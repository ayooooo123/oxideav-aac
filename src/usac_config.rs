// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg libavcodec/aac/aacdec_usac.c (UsacConfig and loudness)
// and libavcodec/aac.h (sample-rate indexing), at 2da55bf.
// Copyright (c) 2024 Lynne <dev@lynne.ee>.
// Copyright (c) 2005-2006 Oded Shimon
// Copyright (c) 2006-2007 Maxim Gavrilov
// Distributed under the GNU Lesser General Public License, version 2.1
// or (at your option) any later version. See LICENSE-LGPL.

//! USAC (AOT 42) AudioSpecificConfig for the 1024-line FD decoder.
//!
//! Mono and stereo, without eSBR, MPS212 or time-warped MDCT, are supported.
//! Length-delimited extension metadata is consumed without affecting audio;
//! program/anchor loudness for the unprocessed layout is retained for the
//! decoder's optional `target_level` normalization. DRC is not applied.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::usac_tables::SAMPLE_RATES;

/// Validated mono/stereo, 1024-line FD configuration and loudness metadata.
#[derive(Clone, Debug)]
pub struct UsacConfig {
    /// Core and output sample rate in Hz (no eSBR rate expansion).
    pub sample_rate: u32,
    /// Number of interleaved output channels.
    pub channels: u8,
    /// ISO/IEC 23003-4 Table A.48: loudness = -57.75 + 0.25 * value.
    pub loudness_method_value: Option<u8>,
    pub(crate) rate_index: u8,
    pub(crate) elements: Vec<ElementConfig>,
}

#[derive(Clone, Debug)]
pub(crate) enum ElementConfig {
    Audio { channels: u8, noise_fill: bool },
    Extension { default_length: u32, fragmented: bool },
}

pub(crate) fn is_usac(data: &[u8]) -> bool {
    let mut bits = BitReader::new(data);
    matches!(audio_object_type(&mut bits), Ok(42))
}

fn audio_object_type(bits: &mut BitReader<'_>) -> Result<u32> {
    let aot = bits.read_u32(5)?;
    Ok(if aot == 31 { 32 + bits.read_u32(6)? } else { aot })
}

pub(crate) fn escaped(bits: &mut BitReader<'_>, a: u32, b: u32, c: u32) -> Result<u32> {
    let mut value = bits.read_u32(a)?;
    if value == (1 << a) - 1 {
        let second = bits.read_u32(b)?;
        value += second;
        if c != 0 && second == (1 << b) - 1 {
            value += bits.read_u32(c)?;
        }
    }
    Ok(value)
}

fn payload_end(bits: &BitReader<'_>, length_bits: u32) -> Result<u64> {
    if length_bits as u64 > bits.bits_remaining() {
        return Err(Error::invalid("AAC USAC: truncated extension payload"));
    }
    Ok(bits.bit_position() + length_bits as u64)
}

fn finish_payload(bits: &mut BitReader<'_>, end: u64) -> Result<()> {
    let remaining = end.checked_sub(bits.bit_position())
        .ok_or_else(|| Error::invalid("AAC USAC: extension exceeds declared length"))?;
    bits.skip(remaining as u32)
}

impl UsacConfig {
    /// Read a complete escaped-AOT AudioSpecificConfig, including its outer
    /// frequency/channel fields and the following UsacConfig().
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut bits = BitReader::new(data);
        if audio_object_type(&mut bits)? != 42 {
            return Err(Error::invalid("AAC USAC: AudioSpecificConfig is not AOT 42"));
        }
        let outer_frequency = bits.read_u32(4)?;
        if outer_frequency == 15 {
            if bits.read_u32(24)? == 0 {
                return Err(Error::invalid("AAC USAC: zero outer sample rate"));
            }
        } else if outer_frequency > 12 {
            return Err(Error::invalid("AAC USAC: reserved outer sample rate"));
        }
        bits.skip(4)?; // Outer channelConfiguration; UsacConfig is authoritative.
        let index = bits.read_u32(5)? as usize;
        let sample_rate = if index == 31 {
            bits.read_u32(24)?
        } else {
            let rate = SAMPLE_RATES[index];
            if rate < 0 {
                return Err(Error::invalid("AAC USAC: reserved sample rate"));
            }
            rate as u32
        };
        if sample_rate == 0 {
            return Err(Error::invalid("AAC USAC: zero sample rate"));
        }
        if bits.read_u32(3)? != 1 {
            return Err(Error::unsupported("AAC USAC: only 1024-line FD without eSBR is supported"));
        }
        let channel_configuration = bits.read_u32(5)?;
        let channels = match channel_configuration {
            1 | 2 => channel_configuration as u8,
            0 => {
                let count = escaped(&mut bits, 5, 8, 16)?;
                if !(1..=2).contains(&count) {
                    return Err(Error::unsupported("AAC USAC: only mono and stereo layouts are supported"));
                }
                for ch in 0..count {
                    let position = bits.read_u32(5)?;
                    let expected = if count == 1 { 2 } else { ch };
                    if position != expected {
                        return Err(Error::unsupported("AAC USAC: noncanonical explicit channel layout"));
                    }
                }
                count as u8
            }
            _ => return Err(Error::unsupported("AAC USAC: only mono and stereo layouts are supported")),
        };
        let count = escaped(&mut bits, 4, 8, 16)? + 1;
        if count > 64 {
            return Err(Error::invalid("AAC USAC: too many configured elements"));
        }
        let mut elements = Vec::with_capacity(count as usize);
        let mut audio_channels = 0;
        for _ in 0..count {
            let kind = bits.read_u32(2)?;
            if kind == 3 {
                let _extension_type = escaped(&mut bits, 4, 8, 16)?;
                let config_length = escaped(&mut bits, 4, 8, 16)?;
                let default_length = if bits.read_bit()? {
                    escaped(&mut bits, 8, 16, 0)? + 1
                } else { 0 };
                let fragmented = bits.read_bit()?;
                // DRC gains and ancillary/preroll metadata are not applied,
                // as in the reference FD decoder's default rendering mode.
                bits.skip(config_length * 8)?;
                elements.push(ElementConfig::Extension { default_length, fragmented });
            } else {
                let mut noise_fill = false;
                if kind != 2 {
                    if bits.read_bit()? {
                        return Err(Error::unsupported("AAC USAC: time-warped MDCT"));
                    }
                    noise_fill = bits.read_bit()?;
                }
                let element_channels = if kind == 1 { 2 } else { 1 };
                audio_channels += element_channels;
                if audio_channels > channels {
                    return Err(Error::invalid("AAC USAC: element channels exceed the declared layout"));
                }
                elements.push(ElementConfig::Audio { channels: element_channels, noise_fill });
            }
        }
        if audio_channels != channels {
            return Err(Error::invalid("AAC USAC: elements do not cover the declared channel layout"));
        }
        let mut loudness_method_value = None;
        if bits.read_bit()? {
            let extensions = escaped(&mut bits, 2, 4, 8)? + 1;
            for _ in 0..extensions {
                let kind = escaped(&mut bits, 4, 8, 16)?;
                let bytes = escaped(&mut bits, 4, 8, 16)?;
                let end = payload_end(&bits, bytes * 8)?;
                match kind {
                    2 => parse_loudness_set(&mut bits, &mut loudness_method_value)?,
                    7 => { bits.skip(16)?; }
                    _ => {}
                }
                finish_payload(&mut bits, end)?;
            }
        }
        let thresholds = [92017, 75132, 55426, 46009, 37566, 27713, 23004, 18783, 13856, 11502, 9391];
        let rate_index = thresholds.iter().position(|&minimum| sample_rate >= minimum).unwrap_or(11) as u8;
        Ok(Self { sample_rate, channels, loudness_method_value, rate_index, elements })
    }
}

fn parse_loudness_info(bits: &mut BitReader<'_>, v1: bool) -> Result<Option<u8>> {
    let drc_set = bits.read_u32(6)?;
    let eq_set = if v1 { bits.read_u32(6)? } else { 0 };
    let downmix = bits.read_u32(7)?;
    if bits.read_bit()? { bits.skip(12)?; }
    if bits.read_bit()? { bits.skip(18)?; }
    let measurements = bits.read_u32(4)?;
    let mut selected = None;
    for _ in 0..measurements {
        let method = bits.read_u32(4)?;
        let width = match method { 7 => 5, 8 => 2, _ => 8 };
        let value = bits.read_u32(width)? as u8;
        bits.skip(6)?; // measurement system and reliability
        if selected.is_none() && drc_set == 0 && eq_set == 0 && downmix == 0 && matches!(method, 1 | 2) {
            selected = Some(value);
        }
    }
    Ok(selected)
}

fn parse_loudness_entries(bits: &mut BitReader<'_>, v1: bool, selected: &mut Option<u8>) -> Result<()> {
    let albums = bits.read_u32(6)?;
    let programs = bits.read_u32(6)?;
    for _ in 0..albums { parse_loudness_info(bits, v1)?; }
    for _ in 0..programs {
        let value = parse_loudness_info(bits, v1)?;
        if selected.is_none() { *selected = value; }
    }
    Ok(())
}

fn parse_loudness_set(bits: &mut BitReader<'_>, selected: &mut Option<u8>) -> Result<()> {
    parse_loudness_entries(bits, false, selected)?;
    if bits.read_bit()? {
        loop {
            let kind = bits.read_u32(4)?;
            if kind == 0 { break; }
            let size_bits = bits.read_u32(4)? + 4;
            let size = bits.read_u32(size_bits)? + 1;
            let end = payload_end(bits, size)?;
            if kind == 1 { parse_loudness_entries(bits, true, selected)?; }
            finish_payload(bits, end)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_configs_resolve_authoritative_geometry() {
        for (bytes, rate, channels) in [
            (&[0xf9, 0x42, 0x41, 0x22, 0x14, 0xc0, 0x00][..], 88200, 2),
            (&[0xf9, 0x48, 0x44, 0x22, 0x14, 0xc0, 0x00][..], 44100, 2),
            (&[0xf9, 0x46, 0x23, 0x21, 0x10, 0xc0, 0x00][..], 48000, 1),
            (&[0xf9, 0x4a, 0x45, 0x22, 0x14, 0xc0, 0x00][..], 32000, 2),
        ] {
            assert!(is_usac(bytes));
            let config = UsacConfig::parse(bytes).unwrap();
            assert_eq!((config.sample_rate, config.channels), (rate, channels));
            assert_eq!(config.elements.len(), 2);
            assert_eq!(config.loudness_method_value, None);
            for length in 0..bytes.len() {
                assert!(UsacConfig::parse(&bytes[..length]).is_err());
            }
        }
        assert!(!is_usac(&[0x12, 0x10]));
        assert!(UsacConfig::parse(&[0x12, 0x10]).is_err());
    }

    #[test]
    fn loudness_extensions_parse_without_consuming_following_syntax() {
        let conformance = [0xf9,0x46,0x43,0x22,0x14,0xc0,0x08,0x5a,0x00,0x11,0x38,0x40,0x02,0x00,0x00,0x2b,0xc0,0x11,0x87,0x2c,0x00];
        let exhale = [0xf9,0x46,0x43,0x22,0x1c,0xc0,0x58,0x52,0x00,0x20,0x00,0xa0,0x40,0x46,0xd0,0xb8,0x00];
        assert!(UsacConfig::parse(&conformance).unwrap().loudness_method_value.is_some());
        assert!(UsacConfig::parse(&exhale).unwrap().loudness_method_value.is_some());
        for bytes in [&conformance[..], &exhale[..]] {
            for length in 0..bytes.len() {
                assert!(UsacConfig::parse(&bytes[..length]).is_err());
            }
        }
    }

    #[test]
    fn two_thousand_seeded_config_mutations_are_bounded() {
        let original = [0xf9,0x46,0x43,0x22,0x1c,0xc0,0x58,0x52,0x00,0x20,0x00,0xa0,0x40,0x46,0xd0,0xb8,0x00];
        let mut state = 0x42ca_ac5e_ed01_0001u64;
        for run in 0..2000 {
            let mut bytes = original;
            for _ in 0..1 + run % 8 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let bit = state as usize % (bytes.len() * 8);
                bytes[bit / 8] ^= 1 << (bit % 8);
            }
            let length = if run % 2 == 0 { bytes.len() } else { state as usize % bytes.len() };
            let result = std::panic::catch_unwind(|| UsacConfig::parse(&bytes[..length]));
            assert!(result.is_ok(), "config mutation {run}, seed {state:#x}");
        }
    }
}
