// SPDX-License-Identifier: LGPL-2.1-or-later
// USAC FD core, ported from FFmpeg libavcodec/aac/aacdec_usac.c,
// aacdec.c (TNS syntax), aacdec_dsp_template.c (TNS synthesis) and
// libavcodec/lpc_functions.h, at 2da55bf.
// Copyright (c) 2024 Lynne <dev@lynne.ee>
// Copyright (c) 2006 Justin Ruggles <justin.ruggles@gmail.com>
// Copyright (c) 2005-2006 Oded Shimon
// Copyright (c) 2006-2007 Maxim Gavrilov
// Copyright (c) 2008-2013 Alex Converse <alex.converse@gmail.com>
// Distributed under the GNU Lesser General Public License, version 2.1
// or (at your option) any later version. See LICENSE-LGPL.

use std::collections::VecDeque;
use std::sync::LazyLock;

use oxideav_core::bits::BitReader;
use oxideav_core::{AudioFormat, AudioFrame, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, SampleFormat};

use crate::filterbank::Filterbank;
use crate::ics_info::{IcsInfo, WindowSequence, WindowShape};
use crate::scale_factor_data::hcod_sf_decode;
use crate::swb_offset::{FrameFamily, long_window_offsets, short_window_offsets};
use crate::usac_arith;
use crate::usac_config::{ElementConfig, UsacConfig};
use crate::usac_tables::{MDST_FILTERS, TNS_REFLECTION};

const N: usize = 1024;
const MAX_BANDS: usize = 128;
const TNS_LONG: [usize; 12] = [31, 31, 34, 40, 42, 51, 47, 47, 43, 43, 43, 40];
const TNS_SHORT: [usize; 12] = [9, 9, 10, 14, 14, 14, 15, 15, 15, 15, 15, 15];
static SCALE_FACTORS: LazyLock<[f32; 428]> = LazyLock::new(||
    std::array::from_fn(|i| 2.0f64.powf((i as f64 - 200.0) * 0.25) as f32));
static NOISE_LEVELS: LazyLock<[f32; 8]> = LazyLock::new(||
    std::array::from_fn(|i| 2.0f32.powf((i as f32 - 14.0) / 3.0)));

fn codec_error(error: crate::Error) -> Error {
    Error::invalid(format!("AAC USAC: {error}"))
}

pub(crate) struct UsacDecoder {
    codec_id: CodecId,
    config: UsacConfig,
    cores: Vec<Core>,
    pending: VecDeque<AudioFrame>,
    gain: f32,
    eof: bool,
}

impl UsacDecoder {
    pub(crate) fn new(params: &CodecParameters) -> Result<Self> {
        let config = UsacConfig::parse(&params.extradata)?;
        let target = params.options.get("target_level")
            .map(|v| v.parse::<i32>().map_err(|_| Error::invalid("AAC USAC: invalid target_level")))
            .transpose()?.unwrap_or(0);
        if !(-63..=0).contains(&target) {
            return Err(Error::invalid("AAC USAC: target_level must be between -63 and 0 dBFS"));
        }
        let gain = match (target, config.loudness_method_value) {
            (0, _) | (_, None) => 1.0,
            (_, Some(value)) => {
                let input_loudness = -57.75f32 + 0.25 * value as f32;
                10.0f32.powf((target as f32 - input_loudness) / 20.0)
            }
        };
        let cores = make_cores(&config)?;
        Ok(Self { codec_id: CodecId::new("aac"), config, cores, pending: VecDeque::new(), gain, eof: false })
    }
}

fn make_cores(config: &UsacConfig) -> Result<Vec<Core>> {
    config.elements.iter().filter_map(|element| match *element {
        ElementConfig::Audio { channels, .. } => Some(Core::new(channels, config.rate_index)),
        ElementConfig::Extension { .. } => None,
    }).collect()
}

impl Decoder for UsacDecoder {
    fn codec_id(&self) -> &CodecId { &self.codec_id }

    fn output_audio_format(&self) -> Option<AudioFormat> {
        Some(AudioFormat { sample_format: SampleFormat::F32, sample_rate: self.config.sample_rate, channels: self.config.channels as u16 })
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.eof { return Err(Error::other("AAC USAC: send_packet after flush")); }
        if packet.data.is_empty() { return Ok(()); }
        let mut bits = BitReader::new(&packet.data);
        let independent = bits.read_bit()?;
        let mut core_index = 0;
        for element in &self.config.elements {
            match *element {
                ElementConfig::Audio { noise_fill, .. } => {
                    self.cores[core_index].decode(&mut bits, independent, noise_fill, self.config.rate_index)?;
                    core_index += 1;
                }
                ElementConfig::Extension { default_length, fragmented } => {
                    if bits.read_bit()? {
                        let length = if bits.read_bit()? { default_length } else {
                            let n = bits.read_u32(8)?;
                            if n == 255 { n + bits.read_u32(16)? - 2 } else { n }
                        };
                        if length != 0 {
                            if fragmented { bits.skip(2)?; }
                            bits.skip(length * 8)?;
                        }
                    }
                }
            }
        }
        let mut bytes = Vec::with_capacity(N * self.config.channels as usize * 4);
        for i in 0..N {
            for core in &self.cores {
                for channel in &core.channels {
                    let value = (channel.pcm[i] / 32768.0) as f32 * self.gain;
                    if !value.is_finite() {
                        return Err(Error::invalid("AAC USAC: nonfinite reconstructed sample"));
                    }
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        self.pending.push_back(AudioFrame { samples: N as u32, pts: packet.pts, data: vec![bytes] });
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.pending.pop_front().map(Frame::Audio).ok_or(if self.eof { Error::Eof } else { Error::NeedMore })
    }

    fn flush(&mut self) -> Result<()> { self.eof = true; Ok(()) }

    fn reset(&mut self) -> Result<()> {
        self.cores = make_cores(&self.config)?;
        self.pending.clear();
        self.eof = false;
        Ok(())
    }
}

struct Channel {
    ics: IcsInfo,
    previous_sequence: WindowSequence,
    previous_shape: WindowShape,
    previous_groups: usize,
    arith: usac_arith::State,
    spectrum: [f32; N],
    scalefactors: [i16; MAX_BANDS],
    noise_seed: u32,
    tns: Tns,
    filterbank: Filterbank,
    transform: [f64; N],
    pcm: Vec<f64>,
}

impl Channel {
    fn new(rate_index: u8, channel: u8) -> Result<Self> {
        let offsets = long_window_offsets(rate_index).map_err(codec_error)?;
        let mut groups = Vec::with_capacity(8);
        groups.push(1);
        Ok(Self {
            ics: IcsInfo {
                family: FrameFamily::Lc1024,
                ics_reserved_bit: false,
                window_sequence: WindowSequence::OnlyLong,
                window_shape: WindowShape::Sine,
                max_sfb: 0,
                scale_factor_grouping: None,
                predictor_data_present: false,
                predictor_data: None,
                ltp_data_present: false,
                ltp_data: None,
                ltp_data_present_pair: None,
                ltp_data_pair: None,
                num_windows: 1,
                num_window_groups: 1,
                window_group_length: groups,
                num_swb: (offsets.len() - 1) as u8,
            },
            previous_sequence: WindowSequence::OnlyLong,
            previous_shape: WindowShape::Sine,
            previous_groups: 1,
            arith: usac_arith::State::default(),
            spectrum: [0.0; N],
            scalefactors: [0; MAX_BANDS],
            // Preserve the reference decoder's FD reset seeds.
            noise_seed: if channel == 1 { 0x10932 } else { 0 },
            tns: Tns::default(),
            filterbank: Filterbank::new(),
            transform: [0.0; N],
            pcm: Vec::new(),
        })
    }

    fn remember_window(&mut self) {
        self.previous_sequence = self.ics.window_sequence;
        self.previous_shape = self.ics.window_shape;
        self.previous_groups = self.ics.num_window_groups as usize;
    }

    fn read_ics(&mut self, bits: &mut BitReader<'_>, rate_index: u8) -> Result<()> {
        self.remember_window();
        self.ics.window_sequence = WindowSequence::from_bits(bits.read_u32(2)? as u8);
        self.ics.window_shape = WindowShape::from_bit(bits.read_bit()?);
        self.ics.max_sfb = bits.read_u32(if self.ics.window_sequence.is_eight_short() { 4 } else { 6 })? as u8;
        self.ics.scale_factor_grouping = if self.ics.window_sequence.is_eight_short() { Some(bits.read_u32(7)? as u8) } else { None };
        self.setup_ics(rate_index)
    }

    fn setup_ics(&mut self, rate_index: u8) -> Result<()> {
        let offsets = offsets(&self.ics, rate_index)?;
        self.ics.num_swb = (offsets.len() - 1) as u8;
        if self.ics.max_sfb > self.ics.num_swb {
            return Err(Error::invalid("AAC USAC: max_sfb exceeds the window's band count"));
        }
        self.ics.window_group_length.clear();
        self.ics.window_group_length.push(1);
        self.ics.num_windows = 1;
        if let Some(mask) = self.ics.scale_factor_grouping {
            self.ics.num_windows = 8;
            for i in 0..7 {
                if mask & (1 << (6 - i)) == 0 {
                    self.ics.window_group_length.push(1);
                } else {
                    *self.ics.window_group_length.last_mut().expect("one window group") += 1;
                }
            }
        }
        self.ics.num_window_groups = self.ics.window_group_length.len() as u8;
        Ok(())
    }

    fn scalefactors(&mut self, bits: &mut BitReader<'_>, global_gain: u32) -> Result<()> {
        let count = self.ics.num_window_groups as usize * self.ics.max_sfb as usize;
        let mut value = global_gain as i16;
        for i in 0..count {
            if i != 0 { value += hcod_sf_decode(bits).map_err(codec_error)? as i16; }
            if !(0..=255).contains(&value) {
                return Err(Error::invalid("AAC USAC: scalefactor outside 0..255"));
            }
            self.scalefactors[i] = value - 100;
        }
        Ok(())
    }

    fn scale(&mut self, rate_index: u8, noise_level: usize, noise_offset: i16) -> Result<()> {
        let swb = offsets(&self.ics, rate_index)?;
        let noise_start = if self.ics.window_sequence.is_eight_short() { 20 } else { 160 };
        let noise = NOISE_LEVELS[noise_level];
        let mut base = 0;
        for (g, &group_length) in self.ics.window_group_length.iter().enumerate() {
            for band in 0..self.ics.max_sfb as usize {
                let scale_index = g * self.ics.max_sfb as usize + band;
                if noise_level != 0 && swb[band] as usize >= noise_start {
                    let mut all_zero = true;
                    for w in 0..group_length as usize {
                        for k in swb[band] as usize..swb[band + 1] as usize {
                            let value = &mut self.spectrum[base + w * 128 + k];
                            if *value == 0.0 {
                                self.noise_seed = self.noise_seed.wrapping_mul(69069).wrapping_add(5);
                                *value = if self.noise_seed & 0x10000 != 0 { -noise } else { noise };
                            } else { all_zero = false; }
                        }
                    }
                    if all_zero { self.scalefactors[scale_index] = (self.scalefactors[scale_index] + noise_offset).max(-200); }
                }
                // Our IMDCT has the opposite sign of av_tx's inverse, so
                // keep positive gains here instead of FFmpeg's negative SF.
                let scale = SCALE_FACTORS[(self.scalefactors[scale_index] + 200) as usize];
                for w in 0..group_length as usize {
                    for k in swb[band] as usize..swb[band + 1] as usize {
                        self.spectrum[base + w * 128 + k] *= scale;
                    }
                }
            }
            base += group_length as usize * 128;
        }
        Ok(())
    }

    fn synthesize(&mut self) -> Result<()> {
        for (dst, &src) in self.transform.iter_mut().zip(&self.spectrum) { *dst = src as f64; }
        self.pcm = self.filterbank.synthesize(&self.transform, &self.ics).map_err(codec_error)?;
        Ok(())
    }
}

fn offsets(ics: &IcsInfo, rate_index: u8) -> Result<&'static [u16]> {
    if ics.window_sequence.is_eight_short() { short_window_offsets(rate_index) } else { long_window_offsets(rate_index) }
        .map_err(codec_error)
}

struct Core {
    channels: Vec<Channel>,
    stereo: Stereo,
}

impl Core {
    fn new(channels: u8, rate_index: u8) -> Result<Self> {
        Ok(Self { channels: (0..channels).map(|ch| Channel::new(rate_index, ch)).collect::<Result<_>>()?, stereo: Stereo::default() })
    }

    fn decode(&mut self, bits: &mut BitReader<'_>, independent: bool, noise_fill: bool, rate_index: u8) -> Result<()> {
        for channel in &mut self.channels {
            channel.tns.counts.fill(0);
            if bits.read_bit()? { return Err(Error::unsupported("AAC USAC: LPD/ACELP core mode")); }
        }
        let mut tns_present = [false; 2];
        self.stereo.common = false;
        self.stereo.common_tns = false;
        if self.channels.len() == 2 {
            self.stereo.parse(bits, &mut self.channels, independent, rate_index, &mut tns_present)?;
        }
        let channel_count = self.channels.len();
        for (ch, channel) in self.channels.iter_mut().enumerate() {
            if channel_count == 1 { tns_present[ch] = bits.read_bit()?; }
            let gain = bits.read_u32(8)?;
            let (noise_level, noise_offset) = if noise_fill {
                (bits.read_u32(3)? as usize, bits.read_u32(5)? as i16 - 16)
            } else { (0, 0) };
            if !self.stereo.common { channel.read_ics(bits, rate_index)?; }
            channel.scalefactors(bits, gain)?;
            if tns_present[ch] { channel.tns.parse(bits, &channel.ics)?; }
            let reset = independent || bits.read_bit()?;
            let swb = offsets(&channel.ics, rate_index)?;
            let len = swb[channel.ics.max_sfb as usize] as usize;
            let n = N / channel.ics.num_windows as usize;
            for (w, spectrum) in channel.spectrum.chunks_exact_mut(n).enumerate() {
                channel.arith.spectrum(bits, reset && w == 0, len, spectrum)?;
            }
            if bits.read_bit()? { return Err(Error::unsupported("AAC USAC: FAC transition from an LPD core")); }
            channel.scale(rate_index, noise_level, noise_offset)?;
        }
        if self.channels.len() == 2 && self.stereo.common {
            if !self.stereo.tns_on_lr { self.apply_tns(rate_index)?; }
            self.stereo.apply(&mut self.channels, rate_index)?;
        }
        if channel_count == 2 {
            self.stereo.prev_re.copy_from_slice(&self.stereo.re);
            self.stereo.prev_im.copy_from_slice(&self.stereo.im);
        }
        if self.channels.len() == 1 || self.stereo.tns_on_lr { self.apply_tns(rate_index)?; }
        for channel in &mut self.channels { channel.synthesize()?; }
        Ok(())
    }

    fn apply_tns(&mut self, rate_index: u8) -> Result<()> {
        let (first, rest) = self.channels.split_at_mut(1);
        let left = &mut first[0];
        left.tns.apply(&mut left.spectrum, &left.ics, rate_index)?;
        if let Some(right) = rest.first_mut() {
            let tns = if self.stereo.common_tns { &left.tns } else { &right.tns };
            tns.apply(&mut right.spectrum, &right.ics, rate_index)?;
        }
        Ok(())
    }
}

struct Stereo {
    common: bool,
    common_tns: bool,
    tns_on_lr: bool,
    mode: u32,
    max_sfb: usize,
    used: [bool; MAX_BANDS],
    direction: bool,
    use_previous: bool,
    re: [f32; MAX_BANDS],
    im: [f32; MAX_BANDS],
    prev_re: [f32; MAX_BANDS],
    prev_im: [f32; MAX_BANDS],
    downmix_im: [f32; N],
}

impl Default for Stereo {
    fn default() -> Self {
        Self { common: false, common_tns: false, tns_on_lr: false, mode: 0, max_sfb: 0,
            used: [false; MAX_BANDS], direction: false, use_previous: false,
            re: [0.0; MAX_BANDS], im: [0.0; MAX_BANDS], prev_re: [0.0; MAX_BANDS],
            prev_im: [0.0; MAX_BANDS], downmix_im: [0.0; N] }
    }
}

impl Stereo {
    fn parse(&mut self, bits: &mut BitReader<'_>, channels: &mut [Channel], independent: bool, rate_index: u8, tns_present: &mut [bool; 2]) -> Result<()> {
        self.re.fill(0.0);
        self.im.fill(0.0);
        self.used.fill(false);
        let tns_active = bits.read_bit()?;
        self.common = bits.read_bit()?;
        if !self.common || independent { self.prev_re.fill(0.0); self.prev_im.fill(0.0); }
        if self.common {
            let (first, rest) = channels.split_at_mut(1);
            let left = &mut first[0];
            let right = &mut rest[0];
            left.read_ics(bits, rate_index)?;
            right.remember_window();
            right.ics.window_sequence = left.ics.window_sequence;
            right.ics.window_shape = left.ics.window_shape;
            right.ics.max_sfb = left.ics.max_sfb;
            right.ics.scale_factor_grouping = left.ics.scale_factor_grouping;
            if !bits.read_bit()? {
                right.ics.max_sfb = bits.read_u32(if right.ics.window_sequence.is_eight_short() { 4 } else { 6 })? as u8;
            }
            right.setup_ics(rate_index)?;
            if [left, right].iter().any(|ch| ch.ics.window_sequence.is_eight_short() != ch.previous_sequence.is_eight_short()) {
                self.prev_re.fill(0.0);
                self.prev_im.fill(0.0);
            }
            self.max_sfb = channels[0].ics.max_sfb.max(channels[1].ics.max_sfb) as usize;
            self.mode = bits.read_u32(2)?;
            let count = channels[0].ics.num_window_groups as usize * self.max_sfb;
            match self.mode {
                1 => for used in &mut self.used[..count] { *used = bits.read_bit()?; },
                2 => self.used[..count].fill(true),
                3 => self.parse_prediction(bits, &channels[0], independent)?,
                _ => {}
            }
        }
        self.tns_on_lr = false;
        if tns_active {
            self.common_tns = self.common && bits.read_bit()?;
            self.tns_on_lr = bits.read_bit()?;
            if self.common_tns {
                let left = &mut channels[0];
                left.tns.parse(bits, &left.ics)?;
            } else if bits.read_bit()? {
                *tns_present = [true, true];
            } else {
                tns_present[1] = bits.read_bit()?;
                tns_present[0] = !tns_present[1];
            }
        }
        Ok(())
    }

    fn parse_prediction(&mut self, bits: &mut BitReader<'_>, left: &Channel, independent: bool) -> Result<()> {
        let groups = left.ics.num_window_groups as usize;
        if bits.read_bit()? { self.used[..groups * self.max_sfb].fill(true); } else {
            for g in 0..groups {
                for sfb in (0..self.max_sfb).step_by(2) {
                    let used = bits.read_bit()?;
                    self.used[g * self.max_sfb + sfb] = used;
                    if sfb + 1 < self.max_sfb { self.used[g * self.max_sfb + sfb + 1] = used; }
                }
            }
        }
        self.direction = bits.read_bit()?;
        let complex = bits.read_bit()?;
        self.use_previous = complex && !independent && bits.read_bit()?;
        let delta_time = !independent && bits.read_bit()?;
        for g in 0..groups {
            for sfb in (0..self.max_sfb).step_by(2) {
                let index = g * self.max_sfb + sfb;
                let (mut re, mut im) = if delta_time {
                    if g != 0 { (self.re[index - self.max_sfb], self.im[index - self.max_sfb]) }
                    else {
                        let previous_group = if left.ics.window_sequence.is_eight_short() && left.previous_sequence.is_eight_short() { left.previous_groups - 1 } else { 0 };
                        let p = previous_group * self.max_sfb + sfb;
                        (self.prev_re[p], self.prev_im[p])
                    }
                } else if sfb != 0 { (self.re[index - 1], self.im[index - 1]) } else { (0.0, 0.0) };
                if self.used[index] {
                    re += -(hcod_sf_decode(bits).map_err(codec_error)? as f32) * 0.1;
                    if complex { im += -(hcod_sf_decode(bits).map_err(codec_error)? as f32) * 0.1; }
                    self.re[index] = re;
                    self.im[index] = im;
                }
                if sfb + 1 < self.max_sfb {
                    self.re[index + 1] = self.re[index];
                    self.im[index + 1] = self.im[index];
                }
            }
        }
        Ok(())
    }

    fn apply(&mut self, channels: &mut [Channel], rate_index: u8) -> Result<()> {
        if self.mode == 0 { return Ok(()); }
        let (first, rest) = channels.split_at_mut(1);
        let left = &mut first[0];
        let right = &mut rest[0];
        let swb = offsets(&left.ics, rate_index)?;
        if self.mode == 3 {
            let mut current = [0.0f32; N];
            let mut previous = [0.0f32; N];
            let sign = if self.direction { -1.0 } else { 1.0 };
            let mut base = 0;
            for (g, &length) in left.ics.window_group_length.iter().enumerate() {
                for sfb in 0..self.max_sfb {
                    for w in 0..length as usize {
                        for k in swb[sfb] as usize..swb[sfb + 1] as usize {
                            let index = base + w * 128 + k;
                            // Match aacdec_usac.c's downmix_prev, which uses
                            // this frame's coefficients before prediction.
                            let downmix = (0.5f64 * (left.spectrum[index] + sign * right.spectrum[index]) as f64) as f32;
                            previous[index] = downmix;
                            current[index] = if self.used[g * self.max_sfb + sfb] { downmix } else { left.spectrum[index] };
                        }
                    }
                }
                base += length as usize * 128;
            }
            let window = match left.ics.window_sequence { WindowSequence::LongStart => 1, WindowSequence::LongStop => 2, _ => 0 };
            let shape = match (left.ics.window_shape, left.previous_shape) {
                (WindowShape::Sine, WindowShape::Sine) => 0,
                (WindowShape::Kbd, WindowShape::Kbd) => 1,
                (WindowShape::Sine, WindowShape::Kbd) => 2,
                (WindowShape::Kbd, WindowShape::Sine) => 3,
            };
            interpolate_imag(&mut self.downmix_im, &current, &MDST_FILTERS[window][shape], 1.0);
            if self.use_previous {
                let window = usize::from(left.ics.window_sequence == WindowSequence::LongStop);
                interpolate_imag(&mut self.downmix_im, &previous, &MDST_FILTERS[window][left.previous_shape as usize], -1.0);
            }
        }
        let mut base = 0;
        for (g, &length) in left.ics.window_group_length.iter().enumerate() {
            for sfb in 0..self.max_sfb {
                let band = g * self.max_sfb + sfb;
                if !self.used[band] { continue; }
                for w in 0..length as usize {
                    for k in swb[sfb] as usize..swb[sfb + 1] as usize {
                        let index = base + w * 128 + k;
                        let a = left.spectrum[index];
                        let b = right.spectrum[index];
                        if self.mode == 3 {
                            let predicted = b - self.re[band] * a - self.im[band] * self.downmix_im[index];
                            left.spectrum[index] = a + predicted;
                            right.spectrum[index] = if self.direction { predicted - a } else { a - predicted };
                        } else {
                            left.spectrum[index] = a + b;
                            right.spectrum[index] = a - b;
                        }
                    }
                }
            }
            base += length as usize * 128;
        }
        Ok(())
    }
}

/// Seven-tap MDCT-to-MDST interpolation with the reference's reflected
/// endpoint samples. Even taps can be negated for previous-frame prediction.
fn interpolate_imag(im: &mut [f32; N], re: &[f32; N], filter: &[f32; 7], even: f32) {
    for i in 0..N {
        let mut value = 0.0;
        for tap in 0..7 {
            let position = i as isize + tap as isize - 3;
            let index = if position < 0 { (-position - 1) as usize }
                else if position >= N as isize { (2 * N as isize - 1 - position) as usize }
                else { position as usize };
            let term = filter[6 - tap] * re[index];
            if tap == 0 { value = term; } else { value += term; }
        }
        im[i] += value * if i % 2 == 0 { even } else { 1.0 };
    }
}

#[derive(Clone, Copy, Default)]
struct TnsFilter {
    length: u8,
    order: u8,
    reverse: bool,
    lpc: [f32; 15],
}

#[derive(Default)]
struct Tns {
    counts: [u8; 8],
    filters: [[TnsFilter; 3]; 8],
}

impl Tns {
    fn parse(&mut self, bits: &mut BitReader<'_>, ics: &IcsInfo) -> Result<()> {
        let short = ics.window_sequence.is_eight_short();
        for w in 0..ics.num_windows as usize {
            let count = bits.read_u32(if short { 1 } else { 2 })? as usize;
            self.counts[w] = count as u8;
            if count == 0 { continue; }
            let resolution = bits.read_u32(1)?;
            for filter in &mut self.filters[w][..count] {
                filter.length = bits.read_u32(if short { 4 } else { 6 })? as u8;
                // USAC long-window TNS order is four bits, not GA AAC's five.
                filter.order = bits.read_u32(if short { 3 } else { 4 })? as u8;
                if filter.order == 0 { continue; }
                filter.reverse = bits.read_bit()?;
                let compress = bits.read_u32(1)?;
                let table = TNS_REFLECTION[(2 * compress + resolution) as usize];
                let width = 3 + resolution - compress;
                for i in 0..filter.order as usize {
                    let reflection = -table[bits.read_u32(width)? as usize];
                    filter.lpc[i] = reflection;
                    for j in 0..(i + 1) / 2 {
                        let front = filter.lpc[j];
                        let back = filter.lpc[i - 1 - j];
                        filter.lpc[j] = front + reflection * back;
                        filter.lpc[i - 1 - j] = back + reflection * front;
                    }
                }
            }
        }
        Ok(())
    }

    fn apply(&self, spectrum: &mut [f32; N], ics: &IcsInfo, rate_index: u8) -> Result<()> {
        let swb = offsets(ics, rate_index)?;
        let cap = if ics.window_sequence.is_eight_short() { TNS_SHORT[rate_index as usize] } else { TNS_LONG[rate_index as usize] };
        let cap = cap.min(ics.max_sfb as usize);
        for w in 0..ics.num_windows as usize {
            let mut bottom = swb.len() - 1;
            for filter in &self.filters[w][..self.counts[w] as usize] {
                let top = bottom;
                bottom = bottom.saturating_sub(filter.length as usize);
                let start = swb[bottom.min(cap)] as usize;
                let end = swb[top.min(cap)] as usize;
                let order = filter.order as usize;
                if order == 0 { continue; }
                for m in 0..end - start {
                    let index = w * 128 + if filter.reverse { end - 1 - m } else { start + m };
                    for j in 1..=m.min(order) {
                        let previous = if filter.reverse { index + j } else { index - j };
                        spectrum[index] -= spectrum[previous] * filter.lpc[j - 1];
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;
    use oxideav_core::TimeBase;

    fn mono_params() -> CodecParameters {
        let mut params = CodecParameters::audio(CodecId::new("aac"));
        params.extradata = vec![0xf9, 0x46, 0x23, 0x21, 0x10, 0xc0, 0x00];
        params
    }

    fn silent_frame(sequence: WindowSequence) -> Packet {
        let mut bits = BitWriter::new();
        bits.write_u32(1, 1); // independent
        bits.write_u32(0, 1); // FD core
        bits.write_u32(0, 1); // no TNS
        bits.write_u32(100, 8);
        bits.write_u32(sequence as u32, 2);
        bits.write_u32(0, 1); // sine
        bits.write_u32(0, if sequence.is_eight_short() { 4 } else { 6 });
        if sequence.is_eight_short() { bits.write_u32(0x7f, 7); }
        bits.write_u32(0, 1); // no FAC
        bits.write_u32(0, 1); // extension absent
        let mut packet = Packet::new(0, TimeBase::new(1, 48000), bits.finish());
        packet.pts = Some(17);
        packet
    }

    #[test]
    fn fd_silence_window_switching_flush_and_reset() {
        let mut decoder = UsacDecoder::new(&mono_params()).unwrap();
        for sequence in [WindowSequence::OnlyLong, WindowSequence::LongStart, WindowSequence::EightShort, WindowSequence::LongStop] {
            decoder.send_packet(&silent_frame(sequence)).unwrap();
            let Frame::Audio(frame) = decoder.receive_frame().unwrap() else { panic!("not audio"); };
            assert_eq!(frame.samples, 1024);
            assert_eq!(frame.pts, Some(17));
            assert_eq!(frame.data[0].len(), N * 4);
            // IEEE -0.0 from a windowed zero coefficient is also silence.
            for (i, bytes) in frame.data[0].chunks_exact(4).enumerate() {
                assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), 0.0, "sample {i}");
            }
            assert!(matches!(decoder.receive_frame(), Err(Error::NeedMore)));
        }
        decoder.flush().unwrap();
        assert!(matches!(decoder.receive_frame(), Err(Error::Eof)));
        assert!(decoder.send_packet(&silent_frame(WindowSequence::OnlyLong)).is_err());
        decoder.reset().unwrap();
        decoder.send_packet(&silent_frame(WindowSequence::OnlyLong)).unwrap();
        assert!(decoder.receive_frame().is_ok());
        assert_eq!(decoder.output_audio_format().unwrap().sample_rate, 48000);
    }

    #[test]
    fn unsupported_core_transitions_do_not_emit_partial_pcm() {
        for mask in [(0, 0x40), (2, 0x08)] { // LPD core, FAC transition
            let mut decoder = UsacDecoder::new(&mono_params()).unwrap();
            let mut packet = silent_frame(WindowSequence::OnlyLong);
            packet.data[mask.0] |= mask.1;
            assert!(decoder.send_packet(&packet).is_err());
            assert!(matches!(decoder.receive_frame(), Err(Error::NeedMore)));
        }
    }

    #[test]
    fn tns_long_order_uses_four_bits_and_supports_fifteen_taps() {
        let mut bits = BitWriter::new();
        bits.write_u32(1, 2); // n_filt
        bits.write_u32(0, 1); // three-bit reflection coefficients
        bits.write_u32(5, 6);
        bits.write_u32(15, 4);
        bits.write_u32(0, 1); // forward
        bits.write_u32(0, 1); // uncompressed
        for _ in 0..15 { bits.write_u32(0, 3); }
        let bytes = bits.finish();
        let mut reader = BitReader::new(&bytes);
        let channel = Channel::new(3, 0).unwrap();
        let mut tns = Tns::default();
        tns.parse(&mut reader, &channel.ics).unwrap();
        assert_eq!(reader.bit_position(), 60);
        assert_eq!(tns.filters[0][0].order, 15);
        assert_eq!(tns.filters[0][0].lpc, [0.0; 15]);
    }
}
