// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg libavcodec/aac/aacdec_ac.c and decode_spectrum_ac in
// aacdec_usac.c at 2da55bf. Copyright (c) 2024 Lynne <dev@lynne.ee>.
// Distributed under the GNU Lesser General Public License, version 2.1
// or (at your option) any later version. See LICENSE-LGPL.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::usac_tables::{HASH, LOOKUP, LSB_CDFS, MSB_CDFS};

pub(super) struct State {
    last: [u8; 513],
    last_len: usize,
    cur: [u8; 4],
    previous: u16,
}

impl Default for State {
    fn default() -> Self {
        Self { last: [0; 513], last_len: 0, cur: [0; 4], previous: 0 }
    }
}

impl State {
    fn map(&mut self, reset: bool, n: usize) {
        if reset {
            self.last.fill(0);
        } else if self.last_len != n {
            let old = self.last;
            for i in 0..n / 2 {
                // FD window lengths differ by exactly eight, so integer
                // resampling is identical to the reference's float ratio.
                self.last[i] = old[i * self.last_len / n];
            }
            self.last[n / 2..].fill(0);
        }
        self.last_len = n;
        self.cur = [1, 0, 0, 0];
        self.previous = (self.last[0] as u16) << 12;
    }

    fn context(&mut self, i: usize) -> u32 {
        let c = (((self.previous as u32 >> 8) + ((self.last[i + 1] as u32) << 8)) << 4)
            + self.cur[1] as u32;
        self.previous = c as u16;
        if i > 3 && self.cur[1] as u16 + self.cur[2] as u16 + (self.cur[3] as u16) < 5 {
            c + 0x10000
        } else {
            c
        }
    }

    fn update(&mut self, i: usize, a: u32, b: u32) {
        // The normative context state uses the low sixteen bits of each
        // reconstructed magnitude, not the potentially 25-bit escape value.
        let sum = (a as u16) as u32 + (b as u16) as u32 + 1;
        self.cur[0] = sum.min(15) as u8;
        self.cur[3] = self.cur[2];
        self.cur[2] = self.cur[1];
        self.cur[1] = self.cur[0];
        self.last[i] = self.cur[0];
    }

    fn finish(&mut self, offset: usize, n: usize) {
        self.last[offset..n / 2].fill(1);
        self.last[n / 2..].fill(0);
    }

    pub(super) fn spectrum(
        &mut self,
        reader: &mut BitReader<'_>,
        reset: bool,
        len: usize,
        coef: &mut [f32],
    ) -> Result<()> {
        let n = coef.len();
        if !matches!(n, 128 | 1024) || len > n || len % 2 != 0 {
            return Err(Error::invalid("AAC USAC: invalid arithmetic spectrum length"));
        }
        self.map(reset, n);
        coef.fill(0.0);
        if len == 0 {
            self.finish(0, n);
            return Ok(());
        }

        let start = reader.bit_position();
        let mut look = Lookahead { bits: *reader, padding: 0 };
        let mut arith = Arithmetic { low: 0, high: 65535, value: look.get(16)? };
        let mut i = 0;
        while i < len / 2 {
            let context = self.context(i);
            let mut level = 0;
            let symbol = loop {
                let pk = probability(context + (level.min(7) << 17));
                let symbol = arith.decode(&mut look, &MSB_CDFS[pk])?;
                if symbol < 16 {
                    break symbol;
                }
                level += 1;
                if level > 23 {
                    return Err(Error::invalid("AAC USAC: arithmetic escape exceeds 23 bits"));
                }
            };
            if symbol == 0 && level != 0 {
                break;
            }
            let mut a = symbol as u32 & 3;
            let mut b = symbol as u32 >> 2;
            for _ in 0..level {
                let table = if a == 0 { 1 } else if b == 0 { 0 } else { 2 };
                let r = arith.decode(&mut look, &LSB_CDFS[table])? as u32;
                a = (a << 1) | (r & 1);
                b = (b << 1) | (r >> 1);
            }
            coef[2 * i] = (a as f64 * (a as f64).cbrt()) as f32;
            coef[2 * i + 1] = (b as f64 * (b as f64).cbrt()) as f32;
            self.update(i, a, b);
            i += 1;
        }

        // The range coder reads fourteen lookahead bits belonging to the
        // following syntax. Advance the real reader only through its code.
        let consumed = look.bits.bit_position() + look.padding as u64 - start - 14;
        reader.skip(u32::try_from(consumed).map_err(|_| Error::invalid("AAC USAC: arithmetic length overflow"))?)?;
        self.finish(i, n);
        for value in &mut coef[..len] {
            if *value != 0.0 && !reader.read_bit()? {
                *value = -*value;
            }
        }
        Ok(())
    }
}

fn probability(context: u32) -> usize {
    let mut lo = -1isize;
    let mut hi = LOOKUP.len() as isize - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let entry = HASH[mid as usize];
        match context.cmp(&(entry >> 8)) {
            std::cmp::Ordering::Less => hi = mid,
            std::cmp::Ordering::Greater => lo = mid,
            std::cmp::Ordering::Equal => return (entry & 255) as usize,
        }
    }
    LOOKUP[hi as usize] as usize
}

/// FFmpeg's padded bitstream reader permits the range coder's final lookahead
/// beyond the packet. No *consumed* syntax may use these virtual zero bits.
struct Lookahead<'a> {
    bits: BitReader<'a>,
    padding: u32,
}

impl Lookahead<'_> {
    fn get(&mut self, n: u32) -> Result<u32> {
        let available = self.bits.bits_remaining().min(n as u64) as u32;
        let missing = n - available;
        if self.padding + missing > 14 {
            return Err(Error::invalid("AAC USAC: truncated arithmetic code"));
        }
        let value = self.bits.read_u32(available)?;
        self.padding += missing;
        Ok(value << missing)
    }
}

struct Arithmetic {
    low: u32,
    high: u32,
    value: u32,
}

impl Arithmetic {
    fn decode(&mut self, reader: &mut Lookahead<'_>, cdf: &[u16]) -> Result<usize> {
        if self.value < self.low || self.value > self.high {
            return Err(Error::invalid("AAC USAC: invalid arithmetic interval"));
        }
        let range = self.high - self.low + 1;
        let c = ((self.value - self.low + 1) << 14) - 1;
        let symbol = cdf.partition_point(|&p| p as u32 * range > c);
        if symbol == cdf.len() {
            return Err(Error::invalid("AAC USAC: invalid arithmetic symbol"));
        }
        if symbol != 0 {
            self.high = self.low + ((range * cdf[symbol - 1] as u32) >> 14) - 1;
        }
        self.low += (range * cdf[symbol] as u32) >> 14;
        loop {
            let offset = if self.high < 32768 {
                0
            } else if self.low >= 32768 {
                32768
            } else if self.low >= 16384 && self.high < 49152 {
                16384
            } else {
                break;
            };
            self.low = (self.low - offset) << 1;
            self.high = ((self.high - offset) << 1) | 1;
            self.value = ((self.value - offset) << 1) | reader.get(1)?;
        }
        Ok(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_tables_are_bounded_and_monotone() {
        for cdf in MSB_CDFS.iter().map(|v| &v[..]).chain(LSB_CDFS.iter().map(|v| &v[..])) {
            assert_eq!(cdf.last(), Some(&0));
            assert!(cdf[0] < 16384);
            assert!(cdf.windows(2).all(|w| w[0] > w[1]));
        }
        assert!(LOOKUP.iter().all(|&v| v < 64));
        assert!(HASH.windows(2).all(|w| w[0] >> 8 < w[1] >> 8));
        for &entry in &HASH[..HASH.len() - 1] {
            assert_eq!(probability(entry >> 8), (entry & 255) as usize);
        }
        assert!(probability(u32::MAX) < 64);
    }

    #[test]
    fn context_maps_between_long_and_short_windows() {
        let mut state = State::default();
        state.map(true, 1024);
        for i in 0..512 { state.last[i] = (i % 16) as u8; }
        state.map(false, 128);
        for i in 0..64 { assert_eq!(state.last[i], (i * 8 % 16) as u8); }
        state.map(false, 1024);
        for i in 0..512 { assert_eq!(state.last[i], (i / 8 * 8 % 16) as u8); }
        state.finish(4, 1024);
        assert!(state.last[4..512].iter().all(|&v| v == 1));
        assert_eq!(state.last[512], 0);
        state.map(true, 128);
        assert!(state.last.iter().all(|&v| v == 0));
    }

    #[test]
    fn malformed_lengths_and_truncated_codes_are_errors() {
        let mut state = State::default();
        let mut out = [0.0; 128];
        assert!(state.spectrum(&mut BitReader::new(&[]), true, 2, &mut out).is_err());
        assert!(state.spectrum(&mut BitReader::new(&[0; 8]), true, 129, &mut out).is_err());
        assert!(state.spectrum(&mut BitReader::new(&[0; 8]), true, 3, &mut out).is_err());
        state.spectrum(&mut BitReader::new(&[]), true, 0, &mut out).unwrap();
        assert_eq!(out, [0.0; 128]);
    }
}
