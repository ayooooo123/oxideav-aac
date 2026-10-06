//! ER AAC ELD low-delay synthesis filterbank (ISO/IEC 14496-3 §4.6.20).
//!
//! Ported from FFmpeg `libavcodec/aac/aacdec_dsp_template.c`
//! (`imdct_and_windowing_eld`) at commit 2da55bf, which is licensed under
//! the GNU Lesser General Public License version 2.1 or later (see
//! `LICENSE-LGPL`). Like FFmpeg, the inverse transform is mapped onto a
//! conventional half-length IMDCT (Chivukula, Reznik, Devarajan,
//! "Efficient algorithms for MPEG-4 AAC-ELD, AAC-LD and AAC-LC
//! filterbanks", ICALIP 2008), and the overlap spans four frames with
//! the `4N − N/4`-tap window read from sample `N/4` on, as the reference
//! decoder does.

use crate::eld_window::{ELD_WINDOW_480, ELD_WINDOW_512};
use crate::{Error, Result};

/// One channel's ELD synthesis state: the three previous half-length
/// IMDCT outputs the four-frame overlap reads.
#[derive(Clone, Debug)]
pub struct EldFilterbank {
    /// Frame length `N` (512 or 480).
    n: usize,
    /// FFmpeg's `saved[0..3N]`: the previous frame's IMDCT output at
    /// `0..N`, the one before at `N..2N`, the oldest at `2N..3N`.
    saved: Vec<f64>,
    /// The reordered spectral lines of the frame being synthesized.
    input: Vec<f64>,
    /// The half-length IMDCT output of the frame being synthesized.
    buf: Vec<f64>,
}

impl EldFilterbank {
    /// A zeroed filterbank for an `n`-line frame (512 or 480).
    pub fn new(n: usize) -> Result<Self> {
        if n != 512 && n != 480 {
            return Err(Error::FilterbankInvalid);
        }
        Ok(EldFilterbank {
            n,
            saved: vec![0.0; 3 * n],
            input: vec![0.0; n],
            buf: vec![0.0; n],
        })
    }

    /// Synthesize one frame of `N` PCM samples from the `N` decoded
    /// spectral lines.
    pub fn synthesize(&mut self, spec: &[f64]) -> Result<Vec<f64>> {
        let n = self.n;
        if spec.len() != n {
            return Err(Error::FilterbankInvalid);
        }
        let (n2, n4) = (n / 2, n / 4);
        let window: &[f32] = if n == 480 {
            &ELD_WINDOW_480
        } else {
            &ELD_WINDOW_512
        };
        let w = |k: usize| f64::from(window[k]);

        // Reorder and negate the lines so a standard IMDCT computes the
        // ELD inverse transform.
        let input = &mut self.input;
        input.copy_from_slice(spec);
        for i in (0..n2).step_by(2) {
            let t = input[i];
            input[i] = -input[n - 1 - i];
            input[n - 1 - i] = t;
            let t = -input[i + 1];
            input[i + 1] = input[n - 2 - i];
            input[n - 2 - i] = t;
        }
        // The half-length IMDCT: the middle `N` samples of the 2N-point
        // inverse transform, with every even sample negated.
        let plan = crate::mdct::imdct_plan(2 * n).ok_or(Error::FilterbankInvalid)?;
        let buf = &mut self.buf;
        plan.half(input, buf);
        for v in buf.iter_mut().step_by(2) {
            *v = -*v;
        }

        let saved = &self.saved;
        let mut out = vec![0.0f64; n];
        for i in n4..n2 {
            out[i - n4] = buf[n2 - 1 - i] * w(i - n4) + saved[i + n2] * w(i + n - n4)
                - saved[n + n2 - 1 - i] * w(i + 2 * n - n4)
                - saved[2 * n + n2 + i] * w(i + 3 * n - n4);
        }
        for i in 0..n2 {
            out[n4 + i] = buf[i] * w(i + n2 - n4)
                - saved[n - 1 - i] * w(i + n2 + n - n4)
                - saved[n + i] * w(i + n2 + 2 * n - n4)
                + saved[3 * n - 1 - i] * w(i + n2 + 3 * n - n4);
        }
        for i in 0..n4 {
            out[n2 + n4 + i] = buf[i + n2] * w(i + n - n4)
                - saved[n2 - 1 - i] * w(i + 2 * n - n4)
                - saved[n + n2 + i] * w(i + 3 * n - n4);
        }

        self.saved.copy_within(0..2 * n, n);
        self.saved[..n].copy_from_slice(buf);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_lengths() {
        assert!(EldFilterbank::new(1024).is_err());
        let mut fb = EldFilterbank::new(480).unwrap();
        assert!(fb.synthesize(&[0.0; 512]).is_err());
    }

    #[test]
    fn silence_in_silence_out() {
        let mut fb = EldFilterbank::new(512).unwrap();
        for _ in 0..4 {
            let out = fb.synthesize(&[0.0; 512]).unwrap();
            assert_eq!(out.len(), 512);
            assert!(out.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn a_frame_rings_through_four_outputs() {
        let mut fb = EldFilterbank::new(512).unwrap();
        let mut spec = vec![0.0; 512];
        spec[7] = 1000.0;
        let mut energies = Vec::new();
        for frame in 0..6 {
            let input = if frame == 0 { spec.clone() } else { vec![0.0; 512] };
            let out = fb.synthesize(&input).unwrap();
            energies.push(out.iter().map(|v| v * v).sum::<f64>());
        }
        assert!(energies[..4].iter().all(|&e| e > 0.0));
        assert!(energies[4..].iter().all(|&e| e == 0.0));
    }
}
