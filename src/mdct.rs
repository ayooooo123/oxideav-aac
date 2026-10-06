//! FFT-based MDCT and inverse MDCT for every transform length the codec
//! runs: 2048 / 1920 / 256 / 240 (AAC LC/Main/LTP and HE-AAC cores),
//! 1024 / 960 (ER AAC LD / ELD), 512 / 64 (SSR bands).
//!
//! Ported from FFmpeg `libavutil/tx_template.c` (`ff_tx_mdct_inv`,
//! `ff_tx_mdct_inv_full`, `ff_tx_mdct_fwd`, `ff_tx_mdct_gen_exp`) at commit
//! 2da55bf, which is licensed under the GNU Lesser General Public License
//! version 2.1 or later (see `LICENSE-LGPL`). As in av_tx, an `N/2`
//! coefficient transform is a pre-rotation, an `N/4`-point complex FFT and
//! a post-rotation. The FFT is a mixed-radix (4, 2, 3, 5)
//! decimation-in-time transform run in place after a digit-reversal
//! gather, in f64 like the rest of the decoder's synthesis chain, so the
//! result equals the direct §4.6.11.3.1 sum to f64 rounding.
//!
//! Complex buffers are interleaved `[re, im]` f64 pairs, which lets the
//! inverse transform run in place in its output slice (av_tx does the
//! same with its destination buffer).

use std::sync::LazyLock;

/// In-place mixed-radix complex FFT of `n` points over an interleaved
/// `[re, im]` f64 buffer whose input has been gathered by [`Fft::perm`].
#[derive(Debug)]
struct Fft {
    /// Transform length in complex points.
    n: usize,
    /// Radices, outermost first; their product is `n`.
    factors: Vec<usize>,
    /// `perm[i]`: the natural-order input index the digit-reversal gather
    /// places at position `i`.
    perm: Vec<u32>,
    /// `e^{sign·2πi·j/n}` for `j < n`, as `[re, im]`.
    twiddles: Vec<[f64; 2]>,
    /// `-1` for the forward transform, `+1` for the inverse one.
    sign: f64,
}

/// Split `n` into radices 4, 2, 3, 5 (fours first), or `None` when another
/// prime divides it.
fn factorize(mut n: usize) -> Option<Vec<usize>> {
    let mut factors = Vec::new();
    while n % 4 == 0 {
        factors.push(4);
        n /= 4;
    }
    if n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }
    for p in [3, 5] {
        while n % p == 0 {
            factors.push(p);
            n /= p;
        }
    }
    (n == 1).then_some(factors)
}

/// The input order a recursive decimation-in-time FFT reads at its leaves:
/// sub-transform `r` of a radix-`p` stage takes every `p`-th input from
/// offset `r`, and its output occupies the `r`-th contiguous block.
fn digit_reversal(perm: &mut [u32], out_base: usize, in_off: usize, in_stride: usize, factors: &[usize]) {
    match factors.split_first() {
        None => perm[out_base] = in_off as u32,
        Some((&p, rest)) => {
            let m: usize = rest.iter().product();
            for r in 0..p {
                digit_reversal(perm, out_base + r * m, in_off + r * in_stride, in_stride * p, rest);
            }
        }
    }
}

#[inline(always)]
fn ld(z: &[f64], i: usize) -> (f64, f64) {
    (z[2 * i], z[2 * i + 1])
}

#[inline(always)]
fn st(z: &mut [f64], i: usize, v: (f64, f64)) {
    z[2 * i] = v.0;
    z[2 * i + 1] = v.1;
}

#[inline(always)]
fn cmul(a: (f64, f64), w: [f64; 2]) -> (f64, f64) {
    (a.0 * w[0] - a.1 * w[1], a.0 * w[1] + a.1 * w[0])
}

impl Fft {
    fn new(n: usize, inverse: bool) -> Option<Self> {
        let factors = factorize(n)?;
        let mut perm = vec![0u32; n];
        digit_reversal(&mut perm, 0, 0, 1, &factors);
        let sign = if inverse { 1.0 } else { -1.0 };
        let twiddles = (0..n)
            .map(|j| {
                let a = 2.0 * core::f64::consts::PI * j as f64 / n as f64;
                [a.cos(), sign * a.sin()]
            })
            .collect();
        Some(Fft {
            n,
            factors,
            perm,
            twiddles,
            sign,
        })
    }

    /// Run the butterfly stages in place on the gathered buffer
    /// `z[..2n]`, innermost stage first.
    fn run(&self, z: &mut [f64]) {
        let n = self.n;
        let z = &mut z[..2 * n];
        let mut m = 1;
        for &p in self.factors.iter().rev() {
            let block = p * m;
            let fstride = n / block;
            for base in (0..n).step_by(block) {
                match p {
                    2 => self.radix2(z, base, m, fstride),
                    3 => self.radix3(z, base, m, fstride),
                    4 => self.radix4(z, base, m, fstride),
                    _ => self.radix5(z, base, m, fstride),
                }
            }
            m = block;
        }
    }

    fn radix2(&self, z: &mut [f64], base: usize, m: usize, fstride: usize) {
        for k in 0..m {
            let (a0, a1) = (base + k, base + k + m);
            let x0 = ld(z, a0);
            let t = cmul(ld(z, a1), self.twiddles[k * fstride]);
            st(z, a0, (x0.0 + t.0, x0.1 + t.1));
            st(z, a1, (x0.0 - t.0, x0.1 - t.1));
        }
    }

    fn radix3(&self, z: &mut [f64], base: usize, m: usize, fstride: usize) {
        // W_3 = -1/2 + sign·i·√3/2.
        let s = self.sign * 0.866_025_403_784_438_6;
        for k in 0..m {
            let (a0, a1, a2) = (base + k, base + k + m, base + k + 2 * m);
            let x0 = ld(z, a0);
            let x1 = cmul(ld(z, a1), self.twiddles[k * fstride]);
            let x2 = cmul(ld(z, a2), self.twiddles[2 * k * fstride]);
            let t = (x1.0 + x2.0, x1.1 + x2.1);
            let d = (x1.0 - x2.0, x1.1 - x2.1);
            let c = (x0.0 - 0.5 * t.0, x0.1 - 0.5 * t.1);
            // ±i·s·d
            let r = (-s * d.1, s * d.0);
            st(z, a0, (x0.0 + t.0, x0.1 + t.1));
            st(z, a1, (c.0 + r.0, c.1 + r.1));
            st(z, a2, (c.0 - r.0, c.1 - r.1));
        }
    }

    fn radix4(&self, z: &mut [f64], base: usize, m: usize, fstride: usize) {
        let sign = self.sign;
        for k in 0..m {
            let (a0, a1, a2, a3) = (base + k, base + k + m, base + k + 2 * m, base + k + 3 * m);
            let x0 = ld(z, a0);
            let x1 = cmul(ld(z, a1), self.twiddles[k * fstride]);
            let x2 = cmul(ld(z, a2), self.twiddles[2 * k * fstride]);
            let x3 = cmul(ld(z, a3), self.twiddles[3 * k * fstride]);
            let t0 = (x0.0 + x2.0, x0.1 + x2.1);
            let t1 = (x0.0 - x2.0, x0.1 - x2.1);
            let t2 = (x1.0 + x3.0, x1.1 + x3.1);
            let t3 = (x1.0 - x3.0, x1.1 - x3.1);
            // sign·i·t3
            let r = (-sign * t3.1, sign * t3.0);
            st(z, a0, (t0.0 + t2.0, t0.1 + t2.1));
            st(z, a2, (t0.0 - t2.0, t0.1 - t2.1));
            st(z, a1, (t1.0 + r.0, t1.1 + r.1));
            st(z, a3, (t1.0 - r.0, t1.1 - r.1));
        }
    }

    fn radix5(&self, z: &mut [f64], base: usize, m: usize, fstride: usize) {
        let (c1, s1) = (0.309_016_994_374_947_45, self.sign * 0.951_056_516_295_153_5);
        let (c2, s2) = (-0.809_016_994_374_947_5, self.sign * 0.587_785_252_292_473_1);
        for k in 0..m {
            let a = [base + k, base + k + m, base + k + 2 * m, base + k + 3 * m, base + k + 4 * m];
            let x0 = ld(z, a[0]);
            let x1 = cmul(ld(z, a[1]), self.twiddles[k * fstride]);
            let x2 = cmul(ld(z, a[2]), self.twiddles[2 * k * fstride]);
            let x3 = cmul(ld(z, a[3]), self.twiddles[3 * k * fstride]);
            let x4 = cmul(ld(z, a[4]), self.twiddles[4 * k * fstride]);
            let b1 = (x1.0 + x4.0, x1.1 + x4.1);
            let b2 = (x2.0 + x3.0, x2.1 + x3.1);
            let d1 = (x1.0 - x4.0, x1.1 - x4.1);
            let d2 = (x2.0 - x3.0, x2.1 - x3.1);
            let p1 = (x0.0 + c1 * b1.0 + c2 * b2.0, x0.1 + c1 * b1.1 + c2 * b2.1);
            let p2 = (x0.0 + c2 * b1.0 + c1 * b2.0, x0.1 + c2 * b1.1 + c1 * b2.1);
            // i·(s1·d1 + s2·d2) and i·(s2·d1 − s1·d2), signs folded in.
            let q1 = (-(s1 * d1.1 + s2 * d2.1), s1 * d1.0 + s2 * d2.0);
            let q2 = (-(s2 * d1.1 - s1 * d2.1), s2 * d1.0 - s1 * d2.0);
            st(z, a[0], (x0.0 + b1.0 + b2.0, x0.1 + b1.1 + b2.1));
            st(z, a[1], (p1.0 + q1.0, p1.1 + q1.1));
            st(z, a[4], (p1.0 - q1.0, p1.1 - q1.1));
            st(z, a[2], (p2.0 + q2.0, p2.1 + q2.1));
            st(z, a[3], (p2.0 - q2.0, p2.1 - q2.1));
        }
    }
}

/// `ff_tx_mdct_gen_exp`: the `n/2` pre/post rotations
/// `(cos α, sin α)·sqrt(|scale|)`, `α = π/2·(i + θ)/(n/2)`, with
/// `θ = 1/8` (plus `n/2` for a negative scale).
fn gen_exp(n: usize, scale: f64) -> Vec<[f64; 2]> {
    let len2 = n / 2;
    let theta = if scale < 0.0 { len2 as f64 } else { 0.0 } + 0.125;
    let s = scale.abs().sqrt();
    (0..len2)
        .map(|i| {
            let alpha = core::f64::consts::FRAC_PI_2 * (i as f64 + theta) / len2 as f64;
            [alpha.cos() * s, alpha.sin() * s]
        })
        .collect()
}

/// Inverse MDCT of `n` coefficients (`N = 2n` output samples):
/// `x[t] = (2/N)·Σ_k X[k]·cos((2π/N)(t + n0)(k + 1/2))`,
/// `n0 = (N/2 + 1)/2` — the §4.6.11.3.1 normalization.
#[derive(Debug)]
pub(crate) struct Imdct {
    /// Coefficients per transform (`N/2`).
    n: usize,
    /// The `n/2`-point inverse FFT.
    fft: Fft,
    /// Natural-order rotations ([`gen_exp`] at scale `−2/N`).
    exp: Vec<[f64; 2]>,
}

impl Imdct {
    /// A plan for `n` coefficients; `None` unless `n` is a multiple of 4
    /// whose half factors into 2, 3 and 5.
    pub(crate) fn new(n: usize) -> Option<Self> {
        if n == 0 || n % 4 != 0 {
            return None;
        }
        Some(Imdct {
            n,
            fft: Fft::new(n / 2, true)?,
            exp: gen_exp(n, -1.0 / n as f64),
        })
    }

    /// `ff_tx_mdct_inv`: the middle `n` samples `x[n/2 .. 3n/2]` of the
    /// `2n`-sample inverse transform of `src[..n]`, into `dst[..n]`.
    pub(crate) fn half(&self, src: &[f64], dst: &mut [f64]) {
        let n = self.n;
        let (len2, len4) = (n / 2, n / 4);
        let src = &src[..n];
        let dst = &mut dst[..n];
        for (i, &j) in self.fft.perm.iter().enumerate() {
            let j = j as usize;
            st(dst, i, cmul((src[n - 1 - 2 * j], src[2 * j]), self.exp[j]));
        }
        self.fft.run(dst);
        debug_assert_eq!(self.exp.len(), len2);
        for i in 0..len4 {
            let (i0, i1) = (len4 + i, len4 - i - 1);
            let (s1re, s1im) = (dst[2 * i1 + 1], dst[2 * i1]);
            let (s0re, s0im) = (dst[2 * i0 + 1], dst[2 * i0]);
            let [e1re, e1im] = self.exp[i1];
            let [e0re, e0im] = self.exp[i0];
            dst[2 * i1] = s1re * e1im - s1im * e1re;
            dst[2 * i0 + 1] = s1re * e1re + s1im * e1im;
            dst[2 * i0] = s0re * e0im - s0im * e0re;
            dst[2 * i1 + 1] = s0re * e0re + s0im * e0im;
        }
    }

    /// `ff_tx_mdct_inv_full`: all `2n` samples of the inverse transform of
    /// `src[..n]`, into `dst[..2n]`.
    pub(crate) fn full(&self, src: &[f64], dst: &mut [f64]) {
        let n = self.n;
        let dst = &mut dst[..2 * n];
        self.half(src, &mut dst[n / 2..n / 2 + n]);
        for i in 0..n / 2 {
            dst[i] = -dst[n - 1 - i];
            dst[2 * n - 1 - i] = dst[n + i];
        }
    }
}

/// Forward MDCT of `N = 2n` samples to `n` coefficients:
/// `X[k] = 2·Σ_t x[t]·cos((2π/N)(t + n0)(k + 1/2))` — the exact analysis
/// pair of [`Imdct`].
#[derive(Debug)]
pub(crate) struct Mdct {
    /// Coefficients per transform (`N/2`).
    n: usize,
    /// The `n/2`-point forward FFT.
    fft: Fft,
    /// `inv_perm[j]`: where natural-order FFT input `j` lands after the
    /// digit-reversal (av_tx's scatter map).
    inv_perm: Vec<u32>,
    /// Natural-order rotations ([`gen_exp`] at scale 2).
    exp: Vec<[f64; 2]>,
}

impl Mdct {
    /// A plan for `n` coefficients; `None` unless `n` is a multiple of 4
    /// whose half factors into 2, 3 and 5.
    pub(crate) fn new(n: usize) -> Option<Self> {
        if n == 0 || n % 4 != 0 {
            return None;
        }
        let fft = Fft::new(n / 2, false)?;
        let mut inv_perm = vec![0u32; n / 2];
        for (i, &j) in fft.perm.iter().enumerate() {
            inv_perm[j as usize] = i as u32;
        }
        Some(Mdct {
            n,
            fft,
            inv_perm,
            exp: gen_exp(n, 2.0),
        })
    }

    /// `ff_tx_mdct_fwd`: `src[..2n]` time samples to `dst[..n]`
    /// coefficients.
    pub(crate) fn forward(&self, src: &[f64], dst: &mut [f64]) {
        let n = self.n;
        let (len2, len4) = (n / 2, n / 4);
        let len3 = 3 * len2;
        let src = &src[..2 * n];
        let dst = &mut dst[..n];
        for i in 0..len2 {
            let k = 2 * i;
            let (re, im) = if k < len2 {
                (
                    -src[len2 + k] + src[len2 - 1 - k],
                    -src[len3 + k] - src[len3 - 1 - k],
                )
            } else {
                (
                    -src[len2 + k] - src[5 * len2 - 1 - k],
                    src[k - len2] - src[len3 - 1 - k],
                )
            };
            let [er, ei] = self.exp[i];
            // av_tx stores the rotated fold with re/im swapped.
            st(
                dst,
                self.inv_perm[i] as usize,
                (re * ei + im * er, re * er - im * ei),
            );
        }
        self.fft.run(dst);
        for i in 0..len4 {
            let (i0, i1) = (len4 + i, len4 - i - 1);
            let s1 = ld(dst, i1);
            let s0 = ld(dst, i0);
            let [e0re, e0im] = self.exp[i0];
            let [e1re, e1im] = self.exp[i1];
            dst[2 * i1 + 1] = s0.0 * e0im - s0.1 * e0re;
            dst[2 * i0] = s0.0 * e0re + s0.1 * e0im;
            dst[2 * i0 + 1] = s1.0 * e1im - s1.1 * e1re;
            dst[2 * i1] = s1.0 * e1re + s1.1 * e1im;
        }
    }
}

// Each length initializes independently: opening an LC stream does not
// build unused LD/ELD, SSR or 960-line plans.
macro_rules! cached_plans {
    ($($length:literal),+ $(,)?) => {
        pub(crate) const CACHED_LENGTHS: &[usize] = &[$($length),+];
        static IMDCT_PLANS: [LazyLock<Imdct>; CACHED_LENGTHS.len()] = [
            $(LazyLock::new(|| Imdct::new($length / 2).expect("fixed IMDCT length"))),+
        ];
        static MDCT_PLANS: [LazyLock<Mdct>; CACHED_LENGTHS.len()] = [
            $(LazyLock::new(|| Mdct::new($length / 2).expect("fixed MDCT length"))),+
        ];
    };
}
cached_plans!(2048, 1920, 1024, 960, 512, 480, 256, 240, 128, 120, 64);

/// The shared inverse plan for transform length `n_transform` (`N`), for
/// the lengths in [`CACHED_LENGTHS`].
pub(crate) fn imdct_plan(n_transform: usize) -> Option<&'static Imdct> {
    let slot = CACHED_LENGTHS.iter().position(|&l| l == n_transform)?;
    Some(&IMDCT_PLANS[slot])
}

/// The shared forward plan for transform length `n_transform` (`N`), for
/// the lengths in [`CACHED_LENGTHS`].
pub(crate) fn mdct_plan(n_transform: usize) -> Option<&'static Mdct> {
    let slot = CACHED_LENGTHS.iter().position(|&l| l == n_transform)?;
    Some(&MDCT_PLANS[slot])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imdct_direct(spec: &[f64], n_transform: usize) -> Vec<f64> {
        let n0 = (n_transform / 2 + 1) as f64 / 2.0;
        let step = 2.0 * core::f64::consts::PI / n_transform as f64;
        (0..n_transform)
            .map(|t| {
                2.0 / n_transform as f64
                    * spec
                        .iter()
                        .enumerate()
                        .map(|(k, &c)| c * (step * (t as f64 + n0) * (k as f64 + 0.5)).cos())
                        .sum::<f64>()
            })
            .collect()
    }

    fn mdct_direct(time: &[f64], n_transform: usize) -> Vec<f64> {
        let n0 = (n_transform / 2 + 1) as f64 / 2.0;
        let step = 2.0 * core::f64::consts::PI / n_transform as f64;
        (0..n_transform / 2)
            .map(|k| {
                2.0 * time
                    .iter()
                    .enumerate()
                    .map(|(t, &x)| x * (step * (t as f64 + n0) * (k as f64 + 0.5)).cos())
                    .sum::<f64>()
            })
            .collect()
    }

    /// Deterministic pseudo-random test signal in [-1, 1).
    fn signal(len: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    #[test]
    fn fft_lengths_factor() {
        for &n_transform in CACHED_LENGTHS {
            assert!(imdct_plan(n_transform).is_some(), "N = {n_transform}");
            assert!(mdct_plan(n_transform).is_some(), "N = {n_transform}");
        }
        assert!(Imdct::new(14).is_none());
        assert!(Imdct::new(28).is_none());
    }

    #[test]
    fn inverse_matches_the_direct_sum() {
        for n_transform in CACHED_LENGTHS.iter().copied().chain([8, 16, 40]) {
            let spec = signal(n_transform / 2, n_transform as u64);
            let want = imdct_direct(&spec, n_transform);
            let mut got = vec![0.0; n_transform];
            Imdct::new(n_transform / 2).unwrap().full(&spec, &mut got);
            let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (t, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-12 * peak, "N = {n_transform}, t = {t}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn forward_matches_the_direct_sum() {
        for n_transform in CACHED_LENGTHS.iter().copied().chain([8, 16, 40]) {
            let time = signal(n_transform, 7 + n_transform as u64);
            let want = mdct_direct(&time, n_transform);
            let mut got = vec![0.0; n_transform / 2];
            Mdct::new(n_transform / 2).unwrap().forward(&time, &mut got);
            let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-12 * peak, "N = {n_transform}, k = {k}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn half_is_the_middle_of_full() {
        let plan = imdct_plan(1920).unwrap();
        let spec = signal(960, 3);
        let mut full = vec![0.0; 1920];
        let mut half = vec![0.0; 960];
        plan.full(&spec, &mut full);
        plan.half(&spec, &mut half);
        assert_eq!(&full[480..1440], &half[..]);
    }
}
