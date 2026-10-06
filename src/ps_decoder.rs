//! Parametric stereo decoding (ISO/IEC 14496-3 §8.6.4, Annex 8.A) —
//! the `ps_data()` parse, the hybrid analysis / synthesis filterbank,
//! the de-correlator with its transient reduction, and the stereo
//! mixing with IPD/OPD phase smoothing.
//!
//! Ported from FFmpeg (commit 2da55bf) `libavcodec/aacps_common.c`
//! (`ff_ps_read_data`), `libavcodec/aacps.c` (`ff_ps_apply` and its
//! helpers), `libavcodec/aacpsdsp_template.c` (the float DSP kernels),
//! `libavcodec/aacps_tablegen.h` (`ps_tableinit`), and
//! `libavcodec/aacpsdata.c` (`ff_k_to_i_20` / `ff_k_to_i_34`). Those
//! files are Copyright (c) 2010 Alex Converse and are licensed under
//! the GNU Lesser General Public License version 2.1 or later; this
//! file is a derivative work under the same license
//! (LGPL-2.1-or-later). The behaviour follows FFmpeg's float decoder
//! exactly, including where it departs from the spec text (the parameter
//! hold across frames without `ps_data()`, the envelope fix-up, the
//! IPD/OPD smoother, the 20/34-band switch), so the decoded output
//! matches FFmpeg's.
//!
//! The Huffman codebooks are the crate's [`crate::ps_huffman`] tables
//! (cross-checked code-for-code against FFmpeg's `aacps_huff_tabs`).

use std::sync::LazyLock;

use oxideav_core::bits::BitReader;

use crate::ps_huffman::{
    ps_huff_dec, HUFF_ICC_DF, HUFF_ICC_DT, HUFF_IID_DF, HUFF_IID_DT, HUFF_IID_FINE_DF,
    HUFF_IID_FINE_DT, HUFF_IPD_DF, HUFF_IPD_DT, HUFF_OPD_DF, HUFF_OPD_DT,
};
use crate::sbr_qmf::Complex;
use crate::Result;

const PS_MAX_NUM_ENV: usize = 5;
const PS_MAX_NR_IIDICC: usize = 34;
const PS_MAX_SSB: usize = 91;
const PS_MAX_AP_BANDS: usize = 50;
const PS_QMF_TIME_SLOTS: usize = 32;
const PS_MAX_DELAY: usize = 14;
const PS_AP_LINKS: usize = 3;
const PS_MAX_AP_DELAY: usize = 5;
/// `numQMFSlots`: FFmpeg runs the PS tool over 32 QMF slots per frame.
const NUM_QMF_SLOTS: usize = 32;

/// Slots of the `X` matrix the PS tool reads: the 32 frame slots plus
/// the 6 look-ahead slots the hybrid analysis filters consume.
pub const PS_X_SLOTS: usize = 38;

/// The SBR `bs_extension_id` of a parametric stereo payload.
const EXTENSION_ID_PS: u32 = 2;

const NUM_ENV_TAB: [[usize; 4]; 2] = [[0, 1, 2, 4], [1, 2, 3, 4]];
const NR_IIDICC_PAR_TAB: [usize; 6] = [10, 20, 34, 10, 20, 34];
const NR_IIDOPD_PAR_TAB: [usize; 6] = [5, 11, 17, 5, 11, 17];

const NR_PAR_BANDS: [usize; 2] = [20, 34];
const NR_IPDOPD_BANDS: [usize; 2] = [11, 17];
const NR_BANDS: [usize; 2] = [71, 91];
const DECAY_CUTOFF: [i32; 2] = [10, 32];
const NR_ALLPASS_BANDS: [usize; 2] = [30, 50];
const SHORT_DELAY_BAND: [usize; 2] = [42, 62];

/// Table 8.48 in FFmpeg's hybrid-band order.
#[rustfmt::skip]
const K_TO_I_20: [u8; 71] = [
     1,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 14, 15,
    15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
];

#[rustfmt::skip]
const K_TO_I_34: [u8; 91] = [
     0,  1,  2,  3,  4,  5,  6,  6,  7,  2,  1,  0, 10, 10,  4,  5,  6,  7,  8,
     9, 10, 11, 12,  9, 14, 11, 12, 13, 14, 15, 16, 13, 16, 17, 18, 19, 20, 21,
    22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29,
    30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
];

const G1_Q2: [f32; 7] = [
    0.0,
    0.018_994_875,
    0.0,
    -0.072_931_39,
    0.0,
    0.305_966_3,
    0.5,
];

/// All-pass link coefficients `a[m]`.
const AP_A: [f32; PS_AP_LINKS] = [0.651_439_05, 0.564_718_1, 0.489_541_66];
const PEAK_DECAY_FACTOR: f32 = 0.765_928_35;
const TRANSIENT_IMPACT: f64 = 1.5;
const A_SMOOTH: f64 = 0.25;
const DECAY_SLOPE: f32 = 0.05;

type HybridFilter = [[f64; 2]; 8];

/// FFmpeg's `ps_tableinit()` tables, computed with the same float
/// arithmetic.
struct PsTables {
    pd_re_smooth: [f64; 512],
    pd_im_smooth: [f64; 512],
    ha: [[[f64; 4]; 8]; 46],
    hb: [[[f64; 4]; 8]; 46],
    f20_0_8: [HybridFilter; 8],
    f34_0_12: [HybridFilter; 12],
    f34_1_8: [HybridFilter; 8],
    f34_2_4: [HybridFilter; 4],
    q_fract_allpass: [[[[f64; 2]; PS_AP_LINKS]; 50]; 2],
    phi_fract: [[[f64; 2]; 50]; 2],
}

static TABLES: LazyLock<PsTables> = LazyLock::new(PsTables::new);

fn make_filters_from_proto(filter: &mut [HybridFilter], proto: &[f32; 7]) {
    let bands = filter.len();
    for (q, f) in filter.iter_mut().enumerate() {
        for n in 0..7 {
            let theta = 2.0 * std::f64::consts::PI * (q as f64 + 0.5) * (n as f64 - 6.0)
                / bands as f64;
            f[n][0] = (f64::from(proto[n]) * theta.cos()) as f32 as f64;
            f[n][1] = (f64::from(proto[n]) * -theta.sin()) as f32 as f64;
        }
    }
}

impl PsTables {
    fn new() -> Self {
        use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};
        let s = FRAC_1_SQRT_2 as f32;
        let ipdopd_sin: [f32; 8] = [0.0, s, 1.0, s, 0.0, -s, -1.0, -s];
        let ipdopd_cos: [f32; 8] = [1.0, s, 0.0, -s, -1.0, -s, 0.0, s];
        let mut pd_re_smooth = [0.0; 512];
        let mut pd_im_smooth = [0.0; 512];
        for pd0 in 0..8 {
            for pd1 in 0..8 {
                for pd2 in 0..8 {
                    let re_smooth: f32 =
                        0.25 * ipdopd_cos[pd0] + 0.5 * ipdopd_cos[pd1] + ipdopd_cos[pd2];
                    let im_smooth: f32 =
                        0.25 * ipdopd_sin[pd0] + 0.5 * ipdopd_sin[pd1] + ipdopd_sin[pd2];
                    let pd_mag = (1.0 / f64::from(im_smooth).hypot(f64::from(re_smooth))) as f32;
                    pd_re_smooth[pd0 * 64 + pd1 * 8 + pd2] = f64::from(re_smooth * pd_mag);
                    pd_im_smooth[pd0 * 64 + pd1 * 8 + pd2] = f64::from(im_smooth * pd_mag);
                }
            }
        }

        #[rustfmt::skip]
        let iid_par_dequant: [f64; 46] = [
            // iid_par_dequant_default
            0.05623413251903, 0.12589254117942, 0.19952623149689, 0.31622776601684,
            0.44668359215096, 0.63095734448019, 0.79432823472428, 1.0,
            1.25892541179417, 1.58489319246111, 2.23872113856834, 3.16227766016838,
            5.01187233627272, 7.94328234724282, 17.7827941003892,
            // iid_par_dequant_fine
            0.00316227766017, 0.00562341325190, 0.01,             0.01778279410039,
            0.03162277660168, 0.05623413251903, 0.07943282347243, 0.11220184543020,
            0.15848931924611, 0.22387211385683, 0.31622776601684, 0.39810717055350,
            0.50118723362727, 0.63095734448019, 0.79432823472428, 1.0,
            1.25892541179417, 1.58489319246111, 1.99526231496888, 2.51188643150958,
            3.16227766016838, 4.46683592150963, 6.30957344480193, 8.91250938133745,
            12.5892541179417, 17.7827941003892, 31.6227766016838, 56.2341325190349,
            100.0,            177.827941003892, 316.227766016837,
        ];
        let icc_invq: [f32; 8] = [1.0, 0.937, 0.84118, 0.60092, 0.36764, 0.0, -0.589, -1.0];
        let acos_icc_invq: [f32; 8] = [
            0.0,
            0.356_855_27,
            0.571_334_66,
            0.926_144_7,
            1.194_326_3,
            (PI / 2.0) as f32,
            2.200_617_1,
            PI as f32,
        ];
        let mut ha = [[[0.0; 4]; 8]; 46];
        let mut hb = [[[0.0; 4]; 8]; 46];
        for iid in 0..46 {
            let c = iid_par_dequant[iid] as f32;
            let c1 = SQRT_2 as f32 / (1.0f32 + c * c).sqrt();
            let c2 = c * c1;
            for icc in 0..8 {
                let alpha = 0.5f32 * acos_icc_invq[icc];
                let beta = alpha * (c1 - c2) * FRAC_1_SQRT_2 as f32;
                ha[iid][icc][0] = f64::from(c2 * (beta + alpha).cos());
                ha[iid][icc][1] = f64::from(c1 * (beta - alpha).cos());
                ha[iid][icc][2] = f64::from(c2 * (beta + alpha).sin());
                ha[iid][icc][3] = f64::from(c1 * (beta - alpha).sin());

                let rho = icc_invq[icc].max(0.05);
                let mut alpha = 0.5f32 * (2.0f32 * c * rho).atan2(c * c - 1.0);
                let mu = c + 1.0 / c;
                let mu = (1.0 + (4.0 * rho * rho - 4.0) / (mu * mu)).sqrt();
                let gamma = ((1.0 - mu) / (1.0 + mu)).sqrt().atan();
                if alpha < 0.0 {
                    alpha = (f64::from(alpha) + PI / 2.0) as f32;
                }
                let (alpha_c, alpha_s) = (alpha.cos(), alpha.sin());
                let (gamma_c, gamma_s) = (gamma.cos(), gamma.sin());
                hb[iid][icc][0] =
                    (SQRT_2 * f64::from(alpha_c) * f64::from(gamma_c)) as f32 as f64;
                hb[iid][icc][1] =
                    (SQRT_2 * f64::from(alpha_s) * f64::from(gamma_c)) as f32 as f64;
                hb[iid][icc][2] =
                    (-SQRT_2 * f64::from(alpha_s) * f64::from(gamma_s)) as f32 as f64;
                hb[iid][icc][3] =
                    (SQRT_2 * f64::from(alpha_c) * f64::from(gamma_s)) as f32 as f64;
            }
        }

        let f_center_20: [i8; 10] = [-3, -1, 1, 3, 5, 7, 10, 14, 18, 22];
        #[rustfmt::skip]
        let f_center_34: [i8; 32] = [
             2,  6, 10, 14, 18, 22, 26, 30,
            34,-10, -6, -2, 51, 57, 15, 21,
            27, 33, 39, 45, 54, 66, 78, 42,
           102, 66, 78, 90,102,114,126, 90,
        ];
        let fractional_delay_links: [f32; 3] = [0.43, 0.75, 0.347];
        let fractional_delay_gain: f32 = 0.39;
        let mut q_fract_allpass = [[[[0.0; 2]; PS_AP_LINKS]; 50]; 2];
        let mut phi_fract = [[[0.0; 2]; 50]; 2];
        for k in 0..NR_ALLPASS_BANDS[0] {
            let f_center = if k < f_center_20.len() {
                f64::from(f_center_20[k]) * 0.125
            } else {
                f64::from(k as f32 - 6.5f32)
            };
            for m in 0..PS_AP_LINKS {
                let theta = -PI * f64::from(fractional_delay_links[m]) * f_center;
                q_fract_allpass[0][k][m] = [theta.cos() as f32 as f64, theta.sin() as f32 as f64];
            }
            let theta = -PI * f64::from(fractional_delay_gain) * f_center;
            phi_fract[0][k] = [theta.cos() as f32 as f64, theta.sin() as f32 as f64];
        }
        for k in 0..NR_ALLPASS_BANDS[1] {
            let f_center = if k < f_center_34.len() {
                f64::from(f_center_34[k]) / 24.0
            } else {
                f64::from(k as f32 - 26.5f32)
            };
            for m in 0..PS_AP_LINKS {
                let theta = -PI * f64::from(fractional_delay_links[m]) * f_center;
                q_fract_allpass[1][k][m] = [theta.cos() as f32 as f64, theta.sin() as f32 as f64];
            }
            let theta = -PI * f64::from(fractional_delay_gain) * f_center;
            phi_fract[1][k] = [theta.cos() as f32 as f64, theta.sin() as f32 as f64];
        }

        let g0_q8: [f32; 7] = [
            0.007_460_829_5,
            0.022_704_21,
            0.045_468_66,
            0.072_661_14,
            0.098_851_09,
            0.117_937_11,
            0.125,
        ];
        let g0_q12: [f32; 7] = [
            0.040_811_8,
            0.038_128_11,
            0.051_449_08,
            0.063_998_31,
            0.074_283_14,
            0.081_003_48,
            0.083_333_33,
        ];
        let g1_q8: [f32; 7] = [
            0.015_656_756,
            0.037_527_164,
            0.054_178_914,
            0.084_170_44,
            0.103_073_44,
            0.122_224_52,
            0.125,
        ];
        let g2_q4: [f32; 7] = [
            -0.059_082_11,
            -0.048_714_984,
            0.0,
            0.077_787_24,
            0.164_863_03,
            0.232_798_57,
            0.25,
        ];
        let mut f20_0_8 = [[[0.0; 2]; 8]; 8];
        let mut f34_0_12 = [[[0.0; 2]; 8]; 12];
        let mut f34_1_8 = [[[0.0; 2]; 8]; 8];
        let mut f34_2_4 = [[[0.0; 2]; 8]; 4];
        make_filters_from_proto(&mut f20_0_8, &g0_q8);
        make_filters_from_proto(&mut f34_0_12, &g0_q12);
        make_filters_from_proto(&mut f34_1_8, &g1_q8);
        make_filters_from_proto(&mut f34_2_4, &g2_q4);

        PsTables {
            pd_re_smooth,
            pd_im_smooth,
            ha,
            hb,
            f20_0_8,
            f34_0_12,
            f34_1_8,
            f34_2_4,
            q_fract_allpass,
            phi_fract,
        }
    }
}

/// Which parameter set a `read_*_data` call fills.
#[derive(Clone, Copy)]
enum Par {
    Iid,
    Icc,
    Ipd,
    Opd,
}

/// One parametric-stereo context (FFmpeg `PSContext` with its
/// `PSCommonContext`), owned by the single-channel SBR element that
/// carries the PS payload.
#[derive(Clone)]
pub struct PsDecoder {
    // PSCommonContext
    start: bool,
    enable_iid: bool,
    iid_quant: bool,
    nr_iid_par: usize,
    nr_ipdopd_par: usize,
    enable_icc: bool,
    icc_mode: u32,
    nr_icc_par: usize,
    enable_ext: bool,
    num_env_old: usize,
    num_env: usize,
    enable_ipdopd: bool,
    border_position: [i32; PS_MAX_NUM_ENV + 1],
    iid_par: [[i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
    icc_par: [[i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
    ipd_par: [[i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
    opd_par: [[i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
    is34bands: bool,
    is34bands_old: bool,
    // PSContext
    in_buf: [[Complex; 44]; 5],
    delay: Vec<[Complex; PS_QMF_TIME_SLOTS + PS_MAX_DELAY]>,
    ap_delay: Vec<[[Complex; PS_QMF_TIME_SLOTS + PS_MAX_AP_DELAY]; PS_AP_LINKS]>,
    peak_decay_nrg: [f64; 34],
    power_smooth: [f64; 34],
    peak_decay_diff_smooth: [f64; 34],
    /// `H11`, `H12`, `H21`, `H22`, each `[re/im][env][band]`.
    h: [[[[f64; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV + 1]; 2]; 4],
    lbuf: Vec<[Complex; 32]>,
    rbuf: Vec<[Complex; 32]>,
    opd_hist: [i8; PS_MAX_NR_IIDICC],
    ipd_hist: [i8; PS_MAX_NR_IIDICC],
}

impl std::fmt::Debug for PsDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PsDecoder")
            .field("start", &self.start)
            .field("num_env", &self.num_env)
            .field("is34bands", &self.is34bands)
            .field("enable_ipdopd", &self.enable_ipdopd)
            .finish_non_exhaustive()
    }
}

impl Default for PsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PsDecoder {
    /// A fresh context: no parameters received, so [`Self::started`]
    /// is false (the caller duplicates the mono signal).
    #[must_use]
    pub fn new() -> Self {
        PsDecoder {
            start: false,
            enable_iid: false,
            iid_quant: false,
            nr_iid_par: 0,
            nr_ipdopd_par: 0,
            enable_icc: false,
            icc_mode: 0,
            nr_icc_par: 0,
            enable_ext: false,
            num_env_old: 0,
            num_env: 0,
            enable_ipdopd: false,
            border_position: [0; PS_MAX_NUM_ENV + 1],
            iid_par: [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
            icc_par: [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
            ipd_par: [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
            opd_par: [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV],
            is34bands: false,
            is34bands_old: false,
            in_buf: [[Complex::default(); 44]; 5],
            delay: vec![[Complex::default(); PS_QMF_TIME_SLOTS + PS_MAX_DELAY]; PS_MAX_SSB],
            ap_delay: vec![
                [[Complex::default(); PS_QMF_TIME_SLOTS + PS_MAX_AP_DELAY]; PS_AP_LINKS];
                PS_MAX_AP_BANDS
            ],
            peak_decay_nrg: [0.0; 34],
            power_smooth: [0.0; 34],
            peak_decay_diff_smooth: [0.0; 34],
            h: [[[[0.0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV + 1]; 2]; 4],
            lbuf: vec![[Complex::default(); 32]; PS_MAX_SSB],
            rbuf: vec![[Complex::default(); 32]; PS_MAX_SSB],
            opd_hist: [0; PS_MAX_NR_IIDICC],
            ipd_hist: [0; PS_MAX_NR_IIDICC],
        }
    }

    /// Whether a header-carrying `ps_data()` has been decoded without
    /// error since the last failure (FFmpeg `ps->common.start`): the PS
    /// tool renders stereo only once this is set.
    #[must_use]
    pub fn started(&self) -> bool {
        self.start
    }

    /// Decode one `sbr_extension()` block whose first `bs_extension_id`
    /// was `EXTENSION_ID_PS`. `data` holds every bit after that id,
    /// zero-padded to a byte: `8 · data.len() − 2` payload bits (the
    /// block is `bs_extension_size` whole bytes). Further extensions in
    /// the remaining bits are walked like FFmpeg's `read_sbr_data` loop:
    /// another PS id parses again, anything else is skipped.
    pub fn read_extension(&mut self, data: &[u8]) {
        let Some(mut bits_left) = (data.len() * 8).checked_sub(2) else {
            return;
        };
        let mut br = BitReader::new(data);
        bits_left -= self.read_data(&mut br, bits_left);
        while bits_left > 7 {
            let Ok(id) = br.read_u32(2) else {
                return;
            };
            bits_left -= 2;
            if id != EXTENSION_ID_PS {
                return;
            }
            bits_left -= self.read_data(&mut br, bits_left);
        }
    }

    /// `ff_ps_read_data`: returns the bits consumed (all of
    /// `bits_left` on a parse error, which also resets the parameters).
    fn read_data(&mut self, br: &mut BitReader<'_>, bits_left: usize) -> usize {
        let start = br.bit_position();
        let parsed = self.parse(br);
        let consumed = (br.bit_position() - start) as usize;
        if parsed.is_ok() && consumed <= bits_left {
            return consumed;
        }
        self.start = false;
        self.iid_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        self.icc_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        self.ipd_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        self.opd_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        bits_left
    }

    fn parse(&mut self, br: &mut BitReader<'_>) -> std::result::Result<(), ()> {
        let bit = |br: &mut BitReader<'_>| br.read_bit().map_err(|_| ());
        let bits = |br: &mut BitReader<'_>, n: u32| br.read_u32(n).map_err(|_| ());

        let header = bit(br)?;
        if header {
            self.enable_iid = bit(br)?;
            if self.enable_iid {
                let iid_mode = bits(br, 3)? as usize;
                if iid_mode > 5 {
                    return Err(());
                }
                self.nr_iid_par = NR_IIDICC_PAR_TAB[iid_mode];
                self.iid_quant = iid_mode > 2;
                self.nr_ipdopd_par = NR_IIDOPD_PAR_TAB[iid_mode];
            }
            self.enable_icc = bit(br)?;
            if self.enable_icc {
                self.icc_mode = bits(br, 3)?;
                if self.icc_mode > 5 {
                    return Err(());
                }
                self.nr_icc_par = NR_IIDICC_PAR_TAB[self.icc_mode as usize];
            }
            self.enable_ext = bit(br)?;
        }

        let frame_class = bit(br)?;
        self.num_env_old = self.num_env;
        self.num_env = NUM_ENV_TAB[usize::from(frame_class)][bits(br, 2)? as usize];

        self.border_position[0] = -1;
        if frame_class {
            for e in 1..=self.num_env {
                self.border_position[e] = bits(br, 5)? as i32;
                if self.border_position[e] < self.border_position[e - 1] {
                    return Err(());
                }
            }
        } else {
            let log2 = match self.num_env {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            };
            for e in 1..=self.num_env {
                self.border_position[e] = ((e * NUM_QMF_SLOTS) >> log2) as i32 - 1;
            }
        }

        if self.enable_iid {
            for e in 0..self.num_env {
                let dt = bit(br)?;
                let table: &[(u8, u32)] = match (dt, self.iid_quant) {
                    (false, false) => &HUFF_IID_DF,
                    (false, true) => &HUFF_IID_FINE_DF,
                    (true, false) => &HUFF_IID_DT,
                    (true, true) => &HUFF_IID_FINE_DT,
                };
                self.read_par(br, Par::Iid, table, e, dt)?;
            }
        } else {
            self.iid_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        }

        if self.enable_icc {
            for e in 0..self.num_env {
                let dt = bit(br)?;
                let table: &[(u8, u32)] = if dt { &HUFF_ICC_DT } else { &HUFF_ICC_DF };
                self.read_par(br, Par::Icc, table, e, dt)?;
            }
        } else {
            self.icc_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        }

        if self.enable_ext {
            let mut cnt = bits(br, 4)? as i64;
            if cnt == 15 {
                cnt += i64::from(bits(br, 8)?);
            }
            cnt *= 8;
            while cnt > 7 {
                let ps_extension_id = bits(br, 2)?;
                cnt -= 2 + self.read_extension_data(br, ps_extension_id)? as i64;
            }
            if cnt < 0 {
                return Err(());
            }
            br.skip(cnt as u32).map_err(|_| ())?;
        }

        // Fix up envelopes: a frame whose last border stops short of the
        // frame end gets a copy of its last envelope reaching the end.
        let ne = self.num_env;
        if ne == 0 || self.border_position[ne] < NUM_QMF_SLOTS as i32 - 1 {
            let source = if ne > 0 {
                ne as i32 - 1
            } else {
                self.num_env_old as i32 - 1
            };
            if source >= 0 && source as usize != ne {
                let source = source as usize;
                if self.enable_iid {
                    self.iid_par[ne] = self.iid_par[source];
                }
                if self.enable_icc {
                    self.icc_par[ne] = self.icc_par[source];
                }
                if self.enable_ipdopd {
                    self.ipd_par[ne] = self.ipd_par[source];
                    self.opd_par[ne] = self.opd_par[source];
                }
            }
            if self.enable_iid {
                let bound = 7 + 8 * i32::from(self.iid_quant);
                for b in 0..self.nr_iid_par {
                    if i32::from(self.iid_par[ne][b]).abs() > bound {
                        return Err(());
                    }
                }
            }
            if self.enable_icc {
                // FFmpeg bounds this check by `nr_iid_par`.
                for b in 0..self.nr_iid_par {
                    if self.icc_par[ne][b] as u8 > 7 {
                        return Err(());
                    }
                }
            }
            self.num_env += 1;
            self.border_position[self.num_env] = NUM_QMF_SLOTS as i32 - 1;
        }

        self.is34bands_old = self.is34bands;
        if self.enable_iid || self.enable_icc {
            self.is34bands = (self.enable_iid && self.nr_iid_par == 34)
                || (self.enable_icc && self.nr_icc_par == 34);
        }

        if !self.enable_ipdopd {
            self.ipd_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
            self.opd_par = [[0; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        }

        if header {
            self.start = true;
        }
        Ok(())
    }

    /// `ps_read_extension_data`: the IPD/OPD extension (id 0); other
    /// ids consume nothing.
    fn read_extension_data(
        &mut self,
        br: &mut BitReader<'_>,
        ps_extension_id: u32,
    ) -> std::result::Result<u64, ()> {
        if ps_extension_id != 0 {
            return Ok(0);
        }
        let count = br.bit_position();
        self.enable_ipdopd = br.read_bit().map_err(|_| ())?;
        if self.enable_ipdopd {
            for e in 0..self.num_env {
                let dt = br.read_bit().map_err(|_| ())?;
                self.read_par(br, Par::Ipd, if dt { &HUFF_IPD_DT } else { &HUFF_IPD_DF }, e, dt)?;
                let dt = br.read_bit().map_err(|_| ())?;
                self.read_par(br, Par::Opd, if dt { &HUFF_OPD_DT } else { &HUFF_OPD_DF }, e, dt)?;
            }
        }
        br.skip(1).map_err(|_| ())?; // reserved_ps
        Ok(br.bit_position() - count)
    }

    /// FFmpeg's `READ_PAR_DATA` family: one envelope of IID / ICC /
    /// IPD / OPD indices, time- (`dt`) or frequency-differential.
    fn read_par(
        &mut self,
        br: &mut BitReader<'_>,
        which: Par,
        table: &[(u8, u32)],
        e: usize,
        dt: bool,
    ) -> std::result::Result<(), ()> {
        let (num, lav, mask) = match which {
            Par::Iid => (self.nr_iid_par, (table.len() as i32 - 1) / 2, false),
            Par::Icc => (self.nr_icc_par, (table.len() as i32 - 1) / 2, false),
            Par::Ipd | Par::Opd => (self.nr_ipdopd_par, 0, true),
        };
        let iid_bound = 7 + 8 * i32::from(self.iid_quant);
        let e_prev = if e > 0 {
            e - 1
        } else {
            self.num_env_old.saturating_sub(1)
        };
        let par = match which {
            Par::Iid => &mut self.iid_par,
            Par::Icc => &mut self.icc_par,
            Par::Ipd => &mut self.ipd_par,
            Par::Opd => &mut self.opd_par,
        };
        let mut val = 0i32;
        for b in 0..num.min(PS_MAX_NR_IIDICC) {
            let delta = ps_huff_dec(br, table, lav).map_err(|_| ())?;
            if dt {
                val = i32::from(par[e_prev][b]) + delta;
            } else {
                val += delta;
            }
            if mask {
                val &= 7;
            }
            par[e][b] = val as i8;
            let stored = i32::from(par[e][b]);
            let bad = match which {
                Par::Iid => stored.abs() > iid_bound,
                Par::Icc => !(0..=7).contains(&stored),
                Par::Ipd | Par::Opd => false,
            };
            if bad {
                return Err(());
            }
        }
        Ok(())
    }

    /// `ff_ps_apply`: run the PS tool over the left channel's `X`
    /// matrix (`l`, rewritten in place) and render the right channel's
    /// into `r`. `top` is `kx + M` of the current SBR frame; only the
    /// first [`NUM_QMF_SLOTS`] slots of either matrix are written.
    pub fn apply(
        &mut self,
        l: &mut [[Complex; 64]; PS_X_SLOTS],
        r: &mut [[Complex; 64]; PS_X_SLOTS],
        top: usize,
    ) -> Result<()> {
        let is34 = usize::from(self.is34bands);
        let top = (top + NR_BANDS[is34]).saturating_sub(64).min(NR_BANDS[is34]);
        for d in &mut self.delay[top..NR_BANDS[is34]] {
            *d = [Complex::default(); PS_QMF_TIME_SLOTS + PS_MAX_DELAY];
        }
        if top < NR_ALLPASS_BANDS[is34] {
            for d in &mut self.ap_delay[top..NR_ALLPASS_BANDS[is34]] {
                *d = [[Complex::default(); PS_QMF_TIME_SLOTS + PS_MAX_AP_DELAY]; PS_AP_LINKS];
            }
        }
        self.hybrid_analysis(l, is34 == 1);
        self.decorrelation(is34 == 1);
        self.stereo_processing(is34 == 1);
        hybrid_synthesis(l, &self.lbuf, is34 == 1);
        hybrid_synthesis(r, &self.rbuf, is34 == 1);
        Ok(())
    }

    fn hybrid_analysis(&mut self, l: &[[Complex; 64]; PS_X_SLOTS], is34: bool) {
        let t = &*TABLES;
        for i in 0..5 {
            for j in 0..PS_X_SLOTS {
                self.in_buf[i][j + 6] = l[j][i];
            }
        }
        let len = NUM_QMF_SLOTS;
        if is34 {
            hybrid4_8_12_cx(&self.in_buf[0], &mut self.lbuf[0..12], &t.f34_0_12, len);
            hybrid4_8_12_cx(&self.in_buf[1], &mut self.lbuf[12..20], &t.f34_1_8, len);
            hybrid4_8_12_cx(&self.in_buf[2], &mut self.lbuf[20..24], &t.f34_2_4, len);
            hybrid4_8_12_cx(&self.in_buf[3], &mut self.lbuf[24..28], &t.f34_2_4, len);
            hybrid4_8_12_cx(&self.in_buf[4], &mut self.lbuf[28..32], &t.f34_2_4, len);
            for i in 5..64 {
                for j in 0..len {
                    self.lbuf[27 + i][j] = l[j][i];
                }
            }
        } else {
            hybrid6_cx(&self.in_buf[0], &mut self.lbuf[0..6], &t.f20_0_8, len);
            hybrid2_re(&self.in_buf[1], &mut self.lbuf[6..8], len, true);
            hybrid2_re(&self.in_buf[2], &mut self.lbuf[8..10], len, false);
            for i in 3..64 {
                for j in 0..len {
                    self.lbuf[7 + i][j] = l[j][i];
                }
            }
        }
        for row in &mut self.in_buf {
            row.copy_within(32..38, 0);
        }
    }

    fn decorrelation(&mut self, is34: bool) {
        let t = &*TABLES;
        let s = usize::from(is34);
        let k_to_i: &[u8] = if is34 { &K_TO_I_34 } else { &K_TO_I_20 };
        let mut power = [[0.0f64; PS_QMF_TIME_SLOTS]; 34];
        let mut transient_gain = [[0.0f64; PS_QMF_TIME_SLOTS]; 34];
        let n_l = NUM_QMF_SLOTS;

        if is34 != self.is34bands_old {
            self.peak_decay_nrg = [0.0; 34];
            self.power_smooth = [0.0; 34];
            self.peak_decay_diff_smooth = [0.0; 34];
            for d in &mut self.delay {
                *d = [Complex::default(); PS_QMF_TIME_SLOTS + PS_MAX_DELAY];
            }
            for d in &mut self.ap_delay {
                *d = [[Complex::default(); PS_QMF_TIME_SLOTS + PS_MAX_AP_DELAY]; PS_AP_LINKS];
            }
        }

        for k in 0..NR_BANDS[s] {
            let i = usize::from(k_to_i[k]);
            for n in 0..n_l {
                let v = self.lbuf[k][n];
                power[i][n] += v.re * v.re + v.im * v.im;
            }
        }

        // Transient detection.
        let peak_decay_factor = f64::from(PEAK_DECAY_FACTOR);
        for i in 0..NR_PAR_BANDS[s] {
            for n in 0..n_l {
                let decayed_peak = peak_decay_factor * self.peak_decay_nrg[i];
                self.peak_decay_nrg[i] = if decayed_peak > power[i][n] {
                    decayed_peak
                } else {
                    power[i][n]
                };
                self.power_smooth[i] += A_SMOOTH * (power[i][n] - self.power_smooth[i]);
                self.peak_decay_diff_smooth[i] += A_SMOOTH
                    * (self.peak_decay_nrg[i] - power[i][n] - self.peak_decay_diff_smooth[i]);
                let denom = TRANSIENT_IMPACT * self.peak_decay_diff_smooth[i];
                transient_gain[i][n] = if denom > self.power_smooth[i] {
                    self.power_smooth[i] / denom
                } else {
                    1.0
                };
            }
        }

        // De-correlation and transient reduction.
        for k in 0..NR_ALLPASS_BANDS[s] {
            let b = usize::from(k_to_i[k]);
            let g_decay_slope = (1.0f32 - DECAY_SLOPE * (k as i32 - DECAY_CUTOFF[s]) as f32)
                .clamp(0.0, 1.0);
            self.delay[k].copy_within(n_l..n_l + PS_MAX_DELAY, 0);
            self.delay[k][PS_MAX_DELAY..].copy_from_slice(&self.lbuf[k][..NUM_QMF_SLOTS]);
            for link in &mut self.ap_delay[k] {
                link.copy_within(NUM_QMF_SLOTS..NUM_QMF_SLOTS + PS_MAX_AP_DELAY, 0);
            }
            decorrelate(
                &mut self.rbuf[k],
                &self.delay[k][PS_MAX_DELAY - 2..],
                &mut self.ap_delay[k],
                &t.phi_fract[s][k],
                &t.q_fract_allpass[s][k],
                &transient_gain[b],
                f64::from(g_decay_slope),
            );
        }
        for k in NR_ALLPASS_BANDS[s]..NR_BANDS[s] {
            let i = usize::from(k_to_i[k]);
            self.delay[k].copy_within(n_l..n_l + PS_MAX_DELAY, 0);
            self.delay[k][PS_MAX_DELAY..].copy_from_slice(&self.lbuf[k][..NUM_QMF_SLOTS]);
            // Bands below SHORT_DELAY_BAND delay by 14 slots, the rest by 1.
            let d = if k < SHORT_DELAY_BAND[s] { 14 } else { 1 };
            for n in 0..n_l {
                self.rbuf[k][n] = self.delay[k][PS_MAX_DELAY - d + n] * transient_gain[i][n];
            }
        }
    }

    fn stereo_processing(&mut self, is34: bool) {
        let t = &*TABLES;
        let s = usize::from(is34);
        let k_to_i: &[u8] = if is34 { &K_TO_I_34 } else { &K_TO_I_20 };
        let h_lut = if self.icc_mode < 3 { &t.ha } else { &t.hb };
        let num_env = self.num_env.min(PS_MAX_NUM_ENV);

        // Remapping.
        if self.num_env_old > 0 {
            let old = self.num_env_old.min(PS_MAX_NUM_ENV);
            for hm in &mut self.h {
                for plane in hm.iter_mut() {
                    plane[0] = plane[old];
                }
            }
        }

        let mut iid_mapped = [[0i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        let mut icc_mapped = [[0i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        let mut ipd_mapped = [[0i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        let mut opd_mapped = [[0i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];
        if is34 {
            remap34(&mut iid_mapped, &self.iid_par, self.nr_iid_par, num_env, true);
            remap34(&mut icc_mapped, &self.icc_par, self.nr_icc_par, num_env, true);
            if self.enable_ipdopd {
                remap34(&mut ipd_mapped, &self.ipd_par, self.nr_ipdopd_par, num_env, false);
                remap34(&mut opd_mapped, &self.opd_par, self.nr_ipdopd_par, num_env, false);
            }
            if !self.is34bands_old {
                for hm in &mut self.h {
                    for plane in hm.iter_mut() {
                        map_val_20_to_34(&mut plane[0]);
                    }
                }
                self.ipd_hist = [0; PS_MAX_NR_IIDICC];
                self.opd_hist = [0; PS_MAX_NR_IIDICC];
            }
        } else {
            remap20(&mut iid_mapped, &self.iid_par, self.nr_iid_par, num_env, true);
            remap20(&mut icc_mapped, &self.icc_par, self.nr_icc_par, num_env, true);
            if self.enable_ipdopd {
                remap20(&mut ipd_mapped, &self.ipd_par, self.nr_ipdopd_par, num_env, false);
                remap20(&mut opd_mapped, &self.opd_par, self.nr_ipdopd_par, num_env, false);
            }
            if self.is34bands_old {
                for hm in &mut self.h {
                    for plane in hm.iter_mut() {
                        map_val_34_to_20(&mut plane[0]);
                    }
                }
                self.ipd_hist = [0; PS_MAX_NR_IIDICC];
                self.opd_hist = [0; PS_MAX_NR_IIDICC];
            }
        }

        // Mixing.
        let iid_offset = 7 + 23 * i32::from(self.iid_quant);
        for e in 0..num_env {
            for b in 0..NR_PAR_BANDS[s] {
                let iid = (i32::from(iid_mapped[e][b]) + iid_offset).clamp(0, 45) as usize;
                let icc = (i32::from(icc_mapped[e][b])).clamp(0, 7) as usize;
                let [mut h11, mut h12, mut h21, mut h22] = h_lut[iid][icc];
                if self.enable_ipdopd && b < NR_IPDOPD_BANDS[s] {
                    let opd_idx = (self.opd_hist[b] as u8 as usize & 0x3F) * 8
                        + (opd_mapped[e][b] as u8 as usize & 7);
                    let ipd_idx = (self.ipd_hist[b] as u8 as usize & 0x3F) * 8
                        + (ipd_mapped[e][b] as u8 as usize & 7);
                    let opd_re = t.pd_re_smooth[opd_idx];
                    let opd_im = t.pd_im_smooth[opd_idx];
                    let ipd_re = t.pd_re_smooth[ipd_idx];
                    let ipd_im = t.pd_im_smooth[ipd_idx];
                    self.opd_hist[b] = (opd_idx & 0x3F) as i8;
                    self.ipd_hist[b] = (ipd_idx & 0x3F) as i8;

                    let ipd_adj_re = opd_re * ipd_re + opd_im * ipd_im;
                    let ipd_adj_im = opd_im * ipd_re - opd_re * ipd_im;
                    let h11i = h11 * opd_im;
                    h11 *= opd_re;
                    let h12i = h12 * ipd_adj_im;
                    h12 *= ipd_adj_re;
                    let h21i = h21 * opd_im;
                    h21 *= opd_re;
                    let h22i = h22 * ipd_adj_im;
                    h22 *= ipd_adj_re;
                    self.h[0][1][e + 1][b] = h11i;
                    self.h[1][1][e + 1][b] = h12i;
                    self.h[2][1][e + 1][b] = h21i;
                    self.h[3][1][e + 1][b] = h22i;
                }
                self.h[0][0][e + 1][b] = h11;
                self.h[1][0][e + 1][b] = h12;
                self.h[2][0][e + 1][b] = h21;
                self.h[3][0][e + 1][b] = h22;
            }
            let start = self.border_position[e];
            let stop = self.border_position[e + 1];
            let span = stop - start;
            let width = 1.0f32 / (if span != 0 { span } else { 1 }) as f32;
            let width = f64::from(width);
            for k in 0..NR_BANDS[s] {
                let b = usize::from(k_to_i[k]);
                let mut h = [[0.0f64; 4]; 2];
                let mut h_step = [[0.0f64; 4]; 2];
                for (j, hm) in self.h.iter().enumerate() {
                    h[0][j] = hm[0][e][b];
                }
                if self.enable_ipdopd {
                    let negate = (is34 && (9..=13).contains(&k)) || (!is34 && k <= 1);
                    for (j, hm) in self.h.iter().enumerate() {
                        h[1][j] = if negate { -hm[1][e][b] } else { hm[1][e][b] };
                    }
                }
                for (j, hm) in self.h.iter().enumerate() {
                    h_step[0][j] = (hm[0][e + 1][b] - h[0][j]) * width;
                }
                if self.enable_ipdopd {
                    for (j, hm) in self.h.iter().enumerate() {
                        h_step[1][j] = (hm[1][e + 1][b] - h[1][j]) * width;
                    }
                }
                if span > 0 {
                    let first = (start + 1).clamp(0, NUM_QMF_SLOTS as i32) as usize;
                    let last = (first + span as usize).min(NUM_QMF_SLOTS);
                    let (lk, rk) = (&mut self.lbuf[k][first..last], &mut self.rbuf[k][first..last]);
                    if self.enable_ipdopd {
                        stereo_interpolate_ipdopd(lk, rk, h, h_step);
                    } else {
                        stereo_interpolate(lk, rk, h[0], h_step[0]);
                    }
                }
            }
        }
    }
}

/// `ps_hybrid_analysis_c`: `out[q]` is filter `q` over the 13 samples
/// `input[0..13]`.
fn hybrid_analysis_kernel(input: &[Complex], filter: &[HybridFilter], out: &mut [Complex]) {
    let mut inre0 = [0.0f64; 6];
    let mut inre1 = [0.0f64; 6];
    let mut inim0 = [0.0f64; 6];
    let mut inim1 = [0.0f64; 6];
    for j in 0..6 {
        inre0[j] = input[j].re + input[12 - j].re;
        inre1[j] = input[j].im - input[12 - j].im;
        inim0[j] = input[j].im + input[12 - j].im;
        inim1[j] = input[j].re - input[12 - j].re;
    }
    for (o, f) in out.iter_mut().zip(filter) {
        let mut sum_re = f[6][0] * input[6].re;
        let mut sum_im = f[6][0] * input[6].im;
        for j in 0..6 {
            sum_re += f[j][0] * inre0[j] - f[j][1] * inre1[j];
            sum_im += f[j][0] * inim0[j] + f[j][1] * inim1[j];
        }
        *o = Complex::new(sum_re, sum_im);
    }
}

/// `hybrid6_cx`: split QMF band 0 into six sub-subbands (20-band mode).
fn hybrid6_cx(input: &[Complex; 44], out: &mut [[Complex; 32]], filter: &[HybridFilter; 8], len: usize) {
    for i in 0..len {
        let mut temp = [Complex::default(); 8];
        hybrid_analysis_kernel(&input[i..i + 13], filter, &mut temp);
        out[0][i] = temp[6];
        out[1][i] = temp[7];
        out[2][i] = temp[0];
        out[3][i] = temp[1];
        out[4][i] = temp[2] + temp[5];
        out[5][i] = temp[3] + temp[4];
    }
}

/// `hybrid4_8_12_cx`: split one QMF band into `filter.len()`
/// sub-subbands (34-band mode).
fn hybrid4_8_12_cx(input: &[Complex; 44], out: &mut [[Complex; 32]], filter: &[HybridFilter], len: usize) {
    let n = filter.len();
    for i in 0..len {
        let mut temp = [Complex::default(); 12];
        hybrid_analysis_kernel(&input[i..i + 13], filter, &mut temp[..n]);
        for (q, o) in out.iter_mut().enumerate().take(n) {
            o[i] = temp[q];
        }
    }
}

/// `hybrid2_re`: split one QMF band into two sub-subbands with the
/// symmetric real filter `g1_Q2`.
fn hybrid2_re(input: &[Complex; 44], out: &mut [[Complex; 32]], len: usize, reverse: bool) {
    let g: [f64; 7] = G1_Q2.map(f64::from);
    let (a, b) = if reverse { (1, 0) } else { (0, 1) };
    for i in 0..len {
        let x = &input[i..i + 13];
        let re_in = g[6] * x[6].re;
        let im_in = g[6] * x[6].im;
        let mut re_op = 0.0f64;
        let mut im_op = 0.0f64;
        for j in (0..6).step_by(2) {
            re_op += g[j + 1] * (x[j + 1].re + x[12 - j - 1].re);
            im_op += g[j + 1] * (x[j + 1].im + x[12 - j - 1].im);
        }
        out[a][i] = Complex::new(re_in + re_op, im_in + im_op);
        out[b][i] = Complex::new(re_in - re_op, im_in - im_op);
    }
}

/// `hybrid_synthesis`: sum the sub-subbands back into QMF bands.
fn hybrid_synthesis(out: &mut [[Complex; 64]; PS_X_SLOTS], input: &[[Complex; 32]], is34: bool) {
    let len = NUM_QMF_SLOTS;
    if is34 {
        for n in 0..len {
            let o = &mut out[n];
            for v in &mut o[..5] {
                *v = Complex::default();
            }
            for row in &input[0..12] {
                o[0] += row[n];
            }
            for row in &input[12..20] {
                o[1] += row[n];
            }
            for i in 0..4 {
                o[2] += input[20 + i][n];
                o[3] += input[24 + i][n];
                o[4] += input[28 + i][n];
            }
            for i in 5..64 {
                o[i] = input[27 + i][n];
            }
        }
    } else {
        for n in 0..len {
            let o = &mut out[n];
            o[0] = input[0][n] + input[1][n] + input[2][n] + input[3][n] + input[4][n] + input[5][n];
            o[1] = input[6][n] + input[7][n];
            o[2] = input[8][n] + input[9][n];
            for i in 3..64 {
                o[i] = input[7 + i][n];
            }
        }
    }
}

/// `ps_decorrelate_c`: the fractional-delay all-pass chain of one
/// hybrid band. `delay` starts two slots before the frame (the `z^-2`).
fn decorrelate(
    out: &mut [Complex; 32],
    delay: &[Complex],
    ap_delay: &mut [[Complex; PS_QMF_TIME_SLOTS + PS_MAX_AP_DELAY]; PS_AP_LINKS],
    phi_fract: &[f64; 2],
    q_fract: &[[f64; 2]; PS_AP_LINKS],
    transient_gain: &[f64; PS_QMF_TIME_SLOTS],
    g_decay_slope: f64,
) {
    let mut ag = [0.0f64; PS_AP_LINKS];
    for (m, a) in ag.iter_mut().enumerate() {
        *a = f64::from(AP_A[m]) * g_decay_slope;
    }
    for n in 0..NUM_QMF_SLOTS {
        let mut in_re = delay[n].re * phi_fract[0] - delay[n].im * phi_fract[1];
        let mut in_im = delay[n].re * phi_fract[1] + delay[n].im * phi_fract[0];
        for m in 0..PS_AP_LINKS {
            let a_re = ag[m] * in_re;
            let a_im = ag[m] * in_im;
            let link = ap_delay[m][n + 2 - m];
            let fd = q_fract[m];
            let apd_re = in_re;
            let apd_im = in_im;
            in_re = link.re * fd[0] - link.im * fd[1];
            in_re -= a_re;
            in_im = link.re * fd[1] + link.im * fd[0];
            in_im -= a_im;
            ap_delay[m][n + 5] = Complex::new(apd_re + ag[m] * in_re, apd_im + ag[m] * in_im);
        }
        out[n] = Complex::new(transient_gain[n] * in_re, transient_gain[n] * in_im);
    }
}

/// `ps_stereo_interpolate_c`.
fn stereo_interpolate(l: &mut [Complex], r: &mut [Complex], h: [f64; 4], h_step: [f64; 4]) {
    let [mut h0, mut h1, mut h2, mut h3] = h;
    for (lv, rv) in l.iter_mut().zip(r.iter_mut()) {
        let (l_re, l_im, r_re, r_im) = (lv.re, lv.im, rv.re, rv.im);
        h0 += h_step[0];
        h1 += h_step[1];
        h2 += h_step[2];
        h3 += h_step[3];
        *lv = Complex::new(h0 * l_re + h2 * r_re, h0 * l_im + h2 * r_im);
        *rv = Complex::new(h1 * l_re + h3 * r_re, h1 * l_im + h3 * r_im);
    }
}

/// `ps_stereo_interpolate_ipdopd_c`.
fn stereo_interpolate_ipdopd(
    l: &mut [Complex],
    r: &mut [Complex],
    h: [[f64; 4]; 2],
    h_step: [[f64; 4]; 2],
) {
    let [mut h00, mut h01, mut h02, mut h03] = h[0];
    let [mut h10, mut h11, mut h12, mut h13] = h[1];
    for (lv, rv) in l.iter_mut().zip(r.iter_mut()) {
        let (l_re, l_im, r_re, r_im) = (lv.re, lv.im, rv.re, rv.im);
        h00 += h_step[0][0];
        h01 += h_step[0][1];
        h02 += h_step[0][2];
        h03 += h_step[0][3];
        h10 += h_step[1][0];
        h11 += h_step[1][1];
        h12 += h_step[1][2];
        h13 += h_step[1][3];
        *lv = Complex::new(
            h00 * l_re + h02 * r_re - h10 * l_im - h12 * r_im,
            h00 * l_im + h02 * r_im + h10 * l_re + h12 * r_re,
        );
        *rv = Complex::new(
            h01 * l_re + h03 * r_re - h11 * l_im - h13 * r_im,
            h01 * l_im + h03 * r_im + h11 * l_re + h13 * r_re,
        );
    }
}

/// Table 8.46: 10 → 20 stereo bands.
fn map_idx_10_to_20(par_mapped: &mut [i8; PS_MAX_NR_IIDICC], par: &[i8; PS_MAX_NR_IIDICC], full: bool) {
    let top = if full {
        9
    } else {
        par_mapped[10] = 0;
        4
    };
    for b in (0..=top).rev() {
        par_mapped[2 * b] = par[b];
        par_mapped[2 * b + 1] = par[b];
    }
}

fn map_idx_34_to_20(par_mapped: &mut [i8; PS_MAX_NR_IIDICC], par: &[i8; PS_MAX_NR_IIDICC], full: bool) {
    let p = |i: usize| i32::from(par[i]);
    par_mapped[0] = ((2 * p(0) + p(1)) / 3) as i8;
    par_mapped[1] = ((p(1) + 2 * p(2)) / 3) as i8;
    par_mapped[2] = ((2 * p(3) + p(4)) / 3) as i8;
    par_mapped[3] = ((p(4) + 2 * p(5)) / 3) as i8;
    par_mapped[4] = ((p(6) + p(7)) / 2) as i8;
    par_mapped[5] = ((p(8) + p(9)) / 2) as i8;
    par_mapped[6] = par[10];
    par_mapped[7] = par[11];
    par_mapped[8] = ((p(12) + p(13)) / 2) as i8;
    par_mapped[9] = ((p(14) + p(15)) / 2) as i8;
    par_mapped[10] = par[16];
    if full {
        par_mapped[11] = par[17];
        par_mapped[12] = par[18];
        par_mapped[13] = par[19];
        par_mapped[14] = ((p(20) + p(21)) / 2) as i8;
        par_mapped[15] = ((p(22) + p(23)) / 2) as i8;
        par_mapped[16] = ((p(24) + p(25)) / 2) as i8;
        par_mapped[17] = ((p(26) + p(27)) / 2) as i8;
        par_mapped[18] = ((p(28) + p(29) + p(30) + p(31)) / 4) as i8;
        par_mapped[19] = ((p(32) + p(33)) / 2) as i8;
    }
}

fn map_val_34_to_20(par: &mut [f64; PS_MAX_NR_IIDICC]) {
    let third = f64::from(0.333_333_33f32);
    par[0] = (2.0 * par[0] + par[1]) * third;
    par[1] = (par[1] + 2.0 * par[2]) * third;
    par[2] = (2.0 * par[3] + par[4]) * third;
    par[3] = (par[4] + 2.0 * par[5]) * third;
    par[4] = (par[6] + par[7]) * 0.5;
    par[5] = (par[8] + par[9]) * 0.5;
    par[6] = par[10];
    par[7] = par[11];
    par[8] = (par[12] + par[13]) * 0.5;
    par[9] = (par[14] + par[15]) * 0.5;
    par[10] = par[16];
    par[11] = par[17];
    par[12] = par[18];
    par[13] = par[19];
    par[14] = (par[20] + par[21]) * 0.5;
    par[15] = (par[22] + par[23]) * 0.5;
    par[16] = (par[24] + par[25]) * 0.5;
    par[17] = (par[26] + par[27]) * 0.5;
    par[18] = (par[28] + par[29] + par[30] + par[31]) * 0.25;
    par[19] = (par[32] + par[33]) * 0.5;
}

fn map_idx_10_to_34(par_mapped: &mut [i8; PS_MAX_NR_IIDICC], par: &[i8; PS_MAX_NR_IIDICC], full: bool) {
    if full {
        par_mapped[28..34].fill(par[9]);
        par_mapped[24..28].fill(par[8]);
        par_mapped[20..24].fill(par[7]);
        par_mapped[18..20].fill(par[6]);
        par_mapped[16..18].fill(par[5]);
    } else {
        par_mapped[16] = 0;
    }
    par_mapped[12..16].fill(par[4]);
    par_mapped[10..12].fill(par[3]);
    par_mapped[6..10].fill(par[2]);
    par_mapped[3..6].fill(par[1]);
    par_mapped[0..3].fill(par[0]);
}

fn map_idx_20_to_34(par_mapped: &mut [i8; PS_MAX_NR_IIDICC], par: &[i8; PS_MAX_NR_IIDICC], full: bool) {
    if full {
        par_mapped[33] = par[19];
        par_mapped[32] = par[19];
        par_mapped[31] = par[18];
        par_mapped[30] = par[18];
        par_mapped[29] = par[18];
        par_mapped[28] = par[18];
        par_mapped[27] = par[17];
        par_mapped[26] = par[17];
        par_mapped[25] = par[16];
        par_mapped[24] = par[16];
        par_mapped[23] = par[15];
        par_mapped[22] = par[15];
        par_mapped[21] = par[14];
        par_mapped[20] = par[14];
        par_mapped[19] = par[13];
        par_mapped[18] = par[12];
        par_mapped[17] = par[11];
    }
    par_mapped[16] = par[10];
    par_mapped[15] = par[9];
    par_mapped[14] = par[9];
    par_mapped[13] = par[8];
    par_mapped[12] = par[8];
    par_mapped[11] = par[7];
    par_mapped[10] = par[6];
    par_mapped[9] = par[5];
    par_mapped[8] = par[5];
    par_mapped[7] = par[4];
    par_mapped[6] = par[4];
    par_mapped[5] = par[3];
    par_mapped[4] = ((i32::from(par[2]) + i32::from(par[3])) / 2) as i8;
    par_mapped[3] = par[2];
    par_mapped[2] = par[1];
    par_mapped[1] = ((i32::from(par[0]) + i32::from(par[1])) / 2) as i8;
    par_mapped[0] = par[0];
}

fn map_val_20_to_34(par: &mut [f64; PS_MAX_NR_IIDICC]) {
    par[33] = par[19];
    par[32] = par[19];
    par[31] = par[18];
    par[30] = par[18];
    par[29] = par[18];
    par[28] = par[18];
    par[27] = par[17];
    par[26] = par[17];
    par[25] = par[16];
    par[24] = par[16];
    par[23] = par[15];
    par[22] = par[15];
    par[21] = par[14];
    par[20] = par[14];
    par[19] = par[13];
    par[18] = par[12];
    par[17] = par[11];
    par[16] = par[10];
    par[15] = par[9];
    par[14] = par[9];
    par[13] = par[8];
    par[12] = par[8];
    par[11] = par[7];
    par[10] = par[6];
    par[9] = par[5];
    par[8] = par[5];
    par[7] = par[4];
    par[6] = par[4];
    par[5] = par[3];
    par[4] = (par[2] + par[3]) * 0.5;
    par[3] = par[2];
    par[2] = par[1];
    par[1] = (par[0] + par[1]) * 0.5;
}

type ParRows = [[i8; PS_MAX_NR_IIDICC]; PS_MAX_NUM_ENV];

fn remap34(mapped: &mut ParRows, par: &ParRows, num_par: usize, num_env: usize, full: bool) {
    for e in 0..num_env {
        match num_par {
            20 | 11 => map_idx_20_to_34(&mut mapped[e], &par[e], full),
            10 | 5 => map_idx_10_to_34(&mut mapped[e], &par[e], full),
            _ => mapped[e] = par[e],
        }
    }
}

fn remap20(mapped: &mut ParRows, par: &ParRows, num_par: usize, num_env: usize, full: bool) {
    for e in 0..num_env {
        match num_par {
            34 | 17 => map_idx_34_to_20(&mut mapped[e], &par[e], full),
            10 | 5 => map_idx_10_to_20(&mut mapped[e], &par[e], full),
            _ => mapped[e] = par[e],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    /// A header'd one-envelope `ps_data()` payload with coarse IID
    /// index `iid_idx` in band 0 (0 elsewhere) and ICC index 0,
    /// prefixed as the SBR extension body (`8·len − 2` bits).
    fn payload(iid_idx: i32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bit(true); // enable_ps_header
        w.write_bit(true); // enable_iid
        w.write_u32(0, 3); // iid_mode 0
        w.write_bit(true); // enable_icc
        w.write_u32(0, 3); // icc_mode 0
        w.write_bit(false); // enable_ext
        w.write_bit(false); // FIX
        w.write_u32(1, 2); // num_env = 1
        w.write_bit(false); // iid_dt = freq
        let (len, code) = HUFF_IID_DF[(iid_idx + 14) as usize];
        w.write_u32(code, u32::from(len));
        let (l0, c0) = HUFF_IID_DF[14];
        for _ in 1..10 {
            w.write_u32(c0, u32::from(l0));
        }
        w.write_bit(false); // icc_dt = freq
        let (li, ci) = HUFF_ICC_DF[7];
        for _ in 0..10 {
            w.write_u32(ci, u32::from(li));
        }
        // Pad so the payload fits the `8·len − 2` bit budget.
        w.write_u32(0, 8);
        w.finish()
    }

    fn x_ones() -> Box<[[Complex; 64]; PS_X_SLOTS]> {
        Box::new([[Complex::new(1.0, 0.0); 64]; PS_X_SLOTS])
    }

    #[test]
    fn starts_only_after_a_header() {
        let mut dec = PsDecoder::new();
        assert!(!dec.started());
        let mut w = BitWriter::new();
        w.write_bit(false); // enable_ps_header = 0
        w.write_bit(false); // frame_class
        w.write_u32(0, 2); // num_env = 0
        w.write_u32(0, 8);
        dec.read_extension(&w.finish());
        assert!(!dec.started());
        dec.read_extension(&payload(7));
        assert!(dec.started());
    }

    #[test]
    fn truncated_payload_resets() {
        let mut dec = PsDecoder::new();
        dec.read_extension(&payload(7));
        assert!(dec.started());
        let p = payload(7);
        dec.read_extension(&p[..2]);
        assert!(!dec.started());
    }

    /// A strong positive IID in band 0 tilts that band's energy left.
    #[test]
    fn iid_tilts_energy_left() {
        let mut dec = PsDecoder::new();
        let p = payload(7);
        let (mut l_e, mut r_e) = (0.0f64, 0.0f64);
        for f in 0..4 {
            dec.read_extension(&p);
            let mut l = x_ones();
            let mut r = x_ones();
            dec.apply(&mut l, &mut r, 32).unwrap();
            if f >= 2 {
                for n in 0..NUM_QMF_SLOTS {
                    l_e += l[n][0].norm_sqr();
                    r_e += r[n][0].norm_sqr();
                }
            }
        }
        assert!(l_e > 50.0 * r_e, "left {l_e} not dominant over right {r_e}");
    }

    /// IID 0 / ICC 0 (full coherence) gives identical channels.
    #[test]
    fn neutral_cues_give_dual_mono() {
        let mut dec = PsDecoder::new();
        let p = payload(0);
        let mut last = None;
        for _ in 0..3 {
            dec.read_extension(&p);
            let mut l = x_ones();
            let mut r = x_ones();
            dec.apply(&mut l, &mut r, 32).unwrap();
            last = Some((l, r));
        }
        let (l, r) = last.unwrap();
        for n in 0..NUM_QMF_SLOTS {
            for k in 0..64 {
                let d = l[n][k] - r[n][k];
                assert!(d.norm_sqr() < 1e-12, "slot {n} band {k}");
            }
        }
    }
}
