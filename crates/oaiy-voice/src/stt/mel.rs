//! The log-mel front end, as NeMo's `FilterbankFeatures` computes it at
//! inference (`AudioToMelSpectrogramPreprocessor` in the Parakeet configs):
//!
//! - pre-emphasis `y[i] = x[i] - 0.97 x[i-1]`, the first sample kept;
//! - a centred STFT (`torch.stft(center=True, pad_mode="constant")`): zero
//!   padding of `n_fft / 2` on both sides, a **symmetric** Hann window of
//!   `win_length` samples (`periodic=False`) placed in the middle of the
//!   `n_fft` frame;
//! - the power spectrum, Slaney mel filters from 0 Hz to Nyquist
//!   (librosa's `filters.mel(norm="slaney", htk=False)`), `ln(x + 2^-24)`;
//! - per-bin normalisation over the valid frames: the mean, and the
//!   **unbiased** standard deviation plus 1e-5.
//!
//! NeMo counts `samples / hop` valid frames; the STFT makes one more, which it
//! masks to zero and leaves out of the statistics. The encoder masks it again
//! before every convolution, so the frame is dropped here: the encoder's
//! output on the valid frames is the same either way.
//!
//! Dither is a training-time augmentation (NeMo adds it only when
//! `self.training`), so there is none here.
//!
//! This runs on the CPU in f64: it is about a millisecond for ten seconds of
//! audio, and the FFT in f64 keeps it within float rounding of NeMo's.

/// The front end's settings (the Parakeet configs' `preprocessor` block).
#[derive(Clone, Debug, PartialEq)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    /// `None` turns pre-emphasis off.
    pub preemph: Option<f32>,
    /// `log_zero_guard_value` (added before the log).
    pub log_guard: f64,
}

impl MelConfig {
    /// Parakeet's: 16 kHz, 25 ms windows every 10 ms, 512-point FFT, 128 mels.
    pub fn parakeet(n_mels: usize) -> Self {
        Self { sample_rate: 16_000, n_fft: 512, win_length: 400, hop_length: 160, n_mels, preemph: Some(0.97), log_guard: 2f64.powi(-24) }
    }
}

/// NeMo's `CONSTANT`, added to each bin's standard deviation.
const STD_GUARD: f64 = 1e-5;

pub struct MelFrontEnd {
    cfg: MelConfig,
    /// The Hann window zero-padded to `n_fft`, centred as `torch.stft` does.
    window: Vec<f64>,
    /// `(n_mels, n_fft / 2 + 1)`, row-major; rounded to f32 like NeMo's buffer.
    filters: Vec<f64>,
    /// Twiddle factors for the radix-2 FFT: `exp(-2 pi i k / n_fft)`.
    twiddle: Vec<(f64, f64)>,
}

/// Mel features for one utterance: `n_mels` rows of `frames` values.
#[derive(Clone, Debug)]
pub struct Features {
    pub n_mels: usize,
    pub frames: usize,
    /// `(n_mels, frames)`, row-major.
    pub data: Vec<f32>,
}

impl MelFrontEnd {
    pub fn new(cfg: MelConfig) -> Self {
        assert!(cfg.n_fft.is_power_of_two() && cfg.win_length <= cfg.n_fft && cfg.hop_length > 0);
        let n = cfg.win_length;
        let mut window = vec![0f64; cfg.n_fft];
        let left = (cfg.n_fft - n) / 2;
        for i in 0..n {
            // torch.hann_window(n, periodic=False), computed in f32 by torch.
            let w = if n == 1 { 1.0 } else { 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos() };
            window[left + i] = w as f32 as f64;
        }
        let filters = slaney_filters(cfg.sample_rate as f64, cfg.n_fft, cfg.n_mels, 0.0, cfg.sample_rate as f64 / 2.0).into_iter().map(|w| w as f64).collect();
        let twiddle = (0..cfg.n_fft / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / cfg.n_fft as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Self { cfg, window, filters, twiddle }
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    /// The number of valid frames for `samples` samples.
    pub fn frames_for(&self, samples: usize) -> usize {
        samples / self.cfg.hop_length
    }

    /// Normalised log-mel features of 16 kHz mono audio in [-1, 1].
    pub fn features(&self, samples: &[f32]) -> Features {
        let cfg = &self.cfg;
        let frames = self.frames_for(samples.len());
        let n_mels = cfg.n_mels;
        if frames == 0 {
            return Features { n_mels, frames: 0, data: Vec::new() };
        }
        // Pre-emphasis in f32, as torch does it.
        let emphasised: Vec<f32> = match cfg.preemph {
            Some(a) => (0..samples.len()).map(|i| if i == 0 { samples[0] } else { samples[i] - a * samples[i - 1] }).collect(),
            None => samples.to_vec(),
        };
        let pad = cfg.n_fft / 2;
        let bins = cfg.n_fft / 2 + 1;
        let mut logmel = vec![0f64; n_mels * frames];
        let (mut re, mut im) = (vec![0f64; cfg.n_fft], vec![0f64; cfg.n_fft]);
        let mut power = vec![0f64; bins];
        for f in 0..frames {
            // Frame f covers padded[f * hop .. f * hop + n_fft], with the
            // signal starting at padded index `pad`.
            let start = (f * cfg.hop_length) as isize - pad as isize;
            for i in 0..cfg.n_fft {
                let s = start + i as isize;
                let x = if s >= 0 && (s as usize) < emphasised.len() { emphasised[s as usize] as f64 } else { 0.0 };
                re[i] = x * self.window[i];
                im[i] = 0.0;
            }
            fft(&mut re, &mut im, &self.twiddle);
            for (b, p) in power.iter_mut().enumerate() {
                *p = re[b] * re[b] + im[b] * im[b];
            }
            for m in 0..n_mels {
                let row = &self.filters[m * bins..(m + 1) * bins];
                let e: f64 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
                logmel[m * frames + f] = (e + cfg.log_guard).ln();
            }
        }
        let mut data = vec![0f32; n_mels * frames];
        for m in 0..n_mels {
            let row = &logmel[m * frames..(m + 1) * frames];
            let mean = row.iter().sum::<f64>() / frames as f64;
            let var = if frames > 1 { row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (frames - 1) as f64 } else { 0.0 };
            let std = var.sqrt() + STD_GUARD;
            for (o, x) in data[m * frames..(m + 1) * frames].iter_mut().zip(row) {
                *o = ((x - mean) / std) as f32;
            }
        }
        Features { n_mels, frames, data }
    }
}

/// In-place radix-2 FFT (`re.len()` a power of two, `twiddle` its first half
/// of roots of unity).
fn fft(re: &mut [f64], im: &mut [f64], twiddle: &[(f64, f64)]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (w_re, w_im) = twiddle[k * step];
                let (a, b) = (start + k, start + k + len / 2);
                let (t_re, t_im) = (re[b] * w_re - im[b] * w_im, re[b] * w_im + im[b] * w_re);
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
            }
        }
        len <<= 1;
    }
}

/// librosa's Slaney mel filterbank (`htk=False`, `norm="slaney"`) as a
/// `(n_mels, n_fft / 2 + 1)` matrix, computed in f64 and rounded to f32 as
/// librosa returns it.
pub fn slaney_filters(sr: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |f: f64| if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp };
    let mel_to_hz = |m: f64| if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m };
    let bins = n_fft / 2 + 1;
    let freqs: Vec<f64> = (0..bins).map(|i| i as f64 * sr / n_fft as f64).collect();
    let (lo, hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let step = (hi - lo) / (n_mels + 1) as f64;
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + step * i as f64)).collect();
    let mut w = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (b, f) in freqs.iter().enumerate() {
            let lower = (f - mel_f[m]) / (mel_f[m + 1] - mel_f[m]);
            let upper = (mel_f[m + 2] - f) / (mel_f[m + 2] - mel_f[m + 1]);
            w[m * bins + b] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_is_samples_over_hop() {
        let mel = MelFrontEnd::new(MelConfig::parakeet(128));
        assert_eq!(mel.features(&[]).frames, 0);
        assert_eq!(mel.features(&[0.1; 159]).frames, 0);
        assert_eq!(mel.features(&vec![0.1; 16_000]).frames, 100);
        assert_eq!(mel.features(&vec![0.1; 16_159]).frames, 100);
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        let n = 16;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 % 5) as f64 - 2.0) * 0.3 + (i as f64 * 0.4).sin()).collect();
        let twiddle: Vec<(f64, f64)> = (0..n / 2).map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (a.cos(), a.sin())
        }).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; n]);
        fft(&mut re, &mut im, &twiddle);
        for k in 0..n {
            let (mut dr, mut di) = (0.0, 0.0);
            for (t, v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                dr += v * a.cos();
                di += v * a.sin();
            }
            assert!((re[k] - dr).abs() < 1e-9 && (im[k] - di).abs() < 1e-9, "bin {k}");
        }
    }

    #[test]
    fn slaney_filters_are_triangles_with_unit_area_per_mel() {
        // Each Slaney filter is scaled to 2 / (bandwidth in Hz): its peak times
        // half its base is one, whatever the band.
        let (sr, n_fft, n_mels) = (16_000.0, 512, 128);
        let w = slaney_filters(sr, n_fft, n_mels, 0.0, 8_000.0);
        let bins = n_fft / 2 + 1;
        assert_eq!(w.len(), n_mels * bins);
        assert!(w.iter().all(|&v| v >= 0.0));
        // The first filter starts at 0 Hz; every filter has some weight.
        for m in 0..n_mels {
            assert!(w[m * bins..(m + 1) * bins].iter().any(|&v| v > 0.0), "mel {m} is empty");
        }
        // The first filter by hand: the Slaney scale is linear (200/3 Hz a
        // mel) below 1 kHz, so its centre is one 129th of 8 kHz's mel value
        // and it spans 0 Hz to twice that; bin 1 is 31.25 Hz.
        let mel_hi = 15.0 + (8000f64 / 1000.0).ln() / (6.4f64.ln() / 27.0);
        let f1 = mel_hi / 129.0 * 200.0 / 3.0;
        let enorm = 2.0 / (2.0 * f1);
        let expect = (31.25 / f1).min((2.0 * f1 - 31.25) / f1) * enorm;
        assert!((w[1] as f64 - expect).abs() < 1e-6, "{} vs {expect}", w[1]);
    }

    #[test]
    fn features_are_normalised_per_bin() {
        let samples: Vec<f32> = (0..32_000).map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.041).cos()) * 0.3).collect();
        let mel = MelFrontEnd::new(MelConfig::parakeet(128));
        let f = mel.features(&samples);
        assert_eq!((f.n_mels, f.frames), (128, 200));
        for m in 0..f.n_mels {
            let row = &f.data[m * f.frames..(m + 1) * f.frames];
            let mean = row.iter().map(|&x| x as f64).sum::<f64>() / f.frames as f64;
            let var = row.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (f.frames - 1) as f64;
            assert!(mean.abs() < 1e-5, "bin {m} mean {mean}");
            // Unit unbiased std, less the 1e-5 guard's share.
            assert!((var.sqrt() - 1.0).abs() < 1e-3 || var < 1e-6, "bin {m} std {}", var.sqrt());
        }
    }
}
