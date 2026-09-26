//! LTX 2.5 audio: the audio VAE decoder (latent to a stereo log-mel spectrogram)
//! and the vocoder with bandwidth extension (mel to a 48 kHz stereo waveform),
//! after the official ltx-core implementation.
//!
//! Everything runs in F32. The reference does the same for the vocoder, whose
//! chained convolutions lose 40-90% of their spectral accuracy in BF16.
use super::store::Store;
use candle_core::{DType, Device, Result, Tensor};
use nrob::json::Json;
use std::collections::HashMap;
use std::path::Path;

/// Audio latent frames per second: 16 kHz mel at hop 160, downsampled 4x in time.
pub const LATENT_RATE: f64 = 25.;

/// One BigVGAN-style generator (the vocoder, or the bandwidth extension's).
struct Generator {
    prefix: &'static str,
    upsample_rates: Vec<usize>,
    upsample_kernels: Vec<usize>,
    resblock_kernels: Vec<usize>,
    dilations: Vec<Vec<usize>>,
    /// Clamp to [-1, 1] at the end (the vocoder does; the BWE residual does not).
    clamp: bool,
}

pub struct AudioDecoder {
    w: HashMap<String, Tensor>,
    levels: usize,
    vocoder: Generator,
    bwe: Generator,
    hop: usize,
    n_fft: usize,
    /// Output over input sample rate of the bandwidth extension (48k / 16k).
    ratio: usize,
    pub sample_rate: usize,
}

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

fn ints(v: Option<&Json>, what: &str) -> Result<Vec<usize>> {
    v.and_then(Json::as_array)
        .ok_or_else(|| msg(format!("audio VAE config lacks {what}")))?
        .iter()
        .map(|x| x.as_i64().filter(|n| *n > 0).map(|n| n as usize).ok_or_else(|| msg(format!("bad {what}"))))
        .collect()
}

impl Generator {
    fn from_config(prefix: &'static str, c: &Json, clamp: bool) -> Result<Self> {
        if c.get("resblock").and_then(Json::as_str) != Some("AMP1") || c.get("activation").and_then(Json::as_str) != Some("snakebeta") {
            candle_core::bail!("audio vocoder: only AMP1 blocks with snakebeta activations are supported");
        }
        let dilations = c
            .get("resblock_dilation_sizes")
            .and_then(Json::as_array)
            .ok_or_else(|| msg("audio VAE config lacks resblock_dilation_sizes"))?
            .iter()
            .map(|d| ints(Some(d), "resblock_dilation_sizes"))
            .collect::<Result<Vec<_>>>()?;
        let g = Self {
            prefix,
            upsample_rates: ints(c.get("upsample_rates"), "upsample_rates")?,
            upsample_kernels: ints(c.get("upsample_kernel_sizes"), "upsample_kernel_sizes")?,
            resblock_kernels: ints(c.get("resblock_kernel_sizes"), "resblock_kernel_sizes")?,
            dilations,
            clamp,
        };
        if g.upsample_rates.len() != g.upsample_kernels.len() || g.resblock_kernels.len() != g.dilations.len() {
            candle_core::bail!("audio vocoder: inconsistent stage configuration");
        }
        Ok(g)
    }
}

impl AudioDecoder {
    /// Load the decoder half of an audio VAE file with its vocoder (the LTX 2.5
    /// layout: `audio_vae.*`, `vocoder.vocoder.*`, `vocoder.bwe_generator.*`,
    /// `vocoder.mel_stft.*`).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let mut store = Store::open(path, 0)?;
        let config = Json::parse(store.index.metadata("config").ok_or_else(|| msg("audio VAE file lacks its config metadata"))?.as_bytes())
            .map_err(candle_core::Error::wrap)?;
        let dd = config
            .get("audio_vae")
            .and_then(|a| a.get("model"))
            .and_then(|m| m.get("params"))
            .and_then(|p| p.get("ddconfig"))
            .ok_or_else(|| msg("audio VAE config lacks audio_vae.model.params.ddconfig"))?;
        for (key, want) in [("z_channels", 8), ("out_ch", 2), ("mel_bins", 64)] {
            if dd.get(key).and_then(Json::as_i64) != Some(want) {
                candle_core::bail!("unsupported audio VAE: {key}");
            }
        }
        if dd.get("norm_type").and_then(Json::as_str) != Some("pixel") || dd.get("causality_axis").and_then(Json::as_str) != Some("height") {
            candle_core::bail!("unsupported audio VAE: needs pixel norm, causal in time");
        }
        let levels = ints(dd.get("ch_mult"), "ch_mult")?.len();
        let voc = config.get("vocoder").ok_or_else(|| msg("audio VAE file lacks a vocoder config"))?;
        let (vc, bc) = match (voc.get("vocoder"), voc.get("bwe")) {
            (Some(v), Some(b)) => (v, b),
            _ => candle_core::bail!("audio VAE file lacks the bandwidth-extension vocoder (LTX 2.5 layout)"),
        };
        let rate = |k: &str| bc.get(k).and_then(Json::as_i64).filter(|n| *n > 0).map(|n| n as usize).ok_or_else(|| msg(format!("bwe config lacks {k}")));
        let (input_rate, sample_rate) = (rate("input_sampling_rate")?, rate("output_sampling_rate")?);
        if sample_rate % input_rate != 0 {
            candle_core::bail!("bandwidth extension must upsample by a whole factor");
        }
        let names: Vec<String> = store
            .index
            .names()
            .filter(|k| {
                k.starts_with("audio_vae.decoder.")
                    || k.starts_with("audio_vae.per_channel_statistics.")
                    || (k.starts_with("vocoder.") && !k.ends_with("inverse_basis"))
            })
            .map(str::to_owned)
            .collect();
        let mut w = HashMap::new();
        for k in names {
            let t = store.tensor_f32(&k, dev)?;
            w.insert(k, t);
        }
        Ok(Self {
            w,
            levels,
            vocoder: Generator::from_config("vocoder.vocoder", vc, true)?,
            bwe: Generator::from_config("vocoder.bwe_generator", bc, false)?,
            hop: rate("hop_length")?,
            n_fft: rate("n_fft")?,
            ratio: sample_rate / input_rate,
            sample_rate,
        })
    }

    fn get(&self, k: &str) -> Result<&Tensor> {
        self.w.get(k).ok_or_else(|| msg(format!("missing audio tensor {k}")))
    }

    /// A 2-D convolution causal in time (height): the past is padded, the
    /// frequency axis symmetrically, both with zeros.
    fn conv2d(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        causal_conv2d(x, self.get(&format!("{prefix}.conv.weight"))?, self.get(&format!("{prefix}.conv.bias"))?)
    }

    fn resnet(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = self.conv2d(&pixel_norm(x)?.silu()?, &format!("{prefix}.conv1"))?;
        let h = self.conv2d(&pixel_norm(&h)?.silu()?, &format!("{prefix}.conv2"))?;
        let shortcut = format!("{prefix}.nin_shortcut");
        let x = if self.w.contains_key(&format!("{shortcut}.conv.weight")) { self.conv2d(x, &shortcut)? } else { x.clone() };
        x + h
    }

    /// Latent `(1, frames, 128)` (the transformer's patchified layout, still
    /// normalized) to a log-mel spectrogram `(1, 2, 4*frames - 3, 64)`.
    pub fn mel(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, frames, width) = latent.dims3()?;
        if width != 128 || frames == 0 {
            candle_core::bail!("audio latent must be (batch, frames, 128)");
        }
        let std = self.get("audio_vae.per_channel_statistics.std-of-means")?;
        let mean = self.get("audio_vae.per_channel_statistics.mean-of-means")?;
        let x = latent.to_dtype(DType::F32)?.broadcast_mul(std)?.broadcast_add(mean)?;
        // Patch index is channel * 16 + frequency.
        let x = x.reshape((b, frames, 8, 16))?.permute((0, 2, 1, 3))?.contiguous()?;
        let p = "audio_vae.decoder";
        let mut h = self.conv2d(&x, &format!("{p}.conv_in"))?;
        h = self.resnet(&h, &format!("{p}.mid.block_1"))?;
        h = self.resnet(&h, &format!("{p}.mid.block_2"))?;
        for level in (0..self.levels).rev() {
            let mut i = 0;
            while self.w.contains_key(&format!("{p}.up.{level}.block.{i}.conv1.conv.weight")) {
                h = self.resnet(&h, &format!("{p}.up.{level}.block.{i}"))?;
                i += 1;
            }
            if level != 0 {
                // Nearest 2x, causal conv, then drop the first time row: 2n - 1.
                let (_, _, t, f) = h.dims4()?;
                h = self.conv2d(&h.upsample_nearest2d(t * 2, f * 2)?, &format!("{p}.up.{level}.upsample.conv"))?;
                h = h.narrow(2, 1, 2 * t - 1)?;
            }
        }
        h = self.conv2d(&pixel_norm(&h)?.silu()?, &format!("{p}.conv_out"))?;
        // Exactly 4 * frames - 3 time rows and 64 mel bins, cropped or zero-padded.
        let (target_t, target_f) = ((frames * 4).saturating_sub(3).max(1), 64);
        let (_, _, t, f) = h.dims4()?;
        let h = h.narrow(2, 0, t.min(target_t))?.narrow(3, 0, f.min(target_f))?;
        h.pad_with_zeros(2, 0, target_t - t.min(target_t))?.pad_with_zeros(3, 0, target_f - f.min(target_f))
    }

    fn conv1d(&self, x: &Tensor, prefix: &str, padding: usize, dilation: usize) -> Result<Tensor> {
        let y = x.conv1d(self.get(&format!("{prefix}.weight"))?, padding, 1, dilation, 1)?;
        match self.w.get(&format!("{prefix}.bias")) {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }

    /// Anti-aliased SnakeBeta: upsample 2x, activate, low-pass and decimate 2x,
    /// with the stored 12-tap filters shared by every channel.
    fn activation(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let up = self.get(&format!("{prefix}.upsample.filter"))?;
        let down = self.get(&format!("{prefix}.downsample.lowpass.filter"))?;
        let (ku, kd) = (up.dim(2)?, down.dim(2)?);
        let pad = ku / 2 - 1;
        let (crop_left, crop_right) = (pad * 2 + (ku - 2) / 2, pad * 2 + (ku - 1) / 2);
        // Channels fold into the batch: one single-channel transposed conv
        // instead of `c` grouped ones.
        let y = (x.reshape((b * c, 1, t))?.pad_with_same(2, pad, pad)?.conv_transpose1d(up, 0, 0, 2, 1, 1)? * 2.)?;
        let len = y.dim(2)?;
        let y = y.narrow(2, crop_left, len - crop_left - crop_right)?.reshape((b, c, 2 * t))?;
        let alpha = self.get(&format!("{prefix}.act.alpha"))?.exp()?.reshape((1, c, 1))?;
        let beta = ((self.get(&format!("{prefix}.act.beta"))?.exp()? + 1e-9)?.recip()?).reshape((1, c, 1))?;
        let y = (&y + y.broadcast_mul(&alpha)?.sin()?.sqr()?.broadcast_mul(&beta)?)?;
        let (left, right) = (kd / 2 - usize::from(kd % 2 == 0), kd / 2);
        y.reshape((b * c, 1, 2 * t))?.pad_with_same(2, left, right)?.conv1d(down, 0, 2, 1, 1)?.reshape((b, c, t))
    }

    fn amp_block(&self, x: &Tensor, prefix: &str, kernel: usize, dilations: &[usize]) -> Result<Tensor> {
        let mut x = x.clone();
        for (j, &d) in dilations.iter().enumerate() {
            let h = self.activation(&x, &format!("{prefix}.acts1.{j}"))?;
            let h = self.conv1d(&h, &format!("{prefix}.convs1.{j}"), (kernel * d - d) / 2, d)?;
            let h = self.activation(&h, &format!("{prefix}.acts2.{j}"))?;
            let h = self.conv1d(&h, &format!("{prefix}.convs2.{j}"), (kernel - 1) / 2, 1)?;
            x = (x + h)?;
        }
        Ok(x)
    }

    /// Stereo mel `(B, 2, frames, 64)` to a waveform `(B, 2, samples)`.
    fn generate(&self, g: &Generator, mel: &Tensor) -> Result<Tensor> {
        let (b, s, frames, bins) = mel.dims4()?;
        let p = g.prefix;
        let mut x = mel.transpose(2, 3)?.contiguous()?.reshape((b, s * bins, frames))?;
        x = self.conv1d(&x, &format!("{p}.conv_pre"), 3, 1)?;
        let blocks = g.resblock_kernels.len();
        for (i, (&stride, &kernel)) in g.upsample_rates.iter().zip(&g.upsample_kernels).enumerate() {
            let w = self.get(&format!("{p}.ups.{i}.weight"))?;
            let bias = self.get(&format!("{p}.ups.{i}.bias"))?;
            x = x.conv_transpose1d(w, (kernel - stride) / 2, 0, stride, 1, 1)?.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)?;
            // The resblocks all read the same input; their mean moves on.
            let mut sum: Option<Tensor> = None;
            for (j, (&k, d)) in g.resblock_kernels.iter().zip(&g.dilations).enumerate() {
                let y = self.amp_block(&x, &format!("{p}.resblocks.{}", i * blocks + j), k, d)?;
                sum = Some(match sum {
                    None => y,
                    Some(acc) => (acc + y)?,
                });
            }
            x = (sum.ok_or_else(|| msg("vocoder stage without resblocks"))? / blocks as f64)?;
        }
        x = self.activation(&x, &format!("{p}.act_post"))?;
        x = self.conv1d(&x, &format!("{p}.conv_post"), 3, 1)?;
        if g.clamp {
            x = x.clamp(-1f32, 1f32)?;
        }
        Ok(x)
    }

    /// Causal log-mel of a waveform `(B, 2, samples)` with the checkpoint's own
    /// STFT basis and filterbank: `(B, 2, frames, 64)`.
    fn log_mel(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, len) = x.dims3()?;
        let basis = self.get("vocoder.mel_stft.stft_fn.forward_basis")?;
        let filters = self.get("vocoder.mel_stft.mel_basis")?;
        let bins = basis.dim(0)? / 2;
        let spec = x.reshape((b * s, 1, len))?.pad_with_zeros(2, self.n_fft - self.hop, 0)?.conv1d(basis, 0, self.hop, 1, 1)?;
        let (re, im) = (spec.narrow(1, 0, bins)?, spec.narrow(1, bins, bins)?);
        let magnitude = (re.sqr()? + im.sqr()?)?.sqrt()?;
        let mel = filters.broadcast_matmul(&magnitude)?.clamp(1e-5f32, f32::MAX)?.log()?;
        let (_, n, frames) = mel.dims3()?;
        mel.reshape((b, s, n, frames))?.transpose(2, 3)?.contiguous()
    }

    /// The bandwidth extension's skip path: Hann-windowed sinc upsampling by
    /// `ratio`, equivalent to torchaudio's resampler (not stored in the file).
    fn resample(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, len) = x.dims3()?;
        let (ratio, rolloff, width_zeros) = (self.ratio, 0.99f64, 6f64);
        let width = (width_zeros / rolloff).ceil() as usize;
        let kernel = 2 * width * ratio + 1;
        let taps: Vec<f32> = (0..kernel)
            .map(|i| {
                let t = (i as f64 / ratio as f64 - width as f64) * rolloff;
                let clamped = t.clamp(-width_zeros, width_zeros);
                let window = (clamped * std::f64::consts::PI / width_zeros / 2.).cos().powi(2);
                let sinc = if t == 0. { 1. } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
                (sinc * window * rolloff / ratio as f64) as f32
            })
            .collect();
        let filter = Tensor::from_vec(taps, (1, 1, kernel), x.device())?;
        let y = (x.reshape((b * s, 1, len))?.pad_with_same(2, width, width)?.conv_transpose1d(&filter, 0, 0, ratio, 1, 1)? * ratio as f64)?;
        let (left, right) = (2 * width * ratio, kernel - ratio);
        let n = y.dim(2)?;
        y.narrow(2, left, n - left - right)?.reshape((b, s, len * ratio))
    }

    /// A log-mel spectrogram `(1, 2, frames, 64)` to a stereo waveform at
    /// `sample_rate`: `(1, 2, samples)`.
    pub fn waveform(&self, mel: &Tensor) -> Result<Tensor> {
        let low = self.generate(&self.vocoder, mel)?;
        let len = low.dim(2)?;
        let low = low.pad_with_zeros(2, 0, (self.hop - len % self.hop) % self.hop)?;
        let residual = self.generate(&self.bwe, &self.log_mel(&low)?)?;
        let skip = self.resample(&low)?;
        (residual + skip)?.clamp(-1f32, 1f32)?.narrow(2, 0, len * self.ratio)
    }

    /// Latent to waveform in one call.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.waveform(&self.mel(latent)?)
    }

    #[cfg(test)]
    fn vocoder_only(&self, mel: &Tensor) -> Result<Tensor> {
        self.generate(&self.vocoder, mel)
    }
}

/// `x / sqrt(mean over channels of x^2 + 1e-6)`.
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    x.broadcast_div(&(x.sqr()?.mean_keepdim(1)? + 1e-6)?.sqrt()?)
}

/// A 2-D convolution causal in time (height): the past is padded, the
/// frequency axis symmetrically, both with zeros.
fn causal_conv2d(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (_, _, kh, kw) = w.dims4()?;
    let x = x.pad_with_zeros(2, kh - 1, 0)?.pad_with_zeros(3, (kw - 1) / 2, kw - 1 - (kw - 1) / 2)?;
    x.conv2d(w, 0, 1, 1, 1)?.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?)
}

/// torchaudio's `functional.resample` (Hann-windowed sinc, 6 zero crossings,
/// rolloff 0.99) for one channel, `from` Hz to `to` Hz.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let g = gcd(from, to);
    let (orig, new) = (from / g, to / g);
    let (zeros, rolloff) = (6f64, 0.99f64);
    let base = orig.min(new) as f64 * rolloff;
    let width = (zeros * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    // One kernel per output phase, computed in F64 and applied in F32.
    let kernels: Vec<Vec<f32>> = (0..new)
        .map(|phase| {
            (0..taps)
                .map(|i| {
                    let idx = (i as f64 - width as f64) / orig as f64;
                    let t = ((-(phase as f64) / new as f64 + idx) * base).clamp(-zeros, zeros);
                    let window = (t * std::f64::consts::PI / zeros / 2.).cos().powi(2);
                    let t = t * std::f64::consts::PI;
                    let sinc = if t == 0. { 1. } else { t.sin() / t };
                    (sinc * window * base / orig as f64) as f32
                })
                .collect()
        })
        .collect();
    // Zeros: `width` before, `width + orig` after.
    let padded: Vec<f32> = std::iter::repeat_n(0f32, width).chain(x.iter().copied()).chain(std::iter::repeat_n(0f32, width + orig)).collect();
    let blocks = (padded.len() - taps) / orig + 1;
    let target = (new as u128 * x.len() as u128).div_ceil(orig as u128) as usize;
    let mut out = Vec::with_capacity(target);
    'outer: for b in 0..blocks {
        let window = &padded[b * orig..b * orig + taps];
        for k in &kernels {
            if out.len() == target {
                break 'outer;
            }
            out.push(window.iter().zip(k).map(|(a, w)| a * w).sum());
        }
    }
    out
}

/// In-place radix-2 FFT (`re.len()` a power of two).
fn fft(re: &mut [f64], im: &mut [f64]) {
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
        let angle = -2. * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (w_re, w_im) = ((angle * k as f64).cos(), (angle * k as f64).sin());
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

/// torchaudio's slaney mel filterbank (`melscale_fbanks`, slaney norm) as a
/// `(mels, bins)` matrix.
fn slaney_filters(bins: usize, f_max: f64, mels: usize, sample_rate: usize) -> Vec<f32> {
    let (sp, min_hz, step) = (200. / 3., 1000f64, 6.4f64.ln() / 27.);
    let hz_to_mel = |f: f64| if f >= min_hz { min_hz / sp + (f / min_hz).ln() / step } else { f / sp };
    let mel_to_hz = |m: f64| if m >= min_hz / sp { min_hz * (step * (m - min_hz / sp)).exp() } else { sp * m };
    let freqs: Vec<f64> = (0..bins).map(|i| (sample_rate / 2) as f64 * i as f64 / (bins - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.), hz_to_mel(f_max));
    let pts: Vec<f64> = (0..mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (mels + 1) as f64)).collect();
    let mut fb = vec![0f32; mels * bins];
    for m in 0..mels {
        let enorm = 2. / (pts[m + 2] - pts[m]);
        for (i, &f) in freqs.iter().enumerate() {
            let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
            let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
            fb[m * bins + i] = (down.min(up).max(0.) * enorm) as f32;
        }
    }
    fb
}

/// The audio VAE encoder: a stereo waveform to the transformer's audio latent
/// `(1, frames, 128)`, normalized, 25 frames a second. The official
/// `AudioProcessor` and `AudioEncoder`, in F32.
pub struct AudioEncoder {
    w: HashMap<String, Tensor>,
    levels: usize,
    pub sample_rate: usize,
    hop: usize,
    n_fft: usize,
    mel_bins: usize,
    f_max: f64,
}

impl AudioEncoder {
    /// `audio_vae.encoder.*` and the latent statistics from an audio VAE file,
    /// or from an LTX 2.3 / Sulphur checkpoint, which carries its own.
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let mut store = Store::open(path, 0)?;
        let config = Json::parse(store.index.metadata("config").ok_or_else(|| msg("audio VAE file lacks its config metadata"))?.as_bytes())
            .map_err(candle_core::Error::wrap)?;
        let vae = config.get("audio_vae").ok_or_else(|| msg("audio VAE config lacks audio_vae"))?;
        let params = vae.get("model").and_then(|m| m.get("params"));
        let dd = params.and_then(|p| p.get("ddconfig")).ok_or_else(|| msg("audio VAE config lacks audio_vae.model.params.ddconfig"))?;
        for (key, want) in [("z_channels", 8), ("in_channels", 2), ("mel_bins", 64)] {
            if dd.get(key).and_then(Json::as_i64) != Some(want) {
                candle_core::bail!("unsupported audio VAE encoder: {key}");
            }
        }
        if dd.get("norm_type").and_then(Json::as_str) != Some("pixel")
            || dd.get("causality_axis").and_then(Json::as_str) != Some("height")
            || dd.get("mid_block_add_attention").and_then(Json::as_bool) == Some(true)
            || dd.get("attn_resolutions").and_then(Json::as_array).is_some_and(|a| !a.is_empty())
            || dd.get("double_z").and_then(Json::as_bool) == Some(false)
        {
            candle_core::bail!("unsupported audio VAE encoder: needs pixel norm, causal in time, no attention");
        }
        let levels = ints(dd.get("ch_mult"), "ch_mult")?.len();
        let pre = vae.get("preprocessing");
        let num = |section: &str, key: &str, default: f64| pre.and_then(|p| p.get(section)).and_then(|s| s.get(key)).and_then(Json::as_f64).unwrap_or(default);
        let names: Vec<String> = store
            .index
            .names()
            .filter(|k| k.starts_with("audio_vae.encoder.") || k.starts_with("audio_vae.per_channel_statistics."))
            .map(str::to_owned)
            .collect();
        if !names.iter().any(|k| k.starts_with("audio_vae.encoder.")) {
            candle_core::bail!("this audio VAE file has no encoder, so it cannot read a soundtrack");
        }
        let mut w = HashMap::new();
        for k in names {
            let t = store.tensor_f32(&k, dev)?;
            w.insert(k, t);
        }
        Ok(Self {
            w,
            levels,
            sample_rate: params.and_then(|p| p.get("sampling_rate")).and_then(Json::as_i64).unwrap_or(16_000) as usize,
            hop: num("stft", "hop_length", 160.) as usize,
            n_fft: num("stft", "filter_length", 1024.) as usize,
            mel_bins: 64,
            f_max: num("mel", "mel_fmax", 8000.),
        })
    }

    fn get(&self, k: &str) -> Result<&Tensor> {
        self.w.get(k).ok_or_else(|| msg(format!("missing audio encoder tensor {k}")))
    }

    fn conv2d(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        causal_conv2d(x, self.get(&format!("{prefix}.conv.weight"))?, self.get(&format!("{prefix}.conv.bias"))?)
    }

    fn resnet(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = self.conv2d(&pixel_norm(x)?.silu()?, &format!("{prefix}.conv1"))?;
        let h = self.conv2d(&pixel_norm(&h)?.silu()?, &format!("{prefix}.conv2"))?;
        let shortcut = format!("{prefix}.nin_shortcut");
        let x = if self.w.contains_key(&format!("{shortcut}.conv.weight")) { self.conv2d(x, &shortcut)? } else { x.clone() };
        x + h
    }

    /// The log-mel spectrogram of a stereo waveform at `sample_rate`,
    /// `(1, 2, T)` to `(1, 2, frames, 64)`: torchaudio's `MelSpectrogram`
    /// (Hann window, centred with reflection, magnitude, slaney mels), then
    /// `log(max(x, 1e-5))`.
    pub fn mel(&self, wave: &Tensor) -> Result<Tensor> {
        let (b, s, len) = wave.dims3()?;
        let dev = wave.device();
        let pad = self.n_fft / 2;
        if len <= pad {
            candle_core::bail!("the soundtrack is too short to read");
        }
        // On the host, in F64 with an FFT: a direct F32 transform loses the
        // quiet bins, which the log then magnifies.
        let rows = wave.to_dtype(DType::F32)?.reshape((b * s, len))?.to_vec2::<f32>()?;
        let n = self.n_fft;
        let bins = n / 2 + 1;
        let frames = len / self.hop + 1;
        let window: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
        let filters = slaney_filters(bins, self.f_max, self.mel_bins, self.sample_rate);
        let mut out = vec![0f32; b * s * frames * self.mel_bins];
        let (mut re, mut im, mut magnitude) = (vec![0f64; n], vec![0f64; n], vec![0f64; bins]);
        for (row, r) in rows.iter().enumerate() {
            // Reflection padding (the edge sample is not repeated).
            let at = |j: isize| -> f64 {
                let j = if j < 0 { -j } else if j >= len as isize { 2 * (len as isize - 1) - j } else { j };
                r[j as usize] as f64
            };
            for f in 0..frames {
                let start = (f * self.hop) as isize - pad as isize;
                for i in 0..n {
                    re[i] = at(start + i as isize) * window[i];
                    im[i] = 0.;
                }
                fft(&mut re, &mut im);
                for k in 0..bins {
                    magnitude[k] = (re[k] * re[k] + im[k] * im[k]).sqrt();
                }
                for m in 0..self.mel_bins {
                    let v: f64 = filters[m * bins..(m + 1) * bins].iter().zip(&magnitude).map(|(w, x)| *w as f64 * x).sum();
                    out[(row * frames + f) * self.mel_bins + m] = v.max(1e-5).ln() as f32;
                }
            }
        }
        Tensor::from_vec(out, (b, s, frames, self.mel_bins), dev)
    }

    /// A log-mel spectrogram `(1, 2, T, 64)` to the latent `(1, frames, 128)`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let p = "audio_vae.encoder";
        let mut h = self.conv2d(mel, &format!("{p}.conv_in"))?;
        for level in 0..self.levels {
            let mut i = 0;
            while self.w.contains_key(&format!("{p}.down.{level}.block.{i}.conv1.conv.weight")) {
                h = self.resnet(&h, &format!("{p}.down.{level}.block.{i}"))?;
                i += 1;
            }
            if level != self.levels - 1 {
                // Causal in time: two rows of past, none after; one frequency
                // column after. A 3x3 convolution with stride 2.
                let d = format!("{p}.down.{level}.downsample.conv");
                let b = self.get(&format!("{d}.bias"))?;
                h = h
                    .pad_with_zeros(2, 2, 0)?
                    .pad_with_zeros(3, 0, 1)?
                    .conv2d(self.get(&format!("{d}.weight"))?, 0, 2, 1, 1)?
                    .broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?)?;
            }
        }
        h = self.resnet(&h, &format!("{p}.mid.block_1"))?;
        h = self.resnet(&h, &format!("{p}.mid.block_2"))?;
        h = self.conv2d(&pixel_norm(&h)?.silu()?, &format!("{p}.conv_out"))?;
        // The means (the first 8 channels), patchified as channel * 16 +
        // frequency, then normalized.
        let means = h.narrow(1, 0, 8)?;
        let (b, c, t, f) = means.dims4()?;
        let x = means.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, t, c * f))?;
        let std = self.get("audio_vae.per_channel_statistics.std-of-means")?;
        let mean = self.get("audio_vae.per_channel_statistics.mean-of-means")?;
        x.broadcast_sub(mean)?.broadcast_div(std)
    }

    /// Stereo samples at `rate` (one vector per channel) to `frames` latent
    /// frames: resampled, padded with silence to cover them, cropped.
    pub fn latent(&self, channels: &[Vec<f32>; 2], rate: usize, frames: usize, dev: &Device) -> Result<Tensor> {
        let wanted = frames * 4 * self.hop + self.n_fft;
        let mut data = Vec::with_capacity(2 * wanted);
        for c in channels {
            let mut x = resample(c, rate, self.sample_rate);
            x.resize(x.len().max(wanted), 0.);
            x.truncate(wanted);
            data.extend(x);
        }
        let wave = Tensor::from_vec(data, (1, 2, wanted), dev)?;
        let latent = self.encode(&self.mel(&wave)?)?;
        if latent.dim(1)? < frames {
            candle_core::bail!("the soundtrack gave {} latent frames, {frames} needed", latent.dim(1)?);
        }
        latent.narrow(1, 0, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let error = (actual - expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        Ok(error / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
    }

    #[test]
    #[ignore = "requires the audio VAE and tools/ltx/audio_reference.py --part decode output; NROB_LTX_GOLDEN, NROB_LTX_AUDIO_VAE"]
    fn audio_decoder_and_vocoder_match_reference() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?);
        let weights = std::env::var("NROB_LTX_AUDIO_VAE").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(std::env::var("NROB_LTX_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let decoder = AudioDecoder::load(Path::new(&weights), &dev)?;
        assert_eq!(decoder.sample_rate, 48_000);
        let latent = read("audio-latent.f32", &[1, 8, 26, 16])?.permute((0, 2, 1, 3))?.contiguous()?.reshape((1, 26, 128))?;
        let mel = decoder.mel(&latent)?;
        let expected = read("audio-mel.f32", &[1, 2, 101, 64])?;
        let e = relative(&mel, &expected)?;
        println!("audio VAE decoder relative RMS error: {e}");
        assert!(e < 1e-5, "decoder {e}");
        // From the reference mel, so each stage is measured on its own.
        let low = decoder.vocoder_only(&expected)?;
        let e = relative(&low, &read("audio-vocoder.f32", &[1, 2, 16160])?)?;
        println!("vocoder relative RMS error: {e}");
        assert!(e < 1e-4, "vocoder {e}");
        let wave = decoder.waveform(&expected)?;
        let e = relative(&wave, &read("audio-wave.f32", &[1, 2, 48480])?)?;
        println!("vocoder + bandwidth extension relative RMS error: {e}");
        assert!(e < 1e-4, "waveform {e}");
        Ok(())
    }

    #[test]
    fn resampling_keeps_length_and_tone() {
        // 24 kHz to 16 kHz: torchaudio's length rule, and a 1 kHz tone stays one.
        let x: Vec<f32> = (0..24_000).map(|i| (2. * std::f32::consts::PI * 1000. * i as f32 / 24_000.).sin()).collect();
        let y = resample(&x, 24_000, 16_000);
        assert_eq!(y.len(), 16_000);
        for i in 1_000..15_000 {
            let want = (2. * std::f32::consts::PI * 1000. * i as f32 / 16_000.).sin();
            assert!((y[i] - want).abs() < 0.02, "sample {i}: {} vs {want}", y[i]);
        }
        assert_eq!(resample(&x[..441], 44_100, 16_000).len(), 160);
    }

    #[test]
    #[ignore = "requires the audio VAE and tools/ltx/audio_reference.py --part encode output; NROB_LTX_GOLDEN, NROB_LTX_AUDIO_VAE"]
    fn audio_encoder_matches_reference() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?);
        let weights = std::env::var("NROB_LTX_AUDIO_VAE").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(std::env::var("NROB_LTX_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let encoder = AudioEncoder::load(Path::new(&weights), &dev)?;
        let wave = read("audio-encode-wave.f32", &[1, 2, 55_200])?.to_vec3::<f32>()?;
        let resampled: Vec<Vec<f32>> = wave[0].iter().map(|c| resample(c, 24_000, 16_000)).collect();
        let r = Tensor::new(resampled, &dev)?.unsqueeze(0)?;
        let e = relative(&r, &read("audio-encode-resampled.f32", &[1, 2, 36_800])?)?;
        println!("resampler relative RMS error: {e}");
        assert!(e < 1e-5, "resampler {e}");
        let mel = encoder.mel(&r)?;
        let expected = read("audio-encode-mel.f32", &[1, 2, 231, 64])?;
        let e = relative(&mel, &expected)?;
        println!("log-mel relative RMS error: {e}");
        // The reference's own run-to-run spread on this signal (half its bins sit
        // at the log floor): torch on the CPU differs by 5e-5 in F32, 4e-5 in F64.
        assert!(e < 2e-4, "mel {e}");
        // From the reference mel, so the encoder is measured on its own.
        let latent = encoder.encode(&expected)?;
        let want = read("audio-encode-latent.f32", &[1, 8, 58, 16])?.permute((0, 2, 1, 3))?.contiguous()?.reshape((1, 58, 128))?;
        let e = relative(&latent, &want)?;
        println!("audio VAE encoder relative RMS error: {e}");
        assert!(e < 1e-5, "encoder {e}");
        Ok(())
    }
}
