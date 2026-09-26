//! What a reusable voice is made of: the speaker embedding (ECAPA-TDNN over a
//! log-mel spectrogram, from the Base model's `speaker_encoder.*`) and the
//! reference clip's codec codes (the speech tokenizer's Mimi encoder,
//! `encoder.*`). Both run once, when a voice is saved, in F32.
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

fn load_prefix(path: &Path, prefix: &str, dev: &Device) -> Result<HashMap<String, Tensor>> {
    let mut store = Store::open(path, 0)?;
    let names: Vec<String> = store.index.names().filter(|k| k.starts_with(prefix)).map(str::to_owned).collect();
    if names.is_empty() {
        return Err(msg(format!("{}: no {prefix}* tensors (the Base model carries the speaker encoder)", path.display())));
    }
    let mut w = HashMap::new();
    for k in names {
        let t = store.tensor_f32(&k, dev)?;
        w.insert(k, t);
    }
    Ok(w)
}

/// Reflect padding on the last axis (candle has zero and replicate only).
fn reflect(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let n = x.dim(D::Minus1)? as i64;
    let idx: Vec<u32> = (-(left as i64)..n + right as i64)
        .map(|i| {
            let mut i = i;
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * (n - 1) - i;
            }
            i as u32
        })
        .collect();
    x.index_select(&Tensor::new(idx.as_slice(), x.device())?, x.rank() - 1)
}

/// librosa's slaney mel filterbank (`htk=False`, `norm="slaney"`).
fn mel_filters(sr: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
    let hz_to_mel = |f: f64| {
        let (f_sp, min_log_hz) = (200. / 3., 1000.);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4f64.ln() / 27.;
        if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
    };
    let mel_to_hz = |m: f64| {
        let (f_sp, min_log_hz) = (200. / 3., 1000.);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4f64.ln() / 27.;
        if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
    };
    let bins = n_fft / 2 + 1;
    let fft: Vec<f64> = (0..bins).map(|i| i as f64 * sr / n_fft as f64).collect();
    let (lo, hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut w = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let enorm = 2. / (mel_f[m + 2] - mel_f[m]);
        for (b, f) in fft.iter().enumerate() {
            let lower = (f - mel_f[m]) / (mel_f[m + 1] - mel_f[m]);
            let upper = (mel_f[m + 2] - f) / (mel_f[m + 2] - mel_f[m + 1]);
            w[m * bins + b] = (lower.min(upper).max(0.) * enorm) as f32;
        }
    }
    w
}

pub struct SpeakerEncoder {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl SpeakerEncoder {
    /// `speaker_encoder.*` from the Base model's `model.safetensors`.
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        Ok(Self { w: load_prefix(path, "speaker_encoder.", dev)?, dev: dev.clone() })
    }

    fn get(&self, k: &str) -> Result<&Tensor> {
        self.w.get(&format!("speaker_encoder.{k}")).ok_or_else(|| msg(format!("missing speaker_encoder.{k}")))
    }

    /// 24 kHz audio to a (1, 128, frames) log-mel: reflect-padded Hann STFT
    /// (1024 / 256), magnitude, slaney mel 0-12 kHz, log.
    pub fn mel(&self, audio: &[f32]) -> Result<Tensor> {
        let (n_fft, hop) = (1024usize, 256usize);
        let bins = n_fft / 2 + 1;
        let x = Tensor::from_slice(audio, (1, 1, audio.len()), &self.dev)?;
        let pad = (n_fft - hop) / 2;
        let x = reflect(&x, pad, pad)?;
        // DFT basis times a periodic Hann window: real rows, then imaginary.
        let mut basis = vec![0f32; 2 * bins * n_fft];
        for k in 0..bins {
            for n in 0..n_fft {
                let w = 0.5 - 0.5 * (2. * std::f64::consts::PI * n as f64 / n_fft as f64).cos();
                let a = 2. * std::f64::consts::PI * (k * n) as f64 / n_fft as f64;
                basis[k * n_fft + n] = (w * a.cos()) as f32;
                basis[(bins + k) * n_fft + n] = (-w * a.sin()) as f32;
            }
        }
        let basis = Tensor::from_vec(basis, (2 * bins, 1, n_fft), &self.dev)?;
        let spec = x.conv1d(&basis, 0, hop, 1, 1)?; // (1, 2*bins, frames)
        let (re, im) = (spec.narrow(1, 0, bins)?, spec.narrow(1, bins, bins)?);
        let magnitude = ((re.sqr()? + im.sqr()?)? + 1e-9)?.sqrt()?;
        let filters = Tensor::from_vec(mel_filters(24_000., n_fft, 128, 0., 12_000.), (128, bins), &self.dev)?;
        filters.broadcast_matmul(&magnitude)?.clamp(1e-5f32, f32::MAX)?.log()
    }

    /// Conv1d with reflect "same" padding, bias, and optional ReLU.
    fn conv(&self, x: &Tensor, name: &str, dilation: usize, relu: bool) -> Result<Tensor> {
        let w = self.get(&format!("{name}.weight"))?;
        let b = self.get(&format!("{name}.bias"))?;
        let k = w.dim(2)?;
        let total = dilation * (k - 1);
        let y = reflect(x, total / 2, total - total / 2)?.conv1d(w, 0, 1, dilation, 1)?.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?;
        if relu { y.relu() } else { Ok(y) }
    }

    fn se_res2net(&self, x: &Tensor, i: usize, dilation: usize) -> Result<Tensor> {
        let p = format!("blocks.{i}");
        let h = self.conv(x, &format!("{p}.tdnn1.conv"), 1, true)?;
        let parts = h.chunk(8, 1)?;
        let mut outs = vec![parts[0].clone()];
        for (j, part) in parts.iter().enumerate().skip(1) {
            let input = if j == 1 { part.clone() } else { (part + &outs[j - 1])? };
            outs.push(self.conv(&input, &format!("{p}.res2net_block.blocks.{}.conv", j - 1), dilation, true)?);
        }
        let h = self.conv(&Tensor::cat(&outs, 1)?, &format!("{p}.tdnn2.conv"), 1, true)?;
        let s = h.mean_keepdim(2)?;
        let s = self.conv(&s, &format!("{p}.se_block.conv1"), 1, true)?;
        let s = candle_nn::ops::sigmoid(&self.conv(&s, &format!("{p}.se_block.conv2"), 1, false)?)?;
        h.broadcast_mul(&s)? + x
    }

    /// Weighted mean and standard deviation over time (weights sum to 1).
    fn statistics(x: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = x.broadcast_mul(weights)?.sum_keepdim(2)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.broadcast_mul(weights)?.sum_keepdim(2)?;
        Ok((mean, var.clamp(1e-12f32, f32::MAX)?.sqrt()?))
    }

    /// The 2048-value speaker embedding of 24 kHz audio.
    pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mel = self.mel(audio)?;
        self.embed_mel(&mel)?.flatten_all()?.to_vec1::<f32>()
    }

    fn embed_mel(&self, mel: &Tensor) -> Result<Tensor> {
        let mut h = self.conv(mel, "blocks.0.conv", 1, true)?;
        let mut skips = Vec::new();
        for (i, d) in [(1, 2), (2, 3), (3, 4)] {
            h = self.se_res2net(&h, i, d)?;
            skips.push(h.clone());
        }
        let h = self.conv(&Tensor::cat(&skips, 1)?, "mfa.conv", 1, true)?;
        let t = h.dim(2)?;
        let uniform = Tensor::full(1f32 / t as f32, (1, 1, t), &self.dev)?;
        let (mean, std) = Self::statistics(&h, &uniform)?;
        let context = Tensor::cat(&[h.clone(), mean.broadcast_as(h.shape())?.contiguous()?, std.broadcast_as(h.shape())?.contiguous()?], 1)?;
        let a = self.conv(&context, "asp.tdnn.conv", 1, true)?.tanh()?;
        let a = candle_nn::ops::softmax(&self.conv(&a, "asp.conv", 1, false)?, 2)?;
        let (mean, std) = Self::statistics(&h, &a)?;
        let pooled = Tensor::cat(&[mean, std], 1)?; // (1, 3072, 1)
        self.conv(&pooled, "fc", 1, false)
    }
}

/// The speech tokenizer's encoder (Mimi): 24 kHz audio to 12.5 Hz frames of
/// 16 codebook ids, the same codes the codec decoder turns back into audio.
pub struct SpeechEncoder {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl SpeechEncoder {
    /// `encoder.*` from `speech_tokenizer/model.safetensors`.
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        Ok(Self { w: load_prefix(path, "encoder.", dev)?, dev: dev.clone() })
    }

    fn get(&self, k: &str) -> Result<&Tensor> {
        self.w.get(k).ok_or_else(|| msg(format!("missing {k}")))
    }

    /// Mimi's causal conv: zero (or replicate) left padding of `k - stride`,
    /// plus right padding so the last frame is whole.
    fn conv(&self, x: &Tensor, name: &str, stride: usize, replicate: bool) -> Result<Tensor> {
        let w = self.get(&format!("{name}.weight"))?;
        let k = w.dim(2)?;
        let len = x.dim(2)?;
        let pad = k - stride;
        let frames = ((len + pad) as f64 - k as f64) / stride as f64 + 1.;
        let ideal = (frames.ceil() as usize - 1) * stride + k - pad;
        let extra = ideal.saturating_sub(len);
        let x = if replicate { x.pad_with_same(2, pad, extra)? } else { x.pad_with_zeros(2, pad, extra)? };
        let y = x.conv1d(w, 0, stride, 1, 1)?;
        match self.w.get(&format!("{name}.bias")) {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }

    /// SEANet: conv 1->64, then per stage (strides 4, 5, 6, 8) a residual
    /// block and a strided conv doubling the channels; ELU; conv to 512.
    fn seanet(&self, audio: &Tensor) -> Result<Tensor> {
        let p = "encoder.encoder.layers";
        let mut h = self.conv(audio, &format!("{p}.0.conv"), 1, false)?;
        let mut layer = 1;
        for stride in [4, 5, 6, 8] {
            let r = self.conv(&h.elu(1.)?, &format!("{p}.{layer}.block.1.conv"), 1, false)?;
            let r = self.conv(&r.elu(1.)?, &format!("{p}.{layer}.block.3.conv"), 1, false)?;
            h = (h + r)?;
            h = self.conv(&h.elu(1.)?, &format!("{p}.{}.conv", layer + 2), stride, false)?;
            layer += 3;
        }
        self.conv(&h.elu(1.)?, &format!("{p}.{}.conv", layer + 1), 1, false)
    }

    fn layer_norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let c = x.broadcast_sub(&mean)?;
        let var = c.sqr()?.mean_keepdim(D::Minus1)?;
        c.broadcast_div(&(var + 1e-5)?.sqrt()?)?.broadcast_mul(self.get(&format!("{name}.weight"))?)?.broadcast_add(self.get(&format!("{name}.bias"))?)
    }

    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        x.broadcast_matmul(&self.get(&format!("{name}.weight"))?.t()?)
    }

    /// 8 causal layers: LayerNorm, attention (8 heads of 64, RoPE 1e4),
    /// LayerNorm, GELU MLP, each branch layer-scaled. `x`: (1, T, 512).
    fn transformer(&self, x: &Tensor) -> Result<Tensor> {
        let (t, heads, hd) = (x.dim(1)?, 8usize, 64usize);
        let inv: Vec<f32> = (0..hd / 2).map(|i| (1. / 10000f64.powf(2. * i as f64 / hd as f64)) as f32).collect();
        let freqs = Tensor::from_vec((0..t).flat_map(|p| inv.iter().map(move |f| p as f32 * f)).collect::<Vec<f32>>(), (t, hd / 2), &self.dev)?;
        let (cos, sin) = (freqs.cos()?, freqs.sin()?);
        let mask: Vec<f32> = (0..t).flat_map(|q| (0..t).map(move |k| if k <= q { 0. } else { f32::NEG_INFINITY })).collect();
        let mask = Tensor::from_vec(mask, (t, t), &self.dev)?;
        let mut h = x.clone();
        let mut i = 0;
        while self.w.contains_key(&format!("encoder.encoder_transformer.layers.{i}.self_attn.q_proj.weight")) {
            let l = format!("encoder.encoder_transformer.layers.{i}");
            let n = self.layer_norm(&h, &format!("{l}.input_layernorm"))?;
            let split = |y: Tensor| -> Result<Tensor> { y.reshape((1, t, heads, hd))?.transpose(1, 2)?.contiguous() };
            let q = candle_nn::rotary_emb::rope(&split(self.linear(&n, &format!("{l}.self_attn.q_proj"))?)?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&split(self.linear(&n, &format!("{l}.self_attn.k_proj"))?)?, &cos, &sin)?;
            let v = split(self.linear(&n, &format!("{l}.self_attn.v_proj"))?)?;
            let s = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?.broadcast_add(&mask)?;
            let a = candle_nn::ops::softmax_last_dim(&s)?.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((1, t, heads * hd))?;
            let a = self.linear(&a, &format!("{l}.self_attn.o_proj"))?;
            h = (h + a.broadcast_mul(self.get(&format!("{l}.self_attn_layer_scale.scale"))?)?)?;
            let n = self.layer_norm(&h, &format!("{l}.post_attention_layernorm"))?;
            let m = self.linear(&self.linear(&n, &format!("{l}.mlp.fc1"))?.gelu_erf()?, &format!("{l}.mlp.fc2"))?;
            h = (h + m.broadcast_mul(self.get(&format!("{l}.mlp_layer_scale.scale"))?)?)?;
            i += 1;
        }
        Ok(h)
    }

    /// Nearest-centroid indices for `x` (T, 256) in one codebook, and the
    /// chosen centroids.
    fn nearest(&self, x: &Tensor, codebook: &str) -> Result<(Vec<u32>, Tensor)> {
        let sum = self.get(&format!("{codebook}.embed_sum"))?;
        let usage = self.get(&format!("{codebook}.cluster_usage"))?;
        let embed = sum.broadcast_div(&usage.clamp(1e-5f32, f32::MAX)?.unsqueeze(1)?)?;
        // |x - e|^2 = |x|^2 - 2 x.e + |e|^2; |x|^2 is the same for every e.
        let d = (embed.sqr()?.sum_keepdim(1)?.t()?.broadcast_sub(&(x.matmul(&embed.t()?)? * 2.)?))?;
        let idx = d.argmin(1)?;
        let chosen = embed.index_select(&idx, 0)?;
        Ok((idx.to_vec1::<u32>()?, chosen))
    }

    /// Up to `quantizers` residual codebooks over the 12.5 Hz embeddings.
    fn quantize(&self, emb: &Tensor, quantizers: usize) -> Result<Vec<Vec<u32>>> {
        let q = "encoder.quantizer";
        let project = |name: &str| -> Result<Tensor> { emb.conv1d(self.get(&format!("{q}.{name}.input_proj.weight"))?, 0, 1, 1, 1)?.squeeze(0)?.t()?.contiguous() };
        let (semantic, _) = self.nearest(&project("semantic_residual_vector_quantizer")?, &format!("{q}.semantic_residual_vector_quantizer.layers.0.codebook"))?;
        let mut codes = vec![semantic];
        let mut residual = project("acoustic_residual_vector_quantizer")?;
        for i in 0..quantizers - 1 {
            let (idx, chosen) = self.nearest(&residual, &format!("{q}.acoustic_residual_vector_quantizer.layers.{i}.codebook"))?;
            residual = (residual - chosen)?;
            codes.push(idx);
        }
        Ok(codes)
    }

    /// 24 kHz audio to frames of 16 codes (as many frames as whole or partial
    /// 1920-sample periods).
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<[u32; 16]>> {
        let x = Tensor::from_slice(audio, (1, 1, audio.len()), &self.dev)?.to_dtype(DType::F32)?;
        let h = self.seanet(&x)?;
        let h = self.transformer(&h.transpose(1, 2)?.contiguous()?)?.transpose(1, 2)?.contiguous()?;
        let h = self.conv(&h, "encoder.downsample.conv", 2, true)?;
        let codes = self.quantize(&h, 16)?;
        let frames = audio.len().div_ceil(super::codec::SAMPLES_PER_FRAME).min(codes[0].len());
        Ok((0..frames).map(|f| std::array::from_fn(|q| codes[q][f])).collect())
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
    fn reflect_padding_mirrors_without_the_edge() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3., 4.]], &Device::Cpu)?;
        assert_eq!(reflect(&x, 2, 2)?.to_vec2::<f32>()?, vec![vec![3., 2., 1., 2., 3., 4., 3., 2.]]);
        Ok(())
    }

    #[test]
    #[ignore = "requires the Base model and clone reference dumps; NROB_TTS_GOLDEN (clone folder), NROB_TTS_BASE"]
    fn encoders_match_reference() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("NROB_TTS_GOLDEN").map_err(candle_core::Error::wrap)?);
        let base = std::path::PathBuf::from(std::env::var("NROB_TTS_BASE").map_err(candle_core::Error::wrap)?);
        let dev = Device::new_cuda(std::env::var("NROB_TTS_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let wav: Vec<f32> = std::fs::read(root.join("ref_wav.f32"))?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let spk = SpeakerEncoder::load(&base.join("model.safetensors"), &dev)?;
        let mel = spk.mel(&wav)?;
        let t = mel.dim(2)?;
        let e = relative(&mel.squeeze(0)?.t()?, &read("spk_mel.f32", &[t, 128])?)?;
        println!("speaker mel ({t} frames): relative RMS error {e}");
        assert!(e < 1e-4, "mel {e}");
        let emb = Tensor::new(spk.embed(&wav)?, &dev)?;
        let e = relative(&emb, &read("spk_embedding_fp32.f32", &[2048])?)?;
        println!("speaker embedding: relative RMS error {e}");
        assert!(e < 1e-3, "speaker embedding {e}");
        let enc = SpeechEncoder::load(&base.join("speech_tokenizer").join("model.safetensors"), &dev)?;
        let x = Tensor::from_slice(&wav, (1, 1, wav.len()), &dev)?;
        let s = enc.seanet(&x)?;
        let e = relative(&s.squeeze(0)?, &read("enc_seanet_strict.f32", &[512, s.dim(2)?])?)?;
        println!("SEANet: relative RMS error {e}");
        assert!(e < 1e-4, "seanet {e}");
        let codes = enc.encode(&wav)?;
        let want: Vec<u32> = std::fs::read(root.join("ref_code_strict.i32"))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect();
        let total = codes.len() * 16;
        let same = codes.iter().flatten().zip(&want).filter(|(a, b)| a == b).count();
        let first = codes.iter().zip(want.chunks(16)).filter(|(a, b)| a[0] == b[0]).count();
        println!("codes: {same} of {total} identical; codebook 0: {first} of {} frames", codes.len());
        assert_eq!(codes.len() * 16, want.len());
        assert!(first * 100 >= codes.len() * 95, "semantic codes differ");
        Ok(())
    }
}
