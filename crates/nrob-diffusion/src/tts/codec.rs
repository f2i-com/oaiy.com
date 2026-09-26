//! The Qwen3-TTS 12 Hz speech-codec decoder: 16 codebook ids per frame to
//! 24 kHz mono audio (1920 samples per frame), after the official
//! `Qwen3TTSTokenizerV2Decoder`. It runs in F32, as the reference does.
//!
//! Stages: split residual VQ (1 + 15 codebooks) -> causal conv -> an 8-layer
//! transformer with 72-frame sliding-window attention -> two ConvNeXt 2x
//! upsamplers -> four SnakeBeta / transposed-conv blocks (8, 5, 4, 3x) -> a
//! final causal conv, clamped to [-1, 1].
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

/// Samples per codec frame (12.5 frames per second at 24 kHz).
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const SAMPLE_RATE: usize = 24_000;
/// The official decoder works on 300-frame chunks with 25 frames of left
/// context; longer inputs are decoded the same way.
const CHUNK: usize = 300;
const LEFT_CONTEXT: usize = 25;

pub struct CodecDecoder {
    w: HashMap<String, Tensor>,
    /// Codebook vectors (2048 x 256) for each of the 16 quantizers.
    codebooks: Vec<Tensor>,
    layers: usize,
    heads: usize,
    window: usize,
    rope_theta: f64,
    upsample_rates: Vec<usize>,
    dev: Device,
}

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

impl CodecDecoder {
    /// Load `decoder.*` from the speech tokenizer's `model.safetensors`
    /// (the encoder half is only needed to clone a voice from audio).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let config: nrob::json::Json = {
            let cfg = path.parent().map(|p| p.join("config.json")).ok_or_else(|| msg("speech tokenizer path has no folder"))?;
            nrob::json::Json::parse(&std::fs::read(&cfg).map_err(|e| msg(format!("{}: {e}", cfg.display())))?).map_err(candle_core::Error::wrap)?
        };
        let dc = config.get("decoder_config").ok_or_else(|| msg("speech tokenizer config lacks decoder_config"))?;
        let int = |k: &str| dc.get(k).and_then(nrob::json::Json::as_i64).map(|v| v as usize).ok_or_else(|| msg(format!("decoder_config lacks {k}")));
        let upsample_rates = dc
            .get("upsample_rates")
            .and_then(nrob::json::Json::as_array)
            .ok_or_else(|| msg("decoder_config lacks upsample_rates"))?
            .iter()
            .filter_map(|v| v.as_i64().map(|v| v as usize))
            .collect::<Vec<_>>();
        let quantizers = int("num_quantizers")?;
        let mut store = Store::open(path, 0)?;
        let names: Vec<String> = store.index.names().filter(|k| k.starts_with("decoder.")).map(str::to_owned).collect();
        let mut w = HashMap::new();
        for k in names {
            let t = store.tensor_f32(&k, dev)?;
            w.insert(k, t);
        }
        let codebook = |prefix: &str, i: usize| -> Result<Tensor> {
            let base = format!("decoder.quantizer.{prefix}.vq.layers.{i}._codebook");
            let sum = w.get(&format!("{base}.embedding_sum")).ok_or_else(|| msg(format!("missing {base}")))?;
            let usage = w.get(&format!("{base}.cluster_usage")).ok_or_else(|| msg(format!("missing {base}")))?;
            sum.broadcast_div(&usage.clamp(1e-5f32, f32::MAX)?.unsqueeze(1)?)
        };
        let mut codebooks = vec![codebook("rvq_first", 0)?];
        for i in 0..quantizers - 1 {
            codebooks.push(codebook("rvq_rest", i)?);
        }
        Ok(Self {
            w,
            codebooks,
            layers: int("num_hidden_layers")?,
            heads: int("num_attention_heads")?,
            window: int("sliding_window")?,
            rope_theta: dc.get("rope_theta").and_then(nrob::json::Json::as_f64).unwrap_or(10000.),
            upsample_rates,
            dev: dev.clone(),
        })
    }

    fn get(&self, k: &str) -> Result<&Tensor> {
        self.w.get(k).ok_or_else(|| msg(format!("missing codec tensor {k}")))
    }

    /// Frames of 16 codes to a 24 kHz waveform, decoded in the official
    /// 300-frame chunks with 25 frames of left context.
    pub fn decode(&self, frames: &[[u32; 16]]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(frames.len() * SAMPLES_PER_FRAME);
        let mut start = 0;
        while start < frames.len() {
            let end = (start + CHUNK).min(frames.len());
            let context = if start > LEFT_CONTEXT { LEFT_CONTEXT } else { start };
            let wave = self.forward(&frames[start - context..end])?;
            out.extend_from_slice(&wave[context * SAMPLES_PER_FRAME..]);
            start = end;
        }
        Ok(out)
    }

    /// One full-sequence decode.
    pub fn forward(&self, frames: &[[u32; 16]]) -> Result<Vec<f32>> {
        let h = self.quantized(frames)?;
        let h = self.pre_conv(&h)?;
        let h = self.transformer(&h.transpose(1, 2)?.contiguous()?)?;
        let h = self.upsample(&h.transpose(1, 2)?.contiguous()?)?;
        let wave = self.vocode(&h)?;
        wave.flatten_all()?.to_vec1::<f32>()
    }

    /// Sum of the 16 codebooks' vectors, each group through its output
    /// projection: (1, 512, T).
    fn quantized(&self, frames: &[[u32; 16]]) -> Result<Tensor> {
        let t = frames.len();
        let ids = |q: usize| -> Result<Tensor> { Tensor::from_vec(frames.iter().map(|f| f[q]).collect::<Vec<u32>>(), t, &self.dev) };
        let first = self.codebooks[0].index_select(&ids(0)?, 0)?;
        let mut rest = self.codebooks[1].index_select(&ids(1)?, 0)?;
        for q in 2..self.codebooks.len() {
            rest = (rest + self.codebooks[q].index_select(&ids(q)?, 0)?)?;
        }
        let project = |x: Tensor, name: &str| -> Result<Tensor> {
            x.t()?.unsqueeze(0)?.contiguous()?.conv1d(self.get(&format!("decoder.quantizer.{name}.output_proj.weight"))?, 0, 1, 1, 1)
        };
        project(first, "rvq_first")? + project(rest, "rvq_rest")?
    }

    fn pre_conv(&self, x: &Tensor) -> Result<Tensor> {
        self.causal_conv(x, "decoder.pre_conv.conv", 1)
    }

    /// A stride-1 causal convolution: the past is zero-padded by `(k-1)*d`.
    fn causal_conv(&self, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let k = w.dim(2)?;
        let y = x.pad_with_zeros(2, (k - 1) * dilation, 0)?.conv1d(w, 0, 1, dilation, 1)?;
        match self.w.get(&format!("{prefix}.bias")) {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }

    /// Transposed convolution trimmed on the right by `kernel - stride`.
    fn causal_trans_conv(&self, x: &Tensor, prefix: &str, stride: usize) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let b = self.get(&format!("{prefix}.bias"))?;
        let k = w.dim(2)?;
        let y = x.conv_transpose1d(w, 0, 0, stride, 1, 1)?.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?;
        let len = y.dim(2)?;
        y.narrow(2, 0, len - (k - stride))
    }

    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.get(&format!("{prefix}.weight"))?.t()?)?;
        match self.w.get(&format!("{prefix}.bias")) {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn rms(&self, x: &Tensor, weight: &str, eps: f64) -> Result<Tensor> {
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(self.get(weight)?)
    }

    /// input_proj, 8 layers of sliding-window attention and MLP with layer
    /// scales, final norm, output_proj. `x`: (1, T, 1024).
    fn transformer(&self, x: &Tensor) -> Result<Tensor> {
        let p = "decoder.pre_transformer";
        let t = x.dim(1)?;
        let mut h = self.linear(x, &format!("{p}.input_proj"))?;
        let dim = h.dim(2)?;
        let head_dim = self.get(&format!("{p}.layers.0.self_attn.q_proj.weight"))?.dim(0)? / self.heads;
        // Rotate-half RoPE from position 0.
        let inv: Vec<f32> = (0..head_dim / 2).map(|i| 1. / self.rope_theta.powf(2. * i as f64 / head_dim as f64) as f32).collect();
        let freqs: Vec<f32> = (0..t).flat_map(|pos| inv.iter().map(move |f| pos as f32 * f)).collect();
        let freqs = Tensor::from_vec(freqs, (t, head_dim / 2), &self.dev)?;
        let (cos, sin) = (freqs.cos()?, freqs.sin()?);
        // Causal, and at most `window` keys back (the current one included).
        let mask: Vec<f32> = (0..t)
            .flat_map(|q| (0..t).map(move |k| if k <= q && k + self.window > q { 0. } else { f32::NEG_INFINITY }))
            .collect();
        let mask = Tensor::from_vec(mask, (t, t), &self.dev)?;
        let scale = 1. / (head_dim as f64).sqrt();
        for i in 0..self.layers {
            let l = format!("{p}.layers.{i}");
            let n = self.rms(&h, &format!("{l}.input_layernorm.weight"), 1e-5)?;
            let split = |y: Tensor| -> Result<Tensor> { y.reshape((1, t, self.heads, head_dim))?.transpose(1, 2)?.contiguous() };
            let q = candle_nn::rotary_emb::rope(&split(self.linear(&n, &format!("{l}.self_attn.q_proj"))?)?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&split(self.linear(&n, &format!("{l}.self_attn.k_proj"))?)?, &cos, &sin)?;
            let v = split(self.linear(&n, &format!("{l}.self_attn.v_proj"))?)?;
            let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(&mask)?;
            let a = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
            let a = a.transpose(1, 2)?.contiguous()?.reshape((1, t, self.heads * head_dim))?;
            let a = self.linear(&a, &format!("{l}.self_attn.o_proj"))?;
            h = (h + a.broadcast_mul(self.get(&format!("{l}.self_attn_layer_scale.scale"))?)?)?;
            let n = self.rms(&h, &format!("{l}.post_attention_layernorm.weight"), 1e-5)?;
            let g = self.linear(&n, &format!("{l}.mlp.gate_proj"))?.silu()?;
            let m = self.linear(&(g * self.linear(&n, &format!("{l}.mlp.up_proj"))?)?, &format!("{l}.mlp.down_proj"))?;
            h = (h + m.broadcast_mul(self.get(&format!("{l}.mlp_layer_scale.scale"))?)?)?;
        }
        let _ = dim;
        let h = self.rms(&h, &format!("{p}.norm.weight"), 1e-5)?;
        self.linear(&h, &format!("{p}.output_proj"))
    }

    /// Two stages of a 2x transposed conv and a ConvNeXt block. (1, C, T).
    fn upsample(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        let mut i = 0;
        while self.w.contains_key(&format!("decoder.upsample.{i}.0.conv.weight")) {
            let p = format!("decoder.upsample.{i}");
            let w = self.get(&format!("{p}.0.conv.weight"))?;
            h = self.causal_trans_conv(&h, &format!("{p}.0.conv"), w.dim(2)?)?;
            h = self.convnext(&h, &format!("{p}.1"))?;
            i += 1;
        }
        Ok(h)
    }

    /// Depthwise causal conv k7, LayerNorm, pointwise MLP with exact GELU,
    /// times gamma, plus the input.
    fn convnext(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let w = self.get(&format!("{p}.dwconv.conv.weight"))?; // (C, 1, k)
        let b = self.get(&format!("{p}.dwconv.conv.bias"))?;
        let (c, _, k) = w.dims3()?;
        let t = x.dim(2)?;
        // Depthwise as k shifted multiply-adds (a grouped conv would run one
        // convolution per channel).
        let padded = x.pad_with_zeros(2, k - 1, 0)?;
        let mut y = b.reshape((1, c, 1))?.broadcast_as((1, c, t))?.contiguous()?;
        for j in 0..k {
            y = (y + padded.narrow(2, j, t)?.broadcast_mul(&w.narrow(2, j, 1)?.reshape((1, c, 1))?)?)?;
        }
        let y = y.transpose(1, 2)?.contiguous()?; // (1, T, C)
        let mean = y.mean_keepdim(D::Minus1)?;
        let centered = y.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let y = centered
            .broadcast_div(&(var + 1e-6)?.sqrt()?)?
            .broadcast_mul(self.get(&format!("{p}.norm.weight"))?)?
            .broadcast_add(self.get(&format!("{p}.norm.bias"))?)?;
        let y = self.linear(&y, &format!("{p}.pwconv1"))?.gelu_erf()?;
        let y = self.linear(&y, &format!("{p}.pwconv2"))?.broadcast_mul(self.get(&format!("{p}.gamma"))?)?;
        x + y.transpose(1, 2)?
    }

    /// `x + sin^2(x * e^alpha) / (e^beta + 1e-9)`, per channel.
    fn snake(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let c = x.dim(1)?;
        let alpha = self.get(&format!("{p}.alpha"))?.exp()?.reshape((1, c, 1))?;
        let beta = ((self.get(&format!("{p}.beta"))?.exp()? + 1e-9)?.recip()?).reshape((1, c, 1))?;
        x + x.broadcast_mul(&alpha)?.sin()?.sqr()?.broadcast_mul(&beta)?
    }

    fn vocode(&self, x: &Tensor) -> Result<Tensor> {
        let p = "decoder.decoder";
        let mut h = self.causal_conv(x, &format!("{p}.0.conv"), 1)?;
        for (i, &rate) in self.upsample_rates.iter().enumerate() {
            let b = format!("{p}.{}.block", i + 1);
            h = self.snake(&h, &format!("{b}.0"))?;
            h = self.causal_trans_conv(&h, &format!("{b}.1.conv"), rate)?;
            for (j, d) in [1, 3, 9].into_iter().enumerate() {
                let u = format!("{b}.{}", j + 2);
                let r = self.snake(&h, &format!("{u}.act1"))?;
                let r = self.causal_conv(&r, &format!("{u}.conv1.conv"), d)?;
                let r = self.snake(&r, &format!("{u}.act2"))?;
                let r = self.causal_conv(&r, &format!("{u}.conv2.conv"), 1)?;
                h = (h + r)?;
            }
        }
        let n = self.upsample_rates.len();
        h = self.snake(&h, &format!("{p}.{}", n + 1))?;
        h = self.causal_conv(&h, &format!("{p}.{}.conv", n + 2), 1)?;
        h.clamp(-1f32, 1f32)?.to_dtype(DType::F32)
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
    #[ignore = "requires the Qwen3-TTS speech tokenizer and reference stages; NROB_TTS_GOLDEN, NROB_TTS_MODEL"]
    fn codec_decoder_matches_reference() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("NROB_TTS_GOLDEN").map_err(candle_core::Error::wrap)?);
        let model = std::path::PathBuf::from(std::env::var("NROB_TTS_MODEL").map_err(candle_core::Error::wrap)?);
        let dev = Device::new_cuda(std::env::var("NROB_TTS_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let codes: Vec<u32> = std::fs::read(root.join("sampled_codes.i32"))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect();
        let frames: Vec<[u32; 16]> = codes.chunks_exact(16).map(|c| c.try_into().unwrap()).collect();
        let t = frames.len();
        let dec = CodecDecoder::load(&model.join("speech_tokenizer").join("model.safetensors"), &dev)?;
        let mut worst = Vec::new();
        let mut check = |name: &str, actual: &Tensor, shape: &[usize], limit: f32| -> Result<()> {
            let e = relative(actual, &read(name, shape)?)?;
            println!("{name}: relative RMS error {e}");
            if e >= limit {
                worst.push(format!("{name} {e}"));
            }
            Ok(())
        };
        let q = dec.quantized(&frames)?;
        check("dec_quantized.f32", &q.squeeze(0)?, &[512, t], 1e-5)?;
        let pc = dec.pre_conv(&q)?;
        check("dec_pre_conv.f32", &pc.squeeze(0)?, &[1024, t], 1e-5)?;
        let pt = dec.transformer(&pc.transpose(1, 2)?.contiguous()?)?;
        check("dec_pre_transformer.f32", &pt.squeeze(0)?, &[t, 1024], 1e-5)?;
        let up = dec.upsample(&pt.transpose(1, 2)?.contiguous()?)?;
        check("dec_upsample1.f32", &up.squeeze(0)?, &[1024, 4 * t], 1e-5)?;
        let wave = Tensor::new(dec.forward(&frames)?, &dev)?;
        check("dec_wave.f32", &wave, &[t * SAMPLES_PER_FRAME], 1e-5)?;
        assert!(worst.is_empty(), "{worst:?}");
        Ok(())
    }
}
