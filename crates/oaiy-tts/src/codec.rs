//! The Qwen3-TTS 12 Hz speech-codec decoder: 16 codebook ids per frame to
//! 24 kHz mono audio (1920 samples per frame), after the official
//! `Qwen3TTSTokenizerV2Decoder`. It runs in F32, as the reference does.
//!
//! Stages: split residual VQ (1 + 15 codebooks) -> causal conv -> an 8-layer
//! transformer with 72-frame sliding-window attention -> two ConvNeXt 2x
//! upsamplers -> four SnakeBeta / transposed-conv blocks (8, 5, 4, 3x) -> a
//! final causal conv, clamped to [-1, 1].
//!
//! Every stage is causal (left-padded convolutions, transposed convolutions
//! trimmed on the right, causal attention), so a frame's audio depends only on
//! that frame and the ones before it. [`CodecStream`] relies on that: it
//! carries each stage's past from chunk to chunk and decodes only the new
//! frames, with the same result as decoding everything at once.
use crate::weights::{load_prefix_f32, tensor_bytes, TensorSource, Weights};
use candle_core::{DType, Device, Result, Tensor, D};
use oaiy_engine::json::Json;
use std::collections::HashMap;
use std::path::Path;

/// Samples per codec frame (12.5 frames per second at 24 kHz).
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const SAMPLE_RATE: usize = 24_000;
/// The official decoder works on 300-frame chunks with 25 frames of left
/// context; longer inputs are decoded the same way.
pub const CHUNK: usize = 300;
pub const LEFT_CONTEXT: usize = 25;

/// The decoder's shape, from `decoder_config` in the speech tokenizer's
/// `config.json`.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    pub quantizers: usize,
    pub layers: usize,
    pub heads: usize,
    pub window: usize,
    pub rope_theta: f64,
    pub upsample_rates: Vec<usize>,
}

impl CodecConfig {
    pub fn from_json(config: &Json) -> Result<Self> {
        let dc = config.get("decoder_config").ok_or_else(|| msg("speech tokenizer config lacks decoder_config"))?;
        let int = |k: &str| dc.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| msg(format!("decoder_config lacks {k}")));
        let upsample_rates = dc
            .get("upsample_rates")
            .and_then(Json::as_array)
            .ok_or_else(|| msg("decoder_config lacks upsample_rates"))?
            .iter()
            .filter_map(|v| v.as_i64().map(|v| v as usize))
            .collect::<Vec<_>>();
        Ok(Self {
            quantizers: int("num_quantizers")?,
            layers: int("num_hidden_layers")?,
            heads: int("num_attention_heads")?,
            window: int("sliding_window")?,
            rope_theta: dc.get("rope_theta").and_then(Json::as_f64).unwrap_or(10000.),
            upsample_rates,
        })
    }
}

pub struct CodecDecoder {
    w: HashMap<String, Tensor>,
    /// Codebook vectors (2048 x 256) for each of the 16 quantizers.
    codebooks: Vec<Tensor>,
    cfg: CodecConfig,
    /// The transformer's RoPE tables by length, and its whole-sequence masks
    /// by length, for short lengths (a streamed chunk's lengths repeat).
    tables: std::sync::Mutex<HashMap<(usize, bool), (Tensor, Tensor)>>,
    /// Stream states after a voice's reference clip, by its codes (a clip is
    /// worked through once, not every line).
    primed: std::sync::Mutex<HashMap<u64, StreamState>>,
    dev: Device,
}

/// Lengths at most this long keep their positional tables.
const SHORT: usize = 128;
/// Reference clips whose primed states are kept.
const PRIMED_VOICES: usize = 16;

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// What a streamed decode carries from one chunk to the next: each causal
/// convolution's last inputs and the transformer's keys and values over its
/// window, in the order the stages meet them. Made as they are first needed.
struct Cursor<'a> {
    states: &'a mut Vec<Tensor>,
    at: usize,
}

impl Cursor<'_> {
    /// The next state, made as zeros of `shape` (the zero padding a whole
    /// decode starts with) the first time.
    fn next(&mut self, shape: &[usize], dev: &Device) -> Result<Tensor> {
        if self.at == self.states.len() {
            self.states.push(Tensor::zeros(shape, DType::F32, dev)?);
        }
        let s = self.states[self.at].clone();
        if s.dims() != shape {
            candle_core::bail!("codec stream state {} is {:?}, not {shape:?}", self.at, s.dims());
        }
        self.at += 1;
        Ok(s)
    }
}

impl CodecDecoder {
    /// Load `decoder.*` from the speech tokenizer's `model.safetensors`
    /// (the encoder half is only needed to clone a voice from audio).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let cfg = path.parent().map(|p| p.join("config.json")).ok_or_else(|| msg("speech tokenizer path has no folder"))?;
        let config = Json::parse(&std::fs::read(&cfg).map_err(|e| msg(format!("{}: {e}", cfg.display())))?).map_err(candle_core::Error::wrap)?;
        Self::from_source(&mut Weights::open(path)?, CodecConfig::from_json(&config)?, dev)
    }

    pub fn from_source(source: &mut (impl TensorSource + ?Sized), cfg: CodecConfig, dev: &Device) -> Result<Self> {
        crate::weights::untracked(dev);
        Self::from_tensors(load_prefix_f32(source, "decoder.", dev)?, cfg, dev)
    }

    /// From F32 tensors named as in the checkpoint (`decoder.*`).
    pub fn from_tensors(w: HashMap<String, Tensor>, cfg: CodecConfig, dev: &Device) -> Result<Self> {
        let codebook = |prefix: &str, i: usize| -> Result<Tensor> {
            let base = format!("decoder.quantizer.{prefix}.vq.layers.{i}._codebook");
            let sum = w.get(&format!("{base}.embedding_sum")).ok_or_else(|| msg(format!("missing {base}")))?;
            let usage = w.get(&format!("{base}.cluster_usage")).ok_or_else(|| msg(format!("missing {base}")))?;
            sum.broadcast_div(&usage.clamp(1e-5f32, f32::MAX)?.unsqueeze(1)?)
        };
        let mut codebooks = vec![codebook("rvq_first", 0)?];
        for i in 0..cfg.quantizers - 1 {
            codebooks.push(codebook("rvq_rest", i)?);
        }
        Ok(Self {
            w,
            codebooks,
            cfg,
            tables: Default::default(),
            primed: Default::default(),
            dev: dev.clone(),
        })
    }

    /// Device bytes held.
    pub fn bytes(&self) -> u64 {
        tensor_bytes(self.w.values().chain(&self.codebooks))
    }

    pub fn device(&self) -> &Device {
        &self.dev
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
        self.forward_tensor(frames)?.flatten_all()?.to_vec1::<f32>()
    }

    /// One full-sequence decode, left on the device: (1, 1, samples).
    pub fn forward_tensor(&self, frames: &[[u32; 16]]) -> Result<Tensor> {
        self.run(&self.ids(frames)?, None)
    }

    /// The frames' codes as one (16, T) tensor on the device (one upload).
    fn ids(&self, frames: &[[u32; 16]]) -> Result<Tensor> {
        let t = frames.len();
        let flat: Vec<u32> = (0..16).flat_map(|q| frames.iter().map(move |f| f[q])).collect();
        Tensor::from_vec(flat, (16, t), &self.dev)
    }

    /// Decode (16, T) codes: (1, 1, T * 1920) samples. Streaming (`stream`:
    /// the carried state and the attention mask over the kept keys and the
    /// new ones), each stage starts from what the chunks before left it
    /// rather than from zeros, and the state moves on in place.
    fn run(&self, ids: &Tensor, stream: Option<(&mut Vec<Tensor>, &Tensor)>) -> Result<Tensor> {
        let (mut cursor, mask) = match stream {
            Some((states, mask)) => (Some(Cursor { states, at: 0 }), Some(mask)),
            None => (None, None),
        };
        let h = self.quantized(ids)?;
        let h = self.causal_conv(&h, "decoder.pre_conv.conv", 1, &mut cursor)?;
        let h = self.transformer(&h.transpose(1, 2)?.contiguous()?, &mut cursor, mask)?;
        let h = self.upsample(&h.transpose(1, 2)?.contiguous()?, &mut cursor)?;
        self.vocode(&h, &mut cursor)
    }

    /// Sum of the 16 codebooks' vectors, each group through its output
    /// projection: (1, 512, T). `codes`: (16, T).
    fn quantized(&self, codes: &Tensor) -> Result<Tensor> {
        let ids = |q: usize| codes.get(q);
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

    /// `x` (1, C, T) behind `ctx` samples of past: zeros for a whole decode,
    /// the carried inputs when streaming (which then become the last `ctx` of
    /// the result's input).
    fn with_past(&self, x: &Tensor, ctx: usize, cursor: &mut Option<Cursor>) -> Result<Tensor> {
        if ctx == 0 {
            return Ok(x.clone());
        }
        match cursor {
            None => x.pad_with_zeros(2, ctx, 0),
            Some(c) => {
                let past = c.next(&[1, x.dim(1)?, ctx], &self.dev)?;
                let full = Tensor::cat(&[&past, x], 2)?;
                let len = full.dim(2)?;
                past.slice_set(&full.narrow(2, len - ctx, ctx)?.contiguous()?, 2, 0)?;
                Ok(full)
            }
        }
    }

    /// A stride-1 causal convolution: the past is `(k-1)*d` samples.
    fn causal_conv(&self, x: &Tensor, prefix: &str, dilation: usize, cursor: &mut Option<Cursor>) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let k = w.dim(2)?;
        let y = self.with_past(x, (k - 1) * dilation, cursor)?.conv1d(w, 0, 1, dilation, 1)?;
        match self.w.get(&format!("{prefix}.bias")) {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }

    /// Transposed convolution trimmed on the right by `kernel - stride`. Each
    /// output depends on `(k-1)/stride` inputs before its own, which a stream
    /// carries.
    fn causal_trans_conv(&self, x: &Tensor, prefix: &str, stride: usize, cursor: &mut Option<Cursor>) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let b = self.get(&format!("{prefix}.bias"))?;
        let k = w.dim(2)?;
        let n = x.dim(2)?;
        let before = if cursor.is_some() { (k - 1) / stride } else { 0 };
        let y = self.with_past(x, before, cursor)?.conv_transpose1d(w, 0, 0, stride, 1, 1)?;
        // Outputs for the new inputs only: after the carried ones', and
        // without the tail that needs inputs yet to come.
        y.narrow(2, before * stride, n * stride)?.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)
    }

    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let y = crate::model::matmul_t(x, self.get(&format!("{prefix}.weight"))?)?;
        match self.w.get(&format!("{prefix}.bias")) {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn rms(&self, x: &Tensor, weight: &str, eps: f64) -> Result<Tensor> {
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(self.get(weight)?)
    }

    fn head_dim(&self) -> Result<usize> {
        Ok(self.get("decoder.pre_transformer.layers.0.self_attn.q_proj.weight")?.dim(0)? / self.cfg.heads)
    }

    /// input_proj, 8 layers of sliding-window attention and MLP with layer
    /// scales, final norm, output_proj. `x`: (1, T, 1024). Streaming, each
    /// layer's keys and values for the `window - 1` frames before are carried
    /// (before RoPE: positions count from the oldest kept key, which leaves
    /// every query-key distance, all attention sees, as in a whole decode).
    fn transformer(&self, x: &Tensor, cursor: &mut Option<Cursor>, stream_mask: Option<&Tensor>) -> Result<Tensor> {
        let p = "decoder.pre_transformer";
        let t = x.dim(1)?;
        let heads = self.cfg.heads;
        let window = self.cfg.window;
        let head_dim = self.head_dim()?;
        let past = if cursor.is_some() { window - 1 } else { 0 };
        let (cos, sin) = self.rope_tables(past + t, head_dim)?;
        let (q_cos, q_sin) = (cos.narrow(0, past, t)?, sin.narrow(0, past, t)?);
        let mask = match stream_mask {
            Some(m) => m.clone(),
            None => self.causal_mask(t)?,
        };
        let mut h = self.linear(x, &format!("{p}.input_proj"))?;
        let scale = 1. / (head_dim as f64).sqrt();
        for i in 0..self.cfg.layers {
            let l = format!("{p}.layers.{i}");
            let n = self.rms(&h, &format!("{l}.input_layernorm.weight"), 1e-5)?;
            let split = |y: Tensor| y.reshape((1, t, heads, head_dim));
            let q = split(self.linear(&n, &format!("{l}.self_attn.q_proj"))?)?;
            let k = split(self.linear(&n, &format!("{l}.self_attn.k_proj"))?)?;
            let v = split(self.linear(&n, &format!("{l}.self_attn.v_proj"))?)?;
            let (k, v) = match cursor {
                None => (k, v),
                Some(c) => {
                    let (kept_k, kept_v) = (c.next(&[1, past, heads, head_dim], &self.dev)?, c.next(&[1, past, heads, head_dim], &self.dev)?);
                    let (k, v) = (Tensor::cat(&[&kept_k, &k], 1)?, Tensor::cat(&[&kept_v, &v], 1)?);
                    kept_k.slice_set(&k.narrow(1, t, past)?, 1, 0)?;
                    kept_v.slice_set(&v.narrow(1, t, past)?, 1, 0)?;
                    (k, v)
                }
            };
            let heads_first = |y: Tensor| y.transpose(1, 2)?.contiguous();
            let q = candle_nn::rotary_emb::rope(&heads_first(q)?, &q_cos, &q_sin)?;
            let k = candle_nn::rotary_emb::rope(&heads_first(k)?, &cos, &sin)?;
            let v = heads_first(v)?;
            let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(&mask)?;
            let a = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
            let a = a.transpose(1, 2)?.contiguous()?.reshape((1, t, heads * head_dim))?;
            let a = self.linear(&a, &format!("{l}.self_attn.o_proj"))?;
            h = (h + a.broadcast_mul(self.get(&format!("{l}.self_attn_layer_scale.scale"))?)?)?;
            let n = self.rms(&h, &format!("{l}.post_attention_layernorm.weight"), 1e-5)?;
            let g = self.linear(&n, &format!("{l}.mlp.gate_proj"))?.silu()?;
            let m = self.linear(&(g * self.linear(&n, &format!("{l}.mlp.up_proj"))?)?, &format!("{l}.mlp.down_proj"))?;
            h = (h + m.broadcast_mul(self.get(&format!("{l}.mlp_layer_scale.scale"))?)?)?;
        }
        let h = self.rms(&h, &format!("{p}.norm.weight"), 1e-5)?;
        self.linear(&h, &format!("{p}.output_proj"))
    }

    /// Rotate-half RoPE tables for positions 0..len; kept for short lengths.
    fn rope_tables(&self, len: usize, head_dim: usize) -> Result<(Tensor, Tensor)> {
        self.table((len, true), || {
            let inv: Vec<f32> = (0..head_dim / 2).map(|i| 1. / self.cfg.rope_theta.powf(2. * i as f64 / head_dim as f64) as f32).collect();
            let freqs: Vec<f32> = (0..len).flat_map(|pos| inv.iter().map(move |f| pos as f32 * f)).collect();
            let freqs = Tensor::from_vec(freqs, (len, head_dim / 2), &self.dev)?;
            Ok((freqs.cos()?, freqs.sin()?))
        })
    }

    /// A whole decode's attention mask: causal, at most `window` keys back
    /// (the current one included).
    fn causal_mask(&self, t: usize) -> Result<Tensor> {
        let window = self.cfg.window;
        Ok(self
            .table((t, false), || {
                let mask: Vec<f32> = (0..t).flat_map(|q| (0..t).map(move |k| if k <= q && k + window > q { 0. } else { f32::NEG_INFINITY })).collect();
                let m = Tensor::from_vec(mask, (t, t), &self.dev)?;
                Ok((m.clone(), m))
            })?
            .0)
    }

    fn table(&self, key: (usize, bool), make: impl FnOnce() -> Result<(Tensor, Tensor)>) -> Result<(Tensor, Tensor)> {
        let short = key.0 <= SHORT;
        if short {
            if let Some(t) = self.tables.lock().map_err(|_| msg("codec tables poisoned by a panic"))?.get(&key) {
                return Ok(t.clone());
            }
        }
        let t = make()?;
        if short {
            self.tables.lock().map_err(|_| msg("codec tables poisoned by a panic"))?.insert(key, t.clone());
        }
        Ok(t)
    }

    /// A streamed chunk's attention mask, (n, window - 1 + n): query `i`
    /// (at kept position `window - 1 + i`) sees itself, the new frames before
    /// it, and the kept ones within the window that are real (`history` of
    /// them; the rest are the zeros a stream starts with).
    fn stream_mask(&self, n: usize, history: usize) -> Result<Tensor> {
        let past = self.cfg.window - 1;
        let first_real = past - history.min(past);
        let mask: Vec<f32> = (0..n).flat_map(|i| (0..past + n).map(move |j| if j >= i.max(first_real) && j <= past + i { 0. } else { f32::NEG_INFINITY })).collect();
        Tensor::from_vec(mask, (n, past + n), &self.dev)
    }

    /// Two stages of a 2x transposed conv and a ConvNeXt block. (1, C, T).
    fn upsample(&self, x: &Tensor, cursor: &mut Option<Cursor>) -> Result<Tensor> {
        let mut h = x.clone();
        let mut i = 0;
        while self.w.contains_key(&format!("decoder.upsample.{i}.0.conv.weight")) {
            let p = format!("decoder.upsample.{i}");
            let w = self.get(&format!("{p}.0.conv.weight"))?;
            h = self.causal_trans_conv(&h, &format!("{p}.0.conv"), w.dim(2)?, cursor)?;
            h = self.convnext(&h, &format!("{p}.1"), cursor)?;
            i += 1;
        }
        Ok(h)
    }

    /// Depthwise causal conv k7, LayerNorm, pointwise MLP with exact GELU,
    /// times gamma, plus the input.
    fn convnext(&self, x: &Tensor, p: &str, cursor: &mut Option<Cursor>) -> Result<Tensor> {
        let w = self.get(&format!("{p}.dwconv.conv.weight"))?; // (C, 1, k)
        let b = self.get(&format!("{p}.dwconv.conv.bias"))?;
        let (c, _, k) = w.dims3()?;
        let t = x.dim(2)?;
        // Depthwise as k shifted multiply-adds (a grouped conv would run one
        // convolution per channel).
        let padded = self.with_past(x, k - 1, cursor)?;
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

    fn vocode(&self, x: &Tensor, cursor: &mut Option<Cursor>) -> Result<Tensor> {
        let p = "decoder.decoder";
        let mut h = self.causal_conv(x, &format!("{p}.0.conv"), 1, cursor)?;
        for (i, &rate) in self.cfg.upsample_rates.iter().enumerate() {
            let b = format!("{p}.{}.block", i + 1);
            h = self.snake(&h, &format!("{b}.0"))?;
            h = self.causal_trans_conv(&h, &format!("{b}.1.conv"), rate, cursor)?;
            for (j, d) in [1, 3, 9].into_iter().enumerate() {
                let u = format!("{b}.{}", j + 2);
                let r = self.snake(&h, &format!("{u}.act1"))?;
                let r = self.causal_conv(&r, &format!("{u}.conv1.conv"), d, cursor)?;
                let r = self.snake(&r, &format!("{u}.act2"))?;
                let r = self.causal_conv(&r, &format!("{u}.conv2.conv"), 1, cursor)?;
                h = (h + r)?;
            }
        }
        let n = self.cfg.upsample_rates.len();
        h = self.snake(&h, &format!("{p}.{}", n + 1))?;
        h = self.causal_conv(&h, &format!("{p}.{}.conv", n + 2), 1, cursor)?;
        h.clamp(-1f32, 1f32)?.to_dtype(DType::F32)
    }

    /// A stream's step: `frames` decoded from `state`, which moves on. On
    /// CUDA a chunk length is captured once as a graph (a step is hundreds
    /// of small kernels, whose launches cost more than their work), with the
    /// state copied in and out around each replay.
    fn step(&self, state: &mut StreamState, frames: &[[u32; 16]]) -> Result<Tensor> {
        let n = frames.len();
        let ids = self.ids(frames)?;
        let mask = self.stream_mask(n, state.history)?;
        let past = self.cfg.window - 1;
        let out = self.run(&ids, Some((&mut state.tensors, &mask)))?;
        state.history = (state.history + n).min(past);
        Ok(out)
    }

    /// Work out (and keep) the stream state after `frames`, a voice's
    /// reference clip, so that its first line starts at once.
    pub fn prime(&self, frames: &[[u32; 16]]) -> Result<()> {
        if !frames.is_empty() {
            self.primed_state(frames)?;
        }
        Ok(())
    }

    /// The state a stream has after `frames` (a voice's reference clip),
    /// worked out once per clip and kept.
    fn primed_state(&self, frames: &[[u32; 16]]) -> Result<StreamState> {
        let key = frames.iter().flatten().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| (h ^ c as u64).wrapping_mul(0x0000_0100_0000_01b3));
        let key = key ^ frames.len() as u64;
        if let Some(s) = self.primed.lock().map_err(|_| msg("primed states poisoned by a panic"))?.get(&key) {
            return s.deep_copy();
        }
        // In pieces, so a long clip never needs a whole-clip decode's memory.
        let mut state = StreamState { tensors: Vec::new(), history: 0 };
        for piece in frames.chunks(PRIME_PIECE) {
            let mask = self.stream_mask(piece.len(), state.history)?;
            self.run(&self.ids(piece)?, Some((&mut state.tensors, &mask)))?;
            state.history = (state.history + piece.len()).min(self.cfg.window - 1);
        }
        let mut primed = self.primed.lock().map_err(|_| msg("primed states poisoned by a panic"))?;
        if primed.len() >= PRIMED_VOICES {
            primed.clear();
        }
        primed.insert(key, state.deep_copy()?);
        Ok(state)
    }
}

/// Frames decoded at a time while priming.
const PRIME_PIECE: usize = 32;

/// A stream's carried state (see [`CodecStream`]).
struct StreamState {
    tensors: Vec<Tensor>,
    /// Real frames behind the transformer's kept keys (at most window - 1).
    history: usize,
}

impl StreamState {
    fn deep_copy(&self) -> Result<Self> {
        Ok(Self { tensors: self.tensors.iter().map(Tensor::copy).collect::<Result<_>>()?, history: self.history })
    }
}

/// Frames in, audio out, as they come. Every stage of the decoder is causal
/// (left-padded convolutions, transposed convolutions trimmed on the right,
/// causal sliding-window attention), so the stream carries what each stage
/// needs from the frames before (its last inputs; the transformer's keys and
/// values over its window) and decodes each chunk from its own frames only:
/// the audio is the whole-sequence decode's, chunk by chunk, at a chunk's cost.
pub struct CodecStream {
    state: StreamState,
}

impl CodecStream {
    pub fn new() -> Self {
        Self { state: StreamState { tensors: Vec::new(), history: 0 } }
    }

    /// Start after `frames` (a cloned voice's reference clip, which the
    /// reference decodes ahead of the new speech and then cuts off): they are
    /// decoded as context only (once per clip; the decoder keeps the result).
    pub fn primed(codec: &CodecDecoder, frames: &[[u32; 16]]) -> Result<Self> {
        if frames.is_empty() {
            return Ok(Self::new());
        }
        Ok(Self { state: codec.primed_state(frames)? })
    }

    /// The samples of `frames` (1920 each), on the host.
    pub fn push(&mut self, codec: &CodecDecoder, frames: &[[u32; 16]]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        codec.step(&mut self.state, frames)?.flatten_all()?.to_vec1::<f32>()
    }
}

impl Default for CodecStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny random codec with the real one's structure (smaller widths and
    /// fewer layers), for checks that need no weights.
    pub(crate) fn tiny_codec(dev: &Device, seed: u64) -> Result<CodecDecoder> {
        let mut rng = crate::sampling::Rng::new(seed);
        let mut w = HashMap::new();
        let mut put = |name: String, shape: &[usize], scale: f32, offset: f32| -> Result<()> {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| offset + scale * (rng.uniform() as f32 * 2. - 1.)).collect();
            w.insert(name, Tensor::from_vec(v, shape, dev)?);
            Ok(())
        };
        let (cb, cbdim, latent, hidden, heads, hd, decoder_dim) = (32usize, 8usize, 12usize, 16usize, 2usize, 8usize, 32usize);
        let q = 4usize;
        for (prefix, n) in [("rvq_first", 1), ("rvq_rest", q - 1)] {
            for i in 0..n {
                let base = format!("decoder.quantizer.{prefix}.vq.layers.{i}._codebook");
                put(format!("{base}.embedding_sum"), &[cb, cbdim], 1., 0.)?;
                put(format!("{base}.cluster_usage"), &[cb], 0.5, 1.)?;
            }
            put(format!("decoder.quantizer.{prefix}.output_proj.weight"), &[latent, cbdim, 1], 0.3, 0.)?;
        }
        put("decoder.pre_conv.conv.weight".into(), &[latent, latent, 3], 0.2, 0.)?;
        put("decoder.pre_conv.conv.bias".into(), &[latent], 0.1, 0.)?;
        let p = "decoder.pre_transformer";
        put(format!("{p}.input_proj.weight"), &[hidden, latent], 0.3, 0.)?;
        put(format!("{p}.input_proj.bias"), &[hidden], 0.1, 0.)?;
        for l in 0..2 {
            let l = format!("{p}.layers.{l}");
            for n in ["q_proj", "k_proj", "v_proj"] {
                put(format!("{l}.self_attn.{n}.weight"), &[heads * hd, hidden], 0.3, 0.)?;
            }
            put(format!("{l}.self_attn.o_proj.weight"), &[hidden, heads * hd], 0.3, 0.)?;
            put(format!("{l}.self_attn_layer_scale.scale"), &[hidden], 0.1, 0.5)?;
            put(format!("{l}.mlp_layer_scale.scale"), &[hidden], 0.1, 0.5)?;
            put(format!("{l}.input_layernorm.weight"), &[hidden], 0.1, 1.)?;
            put(format!("{l}.post_attention_layernorm.weight"), &[hidden], 0.1, 1.)?;
            put(format!("{l}.mlp.gate_proj.weight"), &[2 * hidden, hidden], 0.3, 0.)?;
            put(format!("{l}.mlp.up_proj.weight"), &[2 * hidden, hidden], 0.3, 0.)?;
            put(format!("{l}.mlp.down_proj.weight"), &[hidden, 2 * hidden], 0.3, 0.)?;
        }
        put(format!("{p}.norm.weight"), &[hidden], 0.1, 1.)?;
        put(format!("{p}.output_proj.weight"), &[latent, hidden], 0.3, 0.)?;
        put(format!("{p}.output_proj.bias"), &[latent], 0.1, 0.)?;
        for i in 0..2 {
            let p = format!("decoder.upsample.{i}");
            put(format!("{p}.0.conv.weight"), &[latent, latent, 2], 0.3, 0.)?;
            put(format!("{p}.0.conv.bias"), &[latent], 0.1, 0.)?;
            put(format!("{p}.1.dwconv.conv.weight"), &[latent, 1, 7], 0.3, 0.)?;
            put(format!("{p}.1.dwconv.conv.bias"), &[latent], 0.1, 0.)?;
            put(format!("{p}.1.norm.weight"), &[latent], 0.1, 1.)?;
            put(format!("{p}.1.norm.bias"), &[latent], 0.1, 0.)?;
            put(format!("{p}.1.pwconv1.weight"), &[2 * latent, latent], 0.3, 0.)?;
            put(format!("{p}.1.pwconv1.bias"), &[2 * latent], 0.1, 0.)?;
            put(format!("{p}.1.pwconv2.weight"), &[latent, 2 * latent], 0.3, 0.)?;
            put(format!("{p}.1.pwconv2.bias"), &[latent], 0.1, 0.)?;
            put(format!("{p}.1.gamma"), &[latent], 0.1, 0.3)?;
        }
        // The real rates multiply to 480 (with the 4x above, 1920 a frame).
        let rates = [8usize, 5, 4, 3];
        let p = "decoder.decoder";
        put(format!("{p}.0.conv.weight"), &[decoder_dim, latent, 7], 0.2, 0.)?;
        put(format!("{p}.0.conv.bias"), &[decoder_dim], 0.1, 0.)?;
        let mut ch = decoder_dim;
        for (i, &rate) in rates.iter().enumerate() {
            let b = format!("{p}.{}.block", i + 1);
            let out = ch / 2;
            put(format!("{b}.0.alpha"), &[ch], 0.1, 0.)?;
            put(format!("{b}.0.beta"), &[ch], 0.1, 0.)?;
            put(format!("{b}.1.conv.weight"), &[ch, out, 2 * rate], 0.2, 0.)?;
            put(format!("{b}.1.conv.bias"), &[out], 0.1, 0.)?;
            for j in 0..3 {
                let u = format!("{b}.{}", j + 2);
                put(format!("{u}.act1.alpha"), &[out], 0.1, 0.)?;
                put(format!("{u}.act1.beta"), &[out], 0.1, 0.)?;
                put(format!("{u}.conv1.conv.weight"), &[out, out, 7], 0.2, 0.)?;
                put(format!("{u}.conv1.conv.bias"), &[out], 0.1, 0.)?;
                put(format!("{u}.act2.alpha"), &[out], 0.1, 0.)?;
                put(format!("{u}.act2.beta"), &[out], 0.1, 0.)?;
                put(format!("{u}.conv2.conv.weight"), &[out, out, 1], 0.2, 0.)?;
                put(format!("{u}.conv2.conv.bias"), &[out], 0.1, 0.)?;
            }
            ch = out;
        }
        put(format!("{p}.5.alpha"), &[ch], 0.1, 0.)?;
        put(format!("{p}.5.beta"), &[ch], 0.1, 0.)?;
        put(format!("{p}.6.conv.weight"), &[1, ch, 7], 0.2, 0.)?;
        put(format!("{p}.6.conv.bias"), &[1], 0.01, 0.)?;
        let cfg = CodecConfig { quantizers: q, layers: 2, heads, window: 6, rope_theta: 10000., upsample_rates: rates.to_vec() };
        CodecDecoder::from_tensors(w, cfg, dev)
    }

    pub(crate) fn random_frames(n: usize, seed: u64) -> Vec<[u32; 16]> {
        let mut rng = crate::sampling::Rng::new(seed);
        // The tiny codec has 4 codebooks of 32 entries; the rest are unused.
        (0..n).map(|_| std::array::from_fn(|q| if q < 4 { (rng.uniform() * 32.) as u32 % 32 } else { 0 })).collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0., f32::max)
    }

    #[test]
    fn a_stream_decodes_exactly_what_the_whole_sequence_decodes() -> Result<()> {
        let dev = Device::Cpu;
        let codec = tiny_codec(&dev, 1)?;
        // 20 frames: well past the tiny transformer's 6-frame window.
        let frames = random_frames(20, 2);
        let whole = codec.forward(&frames)?;
        assert_eq!(whole.len(), 20 * SAMPLES_PER_FRAME);
        let mut stream = CodecStream::new();
        let mut streamed = Vec::new();
        for chunk in [1usize, 1, 2, 3, 1, 5, 7] {
            let start = streamed.len() / SAMPLES_PER_FRAME;
            streamed.extend(stream.push(&codec, &frames[start..start + chunk])?);
        }
        let d = max_diff(&streamed, &whole);
        assert!(d < 1e-4, "streamed vs whole: {d}");
        // A prefix decodes to a prefix of the whole.
        let prefix = codec.forward(&frames[..5])?;
        assert!(max_diff(&prefix, &whole[..prefix.len()]) < 1e-4);
        Ok(())
    }

    #[test]
    fn a_primed_stream_decodes_as_the_reference_and_the_new_frames_together() -> Result<()> {
        let dev = Device::Cpu;
        let codec = tiny_codec(&dev, 3)?;
        // Longer than a priming piece, so priming itself is streamed.
        let reference = random_frames(PRIME_PIECE + 7, 4);
        let new = random_frames(6, 5);
        let mut all = reference.clone();
        all.extend_from_slice(&new);
        let whole = codec.forward(&all)?;
        let expected = &whole[reference.len() * SAMPLES_PER_FRAME..];
        for _ in 0..2 {
            // The second time from the kept state.
            let mut stream = CodecStream::primed(&codec, &reference)?;
            let mut out = stream.push(&codec, &new[..2])?;
            out.extend(stream.push(&codec, &new[2..])?);
            let d = max_diff(&out, expected);
            assert!(d < 1e-4, "{d}");
        }
        Ok(())
    }

    fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let error = (actual - expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        Ok(error / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
    }

    #[test]
    #[ignore = "requires the Qwen3-TTS speech tokenizer and reference stages; OAIY_TTS_GOLDEN, OAIY_TTS_MODEL"]
    fn codec_decoder_matches_reference() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("OAIY_TTS_GOLDEN").map_err(candle_core::Error::wrap)?);
        let model = std::path::PathBuf::from(std::env::var("OAIY_TTS_MODEL").map_err(candle_core::Error::wrap)?);
        let dev = Device::new_cuda(std::env::var("OAIY_TTS_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> { Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev) };
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
        let q = dec.quantized(&dec.ids(&frames)?)?;
        check("dec_quantized.f32", &q.squeeze(0)?, &[512, t], 1e-5)?;
        let pc = dec.causal_conv(&q, "decoder.pre_conv.conv", 1, &mut None)?;
        check("dec_pre_conv.f32", &pc.squeeze(0)?, &[1024, t], 1e-5)?;
        let pt = dec.transformer(&pc.transpose(1, 2)?.contiguous()?, &mut None, None)?;
        check("dec_pre_transformer.f32", &pt.squeeze(0)?, &[t, 1024], 1e-5)?;
        let up = dec.upsample(&pt.transpose(1, 2)?.contiguous()?, &mut None)?;
        check("dec_upsample1.f32", &up.squeeze(0)?, &[1024, 4 * t], 1e-5)?;
        let wave = Tensor::new(dec.forward(&frames)?, &dev)?;
        check("dec_wave.f32", &wave, &[t * SAMPLES_PER_FRAME], 1e-5)?;
        assert!(worst.is_empty(), "{worst:?}");
        Ok(())
    }
}
