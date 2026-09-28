//! The autoregressive stage of MiniMax Music 3, after the official diffusers
//! pipeline (`MiniMaxMusic3SemanticGenerationStep`): an 8B Qwen3 language
//! model picks each 25 Hz frame's semantic code, a 4-layer depth decoder
//! picks its seven residual codes, and the frame's eight hidden states (one
//! from the language model, seven from the depth decoder) condition the
//! acoustic stage. Both run with classifier-free guidance, as a batch of two:
//! row 0 sees the prompt, row 1 the same prompt with its words masked.
//!
//! Only the vocabulary the stage can emit is ever touched: the output head
//! keeps the 16384 semantic-code rows and the end-of-audio row (not 200k),
//! and the embedding table is read row by row for the prompt.
use crate::ltx::store::Store;
use crate::residency::{Budget, Resident, Tiered};
use candle_core::{
    quantized::{gguf_file, QMatMul},
    DType, Device, Module, Result, Tensor, D,
};
use std::path::Path;

pub const AUDIO_END: u32 = 151_670;
pub const AUDIO_CFG: u32 = 151_654;
pub const CODE_OFFSET: u32 = 151_675;
pub const SEMANTIC_CODES: usize = 16_384;
pub const CODEBOOKS: usize = 8;
pub const AUDIO_VOCAB: usize = 1_024;
pub const HIDDEN: usize = 4_096;
/// Guidance scale and candidate count for both samplers.
pub const CFG: f32 = 1.5;
pub const TOP_K: usize = 50;

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// A projection: a dense BF16 matrix, or a quantized one.
pub enum Proj {
    Dense(Tensor),
    Quant(QMatMul),
}

impl Proj {
    /// Over the last axis, as one 2-D matmul (a batched one would read the
    /// weight once per batch row).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let n = *dims.last().ok_or_else(|| msg("projection input has no axes"))?;
        let flat = x.reshape((x.elem_count() / n, n))?;
        let y = match self {
            Self::Dense(w) => flat.matmul(&w.t()?)?,
            Self::Quant(q) => q.forward(&flat.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?,
        };
        let mut out = dims;
        *out.last_mut().unwrap() = y.dim(1)?;
        y.reshape(out)
    }
    fn bytes(&self) -> u64 {
        match self {
            Self::Dense(w) => (w.elem_count() * w.dtype().size_in_bytes()) as u64,
            Self::Quant(QMatMul::QTensor(q)) => q.storage_size_in_bytes() as u64,
            Self::Quant(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => (t.elem_count() * t.dtype().size_in_bytes()) as u64,
        }
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(match self {
            Self::Dense(w) => Self::Dense(w.to_device(dev)?),
            Self::Quant(QMatMul::QTensor(q)) => {
                let data = q.data()?;
                Self::Quant(QMatMul::from_qtensor(candle_core::quantized::ggml_file::qtensor_from_ggml(q.dtype(), &data, q.shape().dims().to_vec(), dev)?)?)
            }
            Self::Quant(QMatMul::Tensor(t)) => Self::Quant(QMatMul::Tensor(t.to_device(dev)?)),
            Self::Quant(QMatMul::TensorF16(t)) => Self::Quant(QMatMul::TensorF16(t.to_device(dev)?)),
        })
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, w, eps)
}

fn size(t: &Tensor) -> u64 {
    (t.elem_count() * t.dtype().size_in_bytes()) as u64
}

/// One Qwen3 decoder layer.
pub struct Layer {
    qkv: Proj,
    o: Proj,
    q_norm: Tensor,
    k_norm: Tensor,
    input_norm: Tensor,
    post_norm: Tensor,
    gate_up: Proj,
    down: Proj,
}

impl Resident for Layer {
    fn bytes(&self) -> u64 {
        self.qkv.bytes() + self.o.bytes() + self.gate_up.bytes() + self.down.bytes() + size(&self.q_norm) + size(&self.k_norm) + size(&self.input_norm) + size(&self.post_norm)
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            qkv: self.qkv.to_device(dev)?,
            o: self.o.to_device(dev)?,
            q_norm: self.q_norm.to_device(dev)?,
            k_norm: self.k_norm.to_device(dev)?,
            input_norm: self.input_norm.to_device(dev)?,
            post_norm: self.post_norm.to_device(dev)?,
            gate_up: self.gate_up.to_device(dev)?,
            down: self.down.to_device(dev)?,
        })
    }
}

/// Keys and values so far, per layer: (B, T, kv_heads, head_dim), written
/// in place into buffers sized for the whole song.
pub struct Cache {
    layers: Vec<candle_nn::kv_cache::KvCache>,
    pub len: usize,
}

impl Cache {
    pub fn new(layers: usize, max_len: usize) -> Self {
        Self { layers: (0..layers).map(|_| candle_nn::kv_cache::KvCache::new(1, max_len)).collect(), len: 0 }
    }
}

fn dense(store: &mut Store, keys: &[String], dev: &Device) -> Result<Proj> {
    let ws = keys.iter().map(|k| store.tensor(k, dev, false)).collect::<Result<Vec<_>>>()?;
    Ok(Proj::Dense(if ws.len() == 1 { ws.into_iter().next().unwrap() } else { Tensor::cat(&ws, 0)? }))
}

fn load_layer(store: &mut Store, i: usize, dev: &Device) -> Result<Layer> {
    let l = format!("model.layers.{i}");
    let k = |n: &str| format!("{l}.{n}.weight");
    Ok(Layer {
        qkv: dense(store, &[k("self_attn.q_proj"), k("self_attn.k_proj"), k("self_attn.v_proj")], dev)?,
        o: dense(store, &[k("self_attn.o_proj")], dev)?,
        q_norm: store.tensor(&k("self_attn.q_norm"), dev, false)?,
        k_norm: store.tensor(&k("self_attn.k_norm"), dev, false)?,
        input_norm: store.tensor(&k("input_layernorm"), dev, false)?,
        post_norm: store.tensor(&k("post_attention_layernorm"), dev, false)?,
        gate_up: dense(store, &[k("mlp.gate_proj"), k("mlp.up_proj")], dev)?,
        down: dense(store, &[k("mlp.down_proj")], dev)?,
    })
}

/// Where the language model's weights come from: the published safetensors
/// (BF16), or a quantized file made by [`crate::music::quant::convert`].
enum Source {
    Safe(Store),
    Gguf { content: gguf_file::Content, file: std::fs::File },
}

impl Source {
    fn layer(&mut self, i: usize, dev: &Device) -> Result<Layer> {
        match self {
            Self::Safe(store) => load_layer(store, i, dev),
            Self::Gguf { content, file } => {
                let mut proj = |n: &str| -> Result<Proj> { Ok(Proj::Quant(QMatMul::from_qtensor(content.tensor(file, &format!("layers.{i}.{n}"), dev)?)?)) };
                let (qkv, o, gate_up, down) = (proj("qkv")?, proj("o")?, proj("gate_up")?, proj("down")?);
                let mut norm = |n: &str| -> Result<Tensor> { content.tensor(file, &format!("layers.{i}.{n}"), dev)?.dequantize(dev)?.to_dtype(DType::BF16) };
                Ok(Layer { qkv, o, gate_up, down, q_norm: norm("q_norm")?, k_norm: norm("k_norm")?, input_norm: norm("input_norm")?, post_norm: norm("post_norm")? })
            }
        }
    }

    /// A whole small tensor in BF16.
    fn small(&mut self, safe_key: &str, gguf_key: &str, dev: &Device) -> Result<Tensor> {
        match self {
            Self::Safe(store) => store.tensor(safe_key, dev, false),
            Self::Gguf { content, file } => content.tensor(file, gguf_key, dev)?.dequantize(dev)?.to_dtype(DType::BF16),
        }
    }

    /// Embedding rows (BF16), read one by one: from `embed_tokens` in the
    /// safetensors, or `gguf_key` in the quantized file.
    fn rows(&mut self, gguf_key: &str, safe_key: &str, ids: &[u32], dev: &Device) -> Result<Tensor> {
        match self {
            Self::Safe(store) => store.rows(safe_key, ids, dev),
            Self::Gguf { content, file } => {
                use std::io::{Read, Seek, SeekFrom};
                let info = content.tensor_infos.get(gguf_key).ok_or_else(|| msg(format!("the quantized language model lacks {gguf_key}")))?;
                let (rows, width) = info.shape.dims2()?;
                let dtype = info.ggml_dtype;
                let row_bytes = width / dtype.block_size() * dtype.type_size();
                let mut bytes = vec![0u8; ids.len() * row_bytes];
                for (i, &id) in ids.iter().enumerate() {
                    if id as usize >= rows {
                        candle_core::bail!("row {id} is outside {gguf_key} in the quantized language model");
                    }
                    file.seek(SeekFrom::Start(content.tensor_data_offset + info.offset + id as u64 * row_bytes as u64))?;
                    file.read_exact(&mut bytes[i * row_bytes..(i + 1) * row_bytes])?;
                }
                let q = candle_core::quantized::ggml_file::qtensor_from_ggml(dtype, &bytes, vec![ids.len(), width], &Device::Cpu)?;
                q.dequantize(&Device::Cpu)?.to_dtype(DType::BF16)?.to_device(dev)
            }
        }
    }
}

/// The model's shape, from `config.json` or the quantized file's metadata.
struct Shape {
    count: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f64,
}

pub(crate) fn config_of(dir: &Path) -> Result<oaiy_engine::json::Json> {
    oaiy_engine::json::Json::parse(&std::fs::read(dir.join("config.json")).map_err(|e| msg(format!("{}: {e}", dir.display())))?).map_err(candle_core::Error::wrap)
}

fn shape_of_config(dir: &Path) -> Result<Shape> {
    let config = config_of(dir)?;
    let int = |k: &str| config.get(k).and_then(oaiy_engine::json::Json::as_i64).map(|v| v as usize).ok_or_else(|| msg(format!("language model config lacks {k}")));
    Ok(Shape {
        count: int("num_hidden_layers")?,
        heads: int("num_attention_heads")?,
        kv_heads: int("num_key_value_heads")?,
        head_dim: int("head_dim")?,
        eps: config.get("rms_norm_eps").and_then(oaiy_engine::json::Json::as_f64).unwrap_or(1e-6) as f32,
        theta: config.get("rope_parameters").and_then(|r| r.get("rope_theta")).or_else(|| config.get("rope_theta")).and_then(oaiy_engine::json::Json::as_f64).unwrap_or(1e6),
    })
}

/// Device bytes of the KV cache for `positions` (36 layers, keys and values,
/// two rows, 8 heads of 128, BF16).
pub fn kv_bytes(positions: usize) -> u64 {
    positions as u64 * 36 * 2 * 2 * 8 * 128 * 2
}

/// The global language model (Qwen3, 36 layers).
pub struct Lm {
    source: Source,
    layers: Tiered<Layer>,
    count: usize,
    norm: Tensor,
    /// Output rows the stage can emit: end of audio, then the 16384 codes.
    head: Tensor,
    /// Input rows of the 16384 semantic codes, for the frame feedback.
    codes: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    cos: Tensor,
    sin: Tensor,
    dev: Device,
}

impl Lm {
    /// `dir`: the `language_model` folder; `quantized`: a file made by
    /// [`crate::music::quant::convert`] to use instead of its safetensors.
    pub fn load(dir: &Path, quantized: Option<&Path>, budget: &Budget, dev: &Device, progress: impl FnMut(usize)) -> Result<Self> {
        let (mut source, shape) = match quantized {
            None => (Source::Safe(Store::open(dir, 0)?), shape_of_config(dir)?),
            Some(path) => {
                let mut file = std::fs::File::open(path).map_err(|e| msg(format!("{}: {e}", path.display())))?;
                let content = gguf_file::Content::read(&mut file).map_err(|e| msg(format!("{}: {e}", path.display())))?;
                let u = |k: &str| content.metadata.get(k).and_then(|v| v.to_u32().ok()).map(|v| v as usize).ok_or_else(|| msg(format!("{}: lacks {k}", path.display())));
                let f = |k: &str| content.metadata.get(k).and_then(|v| v.to_f32().ok()).ok_or_else(|| msg(format!("{}: lacks {k}", path.display())));
                let shape = Shape { count: u("music3.layers")?, heads: u("music3.heads")?, kv_heads: u("music3.kv_heads")?, head_dim: u("music3.head_dim")?, eps: f("music3.eps")?, theta: f("music3.rope_theta")? as f64 };
                (Source::Gguf { content, file }, shape)
            }
        };
        let code_rows: Vec<u32> = (CODE_OFFSET..CODE_OFFSET + SEMANTIC_CODES as u32).collect();
        let head_rows: Vec<u32> = std::iter::once(AUDIO_END).chain(code_rows.iter().copied()).collect();
        let (head, codes) = match &mut source {
            Source::Safe(store) => (store.rows("lm_head.weight", &head_rows, dev)?, store.rows("model.embed_tokens.weight", &code_rows, dev)?),
            gguf => (gguf.small("", "head", dev)?, gguf.small("", "codes", dev)?),
        };
        let norm = source.small("model.norm.weight", "norm", dev)?;
        let layers = Tiered::load(shape.count, budget, dev, |i| source.layer(i, dev), progress)?;
        // RoPE tables for the longest prompt and song, in F32 as the reference
        // computes them, then BF16.
        let (head_dim, positions) = (shape.head_dim, crate::music::MAX_PROMPT_TOKENS + crate::music::MAX_FRAMES + 2);
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1. / (shape.theta as f32).powf(2. * i as f32 / head_dim as f32)).collect();
        let freqs: Vec<f32> = (0..positions).flat_map(|p| inv.iter().map(move |f| p as f32 * f)).collect();
        let freqs = Tensor::from_vec(freqs, (positions, half), dev)?;
        Ok(Self {
            source,
            layers,
            count: shape.count,
            norm,
            head,
            codes,
            heads: shape.heads,
            kv_heads: shape.kv_heads,
            head_dim,
            eps: shape.eps,
            cos: freqs.cos()?.to_dtype(DType::BF16)?,
            sin: freqs.sin()?.to_dtype(DType::BF16)?,
            dev: dev.clone(),
        })
    }

    pub fn layers(&self) -> usize {
        self.count
    }
    pub fn report(&self) -> oaiy_engine::json::Json {
        self.layers.report()
    }

    /// Token embeddings, read from the file row by row: (B, T, 4096).
    pub fn embed(&mut self, ids: &[Vec<u32>]) -> Result<Tensor> {
        let rows = ids.iter().map(|r| self.source.rows("text", "model.embed_tokens.weight", r, &self.dev)).collect::<Result<Vec<_>>>()?;
        Tensor::stack(&rows, 0)
    }

    /// Embeddings of semantic codes (not offset): (n, 4096).
    pub fn code_embed(&self, codes: &Tensor) -> Result<Tensor> {
        self.codes.index_select(codes, 0)
    }

    /// Logits over [end of audio, code 0..16383] for (B, 4096) hidden states, F32.
    pub fn logits(&self, h: &Tensor) -> Result<Tensor> {
        h.matmul(&self.head.t()?)?.to_dtype(DType::F32)
    }

    /// `x`: (B, T, 4096) BF16 after `cache`; returns the last position's
    /// normed hidden state, (B, 4096).
    pub fn forward(&mut self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let start = cache.len;
        if start + t > self.cos.dim(0)? {
            candle_core::bail!("the song is longer than the language model's {} positions", self.cos.dim(0)?);
        }
        let cos = self.cos.narrow(0, start, t)?;
        let sin = self.sin.narrow(0, start, t)?;
        let (hd, nq, nkv, eps) = (self.head_dim, self.heads, self.kv_heads, self.eps);
        let dev = self.dev.clone();
        let source = &mut self.source;
        let mut h = x.clone();
        for i in 0..self.count {
            h = self.layers.with(
                i,
                |i| source.layer(i, &dev),
                |l| {
                    let n = rms(&h, &l.input_norm, eps)?;
                    let qkv = l.qkv.forward(&n)?;
                    let q = rms(&qkv.narrow(2, 0, nq * hd)?.reshape((b, t, nq, hd))?, &l.q_norm, eps)?;
                    let k = rms(&qkv.narrow(2, nq * hd, nkv * hd)?.reshape((b, t, nkv, hd))?, &l.k_norm, eps)?;
                    let v = qkv.narrow(2, (nq + nkv) * hd, nkv * hd)?.reshape((b, t, nkv, hd))?.contiguous()?;
                    let q = candle_nn::rotary_emb::rope_thd(&q, &cos, &sin)?;
                    let k = candle_nn::rotary_emb::rope_thd(&k, &cos, &sin)?;
                    let (k, v) = cache.layers[i].append(&k.contiguous()?, &v)?;
                    let a = causal_attention(&q, &k, &v, start)?.reshape((b, t, nq * hd))?;
                    let h = (&h + l.o.forward(&a)?)?;
                    let n = rms(&h, &l.post_norm, eps)?;
                    let gu = l.gate_up.forward(&n)?;
                    let inner = gu.dim(2)? / 2;
                    let m = (gu.narrow(2, 0, inner)?.silu()? * gu.narrow(2, inner, inner)?)?;
                    h + l.down.forward(&m)?
                },
            )?;
        }
        cache.len += t;
        rms(&h.narrow(1, t - 1, 1)?.squeeze(1)?, &self.norm, eps)
    }
}

/// Causal attention over (B, T, H, D) with grouped K/V heads; query `i`
/// sits at position `start + i`. Flash attention when the worker has it
/// (its causal mask is aligned to the last key: the KV-cache case).
fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, start: usize) -> Result<Tensor> {
    let hd = q.dim(3)?;
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() {
        return candle_flash_attn::flash_attn(q, k, v, 1. / (hd as f32).sqrt(), true);
    }
    let (b, nk, kv_heads, _) = k.dims4()?;
    let (_, nq, heads, _) = q.dims4()?;
    let groups = heads / kv_heads;
    let expand = |x: &Tensor| -> Result<Tensor> {
        x.transpose(1, 2)?.unsqueeze(2)?.expand((b, kv_heads, groups, nk, hd))?.reshape((b, heads, nk, hd))?.to_dtype(DType::F32)
    };
    let (kx, vx) = (expand(k)?, expand(v)?);
    let qx = q.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?;
    let mut scores = (qx.matmul(&kx.t()?)? / (hd as f64).sqrt())?;
    if nq > 1 {
        let mask: Vec<f32> = (0..nq).flat_map(|i| (0..nk).map(move |j| if j <= start + i { 0. } else { f32::NEG_INFINITY })).collect();
        scores = scores.broadcast_add(&Tensor::from_vec(mask, (nq, nk), q.device())?)?;
    }
    candle_nn::ops::softmax(&scores, D::Minus1)?.matmul(&vx)?.to_dtype(q.dtype())?.transpose(1, 2)?.contiguous()
}

struct DepthLayer {
    input_norm: Tensor,
    qkv: Proj,
    out: Proj,
    post_norm: Tensor,
    gate_up: Proj,
    down: Proj,
}

/// The depth decoder: within a frame, the seven residual codes one after
/// another (4 causal layers over at most 9 positions, learned positions).
pub struct Depth {
    audio: Tensor,
    projection: Proj,
    positions: Tensor,
    layers: Vec<DepthLayer>,
    norm: Tensor,
    heads_out: Vec<Proj>,
    heads: usize,
}

impl Depth {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let config = oaiy_engine::json::Json::parse(&std::fs::read(dir.join("config.json")).map_err(|e| msg(format!("{}: {e}", dir.display())))?).map_err(candle_core::Error::wrap)?;
        let int = |k: &str, d: usize| config.get(k).and_then(oaiy_engine::json::Json::as_i64).map_or(d, |v| v as usize);
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let mut layers = Vec::new();
        for i in 0..int("num_layers", 4) {
            let k = |n: &str| format!("layers.{i}.{n}.weight");
            layers.push(DepthLayer {
                input_norm: store.tensor(&k("input_layernorm"), dev, false)?,
                qkv: dense(&mut store, &[k("attn.to_q"), k("attn.to_k"), k("attn.to_v")], dev)?,
                out: dense(&mut store, &[k("attn.to_out")], dev)?,
                post_norm: store.tensor(&k("post_attention_layernorm"), dev, false)?,
                gate_up: dense(&mut store, &[k("gate_proj"), k("up_proj")], dev)?,
                down: dense(&mut store, &[k("down_proj")], dev)?,
            });
        }
        let heads_out = (0..CODEBOOKS - 1).map(|i| dense(&mut store, &[format!("audio_heads.{i}.weight")], dev)).collect::<Result<_>>()?;
        Ok(Self {
            audio: store.tensor("audio_embeddings.weight", dev, false)?,
            projection: dense(&mut store, &["projection.weight".into()], dev)?,
            positions: store.tensor("pos_embedding.weight", dev, false)?,
            layers,
            norm: store.tensor("norm.weight", dev, false)?,
            heads_out,
            heads: int("num_attention_heads", 16),
        })
    }

    /// Normed hidden states for `x` (B, S, 4096), already projected.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let (heads, hd) = (self.heads, d / self.heads);
        let mut h = x.broadcast_add(&self.positions.narrow(0, 0, s)?.unsqueeze(0)?)?;
        for l in &self.layers {
            let n = rms(&h, &l.input_norm, 1e-6)?;
            let qkv = l.qkv.forward(&n)?;
            let q = qkv.narrow(2, 0, d)?.reshape((b, s, heads, hd))?.contiguous()?;
            let k = qkv.narrow(2, d, d)?.reshape((b, s, heads, hd))?.contiguous()?;
            let v = qkv.narrow(2, 2 * d, d)?.reshape((b, s, heads, hd))?.contiguous()?;
            let a = causal_attention(&q, &k, &v, 0)?.reshape((b, s, d))?;
            h = (h + l.out.forward(&a)?)?;
            let n = rms(&h, &l.post_norm, 1e-6)?;
            let gu = l.gate_up.forward(&n)?;
            let inner = gu.dim(2)? / 2;
            h = (&h + l.down.forward(&(gu.narrow(2, 0, inner)?.silu()? * gu.narrow(2, inner, inner)?)?)?)?;
        }
        rms(&h, &self.norm, 1e-6)
    }

    /// Residual-code embeddings for codebook `index` (1..7): (n, 4096).
    fn audio_embed(&self, code: u32, index: usize, rows: usize) -> Result<Tensor> {
        let id = code + ((index - 1) * AUDIO_VOCAB) as u32;
        self.audio.index_select(&Tensor::from_vec(vec![id; rows], rows, self.audio.device())?, 0)
    }

    /// The feedback input for a finished frame: every codebook's embedding
    /// summed, scaled by 8^-1/2. `semantic`: (n, 4096) from the language model.
    pub fn frame_embedding(&self, semantic: &Tensor, residual: &[u32]) -> Result<Tensor> {
        let dev = semantic.device();
        let n = semantic.dim(0)?;
        let ids: Vec<u32> = residual.iter().enumerate().map(|(i, c)| c + (i * AUDIO_VOCAB) as u32).collect();
        let extra = self.audio.index_select(&Tensor::from_vec(ids, CODEBOOKS - 1, dev)?, 0)?.sum_keepdim(0)?;
        ((semantic + extra.broadcast_as((n, semantic.dim(1)?))?)? * (CODEBOOKS as f64).powf(-0.5))?.unsqueeze(1)
    }
}

/// splitmix64 with a Box-Muller normal, so a seed means the same song on
/// every device.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = self.uniform().max(f64::MIN_POSITIVE);
            let u2 = self.uniform();
            let r = (-2. * u1.ln()).sqrt();
            let a = 2. * std::f64::consts::PI * u2;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}

/// The `k`-th largest value.
fn kth_largest(v: &[f32], k: usize) -> f32 {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let k = k.min(s.len()).max(1);
    s.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    s[k - 1]
}

/// Top-50 sampling at temperature 1 (the reference's `_sample_top_k`), or the
/// arg max when `rng` is `None`. Masked entries are `-inf`.
pub fn sample_top_k(values: &[f32], rng: Option<&mut Rng>) -> usize {
    // nan/inf -> +-1e9, as the reference cleans them.
    let v: Vec<f32> = values.iter().map(|&x| if x.is_nan() { -1e9 } else { x.clamp(-1e9, 1e9) }).collect();
    let Some(rng) = rng else {
        return v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
    };
    let threshold = kth_largest(&v, TOP_K);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f64> = v.iter().map(|&x| if x < threshold { 0. } else { ((x - max) as f64).exp() }).collect();
    let total: f64 = p.iter().sum();
    let mut r = rng.uniform() * total;
    for (i, &w) in p.iter().enumerate() {
        if w > 0. {
            if r < w {
                return i;
            }
            r -= w;
        }
    }
    p.iter().rposition(|&w| w > 0.).unwrap_or(0)
}

/// Guidance between a conditional and an unconditional row.
fn guide(c: &[f32], u: &[f32]) -> Vec<f32> {
    c.iter().zip(u).map(|(c, u)| u + (c - u) * CFG).collect()
}

/// The next semantic step from (2, 16385) logits: `None` at the end of the
/// song, else the code. Guidance is limited to the conditional row's top 50.
pub fn pick_semantic(logits: &Tensor, rng: Option<&mut Rng>) -> Result<Option<u32>> {
    let rows = logits.to_vec2::<f32>()?;
    let (c, u) = (&rows[0], &rows[1]);
    let threshold = kth_largest(c, TOP_K);
    let guided: Vec<f32> = guide(c, u).into_iter().zip(c).map(|(g, &c)| if c < threshold { f32::NEG_INFINITY } else { g }).collect();
    let i = sample_top_k(&guided, rng);
    Ok(if i == 0 { None } else { Some((i - 1) as u32) })
}

/// Everything the stage keeps of one frame.
pub struct Frame {
    /// Semantic code, then the seven residual codes.
    pub codes: [u32; CODEBOOKS],
    /// The conditional row's hidden states: language model, then depth
    /// decoder steps 1..7, (1, 8 * 4096).
    pub hidden: Tensor,
}

/// The seven residual codes for a frame, the conditional row's depth hidden
/// states, and the frame's feedback embedding. `last`: the language model's
/// (2, 4096) state; `pick(index, guided logits)` chooses each code.
pub fn depth_codes(depth: &Depth, lm: &Lm, last: &Tensor, semantic: u32, pick: &mut dyn FnMut(usize, &[f32]) -> u32) -> Result<([u32; CODEBOOKS], Tensor, Tensor)> {
    let dev = last.device();
    let sem = Tensor::new(&[semantic, semantic], dev)?;
    let sem_embed = lm.code_embed(&sem)?;
    let mut seq = vec![depth.projection.forward(&last.unsqueeze(1)?)?, depth.projection.forward(&sem_embed.unsqueeze(1)?)?];
    let mut codes = [0u32; CODEBOOKS];
    codes[0] = semantic;
    let mut hidden = Vec::with_capacity(CODEBOOKS - 1);
    for (index, slot) in codes.iter_mut().enumerate().skip(1) {
        let h = depth.forward(&Tensor::cat(&seq, 1)?)?;
        let h = h.narrow(1, h.dim(1)? - 1, 1)?.squeeze(1)?.contiguous()?;
        hidden.push(h.narrow(0, 0, 1)?);
        let logits = depth.heads_out[index - 1].forward(&h)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let code = pick(index, &guide(&logits[0], &logits[1]));
        *slot = code;
        if index < CODEBOOKS - 1 {
            let e = depth.audio_embed(code, index, 2)?;
            seq.push(depth.projection.forward(&e.unsqueeze(1)?)?);
        }
    }
    let feedback = depth.frame_embedding(&sem_embed, &codes[1..])?;
    Ok((codes, Tensor::cat(&hidden, 1)?, feedback))
}

/// Generate up to `max_frames` frames for the prompt pair `ids`
/// (conditional, unconditional). `progress(frames)` after each one.
pub fn generate(lm: &mut Lm, depth: &Depth, ids: &[Vec<u32>; 2], max_frames: usize, mut rng: Option<&mut Rng>, mut progress: impl FnMut(usize) -> Result<()>) -> Result<Vec<Frame>> {
    let mut cache = Cache::new(lm.layers(), ids[0].len() + max_frames + 2);
    let x = lm.embed(ids)?;
    let mut last = lm.forward(&x, &mut cache)?;
    let mut frames = Vec::new();
    // The first step only moves past <|audio_start|>; its frame is fed back
    // but not kept.
    for index in 0..=max_frames {
        let Some(semantic) = pick_semantic(&lm.logits(&last)?, rng.as_deref_mut())? else { break };
        let (codes, depth_hidden, feedback) = depth_codes(depth, lm, &last, semantic, &mut |_, g| sample_top_k(g, rng.as_deref_mut()) as u32)?;
        if index > 0 {
            let hidden = Tensor::cat(&[&last.narrow(0, 0, 1)?, &depth_hidden], 1)?.to_device(&Device::Cpu)?;
            frames.push(Frame { codes, hidden });
            progress(frames.len())?;
            if frames.len() >= max_frames {
                break;
            }
        }
        last = lm.forward(&feedback, &mut cache)?;
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::acoustic::golden::{dirs, ints, load, relative};

    #[test]
    fn sampling_keeps_the_top_candidates() {
        let mut rng = Rng::new(3);
        let mut v = vec![f32::NEG_INFINITY; 200];
        for (i, x) in v.iter_mut().enumerate().take(60) {
            *x = i as f32 * 0.01;
        }
        assert_eq!(sample_top_k(&v, None), 59);
        for _ in 0..500 {
            let i = sample_top_k(&v, Some(&mut rng));
            assert!((10..60).contains(&i), "outside the top 50: {i}");
        }
        let n = rng.normals(10_000);
        let mean = n.iter().sum::<f32>() / n.len() as f32;
        let var = n.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n.len() as f32;
        assert!(mean.abs() < 0.05 && (var - 1.).abs() < 0.05, "{mean} {var}");
    }

    #[test]
    fn language_model_matches_reference() -> Result<()> {
        let Some((g, m)) = dirs() else { return Ok(()) };
        let dev = crate::music::acoustic::golden::device();
        let p = oaiy_engine::json::Json::parse(&std::fs::read(g.join("prompt.json"))?).map_err(candle_core::Error::wrap)?;
        let ids = crate::music::prompt_ids(&m, p.get("prompt").and_then(oaiy_engine::json::Json::as_str).unwrap(), p.get("lyrics").and_then(oaiy_engine::json::Json::as_str).unwrap())?;
        let started = std::time::Instant::now();
        let quantized = std::env::var("OAIY_MUSIC_LM").ok().map(std::path::PathBuf::from);
        // A quantized model (OAIY_MUSIC_LM) is held to looser bounds: q8_0
        // roughly doubles BF16's differences, q4_k is ten times them.
        let (tol, margin) = match quantized.as_deref().and_then(|p| p.to_str()) {
            None => (1., 0.25),
            Some(p) if p.contains("q8_0") => (2.5, 1.),
            Some(_) => (15., 4.),
        };
        let mut lm = Lm::load(&m.join("language_model"), quantized.as_deref(), &Budget::default(), &dev, |_| {})?;
        let depth = Depth::load(&m.join("rvq_depth_decoder"), &dev)?;
        println!("loaded in {:.1?}", started.elapsed());

        // The depth decoder on the reference's own inputs.
        for i in 0..7 {
            let y = depth.forward(&load(&g, &format!("depth_in{i}.f32"), &dev).to_dtype(DType::BF16)?)?;
            let err = relative(&y, &load(&g, &format!("depth_out{i}.f32"), &dev));
            println!("depth step {i}: {err:.2e}");
            assert!(err < 2e-2, "depth step {i}: {err}");
        }

        let mut cache = Cache::new(lm.layers(), ids[0].len() + 64);
        let x = lm.embed(&ids)?;
        let last = lm.forward(&x, &mut cache)?;
        let hidden = load(&g, "lm_hidden.f32", &dev);
        let err = relative(&last, &hidden.get(0)?);
        println!("prefill hidden: {err:.2e}");
        assert!(err < 2e-2 * tol, "prefill hidden: {err}");
        // Guided logits over the codes the stage can emit.
        let logits = lm.logits(&last)?.to_vec2::<f32>()?;
        let guided = guide(&logits[0], &logits[1]);
        let reference = load(&g, "lm_logits.f32", &Device::Cpu).get(0)?.get(0)?.to_vec1::<f32>()?;
        let expected: Vec<f32> = std::iter::once(AUDIO_END as usize).chain(CODE_OFFSET as usize..CODE_OFFSET as usize + SEMANTIC_CODES).map(|i| reference[i]).collect();
        let finite: Vec<usize> = (0..expected.len()).filter(|&i| expected[i].is_finite()).collect();
        let a = Tensor::new(finite.iter().map(|&i| guided[i]).collect::<Vec<_>>(), &Device::Cpu)?;
        let e = Tensor::new(finite.iter().map(|&i| expected[i]).collect::<Vec<_>>(), &Device::Cpu)?;
        let err = relative(&a, &e);
        println!("guided logits ({} candidates): {err:.2e}", finite.len());
        assert!(err < 2e-2 * tol, "guided logits: {err}");

        let codes = ints(&g, "codes.i32");
        let reference_hidden = {
            let (a, b) = (load(&g, "cond_in0.f32", &Device::Cpu), load(&g, "cond_in1.f32", &Device::Cpu));
            Tensor::cat(&[&a, &b.narrow(1, 100, b.dim(1)? - 100)?], 1)?.squeeze(0)?
        };
        // The reference's own codes fed back: every frame's hidden states, and
        // how close each of our greedy choices came to the reference's.
        let frames_forced = 40;
        let mut last = last;
        let (mut agree, mut total, mut worst) = (0, 0, 0f32);
        for f in 0..=frames_forced {
            let want: Vec<u32> = codes[f * 8..f * 8 + 8].iter().map(|&c| c as u32).collect();
            let semantic = pick_semantic_forced(&lm.logits(&last)?, want[0], &mut agree, &mut total, &mut worst)?;
            let (got, depth_hidden, feedback) = depth_codes(&depth, &lm, &last, semantic, &mut |index, g| {
                let ours = sample_top_k(g, None) as u32;
                total += 1;
                if ours == want[index] { agree += 1 } else { worst = worst.max(g[ours as usize] - g[want[index] as usize]) }
                want[index]
            })?;
            assert_eq!(got.as_slice(), want.as_slice());
            if f > 0 {
                let h = Tensor::cat(&[&last.narrow(0, 0, 1)?, &depth_hidden], 1)?;
                let err = relative(&h, &reference_hidden.narrow(0, f - 1, 1)?.to_device(&dev)?);
                assert!(err < 3e-2 * tol, "forced frame {f} hidden: {err}");
            }
            last = lm.forward(&feedback, &mut cache)?;
        }
        println!("teacher-forced: {agree} of {total} greedy picks agree; largest margin where they differ {worst:.3}");
        assert!(worst < margin, "a choice differed by {worst}, more than a near tie");

        // A short greedy song: codes, feedback and frame hiddens.
        let n = 24;
        let frames = generate(&mut lm, &depth, &ids, n, None, |_| Ok(()))?;
        let mut same = 0;
        for (i, f) in frames.iter().enumerate() {
            let want: Vec<u32> = codes[(i + 1) * 8..(i + 2) * 8].iter().map(|&c| c as u32).collect();
            if f.codes.as_slice() != want.as_slice() {
                println!("frame {i}: {:?} vs reference {want:?}", f.codes);
                break;
            }
            let err = relative(&f.hidden, &reference_hidden.narrow(0, i, 1)?);
            assert!(err < 5e-2 * tol, "frame {i} hidden: {err}");
            same += 1;
        }
        // Free-running greedy decoding parts ways at the first BF16 near tie
        // (the teacher-forced check above bounds those), so this only reports.
        println!("{same} of {n} greedy frames identical before the first near tie");
        assert_eq!(frames.len(), n);
        Ok(())
    }

    /// The reference's semantic code, counting whether our greedy pick agreed.
    fn pick_semantic_forced(logits: &Tensor, want: u32, agree: &mut usize, total: &mut usize, worst: &mut f32) -> Result<u32> {
        let rows = logits.to_vec2::<f32>()?;
        let threshold = kth_largest(&rows[0], TOP_K);
        let guided: Vec<f32> = guide(&rows[0], &rows[1]).into_iter().zip(&rows[0]).map(|(g, &c)| if c < threshold { f32::NEG_INFINITY } else { g }).collect();
        let ours = sample_top_k(&guided, None);
        *total += 1;
        if ours == want as usize + 1 { *agree += 1 } else { *worst = worst.max(guided[ours] - guided[want as usize + 1]) }
        Ok(want)
    }
}

