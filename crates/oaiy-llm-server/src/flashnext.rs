//! Qwen3.8-Flash-Next (`qwen4_exp`) from its EXL3 checkpoint. No Python runtime/conversion.
//!
//! The Qwen3.5-MoE lineage (Gated DeltaNet and gated full attention, 512 experts with a
//! shared one), with three new parts, each following turboderp's exllamav3 reference:
//!
//! - Gated residual: the residual is 4 fp32 streams. Before each attention and MoE branch, a
//!   per-stream RMS norm and a low-rank sigmoid gate make the branch's input (the gated mean
//!   of the normed streams); the branch output goes back into each stream with a scalar
//!   `2 * sigmoid` weight. A last gate collapses the streams before the head (no final norm).
//! - N-gram embedding (PLE): before layer 1, hashed bigrams and trigrams index a trellis-
//!   quantized table (read from disk as needed), which gates a value into every stream and
//!   feeds a dilated causal conv.
//! - QSA: the full-attention layers attend to the top 512 four-token blocks an indexer picks
//!   (plus the current block). Up to 2051 tokens that is every token, so dense attention runs
//!   there (exactly the same); past it, the indexer scores the blocks and attention reads only
//!   the chosen tokens.
//!
//! The Gated DeltaNet's output gate is `sigmoid(z)` here (Qwen3.5's is `silu(z)`).
//!
//! On CUDA (`load_with_adapters`) the layers split over the cards, PEFT adapters applied, and a decode step runs as
//! a graph per card. Without CUDA (`load_portable`) every EXL3 matrix, the experts' too, is whatever the caller makes
//! of it (on any GPU through WebGPU, else decoded on the CPU: `ggml_rs_wgpu::exl3`), on one backend, uncaptured.
use crate::lora::{adapted, stacked, Adapter, Base, Part, VHeads};
use crate::orcasaq::{inverse, tokenizer, value_map};
use dsv41::safetensors::{Dtype, StIndex};
use ggml_rs::{exl3::{Exl3Data, Experts, PackedLinear}, Backend, Tensor};
use ggml_rs::tensor::round_f16;
#[cfg(feature = "cuda")]
use ggml_rs_cuda::{exl3::{Exl3Experts, Exl3Matrix, HalfMatrix}, CudaBackend};

/// The CUDA cards a decode step is captured on, as graphs.
#[cfg(feature = "cuda")]
pub type Card = CudaBackend;
/// A build without CUDA has no cards to capture on: nothing asks it to (`graphs_enabled` is false).
#[cfg(not(feature = "cuda"))]
pub type Card = NoCard;

#[cfg(not(feature = "cuda"))]
pub struct NoCard;
#[cfg(not(feature = "cuda"))]
impl NoCard {
    fn graph_begin(&self) -> bool {
        false
    }
    fn graph_end(&self) {}
    fn graph_finish(&self) {}
    fn graph_launch(&self) {}
    fn device_address(&self, _: &Tensor) -> u64 {
        0
    }
    fn write_at(&self, _: u64, _: &[f32]) {}
    fn prime_rope_positions(&self, _: &[u32]) {}
}

/// Makes an EXL3 matrix on a device (by its index): packed on the card, or as a portable build places it.
pub type Packer<'a> = &'a (dyn Fn(usize) -> Box<dyn Fn(Exl3Data) -> std::result::Result<Arc<dyn PackedLinear>, String> + Send + Sync + 'a> + Sync);
/// Makes a layer's experts (the shared one last) on a device, given the layer's MLP prefix (for its adapters).
pub type ExpertMaker<'a> = &'a (dyn Fn(usize, &str, Vec<[Exl3Data; 3]>) -> Result<Box<dyn Experts>> + Sync);
use llama_rs::{
    loader::Weight,
    KvCache,
};
use oaiy_engine::{json::Json, Error, Result};
use std::{fs::File, path::Path, sync::Arc};

fn bad(s: impl Into<String>) -> Error {
    Error::Format(format!("Qwen3.8-Flash-Next: {}", s.into()))
}


/// A Qwen3.8-Flash-Next checkpoint: `qwen4_exp`, EXL3.
pub fn detect(path: &Path) -> bool {
    std::fs::read(path.join("config.json")).ok().and_then(|b| Json::parse(&b).ok()).is_some_and(|c| {
        c.get("model_type").and_then(Json::as_str) == Some("qwen4_exp")
            && c.get("quantization_config").and_then(|q| q.get("quant_method")).and_then(Json::as_str) == Some("exl3")
    })
}

fn number(c: &Json, key: &str) -> Result<usize> {
    let v = c.get(key).and_then(Json::as_f64).ok_or_else(|| bad(format!("missing {key}")))?;
    if !v.is_finite() || v < 1.0 || v.fract() != 0.0 || v > 1e9 {
        return Err(bad(format!("invalid {key}")));
    }
    Ok(v as usize)
}

/// The dimensions this implementation reads from `text_config`.
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub vocab: usize,
    pub nk: usize,
    pub nv: usize,
    pub kd: usize,
    pub vd: usize,
    pub conv: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_ff: usize,
    pub shared_ff: usize,
    pub streams: usize,
    pub attention: Vec<bool>,
    /// The block the n-gram layer runs before (0-based).
    pub ple_layer: usize,
    pub ple_dim: usize,
    pub ple_kernel: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub ple_eos: u32,
    pub context_length: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_budget: usize,
    pub index_ratio: usize,
}

impl Config {
    fn read(path: &Path) -> Result<Self> {
        let raw = Json::parse(&std::fs::read(path.join("config.json"))?)?;
        let c = raw.get("text_config").unwrap_or(&raw);
        let rope = c.get("rope_parameters").ok_or_else(|| bad("missing rope parameters"))?;
        let head_dim = number(c, "head_dim")?;
        let types = c.get("layer_types").and_then(Json::as_array).ok_or_else(|| bad("missing layer types"))?;
        let attention: Vec<bool> = types.iter().map(|t| t.as_str() == Some("full_attention")).collect();
        let ple = c.get("ple_layer_ids").and_then(Json::as_array).ok_or_else(|| bad("missing ple_layer_ids"))?;
        if ple.len() != 1 {
            return Err(bad("exactly one n-gram layer is supported"));
        }
        let ple_layer = ple[0].as_f64().filter(|v| *v >= 1.0).ok_or_else(|| bad("invalid ple_layer_ids"))? as usize - 1;
        let cfg = Self {
            hidden: number(c, "hidden_size")?,
            layers: number(c, "num_hidden_layers")?,
            heads: number(c, "num_attention_heads")?,
            kv_heads: number(c, "num_key_value_heads")?,
            head_dim,
            rope_dim: (head_dim as f64 * rope.get("partial_rotary_factor").and_then(Json::as_f64).unwrap_or(0.25)) as usize,
            rope_theta: rope.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e7) as f32,
            eps: c.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            vocab: number(c, "vocab_size")?,
            nk: number(c, "linear_num_key_heads")?,
            nv: number(c, "linear_num_value_heads")?,
            kd: number(c, "linear_key_head_dim")?,
            vd: number(c, "linear_value_head_dim")?,
            conv: number(c, "linear_conv_kernel_dim")?,
            experts: number(c, "num_experts")?,
            top_k: number(c, "num_experts_per_tok")?,
            moe_ff: number(c, "moe_intermediate_size")?,
            shared_ff: number(c, "shared_expert_intermediate_size")?,
            streams: number(c, "hc_count")?,
            attention,
            ple_layer,
            ple_dim: number(c, "ple_embed_dim")?,
            ple_kernel: number(c, "ple_conv_kernel_size")?,
            ngram: number(c, "ngram_size")?,
            heads_per_ngram: number(c, "heads_per_ngram")?,
            ple_eos: number(c, "eos_token_id")? as u32,
            context_length: number(c, "max_position_embeddings")?,
            index_heads: number(c, "indexer_n_heads")?,
            index_dim: number(c, "indexer_head_dim")?,
            index_budget: number(c, "indexer_budget")?,
            index_ratio: number(c, "indexer_compress_ratio")?,
        };
        if cfg.attention.len() != cfg.layers || cfg.ple_layer >= cfg.layers || cfg.heads % cfg.kv_heads != 0
            || cfg.nv % cfg.nk != 0 || cfg.kd != cfg.vd || cfg.streams != 4 || cfg.ngram < 2 || cfg.rope_dim > head_dim
            || cfg.rope_dim > cfg.index_dim || cfg.index_budget % cfg.index_ratio != 0 {
            return Err(bad("unsupported dimensions"));
        }
        Ok(cfg)
    }
}

/// One gated-residual site (or, without write weights, the final stream collapse).
struct HyperMix {
    /// `1 + w`, per stream: `[streams * hidden]`.
    norm: Tensor,
    /// The low-rank gate's down projection, then (at a site) each stream's write logit:
    /// `[rank + writes, streams * hidden]`.
    down: Tensor,
    /// The gate's up projection, with a zero column for each write logit: `[streams * hidden, rank + writes]`.
    up: Tensor,
    rank: usize,
    /// Whether it writes back into the streams (a site) or only collapses them.
    site: bool,
    /// Whether `down` and `up` are f16 packed two to a word (a card's), or f32: without a card they are unpacked
    /// once at load, where the host ops unpacked them on every call (a second of each decode step).
    packed: bool,
}

impl HyperMix {
    /// The branch input (`[rows, hidden]`) and, for a site, each stream's write weight (`[rows, streams]`).
    /// The branch input and, at a site, its write weights. `pending`: the previous branch's
    /// output and write weights, not yet added to `x`: added here, with the norm.
    fn mix(&self, b: &dyn Backend, x: &mut Tensor, pending: Option<(Tensor, Tensor)>, streams: usize, eps: f32) -> (Tensor, Option<Tensor>) {
        let mut prof = profile::Timer::new(b);
        let normed = match pending {
            Some((y, post)) => b.hc_apply_norm(x, &y, &post, &self.norm, streams, eps),
            None => b.hc_norm(x, &self.norm, streams, eps),
        };
        prof.lap("hc:norm");
        let writes = if self.site { streams } else { 0 };
        let (t, post) = if self.packed {
            b.hc_down_gates(&normed, &self.down, self.rank, writes, streams)
        } else {
            let mut t = b.linear(&normed, &self.down);
            let post = b.hc_gates(&mut t, self.rank, writes, streams);
            (t, post)
        };
        prof.lap("hc:down");
        let mixed = if self.packed { b.hc_up_mix(&t, &self.up, &normed, streams) } else { b.hc_mix(&b.linear(&t, &self.up), &normed, streams) };
        prof.lap("hc:up");
        (mixed, self.site.then_some(post))
    }
}

struct Gdn {
    qkv: Weight,
    z: Weight,
    ba: Weight,
    a: Tensor,
    dt_bias: Tensor,
    conv: Tensor,
    norm: Tensor,
    out: Weight,
}

struct Attn {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Tensor,
    k_norm: Tensor,
    /// The QSA indexer: its query heads and raw key, and their norms (`1 + w`).
    index_qk: Weight,
    index_q_norm: Tensor,
    index_k_norm: Tensor,
    /// The cache slot holding this layer's raw indexer keys.
    index_slot: usize,
}

enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

struct Moe {
    /// The router, then the shared expert's gate: `[routed + 1, hidden]`.
    router: Weight,
    /// The routed experts, then the shared one: on CUDA stacked, the MoE in a few launches; elsewhere batched a
    /// layer at a time.
    experts: Box<dyn Experts>,
}

struct Layer {
    device: usize,
    attn_hc: HyperMix,
    mlp_hc: HyperMix,
    mixer: Mixer,
    moe: Moe,
}

/// The hashed n-gram table: trellis-quantized 160-value rows, read from disk as needed.
struct NgramTable {
    /// A handle per reading thread: Windows serializes the reads made through one handle.
    files: Vec<File>,
    start: u64,
    rows: u64,
    row_words: usize,
    bits: usize,
    head_offsets: Vec<i64>,
    head_sizes: Vec<i64>,
    multipliers: Vec<i64>,
    /// `[heads, 160]`.
    bias: Vec<f32>,
    codebook: Vec<f32>,
}

const ROW_DIM: usize = 160;
const MUL1: u64 = 0x83DC_D12D;

/// The 65536 decoded `mul1` values, as fp16 (bit-exact with EXL3's codebook).
fn mul1_codebook() -> Vec<f32> {
    let k_inv = dsv41::formats::f16_to_f32(0x1eee);
    let k_bias = dsv41::formats::f16_to_f32(0xc931);
    (0..65536u64).map(|s| {
        let p = (s * MUL1) & 0xffff_ffff;
        let sum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + ((p >> 24) & 255);
        round_f16((1024 + sum) as f32 * k_inv + k_bias)
    }).collect()
}

impl NgramTable {
    fn open(idx: &StIndex, key: &str, heads: usize) -> Result<Self> {
        let mut shards = Vec::new();
        while let Some(info) = idx.get(&format!("{key}.shard_{}.trellis", shards.len())) { shards.push(info.clone()); }
        if shards.is_empty() {
            shards.push(idx.info(&format!("{key}.trellis"))?.clone());
        }
        let first = &shards[0];
        if first.dtype != Dtype::I16 || first.shape.len() != 2 {
            return Err(bad("n-gram table: expected int16 trellis rows"));
        }
        let row_words = first.shape[1];
        let bits = (row_words - 1) * 16 / ROW_DIM;
        if 1 + ROW_DIM * bits / 16 != row_words {
            return Err(bad("n-gram table: unexpected row width"));
        }
        // The shards sit back to back in one file: one table.
        let row_bytes = (row_words * 2) as u64;
        let mut rows = 0u64;
        for s in &shards {
            if s.shard != first.shard || s.start != first.start + rows * row_bytes || s.shape[1] != row_words {
                return Err(bad("n-gram table shards are not contiguous"));
            }
            rows += s.shape[0] as u64;
        }
        let head_offsets = idx.read_i64(&format!("{key}.head_offsets"))?;
        let head_sizes = idx.read_i64(&format!("{key}.head_vocab_sizes"))?;
        let multipliers = idx.read_i64(&format!("{key}.layer_multipliers"))?;
        let bias = idx.read_f32(&format!("{key}.head_bias"))?;
        if head_offsets.len() != heads || head_sizes.len() != heads || bias.len() != heads * ROW_DIM {
            return Err(bad("n-gram table: head parameters do not match the heads"));
        }
        Ok(Self {
            files: (0..rayon::current_num_threads().max(1)).map(|_| File::open(idx.shard_path(first.shard))).collect::<std::io::Result<_>>()?,
            start: first.start,
            rows,
            row_words,
            bits,
            head_offsets,
            head_sizes,
            multipliers,
            bias,
            codebook: mul1_codebook(),
        })
    }

    /// Row `row` (of hash head `head`), decoded.
    fn row(&self, row: u64, head: usize, out: &mut [f32]) -> Result<()> {
        if row >= self.rows {
            return Err(bad("n-gram row out of range"));
        }
        let mut buf = vec![0u8; self.row_words * 2];
        let file = &self.files[rayon::current_thread_index().unwrap_or(0) % self.files.len()];
        read_at(file, &mut buf, self.start + row * (self.row_words * 2) as u64)?;
        let words: Vec<u16> = buf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let scale = dsv41::formats::f16_to_f32(words[0]);
        let k = self.bits;
        let bit = |i: usize| (words[1 + i / 16] >> (i % 16)) & 1;
        for (i, o) in out.iter_mut().enumerate() {
            let mut state = 0usize;
            for m in 0..16 {
                let src = ((i + ROW_DIM * 16 - m / k) % ROW_DIM) * k + m % k;
                state |= (bit(src) as usize) << m;
            }
            *o = self.codebook[state] * scale + self.bias[head * ROW_DIM + i];
        }
        Ok(())
    }
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], mut at: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.seek_read(&mut buf[done..], at)?;
        if n == 0 { return Err(bad("n-gram table: short read")); }
        done += n;
        at += n as u64;
    }
    Ok(())
}
#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], at: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, at)?;
    Ok(())
}

/// The n-gram layer. Small, once per forward: it runs on the host, but its projections.
struct Ple {
    table: NgramTable,
    key: Weight,
    value: Weight,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// `[streams * hidden, kernel]`.
    conv: Tensor,
}

/// Where the n-gram layer's state lives in the KV cache: its own slot after the layers.
/// `ssm_conv` holds the conv window (`[state_len, streams * hidden]`), `ssm_state` the last
/// `ngram - 1` token ids.
fn ple_slot(cfg: &Config) -> usize {
    cfg.layers
}

pub struct FlashNext {
    pub config: Config,
    pub tokenizer: tokenizer::Tokenizer,
    embed: (File, u64, Dtype),
    layers: Vec<Layer>,
    ple: Ple,
    collapse: HyperMix,
    head: Weight,
    pub devices: Vec<Arc<dyn Backend>>,
    /// The same GPUs, for what uploads EXL3 matrices (the vision tower) and the decode graphs; none without CUDA.
    pub cudas: Vec<Arc<Card>>,
    /// Whether a decode step has run (uncaptured), making the scratch its matrices keep.
    decoded: std::sync::atomic::AtomicBool,
}

/// The EXL3 matrix `name` (`k -> n`, mul1 codebook) on `backend`, its input and output
/// channels optionally reordered.
/// A dense `rows x cols` matrix: kept as f16 (half the memory and reading) when every value is
/// one, as the checkpoint's f16 weights are; f32 otherwise.
/// Without a card, f32 on `backend`.
fn half_or_dense(card: Option<&Arc<Card>>, backend: &Arc<dyn Backend>, values: Vec<f32>, rows: usize, cols: usize) -> Weight {
    #[cfg(feature = "cuda")]
    if let Some(card) = card {
        if cols % 2 == 0 && values.iter().all(|&v| round_f16(v) == v) {
            return Weight::Packed(Arc::new(HalfMatrix::upload(card.clone(), &values, rows, cols)));
        }
    }
    let _ = card;
    Weight::Dense(backend.to_device(Tensor::from_vec(values, vec![rows, cols])))
}

#[cfg(feature = "cuda")]
pub(crate) fn exl3_weight(idx: &StIndex, backend: &Arc<CudaBackend>, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>) -> Result<Weight> {
    let data = exl3_data(idx, name, k, n, input, output)?;
    Ok(Weight::Packed(Arc::new(Exl3Matrix::upload(backend.clone(), data).map_err(bad)?)))
}

/// The EXL3 matrix `name` (`k -> n`), read and checked, on the host.
fn exl3_data(idx: &StIndex, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>) -> Result<Exl3Data> {
    let key = format!("{name}.trellis");
    let info = idx.info(&key)?;
    if info.dtype != Dtype::I16 || info.shape.len() != 3 || info.shape[..2] != [k / 16, n / 16] {
        return Err(bad(format!("{key}: invalid EXL3 tile shape {:?}", info.shape)));
    }
    let mul = idx.read_i64(&format!("{name}.mul1"))?;
    if mul.len() != 1 || mul[0] as u32 != MUL1 as u32 {
        return Err(bad(format!("{name}: unsupported codebook")));
    }
    let suh = idx.read_f32(&format!("{name}.suh"))?;
    let svh = idx.read_f32(&format!("{name}.svh"))?;
    if suh.len() != k || svh.len() != n {
        return Err(bad(format!("{name}: invalid incoherence vector lengths")));
    }
    let bytes = idx.read(&key)?;
    let data = Exl3Data {
        words: bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        suh,
        svh,
        tile_words: info.shape[2],
        input_map: input.unwrap_or_else(|| (0..k as u32).collect()),
        output_map: output.unwrap_or_else(|| (0..n as u32).collect()),
    };
    Ok(data)
}

struct Loader<'a> {
    idx: &'a StIndex,
    backend: Arc<dyn Backend>,
    /// The device's CUDA card (none without CUDA).
    card: Option<Arc<Card>>,
    /// LoRA adapters, applied together.
    lora: &'a [Adapter],
    /// Makes this device's EXL3 matrices.
    packed: Box<dyn Fn(Exl3Data) -> std::result::Result<Arc<dyn PackedLinear>, String> + Send + Sync + 'a>,
}

impl Loader<'_> {
    fn host(&self, name: &str, numel: usize, add_one: bool, map: Option<&[u32]>) -> Result<Vec<f32>> {
        let mut data = self.idx.read_f32(name)?;
        if data.len() != numel {
            return Err(bad(format!("{name}: {} values, expected {numel}", data.len())));
        }
        if add_one { for v in &mut data { *v += 1.0; } }
        if let Some(map) = map {
            let width = data.len() / map.len();
            let src = data.clone();
            for (i, &j) in map.iter().enumerate() {
                data[i * width..(i + 1) * width].copy_from_slice(&src[j as usize * width..(j as usize + 1) * width]);
            }
        }
        Ok(data)
    }
    fn tensor(&self, name: &str, shape: &[usize], add_one: bool, map: Option<&[u32]>) -> Result<Tensor> {
        let data = self.host(name, shape.iter().product(), add_one, map)?;
        Ok(self.backend.to_device(Tensor::from_vec(data, shape.to_vec())))
    }
    fn dense(&self, name: &str, rows: usize, cols: usize) -> Result<Weight> {
        let base = half_or_dense(self.card.as_ref(), &self.backend, self.host(name, rows * cols, false, None)?, rows, cols);
        let target = name.strip_suffix(".weight").unwrap_or(name);
        self.adapt(base, &[Part { name: target.into(), offset: 0, rows, output: None }], cols, rows, None)
    }
    /// `base` (`n x k`) with the adapters' LoRA for `parts` of it, if they have any.
    fn adapt(&self, base: Weight, parts: &[Part<'_>], k: usize, n: usize, input: Option<&[u32]>) -> Result<Weight> {
        Ok(match stacked(self.lora, parts, k, n, input)? {
            Some((a, b, rank)) => adapted(base, a, b, rank, self.backend.clone()),
            None => base,
        })
    }
    /// An EXL3 matrix `k -> n`.
    fn weight(&self, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>) -> Result<Weight> {
        self.weight_parts(name, k, n, input, output, &[])
    }
    /// As `weight`, with LoRA also for `more` parts of it (Hugging Face matrices llama.cpp
    /// splits it into).
    fn weight_parts(&self, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>, more: &[Part<'_>]) -> Result<Weight> {
        let base = Weight::Packed((self.packed)(exl3_data(self.idx, name, k, n, input.clone(), output.clone())?).map_err(bad)?);
        let mut parts = vec![Part { name: name.into(), offset: 0, rows: n, output: output.as_deref() }];
        parts.extend(more.iter().map(|p| Part { name: p.name.clone(), offset: p.offset, rows: p.rows, output: p.output }));
        self.adapt(base, &parts, k, n, input.as_deref())
    }
    fn hyper(&self, p: &str, cfg: &Config, site: bool) -> Result<HyperMix> {
        let width = cfg.streams * cfg.hidden;
        let rank = self.idx.info(&format!("{p}.input_mix_weight_down.weight"))?.shape[0];
        let writes = if site { cfg.streams } else { 0 };
        // The write logits ride along the down projection; the up projection ignores them.
        let mut down = self.host(&format!("{p}.input_mix_weight_down.weight"), rank * width, false, None)?;
        if site {
            down.extend(self.host(&format!("{p}.block_inject_weight.weight"), cfg.streams * width, false, None)?);
        }
        let up = self.host(&format!("{p}.input_mix_weight_up.weight"), width * rank, false, None)?;
        let mut padded = Vec::with_capacity(width * (rank + writes));
        for row in up.chunks_exact(rank) {
            padded.extend_from_slice(row);
            padded.extend(std::iter::repeat_n(0.0, writes));
        }
        // f16 in the checkpoint: kept so on a card, two to a word (half the memory and reading); f32 elsewhere.
        let packed = self.card.is_some();
        let (down, up) = if packed {
            (
                Tensor::from_vec(ggml_rs::tensor::pack_f16(&down), vec![rank + writes, width / 2]),
                Tensor::from_vec(ggml_rs::tensor::pack_f16(&padded), vec![width, (rank + writes) / 2]),
            )
        } else {
            (Tensor::from_vec(down, vec![rank + writes, width]), Tensor::from_vec(padded, vec![width, rank + writes]))
        };
        Ok(HyperMix {
            norm: self.tensor(&format!("{p}.hc_norm.weight"), &[width], true, None)?,
            down: self.backend.to_device(down),
            up: self.backend.to_device(up),
            rank,
            site,
            packed,
        })
    }
}

/// What a LoRA for the checkpoint at `path` must be for.
pub(crate) fn lora_base(path: &Path) -> Result<Base> {
    let cfg = Config::read(path)?;
    Ok(Base::FlashNext(VHeads { k_heads: cfg.nk, v_heads: cfg.nv, k_dim: cfg.kd, v_dim: cfg.vd }))
}

/// A layer's LoRA on projection `which` (0 gate, 1 up, 2 down) of its experts (the shared one
/// last): which have one (`u32::MAX` for none), and their A and B at one rank (the largest,
/// the others padded with zeros).
fn expert_lora(lora: &[Adapter], m: &str, cfg: &Config, which: usize) -> Result<Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize)>> {
    let proj = ["gate_proj", "up_proj", "down_proj"][which];
    let (k, n) = if which == 2 { (cfg.moe_ff, cfg.hidden) } else { (cfg.hidden, cfg.moe_ff) };
    let mut found = Vec::new();
    for e in 0..=cfg.experts {
        let name = if e < cfg.experts { format!("{m}.experts.{e}.{proj}") } else { format!("{m}.shared_expert.{proj}") };
        if let Some(pair) = stacked(lora, &[Part { name, offset: 0, rows: n, output: None }], k, n, None)? {
            found.push((e, pair));
        }
    }
    let Some(rank) = found.iter().map(|(_, p)| p.2).max() else { return Ok(None) };
    let mut slot_of = vec![u32::MAX; cfg.experts + 1];
    let (mut a_all, mut b_all) = (Vec::with_capacity(found.len() * rank * k), Vec::with_capacity(found.len() * n * rank));
    for (slot, (e, (a, b, r))) in found.into_iter().enumerate() {
        slot_of[e] = slot as u32;
        a_all.extend_from_slice(&a);
        a_all.resize(a_all.len() + (rank - r) * k, 0.0);
        for row in b.chunks_exact(r) {
            b_all.extend_from_slice(row);
            b_all.resize(b_all.len() + rank - r, 0.0);
        }
    }
    Ok(Some((slot_of, a_all, b_all, rank)))
}

/// Load the model, its layers split evenly over `devices` (in order).
#[cfg(all(test, feature = "cuda"))]
pub fn load(path: &Path, devices: &[usize]) -> Result<FlashNext> {
    load_with_adapters(path, devices, &[])
}

/// As `load`, with LoRA adapters (from `Adapter::open_for` with `lora_base`) applied together.
#[cfg(feature = "cuda")]
pub(crate) fn load_with_adapters(path: &Path, devices: &[usize], lora: &[Adapter]) -> Result<FlashNext> {
    let cfg = Config::read(path)?;
    let ids: Vec<usize> = if devices.is_empty() { vec![0] } else { devices.to_vec() };
    // Streams of their own, so that decode steps can run as graphs.
    let cudas: Vec<Arc<CudaBackend>> = ids.iter().map(|&d| CudaBackend::new_graphable(d).map(Arc::new).map_err(|e| bad(e.to_string()))).collect::<Result<_>>()?;
    let backends: Vec<Arc<dyn Backend>> = cudas.iter().map(|c| c.clone() as Arc<dyn Backend>).collect();
    let packed = |d: usize| -> Box<dyn Fn(Exl3Data) -> std::result::Result<Arc<dyn PackedLinear>, String> + Send + Sync> {
        let card = cudas[d].clone();
        Box::new(move |data| Exl3Matrix::upload(card.clone(), data).map(|m| Arc::new(m) as Arc<dyn PackedLinear>))
    };
    let experts = |d: usize, m: &str, list: Vec<[Exl3Data; 3]>| -> Result<Box<dyn Experts>> {
        let mut experts = Exl3Experts::upload(cudas[d].clone(), list).map_err(bad)?;
        if !lora.is_empty() {
            for which in 0..3 {
                if let Some((slot_of, a, b, rank)) = expert_lora(lora, m, &cfg, which)? {
                    experts.set_lora(which, &slot_of, &a, &b, rank).map_err(bad)?;
                }
            }
        }
        Ok(Box::new(experts))
    };
    build(path, backends, cudas.clone(), lora, &packed, &experts)
}

/// The bytes of Flash-Next's EXL3 matrices outside its experts (attention, delta-net, the head): a portable build keeps
/// that much of the GPU budget for them, where the experts, loaded first, took it all and left the 248k-row head to the
/// CPU.
#[cfg_attr(feature = "cuda", allow(dead_code))]
pub(crate) fn dense_exl3_bytes(path: &Path) -> Result<u64> {
    let idx = StIndex::open(path)?;
    Ok(idx
        .names()
        .filter(|n| n.ends_with(".trellis") && n.starts_with("model.language_model.") || *n == "lm_head.trellis")
        .filter(|n| !n.contains(".experts.") && !n.contains(".shared_expert."))
        .filter_map(|n| idx.get(n).map(|i| i.nbytes))
        .sum())
}

/// Flash-Next without CUDA: every tensor on `backend`, each EXL3 matrix as `packed` makes it and each layer's experts
/// as `experts` does (on any GPU through WebGPU, else on the CPU), no PEFT adapters, a decode step uncaptured.
/// (A CUDA build loads it on the cards.)
#[cfg_attr(feature = "cuda", allow(dead_code))]
pub(crate) fn load_portable(path: &Path, backend: Arc<dyn Backend>, packed: Packer<'_>, experts: ExpertMaker<'_>) -> Result<FlashNext> {
    build(path, vec![backend], Vec::new(), &[], packed, experts)
}

fn build(path: &Path, backends: Vec<Arc<dyn Backend>>, cudas: Vec<Arc<Card>>, lora: &[Adapter], packed: Packer<'_>, make_experts: ExpertMaker<'_>) -> Result<FlashNext> {
    if !detect(path) {
        return Err(bad("not a Qwen3.8-Flash-Next EXL3 checkpoint"));
    }
    let cfg = Config::read(path)?;
    let tok = tokenizer(path, cfg.vocab)?;
    let idx = StIndex::open(path)?;
    let on = |d: usize| Loader { idx: &idx, backend: backends[d].clone(), card: cudas.get(d).cloned(), lora, packed: packed(d) };
    let p = "model.language_model";
    let h = cfg.hidden;

    let embed = idx.info(&format!("{p}.embed_tokens.weight"))?.clone();
    if embed.shape != [cfg.vocab, h] || !matches!(embed.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32) {
        return Err(bad("invalid embedding table"));
    }

    let (nk, nv, kd, vd, conv) = (cfg.nk, cfg.nv, cfg.kd, cfg.vd, cfg.conv);
    let vm = value_map(nk, nv, vd);
    let hm = value_map(nk, nv, 1);
    let qkv = 2 * nk * kd + nv * vd;
    let qm: Vec<u32> = (0..(2 * nk * kd) as u32).chain(vm.iter().map(|v| v + 2 * (nk * kd) as u32)).collect();
    // A layer is independent of the others: four load at once (the reads, the stacking of its
    // experts and the uploads overlap, on both GPUs).
    let load_layer = |i: usize| -> Result<Layer> {
        let device = i * backends.len() / cfg.layers;
        let l = on(device);
        let lp = format!("{p}.layers.{i}");
        let mixer = if cfg.attention[i] {
            let a = format!("{lp}.self_attn");
            Mixer::Attn(Attn {
                q: l.weight(&format!("{a}.q_proj"), h, 2 * cfg.heads * cfg.head_dim, None, None)?,
                k: l.weight(&format!("{a}.k_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
                v: l.weight(&format!("{a}.v_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
                o: l.weight(&format!("{a}.o_proj"), cfg.heads * cfg.head_dim, h, None, None)?,
                q_norm: l.tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim], true, None)?,
                k_norm: l.tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim], true, None)?,
                // llama.cpp splits it into the indexer's query and key projections.
                index_qk: l.weight_parts(&format!("{a}.indexer.index_qk_proj"), h, (cfg.index_heads + 1) * cfg.index_dim, None, None, &[
                    Part { name: format!("{a}.indexer.index_q_proj"), offset: 0, rows: cfg.index_heads * cfg.index_dim, output: None },
                    Part { name: format!("{a}.indexer.index_k_proj"), offset: cfg.index_heads * cfg.index_dim, rows: cfg.index_dim, output: None },
                ])?,
                index_q_norm: l.tensor(&format!("{a}.indexer.q_layernorm.weight"), &[cfg.index_dim], true, None)?,
                index_k_norm: l.tensor(&format!("{a}.indexer.k_layernorm.weight"), &[cfg.index_dim], true, None)?,
                index_slot: cfg.layers + 1 + cfg.attention[..i].iter().filter(|&&a| a).count(),
            })
        } else {
            let a = format!("{lp}.linear_attn");
            let beta = l.host(&format!("{a}.in_proj_b.weight"), nv * h, false, Some(&hm))?;
            let alpha = l.host(&format!("{a}.in_proj_a.weight"), nv * h, false, Some(&hm))?;
            let ba = Tensor::from_vec([beta, alpha].concat(), vec![2 * nv, h]);
            let mut av = l.host(&format!("{a}.A_log"), nv, false, None)?;
            for v in &mut av { *v = -v.exp(); }
            let av: Vec<f32> = hm.iter().map(|&j| av[j as usize]).collect();
            Mixer::Gdn(Gdn {
                qkv: l.weight(&format!("{a}.in_proj_qkv"), h, qkv, None, Some(qm.clone()))?,
                z: l.weight(&format!("{a}.in_proj_z"), h, nv * vd, None, Some(vm.clone()))?,
                ba: l.adapt(half_or_dense(cudas.get(device), &backends[device], ba.data().to_vec(), 2 * nv, h), &[
                    Part { name: format!("{a}.in_proj_b"), offset: 0, rows: nv, output: Some(&hm) },
                    Part { name: format!("{a}.in_proj_a"), offset: nv, rows: nv, output: Some(&hm) },
                ], h, 2 * nv, None)?,
                a: backends[device].to_device(Tensor::from_vec(av, vec![nv])),
                dt_bias: l.tensor(&format!("{a}.dt_bias"), &[nv], false, Some(&hm))?,
                conv: l.tensor(&format!("{a}.conv1d.weight"), &[qkv, conv], false, Some(&qm))?,
                norm: l.tensor(&format!("{a}.norm.weight"), &[vd], false, None)?,
                out: l.weight(&format!("{a}.out_proj"), nv * vd, h, Some(inverse(&vm)), None)?,
            })
        };
        let m = format!("{lp}.mlp");
        if cfg.shared_ff != cfg.moe_ff {
            return Err(bad("the shared expert must be as wide as the routed ones"));
        }
        // 1539 matrices a layer, read on several threads (each read is small; the latency adds up).
        let names: Vec<String> = (0..cfg.experts).map(|e| format!("{m}.experts.{e}")).chain([format!("{m}.shared_expert")]).collect();
        let read = |p: &String| -> Result<[Exl3Data; 3]> {
            Ok([
                exl3_data(&idx, &format!("{p}.gate_proj"), h, cfg.moe_ff, None, None)?,
                exl3_data(&idx, &format!("{p}.up_proj"), h, cfg.moe_ff, None, None)?,
                exl3_data(&idx, &format!("{p}.down_proj"), cfg.moe_ff, h, None, None)?,
            ])
        };
        let per = names.len().div_ceil(16);
        let experts: Vec<[Exl3Data; 3]> = std::thread::scope(|scope| {
            let parts: Vec<_> = names.chunks(per).map(|part| scope.spawn(|| part.iter().map(read).collect::<Result<Vec<_>>>())).collect();
            parts.into_iter().map(|t| t.join().expect("expert reader")).collect::<Result<Vec<Vec<_>>>>()
        })?.into_iter().flatten().collect();
        let mut router = l.host(&format!("{m}.gate.weight"), cfg.experts * h, false, None)?;
        router.extend(l.host(&format!("{m}.shared_expert_gate.weight"), h, false, None)?);
        let experts = make_experts(device, &m, experts)?;
        let moe = Moe {
            router: l.adapt(half_or_dense(cudas.get(device), &backends[device], router, cfg.experts + 1, h), &[
                Part { name: format!("{m}.gate"), offset: 0, rows: cfg.experts, output: None },
                Part { name: format!("{m}.shared_expert_gate"), offset: cfg.experts, rows: 1, output: None },
            ], h, cfg.experts + 1, None)?,
            experts,
        };
        Ok(Layer {
            device,
            attn_hc: l.hyper(&format!("{lp}.attn_hyper_connection"), &cfg, true)?,
            mlp_hc: l.hyper(&format!("{lp}.mlp_hyper_connection"), &cfg, true)?,
            mixer,
            moe,
        })
    };
    const WORKERS: usize = 4;
    let mut loaded: Vec<Option<Layer>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS).map(|w| {
            let load_layer = &load_layer;
            scope.spawn(move || (w..cfg.layers).step_by(WORKERS).map(|i| load_layer(i).map(|l| (i, l))).collect::<Result<Vec<_>>>())
        }).collect();
        let mut out: Vec<Option<Layer>> = (0..cfg.layers).map(|_| None).collect();
        for w in workers {
            for (i, layer) in w.join().expect("layer loader")? { out[i] = Some(layer); }
        }
        Ok::<_, Error>(out)
    })?;
    let layers: Vec<Layer> = loaded.iter_mut().map(|l| l.take().expect("every layer")).collect();

    // The n-gram layer sits with the block it runs before.
    let pd = layers[cfg.ple_layer].device;
    let l = on(pd);
    let pp = format!("{p}.layers.{}.ple", cfg.ple_layer);
    let width = cfg.streams * h;
    let heads = (cfg.ngram - 1) * cfg.heads_per_ngram;
    if heads * ROW_DIM != cfg.ple_dim {
        return Err(bad("n-gram heads do not make up the embedding width"));
    }
    let ple = Ple {
        table: NgramTable::open(&idx, &format!("{pp}.ple_embedding.ngram_embedding"), heads)?,
        key: l.dense(&format!("{pp}.key_proj.weight"), width, cfg.ple_dim)?,
        value: l.dense(&format!("{pp}.value_proj.weight"), h, cfg.ple_dim)?,
        norm_key: l.tensor(&format!("{pp}.norm_key.weight"), &[width], true, None)?,
        norm_query: l.tensor(&format!("{pp}.norm_query.weight"), &[width], true, None)?,
        norm_conv: l.tensor(&format!("{pp}.norm_conv.weight"), &[width], true, None)?,
        conv: l.tensor(&format!("{pp}.conv1d.weight"), &[width, cfg.ple_kernel], false, None)?,
    };

    let last = on(backends.len() - 1);
    let collapse = last.hyper(&format!("{p}.hyper_connection_mixer"), &cfg, false)?;
    let head = last.weight("lm_head", h, cfg.vocab, None, None)?;
    // Every target of every adapter found its matrix.
    for adapter in lora { adapter.finish()?; }
    let file = File::open(idx.shard_path(embed.shard))?;
    Ok(FlashNext {
        tokenizer: tok,
        embed: (file, embed.start, embed.dtype),
        layers,
        ple,
        collapse,
        head,
        devices: backends,
        cudas,
        decoded: Default::default(),
        config: cfg,
    })
}

impl FlashNext {
    /// A cache for one sequence: attention K/V on each layer's device, the n-gram slot, then
    /// each attention layer's raw indexer keys.
    pub fn new_kv_cache(&self, max_len: usize) -> KvCache {
        let cfg = &self.config;
        let mut backends: Vec<Arc<dyn Backend>> = self.layers.iter().map(|l| self.devices[l.device].clone()).collect();
        let mut heads: Vec<usize> = (0..cfg.layers).map(|i| if cfg.attention[i] { cfg.kv_heads } else { 1 }).collect();
        let mut dims: Vec<usize> = (0..cfg.layers).map(|i| if cfg.attention[i] { cfg.head_dim } else { 1 }).collect();
        backends.push(self.devices[self.layers[cfg.ple_layer].device].clone());
        heads.push(1);
        dims.push(1);
        for layer in self.layers.iter().filter(|l| matches!(l.mixer, Mixer::Attn(_))) {
            backends.push(self.devices[layer.device].clone());
            heads.push(1);
            dims.push(cfg.index_dim);
        }
        KvCache::new_lazy_per_layer_kv(backends, max_len, &heads, &dims)
    }

    /// Token embeddings, `[tokens, hidden]` on the host.
    pub fn embed_text(&self, tokens: &[u32]) -> Result<Tensor> {
        let h = self.config.hidden;
        let (file, start, dtype) = &self.embed;
        let size = dtype.size();
        let mut out = Vec::with_capacity(tokens.len() * h);
        let mut buf = vec![0u8; h * size];
        for &t in tokens {
            if t as usize >= self.config.vocab {
                return Err(bad("token outside the vocabulary"));
            }
            read_at(file, &mut buf, start + (t as u64) * (h * size) as u64)?;
            match dtype {
                Dtype::F32 => out.extend(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
                Dtype::BF16 => out.extend(buf.chunks_exact(2).map(|c| dsv41::formats::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))),
                _ => out.extend(buf.chunks_exact(2).map(|c| dsv41::formats::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))),
            }
        }
        Ok(Tensor::from_vec(out, vec![tokens.len(), h]))
    }

    /// Run `tokens` (whose embeddings, images included, are `embeds`) through the model after
    /// what `kv` holds: the last position's logits, on the host. `positions`: the multimodal
    /// rope positions, when the prompt has images.
    pub fn forward(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, positions: Option<&[[u32; 3]]>) -> Result<Tensor> {
        let cfg = &self.config;
        let seq = tokens.len();
        let past = kv.len;
        let (h, s) = (cfg.hidden, cfg.streams);
        // The embedding, copied into every stream.
        let e = embeds.to_host();
        let mut stack = Vec::with_capacity(seq * s * h);
        for r in 0..seq { for _ in 0..s { stack.extend_from_slice(&e.data()[r * h..(r + 1) * h]); } }
        let mut device = self.layers[0].device;
        let mut x = self.devices[device].to_device(Tensor::from_vec(stack, vec![seq, s * h]));
        let rope: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();
        // The n-gram features (read from the table on the host) go up before any layer runs.
        let ple_device = self.layers[cfg.ple_layer].device;
        let ple_emb = self.ple_embed(self.devices[ple_device].as_ref(), tokens, kv)?;
        // A decode step runs as one graph per device: its hundreds of launches cost the host
        // one each, several microseconds, which is longer than most of them take on the GPU.
        // Captured, nothing may wait on the device or upload mid-step: the caches grow and the
        // rope positions go up first.
        // The first decode step runs uncaptured: it makes the scratch kept between steps.
        let graphs = seq == 1 && past > 0 && positions.is_none() && graphs_enabled() && !profile::on()
            && self.decoded.load(std::sync::atomic::Ordering::Relaxed);
        if graphs {
            self.prepare_step(kv, &rope, past + seq);
        }
        let mut recording = graphs && self.cudas[device].graph_begin();
        // Captured, a later device's layers are recorded while the earlier ones run: its input
        // (written from the previous device's output) is a buffer the step reads in place. Each
        // waits here to be filled, then run: (device, the output to copy, where it goes).
        let mut handoffs: Vec<(usize, Tensor, u64)> = Vec::new();
        // A branch's output waits to be written back into the streams until the next site's
        // norm reads them (one launch for both), or something else needs them whole.
        let mut pending: Option<(Tensor, Tensor)> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.device != device || i == cfg.ple_layer {
                if let Some((y, post)) = pending.take() {
                    self.devices[device].stream_apply(&mut x, &y, &post, s);
                }
            }
            if layer.device != device {
                if recording {
                    if handoffs.is_empty() { self.cudas[device].graph_end(); } else { self.cudas[device].graph_finish(); }
                    let from = std::mem::replace(&mut x, self.devices[layer.device].alloc_zeros(vec![seq, s * h]));
                    device = layer.device;
                    handoffs.push((device, from, self.cudas[device].device_address(&x)));
                    recording = self.cudas[device].graph_begin();
                    assert!(recording, "a later device could not capture its layers");
                } else {
                    device = layer.device;
                    x = self.devices[device].to_device(x.to_host());
                    recording = graphs && self.cudas[device].graph_begin();
                }
            }
            let b = self.devices[device].as_ref();
            let mut prof = profile::Timer::new(b);
            if i == cfg.ple_layer {
                x = self.ple_forward(b, &x, &ple_emb, kv);
                prof.lap("ple");
            }
            let (y, post) = layer.attn_hc.mix(b, &mut x, pending.take(), s, cfg.eps);
            prof.lap("hyper");
            let y = match &layer.mixer {
                Mixer::Attn(a) => { let y = self.attention(b, a, &y, i, kv, &rope, positions); prof.lap("attention"); y }
                Mixer::Gdn(g) => { let y = self.delta_net(b, g, &y, i, kv); prof.lap("delta_net"); y }
            };
            let (y2, post2) = layer.mlp_hc.mix(b, &mut x, Some((y, post.expect("site gate"))), s, cfg.eps);
            prof.lap("hyper");
            let y2 = self.moe(b, &layer.moe, &y2);
            prof.lap("moe");
            pending = Some((y2, post2.expect("site gate")));
        }
        if let Some((y, post)) = pending.take() {
            self.devices[device].stream_apply(&mut x, &y, &post, s);
        }
        kv.commit(seq);
        let last = self.devices.len() - 1;
        if recording && device != last {
            self.cudas[device].graph_finish();
            self.run_handoffs(&mut handoffs);
            recording = false;
        }
        let b = self.devices[last].as_ref();
        let x_last = if device == last { b.slice_axis0_range(&x, seq - 1, 1) } else {
            let host = x.to_host();
            b.to_device(Tensor::from_vec(host.data()[(seq - 1) * s * h..].to_vec(), vec![1, s * h]))
        };
        let mut prof = profile::Timer::new(b);
        let mut x_last = x_last;
        let (mixed, _) = self.collapse.mix(b, &mut x_last, None, s, cfg.eps);
        let logits = self.head.linear(b, &mixed);
        if recording {
            if handoffs.is_empty() { self.cudas[last].graph_end(); } else { self.cudas[last].graph_finish(); }
        }
        self.run_handoffs(&mut handoffs);
        let logits = logits.to_host();
        if seq == 1 { self.decoded.store(true, std::sync::atomic::Ordering::Relaxed); }
        prof.lap("head");
        Ok(logits)
    }

    /// The captured devices after the first, in order: each one's input written from the
    /// previous device's output (which waits for that device), then its graph run.
    fn run_handoffs(&self, handoffs: &mut Vec<(usize, Tensor, u64)>) {
        for (device, from, to) in handoffs.drain(..) {
            self.cudas[device].write_at(to, from.to_host().data());
            self.cudas[device].graph_launch();
        }
    }

    /// Before a captured step: room in the caches for `total` tokens (a cache grown mid-capture
    /// would live in the step's scratch), and the step's rope positions on every device.
    fn prepare_step(&self, kv: &mut KvCache, rope: &[u32], total: usize) {
        let cfg = &self.config;
        for (i, layer) in self.layers.iter().enumerate() {
            let b = self.devices[layer.device].as_ref();
            if let Mixer::Attn(a) = &layer.mixer {
                kv.reserve_layer(b, i, total);
                kv.reserve_layer(b, a.index_slot, total);
            }
            // Recurrent state restored from a checkpoint may still be on the host.
            let slots = if i == cfg.ple_layer { vec![(i, true), (ple_slot(cfg), false)] } else { vec![(i, true)] };
            for (slot, with_state) in slots {
                for (t, keep) in [(&mut kv.ssm_conv[slot], true), (&mut kv.ssm_state[slot], with_state)] {
                    if let Some(v) = t.as_mut().filter(|v| keep && v.is_cpu()) { *v = b.to_device(v.clone()); }
                }
            }
        }
        // Past the dense span, each index block is roped at its start as well.
        let ratio = cfg.index_ratio;
        let blocks_kept = cfg.index_budget / ratio;
        let starts: Vec<u32> = (0..total / ratio).map(|j| (j * ratio) as u32).collect();
        for c in &self.cudas {
            c.prime_rope_positions(rope);
            if total > blocks_kept * ratio + ratio - 1 {
                c.prime_rope_positions(&starts);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, b: &dyn Backend, a: &Attn, x: &Tensor, layer: usize, kv: &mut KvCache, rope: &[u32], positions: Option<&[[u32; 3]]>) -> Tensor {
        let cfg = &self.config;
        let seq = x.dim(0);
        let (nh, nkv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let q_full = a.q.linear(b, x);
        let k = a.k.linear(b, x).reshape(vec![seq, nkv, hd]).expect("k");
        let v = a.v.linear(b, x).reshape(vec![seq, nkv, hd]).expect("v");
        let (q, gate) = b.split_q_and_gate(&q_full, nh, hd);
        let q = q.reshape(vec![seq, nh, hd]).expect("q");
        let mut q = ggml_rs::ops::rmsnorm(b, &q, &a.q_norm, cfg.eps);
        let mut k = ggml_rs::ops::rmsnorm(b, &k, &a.k_norm, cfg.eps);
        if let Some(p) = positions {
            llama_rs::multimodal_rope::text(b, &mut q, p, cfg.rope_dim, cfg.rope_theta);
            llama_rs::multimodal_rope::text(b, &mut k, p, cfg.rope_dim, cfg.rope_theta);
        } else {
            b.rope_partial_neox(&mut q, rope, hd, cfg.rope_dim, cfg.rope_theta);
            b.rope_partial_neox(&mut k, rope, hd, cfg.rope_dim, cfg.rope_theta);
        }
        let past = kv.len;
        let total = past + seq;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut prof = profile::Timer::new(b);
        // The indexer's raw keys are kept for every token: a later, longer step selects over them.
        let (ih, id, ratio) = (cfg.index_heads, cfg.index_dim, cfg.index_ratio);
        let (index_q, raw_k) = b.split_cols(&a.index_qk.linear(b, x), ih * id);
        let raw_k = raw_k.reshape(vec![seq, 1, id]).expect("raw keys");
        kv.append(b, a.index_slot, &raw_k, &raw_k);
        kv.append(b, layer, &k, &v);
        // Every query up to here sees at most the budget's blocks: all of them, i.e. dense.
        let blocks_kept = cfg.index_budget / ratio;
        prof.lap("attn:proj");
        let out = if total <= blocks_kept * ratio + ratio - 1 {
            ggml_rs::ops::attention(b, &q, kv.k_buffer(layer), kv.v_buffer(layer), total, scale, past)
        } else {
            let index_q = index_q.reshape(vec![seq, ih, id]).expect("index queries");
            let mut index_q = ggml_rs::ops::rmsnorm(b, &index_q, &a.index_q_norm, cfg.eps);
            b.rope_partial_neox(&mut index_q, rope, id, cfg.rope_dim, cfg.rope_theta);
            // Complete blocks of raw keys: mean, norm, rope at the block's start.
            let nb = total / ratio;
            // (The pool reads the first nb * ratio keys of the cache in place.)
            let pooled = b.qsa_pool(kv.k_buffer(a.index_slot), nb, ratio).reshape(vec![nb, 1, id]).expect("pooled");
            let mut pooled = ggml_rs::ops::rmsnorm(b, &pooled, &a.index_k_norm, cfg.eps);
            let starts: Vec<u32> = (0..nb).map(|j| (j * ratio) as u32).collect();
            b.rope_partial_neox(&mut pooled, &starts, id, cfg.rope_dim, cfg.rope_theta);
            let pooled = pooled.reshape(vec![nb, id]).expect("pooled");
            let scores = b.qsa_block_scores(&index_q, &pooled, past, ratio, 1.0 / (id as f32).sqrt());
            // Each query: its top blocks, then always its incomplete tail block.
            let sel = b.qsa_select(&scores, past, ratio, blocks_kept);
            prof.lap("attn:select");
            let out = b.sparse_attention(&q, kv.k_buffer(layer), kv.v_buffer(layer), &sel, scale);
            prof.lap("attn:sparse");
            out
        };
        let mut out = out.reshape(vec![seq, nh * hd]).expect("attn");
        b.mul_sigmoid_inplace(&mut out, &gate);
        a.o.linear(b, &out)
    }

    fn delta_net(&self, b: &dyn Backend, g: &Gdn, x: &Tensor, layer: usize, kv: &mut KvCache) -> Tensor {
        let cfg = &self.config;
        let seq = x.dim(0);
        let qkv = g.qkv.linear(b, x);
        let z = g.z.linear(b, x);
        let ba = g.ba.linear(b, x);
        let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
        let mut state = kv.ssm_state[layer].take().unwrap_or_else(|| b.to_device(Tensor::zeros(vec![cfg.nv, cfg.vd, cfg.vd])));
        let mut conv = kv.ssm_conv[layer].take().unwrap_or_else(|| b.to_device(Tensor::zeros(vec![cfg.conv - 1, conv_dim])));
        if state.is_cpu() { state = b.to_device(state); }
        if conv.is_cpu() { conv = b.to_device(conv); }
        let out = b.delta_net_step_sigmoid(&qkv, &z, &ba, &g.conv, &g.a, &g.dt_bias, &g.norm, &mut conv, &mut state,
            seq, cfg.nv, cfg.nk, cfg.vd, cfg.kd, cfg.nv / cfg.nk, 1.0 / (cfg.vd as f32).sqrt(), cfg.eps);
        kv.ssm_state[layer] = Some(state);
        kv.ssm_conv[layer] = Some(conv);
        g.out.linear(b, &out)
    }

    /// Routed experts (grouped by expert) plus the gated shared expert.
    fn moe(&self, b: &dyn Backend, m: &Moe, x: &Tensor) -> Tensor {
        let cfg = &self.config;
        let logits = m.router.linear(b, x);
        m.experts.forward(x, &logits, cfg.top_k)
    }

    /// The hashed n-gram features (`[positions, ple_dim]`) of `history`'s positions past its
    /// `ngram - 1` leading context ids: each position's bigram then trigram hash heads' rows.
    fn ngram_embedding(&self, history: &[i64]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let t = &self.ple.table;
        let ctx = cfg.ngram - 1;
        let eos = cfg.ple_eos as i64;
        let n = history.len();
        // Shifted ids: n-grams never span an eos (a position past one reads eos instead).
        let mut prev_eos = vec![-1i64; n];
        let mut last = -1i64;
        for p in 0..n {
            prev_eos[p] = last;
            if history[p] == eos { last = p as i64; }
        }
        let shifted = |p: usize, shift: usize| -> i64 {
            if shift == 0 { return history[p]; }
            let in_segment = p as i64 - (prev_eos[p] + 1);
            if in_segment >= shift as i64 && p >= shift { history[p - shift] } else { eos }
        };
        let seq = n - ctx;
        // Each position's rows, in head order: they fill `emb` row after row.
        let mut rows = Vec::with_capacity(seq * cfg.ple_dim / ROW_DIM);
        for r in 0..seq {
            let p = ctx + r;
            for ngram in 2..=cfg.ngram {
                let mut mixed = shifted(p, 0).wrapping_mul(t.multipliers[0]);
                for position in 1..ngram {
                    mixed ^= shifted(p, position).wrapping_mul(t.multipliers[position]);
                }
                for j in 0..cfg.heads_per_ngram {
                    let head = (ngram - 2) * cfg.heads_per_ngram + j;
                    rows.push(((mixed.rem_euclid(t.head_sizes[head]) + t.head_offsets[head]) as u64, head));
                }
            }
        }
        // Random reads from a table too big to keep in memory: in parallel, so a cold one
        // (from disk) waits alongside the others instead of after them.
        use rayon::prelude::*;
        let mut emb = vec![0f32; seq * cfg.ple_dim];
        emb.par_chunks_mut(ROW_DIM).zip(rows.par_iter()).try_for_each(|(out, &(row, head))| t.row(row, head, out))?;
        Ok(emb)
    }

    /// The n-gram layer's features for `tokens` (`[tokens, ple_dim]`, on `b`), carrying its
    /// context (the last `ngram - 1` ids) on in the cache.
    fn ple_embed(&self, b: &dyn Backend, tokens: &[u32], kv: &mut KvCache) -> Result<Tensor> {
        let cfg = &self.config;
        let ctx = cfg.ngram - 1;
        let slot = ple_slot(cfg);
        let eos = cfg.ple_eos as i64;
        // The carried context: the last ngram-1 ids (eos at a sequence start).
        let mut history: Vec<i64> = match &kv.ssm_state[slot] {
            Some(t) => t.to_host().data().iter().map(|&v| v as i64).collect(),
            None => vec![eos; ctx],
        };
        history.extend(tokens.iter().map(|&t| t as i64));
        let n = history.len();
        let emb = self.ngram_embedding(&history)?;
        kv.ssm_state[slot] = Some(Tensor::from_vec(history[n - ctx..].iter().map(|&v| v as f32).collect(), vec![ctx]));
        Ok(b.to_device(Tensor::from_vec(emb, vec![tokens.len(), cfg.ple_dim])))
    }

    /// The n-gram layer: the streams plus their gate against the n-gram features `emb`, and
    /// the dilated conv over that (its window carried in the cache, updated in place).
    fn ple_forward(&self, b: &dyn Backend, x: &Tensor, emb: &Tensor, kv: &mut KvCache) -> Tensor {
        let cfg = &self.config;
        let (h, s) = (cfg.hidden, cfg.streams);
        let slot = ple_slot(cfg);
        let ple = &self.ple;
        let key = ple.key.linear(b, emb);
        let value = ple.value.linear(b, emb);
        let (gated, conv_in) = b.ple_gate(&key, x, &value, &ple.norm_key, &ple.norm_query, &ple.norm_conv, s, cfg.eps);
        let state_len = (cfg.ple_kernel - 1) * cfg.ngram;
        let mut window = match kv.ssm_conv[slot].take() {
            Some(t) if !t.is_cpu() => t,
            Some(t) => b.to_device(t),
            None => b.to_device(Tensor::zeros(vec![state_len, s * h])),
        };
        let mut out = x.clone();
        b.ple_conv(&mut out, &gated, &conv_in, &mut window, &ple.conv, cfg.ple_kernel, cfg.ngram);
        kv.ssm_conv[slot] = Some(window);
        out
    }
}

/// Whether decode steps run as graphs (FLASHNEXT_GRAPHS=0 turns them off).
fn graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Graphs are CUDA's: a build without it runs every step uncaptured.
    *ON.get_or_init(|| cfg!(feature = "cuda") && std::env::var("FLASHNEXT_GRAPHS").map_or(true, |v| v != "0"))
}

/// Where a forward's time goes, when `FLASHNEXT_PROFILE` is set: each part synchronizes the
/// GPU at its end (so it slows the run), and the totals print every 64 forwards.
mod profile {
    use ggml_rs::Backend;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static TOTALS: Mutex<(Vec<(&'static str, Duration)>, usize)> = Mutex::new((Vec::new(), 0));
    pub fn on() -> bool { *ON.get_or_init(|| std::env::var_os("FLASHNEXT_PROFILE").is_some()) }
    /// Print the totals so far, per forward, and start over.
    #[allow(dead_code)]
    pub fn print(what: &str) {
        if !on() { return; }
        let mut t = TOTALS.lock().unwrap();
        let n = t.1.max(1) as f64;
        let all: Duration = t.0.iter().map(|e| e.1).sum();
        eprintln!("flash-next profile, {what}: {} forwards ({:.1} ms each)", t.1, all.as_secs_f64() * 1e3 / n);
        for (p, d) in &t.0 { eprintln!("  {p:12} {:8.2} ms/forward", d.as_secs_f64() * 1e3 / n); }
        *t = (Vec::new(), 0);
    }
    /// Start the totals over (e.g. between prefill and decode).
    #[allow(dead_code)]
    pub fn reset() { *TOTALS.lock().unwrap() = (Vec::new(), 0); }
    pub struct Timer<'a> { b: &'a dyn Backend, t: Instant }
    impl<'a> Timer<'a> {
        pub fn new(b: &'a dyn Backend) -> Self { if on() { b.synchronize(); } Self { b, t: Instant::now() } }
        pub fn lap(&mut self, part: &'static str) {
            if !on() { return; }
            self.b.synchronize();
            let now = Instant::now();
            let mut t = TOTALS.lock().unwrap();
            match t.0.iter_mut().find(|(p, _)| *p == part) { Some(e) => e.1 += now - self.t, None => t.0.push((part, now - self.t)) }
            if part == "head" {
                t.1 += 1;
                if t.1 % 64 == 0 {
                    let all: Duration = t.0.iter().map(|e| e.1).sum();
                    eprintln!("flash-next profile over {} forwards ({:.1} ms each):", t.1, all.as_secs_f64() * 1e3 / t.1 as f64);
                    for (p, d) in &t.0 { eprintln!("  {p:10} {:7.2} ms/forward", d.as_secs_f64() * 1e3 / t.1 as f64); }
                }
            }
            self.t = now;
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// Against exllamav3 on the same checkpoint (see exl3-ref/reference.py): FLASHNEXT_MODEL,
    /// FLASHNEXT_REFERENCE (its reference.json), FLASHNEXT_DEVICES (e.g. "0,1").
    #[test]
    #[ignore = "needs the checkpoint, two GPUs and a reference.json"]
    fn matches_the_reference() {
        let path = std::env::var("FLASHNEXT_MODEL").expect("FLASHNEXT_MODEL");
        let reference = Json::parse(&std::fs::read(std::env::var("FLASHNEXT_REFERENCE").expect("FLASHNEXT_REFERENCE")).unwrap()).unwrap();
        let devices: Vec<usize> = std::env::var("FLASHNEXT_DEVICES").unwrap_or_else(|_| "0,1".into()).split(',').map(|d| d.trim().parse().unwrap()).collect();
        let list = |key: &str| reference.get(key).and_then(Json::as_array).unwrap().to_vec();
        let ids: Vec<u32> = list("ids").iter().map(|v| v.as_f64().unwrap() as u32).collect();
        let top: Vec<Vec<u32>> = list("top_ids").iter().map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as u32).collect()).collect();
        let top_logits: Vec<Vec<f32>> = list("top_logits").iter().map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()).collect();
        let t = std::time::Instant::now();
        let model = load(Path::new(&path), &devices).unwrap();
        eprintln!("loaded in {:.1}s; {} tokens", t.elapsed().as_secs_f64(), ids.len());

        // The n-gram features of the first positions.
        let eos = model.config.ple_eos as i64;
        let mut history = vec![eos; model.config.ngram - 1];
        history.extend(ids.iter().map(|&t| t as i64));
        let emb = model.ngram_embedding(&history).unwrap();
        let want: Vec<Vec<f32>> = list("ngram_embedding_first").iter().map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()).collect();
        let dim = model.config.ple_dim;
        let worst = want.iter().enumerate().flat_map(|(r, row)| row.iter().enumerate().map(move |(i, w)| (r, i, *w)))
            .map(|(r, i, w)| (emb[r * dim + i] - w).abs()).fold(0f32, f32::max);
        eprintln!("n-gram features: largest difference {worst:.5}");

        let compare = |pos: usize, logits: &Tensor| -> (bool, f32) {
            let l = logits.data();
            let mine = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
            let diff = top[pos].iter().zip(&top_logits[pos]).map(|(&i, &v)| (l[i as usize] - v).abs()).fold(0f32, f32::max);
            (mine == top[pos][0], diff)
        };
        // Token by token: every position's prediction (a long prompt: prefill all but its last
        // few, then those one at a time).
        let mut kv = model.new_kv_cache(ids.len() + 8);
        let (mut agree, mut worst) = (0, 0f32);
        let first = if ids.len() > 256 { ids.len() - 8 } else { 0 };
        let t = std::time::Instant::now();
        for chunk in ids[..first].chunks(512) {
            let e = model.embed_text(chunk).unwrap();
            model.forward(chunk, &e, &mut kv, None).unwrap();
        }
        eprintln!("prefilled {first} tokens in {:.1}s", t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        for (pos, &id) in ids.iter().enumerate().skip(first) {
            let e = model.embed_text(&[id]).unwrap();
            let logits = model.forward(&[id], &e, &mut kv, None).unwrap();
            let (same, diff) = compare(pos, &logits);
            agree += same as usize;
            worst = worst.max(diff);
            if !same || pos < 3 { eprintln!("  position {pos}: top-1 {} (reference {}), top-5 logit difference {diff:.3}", if same { "same" } else { "DIFFERENT" }, top[pos][0]); }
        }
        let n = ids.len() - first;
        eprintln!("token by token: top-1 agrees at {agree}/{n} positions; largest top-5 logit difference {worst:.3}; {:.1} tokens/s", n as f64 / t.elapsed().as_secs_f64());
        // All at once (prefill chunks): the last prediction.
        if std::env::var("FLASHNEXT_SKIP_PREFILL").is_ok() { assert!(agree * 10 >= n * 9, "differs from the reference"); return; }
        let mut kv = model.new_kv_cache(ids.len() + 8);
        let t = std::time::Instant::now();
        let mut logits = None;
        for chunk in ids.chunks(512) {
            let e = model.embed_text(chunk).unwrap();
            logits = Some(model.forward(chunk, &e, &mut kv, None).unwrap());
        }
        let (same, diff) = compare(ids.len() - 1, &logits.unwrap());
        eprintln!("prefill: last top-1 {}, top-5 logit difference {diff:.3}; {:.0} tokens/s", if same { "same" } else { "DIFFERENT" }, ids.len() as f64 / t.elapsed().as_secs_f64());
        assert!(agree * 10 >= n * 9 && same, "differs from the reference");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod bench {
    use super::*;
    /// Prefill and decode speed (FLASHNEXT_MODEL, FLASHNEXT_REFERENCE for its prompt ids,
    /// FLASHNEXT_DEVICES; FLASHNEXT_PROFILE for where the time goes).
    #[test]
    #[ignore = "needs the checkpoint and two GPUs"]
    fn speed() {
        let path = std::env::var("FLASHNEXT_MODEL").expect("FLASHNEXT_MODEL");
        let reference = Json::parse(&std::fs::read(std::env::var("FLASHNEXT_REFERENCE").expect("FLASHNEXT_REFERENCE")).unwrap()).unwrap();
        let devices: Vec<usize> = std::env::var("FLASHNEXT_DEVICES").unwrap_or_else(|_| "0,1".into()).split(',').map(|d| d.trim().parse().unwrap()).collect();
        let ids: Vec<u32> = reference.get("ids").and_then(Json::as_array).unwrap().iter().map(|v| v.as_f64().unwrap() as u32).collect();
        let model = load(Path::new(&path), &devices).unwrap();
        let n: usize = std::env::var("FLASHNEXT_DECODE").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let mut kv = model.new_kv_cache(ids.len() + n + 8);
        let t = std::time::Instant::now();
        let mut logits = None;
        for chunk in ids.chunks(512) {
            let e = model.embed_text(chunk).unwrap();
            logits = Some(model.forward(chunk, &e, &mut kv, None).unwrap());
        }
        eprintln!("prefill: {} tokens in {:.2}s ({:.0} tokens/s)", ids.len(), t.elapsed().as_secs_f64(), ids.len() as f64 / t.elapsed().as_secs_f64());
        profile::print("prefill");
        let mut logits = logits.unwrap();
        profile::reset();
        let t = std::time::Instant::now();
        let mut out = Vec::new();
        for _ in 0..n {
            let l = logits.data();
            let next = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
            out.push(next);
            let e = model.embed_text(&[next]).unwrap();
            logits = model.forward(&[next], &e, &mut kv, None).unwrap();
        }
        eprintln!("decode: {n} tokens in {:.2}s ({:.1} tokens/s): {:?}", t.elapsed().as_secs_f64(), n as f64 / t.elapsed().as_secs_f64(), model.tokenizer.decode(&out));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod unload_tests {
    fn used() -> String {
        let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader"]).output().unwrap();
        String::from_utf8_lossy(&out.stdout).replace('\n', " ")
    }
    /// A dropped model's GPU memory is free for the next (FLASHNEXT_UNLOAD_MODEL: an OrcaSAQ checkpoint).
    #[test]
    #[ignore = "needs a checkpoint and a GPU"]
    fn unloading_frees_the_gpu() {
        let path = std::env::var("FLASHNEXT_UNLOAD_MODEL").expect("FLASHNEXT_UNLOAD_MODEL");
        let model = crate::orcasaq::load(std::path::Path::new(&path), &[0]).unwrap();
        eprintln!("loaded: {}", used());
        // Dropped where engines drop their models: another thread, with no context bound.
        std::thread::spawn(move || drop(model)).join().unwrap();
        eprintln!("dropped on another thread: {}", used());
        let first = |s: String| s.split_whitespace().next().unwrap().parse::<u64>().unwrap();
        assert!(first(used()) < 2048, "the model's GPU memory was not released");
    }
}
