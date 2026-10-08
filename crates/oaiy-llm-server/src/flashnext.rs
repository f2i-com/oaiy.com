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

/// A build without CUDA has no cards to capture on: nothing asks it to (`graphs_enabled` is false).
pub type Card = NoCard;

pub struct NoCard;
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

#[path = "flashnext_gguf.rs"]
pub(crate) mod gguf_file;

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
        cfg.checked()
    }

    /// The dimensions, where this implementation runs them.
    fn checked(self) -> Result<Self> {
        let cfg = self;
        if cfg.attention.len() != cfg.layers || cfg.ple_layer >= cfg.layers || cfg.heads % cfg.kv_heads != 0
            || cfg.nv % cfg.nk != 0 || cfg.kd != cfg.vd || cfg.streams != 4 || cfg.ngram < 2 || cfg.rope_dim > cfg.head_dim
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
    /// A trellis row's bits a value; 0 for a GGUF's table (IQ4_NL rows, no head bias and no codebook).
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

/// The bytes of a GGUF n-gram table's row: `ROW_DIM` values as IQ4_NL blocks of 32 (18 bytes each).
const IQ4_NL_ROW: usize = ROW_DIM / 32 * 18;

/// The most u16 words an n-gram table's row takes (its scale, then `ROW_DIM` codes of up to 8 bits).
const ROW_WORDS_MAX: usize = 1 + ROW_DIM * 8 / 16;

/// A trellis row of the n-gram table (`bytes`: its f16 scale, then `ROW_DIM` codes of `k` bits, each value's 16-bit
/// state its own code and those of the `16 / k` before it, the row a ring) decoded into `out` with its head's `bias`:
/// each code read once, each state its codes shifted together (where each state's 16 bits were read one at a time).
fn decode_row(bytes: &[u8], k: usize, codebook: &[f32], bias: &[f32], out: &mut [f32]) {
    let word = |i: usize| bytes.get(2 * i..2 * i + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])) as u32;
    let scale = dsv41::formats::f16_to_f32(word(0) as u16);
    let mask = (1u32 << k) - 1;
    let mut codes = [0u32; ROW_DIM];
    for (j, c) in codes.iter_mut().enumerate() {
        let b = j * k;
        let w = word(1 + b / 16) | (word(2 + b / 16) << 16);
        *c = (w >> (b % 16)) & mask;
    }
    let groups = 16usize.div_ceil(k);
    for (i, o) in out.iter_mut().enumerate().take(ROW_DIM) {
        let mut state = 0u32;
        for g in 0..groups {
            state |= codes[(i + ROW_DIM - g) % ROW_DIM] << (g * k);
        }
        *o = codebook[(state & 0xffff) as usize] * scale + bias[i];
    }
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
        if 1 + ROW_DIM * bits / 16 != row_words || !(1..=8).contains(&bits) {
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
        // (a GGUF's table: each row five IQ4_NL blocks, its values as they are)
        if self.bits == 0 {
            let mut buf = [0u8; IQ4_NL_ROW];
            let file = &self.files[rayon::current_thread_index().unwrap_or(0) % self.files.len()];
            read_at(file, &mut buf, self.start + row * IQ4_NL_ROW as u64)?;
            ggml_quants::iq4_nl::dequantize(&buf, &mut out[..ROW_DIM]);
            return Ok(());
        }
        // (a row's 62 bytes or so: on the stack)
        let mut buf = [0u8; 2 * ROW_WORDS_MAX];
        let bytes = &mut buf[..self.row_words * 2];
        let file = &self.files[rayon::current_thread_index().unwrap_or(0) % self.files.len()];
        read_at(file, bytes, self.start + row * (self.row_words * 2) as u64)?;
        decode_row(bytes, self.bits, &self.codebook, &self.bias[head * ROW_DIM..(head + 1) * ROW_DIM], out);
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

/// The token embeddings, a token's row read from the file as it is needed.
enum Embed {
    /// A checkpoint's own floats: the file, the table's first byte and its type.
    Plain(File, u64, Dtype),
    /// A GGUF's quantized rows: the file, the table's first byte, its type and a row's bytes.
    Quant(File, u64, ggml_quants::GgmlType, usize),
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
    embed: Embed,
    layers: Vec<Layer>,
    ple: Ple,
    collapse: HyperMix,
    head: Weight,
    pub devices: Vec<Arc<dyn Backend>>,
    /// The same GPUs, for what uploads EXL3 matrices (the vision tower) and the decode graphs; none without CUDA.
    pub cudas: Vec<Arc<Card>>,
    /// Whether a decode step has run (uncaptured), making the scratch its matrices keep.
    decoded: std::sync::atomic::AtomicBool,
    /// A decode step chained on the devices (WebGPU), made at the first step that can use one.
    chain: std::sync::OnceLock<Option<FnChain>>,
    /// Its multi-token-prediction layer, where asked for ([`load_portable`]'s `mtp`): drafts a chained check takes.
    mtp: Option<FnMtp>,
    /// The positions of the run a chain is recording, where its rows are not the next ones in order (a picture's
    /// tokens, and the text after one): one a row, in three axes ([`Self::forward`] sets and clears it).
    pub(crate) placed: std::sync::Mutex<Option<Vec<[u32; 3]>>>,
    /// Whether the chained run a caller is starting reads its rows' picks (each row's largest logit's token, picked
    /// on the GPU) where it would read their logits: set and cleared around that run ([`Self::check_picks`]).
    pick: std::sync::atomic::AtomicBool,
}

/// Qwen3.8-Flash-Next's multi-token-prediction layer (the checkpoint's `mtp.*`, on the last device with the head): the
/// next token's embedding and the trunk's four streams (after its last layer, before its collapse), each normed (the
/// streams as one row) and projected (the streams one by one), the embedding's projection added to every stream; an
/// attention block with a cache of its own and a MoE block, each behind its hyper-connection; its own collapse, then the
/// trunk's head.
struct FnMtp {
    enorm: Tensor,
    hnorm: Tensor,
    fc_e: Weight,
    fc_h: Weight,
    attn_hc: HyperMix,
    mlp_hc: HyperMix,
    mixer: HyperMix,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Tensor,
    k_norm: Tensor,
    /// The router, then the shared expert's gate, `[experts + 1, hidden]`.
    router: Tensor,
    experts: Box<dyn Experts>,
}

/// The EXL3 matrix `name` (`k -> n`, mul1 codebook) on `backend`, its input and output
/// channels optionally reordered.
/// A dense `rows x cols` matrix: kept as f16 (half the memory and reading) when every value is
/// one, as the checkpoint's f16 weights are; f32 otherwise.
/// Without a card, f32 on `backend`.
fn half_or_dense(card: Option<&Arc<Card>>, backend: &Arc<dyn Backend>, values: Vec<f32>, rows: usize, cols: usize) -> Weight {
    let _ = card;
    Weight::Dense(backend.to_device(Tensor::from_vec(values, vec![rows, cols])))
}

/// The EXL3 matrix `name` (`k -> n`), read and checked, on the host.
pub(crate) fn exl3_data(idx: &StIndex, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>) -> Result<Exl3Data> {
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

/// The multi-token-prediction layer of the EXL3 checkpoint `idx` indexes (its `mtp.*`: an attention block and a MoE
/// block behind their hyper-connections, the projections either side), on `device` as `last` makes its matrices and
/// `make_experts` its experts: None where the checkpoint has none.
fn mtp_layer(idx: &StIndex, cfg: &Config, last: &Loader<'_>, device: usize, make_experts: ExpertMaker<'_>) -> Result<Option<FnMtp>> {
    if idx.get("mtp.fc_embedding.trellis").is_none() {
        return Ok(None);
    }
    let (h, width) = (cfg.hidden, cfg.streams * cfg.hidden);
    let (a, m) = ("mtp.layers.0.self_attn", "mtp.layers.0.mlp");
    let names: Vec<String> = (0..cfg.experts).map(|e| format!("{m}.experts.{e}")).chain([format!("{m}.shared_expert")]).collect();
    let read = |p: &String| -> Result<[Exl3Data; 3]> {
        Ok([
            exl3_data(idx, &format!("{p}.gate_proj"), h, cfg.moe_ff, None, None)?,
            exl3_data(idx, &format!("{p}.up_proj"), h, cfg.moe_ff, None, None)?,
            exl3_data(idx, &format!("{p}.down_proj"), cfg.moe_ff, h, None, None)?,
        ])
    };
    let per = names.len().div_ceil(16);
    let experts: Vec<[Exl3Data; 3]> = std::thread::scope(|scope| {
        let parts: Vec<_> = names.chunks(per).map(|part| scope.spawn(|| part.iter().map(read).collect::<Result<Vec<_>>>())).collect();
        parts.into_iter().map(|t| t.join().expect("expert reader")).collect::<Result<Vec<Vec<_>>>>()
    })?.into_iter().flatten().collect();
    let mut router = last.host(&format!("{m}.gate.weight"), cfg.experts * h, false, None)?;
    router.extend(last.host(&format!("{m}.shared_expert_gate.weight"), h, false, None)?);
    Ok(Some(FnMtp {
        enorm: last.tensor("mtp.pre_fc_norm_embedding.weight", &[h], true, None)?,
        hnorm: last.tensor("mtp.pre_fc_norm_hidden.weight", &[width], true, None)?,
        fc_e: last.weight("mtp.fc_embedding", h, h, None, None)?,
        fc_h: last.weight("mtp.fc_hidden", h, h, None, None)?,
        attn_hc: last.hyper("mtp.layers.0.attn_hyper_connection", cfg, true)?,
        mlp_hc: last.hyper("mtp.layers.0.mlp_hyper_connection", cfg, true)?,
        mixer: last.hyper("mtp.hyper_connection_mixer", cfg, false)?,
        q: last.weight(&format!("{a}.q_proj"), h, 2 * cfg.heads * cfg.head_dim, None, None)?,
        k: last.weight(&format!("{a}.k_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
        v: last.weight(&format!("{a}.v_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
        o: last.weight(&format!("{a}.o_proj"), cfg.heads * cfg.head_dim, h, None, None)?,
        q_norm: last.tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim], true, None)?,
        k_norm: last.tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim], true, None)?,
        router: Tensor::from_vec(router, vec![cfg.experts + 1, h]),
        experts: make_experts(device, m, experts)?,
    }))
}

/// The bytes of Flash-Next's EXL3 matrices outside its experts (attention, delta-net, the head): a portable build keeps
/// that much of the GPU budget for them, where the experts, loaded first, took it all and left the 248k-row head to the
/// CPU.
pub(crate) fn dense_exl3_bytes(path: &Path) -> Result<u64> {
    let idx = StIndex::open(path)?;
    Ok(idx.names().filter(|n| reserved(n)).filter_map(|n| idx.get(n).map(|i| i.nbytes)).sum())
}

/// Whether a tensor is one of the matrices `dense_exl3_bytes` keeps GPU budget for. Not the experts, and not the n-gram
/// table: its rows are trellis-quantized too, but it is read from the disk as needed and never placed on the GPU, and
/// counted (32.6 GB) it left the experts none of a 27 GiB budget.
fn reserved(name: &str) -> bool {
    (name.ends_with(".trellis") && name.starts_with("model.language_model.") || name == "lm_head.trellis")
        && !name.contains(".experts.")
        && !name.contains(".shared_expert.")
        && !name.contains(".ngram_embedding.")
}

#[cfg(test)]
mod reserve_tests {
    #[test]
    fn the_reserve_counts_the_dense_matrices_and_the_head_not_the_experts_or_the_ngram_table() {
        let p = "model.language_model.layers.1";
        for name in [format!("{p}.self_attn.q_proj.trellis"), format!("{p}.linear_attn.in_proj_qkv.trellis"), "lm_head.trellis".into()] {
            assert!(super::reserved(&name), "{name}");
        }
        for name in [
            format!("{p}.mlp.experts.7.gate_proj.trellis"),
            format!("{p}.mlp.shared_expert.down_proj.trellis"),
            format!("{p}.ple.ple_embedding.ngram_embedding.shard_0.trellis"),
            format!("{p}.self_attn.q_proj.suh"),
        ] {
            assert!(!super::reserved(&name), "{name}");
        }
    }
}

/// Flash-Next without CUDA: its layers split over `backends` (a whole layer, its experts too, on each; the head on
/// the last), each EXL3 matrix as `packed` makes it on its device and each layer's experts as `experts` does (on any
/// GPU through WebGPU, else on the CPU), no PEFT adapters, a decode step uncaptured. (A CUDA build loads it on the
/// cards.)
/// `mtp`: its multi-token-prediction layer too (on the last device), for drafting.
pub(crate) fn load_portable(path: &Path, backends: Vec<Arc<dyn Backend>>, packed: Packer<'_>, experts: ExpertMaker<'_>, mtp: bool) -> Result<FlashNext> {
    load_portable_with(path, backends, &[], packed, experts, mtp)
}

/// [`load_portable`] with `lora`'s adapters applied, together, as the weights load: beside each dense projection
/// they adapt, on its device. An adapter that also adapts the routed experts is refused (its targets there are
/// left over when the model is built: `experts` makes them as the checkpoint has them).
pub(crate) fn load_portable_with(path: &Path, backends: Vec<Arc<dyn Backend>>, lora: &[Adapter], packed: Packer<'_>, experts: ExpertMaker<'_>, mtp: bool) -> Result<FlashNext> {
    build(path, backends, Vec::new(), lora, packed, experts, mtp)
}

#[allow(clippy::too_many_arguments)]
fn build(path: &Path, backends: Vec<Arc<dyn Backend>>, cudas: Vec<Arc<Card>>, lora: &[Adapter], packed: Packer<'_>, make_experts: ExpertMaker<'_>, mtp: bool) -> Result<FlashNext> {
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
    // The multi-token-prediction layer, where asked for and the checkpoint has one: on the last device, with the head.
    let mtp = if mtp { mtp_layer(&idx, &cfg, &last, backends.len() - 1, make_experts)? } else { None };
    // Every target of every adapter found its matrix.
    for adapter in lora { adapter.finish()?; }
    let file = File::open(idx.shard_path(embed.shard))?;
    Ok(FlashNext {
        tokenizer: tok,
        embed: Embed::Plain(file, embed.start, embed.dtype),
        layers,
        ple,
        collapse,
        head,
        devices: backends,
        cudas,
        decoded: Default::default(),
        chain: Default::default(),
        mtp,
        placed: Default::default(),
        pick: Default::default(),
        config: cfg,
    })
}

impl FlashNext {
    /// Give the model the multi-token-prediction layer of the EXL3 checkpoint at `exl3` (the same model's: a GGUF
    /// carries none, and Strata takes it from the original checkpoint likewise), on the last device with the head, its
    /// matrices what `packed` makes and its experts what `make_experts` does. Before its first chained step. False
    /// where the checkpoint has no such layer.
    pub(crate) fn attach_mtp(&mut self, exl3: &Path, packed: Packer<'_>, make_experts: ExpertMaker<'_>) -> Result<bool> {
        if !detect(exl3) {
            return Err(bad(format!("{} is not a Qwen3.8-Flash-Next EXL3 checkpoint", exl3.display())));
        }
        let theirs = Config::read(exl3)?;
        if format!("{theirs:?}") != format!("{:?}", self.config) {
            return Err(bad(format!("{} is another model's checkpoint: its MTP layer is not this one's", exl3.display())));
        }
        if self.chain.get().is_some() {
            return Err(bad("an MTP layer is attached before the model's first step"));
        }
        let idx = StIndex::open(exl3)?;
        let device = self.devices.len() - 1;
        let last = Loader { idx: &idx, backend: self.devices[device].clone(), card: self.cudas.get(device).cloned(), lora: &[], packed: packed(device) };
        self.mtp = mtp_layer(&idx, &self.config, &last, device, make_experts)?;
        Ok(self.mtp.is_some())
    }

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
        if tokens.iter().any(|&t| t as usize >= self.config.vocab) {
            return Err(bad("token outside the vocabulary"));
        }
        let (file, start, dtype) = match &self.embed {
            Embed::Plain(file, start, dtype) => (file, start, dtype),
            Embed::Quant(file, start, dtype, row) => {
                let mut out = vec![0f32; tokens.len() * h];
                let mut buf = vec![0u8; *row];
                for (&t, o) in tokens.iter().zip(out.chunks_exact_mut(h)) {
                    read_at(file, &mut buf, start + t as u64 * *row as u64)?;
                    ggml_quants::dequantize(*dtype, &buf, o).map_err(|e| bad(e.to_string()))?;
                }
                return Ok(Tensor::from_vec(out, vec![tokens.len(), h]));
            }
        };
        let size = dtype.size();
        let mut out = Vec::with_capacity(tokens.len() * h);
        let mut buf = vec![0u8; h * size];
        for &t in tokens {
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
        // VENDORED-LOCAL: a decode step or a prompt's chunk chained on the GPUs where they can (WebGPU); a run with a
        // picture in it (or after one) at the positions it is given, where Qwen's rotation of 64 lays its axes out
        // (op by op, a prompt with a picture was read at 12 tokens/s and answered at 4)
        let placed = positions.filter(|p| self.config.rope_dim == 64 && p.len() == tokens.len());
        if (positions.is_none() || placed.is_some()) && !profile::on() && tokens.len() <= self.prompt_rows() {
            *self.placed.lock().unwrap_or_else(|p| p.into_inner()) = placed.map(<[[u32; 3]]>::to_vec);
            let logits = self.forward_chained(tokens, embeds, kv);
            *self.placed.lock().unwrap_or_else(|p| p.into_inner()) = None;
            if let Some(logits) = logits {
                return Ok(logits);
            }
        }
        self.forward_host(tokens, embeds, kv, positions)
    }

    /// [`Self::forward`] op by op (and on CUDA, captured as graphs), never chained: what a chained step is checked
    /// against.
    pub fn forward_host(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, positions: Option<&[[u32; 3]]>) -> Result<Tensor> {
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
        Ok(b.to_device(Tensor::from_vec(self.ple_embed_host(tokens, kv)?, vec![tokens.len(), self.config.ple_dim])))
    }

    /// [`Self::ple_embed`]'s features on the host (`[tokens, ple_dim]`), for a chain to upload as they are.
    fn ple_embed_host(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let mut history = self.ple_history(kv);
        history.extend(tokens.iter().map(|&t| t as i64));
        let emb = self.ngram_embedding(&history)?;
        self.ple_carry(&history, kv);
        Ok(emb)
    }

    /// The n-gram layer's carried context: the last ngram-1 ids (eos at a sequence start).
    fn ple_history(&self, kv: &KvCache) -> Vec<i64> {
        let cfg = &self.config;
        match &kv.ssm_state[ple_slot(cfg)] {
            Some(t) => t.to_host().data().iter().map(|&v| v as i64).collect(),
            None => vec![cfg.ple_eos as i64; cfg.ngram - 1],
        }
    }

    /// The context `history` (the carried one, then a run's tokens) leaves in the cache for the next run.
    fn ple_carry(&self, history: &[i64], kv: &mut KvCache) {
        let ctx = self.config.ngram - 1;
        let n = history.len();
        kv.ssm_state[ple_slot(&self.config)] = Some(Tensor::from_vec(history[n - ctx..].iter().map(|&v| v as f32).collect(), vec![ctx]));
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

// ---------------------------------------------------------------------------
// A decode step chained on the GPUs (WebGPU)
// ---------------------------------------------------------------------------

/// A matrix on a chain's device: f16 two to a word where its values are (the checkpoint's f16 weights; half the bytes
/// a step reads), else f32.
struct ChainMat {
    v: ggml_rs::DeviceVec,
    half: bool,
}

impl ChainMat {
    fn new(c: &dyn ggml_rs::DeviceChain, t: &Tensor) -> Self {
        let host;
        let values = if t.is_device() {
            host = t.to_host();
            host.data()
        } else {
            t.data()
        };
        if let Some(v) = c.vec_f16(values) {
            return ChainMat { v, half: true };
        }
        let v = c.vec(values.len());
        c.upload(&v, values);
        ChainMat { v, half: false }
    }

    /// `y[r] = W x[r]` for `rows` rows (`W` `[n, k]`).
    #[allow(clippy::too_many_arguments)]
    fn mul(&self, rec: &mut dyn ggml_rs::ChainRecorder, n: usize, k: usize, x: &ggml_rs::DeviceVec, y: &ggml_rs::DeviceVec, rows: usize) {
        if self.half {
            rec.matmul_f16_rows(&self.v, n, k, x, y, rows)
        } else {
            rec.matmul_f32_rows(&self.v, n, k, x, y, rows)
        }
    }
}

/// A hyper-connection's matrices on its layer's device (`down` `[rank + writes, streams * hidden]`, `up` `[streams *
/// hidden, rank + writes]`), and its norm (`1 + w`).
impl HcVecs {
    /// The site's projections of `rows` rows of normed streams (`s` of `h`): the down matrix and the gates into `t`
    /// and `post`, the up matrix and the mix into `out`. f16 matrices take each projection with what follows it (one
    /// dispatch for a step's row or a check's few, where two: a decode step 10.5 ms where 11.2), `logits` then scratch
    /// that may stay unwritten.
    #[allow(clippy::too_many_arguments)]
    fn project(&self, rec: &mut dyn ggml_rs::ChainRecorder, normed: &ggml_rs::DeviceVec, t: &ggml_rs::DeviceVec, post: &ggml_rs::DeviceVec, logits: &ggml_rs::DeviceVec, out: &ggml_rs::DeviceVec, rows: usize, s: usize, h: usize) {
        let n = self.rank + self.writes;
        if self.down.half && self.up.half {
            rec.hc_down_gates(&self.down.v, s * h, normed, t, post, rows, self.rank, self.writes, s);
            rec.hc_up_mix(&self.up.v, n, t, logits, normed, out, rows, s, h);
        } else {
            self.down.mul(rec, n, s * h, normed, t, rows);
            rec.hc_gates(t, post, rows, self.rank, self.writes, s);
            self.up.mul(rec, s * h, n, t, logits, rows);
            rec.hc_mix(logits, normed, out, rows, s, h);
        }
    }
}

struct HcVecs {
    norm: ggml_rs::DeviceVec,
    down: ChainMat,
    up: ChainMat,
    rank: usize,
    writes: usize,
}

/// A layer's own vectors on its device.
struct ChainLayer {
    attn_hc: HcVecs,
    mlp_hc: HcVecs,
    mixer: ChainMixer,
    /// The router, then the shared expert's gate, `[experts + 1, hidden]`.
    router: ChainMat,
    /// Its experts are on the host (no GPU had room for them): the chain reads its router and its experts' input
    /// back, they run there, and their sum goes up for the layer's write-back.
    host: bool,
}

enum ChainMixer {
    /// The beta-alpha projection, the conv's weights, `A`, `dt_bias`, the output norm, and which of the recurrent
    /// states' pool is this layer's.
    Gdn { ba: ChainMat, conv: ggml_rs::DeviceVec, a: ggml_rs::DeviceVec, dt: ggml_rs::DeviceVec, norm: ggml_rs::DeviceVec, slot: usize },
    /// The per-head norms, the indexer's (QSA's query and pooled key norms), and which of its device's copy of the
    /// cache is this layer's.
    Attn { q_norm: ggml_rs::DeviceVec, k_norm: ggml_rs::DeviceVec, iq_norm: ggml_rs::DeviceVec, ik_norm: ggml_rs::DeviceVec, slot: usize },
}

/// A device's working vectors for a step (one row).
struct ChainDev {
    x: ggml_rs::DeviceVec,
    normed: ggml_rs::DeviceVec,
    t: ggml_rs::DeviceVec,
    post: ggml_rs::DeviceVec,
    post2: ggml_rs::DeviceVec,
    logits: ggml_rs::DeviceVec,
    y_in: ggml_rs::DeviceVec,
    y_out: ggml_rs::DeviceVec,
    y2_in: ggml_rs::DeviceVec,
    moe_out: ggml_rs::DeviceVec,
    router: ggml_rs::DeviceVec,
    qkv: ggml_rs::DeviceVec,
    z: ggml_rs::DeviceVec,
    ba: ggml_rs::DeviceVec,
    conv: ggml_rs::DeviceVec,
    core: ggml_rs::DeviceVec,
    qfull: ggml_rs::DeviceVec,
    q: ggml_rs::DeviceVec,
    gate: ggml_rs::DeviceVec,
    k: ggml_rs::DeviceVec,
    v: ggml_rs::DeviceVec,
    qn: ggml_rs::DeviceVec,
    kn: ggml_rs::DeviceVec,
    gated: ggml_rs::DeviceVec,
    index: ggml_rs::DeviceVec,
    table: ggml_rs::DeviceVec,
    mixed: ggml_rs::DeviceVec,
    head: ggml_rs::DeviceVec,
    /// The head's rows' picks (`ChainRecorder::argmax_rows`: four values a row).
    picks: ggml_rs::DeviceVec,
}

/// A device's copy of its attention layers' caches (row `t`: K `[kv_heads, head_dim]` then V), `cap` rows, and of their
/// raw indexer keys (`[cap, index_dim]`, QSA's pool reads them past the dense span).
struct ChainKv {
    layers: Vec<ggml_rs::DeviceVec>,
    raw: Vec<ggml_rs::DeviceVec>,
    cap: usize,
    out: ggml_rs::DeviceVec,
    /// QSA's vectors for a step's or a check's rows, made at the first past the dense span (for this `cap`).
    qsa: Option<QsaVecs>,
    /// The `KvCache::id` the rows are a copy of (0: none).
    owner: u64,
}

/// QSA's vectors on a device for runs of up to `rows` rows over a cache of `cap` positions: the indexer's queries (in,
/// normed), the pooled block keys (in, normed) and the RoPE table of the blocks' starts, the block scores, each row's
/// blocks, and the attention's output and parts.
struct QsaVecs {
    rows: usize,
    iq: ggml_rs::DeviceVec,
    iqn: ggml_rs::DeviceVec,
    pooled: ggml_rs::DeviceVec,
    pooledn: ggml_rs::DeviceVec,
    table: ggml_rs::DeviceVec,
    scores: ggml_rs::DeviceVec,
    list: ggml_rs::DeviceVec,
    out: ggml_rs::DeviceVec,
}

/// `v`'s first `len` elements as a vector of their own (the same buffer): a chain's ops take their sizes from their
/// vectors' lengths.
fn first(v: &ggml_rs::DeviceVec, len: usize) -> ggml_rs::DeviceVec {
    assert!(len <= v.len, "{len} of a vector of {}", v.len);
    ggml_rs::DeviceVec { len, inner: Arc::clone(&v.inner) }
}

/// What a chained step changes: the devices' copies of the cache, and the recurrent states the cache's tensors alias.
struct ChainMut {
    kv: Vec<ChainKv>,
    pool: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    /// The n-gram layer's conv window, aliased in the cache as the delta nets' states are.
    ple_window: ggml_rs::DeviceVec,
    /// Each device's attention scratch for a run of a few rows, grown as the positions are.
    rows_out: Vec<ggml_rs::DeviceVec>,
    /// What undoing the last check needs ([`FlashNext::rollback`]), made at the first check.
    undo: Option<Undo>,
    /// The prediction layer's cache, made at the first draft or prompt that fills it.
    mtp_kv: Option<MtpKv>,
    /// Where the trunk's streams kept for the prediction layer are: the position of the first row, and how many.
    mtp_hid: (usize, usize),
}

/// The most rows a chained run takes as a check's (each row as a step's, the experts routed on the GPU, the vectors
/// kept): the token sampled and its drafts.
pub(crate) const CHECK_ROWS: usize = 8;

/// A check undone: each delta net's state and conv window as they were before it and the check's inputs to them (its
/// rows' qkv and beta-alpha, `CHECK_ROWS` rows), in the pool's order; the n-gram layer's window before it; and the
/// check's tokens and the n-gram history before them.
struct Undo {
    backups: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    inputs: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    window: Option<ggml_rs::DeviceVec>,
    tokens: Vec<u32>,
    history: Option<Tensor>,
}

/// A run of a few rows' vectors (a check of drafts, a short chunk), kept so their bind groups are: each device's (the
/// collapse and head for every row), the n-gram layer's, and each device's attention layers' indexer keys (`[rows,
/// index_dim]` each).
struct FewSet {
    devs: Vec<ChainDev>,
    ple: Option<PleVecs>,
    keys: Vec<Vec<ggml_rs::DeviceVec>>,
    /// Each device's one row's query (a row's attention as a step's).
    q1: Vec<ggml_rs::DeviceVec>,
}

/// The multi-token-prediction layer chained (on the last device): its norms, hyper-connections and router, the
/// embedding's broadcast weights (ones), the trunk's streams of the last run's rows (a step's, a check's, a prompt's
/// last), and a pass's vectors for 1 to [`CHECK_ROWS`] rows, made at the first of each.
struct MtpChain {
    enorm: ggml_rs::DeviceVec,
    hnorm: ggml_rs::DeviceVec,
    attn_hc: HcVecs,
    mlp_hc: HcVecs,
    mixer: HcVecs,
    q_norm: ggml_rs::DeviceVec,
    k_norm: ggml_rs::DeviceVec,
    router: ChainMat,
    ones: ggml_rs::DeviceVec,
    hid: ggml_rs::DeviceVec,
    /// A draft's token, its logit and the sum of the exponentials against it (`argmax_softmax`).
    best: ggml_rs::DeviceVec,
    devs: [std::sync::OnceLock<MtpDev>; CHECK_ROWS],
}

/// A pass of the prediction layer's vectors for its rows: a device's working vectors, the next tokens' embeddings
/// (in, normed, projected), the hidden streams (in, normed), the attention's output, and one row's query.
struct MtpDev {
    dv: ChainDev,
    e: ggml_rs::DeviceVec,
    en: ggml_rs::DeviceVec,
    e2: ggml_rs::DeviceVec,
    hin: ggml_rs::DeviceVec,
    hn: ggml_rs::DeviceVec,
    att: ggml_rs::DeviceVec,
    q1: ggml_rs::DeviceVec,
}

/// The prediction layer's cache (row `t`: K `[kv_heads, head_dim]` then V), `cap` rows, the decode attention's
/// scratch for it; the cache it follows ([`KvCache::id`]), and the positions its entries are true ones for
/// (`start..valid`: from the trunk's hidden states, not drafts).
struct MtpKv {
    layer: ggml_rs::DeviceVec,
    cap: usize,
    out: ggml_rs::DeviceVec,
    owner: u64,
    start: usize,
    valid: usize,
}

/// The least probability the prediction layer gives a draft for it to be checked (Strata's `--spec-min-p`): a check's
/// rows cost, and an unlikely draft is seldom taken.
const DRAFT_MIN_P: f64 = 0.5;

/// The n-gram layer on its device: its key and value projections, its norms and its conv's weights (`[streams *
/// hidden, kernel]`).
struct ChainPle {
    key: ChainMat,
    value: ChainMat,
    norm_key: ggml_rs::DeviceVec,
    norm_query: ggml_rs::DeviceVec,
    norm_conv: ggml_rs::DeviceVec,
    conv: ggml_rs::DeviceVec,
}

/// The n-gram layer's working vectors for `rows` rows: the features, the key and value, the gated streams and the
/// conv's input.
struct PleVecs {
    emb: ggml_rs::DeviceVec,
    key: ggml_rs::DeviceVec,
    value: ggml_rs::DeviceVec,
    gated: ggml_rs::DeviceVec,
    conv_in: ggml_rs::DeviceVec,
}

fn ple_vecs(c: &dyn ggml_rs::DeviceChain, cfg: &Config, rows: usize) -> PleVecs {
    let width = cfg.streams * cfg.hidden;
    PleVecs { emb: c.vec(rows * cfg.ple_dim), key: c.vec(rows * width), value: c.vec(rows * cfg.hidden), gated: c.vec(rows * width), conv_in: c.vec(rows * width) }
}

pub(crate) struct FnChain {
    layers: Vec<ChainLayer>,
    devs: Vec<ChainDev>,
    /// Each device's attention layers, in its copy's order.
    attn_of: Vec<Vec<usize>>,
    /// The delta-net layers, in the pool's order.
    gdn: Vec<usize>,
    collapse: HcVecs,
    /// The widest low-rank gate's rank (the scratch's width).
    rank: usize,
    /// Each device's attention layers' indexer keys of a step, copied out as each layer makes its own (a step's
    /// layers on a device run as one submit, and its indexer vector is every layer's).
    keys: Vec<ggml_rs::DeviceVec>,
    /// The n-gram layer chained, where its matrices are dense (else it runs through the host), and a step's vectors.
    ple: Option<(ChainPle, PleVecs)>,
    /// The vectors of runs of 2 to [`CHECK_ROWS`] rows, made at the first of each.
    few: [std::sync::OnceLock<FewSet>; CHECK_ROWS - 1],
    /// A tapped run's later rows' vectors, a set a device, made at the first run that keeps a state from inside.
    seg: std::sync::OnceLock<Vec<TapSeg>>,
    /// The multi-token-prediction layer, where it is loaded and chainable.
    mtp: Option<MtpChain>,
    m: std::sync::Mutex<ChainMut>,
    /// Steps the chain took, for a test that has to know it ran.
    pub(crate) runs: std::sync::atomic::AtomicUsize,
}

/// A device's working vectors for `rows` rows, the collapse and the head for `heads` of them.
fn chain_dev(c: &dyn ggml_rs::DeviceChain, cfg: &Config, rank: usize, rows: usize, heads: usize) -> ChainDev {
    let (h, s, nh, nkv, hd) = (cfg.hidden, cfg.streams, cfg.heads, cfg.kv_heads, cfg.head_dim);
    let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
    let v = |n: usize| c.vec(rows * n);
    ChainDev {
        x: v(s * h),
        normed: v(s * h),
        t: v(rank + s),
        post: v(s),
        post2: v(s),
        logits: v(s * h),
        y_in: v(h),
        y_out: v(h),
        y2_in: v(h),
        moe_out: v(h),
        router: v(cfg.experts + 1),
        qkv: v(conv_dim),
        z: v(cfg.nv * cfg.vd),
        ba: v(2 * cfg.nv),
        conv: v(conv_dim),
        core: v(cfg.nv * cfg.vd),
        qfull: v(2 * nh * hd),
        q: v(nh * hd),
        gate: v(nh * hd),
        k: v(nkv * hd),
        v: v(nkv * hd),
        qn: v(nh * hd),
        kn: v(nkv * hd),
        gated: v(nh * hd),
        index: v((cfg.index_heads + 1) * cfg.index_dim),
        table: v(cfg.rope_dim),
        mixed: c.vec(heads * h),
        head: c.vec(heads * cfg.vocab),
        picks: c.vec(4 * heads.max(1)),
    }
}

fn chain_packed(w: &Weight) -> Option<&dyn PackedLinear> {
    match w {
        Weight::Packed(p) => Some(p.as_ref()),
        _ => None,
    }
}

impl FlashNext {
    /// The chained step's state, made at the first step that can use one: None when a device has no chain, or a
    /// matrix is not where a chain reads it, or a layer's experts are neither there nor on the host (a layer no GPU
    /// had room for: [`ChainLayer::host`]).
    fn chain_state(&self) -> Option<&FnChain> {
        self.chain
            .get_or_init(|| {

                let cfg = &self.config;
                let s = cfg.streams;
                let chains: Vec<&dyn ggml_rs::DeviceChain> = self.devices.iter().map(|b| b.chain()).collect::<Option<_>>()?;
                let held = |d: usize, w: &Weight| chain_packed(w).is_some_and(|p| chains[d].holds_exl3(p));
                if cfg.kd != cfg.vd || ![16, 32, 64, 128].contains(&cfg.kd) || !(2..=8).contains(&cfg.conv) || cfg.rope_dim == 0 || cfg.rope_dim > cfg.head_dim {
                    return None;
                }
                let up = |d: usize, t: &Tensor| {
                    let t = if t.is_device() { t.to_host() } else { t.clone() };
                    let v = chains[d].vec(t.numel());
                    chains[d].upload(&v, t.data());
                    v
                };
                let hc = |d: usize, m: &HyperMix| -> Option<HcVecs> {
                    if m.packed {
                        return None;
                    }
                    let writes = if m.site { s } else { 0 };
                    Some(HcVecs { norm: up(d, &m.norm), down: ChainMat::new(chains[d], &m.down), up: ChainMat::new(chains[d], &m.up), rank: m.rank, writes })
                };
                let mut attn_of = vec![Vec::new(); self.devices.len()];
                let mut gdn = Vec::new();
                let mut layers = Vec::with_capacity(cfg.layers);
                for (i, l) in self.layers.iter().enumerate() {
                    let d = l.device;
                    let dense = |w: &Weight| match w {
                        Weight::Dense(t) => Some(ChainMat::new(chains[d], t)),
                        _ => None,
                    };
                    let mixer = match &l.mixer {
                        Mixer::Gdn(g) => {
                            if ![&g.qkv, &g.z, &g.out].iter().all(|w| held(d, w)) {
                                return None;
                            }
                            gdn.push(i);
                            ChainMixer::Gdn { ba: dense(&g.ba)?, conv: up(d, &g.conv), a: up(d, &g.a), dt: up(d, &g.dt_bias), norm: up(d, &g.norm), slot: gdn.len() - 1 }
                        }
                        Mixer::Attn(a) => {
                            if ![&a.q, &a.k, &a.v, &a.o, &a.index_qk].iter().all(|w| held(d, w)) {
                                return None;
                            }
                            attn_of[d].push(i);
                            ChainMixer::Attn { q_norm: up(d, &a.q_norm), k_norm: up(d, &a.k_norm), iq_norm: up(d, &a.index_q_norm), ik_norm: up(d, &a.index_k_norm), slot: attn_of[d].len() - 1 }
                        }
                    };
                    let host = !chains[d].holds_experts(l.moe.experts.as_ref());
                    if host && (!l.moe.experts.on_host() || std::env::var_os("OAIY_NO_HOST_LAYERS").is_some()) {
                        return None;
                    }
                    layers.push(ChainLayer { attn_hc: hc(d, &l.attn_hc)?, mlp_hc: hc(d, &l.mlp_hc)?, mixer, router: dense(&l.moe.router)?, host });
                }
                // the last layer, the collapse and the head on the last device: a run's layers have written to the cache
                // before it gets there, so nothing may leave it then
                let last = self.devices.len() - 1;
                if self.layers.last().map(|l| l.device) != Some(last) || !held(last, &self.head) {
                    return None;
                }
                let collapse = hc(last, &self.collapse)?;
                let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
                // the prediction layer, where its matrices and experts are where a chain reads them
                let mtp = self.mtp.as_ref().and_then(|mp| {
                    let c = chains[last];
                    if ![&mp.fc_e, &mp.fc_h, &mp.q, &mp.k, &mp.v, &mp.o].iter().all(|w| held(last, w)) || !c.holds_experts(mp.experts.as_ref()) {
                        return None;
                    }
                    let ones = c.vec(CHECK_ROWS * s);
                    c.upload(&ones, &vec![1.0; CHECK_ROWS * s]);
                    Some(MtpChain {
                        enorm: up(last, &mp.enorm),
                        hnorm: up(last, &mp.hnorm),
                        attn_hc: hc(last, &mp.attn_hc)?,
                        mlp_hc: hc(last, &mp.mlp_hc)?,
                        mixer: hc(last, &mp.mixer)?,
                        q_norm: up(last, &mp.q_norm),
                        k_norm: up(last, &mp.k_norm),
                        router: ChainMat::new(c, &mp.router),
                        ones,
                        hid: c.vec(CHECK_ROWS * s * cfg.hidden),
                        best: c.vec(3),
                        devs: Default::default(),
                    })
                });
                let mtp_rank = mtp.as_ref().map_or(0, |m| m.attn_hc.rank.max(m.mlp_hc.rank).max(m.mixer.rank));
                let rank = layers.iter().map(|l| l.attn_hc.rank.max(l.mlp_hc.rank)).max().unwrap_or(0).max(collapse.rank).max(mtp_rank);
                let devs = chains.iter().map(|c| chain_dev(*c, cfg, rank, 1, 1)).collect();
                let keys = chains.iter().zip(&attn_of).map(|(c, a)| c.vec(a.len().max(1) * cfg.index_dim)).collect();
                let kv = chains.iter().map(|c| ChainKv { layers: Vec::new(), raw: Vec::new(), cap: 0, out: c.vec(1), qsa: None, owner: 0 }).collect();
                let pool = gdn
                    .iter()
                    .map(|&i| {
                        let c = chains[self.layers[i].device];
                        (c.vec(cfg.nv * cfg.vd * cfg.kd), c.vec((cfg.conv - 1) * conv_dim))
                    })
                    .collect();
                let pd = self.layers[cfg.ple_layer].device;
                let ple = match (&self.ple.key, &self.ple.value) {
                    (Weight::Dense(k), Weight::Dense(v)) if cfg.ple_kernel >= 1 => {
                        let c = chains[pd];
                        let p = ChainPle { key: ChainMat::new(c, k), value: ChainMat::new(c, v), norm_key: up(pd, &self.ple.norm_key), norm_query: up(pd, &self.ple.norm_query), norm_conv: up(pd, &self.ple.norm_conv), conv: up(pd, &self.ple.conv) };
                        Some((p, ple_vecs(c, cfg, 1)))
                    }
                    _ => None,
                };
                let ple_window = chains[pd].vec((cfg.ple_kernel.max(1) - 1) * cfg.ngram * s * cfg.hidden);
                let rows_out = chains.iter().map(|c| c.vec(1)).collect();
                Some(FnChain { layers, devs, attn_of, gdn, collapse, rank, keys, ple, few: Default::default(), seg: Default::default(), mtp, m: std::sync::Mutex::new(ChainMut { kv, pool, ple_window, rows_out, undo: None, mtp_kv: None, mtp_hid: (0, 0) }), runs: Default::default() })
            })
            .as_ref()
    }

    /// Steps the chain has taken (a test's check that it ran).
    #[allow(dead_code)]
    /// The chain made and its kernels compiled before a first request would wait on them (its matrices packed and
    /// uploaded, some 30 pipelines built): a short prompt and a step on a cache of their own; drafting, a draft and
    /// checks of 2 to 4 rows, each undone. False where nothing is chained.
    pub fn warm_up(&self) -> bool {
        if self.chain_state().is_none() {
            return false;
        }
        let Ok(tokens) = self.tokenizer.encode("The river town kept its market on the north bank.", false) else { return false };
        if tokens.len() < 4 {
            return false;
        }
        let mut kv = self.new_kv_cache(tokens.len() + 8);
        let step = |tokens: &[u32], kv: &mut KvCache| self.embed_text(tokens).and_then(|e| self.forward(tokens, &e, kv, None)).is_ok();
        let mut warmed = step(&tokens, &mut kv) && step(&tokens[..1], &mut kv);
        if warmed && self.drafts() {
            // every draft made, however unlikely: the layer's passes all run
            warmed = self.draft_above(&kv, &tokens[1..2], 3, 0.0).is_some();
            for rows in 2..=4 {
                warmed &= self.check(&tokens[..rows], &mut kv).is_some();
                self.rollback(&mut kv, 1);
            }
        }
        self.chain.get().and_then(|c| c.as_ref()).inspect(|c| c.runs.store(0, std::sync::atomic::Ordering::Relaxed));
        warmed
    }

    #[cfg(test)]
    pub(crate) fn chain_runs(&self) -> usize {
        self.chain.get().and_then(|c| c.as_ref()).map_or(0, |c| c.runs.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// `tokens` (a decode step's one, or a prompt's chunk; their embeddings `embeds`), chained, if the devices can: the
    /// last row's logits. See [`Self::run_chained`].
    fn forward_chained(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache) -> Option<Tensor> {
        let logits = self.run_chained(tokens, embeds, kv, false)?;
        Some(Tensor::from_vec(logits, vec![1, self.config.vocab]))
    }

    /// A check of `tokens` (the token sampled, then its drafts; 2 to [`CHECK_ROWS`] of them) after what `kv` holds:
    /// every row's logits (`[1, vocab]` each), each row as a step would give it. The run is kept undoable
    /// ([`Self::rollback`]). None where it cannot be chained.
    pub fn check(&self, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<Tensor>> {
        if !(2..=CHECK_ROWS).contains(&tokens.len()) || kv.len + tokens.len() > kv.max_len {
            return None;
        }
        let embeds = self.embed_text(tokens).ok()?;
        let logits = self.run_chained(tokens, &embeds, kv, true)?;
        Some(logits.chunks_exact(self.config.vocab).map(|r| Tensor::from_vec(r.to_vec(), vec![1, self.config.vocab])).collect())
    }

    /// [`Self::check`] for a request that samples greedily: each row's token alone, its largest logit's (the first of
    /// equals, as greedy sampling takes it), picked on the GPU. A check's rows of logits are a megabyte each to read
    /// back, for a token each. None as [`Self::check`] (nothing run).
    pub fn check_picks(&self, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<u32>> {
        if !(2..=CHECK_ROWS).contains(&tokens.len()) || kv.len + tokens.len() > kv.max_len {
            return None;
        }
        let embeds = self.embed_text(tokens).ok()?;
        self.pick.store(true, std::sync::atomic::Ordering::Relaxed);
        let got = self.run_chained(tokens, &embeds, kv, true);
        self.pick.store(false, std::sync::atomic::Ordering::Relaxed);
        Some(got?.chunks_exact(4).take(tokens.len()).map(|row| row[0].to_bits()).collect())
    }

    /// A decode step's token for a request that samples greedily, as [`Self::check_picks`]: `token` (its embedding
    /// `embeds`) run after what `kv` holds, the next token picked on the GPU. None where the step cannot be chained
    /// (nothing run: [`Self::forward`] is the caller's).
    pub fn step_pick(&self, token: u32, embeds: &Tensor, kv: &mut KvCache) -> Option<u32> {
        if profile::on() || self.prompt_rows() == 0 {
            return None;
        }
        self.pick.store(true, std::sync::atomic::Ordering::Relaxed);
        let got = self.run_chained(&[token], embeds, kv, false);
        self.pick.store(false, std::sync::atomic::Ordering::Relaxed);
        got.map(|row| row[0].to_bits())
    }

    /// Undo the last check's rows past its first `keep` (the token sampled and the drafts accepted): each delta net's
    /// state and conv window and the n-gram layer's window as they were before it, its first `keep` rows run through
    /// them again, the n-gram history its tokens', and the cache cut back.
    pub fn rollback(&self, kv: &mut KvCache, keep: usize) {
        use ggml_rs::DeltaNet;
        let Some(st) = self.chain_state() else { return };
        let Some(chains) = self.devices.iter().map(|b| b.chain()).collect::<Option<Vec<_>>>() else { return };
        let cfg = &self.config;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        let Some(u) = m.undo.as_mut() else { return };
        let rows = u.tokens.len();
        assert!(keep >= 1 && keep <= rows && rows <= kv.len, "a rollback of {rows} rows to {keep}");
        if keep == rows {
            return;
        }
        let few = st.few[rows - 2].get().expect("the check's vectors");
        let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
        let dn = DeltaNet { rows: keep, v_heads: cfg.nv, k_heads: cfg.nk, k_dim: cfg.kd, v_dim: cfg.vd, scale_q: 1.0 / (cfg.vd as f32).sqrt(), eps: cfg.eps, sigmoid_gate: true };
        let ple_device = self.layers[cfg.ple_layer].device;
        let slot = ple_slot(cfg);
        for (d, c) in chains.iter().enumerate() {
            let dv = &few.devs[d];
            let mut rec = c.begin();
            rec.keep_groups(true);
            for (i, &l) in st.gdn.iter().enumerate().filter(|&(_, &l)| self.layers[l].device == d) {
                let (Some(state), Some(conv)) = (kv.ssm_state[l].as_ref().and_then(|t| c.aliased(t)), kv.ssm_conv[l].as_ref().and_then(|t| c.aliased(t))) else {
                    unreachable!("a check left layer {l}'s state the chain's")
                };
                let ChainMixer::Gdn { conv: w, a, dt, norm, .. } = &st.layers[l].mixer else { unreachable!("layer {l} is a delta net") };
                let (bs, bc) = &u.backups[i];
                let (qkv, ba) = &u.inputs[i];
                rec.copy(bs, 0, &state, 0, state.len);
                rec.copy(bc, 0, &conv, 0, conv.len);
                rec.ssm_conv(qkv, w, &conv, &dv.conv, keep, conv_dim, cfg.conv);
                // the outputs are not wanted: the check's were the kept rows' already
                rec.delta_net(&dv.conv, &dv.z, ba, a, dt, norm, &state, &dv.core, dn);
            }
            if d == ple_device {
                if let (Some(backup), Some((p, _)), Some(v), Some(window)) = (&u.window, &st.ple, &few.ple, kv.ssm_conv[slot].as_ref().and_then(|t| c.aliased(t))) {
                    // the window as the kept rows leave it (the conv's sums into scratch)
                    rec.copy(backup, 0, &window, 0, window.len);
                    rec.ple_conv(&dv.logits, &dv.normed, &v.conv_in, &window, &p.conv, keep, cfg.streams * cfg.hidden, cfg.ple_kernel, cfg.ngram);
                }
            }
            // (nothing of it is read back, and each device's next run goes to its queue behind it: not waited for,
            // where the thread parked for the first card's undoing and then the second's)
            rec.send();
        }
        // the n-gram history: the one before the check, then its kept tokens
        let ctx = cfg.ngram - 1;
        let mut history: Vec<f32> = match &u.history {
            Some(t) => t.to_host().data().to_vec(),
            None => vec![cfg.ple_eos as f32; ctx],
        };
        history.extend(u.tokens[..keep].iter().map(|&t| t as f32));
        kv.ssm_state[slot] = Some(Tensor::from_vec(history[history.len() - ctx..].to_vec(), vec![ctx]));
        kv.len -= rows - keep;
        u.tokens.truncate(keep);
        if m.mtp_hid.0 + m.mtp_hid.1 == kv.len + rows - keep {
            m.mtp_hid.1 = m.mtp_hid.1.min(keep);
        }
    }

    /// `tokens` (a decode step's one, a check's few, or a prompt's chunk; their embeddings `embeds`), chained, if the
    /// devices can: one submit a layer, the layer's work all on its GPU (the previous layer's experts as the host routed
    /// them, the hyper-connections' write-back, norm, gates and mix, the delta net or the attention, the router), only
    /// the router's logits coming back (and an attention layer's K, V and indexer key for the host's cache); the n-gram
    /// features before their layer and the hand-over between devices through the host. A step's and a few rows' (up to
    /// [`CHECK_ROWS`]) experts are routed on their GPU, a device's layers one submit, their vectors kept. The last row's
    /// logits; every row's for a check (`check`: undoable, see [`Self::rollback`]). None leaves the run to `forward`'s
    /// own path: past the dense span (QSA's sparse attention), or with images.
    fn run_chained(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool) -> Option<Vec<f32>> {
        let run = self.run_begin(tokens, embeds, kv, check, &mut None, None, &[])?;
        Some(run.finish(self, kv))
    }

    /// A prompt's chunks in turn, chained, each chunk's first devices' layers run as the last device runs the chunk
    /// before's (`kv` then holds them all): the last chunk's logits, or None where a chunk cannot be chained (the
    /// chunks before it run, `done` of each said; the rest the caller's). `done(i)` once chunk `i` has gone to the
    /// GPUs (its K and V in the host's cache once the next has).
    pub fn forward_chunks(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize)) -> Option<Tensor> {
        self.forward_chunks_tapped(chunks, kv, done, &[]).0
    }

    /// Whether a prompt's chunks of [`Self::prompt_rows`] (`rows` in all, after what `kv` holds) can keep their
    /// recurrent states from inside them once each of `taps` rows is in ([`Self::forward_chunks_tapped`]): chained,
    /// the n-gram layer too (its window a device's vector), and each chunk's rows after the first state it keeps few
    /// ([`TAP_ROWS`]: their recurrences go through vectors of their own). OAIY_NO_TAPS: never.
    pub fn can_tap(&self, rows: usize, kv: &KvCache, taps: &[usize]) -> bool {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || std::env::var_os("OAIY_NO_TAPS").is_some() || profile::on() || rows == 0 || taps.is_empty() {
            return false;
        }
        let most = self.prompt_rows();
        self.chain_state().is_some_and(|st| st.ple.is_some())
            && self.devices.iter().all(|b| b.chain().is_some())
            && (kv.len + rows) / self.config.index_ratio <= 4096
            && taps.windows(2).all(|w| w[0] < w[1])
            && taps[0] > 0
            && taps[taps.len() - 1] <= rows
            && (0..rows).step_by(most).all(|at| {
                let end = (at + most).min(rows);
                taps.iter().find(|&&p| p > at && p <= end).map_or(true, |&p| end - p <= TAP_ROWS)
            })
    }

    /// [`Self::forward_chunks`] with the delta nets' states and conv windows and the n-gram layer's window and history
    /// as they are once each of `taps` rows of the chunks is in (ascending, counted through the chunks): what a
    /// checkpoint there holds, the run not stopping for it. A prompt's last two, before the assistant's header and
    /// before its last token, each ended a run, and a run of few rows costs what a chunk of 64 does (its weights are
    /// decoded once whatever its rows): a follow-up turn's 21 new tokens 102 ms, then 42 and 18 for those two. The
    /// states of the chunks that ran (all of them where the logits are given).
    pub fn forward_chunks_tapped(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize]) -> (Option<Tensor>, Vec<llama_rs::Tapped>) {
        let mut tapped: Vec<llama_rs::Tapped> = Vec::new();
        // each chunk's first row among the chunks'
        let starts: Vec<usize> = chunks.iter().scan(0, |at, (t, _)| { let a = *at; *at += t.len(); Some(a) }).collect();
        let logits = self.chunks_run(chunks, kv, done, taps, &starts, &mut tapped);
        (logits, tapped)
    }

    fn chunks_run(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize], starts: &[usize], tapped: &mut Vec<llama_rs::Tapped>) -> Option<Tensor> {
        // each chunk's n-gram features (random reads of a 32 GB table: some 25 ms a chunk of 512) read on a thread of
        // their own, a chunk ahead of the GPUs
        let ctx = self.config.ngram - 1;
        let mut history = self.ple_history(kv);
        let histories: Vec<Vec<i64>> = chunks
            .iter()
            .map(|(t, _)| {
                history.extend(t.iter().map(|&v| v as i64));
                let h = history.clone();
                history.drain(..history.len() - ctx);
                h
            })
            .collect();
        std::thread::scope(|sc| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Option<Vec<f32>>>(1);
            let histories = &histories;
            sc.spawn(move || {
                for h in histories {
                    if tx.send(self.ngram_embedding(h).ok()).is_err() {
                        break;
                    }
                }
            });
            let mut pending: Option<ChainedRun<'_>> = None;
            let mut last = None;
            // Over two devices each chunk in two parts, its first device's as that device still runs the chunk
            // before's (OAIY_FN_IN_TURN: a chunk whole at a time): where the n-gram layer is chained, the experts
            // routed on their GPU, the layers one device's then the other's, and each device has room for another
            // chunk's vectors.
            let ahead = std::env::var_os("OAIY_FN_IN_TURN").is_none()
                && self.devices.len() == 2
                && self.chain_state().is_some_and(|st| st.ple.is_some() && st.layers.iter().all(|l| !l.host))
                && self.config.experts <= 1024
                && self.config.top_k <= 32
                && std::env::var_os("OAIY_HOST_ROUTE").is_none()
                && self.layers.windows(2).filter(|w| w[0].device != w[1].device).count() == 1
                && self.devices.iter().all(|b| b.chain().is_some_and(|c| c.has_room(1 << 30)));
            let mut parked: Option<(usize, Box<Parked<'_>>)> = None;
            // Each device then has a chunk's work behind the one it runs, with no gap: its pieces two at a time on its
            // queue, each encoded at its turn, until the chunks are in. Without that a card under a power limit ran a
            // tenth as fast for seconds at a time (15,037 tokens in 6.6 to 13.3 s where 5.3; the chunks whole, with
            // their gaps, 5.9 to 7.7).
            struct Fed<'a>(Vec<&'a dyn ggml_rs::DeviceChain>);
            impl Drop for Fed<'_> {
                fn drop(&mut self) {
                    for c in &self.0 {
                        c.pieces_in_flight_at_most(0);
                    }
                }
            }
            let fed = Fed(if ahead { self.devices.iter().filter_map(|b| b.chain()).collect() } else { Vec::new() });
            for c in &fed.0 {
                c.pieces_in_flight_at_most(2);
            }
            let said = std::env::var_os("OAIY_FN_LOG").is_some();
            let began = std::time::Instant::now();
            for (i, (tokens, embeds)) in chunks.iter().enumerate() {
                let t0 = began.elapsed().as_secs_f64() * 1e3;
                let ple = rx.recv().ok().flatten();
                let t1 = began.elapsed().as_secs_f64() * 1e3;
                let had = pending.is_some();
                let fits = tokens.len() <= self.prompt_rows() && !profile::on() && ple.is_some();
                // (the states this chunk keeps: by its own rows)
                let local: Vec<usize> = taps.iter().filter(|&&p| p > starts[i] && p <= starts[i] + tokens.len()).map(|&p| p - starts[i]).collect();
                // (the chunk before's rest goes after this chunk's first part, or before a chunk that goes whole)
                if ahead && fits && tokens.len() > CHECK_ROWS {
                    match self.run_part(tokens, embeds, kv, false, &mut pending, ple, Stage::First, &local) {
                        Some(Went::Parked(p)) => {
                            let mut before = parked.replace((i, p));
                            self.run_rest(&mut before, &mut pending, embeds, kv, done, tapped);
                            if said {
                                eprintln!("  fn chunk {i}: at {t0:.0} ms, features waited {:.0}, its first part and the chunk before's rest in {:.0}", t1 - t0, began.elapsed().as_secs_f64() * 1e3 - t1);
                            }
                            continue;
                        }
                        Some(Went::Run(mut run)) => {
                            // (no second device's part after all: as a chunk whole)
                            self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                            tapped.append(&mut run.tapped);
                            if let Some(p) = pending.replace(run) {
                                p.finish(self, kv);
                            }
                            done(i);
                            continue;
                        }
                        None => {
                            self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                            if let Some(p) = pending.take() {
                                p.finish(self, kv);
                            }
                            return None;
                        }
                    }
                }
                self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                let run = if fits { self.run_begin(tokens, embeds, kv, false, &mut pending, ple, &local) } else { None };
                if said {
                    eprintln!("  fn chunk {i}: at {t0:.0} ms, features waited {:.0}, begun in {:.0} (the one before {})", t1 - t0, began.elapsed().as_secs_f64() * 1e3 - t1, if had && pending.is_none() { "finished inside" } else { "left" });
                }
                let Some(mut run) = run else {
                    if let Some(p) = pending.take() {
                        p.finish(self, kv);
                    }
                    return None;
                };
                tapped.append(&mut run.tapped);
                // (one the run did not take: a chain on one device)
                if let Some(p) = pending.replace(run) {
                    p.finish(self, kv);
                }
                done(i);
            }
            if let Some(&(_, embeds)) = chunks.last() {
                self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
            }
            if let Some(p) = pending.take() {
                last = Some(p.finish(self, kv));
            }
            last.map(|l| Tensor::from_vec(l, vec![1, self.config.vocab]))
        })
    }

    /// A parked chunk's rest ([`Stage::Rest`]; `embeds` any chunk's, not read): its second device's layers recorded
    /// and gone behind the run before's, which is then finished, this chunk's run `pending` in its place.
    fn run_rest<'a>(&'a self, parked: &mut Option<(usize, Box<Parked<'a>>)>, pending: &mut Option<ChainedRun<'a>>, embeds: &Tensor, kv: &mut KvCache, done: &mut dyn FnMut(usize), tapped: &mut Vec<llama_rs::Tapped>) {
        if let Some((j, q)) = parked.take() {
            let Some(Went::Run(mut run)) = self.run_part(&[], embeds, kv, false, pending, None, Stage::Rest(q), &[]) else { panic!("a chunk's second device's part") };
            tapped.append(&mut run.tapped);
            if let Some(p) = pending.replace(run) {
                p.finish(self, kv);
            }
            done(j);
        }
    }

    /// An attention layer's rows of a chained run (`t` of them from `at`, K then V each, and its indexer keys) into
    /// the host's cache.
    fn cache_rows(&self, kv: &mut KvCache, i: usize, at: usize, t: usize, kvrows: Vec<f32>, raw: Vec<f32>) {
        let Mixer::Attn(a) = &self.layers[i].mixer else { unreachable!("layer {i} attends") };
        let b = self.devices[self.layers[i].device].as_ref();
        let id = self.config.index_dim;
        let len = kv.len;
        kv.len = at;
        kv.append(b, a.index_slot, &Tensor::from_vec(raw.clone(), vec![t, 1, id]), &Tensor::from_vec(raw, vec![t, 1, id]));
        kv.append_rows(b, i, &kvrows, t);
        kv.len = len;
    }

    /// [`Self::run_chained`] up to its last device's wait: every device's work gone (the last's running), `kv`
    /// committed; `prev` (a chunk's run before this one's, its last device still running) finished as the next
    /// device's layers are recorded, so the device holds one chunk's scratch at a time.
    fn run_begin<'a>(&'a self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool, prev: &mut Option<ChainedRun<'a>>, ple: Option<Vec<f32>>, taps: &[usize]) -> Option<ChainedRun<'a>> {
        match self.run_part(tokens, embeds, kv, check, prev, ple, Stage::Whole, taps)? {
            Went::Run(run) => Some(run),
            Went::Parked(_) => unreachable!("a whole run stops at no device"),
        }
    }

    /// [`Self::run_begin`] whole, or a prompt's chunk over two devices in two parts ([`Self::forward_chunks`]): its
    /// first device's layers recorded and gone ([`Stage::First`]: `kv` committed, the chunk parked where its next
    /// device's layers begin), then, once the chunk after's first part is gone too, the rest from there
    /// ([`Stage::Rest`]: `tokens` and `embeds` not read, the chunk's own kept with it). So each device has its next
    /// chunk's work behind the one it runs, where a chunk whole left the first device idle while the host recorded
    /// the second's layers and stored rows, and the second while the host recorded the next chunk's first.
    #[allow(clippy::too_many_arguments)]
    fn run_part<'a>(&'a self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool, prev: &mut Option<ChainedRun<'a>>, ple: Option<Vec<f32>>, stage: Stage<'a>, taps: &[usize]) -> Option<Went<'a>> {
        use ggml_rs::{ChainRecorder, DeltaNet};
        use std::sync::atomic::Ordering;
        if std::env::var_os("OAIY_NO_CHAIN").is_some() {
            return None;
        }
        let cfg = &self.config;
        let first_only = matches!(stage, Stage::First);
        let mut parked = match stage {
            Stage::Rest(p) => Some(p),
            _ => None,
        };
        let resumed = parked.is_some();
        let past = parked.as_ref().map_or(kv.len, |p| p.past);
        let t = parked.as_ref().map_or(tokens.len(), |p| p.t);
        let ratio = cfg.index_ratio;
        let phases = std::env::var_os("OAIY_FN_LOG").is_some() && t > 64;
        let clock = std::time::Instant::now();
        let mut marks: Vec<(&str, f64)> = Vec::new();
        let mut mark = |what: &'static str| {
            if phases {
                marks.push((what, clock.elapsed().as_secs_f64() * 1e3));
            }
        };
        // past the dense span the attention layers' queries attend to their QSA blocks (at most 4096 blocks)
        if t == 0 || (past + t) / ratio > 4096 {
            return None;
        }
        let sparse = past + t > cfg.index_budget / ratio * ratio + ratio - 1;
        // a device without QSA's kernels (a workgroup's memory too small for its selection) leaves it to the host path
        if sparse && self.devices.iter().filter_map(|b| b.chain()).any(|c| c.qsa_attention_out_len(1, cfg.heads, cfg.head_dim, cfg.index_budget / ratio, ratio) == 0) {
            return None;
        }
        let st = self.chain_state()?;
        let chains: Vec<&dyn ggml_rs::DeviceChain> = self.devices.iter().map(|b| b.chain()).collect::<Option<_>>()?;
        let (h, s) = (cfg.hidden, cfg.streams);
        let (nh, nkv, hd, rot) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.rope_dim);
        let (kvd, row) = (nkv * hd, 2 * nkv * hd);
        let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
        let eps = cfg.eps;
        let few = (2..=CHECK_ROWS).contains(&t);
        assert!(!check || few, "a check of 2 to {CHECK_ROWS} rows");
        let id_dim = cfg.index_dim;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        // the devices' copies of the attention caches: room for this run, the rows the host wrote since (a resumed
        // chunk's when it began)
        for (d, layers) in st.attn_of.iter().enumerate().filter(|_| !resumed) {
            let g = &mut m.kv[d];
            if g.cap < past + t {
                let cap = (past + t).next_power_of_two().max(256);
                g.layers = (0..layers.len()).map(|i| match g.layers.get(i) { Some(old) => chains[d].resize(old, cap * row), None => chains[d].vec(cap * row) }).collect();
                g.raw = (0..layers.len()).map(|i| match g.raw.get(i) { Some(old) => chains[d].resize(old, cap * id_dim), None => chains[d].vec(cap * id_dim) }).collect();
                g.out = chains[d].vec(chains[d].attention_out_len(nh, hd, cap));
                g.qsa = None;
                g.cap = cap;
            }
            let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
            if from < past {
                for (slot, &l) in layers.iter().enumerate() {
                    let (kh, vh) = (kv.k_buffer(l).to_host(), kv.v_buffer(l).to_host());
                    let mut rows = Vec::with_capacity((past - from) * row);
                    for t in from..past {
                        rows.extend_from_slice(&kh.data()[t * kvd..(t + 1) * kvd]);
                        rows.extend_from_slice(&vh.data()[t * kvd..(t + 1) * kvd]);
                    }
                    chains[d].upload_at(&g.layers[slot], from * row, &rows);
                    // and its raw indexer keys
                    let Mixer::Attn(a) = &self.layers[l].mixer else { unreachable!("layer {l} attends") };
                    let raw = kv.k_buffer(a.index_slot).to_host();
                    chains[d].upload_at(&g.raw[slot], from * id_dim, &raw.data()[from * id_dim..past * id_dim]);
                }
            }
            g.owner = kv.id;
            if sparse && t <= CHECK_ROWS && !layers.is_empty() && g.qsa.is_none() {
                g.qsa = Some(self.qsa_vecs(chains[d], CHECK_ROWS, g.cap));
            }
        }
        // the delta-net layers' recurrent states as the chain's vectors (the cache aliasing them)
        let mut states = Vec::with_capacity(st.gdn.len());
        for (slot, &l) in st.gdn.iter().enumerate() {
            let c = chains[self.layers[l].device];
            let (ps, pc) = &mut m.pool[slot];
            let adopt = |pool: &mut ggml_rs::DeviceVec, t: &mut Option<Tensor>, shape: Vec<usize>| -> ggml_rs::DeviceVec {
                let len: usize = shape.iter().product();
                if let Some(v) = t.as_ref().and_then(|t| c.aliased(t)).filter(|v| v.len == len) {
                    return v;
                }
                let host = t.take().map(|t| t.to_host());
                if Arc::strong_count(&pool.inner) > 1 {
                    *pool = c.vec(len);
                }
                match host.filter(|h| h.numel() == len) {
                    Some(h) => c.upload(pool, h.data()),
                    None => c.zero(pool),
                }
                *t = Some(c.alias(pool, shape));
                pool.clone()
            };
            let sv = adopt(ps, &mut kv.ssm_state[l], vec![cfg.nv, cfg.vd, cfg.kd]);
            let cv = adopt(pc, &mut kv.ssm_conv[l], vec![cfg.conv - 1, conv_dim]);
            states.push((sv, cv));
        }
        // the n-gram layer's window as the chain's vector (the cache aliasing it), where the layer is chained
        let ple_device = self.layers[cfg.ple_layer].device;
        let ple_window = st.ple.as_ref().map(|_| {
            let c = chains[ple_device];
            let shape = vec![(cfg.ple_kernel - 1) * cfg.ngram, s * h];
            let len: usize = shape.iter().product();
            let slot = ple_slot(cfg);
            if let Some(v) = kv.ssm_conv[slot].as_ref().and_then(|t| c.aliased(t)).filter(|v| v.len == len) {
                return v;
            }
            let host = kv.ssm_conv[slot].take().map(|t| t.to_host());
            if Arc::strong_count(&m.ple_window.inner) > 1 {
                m.ple_window = c.vec(len);
            }
            match host.filter(|h| h.numel() == len) {
                Some(hv) => c.upload(&m.ple_window, hv.data()),
                None => c.zero(&m.ple_window),
            }
            kv.ssm_conv[slot] = Some(c.alias(&m.ple_window, shape));
            m.ple_window.clone()
        });
        // The states the run keeps from inside it (`taps`: after that many of its rows, ascending; a resumed chunk's
        // its own): a pair of vectors a delta net and one for the n-gram layer's window, and what the caller is given
        // of them: a checkpoint's tensors, as the cache holds its own (the n-gram layer's history its tokens').
        let (kept, mut tapped): (Vec<Tap>, Vec<llama_rs::Tapped>) = match parked.as_mut() {
            Some(p) => (std::mem::take(&mut p.kept), std::mem::take(&mut p.tapped)),
            None if taps.is_empty() => (Vec::new(), Vec::new()),
            None => {
                assert!(!check && taps.windows(2).all(|w| w[0] < w[1]) && taps[0] > 0 && taps[taps.len() - 1] <= t && t - taps[0] <= TAP_ROWS, "a run of {t} rows keeps states after {taps:?}");
                let before = self.ple_history(kv);
                let ctx = cfg.ngram - 1;
                let kept: Vec<Tap> = taps
                    .iter()
                    .map(|&row| Tap {
                        row,
                        gdn: st.gdn.iter().zip(&states).map(|(&l, (sv, cv))| (chains[self.layers[l].device].vec(sv.len), chains[self.layers[l].device].vec(cv.len))).collect(),
                        window: ple_window.as_ref().map(|w| chains[ple_device].vec(w.len)),
                    })
                    .collect();
                let tapped = kept
                    .iter()
                    .map(|tap| {
                        let none = || (0..kv.ssm_state.len()).map(|_| None).collect::<Vec<Option<Tensor>>>();
                        let (mut ss, mut cs) = (none(), none());
                        for (slot, &l) in st.gdn.iter().enumerate() {
                            let c = chains[self.layers[l].device];
                            ss[l] = Some(c.alias(&tap.gdn[slot].0, vec![cfg.nv, cfg.vd, cfg.kd]));
                            cs[l] = Some(c.alias(&tap.gdn[slot].1, vec![cfg.conv - 1, conv_dim]));
                        }
                        if let Some(w) = &tap.window {
                            cs[ple_slot(cfg)] = Some(chains[ple_device].alias(w, vec![(cfg.ple_kernel - 1) * cfg.ngram, s * h]));
                        }
                        let ids: Vec<i64> = before.iter().copied().chain(tokens[..tap.row].iter().map(|&v| v as i64)).collect();
                        ss[ple_slot(cfg)] = Some(Tensor::from_vec(ids[ids.len() - ctx..].iter().map(|&v| v as f32).collect(), vec![ctx]));
                        llama_rs::Tapped { at: past + tap.row, states: ss, convs: cs }
                    })
                    .collect();
                (kept, tapped)
            }
        };
        let segs: Option<&Vec<TapSeg>> = kept.first().filter(|tap| tap.row < t).map(|_| {
            st.seg.get_or_init(|| {
                chains
                    .iter()
                    .map(|c| {
                        let v = |n: usize| c.vec(TAP_ROWS * n);
                        TapSeg { qkv: v(conv_dim), conv: v(conv_dim), z: v(cfg.nv * cfg.vd), ba: v(2 * cfg.nv), core: v(cfg.nv * cfg.vd), x: v(s * h), gated: v(s * h), conv_in: v(s * h) }
                    })
                    .collect()
            })
        });
        // a step's vectors and a few rows' (their bind groups kept); a prompt's chunk's its own
        let keep = t == 1 || few;
        let few_set = few.then(|| {
            st.few[t - 2].get_or_init(|| {
                let pd = self.layers[cfg.ple_layer].device;
                FewSet {
                    devs: chains.iter().map(|c| chain_dev(*c, cfg, st.rank, t, t)).collect(),
                    ple: st.ple.as_ref().map(|_| ple_vecs(chains[pd], cfg, t)),
                    keys: chains.iter().zip(&st.attn_of).map(|(c, a)| a.iter().map(|_| c.vec(t * cfg.index_dim)).collect()).collect(),
                    q1: chains.iter().map(|c| c.vec(cfg.heads * cfg.head_dim)).collect(),
                }
            })
        });
        let owned: Option<Arc<Vec<ChainDev>>> = (t != 1 && few_set.is_none()).then(|| match &parked {
            Some(p) => Arc::clone(&p.devs),
            None => Arc::new(chains.iter().map(|c| chain_dev(*c, cfg, st.rank, t, 1)).collect()),
        });
        let devs: &[ChainDev] = match (&owned, few_set) {
            (Some(o), _) => o,
            (None, Some(f)) => &f.devs,
            (None, None) => &st.devs,
        };
        // a check: what undoing it needs (the delta nets' and the n-gram window's backups are taken as it runs)
        if check {
            if m.undo.is_none() {
                let ple_window = st.ple.as_ref().map(|_| chains[self.layers[cfg.ple_layer].device].vec((cfg.ple_kernel.max(1) - 1) * cfg.ngram * s * h));
                let (backups, inputs) = st
                    .gdn
                    .iter()
                    .map(|&l| {
                        let c = chains[self.layers[l].device];
                        ((c.vec(cfg.nv * cfg.vd * cfg.kd), c.vec((cfg.conv - 1) * conv_dim)), (c.vec(CHECK_ROWS * conv_dim), c.vec(CHECK_ROWS * 2 * cfg.nv)))
                    })
                    .unzip();
                m.undo = Some(Undo { backups, inputs, window: ple_window, tokens: Vec::new(), history: None });
            }
            let u = m.undo.as_mut().expect("made");
            u.tokens = tokens.to_vec();
            u.history = kv.ssm_state[ple_slot(cfg)].clone();
        }
        // the partial RoPE's sines and cosines at these positions, on every device: the next ones in order, or the
        // ones the run was given, a pair's angle then its axis's position (time, height, width, interleaved over the
        // pairs by Qwen's sections of 11, 11 and 10, as the host's `multimodal_rope::text` rotates them)
        let axis = |k: usize| if k % 3 == 1 && k < 33 { 1 } else if k % 3 == 2 && k < 30 { 2 } else { 0 };
        let given = self.placed.lock().unwrap_or_else(|p| p.into_inner()).clone().filter(|p| p.len() == t);
        let table: Vec<f32> = (0..t)
            .flat_map(|row| {
                let at = given.as_ref().map(|p| p[row]);
                (0..rot / 2).flat_map(move |k| {
                    let pos = at.map_or(past + row, |p| p[axis(k)] as usize);
                    let (sn, cs) = (pos as f32 * cfg.rope_theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                    [sn, cs]
                })
            })
            .collect();
        for (c, dv) in chains.iter().zip(devs).filter(|_| !resumed) {
            c.upload(&dv.table, &table);
        }
        // a prompt's attention scratch, on every device with attention layers (a few rows' kept, grown as positions are)
        let attn_rows: Vec<Option<ggml_rs::DeviceVec>> = match &parked {
            Some(p) => p.attn_rows.clone(),
            None => chains
                .iter()
                .enumerate()
                .map(|(d, c)| {
                    if t == 1 || st.attn_of[d].is_empty() {
                        return None;
                    }
                    let len = c.attention_rows_out_len(t, nh, hd, past + t);
                    if !few {
                        return Some(c.vec(len));
                    }
                    if m.rows_out[d].len < len {
                        m.rows_out[d] = c.vec(len.next_power_of_two());
                    }
                    Some(m.rows_out[d].clone())
                })
                .collect(),
        };
        // a prompt's chunk's QSA vectors (past the dense span) and its rows' raw indexer keys, on every device with
        // attention layers
        let prompt_qsa: Arc<Vec<Option<QsaVecs>>> = match &parked {
            Some(p) => Arc::clone(&p.qsa),
            None => Arc::new(chains.iter().enumerate().map(|(d, c)| (sparse && t > CHECK_ROWS && !st.attn_of[d].is_empty()).then(|| self.qsa_vecs(*c, t, m.kv[d].cap))).collect()),
        };
        let prompt_keys: Vec<Option<ggml_rs::DeviceVec>> = match &parked {
            Some(p) => p.keys.clone(),
            None => chains.iter().enumerate().map(|(d, c)| (t > CHECK_ROWS && !st.attn_of[d].is_empty()).then(|| c.vec(t * id_dim))).collect(),
        };
        // the n-gram features (on the n-gram layer's device, where it is chained), then the embedding in every stream
        // (a prompt's chunk's read already, as the chunk before ran)
        let ple_emb = match ple {
            _ if resumed => Vec::new(),
            Some(f) => {
                let mut history = self.ple_history(kv);
                history.extend(tokens.iter().map(|&t| t as i64));
                self.ple_carry(&history, kv);
                f
            }
            None => self.ple_embed_host(tokens, kv).ok()?,
        };

        let ple_owned: Option<Arc<PleVecs>> = (st.ple.is_some() && t != 1 && !few).then(|| match &parked {
            Some(p) => Arc::clone(p.ple.as_ref().expect("the chunk's n-gram vectors")),
            None => Arc::new(ple_vecs(chains[ple_device], cfg, t)),
        });
        let ple_vs: Option<(&ChainPle, &PleVecs)> = match &st.ple {
            Some((p, step)) if t == 1 => Some((p, step)),
            Some((p, _)) if few => few_set.and_then(|f| f.ple.as_ref()).map(|v| (p, v)),
            Some((p, _)) => ple_owned.as_deref().map(|v| (p, v)),
            None => None,
        };
        // where the layers go on from: the first, or a resumed chunk's next device's first
        let start = parked.as_ref().map_or(0, |p| p.at);
        let mut d = self.layers[start].device;
        let e = match &parked {
            Some(p) => p.e.clone(),
            None => embeds.to_host(),
        };
        if !resumed {
            if let Some((_, v)) = ple_vs {
                chains[ple_device].upload(&v.emb, &ple_emb);
            }
            let mut x0 = Vec::with_capacity(t * s * h);
            for row in e.data().chunks_exact(h).take(t) {
                for _ in 0..s {
                    x0.extend_from_slice(row);
                }
            }
            chains[d].upload(&devs[d].x, &x0);
        }
        let hc = |rec: &mut dyn ChainRecorder, dv: &ChainDev, rows: usize, hcv: &HcVecs, pending: Option<(&ggml_rs::DeviceVec, &ggml_rs::DeviceVec)>, post: &ggml_rs::DeviceVec, out: &ggml_rs::DeviceVec| {
            if let Some((y, p)) = pending {
                rec.stream_apply(&dv.x, y, p, rows, s, h);
            }
            rec.rmsnorm_streams(&dv.x, &hcv.norm, &dv.normed, rows, s, eps);
            hcv.project(&mut *rec, &dv.normed, &dv.t, post, &dv.logits, out, rows, s, h);
        };
        // The experts are routed on their GPU (each layer's router then its experts, a device's layers one submit; a
        // prompt's rows grouped by expert there too, where the tensor cores take them), else by the host between a
        // layer's submits (how many rows each expert takes is what the dispatches are sized by). OAIY_HOST_ROUTE
        // routes on the host.
        let on_gpu = cfg.experts <= 1024 && cfg.top_k <= 32 && std::env::var_os("OAIY_HOST_ROUTE").is_none();
        let undo = m.undo.as_ref().filter(|_| check);
        // a prompt's rows as the host routed them (between a layer's submits); on the GPU, each layer's are recorded
        // after its router, their sums added to the streams there
        enum Routed {
            Host(Vec<Vec<(usize, f32)>>),
            /// A layer's experts run on the host (none of its GPU's): their sums, for the device they go up to.
            Summed(usize, Vec<f32>),
        }
        let experts = |rec: &mut dyn ChainRecorder, dv: &ChainDev, layer: usize, routed: Routed| match routed {
            Routed::Host(assign) => rec.moe_rows(self.layers[layer].moe.experts.as_ref(), &dv.y2_in, &dv.moe_out, &assign),
            Routed::Summed(device, sums) => chains[device].upload(&dv.moe_out, &sums),
        };
        // an attention layer's K and V rows and its indexer keys (`[t, index_dim]`), read back, into the host's cache
        let (iq, id) = (cfg.index_heads * cfg.index_dim, cfg.index_dim);
        let to_cache = |i: usize, kvrows: Vec<f32>, raw: Vec<f32>, kv: &mut KvCache| self.cache_rows(kv, i, past, t, kvrows, raw);
        // the recording open on device `d`, and the attention layers whose rows and keys it reads (in order, before
        // whatever else it reads)
                let mut open: Option<Box<dyn ChainRecorder + '_>> = None;
        let mut attn_reads: Vec<usize> = Vec::new();
        // a device's attention layers' rows and keys read at its handoff, into the host's cache once the next device's
        // layers are recorded (as that device runs them)
        let mut to_store: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
        // a device's recording (its work submitted, its reads its streams and its layers' rows) whose streams go up
        // to the next device (its work held until they do), and its attention layers read
        let mut handoffs: Vec<(Box<dyn ChainRecorder + '_>, usize, Vec<usize>)> = parked.as_mut().map(|p| std::mem::take(&mut p.handoffs)).unwrap_or_default();
        let mut pending: Option<Routed> = None;
        // (OAIY_FN_HOST_LOG: what the run's host layers cost, the waits for their read-backs and their experts)
        static HOST_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let host_log = *HOST_LOG.get_or_init(|| std::env::var_os("OAIY_FN_HOST_LOG").is_some());
        let (mut host_layers, mut host_waited, mut host_ran) = (0usize, 0f64, 0f64);
        if resumed {
            // (as after a handoff: this device's work held till the streams are up)
            let mut r = chains[d].begin();
            r.keep_groups(keep);
            // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
            r.rows_alike(keep);
            r.hold();
            open = Some(r);
        }
        mark("ready");
        for (i, (layer, cl)) in self.layers.iter().zip(&st.layers).enumerate().skip(start) {
            let dev = layer.device;
            let ple_here = i == cfg.ple_layer;
            if dev != d && pending.is_none() && !(ple_here && ple_vs.is_none()) {
                // the next device's layers recorded as this one runs its own (its streams uploaded to the next once
                // it has, the next's work held until then)
                let mut rec = open.take().unwrap_or_else(|| {
                    let mut r = chains[d].begin();
                    r.keep_groups(keep);
                    // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                    r.rows_alike(keep);
                    r
                });
                rec.read(&devs[d].x);
                rec.flush();
                mark("first device recorded and gone");
                handoffs.push((rec, dev, std::mem::take(&mut attn_reads)));
                if first_only {
                    // the chunk parked here: its next device's layers after the next chunk's first part
                    kv.commit(t);
                    kv.dirty_from = usize::MAX;
                    return Some(Went::Parked(Box::new(Parked { past, t, e, devs: owned.clone()?, ple: ple_owned.clone(), attn_rows, qsa: prompt_qsa, keys: prompt_keys, handoffs, at: i, kept, tapped })));
                }
                // the chunk before's last device done with (its scratch back) before this chunk's work there
                                if let Some(p) = prev.take() {
                    p.finish(self, kv);
                }
                mark("the chunk before finished");
                                d = dev;
                let mut r = chains[d].begin();
                r.keep_groups(keep);
                // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                r.rows_alike(keep);
                r.hold();
                open = Some(r);
            } else if dev != d || ple_here && ple_vs.is_none() {
                if let Some(p) = prev.take() {
                    p.finish(self, kv);
                }
                // (a handoff before this one's first: its streams up to its next device before that one's held work
                // is finished below)
                for (from, to, reads) in handoffs.drain(..) {
                    let mut got = from.finish().into_iter();
                    for &a in &reads {
                        let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                        to_store.push((a, kvrows, raw));
                    }
                    chains[to].upload(&devs[to].x, &got.next().expect("the streams"));
                }
                // what this device has pending, then the streams through the host (to the next device, or the
                // n-gram layer where it is not chained)
                let dv = &devs[d];
                let mut rec = open.take().unwrap_or_else(|| {
                    let mut r = chains[d].begin();
                    r.keep_groups(keep);
                    // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                    r.rows_alike(keep);
                    r
                });
                if let Some(routed) = pending.take() {
                    experts(&mut *rec, dv, i - 1, routed);
                    rec.stream_apply(&dv.x, &dv.moe_out, &dv.post2, t, s, h);
                }
                rec.read(&dv.x);
                let mut got = rec.finish().into_iter();
                for &a in &attn_reads {
                    let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                    to_store.push((a, kvrows, raw));
                }
                attn_reads.clear();
                let mut x = Tensor::from_vec(got.next().expect("the streams"), vec![t, s * h]);
                if ple_here && ple_vs.is_none() {
                    let b = self.devices[dev].as_ref();
                    let emb = b.to_device(Tensor::from_vec(ple_emb.clone(), vec![t, cfg.ple_dim]));
                    x = self.ple_forward(b, &b.to_device(x), &emb, kv).to_host();
                }
                d = dev;
                chains[d].upload(&devs[d].x, x.data());
            }
            let dv = &devs[d];
            let rec: &mut dyn ChainRecorder = &mut **open.get_or_insert_with(|| {
                let mut r = chains[d].begin();
                r.keep_groups(keep);
                // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                r.rows_alike(keep);
                // a prompt's chunk's work on its first device held till its handoff too, then let go at once: its
                // pieces submitted one by one as they were recorded ran the GPU's kernels some three times as long
                // (a chunk of 512 some 340 ms of them where held 100; a step's own few go as they are)
                if t > CHECK_ROWS && self.devices.len() > 1 {
                    r.hold();
                }
                r
            });
            if let (true, Some((p, v)), Some(window)) = (ple_here, ple_vs, &ple_window) {
                // the n-gram layer on its device: the last layer's experts written back first
                if let Some(routed) = pending.take() {
                    experts(&mut *rec, dv, i - 1, routed);
                    rec.stream_apply(&dv.x, &dv.moe_out, &dv.post2, t, s, h);
                }
                if let Some(backup) = undo.and_then(|u| u.window.as_ref()) {
                    rec.copy(window, 0, backup, 0, window.len);
                }
                p.key.mul(&mut *rec, s * h, cfg.ple_dim, &v.emb, &v.key, t);
                p.value.mul(&mut *rec, h, cfg.ple_dim, &v.emb, &v.value, t);
                rec.ple_gate(&v.key, &dv.x, &v.value, &p.norm_key, &p.norm_query, &p.norm_conv, &v.gated, &v.conv_in, t, s, h, eps);
                if kept.is_empty() {
                    rec.ple_conv(&dv.x, &v.gated, &v.conv_in, window, &p.conv, t, s * h, cfg.ple_kernel, cfg.ngram);
                } else {
                    // the conv in parts, each kept state's rows then the window's copy: the first part where the rows
                    // are, the later ones (few) through the parts' own vectors and back
                    let w = s * h;
                    let mut done = 0;
                    for (end, to) in kept.iter().map(|tap| (tap.row, tap.window.as_ref())).chain(std::iter::once((t, None))) {
                        let n = end - done;
                        if n > 0 && done == 0 {
                            rec.ple_conv(&dv.x, &v.gated, &v.conv_in, window, &p.conv, n, w, cfg.ple_kernel, cfg.ngram);
                        } else if n > 0 {
                            let seg = &segs.expect("a tapped run's later rows' vectors")[d];
                            let (sx, sg, sc) = (first_of(&seg.x, n * w), first_of(&seg.gated, n * w), first_of(&seg.conv_in, n * w));
                            rec.copy(&dv.x, done * w, &sx, 0, n * w);
                            rec.copy(&v.gated, done * w, &sg, 0, n * w);
                            rec.copy(&v.conv_in, done * w, &sc, 0, n * w);
                            rec.ple_conv(&sx, &sg, &sc, window, &p.conv, n, w, cfg.ple_kernel, cfg.ngram);
                            rec.copy(&sx, 0, &dv.x, done * w, n * w);
                        }
                        if let Some(to) = to {
                            rec.copy(window, 0, to, 0, window.len);
                        }
                        done = end;
                    }
                }
            }
            let applied = pending.take().map(|routed| experts(&mut *rec, dv, i - 1, routed)).is_some();
            hc(&mut *rec, dv, t, &cl.attn_hc, applied.then_some((&dv.moe_out, &dv.post2)), &dv.post, &dv.y_in);
            match (&layer.mixer, &cl.mixer) {
                (Mixer::Gdn(g), ChainMixer::Gdn { ba, conv, a, dt, norm, slot }) => {
                    let (sv, cv) = &states[*slot];
                    // a check's qkv and beta-alpha rows where undoing it finds them, its state and window backed up
                    let (qkv, bav) = match undo {
                        Some(u) => (&u.inputs[*slot].0, &u.inputs[*slot].1),
                        None => (&dv.qkv, &dv.ba),
                    };
                    rec.exl3_rows(chain_packed(&g.qkv)?, &dv.y_in, qkv, t);
                    rec.exl3_rows(chain_packed(&g.z)?, &dv.y_in, &dv.z, t);
                    ba.mul(&mut *rec, 2 * cfg.nv, h, &dv.y_in, bav, t);
                    if let Some(u) = undo {
                        rec.copy(sv, 0, &u.backups[*slot].0, 0, sv.len);
                        rec.copy(cv, 0, &u.backups[*slot].1, 0, cv.len);
                    }
                    let dn = DeltaNet { rows: t, v_heads: cfg.nv, k_heads: cfg.nk, k_dim: cfg.kd, v_dim: cfg.vd, scale_q: 1.0 / (cfg.vd as f32).sqrt(), eps, sigmoid_gate: true };
                    if kept.is_empty() {
                        rec.ssm_conv(qkv, conv, cv, &dv.conv, t, conv_dim, cfg.conv);
                        rec.delta_net(&dv.conv, &dv.z, bav, a, dt, norm, sv, &dv.core, dn);
                    } else {
                        // the recurrence in parts, each kept state's rows then its copies (as the n-gram layer's)
                        let vw = cfg.nv * cfg.vd;
                        let mut done = 0;
                        for (end, to) in kept.iter().map(|tap| (tap.row, Some(&tap.gdn[*slot]))).chain(std::iter::once((t, None))) {
                            let n = end - done;
                            if n > 0 && done == 0 {
                                rec.ssm_conv(qkv, conv, cv, &dv.conv, n, conv_dim, cfg.conv);
                                rec.delta_net(&dv.conv, &dv.z, bav, a, dt, norm, sv, &dv.core, DeltaNet { rows: n, ..dn });
                            } else if n > 0 {
                                let seg = &segs.expect("a tapped run's later rows' vectors")[d];
                                let (sq, sc, sz, sb, so) = (first_of(&seg.qkv, n * conv_dim), first_of(&seg.conv, n * conv_dim), first_of(&seg.z, n * vw), first_of(&seg.ba, n * 2 * cfg.nv), first_of(&seg.core, n * vw));
                                rec.copy(qkv, done * conv_dim, &sq, 0, n * conv_dim);
                                rec.copy(&dv.z, done * vw, &sz, 0, n * vw);
                                rec.copy(bav, done * 2 * cfg.nv, &sb, 0, n * 2 * cfg.nv);
                                rec.ssm_conv(&sq, conv, cv, &sc, n, conv_dim, cfg.conv);
                                rec.delta_net(&sc, &sz, &sb, a, dt, norm, sv, &so, DeltaNet { rows: n, ..dn });
                                rec.copy(&so, 0, &dv.core, done * vw, n * vw);
                            }
                            if let Some((ts, tc)) = to {
                                rec.copy(sv, 0, ts, 0, sv.len);
                                rec.copy(cv, 0, tc, 0, cv.len);
                            }
                            done = end;
                        }
                    }
                    rec.exl3_rows(chain_packed(&g.out)?, &dv.core, &dv.y_out, t);
                }
                (Mixer::Attn(a), ChainMixer::Attn { q_norm, k_norm, iq_norm, ik_norm, slot }) => {
                    let g = &m.kv[d];
                    let kvl = &g.layers[*slot];
                    rec.exl3_rows(chain_packed(&a.q)?, &dv.y_in, &dv.qfull, t);
                    rec.exl3_rows(chain_packed(&a.k)?, &dv.y_in, &dv.k, t);
                    rec.exl3_rows(chain_packed(&a.v)?, &dv.y_in, &dv.v, t);
                    rec.exl3_rows(chain_packed(&a.index_qk)?, &dv.y_in, &dv.index, t);
                    if on_gpu {
                        // the run's indexer keys out of the vector the device's next attention layer writes (a
                        // prompt's are read from the device's copy of them)
                        match few_set {
                            Some(f) => rec.copy_cols(&dv.index, &f.keys[d][*slot], t, id, iq + id, iq),
                            None if t == 1 => rec.copy(&dv.index, iq, &st.keys[d], slot * id, id),
                            None => {}
                        }
                    }
                    // and into the device's copy of the raw keys (QSA's pool reads them past the dense span)
                    match (t, few_set, &prompt_keys[d]) {
                        (1, _, _) => rec.copy(&dv.index, iq, &g.raw[*slot], past * id, id),
                        (_, Some(f), _) => rec.store_rows(&f.keys[d][*slot], &g.raw[*slot], t, id, past, id, 0),
                        (_, None, Some(keys)) => {
                            rec.copy_cols(&dv.index, keys, t, id, iq + id, iq);
                            rec.store_rows(keys, &g.raw[*slot], t, id, past, id, 0);
                        }
                        _ => unreachable!("a run's raw keys have a place"),
                    }
                    rec.copy_cols(&dv.qfull, &dv.q, t * nh, hd, 2 * hd, 0);
                    rec.copy_cols(&dv.qfull, &dv.gate, t * nh, hd, 2 * hd, hd);
                    rec.rmsnorm_rows(&dv.q, q_norm, &dv.qn, t * nh, eps);
                    rec.rmsnorm_rows(&dv.k, k_norm, &dv.kn, t * nkv, eps);
                    rec.rope_partial_rows(&dv.qn, t, nh, hd, rot, &dv.table);
                    rec.rope_partial_rows(&dv.kn, t, nkv, hd, rot, &dv.table);
                    rec.store_rows(&dv.kn, kvl, t, kvd, past, row, 0);
                    rec.store_rows(&dv.v, kvl, t, kvd, past, row, kvd);
                    let scale = 1.0 / (hd as f32).sqrt();
                    let qsa = g.qsa.as_ref().filter(|q| q.rows >= t).or(prompt_qsa[d].as_ref());
                    let out = match (&attn_rows[d], qsa) {
                        (_, Some(q)) if sparse => {
                            // past the dense span: the indexer's queries, the cache's pooled block keys, each row's top
                            // blocks, and its attention over them and its tail block
                            let (ih, nb, keep) = (cfg.index_heads, (past + t) / ratio, cfg.index_budget / ratio);
                            let (iqv, iqn, pooled, pooledn) = (first(&q.iq, t * ih * id), first(&q.iqn, t * ih * id), first(&q.pooled, nb * id), first(&q.pooledn, nb * id));
                            rec.copy_cols(&dv.index, &iqv, t, ih * id, iq + id, 0);
                            rec.rmsnorm_rows(&iqv, iq_norm, &iqn, t * ih, eps);
                            rec.rope_partial_rows(&iqn, t, ih, id, rot, &dv.table);
                            rec.qsa_pool(&g.raw[*slot], &pooled, nb, ratio, id);
                            rec.rmsnorm_rows(&pooled, ik_norm, &pooledn, nb, eps);
                            rec.rope_partial_rows(&pooledn, nb, 1, id, rot, &q.table);
                            rec.qsa_scores(&iqn, &pooledn, &q.scores, t, ih, id, nb, past, ratio, 1.0 / (id as f32).sqrt());
                            rec.qsa_select(&q.scores, &q.list, t, nb, past, ratio, keep);
                            rec.qsa_attention(&dv.qn, kvl, &q.list, &q.out, t, nh, nkv, hd, past, ratio, keep, scale);
                            &q.out
                        }
                        (Some(scratch), _) if few => {
                            // a row at a time as a step attends (the decode kernel's sums: a check's rows a step's bit
                            // for bit)
                            let (q1, qh) = (&few_set.expect("a few rows' vectors").q1[d], nh * hd);
                            for r in 0..t {
                                rec.copy(&dv.qn, r * qh, q1, 0, qh);
                                rec.attention(q1, kvl, &g.out, nh, nkv, hd, 0, past + r + 1, g.cap, scale);
                                rec.copy(&g.out, 0, scratch, r * qh, qh);
                            }
                            scratch
                        }
                        (Some(scratch), _) => {
                            rec.attention_rows(&dv.qn, kvl, scratch, t, nh, nkv, hd, past, None, scale);
                            scratch
                        }
                        (None, _) => {
                            rec.attention(&dv.qn, kvl, &g.out, nh, nkv, hd, 0, past + 1, g.cap, scale);
                            &g.out
                        }
                    };
                    rec.mul_sigmoid(out, &dv.gate, &dv.gated, t * nh * hd);
                    rec.exl3_rows(chain_packed(&a.o)?, &dv.gated, &dv.y_out, t);
                }
                _ => unreachable!("layer {i}'s vectors are its mixer's"),
            }
            hc(&mut *rec, dv, t, &cl.mlp_hc, Some((&dv.y_out, &dv.post)), &dv.post2, &dv.y2_in);
            cl.router.mul(&mut *rec, cfg.experts + 1, h, &dv.y2_in, &dv.router, t);
            if on_gpu && !cl.host && rec.moe_routed_into(layer.moe.experts.as_ref(), &dv.y2_in, &dv.x, &dv.post2, &dv.router, cfg.top_k, t, s) {
                // the experts after it, on the device, their sums into the streams; the recording goes on
                if let ChainMixer::Attn { slot, .. } = &cl.mixer {
                    rec.read_range(&m.kv[d].layers[*slot], past * row, t * row);
                    match few_set {
                        Some(f) => rec.read(&f.keys[d][*slot]),
                        None if t == 1 => rec.read_range(&st.keys[d], slot * id, id),
                        None => rec.read_range(&m.kv[d].raw[*slot], past * id, t * id),
                    }
                    attn_reads.push(i);
                }
                continue;
            }
            // The host between this layer's submits: to route its rows, or (a layer no GPU had room for) to run its
            // experts on their input.
            let width = cfg.experts + 1;
            // (a host layer's shared expert on the device where it is there: its outputs read back with the rest)
            let shared = cl.host && rec.moe_shared(layer.moe.experts.as_ref(), &dv.y2_in, &dv.moe_out, t);
            rec.read(&dv.router);
            if cl.host {
                rec.read_range(&dv.y2_in, 0, t * h);
            }
            if shared {
                rec.read_range(&dv.moe_out, 0, t * h);
            }
            let attn_slot = match &cl.mixer {
                ChainMixer::Attn { slot, .. } => {
                    rec.read_range(&m.kv[d].layers[*slot], past * row, t * row);
                    rec.read(&dv.index);
                    true
                }
                _ => false,
            };
            // (what this device's recording waits on first: the chunk before's run, and the streams of the device
            // before, whose handoff holds this one's work)
            if let Some(p) = prev.take() {
                p.finish(self, kv);
            }
            for (from, to, reads) in handoffs.drain(..) {
                let mut got = from.finish().into_iter();
                for &a in &reads {
                    let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                    to_store.push((a, kvrows, raw));
                }
                chains[to].upload(&devs[to].x, &got.next().expect("the streams"));
            }
            let waiting = std::time::Instant::now();
            let mut got = open.take().expect("the layer's recording").finish().into_iter();
            host_waited += waiting.elapsed().as_secs_f64() * 1e3;
            // (the device's attention layers before this one, routed on it: their rows and keys are this recording's)
            for a in attn_reads.drain(..) {
                let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                to_store.push((a, kvrows, raw));
            }
            let logits = got.next().expect("the router's logits");
            let input = cl.host.then(|| got.next().expect("the experts' input"));
            let shared = shared.then(|| got.next().expect("the shared expert's outputs"));
            if attn_slot {
                let (kvrows, index) = (got.next().expect("the run's K and V"), got.next().expect("the run's indexer keys"));
                let raw: Vec<f32> = index.chunks_exact(iq + id).take(t).flat_map(|r| r[iq..].iter().copied()).collect();
                to_cache(i, kvrows, raw, kv);
            }
            pending = Some(match input {
                Some(x) => {
                    let running = std::time::Instant::now();
                    let (x, logits) = (Tensor::from_vec(x, vec![t, h]), Tensor::from_vec(logits[..t * width].to_vec(), vec![t, width]));
                    let sums = match &shared {
                        Some(shared) => layer.moe.experts.forward_given(&x, &logits, cfg.top_k, shared),
                        None => layer.moe.experts.forward(&x, &logits, cfg.top_k),
                    };
                    host_layers += 1;
                    host_ran += running.elapsed().as_secs_f64() * 1e3;
                    Routed::Summed(d, sums.to_host().data().to_vec())
                }
                None => Routed::Host((0..t).map(|r| ggml_rs::exl3::route(&logits[r * width..(r + 1) * width], cfg.top_k)).collect()),
            });
        }
        // the last layer's experts, its write-back, the streams' collapse and the head, on the last device (the
        // chain's state saw to it)
        debug_assert_eq!(d, self.devices.len() - 1);
        let dv = &devs[d];
        let mut rec = open.take().unwrap_or_else(|| {
            let mut r = chains[d].begin();
            r.keep_groups(keep);
            // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
            r.rows_alike(keep);
            r
        });
        if let Some(routed) = pending.take() {
            experts(&mut *rec, dv, cfg.layers - 1, routed);
            rec.stream_apply(&dv.x, &dv.moe_out, &dv.post2, t, s, h);
        }
        // the trunk's streams for the prediction layer: a step's or a check's rows; a prompt's chunk fills the layer's
        // cache over its positions (but the last, whose next token it does not have) and keeps its last row
        let mut hid = None;
        if let (Some(mc), Some(mp)) = (&st.mtp, &self.mtp) {
            if t <= CHECK_ROWS {
                rec.copy(&dv.x, 0, &mc.hid, 0, t * s * h);
                hid = Some((past, t));
            } else {
                self.mtp_prompt(mp, mc, &mut m, chains[d], &mut *rec, &e, &dv.x, t, past, kv.id);
                rec.copy(&dv.x, (t - 1) * s * h, &mc.hid, 0, s * h);
                hid = Some((past + t - 1, 1));
            }
        }
        // (a greedy request's run: its rows' tokens picked here, those read where megabytes of logits were)
        let pick = self.pick.load(Ordering::Relaxed);
        if check {
            // every row's streams collapsed, then the head
            hc(&mut *rec, dv, t, &st.collapse, None, &dv.post, &dv.mixed);
            rec.exl3_rows(chain_packed(&self.head)?, &dv.mixed, &dv.head, t);
            if pick {
                rec.argmax_rows(&dv.head, t, cfg.vocab, &dv.picks);
                rec.read(&dv.picks);
            } else {
                rec.read(&dv.head);
            }
        } else {
            // the last row's streams collapsed (in a step's own vectors), then the head
            let one = &st.devs[d];
            if t > 1 {
                rec.copy(&dv.x, (t - 1) * s * h, &one.x, 0, s * h);
            }
            hc(&mut *rec, one, 1, &st.collapse, None, &one.post, &one.mixed);
            rec.exl3_rows(chain_packed(&self.head)?, &one.mixed, &one.head, 1);
            if pick {
                rec.argmax_rows(&one.head, 1, cfg.vocab, &one.picks);
                rec.read(&one.picks);
            } else {
                rec.read(&one.head);
            }
        }
        mark("last device recorded");
        if host_log && host_layers > 0 {
            eprintln!("    fn run of {t} at {past}: {host_layers} layers' experts on the host {host_ran:.2} ms, the waits for their inputs {host_waited:.2} ms, {:.2} ms in all so far", clock.elapsed().as_secs_f64() * 1e3);
        }
                // each handoff in turn: its device's streams (once it has run) up to the next, whose held work then goes
        for (from, to, reads) in handoffs.drain(..) {
            let mut got = from.finish().into_iter();
            mark("first device waited for and read");
            for &a in &reads {
                let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                to_store.push((a, kvrows, raw));
            }
            chains[to].upload(&devs[to].x, &got.next().expect("the streams"));
        }
        mark("streams up");
                // the last device's work going (held till its streams were up) as the host stores the others' rows
        rec.flush();
        mark("last device gone");
                if let Some(p) = prev.take() {
            p.finish(self, kv);
        }
        for (a, kvrows, raw) in to_store.drain(..) {
            to_cache(a, kvrows, raw, kv);
        }
        if let Some(at) = hid {
            m.mtp_hid = at;
        }
        if !resumed {
            kv.commit(t);
        }
        kv.dirty_from = usize::MAX;
        if t == 1 {
            self.decoded.store(true, Ordering::Relaxed);
        }
        st.runs.fetch_add(1, Ordering::Relaxed);
        mark("rows stored");
        if phases {
            eprintln!("    fn run of {t} at {past}: {}", marks.iter().map(|(w, ms)| format!("{w} {ms:.0}")).collect::<Vec<_>>().join(", "));
        }
        drop(kept);
        Some(Went::Run(ChainedRun { rec, attn_reads, at: past, t, tapped: std::mem::take(&mut tapped) }))
    }
}

/// The most rows of a chunk after the first state it keeps from inside ([`FlashNext::forward_chunks_tapped`]): those
/// rows' recurrences go through vectors of their own this long (a prompt's tail: the assistant's header and its last
/// token).
pub(crate) const TAP_ROWS: usize = 64;

/// A state a run keeps from inside it: each delta net's state and conv window (a pair a slot, on the layer's device)
/// and the n-gram layer's window as they are once the run's first `row` rows are through them, copied there as the
/// run goes.
struct Tap {
    row: usize,
    gdn: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    window: Option<ggml_rs::DeviceVec>,
}

/// The vectors a tapped run's later rows' recurrences go through on a device ([`TAP_ROWS`] of them): a delta net's
/// conv input and output, gate, beta-alpha and output; the n-gram layer's streams, gate and conv input.
struct TapSeg {
    qkv: ggml_rs::DeviceVec,
    conv: ggml_rs::DeviceVec,
    z: ggml_rs::DeviceVec,
    ba: ggml_rs::DeviceVec,
    core: ggml_rs::DeviceVec,
    x: ggml_rs::DeviceVec,
    gated: ggml_rs::DeviceVec,
    conv_in: ggml_rs::DeviceVec,
}

/// `v`'s first `len` elements as a vector of their own (the same buffer).
fn first_of(v: &ggml_rs::DeviceVec, len: usize) -> ggml_rs::DeviceVec {
    assert!(len <= v.len, "{len} of a vector of {}", v.len);
    ggml_rs::DeviceVec { len, inner: Arc::clone(&v.inner) }
}

/// How much of a chained run [`FlashNext::run_part`] makes: all of it, a prompt chunk's first device's part, or the
/// rest of a chunk parked after that.
enum Stage<'a> {
    Whole,
    First,
    Rest(Box<Parked<'a>>),
}

/// What [`FlashNext::run_part`] left: a run whose last device is running, or a chunk parked after its first device's
/// part.
enum Went<'a> {
    Run(ChainedRun<'a>),
    Parked(Box<Parked<'a>>),
}

/// A prompt's chunk whose first device's layers are recorded and gone: its positions, its embeddings (the prediction
/// layer's), its own vectors on every device, its first device's recording (its streams and rows read at the
/// handoff), and the layer its next device's begin at.
struct Parked<'a> {
    past: usize,
    t: usize,
    e: Tensor,
    devs: Arc<Vec<ChainDev>>,
    ple: Option<Arc<PleVecs>>,
    attn_rows: Vec<Option<ggml_rs::DeviceVec>>,
    qsa: Arc<Vec<Option<QsaVecs>>>,
    keys: Vec<Option<ggml_rs::DeviceVec>>,
    handoffs: Vec<(Box<dyn ggml_rs::ChainRecorder + 'a>, usize, Vec<usize>)>,
    at: usize,
    /// The states the chunk keeps from inside it (their vectors, and what the caller is given of them).
    kept: Vec<Tap>,
    tapped: Vec<llama_rs::Tapped>,
}

/// A chained run whose last device is running: its recording (the run's reads its attention layers' rows, then
/// the logits) and where its rows go in the host's cache.
struct ChainedRun<'a> {
    rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    attn_reads: Vec<usize>,
    at: usize,
    t: usize,
    /// The states it keeps from inside (theirs once the run has run)
    tapped: Vec<llama_rs::Tapped>,
}

impl ChainedRun<'_> {
    /// Its last device waited for: its attention layers' rows into the host's cache, the logits.
    fn finish(self, fnx: &FlashNext, kv: &mut KvCache) -> Vec<f32> {
        let mut got = self.rec.finish().into_iter();
        for &a in &self.attn_reads {
            let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
            fnx.cache_rows(kv, a, self.at, self.t, kvrows, raw);
        }
        kv.dirty_from = usize::MAX;
        got.next().expect("the logits")
    }
}

impl FlashNext {
    /// The devices its layers are over.
    pub fn devices_len(&self) -> usize {
        self.devices.len()
    }

    /// The rows a prompt's chunk has at most: 512 (OAIY_FN_ROWS: as given, 64 to 1,024). A chunk's experts' weights
    /// are decoded once a block of their rows, and a chunk of 512 gives each of the 512 experts some 10 rows of a
    /// block's 32, so a chunk of 1,024 takes less of the GPUs a token (its kernels 298 ms where two of 512 take 342).
    /// It is not the faster for that over two cards (2,148 tokens run together 679 to 691 ms in chunks of 1,024 where
    /// 602 to 612 in 512s: fewer chunks one behind the other), and with 24 GB of weights a card its vectors do not
    /// fit two RTX 5090s' 32 GB (out of memory at the server's second prompt); past 1,638 rows a kernel's grid is
    /// wider than a dispatch may be.
    pub fn prompt_rows(&self) -> usize {
        static ASKED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        ASKED.get_or_init(|| std::env::var("OAIY_FN_ROWS").ok().and_then(|v| v.parse().ok()).filter(|n| (64..=1024).contains(n))).unwrap_or(512)
    }

    /// The layers whose experts run on the host (no GPU had room for them).
    pub fn host_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.moe.experts.on_host()).count()
    }

    /// Whether the chain drafts tokens (its multi-token-prediction layer loaded and chained).
    pub fn drafts(&self) -> bool {
        std::env::var_os("OAIY_NO_CHAIN").is_none() && self.chain_state().is_some_and(|c| c.mtp.is_some())
    }

    /// QSA's vectors on `c` for runs of up to `rows` rows over a cache of `cap` positions.
    fn qsa_vecs(&self, c: &dyn ggml_rs::DeviceChain, rows: usize, cap: usize) -> QsaVecs {
        let cfg = &self.config;
        let (ratio, id, keep) = (cfg.index_ratio, cfg.index_dim, cfg.index_budget / cfg.index_ratio);
        let blocks = cap / ratio;
        let v = |n: usize| c.vec(n.max(1));
        let table = v(blocks * cfg.rope_dim);
        c.upload(&table, &self.rope_table_of((0..blocks).map(|j| j * ratio)));
        QsaVecs {
            rows,
            iq: v(rows * cfg.index_heads * id),
            iqn: v(rows * cfg.index_heads * id),
            pooled: v(blocks * id),
            pooledn: v(blocks * id),
            table,
            scores: v(rows * blocks),
            list: v(rows * keep),
            out: v(c.qsa_attention_out_len(rows, cfg.heads, cfg.head_dim, keep, ratio)),
        }
    }

    /// The prediction layer's cache with room for `len` positions (grown as the trunk's copy is).
    fn mtp_reserve<'a>(c: &dyn ggml_rs::DeviceChain, g: &'a mut Option<MtpKv>, cfg: &Config, len: usize) -> &'a mut MtpKv {
        let row = 2 * cfg.kv_heads * cfg.head_dim;
        let g = g.get_or_insert_with(|| MtpKv { layer: c.vec(1), cap: 0, out: c.vec(1), owner: 0, start: 0, valid: 0 });
        if g.cap < len {
            let cap = len.next_power_of_two().max(256);
            g.layer = if g.cap == 0 { c.vec(cap * row) } else { c.resize(&g.layer, cap * row) };
            g.out = c.vec(c.attention_out_len(cfg.heads, cfg.head_dim, cap));
            g.cap = cap;
        }
        g
    }

    /// The partial RoPE's sines and cosines at positions `at..at + rows`.
    fn rope_table(&self, at: usize, rows: usize) -> Vec<f32> {
        self.rope_table_of(at..at + rows)
    }

    /// The partial RoPE's sines and cosines at `positions`.
    fn rope_table_of(&self, positions: impl Iterator<Item = usize>) -> Vec<f32> {
        let (cfg, rot) = (&self.config, self.config.rope_dim);
        positions
            .flat_map(|pos| {
                (0..rot / 2).flat_map(move |k| {
                    let (sn, cs) = (pos as f32 * cfg.rope_theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                    [sn, cs]
                })
            })
            .collect()
    }

    /// The prediction layer's cache at a prompt's chunk (`t` rows at `past`, its embeddings `e` and its streams after
    /// the trunk's last layer `x`): its K and V for the chunk's positions but the last, each from the row's streams and
    /// the next row's token, where its entries run unbroken from the cache's start up to the chunk (else none: the
    /// next draft's start them over).
    #[allow(clippy::too_many_arguments)]
    fn mtp_prompt(&self, mp: &FnMtp, mc: &MtpChain, m: &mut ChainMut, c: &dyn ggml_rs::DeviceChain, rec: &mut dyn ggml_rs::ChainRecorder, e: &Tensor, x: &ggml_rs::DeviceVec, t: usize, past: usize, owner: u64) {
        let cfg = &self.config;
        let (h, s, eps) = (cfg.hidden, cfg.streams, cfg.eps);
        let (nkv, hd) = (cfg.kv_heads, cfg.head_dim);
        let (kvd, row) = (nkv * hd, 2 * nkv * hd);
        let g = Self::mtp_reserve(c, &mut m.mtp_kv, cfg, past + t);
        if g.owner != owner || g.valid != past || g.start != 0 {
            if past != 0 {
                g.owner = 0;
                g.valid = 0;
                return;
            }
            g.start = 0;
        }
        g.owner = owner;
        let rows = t - 1;
        if rows == 0 {
            g.valid = past;
            return;
        }
        let (Some(fc_e), Some(fc_h), Some(kw), Some(vw)) = (chain_packed(&mp.fc_e), chain_packed(&mp.fc_h), chain_packed(&mp.k), chain_packed(&mp.v)) else { return };
        let v = |n: usize| c.vec(n);
        let (ev, en, e2, hn, xs, normed, tt, post, logits, mixed, kk, vv, kn, table) =
            (v(rows * h), v(rows * h), v(rows * h), v(rows * s * h), v(rows * s * h), v(rows * s * h), v(rows * (self.chain_state().map_or(0, |st| st.rank) + s)), v(rows * s), v(rows * s * h), v(rows * h), v(rows * kvd), v(rows * kvd), v(rows * kvd), v(rows * cfg.rope_dim));
        c.upload(&ev, &e.data()[h..t * h]);
        c.upload(&table, &self.rope_table(past, rows));
        rec.rmsnorm_rows(&ev, &mc.enorm, &en, rows, eps);
        rec.exl3_rows(fc_e, &en, &e2, rows);
        let xv = ggml_rs::DeviceVec { len: rows * s * h, inner: Arc::clone(&x.inner) };
        rec.rmsnorm_rows(&xv, &mc.hnorm, &hn, rows, eps);
        rec.exl3_rows(fc_h, &hn, &xs, rows * s);
        let ones = v(rows * s);
        c.upload(&ones, &vec![1.0; rows * s]);
        rec.stream_apply(&xs, &e2, &ones, rows, s, h);
        let hcv = &mc.attn_hc;
        rec.rmsnorm_streams(&xs, &hcv.norm, &normed, rows, s, eps);
        hcv.project(&mut *rec, &normed, &tt, &post, &logits, &mixed, rows, s, h);
        rec.exl3_rows(kw, &mixed, &kk, rows);
        rec.exl3_rows(vw, &mixed, &vv, rows);
        rec.rmsnorm_rows(&kk, &mc.k_norm, &kn, rows * nkv, eps);
        rec.rope_partial_rows(&kn, rows, nkv, hd, cfg.rope_dim, &table);
        rec.store_rows(&kn, &g.layer, rows, kvd, past, row, 0);
        rec.store_rows(&vv, &g.layer, rows, kvd, past, row, kvd);
        g.valid = past + rows;
    }

    /// Draft up to `k` tokens after `next` (the token sampled for position `kv.len`): the prediction layer's entries
    /// caught up to it first (each position from the trunk's streams there, kept from the last run, and the token after;
    /// `tokens` the tokens at positions up to `kv.len`, ending with `next`), its last row giving the first draft; each
    /// further draft from the layer's own output streams and the draft before. A draft the layer gives less than
    /// [`DRAFT_MIN_P`] ends them (none where the first is such). None where the chain does not draft or the streams it
    /// needs are not kept.
    pub fn draft(&self, kv: &KvCache, tokens: &[u32], k: usize) -> Option<Vec<u32>> {
        self.draft_above(kv, tokens, k, DRAFT_MIN_P)
    }

    /// [`Self::draft`], a draft the layer gives less than `min_p` ending them.
    pub(crate) fn draft_above(&self, kv: &KvCache, tokens: &[u32], k: usize, min_p: f64) -> Option<Vec<u32>> {
        use ggml_rs::ChainRecorder;
        if !self.drafts() || k == 0 {
            return None;
        }
        let (st, mp) = (self.chain_state()?, self.mtp.as_ref()?);
        let mc = st.mtp.as_ref()?;
        let last = self.devices.len() - 1;
        let c = self.devices[last].chain()?;
        let cfg = &self.config;
        let (h, s, eps) = (cfg.hidden, cfg.streams, cfg.eps);
        let (nh, nkv, hd, rot) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.rope_dim);
        let (kvd, row, qh) = (nkv * hd, 2 * nkv * hd, nh * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let n = kv.len;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        let (hid_at, hid_rows) = m.mtp_hid;
        // the rows the streams cover, up to the trunk's last position
        if hid_rows == 0 || hid_at + hid_rows != n || tokens.len() < hid_rows {
            return None;
        }
        let tokens = &tokens[tokens.len() - hid_rows..];
        let g = Self::mtp_reserve(c, &mut m.mtp_kv, cfg, n + k + 1);
        // entries from the streams' first row on; before it, the run of true ones if it reaches it, else none
        if g.owner != kv.id || g.valid < hid_at || g.valid > n {
            g.start = hid_at;
        }
        g.owner = kv.id;
        let dev = |rows: usize| {
            mc.devs[rows - 1].get_or_init(|| {
                let v = |n: usize| c.vec(rows * n);
                MtpDev { dv: chain_dev(c, cfg, st.rank, rows, 1), e: v(h), en: v(h), e2: v(h), hin: v(s * h), hn: v(s * h), att: v(qh), q1: c.vec(qh) }
            })
        };
        let one = dev(1);
        fn packed(w: &Weight) -> &dyn PackedLinear {
            chain_packed(w).expect("a chained layer's matrix")
        }
        let mut drafts = Vec::with_capacity(k);
        // pass 0: the caught-up rows (the streams kept); then a row a draft (the layer's own output)
        for pass in 0..k {
            let (rows, at) = if pass == 0 { (hid_rows, hid_at) } else { (1, n + pass - 1) };
            let md = dev(rows);
            let dv = &md.dv;
            let next: Vec<u32> = if pass == 0 { tokens.to_vec() } else { vec![drafts[pass - 1]] };
            c.upload(&md.e, self.embed_text(&next).ok()?.data());
            c.upload(&dv.table, &self.rope_table(at, rows));
            let mut rec = c.begin();
            rec.keep_groups(true);
            // (a GGUF's head from int8 activations, as the trunk's rows take it: half a draft's GPU time in f32, and
            // a draft is only what a check then takes or refuses)
            rec.rows_alike(true);
            if pass == 0 {
                rec.copy(&mc.hid, 0, &md.hin, 0, rows * s * h);
            } else {
                rec.copy(&one.dv.x, 0, &md.hin, 0, s * h);
            }
            // the inputs: the embedding's projection in every stream of the streams' own
            rec.rmsnorm_rows(&md.e, &mc.enorm, &md.en, rows, eps);
            rec.exl3_rows(packed(&mp.fc_e), &md.en, &md.e2, rows);
            rec.rmsnorm_rows(&md.hin, &mc.hnorm, &md.hn, rows, eps);
            rec.exl3_rows(packed(&mp.fc_h), &md.hn, &dv.x, rows * s);
            rec.stream_apply(&dv.x, &md.e2, &mc.ones, rows, s, h);
            let hc = |rec: &mut dyn ChainRecorder, hcv: &HcVecs, pending: Option<(&ggml_rs::DeviceVec, &ggml_rs::DeviceVec)>, post: &ggml_rs::DeviceVec, out: &ggml_rs::DeviceVec, dv: &ChainDev, rows: usize| {
                if let Some((y, p)) = pending {
                    rec.stream_apply(&dv.x, y, p, rows, s, h);
                }
                rec.rmsnorm_streams(&dv.x, &hcv.norm, &dv.normed, rows, s, eps);
                hcv.project(&mut *rec, &dv.normed, &dv.t, post, &dv.logits, out, rows, s, h);
            };
            // attention over its own cache, a row at a time from its entries' start
            hc(&mut *rec, &mc.attn_hc, None, &dv.post, &dv.y_in, dv, rows);
            rec.exl3_rows(packed(&mp.q), &dv.y_in, &dv.qfull, rows);
            rec.exl3_rows(packed(&mp.k), &dv.y_in, &dv.k, rows);
            rec.exl3_rows(packed(&mp.v), &dv.y_in, &dv.v, rows);
            rec.copy_cols(&dv.qfull, &dv.q, rows * nh, hd, 2 * hd, 0);
            rec.copy_cols(&dv.qfull, &dv.gate, rows * nh, hd, 2 * hd, hd);
            rec.rmsnorm_rows(&dv.q, &mc.q_norm, &dv.qn, rows * nh, eps);
            rec.rmsnorm_rows(&dv.k, &mc.k_norm, &dv.kn, rows * nkv, eps);
            rec.rope_partial_rows(&dv.qn, rows, nh, hd, rot, &dv.table);
            rec.rope_partial_rows(&dv.kn, rows, nkv, hd, rot, &dv.table);
            rec.store_rows(&dv.kn, &g.layer, rows, kvd, at, row, 0);
            rec.store_rows(&dv.v, &g.layer, rows, kvd, at, row, kvd);
            for r in 0..rows {
                rec.copy(&dv.qn, r * qh, &md.q1, 0, qh);
                rec.attention(&md.q1, &g.layer, &g.out, nh, nkv, hd, g.start.min(at + r), at + r + 1, g.cap, scale);
                rec.copy(&g.out, 0, &md.att, r * qh, qh);
            }
            rec.mul_sigmoid(&md.att, &dv.gate, &dv.gated, rows * qh);
            rec.exl3_rows(packed(&mp.o), &dv.gated, &dv.y_out, rows);
            // the experts, their write the layer's output streams
            hc(&mut *rec, &mc.mlp_hc, Some((&dv.y_out, &dv.post)), &dv.post2, &dv.y2_in, dv, rows);
            mc.router.mul(&mut *rec, cfg.experts + 1, h, &dv.y2_in, &dv.router, rows);
            assert!(rec.moe_routed_into(mp.experts.as_ref(), &dv.y2_in, &dv.x, &dv.post2, &dv.router, cfg.top_k, rows, s), "the prediction layer's experts route on their GPU");
            // the last row's streams (the next pass's input), collapsed, then the head: the next draft
            if rows > 1 {
                rec.copy(&dv.x, (rows - 1) * s * h, &one.dv.x, 0, s * h);
            }
            hc(&mut *rec, &mc.mixer, None, &one.dv.post, &one.dv.mixed, &one.dv, 1);
            rec.exl3_rows(chain_packed(&self.head)?, &one.dv.mixed, &one.dv.head, 1);
            // the draft and its probability under the layer (the largest logit's share of their exponentials)
            rec.argmax_softmax(&one.dv.head, &mc.best);
            rec.read(&mc.best);
            let got = rec.finish().pop().expect("the draft");
            let (best, total) = (got[0].to_bits(), got[2] as f64);
            if pass == 0 {
                // the layer's entries are true ones up to the trunk's last position, the draft taken or not
                g.valid = n;
            }
            if 1.0 / total < min_p {
                break;
            }
            drafts.push(best);
        }
        Some(drafts)
    }
}

/// Whether decode steps run as graphs (FLASHNEXT_GRAPHS=0 turns them off).
fn graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Graphs are CUDA's: a build without it runs every step uncaptured.
    *ON.get_or_init(|| false && std::env::var("FLASHNEXT_GRAPHS").map_or(true, |v| v != "0"))
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

#[cfg(test)]
mod row_tests {
    use super::*;

    /// A trellis row of the n-gram table decodes as reading each value's state a bit at a time does (the code the
    /// decode replaced): every width of code, a row of random bytes.
    #[test]
    fn a_trellis_row_decodes_as_its_states_read_a_bit_at_a_time() {
        let codebook = mul1_codebook();
        let bias: Vec<f32> = (0..ROW_DIM).map(|i| i as f32 * 0.01 - 0.5).collect();
        let mut seed = 12345u32;
        for k in 1..=8usize {
            let words = 1 + ROW_DIM * k / 16;
            let bytes: Vec<u8> = (0..2 * words)
                .map(|i| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // a sane f16 scale in the first word
                    match i {
                        0 => 0x00,
                        1 => 0x3c,
                        _ => (seed >> 24) as u8,
                    }
                })
                .collect();
            let w: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let scale = dsv41::formats::f16_to_f32(w[0]);
            let bit = |i: usize| (w[1 + i / 16] >> (i % 16)) & 1;
            let want: Vec<f32> = (0..ROW_DIM)
                .map(|i| {
                    let mut state = 0usize;
                    for m in 0..16 {
                        let src = ((i + ROW_DIM * 16 - m / k) % ROW_DIM) * k + m % k;
                        state |= (bit(src) as usize) << m;
                    }
                    codebook[state] * scale + bias[i]
                })
                .collect();
            let mut got = vec![0f32; ROW_DIM];
            decode_row(&bytes, k, &codebook, &bias, &mut got);
            assert_eq!(got, want, "codes of {k} bits");
        }
    }
}
