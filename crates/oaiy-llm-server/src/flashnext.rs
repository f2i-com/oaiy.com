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
mod chain;
mod chain_part;
mod chain_state;
mod load;
mod ngram;
mod runs;
// (what the files make for each other, and for the rest of the server what they made for it)
pub(crate) use {chain_state::*, load::*, runs::*};
use ngram::*;

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
            // A prompt's chunk on one card: its pieces two at a time on the queue, each encoded at its turn, as two
            // cards' chunks are paced ([`Self::chunks_run`]). Recorded and submitted as fast as the host goes, a card
            // under a power limit ran such chunks a third as fast for whole prompts (4,086 tokens in 8.6 s where 3.5).
            struct Paced<'a>(Vec<&'a dyn ggml_rs::DeviceChain>);
            impl Drop for Paced<'_> {
                fn drop(&mut self) {
                    for c in &self.0 {
                        c.pieces_in_flight_at_most(0);
                    }
                }
            }
            let alone = tokens.len() > CHECK_ROWS && self.devices.len() == 1 && std::env::var_os("OAIY_FN_UNPACED").is_none();
            let paced = Paced(if alone { self.devices.iter().filter_map(|b| b.chain()).collect() } else { Vec::new() });
            for c in &paced.0 {
                c.pieces_in_flight_at_most(2);
            }
            let logits = self.forward_chained(tokens, embeds, kv);
            drop(paced);
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

/// OAIY_ROUTE_DUMP names a file: each row's routed experts of each layer the host routes (OAIY_HOST_ROUTE: all of
/// them) appended to it, a line a row (the layer, the row's position, its experts), for a look at a model's experts'
/// use (which a card that does not hold them all would keep).
fn route_dump(layer: usize, past: usize, assign: &[Vec<(usize, f32)>]) {
    use std::io::Write;
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>> = std::sync::OnceLock::new();
    let Some(file) = FILE.get_or_init(|| {
        let path = std::env::var_os("OAIY_ROUTE_DUMP")?;
        let f = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
        Some(std::sync::Mutex::new(std::io::BufWriter::new(f)))
    }) else {
        return;
    };
    let mut f = file.lock().unwrap_or_else(|p| p.into_inner());
    for (r, a) in assign.iter().enumerate() {
        let experts: Vec<String> = a.iter().map(|(e, _)| e.to_string()).collect();
        let _ = writeln!(f, "{layer} {} {}", past + r, experts.join(" "));
    }
    // (a step's line is on disk when the run is looked at)
    if assign.len() <= 8 && layer % 16 == 15 {
        let _ = f.flush();
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
