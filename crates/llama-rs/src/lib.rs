//! Top-level API. Loads a GGUF, dispatches to the right architecture, runs
//! inference.
//!
//! Supported architectures:
//!   * Llama 1/2/3, Mistral, Qwen-2 (`LlamaModel`)
//!   * Qwen-3 dense (`Qwen3Model`)
//!   * Gemma 3 dense (`Gemma3Model`)
//!   * Gemma 4 (`Gemma4Model`)
//!   * Gemma 3n MatFormer (`Gemma3nModel`) — coherent output
//!
//! Use [`Model::load`] for arch-agnostic loading; the returned enum has a
//! `forward` method that dispatches to the right impl.
//!
//! Roadmap (see `README.md`): Qwen-3.5 (incl. linear-attention), Phi-3,
//! Mixtral / Qwen3-MoE, persistent on-GPU tensors, fused attention kernels.

#![deny(rust_2018_idioms)]

mod chain_decode;
mod chain_qwen35;
pub use chain_qwen35::{Qwen35Chain, SPEC_ROWS};
pub mod chat;
pub mod config;
pub mod gemma3;
pub mod gemma4;
pub mod gemma3n;
pub mod json_fsm;
pub mod gemma4moe;
pub mod kv_cache;
pub mod llama;
pub mod loader;
pub mod mixtral;
pub mod mmproj;
pub mod moe;
// VENDORED-LOCAL: MOE-01/MOE-02 — grouped CUDA MoE decode (see moe_cuda.rs).
#[cfg(feature = "cuda")]
mod moe_cuda;
pub mod qwen3;
pub mod glm5next; // VENDORED-LOCAL: GLM-5.3-Flash
pub mod qwen3moe;
pub mod qwen35;
pub mod qwen35moe;
pub mod sampler;
pub mod vision;
// VENDORED-LOCAL: streaming MoE expert backend — experts read on demand
// from the .gguf file through OAIY's bounded Ecache instead of
// materialized in RAM (`Model::open_streaming`).
pub mod expert_stream;

use std::sync::Arc;

pub use chat::{apply_chat_template, chat_stop_tokens, ChatMessage, Role};
// VENDORED-LOCAL: GLM-5.3-Flash. Its template takes a reasoning effort and a
// choice of thinking or not, which a server needs to pass through.
pub use chat::{glm5next_template_full, glm5next_template_with, ReasoningEffort};
pub use config::{Architecture, LlamaConfig, ModelConfig};
pub use gemma3::Gemma3Model;
pub use gemma3n::Gemma3nModel;
pub use gemma4::Gemma4Model;
pub use gemma4moe::Gemma4MoeModel;
pub use kv_cache::KvCache;
pub use llama::LlamaModel;
pub use mixtral::{MixtralBlock, MixtralModel};
pub use loader::{
    load_tensor_f32, BlockTensors, CommonBlockTensors, CommonTensors, ModelTensors, TensorIndex,
};
pub use qwen3::Qwen3Model;
pub use qwen3moe::{Qwen3MoeBlock, Qwen3MoeModel};
// VENDORED-LOCAL: GLM-5.3-Flash
pub use glm5next::{Glm5NextBlock, Glm5NextConfig, Glm5NextModel, LayerKind};
pub use qwen35::{Qwen35Block, Qwen35Model, SsmConfig};
pub mod multimodal_rope;
pub use qwen35moe::{Qwen35MoeBlock, Qwen35MoeModel};
pub use sampler::{SampleParams, Sampler};
pub use mmproj::{Gemma4VMmProj, MmProj, MmProjConfig, Projector, ProjectorKind, SigLipMmProj};
pub use moe::{moe_forward, MoeFfn};
pub use vision::{preprocess_image, preprocess_image_bytes, VisionConfig};

use ggml_rs::{Backend, Tensor};
use gguf::GgufFile;
use thiserror::Error;
use tokenizer::Tokenizer;

#[derive(Debug, Error)]
pub enum LlamaError {
    #[error(transparent)]
    Gguf(#[from] gguf::GgufError),

    #[error(transparent)]
    Quant(#[from] ggml_quants::QuantError),

    #[error(transparent)]
    Tokenizer(#[from] tokenizer::TokenizerError),

    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),

    #[error("expected tensor `{0}` not found in GGUF")]
    MissingTensor(String),

    #[error("tensor `{name}` had unexpected shape {got:?}, expected {expected:?}")]
    BadTensorShape {
        name:     String,
        got:      Vec<u64>,
        expected: Vec<u64>,
    },

    #[error("config error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, LlamaError>;

/// Architecture-dispatched model. The variant is determined by
/// `general.architecture` in the GGUF.
#[derive(Debug)]
pub enum Model {
    Llama(LlamaModel),
    Qwen3(Qwen3Model),
    Gemma3(Gemma3Model),
    /// Gemma 3n MatFormer. AltUp / PLE / Laurel / activation sparsity / shared
    /// K/V are all wired up and `forward()` produces coherent output. The
    /// non-obvious bit: Gemma 3n hardcodes `attention_scale = 1.0` (not
    /// `1/sqrt(head_dim)` like every other arch). See `gemma3n.rs`.
    Gemma3n(Gemma3nModel),
    /// Gemma 4 (Apache-2.0). Per-layer head_dim / RoPE base / FFN size; SWA
    /// pattern from explicit bool array. See `gemma4.rs` for the gap list.
    Gemma4(Gemma4Model),
    /// Qwen 3.5 — Mamba2/attention hybrid. Loader recognises both layer types
    /// (every `full_attention_interval`-th layer is full attention; the rest are
    /// SSM/Mamba2). Forward not yet implemented — calling `forward` returns a
    /// clear error pointing to task #61.
    Qwen35(Qwen35Model),
    /// Qwen3-MoE — Qwen3 attention backbone + Mixture-of-Experts FFN.
    /// Targets Qwen3-30B-A3B / Qwen3.6-30B-A3B (typically 128 experts, top-8).
    Qwen3Moe(Qwen3MoeModel),
    // VENDORED-LOCAL: GLM-5.3-Flash (`glm5next`). 45-layer trunk of 34 KDA
    /// linear-attention + 11 MLA layers, 288 experts top-8 + 1 shared, and a
    /// 4-stream hyper-connection residual. Loads and validates; `forward`
    /// reports the pending pieces. See `glm5next.rs`.
    Glm5Next(Glm5NextModel),
    /// Mixtral (8x7B / 8x22B) — `arch=llama` in GGUF + `expert_count > 0` in
    /// metadata. Standard Llama attention + per-block MoE FFN.
    Mixtral(MixtralModel),
    /// Gemma 4 MoE (26B-A4B and friends) — `arch=gemma4` + `expert_count > 0`.
    /// Hybrid dense + MoE per layer (shared expert + routed experts), custom
    /// router input pipeline (norm + 1/sqrt(n_embd) + per-channel scale), per-
    /// expert output scalars. See `gemma4moe.rs`.
    Gemma4Moe(Gemma4MoeModel),
    /// Qwen 3.6-MoE (35B-A3B) — `arch=qwen35moe`. SSM hybrid backbone +
    /// 256-expert MoE + sigmoid-gated shared expert per layer (DeepSeek-V2 /
    /// Qwen3-Next pattern). See `qwen35moe.rs`.
    Qwen35Moe(Qwen35MoeModel),
}

impl Model {
    pub fn load(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let arch_str = g.architecture()?.to_string();
        let arch = Architecture::from_str(&arch_str);
        match arch {
            Architecture::Llama | Architecture::Mistral | Architecture::Qwen2 => {
                // Mixtral / other Llama-arch MoE GGUFs set <arch>.expert_count
                // > 0. Detect early and route to the MoE loader.
                let n_experts = g.get_u64(&format!("{arch_str}.expert_count")).unwrap_or(0);
                if n_experts > 0 {
                    Ok(Self::Mixtral(MixtralModel::from_gguf(g, backend)?))
                } else {
                    Ok(Self::Llama(LlamaModel::from_gguf(g, backend)?))
                }
            }
            Architecture::Qwen3 => Ok(Self::Qwen3(Qwen3Model::from_gguf(g, backend)?)),
            Architecture::Gemma3 => Ok(Self::Gemma3(Gemma3Model::from_gguf(g, backend)?)),
            Architecture::Gemma3n => Ok(Self::Gemma3n(Gemma3nModel::from_gguf(g, backend)?)),
            Architecture::Gemma4 => {
                let n_experts = g.get_u64("gemma4.expert_count").unwrap_or(0);
                if n_experts > 0 {
                    Ok(Self::Gemma4Moe(Gemma4MoeModel::from_gguf(g, backend)?))
                } else {
                    Ok(Self::Gemma4(Gemma4Model::from_gguf(g, backend)?))
                }
            }
            Architecture::Qwen35 => Ok(Self::Qwen35(Qwen35Model::from_gguf(g, backend)?)),
            Architecture::Qwen3Moe => Ok(Self::Qwen3Moe(Qwen3MoeModel::from_gguf(g, backend)?)),
            // VENDORED-LOCAL: glm5next loads fully; `forward` is the stub.
            Architecture::Glm5Next => Ok(Self::Glm5Next(Glm5NextModel::from_gguf(g, backend)?)),
            // Qwen3-VL-MoE has the same per-block structure as qwen3moe (full
            // attention + per-block routed MoE, no shared expert), so it routes
            // to the same Qwen3MoeModel — Qwen3MoeModel reads metadata under
            // `<arch>.*` so the namespace difference is handled transparently.
            // The vision tower is loaded separately via the qwen3vl_merger
            // mmproj path.
            Architecture::Qwen3VlMoe => Ok(Self::Qwen3Moe(Qwen3MoeModel::from_gguf(g, backend)?)),
            Architecture::Qwen35Moe => Ok(Self::Qwen35Moe(Qwen35MoeModel::from_gguf(g, backend)?)),
            Architecture::Qwen36MoeVl => Err(LlamaError::Config(
                "Qwen3.6-VL-MoE arch detected. Three pending pieces: MoE loader \
                 (#85), Qwen3-VL vision tower (#84), and integration of both \
                 with the qwen35 dense path. The text-only 27B dense variant \
                 already works (v0.55).".into()
            )),
            Architecture::Unsupported(s) => Err(LlamaError::UnsupportedArch(s)),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        match self {
            Self::Llama(m)    => &m.config,
            Self::Qwen3(m)    => &m.config,
            Self::Gemma3(m)   => &m.config,
            Self::Gemma3n(m)  => &m.config,
            Self::Gemma4(m)   => &m.config,
            Self::Qwen35(m)   => &m.config,
            Self::Qwen3Moe(m) => &m.config,
            Self::Glm5Next(m) => &m.config,
            Self::Mixtral(m)  => &m.config,
            Self::Gemma4Moe(m) => &m.config,
            Self::Qwen35Moe(m) => &m.config,
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Llama(m)    => &m.tokenizer,
            Self::Qwen3(m)    => &m.tokenizer,
            Self::Gemma3(m)   => &m.tokenizer,
            Self::Gemma3n(m)  => &m.tokenizer,
            Self::Gemma4(m)   => &m.tokenizer,
            Self::Qwen35(m)   => &m.tokenizer,
            Self::Qwen3Moe(m) => &m.tokenizer,
            Self::Glm5Next(m) => &m.tokenizer,
            Self::Mixtral(m)  => &m.tokenizer,
            Self::Gemma4Moe(m) => &m.tokenizer,
            Self::Qwen35Moe(m) => &m.tokenizer,
        }
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        match self {
            Self::Llama(m)    => &m.backend,
            Self::Qwen3(m)    => &m.backend,
            Self::Gemma3(m)   => &m.backend,
            Self::Gemma3n(m)  => &m.backend,
            Self::Gemma4(m)   => &m.backend,
            Self::Qwen35(m)   => &m.backend,
            Self::Qwen3Moe(m) => &m.backend,
            Self::Glm5Next(m) => &m.backend,
            Self::Mixtral(m)  => &m.backend,
            Self::Gemma4Moe(m) => &m.backend,
            Self::Qwen35Moe(m) => &m.backend,
        }
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        match self {
            Self::Llama(m)    => m.forward(tokens, kv),
            Self::Qwen3(m)    => m.forward(tokens, kv),
            Self::Gemma3(m)   => m.forward(tokens, kv),
            Self::Gemma3n(m)  => m.forward(tokens, kv).expect("Gemma 3n forward not implemented"),
            Self::Gemma4(m)   => m.forward(tokens, kv),
            Self::Qwen35(m)   => m.forward(tokens, kv).expect("Qwen3.5 forward error"),
            Self::Qwen3Moe(m) => m.forward(tokens, kv),
            Self::Glm5Next(m) => m.forward(tokens, kv).unwrap_or_else(|e| panic!("{e}")),
            Self::Mixtral(m)  => m.forward(tokens, kv),
            Self::Gemma4Moe(m) => m.forward(tokens, kv),
            Self::Qwen35Moe(m) => m.forward(tokens, kv).expect("Qwen3.6-MoE forward error"),
        }
    }

    pub fn new_kv_cache(&self, max_len: usize) -> KvCache {
        let cfg = self.config();
        let len = max_len.min(cfg.context_length);
        // Recurrent Qwen blocks use ssm_state/ssm_conv, never attention K/V.
        // Keep tiny nonzero placeholders (CUDA does not require zero-byte allocs)
        // and allocate full K/V only for actual attention layers.
        if let Self::Qwen35(m) = self {
            let heads: Vec<usize> = m.attention_layers.iter().map(|&a| if a {cfg.n_kv_heads} else {1}).collect();
            let dims: Vec<usize> = m.attention_layers.iter().map(|&a| if a {cfg.head_dim} else {1}).collect();
            if !m.cache_backends.is_empty() {
                return KvCache::new_lazy_per_layer_kv(m.cache_backends.clone(), len, &heads, &dims);
            }
            return KvCache::new_per_layer_kv(self.backend().as_ref(), len, &heads, &dims);
        }
        // Gemma 4 MoE (26B-A4B): per-layer n_kv_heads varies (SWA=8, global=2)
        // AND per-layer head_dim varies. Read both from each layer's loaded
        // tensors so the cache matches the model exactly.
        if let Self::Gemma4Moe(m) = self {
            let head_dims: Vec<usize> = (0..cfg.n_layers).map(|i| cfg.layer_head_dim(i)).collect();
            let n_kv_per_layer: Vec<usize> = m.blocks.iter().enumerate()
                .map(|(i, b)| b.attn_k.shape()[0] / head_dims[i])
                .collect();
            return KvCache::new_per_layer_kv(self.backend().as_ref(), len, &n_kv_per_layer, &head_dims);
        }
        // Gemma 4 (dense) has different head_dim per layer (256 SWA / 512 global) so the
        // cache must be allocated per-layer (uniform n_kv_heads across layers).
        if cfg.head_dim_swa.is_some() {
            let head_dims: Vec<usize> = (0..cfg.n_layers).map(|i| cfg.layer_head_dim(i)).collect();
            KvCache::new_per_layer(self.backend().as_ref(), len, cfg.n_kv_heads, &head_dims)
        } else {
            KvCache::new(self.backend().as_ref(), cfg.n_layers, len, cfg.n_kv_heads, cfg.head_dim)
        }
    }

    /// Take last-position logits from a `[seq, vocab]` tensor and bring them
    /// onto the host (always CPU storage, ready for sampling).
    pub fn last_logits(&self, logits: &Tensor) -> Tensor {
        self.backend().last_row_to_host(logits)
    }

    // VENDORED-LOCAL: SAMPLE-01 — greedy token selection without the
    /// per-token vocab D2H. On CUDA this is one `argmax_last` kernel over
    /// the device logits plus a 4-byte copy; the CPU backend runs the same
    /// earliest-max scan host-side. Returns the argmax of the LAST row
    /// (decode: the only row; prefill: the next-token row).
    pub fn argmax_last_token(&self, logits: &Tensor) -> u32 {
        let ids = self.backend().argmax_last(logits);
        *ids.last().expect("argmax_last_token: empty logits")
    }

    // VENDORED-LOCAL: streaming-expert introspection (expert_stream.rs).
    /// Shared streaming state when the model was opened with
    /// [`Model::open_streaming`]; `None` for resident loads.
    pub fn stream_shared(&self) -> Option<&Arc<expert_stream::StreamShared>> {
        match self {
            Self::Qwen3Moe(m) => m.stream_shared.as_ref(),
            Self::Glm5Next(m) => m.stream_shared.as_ref(),
            Self::Mixtral(m)  => m.stream_shared.as_ref(),
            _ => None,
        }
    }

    /// Expert-cache counters (hits / misses / bytes read) for a streaming
    /// model; `None` for resident loads.
    pub fn expert_cache_stats(&self) -> Option<oaiy_engine::ecache::CacheStats> {
        self.stream_shared().map(|s| s.cache_stats())
    }

    /// VENDORED-LOCAL (CACHE-02): VRAM expert-cache counters for a streaming
    /// model with `--vram-cache` enabled; `None` otherwise (and in non-CUDA
    /// builds, where the whole device cache is compiled out).
    #[cfg(feature = "cuda")]
    pub fn device_cache_stats(&self) -> Option<expert_stream::device_cache::DeviceCacheStats> {
        self.stream_shared().and_then(|s| s.device_cache_stats())
    }

    /// The first streaming fetch/reconstruction failure, if any. The forward
    /// pass is infallible by signature, so a mid-generation I/O error is
    /// recorded here instead — check after each decode step and abort.
    pub fn expert_stream_error(&self) -> Option<String> {
        self.stream_shared().and_then(|s| s.error())
    }

    /// Format `messages` for this model's architecture and tokenize the result.
    /// `add_assistant=true` (the typical case) appends an open assistant turn so
    /// generation continues from there.
    ///
    /// The chat template emits the architecture's BOS marker as part of the string
    /// (e.g. `<bos>`, `<|begin_of_text|>`), so we pass `add_bos=false` to the
    /// tokenizer to avoid double-BOS.
    pub fn encode_chat(&self, messages: &[ChatMessage], add_assistant: bool) -> Result<Vec<u32>> {
        let prompt = apply_chat_template(&self.config().arch, messages, add_assistant);
        Ok(self.tokenizer().encode(&prompt, false)?)
    }

    /// One-shot raw-prompt completion. Tokenises `prompt` with BOS, runs
    /// `generate()` to exhaustion (or until `max_new` / EOS / stop tokens),
    /// returns the decoded response text. Convenience wrapper for batch /
    /// scripting use cases that don't need streaming. Uses default
    /// `SampleParams` (temperature 1.0, top-k 40, top-p 0.95).
    pub fn ask(&self, prompt: &str, max_new: usize) -> Result<String> {
        let prompt_ids = self.tokenizer().encode(prompt, true)?;
        let mut out = String::new();
        for tok in self.generate(&prompt_ids, SampleParams::default(), max_new) {
            out.push_str(&self.tokenizer().decode(&[tok]));
        }
        Ok(out)
    }

    /// One-shot chat completion. Renders `messages` through this arch's chat
    /// template, generates an assistant response, returns the decoded text
    /// (just the new assistant turn, with stop markers stripped). Convenience
    /// wrapper — use `generate()` directly when you need streaming or
    /// per-token control.
    pub fn ask_chat(&self, messages: &[ChatMessage], max_new: usize) -> Result<String> {
        let prompt_ids = self.encode_chat(messages, true)?;
        let mut out = String::new();
        for tok in self.generate(&prompt_ids, SampleParams::default(), max_new) {
            out.push_str(&self.tokenizer().decode(&[tok]));
        }
        Ok(out)
    }

    /// Strict JSON-value chat completion. Same as `ask_chat` but constrains
    /// generation through the JSON FSM (`with_json_value`), so the output is
    /// guaranteed to parse as a top-level JSON object or array of
    /// string/number/bool/null primitives. Auto-stops on the closing `}`/`]`.
    /// Pre-computing the per-state vocab masks runs once on the first call
    /// (~9M byte-checks at 256K vocab); subsequent calls amortise that.
    pub fn ask_json(&self, messages: &[ChatMessage], max_new: usize) -> Result<String> {
        let prompt_ids = self.encode_chat(messages, true)?;
        let iter = self.generate(&prompt_ids, SampleParams::default(), max_new)
            .with_json_value();
        let mut out = String::new();
        for tok in iter {
            out.push_str(&self.tokenizer().decode(&[tok]));
        }
        Ok(out)
    }

    /// Iterator over generated tokens. Prefills `prompt` immediately, then yields
    /// one token per `next()` call. Stops on the tokenizer's EOS, on any per-arch
    /// chat stop token (so this can be used directly with `encode_chat()`), or
    /// after `max_new` tokens.
    ///
    /// The sampler is created from `params` and seeded with the prompt for
    /// repetition-penalty bookkeeping.
    pub fn generate<'m>(
        &'m self,
        prompt: &[u32],
        params: SampleParams,
        max_new: usize,
    ) -> GenerateIter<'m, 'static> {
        let mut kv = self.new_kv_cache(2048);
        let device_greedy = device_greedy_params(&params);
        let (sampler, pending_logits, stop_tokens) =
            self.prefill_for_generate(prompt, params, &mut kv);
        GenerateIter {
            model: self,
            sampler,
            kv: KvHolder::Owned(kv),
            device_greedy,
            next_token: None,
            pending_logits: Some(pending_logits),
            stop_tokens,
            logit_bias: Vec::new(),
            token_history: Vec::new(),
            no_repeat_ngram_size: None,
            token_mask: None,
            min_new: 0,
            balance_stop: None,
            json_fsm: None,
            max_new,
            produced: 0,
            done: false,
        }
    }

    /// Streaming generation that uses an externally-supplied KV cache. Useful
    /// for multi-turn chat where prior turns' KV state should be reused on the
    /// next prefill — the caller keeps `kv` alive across turns and just calls
    /// `generate_with_kv` again with the new prompt suffix. Otherwise
    /// equivalent to [`Model::generate`].
    pub fn generate_with_kv<'m, 'k>(
        &'m self,
        prompt: &[u32],
        params: SampleParams,
        max_new: usize,
        kv: &'k mut KvCache,
    ) -> GenerateIter<'m, 'k> {
        let device_greedy = device_greedy_params(&params);
        let (sampler, pending_logits, stop_tokens) =
            self.prefill_for_generate(prompt, params, kv);
        GenerateIter {
            model: self,
            sampler,
            kv: KvHolder::Borrowed(kv),
            device_greedy,
            next_token: None,
            pending_logits: Some(pending_logits),
            stop_tokens,
            logit_bias: Vec::new(),
            token_history: Vec::new(),
            no_repeat_ngram_size: None,
            token_mask: None,
            min_new: 0,
            balance_stop: None,
            json_fsm: None,
            max_new,
            produced: 0,
            done: false,
        }
    }

    /// Internal helper: seed sampler with the prompt's tokens (for repetition
    /// penalty), run prefill, return the (sampler, pending logits, stop-tokens)
    /// triple. Borrow on `kv` ends with this call so the caller can move it
    /// into the iterator's `KvHolder::Owned` or keep the existing borrow.
    ///
    // VENDORED-LOCAL: SAMPLE-01 — `pending_logits` is the RAW forward output
    /// (`[seq, vocab]`, device-resident on CUDA), not the host last row:
    /// the first `next()` picks between on-device argmax (greedy) and
    /// `last_logits` + the CPU filter chain, so greedy generation never
    /// pays a vocab-sized D2H, not even at prefill.
    fn prefill_for_generate(
        &self,
        prompt: &[u32],
        params: SampleParams,
        kv: &mut KvCache,
    ) -> (Sampler, Tensor, Vec<u32>) {
        let mut sampler = Sampler::new(params);
        for &t in prompt { sampler.observe(t); }
        let logits = self.forward(prompt, kv);
        let stop_tokens: Vec<u32> = chat::chat_stop_tokens(&self.config().arch)
            .iter()
            .filter_map(|s| self.tokenizer().token_id(s))
            .collect();
        (sampler, logits, stop_tokens)
    }
}

/// KV cache holder used by [`GenerateIter`]. Either owns the cache (when
/// created via `Model::generate`) or borrows one supplied externally
/// (`Model::generate_with_kv`). The borrowed variant is what enables KV
/// reuse across chat turns — caller keeps the cache alive between calls and
/// the iterator just appends to it.
enum KvHolder<'k> {
    Owned(KvCache),
    Borrowed(&'k mut KvCache),
}

impl<'k> KvHolder<'k> {
    fn as_mut(&mut self) -> &mut KvCache {
        match self {
            KvHolder::Owned(k) => k,
            KvHolder::Borrowed(k) => k,
        }
    }
}

/// Streaming token iterator returned by [`Model::generate`]. Each call to
/// [`Iterator::next`] runs one decode step.
pub struct GenerateIter<'m, 'k> {
    model: &'m Model,
    sampler: Sampler,
    kv: KvHolder<'k>,
    /// VENDORED-LOCAL: SAMPLE-01 — true when the sampling params reduce to
    /// plain greedy (temperature ≤ 0, no effective repetition penalty), so
    /// token selection CAN run as an on-device argmax. Runtime filters
    /// (logit bias, token mask, JSON FSM, n-gram bans) are checked per step
    /// in `sample_step` — any of them pushes the step back to the CPU chain.
    device_greedy: bool,
    /// Next token to yield, or `None` for the very first call (where we
    /// sample lazily from `pending_logits` so that builder-attached filters
    /// affect the first sample). Subsequent `next()` calls pre-sample one
    /// token ahead so the consumer can decode `current` immediately.
    next_token: Option<u32>,
    /// Prefill logits, populated once by `Model::generate`. Drained on the
    /// first `next()` call to produce the first sampled token under the
    /// fully-configured filter chain. SAMPLE-01: the RAW `[seq, vocab]`
    /// forward output (see prefill_for_generate), not the host last row.
    pending_logits: Option<Tensor>,
    stop_tokens: Vec<u32>,
    /// Additive logit biases (token_id, delta). Empty by default. NEG_INFINITY
    /// effectively bans a token. See [`GenerateIter::with_logit_bias`].
    logit_bias: Vec<(u32, f32)>,
    /// All tokens yielded so far (prompt-relative). Tracked when
    /// `no_repeat_ngram_size` is set, so the n-gram filter can scan history.
    /// `Vec<u32>`; ~4 KB at 1k tokens — cheap.
    token_history: Vec<u32>,
    /// Anti-degeneration filter: forbid sampling any token that would close an
    /// n-gram already seen in `token_history`. None = disabled. Typical: 3.
    no_repeat_ngram_size: Option<usize>,
    /// Optional vocab-sized boolean mask. Pre-computed once via
    /// [`GenerateIter::with_token_filter`]; `mask[i] == false` means token `i`
    /// is forbidden (sampled-out by setting its logit to `-inf` per step).
    /// Compose with `logit_bias` for one-shot adjustments on top of broad masking.
    token_mask: Option<Vec<bool>>,
    /// Minimum number of tokens to emit before EOS / stop_tokens are honoured.
    /// 0 = honour from the very first sampled token.
    min_new: usize,
    /// Auto-stop when a balanced bracket pair has fully closed:
    /// (open_byte, close_byte, current_balance, has_seen_first_open).
    /// Counts opens and closes within decoded token text; stops when balance
    /// returns to 0 after at least one open. None = disabled.
    balance_stop: Option<(u8, u8, i32, bool)>,
    /// Optional JSON-object FSM. Holds a per-state vocab mask plus the live
    /// FSM cursor; if set, the iterator masks logits by the current state's
    /// admissibility set, samples, then advances the FSM by the new token's
    /// bytes. Stops automatically when the FSM reaches the Done state.
    json_fsm: Option<json_fsm::JsonObjectFsm>,
    max_new: usize,
    produced: usize,
    done: bool,
}

impl<'m, 'k> Iterator for GenerateIter<'m, 'k> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.done || self.produced >= self.max_new { return None; }

        // First call: drain prefill logits and sample the first token *now*
        // so that builder filters (with_json_object etc.) take effect.
        if self.next_token.is_none() {
            let logits = self.pending_logits.take()
                .expect("first `next()` but no pending prefill logits");
            self.next_token = Some(self.sample_step(logits));
        }

        let current = self.next_token.expect("next_token must be set after first-sample path");

        // Stop conditions checked on the about-to-be-yielded token. We don't
        // emit the EOS / turn-end marker — it's a control token, not user-visible
        // text — so we return None instead of Some(current) when we hit one.
        // `min_new` defers EOS / stop_tokens until at least N tokens have been
        // emitted (the on-EOS path is the only one this gates — we still want
        // to honour `with_token_filter` etc. if EOS is sampled in the first N).
        let min_satisfied = self.produced >= self.min_new;
        if min_satisfied {
            if let Some(eos) = self.model.tokenizer().eos() {
                if current == eos { self.done = true; return None; }
            }
            if self.stop_tokens.contains(&current) { self.done = true; return None; }
        }

        // Track for n-gram filter (cheap unconditionally; only consulted if
        // no_repeat_ngram_size is set).
        if self.no_repeat_ngram_size.is_some() {
            self.token_history.push(current);
        }

        // Update balanced-bracket counter from the about-to-be-yielded token.
        // Stop *after* yielding the token that closes the outermost pair.
        if let Some((open, close, ref mut bal, ref mut started)) = self.balance_stop {
            let txt = self.model.tokenizer().decode(&[current]);
            for b in txt.bytes() {
                if b == open { *bal += 1; *started = true; }
                else if b == close { *bal -= 1; }
            }
            if *started && *bal <= 0 { self.done = true; }
        }

        // Advance the JSON FSM by the about-to-be-yielded token's bytes. If the
        // FSM has reached its terminal state, also stop after this token.
        if let Some(fsm) = self.json_fsm.as_mut() {
            let txt = self.model.tokenizer().decode(&[current]);
            fsm.observe(&txt);
            if fsm.done() { self.done = true; }
        }

        // Sample the *next* token now, so the caller can decode `current`
        // immediately while we work on what comes after.
        let logits = self.model.forward(&[current], self.kv.as_mut());
        self.next_token = Some(self.sample_step(logits));
        self.produced += 1;
        Some(current)
    }
}

// VENDORED-LOCAL: SAMPLE-01 — does this parameter set reduce to plain
/// greedy? Temperature ≤ 0 makes the sampler argmax-only; a repetition
/// penalty other than 1.0 would still rewrite logits on the CPU path, so it
/// disqualifies the device fast path. (Mirostat needs temperature > 0 to do
/// anything meaningful, and top-k/top-p/min-p are dead code at temperature
/// ≤ 0, matching `Sampler::sample_biased`'s greedy short-circuit.)
fn device_greedy_params(params: &SampleParams) -> bool {
    params.temperature <= 0.0 && matches!(params.repeat_penalty, None | Some(1.0))
}

/// Find every token id that, if sampled next, would close a repeated n-gram.
/// Scans `history` for occurrences of the trailing `(n-1)`-gram and collects
/// the token that immediately followed each one.
///
/// Cost: O(history.len() · (n-1)) per call. For typical n=3 and 1k-token
/// histories that's ~2k comparisons — well below kernel-launch overhead, so
/// this stays out of the per-token critical path.
fn banned_ngram_continuations(history: &[u32], n: usize) -> Vec<u32> {    if n < 2 || history.len() < n { return Vec::new(); }
    let prefix_len = n - 1;
    let prefix_start = history.len() - prefix_len;
    let prefix = &history[prefix_start..];
    let mut out = Vec::new();
    // Search the history *up to but excluding* the prefix's own position, so we
    // don't ban the natural continuation from sampling its first occurrence.
    for i in 0..=prefix_start.saturating_sub(1) {
        if i + n > history.len() { break; }
        if &history[i..i + prefix_len] == prefix {
            let banned = history[i + prefix_len];
            if !out.contains(&banned) { out.push(banned); }
        }
    }
    out
}

impl<'m, 'k> GenerateIter<'m, 'k> {
    /// Number of tokens yielded so far.
    pub fn produced(&self) -> usize { self.produced }

    // VENDORED-LOCAL: SAMPLE-01 — one sampling step from raw `[seq, vocab]`
    /// logits. Greedy runs as an on-device argmax (no vocab D2H) when no
    /// runtime filter is active; anything that rewrites logits (bias, mask,
    /// JSON FSM, n-gram bans) falls back to the CPU filter chain, unchanged.
    /// The CPU greedy sampler and the device argmax both pick the earliest
    /// max index, so the two paths select identical tokens.
    fn sample_step(&mut self, logits: Tensor) -> u32 {
        if self.device_greedy
            && self.logit_bias.is_empty()
            && self.no_repeat_ngram_size.is_none()
            && self.token_mask.is_none()
            && self.json_fsm.is_none()
        {
            let tok = self.model.argmax_last_token(&logits);
            self.sampler.observe(tok);
            return tok;
        }
        let last = self.model.last_logits(&logits);
        self.sample_with_filters(last)
    }

    /// Apply all configured filters (token mask, JSON FSM mask, logit bias,
    /// n-gram bans) to `logits` in place, then sample one token. Used both for
    /// the deferred first-sample (from prefill logits) and the per-step
    /// pre-sample inside `next()`.
    fn sample_with_filters(&mut self, mut last: Tensor) -> u32 {
        // Broad token mask first (covers the largest set of tokens).
        if let Some(mask) = &self.token_mask {
            let data = last.data_mut();
            let n = mask.len().min(data.len());
            for i in 0..n {
                if !mask[i] { data[i] = f32::NEG_INFINITY; }
            }
        }

        // JSON FSM per-state mask: only admits tokens whose decoded bytes
        // legally consume from the current FSM state.
        if let Some(fsm) = self.json_fsm.as_ref() {
            let mask = fsm.current_mask();
            let data = last.data_mut();
            let n = mask.len().min(data.len());
            for i in 0..n {
                if !mask[i] { data[i] = f32::NEG_INFINITY; }
            }
        }

        // Per-step bias: static logit_bias + (optional) n-gram bans.
        let bias_for_step: Vec<(u32, f32)> =
            if let Some(n) = self.no_repeat_ngram_size {
                let banned = banned_ngram_continuations(&self.token_history, n);
                let mut combined = self.logit_bias.clone();
                combined.extend(banned.into_iter().map(|tok| (tok, f32::NEG_INFINITY)));
                combined
            } else {
                self.logit_bias.clone()
            };

        self.sampler.sample_biased(&last, &bias_for_step)
    }

    /// Apply additive logit biases at every decode step. `bias` is a list of
    /// `(token_id, delta)` pairs. Use `f32::NEG_INFINITY` to ban a token, a
    /// large positive value to force one. Compatible with the OpenAI Chat
    /// Completions `logit_bias` parameter. Replaces any prior call.
    pub fn with_logit_bias(mut self, bias: Vec<(u32, f32)>) -> Self {
        self.logit_bias = bias;
        self
    }

    /// Add additional stop tokens — once any of these is sampled, the iterator
    /// terminates without emitting it. Stacks on top of the per-arch chat stop
    /// tokens that `generate` registers automatically.
    pub fn stop_on(mut self, tokens: impl IntoIterator<Item = u32>) -> Self {
        self.stop_tokens.extend(tokens);
        self
    }

    /// Seed the sampler's repetition-penalty history with prior tokens (e.g.
    /// the cached prefix in a multi-turn chat). `generate*` only seeds with
    /// the current prompt; for KV-reuse use cases where the prior conversation
    /// is already in the cache, call this with the cached tokens so repetition
    /// penalty considers the full context. Bounded internally by
    /// `SampleParams::repeat_last_n`.
    pub fn with_history(mut self, tokens: &[u32]) -> Self {
        for &t in tokens { self.sampler.observe(t); }
        self
    }

    /// Anti-degeneration filter (HuggingFace `no_repeat_ngram_size`): forbid
    /// any token that would close an n-gram already seen in the generated
    /// output. `n=3` is the typical value (forbids any sequence of 3 tokens
    /// from repeating). Pass `None` (or don't call) to disable. Useful for
    /// greedy decoding to avoid degenerate "the the the" loops.
    pub fn no_repeat_ngram_size(mut self, n: usize) -> Self {
        self.no_repeat_ngram_size = if n >= 2 { Some(n) } else { None };
        self
    }

    /// Restrict sampling to tokens accepted by `allow`. Walks the full vocab
    /// once at builder time, decoding each token id and calling
    /// `allow(token_id, decoded_text)`; tokens that return `false` are masked
    /// out (logit set to `-inf`) at every subsequent decode step.
    ///
    /// Use for structured-output constraints — e.g. force JSON-like output by
    /// allowing only tokens whose decoded text consists of JSON-meaningful
    /// characters:
    ///
    /// ```ignore
    /// let json_chars = |b: u8| matches!(b,
    ///     b'{'|b'}'|b'['|b']'|b':'|b','|b'"'|b' '|b'\t'|b'\n'
    ///     | b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z'
    ///     | b'.'|b'-'|b'+'|b'_');
    /// let iter = model.generate(&prompt, params, 256)
    ///     .with_token_filter(|_id, text| text.bytes().all(json_chars));
    /// ```
    ///
    /// Composes with `with_logit_bias` and `no_repeat_ngram_size`: mask is
    /// applied first, then bias deltas, then n-gram bans. Calling this method
    /// twice replaces the prior mask.
    pub fn with_token_filter<F>(mut self, allow: F) -> Self
    where
        F: Fn(u32, &str) -> bool,
    {
        let tok = self.model.tokenizer();
        let n = tok.vocab_size();
        let mut mask = vec![false; n];
        for id in 0..n as u32 {
            let text = tok.decode(&[id]);
            mask[id as usize] = allow(id, &text);
        }
        self.token_mask = Some(mask);
        self
    }

    /// Forbid early termination: defer EOS / `stop_on` token recognition until
    /// at least `n` tokens have been emitted. Useful with constrained
    /// sampling where the model might otherwise emit EOS before producing any
    /// content. `with_token_filter` and other masks still apply from token 1.
    pub fn min_new_tokens(mut self, n: usize) -> Self {
        self.min_new = n;
        self
    }

    /// Auto-stop when a balanced bracket pair has fully closed. After at least
    /// one occurrence of `open`, the iterator terminates as soon as `close`
    /// occurrences match. Counts run over the decoded text of every yielded
    /// token, byte-by-byte. Typical usage: `stop_on_balanced(b'{', b'}')` to
    /// emit exactly one JSON object.
    ///
    /// Note: counts inside string literals too — for strict JSON you'd want a
    /// proper FSM. Good enough for "emit one balanced thing then stop"
    /// patterns where the model isn't pathologically embedding `}` chars.
    pub fn stop_on_balanced(mut self, open: u8, close: u8) -> Self {
        self.balance_stop = Some((open, close, 0, false));
        self
    }

    /// Strict JSON-value output. Uses per-state vocab masks so the sampled
    /// token sequence is *guaranteed* to parse as a single flat JSON value:
    /// either an object `{"key": value, ...}` or an array `[value, ...]`,
    /// where each value is a string, integer, `true`, `false`, or `null`.
    /// Stops automatically on the closing `}` / `]`.
    ///
    /// Pre-compute cost: O(n_states · vocab) decodes — ~9M byte-checks for a
    /// 256K vocab, runs in well under a second on a typical machine. Per-step
    /// cost is just a vocab-sized mask scan (already used by `with_token_filter`).
    ///
    /// See [`json_fsm::JsonObjectFsm`] for the supported subset and rationale.
    /// For broader JSON (nesting, escapes, floats) you'd want a PDA-style
    /// constraint with a state stack — the current scope keeps the state count
    /// finite (~35) so per-state masks each fit in a vocab-sized boolean vector.
    pub fn with_json_value(mut self) -> Self {
        self.json_fsm = Some(json_fsm::JsonObjectFsm::new(self.model.tokenizer()));
        self
    }

    /// Backwards-compat alias for [`with_json_value`]. Originally only objects
    /// were supported; the FSM now also accepts top-level arrays under the same
    /// builder.
    pub fn with_json_object(self) -> Self { self.with_json_value() }
}

#[cfg(test)]
mod tests {
    use super::banned_ngram_continuations;

    #[test]
    fn ngram_filter_finds_repeated_completions() {
        // History "A B C ... A B" — completing with C would repeat "A B C".
        let h = vec![1u32, 2, 3, 4, 5, 1, 2];
        let banned = banned_ngram_continuations(&h, 3);
        assert_eq!(banned, vec![3]);
    }

    #[test]
    fn ngram_filter_dedupes_multiple_matches() {
        // Two prior "A B" occurrences both followed by C — banned only listed once.
        let h = vec![1u32, 2, 3, 9, 1, 2, 3, 9, 1, 2];
        let banned = banned_ngram_continuations(&h, 3);
        assert_eq!(banned, vec![3]);
    }

    #[test]
    fn ngram_filter_collects_distinct_completions() {
        // "A B" once followed by C, once by D — both should be banned.
        let h = vec![1u32, 2, 3, 9, 1, 2, 4, 9, 1, 2];
        let banned = banned_ngram_continuations(&h, 3);
        assert_eq!(banned.len(), 2);
        assert!(banned.contains(&3));
        assert!(banned.contains(&4));
    }

    #[test]
    fn ngram_filter_short_history_is_empty() {
        let h = vec![1u32, 2];
        assert_eq!(banned_ngram_continuations(&h, 3), Vec::<u32>::new());
    }

    #[test]
    fn ngram_filter_n_lt_2_is_empty() {
        let h = vec![1u32, 2, 3, 1, 2];
        assert_eq!(banned_ngram_continuations(&h, 1), Vec::<u32>::new());
    }
}
