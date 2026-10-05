//! Llama-family forward pass.
//!
//! Architecture (per layer, with shapes):
//!
//! ```text
//!   x[seq, d]
//!   ├── attn_norm:RMSNorm
//!   ├── q,k,v = Wq·xn, Wk·xn, Wv·xn       -> [seq, n_h*hd] / [seq, n_kv*hd]
//!   ├── reshape, RoPE on q & k
//!   ├── append (k,v) to layer's KV cache
//!   ├── repeat (k,v) along head axis to match q                 (GQA)
//!   ├── scores = bmm_qkt(q, k) ; softmax along kv               -> [seq, n_h, T]
//!   ├── out = bmm_av(scores, v) ; reshape to [seq, d]
//!   ├── x += Wo·out
//!   ├── ffn_norm:RMSNorm
//!   ├── x += W_down·( silu(W_gate·xn2) * W_up·xn2 )             (SwiGLU)
//!   ↓
//!   x[seq, d]
//!   final RMSNorm, then LM head -> logits[seq, vocab]
//! ```
//!
//! Runs entirely on the model's [`Backend`] — for the CUDA backend, weights and
//! activations live on the GPU and we never touch host memory inside the loop.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::ModelTensors;
use crate::{LlamaConfig, Result};

pub struct LlamaModel {
    pub config:    LlamaConfig,
    pub tokenizer: Tokenizer,
    pub weights:   ModelTensors,
    pub backend:   Arc<dyn Backend>,
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("config", &self.config)
            .field("backend", &self.backend.name())
            .field("vocab_size", &self.tokenizer.vocab_size())
            .finish()
    }
}

impl LlamaModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        // MoE detection: Mixtral / Qwen3-MoE / Gemma 4 26B-A4B all set
        // `<arch>.expert_count > 0`. The dense Llama loader doesn't know how to
        // load per-expert ffn weights, so we bail with a clear pointer to #85
        // instead of crashing later with a confusing MissingTensor.
        let arch_name = g.architecture()?.to_string();
        let expert_count_key = format!("{arch_name}.expert_count");
        if let Ok(n_experts) = g.get_u64(&expert_count_key) {
            if n_experts > 0 {
                let used = g.get_u64(&format!("{arch_name}.expert_used_count")).unwrap_or(0);
                return Err(crate::LlamaError::Config(format!(
                    "MoE model detected: {n_experts} experts, top-{used} routing per token. \
                     Loader integration with the new MoeFfn infra is pending (task #85). \
                     The MoE forward composition itself works (see `llama_rs::moe`); the \
                     remaining piece is wiring the per-expert weight loader and replacing \
                     the dense FFN dispatch in this model's forward."
                )));
            }
        }
        let config = LlamaConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let weights = ModelTensors::load(g, &config)?.upload_to(&*backend);
        Ok(Self { config, tokenizer, weights, backend })
    }

    pub fn new_kv_cache(&self, max_len: usize) -> KvCache {
        KvCache::new(
            &*self.backend,
            self.config.n_layers,
            max_len.min(self.config.context_length),
            self.config.n_kv_heads,
            self.config.head_dim,
        )
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let seq = tokens.len();
        let past = kv.len;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let n_rep = cfg.n_rep();
        let scale = 1.0 / (hd as f32).sqrt();

        let mut x = backend.embed_lookup(&self.weights.tok_embd, tokens, cfg.embedding_dim);
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..cfg.n_layers {
            let b = &self.weights.blocks[layer];

            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);
            let [q_flat, k_flat, v_flat]: [Tensor; 3] =
                crate::loader::Weight::linear_many(backend, &xn, &[&b.attn_q, &b.attn_k, &b.attn_v]).try_into().expect("q, k and v");

            let mut q = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let mut k = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
            let v = v_flat.reshape(vec![seq, n_kv, hd]).expect("v reshape");

            let rope_type = cfg.arch.rope_type();
            ops::rope(backend, &mut q, &positions, hd, rope_type, cfg.rope_theta);
            ops::rope(backend, &mut k, &positions, hd, rope_type, cfg.rope_theta);

            kv.append(backend, layer, &k, &v);
            let _ = n_rep; // GQA expansion is now done inside attention
            let kv_len = kv.len + seq;
            let attn_out = ops::attention(
                backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                kv_len, scale, past,
            );

            let attn_t = attn_out
                .reshape(vec![seq, n_h * hd])
                .expect("attn reshape");
            let attn_proj = b.attn_output.linear(backend, &attn_t);
            let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &b.ffn_norm, cfg.rms_eps);
            let activated = b.ffn_pair.swiglu(backend, &xn2);
            let ffn_out = b.ffn_down.linear(backend, &activated);
            ops::add_inplace(backend, &mut x, &ffn_out);
        }

        kv.commit(seq);

        let x = ops::rmsnorm(backend, &x, &self.weights.output_norm, cfg.rms_eps);
        // LM head sees only the last token's residual: callers always sample
        // from the last row (`Model::last_logits`), so for prefill (seq>1) we
        // slice down before the vocab-sized matmul.
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        self.weights.output.linear(backend, &x_last)
    }

    /// Take last-row logits as a host tensor (always on CPU). Use this before
    /// passing logits to a sampler.
    pub fn last_logits(&self, logits: &Tensor) -> Tensor {
        self.backend.last_row_to_host(logits)
    }
}
