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

use crate::llm::kv_cache::KvCache;
use crate::llm::loader::ModelTensors;
use crate::llm::{LlamaConfig, Result};

pub struct LlamaModel {
    pub config:    LlamaConfig,
    pub tokenizer: Tokenizer,
    pub weights:   ModelTensors,
    pub backend:   Arc<dyn Backend>,
    /// a decode step chained on the backend's device ([`crate::llm::chain::decode`]).
    chain: crate::llm::chain::decode::ChainDecoder,
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
                return Err(crate::llm::LlamaError::Config(format!(
                    "MoE model detected: {n_experts} experts, top-{used} routing per token. \
                     Loader integration with the new MoeFfn infra is pending (task #85). \
                     The MoE forward composition itself works (see `crate::llm::moe`); the \
                     remaining piece is wiring the per-expert weight loader and replacing \
                     the dense FFN dispatch in this model's forward."
                )));
            }
        }
        let config = LlamaConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let weights = ModelTensors::load(g, &config)?.upload_to(&*backend);
        Ok(Self { config, tokenizer, weights, backend, chain: Default::default() })
    }

    /// Its layers from `from` on and its head moved to `other` (another card of its backend's kind), whose chain then
    /// runs them after this one's ([`crate::llm::chain::decode::ChainDecoder::split`]): each weight copied there and let
    /// go of here, so each card holds its share of the weights and of the cache. Before the first step; false, and
    /// nothing moved, where a weight is not one a chain holds, `other` has no room, or chains are switched off.
    pub fn split_onto(&mut self, other: Arc<dyn Backend>, from: usize) -> bool {
        use crate::llm::loader::{FfnPair, Weight};
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || from == 0 || from >= self.weights.blocks.len() {
            return false;
        }
        let Some(chain) = other.chain() else { return false };
        let copy = |w: &Weight| match w {
            Weight::Quant(q) => chain.copy_weight(q).map(Weight::Quant),
            _ => None,
        };
        // every weight copied first, so a refusal leaves the model as it was
        let mut moved = Vec::with_capacity(self.weights.blocks.len() - from);
        for b in &self.weights.blocks[from..] {
            let FfnPair::Fused(gu) = &b.ffn_pair else { return false };
            let Some(w) = [&b.attn_q, &b.attn_k, &b.attn_v, &b.attn_output, gu, &b.ffn_down].into_iter().map(copy).collect::<Option<Vec<_>>>() else { return false };
            moved.push(w);
        }
        let Some(output) = copy(&self.weights.output) else { return false };
        for (b, w) in self.weights.blocks[from..].iter_mut().zip(moved) {
            let [q, k, v, o, gu, down]: [Weight; 6] = w.try_into().unwrap_or_else(|_| unreachable!("six weights a block"));
            (b.attn_q, b.attn_k, b.attn_v, b.attn_output, b.ffn_pair, b.ffn_down) = (q, k, v, o, FfnPair::Fused(gu), down);
        }
        self.weights.output = output;
        self.chain.split(other, from)
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
        // a decode step chained on the device when the backend has one for this model.
        let dense = crate::llm::chain::decode::Dense::plain(&self.config, &self.weights);
        let chained = if tokens.len() == 1 {
            self.chain.step(&*self.backend, &dense, tokens[0], kv)
        } else {
            self.chain.prompt(&*self.backend, &dense, tokens, kv)
        };
        if let Some(logits) = chained {
            return logits;
        }
        self.forward_host(tokens, kv)
    }

    /// [`Self::forward`] op by op through the backend, never chained (what a chained step is
    /// checked against).
    pub fn forward_host(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
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
                crate::llm::loader::Weight::linear_many(backend, &xn, &[&b.attn_q, &b.attn_k, &b.attn_v]).try_into().expect("q, k and v");

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

#[cfg(all(test, feature = "webgpu"))]
mod tests {
    use super::*;

    /// A Llama split over two cards ([`LlamaModel::split_onto`]) gives one card's logits to the bit, for a prompt's
    /// chunk and the steps after it: each card runs the same kernels, and the hidden state goes between them whole.
    /// LLAMA_GGUF: a small Llama GGUF; two GPUs.
    #[test]
    #[ignore]
    fn a_llama_split_over_two_cards_gives_one_cards_logits() {
        let Some(path) = std::env::var_os("LLAMA_GGUF") else { return };
        let Ok(a) = ggml_rs_wgpu::WgpuBackend::new(Some(8 << 30)) else { return };
        let Some(b) = a.others(Some(8 << 30)).into_iter().next() else { return };
        let g = GgufFile::open(std::path::Path::new(&path)).unwrap();
        let one = LlamaModel::from_gguf(&g, Arc::new(a.clone())).unwrap();
        let mut two = LlamaModel::from_gguf(&g, Arc::new(a)).unwrap();
        let from = two.config.n_layers / 2;
        assert!(two.split_onto(Arc::new(b), from), "the later layers moved to the second card");
        let prompt = one.tokenizer.encode("The quick brown fox jumps over the lazy dog, and then it runs into the forest where", true).unwrap();
        let (mut kv1, mut kv2) = (one.new_kv_cache(256), two.new_kv_cache(256));
        let (mut l1, mut l2) = (one.last_logits(&one.forward(&prompt, &mut kv1)), two.last_logits(&two.forward(&prompt, &mut kv2)));
        for step in 0..8 {
            assert_eq!(l1.data(), l2.data(), "the logits after {} tokens", prompt.len() + step);
            let next = l1.data().iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i as u32).unwrap();
            l1 = one.last_logits(&one.forward(&[next], &mut kv1));
            l2 = two.last_logits(&two.forward(&[next], &mut kv2));
        }
        assert_eq!(l1.data(), l2.data());
    }
}
