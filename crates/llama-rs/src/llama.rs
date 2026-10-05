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
    /// VENDORED-LOCAL: a decode step chained on the backend's device, made at the first step that can use one
    /// ([`LlamaModel::forward_chained`]); None when the backend has none or a weight is not on it.
    chain: std::sync::OnceLock<Option<ChainState>>,
}

/// VENDORED-LOCAL: what a chained decode step keeps on the device: the norms' weights and its activations.
struct ChainState {
    attn_norms: Vec<ggml_rs::DeviceVec>,
    ffn_norms: Vec<ggml_rs::DeviceVec>,
    output_norm: ggml_rs::DeviceVec,
    /// the residual stream, the normed input, q, k, v, the attention's output, a projection's output, the fused
    /// gate-up, the SwiGLU, the logits
    x: ggml_rs::DeviceVec,
    xn: ggml_rs::DeviceVec,
    q: ggml_rs::DeviceVec,
    k: ggml_rs::DeviceVec,
    v: ggml_rs::DeviceVec,
    attn: ggml_rs::DeviceVec,
    proj: ggml_rs::DeviceVec,
    gate_up: ggml_rs::DeviceVec,
    act: ggml_rs::DeviceVec,
    logits: ggml_rs::DeviceVec,
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
        Ok(Self { config, tokenizer, weights, backend, chain: std::sync::OnceLock::new() })
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

    /// VENDORED-LOCAL: the device state a chained decode step needs, if the backend chains and holds every weight a
    /// step reads (the projections, a fused gate-up, the head).
    fn chain_state(&self) -> Option<&ChainState> {
        self.chain
            .get_or_init(|| {
                let chain = self.backend.chain()?;
                let quant = |w: &crate::loader::Weight| match w {
                    crate::loader::Weight::Quant(q) if chain.holds(q) => Some(()),
                    _ => None,
                };
                for b in &self.weights.blocks {
                    for w in [&b.attn_q, &b.attn_k, &b.attn_v, &b.attn_output, &b.ffn_down] {
                        quant(w)?;
                    }
                    match &b.ffn_pair {
                        crate::loader::FfnPair::Fused(w) => quant(w)?,
                        crate::loader::FfnPair::Split { .. } => return None,
                    }
                }
                quant(&self.weights.output)?;
                let cfg = &self.config;
                let host = |t: &Tensor| if t.is_device() { t.to_host() } else { t.clone() };
                let upload = |t: &Tensor| {
                    let t = host(t);
                    let v = chain.vec(t.numel());
                    chain.upload(&v, t.data());
                    v
                };
                let ff = self.weights.blocks[0].ffn_pair.ff();
                let (d, qd, kvd) = (cfg.embedding_dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
                Some(ChainState {
                    attn_norms: self.weights.blocks.iter().map(|b| upload(&b.attn_norm)).collect(),
                    ffn_norms: self.weights.blocks.iter().map(|b| upload(&b.ffn_norm)).collect(),
                    output_norm: upload(&self.weights.output_norm),
                    x: chain.vec(d),
                    xn: chain.vec(d),
                    q: chain.vec(qd),
                    k: chain.vec(kvd),
                    v: chain.vec(kvd),
                    attn: chain.vec(qd),
                    proj: chain.vec(d),
                    gate_up: chain.vec(2 * ff),
                    act: chain.vec(ff),
                    logits: chain.vec(cfg.vocab_size),
                })
            })
            .as_ref()
    }

    /// VENDORED-LOCAL: a decode step with each layer's weights chained on the device, one submit a layer: the last
    /// layer's output projection, residual, FFN norm, gate-up, SwiGLU, down and residual, then this layer's norm and
    /// q, k and v, which alone come back for the host's RoPE, KV cache and attention. The step's 113 round trips on
    /// WebGPU (a 3B Llama) became 29. The same ops as [`Self::forward`]'s, their sums in the device's order.
    fn forward_chained(&self, token: u32, kv: &mut KvCache, st: &ChainState) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let chain = backend.chain().expect("a chain state comes from a chain");
        let (n_h, n_kv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let past = kv.len;
        let scale = 1.0 / (hd as f32).sqrt();
        let positions = [past as u32];
        let rope_type = cfg.arch.rope_type();
        fn quant(w: &crate::loader::Weight) -> &ggml_rs::QuantizedTensor {
            match w {
                crate::loader::Weight::Quant(q) => q,
                _ => unreachable!("chain_state checked every weight"),
            }
        }
        let emb = backend.embed_lookup(&self.weights.tok_embd, &[token], cfg.embedding_dim);
        let emb = if emb.is_device() { emb.to_host() } else { emb };
        chain.upload(&st.x, emb.data());
        // layer 0's norm and q, k, v
        let mut rec = chain.begin();
        let b0 = &self.weights.blocks[0];
        rec.rmsnorm(&st.x, &st.attn_norms[0], &st.xn, cfg.rms_eps);
        rec.matmul(quant(&b0.attn_q), &st.xn, &st.q);
        rec.matmul(quant(&b0.attn_k), &st.xn, &st.k);
        rec.matmul(quant(&b0.attn_v), &st.xn, &st.v);
        rec.read(&st.q);
        rec.read(&st.k);
        rec.read(&st.v);
        let mut got = rec.finish();
        for layer in 0..cfg.n_layers {
            let b = &self.weights.blocks[layer];
            let v_host = got.pop().expect("v");
            let k_host = got.pop().expect("k");
            let q_host = got.pop().expect("q");
            let mut q = Tensor::from_vec(q_host, vec![1, n_h, hd]);
            let mut k = Tensor::from_vec(k_host, vec![1, n_kv, hd]);
            let v = Tensor::from_vec(v_host, vec![1, n_kv, hd]);
            ops::rope(backend, &mut q, &positions, hd, rope_type, cfg.rope_theta);
            ops::rope(backend, &mut k, &positions, hd, rope_type, cfg.rope_theta);
            kv.append(backend, layer, &k, &v);
            let attn = ops::attention(backend, &q, kv.k_buffer(layer), kv.v_buffer(layer), kv.len + 1, scale, past);
            let attn = if attn.is_device() { attn.to_host() } else { attn };
            chain.upload(&st.attn, attn.data());
            let mut rec = chain.begin();
            // this layer's tail
            rec.matmul(quant(&b.attn_output), &st.attn, &st.proj);
            rec.add(&st.x, &st.proj);
            rec.rmsnorm(&st.x, &st.ffn_norms[layer], &st.xn, cfg.rms_eps);
            let crate::loader::FfnPair::Fused(gu) = &b.ffn_pair else { unreachable!("chain_state checked the pair") };
            rec.matmul(quant(gu), &st.xn, &st.gate_up);
            rec.silu_mul_split(&st.gate_up, &st.act);
            rec.matmul(quant(&b.ffn_down), &st.act, &st.proj);
            rec.add(&st.x, &st.proj);
            if layer + 1 < cfg.n_layers {
                // the next layer's head
                let nb = &self.weights.blocks[layer + 1];
                rec.rmsnorm(&st.x, &st.attn_norms[layer + 1], &st.xn, cfg.rms_eps);
                rec.matmul(quant(&nb.attn_q), &st.xn, &st.q);
                rec.matmul(quant(&nb.attn_k), &st.xn, &st.k);
                rec.matmul(quant(&nb.attn_v), &st.xn, &st.v);
                rec.read(&st.q);
                rec.read(&st.k);
                rec.read(&st.v);
            } else {
                rec.rmsnorm(&st.x, &st.output_norm, &st.xn, cfg.rms_eps);
                rec.matmul(quant(&self.weights.output), &st.xn, &st.logits);
                rec.read(&st.logits);
            }
            got = rec.finish();
        }
        kv.commit(1);
        Tensor::from_vec(got.pop().expect("the logits"), vec![1, cfg.vocab_size])
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        // VENDORED-LOCAL: a decode step chained on the device when the backend has one for this model.
        if tokens.len() == 1 && std::env::var_os("OAIY_NO_CHAIN").is_none() {
            if let Some(st) = self.chain_state() {
                return self.forward_chained(tokens[0], kv, st);
            }
        }
        self.forward_host(tokens, kv)
    }

    /// VENDORED-LOCAL: [`Self::forward`] op by op through the backend, never chained (what a chained step is
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
