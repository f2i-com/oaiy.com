//! Qwen-3 dense forward pass.
//!
//! Qwen-3 = Llama / Qwen-2 + per-layer RMSNorm on Q and K (after projection,
//! before RoPE). Norm weights stored as `blk.{i}.attn_q_norm.weight` /
//! `blk.{i}.attn_k_norm.weight`, both `[head_dim]`.
//!
//! Reference: `class Qwen3Model(Qwen2Model)` in `llama.cpp/convert_hf_to_gguf.py`.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{CommonTensors, TensorIndex};
use crate::{LlamaError, ModelConfig, Result};

#[derive(Debug)]
pub struct Qwen3BlockExtras {
    pub attn_q_norm: Tensor,
    pub attn_k_norm: Tensor,
}

pub struct Qwen3Model {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub common:    CommonTensors,
    pub extras:    Vec<Qwen3BlockExtras>,
    pub backend:   Arc<dyn Backend>,
    /// VENDORED-LOCAL: a decode step chained on the backend's device ([`crate::chain_decode`]).
    chain: crate::chain_decode::ChainDecoder,
}

impl std::fmt::Debug for Qwen3Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Model")
            .field("config", &self.config)
            .field("backend", &self.backend.name())
            .field("vocab_size", &self.tokenizer.vocab_size())
            .finish()
    }
}

impl Qwen3Model {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);
        let common = CommonTensors::load_with_index(&idx, &config)?.upload_to(&*backend);

        let mut extras = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let q_norm = idx.take(&format!("blk.{i}.attn_q_norm.weight"), &[])?;
            let k_norm = idx.take(&format!("blk.{i}.attn_k_norm.weight"), &[])?;
            if q_norm.shape() != [config.head_dim] {
                return Err(LlamaError::BadTensorShape {
                    name: format!("blk.{i}.attn_q_norm"),
                    got: q_norm.shape().iter().map(|&v| v as u64).collect(),
                    expected: vec![config.head_dim as u64],
                });
            }
            if k_norm.shape() != [config.head_dim] {
                return Err(LlamaError::BadTensorShape {
                    name: format!("blk.{i}.attn_k_norm"),
                    got: k_norm.shape().iter().map(|&v| v as u64).collect(),
                    expected: vec![config.head_dim as u64],
                });
            }
            extras.push(Qwen3BlockExtras {
                attn_q_norm: backend.to_device(q_norm),
                attn_k_norm: backend.to_device(k_norm),
            });
        }

        Ok(Self { config, tokenizer, common, extras, backend, chain: Default::default() })
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        // VENDORED-LOCAL: a decode step chained on the device when the backend has one for this model.
        if tokens.len() == 1 {
            let norms = self.extras.iter().map(|e| (&e.attn_q_norm, &e.attn_k_norm)).collect();
            let dense = crate::chain_decode::Dense { cfg: &self.config, common: &self.common, qk_norms: Some(norms) };
            if let Some(logits) = self.chain.step(&*self.backend, &dense, tokens[0], kv) {
                return logits;
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

        let mut x = backend.embed_lookup(&self.common.tok_embd, tokens, cfg.embedding_dim);
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..cfg.n_layers {
            let b = &self.common.blocks[layer];
            let qx = &self.extras[layer];

            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);
            let [q_flat, k_flat, v_flat]: [Tensor; 3] =
                crate::loader::Weight::linear_many(backend, &xn, &[&b.attn_q, &b.attn_k, &b.attn_v]).try_into().expect("q, k and v");

            let q_3d = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let k_3d = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");

            // Qwen-3: per-head Q/K norm before RoPE.
            let mut q = ops::rmsnorm(backend, &q_3d, &qx.attn_q_norm, cfg.rms_eps);
            let mut k = ops::rmsnorm(backend, &k_3d, &qx.attn_k_norm, cfg.rms_eps);
            let v = v_flat.reshape(vec![seq, n_kv, hd]).expect("v reshape");

            let rope_type = cfg.arch.rope_type();
            ops::rope(backend, &mut q, &positions, hd, rope_type, cfg.rope_theta);
            ops::rope(backend, &mut k, &positions, hd, rope_type, cfg.rope_theta);

            kv.append(backend, layer, &k, &v);
            let _ = n_rep;
            let kv_len = kv.len + seq;
            let attn_out = ops::attention(
                backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                kv_len, scale, past,
            );

            let attn_t = attn_out.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let attn_proj = b.attn_output.linear(backend, &attn_t);
            let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &b.ffn_norm, cfg.rms_eps);
            let activated = b.ffn_pair.swiglu(backend, &xn2);
            let ffn_out = b.ffn_down.linear(backend, &activated);
            ops::add_inplace(backend, &mut x, &ffn_out);
        }

        kv.commit(seq);

        let x = ops::rmsnorm(backend, &x, &self.common.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        self.common.output.linear(backend, &x_last)
    }
}
