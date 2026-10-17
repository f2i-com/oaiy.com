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

use crate::llm::kv_cache::KvCache;
use crate::llm::loader::{CommonTensors, TensorIndex};
use crate::llm::{LlamaError, ModelConfig, Result};

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
    /// a decode step chained on the backend's device ([`crate::llm::chain::decode`]).
    chain: crate::llm::chain::decode::ChainDecoder,
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

    /// The model as the chain reads it: Qwen3's per-head q and k norms, and Qwen3-VL's M-RoPE sections.
    fn dense(&self) -> crate::llm::chain::decode::Dense<'_> {
        let norms = self.extras.iter().map(|e| (&e.attn_q_norm, &e.attn_k_norm)).collect();
        crate::llm::chain::decode::Dense { qk_norms: Some(norms), mrope: self.config.mrope_sections, ..crate::llm::chain::decode::Dense::plain(&self.config, &self.common) }
    }

    /// `tokens`' embeddings, on the host (`[tokens, d]`): a prompt's rows, a picture's then spliced in.
    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        let e = self.backend.embed_lookup(&self.common.tok_embd, tokens, self.config.embedding_dim);
        if e.is_device() { e.to_host() } else { e }
    }

    /// Rows given as their embeddings (`embeds`, `[tokens, d]` on the host: a picture's tokens spliced among the
    /// text's) after what `kv` holds, at `positions` (each row's time, height and width: Qwen3-VL's pictures over their
    /// grid, the text after one past its side; None, the rows that follow), with `deepstack[i]` (`[tokens, d]`, zero
    /// where no picture is) added after layer `i`: the last row's logits. Chained on the device (a step's single row
    /// as a step); without a chain, only text at the rows that follow.
    pub fn forward_embeds_positions(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, positions: Option<&[[u32; 3]]>, deepstack: &[Vec<f32>]) -> Result<Tensor> {
        let dense = self.dense();
        let seq = tokens.len();
        let chained = if seq == 1 && deepstack.is_empty() {
            self.chain.step_at(&*self.backend, &dense, tokens[0], kv, positions.and_then(|p| p.first().copied()))
        } else {
            self.chain.prompt_embeds(&*self.backend, &dense, embeds.data(), seq, kv, positions, deepstack)
        };
        if let Some(logits) = chained {
            return Ok(logits);
        }
        let at_rows = positions.is_none_or(|p| p.iter().enumerate().all(|(i, q)| q.iter().all(|&a| a as usize == kv.len + i)));
        if !deepstack.is_empty() || !at_rows {
            return Err(LlamaError::Config("a picture's rows need a GPU the model runs chained on".into()));
        }
        Ok(self.forward_host(tokens, kv))
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        // a decode step chained on the device when the backend has one for this model.
        let dense = self.dense();
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

        let mut x = backend.embed_lookup(&self.common.tok_embd, tokens, cfg.embedding_dim);
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..cfg.n_layers {
            let b = &self.common.blocks[layer];
            let qx = &self.extras[layer];

            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);
            let [q_flat, k_flat, v_flat]: [Tensor; 3] =
                crate::llm::loader::Weight::linear_many(backend, &xn, &[&b.attn_q, &b.attn_k, &b.attn_v]).try_into().expect("q, k and v");

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
