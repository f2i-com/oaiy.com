//! Mixtral 8x7B forward pass — Llama attention block + Mixture-of-Experts FFN.
//!
//! Mixtral uses `general.architecture = "llama"` with extra MoE metadata
//! (`llama.expert_count = 8`, `llama.expert_used_count = 2`) and per-expert
//! FFN tensors (`blk.N.ffn_{gate,up,down}.M.weight`, M = 0..7). Attention is
//! standard Llama (no Q/K-norm — that's a Qwen3 thing).

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{load_lm_head_or_tied, upload_tok_embd_and_lm_head, TensorIndex, Weight};
use crate::moe::{moe_forward, MoeFfn};
use crate::{LlamaError, ModelConfig, Result};

#[derive(Debug)]
pub struct MixtralBlock {
    pub attn_norm:   Tensor,
    pub attn_q:      Weight,
    pub attn_k:      Weight,
    pub attn_v:      Weight,
    pub attn_output: Weight,
    pub ffn_norm:    Tensor,
    pub moe:         MoeFfn,
}

pub struct MixtralModel {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub blocks:    Vec<MixtralBlock>,
    pub tok_embd:  Arc<Tensor>,
    pub output_norm: Tensor,
    pub output:    Weight,
    pub n_experts:      usize,
    pub n_experts_used: usize,
    pub backend:   Arc<dyn Backend>,
    /// VENDORED-LOCAL: shared streaming-expert state when opened with
    /// `from_gguf_streaming`; `None` for the resident load.
    pub stream_shared: Option<Arc<crate::expert_stream::StreamShared>>,
}

impl std::fmt::Debug for MixtralModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixtralModel")
            .field("config", &self.config)
            .field("n_experts", &self.n_experts)
            .field("n_experts_used", &self.n_experts_used)
            .field("backend", &self.backend.name())
            .finish()
    }
}

impl MixtralModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::from_gguf_impl(g, backend, None)
    }

    /// VENDORED-LOCAL: streaming variant — experts stay in the store behind
    /// the bounded cache; everything else loads resident as in `from_gguf`.
    pub fn from_gguf_streaming(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        shared: Arc<crate::expert_stream::StreamShared>,
    ) -> Result<Self> {
        Self::from_gguf_impl(g, backend, Some(shared))
    }

    fn from_gguf_impl(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        stream_shared: Option<Arc<crate::expert_stream::StreamShared>>,
    ) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        let arch_name = g.architecture()?.to_string();
        let n_experts = g.get_u64(&format!("{arch_name}.expert_count"))
            .map(|v| v as usize)
            .map_err(|_| LlamaError::Config(format!("missing {arch_name}.expert_count")))?;
        let n_experts_used = g.get_u64(&format!("{arch_name}.expert_used_count"))
            .map(|v| v as usize)
            .map_err(|_| LlamaError::Config(format!("missing {arch_name}.expert_used_count")))?;

        let tok_embd    = Arc::new(idx.take("token_embd.weight",  &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        const M: usize = 2 * 1024 * 1024 * 1024;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let attn_norm   = idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?;
            let attn_q      = idx.take_weight(&format!("blk.{i}.attn_q.weight"), &[])?;
            let attn_k      = idx.take_weight(&format!("blk.{i}.attn_k.weight"), &[])?;
            let attn_v      = idx.take_weight(&format!("blk.{i}.attn_v.weight"), &[])?;
            let attn_output = idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?;
            let ffn_norm    = idx.take(&format!("blk.{i}.ffn_norm.weight"), &[])?;

            let router = idx.take_weight(&format!("blk.{i}.ffn_gate_inp.weight"), &[])?;
            // VENDORED-LOCAL: streaming mode skips materializing the experts.
            let moe = match &stream_shared {
                Some(shared) => MoeFfn {
                    router,
                    gate_up_experts: Vec::new(),
                    down_experts: Vec::new(),
                    top_k: n_experts_used,
                    stream: Some(shared.layer(i as u32)),
                }
                .move_to_device(&*backend, M),
                None => {
                    let (gate_up_experts, down_experts) =
                        crate::qwen3moe::load_per_layer_experts_compat(&idx, i, n_experts)?;
                    MoeFfn { router, gate_up_experts, down_experts, top_k: n_experts_used, stream: None,
                    }
                        .move_to_device(&*backend, M)
                }
            };

            blocks.push(MixtralBlock {
                attn_norm:   backend.to_device(attn_norm),
                attn_q:      attn_q.try_to_device(&*backend, M),
                attn_k:      attn_k.try_to_device(&*backend, M),
                attn_v:      attn_v.try_to_device(&*backend, M),
                attn_output: attn_output.try_to_device(&*backend, M),
                ffn_norm:    backend.to_device(ffn_norm),
                moe,
            });
        }

        let (tok_embd, output) = upload_tok_embd_and_lm_head(&*backend, tok_embd, output);
        let output = output.try_to_device(&*backend, M);
        Ok(Self {
            config, tokenizer, blocks,
            tok_embd,
            output_norm: backend.to_device(output_norm),
            output,
            n_experts,
            n_experts_used,
            backend,
            stream_shared,
        })
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let seq = tokens.len();
        let past = kv.len;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut x = backend.embed_lookup(&self.tok_embd, tokens, cfg.embedding_dim);
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..cfg.n_layers {
            let b = &self.blocks[layer];
            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);
            let q_flat = b.attn_q.linear(backend, &xn);
            let k_flat = b.attn_k.linear(backend, &xn);
            let v_flat = b.attn_v.linear(backend, &xn);

            let mut q = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let mut k = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
            let v = v_flat.reshape(vec![seq, n_kv, hd]).expect("v reshape");

            let rope_type = cfg.arch.rope_type();
            ops::rope(backend, &mut q, &positions, hd, rope_type, cfg.rope_theta);
            ops::rope(backend, &mut k, &positions, hd, rope_type, cfg.rope_theta);

            kv.append(backend, layer, &k, &v);
            let kv_len = kv.len + seq;
            let attn_out = ops::attention(
                backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                kv_len, scale, past,
            );
            let attn_t = attn_out.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let attn_proj = b.attn_output.linear(backend, &attn_t);
            let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &b.ffn_norm, cfg.rms_eps);
            let ffn_out = moe_forward(backend, &xn2, &b.moe);
            ops::add_inplace(backend, &mut x, &ffn_out);
        }

        kv.commit(seq);
        let x = ops::rmsnorm(backend, &x, &self.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        self.output.linear(backend, &x_last)
    }
}
