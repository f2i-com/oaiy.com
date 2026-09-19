//! Qwen 3.6 MoE — `arch=qwen35moe` + 256-expert MoE + shared expert per layer.
//!
//! Combines three previously separate ports:
//!   * **Qwen 3.5/3.6 SSM hybrid backbone** — every `full_attention_interval`-th
//!     layer is full attention with partial-RoPE 0.25 + MRoPE; the rest are
//!     gated-delta-net (Mamba2-style) layers (per-token CUDA delta-net + conv1d).
//!   * **MoE FFN** — 256 experts, top-8 routing, stacked `ffn_*_exps.weight`
//!     layout sliced into per-expert weights at load (Q4_K byte-range split).
//!     Uses the shared `MoeFfn` infra in `llama_rs::moe`.
//!   * **Shared expert** (DeepSeek-V2 / Qwen3-Next pattern) — a small dense
//!     SwiGLU FFN that runs in parallel with the routed experts; a per-token
//!     sigmoid gate from `ffn_gate_inp_shexp` `[hidden]` controls how much of
//!     the shared expert mixes in. Final FFN output: `moe_out + sigmoid(cur @
//!     gate_inp_shexp) * shexp(cur)`.
//!
//! Target: Qwen3.6-35B-A3B (40 layers, 32 SSM + 8 attention, 256 experts top-8,
//! shared-expert ff=512). On disk Q4_K_M = 20.6 GB.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{load_lm_head_or_tied, upload_tok_embd_and_lm_head, FfnPair, TensorIndex, Weight};
use crate::moe::{moe_forward_with_logits, MoeFfn, MoeOptions};
use crate::qwen3moe::split_stacked_experts;
use crate::qwen35::{fuse_alpha_beta, SsmConfig};
use crate::{LlamaError, ModelConfig, Result};

/// Per-layer MoE+shared-expert FFN bundle. Lives alongside the (SSM- or
/// attention-) attention block tensors.
#[derive(Debug)]
pub struct Qwen35MoeFfn {
    pub post_norm:        Tensor,    // pre-FFN RMSNorm (called post_attention_norm in GGUF)
    /// Routed MoE: router + 256 experts (gate / up / down each).
    pub moe:              MoeFfn,
    /// Dense shared expert (SwiGLU). One per layer, mixes with MoE output.
    /// `shexp_gate` and `shexp_up` auto-fuse into one matmul at load time.
    pub shexp_pair:       FfnPair,
    pub shexp_down:       Weight,
    /// Per-channel `[hidden]` weight; produces a per-token sigmoid gate
    /// scaling the shared expert's contribution.
    pub shexp_router:     Tensor,
}

#[derive(Debug)]
pub enum Qwen35MoeBlock {
    Ssm {
        attn_norm:    Tensor,
        attn_qkv:     Weight,
        attn_gate:    Weight,
        ssm_conv1d:   Tensor,
        ssm_a:        Tensor,
        ssm_ba:       Weight,
        ssm_dt_bias:  Tensor,
        ssm_norm:     Tensor,
        ssm_out:      Weight,
        ffn:          Qwen35MoeFfn,
    },
    Attention {
        attn_norm:    Tensor,
        attn_q:       Weight,
        attn_q_norm:  Tensor,
        attn_k:       Weight,
        attn_k_norm:  Tensor,
        attn_v:       Weight,
        attn_output:  Weight,
        ffn:          Qwen35MoeFfn,
    },
}

#[derive(Debug)]
pub struct Qwen35MoeModel {
    pub config:    ModelConfig,
    pub ssm_cfg:   SsmConfig,
    pub attention_layers: Vec<bool>,
    pub tokenizer: Tokenizer,
    pub blocks:    Vec<Qwen35MoeBlock>,
    pub tok_embd:  Arc<Tensor>,
    pub output_norm: Tensor,
    pub output:    Weight,
    pub backend:   Arc<dyn Backend>,
    pub n_experts: usize,
    pub expert_used: usize,
}

impl Qwen35MoeModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_experts   = g.get_u64("qwen35moe.expert_count")? as usize;
        let expert_used = g.get_u64("qwen35moe.expert_used_count")? as usize;
        if n_experts == 0 {
            return Err(LlamaError::Config(
                "Qwen35MoeModel called with expert_count=0".into(),
            ));
        }

        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        // ----- SSM hyperparameters (same shape as qwen35.rs's SsmConfig) -----
        let get = |suffix: &str| -> Result<u64> {
            Ok(g.get_u64(&format!("qwen35moe.{suffix}"))?)
        };
        let ssm_cfg = SsmConfig {
            conv_kernel:    get("ssm.conv_kernel")?     as usize,
            group_count:    get("ssm.group_count")?     as usize,
            inner_size:     get("ssm.inner_size")?      as usize,
            state_size:     get("ssm.state_size")?      as usize,
            time_step_rank: get("ssm.time_step_rank")?  as usize,
        };

        let interval = g.get_u64("qwen35moe.full_attention_interval")
            .map(|v| v as usize).unwrap_or(4);
        let attention_layers: Vec<bool> = (0..config.n_layers)
            .map(|i| i % interval == interval - 1)
            .collect();

        // ----- Common embeddings + LM head -----
        let tok_embd    = Arc::new(idx.take("token_embd.weight",  &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        const SAFETY_BYTES: usize = 2 * 1024 * 1024 * 1024;
        let upload_dense = |t: Tensor| backend.to_device(t);
        let upload_w = |w: Weight| w.try_to_device(&*backend, SAFETY_BYTES);

        // Tied LM head shares Arc with tok_embd → upload once.
        let (tok_embd_d, output) = upload_tok_embd_and_lm_head(&*backend, tok_embd, output);
        let output_norm_d = upload_dense(output_norm);
        let output_d     = upload_w(output);

        // ----- Per-layer block tensors -----
        let mut blocks = Vec::with_capacity(config.n_layers);
        for (i, &is_attn) in attention_layers.iter().enumerate() {
            // ----- shared FFN (MoE + shared expert) — present on EVERY layer -----
            let post_norm = upload_dense(idx.take(&format!("blk.{i}.post_attention_norm.weight"), &[])?);

            let router = upload_w(idx.take_weight(&format!("blk.{i}.ffn_gate_inp.weight"), &[])?);
            let gate_stacked = idx.take_weight(&format!("blk.{i}.ffn_gate_exps.weight"), &[])?;
            let up_stacked   = idx.take_weight(&format!("blk.{i}.ffn_up_exps.weight"),   &[])?;
            let down_stacked = idx.take_weight(&format!("blk.{i}.ffn_down_exps.weight"), &[])?;
            let gate_experts = split_stacked_experts(gate_stacked, n_experts)?;
            let up_experts   = split_stacked_experts(up_stacked,   n_experts)?;
            let down_experts = split_stacked_experts(down_stacked, n_experts)?;
            // Per-expert gate+up fusion — saves K matmul launches per token per layer.
            let gate_up_experts: Vec<FfnPair> = gate_experts.into_iter().zip(up_experts.into_iter())
                .map(|(g, u)| FfnPair::from_halves(g, u))
                .collect();
            let moe = MoeFfn {
                router, gate_up_experts, down_experts,
                top_k: expert_used,
                stream: None, // VENDORED-LOCAL: resident experts
                #[cfg(feature = "cuda")]
                gpu_plan: std::sync::OnceLock::new(),
            }.move_to_device(&*backend, SAFETY_BYTES);

            // Auto-fuse shexp gate+up at load time (silu_mul_split halves the
            // matmul launches per layer).
            let shexp_gate = idx.take_weight(&format!("blk.{i}.ffn_gate_shexp.weight"), &[])?;
            let shexp_up   = idx.take_weight(&format!("blk.{i}.ffn_up_shexp.weight"),   &[])?;
            let shexp_pair = FfnPair::from_halves(shexp_gate, shexp_up)
                .try_to_device(&*backend, SAFETY_BYTES);
            let shexp_down = upload_w(idx.take_weight(&format!("blk.{i}.ffn_down_shexp.weight"), &[])?);
            // Pre-reshape to `[1, hidden]` at load so the per-forward path can
            // call `linear(xn, &shexp_router)` directly without a device clone.
            let shexp_router_raw = idx.take(&format!("blk.{i}.ffn_gate_inp_shexp.weight"), &[])?;
            let hidden_dim = shexp_router_raw.dim(0);
            let shexp_router = upload_dense(
                shexp_router_raw.reshape(vec![1, hidden_dim]).expect("shexp_router [1, hidden]"),
            );

            let ffn = Qwen35MoeFfn {
                post_norm, moe, shexp_pair, shexp_down, shexp_router,
            };

            // ----- attention block (SSM or full attention) -----
            let blk = if is_attn {
                Qwen35MoeBlock::Attention {
                    attn_norm:   upload_dense(idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?),
                    attn_q:      upload_w(idx.take_weight(&format!("blk.{i}.attn_q.weight"), &[])?),
                    attn_q_norm: upload_dense(idx.take(&format!("blk.{i}.attn_q_norm.weight"), &[])?),
                    attn_k:      upload_w(idx.take_weight(&format!("blk.{i}.attn_k.weight"), &[])?),
                    attn_k_norm: upload_dense(idx.take(&format!("blk.{i}.attn_k_norm.weight"), &[])?),
                    attn_v:      upload_w(idx.take_weight(&format!("blk.{i}.attn_v.weight"), &[])?),
                    attn_output: upload_w(idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?),
                    ffn,
                }
            } else {
                let ssm_alpha = idx.take_weight(&format!("blk.{i}.ssm_alpha.weight"), &[])?;
                let ssm_beta  = idx.take_weight(&format!("blk.{i}.ssm_beta.weight"), &[])?;
                let ssm_ba    = upload_w(fuse_alpha_beta(ssm_beta, ssm_alpha));
                Qwen35MoeBlock::Ssm {
                    attn_norm:   upload_dense(idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?),
                    attn_qkv:    upload_w(idx.take_weight(&format!("blk.{i}.attn_qkv.weight"), &[])?),
                    attn_gate:   upload_w(idx.take_weight(&format!("blk.{i}.attn_gate.weight"), &[])?),
                    ssm_conv1d:  upload_dense(idx.take(&format!("blk.{i}.ssm_conv1d.weight"), &[])?),
                    ssm_a:       upload_dense(idx.take(&format!("blk.{i}.ssm_a"), &[])?),
                    ssm_ba,
                    ssm_dt_bias: upload_dense(idx.take(&format!("blk.{i}.ssm_dt.bias"), &[])?),
                    ssm_norm:    upload_dense(idx.take(&format!("blk.{i}.ssm_norm.weight"), &[])?),
                    ssm_out:     upload_w(idx.take_weight(&format!("blk.{i}.ssm_out.weight"), &[])?),
                    ffn,
                }
            };
            blocks.push(blk);
        }

        Ok(Self {
            config, ssm_cfg, attention_layers, tokenizer, blocks,
            tok_embd: tok_embd_d, output_norm: output_norm_d, output: output_d,
            backend, n_experts, expert_used,
        })
    }

    /// Look up token embeddings — companion to `forward_embeds` for vision splice.
    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        self.backend.embed_lookup(&self.tok_embd, tokens, self.config.embedding_dim)
    }

    /// Vision-language splice helper. Same contract as `Qwen35Model`'s.
    pub fn embed_with_vision_at_placeholder(
        &self,
        prompt:               &str,
        soft_tokens:          &Tensor,
        placeholder_token_id: u32,
        add_bos:              bool,
    ) -> Result<(Vec<u32>, Tensor)> {
        let prompt_ids = self.tokenizer.encode(prompt, add_bos)?;
        let pos = prompt_ids.iter().position(|&id| id == placeholder_token_id)
            .ok_or_else(|| LlamaError::Config(format!(
                "placeholder token id {placeholder_token_id} not found in prompt"
            )))?;
        let n_soft = soft_tokens.dim(0);
        let d = self.config.embedding_dim;
        debug_assert_eq!(soft_tokens.dim(soft_tokens.rank() - 1), d);
        let mut tokens = Vec::with_capacity(prompt_ids.len() - 1 + n_soft);
        tokens.extend_from_slice(&prompt_ids[..pos]);
        tokens.extend(std::iter::repeat(placeholder_token_id).take(n_soft));
        tokens.extend_from_slice(&prompt_ids[pos + 1..]);
        let embeds = self.embed_text(&tokens);
        let mut host = embeds.to_host();
        let dst = host.data_mut();
        let src = soft_tokens.to_host();
        let src_data = src.data();
        let splice_off = pos * d;
        let splice_len = n_soft * d;
        dst[splice_off..splice_off + splice_len].copy_from_slice(&src_data[..splice_len]);
        let embeds = self.backend.to_device(host);
        Ok((tokens, embeds))
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Tensor> {
        let embeds = self.embed_text(tokens);
        self.forward_embeds(&embeds, tokens.len(), kv)
    }

    pub fn forward_embeds(&self, embeds: &Tensor, seq: usize, kv: &mut KvCache) -> Result<Tensor> {
        let cfg = &self.config;
        let backend = &*self.backend;
        let past = kv.len;
        debug_assert_eq!(embeds.dim(0), seq);

        let head_dim = cfg.head_dim;
        let n_h_kv  = cfg.n_kv_heads;
        let q_per_head_full = self.first_attn_q_out_dim() / cfg.n_heads;
        debug_assert_eq!(q_per_head_full, 2 * head_dim,
            "expected q_per_head_full=2*head_dim; got {q_per_head_full} vs {}", 2 * head_dim);
        let n_h = cfg.n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut x = embeds.clone();
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for (layer, blk) in self.blocks.iter().enumerate() {
            match blk {
                Qwen35MoeBlock::Attention { attn_norm, attn_q, attn_q_norm, attn_k, attn_k_norm,
                                            attn_v, attn_output, ffn } => {
                    let xn = ops::rmsnorm(backend, &x, attn_norm, cfg.rms_eps);
                    let q_full = attn_q.linear(backend, &xn);
                    let k_flat = attn_k.linear(backend, &xn);
                    let v_flat = attn_v.linear(backend, &xn);

                    let (q_only, q_gate) = backend.split_q_and_gate(&q_full, n_h, head_dim);
                    let q_3d = q_only.reshape(vec![seq, n_h, head_dim]).expect("q reshape");
                    let k_3d = k_flat.reshape(vec![seq, n_h_kv, head_dim]).expect("k reshape");
                    let v_3d = v_flat.reshape(vec![seq, n_h_kv, head_dim]).expect("v reshape");

                    let mut q = ops::rmsnorm(backend, &q_3d, attn_q_norm, cfg.rms_eps);
                    let mut k = ops::rmsnorm(backend, &k_3d, attn_k_norm, cfg.rms_eps);

                    if cfg.rope_dim < head_dim {
                        backend.rope_partial_neox(&mut q, &positions, head_dim, cfg.rope_dim, cfg.rope_theta);
                        backend.rope_partial_neox(&mut k, &positions, head_dim, cfg.rope_dim, cfg.rope_theta);
                    } else {
                        let rope_type = cfg.arch.rope_type();
                        ops::rope(backend, &mut q, &positions, head_dim, rope_type, cfg.rope_theta);
                        ops::rope(backend, &mut k, &positions, head_dim, rope_type, cfg.rope_theta);
                    }

                    kv.append(backend, layer, &k, &v_3d);
                    let kv_len = kv.len + seq;
                    let attn_out = ops::attention(
                        backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                        kv_len, scale, past,
                    );

                    let mut attn_t = attn_out.reshape(vec![seq, n_h * head_dim]).expect("attn reshape");
                    backend.mul_sigmoid_inplace(&mut attn_t, &q_gate);

                    let attn_proj = attn_output.linear(backend, &attn_t);
                    let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &ffn.post_norm, cfg.rms_eps);
                    let ffn_out = qwen35moe_ffn_forward(backend, &xn2, ffn);
                    ops::add_inplace(backend, &mut x, &ffn_out);
                }
                Qwen35MoeBlock::Ssm { attn_norm, attn_qkv, attn_gate, ssm_conv1d, ssm_a,
                                      ssm_ba, ssm_dt_bias, ssm_norm, ssm_out, ffn } => {
                    let xn = ops::rmsnorm(backend, &x, attn_norm, cfg.rms_eps);
                    let mixed_qkv = attn_qkv.linear(backend, &xn);
                    let z         = attn_gate.linear(backend, &xn);
                    let beta_alpha = ssm_ba.linear(backend, &xn);

                    let num_v_heads = self.ssm_cfg.time_step_rank;
                    let num_k_heads = self.ssm_cfg.group_count;
                    let head_v_dim  = self.ssm_cfg.inner_size / num_v_heads;
                    let head_k_dim  = self.ssm_cfg.state_size;
                    let v_per_k     = num_v_heads / num_k_heads;
                    let conv_kernel = self.ssm_cfg.conv_kernel;
                    let conv_dim    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
                    let scale_q     = 1.0 / (head_v_dim as f32).sqrt();

                    if kv.ssm_state[layer].is_none() {
                        kv.ssm_state[layer] = Some(backend.to_device(
                            Tensor::zeros(vec![num_v_heads, head_v_dim, head_v_dim])));
                    }
                    if kv.ssm_conv[layer].is_none() {
                        kv.ssm_conv[layer] = Some(backend.to_device(
                            Tensor::zeros(vec![conv_kernel - 1, conv_dim])));
                    }
                    let mut state_dev = kv.ssm_state[layer].take().unwrap();
                    if state_dev.is_cpu() { state_dev = backend.to_device(state_dev); }
                    let mut conv_dev  = kv.ssm_conv [layer].take().unwrap();
                    if conv_dev.is_cpu()  { conv_dev  = backend.to_device(conv_dev);  }
                    let attn_out_dev = backend.delta_net_step(
                        &mixed_qkv, &z, &beta_alpha,
                        ssm_conv1d, ssm_a, ssm_dt_bias, ssm_norm,
                        &mut conv_dev, &mut state_dev,
                        seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k,
                        scale_q, cfg.rms_eps,
                    );
                    kv.ssm_state[layer] = Some(state_dev);
                    kv.ssm_conv [layer] = Some(conv_dev);

                    let attn_proj = ssm_out.linear(backend, &attn_out_dev);
                    let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &ffn.post_norm, cfg.rms_eps);
                    let ffn_out = qwen35moe_ffn_forward(backend, &xn2, ffn);
                    ops::add_inplace(backend, &mut x, &ffn_out);
                }
            }
        }

        kv.commit(seq);
        let x = ops::rmsnorm(backend, &x, &self.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        Ok(self.output.linear(backend, &x_last))
    }

    fn first_attn_q_out_dim(&self) -> usize {
        for blk in &self.blocks {
            if let Qwen35MoeBlock::Attention { attn_q, .. } = blk {
                return attn_q.shape()[0];
            }
        }
        self.config.n_heads * self.config.head_dim
    }
}

/// Run a Qwen3.6-MoE FFN block: MoE(top-K experts) + sigmoid-gated shared
/// expert, summed. Caller provides the *already-normalized* `xn` (so they can
/// fuse the upstream residual add into the rmsnorm). Returns `[seq, hidden]`
/// ready to be added to the residual stream.
fn qwen35moe_ffn_forward(backend: &dyn Backend, xn: &Tensor, ffn: &Qwen35MoeFfn) -> Tensor {
    // ----- routed MoE -----
    // Standard router: logits = xn @ router^T, then top-K softmax + per-expert SwiGLU + accumulate.
    let opts = MoeOptions::default(); // SwiGLU (silu+mul), no per-expert output scale
    let router_logits = ffn.moe.router.linear(backend, &xn);
    let moe_out = moe_forward_with_logits(backend, &xn, &ffn.moe, &router_logits, &opts);

    // ----- shared expert -----
    let activated = ffn.shexp_pair.swiglu(backend, &xn);
    let mut shexp = ffn.shexp_down.linear(backend, &activated);

    // Per-token sigmoid gate: shexp_router was pre-reshaped to `[1, hidden]`
    // at load, so `linear(xn, [1, hidden]) = xn @ [1, hidden]^T = [seq, 1]`
    // — no per-forward device clone of the gate weight.
    let mut gate_logits = backend.linear(&xn, &ffn.shexp_router);  // [seq, 1]
    gate_logits = ops::sigmoid(backend, &gate_logits);

    // Apply per-row sigmoid gate fully on device — `mul_inplace_broadcast_axis0`
    // does `shexp[t, j] *= gate_logits[t]` in one kernel launch, no host sync.
    backend.mul_inplace_broadcast_axis0(&mut shexp, &gate_logits);

    // ----- sum -----
    let mut out = moe_out;
    ops::add_inplace(backend, &mut out, &shexp);
    out
}
