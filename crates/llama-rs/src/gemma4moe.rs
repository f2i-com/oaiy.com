//! Gemma 4 MoE — `arch=gemma4` with `gemma4.expert_count > 0`.
//!
//! Hybrid dense + MoE per layer: each block runs both a **shared dense FFN**
//! AND a **sparse MoE FFN** in parallel from the same `attn_out`, sums them,
//! and adds back to the residual. Per
//! `llama.cpp/src/models/gemma4.cpp`'s `is_moe_layer` branch:
//! ```text
//!   cur_mlp = rmsnorm(attn_out, ffn_norm)            # shared dense path
//!   cur_mlp = gelu(gate(cur_mlp)) * up(cur_mlp)      # GELU MLP dense
//!   cur_mlp = down(cur_mlp)
//!   cur_mlp = rmsnorm(cur_mlp, post_ffw_norm_1)
//!
//!   cur_moe = rmsnorm(attn_out, pre_ffw_norm_2)      # MoE path
//!   tmp     = rmsnorm_no_scale(attn_out) / sqrt(n_embd)
//!   tmp     = tmp * ffn_gate_inp_s                   # custom router input
//!   logits  = ffn_gate_inp(tmp)
//!   cur_moe = moe_ffn(cur_moe, logits, GELU, ffn_down_exps_s)
//!   cur_moe = rmsnorm(cur_moe, post_ffw_norm_2)
//!
//!   cur     = cur_mlp + cur_moe
//!   cur     = rmsnorm(cur, post_ffw_norm)
//!   x       = cur + attn_out                          # residual
//!   if out_scale: x *= layer_output_scale
//! ```
//!
//! **Per-layer KV via missing tensors**: Gemma 4 26B-A4B uses tensor omission
//! (NOT `shared_kv_layers` metadata) for KV sharing. Global (non-SWA) layers
//! ship `attn_q` + `attn_k` but **no `attn_v`** — when V is missing, llama.cpp
//! reuses K as V (MQA fallback): `Vcur = Kcur` if `wv == nullptr`. We mirror
//! that by storing `attn_v: Option<Weight>` and using K as V when None.
//! Pattern in 26B-A4B: every 6th layer (5, 11, 17, 23, 29) is a global layer
//! with this MQA fallback; the other 25 are SWA layers with full Q/K/V.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::gemma4::{compute_per_layer_inputs, load_or_unit, Gemma4BlockExtras, Gemma4GlobalTensors};
use crate::kv_cache::KvCache;
use crate::loader::{load_lm_head_or_tied, upload_tok_embd_and_lm_head, FfnPair, TensorIndex, Weight};
use crate::moe::{moe_forward_with_logits, MoeFfn, MoeOptions};
use crate::qwen3moe::split_stacked_experts;
use crate::{LlamaError, ModelConfig, Result};

/// Per-block tensors for Gemma 4 MoE. Differs from `CommonBlockTensors` by
/// making `attn_v` optional (global layers in 26B-A4B omit it for MQA-style
/// V=K fallback).
#[derive(Debug)]
pub struct Gemma4MoeBlockTensors {
    pub attn_norm:   Tensor,
    pub attn_q:      Weight,
    pub attn_k:      Weight,
    /// `None` ⇒ use K as V at attention time (llama.cpp's MQA fallback).
    pub attn_v:      Option<Weight>,
    pub attn_output: Weight,
    pub ffn_norm:    Tensor,
    pub ffn_pair:    FfnPair,
    pub ffn_down:    Weight,
}

/// MoE-specific per-block tensors. Lives alongside both the `CommonBlockTensors`-
/// style attention/dense-FFN and the dense-Gemma-4 `Gemma4BlockExtras` (norms,
/// PLE).
#[derive(Debug)]
pub struct Gemma4MoeBlockExtras {
    pub post_ffw_norm_1: Tensor,
    pub pre_ffw_norm_2:  Tensor,
    pub post_ffw_norm_2: Tensor,
    pub ffn_gate_inp_s:  Tensor,
    pub moe:             MoeFfn,
    pub down_exps_s_host: Vec<f32>,
}

pub struct Gemma4MoeModel {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub tok_embd:    Arc<Tensor>,
    pub output_norm: Tensor,
    pub output:      Weight,
    pub blocks:      Vec<Gemma4MoeBlockTensors>,
    pub global:    Option<Gemma4GlobalTensors>,
    pub extras:    Vec<Gemma4BlockExtras>,
    pub moe_extras: Vec<Gemma4MoeBlockExtras>,
    pub backend:   Arc<dyn Backend>,
    embed_scale_const: Option<f32>,
    per_layer_dim:     usize,
    n_experts: usize,
    expert_used: usize,
}

impl std::fmt::Debug for Gemma4MoeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4MoeModel")
            .field("config", &self.config)
            .field("backend", &self.backend.name())
            .field("vocab_size", &self.tokenizer.vocab_size())
            .field("n_experts", &self.n_experts)
            .field("expert_used", &self.expert_used)
            .finish()
    }
}

impl Gemma4MoeModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_experts = g.get_u64("gemma4.expert_count")? as usize;
        let expert_used = g.get_u64("gemma4.expert_used_count")? as usize;
        if n_experts == 0 {
            return Err(LlamaError::Config(
                "Gemma4MoeModel called with expert_count=0; should have been routed to Gemma4Model".into(),
            ));
        }

        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        // ----- top-level tensors (tok_embd / output_norm / output) -----
        let tok_embd = Arc::new(idx.take("token_embd.weight", &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        const SAFETY_BYTES: usize = 2 * 1024 * 1024 * 1024;
        let upload_dense = |t: Tensor| backend.to_device(t);
        let upload_w = |w: Weight| w.try_to_device(&*backend, SAFETY_BYTES);

        // Tied LM head shares Arc with tok_embd → upload once.
        let (tok_embd_d, output) = upload_tok_embd_and_lm_head(&*backend, tok_embd, output);
        let output_norm_d = upload_dense(output_norm);
        let output_d = upload_w(output);

        // ----- PLE globals (optional) -----
        let per_layer_dim = g.get_u64("gemma4.embedding_length_per_layer_input")
            .unwrap_or(0) as usize;
        let global = if per_layer_dim > 0 {
            let ple_info = g.tensor_by_name("per_layer_token_embd.weight")
                .ok_or_else(|| LlamaError::MissingTensor("per_layer_token_embd.weight".into()))?;
            let n_elem = ple_info.numel() as usize;
            let mut ple_host = vec![0.0f32; n_elem];
            // VENDORED-LOCAL: byte-source seam (mmap'd and source-backed files).
            ggml_quants::dequantize(ple_info.dtype, &g.tensor_bytes(ple_info)?, &mut ple_host)?;
            let vocab = config.vocab_size;
            let row_size = config.n_layers * per_layer_dim;
            let per_layer_token_embd = backend.to_device(
                Tensor::from_vec(ple_host, vec![vocab, row_size]),
            );
            let per_layer_model_proj = backend.to_device(idx.take("per_layer_model_proj.weight", &[])?);
            let per_layer_proj_norm  = backend.to_device(idx.take("per_layer_proj_norm.weight",  &[])?);
            let rope_freq_factors = idx.take("rope_freqs.weight", &[]).ok().map(|t| {
                let host = t.to_host();
                host.data().to_vec()
            });
            Some(Gemma4GlobalTensors {
                per_layer_model_proj, per_layer_proj_norm, per_layer_token_embd, rope_freq_factors,
            })
        } else {
            None
        };

        // ----- per-block tensors (attention + dense FFN, with Optional attn_v) -----
        let mut blocks = Vec::with_capacity(config.n_layers);
        let mut extras = Vec::with_capacity(config.n_layers);
        let mut moe_extras = Vec::with_capacity(config.n_layers);

        for i in 0..config.n_layers {
            let hd_layer = config.layer_head_dim(i);

            // attention: attn_v optional (V=K MQA fallback when absent)
            let attn_v_opt = idx.take_weight(&format!("blk.{i}.attn_v.weight"), &[]).ok();
            blocks.push(Gemma4MoeBlockTensors {
                attn_norm:   upload_dense(idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?),
                attn_q:      upload_w(idx.take_weight(&format!("blk.{i}.attn_q.weight"),      &[])?),
                attn_k:      upload_w(idx.take_weight(&format!("blk.{i}.attn_k.weight"),      &[])?),
                attn_v:      attn_v_opt.map(|w| w.try_to_device(&*backend, SAFETY_BYTES)),
                attn_output: upload_w(idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?),
                ffn_norm:    upload_dense(idx.take(&format!("blk.{i}.ffn_norm.weight"), &[])?),
                ffn_pair:    {
                    let ffn_gate = idx.take_weight(&format!("blk.{i}.ffn_gate.weight"), &[])?;
                    let ffn_up   = idx.take_weight(&format!("blk.{i}.ffn_up.weight"),   &[])?;
                    FfnPair::from_halves(ffn_gate, ffn_up)
                        .try_to_device(&*backend, SAFETY_BYTES)
                },
                ffn_down:    upload_w(idx.take_weight(&format!("blk.{i}.ffn_down.weight"),    &[])?),
            });

            // dense extras (post-norms, PLE)
            let per_layer_inp_gate = if global.is_some() {
                idx.take_weight(&format!("blk.{i}.inp_gate.weight"), &[])
                    .ok().map(|w| w.try_to_device(&*backend, SAFETY_BYTES))
            } else { None };
            let per_layer_proj = if global.is_some() {
                idx.take_weight(&format!("blk.{i}.proj.weight"), &[])
                    .ok().map(|w| w.try_to_device(&*backend, SAFETY_BYTES))
            } else { None };
            let per_layer_post_norm = if global.is_some() {
                idx.take(&format!("blk.{i}.post_norm.weight"), &[])
                    .ok().map(|t| backend.to_device(t))
            } else { None };
            let layer_output_scale = idx.take(&format!("blk.{i}.layer_output_scale.weight"), &[])
                .ok()
                .and_then(|t| t.data().first().copied());

            extras.push(Gemma4BlockExtras {
                attn_post_norm: upload_dense(idx.take(
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &[&format!("blk.{i}.attn_post_norm.weight")],
                )?),
                ffn_post_norm: upload_dense(idx.take(
                    &format!("blk.{i}.post_ffw_norm.weight"),
                    &[&format!("blk.{i}.ffn_post_norm.weight")],
                )?),
                attn_q_norm: upload_dense(load_or_unit(
                    &idx, &format!("blk.{i}.attn_q_norm.weight"), hd_layer,
                )?),
                attn_k_norm: upload_dense(load_or_unit(
                    &idx, &format!("blk.{i}.attn_k_norm.weight"), hd_layer,
                )?),
                per_layer_inp_gate, per_layer_proj, per_layer_post_norm,
                layer_output_scale,
            });

            // MoE extras
            let post_ffw_norm_1 = upload_dense(idx.take(&format!("blk.{i}.post_ffw_norm_1.weight"), &[])?);
            let pre_ffw_norm_2  = upload_dense(idx.take(&format!("blk.{i}.pre_ffw_norm_2.weight"),  &[])?);
            let post_ffw_norm_2 = upload_dense(idx.take(&format!("blk.{i}.post_ffw_norm_2.weight"), &[])?);
            let ffn_gate_inp_s  = upload_dense(idx.take(&format!("blk.{i}.ffn_gate_inp.scale"),     &[])?);

            let router = idx.take_weight(&format!("blk.{i}.ffn_gate_inp.weight"), &[])?;
            let gate_up_stacked = idx.take_weight(&format!("blk.{i}.ffn_gate_up_exps.weight"), &[])?;
            let down_stacked    = idx.take_weight(&format!("blk.{i}.ffn_down_exps.weight"),    &[])?;
            let down_per_expert = split_stacked_experts(down_stacked, n_experts)?;
            // Gemma 4 MoE ships gate+up already concatenated as `ffn_gate_up_exps`
            // — each per-expert tensor is `[2*ff, hidden]`, ready to plug into
            // `FfnPair::Fused`. No need to split halves and re-join.
            let gate_up_per_expert = split_stacked_experts(gate_up_stacked, n_experts)?;
            let gate_up_experts: Vec<FfnPair> = gate_up_per_expert.into_iter()
                .map(FfnPair::Fused)
                .collect();

            let moe = MoeFfn {
                router, gate_up_experts, down_experts: down_per_expert,
                top_k: expert_used,
                stream: None, // VENDORED-LOCAL: resident experts
                #[cfg(feature = "cuda")]
                gpu_plan: std::sync::OnceLock::new(),
            }.move_to_device(&*backend, SAFETY_BYTES);

            let down_exps_s_t = idx.take(&format!("blk.{i}.ffn_down_exps.scale"), &[])?;
            let down_exps_s_host = down_exps_s_t.data().to_vec();

            moe_extras.push(Gemma4MoeBlockExtras {
                post_ffw_norm_1, pre_ffw_norm_2, post_ffw_norm_2,
                ffn_gate_inp_s, moe, down_exps_s_host,
            });
        }

        let embed_scale_const = if config.embedding_scale {
            Some((config.embedding_dim as f32).sqrt())
        } else {
            None
        };

        Ok(Self {
            config, tokenizer,
            tok_embd: tok_embd_d, output_norm: output_norm_d, output: output_d,
            blocks, global, extras, moe_extras, backend,
            embed_scale_const, per_layer_dim,
            n_experts, expert_used,
        })
    }

    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        let backend = &*self.backend;
        let cfg = &self.config;
        let mut x = backend.embed_lookup(&self.tok_embd, tokens, cfg.embedding_dim);
        if let Some(s) = self.embed_scale_const {
            ops::mul_scalar_inplace(backend, &mut x, s);
        }
        x
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        let embeds = self.embed_text(tokens);
        self.forward_embeds(&embeds, tokens, kv)
    }

    pub fn forward_embeds(
        &self,
        embeds: &Tensor,
        tokens: &[u32],
        kv:     &mut KvCache,
    ) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let seq = tokens.len();
        debug_assert_eq!(embeds.dim(0), seq);
        let past = kv.len;
        let n_h = cfg.n_heads;
        let n_layers = cfg.n_layers;
        let pld = self.per_layer_dim;
        let n_embd = cfg.embedding_dim;
        let inv_sqrt_n = 1.0f32 / (n_embd as f32).sqrt();

        let mut x = embeds.clone();

        let per_layer_inputs: Option<Tensor> = self.global.as_ref()
            .map(|g| compute_per_layer_inputs(backend, &x, tokens, &g.per_layer_token_embd,
                                              &g.per_layer_model_proj, &g.per_layer_proj_norm,
                                              cfg.embedding_dim, seq, n_layers, pld, cfg.rms_eps));

        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..n_layers {
            let b  = &self.blocks[layer];
            let gx = &self.extras[layer];
            let mx = &self.moe_extras[layer];

            let hd = cfg.layer_head_dim(layer);
            // Per-layer n_kv_heads — read from K projection's output dim. K is
            // [n_kv * hd, hidden]; n_kv = K.shape[0] / hd.
            let n_kv = b.attn_k.shape()[0] / hd;
            let scale = 1.0f32; // Gemma 4 hardcoded
            let rope_theta = cfg.layer_rope_theta(layer);

            // ----- attention -----
            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);
            let is_swa = cfg.layer_uses_sliding_window(layer);
            let freq_factors: Option<&[f32]> = if !is_swa {
                self.global.as_ref().and_then(|g| g.rope_freq_factors.as_deref())
            } else { None };

            let q_flat = b.attn_q.linear(backend, &xn);
            let q_3d = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let mut q = ops::rmsnorm(backend, &q_3d, &gx.attn_q_norm, cfg.rms_eps);
            let rope_type = cfg.arch.rope_type();
            ops::rope_with_factors(backend, &mut q, &positions, hd, rope_type, rope_theta, freq_factors);

            let k_flat = b.attn_k.linear(backend, &xn);
            // V: either compute fresh (own attn_v), or reuse K (MQA fallback).
            // llama.cpp gemma4.cpp: `Vcur = wv ? linear(wv, cur) : Kcur;`
            let v_flat = match &b.attn_v {
                Some(wv) => wv.linear(backend, &xn),
                None     => k_flat.clone(),
            };

            let v_normed = ops::rmsnorm_no_scale(backend, &v_flat, cfg.rms_eps);
            let k_3d = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
            let v = v_normed.reshape(vec![seq, n_kv, hd]).expect("v reshape");
            let mut k = ops::rmsnorm(backend, &k_3d, &gx.attn_k_norm, cfg.rms_eps);
            ops::rope_with_factors(backend, &mut k, &positions, hd, rope_type, rope_theta, freq_factors);

            kv.append(backend, layer, &k, &v);
            let kv_len = kv.len + seq;
            let sw = if cfg.layer_uses_sliding_window(layer) { cfg.sliding_window } else { None };
            let attn_out_t = ops::attention_swa(
                backend, &q,
                kv.k_buffer(layer), kv.v_buffer(layer),
                kv_len, scale, past, sw,
            );

            let attn_t = attn_out_t.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let mut attn_proj = b.attn_output.linear(backend, &attn_t);
            attn_proj = ops::rmsnorm(backend, &attn_proj, &gx.attn_post_norm, cfg.rms_eps);

            let mut attn_added = x.clone();
            ops::add_inplace(backend, &mut attn_added, &attn_proj);
            // attn_added = attn_out (the input to the FFN block)

            // ----- shared dense FFN branch -----
            let xn_dense = ops::rmsnorm(backend, &attn_added, &b.ffn_norm, cfg.rms_eps);
            let gated = b.ffn_pair.geglu(backend, &xn_dense);
            let mut cur_mlp = b.ffn_down.linear(backend, &gated);
            cur_mlp = ops::rmsnorm(backend, &cur_mlp, &mx.post_ffw_norm_1, cfg.rms_eps);

            // ----- MoE branch -----
            let cur_moe_in = ops::rmsnorm(backend, &attn_added, &mx.pre_ffw_norm_2, cfg.rms_eps);

            // Custom router input: rmsnorm_no_scale(attn_out) / sqrt(n_embd) * ffn_gate_inp_s
            let mut tmp = ops::rmsnorm_no_scale(backend, &attn_added, cfg.rms_eps);
            ops::mul_scalar_inplace(backend, &mut tmp, inv_sqrt_n);
            backend.mul_inplace_broadcast_last(&mut tmp, &mx.ffn_gate_inp_s);
            let logits = mx.moe.router.linear(backend, &tmp);

            let opts = MoeOptions {
                use_gelu: true,
                down_exps_scale_host: Some(&mx.down_exps_s_host),
            };
            let mut cur_moe = moe_forward_with_logits(backend, &cur_moe_in, &mx.moe, &logits, &opts);
            cur_moe = ops::rmsnorm(backend, &cur_moe, &mx.post_ffw_norm_2, cfg.rms_eps);

            // ----- combine + final norm + residual -----
            let mut cur = cur_mlp;
            ops::add_inplace(backend, &mut cur, &cur_moe);
            cur = ops::rmsnorm(backend, &cur, &gx.ffn_post_norm, cfg.rms_eps);
            ops::add_inplace(backend, &mut cur, &attn_added);

            // ----- PLE injection (after combined residual) -----
            // `inp_gate_w.linear(&cur)` borrows `cur`, so we don't need the
            // pre-PLE snapshot — just accumulate `normed` into `cur` directly.
            if let (Some(pli_dev), Some(inp_gate_w), Some(proj_w), Some(post_norm_t))
                = (per_layer_inputs.as_ref(),
                   gx.per_layer_inp_gate.as_ref(),
                   gx.per_layer_proj.as_ref(),
                   gx.per_layer_post_norm.as_ref())
            {
                let mut gated = inp_gate_w.linear(backend, &cur);
                gated = ops::gelu_approx(backend, &gated);
                let layer_pli = ops::slice_axis1_2d(backend, pli_dev, layer);
                ops::mul_inplace(backend, &mut gated, &layer_pli);
                let projected_back = proj_w.linear(backend, &gated);
                let normed = ops::rmsnorm(backend, &projected_back, post_norm_t, cfg.rms_eps);
                ops::add_inplace(backend, &mut cur, &normed);
            }

            if let Some(s) = gx.layer_output_scale {
                ops::mul_scalar_inplace(backend, &mut cur, s);
            }

            x = cur;
        }

        kv.commit(seq);

        let x = ops::rmsnorm(backend, &x, &self.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        let mut logits = self.output.linear(backend, &x_last);
        if let Some(softcap) = cfg.final_logit_softcap {
            ops::mul_scalar_inplace(backend, &mut logits, 1.0 / softcap);
            ops::tanh_inplace(backend, &mut logits);
            ops::mul_scalar_inplace(backend, &mut logits, softcap);
        }
        logits
    }
}
