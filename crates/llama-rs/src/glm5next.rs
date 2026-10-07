// VENDORED-LOCAL: whole module. GLM-5.3-Flash (`arch=glm5next`).
//! GLM-5.3-Flash — `general.architecture = "glm5next"`, MIT, 313B-A17B.
//!
//! Reference: llama.cpp PR #27754 (`unslothai/llama.cpp`, branch
//! `glm5next/upstream`), pinned at `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`,
//! file `src/models/glm5next.cpp`. `glm5next` is **not** in llama.cpp master —
//! master carries the metadata vocabulary (`{arch}.kda.*`,
//! `{arch}.hyper_connection.*`, `SSM_F_A`/`SSM_F_B`/`SSM_G_A`/`SSM_G_B`) and the
//! sibling models the PR builds on: `kimi-linear.cpp` (KDA with low-rank gates),
//! `glm-dsa.cpp` (MLA + lightning indexer), `delta-net-base.cpp`
//! (`build_recurrent_attn`) and `deepseek4.cpp` (hyper-connections — the PR's
//! `glm5next::graph` inherits from `deepseek4::graph` and reuses `build_hc_pre`
//! / `build_hc_post` / `build_hc_sinkhorn` verbatim).
//!
//! ## Shape of the model
//!
//! 46 `blk.*` entries but a **45-layer trunk**: `blk.45` is the NextN/MTP draft
//! block, which is why `hc_*` exists only on `blk.0..=44`. `n_layer` here means
//! the trunk; the MTP block is loaded separately and is not part of a plain
//! decode.
//!
//! Attention alternates by a **per-layer array**, not a modular interval:
//! `glm5next.attention.head_count_kv` is 46 entries of 0/1 where 0 marks a KDA
//! linear-attention layer and 1 a full MLA layer. For the released weights that
//! is 34 KDA + 11 MLA over the trunk, with the MLA layers at
//! 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 — and the stride breaks at the end,
//! so a `% 4` rule would be wrong. `blk.45` (MTP) is MLA-shaped too.
//!
//! FFN: `blk.0..=2` are dense (`leading_dense_block_count = 3`), `blk.3..=45`
//! are MoE — 288 experts, top-8, plus one always-on shared expert.
//!
//! The whole text tower is **NoPE**: `rope.dimension_count = 0` and the GGUF
//! carries no rope tensors. The reference asserts this
//! (`GGML_ASSERT(hparams.n_rot() == 0 && "glm5next MLA is nope-only")`), so a
//! ported rotation from DeepSeek-V4.1 would be wrong here.
//!
//! ## Tensors, per layer kind
//!
//! GGUF dims below are as published; `loader::load_tensor_f32` reverses them, so
//! the Rust `shape()[0]` is the **output** dim. `d_inner = kda.head_dim * n_head`
//! = 8192, `hc_dim = hc_count * n_embd` = 16384, `hc_mix = (2 + hc_count) *
//! hc_count` = 24.
//!
//! | tensor | GGUF dims | present on |
//! |---|---|---|
//! | `attn_norm`, `ffn_norm` | `[n_embd]` | every block |
//! | `hc_{attn,ffn}_fn` | `[hc_dim, hc_mix]` | trunk only (0..=44) |
//! | `hc_{attn,ffn}_base` | `[hc_mix]` | trunk only |
//! | `hc_{attn,ffn}_scale` | `[3]` | trunk only |
//! | `attn_{q,k,v}` | `[n_embd, d_inner]` | KDA |
//! | `ssm_conv1d_{q,k,v}` | `[d_conv, 1, d_inner]` | KDA |
//! | `ssm_f_a` / `ssm_f_b` | `[n_embd, hd]` / `[hd, d_inner]` | KDA |
//! | `ssm_g_a` / `ssm_g_b` | `[n_embd, hd]` / `[hd, d_inner]` | KDA |
//! | `ssm_beta` | `[n_embd, n_head]` | KDA |
//! | `ssm_a` | `[n_head]` | KDA |
//! | `ssm_dt.bias` | `[d_inner]` | KDA |
//! | `ssm_norm` | `[kda.head_dim]` | KDA |
//! | `attn_output` | `[d_inner, n_embd]` | KDA |
//! | `attn_q_a` / `attn_q_a_norm` | `[n_embd, qr]` / `[qr]` | MLA |
//! | `attn_q_b` | `[qr, n_head * qk_head]` | MLA |
//! | `attn_kv_a_mqa` / `attn_kv_a_norm` | `[n_embd, kvr]` / `[kvr]` | MLA |
//! | `attn_k_b` | `[qk_head, kvr, n_head]` | MLA |
//! | `attn_v_b` | `[kvr, v_head, n_head]` | MLA |
//! | `attn_output` | `[n_head * v_head, n_embd]` | MLA |
//! | `indexer.attn_k` | `[n_embd, d_idx]` | MLA |
//! | `indexer.attn_q_b` | `[qr, n_ihead * d_idx]` | MLA |
//! | `indexer.k_norm.{weight,bias}` | `[d_idx]` | MLA |
//! | `indexer.proj` | `[n_embd, n_ihead]` | MLA |
//! | `indexer_compressor_gate` | `[n_embd, d_idx]` | MLA |
//! | `indexer_compressor_ape` | `[d_idx, kpool]` | MLA |
//! | `ffn_{gate,up}` / `ffn_down` | `[n_embd, n_ff]` / `[n_ff, n_embd]` | dense (0..=2) |
//! | `ffn_gate_inp` | `[n_embd, n_expert]` | MoE (3..=45) |
//! | `exp_probs_b.bias` | `[n_expert]` | MoE |
//! | `ffn_{gate,up}_exps` | `[n_embd, n_ff_exp, n_expert]` | MoE |
//! | `ffn_down_exps` | `[n_ff_exp, n_embd, n_expert]` | MoE |
//! | `ffn_{gate,up}_shexp` / `ffn_down_shexp` | `[n_embd, n_ff_sh]` / `[n_ff_sh, n_embd]` | MoE |
//! | `nextn.{eh_proj,enorm,hnorm,shared_head_norm}` | — | `blk.45` only |
//!
//! Per-layer quantisation is **mixed** in the published Q4_K_M: `attn_v` is
//! Q6_K on 17 layers and Q4_K on 17, and `ffn_down_exps` / `ffn_down_shexp` are
//! Q6_K on 19 and Q4_K on 24. `expert_stream` already records a per-layer dtype
//! beside a max-sized record region, so this costs padding, not correctness.
//!
//! ## Status
//!
//! Loader and layer map: done, shape-validated against the metadata.
//!
//! **A complete host-f32 reference forward pass exists in [`forward`]**, and it
//! **runs on the released weights** through [`bridge::HostModel`]: the trunk
//! loop, every stage, both attention paths, 45 layers, 288 experts read from the
//! `.gguf` per dispatch. What it has *not* had is a numerical comparison against
//! llama.cpp — that is the remaining correctness gate. [`Glm5NextModel::forward`]
//! is still the stub, because that path wants the device implementation rather
//! than the host reference.
//!
//! The stage modules, and what each still owes:
//!
//!   1. **KDA scan** — the recurrence landed in [`kda`], not yet wired into
//!      `forward`, and the surrounding stages (conv, gates, output norm/gate)
//!      are still to do. One conv over `q‖k‖v` (SiLU on the conv output, not the
//!      projections), L2-norm on Q and K at the reference's own `1e-6`, a
//!      per-channel log-decay `g = gate_lower_bound * sigmoid(-(ssm_a *
//!      (f_b(f_a(x)) + dt_bias)))` and per-head `beta = sigmoid(ssm_beta(x))`,
//!      then the delta rule. `f`, `g` and `beta` read the **layer input**, not
//!      the convolved q/k/v. Output is RMS-normed by `ssm_norm` and gated by a
//!      **plain sigmoid** of `g_b(g_a(x))`, not a SiLU. This is the one genuinely
//!      new kernel: `qwen35`'s `delta_net_step` decays per head, this decays per
//!      channel, so the state update needs a diagonal.
//!   2. **Hyper-connections** — math landed in [`hc`], not yet wired into
//!      `forward`. Ported from `dsv41::hc` (same formulation: glm5next's
//!      reference inherits `build_hc_pre`/`_post`/`_sinkhorn` from
//!      `deepseek4.cpp` unchanged), with three deliberate divergences
//!      documented there: f32 instead of bf16 activations, an unweighted
//!      [`hc::mean`] collapse instead of DeepSeek-V4.1's learned gated head,
//!      and streams initialised as exact copies. The MTP block has no mixer.
//!   3. **MLA + indexer** — the candidate cache landed in [`kpool`] (geometry,
//!      masks, per-pool bias, `select_k`, pool->cell expansion) and the indexer
//!      in [`indexer`] (pooled key, scores, selection), and the absorbed
//!      attention in [`mla`] (query absorption, the composed mask, MQA over the
//!      latent). None of it is wired into `forward` yet. Absorbed form as in `dsv41::attention`, minus the
//!      query rotation. `kq_scale = 1/sqrt(qk_head)` over the MLA head size, not
//!      the absorbed width. The indexer pools `kpool` cells with a softmax over
//!      the slot axis (`ape` added pre-softmax), scores pooled keys with a ReLU
//!      **between** the per-head dot and the head weighting, and takes top-k over
//!      **pools** before expanding to cells.
//!   4. **MoE routing** — math landed in [`routing`], not yet wired into
//!      `forward`. Sigmoid gating with `exp_probs_b` biasing *selection only*
//!      (`noaux_tc` — the gathered weights stay unbiased), then norm over the
//!      sigmoid values and `expert_weights_scale`. Note this is **not**
//!      `dsv41::moe`'s `sqrt(softplus(...))`: that is llama.cpp's separate
//!      `SQRT_SOFTPLUS` gating func, a different function. Still to do: the
//!      dense/shared-expert FFN wiring, where the shared expert is added
//!      **unscaled** (the routed scale applies to routed weights only).
//!   5. **Vision.** `mmproj` `clip.projector_type = "glm5next"`: a 24-block ViT
//!      (1024 dim, 16 heads, patch 14, 448px, `spatial_merge_size = 2`) and a
//!      patch-merger + SwiGLU projector.

pub mod bridge;
// VENDORED-LOCAL: GLM-5.3-Flash. The CPU tier of the expert hierarchy.
pub mod cpu_experts;
pub mod device;
pub mod forward;
pub mod hc;
pub mod indexer;
pub mod kda;
pub mod kpool;
pub mod mla;
pub mod mmproj;
pub mod routing;
pub mod vision;

use std::sync::Arc;

use ggml_rs::{Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::loader::{load_lm_head_or_tied, TensorIndex, Weight};
use crate::moe::MoeFfn;
use crate::{LlamaError, ModelConfig, Result};

/// Which attention a trunk layer runs. Decoded from the per-layer
/// `attention.head_count_kv` array, never from a modular rule — the released
/// weights break the `% 4` stride at the end of the trunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// KDA linear attention: a recurrent state, no KV cache.
    Kda,
    /// Full multi-head latent attention plus the sparse indexer.
    Mla,
}

/// glm5next hyperparameters that [`ModelConfig`] has no field for.
#[derive(Debug, Clone)]
pub struct Glm5NextConfig {
    /// Trunk depth: `block_count - nextn_predict_layers`.
    pub n_layer: usize,
    /// `block_count`, including the trailing NextN block.
    pub n_layer_all: usize,
    /// `nextn_predict_layers` — 1 for the released weights, 0 if converted
    /// with `--no-mtp`.
    pub n_layer_nextn: usize,
    /// Per-block attention kind, `n_layer_all` entries.
    pub layer_kinds: Vec<LayerKind>,
    /// Blocks `0..n_dense_lead` have a dense FFN; the rest are MoE.
    pub n_dense_lead: usize,

    // --- KDA (linear attention) ---
    pub kda_head_dim: usize,
    pub kda_gate_lower_bound: f32,
    pub ssm_conv_kernel: usize,

    // --- MLA ---
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,

    // --- sparse indexer ---
    pub indexer_n_head: usize,
    pub indexer_head_dim: usize,
    pub indexer_top_k: usize,
    pub indexer_kpool: usize,

    // --- hyper-connections ---
    pub hc_count: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // --- MoE ---
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    /// 0 = softmax, 2 = sigmoid. glm5next ships 2.
    pub expert_gating_func: u32,
    /// Per-layer SwiGLU clamp for the routed experts, `n_layer_all` entries
    /// (10.0 throughout the released weights). Applied as
    /// `clamp(silu(gate), -inf, L) * clamp(up, -L, L)`. A value at or below
    /// 1e-6 disables it, matching the reference's `limit > eps` guard.
    pub swiglu_clamp_exp: Vec<f32>,
    /// Same, for the shared expert — and, because the reference's `build_ffn`
    /// reads this array for every call it makes, also for the leading dense
    /// blocks' FFN.
    pub swiglu_clamp_shexp: Vec<f32>,

    /// LayerNorm epsilon for the indexer's `k_norm`. The reference hardcodes
    /// `1e-6`; without the key it would silently run at 0.
    pub norm_eps: f32,
    pub rms_eps: f32,
}

impl Glm5NextConfig {
    /// `kda.head_dim * n_head` — the KDA q/k/v width.
    pub fn d_inner(&self, n_head: usize) -> usize { self.kda_head_dim * n_head }
    /// `hc_count * n_embd` — one token's full hyper-connection stream.
    pub fn hc_dim(&self, n_embd: usize) -> usize { self.hc_count * n_embd }
    /// `(2 + hc_count) * hc_count` — pre, post and the flattened mixing matrix.
    pub fn hc_mix(&self) -> usize { (2 + self.hc_count) * self.hc_count }

    /// The indexer's selection width, `top_k + kpool - 1`. Below this many
    /// cells the reference skips scoring entirely and attends densely.
    pub fn n_select(&self) -> usize { self.indexer_top_k + self.indexer_kpool - 1 }

    pub fn from_gguf(g: &GgufFile, cfg: &ModelConfig) -> Result<Self> {
        let ns = "glm5next";
        let u = |k: &str| -> Result<u64> {
            g.get_u64(&format!("{ns}.{k}"))
                .map_err(|_| LlamaError::Config(format!("glm5next: missing {ns}.{k}")))
        };
        let uo = |k: &str, d: u64| g.get_u64(&format!("{ns}.{k}")).unwrap_or(d);
        let f = |k: &str| -> Result<f32> {
            g.get_f32(&format!("{ns}.{k}"))
                .map_err(|_| LlamaError::Config(format!("glm5next: missing {ns}.{k}")))
        };

        let n_layer_all = cfg.n_layers;
        let n_layer_nextn = uo("nextn_predict_layers", 0) as usize;
        if n_layer_nextn >= n_layer_all {
            return Err(LlamaError::Config(format!(
                "glm5next: nextn_predict_layers ({n_layer_nextn}) must be < block_count ({n_layer_all})"
            )));
        }
        let n_layer = n_layer_all - n_layer_nextn;

        // The per-layer kind comes from the head_count_kv array that
        // ModelConfig decoded. A scalar there means this is not a glm5next GGUF
        // we can lay out, so fail loudly rather than guess an interval.
        let mask = cfg.recurrent_layers.as_ref().ok_or_else(|| {
            LlamaError::Config(
                "glm5next: attention.head_count_kv must be a per-layer array (0 = KDA, \
                 1 = full MLA); this GGUF stores a scalar"
                    .into(),
            )
        })?;
        if mask.len() != n_layer_all {
            return Err(LlamaError::Config(format!(
                "glm5next: attention.head_count_kv has {} entries, block_count is {n_layer_all}",
                mask.len()
            )));
        }
        let layer_kinds: Vec<LayerKind> = mask
            .iter()
            .map(|&recr| if recr { LayerKind::Kda } else { LayerKind::Mla })
            .collect();
        if !layer_kinds[..n_layer].contains(&LayerKind::Mla) {
            return Err(LlamaError::Config(
                "glm5next: the trunk has no full-attention layer".into(),
            ));
        }
        if !layer_kinds[..n_layer].contains(&LayerKind::Kda) {
            return Err(LlamaError::Config(
                "glm5next: the trunk has no KDA layer; this is not a hybrid model".into(),
            ));
        }

        let indexer_top_k = u("attention.indexer.top_k")? as usize;
        let indexer_kpool = u("attention.indexer.kpool")? as usize;
        if indexer_kpool == 0 || indexer_top_k < indexer_kpool || indexer_top_k % indexer_kpool != 0
        {
            return Err(LlamaError::Config(format!(
                "glm5next: indexer top_k ({indexer_top_k}) must be a non-zero multiple of \
                 kpool ({indexer_kpool})"
            )));
        }

        let kda_gate_lower_bound = f("kda.gate_lower_bound")?;
        if kda_gate_lower_bound >= 0.0 {
            // Absent or non-negative, kimi-k3 selects a softplus branch instead —
            // a different function, silently wrong output rather than an error.
            return Err(LlamaError::Config(format!(
                "glm5next: kda.gate_lower_bound must be negative, got {kda_gate_lower_bound}"
            )));
        }

        let hc_count = u("hyper_connection.count")? as usize;
        if hc_count == 0 {
            return Err(LlamaError::Config(
                "glm5next: hyper_connection.count is 0".into(),
            ));
        }

        let n_expert = u("expert_count")? as usize;
        let n_expert_used = u("expert_used_count")? as usize;
        if n_expert_used == 0 || n_expert_used > n_expert {
            return Err(LlamaError::Config(format!(
                "glm5next: expert_used_count ({n_expert_used}) outside 1..={n_expert}"
            )));
        }
        let n_expert_shared = uo("expert_shared_count", 1) as usize;
        let n_ff_exp = u("expert_feed_forward_length")? as usize;
        let n_ff_shexp = match uo("expert_shared_feed_forward_length", 0) as usize {
            0 => n_ff_exp * n_expert_shared.max(1),
            v => v,
        };

        let rope_dim = uo("rope.dimension_count", 0) as usize;
        if rope_dim != 0 {
            return Err(LlamaError::Config(format!(
                "glm5next is NoPE-only but rope.dimension_count is {rope_dim}"
            )));
        }

        let norm_eps = g
            .get_f32(&format!("{ns}.attention.layer_norm_epsilon"))
            .unwrap_or(1e-6);

        // Stored as a per-layer array; a scalar or a missing key broadcasts.
        let clamp_arr = |k: &str| -> Vec<f32> {
            let key = format!("{ns}.{k}");
            if let Some(gguf::Array::F32(v)) =
                g.metadata().get(&key).and_then(|v| v.as_array())
            {
                if v.len() == n_layer_all {
                    return v.clone();
                }
            }
            let scalar = g.get_f32(&key).unwrap_or(0.0);
            vec![scalar; n_layer_all]
        };

        Ok(Self {
            n_layer,
            n_layer_all,
            n_layer_nextn,
            layer_kinds,
            n_dense_lead: uo("leading_dense_block_count", 0) as usize,

            kda_head_dim: u("kda.head_dim")? as usize,
            kda_gate_lower_bound,
            ssm_conv_kernel: u("ssm.conv_kernel")? as usize,

            q_lora_rank: u("attention.q_lora_rank")? as usize,
            kv_lora_rank: u("attention.kv_lora_rank")? as usize,
            qk_head_dim: u("attention.key_length_mla")? as usize,
            v_head_dim: u("attention.value_length_mla")? as usize,

            indexer_n_head: u("attention.indexer.head_count")? as usize,
            indexer_head_dim: u("attention.indexer.key_length")? as usize,
            indexer_top_k,
            indexer_kpool,

            hc_count,
            hc_sinkhorn_iters: uo("hyper_connection.sinkhorn_iterations", 0) as usize,
            hc_eps: g
                .get_f32(&format!("{ns}.hyper_connection.epsilon"))
                .unwrap_or(1e-6),

            n_expert,
            n_expert_used,
            n_expert_shared,
            n_ff_exp,
            n_ff_shexp,
            expert_weights_scale: g
                .get_f32(&format!("{ns}.expert_weights_scale"))
                .unwrap_or(1.0),
            expert_weights_norm: g
                .get_bool(&format!("{ns}.expert_weights_norm"))
                .unwrap_or(false),
            expert_gating_func: uo("expert_gating_func", 0) as u32,
            swiglu_clamp_exp: clamp_arr("swiglu_clamp_exp"),
            swiglu_clamp_shexp: clamp_arr("swiglu_clamp_shexp"),

            norm_eps,
            rms_eps: cfg.rms_eps,
        })
    }
}

/// Hyper-connection mix parameters for one sublayer. Same layout as
/// `dsv41::hc::HcParams`, which this will be folded into when the forward pass
/// lands — `fn_` is `[hc_mix, hc_dim]` after the loader's dim reversal.
#[derive(Debug)]
pub struct HcParams {
    pub fn_: Weight,
    pub base: Tensor,
    pub scale: Tensor,
}

/// A trunk block's two hyper-connection mixers. `None` on the NextN block,
/// which uses a plain residual.
#[derive(Debug)]
pub struct HcPair {
    pub attn: HcParams,
    pub ffn: HcParams,
}

/// KDA linear-attention weights.
#[derive(Debug)]
pub struct KdaWeights {
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    /// Depthwise conv over the sequence, one per projection. The reference
    /// concatenates all three and runs a single conv so the state rolls back as
    /// one block.
    pub conv_q: Tensor,
    pub conv_k: Tensor,
    pub conv_v: Tensor,
    /// Low-rank per-channel forget gate: `[n_embd -> kda_head_dim -> d_inner]`.
    pub f_a: Weight,
    pub f_b: Weight,
    /// Low-rank output gate, same shape as the forget gate.
    pub g_a: Weight,
    pub g_b: Weight,
    /// Per-head write strength, sigmoid'd.
    pub beta: Weight,
    /// Holds `-exp(A_log)` (the kimi-k3 convention), not `+exp(A_log)`.
    pub a: Tensor,
    pub dt_bias: Tensor,
    pub o_norm: Tensor,
    pub output: Weight,
}

/// Sparse "lightning indexer" weights, present on every MLA layer.
#[derive(Debug)]
pub struct IndexerWeights {
    pub attn_k: Weight,
    pub attn_q_b: Weight,
    /// A LayerNorm **with bias**, unlike every RMSNorm elsewhere in this arch.
    pub k_norm: Tensor,
    pub k_norm_bias: Tensor,
    /// Per-head score weights; sign-unconstrained, and the reference forces F32
    /// accumulation here because bf16 swaps near-tied pools.
    pub proj: Weight,
    /// A second, independent projection cached beside the key — not a reuse of it.
    pub compressor_gate: Weight,
    /// Absolute positional encoding over the `kpool` slots, added pre-softmax.
    pub compressor_ape: Tensor,
}

/// Multi-head latent attention weights, absorbed form.
#[derive(Debug)]
pub struct MlaWeights {
    pub q_a: Weight,
    pub q_a_norm: Tensor,
    pub q_b: Weight,
    pub kv_a_mqa: Weight,
    pub kv_a_norm: Tensor,
    pub k_b: Weight,
    pub v_b: Weight,
    pub output: Weight,
    pub indexer: IndexerWeights,
}

#[derive(Debug)]
pub enum Glm5NextAttn {
    Kda(KdaWeights),
    Mla(MlaWeights),
}

/// Dense FFN on the leading blocks, MoE with a shared expert after.
#[derive(Debug)]
pub enum Glm5NextFfn {
    Dense {
        gate: Weight,
        up: Weight,
        down: Weight,
    },
    Moe {
        moe: MoeFfn,
        /// Biases top-k **selection** only; the routed weights stay unbiased.
        exp_probs_b: Tensor,
        shexp_gate: Weight,
        shexp_up: Weight,
        shexp_down: Weight,
    },
}

/// NextN / MTP draft-head weights, `blk.45` on the released model.
#[derive(Debug)]
pub struct NextNWeights {
    pub eh_proj: Weight,
    pub enorm: Tensor,
    pub hnorm: Tensor,
    pub shared_head_norm: Option<Tensor>,
}

#[derive(Debug)]
pub struct Glm5NextBlock {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    /// `None` on the NextN block only.
    pub hc: Option<HcPair>,
    pub attn: Glm5NextAttn,
    pub ffn: Glm5NextFfn,
    pub nextn: Option<NextNWeights>,
}

impl Glm5NextBlock {
    pub fn kind(&self) -> LayerKind {
        match self.attn {
            Glm5NextAttn::Kda(_) => LayerKind::Kda,
            Glm5NextAttn::Mla(_) => LayerKind::Mla,
        }
    }
}

pub struct Glm5NextModel {
    pub config: ModelConfig,
    pub glm: Glm5NextConfig,
    pub tokenizer: Tokenizer,
    /// `n_layer_all` entries: the trunk, then the NextN block if present.
    ///
    /// Empty on the streaming path, where the weights live in the device model
    /// instead — see [`Weights`].
    pub blocks: Vec<Glm5NextBlock>,
    pub tok_embd: Arc<Tensor>,
    pub output_norm: Tensor,
    pub output: Weight,
    pub backend: Arc<dyn Backend>,
    /// VENDORED-LOCAL: shared streaming-expert state when opened with
    /// `from_gguf_streaming`. `None` for the resident load.
    pub stream_shared: Option<Arc<crate::expert_stream::StreamShared>>,
    // VENDORED-LOCAL: GLM-5.3-Flash. The path that actually decodes.
    /// The optimised representation, when the model was opened streaming.
    ///
    /// Two representations existed side by side for a while and only one of them
    /// could run: `blocks` above expands every trunk tensor to f32 in host memory,
    /// which is 35.7 GB for the released model, and `device::DeviceModel` keeps
    /// them quantised on the backend at 5.97 GB with the experts streamed. Loading
    /// both would spend 35.7 GB of host memory that the expert cache needs — it
    /// wants 165 GB of the 190 — so the streaming constructor builds only this one
    /// and leaves `blocks` empty.
    decoder: Option<Decoder>,
}

// VENDORED-LOCAL: GLM-5.3-Flash.
/// The device model plus the recurrent state a sequence carries.
///
/// The state is a `Mutex` because [`Glm5NextModel::forward`] takes `&self` — the
/// signature every other architecture in this crate uses, where all the state a
/// forward mutates arrives in the `KvCache` argument. glm5next's state does not
/// fit that: a `KvCache` has per-layer K and V plus the two SSM slots, and this
/// architecture also needs the MLA latent cache and the sparse indexer's pooled
/// keys and gates, which have no home there. Rather than widen a type every other
/// architecture shares, the state lives here and `forward` documents that it
/// ignores the argument.
struct Decoder {
    model: Box<device::DeviceModel>,
    state: std::sync::Mutex<forward::State>,
}

impl std::fmt::Debug for Glm5NextModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kda, mla) = self.layer_census();
        f.debug_struct("Glm5NextModel")
            .field("n_layer", &self.glm.n_layer)
            .field("kda_layers", &kda)
            .field("mla_layers", &mla)
            .field("n_expert", &self.glm.n_expert)
            .field("n_expert_used", &self.glm.n_expert_used)
            .field("streamed", &self.stream_shared.is_some())
            .field("backend", &self.backend.name())
            .finish()
    }
}

/// Reject a tensor whose shape is not what the metadata implies, naming both.
/// The loader's dim reversal means `want` is written output-dim first.
fn want_shape(name: &str, got: &[usize], want: &[usize]) -> Result<()> {
    if got != want {
        return Err(LlamaError::Config(format!(
            "glm5next: {name} has shape {got:?}, expected {want:?}"
        )));
    }
    Ok(())
}

impl Glm5NextModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::from_gguf_impl(g, backend, None)
    }

    /// VENDORED-LOCAL: streaming variant — expert tensors stay in the `.gguf`
    /// behind the bounded cache and each MoE block gets a `LayerStream`.
    pub fn from_gguf_streaming(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        shared: Arc<crate::expert_stream::StreamShared>,
    ) -> Result<Self> {
        Self::from_gguf_streaming_with(g, backend, shared, Self::DEFAULT_CONTEXT)
    }

    /// Context length the streaming path allocates state for when the caller does
    /// not say. The MLA latent cache is `max_len * kv_lora` per MLA layer — 2 MB a
    /// layer per 1 000 tokens here — so this is cheap to raise and not free.
    pub const DEFAULT_CONTEXT: usize = 8192;

    // VENDORED-LOCAL: GLM-5.3-Flash. The constructor that can decode.
    /// As [`Self::from_gguf_streaming`], with an explicit context length.
    ///
    /// Builds the device model rather than expanding the trunk to f32, so
    /// [`Self::forward`] works. `blocks` is left empty: see [`Self::blocks`].
    pub fn from_gguf_streaming_with(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        shared: Arc<crate::expert_stream::StreamShared>,
        max_len: usize,
    ) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let glm = Glm5NextConfig::from_gguf(g, &config)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let max_len = max_len.clamp(1, config.context_length.max(1));

        let model = device::DeviceModel::from_gguf(
            g,
            max_len,
            Arc::clone(&backend),
            shared.cache_budget_bytes(),
        )?;
        let state = forward::State::new_on(model.shape(), Arc::clone(&backend))?;

        let idx = TensorIndex::new(g);
        let tok_embd = Arc::new(idx.take("token_embd.weight", &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        Ok(Self {
            config,
            glm,
            tokenizer,
            blocks: Vec::new(),
            tok_embd,
            output_norm,
            output,
            backend,
            stream_shared: Some(shared),
            decoder: Some(Decoder {
                model: Box::new(model),
                state: std::sync::Mutex::new(state),
            }),
        })
    }

    fn from_gguf_impl(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        stream_shared: Option<Arc<crate::expert_stream::StreamShared>>,
    ) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let glm = Glm5NextConfig::from_gguf(g, &config)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        let n_embd = config.embedding_dim;
        let n_head = config.n_heads;
        let d_inner = glm.d_inner(n_head);
        let hd = glm.kda_head_dim;
        let d_conv = glm.ssm_conv_kernel;
        let qr = glm.q_lora_rank;
        let kvr = glm.kv_lora_rank;
        let d_idx = glm.indexer_head_dim;

        let tok_embd = Arc::new(idx.take("token_embd.weight", &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        const M: usize = 2 * 1024 * 1024 * 1024;
        let mut blocks = Vec::with_capacity(glm.n_layer_all);

        for i in 0..glm.n_layer_all {
            let is_nextn = i >= glm.n_layer;

            let attn_norm = idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?;
            let ffn_norm = idx.take(&format!("blk.{i}.ffn_norm.weight"), &[])?;

            // The NextN block has a plain residual, so it carries no mixer. Every
            // trunk block must have one: a missing hc_* there is a truncated file,
            // not an optional feature.
            let hc = if is_nextn {
                None
            } else {
                let load_hc = |which: &str| -> Result<HcParams> {
                    let fn_ = idx.take_weight(&format!("blk.{i}.hc_{which}_fn.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.hc_{which}_fn.weight"),
                        fn_.shape(),
                        &[glm.hc_mix(), glm.hc_dim(n_embd)],
                    )?;
                    Ok(HcParams {
                        fn_,
                        base: idx.take(&format!("blk.{i}.hc_{which}_base.weight"), &[])?,
                        scale: idx.take(&format!("blk.{i}.hc_{which}_scale.weight"), &[])?,
                    })
                };
                Some(HcPair {
                    attn: load_hc("attn")?,
                    ffn: load_hc("ffn")?,
                })
            };

            let attn = match glm.layer_kinds[i] {
                LayerKind::Kda => {
                    let q = idx.take_weight(&format!("blk.{i}.attn_q.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.attn_q.weight"),
                        q.shape(),
                        &[d_inner, n_embd],
                    )?;
                    let f_b = idx.take_weight(&format!("blk.{i}.ssm_f_b.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.ssm_f_b.weight"),
                        f_b.shape(),
                        &[d_inner, hd],
                    )?;
                    let conv_q = idx.take(&format!("blk.{i}.ssm_conv1d_q.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.ssm_conv1d_q.weight"),
                        conv_q.shape(),
                        &[d_inner, 1, d_conv],
                    )?;
                    Glm5NextAttn::Kda(KdaWeights {
                        q,
                        k: idx.take_weight(&format!("blk.{i}.attn_k.weight"), &[])?,
                        v: idx.take_weight(&format!("blk.{i}.attn_v.weight"), &[])?,
                        conv_q,
                        conv_k: idx.take(&format!("blk.{i}.ssm_conv1d_k.weight"), &[])?,
                        conv_v: idx.take(&format!("blk.{i}.ssm_conv1d_v.weight"), &[])?,
                        f_a: idx.take_weight(&format!("blk.{i}.ssm_f_a.weight"), &[])?,
                        f_b,
                        g_a: idx.take_weight(&format!("blk.{i}.ssm_g_a.weight"), &[])?,
                        g_b: idx.take_weight(&format!("blk.{i}.ssm_g_b.weight"), &[])?,
                        beta: idx.take_weight(&format!("blk.{i}.ssm_beta.weight"), &[])?,
                        a: idx.take(&format!("blk.{i}.ssm_a"), &[])?,
                        dt_bias: idx.take(&format!("blk.{i}.ssm_dt.bias"), &[])?,
                        o_norm: idx.take(&format!("blk.{i}.ssm_norm.weight"), &[])?,
                        output: {
                            // KDA's attn_output is [n_embd, d_inner] after the
                            // dim reversal, against MLA's [n_embd, n_head*v_head]
                            // — the one tensor whose shape tells the two layer
                            // kinds apart, so it is worth asserting.
                            let w = idx
                                .take_weight(&format!("blk.{i}.attn_output.weight"), &[])?;
                            want_shape(
                                &format!("blk.{i}.attn_output.weight"),
                                w.shape(),
                                &[n_embd, d_inner],
                            )?;
                            w
                        },
                    })
                }
                LayerKind::Mla => {
                    let q_b = idx.take_weight(&format!("blk.{i}.attn_q_b.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.attn_q_b.weight"),
                        q_b.shape(),
                        &[n_head * glm.qk_head_dim, qr],
                    )?;
                    let k_b = idx.take_weight(&format!("blk.{i}.attn_k_b.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.attn_k_b.weight"),
                        k_b.shape(),
                        &[n_head, kvr, glm.qk_head_dim],
                    )?;
                    let v_b = idx.take_weight(&format!("blk.{i}.attn_v_b.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.attn_v_b.weight"),
                        v_b.shape(),
                        &[n_head, glm.v_head_dim, kvr],
                    )?;
                    let output = idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.attn_output.weight"),
                        output.shape(),
                        &[n_embd, n_head * glm.v_head_dim],
                    )?;

                    let ape = idx.take(&format!("blk.{i}.indexer_compressor_ape.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.indexer_compressor_ape.weight"),
                        ape.shape(),
                        &[glm.indexer_kpool, d_idx],
                    )?;
                    let iq_b = idx.take_weight(&format!("blk.{i}.indexer.attn_q_b.weight"), &[])?;
                    want_shape(
                        &format!("blk.{i}.indexer.attn_q_b.weight"),
                        iq_b.shape(),
                        &[glm.indexer_n_head * d_idx, qr],
                    )?;

                    let indexer = IndexerWeights {
                        attn_k: idx.take_weight(&format!("blk.{i}.indexer.attn_k.weight"), &[])?,
                        attn_q_b: iq_b,
                        k_norm: idx.take(&format!("blk.{i}.indexer.k_norm.weight"), &[])?,
                        // The reference asserts this bias exists: the indexer
                        // k_norm is a LayerNorm, not the RMSNorm used elsewhere.
                        k_norm_bias: idx
                            .take(&format!("blk.{i}.indexer.k_norm.bias"), &[])
                            .map_err(|_| {
                                LlamaError::Config(format!(
                                    "glm5next: blk.{i}.indexer.k_norm.bias is missing; the \
                                     indexer k_norm is a LayerNorm with bias"
                                ))
                            })?,
                        proj: idx.take_weight(&format!("blk.{i}.indexer.proj.weight"), &[])?,
                        compressor_gate: idx.take_weight(
                            &format!("blk.{i}.indexer_compressor_gate.weight"),
                            &[],
                        )?,
                        compressor_ape: ape,
                    };

                    Glm5NextAttn::Mla(MlaWeights {
                        q_a: idx.take_weight(&format!("blk.{i}.attn_q_a.weight"), &[])?,
                        q_a_norm: idx.take(&format!("blk.{i}.attn_q_a_norm.weight"), &[])?,
                        q_b,
                        kv_a_mqa: idx.take_weight(&format!("blk.{i}.attn_kv_a_mqa.weight"), &[])?,
                        kv_a_norm: idx.take(&format!("blk.{i}.attn_kv_a_norm.weight"), &[])?,
                        k_b,
                        v_b,
                        output,
                        indexer,
                    })
                }
            };

            let ffn = if i < glm.n_dense_lead {
                Glm5NextFfn::Dense {
                    gate: idx.take_weight(&format!("blk.{i}.ffn_gate.weight"), &[])?,
                    up: idx.take_weight(&format!("blk.{i}.ffn_up.weight"), &[])?,
                    down: idx.take_weight(&format!("blk.{i}.ffn_down.weight"), &[])?,
                }
            } else {
                let router = idx.take_weight(&format!("blk.{i}.ffn_gate_inp.weight"), &[])?;
                want_shape(
                    &format!("blk.{i}.ffn_gate_inp.weight"),
                    router.shape(),
                    &[glm.n_expert, n_embd],
                )?;

                // VENDORED-LOCAL: streaming mode leaves the experts on disk. The
                // per-layer stream handle is indexed by MoE layer, not by block:
                // blocks 0..n_dense_lead have no experts at all.
                let moe = match &stream_shared {
                    Some(shared) => MoeFfn {
                        router,
                        gate_up_experts: Vec::new(),
                        down_experts: Vec::new(),
                        top_k: glm.n_expert_used,
                        stream: Some(shared.layer((i - glm.n_dense_lead) as u32)),
                    }
                    .move_to_device(&*backend, M),
                    None => {
                        let (gate_up_experts, down_experts) =
                            crate::qwen3moe::load_per_layer_experts_pair(&idx, i, glm.n_expert)?;
                        MoeFfn {
                            router,
                            gate_up_experts,
                            down_experts,
                            top_k: glm.n_expert_used,
                            stream: None,
                        }
                        .move_to_device(&*backend, M)
                    }
                };

                Glm5NextFfn::Moe {
                    moe,
                    exp_probs_b: idx.take(&format!("blk.{i}.exp_probs_b.bias"), &[])?,
                    shexp_gate: idx.take_weight(&format!("blk.{i}.ffn_gate_shexp.weight"), &[])?,
                    shexp_up: idx.take_weight(&format!("blk.{i}.ffn_up_shexp.weight"), &[])?,
                    shexp_down: idx.take_weight(&format!("blk.{i}.ffn_down_shexp.weight"), &[])?,
                }
            };

            let nextn = if is_nextn {
                Some(NextNWeights {
                    eh_proj: idx.take_weight(&format!("blk.{i}.nextn.eh_proj.weight"), &[])?,
                    enorm: idx.take(&format!("blk.{i}.nextn.enorm.weight"), &[])?,
                    hnorm: idx.take(&format!("blk.{i}.nextn.hnorm.weight"), &[])?,
                    shared_head_norm: idx
                        .try_take(&format!("blk.{i}.nextn.shared_head_norm.weight"))
                        .transpose()?,
                })
            } else {
                None
            };

            blocks.push(Glm5NextBlock {
                attn_norm,
                ffn_norm,
                hc,
                attn,
                ffn,
                nextn,
            });
        }

        Ok(Self {
            config,
            glm,
            tokenizer,
            blocks,
            tok_embd,
            output_norm,
            output,
            backend,
            stream_shared,
            // The resident representation cannot decode: see `Weights` on the
            // struct, and `forward` below.
            decoder: None,
        })
    }

    /// How many trunk layers run each attention kind — cheap sanity readout for
    /// `OAIY info`, and what the streamed-vs-resident tests assert on.
    pub fn layer_census(&self) -> (usize, usize) {
        // From the layer map rather than from `blocks`, which the streaming path
        // leaves empty.
        let kda = self.glm.layer_kinds[..self.glm.n_layer]
            .iter()
            .filter(|k| **k == LayerKind::Kda)
            .count();
        (kda, self.glm.n_layer - kda)
    }

    /// The VRAM each card's expert cache holds, in bytes, and how many MoE layers
    /// each card runs — in card order, for a status line.
    ///
    /// Before [`Self::enable_tiering`] the budgets are empty and every MoE layer
    /// belongs to the one backend the model was opened on.
    pub fn tier_layout(&self) -> (Vec<usize>, Vec<usize>) {
        let Some(d) = &self.decoder else { return (Vec::new(), Vec::new()) };
        let (card_of, n_cards) = d.model.experts().card_layout();
        let mut per = vec![0usize; n_cards];
        for &c in card_of {
            if let Some(slot) = per.get_mut(c) {
                *slot += 1;
            }
        }
        (d.model.experts().vram_budgets().to_vec(), per)
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. The prompt state, out and back.
    /// Copy out the state as it stands, for a caller keeping prompts on disk.
    ///
    /// Reading a prompt is the expensive part of a request and a harness sends the
    /// same one every time; this is what lets a later process start where an earlier
    /// one finished. See [`forward::StateSnapshot`] for what it holds and why the
    /// caches are truncated to the tokens actually written.
    pub fn snapshot_state(&self) -> Result<forward::StateSnapshot> {
        let d = self.decoder.as_ref().ok_or_else(|| {
            LlamaError::Config("glm5next: a resident model has no prompt state".into())
        })?;
        let st = d.state.lock().unwrap_or_else(|e| e.into_inner());
        st.snapshot(d.model.shape().kv_lora)
    }

    /// Put a state back. The next [`Self::forward`] continues from it.
    ///
    /// The caller is trusting that the snapshot came from these weights: a state
    /// from another model would decode and restore and then produce nonsense, which
    /// is why the cache that stores them keys every file to a fingerprint.
    pub fn restore_state(&self, snap: &forward::StateSnapshot) -> Result<()> {
        let d = self.decoder.as_ref().ok_or_else(|| {
            LlamaError::Config("glm5next: a resident model has no prompt state".into())
        })?;
        let mut st = d.state.lock().unwrap_or_else(|e| e.into_inner());
        st.restore(snap, d.model.shape().kv_lora)
    }

    /// Whether this model can run [`Self::forward`].
    ///
    /// True when opened with [`Self::from_gguf_streaming`]. The resident
    /// [`Self::from_gguf`] load holds the trunk as f32 for inspection and tests and
    /// has no decoder attached.
    pub fn can_decode(&self) -> bool {
        self.decoder.is_some()
    }

    /// How many tokens of state the decoder was built for.
    pub fn max_len(&self) -> usize {
        self.decoder
            .as_ref()
            .map(|d| d.model.shape().max_len)
            .unwrap_or(0)
    }

    /// Forget the sequence: the next [`Self::forward`] starts from position 0.
    pub fn reset(&self) {
        if let Some(d) = &self.decoder {
            d.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
        }
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. The generic entry point, wired to the
    // optimised path.
    /// Run `tokens` and return the last position's logits, `[1, n_vocab]`.
    ///
    /// **`kv` is not used, and that is deliberate.** Every other architecture here
    /// keeps all of a sequence's state in the `KvCache` it is handed, which works
    /// because their state is per-layer K and V. glm5next carries four other things:
    /// a `[n_head, head_dim, head_dim]` KDA recurrence per KDA layer, that layer's
    /// depthwise-conv ring, the MLA latent cache that serves as both K and V, and
    /// the sparse indexer's pooled keys and gates. Only the first two have anywhere
    /// to live in a `KvCache`. So the state belongs to the model, `reset` clears it,
    /// and the argument is accepted to keep one signature across the enum in
    /// `lib.rs` rather than quietly writing into something that cannot hold it.
    ///
    /// Requires a model opened with [`Self::from_gguf_streaming`]; the resident load
    /// has no decoder. Positions continue from wherever the last call left off, so
    /// prefill is this in a loop and decode is this one token at a time.
    pub fn forward(&self, tokens: &[u32], kv: &mut crate::kv_cache::KvCache) -> Result<Tensor> {
        let Some(d) = &self.decoder else {
            let (kda, mla) = self.layer_census();
            return Err(LlamaError::Config(format!(
                "glm5next: this model was opened resident ({kda} KDA + {mla} MLA layers, \
                 the trunk expanded to f32) and has no decoder. Open it with \
                 `Model::open_streaming` — the released checkpoint is 193 GB and the \
                 routed experts would be about 1.2 TB expanded, so streaming is the \
                 only way it fits."
            )));
        };
        if tokens.is_empty() {
            return Err(LlamaError::Config("glm5next: forward got no tokens".into()));
        }

        let sh = d.model.shape();
        let w = d.model.view();
        let mut st = d.state.lock().unwrap_or_else(|e| e.into_inner());

        // One thing does cross from the argument: a caller that has reset the cache
        // is starting a new sequence, and `oaiy-llm-cli` does exactly that between turns.
        // Mirroring it here means the generic contract works without the caller
        // knowing this architecture keeps its own state.
        if kv.len == 0 && st.len != 0 {
            st.reset();
        }
        if st.len + tokens.len() > sh.max_len {
            return Err(LlamaError::Config(format!(
                "glm5next: {} tokens at position {} exceeds the {}-token state this                  model was opened with",
                tokens.len(),
                st.len,
                sh.max_len
            )));
        }

        // VENDORED-LOCAL: GLM-5.3-Flash. A prompt goes through in chunks.
        //
        // One token at a time, a prompt costs what a generated token costs -- about
        // 100 ms here -- because each one reads all eight routed experts of all 42
        // MoE layers for itself. A coder-cli system prompt of a few thousand tokens
        // is then minutes before the first reply, which is what this fixes:
        // `forward_chunk` walks the layers once for a chunk and its tokens share
        // each expert read. The attention inside still runs token by token, in
        // order, because KDA is a recurrence.
        //
        // A single token takes the one-token path: it has nothing to share, and that
        // path overlaps the CPU tier with the GPU and runs a route through one
        // grouped kernel. GLM5_PREFILL_CHUNK=1 makes every prompt take it too.
        let mut logits = Vec::new();
        if tokens.len() == 1 {
            logits = forward::forward_token(sh, &w, &mut st, tokens[0])?;
        } else {
            for part in tokens.chunks(forward::prefill_chunk()) {
                logits = forward::forward_chunk(sh, &w, &mut st, part)?;
            }
        }
        // And report the position back, so a caller watching `kv.len` for its context
        // limit sees the truth rather than a cache that never fills.
        kv.len = st.len;
        Ok(Tensor::from_vec(logits, vec![1, sh.n_vocab]))
    }
}

// VENDORED-LOCAL: real-model test paths come from the environment, so the
// defaults here need not match any one machine's file names.
/// Where the real-model tests find the released weights: `GLM5_GGUF` (the
/// first shard) and `GLM5_MMPROJ`, or these defaults.
#[cfg(test)]
pub(crate) mod test_paths {
    use std::sync::OnceLock;

    fn var(slot: &'static OnceLock<String>, name: &str, default: &str) -> &'static str {
        slot.get_or_init(|| std::env::var(name).unwrap_or_else(|_| default.to_string()))
    }

    pub fn released() -> &'static str {
        static PATH: OnceLock<String> = OnceLock::new();
        var(&PATH, "GLM5_GGUF", r"D:\glm5.3_flash\Q4_K_M\GLM-5.3-Flash-Q4_K_M-00001-of-00005.gguf")
    }

    pub fn mmproj() -> &'static str {
        static PATH: OnceLock<String> = OnceLock::new();
        var(&PATH, "GLM5_MMPROJ", r"D:\glm5.3_flash\mmproj-GLM-5.3-Flash-F16.gguf")
    }
}

#[cfg(test)]
mod tests {
    //! Metadata-only fixtures built from the **real** GLM-5.3-Flash header
    //! (`GLM-5.3-Flash-GGUF`, Q4_K_M, GGUF v3). Every
    //! value below is what that file actually carries, so these tests pin the
    //! layer map and the derived geometry against the shipped model rather than
    //! against numbers this module invented. No tensors are needed: both config
    //! parsers read metadata only.

    use std::collections::BTreeMap;

    use gguf::{Array, GgufFile, Value};

    use super::*;
    use crate::config::{Architecture, ModelConfig};

    /// Blocks whose `head_count_kv` entry is 1 in the released weights. 3, 7,
    /// ... 43 are the trunk's full-attention layers; 45 is the NextN block. Note
    /// that 43 is followed by 45, not 47: the stride breaks, which is why the
    /// loader reads the array instead of applying a `% 4` rule.
    const MLA_BLOCKS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45];
    const BLOCK_COUNT: usize = 46;

    fn real_metadata() -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        m.insert(
            "general.architecture".to_string(),
            Value::String("glm5next".to_string()),
        );

        for (k, v) in [
            ("vocab_size", 154880u32),
            ("context_length", 1048576),
            ("embedding_length", 4096),
            ("block_count", BLOCK_COUNT as u32),
            ("feed_forward_length", 12288),
            ("attention.head_count", 64),
            ("nextn_predict_layers", 1),
            ("leading_dense_block_count", 3),
            ("kda.head_dim", 128),
            ("ssm.conv_kernel", 4),
            ("attention.q_lora_rank", 1536),
            ("attention.kv_lora_rank", 512),
            ("attention.key_length_mla", 256),
            ("attention.value_length_mla", 256),
            ("attention.indexer.head_count", 32),
            ("attention.indexer.key_length", 128),
            ("attention.indexer.top_k", 2048),
            ("attention.indexer.kpool", 4),
            ("hyper_connection.count", 4),
            ("hyper_connection.sinkhorn_iterations", 20),
            ("expert_count", 288),
            ("expert_used_count", 8),
            ("expert_shared_count", 1),
            ("expert_feed_forward_length", 2048),
            ("expert_shared_feed_forward_length", 2048),
            ("expert_gating_func", 2),
            ("rope.dimension_count", 0),
        ] {
            m.insert(format!("glm5next.{k}"), Value::U32(v));
        }

        for (k, v) in [
            ("attention.layer_norm_rms_epsilon", 1e-5f32),
            ("attention.layer_norm_epsilon", 1e-6),
            ("hyper_connection.epsilon", 1e-6),
            ("kda.gate_lower_bound", -5.0),
            ("expert_weights_scale", 2.5),
        ] {
            m.insert(format!("glm5next.{k}"), Value::F32(v));
        }

        m.insert(
            "glm5next.expert_weights_norm".to_string(),
            Value::Bool(true),
        );

        let kv: Vec<u32> = (0..BLOCK_COUNT)
            .map(|i| u32::from(MLA_BLOCKS.contains(&i)))
            .collect();
        m.insert(
            "glm5next.attention.head_count_kv".to_string(),
            Value::Array(Array::U32(kv)),
        );

        m
    }

    fn file_from(m: &BTreeMap<String, Value>) -> GgufFile {
        let raw = gguf::reader::write_to_vec(m, &[], 32).expect("write gguf");
        GgufFile::from_bytes(raw).expect("parse gguf")
    }

    // The released model, on the faster drive. Gated like the repo's other
    // real-model tests:
    // `cargo test -p llama-rs glm5next::tests::released -- --ignored --nocapture`
    use crate::glm5next::test_paths::released;

    /// The first time any of this meets a real tensor: the split reader, the
    /// metadata parse, the layer map and the shape assertions.
    #[test]
    #[ignore = "needs the released 194 GB model on disk"]
    fn released_split_model_opens_and_parses() {
        let g = GgufFile::open_streaming(released()).expect("open the split model");

        // The split reader must present all five shards as one file.
        assert_eq!(g.n_shards(), 5, "five shards");
        assert_eq!(g.tensors().len(), 1412, "split.tensors.count");
        assert_eq!(g.architecture().expect("arch"), "glm5next");

        let cfg = ModelConfig::from_gguf(&g).expect("model config");
        assert_eq!(cfg.arch, Architecture::Glm5Next);
        assert_eq!(cfg.n_layers, 46);
        assert_eq!(cfg.embedding_dim, 4096);
        assert_eq!(cfg.vocab_size, 154880);
        assert_eq!(cfg.context_length, 1_048_576);
        // The array parse, on the real array.
        assert_eq!(cfg.n_kv_heads, 1, "absorbed MLA is MQA over one latent row");
        let mask = cfg.recurrent_layers.clone().expect("per-layer mask");
        assert_eq!(mask.len(), 46);

        let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm config");
        assert_eq!(glm.n_layer, 45, "the trunk excludes the NextN block");
        assert_eq!(glm.n_layer_nextn, 1);
        assert_eq!(glm.n_dense_lead, 3);
        assert_eq!(glm.n_expert, 288);
        assert_eq!(glm.n_expert_used, 8);
        assert_eq!(glm.n_ff_exp, 2048);
        assert_eq!(glm.kda_head_dim, 128);
        assert_eq!(glm.indexer_kpool, 4);
        assert_eq!(glm.n_select(), 2051);
        assert_eq!(glm.expert_gating_func, 2);
        assert_eq!(glm.swiglu_clamp_exp.len(), 46);
        assert_eq!(glm.swiglu_clamp_exp[0], 10.0);

        let kda = glm.layer_kinds[..glm.n_layer]
            .iter()
            .filter(|k| **k == LayerKind::Kda)
            .count();
        assert_eq!((kda, glm.n_layer - kda), (34, 11), "34 KDA + 11 MLA trunk");
        assert_eq!(glm.layer_kinds[45], LayerKind::Mla, "the MTP block is MLA-shaped");

        println!(
            "glm5next: {} shards, {} tensors, trunk {} ({} KDA + {} MLA), {} experts top-{}",
            g.n_shards(),
            g.tensors().len(),
            glm.n_layer,
            kda,
            glm.n_layer - kda,
            glm.n_expert,
            glm.n_expert_used
        );

        // Tensors must resolve ACROSS shards, with the shapes the loader asserts.
        // blk.0 is in shard 1; blk.45 (NextN) is in the last shard.
        let n_embd = cfg.embedding_dim;
        let d_inner = glm.kda_head_dim * cfg.n_heads;
        for (name, want_shape) in [
            ("token_embd.weight", vec![cfg.vocab_size, n_embd]),
            ("output_norm.weight", vec![n_embd]),
            ("blk.0.attn_q.weight", vec![d_inner, n_embd]),
            ("blk.0.ssm_conv1d_q.weight", vec![d_inner, 1, glm.ssm_conv_kernel]),
            ("blk.3.attn_k_b.weight", vec![cfg.n_heads, glm.kv_lora_rank, glm.qk_head_dim]),
            ("blk.3.indexer_compressor_ape.weight", vec![glm.indexer_kpool, glm.indexer_head_dim]),
            ("blk.0.hc_attn_fn.weight", vec![glm.hc_mix(), glm.hc_dim(n_embd)]),
            ("blk.44.ffn_gate_exps.weight", vec![glm.n_expert, glm.n_ff_exp, n_embd]),
            ("blk.45.nextn.eh_proj.weight", vec![n_embd, 2 * n_embd]),
        ] {
            let t = g
                .tensor_by_name(name)
                .unwrap_or_else(|| panic!("{name} did not resolve across the shards"));
            let got: Vec<usize> = t.shape.iter().map(|&d| d as usize).rev().collect();
            assert_eq!(got, want_shape, "{name} shape (shard {})", g.shard_of(t));
        }

        // hc_* must stop at the trunk, and the NextN block must have none.
        assert!(g.tensor_by_name("blk.44.hc_attn_fn.weight").is_some());
        assert!(
            g.tensor_by_name("blk.45.hc_attn_fn.weight").is_none(),
            "the NextN block has no mHC mixer"
        );
        // Tensors really are spread over the shards.
        let shards: std::collections::BTreeSet<usize> =
            g.tensors().iter().map(|t| g.shard_of(t)).collect();
        assert_eq!(shards.len(), 5, "tensors should occupy all five shards");
    }

    /// The whole streaming load: experts stay in the `.gguf` behind the bounded
    /// cache, everything else is resolved and shape-checked.
    #[test]
    #[ignore = "needs the released 194 GB model on disk"]
    fn released_split_model_loads_streaming() {
        use crate::Model;
        // Resident (non-expert) weights are ~25 GB; give the cache room on top.
        let budget: u64 = 64 << 30;
        let model = Model::open_streaming(released(), ggml_rs::default_backend(), budget)
            .expect("streaming load");

        let cfg = model.config();
        assert_eq!(cfg.arch, Architecture::Glm5Next);
        match &model {
            Model::Glm5Next(m) => {
                let (kda, mla) = m.layer_census();
                assert_eq!((kda, mla), (34, 11));
                assert_eq!(m.blocks.len(), 46, "trunk plus the NextN block");
                assert!(m.stream_shared.is_some(), "experts must be streamed");
                println!(
                    "loaded: {kda} KDA + {mla} MLA, {} blocks, backend {}",
                    m.blocks.len(),
                    m.backend.name()
                );
            }
            other => panic!("expected Model::Glm5Next, got {other:?}"),
        }
    }

    #[test]
    fn arch_string_maps_to_glm5next() {
        assert_eq!(Architecture::from_str("glm5next"), Architecture::Glm5Next);
        assert_eq!(Architecture::Glm5Next.name(), "glm5next");
        assert!(Architecture::Glm5Next.supported());
    }

    #[test]
    fn head_count_kv_array_decodes_to_the_layer_map() {
        let g = file_from(&real_metadata());
        let cfg = ModelConfig::from_gguf(&g).expect("model config");

        assert_eq!(cfg.arch, Architecture::Glm5Next);
        assert_eq!(cfg.n_layers, BLOCK_COUNT);
        assert_eq!(cfg.embedding_dim, 4096);
        assert_eq!(cfg.vocab_size, 154880);
        // Absorbed MLA is MQA over one latent row, so the array maximum is 1 --
        // NOT the 64 that the scalar fallback would have produced.
        assert_eq!(cfg.n_kv_heads, 1);

        let mask = cfg.recurrent_layers.expect("per-layer mask");
        assert_eq!(mask.len(), BLOCK_COUNT);
        for i in 0..BLOCK_COUNT {
            assert_eq!(mask[i], !MLA_BLOCKS.contains(&i), "block {i} recurrent flag");
        }
    }

    #[test]
    fn config_derives_the_trunk_and_geometry() {
        let g = file_from(&real_metadata());
        let cfg = ModelConfig::from_gguf(&g).expect("model config");
        let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm config");

        // 46 blocks, but blk.45 is the NextN draft block.
        assert_eq!(glm.n_layer_all, 46);
        assert_eq!(glm.n_layer_nextn, 1);
        assert_eq!(glm.n_layer, 45);
        assert_eq!(glm.n_dense_lead, 3);

        // 34 KDA + 11 MLA over the trunk; the 12th MLA-shaped block is MTP.
        let kda = glm.layer_kinds[..glm.n_layer]
            .iter()
            .filter(|&&k| k == LayerKind::Kda)
            .count();
        assert_eq!((kda, glm.n_layer - kda), (34, 11));
        assert_eq!(glm.layer_kinds[0], LayerKind::Kda);
        assert_eq!(glm.layer_kinds[3], LayerKind::Mla);
        assert_eq!(glm.layer_kinds[44], LayerKind::Kda);
        assert_eq!(glm.layer_kinds[45], LayerKind::Mla);

        // Derived geometry the tensor shape checks are written against.
        assert_eq!(glm.d_inner(cfg.n_heads), 8192);
        assert_eq!(glm.hc_dim(cfg.embedding_dim), 16384);
        assert_eq!(glm.hc_mix(), 24);
        // top_k + kpool - 1
        assert_eq!(glm.n_select(), 2051);

        assert_eq!(glm.n_expert, 288);
        assert_eq!(glm.n_expert_used, 8);
        assert_eq!(glm.n_ff_exp, 2048);
        assert_eq!(glm.n_ff_shexp, 2048);
        assert_eq!(glm.expert_gating_func, 2, "sigmoid gating");
        assert!(glm.expert_weights_norm);
        assert_eq!(glm.expert_weights_scale, 2.5);
        assert_eq!(glm.kda_gate_lower_bound, -5.0);
        assert_eq!(glm.indexer_kpool, 4);
    }

    /// A scalar `head_count_kv` cannot express the hybrid layout, so the loader
    /// must refuse rather than fall back to an interval rule.
    #[test]
    fn scalar_head_count_kv_is_rejected() {
        let mut m = real_metadata();
        m.insert(
            "glm5next.attention.head_count_kv".to_string(),
            Value::U32(1),
        );
        let g = file_from(&m);
        let cfg = ModelConfig::from_gguf(&g).expect("model config");
        assert!(cfg.recurrent_layers.is_none());
        assert!(Glm5NextConfig::from_gguf(&g, &cfg).is_err());
    }

    /// glm5next is NoPE. A non-zero rope dimension means the file is not the
    /// architecture we think it is.
    #[test]
    fn nonzero_rope_dimension_is_rejected() {
        let mut m = real_metadata();
        m.insert("glm5next.rope.dimension_count".to_string(), Value::U32(128));
        let g = file_from(&m);
        let cfg = ModelConfig::from_gguf(&g).expect("model config");
        assert!(Glm5NextConfig::from_gguf(&g, &cfg).is_err());
    }

    /// A non-negative KDA gate bound selects a different (softplus) branch in
    /// the reference, so it must not be accepted silently.
    #[test]
    fn nonnegative_kda_gate_bound_is_rejected() {
        let mut m = real_metadata();
        m.insert("glm5next.kda.gate_lower_bound".to_string(), Value::F32(0.0));
        let g = file_from(&m);
        let cfg = ModelConfig::from_gguf(&g).expect("model config");
        assert!(Glm5NextConfig::from_gguf(&g, &cfg).is_err());
    }

    /// gemma4 publishes `attention.head_count_kv` as an array too, but its
    /// entries are real per-layer KV head counts rather than a recurrent mask.
    /// Only glm5next may reinterpret them.
    #[test]
    fn other_arches_keep_their_scalar_kv_heads() {
        let mut m = BTreeMap::new();
        m.insert(
            "general.architecture".to_string(),
            Value::String("qwen3moe".to_string()),
        );
        for (k, v) in [
            ("vocab_size", 1000u32),
            ("context_length", 4096),
            ("embedding_length", 256),
            ("block_count", 4),
            ("feed_forward_length", 512),
            ("attention.head_count", 8),
            ("attention.head_count_kv", 4),
        ] {
            m.insert(format!("qwen3moe.{k}"), Value::U32(v));
        }
        let g = file_from(&m);
        let cfg = ModelConfig::from_gguf(&g).expect("model config");
        assert!(cfg.recurrent_layers.is_none());
        assert_eq!(cfg.n_kv_heads, 4, "scalar kv heads must survive untouched");
    }
}
