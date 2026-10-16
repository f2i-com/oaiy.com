//! The weights as the pass reads them: a view a stage (`HcW`, `KdaW`, `IndexerW`, `MlaW`, `MoeW`), a layer's and
//! the model's.

use super::*;

/// One sublayer's hyper-connection mixer.
pub struct HcW<'a> {
    /// `[hc_mix, hc * n_embd]`
    pub fn_: Mat<'a>,
    /// `[hc_mix]`
    pub base: &'a [f32],
    /// `[3]`
    pub scale: &'a [f32],
}

pub struct KdaW<'a> {
    /// `[d_inner, n_embd]` each.
    /// `attn_q` and `attn_k` -- the same dtype, so the device path fuses them.
    pub qk: Pair<'a>,
    pub v: Mat<'a>,
    /// `[d_inner, 1, d_conv]` each — depthwise, so row `c` is
    /// `conv[c*d_conv..(c+1)*d_conv]`.
    pub conv_q: &'a [f32],
    pub conv_k: &'a [f32],
    pub conv_v: &'a [f32],
    /// `[head_dim, n_embd]` then `[d_inner, head_dim]`: the low-rank decay.
    /// `ssm_f_a` and `ssm_g_a`: both Q8_0, both applied to the layer input.
    pub fga: Pair<'a>,
    pub f_b: Mat<'a>,
    pub g_b: Mat<'a>,
    /// `[n_head, n_embd]`
    pub beta: Mat<'a>,
    /// `[n_head]`, holding `-exp(A_log)`.
    pub a: &'a [f32],
    /// `[d_inner]`
    pub dt_bias: &'a [f32],
    /// `[head_dim]`
    pub o_norm: &'a [f32],
    /// `[n_embd, d_inner]`
    pub out: Mat<'a>,
}

pub struct IndexerW<'a> {
    /// `[d_idx, n_embd]`
    pub attn_k: Mat<'a>,
    /// `[n_ihead * d_idx, q_lora]`
    pub attn_q_b: Mat<'a>,
    /// `[d_idx]` each.
    pub k_norm: &'a [f32],
    pub k_norm_bias: &'a [f32],
    /// `[n_ihead, n_embd]`
    pub proj: Mat<'a>,
    /// `[d_idx, n_embd]`
    pub comp_gate: Mat<'a>,
    /// `[kpool, d_idx]`
    pub comp_ape: &'a [f32],
}

pub struct MlaW<'a> {
    /// `[q_lora, n_embd]`
    pub q_a: Mat<'a>,
    /// `[q_lora]`
    pub q_a_norm: &'a [f32],
    /// `[n_head * qk_head, q_lora]`
    pub q_b: Mat<'a>,
    /// `[kv_lora, n_embd]`
    pub kv_a_mqa: Mat<'a>,
    /// `[kv_lora]`
    pub kv_a_norm: &'a [f32],
    /// `[n_head, kv_lora, qk_head]`
    pub k_b: Bat<'a>,
    /// `[n_head, v_head, kv_lora]`
    pub v_b: Bat<'a>,
    /// `[n_embd, n_head * v_head]`
    pub out: Mat<'a>,
    pub indexer: IndexerW<'a>,
}

pub enum AttnW<'a> {
    Kda(KdaW<'a>),
    Mla(MlaW<'a>),
}

pub struct MoeW<'a> {
    /// `[n_expert, n_embd]`
    pub router: Mat<'a>,
    /// `[n_expert]`
    pub probs_b: &'a [f32],
    /// Routed experts, run per dispatch rather than borrowed.
    pub experts: &'a dyn ExpertFfn,
    /// This layer's MoE ordinal, for [`ExpertSource::expert`].
    pub ord: usize,
    /// `[n_ff_shexp, n_embd]`, `[n_ff_shexp, n_embd]`, `[n_embd, n_ff_shexp]`.
    /// The shared expert is always active, so it stays resident.
    /// `ffn_gate_shexp` and `ffn_up_shexp`, fused on the device path.
    pub sh_gate_up: Pair<'a>,
    pub sh_down: Mat<'a>,
}

pub enum FfnW<'a> {
    Dense {
        gate: Mat<'a>,
        up: Mat<'a>,
        down: Mat<'a>,
    },
    Moe(MoeW<'a>),
}

pub struct LayerW<'a> {
    /// `[n_embd]` each.
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub hc_attn: HcW<'a>,
    pub hc_ffn: HcW<'a>,
    pub attn: AttnW<'a>,
    pub ffn: FfnW<'a>,
}

pub struct ModelW<'a> {
    /// `[n_vocab, n_embd]`
    pub tok_embd: &'a [f32],
    /// `[n_embd]`
    pub output_norm: &'a [f32],
    /// `[n_vocab, n_embd]`
    pub output: Mat<'a>,
    /// Trunk layers only — `n_layer` of them, MTP excluded.
    pub layers: Vec<LayerW<'a>>,
}
