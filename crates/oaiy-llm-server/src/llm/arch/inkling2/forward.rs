//! Exact forward pass for Inkling-2 and Inkling-2-Loop.
//!
//! Matches Hugging Face `modeling_inkling2.py` / `modeling_inkling2_loop.py` and
//! llama.cpp `src/models/inkling2.cpp` / `src/models/inkling2-loop.cpp`:
//!   * Token embedding + RMSNorm (skipped for spliced vision soft tokens)
//!   * Block Attention Residuals (`AttnRes`: `mix(blocks, cur, query)`)
//!   * Cross-pass message bank + 4-way low-rank gated `control` (`Q, K, V, FFN`)
//!   * Hybrid SWA + Full Attention (NoPE): L2 clipping with learned scales, full-attn
//!     log-scale `tau(pos)`, dynamic relative-position bias table, per-query attention
//!     sinks, sigmoid output gate
//!   * Static & dynamic causal short convolutions (K, V, post-attn, post-FFN)
//!   * Top-k sigmoid-routed MoE with score correction bias + SiTU activation
//!   * Final AttnRes + RMSNorm + unembed + logit softcap + unpadded vocab slice

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ggml_rs::chain::{ChainRecipe, ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs::Tensor;
use rayon::prelude::*;

#[inline]
fn profile_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_INKLING2_PROFILE").is_some())
}

#[derive(Default)]
struct StepProfile {
    enabled: bool,
    attn_res_ctrl_us: u64,
    attn_proj_us: u64,
    attn_core_us: u64,
    attn_wo_us: u64,
    ffn_pre_us: u64,
    experts_us: u64,
    output_us: u64,
}

use crate::llm::loader::Weight;
use crate::llm::{LlamaError, Result};

use super::experts::{
    f32_axpy_fast, f32_dot_fast, f32_dot_rows_fast, gguf_dot_rows_any, quant_row_bytes, Bf16Linear,
};
use super::state::{
    Inkling2ConvSlotSnapshot, Inkling2ConvSnapshot, Inkling2SlotSnapshot, Inkling2State,
    Inkling2StateSnapshot,
};
use super::{AttnNorms, AttnNormsDev, Inkling2Layer, Inkling2LoopLayer, Inkling2Model};

/// Per-logical-slot GPU-resident packed KV ring cache (`[2, kv_cap, kv_dim]`: K then V) and
/// short-convolution histories (`kv_conv`: `[2, past, kv_dim]` K then V; plus post-attn & post-FFN).
#[derive(Debug)]
pub(crate) struct GpuSlotState {
    pub(crate) is_swa: bool,
    pub(crate) kv_dim: usize,
    pub(crate) kv_cap: usize,
    pub(crate) kv_cache: DeviceVec,
    pub(crate) kv_conv: DeviceVec,
    pub(crate) attn_post_conv: DeviceVec,
    pub(crate) ffn_post_conv: DeviceVec,
}

/// What logits [`Inkling2Model::forward_internal`] should compute and read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputMode {
    /// Intermediate prefill chunk: advance KV/conv states only, skip unembed & readback.
    None,
    /// Compute & return only the final token's logits `[1, unpadded_vocab]`.
    LastOnly,
    /// Compute & return logits for every input token `[seq, unpadded_vocab]`.
    All,
}

/// Persistent GPU decode state + fixed-address per-token working buffers enabling 1-submit
/// 48-layer decode with 100% bind-group cache hits and zero per-token Vulkan allocations.
#[derive(Debug)]
pub(crate) struct GpuDecodeState {
    pub(crate) synced: bool,
    pub(crate) host_dirty: bool,
    pub(crate) recipe: Option<ChainRecipe>,
    pub(crate) slots: Vec<GpuSlotState>,
    /// Lightweight GPU-side prefill-end checkpoint of short-convolution histories (~3.7 MiB)
    /// enabling instant KV prefix reuse across multi-turn chats & agent loops even when
    /// generated thinking blocks are stripped from the next turn's prompt.
    pub(crate) ckpt_pos: Option<usize>,
    pub(crate) ckpt_slots: Vec<(DeviceVec, DeviceVec, DeviceVec)>,
    pub(crate) step_params: DeviceVec,
    pub(crate) attn_split: DeviceVec,
    pub(crate) cur: DeviceVec,
    pub(crate) prelude_cur: DeviceVec,
    pub(crate) blocks: DeviceVec,
    pub(crate) bank: DeviceVec,
    pub(crate) bank_out: DeviceVec,
    pub(crate) msgs: DeviceVec,
    pub(crate) mix_out: DeviceVec,
    pub(crate) mid_qkv: DeviceVec,
    pub(crate) ctrl_qkv: DeviceVec,
    pub(crate) norm_qkv: DeviceVec,
    pub(crate) ffn_inp: DeviceVec,
    pub(crate) ffn_x: DeviceVec,
    pub(crate) attn_gated: DeviceVec,
    pub(crate) attn_proj: DeviceVec,
    pub(crate) attn_conv: DeviceVec,
    pub(crate) logits: DeviceVec,
}

/// Persistent GPU working buffers for single-submit full-GPU batched prefill (`seq >= 1`),
/// kept separate from [`GpuDecodeState`] so multi-turn prefills never invalidate decode's
/// captured [`ChainRecipe`].
#[derive(Debug)]
pub(crate) struct GpuPrefillScratch {
    pub(crate) cap_seq: usize,
    pub(crate) max_blocks: usize,
    pub(crate) tau_params: DeviceVec,
    pub(crate) cur: DeviceVec,
    pub(crate) prelude_cur: DeviceVec,
    pub(crate) blocks: DeviceVec,
    pub(crate) bank: DeviceVec,
    pub(crate) bank_out: DeviceVec,
    pub(crate) msgs: DeviceVec,
    pub(crate) mix_out: DeviceVec,
    pub(crate) mid_qkv: DeviceVec,
    pub(crate) ctrl_qkv: DeviceVec,
    pub(crate) ffn_inp: DeviceVec,
    pub(crate) ffn_x: DeviceVec,
    pub(crate) attn_gated: DeviceVec,
    pub(crate) attn_proj: DeviceVec,
    pub(crate) attn_conv: DeviceVec,
    pub(crate) moe_out: DeviceVec,
    pub(crate) norm_bufs: Vec<DeviceVec>,
    pub(crate) ctrl_out: Vec<DeviceVec>,
    pub(crate) y_bufs: Vec<DeviceVec>,
    pub(crate) router_buf: DeviceVec,
    pub(crate) x_bufs: Vec<DeviceVec>,
    pub(crate) mid_bufs: Vec<DeviceVec>,
    pub(crate) ctrl_gain: Vec<DeviceVec>,
    pub(crate) ctrl_forget: Vec<DeviceVec>,
    pub(crate) logits_all: Option<DeviceVec>,
}

/// Reusable GPU device vectors for `DeviceChain` batched matmuls and fused layer chains,
/// eliminating per-token GPU buffer allocations and enabling 100% bind-group cache hits during
/// decode (`seq == 1`).
#[derive(Debug, Default)]
pub(crate) struct GpuScratch {
    x_bufs: Vec<DeviceVec>,
    y_bufs: Vec<DeviceVec>,
    mid_bufs: Vec<DeviceVec>,
    ctrl_cur: Vec<DeviceVec>,
    ctrl_msg: Vec<DeviceVec>,
    ctrl_gain: Vec<DeviceVec>,
    ctrl_forget: Vec<DeviceVec>,
    ctrl_out: Vec<DeviceVec>,
    norm_bufs: Vec<DeviceVec>,
    router_buf: Option<DeviceVec>,
    decode_state: Option<GpuDecodeState>,
    prefill_scratch: Option<GpuPrefillScratch>,
}

impl GpuScratch {
    pub(crate) fn reset_decode_state(&mut self) {
        if let Some(ds) = self.decode_state.as_mut() {
            ds.synced = false;
            ds.host_dirty = false;
            ds.recipe = None;
            ds.ckpt_pos = None;
        }
    }

    fn ensure_slot(bufs: &mut Vec<DeviceVec>, chain: &dyn DeviceChain, idx: usize, needed: usize) {
        let len = needed.max(1);
        while bufs.len() <= idx {
            bufs.push(chain.vec(1));
        }
        if bufs[idx].len < len {
            bufs[idx] = chain.vec(len);
        }
    }

    /// Cheap zero-copy prefix view over a persistent `DeviceVec` with exact logical length `len`,
    /// preserving the underlying buffer identity for `keep_groups(true)` bind-group caching.
    #[inline]
    pub(crate) fn view(v: &DeviceVec, len: usize) -> DeviceVec {
        debug_assert!(v.len >= len);
        DeviceVec {
            len,
            inner: Arc::clone(&v.inner),
        }
    }

    pub(crate) fn ensure_x(&mut self, chain: &dyn DeviceChain, idx: usize, needed: usize) {
        Self::ensure_slot(&mut self.x_bufs, chain, idx, needed);
    }

    pub(crate) fn ensure_y(&mut self, chain: &dyn DeviceChain, idx: usize, needed: usize) {
        Self::ensure_slot(&mut self.y_bufs, chain, idx, needed);
    }

    pub(crate) fn ensure_router(&mut self, chain: &dyn DeviceChain, needed: usize) {
        let len = needed.max(1);
        if self.router_buf.as_ref().is_none_or(|b| b.len < len) {
            self.router_buf = Some(chain.vec(len));
        }
    }

    pub(crate) fn x(&self, idx: usize) -> &DeviceVec {
        &self.x_bufs[idx]
    }

    pub(crate) fn y(&self, idx: usize) -> &DeviceVec {
        &self.y_bufs[idx]
    }

    pub(crate) fn router(&self) -> &DeviceVec {
        self.router_buf.as_ref().expect("ensure_router must be called first")
    }
}

#[inline]
fn weight_on_chain(chain: &dyn DeviceChain, w: &Weight) -> Option<usize> {
    match w {
        Weight::Quant(qt) if chain.holds(qt) => Some(qt.dim(0)),
        Weight::Packed(p) if chain.holds_exl3(p.as_ref()) => Some(p.shape()[0]),
        _ => None,
    }
}

#[inline]
fn record_weight(
    rec: &mut dyn ChainRecorder,
    w: &Weight,
    x: &DeviceVec,
    y: &DeviceVec,
    seq: usize,
) {
    match w {
        Weight::Quant(qt) => rec.matmul_rows(qt, x, y, seq),
        Weight::Packed(p) => rec.exl3_rows(p.as_ref(), x, y, seq),
        _ => unreachable!("record_weight called on non-device weight"),
    }
}

/// Fast CPU-side linear projection with fused SIMD kernels (`IQ4_NL`, `Q8_0`, `Q4_K`, `Q6_K`,
/// `Bf16Linear`, and 4-row 8-lane `F32`) and optional output-row cap (`max_rows`).
pub(crate) fn linear_host_cpu(
    backend: &dyn ggml_rs::Backend,
    w: &Weight,
    x: &[f32],
    seq: usize,
    in_dim: usize,
    max_rows: usize,
) -> Vec<f32> {
    match w {
        Weight::Quant(qt)
            if !qt.is_device()
                && qt.rank() == 2
                && qt.dim(1) == in_dim
                && in_dim % qt.dtype().block_size() == 0 =>
        {
            let dtype = qt.dtype();
            let n = max_rows.min(qt.dim(0));
            let rb = quant_row_bytes(dtype, in_dim);
            let raw = &qt.bytes()[..n * rb];
            let mut out = vec![0.0f32; seq * n];
            if seq == 1 {
                if n >= 64 {
                    out.par_chunks_mut(32).enumerate().for_each(|(ti, dst)| {
                        let r0 = ti * 32;
                        let slice = &raw[r0 * rb..(r0 + dst.len()) * rb];
                        gguf_dot_rows_any(dtype, slice, x, dst.len(), in_dim, dst)
                            .expect("inkling2: host quant matmul");
                    });
                } else {
                    gguf_dot_rows_any(dtype, raw, x, n, in_dim, &mut out)
                        .expect("inkling2: host quant matmul");
                }
            } else if seq >= 8 {
                out.par_chunks_mut(n).enumerate().for_each(|(row_i, dst)| {
                    let xr = &x[row_i * in_dim..(row_i + 1) * in_dim];
                    gguf_dot_rows_any(dtype, raw, xr, n, in_dim, dst)
                        .expect("inkling2: host quant matmul");
                });
            } else {
                for row_i in 0..seq {
                    let xr = &x[row_i * in_dim..(row_i + 1) * in_dim];
                    out[row_i * n..(row_i + 1) * n]
                        .par_chunks_mut(32)
                        .enumerate()
                        .for_each(|(ti, dst)| {
                            let r0 = ti * 32;
                            let slice = &raw[r0 * rb..(r0 + dst.len()) * rb];
                            gguf_dot_rows_any(dtype, slice, xr, dst.len(), in_dim, dst)
                                .expect("inkling2: host quant matmul");
                        });
                }
            }
            out
        }
        Weight::Packed(p) => {
            if let Some(bf16) = p.as_any().and_then(|a| a.downcast_ref::<Bf16Linear>()) {
                return bf16.linear_rows_slice(x, seq, max_rows);
            }
            let x_t = Tensor::from_vec(x.to_vec(), vec![seq, in_dim]);
            let y_t = p.linear(&x_t);
            let full = if y_t.is_device() {
                y_t.to_host().data().to_vec()
            } else {
                y_t.data().to_vec()
            };
            trim_rows(full, seq, p.shape()[0], max_rows)
        }
        Weight::Dense(t) if !t.is_device() && t.rank() == 2 && t.dim(1) == in_dim => {
            let n = max_rows.min(t.dim(0));
            let wd = &t.data()[..n * in_dim];
            let mut out = vec![0.0f32; seq * n];
            let work = n * in_dim;
            // With AVX-512F 4-row dots, small F32 projections (<1M elements = 4 MB, such as
            // routers and loop controllers) finish in ~20-40 us on 1 core — faster than waking
            // a Rayon thread pool barrier across CCDs.
            const PAR_WORK_MIN: usize = 1_048_576;
            if seq == 1 {
                if n >= 128 && work >= PAR_WORK_MIN {
                    out.par_chunks_mut(64).enumerate().for_each(|(ti, dst)| {
                        let r0 = ti * 64;
                        f32_dot_rows_fast(
                            &wd[r0 * in_dim..(r0 + dst.len()) * in_dim],
                            x,
                            dst.len(),
                            in_dim,
                            dst,
                        );
                    });
                } else {
                    f32_dot_rows_fast(wd, x, n, in_dim, &mut out);
                }
            } else if seq >= 8 && seq * work >= PAR_WORK_MIN {
                out.par_chunks_mut(n).enumerate().for_each(|(row_i, dst)| {
                    f32_dot_rows_fast(wd, &x[row_i * in_dim..(row_i + 1) * in_dim], n, in_dim, dst);
                });
            } else if n >= 128 && work >= PAR_WORK_MIN {
                for row_i in 0..seq {
                    let xr = &x[row_i * in_dim..(row_i + 1) * in_dim];
                    out[row_i * n..(row_i + 1) * n]
                        .par_chunks_mut(64)
                        .enumerate()
                        .for_each(|(ti, dst)| {
                            let r0 = ti * 64;
                            f32_dot_rows_fast(
                                &wd[r0 * in_dim..(r0 + dst.len()) * in_dim],
                                xr,
                                dst.len(),
                                in_dim,
                                dst,
                            );
                        });
                }
            } else {
                for row_i in 0..seq {
                    f32_dot_rows_fast(
                        wd,
                        &x[row_i * in_dim..(row_i + 1) * in_dim],
                        n,
                        in_dim,
                        &mut out[row_i * n..(row_i + 1) * n],
                    );
                }
            }
            out
        }
        _ => {
            let x_t = Tensor::from_vec(x.to_vec(), vec![seq, in_dim]);
            let y_t = w.linear(backend, &x_t);
            let full = if y_t.is_device() {
                y_t.to_host().data().to_vec()
            } else {
                y_t.data().to_vec()
            };
            trim_rows(full, seq, w.shape()[0], max_rows)
        }
    }
}

#[inline]
fn trim_rows(full: Vec<f32>, seq: usize, out_dim: usize, max_rows: usize) -> Vec<f32> {
    if max_rows >= out_dim {
        return full;
    }
    let mut trimmed = Vec::with_capacity(seq * max_rows);
    for r in 0..seq {
        trimmed.extend_from_slice(&full[r * out_dim..r * out_dim + max_rows]);
    }
    trimmed
}

/// Single linear projection using persistent `DeviceChain` scratch when on GPU, or
/// AVX-512/AVX2/auto-vectorized host kernels when on CPU.
pub(crate) fn linear_host_fast(
    backend: &dyn ggml_rs::Backend,
    scratch: &mut GpuScratch,
    w: &Weight,
    x: &[f32],
    seq: usize,
    in_dim: usize,
    max_rows: usize,
) -> Vec<f32> {
    if let Some(chain) = backend.chain() {
        if let Some(out_dim) = weight_on_chain(chain, w) {
            GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, 0, seq * in_dim);
            GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 0, seq * out_dim);
            chain.upload(&scratch.x_bufs[0], x);
            let mut rec = chain.begin();
            rec.keep_groups(seq == 1);
            record_weight(&mut *rec, w, &scratch.x_bufs[0], &scratch.y_bufs[0], seq);
            let read_len = if seq == 1 {
                max_rows.min(out_dim)
            } else {
                seq * out_dim
            };
            rec.read_range(&scratch.y_bufs[0], 0, read_len);
            let full = rec.finish().into_iter().next().expect("1 chain read");
            return if seq == 1 {
                full
            } else {
                trim_rows(full, seq, out_dim, max_rows)
            };
        }
    }
    linear_host_cpu(backend, w, x, seq, in_dim, max_rows)
}

/// Batch multiple linear projections (sharing one or more input vectors of width `in_dim`)
/// into a SINGLE `DeviceChain` GPU submission, overlapping any host-resident weights on CPU
/// while the GPU executes.
pub(crate) fn linear_host_batch(
    backend: &dyn ggml_rs::Backend,
    scratch: &mut GpuScratch,
    inputs: &[&[f32]],
    ops: &[(&Weight, usize)],
    seq: usize,
    in_dim: usize,
) -> Vec<Vec<f32>> {
    if let Some(chain) = backend.chain() {
        let mut gpu_ops: Vec<(usize, &Weight, usize, usize)> = Vec::with_capacity(ops.len());
        let mut cpu_ops: Vec<(usize, &Weight, usize)> = Vec::new();
        for (op_idx, &(w, inp_idx)) in ops.iter().enumerate() {
            if let Some(out_dim) = weight_on_chain(chain, w) {
                gpu_ops.push((op_idx, w, inp_idx, out_dim));
            } else {
                cpu_ops.push((op_idx, w, inp_idx));
            }
        }
        if !gpu_ops.is_empty() {
            let mut uploaded = vec![false; inputs.len()];
            for &(_, _, inp_idx, _) in &gpu_ops {
                if !uploaded[inp_idx] {
                    GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, inp_idx, seq * in_dim);
                    uploaded[inp_idx] = true;
                }
            }
            for (slot, &(_, _, _, out_dim)) in gpu_ops.iter().enumerate() {
                GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, slot, seq * out_dim);
            }
            for (inp_idx, &was_needed) in uploaded.iter().enumerate() {
                if was_needed {
                    chain.upload(&scratch.x_bufs[inp_idx], inputs[inp_idx]);
                }
            }

            let mut rec = chain.begin();
            rec.keep_groups(seq == 1);
            for (slot, &(_, w, inp_idx, out_dim)) in gpu_ops.iter().enumerate() {
                record_weight(&mut *rec, w, &scratch.x_bufs[inp_idx], &scratch.y_bufs[slot], seq);
                rec.read_range(&scratch.y_bufs[slot], 0, seq * out_dim);
            }
            let submitted = rec.submit();

            let mut results: Vec<Option<Vec<f32>>> = vec![None; ops.len()];
            for (op_idx, w, inp_idx) in cpu_ops {
                results[op_idx] = Some(linear_host_cpu(
                    backend,
                    w,
                    inputs[inp_idx],
                    seq,
                    in_dim,
                    usize::MAX,
                ));
            }

            let gpu_reads = submitted.wait();
            for ((op_idx, _, _, _), data) in gpu_ops.into_iter().zip(gpu_reads.into_iter()) {
                results[op_idx] = Some(data);
            }
            return results.into_iter().map(|r| r.expect("op result")).collect();
        }
    }

    ops.iter()
        .map(|&(w, inp_idx)| linear_host_cpu(backend, w, inputs[inp_idx], seq, in_dim, usize::MAX))
        .collect()
}

/// Weighted RMSNorm across the last dimension (`[rows, dim]`).
pub(crate) fn rms_norm_rows(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let src = &x[r * dim..(r + 1) * dim];
        let dst = &mut out[r * dim..(r + 1) * dim];
        let sq = f32_dot_fast(src, src);
        let inv = 1.0 / (sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            dst[i] = src[i] * inv * weight[i];
        }
    }
    out
}

/// Unweighted RMSNorm across the last dimension (`[rows, dim]`).
pub(crate) fn rms_norm_no_weight(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let src = &x[r * dim..(r + 1) * dim];
        let dst = &mut out[r * dim..(r + 1) * dim];
        let sq = f32_dot_fast(src, src);
        let inv = 1.0 / (sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            dst[i] = src[i] * inv;
        }
    }
    out
}

/// Per-head optional learned scale followed by L2 norm clipping to `threshold`:
/// `y = x * scale; y = y * min(1.0, threshold / (||y||_2 + 1e-6))`.
fn l2_clip_heads(
    x: &mut [f32],
    scale: Option<&[f32]>,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    threshold: f32,
) {
    let stride = n_heads * head_dim;
    for t in 0..seq {
        for h in 0..n_heads {
            let off = t * stride + h * head_dim;
            let slice = &mut x[off..off + head_dim];
            if let Some(sc) = scale {
                let s_h = &sc[h * head_dim..(h + 1) * head_dim];
                for d in 0..head_dim {
                    slice[d] *= s_h[d];
                }
            }
            let sq = f32_dot_fast(slice, slice);
            let factor = (threshold / (sq.sqrt() + 1e-6)).min(1.0);
            if factor < 1.0 {
                for v in slice.iter_mut() {
                    *v *= factor;
                }
            }
        }
    }
}

/// Block Attention Residuals (`Inkling2BlockAttentionResidual` / `inkling2.cpp::mix`).
///
/// `normed_blocks` must already have unweighted RMSNorm (`eps = 1e-6`) applied; `cur` is
/// normalized once here unless `cur_is_normed` is true. Computes softmax attention over depth
/// using `n_queries` learned query vectors (`query`: `[n_queries, hidden]`).
///
/// Returns `n_queries` vectors, each of shape `[seq, hidden]`.
pub(crate) fn mix_depth_impl(
    normed_blocks: &[Vec<f32>],
    cur: &[f32],
    cur_is_normed: bool,
    query: &[f32],
    n_queries: usize,
    seq: usize,
    hidden: usize,
) -> Vec<Vec<f32>> {
    let n_states = normed_blocks.len() + 1;
    debug_assert_eq!(cur.len(), seq * hidden);
    debug_assert_eq!(query.len(), n_queries * hidden);

    let owned_cur;
    let cur_n: &[f32] = if cur_is_normed {
        cur
    } else {
        owned_cur = rms_norm_no_weight(cur, hidden, 1e-6);
        &owned_cur
    };

    let inv_sqrt_d = 1.0 / (hidden as f32).sqrt();
    let mut outs: Vec<Vec<f32>> = (0..n_queries).map(|_| vec![0.0f32; seq * hidden]).collect();
    let mut scores = vec![0.0f32; n_states];

    for t in 0..seq {
        let row_off = t * hidden;
        let row_end = row_off + hidden;
        for q_idx in 0..n_queries {
            let q_vec = &query[q_idx * hidden..(q_idx + 1) * hidden];
            let mut max_s = f32::NEG_INFINITY;
            for (s, st) in normed_blocks.iter().enumerate() {
                let sc = f32_dot_fast(&st[row_off..row_end], q_vec) * inv_sqrt_d;
                scores[s] = sc;
                if sc > max_s {
                    max_s = sc;
                }
            }
            {
                let sc = f32_dot_fast(&cur_n[row_off..row_end], q_vec) * inv_sqrt_d;
                scores[normed_blocks.len()] = sc;
                if sc > max_s {
                    max_s = sc;
                }
            }
            let mut sum_exp = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - max_s).exp();
                sum_exp += *sc;
            }
            let inv_sum = 1.0 / sum_exp;
            let dst = &mut outs[q_idx][row_off..row_end];
            for (s, st) in normed_blocks.iter().enumerate() {
                f32_axpy_fast(dst, &st[row_off..row_end], scores[s] * inv_sum);
            }
            f32_axpy_fast(dst, &cur_n[row_off..row_end], scores[normed_blocks.len()] * inv_sum);
        }
    }

    outs
}

#[inline]
pub(crate) fn mix_depth(
    normed_blocks: &[Vec<f32>],
    cur: &[f32],
    query: &[f32],
    n_queries: usize,
    seq: usize,
    hidden: usize,
) -> Vec<Vec<f32>> {
    mix_depth_impl(normed_blocks, cur, false, query, n_queries, seq, hidden)
}

#[inline]
fn build_control_cat(cur: &[f32], msg: &[f32], seq: usize, hidden: usize) -> Vec<f32> {
    let cur_n = rms_norm_no_weight(cur, hidden, 1e-6);
    let msg_n = rms_norm_no_weight(msg, hidden, 1e-6);
    let mut cat = Vec::with_capacity(seq * 2 * hidden);
    for t in 0..seq {
        cat.extend_from_slice(&cur_n[t * hidden..(t + 1) * hidden]);
        cat.extend_from_slice(&msg_n[t * hidden..(t + 1) * hidden]);
    }
    cat
}

#[inline]
fn blend_control(cur: &[f32], msg: &[f32], gain: &[f32], forget: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; cur.len()];
    for i in 0..out.len() {
        let g = gain[i].tanh();
        let f = forget[i].tanh();
        out[i] = (1.0 - f) * cur[i] + g * msg[i];
    }
    out
}

/// Cross-pass gated controller (`inkling2.cpp::control` / `Inkling2CrossPassController`):
/// `g = ctrl_down(concat(rms_norm(cur), rms_norm(msg)))`
/// `gain = tanh(ctrl_gain(g))`, `forget = tanh(ctrl_forget(g))`
/// `out = (1 - forget) * cur + gain * msg`
#[allow(clippy::too_many_arguments)]
fn apply_control(
    backend: &dyn ggml_rs::Backend,
    scratch: &mut GpuScratch,
    cur: &[f32],
    msg: &[f32],
    loop_layer: &Inkling2LoopLayer,
    ctrl_idx: usize,
    seq: usize,
    hidden: usize,
    gate_rank: usize,
) -> Vec<f32> {
    let cat = build_control_cat(cur, msg, seq, hidden);
    if let Some(chain) = backend.chain() {
        let w_dn = &loop_layer.ctrl_down[ctrl_idx];
        let w_gn = &loop_layer.ctrl_gain[ctrl_idx];
        let w_fg = &loop_layer.ctrl_forget[ctrl_idx];
        if weight_on_chain(chain, w_dn).is_some()
            && weight_on_chain(chain, w_gn).is_some()
            && weight_on_chain(chain, w_fg).is_some()
        {
            GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, 0, seq * 2 * hidden);
            GpuScratch::ensure_slot(&mut scratch.mid_bufs, chain, 0, seq * gate_rank);
            GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 0, seq * hidden);
            GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 1, seq * hidden);
            chain.upload(&scratch.x_bufs[0], &cat);
            let mut rec = chain.begin();
            rec.keep_groups(seq == 1);
            record_weight(&mut *rec, w_dn, &scratch.x_bufs[0], &scratch.mid_bufs[0], seq);
            record_weight(&mut *rec, w_gn, &scratch.mid_bufs[0], &scratch.y_bufs[0], seq);
            record_weight(&mut *rec, w_fg, &scratch.mid_bufs[0], &scratch.y_bufs[1], seq);
            rec.read_range(&scratch.y_bufs[0], 0, seq * hidden);
            rec.read_range(&scratch.y_bufs[1], 0, seq * hidden);
            let mut outs = rec.finish();
            let forget = outs.pop().expect("forget");
            let gain = outs.pop().expect("gain");
            return blend_control(cur, msg, &gain, &forget);
        }
    }
    let gate = linear_host_fast(backend, scratch, &loop_layer.ctrl_down[ctrl_idx], &cat, seq, 2 * hidden, usize::MAX);
    let outs = linear_host_batch(
        backend,
        scratch,
        &[&gate],
        &[(&loop_layer.ctrl_gain[ctrl_idx], 0), (&loop_layer.ctrl_forget[ctrl_idx], 0)],
        seq,
        gate_rank,
    );
    blend_control(cur, msg, &outs[0], &outs[1])
}

/// Apply all three independent Q/K/V cross-pass controllers in a single GPU submission
/// when on device (eliminating 8 of 9 submits per looped layer pass), or directly on CPU.
#[allow(clippy::too_many_arguments)]
fn apply_controls_qkv(
    backend: &dyn ggml_rs::Backend,
    scratch: &mut GpuScratch,
    inp_q: &[f32],
    inp_k: &[f32],
    inp_v: &[f32],
    msgs: &[Vec<f32>],
    loop_layer: &Inkling2LoopLayer,
    seq: usize,
    hidden: usize,
    gate_rank: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if let Some(chain) = backend.chain() {
        let all_gpu = (0..3).all(|i| {
            weight_on_chain(chain, &loop_layer.ctrl_down[i]).is_some()
                && weight_on_chain(chain, &loop_layer.ctrl_gain[i]).is_some()
                && weight_on_chain(chain, &loop_layer.ctrl_forget[i]).is_some()
        });
        if all_gpu {
            let cat0 = build_control_cat(inp_q, &msgs[0], seq, hidden);
            let cat1 = build_control_cat(inp_k, &msgs[1], seq, hidden);
            let cat2 = build_control_cat(inp_v, &msgs[2], seq, hidden);
            let cats = [&cat0[..], &cat1[..], &cat2[..]];
            for i in 0..3 {
                GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, i, seq * 2 * hidden);
                GpuScratch::ensure_slot(&mut scratch.mid_bufs, chain, i, seq * gate_rank);
                GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 2 * i, seq * hidden);
                GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 2 * i + 1, seq * hidden);
                chain.upload(&scratch.x_bufs[i], cats[i]);
            }
            let mut rec = chain.begin();
            rec.keep_groups(seq == 1);
            for i in 0..3 {
                record_weight(&mut *rec, &loop_layer.ctrl_down[i], &scratch.x_bufs[i], &scratch.mid_bufs[i], seq);
                record_weight(&mut *rec, &loop_layer.ctrl_gain[i], &scratch.mid_bufs[i], &scratch.y_bufs[2 * i], seq);
                record_weight(&mut *rec, &loop_layer.ctrl_forget[i], &scratch.mid_bufs[i], &scratch.y_bufs[2 * i + 1], seq);
                rec.read_range(&scratch.y_bufs[2 * i], 0, seq * hidden);
                rec.read_range(&scratch.y_bufs[2 * i + 1], 0, seq * hidden);
            }
            let outs = rec.finish();
            return (
                blend_control(inp_q, &msgs[0], &outs[0], &outs[1]),
                blend_control(inp_k, &msgs[1], &outs[2], &outs[3]),
                blend_control(inp_v, &msgs[2], &outs[4], &outs[5]),
            );
        }
    }
    (
        apply_control(backend, scratch, inp_q, &msgs[0], loop_layer, 0, seq, hidden, gate_rank),
        apply_control(backend, scratch, inp_k, &msgs[1], loop_layer, 1, seq, hidden, gate_rank),
        apply_control(backend, scratch, inp_v, &msgs[2], loop_layer, 2, seq, hidden, gate_rank),
    )
}

/// Compute `table[t, h, d] = sum_{r=0..d_rel} R[t, h, r] * rel_bias[d, r]` for `d in 0..active_rel`.
///
/// * `r_cur`: `[seq, n_heads, d_rel]`
/// * `rel_bias`: `[rel_extent, d_rel]` (row-major)
/// * `active_rel`: `<= rel_extent` (only distances actually reachable at current `past_len + seq`)
///
/// Returns `[seq, n_heads, active_rel]`.
fn project_rel_bias(
    r_cur: &[f32],
    rel_bias: &[f32],
    seq: usize,
    n_heads: usize,
    d_rel: usize,
    active_rel: usize,
) -> Vec<f32> {
    let mut table = vec![0.0f32; seq * n_heads * active_rel];
    let compute = |th: usize, dst: &mut [f32]| {
        let r_vec = &r_cur[th * d_rel..(th + 1) * d_rel];
        f32_dot_rows_fast(&rel_bias[..active_rel * d_rel], r_vec, active_rel, d_rel, dst);
    };
    if seq * n_heads * active_rel >= 16_384 {
        table
            .par_chunks_mut(active_rel)
            .enumerate()
            .for_each(|(th, dst)| compute(th, dst));
    } else {
        for (th, dst) in table.chunks_mut(active_rel).enumerate() {
            compute(th, dst);
        }
    }
    table
}

impl Inkling2Model {
    /// Maximum batch size supported by [`Self::forward_prefill_gpu`] on the active device.
    pub(crate) fn max_gpu_prefill_seq(&self) -> usize {
        if !self.full_gpu_decode {
            return 0;
        }
        let Some(chain) = self.backend.chain() else {
            return 0;
        };
        let mut max_rows = 65535usize;
        for layer in &self.layers {
            if let Some(qmoe) = layer.experts.quant_moe() {
                max_rows = max_rows.min(chain.moe_routed_max_rows(qmoe));
            } else {
                return 0;
            }
        }
        max_rows
    }

    /// Optimal prompt prefill chunk size for this model and device:
    /// `1024` (Tensor-Core MoE peak: ~64 tokens/expert, >2,100 tok/s on RTX 5090 + 1,024-token
    /// SWA ring-buffer slack for instant multi-turn checkpoint rewind) on cooperative-matrix
    /// full-GPU devices, `64` otherwise. Override with `OAIY_INKLING2_CHUNK=<16..=2048>`.
    pub fn preferred_prefill_chunk(&self) -> usize {
        let max_gpu = self.max_gpu_prefill_seq();
        if max_gpu >= 512 {
            let default_chunk = if max_gpu >= 1024 { 1024 } else { 512 };
            std::env::var("OAIY_INKLING2_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| (16..=2048).contains(&n))
                .unwrap_or(default_chunk)
                .min(max_gpu)
        } else {
            64
        }
    }

    /// Snapshot all GPU-resident short-convolution histories at the end of prompt prefill so
    /// subsequent multi-turn / agent steps can rewind back to this exact prefix position in
    /// < 0.2 ms (as long as decoded tokens stayed within SWA ring-buffer slack).
    pub fn save_prefill_checkpoint(&self) -> bool {
        let Some(chain) = self.backend.chain() else {
            return false;
        };
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let pos = st.pos;
        if pos == 0 {
            return false;
        }
        let mut scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ds) = scratch_guard.as_mut().and_then(|s| s.decode_state.as_mut()) else {
            return false;
        };
        if !ds.synced {
            return false;
        }
        if ds.ckpt_slots.len() != ds.slots.len() {
            ds.ckpt_slots = ds
                .slots
                .iter()
                .map(|s| {
                    (
                        chain.vec(s.kv_conv.len),
                        chain.vec(s.attn_post_conv.len),
                        chain.vec(s.ffn_post_conv.len),
                    )
                })
                .collect();
        }
        let mut rec = chain.begin();
        rec.keep_groups(false);
        for (slot, (c_kv, c_ap, c_fp)) in ds.slots.iter().zip(&ds.ckpt_slots) {
            rec.copy(&slot.kv_conv, 0, c_kv, 0, slot.kv_conv.len);
            rec.copy(&slot.attn_post_conv, 0, c_ap, 0, slot.attn_post_conv.len);
            rec.copy(&slot.ffn_post_conv, 0, c_fp, 0, slot.ffn_post_conv.len);
        }
        let _ = rec.finish();
        ds.ckpt_pos = Some(pos);
        true
    }

    /// Restore the GPU short-convolution checkpoint saved at `target_pos` and rewind `state.pos`
    /// to `target_pos`. Returns `false` if no matching checkpoint exists or if subsequent decoding
    /// exceeded any SWA slot's ring-buffer slack (`kv_cap - sliding_window`).
    pub fn restore_prefill_checkpoint(&self, target_pos: usize) -> bool {
        let Some(chain) = self.backend.chain() else {
            return false;
        };
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if target_pos == 0 || target_pos > st.pos {
            return false;
        }
        let decoded = st.pos - target_pos;
        let mut scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ds) = scratch_guard.as_mut().and_then(|s| s.decode_state.as_mut()) else {
            return false;
        };
        if !ds.synced || ds.ckpt_pos != Some(target_pos) || ds.ckpt_slots.len() != ds.slots.len() {
            return false;
        }
        let swa = self.cfg.sliding_window;
        if ds
            .slots
            .iter()
            .any(|s| s.is_swa && decoded > s.kv_cap.saturating_sub(swa))
        {
            return false;
        }
        let mut rec = chain.begin();
        rec.keep_groups(false);
        for (slot, (c_kv, c_ap, c_fp)) in ds.slots.iter().zip(&ds.ckpt_slots) {
            rec.copy(c_kv, 0, &slot.kv_conv, 0, slot.kv_conv.len);
            rec.copy(c_ap, 0, &slot.attn_post_conv, 0, slot.attn_post_conv.len);
            rec.copy(c_fp, 0, &slot.ffn_post_conv, 0, slot.ffn_post_conv.len);
        }
        let _ = rec.finish();
        st.pos = target_pos;
        true
    }

    /// Maximum tokens that may be decoded past a live checkpoint before SWA ring buffers
    /// overwrite its sliding-window rows.
    pub fn swa_checkpoint_slack(&self) -> usize {
        let scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(ds) = scratch_guard.as_ref().and_then(|s| s.decode_state.as_ref()) {
            let swa = self.cfg.sliding_window;
            return ds
                .slots
                .iter()
                .filter(|s| s.is_swa)
                .map(|s| s.kv_cap.saturating_sub(swa))
                .min()
                .unwrap_or(usize::MAX);
        }
        self.cfg.sliding_window
    }

    /// Exact serialized byte size of a full [`Inkling2StateSnapshot`] at `pos` tokens, computed
    /// before capturing so RAM & DiskCache budgets can gate capture cheaply.
    pub fn estimate_snapshot_bytes(&self, pos: usize) -> u64 {
        let swa = self.cfg.sliding_window;
        let past = self.cfg.conv_kernel.saturating_sub(1);
        let hidden = self.cfg.hidden;
        let n_slots = self.cfg.n_logical_slots();
        let mut total: u64 = 48; // SNAPSHOT_HEADER_BYTES
        for il in 0..n_slots {
            let ip = self.cfg.layer_phys(il);
            let is_swa = self.cfg.is_swa[ip];
            let kv_dim = self.cfg.kv_head_counts[ip] * self.cfg.head_dims[ip];
            let rows = if is_swa { pos.min(swa) } else { pos };
            let floats = 2 * rows * kv_dim + 2 * past * kv_dim + 2 * past * hidden;
            total = total.saturating_add(25 + 4 * floats as u64);
        }
        total
    }

    /// Capture a lightweight (~3.7 MiB) host-RAM snapshot of all logical slots' short-convolution
    /// histories at the current position. Works on both GPU and CPU paths without mutating state.
    pub fn capture_conv_checkpoint(&self) -> Option<Inkling2ConvSnapshot> {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.pos == 0 {
            return None;
        }
        let scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        if let (Some(chain), Some(ds)) = (
            self.backend.chain(),
            scratch_guard
                .as_ref()
                .and_then(|s| s.decode_state.as_ref())
                .filter(|ds| ds.synced),
        ) {
            let past = self.cfg.conv_kernel.saturating_sub(1);
            let hidden = self.cfg.hidden;
            if past == 0 {
                let slots = (0..ds.slots.len())
                    .map(|_| Inkling2ConvSlotSnapshot {
                        k_conv: Vec::new(),
                        v_conv: Vec::new(),
                        attn_post_conv: Vec::new(),
                        ffn_post_conv: Vec::new(),
                    })
                    .collect();
                return Some(Inkling2ConvSnapshot { pos: st.pos, slots });
            }
            let mut rec = chain.begin();
            rec.keep_groups(false);
            for g_slot in &ds.slots {
                let c_len = past * g_slot.kv_dim;
                rec.read_range(&g_slot.kv_conv, 0, c_len);
                rec.read_range(&g_slot.kv_conv, c_len, c_len);
                rec.read_range(&g_slot.attn_post_conv, 0, past * hidden);
                rec.read_range(&g_slot.ffn_post_conv, 0, past * hidden);
            }
            let mut reads = rec.finish().into_iter();
            let slots = (0..ds.slots.len())
                .map(|_| Inkling2ConvSlotSnapshot {
                    k_conv: reads.next().expect("k_conv"),
                    v_conv: reads.next().expect("v_conv"),
                    attn_post_conv: reads.next().expect("attn_post_conv"),
                    ffn_post_conv: reads.next().expect("ffn_post_conv"),
                })
                .collect();
            return Some(Inkling2ConvSnapshot { pos: st.pos, slots });
        }
        Some(st.capture_conv_checkpoint())
    }

    /// Rewind the live model state (GPU or CPU) back to `ckpt.pos` using a lightweight conv
    /// checkpoint. Returns `false` if SWA ring rows for `ckpt.pos` were already overwritten or
    /// dimensions do not match.
    pub fn restore_conv_checkpoint(&self, ckpt: &Inkling2ConvSnapshot) -> bool {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let target_pos = ckpt.pos;
        if target_pos == 0 || target_pos > st.pos || ckpt.slots.len() != st.slots.len() {
            return false;
        }
        let decoded = st.pos - target_pos;
        let swa = self.cfg.sliding_window;
        let mut scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        if let (Some(chain), Some(ds)) = (
            self.backend.chain(),
            scratch_guard
                .as_mut()
                .and_then(|s| s.decode_state.as_mut())
                .filter(|ds| ds.synced),
        ) {
            if ds.slots.len() != ckpt.slots.len() {
                return false;
            }
            if ds
                .slots
                .iter()
                .any(|s| s.is_swa && decoded > s.kv_cap.saturating_sub(swa))
            {
                return false;
            }
            for (g_slot, c_slot) in ds.slots.iter().zip(&ckpt.slots) {
                if !c_slot.k_conv.is_empty() {
                    chain.upload_at(&g_slot.kv_conv, 0, &c_slot.k_conv);
                    chain.upload_at(&g_slot.kv_conv, c_slot.k_conv.len(), &c_slot.v_conv);
                    chain.upload(&g_slot.attn_post_conv, &c_slot.attn_post_conv);
                    chain.upload(&g_slot.ffn_post_conv, &c_slot.ffn_post_conv);
                }
            }
            ds.ckpt_pos = None;
            st.pos = target_pos;
            return true;
        }
        // CPU path: verify every slot still holds the rows needed at `target_pos` before mutating.
        for s in &st.slots {
            let need_first = if s.is_swa {
                target_pos.saturating_sub(swa)
            } else {
                0
            };
            if s.base_pos > need_first || s.base_pos + s.stored_rows < target_pos {
                return false;
            }
        }
        for (s, c_slot) in st.slots.iter_mut().zip(&ckpt.slots) {
            let keep_rows = target_pos - s.base_pos;
            s.stored_rows = keep_rows;
            s.k_cache.truncate(keep_rows * s.kv_dim);
            s.v_cache.truncate(keep_rows * s.kv_dim);
            s.k_conv.set_hist(&c_slot.k_conv);
            s.v_conv.set_hist(&c_slot.v_conv);
            s.attn_post_conv.set_hist(&c_slot.attn_post_conv);
            s.ffn_post_conv.set_hist(&c_slot.ffn_post_conv);
        }
        st.pos = target_pos;
        true
    }

    /// Capture a complete, self-contained KV + conv snapshot at the current position (GPU or CPU)
    /// without mutating the live inference state.
    pub fn capture_state_snapshot(&self) -> Result<Inkling2StateSnapshot> {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let pos = st.pos;
        let swa = self.cfg.sliding_window;
        let past = self.cfg.conv_kernel.saturating_sub(1);
        let hidden = self.cfg.hidden;
        let scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        if let (Some(chain), Some(ds)) = (
            self.backend.chain(),
            scratch_guard
                .as_ref()
                .and_then(|s| s.decode_state.as_ref())
                .filter(|ds| ds.synced),
        ) {
            return Ok(self.read_gpu_snapshot(chain, ds, pos, None, swa, past, hidden));
        }
        Ok(st.capture_snapshot(hidden, self.cfg.conv_kernel, swa))
    }

    /// Extract a complete KV + conv snapshot at an earlier [`Inkling2ConvSnapshot`] position
    /// after prefill/decode has moved past it, as long as SWA ring rows at `ckpt.pos` are still
    /// live. Returns `None` if `ckpt.pos` has been overwritten.
    pub fn capture_snapshot_at_conv_checkpoint(
        &self,
        ckpt: &Inkling2ConvSnapshot,
    ) -> Option<Inkling2StateSnapshot> {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let target_pos = ckpt.pos;
        if target_pos == 0 || target_pos > st.pos || ckpt.slots.len() != st.slots.len() {
            return None;
        }
        let decoded = st.pos - target_pos;
        let swa = self.cfg.sliding_window;
        let past = self.cfg.conv_kernel.saturating_sub(1);
        let hidden = self.cfg.hidden;
        let scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        if let (Some(chain), Some(ds)) = (
            self.backend.chain(),
            scratch_guard
                .as_ref()
                .and_then(|s| s.decode_state.as_ref())
                .filter(|ds| ds.synced),
        ) {
            if ds.slots.len() != ckpt.slots.len() {
                return None;
            }
            if ds
                .slots
                .iter()
                .any(|s| s.is_swa && decoded > s.kv_cap.saturating_sub(swa))
            {
                return None;
            }
            return Some(self.read_gpu_snapshot(chain, ds, target_pos, Some(ckpt), swa, past, hidden));
        }
        // CPU path: verify every slot holds `[k_first..target_pos)` intact.
        for s in &st.slots {
            let need_first = if s.is_swa {
                target_pos.saturating_sub(swa)
            } else {
                0
            };
            if s.base_pos > need_first || s.base_pos + s.stored_rows < target_pos {
                return None;
            }
        }
        let slots = st
            .slots
            .iter()
            .zip(&ckpt.slots)
            .map(|(s, c)| {
                let k_first = if s.is_swa {
                    target_pos.saturating_sub(swa)
                } else {
                    0
                };
                let count = target_pos.saturating_sub(k_first);
                let row0 = k_first - s.base_pos;
                let off = row0 * s.kv_dim;
                let len = count * s.kv_dim;
                Inkling2SlotSnapshot {
                    is_swa: s.is_swa,
                    kv_dim: s.kv_dim,
                    base_pos: k_first,
                    stored_rows: count,
                    k_cache: s.k_cache[off..off + len].to_vec(),
                    v_cache: s.v_cache[off..off + len].to_vec(),
                    k_conv: c.k_conv.clone(),
                    v_conv: c.v_conv.clone(),
                    attn_post_conv: c.attn_post_conv.clone(),
                    ffn_post_conv: c.ffn_post_conv.clone(),
                }
            })
            .collect();
        Some(Inkling2StateSnapshot {
            pos: target_pos,
            hidden,
            conv_kernel: self.cfg.conv_kernel,
            sliding_window: swa,
            slots,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn read_gpu_snapshot(
        &self,
        chain: &dyn DeviceChain,
        ds: &GpuDecodeState,
        pos: usize,
        conv_override: Option<&Inkling2ConvSnapshot>,
        swa: usize,
        past: usize,
        hidden: usize,
    ) -> Inkling2StateSnapshot {
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let mut plan: Vec<(usize, usize, usize)> = Vec::with_capacity(ds.slots.len());
        for g_slot in &ds.slots {
            let kv_dim = g_slot.kv_dim;
            let k_first = if g_slot.is_swa {
                pos.saturating_sub(swa)
            } else {
                0
            };
            let count = pos.saturating_sub(k_first);
            let ring_start = if count > 0 { k_first % g_slot.kv_cap } else { 0 };
            let first_part = count.min(g_slot.kv_cap - ring_start);
            plan.push((k_first, count, first_part));

            if conv_override.is_none() && past > 0 {
                let c_len = past * kv_dim;
                rec.read_range(&g_slot.kv_conv, 0, c_len);
                rec.read_range(&g_slot.kv_conv, c_len, c_len);
                rec.read_range(&g_slot.attn_post_conv, 0, past * hidden);
                rec.read_range(&g_slot.ffn_post_conv, 0, past * hidden);
            }
            if count > 0 {
                let off_a = ring_start * kv_dim;
                let len_a = first_part * kv_dim;
                let v_base = g_slot.kv_cap * kv_dim;
                rec.read_range(&g_slot.kv_cache, off_a, len_a);
                rec.read_range(&g_slot.kv_cache, v_base + off_a, len_a);
                if first_part < count {
                    let rem = (count - first_part) * kv_dim;
                    rec.read_range(&g_slot.kv_cache, 0, rem);
                    rec.read_range(&g_slot.kv_cache, v_base, rem);
                }
            }
        }
        let mut reads = rec.finish().into_iter();
        let slots = ds
            .slots
            .iter()
            .enumerate()
            .map(|(il, g_slot)| {
                let (k_first, count, first_part) = plan[il];
                let (k_conv, v_conv, attn_post_conv, ffn_post_conv) = if let Some(ov) = conv_override {
                    let c = &ov.slots[il];
                    (
                        c.k_conv.clone(),
                        c.v_conv.clone(),
                        c.attn_post_conv.clone(),
                        c.ffn_post_conv.clone(),
                    )
                } else if past > 0 {
                    (
                        reads.next().expect("k_conv"),
                        reads.next().expect("v_conv"),
                        reads.next().expect("attn_post_conv"),
                        reads.next().expect("ffn_post_conv"),
                    )
                } else {
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                };
                let (k_cache, v_cache) = if count > 0 {
                    let mut k_buf = reads.next().expect("k_a");
                    let mut v_buf = reads.next().expect("v_a");
                    if first_part < count {
                        k_buf.extend_from_slice(&reads.next().expect("k_b"));
                        v_buf.extend_from_slice(&reads.next().expect("v_b"));
                    }
                    (k_buf, v_buf)
                } else {
                    (Vec::new(), Vec::new())
                };
                Inkling2SlotSnapshot {
                    is_swa: g_slot.is_swa,
                    kv_dim: g_slot.kv_dim,
                    base_pos: k_first,
                    stored_rows: count,
                    k_cache,
                    v_cache,
                    k_conv,
                    v_conv,
                    attn_post_conv,
                    ffn_post_conv,
                }
            })
            .collect();
        Inkling2StateSnapshot {
            pos,
            hidden,
            conv_kernel: self.cfg.conv_kernel,
            sliding_window: swa,
            slots,
        }
    }

    /// Validate and restore a complete [`Inkling2StateSnapshot`] into live host + GPU state.
    /// Rejects incompatible snapshots before mutating the live model state.
    pub fn restore_state_snapshot(&self, snap: &Inkling2StateSnapshot) -> Result<()> {
        let n_slots = self.cfg.n_logical_slots();
        let past = self.cfg.conv_kernel.saturating_sub(1);
        let bad = |msg: &str| Err(LlamaError::Config(format!("inkling2 snapshot: {msg}")));
        if snap.pos == 0 || snap.pos > self.cfg.context_length {
            return bad("position out of range");
        }
        if snap.hidden != self.cfg.hidden
            || snap.conv_kernel != self.cfg.conv_kernel
            || snap.sliding_window != self.cfg.sliding_window
            || snap.slots.len() != n_slots
        {
            return bad("model config mismatch");
        }
        for (il, slot) in snap.slots.iter().enumerate() {
            let ip = self.cfg.layer_phys(il);
            let want_swa = self.cfg.is_swa[ip];
            let want_kv_dim = self.cfg.kv_head_counts[ip] * self.cfg.head_dims[ip];
            let want_first = if want_swa {
                snap.pos.saturating_sub(self.cfg.sliding_window)
            } else {
                0
            };
            let want_rows = snap.pos - want_first;
            if slot.is_swa != want_swa
                || slot.kv_dim != want_kv_dim
                || slot.base_pos != want_first
                || slot.stored_rows != want_rows
                || slot.k_cache.len() != want_rows * want_kv_dim
                || slot.v_cache.len() != want_rows * want_kv_dim
                || slot.k_conv.len() != past * want_kv_dim
                || slot.v_conv.len() != past * want_kv_dim
                || slot.attn_post_conv.len() != past * self.cfg.hidden
                || slot.ffn_post_conv.len() != past * self.cfg.hidden
            {
                return bad("slot shape mismatch");
            }
        }

        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.apply_snapshot(snap);
        let mut scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        if self.full_gpu_decode {
            if let Some(chain) = self.backend.chain() {
                let scratch = scratch_guard.get_or_insert_with(GpuScratch::default);
                let ds = scratch
                    .decode_state
                    .get_or_insert_with(|| self.init_gpu_decode_state(chain, st.pos, 1));
                ds.ckpt_pos = None;
                self.sync_host_to_gpu(chain, &st, ds);
            }
        } else if let Some(scratch) = scratch_guard.as_mut() {
            scratch.reset_decode_state();
        }
        Ok(())
    }

    /// Forward pass returning last-token, all-token, or empty logits depending on `out_mode`.
    pub(crate) fn forward_internal(
        &self,
        tokens: &[u32],
        soft: &[(usize, &[f32])],
        state: &mut Inkling2State,
        out_mode: OutputMode,
    ) -> Result<Tensor> {
        let seq = tokens.len();
        let hidden = self.cfg.hidden;
        if seq == 0 {
            return Ok(Tensor::from_vec(Vec::new(), vec![0, self.cfg.vocab_size_unpadded]));
        }

        let mut scratch_guard = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner());
        let scratch = scratch_guard.get_or_insert_with(GpuScratch::default);
        let mut prof = StepProfile {
            enabled: profile_enabled(),
            ..Default::default()
        };
        let t_step_start = if prof.enabled { Some(Instant::now()) } else { None };

        // 1. Token embeddings + embed_norm (IMPORTANT: spliced vision embeddings bypass embed_norm!).
        let mut cur = vec![0.0f32; seq * hidden];
        let mut is_soft = vec![false; seq];
        for &(start, rows) in soft {
            let n_rows = rows.len() / hidden;
            for j in 0..n_rows {
                let pos = start + j;
                if pos < seq {
                    cur[pos * hidden..(pos + 1) * hidden]
                        .copy_from_slice(&rows[j * hidden..(j + 1) * hidden]);
                    is_soft[pos] = true;
                }
            }
        }
        if seq >= 64 {
            cur.par_chunks_mut(hidden)
                .enumerate()
                .try_for_each(|(t, dst)| -> Result<()> {
                    if !is_soft[t] {
                        let raw = self.embed_row(tokens[t])?;
                        let sum_sq: f32 = raw.iter().map(|&v| v * v).sum();
                        let inv = 1.0 / (sum_sq / hidden as f32 + self.cfg.rms_eps).sqrt();
                        for c in 0..hidden {
                            dst[c] = raw[c] * inv * self.embed_norm[c];
                        }
                    }
                    Ok(())
                })?;
        } else {
            for t in 0..seq {
                if !is_soft[t] {
                    let raw = self.embed_row(tokens[t])?;
                    let sum_sq: f32 = raw.iter().map(|&v| v * v).sum();
                    let inv = 1.0 / (sum_sq / hidden as f32 + self.cfg.rms_eps).sqrt();
                    let dst = &mut cur[t * hidden..(t + 1) * hidden];
                    for c in 0..hidden {
                        dst[c] = raw[c] * inv * self.embed_norm[c];
                    }
                }
            }
        }

        let past_len = state.pos;
        let tau: Vec<f32> = (0..seq)
            .map(|t| {
                let pos = past_len + t;
                let ratio = ((pos + 1) as f32 / self.cfg.log_scale_floor as f32).max(1.0);
                1.0 + self.cfg.log_scale_alpha * ratio.ln()
            })
            .collect();

        // Single-submit full-GPU decode (`seq == 1`, no soft vision tokens, last-token logits):
        // runs all 48 logical layers + output unembed + softcap in ONE `DeviceChain` submission
        // with `ChainRecipe` replay and Split-K attention.
        if self.full_gpu_decode && seq == 1 && soft.is_empty() && out_mode == OutputMode::LastOnly {
            return self.forward_decode_gpu(&cur, tau[0], state, scratch, t_step_start);
        }

        // Single-submit full-GPU batched prefill (`seq >= 1`, supports soft vision tokens and
        // `OutputMode::{None, LastOnly, All}`): keeps GPU KV + conv histories live for zero-stall
        // transition into `forward_decode_gpu`!
        if self.full_gpu_decode && seq <= self.max_gpu_prefill_seq() {
            return self.forward_prefill_gpu(&cur, seq, state, scratch, out_mode, t_step_start);
        }

        // If GPU decode/prefill advanced ahead of host `state`, pull GPU KV + conv histories back
        // before running the CPU/partial-GPU fallback path.
        if let (Some(chain), Some(ds)) = (self.backend.chain(), scratch.decode_state.as_mut()) {
            if ds.host_dirty && state.pos > 0 {
                self.sync_gpu_to_host(chain, state, ds);
            }
            ds.synced = false;
            ds.host_dirty = false;
            ds.recipe = None;
        }

        // `blocks` stores block residuals ALREADY normalized with unweighted RMSNorm (eps = 1e-6),
        // because they are immutable once pushed and only consumed by `mix_depth`.
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let n_phys = self.cfg.n_phys_layers;

        if let Some(lc) = &self.cfg.loop_cfg {
            // Prelude: physical layers 0..core_start
            for il in 0..lc.core_start {
                cur = self.run_layer(il, il, cur, &mut blocks, None, None, &tau, past_len, seq, state, scratch, &mut prof)?;
            }

            // Looped core: `num_passes` passes over `core_start..core_end`
            let prelude = cur.clone();
            let prelude_blocks = blocks.clone();
            let core_len = lc.core_end - lc.core_start;

            for pass in 0..lc.num_passes {
                let mut bank: Vec<Vec<f32>> = Vec::new();
                let mut bank_out_normed: Option<Vec<f32>> = None;
                if pass > 0 {
                    // Bank = blocks produced after the first core block + previous pass's output (normalized once).
                    let skip = (prelude_blocks.len() + 1).min(blocks.len());
                    bank = blocks[skip..].to_vec();
                    bank_out_normed = Some(rms_norm_no_weight(&cur, hidden, 1e-6));
                    cur = prelude.clone();
                    blocks = prelude_blocks.clone();
                }

                for ip in lc.core_start..lc.core_end {
                    let il = if pass == 0 {
                        ip
                    } else {
                        n_phys + (pass - 1) * core_len + (ip - lc.core_start)
                    };
                    if pass == 0 {
                        cur = self.run_layer(il, ip, cur, &mut blocks, None, None, &tau, past_len, seq, state, scratch, &mut prof)?;
                    } else {
                        let loop_layer = &self.loop_layers[ip - lc.core_start];
                        let messages = mix_depth_impl(
                            &bank,
                            bank_out_normed.as_deref().expect("bank_out present on pass > 0"),
                            true,
                            &loop_layer.msg_query,
                            4,
                            seq,
                            hidden,
                        );
                        cur = self.run_layer(
                            il,
                            ip,
                            cur,
                            &mut blocks,
                            Some(&messages),
                            Some(loop_layer),
                            &tau,
                            past_len,
                            seq,
                            state,
                            scratch,
                            &mut prof,
                        )?;
                    }
                }
            }

            // Postlude: physical layers core_end..n_phys
            for il in lc.core_end..n_phys {
                cur = self.run_layer(il, il, cur, &mut blocks, None, None, &tau, past_len, seq, state, scratch, &mut prof)?;
            }
        } else {
            // Non-looped `inkling2`: sequential layers 0..n_phys
            for il in 0..n_phys {
                cur = self.run_layer(il, il, cur, &mut blocks, None, None, &tau, past_len, seq, state, scratch, &mut prof)?;
            }
        }

        state.pos += seq;

        // Output AttnRes + RMSNorm
        let unpadded = self.cfg.vocab_size_unpadded;
        if out_mode == OutputMode::None {
            if let Some(t0) = t_step_start {
                let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[inkling2-prof] seq={seq} pos={past_len} total={total_ms:.1}ms | res_ctrl={:.1}ms attn_proj={:.1}ms attn_core={:.1}ms wo={:.1}ms ffn_pre={:.1}ms experts={:.1}ms output=0.0ms",
                    prof.attn_res_ctrl_us as f64 / 1000.0,
                    prof.attn_proj_us as f64 / 1000.0,
                    prof.attn_core_us as f64 / 1000.0,
                    prof.attn_wo_us as f64 / 1000.0,
                    prof.ffn_pre_us as f64 / 1000.0,
                    prof.experts_us as f64 / 1000.0,
                );
            }
            return Ok(Tensor::from_vec(Vec::new(), vec![0, unpadded]));
        }
        let t_out = if prof.enabled { Some(Instant::now()) } else { None };
        let final_mixed = mix_depth(&blocks, &cur, &self.output_res_query, 1, seq, hidden)
            .into_iter()
            .next()
            .expect("1 output query");
        let normed = rms_norm_rows(&final_mixed, &self.output_norm, hidden, self.cfg.rms_eps);

        let (head_in, out_rows) = if out_mode == OutputMode::LastOnly {
            (&normed[(seq - 1) * hidden..seq * hidden], 1usize)
        } else {
            (&normed[..], seq)
        };

        // Unembed (only computing/reading `unpadded` vocab rows) -> in-place softcap
        let mut logits = linear_host_fast(
            self.backend.as_ref(),
            scratch,
            &self.output,
            head_in,
            out_rows,
            hidden,
            unpadded,
        );
        let cap = self.cfg.final_logit_softcap;
        if cap > 0.0 {
            let inv_cap = 1.0 / cap;
            for v in &mut logits {
                *v = cap * (*v * inv_cap).tanh();
            }
        }
        if let (Some(t0), Some(to)) = (t_step_start, t_out) {
            prof.output_us += to.elapsed().as_micros() as u64;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[inkling2-prof] seq={seq} pos={past_len} total={total_ms:.1}ms | res_ctrl={:.1}ms attn_proj={:.1}ms attn_core={:.1}ms wo={:.1}ms ffn_pre={:.1}ms experts={:.1}ms output={:.1}ms",
                prof.attn_res_ctrl_us as f64 / 1000.0,
                prof.attn_proj_us as f64 / 1000.0,
                prof.attn_core_us as f64 / 1000.0,
                prof.attn_wo_us as f64 / 1000.0,
                prof.ffn_pre_us as f64 / 1000.0,
                prof.experts_us as f64 / 1000.0,
                prof.output_us as f64 / 1000.0,
            );
        }
        Ok(Tensor::from_vec(logits, vec![out_rows, unpadded]))
    }

    fn init_gpu_decode_state(
        &self,
        chain: &dyn DeviceChain,
        initial_pos: usize,
        initial_seq: usize,
    ) -> GpuDecodeState {
        let hidden = self.cfg.hidden;
        let past_conv = self.cfg.conv_kernel.saturating_sub(1);
        let n_slots = self.cfg.n_logical_slots();
        let full_init_cap = (initial_pos + initial_seq.max(1)).next_power_of_two().max(2048);
        let swa_init_cap = self.cfg.sliding_window + self.preferred_prefill_chunk().max(2048).max(initial_seq);
        let mut slots = Vec::with_capacity(n_slots);
        let mut max_q_dim = 1usize;
        let mut max_heads = 1usize;
        for il in 0..n_slots {
            let ip = self.cfg.layer_phys(il);
            let is_swa = self.cfg.is_swa[ip];
            let q_dim = self.cfg.head_counts[ip] * self.cfg.head_dims[ip];
            let kv_dim = self.cfg.kv_head_counts[ip] * self.cfg.head_dims[ip];
            max_q_dim = max_q_dim.max(q_dim);
            max_heads = max_heads.max(self.cfg.head_counts[ip]);
            let kv_cap = if is_swa { swa_init_cap } else { full_init_cap };
            slots.push(GpuSlotState {
                is_swa,
                kv_dim,
                kv_cap,
                kv_cache: chain.vec(2 * kv_cap * kv_dim),
                kv_conv: chain.vec((2 * past_conv * kv_dim).max(1)),
                attn_post_conv: chain.vec((past_conv * hidden).max(1)),
                ffn_post_conv: chain.vec((past_conv * hidden).max(1)),
            });
        }
        let gate_rank = self.cfg.loop_cfg.as_ref().map_or(1, |l| l.gate_rank);
        GpuDecodeState {
            synced: false,
            host_dirty: false,
            recipe: None,
            slots,
            ckpt_pos: None,
            ckpt_slots: Vec::new(),
            step_params: chain.vec(4),
            attn_split: chain.vec(max_heads * 16 * 132),
            cur: chain.vec(hidden),
            prelude_cur: chain.vec(hidden),
            blocks: chain.vec(32 * hidden),
            bank: chain.vec(32 * hidden),
            bank_out: chain.vec(hidden),
            msgs: chain.vec(4 * hidden),
            mix_out: chain.vec(3 * hidden),
            mid_qkv: chain.vec(3 * gate_rank),
            ctrl_qkv: chain.vec(3 * hidden),
            norm_qkv: chain.vec(3 * hidden),
            ffn_inp: chain.vec(hidden),
            ffn_x: chain.vec(hidden),
            attn_gated: chain.vec(max_q_dim),
            attn_proj: chain.vec(hidden),
            attn_conv: chain.vec(hidden),
            logits: chain.vec(self.cfg.vocab_size_padded),
        }
    }

    fn sync_host_to_gpu(&self, chain: &dyn DeviceChain, state: &Inkling2State, ds: &mut GpuDecodeState) {
        let swa = self.cfg.sliding_window;
        for (il, g_slot) in ds.slots.iter_mut().enumerate() {
            let h_slot = &state.slots[il];
            let kv_dim = g_slot.kv_dim;
            if !g_slot.is_swa && state.pos >= g_slot.kv_cap {
                let new_cap = (state.pos + 1).next_power_of_two().max(2048);
                g_slot.kv_cache = chain.vec(2 * new_cap * kv_dim);
                g_slot.kv_cap = new_cap;
            }
            let kh = h_slot.k_conv.hist();
            if !kh.is_empty() {
                chain.upload_at(&g_slot.kv_conv, 0, kh);
                chain.upload_at(&g_slot.kv_conv, kh.len(), h_slot.v_conv.hist());
                chain.upload(&g_slot.attn_post_conv, h_slot.attn_post_conv.hist());
                chain.upload(&g_slot.ffn_post_conv, h_slot.ffn_post_conv.hist());
            }
            let k_first = if g_slot.is_swa {
                state.pos.saturating_sub(swa)
            } else {
                0
            }
            .max(h_slot.base_pos);
            let k_end = state.pos.min(h_slot.base_pos + h_slot.stored_rows);
            if k_first < k_end {
                let count = k_end - k_first;
                let src_row0 = k_first - h_slot.base_pos;
                let ring_start = k_first % g_slot.kv_cap;
                let first_part = count.min(g_slot.kv_cap - ring_start);
                let src_a = src_row0 * kv_dim;
                let len_a = first_part * kv_dim;
                let v_base = g_slot.kv_cap * kv_dim;
                chain.upload_at(&g_slot.kv_cache, ring_start * kv_dim, &h_slot.k_cache[src_a..src_a + len_a]);
                chain.upload_at(&g_slot.kv_cache, v_base + ring_start * kv_dim, &h_slot.v_cache[src_a..src_a + len_a]);
                if first_part < count {
                    let rem = (count - first_part) * kv_dim;
                    chain.upload_at(&g_slot.kv_cache, 0, &h_slot.k_cache[src_a + len_a..src_a + len_a + rem]);
                    chain.upload_at(&g_slot.kv_cache, v_base, &h_slot.v_cache[src_a + len_a..src_a + len_a + rem]);
                }
            }
        }
        ds.synced = true;
        ds.host_dirty = false;
        ds.recipe = None;
    }

    fn sync_gpu_to_host(&self, chain: &dyn DeviceChain, state: &mut Inkling2State, ds: &GpuDecodeState) {
        let swa = self.cfg.sliding_window;
        let past_conv = self.cfg.conv_kernel.saturating_sub(1);
        let hidden = self.cfg.hidden;

        let mut rec = chain.begin();
        rec.keep_groups(false);
        let mut plan: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(ds.slots.len());
        for g_slot in &ds.slots {
            let kv_dim = g_slot.kv_dim;
            let k_first = if g_slot.is_swa { state.pos.saturating_sub(swa) } else { 0 };
            let count = state.pos.saturating_sub(k_first);
            let ring_start = if count > 0 { k_first % g_slot.kv_cap } else { 0 };
            let first_part = count.min(g_slot.kv_cap - ring_start);
            plan.push((k_first, count, ring_start, first_part));

            if past_conv > 0 {
                let c_len = past_conv * kv_dim;
                rec.read_range(&g_slot.kv_conv, 0, c_len);
                rec.read_range(&g_slot.kv_conv, c_len, c_len);
                rec.read_range(&g_slot.attn_post_conv, 0, past_conv * hidden);
                rec.read_range(&g_slot.ffn_post_conv, 0, past_conv * hidden);
            }
            if count > 0 {
                let off_a = ring_start * kv_dim;
                let len_a = first_part * kv_dim;
                let v_base = g_slot.kv_cap * kv_dim;
                rec.read_range(&g_slot.kv_cache, off_a, len_a);
                rec.read_range(&g_slot.kv_cache, v_base + off_a, len_a);
                if first_part < count {
                    let rem = (count - first_part) * kv_dim;
                    rec.read_range(&g_slot.kv_cache, 0, rem);
                    rec.read_range(&g_slot.kv_cache, v_base, rem);
                }
            }
        }
        let mut reads = rec.finish().into_iter();
        for (il, &(k_first, count, _, first_part)) in plan.iter().enumerate() {
            let h_slot = &mut state.slots[il];
            if past_conv > 0 {
                h_slot.k_conv.set_hist(&reads.next().expect("k_conv"));
                h_slot.v_conv.set_hist(&reads.next().expect("v_conv"));
                h_slot.attn_post_conv.set_hist(&reads.next().expect("attn_post_conv"));
                h_slot.ffn_post_conv.set_hist(&reads.next().expect("ffn_post_conv"));
            }
            h_slot.base_pos = k_first;
            h_slot.stored_rows = count;
            h_slot.k_cache.clear();
            h_slot.v_cache.clear();
            if count > 0 {
                let mut k_buf = reads.next().expect("k_a");
                let mut v_buf = reads.next().expect("v_a");
                if first_part < count {
                    k_buf.extend_from_slice(&reads.next().expect("k_b"));
                    v_buf.extend_from_slice(&reads.next().expect("v_b"));
                }
                h_slot.k_cache = k_buf;
                h_slot.v_cache = v_buf;
            }
        }
    }

    /// Record one logical layer into an active single-submit full-GPU decode recording.
    #[allow(clippy::too_many_arguments)]
    fn record_decode_layer_gpu(
        &self,
        rec: &mut dyn ChainRecorder,
        il: usize,
        ip: usize,
        n_blocks: &mut usize,
        has_msgs: bool,
        loop_layer: Option<&Inkling2LoopLayer>,
        ds: &GpuDecodeState,
        x_bufs: &[DeviceVec],
        mid_bufs: &[DeviceVec],
        ctrl_gain: &[DeviceVec],
        ctrl_forget: &[DeviceVec],
        ctrl_out: &[DeviceVec],
        norm_bufs: &[DeviceVec],
        y_bufs: &[DeviceVec],
        router_buf: &DeviceVec,
    ) {
        let layer = &self.layers[ip];
        let dev = layer.decode_dev.as_ref().expect("LayerDecodeDev present");
        let norms_dev = layer.attn_norms_dev.as_ref().expect("attn_norms_dev present");
        let ffn_norm_dev = layer.ffn_norm_dev.as_ref().expect("ffn_norm_dev present");
        let router_dev = layer.ffn_gate_inp_padded_dev.as_ref().expect("router_dev present");
        let qmoe = layer.experts.quant_moe().expect("QuantMoe present");
        let slot = &ds.slots[il];

        let hidden = self.cfg.hidden;
        let is_swa = layer.is_swa;
        let n_heads = layer.n_heads;
        let n_kv_heads = layer.n_kv_heads;
        let head_dim = layer.head_dim;
        let q_dim = n_heads * head_dim;
        let d_rel = self.cfg.d_rel;
        let rel_extent = if is_swa { self.cfg.sliding_window } else { self.cfg.rel_extent };
        let conv_k = self.cfg.conv_kernel;
        let n_exp = self.cfg.n_experts;
        let gate_rank = self.cfg.loop_cfg.as_ref().map_or(0, |l| l.gate_rank);

        // 1. Attention input via Block Attention Residuals (`mix_depth`).
        if let Some(ref q_dev) = dev.attn_res_query {
            let nq = if is_swa { 1 } else { 3 };
            rec.inkling2_mix_depth(&ds.blocks, &ds.cur, q_dev, &ds.mix_out, *n_blocks, nq, hidden, false);
        } else {
            rec.copy(&ds.cur, 0, &ds.mix_out, 0, hidden);
            if !is_swa {
                rec.copy(&ds.cur, 0, &ds.mix_out, hidden, hidden);
                rec.copy(&ds.cur, 0, &ds.mix_out, 2 * hidden, hidden);
            }
        }

        // 2. Optional Q/K/V cross-pass controllers + attention pre-norms (batched 3-stream Q/K/V
        //    on quantized attention weights: 3 dispatches instead of 9, or 1 instead of 5 for Separate!).
        let has_ctrl = has_msgs && loop_layer.is_some();
        let shared_x = matches!(norms_dev, AttnNormsDev::Shared(_)) && !has_ctrl;
        let (k_idx, v_idx) = if shared_x { (0usize, 0usize) } else { (1usize, 2usize) };

        let is_q8_split_w = |w: &Weight| -> bool {
            matches!(
                w,
                Weight::Quant(q)
                    if matches!(
                        q.dtype(),
                        ggml_quants::GgmlType::Q8_0
                            | ggml_quants::GgmlType::Q4_0
                            | ggml_quants::GgmlType::IQ4_NL
                            | ggml_quants::GgmlType::Q4_K
                            | ggml_quants::GgmlType::Q5_K
                            | ggml_quants::GgmlType::Q6_K
                            | ggml_quants::GgmlType::Q3_K
                    )
            )
        };
        let q_stream_q8 = if let Some(ref wqrsg) = layer.wqrsg {
            is_q8_split_w(wqrsg)
        } else {
            is_q8_split_w(&layer.wq)
                && is_q8_split_w(&layer.attn_rel)
                && is_q8_split_w(&layer.attn_sink_proj)
                && is_q8_split_w(&layer.attn_gate)
        };
        let kv_stream_q8 = if let (Some(ref wkd), Some(ref wvd)) = (&layer.wk_dyn, &layer.wv_dyn) {
            is_q8_split_w(wkd) && is_q8_split_w(wvd)
        } else {
            is_q8_split_w(&layer.wk)
                && is_q8_split_w(&layer.wv)
                && layer.attn_k_conv_proj.as_ref().is_none_or(&is_q8_split_w)
                && layer.attn_v_conv_proj.as_ref().is_none_or(&is_q8_split_w)
        };
        let all_qkv_q8 = q_stream_q8 && kv_stream_q8;

        let mut use_norm_qkv = false;
        if let (true, Some(ll)) = (has_msgs, loop_layer) {
            let (norm_w_multi, sep_w) = match norms_dev {
                AttnNormsDev::Shared(w) => (w, false),
                AttnNormsDev::Separate { qkv, .. } => (qkv, true),
            };
            if let (Some(dqkv), Some(gfqkv)) = (&ll.ctrl_down_qkv_f16, &ll.ctrl_gf_qkv_f16) {
                let cur_stride = if is_swa { 0 } else { hidden };
                rec.inkling2_ctrl_cat_down_multi(dqkv, &ds.mix_out, 0, cur_stride, &ds.msgs, 0, &ds.mid_qkv, hidden, gate_rank, 3);
                rec.inkling2_ctrl_up_blend_multi(gfqkv, &ds.mid_qkv, &ds.mix_out, 0, cur_stride, &ds.msgs, 0, &ds.ctrl_qkv, hidden, gate_rank, 3);
                if all_qkv_q8
                    && rec.rmsnorm_multi_q8(&ds.ctrl_qkv, norm_w_multi, &ds.norm_qkv, 3, hidden, sep_w, self.cfg.rms_eps)
                {
                    use_norm_qkv = true;
                } else {
                    let norm_ws: [&DeviceVec; 3] = match norms_dev {
                        AttnNormsDev::Shared(w) => [w, w, w],
                        AttnNormsDev::Separate { q, k, v, .. } => [q, k, v],
                    };
                    for i in 0..3 {
                        rec.copy(&ds.ctrl_qkv, i * hidden, &ctrl_out[i], 0, hidden);
                    }
                    for i in 0..3 {
                        let in_v = GpuScratch::view(&ctrl_out[i], hidden);
                        let out_v = GpuScratch::view(&norm_bufs[i], hidden);
                        rec.rmsnorm(&in_v, norm_ws[i], &out_v, self.cfg.rms_eps);
                    }
                }
            } else {
                let norm_ws: [&DeviceVec; 3] = match norms_dev {
                    AttnNormsDev::Shared(w) => [w, w, w],
                    AttnNormsDev::Separate { q, k, v, .. } => [q, k, v],
                };
                for i in 0..3 {
                    let cur_off = if is_swa { 0 } else { i * hidden };
                    let msg_off = i * hidden;
                    rec.inkling2_control_cat(&ds.mix_out, cur_off, &ds.msgs, msg_off, &x_bufs[i], hidden);
                    record_weight(rec, &ll.ctrl_down[i], &x_bufs[i], &mid_bufs[i], 1);
                }
                for i in 0..3 {
                    let cur_off = if is_swa { 0 } else { i * hidden };
                    let msg_off = i * hidden;
                    record_weight(rec, &ll.ctrl_gain[i], &mid_bufs[i], &ctrl_gain[i], 1);
                    record_weight(rec, &ll.ctrl_forget[i], &mid_bufs[i], &ctrl_forget[i], 1);
                    rec.control_blend_at(&ds.mix_out, cur_off, &ds.msgs, msg_off, &ctrl_gain[i], &ctrl_forget[i], &ctrl_out[i], hidden);
                }
                for i in 0..3 {
                    let in_v = GpuScratch::view(&ctrl_out[i], hidden);
                    let out_v = GpuScratch::view(&norm_bufs[i], hidden);
                    rec.rmsnorm(&in_v, norm_ws[i], &out_v, self.cfg.rms_eps);
                }
            }
        } else {
            match norms_dev {
                AttnNormsDev::Shared(w) => {
                    let in_v = GpuScratch::view(&ds.mix_out, hidden);
                    let out_v = GpuScratch::view(&norm_bufs[0], hidden);
                    rec.rmsnorm(&in_v, w, &out_v, self.cfg.rms_eps);
                }
                AttnNormsDev::Separate { q, k, v, qkv } => {
                    if all_qkv_q8
                        && rec.rmsnorm_multi_q8(&ds.mix_out, qkv, &ds.norm_qkv, 3, hidden, true, self.cfg.rms_eps)
                    {
                        use_norm_qkv = true;
                    } else {
                        rec.copy(&ds.mix_out, hidden, &ctrl_out[1], 0, hidden);
                        rec.copy(&ds.mix_out, 2 * hidden, &ctrl_out[2], 0, hidden);
                        let in0 = GpuScratch::view(&ds.mix_out, hidden);
                        let out0 = GpuScratch::view(&norm_bufs[0], hidden);
                        rec.rmsnorm(&in0, q, &out0, self.cfg.rms_eps);
                        for (i, nw) in [(1usize, k), (2usize, v)] {
                            let in_v = GpuScratch::view(&ctrl_out[i], hidden);
                            let out_v = GpuScratch::view(&norm_bufs[i], hidden);
                            rec.rmsnorm(&in_v, nw, &out_v, self.cfg.rms_eps);
                        }
                    }
                }
            }
        }

        // 3. Attention projections: stacked Q-stream (`wqrsg`) + K-stream (`wk_dyn`) + V-stream (`wv_dyn`)
        //    so 6..8 separate Q8_0 matmuls collapse to 3 parallel dispatches per layer.
        let rec_attn_w = |rec: &mut dyn ChainRecorder, w: &Weight, stream_idx: usize, out: &DeviceVec| {
            if use_norm_qkv {
                if let Weight::Quant(qt) = w {
                    let ok = rec.matmul_row_of(qt, &ds.norm_qkv, 3, stream_idx, out);
                    debug_assert!(ok, "matmul_row_of must succeed when rmsnorm_multi_q8 succeeded");
                    return;
                }
            }
            let x_v = GpuScratch::view(&norm_bufs[stream_idx], hidden);
            record_weight(rec, w, &x_v, out, 1);
        };
        let packed_qrsg = if let Some(ref wqrsg) = layer.wqrsg {
            rec_attn_w(rec, wqrsg, 0, &y_bufs[0]);
            true
        } else {
            rec_attn_w(rec, &layer.wq, 0, &y_bufs[0]);
            rec_attn_w(rec, &layer.attn_rel, 0, &y_bufs[3]);
            rec_attn_w(rec, &layer.attn_sink_proj, 0, &y_bufs[4]);
            rec_attn_w(rec, &layer.attn_gate, 0, &y_bufs[5]);
            false
        };
        let (packed_dyn, dyn_k_v, dyn_v_v) = if let (Some(ref wkd), Some(ref wvd)) = (&layer.wk_dyn, &layer.wv_dyn) {
            rec_attn_w(rec, wkd, k_idx, &y_bufs[1]);
            rec_attn_w(rec, wvd, v_idx, &y_bufs[2]);
            (true, None, None)
        } else {
            rec_attn_w(rec, &layer.wk, k_idx, &y_bufs[1]);
            rec_attn_w(rec, &layer.wv, v_idx, &y_bufs[2]);
            let dk = if let Some(ref kp) = layer.attn_k_conv_proj {
                rec_attn_w(rec, kp, k_idx, &y_bufs[6]);
                Some(&y_bufs[6])
            } else {
                None
            };
            let dv = if let Some(ref vp) = layer.attn_v_conv_proj {
                rec_attn_w(rec, vp, v_idx, &y_bufs[7]);
                Some(&y_bufs[7])
            } else {
                None
            };
            (false, dk, dv)
        };

        // 4. Fused K/V conv+clip+norm+store + Fused Q/R clip+tau + GQA rel-bias sink attention
        //    + sigmoid gate (+ inline Q8 quantization of `attn_gated` for `wo` when packed).
        rec.inkling2_kv_conv_clip_store(
            &y_bufs[1],
            &y_bufs[2],
            &dev.kv_conv_w,
            dyn_k_v,
            dyn_v_v,
            packed_dyn,
            &ds.step_params,
            &slot.kv_conv,
            &slot.kv_cache,
            n_kv_heads,
            head_dim,
            conv_k,
            slot.kv_cap,
            self.cfg.qk_clip,
            self.cfg.rms_eps,
        );
        let fused_attn_ok = if packed_qrsg {
            rec.inkling2_attend_fused_packed(
                &y_bufs[0],
                &dev.attn_consts,
                &slot.kv_cache,
                &ds.attn_gated,
                &ds.step_params,
                Some(&ds.attn_split),
                n_heads,
                n_kv_heads,
                head_dim,
                d_rel,
                rel_extent,
                slot.kv_cap,
                if is_swa { self.cfg.sliding_window } else { 0 },
                self.cfg.qk_clip,
                self.cfg.rel_clip,
            )
        } else {
            rec.inkling2_attend_fused(
                &y_bufs[0],
                &y_bufs[3],
                &y_bufs[4],
                &dev.attn_consts,
                &slot.kv_cache,
                &y_bufs[5],
                &ds.attn_gated,
                &ds.step_params,
                n_heads,
                n_kv_heads,
                head_dim,
                d_rel,
                rel_extent,
                slot.kv_cap,
                if is_swa { self.cfg.sliding_window } else { 0 },
                self.cfg.qk_clip,
                self.cfg.rel_clip,
            )
        };
        debug_assert!(fused_attn_ok, "inkling2_attend_fused must succeed");

        // 5. Wo + post-attention short conv + block boundary push / fused residual add.
        let ag_v = GpuScratch::view(&ds.attn_gated, q_dim);
        record_weight(rec, &layer.wo, &ag_v, &ds.attn_proj, 1);
        if ip % self.cfg.attn_res_block_size == 0 {
            rec.inkling2_short_conv_step(&ds.attn_proj, &dev.attn_post_conv, None, &slot.attn_post_conv, &ds.attn_conv, hidden, conv_k, 0);
            rec.inkling2_block_push(&ds.cur, &ds.attn_conv, &ds.blocks, &ds.ffn_inp, hidden, Some(*n_blocks), false);
            *n_blocks += 1;
        } else {
            rec.inkling2_post_conv_add(&ds.attn_proj, &dev.attn_post_conv, &ds.cur, &slot.attn_post_conv, &ds.ffn_inp, hidden, conv_k);
        }

        // 6. MoE FFN: AttnRes -> optional fused FFN ctrl+RMSNorm -> exact-F32 router -> fused QuantMoe + post-FFN conv + residual.
        rec.inkling2_mix_depth(&ds.blocks, &ds.ffn_inp, &dev.ffn_res_query, &ds.ffn_x, *n_blocks, 1, hidden, false);
        let norm_v = GpuScratch::view(&norm_bufs[0], hidden);
        if let (true, Some(ll)) = (has_msgs, loop_layer) {
            if let Some(ref df) = ll.ctrl_down_ffn_f16 {
                rec.inkling2_ctrl_cat_down(df, &ds.ffn_x, 0, &ds.msgs, 3 * hidden, &mid_bufs[0], hidden, gate_rank);
            } else {
                rec.inkling2_control_cat(&ds.ffn_x, 0, &ds.msgs, 3 * hidden, &x_bufs[0], hidden);
                record_weight(rec, &ll.ctrl_down[3], &x_bufs[0], &mid_bufs[0], 1);
            }
            if let Some(ref gf) = ll.ctrl_gf_ffn_f16 {
                rec.inkling2_ctrl_up_blend(gf, &mid_bufs[0], &ds.ffn_x, 0, &ds.msgs, 3 * hidden, &ctrl_out[0], hidden, gate_rank);
            } else {
                record_weight(rec, &ll.ctrl_gain[3], &mid_bufs[0], &ctrl_gain[0], 1);
                record_weight(rec, &ll.ctrl_forget[3], &mid_bufs[0], &ctrl_forget[0], 1);
                rec.control_blend_at(&ds.ffn_x, 0, &ds.msgs, 3 * hidden, &ctrl_gain[0], &ctrl_forget[0], &ctrl_out[0], hidden);
            }
            let in_v = GpuScratch::view(&ctrl_out[0], hidden);
            // `rec.rmsnorm` runs `rmsnorm_q8`, producing BOTH exact-F32 `norm_v` (for router) and
            // cached Q8_0 activations (for `QuantMoe::gu`) in 1 dispatch!
            rec.rmsnorm(&in_v, ffn_norm_dev, &norm_v, self.cfg.rms_eps);
        } else {
            let in_v = GpuScratch::view(&ds.ffn_x, hidden);
            rec.rmsnorm(&in_v, ffn_norm_dev, &norm_v, self.cfg.rms_eps);
        }
        let router_v = GpuScratch::view(router_buf, n_exp + 1);
        rec.matmul_f32_rows(router_dev, n_exp + 1, hidden, &norm_v, &router_v, 1);
        if !rec.moe_routed_post_conv_add(
            qmoe,
            &norm_v,
            &router_v,
            &dev.ffn_post_conv,
            &ds.ffn_inp,
            &slot.ffn_post_conv,
            &ds.cur,
            self.cfg.n_experts_used,
            conv_k,
        ) {
            let moe_out_v = GpuScratch::view(&y_bufs[0], hidden);
            let routed_ok = rec.moe_routed(qmoe, &norm_v, &moe_out_v, &router_v, self.cfg.n_experts_used, 1);
            debug_assert!(routed_ok, "QuantMoe::record_routed must succeed for seq == 1");
            rec.inkling2_post_conv_add(&y_bufs[0], &dev.ffn_post_conv, &ds.ffn_inp, &slot.ffn_post_conv, &ds.cur, hidden, conv_k);
        }
    }

    /// Single-submit full-GPU decode for `seq == 1`: executes ALL 48 logical layers + output
    /// RMSNorm + unembed + softcap in ONE `DeviceChain` submission and reads back only logits.
    fn forward_decode_gpu(
        &self,
        cur_embed: &[f32],
        tau0: f32,
        state: &mut Inkling2State,
        scratch: &mut GpuScratch,
        t_step_start: Option<Instant>,
    ) -> Result<Tensor> {
        let chain = self.backend.chain().expect("full_gpu_decode requires DeviceChain");
        let hidden = self.cfg.hidden;
        let past_len = state.pos;
        let gate_rank = self.cfg.loop_cfg.as_ref().map_or(1, |l| l.gate_rank);
        let n_exp = self.cfg.n_experts;

        // Ensure fixed-size persistent decode scratch slots (never reallocate across tokens).
        let max_q_dim = self
            .cfg
            .head_counts
            .iter()
            .zip(&self.cfg.head_dims)
            .map(|(h, d)| h * d)
            .max()
            .unwrap_or(hidden);
        let max_kv_dim = self
            .cfg
            .kv_head_counts
            .iter()
            .zip(&self.cfg.head_dims)
            .map(|(h, d)| h * d)
            .max()
            .unwrap_or(hidden);
        let max_r_dim = self.cfg.head_counts.iter().copied().max().unwrap_or(1) * self.cfg.d_rel;
        let max_sinks = self.cfg.head_counts.iter().copied().max().unwrap_or(1);
        let max_dyn = self.cfg.kv_head_counts.iter().copied().max().unwrap_or(1) * self.cfg.conv_kernel;
        for i in 0..3 {
            GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, i, 2 * hidden);
            GpuScratch::ensure_slot(&mut scratch.mid_bufs, chain, i, gate_rank);
            GpuScratch::ensure_slot(&mut scratch.ctrl_gain, chain, i, hidden);
            GpuScratch::ensure_slot(&mut scratch.ctrl_forget, chain, i, hidden);
            GpuScratch::ensure_slot(&mut scratch.ctrl_out, chain, i, hidden);
            GpuScratch::ensure_slot(&mut scratch.norm_bufs, chain, i, hidden);
        }
        let max_qrsg = 2 * max_q_dim + max_r_dim + max_sinks;
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 0, max_qrsg.max(hidden));
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 1, max_kv_dim + max_dyn);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 2, max_kv_dim + max_dyn);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 3, max_r_dim);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 4, max_sinks);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 5, max_q_dim);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 6, max_dyn);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 7, max_dyn);
        scratch.ensure_router(chain, n_exp + 1);

        if scratch.decode_state.is_none() {
            scratch.decode_state = Some(self.init_gpu_decode_state(chain, past_len, 1));
        }
        let ds = scratch.decode_state.as_mut().expect("decode_state initialized");
        if !ds.synced {
            self.sync_host_to_gpu(chain, state, ds);
        } else {
            // Dynamically double full-attention packed `[2, kv_cap, kv_dim]` KV cache if context crosses capacity.
            let mut grew_kv = false;
            for g_slot in &mut ds.slots {
                if !g_slot.is_swa && past_len >= g_slot.kv_cap {
                    let old_cap = g_slot.kv_cap;
                    let new_cap = (past_len + 1).next_power_of_two().max(2048);
                    let kv_dim = g_slot.kv_dim;
                    let new_kv = chain.vec(2 * new_cap * kv_dim);
                    let mut grow = chain.begin();
                    grow.keep_groups(false);
                    grow.copy(&g_slot.kv_cache, 0, &new_kv, 0, past_len * kv_dim);
                    grow.copy(&g_slot.kv_cache, old_cap * kv_dim, &new_kv, new_cap * kv_dim, past_len * kv_dim);
                    let _ = grow.finish();
                    g_slot.kv_cache = new_kv;
                    g_slot.kv_cap = new_cap;
                    grew_kv = true;
                }
            }
            if grew_kv {
                ds.recipe = None;
            }
        }

        // Upload only the embedded input token (9 KB) and 16-byte step params `[q_pos, tau]`.
        chain.upload(&ds.cur, cur_embed);
        chain.upload(&ds.step_params, &[f32::from_bits(past_len as u32), tau0, 0.0, 0.0]);

        let ds = scratch.decode_state.as_ref().expect("decode_state");
        let router_buf = scratch.router();
        let unpadded = self.cfg.vocab_size_unpadded;
        let mut rec = chain.begin();
        rec.keep_groups(true);

        let replayed = ds.recipe.as_ref().is_some_and(|r| rec.replay_recipe(r));
        let mut captured_recipe: Option<ChainRecipe> = None;
        if !replayed {
            rec.capture_recipe();
            let mut n_blocks = 0usize;
            let n_phys = self.cfg.n_phys_layers;

            if let Some(lc) = &self.cfg.loop_cfg {
                for il in 0..lc.core_start {
                    self.record_decode_layer_gpu(
                        &mut *rec, il, il, &mut n_blocks, false, None, ds,
                        &scratch.x_bufs, &scratch.mid_bufs, &scratch.ctrl_gain, &scratch.ctrl_forget,
                        &scratch.ctrl_out, &scratch.norm_bufs, &scratch.y_bufs, router_buf,
                    );
                }
                rec.copy(&ds.cur, 0, &ds.prelude_cur, 0, hidden);
                let prelude_n_blocks = n_blocks;
                let core_len = lc.core_end - lc.core_start;

                for pass in 0..lc.num_passes {
                    let mut n_bank = 0usize;
                    if pass > 0 {
                        let skip = (prelude_n_blocks + 1).min(n_blocks);
                        n_bank = n_blocks - skip;
                        if n_bank > 0 {
                            rec.copy(&ds.blocks, skip * hidden, &ds.bank, 0, n_bank * hidden);
                        }
                        // Write `rms_norm_no_weight(ds.cur)` into `ds.bank_out[0..hidden]`.
                        rec.inkling2_block_push(&ds.cur, &ds.attn_conv, &ds.bank_out, &ds.ffn_inp, hidden, Some(0), true);
                        rec.copy(&ds.prelude_cur, 0, &ds.cur, 0, hidden);
                        n_blocks = prelude_n_blocks;
                    }
                    for ip in lc.core_start..lc.core_end {
                        let il = if pass == 0 {
                            ip
                        } else {
                            n_phys + (pass - 1) * core_len + (ip - lc.core_start)
                        };
                        if pass == 0 {
                            self.record_decode_layer_gpu(
                                &mut *rec, il, ip, &mut n_blocks, false, None, ds,
                                &scratch.x_bufs, &scratch.mid_bufs, &scratch.ctrl_gain, &scratch.ctrl_forget,
                                &scratch.ctrl_out, &scratch.norm_bufs, &scratch.y_bufs, router_buf,
                            );
                        } else {
                            let ll = &self.loop_layers[ip - lc.core_start];
                            let mq_dev = ll.msg_query_dev.as_ref().expect("msg_query_dev");
                            rec.inkling2_mix_depth(&ds.bank, &ds.bank_out, mq_dev, &ds.msgs, n_bank, 4, hidden, true);
                            self.record_decode_layer_gpu(
                                &mut *rec, il, ip, &mut n_blocks, true, Some(ll), ds,
                                &scratch.x_bufs, &scratch.mid_bufs, &scratch.ctrl_gain, &scratch.ctrl_forget,
                                &scratch.ctrl_out, &scratch.norm_bufs, &scratch.y_bufs, router_buf,
                            );
                        }
                    }
                }
                for il in lc.core_end..n_phys {
                    self.record_decode_layer_gpu(
                        &mut *rec, il, il, &mut n_blocks, false, None, ds,
                        &scratch.x_bufs, &scratch.mid_bufs, &scratch.ctrl_gain, &scratch.ctrl_forget,
                        &scratch.ctrl_out, &scratch.norm_bufs, &scratch.y_bufs, router_buf,
                    );
                }
            } else {
                for il in 0..n_phys {
                    self.record_decode_layer_gpu(
                        &mut *rec, il, il, &mut n_blocks, false, None, ds,
                        &scratch.x_bufs, &scratch.mid_bufs, &scratch.ctrl_gain, &scratch.ctrl_forget,
                        &scratch.ctrl_out, &scratch.norm_bufs, &scratch.y_bufs, router_buf,
                    );
                }
            }

            // Output AttnRes + RMSNorm (with fused `rmsnorm_q8`) + unembed + softcap -> ONE readback!
            let oq_dev = self.output_res_query_dev.as_ref().expect("output_res_query_dev");
            let on_dev = self.output_norm_dev.as_ref().expect("output_norm_dev");
            rec.inkling2_mix_depth(&ds.blocks, &ds.cur, oq_dev, &ds.ffn_x, n_blocks, 1, hidden, false);
            let in_v = GpuScratch::view(&ds.ffn_x, hidden);
            let norm_v = GpuScratch::view(&scratch.norm_bufs[0], hidden);
            rec.rmsnorm(&in_v, on_dev, &norm_v, self.cfg.rms_eps);
            record_weight(&mut *rec, &self.output, &norm_v, &ds.logits, 1);
            if self.cfg.final_logit_softcap > 0.0 {
                rec.inkling2_softcap(&ds.logits, unpadded, self.cfg.final_logit_softcap);
            }
            captured_recipe = rec.take_recipe();
        }
        rec.read_range(&ds.logits, 0, unpadded);

        let logits = rec.finish().into_iter().next().expect("decode logits");
        state.pos += 1;
        if let Some(ds_mut) = scratch.decode_state.as_mut() {
            ds_mut.host_dirty = true;
            if let Some(r) = captured_recipe {
                ds_mut.recipe = Some(r);
            }
        }
        if let Some(t0) = t_step_start {
            eprintln!(
                "[inkling2-prof] seq=1 pos={past_len} total={:.2}ms (1-submit full-GPU decode)",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(Tensor::from_vec(logits, vec![1, unpadded]))
    }

    fn init_gpu_prefill_scratch(
        &self,
        chain: &dyn DeviceChain,
        cap_seq: usize,
    ) -> GpuPrefillScratch {
        let hidden = self.cfg.hidden;
        let gate_rank = self.cfg.loop_cfg.as_ref().map_or(1, |l| l.gate_rank);
        let n_exp = self.cfg.n_experts;
        let max_blocks = (self.cfg.n_phys_layers.div_ceil(self.cfg.attn_res_block_size) + 2).min(32);
        let max_q_dim = self
            .cfg
            .head_counts
            .iter()
            .zip(&self.cfg.head_dims)
            .map(|(h, d)| h * d)
            .max()
            .unwrap_or(hidden);
        let max_kv_dim = self
            .cfg
            .kv_head_counts
            .iter()
            .zip(&self.cfg.head_dims)
            .map(|(h, d)| h * d)
            .max()
            .unwrap_or(hidden);
        let max_r_dim = self.cfg.head_counts.iter().copied().max().unwrap_or(1) * self.cfg.d_rel;
        let max_sinks = self.cfg.head_counts.iter().copied().max().unwrap_or(1);
        let max_dyn = self.cfg.kv_head_counts.iter().copied().max().unwrap_or(1) * self.cfg.conv_kernel;
        let max_qrsg = 2 * max_q_dim + max_r_dim + max_sinks;
        let need_fallback_ctrl = self.loop_layers.iter().any(|ll| {
            ll.ctrl_down_qkv_f16.is_none()
                || ll.ctrl_gf_qkv_f16.is_none()
                || ll.ctrl_down_ffn_f16.is_none()
                || ll.ctrl_gf_ffn_f16.is_none()
        });
        let sh = cap_seq * hidden;
        let tau_params = chain.vec(4);
        chain.upload(
            &tau_params,
            &[
                self.cfg.qk_clip,
                self.cfg.rel_clip,
                f32::from_bits(self.cfg.log_scale_floor as u32),
                self.cfg.log_scale_alpha,
            ],
        );
        let y_bufs = vec![
            chain.vec(cap_seq * max_qrsg.max(hidden)),
            chain.vec(cap_seq * (max_kv_dim + max_dyn)),
            chain.vec(cap_seq * (max_kv_dim + max_dyn)),
            chain.vec(cap_seq * max_r_dim),
            chain.vec(cap_seq * max_sinks),
            chain.vec(cap_seq * max_q_dim),
            chain.vec(cap_seq * max_dyn),
            chain.vec(cap_seq * max_dyn),
        ];
        let want_ctrl_gemm = cap_seq >= 16 && !self.loop_layers.is_empty();
        let (x_bufs, mid_bufs, ctrl_gain, ctrl_forget) = if need_fallback_ctrl || want_ctrl_gemm {
            (
                vec![chain.vec(2 * sh)],
                vec![chain.vec(cap_seq * gate_rank)],
                vec![chain.vec(sh)],
                vec![chain.vec(sh)],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        GpuPrefillScratch {
            cap_seq,
            max_blocks,
            tau_params,
            cur: chain.vec(sh),
            prelude_cur: chain.vec(sh),
            blocks: chain.vec(max_blocks * sh),
            bank: chain.vec(max_blocks * sh),
            bank_out: chain.vec(sh),
            msgs: chain.vec(4 * sh),
            mix_out: chain.vec(3 * sh),
            mid_qkv: chain.vec(3 * cap_seq * gate_rank),
            ctrl_qkv: chain.vec(3 * sh),
            ffn_inp: chain.vec(sh),
            ffn_x: chain.vec(sh),
            attn_gated: chain.vec(cap_seq * max_q_dim),
            attn_proj: chain.vec(sh),
            attn_conv: chain.vec(sh),
            moe_out: chain.vec(sh),
            norm_bufs: (0..3).map(|_| chain.vec(sh)).collect(),
            ctrl_out: (0..3).map(|_| chain.vec(sh)).collect(),
            y_bufs,
            router_buf: chain.vec(cap_seq * (n_exp + 1)),
            x_bufs,
            mid_bufs,
            ctrl_gain,
            ctrl_forget,
            logits_all: None,
        }
    }

    /// Record one logical layer into an active single-submit full-GPU batched prefill recording.
    #[allow(clippy::too_many_arguments)]
    fn record_prefill_layer_gpu(
        &self,
        chain: &dyn DeviceChain,
        rec: &mut dyn ChainRecorder,
        il: usize,
        ip: usize,
        n_blocks: &mut usize,
        has_msgs: bool,
        loop_layer: Option<&Inkling2LoopLayer>,
        past_len: usize,
        seq: usize,
        ds: &GpuDecodeState,
        pf: &GpuPrefillScratch,
    ) {
        let layer = &self.layers[ip];
        let dev = layer.decode_dev.as_ref().expect("LayerDecodeDev present");
        let norms_dev = layer.attn_norms_dev.as_ref().expect("attn_norms_dev present");
        let ffn_norm_dev = layer.ffn_norm_dev.as_ref().expect("ffn_norm_dev present");
        let router_dev = layer.ffn_gate_inp_padded_dev.as_ref().expect("router_dev present");
        let qmoe = layer.experts.quant_moe().expect("QuantMoe present");
        let slot = &ds.slots[il];

        let hidden = self.cfg.hidden;
        let sh = seq * hidden;
        let is_swa = layer.is_swa;
        let n_heads = layer.n_heads;
        let n_kv_heads = layer.n_kv_heads;
        let head_dim = layer.head_dim;
        let q_dim = n_heads * head_dim;
        let d_rel = self.cfg.d_rel;
        let rel_extent = if is_swa { self.cfg.sliding_window } else { self.cfg.rel_extent };
        let conv_k = self.cfg.conv_kernel;
        let n_exp = self.cfg.n_experts;
        let gate_rank = self.cfg.loop_cfg.as_ref().map_or(0, |l| l.gate_rank);

        // 1. Attention input via Block Attention Residuals (`mix_depth_rows`).
        if let Some(ref q_dev) = dev.attn_res_query {
            let nq = if is_swa { 1 } else { 3 };
            rec.inkling2_mix_depth_rows(&pf.blocks, &pf.cur, q_dev, &pf.mix_out, *n_blocks, nq, seq, hidden, false);
        } else {
            rec.copy(&pf.cur, 0, &pf.mix_out, 0, sh);
            if !is_swa {
                rec.copy(&pf.cur, 0, &pf.mix_out, sh, sh);
                rec.copy(&pf.cur, 0, &pf.mix_out, 2 * sh, sh);
            }
        }

        // 2. Optional Q/K/V cross-pass controllers + attention pre-norms.
        let has_ctrl = has_msgs && loop_layer.is_some();
        let shared_x = matches!(norms_dev, AttnNormsDev::Shared(_)) && !has_ctrl;
        let (k_idx, v_idx) = if shared_x { (0usize, 0usize) } else { (1usize, 2usize) };

        if let (true, Some(ll)) = (has_msgs, loop_layer) {
            let norm_ws: [&DeviceVec; 3] = match norms_dev {
                AttnNormsDev::Shared(w) => [w, w, w],
                AttnNormsDev::Separate { q, k, v, .. } => [q, k, v],
            };
            // For `seq >= 16`, prefer Tensor Core FP16 GEMM (`HalfLinear` / `record_weight`) over
            // the decode-oriented per-token fused F16 matvec, avoiding `seq` redundant weight reads
            // and 2 extra `[seq, hidden]` copies across 24 looped layers!
            let use_qkv_gemm = seq >= 16
                && !pf.x_bufs.is_empty()
                && (0..3).all(|i| {
                    weight_on_chain(chain, &ll.ctrl_down[i]).is_some()
                        && weight_on_chain(chain, &ll.ctrl_gain[i]).is_some()
                        && weight_on_chain(chain, &ll.ctrl_forget[i]).is_some()
                });
            if !use_qkv_gemm && ll.ctrl_down_qkv_f16.is_some() && ll.ctrl_gf_qkv_f16.is_some() {
                let dqkv = ll.ctrl_down_qkv_f16.as_ref().expect("ctrl_down_qkv_f16");
                let gfqkv = ll.ctrl_gf_qkv_f16.as_ref().expect("ctrl_gf_qkv_f16");
                let cur_stride = if is_swa { 0 } else { sh };
                rec.inkling2_ctrl_cat_down_rows(dqkv, &pf.mix_out, 0, cur_stride, &pf.msgs, 0, &pf.mid_qkv, seq, hidden, gate_rank, 3);
                rec.inkling2_ctrl_up_blend_rows(gfqkv, &pf.mid_qkv, &pf.mix_out, 0, cur_stride, &pf.msgs, 0, &pf.ctrl_qkv, seq, hidden, gate_rank, 3);
                let in0 = GpuScratch::view(&pf.ctrl_qkv, sh);
                let out0 = GpuScratch::view(&pf.norm_bufs[0], sh);
                rec.rmsnorm_rows(&in0, norm_ws[0], &out0, seq, self.cfg.rms_eps);
                for i in 1..3 {
                    rec.copy(&pf.ctrl_qkv, i * sh, &pf.ctrl_out[i], 0, sh);
                    let in_v = GpuScratch::view(&pf.ctrl_out[i], sh);
                    let out_v = GpuScratch::view(&pf.norm_bufs[i], sh);
                    rec.rmsnorm_rows(&in_v, norm_ws[i], &out_v, seq, self.cfg.rms_eps);
                }
            } else {
                for i in 0..3 {
                    let cur_off = if is_swa { 0 } else { i * sh };
                    let msg_off = i * sh;
                    rec.inkling2_control_cat_rows(&pf.mix_out, cur_off, &pf.msgs, msg_off, &pf.x_bufs[0], seq, hidden);
                    record_weight(rec, &ll.ctrl_down[i], &pf.x_bufs[0], &pf.mid_bufs[0], seq);
                    record_weight(rec, &ll.ctrl_gain[i], &pf.mid_bufs[0], &pf.ctrl_gain[0], seq);
                    record_weight(rec, &ll.ctrl_forget[i], &pf.mid_bufs[0], &pf.ctrl_forget[0], seq);
                    rec.control_blend_at(&pf.mix_out, cur_off, &pf.msgs, msg_off, &pf.ctrl_gain[0], &pf.ctrl_forget[0], &pf.ctrl_out[i], sh);
                    let in_v = GpuScratch::view(&pf.ctrl_out[i], sh);
                    let out_v = GpuScratch::view(&pf.norm_bufs[i], sh);
                    rec.rmsnorm_rows(&in_v, norm_ws[i], &out_v, seq, self.cfg.rms_eps);
                }
            }
        } else {
            match norms_dev {
                AttnNormsDev::Shared(w) => {
                    let in_v = GpuScratch::view(&pf.mix_out, sh);
                    let out_v = GpuScratch::view(&pf.norm_bufs[0], sh);
                    rec.rmsnorm_rows(&in_v, w, &out_v, seq, self.cfg.rms_eps);
                }
                AttnNormsDev::Separate { q, k, v, .. } => {
                    let in0 = GpuScratch::view(&pf.mix_out, sh);
                    let out0 = GpuScratch::view(&pf.norm_bufs[0], sh);
                    rec.rmsnorm_rows(&in0, q, &out0, seq, self.cfg.rms_eps);
                    for (i, nw) in [(1usize, k), (2usize, v)] {
                        rec.copy(&pf.mix_out, i * sh, &pf.ctrl_out[i], 0, sh);
                        let in_v = GpuScratch::view(&pf.ctrl_out[i], sh);
                        let out_v = GpuScratch::view(&pf.norm_bufs[i], sh);
                        rec.rmsnorm_rows(&in_v, nw, &out_v, seq, self.cfg.rms_eps);
                    }
                }
            }
        }

        // 3. Attention projections (Tensor Core `matmul_rows` on packed `wqrsg`/`wk_dyn`/`wv_dyn`).
        let rec_attn_w = |rec: &mut dyn ChainRecorder, w: &Weight, stream_idx: usize, out: &DeviceVec| {
            let x_v = GpuScratch::view(&pf.norm_bufs[stream_idx], sh);
            record_weight(rec, w, &x_v, out, seq);
        };
        let packed_qrsg = if let Some(ref wqrsg) = layer.wqrsg {
            rec_attn_w(rec, wqrsg, 0, &pf.y_bufs[0]);
            true
        } else {
            rec_attn_w(rec, &layer.wq, 0, &pf.y_bufs[0]);
            rec_attn_w(rec, &layer.attn_rel, 0, &pf.y_bufs[3]);
            rec_attn_w(rec, &layer.attn_sink_proj, 0, &pf.y_bufs[4]);
            rec_attn_w(rec, &layer.attn_gate, 0, &pf.y_bufs[5]);
            false
        };
        let (packed_dyn, dyn_k_v, dyn_v_v) = if let (Some(ref wkd), Some(ref wvd)) = (&layer.wk_dyn, &layer.wv_dyn) {
            rec_attn_w(rec, wkd, k_idx, &pf.y_bufs[1]);
            rec_attn_w(rec, wvd, v_idx, &pf.y_bufs[2]);
            (true, None, None)
        } else {
            rec_attn_w(rec, &layer.wk, k_idx, &pf.y_bufs[1]);
            rec_attn_w(rec, &layer.wv, v_idx, &pf.y_bufs[2]);
            let dk = if let Some(ref kp) = layer.attn_k_conv_proj {
                rec_attn_w(rec, kp, k_idx, &pf.y_bufs[6]);
                Some(&pf.y_bufs[6])
            } else {
                None
            };
            let dv = if let Some(ref vp) = layer.attn_v_conv_proj {
                rec_attn_w(rec, vp, v_idx, &pf.y_bufs[7]);
                Some(&pf.y_bufs[7])
            } else {
                None
            };
            (false, dk, dv)
        };

        // 4. Batched K/V conv+clip+norm+store+hist + Q-tiled FlashAttention + sigmoid gate.
        rec.inkling2_kv_conv_clip_store_rows(
            &pf.y_bufs[1],
            &pf.y_bufs[2],
            &dev.kv_conv_w,
            dyn_k_v,
            dyn_v_v,
            packed_dyn,
            &slot.kv_conv,
            &slot.kv_cache,
            past_len,
            seq,
            n_kv_heads,
            head_dim,
            conv_k,
            slot.kv_cap,
            self.cfg.qk_clip,
            self.cfg.rms_eps,
        );
        let prefill_attn_ok = rec.inkling2_attend_prefill(
            &pf.y_bufs[0],
            (!packed_qrsg).then_some(&pf.y_bufs[3]),
            (!packed_qrsg).then_some(&pf.y_bufs[4]),
            (!packed_qrsg).then_some(&pf.y_bufs[5]),
            packed_qrsg,
            &dev.attn_consts,
            &slot.kv_cache,
            &pf.attn_gated,
            &pf.tau_params,
            past_len,
            seq,
            n_heads,
            n_kv_heads,
            head_dim,
            d_rel,
            rel_extent,
            slot.kv_cap,
            if is_swa { self.cfg.sliding_window } else { 0 },
        );
        debug_assert!(prefill_attn_ok, "inkling2_attend_prefill must succeed");

        // 5. Wo + post-attention static short conv + block boundary push / residual add.
        let ag_v = GpuScratch::view(&pf.attn_gated, seq * q_dim);
        record_weight(rec, &layer.wo, &ag_v, &pf.attn_proj, seq);
        if ip % self.cfg.attn_res_block_size == 0 {
            rec.inkling2_post_conv_rows(&pf.attn_proj, &dev.attn_post_conv, None, &slot.attn_post_conv, &pf.attn_conv, seq, hidden, conv_k);
            rec.inkling2_block_push_rows(&pf.cur, &pf.attn_conv, &pf.blocks, &pf.ffn_inp, seq, hidden, Some(*n_blocks), false);
            *n_blocks += 1;
        } else {
            rec.inkling2_post_conv_rows(&pf.attn_proj, &dev.attn_post_conv, Some(&pf.cur), &slot.attn_post_conv, &pf.ffn_inp, seq, hidden, conv_k);
        }

        // 6. MoE FFN: AttnRes -> optional FFN ctrl -> RMSNorm -> Router -> Tensor Core QuantMoe -> post-FFN conv + residual.
        rec.inkling2_mix_depth_rows(&pf.blocks, &pf.ffn_inp, &dev.ffn_res_query, &pf.ffn_x, *n_blocks, 1, seq, hidden, false);
        let norm_v = GpuScratch::view(&pf.norm_bufs[0], sh);
        if let (true, Some(ll)) = (has_msgs, loop_layer) {
            let use_ffn_gemm = seq >= 16
                && !pf.x_bufs.is_empty()
                && weight_on_chain(chain, &ll.ctrl_down[3]).is_some()
                && weight_on_chain(chain, &ll.ctrl_gain[3]).is_some()
                && weight_on_chain(chain, &ll.ctrl_forget[3]).is_some();
            if use_ffn_gemm {
                rec.inkling2_control_cat_rows(&pf.ffn_x, 0, &pf.msgs, 3 * sh, &pf.x_bufs[0], seq, hidden);
                record_weight(rec, &ll.ctrl_down[3], &pf.x_bufs[0], &pf.mid_bufs[0], seq);
                record_weight(rec, &ll.ctrl_gain[3], &pf.mid_bufs[0], &pf.ctrl_gain[0], seq);
                record_weight(rec, &ll.ctrl_forget[3], &pf.mid_bufs[0], &pf.ctrl_forget[0], seq);
                rec.control_blend_at(&pf.ffn_x, 0, &pf.msgs, 3 * sh, &pf.ctrl_gain[0], &pf.ctrl_forget[0], &pf.ctrl_out[0], sh);
            } else {
                if let Some(ref df) = ll.ctrl_down_ffn_f16 {
                    rec.inkling2_ctrl_cat_down_rows(df, &pf.ffn_x, 0, 0, &pf.msgs, 3 * sh, &pf.mid_qkv, seq, hidden, gate_rank, 1);
                } else {
                    rec.inkling2_control_cat_rows(&pf.ffn_x, 0, &pf.msgs, 3 * sh, &pf.x_bufs[0], seq, hidden);
                    record_weight(rec, &ll.ctrl_down[3], &pf.x_bufs[0], &pf.mid_bufs[0], seq);
                }
                if let Some(ref gf) = ll.ctrl_gf_ffn_f16 {
                    rec.inkling2_ctrl_up_blend_rows(gf, &pf.mid_qkv, &pf.ffn_x, 0, 0, &pf.msgs, 3 * sh, &pf.ctrl_out[0], seq, hidden, gate_rank, 1);
                } else {
                    let mid_src = if ll.ctrl_down_ffn_f16.is_some() { &pf.mid_qkv } else { &pf.mid_bufs[0] };
                    record_weight(rec, &ll.ctrl_gain[3], mid_src, &pf.ctrl_gain[0], seq);
                    record_weight(rec, &ll.ctrl_forget[3], mid_src, &pf.ctrl_forget[0], seq);
                    rec.control_blend_at(&pf.ffn_x, 0, &pf.msgs, 3 * sh, &pf.ctrl_gain[0], &pf.ctrl_forget[0], &pf.ctrl_out[0], sh);
                }
            }
            let in_v = GpuScratch::view(&pf.ctrl_out[0], sh);
            rec.rmsnorm_rows(&in_v, ffn_norm_dev, &norm_v, seq, self.cfg.rms_eps);
        } else {
            let in_v = GpuScratch::view(&pf.ffn_x, sh);
            rec.rmsnorm_rows(&in_v, ffn_norm_dev, &norm_v, seq, self.cfg.rms_eps);
        }
        let router_v = GpuScratch::view(&pf.router_buf, seq * (n_exp + 1));
        rec.matmul_f32_rows(router_dev, n_exp + 1, hidden, &norm_v, &router_v, seq);
        let moe_v = GpuScratch::view(&pf.moe_out, sh);
        let routed_ok = rec.moe_routed(qmoe, &norm_v, &moe_v, &router_v, self.cfg.n_experts_used, seq);
        assert!(routed_ok, "QuantMoe::record_routed must succeed for full-GPU prefill seq={seq}");
        rec.inkling2_post_conv_rows(&pf.moe_out, &dev.ffn_post_conv, Some(&pf.ffn_inp), &slot.ffn_post_conv, &pf.cur, seq, hidden, conv_k);
    }

    /// Single-submit full-GPU batched prefill (`seq >= 1`): executes ALL 48 logical layers on GPU
    /// (with Tensor Core matmuls, Tensor Core QuantMoe, Q-tiled FlashAttention, and live GPU KV/conv
    /// caches) in ONE `DeviceChain` submission, transitioning into `forward_decode_gpu` with zero
    /// KV-cache upload stall!
    fn forward_prefill_gpu(
        &self,
        cur_embed: &[f32],
        seq: usize,
        state: &mut Inkling2State,
        scratch: &mut GpuScratch,
        out_mode: OutputMode,
        t_step_start: Option<Instant>,
    ) -> Result<Tensor> {
        let chain = self.backend.chain().expect("full_gpu_decode requires DeviceChain");
        let hidden = self.cfg.hidden;
        let sh = seq * hidden;
        let past_len = state.pos;
        let swa = self.cfg.sliding_window;

        if scratch.decode_state.is_none() {
            scratch.decode_state = Some(self.init_gpu_decode_state(chain, past_len, seq));
        }
        let ds = scratch.decode_state.as_mut().expect("decode_state initialized");

        // Ensure SWA slots have `kv_cap >= swa + seq` so writing `seq` new rows never overwrites
        // earlier keys still needed by token `t = 0` in the same chunk.
        let need_swa_grow = ds.slots.iter().any(|s| s.is_swa && s.kv_cap < swa + seq);
        if need_swa_grow {
            if ds.synced && ds.host_dirty && past_len > 0 {
                self.sync_gpu_to_host(chain, state, ds);
            }
            let new_swa_cap = (swa + seq).max(swa + self.preferred_prefill_chunk());
            for g_slot in &mut ds.slots {
                if g_slot.is_swa && g_slot.kv_cap < new_swa_cap {
                    g_slot.kv_cache = chain.vec(2 * new_swa_cap * g_slot.kv_dim);
                    g_slot.kv_cap = new_swa_cap;
                }
            }
            ds.synced = false;
            ds.recipe = None;
        }

        if !ds.synced {
            // Also ensure full-attention slots fit `past_len + seq` before syncing.
            for g_slot in &mut ds.slots {
                if !g_slot.is_swa && past_len + seq > g_slot.kv_cap {
                    let new_cap = (past_len + seq).next_power_of_two().max(2048);
                    g_slot.kv_cache = chain.vec(2 * new_cap * g_slot.kv_dim);
                    g_slot.kv_cap = new_cap;
                }
            }
            self.sync_host_to_gpu(chain, state, ds);
        } else {
            let mut grew_kv = false;
            for g_slot in &mut ds.slots {
                if !g_slot.is_swa && past_len + seq > g_slot.kv_cap {
                    let old_cap = g_slot.kv_cap;
                    let new_cap = (past_len + seq).next_power_of_two().max(2048);
                    let kv_dim = g_slot.kv_dim;
                    let new_kv = chain.vec(2 * new_cap * kv_dim);
                    if past_len > 0 {
                        let mut grow = chain.begin();
                        grow.keep_groups(false);
                        grow.copy(&g_slot.kv_cache, 0, &new_kv, 0, past_len * kv_dim);
                        grow.copy(&g_slot.kv_cache, old_cap * kv_dim, &new_kv, new_cap * kv_dim, past_len * kv_dim);
                        let _ = grow.finish();
                    }
                    g_slot.kv_cache = new_kv;
                    g_slot.kv_cap = new_cap;
                    grew_kv = true;
                }
            }
            if grew_kv {
                ds.recipe = None;
            }
        }

        let need_pf_init = scratch.prefill_scratch.as_ref().is_none_or(|pf| pf.cap_seq < seq);
        if need_pf_init {
            let cap_seq = seq.next_power_of_two().max(self.preferred_prefill_chunk().min(512).max(64));
            scratch.prefill_scratch = Some(self.init_gpu_prefill_scratch(chain, cap_seq));
        }
        if out_mode == OutputMode::All {
            let pf_mut = scratch.prefill_scratch.as_mut().expect("prefill_scratch");
            let need = seq * self.cfg.vocab_size_padded;
            if pf_mut.logits_all.as_ref().is_none_or(|v| v.len < need) {
                pf_mut.logits_all = Some(chain.vec(need));
            }
        }

        let ds = scratch.decode_state.as_ref().expect("decode_state");
        let pf = scratch.prefill_scratch.as_ref().expect("prefill_scratch");
        chain.upload_at(&pf.cur, 0, cur_embed);

        let unpadded = self.cfg.vocab_size_unpadded;
        let padded = self.cfg.vocab_size_padded;
        let mut rec = chain.begin();
        rec.keep_groups(false);

        let mut n_blocks = 0usize;
        let n_phys = self.cfg.n_phys_layers;

        if let Some(lc) = &self.cfg.loop_cfg {
            for il in 0..lc.core_start {
                self.record_prefill_layer_gpu(chain, &mut *rec, il, il, &mut n_blocks, false, None, past_len, seq, ds, pf);
            }
            rec.copy(&pf.cur, 0, &pf.prelude_cur, 0, sh);
            let prelude_n_blocks = n_blocks;
            let core_len = lc.core_end - lc.core_start;

            for pass in 0..lc.num_passes {
                let mut n_bank = 0usize;
                if pass > 0 {
                    let skip = (prelude_n_blocks + 1).min(n_blocks);
                    n_bank = n_blocks - skip;
                    if n_bank > 0 {
                        rec.copy(&pf.blocks, skip * sh, &pf.bank, 0, n_bank * sh);
                    }
                    // Write `rms_norm_no_weight(pf.cur)` into `pf.bank_out[0..sh]`.
                    rec.inkling2_block_push_rows(&pf.cur, &pf.attn_conv, &pf.bank_out, &pf.ffn_inp, seq, hidden, Some(0), true);
                    rec.copy(&pf.prelude_cur, 0, &pf.cur, 0, sh);
                    n_blocks = prelude_n_blocks;
                }
                for ip in lc.core_start..lc.core_end {
                    let il = if pass == 0 {
                        ip
                    } else {
                        n_phys + (pass - 1) * core_len + (ip - lc.core_start)
                    };
                    if pass == 0 {
                        self.record_prefill_layer_gpu(chain, &mut *rec, il, ip, &mut n_blocks, false, None, past_len, seq, ds, pf);
                    } else {
                        let ll = &self.loop_layers[ip - lc.core_start];
                        let mq_dev = ll.msg_query_dev.as_ref().expect("msg_query_dev");
                        rec.inkling2_mix_depth_rows(&pf.bank, &pf.bank_out, mq_dev, &pf.msgs, n_bank, 4, seq, hidden, true);
                        self.record_prefill_layer_gpu(chain, &mut *rec, il, ip, &mut n_blocks, true, Some(ll), past_len, seq, ds, pf);
                    }
                }
            }
            for il in lc.core_end..n_phys {
                self.record_prefill_layer_gpu(chain, &mut *rec, il, il, &mut n_blocks, false, None, past_len, seq, ds, pf);
            }
        } else {
            for il in 0..n_phys {
                self.record_prefill_layer_gpu(chain, &mut *rec, il, il, &mut n_blocks, false, None, past_len, seq, ds, pf);
            }
        }

        let out_tensor = match out_mode {
            OutputMode::None => {
                rec.send();
                Tensor::from_vec(Vec::new(), vec![0, unpadded])
            }
            OutputMode::LastOnly => {
                let oq_dev = self.output_res_query_dev.as_ref().expect("output_res_query_dev");
                let on_dev = self.output_norm_dev.as_ref().expect("output_norm_dev");
                rec.inkling2_mix_depth_rows(&pf.blocks, &pf.cur, oq_dev, &pf.ffn_x, n_blocks, 1, seq, hidden, false);
                rec.copy(&pf.ffn_x, (seq - 1) * hidden, &ds.ffn_x, 0, hidden);
                let in_v = GpuScratch::view(&ds.ffn_x, hidden);
                let norm_v = GpuScratch::view(&pf.norm_bufs[0], hidden);
                rec.rmsnorm(&in_v, on_dev, &norm_v, self.cfg.rms_eps);
                record_weight(&mut *rec, &self.output, &norm_v, &ds.logits, 1);
                if self.cfg.final_logit_softcap > 0.0 {
                    rec.inkling2_softcap(&ds.logits, unpadded, self.cfg.final_logit_softcap);
                }
                rec.read_range(&ds.logits, 0, unpadded);
                let logits = rec.finish().into_iter().next().expect("prefill last logits");
                Tensor::from_vec(logits, vec![1, unpadded])
            }
            OutputMode::All => {
                let oq_dev = self.output_res_query_dev.as_ref().expect("output_res_query_dev");
                let on_dev = self.output_norm_dev.as_ref().expect("output_norm_dev");
                let logits_all = pf.logits_all.as_ref().expect("logits_all allocated");
                rec.inkling2_mix_depth_rows(&pf.blocks, &pf.cur, oq_dev, &pf.ffn_x, n_blocks, 1, seq, hidden, false);
                let in_v = GpuScratch::view(&pf.ffn_x, sh);
                let norm_v = GpuScratch::view(&pf.norm_bufs[0], sh);
                rec.rmsnorm_rows(&in_v, on_dev, &norm_v, seq, self.cfg.rms_eps);
                record_weight(&mut *rec, &self.output, &norm_v, logits_all, seq);
                if self.cfg.final_logit_softcap > 0.0 {
                    rec.inkling2_softcap(logits_all, seq * padded, self.cfg.final_logit_softcap);
                }
                rec.read_range(logits_all, 0, seq * padded);
                let full = rec.finish().into_iter().next().expect("prefill all logits");
                Tensor::from_vec(trim_rows(full, seq, padded, unpadded), vec![seq, unpadded])
            }
        };

        state.pos += seq;
        if let Some(ds_mut) = scratch.decode_state.as_mut() {
            ds_mut.synced = true;
            ds_mut.host_dirty = true;
        }
        if let Some(t0) = t_step_start {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[inkling2-prof] seq={seq} pos={past_len} total={ms:.2}ms ({:.1} tok/s, 1-submit full-GPU prefill, out={out_mode:?})",
                seq as f64 / (ms / 1000.0).max(1e-9)
            );
        }
        Ok(out_tensor)
    }

    /// Fused single-submit GPU execution of optional Q/K/V cross-pass controllers + attention
    /// pre-norms (with fused `rmsnorm_q8` on `seq == 1`) + Q/K/V/R/Sink/Gate/[DynK/DynV]
    /// projections. Returns `None` if any required weight is not resident on `DeviceChain`.
    #[allow(clippy::too_many_arguments)]
    fn try_run_attn_proj_gpu(
        &self,
        layer: &Inkling2Layer,
        scratch: &mut GpuScratch,
        inp_q: &[f32],
        inp_k: &[f32],
        inp_v: &[f32],
        messages: Option<&[Vec<f32>]>,
        loop_layer: Option<&Inkling2LoopLayer>,
        seq: usize,
        hidden: usize,
        gate_rank: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let chain = self.backend.chain()?;
        let norms_dev = layer.attn_norms_dev.as_ref()?;
        let has_ctrl = messages.is_some() && loop_layer.is_some();
        if let Some(ll) = loop_layer {
            if messages.is_some() {
                for i in 0..3 {
                    weight_on_chain(chain, &ll.ctrl_down[i])?;
                    weight_on_chain(chain, &ll.ctrl_gain[i])?;
                    weight_on_chain(chain, &ll.ctrl_forget[i])?;
                }
            }
        }
        let shared_x = matches!(norms_dev, AttnNormsDev::Shared(_)) && !has_ctrl;
        let (k_idx, v_idx) = if shared_x { (0usize, 0usize) } else { (1usize, 2usize) };

        let use_qrsg = layer.wqrsg.as_ref().and_then(|w| weight_on_chain(chain, w));
        let use_k_dyn = layer.wk_dyn.as_ref().and_then(|w| weight_on_chain(chain, w));
        let use_v_dyn = layer.wv_dyn.as_ref().and_then(|w| weight_on_chain(chain, w));

        let mut ops: Vec<(&Weight, usize, usize)> = Vec::with_capacity(8);
        if let Some(dim) = use_qrsg {
            ops.push((layer.wqrsg.as_ref().expect("wqrsg"), 0, dim));
        } else {
            ops.push((&layer.wq, 0, weight_on_chain(chain, &layer.wq)?));
            ops.push((&layer.attn_rel, 0, weight_on_chain(chain, &layer.attn_rel)?));
            ops.push((&layer.attn_sink_proj, 0, weight_on_chain(chain, &layer.attn_sink_proj)?));
            ops.push((&layer.attn_gate, 0, weight_on_chain(chain, &layer.attn_gate)?));
        }
        if let Some(dim) = use_k_dyn {
            ops.push((layer.wk_dyn.as_ref().expect("wk_dyn"), k_idx, dim));
        } else {
            ops.push((&layer.wk, k_idx, weight_on_chain(chain, &layer.wk)?));
            if let Some(ref k_proj) = layer.attn_k_conv_proj {
                ops.push((k_proj, k_idx, weight_on_chain(chain, k_proj)?));
            }
        }
        if let Some(dim) = use_v_dyn {
            ops.push((layer.wv_dyn.as_ref().expect("wv_dyn"), v_idx, dim));
        } else {
            ops.push((&layer.wv, v_idx, weight_on_chain(chain, &layer.wv)?));
            if let Some(ref v_proj) = layer.attn_v_conv_proj {
                ops.push((v_proj, v_idx, weight_on_chain(chain, v_proj)?));
            }
        }

        let unpack_rows = |flat: &[f32], dims: &[usize]| -> Vec<Vec<f32>> {
            let total: usize = dims.iter().sum();
            let mut parts: Vec<Vec<f32>> = dims.iter().map(|&d| Vec::with_capacity(seq * d)).collect();
            for t in 0..seq {
                let mut off = t * total;
                for (p, &d) in parts.iter_mut().zip(dims) {
                    p.extend_from_slice(&flat[off..off + d]);
                    off += d;
                }
            }
            parts
        };
        let q_dim = layer.n_heads * layer.head_dim;
        let r_dim = layer.n_heads * self.cfg.d_rel;
        let n_heads = layer.n_heads;
        let kv_dim = layer.n_kv_heads * layer.head_dim;
        let dyn_dim = layer.n_kv_heads * self.cfg.conv_kernel;
        let reorder_raw = |raw: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
            let mut it = raw.into_iter();
            let (q, r, sinks, gate) = if use_qrsg.is_some() {
                let flat = it.next().expect("qrsg");
                let mut p = unpack_rows(&flat, &[q_dim, r_dim, n_heads, q_dim]).into_iter();
                (
                    p.next().expect("q"),
                    p.next().expect("r"),
                    p.next().expect("sinks"),
                    p.next().expect("gate"),
                )
            } else {
                (
                    it.next().expect("q"),
                    it.next().expect("r"),
                    it.next().expect("sinks"),
                    it.next().expect("gate"),
                )
            };
            let (k, dyn_k) = if use_k_dyn.is_some() {
                let flat = it.next().expect("wk_dyn");
                let mut p = unpack_rows(&flat, &[kv_dim, dyn_dim]).into_iter();
                (p.next().expect("k"), Some(p.next().expect("dyn_k")))
            } else {
                let k = it.next().expect("k");
                let dk = layer.attn_k_conv_proj.as_ref().map(|_| it.next().expect("dyn_k"));
                (k, dk)
            };
            let (v, dyn_v) = if use_v_dyn.is_some() {
                let flat = it.next().expect("wv_dyn");
                let mut p = unpack_rows(&flat, &[kv_dim, dyn_dim]).into_iter();
                (p.next().expect("v"), Some(p.next().expect("dyn_v")))
            } else {
                let v = it.next().expect("v");
                let dv = layer.attn_v_conv_proj.as_ref().map(|_| it.next().expect("dyn_v"));
                (v, dv)
            };
            let mut out = vec![q, k, v, r, sinks, gate];
            if let Some(dk) = dyn_k {
                out.push(dk);
            }
            if let Some(dv) = dyn_v {
                out.push(dv);
            }
            out
        };

        let n_streams = if shared_x { 1 } else { 3 };
        let sh = seq * hidden;
        for i in 0..n_streams {
            GpuScratch::ensure_slot(&mut scratch.ctrl_out, chain, i, sh);
            GpuScratch::ensure_slot(&mut scratch.norm_bufs, chain, i, sh);
        }
        for (slot, &(_, _, out_dim)) in ops.iter().enumerate() {
            GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, slot, seq * out_dim);
        }

        if let (Some(msgs), Some(ll)) = (messages, loop_layer) {
            let inps = [inp_q, inp_k, inp_v];
            let cat0 = build_control_cat(inp_q, &msgs[0], seq, hidden);
            let cat1 = build_control_cat(inp_k, &msgs[1], seq, hidden);
            let cat2 = build_control_cat(inp_v, &msgs[2], seq, hidden);
            let cats = [&cat0[..], &cat1[..], &cat2[..]];
            for i in 0..3 {
                GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, i, seq * 2 * hidden);
                GpuScratch::ensure_slot(&mut scratch.mid_bufs, chain, i, seq * gate_rank);
                GpuScratch::ensure_slot(&mut scratch.ctrl_cur, chain, i, sh);
                GpuScratch::ensure_slot(&mut scratch.ctrl_msg, chain, i, sh);
                GpuScratch::ensure_slot(&mut scratch.ctrl_gain, chain, i, sh);
                GpuScratch::ensure_slot(&mut scratch.ctrl_forget, chain, i, sh);
                chain.upload(&scratch.x_bufs[i], cats[i]);
                chain.upload(&scratch.ctrl_cur[i], inps[i]);
                chain.upload(&scratch.ctrl_msg[i], &msgs[i]);
            }
            let mut rec = chain.begin();
            rec.keep_groups(seq == 1);
            let norm_ws: [&DeviceVec; 3] = match norms_dev {
                AttnNormsDev::Shared(w) => [w, w, w],
                AttnNormsDev::Separate { q, k, v, .. } => [q, k, v],
            };
            for i in 0..3 {
                record_weight(&mut *rec, &ll.ctrl_down[i], &scratch.x_bufs[i], &scratch.mid_bufs[i], seq);
                record_weight(&mut *rec, &ll.ctrl_gain[i], &scratch.mid_bufs[i], &scratch.ctrl_gain[i], seq);
                record_weight(&mut *rec, &ll.ctrl_forget[i], &scratch.mid_bufs[i], &scratch.ctrl_forget[i], seq);
                rec.control_blend(
                    &scratch.ctrl_cur[i],
                    &scratch.ctrl_msg[i],
                    &scratch.ctrl_gain[i],
                    &scratch.ctrl_forget[i],
                    &scratch.ctrl_out[i],
                    sh,
                );
                let in_v = GpuScratch::view(&scratch.ctrl_out[i], sh);
                let out_v = GpuScratch::view(&scratch.norm_bufs[i], sh);
                if seq == 1 {
                    rec.rmsnorm(&in_v, norm_ws[i], &out_v, self.cfg.rms_eps);
                } else {
                    rec.rmsnorm_rows(&in_v, norm_ws[i], &out_v, seq, self.cfg.rms_eps);
                }
            }
            for (slot, &(w, inp_idx, out_dim)) in ops.iter().enumerate() {
                let x_v = GpuScratch::view(&scratch.norm_bufs[inp_idx], sh);
                record_weight(&mut *rec, w, &x_v, &scratch.y_bufs[slot], seq);
                rec.read_range(&scratch.y_bufs[slot], 0, seq * out_dim);
            }
            return Some(reorder_raw(rec.finish()));
        }

        // Non-controlled path (prelude, pass 0, postlude): 1 or 3 norm + all attention projections in 1 submit.
        if shared_x {
            chain.upload(&scratch.ctrl_out[0], inp_q);
        } else {
            chain.upload(&scratch.ctrl_out[0], inp_q);
            chain.upload(&scratch.ctrl_out[1], inp_k);
            chain.upload(&scratch.ctrl_out[2], inp_v);
        }
        let mut rec = chain.begin();
        rec.keep_groups(seq == 1);
        match norms_dev {
            AttnNormsDev::Shared(w) => {
                let in_v = GpuScratch::view(&scratch.ctrl_out[0], sh);
                let out_v = GpuScratch::view(&scratch.norm_bufs[0], sh);
                if seq == 1 {
                    rec.rmsnorm(&in_v, w, &out_v, self.cfg.rms_eps);
                } else {
                    rec.rmsnorm_rows(&in_v, w, &out_v, seq, self.cfg.rms_eps);
                }
            }
            AttnNormsDev::Separate { q, k, v, .. } => {
                for (i, nw) in [q, k, v].into_iter().enumerate() {
                    let in_v = GpuScratch::view(&scratch.ctrl_out[i], sh);
                    let out_v = GpuScratch::view(&scratch.norm_bufs[i], sh);
                    if seq == 1 {
                        rec.rmsnorm(&in_v, nw, &out_v, self.cfg.rms_eps);
                    } else {
                        rec.rmsnorm_rows(&in_v, nw, &out_v, seq, self.cfg.rms_eps);
                    }
                }
            }
        }
        for (slot, &(w, inp_idx, out_dim)) in ops.iter().enumerate() {
            let x_v = GpuScratch::view(&scratch.norm_bufs[inp_idx], sh);
            record_weight(&mut *rec, w, &x_v, &scratch.y_bufs[slot], seq);
            rec.read_range(&scratch.y_bufs[slot], 0, seq * out_dim);
        }
        Some(reorder_raw(rec.finish()))
    }

    /// Fused single-submit GPU execution of optional FFN cross-pass controller + FFN RMSNorm +
    /// exact-F32 router projection + `QuantMoe` GPU routing/gate_up/SiTU/down/wsum.
    #[allow(clippy::too_many_arguments)]
    fn try_run_ffn_gpu(
        &self,
        layer: &Inkling2Layer,
        scratch: &mut GpuScratch,
        ffn_x: &[f32],
        messages: Option<&[Vec<f32>]>,
        loop_layer: Option<&Inkling2LoopLayer>,
        seq: usize,
        hidden: usize,
        gate_rank: usize,
    ) -> Option<Vec<f32>> {
        let chain = self.backend.chain()?;
        let qmoe = layer.experts.quant_moe()?;
        if !chain.holds_experts(qmoe) {
            return None;
        }
        let ffn_norm_dev = layer.ffn_norm_dev.as_ref()?;
        let router_dev = layer.ffn_gate_inp_padded_dev.as_ref()?;
        if let (Some(_), Some(ll)) = (messages, loop_layer) {
            weight_on_chain(chain, &ll.ctrl_down[3])?;
            weight_on_chain(chain, &ll.ctrl_gain[3])?;
            weight_on_chain(chain, &ll.ctrl_forget[3])?;
        }

        let sh = seq * hidden;
        let n_exp = self.cfg.n_experts;
        let router_len = seq * (n_exp + 1);
        GpuScratch::ensure_slot(&mut scratch.ctrl_out, chain, 0, sh);
        GpuScratch::ensure_slot(&mut scratch.norm_bufs, chain, 0, sh);
        GpuScratch::ensure_slot(&mut scratch.y_bufs, chain, 0, sh);
        scratch.ensure_router(chain, router_len);

        if let (Some(msgs), Some(ll)) = (messages, loop_layer) {
            let cat = build_control_cat(ffn_x, &msgs[3], seq, hidden);
            GpuScratch::ensure_slot(&mut scratch.x_bufs, chain, 0, seq * 2 * hidden);
            GpuScratch::ensure_slot(&mut scratch.mid_bufs, chain, 0, seq * gate_rank);
            GpuScratch::ensure_slot(&mut scratch.ctrl_cur, chain, 0, sh);
            GpuScratch::ensure_slot(&mut scratch.ctrl_msg, chain, 0, sh);
            GpuScratch::ensure_slot(&mut scratch.ctrl_gain, chain, 0, sh);
            GpuScratch::ensure_slot(&mut scratch.ctrl_forget, chain, 0, sh);
            chain.upload(&scratch.x_bufs[0], &cat);
            chain.upload(&scratch.ctrl_cur[0], ffn_x);
            chain.upload(&scratch.ctrl_msg[0], &msgs[3]);

            let in_v = GpuScratch::view(&scratch.ctrl_out[0], sh);
            let norm_v = GpuScratch::view(&scratch.norm_bufs[0], sh);
            let router_v = GpuScratch::view(scratch.router(), router_len);
            let out_v = GpuScratch::view(&scratch.y_bufs[0], sh);

            let mut rec = chain.begin();
            rec.keep_groups(seq == 1);
            record_weight(&mut *rec, &ll.ctrl_down[3], &scratch.x_bufs[0], &scratch.mid_bufs[0], seq);
            record_weight(&mut *rec, &ll.ctrl_gain[3], &scratch.mid_bufs[0], &scratch.ctrl_gain[0], seq);
            record_weight(&mut *rec, &ll.ctrl_forget[3], &scratch.mid_bufs[0], &scratch.ctrl_forget[0], seq);
            rec.control_blend(
                &scratch.ctrl_cur[0],
                &scratch.ctrl_msg[0],
                &scratch.ctrl_gain[0],
                &scratch.ctrl_forget[0],
                &scratch.ctrl_out[0],
                sh,
            );
            if seq == 1 {
                rec.rmsnorm(&in_v, ffn_norm_dev, &norm_v, self.cfg.rms_eps);
            } else {
                rec.rmsnorm_rows(&in_v, ffn_norm_dev, &norm_v, seq, self.cfg.rms_eps);
            }
            rec.matmul_f32_rows(router_dev, n_exp + 1, hidden, &norm_v, &router_v, seq);
            if rec.moe_routed(qmoe, &norm_v, &out_v, &router_v, self.cfg.n_experts_used, seq) {
                rec.read_range(&out_v, 0, sh);
                return Some(rec.finish().into_iter().next().expect("fused ffn moe out"));
            }
            // Non-coop GPU with `seq > 64`: read router logits and dispatch via host-routed `moe_rows`.
            rec.read_range(&router_v, 0, router_len);
            let padded = rec.finish().into_iter().next().expect("router logits");
            let routing = ggml_rs::exl3::Routing::Sigmoid {
                bias: layer
                    .exp_probs_b
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; n_exp]),
                scale: self.cfg.route_scale,
            };
            let assign: Vec<Vec<(usize, f32)>> = (0..seq)
                .map(|t| routing.route(&padded[t * (n_exp + 1)..(t + 1) * (n_exp + 1)], self.cfg.n_experts_used))
                .collect();
            let mut rec2 = chain.begin();
            rec2.keep_groups(false);
            rec2.moe_rows(qmoe, &norm_v, &out_v, &assign);
            rec2.read_range(&out_v, 0, sh);
            return Some(rec2.finish().into_iter().next().expect("host-routed moe out"));
        }

        // Non-controlled FFN path (prelude, pass 0, postlude): norm + router + QuantMoe in 1 submit.
        chain.upload(&scratch.ctrl_out[0], ffn_x);
        let in_v = GpuScratch::view(&scratch.ctrl_out[0], sh);
        let norm_v = GpuScratch::view(&scratch.norm_bufs[0], sh);
        let router_v = GpuScratch::view(scratch.router(), router_len);
        let out_v = GpuScratch::view(&scratch.y_bufs[0], sh);

        let mut rec = chain.begin();
        rec.keep_groups(seq == 1);
        if seq == 1 {
            rec.rmsnorm(&in_v, ffn_norm_dev, &norm_v, self.cfg.rms_eps);
        } else {
            rec.rmsnorm_rows(&in_v, ffn_norm_dev, &norm_v, seq, self.cfg.rms_eps);
        }
        rec.matmul_f32_rows(router_dev, n_exp + 1, hidden, &norm_v, &router_v, seq);
        if rec.moe_routed(qmoe, &norm_v, &out_v, &router_v, self.cfg.n_experts_used, seq) {
            rec.read_range(&out_v, 0, sh);
            return Some(rec.finish().into_iter().next().expect("fused ffn moe out"));
        }
        rec.read_range(&router_v, 0, router_len);
        let padded = rec.finish().into_iter().next().expect("router logits");
        let routing = ggml_rs::exl3::Routing::Sigmoid {
            bias: layer
                .exp_probs_b
                .clone()
                .unwrap_or_else(|| vec![0.0f32; n_exp]),
            scale: self.cfg.route_scale,
        };
        let assign: Vec<Vec<(usize, f32)>> = (0..seq)
            .map(|t| routing.route(&padded[t * (n_exp + 1)..(t + 1) * (n_exp + 1)], self.cfg.n_experts_used))
            .collect();
        let mut rec2 = chain.begin();
        rec2.keep_groups(false);
        rec2.moe_rows(qmoe, &norm_v, &out_v, &assign);
        rec2.read_range(&out_v, 0, sh);
        Some(rec2.finish().into_iter().next().expect("host-routed moe out"))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &self,
        il: usize,
        ip: usize,
        cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&[Vec<f32>]>,
        loop_layer: Option<&Inkling2LoopLayer>,
        tau: &[f32],
        past_len: usize,
        seq: usize,
        state: &mut Inkling2State,
        scratch: &mut GpuScratch,
        prof: &mut StepProfile,
    ) -> Result<Vec<f32>> {
        let layer: &Inkling2Layer = &self.layers[ip];
        let hidden = self.cfg.hidden;
        let is_swa = layer.is_swa;
        let n_heads = layer.n_heads;
        let n_kv_heads = layer.n_kv_heads;
        let head_dim = layer.head_dim;
        let d_rel = self.cfg.d_rel;
        let rel_extent = if is_swa { self.cfg.sliding_window } else { self.cfg.rel_extent };
        let gate_rank = self.cfg.loop_cfg.as_ref().map_or(0, |l| l.gate_rank);
        let b = self.backend.as_ref();

        let inp_sa = cur;
        let t0 = if prof.enabled { Some(Instant::now()) } else { None };

        // 1. Attention input via Block Attention Residuals (skipped for layer 0).
        let (mut inp_q, mut inp_k, mut inp_v) = if let Some(ref q) = layer.attn_res_query {
            let nq = if is_swa { 1 } else { 3 };
            let mut mixed = mix_depth(blocks, &inp_sa, q, nq, seq, hidden);
            if is_swa {
                let x = mixed.remove(0);
                (x.clone(), x.clone(), x)
            } else {
                let v = mixed.remove(2);
                let k = mixed.remove(1);
                let qx = mixed.remove(0);
                (qx, k, v)
            }
        } else {
            (inp_sa.clone(), inp_sa.clone(), inp_sa.clone())
        };

        let has_dyn_k = layer.attn_k_conv_proj.is_some();
        let has_dyn_v = layer.attn_v_conv_proj.is_some();

        // 2-4. Fused single-submit GPU path: optional QKV cross-pass controllers + attention
        //      pre-norm (with fused rmsnorm_q8 on decode) + Q/K/V/R/Sink/Gate/[DynK/DynV].
        let mut proj_outs = if let Some(gpu_outs) = self.try_run_attn_proj_gpu(
            layer,
            scratch,
            &inp_q,
            &inp_k,
            &inp_v,
            messages,
            loop_layer,
            seq,
            hidden,
            gate_rank,
        ) {
            if let Some(t) = t0 {
                prof.attn_proj_us += t.elapsed().as_micros() as u64;
            }
            gpu_outs.into_iter()
        } else {
            // Fallback (CPU or partial-GPU weights): run controllers, pre-norm, and batched projections.
            if let (Some(msgs), Some(ll)) = (messages, loop_layer) {
                (inp_q, inp_k, inp_v) =
                    apply_controls_qkv(b, scratch, &inp_q, &inp_k, &inp_v, msgs, ll, seq, hidden, gate_rank);
            }
            let shared_x = matches!(&layer.attn_norms, AttnNorms::Shared(_)) && messages.is_none();
            let (x_q, x_k, x_v) = match &layer.attn_norms {
                AttnNorms::Shared(norm) => {
                    if messages.is_some() {
                        (
                            rms_norm_rows(&inp_q, norm, hidden, self.cfg.rms_eps),
                            rms_norm_rows(&inp_k, norm, hidden, self.cfg.rms_eps),
                            rms_norm_rows(&inp_v, norm, hidden, self.cfg.rms_eps),
                        )
                    } else {
                        let n = rms_norm_rows(&inp_q, norm, hidden, self.cfg.rms_eps);
                        (n, Vec::new(), Vec::new())
                    }
                }
                AttnNorms::Separate { q, k, v } => (
                    rms_norm_rows(&inp_q, q, hidden, self.cfg.rms_eps),
                    rms_norm_rows(&inp_k, k, hidden, self.cfg.rms_eps),
                    rms_norm_rows(&inp_v, v, hidden, self.cfg.rms_eps),
                ),
            };
            if let Some(t) = t0 {
                prof.attn_res_ctrl_us += t.elapsed().as_micros() as u64;
            }
            let t1 = if prof.enabled { Some(Instant::now()) } else { None };
            let (inputs_slice, k_idx, v_idx): (Vec<&[f32]>, usize, usize) = if shared_x {
                (vec![&x_q], 0, 0)
            } else {
                (vec![&x_q, &x_k, &x_v], 1, 2)
            };
            let mut ops: Vec<(&Weight, usize)> = Vec::with_capacity(8);
            ops.push((&layer.wq, 0));
            ops.push((&layer.wk, k_idx));
            ops.push((&layer.wv, v_idx));
            ops.push((&layer.attn_rel, 0));
            ops.push((&layer.attn_sink_proj, 0));
            ops.push((&layer.attn_gate, 0));
            if let Some(ref k_proj) = layer.attn_k_conv_proj {
                ops.push((k_proj, k_idx));
            }
            if let Some(ref v_proj) = layer.attn_v_conv_proj {
                ops.push((v_proj, v_idx));
            }
            let out = linear_host_batch(b, scratch, &inputs_slice, &ops, seq, hidden).into_iter();
            if let Some(t) = t1 {
                prof.attn_proj_us += t.elapsed().as_micros() as u64;
            }
            out
        };
        let mut q_cur = proj_outs.next().expect("wq");
        let k_raw = proj_outs.next().expect("wk");
        let v_raw = proj_outs.next().expect("wv");
        let mut r_cur = proj_outs.next().expect("attn_rel");
        let sinks = proj_outs.next().expect("attn_sink_proj");
        let gate_raw = proj_outs.next().expect("attn_gate");
        let dyn_k_opt = if has_dyn_k { proj_outs.next() } else { None };
        let dyn_v_opt = if has_dyn_v { proj_outs.next() } else { None };
        let t2 = if prof.enabled { Some(Instant::now()) } else { None };

        let slot = &mut state.slots[il];

        // K and V causal short convolutions (dynamic on full-attention layers, static on SWA).
        let mut k_cur = if let Some(dyn_k) = dyn_k_opt {
            slot.k_conv.apply_dynamic(&k_raw, &dyn_k, seq, &layer.attn_k_conv, n_kv_heads)
        } else {
            slot.k_conv.apply_static(&k_raw, seq, &layer.attn_k_conv)
        };
        let v_conv = if let Some(dyn_v) = dyn_v_opt {
            slot.v_conv.apply_dynamic(&v_raw, &dyn_v, seq, &layer.attn_v_conv, n_kv_heads)
        } else {
            slot.v_conv.apply_static(&v_raw, seq, &layer.attn_v_conv)
        };

        // L2 clipping on Q, K, R and unweighted RMSNorm on V (per head).
        l2_clip_heads(&mut q_cur, Some(&layer.attn_q_clip), seq, n_heads, head_dim, self.cfg.qk_clip);
        l2_clip_heads(&mut k_cur, None, seq, n_kv_heads, head_dim, self.cfg.qk_clip);
        let v_cur = rms_norm_no_weight(&v_conv, head_dim, self.cfg.rms_eps);
        l2_clip_heads(&mut r_cur, Some(&layer.attn_rel_clip), seq, n_heads, d_rel, self.cfg.rel_clip);

        // Log-scale tau on Q and R for full-attention layers.
        if !is_swa {
            let q_dim = n_heads * head_dim;
            let r_dim = n_heads * d_rel;
            for (t, &tau_t) in tau.iter().enumerate() {
                if (tau_t - 1.0).abs() > 1e-7 {
                    for v in &mut q_cur[t * q_dim..(t + 1) * q_dim] {
                        *v *= tau_t;
                    }
                    for v in &mut r_cur[t * r_dim..(t + 1) * r_dim] {
                        *v *= tau_t;
                    }
                }
            }
        }

        // Update KV cache and compute relative-bias (trimmed to reachable distances) + sink attention.
        slot.append_kv(&k_cur, &v_cur, seq);
        let active_rel = (past_len + seq).min(rel_extent);
        let rel_table = project_rel_bias(&r_cur, &layer.attn_rel_bias, seq, n_heads, d_rel, active_rel);
        let mut attn_out = slot.attend(
            &q_cur,
            &rel_table,
            &sinks,
            seq,
            past_len,
            n_heads,
            n_kv_heads,
            head_dim,
            active_rel,
        );
        slot.maybe_compact();

        // Sigmoid output gate + Wo + post-attention short conv.
        for i in 0..attn_out.len() {
            attn_out[i] *= 1.0 / (1.0 + (-gate_raw[i]).exp());
        }
        if let Some(t) = t2 {
            prof.attn_core_us += t.elapsed().as_micros() as u64;
        }
        let t3 = if prof.enabled { Some(Instant::now()) } else { None };
        let attn_proj = linear_host_fast(b, scratch, &layer.wo, &attn_out, seq, n_heads * head_dim, usize::MAX);
        if let Some(t) = t3 {
            prof.attn_wo_us += t.elapsed().as_micros() as u64;
        }
        let t4 = if prof.enabled { Some(Instant::now()) } else { None };
        let attn_conv = slot.attn_post_conv.apply_static(&attn_proj, seq, &layer.attn_post_conv);

        // 5. Block boundary update (`ip % attn_res_block_size == 0` starts a new block residual,
        //    stored pre-normalized for all subsequent `mix_depth` calls).
        let ffn_inp = if ip % self.cfg.attn_res_block_size == 0 {
            blocks.push(rms_norm_no_weight(&inp_sa, hidden, 1e-6));
            attn_conv
        } else {
            let mut sum = inp_sa;
            for i in 0..sum.len() {
                sum[i] += attn_conv[i];
            }
            sum
        };

        // 6. MoE FFN: AttnRes -> optional cross-pass FFN control -> RMSNorm -> Router + Experts -> post-FFN conv.
        let mut ffn_x = mix_depth(blocks, &ffn_inp, &layer.ffn_res_query, 1, seq, hidden)
            .into_iter()
            .next()
            .expect("1 ffn query");
        let moe_out = if let Some(gpu_moe_out) =
            self.try_run_ffn_gpu(layer, scratch, &ffn_x, messages, loop_layer, seq, hidden, gate_rank)
        {
            if let Some(t) = t4 {
                prof.experts_us += t.elapsed().as_micros() as u64;
            }
            gpu_moe_out
        } else {
            if let (Some(msgs), Some(ll)) = (messages, loop_layer) {
                ffn_x = apply_control(b, scratch, &ffn_x, &msgs[3], ll, 3, seq, hidden, gate_rank);
            }
            let ffn_normed = rms_norm_rows(&ffn_x, &layer.ffn_norm, hidden, self.cfg.rms_eps);
            let router_logits = linear_host_fast(b, scratch, &layer.ffn_gate_inp, &ffn_normed, seq, hidden, usize::MAX);
            if let Some(t) = t4 {
                prof.ffn_pre_us += t.elapsed().as_micros() as u64;
            }
            let t5 = if prof.enabled { Some(Instant::now()) } else { None };
            let out = layer.experts.forward(
                b,
                scratch,
                &ffn_normed,
                &router_logits,
                layer.exp_probs_b.as_deref(),
                seq,
                self.cfg.n_experts,
                self.cfg.n_experts_used,
                self.cfg.route_scale,
                self.cfg.situ_beta,
                self.cfg.situ_linear_beta,
            )?;
            if let Some(t) = t5 {
                prof.experts_us += t.elapsed().as_micros() as u64;
            }
            out
        };
        let ffn_conv = state.slots[il]
            .ffn_post_conv
            .apply_static(&moe_out, seq, &layer.ffn_post_conv);

        let mut out = ffn_inp;
        for i in 0..out.len() {
            out[i] += ffn_conv[i];
        }
        Ok(out)
    }
}
