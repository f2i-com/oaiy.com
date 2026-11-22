//! MoE experts and BF16 / NVFP4 / MXFP8 / GGUF packed kernels for Inkling-2.
//!
//! Supports all four weight representations without conversion or float32 bloat:
//!   * **GGUF** (`ffn_gate_up_exps` + `ffn_down_exps` in any GGML block type, zero-copy mmap
//!     or source-backed bytes).
//!   * **Safetensors NVFP4** (`U8` packed E2M1 nibbles + `F8_E4M3` per-16 scales + `F32` amax /
//!     outer_scale_denominator; `w13` rows interleaved even=gate, odd=up).
//!   * **Safetensors MXFP8** (`F8_E4M3` weights + `U8` UE8M0 per-32 scales; `w13` interleaved).
//!   * **Dense F32** (for unquantized safetensors or synthetic unit tests).
//!
//! Also provides [`Bf16Linear`], a compact 2-byte/weight `PackedLinear` implementation so
//! BF16 dense projections from safetensors run on CPU with zero 2x F32 memory inflation.

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};

use ggml_quants::GgmlType;
use ggml_rs::exl3::PackedLinear;
use ggml_rs::{Backend, QuantizedTensor, Tensor};
use gguf::{GgufFile, TensorInfo as GgufTensorInfo};
use oaiy_engine::io::read_exact_at;
use rayon::prelude::*;

use crate::llm::{LlamaError, Result};

use super::forward::GpuScratch;

/// E4M3FN lookup table (bias 7, no infinities, 0x7f / 0xff are NaN).
pub fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out = [0.0f32; 256];
        for code in 0u16..256 {
            let b = code as u8;
            if (b & 0x7f) == 0x7f {
                out[code as usize] = f32::NAN;
                continue;
            }
            let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
            let exp = ((b >> 3) & 0x0f) as i32;
            let mant = (b & 0x07) as f32;
            let mag = if exp == 0 {
                mant * 2.0f32.powi(-9)
            } else {
                (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
            };
            out[code as usize] = sign * mag;
        }
        out
    })
}

/// FP4 E2M1 lookup table (4-bit nibble -> f32).
pub const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Pre-decoded pair of E2M1 floats for every packed byte `(low_nibble, high_nibble)`.
fn e2m1_byte_table() -> &'static ([f32; 256], [f32; 256]) {
    static TABLE: OnceLock<([f32; 256], [f32; 256])> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut lo = [0.0f32; 256];
        let mut hi = [0.0f32; 256];
        for b in 0usize..256 {
            lo[b] = E2M1_TABLE[b & 0x0f];
            hi[b] = E2M1_TABLE[b >> 4];
        }
        (lo, hi)
    })
}

/// Runtime-dispatched SIMD (AVX-512F / AVX2+FMA / portable) f32 dot product.
#[inline(always)]
pub(crate) fn f32_dot_fast(a: &[f32], b: &[f32]) -> f32 {
    ggml_quants::f32_dot::dot_fast(a, b)
}

/// Runtime-dispatched SIMD 4-row batched f32 matrix-vector product.
#[inline(always)]
pub(crate) fn f32_dot_rows_fast(w: &[f32], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    ggml_quants::f32_dot::dot_rows_fast(w, x, n_rows, k, out);
}

/// Runtime-dispatched SIMD weighted accumulate `dst[i] += weight * src[i]`.
#[inline(always)]
pub(crate) fn f32_axpy_fast(dst: &mut [f32], src: &[f32], weight: f32) {
    ggml_quants::f32_dot::axpy_fast(dst, src, weight);
}

/// Bytes per quantized weight row of `k` elements.
#[inline]
pub(crate) fn quant_row_bytes(dtype: GgmlType, k: usize) -> usize {
    (k / dtype.block_size()) * dtype.type_size()
}

/// Fused SIMD matrix-vector product `out[r] = dot(W[r], x)` for GGUF quantized rows.
///
/// Uses AVX-512 / AVX2 fused kernels for IQ4_NL, Q8_0, Q4_K, and Q6_K, and falls back to
/// per-row dequantization (reusing one tiny row buffer) for all other GGML types.
pub(crate) fn gguf_dot_rows_any(
    dtype: GgmlType,
    raw: &[u8],
    x: &[f32],
    n_rows: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    match dtype {
        GgmlType::IQ4_NL => {
            ggml_quants::iq4_nl::dot_rows_fast(raw, x, n_rows, k, out);
            Ok(())
        }
        GgmlType::Q8_0 => {
            ggml_quants::q8_0::dot_rows_fast(raw, x, n_rows, k, out);
            Ok(())
        }
        GgmlType::Q4_K => {
            ggml_quants::q4_k::dot_rows_fast(raw, x, n_rows, k, out);
            Ok(())
        }
        GgmlType::Q6_K => {
            ggml_quants::q6_k::dot_rows_fast(raw, x, n_rows, k, out);
            Ok(())
        }
        _ => {
            let row_bytes = quant_row_bytes(dtype, k);
            let mut row_f32 = vec![0.0f32; k];
            for r in 0..n_rows {
                ggml_quants::dequantize(dtype, &raw[r * row_bytes..(r + 1) * row_bytes], &mut row_f32)?;
                out[r] = f32_dot_fast(&row_f32, x);
            }
            Ok(())
        }
    }
}

/// Power-of-two UE8M0 scale table: `2^(code - 127)` for `0..255`, `NaN` for `255`.
pub fn ue8m0_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out = [0.0f32; 256];
        for code in 0u16..255 {
            out[code as usize] = 2.0f32.powi(code as i32 - 127);
        }
        out[255] = f32::NAN;
        out
    })
}

#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Inkling-2 SiTU activation (`Inkling2SoftcappedSwiGLU`):
/// `situ(gate, up) = (beta * tanh(gate / beta)) * sigmoid(gate) * (linear_beta * tanh(up / linear_beta))`.
/// Note that `sigmoid` operates on the *raw* (un-softcapped) `gate`.
#[inline]
pub fn situ(gate: f32, up: f32, beta: f32, linear_beta: f32) -> f32 {
    let g_cap = beta * (gate / beta).tanh();
    let sig = 1.0 / (1.0 + (-gate).exp());
    let u_cap = linear_beta * (up / linear_beta).tanh();
    g_cap * sig * u_cap
}

/// Compact host-resident BF16 matrix (`[n, k]`) implementing `PackedLinear`, keeping
/// safetensors dense weights at 2 bytes/element instead of inflating to F32.
#[derive(Debug)]
pub struct Bf16Linear {
    data: Vec<u16>,
    shape: [usize; 2],
}

impl Bf16Linear {
    pub fn from_le_bytes(bytes: &[u8], n: usize, k: usize) -> Result<Self> {
        if bytes.len() != n * k * 2 {
            return Err(LlamaError::Config(format!(
                "Bf16Linear: expected {} bytes for [{n}, {k}], got {}",
                n * k * 2,
                bytes.len()
            )));
        }
        let mut data = Vec::with_capacity(n * k);
        for chunk in bytes.chunks_exact(2) {
            data.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(Self { data, shape: [n, k] })
    }

    pub fn row_f32(&self, row: usize) -> Vec<f32> {
        let k = self.shape[1];
        let slice = &self.data[row * k..(row + 1) * k];
        slice.iter().map(|&b| bf16_bits_to_f32(b)).collect()
    }

    /// Fast host matmul computing only the first `max_rows` rows (`<= shape[0]`), using
    /// 8-lane unrolled accumulators and coarse row-tile parallelism.
    pub fn linear_rows_slice(&self, x_data: &[f32], m: usize, max_rows: usize) -> Vec<f32> {
        let [n_total, k] = self.shape;
        let n = max_rows.min(n_total);
        let mut out = vec![0.0f32; m * n];
        let w = &self.data;

        #[inline(always)]
        fn bf16_dot(row: &[u16], x: &[f32]) -> f32 {
            let k = x.len();
            let chunks = k / 8;
            let mut acc = [0.0f32; 8];
            for c in 0..chunks {
                let base = c * 8;
                let rw = &row[base..base + 8];
                let xx = &x[base..base + 8];
                for j in 0..8 {
                    acc[j] += xx[j] * bf16_bits_to_f32(rw[j]);
                }
            }
            let mut sum =
                ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for i in chunks * 8..k {
                sum += x[i] * bf16_bits_to_f32(row[i]);
            }
            sum
        }

        if m == 1 {
            out.par_chunks_mut(32).enumerate().for_each(|(ti, dst)| {
                let r0 = ti * 32;
                for (i, d) in dst.iter_mut().enumerate() {
                    let r = r0 + i;
                    *d = bf16_dot(&w[r * k..(r + 1) * k], x_data);
                }
            });
        } else if m >= 8 {
            out.par_chunks_mut(n).enumerate().for_each(|(row_i, out_row)| {
                let x_row = &x_data[row_i * k..(row_i + 1) * k];
                for (r, dst) in out_row.iter_mut().enumerate() {
                    *dst = bf16_dot(&w[r * k..(r + 1) * k], x_row);
                }
            });
        } else {
            for row_i in 0..m {
                let x_row = &x_data[row_i * k..(row_i + 1) * k];
                out[row_i * n..(row_i + 1) * n]
                    .par_chunks_mut(32)
                    .enumerate()
                    .for_each(|(ti, dst)| {
                        let r0 = ti * 32;
                        for (i, d) in dst.iter_mut().enumerate() {
                            let r = r0 + i;
                            *d = bf16_dot(&w[r * k..(r + 1) * k], x_row);
                        }
                    });
            }
        }
        out
    }
}

impl PackedLinear for Bf16Linear {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn nbytes(&self) -> usize {
        self.data.len() * 2
    }

    fn linear(&self, x: &Tensor) -> Tensor {
        let x_h = if x.is_device() { x.to_host() } else { x.clone() };
        let x_data = x_h.data();
        let [n, k] = self.shape;
        let m = x_data.len() / k;
        let mut out_shape = x_h.shape().to_vec();
        *out_shape.last_mut().expect("non-empty shape") = n;
        let out = self.linear_rows_slice(x_data, m, n);
        Tensor::from_vec(out, out_shape)
    }
}

/// Bounded in-memory cache of raw packed safetensors expert records so looped core
/// layers (3 passes/token) and hot experts don't re-issue file reads on every pass.
#[derive(Debug)]
pub struct PackedExpertCache {
    budget_bytes: usize,
    inner: Mutex<CacheInner>,
}

#[derive(Debug)]
struct CacheInner {
    used_bytes: usize,
    tick: u64,
    entries: HashMap<(usize, usize), (u64, Arc<PackedExpertRecord>)>,
}

#[derive(Debug)]
pub struct PackedExpertRecord {
    pub w13_bytes: Vec<u8>,
    pub w13_scale: Vec<u8>,
    pub w2_bytes: Vec<u8>,
    pub w2_scale: Vec<u8>,
}

impl PackedExpertRecord {
    fn nbytes(&self) -> usize {
        self.w13_bytes.len() + self.w13_scale.len() + self.w2_bytes.len() + self.w2_scale.len()
    }
}

impl PackedExpertCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            inner: Mutex::new(CacheInner {
                used_bytes: 0,
                tick: 0,
                entries: HashMap::new(),
            }),
        }
    }

    pub fn get_or_load<F>(&self, layer: usize, expert: usize, load: F) -> Result<Arc<PackedExpertRecord>>
    where
        F: FnOnce() -> Result<PackedExpertRecord>,
    {
        if self.budget_bytes > 0 {
            let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            g.tick = g.tick.wrapping_add(1);
            let tick = g.tick;
            if let Some((last_used, rec)) = g.entries.get_mut(&(layer, expert)) {
                *last_used = tick;
                return Ok(Arc::clone(rec));
            }
        }
        let rec = Arc::new(load()?);
        let size = rec.nbytes();
        if self.budget_bytes >= size {
            let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            if g.entries.contains_key(&(layer, expert)) {
                return Ok(rec);
            }
            while g.used_bytes + size > self.budget_bytes && !g.entries.is_empty() {
                if let Some((&victim_key, _)) = g.entries.iter().min_by_key(|(_, (t, _))| *t) {
                    if let Some((_, victim)) = g.entries.remove(&victim_key) {
                        g.used_bytes = g.used_bytes.saturating_sub(victim.nbytes());
                    }
                } else {
                    break;
                }
            }
            g.tick = g.tick.wrapping_add(1);
            let tick = g.tick;
            g.used_bytes += size;
            g.entries.insert((layer, expert), (tick, Arc::clone(&rec)));
        }
        Ok(rec)
    }
}

/// Zero-copy slice into a GGUF file (either mmap-backed or pre-read owned bytes).
#[derive(Debug)]
pub struct GgufExpertSlice {
    file: GgufFile,
    shard: usize,
    base_offset: usize,
    per_expert_bytes: usize,
    dtype: GgmlType,
    /// Owned fallback when opened from a non-mmap `TensorBytes` source.
    owned: Option<Arc<Vec<u8>>>,
}

impl GgufExpertSlice {
    fn from_gguf(g: &GgufFile, info: &GgufTensorInfo, n_experts: usize) -> Result<Self> {
        let total_bytes = info.nbytes() as usize;
        if n_experts == 0 || total_bytes % n_experts != 0 {
            return Err(LlamaError::Config(format!(
                "expert tensor bytes ({total_bytes}) not divisible by n_experts ({n_experts})"
            )));
        }
        let per_expert_bytes = total_bytes / n_experts;
        let shard = g.shard_of(info);
        let owned = if let Some(src) = g.tensor_source_of(info) {
            Some(Arc::new(src.read_tensor(info)?))
        } else {
            None
        };
        Ok(Self {
            file: g.clone(),
            shard,
            base_offset: g.tensor_data_offset(info),
            per_expert_bytes,
            dtype: info.dtype,
            owned,
        })
    }

    fn expert_bytes(&self, expert: usize) -> &[u8] {
        let start = expert * self.per_expert_bytes;
        let end = start + self.per_expert_bytes;
        if let Some(buf) = &self.owned {
            &buf[start..end]
        } else {
            self.file.raw_slice_shard(self.shard, self.base_offset + start, self.per_expert_bytes)
        }
    }
}

#[derive(Debug)]
pub struct Nvfp4LayerSpec {
    pub layer_idx: usize,
    pub file: Arc<File>,
    pub cache: Arc<PackedExpertCache>,
    pub w13_offset: u64,
    pub w13_scale_offset: u64,
    pub w13_amax: Vec<f32>,
    pub w13_denom: f32,
    pub w2_offset: u64,
    pub w2_scale_offset: u64,
    pub w2_amax: Vec<f32>,
    pub w2_denom: f32,
}

#[derive(Debug)]
pub struct Mxfp8LayerSpec {
    pub layer_idx: usize,
    pub file: Arc<File>,
    pub cache: Arc<PackedExpertCache>,
    pub w13_offset: u64,
    pub w13_scale_offset: u64,
    pub w2_offset: u64,
    pub w2_scale_offset: u64,
}

/// Per-layer routed experts storage for Inkling-2.
#[derive(Debug)]
pub enum Inkling2Experts {
    Gguf {
        gate_up: GgufExpertSlice,
        down: GgufExpertSlice,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    },
    /// GPU-resident GGUF experts packed as two contiguous `QuantMoe` buffers (`gate_up` + `down`)
    /// with native GPU sigmoid routing, SiTU activation, and Tensor-Core cooperative-matrix prefill.
    GgufQuantMoe {
        experts: Box<dyn ggml_rs::exl3::Experts>,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    },
    /// GPU-resident GGUF experts (uploaded per-expert onto `DeviceChain` VRAM when `QuantMoe` does
    /// not support the quant type, prioritizing looped core layers that execute 3× per token).
    GgufGpu {
        gate_up: Vec<QuantizedTensor>,
        down: Vec<QuantizedTensor>,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    },
    SafetensorsNvfp4 {
        spec: Nvfp4LayerSpec,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    },
    SafetensorsMxfp8 {
        spec: Mxfp8LayerSpec,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    },
    /// Dense F32 experts: `gate_up` `[n_experts, 2*ff, hidden]` (non-interleaved: `0..ff` gate,
    /// `ff..2*ff` up) and `down` `[n_experts, hidden, ff]`.
    DenseF32 {
        gate_up: Vec<f32>,
        down: Vec<f32>,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    },
}

impl Inkling2Experts {
    pub fn from_gguf(
        g: &GgufFile,
        gate_up_info: &GgufTensorInfo,
        down_info: &GgufTensorInfo,
        n_experts: usize,
        hidden: usize,
        ff: usize,
    ) -> Result<Self> {
        let gate_up = GgufExpertSlice::from_gguf(g, gate_up_info, n_experts)?;
        let down = GgufExpertSlice::from_gguf(g, down_info, n_experts)?;
        Ok(Self::Gguf {
            gate_up,
            down,
            n_experts,
            hidden,
            ff,
        })
    }

    /// Try uploading this GGUF layer's experts into true GPU VRAM if `backend` has at least
    /// `layer_bytes + reserve_bytes` of free VRAM budget. Prefers contiguous `QuantMoe` groups
    /// (single-submit GPU routing + SiTU + Tensor-Core prefill + 0 per-token allocations),
    /// falling back to per-expert `QuantizedTensor`s for quant types without `QuantMoe` kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn try_upload_to_gpu(
        &self,
        backend: &dyn Backend,
        reserve_bytes: usize,
        exp_probs_b: Option<&[f32]>,
        route_scale: f32,
        situ_beta: f32,
        situ_linear_beta: f32,
    ) -> Option<Self> {
        let Self::Gguf { gate_up, down, n_experts, hidden, ff } = self else {
            return None;
        };
        let chain = backend.chain()?;
        let (avail, _budget) = backend.vram_status()?;

        // Fast path: pack into two contiguous GPU buffers via `WgpuBackend::try_quant_experts_situ_leaving`.
        #[cfg(feature = "webgpu")]
        if let Some(wgpu) = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
            let est_bytes = n_experts
                .checked_mul(gate_up.per_expert_bytes + down.per_expert_bytes)
                .map(|b| b + b / 6);
            if est_bytes.is_some_and(|b| avail >= b.saturating_add(reserve_bytes))
                && gate_up.per_expert_bytes % 2 == 0
            {
                let half_gu = gate_up.per_expert_bytes / 2;
                let mut gate_bytes = Vec::with_capacity(*n_experts * half_gu);
                let mut up_bytes = Vec::with_capacity(*n_experts * half_gu);
                let mut down_bytes = Vec::with_capacity(*n_experts * down.per_expert_bytes);
                for e in 0..*n_experts {
                    let gu = gate_up.expert_bytes(e);
                    gate_bytes.extend_from_slice(&gu[..half_gu]);
                    up_bytes.extend_from_slice(&gu[half_gu..]);
                    down_bytes.extend_from_slice(down.expert_bytes(e));
                }
                let bias = exp_probs_b
                    .map(|b| b.to_vec())
                    .unwrap_or_else(|| vec![0.0f32; *n_experts]);
                let data = ggml_rs_wgpu::quant::moe::QuantExpertsData {
                    hidden: *hidden,
                    ff: *ff,
                    experts: *n_experts,
                    gate: (gate_up.dtype, gate_bytes),
                    up: (gate_up.dtype, up_bytes),
                    down: (down.dtype, down_bytes),
                    shared: None,
                    routing: ggml_rs::exl3::Routing::Sigmoid {
                        bias,
                        scale: route_scale,
                    },
                };
                if let Ok(Some(qmoe)) = wgpu.try_quant_experts_situ_leaving(
                    data,
                    reserve_bytes as u64,
                    situ_beta,
                    situ_linear_beta,
                ) {
                    return Some(Self::GgufQuantMoe {
                        experts: qmoe,
                        n_experts: *n_experts,
                        hidden: *hidden,
                        ff: *ff,
                    });
                }
            }
        }
        #[cfg(not(feature = "webgpu"))]
        let _ = (exp_probs_b, route_scale, situ_beta, situ_linear_beta);

        if !crate::llm::loader::dtype_supports_packed_matmul(gate_up.dtype)
            || !crate::llm::loader::dtype_supports_packed_matmul(down.dtype)
        {
            return None;
        }
        let layer_bytes = n_experts.checked_mul(gate_up.per_expert_bytes + down.per_expert_bytes)?;
        if avail < layer_bytes.saturating_add(reserve_bytes) {
            return None;
        }

        let mut gu_dev = Vec::with_capacity(*n_experts);
        let mut dn_dev = Vec::with_capacity(*n_experts);
        for e in 0..*n_experts {
            let gu_qt = QuantizedTensor::from_bytes_cpu(
                gate_up.expert_bytes(e).to_vec(),
                vec![2 * *ff, *hidden],
                gate_up.dtype,
            );
            let gu_up = backend.to_device_quant(gu_qt);
            if !chain.holds(&gu_up) || backend.weight_streamed(&gu_up) {
                return None;
            }
            gu_dev.push(gu_up);

            let dn_qt = QuantizedTensor::from_bytes_cpu(
                down.expert_bytes(e).to_vec(),
                vec![*hidden, *ff],
                down.dtype,
            );
            let dn_up = backend.to_device_quant(dn_qt);
            if !chain.holds(&dn_up) || backend.weight_streamed(&dn_up) {
                return None;
            }
            dn_dev.push(dn_up);
        }

        Some(Self::GgufGpu {
            gate_up: gu_dev,
            down: dn_dev,
            n_experts: *n_experts,
            hidden: *hidden,
            ff: *ff,
        })
    }

    pub(crate) fn quant_moe(&self) -> Option<&dyn ggml_rs::exl3::Experts> {
        match self {
            Self::GgufQuantMoe { experts, .. } => Some(experts.as_ref()),
            _ => None,
        }
    }

    pub fn hidden(&self) -> usize {
        match self {
            Self::Gguf { hidden, .. }
            | Self::GgufQuantMoe { hidden, .. }
            | Self::GgufGpu { hidden, .. }
            | Self::SafetensorsNvfp4 { hidden, .. }
            | Self::SafetensorsMxfp8 { hidden, .. }
            | Self::DenseF32 { hidden, .. } => *hidden,
        }
    }

    /// Run Inkling-2 MoE FFN for `seq` tokens (`x`: `[seq, hidden]`).
    ///
    /// * `router_logits`: `[seq, n_experts]` (f32)
    /// * `exp_probs_b`: optional `[n_experts]` bias added only for top-k selection
    /// * `top_k`: number of routed experts per token (8 for Inkling-2-Mini)
    /// * `route_scale`: scaling factor multiplied into normalized weights (8.0)
    /// * `situ_beta`, `situ_linear_beta`: SiTU softcap constants (4.0, 25.0)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        scratch: &mut GpuScratch,
        x: &[f32],
        router_logits: &[f32],
        exp_probs_b: Option<&[f32]>,
        seq: usize,
        n_experts: usize,
        top_k: usize,
        route_scale: f32,
        situ_beta: f32,
        situ_linear_beta: f32,
    ) -> Result<Vec<f32>> {
        let hidden = self.hidden();
        debug_assert_eq!(x.len(), seq * hidden);
        debug_assert_eq!(router_logits.len(), seq * n_experts);

        // Contiguous GPU `QuantMoe` fallback when invoked standalone (non-fused path):
        if let Self::GgufQuantMoe { experts, .. } = self {
            if let Some(chain) = backend.chain() {
                if chain.holds_experts(experts.as_ref()) {
                    let mut padded = vec![0.0f32; seq * (n_experts + 1)];
                    for t in 0..seq {
                        padded[t * (n_experts + 1)..t * (n_experts + 1) + n_experts]
                            .copy_from_slice(&router_logits[t * n_experts..(t + 1) * n_experts]);
                    }
                    scratch.ensure_x(chain, 0, seq * hidden);
                    scratch.ensure_y(chain, 0, seq * hidden);
                    scratch.ensure_router(chain, seq * (n_experts + 1));
                    let xv = GpuScratch::view(scratch.x(0), seq * hidden);
                    let yv = GpuScratch::view(scratch.y(0), seq * hidden);
                    let lv = GpuScratch::view(scratch.router(), seq * (n_experts + 1));
                    chain.upload(&xv, x);
                    chain.upload(&lv, &padded);
                    let mut rec = chain.begin();
                    rec.keep_groups(seq == 1);
                    if rec.moe_routed(experts.as_ref(), &xv, &yv, &lv, top_k, seq) {
                        rec.read_range(&yv, 0, seq * hidden);
                        return Ok(rec.finish().into_iter().next().expect("quant moe out"));
                    }
                    // Non-coop GPU with `seq > 64`: route on host and dispatch via `moe_rows`.
                    let mut assign = Vec::with_capacity(seq);
                    let routing = ggml_rs::exl3::Routing::Sigmoid {
                        bias: exp_probs_b
                            .map(|b| b.to_vec())
                            .unwrap_or_else(|| vec![0.0f32; n_experts]),
                        scale: route_scale,
                    };
                    for t in 0..seq {
                        assign.push(routing.route(
                            &padded[t * (n_experts + 1)..(t + 1) * (n_experts + 1)],
                            top_k,
                        ));
                    }
                    rec.moe_rows(experts.as_ref(), &xv, &yv, &assign);
                    rec.read_range(&yv, 0, seq * hidden);
                    return Ok(rec.finish().into_iter().next().expect("quant moe out"));
                }
            }
        }

        if seq == 1 {
            return self.forward_decode_one(
                backend,
                scratch,
                x,
                router_logits,
                exp_probs_b,
                n_experts,
                top_k,
                route_scale,
                situ_beta,
                situ_linear_beta,
            );
        }

        // 1. Sigmoid routing in FP32 with optional bias for top-k selection, normalized weights * route_scale.
        // Group tokens by expert so each active expert's weights are read ONCE per chunk.
        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
        let mut scores = vec![0.0f32; n_experts];
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n_experts);

        for t in 0..seq {
            let logits = &router_logits[t * n_experts..(t + 1) * n_experts];
            scored.clear();
            for e in 0..n_experts {
                let s = 1.0 / (1.0 + (-logits[e]).exp());
                scores[e] = s;
                let sel = match exp_probs_b {
                    Some(b) => s + b[e],
                    None => s,
                };
                scored.push((e, sel));
            }
            scored.select_nth_unstable_by(top_k - 1, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            let mut sum = 0.0f32;
            for &(e, _) in &scored[..top_k] {
                sum += scores[e];
            }
            let norm = route_scale / (sum + 1e-20);
            for &(e, _) in &scored[..top_k] {
                buckets[e].push((t, scores[e] * norm));
            }
        }

        // Collect active experts only.
        let active: Vec<(usize, Vec<(usize, f32)>)> = buckets
            .into_iter()
            .enumerate()
            .filter(|(_, toks)| !toks.is_empty())
            .collect();

        // GPU-resident GGUF experts: batch active experts on DeviceChain in chunks of 16.
        if let Self::GgufGpu { gate_up, down, hidden, ff, .. } = self {
            if let Some(chain) = backend.chain() {
                let (hidden, ff) = (*hidden, *ff);
                let mut out = vec![0.0f32; seq * hidden];
                const BATCH: usize = 16;
                for group in active.chunks(BATCH) {
                    let mut x_batches: Vec<Vec<f32>> = Vec::with_capacity(group.len());
                    for (slot, &(_exp, ref toks)) in group.iter().enumerate() {
                        let m = toks.len();
                        let mut xb = Vec::with_capacity(m * hidden);
                        for &(t, _) in toks {
                            xb.extend_from_slice(&x[t * hidden..(t + 1) * hidden]);
                        }
                        scratch.ensure_x(chain, slot, m * hidden);
                        scratch.ensure_y(chain, slot, m * 2 * ff);
                        x_batches.push(xb);
                    }
                    for (slot, xb) in x_batches.iter().enumerate() {
                        chain.upload(scratch.x(slot), xb);
                    }
                    let mut rec = chain.begin();
                    rec.keep_groups(false);
                    for (slot, &(exp, ref toks)) in group.iter().enumerate() {
                        let m = toks.len();
                        rec.matmul_rows(&gate_up[exp], scratch.x(slot), scratch.y(slot), m);
                        rec.read_range(scratch.y(slot), 0, m * 2 * ff);
                    }
                    let gu_outs = rec.finish();

                    let mut mid_batches: Vec<Vec<f32>> = Vec::with_capacity(group.len());
                    for (slot, (gu, &(_exp, ref toks))) in gu_outs.iter().zip(group.iter()).enumerate() {
                        let m = toks.len();
                        let mut mid = vec![0.0f32; m * ff];
                        for row in 0..m {
                            let g = &gu[row * 2 * ff..row * 2 * ff + ff];
                            let u = &gu[row * 2 * ff + ff..(row + 1) * 2 * ff];
                            let dst = &mut mid[row * ff..(row + 1) * ff];
                            for i in 0..ff {
                                dst[i] = situ(g[i], u[i], situ_beta, situ_linear_beta);
                            }
                        }
                        scratch.ensure_x(chain, slot, m * ff);
                        scratch.ensure_y(chain, slot, m * hidden);
                        mid_batches.push(mid);
                    }
                    for (slot, mb) in mid_batches.iter().enumerate() {
                        chain.upload(scratch.x(slot), mb);
                    }
                    let mut rec2 = chain.begin();
                    rec2.keep_groups(false);
                    for (slot, &(exp, ref toks)) in group.iter().enumerate() {
                        let m = toks.len();
                        rec2.matmul_rows(&down[exp], scratch.x(slot), scratch.y(slot), m);
                        rec2.read_range(scratch.y(slot), 0, m * hidden);
                    }
                    let dn_outs = rec2.finish();

                    for (y_batch, &(_exp, ref toks)) in dn_outs.iter().zip(group.iter()) {
                        for (i, &(t, w)) in toks.iter().enumerate() {
                            f32_axpy_fast(
                                &mut out[t * hidden..(t + 1) * hidden],
                                &y_batch[i * hidden..(i + 1) * hidden],
                                w,
                            );
                        }
                    }
                }
                return Ok(out);
            }
        }

        // 2. Evaluate active experts in parallel (zero nested rayon inside `eval_expert`).
        let partial: Result<Vec<Vec<(usize, f32, Vec<f32>)>>> = active
            .into_par_iter()
            .map(|(expert, toks)| {
                let mut local = Vec::with_capacity(toks.len());
                let mut x_batch = Vec::with_capacity(toks.len() * hidden);
                for &(t, _) in &toks {
                    x_batch.extend_from_slice(&x[t * hidden..(t + 1) * hidden]);
                }
                let y_batch = self.eval_expert(
                    expert,
                    &x_batch,
                    toks.len(),
                    situ_beta,
                    situ_linear_beta,
                )?;
                for (i, &(t, w)) in toks.iter().enumerate() {
                    local.push((t, w, y_batch[i * hidden..(i + 1) * hidden].to_vec()));
                }
                Ok(local)
            })
            .collect();

        let partial = partial?;
        let mut out = vec![0.0f32; seq * hidden];
        for items in partial {
            for (t, w, y) in items {
                f32_axpy_fast(&mut out[t * hidden..(t + 1) * hidden], &y, w);
            }
        }
        Ok(out)
    }

    /// Fast decode path (`seq == 1`):
    /// * `GgufGpu`: 2 batched `DeviceChain` submits (8 active `gate_up` -> CPU `SiTU` -> 8 active `down`).
    /// * `Gguf` (host): sequential per-expert chunks (4 quarters × 8 experts = 32 contiguous ~373 KB
    ///   streams across all physical cores, maximizing hardware DRAM prefetching) + AVX-512 reduction.
    /// * `SafetensorsNvfp4` / `SafetensorsMxfp8` / `DenseF32`: flat all-core tiled execution.
    #[allow(clippy::too_many_arguments)]
    fn forward_decode_one(
        &self,
        backend: &dyn Backend,
        scratch: &mut GpuScratch,
        x: &[f32],
        router_logits: &[f32],
        exp_probs_b: Option<&[f32]>,
        n_experts: usize,
        top_k: usize,
        route_scale: f32,
        situ_beta: f32,
        situ_linear_beta: f32,
    ) -> Result<Vec<f32>> {
        const TILE: usize = 32;

        // 1. Top-k sigmoid routing for the single token.
        let mut scores = vec![0.0f32; n_experts];
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let s = 1.0 / (1.0 + (-router_logits[e]).exp());
            scores[e] = s;
            let sel = match exp_probs_b {
                Some(b) => s + b[e],
                None => s,
            };
            scored.push((e, sel));
        }
        scored.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
        let mut sum = 0.0f32;
        for &(e, _) in &scored[..top_k] {
            sum += scores[e];
        }
        let norm = route_scale / (sum + 1e-20);
        let mut chosen: Vec<(usize, f32)> = scored[..top_k]
            .iter()
            .map(|&(e, _)| (e, scores[e] * norm))
            .collect();
        // Sort by expert index for deterministic floating-point accumulation order.
        chosen.sort_unstable_by_key(|&(e, _)| e);

        match self {
            Self::GgufQuantMoe { .. } => {
                unreachable!("GgufQuantMoe handled before forward_decode_one")
            }
            Self::GgufGpu { gate_up, down, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let chain = backend.chain().expect("GgufGpu requires DeviceChain backend");

                // Phase 1: ONE upload of `x` + all `top_k` active `gate_up` matmuls in 1 submit.
                scratch.ensure_x(chain, 0, hidden);
                for slot in 0..top_k {
                    scratch.ensure_y(chain, slot, 2 * ff);
                }
                chain.upload(scratch.x(0), x);
                let mut rec1 = chain.begin();
                rec1.keep_groups(false);
                for (slot, &(exp, _)) in chosen.iter().enumerate() {
                    rec1.matmul_rows(&gate_up[exp], scratch.x(0), scratch.y(slot), 1);
                    rec1.read_range(scratch.y(slot), 0, 2 * ff);
                }
                let gu_outs = rec1.finish();

                // SiTU on host (~8 us) + upload `top_k` mid vectors.
                let mut mids = vec![0.0f32; top_k * ff];
                for (slot, gu) in gu_outs.iter().enumerate() {
                    let g = &gu[..ff];
                    let u = &gu[ff..2 * ff];
                    let dst = &mut mids[slot * ff..(slot + 1) * ff];
                    for i in 0..ff {
                        dst[i] = situ(g[i], u[i], situ_beta, situ_linear_beta);
                    }
                }

                // Phase 2: all `top_k` active `down` matmuls in 1 submit + AVX-512 weighted sum.
                for slot in 0..top_k {
                    scratch.ensure_x(chain, slot, ff);
                    scratch.ensure_y(chain, slot, hidden);
                    chain.upload(scratch.x(slot), &mids[slot * ff..(slot + 1) * ff]);
                }
                let mut rec2 = chain.begin();
                rec2.keep_groups(false);
                for (slot, &(exp, _)) in chosen.iter().enumerate() {
                    rec2.matmul_rows(&down[exp], scratch.x(slot), scratch.y(slot), 1);
                    rec2.read_range(scratch.y(slot), 0, hidden);
                }
                let dn_outs = rec2.finish();

                let mut out = vec![0.0f32; hidden];
                for (y_slot, &(_, weight)) in dn_outs.iter().zip(chosen.iter()) {
                    f32_axpy_fast(&mut out, y_slot, weight);
                }
                Ok(out)
            }
            Self::Gguf { gate_up, down, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let gu_rb = quant_row_bytes(gate_up.dtype, hidden);
                let dn_rb = quant_row_bytes(down.dtype, ff);

                // When thread count <= top_k (e.g. 8 cores on a single 3D V-Cache CCD), run each
                // active expert end-to-end (gate_up -> SiTU -> down) in ONE Rayon barrier so `mid`
                // stays 100% resident in that core's L1d cache with zero cross-core traffic.
                if rayon::current_num_threads() <= top_k {
                    let mut partials = vec![0.0f32; top_k * hidden];
                    partials
                        .par_chunks_mut(hidden)
                        .enumerate()
                        .try_for_each(|(slot, dst)| -> Result<()> {
                            let exp = chosen[slot].0;
                            let gu_raw = gate_up.expert_bytes(exp);
                            let dn_raw = down.expert_bytes(exp);
                            let mut mid = vec![0.0f32; ff];
                            const SUB: usize = 64;
                            let mut g_buf = [0.0f32; SUB];
                            let mut u_buf = [0.0f32; SUB];
                            let mut r0 = 0usize;
                            while r0 < ff {
                                let len = (ff - r0).min(SUB);
                                gguf_dot_rows_any(
                                    gate_up.dtype,
                                    &gu_raw[r0 * gu_rb..(r0 + len) * gu_rb],
                                    x,
                                    len,
                                    hidden,
                                    &mut g_buf[..len],
                                )?;
                                gguf_dot_rows_any(
                                    gate_up.dtype,
                                    &gu_raw[(ff + r0) * gu_rb..(ff + r0 + len) * gu_rb],
                                    x,
                                    len,
                                    hidden,
                                    &mut u_buf[..len],
                                )?;
                                for i in 0..len {
                                    mid[r0 + i] = situ(g_buf[i], u_buf[i], situ_beta, situ_linear_beta);
                                }
                                r0 += len;
                            }
                            gguf_dot_rows_any(down.dtype, dn_raw, &mid, hidden, ff, dst)?;
                            Ok(())
                        })?;
                    let mut out = vec![0.0f32; hidden];
                    for (slot, &(_, weight)) in chosen.iter().enumerate() {
                        f32_axpy_fast(&mut out, &partials[slot * hidden..(slot + 1) * hidden], weight);
                    }
                    return Ok(out);
                }

                // Multi-CCD (>8 threads): split each expert into 4 contiguous quarters (32 sequential
                // ~373 KB streams, zero heap allocs inside parallel tasks) + AVX-512 reduction.
                let ff_chunk = (ff.div_ceil(4).div_ceil(4) * 4).max(32);
                let mut mids = vec![0.0f32; top_k * ff];
                let mut tasks1: Vec<(usize, usize, &mut [f32])> =
                    Vec::with_capacity(top_k * ff.div_ceil(ff_chunk));
                for (slot, mid_slice) in mids.chunks_mut(ff).enumerate() {
                    for (ci, chunk) in mid_slice.chunks_mut(ff_chunk).enumerate() {
                        tasks1.push((slot, ci * ff_chunk, chunk));
                    }
                }
                tasks1
                    .into_par_iter()
                    .try_for_each(|(slot, r0, dst)| -> Result<()> {
                        let total = dst.len();
                        let exp = chosen[slot].0;
                        let raw = gate_up.expert_bytes(exp);
                        const SUB: usize = 64;
                        let mut g_buf = [0.0f32; SUB];
                        let mut u_buf = [0.0f32; SUB];
                        let mut off = 0usize;
                        while off < total {
                            let len = (total - off).min(SUB);
                            let rr = r0 + off;
                            gguf_dot_rows_any(
                                gate_up.dtype,
                                &raw[rr * gu_rb..(rr + len) * gu_rb],
                                x,
                                len,
                                hidden,
                                &mut g_buf[..len],
                            )?;
                            gguf_dot_rows_any(
                                gate_up.dtype,
                                &raw[(ff + rr) * gu_rb..(ff + rr + len) * gu_rb],
                                x,
                                len,
                                hidden,
                                &mut u_buf[..len],
                            )?;
                            for i in 0..len {
                                dst[off + i] = situ(g_buf[i], u_buf[i], situ_beta, situ_linear_beta);
                            }
                            off += len;
                        }
                        Ok(())
                    })?;

                let h_chunk = (hidden.div_ceil(4).div_ceil(4) * 4).max(32);
                let mut partials = vec![0.0f32; top_k * hidden];
                let mut tasks2: Vec<(usize, usize, &mut [f32])> =
                    Vec::with_capacity(top_k * hidden.div_ceil(h_chunk));
                for (slot, part_slice) in partials.chunks_mut(hidden).enumerate() {
                    for (ci, chunk) in part_slice.chunks_mut(h_chunk).enumerate() {
                        tasks2.push((slot, ci * h_chunk, chunk));
                    }
                }
                tasks2
                    .into_par_iter()
                    .try_for_each(|(slot, r0, dst)| -> Result<()> {
                        let len = dst.len();
                        let exp = chosen[slot].0;
                        let mid = &mids[slot * ff..(slot + 1) * ff];
                        let d_raw = &down.expert_bytes(exp)[r0 * dn_rb..(r0 + len) * dn_rb];
                        gguf_dot_rows_any(down.dtype, d_raw, mid, len, ff, dst)?;
                        Ok(())
                    })?;

                let mut out = vec![0.0f32; hidden];
                for (slot, &(_, weight)) in chosen.iter().enumerate() {
                    f32_axpy_fast(&mut out, &partials[slot * hidden..(slot + 1) * hidden], weight);
                }
                Ok(out)
            }
            Self::SafetensorsNvfp4 { spec, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let recs: Vec<Arc<PackedExpertRecord>> = chosen
                    .par_iter()
                    .map(|&(exp, _)| load_nvfp4_record(spec, exp, hidden, ff))
                    .collect::<Result<Vec<_>>>()?;
                let g13s: Vec<f32> = chosen
                    .iter()
                    .map(|&(exp, _)| spec.w13_amax[exp] / spec.w13_denom)
                    .collect();
                let g2s: Vec<f32> = chosen
                    .iter()
                    .map(|&(exp, _)| spec.w2_amax[exp] / spec.w2_denom)
                    .collect();

                let mut mids = vec![0.0f32; top_k * ff];
                let mut tasks: Vec<(usize, usize, &mut [f32])> =
                    Vec::with_capacity(top_k * ff.div_ceil(TILE));
                for (slot, mid_slice) in mids.chunks_mut(ff).enumerate() {
                    for (ti, tile) in mid_slice.chunks_mut(TILE).enumerate() {
                        tasks.push((slot, ti * TILE, tile));
                    }
                }
                tasks.into_par_iter().for_each(|(slot, r0, dst)| {
                    let rec = &recs[slot];
                    let g13 = g13s[slot];
                    for (i, d) in dst.iter_mut().enumerate() {
                        let idx = r0 + i;
                        let gate = nvfp4_dot_row(&rec.w13_bytes, &rec.w13_scale, 2 * idx, hidden, g13, x);
                        let up = nvfp4_dot_row(&rec.w13_bytes, &rec.w13_scale, 2 * idx + 1, hidden, g13, x);
                        *d = situ(gate, up, situ_beta, situ_linear_beta);
                    }
                });

                let mut out = vec![0.0f32; hidden];
                out.par_chunks_mut(TILE).enumerate().for_each(|(ti, dst)| {
                    let r0 = ti * TILE;
                    for (slot, &(_, weight)) in chosen.iter().enumerate() {
                        let rec = &recs[slot];
                        let g2 = g2s[slot];
                        let mid = &mids[slot * ff..(slot + 1) * ff];
                        for (i, d) in dst.iter_mut().enumerate() {
                            *d += weight * nvfp4_dot_row(&rec.w2_bytes, &rec.w2_scale, r0 + i, ff, g2, mid);
                        }
                    }
                });
                Ok(out)
            }
            Self::SafetensorsMxfp8 { spec, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let recs: Vec<Arc<PackedExpertRecord>> = chosen
                    .par_iter()
                    .map(|&(exp, _)| load_mxfp8_record(spec, exp, hidden, ff))
                    .collect::<Result<Vec<_>>>()?;

                let mut mids = vec![0.0f32; top_k * ff];
                let mut tasks: Vec<(usize, usize, &mut [f32])> =
                    Vec::with_capacity(top_k * ff.div_ceil(TILE));
                for (slot, mid_slice) in mids.chunks_mut(ff).enumerate() {
                    for (ti, tile) in mid_slice.chunks_mut(TILE).enumerate() {
                        tasks.push((slot, ti * TILE, tile));
                    }
                }
                tasks.into_par_iter().for_each(|(slot, r0, dst)| {
                    let rec = &recs[slot];
                    for (i, d) in dst.iter_mut().enumerate() {
                        let idx = r0 + i;
                        let gate = mxfp8_dot_row(&rec.w13_bytes, &rec.w13_scale, 2 * idx, hidden, x);
                        let up = mxfp8_dot_row(&rec.w13_bytes, &rec.w13_scale, 2 * idx + 1, hidden, x);
                        *d = situ(gate, up, situ_beta, situ_linear_beta);
                    }
                });

                let mut out = vec![0.0f32; hidden];
                out.par_chunks_mut(TILE).enumerate().for_each(|(ti, dst)| {
                    let r0 = ti * TILE;
                    for (slot, &(_, weight)) in chosen.iter().enumerate() {
                        let rec = &recs[slot];
                        let mid = &mids[slot * ff..(slot + 1) * ff];
                        for (i, d) in dst.iter_mut().enumerate() {
                            *d += weight * mxfp8_dot_row(&rec.w2_bytes, &rec.w2_scale, r0 + i, ff, mid);
                        }
                    }
                });
                Ok(out)
            }
            Self::DenseF32 { gate_up, down, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let mut mids = vec![0.0f32; top_k * ff];
                let mut tasks: Vec<(usize, usize, &mut [f32])> =
                    Vec::with_capacity(top_k * ff.div_ceil(TILE));
                for (slot, mid_slice) in mids.chunks_mut(ff).enumerate() {
                    for (ti, tile) in mid_slice.chunks_mut(TILE).enumerate() {
                        tasks.push((slot, ti * TILE, tile));
                    }
                }
                tasks.into_par_iter().for_each(|(slot, r0, dst)| {
                    let len = dst.len();
                    let exp = chosen[slot].0;
                    let gu_w = &gate_up[exp * 2 * ff * hidden..(exp + 1) * 2 * ff * hidden];
                    let mut g_buf = [0.0f32; TILE];
                    let mut u_buf = [0.0f32; TILE];
                    f32_dot_rows_fast(&gu_w[r0 * hidden..(r0 + len) * hidden], x, len, hidden, &mut g_buf[..len]);
                    f32_dot_rows_fast(&gu_w[(ff + r0) * hidden..(ff + r0 + len) * hidden], x, len, hidden, &mut u_buf[..len]);
                    for i in 0..len {
                        dst[i] = situ(g_buf[i], u_buf[i], situ_beta, situ_linear_beta);
                    }
                });

                let mut out = vec![0.0f32; hidden];
                out.par_chunks_mut(TILE).enumerate().for_each(|(ti, dst)| {
                    let r0 = ti * TILE;
                    let len = dst.len();
                    let mut tmp = [0.0f32; TILE];
                    for (slot, &(exp, weight)) in chosen.iter().enumerate() {
                        let dn_w = &down[exp * hidden * ff..(exp + 1) * hidden * ff];
                        let mid = &mids[slot * ff..(slot + 1) * ff];
                        f32_dot_rows_fast(&dn_w[r0 * ff..(r0 + len) * ff], mid, len, ff, &mut tmp[..len]);
                        for i in 0..len {
                            dst[i] += weight * tmp[i];
                        }
                    }
                });
                Ok(out)
            }
        }
    }

    fn eval_expert(
        &self,
        expert: usize,
        x_batch: &[f32],
        m: usize,
        situ_beta: f32,
        situ_linear_beta: f32,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Gguf { gate_up, down, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let gu = gguf_linear(gate_up, expert, x_batch, m, 2 * ff, hidden)?;
                let mut mid = vec![0.0f32; m * ff];
                for row in 0..m {
                    let g = &gu[row * 2 * ff..row * 2 * ff + ff];
                    let u = &gu[row * 2 * ff + ff..(row + 1) * 2 * ff];
                    let dst = &mut mid[row * ff..(row + 1) * ff];
                    for i in 0..ff {
                        dst[i] = situ(g[i], u[i], situ_beta, situ_linear_beta);
                    }
                }
                gguf_linear(down, expert, &mid, m, hidden, ff)
            }
            Self::GgufGpu { .. } | Self::GgufQuantMoe { .. } => Err(LlamaError::Config(
                "GPU-resident experts require a DeviceChain backend".to_string(),
            )),
            Self::SafetensorsNvfp4 { spec, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let rec = load_nvfp4_record(spec, expert, hidden, ff)?;
                let g13 = spec.w13_amax[expert] / spec.w13_denom;
                let g2 = spec.w2_amax[expert] / spec.w2_denom;
                let mut out = vec![0.0f32; m * hidden];
                let mut mid = vec![0.0f32; ff];
                for row in 0..m {
                    let xr = &x_batch[row * hidden..(row + 1) * hidden];
                    // w13 in safetensors interleaves gate (even row 2*i) and up (odd row 2*i+1).
                    for i in 0..ff {
                        let gate = nvfp4_dot_row(
                            &rec.w13_bytes,
                            &rec.w13_scale,
                            2 * i,
                            hidden,
                            g13,
                            xr,
                        );
                        let up = nvfp4_dot_row(
                            &rec.w13_bytes,
                            &rec.w13_scale,
                            2 * i + 1,
                            hidden,
                            g13,
                            xr,
                        );
                        mid[i] = situ(gate, up, situ_beta, situ_linear_beta);
                    }
                    let yr = &mut out[row * hidden..(row + 1) * hidden];
                    for r in 0..hidden {
                        yr[r] = nvfp4_dot_row(&rec.w2_bytes, &rec.w2_scale, r, ff, g2, &mid);
                    }
                }
                Ok(out)
            }
            Self::SafetensorsMxfp8 { spec, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let rec = load_mxfp8_record(spec, expert, hidden, ff)?;
                let mut out = vec![0.0f32; m * hidden];
                let mut mid = vec![0.0f32; ff];
                for row in 0..m {
                    let xr = &x_batch[row * hidden..(row + 1) * hidden];
                    for i in 0..ff {
                        let gate = mxfp8_dot_row(&rec.w13_bytes, &rec.w13_scale, 2 * i, hidden, xr);
                        let up = mxfp8_dot_row(&rec.w13_bytes, &rec.w13_scale, 2 * i + 1, hidden, xr);
                        mid[i] = situ(gate, up, situ_beta, situ_linear_beta);
                    }
                    let yr = &mut out[row * hidden..(row + 1) * hidden];
                    for r in 0..hidden {
                        yr[r] = mxfp8_dot_row(&rec.w2_bytes, &rec.w2_scale, r, ff, &mid);
                    }
                }
                Ok(out)
            }
            Self::DenseF32 { gate_up, down, hidden, ff, .. } => {
                let (hidden, ff) = (*hidden, *ff);
                let gu_w = &gate_up[expert * 2 * ff * hidden..(expert + 1) * 2 * ff * hidden];
                let dn_w = &down[expert * hidden * ff..(expert + 1) * hidden * ff];
                let mut out = vec![0.0f32; m * hidden];
                let mut mid = vec![0.0f32; ff];
                let mut g_buf = vec![0.0f32; ff];
                let mut u_buf = vec![0.0f32; ff];
                for row in 0..m {
                    let xr = &x_batch[row * hidden..(row + 1) * hidden];
                    f32_dot_rows_fast(&gu_w[..ff * hidden], xr, ff, hidden, &mut g_buf);
                    f32_dot_rows_fast(&gu_w[ff * hidden..], xr, ff, hidden, &mut u_buf);
                    for i in 0..ff {
                        mid[i] = situ(g_buf[i], u_buf[i], situ_beta, situ_linear_beta);
                    }
                    let yr = &mut out[row * hidden..(row + 1) * hidden];
                    f32_dot_rows_fast(dn_w, &mid, hidden, ff, yr);
                }
                Ok(out)
            }
        }
    }
}

fn gguf_linear(
    slice: &GgufExpertSlice,
    expert: usize,
    x: &[f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let dtype = slice.dtype;
    let raw = slice.expert_bytes(expert);
    let mut out = vec![0.0f32; m * n_rows];
    for row in 0..m {
        gguf_dot_rows_any(
            dtype,
            raw,
            &x[row * k..(row + 1) * k],
            n_rows,
            k,
            &mut out[row * n_rows..(row + 1) * n_rows],
        )?;
    }
    Ok(out)
}

fn load_nvfp4_record(
    spec: &Nvfp4LayerSpec,
    expert: usize,
    hidden: usize,
    ff: usize,
) -> Result<Arc<PackedExpertRecord>> {
    spec.cache.get_or_load(spec.layer_idx, expert, || {
        let w13_n = (2 * ff) * (hidden / 2);
        let w13_s = (2 * ff) * (hidden / 16);
        let w2_n = hidden * (ff / 2);
        let w2_s = hidden * (ff / 16);

        let mut w13_bytes = vec![0u8; w13_n];
        let mut w13_scale = vec![0u8; w13_s];
        let mut w2_bytes = vec![0u8; w2_n];
        let mut w2_scale = vec![0u8; w2_s];

        let e = expert as u64;
        read_exact_at(&spec.file, &mut w13_bytes, spec.w13_offset + e * w13_n as u64)
            .map_err(|err| LlamaError::Config(format!("NVFP4 w13 read error: {err}")))?;
        read_exact_at(&spec.file, &mut w13_scale, spec.w13_scale_offset + e * w13_s as u64)
            .map_err(|err| LlamaError::Config(format!("NVFP4 w13_scale read error: {err}")))?;
        read_exact_at(&spec.file, &mut w2_bytes, spec.w2_offset + e * w2_n as u64)
            .map_err(|err| LlamaError::Config(format!("NVFP4 w2 read error: {err}")))?;
        read_exact_at(&spec.file, &mut w2_scale, spec.w2_scale_offset + e * w2_s as u64)
            .map_err(|err| LlamaError::Config(format!("NVFP4 w2_scale read error: {err}")))?;

        Ok(PackedExpertRecord {
            w13_bytes,
            w13_scale,
            w2_bytes,
            w2_scale,
        })
    })
}

fn load_mxfp8_record(
    spec: &Mxfp8LayerSpec,
    expert: usize,
    hidden: usize,
    ff: usize,
) -> Result<Arc<PackedExpertRecord>> {
    spec.cache.get_or_load(spec.layer_idx, expert, || {
        let w13_n = (2 * ff) * hidden;
        let w13_s = (2 * ff) * (hidden / 32);
        let w2_n = hidden * ff;
        let w2_s = hidden * (ff / 32);

        let mut w13_bytes = vec![0u8; w13_n];
        let mut w13_scale = vec![0u8; w13_s];
        let mut w2_bytes = vec![0u8; w2_n];
        let mut w2_scale = vec![0u8; w2_s];

        let e = expert as u64;
        read_exact_at(&spec.file, &mut w13_bytes, spec.w13_offset + e * w13_n as u64)
            .map_err(|err| LlamaError::Config(format!("MXFP8 w13 read error: {err}")))?;
        read_exact_at(&spec.file, &mut w13_scale, spec.w13_scale_offset + e * w13_s as u64)
            .map_err(|err| LlamaError::Config(format!("MXFP8 w13_scale read error: {err}")))?;
        read_exact_at(&spec.file, &mut w2_bytes, spec.w2_offset + e * w2_n as u64)
            .map_err(|err| LlamaError::Config(format!("MXFP8 w2 read error: {err}")))?;
        read_exact_at(&spec.file, &mut w2_scale, spec.w2_scale_offset + e * w2_s as u64)
            .map_err(|err| LlamaError::Config(format!("MXFP8 w2_scale read error: {err}")))?;

        Ok(PackedExpertRecord {
            w13_bytes,
            w13_scale,
            w2_bytes,
            w2_scale,
        })
    })
}

/// Fused dot product of `x` (`len = cols`) against one NVFP4 row (`cols/2` packed nibble bytes,
/// `cols/16` E4M3FN block scales, multiplied by global `outer_scale = amax / denom`).
#[inline]
pub fn nvfp4_dot_row(
    packed: &[u8],
    scales: &[u8],
    row: usize,
    cols: usize,
    outer_scale: f32,
    x: &[f32],
) -> f32 {
    let e4m3 = e4m3_table();
    let (e2m1_lo, e2m1_hi) = e2m1_byte_table();
    let n_blocks = cols / 16;
    let p_row = &packed[row * (cols / 2)..(row + 1) * (cols / 2)];
    let s_row = &scales[row * n_blocks..(row + 1) * n_blocks];
    let mut total = [0.0f32; 8];
    for b in 0..n_blocks {
        let bs = e4m3[s_row[b] as usize];
        let pb = &p_row[b * 8..(b + 1) * 8];
        let xb = &x[b * 16..(b + 1) * 16];
        for j in 0..8 {
            let byte = pb[j] as usize;
            let prod = xb[2 * j] * e2m1_lo[byte] + xb[2 * j + 1] * e2m1_hi[byte];
            total[j] += prod * bs;
        }
    }
    (((total[0] + total[1]) + (total[2] + total[3])) + ((total[4] + total[5]) + (total[6] + total[7])))
        * outer_scale
}

/// Fused dot product of `x` (`len = cols`) against one MXFP8 row (`cols` E4M3FN bytes,
/// `cols/32` UE8M0 power-of-two block scales).
#[inline]
pub fn mxfp8_dot_row(
    packed: &[u8],
    scales: &[u8],
    row: usize,
    cols: usize,
    x: &[f32],
) -> f32 {
    let e4m3 = e4m3_table();
    let ue8m0 = ue8m0_table();
    let n_blocks = cols / 32;
    let p_row = &packed[row * cols..(row + 1) * cols];
    let s_row = &scales[row * n_blocks..(row + 1) * n_blocks];
    let mut total = [0.0f32; 8];
    for b in 0..n_blocks {
        let bs = ue8m0[s_row[b] as usize];
        let pb = &p_row[b * 32..(b + 1) * 32];
        let xb = &x[b * 32..(b + 1) * 32];
        let mut blk = [0.0f32; 8];
        for g in 0..4 {
            let base = g * 8;
            for j in 0..8 {
                blk[j] += xb[base + j] * e4m3[pb[base + j] as usize];
            }
        }
        for j in 0..8 {
            total[j] += blk[j] * bs;
        }
    }
    ((total[0] + total[1]) + (total[2] + total[3])) + ((total[4] + total[5]) + (total[6] + total[7]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_and_nvfp4_and_mxfp8_match_reference() {
        let tbl = e4m3_table();
        // Known E4M3FN values: 0x00=0, 0x38=1.0, 0x40=2.0, 0xb8=-1.0, 0x7e=448.0, 0x7f=NaN
        assert_eq!(tbl[0x00], 0.0);
        assert_eq!(tbl[0x38], 1.0);
        assert_eq!(tbl[0x40], 2.0);
        assert_eq!(tbl[0xb8], -1.0);
        assert_eq!(tbl[0x7e], 448.0);
        assert!(tbl[0x7f].is_nan());
        assert!(tbl[0xff].is_nan());

        // NVFP4: 16 elements = 8 bytes. Nibbles 0..16 -> E2M1_TABLE[0..16], scale 0x40 (2.0), outer 0.5 -> net scale 1.0.
        let packed: Vec<u8> = (0u8..8).map(|j| (2 * j) | ((2 * j + 1) << 4)).collect();
        let scales = vec![0x40u8]; // 2.0
        let x: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let got = nvfp4_dot_row(&packed, &scales, 0, 16, 0.5, &x);
        let want: f32 = (0..16).map(|i| x[i] * E2M1_TABLE[i]).sum();
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");

        // MXFP8: 32 elements, all 0x38 (1.0), scale 128 (2^1 = 2.0).
        let mx_packed = vec![0x38u8; 32];
        let mx_scales = vec![128u8];
        let x32: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let got_mx = mxfp8_dot_row(&mx_packed, &mx_scales, 0, 32, &x32);
        let want_mx: f32 = x32.iter().sum::<f32>() * 2.0;
        assert!((got_mx - want_mx).abs() < 1e-4, "{got_mx} vs {want_mx}");
    }

    #[test]
    fn situ_matches_definition() {
        let g = 3.5f32;
        let u = -12.0f32;
        let got = situ(g, u, 4.0, 25.0);
        let want = (4.0 * (g / 4.0).tanh()) * (1.0 / (1.0 + (-g).exp())) * (25.0 * (u / 25.0).tanh());
        assert!((got - want).abs() < 1e-6);
    }
}
