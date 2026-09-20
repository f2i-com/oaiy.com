//! Load tensors from a GGUF file.
//!
//! Two modes, picked per-tensor based on the GGUF dtype:
//!   * **Quantized weights stay packed** in their native ggml block layout,
//!     wrapped in [`QuantizedTensor`] (Q8_0 and Q4_K supported today). This is
//!     what unblocks ≥7B-parameter models — they no longer get inflated 8× to
//!     F32 at load time.
//!   * **Everything else is dequantized to F32**, including non-quant dtypes
//!     (F32, F16, BF16) and quant types we don't have packed-matmul kernels
//!     for yet (Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, …). Those will fall through to
//!     the CPU dequant + dense linear path.

use std::collections::HashMap;
use std::sync::Arc;

use ggml_quants::GgmlType;
use ggml_rs::{Backend, QuantizedTensor, Tensor};
use gguf::{GgufFile, TensorInfo};

use crate::{LlamaError, ModelConfig, Result};

/// Materialize a GGUF tensor as a contiguous `Tensor<f32>` (always
/// dequantizes if the source is packed). Use for tensors that aren't matmul
/// weights (e.g. norms, embedding tables, KV-cache scales).
pub fn load_tensor_f32(g: &GgufFile, info: &TensorInfo) -> Result<Tensor> {
    // VENDORED-LOCAL: go through the byte-source seam (Cow: borrowed from the
    // mmap, owned from a source-backed file) so both file kinds load here.
    let bytes = g.tensor_bytes(info)?;
    let numel = info.numel() as usize;
    let mut out = vec![0.0f32; numel];

    if !ggml_quants::is_supported(info.dtype) {
        return Err(LlamaError::Quant(ggml_quants::QuantError::Unsupported(info.dtype)));
    }
    ggml_quants::dequantize(info.dtype, &bytes, &mut out)?;

    let mut shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).rev().collect();
    if shape.is_empty() { shape.push(numel); }
    Ok(Tensor::from_vec(out, shape))
}

/// Mmap-backed view into a GGUF file's tensor body. Holds an `Arc<GgufFile>`
/// (the file is itself `Arc`-internal, so this is one Arc clone) plus the
/// absolute byte offset and length. `as_bytes` slices into the live mmap with
/// no copy. Used by `load_tensor_packed` so quantized weights stay zero-copy
/// views into the .gguf file — when VRAM and host RAM are both tight, the OS
/// page cache evicts cold pages to disk transparently (genuine SSD-tier
/// fallback without writing eviction logic ourselves).
struct MmapView {
    file:   GgufFile,
    // VENDORED-LOCAL: which shard of a split GGUF `offset` is relative to.
    // `tensor_data_offset` is shard-relative, so pairing it with shard 0 would
    // read shards 2+ out of the wrong file.
    shard:  usize,
    offset: usize,
    len:    usize,
}

impl ggml_rs::quantized::QuantizedHostBytes for MmapView {
    fn as_bytes(&self) -> &[u8] {
        self.file.raw_slice_shard(self.shard, self.offset, self.len)
    }
}

/// Load a tensor as a packed [`QuantizedTensor`] without dequantizing.
/// Zero-copy: the returned tensor's storage is a view into the GGUF mmap.
fn load_tensor_packed(g: &GgufFile, info: &TensorInfo) -> Result<QuantizedTensor> {
    let mut shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).rev().collect();
    if shape.is_empty() { shape.push(info.numel() as usize); }
    // VENDORED-LOCAL: a source-backed (streaming) file has no mmap to view into —
    // materialize owned bytes instead. Same bytes, same dtype, same kernels.
    // VENDORED-LOCAL: per-shard, not per-file: on a split GGUF one shard may be
    // source-backed while the view offsets belong to another.
    let shard = g.shard_of(info);
    if let Some(src) = g.tensor_source_of(info) {
        let bytes = src.read_tensor(info)?;
        return Ok(QuantizedTensor::from_bytes_cpu(bytes, shape, info.dtype));
    }
    let view = MmapView {
        file:   g.clone(),
        shard,
        offset: g.tensor_data_offset(info),
        len:    info.nbytes() as usize,
    };
    Ok(QuantizedTensor::from_mmap(std::sync::Arc::new(view), shape, info.dtype))
}

/// True if we have a packed-matmul kernel for this dtype today.
// VENDORED-LOCAL: pub(crate) so expert_stream's record->Weight rebuild can
// mirror Weight::load's packed-vs-dense decision.
pub(crate) fn dtype_supports_packed_matmul(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q8_0
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q6_K
    )
}

/// A model weight: either dense F32 or packed quant. Backend-agnostic.
///
/// `TiedEmbed` is a third variant for the LM head when the GGUF has no
/// separate `output.weight` (Llama 3.2-1B, Qwen3.6 family, …): the head reuses
/// the token embedding table. The `Arc<Tensor>` is shared with `tok_embd` so
/// both reference the *same* on-device storage — no dequantize duplication.
/// On big-vocab models (Qwen3.6 vocab=248k × hidden=5120) this saves ~5 GB.
pub enum Weight {
    Dense(Tensor),
    Quant(QuantizedTensor),
    TiedEmbed(Arc<Tensor>),
}

impl std::fmt::Debug for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(t) => write!(f, "Dense({:?})", t),
            Self::Quant(qt) => write!(f, "Quant({:?})", qt),
            Self::TiedEmbed(t) => write!(f, "TiedEmbed({:?})", t),
        }
    }
}

impl Weight {
    /// Load from GGUF, picking packed or dense based on dtype.
    pub fn load(g: &GgufFile, info: &TensorInfo) -> Result<Self> {
        if dtype_supports_packed_matmul(info.dtype) {
            Ok(Self::Quant(load_tensor_packed(g, info)?))
        } else {
            Ok(Self::Dense(load_tensor_f32(g, info)?))
        }
    }

    /// Move onto the backend's preferred storage.
    ///
    /// `TiedEmbed` is left untouched — it's expected to already share its Arc
    /// with the on-device `tok_embd` set up by [`CommonTensors::upload_to`].
    /// Trying to upload it independently would either deep-copy the embed table
    /// (defeats the whole point) or fail outright. Upload tok_embd first via
    /// [`upload_tok_embd_with_tied`].
    pub fn to_device(self, backend: &dyn Backend) -> Self {
        match self {
            Self::Dense(t) => Self::Dense(backend.to_device(t)),
            Self::Quant(qt) => Self::Quant(backend.to_device_quant(qt)),
            Self::TiedEmbed(arc) => Self::TiedEmbed(arc),
        }
    }

    /// Try to place on the backend's preferred storage, but keep host-resident
    /// if VRAM is too tight (with `safety_margin_bytes` reserved for KV cache,
    /// activations, intermediates). When a weight stays host-resident, the
    /// matmul falls through to the CPU op — slower per-tensor but lets us
    /// run models larger than VRAM by mixing placement.
    pub fn try_to_device(self, backend: &dyn Backend, safety_margin_bytes: usize) -> Self {
        match self {
            Self::Dense(t) => Self::Dense(backend.try_to_device(t, safety_margin_bytes)),
            Self::Quant(qt) => Self::Quant(backend.try_to_device_quant(qt, safety_margin_bytes)),
            Self::TiedEmbed(arc) => Self::TiedEmbed(arc),
        }
    }

    /// Dispatch a `linear` call: `y = x · W^T`.
    pub fn linear(&self, backend: &dyn Backend, x: &Tensor) -> Tensor {
        match self {
            Self::Dense(w) => backend.linear(x, w),
            Self::Quant(qw) => backend.linear_q(x, qw),
            Self::TiedEmbed(w) => backend.linear(x, w),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(t) => t.shape(),
            Self::Quant(qt) => qt.shape(),
            Self::TiedEmbed(t) => t.shape(),
        }
    }

    /// Stack two same-dtype, same-inner-dim weights along axis 0:
    ///   `out_shape = [a.shape[0] + b.shape[0], inner]`.
    /// For `Quant(Q4_K | Q5_K | Q6_K | …)` the storage is row-contiguous block
    /// sequences, so this is a byte-concat. For `Dense` it's a row-concat F32
    /// vector. Both inputs must be host-resident — this is intended to run at
    /// load time, before `try_to_device`. Used by FFN gate+up fusion (one matmul
    /// produces `[seq, 2*ff]`, then `silu_mul_split` halves it to `[seq, ff]`).
    pub fn stack_axis0(parts: Vec<Self>) -> Self {
        assert!(!parts.is_empty(), "stack_axis0: need at least one part");
        let first = &parts[0];
        let inner: usize = first.shape().iter().skip(1).product();
        // Validate all parts have the same kind, dtype, and inner shape.
        match first {
            Self::Dense(_) => {
                let mut total_rows = 0usize;
                for p in &parts {
                    let Self::Dense(t) = p else { panic!("stack_axis0: mixed kinds"); };
                    let p_inner: usize = t.shape().iter().skip(1).product();
                    assert_eq!(p_inner, inner, "stack_axis0: inner dim mismatch");
                    total_rows += t.dim(0);
                }
                let mut data = Vec::with_capacity(total_rows * inner);
                for p in parts {
                    let Self::Dense(t) = p else { unreachable!(); };
                    let host = t.to_host();
                    data.extend_from_slice(host.data());
                }
                Self::Dense(Tensor::from_vec(data, vec![total_rows, inner]))
            }
            Self::Quant(qt0) => {
                let dtype = qt0.dtype();
                let mut total_rows = 0usize;
                let mut total_bytes = 0usize;
                for p in &parts {
                    let Self::Quant(qt) = p else { panic!("stack_axis0: mixed kinds"); };
                    assert_eq!(qt.dtype(), dtype, "stack_axis0: dtype mismatch");
                    let p_inner: usize = qt.shape().iter().skip(1).product();
                    assert_eq!(p_inner, inner, "stack_axis0: inner dim mismatch");
                    assert!(qt.is_cpu(), "stack_axis0: parts must be host-resident");
                    total_rows += qt.dim(0);
                    total_bytes += qt.nbytes();
                }
                let mut bytes = Vec::with_capacity(total_bytes);
                for p in parts {
                    let Self::Quant(qt) = p else { unreachable!(); };
                    bytes.extend_from_slice(qt.bytes());
                }
                Self::Quant(QuantizedTensor::from_bytes_cpu(bytes, vec![total_rows, inner], dtype))
            }
            Self::TiedEmbed(_) => panic!("stack_axis0: TiedEmbed is LM-head-only, not stackable"),
        }
    }
}

/// Load the LM head for a model. If `output.weight` (or `lm_head.weight`) is
/// missing the model has tied embeddings — return a `Weight::TiedEmbed` that
/// shares the `Arc<Tensor>` with the caller's `tok_embd`. After a subsequent
/// upload via [`upload_tok_embd_and_lm_head`], both Arcs point at the same
/// on-device storage, eliminating the dequant copy. Saves ~1 GB on
/// Llama 3.2-1B and ~5 GB on Qwen3.6 (vocab=248k).
pub fn load_lm_head_or_tied(
    idx: &TensorIndex<'_>,
    tok_embd: &Arc<Tensor>,
) -> Result<Weight> {
    match idx.take_weight("output.weight", &["lm_head.weight"]) {
        Ok(w) => Ok(w),
        Err(LlamaError::MissingTensor(_)) => Ok(Weight::TiedEmbed(tok_embd.clone())),
        Err(e) => Err(e),
    }
}

/// Upload `tok_embd` to the backend's device storage, fixing up a
/// `Weight::TiedEmbed` LM head to share the same `Arc<Tensor>`. For non-tied
/// LM heads, just uploads tok_embd and returns the head untouched (caller
/// should still upload it separately). Use after [`load_lm_head_or_tied`].
///
/// Invariant: in the tied case, `output` must be the only other Arc holder;
/// otherwise `Arc::try_unwrap` falls through to a deep clone (correct, but
/// negates the memory win — log via debug_assert).
pub fn upload_tok_embd_and_lm_head(
    backend: &dyn Backend,
    tok_embd: Arc<Tensor>,
    output: Weight,
) -> (Arc<Tensor>, Weight) {
    match output {
        Weight::TiedEmbed(lm_arc) => {
            // Drop the second Arc reference so try_unwrap can succeed and we
            // can move the inner Tensor onto the device exactly once.
            drop(lm_arc);
            let inner = Arc::try_unwrap(tok_embd).unwrap_or_else(|arc| {
                debug_assert!(false, "tok_embd Arc has external clones; tied upload will copy");
                (*arc).clone()
            });
            let uploaded = Arc::new(backend.to_device(inner));
            let head = Weight::TiedEmbed(uploaded.clone());
            (uploaded, head)
        }
        other => {
            let inner = Arc::try_unwrap(tok_embd)
                .unwrap_or_else(|arc| (*arc).clone());
            let uploaded = Arc::new(backend.to_device(inner));
            (uploaded, other)
        }
    }
}

/// Index of GGUF tensors by name.
pub struct TensorIndex<'g> {
    by_name: HashMap<&'g str, &'g TensorInfo>,
    file:    &'g GgufFile,
}

impl<'g> TensorIndex<'g> {
    pub fn new(file: &'g GgufFile) -> Self {
        let by_name = file.tensors().iter().map(|t| (t.name.as_str(), t)).collect();
        Self { by_name, file }
    }

    /// Load a tensor as F32 (always dequantizes).
    pub fn take(&self, canonical: &str, alts: &[&str]) -> Result<Tensor> {
        for &name in std::iter::once(&canonical).chain(alts.iter()) {
            if let Some(info) = self.by_name.get(name) {
                return load_tensor_f32(self.file, info);
            }
        }
        Err(LlamaError::MissingTensor(canonical.to_string()))
    }

    /// Load a tensor as a `Weight`, keeping it packed when possible.
    pub fn take_weight(&self, canonical: &str, alts: &[&str]) -> Result<Weight> {
        for &name in std::iter::once(&canonical).chain(alts.iter()) {
            if let Some(info) = self.by_name.get(name) {
                return Weight::load(self.file, info);
            }
        }
        Err(LlamaError::MissingTensor(canonical.to_string()))
    }

    pub fn try_take(&self, name: &str) -> Option<Result<Tensor>> {
        self.by_name.get(name).map(|info| load_tensor_f32(self.file, info))
    }

    pub fn has(&self, name: &str) -> bool { self.by_name.contains_key(name) }
}

/// FFN gate & up projections for a SwiGLU/GeGLU MLP.
///
/// Two layouts:
///   * `Split { gate, up }` — independent weights, two matmuls then
///     `silu_mul` / `gelu_approx_mul`.
///   * `Fused(gate_up)` — byte-stacked along axis 0 (`[2*ff, hidden]`),
///     one matmul producing `[seq, 2*ff]`, then `silu_mul_split` /
///     `gelu_approx_mul_split` halves it back to `[seq, ff]`.
///
/// The loader prefers `Fused` when gate and up share dtype and shape
/// (always true for standard llama-family GGUF dumps); the fused matmul
/// halves the FFN launch count per layer.
#[derive(Debug)]
pub enum FfnPair {
    Split { gate: Weight, up: Weight },
    Fused(Weight),
}

impl FfnPair {
    /// Output dim of one half — i.e. `ff_dim`. For Split this is `gate.shape()[0]`;
    /// for Fused it's `gate_up.shape()[0] / 2`.
    pub fn ff(&self) -> usize {
        match self {
            Self::Split { gate, .. } => gate.shape()[0],
            Self::Fused(w)           => w.shape()[0] / 2,
        }
    }

    // VENDORED-LOCAL: GLM-5.3-Flash.
    /// As [`Self::swiglu`], but clamped at `limit`. `after_silu` picks which side
    /// of the activation the gate clamp lands on — see
    /// [`Backend::swiglu_clamped`]. Stays on the device for both layouts.
    pub fn swiglu_clamped(
        &self,
        backend: &dyn Backend,
        xn: &Tensor,
        limit: f32,
        after_silu: bool,
    ) -> Tensor {
        match self {
            Self::Split { gate, up } => {
                let g = gate.linear(backend, xn);
                let u = up.linear(backend, xn);
                backend.swiglu_clamped(&g, &u, limit, after_silu)
            }
            Self::Fused(w) => {
                let fused = w.linear(backend, xn);
                let ff = fused.dim(fused.rank() - 1) / 2;
                backend.swiglu_clamped_split(&fused, ff, limit, after_silu)
            }
        }
    }

    /// SwiGLU FFN intermediate: `silu(gate(xn)) * up(xn)` -> `[seq, ff]`.
    /// Caller follows up with the down projection.
    pub fn swiglu(&self, backend: &dyn Backend, xn: &Tensor) -> Tensor {
        match self {
            Self::Split { gate, up } => {
                let g = gate.linear(backend, xn);
                let u = up.linear(backend, xn);
                backend.silu_mul(&g, &u)
            }
            Self::Fused(w) => {
                let fused = w.linear(backend, xn);
                let ff = fused.dim(fused.rank() - 1) / 2;
                backend.silu_mul_split(&fused, ff)
            }
        }
    }

    /// GeGLU FFN intermediate: `gelu_approx(gate(xn)) * up(xn)` -> `[seq, ff]`.
    pub fn geglu(&self, backend: &dyn Backend, xn: &Tensor) -> Tensor {
        match self {
            Self::Split { gate, up } => {
                let g = gate.linear(backend, xn);
                let u = up.linear(backend, xn);
                backend.gelu_approx_mul(&g, &u)
            }
            Self::Fused(w) => {
                let fused = w.linear(backend, xn);
                let ff = fused.dim(fused.rank() - 1) / 2;
                backend.gelu_approx_mul_split(&fused, ff)
            }
        }
    }

    /// Move all contained weights onto the backend's preferred storage
    /// (`try_to_device` with `safety_margin_bytes` reserved for KV/activations).
    pub fn try_to_device(self, backend: &dyn Backend, safety_margin_bytes: usize) -> Self {
        match self {
            Self::Split { gate, up } => Self::Split {
                gate: gate.try_to_device(backend, safety_margin_bytes),
                up:   up.try_to_device(backend, safety_margin_bytes),
            },
            Self::Fused(w) => Self::Fused(w.try_to_device(backend, safety_margin_bytes)),
        }
    }

    /// Build from two halves, fusing if both sides are byte-compatible
    /// (same dtype + same shape, host-resident). Otherwise fall back to Split.
    pub fn from_halves(gate: Weight, up: Weight) -> Self {
        let can_fuse = match (&gate, &up) {
            (Weight::Quant(g), Weight::Quant(u)) => {
                g.dtype() == u.dtype() && g.shape() == u.shape() && g.is_cpu() && u.is_cpu()
            }
            (Weight::Dense(g), Weight::Dense(u)) => g.shape() == u.shape(),
            _ => false,
        };
        if can_fuse {
            Self::Fused(Weight::stack_axis0(vec![gate, up]))
        } else {
            Self::Split { gate, up }
        }
    }

    /// Force a Fused pair back into Split form by byte-splitting the stacked
    /// weight along axis 0. Used by gemma3n where activation sparsity is
    /// applied to `gate` between the matmul and the activation, so the unfused
    /// path is required. Only valid for host-resident Fused weights.
    pub fn into_split(self) -> Self {
        match self {
            Self::Split { .. } => self,
            Self::Fused(w) => {
                let (gate, up) = split_weight_axis0_halves(w);
                Self::Split { gate, up }
            }
        }
    }
}

/// Split a Weight along axis 0 into two equal halves. Assumes the input has
/// even-numbered first dimension; for Quant, both halves are valid block-quant
/// rows because per-row block alignment is preserved by the byte split.
fn split_weight_axis0_halves(w: Weight) -> (Weight, Weight) {
    match w {
        Weight::Dense(t) => {
            let total_rows = t.dim(0);
            assert!(total_rows % 2 == 0, "split_weight_axis0_halves: odd row count {total_rows}");
            let half = total_rows / 2;
            let inner: usize = t.shape().iter().skip(1).product::<usize>().max(1);
            let host = t.to_host();
            let src = host.data();
            let split_at = half * inner;
            let gate = Tensor::from_vec(src[..split_at].to_vec(),       vec![half, inner]);
            let up   = Tensor::from_vec(src[split_at..].to_vec(),       vec![half, inner]);
            (Weight::Dense(gate), Weight::Dense(up))
        }
        Weight::Quant(qt) => {
            assert!(qt.is_cpu(), "split_weight_axis0_halves: must be host-resident");
            let total_rows = qt.dim(0);
            assert!(total_rows % 2 == 0, "split_weight_axis0_halves: odd row count {total_rows}");
            let half = total_rows / 2;
            let inner: usize = qt.shape().iter().skip(1).product::<usize>().max(1);
            let dtype = qt.dtype();
            let bytes = qt.bytes();
            let split_at = bytes.len() / 2;
            let gate_bytes = bytes[..split_at].to_vec();
            let up_bytes   = bytes[split_at..].to_vec();
            let gate = QuantizedTensor::from_bytes_cpu(gate_bytes, vec![half, inner], dtype);
            let up   = QuantizedTensor::from_bytes_cpu(up_bytes,   vec![half, inner], dtype);
            (Weight::Quant(gate), Weight::Quant(up))
        }
        Weight::TiedEmbed(_) => panic!("split_weight_axis0_halves: TiedEmbed is LM-head-only"),
    }
}

/// Tensors shared across all Llama-family architectures.
#[derive(Debug)]
pub struct CommonBlockTensors {
    pub attn_norm:   Tensor,
    pub attn_q:      Weight,
    pub attn_k:      Weight,
    pub attn_v:      Weight,
    pub attn_output: Weight,
    pub ffn_norm:    Tensor,
    pub ffn_pair:    FfnPair,
    pub ffn_down:    Weight,
}

#[derive(Debug)]
pub struct CommonTensors {
    /// Token embedding lookup table. Always dense F32 (we don't yet have a
    /// packed-gather kernel; embedding tables are usually a small fraction of
    /// total model bytes).
    /// Token embedding table. `Arc`-shared with the LM head when tied (saves a
    /// dequantize copy of the embed table — 1 GB on Llama 3.2-1B, 5 GB on
    /// Qwen3.6 248k-vocab).
    pub tok_embd:    Arc<Tensor>,
    pub output_norm: Tensor,
    /// LM head — packed if quantized in the GGUF, or `TiedEmbed` sharing the
    /// `tok_embd` Arc when the GGUF has no separate `output.weight`.
    pub output:      Weight,
    pub blocks:      Vec<CommonBlockTensors>,
}

impl CommonTensors {
    pub fn load(g: &GgufFile, cfg: &ModelConfig) -> Result<Self> {
        Self::load_with_index(&TensorIndex::new(g), cfg)
    }

    pub fn load_with_index(idx: &TensorIndex<'_>, cfg: &ModelConfig) -> Result<Self> {
        let tok_embd = Arc::new(idx.take("token_embd.weight", &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(idx, &tok_embd)?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let ffn_gate = idx.take_weight(&format!("blk.{i}.ffn_gate.weight"), &[])?;
            let ffn_up   = idx.take_weight(&format!("blk.{i}.ffn_up.weight"),   &[])?;
            blocks.push(CommonBlockTensors {
                attn_norm:   idx.take(&format!("blk.{i}.attn_norm.weight"),    &[])?,
                attn_q:      idx.take_weight(&format!("blk.{i}.attn_q.weight"),      &[])?,
                attn_k:      idx.take_weight(&format!("blk.{i}.attn_k.weight"),      &[])?,
                attn_v:      idx.take_weight(&format!("blk.{i}.attn_v.weight"),      &[])?,
                attn_output: idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?,
                ffn_norm:    idx.take(&format!("blk.{i}.ffn_norm.weight"),    &[])?,
                // Auto-fuse gate+up if dtype/shape match — saves one matmul launch
                // per FFN per layer (silu_mul_split does the back-end split).
                ffn_pair:    FfnPair::from_halves(ffn_gate, ffn_up),
                ffn_down:    idx.take_weight(&format!("blk.{i}.ffn_down.weight"),    &[])?,
            });
        }

        Self::verify_shapes(&tok_embd, &output_norm, &output, &blocks, cfg)?;
        Ok(Self { tok_embd, output_norm, output, blocks })
    }

    fn verify_shapes(
        tok_embd:    &Tensor,
        output_norm: &Tensor,
        output:      &Weight,
        blocks:      &[CommonBlockTensors],
        cfg:         &ModelConfig,
    ) -> Result<()> {
        let expect_dense = |t: &Tensor, expected: &[usize], name: &str| -> Result<()> {
            if t.shape() != expected {
                Err(LlamaError::BadTensorShape {
                    name: name.into(),
                    got: t.shape().iter().map(|&v| v as u64).collect(),
                    expected: expected.iter().map(|&v| v as u64).collect(),
                })
            } else { Ok(()) }
        };
        let expect_w = |w: &Weight, expected: &[usize], name: &str| -> Result<()> {
            if w.shape() != expected {
                Err(LlamaError::BadTensorShape {
                    name: name.into(),
                    got: w.shape().iter().map(|&v| v as u64).collect(),
                    expected: expected.iter().map(|&v| v as u64).collect(),
                })
            } else { Ok(()) }
        };

        expect_dense(tok_embd,    &[cfg.vocab_size, cfg.embedding_dim], "token_embd")?;
        expect_dense(output_norm, &[cfg.embedding_dim],                 "output_norm")?;
        expect_w(output,          &[cfg.vocab_size, cfg.embedding_dim], "output")?;

        for (i, b) in blocks.iter().enumerate() {
            // Per-layer head_dim (Gemma 4) means q/k/v sizes vary by layer.
            let hd = cfg.layer_head_dim(i);
            let q_dim = cfg.n_heads * hd;
            let k_dim = cfg.n_kv_heads * hd;
            expect_dense(&b.attn_norm,   &[cfg.embedding_dim],             &format!("blk.{i}.attn_norm"))?;
            expect_w(&b.attn_q,      &[q_dim, cfg.embedding_dim],      &format!("blk.{i}.attn_q"))?;
            expect_w(&b.attn_k,      &[k_dim, cfg.embedding_dim],      &format!("blk.{i}.attn_k"))?;
            expect_w(&b.attn_v,      &[k_dim, cfg.embedding_dim],      &format!("blk.{i}.attn_v"))?;
            expect_w(&b.attn_output, &[cfg.embedding_dim, q_dim],      &format!("blk.{i}.attn_output"))?;
            expect_dense(&b.ffn_norm,    &[cfg.embedding_dim],             &format!("blk.{i}.ffn_norm"))?;
            // Per-layer FFN dim (Gemma 3n / Gemma 4 store as array). Falls back
            // to scalar `ff_dim`; if that's 0 too (no metadata), skip the check
            // and trust the loaded shape.
            let ff = cfg.layer_ff_dim(i);
            if ff != 0 {
                match &b.ffn_pair {
                    FfnPair::Split { gate, up } => {
                        expect_w(gate, &[ff, cfg.embedding_dim], &format!("blk.{i}.ffn_gate"))?;
                        expect_w(up,   &[ff, cfg.embedding_dim], &format!("blk.{i}.ffn_up"))?;
                    }
                    FfnPair::Fused(w) => {
                        expect_w(w, &[2 * ff, cfg.embedding_dim], &format!("blk.{i}.ffn_gate_up"))?;
                    }
                }
                expect_w(&b.ffn_down,    &[cfg.embedding_dim, ff], &format!("blk.{i}.ffn_down"))?;
            }
        }
        Ok(())
    }

    /// Move all weights onto the backend's preferred storage. CPU = no-op.
    /// Big Weights (q/k/v/output, ffn_gate/up/down) use `try_to_device` with a
    /// 2 GB safety margin so they fall back to host-resident if VRAM is too
    /// tight; the matmul path then dispatches to the CPU op transparently.
    /// Norms are always device-resident (tiny, hot).
    pub fn upload_to(self, backend: &dyn Backend) -> Self {
        const M: usize = 2 * 1024 * 1024 * 1024;
        // Tied LM head: upload tok_embd once and share the Arc with the head.
        let (tok_embd, output) = upload_tok_embd_and_lm_head(backend, self.tok_embd, self.output);
        let output = output.try_to_device(backend, M);
        Self {
            tok_embd,
            output_norm: backend.to_device(self.output_norm),
            output,
            blocks: self.blocks.into_iter().map(|b| CommonBlockTensors {
                attn_norm:   backend.to_device(b.attn_norm),
                attn_q:      b.attn_q.try_to_device(backend, M),
                attn_k:      b.attn_k.try_to_device(backend, M),
                attn_v:      b.attn_v.try_to_device(backend, M),
                attn_output: b.attn_output.try_to_device(backend, M),
                ffn_norm:    backend.to_device(b.ffn_norm),
                ffn_pair:    b.ffn_pair.try_to_device(backend, M),
                ffn_down:    b.ffn_down.try_to_device(backend, M),
            }).collect(),
        }
    }
}

// ----- backwards-compat aliases ----------------------------------------------

pub type ModelTensors = CommonTensors;
pub type BlockTensors = CommonBlockTensors;
