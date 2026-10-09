//! mmproj.gguf loader + SigLIP-400M vision tower scaffolding.
//!
//! `mmproj.gguf` is the companion file shipped alongside multimodal models
//! (Gemma 3, Gemma 4, Llava). It contains:
//!   * **Vision tower** (`v.*` tensors): SigLIP-400M for Gemma, CLIP-ViT-L/14
//!     for Llava. A vanilla pre-LN ViT — patch convolution → position embed →
//!     N transformer blocks → final LN.
//!   * **Multimodal projector** (`mm.*` tensors): a small adapter that maps
//!     the vision-tower output [n_patches, vision_dim] into the LM's text
//!     embedding space [n_soft_tokens, lm_hidden_dim].
//!
//! Status (this commit): **loader + config only**. The forward pass is laid
//! out as a sequence of stub functions; each will be filled in once the
//! supporting backend ops exist (layer_norm, bias-add, image-to-patches
//! unfold). Loading works against any standard mmproj.gguf and validates
//! shapes, so we'll catch tensor-naming / dimension mismatches the moment a
//! real file is supplied — no guesswork at forward time.

use std::sync::Arc;

use ggml_rs::{Backend, Tensor};
use gguf::GgufFile;

use crate::loader::{TensorIndex, Weight};
use crate::{LlamaError, Result};

mod gemma4v;
mod qwen3vl;
mod siglip;

// (what this file gave the crate and its users, and what the files make for each other)
pub use {gemma4v::*, qwen3vl::*, siglip::*};

/// Which downstream LM family this mmproj feeds. Determines both the
/// vision-tower architecture and the projector adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorKind {
    /// Gemma 3 (`projector_type = "gemma3"`): vanilla SigLIP-400M tower
    /// (pre-LN ViT, GeLU MLP, biases everywhere, post-LN) + 16× spatial
    /// avg-pool (4096 → 256 patches) → RMSNorm (`mm.soft_emb_norm.weight`)
    /// → linear (`mm.input_projection.weight`). 256 soft tokens out.
    Gemma3,
    /// Llava-1.5 (`projector_type = "mlp"`): CLIP-ViT-L/14 vanilla tower +
    /// linear (`mm.0.{weight,bias}`) → GeLU → linear (`mm.2.{weight,bias}`).
    /// One soft token per patch — 576 for CLIP-L/14.
    Mlp,
    /// Gemma 4 (`projector_type = "gemma4v"`): bespoke RMSNorm-everywhere
    /// vision tower (no biases, SwiGLU FFN, sandwich norms, per-head Q/K-norm,
    /// multi-tile position embedding) + single linear projector. **Loader
    /// recognises but does not yet support this variant** — needs a separate
    /// forward pipeline; tracked under task #45 Step 4.
    Gemma4V,
    /// Qwen3-VL (`projector_type = "qwen3vl_merger"`): pre-LN ViT with FUSED
    /// `attn_qkv` projection (vs separate q/k/v in SigLIP), GeLU MLP with biases,
    /// 27 blocks at hidden=1152, 768×768 input / patch=16. Plus DeepStack:
    /// per-block bool flag (`clip.vision.is_deepstack_layers`) marks blocks
    /// whose outputs are also extracted, normed, projected via per-block
    /// `deepstack_norm`/`fc1`/`fc2`, and concatenated as additional context
    /// tokens. Projector head is a 2-layer MLP (`mm.0.weight` 4608×4608 →
    /// GELU → `mm.2.weight` 4608×5120) preceded by a 2×2 spatial merge that
    /// concatenates 4 neighbouring patches into one 4608-dim row. **Loader
    /// recognises but does not yet support this variant** — tracked under #84.
    Qwen3Vl,
}

impl ProjectorKind {
    /// Returns the tensor-name prefix layout this projector uses for the
    /// vanilla pre-LN ViT tower (for variants that have one). Returns None
    /// for variants whose tower deviates structurally (e.g. Gemma4V).
    pub fn supports_siglip_loader(&self) -> bool {
        matches!(self, Self::Gemma3 | Self::Mlp)
    }
}

/// Parsed `clip.*` metadata describing the vision tower + projector. Mirrors
/// the schema llama.cpp's `clip.cpp` writes out when converting from
/// HF-format weights.
#[derive(Debug, Clone)]
pub struct MmProjConfig {
    /// Square input side. SigLIP-Gemma: 896. CLIP-L/14: 336.
    pub image_size: usize,
    /// Patch side in pixels. SigLIP-Gemma: 14. CLIP-L/14: 14.
    pub patch_size: usize,
    /// ViT hidden dim. SigLIP-400M: 1152. CLIP-L/14: 1024.
    pub embedding_dim: usize,
    /// Number of transformer blocks. SigLIP-400M: 27. CLIP-L/14: 24.
    pub n_layers: usize,
    /// Multi-head count. SigLIP-400M: 16. CLIP-L/14: 16.
    pub n_heads: usize,
    /// Per-head dim — derived as `embedding_dim / n_heads`.
    pub head_dim: usize,
    /// FFN inner dim. SigLIP-400M: 4304. CLIP-L/14: 4096.
    pub ff_dim: usize,
    /// LayerNorm epsilon — SigLIP / CLIP both use 1e-6 by default; we still
    /// read it from metadata in case the converter wrote a different value.
    pub layer_norm_eps: f32,
    /// Which projector head to use. Inferred from `clip.projector_type`
    /// metadata; defaults to Gemma if absent (since Gemma 3 4B / E2B is the
    /// primary target).
    pub projector: ProjectorKind,
    /// Per-channel pixel mean for normalisation. Read from
    /// `clip.vision.image_mean` if present, else SigLIP default `[0.5; 3]`.
    pub mean: [f32; 3],
    /// Per-channel pixel std. Read from `clip.vision.image_std`, else `[0.5; 3]`.
    pub std: [f32; 3],
}

impl MmProjConfig {
    /// Number of raw patches (before any projector pooling): `(image_size/patch_size)^2`.
    pub fn n_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Number of soft tokens emitted to the LM after the projector.
    /// Gemma3 collapses 4×4 patch blocks via avg-pool, so 4096 → 256. Mlp
    /// passes patches through 1:1. Gemma4V's pooling scheme isn't published;
    /// returns the raw patch count as a placeholder until the loader lands.
    pub fn n_soft_tokens(&self) -> usize {
        match self.projector {
            ProjectorKind::Gemma3  => self.n_patches() / 16,
            ProjectorKind::Mlp     => self.n_patches(),
            ProjectorKind::Gemma4V => self.n_patches(),
            // Qwen3-VL: 2x2 spatial merge → patches/4. For 768/16 = 48-side
            // patch grid, that's 24×24 = 576 soft tokens per image.
            ProjectorKind::Qwen3Vl => self.n_patches() / 4,
        }
    }

    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        // Same get_u64/get_f32 pattern as ModelConfig, but always under
        // the `clip.*` namespace — mmproj.gguf has its own architecture name.
        let get_u64 = |key: &str| -> Result<u64> { Ok(g.get_u64(key)?) };
        let get_f32 = |key: &str| -> Result<f32> { Ok(g.get_f32(key)?) };

        let image_size    = get_u64("clip.vision.image_size")? as usize;
        let patch_size    = get_u64("clip.vision.patch_size")? as usize;
        let embedding_dim = get_u64("clip.vision.embedding_length")? as usize;
        let ff_dim        = get_u64("clip.vision.feed_forward_length")? as usize;
        let n_layers      = get_u64("clip.vision.block_count")? as usize;
        let n_heads       = get_u64("clip.vision.attention.head_count")? as usize;
        let layer_norm_eps = get_f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6);

        if n_heads == 0 || embedding_dim == 0 {
            return Err(LlamaError::Config("zero-sized vision dimension".into()));
        }
        if embedding_dim % n_heads != 0 {
            return Err(LlamaError::Config(format!(
                "vision embedding_dim ({embedding_dim}) not divisible by n_heads ({n_heads})"
            )));
        }
        let head_dim = embedding_dim / n_heads;

        // The schema for `projector_type` differs across mmproj versions:
        //   * Gemma 3 mmprojs (single-modality): `clip.projector_type`
        //   * Gemma 4 mmprojs (vision + audio): `clip.vision.projector_type`
        // and a separate `clip.audio.projector_type` for the audio tower.
        // Try the vision-namespaced key first, fall back to the older one.
        let projector_type = g.metadata().get("clip.vision.projector_type")
            .or_else(|| g.metadata().get("clip.projector_type"))
            .and_then(|v| v.as_str());
        let projector = match projector_type {
            Some("gemma3")          => ProjectorKind::Gemma3,
            Some("gemma4v")         => ProjectorKind::Gemma4V,
            Some("qwen3vl_merger")  => ProjectorKind::Qwen3Vl,
            Some("mlp")     => ProjectorKind::Mlp,
            None            => ProjectorKind::Gemma3,  // legacy default
            Some(other)     => return Err(LlamaError::Config(
                format!("unsupported clip.projector_type {other:?}")
            )),
        };

        // Image-norm constants. Read array-form if present, else fall back to
        // SigLIP's centered-and-scaled default (which is also what HF's SigLIP
        // image processor emits).
        let mean = read_three_f32(g, "clip.vision.image_mean").unwrap_or([0.5, 0.5, 0.5]);
        let std  = read_three_f32(g, "clip.vision.image_std" ).unwrap_or([0.5, 0.5, 0.5]);

        Ok(Self { image_size, patch_size, embedding_dim, n_layers, n_heads,
                  head_dim, ff_dim, layer_norm_eps, projector, mean, std })
    }
}

/// Host-side 2D transpose: `[r, c]` → `[c, r]`. Used for the Gemma projector's
/// `mm.input_projection.weight`, which is stored in HF `nn.Parameter` order
/// `[vision_dim, lm_hidden]` rather than `nn.Linear`'s `[out, in]`. Errors if
/// the input isn't 2D.
fn transpose_2d_host(t: Tensor) -> Result<Tensor> {
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(LlamaError::Config(format!(
            "transpose_2d_host expects 2D tensor, got shape {shape:?}"
        )));
    }
    let r = shape[0];
    let c = shape[1];
    let host = t.to_host();
    let src = host.data();
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = src[i * c + j];
        }
    }
    Ok(Tensor::from_vec(out, vec![c, r]))
}

fn read_three_f32(g: &GgufFile, key: &str) -> Option<[f32; 3]> {
    let arr = g.metadata().get(key)?.as_array()?;
    if let gguf::Array::F32(v) = arr {
        if v.len() == 3 { return Some([v[0], v[1], v[2]]); }
    }
    None
}

/// Loaded mmproj weights. Top-level dispatcher over the supported vision-tower
/// flavours: vanilla SigLIP/CLIP (Gemma 3, Llava), Gemma 4's bespoke
/// RMSNorm/SwiGLU/sandwich-norm tower, and Qwen3-VL's fused-QKV ViT with
/// 2x2 spatial-merge MLP head. Construct with [`MmProj::from_gguf`].
#[derive(Debug)]
pub enum MmProj {
    /// Vanilla SigLIP/CLIP-style ViT (Gemma 3, Llava).
    SigLip(SigLipMmProj),
    /// Gemma 4's bespoke vision tower — RMSNorm everywhere, SwiGLU FFN,
    /// sandwich norms, per-head Q/K-norm, no biases. Single linear projector.
    Gemma4V(Gemma4VMmProj),
    /// Qwen3-VL ViT — pre-LN (LayerNorm), fused QKV with bias, GeLU MLP with
    /// biases, post-LN, then 2x2 spatial merge → 2-layer MLP projector head
    /// (`mm.0` → GELU → `mm.2`). 768x768 input @ patch=16 → 2304 patches →
    /// 576 soft tokens at 5120-dim.
    Qwen3Vl(Qwen3VlMmProj),
}

impl MmProj {
    /// Open an mmproj.gguf and load it via the right loader for its
    /// projector type. Either flavour validates tensor shapes against the
    /// parsed config — a mismatch surfaces as `BadTensorShape` rather than a
    /// downstream NaN.
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = MmProjConfig::from_gguf(g)?;
        let _ = backend;
        match config.projector {
            ProjectorKind::Gemma3 | ProjectorKind::Mlp =>
                Ok(Self::SigLip(SigLipMmProj::from_gguf_with_config(g, config, backend.clone())?)),
            ProjectorKind::Gemma4V =>
                Ok(Self::Gemma4V(Gemma4VMmProj::from_gguf_with_config(g, config, backend.clone())?)),
            ProjectorKind::Qwen3Vl =>
                Ok(Self::Qwen3Vl(Qwen3VlMmProj::from_gguf_with_config(g, config, backend.clone())?)),
        }
    }

    pub fn config(&self) -> &MmProjConfig {
        match self {
            Self::SigLip(m)  => &m.config,
            Self::Gemma4V(m) => &m.config,
            Self::Qwen3Vl(m) => &m.config,
        }
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        match self {
            Self::SigLip(m)  => &m.backend,
            Self::Gemma4V(m) => &m.backend,
            Self::Qwen3Vl(m) => &m.backend,
        }
    }

    pub fn n_blocks(&self) -> usize {
        match self {
            Self::SigLip(m)  => m.blocks.len(),
            Self::Gemma4V(m) => m.blocks.len(),
            Self::Qwen3Vl(m) => m.blocks.len(),
        }
    }

    /// Run the full vision pipeline: image → soft tokens in LM hidden space.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        match self {
            Self::SigLip(m)  => m.forward(image),
            Self::Gemma4V(m) => m.forward(image),
            Self::Qwen3Vl(m) => m.forward(image),
        }
    }
}

/// Reshape a CHW image `[3, H, W]` into row-major patches `[N, 3*P*P]` where
/// row `n` corresponds to patch `(ny, nx) = (n / W_p, n % W_p)` and within a
/// row the layout is `(channel, py, px)` — matching how the GGUF conv weight
/// is stored, so `patches @ W_flat^T` reproduces a stride-P 2D convolution
/// exactly. Always runs on host (image is host-resident from preprocess).
fn unfold_patches_to_host(img: &Tensor, p: usize) -> Result<Tensor> {
    let host = img.to_host();
    let s = host.shape();
    if s.len() != 3 || s[0] != 3 || s[1] % p != 0 || s[2] % p != 0 {
        return Err(LlamaError::Config(format!(
            "image shape {:?} not divisible by patch_size {p} (expected [3, H, W])", s
        )));
    }
    let h = s[1];
    let w = s[2];
    let n_y = h / p;
    let n_x = w / p;
    let n_patches = n_y * n_x;
    let in_dim = 3 * p * p;
    let img_data = host.data();
    let mut out = vec![0.0f32; n_patches * in_dim];
    for ny in 0..n_y {
        for nx in 0..n_x {
            let n = ny * n_x + nx;
            let dst_off = n * in_dim;
            for c in 0..3 {
                let chan_off = c * h * w;
                for ky in 0..p {
                    let src_row_off = chan_off + (ny * p + ky) * w + nx * p;
                    let dst_row_off = dst_off + (c * p + ky) * p;
                    out[dst_row_off..dst_row_off + p]
                        .copy_from_slice(&img_data[src_row_off..src_row_off + p]);
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, vec![n_patches, in_dim]))
}

/// LayerNorm over the last axis: `y = (x - mean) / sqrt(var + eps) * w + b`.
/// Different from RMSNorm (which doesn't subtract mean and has no bias) — the
/// vision tower needs this because SigLIP / CLIP both use full LayerNorm.
/// Host-side fallback; `to_host` + `to_device` round-trip is fine because
/// the vision pass runs once per image, not per token.
fn layer_norm(b: &dyn Backend, x: &Tensor, w: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let mut host = x.to_host();
    let last = host.dim(host.rank() - 1);
    let n_rows = host.numel() / last;
    let w_host = w.to_host();
    let b_host = bias.to_host();
    let w_data = w_host.data();
    let b_data = b_host.data();
    let data = host.data_mut();
    for r in 0..n_rows {
        let off = r * last;
        let row = &mut data[off..off + last];
        let mean = row.iter().sum::<f32>() / last as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / last as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..last {
            row[j] = (row[j] - mean) * inv * w_data[j] + b_data[j];
        }
    }
    b.to_device(host)
}

/// `x[..., j] += bias[j]` — broadcast-add along the last axis. Counterpart
/// to `Backend::mul_inplace_broadcast_last`. Used after every linear in the
/// vision tower, since SigLIP / CLIP attention and FFN projections all have
/// per-channel biases (text-tower Llama / Gemma do not).
fn add_bias_inplace(b: &dyn Backend, x: &mut Tensor, bias: &Tensor) {
    let last = bias.numel();
    debug_assert_eq!(x.dim(x.rank() - 1), last);
    let mut host = x.to_host();
    let bias_host = bias.to_host();
    let bias_data = bias_host.data();
    let n_rows = host.numel() / last;
    let data = host.data_mut();
    for r in 0..n_rows {
        let off = r * last;
        for j in 0..last { data[off + j] += bias_data[j]; }
    }
    *x = b.to_device(host);
}

#[cfg(test)]
mod tests {
    use super::*;

    use ggml_rs::default_backend;

    /// Round-trip a tiny [3, 4, 4] image with patch_size=2 → [4, 12].
    /// Each patch should contain its 4 (channel * 2*2) pixel values in the
    /// (c, ky, kx) order matching how the GGUF conv weight is laid out.
    #[test]
    fn unfold_patches_layout() {
        // Build [3, 4, 4] with channel c filled with c+1 in every pixel.
        let mut data = Vec::new();
        for c in 0..3 {
            for _y in 0..4 {
                for _x in 0..4 {
                    data.push((c + 1) as f32);
                }
            }
        }
        let img = Tensor::from_vec(data, vec![3, 4, 4]);
        let patches = unfold_patches_to_host(&img, 2).unwrap();
        // 4 patches, each with 3*2*2 = 12 elements.
        assert_eq!(patches.shape(), &[4, 12]);
        // Within each row: 4 pixels of channel 0 (=1.0), then 4 of channel 1
        // (=2.0), then 4 of channel 2 (=3.0).
        for row_n in 0..4 {
            for j in 0..4  { assert_eq!(patches.data()[row_n * 12 + j],     1.0); }
            for j in 4..8  { assert_eq!(patches.data()[row_n * 12 + j],     2.0); }
            for j in 8..12 { assert_eq!(patches.data()[row_n * 12 + j],     3.0); }
        }
    }

    /// Per-patch positional check: a uniform-channel image still respects the
    /// (ny, nx) → (y, x) sub-region mapping. Set channel 0 to a per-pixel
    /// value `y * 4 + x` so we can read out where each patch came from.
    #[test]
    fn unfold_patches_positional() {
        let mut data = vec![0.0f32; 3 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                // channel 0 only; channels 1, 2 stay 0.
                data[y * 4 + x] = (y * 4 + x) as f32;
            }
        }
        let img = Tensor::from_vec(data, vec![3, 4, 4]);
        let patches = unfold_patches_to_host(&img, 2).unwrap();
        // Patch (0, 0) = upper-left 2x2 = pixels (0,0), (0,1), (1,0), (1,1)
        //                                     = 0,    1,    4,    5
        // Layout: row 0 of channel 0 = pixels (0,0) and (0,1); row 1 = (1,0), (1,1).
        let p00 = &patches.data()[0..12];
        assert_eq!(&p00[0..4], &[0.0, 1.0, 4.0, 5.0]);
        // Patch (1, 1) = lower-right 2x2 = pixels (2,2), (2,3), (3,2), (3,3)
        //                                       = 10,   11,   14,   15
        let p11 = &patches.data()[36..48];
        assert_eq!(&p11[0..4], &[10.0, 11.0, 14.0, 15.0]);
    }

    /// Verify LayerNorm output: with weight=1, bias=0, `(x - mean) / std`
    /// should produce a row with mean ≈ 0 and var ≈ 1.
    #[test]
    fn layer_norm_matches_definition() {
        let backend = default_backend();
        // Two rows of length 4: [1, 2, 3, 4] and [10, 20, 30, 40].
        // Mean1 = 2.5, var1 = 1.25; Mean2 = 25, var2 = 125.
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], vec![2, 4]);
        let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
        let y = layer_norm(&*backend, &x, &w, &b, 1e-5);
        let d = y.data();
        // Each row should sum to ~0 and have RMS ~1. Last value of row 0:
        // (4 - 2.5) / sqrt(1.25 + 1e-5) ≈ 1.3416.
        let expected = (4.0_f32 - 2.5) / (1.25_f32 + 1e-5).sqrt();
        assert!((d[3] - expected).abs() < 1e-4, "got {} expected {}", d[3], expected);
        // Both rows should have mean ≈ 0.
        for r in 0..2 {
            let row_sum: f32 = d[r*4..r*4+4].iter().sum();
            assert!(row_sum.abs() < 1e-4, "row {r} sum = {row_sum}");
        }
    }

    /// Verify `add_bias_inplace` broadcasts along the last axis.
    #[test]
    fn add_bias_broadcasts_last_axis() {
        let backend = default_backend();
        let mut x = Tensor::from_vec(vec![0.0; 6], vec![3, 2]);
        let bias = Tensor::from_vec(vec![10.0, 100.0], vec![2]);
        add_bias_inplace(&*backend, &mut x, &bias);
        assert_eq!(x.data(), &[10.0, 100.0, 10.0, 100.0, 10.0, 100.0]);
    }

    #[test]
    fn config_n_patches_and_soft_tokens() {
        let cfg = MmProjConfig {
            image_size: 896, patch_size: 14, embedding_dim: 1152,
            n_layers: 27, n_heads: 16, head_dim: 72, ff_dim: 4304,
            layer_norm_eps: 1e-6, projector: ProjectorKind::Gemma3,
            mean: [0.5; 3], std: [0.5; 3],
        };
        assert_eq!(cfg.n_patches(), 64 * 64);
        // Gemma projector pools 4x4 patch blocks → 256 soft tokens.
        assert_eq!(cfg.n_soft_tokens(), 256);

        let llava = MmProjConfig {
            image_size: 336, patch_size: 14, embedding_dim: 1024,
            n_layers: 24, n_heads: 16, head_dim: 64, ff_dim: 4096,
            layer_norm_eps: 1e-5, projector: ProjectorKind::Mlp,
            mean: [0.481, 0.458, 0.408], std: [0.269, 0.261, 0.276],
        };
        assert_eq!(llava.n_patches(), 24 * 24);
        // Mlp projector emits one soft token per patch.
        assert_eq!(llava.n_soft_tokens(), 576);
    }
}
