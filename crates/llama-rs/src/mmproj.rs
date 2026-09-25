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

/// One transformer block of the SigLIP / CLIP vision tower. Pre-LN
/// architecture: `x + Attn(LN1(x))`; `x + MLP(LN2(x))`. All projections have
/// biases (unlike Llama/Gemma text towers).
#[derive(Debug)]
pub struct VisionBlock {
    pub ln1_w:       Tensor,
    pub ln1_b:       Tensor,
    pub attn_q:      Weight,
    pub attn_q_b:    Tensor,
    pub attn_k:      Weight,
    pub attn_k_b:    Tensor,
    pub attn_v:      Weight,
    pub attn_v_b:    Tensor,
    pub attn_out:    Weight,
    pub attn_out_b:  Tensor,
    pub ln2_w:       Tensor,
    pub ln2_b:       Tensor,
    pub ffn_up:      Weight,
    pub ffn_up_b:    Tensor,
    pub ffn_down:    Weight,
    pub ffn_down_b:  Tensor,
}

/// The Gemma multimodal projector: `avg_pool_4x4(patches) → RMSNorm → linear`.
#[derive(Debug)]
pub struct GemmaProjector {
    /// Per-channel RMSNorm weight applied to the post-pool patches.
    /// Shape `[vision_dim]`.
    pub soft_emb_norm: Tensor,
    /// `[lm_hidden_dim, vision_dim]` linear mapping into the LM token-embedding
    /// space. No bias. Quantized in the GGUF, so we keep it packed.
    pub input_projection: Weight,
}

/// Llava-style 2-layer projector with GeLU between.
#[derive(Debug)]
pub struct MlpProjector {
    pub linear0:   Weight,
    pub linear0_b: Tensor,
    pub linear1:   Weight,
    pub linear1_b: Tensor,
}

#[derive(Debug)]
pub enum Projector {
    Gemma(GemmaProjector),
    Mlp(MlpProjector),
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

/// Vanilla SigLIP/CLIP-style mmproj: pre-LN ViT with biased projections, GeLU
/// MLP, single LayerNorm per sublayer, optional pre-LN, mandatory post-LN.
/// Used by Gemma 3 (with Gemma avg-pool projector) and Llava (with Mlp
/// projector).
#[derive(Debug)]
pub struct SigLipMmProj {
    pub config:        MmProjConfig,
    /// Patch-embedding convolution weight. Stored shape: `[D, 3, P, P]` where
    /// D is `embedding_dim` and P is `patch_size`. Acts as a stride-P conv
    /// from [3, H, W] to [D, H/P, W/P]. We treat it as the matmul
    /// `patches[N, 3*P*P] @ W^T` after unfolding the input.
    pub patch_embd:    Tensor,
    pub patch_embd_b:  Tensor,
    /// Learned position-embedding table. Shape `[n_patches, D]`. Added
    /// element-wise to the patch-embedded sequence before block 0.
    pub position_embd: Tensor,
    /// Optional pre-block layer-norm (CLIP has it, SigLIP does not).
    pub pre_ln:        Option<(Tensor, Tensor)>,
    pub blocks:        Vec<VisionBlock>,
    /// Final post-block layer-norm `(weight, bias)`. Always present.
    pub post_ln:       (Tensor, Tensor),
    pub projector:     Projector,
    pub backend:       Arc<dyn Backend>,
}

impl SigLipMmProj {
    /// Loader for the vanilla SigLIP/CLIP path. Called by
    /// [`MmProj::from_gguf`] after dispatching on `projector_kind`. Loads +
    /// shape-validates every tensor against the parsed config — any mismatch
    /// surfaces as `BadTensorShape` rather than a downstream NaN.
    pub fn from_gguf_with_config(
        g: &GgufFile,
        config: MmProjConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        debug_assert!(config.projector.supports_siglip_loader());
        let idx = TensorIndex::new(g);

        // ----- patch embedding (conv as a flattened matmul) ---------------
        // The HF→GGUF converter writes the conv weight in NCHW order
        // `[D, 3, P, P]`. We load it as F32 since the shape isn't a 2D matrix
        // — backend.linear can't consume it directly. The forward pass will
        // reshape it to `[D, 3*P*P]` before matmul.
        let patch_embd   = idx.take("v.patch_embd.weight", &[])?;
        let patch_embd_b = idx.take("v.patch_embd.bias",   &[])?;
        let position_embd = idx.take("v.position_embd.weight", &[])?;

        // ----- optional pre-LN (CLIP family only) -------------------------
        let pre_ln = match (idx.try_take("v.pre_ln.weight"), idx.try_take("v.pre_ln.bias")) {
            (Some(w), Some(b)) => Some((w?, b?)),
            _ => None,
        };

        // ----- transformer blocks -----------------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(VisionBlock {
                ln1_w:      idx.take(&format!("v.blk.{i}.ln1.weight"),     &[])?,
                ln1_b:      idx.take(&format!("v.blk.{i}.ln1.bias"),       &[])?,
                attn_q:     idx.take_weight(&format!("v.blk.{i}.attn_q.weight"),   &[])?,
                attn_q_b:   idx.take(&format!("v.blk.{i}.attn_q.bias"),    &[])?,
                attn_k:     idx.take_weight(&format!("v.blk.{i}.attn_k.weight"),   &[])?,
                attn_k_b:   idx.take(&format!("v.blk.{i}.attn_k.bias"),    &[])?,
                attn_v:     idx.take_weight(&format!("v.blk.{i}.attn_v.weight"),   &[])?,
                attn_v_b:   idx.take(&format!("v.blk.{i}.attn_v.bias"),    &[])?,
                attn_out:   idx.take_weight(&format!("v.blk.{i}.attn_out.weight"), &[])?,
                attn_out_b: idx.take(&format!("v.blk.{i}.attn_out.bias"),  &[])?,
                ln2_w:      idx.take(&format!("v.blk.{i}.ln2.weight"),     &[])?,
                ln2_b:      idx.take(&format!("v.blk.{i}.ln2.bias"),       &[])?,
                ffn_up:     idx.take_weight(&format!("v.blk.{i}.ffn_up.weight"),   &[])?,
                ffn_up_b:   idx.take(&format!("v.blk.{i}.ffn_up.bias"),    &[])?,
                ffn_down:   idx.take_weight(&format!("v.blk.{i}.ffn_down.weight"), &[])?,
                ffn_down_b: idx.take(&format!("v.blk.{i}.ffn_down.bias"),  &[])?,
            });
        }

        // ----- post-LN (always present) -----------------------------------
        let post_ln = (
            idx.take("v.post_ln.weight", &[])?,
            idx.take("v.post_ln.bias",   &[])?,
        );

        // ----- projector head ---------------------------------------------
        // Note on `mm.input_projection.weight`: the Gemma converter writes
        // this matrix in HuggingFace's `nn.Parameter([vision_dim, lm_hidden])`
        // form (used as `out = in @ W` in the reference implementation),
        // *not* in `nn.Linear`'s `[out, in]` form. Our `linear(x, w) = x @ w^T`
        // expects `[out, in]`, so we transpose at load time. F16 dequantises
        // to F32 before the transpose; the matrix is small (1152×2560 ≈ 12MB)
        // so the host round-trip is cheap.
        let projector = match config.projector {
            ProjectorKind::Gemma3 => {
                let raw = idx.take("mm.input_projection.weight", &[])?;
                Projector::Gemma(GemmaProjector {
                    soft_emb_norm:    idx.take("mm.soft_emb_norm.weight", &[])?,
                    input_projection: Weight::Dense(transpose_2d_host(raw)?),
                })
            }
            ProjectorKind::Mlp => Projector::Mlp(MlpProjector {
                linear0:   idx.take_weight("mm.0.weight", &[])?,
                linear0_b: idx.take("mm.0.bias", &[])?,
                linear1:   idx.take_weight("mm.2.weight", &[])?,
                linear1_b: idx.take("mm.2.bias", &[])?,
            }),
            // Gated by `supports_siglip_loader()` above — unreachable.
            ProjectorKind::Gemma4V => unreachable!("Gemma4V loader not yet implemented"),
            ProjectorKind::Qwen3Vl => unreachable!("Qwen3Vl handled by separate loader (returns Err in MmProj::from_gguf)"),
        };

        verify_shapes(&config, &patch_embd, &patch_embd_b, &position_embd,
                      pre_ln.as_ref(), &blocks, &post_ln, &projector)?;

        // Move everything onto the backend's preferred storage. Norms / biases
        // stay dense; projection matrices use `Weight::try_to_device` which
        // keeps packed quants packed AND falls back to host-resident if the
        // remaining VRAM budget would be breached. The vision tower is one-shot
        // per image, so a host-resident weight just means a slightly slower
        // single-image preprocess — not per-token cost.
        const M: usize = 1 * 1024 * 1024 * 1024;       // 1 GB margin (vision is one-shot)
        let upload = |t: Tensor| backend.to_device(t);
        let blocks: Vec<VisionBlock> = blocks.into_iter().map(|b| VisionBlock {
            ln1_w:      upload(b.ln1_w),
            ln1_b:      upload(b.ln1_b),
            attn_q:     b.attn_q.try_to_device(&*backend, M),
            attn_q_b:   upload(b.attn_q_b),
            attn_k:     b.attn_k.try_to_device(&*backend, M),
            attn_k_b:   upload(b.attn_k_b),
            attn_v:     b.attn_v.try_to_device(&*backend, M),
            attn_v_b:   upload(b.attn_v_b),
            attn_out:   b.attn_out.try_to_device(&*backend, M),
            attn_out_b: upload(b.attn_out_b),
            ln2_w:      upload(b.ln2_w),
            ln2_b:      upload(b.ln2_b),
            ffn_up:     b.ffn_up.try_to_device(&*backend, M),
            ffn_up_b:   upload(b.ffn_up_b),
            ffn_down:   b.ffn_down.try_to_device(&*backend, M),
            ffn_down_b: upload(b.ffn_down_b),
        }).collect();
        let projector = match projector {
            Projector::Gemma(p) => Projector::Gemma(GemmaProjector {
                soft_emb_norm:    upload(p.soft_emb_norm),
                input_projection: p.input_projection.try_to_device(&*backend, M),
            }),
            Projector::Mlp(p) => Projector::Mlp(MlpProjector {
                linear0:   p.linear0.try_to_device(&*backend, M),
                linear0_b: upload(p.linear0_b),
                linear1:   p.linear1.try_to_device(&*backend, M),
                linear1_b: upload(p.linear1_b),
            }),
        };
        let pre_ln = pre_ln.map(|(w, b)| (upload(w), upload(b)));
        let post_ln = (upload(post_ln.0), upload(post_ln.1));
        // `patch_embd` keeps its NCHW shape on host until forward time, since
        // the upcoming unfold step happens CPU-side anyway.

        Ok(Self {
            config, patch_embd, patch_embd_b, position_embd,
            pre_ln, blocks, post_ln, projector, backend,
        })
    }
}

fn verify_shapes(
    cfg:           &MmProjConfig,
    patch_embd:    &Tensor,
    patch_embd_b:  &Tensor,
    position_embd: &Tensor,
    pre_ln:        Option<&(Tensor, Tensor)>,
    blocks:        &[VisionBlock],
    post_ln:       &(Tensor, Tensor),
    projector:     &Projector,
) -> Result<()> {
    let bad = |name: &str, got: &[usize], expected: &[usize]| -> LlamaError {
        LlamaError::BadTensorShape {
            name: name.into(),
            got: got.iter().map(|&v| v as u64).collect(),
            expected: expected.iter().map(|&v| v as u64).collect(),
        }
    };
    let check_dense = |t: &Tensor, expected: &[usize], name: &str| -> Result<()> {
        if t.shape() != expected { Err(bad(name, t.shape(), expected)) } else { Ok(()) }
    };
    let check_w = |w: &Weight, expected: &[usize], name: &str| -> Result<()> {
        if w.shape() != expected { Err(bad(name, w.shape(), expected)) } else { Ok(()) }
    };

    let d = cfg.embedding_dim;
    let p = cfg.patch_size;
    let np = cfg.n_patches();
    let ff = cfg.ff_dim;

    // Patch conv: [D, 3, P, P]. Bias: [D]. Position embed: [N, D].
    check_dense(patch_embd,    &[d, 3, p, p], "v.patch_embd")?;
    check_dense(patch_embd_b,  &[d],          "v.patch_embd.bias")?;
    check_dense(position_embd, &[np, d],      "v.position_embd")?;

    if let Some((w, b)) = pre_ln {
        check_dense(w, &[d], "v.pre_ln.weight")?;
        check_dense(b, &[d], "v.pre_ln.bias")?;
    }
    check_dense(&post_ln.0, &[d], "v.post_ln.weight")?;
    check_dense(&post_ln.1, &[d], "v.post_ln.bias")?;

    for (i, blk) in blocks.iter().enumerate() {
        check_dense(&blk.ln1_w,     &[d],    &format!("v.blk.{i}.ln1.weight"))?;
        check_dense(&blk.ln1_b,     &[d],    &format!("v.blk.{i}.ln1.bias"))?;
        check_w    (&blk.attn_q,    &[d, d], &format!("v.blk.{i}.attn_q.weight"))?;
        check_dense(&blk.attn_q_b,  &[d],    &format!("v.blk.{i}.attn_q.bias"))?;
        check_w    (&blk.attn_k,    &[d, d], &format!("v.blk.{i}.attn_k.weight"))?;
        check_dense(&blk.attn_k_b,  &[d],    &format!("v.blk.{i}.attn_k.bias"))?;
        check_w    (&blk.attn_v,    &[d, d], &format!("v.blk.{i}.attn_v.weight"))?;
        check_dense(&blk.attn_v_b,  &[d],    &format!("v.blk.{i}.attn_v.bias"))?;
        check_w    (&blk.attn_out,  &[d, d], &format!("v.blk.{i}.attn_out.weight"))?;
        check_dense(&blk.attn_out_b,&[d],    &format!("v.blk.{i}.attn_out.bias"))?;
        check_dense(&blk.ln2_w,     &[d],    &format!("v.blk.{i}.ln2.weight"))?;
        check_dense(&blk.ln2_b,     &[d],    &format!("v.blk.{i}.ln2.bias"))?;
        check_w    (&blk.ffn_up,    &[ff, d],&format!("v.blk.{i}.ffn_up.weight"))?;
        check_dense(&blk.ffn_up_b,  &[ff],   &format!("v.blk.{i}.ffn_up.bias"))?;
        check_w    (&blk.ffn_down,  &[d, ff],&format!("v.blk.{i}.ffn_down.weight"))?;
        check_dense(&blk.ffn_down_b,&[d],    &format!("v.blk.{i}.ffn_down.bias"))?;
    }

    match projector {
        Projector::Gemma(p) => {
            check_dense(&p.soft_emb_norm, &[d], "mm.soft_emb_norm.weight")?;
            // [lm_hidden, d] — we don't know lm_hidden here without the LM
            // config, so just sanity-check the inner dim matches.
            let s = p.input_projection.shape();
            if s.len() != 2 || s[1] != d {
                return Err(bad("mm.input_projection.weight", s, &[0, d]));
            }
        }
        Projector::Mlp(p) => {
            // First layer: [hidden, vision_dim], second: [hidden, hidden].
            // Hidden dim is implicit — just check inner dim consistency.
            let s0 = p.linear0.shape();
            if s0.len() != 2 || s0[1] != d {
                return Err(bad("mm.0.weight", s0, &[0, d]));
            }
            let s1 = p.linear1.shape();
            if s1.len() != 2 || s1[0] != s0[0] || s1[1] != s0[0] {
                return Err(bad("mm.2.weight", s1, &[s0[0], s0[0]]));
            }
            check_dense(&p.linear0_b, &[s0[0]], "mm.0.bias")?;
            check_dense(&p.linear1_b, &[s1[0]], "mm.2.bias")?;
        }
    }
    Ok(())
}

impl SigLipMmProj {
    /// Run image preprocessing → ViT → projector. Input is a host tensor
    /// `[3, H, W]` produced by [`crate::preprocess_image`]; output is the
    /// LM-space soft-token tensor `[n_soft_tokens, lm_hidden_dim]` ready to be
    /// spliced into the text token embedding stream.
    ///
    /// Cost: one-shot per image (not per token), so the LayerNorm /
    /// bias-add / patch-unfold paths run on host even when the backend is
    /// CUDA — same pattern as `rmsnorm_no_scale` in its initial port. GPU
    /// fast paths can be added later if vision becomes a hotspot.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        if image.rank() != 3 || image.dim(0) != 3
            || image.dim(1) != cfg.image_size || image.dim(2) != cfg.image_size
        {
            return Err(LlamaError::Config(format!(
                "image shape {:?} doesn't match expected [3, {sz}, {sz}]",
                image.shape(), sz = cfg.image_size
            )));
        }
        let backend = &*self.backend;

        // ----- 1. Patch-conv as flattened matmul -------------------------
        // Unfold the image into [n_patches, 3*P*P] so the conv becomes a
        // standard linear. The conv weight is stored as [D, 3, P, P] in
        // GGUF; reshape to [D, 3*P*P] and then `backend.linear` does the
        // job (linear computes y = x · W^T, so [N, 3PP] @ [D, 3PP]^T = [N, D]).
        let patches = unfold_patches_to_host(image, cfg.patch_size)?;
        let patches = backend.to_device(patches);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_w_host = self.patch_embd.to_host();
        let patch_w_flat = patch_w_host.reshape(vec![cfg.embedding_dim, in_dim])
            .map_err(|e| LlamaError::Config(format!("patch_embd reshape: {e:?}")))?;
        let patch_w_flat = backend.to_device(patch_w_flat);
        let mut x = backend.linear(&patches, &patch_w_flat);
        add_bias_inplace(backend, &mut x, &self.patch_embd_b);

        // ----- 2. Position embedding (additive) --------------------------
        backend.add_inplace(&mut x, &self.position_embd);

        // ----- 3. Optional pre-LN (CLIP only) ----------------------------
        if let Some((w, b)) = &self.pre_ln {
            x = layer_norm(backend, &x, w, b, cfg.layer_norm_eps);
        }

        // ----- 4. Transformer blocks -------------------------------------
        for blk in &self.blocks {
            x = vision_block_forward(backend, &x, blk, cfg);
        }

        // ----- 5. Post-LN ------------------------------------------------
        x = layer_norm(backend, &x, &self.post_ln.0, &self.post_ln.1, cfg.layer_norm_eps);

        // ----- 6. Projector head -----------------------------------------
        let out = match &self.projector {
            Projector::Gemma(p) => gemma_project(backend, &x, p, cfg),
            Projector::Mlp(p)   => mlp_project(backend, &x, p),
        };
        Ok(out)
    }
}

// ============================================================================
// Gemma 4 V mmproj — bespoke vision tower
// ============================================================================
//
// Gemma 4 (E2B / E4B) ships a vision tower that's structurally a Gemma-style
// transformer applied to image patches, not a vanilla SigLIP. Differences
// from SigLIP:
//   * RMSNorm everywhere (no LayerNorm bias terms).
//   * Sandwich norms — both pre and post-norm around attention and FFN
//     (mirrors Gemma 3's text tower's `attn_post_norm` + `ffn_post_norm`).
//   * Per-head Q/K-norm (RMSNorm of size head_dim) before attention scaling.
//   * SwiGLU FFN — `ffn_gate` + `ffn_up` + `ffn_down` (vs vanilla GeLU MLP).
//   * No biases on any projection.
//   * No global post-LN — the final norm is the per-block ffn_post_norm.
//   * Smaller dimensions: SigLIP-base sized (768 dim, 16 blocks, 12 heads).
//   * Multi-tile position embedding (~20480 entries) so the same tower can
//     handle pan-and-scan crops at varying tile counts. For a single
//     224×224 image, we slice the first `n_patches` (=196) positions.
// And the projector:
//   * Just a single `mm.input_projection.weight` linear (no soft_emb_norm,
//     no avg-pool). 1:1 patches → soft tokens.

/// One pre-LN+post-LN sandwich block of Gemma 4's vision tower. RMSNorm on
/// both sides of each sublayer, no biases anywhere, SwiGLU FFN.
#[derive(Debug)]
pub struct Gemma4VBlock {
    pub ln1:        Tensor,        // pre-attention RMSNorm scale
    pub attn_q:     Weight,
    pub attn_q_norm: Tensor,       // per-head RMSNorm scale (size head_dim)
    pub attn_k:     Weight,
    pub attn_k_norm: Tensor,
    pub attn_v:     Weight,
    pub attn_out:   Weight,
    pub attn_post:  Tensor,        // post-attention RMSNorm scale
    pub ln2:        Tensor,        // pre-FFN RMSNorm scale
    pub ffn_gate:   Weight,
    pub ffn_up:     Weight,
    pub ffn_down:   Weight,
    pub ffn_post:   Tensor,        // post-FFN RMSNorm scale
}

#[derive(Debug)]
pub struct Gemma4VProjector {
    /// `[lm_hidden, vision_dim]` after our load-time transpose. Same convention
    /// as the Gemma 3 projector.
    pub input_projection: Weight,
}

#[derive(Debug)]
pub struct Gemma4VMmProj {
    pub config:        MmProjConfig,
    pub patch_embd:    Tensor,            // [D, 3, P, P]
    pub position_embd: Tensor,            // [N_max, D] — slice first n_patches
    pub blocks:        Vec<Gemma4VBlock>,
    pub projector:     Gemma4VProjector,
    pub backend:       Arc<dyn Backend>,
}

impl Gemma4VMmProj {
    pub fn from_gguf_with_config(
        g: &GgufFile,
        config: MmProjConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        debug_assert_eq!(config.projector, ProjectorKind::Gemma4V);
        let idx = TensorIndex::new(g);

        // ----- patch + position embed ------------------------------------
        let patch_embd    = idx.take("v.patch_embd.weight",    &[])?;
        let position_embd = idx.take("v.position_embd.weight", &[])?;

        // ----- transformer blocks ---------------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(Gemma4VBlock {
                ln1:         idx.take(&format!("v.blk.{i}.ln1.weight"),            &[])?,
                attn_q:      idx.take_weight(&format!("v.blk.{i}.attn_q.weight"),  &[])?,
                attn_q_norm: idx.take(&format!("v.blk.{i}.attn_q_norm.weight"),    &[])?,
                attn_k:      idx.take_weight(&format!("v.blk.{i}.attn_k.weight"),  &[])?,
                attn_k_norm: idx.take(&format!("v.blk.{i}.attn_k_norm.weight"),    &[])?,
                attn_v:      idx.take_weight(&format!("v.blk.{i}.attn_v.weight"),  &[])?,
                attn_out:    idx.take_weight(&format!("v.blk.{i}.attn_out.weight"),&[])?,
                attn_post:   idx.take(&format!("v.blk.{i}.attn_post_norm.weight"), &[])?,
                ln2:         idx.take(&format!("v.blk.{i}.ln2.weight"),            &[])?,
                ffn_gate:    idx.take_weight(&format!("v.blk.{i}.ffn_gate.weight"),&[])?,
                ffn_up:      idx.take_weight(&format!("v.blk.{i}.ffn_up.weight"),  &[])?,
                ffn_down:    idx.take_weight(&format!("v.blk.{i}.ffn_down.weight"),&[])?,
                ffn_post:    idx.take(&format!("v.blk.{i}.ffn_post_norm.weight"),  &[])?,
            });
        }

        // ----- projector (single linear, NO transpose) ------------------
        // Unlike Gemma 3 (which stores `mm.input_projection` as
        // `[vision_dim, lm_hidden]` and needs transposing), Gemma 4 V already
        // stores the projector in `[lm_hidden, vision_dim]` order — exactly
        // what `linear(x, w) = x @ w^T` expects. Use as-is.
        let projector = Gemma4VProjector {
            input_projection: idx.take_weight("mm.input_projection.weight", &[])?,
        };

        verify_gemma4v_shapes(&config, &patch_embd, &position_embd, &blocks, &projector)?;

        // ----- upload to backend ----------------------------------------
        // Vision is one-shot per image; 1 GB safety margin is plenty.
        const M: usize = 1 * 1024 * 1024 * 1024;
        let upload = |t: Tensor| backend.to_device(t);
        let blocks: Vec<Gemma4VBlock> = blocks.into_iter().map(|b| Gemma4VBlock {
            ln1:         upload(b.ln1),
            attn_q:      b.attn_q.try_to_device(&*backend, M),
            attn_q_norm: upload(b.attn_q_norm),
            attn_k:      b.attn_k.try_to_device(&*backend, M),
            attn_k_norm: upload(b.attn_k_norm),
            attn_v:      b.attn_v.try_to_device(&*backend, M),
            attn_out:    b.attn_out.try_to_device(&*backend, M),
            attn_post:   upload(b.attn_post),
            ln2:         upload(b.ln2),
            ffn_gate:    b.ffn_gate.try_to_device(&*backend, M),
            ffn_up:      b.ffn_up.try_to_device(&*backend, M),
            ffn_down:    b.ffn_down.try_to_device(&*backend, M),
            ffn_post:    upload(b.ffn_post),
        }).collect();
        let projector = Gemma4VProjector {
            input_projection: projector.input_projection.try_to_device(&*backend, M),
        };
        let position_embd = upload(position_embd);
        // patch_embd kept on host until forward time (unfold runs CPU-side).

        Ok(Self { config, patch_embd, position_embd, blocks, projector, backend })
    }

    /// Run image preprocessing → Gemma 4 V tower → projector. Same input/output
    /// contract as [`SigLipMmProj::forward`]: `[3, H, W]` → `[n_soft, lm_hidden]`.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let backend = &*self.backend;
        if image.rank() != 3 || image.dim(0) != 3
            || image.dim(1) != cfg.image_size || image.dim(2) != cfg.image_size
        {
            return Err(LlamaError::Config(format!(
                "image shape {:?} doesn't match expected [3, {sz}, {sz}]",
                image.shape(), sz = cfg.image_size
            )));
        }

        // ----- 1. Patch-conv as flattened matmul ------------------------
        let patches = unfold_patches_to_host(image, cfg.patch_size)?;
        let patches = backend.to_device(patches);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_w_host = self.patch_embd.to_host();
        let patch_w_flat = patch_w_host.reshape(vec![cfg.embedding_dim, in_dim])
            .map_err(|e| LlamaError::Config(format!("patch_embd reshape: {e:?}")))?;
        let patch_w_flat = backend.to_device(patch_w_flat);
        let mut x = backend.linear(&patches, &patch_w_flat);
        // No patch_embd_b — Gemma 4 V conv has no bias.

        // ----- 2. Position embed (axial: row + col) ----------------------
        // The table is `[2, max_pos, D]`. For patch (py, px) at flat index
        // `py*side + px`, we add `pos[0][py] + pos[1][px]`. Build the per-
        // patch slice on host then add as one tensor.
        let pe_host = self.position_embd.to_host();
        let pe_shape = pe_host.shape();
        let max_pos = pe_shape[1];
        let d = cfg.embedding_dim;
        let side = cfg.image_size / cfg.patch_size;
        let np = side * side;
        let pe_data = pe_host.data();
        let row_off = 0;                  // pos[0] starts at index 0
        let col_off = max_pos * d;        // pos[1] starts at max_pos*d
        let mut pos_block = vec![0.0f32; np * d];
        for py in 0..side {
            let row_base = row_off + py * d;
            for px in 0..side {
                let col_base = col_off + px * d;
                let dst = (py * side + px) * d;
                for j in 0..d {
                    pos_block[dst + j] = pe_data[row_base + j] + pe_data[col_base + j];
                }
            }
        }
        let pos_block = backend.to_device(Tensor::from_vec(pos_block, vec![np, d]));
        backend.add_inplace(&mut x, &pos_block);

        // ----- 3. Transformer blocks ------------------------------------
        for blk in &self.blocks {
            x = gemma4v_block_forward(backend, &x, blk, cfg);
        }

        // ----- 4. No global post-LN (per-block sandwich handles it) ------

        // ----- 5. Projector: single linear -----------------------------
        let out = self.projector.input_projection.linear(backend, &x);
        Ok(out)
    }
}

/// One Gemma-style pre-LN+post-LN sandwich block applied to image patches.
///   `x' = x + attn_post(Attn(ln1(x)))`
///   `out = x' + ffn_post(MLP(ln2(x')))`
/// Bidirectional attention via `past = seq` trick (no causal mask).
fn gemma4v_block_forward(
    b:   &dyn Backend,
    x:   &Tensor,
    blk: &Gemma4VBlock,
    cfg: &MmProjConfig,
) -> Tensor {
    let n_seq = x.dim(0);
    let n_h   = cfg.n_heads;
    let hd    = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let eps   = cfg.layer_norm_eps;

    // ----- Pre-attn RMSNorm + Q/K/V proj (no bias) -------------------
    let xn = ggml_rs::ops::rmsnorm(b, x, &blk.ln1, eps);
    let q  = blk.attn_q.linear(b, &xn);
    let k  = blk.attn_k.linear(b, &xn);
    let v  = blk.attn_v.linear(b, &xn);

    let q_3d = q.reshape(vec![n_seq, n_h, hd]).expect("Q reshape");
    let k_3d = k.reshape(vec![n_seq, n_h, hd]).expect("K reshape");
    let v_3d = v.reshape(vec![n_seq, n_h, hd]).expect("V reshape");

    // Per-head Q/K RMSNorm (Gemma 3 does this in the text tower too).
    let q_3d = ggml_rs::ops::rmsnorm(b, &q_3d, &blk.attn_q_norm, eps);
    let k_3d = ggml_rs::ops::rmsnorm(b, &k_3d, &blk.attn_k_norm, eps);

    // Bidirectional attention via past = n_seq (mask never fires).
    let attn = b.attention(&q_3d, &k_3d, &v_3d, n_seq, scale, n_seq, None);
    let attn = attn.reshape(vec![n_seq, n_h * hd]).expect("attn reshape");

    let mut out = blk.attn_out.linear(b, &attn);
    // Post-attn RMSNorm (Gemma sandwich), then residual.
    out = ggml_rs::ops::rmsnorm(b, &out, &blk.attn_post, eps);
    b.add_inplace(&mut out, x);

    // ----- Pre-FFN RMSNorm + SwiGLU + post-FFN norm + residual --------
    let xn2 = ggml_rs::ops::rmsnorm(b, &out, &blk.ln2, eps);
    let gate = blk.ffn_gate.linear(b, &xn2);
    let up   = blk.ffn_up.linear(b, &xn2);
    // Gemma uses approx-GeLU on the gate, then multiplies by up — fused.
    let gated = b.gelu_approx_mul(&gate, &up);
    let mut down = blk.ffn_down.linear(b, &gated);
    down = ggml_rs::ops::rmsnorm(b, &down, &blk.ffn_post, eps);
    b.add_inplace(&mut down, &out);
    down
}

fn verify_gemma4v_shapes(
    cfg:           &MmProjConfig,
    patch_embd:    &Tensor,
    position_embd: &Tensor,
    blocks:        &[Gemma4VBlock],
    projector:     &Gemma4VProjector,
) -> Result<()> {
    let bad = |name: &str, got: &[usize], expected: &[usize]| -> LlamaError {
        LlamaError::BadTensorShape {
            name: name.into(),
            got: got.iter().map(|&v| v as u64).collect(),
            expected: expected.iter().map(|&v| v as u64).collect(),
        }
    };
    let check_dense = |t: &Tensor, expected: &[usize], name: &str| -> Result<()> {
        if t.shape() != expected { Err(bad(name, t.shape(), expected)) } else { Ok(()) }
    };
    let check_w = |w: &Weight, expected: &[usize], name: &str| -> Result<()> {
        if w.shape() != expected { Err(bad(name, w.shape(), expected)) } else { Ok(()) }
    };

    let d  = cfg.embedding_dim;
    let p  = cfg.patch_size;
    let ff = cfg.ff_dim;
    let hd = cfg.head_dim;

    check_dense(patch_embd, &[d, 3, p, p], "v.patch_embd")?;

    // Position embed is `[2, max_positions, D]` — two axial embedding tables
    // (row + column), each indexed by a per-axis position. For a single
    // 224×224 image with patch 16 we use indices 0..14 from each axis. The
    // outer "2" is the row/col split; we just sanity-check the inner dim.
    let pe_shape = position_embd.shape();
    if pe_shape.len() != 3 || pe_shape[0] != 2 || pe_shape[2] != d {
        return Err(bad("v.position_embd", pe_shape, &[2, 0, d]));
    }
    let side = cfg.image_size / cfg.patch_size;
    if pe_shape[1] < side {
        return Err(LlamaError::Config(format!(
            "v.position_embd axial table has only {} positions per axis, need at least {} for {}×{} input",
            pe_shape[1], side, cfg.image_size, cfg.image_size
        )));
    }

    for (i, blk) in blocks.iter().enumerate() {
        check_dense(&blk.ln1,         &[d],     &format!("v.blk.{i}.ln1"))?;
        check_w    (&blk.attn_q,      &[d, d],  &format!("v.blk.{i}.attn_q"))?;
        check_dense(&blk.attn_q_norm, &[hd],    &format!("v.blk.{i}.attn_q_norm"))?;
        check_w    (&blk.attn_k,      &[d, d],  &format!("v.blk.{i}.attn_k"))?;
        check_dense(&blk.attn_k_norm, &[hd],    &format!("v.blk.{i}.attn_k_norm"))?;
        check_w    (&blk.attn_v,      &[d, d],  &format!("v.blk.{i}.attn_v"))?;
        check_w    (&blk.attn_out,    &[d, d],  &format!("v.blk.{i}.attn_out"))?;
        check_dense(&blk.attn_post,   &[d],     &format!("v.blk.{i}.attn_post_norm"))?;
        check_dense(&blk.ln2,         &[d],     &format!("v.blk.{i}.ln2"))?;
        check_w    (&blk.ffn_gate,    &[ff, d], &format!("v.blk.{i}.ffn_gate"))?;
        check_w    (&blk.ffn_up,      &[ff, d], &format!("v.blk.{i}.ffn_up"))?;
        check_w    (&blk.ffn_down,    &[d, ff], &format!("v.blk.{i}.ffn_down"))?;
        check_dense(&blk.ffn_post,    &[d],     &format!("v.blk.{i}.ffn_post_norm"))?;
    }

    // Projector: [lm_hidden, vision_dim] after our transpose. Inner dim = d.
    let s = projector.input_projection.shape();
    if s.len() != 2 || s[1] != d {
        return Err(bad("mm.input_projection.weight", s, &[0, d]));
    }
    Ok(())
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

/// One pre-LN ViT block. Standard pattern:
///   `x' = x + Attn(LN(x))`
///   `out = x' + MLP(LN(x'))`
/// Vision attention is **bidirectional** (no causal mask) — we get that out
/// of the existing causal `attention()` op by setting `past = seq`, which
/// makes the mask predicate `t > past + s` false for every (t, s) in the
/// valid range. Avoids needing a separate non-causal kernel until vision
/// becomes hot enough to warrant one.
fn vision_block_forward(
    b:   &dyn Backend,
    x:   &Tensor,
    blk: &VisionBlock,
    cfg: &MmProjConfig,
) -> Tensor {
    let n_seq = x.dim(0);
    let n_h   = cfg.n_heads;
    let hd    = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    // ----- Pre-attn LN + Q/K/V projections + biases -------------------
    let h0 = layer_norm(b, x, &blk.ln1_w, &blk.ln1_b, cfg.layer_norm_eps);
    let mut q = blk.attn_q.linear(b, &h0); add_bias_inplace(b, &mut q, &blk.attn_q_b);
    let mut k = blk.attn_k.linear(b, &h0); add_bias_inplace(b, &mut k, &blk.attn_k_b);
    let mut v = blk.attn_v.linear(b, &h0); add_bias_inplace(b, &mut v, &blk.attn_v_b);

    let q3 = q.reshape(vec![n_seq, n_h, hd]).expect("Q reshape");
    let k3 = k.reshape(vec![n_seq, n_h, hd]).expect("K reshape");
    let v3 = v.reshape(vec![n_seq, n_h, hd]).expect("V reshape");

    // No RoPE — vision uses learned absolute positions (already added at the
    // start of forward()). past = n_seq disables the causal mask.
    let attn = b.attention(&q3, &k3, &v3, n_seq, scale, n_seq, None);
    let attn = attn.reshape(vec![n_seq, n_h * hd]).expect("attn reshape");

    let mut out = blk.attn_out.linear(b, &attn);
    add_bias_inplace(b, &mut out, &blk.attn_out_b);

    // Residual: out += x
    b.add_inplace(&mut out, x);

    // ----- Pre-FFN LN + MLP + bias + residual -------------------------
    let h1 = layer_norm(b, &out, &blk.ln2_w, &blk.ln2_b, cfg.layer_norm_eps);
    let mut up = blk.ffn_up.linear(b, &h1); add_bias_inplace(b, &mut up, &blk.ffn_up_b);
    let activated = b.gelu_approx(&up);
    let mut down = blk.ffn_down.linear(b, &activated);
    add_bias_inplace(b, &mut down, &blk.ffn_down_b);

    b.add_inplace(&mut down, &out);
    down
}

/// Gemma 3 / 4 multimodal projector: 4×4 spatial avg-pool over the 64×64
/// patch grid (4096 → 256 soft tokens), then RMSNorm with `soft_emb_norm`,
/// then linear to LM hidden dim.
fn gemma_project(
    b:   &dyn Backend,
    x:   &Tensor,
    p:   &GemmaProjector,
    cfg: &MmProjConfig,
) -> Tensor {
    let np = cfg.n_patches();
    let d  = cfg.embedding_dim;
    // Side length of the patch grid (e.g. 64 for SigLIP-Gemma).
    let side = (np as f32).sqrt() as usize;
    debug_assert_eq!(side * side, np, "n_patches {np} isn't a perfect square");
    let pool = 4;
    let new_side = side / pool;          // 16
    let n_pooled = new_side * new_side;  // 256

    // Avg-pool on host: vision is one-shot, and the output (256 × D) is small
    // enough that the d2h round-trip is negligible.
    let host = x.to_host();
    let data = host.data();
    let mut pooled = vec![0.0f32; n_pooled * d];
    let inv = 1.0 / ((pool * pool) as f32);
    for big_y in 0..new_side {
        for big_x in 0..new_side {
            let out_off = (big_y * new_side + big_x) * d;
            for dy in 0..pool {
                for dx in 0..pool {
                    let ny = big_y * pool + dy;
                    let nx = big_x * pool + dx;
                    let in_off = (ny * side + nx) * d;
                    for j in 0..d {
                        pooled[out_off + j] += data[in_off + j];
                    }
                }
            }
            for j in 0..d {
                pooled[out_off + j] *= inv;
            }
        }
    }
    let pooled = b.to_device(Tensor::from_vec(pooled, vec![n_pooled, d]));

    // RMSNorm with the per-channel `soft_emb_norm` weight. Gemma uses the
    // same eps here as the LM's RMSNorm (1e-6 in practice).
    let normed = b.rmsnorm(&pooled, &p.soft_emb_norm, 1e-6);

    // Project into LM hidden dim. Output: [n_pooled, lm_hidden].
    p.input_projection.linear(b, &normed)
}

/// Llava-style MLP projector: linear → GeLU → linear, both with bias.
fn mlp_project(b: &dyn Backend, x: &Tensor, p: &MlpProjector) -> Tensor {
    let mut h = p.linear0.linear(b, x);
    add_bias_inplace(b, &mut h, &p.linear0_b);
    let activated = b.gelu_approx(&h);
    let mut out = p.linear1.linear(b, &activated);
    add_bias_inplace(b, &mut out, &p.linear1_b);
    out
}

// ============================================================================
// Qwen3-VL mmproj — fused-QKV ViT + 2x2 spatial-merge MLP projector
// ============================================================================
//
// Qwen3-VL ships its vision tower as a "merger" projector. Differences from
// SigLIP:
//   * Fused `attn_qkv.weight` (with bias) instead of separate q/k/v.
//   * Pre-LN with full LayerNorm (mean + var, with bias) — same as SigLIP/CLIP.
//   * GeLU MLP with biases on `ffn_up` and `ffn_down`. NO `ffn_gate` (Qwen3-VL
//     uses a vanilla 2-layer MLP, not SwiGLU).
//   * Post-LN at the end of the tower.
//   * Temporal patch embed: TWO `v.patch_embd.weight` tensors named `.weight`
//     and `.weight.1`. For static images (single time step) these are summed at
//     load time — equivalent to feeding the same image at both temporal slots.
//   * 2x2 spatial merge collapses 4 neighbouring patches into one 4*D row, then
//     a 2-layer MLP (`mm.0` → GELU → `mm.2`) projects into LM hidden dim.
// And optional DeepStack (when `clip.vision.is_deepstack_layers` flags any
// block AND per-block `deepstack_norm`/`fc1`/`fc2` tensors are present): each
// flagged block's output is also extracted, normed, projected, and concatenated
// as additional context tokens. The current Q3.6-27B mmproj declares the bool
// array but ships no such per-block tensors → DeepStack is inactive for now;
// loader detects this and emits warning rather than failing.

/// One pre-LN ViT block as used by the Qwen3-VL merger. Fused QKV reduces
/// three matmuls into one, lowering memory bandwidth at decode time.
#[derive(Debug)]
pub struct Qwen3VlBlock {
    pub ln1_w:        Tensor,
    pub ln1_b:        Tensor,
    pub attn_qkv:     Weight,    // [3*D, D] — Q | K | V stacked along output axis
    pub attn_qkv_b:   Tensor,    // [3*D]
    pub attn_out:     Weight,    // [D, D]
    pub attn_out_b:   Tensor,    // [D]
    pub ln2_w:        Tensor,
    pub ln2_b:        Tensor,
    pub ffn_up:       Weight,    // [ff, D]
    pub ffn_up_b:     Tensor,    // [ff]
    pub ffn_down:     Weight,    // [D, ff]
    pub ffn_down_b:   Tensor,    // [D]
}

/// 2-layer MLP merger head. The 2x2 spatial reshape happens at forward time;
/// these are just the two linear layers + their biases.
#[derive(Debug)]
pub struct Qwen3VlProjector {
    /// `[merge_dim, merge_dim]` where `merge_dim = 4 * vision_dim` (the
    /// 4-patch concatenation). For 1152-dim vision: 4608x4608.
    pub mm0:    Weight,
    pub mm0_b:  Tensor,
    /// `[lm_hidden, merge_dim]`. For Qwen3.6-27B: 5120x4608.
    pub mm2:    Weight,
    pub mm2_b:  Tensor,
}

#[derive(Debug)]
pub struct Qwen3VlMmProj {
    pub config:        MmProjConfig,
    /// `[D, 3, P, P]` patch convolution. If a `.weight.1` temporal slot exists
    /// in the GGUF, it has been summed into this tensor at load time so the
    /// runtime path stays a single conv. (For a static image, summing the
    /// two temporal weights is equivalent to feeding [img, img] through the
    /// 2-temporal-step conv.)
    pub patch_embd:    Tensor,
    pub patch_embd_b:  Tensor,                 // [D]
    pub position_embd: Tensor,                 // [n_patches, D]
    pub blocks:        Vec<Qwen3VlBlock>,
    pub post_ln_w:     Tensor,                 // [D]
    pub post_ln_b:     Tensor,                 // [D]
    pub projector:     Qwen3VlProjector,
    pub backend:       Arc<dyn Backend>,
}

impl Qwen3VlMmProj {
    pub fn from_gguf_with_config(
        g: &GgufFile,
        config: MmProjConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        debug_assert_eq!(config.projector, ProjectorKind::Qwen3Vl);
        let idx = TensorIndex::new(g);

        // ----- patch conv (sum the two temporal slots if both present) ---
        let patch_embd_0 = idx.take("v.patch_embd.weight", &[])?;
        let patch_embd = match idx.try_take("v.patch_embd.weight.1") {
            Some(Ok(t1)) => sum_two_dense(patch_embd_0, t1)?,
            Some(Err(e)) => return Err(e),
            None         => patch_embd_0,
        };
        let patch_embd_b  = idx.take("v.patch_embd.bias",   &[])?;
        let position_embd = idx.take("v.position_embd.weight", &[])?;

        // ----- blocks ----------------------------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(Qwen3VlBlock {
                ln1_w:      idx.take(&format!("v.blk.{i}.ln1.weight"), &[])?,
                ln1_b:      idx.take(&format!("v.blk.{i}.ln1.bias"),   &[])?,
                attn_qkv:   idx.take_weight(&format!("v.blk.{i}.attn_qkv.weight"), &[])?,
                attn_qkv_b: idx.take(&format!("v.blk.{i}.attn_qkv.bias"), &[])?,
                attn_out:   idx.take_weight(&format!("v.blk.{i}.attn_out.weight"), &[])?,
                attn_out_b: idx.take(&format!("v.blk.{i}.attn_out.bias"), &[])?,
                ln2_w:      idx.take(&format!("v.blk.{i}.ln2.weight"), &[])?,
                ln2_b:      idx.take(&format!("v.blk.{i}.ln2.bias"),   &[])?,
                ffn_up:     idx.take_weight(&format!("v.blk.{i}.ffn_up.weight"), &[])?,
                ffn_up_b:   idx.take(&format!("v.blk.{i}.ffn_up.bias"), &[])?,
                ffn_down:   idx.take_weight(&format!("v.blk.{i}.ffn_down.weight"), &[])?,
                ffn_down_b: idx.take(&format!("v.blk.{i}.ffn_down.bias"), &[])?,
            });
        }

        // ----- post-LN + projector -------------------------------------
        let post_ln_w = idx.take("v.post_ln.weight", &[])?;
        let post_ln_b = idx.take("v.post_ln.bias",   &[])?;

        // The projector dims aren't fully fixed by config — `mm.0` is square
        // `[merge_dim, merge_dim]` and `mm.2` is `[lm_hidden, merge_dim]`.
        // We pull them with no shape constraint and just sanity-check them in
        // verify_qwen3vl_shapes below.
        let mm0   = idx.take_weight("mm.0.weight", &[])?;
        let mm0_b = idx.take("mm.0.bias", &[])?;
        let mm2   = idx.take_weight("mm.2.weight", &[])?;
        let mm2_b = idx.take("mm.2.bias", &[])?;
        let projector = Qwen3VlProjector { mm0, mm0_b, mm2, mm2_b };

        verify_qwen3vl_shapes(&config, &patch_embd, &position_embd, &blocks, &projector)?;

        // ----- upload to backend (1 GB margin — vision is one-shot) ----
        const M: usize = 1024 * 1024 * 1024;
        let upload = |t: Tensor| backend.to_device(t);
        let blocks: Vec<Qwen3VlBlock> = blocks.into_iter().map(|b| Qwen3VlBlock {
            ln1_w:      upload(b.ln1_w),
            ln1_b:      upload(b.ln1_b),
            attn_qkv:   b.attn_qkv.try_to_device(&*backend, M),
            attn_qkv_b: upload(b.attn_qkv_b),
            attn_out:   b.attn_out.try_to_device(&*backend, M),
            attn_out_b: upload(b.attn_out_b),
            ln2_w:      upload(b.ln2_w),
            ln2_b:      upload(b.ln2_b),
            ffn_up:     b.ffn_up.try_to_device(&*backend, M),
            ffn_up_b:   upload(b.ffn_up_b),
            ffn_down:   b.ffn_down.try_to_device(&*backend, M),
            ffn_down_b: upload(b.ffn_down_b),
        }).collect();
        let projector = Qwen3VlProjector {
            mm0:   projector.mm0.try_to_device(&*backend, M),
            mm0_b: upload(projector.mm0_b),
            mm2:   projector.mm2.try_to_device(&*backend, M),
            mm2_b: upload(projector.mm2_b),
        };
        let post_ln_w = upload(post_ln_w);
        let post_ln_b = upload(post_ln_b);
        let position_embd = upload(position_embd);
        // patch_embd kept on host until forward time (unfold runs CPU-side).

        Ok(Self {
            config, patch_embd, patch_embd_b: upload(patch_embd_b),
            position_embd, blocks, post_ln_w, post_ln_b, projector, backend,
        })
    }

    /// Image → soft tokens. Same contract as the other mmproj variants.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let backend = &*self.backend;
        if image.rank() != 3 || image.dim(0) != 3
            || image.dim(1) != cfg.image_size || image.dim(2) != cfg.image_size
        {
            return Err(LlamaError::Config(format!(
                "image shape {:?} doesn't match expected [3, {sz}, {sz}]",
                image.shape(), sz = cfg.image_size
            )));
        }

        // ----- 1. Patch conv as flattened matmul + bias ------------------
        let patches = unfold_patches_to_host(image, cfg.patch_size)?;
        let patches = backend.to_device(patches);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_w_host = self.patch_embd.to_host();
        let patch_w_flat = patch_w_host.reshape(vec![cfg.embedding_dim, in_dim])
            .map_err(|e| LlamaError::Config(format!("patch_embd reshape: {e:?}")))?;
        let patch_w_flat = backend.to_device(patch_w_flat);
        let mut x = backend.linear(&patches, &patch_w_flat);
        backend.add_inplace_broadcast_last(&mut x, &self.patch_embd_b);

        // ----- 2. Add learned absolute positions ------------------------
        backend.add_inplace(&mut x, &self.position_embd);

        // ----- 3. Transformer blocks ------------------------------------
        for blk in &self.blocks {
            x = qwen3vl_block_forward(backend, &x, blk, cfg);
        }

        // ----- 4. Post-LN ----------------------------------------------
        x = backend.layer_norm(&x, &self.post_ln_w, &self.post_ln_b, cfg.layer_norm_eps);

        // ----- 5. 2x2 spatial merge → [N/4, 4*D] -----------------------
        // The patch grid is `side` x `side`; collapse 2x2 neighbourhoods into
        // one row by concatenating their D-vectors in row-major order. Done
        // on host because it's a contiguous shuffle, not a matmul, and the
        // tensor is small (2304 * 1152 = ~2.7M elements).
        let side = cfg.image_size / cfg.patch_size;
        let new_side = side / 2;
        let d = cfg.embedding_dim;
        let merge_dim = 4 * d;
        let n_merged = new_side * new_side;
        let host = x.to_host();
        let src = host.data();
        let mut merged = vec![0.0f32; n_merged * merge_dim];
        for by in 0..new_side {
            for bx in 0..new_side {
                let dst_off = (by * new_side + bx) * merge_dim;
                // Sub-quadrant order: (0,0), (0,1), (1,0), (1,1) — row-major
                // within the 2x2 block. Matches HF's `Qwen3VLPatchMerger`.
                for sy in 0..2 {
                    for sx in 0..2 {
                        let py = by * 2 + sy;
                        let px = bx * 2 + sx;
                        let src_off = (py * side + px) * d;
                        let quad = sy * 2 + sx;
                        let dst_quad_off = dst_off + quad * d;
                        merged[dst_quad_off..dst_quad_off + d]
                            .copy_from_slice(&src[src_off..src_off + d]);
                    }
                }
            }
        }
        let mut h = backend.to_device(Tensor::from_vec(merged, vec![n_merged, merge_dim]));

        // ----- 6. mm.0 → GELU → mm.2 -----------------------------------
        let mut h0 = self.projector.mm0.linear(backend, &h);
        backend.add_inplace_broadcast_last(&mut h0, &self.projector.mm0_b);
        // VENDORED-LOCAL: HF's patch merger uses erf GELU; its ViT blocks
        // use tanh GELU. Evaluate this once per image on host (not per token).
        h = qwen_merger_gelu(backend, &h0);
        let mut out = self.projector.mm2.linear(backend, &h);
        backend.add_inplace_broadcast_last(&mut out, &self.projector.mm2_b);
        Ok(out)
    }
}

fn qwen_merger_gelu(b: &dyn Backend, x: &Tensor) -> Tensor {
    let mut host=x.to_host();
    for v in host.data_mut() {
        // Abramowitz-Stegun 7.1.26: absolute erf error below 1.5e-7.
        let z=*v*std::f32::consts::FRAC_1_SQRT_2;
        let t=1.0/(1.0+0.3275911*z.abs());
        let erf=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*(-z*z).exp();
        *v *= 0.5*(1.0+erf.copysign(z));
    }
    b.to_device(host)
}

/// Pre-LN attention + FFN block for the Qwen3-VL ViT. Fused QKV split into
/// three [N, D] tensors, bidirectional self-attention (no causal mask via
/// `past = n_seq` trick), GeLU MLP.
fn qwen3vl_block_forward(
    b:   &dyn Backend,
    x:   &Tensor,
    blk: &Qwen3VlBlock,
    cfg: &MmProjConfig,
) -> Tensor {
    let n_seq = x.dim(0);
    let n_h   = cfg.n_heads;
    let hd    = cfg.head_dim;
    let d     = cfg.embedding_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    // ----- Pre-LN + fused QKV --------------------------------------------
    let h0 = b.layer_norm(x, &blk.ln1_w, &blk.ln1_b, cfg.layer_norm_eps);
    let mut qkv = blk.attn_qkv.linear(b, &h0);            // [N, 3*D]
    b.add_inplace_broadcast_last(&mut qkv, &blk.attn_qkv_b);

    // Split [N, 3*D] → q, k, v each [N, D] via the backend's GPU-resident
    // 3-way splitter — no host round-trip on CUDA.
    let (q, k, v) = b.split_qkv_3way(&qkv, d);

    let mut q3 = q.reshape(vec![n_seq, n_h, hd]).expect("Q reshape");
    let mut k3 = k.reshape(vec![n_seq, n_h, hd]).expect("K reshape");
    let v3 = v.reshape(vec![n_seq, n_h, hd]).expect("V reshape");

    crate::multimodal_rope::vision(b, &mut q3, cfg.image_size / cfg.patch_size);
    crate::multimodal_rope::vision(b, &mut k3, cfg.image_size / cfg.patch_size);

    // Bidirectional via past = n_seq.
    let attn = b.attention(&q3, &k3, &v3, n_seq, scale, n_seq, None);
    let attn = attn.reshape(vec![n_seq, n_h * hd]).expect("attn reshape");

    let mut out = blk.attn_out.linear(b, &attn);
    b.add_inplace_broadcast_last(&mut out, &blk.attn_out_b);
    b.add_inplace(&mut out, x);

    // ----- Pre-LN + 2-layer GeLU MLP + residual --------------------------
    let h1 = b.layer_norm(&out, &blk.ln2_w, &blk.ln2_b, cfg.layer_norm_eps);
    let mut up = blk.ffn_up.linear(b, &h1);
    b.add_inplace_broadcast_last(&mut up, &blk.ffn_up_b);
    let activated = b.gelu_approx(&up);
    let mut down = blk.ffn_down.linear(b, &activated);
    b.add_inplace_broadcast_last(&mut down, &blk.ffn_down_b);
    b.add_inplace(&mut down, &out);
    down
}

/// Sum two same-shape dense tensors element-wise into a fresh tensor. Used to
/// fold the two temporal-slot patch convs into a single conv at load time.
fn sum_two_dense(a: Tensor, b: Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(LlamaError::Config(format!(
            "patch_embd temporal slot 0 shape {:?} != slot 1 shape {:?}",
            a.shape(), b.shape()
        )));
    }
    let shape = a.shape().to_vec();
    let a_h = a.to_host();
    let b_h = b.to_host();
    let mut out = a_h.data().to_vec();
    for (o, v) in out.iter_mut().zip(b_h.data()) {
        *o += *v;
    }
    Ok(Tensor::from_vec(out, shape))
}

fn verify_qwen3vl_shapes(
    cfg:           &MmProjConfig,
    patch_embd:    &Tensor,
    position_embd: &Tensor,
    blocks:        &[Qwen3VlBlock],
    projector:     &Qwen3VlProjector,
) -> Result<()> {
    let bad = |name: &str, got: &[usize], expected: &[usize]| -> LlamaError {
        LlamaError::BadTensorShape {
            name: name.into(),
            got: got.iter().map(|&v| v as u64).collect(),
            expected: expected.iter().map(|&v| v as u64).collect(),
        }
    };
    let d = cfg.embedding_dim;
    let p = cfg.patch_size;
    let np = cfg.n_patches();

    // Patch conv shape: [D, 3, P, P]
    if patch_embd.shape() != [d, 3, p, p] {
        return Err(bad("v.patch_embd", patch_embd.shape(), &[d, 3, p, p]));
    }
    if position_embd.shape() != [np, d] {
        return Err(bad("v.position_embd", position_embd.shape(), &[np, d]));
    }

    // mm.0: [merge_dim, merge_dim] where merge_dim = 4*D.
    let merge_dim = 4 * d;
    let s0 = projector.mm0.shape();
    if s0.len() != 2 || s0[0] != merge_dim || s0[1] != merge_dim {
        return Err(bad("mm.0.weight", s0, &[merge_dim, merge_dim]));
    }
    if projector.mm0_b.shape() != [merge_dim] {
        return Err(bad("mm.0.bias", projector.mm0_b.shape(), &[merge_dim]));
    }
    // mm.2: [lm_hidden, merge_dim] — lm_hidden unknown here, just check inner.
    let s2 = projector.mm2.shape();
    if s2.len() != 2 || s2[1] != merge_dim {
        return Err(bad("mm.2.weight", s2, &[0, merge_dim]));
    }
    if projector.mm2_b.shape() != [s2[0]] {
        return Err(bad("mm.2.bias", projector.mm2_b.shape(), &[s2[0]]));
    }

    // Block-level shapes are already checked by `idx.take_weight(.., expected)`
    // calls in the loader. This pass just guards against zero-block mmprojs.
    if blocks.len() != cfg.n_layers {
        return Err(LlamaError::Config(format!(
            "loaded {} vision blocks, expected {}", blocks.len(), cfg.n_layers
        )));
    }
    Ok(())
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
