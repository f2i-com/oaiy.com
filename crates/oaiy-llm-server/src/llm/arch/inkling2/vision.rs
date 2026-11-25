//! Inkling-2 Vision Tower (`inkling2v` mmproj GGUF & native `model.visual.*` safetensors).
//!
//! Architecture (verified against HF `Inkling2VisionModel` & `Inkling2ImageProcessor`
//! and llama.cpp `mtmd_graph_inkling2v`):
//!   * Preprocessor: composite alpha onto 8×8 checkerboard (204 / 153), pad right/bottom
//!     to a multiple of `patch_size * spatial_merge_size` (= 40) with fill value `-1.0` on
//!     the `0..255` scale, rescale `/ 255.0`, normalize `(x - mean) / std` (`0.5 / 0.5`).
//!   * Patch embedding (`20×20×3 -> 1024`), bilinear-interpolated 2D position embedding
//!     (`32×32 -> H_p×W_p`, `align_corners = false`), prepend CLS token
//!     (`class_embedding + cls_position_embedding`) at position `(0, 0)`.
//!   * `pre_layernorm` (LayerNorm, `eps = 1e-5`) over `[CLS, patches...]`.
//!   * 23 ViT encoder layers: Pre-LN, full bidirectional self-attention with 2D axial
//!     pairwise RoPE (`theta = 10000`, width first then height, 1-indexed `(row+1, col+1)`),
//!     LayerScale (`ls1` / `ls2`), exact GELU (`erf`) MLP.
//!   * Drop CLS token, 2×2 spatial merge (`4 * 1024 = 4096`), biasless 2-layer GELU MLP
//!     (`4096 -> 4096 -> 2304`).

use std::sync::{Arc, OnceLock};

use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use ggml_rs::{Backend, Tensor};
use gguf::GgufFile;
use oaiy_engine::safetensors::StIndex;
use rayon::prelude::*;

use crate::llm::loader::{load_tensor_f32, TensorIndex, Weight};
use crate::llm::{LlamaError, Result};

use super::forward::{linear_host_batch, linear_host_fast, GpuScratch};

/// Maximum merged image tokens per image before downscaling (40x40 px per token:
/// 1024 tokens ≈ 1280×1280 px, matching practical context budgets while preserving
/// native resolution for all normal screenshots and photos).
pub const DEFAULT_MAX_IMAGE_TOKENS: usize = 1024;

#[derive(Debug, Clone)]
pub struct Inkling2VisionConfig {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub posemb_grid_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ff_dim: usize,
    pub num_layers: usize,
    pub proj_hidden_dim: usize,
    pub text_hidden_size: usize,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Inkling2VisionConfig {
    pub fn cell_size(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }
}

/// Preprocessed image ready for [`Inkling2Vision::encode`].
#[derive(Debug, Clone)]
pub struct PreparedInkling2Image {
    /// Patches laid out in merged-block order:
    /// `[rows * cols * merge * merge, patch * patch * 3]` (HWC inside each patch).
    pub patches: Vec<f32>,
    /// Merged grid height (in language-model image tokens).
    pub rows: usize,
    /// Merged grid width (in language-model image tokens).
    pub cols: usize,
}

impl PreparedInkling2Image {
    pub fn n_tokens(&self) -> usize {
        self.rows * self.cols
    }
}

/// Compute resized `(width, height)` so that `height.div_ceil(cell) * width.div_ceil(cell) <= max_tokens.max(1)`
/// even for extreme aspect ratios like `1 x 1_000_000`.
fn fit_dimensions(width: usize, height: usize, cell: usize, max_tokens: usize) -> (usize, usize) {
    let cell = cell.max(1);
    let max_tok = max_tokens.max(1);
    let est_tokens = height.div_ceil(cell).saturating_mul(width.div_ceil(cell));
    if est_tokens <= max_tok {
        return (width, height);
    }
    let scale = (max_tok as f64 / est_tokens as f64).sqrt();
    let mut new_w = ((width as f64 * scale).floor() as usize).max(1);
    let mut new_h = ((height as f64 * scale).floor() as usize).max(1);
    let mut grid_w = new_w.div_ceil(cell);
    let mut grid_h = new_h.div_ceil(cell);
    if grid_h.saturating_mul(grid_w) > max_tok {
        if new_w >= new_h {
            let cap_w = (max_tok / grid_h).max(1).saturating_mul(cell);
            new_w = new_w.min(cap_w);
            grid_w = new_w.div_ceil(cell);
        } else {
            let cap_h = (max_tok / grid_w).max(1).saturating_mul(cell);
            new_h = new_h.min(cap_h);
            grid_h = new_h.div_ceil(cell);
        }
        if grid_h.saturating_mul(grid_w) > max_tok {
            let cap_h = (max_tok / grid_w).max(1).saturating_mul(cell);
            new_h = new_h.min(cap_h);
            grid_h = new_h.div_ceil(cell);
            if grid_h.saturating_mul(grid_w) > max_tok {
                let cap_w = (max_tok / grid_h).max(1).saturating_mul(cell);
                new_w = new_w.min(cap_w);
            }
        }
    }
    (new_w.max(1), new_h.max(1))
}

/// Decode image bytes and run the exact `Inkling2ImageProcessor` preprocessing.
pub fn preprocess_image_bytes(
    bytes: &[u8],
    cfg: &Inkling2VisionConfig,
    max_tokens: usize,
) -> std::result::Result<PreparedInkling2Image, String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("decode image: {e}"))?;
    let mut width = img.width() as usize;
    let mut height = img.height() as usize;
    if width == 0 || height == 0 {
        return Err("empty image".into());
    }

    let cell = cfg.cell_size();
    let (new_w, new_h) = fit_dimensions(width, height, cell, max_tokens);
    let img = if new_w != width || new_h != height {
        width = new_w;
        height = new_h;
        img.resize_exact(new_w as u32, new_h as u32, image::imageops::FilterType::CatmullRom)
    } else {
        img
    };

    // Composite alpha over 8x8 checkerboard (204 / 153) matching `Inkling2ImageProcessor.convert_to_rgb`.
    let has_alpha = img.color().has_alpha();
    let mut rgb = vec![0u8; height * width * 3];
    if has_alpha {
        let rgba = img.to_rgba8();
        for y in 0..height {
            for x in 0..width {
                let p = rgba.get_pixel(x as u32, y as u32).0;
                let bg: i32 = if ((y / 8) + (x / 8)) % 2 == 0 { 204 } else { 153 };
                let a = p[3] as i32;
                let off = (y * width + x) * 3;
                for c in 0..3 {
                    rgb[off + c] = ((p[c] as i32 * a + bg * (255 - a) + 127) / 255) as u8;
                }
            }
        }
    } else {
        let raw = img.to_rgb8();
        rgb.copy_from_slice(raw.as_raw());
    }

    preprocess_rgb(&rgb, width, height, cfg)
}

/// Preprocess an RGB byte buffer `[height, width, 3]` into Inkling-2 merged-order patches.
pub fn preprocess_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    cfg: &Inkling2VisionConfig,
) -> std::result::Result<PreparedInkling2Image, String> {
    if width == 0 || height == 0 {
        return Err("image has zero width or height".into());
    }
    if rgb.len() != height * width * 3 {
        return Err(format!("expected {} RGB bytes, got {}", height * width * 3, rgb.len()));
    }
    let patch = cfg.patch_size;
    let merge = cfg.spatial_merge_size;
    let cell = patch * merge;
    let rows = height.div_ceil(cell);
    let cols = width.div_ceil(cell);

    let pad_vals = [
        (-1.0f32 / 255.0 - cfg.mean[0]) / cfg.std[0],
        (-1.0f32 / 255.0 - cfg.mean[1]) / cfg.std[1],
        (-1.0f32 / 255.0 - cfg.mean[2]) / cfg.std[2],
    ];

    let patch_dim = patch * patch * 3;
    let n_patches = rows * cols * merge * merge;
    let mut patches = vec![0.0f32; n_patches * patch_dim];

    let mut p_idx = 0usize;
    for mr in 0..rows {
        for mc in 0..cols {
            for sr in 0..merge {
                for sc in 0..merge {
                    let y0 = (mr * merge + sr) * patch;
                    let x0 = (mc * merge + sc) * patch;
                    let dst = &mut patches[p_idx * patch_dim..(p_idx + 1) * patch_dim];
                    let mut d = 0usize;
                    for py in 0..patch {
                        let y = y0 + py;
                        for px in 0..patch {
                            let x = x0 + px;
                            if y < height && x < width {
                                let off = (y * width + x) * 3;
                                dst[d] = (rgb[off] as f32 / 255.0 - cfg.mean[0]) / cfg.std[0];
                                dst[d + 1] = (rgb[off + 1] as f32 / 255.0 - cfg.mean[1]) / cfg.std[1];
                                dst[d + 2] = (rgb[off + 2] as f32 / 255.0 - cfg.mean[2]) / cfg.std[2];
                            } else {
                                dst[d] = pad_vals[0];
                                dst[d + 1] = pad_vals[1];
                                dst[d + 2] = pad_vals[2];
                            }
                            d += 3;
                        }
                    }
                    p_idx += 1;
                }
            }
        }
    }

    Ok(PreparedInkling2Image { patches, rows, cols })
}

#[derive(Debug)]
struct ViTLayer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    ls1: Vec<f32>,
    ls2: Vec<f32>,
    q_w: Weight,
    q_b: Vec<f32>,
    k_w: Weight,
    k_b: Vec<f32>,
    v_w: Weight,
    v_b: Vec<f32>,
    o_w: Weight,
    o_b: Vec<f32>,
    fc1_w: Weight,
    fc1_b: Vec<f32>,
    fc2_w: Weight,
    fc2_b: Vec<f32>,
}

#[derive(Debug)]
enum GpuVisionLin {
    /// Weight is already resident on `DeviceChain` as `Weight::Quant` or `Weight::Packed`.
    OnDevice,
    /// Dense F32/F16 weight uploaded once as packed F16 for Tensor Core `matmul_f16_rows_ranged`.
    Half { w: DeviceVec, out_dim: usize, in_dim: usize },
    /// Fallback dense F32 weight on `DeviceChain`.
    F32 { w: DeviceVec, out_dim: usize, in_dim: usize },
}

impl GpuVisionLin {
    fn from_weight(chain: &dyn DeviceChain, w: &Weight) -> Option<Self> {
        match w {
            Weight::Quant(qt) if chain.holds(qt) => Some(Self::OnDevice),
            Weight::Packed(p) if chain.holds_exl3(p.as_ref()) => Some(Self::OnDevice),
            Weight::Dense(t) if t.rank() == 2 => {
                let (out_dim, in_dim) = (t.dim(0), t.dim(1));
                let host = if t.is_device() { t.to_host() } else { t.clone() };
                if in_dim % 2 == 0 {
                    if let Some(hw) = chain.vec_f16_rounded(host.data()) {
                        return Some(Self::Half { w: hw, out_dim, in_dim });
                    }
                }
                let fw = chain.vec(host.numel());
                chain.upload(&fw, host.data());
                Some(Self::F32 { w: fw, out_dim, in_dim })
            }
            _ => None,
        }
    }

    fn record(&self, rec: &mut dyn ChainRecorder, weight: &Weight, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match self {
            Self::OnDevice => match weight {
                Weight::Quant(qt) => rec.matmul_rows(qt, x, y, rows),
                Weight::Packed(p) => rec.exl3_rows(p.as_ref(), x, y, rows),
                _ => unreachable!("GpuVisionLin::OnDevice on non-device weight"),
            },
            Self::Half { w, out_dim, in_dim } => {
                rec.matmul_f16_rows_ranged(w, *out_dim, *in_dim, x, y, rows);
            }
            Self::F32 { w, out_dim, in_dim } => {
                rec.matmul_f32_rows(w, *out_dim, *in_dim, x, y, rows);
            }
        }
    }
}

#[derive(Debug)]
struct GpuViTLayer {
    ln1: DeviceVec,
    ln2: DeviceVec,
    ls1: DeviceVec,
    ls2: DeviceVec,
    q: GpuVisionLin,
    q_b: DeviceVec,
    k: GpuVisionLin,
    k_b: DeviceVec,
    v: GpuVisionLin,
    v_b: DeviceVec,
    o: GpuVisionLin,
    o_b: DeviceVec,
    fc1: GpuVisionLin,
    fc1_b: DeviceVec,
    fc2: GpuVisionLin,
    fc2_b: DeviceVec,
}

#[derive(Debug)]
struct GpuVisionTower {
    pre_ln: DeviceVec,
    layers: Vec<GpuViTLayer>,
    mm1: GpuVisionLin,
    mm2: GpuVisionLin,
}

/// Loaded Inkling-2 Vision Tower + 2×2 Spatial Merger MLP.
#[derive(Debug)]
pub struct Inkling2Vision {
    pub cfg: Inkling2VisionConfig,
    backend: Arc<dyn Backend>,
    /// `[hidden, patch * patch * 3]` in HWC order (matches safetensors directly; GGUF CHW is permuted at load).
    patch_w: Weight,
    /// `[grid * grid, hidden]`
    pos_embd: Vec<f32>,
    /// `[hidden]` (`class_embedding + cls_position_embedding`)
    cls_token: Vec<f32>,
    pre_ln_w: Vec<f32>,
    pre_ln_b: Vec<f32>,
    layers: Vec<ViTLayer>,
    mm1_w: Weight,
    mm2_w: Weight,
    gpu_tower: OnceLock<Option<GpuVisionTower>>,
}

impl Inkling2Vision {
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    pub fn n_blocks(&self) -> usize {
        self.layers.len()
    }

    /// Load from a llama.cpp `mmproj` GGUF (`clip.vision.projector_type = "inkling2v"`).
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let ptype = g
            .metadata()
            .get("clip.vision.projector_type")
            .or_else(|| g.metadata().get("clip.projector_type"))
            .and_then(|v| v.as_str());
        if ptype != Some("inkling2v") {
            return Err(LlamaError::Config(format!(
                "expected inkling2v mmproj projector_type, got {ptype:?}"
            )));
        }

        let image_size = g.get_u64("clip.vision.image_size")? as usize;
        let patch_size = g.get_u64("clip.vision.patch_size")? as usize;
        let spatial_merge_size = g.get_u64("clip.vision.spatial_merge_size").unwrap_or(2) as usize;
        let hidden_size = g.get_u64("clip.vision.embedding_length")? as usize;
        let num_heads = g.get_u64("clip.vision.attention.head_count")? as usize;
        let ff_dim = g.get_u64("clip.vision.feed_forward_length")? as usize;
        let num_layers = g.get_u64("clip.vision.block_count")? as usize;
        let text_hidden_size = g.get_u64("clip.vision.projection_dim")? as usize;
        let layer_norm_eps = g
            .get_f32("clip.vision.attention.layer_norm_epsilon")
            .unwrap_or(1e-5);
        let mean = read_three_f32(g, "clip.vision.image_mean").unwrap_or([0.5; 3]);
        let std = read_three_f32(g, "clip.vision.image_std").unwrap_or([0.5; 3]);

        if patch_size == 0 || num_heads == 0 || hidden_size % num_heads != 0 {
            return Err(LlamaError::Config("invalid inkling2v vision dimensions".into()));
        }
        let posemb_grid_size = image_size / patch_size;

        let idx = TensorIndex::new(g);
        let req = |name: &str| -> Result<&gguf::TensorInfo> {
            idx.info(name).ok_or_else(|| LlamaError::MissingTensor(name.to_string()))
        };
        let load_vec = |name: &str| -> Result<Vec<f32>> {
            Ok(load_tensor_f32(g, req(name)?)?.data().to_vec())
        };
        let load_w = |name: &str| -> Result<Weight> {
            Ok(Weight::load(g, req(name)?)?.to_device(backend.as_ref()))
        };

        // GGUF `v.patch_embd.weight` has GGML dims [patch, patch, 3, hidden] -> row-major [hidden, 3, patch, patch].
        // Permute to [hidden, patch, patch, 3] (HWC) so forward is unified with safetensors.
        let raw_patch = load_tensor_f32(g, req("v.patch_embd.weight")?)?;
        let patch_dim = patch_size * patch_size * 3;
        if raw_patch.numel() != hidden_size * patch_dim {
            return Err(LlamaError::Config(format!(
                "v.patch_embd.weight numel {} != {}",
                raw_patch.numel(),
                hidden_size * patch_dim
            )));
        }
        let src = raw_patch.data();
        let mut hwc = vec![0.0f32; hidden_size * patch_dim];
        let pp = patch_size * patch_size;
        for o in 0..hidden_size {
            for c in 0..3 {
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        let src_i = o * (3 * pp) + c * pp + py * patch_size + px;
                        let dst_i = o * patch_dim + (py * patch_size + px) * 3 + c;
                        hwc[dst_i] = src[src_i];
                    }
                }
            }
        }
        let patch_w = Weight::Dense(Tensor::from_vec(hwc, vec![hidden_size, patch_dim])).to_device(backend.as_ref());

        let pos_embd = load_vec("v.position_embd.weight")?;
        let class_embd = load_vec("v.class_embd")?;
        let class_pos = load_vec("v.class_position_embd.weight")?;
        let cls_token: Vec<f32> = class_embd.iter().zip(&class_pos).map(|(a, b)| a + b).collect();
        let pre_ln_w = load_vec("v.pre_ln.weight")?;
        let pre_ln_b = load_vec("v.pre_ln.bias")?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(ViTLayer {
                ln1_w: load_vec(&format!("v.blk.{i}.ln1.weight"))?,
                ln1_b: load_vec(&format!("v.blk.{i}.ln1.bias"))?,
                ln2_w: load_vec(&format!("v.blk.{i}.ln2.weight"))?,
                ln2_b: load_vec(&format!("v.blk.{i}.ln2.bias"))?,
                ls1: load_vec(&format!("v.blk.{i}.ls1.weight"))?,
                ls2: load_vec(&format!("v.blk.{i}.ls2.weight"))?,
                q_w: load_w(&format!("v.blk.{i}.attn_q.weight"))?,
                q_b: load_vec(&format!("v.blk.{i}.attn_q.bias"))?,
                k_w: load_w(&format!("v.blk.{i}.attn_k.weight"))?,
                k_b: load_vec(&format!("v.blk.{i}.attn_k.bias"))?,
                v_w: load_w(&format!("v.blk.{i}.attn_v.weight"))?,
                v_b: load_vec(&format!("v.blk.{i}.attn_v.bias"))?,
                o_w: load_w(&format!("v.blk.{i}.attn_out.weight"))?,
                o_b: load_vec(&format!("v.blk.{i}.attn_out.bias"))?,
                fc1_w: load_w(&format!("v.blk.{i}.ffn_up.weight"))?,
                fc1_b: load_vec(&format!("v.blk.{i}.ffn_up.bias"))?,
                fc2_w: load_w(&format!("v.blk.{i}.ffn_down.weight"))?,
                fc2_b: load_vec(&format!("v.blk.{i}.ffn_down.bias"))?,
            });
        }

        let mm1_w = load_w("mm.1.weight")?;
        let mm2_w = load_w("mm.2.weight")?;
        let proj_hidden_dim = mm1_w.shape()[0];

        let cfg = Inkling2VisionConfig {
            patch_size,
            spatial_merge_size,
            posemb_grid_size,
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
            ff_dim,
            num_layers,
            proj_hidden_dim,
            text_hidden_size,
            layer_norm_eps,
            rope_theta: 10000.0,
            mean,
            std,
        };

        Ok(Self {
            cfg,
            backend,
            patch_w,
            pos_embd,
            cls_token,
            pre_ln_w,
            pre_ln_b,
            layers,
            mm1_w,
            mm2_w,
            gpu_tower: OnceLock::new(),
        })
    }

    /// Load the built-in vision tower (`model.visual.*` or `model.vision_tower.*`) directly
    /// from an Inkling-2 Hugging Face safetensors checkpoint directory.
    pub fn from_safetensors(
        index: &StIndex,
        cfg: Inkling2VisionConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        use super::safetensors::{read_st_f32, read_st_weight};

        let raw_layout = index.get("model.visual.backbone.ln_pre.weight").is_some();
        let load_vec = |raw: &str, hf: &str| -> Result<Vec<f32>> {
            let name = if raw_layout { raw } else { hf };
            read_st_f32(index, name)
        };
        let load_w = |raw: &str, hf: &str| -> Result<Weight> {
            let name = if raw_layout { raw } else { hf };
            read_st_weight(index, name, backend.as_ref())
        };

        let patch_w = load_w(
            "model.visual.backbone.embeddings.patch_embedding.weight",
            "model.vision_tower.embeddings.patch_embedding.weight",
        )?;
        let pos_embd = load_vec(
            "model.visual.backbone.embeddings.position_embedding",
            "model.vision_tower.embeddings.position_embedding.weight",
        )?;
        let class_embd = load_vec(
            "model.visual.backbone.embeddings.class_embedding",
            "model.vision_tower.embeddings.class_embedding",
        )?;
        let class_pos = load_vec(
            "model.visual.backbone.embeddings.cls_position_embedding",
            "model.vision_tower.embeddings.cls_position_embedding",
        )?;
        let cls_token: Vec<f32> = class_embd.iter().zip(&class_pos).map(|(a, b)| a + b).collect();
        let pre_ln_w = load_vec(
            "model.visual.backbone.ln_pre.weight",
            "model.vision_tower.pre_layernorm.weight",
        )?;
        let pre_ln_b = load_vec(
            "model.visual.backbone.ln_pre.bias",
            "model.vision_tower.pre_layernorm.bias",
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(ViTLayer {
                ln1_w: load_vec(
                    &format!("model.visual.backbone.layers.{i}.ln_1.weight"),
                    &format!("model.vision_tower.layers.{i}.layernorm_before.weight"),
                )?,
                ln1_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.ln_1.bias"),
                    &format!("model.vision_tower.layers.{i}.layernorm_before.bias"),
                )?,
                ln2_w: load_vec(
                    &format!("model.visual.backbone.layers.{i}.ln_2.weight"),
                    &format!("model.vision_tower.layers.{i}.layernorm_after.weight"),
                )?,
                ln2_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.ln_2.bias"),
                    &format!("model.vision_tower.layers.{i}.layernorm_after.bias"),
                )?,
                ls1: load_vec(
                    &format!("model.visual.backbone.layers.{i}.ls_1.gamma"),
                    &format!("model.vision_tower.layers.{i}.lambda_1"),
                )?,
                ls2: load_vec(
                    &format!("model.visual.backbone.layers.{i}.ls_2.gamma"),
                    &format!("model.vision_tower.layers.{i}.lambda_2"),
                )?,
                q_w: load_w(
                    &format!("model.visual.backbone.layers.{i}.self_attn.q_proj.weight"),
                    &format!("model.vision_tower.layers.{i}.self_attn.q_proj.weight"),
                )?,
                q_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.self_attn.q_proj.bias"),
                    &format!("model.vision_tower.layers.{i}.self_attn.q_proj.bias"),
                )?,
                k_w: load_w(
                    &format!("model.visual.backbone.layers.{i}.self_attn.k_proj.weight"),
                    &format!("model.vision_tower.layers.{i}.self_attn.k_proj.weight"),
                )?,
                k_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.self_attn.k_proj.bias"),
                    &format!("model.vision_tower.layers.{i}.self_attn.k_proj.bias"),
                )?,
                v_w: load_w(
                    &format!("model.visual.backbone.layers.{i}.self_attn.v_proj.weight"),
                    &format!("model.vision_tower.layers.{i}.self_attn.v_proj.weight"),
                )?,
                v_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.self_attn.v_proj.bias"),
                    &format!("model.vision_tower.layers.{i}.self_attn.v_proj.bias"),
                )?,
                o_w: load_w(
                    &format!("model.visual.backbone.layers.{i}.self_attn.out_proj.weight"),
                    &format!("model.vision_tower.layers.{i}.self_attn.out_proj.weight"),
                )?,
                o_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.self_attn.out_proj.bias"),
                    &format!("model.vision_tower.layers.{i}.self_attn.out_proj.bias"),
                )?,
                fc1_w: load_w(
                    &format!("model.visual.backbone.layers.{i}.mlp.c_fc.weight"),
                    &format!("model.vision_tower.layers.{i}.mlp.fc1.weight"),
                )?,
                fc1_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.mlp.c_fc.bias"),
                    &format!("model.vision_tower.layers.{i}.mlp.fc1.bias"),
                )?,
                fc2_w: load_w(
                    &format!("model.visual.backbone.layers.{i}.mlp.c_proj.weight"),
                    &format!("model.vision_tower.layers.{i}.mlp.fc2.weight"),
                )?,
                fc2_b: load_vec(
                    &format!("model.visual.backbone.layers.{i}.mlp.c_proj.bias"),
                    &format!("model.vision_tower.layers.{i}.mlp.fc2.bias"),
                )?,
            });
        }

        let mm1_w = load_w("model.visual.proj_in.weight", "model.vision_tower.merger.linear_1.weight")?;
        let mm2_w = load_w("model.visual.proj_out.weight", "model.vision_tower.merger.linear_2.weight")?;

        Ok(Self {
            cfg,
            backend,
            patch_w,
            pos_embd,
            cls_token,
            pre_ln_w,
            pre_ln_b,
            layers,
            mm1_w,
            mm2_w,
            gpu_tower: OnceLock::new(),
        })
    }

    /// Encode a preprocessed image into `[rows * cols, text_hidden_size]` soft embeddings.
    pub fn encode(&self, img: &PreparedInkling2Image) -> Result<Vec<f32>> {
        let d = self.cfg.hidden_size;
        let merge = self.cfg.spatial_merge_size;
        let patch_dim = self.cfg.patch_size * self.cfg.patch_size * 3;
        let n_merged = img.rows * img.cols;
        let n_patches = n_merged * merge * merge;
        let n_seq = n_patches + 1; // +1 for CLS
        let hp = img.rows * merge;
        let wp = img.cols * merge;
        let b = self.backend.as_ref();
        let mut scratch = GpuScratch::default();

        // 1. Linear patch embedding: `[n_patches, patch_dim] -> [n_patches, d]`.
        let patch_emb = linear_host_fast(b, &mut scratch, &self.patch_w, &img.patches, n_patches, patch_dim, usize::MAX);

        // 2. Interpolated 2D position embeddings (bilinear, align_corners=False) + prepend CLS.
        let mut x = vec![0.0f32; n_seq * d];
        x[..d].copy_from_slice(&self.cls_token);

        let g = self.cfg.posemb_grid_size;
        let mut coords: Vec<(usize, usize)> = Vec::with_capacity(n_seq);
        coords.push((0, 0)); // CLS sits at (0, 0)

        let mut p_idx = 0usize;
        for mr in 0..img.rows {
            for mc in 0..img.cols {
                for sr in 0..merge {
                    for sc in 0..merge {
                        let r = mr * merge + sr;
                        let c = mc * merge + sc;
                        coords.push((r + 1, c + 1)); // 1-indexed (row, col)

                        let src_y = (((r as f32 + 0.5) * g as f32 / hp as f32) - 0.5)
                            .clamp(0.0, (g - 1) as f32);
                        let src_x = (((c as f32 + 0.5) * g as f32 / wp as f32) - 0.5)
                            .clamp(0.0, (g - 1) as f32);
                        let y0 = src_y.floor() as usize;
                        let x0 = src_x.floor() as usize;
                        let y1 = (y0 + 1).min(g - 1);
                        let x1 = (x0 + 1).min(g - 1);
                        let wy = src_y - y0 as f32;
                        let wx = src_x - x0 as f32;

                        let w00 = (1.0 - wy) * (1.0 - wx);
                        let w01 = (1.0 - wy) * wx;
                        let w10 = wy * (1.0 - wx);
                        let w11 = wy * wx;

                        let p00 = &self.pos_embd[(y0 * g + x0) * d..(y0 * g + x0 + 1) * d];
                        let p01 = &self.pos_embd[(y0 * g + x1) * d..(y0 * g + x1 + 1) * d];
                        let p10 = &self.pos_embd[(y1 * g + x0) * d..(y1 * g + x0 + 1) * d];
                        let p11 = &self.pos_embd[(y1 * g + x1) * d..(y1 * g + x1 + 1) * d];

                        let pe = &patch_emb[p_idx * d..(p_idx + 1) * d];
                        let dst = &mut x[(p_idx + 1) * d..(p_idx + 2) * d];
                        for k in 0..d {
                            dst[k] = pe[k] + w00 * p00[k] + w01 * p01[k] + w10 * p10[k] + w11 * p11[k];
                        }
                        p_idx += 1;
                    }
                }
            }
        }

        // 3. Pre-LN + 23 ViT encoder layers + 2x2 merger MLP:
        //    execute the entire tower in ONE `DeviceChain` submission on GPU when available
        //    (eliminating 100+ CPU-GPU round trips and CPU O(N²) ViT attention).
        if std::env::var_os("OAIY_NO_CHAIN").is_none() {
            if let Some(out) = self.encode_gpu(&x, &coords, n_seq, n_patches, n_merged) {
                return Ok(out);
            }
        }

        // CPU/fallback path: Pre-LN over [CLS, patches...].
        x = layer_norm_rows(&x, &self.pre_ln_w, &self.pre_ln_b, d, self.cfg.layer_norm_eps);

        // 4. Precompute 2D axial pairwise RoPE (cos, sin) tables: width first, then height.
        let (cos_tbl, sin_tbl) = build_axial_rope(&coords, self.cfg.head_dim, self.cfg.rope_theta);

        // 5. ViT encoder layers.
        let n_heads = self.cfg.num_heads;
        let head_dim = self.cfg.head_dim;
        let ff_dim = self.cfg.ff_dim;
        let eps = self.cfg.layer_norm_eps;

        for layer in &self.layers {
            let ln1 = layer_norm_rows(&x, &layer.ln1_w, &layer.ln1_b, d, eps);
            let mut qkv = linear_host_batch(
                b,
                &mut scratch,
                &[&ln1],
                &[(&layer.q_w, 0), (&layer.k_w, 0), (&layer.v_w, 0)],
                n_seq,
                d,
            )
            .into_iter();
            let mut q = qkv.next().expect("q");
            let mut k = qkv.next().expect("k");
            let mut v = qkv.next().expect("v");
            add_bias_inplace(&mut q, &layer.q_b, n_seq, d);
            add_bias_inplace(&mut k, &layer.k_b, n_seq, d);
            add_bias_inplace(&mut v, &layer.v_b, n_seq, d);

            apply_pairwise_rope(&mut q, &cos_tbl, &sin_tbl, n_seq, n_heads, head_dim);
            apply_pairwise_rope(&mut k, &cos_tbl, &sin_tbl, n_seq, n_heads, head_dim);

            let attn_out = vit_attention(&q, &k, &v, n_seq, n_heads, head_dim);
            let proj = linear_bias(b, &mut scratch, &layer.o_w, &layer.o_b, &attn_out, n_seq, d, d);
            for t in 0..n_seq {
                let off = t * d;
                for c in 0..d {
                    x[off + c] += layer.ls1[c] * proj[off + c];
                }
            }

            let ln2 = layer_norm_rows(&x, &layer.ln2_w, &layer.ln2_b, d, eps);
            let mut h = linear_bias(b, &mut scratch, &layer.fc1_w, &layer.fc1_b, &ln2, n_seq, d, ff_dim);
            for val in &mut h {
                *val = gelu_exact(*val);
            }
            let mlp_out = linear_bias(b, &mut scratch, &layer.fc2_w, &layer.fc2_b, &h, n_seq, ff_dim, d);
            for t in 0..n_seq {
                let off = t * d;
                for c in 0..d {
                    x[off + c] += layer.ls2[c] * mlp_out[off + c];
                }
            }
        }

        // 6. Drop CLS token; patches are already contiguous in 2x2 merged-block order!
        let merged_dim = merge * merge * d;
        let patches_only = &x[d..];
        let mut h1 = linear_host_fast(b, &mut scratch, &self.mm1_w, patches_only, n_merged, merged_dim, usize::MAX);
        for val in &mut h1 {
            *val = gelu_exact(*val);
        }
        Ok(linear_host_fast(
            b,
            &mut scratch,
            &self.mm2_w,
            &h1,
            n_merged,
            self.cfg.proj_hidden_dim,
            usize::MAX,
        ))
    }

    fn build_gpu_tower(&self, chain: &dyn DeviceChain) -> Option<GpuVisionTower> {
        let up = |v: &[f32]| -> DeviceVec {
            let dv = chain.vec(v.len());
            chain.upload(&dv, v);
            dv
        };
        let up_ln = |w: &[f32], b: &[f32]| -> DeviceVec {
            let mut mods = Vec::with_capacity(w.len() + b.len());
            mods.extend(w.iter().map(|&v| v - 1.0));
            mods.extend_from_slice(b);
            up(&mods)
        };
        let pre_ln = up_ln(&self.pre_ln_w, &self.pre_ln_b);
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(GpuViTLayer {
                ln1: up_ln(&l.ln1_w, &l.ln1_b),
                ln2: up_ln(&l.ln2_w, &l.ln2_b),
                ls1: up(&l.ls1),
                ls2: up(&l.ls2),
                q: GpuVisionLin::from_weight(chain, &l.q_w)?,
                q_b: up(&l.q_b),
                k: GpuVisionLin::from_weight(chain, &l.k_w)?,
                k_b: up(&l.k_b),
                v: GpuVisionLin::from_weight(chain, &l.v_w)?,
                v_b: up(&l.v_b),
                o: GpuVisionLin::from_weight(chain, &l.o_w)?,
                o_b: up(&l.o_b),
                fc1: GpuVisionLin::from_weight(chain, &l.fc1_w)?,
                fc1_b: up(&l.fc1_b),
                fc2: GpuVisionLin::from_weight(chain, &l.fc2_w)?,
                fc2_b: up(&l.fc2_b),
            });
        }
        let mm1 = GpuVisionLin::from_weight(chain, &self.mm1_w)?;
        let mm2 = GpuVisionLin::from_weight(chain, &self.mm2_w)?;
        Some(GpuVisionTower { pre_ln, layers, mm1, mm2 })
    }

    fn encode_gpu(
        &self,
        x_host: &[f32],
        coords: &[(usize, usize)],
        n_seq: usize,
        n_patches: usize,
        n_merged: usize,
    ) -> Option<Vec<f32>> {
        if n_seq == 0 || n_patches == 0 || n_merged == 0 {
            return None;
        }
        let chain = self.backend.chain()?;
        let tower = self.gpu_tower.get_or_init(|| self.build_gpu_tower(chain)).as_ref()?;

        let d = self.cfg.hidden_size;
        let n_heads = self.cfg.num_heads;
        let head_dim = self.cfg.head_dim;
        let ff_dim = self.cfg.ff_dim;
        let eps = self.cfg.layer_norm_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Build interleaved `[sin, cos]` table matching `ChainRecorder::rope_rows(..., neox=false)`.
        let (cos_tbl, sin_tbl) = build_axial_rope(coords, head_dim, self.cfg.rope_theta);
        let mut rope_host = Vec::with_capacity(n_seq * head_dim);
        for (&s, &c) in sin_tbl.iter().zip(&cos_tbl) {
            rope_host.push(s);
            rope_host.push(c);
        }

        let normed = chain.vec(n_seq * d);
        let x_buf = chain.vec(n_seq * d);
        let q = chain.vec(n_seq * d);
        let k = chain.vec(n_seq * d);
        let v = chain.vec(n_seq * d);
        let kv = chain.vec(n_seq * 2 * d);
        let att_len = chain.attention_rows_full_out_len(n_seq, n_heads, head_dim, n_seq);
        let att = chain.vec(att_len);
        let att_view = DeviceVec { len: n_seq * d, inner: Arc::clone(&att.inner) };
        let proj = chain.vec(n_seq * d);
        let mlp_h = chain.vec(n_seq * ff_dim);
        let mlp_act = chain.vec(n_seq * ff_dim);
        let rope_tbl = chain.vec(rope_host.len());
        let merged = chain.vec(n_patches * d);
        let mm_h = chain.vec(n_merged * self.cfg.proj_hidden_dim);
        let mm_act = chain.vec(n_merged * self.cfg.proj_hidden_dim);
        let out = chain.vec(n_merged * self.cfg.text_hidden_size);

        chain.upload(&normed, x_host);
        chain.upload(&rope_tbl, &rope_host);

        let mut rec = chain.begin();
        rec.keep_groups(false);

        // Pre-LN: `normed` -> `x_buf`.
        rec.norm_mod_rows(&normed, &x_buf, n_seq, d, &tower.pre_ln, 0, Some(d), RowNorm::Layer, eps);

        // 23 ViT encoder layers.
        for (layer, gl) in self.layers.iter().zip(&tower.layers) {
            rec.norm_mod_rows(&x_buf, &normed, n_seq, d, &gl.ln1, 0, Some(d), RowNorm::Layer, eps);

            gl.q.record(&mut *rec, &layer.q_w, &normed, &q, n_seq);
            rec.add_bias_rows(&q, &gl.q_b, n_seq, d);
            gl.k.record(&mut *rec, &layer.k_w, &normed, &k, n_seq);
            rec.add_bias_rows(&k, &gl.k_b, n_seq, d);
            gl.v.record(&mut *rec, &layer.v_w, &normed, &v, n_seq);
            rec.add_bias_rows(&v, &gl.v_b, n_seq, d);

            rec.rope_rows(&q, n_seq, n_heads, head_dim, &rope_tbl, false);
            rec.rope_rows(&k, n_seq, n_heads, head_dim, &rope_tbl, false);

            rec.store_rows(&k, &kv, n_seq, d, 0, 2 * d, 0);
            rec.store_rows(&v, &kv, n_seq, d, 0, 2 * d, d);
            rec.attention_rows_full(&q, &kv, &att, n_seq, n_heads, n_heads, head_dim, n_seq, scale);

            gl.o.record(&mut *rec, &layer.o_w, &att_view, &proj, n_seq);
            rec.add_bias_rows(&proj, &gl.o_b, n_seq, d);
            rec.add_gated_rows(&x_buf, &proj, n_seq, d, &gl.ls1, 0, false);

            rec.norm_mod_rows(&x_buf, &normed, n_seq, d, &gl.ln2, 0, Some(d), RowNorm::Layer, eps);
            gl.fc1.record(&mut *rec, &layer.fc1_w, &normed, &mlp_h, n_seq);
            rec.add_bias_rows(&mlp_h, &gl.fc1_b, n_seq, ff_dim);
            rec.gelu_erf(&mlp_h, &mlp_act, n_seq * ff_dim);
            gl.fc2.record(&mut *rec, &layer.fc2_w, &mlp_act, &proj, n_seq);
            rec.add_bias_rows(&proj, &gl.fc2_b, n_seq, d);
            rec.add_gated_rows(&x_buf, &proj, n_seq, d, &gl.ls2, 0, false);
        }

        // Drop CLS token + 2x2 spatial merger MLP (`4096 -> 4096 -> text_hidden`).
        rec.copy(&x_buf, d, &merged, 0, n_patches * d);
        tower.mm1.record(&mut *rec, &self.mm1_w, &merged, &mm_h, n_merged);
        rec.gelu_erf(&mm_h, &mm_act, n_merged * self.cfg.proj_hidden_dim);
        tower.mm2.record(&mut *rec, &self.mm2_w, &mm_act, &out, n_merged);
        rec.read(&out);

        rec.finish().into_iter().next()
    }
}

fn read_three_f32(g: &GgufFile, key: &str) -> Option<[f32; 3]> {
    let arr = g.metadata().get(key)?.as_array()?;
    if let gguf::Array::F32(v) = arr {
        if v.len() == 3 {
            return Some([v[0], v[1], v[2]]);
        }
    }
    None
}

#[inline]
fn add_bias_inplace(y: &mut [f32], bias: &[f32], rows: usize, out_dim: usize) {
    for r in 0..rows {
        let row = &mut y[r * out_dim..(r + 1) * out_dim];
        for c in 0..out_dim {
            row[c] += bias[c];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_bias(
    backend: &dyn Backend,
    scratch: &mut GpuScratch,
    w: &Weight,
    bias: &[f32],
    x: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = linear_host_fast(backend, scratch, w, x, rows, in_dim, usize::MAX);
    add_bias_inplace(&mut y, bias, rows, out_dim);
    y
}

fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let src = &x[r * dim..(r + 1) * dim];
        let dst = &mut out[r * dim..(r + 1) * dim];
        let mean = src.iter().sum::<f32>() / dim as f32;
        let var = src.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            dst[c] = (src[c] - mean) * inv * w[c] + b[c];
        }
    }
    out
}

/// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))` using Abramowitz & Stegun 7.1.26 (`|err| < 1.5e-7`).
#[inline]
pub(crate) fn gelu_exact(x: f32) -> f32 {
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let z = (x as f64) * INV_SQRT2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * az);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let erf = sign * (1.0 - poly * (-az * az).exp());
    (0.5 * (x as f64) * (1.0 + erf)) as f32
}

/// Build `[n_seq, head_dim / 2]` cos/sin tables: width frequencies in `0..head_dim/4`,
/// height frequencies in `head_dim/4..head_dim/2`.
fn build_axial_rope(coords: &[(usize, usize)], head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let spatial_dim = head_dim / 2;
    let n_freqs = spatial_dim / 2;
    let inv_freq: Vec<f32> = (0..n_freqs)
        .map(|k| 1.0 / theta.powf((2 * k) as f32 / spatial_dim as f32))
        .collect();
    let n_pairs = head_dim / 2;
    let mut cos_tbl = vec![0.0f32; coords.len() * n_pairs];
    let mut sin_tbl = vec![0.0f32; coords.len() * n_pairs];
    for (i, &(h_pos, w_pos)) in coords.iter().enumerate() {
        let base = i * n_pairs;
        for k in 0..n_freqs {
            let aw = (w_pos as f32) * inv_freq[k];
            let ah = (h_pos as f32) * inv_freq[k];
            cos_tbl[base + k] = aw.cos();
            sin_tbl[base + k] = aw.sin();
            cos_tbl[base + n_freqs + k] = ah.cos();
            sin_tbl[base + n_freqs + k] = ah.sin();
        }
    }
    (cos_tbl, sin_tbl)
}

/// Pairwise rotary position embedding: `(x_{2p}, x_{2p+1}) -> (x_{2p}*c - x_{2p+1}*s, x_{2p+1}*c + x_{2p}*s)`.
fn apply_pairwise_rope(
    x: &mut [f32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    n_seq: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let n_pairs = head_dim / 2;
    let d = n_heads * head_dim;
    for t in 0..n_seq {
        let c_row = &cos_tbl[t * n_pairs..(t + 1) * n_pairs];
        let s_row = &sin_tbl[t * n_pairs..(t + 1) * n_pairs];
        for h in 0..n_heads {
            let slice = &mut x[t * d + h * head_dim..t * d + (h + 1) * head_dim];
            for p in 0..n_pairs {
                let x0 = slice[2 * p];
                let x1 = slice[2 * p + 1];
                let c = c_row[p];
                let s = s_row[p];
                slice[2 * p] = x0 * c - x1 * s;
                slice[2 * p + 1] = x1 * c + x0 * s;
            }
        }
    }
}

/// Full bidirectional multi-head self-attention for ViT (`[n_seq, n_heads * head_dim]`).
fn vit_attention(q: &[f32], k: &[f32], v: &[f32], n_seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let d = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_seq * d];

    out.par_chunks_mut(head_dim).enumerate().for_each(|(th, dst)| {
        let t = th / n_heads;
        let h = th % n_heads;
        let q_vec = &q[t * d + h * head_dim..t * d + (h + 1) * head_dim];

        let mut max_s = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        dst.fill(0.0);

        for s in 0..n_seq {
            let k_vec = &k[s * d + h * head_dim..s * d + (h + 1) * head_dim];
            let v_vec = &v[s * d + h * head_dim..s * d + (h + 1) * head_dim];
            let mut dot = 0.0f32;
            for j in 0..head_dim {
                dot += q_vec[j] * k_vec[j];
            }
            let score = dot * scale;
            if score > max_s {
                let rescale = (max_s - score).exp();
                sum_exp = sum_exp * rescale + 1.0;
                for j in 0..head_dim {
                    dst[j] = dst[j] * rescale + v_vec[j];
                }
                max_s = score;
            } else {
                let w = (score - max_s).exp();
                sum_exp += w;
                for j in 0..head_dim {
                    dst[j] += w * v_vec[j];
                }
            }
        }
        let inv = 1.0 / sum_exp;
        for val in dst.iter_mut() {
            *val *= inv;
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocessor_pads_with_minus_one_and_composites_checkerboard() {
        let cfg = Inkling2VisionConfig {
            patch_size: 20,
            spatial_merge_size: 2,
            posemb_grid_size: 32,
            hidden_size: 16,
            num_heads: 2,
            head_dim: 8,
            ff_dim: 32,
            num_layers: 1,
            proj_hidden_dim: 16,
            text_hidden_size: 16,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        // 10x10 solid white RGB image -> pads to 40x40 (1 merged token = 4 patches of 20x20).
        let rgb = vec![255u8; 10 * 10 * 3];
        let prep = preprocess_rgb(&rgb, 10, 10, &cfg).unwrap();
        assert_eq!((prep.rows, prep.cols, prep.n_tokens()), (1, 1, 1));
        assert_eq!(prep.patches.len(), 4 * 20 * 20 * 3);
        // Pixel (0, 0) in top-left subpatch -> (255/255 - 0.5)/0.5 = 1.0.
        assert!((prep.patches[0] - 1.0).abs() < 1e-6);
        // Pixel (0, 15) in top-left subpatch -> padded with -1.0 on 0..255 scale -> (-1/255 - 0.5)/0.5.
        let want_pad = (-1.0f32 / 255.0 - 0.5) / 0.5;
        assert!((prep.patches[15 * 3] - want_pad).abs() < 1e-6);
    }

    #[test]
    fn exact_gelu_matches_known_points() {
        assert!(gelu_exact(0.0).abs() < 1e-6);
        assert!((gelu_exact(1.0) - 0.84134475).abs() < 1e-5);
        assert!((gelu_exact(-1.0) - (-0.15865525)).abs() < 1e-5);
        assert!((gelu_exact(3.0) - 2.99595).abs() < 1e-4);
    }

    #[test]
    fn extreme_aspect_ratios_never_exceed_max_tokens() {
        let cell = 40usize;
        for max_tok in [1usize, 10, 256, 1024] {
            for (w, h) in [
                (1_000_000, 1),
                (1, 1_000_000),
                (10_000_000, 2),
                (2, 10_000_000),
                (2000, 2000),
                (1280, 720),
            ] {
                let (nw, nh) = fit_dimensions(w, h, cell, max_tok);
                let tokens = nh.div_ceil(cell) * nw.div_ceil(cell);
                assert!(
                    tokens <= max_tok,
                    "{w}x{h} with max_tok={max_tok} resized to {nw}x{nh} ({tokens} tokens)"
                );
                assert!(nw >= 1 && nh >= 1);
            }
        }
    }
}
