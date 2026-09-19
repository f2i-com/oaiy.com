//! Vision-tower scaffolding for multimodal models. Step 1 of multimodal
//! support: image preprocessing pipeline that turns a path/bytes into a
//! `[3, H, W]` F32 tensor matching SigLIP / CLIP input conventions.
//!
//! Targets: Gemma 3 vision and Gemma 4 vision (both use Google's SigLIP-400M
//! tower with 896×896 input and patch size 14, producing 64×64 = 4096 patches
//! of 1152-dim embeddings; a 2-layer MLP then projects to the LM hidden dim).
//! Llava-style CLIP variants use 336×336 / 224×224 with different normalisation
//! constants; we'd add another `VisionConfig` for them.
//!
//! Status (this commit): **preprocessing only**. The vision tower forward
//! pass and mmproj.gguf loader are stubs to be filled in once a real
//! mmproj.gguf is available for testing. The preprocess pipeline is
//! self-testable with synthetic images.

use std::path::Path;

use ggml_rs::Tensor;
use image::imageops::FilterType;

use crate::{LlamaError, Result};

/// Vision encoder configuration. Different vision-LLM families use different
/// input sizes and normalisation constants — this struct lets the caller pick
/// the right preset rather than hard-coding it.
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig {
    /// Input image side length in pixels (square — vision towers always crop
    /// to a square). SigLIP-400M (Gemma 3/4): 896. CLIP-ViT-L/14 (Llava-1.5):
    /// 336. CLIP-ViT-B/16 (Llava-1.0): 224.
    pub image_size: usize,
    /// Patch size in pixels — image is split into `(image_size/patch_size)^2`
    /// non-overlapping square patches. SigLIP: 14. CLIP-L/14: 14. CLIP-B/16: 16.
    pub patch_size: usize,
    /// Per-channel mean for normalisation. Subtracted from `pixel/255.0`.
    /// SigLIP: `[0.5, 0.5, 0.5]` (centered 0..1 range). CLIP/OpenAI:
    /// `[0.48145, 0.4578, 0.40821]`.
    pub mean: [f32; 3],
    /// Per-channel std for normalisation. Divided after mean subtraction.
    /// SigLIP: `[0.5, 0.5, 0.5]` (yielding -1..1 output). CLIP/OpenAI:
    /// `[0.26863, 0.26130, 0.27577]`.
    pub std: [f32; 3],
}

impl VisionConfig {
    /// SigLIP-400M as used by Gemma 3 (4B/12B/27B vision) and Gemma 4
    /// (E2B/E4B). 896×896 input, patch 14, centered-and-scaled normalisation
    /// to a -1..1 output range.
    pub const SIGLIP_GEMMA: Self = Self {
        image_size: 896,
        patch_size: 14,
        mean: [0.5, 0.5, 0.5],
        std:  [0.5, 0.5, 0.5],
    };

    /// CLIP-ViT-L/14 as used by Llava-1.5. 336×336 input, patch 14, OpenAI's
    /// per-channel normalisation constants from the original CLIP paper.
    pub const CLIP_LLAVA: Self = Self {
        image_size: 336,
        patch_size: 14,
        mean: [0.48145466, 0.4578275, 0.40821073],
        std:  [0.26862954, 0.26130258, 0.27577711],
    };

    /// Number of patch tokens this config produces per image:
    /// `(image_size/patch_size)^2`. Gemma 3/4: 4096. Llava-1.5: 576.
    pub fn n_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }
}

/// Load an image from `path`, resize to `cfg.image_size × cfg.image_size`
/// using bilinear filtering, normalise per `cfg.mean` / `cfg.std`, and return
/// a `[3, H, W]` F32 tensor in CHW (channels-first) layout — what every vision
/// transformer expects as patch-conv input.
///
/// Resize strategy: a single bilinear resize to the target square. This
/// matches Gemma 3's pan_and_scan-disabled mode and Llava-1.5's default; some
/// configurations (Gemma 3 Hi-Res, Llava-Next) tile the image into multiple
/// crops first — that needs a separate pipeline that returns `Vec<Tensor>`.
///
/// Cost: dominated by the resize (~5 ms for a typical 1024×1024 → 896×896
/// bilinear). Not in the per-token critical path — runs once per image.
pub fn preprocess_image(path: &Path, cfg: &VisionConfig) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| LlamaError::Config(format!("failed to load image {path:?}: {e}")))?;
    preprocess_dynamic_image(img, cfg)
}

/// Same as [`preprocess_image`] but takes raw bytes — useful for in-memory
/// images (HTTP uploads, embedded data) without writing to disk first.
pub fn preprocess_image_bytes(bytes: &[u8], cfg: &VisionConfig) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| LlamaError::Config(format!("failed to decode image bytes: {e}")))?;
    preprocess_dynamic_image(img, cfg)
}

fn preprocess_dynamic_image(img: image::DynamicImage, cfg: &VisionConfig) -> Result<Tensor> {
    let size = cfg.image_size as u32;
    // Bilinear resize to the target square. `Triangle` is the image crate's
    // bilinear (separable triangle filter); matches Pillow's `BILINEAR` which
    // is what the HF SigLIP / CLIP processors use by default.
    let resized = img
        .resize_exact(size, size, FilterType::Triangle)
        .to_rgb8();

    // Normalise per channel and lay out as CHW. SigLIP / CLIP both consume
    // [3, H, W] in (R, G, B) channel order.
    let h = cfg.image_size;
    let w = cfg.image_size;
    let mut data = vec![0.0f32; 3 * h * w];
    let raw = resized.as_raw();   // length 3*h*w in HWC order (R, G, B per pixel).
    let mean = cfg.mean;
    let std = cfg.std;
    for y in 0..h {
        for x in 0..w {
            let src_off = (y * w + x) * 3;
            let r = raw[src_off    ] as f32 / 255.0;
            let g = raw[src_off + 1] as f32 / 255.0;
            let b = raw[src_off + 2] as f32 / 255.0;
            let dst_idx = y * w + x;
            data[0 * h * w + dst_idx] = (r - mean[0]) / std[0];
            data[1 * h * w + dst_idx] = (g - mean[1]) / std[1];
            data[2 * h * w + dst_idx] = (b - mean[2]) / std[2];
        }
    }
    Ok(Tensor::from_vec(data, vec![3, h, w]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_n_patches() {
        assert_eq!(VisionConfig::SIGLIP_GEMMA.n_patches(), 64 * 64);
        assert_eq!(VisionConfig::CLIP_LLAVA.n_patches(), 24 * 24);
    }

    /// Synthesise a solid-colour 256×256 PNG, run preprocess, verify the
    /// output tensor has the right shape and approximately the expected
    /// per-channel values after SigLIP normalisation.
    #[test]
    fn preprocess_solid_colour_image() {
        // Build a solid mid-grey (128, 128, 128) image in memory.
        let mut img = image::RgbImage::new(256, 256);
        for px in img.pixels_mut() { *px = image::Rgb([128, 128, 128]); }
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let t = preprocess_image_bytes(&buf, &VisionConfig::SIGLIP_GEMMA).unwrap();
        assert_eq!(t.shape(), &[3, 896, 896]);

        // SigLIP normalisation: (128/255 - 0.5) / 0.5 ≈ 0.00392.
        let expected = (128.0_f32 / 255.0 - 0.5) / 0.5;
        let data = t.data();
        // Sample a few pixels — bilinear resize of a solid image preserves
        // colour exactly, so every output pixel should be the same value.
        for &offset in &[0usize, 1000, 100_000, data.len() - 1] {
            let v = data[offset];
            assert!((v - expected).abs() < 1e-4,
                "pixel {offset}: got {v}, expected {expected}");
        }
    }

    /// Verify the channel-first layout: a red-only image should put values in
    /// channel 0 and 0-baseline-equivalent in channels 1 and 2.
    #[test]
    fn preprocess_red_image_channel_layout() {
        let mut img = image::RgbImage::new(64, 64);
        for px in img.pixels_mut() { *px = image::Rgb([255, 0, 0]); }
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let t = preprocess_image_bytes(&buf, &VisionConfig::SIGLIP_GEMMA).unwrap();
        let h = 896;
        let w = 896;
        let data = t.data();

        // Channel 0 (R): (255/255 - 0.5)/0.5 = 1.0
        assert!((data[h * w / 2 + 100] - 1.0).abs() < 1e-3);
        // Channels 1, 2 (G, B): (0/255 - 0.5)/0.5 = -1.0
        assert!((data[h * w + h * w / 2 + 100] - (-1.0)).abs() < 1e-3);
        assert!((data[2 * h * w + h * w / 2 + 100] - (-1.0)).abs() < 1e-3);
    }
}
