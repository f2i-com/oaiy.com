//! Bounded local reference-image decoding and Qwen3-VL patch ordering.
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

pub struct Reference {
    pub rgba: image::RgbaImage,
    pub h: usize,
    pub w: usize,
}
impl Reference {
    pub fn load(path: &Path, size: usize) -> Result<Self> {
        let meta = std::fs::metadata(path)?;
        if !meta.is_file() || meta.len() > 32 * 1024 * 1024 {
            candle_core::bail!(
                "reference must be a regular image file of at most 32 MiB: {}",
                path.display()
            );
        }
        let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(16384);
        limits.max_image_height = Some(16384);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader.decode().map_err(candle_core::Error::wrap)?;
        let ratio = decoded.width() as f64 / decoded.height() as f64;
        if !(1. / 8. ..=8.).contains(&ratio) {
            candle_core::bail!("reference aspect ratio must be between 1:8 and 8:1");
        }
        let w = ((size as f64 * ratio.sqrt() / 32.).round() as usize * 32).max(32);
        let h = ((size as f64 / ratio.sqrt() / 32.).round() as usize * 32).max(32);
        let rgba = image::imageops::resize(
            &decoded.to_rgba8(),
            w as u32,
            h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        Ok(Self { rgba, h, w })
    }
    pub fn pixels(&self, dev: &Device, dtype: DType) -> Result<Tensor> {
        let data: Vec<f32> = self
            .rgba
            .as_raw()
            .iter()
            .map(|&v| v as f32 / 127.5 - 1.)
            .collect();
        Tensor::from_vec(data, (1, self.h, self.w, 4), dev)?
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .to_dtype(dtype)
    }
    pub fn patches(&self, dev: &Device, dtype: DType) -> Result<Tensor> {
        let mut data = Vec::with_capacity(self.h * self.w * 6);
        for (row, col) in patch_positions(self.h / 16, self.w / 16) {
            for c in 0..3 {
                for _ in 0..2 {
                    for y in 0..16 {
                        for x in 0..16 {
                            let p = self
                                .rgba
                                .get_pixel((col * 16 + x) as u32, (row * 16 + y) as u32)
                                .0;
                            // The VLM sees RGB composited on white; the VAE keeps alpha.
                            let rgb =
                                (p[c] as f32 * p[3] as f32 / 255. + 255. - p[3] as f32).round();
                            data.push(rgb / 127.5 - 1.);
                        }
                    }
                }
            }
        }
        Tensor::from_vec(data, (self.h * self.w / 256, 1536), dev)?.to_dtype(dtype)
    }
}
pub fn patch_positions(h: usize, w: usize) -> Vec<(usize, usize)> {
    let mut positions = Vec::with_capacity(h * w);
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    positions.push((y + dy, x + dx));
                }
            }
        }
    }
    positions
}
#[test]
fn patches_merge_in_two_by_two_groups() {
    assert_eq!(
        patch_positions(2, 4),
        vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3)
        ]
    );
}

#[test]
fn transparent_reference_keeps_alpha_but_vision_sees_white() -> Result<()> {
    let image = Reference {
        rgba: image::RgbaImage::from_pixel(32, 32, image::Rgba([0, 0, 0, 0])),
        h: 32,
        w: 32,
    };
    let rgba = image
        .pixels(&Device::Cpu, DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(rgba.iter().all(|v| *v == -1.));
    let patches = image.patches(&Device::Cpu, DType::F32)?.to_vec2::<f32>()?;
    assert_eq!(patches.len(), 4);
    assert!(patches.iter().flatten().all(|v| *v == 1.));
    Ok(())
}

#[test]
fn png_jpeg_and_webp_references_decode_with_preserved_aspect() -> Result<()> {
    let root = std::env::temp_dir().join(format!("nrob-reference-codecs-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let source = image::RgbImage::from_pixel(128, 64, image::Rgb([200, 100, 50]));
    for extension in ["png", "jpg", "webp"] {
        let path = root.join(format!("reference.{extension}"));
        source.save(&path).map_err(candle_core::Error::wrap)?;
        let decoded = Reference::load(&path, 256)?;
        assert_eq!((decoded.w, decoded.h), (352, 192));
        assert_eq!(decoded.rgba.get_pixel(0, 0).0[3], 255);
        std::fs::remove_file(path)?;
    }
    std::fs::remove_dir(root)?;
    Ok(())
}
