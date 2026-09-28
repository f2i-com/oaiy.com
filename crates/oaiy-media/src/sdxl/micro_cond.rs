// Ported from the user-owned F2I plugin-diffusion SDXL implementation.
//! SDXL micro-conditioning — the 1536-d "image size + crop" vector
//! that gets concatenated with the pooled CLIP-G output to form the
//! UNet's 2816-d ADM `y_label`.
//!
//! SDXL was trained with explicit conditioning on:
//! * `original_size_h, original_size_w` — the source image dims
//! * `crop_coords_top, crop_coords_left` — the random crop offset
//! * `target_size_h, target_size_w` — the output dims
//!
//! Each scalar is embedded via the standard sinusoidal formula at
//! `dim = block_out_channels[0] = 320` (wait — diffusers uses 256, not
//! 320). Verified: 6 × 256 = 1536, and 1280 (pooled CLIP-G) + 1536 = 2816
//! = `label_emb_in_dim`. So **micro-cond embed dim is 256**.
//!
//! ## Formula
//!
//! ```text
//!   sin/cos sinusoidal(scalar, dim=256, max_period=10000) → (256,)
//!   concat 6 of them → (1536,)
//!   concat with pooled_clip_g (1280,) → (2816,) → reshape (B, 2816)
//! ```
//!
//! ## Recommended values (for stock SDXL 1.0 text-to-image)
//!
//! * `original_size = target_size = (H, W)` (no resize)
//! * `crop_coords = (0, 0)` (no crop)
//! * `aesthetic_score` for the SDXL Refiner only — base ignores it

use candle_core::{Device, Result, Tensor, D};

/// Sinusoidal embedding of a single scalar to `dim`-d vector.
/// Output dtype is F32; the caller casts as needed.
fn sinusoidal_scalar(value: f64, dim: usize, max_period: f64, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut freqs: Vec<f32> = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = -(max_period.ln()) * (i as f64) / (half as f64);
        freqs.push(exponent.exp() as f32);
    }
    let freqs_t = Tensor::from_vec(freqs, (half,), device)?;
    let v = Tensor::from_vec(vec![value as f32], (1usize,), device)?;
    let args = v.broadcast_mul(&freqs_t)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let combined = Tensor::cat(&[&cos, &sin], D::Minus1)?;
    Ok(combined.squeeze(0)?) // (dim,)
}

/// Build the 6×256 = 1536-d micro-conditioning vector. Returns shape
/// `(1536,)` — caller broadcasts/repeats per batch.
pub fn build_micro_cond_vec(
    original_size: (u32, u32),
    crop_coords: (u32, u32),
    target_size: (u32, u32),
    device: &Device,
) -> Result<Tensor> {
    let dim = 256usize;
    let max_period = 10_000f64;
    let parts = [
        original_size.0 as f64,
        original_size.1 as f64,
        crop_coords.0 as f64,
        crop_coords.1 as f64,
        target_size.0 as f64,
        target_size.1 as f64,
    ];
    let mut chunks: Vec<Tensor> = Vec::with_capacity(parts.len());
    for v in parts {
        chunks.push(sinusoidal_scalar(v, dim, max_period, device)?);
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Build the SDXL `y_label` conditioning vector `(B, 2816)`:
/// concatenation of the pooled CLIP-G output `(B, 1280)` and the
/// 1536-d micro-conditioning (broadcast across batch).
///
/// The output dtype matches `pooled_clip_g`.
pub fn build_label_y(
    pooled_clip_g: &Tensor,
    original_size: (u32, u32),
    crop_coords: (u32, u32),
    target_size: (u32, u32),
) -> Result<Tensor> {
    let (b, _) = pooled_clip_g.dims2()?;
    let micro = build_micro_cond_vec(
        original_size,
        crop_coords,
        target_size,
        pooled_clip_g.device(),
    )?;
    let micro = micro.to_dtype(pooled_clip_g.dtype())?;
    // Repeat micro across the batch.
    let micro = micro.unsqueeze(0)?.expand((b, micro.dims1()?))?;
    Tensor::cat(&[pooled_clip_g, &micro], D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_cond_shape_is_1536() {
        let dev = Device::Cpu;
        let v = build_micro_cond_vec((1024, 1024), (0, 0), (1024, 1024), &dev).unwrap();
        assert_eq!(v.dims(), &[1536]);
    }

    #[test]
    fn label_y_shape_is_2816() {
        let dev = Device::Cpu;
        let pooled = Tensor::zeros((1usize, 1280), candle_core::DType::F32, &dev).unwrap();
        let y = build_label_y(&pooled, (1024, 1024), (0, 0), (1024, 1024)).unwrap();
        assert_eq!(y.dims(), &[1, 2816]);
    }
}
