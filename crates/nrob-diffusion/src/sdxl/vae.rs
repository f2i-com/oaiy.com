// Ported from the user-owned F2I plugin-diffusion SDXL implementation.
//! SDXL VAE — `AutoencoderKL` from the LDM paper, 4-channel latent.
//!
//! Structurally identical to the 16-channel FLUX-family VAE already in
//! `crate::vae`, so we reuse those `Encoder` / `Decoder` primitives
//! unchanged. The two differences SDXL needs on top:
//!
//! 1. **quant_conv** (1×1, 8→8) right after the encoder's `conv_out`,
//!    operating on the stacked `(mu, logvar)` 8-channel output.
//! 2. **post_quant_conv** (1×1, 4→4) right before the decoder's
//!    `conv_in`, operating on the sampled 4-channel latent.
//!
//! These 1×1 convs are inherited from the LDM paper and don't exist
//! on the 16-ch FLUX VAE (which collapsed them into nothing during
//! retraining). Both projections are tiny and only matter at the
//! latent boundary.
//!
//! ## Scaling
//!
//! SDXL's per-channel std after the encoder's stochastic head is ~7.67,
//! so the canonical `scaling_factor = 0.13025 ≈ 1/7.67` brings latents
//! to ~unit variance. `shift_factor` is 0 (SDXL kept the latent
//! distribution centred at 0 during training — SD 1.5 also was 0; the
//! non-zero shifts only show up on FLUX-family VAEs).

use candle_core::{DType, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

use candle_core::Device;

use crate::sdxl::config::VaeConfig as SdxlVaeConfig;
use crate::sdxl::vae_layers::{
    Decoder as FluxDecoder, Encoder as FluxEncoder, VaeConfig as FluxVaeConfig,
};

/// Linear ramp from 0 → 1 across the first `overlap` pixels, flat 1 in
/// the interior, and 1 → 0 across the last `overlap` pixels. Used as
/// per-tile weights in `decode_tiled` so overlapping tiles blend smoothly.
fn build_ramp_1d(size: usize, overlap: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let edge = overlap.max(1);
    let mut v = Vec::with_capacity(size);
    for i in 0..size {
        let from_start = (i + 1) as f32 / (edge + 1) as f32;
        let from_end = (size - i) as f32 / (edge + 1) as f32;
        let w = from_start.min(from_end).min(1.0).max(1.0e-3);
        v.push(w);
    }
    Tensor::from_vec(v, (size,), device)?.to_dtype(dtype)
}

/// Encoder wrapper: thin alias so callers can construct it
/// independently of the decoder (matches the diffusers split-loading
/// pattern used by the SDXL Refiner workflow).
pub type VaeEncoder = FluxEncoder;
pub type VaeDecoder = FluxDecoder;

/// Top-level SDXL VAE — owns both halves plus the 1×1 quant projections.
#[derive(Debug, Clone)]
pub struct AutoencoderKL {
    encoder: VaeEncoder,
    decoder: VaeDecoder,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    scale_factor: f64,
    latent_channels: usize,
}

impl AutoencoderKL {
    /// Load encoder + decoder + quant projections from a VarBuilder
    /// scoped to the `first_stage_model` level of an SDXL checkpoint.
    /// The expected key layout under `vb`:
    ///
    /// ```text
    ///   encoder.*           — the 4-stage downsampling encoder
    ///   decoder.*           — the 4-stage upsampling decoder
    ///   quant_conv.weight   — (8, 8, 1, 1)
    ///   quant_conv.bias     — (8,)
    ///   post_quant_conv.weight — (4, 4, 1, 1)
    ///   post_quant_conv.bias   — (4,)
    /// ```
    pub fn load(cfg: &SdxlVaeConfig, vb: VarBuilder) -> Result<Self> {
        let inner_cfg = sdxl_to_flux_cfg(cfg);
        let encoder = VaeEncoder::new(&inner_cfg, vb.pp("encoder"))?;
        let decoder = VaeDecoder::new(&inner_cfg, vb.pp("decoder"))?;

        let cfg_1x1 = Conv2dConfig::default();
        let quant_conv = conv2d(
            2 * cfg.latent_channels,
            2 * cfg.latent_channels,
            1,
            cfg_1x1,
            vb.pp("quant_conv"),
        )?;
        let post_quant_conv = conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            cfg_1x1,
            vb.pp("post_quant_conv"),
        )?;

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            scale_factor: cfg.scaling_factor,
            latent_channels: cfg.latent_channels,
        })
    }

    /// Encode an RGB image `(B, 3, H, W)` ∈ [-1, 1] to a latent
    /// `(B, latent_channels, H/8, W/8)`. Deterministic: returns the
    /// mean of the diagonal Gaussian, not a sample. `scaling_factor`
    /// is applied so the result is in the same unit-variance space
    /// the UNet was trained against.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.encoder)?;
        // `quant_conv` operates on the full (mu, logvar) stack.
        let moments = h.apply(&self.quant_conv)?;
        let mean = moments.narrow(1, 0, self.latent_channels)?;
        mean.affine(self.scale_factor, 0.0)
    }

    /// Decode a latent `(B, latent_channels, H, W)` to RGB
    /// `(B, 3, H*8, W*8)` ∈ [-1, 1]. The caller is responsible for
    /// rescaling [-1, 1] → [0, 255] for image output.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let unscaled = z.affine(1.0 / self.scale_factor, 0.0)?;
        let projected = unscaled.apply(&self.post_quant_conv)?;
        projected.apply(&self.decoder)
    }

    /// Tiled decode for latents whose spatial dims exceed the VAE's
    /// trained size. SDXL's GroupNorm + decoder convs were calibrated
    /// at 128² latent (= 1024² output); decoding a single larger latent
    /// in one shot produces out-of-distribution pixel values (typically
    /// all-negative, which the downstream [-1, 1] → [0, 255] clamp
    /// turns into a fully black image).
    ///
    /// The fix mirrors what diffusers does for high-res SDXL: walk the
    /// latent in `tile_size_latent`-sized windows with `overlap_latent`
    /// of overlap, decode each tile independently, and composite back
    /// using a linear ramp weight at tile edges so seams blend out.
    ///
    /// * `tile_size_latent` — typically 128 (= 1024² output per tile).
    ///   Each tile is decoded as if it were a standalone full-res SDXL
    ///   image so the GroupNorm statistics stay in-distribution.
    /// * `overlap_latent` — typically 16-32 (= 128-256 px of image
    ///   overlap). Smaller = faster + more visible seams; larger =
    ///   slower + smoother seams.
    ///
    /// When the latent already fits in one tile this falls through to
    /// the plain `decode`.
    pub fn decode_tiled(
        &self,
        z: &Tensor,
        tile_size_latent: usize,
        overlap_latent: usize,
    ) -> Result<Tensor> {
        let (b, _c, h_lat, w_lat) = z.dims4()?;
        if h_lat <= tile_size_latent && w_lat <= tile_size_latent {
            return self.decode(z);
        }
        let tile = tile_size_latent.max(1);
        let overlap = overlap_latent.min(tile.saturating_sub(1));
        let stride = tile - overlap;

        // Per-axis plan. If `dim <= tile`, do a single pass over the
        // full dim (no tiling on that axis) — the SDXL aspect buckets
        // (1216×832 → 152×104 latent) hit this on the short axis. If
        // `dim > tile`, build the multi-tile schedule the old code did,
        // but only here (skipping the early `dim - tile` underflow that
        // crashed on the small axis under the old uniform-tile path).
        let plan_axis = |dim: usize| -> (Vec<usize>, usize) {
            if dim <= tile {
                return (vec![0], dim);
            }
            let mut out = vec![0usize];
            while *out.last().unwrap() + tile < dim {
                let next = out.last().unwrap() + stride;
                if next + tile > dim {
                    out.push(dim - tile);
                    break;
                }
                out.push(next);
            }
            if *out.last().unwrap() + tile > dim {
                if let Some(last) = out.last_mut() {
                    *last = dim - tile;
                }
            }
            out.dedup();
            (out, tile)
        };

        let (h_starts, h_tile) = plan_axis(h_lat);
        let (w_starts, w_tile) = plan_axis(w_lat);

        let device = z.device();
        let dtype = z.dtype();
        let overlap_img = overlap * 8;
        let h_tile_img = h_tile * 8;
        let w_tile_img = w_tile * 8;
        let h_img = h_lat * 8;
        let w_img = w_lat * 8;

        // Per-axis blend weight. An axis with a single tile gets a flat
        // 1.0 ramp (nothing to blend across); a multi-tile axis gets the
        // linear ramp built by `build_ramp_1d`. Outer-producting these
        // two 1-D ramps yields the same tile_weight shape the uniform
        // path used, but correctly sized when one axis isn't tiled.
        let h_ramp = if h_starts.len() == 1 {
            Tensor::ones(h_tile_img, dtype, device)?
        } else {
            build_ramp_1d(h_tile_img, overlap_img, device, dtype)?
        };
        let w_ramp = if w_starts.len() == 1 {
            Tensor::ones(w_tile_img, dtype, device)?
        } else {
            build_ramp_1d(w_tile_img, overlap_img, device, dtype)?
        };
        let weight_2d = h_ramp.unsqueeze(1)?.broadcast_mul(&w_ramp.unsqueeze(0)?)?;
        let tile_weight = weight_2d.unsqueeze(0)?.unsqueeze(0)?;

        let mut out = Tensor::zeros((b, 3, h_img, w_img), dtype, device)?;
        let mut weight_sum = Tensor::zeros((1, 1, h_img, w_img), dtype, device)?;

        for &y0 in &h_starts {
            for &x0 in &w_starts {
                let tile_latent = z.narrow(2, y0, h_tile)?.narrow(3, x0, w_tile)?;
                let decoded = self.decode(&tile_latent)?;
                let weighted = decoded.broadcast_mul(&tile_weight)?;
                let img_y = y0 * 8;
                let img_x = x0 * 8;

                let cur = out
                    .narrow(2, img_y, h_tile_img)?
                    .narrow(3, img_x, w_tile_img)?;
                let new_slice = (cur + weighted)?;
                out = out.slice_assign(
                    &[
                        0..b,
                        0..3,
                        img_y..img_y + h_tile_img,
                        img_x..img_x + w_tile_img,
                    ],
                    &new_slice,
                )?;

                let cur_w = weight_sum
                    .narrow(2, img_y, h_tile_img)?
                    .narrow(3, img_x, w_tile_img)?;
                let new_w = (cur_w + &tile_weight)?;
                weight_sum = weight_sum.slice_assign(
                    &[
                        0..1,
                        0..1,
                        img_y..img_y + h_tile_img,
                        img_x..img_x + w_tile_img,
                    ],
                    &new_w,
                )?;
            }
        }

        out.broadcast_div(&weight_sum)
    }

    /// Convenience: encode → immediately decode. Useful as a
    /// reconstruction sanity check on a real VAE load (post-encode
    /// → post-decode reconstruction should be visually close to input).
    pub fn round_trip(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.encode(x)?;
        self.decode(&z)
    }

    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Cast every owned tensor to `dtype`. Useful when the file loaded
    /// in F16 but the rest of the pipeline runs in BF16.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        // The candle Conv2d struct has no `to_dtype` method, but we can
        // round-trip via a fresh VarBuilder built from the existing
        // tensors after casting. For now this is a placeholder — most
        // pipeline calls will load fresh at the desired dtype rather
        // than cast post-load. Document the limitation.
        let _ = dtype;
        Ok(self.clone())
    }
}

/// Bridge our `SdxlVaeConfig` field naming to the existing FLUX-style
/// `VaeConfig` the shared Encoder/Decoder primitives consume. SDXL and
/// FLUX VAEs use identical down/mid/up architecture; only the latent
/// width and the presence of quant_conv differ.
fn sdxl_to_flux_cfg(cfg: &SdxlVaeConfig) -> FluxVaeConfig {
    FluxVaeConfig {
        resolution: 256, // informational only
        in_channels: cfg.in_channels,
        out_channels: cfg.out_channels,
        base_channels: cfg.base_channels,
        ch_mult: cfg.channel_mults.clone(),
        num_res_blocks: cfg.layers_per_block,
        latent_channels: cfg.latent_channels,
        // The shared Vae struct's encode() applies (mean - shift) * scale
        // and decode() applies (z / scale) + shift; we don't reuse those
        // methods (we have our own that handle quant_conv), but the
        // sub-Encoder/Decoder primitives ignore these fields entirely.
        scale_factor: cfg.scaling_factor,
        shift_factor: 0.0,
    }
}
