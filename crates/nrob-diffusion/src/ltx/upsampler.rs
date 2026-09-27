//! The LTX latent spatial upsampler (`LatentUpsampler`, 3D, x2): the bridge
//! between the reference's two stages. Stage one renders at half size; this
//! doubles the latent's height and width, and stage two refines it.
//!
//! In the reference: 3D convolutions with zero padding, GroupNorm(32) and
//! SiLU, four residual blocks, a per-frame 2D convolution to four times the
//! channels and a 2x2 pixel shuffle, four more residual blocks, and a last 3D
//! convolution back to 128 channels. It works on un-normalized latents.
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub struct Upsampler {
    w: HashMap<String, Tensor>,
    blocks: usize,
}

impl Upsampler {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let mut store = super::store::Store::open(path, 0)?;
        let config = store
            .index
            .metadata("config")
            .ok_or_else(|| candle_core::Error::Msg("the latent upsampler lacks its config".into()))?
            .to_owned();
        let c = nrob::json::Json::parse(config.as_bytes()).map_err(candle_core::Error::wrap)?;
        let int = |k: &str| c.get(k).and_then(nrob::json::Json::as_i64);
        let flag = |k: &str| c.get(k).and_then(nrob::json::Json::as_bool);
        if int("in_channels") != Some(128)
            || int("dims") != Some(3)
            || flag("spatial_upsample") != Some(true)
            || flag("temporal_upsample") == Some(true)
            || flag("rational_resampler") == Some(true)
            // Missing keys take the reference's defaults (x2, no rational resampler).
            || c.get("spatial_scale").and_then(nrob::json::Json::as_f64).unwrap_or(2.) != 2.
        {
            candle_core::bail!("only the x2 spatial LTX latent upsampler is supported");
        }
        let blocks = int("num_blocks_per_stage").unwrap_or(4) as usize;
        let names: Vec<String> = store.index.names().map(str::to_owned).collect();
        let mut w = HashMap::new();
        for name in names {
            let t = store.tensor(&name, dev, false)?.to_dtype(DType::F32)?;
            w.insert(name, t);
        }
        Ok(Self { w, blocks })
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("latent upsampler lacks {name}")))
    }

    /// A 3x3x3 convolution with zero padding (PyTorch `Conv3d(padding=1)`),
    /// as a sum of 2D convolutions over the three time offsets.
    fn conv3d(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let weight = self.get(&format!("{prefix}.weight"))?;
        let bias = self.get(&format!("{prefix}.bias"))?;
        let (b, c, t, h, w) = x.dims5()?;
        let out = weight.dim(0)?;
        let zeros = Tensor::zeros((b, c, 1, h, w), x.dtype(), x.device())?;
        let padded = Tensor::cat(&[&zeros, x, &zeros], 2)?;
        let mut acc: Option<Tensor> = None;
        for k in 0..3 {
            let kernel = weight.narrow(2, k, 1)?.squeeze(2)?.contiguous()?;
            let frames = padded
                .narrow(2, k, t)?
                .permute((0, 2, 1, 3, 4))?
                .contiguous()?
                .reshape((b * t, c, h, w))?;
            let y = frames
                .conv2d(&kernel, 1, 1, 1, 1)?
                .reshape((b, t, out, h, w))?
                .permute((0, 2, 1, 3, 4))?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        acc.ok_or_else(|| candle_core::Error::Msg("empty kernel".into()))?
            .broadcast_add(&bias.reshape((1, out, 1, 1, 1))?)
    }

    /// GroupNorm with 32 groups over (channels / 32, time, height, width).
    fn group_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let g = x.reshape((b, 32, (c / 32) * t * h * w))?;
        let mean = g.mean_keepdim(2)?;
        let centered = g.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?;
        let normed = centered
            .broadcast_div(&(var + 1e-5)?.sqrt()?)?
            .reshape((b, c, t, h, w))?;
        normed
            .broadcast_mul(&self.get(&format!("{prefix}.weight"))?.reshape((1, c, 1, 1, 1))?)?
            .broadcast_add(&self.get(&format!("{prefix}.bias"))?.reshape((1, c, 1, 1, 1))?)
    }

    fn res_block(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = self.conv3d(x, &format!("{prefix}.conv1"))?;
        let h = candle_nn::ops::silu(&self.group_norm(&h, &format!("{prefix}.norm1"))?)?;
        let h = self.conv3d(&h, &format!("{prefix}.conv2"))?;
        let h = self.group_norm(&h, &format!("{prefix}.norm2"))?;
        candle_nn::ops::silu(&(h + x)?)
    }

    /// (1, 128, frames, height, width) un-normalized latent, to twice the
    /// height and width.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = self.conv3d(&latent.to_dtype(DType::F32)?, "initial_conv")?;
        x = candle_nn::ops::silu(&self.group_norm(&x, "initial_norm")?)?;
        for i in 0..self.blocks {
            x = self.res_block(&x, &format!("res_blocks.{i}"))?;
        }
        // Per frame: a 2D convolution to 4x the channels, then a 2x2 pixel
        // shuffle, `b (c p1 p2) h w -> b c (h p1) (w p2)`.
        let (b, c, t, h, w) = x.dims5()?;
        let frames = x.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((b * t, c, h, w))?;
        let up = frames
            .conv2d(&self.get("upsampler.0.weight")?.contiguous()?, 1, 1, 1, 1)?
            .broadcast_add(&self.get("upsampler.0.bias")?.reshape((1, 4 * c, 1, 1))?)?;
        let up = up
            .reshape((b * t, c, 2, 2, h, w))?
            .permute((0, 1, 4, 2, 5, 3))?
            .contiguous()?
            .reshape((b * t, c, 2 * h, 2 * w))?;
        x = up.reshape((b, t, c, 2 * h, 2 * w))?.permute((0, 2, 1, 3, 4))?.contiguous()?;
        for i in 0..self.blocks {
            x = self.res_block(&x, &format!("post_upsample_res_blocks.{i}"))?;
        }
        self.conv3d(&x, "final_conv")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_shuffle_matches_the_reference_layout() -> Result<()> {
        // Channel index = c * 4 + p1 * 2 + p2; output (h * 2 + p1, w * 2 + p2).
        let dev = Device::Cpu;
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let x = Tensor::from_vec(values, (1, 8, 1, 1), &dev)?;
        let (bt, c, h, w) = (1, 2, 1, 1);
        let y = x
            .reshape((bt, c, 2, 2, h, w))?
            .permute((0, 1, 4, 2, 5, 3))?
            .contiguous()?
            .reshape((bt, c, 2 * h, 2 * w))?;
        assert_eq!(y.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2., 3., 4., 5., 6., 7.]);
        Ok(())
    }

    /// Against `tools/ltx/upsampler_reference.py` (the official modules, F32).
    #[test]
    #[ignore = "needs the LTX latent upsampler and its reference; NROB_LTX_UPSAMPLER, NROB_LTX_UPSAMPLER_REF"]
    fn upsampler_matches_reference() -> Result<()> {
        let (Some(path), Some(reference)) = (std::env::var_os("NROB_LTX_UPSAMPLER"), std::env::var_os("NROB_LTX_UPSAMPLER_REF")) else {
            return Ok(());
        };
        let dev = Device::Cpu;
        let up = Upsampler::load(Path::new(&path), &dev)?;
        // Read at full precision (the model store hands out BF16).
        let r = candle_core::safetensors::load(Path::new(&reference), &dev)?;
        let input = r.get("input").ok_or_else(|| candle_core::Error::Msg("no input".into()))?;
        let expected = r.get("output").ok_or_else(|| candle_core::Error::Msg("no output".into()))?;
        let actual = up.forward(input)?;
        assert_eq!(actual.dims(), expected.dims());
        let error = (&actual - expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt()
            / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        eprintln!("upsampler relative RMS error {error:e}");
        assert!(error < 1e-3, "relative error {error}");
        Ok(())
    }

    #[test]
    #[ignore = "needs the LTX latent upsampler; NROB_LTX_UPSAMPLER"]
    fn upsampler_doubles_height_and_width() -> Result<()> {
        let Some(path) = std::env::var_os("NROB_LTX_UPSAMPLER") else { return Ok(()) };
        let dev = Device::Cpu;
        let up = Upsampler::load(Path::new(&path), &dev)?;
        let x = Tensor::randn(0f32, 1., (1, 128, 2, 3, 4), &dev)?;
        let y = up.forward(&x)?;
        assert_eq!(y.dims5()?, (1, 128, 2, 6, 8));
        assert!(y.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
        Ok(())
    }
}

