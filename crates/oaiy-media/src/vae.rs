//! Image specialization of the released Qwen Image 2.1 RGBA VAE encoder/decoder.
//! Temporal upsampling skips time_conv for the first (only) frame.
use crate::{math, weights::Weights};
use candle_core::{DType, Device, Result, Tensor};
use std::{collections::HashMap, path::Path};

pub struct Vae {
    tensors: HashMap<String, Tensor>,
    mean: Tensor,
    std: Tensor,
}
impl Vae {
    pub fn load(root: &Path, dev: &Device, dtype: DType) -> Result<Self> {
        Self::load_part(root, dev, dtype, false)
    }
    pub fn load_encoder(root: &Path, dev: &Device, dtype: DType) -> Result<Self> {
        Self::load_part(root, dev, dtype, true)
    }
    fn load_part(root: &Path, dev: &Device, dtype: DType, encode: bool) -> Result<Self> {
        let mut w = Weights::open(&root.join("vae"))?;
        let mut tensors = HashMap::new();
        for name in w.names() {
            if (name.starts_with(if encode { "encoder." } else { "decoder." })
                || name.starts_with(if encode {
                    "quant_conv."
                } else {
                    "post_quant_conv."
                }))
                && !name.contains("time_conv")
            {
                tensors.insert(name.clone(), w.tensor(&name, dev, dtype)?);
            }
        }
        let config = oaiy_engine::json::Json::parse(&std::fs::read(root.join("vae/config.json"))?)
            .map_err(candle_core::Error::wrap)?;
        let load = |key: &str| -> Result<Tensor> {
            let values = config
                .get(key)
                .and_then(|x| x.as_array())
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {key}")))?;
            let data = values
                .iter()
                .map(|x| {
                    x.as_f64()
                        .map(|n| n as f32)
                        .ok_or_else(|| candle_core::Error::Msg(format!("invalid {key}")))
                })
                .collect::<Result<Vec<_>>>()?;
            Tensor::from_vec(data, (1, 64, 1, 1), dev)?.to_dtype(dtype)
        };
        Ok(Self {
            tensors,
            mean: load("latents_mean")?,
            std: load("latents_std")?,
        })
    }
    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing VAE tensor {name}")))
    }
    fn conv(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.tensor(&format!("{name}.weight"))?;
        let b = self.tensor(&format!("{name}.bias"))?;
        x.conv2d(w, w.dim(2)? / 2, 1, 1, 1)?
            .broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)
    }
    fn norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let f = x.to_dtype(DType::F32)?;
        let c = x.dim(1)?;
        let norm = f
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f32::MAX as f64)?;
        let f = f.broadcast_div(&norm)?.to_dtype(x.dtype())?;
        (f * (c as f64).sqrt())?.broadcast_mul(
            &self
                .tensor(&format!("{name}.gamma"))?
                .reshape((1, c, 1, 1))?,
        )
    }
    fn residual(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let shortcut = if self
            .tensors
            .contains_key(&format!("{p}.conv_shortcut.weight"))
        {
            self.conv(x, &format!("{p}.conv_shortcut"))?
        } else {
            x.clone()
        };
        let h = self.conv(
            &candle_nn::ops::silu(&self.norm(x, &format!("{p}.norm1"))?)?,
            &format!("{p}.conv1"),
        )?;
        self.conv(
            &candle_nn::ops::silu(&self.norm(&h, &format!("{p}.norm2"))?)?,
            &format!("{p}.conv2"),
        )? + shortcut
    }
    pub fn decode(&self, latent: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let mut x = latent
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, 64, h, w))?
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)?;
        x = self.conv(&x, "post_quant_conv")?;
        x = self.conv(&x, "decoder.conv_in")?;
        x = self.residual(&x, "decoder.mid_block.resnets.0")?;
        let p = "decoder.mid_block.attentions.0";
        let qkv = self.conv(
            &self.norm(&x, &format!("{p}.norm"))?,
            &format!("{p}.to_qkv"),
        )?;
        let c = x.dim(1)?;
        let reshape = |start| {
            qkv.narrow(1, start, c)?
                .reshape((1, 1, c, h * w))?
                .transpose(2, 3)?
                .contiguous()
        };
        let attended = math::attention(&reshape(0)?, &reshape(c)?, &reshape(2 * c)?, 0)?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((1, c, h, w))?;
        x = (x + self.conv(&attended, &format!("{p}.proj"))?)?;
        x = self.residual(&x, "decoder.mid_block.resnets.1")?;
        for i in 0..5 {
            let before = x.clone();
            for j in 0..3 {
                x = self.residual(&x, &format!("decoder.up_blocks.{i}.resnets.{j}"))?;
            }
            if i < 4 {
                let (_, cout, h, w) = x.dims4()?;
                x = self.conv(
                    &x.upsample_nearest2d(h * 2, w * 2)?,
                    &format!("decoder.up_blocks.{i}.upsampler.resample.1"),
                )?;
                let ft = if i < 3 { 2 } else { 1 };
                let cin = before.dim(1)?;
                let repeats = cout * ft * 4 / cin;
                // Exact repeat_interleave + pixel shuffle, retaining the last
                // temporal slot for the first frame (Wan residual shortcut).
                let skip = before
                    .unsqueeze(2)?
                    .broadcast_as((1, cin, repeats, h, w))?
                    .contiguous()?
                    .reshape(&[1, cout, ft, 2, 2, h, w])?
                    .narrow(2, ft - 1, 1)?
                    .squeeze(2)?
                    .permute((0, 1, 4, 2, 5, 3))?
                    .contiguous()?
                    .reshape((1, cout, h * 2, w * 2))?;
                x = (x + skip)?;
            }
        }
        self.conv(
            &candle_nn::ops::silu(&self.norm(&x, "decoder.norm_out")?)?,
            "decoder.conv_out",
        )
    }
    pub fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let mut x = self.conv(pixels, "encoder.conv_in")?;
        for i in 0..5 {
            let before = x.clone();
            for j in 0..2 {
                x = self.residual(&x, &format!("encoder.down_blocks.{i}.resnets.{j}"))?;
            }
            if i < 4 {
                let p = format!("encoder.down_blocks.{i}.downsampler.resample.1");
                let b = self.tensor(&format!("{p}.bias"))?;
                x = x
                    .pad_with_zeros(2, 0, 1)?
                    .pad_with_zeros(3, 0, 1)?
                    .conv2d(self.tensor(&format!("{p}.weight"))?, 0, 2, 1, 1)?
                    .broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)?;
            }
            x = (&x
                + average_shortcut(
                    &before,
                    x.dim(1)?,
                    if (1..4).contains(&i) { 2 } else { 1 },
                    if i < 4 { 2 } else { 1 },
                )?)?;
        }
        x = self.residual(&x, "encoder.mid_block.resnets.0")?;
        let p = "encoder.mid_block.attentions.0";
        let qkv = self.conv(
            &self.norm(&x, &format!("{p}.norm"))?,
            &format!("{p}.to_qkv"),
        )?;
        let (_, c, h, w) = x.dims4()?;
        let part = |start| {
            qkv.narrow(1, start, c)?
                .reshape((1, 1, c, h * w))?
                .transpose(2, 3)?
                .contiguous()
        };
        let attended = math::attention(&part(0)?, &part(c)?, &part(2 * c)?, 0)?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((1, c, h, w))?;
        x = (x + self.conv(&attended, &format!("{p}.proj"))?)?;
        x = self.residual(&x, "encoder.mid_block.resnets.1")?;
        x = self.conv(
            &candle_nn::ops::silu(&self.norm(&x, "encoder.norm_out")?)?,
            "encoder.conv_out",
        )?;
        self.conv(&x, "quant_conv")?
            .narrow(1, 0, 64)?
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?
            .reshape((1, 64, h * w))?
            .transpose(1, 2)?
            .contiguous()
    }
}

fn average_shortcut(x: &Tensor, cout: usize, ft: usize, fs: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x = x.unsqueeze(2)?.pad_with_zeros(2, ft - 1, 0)?;
    x.reshape(&[b, c, ft, h / fs, fs, w / fs, fs])?
        .permute(&[0, 1, 2, 4, 6, 3, 5][..])?
        .contiguous()?
        .reshape((b, cout, c * ft * fs * fs / cout, h / fs, w / fs))?
        .mean(2)
}
#[test]
fn first_frame_shortcut_keeps_zero_temporal_padding() -> Result<()> {
    let x = Tensor::new(&[[[[2f32, 4.], [6., 8.]]]], &Device::Cpu)?;
    assert_eq!(
        average_shortcut(&x, 1, 2, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        vec![2.5]
    );
    Ok(())
}
