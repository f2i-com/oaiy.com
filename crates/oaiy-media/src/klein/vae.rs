//! AutoencoderKLFlux2: 32-channel KL VAE, 2x2 channel patches, and
//! non-affine BatchNorm using the published running statistics.
use super::math;
use crate::{
    sdxl::vae_layers::{Decoder, Encoder, VaeConfig},
    weights::Weights,
};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};
use std::{collections::HashMap, path::Path};

pub struct Vae {
    decoder: Decoder,
    post: Option<Conv2d>,
    encoder: Option<Encoder>,
    quant: Option<Conv2d>,
    mean: Tensor,
    std: Tensor,
}
impl Vae {
    pub fn load(path: &Path, d: &Device, encode: bool) -> Result<Self> {
        // Published Flux2 VAE is F32. Keeping its group norms and decoder F32
        // also honors force_upcast; denoising remains BF16 on CUDA.
        let mut w = Weights::open(path)?;
        let mut tensors = HashMap::new();
        let original = w.has("decoder.mid.block_1.conv1.weight");
        for name in w.names() {
            if name.starts_with("decoder.")
                || name.starts_with("post_quant_conv.")
                || (encode && (name.starts_with("encoder.") || name.starts_with("quant_conv.")))
            {
                let mapped = if original {
                    name.clone()
                } else {
                    map_key(&name)
                };
                let mut t = w.tensor(&name, d, DType::F32)?;
                if mapped.contains(".mid.attn_1.") && t.rank() == 2 {
                    let (a, b) = t.dims2()?;
                    t = t.reshape((a, b, 1, 1))?;
                }
                tensors.insert(mapped, t);
            }
        }
        let mean = w
            .tensor("bn.running_mean", d, DType::F32)?
            .reshape((1, 128, 1, 1))?;
        let var = w
            .tensor("bn.running_var", d, DType::F32)?
            .reshape((1, 128, 1, 1))?;
        let std = (var + 1e-4)?.sqrt()?;
        let cfg = VaeConfig {
            resolution: 256,
            in_channels: 3,
            out_channels: 3,
            base_channels: 128,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            latent_channels: 32,
            scale_factor: 1.,
            shift_factor: 0.,
        };
        let vb = VarBuilder::from_tensors(tensors, DType::F32, d);
        let decoder = Decoder::new(&cfg, vb.pp("decoder"))?;
        let post = if w.has("post_quant_conv.weight") {
            Some(candle_nn::conv2d(
                32,
                32,
                1,
                Conv2dConfig::default(),
                vb.pp("post_quant_conv"),
            )?)
        } else {
            None
        };
        let encoder = if encode {
            Some(Encoder::new(&cfg, vb.pp("encoder"))?)
        } else {
            None
        };
        let quant = if encode && w.has("quant_conv.weight") {
            Some(candle_nn::conv2d(
                64,
                64,
                1,
                Conv2dConfig::default(),
                vb.pp("quant_conv"),
            )?)
        } else {
            None
        };
        Ok(Self {
            decoder,
            post,
            encoder,
            quant,
            mean,
            std,
        })
    }
    pub fn decode(&self, tokens: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let normalized = math::unpack(&tokens.to_dtype(DType::F32)?, h, w)?;
        let x = normalized
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)?;
        let mut x = math::unpatchify(&x)?;
        if let Some(post) = &self.post {
            x = post.forward(&x)?;
        }
        self.decoder.forward(&x)
    }
    pub fn encode(&self, rgb: &Tensor) -> Result<Tensor> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Flux2 encoder was not loaded".into()))?;
        let mut moments = encoder.forward(&rgb.to_dtype(DType::F32)?)?;
        if let Some(quant) = &self.quant {
            moments = quant.forward(&moments)?;
        }
        let patches = math::patchify(&moments.narrow(1, 0, 32)?)?;
        math::pack(
            &patches
                .broadcast_sub(&self.mean)?
                .broadcast_div(&self.std)?,
        )
    }
}
fn map_key(k: &str) -> String {
    let mut s = k
        .replace("conv_norm_out", "norm_out")
        .replace(".conv_shortcut.", ".nin_shortcut.");
    for side in ["encoder", "decoder"] {
        for (source, target) in [
            ("mid_block.resnets.0", "mid.block_1"),
            ("mid_block.resnets.1", "mid.block_2"),
            ("mid_block.attentions.0", "mid.attn_1"),
        ] {
            s = s.replace(&format!("{side}.{source}"), &format!("{side}.{target}"));
        }
    }
    for i in 0..4 {
        s = s.replace(
            &format!("encoder.down_blocks.{i}.resnets."),
            &format!("encoder.down.{i}.block."),
        );
        s = s.replace(
            &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
            &format!("encoder.down.{i}.downsample.conv"),
        );
        s = s.replace(
            &format!("decoder.up_blocks.{i}.resnets."),
            &format!("decoder.up.{}.block.", 3 - i),
        );
        s = s.replace(
            &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
            &format!("decoder.up.{}.upsample.conv", 3 - i),
        );
    }
    if s.contains(".mid.attn_1.") {
        s = s
            .replace(".group_norm.", ".norm.")
            .replace(".to_q.", ".q.")
            .replace(".to_k.", ".k.")
            .replace(".to_v.", ".v.")
            .replace(".to_out.0.", ".proj_out.");
    }
    s
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires OAIY_KLEIN_VAE; native CPU component smoke, no image artifact"]
    fn published_vae_loads_and_encodes_decodes_the_packed_latent_shape() -> Result<()> {
        let p = std::env::var("OAIY_KLEIN_VAE").map_err(candle_core::Error::wrap)?;
        let vae = Vae::load(Path::new(&p), &Device::Cpu, true)?;
        let input = Tensor::zeros((1, 3, 16, 16), DType::F32, &Device::Cpu)?;
        let tokens = vae.encode(&input)?;
        assert_eq!(tokens.dims(), [1, 1, 128]);
        let rgb = vae.decode(&tokens, 1, 1)?;
        assert_eq!(rgb.dims(), [1, 3, 16, 16]);
        assert!(rgb
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite()));
        Ok(())
    }
    #[test]
    fn diffusers_vae_levels_follow_the_reversed_decoder_order() {
        assert_eq!(
            map_key("decoder.up_blocks.0.resnets.2.conv_shortcut.weight"),
            "decoder.up.3.block.2.nin_shortcut.weight"
        );
        assert_eq!(
            map_key("encoder.mid_block.attentions.0.to_q.weight"),
            "encoder.mid.attn_1.q.weight"
        );
    }
}
