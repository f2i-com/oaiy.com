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
/// The decoder's shape as [`crate::sdxl_vae_wgpu`] takes it, and each packed channel's mean and deviation (the
/// published batch-norm statistics, 128 of them): what a decode on WebGPU needs of the file at `path` beside its
/// tensors. Whether the file is in the original layout (else diffusers': [`map_key`] its names).
#[cfg(feature = "webgpu")]
pub(crate) fn decoder_on_webgpu(path: &Path, gpu: ggml_rs_wgpu::WgpuBackend) -> Result<(crate::sdxl_vae_wgpu::WgpuSdxlVae, Vec<f32>, Vec<f32>)> {
    let mut w = Weights::open(path)?;
    let original = w.has("decoder.mid.block_1.conv1.weight");
    let cfg = crate::sdxl::config::VaeConfig { base_channels: 128, channel_mults: vec![1, 2, 4, 4], layers_per_block: 2, latent_channels: 32, scaling_factor: 1., in_channels: 3, out_channels: 3, norm_groups: 32 };
    let mean = w.tensor("bn.running_mean", &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let std: Vec<f32> = w.tensor("bn.running_var", &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| (v + 1e-4).sqrt()).collect();
    if mean.len() != 128 || std.len() != 128 {
        candle_core::bail!("a Flux2 VAE's batch-norm statistics are 128 values, not {} and {}", mean.len(), std.len());
    }
    let map = |key: &str| if original { key.to_owned() } else { map_key(key) };
    let vae = crate::sdxl_vae_wgpu::WgpuSdxlVae::load_mapped(&mut w, "", &cfg, gpu, &map, false)?;
    Ok((vae, mean, std))
}

/// A picture's latent from its tokens (`[h w, 128]`, a token a row) as [`Vae::decode`] unpacks them: each channel
/// times its deviation plus its mean, then a token's 128 as its 2x2 pixels of 32 (channel `4 c + 2 dy + dx` pixel
/// `(2 y + dy, 2 x + dx)`'s channel `c`): `[2 h * 2 w, 32]`, a pixel a row.
#[cfg(feature = "webgpu")]
pub(crate) fn unpack_rows(tokens: &[f32], h: usize, w: usize, mean: &[f32], std: &[f32]) -> Vec<f32> {
    let mut latent = vec![0f32; 4 * h * w * 32];
    for (at, token) in tokens.chunks_exact(128).take(h * w).enumerate() {
        let (y, x) = (at / w, at % w);
        for (p, v) in token.iter().enumerate() {
            let (c, dy, dx) = (p / 4, p % 4 / 2, p % 2);
            latent[((2 * y + dy) * 2 * w + 2 * x + dx) * 32 + c] = v * std[p] + mean[p];
        }
    }
    latent
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
    /// The host's unpacking for the WebGPU decoder is [`Vae::decode`]'s: its statistics, then its 2x2 patches.
    #[cfg(feature = "webgpu")]
    #[test]
    fn the_rows_unpacked_for_webgpu_are_the_tensors() -> Result<()> {
        let (h, w) = (3usize, 5usize);
        let tokens: Vec<f32> = (0..h * w * 128).map(|i| ((i * 37 % 101) as f32 - 50.) / 17.).collect();
        let mean: Vec<f32> = (0..128).map(|i| i as f32 * 0.01 - 0.3).collect();
        let std: Vec<f32> = (0..128).map(|i| 0.5 + i as f32 * 0.003).collect();
        let t = Tensor::from_vec(tokens.clone(), (1, h * w, 128), &Device::Cpu)?;
        let (m, s) = (Tensor::from_vec(mean.clone(), (1, 128, 1, 1), &Device::Cpu)?, Tensor::from_vec(std.clone(), (1, 128, 1, 1), &Device::Cpu)?);
        let want = math::unpatchify(&math::unpack(&t, h, w)?.broadcast_mul(&s)?.broadcast_add(&m)?)?;
        // (a pixel a row, as the decoder on WebGPU takes it)
        let want = want.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(unpack_rows(&tokens, h, w, &mean, &std), want);
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
