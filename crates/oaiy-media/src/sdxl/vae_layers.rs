// Ported from the user-owned F2I plugin-diffusion VAE primitives.
//! F2I-owned VAE — variational autoencoder for the 16-channel latent
//! diffusion family (FLUX / Qwen-Image / SD3).
//!
//! Architecture (encode side):
//!   conv_in → [ResBlock × N, Downsample] × len(ch_mult) →
//!   ResBlock → AttnBlock → ResBlock → norm → swish → conv_out
//!
//! Decoder is the symmetric mirror: conv_in → ResBlock → AttnBlock →
//! ResBlock → [Upsample, ResBlock × N] × len(ch_mult) → norm → swish →
//! conv_out → 3-channel RGB.
//!
//! Reference: the published 16-ch latent VAE used by FLUX (and reused
//! by Qwen-Image and other 16-ch DiTs). The code below is a fresh port
//! — we use candle-nn's Conv2d/GroupNorm primitives but the network is
//! ours.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{conv2d, group_norm, Activation, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

#[derive(Debug, Clone)]
pub struct VaeConfig {
    /// Spatial size the VAE was trained at (informational only).
    pub resolution: usize,
    /// RGB input channels (always 3 for these VAEs).
    pub in_channels: usize,
    /// RGB output channels (always 3).
    pub out_channels: usize,
    /// Base channel count; each level scales by `ch_mult[i]`.
    pub base_channels: usize,
    /// Per-level channel multiplier. `len() - 1` downsamples in encode.
    pub ch_mult: Vec<usize>,
    /// ResBlocks per resolution.
    pub num_res_blocks: usize,
    /// Latent channel count. 16 for the FLUX-family VAE.
    pub latent_channels: usize,
    /// Multiplicative factor applied to (encoded - shift) → latent.
    pub scale_factor: f64,
    /// Mean of the latent distribution; subtracted before scale.
    pub shift_factor: f64,
}

impl VaeConfig {
    /// Standard 16-channel latent VAE (FLUX dev/schnell). Qwen-Image
    /// uses these same dimensions; the trained weights differ but the
    /// architecture is shape-identical.
    pub fn flux_16ch() -> Self {
        Self {
            resolution: 256,
            in_channels: 3,
            out_channels: 3,
            base_channels: 128,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            latent_channels: 16,
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    /// Optional 1x1 projection used when in/out channel count differs.
    nin_shortcut: Option<Conv2d>,
}

impl ResBlock {
    fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(32, in_ch, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(in_ch, out_ch, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(32, out_ch, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(out_ch, out_ch, 3, conv_cfg, vb.pp("conv2"))?;
        // The 1x1 skip is only needed when channel counts differ;
        // otherwise we add the input directly to the residual stream.
        let nin_shortcut = if in_ch != out_ch {
            Some(conv2d(
                in_ch,
                out_ch,
                1,
                Conv2dConfig::default(),
                vb.pp("nin_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
        })
    }
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.norm1)?
            .apply(&Activation::Swish)?
            .apply(&self.conv1)?
            .apply(&self.norm2)?
            .apply(&Activation::Swish)?
            .apply(&self.conv2)?;
        let skip = match &self.nin_shortcut {
            Some(s) => xs.apply(s)?,
            None => xs.clone(),
        };
        h + skip
    }
}

/// Single-head self-attention block over an HxW grid. Used at the
/// bottleneck of both encoder and decoder.
#[derive(Debug, Clone)]
struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl AttnBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg1x1 = Conv2dConfig::default();
        let norm = group_norm(32, channels, 1e-6, vb.pp("norm"))?;
        let q = conv2d(channels, channels, 1, cfg1x1, vb.pp("q"))?;
        let k = conv2d(channels, channels, 1, cfg1x1, vb.pp("k"))?;
        let v = conv2d(channels, channels, 1, cfg1x1, vb.pp("v"))?;
        let proj_out = conv2d(channels, channels, 1, cfg1x1, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            q,
            k,
            v,
            proj_out,
        })
    }
}

impl Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.norm)?;
        let q = h.apply(&self.q)?;
        let k = h.apply(&self.k)?;
        let v = h.apply(&self.v)?;
        let (b, c, hh, w) = q.dims4()?;
        let q = q.flatten_from(2)?.transpose(1, 2)?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.transpose(1, 2)?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.transpose(1, 2)?.unsqueeze(1)?;
        let out = crate::math::attention(&q, &k, &v, 0)?
            .squeeze(1)?
            .transpose(1, 2)?
            .reshape((b, c, hh, w))?;
        let projected = out.apply(&self.proj_out)?;
        xs + projected
    }
}

#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        // Asymmetric padding (1 right, 1 bottom) — matches the
        // black-forest-labs / FLUX reference. Without it the spatial
        // dims drop by an extra row/col after the stride-2 conv.
        let cfg = Conv2dConfig {
            stride: 2,
            padding: 0,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pad (left=0, right=1, top=0, bottom=1) so the stride-2 conv
        // halves dims cleanly. candle's pad takes (left, right, top, bottom).
        let x = xs.pad_with_zeros(D::Minus1, 0, 1)?;
        let x = x.pad_with_zeros(D::Minus2, 0, 1)?;
        x.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Nearest-neighbour 2x upsample then 3x3 refine — same recipe
        // as Stable Diffusion 1/2/XL/3 and FLUX.
        let (_, _, h, w) = xs.dims4()?;
        let upsampled = xs.upsample_nearest2d(h * 2, w * 2)?;
        upsampled.apply(&self.conv)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    down: Vec<DownLevel>,
    mid_block_1: ResBlock,
    mid_attn: AttnBlock,
    mid_block_2: ResBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

#[derive(Debug, Clone)]
struct DownLevel {
    blocks: Vec<ResBlock>,
    downsample: Option<Downsample>,
}

impl Encoder {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(
            cfg.in_channels,
            cfg.base_channels,
            3,
            conv_cfg,
            vb.pp("conv_in"),
        )?;

        let mut down = Vec::with_capacity(cfg.ch_mult.len());
        let mut block_in = cfg.base_channels;
        let in_ch_mult = std::iter::once(1)
            .chain(cfg.ch_mult.iter().copied())
            .collect::<Vec<_>>();
        for (i_level, &mult) in cfg.ch_mult.iter().enumerate() {
            let vb_lv = vb.pp(format!("down.{i_level}"));
            block_in = cfg.base_channels * in_ch_mult[i_level];
            let block_out = cfg.base_channels * mult;
            let mut blocks = Vec::with_capacity(cfg.num_res_blocks);
            let vb_blocks = vb_lv.pp("block");
            for i_block in 0..cfg.num_res_blocks {
                let b = ResBlock::new(block_in, block_out, vb_blocks.pp(i_block))?;
                blocks.push(b);
                block_in = block_out;
            }
            let downsample = if i_level != cfg.ch_mult.len() - 1 {
                Some(Downsample::new(block_in, vb_lv.pp("downsample"))?)
            } else {
                None
            };
            down.push(DownLevel { blocks, downsample });
        }

        let mid_block_1 = ResBlock::new(block_in, block_in, vb.pp("mid.block_1"))?;
        let mid_attn = AttnBlock::new(block_in, vb.pp("mid.attn_1"))?;
        let mid_block_2 = ResBlock::new(block_in, block_in, vb.pp("mid.block_2"))?;
        let norm_out = group_norm(32, block_in, 1e-6, vb.pp("norm_out"))?;
        // Encoder produces 2 * latent_channels (mean + logvar) for the
        // diagonal Gaussian sample.
        let conv_out = conv2d(
            block_in,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down,
            mid_block_1,
            mid_attn,
            mid_block_2,
            norm_out,
            conv_out,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?;
        for level in &self.down {
            for block in &level.blocks {
                h = h.apply(block)?;
            }
            if let Some(ds) = &level.downsample {
                h = h.apply(ds)?;
            }
        }
        h.apply(&self.mid_block_1)?
            .apply(&self.mid_attn)?
            .apply(&self.mid_block_2)?
            .apply(&self.norm_out)?
            .apply(&Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block_1: ResBlock,
    mid_attn: AttnBlock,
    mid_block_2: ResBlock,
    up: Vec<UpLevel>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

#[derive(Debug, Clone)]
struct UpLevel {
    blocks: Vec<ResBlock>,
    upsample: Option<Upsample>,
}

impl Decoder {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut block_in = cfg.base_channels * cfg.ch_mult.last().copied().unwrap_or(1);
        let conv_in = conv2d(cfg.latent_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;
        let mid_block_1 = ResBlock::new(block_in, block_in, vb.pp("mid.block_1"))?;
        let mid_attn = AttnBlock::new(block_in, vb.pp("mid.attn_1"))?;
        let mid_block_2 = ResBlock::new(block_in, block_in, vb.pp("mid.block_2"))?;

        let mut up = Vec::with_capacity(cfg.ch_mult.len());
        // Decoder iterates ch_mult in reverse; level index in the
        // checkpoint is the *encoder-style* index (so ch_mult.len()-1
        // is the bottleneck level).
        for (i_rev, &mult) in cfg.ch_mult.iter().enumerate().rev() {
            let vb_lv = vb.pp(format!("up.{i_rev}"));
            let block_out = cfg.base_channels * mult;
            let mut blocks = Vec::with_capacity(cfg.num_res_blocks + 1);
            let vb_blocks = vb_lv.pp("block");
            // Decoder has num_res_blocks+1 ResBlocks per level (one extra
            // post-skip-merge). We don't have skips here — straight
            // through — but the count matches the reference.
            for i_block in 0..(cfg.num_res_blocks + 1) {
                let b = ResBlock::new(block_in, block_out, vb_blocks.pp(i_block))?;
                blocks.push(b);
                block_in = block_out;
            }
            let upsample = if i_rev != 0 {
                Some(Upsample::new(block_in, vb_lv.pp("upsample"))?)
            } else {
                None
            };
            up.push(UpLevel { blocks, upsample });
        }

        let norm_out = group_norm(32, block_in, 1e-6, vb.pp("norm_out"))?;
        let conv_out = conv2d(block_in, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn,
            mid_block_2,
            up,
            norm_out,
            conv_out,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.conv_in)?
            .apply(&self.mid_block_1)?
            .apply(&self.mid_attn)?
            .apply(&self.mid_block_2)?;
        let mut h = h;
        for level in &self.up {
            for block in &level.blocks {
                h = h.apply(block)?;
            }
            if let Some(us) = &level.upsample {
                h = h.apply(us)?;
            }
        }
        h.apply(&self.norm_out)?
            .apply(&Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ---------------------------------------------------------------------------
// Top-level VAE wrapper — owns encoder + decoder + scale/shift constants.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Vae {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub scale_factor: f64,
    pub shift_factor: f64,
    pub latent_channels: usize,
}

impl Vae {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            scale_factor: cfg.scale_factor,
            shift_factor: cfg.shift_factor,
            latent_channels: cfg.latent_channels,
        })
    }

    /// Encode an RGB image (b, 3, h, w) to a latent (b, latent_channels, h/8, w/8).
    /// The diagonal Gaussian is collapsed to its mean (deterministic).
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.encoder)?;
        // Encoder output is (b, 2*latent_channels, h/8, w/8); take the
        // mean half (channel 0..latent_channels).
        let mean = h.narrow(1, 0, self.latent_channels)?;
        (mean - self.shift_factor)? * self.scale_factor
    }

    /// Decode a latent (b, latent_channels, h, w) to RGB (b, 3, h*8, w*8).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let scaled = ((z / self.scale_factor)? + self.shift_factor)?;
        scaled.apply(&self.decoder)
    }
}
