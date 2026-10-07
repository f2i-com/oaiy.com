// Ported from the user-owned F2I plugin-diffusion SDXL implementation.
//! SDXL UNet — the diffusion model.
//!
//! Layout (matching `sd_xl_base_1.0.safetensors` post-strip of
//! `model.diffusion_model.`):
//!
//! ```text
//!   time_embed.0          Linear(320  → 1280)
//!   time_embed.2          Linear(1280 → 1280)         (.1 is SiLU)
//!   label_emb.0.0         Linear(2816 → 1280)
//!   label_emb.0.2         Linear(1280 → 1280)         (.1 is SiLU)
//!
//!   input_blocks.0.0      Conv2d 3x3 (4 → 320)        — initial conv
//!   input_blocks.1, 2     ResnetBlock @ 320
//!   input_blocks.3.0      Downsample (Conv2d 3x3 s=2) 320 → 320
//!   input_blocks.4, 5     ResnetBlock @ 320→640 + SpatialTransformer (2 layers)
//!   input_blocks.6.0      Downsample 640 → 640
//!   input_blocks.7, 8     ResnetBlock @ 640→1280 + SpatialTransformer (10 layers)
//!
//!   middle_block.0        ResnetBlock @ 1280
//!   middle_block.1        SpatialTransformer (10 layers)
//!   middle_block.2        ResnetBlock @ 1280
//!
//!   output_blocks.0, 1, 2 ResnetBlock + SpatialTransformer (10 layers) @ 1280
//!                         (output_blocks.2 also has Upsample 1280 → 1280)
//!   output_blocks.3, 4, 5 ResnetBlock + SpatialTransformer (2 layers)  @ 640
//!                         (output_blocks.5 also has Upsample 640 → 640)
//!   output_blocks.6, 7, 8 ResnetBlock @ 320
//!
//!   out.0                 GroupNorm(32, 320)
//!   out.2                 Conv2d 3x3 (320 → 4)        (.1 is SiLU)
//! ```
//!
//! ## Forward
//!
//! `forward(x, timesteps, context, y_label)` where:
//! * `x`             — noisy latent `(B, 4, H/8, W/8)`
//! * `timesteps`     — `(B,)` integer-ish sigma values (cast to F32 sinusoidal)
//! * `context`       — `(B, 77, 2048)` from dual CLIP (L+G penultimate concat)
//! * `y_label`       — `(B, 2816)` ADM conditioning (pooled CLIP-G [1280]
//!                     + sinusoidal embeddings of (orig_h, orig_w, crop_top,
//!                     crop_left, target_h, target_w) [6 × 256 = 1536])
//!
//! Returns `(B, 4, H/8, W/8)` predicted noise.

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{
    conv2d, group_norm, layer_norm, linear, linear_no_bias, Activation, Conv2d, Conv2dConfig,
    GroupNorm, LayerNorm, Linear, VarBuilder,
};

use crate::sdxl::config::UNetConfig;

// ---------------------------------------------------------------------------
// Sinusoidal timestep embedding
// ---------------------------------------------------------------------------

/// Build the classic sinusoidal embedding `(B, dim)` for a vector of
/// timestep values. Same formula as the original DDPM / OpenAI UNet:
///
/// ```text
///   half = dim / 2
///   freq = exp(-log(max_period) * range(half) / half)
///   args = t[:, None] * freq[None, :]
///   emb  = cat([cos(args), sin(args)], dim=-1)
/// ```
///
/// Note `cos` first, then `sin` — this matches the OpenAI / Stability
/// AI convention (HF diffusers does sin-then-cos which is functionally
/// identical but the first half of the embed vector swaps).
pub fn timestep_embedding(
    t: &Tensor,
    dim: usize,
    max_period: f64,
    device: &Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| {
            let exponent = -(max_period.ln()) * (i as f64) / (half as f64);
            exponent.exp() as f32
        })
        .collect();
    let freqs = Tensor::from_vec(freqs, (half,), device)?.to_dtype(t.dtype())?;
    // t: (B,) -> (B, half)
    let args = t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

// ---------------------------------------------------------------------------
// ResnetBlock (the U-Net residual block that also takes the time embedding)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ResnetBlock {
    in_norm: GroupNorm,
    in_conv: Conv2d,
    emb_proj: Linear,
    out_norm: GroupNorm,
    out_conv: Conv2d,
    /// 1×1 skip-connection conv when input/output channel counts differ.
    skip: Option<Conv2d>,
}

impl ResnetBlock {
    fn load(in_ch: usize, out_ch: usize, emb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let in_norm = group_norm(32, in_ch, 1e-5, vb.pp("in_layers").pp("0"))?;
        let in_conv = conv2d(in_ch, out_ch, 3, conv_cfg, vb.pp("in_layers").pp("2"))?;
        let emb_proj = linear(emb_dim, out_ch, vb.pp("emb_layers").pp("1"))?;
        let out_norm = group_norm(32, out_ch, 1e-5, vb.pp("out_layers").pp("0"))?;
        let out_conv = conv2d(out_ch, out_ch, 3, conv_cfg, vb.pp("out_layers").pp("3"))?;
        let skip = if in_ch != out_ch {
            Some(conv2d(
                in_ch,
                out_ch,
                1,
                Conv2dConfig::default(),
                vb.pp("skip_connection"),
            )?)
        } else {
            None
        };
        Ok(Self {
            in_norm,
            in_conv,
            emb_proj,
            out_norm,
            out_conv,
            skip,
        })
    }

    fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let h = x
            .apply(&self.in_norm)?
            .apply(&Activation::Swish)?
            .apply(&self.in_conv)?;
        // emb: (B, emb_dim). Apply SiLU before projecting (matches the
        // CompVis "emb_layers = nn.Sequential(nn.SiLU(), linear)" layout).
        let e = emb
            .apply(&Activation::Swish)?
            .apply(&self.emb_proj)?
            .unsqueeze(2)?
            .unsqueeze(3)?;
        let h = h.broadcast_add(&e)?;
        let h = h
            .apply(&self.out_norm)?
            .apply(&Activation::Swish)?
            .apply(&self.out_conv)?;
        let skip = match &self.skip {
            Some(s) => x.apply(s)?,
            None => x.clone(),
        };
        h + skip
    }
}

// ---------------------------------------------------------------------------
// CrossAttention — the heart of the spatial transformer
// ---------------------------------------------------------------------------

/// Standard attention block used in both `attn1` (self-attention,
/// `context = None`) and `attn2` (cross-attention to text, `context =
/// Some(text_emb)`). The CompVis layout has no bias on Q/K/V and bias
/// only on `to_out.0`. Heads share a single projection per (q, k, v).
#[derive(Debug, Clone)]
struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn load(
        query_dim: usize,
        context_dim: usize,
        n_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = n_heads * head_dim;
        // CompVis SDXL attention has no bias on Q/K/V projections.
        let to_q = linear_no_bias(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear_no_bias(context_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear_no_bias(context_dim, inner_dim, vb.pp("to_v"))?;
        // to_out is wrapped in a list (`to_out.0` is the linear, `to_out.1`
        // is a dropout that's a no-op at inference).
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out").pp("0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            n_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let kv_src = context.unwrap_or(x);
        let (b, n_q, _) = x.dims3()?;
        let (b_k, n_kv, _) = kv_src.dims3()?;
        if b != b_k {
            candle_core::bail!("CrossAttention: batch mismatch {b} vs {b_k}");
        }
        let q = x.apply(&self.to_q)?;
        let k = kv_src.apply(&self.to_k)?;
        let v = kv_src.apply(&self.to_v)?;

        // (B, N, n_heads*head_dim) -> (B, n_heads, N, head_dim)
        let split = |t: &Tensor, n: usize| -> Result<Tensor> {
            t.reshape((b, n, self.n_heads, self.head_dim))?
                .transpose(1, 2)
        };
        let q = split(&q, n_q)?.contiguous()?;
        let k = split(&k, n_kv)?.contiguous()?;
        let v = split(&v, n_kv)?.contiguous()?;

        // Reuse bounded/flash attention instead of allocating the full spatial score matrix.
        let out = crate::math::attention(&q, &k, &v, 0)?;
        // (B, n_heads, N, head_dim) -> (B, N, n_heads*head_dim)
        let out = out
            .transpose(1, 2)?
            .reshape((b, n_q, self.n_heads * self.head_dim))?;
        out.apply(&self.to_out)
    }
}

// ---------------------------------------------------------------------------
// GEGLU feed-forward
// ---------------------------------------------------------------------------
//
// CompVis FF layout is `nn.Sequential(GEGLU(d, 4*d), nn.Dropout, Linear(4*d, d))`.
// GEGLU itself contains a single Linear from d to 2*(4*d) = 8*d, then
// splits along the channel dim into (gate, value) and returns
// `gate * GELU(value)`. So the on-disk shapes are:
//   * ff.net.0.proj.weight   (8*d, d)
//   * ff.net.0.proj.bias     (8*d,)
//   * ff.net.2.weight        (d, 4*d)
//   * ff.net.2.bias          (d,)
//
// Wait — SDXL uses `mult=4` BUT the shapes I see in the header are
// fc1: (5120, 640), fc2: (640, 2560) for the 640-channel level. That's
// inner = 4*640 = 2560, and GEGLU's fc1 outputs 2*inner = 5120. ✓

#[derive(Debug, Clone)]
struct GeGluFeedForward {
    fc1: Linear,
    fc2: Linear,
    inner_dim: usize,
}

impl GeGluFeedForward {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Multiplier hard-coded to 4 — every SDXL SpatialTransformer FF
        // uses this expansion.
        let inner_dim = 4 * dim;
        let fc1 = linear(dim, 2 * inner_dim, vb.pp("net").pp("0").pp("proj"))?;
        let fc2 = linear(inner_dim, dim, vb.pp("net").pp("2"))?;
        Ok(Self {
            fc1,
            fc2,
            inner_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.fc1)?;
        // Split last dim into (gate, value) of size inner_dim each.
        let gate = h.narrow(D::Minus1, 0, self.inner_dim)?;
        let value = h.narrow(D::Minus1, self.inner_dim, self.inner_dim)?;
        // Standard GELU on `value`, then elementwise multiply by gate.
        let act = value.apply(&Activation::Gelu)?;
        let h = (gate * act)?;
        h.apply(&self.fc2)
    }
}

// ---------------------------------------------------------------------------
// BasicTransformerBlock — one (self-attn + cross-attn + ff) unit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: CrossAttention,
    norm2: LayerNorm,
    attn2: CrossAttention,
    norm3: LayerNorm,
    ff: GeGluFeedForward,
}

impl BasicTransformerBlock {
    fn load(
        dim: usize,
        context_dim: usize,
        n_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn1 = CrossAttention::load(dim, dim, n_heads, head_dim, vb.pp("attn1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let attn2 = CrossAttention::load(dim, context_dim, n_heads, head_dim, vb.pp("attn2"))?;
        let norm3 = layer_norm(dim, 1e-5, vb.pp("norm3"))?;
        let ff = GeGluFeedForward::load(dim, vb.pp("ff"))?;
        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        // attn1: self-attention (context = None → uses x)
        let h = (x + self.attn1.forward(&x.apply(&self.norm1)?, None)?)?;
        // attn2: cross-attention to text context
        let h = (&h + self.attn2.forward(&h.apply(&self.norm2)?, Some(context))?)?;
        // FF: GEGLU
        &h + self.ff.forward(&h.apply(&self.norm3)?)?
    }
}

// ---------------------------------------------------------------------------
// SpatialTransformer — wraps N transformer blocks for a 2D feature map
// ---------------------------------------------------------------------------

/// Operates on a `(B, C, H, W)` feature map: reshape → Linear proj_in →
/// N transformer blocks (each consuming text context) → Linear proj_out
/// → reshape back. The +residual is on the OUTSIDE — the caller is
/// expected to add `x` after this returns (which the encoder/decoder
/// blocks do via "h + spatial_transformer(h)").
///
/// SDXL uses `use_linear_projection=True` so proj_in/proj_out are
/// Linear (2D weight), not 1×1 Conv (4D weight). The GroupNorm on the
/// outside is the typical pre-norm.
#[derive(Debug, Clone)]
struct SpatialTransformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<BasicTransformerBlock>,
    proj_out: Linear,
}

impl SpatialTransformer {
    fn load(
        channels: usize,
        n_heads: usize,
        head_dim: usize,
        n_layers: usize,
        context_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = n_heads * head_dim;
        let norm = group_norm(32, channels, 1e-6, vb.pp("norm"))?;
        let proj_in = linear(channels, inner_dim, vb.pp("proj_in"))?;
        let mut blocks = Vec::with_capacity(n_layers);
        let tb_vb = vb.pp("transformer_blocks");
        for i in 0..n_layers {
            blocks.push(BasicTransformerBlock::load(
                inner_dim,
                context_dim,
                n_heads,
                head_dim,
                tb_vb.pp(i.to_string()),
            )?);
        }
        let proj_out = linear(inner_dim, channels, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            proj_in,
            blocks,
            proj_out,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let residual = x.clone();
        let h_norm = x.apply(&self.norm)?;
        // (B, C, H, W) -> (B, H*W, C)
        let flat = h_norm.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let mut t = flat.apply(&self.proj_in)?;
        for block in &self.blocks {
            t = block.forward(&t, context)?;
        }
        let t = t.apply(&self.proj_out)?;
        // (B, H*W, C) -> (B, C, H, W)
        let t = t.transpose(1, 2)?.reshape((b, c, h, w))?;
        t + residual
    }
}

// ---------------------------------------------------------------------------
// Downsample (Conv2d 3x3 stride=2) and Upsample (nearest + Conv2d 3x3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        // Note: the on-disk weight name path for the UNet downsample is
        // `.op.{weight,bias}` (matching CompVis's `nn.Conv2d` inside
        // `Downsample(use_conv=True)` wrapper).
        let conv = conv2d(channels, channels, 3, cfg, vb.pp("op"))?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        x.upsample_nearest2d(h * 2, w * 2)?.apply(&self.conv)
    }
}

// ---------------------------------------------------------------------------
// Block variants — the per-position sequence in input_blocks / output_blocks
// ---------------------------------------------------------------------------

/// Each input/output block is composed of 1-3 modules whose order on
/// disk depends on the block index. We model it as a list of typed
/// modules so the forward loop just iterates and dispatches.
#[derive(Debug, Clone)]
enum BlockOp {
    /// Initial conv at `input_blocks.0.0` — 4 → 320.
    InitialConv(Conv2d),
    /// A ResnetBlock — always at offset `.0` in input_blocks 1+ and
    /// output_blocks.
    Resnet(ResnetBlock),
    /// A SpatialTransformer at offset `.1` — present iff
    /// `transformer_layers > 0` for this resolution.
    Spatial(SpatialTransformer),
    /// Downsample at offset `.0` for input_blocks at downsample positions
    /// (3, 6 in SDXL).
    Down(Downsample),
    /// Upsample at the end of output_blocks 2 and 5 (= the LAST
    /// output_block at each non-final resolution).
    Up(Upsample),
}

impl BlockOp {
    /// Apply the op. ResnetBlock needs `emb`; SpatialTransformer needs
    /// `context`; others ignore both. Returns the new tensor.
    fn forward(&self, h: &Tensor, emb: &Tensor, context: &Tensor) -> Result<Tensor> {
        match self {
            BlockOp::InitialConv(c) => h.apply(c),
            BlockOp::Resnet(r) => r.forward(h, emb),
            BlockOp::Spatial(s) => s.forward(h, context),
            BlockOp::Down(d) => d.forward(h),
            BlockOp::Up(u) => u.forward(h),
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    ops: Vec<BlockOp>,
}

impl Block {
    fn forward(&self, h: &Tensor, emb: &Tensor, context: &Tensor) -> Result<Tensor> {
        let mut h = h.clone();
        for op in &self.ops {
            h = op.forward(&h, emb, context)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Top-level UNet
// ---------------------------------------------------------------------------

/// SDXL UNet. Hand-ported, no candle-transformers dependency.
#[derive(Debug, Clone)]
pub struct UNet2DConditionModel {
    cfg: UNetConfig,
    time_embed_0: Linear,
    time_embed_2: Linear,
    label_emb_0_0: Linear,
    label_emb_0_2: Linear,
    input_blocks: Vec<Block>,
    middle_block: Block,
    output_blocks: Vec<Block>,
    out_norm: GroupNorm,
    out_conv: Conv2d,
}

impl UNet2DConditionModel {
    /// Load from a VarBuilder whose prefix is the stripped
    /// `model.diffusion_model.*` namespace (i.e. the splitter output).
    pub fn load(cfg: &UNetConfig, vb: VarBuilder) -> Result<Self> {
        let time_dim = cfg.time_embed_dim;
        // Time embed MLP: 320 → 1280 → 1280. Input is the sinusoidal
        // embedding of dim = block_out_channels[0] = 320.
        let time_in_dim = cfg.block_out_channels[0];
        let time_embed_0 = linear(time_in_dim, time_dim, vb.pp("time_embed").pp("0"))?;
        let time_embed_2 = linear(time_dim, time_dim, vb.pp("time_embed").pp("2"))?;
        // Label/ADM embed: 2816 → 1280 → 1280. Note the doubled prefix
        // `label_emb.0.{0,2}` (CompVis wraps it in nn.Sequential of
        // nn.Sequential).
        let label_emb_0_0 = linear(
            cfg.label_emb_in_dim,
            time_dim,
            vb.pp("label_emb").pp("0").pp("0"),
        )?;
        let label_emb_0_2 = linear(time_dim, time_dim, vb.pp("label_emb").pp("0").pp("2"))?;

        // ----- input_blocks -----
        let input_blocks = build_input_blocks(cfg, vb.pp("input_blocks"))?;

        // ----- middle_block: ResnetBlock @ 1280, SpatialTransformer @ 1280, ResnetBlock @ 1280
        let mid_ch = *cfg.block_out_channels.last().unwrap();
        let middle_block = {
            let mb = vb.pp("middle_block");
            let mid_n_heads = mid_ch / cfg.attention_head_dim;
            let mid_layers = *cfg.transformer_layers_per_block.last().unwrap();
            let ops = vec![
                BlockOp::Resnet(ResnetBlock::load(mid_ch, mid_ch, time_dim, mb.pp("0"))?),
                BlockOp::Spatial(SpatialTransformer::load(
                    mid_ch,
                    mid_n_heads,
                    cfg.attention_head_dim,
                    mid_layers,
                    cfg.cross_attention_dim,
                    mb.pp("1"),
                )?),
                BlockOp::Resnet(ResnetBlock::load(mid_ch, mid_ch, time_dim, mb.pp("2"))?),
            ];
            Block { ops }
        };

        // ----- output_blocks -----
        let output_blocks = build_output_blocks(cfg, vb.pp("output_blocks"))?;

        // ----- output projection -----
        let final_ch = cfg.block_out_channels[0];
        let out_norm = group_norm(cfg.norm_groups, final_ch, 1e-5, vb.pp("out").pp("0"))?;
        let out_conv = conv2d(
            final_ch,
            cfg.out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("out").pp("2"),
        )?;

        Ok(Self {
            cfg: cfg.clone(),
            time_embed_0,
            time_embed_2,
            label_emb_0_0,
            label_emb_0_2,
            input_blocks,
            middle_block,
            output_blocks,
            out_norm,
            out_conv,
        })
    }

    /// Forward pass.
    ///
    /// `timesteps`: `(B,)` integer-like float tensor (cast happens inside).
    /// `context`:   `(B, T, cross_attention_dim)` — 2048 for stock SDXL.
    /// `y_label`:   `(B, label_emb_in_dim)` — 2816 for stock SDXL.
    pub fn forward(
        &self,
        x: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        y_label: &Tensor,
    ) -> Result<Tensor> {
        self.forward_traced(x, timesteps, context, y_label, None)
    }

    /// [`Self::forward`], with each block's output pushed to `trace`: the input blocks', the middle's, then the
    /// output blocks' (what another backend's blocks are held against).
    pub fn forward_traced(
        &self,
        x: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        y_label: &Tensor,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let mut keep = |h: &Tensor| {
            if let Some(t) = trace.as_deref_mut() {
                t.push(h.clone());
            }
        };
        // ---- timestep + label embeddings ----
        let t_in_dim = self.cfg.block_out_channels[0];
        let t =
            timestep_embedding(timesteps, t_in_dim, 10_000f64, x.device())?.to_dtype(x.dtype())?;
        let t_emb = t
            .apply(&self.time_embed_0)?
            .apply(&Activation::Swish)?
            .apply(&self.time_embed_2)?;
        let y_emb = y_label
            .to_dtype(x.dtype())?
            .apply(&self.label_emb_0_0)?
            .apply(&Activation::Swish)?
            .apply(&self.label_emb_0_2)?;
        let emb = (t_emb + y_emb)?;

        // ---- down path ----
        let mut h = x.clone();
        let mut skips: Vec<Tensor> = Vec::with_capacity(self.input_blocks.len());
        for block in &self.input_blocks {
            h = block.forward(&h, &emb, context)?;
            keep(&h);
            skips.push(h.clone());
        }

        // ---- middle ----
        h = self.middle_block.forward(&h, &emb, context)?;
        keep(&h);

        // ---- up path ----
        for block in &self.output_blocks {
            let skip = skips.pop().ok_or_else(|| {
                candle_core::Error::Msg(
                    "UNet up-path popped more skips than down-path pushed".into(),
                )
            })?;
            h = Tensor::cat(&[&h, &skip], 1)?;
            h = block.forward(&h, &emb, context)?;
            keep(&h);
        }
        if !skips.is_empty() {
            candle_core::bail!(
                "UNet skip-connection imbalance: {} unused after up-path",
                skips.len()
            );
        }

        // ---- output projection ----
        h.apply(&self.out_norm)?
            .apply(&Activation::Swish)?
            .apply(&self.out_conv)
    }

    pub fn config(&self) -> &UNetConfig {
        &self.cfg
    }
}

// ---------------------------------------------------------------------------
// Block builders — SDXL's specific topology
// ---------------------------------------------------------------------------

/// Build `input_blocks.0..8` for stock SDXL. Topology hard-coded to
/// match `sd_xl_base_1.0.safetensors`:
///
/// ```text
///   0 = [InitialConv(4→320)]
///   1 = [ResBlock 320→320]                 ← skip 1
///   2 = [ResBlock 320→320]                 ← skip 2
///   3 = [Downsample 320→320]               ← skip 3
///   4 = [ResBlock 320→640, Spatial(2)]     ← skip 4
///   5 = [ResBlock 640→640, Spatial(2)]     ← skip 5
///   6 = [Downsample 640→640]               ← skip 6
///   7 = [ResBlock 640→1280, Spatial(10)]   ← skip 7
///   8 = [ResBlock 1280→1280, Spatial(10)]  ← skip 8
/// ```
fn build_input_blocks(cfg: &UNetConfig, vb: VarBuilder) -> Result<Vec<Block>> {
    let time_dim = cfg.time_embed_dim;
    let cs = &cfg.block_out_channels; // [320, 640, 1280]
    let tlpb = &cfg.transformer_layers_per_block; // [0, 2, 10]
    let cad = cfg.cross_attention_dim;
    let head_dim = cfg.attention_head_dim;

    let mut blocks: Vec<Block> = Vec::with_capacity(9);

    // 0: initial conv 4 → cs[0]
    {
        let cfg_c = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(cfg.in_channels, cs[0], 3, cfg_c, vb.pp("0").pp("0"))?;
        blocks.push(Block {
            ops: vec![BlockOp::InitialConv(conv)],
        });
    }

    // For each of the 3 resolution levels:
    //   2 ResnetBlock(+SpatialTransformer if tlpb[i] > 0) blocks,
    //   then a Downsample (except at the deepest level).
    let mut block_idx = 1usize;
    let mut in_ch = cs[0];
    for (i, &out_ch) in cs.iter().enumerate() {
        for _ in 0..cfg.layers_per_block {
            let mut ops = vec![BlockOp::Resnet(ResnetBlock::load(
                in_ch,
                out_ch,
                time_dim,
                vb.pp(block_idx.to_string()).pp("0"),
            )?)];
            if tlpb[i] > 0 {
                let n_heads = out_ch / head_dim;
                ops.push(BlockOp::Spatial(SpatialTransformer::load(
                    out_ch,
                    n_heads,
                    head_dim,
                    tlpb[i],
                    cad,
                    vb.pp(block_idx.to_string()).pp("1"),
                )?));
            }
            blocks.push(Block { ops });
            block_idx += 1;
            in_ch = out_ch;
        }
        if i < cs.len() - 1 {
            // Downsample — single op at .0
            blocks.push(Block {
                ops: vec![BlockOp::Down(Downsample::load(
                    out_ch,
                    vb.pp(block_idx.to_string()).pp("0"),
                )?)],
            });
            block_idx += 1;
        }
    }

    Ok(blocks)
}

/// Build `output_blocks.0..8` for stock SDXL.
///
/// Output blocks mirror the input blocks, but in REVERSE order. Each
/// output block receives the previous up-path activation concatenated
/// with the matching skip from the down path. The skip's channel count
/// is included in the ResnetBlock's `in_ch` calculation.
///
/// Resolution levels are walked deepest→shallowest:
///
/// ```text
///   0 = [ResBlock 1280+1280→1280,    Spatial(10)]
///   1 = [ResBlock 1280+1280→1280,    Spatial(10)]
///   2 = [ResBlock 1280+640 →1280,    Spatial(10),  Upsample 1280→1280]
///   3 = [ResBlock 1280+1280→640,     Spatial(2)]
///   4 = [ResBlock  640+640 →640,     Spatial(2)]
///   5 = [ResBlock  640+320 →640,     Spatial(2),   Upsample 640→640]
///   6 = [ResBlock  640+640 →320]
///   7 = [ResBlock  320+320 →320]
///   8 = [ResBlock  320+320 →320]
/// ```
///
/// The skip channels per output_block index come from iterating the
/// input skips in reverse — see the helper below for the explicit table.
fn build_output_blocks(cfg: &UNetConfig, vb: VarBuilder) -> Result<Vec<Block>> {
    let time_dim = cfg.time_embed_dim;
    let cs = &cfg.block_out_channels; // [320, 640, 1280]
    let tlpb = &cfg.transformer_layers_per_block; // [0, 2, 10]
    let cad = cfg.cross_attention_dim;
    let head_dim = cfg.attention_head_dim;

    // Pre-compute the per-output-block channel topology. SDXL's pattern
    // is fixed at layers_per_block=2 → 3 ResBlocks per resolution
    // (the +1 captures the post-skip-merge block).
    let skip_chs = collect_input_skip_channels(cfg);
    let mut skip_chs_rev = skip_chs.clone();
    skip_chs_rev.reverse();

    let mut blocks: Vec<Block> = Vec::new();
    let mut block_idx = 0usize;
    let n_levels = cs.len(); // 3
                             // Walk levels deepest → shallowest. Each level has
                             // `layers_per_block + 1` = 3 ResnetBlocks (the extra one handles
                             // the skip from the level boundary).
    for (lev_rev, &out_ch) in cs.iter().rev().enumerate() {
        let i = n_levels - 1 - lev_rev;
        let n_blocks_here = cfg.layers_per_block + 1;
        let mut prev_ch_in = if i == n_levels - 1 {
            // Topmost-resolution (deepest) up-block: input from the
            // middle block, which has cs.last() channels.
            *cs.last().unwrap()
        } else {
            // For shallower levels the first up-block receives the
            // upsampled output from the level below, with the deeper
            // level's out_ch.
            cs[i + 1]
        };
        for j in 0..n_blocks_here {
            let skip_ch = skip_chs_rev[block_idx];
            let in_ch = prev_ch_in + skip_ch;
            let mut ops = vec![BlockOp::Resnet(ResnetBlock::load(
                in_ch,
                out_ch,
                time_dim,
                vb.pp(block_idx.to_string()).pp("0"),
            )?)];
            if tlpb[i] > 0 {
                let n_heads = out_ch / head_dim;
                ops.push(BlockOp::Spatial(SpatialTransformer::load(
                    out_ch,
                    n_heads,
                    head_dim,
                    tlpb[i],
                    cad,
                    vb.pp(block_idx.to_string()).pp("1"),
                )?));
            }
            // Upsample on the LAST block of every level except the
            // shallowest (index 0). The upsample sub-module key is at
            // a different `.X` depending on whether the SpatialTransformer
            // was present.
            if j == n_blocks_here - 1 && i != 0 {
                // Upsample lives at index 2 if Spatial is present,
                // else index 1.
                let up_pos = if tlpb[i] > 0 { "2" } else { "1" };
                ops.push(BlockOp::Up(Upsample::load(
                    out_ch,
                    vb.pp(block_idx.to_string()).pp(up_pos),
                )?));
            }
            blocks.push(Block { ops });
            block_idx += 1;
            prev_ch_in = out_ch;
        }
    }
    Ok(blocks)
}

/// Per-input-block output channel count = the channel count that gets
/// pushed onto the skip stack after each input_block. Needed by the
/// output-block builder to compute correct in-channel counts after
/// cat(h, skip).
fn collect_input_skip_channels(cfg: &UNetConfig) -> Vec<usize> {
    let cs = &cfg.block_out_channels;
    let mut out = Vec::with_capacity(9);
    out.push(cs[0]); // block 0 (initial conv)
    let mut cur = cs[0];
    for (i, &out_ch) in cs.iter().enumerate() {
        for _ in 0..cfg.layers_per_block {
            cur = out_ch;
            out.push(cur);
        }
        if i < cs.len() - 1 {
            out.push(cur);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn skip_channels_match_sdxl_topology() {
        let cfg = UNetConfig::sdxl_1_0();
        let skips = collect_input_skip_channels(&cfg);
        // 1 initial + 2 res + 1 down + 2 res + 1 down + 2 res = 9
        assert_eq!(skips, vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280]);
    }

    #[test]
    fn timestep_embedding_shape() {
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![0.0f32, 100.0, 999.0], (3usize,), &dev).unwrap();
        let e = timestep_embedding(&t, 320, 10_000.0, &dev).unwrap();
        assert_eq!(e.dims(), &[3, 320]);
        // At t=0, cos(0)=1, sin(0)=0 → first half all 1s, second half all 0s.
        let row0: Vec<f32> = e.i(0).unwrap().to_vec1().unwrap();
        assert!((row0[0] - 1.0).abs() < 1e-5);
        assert!(row0[160].abs() < 1e-5);
    }
}
