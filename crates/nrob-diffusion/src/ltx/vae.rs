//! Native LTX convolutional video VAE, adapted from the local plugin-diffusion Rust implementation.
//! Uses positioned safetensors reads; no memory mapping or inference subprocess.
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

use crate::math::layer_norm;
fn layer_norm_no_params(x: &Tensor, _eps: f64) -> Result<Tensor> {
    layer_norm(x)
}

// ---------------------------------------------------------------------------
// PadMode / config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Zero,
    Reflect,
}

#[derive(Debug, Clone)]
pub enum EncoderBlockSpec {
    /// Stack of `num_layers` ResnetBlock3D, no resampling.
    ResX { num_layers: usize },
    /// `SpaceToDepthDownsample` stride (1, 2, 2). `multiplier` doubles
    /// channel count: `out = in * multiplier`. Conv inside outputs
    /// `out / stride_volume` channels which get pixel-shuffled up.
    CompressSpaceRes { multiplier: usize },
    /// `SpaceToDepthDownsample` stride (2, 1, 1).
    CompressTimeRes { multiplier: usize },
    /// `SpaceToDepthDownsample` stride (2, 2, 2).
    CompressAllRes { multiplier: usize },
}

#[derive(Debug, Clone)]
pub enum DecoderBlockSpec {
    /// Stack of `num_layers` ResnetBlock3D, no resampling.
    ResX { num_layers: usize },
    /// `DepthToSpaceUpsample` stride (2, 1, 1). `multiplier` divides
    /// channel count: `out = in * stride_volume / multiplier` after
    /// pixel-shuffle.
    CompressTime { multiplier: usize },
    /// `DepthToSpaceUpsample` stride (1, 2, 2).
    CompressSpace { multiplier: usize },
    /// `DepthToSpaceUpsample` stride (2, 2, 2).
    CompressAll { multiplier: usize },
}

#[derive(Debug, Clone)]
pub struct LtxVaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub base_channels: usize,
    /// Spec list in CONFIG order — same order as `decoder_blocks` in the
    /// JSON config. The constructor reverses internally to walk from
    /// latent → pixels.
    pub decoder_blocks: Vec<DecoderBlockSpec>,
    pub spatial_padding_mode: PadMode,
    pub causal_decoder: bool,
    /// Encoder-specific block list. Walked in order (not reversed).
    pub encoder_blocks: Vec<EncoderBlockSpec>,
    pub encoder_spatial_padding_mode: PadMode,
}

impl LtxVaeConfig {
    pub fn ltx_2_3_22b() -> Self {
        use DecoderBlockSpec as D;
        use EncoderBlockSpec as E;
        Self {
            latent_channels: 128,
            out_channels: 3,
            patch_size: 4,
            base_channels: 128,
            decoder_blocks: vec![
                D::ResX { num_layers: 4 },
                D::CompressSpace { multiplier: 2 },
                D::ResX { num_layers: 6 },
                D::CompressTime { multiplier: 2 },
                D::ResX { num_layers: 4 },
                D::CompressAll { multiplier: 1 },
                D::ResX { num_layers: 2 },
                D::CompressAll { multiplier: 2 },
                D::ResX { num_layers: 2 },
            ],
            spatial_padding_mode: PadMode::Zero,
            causal_decoder: false,
            encoder_blocks: vec![
                E::ResX { num_layers: 4 },
                E::CompressSpaceRes { multiplier: 2 },
                E::ResX { num_layers: 6 },
                E::CompressTimeRes { multiplier: 2 },
                E::ResX { num_layers: 4 },
                E::CompressAllRes { multiplier: 2 },
                E::ResX { num_layers: 2 },
                E::CompressAllRes { multiplier: 1 },
                E::ResX { num_layers: 2 },
            ],
            encoder_spatial_padding_mode: PadMode::Zero,
        }
    }
}

// ---------------------------------------------------------------------------
// Causal 3D conv via 2D slice accumulation.
// ---------------------------------------------------------------------------

/// Reflect-pad a single dim by `pad` on each side. Implements PyTorch's
/// `F.pad(..., mode='reflect')` semantics (boundary excluded).
fn reflect_pad_dim(x: &Tensor, dim: usize, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let n = x.dim(dim)?;
    if n < pad + 1 {
        candle_core::bail!(
            "reflect_pad_dim: dim {} length {} too small for pad {}",
            dim,
            n,
            pad
        );
    }
    // candle's `flip` is implemented via index_select internally and
    // requires a contiguous input. `narrow` along an inner dim leaves
    // non-contiguous strides, so materialize before flipping.
    let left = x.narrow(dim, 1, pad)?.contiguous()?.flip(&[dim])?;
    let right = x
        .narrow(dim, n - pad - 1, pad)?
        .contiguous()?
        .flip(&[dim])?;
    Tensor::cat(&[&left, x, &right], dim)
}

/// 5D-aware time padding. `causal=true` repeats the first frame
/// `time_kernel - 1` times on the left; `causal=false` repeats the
/// first AND last frames `(time_kernel - 1) / 2` times each.
fn time_pad_5d(x: &Tensor, time_kernel: usize, causal: bool) -> Result<Tensor> {
    if time_kernel <= 1 {
        return Ok(x.clone());
    }
    if causal {
        let pad_n = time_kernel - 1;
        let first = x.narrow(2, 0, 1)?;
        let pad = first.repeat((1, 1, pad_n, 1, 1))?;
        Tensor::cat(&[&pad, x], 2)
    } else {
        let pad_n = (time_kernel - 1) / 2;
        if pad_n == 0 {
            return Ok(x.clone());
        }
        let first = x.narrow(2, 0, 1)?.repeat((1, 1, pad_n, 1, 1))?;
        let t = x.dim(2)?;
        let last = x.narrow(2, t - 1, 1)?.repeat((1, 1, pad_n, 1, 1))?;
        Tensor::cat(&[&first, x, &last], 2)
    }
}

/// Run a causal 3D convolution by reducing to a sum of 2D convs.
///
/// `weight` shape `(C_out, C_in, kt, kh, kw)`. Stride along all axes
/// is 1 (the only stride used in the LTX VAE; `DepthToSpaceUpsample`
/// handles "stride 2" separately by pixel-shuffle).
pub fn causal_conv3d_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    causal: bool,
    spatial_pad_mode: PadMode,
) -> Result<Tensor> {
    let w_dims = weight.dims();
    if w_dims.len() != 5 {
        candle_core::bail!(
            "causal_conv3d_forward: weight must be 5D (C_out, C_in, kt, kh, kw), got {:?}",
            w_dims
        );
    }
    let c_out = w_dims[0];
    let kt = w_dims[2];
    let kh = w_dims[3];
    let kw = w_dims[4];
    if kt == 0 || kh == 0 || kw == 0 || kt % 2 == 0 || kh % 2 == 0 || kw % 2 == 0 {
        candle_core::bail!("LTX convolution requires nonzero odd kernel dimensions");
    }
    let h_pad = kh / 2;
    let w_pad = kw / 2;

    // Time padding (causal vs symmetric).
    let x_padded = time_pad_5d(x, kt, causal)?;

    // Spatial padding follows the checkpoint configuration.
    // With reflect we pre-pad and pass conv2d padding=0;
    // with zero we let conv2d apply the pad itself.
    let (x_padded, conv_h_pad, conv_w_pad) = match spatial_pad_mode {
        PadMode::Reflect => {
            let xh = reflect_pad_dim(&x_padded, 3, h_pad)?;
            let xhw = reflect_pad_dim(&xh, 4, w_pad)?;
            (xhw, 0usize, 0usize)
        }
        PadMode::Zero => (x_padded, h_pad, w_pad),
    };

    let dims = x_padded.dims5()?;
    let (b, c_in, t_padded, h_p, w_p) = (dims.0, dims.1, dims.2, dims.3, dims.4);
    let t_out = t_padded - kt + 1;
    let weight_dtype = weight.dtype();

    // candle's conv2d takes a single padding scalar (square pad). The
    // VAE only uses kernel=3, so kh==kw and h_pad==w_pad — no ragged
    // case to worry about.
    if conv_h_pad != conv_w_pad {
        candle_core::bail!(
            "causal_conv3d_forward: candle conv2d requires square padding, got h={} w={}",
            conv_h_pad,
            conv_w_pad
        );
    }
    let conv_pad = conv_h_pad;

    let mut acc: Option<Tensor> = None;
    for kt_idx in 0..kt {
        // Weight slice at time offset kt_idx → (C_out, C_in, kh, kw).
        // Materialize the 4D slice — `narrow(2, ..)` leaves non-contiguous
        // strides over the time axis and candle's conv2d (esp. on CUDA)
        // expects a row-major kernel.
        let w_slice = weight.narrow(2, kt_idx, 1)?.squeeze(2)?.contiguous()?;
        // Activation slice at the matching input time positions.
        let x_slice = x_padded.narrow(2, kt_idx, t_out)?;
        // Fold time into batch: (B, C_in, t_out, H, W) → (B*t_out, C_in, H, W).
        let x_2d = x_slice
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((b * t_out, c_in, h_p, w_p))?
            .to_dtype(weight_dtype)?;
        let y_2d = x_2d.conv2d(&w_slice, conv_pad, 1, 1, 1)?;
        let h_o = y_2d.dim(2)?;
        let w_o = y_2d.dim(3)?;
        // Unfold back to 5D and reorder axes to (B, C_out, t_out, H', W').
        let y_3d = y_2d
            .reshape((b, t_out, c_out, h_o, w_o))?
            .permute((0, 2, 1, 3, 4))?;
        acc = Some(match acc {
            Some(a) => a.broadcast_add(&y_3d)?,
            None => y_3d,
        });
    }
    let out =
        acc.ok_or_else(|| candle_core::Error::Msg("empty temporal convolution kernel".into()))?;

    if let Some(b_t) = bias {
        let b_b = b_t.reshape((1, c_out, 1, 1, 1))?;
        out.broadcast_add(&b_b)
    } else {
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CausalConv3d wrapper — owns weight + optional bias, exposes forward.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CausalConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    spatial_pad_mode: PadMode,
}

impl CausalConv3d {
    pub fn load(prefix: &str, spatial_pad_mode: PadMode, vb: VarBuilder) -> Result<Self> {
        // Wan2GP wraps the actual conv under `<prefix>.conv.{weight,bias}`.
        let p = vb.pp(prefix).pp("conv");
        // Weight shape is dynamic across blocks — we don't know it at
        // construction time, so use VarBuilder's no-shape get.
        let weight = p.get_unchecked("weight")?;
        let bias = p.get_unchecked("bias").ok();
        Ok(Self {
            weight,
            bias,
            spatial_pad_mode,
        })
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        causal_conv3d_forward(
            x,
            &self.weight,
            self.bias.as_ref(),
            causal,
            self.spatial_pad_mode,
        )
    }

    pub fn weight_dtype(&self) -> DType {
        self.weight.dtype()
    }

    pub fn weight_device(&self) -> Device {
        self.weight.device().clone()
    }
}

// 1×1×1 conv (used as the channel-changing shortcut in ResnetBlock3D).
#[derive(Debug, Clone)]
pub struct LinearConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearConv3d {
    pub fn load(prefix: &str, vb: VarBuilder) -> Result<Self> {
        let p = vb.pp(prefix);
        // For `make_linear_nd` (dims=3), the wrapper is a plain Conv3d
        // with kernel_size=1 — no `.conv` sub-module nesting.
        let weight = p.get_unchecked("weight")?;
        let bias = p.get_unchecked("bias").ok();
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1x1x1 → just a per-pixel matmul. Reshape to (B*T*H*W, C_in)
        // then matmul against the weight flattened to (C_out, C_in).
        let dims = x.dims5()?;
        let (b, c_in, t, h, w) = dims;
        let w_dims = self.weight.dims();
        let c_out = w_dims[0];
        // Weight (C_out, C_in, 1, 1, 1) → (C_out, C_in)
        let w_flat = self.weight.reshape((c_out, c_in))?;
        // Activation (B, C_in, T, H, W) → (B, T, H, W, C_in) → (-1, C_in)
        let x_flat = x
            .permute((0, 2, 3, 4, 1))?
            .contiguous()?
            .reshape((b * t * h * w, c_in))?
            .to_dtype(w_flat.dtype())?;
        let y = x_flat.matmul(&w_flat.t()?)?;
        let mut y = y
            .reshape((b, t, h, w, c_out))?
            .permute((0, 4, 1, 2, 3))?
            .contiguous()?;
        if let Some(b_t) = &self.bias {
            let b_b = b_t.reshape((1, c_out, 1, 1, 1))?;
            y = y.broadcast_add(&b_b)?;
        }
        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// PixelNorm — parameterless RMS norm along the channel axis (dim=1).
// ---------------------------------------------------------------------------

/// Per-pixel RMS normalisation over the channel dim. Done in F32 for
/// numerical stability (matches the F32 RMS-norm convention used in
/// `ltx_block::apply_rms_norm`).
fn pixel_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let mean_sq = x32.sqr()?.mean_keepdim(1)?;
    let inv_rms = (mean_sq + eps)?.sqrt()?.recip()?;
    x32.broadcast_mul(&inv_rms)?.to_dtype(in_dtype)
}

// ---------------------------------------------------------------------------
// ResnetBlock3D — norm1 → SiLU → conv1 → norm2 → SiLU → conv2 + shortcut.
// PixelNorm-only path (the 22B config doesn't use group_norm here).
// `inject_noise` and `timestep_conditioning` are off in production and
// are not implemented in this initial port.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ResnetBlock3D {
    #[allow(dead_code)]
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    conv1: CausalConv3d,
    conv2: CausalConv3d,
    /// Channel-changing 1×1×1 conv when `in != out`, else identity.
    conv_shortcut: Option<LinearConv3d>,
    /// `nn.GroupNorm(num_groups=1, num_channels=in)` applied to the
    /// shortcut path when `in != out`. Skipped (Identity) otherwise.
    norm3_weight: Option<Tensor>,
    norm3_bias: Option<Tensor>,
    eps: f64,
}

impl ResnetBlock3D {
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        spatial_pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = CausalConv3d::load("conv1", spatial_pad_mode, vb.clone())?;
        let conv2 = CausalConv3d::load("conv2", spatial_pad_mode, vb.clone())?;
        let (conv_shortcut, norm3_weight, norm3_bias) = if in_channels != out_channels {
            let cs = LinearConv3d::load("conv_shortcut", vb.clone())?;
            // GroupNorm(num_groups=1, num_channels=in) — affine=True so
            // both weight (in_channels,) and bias (in_channels,) are on disk.
            let pn = vb.pp("norm3");
            let w = pn.get(in_channels, "weight")?;
            let b = pn.get(in_channels, "bias")?;
            (Some(cs), Some(w), Some(b))
        } else {
            (None, None, None)
        };
        Ok(Self {
            in_channels,
            out_channels,
            conv1,
            conv2,
            conv_shortcut,
            norm3_weight,
            norm3_bias,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        // Main branch.
        let h = pixel_norm(x, self.eps)?;
        let h = h.apply(&Activation::Silu)?;
        let h = self.conv1.forward(&h, causal)?;
        let h = pixel_norm(&h, self.eps)?;
        let h = h.apply(&Activation::Silu)?;
        let h = self.conv2.forward(&h, causal)?;

        // Shortcut branch.
        let shortcut = if let Some(cs) = &self.conv_shortcut {
            // GroupNorm(num_groups=1) is essentially LayerNorm over (C, T, H, W)
            // — normalise across (C, T, H, W) per (B,) sample, with per-channel
            // affine. Implementation: (B, C, T, H, W) → (B, C*T*H*W) for the
            // mean/var computation, then apply per-channel weight/bias.
            // Equivalent: parameterless LN over the flattened (C, T, H, W) tail,
            // followed by `weight[None, :, None, None, None] * x + bias[None, :, ...]`.
            let dims = x.dims5()?;
            let (b, c, t, hh, ww) = dims;
            let flat = x.reshape((b, c * t * hh * ww))?;
            let normed = layer_norm_no_params(&flat, self.eps)?;
            let normed = normed.reshape((b, c, t, hh, ww))?;
            let w = self
                .norm3_weight
                .as_ref()
                .unwrap()
                .reshape((1, c, 1, 1, 1))?
                .to_dtype(normed.dtype())?;
            let bb = self
                .norm3_bias
                .as_ref()
                .unwrap()
                .reshape((1, c, 1, 1, 1))?
                .to_dtype(normed.dtype())?;
            let normed = normed.broadcast_mul(&w)?.broadcast_add(&bb)?;
            cs.forward(&normed)?
        } else {
            x.clone()
        };

        h.broadcast_add(&shortcut)
    }
}

// ---------------------------------------------------------------------------
// UNetMidBlock3D — stack of ResnetBlock3D (no resampling).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UNetMidBlock3D {
    res_blocks: Vec<ResnetBlock3D>,
}

impl UNetMidBlock3D {
    pub fn load(
        in_channels: usize,
        num_layers: usize,
        spatial_pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let res_blocks = (0..num_layers)
            .map(|i| {
                ResnetBlock3D::load(
                    in_channels,
                    in_channels,
                    spatial_pad_mode,
                    vb.pp(format!("res_blocks.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { res_blocks })
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.res_blocks {
            h = block.forward(&h, causal)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// DepthToSpaceUpsample — 3×3×3 causal conv → pixel-shuffle along (t, h, w).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DepthToSpaceUpsample {
    conv: CausalConv3d,
    /// (st, sh, sw) — pixel-shuffle factors per axis.
    stride: (usize, usize, usize),
}

impl DepthToSpaceUpsample {
    pub fn load(
        _in_channels: usize,
        stride: (usize, usize, usize),
        spatial_pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = CausalConv3d::load("conv", spatial_pad_mode, vb)?;
        Ok(Self { conv, stride })
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let y = self.conv.forward(x, causal)?;
        // Wan2GP's einops: "b (c p1 p2 p3) d h w -> b c (d p1) (h q) (w r)"
        // i.e. depth-to-space along (t, h, w) by stride factors.
        let (st, sh, sw) = self.stride;
        let dims = y.dims5()?;
        let (b, c_packed, d, h, w) = dims;
        let p_total = st * sh * sw;
        if c_packed % p_total != 0 {
            candle_core::bail!(
                "DepthToSpaceUpsample: channel count {} not divisible by stride volume {}",
                c_packed,
                p_total
            );
        }
        let c_out = c_packed / p_total;
        // Reshape (B, C*p1*p2*p3, D, H, W) → (B, C, p1, p2, p3, D, H, W)
        // then permute to interleave the p axes with their D/H/W partners,
        // and finally reshape to (B, C, D*p1, H*p2, W*p3).
        //
        // Concretely we want the output index ordering:
        //   out[b, c, d*p1 + i, h*p2 + j, w*p3 + k]
        //     = in[b, c*p1*p2*p3 + i*p2*p3 + j*p3 + k, d, h, w]
        let reshaped = y.reshape(&[b, c_out, st, sh, sw, d, h, w])?;
        // Move (st, sh, sw) next to (d, h, w):
        //   (b, c, st, sh, sw, d, h, w)
        //     → (b, c, d, st, h, sh, w, sw)
        //     dims:   0  1  5   2   6   3   7   4
        let permuted = reshaped
            .permute([0usize, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?;
        let mut up = permuted.reshape((b, c_out, d * st, h * sh, w * sw))?;
        // Wan2GP drops the first frame after temporal upsampling
        // (`if self.stride[0] == 2: x = x[:, :, 1:, :, :]`). This trims
        // the duplicated boundary frame the causal pad introduced.
        if st == 2 {
            up = up.narrow(2, 1, d * st - 1)?;
        }
        Ok(up)
    }
}

// ---------------------------------------------------------------------------
// SpaceToDepthDownsample — encoder counterpart of DepthToSpaceUpsample.
// 3×3×3 causal conv → space-to-depth pixel-shuffle along (t, h, w),
// plus a residual skip path that averages the packed-channel groups.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpaceToDepthDownsample {
    conv: CausalConv3d,
    /// (st, sh, sw) — space-to-depth factors per axis.
    stride: (usize, usize, usize),
    /// Channels in the SHORTCUT path before averaging — equals
    /// `in_channels * st*sh*sw`. The skip output is reshaped from
    /// `(B, in*stride_volume, ...)` to `(B, out_channels, group_size, ...)`
    /// and averaged over `group_size`.
    in_channels: usize,
    out_channels: usize,
}

impl SpaceToDepthDownsample {
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        spatial_pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = CausalConv3d::load("conv", spatial_pad_mode, vb)?;
        Ok(Self {
            conv,
            stride,
            in_channels,
            out_channels,
        })
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let (st, sh, sw) = self.stride;
        let stride_vol = st * sh * sw;

        // Encoder is always causal in LTX 2.3; the temporal "pre-pad" in
        // SpaceToDepthDownsample only kicks in for stride[0]==2 — it
        // duplicates the first frame so that the time dim becomes
        // divisible by 2 after the space-to-depth pack.
        let x_in = if st == 2 {
            let first = x.narrow(2, 0, 1)?;
            Tensor::cat(&[&first, x], 2)?
        } else {
            x.clone()
        };

        // Space-to-depth pack. (B, c, d*p1, h*p2, w*p3) →
        //                     (B, c*p1*p2*p3, d, h, w)
        // Channel decomposition order in einops: (c, p1, p2, p3) outer→inner.
        let dims = x_in.dims5()?;
        let (b, c, dt, hh, ww) = dims;
        if dt % st != 0 || hh % sh != 0 || ww % sw != 0 {
            candle_core::bail!(
                "SpaceToDepthDownsample: input shape {:?} not divisible by stride {:?}",
                dims,
                self.stride
            );
        }
        let d_o = dt / st;
        let h_o = hh / sh;
        let w_o = ww / sw;

        // Reshape (B, c, d_o, p1, h_o, p2, w_o, p3)
        let split = x_in.reshape(&[b, c, d_o, st, h_o, sh, w_o, sw])?;
        // Permute to (B, c, p1, p2, p3, d_o, h_o, w_o):
        //   source: 0=b 1=c 2=d 3=p1 4=h 5=p2 6=w 7=p3
        //   target permute spec [0, 1, 3, 5, 7, 2, 4, 6]
        let packed = split
            .permute([0usize, 1, 3, 5, 7, 2, 4, 6])?
            .contiguous()?
            .reshape((b, c * stride_vol, d_o, h_o, w_o))?;

        // Skip-connection branch: split channel into (out_channels,
        // group_size) and average over the group dim.
        let group_size = self.in_channels * stride_vol / self.out_channels;
        let skip = packed
            .reshape((b, self.out_channels, group_size, d_o, h_o, w_o))?
            .mean(2)?;

        // Main branch: causal 3D conv, then same space-to-depth pack.
        let conv_out = self.conv.forward(&x_in, causal)?;
        let conv_dims = conv_out.dims5()?;
        let (cb, cc, cdt, chh, cww) = conv_dims;
        let conv_packed = conv_out
            .reshape(&[cb, cc, cdt / st, st, chh / sh, sh, cww / sw, sw])?
            .permute([0usize, 1, 3, 5, 7, 2, 4, 6])?
            .contiguous()?
            .reshape((cb, cc * stride_vol, cdt / st, chh / sh, cww / sw))?;

        conv_packed.broadcast_add(&skip)
    }
}

// ---------------------------------------------------------------------------
// PerChannelStatistics — `(x * std-of-means) + mean-of-means`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PerChannelStatistics {
    /// (C,)
    std_of_means: Tensor,
    /// (C,)
    mean_of_means: Tensor,
}

impl PerChannelStatistics {
    pub fn load(latent_channels: usize, vb: VarBuilder) -> Result<Self> {
        // Wan2GP buffers are named with literal hyphens in the key —
        // `std-of-means` / `mean-of-means` — and Python's
        // `register_buffer` writes them through verbatim.
        let std_of_means = vb.get(latent_channels, "std-of-means")?;
        let mean_of_means = vb.get(latent_channels, "mean-of-means")?;
        Ok(Self {
            std_of_means,
            mean_of_means,
        })
    }

    pub fn un_normalize(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let s = self
            .std_of_means
            .reshape((1, self.std_of_means.dim(0)?, 1, 1, 1))?
            .to_dtype(dtype)?;
        let m = self
            .mean_of_means
            .reshape((1, self.mean_of_means.dim(0)?, 1, 1, 1))?
            .to_dtype(dtype)?;
        x.broadcast_mul(&s)?.broadcast_add(&m)
    }

    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let s = self
            .std_of_means
            .reshape((1, self.std_of_means.dim(0)?, 1, 1, 1))?
            .to_dtype(dtype)?;
        let m = self
            .mean_of_means
            .reshape((1, self.mean_of_means.dim(0)?, 1, 1, 1))?
            .to_dtype(dtype)?;
        x.broadcast_sub(&m)?.broadcast_div(&s)
    }
}

// ---------------------------------------------------------------------------
// VideoDecoder — the production decoder path.
// ---------------------------------------------------------------------------

/// Materialised decoder block — variant tracks which forward path to use.
#[derive(Debug, Clone)]
enum UpBlock {
    Mid(UNetMidBlock3D),
    Up(DepthToSpaceUpsample),
}

impl UpBlock {
    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        match self {
            UpBlock::Mid(b) => b.forward(x, causal),
            UpBlock::Up(b) => b.forward(x, causal),
        }
    }
}

pub struct LtxVideoDecoder {
    pub config: LtxVaeConfig,
    pub device: Device,

    pub per_channel_statistics: PerChannelStatistics,
    pub conv_in: CausalConv3d,
    up_blocks: Vec<UpBlock>,
    pub conv_out: CausalConv3d,
}

impl LtxVideoDecoder {
    /// Load the decoder from a Wan2GP-style VAE checkpoint. The
    /// expected key prefixes are `vae.decoder.*` and
    /// `vae.per_channel_statistics.*`.
    pub fn load(
        weights_path: &Path,
        config: LtxVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = safe_builder(weights_path, device, dtype, false)?;
        Self::build(config, device, vb)
    }

    fn build(config: LtxVaeConfig, device: &Device, vb: VarBuilder) -> Result<Self> {
        let pcs = PerChannelStatistics::load(
            config.latent_channels,
            vb.pp("vae").pp("per_channel_statistics"),
        )?;

        let dec = vb.pp("vae").pp("decoder");
        let pad = config.spatial_padding_mode;
        let conv_in = CausalConv3d::load("conv_in", pad, dec.clone())?;

        // Walk decoder_blocks in REVERSED order (config[-1] is up_blocks[0]).
        let mut up_blocks: Vec<UpBlock> = Vec::new();
        let mut feature_channels = config.base_channels * 8;
        for (i, spec) in config.decoder_blocks.iter().rev().enumerate() {
            let block_vb = dec.pp(format!("up_blocks.{i}"));
            let (block, out_ch) = match spec {
                DecoderBlockSpec::ResX { num_layers } => {
                    let mid = UNetMidBlock3D::load(feature_channels, *num_layers, pad, block_vb)?;
                    (UpBlock::Mid(mid), feature_channels)
                }
                DecoderBlockSpec::CompressTime { multiplier } => {
                    let up =
                        DepthToSpaceUpsample::load(feature_channels, (2, 1, 1), pad, block_vb)?;
                    (UpBlock::Up(up), feature_channels / multiplier)
                }
                DecoderBlockSpec::CompressSpace { multiplier } => {
                    let up =
                        DepthToSpaceUpsample::load(feature_channels, (1, 2, 2), pad, block_vb)?;
                    (UpBlock::Up(up), feature_channels / multiplier)
                }
                DecoderBlockSpec::CompressAll { multiplier } => {
                    let up =
                        DepthToSpaceUpsample::load(feature_channels, (2, 2, 2), pad, block_vb)?;
                    (UpBlock::Up(up), feature_channels / multiplier)
                }
            };
            up_blocks.push(block);
            feature_channels = out_ch;
        }

        // After the upsampling stack, conv_out emits
        //   feature_channels → out_channels * patch_size**2
        // (e.g. 128 → 3 * 16 = 48 for the production config).
        let conv_out = CausalConv3d::load("conv_out", pad, dec.clone())?;

        Ok(Self {
            config,
            device: device.clone(),
            per_channel_statistics: pcs,
            conv_in,
            up_blocks,
            conv_out,
        })
    }

    /// Decode a `(B, latent_channels, F', H', W')` latent into a
    /// `(B, out_channels, F, H, W)` video where `F = 8*(F'-1) + 1`,
    /// `H = patch_size * H' * (spatial up-block factors)`,
    /// `W = patch_size * W' * (spatial up-block factors)`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let causal = self.config.causal_decoder;

        // Denormalise.
        let x = self.per_channel_statistics.un_normalize(latent)?;

        // conv_in (latent_channels → base_channels * 8).
        let mut h = self.conv_in.forward(&x, causal)?;

        // Up blocks.
        for block in &self.up_blocks {
            h = block.forward(&h, causal)?;
        }

        // conv_norm_out (PixelNorm) → SiLU → conv_out.
        let h = pixel_norm(&h, 1e-6)?;
        let h = h.apply(&Activation::Silu)?;
        let h = self.conv_out.forward(&h, causal)?;

        // Final unpatchify: (B, out_channels * p^2, F, H', W')
        //   → (B, out_channels, F, H' * p, W' * p)
        unpatchify_5d(&h, self.config.patch_size, 1)
    }

    /// Spatial-only tiled decode. Splits the latent into overlapping
    /// (h, w) tiles, decodes each, blends with a 2D trapezoidal mask
    /// (linear ramp on shared edges, flat in the centre). Avoids
    /// allocating a single full-size activation buffer on the GPU when
    /// targeting high-res output (1080p+). Temporal tiling NOT yet
    /// supported — pass the full F' axis in one tile.
    ///
    /// `tile_h`, `tile_w` — tile size in LATENT space (post-VAE-downscale).
    /// `overlap_h`, `overlap_w` — overlap between adjacent tiles, also
    /// in LATENT space; must satisfy `overlap < tile`.
    ///
    /// Returns a `(B, 3, F, H, W)` video on CPU (the decode is run on
    /// the encoder's device, but tiles are blended on CPU to keep GPU
    /// memory bounded by `O(tile_size² × decoder activations)`).
    pub fn decode_tiled(
        &self,
        latent: &Tensor,
        tile_h: usize,
        tile_w: usize,
        overlap_h: usize,
        overlap_w: usize,
    ) -> Result<Tensor> {
        if overlap_h >= tile_h || overlap_w >= tile_w {
            candle_core::bail!(
                "decode_tiled: overlap ({}×{}) must be smaller than tile ({}×{})",
                overlap_h,
                overlap_w,
                tile_h,
                tile_w
            );
        }
        let dims = latent.dims5()?;
        let (b, _c, f_lat, h_lat, w_lat) = dims;

        // Pixel-space dimensions. The decoder up-blocks contribute 8×
        // spatial; the final patch_size unpatchify contributes another
        // ×patch_size; total spatial = patch_size × 8 (= 32 for 22B).
        // Time: causal VAE expands `(F'-1) × 8 + 1`.
        let p = self.config.patch_size;
        let scale_t: usize = 8; // causal VAE temporal stride
        let spatial_scale: usize = 8 * p; // up-blocks × patchify
        let f_pix = if f_lat == 1 {
            1
        } else {
            (f_lat - 1) * scale_t + 1
        };
        let h_pix = h_lat * spatial_scale;
        let w_pix = w_lat * spatial_scale;

        let h_intervals = build_intervals(h_lat, tile_h, overlap_h);
        let w_intervals = build_intervals(w_lat, tile_w, overlap_w);

        // Blend on the host so even a large output cannot crowd the decoder
        // out of VRAM. Direct row accumulation avoids copying the complete
        // output tensor for every tile (Tensor::slice_assign is out of place).
        let mut acc = vec![0f32; b * 3 * f_pix * h_pix * w_pix];
        let mut wsum = vec![0f32; h_pix * w_pix];
        for &(h_lo, h_hi, h_left, h_right) in &h_intervals {
            for &(w_lo, w_hi, w_left, w_right) in &w_intervals {
                let tile_latent = latent
                    .narrow(3, h_lo, h_hi - h_lo)?
                    .narrow(4, w_lo, w_hi - w_lo)?
                    .contiguous()?;
                let tile = self
                    .decode(&tile_latent)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
                let (_, _, tf, th, tw) = tile.dims5()?;
                if tf != f_pix {
                    candle_core::bail!("unexpected VAE tile frame count");
                }
                let values = tile.flatten_all()?.to_vec1::<f32>()?;
                let hm = trapezoidal_mask_1d(th, h_left * spatial_scale, h_right * spatial_scale)?
                    .to_vec1::<f32>()?;
                let wm = trapezoidal_mask_1d(tw, w_left * spatial_scale, w_right * spatial_scale)?
                    .to_vec1::<f32>()?;
                let y0 = h_lo * spatial_scale;
                let x0 = w_lo * spatial_scale;
                for y in 0..th {
                    for x in 0..tw {
                        wsum[(y0 + y) * w_pix + x0 + x] += hm[y] * wm[x];
                    }
                }
                for plane in 0..b * 3 * f_pix {
                    for y in 0..th {
                        let source = (plane * th + y) * tw;
                        let dest = (plane * h_pix + y0 + y) * w_pix + x0;
                        for x in 0..tw {
                            acc[dest + x] += values[source + x] * hm[y] * wm[x];
                        }
                    }
                }
            }
        }
        let plane_size = h_pix * w_pix;
        for plane in acc.chunks_mut(plane_size) {
            for (value, weight) in plane.iter_mut().zip(&wsum) {
                if *weight <= 0. {
                    candle_core::bail!("uncovered VAE tile pixel");
                }
                *value /= weight;
            }
        }
        Tensor::from_vec(acc, (b, 3, f_pix, h_pix, w_pix), &Device::Cpu)
    }
}

/// Build (lo, hi, left_ramp, right_ramp) intervals for a single axis of
/// length `axis_len`, given `tile_size` and `overlap`. The first tile's
/// `left_ramp = 0` (no leftward neighbour), and the last tile's
/// `right_ramp = 0` — those edges keep full weight (= 1) so the boundary
/// stays at unit blend instead of fading.
fn build_intervals(
    axis_len: usize,
    tile_size: usize,
    overlap: usize,
) -> Vec<(usize, usize, usize, usize)> {
    if axis_len <= tile_size {
        return vec![(0, axis_len, 0, 0)];
    }
    let stride = tile_size - overlap;
    let mut out: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut lo = 0usize;
    loop {
        let hi = (lo + tile_size).min(axis_len);
        let is_first = out.is_empty();
        let is_last = hi == axis_len;
        let left = if is_first { 0 } else { overlap };
        let right = if is_last { 0 } else { overlap };
        out.push((lo, hi, left, right));
        if is_last {
            break;
        }
        lo += stride;
    }
    out
}

/// Trapezoidal 1D mask: linear ramp from 0→1 over the leftmost
/// `left_ramp` cells, flat 1 in the middle, linear ramp 1→0 over the
/// rightmost `right_ramp` cells. F32. When ramps are 0, returns all-1s.
fn trapezoidal_mask_1d(length: usize, left_ramp: usize, right_ramp: usize) -> Result<Tensor> {
    let mut v: Vec<f32> = vec![1.0; length];
    if left_ramp > 0 {
        for i in 0..left_ramp.min(length) {
            v[i] = (i as f32 + 1.0) / (left_ramp as f32 + 1.0);
        }
    }
    if right_ramp > 0 {
        for i in 0..right_ramp.min(length) {
            let idx = length - 1 - i;
            let edge_v = (i as f32 + 1.0) / (right_ramp as f32 + 1.0);
            // Take the min of any prior left ramp and this right ramp so
            // the corners sum to 1 instead of double-counting.
            v[idx] = v[idx].min(edge_v);
        }
    }
    Tensor::from_vec(v, length, &Device::Cpu)
}

// ---------------------------------------------------------------------------
// LtxVideoEncoder — produces (means, repeated_logvar) latent format.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum DownBlock {
    Mid(UNetMidBlock3D),
    Down(SpaceToDepthDownsample),
}

impl DownBlock {
    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        match self {
            DownBlock::Mid(b) => b.forward(x, causal),
            DownBlock::Down(b) => b.forward(x, causal),
        }
    }
}

pub struct LtxVideoEncoder {
    pub config: LtxVaeConfig,
    pub device: Device,

    pub per_channel_statistics: PerChannelStatistics,
    pub conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock>,
    pub conv_out: CausalConv3d,
}

impl LtxVideoEncoder {
    pub fn load(
        weights_path: &Path,
        config: LtxVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = safe_builder(weights_path, device, dtype, true)?;
        Self::build(config, device, vb)
    }

    fn build(config: LtxVaeConfig, device: &Device, vb: VarBuilder) -> Result<Self> {
        let pcs = PerChannelStatistics::load(
            config.latent_channels,
            vb.pp("vae").pp("per_channel_statistics"),
        )?;

        let enc = vb.pp("vae").pp("encoder");
        let pad = config.encoder_spatial_padding_mode;

        // Encoder conv_in takes patchified input (in_channels * patch_size²)
        // and emits `latent_channels` features (the trunk starts at the
        // latent dim and grows through the down-blocks).
        let conv_in = CausalConv3d::load("conv_in", pad, enc.clone())?;

        let mut down_blocks: Vec<DownBlock> = Vec::new();
        let mut feature_channels = config.latent_channels;
        for (i, spec) in config.encoder_blocks.iter().enumerate() {
            let block_vb = enc.pp(format!("down_blocks.{i}"));
            let (block, out_ch) = match spec {
                EncoderBlockSpec::ResX { num_layers } => {
                    let mid = UNetMidBlock3D::load(feature_channels, *num_layers, pad, block_vb)?;
                    (DownBlock::Mid(mid), feature_channels)
                }
                EncoderBlockSpec::CompressSpaceRes { multiplier } => {
                    let out_ch = feature_channels * multiplier;
                    let down = SpaceToDepthDownsample::load(
                        feature_channels,
                        out_ch,
                        (1, 2, 2),
                        pad,
                        block_vb,
                    )?;
                    (DownBlock::Down(down), out_ch)
                }
                EncoderBlockSpec::CompressTimeRes { multiplier } => {
                    let out_ch = feature_channels * multiplier;
                    let down = SpaceToDepthDownsample::load(
                        feature_channels,
                        out_ch,
                        (2, 1, 1),
                        pad,
                        block_vb,
                    )?;
                    (DownBlock::Down(down), out_ch)
                }
                EncoderBlockSpec::CompressAllRes { multiplier } => {
                    let out_ch = feature_channels * multiplier;
                    let down = SpaceToDepthDownsample::load(
                        feature_channels,
                        out_ch,
                        (2, 2, 2),
                        pad,
                        block_vb,
                    )?;
                    (DownBlock::Down(down), out_ch)
                }
            };
            down_blocks.push(block);
            feature_channels = out_ch;
        }

        // conv_out: feature_channels → latent_channels + 1 (UNIFORM logvar).
        let conv_out = CausalConv3d::load("conv_out", pad, enc.clone())?;
        let _ = feature_channels;

        Ok(Self {
            config,
            device: device.clone(),
            per_channel_statistics: pcs,
            conv_in,
            down_blocks,
            conv_out,
        })
    }

    /// Encode a `(B, in_channels, F, H, W)` video into latent means
    /// `(B, latent_channels, F', H', W')` where `F' = (F-1)/8 + 1`,
    /// `H' = H / (4 * spatial_compress)`, etc.
    ///
    /// Frame count must satisfy `(F - 1) % 8 == 0`. The encoder runs
    /// in CAUSAL mode (matching Wan2GP's `forward(self, sample)` which
    /// uses the default `causal=True` everywhere).
    ///
    /// Returns ONLY the latent means (the leading `latent_channels` of
    /// the post-conv_out (B, latent_channels + 1, ...) tensor — the
    /// trailing channel is the shared UNIFORM logvar that Wan2GP repeats
    /// out and concatenates downstream, which we don't need for
    /// deterministic encode-then-decode pipelines).
    pub fn encode_means(&self, video: &Tensor) -> Result<Tensor> {
        let frames = video.dim(2)?;
        if (frames.saturating_sub(1)) % 8 != 0 {
            candle_core::bail!(
                "LtxVideoEncoder: F must be 1 + 8*k (got {}); valid: 1, 9, 17, 25, 33, …",
                frames
            );
        }

        // patchify spatial 4×4 → channels (no temporal patch).
        let p = self.config.patch_size;
        let patched = patchify_5d(video, p, 1)?;
        let mut h = self.conv_in.forward(&patched, true)?;
        for block in &self.down_blocks {
            h = block.forward(&h, true)?;
        }
        let h = pixel_norm(&h, 1e-6)?;
        let h = h.apply(&Activation::Silu)?;
        let h = self.conv_out.forward(&h, true)?;
        // First `latent_channels` are the means; trailing 1 channel is
        // the UNIFORM logvar (used for VAE sampling, not needed when we
        // just want the deterministic mean for decode-only pipelines).
        let means = h.narrow(1, 0, self.config.latent_channels)?;
        // Wan2GP's `VideoEncoder.forward` returns
        // `per_channel_statistics.normalize(means)` — the means are
        // shifted by `mean-of-means` and divided by `std-of-means` so the
        // downstream diffusion sees ~unit-variance latents. The decoder's
        // `un_normalize` is the inverse, so encode-then-decode round-trips.
        self.per_channel_statistics.normalize(&means)
    }
}

/// Patchify spatial 4×4 patches into channels (no temporal). Inverse of
/// `unpatchify_5d`. Mirrors Wan2GP's
/// `b c (f p) (h q) (w r) -> b (c p r q) f h w` einops rearrange,
/// where the channel decomposition is (c, p_t, p_w, p_h).
pub fn patchify_5d(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 0 || patch_size_t == 0 {
        candle_core::bail!("patch sizes must be nonzero");
    }
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }
    let (b, c, f, h, w) = x.dims5()?;
    if f % patch_size_t != 0 || h % patch_size_hw != 0 || w % patch_size_hw != 0 {
        candle_core::bail!(
            "patchify_5d: shape ({}, {}, {}, {}, {}) not divisible by patches ({}, {})",
            b,
            c,
            f,
            h,
            w,
            patch_size_t,
            patch_size_hw
        );
    }
    let f_p = f / patch_size_t;
    let h_p = h / patch_size_hw;
    let w_p = w / patch_size_hw;
    // Split spatial dims out: (B, C, f_p, p_t, h_p, p_h, w_p, p_w)
    let split = x.reshape(&[
        b,
        c,
        f_p,
        patch_size_t,
        h_p,
        patch_size_hw,
        w_p,
        patch_size_hw,
    ])?;
    // Permute to (B, C, p_t, p_w, p_h, f_p, h_p, w_p):
    //   source: 0=b 1=c 2=f_p 3=p_t 4=h_p 5=p_h 6=w_p 7=p_w
    //   permute spec [0, 1, 3, 7, 5, 2, 4, 6]
    // Note: einops channel order is (c, p_t, p_w, p_h) — so p_w (source 7)
    // sits between p_t and p_h.
    let permuted = split.permute([0usize, 1, 3, 7, 5, 2, 4, 6])?.contiguous()?;
    let p_total = patch_size_t * patch_size_hw * patch_size_hw;
    permuted.reshape((b, c * p_total, f_p, h_p, w_p))
}

/// Unpatchify a 5D tensor — inverse of the `patchify` operation that
/// folds spatial patches into channels. Ports Wan2GP's
/// `b (c p r q) f h w -> b c (f p) (h q) (w r)` einops rearrange,
/// where the variable bindings are `p → f (time)`, `q → h (height)`,
/// `r → w (width)` and the channel decomposition is therefore
/// (c, p_t, p_w, p_h) in slowest→fastest order — NOTE the W patch
/// sits between c and the H patch, NOT after it.
pub fn unpatchify_5d(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims5()?;
    let (b, c_packed, f, h, w) = dims;
    let p_total = patch_size_hw
        .checked_mul(patch_size_hw)
        .and_then(|v| v.checked_mul(patch_size_t))
        .filter(|&v| v > 0)
        .ok_or_else(|| candle_core::Error::Msg("invalid patch volume".into()))?;
    if c_packed % p_total != 0 {
        candle_core::bail!(
            "unpatchify_5d: channel count {} not divisible by patch volume {}",
            c_packed,
            p_total
        );
    }
    let c_out = c_packed / p_total;
    // Channel split in einops order: (c, p_t, p_w, p_h).
    let reshaped = x.reshape(&[
        b,
        c_out,
        patch_size_t,
        patch_size_hw, // p_w
        patch_size_hw, // p_h
        f,
        h,
        w,
    ])?;
    // Move each patch axis next to its spatial partner:
    //   (b, c, p_t, p_w, p_h, f, h, w)        index: 0 1  2   3   4  5 6 7
    //     → (b, c, f, p_t, h, p_h, w, p_w)
    //   permute dims:        0  1  5   2   6   4   7   3
    let permuted = reshaped
        .permute([0usize, 1, 5, 2, 6, 4, 7, 3])?
        .contiguous()?;
    permuted.reshape((
        b,
        c_out,
        f * patch_size_t,
        h * patch_size_hw,
        w * patch_size_hw,
    ))
}

fn safe_builder(
    path: &Path,
    device: &Device,
    dtype: DType,
    encoder: bool,
) -> Result<VarBuilder<'static>> {
    let mut store = super::store::Store::open(path, 0)?;
    let component = if encoder { "encoder." } else { "decoder." };
    let names: Vec<_> = store
        .index
        .names()
        .filter(|k| {
            let k = k.strip_prefix("vae.").unwrap_or(k);
            k.starts_with(component) || k.starts_with("per_channel_statistics.")
        })
        .map(str::to_owned)
        .collect();
    let mut tensors = std::collections::HashMap::new();
    for name in names {
        let key = if name.starts_with("vae.") {
            name.clone()
        } else {
            format!("vae.{name}")
        };
        tensors.insert(key, store.tensor(&name, device, false)?.to_dtype(dtype)?);
    }
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}
