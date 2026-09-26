use super::store::{Group, Store};
use crate::math::{attention, heads, layer_norm, unheads};
use candle_core::{DType, Device, Result, Tensor, D};

pub const PREFIX: &str = "model.diffusion_model.";
pub fn norm(x: &Tensor) -> Result<Tensor> {
    rms(
        x,
        &Tensor::ones(x.dim(D::Minus1)?, x.dtype(), x.device())?,
        1e-6,
    )
}
fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, &weight.contiguous()?, eps as f32)
}
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
    first_frame_tokens: usize,
}
impl Rope {
    /// Split rotary tables for `dim` channels over `heads` heads: each head
    /// rotates its first half against its second. Frequencies are spread over
    /// the axes, and any remainder is padded with identity rotations first.
    pub fn positions(positions: &[Vec<f32>], maxima: &[f32], dim: usize, heads: usize, dev: &Device) -> Result<Self> {
        let axes = maxima.len();
        if axes == 0 || positions.iter().any(|p| p.len() != axes) || dim % (2 * heads) != 0 {
            candle_core::bail!("invalid LTX rotary coordinates");
        }
        let half = dim / 2;
        let count = dim / (2 * axes);
        let pad = half - count * axes;
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(cos.capacity());
        for p in positions {
            cos.extend(std::iter::repeat_n(1f32, pad));
            sin.extend(std::iter::repeat_n(0f32, pad));
            for j in 0..count {
                let freq = (10000f64.powf(j as f64 / (count - 1) as f64)
                    * std::f64::consts::FRAC_PI_2) as f32;
                for axis in 0..axes {
                    let phase = freq * (2. * p[axis] / maxima[axis] - 1.);
                    cos.push(phase.cos());
                    sin.push(phase.sin());
                }
            }
        }
        let shape = (1, positions.len(), heads, half / heads);
        Ok(Self {
            first_frame_tokens: 0,
            cos: Tensor::from_vec(cos, shape, dev)?
                .transpose(1, 2)?
                .to_dtype(DType::BF16)?,
            sin: Tensor::from_vec(sin, shape, dev)?
                .transpose(1, 2)?
                .to_dtype(DType::BF16)?,
        })
    }
    #[cfg(test)]
    pub fn video(
        frames: usize,
        height: usize,
        width: usize,
        fps: usize,
        dev: &Device,
    ) -> Result<Self> {
        Self::video_with_end(frames, height, width, fps, false, dev)
    }
    pub fn video_with_end(
        frames: usize,
        height: usize,
        width: usize,
        fps: usize,
        end_image: bool,
        dev: &Device,
    ) -> Result<Self> {
        let positions = Self::video_positions(frames, height, width, fps, end_image);
        let mut rope = Self::positions(&positions, &[20., 2048., 2048.], 4096, 32, dev)?;
        rope.first_frame_tokens = height * width;
        Ok(rope)
    }
    /// The video tokens' side of audio-video cross-attention: time only, at
    /// the audio stream's width (2048 channels, 32 heads of 64).
    pub fn video_cross(
        frames: usize,
        height: usize,
        width: usize,
        fps: usize,
        end_image: bool,
        dev: &Device,
    ) -> Result<Self> {
        let times: Vec<Vec<f32>> = Self::video_positions(frames, height, width, fps, end_image)
            .into_iter()
            .map(|p| vec![p[0]])
            .collect();
        Self::positions(&times, &[20.], 2048, 32, dev)
    }
    /// Audio latent frame `i` covers mel frames `[max(4i - 3, 0), 4i + 1)` at
    /// 100 per second; its rotary position is the middle, in seconds. The same
    /// table serves audio self-attention and the audio side of cross-attention.
    pub fn audio(frames: usize, dev: &Device) -> Result<Self> {
        let times: Vec<Vec<f32>> = (0..frames)
            .map(|i| {
                let start = (4 * i).saturating_sub(3) as f32 * 0.01;
                let end = (4 * i + 1) as f32 * 0.01;
                vec![(start + end) / 2.]
            })
            .collect();
        Self::positions(&times, &[20.], 2048, 32, dev)
    }
    fn video_positions(
        frames: usize,
        height: usize,
        width: usize,
        fps: usize,
        end_image: bool,
    ) -> Vec<Vec<f32>> {
        let mut positions = Vec::new();
        for t in 0..frames {
            for y in 0..height {
                for x in 0..width {
                    let start = (t * 8).saturating_sub(7);
                    let end = (t + 1) * 8 - 7;
                    positions.push(vec![
                        (start + end) as f32 / (2 * fps) as f32,
                        (y as f32 + 0.5) * 32.,
                        (x as f32 + 0.5) * 32.,
                    ]);
                }
            }
        }
        if end_image {
            let frame_index = (frames - 1) * 8;
            for y in 0..height {
                for x in 0..width {
                    positions.push(vec![
                        (frame_index as f32 + 0.5) / fps as f32,
                        (y as f32 + 0.5) * 32.,
                        (x as f32 + 0.5) * 32.,
                    ]);
                }
            }
        }
        positions
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let half = x.dim(3)? / 2;
        let a = x.narrow(3, 0, half)?;
        let b = x.narrow(3, half, half)?;
        // The reference rounds x*cos first, then uses an FP32 addcmul
        // accumulator for the sine term before returning to BF16.
        let sin = self.sin.to_dtype(DType::F32)?;
        Tensor::cat(
            &[
                (a.broadcast_mul(&self.cos)?.to_dtype(DType::F32)?
                    - b.to_dtype(DType::F32)?.broadcast_mul(&sin)?)?
                .to_dtype(x.dtype())?,
                (b.broadcast_mul(&self.cos)?.to_dtype(DType::F32)?
                    + a.to_dtype(DType::F32)?.broadcast_mul(&sin)?)?
                .to_dtype(x.dtype())?,
            ],
            3,
        )
    }
}
pub fn attn(
    w: &Group,
    prefix: &str,
    x: &Tensor,
    context: &Tensor,
    rope: Option<&Rope>,
) -> Result<Tensor> {
    attn_pe(w, prefix, x, context, rope, rope)
}
/// Attention whose queries and keys rotate by different tables (audio-video
/// cross-attention places each side on its own timeline).
fn attn_pe(
    w: &Group,
    prefix: &str,
    x: &Tensor,
    context: &Tensor,
    q_rope: Option<&Rope>,
    k_rope: Option<&Rope>,
) -> Result<Tensor> {
    let mut q = heads(
        &rms(
            &w.linear(&format!("{prefix}.to_q"), x)?,
            w.get(&format!("{prefix}.q_norm.weight"))?,
            1e-6,
        )?,
        32,
    )?;
    let mut k = heads(
        &rms(
            &w.linear(&format!("{prefix}.to_k"), context)?,
            w.get(&format!("{prefix}.k_norm.weight"))?,
            1e-6,
        )?,
        32,
    )?;
    let v = heads(&w.linear(&format!("{prefix}.to_v"), context)?, 32)?;
    if let Some(r) = q_rope {
        q = r.apply(&q)?;
    }
    if let Some(r) = k_rope {
        k = r.apply(&k)?;
    }
    let mut y = attention(&q, &k, &v, 0)?;
    if w.tensors
        .contains_key(&format!("{prefix}.to_gate_logits.weight"))
    {
        let logits = w.linear(&format!("{prefix}.to_gate_logits"), x)?;
        let gate = (candle_nn::ops::sigmoid(&logits.to_dtype(DType::F32)?)?
            .to_dtype(logits.dtype())?
            * 2.)?;
        y = y.broadcast_mul(&gate.transpose(1, 2)?.unsqueeze(3)?)?;
    }
    w.linear(&format!("{prefix}.to_out.0"), &unheads(&y)?)
}
pub fn ff(w: &Group, prefix: &str, x: &Tensor) -> Result<Tensor> {
    let h = w.linear(&format!("{prefix}.net.0.proj"), x)?;
    // PyTorch computes activation internals in float32, including for BF16 inputs.
    w.linear(
        &format!("{prefix}.net.2"),
        &h.to_dtype(DType::F32)?.gelu()?.to_dtype(h.dtype())?,
    )
}
fn affine(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.)?)?.broadcast_add(shift)
}
fn time_embedding(w: &Group, prefix: &str, sigma: f64, dev: &Device) -> Result<(Tensor, Tensor)> {
    let phase: Vec<_> = (0..128)
        .map(|j| (sigma as f32 * 1000.) * (-10000f32.ln() * j as f32 / 128.).exp())
        .collect();
    let values: Vec<_> = phase
        .iter()
        .map(|p| p.cos())
        .chain(phase.iter().map(|p| p.sin()))
        .collect();
    let t = Tensor::from_vec(values, (1, 1, 256), dev)?.to_dtype(DType::BF16)?;
    let t = w
        .linear(&format!("{prefix}.emb.timestep_embedder.linear_1"), &t)?
        .to_dtype(DType::F32)?
        .silu()?
        .to_dtype(DType::BF16)?;
    let t = w.linear(&format!("{prefix}.emb.timestep_embedder.linear_2"), &t)?;
    Ok((
        w.linear(
            &format!("{prefix}.linear"),
            &t.to_dtype(DType::F32)?.silu()?.to_dtype(DType::BF16)?,
        )?,
        t,
    ))
}
/// The text connector for one stream (`video_embeddings_connector`, or
/// `audio_embeddings_connector` at 2048 channels): prompt features padded to
/// 1024 tokens with learned registers, eight gated blocks, a final norm.
pub fn connector(
    store: &mut Store,
    features: &Tensor,
    name: &str,
    dev: &Device,
    mut progress: impl FnMut(usize),
) -> Result<Tensor> {
    let prefix = format!("{PREFIX}{name}.");
    let registers = store.tensor(&format!("{prefix}learnable_registers"), dev, false)?;
    let n = features.dim(1)?;
    let ids: Vec<u32> = (n..1024).map(|i| (i % 128) as u32).collect();
    let padding = registers
        .index_select(&Tensor::new(ids.as_slice(), dev)?, 0)?
        .unsqueeze(0)?;
    let mut x = Tensor::cat(&[features, &padding], 1)?;
    let rope = Rope::positions(
        &(0..1024).map(|i| vec![i as f32]).collect::<Vec<_>>(),
        &[4096.],
        features.dim(2)?,
        32,
        dev,
    )?;
    for i in 0..8 {
        let w = store.group(
            &format!("{prefix}transformer_1d_blocks.{i}."),
            dev,
            false,
            |_| true,
        )?;
        let h = norm(&x)?;
        x = (x + attn(&w, "attn1", &h, &h, Some(&rope))?)?;
        let h = ff(&w, "ff", &norm(&x)?)?;
        x = (x + h)?;
        progress(i + 1);
    }
    norm(&x)
}
fn video_weight(k: &str) -> bool {
    k.starts_with("attn1.")
        || k.starts_with("attn2.")
        || k.starts_with("ff.")
        || k == "scale_shift_table"
        || k == "prompt_scale_shift_table"
}
/// Every tensor of a block: the audio stream and audio-video cross-attention too.
fn av_weight(_: &str) -> bool {
    true
}
/// One stream's names within a block.
struct Stream {
    attn1: &'static str,
    attn2: &'static str,
    table: &'static str,
    prompt_table: &'static str,
    ff: &'static str,
}
const VIDEO: Stream = Stream {
    attn1: "attn1",
    attn2: "attn2",
    table: "scale_shift_table",
    prompt_table: "prompt_scale_shift_table",
    ff: "ff",
};
const AUDIO: Stream = Stream {
    attn1: "audio_attn1",
    attn2: "audio_attn2",
    table: "audio_scale_shift_table",
    prompt_table: "audio_prompt_scale_shift_table",
    ff: "audio_ff",
};
/// Self-attention, then text cross-attention. `m` is the stream's nine
/// modulation rows with its block table added: (shift, scale, gate) for
/// self-attention (0-2), the feed-forward (3-5) and text attention (6-8).
fn attend(
    w: &Group,
    s: &Stream,
    mut x: Tensor,
    context: &Tensor,
    m: &Tensor,
    prompt: Option<&Tensor>,
    rope: &Rope,
) -> Result<Tensor> {
    let slot = |j| m.narrow(2, j, 1)?.squeeze(2);
    let h = affine(&norm(&x)?, &slot(0)?, &slot(1)?)?;
    x = (x + attn(w, s.attn1, &h, &h, Some(rope))?.broadcast_mul(&slot(2)?)?)?;
    let mut pm = w.get(s.prompt_table)?.unsqueeze(0)?;
    if let Some(p) = prompt {
        pm = (pm + p)?;
    }
    let h = affine(&norm(&x)?, &slot(6)?, &slot(7)?)?;
    let c = affine(context, &pm.narrow(1, 0, 1)?, &pm.narrow(1, 1, 1)?)?;
    x + attn(w, s.attn2, &h, &c, None)?.broadcast_mul(&slot(8)?)?
}
fn feed(w: &Group, s: &Stream, x: Tensor, m: &Tensor) -> Result<Tensor> {
    let slot = |j| m.narrow(2, j, 1)?.squeeze(2);
    let h = affine(&norm(&x)?, &slot(3)?, &slot(4)?)?;
    x + ff(w, s.ff, &h)?.broadcast_mul(&slot(5)?)?
}
fn block(
    w: &Group,
    x: Tensor,
    context: &Tensor,
    modulation: &Tensor,
    prompt: Option<&Tensor>,
    rope: &Rope,
) -> Result<Tensor> {
    let m = modulation.broadcast_add(w.get(VIDEO.table)?)?;
    let x = attend(w, &VIDEO, x, context, &m, prompt, rope)?;
    feed(w, &VIDEO, x, &m)
}
/// The audio stream's inputs for one denoising step, computed once and
/// shared by every block.
struct AudioStep<'a> {
    context: &'a Tensor,
    /// (1, 1, 9, 2048) audio modulation, and (1, 2, 2048) prompt modulation.
    modulation: Tensor,
    prompt: Option<Tensor>,
    /// Cross-attention (scale, shift) rows: per video token (1, Tv, 4, 4096),
    /// and for audio (1, 1, 4, 2048). Rows 0-1 serve audio-to-video, 2-3
    /// video-to-audio.
    video_ss: Tensor,
    audio_ss: Tensor,
    /// Output gates of audio-to-video (1, 1, 4096) and video-to-audio (1, 1, 2048).
    video_gate: Tensor,
    audio_gate: Tensor,
    rope: &'a Rope,
    video_cross: &'a Rope,
    isolated: bool,
}
/// Audio and video attend to each other. Both directions read the states
/// from before either update, so their order does not matter.
fn cross(w: &Group, vx: Tensor, ax: Tensor, s: &AudioStep) -> Result<(Tensor, Tensor)> {
    let tv = w.get("scale_shift_table_a2v_ca_video")?;
    let ta = w.get("scale_shift_table_a2v_ca_audio")?;
    // The block tables hold (scale, shift) pairs, then the gate.
    let vss = s.video_ss.broadcast_add(&tv.narrow(0, 0, 4)?.unsqueeze(0)?.unsqueeze(0)?)?;
    let ass = s.audio_ss.broadcast_add(&ta.narrow(0, 0, 4)?.unsqueeze(0)?.unsqueeze(0)?)?;
    let vrow = |j| vss.narrow(2, j, 1)?.squeeze(2);
    let arow = |j| ass.narrow(2, j, 1)?.squeeze(2);
    let vgate = s.video_gate.broadcast_add(&tv.narrow(0, 4, 1)?)?;
    let agate = s.audio_gate.broadcast_add(&ta.narrow(0, 4, 1)?)?;
    let (vn, an) = (norm(&vx)?, norm(&ax)?);
    let q = affine(&vn, &vrow(1)?, &vrow(0)?)?;
    let k = affine(&an, &arow(1)?, &arow(0)?)?;
    let a2v = attn_pe(w, "audio_to_video_attn", &q, &k, Some(s.video_cross), Some(s.rope))?;
    let q = affine(&an, &arow(3)?, &arow(2)?)?;
    let k = affine(&vn, &vrow(3)?, &vrow(2)?)?;
    let v2a = attn_pe(w, "video_to_audio_attn", &q, &k, Some(s.rope), Some(s.video_cross))?;
    Ok(((vx + a2v.broadcast_mul(&vgate)?)?, (ax + v2a.broadcast_mul(&agate)?)?))
}
fn av_block(
    w: &Group,
    vx: Tensor,
    ax: Tensor,
    context: &Tensor,
    modulation: &Tensor,
    prompt: Option<&Tensor>,
    rope: &Rope,
    a: &AudioStep,
) -> Result<(Tensor, Tensor)> {
    let vm = modulation.broadcast_add(w.get(VIDEO.table)?)?;
    let am = a.modulation.broadcast_add(w.get(AUDIO.table)?)?;
    let vx = attend(w, &VIDEO, vx, context, &vm, prompt, rope)?;
    let ax = attend(w, &AUDIO, ax, a.context, &am, a.prompt.as_ref(), a.rope)?;
    let (vx, ax) = if a.isolated { (vx, ax) } else { cross(w, vx, ax, a)? };
    Ok((feed(w, &VIDEO, vx, &vm)?, feed(w, &AUDIO, ax, &am)?))
}

/// Whether `size` more bytes of weights can stay on the device while leaving
/// room to stream a block the size of this one (the rest go to RAM or SSD).
/// Measured, not estimated: decoding and the allocator hold more than the
/// weights' own bytes, and other programs may share the GPU.
fn fits_on_device(dev: &Device, size: u64) -> Result<bool> {
    #[cfg(feature = "cuda")]
    if let Ok(cuda) = dev.as_cuda_device() {
        let free = cuda.cuda_stream().context().mem_get_info().map_err(candle_core::Error::wrap)?.0 as u64;
        return Ok(free >= 3 * size + (2 << 30));
    }
    let _ = (dev, size);
    Ok(true)
}

/// What the audio stream brings to one forward pass.
pub struct AudioInput<'a> {
    /// Noisy audio latent (1, frames, 128).
    pub latent: &'a Tensor,
    /// Audio connector output (1, 1024, 2048).
    pub context: &'a Tensor,
    pub rope: &'a Rope,
    pub video_cross: &'a Rope,
    /// The audio stream's own sigma: the video's while both are denoised, 0
    /// when the audio is a fixed condition (a soundtrack the picture follows).
    pub sigma: f64,
    /// Skip the audio-video cross-attention in every block (both ways): the
    /// "isolated modality" pass that audio-to-video guidance contrasts with.
    pub isolated: bool,
}

pub struct Transformer {
    store: Store,
    global: Group,
    resident: Vec<Option<Group>>,
    /// Blocks carry the audio stream and audio-video cross-attention.
    audio: bool,
    layers: usize,
    pub gpu_bytes: u64,
    gpu_budget: u64,
    require_gpu: bool,
    #[cfg(test)]
    last_hidden: Option<Tensor>,
    #[cfg(test)]
    first_hidden: Option<Tensor>,
    #[cfg(test)]
    first_audio_hidden: Option<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the complete official transformer reference; NROB_LTX_GOLDEN"]
    fn full_video_transformer_matches_reference() -> Result<()> {
        full_reference(0, false)
    }
    #[test]
    #[ignore = "requires official per-token timestep reference; NROB_LTX_GOLDEN"]
    fn image_conditioned_transformer_matches_reference() -> Result<()> {
        full_reference(4, false)
    }
    #[test]
    #[ignore = "requires official start/end-keyframe reference; NROB_LTX_GOLDEN"]
    fn endpoint_conditioned_transformer_matches_reference() -> Result<()> {
        full_reference(16, true)
    }
    fn full_reference(conditioned_tokens: usize, end_frame: bool) -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
        );
        let weights = std::env::var("NROB_LTX_CHECKPOINT").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(
            std::env::var("NROB_LTX_TEST_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let tokens = if end_frame { 48 } else { 16 };
        let end_tokens = if end_frame { 16 } else { 0 };
        let prefix = if end_frame {
            "end-transformer"
        } else if conditioned_tokens == 0 {
            "transformer"
        } else {
            "i2v-transformer"
        };
        let x = read(&format!("{prefix}-input.f32"), &[1, tokens, 128])?.to_dtype(DType::BF16)?;
        let context =
            read(&format!("{prefix}-context.f32"), &[1, 8, 4096])?.to_dtype(DType::BF16)?;
        let expected = read(
            if end_frame {
                "end-transformer-output.f32"
            } else if conditioned_tokens == 0 {
                "transformer-output.f32"
            } else {
                "i2v-transformer-output.f32"
            },
            &[1, tokens, 128],
        )?;
        let store = Store::open(std::path::Path::new(&weights), 0)?;
        let mut model = Transformer::new(store, &dev, 26 << 30, true, false)?;
        let rope = Rope::video_with_end(if end_frame { 2 } else { 1 }, 4, 4, 24, end_frame, &dev)?;
        let actual = model
            .forward(
                &x,
                &context,
                0.725,
                &rope,
                conditioned_tokens,
                end_tokens,
                None,
                |_| {},
            )?
            .0
            .to_dtype(DType::F32)?;
        if end_frame {
            let first = model.first_hidden.as_ref().unwrap().to_dtype(DType::F32)?;
            let reference = read("end-first-block.f32", &[1, tokens, 4096])?;
            let error = ((first - &reference)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?
                / reference.sqr()?.mean_all()?.to_scalar::<f32>()?)
            .sqrt();
            println!("First video block relative RMS error: {error}");
            assert!(error < 0.005, "first block relative RMS {error}");
        }
        let mut hidden_error = 0.;
        if conditioned_tokens > 0 {
            let hidden = model.last_hidden.as_ref().unwrap().to_dtype(DType::F32)?;
            let reference = read(
                if end_frame {
                    "end-hidden.f32"
                } else {
                    "i2v-hidden.f32"
                },
                &[1, tokens, 4096],
            )?;
            let relative = ((hidden - &reference)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?
                / reference.sqr()?.mean_all()?.to_scalar::<f32>()?)
            .sqrt();
            println!("Final hidden-state relative RMS error: {relative}");
            hidden_error = relative;
        }
        // Clean-frame velocities are discarded by masked Euler sampling.
        let actual = actual.narrow(
            1,
            conditioned_tokens,
            tokens - conditioned_tokens - end_tokens,
        )?;
        let expected = expected.narrow(
            1,
            conditioned_tokens,
            tokens - conditioned_tokens - end_tokens,
        )?;
        let error = (&actual - &expected)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        println!(
            "Complete video transformer relative RMS error: {}",
            error / scale
        );
        // LTX 2.5's learned marker changes the activation distribution. Its
        // BF16 backends accumulate more error across 48 layers; separately
        // require the first conditioned block to agree within 0.5% above.
        let keyframe_embedding = model
            .global
            .tensors
            .contains_key("keyframes_abs_pos_embedding");
        let (hidden_limit, velocity_limit) = if end_frame && keyframe_embedding {
            (0.05, 0.12)
        } else {
            (0.04, if conditioned_tokens == 0 { 0.05 } else { 0.10 })
        };
        assert!(
            hidden_error < hidden_limit,
            "conditioned hidden-state relative RMS {hidden_error}"
        );
        assert!(
            // Mixed zero/noisy timesteps amplify small BF16 backend differences
            // in the final projection; also bound the pre-projection state above.
            error / scale < velocity_limit,
            "full transformer relative RMS error {}",
            error / scale
        );
        Ok(())
    }
    fn golden() -> Result<(std::path::PathBuf, String, Device)> {
        let root = std::path::PathBuf::from(std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?);
        let weights = std::env::var("NROB_LTX_CHECKPOINT").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(std::env::var("NROB_LTX_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        Ok((root, weights, dev))
    }
    fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual.to_dtype(DType::F32)?;
        let error = (&actual - expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        Ok(error / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
    }
    #[test]
    #[ignore = "requires tools/ltx/audio_reference.py --part transformer output; NROB_LTX_GOLDEN, NROB_LTX_CHECKPOINT"]
    fn audio_video_blocks_match_reference() -> Result<()> {
        let (root, weights, dev) = golden()?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let x = read("av-video-input.f32", &[1, 16, 128])?.to_dtype(DType::BF16)?;
        let context = read("av-video-context.f32", &[1, 8, 4096])?.to_dtype(DType::BF16)?;
        let ax = read("av-audio-input.f32", &[1, 6, 128])?.to_dtype(DType::BF16)?;
        let actx = read("av-audio-context.f32", &[1, 8, 2048])?.to_dtype(DType::BF16)?;
        let store = Store::open(std::path::Path::new(&weights), 0)?;
        let mut model = Transformer::new(store, &dev, 26 << 30, false, true)?;
        // The reference ran the first two blocks.
        model.layers = 2;
        let rope = Rope::video_with_end(1, 4, 4, 24, false, &dev)?;
        let video_cross = Rope::video_cross(1, 4, 4, 24, false, &dev)?;
        let audio_rope = Rope::audio(6, &dev)?;
        let audio = AudioInput { latent: &ax, context: &actx, rope: &audio_rope, video_cross: &video_cross, sigma: 0.725, isolated: false };
        let (video, audio) = model.forward(&x, &context, 0.725, &rope, 4, 0, Some(audio), |_| {})?;
        let (first_video, first_audio) = (model.first_hidden.clone().unwrap(), model.first_audio_hidden.clone().unwrap());
        // The same pass with the audio frozen as a condition (sigma 0 throughout).
        let frozen = AudioInput { latent: &ax, context: &actx, rope: &audio_rope, video_cross: &video_cross, sigma: 0., isolated: false };
        let (frozen_video, frozen_audio) = model.forward(&x, &context, 0.725, &rope, 4, 0, Some(frozen), |_| {})?;
        let checks = [
            ("first block, video", first_video, read("av-block0-video.f32", &[1, 16, 4096])?, 0.01),
            ("first block, audio", first_audio, read("av-block0-audio.f32", &[1, 6, 2048])?, 0.01),
            // Clean conditioning tokens' velocities are discarded by the sampler.
            ("video velocity", video.narrow(1, 4, 12)?, read("av-video-output.f32", &[1, 16, 128])?.narrow(1, 4, 12)?, 0.03),
            ("audio velocity", audio.unwrap(), read("av-audio-output.f32", &[1, 6, 128])?, 0.03),
            ("video velocity, frozen audio", frozen_video.narrow(1, 4, 12)?, read("av-frozen-video-output.f32", &[1, 16, 128])?.narrow(1, 4, 12)?, 0.03),
            ("audio output, frozen audio", frozen_audio.unwrap(), read("av-frozen-audio-output.f32", &[1, 6, 128])?, 0.03),
        ];
        let mut failed = Vec::new();
        for (name, actual, expected, limit) in checks {
            let e = relative(&actual, &expected)?;
            println!("{name}: relative RMS error {e}");
            if e >= limit {
                failed.push(format!("{name} {e}"));
            }
        }
        assert!(failed.is_empty(), "{failed:?}");
        Ok(())
    }
    #[test]
    #[ignore = "requires tools/ltx/audio_reference.py --part transformer output; NROB_LTX_GOLDEN, NROB_LTX_CHECKPOINT"]
    fn audio_connector_matches_reference() -> Result<()> {
        let (root, weights, dev) = golden()?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let mut store = Store::open(std::path::Path::new(&weights), 0)?;
        let features = read("audio-connector-input.f32", &[1, 12, 2048])?.to_dtype(DType::BF16)?;
        let out = connector(&mut store, &features, "audio_embeddings_connector", &dev, |_| {})?;
        let e = relative(&out, &read("audio-connector-output.f32", &[1, 1024, 2048])?)?;
        println!("audio connector relative RMS error: {e}");
        assert!(e < 0.01, "audio connector {e}");
        Ok(())
    }
    #[test]
    #[ignore = "requires LTX checkpoint and reference activations; NROB_LTX_GOLDEN"]
    fn video_block_ram_and_ssd_match_reference() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
        );
        let weights = std::env::var("NROB_LTX_CHECKPOINT").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(
            std::env::var("NROB_LTX_TEST_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )?;
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let x = read("dit-input.f32", &[1, 16, 4096])?.to_dtype(DType::BF16)?;
        let context = read("dit-context.f32", &[1, 8, 4096])?.to_dtype(DType::BF16)?;
        let m = read("dit-modulation.f32", &[1, 1, 9, 4096])?.to_dtype(DType::BF16)?;
        let pm = read("dit-prompt.f32", &[1, 2, 4096])?.to_dtype(DType::BF16)?;
        let rope = Rope::video(1, 4, 4, 24, &dev)?;
        let expected = read("dit-output.f32", &[1, 16, 4096])?;
        let mut previous: Option<Tensor> = None;
        for budget in [0, 1 << 30] {
            let mut store = Store::open(std::path::Path::new(&weights), budget)?;
            let prefix = format!("{PREFIX}transformer_blocks.0.");
            for repeat in 0..2 {
                let before = store.disk_bytes;
                let w = store.group(&prefix, &dev, true, video_weight)?;
                assert_eq!(store.group_bytes(&prefix, video_weight)?, w.bytes);
                if repeat == 1 && budget > 0 {
                    assert_eq!(
                        before, store.disk_bytes,
                        "RAM cache must avoid re-reading weights"
                    );
                }
                if budget == 0 {
                    assert_eq!(store.host_bytes, 0);
                    assert!(store.disk_bytes > before);
                }
                let y =
                    block(&w, x.clone(), &context, &m, Some(&pm), &rope)?.to_dtype(DType::F32)?;
                let error = (&y - &expected)?
                    .sqr()?
                    .mean_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                assert!(
                    error / scale < 0.025,
                    "DiT relative RMS error {}",
                    error / scale
                );
                if let Some(p) = &previous {
                    assert_eq!((&y - p)?.abs()?.max_all()?.to_scalar::<f32>()?, 0.);
                }
                previous = Some(y);
            }
        }
        Ok(())
    }
}
impl Transformer {
    /// `audio`: load (and run) the audio stream too. Its weights add about 44%
    /// to every block, and the budget below counts them.
    pub fn new(mut store: Store, dev: &Device, gpu_budget: u64, require_gpu: bool, audio: bool) -> Result<Self> {
        let filter = if audio { av_weight } else { video_weight };
        let mut required = 0;
        for i in 0..48 {
            required += store.group_bytes(&format!("{PREFIX}transformer_blocks.{i}."), filter)?;
        }
        if require_gpu && required > gpu_budget {
            candle_core::bail!("video transformer needs {:.1} GiB of weight VRAM; budget is {:.1} GiB. Use auto, ram or ssd", required as f64 / 1073741824., gpu_budget as f64 / 1073741824.);
        }
        let global = store.group(PREFIX, dev, false, |k| {
            k.starts_with("patchify_proj.")
                || k.starts_with("adaln_single.")
                || k.starts_with("prompt_adaln_single.")
                || k.starts_with("proj_out.")
                || k == "scale_shift_table"
                || k == "keyframes_abs_pos_embedding"
                || (audio
                    && (k.starts_with("audio_patchify_proj.")
                        || k.starts_with("audio_adaln_single.")
                        || k.starts_with("audio_prompt_adaln_single.")
                        || k.starts_with("audio_proj_out.")
                        || k == "audio_scale_shift_table"
                        || k.starts_with("av_ca_")))
        })?;
        Ok(Self {
            store,
            global,
            resident: (0..48).map(|_| None).collect(),
            audio,
            layers: 48,
            gpu_bytes: 0,
            gpu_budget,
            require_gpu,
            #[cfg(test)]
            last_hidden: None,
            #[cfg(test)]
            first_hidden: None,
            #[cfg(test)]
            first_audio_hidden: None,
        })
    }
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.gpu_bytes, self.store.host_bytes, self.store.disk_bytes)
    }
    /// Video velocity, and audio velocity when `audio` is given (the model
    /// must have been built with its audio stream).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        latent: &Tensor,
        context: &Tensor,
        sigma: f64,
        rope: &Rope,
        conditioned_tokens: usize,
        end_tokens: usize,
        audio: Option<AudioInput>,
        mut progress: impl FnMut(usize),
    ) -> Result<(Tensor, Option<Tensor>)> {
        let dev = latent.device();
        if audio.is_some() && !self.audio {
            candle_core::bail!("this transformer was loaded without its audio stream");
        }
        let tokens = latent.dim(1)?;
        if conditioned_tokens + end_tokens >= tokens {
            candle_core::bail!("conditioning must leave generated video tokens");
        }
        let (mut modulation, mut embedded) =
            time_embedding(&self.global, "adaln_single", sigma, dev)?;
        if conditioned_tokens > 0 || end_tokens > 0 {
            // Only two distinct timesteps: clean endpoint images and noisy video.
            let (clean_modulation, clean_embedded) =
                time_embedding(&self.global, "adaln_single", 0., dev)?;
            let expand = |clean: &Tensor, noisy: &Tensor| -> Result<Tensor> {
                let mut parts = Vec::new();
                if conditioned_tokens > 0 {
                    parts.push(clean.broadcast_as((1, conditioned_tokens, clean.dim(2)?))?);
                }
                parts.push(noisy.broadcast_as((
                    1,
                    tokens - conditioned_tokens - end_tokens,
                    noisy.dim(2)?,
                ))?);
                if end_tokens > 0 {
                    parts.push(clean.broadcast_as((1, end_tokens, clean.dim(2)?))?);
                }
                Tensor::cat(&parts, 1)
            };
            modulation = expand(&clean_modulation, &modulation)?;
            embedded = expand(&clean_embedded, &embedded)?;
        }
        // Audio: every timestep is the audio's scalar sigma. The video tokens'
        // cross-attention scale/shift follows their own (per-token) timesteps;
        // each gate follows the other stream's sigma.
        let mut audio_state = None;
        if let Some(a) = &audio {
            let (mut video_ss, _) = time_embedding(&self.global, "av_ca_video_scale_shift_adaln_single", sigma, dev)?;
            if conditioned_tokens > 0 || end_tokens > 0 {
                let (clean, _) = time_embedding(&self.global, "av_ca_video_scale_shift_adaln_single", 0., dev)?;
                let mut parts = Vec::new();
                if conditioned_tokens > 0 {
                    parts.push(clean.broadcast_as((1, conditioned_tokens, clean.dim(2)?))?);
                }
                parts.push(video_ss.broadcast_as((1, tokens - conditioned_tokens - end_tokens, video_ss.dim(2)?))?);
                if end_tokens > 0 {
                    parts.push(clean.broadcast_as((1, end_tokens, clean.dim(2)?))?);
                }
                video_ss = Tensor::cat(&parts, 1)?;
            }
            let (audio_modulation, audio_embedded) = time_embedding(&self.global, "audio_adaln_single", a.sigma, dev)?;
            let prompt = if self.global.tensors.contains_key("audio_prompt_adaln_single.linear.weight") {
                Some(time_embedding(&self.global, "audio_prompt_adaln_single", a.sigma, dev)?.0.reshape((1, 2, 2048))?)
            } else {
                None
            };
            let step = AudioStep {
                context: a.context,
                modulation: audio_modulation.reshape((1, 1, 9, 2048))?,
                prompt,
                video_ss: video_ss.reshape((1, video_ss.dim(1)?, 4, 4096))?,
                audio_ss: time_embedding(&self.global, "av_ca_audio_scale_shift_adaln_single", a.sigma, dev)?.0.reshape((1, 1, 4, 2048))?,
                video_gate: time_embedding(&self.global, "av_ca_a2v_gate_adaln_single", a.sigma, dev)?.0,
                audio_gate: time_embedding(&self.global, "av_ca_v2a_gate_adaln_single", sigma, dev)?.0,
                rope: a.rope,
                video_cross: a.video_cross,
                isolated: a.isolated,
            };
            let ax = self.global.linear("audio_patchify_proj", a.latent)?;
            audio_state = Some((step, ax, audio_embedded));
        }
        let modulation = modulation.reshape((1, modulation.dim(1)?, 9, 4096))?;
        let prompt = if self
            .global
            .tensors
            .contains_key("prompt_adaln_single.linear.weight")
        {
            Some(
                time_embedding(&self.global, "prompt_adaln_single", sigma, dev)?
                    .0
                    .reshape((1, 2, 4096))?,
            )
        } else {
            None
        };
        let mut x = self.global.linear("patchify_proj", latent)?;
        if let Some(embedding) = self.global.tensors.get("keyframes_abs_pos_embedding") {
            // LTX 2.5 marks the first video latent (one pixel frame). Supplied
            // end-keyframe tokens remain unmarked in the official conditioner.
            let n = rope.first_frame_tokens;
            if n > 0 {
                let first = x.narrow(1, 0, n)?.broadcast_add(embedding)?;
                x = if n == tokens {
                    first
                } else {
                    Tensor::cat(&[first, x.narrow(1, n, tokens - n)?], 1)?
                };
            }
        }
        let filter = if self.audio { av_weight } else { video_weight };
        for i in 0..self.layers {
            let temporary;
            let w = if let Some(w) = self.resident[i].as_ref() {
                w
            } else {
                let prefix = format!("{PREFIX}transformer_blocks.{i}.");
                let size = self.store.group_bytes(&prefix, filter)?;
                let keep = self.gpu_bytes + size <= self.gpu_budget && fits_on_device(dev, size)?;
                temporary = self.store.group(&prefix, dev, !keep, filter)?;
                if keep {
                    self.gpu_bytes += temporary.bytes;
                    self.resident[i] = Some(temporary);
                    self.resident[i]
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("resident block missing".into()))?
                } else {
                    if self.require_gpu {
                        candle_core::bail!("GPU weight budget exceeded");
                    }
                    &temporary
                }
            };
            match audio_state.as_mut() {
                Some((step, ax, _)) => {
                    let (v, a) = av_block(w, x, ax.clone(), context, &modulation, prompt.as_ref(), rope, step)?;
                    x = v;
                    *ax = a;
                }
                None => x = block(w, x, context, &modulation, prompt.as_ref(), rope)?,
            }
            #[cfg(test)]
            if i == 0 {
                self.first_hidden = Some(x.clone());
                self.first_audio_hidden = audio_state.as_ref().map(|(_, ax, _)| ax.clone());
            }
            progress(i + 1);
        }
        #[cfg(test)]
        {
            self.last_hidden = Some(x.clone());
        }
        let video = self.output("scale_shift_table", "proj_out", &x, &embedded)?;
        let audio = match audio_state {
            Some((_, ax, embedded)) => Some(self.output("audio_scale_shift_table", "audio_proj_out", &ax, &embedded)?),
            None => None,
        };
        Ok((video, audio))
    }
    /// LayerNorm, the (shift, scale) of `table` plus the timestep embedding, projection.
    fn output(&self, table: &str, proj: &str, x: &Tensor, embedded: &Tensor) -> Result<Tensor> {
        let m = self
            .global
            .get(table)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_add(&embedded.unsqueeze(2)?)?;
        self.global.linear(
            proj,
            &affine(
                &layer_norm(x)?,
                &m.narrow(2, 0, 1)?.squeeze(2)?,
                &m.narrow(2, 1, 1)?.squeeze(2)?,
            )?,
        )
    }
}
