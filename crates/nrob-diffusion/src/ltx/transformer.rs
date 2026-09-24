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
}
impl Rope {
    pub fn positions(positions: &[Vec<f32>], maxima: &[f32], dev: &Device) -> Result<Self> {
        let axes = maxima.len();
        if axes == 0 || positions.iter().any(|p| p.len() != axes) {
            candle_core::bail!("invalid LTX rotary coordinates");
        }
        let count = 4096 / (2 * axes);
        let pad = 2048 - count * axes;
        let mut cos = Vec::with_capacity(positions.len() * 2048);
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
        let shape = (1, positions.len(), 32, 64);
        Ok(Self {
            cos: Tensor::from_vec(cos, shape, dev)?
                .transpose(1, 2)?
                .to_dtype(DType::BF16)?,
            sin: Tensor::from_vec(sin, shape, dev)?
                .transpose(1, 2)?
                .to_dtype(DType::BF16)?,
        })
    }
    pub fn video(
        frames: usize,
        height: usize,
        width: usize,
        fps: usize,
        dev: &Device,
    ) -> Result<Self> {
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
        Self::positions(&positions, &[20., 2048., 2048.], dev)
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let a = x.narrow(3, 0, 64)?;
        let b = x.narrow(3, 64, 64)?;
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
    if let Some(r) = rope {
        q = r.apply(&q)?;
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
pub fn ff(w: &Group, x: &Tensor) -> Result<Tensor> {
    let h = w.linear("ff.net.0.proj", x)?;
    // PyTorch computes activation internals in float32, including for BF16 inputs.
    w.linear(
        "ff.net.2",
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
pub fn connector(
    store: &mut Store,
    features: &Tensor,
    dev: &Device,
    mut progress: impl FnMut(usize),
) -> Result<Tensor> {
    let prefix = format!("{PREFIX}video_embeddings_connector.");
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
        let h = ff(&w, &norm(&x)?)?;
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
fn block(
    w: &Group,
    mut x: Tensor,
    context: &Tensor,
    modulation: &Tensor,
    prompt: Option<&Tensor>,
    rope: &Rope,
) -> Result<Tensor> {
    let m = modulation.broadcast_add(w.get("scale_shift_table")?)?;
    let slot = |j| m.narrow(2, j, 1)?.squeeze(2);
    let h = affine(&norm(&x)?, &slot(0)?, &slot(1)?)?;
    x = (x + attn(w, "attn1", &h, &h, Some(rope))?.broadcast_mul(&slot(2)?)?)?;
    let mut pm = w.get("prompt_scale_shift_table")?.unsqueeze(0)?;
    if let Some(p) = prompt {
        pm = (pm + p)?;
    }
    let h = affine(&norm(&x)?, &slot(6)?, &slot(7)?)?;
    let c = affine(context, &pm.narrow(1, 0, 1)?, &pm.narrow(1, 1, 1)?)?;
    x = (x + attn(w, "attn2", &h, &c, None)?.broadcast_mul(&slot(8)?)?)?;
    let h = affine(&norm(&x)?, &slot(3)?, &slot(4)?)?;
    x = (x + ff(w, &h)?.broadcast_mul(&slot(5)?)?)?;
    Ok(x)
}

pub struct Transformer {
    store: Store,
    global: Group,
    resident: Vec<Option<Group>>,
    pub gpu_bytes: u64,
    gpu_budget: u64,
    require_gpu: bool,
    #[cfg(test)]
    last_hidden: Option<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the complete official transformer reference; NROB_LTX_GOLDEN"]
    fn full_video_transformer_matches_reference() -> Result<()> {
        full_reference(0)
    }
    #[test]
    #[ignore = "requires official per-token timestep reference; NROB_LTX_GOLDEN"]
    fn image_conditioned_transformer_matches_reference() -> Result<()> {
        full_reference(4)
    }
    fn full_reference(conditioned_tokens: usize) -> Result<()> {
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
        let prefix = if conditioned_tokens == 0 {
            "transformer"
        } else {
            "i2v-transformer"
        };
        let x = read(&format!("{prefix}-input.f32"), &[1, 16, 128])?.to_dtype(DType::BF16)?;
        let context =
            read(&format!("{prefix}-context.f32"), &[1, 8, 4096])?.to_dtype(DType::BF16)?;
        let expected = read(
            if conditioned_tokens == 0 {
                "transformer-output.f32"
            } else {
                "i2v-transformer-output.f32"
            },
            &[1, 16, 128],
        )?;
        let store = Store::open(std::path::Path::new(&weights), 0)?;
        let mut model = Transformer::new(store, &dev, 26 << 30, true)?;
        let rope = Rope::video(1, 4, 4, 24, &dev)?;
        let actual = model
            .forward(&x, &context, 0.725, &rope, conditioned_tokens, |_| {})?
            .to_dtype(DType::F32)?;
        if conditioned_tokens > 0 {
            let hidden = model.last_hidden.as_ref().unwrap().to_dtype(DType::F32)?;
            let reference = read("i2v-hidden.f32", &[1, 16, 4096])?;
            let relative = ((hidden - &reference)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?
                / reference.sqr()?.mean_all()?.to_scalar::<f32>()?)
            .sqrt();
            assert!(
                relative < 0.04,
                "conditioned hidden-state relative RMS {relative}"
            );
        }
        // Clean-frame velocities are discarded by masked Euler sampling.
        let actual = actual.narrow(1, conditioned_tokens, 16 - conditioned_tokens)?;
        let expected = expected.narrow(1, conditioned_tokens, 16 - conditioned_tokens)?;
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
        assert!(
            // Mixed zero/noisy timesteps amplify small BF16 backend differences
            // in the final projection; also bound the pre-projection state above.
            error / scale < if conditioned_tokens == 0 { 0.05 } else { 0.10 },
            "full transformer relative RMS error {}",
            error / scale
        );
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
    pub fn new(mut store: Store, dev: &Device, gpu_budget: u64, require_gpu: bool) -> Result<Self> {
        let mut required = 0;
        for i in 0..48 {
            required +=
                store.group_bytes(&format!("{PREFIX}transformer_blocks.{i}."), video_weight)?;
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
        })?;
        Ok(Self {
            store,
            global,
            resident: (0..48).map(|_| None).collect(),
            gpu_bytes: 0,
            gpu_budget,
            require_gpu,
            #[cfg(test)]
            last_hidden: None,
        })
    }
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.gpu_bytes, self.store.host_bytes, self.store.disk_bytes)
    }
    pub fn forward(
        &mut self,
        latent: &Tensor,
        context: &Tensor,
        sigma: f64,
        rope: &Rope,
        conditioned_tokens: usize,
        mut progress: impl FnMut(usize),
    ) -> Result<Tensor> {
        let dev = latent.device();
        let tokens = latent.dim(1)?;
        if conditioned_tokens >= tokens {
            candle_core::bail!("conditioning must leave generated video tokens");
        }
        let (mut modulation, mut embedded) =
            time_embedding(&self.global, "adaln_single", sigma, dev)?;
        if conditioned_tokens > 0 {
            // Only two distinct timesteps: clean first frame and noisy video.
            let (clean_modulation, clean_embedded) =
                time_embedding(&self.global, "adaln_single", 0., dev)?;
            let expand = |clean: &Tensor, noisy: &Tensor| -> Result<Tensor> {
                Tensor::cat(
                    &[
                        clean.broadcast_as((1, conditioned_tokens, clean.dim(2)?))?,
                        noisy.broadcast_as((1, tokens - conditioned_tokens, noisy.dim(2)?))?,
                    ],
                    1,
                )
            };
            modulation = expand(&clean_modulation, &modulation)?;
            embedded = expand(&clean_embedded, &embedded)?;
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
        for i in 0..48 {
            let temporary;
            let w = if let Some(w) = self.resident[i].as_ref() {
                w
            } else {
                let prefix = format!("{PREFIX}transformer_blocks.{i}.");
                let size = self.store.group_bytes(&prefix, video_weight)?;
                let keep = self.gpu_bytes + size <= self.gpu_budget;
                temporary = self.store.group(&prefix, dev, !keep, video_weight)?;
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
            x = block(w, x, context, &modulation, prompt.as_ref(), rope)?;
            progress(i + 1);
        }
        #[cfg(test)]
        {
            self.last_hidden = Some(x.clone());
        }
        let m = self
            .global
            .get("scale_shift_table")?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_add(&embedded.unsqueeze(2)?)?;
        self.global.linear(
            "proj_out",
            &affine(
                &layer_norm(&x)?,
                &m.narrow(2, 0, 1)?.squeeze(2)?,
                &m.narrow(2, 1, 1)?.squeeze(2)?,
            )?,
        )
    }
}
