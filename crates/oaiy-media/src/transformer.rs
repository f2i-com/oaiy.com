//! Native port of the Qwen Image 2.1 single-stream transformer (Apache-2.0,
//! Qwen/Hugging Face). Text tokens are causal and conditioned at time zero.
use crate::residency::{Budget, Resident, Tiered};
use crate::text::Conditioning;
use crate::{
    lora::Loras,
    math::*,
    weights::{bytes_of, Linear, Weights},
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::{Path, PathBuf};

pub struct Prefix {
    kv: Vec<(Tensor, Tensor)>,
    position: usize,
}

struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: Tensor,
    kn: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Resident for Block {
    fn bytes(&self) -> u64 {
        [&self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down]
            .iter()
            .map(|l| l.bytes())
            .sum::<u64>()
            + bytes_of(&self.qn)
            + bytes_of(&self.kn)
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            q: self.q.to_device(dev)?,
            k: self.k.to_device(dev)?,
            v: self.v.to_device(dev)?,
            o: self.o.to_device(dev)?,
            qn: self.qn.to_device(dev)?,
            kn: self.kn.to_device(dev)?,
            gate: self.gate.to_device(dev)?,
            up: self.up.to_device(dev)?,
            down: self.down.to_device(dev)?,
        })
    }
}
/// Transformer block `i`, read from the published weights onto `device`.
fn load_block(
    w: &mut Weights,
    lora: &mut Loras,
    i: usize,
    device: &Device,
    dtype: DType,
) -> Result<Block> {
    let p = format!("transformer_blocks.{i}");
    let mut linear = |name: &str| w.linear(&format!("{p}.{name}"), device, dtype, lora);
    let (q, k, v, o) = (
        linear("attn.to_q")?,
        linear("attn.to_k")?,
        linear("attn.to_v")?,
        linear("attn.to_out.0")?,
    );
    let (gate, up, down) = (
        linear("img_mlp.gate_layer")?,
        linear("img_mlp.proj")?,
        linear("img_mlp.out")?,
    );
    Ok(Block {
        q,
        k,
        v,
        o,
        gate,
        up,
        down,
        qn: w.tensor(&format!("{p}.attn.norm_q.weight"), device, dtype)?,
        kn: w.tensor(&format!("{p}.attn.norm_k.weight"), device, dtype)?,
    })
}
const BLOCKS: usize = 32;
pub struct Transformer {
    img: Linear,
    text1: Linear,
    text2: Linear,
    text_norm: Tensor,
    time1: Linear,
    time2: Linear,
    modulation: Linear,
    norm_out: Linear,
    out: Linear,
    /// Each block on the GPU, in RAM or streamed from the SSD (see `residency`).
    blocks: Tiered<Block>,
    /// Kept for blocks that are re-read on every pass.
    weights: Weights,
    lora: Loras,
    /// Adapters that fit only in part: what was left out.
    lora_notes: Vec<String>,
    device: Device,
    dtype: DType,
}
impl Transformer {
    /// Reference/text tokens are time-zero conditions. Their per-layer keys and
    /// values are independent of the seed and all six denoising timesteps.
    pub fn prepare(&mut self, text: &Conditioning, refs: &[(Tensor, usize, usize)]) -> Result<Prefix> {
        if text.spans.len() != refs.len() {
            candle_core::bail!("conditioning/reference count mismatch");
        }
        let projected = self.text2.forward(
            &self
                .text1
                .forward(&rms(&text.states, &self.text_norm, 1e-6)?)?
                .gelu()?,
        )?;
        let mut parts = Vec::new();
        let mut segments = Vec::new();
        let mut positions = Vec::new();
        let mut cursor = 0;
        let mut position = 0;
        for (&(start, len), (latent, h, w)) in text.spans.iter().zip(refs) {
            if latent.dim(1)? != h * w || len * 4 != h * w {
                candle_core::bail!("reference latent/vision grid mismatch");
            }
            if start > cursor {
                let n = start - cursor;
                parts.push(projected.narrow(1, cursor, n)?);
                segments.push((n, true));
                positions.extend((position..position + n).map(|p| [p as f64; 3]));
                position += n;
            }
            parts.push(self.img.forward(latent)?);
            segments.push((h * w, false));
            image_positions(&mut positions, position, *h, *w);
            position += h.max(w);
            cursor = start + len;
        }
        let n = projected.dim(1)? - cursor;
        if n > 0 {
            parts.push(projected.narrow(1, cursor, n)?);
            segments.push((n, true));
            positions.extend((position..position + n).map(|p| [p as f64; 3]));
            position += n;
        }
        let mut x = Tensor::cat(&parts, 1)?;
        let (cos, sin) = position_rope(&positions, &self.device, self.dtype)?;
        let time = self.time(0.)?;
        let mods = self
            .modulation
            .forward(&candle_nn::ops::silu(&time)?)?
            .unsqueeze(0)?;
        let mut kv = Vec::new();
        let (device, dtype) = (self.device.clone(), self.dtype);
        for i in 0..BLOCKS {
            let (w, lora) = (&mut self.weights, &mut self.lora);
            let (next, k, v) = self.blocks.with(i, |i| load_block(w, lora, i, &device, dtype), |b| {
                let norm = layer_norm(&x)?.broadcast_mul(&(mods.narrow(2, 0, 4096)? + 1.)?)?;
                let rotate = |v: Tensor| candle_nn::rotary_emb::rope_i(&v.contiguous()?, &cos, &sin);
                let q = rotate(rms(&heads(&b.q.forward(&norm)?, 32)?, &b.qn, 1e-6)?)?;
                let k = rotate(rms(&heads(&b.k.forward(&norm)?, 32)?, &b.kn, 1e-6)?)?;
                let v = heads(&b.v.forward(&norm)?, 32)?;
                let x = (&x + b
                    .o
                    .forward(&unheads(&block_attention(&q, &k, &v, &segments)?)?)?
                    .broadcast_mul(&mods.narrow(2, 4096, 4096)?.tanh()?)?)?;
                let norm = layer_norm(&x)?.broadcast_mul(&(mods.narrow(2, 8192, 4096)? + 1.)?)?;
                let x = (&x + swiglu(&norm, &b.gate, &b.up, &b.down)?
                    .broadcast_mul(&mods.narrow(2, 12288, 4096)?.tanh()?)?)?;
                Ok((x, k, v))
            })?;
            x = next;
            kv.push((k, v));
        }
        Ok(Prefix { kv, position })
    }
    fn time(&self, sigma: f64) -> Result<Tensor> {
        let mut values = Vec::new();
        for kind in 0..2 {
            for j in 0..128 {
                let a = 1000. * sigma / 10000f64.powf(j as f64 / 128.);
                values.push(if kind == 0 {
                    a.cos() as f32
                } else {
                    a.sin() as f32
                });
            }
        }
        self.time2
            .forward(&candle_nn::ops::silu(&self.time1.forward(
                &Tensor::from_vec(values, (1, 256), &self.device)?.to_dtype(self.dtype)?,
            )?)?)
    }
    pub fn conditioned(
        &mut self,
        latent: &Tensor,
        prefix: &Prefix,
        sigma: f64,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let mut positions = Vec::new();
        image_positions(&mut positions, prefix.position, h, w);
        let (cos, sin) = position_rope(&positions, &self.device, self.dtype)?;
        let time = self.time(sigma)?;
        let mods = self
            .modulation
            .forward(&candle_nn::ops::silu(&time)?)?
            .unsqueeze(0)?;
        let mut x = self.img.forward(latent)?;
        let (device, dtype) = (self.device.clone(), self.dtype);
        for (i, (pk, pv)) in prefix.kv.iter().enumerate() {
            let (w, lora) = (&mut self.weights, &mut self.lora);
            x = self.blocks.with(i, |i| load_block(w, lora, i, &device, dtype), |b| {
                let norm = layer_norm(&x)?.broadcast_mul(&(mods.narrow(2, 0, 4096)? + 1.)?)?;
                let rotate = |v: Tensor| candle_nn::rotary_emb::rope_i(&v.contiguous()?, &cos, &sin);
                let q = rotate(rms(&heads(&b.q.forward(&norm)?, 32)?, &b.qn, 1e-6)?)?;
                let k = rotate(rms(&heads(&b.k.forward(&norm)?, 32)?, &b.kn, 1e-6)?)?;
                let v = heads(&b.v.forward(&norm)?, 32)?;
                let k = Tensor::cat(&[pk, &k], 2)?;
                let v = Tensor::cat(&[pv, &v], 2)?;
                let x = (&x + b
                    .o
                    .forward(&unheads(&attention(&q, &k, &v, 0)?)?)?
                    .broadcast_mul(&mods.narrow(2, 4096, 4096)?.tanh()?)?)?;
                let norm = layer_norm(&x)?.broadcast_mul(&(mods.narrow(2, 8192, 4096)? + 1.)?)?;
                &x + swiglu(&norm, &b.gate, &b.up, &b.down)?
                    .broadcast_mul(&mods.narrow(2, 12288, 4096)?.tanh()?)?
            })?;
        }
        let scale = (self
            .norm_out
            .forward(&candle_nn::ops::silu(&time)?)?
            .unsqueeze(0)?
            + 1.)?;
        self.out.forward(&layer_norm(&x)?.broadcast_mul(&scale)?)
    }
    /// `adapter` is the turbo adapter (strength 1); `loras` are more adapters,
    /// each with its strength, applied with it.
    pub fn load(
        path: &Path,
        adapter: Option<&Path>,
        loras: &[(PathBuf, f64)],
        device: &Device,
        dtype: DType,
        budget: &Budget,
        progress: impl FnMut(usize),
    ) -> Result<Self> {
        let mut w = Weights::open(path)?;
        let list: Vec<(PathBuf, f64)> = adapter.map(|a| (a.to_owned(), 1.)).into_iter().chain(loras.iter().cloned()).collect();
        let mut lora = Loras::open(&list)?;
        if adapter.is_some() && !lora.first_adapts("transformer_blocks.0.attn.to_q") {
            candle_core::bail!("not a supported turbo adapter (expected LoRA factors for transformer_blocks.0.attn.to_q)");
        }
        let mut linear = |name: &str| w.linear(name, device, dtype, &mut lora);
        let img = linear("img_in")?;
        let (text1, text2) = (linear("txt_in.in_layer")?, linear("txt_in.out_layer")?);
        let (time1, time2) = (
            linear("time_text_embed.timestep_embedder.linear_1")?,
            linear("time_text_embed.timestep_embedder.linear_2")?,
        );
        let (modulation, norm_out, out) = (
            linear("modulation.1")?,
            linear("norm_out.linear")?,
            linear("proj_out")?,
        );
        let text_norm = (w.tensor("txt_in.text_norm.weight", device, DType::F32)? + 1.)?;
        // The small global projections above always stay on the GPU; only the
        // 32 blocks (nearly all of the weights) are tiered.
        let blocks = Tiered::load(
            BLOCKS,
            budget,
            device,
            |i| load_block(&mut w, &mut lora, i, device, dtype),
            progress,
        )?;
        // Every block has been read once: an adapter that fit nothing is for another model.
        let lora_notes = lora.check()?;
        Ok(Self {
            img,
            text1,
            text2,
            text_norm,
            time1,
            time2,
            modulation,
            norm_out,
            out,
            blocks,
            weights: w,
            lora,
            lora_notes,
            device: device.clone(),
            dtype,
        })
    }

    /// Where the blocks live and how much was streamed, for the job result.
    pub fn residency(&self) -> oaiy_engine::json::Json {
        self.blocks.report()
    }

    /// LoRA adapters that fit only in part, and what of them was left out.
    pub fn lora_notes(&self) -> &[String] {
        &self.lora_notes
    }

    pub fn forward(
        &mut self,
        latents: &Tensor,
        text: &Tensor,
        sigma: f64,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let nt = text.dim(1)?;
        let ni = h * w;
        let text = self.text2.forward(
            &self
                .text1
                .forward(&rms(text, &self.text_norm, 1e-6)?)?
                .gelu()?,
        )?;
        let mut x = Tensor::cat(&[text, self.img.forward(latents)?], 1)?;
        let mut times = Vec::new();
        for t in [0., sigma] {
            for kind in 0..2 {
                for j in 0..128 {
                    let a = 1000. * t / 10000f64.powf(j as f64 / 128.);
                    times.push(if kind == 0 {
                        a.cos() as f32
                    } else {
                        a.sin() as f32
                    });
                }
            }
        }
        let time = self
            .time2
            .forward(&candle_nn::ops::silu(&self.time1.forward(
                &Tensor::from_vec(times, (2, 256), &self.device)?.to_dtype(self.dtype)?,
            )?)?)?;
        let modulation = self.modulation.forward(&candle_nn::ops::silu(&time)?)?;
        let select = |t: &Tensor| -> Result<Tensor> {
            let d = t.dim(1)?;
            Tensor::cat(
                &[
                    t.narrow(0, 0, 1)?.unsqueeze(0)?.broadcast_as((1, nt, d))?,
                    t.narrow(0, 1, 1)?.unsqueeze(0)?.broadcast_as((1, ni, d))?,
                ],
                1,
            )
        };
        let mods = select(&modulation)?;
        let scale1 = (mods.narrow(2, 0, 4096)? + 1.)?;
        let gate1 = mods.narrow(2, 4096, 4096)?.tanh()?;
        let scale2 = (mods.narrow(2, 8192, 4096)? + 1.)?;
        let gate2 = mods.narrow(2, 12288, 4096)?.tanh()?;
        let (cos, sin) = rope(nt, h, w, &self.device, self.dtype)?;
        let (device, dtype) = (self.device.clone(), self.dtype);
        for i in 0..BLOCKS {
            let (weights, lora) = (&mut self.weights, &mut self.lora);
            x = self.blocks.with(i, |i| load_block(weights, lora, i, &device, dtype), |b| {
                let norm = layer_norm(&x)?.mul(&scale1)?;
                let rotate = |v: Tensor| candle_nn::rotary_emb::rope_i(&v.contiguous()?, &cos, &sin);
                let q = rotate(rms(&heads(&b.q.forward(&norm)?, 32)?, &b.qn, 1e-6)?)?;
                let k = rotate(rms(&heads(&b.k.forward(&norm)?, 32)?, &b.kn, 1e-6)?)?;
                let v = heads(&b.v.forward(&norm)?, 32)?;
                let x = (&x + b
                    .o
                    .forward(&unheads(&attention(&q, &k, &v, nt)?)?)?
                    .mul(&gate1)?)?;
                let norm = layer_norm(&x)?.mul(&scale2)?;
                &x + swiglu(&norm, &b.gate, &b.up, &b.down)?.mul(&gate2)?
            })?;
        }
        let scale = (self
            .norm_out
            .forward(&candle_nn::ops::silu(&time.narrow(0, 1, 1)?)?)?
            .unsqueeze(0)?
            + 1.)?;
        self.out
            .forward(&layer_norm(&x.narrow(1, nt, ni)?)?.broadcast_mul(&scale)?)
    }
}

fn image_positions(positions: &mut Vec<[f64; 3]>, position: usize, h: usize, w: usize) {
    for y in 0..h {
        for x in 0..w {
            positions.push([
                position as f64,
                y as f64 - (h - h / 2) as f64,
                x as f64 - (w - w / 2) as f64,
            ]);
        }
    }
}
fn position_rope(positions: &[[f64; 3]], dev: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for pos in positions {
        for (axis, dim) in [16, 56, 56].into_iter().enumerate() {
            for j in 0..dim / 2 {
                let a = pos[axis] / 10000f64.powf((j * 2) as f64 / dim as f64);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (positions.len(), 64), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (positions.len(), 64), dev)?.to_dtype(dtype)?,
    ))
}

fn rope(nt: usize, h: usize, w: usize, dev: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for i in 0..nt + h * w {
        let pos = if i < nt {
            [i as f64; 3]
        } else {
            [
                nt as f64,
                ((i - nt) / w) as f64 - (h - h / 2) as f64,
                ((i - nt) % w) as f64 - (w - w / 2) as f64,
            ]
        };
        for (axis, dim) in [16, 56, 56].into_iter().enumerate() {
            for j in 0..dim / 2 {
                let a = pos[axis] / 10000f64.powf((j * 2) as f64 / dim as f64);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (nt + h * w, 64), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (nt + h * w, 64), dev)?.to_dtype(dtype)?,
    ))
}
