//! Native port of the Qwen Image 2.1 single-stream transformer (Apache-2.0,
//! Qwen/Hugging Face). Text tokens are causal and conditioned at time zero.
use crate::{
    math::*,
    weights::{Linear, Weights},
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

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
    blocks: Vec<Block>,
    device: Device,
    dtype: DType,
}
impl Transformer {
    pub fn load(
        path: &Path,
        adapter: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut w = Weights::open(path)?;
        let mut lora = adapter.map(Weights::open).transpose()?;
        if let Some(l) = &lora {
            if !l.has("transformer.transformer_blocks.0.attn.to_q.lora_A.weight")
                && !l.has("transformer_blocks.0.attn.to_q.lora_A.weight")
            {
                candle_core::bail!(
                    "not a supported Viggle runtime LoRA (expected diffusers lora_A/lora_B keys)"
                );
            }
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
        let mut blocks = Vec::new();
        for i in 0..32 {
            let p = format!("transformer_blocks.{i}");
            let mut linear =
                |name: &str| w.linear(&format!("{p}.{name}"), device, dtype, &mut lora);
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
            blocks.push(Block {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                qn: w.tensor(&format!("{p}.attn.norm_q.weight"), device, dtype)?,
                kn: w.tensor(&format!("{p}.attn.norm_k.weight"), device, dtype)?,
            });
        }
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
            device: device.clone(),
            dtype,
        })
    }

    pub fn forward(
        &self,
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
        for b in &self.blocks {
            let norm = layer_norm(&x)?.mul(&scale1)?;
            let rotate = |v: Tensor| candle_nn::rotary_emb::rope_i(&v.contiguous()?, &cos, &sin);
            let q = rotate(rms(&heads(&b.q.forward(&norm)?, 32)?, &b.qn, 1e-6)?)?;
            let k = rotate(rms(&heads(&b.k.forward(&norm)?, 32)?, &b.kn, 1e-6)?)?;
            let v = heads(&b.v.forward(&norm)?, 32)?;
            x = (x + b
                .o
                .forward(&unheads(&attention(&q, &k, &v, nt)?)?)?
                .mul(&gate1)?)?;
            let norm = layer_norm(&x)?.mul(&scale2)?;
            x = (x + swiglu(&norm, &b.gate, &b.up, &b.down)?.mul(&gate2)?)?;
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
