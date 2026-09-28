//! Native Qwen3-VL vision tower, including three DeepStack feature mergers.
//! Architecture: Qwen/Transformers, Apache-2.0.
use crate::{
    math::{attention, heads, unheads},
    reference::{patch_positions, Reference},
    weights::{Linear, Weights},
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

struct Norm {
    weight: Tensor,
    bias: Tensor,
}
impl Norm {
    fn load(w: &mut Weights, p: &str, d: &Device, ty: DType) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(&format!("{p}.weight"), d, ty)?,
            bias: w.tensor(&format!("{p}.bias"), d, ty)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(&x.contiguous()?, &self.weight, &self.bias, 1e-6)
    }
}
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}
struct Merger {
    norm: Norm,
    fc1: Linear,
    fc2: Linear,
    post: bool,
}
impl Merger {
    fn load(w: &mut Weights, p: &str, post: bool, d: &Device, ty: DType) -> Result<Self> {
        Ok(Self {
            norm: Norm::load(w, &format!("{p}.norm"), d, ty)?,
            fc1: w.linear(&format!("{p}.linear_fc1"), d, ty, &mut crate::lora::Loras::default())?,
            fc2: w.linear(&format!("{p}.linear_fc2"), d, ty, &mut crate::lora::Loras::default())?,
            post,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.elem_count() / 4608;
        let x = if self.post {
            self.norm.forward(&x.reshape((n, 4608))?)?
        } else {
            self.norm.forward(x)?.reshape((n, 4608))?
        };
        self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)
    }
}
pub struct Features {
    pub embedding: Tensor,
    pub deep: Vec<Tensor>,
    pub h: usize,
    pub w: usize,
}
pub struct VisionEncoder {
    patch: Tensor,
    bias: Tensor,
    pos: Tensor,
    blocks: Vec<Block>,
    merger: Merger,
    deep: Vec<Merger>,
    device: Device,
    dtype: DType,
}
impl VisionEncoder {
    pub fn load(root: &Path, d: &Device, ty: DType) -> Result<Self> {
        let mut w = Weights::open(root)?;
        let p = "model.visual";
        let patch = w
            .tensor(&format!("{p}.patch_embed.proj.weight"), d, ty)?
            .reshape((1152, 1536))?;
        let bias = w.tensor(&format!("{p}.patch_embed.proj.bias"), d, ty)?;
        let pos = w.tensor(&format!("{p}.pos_embed.weight"), d, DType::F32)?;
        let mut blocks = Vec::new();
        for i in 0..27 {
            let p = format!("{p}.blocks.{i}");
            blocks.push(Block {
                norm1: Norm::load(&mut w, &format!("{p}.norm1"), d, ty)?,
                norm2: Norm::load(&mut w, &format!("{p}.norm2"), d, ty)?,
                qkv: w.linear(&format!("{p}.attn.qkv"), d, ty, &mut crate::lora::Loras::default())?,
                proj: w.linear(&format!("{p}.attn.proj"), d, ty, &mut crate::lora::Loras::default())?,
                fc1: w.linear(&format!("{p}.mlp.linear_fc1"), d, ty, &mut crate::lora::Loras::default())?,
                fc2: w.linear(&format!("{p}.mlp.linear_fc2"), d, ty, &mut crate::lora::Loras::default())?,
            });
        }
        let merger = Merger::load(&mut w, &format!("{p}.merger"), false, d, ty)?;
        let deep = (0..3)
            .map(|i| {
                Merger::load(
                    &mut w,
                    &format!("{p}.deepstack_merger_list.{i}"),
                    true,
                    d,
                    ty,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            patch,
            bias,
            pos,
            blocks,
            merger,
            deep,
            device: d.clone(),
            dtype: ty,
        })
    }
    pub fn encode(&self, image: &Reference) -> Result<Features> {
        let (h, w) = (image.h / 16, image.w / 16);
        let positions = patch_positions(h, w);
        let mut ids = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut factors = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for &(y, x) in &positions {
            let yy = y as f32 * 47. / (h - 1) as f32;
            let xx = x as f32 * 47. / (w - 1) as f32;
            let (y0, x0) = (yy.floor() as u32, xx.floor() as u32);
            let (dy, dx) = (yy - y0 as f32, xx - x0 as f32);
            for (i, (py, px, f)) in [
                (y0, x0, (1. - dy) * (1. - dx)),
                (y0, (x0 + 1).min(47), (1. - dy) * dx),
                ((y0 + 1).min(47), x0, dy * (1. - dx)),
                ((y0 + 1).min(47), (x0 + 1).min(47), dy * dx),
            ]
            .into_iter()
            .enumerate()
            {
                ids[i].push(py * 48 + px);
                factors[i].push(f);
            }
            for p in [y, x] {
                for j in 0..18 {
                    let a = p as f64 / 10000f64.powf(j as f64 / 18.);
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        let mut pos = Tensor::zeros((h * w, 1152), DType::F32, &self.device)?;
        for i in 0..4 {
            pos = (pos
                + self
                    .pos
                    .index_select(&Tensor::new(ids[i].as_slice(), &self.device)?, 0)?
                    .broadcast_mul(
                        &Tensor::new(factors[i].as_slice(), &self.device)?.unsqueeze(1)?,
                    )?)?;
        }
        let mut x = image
            .patches(&self.device, self.dtype)?
            .matmul(&self.patch.t()?)?
            .broadcast_add(&self.bias)?
            .add(&pos.to_dtype(self.dtype)?)?
            .unsqueeze(0)?;
        let cos = Tensor::from_vec(cos, (h * w, 36), &self.device)?;
        let sin = Tensor::from_vec(sin, (h * w, 36), &self.device)?;
        let mut deep = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            let qkv = b.qkv.forward(&b.norm1.forward(&x)?)?;
            let rotate = |t: Tensor| {
                candle_nn::rotary_emb::rope(&t.to_dtype(DType::F32)?.contiguous()?, &cos, &sin)?
                    .to_dtype(self.dtype)
            };
            let q = rotate(heads(&qkv.narrow(2, 0, 1152)?, 16)?)?;
            let k = rotate(heads(&qkv.narrow(2, 1152, 1152)?, 16)?)?;
            let v = heads(&qkv.narrow(2, 2304, 1152)?, 16)?;
            x = (x + b.proj.forward(&unheads(&attention(&q, &k, &v, 0)?)?)?)?;
            x = (&x
                + b.fc2
                    .forward(&b.fc1.forward(&b.norm2.forward(&x)?)?.gelu()?)?)?;
            if [8, 16, 24].contains(&i) {
                deep.push(self.deep[deep.len()].forward(&x)?.unsqueeze(0)?);
            }
        }
        Ok(Features {
            embedding: self.merger.forward(&x)?.unsqueeze(0)?,
            deep,
            h: h / 2,
            w: w / 2,
        })
    }
}
