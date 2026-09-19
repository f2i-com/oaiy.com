//! The vision tower on a GPU: [`dsv41::vision::VisionTower`] (the CPU
//! oracle) with tiled kernels — a bf16-weight GEMM for every linear layer,
//! one-thread-per-query flash attention, the 2-D rotary and the gates. About
//! 1 GB of bf16 weights; a 1,500-patch image takes well under a second.

use cudarc::driver::{CudaSlice, CudaView};
use dsv41::config::{Config, VisionConfig};
use dsv41::safetensors::{Dtype, StIndex};
use dsv41::vision::{self, Prepared};
use nrob::{Error, Result};

use crate::gpu::Gpu;

/// The reference's RMSNorm eps in the tower.
const NORM_EPS: f32 = 1e-6;

/// A bf16 `nn.Linear` on the device (a zero bias when it has none).
struct Lin {
    w: CudaSlice<u16>,
    b: CudaSlice<f32>,
    n: usize,
    k: usize,
}

impl Lin {
    fn load(g: &Gpu, idx: &StIndex, prefix: &str, bias: bool) -> Result<Lin> {
        let name = format!("{prefix}.weight");
        let info = idx.info(&name)?;
        let [n, k] = info.shape[..] else {
            return Err(Error::Format(format!("{name}: expected 2-D, got {:?}", info.shape)));
        };
        if info.dtype != Dtype::BF16 {
            return Err(Error::Format(format!("{name}: expected bf16, got {:?}", info.dtype)));
        }
        let w: Vec<u16> = idx.read(&name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let b = if bias { idx.read_f32(&format!("{prefix}.bias"))? } else { vec![0.0; n] };
        if w.len() != n * k || b.len() != n {
            return Err(Error::Format(format!("{prefix}: size mismatch")));
        }
        Ok(Lin { w: g.upload(&w)?, b: g.upload(&b)?, n, k })
    }

    fn forward(&self, g: &Gpu, x: &CudaView<'_, f32>, t: usize) -> Result<CudaSlice<f32>> {
        let mut y = g.alloc::<f32>(t * self.n)?;
        g.gemm_bf16(x, &self.w, &self.b, &mut y, t, self.n, self.k, true)?;
        Ok(y)
    }

    fn bytes(&self) -> usize {
        self.n * self.k * 2 + self.n * 4
    }
}

struct Block {
    norm1: CudaSlice<f32>,
    wqkv: Lin,
    wo: Lin,
    norm2: CudaSlice<f32>,
    w1: Lin,
    w2: Lin,
}

/// The ViT and aligner on one device, with the span delimiters' embeddings
/// (host side: they go straight into the LLM's input).
pub struct GpuVision {
    pub cfg: VisionConfig,
    patch: Lin,
    blocks: Vec<Block>,
    norm: CudaSlice<f32>,
    aligner: [Lin; 2],
    pub start: Vec<f32>,
    pub newline: Vec<f32>,
    pub end: Vec<f32>,
}

impl GpuVision {
    pub fn load(g: &Gpu, idx: &StIndex, cfg: &Config) -> Result<GpuVision> {
        let vc = cfg.vision.clone().ok_or_else(|| Error::Unsupported("this checkpoint has no vision tower".into()))?;
        if vc.head_dim() != 64 {
            return Err(Error::Unsupported(format!("vision head size {} (the attention kernel does 64)", vc.head_dim())));
        }
        let vec = |name: &str| -> Result<CudaSlice<f32>> { g.upload(&idx.read_f32(name)?) };
        let blocks = (0..vc.n_layers)
            .map(|i| {
                let p = format!("vision.blocks.{i}");
                Ok(Block {
                    norm1: vec(&format!("{p}.norm1.weight"))?,
                    wqkv: Lin::load(g, idx, &format!("{p}.attn.wqkv"), true)?,
                    wo: Lin::load(g, idx, &format!("{p}.attn.wo"), true)?,
                    norm2: vec(&format!("{p}.norm2.weight"))?,
                    w1: Lin::load(g, idx, &format!("{p}.mlp.w1"), false)?,
                    w2: Lin::load(g, idx, &format!("{p}.mlp.w2"), false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GpuVision {
            patch: Lin::load(g, idx, "vision.patch_embed.proj", true)?,
            blocks,
            norm: vec("vision.norm.weight")?,
            aligner: [Lin::load(g, idx, "aligner.w1", true)?, Lin::load(g, idx, "aligner.w2", true)?],
            start: idx.read_f32("image_start")?,
            newline: idx.read_f32("image_newline")?,
            end: idx.read_f32("image_end")?,
            cfg: vc,
        })
    }

    /// Device memory the weights take.
    pub fn bytes(&self) -> usize {
        let blocks: usize = self.blocks.iter().map(|b| b.wqkv.bytes() + b.wo.bytes() + b.w1.bytes() + b.w2.bytes() + 8 * self.cfg.dim).sum();
        blocks + self.patch.bytes() + self.aligner[0].bytes() + self.aligner[1].bytes() + 4 * self.cfg.dim
    }

    /// Patch embedding, `[patches][dim]` on the device.
    pub fn patch_embed(&self, g: &Gpu, img: &Prepared) -> Result<CudaSlice<f32>> {
        let x = g.upload(&img.patches)?;
        self.patch.forward(g, &x.as_view(), img.n_patches())
    }

    /// The image's rotary tables on the device.
    pub fn rope(&self, g: &Gpu, img: &Prepared) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let (c, s) = vision::rope_tables(img.n_vit_h, img.n_vit_w, self.cfg.head_dim() / 2, self.cfg.rope_theta);
        Ok((g.upload(&c)?, g.upload(&s)?))
    }

    /// ViT block `i` over `x` (`[n][dim]`).
    pub fn block(&self, g: &Gpu, i: usize, x: &CudaSlice<f32>, n: usize, rope: &(CudaSlice<f32>, CudaSlice<f32>)) -> Result<CudaSlice<f32>> {
        let b = &self.blocks[i];
        let (d, heads, inter) = (self.cfg.dim, self.cfg.n_heads, self.cfg.inter_dim);
        let mut h = g.alloc::<f32>(n * d)?;
        g.rmsnorm(&x.as_view(), &b.norm1, &mut h.slice_mut(..), n, d, NORM_EPS)?;
        let mut qkv = b.wqkv.forward(g, &h.as_view(), n)?;
        g.vit_rope(&mut qkv, &rope.0, &rope.1, n, heads, self.cfg.head_dim())?;
        let mut att = g.alloc::<f32>(n * d)?;
        g.vit_attn(&qkv, &mut att, n, heads)?;
        let o = b.wo.forward(g, &att.as_view(), n)?;
        let mut x1 = g.alloc::<f32>(n * d)?;
        g.add_round(x, &o.as_view(), &mut x1, n * d)?;
        g.rmsnorm(&x1.as_view(), &b.norm2, &mut h.slice_mut(..), n, d, NORM_EPS)?;
        let gu = b.w1.forward(g, &h.as_view(), n)?;
        let mut m = g.alloc::<f32>(n * inter)?;
        g.silu_mul(&gu, &mut m, inter, n)?;
        let y = b.w2.forward(g, &m.as_view(), n)?;
        let mut out = g.alloc::<f32>(n * d)?;
        g.add_round(&x1, &y.as_view(), &mut out, n * d)?;
        Ok(out)
    }

    /// The ViT's output features, `[patches][dim]` on the device.
    pub fn vit(&self, g: &Gpu, img: &Prepared) -> Result<CudaSlice<f32>> {
        let n = img.n_patches();
        let rope = self.rope(g, img)?;
        let mut x = self.patch_embed(g, img)?;
        for i in 0..self.blocks.len() {
            x = self.block(g, i, &x, n, &rope)?;
        }
        let mut y = g.alloc::<f32>(n * self.cfg.dim)?;
        g.rmsnorm(&x.as_view(), &self.norm, &mut y.slice_mut(..), n, self.cfg.dim, NORM_EPS)?;
        Ok(y)
    }

    /// The aligner over ViT features (host `[patches][dim]`): the image's
    /// LLM rows, `[n_llm_h * n_llm_w][llm dim]` on the host.
    pub fn align(&self, g: &Gpu, feats: &[f32], img: &Prepared) -> Result<Vec<f32>> {
        let r = self.cfg.downsample;
        let u = vision::unfold(feats, img.n_vit_h, img.n_vit_w, self.cfg.dim, r);
        let rows = u.len() / (self.cfg.dim * r * r);
        let u = g.upload(&u)?;
        let mut y = self.aligner[0].forward(g, &u.as_view(), rows)?;
        g.gelu_round(&mut y, rows * self.aligner[0].n)?;
        g.download(&self.aligner[1].forward(g, &y.as_view(), rows)?)
    }

    /// The image's aligner rows (the reference `encode_image`).
    pub fn encode(&self, g: &Gpu, img: &Prepared) -> Result<Vec<f32>> {
        let feats = g.download(&self.vit(g, img)?)?;
        self.align(g, &feats, img)
    }

    /// The whole span's input embeddings, `[n_tokens][llm dim]`.
    pub fn span(&self, g: &Gpu, img: &Prepared) -> Result<Vec<f32>> {
        let aligned = self.encode(g, img)?;
        Ok(vision::span_rows(img, &aligned, &self.start, &self.newline, &self.end))
    }
}
