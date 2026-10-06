//! Sparse voxel tensors for Pixal3D (TRELLIS.2's `SparseTensor`, one batch):
//! voxel coordinates on the CPU, features on the device, and the operations its
//! decoders are made of.
//!
//! A submanifold 3×3×3 convolution is computed from the output's side: each
//! voxel's 27 neighbours (in kernel order, a missing one pointing at a zero row)
//! are gathered into one [N, 27·Cin] matrix and multiplied by the kernel as one
//! [27·Cin, Cout] matrix, in chunks of voxels so memory stays bounded. The
//! neighbour table is built once per level (coordinates) and shared by every
//! convolution on it. A dense grid is the same thing with every voxel present,
//! so the dense decoder's `Conv3d` (zero padding) uses it too.
use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;
use std::sync::Arc;

/// The voxels of one level and their neighbours, for every convolution on it.
pub struct Level {
    pub coords: Vec<[i32; 3]>,
    /// The grid's size along each axis.
    pub res: usize,
    /// For each voxel, its 27 neighbours' rows in kernel order (x, then y, then z
    /// offset from -1 to 1); `coords.len()` (the zero row) where there is none.
    neighbours: Tensor,
}

fn key(c: [i32; 3]) -> u64 {
    ((c[0] as u64 & 0x1f_ffff) << 42) | ((c[1] as u64 & 0x1f_ffff) << 21) | (c[2] as u64 & 0x1f_ffff)
}

impl Level {
    pub fn new(coords: Vec<[i32; 3]>, res: usize, dev: &Device) -> Result<Arc<Self>> {
        let n = coords.len();
        let mut index: HashMap<u64, u32> = HashMap::with_capacity(n * 2);
        for (i, c) in coords.iter().enumerate() {
            index.insert(key(*c), i as u32);
        }
        let mut table = vec![n as u32; n * 27];
        for (i, c) in coords.iter().enumerate() {
            let mut k = 0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let p = [c[0] + dx, c[1] + dy, c[2] + dz];
                        if p.iter().all(|&v| v >= 0 && (v as usize) < res) {
                            if let Some(&j) = index.get(&key(p)) {
                                table[i * 27 + k] = j;
                            }
                        }
                        k += 1;
                    }
                }
            }
        }
        Ok(Arc::new(Self { neighbours: Tensor::from_vec(table, n * 27, dev)?, coords, res }))
    }

    /// Every voxel of an `res`³ grid, x slowest (as a dense tensor's flattening).
    pub fn dense(res: usize, dev: &Device) -> Result<Arc<Self>> {
        let mut coords = Vec::with_capacity(res * res * res);
        for x in 0..res as i32 {
            for y in 0..res as i32 {
                for z in 0..res as i32 {
                    coords.push([x, y, z]);
                }
            }
        }
        Self::new(coords, res, dev)
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Rows of voxels gathered per matmul: [rows·27, Cin] must fit comfortably.
fn chunk_rows(cin: usize) -> usize {
    ((1usize << 28) / (27 * cin * 2)).clamp(1024, 1 << 20)
}

/// A 3×3×3 submanifold convolution's weights, as one matrix.
pub struct Conv {
    /// [27·Cin, Cout].
    w: Tensor,
    b: Option<Tensor>,
    cin: usize,
}

impl Conv {
    /// From sparse (flex_gemm) layout [Cout, 3, 3, 3, Cin].
    pub fn sparse(w: &Tensor, b: Option<Tensor>, dtype: DType) -> Result<Self> {
        let (co, kd, kh, kw, ci) = w.dims5()?;
        if (kd, kh, kw) != (3, 3, 3) {
            candle_core::bail!("only 3×3×3 sparse convolutions are supported");
        }
        let w = w.to_dtype(dtype)?.reshape((co, 27 * ci))?.t()?.contiguous()?;
        Ok(Self { w, b: b.map(|b| b.to_dtype(dtype)).transpose()?, cin: ci })
    }

    /// From PyTorch `Conv3d` layout [Cout, Cin, 3, 3, 3].
    pub fn dense(w: &Tensor, b: Option<Tensor>, dtype: DType) -> Result<Self> {
        let w = w.permute((0, 2, 3, 4, 1))?.contiguous()?;
        Self::sparse(&w, b, dtype)
    }

    pub fn forward(&self, level: &Level, x: &Tensor) -> Result<Tensor> {
        let n = level.len();
        let (rows, cin) = x.dims2()?;
        if rows != n || cin != self.cin {
            candle_core::bail!("sparse conv: {rows}×{cin} features for {n} voxels and {} input channels", self.cin);
        }
        let x = x.to_dtype(self.w.dtype())?;
        let padded = Tensor::cat(&[&x, &Tensor::zeros((1, cin), x.dtype(), x.device())?], 0)?;
        let step = chunk_rows(cin);
        let mut outs = Vec::with_capacity(n.div_ceil(step));
        let mut at = 0;
        while at < n {
            let m = step.min(n - at);
            let ids = level.neighbours.narrow(0, at * 27, m * 27)?;
            let g = padded.index_select(&ids, 0)?.reshape((m, 27 * cin))?;
            outs.push(g.matmul(&self.w)?);
            at += m;
        }
        let y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// LayerNorm over channels in F32 (TRELLIS's LayerNorm32), back in the input's dtype.
pub fn layer_norm(x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    if let Some(w) = w {
        y = y.broadcast_mul(&w.to_dtype(DType::F32)?)?;
    }
    if let Some(b) = b {
        y = y.broadcast_add(&b.to_dtype(DType::F32)?)?;
    }
    y.to_dtype(dtype)
}

/// The children each voxel is split into (8 flags per voxel, bit i = offset along axis i).
pub struct Subdivision {
    pub parent: Vec<u32>,
    pub child: Vec<u8>,
}

impl Subdivision {
    /// Children whose logit is positive, in voxel order then child order.
    pub fn from_logits(logits: &Tensor) -> Result<Self> {
        let flags: Vec<Vec<f32>> = logits.to_dtype(DType::F32)?.to_vec2()?;
        let mut parent = Vec::new();
        let mut child = Vec::new();
        for (i, row) in flags.iter().enumerate() {
            for (s, &v) in row.iter().enumerate() {
                if v > 0. {
                    parent.push(i as u32);
                    child.push(s as u8);
                }
            }
        }
        Ok(Self { parent, child })
    }

    /// The children's coordinates at the next (twice as fine) level.
    pub fn coords(&self, level: &Level) -> Vec<[i32; 3]> {
        self.coords_of(&level.coords)
    }

    /// [`Self::coords`] of a level's voxels at `coords`.
    pub fn coords_of(&self, coords: &[[i32; 3]]) -> Vec<[i32; 3]> {
        self.parent
            .iter()
            .zip(&self.child)
            .map(|(&p, &s)| {
                let c = coords[p as usize];
                [c[0] * 2 + (s & 1) as i32, c[1] * 2 + ((s >> 1) & 1) as i32, c[2] * 2 + ((s >> 2) & 1) as i32]
            })
            .collect()
    }

    /// Channel-to-space: a parent's features [N, 8·C] are its children's [C] each.
    pub fn channel_to_space(&self, x: &Tensor) -> Result<Tensor> {
        let (n, c8) = x.dims2()?;
        let rows: Vec<u32> = self.parent.iter().zip(&self.child).map(|(&p, &s)| p * 8 + s as u32).collect();
        let ids = Tensor::from_vec(rows, self.parent.len(), x.device())?;
        x.reshape((n * 8, c8 / 8))?.index_select(&ids, 0)
    }
}
