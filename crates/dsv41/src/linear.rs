//! Dense weights and the reference `linear()`.
//!
//! Three storage kinds occur in the trunk:
//! - **fp8** e4m3 `[n, k]` with one e8m0 scale per 32x32 tile (attention
//!   projections, shared experts, Engram `wkv`, indexer `wq_b`). The
//!   reference first quantizes the activation to fp8 (per 32, power-of-two
//!   scale), then runs an fp32-accumulating GEMM and returns bf16.
//! - **bf16** `[n, k]` (router, compressor, indexer `weights_proj`/`wk`,
//!   `wo_a` after the reference's own fp8->bf16 conversion, embed, head).
//!   A plain GEMM; the caller says whether the output is rounded to bf16
//!   (`F.linear` in bf16) or kept in f32 (the router and the head upcast).
//! - **f32** (the hyper-connection projections).
//!
//! Weights stay in their stored form in RAM (fp8 is 1 byte/weight) and each
//! output row is dequantized once per call, then dotted with every token's
//! activation, so a prefill of t tokens reads the weight once.
//!
//! A weight can also be held elsewhere ([`Weight::Device`], a GPU's: see
//! [`crate::model::Model::offload`]): the activation is still quantized here and the
//! result rounded here, so only the sums are made there.

use std::sync::{Arc, Mutex};

use oaiy_engine::backend::parallel_rows;
use oaiy_engine::{Error, Result};

use crate::formats::{bf16_to_f32, e8m0_to_f32, fake_quant_fp8, fp8_e4m3_to_f32, to_bf16};
use crate::safetensors::{Dtype, StIndex};

/// fp8 tile edge (weights) and activation block.
pub const FP8_BLOCK: usize = 32;

/// Output dtype of a GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// Rounded to bf16 (the reference's default dtype).
    Bf16,
    /// Left in f32.
    F32,
}

/// A dense weight held elsewhere (on a GPU), which [`Weight::Device`] runs: the f32 sums of `x` (`[t, k]`, already
/// quantized as the reference quantizes it for an fp8 weight) against rows `rows`, `[t, rows.len()]`.
pub trait DenseKernel: Send + Sync + std::any::Any {
    fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32>;

    /// Several kernels' sums in one call of this kernel's kind (a GPU's: one submit, one round trip for all), each
    /// `(kernel, x, t, rows)`'s in order; None when it cannot take them together (another kind among them), and the
    /// caller makes them one at a time.
    fn forward_many(&self, items: &[(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)]) -> Option<Vec<Vec<f32>>> {
        let _ = items;
        None
    }
}

/// Several weights' [`Weight::forward_rows`] at once, each `(weight, x, t, rows, out)`'s result in order and the same
/// as alone. Weights a device holds go in one call when its kernel takes them together: a decode step's dense calls
/// were each a round trip to the GPU, 667 of them a token, a third of them the eight groups of each layer's `wo_a`.
pub fn forward_together(items: &[(&Weight, &[f32], usize, std::ops::Range<usize>, Out)]) -> Vec<Vec<f32>> {
    let kernels: Option<Vec<&dyn DenseKernel>> =
        items.iter().map(|(w, ..)| if let Weight::Device { kernel, .. } = w { Some(&**kernel) } else { None }).collect();
    if let Some(kernels) = kernels.filter(|k| k.len() > 1) {
        let prepared: Vec<(Vec<f32>, Out)> = items
            .iter()
            .map(|(w, x, t, rows, out)| {
                assert_eq!(x.len(), t * w.k(), "linear: input is not [t, k]");
                assert!(rows.end <= w.n(), "linear: row range past the weight");
                w.prepare(x, *out)
            })
            .collect();
        let many: Vec<(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)> =
            kernels.iter().zip(items).zip(&prepared).map(|((k, (_, _, t, rows, _)), (x, _))| (*k, x.as_slice(), *t, rows.clone())).collect();
        let start = std::time::Instant::now();
        if let Some(mut ys) = kernels[0].forward_many(&many) {
            crate::profile::add(crate::profile::Part::DeviceDense, start);
            for (y, (_, out)) in ys.iter_mut().zip(&prepared) {
                if *out == Out::Bf16 {
                    for v in y.iter_mut() {
                        *v = to_bf16(*v);
                    }
                }
            }
            return ys;
        }
    }
    items.iter().map(|(w, x, t, rows, out)| w.forward_rows(x, *t, rows.clone(), *out)).collect()
}

pub enum Weight {
    Fp8 { w: Vec<u8>, s: Vec<u8>, n: usize, k: usize },
    Bf16 { w: Vec<u16>, n: usize, k: usize },
    F32 { w: Vec<f32>, n: usize, k: usize },
    /// Held by `kernel` (a GPU's), with what the forward pass still does here: `fp8` says the activation is quantized
    /// and the result rounded to bf16 as for [`Weight::Fp8`]. It has no host rows.
    Device { kernel: Arc<dyn DenseKernel>, n: usize, k: usize, fp8: bool },
}

impl Weight {
    /// Load `{prefix}.weight` (and `{prefix}.scale` for fp8) as stored.
    pub fn load(idx: &StIndex, prefix: &str) -> Result<Weight> {
        let name = format!("{prefix}.weight");
        let info = idx.info(&name)?;
        let [n, k] = info.shape[..] else {
            return Err(Error::Format(format!("{name}: expected 2-D, got {:?}", info.shape)));
        };
        Ok(match info.dtype {
            Dtype::F8E4M3 => {
                let s = idx.read(&format!("{prefix}.scale"))?;
                if s.len() != n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) {
                    return Err(Error::Format(format!("{prefix}.scale: {} scales for [{n}, {k}]", s.len())));
                }
                Weight::Fp8 { w: idx.read(&name)?, s, n, k }
            }
            Dtype::BF16 => Weight::Bf16 { w: to_u16(&idx.read(&name)?), n, k },
            Dtype::F32 => Weight::F32 { w: idx.read_f32(&name)?, n, k },
            other => return Err(Error::Format(format!("{name}: {other:?} is not a dense weight dtype"))),
        })
    }

    /// Load an fp8 tensor and dequantize it to bf16, as the reference
    /// `convert.py` does for `wo_a` (exact: fp8 x 2^k fits bf16).
    pub fn load_fp8_as_bf16(idx: &StIndex, prefix: &str) -> Result<Weight> {
        match Self::load(idx, prefix)? {
            Weight::Fp8 { w, s, n, k } => {
                let kb = k.div_ceil(FP8_BLOCK);
                let mut out = Vec::with_capacity(n * k);
                for r in 0..n {
                    for c in 0..k {
                        let v = fp8_e4m3_to_f32(w[r * k + c]) * e8m0_to_f32(s[(r / FP8_BLOCK) * kb + c / FP8_BLOCK]);
                        out.push(crate::formats::f32_to_bf16(v));
                    }
                }
                Ok(Weight::Bf16 { w: out, n, k })
            }
            _ => Err(Error::Format(format!("{prefix}: expected fp8"))),
        }
    }

    pub fn n(&self) -> usize {
        match self {
            Weight::Fp8 { n, .. } | Weight::Bf16 { n, .. } | Weight::F32 { n, .. } | Weight::Device { n, .. } => *n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Weight::Fp8 { k, .. } | Weight::Bf16 { k, .. } | Weight::F32 { k, .. } | Weight::Device { k, .. } => *k,
        }
    }

    /// Row `r` dequantized to f32.
    pub fn row(&self, r: usize, dst: &mut [f32]) {
        match self {
            Weight::Fp8 { w, s, k, .. } => {
                let kb = k.div_ceil(FP8_BLOCK);
                let srow = &s[(r / FP8_BLOCK) * kb..];
                for (b, chunk) in w[r * k..(r + 1) * k].chunks(FP8_BLOCK).enumerate() {
                    let sc = e8m0_to_f32(srow[b]);
                    for (j, &q) in chunk.iter().enumerate() {
                        dst[b * FP8_BLOCK + j] = fp8_e4m3_to_f32(q) * sc;
                    }
                }
            }
            Weight::Bf16 { w, k, .. } => {
                for (d, &b) in dst.iter_mut().zip(&w[r * k..(r + 1) * k]) {
                    *d = bf16_to_f32(b);
                }
            }
            Weight::F32 { w, k, .. } => dst.copy_from_slice(&w[r * k..(r + 1) * k]),
            Weight::Device { .. } => panic!("a weight on a device has no host rows (the embedding is never offloaded)"),
        }
    }

    /// The reference `linear(x, W)` for `t` rows of `x` (row-major `[t, k]`),
    /// returning `[t, n]`. fp8 weights always quantize the activation and
    /// return bf16 (`out` is ignored for them, as in the reference).
    pub fn forward(&self, x: &[f32], t: usize, out: Out) -> Vec<f32> {
        self.forward_rows(x, t, 0..self.n(), out)
    }

    /// The activation as this weight takes it, and the rounding of its result: an fp8 weight's quantized and its
    /// result bf16, as the reference does.
    fn prepare(&self, x: &[f32], out: Out) -> (Vec<f32>, Out) {
        match self {
            Weight::Fp8 { .. } | Weight::Device { fp8: true, .. } => (fake_quant_fp8(x, FP8_BLOCK), Out::Bf16),
            _ => (x.to_vec(), out),
        }
    }

    /// [`forward`](Self::forward) restricted to output rows `rows`, giving
    /// `[t, rows.len()]` — one group of a block-diagonal projection (`wo_a`).
    pub fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>, out: Out) -> Vec<f32> {
        let k = self.k();
        assert_eq!(x.len(), t * k, "linear: input is not [t, k]");
        assert!(rows.end <= self.n(), "linear: row range past the weight");
        let (first, n) = (rows.start, rows.len());
        let (xin, out) = self.prepare(x, out);
        let start = std::time::Instant::now();
        if let Weight::Device { kernel, .. } = self {
            let mut y = kernel.forward_rows(&xin, t, rows);
            crate::profile::add(crate::profile::Part::DeviceDense, start);
            if out == Out::Bf16 {
                for v in &mut y {
                    *v = to_bf16(*v);
                }
            }
            return y;
        }
        let y = Mutex::new(vec![0.0f32; t * n]);
        parallel_rows(n, 16, &|b, e| {
            let mut row = vec![0.0f32; k];
            let mut buf = vec![0.0f32; (e - b) * t];
            for o in b..e {
                self.row(first + o, &mut row);
                for tt in 0..t {
                    let xr = &xin[tt * k..(tt + 1) * k];
                    let mut acc = 0.0f32;
                    for (a, w) in xr.iter().zip(&row) {
                        acc += a * w;
                    }
                    buf[(o - b) * t + tt] = if out == Out::Bf16 { to_bf16(acc) } else { acc };
                }
            }
            let mut y = y.lock().unwrap_or_else(|p| p.into_inner());
            for o in b..e {
                for tt in 0..t {
                    y[tt * n + o] = buf[(o - b) * t + tt];
                }
            }
        });
        crate::profile::add(crate::profile::Part::HostDense, start);
        y.into_inner().unwrap_or_else(|p| p.into_inner())
    }
}

/// A 1-D bf16/f32 vector (norm weights, biases, sinks) as f32.
pub fn load_vec(idx: &StIndex, name: &str) -> Result<Vec<f32>> {
    idx.read_f32(name)
}

fn to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_forward_quantizes_activation_and_rounds_to_bf16() {
        // 2 x 32 weight, all 1.0 (0x38), scale 2^0 (127)
        let w = Weight::Fp8 { w: vec![0x38; 64], s: vec![127], n: 2, k: 32 };
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let y = w.forward(&x, 1, Out::F32);
        let expect: f32 = fake_quant_fp8(&x, 32).iter().sum();
        assert_eq!(y, vec![to_bf16(expect); 2]);
    }

    #[test]
    fn bf16_forward_honours_out_dtype() {
        let one = crate::formats::f32_to_bf16(1.0);
        let w = Weight::Bf16 { w: vec![one; 3], n: 1, k: 3 };
        let x = [1.0, 2f32.powi(-9), 2f32.powi(-10)];
        assert_eq!(w.forward(&x, 1, Out::F32)[0], 1.0 + 2f32.powi(-9) + 2f32.powi(-10));
        assert_eq!(w.forward(&x, 1, Out::Bf16)[0], 1.0);
    }
}
