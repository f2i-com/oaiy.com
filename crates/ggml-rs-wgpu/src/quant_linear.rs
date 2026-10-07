//! A GGUF's matrix where a chain takes a packed projection ([`ggml_rs::exl3::PackedLinear`], as an EXL3 one is):
//! Qwen3.8-Flash-Next from a GGUF, its dense matrices in whatever type the file gives each (Q2_0, the K-quants, Q4_0,
//! Q5_0, IQ4_NL, IQ4_XS, Q8_0) on the device as the quantized matmuls read them. No transforms either side of it and
//! no channel maps (EXL3's): a recorder's `exl3_rows` of one is its `matmul_rows`.
use crate::WgpuBackend;
use ggml_rs::exl3::PackedLinear;
use ggml_rs::{Backend, DeviceChain, DeviceVec, QuantizedTensor, Tensor};
use std::sync::Arc;

pub struct QuantLinear {
    backend: WgpuBackend,
    pub(crate) w: QuantizedTensor,
    shape: [usize; 2],
}

impl std::fmt::Debug for QuantLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantLinear({:?} {:?})", self.w.dtype(), self.shape)
    }
}

impl QuantLinear {
    /// Its rows and columns (`[n, k]`: `k` inputs to `n` outputs).
    pub(crate) fn kn(&self) -> (usize, usize) {
        (self.shape[1], self.shape[0])
    }
}

impl PackedLinear for QuantLinear {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.w.nbytes()
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        self.backend.linear_q(x, &self.w)
    }
}

/// A matrix of floats where a chain takes a packed projection: a GGUF's BF16 matrices (Qwen3.8-Flash-Next's
/// indexer's), on the device as f16 two to a word (each value rounded to the nearest), a recorder's `exl3_rows` of
/// one its `matmul_f16_rows`. The host's path multiplies the floats it was made of.
pub struct HalfLinear {
    pub(crate) w: DeviceVec,
    host: Tensor,
    shape: [usize; 2],
    backend: WgpuBackend,
}

impl std::fmt::Debug for HalfLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HalfLinear({:?})", self.shape)
    }
}

impl HalfLinear {
    pub(crate) fn kn(&self) -> (usize, usize) {
        (self.shape[1], self.shape[0])
    }

    pub(crate) fn is_on(&self, gpu: &Arc<crate::Gpu>) -> bool {
        Arc::ptr_eq(&self.backend.gpu, gpu)
    }
}

impl PackedLinear for HalfLinear {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.shape[0] * self.shape[1] * 2
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        self.backend.linear(x, &self.host)
    }
}

/// A packed projection with a low-rank update beside it, where a chain takes a packed projection: a LoRA adapter's
/// `y = W x + B (A x)`, the base as it is (EXL3, a GGUF's blocks, another of these) and `A` (`[r, k]`) and `B` (`[n,
/// r]`, the adapter's scale in it) as f16 matrices on its device. A recorder's `exl3_rows` of one is the base's rows,
/// then the two small products, added to them; the host's path makes the same sums. The base is never rewritten.
pub struct LowRank {
    pub(crate) base: Arc<dyn PackedLinear>,
    pub(crate) a: Arc<dyn PackedLinear>,
    pub(crate) b: Arc<dyn PackedLinear>,
    shape: [usize; 2],
    rank: usize,
}

impl std::fmt::Debug for LowRank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LowRank({:?} rank {} on {:?})", self.shape, self.rank, self.base)
    }
}

impl LowRank {
    /// Its columns and rows (`k` inputs to `n` outputs).
    pub(crate) fn kn(&self) -> (usize, usize) {
        (self.shape[1], self.shape[0])
    }

    /// The update's rank as held (an odd one is one more: f16 values go two to a word).
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }
}

impl PackedLinear for LowRank {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.base.nbytes() + self.a.nbytes() + self.b.nbytes()
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        let y = self.base.linear(x).to_host();
        let update = self.b.linear(&self.a.linear(x)).to_host();
        Tensor::from_vec(y.data().iter().zip(update.data()).map(|(y, u)| y + u).collect(), y.shape().to_vec())
    }
}

impl WgpuBackend {
    /// `base` (`[n, k]`) with the low-rank update `b @ a` beside it (`a`: `[rank, k]`, `b`: `[n, rank]`, row-major, a
    /// LoRA's scale already in one of them), on this device: a packed projection a chain runs. An error where the
    /// shapes do not agree or a value is past f16's range.
    pub fn low_rank(&self, base: Arc<dyn PackedLinear>, mut a: Vec<f32>, mut b: Vec<f32>, mut rank: usize) -> Result<Arc<dyn PackedLinear>, String> {
        let &[n, k] = base.shape() else { return Err(format!("a low-rank update of a projection of shape {:?}", base.shape())) };
        if rank == 0 || a.len() != rank * k || b.len() != n * rank {
            return Err(format!("a rank {rank} update of {} and {} floats beside a projection [{n}, {k}]", a.len(), b.len()));
        }
        // an odd rank gains a row of zeros in A and a column of zeros in B: nothing is added to the product
        if rank % 2 != 0 {
            a.extend(std::iter::repeat_n(0.0, k));
            b = b.chunks_exact(rank).flat_map(|row| row.iter().copied().chain(std::iter::once(0.0))).collect();
            rank += 1;
        }
        let a = self.half_linear(a, rank, k)?;
        let b = self.half_linear(b, n, rank)?;
        Ok(Arc::new(LowRank { base, a, b, shape: [n, k], rank }))
    }

    /// `values` (`[n, k]`, row-major; `k` even) on this device as f16, a packed projection a chain runs. An error
    /// where a value is past f16's range.
    pub fn half_linear(&self, values: Vec<f32>, n: usize, k: usize) -> Result<Arc<dyn PackedLinear>, String> {
        if values.len() != n * k || k % 2 != 0 {
            return Err(format!("a projection of {} floats as [{n}, {k}]", values.len()));
        }
        let w = self.vec_f16_rounded(&values).ok_or_else(|| format!("a projection [{n}, {k}] with a value past f16's range"))?;
        Ok(Arc::new(HalfLinear { w, host: Tensor::from_vec(values, vec![n, k]), shape: [n, k], backend: self.clone() }))
    }

    /// `w` (`[n, k]`, a GGUF's blocks) on this device as a packed projection a chain runs. An error where its type
    /// has no kernel here or the device has no room for it.
    pub fn quant_linear(&self, w: QuantizedTensor) -> Result<Arc<dyn PackedLinear>, String> {
        let &[n, k] = w.shape() else { return Err(format!("a quantized projection of shape {:?}", w.shape())) };
        let dtype = w.dtype();
        if !Self::supports(dtype) {
            return Err(format!("{dtype:?} has no WebGPU kernel"));
        }
        let w = self.to_device_quant(w);
        if !w.is_device() {
            return Err(format!("no room on the device for a {dtype:?} projection [{n}, {k}]"));
        }
        Ok(Arc::new(QuantLinear { backend: self.clone(), w, shape: [n, k] }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_quants::GgmlType;
    use ggml_rs::DeviceChain;

    /// A GGUF matrix as a packed projection is its dequantized weights' product, by the host's path (`linear`) and in
    /// a chain (`exl3_rows`, a step's one row, a check's few and a prompt's many), for the types a Flash-Next GGUF's
    /// dense matrices come in.
    #[test]
    fn a_quantized_projection_in_a_chain_is_its_weights() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (n, k) = (96usize, 512usize);
        for dtype in [GgmlType::Q2_0, GgmlType::Q4_0, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::IQ4_NL, GgmlType::IQ4_XS, GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
            let (elems, bytes) = (dtype.block_size(), dtype.type_size());
            let mut raw: Vec<u8> = (0..n * k / elems * bytes).map(|_| next() as u8).collect();
            // (each block's scales modest and finite: the types' f16 fields)
            let at: &[usize] = match dtype {
                GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
                GgmlType::Q3_K => &[108],
                GgmlType::Q6_K => &[208],
                _ => &[0],
            };
            for block in raw.chunks_exact_mut(bytes) {
                for &o in at {
                    let scale = half::f16::from_f32(0.002 + 0.02 * ((next() >> 40) as f32 / (1u64 << 24) as f32));
                    block[o..o + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                }
            }
            let mut dense = vec![0f32; n * k];
            ggml_quants::dequantize(dtype, &raw, &mut dense).unwrap();
            let w = b.quant_linear(QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype)).unwrap();
            check(&b, w, &dense, &format!("{dtype:?}"), &mut next);
        }
        // a matrix of floats, as f16
        let dense: Vec<f32> = (0..n * k).map(|_| ((next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.05).collect();
        check(&b, b.half_linear(dense.clone(), n, k).unwrap(), &dense, "floats as f16", &mut next);

        fn check(b: &WgpuBackend, w: Arc<dyn PackedLinear>, dense: &[f32], dtype: &str, next: &mut dyn FnMut() -> u64) {
            let (n, k) = (w.shape()[0], w.shape()[1]);
            assert!(DeviceChain::holds_exl3(b, w.as_ref()), "{dtype}: held where a chain reads it");
            for rows in [1usize, 3, 70] {
                let x: Vec<f32> = (0..rows * k).map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
                let want: Vec<f64> = (0..rows * n).map(|i| (0..k).map(|j| dense[(i % n) * k + j] as f64 * x[(i / n) * k + j] as f64).sum()).collect();
                let size = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                let host = w.linear(&Tensor::from_vec(x.clone(), vec![rows, k]));
                let (xv, yv) = (DeviceChain::vec(b, rows * k), DeviceChain::vec(b, rows * n));
                DeviceChain::upload(b, &xv, &x);
                let mut rec = DeviceChain::begin(b);
                rec.exl3_rows(w.as_ref(), &xv, &yv, rows);
                rec.read(&yv);
                let chained = rec.finish().pop().unwrap();
                for (what, got) in [("the host's path", host.data()), ("a chain", &chained[..])] {
                    let worst = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0f64, f64::max);
                    // (a prompt's rows on the tensor cores read the inputs as f16; a check's few rows as int8 where
                    // the type has that kernel, as llama.cpp's MMQ does: some 0.4% of the largest)
                    let allowed = if (2..=8).contains(&rows) { 1.5e-2 } else { 2e-3 };
                    eprintln!("{dtype}, {rows} rows by {what}: the worst error {:.1e} of the largest", worst / size);
                    assert!(worst <= size * allowed, "{dtype}, {rows} rows by {what}: {worst} of {size}");
                }
            }
        }
    }

    /// A projection with a low-rank update beside it (a LoRA adapter's) is `(W + B A) x`, by the host's path and in
    /// a chain: over a matrix of floats and over a quantized one, an even rank and an odd one (which gains a zero),
    /// one over another (two adapters), and under a SwiGLU's product.
    #[test]
    fn a_low_rank_update_beside_a_projection_is_the_sum_of_both() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut seed = 0x0bad_5eed_1234_5677u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (n, k) = (96usize, 512usize);
        let mut floats = |len: usize, scale: f32| -> Vec<f32> { (0..len).map(|_| ((next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale).collect() };
        // `dense + b @ a`, as f64 sums rounded once
        let with = |dense: &[f32], a: &[f32], bm: &[f32], rank: usize| -> Vec<f32> {
            (0..n * k).map(|i| (dense[i] as f64 + (0..rank).map(|r| bm[(i / k) * rank + r] as f64 * a[r * k + i % k] as f64).sum::<f64>()) as f32).collect()
        };
        let mut cases: Vec<(String, Arc<dyn PackedLinear>, Vec<f32>)> = Vec::new();
        let dense = floats(n * k, 0.05);
        for rank in [4usize, 3] {
            let (a, bm) = (floats(rank * k, 0.05), floats(n * rank, 0.5));
            let w = b.low_rank(b.half_linear(dense.clone(), n, k).unwrap(), a.clone(), bm.clone(), rank).unwrap();
            cases.push((format!("floats, rank {rank}"), w, with(&dense, &a, &bm, rank)));
        }
        // two adapters, one over the other
        let (a1, b1, a2, b2) = (floats(2 * k, 0.05), floats(n * 2, 0.5), floats(6 * k, 0.05), floats(n * 6, 0.5));
        let first = b.low_rank(b.half_linear(dense.clone(), n, k).unwrap(), a1.clone(), b1.clone(), 2).unwrap();
        cases.push(("floats, two updates".into(), b.low_rank(first, a2.clone(), b2.clone(), 6).unwrap(), with(&with(&dense, &a1, &b1, 2), &a2, &b2, 6)));
        // a quantized base
        let dtype = GgmlType::Q4_0;
        let (elems, bytes) = (dtype.block_size(), dtype.type_size());
        let mut raw: Vec<u8> = floats(n * k / elems * bytes, 1.0).iter().map(|v| (v * 127.0) as i8 as u8).collect();
        for block in raw.chunks_exact_mut(bytes) {
            block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        }
        let mut quant = vec![0f32; n * k];
        ggml_quants::dequantize(dtype, &raw, &mut quant).unwrap();
        let (a, bm) = (floats(8 * k, 0.05), floats(n * 8, 0.5));
        let w = b.low_rank(b.quant_linear(QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype)).unwrap(), a.clone(), bm.clone(), 8).unwrap();
        cases.push(("Q4_0, rank 8".into(), w, with(&quant, &a, &bm, 8)));

        for (what, w, dense) in &cases {
            assert!(DeviceChain::holds_exl3(&b, w.as_ref()), "{what}: held where a chain reads it");
            for rows in [1usize, 3, 70] {
                let x = floats(rows * k, 1.0);
                let want: Vec<f64> = (0..rows * n).map(|i| (0..k).map(|j| dense[(i % n) * k + j] as f64 * x[(i / n) * k + j] as f64).sum()).collect();
                let size = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                let host = w.linear(&Tensor::from_vec(x.clone(), vec![rows, k]));
                // (the output a work vector longer than these rows, as a model's are: what is past them is left)
                let (xv, yv) = (DeviceChain::vec(&b, rows * k), DeviceChain::vec(&b, rows * n + 5));
                DeviceChain::upload(&b, &xv, &x);
                DeviceChain::upload(&b, &yv, &vec![7.0; rows * n + 5]);
                let mut rec = DeviceChain::begin(&b);
                rec.exl3_rows(w.as_ref(), &xv, &yv, rows);
                rec.read(&yv);
                let chained = rec.finish().pop().unwrap();
                assert!(chained[rows * n..].iter().all(|&v| v == 7.0), "{what}, {rows} rows: the sum went past its rows");
                for (path, got) in [("the host's path", host.data()), ("a chain", &chained[..rows * n])] {
                    let worst = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0f64, f64::max);
                    let allowed = if (2..=8).contains(&rows) { 1.5e-2 } else { 3e-3 };
                    eprintln!("{what}, {rows} rows by {path}: the worst error {:.1e} of the largest", worst / size);
                    assert!(worst <= size * allowed, "{what}, {rows} rows by {path}: {worst} of {size}");
                }
            }
        }
        // several projections in one recording, each's output the next one's input and their scratch shared (a
        // model's layers: a step's one row with its bind groups kept, a check's few rows, a prompt's many)
        let square: Vec<(Arc<dyn PackedLinear>, Vec<f32>)> = (0..3)
            .map(|_| {
                let (w, a, bm) = (floats(k * k, 0.04), floats(4 * k, 0.05), floats(k * 4, 0.5));
                let sum: Vec<f32> = (0..k * k).map(|i| (w[i] as f64 + (0..4).map(|r| bm[(i / k) * 4 + r] as f64 * a[r * k + i % k] as f64).sum::<f64>()) as f32).collect();
                (b.low_rank(b.half_linear(w, k, k).unwrap(), a, bm, 4).unwrap(), sum)
            })
            .collect();
        for rows in [1usize, 3, 70] {
            for keep in [true, false] {
                let x = floats(rows * k, 1.0);
                let mut want: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                for (_, dense) in &square {
                    want = (0..rows * k).map(|i| (0..k).map(|j| dense[(i % k) * k + j] as f64 * want[(i / k) * k + j]).sum()).collect();
                }
                let size = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                let vs: Vec<DeviceVec> = (0..=square.len()).map(|_| DeviceChain::vec(&b, rows * k)).collect();
                DeviceChain::upload(&b, &vs[0], &x);
                // (twice over, as a model's steps follow one another: the second recording meets what the first kept)
                for turn in 0..2 {
                    let mut rec = DeviceChain::begin(&b);
                    rec.keep_groups(keep);
                    for (i, (w, _)) in square.iter().enumerate() {
                        rec.exl3_rows(w.as_ref(), &vs[i], &vs[i + 1], rows);
                    }
                    rec.read(&vs[square.len()]);
                    let got = rec.finish().pop().unwrap();
                    let worst = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0f64, f64::max);
                    eprintln!("three in a row, {rows} rows, groups kept {keep}, turn {turn}: the worst error {:.1e} of the largest", worst / size);
                    assert!(worst <= size * 3e-2, "three in a row, {rows} rows, groups kept {keep}, turn {turn}: {worst} of {size}");
                }
            }
        }
        // the shapes must agree
        assert!(b.low_rank(b.half_linear(dense.clone(), n, k).unwrap(), vec![0.0; 4 * k], vec![0.0; n * 3], 4).is_err());
        assert!(b.low_rank(b.half_linear(dense, n, k).unwrap(), Vec::new(), Vec::new(), 0).is_err());
    }
}
