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

impl WgpuBackend {
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
                    assert!(worst <= size * allowed, "{dtype}, {rows} rows by {what}: {worst} of {size}");
                }
            }
        }
    }
}
