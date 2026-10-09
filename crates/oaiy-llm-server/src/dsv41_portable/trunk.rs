//! The dense trunk on the WebGPU adapter: its weights as the CPU model's kernels, and their placement.

use super::*;

/// A dense weight on the WebGPU adapter, as the CPU model's [`DenseKernel`].
pub(super) struct WgpuDense(pub(super) Arc<DenseGpu>);

impl DenseKernel for WgpuDense {
    fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
        self.0.forward(x, t, rows)
    }

    /// Weights on this adapter in one submit and one read back.
    fn forward_many(&self, items: &[(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)]) -> Option<Vec<Vec<f32>>> {
        let batch: Option<Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)>> = items
            .iter()
            .map(|(k, x, t, rows)| {
                let w = &(*k as &dyn std::any::Any).downcast_ref::<WgpuDense>()?.0;
                w.same_device(&self.0).then(|| (&**w, *x, *t, rows.clone()))
            })
            .collect();
        Some(ggml_rs_wgpu::dense::forward_batch(&batch?))
    }

    /// The unit and the others on this adapter in one submit, the SwiGLU on the device
    /// ([`ggml_rs_wgpu::dense::forward_units`]).
    fn forward_gated(
        &self,
        up: &dyn DenseKernel,
        down: &dyn DenseKernel,
        x: &[f32],
        limit: f32,
        others: &[(&dyn DenseKernel, &[f32], std::ops::Range<usize>)],
    ) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
        let here = |k: &dyn DenseKernel| -> Option<Arc<DenseGpu>> {
            let w = &(k as &dyn std::any::Any).downcast_ref::<WgpuDense>()?.0;
            w.same_device(&self.0).then(|| Arc::clone(w))
        };
        let (up, down) = (here(up)?, here(down)?);
        let held: Vec<Arc<DenseGpu>> = others.iter().map(|(k, ..)| here(*k)).collect::<Option<_>>()?;
        let with: Vec<(&DenseGpu, &[f32], std::ops::Range<usize>)> = held.iter().zip(others).map(|(w, (_, x, rows))| (&**w, *x, rows.clone())).collect();
        let unit = ggml_rs_wgpu::dense::Unit { gate: &self.0, up: &up, down: &down, x, weight: 1.0 };
        let (mut downs, sums) = ggml_rs_wgpu::dense::forward_units(&[unit], &with, limit)?;
        Some((downs.pop()?, sums))
    }

    /// The projections on this adapter in one submit, the row between them made on the device
    /// ([`ggml_rs_wgpu::dense::forward_chained`]).
    fn forward_after(&self, first: &[(&dyn DenseKernel, &[f32], std::ops::Range<usize>)], quantized: bool) -> Option<Vec<f32>> {
        let held: Vec<Arc<DenseGpu>> = first
            .iter()
            .map(|(k, ..)| {
                let w = &(*k as &dyn std::any::Any).downcast_ref::<WgpuDense>()?.0;
                w.same_device(&self.0).then(|| Arc::clone(w))
            })
            .collect::<Option<_>>()?;
        let with: Vec<(&DenseGpu, &[f32], std::ops::Range<usize>)> = held.iter().zip(first).map(|(w, (_, x, rows))| (&**w, *x, rows.clone())).collect();
        ggml_rs_wgpu::dense::forward_chained(&ggml_rs_wgpu::dense::Chain { first: &with, then: &self.0, quantized })
    }
}

/// Put `model`'s dense trunk on `gpu` while its budget holds it: how many matrices went, and their stored bytes.
pub(crate) fn offload(model: &mut dsv41::model::Model, gpu: &WgpuBackend) -> (usize, u64) {
    model.offload(|_name, w| {
        let (data, n, k, fp8) = match w {
            Weight::Fp8 { w, s, n, k } => {
                let scales = s.iter().map(|&b| dsv41::formats::e8m0_to_f32(b)).collect();
                (DenseData::Fp8 { w: w.clone(), scales, n: *n, k: *k }, *n, *k, true)
            }
            Weight::Bf16 { w, n, k } => (DenseData::Bf16 { w: w.clone(), n: *n, k: *k }, *n, *k, false),
            _ => return None,
        };
        let placed = gpu.dense(data).ok().flatten()?;
        Some(Weight::Device { kernel: Arc::new(WgpuDense(placed)), n, k, fp8 })
    })
}
