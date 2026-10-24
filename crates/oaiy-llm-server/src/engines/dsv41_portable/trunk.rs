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

/// What a [`WgpuChained`] multiplies: an EXL3 projection on the adapter (`WgpuBackend::exl3`), or a matrix of BF16
/// values there as the checkpoint's bytes lie (an EXL3 checkpoint's other matrices: its router and indexer weights,
/// stored as f16, that bf16 holds exactly; most of its `wo_b` and shared experts' `w2`), counted in the weight budget.
enum Mat {
    Exl3(Arc<dyn ggml_rs::exl3::PackedLinear>),
    Bf16(ggml_rs_wgpu::chain::WeightVec),
}

/// A matrix of a chained recording on the WebGPU adapter, as the CPU model's [`DenseKernel`]: it gives every row of its
/// output, of which the rows asked for are taken. Its activation is taken as it is (an EXL3 projection's sums are its own
/// transforms', f16 where exllamav3 rounds), so it is no fp8 weight. Its calls, and those of every other such matrix of
/// the same call, go through one chained recording on its adapter (`ggml_rs_wgpu::exl3::chained`): the transforms on the
/// device, a step's buffers kept, a prompt on the tensor cores; a BF16 router, `wo_b` or shared `w2` beside the EXL3
/// projections there (each kind its own call before, a round trip more for each a layer).
pub(super) struct WgpuChained {
    mat: Mat,
    n: usize,
    k: usize,
    b: WgpuBackend,
}

impl WgpuChained {
    /// `rows` of each of `t` rows of `all` (the matrix's whole output).
    fn rows_of(&self, all: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
        let mut y = Vec::with_capacity(t * rows.len());
        for tt in 0..t {
            y.extend_from_slice(&all[tt * self.n + rows.start..tt * self.n + rows.end]);
        }
        y
    }

    fn chain_mat(&self) -> ggml_rs_wgpu::exl3::ChainMat<'_> {
        match &self.mat {
            Mat::Exl3(lin) => ggml_rs_wgpu::exl3::ChainMat::Exl3(&**lin),
            Mat::Bf16(w) => ggml_rs_wgpu::exl3::ChainMat::Bf16 { w: &w.v, n: self.n, k: self.k },
        }
    }
}

impl DenseKernel for WgpuChained {
    fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
        if let Some(all) = ggml_rs_wgpu::exl3::chained(&self.b, &[(self.chain_mat(), x)]).and_then(|mut v| v.pop()) {
            return self.rows_of(&all, t, rows);
        }
        let Mat::Exl3(lin) = &self.mat else { panic!("a BF16 matrix of {} rows the device would not multiply", t) };
        let all = lin.linear(&ggml_rs::Tensor::from_vec(x.to_vec(), vec![t, self.k])).to_host();
        self.rows_of(all.data(), t, rows)
    }

    /// The call's matrices on this adapter in one submit and one read back (`ggml_rs_wgpu::exl3::chained`, else, all
    /// EXL3, `exl3_many`).
    fn forward_many(&self, items: &[(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)]) -> Option<Vec<Vec<f32>>> {
        let mine: Vec<&WgpuChained> = items.iter().map(|(k, ..)| (*k as &dyn std::any::Any).downcast_ref::<WgpuChained>()).collect::<Option<_>>()?;
        let mats: Vec<(ggml_rs_wgpu::exl3::ChainMat<'_>, &[f32])> = mine.iter().zip(items).map(|(w, (_, x, ..))| (w.chain_mat(), *x)).collect();
        let all = ggml_rs_wgpu::exl3::chained(&self.b, &mats).or_else(|| {
            let lins: Vec<(&dyn ggml_rs::exl3::PackedLinear, &[f32])> =
                mine.iter().zip(items).map(|(w, (_, x, ..))| match &w.mat { Mat::Exl3(l) => Some((&**l, *x)), Mat::Bf16(_) => None }).collect::<Option<_>>()?;
            ggml_rs_wgpu::exl3::exl3_many(&lins)
        })?;
        Some(all.iter().zip(&mine).zip(items).map(|((all, w), (_, _, t, rows))| w.rows_of(all, *t, rows.clone())).collect())
    }
}

/// Put `model`'s dense trunk on `gpu` while its budget holds it: how many matrices went, and their stored bytes. An EXL3
/// checkpoint's BF16 matrices (its router and indexer weights, stored as f16, among them) go into its EXL3 projections'
/// chained recordings, so every call of its trunk is one round trip whatever kinds it holds; every fp8 checkpoint's
/// matrices go to the dense kernels.
pub(crate) fn offload(model: &mut crate::dsv41::model::Model, gpu: &WgpuBackend) -> (usize, u64) {
    let exl3 = matches!(model.expert_format(), crate::dsv41::expert::ExpertFormat::Exl3 { .. });
    model.offload(placer(gpu, exl3))
}

/// DSpark's drafter's dense weights on `gpu`, as [`offload`] puts the trunk's; `exl3`: its checkpoint is EXL3's.
pub(crate) fn offload_drafter(drafter: &mut crate::dsv41::dspark::Drafter, gpu: &WgpuBackend, exl3: bool) -> (usize, u64) {
    drafter.offload(placer(gpu, exl3))
}

/// What puts a dense weight on `gpu` ([`offload`]).
fn placer(gpu: &WgpuBackend, exl3: bool) -> impl FnMut(&str, &Weight) -> Option<Weight> + '_ {
    move |_name, w| {
        if let Weight::Exl3 { data, n, k, .. } = w {
            let lin = gpu.exl3(crate::dsv41::linear::exl3_copy(data)).ok()?;
            return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::Exl3(lin), n: *n, k: *k, b: gpu.clone() }), n: *n, k: *k, fp8: false });
        }
        if let (true, Weight::Bf16 { w, n, k }) = (exl3, &*w) {
            if *k % 2 == 0 && *n <= 65535 {
                let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
                let m = gpu.weight_of_bytes(&bytes)?;
                return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::Bf16(m), n: *n, k: *k, b: gpu.clone() }), n: *n, k: *k, fp8: false });
            }
        }
        let (data, n, k, fp8) = match w {
            Weight::Fp8 { w, s, n, k } => {
                let scales = s.iter().map(|&b| oaiy_engine::formats::e8m0_to_f32(b)).collect();
                (DenseData::Fp8 { w: w.clone(), scales, n: *n, k: *k }, *n, *k, true)
            }
            Weight::Bf16 { w, n, k } => (DenseData::Bf16 { w: w.clone(), n: *n, k: *k }, *n, *k, false),
            _ => return None,
        };
        let placed = gpu.dense(data).ok().flatten()?;
        Some(Weight::Device { kernel: Arc::new(WgpuDense(placed)), n, k, fp8 })
    }
}
