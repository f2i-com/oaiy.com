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
    fn forward_after(&self, first: &[(&dyn DenseKernel, &[f32], std::ops::Range<usize>)], t: usize, quantized: bool) -> Option<Vec<f32>> {
        if t != 1 {
            return None;
        }
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
    F32(ggml_rs_wgpu::chain::WeightVec),
    /// A stack's parts together, multiplied only by a program's step (`Step::MulGroup`): an EXL3 checkpoint's, or a
    /// GGUF's.
    Group(ggml_rs_wgpu::exl3::Exl3Group),
    QuantGroup(ggml_rs_wgpu::quant::moe::QuantGroup),
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
            Mat::F32(w) => ggml_rs_wgpu::exl3::ChainMat::F32 { w: &w.v, n: self.n, k: self.k },
            Mat::Group(_) | Mat::QuantGroup(_) => panic!("a stack's group is multiplied by a program's step alone"),
        }
    }
}

impl DenseKernel for WgpuChained {
    fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
        if let Some(all) = ggml_rs_wgpu::exl3::chained(&self.b, &[(self.chain_mat(), x)]).and_then(|mut v| v.pop()) {
            return self.rows_of(&all, t, rows);
        }
        let Mat::Exl3(lin) = &self.mat else { panic!("a matrix of floats of {} rows the device would not multiply", t) };
        let all = lin.linear(&ggml_rs::Tensor::from_vec(x.to_vec(), vec![t, self.k])).to_host();
        self.rows_of(all.data(), t, rows)
    }

    /// The call's matrices on this adapter in one submit and one read back (`ggml_rs_wgpu::exl3::chained`, else, all
    /// EXL3, `exl3_many`).
    fn forward_many(&self, items: &[(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)]) -> Option<Vec<Vec<f32>>> {
        let mine: Vec<&WgpuChained> = items.iter().map(|(k, ..)| self.here(*k)).collect::<Option<_>>()?;
        let mats: Vec<(ggml_rs_wgpu::exl3::ChainMat<'_>, &[f32])> = mine.iter().zip(items).map(|(w, (_, x, ..))| (w.chain_mat(), *x)).collect();
        let all = ggml_rs_wgpu::exl3::chained(&self.b, &mats).or_else(|| {
            let lins: Vec<(&dyn ggml_rs::exl3::PackedLinear, &[f32])> =
                mine.iter().zip(items).map(|(w, (_, x, ..))| match &w.mat { Mat::Exl3(l) => Some((&**l, *x)), Mat::Bf16(_) | Mat::F32(_) | Mat::Group(_) | Mat::QuantGroup(_) => None }).collect::<Option<_>>()?;
            ggml_rs_wgpu::exl3::exl3_many(&lins)
        })?;
        Some(all.iter().zip(&mine).zip(items).map(|((all, w), (_, _, t, rows))| w.rows_of(all, *t, rows.clone())).collect())
    }

    /// The first matrices and this one in one chained recording, the rows between them rounded on the device
    /// (`ggml_rs_wgpu::exl3::chained_after`): an EXL3 checkpoint's `wo_a` groups then its `wo_b`.
    fn forward_after(&self, first: &[(&dyn DenseKernel, &[f32], std::ops::Range<usize>)], _t: usize, quantized: bool) -> Option<Vec<f32>> {
        if quantized {
            return None;
        }
        let mine: Vec<&WgpuChained> = first.iter().map(|(k, ..)| self.here(*k)).collect::<Option<_>>()?;
        let with: Vec<(ggml_rs_wgpu::exl3::ChainMat<'_>, &[f32], std::ops::Range<usize>)> = mine.iter().zip(first).map(|(w, (_, x, rows))| (w.chain_mat(), *x, rows.clone())).collect();
        ggml_rs_wgpu::exl3::chained_after(&self.b, &with, self.chain_mat())
    }

    /// Both stages in one chained recording, the norm between them on the device to the host's bit
    /// (`ggml_rs_wgpu::exl3::chained_normed`): a layer's projections of x, then its queries and its indexer's.
    fn forward_normed(&self, first: &[(&dyn DenseKernel, &[f32], usize)], normed: usize, weight: &[f32], eps: f32, then: &[&dyn DenseKernel]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let firsts: Vec<(ggml_rs_wgpu::exl3::ChainMat<'_>, &[f32])> = first.iter().map(|(k, x, _)| self.here(*k).map(|w| (w.chain_mat(), *x))).collect::<Option<_>>()?;
        let thens: Vec<ggml_rs_wgpu::exl3::ChainMat<'_>> = then.iter().map(|k| self.here(*k).map(WgpuChained::chain_mat)).collect::<Option<_>>()?;
        ggml_rs_wgpu::exl3::chained_normed(&self.b, &firsts, normed, weight, eps, &thens)
    }

    /// The program as one chained recording on this adapter (`ggml_rs_wgpu::exl3::run_program`), each matrix one of
    /// these on it.
    fn run_program(&self, steps: &[crate::dsv41::linear::Step<'_>], reads: &[(usize, usize, usize)]) -> Option<Vec<Vec<f32>>> {
        ggml_rs_wgpu::exl3::run_program(&self.b, &self.program(steps)?, reads)
    }

    /// The program as two chained recordings on this adapter, one after the other (`ggml_rs_wgpu::exl3::
    /// run_program_parts`).
    fn run_program_parts(&self, steps: &[crate::dsv41::linear::Step<'_>], reads: &[(usize, usize, usize)], split: usize, later: &[(usize, usize, usize)]) -> Option<(Vec<Vec<f32>>, crate::dsv41::linear::Later<'_>)> {
        let (now, then) = ggml_rs_wgpu::exl3::run_program_parts(&self.b, &self.program(steps)?, reads, split, later)?;
        Some((now, Box::new(move || then.wait())))
    }
}

impl WgpuChained {
    /// `steps` as this adapter's program's steps, each matrix one of these on it; None where one is not.
    fn program<'s>(&self, steps: &[crate::dsv41::linear::Step<'s>]) -> Option<Vec<ggml_rs_wgpu::exl3::ProgramStep<'s>>> {
        use crate::dsv41::linear::Step as S;
        use ggml_rs_wgpu::exl3::ProgramStep as G;
        // (a matrix on another card the program refuses: a split trunk's layers are each on their own)
        fn on_device<'w>(w: &'w Weight, b: &WgpuBackend) -> Option<&'w WgpuChained> {
            let Weight::Device { kernel, .. } = w else { return None };
            chained_of(&**kernel).filter(|m| m.b.same_device(b))
        }
        steps
            .iter()
            .map(|s| {
                Some(match *s {
                    S::Input(x) => G::Input(x),
                    S::Scratch(len) => G::Scratch(len),
                    S::Mul { w, x, rows } => G::Mul { m: on_device(w, &self.b)?.chain_mat(), x, rows },
                    S::MulGroup { group, x, jobs } => match &on_device(group, &self.b)?.mat {
                        Mat::Group(g) => G::MulGroup { g: ggml_rs_wgpu::exl3::GroupMat::Exl3(g), x, jobs },
                        Mat::QuantGroup(g) => G::MulGroup { g: ggml_rs_wgpu::exl3::GroupMat::Quant(g), x, jobs },
                        _ => return None,
                    },
                    S::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => G::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows },
                    S::Norm { x, weight, eps, rows, width } => G::Norm { x, weight, eps, rows, width },
                    S::HcPost { out, res, mix, d } => G::HcPost { out, res, mix, d },
                    S::HcPre { h, mix, d } => G::HcPre { h, mix, d },
                    S::SwigluBf16 { gate, up, len, silu, limit } => G::SwigluBf16 { gate, up, len, silu, limit },
                    S::Candidates { logits, bias, held, layer, k } => G::Candidates { logits, bias, table: held.downcast_ref::<super::exl3::WgpuExl3Experts>()?.fused_parts(layer)?.1, k },
                    S::Held { held, layer, x, cands, limit } => G::Held { slots: held.downcast_ref::<super::exl3::WgpuExl3Experts>()?.fused_parts(layer)?.0, x, cands, limit },
                    S::State(v) => G::State(v),
                    S::Rope { x, rows, stride, at, half, cos, sin, pos, word, inverse } => G::Rope { x, rows, stride, at, half, cos, sin, pos, word, inverse },
                    S::Fp8 { x, len } => G::Fp8 { x, len },
                    S::Fp4 { x, len, block, e4m3 } => G::Fp4 { x, len, block, e4m3 },
                    S::PutRow { src, dst, width, pos, word } => G::PutRow { src, dst, width, pos, word },
                    S::Pool { kv, score, ratio, width } => G::Pool { kv, score, ratio, width },
                    S::IndexScores { q, keys, weights, heads, dim, scale, pos, word } => G::IndexScores { q, keys, weights, heads, dim, scale, pos, word },
                    S::TopK { scores, dst, k, pos, word } => G::TopK { scores, dst, k, pos, word },
                    S::Attend { q, heads, dim, window, win, compressed, sink, scale, pos, word } => G::Attend { q, heads, dim, window, win, compressed, sink, scale, pos, word },
                    S::HcMix { proj, pos, word, base, scale, n, norm_eps, iters, hc_eps } => G::HcMix { proj, pos, word, base, scale, n, norm_eps, iters, hc_eps },
                })
            })
            .collect()
    }
}

/// `k` as a matrix of a chained recording, where it is one.
fn chained_of(k: &dyn DenseKernel) -> Option<&WgpuChained> {
    (k as &dyn std::any::Any).downcast_ref()
}

impl WgpuChained {
    /// `k` as a matrix of a chained recording on this one's card, where it is one.
    fn here<'a>(&self, k: &'a dyn DenseKernel) -> Option<&'a WgpuChained> {
        chained_of(k).filter(|m| m.b.same_device(&self.b))
    }
}

/// Put `model`'s dense trunk on `gpu` while its budget holds it: how many matrices went, and their stored bytes. An EXL3
/// checkpoint's BF16 matrices (its router and indexer weights, stored as f16, among them) go into its EXL3 projections'
/// chained recordings, so every call of its trunk is one round trip whatever kinds it holds; every fp8 checkpoint's
/// matrices go to the dense kernels.
pub(crate) fn offload(model: &mut crate::dsv41::model::Model, gpu: &WgpuBackend) -> (usize, u64) {
    // (an EXL3 or a GGUF checkpoint: every matrix a chained recording's)
    let exl3 = !matches!(model.expert_format(), crate::dsv41::expert::ExpertFormat::Fp4);
    model.offload(placer(gpu, exl3))
}

/// [`offload`] over `cards`, a share of the layers each (`home`: each layer's card, [`layer_homes`]), the rest (the
/// head) on the first: a layer's calls go to the card that holds its weights and its experts, each card's share of a
/// step's work its own (where the second card held experts alone, its share of a step was a few calls a layer, and
/// it spent the step in a low power state, its memory clocked at a seventeenth of its rate).
pub(crate) fn offload_split(model: &mut crate::dsv41::model::Model, cards: &[Arc<WgpuBackend>], home: &[usize]) -> (usize, u64) {
    // (an EXL3 or a GGUF checkpoint: every matrix a chained recording's)
    let exl3 = !matches!(model.expert_format(), crate::dsv41::expert::ExpertFormat::Fp4);
    let mut placers: Vec<_> = cards.iter().map(|c| placer(c, exl3)).collect();
    model.offload(|name, w| {
        let layer = name.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok());
        let card = layer.and_then(|l| home.get(l).copied()).unwrap_or(0).min(placers.len() - 1);
        (placers[card])(name, w)
    })
}

/// Each of `layers` layers' card of `cards`: the layers in runs of equal length, the first run the first card's.
pub(crate) fn layer_homes(layers: usize, cards: usize) -> Vec<usize> {
    (0..layers).map(|l| l * cards.max(1) / layers.max(1)).collect()
}

/// DSpark's drafter's dense weights on `gpu`, as [`offload`] puts the trunk's; `exl3`: its checkpoint is EXL3's.
pub(crate) fn offload_drafter(drafter: &mut crate::dsv41::dspark::Drafter, gpu: &WgpuBackend, exl3: bool) -> (usize, u64) {
    drafter.offload(placer(gpu, exl3))
}

/// What puts a dense weight on `gpu` ([`offload`]).
pub(super) fn placer(gpu: &WgpuBackend, exl3: bool) -> impl FnMut(&str, &Weight) -> Option<Weight> + '_ {
    move |_name, w| {
        // (a stack of EXL3 projections of one shape, an EXL3 checkpoint's `wo_a`: its parts together as well; a GGUF's of
        // one type and shape alike, through the experts' kernels: OAIY_DSV41_GGUF_GROUP=0, each part its own matmul)
        if let Weight::Stacked { parts, rows, k, .. } = w {
            if let Some(Weight::Ggml { dtype, n, k: pk, .. }) = parts.first() {
                if std::env::var("OAIY_DSV41_GGUF_GROUP").as_deref() == Ok("0") {
                    return None;
                }
                let bytes: Vec<&[u8]> = parts.iter().map(|p| if let Weight::Ggml { dtype: t, bytes, n: pn, k: kk } = p { (t == dtype && (pn, kk) == (n, pk)).then_some(&bytes[..]) } else { None }).collect::<Option<_>>()?;
                let group = gpu.quant_group(*dtype, &bytes, *n, *pk)?;
                return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::QuantGroup(group), n: *rows, k: *k, b: gpu.clone() }), n: *rows, k: *k, fp8: false });
            }
            let datas: Vec<&ggml_rs::exl3::Exl3Data> = parts.iter().map(|p| if let Weight::Exl3 { data, .. } = p { Some(&**data) } else { None }).collect::<Option<_>>()?;
            let group = gpu.exl3_group(&datas)?;
            return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::Group(group), n: *rows, k: *k, b: gpu.clone() }), n: *rows, k: *k, fp8: false });
        }
        if let Weight::Exl3 { data, n, k, .. } = w {
            let lin = gpu.exl3(crate::dsv41::linear::exl3_copy(data)).ok()?;
            return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::Exl3(lin), n: *n, k: *k, b: gpu.clone() }), n: *n, k: *k, fp8: false });
        }
        // (a GGUF's matrix: its blocks as a quantized matrix there, which a chained recording multiplies as it does an
        // EXL3 projection; a type with no kernel there as f16)
        if let Weight::Ggml { dtype, bytes, n, k } = w {
            let lin = gpu.quant_linear(ggml_rs::QuantizedTensor::from_bytes_cpu(bytes.to_vec(), vec![*n, *k], *dtype)).ok()?;
            return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::Exl3(lin), n: *n, k: *k, b: gpu.clone() }), n: *n, k: *k, fp8: false });
        }
        if let (true, Weight::Bf16 { w, n, k }) = (exl3, &*w) {
            if *k % 2 == 0 && *n <= 65535 {
                let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
                let m = gpu.weight_of_bytes(&bytes)?;
                return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::Bf16(m), n: *n, k: *k, b: gpu.clone() }), n: *n, k: *k, fp8: false });
            }
        }
        // (and its matrices of f32 values, the hyper-connections' mixing projections among them: a decode step's are made
        // in the sublayer's own call there, where the host read 157 MB of them a token)
        if let (true, Weight::F32 { w, n, k }) = (exl3, &*w) {
            if *k > 0 && *n <= 65535 {
                let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
                let m = gpu.weight_of_bytes(&bytes)?;
                return Some(Weight::Device { kernel: Arc::new(WgpuChained { mat: Mat::F32(m), n: *n, k: *k, b: gpu.clone() }), n: *n, k: *k, fp8: false });
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
