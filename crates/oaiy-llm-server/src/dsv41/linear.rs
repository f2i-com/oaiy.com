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
//! - **EXL3** (exllamav3's mul1 trellis, [`Weight::Exl3`]): a checkpoint quantized to EXL3 stores most of the trunk
//!   this way (`{prefix}.trellis`, `suh`, `svh`, `mul1` where the original has `weight` and `scale`), and keeps the
//!   rest as it was. Such a projection takes its activation as it is (no fp8 quantization: the trellis has its own
//!   transforms) and its result is rounded as the caller asks, as a bf16 weight's.
//!
//! Weights stay in their stored form in RAM (fp8 is 1 byte/weight) and each
//! output row is dequantized once per call, then dotted with every token's
//! activation, so a prefill of t tokens reads the weight once.
//!
//! A weight can also be held elsewhere ([`Weight::Device`], a GPU's: see
//! [`crate::dsv41::model::Model::offload`]): the activation is still quantized here and the
//! result rounded here, so only the sums are made there.

use std::sync::{Arc, Mutex};

use oaiy_engine::backend::parallel_rows;
use oaiy_engine::{Error, Result};

use ggml_rs::exl3::{Exl3Data, PackedLinear};

use crate::dsv41::formats::{bf16_to_f32, e8m0_to_f32, fake_quant_fp8, fp8_e4m3_to_f32, to_bf16};
use crate::dsv41::safetensors::{Dtype, StIndex};

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

    /// A gated unit in one call of this kernel's kind (a GPU's: one submit, one round trip), this kernel its gate:
    /// the sums of `x` (one row, quantized already) against it and against `up`, their SwiGLU (each sum rounded to
    /// bf16 and clamped at `limit`, `silu(gate) * up` rounded to bf16) quantized to fp8 as `down` takes its input, and
    /// the sums of that against `down`; and with them `others`' sums of one row each. `(down's sums, the others')`,
    /// or None when it cannot (another kind among them), and the caller makes each projection on its own.
    #[allow(clippy::type_complexity)]
    fn forward_gated(
        &self,
        up: &dyn DenseKernel,
        down: &dyn DenseKernel,
        x: &[f32],
        limit: f32,
        others: &[(&dyn DenseKernel, &[f32], std::ops::Range<usize>)],
    ) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
        let _ = (up, down, x, limit, others);
        None
    }

    /// A projection of projections in one call of this kernel's kind (a GPU's: one submit, one round trip), this
    /// kernel the last: `first`'s sums of `t` rows each, each row's rounded to bf16 and laid end to end, quantized to
    /// fp8 where `quantized` (as an fp8 weight takes its activation), against this kernel; its sums. None when it
    /// cannot (another kind among them, a row that is not whole 32s, more rows than it takes), and the caller makes
    /// each projection on its own.
    fn forward_after(&self, first: &[(&dyn DenseKernel, &[f32], std::ops::Range<usize>)], t: usize, quantized: bool) -> Option<Vec<f32>> {
        let _ = (first, t, quantized);
        None
    }

    /// Two stages in one call of this kernel's kind (a GPU's: one submit, one round trip): `first`'s sums of their
    /// rows (`(kernel, x, t)`, every row of each output), then `then`'s of the rows of `first[normed]`'s sums normed as
    /// [`crate::dsv41::ops::rmsnorm`] norms them rounded to bf16 (by `weight`, `eps`), to the bit. `(first's, then's)`,
    /// or None when it cannot, and the caller makes each stage on its own.
    #[allow(clippy::type_complexity)]
    fn forward_normed(&self, first: &[(&dyn DenseKernel, &[f32], usize)], normed: usize, weight: &[f32], eps: f32, then: &[&dyn DenseKernel]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let _ = (first, normed, weight, eps, then);
        None
    }

    /// A program of steps ([`Step`]) in one call of this kernel's kind (a GPU's: one recording, one read back), its
    /// matrices all of that kind: the vectors `reads` names (`(step, first value, values)`), in order. None where it
    /// cannot (another kind among them), and the caller makes the steps as before.
    fn run_program(&self, steps: &[Step<'_>], reads: &[(usize, usize, usize)]) -> Option<Vec<Vec<f32>>> {
        let _ = (steps, reads);
        None
    }

    /// [`Self::run_program`] in two parts submitted one after the other, `steps[..split]` then the rest: the first's
    /// reads (`reads`) once it is done, and what gives the second's (`later`) once it is (the device making them
    /// meanwhile). None where it cannot.
    #[allow(clippy::type_complexity)]
    fn run_program_parts(&self, steps: &[Step<'_>], reads: &[(usize, usize, usize)], split: usize, later: &[(usize, usize, usize)]) -> Option<(Vec<Vec<f32>>, Later<'_>)> {
        let _ = (steps, reads, split, later);
        None
    }
}

/// The later part of a program run in two ([`run_program_parts`]): its reads, waited for when called.
pub type Later<'a> = Box<dyn FnOnce() -> Vec<Vec<f32>> + 'a>;

/// A step of a device program ([`run_program`]): each makes a vector (its number the step's own) or writes one. A
/// decode step's sublayers, their hyper-connections' mixing and their norms in one call where each was a round trip
/// of the host's work.
pub enum Step<'a> {
    /// The host's values.
    Input(&'a [f32]),
    /// A vector of this many values, written by the steps after.
    Scratch(usize),
    /// `w`'s sums of `rows` rows of vector `x` (a device's weight, one part of a stack), `[rows, w.n()]`, in f32.
    Mul { w: &'a Weight, x: usize, rows: usize },
    /// A stack's parts together (`group`, its [`Weight::Stacked`]'s), each job (`(part, row of x)`) in one pass: each
    /// job's sums, `[jobs, part's n]`, each part's own projection's to the bit.
    MulGroup { group: &'a Weight, x: usize, jobs: &'a [(u32, u32)] },
    /// `width` columns from `src_at` of `rows` rows of `src` (rows `src_stride` apart), each rounded to bf16, into `dst`
    /// from `dst_at` (rows `dst_stride` apart).
    CopyColsBf16 { src: usize, src_stride: usize, src_at: usize, dst: usize, dst_stride: usize, dst_at: usize, width: usize, rows: usize },
    /// [`crate::dsv41::ops::rmsnorm`] of `rows` rows of `x` (`width` each) by `weight`, to the bit.
    Norm { x: usize, weight: &'a [f32], eps: f32, rows: usize, width: usize },
    /// [`crate::dsv41::hc::post`] of `out` (`[d]`) into the residual's copies `res` by `mix` ([`crate::dsv41::hc::Mix`]
    /// laid out by [`crate::dsv41::hc::Mix::flat`]), within roundings of the host's.
    HcPost { out: usize, res: usize, mix: usize, d: usize },
    /// [`crate::dsv41::hc::pre`] of the copies `h` by `mix`'s pre, within roundings of the host's.
    HcPre { h: usize, mix: usize, d: usize },
    /// A gated unit's activation of its projections `gate` and `up` (`len` each, rounded to bf16 first): the host's
    /// `bf16(silu(min(g, limit)) * clamp(u, ±limit))` to the bit, `silu` its silu of every bf16 gate by the gate's bits
    /// ([`crate::dsv41::moe::silu_bf16_table`]).
    SwigluBf16 { gate: usize, up: usize, len: usize, silu: &'a [f32], limit: f32 },
    /// The `k` routed experts of `layer` the router's `logits` (`bias` its correction bias) likely picks, by the device's
    /// own scores (a guess: the host routes as before, and takes those of them it picked), `held` the experts' kernel
    /// ([`crate::dsv41::expert::ExpertsKernel::fused`]): `[4k]`, u32 bits: the held pass's jobs, the experts, their
    /// slots (`u32::MAX` where not held).
    Candidates { logits: usize, bias: &'a [f32], held: &'a dyn std::any::Any, layer: u32, k: usize },
    /// The held experts of `cands` (a `Candidates` step), each on row 0 of `x`, as the kernel makes them alone (to the
    /// bit): `[k, dim]`, unweighted, the row of one not held whatever it was.
    Held { held: &'a dyn std::any::Any, layer: u32, x: usize, cands: usize, limit: f32 },
}

/// `steps` as one call where a device holds every matrix they multiply and takes them together
/// ([`DenseKernel::run_program`]): the vectors `reads` names. None where it does not.
pub fn run_program(steps: &[Step<'_>], reads: &[(usize, usize, usize)]) -> Option<Vec<Vec<f32>>> {
    let first = steps.iter().find_map(|s| match s {
        Step::Mul { w: Weight::Device { kernel, .. }, .. } | Step::MulGroup { group: Weight::Device { kernel, .. }, .. } => Some(Arc::clone(kernel)),
        _ => None,
    })?;
    let start = std::time::Instant::now();
    let made = first.run_program(steps, reads)?;
    crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
    Some(made)
}

/// [`run_program`] in two parts ([`DenseKernel::run_program_parts`]) on the device `on` (one of the program's weights)
/// is on: the first's reads, and what gives the second's.
#[allow(clippy::type_complexity)]
pub fn run_program_parts<'k>(on: &'k Weight, steps: &[Step<'_>], reads: &[(usize, usize, usize)], split: usize, later: &[(usize, usize, usize)]) -> Option<(Vec<Vec<f32>>, Later<'k>)> {
    let Weight::Device { kernel, .. } = on else { return None };
    let start = std::time::Instant::now();
    let made = kernel.run_program_parts(steps, reads, split, later)?;
    crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
    Some(made)
}

/// `first`'s [`Weight::forward_rows`] of their rows (`(weight, x, t, rows, out)`, as [`forward_together`] takes them),
/// and `then`'s (`(weight, rows, out)`) of `rmsnorm(first[normed]'s output, weight, eps)`, `first[normed]` rounded to
/// bf16 (a layer's queries' low-rank row, then its queries and its indexer's): in one call where a device holds them
/// all and takes them together ([`DenseKernel::forward_normed`]), the norm made there to the host's bit; None where it
/// does not, and the caller makes them as before.
#[allow(clippy::type_complexity)]
pub fn normed_together(
    first: &[(&Weight, &[f32], usize, std::ops::Range<usize>, Out)],
    normed: usize,
    weight: &[f32],
    eps: f32,
    then: &[(&Weight, std::ops::Range<usize>, Out)],
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let device = |w: &Weight| if let Weight::Device { kernel, fp8: false, .. } = w { Some(Arc::clone(kernel)) } else { None };
    let firsts: Vec<Arc<dyn DenseKernel>> = first.iter().map(|(w, ..)| device(w)).collect::<Option<_>>()?;
    let thens: Vec<Arc<dyn DenseKernel>> = then.iter().map(|(w, ..)| device(w)).collect::<Option<_>>()?;
    if first.get(normed)?.4 != Out::Bf16 {
        return None;
    }
    for (w, x, t, rows, _) in first {
        assert_eq!(x.len(), t * w.k(), "linear: input is not [t, k]");
        assert!(rows.end <= w.n(), "linear: row range past the weight");
    }
    let with: Vec<(&dyn DenseKernel, &[f32], usize)> = firsts.iter().zip(first).map(|(k, (_, x, t, ..))| (&**k, *x, *t)).collect();
    let after: Vec<&dyn DenseKernel> = thens.iter().map(|k| &**k).collect();
    let start = std::time::Instant::now();
    let (alls, afters) = firsts[0].forward_normed(&with, normed, weight, eps, &after)?;
    crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
    let t = first[normed].2;
    // each output's rows asked for, of every row of it, rounded as asked
    let take = |all: &[f32], whole: usize, t: usize, rows: &std::ops::Range<usize>, out: Out| -> Vec<f32> {
        let mut y = Vec::with_capacity(t * rows.len());
        for tt in 0..t {
            y.extend(all[tt * whole + rows.start..tt * whole + rows.end].iter().map(|&v| if out == Out::Bf16 { to_bf16(v) } else { v }));
        }
        y
    };
    let ys = first.iter().zip(&alls).map(|((w, _, t, rows, out), all)| take(all, w.n(), *t, rows, *out)).collect();
    let qs = then.iter().zip(&afters).map(|((w, rows, out), all)| take(all, w.n(), t, rows, *out)).collect();
    Some((ys, qs))
}

/// `then`'s [`Weight::forward`] of the `t` rows `first` make: each `(weight, x, rows)`'s [`Weight::forward_rows`] of
/// those rows, each row's rounded to bf16 and laid end to end. One call where a device holds them all and takes them
/// together ([`DenseKernel::forward_after`]), the rows between them never brought here; None where it does not, and the
/// caller makes them as before. The roundings between the two are the reference's and are made of the values' bits,
/// there as here: the result is the same to the bit.
pub fn chained_together(first: &[(&Weight, &[f32], std::ops::Range<usize>)], then: &Weight, t: usize, out: Out) -> Option<Vec<f32>> {
    let Weight::Device { kernel: last, .. } = then else { return None };
    // (a stack's rows within one of its parts are that part's, as `forward_together` takes them)
    let resolved: Vec<(&Weight, &[f32], std::ops::Range<usize>)> = first
        .iter()
        .map(|(w, x, rows)| match w {
            Weight::Stacked { parts, rows: each, .. } if !rows.is_empty() && rows.start / each == (rows.end - 1) / each => {
                let p = rows.start / each;
                (&parts[p], *x, rows.start - p * each..rows.end - p * each)
            }
            _ => (*w, *x, rows.clone()),
        })
        .collect();
    let first = &resolved[..];
    let kernels: Vec<&dyn DenseKernel> = first.iter().map(|(w, ..)| if let Weight::Device { kernel, .. } = w { Some(&**kernel) } else { None }).collect::<Option<_>>()?;
    let width: usize = first.iter().map(|(_, _, rows)| rows.len()).sum();
    if width != then.k() || width % FP8_BLOCK != 0 {
        return None;
    }
    let prepared: Vec<Vec<f32>> = first
        .iter()
        .map(|(w, x, rows)| {
            assert_eq!(x.len(), t * w.k(), "linear: input is not [t, k]");
            assert!(rows.end <= w.n(), "linear: row range past the weight");
            w.prepare(x, Out::Bf16).0
        })
        .collect();
    let with: Vec<(&dyn DenseKernel, &[f32], std::ops::Range<usize>)> = kernels.iter().zip(first).zip(&prepared).map(|((k, (_, _, rows)), x)| (*k, x.as_slice(), rows.clone())).collect();
    let start = std::time::Instant::now();
    let mut y = last.forward_after(&with, t, then.quantizes())?;
    crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
    if out == Out::Bf16 || then.quantizes() {
        for v in y.iter_mut() {
            *v = to_bf16(*v);
        }
    }
    Some(y)
}

/// A token's fp8 gated unit (the shared expert: `gate` and `up` of `x`, the SwiGLU of the two clamped at `limit`,
/// `down` of it, rounded to bf16) and `others`' [`Weight::forward_rows`] of one row each (`(weight, x, rows, out)`),
/// in one call where a device holds them all and takes them together ([`DenseKernel::forward_gated`]); None where it
/// does not, and the caller makes them as before. The unit's roundings are the reference's at the same points; where
/// a device makes the SwiGLU its exponential is its own, so an element now and then is a bf16 step from the host's.
#[allow(clippy::type_complexity)]
pub fn gated_together(gate: &Weight, up: &Weight, down: &Weight, x: &[f32], limit: f32, others: &[(&Weight, &[f32], std::ops::Range<usize>, Out)]) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
    fn fp8(w: &Weight) -> Option<&dyn DenseKernel> {
        if let Weight::Device { kernel, fp8: true, .. } = w {
            Some(&**kernel)
        } else {
            None
        }
    }
    let (g, u, d) = (fp8(gate)?, fp8(up)?, fp8(down)?);
    let kernels: Vec<&dyn DenseKernel> = others.iter().map(|(w, ..)| if let Weight::Device { kernel, .. } = w { Some(&**kernel) } else { None }).collect::<Option<_>>()?;
    assert_eq!(x.len(), gate.k(), "linear: input is not [1, k]");
    assert!(gate.k() == up.k() && gate.n() == up.n() && gate.n() == down.k(), "linear: a gated unit's projections do not fit each other");
    let (xq, _) = gate.prepare(x, Out::Bf16);
    let prepared: Vec<(Vec<f32>, Out)> = others
        .iter()
        .map(|(w, x, rows, out)| {
            assert_eq!(x.len(), w.k(), "linear: input is not [1, k]");
            assert!(rows.end <= w.n(), "linear: row range past the weight");
            w.prepare(x, *out)
        })
        .collect();
    let with: Vec<(&dyn DenseKernel, &[f32], std::ops::Range<usize>)> = kernels.iter().zip(others).zip(&prepared).map(|((k, (_, _, rows, _)), (x, _))| (*k, x.as_slice(), rows.clone())).collect();
    let start = std::time::Instant::now();
    let (mut y, mut ys) = g.forward_gated(u, d, &xq, limit, &with)?;
    crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
    for v in y.iter_mut() {
        *v = to_bf16(*v);
    }
    for (y, (_, out)) in ys.iter_mut().zip(&prepared) {
        if *out == Out::Bf16 {
            for v in y.iter_mut() {
                *v = to_bf16(*v);
            }
        }
    }
    Some((y, ys))
}

/// Several weights' [`Weight::forward_rows`] at once, each `(weight, x, t, rows, out)`'s result in order and the same
/// as alone. Weights a device holds go in one call when its kernel takes them together: a decode step's dense calls
/// were each a round trip to the GPU, 667 of them a token, a third of them the eight groups of each layer's `wo_a`.
pub fn forward_together(items: &[(&Weight, &[f32], usize, std::ops::Range<usize>, Out)]) -> Vec<Vec<f32>> {
    // A stack's rows within one of its parts are that part's (an EXL3 checkpoint's `wo_a`, asked for a group at a time).
    let resolved: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> = items
        .iter()
        .map(|(w, x, t, rows, out)| match w {
            Weight::Stacked { parts, rows: each, .. } if !rows.is_empty() && rows.start / each == (rows.end - 1) / each => {
                let p = rows.start / each;
                (&parts[p], *x, *t, rows.start - p * each..rows.end - p * each, *out)
            }
            _ => (*w, *x, *t, rows.clone(), *out),
        })
        .collect();
    let items = &resolved[..];
    let kernels: Option<Vec<&dyn DenseKernel>> =
        items.iter().map(|(w, ..)| if let Weight::Device { kernel, .. } = w { Some(&**kernel) } else { None }).collect();
    // Kernels of more than one kind (an EXL3 checkpoint's projections beside its fp8 and bf16 ones): each kind's in a
    // call of its own, a round trip each, rather than one each.
    if let Some(kernels) = &kernels {
        let kind = |k: &dyn DenseKernel| (k as &dyn std::any::Any).type_id();
        if kernels.len() > 2 && kernels.iter().any(|k| kind(*k) != kind(kernels[0])) {
            let mut kinds: Vec<std::any::TypeId> = Vec::new();
            for k in kernels {
                if !kinds.contains(&kind(*k)) {
                    kinds.push(kind(*k));
                }
            }
            let mut ys: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
            for of in kinds {
                let mine: Vec<usize> = (0..items.len()).filter(|&i| kind(kernels[i]) == of).collect();
                let these: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> = mine.iter().map(|&i| items[i].clone()).collect();
                for (i, y) in mine.into_iter().zip(forward_together(&these)) {
                    ys[i] = y;
                }
            }
            return ys;
        }
    }
    if let Some(kernels) = kernels.filter(|k| k.len() > 1) {
        // Each input as its weights take it, made once for the weights that take it the same way (the queries' and the
        // window's projections both quantize a layer's x, the shared expert's gate and up likewise: quantized for each,
        // it was a tenth of a millisecond a layer, and each copy an upload of its own).
        let mut made: Vec<((*const f32, usize, bool), Vec<f32>)> = Vec::new();
        let prepared: Vec<(usize, Out)> = items
            .iter()
            .map(|(w, x, t, rows, out)| {
                assert_eq!(x.len(), t * w.k(), "linear: input is not [t, k]");
                assert!(rows.end <= w.n(), "linear: row range past the weight");
                let key = (x.as_ptr(), x.len(), w.quantizes());
                let at = made.iter().position(|(k, _)| *k == key).unwrap_or_else(|| {
                    made.push((key, w.prepare(x, *out).0));
                    made.len() - 1
                });
                (at, if w.quantizes() { Out::Bf16 } else { *out })
            })
            .collect();
        let many: Vec<(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)> =
            kernels.iter().zip(items).zip(&prepared).map(|((k, (_, _, t, rows, _)), (at, _))| (*k, made[*at].1.as_slice(), *t, rows.clone())).collect();
        let start = std::time::Instant::now();
        if let Some(mut ys) = kernels[0].forward_many(&many) {
            crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
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
    /// EXL3 as the checkpoint stores it (`[n, k]`), run here by the host's decoder, made when first asked for: a
    /// projection a GPU takes ([`crate::dsv41::model::Model::offload`]) never needs one. It has no host rows.
    Exl3 { data: Arc<Exl3Data>, host: std::sync::OnceLock<Arc<dyn PackedLinear>>, n: usize, k: usize },
    /// Projections stacked by output rows, `rows` each (`wo_a`'s groups, which an EXL3 checkpoint stores a slice each):
    /// row `r` is part `r / rows`'s row `r % rows`. Each part takes the same `k` inputs. `group`: the parts together on a
    /// device as well (a [`Weight::Device`]), where it takes them so: a step's row through every part in one pass
    /// ([`Step::MulGroup`]), where each part's was a pass of its own.
    Stacked { parts: Vec<Weight>, rows: usize, k: usize, group: Option<Box<Weight>> },
}

/// A copy of `d` (the type has no `Clone`): the host's decoder takes its own, as a GPU's upload does.
pub(crate) fn exl3_copy(d: &Exl3Data) -> Exl3Data {
    Exl3Data { words: d.words.clone(), suh: d.suh.clone(), svh: d.svh.clone(), tile_words: d.tile_words, input_map: d.input_map.clone(), output_map: d.output_map.clone() }
}

/// The host's EXL3 decoder for `d` (the WebGPU build's: [`Weight::load`] refuses an EXL3 weight in any other, so this is
/// never asked for there). `d` was validated when it was read.
#[cfg(feature = "webgpu")]
fn exl3_host(d: &Exl3Data) -> Arc<dyn PackedLinear> {
    ggml_rs_wgpu::exl3::exl3_cpu(exl3_copy(d)).expect("an EXL3 weight is validated when it is read")
}

#[cfg(not(feature = "webgpu"))]
fn exl3_host(_: &Exl3Data) -> Arc<dyn PackedLinear> {
    unreachable!("an EXL3 weight is refused when read, without the WebGPU build")
}

/// The multiplier mul1's codebook stores beside each EXL3 projection (`{prefix}.mul1`): the only codebook decoded here.
const MUL1: i64 = 0x83dc_d12d_u32 as i32 as i64;

/// `{prefix}`'s EXL3 projection (`trellis` `[k/16, n/16, 16Â·bits]` i16, `suh` `[k]`, `svh` `[n]`, `mul1`), checked.
fn load_exl3(idx: &StIndex, prefix: &str) -> Result<Weight> {
    if cfg!(not(feature = "webgpu")) {
        return Err(Error::Unsupported(format!("{prefix}: an EXL3 checkpoint needs the WebGPU build of the server")));
    }
    let name = format!("{prefix}.trellis");
    let info = idx.info(&name)?;
    let [kt, nt, tw] = info.shape[..] else {
        return Err(Error::Format(format!("{name}: expected [k/16, n/16, words], got {:?}", info.shape)));
    };
    if info.dtype != Dtype::I16 {
        return Err(Error::Format(format!("{name}: {:?}, expected I16", info.dtype)));
    }
    let mul1 = idx.read_i64(&format!("{prefix}.mul1"))?;
    if mul1.first() != Some(&MUL1) {
        return Err(Error::Unsupported(format!("{prefix}: a codebook other than mul1 ({mul1:?})")));
    }
    let suh = idx.read_f32(&format!("{prefix}.suh"))?;
    let svh = idx.read_f32(&format!("{prefix}.svh"))?;
    let (k, n) = (suh.len(), svh.len());
    if (kt * 16, nt * 16) != (k, n) {
        return Err(Error::Format(format!("{prefix}: trellis {:?} does not fit suh {k} and svh {n}", info.shape)));
    }
    let bytes = idx.read(&name)?;
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let data = Exl3Data { words, suh, svh, tile_words: tw, input_map: (0..k as u32).collect(), output_map: (0..n as u32).collect() };
    data.validate().map_err(|e| Error::Format(format!("{prefix}: {e}")))?;
    Ok(Weight::Exl3 { data: Arc::new(data), host: std::sync::OnceLock::new(), n, k })
}

impl Weight {
    /// Load `{prefix}.weight` (and `{prefix}.scale` for fp8) as stored; where there is none, the EXL3 projection an
    /// EXL3 checkpoint stores in its place (`{prefix}.trellis` and its parts). A weight stored both ways (the Engram's
    /// `wkv`) is taken as its `weight`.
    pub fn load(idx: &StIndex, prefix: &str) -> Result<Weight> {
        let name = format!("{prefix}.weight");
        if idx.get(&name).is_none() && idx.get(&format!("{prefix}.trellis")).is_some() {
            return load_exl3(idx, prefix);
        }
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
            // (f16, as an EXL3 checkpoint keeps its router and the indexer's `weights_proj`: as bf16 where every value
            // is one (the original's bf16 written as f16, which a GPU then holds as it holds the trunk's: on the CPU
            // the router was a millisecond a layer), else held as f32, which holds every f16 exactly, where bf16 would
            // round its mantissa)
            Dtype::F16 => {
                let w = idx.read_f32(&name)?;
                let bf: Vec<u16> = w.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                if w.iter().zip(&bf).all(|(&v, &b)| f32::from_bits((b as u32) << 16) == v) {
                    Weight::Bf16 { w: bf, n, k }
                } else {
                    Weight::F32 { w, n, k }
                }
            }
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
                        out.push(crate::dsv41::formats::f32_to_bf16(v));
                    }
                }
                Ok(Weight::Bf16 { w: out, n, k })
            }
            _ => Err(Error::Format(format!("{prefix}: expected fp8"))),
        }
    }

    /// A block-diagonal projection of `groups` groups (`wo_a`): [`load_fp8_as_bf16`](Self::load_fp8_as_bf16)'s, or
    /// in an EXL3 checkpoint, which stores each group's rows as a projection of its own (`{prefix}.slice.{g}`), those
    /// stacked ([`Weight::Stacked`]).
    pub fn load_groups(idx: &StIndex, prefix: &str, groups: usize) -> Result<Weight> {
        if idx.get(&format!("{prefix}.weight")).is_none() && idx.get(&format!("{prefix}.slice.0.trellis")).is_some() {
            let parts: Vec<Weight> = (0..groups).map(|g| load_exl3(idx, &format!("{prefix}.slice.{g}"))).collect::<Result<_>>()?;
            let (rows, k) = (parts[0].n(), parts[0].k());
            if parts.iter().any(|p| p.n() != rows || p.k() != k) {
                return Err(Error::Format(format!("{prefix}: its slices are not all [{rows}, {k}]")));
            }
            return Ok(Weight::Stacked { parts, rows, k, group: None });
        }
        Self::load_fp8_as_bf16(idx, prefix)
    }

    pub fn n(&self) -> usize {
        match self {
            Weight::Fp8 { n, .. } | Weight::Bf16 { n, .. } | Weight::F32 { n, .. } | Weight::Device { n, .. } | Weight::Exl3 { n, .. } => *n,
            Weight::Stacked { parts, rows, .. } => parts.len() * rows,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Weight::Fp8 { k, .. } | Weight::Bf16 { k, .. } | Weight::F32 { k, .. } | Weight::Device { k, .. } | Weight::Exl3 { k, .. } | Weight::Stacked { k, .. } => *k,
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
            Weight::Exl3 { .. } | Weight::Stacked { .. } => panic!("an EXL3 weight has no host rows (the embedding is stored bf16)"),
        }
    }

    /// The reference `linear(x, W)` for `t` rows of `x` (row-major `[t, k]`),
    /// returning `[t, n]`. fp8 weights always quantize the activation and
    /// return bf16 (`out` is ignored for them, as in the reference).
    pub fn forward(&self, x: &[f32], t: usize, out: Out) -> Vec<f32> {
        self.forward_rows(x, t, 0..self.n(), out)
    }

    /// Whether this weight takes its activation quantized to fp8 (and gives its result in bf16): an fp8 weight.
    fn quantizes(&self) -> bool {
        match self {
            Weight::Stacked { parts, .. } => parts.first().is_some_and(Weight::quantizes),
            w => matches!(w, Weight::Fp8 { .. } | Weight::Device { fp8: true, .. }),
        }
    }

    /// The activation as this weight takes it, and the rounding of its result: an fp8 weight's quantized and its
    /// result bf16, as the reference does.
    fn prepare(&self, x: &[f32], out: Out) -> (Vec<f32>, Out) {
        if self.quantizes() {
            (fake_quant_fp8(x, FP8_BLOCK), Out::Bf16)
        } else {
            (x.to_vec(), out)
        }
    }

    /// [`forward`](Self::forward) restricted to output rows `rows`, giving
    /// `[t, rows.len()]` â€” one group of a block-diagonal projection (`wo_a`).
    pub fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>, out: Out) -> Vec<f32> {
        let k = self.k();
        assert_eq!(x.len(), t * k, "linear: input is not [t, k]");
        assert!(rows.end <= self.n(), "linear: row range past the weight");
        let (first, n) = (rows.start, rows.len());
        // A stack's rows are its parts' (each asked for its own, its activation prepared as it takes it).
        if let Weight::Stacked { parts, rows: each, .. } = self {
            let mut y = vec![0.0f32; t * n];
            let mut r = rows.start;
            while r < rows.end {
                let (p, from) = (r / each, r % each);
                let to = (from + rows.end - r).min(*each);
                let part = parts[p].forward_rows(x, t, from..to, out);
                let (w, at) = (to - from, r - rows.start);
                for tt in 0..t {
                    y[tt * n + at..tt * n + at + w].copy_from_slice(&part[tt * w..(tt + 1) * w]);
                }
                r += w;
            }
            return y;
        }
        let (xin, out) = self.prepare(x, out);
        let start = std::time::Instant::now();
        if let Weight::Exl3 { data, host, k, .. } = self {
            // The decoder gives every row; those asked for are taken from them.
            let lin = host.get_or_init(|| exl3_host(data));
            let all = lin.linear(&ggml_rs::Tensor::from_vec(xin, vec![t, *k])).to_host();
            let (all, whole) = (all.data(), self.n());
            let mut y = Vec::with_capacity(t * n);
            for tt in 0..t {
                y.extend(all[tt * whole + first..tt * whole + first + n].iter().map(|&v| if out == Out::Bf16 { to_bf16(v) } else { v }));
            }
            crate::dsv41::profile::add(crate::dsv41::profile::Part::HostDense, start);
            return y;
        }
        if let Weight::Device { kernel, .. } = self {
            let mut y = kernel.forward_rows(&xin, t, rows);
            crate::dsv41::profile::add(crate::dsv41::profile::Part::DeviceDense, start);
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
        crate::dsv41::profile::add(crate::dsv41::profile::Part::HostDense, start);
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
        let one = crate::dsv41::formats::f32_to_bf16(1.0);
        let w = Weight::Bf16 { w: vec![one; 3], n: 1, k: 3 };
        let x = [1.0, 2f32.powi(-9), 2f32.powi(-10)];
        assert_eq!(w.forward(&x, 1, Out::F32)[0], 1.0 + 2f32.powi(-9) + 2f32.powi(-10));
        assert_eq!(w.forward(&x, 1, Out::Bf16)[0], 1.0);
    }
}
