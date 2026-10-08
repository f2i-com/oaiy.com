//! A MoE layer's GGUF experts on the host, for the layers no GPU has room for (Qwen3.8-Flash-Next's Q2_0 experts are
//! 34 GB, a 32 GB card holds some 35 of its 48 layers'): a chain reads such a layer's router and input back, these
//! run, and their sum goes up for the layer's write-back.
//!
//! [`crate::quant_moe::QuantMoeCpu`] is the reference the GPU's kernels are held against: it dequantises each expert's
//! three matrices whole to f32 for a call (20 MB an expert, 200 MB a row of ten). [`QuantMoeHost`] reads a Q2_0 matrix
//! as it lies in the GGUF (1.4 MB an expert): a block's weights are `(code - 1) * scale`, so a row's product with `x`
//! is, over its blocks, `scale * (sum of code * x - sum of x)`; a block's sum of `x` is made once for every row, and
//! `sum of code * x` takes a word's 16 codes as 16 floats (a shift a lane and a mask: one register on AVX-512, what a
//! compiler makes of the same loop elsewhere), a multiply-add with 16 of `x`.
//! Each row's experts are summed in the order it was routed, each expert's output made by itself, so a row's sum is
//! the same bits whatever rows are beside it (a check of drafted tokens rests on that).
//!
//! The shared expert is dense (20 MB of f32 a layer, cold at each step: more to read than a row's ten routed experts
//! together), so where the layer has a GPU it stays there ([`SharedOnGpu`], 10 MB as f16): the chain records it before
//! it reads the layer's router and input back, and the host adds its output by its gate (`Experts::forward_given`).
use crate::quant_moe::{Dense, QuantExpertsData};
use crate::WgpuBackend;
use ggml_quants::GgmlType;
use ggml_rs::exl3::{route, Experts};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, Tensor};
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// A Q2_0 block's weights, and its bytes (an f16 scale, then 2 bits a weight).
const BLOCK: usize = 64;
const BLOCK_BYTES: usize = 18;

/// The rows of a call up to which each routed expert is a task of its own and the caller's thread works beside them
/// (a step's row, a check's few).
const FEW_ROWS: usize = 8;

/// How the host's rows are multiplied: by what the CPU has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Level {
    /// AVX-512F: a word's 16 codes a register.
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// AVX2 and FMA: the portable loops, compiled for them.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Portable,
}

impl Level {
    /// The best the CPU has (OAIY_HOST_EXPERTS_PORTABLE: the portable loops).
    fn detect() -> Level {
        if std::env::var_os("OAIY_HOST_EXPERTS_PORTABLE").is_some() {
            return Level::Portable;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Level::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                return Level::Avx2;
            }
        }
        Level::Portable
    }
}

/// A block's scale: its f16 as f32, exactly (by shifts: a scale is read for every 64 weights of every row).
#[inline(always)]
fn scale(block: &[u8]) -> f32 {
    let half = u16::from_le_bytes([block[0], block[1]]) as u32;
    let (sign, exponent, mantissa) = (half >> 15 << 31, (half >> 10) & 0x1f, half & 0x3ff);
    match exponent {
        // zero and the subnormals: the mantissa in units of 2^-24
        0 => f32::from_bits(sign | (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits()),
        0x1f => f32::from_bits(sign | 0x7f80_0000 | mantissa << 13),
        _ => f32::from_bits(sign | (exponent + 112) << 23 | mantissa << 13),
    }
}

/// `out[r]`: Q2_0 row `r` of `w` (rows of `k` weights as a GGUF has them, `out.len()` of them) times `x`; `sums[b]`
/// the sum of `x` over block `b`. A word's 16 codes as floats times 16 of `x`, a block's four words summed in 16 lanes,
/// the lanes added to the row's by the block's scale; the row is its lanes' sum less its blocks' `scale * sum of x`.
#[inline(always)]
fn q2_rows_portable(w: &[u8], k: usize, x: &[f32], sums: &[f32], out: &mut [f32]) {
    let row_bytes = k / BLOCK * BLOCK_BYTES;
    for (row, o) in w.chunks_exact(row_bytes).zip(out.iter_mut()) {
        let (mut total, mut less) = ([0f32; 16], 0f32);
        for ((block, xb), s) in row.chunks_exact(BLOCK_BYTES).zip(x.chunks_exact(BLOCK)).zip(sums) {
            let mut lanes = [0f32; 16];
            for (word, x16) in block[2..].chunks_exact(4).zip(xb.chunks_exact(16)) {
                let codes = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                for (i, lane) in lanes.iter_mut().enumerate() {
                    *lane += ((codes >> (2 * i)) & 3) as f32 * x16[i];
                }
            }
            let d = scale(block);
            less += d * s;
            for (t, lane) in total.iter_mut().zip(&lanes) {
                *t += d * lane;
            }
        }
        *o = total.iter().sum::<f32>() - less;
    }
}

/// `out[r]`: row `r` of `w` (f32, rows of `k`, a multiple of 32) times `x`.
#[inline(always)]
fn dense_rows_portable(w: &[f32], k: usize, x: &[f32], out: &mut [f32]) {
    for (row, o) in w.chunks_exact(k).zip(out.iter_mut()) {
        let mut lanes = [0f32; 32];
        for (wc, xc) in row.chunks_exact(32).zip(x.chunks_exact(32)) {
            for i in 0..32 {
                lanes[i] += wc[i] * xc[i];
            }
        }
        *o = lanes.iter().sum();
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{BLOCK, BLOCK_BYTES};
    use std::arch::x86_64::*;

    /// [`super::q2_rows_portable`] on AVX-512, a word's 16 codes in one register: broadcast, shifted a lane, masked,
    /// converted, then multiplied by 16 of `x` into the block's sum; a block's four words are a chain of their own
    /// (blocks run side by side), its sum added to the row's by its scale; a row's scales are converted from f16
    /// sixteen at a time, their products with `x`'s block sums taken off at the end.
    ///
    /// # Safety
    /// The CPU must have AVX-512F (`is_x86_feature_detected!`).
    #[target_feature(enable = "avx512f")]
    pub unsafe fn q2_rows(w: &[u8], k: usize, x: &[f32], sums: &[f32], out: &mut [f32]) {
        let blocks = k / BLOCK;
        let row_bytes = blocks * BLOCK_BYTES;
        assert!(k % BLOCK == 0 && x.len() >= k && sums.len() >= blocks && w.len() >= out.len() * row_bytes, "Q2_0 rows of {k}");
        // a row's scales and x's block sums, to whole sixteens
        let padded = blocks.div_ceil(16) * 16;
        let (mut halves, mut scales, mut xsums) = (vec![0u16; padded], vec![0f32; padded], vec![0f32; padded]);
        xsums[..blocks].copy_from_slice(&sums[..blocks]);
        let shifts = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let three = _mm512_set1_epi32(3);
        for (row, o) in w.chunks_exact(row_bytes).zip(out.iter_mut()) {
            for (half, block) in halves.iter_mut().zip(row.chunks_exact(BLOCK_BYTES)) {
                *half = u16::from_le_bytes([block[0], block[1]]);
            }
            let mut less = _mm512_setzero_ps();
            for at in (0..padded).step_by(16) {
                // SAFETY: the three vectors are `padded` long, a multiple of 16 that `at + 16` does not pass.
                unsafe {
                    let d = _mm512_cvtph_ps(_mm256_loadu_si256(halves.as_ptr().add(at) as *const __m256i));
                    _mm512_storeu_ps(scales.as_mut_ptr().add(at), d);
                    less = _mm512_fmadd_ps(d, _mm512_loadu_ps(xsums.as_ptr().add(at)), less);
                }
            }
            let (mut t0, mut t1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
            for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
                let codes = |j: usize| {
                    let word = i32::from_le_bytes([block[2 + 4 * j], block[3 + 4 * j], block[4 + 4 * j], block[5 + 4 * j]]);
                    _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(word), shifts), three))
                };
                // SAFETY: x holds k values (asserted), and block b's 64 end at 64 b + 64 <= k.
                let xp = unsafe { x.as_ptr().add(b * BLOCK) };
                let (x0, x1, x2, x3) = unsafe { (_mm512_loadu_ps(xp), _mm512_loadu_ps(xp.add(16)), _mm512_loadu_ps(xp.add(32)), _mm512_loadu_ps(xp.add(48))) };
                let front = _mm512_fmadd_ps(codes(1), x1, _mm512_mul_ps(codes(0), x0));
                let back = _mm512_fmadd_ps(codes(3), x3, _mm512_mul_ps(codes(2), x2));
                let d = _mm512_set1_ps(scales[b]);
                if b % 2 == 0 {
                    t0 = _mm512_fmadd_ps(d, _mm512_add_ps(front, back), t0);
                } else {
                    t1 = _mm512_fmadd_ps(d, _mm512_add_ps(front, back), t1);
                }
            }
            *o = _mm512_reduce_add_ps(_mm512_sub_ps(_mm512_add_ps(t0, t1), less));
        }
    }

    /// The portable loops compiled for AVX-512 (the shared expert's f32 rows).
    ///
    /// # Safety
    /// The CPU must have AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dense_rows_512(w: &[f32], k: usize, x: &[f32], out: &mut [f32]) {
        super::dense_rows_portable(w, k, x, out)
    }

    /// The portable loops compiled for AVX2 and FMA.
    ///
    /// # Safety
    /// The CPU must have AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q2_rows_avx2(w: &[u8], k: usize, x: &[f32], sums: &[f32], out: &mut [f32]) {
        super::q2_rows_portable(w, k, x, sums, out)
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dense_rows_avx2(w: &[f32], k: usize, x: &[f32], out: &mut [f32]) {
        super::dense_rows_portable(w, k, x, out)
    }
}

fn q2_rows(level: Level, w: &[u8], k: usize, x: &[f32], sums: &[f32], out: &mut [f32]) {
    match level {
        // SAFETY: Level::detect found the features these are compiled for.
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { x86::q2_rows(w, k, x, sums, out) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { x86::q2_rows_avx2(w, k, x, sums, out) },
        Level::Portable => q2_rows_portable(w, k, x, sums, out),
    }
}

fn dense_rows(level: Level, w: &[f32], k: usize, x: &[f32], out: &mut [f32]) {
    match level {
        // SAFETY: Level::detect found the features these are compiled for.
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { x86::dense_rows_512(w, k, x, out) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { x86::dense_rows_avx2(w, k, x, out) },
        Level::Portable => dense_rows_portable(w, k, x, out),
    }
}

/// `x`'s sums over blocks of 64.
fn block_sums(x: &[f32]) -> Vec<f32> {
    x.chunks_exact(BLOCK).map(|b| b.iter().sum()).collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// A host layer's shared expert on its GPU: gate, up and down as a chain multiplies by them, and the scratch a step's
/// row and a check's few keep (their bind groups kept with them).
struct SharedOnGpu {
    b: WgpuBackend,
    mats: [Dense; 3],
    /// The bytes counted against the backend's budget, given back when the layer goes.
    bytes: u64,
    /// By rows: gate's output, up's, and their SwiGLU.
    kept: Mutex<Vec<(usize, Arc<[DeviceVec; 3]>)>>,
}

impl Drop for SharedOnGpu {
    fn drop(&mut self) {
        self.b.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A layer's Q2_0 experts on the host, read as the GGUF holds them. See the module's notes.
pub struct QuantMoeHost {
    data: QuantExpertsData,
    level: Level,
    gpu: Option<SharedOnGpu>,
}

impl std::fmt::Debug for QuantMoeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shared = if self.gpu.is_some() { "on the host, the shared one on its GPU" } else { "and the shared one, on the host" };
        write!(f, "QuantMoeHost({} routed experts of {}x{} in Q2_0 {shared}: {:?})", self.data.experts, self.data.hidden, self.data.ff, self.level)
    }
}

impl QuantMoeHost {
    /// The experts of `data`, where its routed ones are Q2_0 and their widths whole blocks: else `data` back.
    fn try_new(data: QuantExpertsData, level: Level) -> Result<Self, QuantExpertsData> {
        let q2 = |t: GgmlType| t == GgmlType::Q2_0;
        if !(q2(data.gate.0) && q2(data.up.0) && q2(data.down.0)) || data.hidden % BLOCK != 0 || data.ff % BLOCK != 0 {
            return Err(data);
        }
        Ok(QuantMoeHost { data, level, gpu: None })
    }

    /// With the shared expert on `b` as well, where its budget has the room (f16 where its values are f16's: 10 MB of
    /// Flash-Next's): a chain on `b` then runs it ([`Self::record_shared`]).
    fn beside(mut self, b: &WgpuBackend) -> Self {
        let (h, f) = (self.data.hidden, self.data.ff);
        let bytes = (3 * h * f * 4) as u64;
        let prev = b.used.fetch_add(bytes, Ordering::Relaxed);
        if prev + bytes > b.budget {
            b.used.fetch_sub(bytes, Ordering::Relaxed);
            return self;
        }
        let s = &self.data.shared;
        let mats = [Dense::new(b, &s[0], f, h), Dense::new(b, &s[1], f, h), Dense::new(b, &s[2], h, f)];
        self.gpu = Some(SharedOnGpu { b: b.clone(), mats, bytes, kept: Mutex::new(Vec::new()) });
        self
    }

    /// The shared expert on each of `rows` rows of `x` into `out`, recorded where `rec`'s device holds it (see
    /// `ChainRecorder::moe_shared`).
    pub(crate) fn record_shared(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, rows: usize) -> bool {
        let Some(g) = self.gpu.as_ref().filter(|g| std::ptr::eq(Arc::as_ptr(&g.b.gpu), rec.gpu())) else { return false };
        let (h, f) = (self.data.hidden, self.data.ff);
        assert!(rows > 0 && x.len >= rows * h && out.len >= rows * h, "a shared expert on {rows} rows of {h}");
        // a step's row, a check's few: the kept scratch; a prompt's: the recording's
        let st = if rows <= FEW_ROWS && rec.keeps() {
            let mut kept = g.kept.lock().unwrap_or_else(|p| p.into_inner());
            match kept.iter().find(|(r, _)| *r == rows) {
                Some((_, st)) => Arc::clone(st),
                None => {
                    let st = Arc::new([g.b.vec(rows * f), g.b.vec(rows * f), g.b.vec(rows * f)]);
                    kept.push((rows, Arc::clone(&st)));
                    st
                }
            }
        } else {
            Arc::new([rec.scratch(rows * f), rec.scratch(rows * f), rec.scratch(rows * f)])
        };
        g.mats[0].rows(rec, x, &st[0], rows);
        g.mats[1].rows(rec, x, &st[1], rows);
        rec.silu_mul(&st[0], &st[1], &st[2], rows * f);
        g.mats[2].rows(rec, &st[2], out, rows);
        true
    }

    /// Routed expert `e` on `x` (`sums` its blocks' sums): `down(silu(gate x) * up x)`, on the core that has it (its
    /// gate and up side by side and its down in pieces were tried for a step's row, whose ten experts leave most cores
    /// idle: slower, the pool's threads woken for each piece).
    fn routed(&self, e: usize, x: &[f32], sums: &[f32]) -> Vec<f32> {
        let d = &self.data;
        let (h, f) = (d.hidden, d.ff);
        let (wide, narrow) = (f * (h / BLOCK * BLOCK_BYTES), h * (f / BLOCK * BLOCK_BYTES));
        let (mut a, mut b, mut y) = (vec![0f32; f], vec![0f32; f], vec![0f32; h]);
        q2_rows(self.level, &d.gate.1[e * wide..(e + 1) * wide], h, x, sums, &mut a);
        q2_rows(self.level, &d.up.1[e * wide..(e + 1) * wide], h, x, sums, &mut b);
        for (a, b) in a.iter_mut().zip(&b) {
            *a = silu(*a) * b;
        }
        q2_rows(self.level, &d.down.1[e * narrow..(e + 1) * narrow], f, &a, &block_sums(&a), &mut y);
        y
    }

    /// The shared expert on `x`.
    fn shared(&self, x: &[f32]) -> Vec<f32> {
        let d = &self.data;
        let (h, f) = (d.hidden, d.ff);
        let (mut a, mut b, mut y) = (vec![0f32; f], vec![0f32; f], vec![0f32; h]);
        dense_rows(self.level, &d.shared[0], h, x, &mut a);
        dense_rows(self.level, &d.shared[1], h, x, &mut b);
        for (a, b) in a.iter_mut().zip(&b) {
            *a = silu(*a) * b;
        }
        dense_rows(self.level, &d.shared[2], f, &a, &mut y);
        y
    }
}

/// A piece of a call's work: a routed expert on its rows (each with where in its row's list it is), or the shared
/// expert on a row.
enum Task<'a> {
    Routed(usize, &'a [(usize, usize)]),
    Shared(usize),
}

impl QuantMoeHost {
    /// [`Experts::forward`]; `given`: the shared expert's outputs, a device's ([`Experts::forward_given`]).
    fn sum(&self, x: &Tensor, logits: &Tensor, top_k: usize, given: Option<&[f32]>) -> Tensor {
        let d = &self.data;
        let (h, n) = (d.hidden, d.experts);
        let x = x.to_host();
        let logits = logits.to_host();
        let (xs, ls) = (x.data(), logits.data());
        let rows = xs.len() / h;
        assert!(given.is_none_or(|g| g.len() >= rows * h), "the shared expert's outputs for {rows} rows");
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * (n + 1)..(r + 1) * (n + 1)], top_k)).collect();
        let sums: Vec<Vec<f32>> = xs.chunks_exact(h).map(block_sums).collect();
        // each routed expert's rows, and where in a row's list it is: an expert's rows one task (its matrices read on
        // one core for them all)
        let mut by: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (r, a) in assign.iter().enumerate() {
            for (at, &(e, _)) in a[..a.len() - 1].iter().enumerate() {
                by[e].push((r, at));
            }
        }
        let each = assign.first().map_or(0, |a| a.len() - 1);
        // (the shared expert's rows first: each the most to read)
        let shared = if given.is_none() { 0..rows } else { 0..0 };
        let tasks: Vec<Task<'_>> = shared.map(Task::Shared).chain(by.iter().enumerate().filter(|(_, rs)| !rs.is_empty()).map(|(e, rs)| Task::Routed(e, rs))).collect();
        // an output's place: row r's routed expert `at` is `r each + at`, its shared expert `rows each + r`
        let (sums, row) = (&sums, |r: usize| &xs[r * h..(r + 1) * h]);
        let work = |task: &Task<'_>| -> Vec<(usize, Vec<f32>)> {
            match *task {
                Task::Routed(e, rs) => rs.iter().map(|&(r, at)| (r * each + at, self.routed(e, row(r), &sums[r]))).collect(),
                Task::Shared(r) => vec![(rows * each + r, self.shared(row(r)))],
            }
        };
        let done: Vec<(usize, Vec<f32>)> = if rows <= FEW_ROWS {
            // a step's row, a check's few: each task the pool's but the first, which this thread takes while they run
            // (it would only wait)
            let done = Mutex::new(Vec::with_capacity(rows * (each + 1)));
            rayon::in_place_scope(|s| {
                for task in tasks.iter().skip(1) {
                    let (done, work) = (&done, &work);
                    s.spawn(move |_| {
                        let ys = work(task);
                        done.lock().unwrap_or_else(|p| p.into_inner()).extend(ys);
                    });
                }
                if let Some(first) = tasks.first() {
                    let ys = work(first);
                    done.lock().unwrap_or_else(|p| p.into_inner()).extend(ys);
                }
            });
            done.into_inner().unwrap_or_else(|p| p.into_inner())
        } else {
            tasks.par_iter().flat_map_iter(work).collect()
        };
        let mut slots: Vec<Option<Vec<f32>>> = (0..rows * (each + 1)).map(|_| None).collect();
        for (at, y) in done {
            slots[at] = Some(y);
        }
        // a row's experts summed as it was routed, then the shared one by its gate's sigmoid
        let mut out = vec![0f32; rows * h];
        for (r, o) in out.chunks_exact_mut(h).enumerate() {
            let last = match given {
                Some(g) => &g[r * h..(r + 1) * h],
                None => slots[rows * each + r].as_deref().expect("the shared expert's output"),
            };
            let terms = slots[r * each..(r + 1) * each].iter().map(|y| y.as_deref().expect("each routed expert's output")).chain(std::iter::once(last));
            for (y, &(_, w)) in terms.zip(&assign[r]) {
                for (o, v) in o.iter_mut().zip(y) {
                    *o += w * v;
                }
            }
        }
        Tensor::from_vec(out, vec![rows, h])
    }
}

impl Experts for QuantMoeHost {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        self.sum(x, logits, top_k, None)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn on_host(&self) -> bool {
        true
    }

    fn forward_given(&self, x: &Tensor, logits: &Tensor, top_k: usize, shared: &[f32]) -> Tensor {
        self.sum(x, logits, top_k, Some(shared))
    }
}

/// The experts of `data` on the host for a model's use: read as they lie where they are Q2_0 ([`QuantMoeHost`]), else
/// the reference ([`crate::quant_moe::QuantMoeCpu`], each expert dequantised for its call).
pub fn quant_experts_host(data: QuantExpertsData) -> Result<Box<dyn Experts>, String> {
    data.validate()?;
    match QuantMoeHost::try_new(data, Level::detect()) {
        Ok(fast) => Ok(Box::new(fast)),
        Err(data) => crate::quant_moe::quant_experts_cpu(data),
    }
}

/// [`quant_experts_host`] for a layer whose GPU is `b` (no room there for its routed experts): its shared expert on
/// `b`, for the chain that runs the layer.
pub fn quant_experts_host_beside(b: &WgpuBackend, data: QuantExpertsData) -> Result<Box<dyn Experts>, String> {
    data.validate()?;
    match QuantMoeHost::try_new(data, Level::detect()) {
        Ok(fast) => Ok(Box::new(fast.beside(b))),
        Err(data) => crate::quant_moe::quant_experts_cpu(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_moe::tests::{copy, experts, inputs, worst};

    /// Every level this CPU has.
    fn levels() -> Vec<Level> {
        let mut all = vec![Level::Portable];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                all.push(Level::Avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                all.push(Level::Avx512);
            }
        }
        all
    }

    /// The experts read as they lie give the reference's sums (which dequantises each matrix whole and sums in f64),
    /// within f32's rounding: at a small shape and at Flash-Next's, a step's row, a check's few and a prompt's many,
    /// on every level the CPU has.
    #[test]
    fn the_hosts_experts_read_as_they_lie_are_the_reference() {
        for (hidden, ff, count, top_k, rows) in [(128usize, 64usize, 24usize, 4usize, 1usize), (128, 64, 24, 4, 37), (256, 192, 12, 3, 5), (2560, 640, 14, 10, 1), (2560, 640, 14, 10, 4)] {
            let data = experts(31 + rows as u64, hidden, ff, count, true);
            let (x, logits) = inputs(7 + hidden as u64, rows, hidden, count);
            let (xt, lt) = (Tensor::from_vec(x, vec![rows, hidden]), Tensor::from_vec(logits, vec![rows, count + 1]));
            let want = crate::quant_moe::quant_experts_cpu(copy(&data)).unwrap().forward(&xt, &lt, top_k);
            for level in levels() {
                let fast = QuantMoeHost::try_new(copy(&data), level).ok().expect("Q2_0 experts");
                let got = fast.forward(&xt, &lt, top_k);
                let err = worst(got.data(), want.data());
                assert!(err < 2e-4, "{hidden}x{ff}, {rows} rows, {level:?}: the worst {err:.3e} of the reference's RMS");
            }
        }
    }

    /// A row's sum is the same bits alone and among other rows (a check of drafted tokens is held to its steps').
    #[test]
    fn a_rows_sum_is_its_own_whatever_rows_are_beside_it() {
        let (hidden, ff, count, top_k, rows) = (256usize, 128usize, 20usize, 5usize, 9usize);
        let data = experts(77, hidden, ff, count, true);
        let (x, logits) = inputs(5, rows, hidden, count);
        for level in levels() {
            let fast = QuantMoeHost::try_new(copy(&data), level).ok().expect("Q2_0 experts");
            let all = fast.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k);
            for r in 0..rows {
                let one = fast.forward(&Tensor::from_vec(x[r * hidden..(r + 1) * hidden].to_vec(), vec![1, hidden]), &Tensor::from_vec(logits[r * (count + 1)..(r + 1) * (count + 1)].to_vec(), vec![1, count + 1]), top_k);
                assert_eq!(one.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), all.data()[r * hidden..(r + 1) * hidden].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "row {r}, {level:?}");
            }
        }
    }

    /// The shared expert run by its GPU's chain and the routed ones on the host give the sums the host gives by itself:
    /// a step's row and a check's few within f32's rounding, a prompt's many within the tensor cores' f16; and the
    /// layer gone, its bytes are the budget's again.
    #[test]
    fn a_shared_expert_on_its_gpu_and_the_routed_on_the_host_are_the_hosts_sums() {
        use ggml_rs::DeviceChain;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let before = b.usage().0;
        for (hidden, ff, count, top_k, rows) in [(128usize, 64usize, 24usize, 4usize, 1usize), (256, 192, 12, 3, 5), (128, 64, 24, 4, 37)] {
            let data = experts(19 + rows as u64, hidden, ff, count, true);
            let host = QuantMoeHost::try_new(copy(&data), Level::detect()).ok().expect("Q2_0 experts").beside(&b);
            assert!(host.gpu.is_some() && b.usage().0 > before, "the shared expert on the GPU, counted");
            let (x, logits) = inputs(41 + hidden as u64, rows, hidden, count);
            let (xd, out) = (b.vec(rows * hidden), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xd, &x);
            let mut rec = b.begin();
            assert!(rec.moe_shared(&host, &xd, &out, rows), "recorded on its GPU");
            rec.read(&out);
            let shared = rec.finish().pop().expect("the shared expert's outputs");
            let (xt, lt) = (Tensor::from_vec(x, vec![rows, hidden]), Tensor::from_vec(logits, vec![rows, count + 1]));
            let (got, want) = (host.forward_given(&xt, &lt, top_k, &shared[..rows * hidden]), host.forward(&xt, &lt, top_k));
            let err = worst(got.data(), want.data());
            assert!(err < if rows > 8 { 2e-2 } else { 2e-4 }, "{hidden}x{ff}, {rows} rows: the worst {err:.3e} of the host's RMS");
        }
        assert_eq!(b.usage().0, before, "the layers gone, their bytes back");
    }

    /// Experts of another type go to the reference.
    #[test]
    fn experts_not_q2_0_are_the_references() {
        let mut data = experts(3, 128, 64, 4, true);
        data.gate.0 = GgmlType::Q4_0;
        assert!(QuantMoeHost::try_new(data, Level::Portable).is_err());
    }

    /// What a layer of Flash-Next's experts costs on the host: a step's row, a check's four, a prompt's 512.
    #[test]
    #[ignore = "a timing: 0.7 GB of experts made at random; run with --nocapture"]
    fn measure_a_layers_experts_on_the_host() {
        let (hidden, ff, count, top_k) = (2560usize, 640usize, 512usize, 10usize);
        let data = experts(11, hidden, ff, count, true);
        for level in levels() {
            let fast = QuantMoeHost::try_new(copy(&data), level).ok().expect("Q2_0 experts");
            // one routed expert and the shared one, each on this core
            let (x, _) = inputs(3, 1, hidden, count);
            let sums = block_sums(&x);
            let t = std::time::Instant::now();
            for e in 0..200 {
                std::hint::black_box(fast.routed(e, &x, &sums));
            }
            let one = t.elapsed().as_secs_f64() * 1e3 / 200.0;
            let t = std::time::Instant::now();
            for _ in 0..50 {
                std::hint::black_box(fast.shared(&x));
            }
            eprintln!("{level:?}: a routed expert {one:.3} ms, the shared one {:.3} ms, each on one core", t.elapsed().as_secs_f64() * 1e3 / 50.0);
            for rows in [1usize, 4, 512] {
                let (x, logits) = inputs(rows as u64, rows, hidden, count);
                let (xt, lt) = (Tensor::from_vec(x, vec![rows, hidden]), Tensor::from_vec(logits, vec![rows, count + 1]));
                let runs = if rows == 512 { 2 } else { 20 };
                let best = (0..runs)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        std::hint::black_box(fast.forward(&xt, &lt, top_k));
                        t.elapsed().as_secs_f64() * 1e3
                    })
                    .fold(f64::MAX, f64::min);
                eprintln!("{level:?}: {rows} rows {best:.2} ms");
            }
        }
        let slow = crate::quant_moe::quant_experts_cpu(data).unwrap();
        let (x, logits) = inputs(1, 1, hidden, count);
        let t = std::time::Instant::now();
        std::hint::black_box(slow.forward(&Tensor::from_vec(x, vec![1, hidden]), &Tensor::from_vec(logits, vec![1, count + 1]), top_k));
        eprintln!("the reference: 1 row {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
