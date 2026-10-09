//! A MoE layer's experts as a GGUF holds them (Qwen3.8-Flash-Next's GSQ-RCO files: 512 routed experts a layer, their
//! gate, up and down matrices each one tensor of quant blocks), beside [`crate::exl3`]'s EXL3 ones and run as those
//! are: a row's experts a job list (job `j` matrix `jobs[2j]` on input row `jobs[2j + 1]`, its result row `j`), routed
//! on the host or on the GPU ([`crate::exl3::record_route`]), grouped by expert for a prompt's rows, each row's weighted sum
//! made on the GPU. What differs is the matmul: a GGUF's weights are the weights (no Hadamard transforms, no channel
//! maps). The tensor cores' kernel decodes them by one function a type ([`Kind::wgsl`]'s `w8`: eight weights of a
//! row); the kernel for a step's row and a check's few ([`few_source`]) is written for its type, a word's weights as
//! vectors. Another type is its decode function, its rows' layout on the GPU and, for a lookup-table type, its grid.
//!
//! Q2_0 (ggml type 42: 64 weights a block of 18 bytes, an f16 scale then 2 bits a weight, a weight `(code - 1) *
//! scale`) is held a row at a time: its codes' words (16 weights a word, weight `i` its bits `2 i`), then its blocks'
//! scales, f16, two a word: the GGUF's bytes, no more, each word aligned.
//!
//! The grid types the GSQ-RCO IQ2_XS file keeps its experts' gate and up matrices in (IQ2_S, IQ2_XXS, IQ1_M: 256
//! weights a block, a group of eight an entry of ggml's grid, with its signs and a scale of its own) are held a block
//! in whole words, its fields each on a word's edge (a 32's four index bytes one word, its signs another: the GGUF's
//! bytes in another order, and a pad). The grid is a storage buffer of the layer's (binding 4), two bits a weight (a
//! grid's bytes take four values at most), so a group is one load of it: as WGSL constants such tables are copied at
//! each call, which once ran a dispatch past Windows' two seconds and reset the driver (`crate::shaders::layout`'s
//! note).
use crate::exl3::{coop_on, many_order, moe_rows_for, record_route, Step, FEW_MAX, GROUP, MANY_CLEAR, MANY_COUNT, MANY_SCAN, MANY_SCATTER, WSUM_APPLY, WSUM_ROWS};
use crate::{chunk_limit, Gpu, WgpuBackend};
use ggml_quants::GgmlType;
use ggml_rs::exl3::{route, Experts};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, Tensor};
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

mod cache;
mod host;
mod kernels;

// (what this file gave the crate and its users, and what the files make for each other)
pub use {cache::*, host::*};
use kernels::*;

/// One kind of a layer's routed experts' projection as a group (their gate and up matrices, or their down ones): their
/// rows in one buffer, a matrix's after another.
struct Group {
    words: wgpu::Buffer,
    kind: Kind,
    k: usize,
    n: usize,
    /// Words a row, and a matrix.
    rw: usize,
    mwords: usize,
    /// A grid type's table ([`Kind::table`]), which its kernels read at binding 4.
    table: Option<wgpu::Buffer>,
    /// Where the card holds only some of the layer's experts ([`Cache`]): every expert's rows in the host's memory
    /// (`words` then the card's slots').
    cold: Option<wgpu::Buffer>,
    /// The group's matrices an expert (2: gate and up; 1: down).
    mats: usize,
}

/// How a group's jobs are taken: each a block of its own (a step's, no order), or in blocks of one matrix (the order,
/// its blocks, the jobs a block): up to [`FEW_MAX`] a block in f32, 16 and more on the tensor cores.
#[derive(Clone, Copy)]
enum Order<'a> {
    Jobs,
    Blocks(&'a DeviceVec, usize, usize),
    /// As `Blocks`, the order's first block not its vector's first (the vector, that block, the blocks, the jobs a
    /// block): a prompt's experts of few rows, kept after its others' blocks.
    At(&'a DeviceVec, usize, usize, usize),
}

/// The shared expert's matrix: f16 where every value is one exactly (a Q2_0 matrix's are), else f32.
pub(crate) struct Dense {
    w: DeviceVec,
    half: bool,
    n: usize,
    k: usize,
}

impl Dense {
    pub(crate) fn new(b: &WgpuBackend, values: &[f32], n: usize, k: usize) -> Self {
        match DeviceChain::vec_f16(b, values) {
            Some(w) => Dense { w, half: true, n, k },
            None => {
                let w = b.vec(values.len());
                DeviceChain::upload(b, &w, values);
                Dense { w, half: false, n, k }
            }
        }
    }

    /// [`Self::rows`] of a SwiGLU's gate rows then its up rows (`[2 ff, k]`), and the SwiGLU, into `out` (`[rows,
    /// ff]`): one dispatch where the recorder has the two as one kernel (`fused` then unwritten), else the matmul into
    /// `fused` and the SwiGLU of it.
    pub(crate) fn rows_swiglu(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        if self.half && rec.matmul_f16_swiglu_rows(&self.w, self.n / 2, self.k, x, out, rows) {
            return;
        }
        self.rows(rec, x, fused, rows);
        rec.silu_mul_split_rows(fused, out, rows);
    }

    /// `first`'s rows then `second`'s as one matrix (`[n, k]`).
    pub(crate) fn stacked(b: &WgpuBackend, first: &[f32], second: &[f32], n: usize, k: usize) -> Self {
        let both: Vec<f32> = first.iter().chain(second).copied().collect();
        Dense::new(b, &both, n, k)
    }

    pub(crate) fn rows(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        if self.half {
            rec.matmul_f16_rows(&self.w, self.n, self.k, x, y, rows);
        } else {
            rec.matmul_f32_rows(&self.w, self.n, self.k, x, y, rows);
        }
    }
}

/// A MoE layer's GGUF experts on the GPU as groups, as [`crate::exl3::Exl3MoeGrouped`] holds EXL3's: the routed ones'
/// gate and up matrices in one buffer (matrix `2e` expert `e`'s gate, `2e + 1` its up), their down matrices in
/// another, the shared expert dense.
pub struct QuantMoe {
    b: WgpuBackend,
    routed: usize,
    hidden: usize,
    ff: usize,
    gu: Group,
    down: Group,
    /// The shared expert: its gate and up matrices one below the other (one matmul gives both, a dispatch fewer a
    /// layer), and its down one.
    shared: [Dense; 2],
    /// Whether a prompt's blocks go to the tensor cores (where the device has them; OAIY_QUANT_MOE_NO_COOP: no).
    coop: bool,
    /// The bytes counted against the backend's budget, given back when the layer goes.
    bytes: u64,
    /// Where the card holds only some of the routed experts: which, and the rest's place in the host's memory.
    cache: Option<Arc<Cache>>,
}

impl std::fmt::Debug for QuantMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantMoe({} routed experts of {}x{} in {} as two groups, and the shared one)", self.routed, self.hidden, self.ff, self.gu.kind.tag())
    }
}

impl Drop for QuantMoe {
    fn drop(&mut self) {
        self.b.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// What tells this module's scratch from EXL3's of the same shape, in the caches the two share.
const QUANT: usize = 1 << 40;

impl QuantMoe {
    /// The layer's experts as groups on `b`, if they can be: their types ones the kernels decode, each group within a
    /// binding, and all of them within the budget less `reserve`. Else `data` back, for the host.
    fn try_new(b: &WgpuBackend, data: QuantExpertsData, reserve: u64) -> Result<Self, QuantExpertsData> {
        Self::make(b, data, reserve, None)
    }

    /// [`Self::try_new`], the card holding `slots` of the routed experts where that is fewer than all of them (the
    /// first so many to begin with), every expert's matrices in the host's memory for the kernels to read the others
    /// from ([`Cache`]); `data` back too where the device's kernels cannot read the host's memory.
    fn make(b: &WgpuBackend, data: QuantExpertsData, reserve: u64, slots: Option<usize>) -> Result<Self, QuantExpertsData> {
        let (h, f, e) = (data.hidden, data.ff, data.experts);
        let (Some(kg), Some(ku), Some(kd)) = (Kind::of(data.gate.0), Kind::of(data.up.0), Kind::of(data.down.0)) else { return Err(data) };
        // (the tensor cores' tiles are 16 by 16, a scale's block 64; a grid type's block 256)
        if kg != ku || h % 64 != 0 || f % 64 != 0 || !kg.fits(h) || !kd.fits(f) {
            return Err(data);
        }
        let (gw, dw) = (kg.row_words(h), kd.row_words(f));
        let gu_bytes = (2 * e * f * gw * 4) as u64;
        let d_bytes = (e * h * dw * 4) as u64;
        let limit = chunk_limit(&b.gpu.limits);
        // (a job's matrix and row index its words in 32 bits)
        if gu_bytes > limit || d_bytes > limit || gu_bytes / 4 > u32::MAX as u64 || d_bytes / 4 > u32::MAX as u64 {
            return Err(data);
        }
        let held = slots.map_or(e, |s| s.clamp(1, e));
        let part = held < e;
        if part && !b.host_weights() {
            return Err(data);
        }
        let shared_bytes = (3 * h * f * 4) as u64;
        let total = (gu_bytes + d_bytes) / e as u64 * held as u64 + shared_bytes + if part { ((4 * e + 4 + 3 * TAKE) * 4) as u64 } else { 0 };
        let prev = b.used.fetch_add(total, Ordering::Relaxed);
        if prev + total > b.budget.saturating_sub(reserve) {
            b.used.fetch_sub(total, Ordering::Relaxed);
            return Err(data);
        }
        // a group's rows packed on every core, a matrix's after another (`which`: the tensors a matrix comes from in turn)
        let group = |which: &[&(GgmlType, Vec<u8>)], kind: Kind, n: usize, k: usize| -> Group {
            let (rw, rb) = (kind.row_words(k), row_bytes(which[0].0, k));
            let mwords = n * rw;
            let mut words = vec![0u32; e * which.len() * mwords];
            words.par_chunks_mut(mwords).enumerate().for_each(|(m, out)| {
                let src = &which[m % which.len()].1[(m / which.len()) * n * rb..];
                for (r, row) in out.chunks_exact_mut(rw).enumerate() {
                    kind.pack_row(&src[r * rb..(r + 1) * rb], k, row);
                }
            });
            // (the words' bytes as they lie: little-endian, as the kernels read them)
            let bytes: &[u8] = bytemuck::cast_slice(&words);
            let table = kind.table().map(|t| {
                let bytes: &[u8] = bytemuck::cast_slice(&t);
                b.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0
            });
            // (the card's slots: the first experts' to begin with; all of them in the host's memory beside)
            let hot = &bytes[..held * which.len() * mwords * 4];
            let cold = part.then(|| b.gpu.host_buffer(bytes).expect("a storage buffer in the host's memory"));
            Group { words: b.gpu.upload_rows(hot, hot.len(), 1).remove(0).0, kind, k, n, rw, mwords, table, cold, mats: which.len() }
        };
        let gu = group(&[&data.gate, &data.up], kg, f, h);
        let down = group(&[&data.down], kd, h, f);
        let shared = [Dense::stacked(b, &data.shared[0], &data.shared[1], 2 * f, h), Dense::new(b, &data.shared[2], h, f)];
        let coop = coop_on(&b.gpu) && std::env::var_os("OAIY_QUANT_MOE_NO_COOP").is_none();
        let cache = part.then(|| {
            // each expert's slot (the first `held` their own), no pass yet, no place in a prompt's scratch; the first
            // victims the last of the card's experts (none has been used)
            let mut state = vec![0u32; 4 * e + 4 + 3 * TAKE];
            for (x, s) in state[..e].iter_mut().enumerate() {
                *s = if x < held { x as u32 } else { MISS };
            }
            state[3 * e + 4..4 * e + 4].fill(MISS);
            let victims: Vec<u32> = (0..TAKE).map(|i| if i < held { (held - 1 - i) as u32 } else { MISS }).collect();
            state[4 * e + 4..4 * e + 4 + TAKE].copy_from_slice(&victims);
            let bytes: &[u8] = bytemuck::cast_slice(&state);
            let of = |g: &Group| (g.cold.clone().expect("a part-held group's rows in the host's memory"), g.words.clone(), (g.mats * g.mwords * 4) as u64);
            // (what a prompt's scratch of this layer's experts takes, for the device's largest)
            let lens = [(e - held) * gu.mats * gu.mwords, (e - held) * down.mats * down.mwords];
            let device = Arc::as_ptr(&b.gpu) as usize;
            let mut most = STAGE_MOST.lock().unwrap_or_else(|p| p.into_inner());
            match most.iter_mut().find(|(d, _)| *d == device) {
                Some((_, m)) => *m = [m[0].max(lens[0]), m[1].max(lens[1])],
                None => most.push((device, lens)),
            }
            drop(most);
            Arc::new(Cache {
                gpu: Arc::clone(&b.gpu),
                experts: e,
                slots: held,
                state: b.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0,
                groups: [of(&gu), of(&down)],
                host: Mutex::new(Held { map: state[..e].to_vec(), seen: [0, 0], victims }),
                counts: [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)],
            })
        });
        Ok(QuantMoe { b: b.clone(), routed: e, hidden: h, ff: f, gu, down, shared, coop, bytes: total, cache })
    }

    pub(crate) fn is_on(&self, gpu: &Arc<Gpu>) -> bool {
        Arc::ptr_eq(&self.b.gpu, gpu)
    }

    /// Scratch for `rows` rows of `top_k` experts, from `vec` (EXL3's [`Step`], what its caches keep: the vectors its
    /// transforms take here the SwiGLUs', the shared expert's in `part_gu` and each pair's in `xh_d`). `few`: a check's
    /// rows, whose jobs [`GROUP`] orders.
    fn scratch(&self, vec: &mut dyn FnMut(usize) -> DeviceVec, rows: usize, top_k: usize, few: bool) -> Step {
        let (h, f) = (self.hidden, self.ff);
        let pairs = rows * top_k;
        Step {
            top_k,
            jobs_gu: vec(4 * pairs),
            jobs_d: vec(2 * pairs),
            w: vec(rows * (top_k + 1)),
            xh_gu: vec(1),
            part_gu: vec(rows * f),
            out_gu: vec(2 * pairs * f),
            xh_d: vec(pairs * f),
            part_d: vec(1),
            out_d: vec(pairs * h),
            // (the shared expert's gate and up outputs side by side)
            sg: vec(rows * 2 * f),
            su: vec(1),
            sd: vec(rows * h),
            order_gu: vec(if few { 2 * pairs * rows } else { 1 }),
            order_d: vec(if few { pairs * rows } else { 1 }),
        }
    }

    /// The kept scratch of `rows` routed rows (a step's one, a check's few): one for every layer of this shape on the
    /// device (a layer's experts are done before the next layer's start), its bind groups kept.
    fn step(&self, rows: usize, top_k: usize) -> Arc<Step> {
        let key = [rows, top_k, self.hidden, self.ff, QUANT, QUANT];
        let mut s = self.b.gpu.moe_steps.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, st)) = s.iter().find(|(k, _)| *k == key) {
            return Arc::clone(st);
        }
        let st = Arc::new(self.scratch(&mut |n| self.b.vec(n), rows, top_k, rows > 1));
        s.push((key, Arc::clone(&st)));
        st
    }

    /// One group's jobs (`jobs`, `count` of them) on `x`'s rows into `y`'s (job `j`'s row `j`).
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn group_pass(&self, rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, jobs: &DeviceVec, count: usize, order: Order<'_>, y: &DeviceVec) {
        self.group_pass_from(rec, g, x, jobs, count, order, y, None)
    }

    /// [`Self::group_pass`]; `stage`: where a part-held layer's experts with no slot on the card were copied for this
    /// run ([`Cache::stage`]: a prompt's), else its kernels read them from the host's memory.
    #[allow(clippy::too_many_arguments)]
    fn group_pass_from(&self, rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, jobs: &DeviceVec, count: usize, order: Order<'_>, y: &DeviceVec, stage: Option<&DeviceVec>) {
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let (xb, jb, yb) = (buf(x), buf(jobs), buf(y));
        let ntiles = (g.n / 16) as u32;
        let (ob, base, blocks, rows, identity) = match order {
            Order::Jobs => (d.clone(), 0, count, 1, 1),
            Order::Blocks(o, blocks, rows) => (buf(o), 0, blocks, rows, 0),
            Order::At(o, base, blocks, rows) => (buf(o), base, blocks, rows, 0),
        };
        let coop = rows > FEW_MAX;
        // (a prompt's experts of few rows: eight lanes a row, 32 output rows a workgroup. Most of that pass's blocks
        // are empty, an expert of more rows or of none, and a workgroup of an empty block still starts)
        let lanes = if matches!(order, Order::At(..)) { 8 } else { few_lanes(g.kind, g.k) };
        let (name, source) = kernel(g.kind, coop, rows, g.k, if self.cache.is_some() { g.mats } else { 0 }, stage.is_some(), lanes);
        // (a card holding some of the experts: the rest's rows in the host's memory or where this run's were copied,
        // and the layer's state)
        let staged = stage.map(|s| buf(s));
        let (cold, state) = match (&self.cache, &staged) {
            (Some(c), Some(s)) => (s, &c.state),
            (Some(c), None) => (g.cold.as_ref().expect("a part-held group's rows in the host's memory"), &c.state),
            (None, _) => (&d, &drw),
        };
        // as many blocks a pass as the grid's third axis takes
        for first in (0..blocks).step_by(65535) {
            let these = 65535.min(blocks - first) as u32;
            let words = [g.n as u32, g.k as u32, g.rw as u32, identity, g.mwords as u32, (base + first) as u32, 0, self.routed as u32];
            let groups = (g.n as u32).div_ceil((256 / lanes) as u32);
            let grid = if coop { (ntiles.div_ceil(8), 1, these) } else { (groups.min(65535), groups.div_ceil(65535), these) };
            rec.dispatch_wide(name, source, [&g.words, &xb, &jb, &ob, g.table.as_ref().unwrap_or(&d), cold, &yb, state], &words, grid);
        }
        rec.weigh(2.0 * count as f64 * (g.n * g.k) as f64);
    }

    /// Record `assign`'s experts for each row of `x` into `out` (see `ChainRecorder::moe_rows`).
    pub(crate) fn record(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]) {
        let rows = assign.len();
        let h = self.hidden;
        let top_k = assign.first().map_or(0, |a| a.len().saturating_sub(1));
        assert!(rows > 0 && top_k > 0 && assign.iter().all(|a| a.len() == top_k + 1 && a[top_k].0 == self.routed), "moe: each row's routed experts, then the shared one");
        assert!(x.len >= rows * h && out.len >= rows * h, "moe: {rows} rows of {h}");
        // the jobs: gate and up of each (row, expert) on its row of x, then down of each on its hidden row
        let mut jobs_gu = Vec::with_capacity(4 * rows * top_k);
        let mut jobs_d = Vec::with_capacity(2 * rows * top_k);
        let mut w = Vec::with_capacity(rows * (top_k + 1));
        for (r, a) in assign.iter().enumerate() {
            for (j, &(e, wt)) in a[..top_k].iter().enumerate() {
                assert!(e < self.routed, "moe: expert {e} of {}", self.routed);
                jobs_gu.extend([2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32]);
                jobs_d.extend([e as u32, (r * top_k + j) as u32]);
                w.push(wt);
            }
            w.push(a[top_k].1);
        }
        let b = rec.backend().clone();
        // a step's one row: the kept scratch (its bind groups kept); else this call's (from the pool, given back when
        // the recording has run)
        let st = if rows == 1 && rec.keeps() { self.step(1, top_k) } else { Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, false)) };
        let up = |v: &DeviceVec, data: &[u32]| DeviceChain::upload(&b, v, &data.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>());
        up(&st.jobs_gu, &jobs_gu);
        up(&st.jobs_d, &jobs_d);
        DeviceChain::upload(&b, &st.w, &w);
        // a prompt's rows: each expert's in blocks, its weights decoded once a block (16 on the tensor cores)
        let block = if self.coop { 16 } else { FEW_MAX };
        let mut order = |jobs: &[u32]| {
            let o = many_order(jobs, block);
            let v = rec.scratch(o.len());
            up(&v, &o);
            (v, o.len() / block)
        };
        let orders = (rows > 1).then(|| (order(&jobs_gu), order(&jobs_d)));
        let (ogu, od) = match &orders {
            Some(((g, gn), (d, dn))) => (Order::Blocks(g, *gn, block), Order::Blocks(d, *dn, block)),
            None => (Order::Jobs, Order::Jobs),
        };
        self.run(rec, &st, x, out, rows, ogu, od, None, None);
    }

    /// `rows` rows' experts routed on the GPU from the router's `logits` (`[rows, routed + 1]`) and recorded into
    /// `out` (see `ChainRecorder::moe_routed`), as [`crate::exl3::Exl3MoeGrouped::record_routed`] records EXL3's:
    /// [`record_route`] writes the jobs and weights where [`Self::record`] uploads them. `into`: each row's sum added to its
    /// streams (the streams, their write weights, how many) where it would be `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_routed(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, into: Option<(&DeviceVec, &DeviceVec, usize)>) -> bool {
        // a prompt's rows (more than a check's) grouped by expert on the GPU, where the tensor cores take its blocks
        let many = rows > FEW_MAX && self.coop;
        if self.routed > 1024 || top_k == 0 || top_k > 32.min(self.routed) || rows == 0 || (rows > 64 && !many) || rows > 65535 || logits.len < rows * (self.routed + 1) {
            return false;
        }
        assert!(x.len >= rows * self.hidden && (into.is_some() || out.len >= rows * self.hidden), "moe: {rows} rows of {}", self.hidden);
        let pairs = rows * top_k;
        // a check's few rows: an expert the rows share decoded once for them (its jobs one block)
        let grouped = (2..=FEW_MAX).contains(&rows) && pairs <= 256;
        // (a prompt's scratch the recording's, each layer's in turn)
        let st = if rec.keeps() && !many && (rows == 1 || grouped) {
            self.step(rows, top_k)
        } else if many {
            let key = [rows, top_k, self.hidden, self.ff + QUANT];
            match rec.moe_tmp.take() {
                Some((k, st)) if k == key => {
                    rec.moe_tmp = Some((k, Arc::clone(&st)));
                    st
                }
                _ => {
                    let st = Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, false));
                    rec.moe_tmp = Some((key, Arc::clone(&st)));
                    st
                }
            }
        } else {
            Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, grouped))
        };
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        record_route(rec, &buf(logits), &st, self.routed, top_k, rows);
        if many {
            // blocks of some three times the jobs an expert has on average; the most blocks the experts could fill (a
            // part-filled one each at most), the grid that wide
            let bs = moe_rows_for(pairs, self.routed);
            let blocks = pairs.div_ceil(bs) + self.routed;
            // An expert with FEW_MAX of the chunk's rows or fewer takes no such block: its jobs go to a block of its
            // own of that many, after the others' in the same vectors, for the few rows' kernel. Of a chunk of 512's
            // 371 blocks of 32 rows a Flash-Next layer, 43% held 8 rows or fewer (158 of the 299 experts it used, a
            // tenth of its pairs), each a whole block's decode on the tensor cores.
            let few = if std::env::var_os("OAIY_MOE_NO_TAIL").is_some() { 0 } else { FEW_MAX };
            let (tail_g, tail_d) = (2 * bs * blocks, (bs * blocks + 3 * self.routed).next_multiple_of(FEW_MAX));
            let (og, od) = (rec.scratch(tail_g + 2 * self.routed * few), rec.scratch(tail_d + self.routed * few));
            let words = [pairs as u32, blocks as u32, self.routed as u32, bs as u32, few as u32, tail_d as u32, tail_g as u32];
            let groups = ((2 * bs * blocks) as u32).div_ceil(256);
            let jd = buf(&st.jobs_d);
            rec.dispatch_wide("moe-many-clear", MANY_CLEAR, [&d, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, (groups.min(65535), groups.div_ceil(65535), 1));
            rec.dispatch_wide("moe-many-count", MANY_COUNT, [&jd, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            rec.dispatch_wide("moe-many-scan", MANY_SCAN, [&d, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, (1, 1, 1));
            rec.dispatch_wide("moe-many-scatter", MANY_SCATTER, [&jd, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            let tail = (few > 0).then(|| [Order::At(&og, tail_g / few, 2 * self.routed, few), Order::At(&od, tail_d / few, self.routed, few)]);
            self.run(rec, &st, x, out, rows, Order::Blocks(&og, 2 * blocks, bs), Order::Blocks(&od, blocks, bs), tail, into);
            return true;
        }
        let (ogu, od) = if grouped {
            rec.dispatch_wide("moe-group", GROUP, [&buf(&st.jobs_d), &d, &d, &d, &d, &d, &buf(&st.order_gu), &buf(&st.order_d)], &[pairs as u32, rows as u32], (1, 1, 1));
            (Order::Blocks(&st.order_gu, 2 * pairs, rows), Order::Blocks(&st.order_d, pairs, rows))
        } else {
            (Order::Jobs, Order::Jobs)
        };
        self.run(rec, &st, x, out, rows, ogu, od, None, into);
        true
    }

    /// The experts' work once `st` holds the jobs and weights: gate and up, each pair's SwiGLU, down, the shared
    /// expert on every row, and each row's weighted sum (into `out`, or added to the streams `into` names). `tail`: a
    /// prompt's experts of few rows, their gate and up order and their down one, run after the others' of each.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, rec: &mut crate::chain::Recorder<'_>, st: &Step, x: &DeviceVec, out: &DeviceVec, rows: usize, ogu: Order<'_>, od: Order<'_>, tail: Option<[Order<'_>; 2]>, into: Option<(&DeviceVec, &DeviceVec, usize)>) {
        let (h, f, top_k) = (self.hidden, self.ff, st.top_k);
        let pairs = rows * top_k;
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        // (a part-held layer: its state read as the recording finishes; the pass's experts with no slot taken in
        // first, and a prompt's rows' other such copied into the card's scratch for the tensor cores' kernels)
        let stage = self.cache.as_ref().and_then(|c| {
            c.watch(rec);
            c.admit(rec, &st.jobs_d, pairs, matches!(ogu, Order::Blocks(_, _, rows) if rows > FEW_MAX))
        });
        self.group_pass_from(rec, &self.gu, x, &st.jobs_gu, 2 * pairs, ogu, &st.out_gu, stage.as_ref().map(|s| &s[0]));
        if let Some([gu, _]) = tail {
            self.group_pass_from(rec, &self.gu, x, &st.jobs_gu, 2 * pairs / 10, gu, &st.out_gu, stage.as_ref().map(|s| &s[0]));
        }
        let groups = ((pairs * f) as u32).div_ceil(256);
        rec.dispatch_wide("quant-moe-swiglu", SWIGLU_PAIRS, [&buf(&st.out_gu), &d, &d, &d, &d, &d, &buf(&st.xh_d), &drw], &[f as u32, pairs as u32], (groups.min(65535), groups.div_ceil(65535).max(1), 1));
        self.group_pass_from(rec, &self.down, &st.xh_d, &st.jobs_d, pairs, od, &st.out_d, stage.as_ref().map(|s| &s[1]));
        if let Some([_, down]) = tail {
            self.group_pass_from(rec, &self.down, &st.xh_d, &st.jobs_d, pairs / 10, down, &st.out_d, stage.as_ref().map(|s| &s[1]));
        }
        // the shared expert on every row: its gate and up with their SwiGLU, then its down projection with the
        // experts' sum into the streams, each one dispatch for a step's row or a check's few
        self.shared[0].rows_swiglu(rec, x, &st.sg, &st.part_gu, rows);
        if let Some((xs, post, streams)) = into {
            assert!(xs.len >= rows * streams * h && post.len >= rows * streams, "moe: {rows} rows' {streams} streams");
            let down = &self.shared[1];
            if down.half && rec.shared_down_wsum(&down.w, h, f, &st.part_gu, &st.out_d, &st.w, post, xs, rows, top_k, streams) {
                return;
            }
        }
        self.shared[1].rows(rec, &st.part_gu, &st.sd, rows);
        let groups = (((rows * h) as u32).div_ceil(256), 1, 1);
        match into {
            Some((xs, post, streams)) => {
                rec.dispatch_wide("moe-wsum-apply", WSUM_APPLY, [&buf(&st.out_d), &buf(&st.sd), &buf(&st.w), &buf(post), &d, &d, &buf(xs), &drw], &[h as u32, top_k as u32, rows as u32, streams as u32], groups);
            }
            None => rec.dispatch_wide("moe-wsum-rows", WSUM_ROWS, [&buf(&st.out_d), &buf(&st.sd), &buf(&st.w), &d, &d, &d, &buf(out), &drw], &[h as u32, top_k as u32, rows as u32], groups),
        }
    }
}

impl Experts for QuantMoe {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn part_held(&self) -> bool {
        self.cache.is_some()
    }

    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let h = self.hidden;
        let x = x.to_host();
        let logits = logits.to_host();
        let rows = x.numel() / h;
        let width = self.routed + 1;
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect();
        let b = &self.b;
        let (xd, out) = (b.vec(rows * h), b.vec(rows * h));
        DeviceChain::upload(b, &xd, x.data());
        let mut rec = b.begin();
        rec.keep_groups(false);
        rec.moe_rows(self, &xd, &out, &assign);
        rec.read(&out);
        let y = rec.finish().pop().expect("the experts' sum");
        Tensor::from_vec(y, vec![rows, h])
    }
}

impl WgpuBackend {
    /// A MoE layer's experts as a GGUF holds them: on the GPU as groups ([`QuantMoe`]) where the kernels decode their
    /// types and the weight budget holds them, else on the host, their shared expert here
    /// ([`crate::quant_host::quant_experts_host_beside`]).
    pub fn quant_experts(&self, data: QuantExpertsData) -> Result<Box<dyn Experts>, String> {
        self.quant_experts_leaving(data, 0)
    }

    /// As [`Self::quant_experts`] where the card has room for `slots` of the layer's routed experts only: it holds
    /// those (the ones last used, in time) and its kernels read the rest from the host's memory, which holds them
    /// all ([`Cache`]). On the host as [`Self::quant_experts`]'s where the device cannot ([`Self::host_weights`]).
    pub fn quant_experts_cached(&self, data: QuantExpertsData, slots: usize) -> Result<Box<dyn Experts>, String> {
        data.validate()?;
        match QuantMoe::make(self, data, 0, Some(slots)) {
            Ok(g) => Ok(Box::new(g)),
            Err(data) => crate::quant_host::quant_experts_host_beside(self, data),
        }
    }

    /// As [`Self::quant_experts`], leaving `reserve` bytes of the budget for the model's other matrices (loaded after).
    pub fn quant_experts_leaving(&self, data: QuantExpertsData, reserve: u64) -> Result<Box<dyn Experts>, String> {
        data.validate()?;
        match QuantMoe::try_new(self, data, reserve) {
            Ok(g) => Ok(Box::new(g)),
            Err(data) => crate::quant_host::quant_experts_host_beside(self, data),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
