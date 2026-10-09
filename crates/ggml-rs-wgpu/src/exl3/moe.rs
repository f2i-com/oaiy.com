//! A MoE layer's experts: their jobs' order and blocks, a step's recorded passes, and the layer grouped on the
//! GPU (`Exl3MoeGrouped`) or on the host (`Exl3MoeHost`).

use super::*;

/// How a group's jobs are taken: each a workgroup's (a step's, [`g_mm_source`]); a prompt's in blocks of one matrix
/// ([`g_many`], the order and its blocks); a check's few rows' grouped on the GPU in blocks of `rows` ([`few_kernel`], the
/// order and the most blocks it can have), each job's sums the one-job kernel's.
#[derive(Clone, Copy)]
pub(super) enum Order<'a> {
    Jobs,
    /// The order, its blocks, and the jobs a block.
    Many(&'a DeviceVec, usize, usize),
    Few(&'a DeviceVec, usize, usize),
}

/// An unused place in a block of [`g_many`]'s job order.
pub(crate) const NONE: u32 = u32::MAX;

/// A job list's (`jobs`: pairs of matrix and input row) order for [`g_many`]: its jobs grouped by matrix (each
/// matrix's in their list's order), in blocks of `rows`, a block one matrix's and its unused places [`NONE`].
pub(crate) fn many_order(jobs: &[u32], rows: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..(jobs.len() / 2) as u32).collect();
    idx.sort_by_key(|&j| jobs[2 * j as usize]);
    let mut order = Vec::with_capacity(idx.len() + rows);
    for same in idx.chunk_by(|&a, &b| jobs[2 * a as usize] == jobs[2 * b as usize]) {
        for block in same.chunks(rows) {
            order.extend_from_slice(block);
            order.resize(order.len().next_multiple_of(rows), NONE);
        }
    }
    order
}

/// Rows a block of a MoE layer's experts takes: a prompt routes a few rows to each, and a block decodes its expert's
/// weights whatever its rows (Qwen3.8-Flash-Next's chunk of 512, about 10 rows an expert: its experts 276 ms in blocks
/// of 32, 378 in blocks of 16; OAIY_EXL3_MOE_BLOCK=16 for those).
fn moe_block() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| match std::env::var("OAIY_EXL3_MOE_BLOCK").ok().and_then(|v| v.parse().ok()) {
        Some(16) => 16,
        _ => 32,
    })
}

/// Whether a prompt's EXL3 matmuls go through the tensor cores ([`g_coop`]): where the device has them.
pub(crate) fn coop_on(gpu: &Gpu) -> bool {
    gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
}

/// Rows a block of a prompt's experts' jobs takes: [`g_coop`]'s 16 (an expert's some 10 rows of a 512-row chunk in
/// one fragment of the tensor cores), else [`moe_block`]'s.
fn many_rows(gpu: &Gpu) -> usize {
    if coop_on(gpu) { 16 } else { moe_block() }
}

/// The jobs a block of a prompt's experts grouped on the GPU takes: about three times the jobs an expert has on
/// average (16 to 128), so a busy expert's tiles are decoded for as few blocks as a quiet one's empty places cost
/// columns (Qwen3.8-Flash-Next's chunk of 512, some 10 jobs an expert: blocks of 32 308 ms, of 16 337, of 64 339).
/// Blocks of 32 up to some 24 jobs an expert: a chunk of 1,024, 20 an expert, its experts' matmuls 118 ms with them
/// and 149 with blocks of 64 (a workgroup of 64 an SM, where two of 32).
pub(crate) fn moe_rows_for(pairs: usize, experts: usize) -> usize {
    let target = 3 * pairs / experts.max(1);
    if (16..=72).contains(&target) {
        return 32;
    }
    [16, 32, 64, 128].into_iter().find(|&b| b >= target).unwrap_or(128)
}

/// One kind of a layer's routed experts' projection as a group (their gate and up matrices, or their down ones): their
/// words in one buffer, a matrix's after another, and their transforms' tables (the maps the identity).
pub(super) struct Group {
    words: wgpu::Buffer,
    pub(super) suh: DeviceVec,
    pub(super) svh: DeviceVec,
    pub(super) k: usize,
    pub(super) n: usize,
    tw: usize,
    /// Words a matrix.
    mwords: usize,
    splits: u32,
}

/// A decode step's scratch: one row, `top_k` experts (the bind groups of a step made once).
pub(crate) struct Step {
    pub(crate) top_k: usize,
    pub(crate) jobs_gu: DeviceVec,
    pub(crate) jobs_d: DeviceVec,
    pub(crate) w: DeviceVec,
    pub(crate) xh_gu: DeviceVec,
    pub(crate) part_gu: DeviceVec,
    pub(crate) out_gu: DeviceVec,
    pub(crate) xh_d: DeviceVec,
    pub(crate) part_d: DeviceVec,
    pub(crate) out_d: DeviceVec,
    pub(crate) sg: DeviceVec,
    pub(crate) su: DeviceVec,
    pub(crate) sd: DeviceVec,
    /// A check's jobs grouped by matrix ([`GROUP`]): gate and up, down.
    pub(crate) order_gu: DeviceVec,
    pub(crate) order_d: DeviceVec,
}

/// A MoE layer's experts on the GPU as groups (Qwen3.8-Flash-Next's 512 routed ones): their gate and up matrices in
/// one buffer (matrix `2e` expert `e`'s gate, `2e + 1` its up), their down matrices in another, each group's
/// transforms' tables beside it; the shared expert as projections of its own. A row's (a step's) experts run in a
/// few dispatches from a job list, their transforms on the GPU too, where a projection was a dispatch (and its
/// transforms the host's) in a layer's two round trips.
pub struct Exl3MoeGrouped {
    b: WgpuBackend,
    routed: usize,
    hidden: usize,
    ff: usize,
    pub(super) gu: Group,
    down: Group,
    shared: [Exl3Gpu; 3],
}

impl std::fmt::Debug for Exl3MoeGrouped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exl3MoeGrouped({} routed experts of {}x{} as two groups, and the shared one)", self.routed, self.hidden, self.ff)
    }
}

impl Exl3MoeGrouped {
    /// The layer's experts as groups on `b`, if they can be: every routed expert's projections of one shape and
    /// bitrate, their maps the identity, each group within a binding, and all of them within the budget less `reserve`.
    /// None leaves them to [`Exl3MoeHost`].
    pub(super) fn try_new(b: &WgpuBackend, experts: &[[Exl3Data; 3]], reserve: u64) -> Result<Option<Exl3MoeGrouped>, String> {
        let Some((shared, routed)) = experts.split_last() else { return Ok(None) };
        let Some(first) = routed.first() else { return Ok(None) };
        let (hidden, ff) = (first[0].suh.len(), first[0].svh.len());
        let identity = |m: &[u32]| m.iter().enumerate().all(|(i, &v)| v as usize == i);
        for e in routed.iter().chain([shared]) {
            for d in e.iter() {
                d.validate()?;
            }
        }
        let same = routed.iter().all(|e| {
            [(hidden, ff), (hidden, ff), (ff, hidden)].iter().zip(e.iter()).all(|(&(k, n), d)| d.suh.len() == k && d.svh.len() == n && d.tile_words == e[0].tile_words && identity(&d.input_map) && identity(&d.output_map))
                && e[0].tile_words == first[0].tile_words
                && e[2].tile_words == first[2].tile_words
        });
        let shapes = [(hidden, ff), (hidden, ff), (ff, hidden)];
        if !same || shared.iter().zip(shapes).any(|(d, (k, n))| d.suh.len() != k || d.svh.len() != n) {
            return Ok(None);
        }
        let limit = chunk_limit(&b.gpu.limits);
        let mwords = |k: usize, n: usize, tw: usize| k / 16 * n / 16 * (tw / 2);
        let (gu_words, d_words) = (mwords(hidden, ff, first[0].tile_words), mwords(ff, hidden, first[2].tile_words));
        let gu_bytes = (2 * routed.len() * gu_words * 4) as u64;
        let d_bytes = (routed.len() * d_words * 4) as u64;
        let shared_bytes: u64 = shared.iter().map(|d| d.words.len() as u64 * 4).sum();
        let tables = (routed.len() * 3 * (hidden + ff) * 4) as u64;
        let total = gu_bytes + d_bytes + shared_bytes + tables;
        if gu_bytes > limit || d_bytes > limit {
            return Ok(None);
        }
        let prev = b.used.fetch_add(total, Ordering::Relaxed);
        if prev + total > b.budget.saturating_sub(reserve) {
            b.used.fetch_sub(total, Ordering::Relaxed);
            return Ok(None);
        }
        // the groups' words and tables; the shared expert's projections count themselves, so their share is given back
        b.used.fetch_sub(shared_bytes, Ordering::Relaxed);
        let group = |which: &[usize], k: usize, n: usize, tw: usize| -> Group {
            let mw = mwords(k, n, tw);
            let mut bytes = Vec::with_capacity(routed.len() * which.len() * mw * 4);
            let (mut suh, mut svh) = (Vec::new(), Vec::new());
            for e in routed {
                for &p in which {
                    bytes.extend(e[p].words.iter().flat_map(|w| w.to_le_bytes()));
                    suh.extend_from_slice(&e[p].suh);
                    svh.extend_from_slice(&e[p].svh);
                }
            }
            let words = b.gpu.upload_rows(&bytes, bytes.len(), 1).remove(0).0;
            let up = |v: &[f32]| {
                use ggml_rs::DeviceChain;
                let d = b.vec(v.len());
                DeviceChain::upload(b, &d, v);
                d
            };
            let ntiles = n / 16;
            let splits = 4096usize.div_ceil(ntiles).next_power_of_two().min(8).min(k / 16).max(1) as u32;
            Group { words, suh: up(&suh), svh: up(&svh), k, n, tw, mwords: mw, splits }
        };
        let gu = group(&[0, 1], hidden, ff, first[0].tile_words);
        let down = group(&[2], ff, hidden, first[2].tile_words);
        let [sg, su, sd] = shared;
        let one = |d: &Exl3Data| Exl3Gpu::upload(b, Exl3Data { words: d.words.clone(), suh: d.suh.clone(), svh: d.svh.clone(), tile_words: d.tile_words, input_map: d.input_map.clone(), output_map: d.output_map.clone() }, None);
        let shared = [one(sg), one(su), one(sd)];
        if shared.iter().any(|s| s.single_chunk().is_none()) {
            return Ok(None);
        }
        // the routed groups' bytes stay counted as long as the layer lives
        Ok(Some(Exl3MoeGrouped { b: b.clone(), routed: routed.len(), hidden, ff, gu, down, shared }))
    }

    pub(crate) fn is_on(&self, gpu: &Arc<Gpu>) -> bool {
        Arc::ptr_eq(&self.b.gpu, gpu)
    }

    /// Scratch for `rows` rows of `top_k` experts, from `vec`: each job one row (`one`: a step's, a check's) the groups'
    /// splits, a prompt's (each expert's rows in blocks) one split.
    pub(super) fn scratch(&self, vec: &mut dyn FnMut(usize) -> DeviceVec, rows: usize, top_k: usize, one: bool) -> Step {
        let (h, f) = (self.hidden, self.ff);
        let pairs = rows * top_k;
        let (sgu, sd) = if one { (self.gu.splits as usize, self.down.splits as usize) } else { (1, 1) };
        Step {
            top_k,
            jobs_gu: vec(4 * pairs),
            jobs_d: vec(2 * pairs),
            w: vec(rows * (top_k + 1)),
            xh_gu: vec(2 * pairs * h),
            part_gu: vec(2 * pairs * sgu * f),
            out_gu: vec(2 * pairs * f),
            xh_d: vec(pairs * f),
            part_d: vec(pairs * sd * h),
            out_d: vec(pairs * h),
            sg: vec(rows * f),
            su: vec(rows * f),
            sd: vec(rows * h),
            order_gu: vec(if one && rows > 1 { 2 * pairs * rows } else { 1 }),
            order_d: vec(if one && rows > 1 { pairs * rows } else { 1 }),
        }
    }

    /// The kept scratch of `rows` routed rows (a step's one, a check's few), each job one row: one for every layer of
    /// this shape on the device (a layer's experts are done before the next layer's start), its bind groups kept.
    fn step(&self, rows: usize, top_k: usize) -> Arc<Step> {
        let key = [rows, top_k, self.hidden, self.ff, self.gu.splits as usize, self.down.splits as usize];
        let mut s = self.b.gpu.moe_steps.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, st)) = s.iter().find(|(k, _)| *k == key) {
            return Arc::clone(st);
        }
        let st = Arc::new(self.scratch(&mut |n| self.b.vec(n), rows, top_k, true));
        s.push((key, Arc::clone(&st)));
        st
    }

    /// One group's jobs (`jobs`, `count` of them) on `x`: its input transforms, matmul and output transforms into `y`.
    /// `order`: a prompt's jobs in blocks of one matrix (and the blocks' count), each tile decoded once a block, in one
    /// split (a prompt has workgroups enough without, and its partial sums are the smaller).
    #[allow(clippy::too_many_arguments)]
    /// `pairs`: `x` the gate and up rows of each hidden row (`2 r` and `2 r + 1`), the input their SwiGLU.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn group_pass(&self, rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, pairs: bool, jobs: &DeviceVec, count: usize, order: Order<'_>, xh: &DeviceVec, part: &DeviceVec, y: &DeviceVec) {
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let (xb, jb, xhb, pb, yb) = (buf(x), buf(jobs), buf(xh), buf(part), buf(y));
        let (suh, svh) = (buf(&g.suh), buf(&g.svh));
        if pairs {
            rec.dispatch_wide("exl3-pre-swiglu", chain_shader("pre-swiglu"), [&xb, &suh, &d, &jb, &xb, &d, &xhb, &drw], &[g.k as u32, 1, 2, 0, 2, 1], ((g.k / 128) as u32, count as u32, 1));
        } else {
            rec.dispatch_wide("exl3-pre", chain_shader("pre"), [&xb, &suh, &d, &jb, &d, &d, &xhb, &drw], &[g.k as u32, 1], ((g.k / 128) as u32, count as u32, 1));
        }
        let ntiles = (g.n / 16) as u32;
        let splits = if matches!(order, Order::Many(..)) { 1 } else { g.splits };
        // as many jobs (or blocks) a pass as the grid's third axis takes
        let per = (65535 / splits) as usize;
        match order {
            Order::Many(order, blocks, rows) => {
                let ob = buf(order);
                for first in (0..blocks).step_by(per) {
                    let these = per.min(blocks - first) as u32;
                    if coop_on(rec.gpu()) {
                        // 8 tile columns a workgroup on the tensor cores
                        let src = g_coop(rows);
                        rec.dispatch_wide(coop_name(rows), &src, [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, 1, g.mwords as u32, first as u32], (ntiles.div_ceil(8), 1, these));
                    } else {
                        rec.dispatch_wide(many_name(rows), &g_many(rows), [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), these * splits));
                    }
                }
            }
            Order::Few(order, blocks, rows) => {
                let ob = buf(order);
                for first in (0..blocks).step_by(per) {
                    let these = per.min(blocks - first) as u32;
                    let (name, source) = few_kernel(rows, g.tw);
                    rec.dispatch_wide(name, source, [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), these * splits));
                }
            }
            Order::Jobs => {
                for first in (0..count).step_by(per) {
                    let jobs = per.min(count - first) as u32;
                    // (tiles of up to 4 bits a weight through the kernel that finds a code in three words: OAIY_EXL3_GENERAL
                    // keeps every tile on the general one)
                    let (name, source) = one_row_kernel(g.tw as usize);
                    rec.dispatch_wide(name, source, [&g.words, &xhb, &jb, &d, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, g.splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), jobs * g.splits));
                }
            }
        }
        rec.dispatch_wide("exl3-post", chain_shader("post"), [&pb, &svh, &jb, &d, &d, &d, &yb, &drw], &[g.n as u32, splits], ((g.n / 128) as u32, count as u32, 1));
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
                jobs_gu.extend([2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32]);
                jobs_d.extend([e as u32, (r * top_k + j) as u32]);
                w.push(wt);
            }
            w.push(a[top_k].1);
        }
        let b = rec.backend().clone();
        // a step's one row: the kept scratch (its bind groups kept); else this call's (from the pool, given back when
        // the recording has run)
        let st = if rows == 1 && rec.keeps() { self.step(1, top_k) } else { Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, rows == 1)) };
        let up = |v: &DeviceVec, data: &[u32]| DeviceChain::upload(&b, v, &data.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>());
        up(&st.jobs_gu, &jobs_gu);
        up(&st.jobs_d, &jobs_d);
        DeviceChain::upload(&b, &st.w, &w);
        // a prompt's rows: each expert's in blocks, a tile decoded once a block
        let block = many_rows(rec.gpu());
        let mut order = |jobs: &[u32]| {
            let o = many_order(jobs, block);
            let v = rec.scratch(o.len());
            up(&v, &o);
            (v, o.len() / block)
        };
        let orders = (rows > 1).then(|| (order(&jobs_gu), order(&jobs_d)));
        let (ogu, od) = match &orders {
            Some(((g, gn), (d, dn))) => (Order::Many(g, *gn, block), Order::Many(d, *dn, block)),
            None => (Order::Jobs, Order::Jobs),
        };
        self.run(rec, &st, x, out, rows, ogu, od, None);
    }

    /// `rows` rows' experts (a step's one, a check's few) routed on the GPU from the router's `logits` (`[rows, routed +
    /// 1]`) and recorded into `out` (see `ChainRecorder::moe_routed`): [`record_route`] writes the jobs and weights where
    /// [`Self::record`] uploads them, each job one row (so each row's sums are a step's). `into`: each row's sum added to
    /// its streams (the streams, their write weights, how many) where it would be `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_routed(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, into: Option<(&DeviceVec, &DeviceVec, usize)>) -> bool {
        // a prompt's rows (more than a check's) grouped by expert on the GPU, where the tensor cores take its blocks
        let many = rows > FEW_MAX && coop_on(rec.gpu());
        if self.routed > 1024 || top_k == 0 || top_k > 32.min(self.routed) || rows == 0 || (rows > 64 && !many) || rows > 65535 || logits.len < rows * (self.routed + 1) {
            return false;
        }
        assert!(x.len >= rows * self.hidden && (into.is_some() || out.len >= rows * self.hidden), "moe: {rows} rows of {}", self.hidden);
        // (a prompt's orders its own: the scratch's are a check's; its scratch the recording's, each layer's in turn)
        let st = if rec.keeps() && !many {
            self.step(rows, top_k)
        } else if many {
            let key = [rows, top_k, self.hidden, self.ff];
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
            Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, true))
        };
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        record_route(rec, &buf(logits), &st, self.routed, top_k, rows);
        let pairs = rows * top_k;
        if many {
            // blocks of as many jobs as an expert has on average (16 to 64: an expert's tiles decoded once a block,
            // what an empty place of it costs a fragment's columns); the most blocks the experts could fill (a
            // part-filled one each at most), the grid that wide
            let bs = moe_rows_for(pairs, self.routed);
            let blocks = pairs.div_ceil(bs) + self.routed;
            let (og, od) = (rec.scratch(2 * bs * blocks), rec.scratch(bs * blocks + 3 * self.routed));
            let words = [pairs as u32, blocks as u32, self.routed as u32, bs as u32];
            let clear = (2 * bs * blocks) as u32;
            let groups = clear.div_ceil(256);
            let jd = buf(&st.jobs_d);
            rec.dispatch_wide("moe-many-clear", MANY_CLEAR, [&d, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, (groups.min(65535), groups.div_ceil(65535), 1));
            rec.dispatch_wide("moe-many-count", MANY_COUNT, [&jd, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            rec.dispatch_wide("moe-many-scan", MANY_SCAN, [&d, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, (1, 1, 1));
            rec.dispatch_wide("moe-many-scatter", MANY_SCATTER, [&jd, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            self.run(rec, &st, x, out, rows, Order::Many(&og, 2 * blocks, bs), Order::Many(&od, blocks, bs), into);
            return true;
        }
        // a check's few rows: an expert the rows share decoded once for them (its jobs one block; OAIY_MOE_UNGROUPED:
        // a job each)
        let grouped = (2..=FEW_MAX).contains(&rows) && pairs <= 256 && std::env::var_os("OAIY_MOE_UNGROUPED").is_none();
        let (ogu, od) = if grouped {
            rec.dispatch_wide("moe-group", GROUP, [&buf(&st.jobs_d), &d, &d, &d, &d, &d, &buf(&st.order_gu), &buf(&st.order_d)], &[pairs as u32, rows as u32], (1, 1, 1));
            (Order::Few(&st.order_gu, 2 * pairs, rows), Order::Few(&st.order_d, pairs, rows))
        } else {
            (Order::Jobs, Order::Jobs)
        };
        self.run(rec, &st, x, out, rows, ogu, od, into);
        true
    }

    /// The experts' work once `st` holds the jobs and weights: gate and up, SwiGLU, down, the shared expert on every
    /// row, and each row's weighted sum (into `out`, or added to the streams `into` names); the gate and up jobs taken as
    /// `ogu` has them, the down jobs as `od`, each SwiGLU computed as its down projection reads it.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, rec: &mut crate::chain::Recorder<'_>, st: &Step, x: &DeviceVec, out: &DeviceVec, rows: usize, ogu: Order<'_>, od: Order<'_>, into: Option<(&DeviceVec, &DeviceVec, usize)>) {
        use ggml_rs::ChainRecorder;
        let (h, top_k) = (self.hidden, st.top_k);
        let pairs = rows * top_k;
        let (jgu, jd, wv, xh_gu, part_gu, out_gu, xh_d, part_d, out_d, sg, su, sd) = (&st.jobs_gu, &st.jobs_d, &st.w, &st.xh_gu, &st.part_gu, &st.out_gu, &st.xh_d, &st.part_d, &st.out_d, &st.sg, &st.su, &st.sd);
        self.group_pass(rec, &self.gu, x, false, jgu, 2 * pairs, ogu, xh_gu, part_gu, out_gu);
        self.group_pass(rec, &self.down, out_gu, true, jd, pairs, od, xh_d, part_d, out_d);
        // the shared expert on every row
        rec.exl3_rows(&self.shared[0], x, sg, rows);
        rec.exl3_rows(&self.shared[1], x, su, rows);
        rec.exl3_rows_swiglu(&self.shared[2], sg, su, sd, rows);
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        match into {
            Some((xs, post, streams)) => {
                assert!(xs.len >= rows * streams * h && post.len >= rows * streams, "moe: {rows} rows' {streams} streams");
                rec.dispatch_wide("moe-wsum-apply", WSUM_APPLY, [&buf(out_d), &buf(sd), &buf(wv), &buf(post), &d, &d, &buf(xs), &drw], &[h as u32, top_k as u32, rows as u32, streams as u32], (((rows * h) as u32).div_ceil(256), 1, 1));
            }
            None => rec.dispatch_wide("moe-wsum-rows", WSUM_ROWS, [&buf(out_d), &buf(sd), &buf(wv), &d, &d, &d, &buf(out), &drw], &[h as u32, top_k as u32, rows as u32], (((rows * h) as u32).div_ceil(256), 1, 1)),
        }
    }
}

impl ggml_rs::exl3::Experts for Exl3MoeGrouped {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
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

/// One projection of an expert: its packed weights on the GPU, or decoded on the CPU.
#[derive(Debug)]
pub(super) enum Proj {
    Gpu(Exl3Gpu),
    Cpu(Exl3Cpu),
}

impl Proj {
    pub(super) fn t(&self) -> &Transform {
        match self {
            Proj::Gpu(g) => &g.t,
            Proj::Cpu(c) => &c.t,
        }
    }
}

/// A MoE layer's EXL3 experts (the shared one last) without CUDA: each projection on the GPU while the weight budget
/// holds it, else decoded on the CPU, routed on the host exactly as `ggml_rs_cuda::exl3::Exl3Experts` routes.
///
/// A layer's work goes as two batches: every expert's gate and up projections for the rows routed to it, then every
/// down projection. The GPU's of a batch are recorded into one command encoder and read back together, one submit
/// for the lot (a decode step of Qwen3.8-Flash-Next would otherwise wait on 33 a layer); the CPU's run meanwhile, an
/// expert a thread.
pub struct Exl3MoeHost {
    experts: Vec<[Proj; 3]>,
    hidden: usize,
    ff: usize,
    gpu: Option<(Arc<Gpu>, Arc<Mutex<()>>)>,
}

impl std::fmt::Debug for Exl3MoeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let on_gpu = self.experts.iter().flatten().filter(|p| matches!(p, Proj::Gpu(_))).count();
        write!(f, "Exl3MoeHost({} experts of {}x{}, {} of {} projections on the GPU)", self.experts.len(), self.hidden, self.ff, on_gpu, 3 * self.experts.len())
    }
}

impl Exl3MoeHost {
    pub(super) fn new(experts: Vec<[Proj; 3]>, gpu: Option<(Arc<Gpu>, Arc<Mutex<()>>)>) -> Result<Self, String> {
        let first = experts.first().ok_or("no experts")?;
        let (hidden, ff) = (first[0].t().k, first[0].t().n);
        for e in &experts {
            let shapes = [e[0].t(), e[1].t(), e[2].t()].map(|t| (t.k, t.n));
            if shapes != [(hidden, ff), (hidden, ff), (ff, hidden)] {
                return Err("every expert needs gate and up of hidden -> ff, and down of ff -> hidden".into());
            }
        }
        Ok(Self { experts, hidden, ff, gpu })
    }

    /// Each job's projection applied to its prepared rows: `(expert, which projection, rows [count, k])`. The GPU's
    /// in one submit, the CPU's on threads meanwhile; each result post-transformed, `[count, n]`, in job order.
    fn batch(&self, jobs: &[(usize, usize, Vec<f32>)]) -> Vec<Vec<f32>> {
        let mut out: Vec<Option<Vec<f32>>> = (0..jobs.len()).map(|_| None).collect();
        let cpu_jobs: Vec<usize> = (0..jobs.len()).filter(|&i| matches!(self.experts[jobs[i].0][jobs[i].1], Proj::Cpu(_))).collect();
        let gpu_jobs: Vec<usize> = (0..jobs.len()).filter(|&i| matches!(self.experts[jobs[i].0][jobs[i].1], Proj::Gpu(_))).collect();
        let finish = |i: usize, mut y: Vec<f32>| -> Vec<f32> {
            let t = self.experts[jobs[i].0][jobs[i].1].t();
            let rows = jobs[i].2.len() / t.k;
            let mut o = vec![0f32; rows * t.n];
            for (yr, or) in y.chunks_exact_mut(t.n).zip(o.chunks_exact_mut(t.n)) {
                t.post(yr, or);
            }
            o
        };
        std::thread::scope(|scope| {
            // The CPU's experts, one a thread, while the GPU works.
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
            let per = cpu_jobs.len().div_ceil(threads).max(1);
            let cpu_work: Vec<_> = cpu_jobs
                .chunks(per)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|&i| {
                                let Proj::Cpu(c) = &self.experts[jobs[i].0][jobs[i].1] else { unreachable!() };
                                (i, c.matmul(&jobs[i].2, jobs[i].2.len() / c.t.k, 1))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            if let (Some((gpu, serial)), false) = (&self.gpu, gpu_jobs.is_empty()) {
                let _one = serial.lock().unwrap_or_else(|p| p.into_inner());
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                // A job of more rows than a pass takes goes as several passes.
                let mut passes = Vec::new();
                let mut owner = Vec::new();
                for &i in &gpu_jobs {
                    let Proj::Gpu(g) = &self.experts[jobs[i].0][jobs[i].1] else { unreachable!() };
                    let rows = jobs[i].2.len() / g.t.k;
                    for start in (0..rows).step_by(ROWS) {
                        let count = ROWS.min(rows - start);
                        passes.push(g.record(&mut enc, &jobs[i].2[start * g.t.k..(start + count) * g.t.k], count));
                        owner.push(i);
                    }
                }
                for (i, y) in owner.into_iter().zip(read_back(gpu, enc, &passes)) {
                    out[i].get_or_insert_with(Vec::new).extend(y);
                }
            }
            for w in cpu_work {
                for (i, y) in w.join().expect("an EXL3 expert worker panicked") {
                    out[i] = Some(y);
                }
            }
        });
        out.into_iter().enumerate().map(|(i, y)| finish(i, y.expect("every job ran"))).collect()
    }
}

impl ggml_rs::exl3::Experts for Exl3MoeHost {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let (h, f) = (self.hidden, self.ff);
        let x = x.to_host();
        let logits = logits.to_host();
        let rows = x.numel() / h;
        let width = self.experts.len();
        // Each row's (expert, weight) assignments, in the row's own order.
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect();
        // The rows routed to each expert, in row order: (row, its place among the row's assignments).
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); width];
        for (r, a) in assign.iter().enumerate() {
            for (j, &(e, _)) in a.iter().enumerate() {
                by_expert[e].push((r, j));
            }
        }
        let used: Vec<usize> = (0..width).filter(|&e| !by_expert[e].is_empty()).collect();
        // Gate and up, every expert's rows at once.
        let mut jobs = Vec::new();
        for &e in &used {
            for which in 0..2 {
                let t = self.experts[e][which].t();
                let mut xh = vec![0f32; by_expert[e].len() * h];
                for (slot, &(r, _)) in by_expert[e].iter().enumerate() {
                    t.pre(&x.data()[r * h..(r + 1) * h], &mut xh[slot * h..(slot + 1) * h]);
                }
                jobs.push((e, which, xh));
            }
        }
        let gu = self.batch(&jobs);
        // silu(gate) * up, then down, every expert's rows at once.
        let mut jobs = Vec::new();
        for (n, &e) in used.iter().enumerate() {
            let (g, u) = (&gu[2 * n], &gu[2 * n + 1]);
            let hidden: Vec<f32> = g.iter().zip(u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let t = self.experts[e][2].t();
            let mut xh = vec![0f32; by_expert[e].len() * f];
            for slot in 0..by_expert[e].len() {
                t.pre(&hidden[slot * f..(slot + 1) * f], &mut xh[slot * f..(slot + 1) * f]);
            }
            jobs.push((e, 2, xh));
        }
        let down = self.batch(&jobs);
        // Each row's experts summed in its own order, each weighted: the same sum every run.
        let mut placed: Vec<Vec<Option<&[f32]>>> = assign.iter().map(|a| vec![None; a.len()]).collect();
        for (n, &e) in used.iter().enumerate() {
            for (slot, &(r, j)) in by_expert[e].iter().enumerate() {
                placed[r][j] = Some(&down[n][slot * h..(slot + 1) * h]);
            }
        }
        let mut out = vec![0f32; rows * h];
        for (r, row) in out.chunks_exact_mut(h).enumerate() {
            for (j, &(_, w)) in assign[r].iter().enumerate() {
                let y = placed[r][j].expect("every assignment computed");
                for (o, v) in row.iter_mut().zip(y) {
                    *o += w * v;
                }
            }
        }
        Tensor::from_vec(out, vec![rows, h])
    }
}
