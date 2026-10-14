//! The recorder's attention: a prompt's rows (runs, tiled, or on the tensor cores), a step's query, and QSA's.
use super::*;

impl Recorder<'_> {
    /// [`ChainRecorder::attention_rows`] in f32: the positions in runs of 256 a workgroup a (head, run, query), then the
    /// runs joined.
    pub(crate) fn attention_rows_f32(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) {
        self.attention_rows_f32_masked(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, false)
    }

    /// [`Self::attention_rows_f32`], with `full` every query over every position (no causal mask): in one pass
    /// ([`ATTENTION_TILED`]) where [`attention_tiled_for`], its queries in chunks of bounded work
    /// ([`ATTENTION_DISPATCH_FLOPS`]); else in runs joined ([`Self::attention_rows_runs`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_f32_masked(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) {
        if full && window.is_none() && wide_attention_for(self.gpu(), head_dim, rows) && n_kv > 0 && n_h % n_kv == 0 {
            self.attention_rows_wide(q, kv, out, rows, n_h, n_kv, head_dim, past, scale, None);
            return;
        }
        // (a causal attention of heads the tensor cores' kernels do not take, 384 or 512 wide: Gemma 4's global layers'
        // 512, on the f32 tiles at some 280 GFLOPS on an NVIDIA A2; as matmuls, each run's scores masked)
        if !full && matches!(head_dim, 384 | 512) && rows >= 16 && n_kv > 0 && n_h % n_kv == 0 && (past + rows).div_ceil(16) * 16 <= 65535 && self.gpu().coop16() && wide_on() {
            self.attention_rows_wide(q, kv, out, rows, n_h, n_kv, head_dim, past + rows, scale, Some((past, window)));
            return;
        }
        if attention_tiled_for(rows, head_dim) {
            let per_row = 4.0 * (if full { past } else { past + rows }).max(1) as f64 * (n_h * head_dim) as f64;
            let tq = attention_tile(head_dim).0;
            let chunk = (((ATTENTION_DISPATCH_FLOPS / per_row) as usize) / tq * tq).max(tq);
            self.attention_rows_tiled(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, full, chunk);
        } else {
            self.attention_rows_runs(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, full);
        }
    }

    /// A full attention of heads wider than the tiled kernels take ([`wide_attention_for`]: Qwen Image 2.1's VAE's
    /// single head of 1,152) as matmuls on the tensor cores, a head at a time and its queries a run at a time: the run's
    /// scores against every key (`Q Kᵀ`, the keys f16 as an f16 matmul's weights), each row's weights (exp, the row's
    /// largest 1: f16's range) and their sum ([`SOFTMAX_ROWS_SUMS`]), their values (`P V`, the values transposed into the
    /// weights, [`TRANSPOSE_F16`]), over the sum. The runs' kernel it stands in for read every key and value again for each
    /// query, its scratch the queries times the runs times the head: a 1024x1024 picture's VAE attention 2.4 s on an
    /// NVIDIA A2, a 2048x2048 one's scratch 4.9 GB, past the card's largest buffer.
    ///
    /// `causal`: the queries' first position and their window (Some), each query then over the positions to its own
    /// (within its window) as the f32 kernels take them ([`SOFTMAX_ROWS_CAUSAL`]); None: every query over every
    /// position.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_wide(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, kv_len: usize, scale: f32, causal: Option<(usize, Option<usize>)>) {
        let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
        // (the keys padded to 16: their scores' rows, the values' matmul's sums; padding's weights 0. The queries and
        // keys padded to 16's too, with zeros, as the scores' matmul takes them: Qwen3-VL's vision heads of 72 as 80)
        let kp = kv_len.div_ceil(16) * 16;
        let hp = head_dim.div_ceil(16) * 16;
        assert!(rows > 0 && kv_len > 0 && kp <= 65535 && kv.len >= kv_len * row && q.len >= rows * qs && out.len >= rows * qs, "chain: a wide attention's buffers");
        // a run of queries: its scores at most 64M (256 MB)
        let run = ((64usize << 20) / kp / 64 * 64).clamp(64, rows.div_ceil(64) * 64);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let group = n_h / n_kv;
        for kh in 0..n_kv {
            let k16 = self.scratch(kp * hp / 2);
            let vt16 = self.scratch(head_dim * kp / 2);
            let words = [kv_len as u32, row as u32, (kh * head_dim) as u32, head_dim as u32, 0, 1, kp as u32, hp as u32];
            self.dispatch_wide("chain-slice-rows", SLICE_ROWS, [buffer(kv), &d, &d, &d, &d, &d, buffer(&k16), &drw], &words, grid(((kp * hp / 2) as u32).div_ceil(256)));
            let words = [kv_len as u32, row as u32, ((n_kv + kh) * head_dim) as u32, head_dim as u32, kp as u32, 0, 0, 0];
            self.dispatch_wide("chain-transpose-f16", TRANSPOSE_F16, [buffer(kv), &d, &d, &d, &d, &d, buffer(&vt16), &drw], &words, grid(((head_dim * kp / 2) as u32).div_ceil(256)));
            for h in kh * group..(kh + 1) * group {
                for r0 in (0..rows).step_by(run) {
                    let m = run.min(rows - r0);
                    let (qh, scores, sums, o) = (self.scratch(m * hp), self.scratch(m * kp), self.scratch(m), self.scratch(m * head_dim));
                    let words = [m as u32, qs as u32, (h * head_dim) as u32, head_dim as u32, r0 as u32, 0, m as u32, hp as u32];
                    self.dispatch_wide("chain-slice-rows", SLICE_ROWS, [buffer(q), &d, &d, &d, &d, &d, buffer(&qh), &drw], &words, grid(((m * hp) as u32).div_ceil(256)));
                    self.matmul_f16_rows(&k16, kp, hp, &qh, &scores, m);
                    match causal {
                        None => self.dispatch_wide("chain-softmax-rows-sums", SOFTMAX_ROWS_SUMS, [&d, &d, &d, &d, &d, &d, buffer(&scores), buffer(&sums)], &[kp as u32, kv_len as u32, m as u32, scale.to_bits()], grid(m as u32)),
                        Some((past, window)) => {
                            let words = [kp as u32, kv_len as u32, m as u32, scale.to_bits(), (past + r0) as u32, window.unwrap_or(0) as u32, 0, 0];
                            self.dispatch_wide("chain-softmax-rows-causal", SOFTMAX_ROWS_CAUSAL, [&d, &d, &d, &d, &d, &d, buffer(&scores), buffer(&sums)], &words, grid(m as u32))
                        }
                    }
                    self.matmul_f16_rows(&vt16, head_dim, kp, &scores, &o, m);
                    let words = [m as u32, head_dim as u32, qs as u32, (h * head_dim) as u32, r0 as u32, 0, 0, 0];
                    self.dispatch_wide("chain-divide-rows-into", DIVIDE_ROWS_INTO, [buffer(&o), buffer(&sums), &d, &d, &d, &d, buffer(out), &drw], &words, grid(((m * head_dim) as u32).div_ceil(256)));
                    for v in [qh, scores, sums, o] {
                        self.done_with(&v);
                    }
                }
            }
            self.done_with(&k16);
            self.done_with(&vt16);
        }
    }

    /// [`ATTENTION_TILED`]'s attention of `rows` queries, `chunk` of them a dispatch, each dispatch's work weighed
    /// ([`Recorder::weigh`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_tiled(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool, chunk: usize) {
        // (a full attention's `past` its positions, whatever its queries)
        let kv_len = if full { past } else { past + rows };
        assert!(
            rows > 0 && n_kv > 0 && n_h % n_kv == 0 && chunk > 0 && kv.len >= kv_len * 2 * n_kv * head_dim && q.len >= rows * n_h * head_dim && out.len >= rows * n_h * head_dim && window.unwrap_or(0) < 1 << 31,
            "chain: a prompt's attention's buffers"
        );
        let name = match head_dim {
            64 => "chain-attention-tiled-64",
            128 => "chain-attention-tiled-128",
            256 => "chain-attention-tiled-256",
            384 => "chain-attention-tiled-384",
            _ => "chain-attention-tiled-512",
        };
        let tq = attention_tile(head_dim).0;
        let pipeline = self.gpu().named_pipeline(name, || attention_tiled(head_dim));
        let mask = full as u32 | (window.unwrap_or(0) as u32) << 1;
        let mut first = 0;
        while first < rows {
            let n = chunk.min(rows - first);
            let params = self.uniform(&[n_h as u32, n_kv as u32, past as u32, n as u32, kv_len as u32, scale.to_bits(), first as u32, mask]);
            self.dispatch(&pipeline, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, n.div_ceil(tq) as u32, 1));
            self.weigh(4.0 * n as f64 * kv_len as f64 * (n_h * head_dim) as f64);
            first += n;
        }
    }

    /// [`Self::attention_rows_f32_masked`] in runs of 256 positions a workgroup a (head, run, query), the runs then
    /// joined: any head's width, its out [`attention_runs_out_len`] long.
    #[allow(clippy::too_many_arguments)]
    /// [`ChainRecorder::attention`], its parts a KV head's query heads together with `group` where they can be
    /// ([`attention_group_for`]), else a workgroup a query head.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_by(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32, group: bool) {
        let runs = kv_len.saturating_sub(lo).div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= cap * 2 * n_kv * head_dim && kv_len <= cap && out.len >= n_h * head_dim + n_h * runs * (head_dim + 2),
            "chain: attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, kv_len as u32, lo as u32, runs as u32, scale.to_bits()]);
        let per_kv = if n_kv > 0 && n_h % n_kv == 0 { n_h / n_kv } else { 0 };
        if let Some(g) = attention_group_of(per_kv, head_dim, self.gpu().limits.max_compute_workgroup_storage_size, 0).filter(|_| group) {
            // a KV head's query heads together (or in whole shares): its keys and values read once a group
            const NAMES: [&str; 9] = ["", "", "chain-attention-group-2", "chain-attention-group-3", "chain-attention-group-4", "chain-attention-group-5", "chain-attention-group-6", "chain-attention-group-7", "chain-attention-group-8"];
            let part = self.gpu().named_pipeline(NAMES[g], || attention_part_group(g));
            self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, ((n_h / g) as u32, runs as u32, 1));
        } else {
            // a head a multiple of 4 wide (at most 512): the vec4 kernel
            let part = if head_dim % 4 == 0 && head_dim <= 512 { self.gpu().named_pipeline("chain-attention-part4", || ATTENTION_PART4.to_string()) } else { self.named("chain-attention-part", ATTENTION_PART) };
            self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, 1));
        }
        let join = self.named("chain-attention-join", ATTENTION_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, 1, 1));
    }

    pub(crate) fn attention_rows_runs(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) {
        self.attention_rows_runs_by(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, full, head_dim % 4 == 0 && head_dim <= 512);
    }

    /// [`Self::attention_rows_runs`], its parts by [`ATTENTION_ROWS_PART4`] with `fours` (a head a multiple of 4 wide,
    /// at most 512), else a sum a load ([`ATTENTION_ROWS_PART`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_runs_by(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool, fours: bool) {
        // (a full attention's `past` its positions, whatever its queries)
        let kv_len = if full { past } else { past + rows };
        let runs = kv_len.div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= kv_len * 2 * n_kv * head_dim && q.len >= rows * n_h * head_dim && out.len >= rows * n_h * head_dim + rows * n_h * runs * (head_dim + 2),
            "chain: a prompt's attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, past as u32, window.unwrap_or(0) as u32, runs as u32, scale.to_bits(), rows as u32]);
        // (a full attention: every query's positions all `past` of them)
        let every = |body: &str| {
            let full = body.replace("    let hi = past + s + 1u;\n", "    let hi = past;\n");
            assert_ne!(full, body, "the parts' causal limit");
            full
        };
        let part = match (fours, full) {
            (true, true) => self.gpu().named_pipeline("chain-attention-rows-part4-full", || every(ATTENTION_ROWS_PART4)),
            (true, false) => self.gpu().named_pipeline("chain-attention-rows-part4", || ATTENTION_ROWS_PART4.to_string()),
            (false, true) => self.gpu().named_pipeline("chain-attention-rows-part-full", || format!("{HEAD}{}", every(ATTENTION_ROWS_PART))),
            (false, false) => self.named("chain-attention-rows-part", ATTENTION_ROWS_PART),
        };
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, rows as u32));
        let join = self.named("chain-attention-rows-join", ATTENTION_ROWS_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, rows as u32, 1));
        self.weigh(4.0 * rows as f64 * kv_len as f64 * (n_h * head_dim) as f64);
    }

    /// [`ChainRecorder::attention_rows`] on the tensor cores ([`ATTENTION_COOP`]): the queries and the cache's rows
    /// as f16 (the cache's each time, to the last query: a few microseconds a layer), the scores f16 into f32 and the
    /// softmax in f32, its weights f16. False (nothing recorded) where the device has no tensor cores, a window is
    /// kept, the head is not 64, 128 or 256 wide, or there are fewer than 16 rows.
    pub(crate) fn attention_rows_coop(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) -> bool {
        self.attention_rows_coop_masked(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, false)
    }

    /// [`Self::attention_rows_coop`], with `full` every query over every position (no causal mask).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_coop_masked(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) -> bool {
        self.attention_rows_coop_of(q, kv, false, out, rows, n_h, n_kv, head_dim, past, window, scale, full)
    }

    /// [`Self::attention_rows_coop_masked`] over a cache that is `halves` (its rows f16, two a word, as
    /// [`ChainRecorder::halve`] writes them) or f32.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_coop_of(&mut self, q: &DeviceVec, kv: &DeviceVec, halves: bool, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) -> bool {
        // (the cache's values: two a word where it is halves)
        let kv_values = if halves { kv.len * 2 } else { kv.len };
        let name = match (head_dim, full) {
            (64, false) => "chain-attention-coop-64",
            (128, false) => "chain-attention-coop-128",
            (256, false) => "chain-attention-coop-256",
            (64, true) => "chain-attention-coop-full-64",
            (128, true) => "chain-attention-coop-full-128",
            (256, true) => "chain-attention-coop-full-256",
            _ => return false,
        };
        if rows < 16 || n_kv == 0 || n_h % n_kv != 0 || !self.gpu().coop16() {
            return false;
        }
        // (OAIY_ATTENTION_KERNEL=rows: a full attention's queries 64 a workgroup, each subgroup's 16 rows its own)
        if full && window.is_none() && matches!(head_dim, 64 | 128) && rows_kernel() {
            let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
            assert!(kv_values >= past * row && q.len >= rows * qs && out.len >= rows.div_ceil(64) * 64 * qs, "chain: a full attention's buffers");
            let (q16, kv16) = self.attention_f16(q, kv, halves, rows, past, qs, row, Some((n_kv, head_dim)));
            let words = [n_h as u32, n_kv as u32, past as u32, rows as u32, past as u32, scale.to_bits(), 0, 0];
            let name = if head_dim == 64 { "chain-attention-coop-rows-64" } else { "chain-attention-coop-rows-128" };
            let pipeline = self.gpu().named_pipeline(name, || attention_coop_rows(head_dim));
            self.dispatch_kept(&pipeline, buffer(&kv16), buffer(&q16), buffer(out), &words, (n_h as u32, rows.div_ceil(64) as u32, 1));
            self.att16 = Some((q16, kv16));
            self.weigh(4.0 * rows as f64 * past as f64 * (n_h * head_dim) as f64);
            return true;
        }
        // (a full attention's `past` its positions, whatever its queries)
        let kv_len = if full { past } else { past + rows };
        let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
        let rp = rows.div_ceil(32) * 32;
        // the padding's rows are stored past the output (in a causal attention's scratch, at least as long: 16 rows or
        // more; a full one's out is as long as they need, `attention_rows_full_out_len`)
        assert!(
            kv_values >= kv_len * row && q.len >= rows * qs && out.len >= rp * qs && (full || out.len >= self.backend.attention_rows_out_len(rows, n_h, head_dim, kv_len)),
            "chain: a prompt's attention's buffers"
        );
        // (the keys 128 a block, their scores once; OAIY_ATTENTION_PASSES=2: twice, the first pass each query's largest
        // score and sum; OAIY_ATTENTION_NARROW: twice, the keys 32 a block)
        static KERNEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let kernel = *KERNEL.get_or_init(|| if std::env::var_os("OAIY_ATTENTION_NARROW").is_some() { 0 } else if std::env::var("OAIY_ATTENTION_PASSES").is_ok_and(|v| v == "2") { 2 } else { 1 });
        let words = [n_h as u32, n_kv as u32, past as u32, rows as u32, kv_len as u32, scale.to_bits(), 0, 0];
        let window = window.filter(|_| !full);
        // A cache held as halves is read where it lies, its rows' fragments loaded as they are (the one pass's kernel
        // or, with OAIY_ATTENTION_PASSES=2, the two passes'): not copied a fragment at a time first, a layer's cache
        // again (512 MB at 79,000 of the 27B's positions, which a 24 GB card beside its weights had not). Its rows
        // past `kv_len` are zeros or its own, whose weights are none (a cache without rows to 128 past them is copied).
        // (an attention whose query heads share their KV heads 1, 2 or 4 at a time: [`attention_coop_grouped`]'s kernel,
        // a KV head's queries a workgroup, the cache's rows read once for all of them; OAIY_ATTENTION_GROUPED=0: the
        // kernels below. A full one too, its `past` its every position, so each query sees them all: Qwen Image 2.1's 32
        // heads of 128 over a 1024 x 1024 picture's 4,127 positions in 3.9 s of a step's 13.2 with the one-pass kernel
        // on an NVIDIA A2)
        let g = n_h / n_kv;
        // (8 or more query heads a KV head as virtual KV heads of 4 each: Gemma 4's 8 on one)
        let (g, dup) = if g > 4 && g % 4 == 0 { (4, g / 4) } else { (g, 1) };
        // (its workgroup's memory near 48 KB, 256 threads; a window its own mask: the kernels below have none)
        let grouped = kernel == 1 && matches!(g, 1 | 2 | 4) && grouped_on() && self.gpu().limits.max_compute_workgroup_storage_size >= 48 << 10;
        if window.is_some_and(|w| w >= 1 << 31) || (window.is_some() || dup > 1) && !(grouped && (!halves || halves_in_place())) {
            return false;
        }
        if grouped && (!halves || halves_in_place()) {
            let words = [n_h as u32, (n_kv * dup) as u32, past as u32, rows as u32, kv_len as u32, scale.to_bits(), window.unwrap_or(0) as u32, dup as u32];
            let rp = rows.div_ceil(64) * 64;
            // (its workgroups' rows to 64: a full attention's out as long, `attention_rows_full_out_len`)
            assert!(!full || out.len >= rp * qs, "chain: a full attention's output");
            // the cache's rows as they lie where they are halves with room to 128 positions past `kv_len`, else as an
            // f16 copy of them so padded
            let (q16, kv16) = if halves && kv.len * 2 >= kv_len.div_ceil(128) * 128 * row {
                (self.attention_q16(q, rows, rp, qs), None)
            } else {
                let (q16, kv16) = self.attention_f16(q, kv, halves, rows, kv_len, qs, row, None);
                (q16, Some(kv16))
            };
            const NAMES: [[&str; 3]; 3] = [
                ["chain-attention-grouped-64-1", "chain-attention-grouped-128-1", "chain-attention-grouped-256-1"],
                ["chain-attention-grouped-64-2", "chain-attention-grouped-128-2", "chain-attention-grouped-256-2"],
                ["chain-attention-grouped-64-4", "chain-attention-grouped-128-4", "chain-attention-grouped-256-4"],
            ];
            let at = match head_dim { 64 => 0, 128 => 1, _ => 2 };
            let pipeline = self.gpu().named_pipeline(NAMES[g / 2][at], || attention_coop_grouped(head_dim, g));
            let per = 64 / g;
            self.dispatch_kept(&pipeline, kv16.as_ref().map_or(buffer(kv), buffer), buffer(&q16), buffer(out), &words, ((n_kv * dup) as u32, rows.div_ceil(per) as u32, 1));
            match kv16 {
                Some(kv16) => self.att16 = Some((q16, kv16)),
                // (the next layer's queries in it again)
                None => self.done_with(&q16),
            }
            self.weigh(4.0 * rows as f64 * kv_len as f64 * (n_h * head_dim) as f64);
            return true;
        }
        if halves && !full && halves_in_place() && kv.len * 2 >= kv_len.div_ceil(128) * 128 * row {
            let rp = rows.div_ceil(64) * 64;
            let q16 = self.attention_q16(q, rows, rp, qs);
            const NAMES: [[&str; 3]; 2] = [
                ["chain-attention-coop-one-halves-64", "chain-attention-coop-one-halves-128", "chain-attention-coop-one-halves-256"],
                ["chain-attention-coop-wide-halves-64", "chain-attention-coop-wide-halves-128", "chain-attention-coop-wide-halves-256"],
            ];
            let (two, at) = (kernel != 1, match head_dim { 64 => 0, 128 => 1, _ => 2 });
            let pipeline = self.gpu().named_pipeline(NAMES[usize::from(two)][at], || if two { attention_coop_wide(head_dim, false) } else { attention_coop_one_rows(head_dim) });
            self.dispatch_kept(&pipeline, buffer(kv), buffer(&q16), buffer(out), &words, (n_h as u32, (rows.div_ceil(32)) as u32, 1));
            // (the next layer's queries in it again)
            self.done_with(&q16);
            self.weigh(4.0 * rows as f64 * kv_len as f64 * (n_h * head_dim) as f64);
            return true;
        }
        // (the one pass reads the cache's copy a fragment at a time)
        let (q16, kv16) = self.attention_f16(q, kv, halves, rows, kv_len, qs, row, (kernel == 1).then_some((n_kv, head_dim)));
        let pipeline = self.gpu().named_pipeline(name, || match (kernel, full) {
            (1, _) => attention_coop_one(head_dim, full),
            (2, _) => attention_coop_wide(head_dim, full),
            (_, true) => attention_coop_full(head_dim),
            (_, false) => attention_coop(head_dim),
        });
        self.dispatch_kept(&pipeline, buffer(&kv16), buffer(&q16), buffer(out), &words, (n_h as u32, (rows.div_ceil(32)) as u32, 1));
        self.att16 = Some((q16, kv16));
        self.weigh(4.0 * rows as f64 * kv_len as f64 * (n_h * head_dim) as f64);
        true
    }

    /// A prompt's queries (`rows` of `qs`) as f16 for the tensor cores, padded to `padded` rows with zeros, in a scratch
    /// of the recording's (the cache's rows left where they are).
    fn attention_q16(&mut self, q: &DeviceVec, rows: usize, padded: usize, qs: usize) -> DeviceVec {
        let q16 = self.scratch(padded * qs / 2);
        let conv = self.gpu().named_pipeline("chain-x-f16", || crate::shaders::X_F16.to_string());
        let d = self.gpu().dummy().clone();
        let groups = ((padded * qs / 2) as u32).div_ceil(256);
        self.dispatch_kept(&conv, &d, buffer(q), buffer(&q16), &[qs as u32, rows as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        q16
    }

    /// A prompt's queries (`rows` of `qs`) and its cache's rows (`kv_len` of `row`) as f16 for the tensor cores'
    /// attention, each padded to 32 (the copies one pair a recording, grown as it needs: each attention's converted
    /// as it runs; put back in `att16` once used).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_f16(&mut self, q: &DeviceVec, kv: &DeviceVec, kv_halves: bool, rows: usize, kv_len: usize, qs: usize, row: usize, tiled: Option<(usize, usize)>) -> (DeviceVec, DeviceVec) {
        // (the keys to a block of the wide kernel's: 128; the queries to the rows kernel's workgroup: 64)
        let (rp, kp) = (rows.div_ceil(64) * 64, kv_len.div_ceil(128) * 128);
        let (q16, kv16) = match self.att16.take() {
            Some((a, b)) if a.len >= rp * qs / 2 && b.len >= kp * row / 2 => (a, b),
            _ => (self.scratch(rp * qs / 2), self.scratch(kp * row / 2)),
        };
        let conv = self.gpu().named_pipeline("chain-x-f16", || crate::shaders::X_F16.to_string());
        let d = self.gpu().dummy().clone();
        for (src, dst, width, n, padded) in [(q, &q16, qs, rows, rp), (kv, &kv16, row, kv_len, kp)] {
            let groups = ((padded * width / 2) as u32).div_ceil(256);
            // (a cache held as halves: its words as they are, laid out the same)
            let halved = kv_halves && std::ptr::eq(dst, &kv16);
            match tiled.filter(|_| std::ptr::eq(dst, &kv16)) {
                // the cache's a fragment at a time (`tiled`: its KV heads and their width), the one-pass kernels'
                Some((n_kv, head_dim)) => {
                    let tile = if halved {
                        self.gpu().named_pipeline("chain-kv-halves-tiled", || crate::shaders::KV_HALVES_TILED.to_string())
                    } else {
                        self.gpu().named_pipeline("chain-kv-f16-tiled", || crate::shaders::KV_F16_TILED.to_string())
                    };
                    self.dispatch_kept(&tile, &d, buffer(src), buffer(dst), &[n_kv as u32, head_dim as u32, n as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
                }
                None if halved => {
                    let copy = self.gpu().named_pipeline("chain-halves-padded", || crate::shaders::HALVES_PADDED.to_string());
                    self.dispatch_kept(&copy, &d, buffer(src), buffer(dst), &[width as u32, n as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
                }
                None => self.dispatch_kept(&conv, &d, buffer(src), buffer(dst), &[width as u32, n as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1)),
            }
        }
        (q16, kv16)
    }

    /// [`ChainRecorder::qsa_attention`] in f32: a query's entries in runs of 256, a workgroup a (group of a KV head's
    /// query heads, run, query) where they can be taken together ([`qsa_attention_part_group`]; OAIY_QSA_BY_HEAD: not),
    /// else a (head, run, query) ([`QSA_ATTENTION_PART`]), the runs joined.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_attention_f32(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) {
        static BY_HEAD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let group = !*BY_HEAD.get_or_init(|| std::env::var_os("OAIY_QSA_BY_HEAD").is_some());
        self.qsa_attention_f32_by(q, kv, list, out, rows, n_h, n_kv, head_dim, first, ratio, keep, scale, group)
    }

    /// [`Self::qsa_attention_f32`], its parts a group of query heads at once with `group` where they can be.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_attention_f32_by(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32, group: bool) {
        let runs = (keep * ratio + ratio).div_ceil(256);
        assert!(
            head_dim % 4 == 0 && head_dim <= 512 && q.len >= rows * n_h * head_dim && kv.len >= (first + rows) * 2 * n_kv * head_dim && list.len >= rows * keep && out.len >= self.backend.qsa_attention_out_len(rows, n_h, head_dim, keep, ratio),
            "chain: QSA's attention's buffers"
        );
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let words = [n_h as u32, n_kv as u32, head_dim as u32, first as u32, ratio as u32, runs as u32, scale.to_bits(), keep as u32];
        let per_kv = if n_kv > 0 && n_h % n_kv == 0 { n_h / n_kv } else { 0 };
        // (the entries' positions: 1 KB of the workgroup's memory besides the step's kernel's)
        match attention_group_of(per_kv, head_dim, self.gpu().limits.max_compute_workgroup_storage_size, 1024).filter(|_| group) {
            Some(g) => {
                const NAMES: [&str; 9] = ["", "", "chain-qsa-attention-group-2", "chain-qsa-attention-group-3", "chain-qsa-attention-group-4", "chain-qsa-attention-group-5", "chain-qsa-attention-group-6", "chain-qsa-attention-group-7", "chain-qsa-attention-group-8"];
                let source = self.gpu().named_source(NAMES[g], || qsa_attention_part_group(g));
                self.dispatch_wide(NAMES[g], source, [buffer(kv), buffer(q), buffer(list), &dd, &dd, &dd, buffer(out), &drw], &words, ((n_h / g) as u32, runs as u32, rows as u32));
            }
            None => self.dispatch_wide("chain-qsa-attention-part", QSA_ATTENTION_PART, [buffer(kv), buffer(q), buffer(list), &dd, &dd, &dd, buffer(out), &drw], &words, (n_h as u32, runs as u32, rows as u32)),
        }
        // the runs joined as a prompt's are (an empty run's sum 0 adds nothing)
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, first as u32, 0, runs as u32, scale.to_bits(), rows as u32]);
        let join = self.named("chain-attention-rows-join", ATTENTION_ROWS_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, rows as u32, 1));
    }

    /// [`ChainRecorder::qsa_attention`] of a prompt's rows on the tensor cores ([`attention_coop_masked`]): the kept
    /// blocks as each query's bitmask ([`QSA_MASK`]), then every position up to the query's on the tensor cores but
    /// those of a block it did not keep (the dense span's work past it, which the tensor cores still do the sooner:
    /// 512 of some 640 blocks kept at 2,560 positions). False (nothing recorded) as [`Self::attention_rows_coop`] is.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_attention_coop(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) -> bool {
        let name = match head_dim {
            64 => "chain-qsa-attention-coop-64",
            128 => "chain-qsa-attention-coop-128",
            256 => "chain-qsa-attention-coop-256",
            _ => return false,
        };
        if rows < 16 || ratio == 0 || n_kv == 0 || n_h % n_kv != 0 || !self.gpu().coop16() {
            return false;
        }
        let kv_len = first + rows;
        let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
        // the padding's rows go past the output, in its scratch (at least as long as they: 16 rows or more)
        assert!(out.len >= rows.div_ceil(32) * 32 * qs, "chain: QSA's attention's buffers");
        let mw = (kv_len / ratio).div_ceil(32).max(1);
        let mask = self.scratch(rows * mw);
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let words = (rows * mw) as u32;
        let groups = words.div_ceil(256);
        self.dispatch_wide("chain-qsa-mask", QSA_MASK, [buffer(list), &d, &d, &d, &d, &d, buffer(&mask), &drw], &[rows as u32, keep as u32, mw as u32, first as u32, ratio as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        // (the keys 128 a block, their scores once; OAIY_ATTENTION_PASSES=2 or OAIY_ATTENTION_NARROW: twice, 32 a block)
        static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let old = *OLD.get_or_init(|| std::env::var_os("OAIY_ATTENTION_NARROW").is_some() || std::env::var("OAIY_ATTENTION_PASSES").is_ok_and(|v| v == "2"));
        let (q16, kv16) = self.attention_f16(q, kv, false, rows, kv_len, qs, row, (!old).then_some((n_kv, head_dim)));
        let src = if old { attention_coop_masked(head_dim) } else { attention_coop_one_masked(head_dim) };
        let words = [n_h as u32, n_kv as u32, first as u32, rows as u32, kv_len as u32, scale.to_bits(), ratio as u32, mw as u32];
        self.dispatch_wide(name, &src, [buffer(&kv16), buffer(&q16), buffer(&mask), &d, &d, &d, buffer(out), &drw], &words, (n_h as u32, rows.div_ceil(32) as u32, 1));
        self.att16 = Some((q16, kv16));
        true
    }
}

/// Whether a prompt's attention reads a cache held as halves where it lies (OAIY_HALVES_TILED=1: copied a fragment at a
/// time for the one-pass kernel, as an f32 cache is).
fn halves_in_place() -> bool {
    static IN_PLACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *IN_PLACE.get_or_init(|| std::env::var("OAIY_HALVES_TILED").map_or(true, |v| v != "1"))
}

/// Whether a full attention of heads `head_dim` wide is [`Recorder::attention_rows_wide`]'s: a head no other kernel takes
/// (wider than the tiled kernels' 512, or of another width than theirs and the tensor cores' kernels': Qwen3-VL's
/// vision tower's 72), on the tensor cores (OAIY_WIDE_ATTENTION=0: the runs' kernel). A 1024x1024 reference's 4,096
/// patches through the vision tower's 27 blocks of 16 heads of 72: 45.5 s on an NVIDIA A2 with the runs' kernel.
pub(crate) fn wide_attention_for(gpu: &Gpu, head_dim: usize, rows: usize) -> bool {
    // (a few queries' attention, a check of drafts' over a cache, the runs' kernel's: a matmul a head is more work
    // than their dot products; a head wider than the others' is this one's whatever its queries)
    let other = !matches!(head_dim, 64 | 128 | 256 | 384 | 512) && rows >= 16;
    (head_dim > 512 || other) && head_dim % 4 == 0 && gpu.coop16() && wide_on()
}

/// Whether attentions the tensor cores' kernels do not take are matmuls ([`Recorder::attention_rows_wide`];
/// OAIY_WIDE_ATTENTION=0: the f32 kernels).
fn wide_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OAIY_WIDE_ATTENTION").map_or(true, |v| v != "0"))
}

/// Whether an attention takes [`attention_coop_grouped`]'s kernel where it can (OAIY_ATTENTION_GROUPED=0: not).
fn grouped_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OAIY_ATTENTION_GROUPED").map_or(true, |v| v != "0"))
}

/// Whether a full attention takes [`attention_coop_rows`]'s kernel (OAIY_ATTENTION_KERNEL=rows).
pub(crate) fn rows_kernel() -> bool {
    static ROWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| std::env::var("OAIY_ATTENTION_KERNEL").is_ok_and(|v| v == "rows"))
}
