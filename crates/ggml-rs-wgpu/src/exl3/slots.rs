//! EXL3 experts in slots on a GPU, for a model that routes and combines its experts itself (DeepSeek-V4.1's EXL3
//! checkpoints): each slot an expert's three projections, put in from its record as the file lays it out
//! ([`RecordLayout`]) whenever the model wants it there, and a call's jobs (a slot and some rows each) run through the
//! groups' kernels as [`Exl3MoeGrouped`]'s are: gate and up, the SwiGLU as the down projection reads it (clamped at the
//! model's limit), then down. Each job's output is its own (`[rows, hidden]`, unweighted): the model weights and sums
//! them in its own order. No shared expert (the model runs its own).

use super::*;
use ggml_rs::ChainRecorder;

/// Experts of one shape and bitrate in slots on one GPU ([`WgpuBackend::exl3_slots`]).
pub struct Exl3Slots {
    b: WgpuBackend,
    gu: Group,
    down: Group,
    slots: usize,
    hidden: usize,
    ff: usize,
    /// Bytes counted against the weight budget while these live.
    nbytes: u64,
    /// Whether the host writes the matrices' words in place ([`WgpuBackend::exl3_slots_mapped`]).
    mapped: bool,
}

impl std::fmt::Debug for Exl3Slots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exl3Slots({} slots of {}x{} experts at {}/{} words a tile)", self.slots, self.hidden, self.ff, self.gu.tw, self.down.tw)
    }
}

impl Drop for Exl3Slots {
    fn drop(&mut self) {
        self.b.used.fetch_sub(self.nbytes, Ordering::Relaxed);
    }
}

impl WgpuBackend {
    /// Up to `slots` slots for `hidden -> ff` experts at `tw` words a tile (the gate's and up's, the down's), as many as
    /// a binding holds (a group's matrices are one buffer) and the weight budget has room for; None if not one.
    pub fn exl3_slots(&self, slots: usize, hidden: usize, ff: usize, tw: [usize; 2]) -> Option<Exl3Slots> {
        self.exl3_slots_with(slots, hidden, ff, tw, false)
    }

    /// [`Self::exl3_slots`] whose matrices' words the host writes in place ([`Exl3Slots::put_many`]): buffers in the
    /// card's memory it maps for writing (through its BAR, where the device has mappable storage buffers; else as
    /// [`Self::exl3_slots`]'s), for slots that take records as passes ask for them. Measured on an RTX 5090, the host's
    /// threads write there at 21.7 GB/s over eight lanes and 11.1 over four, where the queue's writes went at 2.6 to 3.5
    /// (its staging in the host's memory, copied by the GPU at 1.7 to 2.5).
    pub fn exl3_slots_mapped(&self, slots: usize, hidden: usize, ff: usize, tw: [usize; 2]) -> Option<Exl3Slots> {
        let can = self.gpu.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        self.exl3_slots_with(slots, hidden, ff, tw, can)
    }

    fn exl3_slots_with(&self, slots: usize, hidden: usize, ff: usize, tw: [usize; 2], mapped: bool) -> Option<Exl3Slots> {
        if slots == 0 || hidden % 128 != 0 || ff % 128 != 0 || !tw.iter().all(|t| matches!(t, 16 | 24 | 32 | 40 | 48 | 56 | 64 | 80 | 96 | 112 | 128)) {
            return None;
        }
        let mwords = |k: usize, n: usize, tw: usize| k / 16 * n / 16 * (tw / 2);
        let (gu_words, d_words) = (mwords(hidden, ff, tw[0]), mwords(ff, hidden, tw[1]));
        let per = (2 * gu_words * 4 + d_words * 4 + 3 * (hidden + ff) * 4) as u64;
        let limit = chunk_limit(&self.gpu.limits);
        let by_binding = (limit / (2 * gu_words * 4) as u64).min(limit / (d_words * 4) as u64) as usize;
        let free = self.budget.saturating_sub(self.used.load(Ordering::Relaxed));
        let n = slots.min(by_binding).min((free / per) as usize);
        if n == 0 {
            return None;
        }
        let nbytes = n as u64 * per;
        self.used.fetch_add(nbytes, Ordering::Relaxed);
        let group = |matrices: usize, k: usize, n: usize, tw: usize| -> Group {
            let mw = mwords(k, n, tw);
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
            let words = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-exl3-slots"),
                size: (matrices * mw * 4) as u64,
                usage: if mapped { usage | wgpu::BufferUsages::MAP_WRITE } else { usage },
                mapped_at_creation: false,
            });
            let ntiles = n / 16;
            let splits = 4096usize.div_ceil(ntiles).next_power_of_two().min(8).min(k / 16).max(1) as u32;
            Group { words, suh: self.vec(matrices * k), svh: self.vec(matrices * n), k, n, tw, mwords: mw, splits }
        };
        let gu = group(2 * n, hidden, ff, tw[0]);
        let down = group(n, ff, hidden, tw[1]);
        Some(Exl3Slots { b: self.clone(), gu, down, slots: n, hidden, ff, nbytes, mapped })
    }
}

impl Exl3Slots {
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The bytes these take on the GPU.
    pub fn nbytes(&self) -> u64 {
        self.nbytes
    }

    /// Whether an expert laid out as `layout` lays it out goes into these slots (its shapes and bitrates theirs).
    pub fn fits(&self, layout: &RecordLayout) -> bool {
        let [g, u, d] = &layout.proj;
        (g.k, g.n, g.tile_words) == (self.hidden, self.ff, self.gu.tw) && (u.k, u.n, u.tile_words) == (self.hidden, self.ff, self.gu.tw) && (d.k, d.n, d.tile_words) == (self.ff, self.hidden, self.down.tw)
    }

    /// Slot `slot`'s expert made `record`'s (as `layout` lays it out, which [`Self::fits`]): its matrices' words and
    /// their transforms' tables written where the groups keep that slot's, there for every call after.
    pub fn put(&self, slot: usize, record: &[u8], layout: &RecordLayout) {
        assert!(slot < self.slots && self.fits(layout), "exl3 slots: slot {slot} of {}, or an expert of another shape", self.slots);
        if self.mapped {
            return self.put_many(slot, &[record], layout);
        }
        let gpu = &self.b.gpu;
        let floats = |at: usize, len: usize| -> Vec<u8> {
            (0..len).flat_map(|i| half::f16::from_bits(u16::from_le_bytes([record[at + 2 * i], record[at + 2 * i + 1]])).to_f32().to_le_bytes()).collect()
        };
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector").clone();
        for (which, (g, m)) in [(&self.gu, 2 * slot), (&self.gu, 2 * slot + 1), (&self.down, slot)].into_iter().enumerate() {
            let p = &layout.proj[which];
            gpu.write(&g.words, (m * g.mwords * 4) as u64, &record[p.trellis..p.trellis + g.mwords * 4]);
        }
        let (g, u, d) = (&layout.proj[0], &layout.proj[1], &layout.proj[2]);
        gpu.write(&buf(&self.gu.suh), (2 * slot * self.gu.k * 4) as u64, &[floats(g.suh, g.k), floats(u.suh, u.k)].concat());
        gpu.write(&buf(&self.gu.svh), (2 * slot * self.gu.n * 4) as u64, &[floats(g.svh, g.n), floats(u.svh, u.n)].concat());
        gpu.write(&buf(&self.down.suh), (slot * self.down.k * 4) as u64, &floats(d.suh, d.k));
        gpu.write(&buf(&self.down.svh), (slot * self.down.n * 4) as u64, &floats(d.svh, d.n));
    }

    /// Slots `first..` made `records`' experts (each as `layout` lays it out), as [`Self::put`] makes one. Mapped slots
    /// ([`WgpuBackend::exl3_slots_mapped`]) take the gate's and up's words of them all and the down's (the slots' words
    /// lie together) written in place by the host's threads, at once; others, each record's as [`Self::put`] writes
    /// it. Their tables in one write each. Nothing may be running on these slots.
    pub fn put_many(&self, first: usize, records: &[&[u8]], layout: &RecordLayout) {
        let n = records.len();
        assert!(first + n <= self.slots && self.fits(layout), "exl3 slots: slots {first}..{} of {}, or an expert of another shape", first + n, self.slots);
        if n == 0 {
            return;
        }
        if !self.mapped {
            for (j, record) in records.iter().enumerate() {
                self.put(first + j, record, layout);
            }
            return;
        }
        let gpu = &self.b.gpu;
        let [pg, pu, pd] = &layout.proj;
        // the words: each slot's gate then up (the group's matrices 2s and 2s + 1), and its down
        let (gw, dw) = (self.gu.mwords * 4, self.down.mwords * 4);
        let gu_fill = |from: usize, part: &mut [u8]| {
            let mut at = 0;
            while at < part.len() {
                let byte = from + at;
                let (j, m, o) = (byte / (2 * gw), byte % (2 * gw) / gw, byte % gw);
                let start = [pg.trellis, pu.trellis][m] + o;
                let take = (gw - o).min(part.len() - at);
                part[at..at + take].copy_from_slice(&records[j][start..start + take]);
                at += take;
            }
        };
        let d_fill = |from: usize, part: &mut [u8]| {
            let mut at = 0;
            while at < part.len() {
                let byte = from + at;
                let (j, o) = (byte / dw, byte % dw);
                let take = (dw - o).min(part.len() - at);
                part[at..at + take].copy_from_slice(&records[j][pd.trellis + o..pd.trellis + o + take]);
                at += take;
            }
        };
        gpu.write_mapped(&[
            (&self.gu.words, (2 * first * gw) as u64, 2 * n * gw, &gu_fill as &(dyn Fn(usize, &mut [u8]) + Sync)),
            (&self.down.words, (first * dw) as u64, n * dw, &d_fill),
        ]);
        let floats = |record: &[u8], at: usize, len: usize| -> Vec<u8> {
            (0..len).flat_map(|i| half::f16::from_bits(u16::from_le_bytes([record[at + 2 * i], record[at + 2 * i + 1]])).to_f32().to_le_bytes()).collect()
        };
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector").clone();
        let suh_gu: Vec<u8> = records.iter().flat_map(|r| [floats(r, pg.suh, pg.k), floats(r, pu.suh, pu.k)].concat()).collect();
        let svh_gu: Vec<u8> = records.iter().flat_map(|r| [floats(r, pg.svh, pg.n), floats(r, pu.svh, pu.n)].concat()).collect();
        let suh_d: Vec<u8> = records.iter().flat_map(|r| floats(r, pd.suh, pd.k)).collect();
        let svh_d: Vec<u8> = records.iter().flat_map(|r| floats(r, pd.svh, pd.n)).collect();
        gpu.write(&buf(&self.gu.suh), (2 * first * self.gu.k * 4) as u64, &suh_gu);
        gpu.write(&buf(&self.gu.svh), (2 * first * self.gu.n * 4) as u64, &svh_gu);
        gpu.write(&buf(&self.down.suh), (first * self.down.k * 4) as u64, &suh_d);
        gpu.write(&buf(&self.down.svh), (first * self.down.n * 4) as u64, &svh_d);
    }

    /// `to`'s slot `at` made what slot `from` of these holds (both sets on one GPU, of one shape and bitrate): its
    /// matrices' words and tables copied by the GPU, in its turn after the work submitted before it (and before any
    /// later write into these, which waits for it). An expert a pass put into scratch slots is kept that way for a
    /// copy of the card's own memory, where its record written again was the upload it had just had.
    pub fn copy_slot(&self, from: usize, to: &Exl3Slots, at: usize) {
        assert!(Arc::ptr_eq(&self.b.gpu, &to.b.gpu) && from < self.slots && at < to.slots, "exl3 slots: a copy between GPUs, or past the slots");
        assert!((self.hidden, self.ff, self.gu.tw, self.down.tw) == (to.hidden, to.ff, to.gu.tw, to.down.tw), "exl3 slots: a copy between experts of other shapes");
        let gpu = &self.b.gpu;
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector").clone();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("oaiy-exl3-slot-copy") });
        for (a, b, per) in [(&self.gu, &to.gu, 2usize), (&self.down, &to.down, 1)] {
            let words = (per * a.mwords * 4) as u64;
            enc.copy_buffer_to_buffer(&a.words, from as u64 * words, &b.words, at as u64 * words, words);
            let (suh, svh) = ((per * a.k * 4) as u64, (per * a.n * 4) as u64);
            enc.copy_buffer_to_buffer(&buf(&a.suh), from as u64 * suh, &buf(&b.suh), at as u64 * suh, suh);
            enc.copy_buffer_to_buffer(&buf(&a.svh), from as u64 * svh, &buf(&b.svh), at as u64 * svh, svh);
        }
        gpu.queue().submit([enc.finish()]);
    }

    /// `jobs`' experts (each a slot and its rows, `[rows, hidden]`), in one recording: gate and up, the SwiGLU clamped
    /// at `limit` (0: none) as the down projection reads it, then down. Each job's output, `[rows, hidden]`, unweighted.
    /// A job of one row runs as a decode step's (a workgroup a tile column, split for a large GPU), more rows as a
    /// prompt's (each expert's rows in blocks, a tile decoded once a block).
    pub fn forward(&self, jobs: &[(usize, &[f32])], limit: f32) -> Vec<Vec<f32>> {
        Self::forward_many(&[(self, jobs)], limit).pop().expect("one set of slots, one set of outputs")
    }

    /// [`Self::forward`] of several sets of slots on one GPU (a card's pools: a binding holds some 240 experts), in one
    /// recording and one read back: each set's jobs' outputs. A decode step's held experts lie in several pools of a
    /// card, each pool's call its own round trip before.
    ///
    /// A job of one row is recorded as a decode step's and a job of more as a prompt's whatever it is called with (a
    /// set's jobs of each kind recorded apart): its sums are made in one order, so its output is the same bits
    /// whichever jobs share its call. (A call's jobs went one way for all of them, as its rows had it, and a prompt's
    /// output then hung on how its experts happened to be grouped.)
    pub fn forward_many(parts: &[(&Exl3Slots, &[(usize, &[f32])])], limit: f32) -> Vec<Vec<Vec<f32>>> {
        let Some(&(first, _)) = parts.first() else { return Vec::new() };
        assert!(parts.iter().all(|(s, _)| Arc::ptr_eq(&s.b.gpu, &first.b.gpu)), "exl3 slots: sets on more than one GPU in one call");
        let mut outs: Vec<Vec<Vec<f32>>> = parts.iter().map(|(_, jobs)| vec![Vec::new(); jobs.len()]).collect();
        // (set, its jobs' places in its list, the jobs): each set's jobs of one row, then of more
        let mut subs: Vec<(usize, Vec<usize>, Vec<(usize, &[f32])>)> = Vec::new();
        for (i, (s, jobs)) in parts.iter().enumerate() {
            for one in [true, false] {
                let mine: Vec<usize> = (0..jobs.len()).filter(|&j| !jobs[j].1.is_empty() && (jobs[j].1.len() == s.hidden) == one).collect();
                if !mine.is_empty() {
                    let these = mine.iter().map(|&j| jobs[j]).collect();
                    subs.push((i, mine, these));
                }
            }
        }
        if subs.is_empty() {
            return outs;
        }
        let mut rec = crate::chain::Recorder::new(&first.b);
        rec.keep_groups(false);
        let recorded: Vec<(DeviceVec, Vec<usize>)> = subs.iter().map(|(i, _, these)| parts[*i].0.record(&mut rec, these, limit).expect("jobs with rows")).collect();
        for (out, _) in &recorded {
            rec.read(out);
        }
        for (((i, mine, _), (_, rows)), y) in subs.iter().zip(&recorded).zip(Box::new(rec).finish()) {
            let hidden = parts[*i].0.hidden;
            let mut at = 0;
            for (&j, &n) in mine.iter().zip(rows) {
                outs[*i][j] = y[at..at + n * hidden].to_vec();
                at += n * hidden;
            }
        }
        outs
    }

    /// `jobs` recorded into `rec` as [`Self::forward`] runs them: the vector their outputs will be in (`[rows, hidden]`
    /// each, in order) and each job's rows; None where they have no rows.
    fn record(&self, rec: &mut crate::chain::Recorder, jobs: &[(usize, &[f32])], limit: f32) -> Option<(DeviceVec, Vec<usize>)> {
        let (h, f) = (self.hidden, self.ff);
        let rows: Vec<usize> = jobs.iter().map(|(_, x)| x.len() / h).collect();
        let pairs: usize = rows.iter().sum();
        if pairs == 0 {
            return None;
        }
        assert!(jobs.iter().all(|&(s, x)| s < self.slots && x.len() % h == 0), "exl3 slots: a job's slot or rows");
        // the jobs: gate and up of each row on its row of x, then down of each on its hidden row
        let (mut jobs_gu, mut jobs_d, mut xs) = (Vec::with_capacity(4 * pairs), Vec::with_capacity(2 * pairs), Vec::with_capacity(pairs * h));
        let mut r = 0u32;
        for &(slot, x) in jobs {
            for row in x.chunks_exact(h) {
                jobs_gu.extend([2 * slot as u32, r, 2 * slot as u32 + 1, r]);
                jobs_d.extend([slot as u32, r]);
                xs.extend_from_slice(row);
                r += 1;
            }
        }
        let one = rows.iter().all(|&n| n == 1);
        let b = &self.b;
        let up = |v: &DeviceVec, data: &[u32]| DeviceChain::upload(b, v, &data.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>());
        let (x, jgu, jd) = (rec.scratch(pairs * h), rec.scratch(4 * pairs), rec.scratch(2 * pairs));
        DeviceChain::upload(b, &x, &xs);
        up(&jgu, &jobs_gu);
        up(&jd, &jobs_d);
        let (sgu, sd) = if one { (self.gu.splits as usize, self.down.splits as usize) } else { (1, 1) };
        let (xh_gu, part_gu, out_gu) = (rec.scratch(2 * pairs * h), rec.scratch(2 * pairs * sgu * f), rec.scratch(2 * pairs * f));
        let (xh_d, part_d, out_d) = (rec.scratch(pairs * f), rec.scratch(pairs * sd * h), rec.scratch(pairs * h));
        if one {
            group_pass(rec, &self.gu, &x, false, &jgu, 2 * pairs, Order::Jobs, &xh_gu, &part_gu, &out_gu, 0.0);
            group_pass(rec, &self.down, &out_gu, true, &jd, pairs, Order::Jobs, &xh_d, &part_d, &out_d, limit);
        } else {
            // a prompt's rows: each expert's in blocks, a tile decoded once a block
            let block = many_rows(rec.gpu());
            let (ogu, od) = (many_order(&jobs_gu, block), many_order(&jobs_d, block));
            let (ogv, odv) = (rec.scratch(ogu.len()), rec.scratch(od.len()));
            up(&ogv, &ogu);
            up(&odv, &od);
            group_pass(rec, &self.gu, &x, false, &jgu, 2 * pairs, Order::Many(&ogv, ogu.len() / block, block), &xh_gu, &part_gu, &out_gu, 0.0);
            group_pass(rec, &self.down, &out_gu, true, &jd, pairs, Order::Many(&odv, od.len() / block, block), &xh_d, &part_d, &out_d, limit);
        }
        Some((out_d, rows))
    }
}
