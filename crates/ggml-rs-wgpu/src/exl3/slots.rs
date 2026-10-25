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
    /// A step's vectors (jobs of one row each, up to [`STEP_JOBS`]), made at its first call and kept with their bind
    /// groups for every call after; held by the call that uses them.
    step: Mutex<Option<StepScratch>>,
    /// [`silu_table`]'s for the limit the calls give, on this card: the down projection's SwiGLU looks it up.
    silu: Mutex<Option<(u32, DeviceVec)>>,
}

/// What a begun call holds until it is finished: the locks of the kept vectors it used (an [`Exl3Slots`]' or a
/// [`crate::quant::moe::QuantSlots`]').
pub(crate) trait Held {}
impl<T> Held for T {}

/// A call of sets of slots begun ([`Exl3Slots::begin_many`], [`crate::quant::moe::QuantSlots::begin_many`]): its
/// recording submitted, each set's jobs' outputs to come ([`SlotsBegun::finish`]). The sets' kept vectors it used are
/// held until then.
pub struct SlotsBegun<'a> {
    pub(crate) launched: Option<crate::chain::Launched<'a>>,
    pub(crate) held: Vec<Box<dyn Held + 'a>>,
    /// Each part of the recording: its set, its jobs' places in the set's list, each job's rows, the set's width.
    pub(crate) subs: Vec<(usize, Vec<usize>, Vec<usize>, usize)>,
    /// Each set's jobs.
    pub(crate) jobs: Vec<usize>,
}

impl SlotsBegun<'_> {
    /// Whether the call is done (its outputs there to read).
    pub fn done(&self) -> bool {
        self.launched.as_ref().is_none_or(|l| l.done())
    }

    /// Each set's jobs' outputs (`[rows, hidden]` each, unweighted), once the call is done.
    pub fn finish(self) -> Vec<Vec<Vec<f32>>> {
        let SlotsBegun { launched, held, subs, jobs } = self;
        let mut outs: Vec<Vec<Vec<f32>>> = jobs.iter().map(|&n| vec![Vec::new(); n]).collect();
        let Some(launched) = launched else { return outs };
        let ys = launched.wait();
        drop(held);
        for ((i, mine, rows, hidden), y) in subs.into_iter().zip(ys) {
            let mut at = 0;
            for (j, n) in mine.into_iter().zip(rows) {
                outs[i][j] = y[at..at + n * hidden].to_vec();
                at += n * hidden;
            }
        }
        outs
    }
}

/// One-row jobs a call's kept vectors take ([`Exl3Slots::step`]): a decode step's or a draft's experts in one set of
/// slots, most of a layer's six or three. A call of more goes through the recording's scratch.
const STEP_JOBS: usize = 8;

/// [`Exl3Slots::record`]'s vectors for up to [`STEP_JOBS`] jobs of one row: the inputs, the two job lists, and each
/// group's transformed inputs, partial sums and outputs.
struct StepScratch {
    x: DeviceVec,
    jgu: DeviceVec,
    jd: DeviceVec,
    xh_gu: DeviceVec,
    part_gu: DeviceVec,
    out_gu: DeviceVec,
    xh_d: DeviceVec,
    part_d: DeviceVec,
    out_d: DeviceVec,
}

/// [`Exl3Slots::record_fused`]'s vectors beside its inputs and output: each group's transformed inputs and partial sums,
/// and the gate's and up's outputs.
pub(super) struct FusedScratch {
    xh_gu: DeviceVec,
    part_gu: DeviceVec,
    out_gu: DeviceVec,
    xh_d: DeviceVec,
    part_d: DeviceVec,
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
            let splits = step_splits(k, n) as u32;
            Group { words, suh: self.vec(matrices * k), svh: self.vec(matrices * n), k, n, tw, mwords: mw, splits }
        };
        let gu = group(2 * n, hidden, ff, tw[0]);
        let down = group(n, ff, hidden, tw[1]);
        Some(Exl3Slots { b: self.clone(), gu, down, slots: n, hidden, ff, nbytes, mapped, step: Mutex::new(None), silu: Mutex::new(None) })
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
        // (the writes sent, and their staging let go: slots filled one after another while nothing else is submitted
        // held every record's staging until the next pass)
        gpu.queue().submit([]);
        gpu.wait(None);
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
        self.copy_slot_marked(from, to, at, &[]);
    }

    /// [`Self::copy_slot`], and `marks` (`(table, value's place, value)` each: the table of where a layer's experts are
    /// that a device program reads, [`super::ProgramStep::Candidates`]) written in the same submission: a recording
    /// submitted before it finds the old expert in the slot and the old table, one after it the new in both.
    pub fn copy_slot_marked(&self, from: usize, to: &Exl3Slots, at: usize, marks: &[(&DeviceVec, usize, u32)]) {
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
        // (the marks' writes go to the queue with the next submission, this one: no other submits between them, as
        // every recording submits under the same lock)
        let _one = self.b.serial.lock().unwrap_or_else(|p| p.into_inner());
        for (table, place, value) in marks {
            assert!(*place < table.len, "exl3 slots: a mark past its table");
            gpu.queue().write_buffer(&buf(table), (*place * 4) as u64, &value.to_le_bytes());
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
        Self::begin_many(parts, limit).finish()
    }

    /// [`Self::forward_many`] begun: recorded and submitted, its outputs left for [`SlotsBegun::finish`], so what the
    /// caller does meanwhile runs beside the card's work (a decode step's held experts on two cards and the reads of
    /// the rest, with no thread started for either: a thread a card a layer before).
    pub fn begin_many<'a>(parts: &[(&'a Exl3Slots, &[(usize, &[f32])])], limit: f32) -> SlotsBegun<'a> {
        let jobs: Vec<usize> = parts.iter().map(|(_, jobs)| jobs.len()).collect();
        let Some(&(first, _)) = parts.first() else { return SlotsBegun { launched: None, held: Vec::new(), subs: Vec::new(), jobs } };
        assert!(parts.iter().all(|(s, _)| Arc::ptr_eq(&s.b.gpu, &first.b.gpu)), "exl3 slots: sets on more than one GPU in one call");
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
            return SlotsBegun { launched: None, held: Vec::new(), subs: Vec::new(), jobs };
        }
        let mut rec = crate::chain::Recorder::new(&first.b);
        // A step's jobs (one row each, a few to a set of slots) through each set's kept vectors, their bind groups kept
        // with them: making them was a quarter of a call of six experts in three sets. Kept for a recording of nothing
        // else, so the device's kept groups hold the same few call after call (the recording's scratch is whatever the
        // pool hands out, and groups kept of it filled the device's store with new ones a call, the trunk's own among
        // what it let go). A prompt's are made anew.
        let mut steps: Vec<Option<std::sync::MutexGuard<'a, Option<StepScratch>>>> = subs
            .iter()
            .map(|(i, _, these)| {
                let s = parts[*i].0;
                (these.len() <= STEP_JOBS && these.iter().all(|(_, x)| x.len() == s.hidden)).then(|| s.step.try_lock().ok()).flatten()
            })
            .collect();
        rec.keep_groups(steps.iter().all(Option::is_some));
        let recorded: Vec<(DeviceVec, Vec<usize>)> = subs
            .iter()
            .zip(&mut steps)
            .map(|((i, _, these), step)| {
                let s = parts[*i].0;
                let step = step.as_mut().map(|g| &*g.get_or_insert_with(|| s.step_scratch()));
                s.record(&mut rec, these, limit, step).expect("jobs with rows")
            })
            .collect();
        for (out, _) in &recorded {
            rec.read(out);
        }
        let launched = Some(Box::new(rec).launch());
        let subs = subs.into_iter().zip(recorded).map(|((i, mine, _), (_, rows))| (i, mine, rows, parts[i].0.hidden)).collect();
        let held = steps.into_iter().flatten().map(|g| Box::new(g) as Box<dyn Held + 'a>).collect();
        SlotsBegun { launched, held, subs, jobs }
    }
    /// [`silu_table`]'s for `limit` on this card, uploaded when first asked for (again for another limit).
    fn silu(&self, limit: f32) -> DeviceVec {
        let mut held = self.silu.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, v)) = held.as_ref().filter(|(bits, _)| *bits == limit.to_bits()) {
            return v.clone();
        }
        let table = silu_table(limit);
        let v = self.b.vec(table.len());
        DeviceChain::upload(&self.b, &v, &table);
        *held = Some((limit.to_bits(), v.clone()));
        v
    }

    /// Whether these are on `b`'s device.
    pub(super) fn on(&self, b: &WgpuBackend) -> bool {
        self.b.same_device(b)
    }

    /// The hidden width these experts take and give.
    pub(super) fn hidden(&self) -> usize {
        self.hidden
    }

    /// Their gates' and ups' words' buffer (what knows them on their device).
    pub(super) fn words(&self) -> &wgpu::Buffer {
        &self.gu.words
    }

    /// A fused pass's vectors for `k` jobs ([`Self::record_fused`]), on the device for good (a program's arena's).
    pub(super) fn fused_scratch(&self, k: usize) -> FusedScratch {
        let (h, f) = (self.hidden, self.ff);
        let (sgu, sd) = (self.gu.splits as usize, self.down.splits as usize);
        let v = |len: usize| self.b.vec(len);
        FusedScratch { xh_gu: v(2 * k * h), part_gu: v(2 * k * sgu * f), out_gu: v(2 * k * f), xh_d: v(k * f), part_d: v(k * sd * h) }
    }

    /// `k` one-row jobs of row 0 of `x` (`[hidden]`), their gate and up jobs `jgu` (`(2 slot, 0)` and `(2 slot + 1, 0)`
    /// each) and down jobs `jd` (`(slot, job)`) written on the device (a fused step's candidates, [`super::ProgramStep::
    /// Held`]), recorded into `rec` through `s` into `out` (`[k, hidden]`): each job's output as [`Self::forward`] makes
    /// it (the same kernels, splits and order), a job whose slot is `u32::MAX` skipped (its row left as it was).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_fused(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, jgu: &DeviceVec, jd: &DeviceVec, k: usize, s: &FusedScratch, out: &DeviceVec, limit: f32) {
        let silu = self.silu(limit);
        group_pass_skipping(rec, &self.gu, x, false, jgu, 2 * k, &s.xh_gu, &s.part_gu, &s.out_gu, 0.0, None);
        group_pass_skipping(rec, &self.down, &s.out_gu, true, jd, k, &s.xh_d, &s.part_d, out, limit, Some(&silu));
    }

    /// A step's kept vectors for these slots ([`StepScratch`]).
    fn step_scratch(&self) -> StepScratch {
        let (h, f, n) = (self.hidden, self.ff, STEP_JOBS);
        let (sgu, sd) = (self.gu.splits as usize, self.down.splits as usize);
        let v = |len: usize| self.b.vec(len);
        StepScratch { x: v(n * h), jgu: v(4 * n), jd: v(2 * n), xh_gu: v(2 * n * h), part_gu: v(2 * n * sgu * f), out_gu: v(2 * n * f), xh_d: v(n * f), part_d: v(n * sd * h), out_d: v(n * h) }
    }

    /// `jobs` recorded into `rec` as [`Self::forward`] runs them, through `step`'s vectors where given (jobs of one row
    /// each, no more than it takes): the vector their outputs will be in (`[rows, hidden]` each, in order) and each
    /// job's rows; None where they have no rows.
    fn record(&self, rec: &mut crate::chain::Recorder, jobs: &[(usize, &[f32])], limit: f32, step: Option<&StepScratch>) -> Option<(DeviceVec, Vec<usize>)> {
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
        let silu = self.silu(limit);
        let b = &self.b;
        let up = |v: &DeviceVec, data: &[u32]| DeviceChain::upload(b, v, &data.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>());
        let (sgu, sd) = if one { (self.gu.splits as usize, self.down.splits as usize) } else { (1, 1) };
        let (x, jgu, jd, xh_gu, part_gu, out_gu, xh_d, part_d, out_d) = match step.filter(|_| one && pairs <= STEP_JOBS) {
            // (the outputs' vector as long as these jobs' rows: what is read back)
            Some(s) => (s.x.clone(), s.jgu.clone(), s.jd.clone(), s.xh_gu.clone(), s.part_gu.clone(), s.out_gu.clone(), s.xh_d.clone(), s.part_d.clone(), DeviceVec { len: pairs * h, inner: s.out_d.inner.clone() }),
            None => (
                rec.scratch(pairs * h),
                rec.scratch(4 * pairs),
                rec.scratch(2 * pairs),
                rec.scratch(2 * pairs * h),
                rec.scratch(2 * pairs * sgu * f),
                rec.scratch(2 * pairs * f),
                rec.scratch(pairs * f),
                rec.scratch(pairs * sd * h),
                rec.scratch(pairs * h),
            ),
        };
        DeviceChain::upload(b, &x, &xs);
        up(&jgu, &jobs_gu);
        up(&jd, &jobs_d);
        if one {
            group_pass(rec, &self.gu, &x, false, &jgu, 2 * pairs, Order::Jobs, &xh_gu, &part_gu, &out_gu, 0.0, None);
            group_pass(rec, &self.down, &out_gu, true, &jd, pairs, Order::Jobs, &xh_d, &part_d, &out_d, limit, Some(&silu));
        } else {
            // a prompt's rows: each expert's in blocks, a tile decoded once a block
            let block = many_rows(rec.gpu());
            let (ogu, od) = (many_order(&jobs_gu, block), many_order(&jobs_d, block));
            let (ogv, odv) = (rec.scratch(ogu.len()), rec.scratch(od.len()));
            up(&ogv, &ogu);
            up(&odv, &od);
            group_pass(rec, &self.gu, &x, false, &jgu, 2 * pairs, Order::Many(&ogv, ogu.len() / block, block), &xh_gu, &part_gu, &out_gu, 0.0, None);
            group_pass(rec, &self.down, &out_gu, true, &jd, pairs, Order::Many(&odv, od.len() / block, block), &xh_d, &part_d, &out_d, limit, Some(&silu));
        }
        Some((out_d, rows))
    }
}
