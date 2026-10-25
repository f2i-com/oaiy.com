//! A GGUF's experts in slots on a GPU, for a model that routes and combines its experts itself (DeepSeek-V4.1's GGUF
//! quantizations), as [`crate::exl3::Exl3Slots`] holds an EXL3 checkpoint's: each slot an expert's three matrices, put
//! in from its record (each matrix the GGUF's blocks, [`QuantLayout`]) whenever the model wants it there, its rows laid
//! out as [`QuantMoe`]'s kernels read them ([`Kind::pack_row`]); a call's jobs (a slot and some rows each) through those
//! kernels: gate and up, the model's SwiGLU between ([`SWIGLU_BF16`]), then down. Each job's output is its own (`[rows,
//! hidden]`, unweighted): the model weights and sums them in its own order. No shared expert (the model runs its own).

use super::*;
use crate::exl3::SlotsBegun;

/// Where an expert's record holds one of its matrices: its blocks' first byte, their type, and its shape (`n` rows of
/// `k` weights, a row's blocks after another's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantProj {
    pub at: usize,
    pub dtype: GgmlType,
    pub k: usize,
    pub n: usize,
}

/// An expert's record: its bytes, and where its gate's, up's and down's blocks are in it.
#[derive(Clone, Copy, Debug)]
pub struct QuantLayout {
    pub bytes: usize,
    pub proj: [QuantProj; 3],
}

/// One-row jobs a call's kept vectors take ([`QuantSlots::begin_many`]): a decode step's experts in one set of slots,
/// most of a layer's six. A call of more goes through the recording's scratch.
const STEP_JOBS: usize = 8;

/// Each pair's SwiGLU as DeepSeek-V4.1's routed experts make it, as the down projection reads it: the gate's and up's
/// sums rounded to bf16, the up clamped to `[-limit, limit]` and the gate to `limit` at most (`limit` 0: neither),
/// `silu(gate) * up` rounded to bf16. `p[0]`: ff, pairs, the limit's bits. The pairs' sums as the gate and up pass
/// leaves them: pair `j`'s gate row `2 j`, its up row `2 j + 1`.
const SWIGLU_BF16: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(6) var<storage, read_write> act: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].x;
    if (i >= p[0].y * ff) { return; }
    let pair = i / ff;
    let c = i % ff;
    var g = bf16(y[2u * pair * ff + c]);
    var u = bf16(y[(2u * pair + 1u) * ff + c]);
    let limit = bitcast<f32>(p[0].z);
    if (limit > 0.0) {
        u = clamp(u, -limit, limit);
        g = min(g, limit);
    }
    act[i] = bf16((g / (1.0 + exp(-g))) * u);
}
"#;

/// [`QuantSlots::record`]'s vectors for up to [`STEP_JOBS`] jobs of one row: the inputs, the two job lists, the gate's
/// and up's sums, the SwiGLU's and the outputs.
struct StepScratch {
    x: DeviceVec,
    jgu: DeviceVec,
    jd: DeviceVec,
    out_gu: DeviceVec,
    act: DeviceVec,
    out_d: DeviceVec,
}

/// Experts of one shape and pair of types (their gate's and up's, their down's) in slots on one GPU
/// ([`WgpuBackend::quant_slots`]).
pub struct QuantSlots {
    b: WgpuBackend,
    /// Slot `s`'s gate matrix `2 s` of `gu`, its up `2 s + 1`; its down matrix `s` of `down`.
    gu: Group,
    down: Group,
    slots: usize,
    hidden: usize,
    ff: usize,
    /// Bytes counted against the weight budget while these live.
    nbytes: u64,
    /// Whether the host writes the matrices' words in place ([`WgpuBackend::quant_slots_mapped`]).
    mapped: bool,
    /// A step's vectors (jobs of one row each, up to [`STEP_JOBS`]), made at its first call and kept with their bind
    /// groups for every call after; held by the call that uses them.
    step: Mutex<Option<StepScratch>>,
}

impl std::fmt::Debug for QuantSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantSlots({} slots of {}x{} experts in {} and {})", self.slots, self.hidden, self.ff, self.gu.kind.tag(), self.down.kind.tag())
    }
}

impl Drop for QuantSlots {
    fn drop(&mut self) {
        self.b.used.fetch_sub(self.nbytes, Ordering::Relaxed);
    }
}

/// The words a slot's matrices take, `hidden -> ff` experts of `types` (their gate's and up's, their down's): a gate's
/// (or up's) and a down's; None for a type the kernels do not decode or rows not whole blocks of it.
fn matrix_words(hidden: usize, ff: usize, types: [GgmlType; 2]) -> Option<([Kind; 2], [usize; 2])> {
    let (kg, kd) = (Kind::of(types[0])?, Kind::of(types[1])?);
    (kg.fits(hidden) && kd.fits(ff)).then(|| ([kg, kd], [kg.row_words(hidden) * ff, kd.row_words(ff) * hidden]))
}

impl WgpuBackend {
    /// Up to `slots` slots for `hidden -> ff` experts of `types` (their gate's and up's type, their down's), as many as
    /// a binding holds (a group's matrices are one buffer) and the weight budget has room for; None if not one, or
    /// where the kernels do not decode the types.
    pub fn quant_slots(&self, slots: usize, hidden: usize, ff: usize, types: [GgmlType; 2]) -> Option<QuantSlots> {
        self.quant_slots_with(slots, hidden, ff, types, false)
    }

    /// [`Self::quant_slots`] whose matrices' words the host writes in place ([`QuantSlots::put_many`]): buffers in the
    /// card's memory it maps for writing (through its BAR, where the device has mappable storage buffers; else as
    /// [`Self::quant_slots`]'s), for slots that take records as passes ask for them.
    pub fn quant_slots_mapped(&self, slots: usize, hidden: usize, ff: usize, types: [GgmlType; 2]) -> Option<QuantSlots> {
        let can = self.gpu.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        self.quant_slots_with(slots, hidden, ff, types, can)
    }

    /// The bytes a slot of [`Self::quant_slots`]' takes on a card; None where it makes none.
    pub fn quant_slot_bytes(hidden: usize, ff: usize, types: [GgmlType; 2]) -> Option<u64> {
        matrix_words(hidden, ff, types).map(|(_, [g, d])| ((2 * g + d) * 4) as u64)
    }

    fn quant_slots_with(&self, slots: usize, hidden: usize, ff: usize, types: [GgmlType; 2], mapped: bool) -> Option<QuantSlots> {
        let ([kg, kd], [gw, dw]) = matrix_words(hidden, ff, types)?;
        let per = ((2 * gw + dw) * 4) as u64;
        let limit = chunk_limit(&self.gpu.limits);
        let by_binding = (limit / (2 * gw * 4) as u64).min(limit / (dw * 4) as u64) as usize;
        let free = self.budget.saturating_sub(self.used.load(Ordering::Relaxed));
        let n = slots.min(by_binding).min((free / per) as usize);
        // (a write of the host's lands on a word's edge: a slot's matrices whole words, and two of them for a mapping's
        // offset, a multiple of eight bytes)
        if n == 0 || gw % 2 != 0 || dw % 2 != 0 {
            return None;
        }
        let nbytes = n as u64 * per;
        self.used.fetch_add(nbytes, Ordering::Relaxed);
        let group = |matrices: usize, kind: Kind, k: usize, rows: usize, mwords: usize| -> Group {
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
            let words = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-quant-slots"),
                size: (matrices * mwords * 4) as u64,
                usage: if mapped { usage | wgpu::BufferUsages::MAP_WRITE } else { usage },
                mapped_at_creation: false,
            });
            let table = kind.table().map(|t| {
                let bytes: &[u8] = bytemuck::cast_slice(&t);
                self.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0
            });
            Group { words, kind, k, n: rows, rw: kind.row_words(k), mwords, table, cold: None, mats: matrices / n }
        };
        let gu = group(2 * n, kg, hidden, ff, gw);
        let down = group(n, kd, ff, hidden, dw);
        Some(QuantSlots { b: self.clone(), gu, down, slots: n, hidden, ff, nbytes, mapped, step: Mutex::new(None) })
    }
}

/// The bytes of slots' matrices from byte `from` of a group's (`g`, `mats` matrices a slot, the first of them slot
/// `first`'s) into `part`: each row the record's blocks as the kernels read them ([`Kind::pack_row`]), from `records[j]`
/// (slot `first + j`'s) at `projs[m]` (its matrix `m` of the slot's).
fn pack_part(g: &Group, mats: usize, projs: &[QuantProj], records: &[&[u8]], from: usize, part: &mut [u8]) {
    let (row_bytes, mbytes) = (g.rw * 4, g.mwords * 4);
    let mut row = vec![0u32; g.rw];
    let mut at = 0;
    while at < part.len() {
        let byte = from + at;
        let (matrix, o) = (byte / mbytes, byte % mbytes);
        let (j, m) = (matrix / mats, matrix % mats);
        let (r, ro) = (o / row_bytes, o % row_bytes);
        let p = &projs[m];
        let src = row_bytes_of(p);
        g.kind.pack_row(&records[j][p.at + r * src..p.at + (r + 1) * src], g.k, &mut row);
        let take = (row_bytes - ro).min(part.len() - at);
        let bytes: &[u8] = bytemuck::cast_slice(&row);
        part[at..at + take].copy_from_slice(&bytes[ro..ro + take]);
        at += take;
    }
}

/// What a record's put takes: its rows packed on the host's threads into memory of its own, and the same bytes written
/// into a card's mapped memory with nothing to pack, at DeepSeek-V4.1's shape.
#[cfg(test)]
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_a_records_packing_and_write() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    let (hidden, ff) = (5120usize, 2304usize);
    for types in [[GgmlType::IQ1_M, GgmlType::IQ1_M], [GgmlType::IQ2_XXS, GgmlType::IQ2_XS]] {
        let gb = ff * hidden / 256 * types[0].type_size();
        let db = hidden * ff / 256 * types[1].type_size();
        let layout = QuantLayout {
            bytes: 2 * gb + db,
            proj: [QuantProj { at: 0, dtype: types[0], k: hidden, n: ff }, QuantProj { at: gb, dtype: types[0], k: hidden, n: ff }, QuantProj { at: 2 * gb, dtype: types[1], k: ff, n: hidden }],
        };
        let record: Vec<u8> = (0..layout.bytes).map(|i| (i * 131 % 251) as u8).collect();
        let slots = b.quant_slots_mapped(8, hidden, ff, types).expect("room for eight slots");
        let gw = slots.gu.mwords * 4;
        let mut packed = vec![0u8; 2 * gw];
        let reps = 20;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            pack_part(&slots.gu, 2, &[layout.proj[0], layout.proj[1]], &[&record[..]], 0, &mut packed);
        }
        let one = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            packed.par_chunks_mut(gw / 8).enumerate().for_each(|(i, part)| pack_part(&slots.gu, 2, &[layout.proj[0], layout.proj[1]], &[&record[..]], i * (gw / 8), part));
        }
        let many = t.elapsed().as_secs_f64() / reps as f64;
        let copy = |from: usize, part: &mut [u8]| part.copy_from_slice(&packed[from..from + part.len()]);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            b.gpu.write_mapped(&[(&slots.gu.words, 0, 2 * gw, &copy as &(dyn Fn(usize, &mut [u8]) + Sync))]);
        }
        let write = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            slots.put(0, &record, &layout);
        }
        let put = t.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "{types:?}: a record's gate and up ({:.1} MB) packed in {:.2} ms on one thread, {:.2} ms on eight; written mapped with nothing to pack {:.2} ms ({:.1} GB/s); a whole put {:.2} ms",
            (2 * gw) as f64 / 1e6,
            one * 1e3,
            many * 1e3,
            write * 1e3,
            (2 * gw) as f64 / write / 1e9,
            put * 1e3
        );
    }
}

/// A matrix's row's bytes in its record.
fn row_bytes_of(p: &QuantProj) -> usize {
    row_bytes(p.dtype, p.k)
}

/// One group's jobs (`jobs`, `count` of them) on `x`'s rows into `y`'s (job `j`'s row `j`), as [`QuantMoe`]'s pass
/// runs a layer the card holds whole; `skip`: a job whose matrix is `u32::MAX` skipped, its row left as it was.
#[allow(clippy::too_many_arguments)]
pub(super) fn pass(rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, jobs: &DeviceVec, count: usize, order: Order<'_>, y: &DeviceVec, slots: usize, skip: bool) {
    let d = rec.gpu().dummy().clone();
    let drw = rec.gpu().dummy_rw().clone();
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
    let (xb, jb, yb) = (buf(x), buf(jobs), buf(y));
    let ntiles = (g.n / 16) as u32;
    let (ob, blocks, rows, identity) = match order {
        Order::Jobs => (d.clone(), count, 1, 1),
        Order::Blocks(o, blocks, rows) => (buf(o), blocks, rows, 0),
        Order::At(..) => unreachable!("a slots' pass orders its blocks from the first"),
    };
    let coop = rows > FEW_MAX;
    let lanes = few_lanes(g.kind, g.k);
    let (name, source) = kernel(g.kind, coop, rows, g.k, 0, false, lanes, skip);
    for first in (0..blocks).step_by(65535) {
        let these = 65535.min(blocks - first) as u32;
        let words = [g.n as u32, g.k as u32, g.rw as u32, identity, g.mwords as u32, first as u32, 0, slots as u32];
        let groups = (g.n as u32).div_ceil((256 / lanes) as u32);
        let grid = if coop { (ntiles.div_ceil(8), 1, these) } else { (groups.min(65535), groups.div_ceil(65535), these) };
        rec.dispatch_wide(name, source, [&g.words, &xb, &jb, &ob, g.table.as_ref().unwrap_or(&d), &d, &yb, &drw], &words, grid);
    }
    rec.weigh(2.0 * count as f64 * (g.n * g.k) as f64);
}

impl QuantSlots {
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The bytes these take on the GPU.
    pub fn nbytes(&self) -> u64 {
        self.nbytes
    }

    /// Whether an expert laid out as `layout` lays it out goes into these slots (its shapes and types theirs).
    pub fn fits(&self, layout: &QuantLayout) -> bool {
        let [g, u, d] = &layout.proj;
        let kind = |p: &QuantProj| Kind::of(p.dtype);
        (g.k, g.n, u.k, u.n, d.k, d.n) == (self.hidden, self.ff, self.hidden, self.ff, self.ff, self.hidden)
            && kind(g) == Some(self.gu.kind)
            && kind(u) == Some(self.gu.kind)
            && kind(d) == Some(self.down.kind)
            && layout.proj.iter().all(|p| p.at + p.n * row_bytes_of(p) <= layout.bytes)
    }

    /// Slot `slot`'s expert made `record`'s (as `layout` lays it out, which [`Self::fits`]): its matrices' rows written
    /// where the groups keep that slot's, there for every call after.
    pub fn put(&self, slot: usize, record: &[u8], layout: &QuantLayout) {
        self.put_many(slot, &[record], layout);
    }

    /// Slots `first..` made `records`' experts (each as `layout` lays it out). Mapped slots ([`WgpuBackend::
    /// quant_slots_mapped`]) take the gate's and up's rows of them all and the down's written in place by the host's
    /// threads, at once; others, the rows packed here and written by the queue. Nothing may be running on these slots.
    pub fn put_many(&self, first: usize, records: &[&[u8]], layout: &QuantLayout) {
        let n = records.len();
        assert!(first + n <= self.slots && self.fits(layout), "quant slots: slots {first}..{} of {}, or an expert of another shape", first + n, self.slots);
        assert!(records.iter().all(|r| r.len() >= layout.bytes), "quant slots: a record shorter than its layout");
        if n == 0 {
            return;
        }
        let [pg, pu, pd] = layout.proj;
        let (gw, dw) = (self.gu.mwords * 4, self.down.mwords * 4);
        let gu_fill = |from: usize, part: &mut [u8]| pack_part(&self.gu, 2, &[pg, pu], records, from, part);
        let d_fill = |from: usize, part: &mut [u8]| pack_part(&self.down, 1, &[pd], records, from, part);
        let gpu = &self.b.gpu;
        if self.mapped {
            gpu.write_mapped(&[
                (&self.gu.words, (2 * first * gw) as u64, 2 * n * gw, &gu_fill as &(dyn Fn(usize, &mut [u8]) + Sync)),
                (&self.down.words, (first * dw) as u64, n * dw, &d_fill),
            ]);
            return;
        }
        let mut gu = vec![0u8; 2 * n * gw];
        gu.par_chunks_mut(gw).enumerate().for_each(|(i, part)| gu_fill(i * gw, part));
        let mut down = vec![0u8; n * dw];
        down.par_chunks_mut(dw).enumerate().for_each(|(i, part)| d_fill(i * dw, part));
        gpu.write(&self.gu.words, (2 * first * gw) as u64, &gu);
        gpu.write(&self.down.words, (first * dw) as u64, &down);
        // (the writes sent, and their staging let go)
        gpu.queue().submit([]);
        gpu.wait(None);
    }

    /// `to`'s slot `at` made what slot `from` of these holds (both sets on one GPU, of one shape and types): its
    /// matrices' words copied by the GPU, in its turn after the work submitted before it, and `marks` (`(table, value's
    /// place, value)` each) written in the same submission, as [`crate::exl3::Exl3Slots::copy_slot_marked`] copies.
    pub fn copy_slot_marked(&self, from: usize, to: &QuantSlots, at: usize, marks: &[(&DeviceVec, usize, u32)]) {
        assert!(Arc::ptr_eq(&self.b.gpu, &to.b.gpu) && from < self.slots && at < to.slots, "quant slots: a copy between GPUs, or past the slots");
        assert!((self.hidden, self.ff, self.gu.kind, self.down.kind) == (to.hidden, to.ff, to.gu.kind, to.down.kind), "quant slots: a copy between experts of other shapes");
        let gpu = &self.b.gpu;
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector").clone();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("oaiy-quant-slot-copy") });
        for (a, b, per) in [(&self.gu, &to.gu, 2usize), (&self.down, &to.down, 1)] {
            let words = (per * a.mwords * 4) as u64;
            enc.copy_buffer_to_buffer(&a.words, from as u64 * words, &b.words, at as u64 * words, words);
        }
        let _one = self.b.serial.lock().unwrap_or_else(|p| p.into_inner());
        for (table, place, value) in marks {
            assert!(*place < table.len, "quant slots: a mark past its table");
            gpu.queue().write_buffer(&buf(table), (*place * 4) as u64, &value.to_le_bytes());
        }
        gpu.queue().submit([enc.finish()]);
    }

    /// `jobs`' experts (each a slot and its rows, `[rows, hidden]`), in one recording: gate and up, the SwiGLU clamped
    /// at `limit` (0: none), then down. Each job's output, `[rows, hidden]`, unweighted. A job of one row runs as a
    /// decode step's (a workgroup some rows of the matrix), more rows as a prompt's (each expert's rows in blocks, a
    /// row's weights decoded once a block).
    pub fn forward(&self, jobs: &[(usize, &[f32])], limit: f32) -> Vec<Vec<f32>> {
        Self::forward_many(&[(self, jobs)], limit).pop().expect("one set of slots, one set of outputs")
    }

    /// [`Self::forward`] of several sets of slots on one GPU, in one recording and one read back.
    pub fn forward_many(parts: &[(&QuantSlots, &[(usize, &[f32])])], limit: f32) -> Vec<Vec<Vec<f32>>> {
        Self::begin_many(parts, limit).finish()
    }

    /// [`Self::forward_many`] begun: recorded and submitted, its outputs left for [`SlotsBegun::finish`]. A job of one
    /// row is recorded as a decode step's and a job of more as a prompt's whatever it is called with (a set's jobs of
    /// each kind recorded apart), so its output is the same bits whichever jobs share its call.
    pub fn begin_many<'a>(parts: &[(&'a QuantSlots, &[(usize, &[f32])])], limit: f32) -> SlotsBegun<'a> {
        let jobs: Vec<usize> = parts.iter().map(|(_, jobs)| jobs.len()).collect();
        let Some(&(first, _)) = parts.first() else { return SlotsBegun { launched: None, held: Vec::new(), subs: Vec::new(), jobs } };
        assert!(parts.iter().all(|(s, _)| Arc::ptr_eq(&s.b.gpu, &first.b.gpu)), "quant slots: sets on more than one GPU in one call");
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
        // (a step's jobs through each set's kept vectors, their bind groups kept with them, where every part has its
        // own: as Exl3Slots' calls)
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
        let held = steps.into_iter().flatten().map(|g| Box::new(g) as Box<dyn crate::exl3::Held + 'a>).collect();
        SlotsBegun { launched, held, subs, jobs }
    }

    /// A step's kept vectors for these slots ([`StepScratch`]).
    fn step_scratch(&self) -> StepScratch {
        let (h, f, n) = (self.hidden, self.ff, STEP_JOBS);
        let v = |len: usize| self.b.vec(len);
        StepScratch { x: v(n * h), jgu: v(4 * n), jd: v(2 * n), out_gu: v(2 * n * f), act: v(n * f), out_d: v(n * h) }
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
        assert!(jobs.iter().all(|&(s, x)| s < self.slots && x.len() % h == 0), "quant slots: a job's slot or rows");
        // the jobs: gate and up of each row on its row of x, then down of each on its SwiGLU's row
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
        let (x, jgu, jd, out_gu, act, out_d) = match step.filter(|_| one && pairs <= STEP_JOBS) {
            // (the outputs' vector as long as these jobs' rows: what is read back)
            Some(s) => (s.x.clone(), s.jgu.clone(), s.jd.clone(), s.out_gu.clone(), s.act.clone(), DeviceVec { len: pairs * h, inner: s.out_d.inner.clone() }),
            None => (rec.scratch(pairs * h), rec.scratch(4 * pairs), rec.scratch(2 * pairs), rec.scratch(2 * pairs * f), rec.scratch(pairs * f), rec.scratch(pairs * h)),
        };
        DeviceChain::upload(b, &x, &xs);
        up(&jgu, &jobs_gu);
        up(&jd, &jobs_d);
        let swiglu = |rec: &mut crate::chain::Recorder| swiglu_pairs(rec, &out_gu, &act, f, pairs, limit);
        if one {
            pass(rec, &self.gu, &x, &jgu, 2 * pairs, Order::Jobs, &out_gu, self.slots, false);
            swiglu(rec);
            pass(rec, &self.down, &act, &jd, pairs, Order::Jobs, &out_d, self.slots, false);
        } else {
            // a prompt's rows: each expert's in blocks, its weights decoded once a block (16 rows on the tensor cores)
            let block = if coop_on(&b.gpu) && std::env::var_os("OAIY_QUANT_MOE_NO_COOP").is_none() { 16 } else { FEW_MAX };
            let (ogu, od) = (many_order(&jobs_gu, block), many_order(&jobs_d, block));
            let (ogv, odv) = (rec.scratch(ogu.len()), rec.scratch(od.len()));
            up(&ogv, &ogu);
            up(&odv, &od);
            pass(rec, &self.gu, &x, &jgu, 2 * pairs, Order::Blocks(&ogv, ogu.len() / block, block), &out_gu, self.slots, false);
            swiglu(rec);
            pass(rec, &self.down, &act, &jd, pairs, Order::Blocks(&odv, od.len() / block, block), &out_d, self.slots, false);
        }
        Some((out_d, rows))
    }

    /// Whether these are on `b`'s device.
    pub(crate) fn on(&self, b: &WgpuBackend) -> bool {
        self.b.same_device(b)
    }

    /// The hidden width these experts take and give.
    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    /// Their gates' and ups' words' buffer (what knows them on their device).
    pub(crate) fn words(&self) -> &wgpu::Buffer {
        &self.gu.words
    }

    /// A fused pass's vectors for `k` jobs ([`Self::record_fused`]), on the device for good (a program's arena's).
    pub(crate) fn fused_scratch(&self, k: usize) -> QuantFused {
        QuantFused { out_gu: self.b.vec(2 * k * self.ff), act: self.b.vec(k * self.ff) }
    }

    /// `k` one-row jobs of row 0 of `x` (`[hidden]`), their gate and up jobs `jgu` (`(2 slot, 0)` and `(2 slot + 1, 0)`
    /// each) and down jobs `jd` (`(slot, job)`) written on the device (a fused step's candidates, `exl3::ProgramStep::
    /// Held`), recorded into `rec` through `s` into `out` (`[k, hidden]`): each job's output as [`Self::forward`] makes
    /// it (the same kernels and order), a job whose slot is `u32::MAX` skipped (its row left as it was).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_fused(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, jgu: &DeviceVec, jd: &DeviceVec, k: usize, s: &QuantFused, out: &DeviceVec, limit: f32) {
        pass(rec, &self.gu, x, jgu, 2 * k, Order::Jobs, &s.out_gu, self.slots, true);
        swiglu_pairs(rec, &s.out_gu, &s.act, self.ff, k, limit);
        pass(rec, &self.down, &s.act, jd, k, Order::Jobs, out, self.slots, true);
    }
}

/// [`QuantSlots::record_fused`]'s vectors beside its inputs and output: the gate's and up's sums, the SwiGLU's.
pub(crate) struct QuantFused {
    out_gu: DeviceVec,
    act: DeviceVec,
}

/// [`SWIGLU_BF16`] of `pairs` pairs of `ff` (`gu`, as the gate and up pass leaves them) into `act`.
fn swiglu_pairs(rec: &mut crate::chain::Recorder<'_>, gu: &DeviceVec, act: &DeviceVec, ff: usize, pairs: usize, limit: f32) {
    let (d, drw) = (rec.gpu().dummy().clone(), rec.gpu().dummy_rw().clone());
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
    let groups = ((pairs * ff) as u32).div_ceil(256);
    rec.dispatch_wide("quant-slots-swiglu-bf16", SWIGLU_BF16, [&buf(gu), &d, &d, &d, &d, &d, &buf(act), &drw], &[ff as u32, pairs as u32, limit.to_bits()], (groups.min(65535), groups.div_ceil(65535).max(1), 1));
}
