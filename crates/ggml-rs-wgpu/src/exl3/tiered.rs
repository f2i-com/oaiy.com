//! A MoE layer's EXL3 experts where neither the GPU nor the host's memory holds them all (Qwen3.8-Flash-Next's 46.7 GB
//! of routed experts beside a 24 GB card on a 24 GB Mac): each routed expert is one record, its three projections as
//! the checkpoint lays them out side by side (1.86 MB), read when a row is routed to it (from the drive, or the page
//! cache where it was read before) and computed on the host's cores, its packed words as they lie.
//!
//! Where the GPU has room, it holds some of them ([`Tier`]): slots of grouped experts (`Exl3MoeGrouped`'s groups,
//! a slot an expert's place in them), each expert's slot in a map on the device. A step's and a check's rows are routed
//! there as every layer's are, each expert sent to its slot; one it does not hold is weighted 0 there, and the host,
//! which reads the layer's router and input back, adds the rest ([`Exl3MoeTiered::forward_rest`]). A prompt's rows are
//! routed on the host, which first takes the experts the chunk uses into slots of experts it does not
//! ([`Exl3MoeTiered::plan_rows`]). The slots are kept by recency: an expert a step or a check misses is taken in on its
//! second miss (on its first while a slot is free), a few a layer a step at most, and a prompt's experts as its chunk
//! asks for them. The shared expert is on the GPU, where the chain runs it beside the layer's experts.

use super::*;
use rayon::prelude::*;

/// Where one projection lies in an expert's record (byte offsets of its `suh` and `svh`, f16, its codebook's
/// multiplier, an i32 that must be mul1's, and its trellis), and its shape (`k -> n`; the identity maps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjAt {
    pub suh: usize,
    pub svh: usize,
    pub mul1: usize,
    pub trellis: usize,
    pub k: usize,
    pub n: usize,
    pub tile_words: usize,
}

impl ProjAt {
    fn trellis_bytes(&self) -> usize {
        self.k / 16 * (self.n / 16) * self.tile_words * 2
    }
}

/// An expert's record: its bytes, and where its gate, up and down projections lie in them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordLayout {
    pub bytes: usize,
    pub proj: [ProjAt; 3],
}

impl RecordLayout {
    /// Whether it is a layer's of `hidden -> ff` experts, each part within the record.
    pub fn check(&self, hidden: usize, ff: usize) -> Result<(), String> {
        for (i, (p, (k, n))) in self.proj.iter().zip([(hidden, ff), (hidden, ff), (ff, hidden)]).enumerate() {
            let fits = |at: usize, len: usize| at.checked_add(len).is_some_and(|end| end <= self.bytes);
            if (p.k, p.n) != (k, n) || k % 128 != 0 || n % 128 != 0 || !matches!(p.tile_words, 16 | 24 | 32 | 40 | 48 | 56 | 64 | 80 | 96 | 112 | 128) {
                return Err(format!("an expert's projection {i} is {}x{} at {} words a tile, where {k}x{n}", p.k, p.n, p.tile_words));
            }
            if !fits(p.suh, 2 * k) || !fits(p.svh, 2 * n) || !fits(p.mul1, 4) || !fits(p.trellis, p.trellis_bytes()) {
                return Err(format!("an expert's projection {i} lies past its record's {} bytes", self.bytes));
            }
        }
        Ok(())
    }
}

/// The host's source of a layer's routed experts' records.
pub trait ExpertSource: Send + Sync + std::fmt::Debug {
    /// The experts it has records of.
    fn experts(&self) -> usize;
    /// Expert `e`'s record into `into` (resized to the record's bytes).
    fn read(&self, e: usize, into: &mut Vec<u8>) -> std::io::Result<()>;
    /// The bytes read so far.
    fn bytes_read(&self) -> u64 {
        0
    }
}

/// Records in files: expert `e`'s `bytes` at `at[e]` (a file of `files`, and an offset), read through the page cache
/// (which keeps what it can of them in the host's memory).
#[derive(Debug)]
pub struct FileRecords {
    files: Vec<Arc<std::fs::File>>,
    at: Vec<(u32, u64)>,
    bytes: usize,
    read: AtomicU64,
}

impl FileRecords {
    pub fn new(files: Vec<Arc<std::fs::File>>, at: Vec<(u32, u64)>, bytes: usize) -> Result<Self, String> {
        if at.iter().any(|&(f, _)| f as usize >= files.len()) {
            return Err("an expert's record is in a file not given".into());
        }
        Ok(Self { files, at, bytes, read: AtomicU64::new(0) })
    }
}

impl ExpertSource for FileRecords {
    fn experts(&self) -> usize {
        self.at.len()
    }

    fn read(&self, e: usize, into: &mut Vec<u8>) -> std::io::Result<()> {
        let (f, offset) = self.at[e];
        into.resize(self.bytes, 0);
        read_exact_at(&self.files[f as usize], into, offset)?;
        self.read.fetch_add(self.bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    fn bytes_read(&self) -> u64 {
        self.read.load(Ordering::Relaxed)
    }
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = std::os::windows::fs::FileExt::seek_read(file, buf, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// mul1's multiplier, as a checkpoint stores it (an i32's bytes).
const MUL1: [u8; 4] = 0x83dc_d12du32.to_le_bytes();

/// A record's projection's transforms: its scales as f32, the maps the identity.
fn transform(record: &[u8], p: &ProjAt) -> Transform {
    assert_eq!(record[p.mul1..p.mul1 + 4], MUL1, "an expert's projection of a codebook other than mul1");
    let halves = |at: usize, len: usize| (0..len).map(|i| half::f16::from_bits(u16::from_le_bytes([record[at + 2 * i], record[at + 2 * i + 1]])).to_f32()).collect();
    Transform { k: p.k, n: p.n, suh: halves(p.suh, p.k), svh: halves(p.svh, p.n), input_map: (0..p.k as u32).collect(), output_map: (0..p.n as u32).collect() }
}

/// Tiles a task of the host's experts decodes: a projection's tile columns in runs of about this many tiles (a gate's
/// or an up's one column of 160, a down's four of 40), so even one expert's projections are tasks enough for every
/// core (some 40 us each).
const TASK_TILES: usize = 320;

/// A MoE layer's experts, the routed ones read as records from `source` and computed on the host, the shared one on
/// the GPU (see the module's notes).
pub struct Exl3MoeTiered {
    b: WgpuBackend,
    routed: usize,
    hidden: usize,
    ff: usize,
    layout: RecordLayout,
    source: Arc<dyn ExpertSource>,
    /// The shared expert's gate, up and down projections (on the GPU while the budget holds them).
    shared: [Arc<dyn PackedLinear>; 3],
    /// Whether the chain can run the shared expert (its projections each one buffer on the GPU).
    shared_chained: bool,
    /// A step's and a check's shared-expert scratch (gate and up), a row count's each.
    kept: Mutex<Vec<(usize, Arc<[DeviceVec; 2]>)>>,
    /// An adapter's low-rank updates beside the routed experts' gate, up and down projections (an expert a matrix).
    lora: [LoraTables; 3],
    /// The experts the GPU holds, where it has room for some.
    tier: Option<Tier>,
    /// Records being read ahead for the next run (the experts the layer before's input predicts it will route to that
    /// the GPU does not hold), each set once read.
    ahead: Mutex<Vec<(usize, Arc<std::sync::OnceLock<Arc<Vec<u8>>>>)>>,
}

impl std::fmt::Debug for Exl3MoeTiered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slots = self.tier.as_ref().map_or(0, |t| t.grouped.slots());
        write!(f, "Exl3MoeTiered({} routed experts of {}x{} read as records of {} bytes, {slots} of them held on the GPU, the shared one {})", self.routed, self.hidden, self.ff, self.layout.bytes, if self.shared_chained { "on the GPU" } else { "where the budget put it" })
    }
}

/// The GPU's slots of a layer's experts: grouped as a layer's experts are, each expert's slot on the device (u32s,
/// [`NONE`] for none) and what the host keeps of them.
struct Tier {
    grouped: Exl3MoeGrouped,
    map: DeviceVec,
    state: Mutex<TierState>,
}

/// What the host knows of a layer's slots: each expert's slot ([`NONE`]: none) and each slot's expert, the run each
/// slot's expert was last used in, the run each expert not held was last missed in (0: not lately), and counts.
#[derive(Default)]
struct TierState {
    slot_of: Vec<u32>,
    owner: Vec<u32>,
    last: Vec<u64>,
    missed: Vec<u64>,
    /// The runs so far (a step's, a check's, a prompt chunk's: each one).
    runs: u64,
    hits: u64,
    misses: u64,
    taken: u64,
}

/// Over every layer's slots since the process began: routings to experts held, and not; experts taken in; records
/// read for the host, with their bytes; and the microseconds the host spent reading them, running them, and taking
/// them in. What a request's cost of them was is the difference over it.
static COUNTS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// [`COUNTS`]: routings held and not, experts taken in, records read and their bytes, the host's microseconds
/// reading, running and taking in, and records read ahead that a run used and that none did.
pub fn tiered_experts() -> [u64; 10] {
    COUNTS.each_ref().map(|c| c.load(Ordering::Relaxed))
}

/// The microseconds since `t`, counted at `i`.
fn timed(i: usize, t: std::time::Instant) {
    counted(i, t.elapsed().as_micros() as u64);
}

fn counted(i: usize, n: u64) {
    COUNTS[i].fetch_add(n, Ordering::Relaxed);
}

/// The threads records are read ahead on (the drive's reads side by side, apart from the cores' work).
fn io_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(6).thread_name(|i| format!("oaiy-expert-read-{i}")).build().expect("the expert readers' threads"))
}

/// Experts a step or a check takes into slots at most a layer: each an upload of a record (1.86 MB, some 0.8 ms of a
/// Thunderbolt 3 link), where a miss costs the host's cores some 0.3 ms (`OAIY_FN_TAKE`).
fn take_per_run() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("OAIY_FN_TAKE").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// Runs within which an expert's second miss takes it in (`OAIY_FN_TAKE_WINDOW`; 0: its first miss does).
fn take_window() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("OAIY_FN_TAKE_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(256))
}

impl Tier {
    /// Expert `e` taken into a slot (`record` its record), if one can be had: a free one, else the least recently used
    /// whose expert this run does not use (`busy`). Its slot and the map written (in the queue's order).
    fn take(&self, st: &mut TierState, e: usize, record: &[u8], layout: &RecordLayout, busy: &dyn Fn(usize) -> bool) -> bool {
        let now = st.runs;
        let free = st.owner.iter().position(|&o| o == NONE);
        let slot = free.or_else(|| (0..st.owner.len()).filter(|&s| st.last[s] < now && !busy(st.owner[s] as usize)).min_by_key(|&s| st.last[s]));
        let Some(slot) = slot else { return false };
        let gpu = &self.grouped.b.gpu;
        let map = self.map.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector");
        let old = st.owner[slot];
        if old != NONE {
            st.slot_of[old as usize] = NONE;
            gpu.write(map, old as u64 * 4, &NONE.to_le_bytes());
        }
        self.grouped.put(slot, record, layout);
        st.owner[slot] = e as u32;
        st.slot_of[e] = slot as u32;
        st.last[slot] = now;
        st.missed[e] = 0;
        st.taken += 1;
        counted(2, 1);
        gpu.write(map, e as u64 * 4, &(slot as u32).to_le_bytes());
        true
    }
}

impl WgpuBackend {
    /// A MoE layer's routed experts read as `layout`'s records from `source` and computed on the host, and its
    /// `shared` expert (gate, up, down) on this GPU while the budget holds it.
    ///
    /// `slots` of them held on this GPU (taken in as they are used), where the budget less `reserve` has room for
    /// them: the experts then routed on the GPU, each to its slot, the host adding the rest.
    pub fn exl3_experts_tiered(&self, layout: RecordLayout, source: Arc<dyn ExpertSource>, shared: [Exl3Data; 3], slots: usize, reserve: u64) -> Result<Box<dyn ggml_rs::exl3::Experts>, String> {
        let routed = source.experts();
        let [g, u, d] = shared;
        let (hidden, ff) = (g.suh.len(), g.svh.len());
        layout.check(hidden, ff)?;
        if routed == 0 || u.suh.len() != hidden || u.svh.len() != ff || d.suh.len() != ff || d.svh.len() != hidden {
            return Err(format!("a layer of {routed} experts of {hidden}x{ff}, the shared one of other shapes"));
        }
        let shared = [g, u, d];
        let grouped = if slots > 0 && routed <= 1024 { Exl3MoeGrouped::with_slots(self, slots.min(routed), hidden, ff, [layout.proj[0].tile_words, layout.proj[2].tile_words], &shared, reserve)? } else { None };
        let tier = grouped.map(|grouped| {
            let n = grouped.slots();
            let map = self.vec(routed);
            upload_u32(self, &map, &vec![NONE; routed]);
            let state = TierState { slot_of: vec![NONE; routed], owner: vec![NONE; n], last: vec![0; n], missed: vec![0; routed], ..Default::default() };
            Tier { grouped, map, state: Mutex::new(state) }
        });
        let shared: [Arc<dyn PackedLinear>; 3] = match &tier {
            Some(t) => t.grouped.shared.clone(),
            None => {
                let [g, u, d] = shared;
                [self.exl3(g)?, self.exl3(u)?, self.exl3(d)?]
            }
        };
        let shared_chained = shared.iter().all(|s| s.as_any().and_then(|a| a.downcast_ref::<Exl3Gpu>()).is_some_and(|g| g.is_on(&self.gpu) && g.single_chunk().is_some()));
        Ok(Box::new(Exl3MoeTiered { b: self.clone(), routed, hidden, ff, layout, source, shared, shared_chained, kept: Mutex::new(Vec::new()), lora: Default::default(), tier, ahead: Mutex::new(Vec::new()) }))
    }
}

impl Exl3MoeTiered {
    /// The shared expert recorded on `x`'s `rows` into `out` (unweighted), where the chain can run it on `rec`'s GPU.
    pub(crate) fn record_shared(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, rows: usize) -> bool {
        use ggml_rs::ChainRecorder;
        if !self.shared_chained || !std::ptr::eq(Arc::as_ptr(&self.b.gpu), rec.gpu()) {
            return false;
        }
        let (h, f) = (self.hidden, self.ff);
        assert!(rows > 0 && x.len >= rows * h && out.len >= rows * h, "a shared expert on {rows} rows of {h}");
        // a step's row, a check's few: the kept scratch (its bind groups kept); a prompt's: the recording's
        let st = if rows <= 8 && rec.keeps() {
            let mut kept = self.kept.lock().unwrap_or_else(|p| p.into_inner());
            match kept.iter().find(|(r, _)| *r == rows) {
                Some((_, st)) => Arc::clone(st),
                None => {
                    let st = Arc::new([self.b.vec(rows * f), self.b.vec(rows * f)]);
                    kept.push((rows, Arc::clone(&st)));
                    st
                }
            }
        } else {
            Arc::new([rec.scratch(rows * f), rec.scratch(rows * f)])
        };
        rec.exl3_rows(&*self.shared[0], x, &st[0], rows);
        rec.exl3_rows(&*self.shared[1], x, &st[1], rows);
        rec.exl3_rows_swiglu(&*self.shared[2], &st[0], &st[1], out, rows);
        true
    }

    /// The shared expert's outputs for `x`'s rows (unweighted), each projection where it is.
    fn shared_rows(&self, x: &Tensor) -> Vec<f32> {
        let g = self.shared[0].linear(x).to_host();
        let u = self.shared[1].linear(x).to_host();
        let hidden: Vec<f32> = g.data().iter().zip(u.data()).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
        let rows = x.numel() / self.hidden;
        self.shared[2].linear(&Tensor::from_vec(hidden, vec![rows, self.ff])).to_host().data().to_vec()
    }

    /// `used`'s records, read side by side.
    fn records(&self, used: &[usize]) -> Vec<Arc<Vec<u8>>> {
        let t = std::time::Instant::now();
        // (those read ahead taken as they are, waited for where still being read)
        let ahead = std::mem::take(&mut *self.ahead.lock().unwrap_or_else(|p| p.into_inner()));
        let mut got: Vec<Option<Arc<Vec<u8>>>> = used.iter().map(|&e| ahead.iter().find(|(a, _)| *a == e).map(|(_, r)| Arc::clone(r.wait()))).collect();
        let found = got.iter().filter(|r| r.is_some()).count();
        counted(8, found as u64);
        counted(9, (ahead.len() - found) as u64);
        let rest: Vec<usize> = used.iter().zip(&got).filter(|(_, r)| r.is_none()).map(|(&e, _)| e).collect();
        counted(3, rest.len() as u64);
        counted(4, (rest.len() * self.layout.bytes) as u64);
        let mut read = rest
            .par_iter()
            .map(|&e| {
                let mut v = Vec::new();
                self.source.read(e, &mut v).unwrap_or_else(|err| panic!("expert {e}'s record could not be read: {err}"));
                Arc::new(v)
            })
            .collect::<Vec<_>>()
            .into_iter();
        for r in got.iter_mut().filter(|r| r.is_none()) {
            *r = read.next();
        }
        timed(5, t);
        got.into_iter().map(|r| r.expect("every record read")).collect()
    }

    /// The next run's records read ahead, on threads of their own: the experts the GPU does not hold among each row's
    /// top `top_k` + 2 by `logits` (the next layer's router on this one's input, which predicts most of what it routes
    /// to). What was read ahead and not yet used is let go.
    fn read_ahead(&self, logits: &Tensor, top_k: usize) {
        let Some(tier) = &self.tier else { return };
        let logits = logits.to_host();
        let width = self.routed + 1;
        let rows = logits.numel() / width;
        let mut want: Vec<usize> = Vec::new();
        {
            let st = tier.state.lock().unwrap_or_else(|p| p.into_inner());
            for r in 0..rows {
                for (e, _) in route(&logits.data()[r * width..(r + 1) * width], (top_k + 2).min(self.routed)).into_iter().take(top_k + 2) {
                    if e < self.routed && st.slot_of[e] == NONE && !want.contains(&e) {
                        want.push(e);
                    }
                }
            }
        }
        let jobs: Vec<(usize, Arc<std::sync::OnceLock<Arc<Vec<u8>>>>)> = want.iter().map(|&e| (e, Arc::new(std::sync::OnceLock::new()))).collect();
        for (e, cell) in &jobs {
            let (source, e, cell) = (Arc::clone(&self.source), *e, Arc::clone(cell));
            io_pool().spawn(move || {
                let mut v = Vec::new();
                // (a read that fails leaves an empty record: the run reads it again itself)
                let ok = source.read(e, &mut v).is_ok();
                let _ = cell.set(Arc::new(if ok { v } else { Vec::new() }));
            });
        }
        counted(3, jobs.len() as u64);
        counted(4, (jobs.len() * self.layout.bytes) as u64);
        *self.ahead.lock().unwrap_or_else(|p| p.into_inner()) = jobs;
    }

    /// Each of `used`'s experts' outputs (`[its rows, hidden]`) for its rows of `x` (`by[i]`, in row order), from its
    /// record (`records[i]`): its gate and up projections, their SwiGLU and its down projection, each projection's
    /// columns in runs that are tasks of their own on the host's cores.
    fn run_host(&self, x: &[f32], used: &[usize], by: &[Vec<usize>], records: &[Arc<Vec<u8>>]) -> Vec<Vec<f32>> {
        let (h, f) = (self.hidden, self.ff);
        let ts: Vec<[Transform; 3]> = records.par_iter().map(|r| self.layout.proj.map(|p| transform(r, &p))).collect();
        // A pass of some projections (`which`) of every expert: each task a run of an expert's tile columns, all its
        // rows; each projection's sums put together from its tasks', then its output transform.
        let pass = |which: &[usize], inputs: &[&[Vec<f32>]]| -> Vec<Vec<Vec<f32>>> {
            let tasks: Vec<(usize, usize, usize, usize)> = which
                .iter()
                .enumerate()
                .flat_map(|(wi, &w)| {
                    let p = self.layout.proj[w];
                    let (ntiles, cols) = (p.n / 16, (TASK_TILES / (p.k / 16)).max(1));
                    (0..used.len()).flat_map(move |i| (0..ntiles).step_by(cols).map(move |c| (wi, i, c, (c + cols).min(ntiles))))
                })
                .collect();
            let sums: Vec<Vec<f32>> = tasks
                .par_iter()
                .map(|&(wi, i, c, end)| {
                    let p = self.layout.proj[which[wi]];
                    let words = LeWords(&records[i][p.trellis..p.trellis + p.trellis_bytes()]);
                    tile_columns(&words, p.k, p.n, p.tile_words, &inputs[wi][i], by[i].len(), c..end)
                })
                .collect();
            let mut raw: Vec<Vec<Vec<f32>>> = which.iter().map(|&w| by.iter().map(|rows| vec![0f32; rows.len() * self.layout.proj[w].n]).collect()).collect();
            for (&(wi, i, c, _), s) in tasks.iter().zip(&sums) {
                let n = self.layout.proj[which[wi]].n;
                let (m, width) = (by[i].len(), s.len() / by[i].len());
                for r in 0..m {
                    raw[wi][i][r * n + c * 16..r * n + c * 16 + width].copy_from_slice(&s[r * width..(r + 1) * width]);
                }
            }
            raw.into_iter()
                .enumerate()
                .map(|(wi, mut per)| {
                    let (w, n) = (which[wi], self.layout.proj[which[wi]].n);
                    per.par_iter_mut()
                        .enumerate()
                        .map(|(i, y)| {
                            let mut o = vec![0f32; y.len()];
                            for (yr, or) in y.chunks_exact_mut(n).zip(o.chunks_exact_mut(n)) {
                                ts[i][w].post(yr, or);
                            }
                            o
                        })
                        .collect()
                })
                .collect()
        };
        // each expert's rows prepared for a projection (each its own input transform)
        let prep = |which: usize, rows_of: &(dyn Fn(usize, usize) -> Vec<f32> + Sync)| -> Vec<Vec<f32>> {
            let k = self.layout.proj[which].k;
            (0..used.len())
                .into_par_iter()
                .map(|i| {
                    let mut xh = vec![0f32; by[i].len() * k];
                    for (slot, chunk) in xh.chunks_exact_mut(k).enumerate() {
                        ts[i][which].pre(&rows_of(i, slot), chunk);
                    }
                    xh
                })
                .collect()
        };
        let row_of = |i: usize, slot: usize| x[by[i][slot] * h..(by[i][slot] + 1) * h].to_vec();
        // gate and up together, then down
        let (gx, ux) = (prep(0, &row_of), prep(1, &row_of));
        let mut gu = pass(&[0, 1], &[&gx, &ux]).into_iter();
        let (mut gate, mut up) = (gu.next().expect("the gates"), gu.next().expect("the ups"));
        // (an adapter's low-rank updates of the gate and up projections, from each expert's rows as they are)
        if self.lora[0].rank != 0 || self.lora[1].rank != 0 {
            for (i, &e) in used.iter().enumerate() {
                let rows: Vec<f32> = by[i].iter().flat_map(|&r| x[r * h..(r + 1) * h].iter().copied()).collect();
                self.lora[0].add(e, h, f, &rows, &mut gate[i]);
                self.lora[1].add(e, h, f, &rows, &mut up[i]);
            }
        }
        let hidden: Vec<Vec<f32>> = gate.iter().zip(&up).map(|(g, u)| g.iter().zip(u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect()).collect();
        let dx = prep(2, &|i: usize, slot: usize| hidden[i][slot * f..(slot + 1) * f].to_vec());
        let mut down = pass(&[2], &[&dx]).pop().expect("the downs");
        for (i, &e) in used.iter().enumerate() {
            self.lora[2].add(e, f, h, &hidden[i], &mut down[i]);
        }
        down
    }

    /// The sums of each row's routed experts that `skip` does not pass over (`assign[r]`: the row's experts and
    /// weights, the shared one last and not among them), each weighted, in the row's own order; the experts read and
    /// run on the host. Also each expert's record, by expert, of those run.
    fn sum_host(&self, x: &[f32], assign: &[Vec<(usize, f32)>], skip: &dyn Fn(usize) -> bool, records: Option<Vec<(usize, Arc<Vec<u8>>)>>) -> (Vec<f32>, Vec<(usize, Arc<Vec<u8>>)>) {
        let h = self.hidden;
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.routed];
        for (r, a) in assign.iter().enumerate() {
            for (j, &(e, _)) in a[..a.len() - 1].iter().enumerate() {
                if !skip(e) {
                    by_expert[e].push((r, j));
                }
            }
        }
        let used: Vec<usize> = (0..self.routed).filter(|&e| !by_expert[e].is_empty()).collect();
        let by: Vec<Vec<usize>> = used.iter().map(|&e| by_expert[e].iter().map(|&(r, _)| r).collect()).collect();
        // (records already read are taken as they are, the rest read now)
        let mut have: Vec<Option<Arc<Vec<u8>>>> = vec![None; self.routed];
        for (e, r) in records.into_iter().flatten() {
            have[e] = Some(r);
        }
        let missing: Vec<usize> = used.iter().copied().filter(|&e| have[e].is_none()).collect();
        for (e, r) in missing.iter().zip(self.records(&missing)) {
            have[*e] = Some(r);
        }
        let mut records: Vec<Arc<Vec<u8>>> = used.iter().map(|&e| have[e].take().expect("every record read")).collect();
        // (a record read ahead that failed is read again here)
        for (r, &e) in records.iter_mut().zip(&used) {
            if r.len() != self.layout.bytes {
                *r = self.records(&[e]).pop().expect("a record");
            }
        }
        let t = std::time::Instant::now();
        let outs = self.run_host(x, &used, &by, &records);
        timed(6, t);
        let mut placed: Vec<Vec<Option<&[f32]>>> = assign.iter().map(|a| vec![None; a.len()]).collect();
        for (n, &e) in used.iter().enumerate() {
            for (slot, &(r, j)) in by_expert[e].iter().enumerate() {
                placed[r][j] = Some(&outs[n][slot * h..(slot + 1) * h]);
            }
        }
        let mut out = vec![0f32; assign.len() * h];
        for (r, row) in out.chunks_exact_mut(h).enumerate() {
            let a = &assign[r];
            for (j, &(_, w)) in a[..a.len() - 1].iter().enumerate() {
                if let Some(y) = placed[r][j] {
                    for (o, v) in row.iter_mut().zip(y) {
                        *o += w * v;
                    }
                }
            }
        }
        (out, used.into_iter().zip(records).collect())
    }

    /// Each row's routed experts (`assign`, the shared one last) summed in its own order, each weighted, and then the
    /// shared expert's `shared` outputs by its gate: every routed expert on the host.
    fn sum_rows(&self, x: &[f32], assign: &[Vec<(usize, f32)>], shared: &[f32]) -> Vec<f32> {
        let h = self.hidden;
        let (mut out, _) = self.sum_host(x, assign, &|_| false, None);
        for (r, row) in out.chunks_exact_mut(h).enumerate() {
            let ws = assign[r][assign[r].len() - 1].1;
            for (o, v) in row.iter_mut().zip(&shared[r * h..(r + 1) * h]) {
                *o += ws * v;
            }
        }
        out
    }

    /// The rows' routing, as the GPU routes them.
    fn route_rows(&self, logits: &Tensor, rows: usize, top_k: usize) -> Vec<Vec<(usize, f32)>> {
        let logits = logits.to_host();
        let width = self.routed + 1;
        assert!(logits.numel() >= rows * width, "tiered experts: {rows} rows' logits");
        (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect()
    }

    /// A chain's run of `x`'s rows routed on the GPU, each expert to its slot ([`Self::record_routed`]): the experts it
    /// does not hold, run here and summed by their weights (`[rows, hidden]`), None where it held every one. Their
    /// slots' recency kept, and the ones missed again taken in (a few a run).
    fn rest(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Option<Tensor> {
        let tier = self.tier.as_ref()?;
        let h = self.hidden;
        let x = x.to_host();
        let rows = x.numel() / h;
        let assign = self.route_rows(logits, rows, top_k);
        let held: Vec<bool> = {
            let mut st = tier.state.lock().unwrap_or_else(|p| p.into_inner());
            st.runs += 1;
            let now = st.runs;
            let mut held = vec![false; self.routed];
            for a in &assign {
                for &(e, _) in &a[..a.len() - 1] {
                    let s = st.slot_of[e];
                    if s != NONE {
                        held[e] = true;
                        st.last[s as usize] = now;
                        st.hits += 1;
                        counted(0, 1);
                    } else {
                        st.misses += 1;
                        counted(1, 1);
                    }
                }
            }
            held
        };
        if assign.iter().all(|a| a[..a.len() - 1].iter().all(|&(e, _)| held[e])) {
            return None;
        }
        let (out, records) = self.sum_host(x.data(), &assign, &|e| held[e], None);
        // the experts missed again (or while a slot is free) taken in, a few a run
        let mut st = tier.state.lock().unwrap_or_else(|p| p.into_inner());
        let now = st.runs;
        let used: Vec<bool> = {
            let mut u = vec![false; self.routed];
            for a in &assign {
                for &(e, _) in &a[..a.len() - 1] {
                    u[e] = true;
                }
            }
            u
        };
        let (window, most) = (take_window(), take_per_run());
        let mut took = 0;
        let t = std::time::Instant::now();
        for (e, record) in &records {
            let free = st.owner.contains(&NONE);
            let again = window == 0 || (st.missed[*e] != 0 && now - st.missed[*e] <= window);
            if took < most && (free || again) && tier.take(&mut st, *e, record, &self.layout, &|o| used[o]) {
                took += 1;
            } else {
                st.missed[*e] = now;
            }
        }
        timed(7, t);
        Some(Tensor::from_vec(out, vec![rows, h]))
    }

    /// A prompt's rows routed on the host ([`Self::plan`]): the experts its chunk uses that the GPU does not hold taken
    /// into the slots of experts it does not use (the most used first), then each row's experts as the GPU runs them
    /// (each by its slot; one still not held weighted 0 at slot 0, the shared one at the slots' count) and the sums of
    /// those still not held, run here (None: none).
    fn plan(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> (Vec<Vec<(usize, f32)>>, Option<Tensor>) {
        let tier = self.tier.as_ref().expect("a plan is of experts the GPU holds some of");
        let h = self.hidden;
        let x = x.to_host();
        let rows = x.numel() / h;
        let assign = self.route_rows(logits, rows, top_k);
        let mut count = vec![0usize; self.routed];
        for a in &assign {
            for &(e, _) in &a[..a.len() - 1] {
                count[e] += 1;
            }
        }
        let mut st = tier.state.lock().unwrap_or_else(|p| p.into_inner());
        st.runs += 1;
        let now = st.runs;
        // what the chunk uses and the GPU lacks, the most used first: taken in while slots of experts it does not use
        // (or free ones) last
        let mut lacking: Vec<usize> = (0..self.routed).filter(|&e| count[e] > 0 && st.slot_of[e] == NONE).collect();
        lacking.sort_by_key(|&e| (std::cmp::Reverse(count[e]), e));
        let room = st.owner.iter().filter(|&&o| o == NONE || count[o as usize] == 0).count();
        let taking: Vec<usize> = lacking.iter().copied().take(room).collect();
        drop(st);
        let read = self.records(&taking);
        let mut st = tier.state.lock().unwrap_or_else(|p| p.into_inner());
        let t = std::time::Instant::now();
        for (&e, record) in taking.iter().zip(&read) {
            tier.take(&mut st, e, record, &self.layout, &|o| count[o] > 0);
        }
        timed(7, t);
        let slots = tier.grouped.slots();
        let mut gpu = Vec::with_capacity(rows);
        for a in &assign {
            let mut g: Vec<(usize, f32)> = a[..a.len() - 1]
                .iter()
                .map(|&(e, w)| match st.slot_of[e] {
                    NONE => (0, 0.0),
                    s => {
                        st.last[s as usize] = now;
                        (s as usize, w)
                    }
                })
                .collect();
            g.push((slots, a[a.len() - 1].1));
            gpu.push(g);
        }
        let held: Vec<bool> = st.slot_of.iter().map(|&s| s != NONE).collect();
        let (hits, misses) = assign.iter().flat_map(|a| a[..a.len() - 1].iter()).fold((0, 0), |(h, m), &(e, _)| if held[e] { (h + 1, m) } else { (h, m + 1) });
        st.hits += hits;
        st.misses += misses;
        counted(0, hits);
        counted(1, misses);
        drop(st);
        if misses == 0 {
            return (gpu, None);
        }
        let (out, _) = self.sum_host(x.data(), &assign, &|e| held[e], None);
        (gpu, Some(Tensor::from_vec(out, vec![rows, h])))
    }

    /// [`Self::sum_host`] of every routed expert in `assign`, for a measurement.
    #[cfg(test)]
    pub(super) fn time_sum_host(&self, x: &[f32], assign: &[Vec<(usize, f32)>]) -> Vec<f32> {
        self.sum_host(x, assign, &|_| false, None).0
    }

    /// Whether these run on `gpu` (some of them held there).
    pub(crate) fn held_on(&self, gpu: &Arc<Gpu>) -> bool {
        self.tier.as_ref().is_some_and(|t| t.grouped.is_on(gpu))
    }

    /// [`Exl3MoeGrouped::record_routed`] of the experts the GPU holds, each routed to its slot (one it does not hold
    /// weighted 0: [`Self::rest`] is the host's to add).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_routed(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, into: Option<(&DeviceVec, &DeviceVec, usize)>) -> bool {
        let Some(t) = &self.tier else { return false };
        // (a prompt's rows are planned on the host: their experts taken in first)
        rows <= 8 && t.grouped.record_routed_mapped(rec, x, out, logits, self.routed, Some(&t.map), top_k, rows, into)
    }

    /// [`Exl3MoeGrouped::record`] of rows as [`Self::plan`] gave them (each expert by its slot).
    pub(crate) fn record(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]) {
        self.tier.as_ref().expect("rows planned for experts the GPU holds some of").grouped.record(rec, x, out, assign)
    }

    /// The slots' counts so far: experts the runs found held and not, and those taken in.
    pub fn tier_counts(&self) -> Option<(u64, u64, u64)> {
        self.tier.as_ref().map(|t| {
            let st = t.state.lock().unwrap_or_else(|p| p.into_inner());
            (st.hits, st.misses, st.taken)
        })
    }
}

impl ggml_rs::exl3::Experts for Exl3MoeTiered {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let x = x.to_host();
        let shared = self.shared_rows(&x);
        self.forward_given(&x, logits, top_k, &shared)
    }

    fn on_host(&self) -> bool {
        self.tier.is_none()
    }

    fn host_rest(&self) -> bool {
        self.tier.is_some()
    }

    fn forward_rest(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Option<Tensor> {
        self.rest(x, logits, top_k)
    }

    fn plan_rows(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Option<(Vec<Vec<(usize, f32)>>, Option<Tensor>)> {
        self.tier.as_ref()?;
        Some(self.plan(x, logits, top_k))
    }

    fn rest_counts(&self) -> Option<(u64, u64, u64)> {
        self.tier_counts()
    }

    fn read_ahead(&self, logits: &Tensor, top_k: usize) {
        Exl3MoeTiered::read_ahead(self, logits, top_k)
    }

    fn forward_given(&self, x: &Tensor, logits: &Tensor, top_k: usize, shared: &[f32]) -> Tensor {
        let h = self.hidden;
        let x = x.to_host();
        let logits = logits.to_host();
        let rows = x.numel() / h;
        let width = self.routed + 1;
        assert!(logits.numel() >= rows * width && shared.len() >= rows * h, "tiered experts: {rows} rows' logits and shared outputs");
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect();
        Tensor::from_vec(self.sum_rows(x.data(), &assign, shared), vec![rows, h])
    }

    fn low_rank(&mut self, which: usize, slot_of: &[u32], a: &[f32], b: &[f32], rank: usize) -> Result<(), String> {
        if which > 2 || slot_of.len() != self.routed + 1 {
            return Err(format!("a low-rank update of projection {which} for {} experts, of {} and the shared one", slot_of.len(), self.routed));
        }
        let (k, n) = if which == 2 { (self.ff, self.hidden) } else { (self.hidden, self.ff) };
        if rank == 0 || a.is_empty() || a.len() % (rank * k) != 0 || b.len() != a.len() / (rank * k) * n * rank {
            return Err(format!("a rank {rank} update of {} and {} floats beside experts' projections [{n}, {k}]", a.len(), b.len()));
        }
        let at: Vec<(usize, u32)> = slot_of[..self.routed].iter().enumerate().filter(|(_, &s)| s != NONE).map(|(e, &s)| (e, s)).collect();
        if !at.is_empty() && self.tier.is_some() {
            return Err("a low-rank update of routed experts the GPU holds some of (not yet: the updates would follow the experts' slots)".into());
        }
        if slot_of[self.routed] != NONE {
            let s = slot_of[self.routed] as usize;
            self.shared[which] = self.b.low_rank(Arc::clone(&self.shared[which]), a[s * rank * k..(s + 1) * rank * k].to_vec(), b[s * n * rank..(s + 1) * n * rank].to_vec(), rank)?;
            if let Some(t) = &mut self.tier {
                t.grouped.shared[which] = Arc::clone(&self.shared[which]);
            }
        }
        if !at.is_empty() {
            self.lora[which] = std::mem::take(&mut self.lora[which]).with(self.routed, &at, a, b, rank, k, n)?;
        }
        Ok(())
    }
}
