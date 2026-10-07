//! WebGPU backend for ggml-rs: quantized weights on any GPU wgpu reaches
//! (Direct3D 12, Vulkan, Metal), for machines without CUDA.
//!
//! The load-bearing op of a GGUF model is `linear_q`: every projection reads a
//! quantized weight matrix. This backend uploads those matrices to the GPU in
//! their GGML block layout (so VRAM use equals the file's size) and runs the
//! matmul in WGSL (see [`shaders`]); everything else -- norms, RoPE, attention,
//! recurrent state -- runs on [`CpuBackend`], with activations on the host.
//! A projection is one upload of `x`, one dispatch and one read-back.
//!
//! Weights beyond the VRAM budget, or in a type without a shader, stay on the
//! host and take `CpuBackend`'s path: a model larger than the GPU still runs,
//! the rest of it on the CPU.

// VENDORED-LOCAL: this crate is OAIY's addition beside ggml-rs-cuda.

pub mod chain;
pub mod dense;
pub mod exl3;
pub mod quant_linear;
pub mod quant_moe;
pub mod shaders;

/// Where a GGUF model's time goes on this backend: counters any thread adds to and a timing test reads and resets.
pub mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// A quantized projection's call: all of it, and the part spent waiting for the GPU (submit to mapped).
    pub static LINEAR: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    pub static LINEAR_WAIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    /// The attention (on the CPU: the cache is on the host).
    pub static ATTENTION: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    /// A chain's run: encoding its dispatches (to the submit), and the GPU's part (the submit to its reads mapped).
    pub static CHAIN_ENCODE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    pub static CHAIN_WAIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

    pub(crate) fn add(counter: &[AtomicU64; 2], start: Instant) {
        counter[0].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        counter[1].fetch_add(1, Ordering::Relaxed);
    }

    /// The counters since the last call, as one line, and reset.
    pub fn take_line() -> String {
        let take = |c: &[AtomicU64; 2]| (c[0].swap(0, Ordering::Relaxed) as f64 / 1e9, c[1].swap(0, Ordering::Relaxed));
        let (l, w, a) = (take(&LINEAR), take(&LINEAR_WAIT), take(&ATTENTION));
        let (e, cw) = (take(&CHAIN_ENCODE), take(&CHAIN_WAIT));
        format!("projections {:.3} s ({}), of it waiting for the GPU {:.3} s; attention {:.3} s ({}); chains encoding {:.3} s ({}), on the GPU {:.3} s", l.0, l.1, w.0, a.0, a.1, e.0, e.1, cw.0)
    }

    /// A chain's kernels timed on the GPU (`OAIY_CHAIN_PROFILE`): each dispatch in a pass of its own between two
    /// timestamps (which costs a little), its time added to its kernel's.
    pub fn chain_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("OAIY_CHAIN_PROFILE").is_some())
    }

    /// A chain's pieces timed on the GPU (`OAIY_PIECE_STAMPS`): each submitted pass between two timestamps, and a
    /// recording's busy time (its passes' own) against its span (its first's start to its last's end) added up.
    pub fn pieces_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("OAIY_PIECE_STAMPS").is_some())
    }

    /// Recordings' pieces' busy time and span (ns), and the pieces.
    pub(crate) static PIECES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

    /// The pieces' busy time and span since the last call (ms), and how many, and reset.
    pub fn take_pieces() -> (f64, f64, u64) {
        let take = |i: usize| PIECES[i].swap(0, Ordering::Relaxed);
        (take(0) as f64 / 1e6, take(1) as f64 / 1e6, take(2))
    }

    /// Each kernel's GPU time (ns) and dispatches.
    pub(crate) static KERNELS: std::sync::Mutex<std::collections::BTreeMap<&'static str, (u64, u64)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

    /// The kernels' GPU time since the last call (ms, and dispatches), the most first, and reset.
    pub fn take_kernels() -> Vec<(&'static str, f64, u64)> {
        let mut k = KERNELS.lock().unwrap_or_else(|p| p.into_inner());
        let mut v: Vec<(&'static str, f64, u64)> = std::mem::take(&mut *k).into_iter().map(|(n, (ns, c))| (n, ns as f64 / 1e6, c)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

use ggml_quants::GgmlType;
use ggml_rs::{Backend, CpuBackend, QuantizedDeviceStorage, QuantizedTensor, RopeType, Tensor};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const GIB: u64 = 1 << 30;
/// The scratch a GPU's pool keeps between chains' runs: a prompt's layer's (Qwen3.8-Flash-Next's at 512 rows about
/// 0.6 GiB), within the 4 GiB a card's budget leaves.
const POOL_BYTES: u64 = 3 * GIB / 2;
/// The read-backs' staging a GPU keeps between chains' runs: two of a prompt's chunks' (some 70 MB each).
const STAGING_BYTES: u64 = GIB / 4;

/// The adapter's own memory where its API says: Vulkan's largest device-local heap (a discrete card's VRAM). None on
/// Direct3D 12 and Metal.
fn device_memory(adapter: &wgpu::Adapter) -> Option<u64> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        // SAFETY: the adapter's handles are only read, by a query that creates and frees nothing.
        let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let props = unsafe { a.shared_instance().raw_instance().get_physical_device_memory_properties(a.raw_physical_device()) };
        let heaps = &props.memory_heaps[..(props.memory_heap_count as usize).min(props.memory_heaps.len())];
        // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        heaps.iter().filter(|h| h.flags.as_raw() & 1 != 0).map(|h| h.size).max()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = adapter;
        None
    }
}

/// Vulkan's budget for this process on the adapter's largest device-local heap (VK_EXT_memory_budget: what the OS
/// lets it keep there, the rest of the computer's use of the card taken off) and its use of it now. None on other
/// APIs.
fn heap_budget(adapter: &wgpu::Adapter) -> Option<(u64, u64)> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        // SAFETY: the adapter's handles are only read, by a query that creates and frees nothing.
        let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let instance = a.shared_instance().raw_instance();
        let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let heaps = {
            let mut props = ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe { instance.get_physical_device_memory_properties2(a.raw_physical_device(), &mut props) };
            props.memory_properties
        };
        let n = (heaps.memory_heap_count as usize).min(heaps.memory_heaps.len());
        (0..n).filter(|&i| heaps.memory_heaps[i].flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| (budget.heap_budget[i], budget.heap_usage[i])).max_by_key(|&(b, _)| b)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = adapter;
        None
    }
}

/// The weights a discrete card with `memory` bytes holds by default: all but 4 GiB (the cache, the work buffers and
/// the rest of the computer's use of it), or half of a card under 8 GiB.
fn discrete_budget(memory: u64) -> u64 {
    memory.saturating_sub(4 * GIB).max(memory / 2)
}

/// Weight bytes one storage binding may cover: rows are split across buffers
/// below the adapter's binding limit (a 152k-vocab Q6_K output is ~640 MB).
fn chunk_limit(limits: &wgpu::Limits) -> u64 {
    (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size).min(1 << 30) & !3
}

/// The driver's calls a [`Gpu`]'s watchdog watches ([`Gpu::wait`]'s polls, a second's timeout each): how many are in
/// progress, and when one last began or returned (ms since the first watched). None returned in [`HUNG`] has hung in
/// the driver: NVIDIA's Vulkan, its device lost to a reset (Windows' TDR: a submission past 2 s), spins in its fence
/// wait whatever the timeout; said, and the process ended where it is one job's ([`end_process_on_hang`]).
#[derive(Default)]
struct Watch {
    calls: AtomicU64,
    stamp: AtomicU64,
}

/// The bytes of a WGSL module's workgroup variables, as its layout has them (None: it does not parse here).
fn workgroup_bytes(source: &str) -> Option<u64> {
    use wgpu::naga;
    let module = naga::front::wgsl::parse_str(source).ok()?;
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).ok()?;
    Some(module.global_variables.iter().filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup).map(|(_, g)| layouter[g.ty].size as u64).sum())
}

/// How long a watched call may go without one returning.
const HUNG: std::time::Duration = std::time::Duration::from_secs(30);

/// Milliseconds since the first call (never 0).
fn watch_clock() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

impl Watch {
    /// A watched call begun, ended when the guard is dropped.
    fn enter(&self) -> Watched<'_> {
        self.stamp.store(watch_clock(), Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Watched(self)
    }
}

struct Watched<'a>(&'a Watch);

impl Drop for Watched<'_> {
    fn drop(&mut self) {
        self.0.stamp.store(watch_clock(), Ordering::Relaxed);
        self.0.calls.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Whether a hung driver call ends the process ([`end_process_on_hang`]); else it is only said, once.
static END_ON_HANG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A driver call hung on a lost device ([`HUNG`] without one returning) ends this process, saying why on stderr: for a
/// process of one job (the media worker: its last line the job's error, rather than a job that never ends). Elsewhere
/// (a server's other models on other devices) it is only said.
pub fn end_process_on_hang() {
    END_ON_HANG.store(true, Ordering::Relaxed);
}

/// `watch`'s calls checked every second while its GPU lives: one hung ([`HUNG`]) said on stderr, and the process ended
/// where [`end_process_on_hang`] has asked.
fn watchdog(watch: std::sync::Weak<Watch>, lost: std::sync::Weak<Mutex<Option<String>>>, name: String) {
    let mut said = false;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let Some(w) = watch.upgrade() else { return };
        if w.calls.load(Ordering::Relaxed) == 0 {
            said = false;
            continue;
        }
        let quiet = watch_clock().saturating_sub(w.stamp.load(Ordering::Relaxed));
        if quiet > HUNG.as_millis() as u64 && !said {
            let why = lost.upgrade().and_then(|l| l.lock().unwrap_or_else(|p| p.into_inner()).clone());
            let end = END_ON_HANG.load(Ordering::Relaxed);
            eprintln!(
                "webgpu: {name}'s driver has not returned from a wait in {} s: its device is lost{} (a reset: a submission past the OS's GPU time limit, Windows' 2 s TDR){}",
                quiet / 1000,
                why.map_or(String::new(), |w| format!(" ({w})")),
                if end { "; ending the process" } else { "" }
            );
            if end {
                end_process();
            }
            said = true;
        }
    }
}

/// OAIY_PIECES_IN_FLIGHT's, where it is set: every device's pieces in flight at most that many, whatever was asked
/// of it ([`WgpuBackend::pieces_in_flight_at_most`]; 0: as many as are recorded).
fn pieces_in_flight_asked() -> Option<usize> {
    static ASKED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| std::env::var("OAIY_PIECES_IN_FLIGHT").ok().and_then(|v| v.parse().ok()))
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.feed.state.lock().unwrap_or_else(|p| p.into_inner()).closed = true;
        self.feed.changed.notify_all();
    }
}

/// This process ended at once, its exit code 3: no DLL's detach run (a GPU driver's would wait for its hung threads).
fn end_process() -> ! {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn TerminateProcess(process: isize, code: u32) -> i32;
        }
        // SAFETY: the current process's pseudo-handle, ended
        unsafe {
            TerminateProcess(GetCurrentProcess(), 3);
        }
    }
    std::process::abort()
}

/// A dispatch of a chain's: its pipeline, its bind group and its grid of workgroups.
pub(crate) type Dispatch = (Arc<wgpu::ComputePipeline>, wgpu::BindGroup, (u32, u32, u32));

/// A chain's piece on its way to a device's queue ([`Gpu::submit_piece`], [`Gpu::feed`]): gone (its submission), or
/// to go once the device's pieces in flight allow, its submission put here then.
pub(crate) enum Piece {
    Gone(wgpu::SubmissionIndex),
    Fed(Arc<Mutex<Option<wgpu::SubmissionIndex>>>),
}

/// What a device's feed is handed: a piece's dispatches, encoded when its turn comes, command buffers as they are,
/// or a write of the host's into a buffer (its bytes, at an offset) made in its turn.
enum Fare {
    Dispatches(Vec<Dispatch>),
    Commands(Vec<wgpu::CommandBuffer>),
    Write(wgpu::Buffer, u64, Vec<u8>),
}

/// `data` (a multiple of 4 bytes) into `buffer` at `offset` through a write's staging memory, copied there from every
/// core: one core's copy into the card's memory (Resizable BAR) took 1.4 GB/s, most of a weight's load. Small
/// writes as `write_buffer`'s.
fn write_bytes(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    let size = match wgpu::BufferSize::new(data.len() as u64) {
        Some(size) if data.len() >= 8 << 20 => size,
        _ => return queue.write_buffer(buffer, offset, data),
    };
    let Some(mut view) = queue.write_buffer_with(buffer, offset, size) else { return queue.write_buffer(buffer, offset, data) };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let each = data.len().div_ceil(threads).div_ceil(4096) * 4096;
    let mut whole = view.slice(..);
    let base = whole.as_raw_ptr().cast::<u8>().as_ptr() as usize;
    std::thread::scope(|s| {
        for (i, part) in data.chunks(each).enumerate() {
            let at = base + i * each;
            // SAFETY: each thread writes its own bytes of the staging memory, mapped for writing until the view
            // (which outlives this scope) is dropped; nothing reads them
            s.spawn(move || unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), at as *mut u8, part.len()) });
        }
    });
}

/// A device's pieces waiting for its queue, where its pieces in flight are limited
/// ([`WgpuBackend::pieces_in_flight_at_most`]): a thread of the device's ([`feed_pieces`]) encodes and submits each in
/// turn once all but the limit less one of those before it have run, so whoever recorded them goes on meanwhile (a
/// prompt over two cards records one's chunk as the other runs). A piece's command buffer is made there, just before
/// it is submitted: an RTX 5090 under a power limit runs a prompt's chunks steadily so, and throttles itself for
/// seconds at a time through the same pieces encoded as they were recorded and submitted in the same turn, however
/// few of them were on its queue at once (15,360 tokens of the 27B, four times over: 6.8 s each after the first's
/// once-only second, where 9.2, 13.9, 13.1 and 6.8). Anything else of the queue's waits for the pieces handed over
/// first ([`Gpu::queue`]): a write is before the submissions after it.
#[derive(Default)]
struct Feed {
    state: Mutex<Fed>,
    changed: std::sync::Condvar,
}

#[derive(Default)]
struct Fed {
    /// What was handed over and is not yet submitted, in turn, each with where its submission is wanted.
    waiting: std::collections::VecDeque<(Fare, Arc<Mutex<Option<wgpu::SubmissionIndex>>>)>,
    /// How many were handed over, and how many of them are submitted.
    handed: u64,
    gone: u64,
    /// The device's thread runs (from the first piece handed over), and is to end (the device dropped).
    fed: bool,
    closed: bool,
    /// Why the thread stopped, if it did: a lost device, or an error of the queue's.
    failed: Option<String>,
}

/// Waits for `device`'s submission `index` (else everything submitted), a second at a time: Err once its loss has been
/// said, or the wait itself fails.
fn wait_for(device: &wgpu::Device, watch: &Watch, lost: &Mutex<Option<String>>, index: Option<wgpu::SubmissionIndex>) -> Result<(), String> {
    loop {
        let polled = {
            let _watched = watch.enter();
            device.poll(wgpu::PollType::Wait { submission_index: index.clone(), timeout: Some(std::time::Duration::from_secs(1)) })
        };
        match polled {
            Ok(_) => return Ok(()),
            Err(wgpu::PollError::Timeout) => {}
            Err(e) => return Err(format!("waiting for the GPU: {e}")),
        }
        if let Some(why) = lost.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            return Err(format!("the device was lost ({why})"));
        }
    }
}

/// `dispatches` as one pass's command buffer.
fn encode(device: &wgpu::Device, dispatches: &[Dispatch]) -> wgpu::CommandBuffer {
    let mut piece = device.create_command_encoder(&Default::default());
    {
        let mut pass = piece.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for (pipeline, group, (x, y, z)) in dispatches {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*x, *y, *z);
        }
    }
    piece.finish()
}

/// A device's [`Feed`]'s thread: what was handed over submitted in turn (a piece's dispatches encoded then), each
/// once the device's pieces in flight are fewer than `limit` (as it is then; 0: at once). It ends with the device, or
/// at a failure (said to whoever waits on the feed next).
fn feed_pieces(feed: Arc<Feed>, device: wgpu::Device, queue: wgpu::Queue, watch: Arc<Watch>, lost: Arc<Mutex<Option<String>>>, limit: Arc<std::sync::atomic::AtomicUsize>) {
    let mut flying: std::collections::VecDeque<wgpu::SubmissionIndex> = std::collections::VecDeque::new();
    loop {
        let (fare, slot) = {
            let mut s = feed.state.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(next) = s.waiting.pop_front() {
                    break next;
                }
                if s.closed {
                    return;
                }
                s = feed.changed.wait(s).unwrap_or_else(|p| p.into_inner());
            }
        };
        // (an error of the queue's is a panic of its handler's: this thread's, so said to the feed's users)
        let gone = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // (a write: made now, after the pieces before it and before those after; nothing of the GPU's to wait for)
            let fare = match fare {
                Fare::Write(buffer, offset, data) => {
                    write_bytes(&queue, &buffer, offset, &data);
                    return Ok::<_, String>(None);
                }
                fare => fare,
            };
            let most = pieces_in_flight_asked().unwrap_or_else(|| limit.load(Ordering::Relaxed));
            while most > 0 && flying.len() >= most {
                let oldest = flying.pop_front().expect("a piece in flight");
                wait_for(&device, &watch, &lost, Some(oldest))?;
            }
            if most == 0 {
                flying.clear();
            }
            let commands = match fare {
                Fare::Dispatches(dispatches) => vec![encode(&device, &dispatches)],
                Fare::Commands(commands) => commands,
                Fare::Write(..) => unreachable!("a write is made above"),
            };
            let index = queue.submit(commands);
            flying.push_back(index.clone());
            Ok(Some(index))
        }));
        let mut s = feed.state.lock().unwrap_or_else(|p| p.into_inner());
        match gone {
            Ok(Ok(index)) => {
                *slot.lock().unwrap_or_else(|p| p.into_inner()) = index;
                s.gone += 1;
            }
            Ok(Err(why)) => s.failed = Some(why),
            Err(panic) => s.failed = Some(panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|m| m.to_string())).unwrap_or_else(|| "its queue's thread panicked".into())),
        }
        let failed = s.failed.is_some();
        drop(s);
        feed.changed.notify_all();
        if failed {
            return;
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    /// The device's queue as it is; [`Gpu::queue`] for anyone's use of it but the feed's.
    queue_raw: wgpu::Queue,
    /// Why the device was lost, once its callback has said (a wait then fails rather than waiting on).
    lost: Arc<Mutex<Option<String>>>,
    /// The driver's calls a watchdog thread watches (a hung one ends the process).
    watch: Arc<Watch>,
    /// The chains' pieces on their way to the queue where the device's pieces in flight are limited ([`Gpu::feed`]),
    /// and how many may be in flight at once (0: as many as are recorded).
    feed: Arc<Feed>,
    in_flight_limit: Arc<std::sync::atomic::AtomicUsize>,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: Mutex<HashMap<(GgmlType, u8), Arc<wgpu::ComputePipeline>>>,
    /// The EXL3 matmul's pipelines (`exl3::shader`): for one row, and for several. Made when first used.
    exl3: Mutex<[Option<Arc<wgpu::ComputePipeline>>; 2]>,
    /// Other kernels' pipelines by name (`dense`), made when first used.
    named: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
    /// The pipelines' names by address, for a chain's profile.
    names: Mutex<HashMap<usize, &'static str>>,
    /// Chains' scratch buffers between their runs (bytes, buffer): a prompt's layer takes the last one's, where a new
    /// buffer is allocated and cleared before its first use.
    pool: Mutex<Vec<(u64, wgpu::Buffer)>>,
    /// Read-backs' staging buffers (host memory the GPU copies into) a chain's reads use again: a prompt's chunk read
    /// 64 MB of its cache's rows into new ones, each a wait of the OS's before the GPU could start the next.
    staging: Mutex<Vec<(u64, wgpu::Buffer)>>,
    /// A chain's bind groups that are the same step after step (`chain`): by pipeline, buffers and parameters.
    chain_groups: Mutex<HashMap<chain::GroupKey, wgpu::BindGroup>>,
    /// The layout of the chain's kernels of eight buffers (a gated delta net's: six read, two written, then the
    /// parameters), made when first used, and their bind groups as `chain_groups`.
    wide: std::sync::OnceLock<(wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    chain_groups_wide: Mutex<HashMap<chain::WideKey, wgpu::BindGroup>>,
    /// Small buffers for a kernel's bindings it does not use: one read, one written.
    dummy: std::sync::OnceLock<wgpu::Buffer>,
    dummy_rw: std::sync::OnceLock<wgpu::Buffer>,
    limits: wgpu::Limits,
    /// Upload bytes written since the queue was last flushed.
    staged: AtomicU64,
    /// The EXL3 projections' scratch for a few rows (a check of drafts), shared by them all, made when first used.
    few: std::sync::OnceLock<exl3::FewScratch>,
    /// Grouped experts' kept scratch of a step or a check (rows, top k, hidden, ff and the groups' splits): one for
    /// every layer of that shape.
    moe_steps: Mutex<Vec<([usize; 6], Arc<exl3::Step>)>>,
    /// How many units (SMs) the tensor cores' matmuls share out ([`Gpu::coop_units`]), counted when first asked.
    coop_units: std::sync::OnceLock<u32>,
}

/// A weight matrix on the GPU: its rows in one or more buffers.
struct WgpuQuant {
    gpu: Arc<Gpu>,
    dtype: GgmlType,
    /// `(buffer, first_row, rows)`; each buffer holds whole rows.
    chunks: Vec<(wgpu::Buffer, u32, u32)>,
    row_bytes: usize,
    nbytes: usize,
    used: Arc<AtomicU64>,
}

impl std::fmt::Debug for WgpuQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuQuant({:?}, {} bytes, {} chunk(s))", self.dtype, self.nbytes, self.chunks.len())
    }
}

impl Drop for WgpuQuant {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes as u64, Ordering::Relaxed);
    }
}

impl WgpuQuant {
    /// The weights as the GPU holds them.
    fn gpu_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nbytes);
        for (buffer, _, rows) in &self.chunks {
            let len = *rows as usize * self.row_bytes;
            out.extend_from_slice(&self.gpu.read(buffer, len as u64)[..len]);
        }
        out
    }
}

impl QuantizedDeviceStorage for WgpuQuant {
    fn nbytes(&self) -> usize {
        self.nbytes
    }
    fn dtype(&self) -> GgmlType {
        self.dtype
    }
    fn device_name(&self) -> &str {
        "webgpu"
    }
    fn copy_to_host(&self) -> Vec<u8> {
        let out = self.gpu_bytes();
        // ggml's layout, where the GPU's blocks are padded
        match shaders::padded_block(self.dtype) {
            Some((host, gpu, at)) => shaders::pad_blocks(&out, gpu, host, at),
            None => out,
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_to_device(&self) -> Box<dyn QuantizedDeviceStorage> {
        let bytes = self.gpu_bytes();
        self.used.fetch_add(self.nbytes as u64, Ordering::Relaxed);
        Box::new(WgpuQuant {
            gpu: Arc::clone(&self.gpu),
            dtype: self.dtype,
            chunks: self.gpu.upload_rows(&bytes, self.row_bytes, self.chunks.len()),
            row_bytes: self.row_bytes,
            nbytes: self.nbytes,
            used: Arc::clone(&self.used),
        })
    }
}

impl Gpu {
    /// The EXL3 projections' few-rows scratch ([`exl3::FewScratch`]), made by `b` when first asked for.
    pub(crate) fn few(&self, b: &WgpuBackend) -> &exl3::FewScratch {
        self.few.get_or_init(|| exl3::FewScratch::new(b))
    }

    /// How many units (SMs) the GPU's workgroups are shared out to ([`shaders::COOP_UNITS_PROBE`]'s count of 4,096
    /// workgroups of 1024 threads that each work about 0.1 ms: those started before any had finished), counted once
    /// (a millisecond or so); 1 where a workgroup cannot be that large.
    pub(crate) fn coop_units(&self) -> u32 {
        *self.coop_units.get_or_init(|| {
            if self.limits.max_compute_invocations_per_workgroup < 1024 || self.limits.max_compute_workgroup_size_x < 1024 {
                return 1;
            }
            let pipeline = self.named_pipeline("coop-units-probe", || shaders::COOP_UNITS_PROBE.to_string());
            let counts = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-coop-units"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let params = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-coop-units-params"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut words = [0u8; 32];
            words[..4].copy_from_slice(&65536u32.to_le_bytes());
            self.queue().write_buffer(&params, 0, &words);
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-coop-units"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.dummy().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.dummy().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                ],
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(4096, 1, 1);
            }
            self.queue().submit([enc.finish()]);
            let got = self.read(&counts, 8);
            u32::from_le_bytes([got[0], got[1], got[2], got[3]]).max(1)
        })
    }

    /// Copy `len` bytes of `src` back to the host.
    fn read(&self, src: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let len = len.div_ceil(4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-readback"),
            size: len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, len);
        self.queue().submit([enc.finish()]);
        self.map_read(&staging, len)
    }

    /// The device's queue, for a write or a submission of the caller's own: after everything handed to the feed has
    /// gone to it (a write is before the submissions after it; a piece still waiting would run after a write made
    /// since it was recorded).
    fn queue(&self) -> &wgpu::Queue {
        self.settle();
        &self.queue_raw
    }

    /// Whether the device's pieces in flight are limited: its chains' pieces go by its feed.
    pub(crate) fn feeds(&self) -> bool {
        pieces_in_flight_asked().unwrap_or_else(|| self.in_flight_limit.load(Ordering::Relaxed)) > 0
    }

    /// Waits until everything handed to the feed is submitted (at once where nothing is waiting).
    fn settle(&self) {
        let mut s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
        while s.gone < s.handed {
            if let Some(why) = &s.failed {
                panic!("webgpu: {why}");
            }
            s = self.feed.changed.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// `fare` handed to the feed (its thread begun at the first), where its submission will be said.
    fn hand(&self, fare: Fare) -> Piece {
        let slot = Arc::new(Mutex::new(None));
        let mut s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(why) = &s.failed {
            panic!("webgpu: {why}");
        }
        s.waiting.push_back((fare, Arc::clone(&slot)));
        s.handed += 1;
        if !s.fed {
            s.fed = true;
            let (feed, device, queue, watch, lost, limit) = (Arc::clone(&self.feed), self.device.clone(), self.queue_raw.clone(), Arc::clone(&self.watch), Arc::clone(&self.lost), Arc::clone(&self.in_flight_limit));
            std::thread::Builder::new().name("oaiy-webgpu-feed".into()).spawn(move || feed_pieces(feed, device, queue, watch, lost, limit)).expect("webgpu: a thread for the device's queue");
        }
        drop(s);
        self.feed.changed.notify_all();
        Piece::Fed(slot)
    }

    /// A chain's piece as its dispatches: where the device's pieces in flight are limited
    /// ([`WgpuBackend::pieces_in_flight_at_most`]), to its feed, which encodes and submits it once all but the limit
    /// less one of those before it have run (the caller goes on at once); else encoded and submitted here.
    pub(crate) fn feed(&self, dispatches: Vec<Dispatch>) {
        if self.feeds() {
            self.hand(Fare::Dispatches(dispatches));
        } else {
            self.queue().submit([encode(&self.device, &dispatches)]);
        }
    }

    /// A chain's piece its recorder encoded (its command buffers, in turn) submitted: where the device's pieces in
    /// flight are limited, by its feed a command buffer a piece, and waited for until the last of them has gone to the
    /// queue (all but the limit less one of those before it have run: such a piece keeps its recorder to the queue's
    /// pace). The piece returned is the last of them.
    pub(crate) fn submit_piece(&self, commands: Vec<wgpu::CommandBuffer>) -> Piece {
        if !self.feeds() || commands.is_empty() {
            return Piece::Gone(self.queue().submit(commands));
        }
        let mut last = None;
        for command in commands {
            last = Some(self.hand(Fare::Commands(vec![command])));
        }
        Piece::Gone(self.gone(last.expect("a command buffer")))
    }

    /// A recording's copies out (its reads, its timestamps) submitted after its pieces: by the feed where the device's
    /// pieces in flight are limited (the caller goes on at once), else here.
    pub(crate) fn submit_after(&self, commands: Vec<wgpu::CommandBuffer>) -> Piece {
        if self.feeds() {
            self.hand(Fare::Commands(commands))
        } else {
            Piece::Gone(self.queue().submit(commands))
        }
    }

    /// `piece`'s submission, once it has gone to the queue (a buffer it copies into can be mapped from then, not
    /// before).
    pub(crate) fn gone(&self, piece: Piece) -> wgpu::SubmissionIndex {
        let slot = match piece {
            Piece::Gone(index) => return index,
            Piece::Fed(slot) => slot,
        };
        let mut s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(index) = slot.lock().unwrap_or_else(|p| p.into_inner()).clone() {
                return index;
            }
            if let Some(why) = &s.failed {
                panic!("webgpu: {why}");
            }
            s = self.feed.changed.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Waits for submission `index` (else everything submitted, the feed's pieces too), a second at a time: a lost
    /// device (its callback has said so) fails the job rather than leaving it waiting for good.
    pub(crate) fn wait(&self, index: Option<wgpu::SubmissionIndex>) {
        if index.is_none() {
            self.settle();
        }
        if let Err(why) = wait_for(&self.device, &self.watch, &self.lost, index) {
            panic!("webgpu: {why}");
        }
    }

    fn map_read(&self, staging: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let slice = staging.slice(..len);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.wait(None);
        let out = slice.get_mapped_range().expect("webgpu: mapping a finished buffer").to_vec();
        staging.unmap();
        out
    }

    /// `data` (a multiple of 4 bytes) into `buffer` at `offset` ([`write_bytes`]): at once, or, behind pieces still
    /// waiting for the queue, in its turn after them by the feed (a copy of the bytes), the caller going on (a
    /// prompt's next chunk is recorded as the one before runs: its embeddings' upload waiting for the queue kept the
    /// recording to the GPU's last pieces).
    pub(crate) fn write(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let holds = {
            let s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
            s.gone < s.handed && s.failed.is_none()
        };
        if holds {
            self.hand(Fare::Write(buffer.clone(), offset, data.to_vec()));
        } else {
            write_bytes(self.queue(), buffer, offset, data);
        }
    }

    /// Upload whole rows into buffers below the binding limit.
    fn upload_rows(&self, bytes: &[u8], row_bytes: usize, _hint: usize) -> Vec<(wgpu::Buffer, u32, u32)> {
        let rows = bytes.len() / row_bytes;
        let per = ((chunk_limit(&self.limits) as usize) / row_bytes).max(1);
        let mut chunks = Vec::new();
        let mut r = 0;
        while r < rows {
            let n = per.min(rows - r);
            let data = &bytes[r * row_bytes..(r + n) * row_bytes];
            let size = (data.len() as u64).div_ceil(4) * 4;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-weight"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if data.len() % 4 == 0 {
                self.write(&buffer, 0, data);
            } else {
                let mut padded = data.to_vec();
                padded.resize(size as usize, 0);
                self.write(&buffer, 0, &padded);
            }
            chunks.push((buffer, r as u32, n as u32));
            r += n;
            // `write_buffer` stages through host-visible memory that is only
            // recycled after a submission completes: flush as uploads pile up,
            // or loading a 16 GB model exhausts the staging pool.
            let pending = self.staged.fetch_add(size, Ordering::Relaxed) + size;
            if pending >= 256 << 20 {
                self.staged.store(0, Ordering::Relaxed);
                self.queue().submit([]);
                self.wait(None);
            }
        }
        chunks
    }

    /// A kernel's module, its workgroup memory checked first where the device has less than 48 KB of it (Apple's 32,
    /// OAIY_PORTABLE_LIMITS's): wgpu 30 does not check it, and a kernel past the limit fails where it runs, if its
    /// driver says at all.
    fn shader(&self, name: &str, source: String) -> wgpu::ShaderModule {
        if self.limits.max_compute_workgroup_storage_size < 48 << 10 {
            if let Some(bytes) = workgroup_bytes(&source) {
                assert!(
                    bytes <= self.limits.max_compute_workgroup_storage_size as u64,
                    "webgpu: kernel {name} takes {bytes} bytes of a workgroup's memory, past this device's {}",
                    self.limits.max_compute_workgroup_storage_size
                );
            }
        }
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(source.into()) })
    }

    fn exl3_pipeline(&self, many: bool) -> Arc<wgpu::ComputePipeline> {
        let mut slots = self.exl3.lock().unwrap_or_else(|p| p.into_inner());
        let slot = &mut slots[many as usize];
        if let Some(p) = slot.as_ref() {
            return Arc::clone(p);
        }
        let module = self.shader("oaiy-exl3", exl3::shader(many));
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oaiy-exl3"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        *slot = Some(Arc::clone(&pipeline));
        pipeline
    }

    /// The pipeline `name`, made from the WGSL `source` gives the first time it is asked for.
    fn named_pipeline(&self, name: &'static str, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        self.named_pipeline_in(name, &self.pipeline_layout, source)
    }

    /// The layout of eight storage buffers (the first six read, the last two written) and the parameters at 8.
    fn wide_layout(&self) -> &(wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        self.wide.get_or_init(|| {
            let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8).map(|b| storage(b, b < 6)).collect();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("oaiy-chain-wide"), entries: &entries });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("oaiy-chain-wide"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            (layout, pipeline_layout)
        })
    }

    /// A small buffer for the bindings a kernel does not read.
    fn dummy(&self) -> &wgpu::Buffer {
        self.dummy.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
        })
    }

    /// Another, for a written binding a kernel does not write (one buffer may not be bound written twice).
    fn dummy_rw(&self) -> &wgpu::Buffer {
        self.dummy_rw.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-dummy-rw"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
        })
    }

    /// A named kernel of eight buffers ([`Gpu::wide_layout`]).
    /// A scratch buffer of `bytes` (a power of two) from the pool, or a new one.
    pub(crate) fn pooled(&self, bytes: u64) -> wgpu::Buffer {
        let taken = {
            let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
            pool.iter().position(|(b, _)| *b == bytes).map(|i| pool.swap_remove(i).1)
        };
        taken.unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-chain-scratch"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
    }

    /// A staging buffer of `bytes` (a power of two) for a read-back: one a read before used, or a new one.
    pub(crate) fn staging(&self, bytes: u64) -> wgpu::Buffer {
        let taken = {
            let mut pool = self.staging.lock().unwrap_or_else(|p| p.into_inner());
            pool.iter().position(|(b, _)| *b == bytes).map(|i| pool.swap_remove(i).1)
        };
        taken.unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-chain-read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Staging buffers back (unmapped), to be used again; past [`STAGING_BYTES`] the largest are let go.
    pub(crate) fn unstage(&self, buffers: Vec<(u64, wgpu::Buffer)>) {
        let mut pool = self.staging.lock().unwrap_or_else(|p| p.into_inner());
        pool.extend(buffers);
        let mut total: u64 = pool.iter().map(|(b, _)| b).sum();
        if total > STAGING_BYTES {
            pool.sort_by_key(|(b, _)| *b);
            while total > STAGING_BYTES {
                let Some((b, _)) = pool.pop() else { break };
                total -= b;
            }
        }
    }

    /// Scratch buffers back to the pool, once what used them has run; past [`POOL_BYTES`] the largest are let go.
    pub(crate) fn unpool(&self, buffers: Vec<(u64, wgpu::Buffer)>) {
        let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
        pool.extend(buffers);
        let mut total: u64 = pool.iter().map(|(b, _)| b).sum();
        if total > POOL_BYTES {
            pool.sort_by_key(|(b, _)| *b);
            while total > POOL_BYTES {
                let Some((b, _)) = pool.pop() else { break };
                total -= b;
            }
        }
    }

    /// A pipeline's name, for a chain's profile.
    pub(crate) fn name_of(&self, pipeline: &Arc<wgpu::ComputePipeline>) -> &'static str {
        self.names.lock().unwrap_or_else(|p| p.into_inner()).get(&(Arc::as_ptr(pipeline) as usize)).copied().unwrap_or("other")
    }

    fn named_pipeline_wide(&self, name: &'static str, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let layout = &self.wide_layout().1;
        self.named_pipeline_in(name, layout, source)
    }

    fn named_pipeline_in(&self, name: &'static str, layout: &wgpu::PipelineLayout, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self.named.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(name) {
            return Arc::clone(p);
        }
        let module = self.shader(name, source());
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert(name, Arc::clone(&pipeline));
        self.names.lock().unwrap_or_else(|p| p.into_inner()).insert(Arc::as_ptr(&pipeline) as usize, name);
        pipeline
    }

    /// `dtype`'s kernel for `m` rows of `x`: the decode kernel for one, the one-row kernel for a few, the tiled one
    /// a prompt takes.
    fn pipeline(&self, dtype: GgmlType, m: usize) -> Option<Arc<wgpu::ComputePipeline>> {
        let kind = shaders::kind(dtype, m);
        let mut cache = self.pipelines.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(&(dtype, kind)) {
            return Some(Arc::clone(p));
        }
        let source = match kind {
            1 => shaders::source_many(dtype)?,
            2 => shaders::source_decode(dtype)?,
            3 => shaders::source_multi(dtype)?,
            _ => shaders::source(dtype)?,
        };
        let module = self.shader(&format!("oaiy-linear-q ({dtype:?}, kind {kind})"), source);
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oaiy-linear-q"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert((dtype, kind), Arc::clone(&pipeline));
        // a name for a chain's profile (a few, made once each)
        let name: &'static str = Box::leak(format!("matmul-{dtype:?}-{}", ["one", "tiled", "decode", "multi"][kind as usize]).into_boxed_str());
        self.names.lock().unwrap_or_else(|p| p.into_inner()).insert(Arc::as_ptr(&pipeline) as usize, name);
        Some(pipeline)
    }
}

/// Which adapter the backend runs on, for logs.
#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    /// Where the adapter sits on the PCI bus, as the API says (two cards of one model differ only here).
    pub pci_bus_id: String,
}

pub struct WgpuBackend {
    cpu: CpuBackend,
    gpu: Arc<Gpu>,
    budget: u64,
    used: Arc<AtomicU64>,
    summary: AdapterSummary,
    /// One projection at a time: the dispatch and read-back share the queue (EXL3 weights hold it too).
    serial: Arc<Mutex<()>>,
    /// The adapter it was opened on (its memory budget asked of it).
    raw_adapter: wgpu::Adapter,
}

/// Another handle on the same adapter (its device, its budget's count and its lock shared): what a model's parts keep
/// to record chains of their own (an expert group).
impl Clone for WgpuBackend {
    fn clone(&self) -> Self {
        Self { cpu: CpuBackend::new(), gpu: Arc::clone(&self.gpu), budget: self.budget, used: Arc::clone(&self.used), summary: self.summary.clone(), serial: Arc::clone(&self.serial), raw_adapter: self.raw_adapter.clone() }
    }
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBackend({} via {}, {} of {} GiB used)", self.summary.name, self.summary.backend,
            self.used.load(Ordering::Relaxed) / GIB, self.budget / GIB)
    }
}

impl WgpuBackend {
    /// A gated delta net's step on the GPU (the chain's conv and recurrence kernels, one submit), its recurrent state
    /// and conv window kept there between calls as tensors that are the GPU's own vectors (`DeviceChain::alias`): a
    /// state on the host (a new one, a restore) is taken up into one. None for a shape the kernels do not take; the
    /// caller runs the host's. Where a model calls the backend's step itself (Qwen3.8-Flash-Next, a Qwen3.5 prompt
    /// with an image), this takes the recurrence off the host: 7.8 ms a layer there for Flash-Next.
    #[allow(clippy::too_many_arguments)]
    fn delta_net_gpu(
        &self, mixed_qkv: &Tensor, z_in: &Tensor, beta_alpha: &Tensor, conv_weight: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor, ssm_norm: &Tensor,
        conv_state: &mut Tensor, state: &mut Tensor, d: ggml_rs::DeltaNet,
    ) -> Option<Tensor> {
        use ggml_rs::DeviceChain;
        let seq = d.rows;
        let kern = conv_weight.dim(conv_weight.rank() - 1);
        let ch = 2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim;
        let supported = d.k_dim == d.v_dim && [16, 32, 64, 128].contains(&d.k_dim) && d.k_heads > 0 && (2..=8).contains(&kern) && seq > 0;
        if !supported || mixed_qkv.numel() != seq * ch || z_in.numel() != seq * d.v_heads * d.v_dim || beta_alpha.numel() != seq * 2 * d.v_heads
            || conv_weight.numel() != ch * kern
        {
            return None;
        }
        // the state and the conv window as this adapter's vectors: the ones they alias, or the host's taken up
        let adopt = |t: &mut Tensor, shape: Vec<usize>| -> ggml_rs::DeviceVec {
            let len: usize = shape.iter().product();
            if let Some(v) = self.aliased(t).filter(|v| v.len == len) {
                return v;
            }
            let host = t.to_host();
            let v = self.vec(len);
            if host.numel() == len {
                DeviceChain::upload(self, &v, host.data());
            }
            *t = self.alias(&v, shape);
            v
        };
        let st = adopt(state, vec![d.v_heads, d.v_dim, d.k_dim]);
        let cv = adopt(conv_state, vec![kern - 1, ch]);
        let up = |t: &Tensor| {
            let h = if t.is_device() { t.to_host() } else { t.clone() };
            let v = self.vec(h.numel());
            DeviceChain::upload(self, &v, h.data());
            v
        };
        let (qkv, z, ba, cw, a, dt, nm) = (up(mixed_qkv), up(z_in), up(beta_alpha), up(conv_weight), up(ssm_a), up(dt_bias), up(ssm_norm));
        let (conv_out, out) = (self.vec(seq * ch), self.vec(seq * d.v_heads * d.v_dim));
        let mut rec = self.begin();
        // these vectors are this call's: no bind groups kept to hold them
        rec.keep_groups(false);
        rec.ssm_conv(&qkv, &cw, &cv, &conv_out, seq, ch, kern);
        rec.delta_net(&conv_out, &z, &ba, &a, &dt, &nm, &st, &out, d);
        rec.read(&out);
        let got = rec.finish().pop().expect("the delta net's output");
        Some(Tensor::from_vec(got, vec![seq, d.v_heads * d.v_dim]))
    }

    /// Open the best adapter wgpu finds, or the one `OAIY_WEBGPU_ADAPTER` names
    /// (part of its name, any case: "radeon", "arc", "5090", or of its PCI bus id:
    /// "03:00" of two cards alike), for a computer with more than one GPU. `budget_bytes` caps the weights placed on it (WebGPU
    /// cannot report free memory); `None` picks a default: a discrete card's
    /// memory less 4 GiB where Vulkan says how much it has (27.8 GiB of a 32 GB
    /// card), else 8 GiB; 2 GiB integrated, none for software.
    ///
    /// Vulkan, D3D12 and Metal only, unless `WGPU_BACKEND` names others: an
    /// instance with OpenGL too starts a WGL thread in NVIDIA's GL driver, and
    /// that thread's exit, as an instance drops, deadlocked on the loader lock
    /// against another thread opening Vulkan (one test run in about forty hung).
    pub fn new(budget_bytes: Option<u64>) -> Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let adapter = match std::env::var("OAIY_WEBGPU_ADAPTER").ok().map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()) {
            Some(wanted) => {
                let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
                let names: Vec<String> = adapters.iter().map(|a| { let i = a.get_info(); format!("{} ({:?})", i.name, i.backend) }).collect();
                adapters
                    .into_iter()
                    .find(|a| {
                        let i = a.get_info();
                        i.name.to_lowercase().contains(&wanted) || i.device_pci_bus_id.to_lowercase().contains(&wanted)
                    })
                    .ok_or_else(|| format!("OAIY_WEBGPU_ADAPTER={wanted}: no WebGPU adapter has that in its name or PCI bus id; there are {}", names.join(", ")))?
            }
            None => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .map_err(|e| format!("no WebGPU adapter: {e}"))?,
        };
        Self::open(adapter, budget_bytes)
    }

    /// The computer's `index`-th GPU (0 the first) as a CUDA device index counts them: the discrete ones on the API
    /// `new` picks, in their order on the PCI bus (nvidia-smi's, and CUDA's of cards alike), else every adapter on it;
    /// `OAIY_WEBGPU_ADAPTER`, where set, naming one instead (as for `new`). A worker told which GPU to use (an image
    /// model kept off a chat model's).
    pub fn nth(index: usize, budget_bytes: Option<u64>) -> Result<Self, String> {
        if std::env::var("OAIY_WEBGPU_ADAPTER").is_ok_and(|s| !s.trim().is_empty()) {
            return Self::new(budget_bytes);
        }
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let best = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let api = best.get_info().backend;
        let all: Vec<wgpu::Adapter> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY)).into_iter().filter(|a| a.get_info().backend == api).collect();
        let discrete = all.iter().any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
        let mut list: Vec<wgpu::Adapter> = all.into_iter().filter(|a| !discrete || a.get_info().device_type == wgpu::DeviceType::DiscreteGpu).collect();
        list.sort_by_key(|a| a.get_info().device_pci_bus_id);
        let count = list.len();
        let adapter = list.into_iter().nth(index).ok_or_else(|| format!("GPU {index}: the computer has {count} on {api:?}"))?;
        Self::open(adapter, budget_bytes)
    }

    /// The computer's other discrete GPUs on this one's API, each opened as a backend of its own (a budget each, as
    /// `new` picks one): two cards of one model differ only in where they sit on the PCI bus. For a model whose
    /// weights do not fit one card (Qwen3.8-Flash-Next's 46 GB of experts).
    pub fn others(&self, budget_bytes: Option<u64>) -> Vec<WgpuBackend> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY));
        let mut out = Vec::new();
        for adapter in adapters {
            let info = adapter.get_info();
            let same_api = format!("{:?}", info.backend) == self.summary.backend;
            if !same_api || info.device_type != wgpu::DeviceType::DiscreteGpu || info.device_pci_bus_id == self.summary.pci_bus_id {
                continue;
            }
            if let Ok(b) = Self::open(adapter, budget_bytes) {
                out.push(b);
            }
        }
        out
    }

    /// A backend on `adapter`.
    fn open(adapter: wgpu::Adapter, budget_bytes: Option<u64>) -> Result<Self, String> {
        let info = adapter.get_info();
        let summary = AdapterSummary {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            pci_bus_id: info.device_pci_bus_id.clone(),
        };
        let mut limits = adapter.limits();
        if let Some(v) = std::env::var_os("OAIY_PORTABLE_LIMITS") {
            // a workgroup's memory as given (bytes; else Apple's GPUs' 32,768, the least a native adapter has: WebGPU's
            // default, a browser's, is 16,384), and its invocations as WebGPU's defaults (256): other GPUs' limits (a
            // kernel past them refused, Gpu::shader), the rest the adapter's
            let d = wgpu::Limits::default();
            let bytes = v.to_str().and_then(|s| s.parse::<u32>().ok()).filter(|&b| b >= 1024).unwrap_or(32 << 10);
            limits.max_compute_workgroup_storage_size = limits.max_compute_workgroup_storage_size.min(bytes);
            limits.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup.min(d.max_compute_invocations_per_workgroup);
            limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.min(d.max_compute_workgroup_size_x);
            limits.max_compute_workgroup_size_y = limits.max_compute_workgroup_size_y.min(d.max_compute_workgroup_size_y);
            limits.max_compute_workgroup_size_z = limits.max_compute_workgroup_size_z.min(d.max_compute_workgroup_size_z);
        }
        let timestamps = if profile::chain_on() || profile::pieces_on() { adapter.features() & wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        // the tensor cores' matrices (Vulkan's cooperative matrices) and f16 in shaders, where the adapter has them: a
        // prompt's matmuls through them
        let coop = adapter.features() & (wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16);
        // (the kernels' fragments are 16 x 16 x 16, f16 into f32 sums and f16 ones: an adapter with only other shapes,
        // Metal's 8 x 8, has none of them)
        let shapes = adapter.cooperative_matrix_properties();
        let shape = |sums: wgpu::CooperativeScalarType| shapes.iter().any(|p| (p.m_size, p.n_size, p.k_size) == (16, 16, 16) && p.ab_type == wgpu::CooperativeScalarType::F16 && p.cr_type == sums);
        let fits = shape(wgpu::CooperativeScalarType::F32) && shape(wgpu::CooperativeScalarType::F16);
        let coop = if coop == wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16 && fits && std::env::var_os("OAIY_NO_COOP").is_none() { coop } else { wgpu::Features::empty() };
        // SAFETY: wgpu's cooperative matrices are an experimental feature (its implementation may misbehave where
        // misused); only the prompt kernels use them, each checked against the f32 kernels (OAIY_NO_COOP: none).
        let experimental = if coop.is_empty() { wgpu::ExperimentalFeatures::disabled() } else { unsafe { wgpu::ExperimentalFeatures::enabled() } };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("oaiy"),
            required_features: timestamps | coop,
            required_limits: limits.clone(),
            experimental_features: experimental,
            ..Default::default()
        }))
        .map_err(|e| format!("WebGPU device on {}: {e}", info.name))?;
        // its loss noted (a TDR, a driver's reset): a wait for it fails then
        let lost = Arc::new(Mutex::new(None));
        {
            let lost = Arc::clone(&lost);
            device.set_device_lost_callback(move |reason, message| {
                *lost.lock().unwrap_or_else(|p| p.into_inner()) = Some(format!("{reason:?}: {message}"));
            });
        }
        {
            // (an error once it is lost said to be that: its new buffers are invalid, the first use of one the error)
            let lost = Arc::clone(&lost);
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                if let Some(why) = lost.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                    panic!("webgpu: the device was lost ({why}): {e}");
                }
                panic!("wgpu error: {e}");
            }));
        }
        let watch = Arc::new(Watch::default());
        {
            let (w, l, name) = (Arc::downgrade(&watch), Arc::downgrade(&lost), info.name.clone());
            std::thread::Builder::new()
                .name("oaiy-webgpu-watchdog".into())
                .spawn(move || watchdog(w, l, name))
                .map_err(|e| format!("WebGPU device on {}: its watchdog: {e}", info.name))?;
        }
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oaiy-linear-q"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(
                    3,
                    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oaiy-linear-q"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let budget = budget_bytes.unwrap_or(match info.device_type {
            wgpu::DeviceType::DiscreteGpu => device_memory(&adapter).map_or(8 * GIB, discrete_budget),
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => 2 * GIB,
            _ => 0,
        });
        Ok(Self {
            cpu: CpuBackend::new(),
            gpu: Arc::new(Gpu { device, queue_raw: queue, lost, watch, feed: Arc::new(Feed::default()), in_flight_limit: Arc::new(std::sync::atomic::AtomicUsize::new(0)), layout, pipeline_layout, pipelines: Mutex::new(HashMap::new()), exl3: Mutex::new([None, None]), named: Mutex::new(HashMap::new()), names: Mutex::new(HashMap::new()), pool: Mutex::new(Vec::new()), staging: Mutex::new(Vec::new()), chain_groups: Mutex::new(HashMap::new()), wide: std::sync::OnceLock::new(), chain_groups_wide: Mutex::new(HashMap::new()), dummy: std::sync::OnceLock::new(), dummy_rw: std::sync::OnceLock::new(), limits, staged: AtomicU64::new(0), few: std::sync::OnceLock::new(), moe_steps: Mutex::new(Vec::new()), coop_units: std::sync::OnceLock::new() }),
            budget,
            used: Arc::new(AtomicU64::new(0)),
            summary,
            serial: Arc::new(Mutex::new(())),
            raw_adapter: adapter,
        })
    }

    pub fn adapter(&self) -> &AdapterSummary {
        &self.summary
    }

    /// How fast the host's bytes go up to the card and come back (GB/s each, timed on 64 MB three times): a card on
    /// fewer PCIe lanes is the slower (Qwen3.8 27B's prompts over two cards put the one that moves more on the faster).
    pub fn transfer_rates(&self) -> (f64, f64) {
        let len = 64usize << 20;
        let data = vec![0u8; len];
        let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-transfer-probe"),
            size: len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let up = || {
            self.gpu.queue().write_buffer(&buf, 0, &data);
            let i = self.gpu.queue().submit([]);
            self.gpu.wait(Some(i));
        };
        up();
        let _ = self.gpu.read(&buf, len as u64);
        let t = std::time::Instant::now();
        for _ in 0..3 {
            up();
        }
        let up_rate = 3.0 * len as f64 / t.elapsed().as_secs_f64() / 1e9;
        let t = std::time::Instant::now();
        for _ in 0..3 {
            let _ = self.gpu.read(&buf, len as u64);
        }
        (up_rate, 3.0 * len as f64 / t.elapsed().as_secs_f64() / 1e9)
    }

    /// The OS's budget for this process on the card's memory, and its use of it now (Vulkan's): None where the API
    /// does not say.
    /// Wait for the GPU's work and let go of the buffers dropped since (wgpu frees them at a poll or submit: a stage's
    /// vectors dropped before the next stage makes its own would otherwise share the card with them).
    pub fn settle(&self) {
        self.gpu.wait(None);
    }

    /// Let go of what the device keeps between chains' runs (the kept bind groups and the vectors they hold, the
    /// scratch and staging pools), then [`Self::settle`]: the room for another model's work on the same card (a
    /// diffusion step's kept some 10 GB past its weights).
    pub fn release_cached(&self) {
        self.gpu.chain_groups.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.gpu.chain_groups_wide.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.gpu.pool.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.gpu.staging.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.settle();
    }

    /// Whether the adapter has cooperative matrices (the tensor cores' kernels: f16, NVFP4 and the K-quants' and Q8_0's
    /// for a prompt's rows).
    pub fn tensor_cores(&self) -> bool {
        self.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    }

    /// The most bytes one vector an op reads or writes may hold (the adapter's storage binding limit: 2 GB here).
    pub fn max_binding(&self) -> u64 {
        (self.gpu.limits.max_storage_buffer_binding_size as u64).min(self.gpu.limits.max_buffer_size)
    }

    pub fn memory_budget(&self) -> Option<(u64, u64)> {
        heap_budget(&self.raw_adapter)
    }

    /// At most `pieces` of the chains' pieces on this device's queue at once from here on (0: as many as are recorded,
    /// as it is until asked), each encoded just before it is submitted ([`Feed`]): for a loop of short steps, or a
    /// prompt's chunks, that runs the GPU for seconds on end. A recording's pieces otherwise go to the GPU as they are
    /// recorded, a step's dozen queued within its first milliseconds; twelve such steps a second (MiniMax Music 3's
    /// transformer: 800 dispatches a step, 82 ms), or a prompt's chunks one after another (the 27B's 512 tokens in
    /// 0.2 s), and an RTX 5090 under a power limit (402 W of its 575) fell, after some 5 to 7 s and then every few, into
    /// seconds of running a tenth as fast (a matmul over large weights 12 to 17 times as long, an attention 2 to 3: its
    /// memory busy three times as much of the time, its clock no lower and its limiter less at work than before),
    /// where so fed it does so once, for a second or two, and settles (Candle's CUDA, a kernel at a time, never
    /// does). The pieces' number on the queue is the lesser part of it: with one, two or three on the queue but each
    /// encoded when it was recorded, the card fell so as before; two encoded at their turn cost a step or a chunk
    /// nothing, three some 6% of a long prompt's chunk. Not for long dispatches (Pixal3D's flows over 15,000 voxels,
    /// 1.2 s a pass: steady as they are, and with a limit the throttle found them now and then).
    pub fn pieces_in_flight_at_most(&self, pieces: usize) {
        self.gpu.in_flight_limit.store(pieces, Ordering::Relaxed);
    }

    /// Bytes of weights placed on the GPU, and the budget.
    pub fn usage(&self) -> (u64, u64) {
        (self.used.load(Ordering::Relaxed), self.budget)
    }

    /// [`ggml_rs::DeviceChain::copy_weight`]: `w`'s buffers as another adapter holds them (blocks padded alike) read
    /// back and put on this one, within its budget.
    pub(crate) fn copy_weight(&self, w: &QuantizedTensor) -> Option<QuantizedTensor> {
        let q = w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?;
        if Arc::ptr_eq(&q.gpu, &self.gpu) {
            return None;
        }
        let prev = self.used.fetch_add(q.nbytes as u64, Ordering::Relaxed);
        if prev + q.nbytes as u64 > self.budget {
            self.used.fetch_sub(q.nbytes as u64, Ordering::Relaxed);
            return None;
        }
        let chunks = self.gpu.upload_rows(&q.gpu_bytes(), q.row_bytes, 1);
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype: q.dtype, chunks, row_bytes: q.row_bytes, nbytes: q.nbytes, used: Arc::clone(&self.used) };
        Some(QuantizedTensor::from_device(Box::new(storage), w.shape().to_vec()))
    }

    /// Whether `dtype` has a GPU kernel.
    pub fn supports(dtype: GgmlType) -> bool {
        shaders::layout(dtype).is_some()
    }

    fn upload(&self, w: QuantizedTensor) -> QuantizedTensor {
        let Some((elems, block_bytes, _)) = shaders::layout(w.dtype()) else { return w };
        if w.is_device() || w.rank() != 2 || w.dim(1) % elems as usize != 0 || self.gpu.pipeline(w.dtype(), 2).is_none() {
            return w;
        }
        let row_bytes = w.dim(1) / elems as usize * block_bytes as usize;
        // ggml's bytes, and the GPU's: Q3_K's blocks padded (`shaders::padded_block`)
        let padded = shaders::padded_block(w.dtype());
        let host_row = padded.map_or(row_bytes, |(host, gpu, _)| row_bytes / gpu * host);
        if w.bytes().len() != host_row * w.dim(0) || row_bytes as u64 > chunk_limit(&self.gpu.limits) {
            return w;
        }
        let nbytes = row_bytes * w.dim(0);
        // Reserve before uploading so concurrent loads cannot overshoot together.
        let prev = self.used.fetch_add(nbytes as u64, Ordering::Relaxed);
        if prev + nbytes as u64 > self.budget {
            self.used.fetch_sub(nbytes as u64, Ordering::Relaxed);
            return w;
        }
        let chunks = match padded {
            Some((host, gpu, at)) => self.gpu.upload_rows(&shaders::pad_blocks(w.bytes(), host, gpu, at), row_bytes, 1),
            None => self.gpu.upload_rows(w.bytes(), row_bytes, 1),
        };
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype: w.dtype(), chunks, row_bytes, nbytes, used: Arc::clone(&self.used) };
        QuantizedTensor::from_device(Box::new(storage), w.shape().to_vec())
    }

    /// `y = x · Wᵀ` with `W` on the GPU.
    fn linear_gpu(&self, x: &Tensor, q: &WgpuQuant, shape: &[usize]) -> Tensor {
        let start = std::time::Instant::now();
        let y = self.linear_gpu_inner(x, q, shape);
        profile::add(&profile::LINEAR, start);
        y
    }

    fn linear_gpu_inner(&self, x: &Tensor, q: &WgpuQuant, shape: &[usize]) -> Tensor {
        let (n, k) = (shape[0], shape[1]);
        let host;
        let x = if x.is_device() {
            host = x.to_host();
            &host
        } else {
            x
        };
        let m = x.numel() / k;
        let mut out_shape = x.shape().to_vec();
        *out_shape.last_mut().expect("x has a last axis") = n;
        if m == 0 || n == 0 {
            return Tensor::from_vec(Vec::new(), out_shape);
        }
        // x and y are single bindings: a long prompt's rows go in batches that
        // stay under the adapter's binding and buffer limits.
        let limit = chunk_limit(&self.gpu.limits) as usize;
        let rows = (limit / (4 * k.max(n))).max(1);
        if m > rows {
            let data = x.data();
            let mut y = Vec::with_capacity(m * n);
            for start in (0..m).step_by(rows) {
                let count = rows.min(m - start);
                let part = Tensor::from_vec(data[start * k..(start + count) * k].to_vec(), vec![count, k]);
                y.extend_from_slice(self.linear_gpu_inner(&part, q, shape).data());
            }
            return Tensor::from_vec(y, out_shape);
        }
        self.linear_gpu_batch(x, &[(q, shape)]).pop().expect("one weight, one result")
    }

    /// Several weights of one input, `x` (`[m, k]` on the host, few enough rows for one binding), in one submit and
    /// one read back: each weight's `[m, n]`, in order. The input goes up once.
    fn linear_gpu_batch(&self, x: &Tensor, ws: &[(&WgpuQuant, &[usize])]) -> Vec<Tensor> {
        let k = ws[0].1[1];
        let m = x.numel() / k;
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let gpu = &self.gpu;
        let bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for f in v {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out
        };
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let xbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-x"), size: (x.numel() * 4) as u64, usage, mapped_at_creation: false });
        gpu.queue().write_buffer(&xbuf, 0, &bytes(x.data()));
        let sizes: Vec<u64> = ws.iter().map(|(_, shape)| (m * shape[0] * 4) as u64).collect();
        let total: u64 = sizes.iter().sum();
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-y-read"),
            size: total,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut ybufs = Vec::with_capacity(ws.len());
        for ((q, shape), &ysize) in ws.iter().zip(&sizes) {
            let n = shape[0];
            let pipeline = gpu.pipeline(q.dtype, m).expect("uploaded weights have a pipeline");
            let ybuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-y"),
                size: ysize,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut groups = Vec::new();
            for (buffer, row0, rows) in &q.chunks {
                let params: Vec<u8> = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let pbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("oaiy-params"),
                    size: params.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue().write_buffer(&pbuf, 0, &params);
                let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("oaiy-linear-q"),
                    layout: &gpu.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: ybuf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: pbuf.as_entire_binding() },
                    ],
                });
                groups.push((group, *rows));
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                for (group, rows) in &groups {
                    pass.set_bind_group(0, group, &[]);
                    let (gx, gy, gz) = shaders::grid(q.dtype, m, *rows);
                    pass.dispatch_workgroups(gx, gy, gz);
                }
            }
            ybufs.push(ybuf);
        }
        let mut at = 0;
        for (ybuf, &ysize) in ybufs.iter().zip(&sizes) {
            enc.copy_buffer_to_buffer(ybuf, 0, &staging, at, ysize);
            at += ysize;
        }
        let waiting = std::time::Instant::now();
        gpu.queue().submit([enc.finish()]);
        let slice = staging.slice(..total);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.wait(None);
        profile::add(&profile::LINEAR_WAIT, waiting);
        // each output made straight from the mapped bytes (a copy of them first was a second pass over a prompt's
        // outputs, up to 131 MB a call)
        let raw = slice.get_mapped_range().expect("webgpu: mapping a finished buffer");
        let mut at = 0usize;
        let out = ws
            .iter()
            .zip(&sizes)
            .map(|((_, shape), &ysize)| {
                let y: Vec<f32> = raw[at..at + ysize as usize].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                at += ysize as usize;
                let mut out_shape = x.shape().to_vec();
                *out_shape.last_mut().expect("x has a last axis") = shape[0];
                Tensor::from_vec(y, out_shape)
            })
            .collect();
        drop(raw);
        staging.unmap();
        out
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn delta_net_step(
        &self, mixed_qkv: &Tensor, z_in: &Tensor, beta_alpha: &Tensor, conv_weight: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor, ssm_norm: &Tensor,
        conv_state: &mut Tensor, state: &mut Tensor, seq: usize, num_v_heads: usize, num_k_heads: usize, head_v_dim: usize, head_k_dim: usize,
        v_per_k: usize, scale_q: f32, eps: f32,
    ) -> Tensor {
        let d = ggml_rs::DeltaNet { rows: seq, v_heads: num_v_heads, k_heads: num_k_heads, k_dim: head_k_dim, v_dim: head_v_dim, scale_q, eps, sigmoid_gate: false };
        match self.delta_net_gpu(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, d) {
            Some(out) => out,
            None => self.cpu.delta_net_step(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps),
        }
    }
    fn delta_net_step_sigmoid(
        &self, mixed_qkv: &Tensor, z_in: &Tensor, beta_alpha: &Tensor, conv_weight: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor, ssm_norm: &Tensor,
        conv_state: &mut Tensor, state: &mut Tensor, seq: usize, num_v_heads: usize, num_k_heads: usize, head_v_dim: usize, head_k_dim: usize,
        v_per_k: usize, scale_q: f32, eps: f32,
    ) -> Tensor {
        let d = ggml_rs::DeltaNet { rows: seq, v_heads: num_v_heads, k_heads: num_k_heads, k_dim: head_k_dim, v_dim: head_v_dim, scale_q, eps, sigmoid_gate: true };
        match self.delta_net_gpu(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, d) {
            Some(out) => out,
            None => self.cpu.delta_net_step_sigmoid(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps),
        }
    }
    fn vram_status(&self) -> Option<(usize, usize)> {
        let (used, budget) = self.usage();
        Some((budget.saturating_sub(used) as usize, budget as usize))
    }
    fn to_device_quant(&self, w: QuantizedTensor) -> QuantizedTensor {
        self.upload(w)
    }
    fn try_to_device_quant(&self, w: QuantizedTensor, _safety_margin_bytes: usize) -> QuantizedTensor {
        // The budget is the whole weight allowance; activations live on the host.
        self.upload(w)
    }
    fn linear_q(&self, x: &Tensor, w: &QuantizedTensor) -> Tensor {
        if let Some(q) = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()) {
            return self.linear_gpu(x, q, w.shape());
        }
        self.cpu.linear_q(x, w)
    }
    /// Weights of one input that are all on this adapter, in one submit: a layer's q, k and v were three round trips
    /// a decode step.
    fn linear_q_many(&self, x: &Tensor, ws: &[&QuantizedTensor]) -> Vec<Tensor> {
        let on_gpu: Option<Vec<(&WgpuQuant, &[usize])>> =
            ws.iter().map(|w| Some((w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?, w.shape()))).collect();
        let host;
        let x = if x.is_device() {
            host = x.to_host();
            &host
        } else {
            x
        };
        match on_gpu {
            Some(on_gpu) if on_gpu.len() > 1 && !x.data().is_empty() => {
                let k = on_gpu[0].1[1];
                let m = x.numel() / k;
                let widest = on_gpu.iter().map(|(_, s)| s[0]).max().unwrap_or(0);
                let fits = m <= (chunk_limit(&self.gpu.limits) as usize / (4 * k.max(widest))).max(1);
                if on_gpu.iter().all(|(_, s)| s[1] == k && s[0] > 0) && fits {
                    let start = std::time::Instant::now();
                    let ys = self.linear_gpu_batch(x, &on_gpu);
                    profile::add(&profile::LINEAR, start);
                    return ys;
                }
                ws.iter().map(|w| self.linear_q(x, w)).collect()
            }
            _ => ws.iter().map(|w| self.linear_q(x, w)).collect(),
        }
    }

    // Everything else is CpuBackend's, forwarded explicitly so its optimized
    // overrides are kept rather than the trait's defaults.
    fn embed_lookup(&self, table: &Tensor, tokens: &[u32], embedding_dim: usize) -> Tensor {
        self.cpu.embed_lookup(table, tokens, embedding_dim)
    }
    fn linear(&self, x: &Tensor, w: &Tensor) -> Tensor {
        self.cpu.linear(x, w)
    }
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        self.cpu.rmsnorm(x, weight, eps)
    }
    fn silu_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        self.cpu.silu_mul_split(fused, ff)
    }
    fn chain(&self) -> Option<&dyn ggml_rs::chain::DeviceChain> {
        Some(self)
    }
    fn gelu_approx_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        self.cpu.gelu_approx_mul_split(fused, ff)
    }
    fn softmax_last(&self, x: &mut Tensor) {
        self.cpu.softmax_last(x)
    }
    fn silu(&self, x: &Tensor) -> Tensor {
        self.cpu.silu(x)
    }
    fn gelu_approx(&self, x: &Tensor) -> Tensor {
        self.cpu.gelu_approx(x)
    }
    fn add_inplace(&self, x: &mut Tensor, y: &Tensor) {
        self.cpu.add_inplace(x, y)
    }
    fn mul_inplace(&self, x: &mut Tensor, y: &Tensor) {
        self.cpu.mul_inplace(x, y)
    }
    fn rope(&self, x: &mut Tensor, positions: &[u32], head_dim: usize, rope_type: RopeType, theta: f32, freq_factors: Option<&[f32]>) {
        self.cpu.rope(x, positions, head_dim, rope_type, theta, freq_factors)
    }
    fn repeat_kv(&self, x: &Tensor, n_rep: usize) -> Tensor {
        self.cpu.repeat_kv(x, n_rep)
    }
    fn bmm_qkt(&self, q: &Tensor, k: &Tensor, scale: f32, past: usize) -> Tensor {
        self.cpu.bmm_qkt(q, k, scale, past)
    }
    fn bmm_av(&self, scores: &Tensor, v: &Tensor) -> Tensor {
        self.cpu.bmm_av(scores, v)
    }
    /// The CPU's fused attention (the cache is on the host): the default would copy the cache and take the softmax on
    /// one thread.
    fn attention(&self, q: &Tensor, k_buffer: &Tensor, v_buffer: &Tensor, kv_len: usize, scale: f32, past: usize, sliding_window: Option<usize>) -> Tensor {
        let start = std::time::Instant::now();
        let host = |t: &Tensor| if t.is_device() { t.to_host() } else { t.clone() };
        let out = if q.is_device() || k_buffer.is_device() || v_buffer.is_device() {
            self.cpu.attention(&host(q), &host(k_buffer), &host(v_buffer), kv_len, scale, past, sliding_window)
        } else {
            self.cpu.attention(q, k_buffer, v_buffer, kv_len, scale, past, sliding_window)
        };
        profile::add(&profile::ATTENTION, start);
        out
    }
    fn argmax_last(&self, x: &Tensor) -> Vec<u32> {
        self.cpu.argmax_last(x)
    }
}

#[cfg(test)]
mod tests;

/// `(available, total)` bytes of the computer's memory, for a build without CUDA: what the portable engine sizes its
/// expert cache from (a streamed MoE model such as GLM-5.3-Flash), as the CUDA build does from
/// `ggml_rs_cuda::host_memory`. Without it the portable build assumed 32 GB on any computer, so a model of 190 GB read
/// most of each token's experts from the disk on a computer with 192 GB.
///
/// The same reader as `ggml_rs_cuda::host_memory`, here because this is the crate every portable build links and
/// `ggml-rs` and `llama-rs` keep no `unsafe` at all: on Windows it is one documented call. `None` on a platform it has
/// no reader for, so a caller keeps a fallback.
pub fn host_memory() -> Option<(usize, usize)> {
    host_memory_impl()
}

#[cfg(windows)]
fn host_memory_impl() -> Option<(usize, usize)> {
    // MEMORYSTATUSEX, exactly as documented: `length` must be set by the caller and every other field is written by
    // the call.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `s` is a fully initialised MEMORYSTATUSEX of the size its own `length` field declares, and the call only
    // writes into it. The pointer is valid for the duration of the call and nothing retains it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
    (ok != 0).then_some((s.avail_phys as usize, s.total_phys as usize))
}

#[cfg(target_os = "linux")]
fn host_memory_impl() -> Option<(usize, usize)> {
    // /proc/meminfo reports both, in kB. MemAvailable is the kernel's own estimate of what a new allocation can have;
    // MemFree undercounts badly because of the page cache.
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<usize> {
        text.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok()).map(|kb| kb * 1024)
    };
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn host_memory_impl() -> Option<(usize, usize)> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    /// Whatever the platform, the answer must be self-consistent or absent.
    #[test]
    fn host_memory_is_plausible_or_absent() {
        if let Some((free, total)) = super::host_memory() {
            assert!(total > (1 << 30), "total {total} is implausibly small");
            assert!(free <= total, "free {free} exceeds total {total}");
        }
    }
}
