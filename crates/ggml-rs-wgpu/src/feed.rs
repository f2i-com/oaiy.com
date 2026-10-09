//! A device's queue: the pieces of work recordings hand it, and the thread that encodes and submits them in turn.

use super::*;

/// OAIY_PIECES_IN_FLIGHT's, where it is set: every device's pieces in flight at most that many, whatever was asked
/// of it ([`WgpuBackend::pieces_in_flight_at_most`]; 0: as many as are recorded).
pub(super) fn pieces_in_flight_asked() -> Option<usize> {
    static ASKED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| std::env::var("OAIY_PIECES_IN_FLIGHT").ok().and_then(|v| v.parse().ok()))
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
pub(super) enum Fare {
    Dispatches(Vec<Dispatch>),
    Commands(Vec<wgpu::CommandBuffer>),
    Write(wgpu::Buffer, u64, Vec<u8>),
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
pub(super) struct Feed {
    pub(super) state: Mutex<Fed>,
    pub(super) changed: std::sync::Condvar,
}

#[derive(Default)]
pub(super) struct Fed {
    /// What was handed over and is not yet submitted, in turn, each with where its submission is wanted.
    pub(super) waiting: std::collections::VecDeque<(Fare, Arc<Mutex<Option<wgpu::SubmissionIndex>>>)>,
    /// How many were handed over, and how many of them are submitted.
    pub(super) handed: u64,
    pub(super) gone: u64,
    /// The device's thread runs (from the first piece handed over), and is to end (the device dropped).
    pub(super) fed: bool,
    pub(super) closed: bool,
    /// Why the thread stopped, if it did: a lost device, or an error of the queue's.
    pub(super) failed: Option<String>,
}

/// Waits for `device`'s submission `index` (else everything submitted), a second at a time: Err once its loss has been
/// said, or the wait itself fails.
pub(super) fn wait_for(device: &wgpu::Device, watch: &Watch, lost: &Mutex<Option<String>>, index: Option<wgpu::SubmissionIndex>) -> Result<(), String> {
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
pub(super) fn encode(device: &wgpu::Device, dispatches: &[Dispatch]) -> wgpu::CommandBuffer {
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

/// OAIY_PIPELINE_LOG: a kernel's pipeline said as it is made (its shader read and its pipeline built since `began`),
/// with the process's time: which kernels a server's first request makes, and for how long.
pub(super) fn made(name: &str, began: std::time::Instant) {
    static ON: std::sync::OnceLock<Option<std::time::Instant>> = std::sync::OnceLock::new();
    if let Some(start) = ON.get_or_init(|| std::env::var_os("OAIY_PIPELINE_LOG").map(|_| std::time::Instant::now())) {
        eprintln!("    pipeline {name}: {:.1} ms, at {:.2} s", began.elapsed().as_secs_f64() * 1e3, start.elapsed().as_secs_f64());
    }
}

/// A device's [`Feed`]'s thread: what was handed over submitted in turn (a piece's dispatches encoded then), each
/// once the device's pieces in flight are fewer than `limit` (as it is then; 0: at once). It ends with the device, or
/// at a failure (said to whoever waits on the feed next).
pub(super) fn feed_pieces(feed: Arc<Feed>, device: wgpu::Device, queue: wgpu::Queue, watch: Arc<Watch>, lost: Arc<Mutex<Option<String>>>, limit: Arc<std::sync::atomic::AtomicUsize>) {
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
