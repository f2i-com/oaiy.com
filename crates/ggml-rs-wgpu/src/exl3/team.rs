//! Worker threads that stay, for a decode step's experts on the host ([`super::step_experts_host`]): a step's work is a
//! fraction of a millisecond a layer, forty layers a token, and rayon's threads (and the cores under them) slept
//! between layers, so a lone expert's 3.5 ms of one thread's work took 0.95 ms of 16 cores, where the same cores kept
//! busy give it 0.3. These spin between jobs for a while (the next layer's comes within a few milliseconds) and sleep
//! after it.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// How long a worker spins for the next job before it sleeps: a decode step's layers come some 2.5 ms apart
/// (OAIY_STEP_SPIN_US another time).
fn spin() -> std::time::Duration {
    static SPIN: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| std::time::Duration::from_micros(std::env::var("OAIY_STEP_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(5000)))
}

/// A job: `f(i)` for every `i` below `n`. `f` is the caller's, alive until the call returns, which is after every
/// task it hands out has run.
#[derive(Clone, Copy)]
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    generation: u64,
}

// SAFETY: `f` points to a `Sync` closure, called from the workers only while the caller that owns it waits for them
// ([`Team::run`]); the pointer itself is plain data.
unsafe impl Send for Job {}

struct Shared {
    /// The job's number: a worker takes up a job when it changes.
    generation: AtomicU64,
    job: Mutex<Option<Job>>,
    /// The next task: the job's number (its low 32 bits) above the task's index, so a worker still at the last job
    /// claims nothing of this one.
    next: AtomicU64,
    /// Tasks claimed and not yet done, and those not yet claimed: the caller waits for none.
    pending: AtomicUsize,
    sleep: Mutex<()>,
    wake: Condvar,
}

pub(super) struct Team {
    shared: Arc<Shared>,
    /// One call at a time.
    turn: Mutex<()>,
}

impl Team {
    /// `threads` workers beside the caller (which works too).
    fn new(threads: usize) -> Team {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            job: Mutex::new(None),
            next: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        for i in 0..threads {
            let shared = Arc::clone(&shared);
            // (a worker that cannot be started leaves its share to the others and the caller)
            let _ = std::thread::Builder::new().name(format!("oaiy-step-expert-{i}")).spawn(move || work(&shared));
        }
        Team { shared, turn: Mutex::new(()) }
    }

    /// `f(i)` for every `i` below `n`, on the workers and this thread, each once; returns once every one has run.
    pub(super) fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        let _turn = self.turn.lock().unwrap_or_else(|p| p.into_inner());
        let s = &*self.shared;
        let generation = s.generation.load(Ordering::Relaxed) + 1;
        // SAFETY: the pointer's lifetime is erased to hand it to the workers; this call does not return until every
        // task has run (`pending` back to 0), and after that no worker calls it: each claims a task by its number and
        // an index below `n`, and none is left.
        let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        s.pending.store(n, Ordering::Relaxed);
        s.next.store((generation & 0xffff_ffff) << 32, Ordering::Relaxed);
        *s.job.lock().unwrap_or_else(|p| p.into_inner()) = Some(Job { f, n, generation });
        s.generation.store(generation, Ordering::Release);
        {
            let _g = s.sleep.lock().unwrap_or_else(|p| p.into_inner());
            s.wake.notify_all();
        }
        take_tasks(s, Job { f, n, generation });
        while s.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        *s.job.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// Claim and run `job`'s tasks while there are any.
fn take_tasks(s: &Shared, job: Job) {
    let tag = (job.generation & 0xffff_ffff) << 32;
    loop {
        let v = s.next.load(Ordering::Acquire);
        if v & !0xffff_ffff != tag || (v & 0xffff_ffff) as usize >= job.n {
            return;
        }
        if s.next.compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            continue;
        }
        // SAFETY: a task of this job was claimed, so its caller is still waiting for it ([`Team::run`]).
        unsafe { (*job.f)((v & 0xffff_ffff) as usize) };
        s.pending.fetch_sub(1, Ordering::Release);
    }
}

/// A worker: each new job's tasks taken as they come, spinning for [`spin`] between jobs, then sleeping.
fn work(s: &Shared) {
    let mut seen = 0u64;
    let spin = spin();
    loop {
        let idle = std::time::Instant::now();
        loop {
            if s.generation.load(Ordering::Acquire) != seen {
                break;
            }
            if idle.elapsed() < spin {
                std::hint::spin_loop();
                continue;
            }
            let g = s.sleep.lock().unwrap_or_else(|p| p.into_inner());
            if s.generation.load(Ordering::Acquire) == seen {
                drop(s.wake.wait(g).unwrap_or_else(|p| p.into_inner()));
            }
        }
        seen = s.generation.load(Ordering::Acquire);
        let job = *s.job.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(job) = job.filter(|j| j.generation == seen) {
            take_tasks(s, job);
        }
    }
}

/// The team a process's steps share: as many workers as the host has cores beside the caller (logical processors
/// halved: a second thread on a core shares its vectors), OAIY_STEP_THREADS another number.
pub(super) fn team() -> &'static Team {
    static TEAM: std::sync::OnceLock<Team> = std::sync::OnceLock::new();
    TEAM.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get()).div_ceil(2);
        let threads = std::env::var("OAIY_STEP_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(cores);
        Team::new(threads.saturating_sub(1))
    })
}
