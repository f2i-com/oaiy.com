//! Worker threads that stay: a job is handed to one that is already there, where a scope's threads are started and
//! joined at every call (some 60 us each on Windows). A decode step's parallel work is a fraction of a millisecond a
//! layer, forty layers a token: the routed experts on the CPU ([`crate::cpu_experts`], two phases a layer) and the
//! sparse attention's heads ([`crate::attention`]).
//!
//! Jobs own what they read (through `Arc`s), so the workers need no lifetime tricks and the crate no `unsafe`.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

/// A job: its number among its call's, and what it made.
pub(crate) type Job = Box<dyn FnOnce() -> (usize, Vec<f32>) + Send>;

/// A fixed set of worker threads fed round-robin, results tagged by job index.
pub(crate) struct Pool {
    senders: Vec<Sender<Job>>,
    results: Receiver<(usize, Vec<f32>)>,
    handles: Vec<JoinHandle<()>>,
}

impl Pool {
    /// `threads` workers, named `{name}-{i}`.
    pub(crate) fn new(threads: usize, name: &str) -> Pool {
        let (res_tx, results) = channel::<(usize, Vec<f32>)>();
        let mut senders = Vec::with_capacity(threads);
        let mut handles = Vec::with_capacity(threads);
        for i in 0..threads {
            let (tx, rx) = channel::<Job>();
            let res_tx = res_tx.clone();
            let h = std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .spawn(move || {
                    while let Some(job) = recv_spinning(&rx) {
                        if res_tx.send(job()).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawn a worker");
            senders.push(tx);
            handles.push(h);
        }
        Pool { senders, results, handles }
    }

    pub(crate) fn threads(&self) -> usize {
        self.senders.len()
    }

    /// Run every job, returning their outputs in job order.
    pub(crate) fn run(&self, jobs: Vec<Job>) -> Vec<Vec<f32>> {
        let n = jobs.len();
        for (i, job) in jobs.into_iter().enumerate() {
            self.senders[i % self.senders.len()].send(job).expect("a worker alive");
        }
        let mut out = vec![Vec::new(); n];
        for _ in 0..n {
            let (i, v) = recv_spinning(&self.results).expect("a worker alive");
            out[i] = v;
        }
        out
    }
}

/// How long a thread polls for its next message before parking. A job's two
/// phases follow each other within microseconds and layers within about a
/// millisecond; waking a parked thread costs tens of microseconds on Windows,
/// twice per job, which is a fifth of a one-expert job.
const SPIN: std::time::Duration = std::time::Duration::from_micros(100);

/// `rx.recv()`, polling for [`SPIN`] first. `None` when the channel is closed.
fn recv_spinning<T>(rx: &Receiver<T>) -> Option<T> {
    let t0 = std::time::Instant::now();
    loop {
        match rx.try_recv() {
            Ok(v) => return Some(v),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
            Err(std::sync::mpsc::TryRecvError::Empty) if t0.elapsed() < SPIN => std::hint::spin_loop(),
            Err(std::sync::mpsc::TryRecvError::Empty) => return rx.recv().ok(),
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.senders.clear(); // closes every job channel; workers leave their loops
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// The most workers [`shared`]'s pool has: a decode step's heads are a couple of milliseconds of one thread's work.
const SHARED_THREADS: usize = 16;

/// A pool for a model's small parallel work between its experts (the attention's heads), made at its first use and
/// kept for the process: one call at a time (the lock is its caller's turn).
pub(crate) fn shared() -> &'static Mutex<Pool> {
    static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(SHARED_THREADS);
        Mutex::new(Pool::new(threads, "dsv41-worker"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_come_back_in_their_order_whatever_worker_ran_them() {
        let pool = Pool::new(3, "dsv41-test");
        for _ in 0..3 {
            let jobs: Vec<Job> = (0..10).map(|i| Box::new(move || (i, vec![i as f32; i + 1])) as Job).collect();
            let out = pool.run(jobs);
            assert_eq!(out.len(), 10);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(v, &vec![i as f32; i + 1]);
            }
        }
        assert!(shared().lock().unwrap().threads() >= 1);
    }
}
