//! Where a forward pass's time goes: counters any thread adds to (an `Instant` and an atomic add a region, so they
//! stay on), read and reset by whoever measures (the timing tests, a serving log). Wall time of each region, so the
//! parts that run on several threads at once are counted once.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A region of the forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Waiting for a layer's routed experts' records (the cache, or the drive).
    ExpertRead,
    /// Routed experts' matmuls on a GPU (`moe::Experts::gpu`).
    ExpertGpu,
    /// Routed experts' matmuls on the CPU.
    ExpertCpu,
    /// A dense weight held on a device (`linear::Weight::Device`): one call.
    DeviceDense,
    /// A dense weight on the CPU: one call.
    HostDense,
    /// The sparse attention over the gathered positions.
    SparseAttention,
    /// The indexer's scores.
    IndexScores,
}

const PARTS: [Part; 7] = [Part::ExpertRead, Part::ExpertGpu, Part::ExpertCpu, Part::DeviceDense, Part::HostDense, Part::SparseAttention, Part::IndexScores];
static NANOS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static CALLS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

impl Part {
    fn index(self) -> usize {
        PARTS.iter().position(|p| *p == self).expect("every part is listed")
    }

    pub fn name(self) -> &'static str {
        match self {
            Part::ExpertRead => "expert reads",
            Part::ExpertGpu => "experts on the GPU",
            Part::ExpertCpu => "experts on the CPU",
            Part::DeviceDense => "dense on the device",
            Part::HostDense => "dense on the CPU",
            Part::SparseAttention => "sparse attention",
            Part::IndexScores => "index scores",
        }
    }
}

/// Count the time since `start` against `part`.
pub fn add(part: Part, start: Instant) {
    let i = part.index();
    NANOS[i].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    CALLS[i].fetch_add(1, Ordering::Relaxed);
}

/// Every part's seconds and calls since the last `take`, and reset them.
pub fn take() -> Vec<(Part, f64, u64)> {
    PARTS
        .iter()
        .map(|&p| {
            let i = p.index();
            (p, NANOS[i].swap(0, Ordering::Relaxed) as f64 / 1e9, CALLS[i].swap(0, Ordering::Relaxed))
        })
        .collect()
}

/// [`take`] as one line: the parts that ran, `name s (calls)`.
pub fn take_line() -> String {
    take().into_iter().filter(|(_, _, c)| *c > 0).map(|(p, s, c)| format!("{} {s:.2} s ({c})", p.name())).collect::<Vec<_>>().join(", ")
}
