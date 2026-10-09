//! Where a GGUF model's time goes on this backend: counters any thread adds to and a timing test reads and resets.

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

/// A dense call ([`crate::dense::forward_batch`]): what it makes before its submit (the inputs' uploads, its
/// buffers and bind groups, the passes), the submit, and the wait for its read-back; and the bytes of the weights
/// it read (its second number the weights).
pub static DENSE_MAKE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static DENSE_SUBMIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static DENSE_WAIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static DENSE_BYTES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// The dense calls' counters since the last call, as one line, and reset.
pub fn take_dense_line() -> String {
    let take = |c: &[AtomicU64; 2]| (c[0].swap(0, Ordering::Relaxed) as f64, c[1].swap(0, Ordering::Relaxed));
    let (m, s, w, b) = (take(&DENSE_MAKE), take(&DENSE_SUBMIT), take(&DENSE_WAIT), take(&DENSE_BYTES));
    format!("dense calls {}: made in {:.3} s, submitted in {:.3} s, waited for {:.3} s; {} weights of {:.2} GB", m.1, m.0 / 1e9, s.0 / 1e9, w.0 / 1e9, b.1, b.0 / 1e9)
}

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
