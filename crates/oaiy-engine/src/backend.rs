//! Row-parallel loops and a couple of machine probes.
//!
//! [`parallel_rows`] splits `[0, n)` into blocks and runs them on scoped
//! threads that pull the next block from a shared counter. Each block is a
//! disjoint range of rows, so a caller that writes only its own rows gets
//! the same bits whatever the thread count or the order blocks finish in.
//! Scoped threads let the closure borrow the caller's data without `unsafe`
//! or `'static` bounds; spawning per call costs microseconds, next to the
//! milliseconds of work a call is worth splitting for.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Upper bound on worker threads for one loop.
const MAX_THREADS: usize = 64;

/// Logical CPUs this process may use (at least 1).
pub fn hardware_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Installed RAM in bytes, or 0 when unknown. `OAIY_RAM_BYTES` overrides
/// the probe. Without FFI (the core is std-only) the only portable source
/// is Linux's `/proc/meminfo`; elsewhere set the variable.
pub fn physical_ram() -> u64 {
    if let Some(n) = std::env::var("OAIY_RAM_BYTES").ok().and_then(|v| v.trim().parse().ok()) {
        return n;
    }
    meminfo_total().unwrap_or(0)
}

fn meminfo_total() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// Run `f(begin, end)` over blocks covering `[0, n)`, on up to
/// [`hardware_concurrency`] threads (the caller's included), and return
/// when every block is done. Block starts are multiples of `min_chunk`, so
/// callers with blocked data (quant blocks, tiles) never see a split block.
/// Small jobs (`n <= min_chunk`, or one CPU) run inline as `f(0, n)`.
pub fn parallel_rows(n: usize, min_chunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    let unit = min_chunk.max(1);
    let threads = hardware_concurrency().min(MAX_THREADS);
    if threads < 2 || n <= unit {
        f(0, n);
        return;
    }
    // one even share per thread, at least one unit, in whole units
    let block = n.div_ceil(threads).max(unit).div_ceil(unit) * unit;
    let next = AtomicUsize::new(0);
    let worker = || loop {
        let begin = next.fetch_add(block, Ordering::Relaxed);
        if begin >= n {
            break;
        }
        f(begin, (begin + block).min(n));
    };
    let helpers = n.div_ceil(block).min(threads) - 1;
    std::thread::scope(|s| {
        for _ in 0..helpers {
            s.spawn(worker);
        }
        worker();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn small_jobs_run_inline() {
        let seen = Mutex::new(Vec::new());
        parallel_rows(10, 64, &|b, e| seen.lock().unwrap().push((b, e)));
        assert_eq!(*seen.lock().unwrap(), [(0, 10)]);
        let seen = Mutex::new(Vec::new());
        parallel_rows(0, 1, &|b, e| seen.lock().unwrap().push((b, e)));
        assert_eq!(*seen.lock().unwrap(), [(0, 0)]);
    }

    #[test]
    fn blocks_cover_the_range_once_on_unit_boundaries() {
        for (n, unit) in [(1000, 64), (1001, 1), (65, 64), (4096, 32), (7, 3)] {
            let count = Mutex::new(vec![0u8; n]);
            parallel_rows(n, unit, &|b, e| {
                assert!(b < e && e <= n);
                assert_eq!(b % unit, 0, "block start {b} not on a {unit} boundary");
                let mut c = count.lock().unwrap();
                for x in &mut c[b..e] {
                    *x += 1;
                }
            });
            assert!(count.lock().unwrap().iter().all(|&c| c == 1), "n={n} unit={unit}");
        }
    }

    #[test]
    fn row_results_do_not_depend_on_scheduling() {
        let n = 777;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let serial: Vec<f32> = input.iter().map(|v| v.tanh() * 3.0).collect();
        let out = Mutex::new(vec![0f32; n]);
        parallel_rows(n, 8, &|b, e| {
            let part: Vec<f32> = input[b..e].iter().map(|v| v.tanh() * 3.0).collect();
            out.lock().unwrap()[b..e].copy_from_slice(&part);
        });
        let out = out.into_inner().unwrap();
        assert!(out.iter().zip(&serial).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn probes() {
        assert!(hardware_concurrency() >= 1);
        std::env::set_var("OAIY_RAM_BYTES", "123456789");
        assert_eq!(physical_ram(), 123_456_789);
        std::env::remove_var("OAIY_RAM_BYTES");
        let ram = physical_ram();
        assert!(ram == 0 || ram > 1 << 28, "implausible RAM {ram}");
    }
}
