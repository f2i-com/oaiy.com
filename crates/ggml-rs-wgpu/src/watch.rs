//! The watchdog: a wait on the device that has hung for good ends the job, or the process where asked.

use super::*;

/// The driver's calls a [`Gpu`]'s watchdog watches ([`Gpu::wait`]'s polls, a second's timeout each): how many are in
/// progress, and when one last began or returned (ms since the first watched). None returned in [`HUNG`] has hung in
/// the driver: NVIDIA's Vulkan, its device lost to a reset (Windows' TDR: a submission past 2 s), spins in its fence
/// wait whatever the timeout; said, and the process ended where it is one job's ([`end_process_on_hang`]).
#[derive(Default)]
pub(super) struct Watch {
    pub(super) calls: AtomicU64,
    pub(super) stamp: AtomicU64,
}

/// How long a watched call may go without one returning.
pub(super) const HUNG: std::time::Duration = std::time::Duration::from_secs(30);

/// Milliseconds since the first call (never 0).
pub(super) fn watch_clock() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

impl Watch {
    /// A watched call begun, ended when the guard is dropped.
    pub(super) fn enter(&self) -> Watched<'_> {
        self.stamp.store(watch_clock(), Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Watched(self)
    }
}

pub(super) struct Watched<'a>(&'a Watch);

impl Drop for Watched<'_> {
    fn drop(&mut self) {
        self.0.stamp.store(watch_clock(), Ordering::Relaxed);
        self.0.calls.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Whether a hung driver call ends the process ([`end_process_on_hang`]); else it is only said, once.
pub(super) static END_ON_HANG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A driver call hung on a lost device ([`HUNG`] without one returning) ends this process, saying why on stderr: for a
/// process of one job (the media worker: its last line the job's error, rather than a job that never ends). Elsewhere
/// (a server's other models on other devices) it is only said.
pub fn end_process_on_hang() {
    END_ON_HANG.store(true, Ordering::Relaxed);
}

/// `watch`'s calls checked every second while its GPU lives: one hung ([`HUNG`]) said on stderr, and the process ended
/// where [`end_process_on_hang`] has asked.
pub(super) fn watchdog(watch: std::sync::Weak<Watch>, lost: std::sync::Weak<Mutex<Option<String>>>, name: String) {
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

/// This process ended at once, its exit code 3: no DLL's detach run (a GPU driver's would wait for its hung threads).
pub(super) fn end_process() -> ! {
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
