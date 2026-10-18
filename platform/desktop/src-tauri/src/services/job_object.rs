//! Tying every child process's life to this one.
//!
//! On Windows a child does NOT die with its parent. `TerminateProcess` — which
//! is what Task Manager's End Task and PowerShell's `Stop-Process -Force` issue
//! — runs no cleanup code, so every cooperative teardown this app has is skipped
//! and each spawned service keeps running, holding its port.
//!
//! That is not hypothetical. Force-killing OAIY Desktop left
//! `aokie-voice-server --mode stt --port 8781` and its `--mode tts --port 8782`
//! sibling alive; the next launch could not bind either port, retried five
//! times over about a minute, and parked both services as "needs Repair".
//!
//! A job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is the only mechanism
//! that fixes this class of problem rather than one instance of it. The kernel
//! closes our handles when the process dies — however it dies, including a
//! crash or a kill we never get to see — and closing the last handle to the job
//! terminates everything still inside it. No cooperation required, which is
//! exactly the property every existing teardown path lacks.
//!
//! Deliberately NOT a replacement for the graceful paths. Those still run on a
//! normal exit and stop children in the right order, giving them a chance to
//! flush. This is the backstop for when they do not run at all.

/// Put `pid` under this process's kill-on-close job.
///
/// Best-effort by design: a machine where the job cannot be created is a
/// machine that behaves exactly as it did before this existed, which is worse
/// than the fix but no worse than the status quo. Never blocks a spawn.
pub fn adopt(pid: u32) {
    #[cfg(windows)]
    windows_impl::adopt(pid);
    #[cfg(not(windows))]
    let _ = pid;
}

/// A job of one child's own, nested in the process-wide one: dropped, it ends whatever is still in it, which is what
/// the child started and left running (a helper it ran detached, a grandchild), however the child itself ended.
///
/// For a child whose leftovers must end with it, not with the app. On Windows nothing else finds them: `taskkill /T`
/// walks the tree from a live parent, so once a flow's CLI had exited, what it left kept running until OAIY quit.
/// Elsewhere this holds nothing: a process group does the same there.
pub struct Tree {
    #[cfg(windows)]
    _job: Option<windows_impl::OwnJob>,
}

/// [`adopt`] `pid`, and put it under a job of its own besides ([`Tree`]). Best-effort like `adopt`: a job that cannot
/// be made or joined leaves the child as `adopt` alone would.
pub fn adopt_tree(pid: u32) -> Tree {
    adopt(pid);
    Tree {
        #[cfg(windows)]
        _job: windows_impl::own_job(pid),
    }
}

/// Whether the backstop is actually active, for diagnostics.
pub fn active() -> bool {
    #[cfg(windows)]
    {
        windows_impl::job_handle().is_some()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(windows)]
mod windows_impl {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    /// A raw HANDLE is a pointer and therefore not `Send`/`Sync`. The handle is
    /// only ever read, and the kernel object it names outlives every use, so
    /// wrapping it is sound — but it has to be explicit.
    struct JobHandle(HANDLE);
    unsafe impl Send for JobHandle {}
    unsafe impl Sync for JobHandle {}

    static JOB: OnceLock<Option<JobHandle>> = OnceLock::new();

    /// The process-wide job, created once on first use.
    ///
    /// Never closed. Closing it is precisely what kills the children, so the
    /// handle must live until the process does — at which point the kernel
    /// closes it for us, which is the entire mechanism.
    pub(super) fn job_handle() -> Option<HANDLE> {
        JOB.get_or_init(|| {
            // SAFETY: a null name creates an anonymous job owned by this
            // process; the returned handle is checked before use.
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job.is_null() {
                log::warn!(
                    "could not create the child-process job; children may outlive a forced exit"
                );
                return None;
            }

            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `info` is a correctly-sized, fully-initialised struct of
            // the type this information class expects.
            let ok = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if ok == 0 {
                // A job without the limit set would silently do nothing, which
                // is worse than no job at all: it would look like protection.
                log::warn!("could not set kill-on-close on the child job; not using it");
                unsafe { CloseHandle(job) };
                return None;
            }
            Some(JobHandle(job))
        })
        .as_ref()
        .map(|h| h.0)
    }

    pub(super) fn adopt(pid: u32) {
        let Some(job) = job_handle() else { return };
        // PROCESS_SET_QUOTA | PROCESS_TERMINATE is the documented minimum for
        // AssignProcessToJobObject. Asking for less fails; asking for ALL_ACCESS
        // would need rights we have no reason to hold.
        // SAFETY: pid names a child we just spawned; a failed open returns null,
        // which is checked.
        let handle = unsafe { OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid) };
        if handle.is_null() {
            log::warn!("could not open child pid {pid} to place it under the job");
            return;
        }
        // SAFETY: both handles are non-null and owned by this process.
        let ok = unsafe { AssignProcessToJobObject(job, handle) };
        if ok == 0 {
            // Nested jobs are supported from Windows 8 onward, so the common
            // cause here is a child that already belongs to a job which forbids
            // breakaway. Worth a line, not worth failing the spawn.
            log::warn!("could not place child pid {pid} under the job; it may outlive a forced exit");
        }
        // Closing OUR handle to the process does not remove it from the job —
        // membership is a property of the process, not of this handle.
        unsafe { CloseHandle(handle) };
    }

    /// A job of one child's own ([`super::Tree`]). Its handle is the only one (not inheritable), so closing it, with
    /// kill-on-close set, ends every process still in the job.
    pub(super) struct OwnJob(HANDLE);

    impl Drop for OwnJob {
        fn drop(&mut self) {
            // SAFETY: the handle is this value's own, open since `own_job` made it, and closed only here.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// Make a kill-on-close job and put `pid` in it. Joined after the process-wide job ([`adopt`]), it nests inside
    /// that one (Windows 8 and later): a process in a job can join only an empty one, or one of its own hierarchy.
    pub(super) fn own_job(pid: u32) -> Option<OwnJob> {
        // SAFETY: a null name makes an anonymous job of this process's; the handle is checked before use.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return None;
        }
        let job = OwnJob(job);
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `info` is a fully initialised struct of the type this information class expects.
        let ok = unsafe {
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return None;
        }
        // SAFETY: as in `adopt`: a failed open is null, which is checked.
        let handle = unsafe { OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid) };
        if handle.is_null() {
            return None;
        }
        // SAFETY: both handles are non-null and this process's own.
        let joined = unsafe { AssignProcessToJobObject(job.0, handle) } != 0;
        unsafe { CloseHandle(handle) };
        if !joined {
            log::warn!("could not give child pid {pid} a job of its own; what it leaves may run until OAIY quits");
        }
        joined.then_some(job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adopting_a_pid_that_does_not_exist_is_survivable() {
        // The backstop must never be able to take down a spawn. pid 0 is the
        // system idle process and cannot be opened with these rights, so this
        // exercises the failure path rather than the happy one.
        adopt(0);
        adopt(u32::MAX);
    }

    #[cfg(windows)]
    #[test]
    fn the_job_is_created_once_and_reused() {
        // Re-creating it per child would put each one in its own job, and the
        // first handle drop would kill an unrelated service.
        let first = super::windows_impl::job_handle();
        let second = super::windows_impl::job_handle();
        assert_eq!(first.is_some(), second.is_some());
        if let (Some(a), Some(b)) = (first, second) {
            assert!(std::ptr::eq(a, b), "the job handle must be a singleton");
        }
    }

    /// What a child leaves running ends when its own job is dropped: a grandchild a batch file started and left behind
    /// (as a flow's CLI may leave a helper) wrote its file after its parent had exited. Beside it, one whose job is
    /// kept writes its file, so the test sees a survivor when there is one.
    #[cfg(windows)]
    #[test]
    fn a_tree_ends_what_its_child_left_running() {
        use std::process::{Command, Stdio};
        let dir = std::env::temp_dir().join(format!("oaiy-job-tree-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The child waits a second (it is in its jobs before it starts anything), starts the grandchild and exits; the
        // grandchild writes its file three seconds later.
        let script = |name: &str| {
            let inner = dir.join(format!("{name}-inner.cmd"));
            let marker = dir.join(format!("{name}.txt"));
            std::fs::write(&inner, format!("@echo off\r\nping -n 4 127.0.0.1 >nul\r\necho late> \"{}\"\r\n", marker.display())).unwrap();
            let outer = dir.join(format!("{name}.cmd"));
            std::fs::write(&outer, format!("@echo off\r\nping -n 2 127.0.0.1 >nul\r\nstart \"\" /b cmd /c \"{}\"\r\n", inner.display())).unwrap();
            outer
        };
        let spawn = |outer: &std::path::Path| {
            Command::new("cmd").arg("/c").arg(outer).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap()
        };
        let (ended, kept) = (script("ended"), script("kept"));
        let mut a = spawn(&ended);
        let tree_a = adopt_tree(a.id());
        let mut b = spawn(&kept);
        let tree_b = adopt_tree(b.id());
        assert!(a.wait().unwrap().success() && b.wait().unwrap().success());
        drop(tree_a);
        std::thread::sleep(std::time::Duration::from_secs(6));
        assert!(dir.join("kept.txt").exists(), "the kept job's grandchild ran on, so a survivor is seen");
        assert!(!dir.join("ended.txt").exists(), "the dropped job's grandchild was ended with it");
        drop(tree_b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn the_backstop_reports_itself_active_on_windows() {
        // If this ever starts failing on a normal dev box, the fix has silently
        // stopped protecting anything.
        assert!(active(), "the job object should be available on Windows");
    }
}
