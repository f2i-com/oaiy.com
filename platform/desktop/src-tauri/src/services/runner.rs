//! Process runner — spawns + monitors a service child process.
//!
//! Each `Runner` owns:
//!   - the spawned `Child` (kept alive until stop() or process exit)
//!   - a thread reading the child's stdout/stderr into a ring buffer
//!   - a tokio task watching for unexpected exit (so the registry can
//!     flip status to `Errored` when the process dies on its own)
//!
//! Logs are bounded to MAX_LOG_LINES to keep memory tight; older lines
//! drop off the front as new ones arrive.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

pub const MAX_LOG_LINES: usize = 1000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LogLine {
    pub timestamp: DateTime<Utc>,
    /// "stdout" or "stderr"
    pub stream: &'static str,
    pub text: String,
}

/// Shared ring buffer + exit watcher state. Cloned cheaply; the Mutex
/// is fine for the small N of reads/writes we do.
#[derive(Clone)]
pub struct LogBuffer(Arc<Mutex<VecDeque<LogLine>>>);

impl LogBuffer {
    /// Create a standalone buffer. Used by `Runner::spawn` and by native
    /// (non-subprocess) jobs like the Python install that want to stream
    /// progress into the same `/logs` UI.
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(VecDeque::with_capacity(MAX_LOG_LINES))))
    }

    pub fn push(&self, stream: &'static str, text: String) {
        let line = LogLine {
            timestamp: Utc::now(),
            stream,
            text,
        };
        if let Ok(mut buf) = self.0.lock() {
            if buf.len() >= MAX_LOG_LINES {
                buf.pop_front();
            }
            buf.push_back(line);
        }
    }

    pub fn snapshot(&self, tail: Option<usize>) -> Vec<LogLine> {
        let buf = match self.0.lock() {
            Ok(b) => b,
            Err(_) => return vec![],
        };
        match tail {
            Some(n) if n < buf.len() => buf.iter().skip(buf.len() - n).cloned().collect(),
            _ => buf.iter().cloned().collect(),
        }
    }
}

/// A live, running service process.
pub struct Runner {
    /// The spawned child. `Some` while alive; `None` after stop() or
    /// confirmed exit.
    pub child: Arc<Mutex<Option<Child>>>,
    pub pid: u32,
    pub logs: LogBuffer,
    pub started_at: DateTime<Utc>,
}

pub struct SpawnConfig<'a> {
    pub command: &'a str,
    pub args: &'a [String],
    pub env: &'a std::collections::HashMap<String, String>,
    pub cwd: Option<&'a str>,
}

impl Runner {
    /// Spawn the configured process. Returns the live Runner on success.
    pub fn spawn(cfg: SpawnConfig<'_>) -> std::io::Result<Self> {
        let logs = LogBuffer::new();

        let mut cmd = Command::new(cfg.command);
        cmd.args(cfg.args);
        for (k, v) in cfg.env {
            cmd.env(k, v);
        }
        if let Some(d) = cfg.cwd {
            cmd.current_dir(d);
        }
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        // On Windows, hide the console window for the spawned service.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        // On Unix, put the spawned service in its own process group so the
        // negative-pid group kill in kill_process_tree reaches descendants.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let mut child = cmd.spawn()?;
        // The backstop for a parent that is KILLED rather than asked to stop.
        // Without it this exact child — a voice server holding 8781 — outlives
        // OAIY and blocks the next launch from binding. See `job_object`.
        crate::services::job_object::adopt(child.id());
        let pid = child.id();

        // Capture stdout + stderr on dedicated threads. We use std::thread
        // rather than tokio here because std::process::Child's pipes are
        // blocking and tokio's spawn_blocking is the wrong tool — the
        // threads sit waiting on read() forever, that's fine.
        if let Some(stdout) = child.stdout.take() {
            let logs = logs.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    logs.push("stdout", line);
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            let logs = logs.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    logs.push("stderr", line);
                }
            });
        }

        let child = Arc::new(Mutex::new(Some(child)));
        Ok(Self {
            child,
            pid,
            logs,
            started_at: Utc::now(),
        })
    }

    /// Returns true if the child has exited (and clears the slot).
    pub fn check_exited(&self) -> Option<i32> {
        let mut guard = match self.child.lock() {
            Ok(g) => g,
            Err(_) => return None,
        };
        if let Some(child) = guard.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    *guard = None;
                    Some(status.code().unwrap_or(-1))
                }
                Ok(None) => None,    // still running
                Err(_) => None,
            }
        } else {
            // Already gone — return a sentinel "exited with unknown code".
            Some(-1)
        }
    }

    /// Cheap, non-consuming liveness peek: true while the child slot is still
    /// occupied (it's cleared by check_exited()/stop() once the process exits).
    /// Lets a health re-probe distinguish a service still coming up (alive) from
    /// a crashed one whose runner is kept only for its logs.
    pub fn is_alive(&self) -> bool {
        self.child.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Send SIGTERM (Unix) or TerminateProcess (Windows) and wait briefly.
    /// Returns Ok even if the child was already gone.
    pub fn stop(&self) -> std::io::Result<()> {
        let mut guard = self
            .child
            .lock()
            .map_err(|_| std::io::Error::other("runner mutex poisoned"))?;
        if let Some(mut child) = guard.take() {
            // best-effort kill; std::process::Child::kill is the cross-platform
            // path. On unix this sends SIGKILL — for a graceful SIGTERM we'd
            // need libc, but a forceful kill is fine for Phase 2.
            let _ = child.kill();
            // Reap so we don't leak a zombie.
            let _ = child.wait();
        }
        Ok(())
    }
}
