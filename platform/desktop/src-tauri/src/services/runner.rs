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

/// Is `name` one of OAIY's own credentials: `OAIY_SERVER_TOKEN`, `OAIY_HF_TOKEN`
/// or any other `OAIY_*TOKEN`?
///
/// Compared without regard to case, because environment names on Windows do not
/// have one.
pub(crate) fn is_oaiy_token(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("OAIY_") && name.ends_with("TOKEN")
}

/// Take OAIY's own tokens out of what `cmd` inherits from this process.
///
/// `Command` hands its child the whole environment of this one, and this one
/// holds the headless server's bearer (`OAIY_SERVER_TOKEN`, which is the key to
/// every privileged route and so to running code on the machine) and the
/// Hugging Face token. A model server, a Python service, a custom node or a
/// browser it installs is code that is not ours, and so is every package a
/// venv's `pip` fetches. None of them needs either token, so none is given one:
/// a package or service that sends its own environment home, logs it (a
/// service's output goes into the log buffer the API serves) or passes it on to
/// what it starts no longer takes them along.
///
/// This is NOT a defence against hostile code that goes looking as the same
/// user. On Linux such code can read this process's own `/proc/<pid>/environ`,
/// which holds the environment it started with, and the owner-only files (a paired
/// app's token, which the auth guard accepts like the configured one, and the
/// providers' keys) are readable by that user by design. Keeping the bearer out
/// of the environment altogether needs the server to take it from a credential
/// file, which it does not yet.
///
/// Only what is inherited is scrubbed. Call this BEFORE applying a step's own
/// `env`, so a value a template sets on purpose still arrives. No shipped
/// install script or template reads a token (the models are public revisions
/// pinned by SHA-256, and downloads run in this process with the saved token),
/// so no step is given one.
pub(crate) fn scrub_inherited_tokens(cmd: &mut Command) {
    for (name, _) in std::env::vars_os() {
        if name.to_str().is_some_and(is_oaiy_token) {
            cmd.env_remove(&name);
        }
    }
}

impl Runner {
    /// Spawn the configured process. Returns the live Runner on success.
    pub fn spawn(cfg: SpawnConfig<'_>) -> std::io::Result<Self> {
        let logs = LogBuffer::new();

        let mut cmd = Command::new(cfg.command);
        cmd.args(cfg.args);
        // Managed services and the installers both come through here.
        scrub_inherited_tokens(&mut cmd);
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

/// What the tests of the child environment share: a way to plant variables in
/// THIS process, and a child that prints the environment it actually received.
///
/// Nothing here ever prints a value. A test that dumped the environment on
/// failure would put the developer's real keys in a log, so the tests only ever
/// name the variables they checked.
#[cfg(test)]
pub(crate) mod env_probe {
    use super::LogLine;
    use std::collections::HashSet;

    /// Printed after the environment, so a test knows the child has said everything.
    pub const END: &str = "__END_OF_ENV__";

    /// Variables planted in this process's environment, taken out again on drop
    /// (assertion or not) so no other test in the binary inherits them. Held one
    /// at a time, so two of these tests planting the same name cannot take it out
    /// from under each other.
    pub struct Planted {
        names: Vec<String>,
        _one_at_a_time: std::sync::MutexGuard<'static, ()>,
    }

    impl Planted {
        pub fn new(vars: &[(&str, &str)]) -> Self {
            static PLANTING: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let one_at_a_time = PLANTING.lock().unwrap_or_else(|e| e.into_inner());
            for (name, value) in vars {
                std::env::set_var(name, value);
            }
            Planted {
                names: vars.iter().map(|(name, _)| name.to_string()).collect(),
                _one_at_a_time: one_at_a_time,
            }
        }
    }

    impl Drop for Planted {
        fn drop(&mut self) {
            for name in &self.names {
                std::env::remove_var(name);
            }
        }
    }

    /// A program and arguments that print every variable of their environment as
    /// `NAME=value` lines, then [`END`].
    pub fn env_dump() -> (String, Vec<String>) {
        if cfg!(windows) {
            ("cmd.exe".into(), vec!["/C".into(), format!("set & echo {END}")])
        } else {
            ("sh".into(), vec!["-c".into(), format!("env; echo {END}")])
        }
    }

    /// Has the child said everything?
    pub fn is_complete(lines: &[LogLine]) -> bool {
        lines.iter().any(|l| l.text.trim() == END)
    }

    /// The NAMES the child's environment had, upper-cased (on Windows they have
    /// no case, and `set` prints them as they were stored).
    pub fn names(lines: &[LogLine]) -> HashSet<String> {
        lines
            .iter()
            .filter(|l| l.stream == "stdout")
            .filter_map(|l| l.text.split_once('='))
            .filter(|(name, _)| !name.is_empty() && !name.starts_with('='))
            .map(|(name, _)| name.to_ascii_uppercase())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::env_probe::{env_dump, is_complete, names, Planted};
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::time::{Duration, Instant};

    #[test]
    fn oaiy_tokens_are_recognised_by_name_and_only_those() {
        for token in [
            "OAIY_SERVER_TOKEN",
            "OAIY_HF_TOKEN",
            "OAIY_SOMETHING_NEW_TOKEN",
            "OAIY_TOKEN",
            "oaiy_hf_token",
            "Oaiy_Server_Token",
        ] {
            assert!(is_oaiy_token(token), "{token} is one of OAIY's tokens");
        }
        for other in [
            "OAIY_DATA_DIR",
            "OAIY_BIN_DIR",
            "OAIY_MODELS_DIR",
            "OAIY_SERVER_URL",
            "OAIY_TOKEN_FILE",
            "HF_TOKEN",
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "MY_OAIY_TOKEN",
            "PATH",
            "",
        ] {
            assert!(!is_oaiy_token(other), "{other} is not");
        }
    }

    /// Spawn the env dump through `Runner::spawn`, the one call both the managed
    /// services and the installers are started with, and return everything the
    /// child printed of the environment it actually received.
    fn printed_by_a_spawned_child(explicit: &HashMap<String, String>) -> Vec<LogLine> {
        let (command, args) = env_dump();
        let runner = Runner::spawn(SpawnConfig { command: &command, args: &args, env: explicit, cwd: None })
            .expect("spawn a child that prints its environment");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let lines = runner.logs.snapshot(None);
            if is_complete(&lines) {
                return lines;
            }
            assert!(Instant::now() < deadline, "the child never finished printing its environment");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// The names of the variables the child received.
    fn received_by_a_spawned_child(explicit: &HashMap<String, String>) -> HashSet<String> {
        names(&printed_by_a_spawned_child(explicit))
    }

    #[test]
    fn a_spawned_child_does_not_inherit_oaiy_tokens_but_gets_everything_else() {
        // The headless server carries its real bearer and its Hugging Face token in
        // its own environment. Plant both, another OAIY_*TOKEN, and variables that
        // must still arrive: not a token, and not what the child could run without.
        let _planted = Planted::new(&[
            ("OAIY_SERVER_TOKEN", "planted-server-value"),
            ("OAIY_HF_TOKEN", "planted-hf-value"),
            ("OAIY_SCRUB_TEST_EXTRA_TOKEN", "planted"),
            ("oaiy_scrub_test_lower_token", "planted"),
            ("OAIY_SCRUB_TEST_KEEP", "planted"),
            ("OAIY_SCRUB_TEST_TOKEN_FILE", "planted"),
        ]);
        // What a template sets for a step: only what is INHERITED is scrubbed, so a
        // variable of the step's own that is named like a token still arrives.
        let mut explicit = HashMap::new();
        explicit.insert("OAIY_DATA_DIR".to_string(), "the-data-dir".to_string());
        explicit.insert("OAIY_SCRUB_TEST_STEP_TOKEN".to_string(), "on-purpose".to_string());

        let seen = received_by_a_spawned_child(&explicit);

        for gone in [
            "OAIY_SERVER_TOKEN",
            "OAIY_HF_TOKEN",
            "OAIY_SCRUB_TEST_EXTRA_TOKEN",
            "OAIY_SCRUB_TEST_LOWER_TOKEN",
        ] {
            assert!(!seen.contains(gone), "{gone} reached the child");
        }
        for arrived in [
            "OAIY_SCRUB_TEST_KEEP",
            "OAIY_SCRUB_TEST_TOKEN_FILE",
            "OAIY_DATA_DIR",
            "OAIY_SCRUB_TEST_STEP_TOKEN",
            "PATH",
        ] {
            assert!(seen.contains(arrived), "{arrived} should have reached the child (is the probe reading its environment at all?)");
        }
    }

    /// The order the doc comment promises: the inherited tokens are taken out FIRST and the step's own
    /// variables applied after, so a template that sets an `OAIY_*TOKEN` on purpose is given it even
    /// when this process holds a token of the same name. Scrubbing after the step's variables would
    /// take the step's own value out with the inherited one.
    #[test]
    fn a_step_that_names_a_token_this_process_also_holds_gets_its_own_value() {
        let _planted = Planted::new(&[("OAIY_SCRUB_TEST_OVERLAP_TOKEN", "the-servers-own")]);
        let mut explicit = HashMap::new();
        explicit.insert("OAIY_SCRUB_TEST_OVERLAP_TOKEN".to_string(), "the-steps-own".to_string());

        let printed = printed_by_a_spawned_child(&explicit);

        // Both values are made up for the test; neither is anyone's credential.
        let value = printed
            .iter()
            .filter(|l| l.stream == "stdout")
            .find_map(|l| l.text.trim().strip_prefix("OAIY_SCRUB_TEST_OVERLAP_TOKEN="))
            .map(str::to_owned);
        assert_eq!(value.as_deref(), Some("the-steps-own"), "the child should hold the step's own value, and only that");
    }
}
