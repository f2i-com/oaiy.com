//! Embedded Python runtime + venv management.
//!
//! OAIY Desktop bundles a portable Python distribution (python-build-
//! standalone) under `${dataDir}/python/`. Services that need Python
//! (ComfyUI, custom scripts, etc.) reference this single interpreter and
//! create venvs under `${dataDir}/venvs/<name>/`. Venvs are user-named
//! and reusable — two services can point at the same venv to share
//! installed packages (saves multi-GB for torch + diffusers).
//!
//! Why bundle Python at all? Two reasons:
//!   1. We don't want to rely on whatever python is on PATH (could be
//!      a Microsoft Store stub, a wrong-version system Python, etc.).
//!   2. Users can see exactly where everything lives + uninstall by
//!      deleting one directory.
//!
//! Install + venv creation run NATIVELY in-process (download + gunzip +
//! untar via flate2/tar; venv/pip via direct subprocess) — no PowerShell,
//! no `.ps1`, no `tar.exe`. Progress streams into a `LogBuffer` so the
//! UI's LogsViewer shows it the same way it shows service logs.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use super::runner::{LogBuffer, LogLine};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VenvInfo {
    pub name: String,
    pub path: String,
    pub python_executable: Option<String>,
    pub size_bytes: u64,
    pub created: Option<DateTime<Utc>>,
    /// Services currently bound to this venv (purely informational —
    /// venv reuse is consensual, no locking).
    #[serde(default)]
    pub bound_services: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PythonSnapshot {
    pub installed: bool,
    pub runtime_dir: String,
    pub interpreter_path: Option<String>,
    pub venvs_dir: String,
    pub venvs: Vec<VenvInfo>,
    /// Status of an in-flight install/venv job, if any.
    pub current_job: Option<JobStatus>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    InstallRuntime,
    CreateVenv,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub kind: JobKind,
    pub target: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
}

pub struct Python {
    /// `${dataDir}/python/`
    runtime_dir: PathBuf,
    /// `${dataDir}/venvs/`
    venvs_dir: PathBuf,
    /// One active install or venv-create job at a time. The LogsViewer
    /// reads its logs via /api/python/logs.
    current: Arc<Mutex<Option<ActiveJob>>>,
}

/// An in-flight Python job. Both install + venv-create run natively on a
/// background thread (no shell / `.ps1`) and stream progress into `logs`;
/// `done` flips to `Some(exit_code)` when the thread finishes, which
/// `reap_exited` folds into the `JobStatus`.
struct ActiveJob {
    status: JobStatus,
    logs: LogBuffer,
    done: Arc<Mutex<Option<i32>>>,
}

pub type PythonHandle = Arc<Python>;

impl Python {
    pub fn new(data_dir: PathBuf) -> Self {
        let runtime_dir = data_dir.join("python");
        let venvs_dir = data_dir.join("venvs");
        let _ = std::fs::create_dir_all(&venvs_dir);
        Self {
            runtime_dir,
            venvs_dir,
            current: Arc::new(Mutex::new(None)),
        }
    }

    pub fn into_handle(self) -> PythonHandle {
        Arc::new(self)
    }

    /// Path to the bundled python interpreter, if installed. We probe
    /// the conventional PBS layout (python.exe in install root on
    /// Windows, bin/python3 on Unix).
    pub fn interpreter_path(&self) -> Option<PathBuf> {
        let candidates = if cfg!(windows) {
            vec![
                self.runtime_dir.join("python.exe"),
                self.runtime_dir.join("python").join("python.exe"),
            ]
        } else {
            vec![
                self.runtime_dir.join("bin").join("python3"),
                self.runtime_dir.join("python").join("bin").join("python3"),
            ]
        };
        candidates.into_iter().find(|p| p.exists())
    }

    pub fn snapshot(&self) -> PythonSnapshot {
        let interpreter = self.interpreter_path();
        let venvs = self.list_venvs();
        let current_job = self.current.lock().ok().and_then(|g| {
            g.as_ref().map(|j| j.status.clone())
        });
        PythonSnapshot {
            installed: interpreter.is_some(),
            runtime_dir: self.runtime_dir.display().to_string(),
            interpreter_path: interpreter.map(|p| p.display().to_string()),
            venvs_dir: self.venvs_dir.display().to_string(),
            venvs,
            current_job,
        }
    }

    /// Logs of the currently running job, if any. The HTTP layer wires
    /// this up so the UI can stream into the same LogsViewer used for
    /// services.
    pub fn current_logs(&self, tail: Option<usize>) -> Option<Vec<LogLine>> {
        let g = self.current.lock().ok()?;
        let job = g.as_ref()?;
        Some(job.logs.snapshot(tail))
    }

    /// Kick off the python-runtime install. Returns immediately; the UI
    /// polls /api/python to see job progress + completion. Runs natively
    /// (download + extract + ensurepip) on a background thread — no shell.
    pub fn install_runtime(&self) -> Result<(), String> {
        if self.current_job_running() {
            return Err("a python job is already running; wait for it to finish".into());
        }
        if self.interpreter_path().is_some() {
            return Err("python runtime already installed".into());
        }

        let dest = self.runtime_dir.clone();
        let logs = LogBuffer::new();
        let done = Arc::new(Mutex::new(None));
        let (logs_t, done_t) = (logs.clone(), done.clone());
        // Atomically claim the job slot BEFORE spawning (closes the check-then-set race
        // — same as create_or_reuse_venv).
        if !self.set_native_job(JobKind::InstallRuntime, "python-runtime".into(), logs, done) {
            return Err("a python job is already running; wait for it to finish".into());
        }
        std::thread::spawn(move || {
            let code = match install_python_native(&dest, &logs_t) {
                Ok(()) => 0,
                Err(e) => {
                    logs_t.push("stderr", format!("install failed: {e}"));
                    1
                }
            };
            if let Ok(mut g) = done_t.lock() {
                *g = Some(code);
            }
        });
        Ok(())
    }

    /// Create a venv at `${venvs_dir}/<name>/`. If it already exists, the
    /// existing venv is reused and any new `requirements` are pip-installed
    /// into it (this is how two services share one venv to save space).
    /// Runs natively — the bundled interpreter is invoked directly for
    /// `-m venv` then `-m pip install`, no shell wrapper.
    pub fn create_or_reuse_venv(
        &self,
        name: &str,
        requirements: &[String],
    ) -> Result<String, String> {
        if self.current_job_running() {
            return Err("a python job is already running; wait for it to finish".into());
        }
        if !valid_venv_name(name) {
            return Err("invalid venv name (use letters, digits, dash, underscore)".into());
        }
        // Reject flag-shaped requirements — legit package specs never start
        // with '-', and accepting them would let a caller inject arbitrary
        // pip flags (e.g. --index-url, --pre) into the install command.
        if requirements.iter().any(|r| r.trim_start().starts_with('-')) {
            return Err("requirements must be package specs, not pip flags".into());
        }
        let interpreter = self
            .interpreter_path()
            .ok_or("python runtime not installed — POST /api/python/install first")?;

        let venv_dir = self.venvs_dir.join(name);
        let already = venv_dir.exists();
        let venv_python = self.venv_python(&venv_dir);

        let logs = LogBuffer::new();
        let done = Arc::new(Mutex::new(None));
        let (logs_t, done_t) = (logs.clone(), done.clone());
        let reqs: Vec<String> = requirements.to_vec();
        let venv_dir_t = venv_dir.clone();
        // Claim the single-job slot ATOMICALLY before spawning the worker — two
        // concurrent calls must not both start a build on the same venv. (The
        // current_job_running() check at the top is a friendly fast-path; this is the
        // real guard against the check-then-set race.)
        if !self.set_native_job(JobKind::CreateVenv, name.to_string(), logs, done) {
            return Err("a python job is already running; wait for it to finish".into());
        }
        std::thread::spawn(move || {
            let mut code = 0;
            if already {
                logs_t.push(
                    "stdout",
                    format!("reusing existing venv at {}", venv_dir_t.display()),
                );
            } else {
                logs_t.push("stdout", format!("creating venv at {}", venv_dir_t.display()));
                code = run_logged(
                    &interpreter,
                    &["-m".into(), "venv".into(), venv_dir_t.display().to_string()],
                    &logs_t,
                );
            }
            if code == 0 && !reqs.is_empty() {
                logs_t.push("stdout", format!("pip install --upgrade {}", reqs.join(" ")));
                let mut args = vec![
                    "-m".to_string(),
                    "pip".to_string(),
                    "install".to_string(),
                    "--upgrade".to_string(),
                ];
                args.extend(reqs);
                code = run_logged(&venv_python, &args, &logs_t);
            }
            if code == 0 {
                logs_t.push("stdout", "done".into());
            }
            if let Ok(mut g) = done_t.lock() {
                *g = Some(code);
            }
        });

        Ok(venv_dir.display().to_string())
    }

    pub fn delete_venv(&self, name: &str) -> Result<(), String> {
        // Refuse to delete a venv that an in-flight create-venv job is
        // populating — deleting it out from under the job races the pip
        // install and leaves a half-built tree.
        if let Ok(g) = self.current.lock() {
            if let Some(job) = g.as_ref() {
                if job.status.finished_at.is_none()
                    && job.status.kind == JobKind::CreateVenv
                    && job.status.target == name
                {
                    return Err(
                        "a venv job is running for this name; wait for it to finish".into(),
                    );
                }
            }
        }
        if !valid_venv_name(name) {
            return Err("invalid venv name".into());
        }
        let venv_dir = self.venvs_dir.join(name);
        if !venv_dir.starts_with(&self.venvs_dir) {
            return Err("path escapes venvs dir".into());
        }
        std::fs::remove_dir_all(&venv_dir).map_err(|e| format!("delete failed: {e}"))?;
        Ok(())
    }

    /// Poll the active job. Cheap — call from the same reap_exited tick
    /// that handles services. Folds JobStatus.finished_at + exit_code on
    /// completion so the snapshot UI flips state without polling on the
    /// frontend.
    pub fn reap_exited(&self) {
        let mut g = match self.current.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let Some(job) = g.as_mut() else { return };
        if job.status.finished_at.is_some() {
            return; // already finalised
        }
        let code = job.done.lock().ok().and_then(|d| *d);
        if let Some(code) = code {
            job.status.finished_at = Some(Utc::now());
            job.status.exit_code = Some(code);
            if code != 0 {
                job.status.error = Some(format!("exit code {code}"));
            }
        }
    }

    // ---- helpers ----

    fn current_job_running(&self) -> bool {
        let g = match self.current.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        match g.as_ref() {
            Some(j) => j.status.finished_at.is_none(),
            None => false,
        }
    }

    /// Atomically claim the single-job slot. Returns false (and claims nothing) if a
    /// job is already running. The running-check and the set happen under ONE lock,
    /// so two concurrent callers can't both pass a separate check and then both start
    /// — which previously let two `python -m venv`/`pip install` runs race on the same
    /// venv directory and corrupt it. Callers must claim BEFORE spawning the worker.
    #[must_use]
    fn set_native_job(
        &self,
        kind: JobKind,
        target: String,
        logs: LogBuffer,
        done: Arc<Mutex<Option<i32>>>,
    ) -> bool {
        let Ok(mut g) = self.current.lock() else {
            return false;
        };
        if matches!(g.as_ref(), Some(j) if j.status.finished_at.is_none()) {
            return false;
        }
        *g = Some(ActiveJob {
            status: JobStatus {
                kind,
                target,
                started_at: Utc::now(),
                finished_at: None,
                exit_code: None,
                error: None,
            },
            logs,
            done,
        });
        true
    }

    fn venv_python(&self, venv_dir: &Path) -> PathBuf {
        if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        }
    }

    fn list_venvs(&self) -> Vec<VenvInfo> {
        let mut out = Vec::new();
        let read = match std::fs::read_dir(&self.venvs_dir) {
            Ok(r) => r,
            Err(_) => return out,
        };
        for entry in read.flatten() {
            let p = entry.path();
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if !meta.is_dir() {
                continue;
            }
            let name = match p.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let py = self.venv_python(&p);
            let created = meta
                .created()
                .or_else(|_| meta.modified())
                .ok()
                .and_then(|t| {
                    let d = t.duration_since(std::time::UNIX_EPOCH).ok()?;
                    DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos())
                });
            out.push(VenvInfo {
                name,
                path: p.display().to_string(),
                python_executable: py.exists().then(|| py.display().to_string()),
                size_bytes: dir_size_shallow(&p),
                created,
                bound_services: Vec::new(),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

/// Download python-build-standalone + extract it under `dest`, then
/// bootstrap pip. Pure Rust — no `tar.exe` / PowerShell. Streams progress
/// into `logs`. Blocking; call from a background thread.
fn install_python_native(dest: &Path, logs: &LogBuffer) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("mkdir {}: {e}", dest.display()))?;

    // Pinned PBS release (bump when a newer one is stable). The
    // `install_only` flavour skips the cpython source/headers we don't need.
    const RELEASE: &str = "20260510";
    const PY_VERSION: &str = "3.12.13";
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        other => return Err(format!("unsupported CPU arch '{other}' (need x86_64 or aarch64)")),
    };
    let triple = if cfg!(target_os = "windows") {
        "pc-windows-msvc"
    } else if cfg!(target_os = "macos") {
        "apple-darwin"
    } else {
        "unknown-linux-gnu"
    };
    let asset = format!("cpython-{PY_VERSION}+{RELEASE}-{arch}-{triple}-install_only.tar.gz");
    let url = format!(
        "https://github.com/astral-sh/python-build-standalone/releases/download/{RELEASE}/{asset}"
    );

    logs.push("stdout", format!("downloading {url}"));
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut resp = client
        .get(&url)
        .send()
        .map_err(|e| format!("download request: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download returned HTTP {}", resp.status()));
    }
    let total = resp.content_length();

    let tmp = std::env::temp_dir().join(&asset);
    {
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("create temp {}: {e}", tmp.display()))?;
        let mut buf = vec![0u8; 64 * 1024];
        let mut got: u64 = 0;
        let mut last: u64 = 0;
        loop {
            let n = resp.read(&mut buf).map_err(|e| format!("download read: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| format!("write temp: {e}"))?;
            got += n as u64;
            if got - last >= 8 * 1024 * 1024 {
                last = got;
                let msg = match total {
                    Some(t) => format!(
                        "downloaded {:.0} / {:.0} MB",
                        got as f64 / 1.0e6,
                        t as f64 / 1.0e6
                    ),
                    None => format!("downloaded {:.0} MB", got as f64 / 1.0e6),
                };
                logs.push("stdout", msg);
            }
        }
    }

    logs.push("stdout", "extracting...".into());
    // PBS install_only tarballs unpack as a single top-level `python/` dir;
    // strip it so the interpreter lands at <dest>/python.exe (Windows) or
    // <dest>/bin/python3 (Unix).
    let tar_gz = std::fs::File::open(&tmp).map_err(|e| format!("open temp: {e}"))?;
    let gz = flate2::read::GzDecoder::new(BufReader::new(tar_gz));
    let mut archive = tar::Archive::new(gz);
    archive.set_preserve_permissions(true);
    let entries = archive.entries().map_err(|e| format!("read archive: {e}"))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("archive entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("entry path: {e}"))?
            .into_owned();
        let stripped: PathBuf = path.components().skip(1).collect();
        if stripped.as_os_str().is_empty() {
            continue;
        }
        // Defend against path traversal in a garbled/malicious archive.
        if stripped
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            continue;
        }
        let out = dest.join(&stripped);
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        entry
            .unpack(&out)
            .map_err(|e| format!("unpack {}: {e}", stripped.display()))?;
    }
    let _ = std::fs::remove_file(&tmp);

    let python = if cfg!(windows) {
        dest.join("python.exe")
    } else {
        dest.join("bin").join("python3")
    };
    if !python.exists() {
        return Err(format!(
            "expected {} after extraction but it's missing — PBS layout may have changed",
            python.display()
        ));
    }

    logs.push("stdout", "bootstrapping pip (ensurepip)...".into());
    let code = run_logged(
        &python,
        &["-m".into(), "ensurepip".into(), "--upgrade".into()],
        logs,
    );
    if code != 0 {
        logs.push(
            "stderr",
            format!(
                "ensurepip exited {code}; venvs still create but pip may need a manual \
                 'python -m ensurepip'"
            ),
        );
    }
    logs.push("stdout", format!("done. python is at {}", python.display()));
    Ok(())
}

/// Run a subprocess, streaming its stdout/stderr into `logs` line by line.
/// Returns the exit code (or -1 on spawn/wait failure). No shell — the
/// program is invoked directly, so there's no quoting/encoding surface.
fn run_logged(program: &Path, args: &[String], logs: &LogBuffer) -> i32 {
    let mut cmd = Command::new(program);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            logs.push("stderr", format!("failed to run {}: {e}", program.display()));
            return -1;
        }
    };
    let mut handles = Vec::new();
    if let Some(out) = child.stdout.take() {
        let l = logs.clone();
        handles.push(std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                l.push("stdout", line);
            }
        }));
    }
    if let Some(err) = child.stderr.take() {
        let l = logs.clone();
        handles.push(std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                l.push("stderr", line);
            }
        }));
    }
    let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
    for h in handles {
        let _ = h.join();
    }
    code
}

fn valid_venv_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() < 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// One-level dir size — accurate enough for the UI's "this venv takes
/// 2.3 GB" line without the cost of a full deep walk on every snapshot.
/// (A deep walk on a torch venv with 10k files takes hundreds of ms.)
fn dir_size_shallow(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(m) = e.metadata() {
                if m.is_file() {
                    total += m.len();
                } else if m.is_dir() {
                    // One more level for site-packages — biggest user
                    // of disk in a typical venv.
                    let inner = e.path();
                    if let Ok(rd2) = std::fs::read_dir(&inner) {
                        for e2 in rd2.flatten() {
                            if let Ok(m2) = e2.metadata() {
                                if m2.is_file() {
                                    total += m2.len();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_venv_names() {
        assert!(!valid_venv_name(""));
        assert!(!valid_venv_name("../escape"));
        assert!(!valid_venv_name("with space"));
        assert!(!valid_venv_name("with/slash"));
    }

    #[test]
    fn accepts_good_venv_names() {
        assert!(valid_venv_name("comfyui"));
        assert!(valid_venv_name("torch-cuda121"));
        assert!(valid_venv_name("Default_2"));
    }
}
