//! OAIY Desktop — entry point.
//!
//! Phase 1: bring up the Tauri shell + a tray icon + a localhost HTTP API
//! that oaiy-web can discover.
//! Phase 2: load the service registry, manage child processes, expose
//! /api/services/* and stop everything cleanly on exit.

pub mod ai;
pub mod http;
pub mod services;
pub mod bridge;
pub mod plugins;

/// Port the localhost API binds to. Fixed so oaiy-web's detection probe has a
/// stable target. Shared by both binaries (the GUI and the headless server).
pub const DESKTOP_PORT: u16 = 17972;

/// File logging for the GUI build.
///
/// The headless binary installs a stderr logger. The GUI installed NONE, so all
/// 37 `log::` call sites in this crate wrote to nowhere in the packaged app —
/// including "HTTP server exited", "registry init failed", and every plugin
/// autostart failure. That is not a cosmetic gap: a startup hang in this app was
/// diagnosable only by rebuilding it with `eprintln!`s, which is not something a
/// user can do. A desktop app that cannot say what went wrong has to be debugged
/// by whoever compiled it.
///
/// Deliberately small: no rotation library, no async appender. The write rate is
/// a few lines per minute and the file is capped and rolled once, which is all
/// that is needed to answer "what happened just before it broke".
mod applog {
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Roll at 2 MiB, keeping one previous file. Enough to cover several
    /// sessions; small enough that a user can open it and a support request can
    /// carry it.
    const MAX_BYTES: u64 = 2 * 1024 * 1024;
    /// Records held before the data directory is known. The interesting failures
    /// happen during startup, so dropping those would defeat the purpose — but
    /// this is bounded, because a logger that can exhaust memory is worse than
    /// no logger.
    const PREBUFFER_LINES: usize = 512;

    #[derive(Default)]
    struct Sink {
        path: Option<PathBuf>,
        pending: Vec<String>,
    }

    pub struct FileLogger {
        sink: Mutex<Sink>,
    }

    pub static LOGGER: FileLogger = FileLogger { sink: Mutex::new(Sink { path: None, pending: Vec::new() }) };

    impl FileLogger {
        /// Point the logger at `<data_dir>/logs/oaiy-desktop.log` and flush
        /// whatever was buffered before the path was known.
        pub fn attach(&self, data_dir: &std::path::Path) {
            let dir = data_dir.join("logs");
            if std::fs::create_dir_all(&dir).is_err() {
                return;
            }
            let path = dir.join("oaiy-desktop.log");
            let Ok(mut sink) = self.sink.lock() else { return };
            let pending = std::mem::take(&mut sink.pending);
            sink.path = Some(path.clone());
            drop(sink);
            for line in pending {
                self.append(&path, &line);
            }
        }

        /// Where the log lives, for the UI to show and open.
        pub fn path(&self) -> Option<PathBuf> {
            self.sink.lock().ok().and_then(|s| s.path.clone())
        }

        fn append(&self, path: &std::path::Path, line: &str) {
            // Roll before writing so the cap is a real ceiling rather than a
            // threshold the last write is allowed to blow past.
            if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) >= MAX_BYTES {
                let _ = std::fs::rename(path, path.with_extension("log.1"));
            }
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{line}");
            }
        }
    }

    impl log::Log for FileLogger {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }

        fn log(&self, record: &log::Record) {
            let line = format!(
                "{} [{}] {}: {}",
                chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                record.level(),
                record.target(),
                record.args()
            );
            let path = {
                let Ok(mut sink) = self.sink.lock() else { return };
                match sink.path.clone() {
                    Some(p) => Some(p),
                    None => {
                        // Not attached yet: hold it, bounded. Keeping the OLDEST
                        // is right — the first failure explains the rest.
                        if sink.pending.len() < PREBUFFER_LINES {
                            sink.pending.push(line.clone());
                        }
                        None
                    }
                }
            };
            if let Some(p) = path {
                self.append(&p, &line);
            }
        }

        fn flush(&self) {}
    }
}

/// Keep a spawned console program from flashing a window.
///
/// The release GUI is built with `windows_subsystem = "windows"`, so the process
/// has no console of its own — and Windows gives a console-subsystem child a
/// brand new visible one unless told otherwise. Every probe this app runs
/// (`nvidia-smi`, `where`, `node -v`) is such a program, and some of them ran
/// per UI poll, so black windows flickered across the user's desktop while they
/// worked. Reported as "it was spawning multiple terminal windows".
///
/// The service runner, plugin host, python installer and CLI worker already do
/// this; these probes were simply missed.
#[cfg(windows)]
fn hide_console(cmd: &mut std::process::Command) {
    use std::os::windows::process::CommandExt as _;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console(_cmd: &mut std::process::Command) {}

/// `cmd.pipe_hidden()` — chainable [`hide_console`], so a probe's spawn stays a
/// single expression instead of needing a `let mut` dance around a cfg block.
pub trait HiddenCommand {
    fn pipe_hidden(&mut self) -> &mut Self;
}

impl HiddenCommand for std::process::Command {
    fn pipe_hidden(&mut self) -> &mut Self {
        hide_console(self);
        self
    }
}

pub mod companion;

// Everything below `open_path` is the GUI companion (Tauri), gated behind the
// default `gui` feature: `cargo build --bin oaiy-server --no-default-features`
// builds the headless server WITHOUT tauri/webkit2gtk. `http` + `services`
// above are tauri-free and shared by both binaries.
#[cfg(feature = "gui")]
mod migrate;
#[cfg(feature = "gui")]
mod tray;
#[cfg(feature = "gui")]
pub use gui::run;

#[cfg(feature = "gui")]
mod gui {
use super::DESKTOP_PORT;
use crate::HiddenCommand as _;
use crate::{http, migrate, tray};
use crate::http::DesktopConfig;
use crate::migrate::{MigratePlan, MigrationHandle, MigrationProgress};
use crate::services::catalog::CatalogHandle;
use crate::services::downloads::{Downloads, DownloadsHandle};
use crate::services::python::{Python, PythonHandle};
use crate::services::registry::{Registry, RegistryHandle};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{Manager, RunEvent, WindowEvent};

/// Resolve a Windows system binary to its absolute `%SystemRoot%` path, so a planted exe of the
/// same name on the process CWD or an early PATH entry can't be launched instead (mirrors
/// services::registry::system32_exe). Falls back to the bare name when SystemRoot is unset or the
/// absolute path is missing (e.g. nvidia-smi on a non-standard NVIDIA install). On non-Windows the
/// bare name is returned for the standard PATH lookup.
fn resolved_system_exe(subdir: &str, rel_win: &str, bare: &str) -> String {
    #[cfg(windows)]
    {
        if let Some(root) = std::env::var_os("SystemRoot") {
            let mut p = PathBuf::from(&root);
            if !subdir.is_empty() {
                p.push(subdir);
            }
            p.push(rel_win);
            if p.exists() {
                return p.display().to_string();
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (subdir, rel_win);
    }
    bare.to_string()
}

/// Tauri command: open a folder in the OS file manager. Surfaced to the
/// React UI via `window.__TAURI_INTERNALS__.invoke('open_path', ...)` and
/// wired to every path the dashboard shows (data dir, models dir, venv
/// dirs, individual model file parents). On Windows this is `explorer
/// /select,<path>` to highlight the file when given a file path, or just
/// `explorer <dir>` for a directory.
#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let p = std::path::PathBuf::from(&path);
    if !p.exists() {
        return Err(format!("path not found: {path}"));
    }

    #[cfg(target_os = "windows")]
    {
        // explorer.exe lives in the Windows dir (not System32); resolve it absolutely so a
        // planted explorer.exe on the CWD/PATH can't be launched instead.
        let explorer = resolved_system_exe("", "explorer.exe", "explorer");
        let result = if p.is_file() {
            // `/select,` and the path MUST be a single argv token, otherwise
            // Explorer treats them as two arguments and never highlights the
            // file. std quotes paths containing spaces, which Explorer accepts.
            std::process::Command::new(&explorer)
                .arg(format!("/select,{path}"))
                .spawn()
        } else {
            std::process::Command::new(&explorer).arg(&path).spawn()
        };
        result.map_err(|e| format!("explorer spawn failed: {e}"))?;
    }
    #[cfg(target_os = "macos")]
    {
        let arg = if p.is_file() {
            vec!["-R".to_string(), path.clone()]
        } else {
            vec![path.clone()]
        };
        std::process::Command::new("open")
            .args(&arg)
            .spawn()
            .map_err(|e| format!("open spawn failed: {e}"))?;
    }
    #[cfg(target_os = "linux")]
    {
        let target = if p.is_file() {
            p.parent().unwrap_or(&p).display().to_string()
        } else {
            path.clone()
        };
        std::process::Command::new("xdg-open")
            .arg(target)
            .spawn()
            .map_err(|e| format!("xdg-open spawn failed: {e}"))?;
    }
    Ok(())
}

/// Open an external URL (e.g. https://oaiy.com) in the system default browser.
/// Unlike `open_path`, this does NOT existence-check — it hands the URL to the
/// OS handler. Guarded to http/https so the UI can't ask us to launch arbitrary
/// schemes. Invoked from the header's oaiy.com link.
#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("only http(s) URLs are allowed".into());
    }
    // Reject ASCII control chars and shell-dangerous metacharacters so the URL
    // can't be used to inject commands into a launcher. '%' is intentionally
    // allowed (valid percent-encoding).
    if url
        .chars()
        .any(|c| c.is_ascii_control() || matches!(c, '"' | '&' | '^' | '|' | '<' | '>' | '`'))
    {
        return Err("URL contains disallowed characters".into());
    }
    #[cfg(target_os = "windows")]
    std::process::Command::new(resolved_system_exe("System32", "rundll32.exe", "rundll32"))
        .args(["url.dll,FileProtocolHandler", &url])
        .spawn()
        .map_err(|e| format!("open url failed: {e}"))?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg(&url)
        .spawn()
        .map_err(|e| format!("open url failed: {e}"))?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open")
        .arg(&url)
        .spawn()
        .map_err(|e| format!("open url failed: {e}"))?;
    Ok(())
}

// DESKTOP_PORT is declared at the crate root (above mod gui) so the headless
// oaiy-server shares it; here it's in scope via `use super::DESKTOP_PORT`.

/// The OS-default data dir (`%APPDATA%/<id>/` on Windows, etc.). This is
/// where everything lives unless the user has chosen a custom folder.
fn default_data_dir(app: &tauri::AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("OAIY"))
}

/// Path to the tiny bootstrap pointer file that records the user's chosen
/// data dir. It MUST live at a fixed OS location (the config dir), never
/// inside the data dir itself — otherwise we couldn't find it after the
/// user relocates their data folder.
fn config_pointer_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join(CONFIG_POINTER_NAME))
}

/// Bootstrap pointer filename.
const CONFIG_POINTER_NAME: &str = "desktop-config.json";

/// Process-wide lock serializing read-modify-write cycles on the pointer
/// file. Every mutating writer (`write_config_str`, `write_extra_model_dirs`,
/// `write_hf_token`) takes this guard BEFORE reading, so concurrent mutations
/// can't read a stale object and clobber each other's keys.
fn pointer_lock() -> &'static Mutex<()> {
    static POINTER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    POINTER_LOCK.get_or_init(|| Mutex::new(()))
}

/// Atomically replace `path`'s contents: write to a sibling `<path>.tmp` then
/// rename over the target (rename is atomic within a dir). Cleans up the temp
/// file on any error so we never leave a partial `.tmp` behind.
fn atomic_write(path: &Path, contents: &str) -> Result<(), String> {
    let tmp = {
        let mut t = path.as_os_str().to_owned();
        t.push(".tmp");
        PathBuf::from(t)
    };
    if let Err(e) = std::fs::write(&tmp, contents) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("write pointer: {e}"));
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("write pointer: {e}"));
    }
    Ok(())
}

/// Read the whole pointer object (BOM-tolerant), or an empty map. The file
/// holds several keys now (`dataDir`, `modelsDir`), so every read/write goes
/// through this to avoid one key clobbering another.
fn read_config_obj(app: &tauri::AppHandle) -> serde_json::Map<String, serde_json::Value> {
    config_pointer_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.strip_prefix('\u{feff}').unwrap_or(&s).to_string())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

/// Read one non-empty string key from the pointer object, if present.
fn read_config_str(app: &tauri::AppHandle, key: &str) -> Option<String> {
    read_config_obj(app)
        .get(key)
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
}

/// Set (Some) or clear (None) one key, PRESERVING the other keys. Deletes
/// the file only when clearing the last key leaves it empty.
fn write_config_str(app: &tauri::AppHandle, key: &str, val: Option<&str>) -> Result<(), String> {
    let _guard = pointer_lock().lock().unwrap_or_else(|e| e.into_inner());
    let p = config_pointer_path(app).ok_or("cannot resolve config dir")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir config dir: {e}"))?;
    }
    let mut obj = read_config_obj(app);
    match val {
        Some(v) if !v.trim().is_empty() => {
            obj.insert(key.to_string(), serde_json::Value::String(v.to_string()));
        }
        _ => {
            obj.remove(key);
        }
    }
    if obj.is_empty() {
        let _ = std::fs::remove_file(&p);
        return Ok(());
    }
    let body = serde_json::Value::Object(obj);
    let pretty = serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string());
    atomic_write(&p, &pretty)
}

/// The user's chosen data dir from the pointer file, if any (else OS default).
fn read_data_dir_override(app: &tauri::AppHandle) -> Option<String> {
    read_config_str(app, "dataDir")
}

/// Write (Some) or clear (None) the data-dir override, keeping other keys.
fn write_data_dir_override(app: &tauri::AppHandle, dir: Option<&str>) -> Result<(), String> {
    write_config_str(app, "dataDir", dir)
}

/// The user's chosen models dir, if set. When unset, models live under
/// `<dataDir>/models` (the default). Kept separate from the data dir so a
/// user can park a big model library on another drive without relocating
/// venvs/templates (which can't move — absolute paths baked into venvs).
fn read_models_dir_override(app: &tauri::AppHandle) -> Option<String> {
    read_config_str(app, "modelsDir")
}

fn write_models_dir_override(app: &tauri::AppHandle, dir: Option<&str>) -> Result<(), String> {
    write_config_str(app, "modelsDir", dir)
}

/// The GGUF a single-model server (llama.cpp) should load, if the user picked
/// one in its Model selector. Unset ⇒ no model selected (no implicit default).
fn read_llama_model_override(app: &tauri::AppHandle) -> Option<String> {
    read_config_str(app, "llamaModel")
}

/// The multimodal projector chosen beside the model, if any.
fn read_llama_mmproj_override(app: &tauri::AppHandle) -> Option<String> {
    read_config_str(app, "llamaMmproj")
}

fn write_llama_mmproj_override(app: &tauri::AppHandle, path: Option<&str>) -> Result<(), String> {
    write_config_str(app, "llamaMmproj", path)
}

fn write_llama_model_override(app: &tauri::AppHandle, model: Option<&str>) -> Result<(), String> {
    write_config_str(app, "llamaModel", model)
}

/// The model NAME a multi-model server (Ollama) should use, if the user picked
/// one in its Model selector. Unset ⇒ the pre-pulled default (qwen2.5:0.5b).
fn read_ollama_model_override(app: &tauri::AppHandle) -> Option<String> {
    read_config_str(app, "ollamaModel")
}

fn write_ollama_model_override(app: &tauri::AppHandle, model: Option<&str>) -> Result<(), String> {
    write_config_str(app, "ollamaModel", model)
}

/// Per-service GPU pins (serviceId → GPU index) the user set in the GPU picker, stored as a
/// JSON object under `serviceGpus`. Applied as CUDA_VISIBLE_DEVICES at start() so heavy
/// services don't all default to GPU 0 and exhaust its VRAM.
fn read_service_gpus(app: &tauri::AppHandle) -> std::collections::HashMap<String, u32> {
    read_config_str(app, "serviceGpus")
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_service_gpus(
    app: &tauri::AppHandle,
    map: &std::collections::HashMap<String, u32>,
) -> Result<(), String> {
    if map.is_empty() {
        return write_config_str(app, "serviceGpus", None);
    }
    let s = serde_json::to_string(map).map_err(|e| format!("serialize serviceGpus: {e}"))?;
    write_config_str(app, "serviceGpus", Some(&s))
}

/// Additional model search roots beyond the primary models dir, stored as a
/// JSON array under `extraModelDirs`. These are read-only weight folders the
/// user registers in Settings (e.g. `E:\ckpts`) so a service can scan several
/// drives via `${modelDirs}` / `OAIY_MODEL_DIRS`. Empty when none configured.
fn read_extra_model_dirs(app: &tauri::AppHandle) -> Vec<String> {
    read_config_obj(app)
        .get("extraModelDirs")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Persist the additional model roots, PRESERVING other config keys. An empty
/// list removes the key (and the file if it was the last key).
fn write_extra_model_dirs(app: &tauri::AppHandle, dirs: &[String]) -> Result<(), String> {
    let _guard = pointer_lock().lock().unwrap_or_else(|e| e.into_inner());
    let p = config_pointer_path(app).ok_or("cannot resolve config dir")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir config dir: {e}"))?;
    }
    let mut obj = read_config_obj(app);
    if dirs.is_empty() {
        obj.remove("extraModelDirs");
    } else {
        obj.insert(
            "extraModelDirs".to_string(),
            serde_json::Value::Array(
                dirs.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
    if obj.is_empty() {
        let _ = std::fs::remove_file(&p);
        return Ok(());
    }
    let body = serde_json::Value::Object(obj);
    let pretty = serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string());
    atomic_write(&p, &pretty)
}

/// Case-insensitive-on-Windows path key for de-duping model dirs (trailing
/// separators ignored). Mirrors `combine_model_dirs` in registry.rs.
fn model_dir_key(s: &str) -> String {
    let s = s.trim().trim_end_matches(['/', '\\']).to_string();
    if cfg!(windows) {
        s.to_lowercase()
    } else {
        s
    }
}

/// Path to the HuggingFace token file. Lives in the FIXED config dir (not
/// the data dir) so it survives a data-folder move, alongside the data-dir
/// pointer. Plaintext, matching the HF CLI's own `~/.cache/huggingface/token`
/// convention for a local single-user tool.
fn hf_token_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("hf-token"))
}

/// Read the saved HuggingFace token, if any (trimmed; BOM-tolerant for
/// hand edits). None when unset/empty.
fn read_hf_token(app: &tauri::AppHandle) -> Option<String> {
    let p = hf_token_path(app)?;
    let s = std::fs::read_to_string(p).ok()?;
    let s = s.strip_prefix('\u{feff}').unwrap_or(&s).trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Persist (Some) or clear (None/empty) the HuggingFace token.
fn write_hf_token(app: &tauri::AppHandle, token: Option<&str>) -> Result<(), String> {
    let _guard = pointer_lock().lock().unwrap_or_else(|e| e.into_inner());
    let p = hf_token_path(app).ok_or("cannot resolve config dir")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir config dir: {e}"))?;
    }
    match token {
        Some(t) if !t.trim().is_empty() => {
            atomic_write(&p, t.trim())?;
        }
        _ => {
            let _ = std::fs::remove_file(&p);
        }
    }
    Ok(())
}

/// Resolve the active per-user data directory. Honours a user-chosen
/// folder from the pointer file (falling back to the OS default if that
/// folder can't be created), so models/venvs/services live wherever the
/// user wants them — somewhere easy to browse, not buried in AppData.
fn resolve_data_dir(app: &tauri::AppHandle) -> PathBuf {
    if let Some(custom) = read_data_dir_override(app) {
        let p = PathBuf::from(&custom);
        if std::fs::create_dir_all(&p).is_ok() {
            return p;
        }
        log::warn!(
            "configured data dir {custom} is unusable; falling back to the default"
        );
    }
    default_data_dir(app)
}

/// Resolve the active models directory: the user's `modelsDir` override
/// (when set + creatable) else `<dataDir>/models`. This is where downloads
/// land and where the install scripts' `OAIY_MODELS_DIR` points.
fn resolve_models_dir(app: &tauri::AppHandle, data_dir: &std::path::Path) -> PathBuf {
    if let Some(custom) = read_models_dir_override(app) {
        let p = PathBuf::from(&custom);
        if std::fs::create_dir_all(&p).is_ok() {
            return p;
        }
        log::warn!("configured models dir {custom} is unusable; falling back to <dataDir>/models");
    }
    data_dir.join("models")
}

// `DesktopConfig` lives in `http.rs`; it's imported at the top of `mod gui`.
// The GUI builds it from AppHandle paths via `config_snapshot` +
// `TauriConfigProvider` below.

/// Build the config snapshot from AppHandle paths. Used by the Tauri command +
/// the GUI's `ConfigProvider`.
pub(crate) fn config_snapshot(app: &tauri::AppHandle, registry: &RegistryHandle) -> DesktopConfig {
    // One lock: pull both the active data dir and the active models dir.
    let (active_dir, models_active_dir) = registry
        .lock()
        .ok()
        .map(|r| {
            (
                r.data_dir().display().to_string(),
                r.models_dir().display().to_string(),
            )
        })
        .unwrap_or_default();
    let default_dir = default_data_dir(app).display().to_string();
    let configured_dir = read_data_dir_override(app);
    let effective = configured_dir.clone().unwrap_or_else(|| default_dir.clone());
    // Normalise trailing separators for the comparison so e.g.
    // "D:\OAIY" and "D:\OAIY\" don't read as a pending change.
    let norm = |s: &str| s.trim_end_matches(['/', '\\']).to_lowercase();

    // Models dir: default is <pending data dir>/models, so the "pending"
    // readout reflects where models will live after a restart that also
    // changes the data dir.
    let models_default_dir = pending_data_dir(app).join("models").display().to_string();
    let models_configured_dir = read_models_dir_override(app);
    let models_effective = models_configured_dir
        .clone()
        .unwrap_or_else(|| models_default_dir.clone());

    DesktopConfig {
        restart_required: norm(&effective) != norm(&active_dir),
        is_custom: configured_dir.is_some(),
        active_dir,
        default_dir,
        configured_dir,
        models_restart_required: norm(&models_effective) != norm(&models_active_dir),
        models_is_custom: models_configured_dir.is_some(),
        models_active_dir,
        models_default_dir,
        models_configured_dir,
        llama_model: read_llama_model_override(app),
        llama_mmproj: read_llama_mmproj_override(app),
        ollama_model: read_ollama_model_override(app),
    }
}

/// AppHandle-backed [`http::ConfigProvider`] for the GUI build — feeds the
/// HTTP `GET /api/config` handler the same snapshot the Tauri command returns.
struct TauriConfigProvider {
    app: tauri::AppHandle,
}

impl http::ConfigProvider for TauriConfigProvider {
    fn snapshot(&self, registry: &RegistryHandle) -> DesktopConfig {
        config_snapshot(&self.app, registry)
    }
}

/// Tauri command: current data-dir configuration for the Settings panel.
#[tauri::command]
fn get_config(app: tauri::AppHandle, registry: tauri::State<RegistryHandle>) -> DesktopConfig {
    config_snapshot(&app, &registry)
}

/// Reject a user-supplied directory that isn't a plain local absolute path BEFORE we touch it.
/// A UNC share (`\\server\share`) or device/verbatim namespace (`\\?\`, `\\.\`) would (a) trigger
/// an outbound SMB connection on the create_dir_all / write-probe / is_dir stat below — leaking the
/// user's NTLM hash and hanging for the SMB timeout — and (b) become a copy-then-delete migration
/// DESTINATION pointing at a network share. Mirrors the guards already on delete_model /
/// run_command_exists. Cross-drive local roots (`D:\`, `E:\ckpts`) stay allowed.
fn validate_local_dir_path(path: &str) -> Result<(), String> {
    use std::path::{Component, Prefix};
    let mut comps = Path::new(path).components();
    match comps.next() {
        // Windows drive path (C:\, D:\) or its extended-length form (\\?\C:\) — both LOCAL. Reject
        // UNC / device / non-disk verbatim prefixes. Require a following RootDir so a drive-RELATIVE
        // path (C:foo, resolved against the process CWD) is refused too.
        Some(Component::Prefix(pre)) => {
            if !matches!(pre.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)) {
                return Err("network or device paths (UNC \\\\server\\share, \\\\?\\, \\\\.\\) aren't allowed — choose a local folder".into());
            }
            if matches!(comps.next(), Some(Component::RootDir)) {
                Ok(())
            } else {
                Err("please choose an absolute folder (e.g. C:\\models), not a drive-relative path".into())
            }
        }
        // POSIX absolute path (/…).
        Some(Component::RootDir) => Ok(()),
        // Relative / empty — refuse rather than persist an ambiguous root.
        _ => Err("please choose an absolute local folder".into()),
    }
}

/// Tauri command: set (or, with an empty string, reset) the data dir.
/// Validates that the folder is creatable + writable before persisting
/// the pointer. Takes effect on the next launch (the UI prompts to
/// restart) — we deliberately don't hot-swap the live registry / in-
/// flight downloads / running service env, which would be error-prone.
#[tauri::command]
fn set_data_dir(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return write_data_dir_override(&app, None);
    }
    validate_local_dir_path(trimmed)?;
    let p = PathBuf::from(trimmed);
    std::fs::create_dir_all(&p).map_err(|e| format!("can't create that folder: {e}"))?;
    // Writability probe — a read-only or permission-denied folder is a
    // common foot-gun; catch it now rather than on first download.
    let probe = p.join(".oaiy-write-test");
    std::fs::write(&probe, b"ok").map_err(|e| format!("that folder isn't writable: {e}"))?;
    let _ = std::fs::remove_file(&probe);
    write_data_dir_override(&app, Some(trimmed))
}

/// Tauri command: set (or, with an empty string, reset to `<dataDir>/models`)
/// the models dir. Same creatable+writable validation as the data dir.
/// Applies on next launch (Downloads + the install scripts' OAIY_MODELS_DIR
/// capture it at startup); the UI prompts to restart.
#[tauri::command]
fn set_models_dir(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return write_models_dir_override(&app, None);
    }
    validate_local_dir_path(trimmed)?;
    let p = PathBuf::from(trimmed);
    std::fs::create_dir_all(&p).map_err(|e| format!("can't create that folder: {e}"))?;
    let probe = p.join(".oaiy-write-test");
    std::fs::write(&probe, b"ok").map_err(|e| format!("that folder isn't writable: {e}"))?;
    let _ = std::fs::remove_file(&probe);
    write_models_dir_override(&app, Some(trimmed))
}

/// Push the extra dirs into the LIVE registry and return the registry-
/// normalized view (`extra_model_dirs()` — deduped, primary stripped) — the
/// exact list `list_model_dirs` reports. Add/remove return this so their reply
/// can't diverge from a subsequent refresh (no "shows then vanishes" entry).
/// Falls back to the raw list only if the registry lock is poisoned.
fn apply_and_list_extra_dirs(registry: &RegistryHandle, dirs: Vec<String>) -> Vec<String> {
    match registry.lock() {
        Ok(mut r) => {
            r.set_extra_model_dirs(dirs.iter().map(PathBuf::from).collect());
            r.extra_model_dirs()
        }
        Err(_) => dirs,
    }
}

/// Tauri command: the additional model folders the user registered (beyond
/// the primary models dir). Read live from the registry so it reflects any
/// add/remove done this session without a restart.
#[tauri::command]
fn list_model_dirs(registry: tauri::State<RegistryHandle>) -> Vec<String> {
    registry
        .lock()
        .map(|r| r.extra_model_dirs())
        .unwrap_or_default()
}

/// Tauri command: register an additional (read-only) model folder. Validates
/// it exists, persists it, and updates the LIVE registry so the next service
/// start sees it via `${modelDirs}` / `OAIY_MODEL_DIRS` — no restart needed.
/// Returns the updated extra-dirs list. A folder that's already registered (or
/// is the primary) is a no-op / error respectively.
#[tauri::command]
fn add_model_dir(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    path: String,
) -> Result<Vec<String>, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("path is empty".into());
    }
    validate_local_dir_path(trimmed)?;
    if !PathBuf::from(trimmed).is_dir() {
        return Err(format!("not a folder: {trimmed}"));
    }
    let target = model_dir_key(trimmed);
    // Don't shadow the primary models dir.
    let primary_key = registry
        .lock()
        .ok()
        .map(|r| model_dir_key(&r.models_dir().display().to_string()));
    if primary_key.as_deref() == Some(target.as_str()) {
        return Err("that's already the primary models folder".into());
    }
    let mut dirs = read_extra_model_dirs(&app);
    // Persist only when it's genuinely new; either way return the normalized view.
    if !dirs.iter().any(|d| model_dir_key(d) == target) {
        dirs.push(trimmed.to_string());
        write_extra_model_dirs(&app, &dirs)?;
    }
    Ok(apply_and_list_extra_dirs(&registry, dirs))
}

/// Tauri command: remove a previously-registered model folder. Persists +
/// updates the live registry. Returns the updated list.
#[tauri::command]
fn remove_model_dir(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    path: String,
) -> Result<Vec<String>, String> {
    let target = model_dir_key(&path);
    let mut dirs = read_extra_model_dirs(&app);
    dirs.retain(|d| model_dir_key(d) != target);
    write_extra_model_dirs(&app, &dirs)?;
    Ok(apply_and_list_extra_dirs(&registry, dirs))
}

/// Tauri command: the loadable GGUFs found across the model search roots —
/// the options for the llama.cpp Model picker.
/// `async` for the same reason as [`list_gpus`]: this walks the model
/// directories, which is instant on an empty install and very much not on a
/// library of 40 GB checkpoints.
#[tauri::command(async)]
fn list_gguf_models(registry: tauri::State<RegistryHandle>) -> Vec<String> {
    registry
        .lock()
        .map(|r| r.list_gguf_models())
        .unwrap_or_default()
}

/// Tauri command: set (or clear, with '') the GGUF a single-model server
/// (llama.cpp) loads. Validates the file exists, persists it, and updates the
/// LIVE registry so the next start loads it via `${llamaModel}` — no restart.
#[tauri::command]
fn set_llama_model(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    path: String,
) -> Result<(), String> {
    let trimmed = path.trim();
    let value = if trimmed.is_empty() { None } else { Some(trimmed) };
    if let Some(v) = value {
        if !PathBuf::from(v).is_file() {
            return Err(format!("not a file: {v}"));
        }
    }
    write_llama_model_override(&app, value)?;
    if let Ok(mut r) = registry.lock() {
        r.set_llama_model(value.map(str::to_string));
    }
    Ok(())
}

/// Tauri command: set (or clear, with '') the multimodal projector loaded
/// beside the llama.cpp model.
///
/// Clearing is a first-class action, not an oversight: a projector costs real
/// VRAM (~1.2 GiB for gemma-4-e2b) and forces llama.cpp's device fitting off,
/// so a user who only wants text should be able to put the model back to
/// text-only without re-picking it.
#[tauri::command]
fn set_llama_mmproj(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    path: String,
) -> Result<(), String> {
    let trimmed = path.trim();
    let value = if trimmed.is_empty() { None } else { Some(trimmed) };
    if let Some(v) = value {
        if !PathBuf::from(v).is_file() {
            return Err(format!("not a file: {v}"));
        }
    }
    write_llama_mmproj_override(&app, value)?;
    if let Ok(mut r) = registry.lock() {
        r.set_llama_mmproj(value.map(str::to_string));
    }
    Ok(())
}

/// Tauri command: the projector files OAIY can see, for the picker.
///
/// The main model list deliberately EXCLUDES `mmproj*` files — a projector is
/// not a model you can run — so they need their own listing or they would be
/// invisible everywhere.
#[tauri::command(async)]
fn list_mmproj_files(registry: tauri::State<RegistryHandle>) -> Vec<String> {
    registry
        .lock()
        .map(|r| r.list_mmproj_files())
        .unwrap_or_default()
}

/// Tauri command: set (or clear, with '') the Ollama model NAME a node uses.
/// Persists it + updates the LIVE registry so the next /api/services snapshot
/// resolves `${ollamaModel}` in the node body — no restart.
#[tauri::command]
fn set_ollama_model(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    model: String,
) -> Result<(), String> {
    let trimmed = model.trim();
    let value = if trimmed.is_empty() { None } else { Some(trimmed) };
    write_ollama_model_override(&app, value)?;
    if let Ok(mut r) = registry.lock() {
        r.set_ollama_model(value.map(str::to_string));
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct GpuInfo {
    index: u32,
    name: String,
}

/// Tauri command: the CUDA GPUs present (index + name), via nvidia-smi. Empty on a box
/// without an NVIDIA GPU / nvidia-smi — the GPU picker then hides itself.
/// `async` so Tauri runs it OFF the main thread: this spawns `nvidia-smi`,
/// which costs ~60ms on a healthy machine and much more on a busy one. Every
/// service card mounts a GPU picker, so on the plain `#[tauri::command]` form
/// that stall landed on the UI thread exactly as the Services panel rendered.
#[tauri::command(async)]
fn list_gpus() -> Vec<GpuInfo> {
    // Hard deadline. `nvidia-smi` is a third-party binary talking to a kernel
    // driver, and it does not always come back: a busy GPU, a half-installed
    // driver, or a hung one leaves it blocked indefinitely. This used to run
    // synchronously in app setup, so a stuck probe meant the HTTP server never
    // bound and OAIY opened as a dead window with no explanation — observed on
    // this machine, and the reason the timeout exists rather than being a
    // theoretical nicety.
    //
    // `stdin` is explicitly null: a child that inherits a console-less parent's
    // stdin can block on a read that will never be answered.
    const GPU_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    let mut child = match std::process::Command::new(resolved_system_exe(
        "System32",
        "nvidia-smi.exe",
        "nvidia-smi",
    ))
    .args(["--query-gpu=index,name", "--format=csv,noheader,nounits"])
    .stdin(std::process::Stdio::null())
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::null())
    .pipe_hidden()
    .spawn()
    {
        Ok(c) => c,
        // No NVIDIA tooling on this box: not an error, just no GPUs to list.
        Err(_) => return Vec::new(),
    };

    let deadline = std::time::Instant::now() + GPU_PROBE_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Ok(None) => {
                // Wedged. Kill it and report no GPUs — a missing GPU list
                // degrades one picker; a blocked probe used to take the whole
                // app down with it.
                let _ = child.kill();
                let _ = child.wait();
                log::warn!("nvidia-smi did not respond within {GPU_PROBE_TIMEOUT:?}; treating this machine as having no NVIDIA GPUs");
                return Vec::new();
            }
            Err(_) => return Vec::new(),
        }
    }

    let out = match child.wait_with_output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ',');
            let index = parts.next()?.trim().parse::<u32>().ok()?;
            let name = parts.next()?.trim().to_string();
            Some(GpuInfo { index, name })
        })
        .collect()
}

/// Tauri command: pin a service to a GPU index (CUDA_VISIBLE_DEVICES), or clear with a null
/// `gpu`. Persists to config + applies to the live registry; takes effect on the next start.
#[tauri::command]
fn set_service_gpu(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    id: String,
    gpu: Option<u32>,
) -> Result<(), String> {
    let mut map = read_service_gpus(&app);
    match gpu {
        Some(n) => {
            map.insert(id.clone(), n);
        }
        None => {
            map.remove(&id);
        }
    }
    write_service_gpus(&app, &map)?;
    if let Ok(mut r) = registry.lock() {
        r.set_service_gpu(&id, gpu);
    }
    Ok(())
}

/// Tauri command: the models pulled into the running Ollama server (its
/// `/api/tags`) — the options for the Ollama Model picker. Empty + an error
/// when Ollama isn't running.
#[tauri::command]
async fn list_ollama_models(
    registry: tauri::State<'_, RegistryHandle>,
) -> Result<Vec<String>, String> {
    let port = registry
        .lock()
        .ok()
        .and_then(|r| r.service_port("ollama"))
        .unwrap_or(11434);
    let url = format!("http://127.0.0.1:{port}/api/tags");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|_| "Ollama isn't reachable — start it first, then refresh.".to_string())?;
    // `.text()` + serde_json avoids needing reqwest's `json` feature.
    let text = resp.text().await.map_err(|e| e.to_string())?;
    let body: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut models: Vec<String> = match body["models"].as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|m| m["name"].as_str().map(String::from))
            .collect(),
        None => Vec::new(),
    };
    models.sort();
    Ok(models)
}

/// Explain a fatal API-server failure to the user, then quit.
///
/// The bind failure that matters in practice is "OAIY is already running" —
/// launching a second copy leaves a window with no API behind it, which looks
/// like the app is simply broken. It is also actively harmful: two instances
/// share one data directory, so two plugin hosts end up fighting over the same
/// hardware (the phone bridge owns a Bluetooth dongle, and only one process can).
///
/// So the second instance identifies WHO holds the port before it complains —
/// another OAIY reads very differently from an unrelated program — and then
/// exits rather than lingering as a broken window.
async fn report_fatal_server_error(app: &tauri::AppHandle, detail: &str) {
    use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

    let ours = port_holder_is_oaiy().await;
    let (title, body) = if ours {
        (
            "OAIY is already running",
            format!(
                "Another copy of OAIY Desktop already has port {DESKTOP_PORT}.\n\n\
                 Only one can run at a time — they share the same data folder, and \
                 two plugin hosts would fight over the same hardware. Use the copy \
                 that is already open (check the system tray).",
            ),
        )
    } else {
        (
            "OAIY could not start its local API",
            format!(
                "Port {DESKTOP_PORT} is being used by another program, so OAIY \
                 cannot serve its local API and nothing in this window will work.\n\n\
                 Close whatever is using that port and start OAIY again.\n\n{detail}"
            ),
        )
    };

    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .message(body)
        .title(title)
        .kind(MessageDialogKind::Error)
        .show(move |_| {
            let _ = tx.send(());
        });
    let _ = rx.await;
    app.exit(1);
}

/// Is the process holding our port another OAIY, or something unrelated?
///
/// Asks the same `/api/health` handshake every client uses — a fixed loopback
/// port is trivially squatted, so identity is asserted rather than assumed.
async fn port_holder_is_oaiy() -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    else {
        return false;
    };
    let url = format!("http://127.0.0.1:{DESKTOP_PORT}/api/health");
    let Ok(resp) = client.get(&url).send().await else {
        return false;
    };
    let Ok(text) = resp.text().await else { return false };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("product").and_then(|p| p.as_str()).map(str::to_string))
        .is_some_and(|product| product == http::PRODUCT_ID)
}

/// Tauri command: where the app's log file is, if logging is attached.
///
/// Exposed so Settings can offer "Open logs" — a log the user cannot find is
/// barely better than no log, and this file is the first thing to ask for when
/// something goes wrong on a machine we cannot reach.
#[tauri::command]
fn log_path() -> Option<String> {
    crate::applog::LOGGER.path().map(|p| p.display().to_string())
}

/// Tauri command: open a native folder picker, returning the chosen path
/// (or None if cancelled). Non-blocking via a oneshot so we never stall
/// the UI thread.
#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |f| {
        let _ = tx.send(f);
    });
    let picked = rx.await.map_err(|e| e.to_string())?;
    Ok(picked.and_then(|fp| fp.as_path().map(|p| p.display().to_string())))
}

/// Tauri command: relaunch the app so a new data/models dir takes effect.
///
/// A packaged build relaunches its own binary and reloads the bundled frontend
/// from disk — clean. Under `tauri dev`, though, `app.restart()` relaunches the
/// binary but the Vite dev server is owned by the `tauri dev` supervisor and is
/// torn down with the old process, so the new window loads a dead
/// `http://localhost:17973` ("Hmmm… can't reach this page"). So in a debug build
/// we DON'T relaunch — we leave the working window in place and tell the user to
/// re-run the dev command to apply the change.
#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    #[cfg(debug_assertions)]
    {
        use tauri_plugin_notification::NotificationExt;
        let _ = app
            .notification()
            .builder()
            .title("Restart needed to apply the change")
            .body(
                "Running under `tauri dev`, so the app can't relaunch itself here \
                 (it would lose the Vite dev server and show \"can't reach this page\"). \
                 Stop and re-run `npm run tauri:dev` to apply. The packaged app restarts \
                 on its own.",
            )
            .show();
    }
    #[cfg(not(debug_assertions))]
    {
        app.restart();
    }
}

/// The data dir the *next* launch will use — the configured override, or
/// the OS default when none/reset. Mirrors `config_snapshot`'s `effective`.
fn pending_data_dir(app: &tauri::AppHandle) -> PathBuf {
    read_data_dir_override(app)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_data_dir(app))
}

/// Tauri command: what a data-folder migration would move (old → pending).
#[tauri::command]
fn migration_plan(app: tauri::AppHandle, registry: tauri::State<RegistryHandle>) -> MigratePlan {
    let old = match registry.lock().ok().map(|r| r.data_dir().to_path_buf()) {
        Some(o) => o,
        None => return MigratePlan::default(),
    };
    migrate::plan(&old, &pending_data_dir(&app))
}

/// Tauri command: start copying/moving the user's data to the pending
/// folder on a background thread. The UI polls `migration_status`.
#[tauri::command]
fn start_migration(
    app: tauri::AppHandle,
    registry: tauri::State<RegistryHandle>,
    migration: tauri::State<MigrationHandle>,
    mode: String,
) -> Result<(), String> {
    let mode = migrate::Mode::parse(&mode).ok_or("mode must be 'copy' or 'move'")?;
    let old = registry
        .lock()
        .map_err(|_| "registry is busy")?
        .data_dir()
        .to_path_buf();
    let new = pending_data_dir(&app);
    // Belt-and-suspenders: reject a network/device destination before any copy+delete, in case a
    // UNC override was persisted before set_data_dir validated it.
    validate_local_dir_path(&new.to_string_lossy())?;
    if !migrate::plan(&old, &new).can_migrate {
        return Err("nothing to migrate — no pending folder change, or the old folder is empty".into());
    }
    {
        let g = migration.lock().map_err(|_| "migration state is busy")?;
        if g.running {
            return Err("a migration is already running".into());
        }
    }
    let handle: MigrationHandle = (*migration).clone();
    std::thread::spawn(move || migrate::run(old, new, mode, handle));
    Ok(())
}

/// Tauri command: current migration progress (polled by the UI).
#[tauri::command]
fn migration_status(migration: tauri::State<MigrationHandle>) -> MigrationProgress {
    migration.lock().map(|g| g.clone()).unwrap_or_default()
}

/// Tauri command: whether a HuggingFace token is currently set. We never
/// hand the token itself back to the UI — only its presence.
#[tauri::command]
fn get_hf_token_status(downloads: tauri::State<DownloadsHandle>) -> bool {
    downloads.has_token()
}

/// Tauri command: set (or clear, with an empty string) the HuggingFace
/// token. Persists it AND updates the live downloader so the next gated
/// download picks it up without a restart.
#[tauri::command]
fn set_hf_token(
    app: tauri::AppHandle,
    downloads: tauri::State<DownloadsHandle>,
    token: String,
) -> Result<(), String> {
    let trimmed = token.trim();
    let value = if trimmed.is_empty() { None } else { Some(trimmed) };
    write_hf_token(&app, value)?;
    downloads.set_token(value.map(str::to_string));
    Ok(())
}

pub fn run() {
    // Before anything else: without this, every log:: call in the GUI went
    // nowhere. main.rs sets `windows_subsystem = "windows"` so there is no
    // console either, which is why a startup hang in this app could only be
    // diagnosed by rebuilding it with eprintln!s. Records emitted before the
    // data directory is known are buffered and flushed by `attach` below.
    let _ = log::set_logger(&crate::applog::LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![
            open_path,
            log_path,
            open_url,
            get_config,
            set_data_dir,
            set_models_dir,
            pick_folder,
            restart_app,
            migration_plan,
            start_migration,
            migration_status,
            get_hf_token_status,
            set_hf_token,
            list_model_dirs,
            add_model_dir,
            remove_model_dir,
            list_gguf_models,
            set_llama_model,
            set_llama_mmproj,
            list_mmproj_files,
            set_ollama_model,
            list_gpus,
            set_service_gpu,
            list_ollama_models
        ])
        .setup(|app| {
            // Build the registry once, share it with both the HTTP server
            // and Tauri-managed state. Failures here are non-fatal — we
            // fall back to an empty registry so the tray still works and
            // the user sees a clear error in the LogsViewer.
            let data_dir = resolve_data_dir(app.handle());
            // Models live under a separately-configurable dir (the user can
            // park a big library on another drive); defaults to <dataDir>/models.
            // The log file lives with the data it describes, and moves when the
            // user relocates the data folder.
            crate::applog::LOGGER.attach(&data_dir);
            log::info!("OAIY Desktop {} starting (data={})", env!("CARGO_PKG_VERSION"), data_dir.display());

            let models_dir = resolve_models_dir(app.handle(), &data_dir);
            // Additional read-only weight folders the user registered (e.g.
            // E:\ckpts) — joined with the primary into the ${modelDirs} search
            // list so a service can scan several drives.
            let extra_model_dirs: Vec<PathBuf> = read_extra_model_dirs(app.handle())
                .into_iter()
                .map(PathBuf::from)
                .collect();
            let registry: RegistryHandle =
                match Registry::init(data_dir.clone(), models_dir.clone(), extra_model_dirs) {
                    Ok(r) => Arc::new(Mutex::new(r)),
                    Err(e) => {
                        log::error!("registry init failed at {}: {e}", data_dir.display());
                        // Empty placeholder; the UI surfaces "no templates" cleanly.
                        let fallback = std::env::temp_dir().join("oaiy-desktop-fallback");
                        let fb_models = fallback.join("models");
                        // Even the temp-dir fallback does filesystem work and can
                        // fail (read-only/full temp). Don't panic — degrade to a
                        // truly in-memory empty registry so the tray/UI still runs.
                        let reg = Registry::init(fallback.clone(), fb_models.clone(), Vec::new())
                            .unwrap_or_else(|e2| {
                                log::error!(
                                    "temp-dir fallback registry init also failed: {e2}; running with an empty in-memory registry"
                                );
                                Registry::empty(fallback, fb_models)
                            });
                        Arc::new(Mutex::new(reg))
                    }
                };
            // Apply the saved llama.cpp model selection (if any) to the live
            // registry so the next flow-triggered start loads it — no restart.
            if let Ok(mut r) = registry.lock() {
                r.set_llama_model(read_llama_model_override(app.handle()));
                r.set_llama_mmproj(read_llama_mmproj_override(app.handle()));
                r.set_ollama_model(read_ollama_model_override(app.handle()));
                // Drop GPU pins to cards that no longer exist (removed / re-imaged box) —
                // otherwise start() would export CUDA_VISIBLE_DEVICES at a missing index and
                // CUDA would see ZERO devices (silent CPU fallback / hard crash). Only prune
                // when enumeration actually succeeds, so a transient nvidia-smi hiccup can't
                // wipe valid pins.
                r.set_service_gpus(read_service_gpus(app.handle()));
                // Backfill install-completion markers for venv services installed before the
                // marker existed, so they don't suddenly read as not-installed.
                r.backfill_install_markers();
            }
            // Build Downloads + Python + Catalog helpers from the
            // registry's data dir so all four share `${dataDir}`
            // consistently. All are Tauri-managed so any future Tauri
            // command can reach them alongside the HTTP layer.
            // Prune GPU pins on a background thread. This used to run inline,
            // and `list_gpus` shells out to nvidia-smi — a third-party binary
            // talking to a kernel driver, which on a busy or half-broken driver
            // simply does not return. Setup then never finished, so the HTTP
            // server never bound and OAIY opened as a window where nothing
            // worked and nothing said why. Observed here; hence both this and
            // the timeout inside `list_gpus`.
            //
            // Nothing needs the result promptly: a stale pin only matters at the
            // next service start, which is a human action seconds away at best.
            {
                let registry_for_gpus = registry.clone();
                let app_for_gpus = app.handle().clone();
                std::thread::spawn(move || {
                    let available = list_gpus();
                    // Only prune when enumeration actually SUCCEEDED, so a
                    // transient nvidia-smi failure cannot wipe valid pins.
                    if available.is_empty() {
                        return;
                    }
                    let valid: std::collections::HashSet<u32> =
                        available.iter().map(|g| g.index).collect();
                    let mut gpus = read_service_gpus(&app_for_gpus);
                    let before = gpus.len();
                    gpus.retain(|_, idx| valid.contains(idx));
                    if gpus.len() != before {
                        log::warn!(
                            "dropped {} GPU pin(s) for device(s) no longer present",
                            before - gpus.len()
                        );
                        let _ = write_service_gpus(&app_for_gpus, &gpus);
                    }
                    if let Ok(mut r) = registry_for_gpus.lock() {
                        r.set_service_gpus(gpus);
                    }
                });
            }

            let downloads: DownloadsHandle = Downloads::new(models_dir.clone()).into_handle();
            // Load the saved HuggingFace token (if any) so gated downloads
            // work from the first launch without re-entering it.
            downloads.set_token(read_hf_token(app.handle()));
            let python: PythonHandle = Python::new(data_dir.clone()).into_handle();
            let catalog = CatalogHandle::new(data_dir.clone());

            app.manage(registry.clone());
            app.manage(downloads.clone());
            app.manage(python.clone());
            app.manage(catalog.clone());
            // Poll-able state for an in-progress data-folder migration.
            let migration: MigrationHandle = Arc::new(Mutex::new(MigrationProgress::default()));
            app.manage(migration);

            // Spawn the localhost HTTP server on its own task. Errors here
            // are non-fatal for the tray itself — the user can still
            // interact with the UI, just no API.
            let app_handle = app.handle().clone();
            // Restore whatever was running before the app last closed, so a
            // reboot doesn't silently take the local runtime offline (a paired
            // page's AI calls would 503 and trigger-fired flows would fail until
            // someone opened this window and clicked Start on each service).
            {
                let started = registry
                    .lock()
                    .map(|mut r| r.autostart_remembered())
                    .unwrap_or_default();
                if !started.is_empty() {
                    log::info!("autostarted {} service(s): {}", started.len(), started.join(", "));
                }
            }

            let registry_for_http = registry.clone();
            let downloads_for_http = downloads.clone();
            let python_for_http = python.clone();
            let catalog_for_http = catalog.clone();
            // Plugins live under the data dir so relocating that folder takes them
            // along. Plugins hold their own state — Aokie keeps phone pairing keys
            // in its data dir — and leaving that behind on a move reads as data loss.
            let plugins_root_for_http = data_dir.join("plugins");
            let data_dir_for_bridge = data_dir.clone();
            let app_for_http = app.handle().clone();
            // A stable per-install id, so run history on a consumer can tell two
            // machines apart. Derived from the data dir rather than minted fresh
            // each launch, which would make every restart look like a new device.
            let device_id_for_http = crate::stable_device_id(&data_dir);
            let config_provider: Arc<dyn http::ConfigProvider> =
                Arc::new(TauriConfigProvider { app: app_handle });
            let app_for_dialog = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Bridge state. The plugins root sits under the data dir so a
                // relocated data folder takes its plugins with it — plugins hold
                // their own state (Aokie keeps phone pairing keys), and leaving
                // that behind on a move would look like data loss.
                // Portable Node lives under the data dir; the worker resolves it
                // (or the system one) per run.
                let node_for_http =
                    crate::services::node_runtime::new_handle(data_dir_for_bridge.clone());
                let bridge_for_http = crate::build_bridge_state(
                    plugins_root_for_http.clone(),
                    data_dir_for_bridge.clone(),
                    device_id_for_http.clone(),
                    Some(node_for_http.clone()),
                );
                // Managed so the Exit arm can stop plugin children alongside
                // services — an orphaned plugin keeps holding its hardware.
                app_for_http.manage(bridge_for_http.host.clone());
                // AI provider store under <data>/ai so it moves with the data dir
                // (like bridge/pairings.json). Holds provider API keys plaintext,
                // guarded by the full/public split — never over the wire.
                let ai_providers_for_http =
                    crate::ai::open_handle(data_dir_for_bridge.join("ai").join("providers.json"));
                // The ChatGPT connector: its OAuth lives in a CODEX_HOME under
                // the data dir, owned by the codex child, not by us.
                let ai_codex_for_http = crate::ai::codex::new_handle(&data_dir_for_bridge);
                if let Err(e) = http::serve(
                    DESKTOP_PORT,
                    config_provider,
                    // GUI: webview-origin auth, plus an OPTIONAL bearer token
                    // (set OAIY_SERVER_TOKEN) so the CLI can drive this companion
                    // without locking out the webview (gui_mode = true below).
                    std::env::var("OAIY_SERVER_TOKEN").ok().filter(|s| !s.is_empty()),
                    true,
                    registry_for_http,
                    downloads_for_http,
                    python_for_http,
                    catalog_for_http,
                    bridge_for_http,
                    data_dir_for_bridge.clone(),
                    ai_providers_for_http,
                    ai_codex_for_http,
                    node_for_http,
                )
                .await
                {
                    log::error!("HTTP server exited: {e}");
                    // A log nobody reads is not a report. Without the API this
                    // window is inert — every panel loads forever and nothing
                    // works — so say what happened and stop, rather than
                    // presenting a running app that silently does nothing.
                    report_fatal_server_error(&app_for_dialog, &e.to_string()).await;
                }
            });

            // Periodically reap exited child processes so the UI status
            // flips from "Running" to "Stopped"/"Errored" within a tick.
            // Cheap — walks only services that think they're running or
            // installing, plus the optional Python install/venv job.
            let registry_for_reaper = registry.clone();
            let python_for_reaper = python.clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    if let Ok(mut reg) = registry_for_reaper.lock() {
                        reg.reap_exited();
                        // Retry anything whose crash backoff has elapsed.
                        reg.run_scheduled_restarts();
                    }
                    python_for_reaper.reap_exited();
                }
            });

            // Health-probe ticker: every 10s, walk Running services, hit
            // their health URLs out-of-lock, then fold results back in.
            // Catches the case where the process spawned fine but didn't
            // actually bind its port (config error, port collision, etc.)
            // — without this, the UI would happily say "Running" forever.
            let registry_for_health = registry.clone();
            tauri::async_runtime::spawn(async move {
                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(3))
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("health: client build failed: {e}");
                        return;
                    }
                };
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    let targets: Vec<(String, String, u64)> =
                        match registry_for_health.lock() {
                            Ok(r) => r.health_targets(),
                            Err(_) => continue,
                        };
                    if targets.is_empty() {
                        continue;
                    }
                    let mut results = Vec::with_capacity(targets.len());
                    for (id, url, timeout) in targets {
                        let req = client
                            .get(&url)
                            .timeout(std::time::Duration::from_secs(timeout.min(10)));
                        let ok = matches!(req.send().await, Ok(r) if r.status().is_success());
                        results.push((id, ok));
                    }
                    if let Ok(mut r) = registry_for_health.lock() {
                        r.apply_health_results(&results);
                    }
                }
            });

            // Build the tray icon + menu. tray::setup hides the main
            // window on close so OAIY Desktop stays alive in the tray.
            tray::setup(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // Hide-on-close instead of quit. The window can be reopened
            // from the tray menu. Quit lives explicitly under tray > Quit.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("error building tauri application")
        .run(|app_handle, event| {
            match event {
                // Keep the app alive only for the IMPLICIT exit (last window
                // closed => code is None) so it stays in the tray. An EXPLICIT
                // app.exit()/app.restart() carries Some(code) and MUST pass through
                // — otherwise tray > Quit and the packaged restart become no-ops and
                // the RunEvent::Exit arm below (which runs stop_all) never fires,
                // orphaning every running service.
                RunEvent::ExitRequested { code, api, .. } => {
                    if code.is_none() {
                        api.prevent_exit();
                    }
                }
                // On real Exit (tray > Quit, or programmatic app.exit),
                // stop every running service so we don't leak orphaned
                // processes. Done synchronously — the user just clicked
                // Quit and is waiting; a few hundred ms is fine.
                RunEvent::Exit => {
                    // Plugins first: they are lighter to stop than model servers,
                    // and a plugin holding hardware (Aokie's dongle) should get
                    // its graceful shutdown before anything slow runs.
                    if let Some(host) = app_handle.try_state::<std::sync::Arc<crate::plugins::PluginHost>>() {
                        log::info!("stopping all plugins on exit");
                        host.stop_all();
                    }
                    if let Some(reg) = app_handle.try_state::<RegistryHandle>() {
                        // Recover from a poisoned mutex — stopping services on exit
                        // matters more than poison-safety (else they're orphaned).
                        let mut r = reg.lock().unwrap_or_else(|e| e.into_inner());
                        log::info!("stopping all services on exit");
                        r.stop_all();
                    }
                }
                _ => {}
            }
        });
}
} // mod gui

/// Build the bridge state both binaries share: ledger, plugin registry + host,
/// trigger store, flow store — and start the flow worker.
///
/// One constructor rather than two hand-rolled blocks, because the two binaries
/// drifting apart here means "works in the GUI, broken headless" (or vice
/// versa), which is the least-tested path by definition.
pub fn build_bridge_state(
    plugins_root: std::path::PathBuf,
    data_dir: std::path::PathBuf,
    device_id: String,
    // Resolves the Node runtime the bundled CLI runs under.
    node: Option<crate::services::node_runtime::NodeHandle>,
) -> bridge::BridgeState {
    // Durable: runs survive a restart, and a run left mid-execution is finalised
    // on reload so a standalone consumer (no ledger of its own) always gets a
    // terminal answer. Under <data>/bridge/ so it moves with a relocated data dir.
    let ledger = bridge::ledger::open_handle(data_dir.join("bridge").join("ledger.jsonl"));
    let plugins = plugins::registry::new_handle(plugins_root);
    let triggers: plugins::TriggerStoreHandle = std::sync::Arc::new(std::sync::Mutex::new(
        plugins::TriggerStore::load(data_dir.join("triggers.json")),
    ));
    // Durable alongside the ledger: an event shed at 3am must still be there in
    // the morning, which is the entire failure this queue exists to fix.
    let dead = bridge::deadletters::open_handle(
        data_dir.join("bridge").join("deadletters.jsonl"),
    );
    let host = plugins::PluginHost::new(
        plugins.clone(),
        ledger.clone(),
        triggers,
        dead.clone(),
        env!("CARGO_PKG_VERSION").to_string(),
        cfg!(debug_assertions),
    );
    let flows = std::sync::Arc::new(bridge::FlowStore::new(data_dir.join("flows")));
    // Durable pairing tokens under <data>/bridge, so a paired consumer stays
    // paired across restarts.
    let pairing = bridge::pairing::open_handle(data_dir.join("bridge").join("pairings.json"));
    // The worker owns nothing the state needs back; its stop flag is dropped
    // deliberately — it runs for the process lifetime, and the ledger being
    // in-memory means there is nothing to hand over on exit.
    let _ = bridge::Worker::start(ledger.clone(), flows.clone(), device_id.clone(), node.clone());
    bridge::BridgeState {
        ledger,
        dead,
        plugins,
        host,
        flows,
        pairing,
        device_id,
        node,
    }
}

/// A stable per-install device id.
///
/// Derived from the data dir path rather than minted per launch: a consumer
/// reads `deviceId` to attribute runs to a machine, and a fresh id every restart
/// would make one PC look like an endless stream of new devices in run history.
/// Hashed rather than sent raw because the path contains the OS username.
pub fn stable_device_id(data_dir: &std::path::Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data_dir.to_string_lossy().to_lowercase().hash(&mut h);
    format!("dev_{:016x}", h.finish())
}
