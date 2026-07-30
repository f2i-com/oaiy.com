//! Registry — in-memory store of templates + their runtime state.
//!
//! On init, seeds the user's config dir with built-in JSON templates
//! and shell scripts (so the user can edit them in place to add new
//! services or tweak existing ones), then loads every *.json under
//! `templates/` into the in-memory map.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::runner::{LogLine, Runner, SpawnConfig};
use super::template::{substitute, InstallSpec, NodeSpec, ServiceTemplate, UninstallSpec};

/// Built-in templates embedded at compile time. Seeded to disk on first
/// run; users can edit the on-disk copies to customise.
const BUILTIN_TEMPLATES: &[(&str, &str)] = &[
    (
        "llama-cpp.json",
        include_str!("../../resources/templates/llama-cpp.json"),
    ),
    (
        "ollama.json",
        include_str!("../../resources/templates/ollama.json"),
    ),
    (
        "playwright-browser.json",
        include_str!("../../resources/templates/playwright-browser.json"),
    ),
    (
        "ltx2-video.json",
        include_str!("../../resources/templates/ltx2-video.json"),
    ),
    (
        "lance.json",
        include_str!("../../resources/templates/lance.json"),
    ),
    (
        "krea2.json",
        include_str!("../../resources/templates/krea2.json"),
    ),
];

const BUILTIN_SCRIPTS: &[(&str, &str)] = &[
    (
        "install-ollama.ps1",
        include_str!("../../resources/scripts/install-ollama.ps1"),
    ),
    // (Python install is now native Rust — see Python::install_runtime in
    // services/python.rs — so there's no install-python.ps1 to seed.)
    (
        "install-playwright.ps1",
        include_str!("../../resources/scripts/install-playwright.ps1"),
    ),
    // The Playwright HTTP server itself — run by the playwright-browser
    // service's run.command. Seeded to scripts/ alongside the installers.
    (
        "playwright_server.py",
        include_str!("../../resources/scripts/playwright_server.py"),
    ),
    // Git-free GitHub-zip fetcher used by the Python-model installers.
    (
        "fetch_zip.py",
        include_str!("../../resources/scripts/fetch_zip.py"),
    ),
    // LTX-2.3 video service: one-click .bat installer + its HTTP server.
    (
        "install-ltx2.bat",
        include_str!("../../resources/scripts/install-ltx2.bat"),
    ),
    (
        "ltx2_server.py",
        include_str!("../../resources/scripts/ltx2_server.py"),
    ),
    // Lance (image+video) service: one-click .bat installer. It runs the
    // repo's own lance_gradio.py; lance_server.py (below) wraps run_task() in a
    // clean JSON API for oaiy-web while still mounting that Gradio UI.
    (
        "install-lance.bat",
        include_str!("../../resources/scripts/install-lance.bat"),
    ),
    // Optional multi-GPU sharding / CPU-offload patch for Lance's gradio
    // server. install-lance.bat applies it after fetch; inert at runtime
    // unless OAIY_LANCE_SHARD=1 / OAIY_LANCE_OFFLOAD=1.
    (
        "patch_lance_sharding.py",
        include_str!("../../resources/scripts/patch_lance_sharding.py"),
    ),
    // Patch config_factory.get_model_path to honor LANCE_MODEL_BASE_DIR for the
    // ViT + Wan VAE (their paths come from a relative "downloads/..." yaml, unlike
    // the main model). Without it, off-repo weights load the LLM but 404 the ViT.
    (
        "patch_lance_paths.py",
        include_str!("../../resources/scripts/patch_lance_paths.py"),
    ),
    // SDPA-backed `flash_attn` shim. Lance hard-imports flash_attn_varlen_func;
    // there's no matching Windows/Blackwell wheel, so install-lance.bat copies
    // this into the venv's site-packages as flash_attn.py (unless
    // OAIY_LANCE_FLASH=1, which builds the real thing).
    (
        "flash_attn_shim.py",
        include_str!("../../resources/scripts/flash_attn_shim.py"),
    ),
    // Thin JSON API (FastAPI) wrapping lance_gradio.run_task so oaiy-web can
    // drive Lance over HTTP; also mounts the Gradio UI at /ui.
    (
        "lance_server.py",
        include_str!("../../resources/scripts/lance_server.py"),
    ),
    // --- Linux install scripts (.sh) — seeded alongside the .ps1/.bat so
    // oaiy-server can install services on a headless Linux host. ollama /
    // playwright / llama-cpp are real installers; ltx2 / lance print manual
    // setup guidance (their GPU installs need a real Linux+CUDA box to port).
    (
        "install-ollama.sh",
        include_str!("../../resources/scripts/install-ollama.sh"),
    ),
    (
        "install-playwright.sh",
        include_str!("../../resources/scripts/install-playwright.sh"),
    ),
    (
        "install-ltx2.sh",
        include_str!("../../resources/scripts/install-ltx2.sh"),
    ),
    (
        "install-lance.sh",
        include_str!("../../resources/scripts/install-lance.sh"),
    ),
    // NOTE: Krea-2 Turbo and Llama.cpp Server ship as SELF-CONTAINED packages --
    // their scripts live inline in the `files` map of their own templates
    // (krea2.json: krea2_server.py / krea2_gguf.py / install-krea2.ps1+.sh;
    // llama-cpp.json: install-llama-cpp.ps1+.sh) and are materialized on load,
    // so they are not listed here.
];

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Stopped,
    Installing,
    Starting,
    Running,
    Errored,
}

/// Runtime status of one service.
#[derive(Clone)]
pub struct ServiceRuntime {
    pub template: ServiceTemplate,
    pub status: ServiceStatus,
    pub error: Option<String>,
    pub runner: Option<Arc<Runner>>,
    /// Transient Runner that owns the in-flight install script process.
    /// Set by `install_streaming`, cleared by `reap_exited` once the
    /// script exits. While set, `logs()` returns its logs (so the UI's
    /// LogsViewer shows install progress without a separate endpoint).
    pub installer: Option<Arc<Runner>>,
    pub port: u16,
    pub last_status_change: DateTime<Utc>,
}

impl ServiceRuntime {
    fn new(template: ServiceTemplate) -> Self {
        let port = template.default_port;
        Self {
            template,
            status: ServiceStatus::Stopped,
            error: None,
            runner: None,
            installer: None,
            port,
            last_status_change: Utc::now(),
        }
    }

    fn set_status(&mut self, s: ServiceStatus, error: Option<String>) {
        if self.status != s || self.error != error {
            self.last_status_change = Utc::now();
        }
        self.status = s;
        self.error = error;
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceSnapshot {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub status: ServiceStatus,
    pub error: Option<String>,
    pub port: u16,
    pub default_port: u16,
    pub pid: Option<u32>,
    pub started_at: Option<DateTime<Utc>>,
    pub last_status_change: DateTime<Utc>,
    pub docs_url: Option<String>,
    pub installable: bool,
    /// True when the template declares an `uninstall` spec — the desktop app
    /// shows an Uninstall button for it.
    pub uninstallable: bool,
    /// True when the service's run executable actually exists on disk (resolved the way
    /// start() does). Lets the UI show ONE button: Install when not installed, else Uninstall.
    pub installed: bool,
    /// The GPU index this service is pinned to (CUDA_VISIBLE_DEVICES), or None for default
    /// placement. Drives the GPU picker on the card.
    pub gpu: Option<u32>,
    /// How to call this service as a flow node (from the template), so the web
    /// app can surface it pre-wired. None → the client falls back to a convention.
    pub node: Option<NodeSpec>,
}

/// Result of `ensure_by_port` — surfaced to oaiy-web so it can tell the
/// user "started Ollama for you" vs "no companion service on that port".
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnsureByPortResult {
    /// A companion service is configured for the requested port.
    pub found: bool,
    /// It was already running (no action taken).
    pub already_running: bool,
    /// We spawned it just now.
    pub started: bool,
    pub id: Option<String>,
    pub name: Option<String>,
    /// Set when found but the spawn failed.
    pub error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegistrySnapshot {
    pub services: Vec<ServiceSnapshot>,
    /// Serialized as `dataDir` — the TS `RegistrySnapshot` interface +
    /// ServicesPanel read camelCase, like every other snapshot struct.
    /// Without the rename this came across as `data_dir` and the
    /// "configs live under …" line + open-folder button got `undefined`.
    pub data_dir: String,
}

/// Combine the primary models dir with extra roots into an ordered, deduped
/// search list (primary first). Skips empties + duplicates (case-insensitive
/// on Windows, where the filesystem is).
fn combine_model_dirs(primary: &Path, extra: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::with_capacity(1 + extra.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Keep this identical to lib.rs `model_dir_key` (trim whitespace, then
    // trailing separators, then lowercase on Windows) so the two dedup layers
    // never disagree about whether two paths are the same.
    let norm = |p: &Path| {
        let s = p.display().to_string();
        let s = s.trim().trim_end_matches(['/', '\\']).to_string();
        if cfg!(windows) {
            s.to_lowercase()
        } else {
            s
        }
    };
    for p in std::iter::once(primary.to_path_buf()).chain(extra) {
        if p.as_os_str().is_empty() {
            continue;
        }
        let k = norm(&p);
        if k.is_empty() || !seen.insert(k) {
            continue;
        }
        out.push(p);
    }
    out
}

/// Join model roots with the OS path separator (`;` on Windows, `:` elsewhere)
/// for the `${modelDirs}` placeholder + `OAIY_MODEL_DIRS` env. Services split on
/// `os.pathsep` to recover the list.
fn join_model_dirs(dirs: &[PathBuf]) -> String {
    let sep = if cfg!(windows) { ";" } else { ":" };
    dirs.iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(sep)
}

/// On Unix, rewrite a Windows venv interpreter path that a template hardcodes
/// (`…/venvs/NAME/Scripts/python.exe`) into its Unix equivalent
/// (`…/venvs/NAME/bin/python`). No-op on Windows. Lets the same templates run a
/// venv-based service on either OS without per-OS run.command variants.
#[cfg(windows)]
fn os_fix_path(s: String) -> String {
    s
}
#[cfg(not(windows))]
fn os_fix_path(s: String) -> String {
    s.replace("\\Scripts\\python.exe", "/bin/python")
        .replace("/Scripts/python.exe", "/bin/python")
        .replace("/Scripts/pythonw.exe", "/bin/python")
}

/// The standard per-user install location an installer like Ollama's `OllamaSetup.exe` drops
/// `<cmd>.exe` into: `%LOCALAPPDATA%\Programs\<cmd>\<cmd>.exe`. Probed directly because such
/// installers add themselves only to the PERSISTED registry PATH — the already-running
/// OAIY Desktop's in-process PATH (captured at launch) doesn't pick that up until a restart, so
/// without this a just-installed ollama reads as "not installed" and refuses to start. (The
/// Windows FS is case-insensitive, so the bare `ollama` resolves the installer's `Ollama` dir.)
#[cfg(windows)]
fn user_programs_exe(cmd: &str) -> Option<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")?;
    let p = Path::new(&local)
        .join("Programs")
        .join(cmd)
        .join(format!("{cmd}.exe"));
    p.exists().then_some(p)
}
#[cfg(not(windows))]
fn user_programs_exe(_cmd: &str) -> Option<PathBuf> {
    None
}

/// Resolve a Windows system binary to its absolute `%SystemRoot%\System32` path so the
/// companion never launches a planted `cmd.exe`/`powershell.exe`/`taskkill.exe` from an
/// attacker-writable working dir or PATH entry (binary-planting defense-in-depth). Falls back
/// to the bare name if SystemRoot is somehow unset.
#[cfg(windows)]
fn system32_exe(rel: &str) -> String {
    std::env::var_os("SystemRoot")
        .map(|root| {
            Path::new(&root)
                .join("System32")
                .join(rel)
                .display()
                .to_string()
        })
        .unwrap_or_else(|| rel.to_string())
}
#[cfg(not(windows))]
fn system32_exe(rel: &str) -> String {
    rel.to_string()
}

/// Normalize a path string for managed-root containment checks: drop trailing
/// separators, and on Windows fold `/`→`\` and lowercase (so a `${dataDir}/x`
/// expansion mixing separators still compares equal to its backslash root).
fn norm_path_key(s: &str) -> String {
    let s = s.trim_end_matches(['/', '\\']);
    if cfg!(windows) {
        s.replace('/', "\\").to_lowercase()
    } else {
        s.to_string()
    }
}

pub struct Registry {
    services: HashMap<String, ServiceRuntime>,
    /// Root config dir — `${dataDir}` placeholder. Contains
    /// `templates/`, `scripts/`, `bin/`, `venvs/` subdirs.
    data_dir: PathBuf,
    /// Resolved models dir — the `${modelsDir}` placeholder + the
    /// `OAIY_MODELS_DIR` install env. Usually `<dataDir>/models`, but can be
    /// a user-chosen folder on another drive (set in Settings).
    models_dir: PathBuf,
    /// All model search roots, primary first: `[models_dir] ++ extra`. Powers
    /// the `${modelDirs}` placeholder + `OAIY_MODEL_DIRS` env (os-pathsep list)
    /// so a service can scan several drives (e.g. `E:\models` AND `E:\ckpts`)
    /// for its weights. Deduped; primary stays index 0 for back-compat.
    model_dirs: Vec<PathBuf>,
    /// The GGUF a single-model server (llama.cpp) should load — the
    /// `${llamaModel}` placeholder. `None` ⇒ no model selected; start() refuses
    /// to spawn (no implicit default). Set live from the Model picker.
    llama_model: Option<String>,
    /// The model NAME a multi-model server (Ollama) should use — substituted
    /// into the node body template as `${ollamaModel}`. `None` ⇒ the small
    /// default the installer pre-pulls (qwen2.5:0.5b). Set live from its picker.
    ollama_model: Option<String>,
    /// Per-service GPU pin: serviceId → GPU index, applied as `CUDA_VISIBLE_DEVICES` at
    /// start() so heavy services don't all default to GPU 0 and thrash. Absent ⇒ the
    /// service's own default (e.g. krea2 keeps DIT on GPU 0 + encoder on GPU 1). Set live
    /// from the GPU picker; takes effect on the next start.
    service_gpus: HashMap<String, u32>,
}

/// Seed a built-in plumbing script, refreshing it when we ship a new
/// version — but never clobbering a copy the user has hand-edited. A
/// hidden `.<name>.seed` snapshot records what we last wrote, so we can
/// tell an untouched auto-seed (safe to refresh) from a user edit (leave
/// alone). Same policy as the model catalog (see catalog.rs):
///   - target missing                 → write file + snapshot
///   - target == snapshot, bundle new → refresh both (ship the fix)
///   - target != snapshot             → user edited; leave it alone
///   - snapshot missing (older build) → treat as old auto-seed; refresh
///     and start tracking. (Scripts are plumbing, rarely hand-edited, so
///     this one-time refresh is an acceptable trade for shipping fixes.)
fn seed_builtin_script(target: &Path, snapshot: &Path, body: &str) -> std::io::Result<()> {
    // Shell scripts must be LF — a CRLF copy (e.g. from a core.autocrlf=true
    // checkout that reached include_str!) breaks under `sh` on Linux. Normalize
    // .sh bodies to LF defensively; the repo .gitattributes is the primary guard.
    let lf;
    let body: &str = if target.extension().and_then(|e| e.to_str()) == Some("sh") {
        lf = body.replace("\r\n", "\n");
        &lf
    } else {
        body
    };
    match std::fs::read_to_string(target).ok() {
        None => {
            std::fs::write(target, body)?;
            let _ = std::fs::write(snapshot, body);
        }
        Some(current) => match std::fs::read_to_string(snapshot).ok() {
            Some(snap) => {
                if current.trim() == snap.trim() && current.trim() != body.trim() {
                    std::fs::write(target, body)?;
                    let _ = std::fs::write(snapshot, body);
                }
            }
            None => {
                std::fs::write(target, body)?;
                let _ = std::fs::write(snapshot, body);
            }
        },
    }
    Ok(())
}

/// A package-file name is safe iff it's a bare filename (no path separator / `..` /
/// drive prefix) and not hidden — package files live directly in `scripts/`.
fn is_safe_script_name(name: &str) -> bool {
    Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|f| f == name)
        .unwrap_or(false)
        && !name.starts_with('.')
}

/// Script names OWNED by the built-ins: every `BUILTIN_SCRIPTS` key plus every file a built-in
/// TEMPLATE bundles (krea2_server.py, install-krea2.ps1, …). An imported package must never be
/// allowed to write any of these — they're seeded from the trusted compiled-in source, and
/// overwriting one would trojan a DIFFERENT, trusted service (RCE the next time the user
/// installs/starts it).
fn reserved_script_names() -> &'static std::collections::HashSet<String> {
    static SET: std::sync::OnceLock<std::collections::HashSet<String>> = std::sync::OnceLock::new();
    SET.get_or_init(|| {
        let mut s = std::collections::HashSet::new();
        for (name, _) in BUILTIN_SCRIPTS {
            s.insert((*name).to_string());
        }
        for (_, body) in BUILTIN_TEMPLATES {
            if let Ok(t) = serde_json::from_str::<ServiceTemplate>(body) {
                for k in t.files.keys() {
                    s.insert(k.clone());
                }
            }
        }
        s
    })
}

/// Seed every built-in template's bundled `files` from the COMPILED-IN source (trusted), so the
/// reserved-name guard in `materialize_package_files` can then refuse an imported package from
/// overwriting them.
fn seed_builtin_template_files(scripts_dir: &Path) {
    for (_, body) in BUILTIN_TEMPLATES {
        if let Ok(t) = serde_json::from_str::<ServiceTemplate>(body) {
            for (name, fbody) in &t.files {
                if !is_safe_script_name(name) {
                    continue;
                }
                let target = scripts_dir.join(name);
                let snap = scripts_dir.join(format!(".{name}.seed"));
                if let Err(e) = seed_builtin_script(&target, &snap, fbody) {
                    log::warn!("could not seed built-in template file {name}: {e}");
                }
            }
        }
    }
}

/// Write a NON-BUILT-IN (imported) package's bundled `files` into the shared scripts dir so its
/// install/run commands resolve. SECURITY: refuses any name owned by a built-in script/template
/// (so a malicious import can't overwrite a trusted service's script — cross-service RCE) and
/// any non-reserved name already owned by a DIFFERENT package (tracked via a `.<name>.owner`
/// sidecar). Built-in templates' own files are seeded by `seed_builtin_template_files`, not here.
fn materialize_package_files(scripts_dir: &Path, template_id: &str, files: &HashMap<String, String>) {
    for (name, body) in files {
        if !is_safe_script_name(name) {
            log::warn!("skipping package file with unsafe/hidden name: {name}");
            continue;
        }
        if reserved_script_names().contains(name) {
            log::warn!(
                "template '{template_id}': refusing to overwrite built-in script '{name}'"
            );
            continue;
        }
        let target = scripts_dir.join(name);
        let owner = scripts_dir.join(format!(".{name}.owner"));
        // Don't clobber a non-reserved script another package (or an unmarked pre-existing
        // file) owns — only refresh one this same template wrote.
        if target.exists()
            && std::fs::read_to_string(&owner)
                .ok()
                .as_deref()
                .map(str::trim)
                != Some(template_id)
        {
            log::warn!(
                "template '{template_id}': refusing to overwrite script '{name}' owned by another package"
            );
            continue;
        }
        let snap = scripts_dir.join(format!(".{name}.seed"));
        match seed_builtin_script(&target, &snap, body) {
            Ok(()) => {
                let _ = std::fs::write(&owner, template_id);
            }
            Err(e) => log::warn!("could not materialize package file {name}: {e}"),
        }
    }
}

impl Registry {
    /// Seed user dir if empty and load every template. `models_dir` is the
    /// resolved downloads/weights root (override or `<dataDir>/models`);
    /// `extra_model_dirs` are additional read-only search roots (e.g.
    /// `E:\ckpts`) the user registered in Settings — together they form the
    /// `${modelDirs}` / `OAIY_MODEL_DIRS` search list.
    pub fn init(
        data_dir: PathBuf,
        models_dir: PathBuf,
        extra_model_dirs: Vec<PathBuf>,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(data_dir.join("templates"))?;
        std::fs::create_dir_all(data_dir.join("scripts"))?;
        std::fs::create_dir_all(data_dir.join("bin"))?;
        std::fs::create_dir_all(&models_dir)?;

        // Templates re-seed on change via a snapshot — same policy as the
        // scripts below (seed_builtin_script): ship fixes to built-in
        // templates without clobbering a copy the user has hand-edited. A
        // missing snapshot (older build that used write-if-not-exists) is
        // treated as an untouched auto-seed and refreshed once, then tracked
        // — so e.g. a corrected lance.json reaches existing installs.
        for (name, body) in BUILTIN_TEMPLATES {
            let p = data_dir.join("templates").join(name);
            let snap = data_dir.join("templates").join(format!(".{name}.seed"));
            if let Err(e) = seed_builtin_script(&p, &snap, body) {
                log::warn!("could not seed built-in template {name}: {e}");
            }
        }
        // Scripts are plumbing (install .ps1 + the playwright server .py),
        // not user-facing config like templates. We still ship fixes to
        // them, so — unlike templates — re-seed on change via a snapshot,
        // never clobbering a copy the user has actually edited. Mirrors the
        // catalog's seed-snapshot policy (catalog.rs).
        for (name, body) in BUILTIN_SCRIPTS {
            let p = data_dir.join("scripts").join(name);
            let snap = data_dir.join("scripts").join(format!(".{name}.seed"));
            if let Err(e) = seed_builtin_script(&p, &snap, body) {
                log::warn!("could not seed built-in script {name}: {e}");
            }
        }
        // Seed built-in TEMPLATES' bundled files (krea2_server.py, install-krea2.ps1, …) from
        // the trusted compiled-in source, so the reserved-name guard in
        // materialize_package_files refuses any imported package from overwriting them.
        seed_builtin_template_files(&data_dir.join("scripts"));

        let mut services = HashMap::new();
        for entry in std::fs::read_dir(data_dir.join("templates"))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(&path)?;
            match serde_json::from_str::<ServiceTemplate>(&raw) {
                Ok(t) => {
                    // Self-contained packages carry their scripts inline; lay
                    // them down so install/run can find them by name.
                    if !t.files.is_empty() {
                        materialize_package_files(&data_dir.join("scripts"), &t.id, &t.files);
                    }
                    let id = t.id.clone();
                    services.insert(id, ServiceRuntime::new(t));
                }
                Err(e) => {
                    log::warn!("skipping bad template {}: {}", path.display(), e);
                }
            }
        }

        log::info!(
            "registry loaded {} service(s) from {}",
            services.len(),
            data_dir.display()
        );
        let model_dirs = combine_model_dirs(&models_dir, extra_model_dirs);
        Ok(Self {
            services,
            data_dir,
            models_dir,
            model_dirs,
            llama_model: None,
            ollama_model: None,
            service_gpus: HashMap::new(),
        })
    }

    /// Last-resort in-memory registry with NO templates and NO filesystem work.
    /// Used when both the data dir AND the temp-dir fallback are unwritable, so
    /// the tray/UI still comes up (degraded) instead of panicking at startup.
    pub fn empty(data_dir: PathBuf, models_dir: PathBuf) -> Self {
        let model_dirs = combine_model_dirs(&models_dir, Vec::new());
        Self {
            services: HashMap::new(),
            data_dir,
            models_dir,
            model_dirs,
            llama_model: None,
            ollama_model: None,
            service_gpus: HashMap::new(),
        }
    }

    /// Root config dir. Used by the gui-feature config/migration Tauri commands;
    /// the `#[allow(dead_code)]` keeps the headless (`--no-default-features`)
    /// build — where those callers are cfg'd out — warning-free.
    #[allow(dead_code)]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Resolved models dir (override or `<dataDir>/models`). Used by the
    /// config snapshot to report where models actually live.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// All model search roots, primary first. Powers `${modelDirs}`.
    /// (Exposed for completeness alongside `extra_model_dirs()`; the env/ctx
    /// paths read the field directly via `join_model_dirs`.)
    #[allow(dead_code)]
    pub fn model_dirs(&self) -> &[PathBuf] {
        &self.model_dirs
    }

    /// The additional roots beyond the primary `models_dir` (what Settings
    /// shows + edits). Just `model_dirs[1..]`.
    pub fn extra_model_dirs(&self) -> Vec<String> {
        self.model_dirs
            .iter()
            .skip(1)
            .map(|p| p.display().to_string())
            .collect()
    }

    /// Replace the extra search roots (live — the next service start picks
    /// them up via `${modelDirs}` / `OAIY_MODEL_DIRS`). Primary stays index 0.
    pub fn set_extra_model_dirs(&mut self, extra: Vec<PathBuf>) {
        self.model_dirs = combine_model_dirs(&self.models_dir, extra);
    }

    /// The configured single-model override (`${llamaModel}`), if any. `None`
    /// means no model is selected (no implicit default; start() refuses to spawn).
    pub fn llama_model(&self) -> Option<String> {
        self.llama_model.clone()
    }

    /// Set (or clear) the GGUF a single-model server loads. Live — the next
    /// service start reads it via `${llamaModel}`, no restart needed.
    pub fn set_llama_model(&mut self, model: Option<String>) {
        self.llama_model = clean_model_opt(model);
    }

    /// The configured Ollama model name (`${ollamaModel}`), if any.
    pub fn ollama_model(&self) -> Option<String> {
        self.ollama_model.clone()
    }

    /// Set (or clear) the Ollama model name. Live — the next /api/services
    /// snapshot resolves `${ollamaModel}` in the node body, no restart needed.
    pub fn set_ollama_model(&mut self, model: Option<String>) {
        self.ollama_model = clean_model_opt(model);
    }

    /// The GPU index pinned for a service, if any (`None` ⇒ default placement).
    pub fn service_gpu(&self, id: &str) -> Option<u32> {
        self.service_gpus.get(id).copied()
    }

    /// Pin a service to a GPU index (or clear with `None`). Applied as
    /// `CUDA_VISIBLE_DEVICES` on the next start(); no effect on a running process.
    pub fn set_service_gpu(&mut self, id: &str, gpu: Option<u32>) {
        match gpu {
            Some(n) => {
                self.service_gpus.insert(id.to_string(), n);
            }
            None => {
                self.service_gpus.remove(id);
            }
        }
    }

    /// Replace the whole per-service GPU map (loads the saved config at startup).
    pub fn set_service_gpus(&mut self, map: HashMap<String, u32>) {
        self.service_gpus = map;
    }

    /// The live port of a service by id (e.g. to query Ollama's /api/tags for
    /// the list of pulled models).
    pub fn service_port(&self, id: &str) -> Option<u16> {
        self.services.get(id).map(|s| s.port)
    }

    /// Every loadable *.gguf at the top level of any model search root
    /// (primary + extras), deduped + sorted. Excludes multimodal projector
    /// files (`mmproj*`), which aren't a standalone model. Powers the
    /// llama.cpp Model picker.
    pub fn list_gguf_models(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for dir in &self.model_dirs {
            let Ok(rd) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in rd.flatten() {
                let p = entry.path();
                let is_gguf = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(false);
                if !is_gguf {
                    continue;
                }
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.to_ascii_lowercase().starts_with("mmproj") {
                    continue;
                }
                let s = p.display().to_string();
                if seen.insert(s.clone()) {
                    out.push(s);
                }
            }
        }
        out.sort();
        out
    }

    /// True when `p` resolves inside a managed root (`${dataDir}` — which
    /// contains bin/scripts/venvs/services — or any model dir). The uninstall
    /// guard so a template's declared paths can't reach outside what OAIY owns.
    fn path_within_managed_root(&self, p: &Path) -> bool {
        let target = norm_path_key(&p.display().to_string());
        if target.is_empty() {
            return false;
        }
        std::iter::once(&self.data_dir)
            .chain(self.model_dirs.iter())
            .any(|root| {
                let r = norm_path_key(&root.display().to_string());
                !r.is_empty()
                    && (target == r
                        || target.starts_with(&format!("{r}{}", std::path::MAIN_SEPARATOR)))
            })
    }

    /// True when `p` is EXACTLY one of the managed roots (data dir or a model
    /// dir). Used to refuse a glob sitting directly in a root, which would
    /// enumerate the whole root.
    fn is_managed_root(&self, p: &Path) -> bool {
        let t = norm_path_key(&p.display().to_string());
        !t.is_empty()
            && std::iter::once(&self.data_dir)
                .chain(self.model_dirs.iter())
                .any(|r| norm_path_key(&r.display().to_string()) == t)
    }

    /// True when `p` is EXACTLY a managed root OR a SHARED structural dir under the data dir
    /// (`bin`/`venvs`/`services`/`templates`/`scripts`). Uninstall must refuse these: removing
    /// one would wipe EVERY service's files (all venvs, all binaries, all templates), not just
    /// this service's. Paths strictly UNDER them (e.g. `${dataDir}/venvs/<id>`,
    /// `${binDir}/llama-*.exe`) are fine — this is an equality check, so they pass through.
    fn is_protected_uninstall_root(&self, p: &Path) -> bool {
        if self.is_managed_root(p) {
            return true;
        }
        let t = norm_path_key(&p.display().to_string());
        !t.is_empty()
            && ["bin", "venvs", "services", "templates", "scripts"]
                .iter()
                .any(|sub| norm_path_key(&self.data_dir.join(sub).display().to_string()) == t)
    }

    /// Remove the files/dirs a service's `uninstall` spec declares, so the user
    /// can clean-reinstall (e.g. swap an old llama.cpp build for a new one).
    /// Each path is placeholder-expanded; a `*` in the final segment globs that
    /// dir. Every path is guarded to stay inside a managed root and to contain
    /// no `..`. The service must be stopped. Returns the count removed.
    pub fn uninstall(&mut self, id: &str) -> Result<usize, String> {
        let (status, spec, port) = {
            let svc = self.services.get(id).ok_or("unknown service")?;
            (svc.status, svc.template.uninstall.clone(), svc.port)
        };
        if matches!(status, ServiceStatus::Running | ServiceStatus::Starting) {
            return Err("stop the service before uninstalling it".into());
        }
        let spec: UninstallSpec = spec.ok_or("this service has no uninstall defined")?;

        // Glob-aware single-segment removal (prefix*suffix on the file name).
        fn remove_matching(resolved: &str) -> usize {
            let p = Path::new(resolved);
            let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if fname.contains('*') {
                let dir = p.parent().unwrap_or_else(|| Path::new("."));
                let (pre, suf) = fname.split_once('*').unwrap_or((fname, ""));
                // A bare `*` (empty prefix AND suffix) matches EVERY entry in the
                // dir — refuse it so an over-broad spec can't wipe a whole dir.
                if pre.is_empty() && suf.is_empty() {
                    return 0;
                }
                let mut n = 0;
                if let Ok(rd) = std::fs::read_dir(dir) {
                    for e in rd.flatten() {
                        let nm = e.file_name();
                        let nm = nm.to_str().unwrap_or("");
                        if nm.len() >= pre.len() + suf.len()
                            && nm.starts_with(pre)
                            && nm.ends_with(suf)
                        {
                            let t = e.path();
                            let ok = if t.is_dir() {
                                std::fs::remove_dir_all(&t).is_ok()
                            } else {
                                std::fs::remove_file(&t).is_ok()
                            };
                            if ok {
                                n += 1;
                            }
                        }
                    }
                }
                n
            } else if p.is_dir() {
                usize::from(std::fs::remove_dir_all(p).is_ok())
            } else if p.is_file() {
                usize::from(std::fs::remove_file(p).is_ok())
            } else {
                0
            }
        }

        let ctx = self.ctx(port);
        let mut removed = 0usize;
        for raw in &spec.paths {
            let resolved = os_fix_path(substitute(raw, &ctx));
            if resolved.split(['/', '\\']).any(|c| c == "..") {
                log::warn!("uninstall {id}: skipping path with '..': {resolved}");
                continue;
            }
            if !self.path_within_managed_root(Path::new(&resolved)) {
                log::warn!("uninstall {id}: skipping out-of-root path: {resolved}");
                continue;
            }
            // Refuse a path that IS a managed root or a shared structural dir (${dataDir},
            // ${modelsDir}, ${binDir}, ${dataDir}/venvs, …). remove_dir_all on one of these
            // would nuke every OTHER service's files + the model library — a hostile or buggy
            // template's "Uninstall <ServiceX>" must only touch ServiceX's own subtree.
            if self.is_protected_uninstall_root(Path::new(&resolved)) {
                log::warn!("uninstall {id}: refusing managed-root/structural path: {resolved}");
                continue;
            }
            // Refuse a glob sitting DIRECTLY in a managed root (e.g. `${modelsDir}/*`
            // or `${modelsDir}/Q*`) — it could enumerate the user's whole library.
            let pb = Path::new(&resolved);
            if pb.file_name().and_then(|n| n.to_str()).is_some_and(|f| f.contains('*'))
                && pb.parent().is_some_and(|par| self.is_managed_root(par))
            {
                log::warn!("uninstall {id}: refusing root-level glob: {resolved}");
                continue;
            }
            removed += remove_matching(&resolved);
        }
        if let Some(svc) = self.services.get_mut(id) {
            svc.set_status(ServiceStatus::Stopped, None);
            svc.error = None;
        }
        log::info!("uninstall {id}: removed {removed} item(s)");
        Ok(removed)
    }

    /// Whether the service's run executable exists on disk. Mirrors start()'s resolution: a
    /// path command must exist; a bare command resolves to ${binDir} first, then PATH (e.g. a
    /// system-installed `ollama`). The fallback "is it installed" signal for a service with no
    /// install-completion marker.
    fn run_command_exists(&self, run_command: &str, port: u16) -> bool {
        let raw = os_fix_path(substitute(run_command, &self.ctx(port)));
        if raw.contains(std::path::is_separator) {
            // NEVER stat a UNC / network path. run.command is attacker-controlled by an
            // imported template, and this runs on every (unprivileged) GET /api/services
            // poll before the service is ever started. A value like `\\attacker\share\x`
            // would make .exists() do an outbound SMB/WebDAV stat — leaking the user's NTLM
            // hash and hanging for the SMB timeout while holding the registry lock (DoS).
            if raw.starts_with("\\\\") || raw.starts_with("//") {
                return false;
            }
            return Path::new(&raw).exists();
        }
        let bin = self.data_dir.join("bin").join(&raw);
        if bin.with_extension("exe").exists() || bin.exists() {
            return true;
        }
        std::env::var("PATH").is_ok_and(|path| {
            std::env::split_paths(&path).any(|dir| {
                let p = dir.join(&raw);
                p.exists() || p.with_extension("exe").exists()
            })
        }) || user_programs_exe(&raw).is_some()
    }

    /// The resolved path of a service's install-completion marker, if it declares one.
    fn install_marker_path(&self, template: &ServiceTemplate, port: u16) -> Option<String> {
        template
            .installed_marker
            .as_ref()
            .map(|m| os_fix_path(substitute(m, &self.ctx(port))))
    }

    /// Whether the service is installed. When the template declares an `installedMarker` — a
    /// venv service whose interpreter exists after step 1 of a multi-GB install, BEFORE the
    /// pip/download that can fail — that marker (written only when the installer exits 0, see
    /// reap_exited) is the source of truth, so a partial/failed install correctly reads as
    /// not-installed (Start hidden, Install shown). Otherwise fall back to "run exe exists".
    fn run_installed(&self, template: &ServiceTemplate, port: u16) -> bool {
        if let Some(marker) = self.install_marker_path(template, port) {
            return Path::new(&marker).exists();
        }
        self.run_command_exists(&template.run.command, port)
    }

    /// One-time migration for installs that predate the install-completion marker: their venv
    /// interpreter exists but the marker doesn't, so they'd wrongly read as not-installed.
    /// Backfill the marker when the run executable IS present (the prior "installed" heuristic)
    /// but the marker is missing — preserving existing installs' behavior, while NEW installs
    /// get the accurate marker from reap_exited (so a partial install reads as not-installed).
    pub fn backfill_install_markers(&self) {
        // ONE-TIME migration, gated by a persisted sentinel. WITHOUT this gate backfill re-runs
        // every launch and — because run_command_exists only sees the venv interpreter, created
        // at install step 1 — would re-bless a NEW failed install (interpreter present, marker
        // absent) as installed on the next restart, defeating the marker's whole purpose. A
        // pre-feature install already read installed under the old run-exe heuristic, so
        // migrating it ONCE is behavior-preserving; afterwards only reap_exited (installer exit
        // 0) ever writes a marker, so a post-migration partial install stays not-installed.
        let sentinel = self.data_dir.join(".install-markers-backfilled");
        if sentinel.exists() {
            return;
        }
        for svc in self.services.values() {
            let Some(marker) = self.install_marker_path(&svc.template, svc.port) else {
                continue;
            };
            if Path::new(&marker).exists()
                || !self.run_command_exists(&svc.template.run.command, svc.port)
            {
                continue;
            }
            if let Some(parent) = Path::new(&marker).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&marker, "installed by oaiy\n");
        }
        let _ = std::fs::write(&sentinel, "");
    }

    pub fn snapshot(&self) -> RegistrySnapshot {
        let mut services: Vec<ServiceSnapshot> = self
            .services
            .values()
            .map(|s| ServiceSnapshot {
                id: s.template.id.clone(),
                name: s.template.name.clone(),
                description: s.template.description.clone(),
                category: s.template.category.clone(),
                status: s.status,
                error: s.error.clone(),
                port: s.port,
                default_port: s.template.default_port,
                pid: s.runner.as_ref().map(|r| r.pid),
                started_at: s.runner.as_ref().map(|r| r.started_at),
                last_status_change: s.last_status_change,
                docs_url: s.template.docs_url.clone(),
                installable: !matches!(s.template.install, InstallSpec::None),
                uninstallable: s.template.uninstall.is_some(),
                installed: self.run_installed(&s.template, s.port),
                gpu: self.service_gpus.get(&s.template.id).copied(),
                node: s.template.node.as_ref().map(|n| {
                    // Resolve companion-side `${...}` placeholders (e.g.
                    // `${ollamaModel}`) in the node body BEFORE the web app sees
                    // it — the web compiler only handles `{{...}}` vars.
                    let mut n = n.clone();
                    if let Some(bt) = &n.body_template {
                        n.body_template = Some(substitute(bt, &self.ctx(s.port)));
                    }
                    n
                }),
            })
            .collect();
        services.sort_by(|a, b| a.category.cmp(&b.category).then(a.name.cmp(&b.name)));
        RegistrySnapshot {
            services,
            data_dir: self.data_dir.display().to_string(),
        }
    }

    /// Map venv name → ids of services whose run spec references it
    /// (`.../venvs/<name>/...`). Powers the Python tab's "used by …" line so
    /// a venv shows which services depend on it before you delete it. The
    /// Python module doesn't know about the registry, so the HTTP layer
    /// folds this into the python snapshot's `bound_services`.
    pub fn venv_usage(&self) -> HashMap<String, Vec<String>> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        for (id, svc) in &self.services {
            let run = &svc.template.run;
            let mut fields: Vec<&str> = vec![run.command.as_str()];
            fields.extend(run.args.iter().map(|s| s.as_str()));
            fields.extend(run.env.values().map(|s| s.as_str()));
            if let Some(c) = &run.cwd {
                fields.push(c.as_str());
            }
            let mut seen = std::collections::HashSet::new();
            for f in fields {
                if let Some(name) = venv_name_in(f) {
                    if seen.insert(name.clone()) {
                        out.entry(name).or_default().push(id.clone());
                    }
                }
            }
        }
        for ids in out.values_mut() {
            ids.sort();
        }
        out
    }

    /// Recent log lines for `id`. While an install is in flight the
    /// installer's logs take precedence (that's what the user is staring
    /// at the LogsViewer for). Once install completes the installer is
    /// dropped and we fall back to the main service runner's logs.
    pub fn logs(&self, id: &str, tail: Option<usize>) -> Option<Vec<LogLine>> {
        let svc = self.services.get(id)?;
        // A live service shows its own process logs.
        if matches!(svc.status, ServiceStatus::Running | ServiceStatus::Starting) {
            if let Some(r) = &svc.runner {
                return Some(r.logs.snapshot(tail));
            }
        }
        // While installing, the installer's output is what the user is watching
        // (a stale runner from a previous run must not shadow it).
        if svc.status == ServiceStatus::Installing {
            if let Some(installer) = &svc.installer {
                return Some(installer.logs.snapshot(tail));
            }
        }
        // Stopped / Errored: prefer the runner if it actually produced output —
        // that's the most recent activity (e.g. a crash traceback). We KEEP the
        // runner after exit now, so crash logs survive. Fall back to the
        // installer (an install that failed before the service ever ran), which
        // we also KEEP after it exits.
        if let Some(r) = &svc.runner {
            let snap = r.logs.snapshot(tail);
            if !snap.is_empty() {
                return Some(snap);
            }
        }
        if let Some(installer) = &svc.installer {
            return Some(installer.logs.snapshot(tail));
        }
        None
    }

    /// Build the placeholder context for substitute().
    fn ctx(&self, port: u16) -> HashMap<&'static str, String> {
        let mut m = HashMap::new();
        m.insert("port", port.to_string());
        m.insert("dataDir", self.data_dir.display().to_string());
        m.insert("binDir", self.data_dir.join("bin").display().to_string());
        m.insert("modelsDir", self.models_dir.display().to_string());
        // Where a self-contained package's `files` are materialized, so its
        // run.command/args/env/cwd can reference bundled scripts by
        // `${scriptsDir}/name` (matches materialize_package_files' target dir).
        m.insert("scriptsDir", self.data_dir.join("scripts").display().to_string());
        // All search roots joined by the OS path separator (`;` on Windows),
        // so a service env like `"LTX2_MODEL_DIRS": "${modelDirs}"` can scan
        // several drives. Primary is always first.
        m.insert("modelDirs", join_model_dirs(&self.model_dirs));
        // Single-model LLM servers (llama.cpp) load ONE gguf via `-m
        // ${llamaModel}`. Inserted ONLY when the user has explicitly picked a
        // model — there is NO implicit default. Left unset, `${llamaModel}`
        // stays unsubstituted and start() refuses to spawn (a clear "pick a
        // model" error) rather than guessing a `model.gguf` that may not exist.
        if let Some(model) = self.llama_model.clone().filter(|s| !s.trim().is_empty()) {
            m.insert("llamaModel", model);
        }
        // Multi-model servers (Ollama) take a model NAME per request; the
        // companion resolves the user's pick into the node body template via
        // `${ollamaModel}`. ALWAYS set (with the pre-pulled default) so a node
        // never ships a literal `${ollamaModel}`.
        m.insert(
            "ollamaModel",
            self.ollama_model
                .clone()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "qwen2.5:0.5b".to_string()),
        );
        m
    }

    /// Spawn the service identified by `id`. Returns Err if the template
    /// doesn't exist or the spawn fails. Currently a no-op when the
    /// service is already Running.
    pub fn start(&mut self, id: &str) -> Result<(), String> {
        // Extract everything we need under a read-only borrow first so
        // we can call `self.ctx(...)` (which also borrows `self`) without
        // hitting the borrow checker. `RunSpec.clone()` is cheap — small
        // String/Vec/HashMap allocs, and start() isn't hot.
        let (port, run_spec, status) = {
            let svc = self.services.get(id).ok_or("unknown service")?;
            (svc.port, svc.template.run.clone(), svc.status)
        };
        if status == ServiceStatus::Running {
            return Ok(());
        }
        // An install in flight owns svc.installer + the bin/venv dirs. Falling
        // through would set_status(Starting) (clobbering Installing so the
        // installer is never reaped), spawn over a half-built install, and orphan
        // the installer. install_streaming() refuses the reverse collision; mirror
        // it here. ensure_by_port funnels this Err into a benign "not started".
        if status == ServiceStatus::Installing {
            return Err(format!(
                "{id}: install in progress — wait for it to finish before starting"
            ));
        }

        let ctx = self.ctx(port);

        // Resolve the command — if it doesn't already have a path
        // separator, try ${binDir} first, fall back to PATH lookup
        // (which Command::spawn does naturally).
        let raw_cmd = os_fix_path(substitute(&run_spec.command, &ctx));
        let resolved_cmd = if raw_cmd.contains(std::path::is_separator) {
            raw_cmd
        } else {
            let bin_dir = self.data_dir.join("bin").join(&raw_cmd);
            let candidates = [bin_dir.with_extension("exe"), bin_dir.clone()];
            candidates
                .iter()
                .find(|p| p.exists())
                .map(|p| p.display().to_string())
                // Resolve a just-installed system tool (e.g. ollama) to its real path so the
                // spawn doesn't fall back to the bare name + OAIY Desktop's stale PATH (which
                // wouldn't find it until a restart) and Error out.
                .or_else(|| user_programs_exe(&raw_cmd).map(|p| p.display().to_string()))
                .unwrap_or(raw_cmd)
        };

        let args: Vec<String> = run_spec
            .args
            .iter()
            .map(|a| os_fix_path(substitute(a, &ctx)))
            .collect();
        let mut env: HashMap<String, String> = run_spec
            .env
            .iter()
            .map(|(k, v)| (k.clone(), substitute(v, &ctx)))
            .collect();
        // Pin to a chosen GPU if the user assigned one in the GPU picker — so e.g.
        // llama.cpp runs on GPU 1 while krea2 keeps GPU 0, instead of both defaulting to
        // GPU 0 and exhausting its VRAM. CUDA_VISIBLE_DEVICES re-indexes, so the service
        // sees the chosen card as cuda:0 (a multi-GPU service like krea2 then runs on the
        // one card — encoder falls back to CPU; left unset it keeps its dual-GPU default).
        if let Some(gpu) = self.service_gpus.get(id).copied() {
            // The picker shows nvidia-smi indices (PCI-bus order), but CUDA defaults to
            // CUDA_DEVICE_ORDER=FASTEST_FIRST — so on a HETEROGENEOUS box "GPU 1" could map
            // to a different physical card than the user picked. Force PCI_BUS_ID so the two
            // index spaces line up; don't clobber a template that set its own order.
            env.entry("CUDA_DEVICE_ORDER".to_string())
                .or_insert_with(|| "PCI_BUS_ID".to_string());
            env.insert("CUDA_VISIBLE_DEVICES".to_string(), gpu.to_string());
        }
        let cwd = run_spec.cwd.as_deref().map(|c| os_fix_path(substitute(c, &ctx)));

        // A required, user-chosen value the template references but that's unset
        // leaves `${llamaModel}` literal (no implicit default) — refuse to spawn
        // with a bogus path and tell the user exactly what to do.
        if resolved_cmd.contains("${llamaModel}")
            || args.iter().any(|a| a.contains("${llamaModel}"))
            || env.values().any(|v| v.contains("${llamaModel}"))
            || cwd.as_deref().is_some_and(|c| c.contains("${llamaModel}"))
        {
            return Err(format!(
                "{id}: no model selected — pick one in the service's Model selector first"
            ));
        }

        // Default the working dir to the data dir (not the inherited process CWD) when the
        // template doesn't set one, for the same reason as install: a bare helper the run
        // command shells out to can't then be hijacked by a planted binary in an attacker-
        // writable CWD (Windows searches CWD before System32/PATH).
        let data_dir_cwd = self.data_dir.display().to_string();
        let cfg = SpawnConfig {
            command: &resolved_cmd,
            args: &args,
            env: &env,
            cwd: cwd.as_deref().or(Some(data_dir_cwd.as_str())),
        };

        // Now take a mutable borrow to update status + stash the runner.
        let svc = self.services.get_mut(id).ok_or("unknown service")?;
        // Tear down any lingering process from a previous (Errored / health-failed)
        // run before replacing it. Runner has no Drop, so simply overwriting
        // svc.runner below would orphan the old process (zombie + port conflict).
        if let Some(old) = svc.runner.take() {
            // Only kill the tree if that process is STILL OURS. Reaching here
            // after a crash or a failed health check is the common case, and a
            // dead child's pid is not just useless — the OS recycles pids, so
            // `kill_process_tree` on a stale one force-kills whatever unrelated
            // process now holds it. check_exited() reaps and reports: None means
            // still running (safe to kill), Some(code) means already gone.
            match old.check_exited() {
                None => kill_process_tree(old.pid),
                Some(_code) => { /* already exited — pid may be recycled, do not touch it */ }
            }
            // Idempotent, and reaps the child if check_exited() didn't.
            let _ = old.stop();
        }
        svc.set_status(ServiceStatus::Starting, None);

        match Runner::spawn(cfg) {
            Ok(runner) => {
                svc.runner = Some(Arc::new(runner));
                // We optimistically flip to Running. A future health-check
                // task can downgrade to Errored / upgrade to confirmed
                // Running, but the immediate UI feedback is "it's
                // starting → it spawned, looking good".
                svc.set_status(ServiceStatus::Running, None);
                Ok(())
            }
            Err(e) => {
                let msg = format!("spawn failed: {e}");
                svc.set_status(ServiceStatus::Errored, Some(msg.clone()));
                Err(msg)
            }
        }
    }

    pub fn stop(&mut self, id: &str) -> Result<(), String> {
        let svc = self.services.get_mut(id).ok_or("unknown service")?;
        if let Some(runner) = svc.runner.take() {
            // The runner may have spawned children (a shell wrapper, python
            // subprocesses); a plain kill of the direct child would orphan
            // them, so kill the whole process tree first, then stop() to reap
            // the direct child and clear the slot.
            kill_process_tree(runner.pid);
            let _ = runner.stop();
        }
        // Stop during an in-flight install: the live process tree is owned by
        // svc.installer (the script + its pip/curl/git children), NOT runner — so
        // runner.take() above is a no-op for it. Tear the installer's tree down too
        // (keeping the handle so its log buffer stays visible, like cancel_install).
        // Otherwise it keeps pulling GBs invisibly (reap_exited only reaps it while
        // status==Installing) and a later start() would spawn over the half-built
        // install, the exact collision start()/install_streaming() guard against.
        if svc.status == ServiceStatus::Installing {
            if let Some(installer) = &svc.installer {
                kill_process_tree(installer.pid);
                let _ = installer.stop();
            }
        }
        svc.set_status(ServiceStatus::Stopped, None);
        Ok(())
    }

    /// Ensure the service listening on `port` is running, starting it if
    /// needed. This is OAIY Desktop-side mirror of the desktop's
    /// `ensure_service_ready_by_port`: oaiy-web calls it (over HTTP) right
    /// before hitting a `127.0.0.1:<port>` endpoint that an OAIY Desktop
    /// service owns, so picking a stopped companion service in a flow and
    /// running it "just works" — no manual Start click first.
    ///
    /// Returns immediately after kicking off the spawn (the AI/HTTP nodes
    /// already retry on connection refused / 503 while a server warms up),
    /// so the HTTP handler never blocks for the full boot time.
    pub fn ensure_by_port(&mut self, port: u16) -> EnsureByPortResult {
        // Find the service configured for this port. `port` on the runtime
        // reflects the active/overridable port; fall back to default_port.
        let id = self
            .services
            .values()
            .find(|s| s.port == port || s.template.default_port == port)
            .map(|s| s.template.id.clone());

        let Some(id) = id else {
            return EnsureByPortResult {
                found: false,
                already_running: false,
                started: false,
                id: None,
                name: None,
                error: None,
            };
        };

        let (name, status) = self
            .services
            .get(&id)
            .map(|s| (s.template.name.clone(), s.status))
            .unwrap_or((String::new(), ServiceStatus::Stopped));

        if status == ServiceStatus::Running {
            return EnsureByPortResult {
                found: true,
                already_running: true,
                started: false,
                id: Some(id),
                name: Some(name),
                error: None,
            };
        }

        match self.start(&id) {
            Ok(()) => EnsureByPortResult {
                found: true,
                already_running: false,
                started: true,
                id: Some(id),
                name: Some(name),
                error: None,
            },
            Err(e) => EnsureByPortResult {
                found: true,
                already_running: false,
                started: false,
                id: Some(id),
                name: Some(name),
                error: Some(e),
            },
        }
    }

    /// Poll for unexpected exits and update status. Cheap — only walks
    /// services that currently think they're Running or Installing.
    pub fn reap_exited(&mut self) {
        // Services whose install just succeeded (exit 0) + declare an install-completion marker.
        // Collected here, written AFTER the loop so we can resolve the marker path via self.ctx
        // (an immutable borrow) without conflicting with the mutable services iteration.
        let mut mark_installed: Vec<(u16, String)> = Vec::new();
        for svc in self.services.values_mut() {
            // Reap the install script if one's in flight.
            if svc.status == ServiceStatus::Installing {
                if let Some(installer) = &svc.installer {
                    if let Some(code) = installer.check_exited() {
                        if code == 0 {
                            if let Some(marker) = svc.template.installed_marker.clone() {
                                mark_installed.push((svc.port, marker));
                            }
                        }
                        // KEEP the installer (and its LogBuffer) so a failed
                        // install's output stays visible in the LogsViewer —
                        // previously we dropped it here, so the logs vanished
                        // the instant the install errored. install_streaming
                        // replaces it on the next install; logs() prefers the
                        // runner once the service is actually running.
                        let err = if code == 0 {
                            None
                        } else {
                            Some(format!("install failed (exit code {code}) — open Logs for details"))
                        };
                        svc.set_status(
                            if code == 0 {
                                ServiceStatus::Stopped
                            } else {
                                ServiceStatus::Errored
                            },
                            err,
                        );
                    }
                }
                continue;
            }
            if svc.status != ServiceStatus::Running {
                continue;
            }
            if let Some(runner) = &svc.runner {
                if let Some(code) = runner.check_exited() {
                    let msg = format!("process exited (code {code}) — open Logs for details");
                    let err = if code == 0 { None } else { Some(msg) };
                    // KEEP the runner (and its LogBuffer) so a crashed service's
                    // stderr/traceback stays visible in the LogsViewer — dropping
                    // it here made crash logs vanish the instant the process died
                    // (e.g. Lance's "No module named flash_attn"). start() replaces
                    // it on restart; logs() prefers the runner's output once it has
                    // any, falling back to the installer otherwise.
                    svc.set_status(
                        if code == 0 { ServiceStatus::Stopped } else { ServiceStatus::Errored },
                        err,
                    );
                }
            }
        }
        // Write the install-completion marker for any service whose installer just exited 0, so
        // run_installed() flips it to installed=true. A partial/failed install (non-zero exit →
        // no marker) correctly stays not-installed even though its venv interpreter exists.
        for (port, marker) in mark_installed {
            let resolved = os_fix_path(substitute(&marker, &self.ctx(port)));
            if let Some(parent) = Path::new(&resolved).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&resolved, "installed by oaiy\n") {
                log::warn!("could not write install marker {resolved}: {e}");
            }
        }
    }

    /// Snapshot of (service id, resolved health URL) pairs for every
    /// service that's currently Running AND has a health template. The
    /// background health ticker uses this — it walks the list outside
    /// the lock, hits each URL, then folds results back in via
    /// `apply_health_results`. Keeping the lock-time short matters
    /// because users hit /api/services constantly and we don't want a
    /// slow check blocking the UI.
    pub fn health_targets(&self) -> Vec<(String, String, u64)> {
        let mut out = Vec::new();
        for (id, svc) in &self.services {
            // Probe Running services (to catch a real failure) AND Errored ones
            // whose process is STILL ALIVE — so a service wrongly faulted while its
            // model was loading can RECOVER, but a crashed one (runner kept only
            // for its logs, child slot already reaped) is NOT resurrected by a
            // stale/reused listener answering on its port.
            let probe = svc.status == ServiceStatus::Running
                || (svc.status == ServiceStatus::Errored
                    && svc.runner.as_ref().is_some_and(|r| r.is_alive()));
            if !probe {
                continue;
            }
            let Some(spec) = &svc.template.health else { continue };
            let url = spec.url.replace("${port}", &svc.port.to_string());
            out.push((id.clone(), url, spec.timeout_secs));
        }
        out
    }

    /// Apply health-probe results. A success RECOVERS a service that was
    /// previously (wrongly) Errored — e.g. faulted while its model was still
    /// loading. A failure only faults a Running service AFTER its startup grace
    /// window (the health timeout), so a slow model load isn't flagged.
    pub fn apply_health_results(&mut self, results: &[(String, bool)]) {
        let now = Utc::now();
        for (id, ok) in results {
            let Some(svc) = self.services.get_mut(id) else { continue };
            if *ok {
                // Healthy probe: bring a service that was faulted while it was
                // still coming up back to Running — but ONLY if its process is
                // genuinely alive, so a dead service isn't resurrected by some
                // other listener that happens to answer on its port.
                if svc.status == ServiceStatus::Errored
                    && svc.runner.as_ref().is_some_and(|r| r.is_alive())
                {
                    svc.set_status(ServiceStatus::Running, None);
                }
                continue;
            }
            // Failed probe: only fault a Running service, and only once it's had
            // its health-timeout to come up (the model may still be loading —
            // llama.cpp doesn't bind its port until after the model loads).
            if svc.status != ServiceStatus::Running {
                continue;
            }
            let grace = svc
                .template
                .health
                .as_ref()
                .map(|h| h.timeout_secs)
                .unwrap_or(30) as i64;
            let within_grace = svc
                .runner
                .as_ref()
                .map(|r| (now - r.started_at).num_seconds() < grace)
                .unwrap_or(false);
            if within_grace {
                continue;
            }
            svc.set_status(
                ServiceStatus::Errored,
                Some("health check failed — process is running but not responding on its port".into()),
            );
        }
    }

    /// Resolve the install script path for `id` on the current OS, or
    /// return Err if the template doesn't have one.
    pub fn install_script(&self, id: &str) -> Result<PathBuf, String> {
        let svc = self.services.get(id).ok_or("unknown service")?;
        let (win, unix) = match &svc.template.install {
            InstallSpec::None => return Err("template has no install script".into()),
            InstallSpec::Script { windows, unix } => (windows.as_deref(), unix.as_deref()),
        };
        let name = if cfg!(windows) { win } else { unix }.ok_or_else(|| {
            format!(
                "no install script for {} on {}",
                id,
                if cfg!(windows) { "windows" } else { "unix" }
            )
        })?;
        // The install script must live directly under scripts/. Reject any
        // absolute path, `..`, or path separator so a template's install field
        // can't steer the installer at an arbitrary file (Path::join replaces
        // the base on an absolute component and honours `..`). Mirrors the
        // bare-name guard in materialize_package_files / delete_model.
        let is_bare = Path::new(name)
            .file_name()
            .and_then(|s| s.to_str())
            .map(|f| f == name)
            .unwrap_or(false);
        if !is_bare || name.starts_with('.') {
            return Err(format!("invalid install script name: {name}"));
        }
        let p = self.data_dir.join("scripts").join(name);
        if !p.exists() {
            return Err(format!("install script missing: {}", p.display()));
        }
        Ok(p)
    }

    /// Kick off the install script in the background. Returns immediately;
    /// the script's stdout/stderr stream into the service's installer
    /// LogBuffer (visible via /api/services/:id/logs) and `reap_exited()`
    /// flips the status to Stopped or Errored on completion.
    ///
    /// Why streaming instead of blocking `.output()`? Some installs take
    /// minutes (llama.cpp CUDA zip is ~500 MB), and the user wants to
    /// see "Downloading…" + "Extracting…" lines tick by — same UX as
    /// running the .ps1 in a console themselves.
    pub fn install_streaming(&mut self, id: &str) -> Result<(), String> {
        let script = self.install_script(id)?;

        // Block double-installs, and refuse to install over a LIVE service —
        // its running process would be orphaned (Runner has no Drop), so the
        // user must stop it first.
        if let Some(svc) = self.services.get(id) {
            if svc.status == ServiceStatus::Installing {
                return Err("install already in progress".into());
            }
            if matches!(svc.status, ServiceStatus::Running | ServiceStatus::Starting) {
                return Err("stop the service before installing/reinstalling it".into());
            }
        }
        // Errored-with-live-runner: a service faulted while still coming up (a slow
        // model load past the health grace) keeps its process ALIVE while status is
        // Errored — which the guard above doesn't catch. Runner has no Drop, so
        // installing over it would orphan that process: it locks the live binary /
        // venv the installer overwrites + holds the port, and once status flips to
        // Installing reap_exited never reaps it. Tear it down first, exactly as
        // start() does before respawning, so Reinstall just works from Errored.
        if let Some(svc) = self.services.get_mut(id) {
            if svc.runner.as_ref().is_some_and(|r| r.is_alive()) {
                if let Some(old) = svc.runner.take() {
                    kill_process_tree(old.pid);
                    let _ = old.stop();
                }
            }
        }

        let bin_dir = self.data_dir.join("bin").display().to_string();
        let models_dir = self.models_dir.display().to_string();
        let data_dir = self.data_dir.display().to_string();
        let venvs_dir = self.data_dir.join("venvs").display().to_string();
        let scripts_dir = self.data_dir.join("scripts").display().to_string();

        // Dispatch by script extension so a service can ship a one-click
        // `.bat` (cmd) installer as well as a `.ps1`. A `.bat` sidesteps the
        // PowerShell-5.1 ANSI-codepage parse trap entirely (see python.rs)
        // and is double-clickable outside the app too.
        let script_str = script.display().to_string();
        let ext = script
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let (cmd_str, args_vec): (String, Vec<String>) = if cfg!(windows) {
            match ext.as_str() {
                "bat" | "cmd" => (system32_exe("cmd.exe"), vec!["/C".into(), script_str]),
                _ => (
                    system32_exe("WindowsPowerShell\\v1.0\\powershell.exe"),
                    vec![
                        "-NoProfile".into(),
                        "-ExecutionPolicy".into(),
                        "Bypass".into(),
                        "-File".into(),
                        script_str,
                    ],
                ),
            }
        } else {
            ("sh".to_string(), vec![script_str])
        };

        let mut env: HashMap<String, String> = HashMap::new();
        env.insert("OAIY_BIN_DIR".into(), bin_dir);
        env.insert("OAIY_MODELS_DIR".into(), models_dir);
        // All registered model roots (primary + extras), os-pathsep-joined, so
        // an installer can scan every drive the user added in Settings.
        env.insert("OAIY_MODEL_DIRS".into(), join_model_dirs(&self.model_dirs));
        // Extra vars so a Python-service installer can locate the bundled
        // interpreter (`%OAIY_DATA_DIR%\python\python.exe`), the shared venvs
        // root, and seeded helper scripts without hard-coding the data dir.
        env.insert("OAIY_DATA_DIR".into(), data_dir);
        env.insert("OAIY_VENVS_DIR".into(), venvs_dir);
        env.insert("OAIY_SCRIPTS_DIR".into(), scripts_dir);

        let cfg = SpawnConfig {
            command: &cmd_str,
            args: &args_vec,
            env: &env,
            // Run the installer in the data dir, not the inherited process CWD: the install
            // scripts shell out to bare helpers (curl/tar/where/git/robocopy) and Windows
            // searches the CWD before System32/PATH, so a planted binary in an attacker-
            // writable CWD would otherwise run during install. (Reuses the OAIY_DATA_DIR value.)
            cwd: env.get("OAIY_DATA_DIR").map(String::as_str),
        };

        let runner = Runner::spawn(cfg).map_err(|e| {
            let msg = format!("install spawn failed: {e}");
            if let Some(svc) = self.services.get_mut(id) {
                svc.set_status(ServiceStatus::Errored, Some(msg.clone()));
            }
            msg
        })?;

        let svc = self.services.get_mut(id).ok_or("unknown service")?;
        svc.installer = Some(Arc::new(runner));
        svc.set_status(ServiceStatus::Installing, None);
        Ok(())
    }

    /// Cancel an in-flight install. The installer is `cmd /C <script>` (or
    /// `sh`), which spawns children (pip, python downloads); killing just the
    /// shell would orphan those, so we kill the whole process TREE. The
    /// installer's LogBuffer is kept so the user still sees where it stopped.
    pub fn cancel_install(&mut self, id: &str) -> Result<(), String> {
        let svc = self.services.get_mut(id).ok_or("unknown service")?;
        if svc.status != ServiceStatus::Installing {
            return Err("no install is in progress for this service".into());
        }
        if let Some(installer) = &svc.installer {
            kill_process_tree(installer.pid);
            let _ = installer.stop();
        }
        svc.set_status(ServiceStatus::Stopped, Some("install cancelled".into()));
        Ok(())
    }

    /// Stop every running service AND tear down any in-flight installer
    /// (called on app exit).
    pub fn stop_all(&mut self) {
        let ids: Vec<String> = self.services.keys().cloned().collect();
        for id in ids {
            let _ = self.stop(&id);
            // stop() only handles the runner; an in-flight install is owned by
            // svc.installer (cmd /C <script> → pip/curl/powershell children).
            // Tear it down too so quitting doesn't leave a detached download or
            // build running unmanaged.
            if let Some(svc) = self.services.get_mut(&id) {
                if let Some(installer) = svc.installer.take() {
                    kill_process_tree(installer.pid);
                    let _ = installer.stop();
                }
            }
        }
    }

    /// Create or replace a service template from a UI form. Writes to
    /// `templates/<id>.json` and refreshes the in-memory entry. Refuses
    /// to clobber a currently-Running service so the user doesn't lose
    /// the live runner reference.
    pub fn add_template(&mut self, template: ServiceTemplate) -> Result<(), String> {
        if !valid_service_id(&template.id) {
            return Err("service id must be lowercase letters/digits/dash/underscore (1–64 chars)".into());
        }
        if let Some(existing) = self.services.get(&template.id) {
            if existing.status == ServiceStatus::Running
                || existing.status == ServiceStatus::Installing
            {
                return Err(format!(
                    "service {} is {:?}; stop it before editing",
                    template.id, existing.status
                ));
            }
        }
        let path = self
            .data_dir
            .join("templates")
            .join(format!("{}.json", template.id));
        let json = serde_json::to_string_pretty(&template)
            .map_err(|e| format!("serialize: {e}"))?;
        // Write atomically: lay down a sibling .tmp then rename over the
        // target. rename is atomic on the same volume, so a crash mid-write
        // can never leave a half-written / corrupt template behind.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("rename {} -> {}: {e}", tmp.display(), path.display()));
        }
        // Lay down any bundled scripts so a self-contained imported package is
        // immediately installable/runnable.
        if !template.files.is_empty() {
            materialize_package_files(&self.data_dir.join("scripts"), &template.id, &template.files);
        }
        let id = template.id.clone();
        self.services.insert(id, ServiceRuntime::new(template));
        Ok(())
    }

    /// Build a shareable, self-contained package for `id`: the template with
    /// every script it references inlined into `files`, so the result is one
    /// JSON that installs + runs anywhere with no recompile. Best-effort
    /// bundling: includes the install script(s) plus any file in the scripts
    /// dir whose name appears in the template JSON or the install script's own
    /// text (so helper scripts the installer calls come along too).
    pub fn export_package(&self, id: &str) -> Result<ServiceTemplate, String> {
        let svc = self.services.get(id).ok_or("unknown service")?;
        let mut t = svc.template.clone();
        // A self-contained template already declares its authoritative file set;
        // capture it so export round-trips those completely (re-read from disk to
        // pick up hand-edits), regardless of whether each name appears in `hay`.
        let declared: Vec<String> = t.files.keys().cloned().collect();
        t.files.clear();
        let scripts_dir = self.data_dir.join("scripts");

        // Haystack of names to search for: the serialized template (catches
        // run.command/args refs) + the install scripts' own text (catches
        // helpers the installer shells out to, e.g. fetch_zip.py).
        let mut hay = serde_json::to_string(&t).unwrap_or_default();
        if let InstallSpec::Script { windows, unix } = &t.install {
            for n in [windows.as_deref(), unix.as_deref()].into_iter().flatten() {
                hay.push(' ');
                hay.push_str(n);
                if let Ok(s) = std::fs::read_to_string(scripts_dir.join(n)) {
                    hay.push('\n');
                    hay.push_str(&s);
                }
            }
        }

        let mut files = HashMap::new();
        // The template's own declared files are authoritative — always include
        // them (re-read for hand-edits), so a self-contained package round-trips
        // even if a name doesn't happen to appear in the haystack.
        for name in &declared {
            if let Ok(body) = std::fs::read_to_string(scripts_dir.join(name)) {
                files.insert(name.clone(), body);
            }
        }
        // Plus any other script in the dir whose name is referenced by the
        // template/install text (helpers the installer shells out to).
        if let Ok(rd) = std::fs::read_dir(&scripts_dir) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_file() {
                    continue;
                }
                let Some(fname) = p.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                // Skip hidden snapshot files + already-included declared files.
                if fname.starts_with('.') || files.contains_key(fname) || !hay.contains(fname) {
                    continue;
                }
                if let Ok(body) = std::fs::read_to_string(&p) {
                    files.insert(fname.to_string(), body);
                }
            }
        }
        t.files = files;
        Ok(t)
    }

    /// Re-scan the templates dir and register any package whose id isn't loaded
    /// yet — so dropping a `*.json` into `templates/` (or an Import) shows up
    /// live, no restart. Existing services (incl. running ones) are left
    /// untouched, so this is safe to call on every poll. Returns how many were
    /// newly added. Bundled `files` are materialized for the new ones.
    pub fn reload_new_templates(&mut self) -> usize {
        let dir = self.data_dir.join("templates");
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return 0;
        };
        let scripts_dir = self.data_dir.join("scripts");
        let mut added = 0usize;
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(&path) else {
                continue;
            };
            let t: ServiceTemplate = match serde_json::from_str(&raw) {
                Ok(t) => t,
                Err(e) => {
                    log::warn!("skipping bad template {}: {}", path.display(), e);
                    continue;
                }
            };
            // Already known (by id) — don't disturb its in-memory/runtime state.
            if self.services.contains_key(&t.id) {
                continue;
            }
            if !t.files.is_empty() {
                materialize_package_files(&scripts_dir, &t.id, &t.files);
            }
            log::info!("dynamically loaded service '{}' from {}", t.id, path.display());
            let id = t.id.clone();
            self.services.insert(id, ServiceRuntime::new(t));
            added += 1;
        }
        added
    }

    /// Remove a template + its on-disk JSON. Refuses if Running.
    pub fn delete_template(&mut self, id: &str) -> Result<(), String> {
        if let Some(svc) = self.services.get(id) {
            if svc.status == ServiceStatus::Running
                || svc.status == ServiceStatus::Installing
            {
                return Err(format!("service {id} is {:?}; stop it first", svc.status));
            }
        } else {
            return Err("unknown service".into());
        }
        let path = self
            .data_dir
            .join("templates")
            .join(format!("{}.json", id));
        // Best-effort file removal — if it's missing on disk but in
        // memory (shouldn't happen normally) we still proceed.
        let _ = std::fs::remove_file(&path);
        self.services.remove(id);
        Ok(())
    }
}

/// Kill a process and its descendants. The installer shell spawns pip /
/// python downloaders; a plain kill of the shell would orphan them (they'd
/// keep downloading), so use the OS tree-kill.
pub(crate) fn kill_process_tree(pid: u32) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = std::process::Command::new(system32_exe("taskkill.exe"))
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }
    #[cfg(not(windows))]
    {
        // Negative pid = the process GROUP (the shell + its children). Graceful
        // SIGTERM, then escalate to SIGKILL after a short grace so a descendant
        // that ignores/slow-handles TERM (pip/curl/git mid-download, a Chromium
        // renderer under Playwright) can't linger and keep a port bound — matching
        // the forceful `taskkill /T /F` on Windows. Detached (no .wait()) so app
        // exit / a held Registry lock never blocks on the grace sleep.
        let _ = std::process::Command::new("sh")
            .args([
                "-c",
                &format!("kill -TERM -{pid} 2>/dev/null; sleep 2; kill -KILL -{pid} 2>/dev/null"),
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

fn valid_service_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Trim a model selection and treat all-whitespace/empty as "unset" (`None`).
/// Shared by `set_llama_model` / `set_ollama_model`.
fn clean_model_opt(model: Option<String>) -> Option<String> {
    model.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Extract the venv name from a path-ish string containing a `venvs/<name>`
/// (or `venvs\<name>`) segment — e.g. `${dataDir}/venvs/ltx2/Scripts/python.exe`
/// → `Some("ltx2")`. Returns None when there's no such segment.
fn venv_name_in(s: &str) -> Option<String> {
    let norm = s.replace('\\', "/");
    let idx = norm.find("venvs/")?;
    let rest = &norm[idx + "venvs/".len()..];
    let name: String = rest.chars().take_while(|&c| c != '/').collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

pub type RegistryHandle = Arc<Mutex<Registry>>;

#[cfg(test)]
mod tests {
    use super::venv_name_in;

    #[test]
    fn extracts_venv_name_from_run_command() {
        // Forward + back slashes, with the `${dataDir}` placeholder intact.
        assert_eq!(
            venv_name_in("${dataDir}/venvs/ltx2/Scripts/python.exe").as_deref(),
            Some("ltx2")
        );
        assert_eq!(
            venv_name_in("${dataDir}\\venvs\\lance\\Scripts\\python.exe").as_deref(),
            Some("lance")
        );
        assert_eq!(
            venv_name_in("C:/data/venvs/playwright/Scripts/python.exe").as_deref(),
            Some("playwright")
        );
    }

    #[test]
    fn no_venv_segment_returns_none() {
        assert_eq!(venv_name_in("ollama"), None);
        assert_eq!(venv_name_in("${binDir}/llama-server.exe"), None);
        // `venvs/` with nothing after it is not a usable name.
        assert_eq!(venv_name_in("x/venvs/"), None);
    }

    #[test]
    fn all_builtin_templates_deserialize() {
        for (name, body) in super::BUILTIN_TEMPLATES {
            let t: super::ServiceTemplate = serde_json::from_str(body)
                .unwrap_or_else(|e| panic!("builtin template {name} failed to deserialize: {e}"));
            assert!(!t.id.is_empty(), "builtin template {name} has empty id");
        }
    }

    #[test]
    fn krea2_has_uninstall_and_central_models() {
        let (_, body) = super::BUILTIN_TEMPLATES
            .iter()
            .find(|(n, _)| *n == "krea2.json")
            .expect("krea2.json builtin");
        let t: super::ServiceTemplate = serde_json::from_str(body).expect("krea2 deserializes");
        // Uninstall removes the venv + repo (program files) so the button appears...
        let paths = &t.uninstall.expect("krea2 declares an uninstall spec").paths;
        assert!(
            paths.iter().any(|p| p == "${dataDir}/venvs/krea2"),
            "krea2 uninstall should remove its venv, got {paths:?}"
        );
        // ...but must NOT delete the central models (matches the "models not touched"
        // convention in the uninstall confirm dialog + avoids deleting a big download).
        assert!(
            !paths.iter().any(|p| p.contains("modelsDir")),
            "krea2 uninstall should NOT delete the central models dir, got {paths:?}"
        );
        // Models centralized: checkpoint + HF cache live under ${modelsDir}.
        assert!(
            body.contains("${modelsDir}/krea2/checkpoints"),
            "krea2 checkpoint should live under the central models dir"
        );
    }

    #[test]
    fn uninstall_refuses_managed_roots_and_structural_dirs() {
        use std::path::PathBuf;
        let data = PathBuf::from("C:/oaiy/data");
        let models = PathBuf::from("C:/oaiy/models");
        let reg = super::Registry::empty(data.clone(), models.clone());
        // The data dir, the models dir, and the shared structural dirs must ALL be refused —
        // an uninstall removing one would remove_dir_all every OTHER service's files / the
        // whole model library, not just the service being uninstalled.
        let mut protected: Vec<PathBuf> = ["bin", "venvs", "services", "templates", "scripts"]
            .iter()
            .map(|s| data.join(s))
            .collect();
        protected.push(data.clone());
        protected.push(models.clone());
        for p in &protected {
            assert!(
                reg.is_protected_uninstall_root(p),
                "uninstall must REFUSE the root/structural path {}",
                p.display()
            );
        }
        // Per-service subtrees + the llama-cpp glob target sit STRICTLY under a structural
        // dir; the guard is an equality check (not a prefix), so they still pass through and
        // real built-in uninstalls keep working.
        for p in [
            data.join("venvs/krea2"),
            data.join("services/ltx2"),
            data.join("bin/llama-server.exe"),
            models.join("krea2/checkpoints"),
        ] {
            assert!(
                !reg.is_protected_uninstall_root(&p),
                "uninstall must ALLOW the per-service path {}",
                p.display()
            );
        }
    }

    #[test]
    fn package_files_cannot_overwrite_builtin_or_others_scripts() {
        use std::collections::HashMap;
        let dir = std::env::temp_dir().join(format!("oaiy-mpf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let reserved = super::reserved_script_names();
        assert!(
            !reserved.is_empty(),
            "reserved set should be populated from the built-ins"
        );
        // A name owned by a built-in script / built-in template.
        let reserved_name = reserved.iter().next().unwrap().clone();
        std::fs::write(dir.join(&reserved_name), "GENUINE").unwrap();
        // An imported package must NOT overwrite a built-in's script (cross-service RCE).
        let mut evil = HashMap::new();
        evil.insert(reserved_name.clone(), "EVIL".to_string());
        super::materialize_package_files(&dir, "attacker-pkg", &evil);
        assert_eq!(
            std::fs::read_to_string(dir.join(&reserved_name)).unwrap(),
            "GENUINE",
            "an imported package must not overwrite a built-in script"
        );

        // A package writes its OWN non-reserved helper → allowed + owned.
        let mut a = HashMap::new();
        a.insert("pkgA_helper.py".to_string(), "A".to_string());
        super::materialize_package_files(&dir, "pkgA", &a);
        assert_eq!(std::fs::read_to_string(dir.join("pkgA_helper.py")).unwrap(), "A");
        // A DIFFERENT package must NOT clobber pkgA's helper.
        let mut b = HashMap::new();
        b.insert("pkgA_helper.py".to_string(), "B".to_string());
        super::materialize_package_files(&dir, "pkgB", &b);
        assert_eq!(
            std::fs::read_to_string(dir.join("pkgA_helper.py")).unwrap(),
            "A",
            "package B must not overwrite package A's script"
        );
        // pkgA CAN refresh its own helper.
        let mut a2 = HashMap::new();
        a2.insert("pkgA_helper.py".to_string(), "A2".to_string());
        super::materialize_package_files(&dir, "pkgA", &a2);
        assert_eq!(
            std::fs::read_to_string(dir.join("pkgA_helper.py")).unwrap(),
            "A2"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn installed_marker_gates_installed_over_run_executable() {
        let data = std::env::temp_dir().join(format!("oaiy-marker-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        std::fs::create_dir_all(&data).unwrap();
        let reg = super::Registry::empty(data.clone(), data.join("models"));

        let json = r#"{
            "id": "svc", "name": "Svc", "description": "", "category": "test",
            "defaultPort": 9999,
            "installedMarker": "${dataDir}/venvs/svc/.oaiy-installed",
            "run": { "command": "${dataDir}/venvs/svc/Scripts/python.exe", "args": [] }
        }"#;
        let t: super::ServiceTemplate = serde_json::from_str(json).unwrap();

        // The venv interpreter exists (a partial install creates it at step 1) BUT no marker →
        // must read as NOT installed (this is the whole point of the marker).
        let interp = data.join("venvs/svc/Scripts/python.exe");
        std::fs::create_dir_all(interp.parent().unwrap()).unwrap();
        std::fs::write(&interp, "").unwrap();
        assert!(
            !reg.run_installed(&t, 9999),
            "interpreter present but no marker must read as not-installed"
        );

        // Marker present (installer exited 0) → installed.
        std::fs::write(data.join("venvs/svc/.oaiy-installed"), "").unwrap();
        assert!(reg.run_installed(&t, 9999), "marker present → installed");

        // backfill: a pre-marker install (interp present, marker removed) gets the marker back.
        std::fs::remove_file(data.join("venvs/svc/.oaiy-installed")).unwrap();
        let mut reg2 = super::Registry::empty(data.clone(), data.join("models"));
        reg2.services
            .insert("svc".to_string(), super::ServiceRuntime::new(t.clone()));
        reg2.backfill_install_markers();
        assert!(
            data.join("venvs/svc/.oaiy-installed").exists(),
            "backfill should restore the marker for an existing install (interp present)"
        );

        // backfill is ONE-TIME (sentinel persisted): a NEW partial install after migration
        // (interpreter present, marker absent) must NOT be re-blessed on a later run/restart.
        std::fs::remove_file(data.join("venvs/svc/.oaiy-installed")).unwrap();
        reg2.backfill_install_markers();
        assert!(
            !data.join("venvs/svc/.oaiy-installed").exists(),
            "one-time backfill must not resurrect a post-migration partial install"
        );

        let _ = std::fs::remove_dir_all(&data);
    }
}
