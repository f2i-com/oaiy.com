//! Service template — declarative description of an installable + runnable
//! local AI service. One JSON file per service; users can drop their own
//! into `%APPDATA%/com.oaiy/templates/` to extend the set without
//! recompiling.
//!
//! Path placeholders supported in `args` / `env` / `cwd`:
//!   ${port}        — the resolved port from the running config
//!   ${dataDir}     — `%APPDATA%/com.oaiy/`
//!   ${binDir}      — `%APPDATA%/com.oaiy/bin/`
//!   ${modelsDir}   — `%APPDATA%/com.oaiy/models/`
//!   ${scriptsDir}  — `%APPDATA%/com.oaiy/scripts/` (where `files` land)
//!
//! A template can be fully self-contained ("plug-and-play"): set `files` to a
//! map of `filename -> contents` and those scripts are written into
//! `${scriptsDir}` on load, so the install/run commands that reference them
//! resolve with no recompile and no separate files to ship. Drop one JSON into
//! the templates dir (or Import it) and the service is ready.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceTemplate {
    /// Stable identifier — used as the URL path segment + on-disk filename.
    /// Lowercase ASCII + hyphens recommended.
    pub id: String,

    /// User-visible name (e.g. "Ollama", "Llama.cpp Server").
    pub name: String,

    /// One-line description shown in the UI.
    pub description: String,

    /// Free-form category for grouping in the UI. Suggested values:
    /// "LLM", "Image Generation", "Video Generation", "Speech", "Browser".
    pub category: String,

    /// Default TCP port the service listens on. Used in placeholder
    /// substitution. Users can override per-run in Phase 3.
    pub default_port: u16,

    /// How to install (if anything beyond running is needed).
    #[serde(default)]
    pub install: InstallSpec,

    /// What an Uninstall removes (optional). Present ⇒ the service shows an
    /// Uninstall button in the desktop app.
    #[serde(default)]
    pub uninstall: Option<UninstallSpec>,

    /// A placeholder-expanded path the companion writes when this service's installer EXITS 0,
    /// and checks for `installed`. For a venv service whose interpreter exists right after the
    /// venv is created (step 1) — BEFORE the multi-GB pip/model download that can fail — the
    /// interpreter merely existing isn't "installed"; this marker, written only on success, is.
    /// Put it inside the dir Uninstall removes (e.g. `${dataDir}/venvs/<id>/.oaiy-installed`) so
    /// it's cleaned up. None ⇒ fall back to "the run executable exists".
    #[serde(default)]
    pub installed_marker: Option<String>,

    /// How to run the service.
    pub run: RunSpec,

    /// How to verify it's actually serving traffic.
    #[serde(default)]
    pub health: Option<HealthSpec>,

    /// Optional doc/help URL shown next to the service in the UI.
    #[serde(default)]
    pub docs_url: Option<String>,

    /// How a client (the web app's palette + service nodes) should CALL this
    /// service: request path (appended to the base URL), method, body template,
    /// and which JSON field to read back — so it surfaces as a ready-to-run node
    /// without the web app hard-coding per-service knowledge. LLM services set
    /// `apiFormat: "openai"` instead. Absent → the web app falls back to a
    /// category convention.
    #[serde(default)]
    pub node: Option<NodeSpec>,

    /// Bundled scripts that make this a self-contained, plug-and-play package:
    /// `filename -> file contents`. On load each is written into
    /// `${scriptsDir}` (via the same snapshot policy as built-in scripts, so
    /// updates ship but hand-edits survive), so install/run commands that
    /// reference them by name resolve. Built-in services leave this empty
    /// (their scripts are compiled in); it's for user-authored / imported
    /// packages, and `export` fills it from disk. Names must be bare filenames
    /// (no path separators).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub files: HashMap<String, String>,
}

/// How to call a service as a flow node — declared by the template so the web
/// app surfaces it as a pre-wired Service Call (or LLM) node, with no
/// per-service knowledge baked into the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeSpec {
    /// Request path appended to `http://127.0.0.1:<port>` (e.g. "/generate").
    #[serde(default)]
    pub endpoint: Option<String>,
    /// HTTP method; the web app defaults to POST.
    #[serde(default)]
    pub method: Option<String>,
    /// Request body template, with `{{input}}` / `{{inputRaw}}` placeholders.
    #[serde(default)]
    pub body_template: Option<String>,
    /// Dot-path into the JSON response to surface (e.g. "imageUrl").
    #[serde(default)]
    pub response_path: Option<String>,
    /// "openai" → surface as an OpenAI-compatible LLM node.
    #[serde(default)]
    pub api_format: Option<String>,
    /// Palette icon (emoji or lucide name).
    #[serde(default)]
    pub icon: Option<String>,
}

/// Install variants. `None` assumes the binary is already on PATH or in
/// `${binDir}` (the typical case for bundled sidecars).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum InstallSpec {
    /// No install needed — the run command is expected to work as-is.
    #[default]
    None,

    /// Run a platform-specific script. Path is relative to the seeded
    /// scripts dir (`%APPDATA%/com.oaiy/scripts/`).
    Script {
        /// Windows PowerShell script filename (.ps1)
        #[serde(default)]
        windows: Option<String>,
        /// Unix shell script filename (.sh)
        #[serde(default)]
        unix: Option<String>,
    },
}

/// What an Uninstall removes: files/dirs under the managed roots
/// (`${binDir}` / `${dataDir}` / `${modelsDir}`). Each entry is placeholder-
/// expanded; a `*` is allowed in the final path segment (e.g. `${binDir}/llama-*.exe`).
/// The registry guards every resolved path to stay inside a managed root.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UninstallSpec {
    #[serde(default)]
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSpec {
    /// Executable name. If it contains a path separator, used as-is;
    /// otherwise resolved against `${binDir}` first, then PATH.
    pub command: String,

    /// Argument list. Each entry can contain `${port}` / `${dataDir}` /
    /// `${binDir}` / `${modelsDir}` placeholders that get expanded at
    /// spawn time.
    #[serde(default)]
    pub args: Vec<String>,

    /// Extra env vars to set on the child. Placeholders accepted.
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Working directory for the spawn. Placeholders accepted. Defaults
    /// to `${dataDir}` when not set.
    #[serde(default)]
    pub cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthSpec {
    /// URL to GET after start; non-error status = healthy. Placeholders
    /// accepted (typically `http://127.0.0.1:${port}/...`).
    pub url: String,

    /// Max time to wait for the service to come up, in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_timeout() -> u64 {
    30
}

/// Substitute `${name}` placeholders in `s` using the given context.
pub fn substitute(s: &str, ctx: &HashMap<&'static str, String>) -> String {
    let mut out = s.to_string();
    for (k, v) in ctx {
        let needle = format!("${{{}}}", k);
        out = out.replace(&needle, v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_replaces_placeholders() {
        let mut ctx: HashMap<&'static str, String> = HashMap::new();
        ctx.insert("port", "11434".to_string());
        ctx.insert("binDir", "C:\\bin".to_string());
        assert_eq!(
            substitute("serve --port ${port}", &ctx),
            "serve --port 11434"
        );
        assert_eq!(
            substitute("${binDir}\\ollama.exe", &ctx),
            "C:\\bin\\ollama.exe"
        );
    }
}
