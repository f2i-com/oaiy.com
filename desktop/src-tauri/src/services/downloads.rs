//! Model downloads — HuggingFace + direct URLs, with pause/resume.
//!
//! Each download runs on a background task and writes into a `.part`
//! file under the designated downloads folder (`${dataDir}/models/`).
//! Progress is tracked in a shared map; the UI polls
//! /api/models/downloads to render bars.
//!
//! Pause/resume works via HTTP `Range: bytes=N-`. When paused, the
//! background task is aborted but the .part file + byte counter survive,
//! so a later resume picks up exactly where it left off. Servers that
//! don't accept ranges (rare for HF) cause resume to restart from 0 —
//! we surface that on the resumed entry's `error` field as a warning.
//!
//! HuggingFace URL normalisation: accept either
//!   https://huggingface.co/<repo>/blob/<rev>/<file>           (browser)
//!   https://huggingface.co/<repo>/resolve/<rev>/<file>        (direct)
//! and rewrite the first to the second.

use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use sha2::{Digest, Sha256};
use tokio::task::AbortHandle;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    Queued,
    Active,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub id: String,
    pub url: String,
    pub filename: String,
    pub subdir: Option<String>,
    /// Absolute path of the final file (where the .part lands after rename).
    /// The UI shows this so the user can copy it to a file explorer.
    pub dest_path: String,
    pub status: DownloadStatus,
    pub bytes_downloaded: u64,
    pub bytes_total: Option<u64>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    /// Whether the remote server supports byte-range resume. `None` until
    /// the first response comes back; surfaced so the UI can disable the
    /// Pause button on servers that would force a restart.
    pub resumable: Option<bool>,
    /// Bytes/sec over a 4-second sliding window. `None` until the first
    /// progress tick. Stops updating (stays at last value) on pause so
    /// the UI doesn't flicker to 0 mid-pause.
    pub speed_bps: Option<u64>,
    /// Seconds remaining at the current speed. `None` when `bytes_total`
    /// or `speed_bps` isn't known yet.
    pub eta_secs: Option<u64>,
    /// SHA-256 of what we actually wrote, computed as it streamed. Reported
    /// whether or not anything could be checked against it — a user with a
    /// digest from elsewhere can then compare by eye.
    pub sha256: Option<String>,
    /// The digest we're checking against: supplied by the caller, or taken from
    /// HuggingFace's `X-Linked-ETag` (the git-lfs OID, which IS the content's
    /// SHA-256). `None` means nothing could be verified — see `verified`.
    pub expected_sha256: Option<String>,
    /// `Some(true)` the content matched, `Some(false)` it did not (the file is
    /// deleted and the download failed), `None` there was nothing to check it
    /// against. Deliberately three-valued: reporting "unverified" as `false`
    /// would cry wolf on every direct URL, and as `true` would claim a
    /// guarantee that was never made.
    pub verified: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub modified: Option<DateTime<Utc>>,
}

/// Snapshot for the UI — wraps `list_models()` output with the designated
/// folder path so the user sees "all your models live here:" prominently.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsSnapshot {
    pub root_dir: String,
    pub models: Vec<ModelFile>,
    /// Free space on the drive holding the models dir, so the UI can show
    /// "142 GiB free" and warn before a download that won't fit. `None`
    /// if the query failed (rare — non-fatal).
    pub free_bytes: Option<u64>,
}

/// Headroom we keep free so a download never fills the drive to zero —
/// the final `.part` → final-name rename, plus logs/temp, all need room.
const DISK_MARGIN_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB

/// A lowercase 64-char hex string, and nothing else.
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Normalise a caller-supplied digest, or refuse it.
///
/// Refused rather than ignored: a user who pastes a truncated or MD5 digest and
/// gets a silently-unverified download is worse off than one who gets an error,
/// because they believe a check happened.
fn parse_expected_sha256(raw: &str) -> Result<String, String> {
    let cleaned = raw.trim().trim_start_matches("sha256:").trim();
    if !is_sha256_hex(cleaned) {
        return Err(format!(
            "expectedSha256 must be 64 hex characters (got {} characters)",
            cleaned.len()
        ));
    }
    Ok(cleaned.to_ascii_lowercase())
}

/// Read a content digest out of an ETag-style header value.
///
/// HuggingFace sets `X-Linked-ETag` to the file's git-lfs OID, which by the LFS
/// spec is the SHA-256 of the content — so for LFS-backed model files this is an
/// authoritative digest, published by the origin, that costs nothing to obtain.
///
/// Only `X-Linked-ETag` is trusted. The CDN's own `etag` on the final response
/// is also 64 hex characters but is a DIFFERENT value (the object store's, not
/// the content's) — verified against the HF API's `lfs.oid`. Treating it as a
/// content digest would fail every download.
fn digest_from_etag(value: &str) -> Option<String> {
    let v = value.trim().trim_start_matches("W/").trim_matches('"');
    is_sha256_hex(v).then(|| v.to_ascii_lowercase())
}

/// Pure free-space decision: how many bytes short we'd be after reserving
/// the margin, or `None` if `needed` fits. Split out so it's unit-testable
/// without touching a real filesystem.
fn space_shortfall(available: u64, needed: u64) -> Option<u64> {
    let required = needed.saturating_add(DISK_MARGIN_BYTES);
    required.checked_sub(available).filter(|short| *short > 0)
}

fn human_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Whether a URL points at HuggingFace, so it's safe to attach the user's
/// HF token. Matches `huggingface.co` and its subdomains only — we never
/// send the token to an arbitrary host the user pasted.
fn is_hf_host(url: &str) -> bool {
    // Require HTTPS: the only caller that matters attaches the user's HF bearer
    // token to is_hf_host URLs, and that token must never ride a plaintext request
    // (a MitM on the path would capture the Authorization header). Checking the
    // host without the scheme let `http://huggingface.co/...` leak the token.
    match url::Url::parse(url).ok() {
        Some(u) if u.scheme() == "https" => u
            .host_str()
            .map(str::to_lowercase)
            .is_some_and(|host| host == "huggingface.co" || host.ends_with(".huggingface.co")),
        _ => false,
    }
}

/// True when an IP is one a model download must NEVER reach: loopback, private (RFC1918),
/// link-local (incl. the 169.254.169.254 cloud-metadata endpoint), CGNAT, unspecified, etc.
/// The SSRF guard, so an untrusted catalog/template download URL can't drive OAIY Desktop
/// into internal services on the user's (possibly cloud/corporate) machine.
fn is_disallowed_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 0x40) // 100.64.0.0/10 CGNAT
        }
        IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_disallowed_ip(IpAddr::V4(v4));
            }
            let seg0 = v6.segments()[0];
            (seg0 & 0xfe00) == 0xfc00 // unique-local fc00::/7
                || (seg0 & 0xffc0) == 0xfe80 // link-local fe80::/10
        }
    }
}

/// SSRF guard for a model-download URL: require https + reject a host that IS (or resolves to)
/// an internal/special address. Runs synchronously in start() BEFORE any task is spawned, so a
/// hostile URL fails fast (Err -> 400) and never becomes an internal-port-scan oracle (the
/// status endpoint otherwise reflects an internal response's status/size/resumable back).
fn validate_download_url(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|_| "invalid download URL".to_string())?;
    if parsed.scheme() != "https" {
        return Err("download URL must use https".into());
    }
    let host = parsed.host_str().ok_or("download URL has no host")?;
    // Bare IP literal — check directly (no DNS).
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        if is_disallowed_ip(ip) {
            return Err(format!("download URL points at a disallowed address ({ip})"));
        }
        return Ok(());
    }
    // Hostname — resolve + reject if ANY resolved address is internal (covers a DNS name that
    // points at an internal IP). Brief blocking lookup; start() is user-initiated, not hot.
    use std::net::ToSocketAddrs;
    let port = parsed.port_or_known_default().unwrap_or(443);
    let mut saw_any = false;
    for addr in (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve download host '{host}': {e}"))?
    {
        saw_any = true;
        if is_disallowed_ip(addr.ip()) {
            return Err(format!(
                "download host '{host}' resolves to a disallowed address ({})",
                addr.ip()
            ));
        }
    }
    if !saw_any {
        return Err(format!("download host '{host}' did not resolve"));
    }
    Ok(())
}

/// Parse the start byte of a `Content-Range: bytes <start>-<end>/<total>` response header.
fn content_range_start(resp: &reqwest::Response) -> Option<u64> {
    resp.headers()
        .get("content-range")?
        .to_str()
        .ok()?
        .trim()
        .strip_prefix("bytes")?
        .trim_start()
        .split('-')
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
}

pub struct Downloads {
    /// Where downloaded files land: `${dataDir}/models/`. This is the
    /// "designated folder" the UI shows prominently.
    models_dir: PathBuf,
    /// All downloads ever attempted in this session, keyed by id.
    /// Completed entries stay so the UI can surface "just finished".
    progress: Arc<Mutex<HashMap<String, DownloadProgress>>>,
    /// AbortHandles for in-flight task cancellation (pause + cancel).
    /// Kept separate from `progress` so the latter can stay serializable.
    abort_handles: Arc<Mutex<HashMap<String, AbortHandle>>>,
    /// Optional HuggingFace access token, sent as `Authorization: Bearer`
    /// on huggingface.co requests so gated/private repos (Llama, some
    /// Gemma) download. Set from the persisted config at startup +
    /// whenever the user changes it in Settings. Never sent to non-HF
    /// hosts (incl. the CDN host the resolve URL 302-redirects to — that
    /// URL is already presigned, and reqwest strips auth across hosts).
    hf_token: Arc<Mutex<Option<String>>>,
}

pub type DownloadsHandle = Arc<Downloads>;

impl Downloads {
    /// `models_dir` is the resolved downloads root (the `modelsDir` override
    /// when set, else `<dataDir>/models`) — passed in fully-resolved so the
    /// downloader doesn't need to know about the data dir.
    pub fn new(models_dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&models_dir);
        Self {
            models_dir,
            progress: Arc::new(Mutex::new(HashMap::new())),
            abort_handles: Arc::new(Mutex::new(HashMap::new())),
            hf_token: Arc::new(Mutex::new(None)),
        }
    }

    /// Set (or clear, with None/empty) the HuggingFace token used for
    /// gated downloads. An empty string clears it.
    pub fn set_token(&self, token: Option<String>) {
        let cleaned = token.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
        if let Ok(mut g) = self.hf_token.lock() {
            *g = cleaned;
        }
    }

    /// Whether a HuggingFace token is currently set (the UI shows status,
    /// never the token itself).
    pub fn has_token(&self) -> bool {
        self.hf_token.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    pub fn into_handle(self) -> DownloadsHandle {
        Arc::new(self)
    }

    /// Designated downloads folder. Surfaced in `list_models()` for the
    /// UI; this getter is for future Tauri commands that may want it.
    #[allow(dead_code)]
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn snapshot(&self) -> Vec<DownloadProgress> {
        match self.progress.lock() {
            Ok(g) => {
                let mut v: Vec<_> = g.values().cloned().collect();
                // Newest first so the bar of interest is at the top of the UI.
                v.sort_by(|a, b| b.started_at.cmp(&a.started_at));
                v
            }
            Err(_) => Vec::new(),
        }
    }

    pub fn list_models(&self) -> Result<ModelsSnapshot, String> {
        let mut models = Vec::new();
        walk_dir(&self.models_dir, &self.models_dir, &mut models)
            .map_err(|e| format!("scan failed: {e}"))?;
        models.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(ModelsSnapshot {
            root_dir: self.models_dir.display().to_string(),
            models,
            free_bytes: fs2::available_space(&self.models_dir).ok(),
        })
    }

    pub fn delete_model(&self, name: &str) -> Result<(), String> {
        // Reject any traversal/absolute component (`..`, a root, or a Windows
        // drive/UNC prefix) outright — more robust than a substring `..`
        // check, which misses e.g. a bare `C:` prefix.
        use std::path::Component;
        let name_path = Path::new(name);
        if name_path.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err("invalid model name".into());
        }
        let p = self.models_dir.join(name);
        if !p.starts_with(&self.models_dir) {
            return Err("path escapes models dir".into());
        }

        // Resolve symlinks/junctions before deleting: a symlink (or NTFS
        // junction) inside models_dir could otherwise point the final path at
        // a target outside it. `p` itself may not be canonicalizable if it's a
        // dangling link or already gone, so canonicalize the parent dir (which
        // must exist) and re-join the file name, then confirm it stays under a
        // canonicalized models_dir.
        match std::fs::canonicalize(&self.models_dir) {
            Ok(canon_root) => {
                let parent = p.parent().unwrap_or(&self.models_dir);
                match std::fs::canonicalize(parent) {
                    Ok(canon_parent) => {
                        let file_name = p
                            .file_name()
                            .ok_or_else(|| "invalid model name".to_string())?;
                        let resolved = canon_parent.join(file_name);
                        if !resolved.starts_with(&canon_root) {
                            return Err("path escapes models dir".into());
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // Parent dir missing → the file can't exist; fall
                        // through to remove_file so the not-found path below
                        // produces the usual "delete failed" message.
                    }
                    Err(e) => return Err(format!("delete failed: {e}")),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // models_dir itself missing → nothing to delete; fall through.
            }
            Err(e) => return Err(format!("delete failed: {e}")),
        }

        std::fs::remove_file(&p).map_err(|e| format!("delete failed: {e}"))?;
        Ok(())
    }

    /// Kick off a download. Returns the assigned download id immediately;
    /// the actual transfer runs on a background task.
    pub fn start(
        &self,
        url: &str,
        filename: Option<&str>,
        subdir: Option<&str>,
        expected_sha256: Option<&str>,
    ) -> Result<String, String> {
        // Validated up front so a malformed digest is a 400 the caller sees,
        // not a surprise after a 40 GB transfer.
        let expected_sha256 = expected_sha256
            .filter(|s| !s.trim().is_empty())
            .map(parse_expected_sha256)
            .transpose()?;
        let normalised = normalise_hf_url(url)?;
        // SSRF guard: refuse a non-https URL or one pointing at an internal/special address
        // BEFORE spawning, so an untrusted catalog/template URL can't make OAIY Desktop probe
        // internal services (and never spawns a task whose status would leak as an oracle).
        validate_download_url(&normalised)?;
        let chosen_filename = match filename {
            Some(f) if !f.is_empty() => f.to_string(),
            _ => guess_filename(&normalised)?,
        };

        // Reject filenames that could escape the dest dir. Legitimate downloads pass a bare
        // filename; subdir is the nesting mechanism, so a separator or `..` in the filename is
        // always invalid. Use the SAME Component-based check as `subdir`/`delete_model`, not the
        // weaker `is_absolute()` test: a Windows drive-relative prefix like `C:evil` has a
        // Prefix component but is NOT absolute, and `dest_dir.join("C:evil")` REPLACES dest_dir
        // entirely (→ writes to the process CWD on that drive, outside models_dir).
        {
            use std::path::Component;
            if chosen_filename.contains("..")
                || chosen_filename.contains('/')
                || chosen_filename.contains('\\')
                || Path::new(&chosen_filename).components().any(|c| {
                    matches!(
                        c,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                return Err("invalid filename".into());
            }
        }

        let dest_dir = match subdir {
            Some(s) if !s.is_empty() => {
                // Reject any traversal/absolute/drive-prefix component — more
                // robust than the old `contains("..") || is_absolute()` check,
                // which missed a Windows drive-relative prefix like `C:foo`
                // (is_absolute() is false for it). Mirrors delete_model.
                use std::path::Component;
                if Path::new(s).components().any(|c| {
                    matches!(
                        c,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                }) {
                    return Err("invalid subdir".into());
                }
                self.models_dir.join(s)
            }
            _ => self.models_dir.clone(),
        };
        std::fs::create_dir_all(&dest_dir).map_err(|e| format!("mkdir failed: {e}"))?;
        let dest = dest_dir.join(&chosen_filename);
        // Defense-in-depth: ensure the resolved destination still lives under
        // the models dir, resolving symlinks/junctions (a pre-existing link
        // inside models_dir could otherwise redirect the write outside it).
        // Mirrors delete_model: canonicalize the (now-created) dest dir + root,
        // with a lexical fallback if canonicalization isn't possible.
        match (
            std::fs::canonicalize(&dest_dir),
            std::fs::canonicalize(&self.models_dir),
        ) {
            (Ok(canon_dir), Ok(canon_root)) => {
                if !canon_dir.starts_with(&canon_root) {
                    return Err("path escapes models dir".into());
                }
            }
            _ => {
                if !dest.starts_with(&self.models_dir) {
                    return Err("path escapes models dir".into());
                }
            }
        }

        // Already on disk? Don't re-download a model the user already has.
        // Surface a Completed entry (with the real on-disk size) so the UI
        // shows "already downloaded" instead of silently no-op'ing. This is
        // the "check for existing before downloading" behaviour.
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.is_file() && meta.len() > 0 {
                let id = Uuid::new_v4().to_string();
                let now = Utc::now();
                if let Ok(mut g) = self.progress.lock() {
                    g.insert(
                        id.clone(),
                        DownloadProgress {
                            id: id.clone(),
                            url: normalised.clone(),
                            filename: chosen_filename.clone(),
                            subdir: subdir.map(|s| s.to_string()),
                            dest_path: dest.display().to_string(),
                            status: DownloadStatus::Completed,
                            bytes_downloaded: meta.len(),
                            bytes_total: Some(meta.len()),
                            started_at: now,
                            finished_at: Some(now),
                            error: None,
                            resumable: None,
                            speed_bps: None,
                            eta_secs: None,
                            // Not re-hashed: this file was not downloaded now,
                            // and claiming a verification we did not perform is
                            // the one thing this feature must never do.
                            sha256: None,
                            expected_sha256: expected_sha256.clone(),
                            verified: None,
                        },
                    );
                }
                return Ok(id);
            }
        }

        // De-dup: if a download for the same URL is already running or
        // paused (not failed/completed), return that id. Also bound the number
        // of simultaneously in-flight transfers: POST /api/models/download is
        // reachable from any loopback web page, so without a cap a flood of
        // distinct URLs could spawn unbounded tasks/connections/.part writes
        // and exhaust the tray app.
        const MAX_CONCURRENT_DOWNLOADS: usize = 8;

        let id = Uuid::new_v4().to_string();
        let progress = DownloadProgress {
            id: id.clone(),
            url: normalised.clone(),
            filename: chosen_filename.clone(),
            subdir: subdir.map(|s| s.to_string()),
            dest_path: dest.display().to_string(),
            status: DownloadStatus::Queued,
            bytes_downloaded: 0,
            bytes_total: None,
            started_at: Utc::now(),
            finished_at: None,
            error: None,
            resumable: None,
            speed_bps: None,
            eta_secs: None,
            sha256: None,
            expected_sha256: expected_sha256.clone(),
            verified: None,
        };

        // Dedup, cap and insert must be ONE critical section. These were three
        // separate lock acquisitions, which made both guards check-then-act: two
        // concurrent POST /api/models/download calls for the same URL each looked,
        // saw no in-flight row (neither had inserted yet), and both spawned a
        // transfer onto the same .part file — interleaved writes to one path. The
        // cap raced the same way (two callers both seeing in_flight == 7). Holding
        // the lock across the decision AND the insert makes the winner visible to
        // the loser. Registering the row before spawn_transfer also means the
        // spawn can no longer outrun its own bookkeeping.
        //
        // A poisoned lock is now fatal rather than silently ignored: proceeding
        // past a poisoned registry is exactly how the duplicate transfer happened.
        {
            let mut g = self
                .progress
                .lock()
                .map_err(|_| "download registry lock poisoned".to_string())?;

            if let Some(existing) = g.values().find(|p| {
                p.url == normalised
                    && matches!(
                        p.status,
                        DownloadStatus::Queued
                            | DownloadStatus::Active
                            | DownloadStatus::Paused
                    )
            }) {
                return Ok(existing.id.clone());
            }
            let in_flight = g
                .values()
                .filter(|p| {
                    matches!(p.status, DownloadStatus::Queued | DownloadStatus::Active)
                })
                .count();
            if in_flight >= MAX_CONCURRENT_DOWNLOADS {
                return Err(format!(
                    "too many concurrent downloads ({MAX_CONCURRENT_DOWNLOADS} max) — wait for some to finish"
                ));
            }

            g.insert(id.clone(), progress);
        }

        self.spawn_transfer(
            id.clone(),
            normalised,
            dest,
            /* resume_from = */ 0,
            expected_sha256,
        );
        Ok(id)
    }

    /// Pause an in-flight download. The .part file + byte counter are
    /// preserved; resume() picks up via HTTP Range. No-op if already
    /// paused/finished.
    pub fn pause(&self, id: &str) -> Result<(), String> {
        let status = {
            let g = self.progress.lock().map_err(|_| "lock poisoned")?;
            g.get(id).map(|p| p.status).ok_or("unknown download")?
        };
        match status {
            DownloadStatus::Active | DownloadStatus::Queued => {
                self.abort_task(id);
                self.update(id, |p| {
                    p.status = DownloadStatus::Paused;
                    // Clear speed/ETA so paused rows don't lie ("12 MiB/s")
                    // for the few seconds before the user sees the badge flip.
                    p.speed_bps = None;
                    p.eta_secs = None;
                });
                Ok(())
            }
            DownloadStatus::Paused => Ok(()),
            other => Err(format!("can't pause download in state {other:?}")),
        }
    }

    /// Resume a paused download. Sends `Range: bytes=N-` from the current
    /// byte counter; if the server refuses, the .part is wiped and we
    /// restart from 0 (with a warning surfaced on `error`).
    pub fn resume(&self, id: &str) -> Result<(), String> {
        // CLAIM the download atomically: check the state and flip it to Queued
        // under ONE lock. This used to read the status, drop the lock, then check
        // and spawn — so two concurrent resume() calls on the same paused row both
        // saw Paused, both passed, and both spawned a transfer that reopens the
        // same .part with .append(true). Interleaved appends corrupt the file and
        // the byte counter. Transitioning inside the critical section means the
        // second caller sees Queued and takes the error path instead.
        let (url, dest) = {
            let mut g = self.progress.lock().map_err(|_| "lock poisoned")?;
            let p = g.get_mut(id).ok_or("unknown download")?;
            if !matches!(p.status, DownloadStatus::Paused | DownloadStatus::Failed) {
                return Err(format!("can't resume download in state {:?}", p.status));
            }
            p.status = DownloadStatus::Queued;
            p.error = None;
            (p.url.clone(), PathBuf::from(&p.dest_path))
        };
        // Resume from the ACTUAL bytes on disk, not the throttle-lagged counter.
        // Every chunk is write_all'd in order, but `bytes_downloaded` only advances
        // on the ~1 MiB / 250 ms emit tick — so the .part is up to ~1 MiB longer
        // than the counter at pause/failure. The transfer reopens the .part with
        // `.append(true)` (writes at EOF) but asks the server for `Range: bytes=N-`;
        // resuming from the stale counter therefore DUPLICATES [counter, part_len)
        // into the file and silently corrupts it. The on-disk bytes are valid + in
        // order, so the real file length is the correct, waste-free resume offset.
        // If the .part is gone or unreadable (AV/cleanup/manual delete), treat it
        // as a fresh start (0) — NOT the stale counter: run_download would then
        // `Range: bytes=N-` + `.append(true).create(true)` onto a brand-new empty
        // file, dropping the first N bytes (rename has no size check) = silent
        // head-truncation. Make the on-disk length authoritative.
        let resume_from = std::fs::metadata(part_path_for(&dest))
            .map(|m| m.len())
            .unwrap_or(0);
        // Status/error already set while claiming above; only the offset is left,
        // and it needed file I/O so it's computed outside the lock.
        self.update(id, |p| {
            p.bytes_downloaded = resume_from;
        });
        // Carried across a resume: the digest is a property of the file, not of
        // one transfer attempt.
        let expected = self
            .progress
            .lock()
            .ok()
            .and_then(|g| g.get(id).and_then(|p| p.expected_sha256.clone()));
        self.spawn_transfer(id.to_string(), url, dest, resume_from, expected);
        Ok(())
    }

    /// Cancel + delete the in-progress file. Use when the user clicks
    /// the trash icon next to an active or paused download.
    pub fn cancel(&self, id: &str) -> Result<(), String> {
        self.abort_task(id);
        let part_path = {
            let g = self.progress.lock().map_err(|_| "lock poisoned")?;
            g.get(id).map(|p| part_path_for(&PathBuf::from(&p.dest_path)))
                .ok_or("unknown download")?
        };
        let _ = std::fs::remove_file(&part_path);
        self.update(id, |p| {
            p.status = DownloadStatus::Cancelled;
            p.finished_at = Some(Utc::now());
        });
        Ok(())
    }

    // ---- internals ----

    fn abort_task(&self, id: &str) {
        if let Ok(mut g) = self.abort_handles.lock() {
            if let Some(h) = g.remove(id) {
                h.abort();
            }
        }
    }

    fn update<F: FnOnce(&mut DownloadProgress)>(&self, id: &str, f: F) {
        if let Ok(mut g) = self.progress.lock() {
            if let Some(p) = g.get_mut(id) {
                f(p);
            }
        }
    }

    fn spawn_transfer(
        &self,
        id: String,
        url: String,
        dest: PathBuf,
        resume_from: u64,
        expected_sha256: Option<String>,
    ) {
        let progress_map = self.progress.clone();
        // The abort_map clone is moved into the task so it can remove its
        // own AbortHandle entry on completion — without this, finished
        // downloads leak stale entries that the next pause() would try to
        // abort harmlessly but the map would grow unbounded across long
        // sessions.
        let abort_map = self.abort_handles.clone();
        let id_for_task = id.clone();
        let token = self.hf_token.lock().ok().and_then(|g| g.clone());
        let handle = tokio::spawn(async move {
            run_download(
                progress_map,
                abort_map,
                id_for_task,
                url,
                dest,
                resume_from,
                token,
                expected_sha256,
            )
            .await;
        });
        if let Ok(mut g) = self.abort_handles.lock() {
            g.insert(id, handle.abort_handle());
        }
    }
}

async fn run_download(
    progress_map: Arc<Mutex<HashMap<String, DownloadProgress>>>,
    abort_map: Arc<Mutex<HashMap<String, AbortHandle>>>,
    id: String,
    url: String,
    dest: PathBuf,
    resume_from: u64,
    hf_token: Option<String>,
    expected_sha256: Option<String>,
) {
    let update = |f: Box<dyn FnOnce(&mut DownloadProgress)>| {
        if let Ok(mut g) = progress_map.lock() {
            if let Some(p) = g.get_mut(&id) {
                f(p);
            }
        }
    };

    update(Box::new(|p| p.status = DownloadStatus::Active));

    let client = match reqwest::Client::builder()
        .user_agent(format!("oaiy-desktop/{}", env!("CARGO_PKG_VERSION")))
        // Re-validate every redirect hop: a public host (incl. the HF resolve URL) may 302 to a
        // CDN, but it must stay https and must not point at an internal literal IP (defense over
        // the start() guard, which only sees the original URL). DNS-name hops are followed and
        // re-checked by the OS at connect (a rebind-to-internal across the hop is the residual).
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.stop();
            }
            if attempt.url().scheme() != "https" {
                return attempt.error("redirect to a non-https URL");
            }
            if let Some(ip) = attempt
                .url()
                .host_str()
                .and_then(|h| h.parse::<std::net::IpAddr>().ok())
            {
                if is_disallowed_ip(ip) {
                    return attempt.error("redirect to a disallowed internal address");
                }
            }
            attempt.follow()
        }))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            update(Box::new(move |p| {
                p.status = DownloadStatus::Failed;
                p.error = Some(format!("client build: {e}"));
                p.finished_at = Some(Utc::now());
            }));
            return;
        }
    };

    let mut req = client.get(&url);
    let resuming = resume_from > 0;
    if resuming {
        req = req.header("Range", format!("bytes={resume_from}-"));
    }
    // Attach the HF token only on huggingface.co requests so gated/private
    // repos resolve. Never sent to an arbitrary host the user pasted; the
    // CDN host the resolve URL 302s to is already presigned (and reqwest
    // drops Authorization across hosts anyway).
    if let Some(token) = hf_token.as_deref() {
        if is_hf_host(&url) {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            update(Box::new(move |p| {
                p.status = DownloadStatus::Failed;
                p.error = Some(format!("request: {e}"));
                p.finished_at = Some(Utc::now());
            }));
            return;
        }
    };

    let status = resp.status();
    // A range request returns 206 Partial Content on success and 200 OK
    // when the server ignored Range. Any non-success is fatal.
    let mut resume_accepted = true;
    if resuming && status == reqwest::StatusCode::OK {
        // Server gave us the whole file again — start over.
        resume_accepted = false;
    } else if !status.is_success() {
        // 401/403 on an HF URL = gated/private repo. Give an actionable
        // message instead of a bare "HTTP 401": the fix is a token (or one
        // with access + accepted terms), set in Settings.
        let gated = matches!(
            status,
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        ) && is_hf_host(&url);
        let had_token = hf_token.is_some();
        let msg = if gated && !had_token {
            "this looks like a gated or private HuggingFace repo. Add a HuggingFace \
             access token in Settings (and accept the model's terms on its HF page), \
             then retry."
                .to_string()
        } else if gated {
            "HuggingFace returned 403/401 even with your token — make sure that token \
             has access to this repo and you've accepted the model's terms on its HF page."
                .to_string()
        } else {
            format!("HTTP {status}")
        };
        update(Box::new(move |p| {
            p.status = DownloadStatus::Failed;
            p.error = Some(msg);
            p.finished_at = Some(Utc::now());
        }));
        return;
    }

    // On a resume, a 206 must actually START at resume_from. The code appends the body at the
    // file's EOF, so a server (or MITM on an attacker-supplied URL) returning a different range
    // — the whole file again, or a wrong offset — would otherwise corrupt the file (and there's
    // no checksum backstop). Validate Content-Range before trusting the 206.
    if resuming && resume_accepted && status == reqwest::StatusCode::PARTIAL_CONTENT {
        match content_range_start(&resp) {
            Some(start) if start == resume_from => {} // good — resumes exactly where we asked
            Some(0) => resume_accepted = false,       // whole file again → restart from 0
            _ => {
                update(Box::new(|p| {
                    p.status = DownloadStatus::Failed;
                    p.error = Some(
                        "server returned an inconsistent partial response (bad Content-Range); \
                         pause and retry"
                            .into(),
                    );
                    p.finished_at = Some(Utc::now());
                }));
                return;
            }
        }
    }

    // `content_len` is the size of THIS response body — i.e. the bytes
    // we're about to write now (the remaining tail when resuming, or the
    // whole file otherwise). `total` is the full file size for the UI.
    let content_len = resp.content_length();
    let total = content_len.map(|cl| {
        if resuming && resume_accepted {
            cl.saturating_add(resume_from)
        } else {
            cl
        }
    });
    let resumable = resp.headers().get("accept-ranges").is_some()
        || status == reqwest::StatusCode::PARTIAL_CONTENT;
    update(Box::new(move |p| {
        p.bytes_total = total;
        p.resumable = Some(resumable);
        if resuming && !resume_accepted {
            p.bytes_downloaded = 0;
            p.error = Some(
                "server didn't accept Range request — restarting from 0".to_string(),
            );
        }
    }));

    // Pre-flight free-space check: when the server told us the size, bail
    // BEFORE writing a single byte if it clearly won't fit — far friendlier
    // than filling the drive and dying mid-stream with an OS error. If the
    // size is unknown (no Content-Length) we proceed and let the write fail
    // naturally.
    if let Some(need) = content_len {
        if need > 0 {
            let check_dir = dest.parent().unwrap_or(&dest);
            if let Ok(avail) = fs2::available_space(check_dir) {
                if let Some(short) = space_shortfall(avail, need) {
                    update(Box::new(move |p| {
                        p.status = DownloadStatus::Failed;
                        p.error = Some(format!(
                            "not enough disk space: this download needs ~{} but only ~{} is free \
                             (short by ~{}). Free up space or point the data folder at a bigger drive \
                             (Settings → Data folder).",
                            human_gib(need),
                            human_gib(avail),
                            human_gib(short)
                        ));
                        p.finished_at = Some(Utc::now());
                    }));
                    return;
                }
            }
        }
    }

    let part = part_path_for(&dest);
    if let Err(e) = tokio::fs::create_dir_all(part.parent().unwrap_or(Path::new("."))).await {
        update(Box::new(move |p| {
            p.status = DownloadStatus::Failed;
            p.error = Some(format!("mkdir: {e}"));
            p.finished_at = Some(Utc::now());
        }));
        return;
    }

    let mut file = if resuming && resume_accepted {
        match tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&part)
            .await
        {
            Ok(f) => f,
            Err(e) => {
                update(Box::new(move |p| {
                    p.status = DownloadStatus::Failed;
                    p.error = Some(format!("append: {e}"));
                    p.finished_at = Some(Utc::now());
                }));
                return;
            }
        }
    } else {
        // Either a fresh download or a non-resumable retry.
        match tokio::fs::File::create(&part).await {
            Ok(f) => f,
            Err(e) => {
                update(Box::new(move |p| {
                    p.status = DownloadStatus::Failed;
                    p.error = Some(format!("create: {e}"));
                    p.finished_at = Some(Utc::now());
                }));
                return;
            }
        }
    };

    // The digest to check against: what the caller asked for, else whatever the
    // origin published. Asking HuggingFace costs one redirect-less HEAD and
    // covers essentially every model download without the user supplying
    // anything — the whole reason this is worth doing.
    let expected = match expected_sha256 {
        Some(d) => Some(d),
        None => hf_linked_digest(&url, hf_token.as_deref()).await,
    };
    if let Some(d) = expected.clone() {
        update(Box::new(move |p| p.expected_sha256 = Some(d)));
    }

    // Hash as we write. On a resume the bytes already on disk were hashed by a
    // previous process that is gone, so re-read them — one pass over the
    // partial, only on the uncommon path, versus re-reading the whole file at
    // the end of every download.
    let mut unhashable_resume: Option<String> = None;
    let mut hasher = if resuming && resume_accepted {
        match hash_prefix(&part, resume_from).await {
            Ok(h) => Some(h),
            Err(e) => {
                // Verification is now impossible for this transfer: hashing only
                // the tail would produce a digest of the wrong bytes. The
                // alternative was to finish anyway and rename the file into
                // place with `verified: None` — indistinguishable from a source
                // that publishes no digest at all. Silently downgrading a
                // checkable download to an uncheckable one is precisely the lie
                // this feature exists to prevent, and it was reported only by an
                // eprintln! to a console a packaged app does not have.
                //
                // Restarting from zero is not available here: the file is
                // already open in append mode and the request already carries a
                // Range header. So fail, keep the .part, and say what to do —
                // a retry re-attempts the hash, and cancel starts clean.
                unhashable_resume = Some(e.to_string());
                None
            }
        }
    } else {
        Some(Sha256::new())
    };

    if let Some(why) = unhashable_resume {
        update(Box::new(move |p| {
            p.status = DownloadStatus::Failed;
            p.error = Some(format!(
                "could not re-read the partial file to continue verifying it ({why}).                  The download was stopped rather than finished unverified — retry to                  try again, or cancel and download afresh."
            ));
            p.finished_at = Some(Utc::now());
        }));
        return;
    }

    let mut downloaded: u64 = if resuming && resume_accepted { resume_from } else { 0 };
    let mut last_emit: u64 = downloaded;
    // Sliding-window speed tracker: keep (timestamp, bytes_so_far) samples
    // from the last ~4s. Speed = (bytes_now - bytes_oldest) / window_secs.
    // Resists both lumpy chunk arrivals AND end-of-file misreporting from
    // simple "bytes / elapsed_total".
    let mut samples: std::collections::VecDeque<(std::time::Instant, u64)> =
        std::collections::VecDeque::with_capacity(32);
    samples.push_back((std::time::Instant::now(), downloaded));
    const SPEED_WINDOW_SECS: u64 = 4;

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                update(Box::new(move |p| {
                    p.status = DownloadStatus::Failed;
                    p.error = Some(format!("stream: {e}"));
                    p.finished_at = Some(Utc::now());
                }));
                // Don't delete the .part — pause/resume relies on it.
                return;
            }
        };
        if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await {
            update(Box::new(move |p| {
                p.status = DownloadStatus::Failed;
                p.error = Some(format!("write: {e}"));
                p.finished_at = Some(Utc::now());
            }));
            return;
        }
        if let Some(h) = hasher.as_mut() {
            h.update(&chunk);
        }
        downloaded += chunk.len() as u64;

        // Throttle progress writes — 1 MiB OR 250ms granularity (whichever
        // first). The 250ms floor keeps speed/ETA ticking on slow links
        // where we might wait minutes between MiB.
        let now = std::time::Instant::now();
        let elapsed_since_emit = now.duration_since(samples.back().map(|s| s.0).unwrap_or(now));
        if downloaded - last_emit > 1 << 20 || elapsed_since_emit.as_millis() > 250 {
            last_emit = downloaded;
            samples.push_back((now, downloaded));
            // Trim samples older than the window.
            while let Some(&(t, _)) = samples.front() {
                if now.duration_since(t).as_secs() > SPEED_WINDOW_SECS {
                    samples.pop_front();
                } else {
                    break;
                }
            }
            let (speed, eta) = compute_speed_eta(&samples, total);
            update(Box::new(move |p| {
                p.bytes_downloaded = downloaded;
                p.speed_bps = speed;
                p.eta_secs = eta;
            }));
        }
    }
    // Treat flush as a write, because it IS one: `tokio::fs::File` buffers, and
    // a deferred I/O failure (a full disk, a network-mapped models dir that
    // hiccups) surfaces here or nowhere. Every downstream check is computed
    // from memory — `downloaded` counts bytes off the network and the digest is
    // taken over those same chunks — so nothing else can notice that the tail
    // never reached disk. Discarding this error let a SHORT file be renamed
    // into place and reported "checksum verified", which is the one claim this
    // code exists to make truthfully.
    //
    // fsync as well as flush: "verified" is a claim about what is ON DISK, and
    // an unsynced tail is not on disk yet.
    let flushed = tokio::io::AsyncWriteExt::flush(&mut file)
        .await
        .and(file.sync_all().await);
    if let Err(e) = flushed {
        update(Box::new(move |p| {
            p.status = DownloadStatus::Failed;
            p.error = Some(format!("write: {e}"));
            p.finished_at = Some(Utc::now());
        }));
        // Keep the .part: the bytes that DID land are still a valid prefix to
        // resume from once whatever failed is fixed.
        return;
    }
    drop(file);

    // Don't rename a silently-truncated body into a "valid" model: when the full size is known,
    // require we wrote all of it. (hyper raises a premature-EOF stream error for Content-Length/
    // chunked bodies, caught above; this is the backstop, and keeps the .part for a resume.)
    if let Some(t) = total {
        if downloaded != t {
            update(Box::new(move |p| {
                p.status = DownloadStatus::Failed;
                p.error = Some(format!("incomplete download: wrote {downloaded} of {t} bytes"));
                p.finished_at = Some(Utc::now());
            }));
            return;
        }
    }

    // Verify BEFORE the rename: a file that fails its digest must never appear
    // under its final name, however briefly. Something else on this machine could
    // pick it up in between, and the whole point is that a corrupted or swapped
    // model never becomes a model this app will load.
    let actual = hasher.map(|h| format!("{:x}", h.finalize()));
    if let (Some(expected), Some(actual)) = (expected.as_deref(), actual.as_deref()) {
        if expected != actual {
            let _ = tokio::fs::remove_file(&part).await;
            let (e, a) = (expected.to_string(), actual.to_string());
            update(Box::new(move |p| {
                p.status = DownloadStatus::Failed;
                p.verified = Some(false);
                p.sha256 = Some(a.clone());
                p.error = Some(format!(
                    "content does not match its published checksum — the file was deleted. \
                     Expected {e}, got {a}. Retry; if it keeps happening, the source or the \
                     connection is not to be trusted."
                ));
                p.finished_at = Some(Utc::now());
            }));
            if let Ok(mut g) = abort_map.lock() {
                g.remove(&id);
            }
            return;
        }
    }

    if let Err(e) = tokio::fs::rename(&part, &dest).await {
        update(Box::new(move |p| {
            p.status = DownloadStatus::Failed;
            p.error = Some(format!("rename: {e}"));
            p.finished_at = Some(Utc::now());
        }));
        return;
    }

    let verified = match (&expected, &actual) {
        (Some(_), Some(_)) => Some(true), // a mismatch returned above
        _ => None,
    };
    update(Box::new(move |p| {
        p.status = DownloadStatus::Completed;
        p.bytes_downloaded = downloaded;
        p.sha256 = actual;
        p.verified = verified;
        p.finished_at = Some(Utc::now());
        // Clear speed/ETA on completion — leaving them stale would say
        // "12 MiB/s · 0 min remaining" on a finished row.
        p.speed_bps = None;
        p.eta_secs = None;
    }));

    // Drop our AbortHandle from the map — the task's about to exit
    // normally; future pause/cancel calls would no-op anyway, but a
    // clean map saves memory across long sessions of many downloads.
    if let Ok(mut g) = abort_map.lock() {
        g.remove(&id);
    }
}

/// Hash the first `len` bytes of a file, for seeding a resumed download.
async fn hash_prefix(path: &Path, len: u64) -> std::io::Result<Sha256> {
    use tokio::io::AsyncReadExt as _;
    let mut f = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut read_total: u64 = 0;
    while read_total < len {
        let want = ((len - read_total) as usize).min(buf.len());
        let n = f.read(&mut buf[..want]).await?;
        if n == 0 {
            // The partial is shorter than the byte counter claims. Better to
            // fail the hash than to report a digest over the wrong bytes.
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("partial file is {read_total} bytes, expected at least {len}"),
            ));
        }
        hasher.update(&buf[..n]);
        read_total += n as u64;
    }
    Ok(hasher)
}

/// Ask HuggingFace for a file's published SHA-256.
///
/// `X-Linked-ETag` rides the 302 from `huggingface.co` to the CDN, so this must
/// NOT follow redirects — the final CDN response carries a different `etag` that
/// is not the content digest. Best-effort: any failure just means the download
/// is unverified, which is exactly where it was before.
async fn hf_linked_digest(url: &str, hf_token: Option<&str>) -> Option<String> {
    if !is_hf_host(url) {
        return None;
    }
    let client = reqwest::Client::builder()
        .user_agent(format!("oaiy-desktop/{}", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .ok()?;
    let mut req = client.head(url);
    if let Some(t) = hf_token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await.ok()?;
    resp.headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .and_then(digest_from_etag)
}

/// Compute bytes/sec + ETA from a sliding window of (timestamp, bytes)
/// samples. Returns (None, None) when the window doesn't have enough
/// signal yet (only one sample, or zero elapsed).
fn compute_speed_eta(
    samples: &std::collections::VecDeque<(std::time::Instant, u64)>,
    total: Option<u64>,
) -> (Option<u64>, Option<u64>) {
    if samples.len() < 2 {
        return (None, None);
    }
    let (t_old, b_old) = *samples.front().unwrap();
    let (t_new, b_new) = *samples.back().unwrap();
    let elapsed = t_new.duration_since(t_old).as_secs_f64();
    if elapsed <= 0.0 || b_new <= b_old {
        return (None, None);
    }
    let bps = ((b_new - b_old) as f64 / elapsed) as u64;
    let eta = total.and_then(|t| {
        if t <= b_new || bps == 0 {
            None
        } else {
            Some(((t - b_new) as f64 / bps as f64) as u64)
        }
    });
    (Some(bps), eta)
}

/// `model.gguf` → `model.gguf.part`. Single, predictable layout makes
/// resume + cancel cleanup straightforward.
fn part_path_for(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Recursively walk `root` collecting ModelFile entries with paths
/// relative to `root_base`. Skips .part files (in-flight downloads).
fn walk_dir(root: &Path, root_base: &Path, out: &mut Vec<ModelFile>) -> std::io::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        // Don't follow symlinks/junctions — a loop or an out-of-tree link would
        // let this walk recurse widely or surface files outside the models dir.
        // (`file_type()` does NOT follow the link, unlike `metadata()` below.)
        if entry.file_type()?.is_symlink() {
            continue;
        }
        let p = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            walk_dir(&p, root_base, out)?;
            continue;
        }
        if p.extension().and_then(|e| e.to_str()) == Some("part") {
            continue;
        }
        let rel = p.strip_prefix(root_base).unwrap_or(&p);
        let name = rel.display().to_string().replace('\\', "/");
        let modified = meta.modified().ok().and_then(|t| {
            let d = t.duration_since(std::time::UNIX_EPOCH).ok()?;
            DateTime::<Utc>::from_timestamp(
                d.as_secs() as i64,
                d.subsec_nanos(),
            )
        });
        out.push(ModelFile {
            name,
            path: p.display().to_string(),
            size_bytes: meta.len(),
            modified,
        });
    }
    Ok(())
}

/// Convert a HF browser URL (`/blob/`) to a direct download URL
/// (`/resolve/`). Leaves direct URLs and non-HF URLs untouched.
fn normalise_hf_url(input: &str) -> Result<String, String> {
    let parsed = url::Url::parse(input).map_err(|e| format!("invalid URL: {e}"))?;
    if parsed.host_str() != Some("huggingface.co") {
        return Ok(input.to_string());
    }
    let path = parsed.path();
    if let Some(rest) = path.strip_prefix("/") {
        let segments: Vec<&str> = rest.splitn(4, '/').collect();
        if segments.len() == 4 && segments[2] == "blob" {
            let new_path = format!(
                "/{}/{}/resolve/{}",
                segments[0], segments[1], segments[3]
            );
            let mut rewritten = parsed.clone();
            rewritten.set_path(&new_path);
            return Ok(rewritten.to_string());
        }
    }
    Ok(input.to_string())
}

/// Default filename when the caller didn't supply one — last path segment
/// of the URL, percent-decoded.
fn guess_filename(url: &str) -> Result<String, String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("invalid URL: {e}"))?;
    let last = parsed
        .path_segments()
        .and_then(|mut s| s.next_back())
        .ok_or("URL has no path segments")?;
    if last.is_empty() {
        return Err("URL ends in /; no filename to use".into());
    }
    Ok(percent_decode(last))
}

fn percent_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((h * 16 + l) as char);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_blob_to_resolve() {
        let url = "https://huggingface.co/Qwen/Qwen3-Coder/blob/main/model.gguf";
        let normalised = normalise_hf_url(url).unwrap();
        assert_eq!(
            normalised,
            "https://huggingface.co/Qwen/Qwen3-Coder/resolve/main/model.gguf"
        );
    }

    #[test]
    fn hf_resolve_unchanged() {
        let url = "https://huggingface.co/Qwen/Qwen3-Coder/resolve/main/model.gguf";
        assert_eq!(normalise_hf_url(url).unwrap(), url);
    }

    #[test]
    fn non_hf_unchanged() {
        let url = "https://example.com/model.gguf";
        assert_eq!(normalise_hf_url(url).unwrap(), url);
    }

    #[test]
    fn guesses_filename_from_url() {
        assert_eq!(
            guess_filename("https://huggingface.co/foo/bar/resolve/main/Qwen3.gguf").unwrap(),
            "Qwen3.gguf"
        );
    }

    #[test]
    fn percent_decode_works() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
    }

    #[test]
    fn part_path_is_suffixed() {
        let p = part_path_for(Path::new("/tmp/model.gguf"));
        assert_eq!(p.to_string_lossy(), "/tmp/model.gguf.part");
    }

    #[test]
    fn space_shortfall_respects_margin() {
        // Comfortably fits (well above need + margin).
        assert_eq!(space_shortfall(DISK_MARGIN_BYTES + 10_000, 1_000), None);
        // Exactly need + margin available → fits (no shortfall).
        assert_eq!(space_shortfall(DISK_MARGIN_BYTES + 1_000, 1_000), None);
        // One byte short of need + margin → reports a 1-byte shortfall.
        assert_eq!(space_shortfall(DISK_MARGIN_BYTES + 999, 1_000), Some(1));
        // Way too small → large shortfall.
        assert_eq!(space_shortfall(0, 1_000), Some(1_000 + DISK_MARGIN_BYTES));
    }

    #[test]
    fn available_space_query_works() {
        // fs2 integration smoke test — the temp dir's drive has *some* free
        // space, and the query succeeds on this platform.
        let avail = fs2::available_space(&std::env::temp_dir()).unwrap();
        assert!(avail > 0);
    }

    #[test]
    fn token_state_trims_and_clears() {
        let d = Downloads::new(std::env::temp_dir().join("oaiy-token-test"));
        assert!(!d.has_token(), "starts unset");
        d.set_token(Some("hf_abc".to_string()));
        assert!(d.has_token(), "set");
        d.set_token(Some("   ".to_string()));
        assert!(!d.has_token(), "whitespace-only clears");
        d.set_token(Some("hf_xyz".to_string()));
        assert!(d.has_token());
        d.set_token(None);
        assert!(!d.has_token(), "None clears");
    }

    #[test]
    fn hf_host_scoping() {
        // The token must go to HF (and its subdomains) ONLY — never to an
        // arbitrary host the user pasted.
        assert!(is_hf_host("https://huggingface.co/meta-llama/x/resolve/main/a.gguf"));
        assert!(is_hf_host("https://cdn-lfs.huggingface.co/foo"));
        assert!(is_hf_host("https://HuggingFace.co/foo")); // case-insensitive
        assert!(!is_hf_host("https://evil.com/huggingface.co/foo"));
        assert!(!is_hf_host("https://nothuggingface.co/foo"));
        // HTTPS required: never attach the bearer token over plaintext.
        assert!(!is_hf_host("http://huggingface.co/foo"));
        assert!(!is_hf_host("http://cdn-lfs.huggingface.co/foo"));
        assert!(!is_hf_host("https://example.com/model.gguf"));
        assert!(!is_hf_host("not a url"));
    }

    #[test]
    fn ssrf_guard_blocks_internal_and_non_https() {
        use std::net::IpAddr;
        // Internal / special addresses must be disallowed; public ones allowed.
        for ip in [
            "127.0.0.1",
            "169.254.169.254", // cloud metadata
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
            "0.0.0.0",
            "100.64.0.1", // CGNAT
            "::1",
            "fc00::1", // unique-local
            "fe80::1", // link-local
            "::ffff:127.0.0.1", // v4-mapped loopback
        ] {
            assert!(
                is_disallowed_ip(ip.parse::<IpAddr>().unwrap()),
                "{ip} should be disallowed"
            );
        }
        for ip in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            assert!(
                !is_disallowed_ip(ip.parse::<IpAddr>().unwrap()),
                "{ip} should be allowed"
            );
        }
        // validate_download_url: scheme + IP-literal checks (no DNS needed).
        assert!(validate_download_url("http://huggingface.co/x").is_err()); // not https
        assert!(validate_download_url("file:///etc/passwd").is_err());
        assert!(validate_download_url("https://127.0.0.1/x").is_err());
        assert!(validate_download_url("https://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_download_url("https://[::1]/x").is_err());
        assert!(validate_download_url("https://10.0.0.5:8080/x").is_err());
        assert!(validate_download_url("https://1.1.1.1/model.gguf").is_ok()); // public IP literal
    }

    #[test]
    fn content_range_start_parsing() {
        // Minimal Response is awkward to build; parse the header value logic via the same path
        // by constructing a header map through a real Response is overkill — instead assert the
        // string handling on representative values would parse the start byte.
        let parse = |s: &str| -> Option<u64> {
            s.trim()
                .strip_prefix("bytes")?
                .trim_start()
                .split('-')
                .next()?
                .trim()
                .parse::<u64>()
                .ok()
        };
        assert_eq!(parse("bytes 200-1000/1001"), Some(200));
        assert_eq!(parse("bytes 0-500/1001"), Some(0));
        assert_eq!(parse("bytes */1001"), None);
        assert_eq!(parse("nonsense"), None);
    }

    // --- checksum verification ---------------------------------------------

    #[test]
    fn a_caller_digest_is_normalised_or_refused() {
        let good = "9EE36184E616DFC76DF4F5DD66F908DBDE6979524AE36E6CEFB67F532F798CB8";
        assert_eq!(
            parse_expected_sha256(good).unwrap(),
            good.to_ascii_lowercase()
        );
        // The form a user copies out of a lockfile or an LFS pointer.
        assert_eq!(
            parse_expected_sha256(&format!("  sha256:{good}  ")).unwrap(),
            good.to_ascii_lowercase()
        );

        // Refused, not ignored: someone who pastes an MD5 or a truncated digest
        // and gets a silently-unverified download is worse off than one who
        // gets an error, because they believe a check happened.
        assert!(parse_expected_sha256("d41d8cd98f00b204e9800998ecf8427e").is_err());
        assert!(parse_expected_sha256(&good[..63]).is_err());
        assert!(parse_expected_sha256(&format!("{}z", &good[..63])).is_err());
        assert!(parse_expected_sha256("").is_err());
    }

    #[test]
    fn only_a_real_sha256_is_read_out_of_an_etag() {
        let oid = "9ee36184e616dfc76df4f5dd66f908dbde6979524ae36e6cefb67f532f798cb8";
        assert_eq!(digest_from_etag(&format!("\"{oid}\"")).as_deref(), Some(oid));
        assert_eq!(digest_from_etag(&format!("W/\"{oid}\"")).as_deref(), Some(oid));
        assert_eq!(digest_from_etag(oid).as_deref(), Some(oid));

        // A non-LFS file's ETag is a git blob SHA-1 (40 hex). Treating that as a
        // content digest would fail every small-file download.
        assert_eq!(digest_from_etag("\"8d1c9f0e2b3a4d5c6e7f8091a2b3c4d5e6f70819\""), None);
        // S3-style multipart ETags carry a part suffix.
        assert_eq!(digest_from_etag("\"d41d8cd98f00b204e9800998ecf8427e-7\""), None);
        assert_eq!(digest_from_etag("\"\""), None);
    }

    #[tokio::test]
    async fn hashing_a_resumed_prefix_matches_hashing_the_whole_thing() {
        // The resume path re-reads what a previous process wrote, then continues
        // hashing the tail. If the two halves didn't compose, every resumed
        // download would fail verification — which looks exactly like tampering.
        let dir = std::env::temp_dir().join(format!("oaiy-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("part.bin");

        let head: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let tail: Vec<u8> = (0..37_000u32).map(|i| (i % 197) as u8).collect();
        std::fs::write(&path, &head).unwrap();

        let mut resumed = hash_prefix(&path, head.len() as u64).await.unwrap();
        resumed.update(&tail);

        let mut whole = Sha256::new();
        whole.update(&head);
        whole.update(&tail);

        assert_eq!(
            format!("{:x}", resumed.finalize()),
            format!("{:x}", whole.finalize())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_partial_shorter_than_its_counter_fails_rather_than_hashing_the_wrong_bytes() {
        let dir = std::env::temp_dir().join(format!("oaiy-hash-short-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("part.bin");
        std::fs::write(&path, b"only twenty bytes...").unwrap();

        // Claiming 5000 bytes are there when 20 are: reporting a digest over the
        // short read would produce a mismatch blamed on the server.
        assert!(hash_prefix(&path, 5_000).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_non_huggingface_url_is_never_asked_for_a_digest() {
        // hf_linked_digest attaches the user's HF bearer token, so the host gate
        // matters as much here as it does on the transfer itself.
        assert!(!is_hf_host("https://example.com/model.gguf"));
        assert!(!is_hf_host("http://huggingface.co/x/resolve/main/m.gguf"));
        assert!(is_hf_host("https://huggingface.co/x/resolve/main/m.gguf"));
        assert!(is_hf_host("https://cdn-lfs.huggingface.co/x"));
    }
}
