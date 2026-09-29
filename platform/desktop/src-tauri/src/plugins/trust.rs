//! Package trust: who made a plugin package, and whether it is still the package they made.
//!
//! A plugin is native code that OAIY starts with the person's own permissions, so the
//! folder it lives in is part of the trust boundary. This module is the check on that
//! folder. It is a port of the verifier FormLogic Desktop had (`package_trust.rs`), made
//! to read exactly what Aokie's `crates/package-signer` writes.
//!
//! # The format
//!
//! A release bundle carries `package-manifest.json`, an envelope
//! `{format: 1, alg: "Ed25519", keyId, payloadB64, signature}`:
//!
//! - `payloadB64` is base64 (standard) of the exact payload bytes the signer signed:
//!   `{name, version, createdAt, files: [{path, sha256, size}]}`, one entry for every
//!   file in the bundle except the envelope itself, with forward-slash relative paths;
//! - `signature` is base64url (no padding) of the detached Ed25519 signature over those
//!   bytes.
//!
//! What was signed is byte for byte what is verified, so there is no canonical-JSON step
//! to get wrong.
//!
//! # Who is trusted
//!
//! Publisher keys are pinned in `resources/trusted-publishers.json`, compiled into the
//! binary: an id (the envelope's `keyId`), a name, the public key and the plugin ids the
//! key may sign. A signature that verifies under a key that is not pinned, or is pinned
//! for other plugins, is worth nothing: a publisher that may sign `aokie` may not sign a
//! plugin called `wallet`.
//!
//! # What is decided
//!
//! | Package | Release build | Developer build |
//! |---|---|---|
//! | signature and every file check out | `verified` | `verified` |
//! | a signature that does not check out | `quarantined`, never started | same |
//! | no signature, the person trusted this exact package | `trusted-local` | same |
//! | no signature | `unsigned`, never started | `unsigned-dev`, started |
//!
//! Quarantine is not something a developer build waives. A package that carries a
//! signature is claiming to be a release, and one that has drifted from it (a binary
//! copied over the top of an installed release, say) is exactly the state that must not
//! run silently. To work on a plugin, install it without `package-manifest.json`.
//!
//! **The bundle is immutable.** Verification fails on a file whose digest differs and on
//! any file the signature does not list, executable or not, as Aokie's own
//! `package-signer verify` does. That is why a plugin's writable directory is
//! `<data>/plugin-data/<id>`, outside the bundle (see `runner::plugin_data_dir`).
//!
//! # When it runs
//!
//! - when a plugin is **scanned** (`PluginRegistry::scan`), for the listing and for what
//!   the plugin may contribute. A cheap fingerprint of the folder (paths, sizes and
//!   times, no reads) decides whether the answer of the last scan still stands, so the
//!   listing that is polled every couple of seconds does not hash a hundred megabytes
//!   each time. The fingerprint cannot see a file rewritten to the same size with its
//!   time put back, so the one file a scan builds everything from, `manifest.json`, is
//!   held to the answer another way: the scan hands over the bytes it parsed
//!   ([`TrustService::assess_read`]) and the verdict only stands for a folder whose
//!   `manifest.json` has exactly those bytes;
//! - when a plugin is **installed**, on the staged copy, before it replaces anything;
//! - **immediately before each launch**, in full and never from the cache. A scan's
//!   answer is only ever a display: what decides whether a process starts is the
//!   [`LaunchPermit`] that [`TrustService::authorize_launch`] hands out, and
//!   `PluginProcess::spawn` cannot be called without one. The permit carries the folder
//!   that was checked and the manifest parsed from the very bytes that were hashed, and
//!   `spawn` starts that manifest's entry and no other: a second read of `manifest.json`
//!   could hand the launch a different file from the one the signature covers.
//!
//! The launch check and the process creation are still two steps, so a process that can
//! write to the plugin folder can in principle win a race between them (by swapping the
//! entry executable itself). The folder is in the person's own data directory: this
//! guards against a tampered download, a swapped file and a stale copy, not against
//! malware already running as the person.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::manifest::{ManifestError, PluginManifest, MAX_MANIFEST_BYTES};

/// The envelope Aokie's `package-signer` writes at the root of a bundle.
pub const PACKAGE_MANIFEST_FILE: &str = "package-manifest.json";

/// The file the plugin is described in, and which a launch is built from.
const MANIFEST_FILE: &str = "manifest.json";

/// Where the person's "trust this exact package" decisions are kept, in the plugins root
/// beside `disabled.json`.
pub const LOCAL_TRUST_FILE: &str = "trusted-plugins.json";

/// The developer switch OAIY already has for plugins (see [`developer_mode`]).
pub const DEV_MODE_ENV: &str = "OAIY_PLUGIN_DEV_MODE";

/// The envelope is a few kilobytes for a hundred files. A megabyte-scale one is not a
/// package manifest.
const MAX_ENVELOPE_BYTES: u64 = 4 * 1024 * 1024;
/// As many files as the installer will unpack (`install::MAX_ENTRIES`).
const MAX_FILES: usize = 20_000;
/// The most that is hashed for one package. The installer stops at 512 MiB; a folder
/// copied in by hand may be larger, but not without limit: this runs before every launch.
const MAX_PACKAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Extensions that run or load. Used only to say "executable" in a reason: every
/// unlisted file fails a signed package, whatever it is called.
const LOADABLE_EXTS: &[&str] = &[
    "exe", "dll", "com", "cmd", "bat", "ps1", "js", "mjs", "cjs", "py", "sh", "node", "scr", "vbs",
    "wsf", "msi", "jar", "lnk", "sys", "ocx", "cpl", "hta", "pif",
];

// ---------------------------------------------------------------------------------
// The verdict
// ---------------------------------------------------------------------------------

/// What is known about a plugin package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrustState {
    /// Signed by a pinned publisher for this plugin, every file as signed.
    Verified,
    /// Carries a signature that does not check out. Never started.
    Quarantined,
    /// No signature, a release build, and the person has not trusted this package.
    /// Never started.
    Unsigned,
    /// No signature, and this is a developer build (or the developer switch is on).
    UnsignedDev,
    /// No signature, and the person trusted this exact package (its digest).
    TrustedLocal,
}

impl TrustState {
    pub fn allows_launch(self) -> bool {
        matches!(self, TrustState::Verified | TrustState::UnsignedDev | TrustState::TrustedLocal)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TrustState::Verified => "verified",
            TrustState::Quarantined => "quarantined",
            TrustState::Unsigned => "unsigned",
            TrustState::UnsignedDev => "unsigned-dev",
            TrustState::TrustedLocal => "trusted-local",
        }
    }
}

/// The trust of one package, as `GET /api/plugins` reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageTrust {
    pub state: TrustState,
    /// Who signed it (a verified package only): the pinned publisher's name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub publisher: Option<String>,
    /// The pinned key that verified it (a verified package only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    /// The release the signer wrote into the signed payload (a verified package only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Why, for every state that is not `verified`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// When the person trusted this package (`trusted-local` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trusted_at: Option<String>,
}

impl PackageTrust {
    pub fn allows_launch(&self) -> bool {
        self.state.allows_launch()
    }

    fn bare(state: TrustState, reason: Option<String>) -> Self {
        Self { state, publisher: None, key_id: None, version: None, reason, trusted_at: None }
    }
}

/// What lets a process start. Only [`TrustService::authorize_launch`] makes one, and
/// `PluginProcess::spawn` takes one, so a new way to start a plugin cannot skip the
/// check by forgetting it.
///
/// It carries what was checked and `spawn` uses that and nothing else: the folder, and
/// the manifest parsed from the very bytes of `manifest.json` that were hashed. A launch
/// that read `manifest.json` again for itself, or was handed a manifest read at some
/// earlier scan, could start an entry the verified manifest never named.
#[derive(Debug)]
pub struct LaunchPermit {
    trust: PackageTrust,
    dir: PathBuf,
    manifest: PluginManifest,
}

impl LaunchPermit {
    /// What the package was found to be a moment ago (logged with each launch).
    pub fn trust(&self) -> &PackageTrust {
        &self.trust
    }

    /// The folder that was checked: the only one a process is started from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The manifest of what was checked, parsed from the bytes that were hashed.
    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    /// For tests of the process layer, which have nothing to do with trust.
    #[cfg(test)]
    pub(crate) fn unchecked_for_tests(dir: &Path, manifest: PluginManifest) -> Self {
        Self { trust: PackageTrust::bare(TrustState::UnsignedDev, None), dir: dir.to_path_buf(), manifest }
    }
}

/// Why a launch was not authorised.
#[derive(Debug)]
pub enum LaunchRefusal {
    /// The package may not run. The verdict is what the plugin's record should show.
    Untrusted(Box<PackageTrust>),
    /// The package may run, but its `manifest.json` does not load now (it changed since
    /// the scan, or was never valid).
    Manifest(ManifestError),
}

// ---------------------------------------------------------------------------------
// The policy: what counts as a developer
// ---------------------------------------------------------------------------------

/// Whether an unsigned package may run without the person's say-so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustPolicy {
    developer: bool,
}

impl TrustPolicy {
    pub fn developer() -> Self {
        Self { developer: true }
    }

    pub fn release() -> Self {
        Self { developer: false }
    }

    pub fn is_developer(self) -> bool {
        self.developer
    }

    /// The policy of this build and environment.
    ///
    /// Test builds are developer builds: the plugin fixtures in this crate's tests are
    /// unsigned, and a `cargo test --release` should not turn every one of them into a
    /// refusal. The rules themselves are tested with [`TrustPolicy::release`].
    pub fn for_build() -> Self {
        if cfg!(test) {
            return Self::developer();
        }
        Self { developer: developer_mode(std::env::var(DEV_MODE_ENV).ok().as_deref(), cfg!(debug_assertions)) }
    }
}

/// Is this a developer's OAIY, for the purpose of running an unsigned plugin?
///
/// A debug build always is: that is `tauri dev`, the owner's own flow, and it stays as it
/// was. In a release build only an explicit `OAIY_PLUGIN_DEV_MODE=1` (or `true`) is.
///
/// This is not `lib.rs`'s `plugin_development_mode`, which answers a different question,
/// whether a plugin should simulate its hardware. There `OAIY_PLUGIN_DEV_MODE=0` in a
/// debug build means "use the real dongle", which is how the owner answers a real phone
/// line from `tauri dev`; it must not turn the same build into one that refuses the
/// unsigned plugin that answers it. And there a value nobody recognises means
/// "simulate", the safe side for hardware; here it means "not a developer", the safe
/// side for trust.
pub fn developer_mode(env_value: Option<&str>, debug_build: bool) -> bool {
    debug_build || matches!(env_value.map(str::trim), Some("1" | "true"))
}

// ---------------------------------------------------------------------------------
// Pinned publishers
// ---------------------------------------------------------------------------------

/// One pinned key: a publisher's name, the key, and the plugins it may sign.
#[derive(Debug, Clone)]
pub struct Publisher {
    /// The envelope's `keyId`.
    pub key_id: String,
    pub name: String,
    key: VerifyingKey,
    pub plugins: BTreeSet<String>,
}

/// The publisher keys this OAIY trusts.
#[derive(Debug, Clone, Default)]
pub struct Publishers(Vec<Publisher>);

#[derive(Deserialize)]
struct PublishersFile {
    version: u32,
    publishers: Vec<PublisherEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PublisherEntry {
    id: String,
    name: String,
    public_key: String,
    plugins: Vec<String>,
}

static EMBEDDED: OnceLock<Publishers> = OnceLock::new();

impl Publishers {
    /// The keys compiled into this build (`resources/trusted-publishers.json`).
    ///
    /// A file that does not parse trusts nobody and says so in the log: a broken pin file
    /// must never widen what is accepted. A test keeps the shipped one parsing.
    pub fn embedded() -> &'static Publishers {
        EMBEDDED.get_or_init(|| {
            match Publishers::parse(include_str!("../../resources/trusted-publishers.json")) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("resources/trusted-publishers.json is unusable, so no publisher is trusted: {e}");
                    Publishers::default()
                }
            }
        })
    }

    /// Read a pin file. Strict: one bad entry rejects the file, because a half-read pin
    /// list is a different trust decision from the one written down.
    pub fn parse(text: &str) -> Result<Publishers, String> {
        let file: PublishersFile = serde_json::from_str(text).map_err(|e| format!("not a publishers file: {e}"))?;
        if file.version != 1 {
            return Err(format!("unsupported publishers file version {}", file.version));
        }
        let mut out: Vec<Publisher> = Vec::new();
        for entry in file.publishers {
            let id = entry.id.trim().to_string();
            if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
                return Err(format!("publisher key id {id:?} is not a plain identifier"));
            }
            if out.iter().any(|p| p.key_id == id) {
                return Err(format!("publisher key id {id:?} appears twice"));
            }
            let name = entry.name.trim().to_string();
            if name.is_empty() || name.len() > 100 {
                return Err(format!("publisher {id:?} needs a name"));
            }
            let raw = base64::engine::general_purpose::STANDARD
                .decode(entry.public_key.trim())
                .map_err(|e| format!("publisher {id:?}: the public key is not base64: {e}"))?;
            let bytes = <[u8; 32]>::try_from(raw.as_slice()).map_err(|_| format!("publisher {id:?}: the public key must be 32 bytes"))?;
            let key = VerifyingKey::from_bytes(&bytes).map_err(|e| format!("publisher {id:?}: not a valid Ed25519 key: {e}"))?;
            let plugins: BTreeSet<String> = entry.plugins.iter().map(|p| p.trim().to_string()).collect();
            if plugins.is_empty() || plugins.iter().any(|p| !crate::plugins::install::valid_plugin_id(p)) {
                return Err(format!("publisher {id:?} must list the plugin ids it may sign, each a plain plugin id"));
            }
            out.push(Publisher { key_id: id, name, key, plugins });
        }
        Ok(Publishers(out))
    }

    fn get(&self, key_id: &str) -> Option<&Publisher> {
        self.0.iter().find(|p| p.key_id == key_id)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ---------------------------------------------------------------------------------
// The envelope
// ---------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Envelope {
    format: u32,
    alg: String,
    key_id: String,
    payload_b64: String,
    signature: String,
}

/// The signed payload. Its `name` and `createdAt` are the signer's own bookkeeping and
/// mean nothing to this check (Aokie's CI signs `aokie-plugin`, the operator's own key
/// signed `aokie`): who may sign what is decided by the pinned key's plugin list.
#[derive(Deserialize)]
struct Payload {
    version: String,
    files: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct FileEntry {
    path: String,
    sha256: String,
    size: u64,
}

/// An envelope whose signature checked out under a pinned key for this plugin.
struct Opened<'a> {
    publisher: &'a Publisher,
    payload: Payload,
}

/// Check an envelope: format, a pinned key, the signature over the payload bytes, and
/// that this publisher may sign this plugin. Says nothing yet about the files.
fn open_envelope<'a>(text: &str, publishers: &'a Publishers, plugin_id: &str) -> Result<Opened<'a>, String> {
    // A byte order mark is what a text editor or PowerShell adds when the file is saved
    // again. The signature is over the payload inside, not the file, so it changes nothing.
    let envelope: Envelope =
        serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| format!("{PACKAGE_MANIFEST_FILE} is malformed: {e}"))?;
    // Neither of the two earlier verifiers looked at `format`. A format this code has not
    // seen may mean something else by its paths or its digests, and verifying it as if it
    // did not is how a new format gets waved through.
    if envelope.format != 1 {
        return Err(format!("{PACKAGE_MANIFEST_FILE} is format {}, which this OAIY cannot read", envelope.format));
    }
    if envelope.alg != "Ed25519" {
        return Err(format!("unsupported signature algorithm {:?}", envelope.alg));
    }
    let Some(publisher) = publishers.get(&envelope.key_id) else {
        return Err(format!("it is signed with the key {:?}, which this OAIY does not trust", envelope.key_id));
    };

    let payload_bytes = base64::engine::general_purpose::STANDARD
        .decode(envelope.payload_b64.trim())
        .map_err(|e| format!("the signed payload is not base64: {e}"))?;
    let signature_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(envelope.signature.trim().trim_end_matches('='))
        .map_err(|e| format!("the signature is not base64: {e}"))?;
    let signature = Signature::from_bytes(
        &<[u8; 64]>::try_from(signature_bytes.as_slice()).map_err(|_| "the signature must be 64 bytes".to_string())?,
    );
    // Strict: also refuses the malleable encodings a plain verify accepts. Whatever the
    // signer wrote passes, because it writes only canonical signatures.
    publisher
        .key
        .verify_strict(&payload_bytes, &signature)
        .map_err(|_| "the signature does not match the package manifest".to_string())?;

    if !publisher.plugins.contains(plugin_id) {
        return Err(format!("{} (key {}) is not allowed to sign the {plugin_id:?} plugin", publisher.name, publisher.key_id));
    }
    let payload: Payload = serde_json::from_slice(&payload_bytes).map_err(|e| format!("the signed payload is malformed: {e}"))?;
    Ok(Opened { publisher, payload })
}

/// A path the signer could have written and that stays inside the bundle.
///
/// Stricter than the signer's own `..` test: it also refuses what Windows would resolve
/// somewhere else (a drive or a stream after a colon, a backslash, a trailing dot or
/// space), and the environment-variable-shaped names Aokie's signer refuses to sign
/// because something once parked files under a literal `%SystemDrive%`.
fn safe_listed_path(rel: &str) -> Result<(), String> {
    let bad = |why: &str| Err(format!("the signature lists an unusable path {rel:?} ({why})"));
    if rel.is_empty() || rel.len() > 1024 {
        return bad("empty or too long");
    }
    if rel.starts_with('/') || rel.contains('\\') || rel.contains(':') || rel.contains("..") || rel.chars().any(|c| c.is_control()) {
        return bad("it does not stay inside the package");
    }
    for part in rel.split('/') {
        if part.is_empty() || part == "." || part.starts_with('%') || part.ends_with('.') || part.ends_with(' ') {
            return bad("a component the package cannot have");
        }
    }
    if rel == PACKAGE_MANIFEST_FILE {
        return bad("the envelope cannot list itself");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------
// Reading the folder
// ---------------------------------------------------------------------------------

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

enum HashFailure {
    Missing,
    /// More than the budget allows: the package is not verified, however it is.
    TooLarge,
    Other(String),
}

/// What `path` is, if it is a regular file. A symbolic link or junction is refused: the
/// bytes it leads to are not in the package.
fn regular_file_metadata(path: &Path) -> Result<std::fs::Metadata, HashFailure> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => HashFailure::Missing,
        _ => HashFailure::Other(e.to_string()),
    })?;
    if meta.file_type().is_symlink() {
        return Err(HashFailure::Other("it is a symbolic link".into()));
    }
    if !meta.is_file() {
        return Err(HashFailure::Other("it is not a regular file".into()));
    }
    Ok(meta)
}

/// SHA-256 of some bytes, as lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Like [`hash_regular`], but reads the file whole into memory (at most `cap` bytes) and
/// gives the bytes back with their digest and size. What was hashed is then exactly what
/// the caller goes on to use: there is no second read for a swap to slip into.
fn hash_regular_keeping(path: &Path, budget: &mut u64, cap: u64) -> Result<(String, u64, Vec<u8>), HashFailure> {
    let too_large = || HashFailure::Other(format!("it is larger than the {cap} bytes allowed here"));
    let meta = regular_file_metadata(path)?;
    if meta.len() > cap {
        return Err(too_large());
    }
    *budget = budget.checked_sub(meta.len()).ok_or(HashFailure::TooLarge)?;
    let file = std::fs::File::open(path).map_err(|e| HashFailure::Other(e.to_string()))?;
    let mut bytes = Vec::new();
    file.take(cap + 1).read_to_end(&mut bytes).map_err(|e| HashFailure::Other(e.to_string()))?;
    if bytes.len() as u64 > cap {
        return Err(too_large());
    }
    Ok((sha256_hex(&bytes), bytes.len() as u64, bytes))
}

/// SHA-256 and size of a regular file, spending `budget` bytes.
///
/// Refuses a symbolic link or junction: the bytes it leads to are not in the package.
fn hash_regular(path: &Path, budget: &mut u64) -> Result<(String, u64), HashFailure> {
    let meta = regular_file_metadata(path)?;
    *budget = budget.checked_sub(meta.len()).ok_or(HashFailure::TooLarge)?;
    let mut file = std::fs::File::open(path).map_err(|e| HashFailure::Other(e.to_string()))?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| HashFailure::Other(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        size += n as u64;
    }
    Ok((hex(&hasher.finalize()), size))
}

/// Every regular file under `dir`, as sorted forward-slash relative paths. A symbolic
/// link or junction anywhere is an error: a package is files, not pointers.
fn list_files(dir: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), String::new())];
    while let Some((path, prefix)) = stack.pop() {
        let entries = std::fs::read_dir(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("cannot read an entry of {}: {e}", path.display()))?;
            let file_type = entry.file_type().map_err(|e| format!("cannot stat {}: {e}", entry.path().display()))?;
            let rel = format!("{prefix}{}", entry.file_name().to_string_lossy());
            if file_type.is_symlink() {
                return Err(format!("a symbolic link is present: {rel}"));
            }
            if file_type.is_dir() {
                stack.push((entry.path(), format!("{rel}/")));
                continue;
            }
            out.push(rel);
            if out.len() > MAX_FILES {
                return Err("the package has too many files".into());
            }
        }
    }
    out.sort();
    Ok(out)
}

/// A hash of every file's path, kind, size and modification time, from `stat` alone.
///
/// Decides whether the answer of the last scan still stands. It is not evidence of
/// anything: a file rewritten to the same size with its time put back has the same
/// fingerprint, which is why a launch never asks it.
fn fingerprint(dir: &Path) -> Result<String, String> {
    let mut rows: Vec<String> = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), String::new())];
    while let Some((path, prefix)) = stack.pop() {
        let entries = std::fs::read_dir(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            // Not `entry.metadata()`: on Windows that is the directory's cached copy, which
            // lags a file that was just written until its handle is closed.
            let meta = std::fs::symlink_metadata(entry.path()).map_err(|e| e.to_string())?;
            let file_type = meta.file_type();
            let rel = format!("{prefix}{}", entry.file_name().to_string_lossy());
            let mtime = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let kind = if file_type.is_symlink() { 'l' } else if file_type.is_dir() { 'd' } else { 'f' };
            rows.push(format!("{rel}\0{kind}\0{}\0{mtime}", meta.len()));
            if file_type.is_dir() && !file_type.is_symlink() {
                stack.push((entry.path(), format!("{rel}/")));
            }
            if rows.len() > MAX_FILES * 2 {
                return Err("the package has too many files".into());
            }
        }
    }
    rows.sort();
    let mut hasher = Sha256::new();
    for row in &rows {
        hasher.update(row.as_bytes());
        hasher.update(b"\n");
    }
    Ok(hex(&hasher.finalize()))
}

/// The digest a "trust this package" decision is bound to: a hash over every file's
/// path, size and SHA-256, so a change to any file, a new file or a missing one is a
/// different package.
pub fn package_digest(dir: &Path) -> Result<String, String> {
    digest_walk(dir).map(|(digest, _)| digest)
}

/// [`package_digest`], and the bytes of the folder's `manifest.json` exactly as they were
/// hashed into it (`None` when there is none).
fn digest_walk(dir: &Path) -> Result<(String, Option<Vec<u8>>), String> {
    let files = list_files(dir)?;
    let mut budget = MAX_PACKAGE_BYTES;
    let mut hasher = Sha256::new();
    let mut manifest_json = None;
    hasher.update(b"oaiy-package-digest-v1\n");
    for rel in &files {
        let path = dir.join(rel);
        let hashed = if rel == MANIFEST_FILE {
            hash_regular_keeping(&path, &mut budget, MAX_MANIFEST_BYTES).map(|(sha, size, bytes)| (sha, size, Some(bytes)))
        } else {
            hash_regular(&path, &mut budget).map(|(sha, size)| (sha, size, None))
        };
        let (sha, size, bytes) = match hashed {
            Ok(v) => v,
            Err(HashFailure::Missing) => return Err(format!("{rel} vanished while it was read")),
            Err(HashFailure::TooLarge) => return Err("the package is too large to hash".into()),
            Err(HashFailure::Other(e)) => return Err(format!("cannot read {rel}: {e}")),
        };
        if bytes.is_some() {
            manifest_json = bytes;
        }
        hasher.update(rel.as_bytes());
        hasher.update([0]);
        hasher.update(size.to_string().as_bytes());
        hasher.update([0]);
        hasher.update(sha.as_bytes());
        hasher.update(b"\n");
    }
    Ok((format!("sha256:{}", hex(&hasher.finalize())), manifest_json))
}

fn is_loadable(rel: &str) -> bool {
    Path::new(rel)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| LOADABLE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Up to five names, and how many more.
fn list_some(names: &[String]) -> String {
    let shown: Vec<&str> = names.iter().take(5).map(String::as_str).collect();
    match names.len().saturating_sub(shown.len()) {
        0 => shown.join(", "),
        more => format!("{} and {more} more", shown.join(", ")),
    }
}

/// Every listed file must be there as signed, and no other file may be. Gives back the
/// bytes of `manifest.json` exactly as they were hashed, so what is started from the
/// package is the file that was checked and not a second read of it.
fn check_files(dir: &Path, payload: &Payload) -> Result<Vec<u8>, String> {
    if payload.files.is_empty() {
        return Err("the signature lists no files".into());
    }
    if payload.files.len() > MAX_FILES {
        return Err("the signature lists too many files".into());
    }
    let mut listed: HashSet<&str> = HashSet::new();
    for file in &payload.files {
        safe_listed_path(&file.path)?;
        if !listed.insert(file.path.as_str()) {
            return Err(format!("the signature lists {:?} twice", file.path));
        }
    }
    // Every file already has to match its digest, so this is about the entry command: a
    // manifest.json the signature does not cover could point at anything.
    if !listed.contains(MANIFEST_FILE) {
        return Err("the signature does not cover manifest.json, so the command it starts is not pinned".into());
    }

    let mut budget = MAX_PACKAGE_BYTES;
    let mut changed: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut manifest_json: Option<Vec<u8>> = None;
    for file in &payload.files {
        let path = dir.join(&file.path);
        let hashed = if file.path == MANIFEST_FILE {
            hash_regular_keeping(&path, &mut budget, MAX_MANIFEST_BYTES).map(|(sha, size, bytes)| (sha, size, Some(bytes)))
        } else {
            hash_regular(&path, &mut budget).map(|(sha, size)| (sha, size, None))
        };
        match hashed {
            Ok((sha, size, bytes)) => {
                if sha != file.sha256.to_ascii_lowercase() || size != file.size {
                    changed.push(file.path.clone());
                } else if bytes.is_some() {
                    manifest_json = bytes;
                }
            }
            Err(HashFailure::Missing) => missing.push(file.path.clone()),
            Err(HashFailure::TooLarge) => return Err("the package is too large to verify".into()),
            Err(HashFailure::Other(e)) => changed.push(format!("{} ({e})", file.path)),
        }
    }

    let mut executables: Vec<String> = Vec::new();
    let mut others: Vec<String> = Vec::new();
    for rel in list_files(dir)? {
        if rel == PACKAGE_MANIFEST_FILE || listed.contains(rel.as_str()) {
            continue;
        }
        if is_loadable(&rel) {
            executables.push(rel);
        } else {
            others.push(rel);
        }
    }

    let mut problems: Vec<String> = Vec::new();
    if !changed.is_empty() {
        problems.push(format!("digest mismatch: {}", list_some(&changed)));
    }
    if !missing.is_empty() {
        problems.push(format!("listed file missing: {}", list_some(&missing)));
    }
    if !executables.is_empty() {
        problems.push(format!("unlisted executable present: {}", list_some(&executables)));
    }
    if !others.is_empty() {
        problems.push(format!("unlisted file present: {}", list_some(&others)));
    }
    if !problems.is_empty() {
        return Err(problems.join("; "));
    }
    // Listed, and neither changed nor missing, so it was read above.
    manifest_json.ok_or_else(|| "manifest.json could not be read".to_string())
}

// ---------------------------------------------------------------------------------
// The person's own trust
// ---------------------------------------------------------------------------------

/// A package the person chose to trust, and the digest of exactly what they trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalTrust {
    pub digest: String,
    pub trusted_at: String,
}

#[derive(Serialize, Deserialize)]
struct LocalTrustFile {
    version: u32,
    #[serde(default)]
    plugins: BTreeMap<String, LocalTrust>,
}

#[derive(Default)]
struct StoreState {
    loaded: bool,
    stamp: Option<(u64, Option<SystemTime>)>,
    map: BTreeMap<String, LocalTrust>,
}

/// `<plugins root>/trusted-plugins.json`. Read again whenever the file changes, so the
/// desktop and a headless server sharing a data directory see each other's decisions.
struct LocalStore {
    path: PathBuf,
    state: Mutex<StoreState>,
}

impl LocalStore {
    fn new(path: PathBuf) -> Self {
        Self { path, state: Mutex::new(StoreState::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn refresh(&self, state: &mut StoreState) {
        let stamp = std::fs::metadata(&self.path).ok().map(|m| (m.len(), m.modified().ok()));
        if state.loaded && stamp == state.stamp {
            return;
        }
        state.loaded = true;
        state.stamp = stamp;
        // A store that cannot be read trusts nothing: the person is asked again, which is
        // the safe way to be wrong.
        state.map = match std::fs::read_to_string(&self.path) {
            Ok(text) => match serde_json::from_str::<LocalTrustFile>(&text) {
                Ok(file) if file.version == 1 => file.plugins,
                Ok(file) => {
                    log::warn!("{} is version {}, which this OAIY cannot read; no package is trusted", self.path.display(), file.version);
                    BTreeMap::new()
                }
                Err(e) => {
                    log::warn!("{} is unreadable, so no package is trusted: {e}", self.path.display());
                    BTreeMap::new()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => {
                log::warn!("cannot read {}: {e}", self.path.display());
                BTreeMap::new()
            }
        };
    }

    fn get(&self, id: &str) -> Option<LocalTrust> {
        let mut state = self.lock();
        self.refresh(&mut state);
        state.map.get(id).cloned()
    }

    /// Write the map: temp and rename, so a crash leaves the old decisions rather than a
    /// truncated file that reads as "nobody is trusted".
    fn persist(&self, state: &mut StoreState) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let file = LocalTrustFile { version: 1, plugins: state.map.clone() };
        let body = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, body).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("cannot replace {}: {e}", self.path.display()))?;
        state.stamp = std::fs::metadata(&self.path).ok().map(|m| (m.len(), m.modified().ok()));
        Ok(())
    }

    fn put(&self, id: &str, trust: LocalTrust) -> Result<(), String> {
        let mut state = self.lock();
        self.refresh(&mut state);
        state.map.insert(id.to_string(), trust);
        self.persist(&mut state)
    }

    fn remove(&self, id: &str) {
        let mut state = self.lock();
        self.refresh(&mut state);
        if state.map.remove(id).is_some() {
            if let Err(e) = self.persist(&mut state) {
                log::warn!("could not forget the trust given to {id}: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// The service
// ---------------------------------------------------------------------------------

/// How much of the last answer may be reused.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reuse {
    /// A scan: the answer of the last one, while the folder looks the same.
    Cached,
    /// A launch: hash everything again, and remember what was found.
    Fresh,
    /// A staged copy at install: nothing is read from or written to the cache.
    Never,
}

struct Remembered {
    fingerprint: String,
    /// The local trust record the answer was made under (its digest).
    local: Option<String>,
    verdict: PackageTrust,
    /// See [`Assessed::manifest_sha256`].
    manifest_sha256: Option<String>,
}

/// A verdict, and what the check behind it saw of `manifest.json`.
struct Assessed {
    trust: PackageTrust,
    /// SHA-256 of the `manifest.json` the check hashed, for a package that was verified
    /// (a signature that held, or the person's trust in a digest): the verdict is about
    /// a folder whose manifest is exactly that file. `None` when nothing was verified.
    manifest_sha256: Option<String>,
    /// The bytes that were hashed. Kept only by a check that ran just now, never by the
    /// cache, so that a launch can start what was verified rather than read it again.
    manifest_json: Option<Vec<u8>>,
}

impl Assessed {
    /// A verdict that rests on no verification of the folder's files.
    fn unverified(trust: PackageTrust) -> Self {
        Self { trust, manifest_sha256: None, manifest_json: None }
    }

    /// Is `manifest_json` (bytes a caller parsed) the manifest this verdict was made for?
    /// A verdict that verified nothing has no manifest to disagree with.
    fn covers(&self, manifest_json: &[u8]) -> bool {
        self.manifest_sha256.as_deref().map_or(true, |sha| sha == sha256_hex(manifest_json))
    }
}

/// Verifies plugin packages under one policy and one set of pinned keys.
pub struct TrustService {
    policy: TrustPolicy,
    publishers: Publishers,
    local: LocalStore,
    remembered: Mutex<HashMap<String, Remembered>>,
}

impl TrustService {
    pub fn new(policy: TrustPolicy, publishers: Publishers, local_store: PathBuf) -> Arc<Self> {
        Arc::new(Self { policy, publishers, local: LocalStore::new(local_store), remembered: Mutex::new(HashMap::new()) })
    }

    /// The service of this build for the plugins in `root`: the pinned keys compiled in,
    /// the policy of the build and environment, the person's decisions in `root`.
    pub fn for_root(root: &Path) -> Arc<Self> {
        Self::new(TrustPolicy::for_build(), Publishers::embedded().clone(), root.join(LOCAL_TRUST_FILE))
    }

    fn remembered(&self) -> std::sync::MutexGuard<'_, HashMap<String, Remembered>> {
        self.remembered.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What a listing shows: as recent as the folder's fingerprint allows. Cheap for a
    /// package that has not changed, and for an unsigned one it is a single `stat`.
    ///
    /// The fingerprint cannot see a file rewritten to the same size with its time put
    /// back, so this is only a display. A scan that builds a plugin from `manifest.json`
    /// asks [`TrustService::assess_read`] instead.
    pub fn assess(&self, dir: &Path, id: &str) -> PackageTrust {
        self.assess_with(dir, id, Reuse::Cached).trust
    }

    /// What a scan shows for a package whose `manifest.json` it has just read (the bytes
    /// it parsed the plugin from). The answer of the last check stands while the folder
    /// looks the same and that manifest is the one the check verified. If it is not (a
    /// swap the fingerprint could not see, or a change between the scan's read and the
    /// check) the package is looked at again from its bytes, and one whose manifest
    /// still differs from what was verified is not trusted: the scan must not build a
    /// plugin, its commands or its screens from a file the verdict is not about.
    pub fn assess_read(&self, dir: &Path, id: &str, manifest_json: &[u8]) -> PackageTrust {
        let remembered = self.assess_with(dir, id, Reuse::Cached);
        if remembered.covers(manifest_json) {
            return remembered.trust;
        }
        let fresh = self.assess_with(dir, id, Reuse::Fresh);
        if fresh.covers(manifest_json) {
            return fresh.trust;
        }
        match fresh.trust.state {
            TrustState::TrustedLocal => self.unsigned(Some("It changed while it was being checked.")),
            _ => self.quarantined("manifest.json changed while it was being read"),
        }
    }

    /// The whole check again, from the bytes, whatever was remembered. What a launch
    /// stands on; also forgets a stale answer, so the listing agrees with it.
    pub fn assess_fresh(&self, dir: &Path, id: &str) -> PackageTrust {
        self.assess_with(dir, id, Reuse::Fresh).trust
    }

    /// The check on a copy that is not (yet) where the plugin lives, at install. Uses and
    /// leaves no memory, since the folder it will end up in is another path.
    pub fn assess_staged(&self, dir: &Path, id: &str) -> PackageTrust {
        self.assess_with(dir, id, Reuse::Never).trust
    }

    /// Verify immediately before a launch and, if the package may run, say so with a
    /// permit. The failure is what the record should show.
    ///
    /// The permit's manifest is parsed from the bytes of `manifest.json` that this check
    /// hashed, so the entry that is started is the one the signature (or the person's
    /// trust) covers. Only a package that nothing was verified for, a developer's, is
    /// read here for the first time.
    pub fn authorize_launch(&self, dir: &Path, id: &str) -> Result<LaunchPermit, LaunchRefusal> {
        let assessed = self.assess_with(dir, id, Reuse::Fresh);
        if !assessed.trust.allows_launch() {
            return Err(LaunchRefusal::Untrusted(Box::new(assessed.trust)));
        }
        let manifest = match &assessed.manifest_json {
            Some(bytes) => PluginManifest::parse(bytes, dir),
            None => PluginManifest::read(dir).map(|(manifest, _)| manifest),
        }
        .map_err(LaunchRefusal::Manifest)?;
        Ok(LaunchPermit { trust: assessed.trust, dir: dir.to_path_buf(), manifest })
    }

    fn quarantined(&self, detail: impl AsRef<str>) -> PackageTrust {
        let mut reason = format!("Quarantined: {}.", detail.as_ref().trim_end_matches('.'));
        if self.policy.developer {
            // Said only to a developer: to anyone else it reads as a way round the check.
            reason.push_str(
                " To run a build you changed yourself, delete package-manifest.json from the plugin's folder so it counts as unsigned.",
            );
        }
        PackageTrust::bare(TrustState::Quarantined, Some(reason))
    }

    fn unsigned(&self, note: Option<&str>) -> PackageTrust {
        let lead = note.map(|n| format!("{n} ")).unwrap_or_default();
        if self.policy.developer {
            PackageTrust::bare(
                TrustState::UnsignedDev,
                Some(format!("{lead}Not signed. It runs because this is a developer build; a release build would ask you to trust it first.")),
            )
        } else {
            PackageTrust::bare(
                TrustState::Unsigned,
                Some(format!(
                    "{lead}Not signed by a publisher this OAIY trusts. If you built it yourself or know where it came from, you can trust this exact package."
                )),
            )
        }
    }

    fn assess_with(&self, dir: &Path, id: &str, reuse: Reuse) -> Assessed {
        let signed = match std::fs::symlink_metadata(dir.join(PACKAGE_MANIFEST_FILE)) {
            Ok(meta) if meta.is_file() => true,
            Ok(_) => return Assessed::unverified(self.quarantined(format!("{PACKAGE_MANIFEST_FILE} is not a regular file"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            // Present or not, it could not be looked at: not a reason to call it unsigned.
            Err(e) => return Assessed::unverified(self.quarantined(format!("{PACKAGE_MANIFEST_FILE} cannot be read: {e}"))),
        };
        let local = if signed { None } else { self.local.get(id) };
        if !signed && local.is_none() {
            return Assessed::unverified(self.unsigned(None));
        }

        let print = match fingerprint(dir) {
            Ok(p) => p,
            Err(e) if signed => return Assessed::unverified(self.quarantined(format!("the package folder cannot be read: {e}"))),
            Err(e) => return Assessed::unverified(self.unsigned(Some(format!("The package folder cannot be read ({e}).").as_str()))),
        };
        let local_digest = local.as_ref().map(|l| l.digest.clone());
        if reuse == Reuse::Cached {
            if let Some(hit) = self.remembered().get(id) {
                if hit.fingerprint == print && hit.local == local_digest {
                    return Assessed { trust: hit.verdict.clone(), manifest_sha256: hit.manifest_sha256.clone(), manifest_json: None };
                }
            }
        }

        let assessed = match &local {
            None => self.verify_signed(dir, id),
            Some(record) => self.check_local(dir, record),
        };
        if reuse != Reuse::Never {
            self.remembered().insert(
                id.to_string(),
                Remembered {
                    fingerprint: print,
                    local: local_digest,
                    verdict: assessed.trust.clone(),
                    manifest_sha256: assessed.manifest_sha256.clone(),
                },
            );
        }
        assessed
    }

    fn verify_signed(&self, dir: &Path, id: &str) -> Assessed {
        let path = dir.join(PACKAGE_MANIFEST_FILE);
        let text = match std::fs::metadata(&path) {
            Ok(m) if m.len() > MAX_ENVELOPE_BYTES => {
                return Assessed::unverified(self.quarantined(format!("{PACKAGE_MANIFEST_FILE} is too large to be a package manifest")))
            }
            Ok(_) => match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => return Assessed::unverified(self.quarantined(format!("{PACKAGE_MANIFEST_FILE} cannot be read: {e}"))),
            },
            Err(e) => return Assessed::unverified(self.quarantined(format!("{PACKAGE_MANIFEST_FILE} cannot be read: {e}"))),
        };
        let opened = match open_envelope(&text, &self.publishers, id) {
            Ok(o) => o,
            Err(e) => return Assessed::unverified(self.quarantined(e)),
        };
        match check_files(dir, &opened.payload) {
            Ok(manifest_json) => Assessed {
                trust: PackageTrust {
                    state: TrustState::Verified,
                    publisher: Some(opened.publisher.name.clone()),
                    key_id: Some(opened.publisher.key_id.clone()),
                    version: Some(opened.payload.version.clone()),
                    reason: None,
                    trusted_at: None,
                },
                manifest_sha256: Some(sha256_hex(&manifest_json)),
                manifest_json: Some(manifest_json),
            },
            Err(e) => Assessed::unverified(self.quarantined(e)),
        }
    }

    fn check_local(&self, dir: &Path, record: &LocalTrust) -> Assessed {
        match digest_walk(dir) {
            Ok((digest, manifest_json)) if digest == record.digest => Assessed {
                trust: PackageTrust {
                    state: TrustState::TrustedLocal,
                    publisher: None,
                    key_id: None,
                    version: None,
                    reason: Some(format!(
                        "You trusted this exact package on {}. A change to any of its files ends that.",
                        record.trusted_at.get(..10).unwrap_or(&record.trusted_at)
                    )),
                    trusted_at: Some(record.trusted_at.clone()),
                },
                manifest_sha256: manifest_json.as_deref().map(sha256_hex),
                manifest_json,
            },
            Ok(_) => Assessed::unverified(self.unsigned(Some("It changed since you trusted it."))),
            Err(e) => Assessed::unverified(self.unsigned(Some(format!("It cannot be compared with what you trusted ({e}).").as_str()))),
        }
    }

    /// Record that the person trusts this exact, unsigned package.
    ///
    /// Takes the plugin's id and the folder the registry already holds for it, never a
    /// path or a URL from a caller. A package that carries a signature is refused: it is
    /// verified or quarantined by that signature, and no click turns a failed one into a
    /// good one.
    pub fn trust_local(&self, dir: &Path, id: &str) -> Result<PackageTrust, String> {
        if !crate::plugins::install::valid_plugin_id(id) {
            return Err(format!("{id:?} is not a plugin id"));
        }
        match std::fs::symlink_metadata(dir.join(PACKAGE_MANIFEST_FILE)) {
            Ok(_) => {
                return Err(format!(
                    "{id} carries a signature ({PACKAGE_MANIFEST_FILE}). A signed package is judged by its signature alone, so it cannot be trusted by hand."
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("cannot look at {PACKAGE_MANIFEST_FILE}: {e}")),
        }
        let digest = package_digest(dir).map_err(|e| format!("{id} cannot be trusted: {e}"))?;
        let trusted_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        self.local.put(id, LocalTrust { digest, trusted_at })?;
        self.remembered().remove(id);
        Ok(self.assess_fresh(dir, id))
    }

    /// Drop the person's trust in a plugin (it was uninstalled).
    pub fn forget_local(&self, id: &str) {
        self.local.remove(id);
        self.remembered().remove(id);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use std::sync::atomic::{AtomicU32, Ordering};

    // -----------------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------------

    static N: AtomicU32 = AtomicU32::new(0);

    /// A scratch folder, removed when the test is done.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!("oaiy-trust-{tag}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        /// A package folder inside, ready to fill.
        fn package(&self, name: &str) -> PathBuf {
            let dir = self.0.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A signing key made for the test, standing in for a publisher's.
    pub(crate) struct TestKey {
        pub(crate) key_id: String,
        signing: SigningKey,
    }

    impl TestKey {
        pub(crate) fn generate(key_id: &str) -> Self {
            let mut seed = [0u8; 32];
            getrandom::getrandom(&mut seed).unwrap();
            Self { key_id: key_id.to_string(), signing: SigningKey::from_bytes(&seed) }
        }

        pub(crate) fn public_b64(&self) -> String {
            base64::engine::general_purpose::STANDARD.encode(self.signing.verifying_key().to_bytes())
        }

        /// The pin list a host would carry for this key, allowed to sign `plugins`.
        pub(crate) fn pinned_for(&self, name: &str, plugins: &[&str]) -> Publishers {
            let file = serde_json::json!({
                "version": 1,
                "publishers": [{ "id": self.key_id, "name": name, "publicKey": self.public_b64(), "plugins": plugins }],
            });
            Publishers::parse(&file.to_string()).unwrap()
        }

        /// Sign a folder as `package-signer sign` does: an entry for every file but the
        /// envelope, sorted, then the detached signature over the payload bytes.
        pub(crate) fn sign(&self, dir: &Path, name: &str, version: &str) {
            self.write_envelope(dir, &self.payload_of(dir, name, version));
        }

        pub(crate) fn payload_of(&self, dir: &Path, name: &str, version: &str) -> Vec<u8> {
            let mut budget = u64::MAX;
            let files: Vec<serde_json::Value> = list_files(dir)
                .unwrap()
                .into_iter()
                .filter(|rel| rel != PACKAGE_MANIFEST_FILE)
                .map(|rel| {
                    let (sha, size) = hash_regular(&dir.join(&rel), &mut budget).ok().unwrap();
                    serde_json::json!({ "path": rel, "sha256": sha, "size": size })
                })
                .collect();
            serde_json::to_vec(&serde_json::json!({ "name": name, "version": version, "createdAt": "2026-09-29T00:00:00Z", "files": files })).unwrap()
        }

        pub(crate) fn write_envelope(&self, dir: &Path, payload: &[u8]) {
            let signature = self.signing.sign(payload);
            let envelope = serde_json::json!({
                "format": 1,
                "alg": "Ed25519",
                "keyId": self.key_id,
                "payloadB64": base64::engine::general_purpose::STANDARD.encode(payload),
                "signature": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(signature.to_bytes()),
            });
            std::fs::write(dir.join(PACKAGE_MANIFEST_FILE), serde_json::to_string_pretty(&envelope).unwrap()).unwrap();
        }
    }

    pub(crate) const MANIFEST: &str = r#"{"schemaVersion":1,"id":"demo","name":"Demo","version":"1.0.0","pluginApiVersion":1,"entry":{"kind":"process","command":"demo-plugin.exe"}}"#;

    /// The files of a small plugin.
    pub(crate) fn fill(dir: &Path) {
        std::fs::write(dir.join("manifest.json"), MANIFEST).unwrap();
        std::fs::write(dir.join("demo-plugin.exe"), b"demo plugin executable bytes").unwrap();
        std::fs::create_dir_all(dir.join("ui")).unwrap();
        std::fs::write(dir.join("ui").join("index.html"), b"<p>hello</p>").unwrap();
    }

    fn service(policy: TrustPolicy, publishers: Publishers, scratch: &Scratch) -> Arc<TrustService> {
        TrustService::new(policy, publishers, scratch.path().join(LOCAL_TRUST_FILE))
    }

    /// A signed `demo` package and a service that pins its key for `demo`.
    fn signed(policy: TrustPolicy) -> (Scratch, PathBuf, TestKey, Arc<TrustService>) {
        let scratch = Scratch::new("signed");
        let dir = scratch.package("demo");
        fill(&dir);
        let key = TestKey::generate("test-key-1");
        key.sign(&dir, "demo-plugin", "1.0.0");
        let svc = service(policy, key.pinned_for("Demo Co", &["demo"]), &scratch);
        (scratch, dir, key, svc)
    }

    fn quarantine_reason(t: &PackageTrust) -> &str {
        assert_eq!(t.state, TrustState::Quarantined, "{t:?}");
        t.reason.as_deref().unwrap()
    }

    // -----------------------------------------------------------------------------
    // What the real signer writes
    // -----------------------------------------------------------------------------

    /// A bundle built and signed by Aokie's actual `package-signer` (built from
    /// `crates/package-signer` in the Aokie repository, run as
    /// `package-signer sign --dir bundle --name demo-plugin --version 1.0.0 --key-id test-key-1`,
    /// with a key made by its `keygen` for this test). The three files are `MANIFEST` and
    /// the two below.
    const REAL_SIGNER_ENVELOPE: &str = r#"{
  "format": 1,
  "alg": "Ed25519",
  "keyId": "test-key-1",
  "payloadB64": "eyJuYW1lIjoiZGVtby1wbHVnaW4iLCJ2ZXJzaW9uIjoiMS4wLjAiLCJjcmVhdGVkQXQiOiIyMDI2LTA5LTI5VDA5OjU2OjAzWiIsImZpbGVzIjpbeyJwYXRoIjoiZGVtby1wbHVnaW4uZXhlIiwic2hhMjU2IjoiNzU5YmU3NjlkZDE0MGZkYWFjNTVkNDJkMTkzMGE0MjYyMjZiMTEwZGI1ZmNiMDJlNTQ2MDY2ZWIwM2YxMmM2OSIsInNpemUiOjI4fSx7InBhdGgiOiJtYW5pZmVzdC5qc29uIiwic2hhMjU2IjoiMmE0NTIxYzQ0ODlmYTE5MTcxMTBjZmIwOGUyYjhmMmY3M2QyMGMwZTllYzJkMmYyNjA1OWFjMGU3OWY4OWEzOSIsInNpemUiOjEzOX0seyJwYXRoIjoidWkvaW5kZXguaHRtbCIsInNoYTI1NiI6ImE1NjUyYmUxY2E4NjRkMzZkMjVjZmI1NGE0MWYzODRlMmRlMWIzYWNmNzUxM2E5MjVkNzJlZDcyNThmZGMwYWUiLCJzaXplIjoxMn1dfQ==",
  "signature": "0_ZVcHEn9cDV31A5duD9ySYeLPjpJLii522GSR5URn2r_Sv8-fW3Dzsl4XH03IL85dUI74w817nSO1CWuExhDw"
}"#;
    /// The public half of the key that signed it (a test key, made only for this vector).
    const REAL_SIGNER_TEST_PUBLIC_KEY: &str = "FBY5jCcCLZDSG20KX7f1UPo8sZ48ECERcCWq/iip1m8=";

    fn real_signer_pins(plugins: &[&str]) -> Publishers {
        let file = serde_json::json!({
            "version": 1,
            "publishers": [{ "id": "test-key-1", "name": "Real Signer Test", "publicKey": REAL_SIGNER_TEST_PUBLIC_KEY, "plugins": plugins }],
        });
        Publishers::parse(&file.to_string()).unwrap()
    }

    #[test]
    fn a_bundle_signed_by_the_real_package_signer_verifies() {
        let scratch = Scratch::new("real");
        let dir = scratch.package("demo");
        fill(&dir);
        std::fs::write(dir.join(PACKAGE_MANIFEST_FILE), REAL_SIGNER_ENVELOPE).unwrap();
        let svc = service(TrustPolicy::release(), real_signer_pins(&["demo"]), &scratch);

        let t = svc.assess_fresh(&dir, "demo");
        assert_eq!(t.state, TrustState::Verified, "{t:?}");
        assert_eq!(t.publisher.as_deref(), Some("Real Signer Test"));
        assert_eq!(t.key_id.as_deref(), Some("test-key-1"));
        assert_eq!(t.version.as_deref(), Some("1.0.0"));
        assert!(t.reason.is_none());

        // The same bytes, one changed: the real signer's own `verify` says "digest
        // mismatch", and so does this.
        std::fs::write(dir.join("demo-plugin.exe"), b"demo plugin executable bytez").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("digest mismatch: demo-plugin.exe"), "{t:?}");
    }

    #[test]
    fn the_real_signers_envelope_is_refused_under_another_key_or_for_another_plugin() {
        let scratch = Scratch::new("real-wrong");
        let dir = scratch.package("demo");
        fill(&dir);
        std::fs::write(dir.join(PACKAGE_MANIFEST_FILE), REAL_SIGNER_ENVELOPE).unwrap();

        // Pinned, but for a different plugin.
        let svc = service(TrustPolicy::release(), real_signer_pins(&["other"]), &scratch);
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("not allowed to sign the \"demo\" plugin"), "{t:?}");

        // A different key under the same id: the signature does not match it.
        let other = TestKey::generate("test-key-1");
        let svc = service(TrustPolicy::release(), other.pinned_for("Someone Else", &["demo"]), &scratch);
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("signature does not match"), "{t:?}");
    }

    /// A real Aokie release manifest (version 0.1.0), signed with the production key:
    /// what the pinned key is for. Only the envelope is kept here, not the 100 MB bundle
    /// it describes, so this proves the key and the parsing, not the digests.
    const AOKIE_RELEASE_ENVELOPE: &str = include_str!("fixtures/aokie-0.1.0.package-manifest.json");

    #[test]
    fn the_pinned_aokie_key_verifies_a_real_aokie_release_manifest() {
        let opened = open_envelope(AOKIE_RELEASE_ENVELOPE, Publishers::embedded(), "aokie").expect("the pinned key must verify Aokie's own release manifest");
        assert_eq!(opened.publisher.name, "Aokie");
        assert_eq!(opened.publisher.key_id, "fl-aokie-2026a");
        assert_eq!(opened.payload.version, "0.1.0");
        assert_eq!(opened.payload.files.len(), 21);
        assert!(opened.payload.files.iter().any(|f| f.path == "manifest.json"));
        assert!(opened.payload.files.iter().any(|f| f.path == "aokie-plugin.exe"));
        for f in &opened.payload.files {
            safe_listed_path(&f.path).unwrap();
        }

        // The publisher that signs Aokie does not get to sign anything else.
        let err = open_envelope(AOKIE_RELEASE_ENVELOPE, Publishers::embedded(), "wallet").err().unwrap();
        assert!(err.contains("not allowed to sign the \"wallet\" plugin"), "{err}");
    }

    #[test]
    fn the_shipped_pin_file_parses_and_pins_aokie_for_aokie_alone() {
        let text = include_str!("../../resources/trusted-publishers.json");
        let pins = Publishers::parse(text).expect("resources/trusted-publishers.json");
        assert!(!pins.is_empty());
        assert!(!Publishers::embedded().is_empty(), "the embedded copy must load");
        let aokie = pins.get("fl-aokie-2026a").expect("the Aokie release key");
        assert_eq!(aokie.name, "Aokie");
        assert_eq!(aokie.plugins.iter().map(String::as_str).collect::<Vec<_>>(), ["aokie"]);
    }

    // -----------------------------------------------------------------------------
    // The pin file
    // -----------------------------------------------------------------------------

    #[test]
    fn a_pin_file_with_a_bad_entry_is_refused_whole() {
        let key = TestKey::generate("k").public_b64();
        let ok = |extra: serde_json::Value| {
            let mut entry = serde_json::json!({ "id": "k1", "name": "N", "publicKey": key, "plugins": ["demo"] });
            for (k, v) in extra.as_object().unwrap() {
                entry[k] = v.clone();
            }
            Publishers::parse(&serde_json::json!({ "version": 1, "publishers": [entry] }).to_string())
        };
        assert!(ok(serde_json::json!({})).is_ok());
        assert!(ok(serde_json::json!({ "plugins": [] })).is_err(), "a key that may sign nothing is a mistake");
        assert!(ok(serde_json::json!({ "plugins": ["../evil"] })).is_err());
        assert!(ok(serde_json::json!({ "publicKey": "AAAA" })).is_err(), "a short key");
        assert!(ok(serde_json::json!({ "publicKey": "not base64!" })).is_err());
        assert!(ok(serde_json::json!({ "id": "" })).is_err());
        assert!(ok(serde_json::json!({ "name": " " })).is_err());
        assert!(Publishers::parse(r#"{"version":2,"publishers":[]}"#).is_err());
        assert!(Publishers::parse("[]").is_err());
        let twice = serde_json::json!({ "version": 1, "publishers": [
            { "id": "k1", "name": "N", "publicKey": key, "plugins": ["demo"] },
            { "id": "k1", "name": "M", "publicKey": key, "plugins": ["demo"] },
        ]});
        assert!(Publishers::parse(&twice.to_string()).is_err(), "one id, two keys");
        assert!(Publishers::parse(r#"{"version":1,"publishers":[]}"#).unwrap().is_empty());
    }

    // -----------------------------------------------------------------------------
    // A signed package that verifies
    // -----------------------------------------------------------------------------

    #[test]
    fn a_signed_package_that_verifies_is_verified_with_its_publisher() {
        let (_s, dir, key, svc) = signed(TrustPolicy::release());
        let t = svc.assess(&dir, "demo");
        assert_eq!(t.state, TrustState::Verified, "{t:?}");
        assert_eq!(t.publisher.as_deref(), Some("Demo Co"));
        assert_eq!(t.key_id.as_deref(), Some(key.key_id.as_str()));
        assert_eq!(t.version.as_deref(), Some("1.0.0"));
        assert!(t.reason.is_none());
        assert!(t.allows_launch());
    }

    #[test]
    fn a_verified_package_is_verified_in_a_developer_build_too() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::developer());
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Verified);
    }

    #[test]
    fn the_verdict_serializes_the_way_the_dashboard_reads_it() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let v = serde_json::to_value(svc.assess(&dir, "demo")).unwrap();
        assert_eq!(v["state"], "verified");
        assert_eq!(v["publisher"], "Demo Co");
        assert_eq!(v["keyId"], "test-key-1");
        assert!(v.get("reason").is_none());
        for (state, wire) in [
            (TrustState::Quarantined, "quarantined"),
            (TrustState::Unsigned, "unsigned"),
            (TrustState::UnsignedDev, "unsigned-dev"),
            (TrustState::TrustedLocal, "trusted-local"),
        ] {
            assert_eq!(serde_json::to_value(state).unwrap(), wire);
            assert_eq!(state.as_str(), wire);
        }
    }

    // -----------------------------------------------------------------------------
    // A signed package that fails: each way
    // -----------------------------------------------------------------------------

    #[test]
    fn a_changed_listed_file_quarantines() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        std::fs::write(dir.join("demo-plugin.exe"), b"EVIL executable bytes!!!!!!!").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("digest mismatch: demo-plugin.exe"), "{t:?}");
        assert!(!t.allows_launch());
        assert!(t.publisher.is_none(), "a failed package borrows nothing from the publisher");
    }

    #[test]
    fn a_change_that_keeps_the_size_is_still_a_change() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let same_length = vec![b'x'; b"demo plugin executable bytes".len()];
        std::fs::write(dir.join("demo-plugin.exe"), same_length).unwrap();
        assert!(quarantine_reason(&svc.assess_fresh(&dir, "demo")).contains("digest mismatch"));
    }

    #[test]
    fn a_listed_file_that_is_gone_quarantines() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        std::fs::remove_file(dir.join("ui").join("index.html")).unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("listed file missing: ui/index.html"), "{t:?}");
    }

    #[test]
    fn an_unlisted_executable_quarantines() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        std::fs::write(dir.join("evil.dll"), b"hijack").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("unlisted executable present: evil.dll"), "{t:?}");
        // Also one dropped into a folder the signature does know.
        std::fs::remove_file(dir.join("evil.dll")).unwrap();
        assert_eq!(svc.assess_fresh(&dir, "demo").state, TrustState::Verified);
        std::fs::write(dir.join("ui").join("helper.exe"), b"hijack").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("unlisted executable present: ui/helper.exe"), "{t:?}");
    }

    #[test]
    fn any_unlisted_file_quarantines_as_package_signer_verify_does() {
        // The bundle is immutable: a `.bak` left beside a replaced binary, or a note, is
        // a change to what was signed, and `package-signer verify` fails it too.
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        std::fs::write(dir.join("demo-plugin.exe.bak"), b"old").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("unlisted file present: demo-plugin.exe.bak"), "{t:?}");
        std::fs::remove_file(dir.join("demo-plugin.exe.bak")).unwrap();
        std::fs::create_dir_all(dir.join("data")).unwrap();
        std::fs::write(dir.join("data").join("settings.json"), b"{}").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("unlisted file present: data/settings.json"), "{t:?}");
    }

    #[test]
    fn a_file_dropped_in_an_environment_shaped_folder_quarantines() {
        // Aokie's AK-002: files once landed under a literal `%SystemDrive%`.
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        std::fs::create_dir_all(dir.join("%SystemDrive%").join("ProgramData")).unwrap();
        std::fs::write(dir.join("%SystemDrive%").join("ProgramData").join("cache.bin"), b"parked").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("unlisted file present: %SystemDrive%/ProgramData/cache.bin"), "{t:?}");
    }

    #[test]
    fn a_wrong_or_forged_signature_quarantines() {
        let (_s, dir, key, svc) = signed(TrustPolicy::release());
        let path = dir.join(PACKAGE_MANIFEST_FILE);
        let good = std::fs::read_to_string(&path).unwrap();

        // A signature with one character changed.
        let mut envelope: serde_json::Value = serde_json::from_str(&good).unwrap();
        let mut sig: Vec<char> = envelope["signature"].as_str().unwrap().chars().collect();
        sig[10] = if sig[10] == 'A' { 'B' } else { 'A' };
        envelope["signature"] = serde_json::json!(sig.into_iter().collect::<String>());
        std::fs::write(&path, envelope.to_string()).unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("signature does not match"), "{t:?}");

        // A different payload under the old signature: the payload is what was signed,
        // so its digests are never consulted.
        let mut envelope: serde_json::Value = serde_json::from_str(&good).unwrap();
        let forged = serde_json::json!({ "name": "demo-plugin", "version": "1.0.0", "createdAt": "t", "files": [
            { "path": "manifest.json", "sha256": "00".repeat(32), "size": 1 }
        ]});
        envelope["payloadB64"] = serde_json::json!(base64::engine::general_purpose::STANDARD.encode(forged.to_string()));
        std::fs::write(&path, envelope.to_string()).unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("signature does not match"), "{t:?}");

        // Signed by a key that is not the pinned one, under the pinned id.
        let impostor = TestKey::generate(&key.key_id);
        impostor.sign(&dir, "demo-plugin", "1.0.0");
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("signature does not match"), "{t:?}");
    }

    #[test]
    fn garbage_in_the_envelope_quarantines_and_never_panics() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let path = dir.join(PACKAGE_MANIFEST_FILE);
        for text in ["", "{", "[]", "null", r#"{"format":1}"#, "\u{0}\u{0}", r#"{"format":1,"alg":"Ed25519","keyId":"test-key-1","payloadB64":"!!","signature":"!!"}"#] {
            std::fs::write(&path, text).unwrap();
            let t = svc.assess_fresh(&dir, "demo");
            assert_eq!(t.state, TrustState::Quarantined, "{text:?}: {t:?}");
        }
        // An envelope that is not a file.
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert_eq!(svc.assess_fresh(&dir, "demo").state, TrustState::Quarantined);
    }

    #[test]
    fn an_envelope_format_or_algorithm_this_code_does_not_know_quarantines() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let path = dir.join(PACKAGE_MANIFEST_FILE);
        let good: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for (field, value, said) in [("format", serde_json::json!(2), "format 2"), ("alg", serde_json::json!("RS256"), "unsupported signature algorithm")] {
            let mut envelope = good.clone();
            envelope[field] = value;
            std::fs::write(&path, envelope.to_string()).unwrap();
            let t = svc.assess_fresh(&dir, "demo");
            assert!(quarantine_reason(&t).contains(said), "{field}: {t:?}");
        }
    }

    #[test]
    fn an_envelope_saved_again_with_a_byte_order_mark_still_verifies() {
        // Windows tools add one when a file is saved. The signature covers the payload
        // inside the envelope, not the envelope's bytes, so it is no change to the package.
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let path = dir.join(PACKAGE_MANIFEST_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, format!("\u{feff}{text}")).unwrap();
        assert_eq!(svc.assess_fresh(&dir, "demo").state, TrustState::Verified);
    }

    #[test]
    fn a_publisher_that_is_not_pinned_quarantines() {
        let scratch = Scratch::new("unpinned");
        let dir = scratch.package("demo");
        fill(&dir);
        let stranger = TestKey::generate("stranger-1");
        stranger.sign(&dir, "demo-plugin", "1.0.0");
        // The host pins somebody else.
        let pinned = TestKey::generate("test-key-1");
        let svc = service(TrustPolicy::release(), pinned.pinned_for("Demo Co", &["demo"]), &scratch);
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("\"stranger-1\", which this OAIY does not trust"), "{t:?}");
        // And a host that pins nobody trusts no signature at all.
        let svc = service(TrustPolicy::release(), Publishers::default(), &scratch);
        assert_eq!(svc.assess_fresh(&dir, "demo").state, TrustState::Quarantined);
    }

    #[test]
    fn a_publisher_that_may_not_sign_this_plugin_quarantines() {
        let scratch = Scratch::new("notallowed");
        let dir = scratch.package("demo");
        fill(&dir);
        let key = TestKey::generate("test-key-1");
        key.sign(&dir, "demo-plugin", "1.0.0");
        // Pinned, its signature is good, and it is a publisher for other plugins only.
        let svc = service(TrustPolicy::release(), key.pinned_for("Demo Co", &["aokie"]), &scratch);
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("Demo Co (key test-key-1) is not allowed to sign the \"demo\" plugin"), "{t:?}");
        assert!(!t.allows_launch());
    }

    #[test]
    fn a_signature_that_does_not_cover_manifest_json_quarantines() {
        let scratch = Scratch::new("nomanifest");
        let dir = scratch.package("demo");
        fill(&dir);
        let key = TestKey::generate("test-key-1");
        // Signed by a compromised or careless signer that left the entry command out.
        let mut payload: serde_json::Value = serde_json::from_slice(&key.payload_of(&dir, "demo-plugin", "1.0.0")).unwrap();
        payload["files"].as_array_mut().unwrap().retain(|f| f["path"] != "manifest.json");
        key.write_envelope(&dir, payload.to_string().as_bytes());
        let svc = service(TrustPolicy::release(), key.pinned_for("Demo Co", &["demo"]), &scratch);
        let t = svc.assess_fresh(&dir, "demo");
        assert!(quarantine_reason(&t).contains("does not cover manifest.json"), "{t:?}");
    }

    #[test]
    fn a_signature_that_lists_an_unsafe_path_quarantines() {
        for bad in ["../outside.exe", "/abs.exe", "C:/Windows/cmd.exe", "a\\b.exe", "dir/../x", "%SystemDrive%/x.dll", "trailing.", "sp ace /x", "", "a//b", "./x", PACKAGE_MANIFEST_FILE] {
            assert!(safe_listed_path(bad).is_err(), "{bad:?} must be refused");
        }
        for good in ["manifest.json", "ui/receptionist/app.js", "driver-package/aokie_winusb_bluetooth.inf", "onnxruntime_1.25.0.dll", "a b/c d.txt"] {
            assert!(safe_listed_path(good).is_ok(), "{good:?} must be accepted");
        }

        let scratch = Scratch::new("unsafepath");
        let dir = scratch.package("demo");
        fill(&dir);
        let key = TestKey::generate("test-key-1");
        let mut payload: serde_json::Value = serde_json::from_slice(&key.payload_of(&dir, "demo-plugin", "1.0.0")).unwrap();
        payload["files"].as_array_mut().unwrap().push(serde_json::json!({ "path": "../elsewhere.exe", "sha256": "00".repeat(32), "size": 0 }));
        key.write_envelope(&dir, payload.to_string().as_bytes());
        let svc = service(TrustPolicy::release(), key.pinned_for("Demo Co", &["demo"]), &scratch);
        assert!(quarantine_reason(&svc.assess_fresh(&dir, "demo")).contains("unusable path"));
    }

    #[test]
    fn a_signature_that_lists_a_file_twice_or_none_quarantines() {
        let scratch = Scratch::new("dupes");
        let dir = scratch.package("demo");
        fill(&dir);
        let key = TestKey::generate("test-key-1");
        let mut payload: serde_json::Value = serde_json::from_slice(&key.payload_of(&dir, "demo-plugin", "1.0.0")).unwrap();
        let first = payload["files"][0].clone();
        payload["files"].as_array_mut().unwrap().push(first);
        key.write_envelope(&dir, payload.to_string().as_bytes());
        let svc = service(TrustPolicy::release(), key.pinned_for("Demo Co", &["demo"]), &scratch);
        assert!(quarantine_reason(&svc.assess_fresh(&dir, "demo")).contains("twice"));

        payload["files"] = serde_json::json!([]);
        key.write_envelope(&dir, payload.to_string().as_bytes());
        assert!(quarantine_reason(&svc.assess_fresh(&dir, "demo")).contains("lists no files"));
    }

    #[test]
    fn the_reason_names_what_changed_and_how_many_more() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        for i in 0..8 {
            std::fs::write(dir.join(format!("extra{i}.txt")), b"x").unwrap();
        }
        std::fs::write(dir.join("manifest.json"), b"{}").unwrap();
        std::fs::write(dir.join("demo-plugin.exe"), b"changed").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        let reason = quarantine_reason(&t);
        assert!(reason.contains("digest mismatch: demo-plugin.exe, manifest.json"), "{reason}");
        assert!(reason.contains("unlisted file present:") && reason.contains("and 3 more"), "{reason}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_in_a_signed_package_quarantines() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        std::os::unix::fs::symlink("/etc/passwd", dir.join("link")).unwrap();
        assert!(quarantine_reason(&svc.assess_fresh(&dir, "demo")).contains("symbolic link"));
    }

    #[test]
    fn a_quarantine_tells_a_developer_how_to_run_their_own_build_and_nobody_else() {
        let (_s, dir, _k, dev) = signed(TrustPolicy::developer());
        std::fs::write(dir.join("demo-plugin.exe"), b"my own build").unwrap();
        let t = dev.assess_fresh(&dir, "demo");
        assert_eq!(t.state, TrustState::Quarantined, "a developer build does not waive quarantine: {t:?}");
        assert!(quarantine_reason(&t).contains("delete package-manifest.json"), "{t:?}");

        let (_s2, dir2, _k2, release) = signed(TrustPolicy::release());
        std::fs::write(dir2.join("demo-plugin.exe"), b"tampered").unwrap();
        let t = release.assess_fresh(&dir2, "demo");
        assert!(!quarantine_reason(&t).contains("delete package-manifest.json"), "{t:?}");
    }

    // -----------------------------------------------------------------------------
    // Unsigned packages, and the developer switch
    // -----------------------------------------------------------------------------

    #[test]
    fn an_unsigned_package_is_refused_in_a_release_build() {
        let scratch = Scratch::new("unsigned");
        let dir = scratch.package("demo");
        fill(&dir);
        let svc = service(TrustPolicy::release(), Publishers::default(), &scratch);
        let t = svc.assess(&dir, "demo");
        assert_eq!(t.state, TrustState::Unsigned);
        assert!(!t.allows_launch());
        assert!(t.reason.as_deref().unwrap().contains("trust this exact package"), "{t:?}");
        assert!(svc.authorize_launch(&dir, "demo").is_err());
    }

    #[test]
    fn an_unsigned_package_runs_in_a_developer_build_as_unsigned_dev() {
        let scratch = Scratch::new("unsigned-dev");
        let dir = scratch.package("demo");
        fill(&dir);
        let svc = service(TrustPolicy::developer(), Publishers::default(), &scratch);
        let t = svc.assess(&dir, "demo");
        assert_eq!(t.state, TrustState::UnsignedDev);
        assert!(t.allows_launch());
        let permit = svc.authorize_launch(&dir, "demo").expect("developer mode starts it");
        assert_eq!(permit.trust().state, TrustState::UnsignedDev);
    }

    #[test]
    fn an_unsigned_package_costs_no_reading() {
        // Scanning runs every few seconds. With nothing signed and nothing trusted, the
        // answer is one `stat` of the envelope's path: no walk, no hashing.
        let scratch = Scratch::new("cheap");
        let dir = scratch.package("demo");
        fill(&dir);
        std::fs::write(dir.join("big.bin"), vec![7u8; 4 * 1024 * 1024]).unwrap();
        let svc = service(TrustPolicy::release(), Publishers::default(), &scratch);
        svc.assess(&dir, "demo");
        assert!(svc.remembered().is_empty(), "nothing was hashed, so nothing is remembered");
    }

    #[test]
    fn what_counts_as_a_developer() {
        // A debug build is one, whatever OAIY_PLUGIN_DEV_MODE says: `=0` there means "use
        // the real dongle", which is how the owner answers a real line from `tauri dev`.
        for value in [None, Some("1"), Some("true"), Some("0"), Some("false"), Some("typo"), Some("")] {
            assert!(developer_mode(value, true), "a debug build with {value:?}");
        }
        // A release build only when asked to, in words it knows.
        assert!(developer_mode(Some("1"), false));
        assert!(developer_mode(Some("true"), false));
        assert!(developer_mode(Some(" 1 "), false));
        assert!(!developer_mode(None, false));
        assert!(!developer_mode(Some("0"), false));
        assert!(!developer_mode(Some("false"), false));
        // A value nobody recognises never waives trust (for hardware it means "simulate").
        assert!(!developer_mode(Some("typo"), false));
        assert!(!developer_mode(Some("yes"), false));
        assert!(!developer_mode(Some(""), false));
        // Test builds are developer builds.
        assert!(TrustPolicy::for_build().is_developer());
        assert!(!TrustPolicy::release().is_developer());
    }

    // -----------------------------------------------------------------------------
    // The person's own trust, bound to a digest
    // -----------------------------------------------------------------------------

    fn unsigned_release() -> (Scratch, PathBuf, Arc<TrustService>) {
        let scratch = Scratch::new("local");
        let dir = scratch.package("demo");
        fill(&dir);
        let svc = service(TrustPolicy::release(), Publishers::default(), &scratch);
        (scratch, dir, svc)
    }

    #[test]
    fn a_package_the_person_trusted_runs_in_a_release_build() {
        let (_s, dir, svc) = unsigned_release();
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Unsigned);

        let t = svc.trust_local(&dir, "demo").unwrap();
        assert_eq!(t.state, TrustState::TrustedLocal, "{t:?}");
        assert!(t.trusted_at.is_some());
        assert!(t.allows_launch());
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::TrustedLocal, "and the listing agrees");
        assert!(svc.authorize_launch(&dir, "demo").is_ok());
    }

    #[test]
    fn trust_is_bound_to_the_packages_digest_so_any_change_revokes_it() {
        let (_s, dir, svc) = unsigned_release();
        svc.trust_local(&dir, "demo").unwrap();
        assert!(svc.authorize_launch(&dir, "demo").is_ok());

        // A changed file.
        std::fs::write(dir.join("demo-plugin.exe"), b"a different build").unwrap();
        let t = svc.assess_fresh(&dir, "demo");
        assert_eq!(t.state, TrustState::Unsigned, "{t:?}");
        assert!(t.reason.as_deref().unwrap().contains("changed since you trusted it"), "{t:?}");
        assert!(svc.authorize_launch(&dir, "demo").is_err());

        // Trusting again binds to the new bytes...
        svc.trust_local(&dir, "demo").unwrap();
        assert!(svc.authorize_launch(&dir, "demo").is_ok());
        // ...a new file is a change...
        std::fs::write(dir.join("extra.dll"), b"new").unwrap();
        assert!(svc.authorize_launch(&dir, "demo").is_err());
        std::fs::remove_file(dir.join("extra.dll")).unwrap();
        assert!(svc.authorize_launch(&dir, "demo").is_ok(), "the same bytes again are the same package");
        // ...and so is a missing one.
        std::fs::remove_file(dir.join("ui").join("index.html")).unwrap();
        assert!(svc.authorize_launch(&dir, "demo").is_err());
    }

    #[test]
    fn trust_in_one_plugin_is_not_trust_in_another() {
        let (scratch, dir, svc) = unsigned_release();
        svc.trust_local(&dir, "demo").unwrap();
        let twin = scratch.package("twin");
        fill(&twin);
        assert_eq!(svc.assess_fresh(&twin, "twin").state, TrustState::Unsigned, "the same bytes under another id are not trusted");
    }

    #[test]
    fn trust_survives_a_restart_and_forgetting_it_takes_it_away() {
        let (scratch, dir, svc) = unsigned_release();
        svc.trust_local(&dir, "demo").unwrap();
        // A second service over the same file, as after a restart or in the headless server.
        let reopened = service(TrustPolicy::release(), Publishers::default(), &scratch);
        assert_eq!(reopened.assess(&dir, "demo").state, TrustState::TrustedLocal);

        svc.forget_local("demo");
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Unsigned);
        assert_eq!(reopened.assess(&dir, "demo").state, TrustState::Unsigned, "the other reads the file again when it changes");
    }

    #[test]
    fn a_store_that_cannot_be_read_trusts_nobody() {
        let (scratch, dir, svc) = unsigned_release();
        svc.trust_local(&dir, "demo").unwrap();
        std::fs::write(scratch.path().join(LOCAL_TRUST_FILE), b"{ this is not json").unwrap();
        let fresh = service(TrustPolicy::release(), Publishers::default(), &scratch);
        assert_eq!(fresh.assess(&dir, "demo").state, TrustState::Unsigned);
        std::fs::write(scratch.path().join(LOCAL_TRUST_FILE), br#"{"version":9,"plugins":{}}"#).unwrap();
        let fresh = service(TrustPolicy::release(), Publishers::default(), &scratch);
        assert_eq!(fresh.assess(&dir, "demo").state, TrustState::Unsigned);
    }

    #[test]
    fn a_signed_package_cannot_be_trusted_by_hand() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let err = svc.trust_local(&dir, "demo").unwrap_err();
        assert!(err.contains("cannot be trusted by hand"), "{err}");
        // Tampered, it is still judged by its signature alone.
        std::fs::write(dir.join("demo-plugin.exe"), b"tampered").unwrap();
        assert!(svc.trust_local(&dir, "demo").is_err());
        assert_eq!(svc.assess_fresh(&dir, "demo").state, TrustState::Quarantined);
    }

    #[test]
    fn trusting_refuses_what_is_not_a_plugin_id_and_a_link() {
        let (_s, dir, svc) = unsigned_release();
        for id in ["", "../demo", "a/b", "DEMO", "de mo"] {
            assert!(svc.trust_local(&dir, id).is_err(), "{id:?}");
        }
    }

    #[test]
    fn a_local_trust_record_does_not_outrank_a_signature() {
        // Trusted while unsigned; later a signature appears (or is forged). The signature
        // decides from then on.
        let (scratch, dir, svc) = unsigned_release();
        svc.trust_local(&dir, "demo").unwrap();
        let stranger = TestKey::generate("stranger-1");
        stranger.sign(&dir, "demo-plugin", "1.0.0");
        let t = svc.assess_fresh(&dir, "demo");
        assert_eq!(t.state, TrustState::Quarantined, "{t:?}");
        drop(scratch);
    }

    // -----------------------------------------------------------------------------
    // Scan, then launch
    // -----------------------------------------------------------------------------

    #[test]
    fn a_scan_reuses_its_answer_until_the_folder_changes() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Verified);
        assert_eq!(svc.remembered().len(), 1);
        // Same folder: the answer of the last scan, without hashing again.
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Verified);
        // A changed size shows up in the fingerprint, and the scan looks again.
        std::fs::write(dir.join("demo-plugin.exe"), b"a longer, different executable").unwrap();
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Quarantined);
    }

    #[test]
    fn a_file_swapped_after_the_scan_and_before_the_launch_does_not_slip_through() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        // The scan says verified.
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Verified);

        // The worst swap for a fingerprint: the same length, the modified time put back.
        let exe = dir.join("demo-plugin.exe");
        let before = std::fs::metadata(&exe).unwrap().modified().unwrap();
        std::fs::write(&exe, vec![b'!'; b"demo plugin executable bytes".len()]).unwrap();
        std::fs::File::options().write(true).open(&exe).unwrap().set_modified(before).unwrap();
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Verified, "a listing can be stale: it is only ever a display");

        // The launch does not ask the listing.
        let refused = refused_launch(svc.authorize_launch(&dir, "demo"));
        assert!(quarantine_reason(&refused).contains("digest mismatch: demo-plugin.exe"), "{refused:?}");
        // And what it found is what the listing shows from then on.
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Quarantined);
    }

    /// The verdict of a launch that was refused because of the package.
    fn refused_launch(outcome: Result<LaunchPermit, LaunchRefusal>) -> PackageTrust {
        match outcome {
            Err(LaunchRefusal::Untrusted(verdict)) => *verdict,
            other => panic!("the launch should have been refused for its package: {other:?}"),
        }
    }

    #[test]
    fn a_launch_permit_is_given_only_to_a_package_that_may_run() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let permit = svc.authorize_launch(&dir, "demo").unwrap();
        assert_eq!(permit.trust().state, TrustState::Verified);
        std::fs::write(dir.join("manifest.json"), b"{}").unwrap();
        assert!(svc.authorize_launch(&dir, "demo").is_err());
    }

    // -----------------------------------------------------------------------------
    // What a scan and a launch build a plugin from is what was verified
    // -----------------------------------------------------------------------------

    /// `MANIFEST` with another entry command of the same length.
    fn same_length_manifest() -> String {
        let swapped = MANIFEST.replace("demo-plugin.exe", "demo-plugin.cmd");
        assert_eq!(swapped.len(), MANIFEST.len());
        swapped
    }

    /// Rewrite a file with other bytes of the same length and put its modified time back:
    /// the folder's fingerprint (paths, sizes, times) does not move.
    fn swap_keeping_size_and_time(path: &Path, bytes: &[u8]) {
        let before = std::fs::metadata(path).unwrap().modified().unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), bytes.len() as u64);
        std::fs::write(path, bytes).unwrap();
        std::fs::File::options().write(true).open(path).unwrap().set_modified(before).unwrap();
    }

    #[test]
    fn a_manifest_swapped_behind_the_fingerprint_is_not_trusted_by_the_scan() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        assert_eq!(svc.assess_read(&dir, "demo", MANIFEST.as_bytes()).state, TrustState::Verified);

        let swapped = same_length_manifest();
        swap_keeping_size_and_time(&dir.join("manifest.json"), swapped.as_bytes());
        // The fingerprint cannot tell, so a plain listing still says verified...
        assert_eq!(svc.assess(&dir, "demo").state, TrustState::Verified);
        // ...but a scan that has just parsed that manifest is told the verdict is not about it.
        let t = svc.assess_read(&dir, "demo", swapped.as_bytes());
        assert!(quarantine_reason(&t).contains("digest mismatch: manifest.json"), "{t:?}");
        assert!(!t.allows_launch());
    }

    #[test]
    fn a_verdict_is_only_ever_about_the_manifest_that_was_verified() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        // The folder is untouched, but the bytes the scan parsed are not the signed ones
        // (they were read a moment before a swap was put back, say).
        let t = svc.assess_read(&dir, "demo", same_length_manifest().as_bytes());
        assert!(quarantine_reason(&t).contains("manifest.json changed while it was being read"), "{t:?}");
        // The signed bytes are what it is about.
        assert_eq!(svc.assess_read(&dir, "demo", MANIFEST.as_bytes()).state, TrustState::Verified);
    }

    #[test]
    fn a_trusted_package_is_trusted_for_the_manifest_that_was_hashed() {
        let (_s, dir, svc) = unsigned_release();
        svc.trust_local(&dir, "demo").unwrap();
        assert_eq!(svc.assess_read(&dir, "demo", MANIFEST.as_bytes()).state, TrustState::TrustedLocal);

        let swapped = same_length_manifest();
        swap_keeping_size_and_time(&dir.join("manifest.json"), swapped.as_bytes());
        let t = svc.assess_read(&dir, "demo", swapped.as_bytes());
        assert_eq!(t.state, TrustState::Unsigned, "{t:?}");
        assert!(t.reason.as_deref().unwrap().contains("changed since you trusted it"), "{t:?}");
        // And the bytes of the trusted manifest, offered for the folder that now holds another.
        assert_ne!(svc.assess_read(&dir, "demo", MANIFEST.as_bytes()).state, TrustState::TrustedLocal);
    }

    #[test]
    fn the_permit_carries_the_folder_and_the_manifest_that_were_hashed() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        let permit = svc.authorize_launch(&dir, "demo").unwrap();
        assert_eq!(permit.dir(), dir.as_path());
        assert_eq!(permit.manifest().id, "demo");
        assert_eq!(permit.manifest().entry.command, "demo-plugin.exe");

        // What the check kept is the bytes it hashed, and the cache never hands them out.
        let checked = svc.assess_with(&dir, "demo", Reuse::Fresh);
        assert_eq!(checked.manifest_json.as_deref(), Some(MANIFEST.as_bytes()));
        assert_eq!(checked.manifest_sha256.as_deref(), Some(sha256_hex(MANIFEST.as_bytes()).as_str()));
        let remembered = svc.assess_with(&dir, "demo", Reuse::Cached);
        assert!(remembered.manifest_json.is_none());
        assert_eq!(remembered.manifest_sha256, checked.manifest_sha256);
    }

    #[test]
    fn an_unsigned_package_in_a_developer_build_is_read_at_the_launch_and_a_broken_one_is_refused() {
        let scratch = Scratch::new("dev-launch");
        let dir = scratch.package("demo");
        fill(&dir);
        let svc = service(TrustPolicy::developer(), Publishers::default(), &scratch);
        let permit = svc.authorize_launch(&dir, "demo").unwrap();
        assert_eq!(permit.trust().state, TrustState::UnsignedDev);
        assert_eq!(permit.manifest().entry.command, "demo-plugin.exe");

        std::fs::write(dir.join("manifest.json"), b"{ not json").unwrap();
        match svc.authorize_launch(&dir, "demo") {
            Err(LaunchRefusal::Manifest(ManifestError::Malformed(_))) => {}
            other => panic!("a manifest that does not load is a refusal of its own: {other:?}"),
        }
    }

    #[test]
    fn a_staged_copy_is_checked_without_touching_the_scans_memory() {
        let (_s, dir, _k, svc) = signed(TrustPolicy::release());
        assert_eq!(svc.assess_staged(&dir, "demo").state, TrustState::Verified);
        assert!(svc.remembered().is_empty());
    }

    #[test]
    fn the_folder_digest_is_stable_and_sees_every_kind_of_change() {
        let scratch = Scratch::new("digest");
        let dir = scratch.package("demo");
        fill(&dir);
        let a = package_digest(&dir).unwrap();
        assert!(a.starts_with("sha256:") && a.len() == 7 + 64);
        assert_eq!(a, package_digest(&dir).unwrap());
        std::fs::write(dir.join("ui").join("index.html"), b"<p>hellp</p>").unwrap();
        let b = package_digest(&dir).unwrap();
        assert_ne!(a, b, "a changed byte");
        std::fs::rename(dir.join("ui").join("index.html"), dir.join("ui").join("home.html")).unwrap();
        assert_ne!(b, package_digest(&dir).unwrap(), "a renamed file");
    }
}
