//! The provider's PRELUDE, fetched, verified, cached and handed to every lane
//! that runs the provider's own script source.
//!
//! ```text
//!   GET {scriptProfile.path}  (If-None-Match: <last ETag>)
//!        ├─ 200 ─► verify digest + preamble declarations ─► memory + <data>/link
//!        ├─ 304 ─► keep what is held, mark it fresh
//!        └─ 503 / unreachable ─► keep the last good copy; if there is none, NONE
//!                                    │
//!   conditions ─┐                    ▼
//!   app logic  ─┼─► batch `{v:1, profile:<the document, verbatim>, jobs:[…]}`
//!   flow runs  ─┘─► `run --profile <file>`
//! ```
//!
//! # Why this exists at all
//!
//! Every logic lane on this desktop now sends the author's source to ZIPP
//! unchanged. That is right, and it was not enough: the author did not write
//! plain JavaScript. They wrote JavaScript *plus a standard library the provider
//! publishes* — its own helpers for email addresses, totals and emptiness —
//! which the provider's browser compiles ahead of every expression and this
//! desktop, until now, did not. The same condition therefore FIRED in the
//! browser and threw "… is not defined" here. Not a refusal anybody could see:
//! a ReferenceError, an `Unknown` verdict, and an automation that quietly
//! stopped.
//!
//! So the prelude is fetched from the provider and travels with every batch.
//!
//! # It is verified before it is used, every time it is read
//!
//! This is program text that runs on the user's machine, fetched over the
//! network from a server that can be wrong — compromised, half-deployed, or
//! simply serving a stale artifact. So:
//!
//! * `preambleSha256` is RECOMPUTED over the preamble and must match. A
//!   mismatch refuses the whole document: it is not cached, it does not replace
//!   the copy already held, and the lanes carry on as if the fetch had failed.
//!   The CLI verifies it again per request — two independent checks, and the
//!   one here refuses before a single job is built;
//! * a preamble that DECLARES a name the engine's own preamble, the leaf-script
//!   envelope or a lane's globals bind is refused. `var host = …` would shadow
//!   the host bridge; `var __emit = …` would forge replies; `var event = …`
//!   would shadow the very event a condition is about. None of them is a
//!   detectable failure at the other end — they are a different program that
//!   still runs;
//! * the copy on disk is verified on the way IN, not trusted because this
//!   process wrote it. A file under `<data>` is a file, and the whole argument
//!   above is about not running text nobody checked.
//!
//! # What a lane does without one
//!
//! Split by whether anybody else can do the work.
//!
//! * The LINKED lanes — the account's binding conditions, its app-logic scripts,
//!   its claimed and sealed flow runs — refuse. Nothing is lost by refusing:
//!   the heartbeat stops advertising the engine in the same breath
//!   ([`crate::link::heartbeat`]), the provider's own gate reads a desktop that
//!   takes nothing, and its browser runs the work correctly, prelude and all.
//!   Carrying on would run a DIFFERENT language against the same source and
//!   silently lose every script that touches the standard library, while the
//!   browser stood by having deferred to us.
//! * The LOCAL trigger lane carries on. Those bindings are this desktop's own;
//!   no browser will pick them up, so refusing them loses ALL local automation
//!   to protect the prelude-using subset of it. It gets the profile whenever
//!   there is one.
//! * A provider that declares no `scriptProfile` at all is untouched — no
//!   fetch, no refusal, exactly the behaviour it had before this module.
//!
//! # Nothing here knows a provider
//!
//! The path is the descriptor's, the document is the provider's, and it is
//! carried verbatim from the socket to the batch. This module never reads a
//! field of it beyond the three the protocol defines (`preamble`,
//! `preambleSha256`, `instructionSteps`) — `python` and its modes included,
//! which are the requester's to spend and the runner's to unfold.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::bridge::script_host::Prelude;

use super::descriptor::ScriptProfileSpec;
use super::LinkedAccount;

/// How long the network leg waits. Shorter than the lanes' own 30 s: this runs
/// on the heartbeat thread beside a beat that must not be delayed, and a
/// provider slow enough to need longer is a provider the last good copy covers.
const HTTP_TIMEOUT: Duration = Duration::from_secs(15);

/// How long a failed fetch is left alone. Same reasoning as the app-logic
/// catalogue's: a provider that is down must not cost every caller a timeout.
const FETCH_RETRY_AFTER: Duration = Duration::from_secs(30);

/// The largest document that will be read off the socket.
///
/// The preamble is program text, not a payload, and the whole document has to
/// fit inside a script request beside the jobs
/// ([`crate::bridge::script_host::MAX_REQUEST_BYTES`]). A provider that serves
/// more than this is not serving a prelude.
pub const MAX_PROFILE_BYTES: usize = 512 * 1024;

/// Names the ENGINE's own preamble binds — `zipp-executor.ts`
/// `ZIPP_PREAMBLE_GLOBALS`, vendored as data and pinned against that file by a
/// test. A preamble redeclaring one of these does not fail: it shadows the
/// bridge for every job in the batch.
const ZIPP_PREAMBLE_GLOBALS: &[&str] = &[
    "window",
    "navigator",
    "localStorage",
    "db",
    "host",
    "__zEvents",
    "__zHostQueue",
    "__zHostCbs",
    "__zHostId",
    "__zippHostCall",
];

/// Names the leaf-script ENVELOPE's program binds — `zipp-script.ts`
/// `SCRIPT_ENVELOPE_GLOBALS`, likewise vendored and pinned. A preamble
/// redeclaring `__emit` would forge this batch's replies.
const SCRIPT_ENVELOPE_GLOBALS: &[&str] = &[
    "__replies", "__emit", "__out", "__ctx", "__args", "__k", "__fn", "__asBody", "__jsonParse",
    "__isArray", "__hasOwn",
];

/// The names a LANE installs as `globals` for its own jobs.
///
/// The one collision neither the engine nor the CLI can see: `globals` are
/// installed after the preamble, so a preamble declaring `var event = {}` is
/// legal everywhere and simply answers a different question. This desktop only
/// ever installs one name, and it installs it in every condition.
const LANE_GLOBALS: &[&str] = &["event"];

/// The profile as it will travel: the document verbatim, plus the two figures
/// this side reads from it.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    document: Value,
    preamble_sha256: String,
    instruction_steps: Option<u64>,
    /// Serialized length of `document`, so a batch planner can charge for it
    /// without re-serializing on every job.
    bytes: usize,
}

impl Profile {
    /// The document as the provider served it. Handed to the CLI unchanged —
    /// it validates the whole thing again, and anything rebuilt here is
    /// something the two sides could disagree about.
    pub fn document(&self) -> &Value {
        &self.document
    }

    pub fn preamble_sha256(&self) -> &str {
        &self.preamble_sha256
    }

    /// `instructionSteps`, when the provider set one. Its presence is what
    /// makes `--instruction-budget` illegal beside `--profile`.
    pub fn instruction_steps(&self) -> Option<u64> {
        self.instruction_steps
    }

    /// What carrying this profile costs a request, in bytes.
    pub fn request_bytes(&self) -> usize {
        self.bytes
    }

    /// Write the document where a CLI run can be pointed at it (`--profile`).
    ///
    /// Written afresh into the run's own directory rather than shared from the
    /// cache: a refresh renaming over a file a child has open is a failure on
    /// Windows, and a run should use the profile it was prepared with even if
    /// the provider publishes another one while it runs.
    pub fn write_to(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.document.to_string())
            .map_err(|e| format!("could not write the script profile to {}: {e}", path.display()))
    }
}

/// Lower-case hex sha256 of UTF-8 text — how `preambleSha256` is computed on
/// every side of this protocol.
pub fn sha256_hex(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}

/// Check a served document and take it, or say why it cannot be taken.
///
/// Pure: no clock, no socket, no disk — so every refusal is pinned by a test
/// with a literal document.
pub fn verify(document: Value) -> Result<Profile, String> {
    let obj = document
        .as_object()
        .ok_or_else(|| "the script profile is not a JSON object".to_string())?;
    match obj.get("v").and_then(Value::as_u64) {
        Some(1) => {}
        other => {
            return Err(format!(
                "the script profile declares v={} and this desktop speaks v=1",
                other.map_or("nothing".to_string(), |v| v.to_string())
            ))
        }
    }
    let preamble = obj
        .get("preamble")
        .and_then(Value::as_str)
        .ok_or_else(|| "the script profile carries no preamble".to_string())?;
    let declared = obj
        .get("preambleSha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "the script profile carries no preambleSha256".to_string())?;
    let actual = sha256_hex(preamble);
    if actual != declared {
        // Named in full on both sides. This is the message an operator reads
        // when a provider is serving something it did not build, and "the
        // digest did not match" without the figures is not reviewable.
        return Err(format!(
            "the script profile's preamble does not match its own digest (declared {declared}, \
             computed {actual} over {} bytes) — it was NOT used",
            preamble.len()
        ));
    }
    if let Some(name) = preamble_collision(preamble) {
        return Err(format!(
            "the script profile's preamble declares {name:?}, a name the engine, the leaf-script \
             envelope or a condition's own globals bind — it was NOT used"
        ));
    }
    let instruction_steps = match obj.get("instructionSteps") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(|| format!("the script profile's instructionSteps is not a positive whole number: {v}"))?,
        ),
    };
    let bytes = serde_json::to_vec(&document).map(|b| b.len()).unwrap_or(usize::MAX);
    if bytes > MAX_PROFILE_BYTES {
        return Err(format!(
            "the script profile is {bytes} bytes and this desktop carries at most {MAX_PROFILE_BYTES}"
        ));
    }
    let preamble_sha256 = declared.to_string();
    Ok(Profile { document, preamble_sha256, instruction_steps, bytes })
}

/// A top-level declaration in `preamble` of a name something else binds.
///
/// The same scan the leaf-script envelope runs (`detectPreambleCollision`,
/// `zipp-script.ts`) — a declaration scan, not a parse, because this crate has
/// no JavaScript parser and a scan refuses what it can see rather than guessing
/// about what it cannot. The CLI has a parser and runs the wider check again.
fn preamble_collision(preamble: &str) -> Option<String> {
    for line in preamble.split('\n') {
        let Some(name) = declared_name(line) else {
            continue;
        };
        if ZIPP_PREAMBLE_GLOBALS.contains(&name)
            || SCRIPT_ENVELOPE_GLOBALS.contains(&name)
            || LANE_GLOBALS.contains(&name)
        {
            return Some(name.to_string());
        }
    }
    None
}

/// The identifier a `var`/`let`/`const`/`class`/`function` line declares, if the
/// line opens one at the top level (no leading indent but spaces and tabs).
fn declared_name(line: &str) -> Option<&str> {
    let rest = line.trim_start_matches([' ', '\t']);
    let rest = ["var", "let", "const", "class", "async function", "function"]
        .iter()
        .find_map(|kw| {
            rest.strip_prefix(*kw)
                .filter(|after| after.starts_with([' ', '\t', '*']))
        })?;
    let rest = rest.trim_start_matches([' ', '\t', '*']);
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))
        .unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

// ---------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------

#[derive(Default)]
struct State {
    held: Option<Arc<Profile>>,
    /// The `ETag` header exactly as the provider wrote it, quotes and all, to
    /// be echoed in `If-None-Match`. Memory only, deliberately: a validator
    /// that outlives its document answers 304 for a document nobody holds, and
    /// one unconditional GET per app start is a cheaper thing to be wrong about.
    etag: Option<String>,
    /// When the held copy was last CONFIRMED current — by a 200 or by a 304.
    fresh_at: Option<Instant>,
    /// When a failed fetch may be tried again.
    retry_at: Option<Instant>,
    /// Has the disk copy been looked at in this process yet?
    read_disk: bool,
    /// The last refusal said out loud, so a provider serving a bad document
    /// does not fill the log with one line per event.
    said: Option<String>,
}

/// The process-wide profile cache.
///
/// One per process rather than one per lane: four independent threads — the
/// heartbeat, the flow runner, the sealed-flow relay and the plugin event
/// thread — all need the same answer, and two caches would mean two fetches,
/// two ETags and two different preludes running side by side.
pub struct ProfileCache {
    inner: Mutex<State>,
}

static GLOBAL: OnceLock<ProfileCache> = OnceLock::new();

impl Default for ProfileCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileCache {
    pub fn new() -> Self {
        Self { inner: Mutex::new(State::default()) }
    }

    pub fn global() -> &'static ProfileCache {
        GLOBAL.get_or_init(ProfileCache::new)
    }

    /// `<data>/link/script-profile.json` — beside the stored link, because it
    /// belongs to the link and goes when the link goes.
    pub fn path(data_dir: &Path) -> PathBuf {
        data_dir.join("link").join("script-profile.json")
    }

    /// What is held right now, without a fetch and without touching the disk.
    ///
    /// For callers that must not block — the heartbeat's token decision reads
    /// this, and a beat is not the place to discover a provider is down.
    pub fn current(&self) -> Option<Arc<Profile>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).held.clone()
    }

    /// The profile for a lane about to run the provider's source: memory, else
    /// the disk copy, else one fetch (rate-limited after a failure).
    ///
    /// Never blocks twice on a provider that is down: the first failure sets a
    /// retry gate, and every caller inside that window gets the answer straight
    /// away instead of another timeout on the event thread.
    pub fn load(
        &self,
        account: &LinkedAccount,
        spec: &ScriptProfileSpec,
        data_dir: &Path,
    ) -> Option<Arc<Profile>> {
        {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if state.held.is_some() {
                return state.held.clone();
            }
            if !state.read_disk {
                state.read_disk = true;
                if let Some(profile) = read_disk(data_dir) {
                    state.held = Some(Arc::new(profile));
                    // NOT marked fresh: a copy off the disk has not been
                    // confirmed with the provider in this process, so the next
                    // revalidation happens at once rather than a TTL later.
                    return state.held.clone();
                }
            }
            if state.retry_at.is_some_and(|t| Instant::now() < t) {
                return None;
            }
        }
        self.refresh(account, spec, data_dir);
        self.current()
    }

    /// Fetch if the held copy is older than the TTL (or there is none).
    ///
    /// Called from the heartbeat tick: it already has the account and the
    /// descriptor, it already does HTTP on its own thread, and it is the thread
    /// whose capability tokens depend on the answer. Doing the polling anywhere
    /// else would put a periodic 15-second timeout on the event path.
    pub fn poll(&self, account: &LinkedAccount, spec: &ScriptProfileSpec, data_dir: &Path) {
        let ttl = Duration::from_secs(spec.ttl_seconds);
        {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if !state.read_disk {
                state.read_disk = true;
                if let Some(profile) = read_disk(data_dir) {
                    state.held = Some(Arc::new(profile));
                }
            }
            if state.retry_at.is_some_and(|t| Instant::now() < t) {
                return;
            }
            if state.held.is_some() && state.fresh_at.is_some_and(|at| at.elapsed() < ttl) {
                return;
            }
        }
        self.refresh(account, spec, data_dir);
    }

    /// One conditional GET, and what it does to the cache.
    fn refresh(&self, account: &LinkedAccount, spec: &ScriptProfileSpec, data_dir: &Path) {
        let etag = self.inner.lock().unwrap_or_else(|e| e.into_inner()).etag.clone();
        let outcome = fetch(account, spec, etag.as_deref());
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match outcome {
            Ok(Fetched::NotModified) => {
                // The one case with nothing to verify and nothing to write: the
                // provider has confirmed the document this desktop already
                // holds. Only its freshness moves.
                state.fresh_at = Some(Instant::now());
                state.retry_at = None;
                state.said = None;
            }
            Ok(Fetched::Document { document, etag }) => match verify(document) {
                Ok(profile) => {
                    if state.held.as_deref() != Some(&profile) {
                        log::info!(
                            "script profile: the provider's prelude is {} bytes, digest {}",
                            profile.request_bytes(),
                            profile.preamble_sha256()
                        );
                        write_disk(data_dir, &profile);
                    }
                    state.held = Some(Arc::new(profile));
                    state.etag = etag;
                    state.fresh_at = Some(Instant::now());
                    state.retry_at = None;
                    state.said = None;
                }
                Err(why) => {
                    // Refused, and the copy already held is left exactly where
                    // it is. A provider that starts serving a bad document must
                    // not be able to take a good one away — and the ETag is
                    // dropped so the next attempt asks unconditionally rather
                    // than being told "unchanged" about the thing it refused.
                    state.etag = None;
                    state.retry_at = Some(Instant::now() + FETCH_RETRY_AFTER);
                    Self::say(&mut state, format!("script profile REFUSED: {why}"));
                }
            },
            Err(why) => {
                state.retry_at = Some(Instant::now() + FETCH_RETRY_AFTER);
                let line = match state.held.as_ref() {
                    Some(p) => format!(
                        "script profile: could not refresh the provider's prelude ({why}); running \
                         the copy already held (digest {})",
                        p.preamble_sha256()
                    ),
                    None => format!(
                        "script profile: the provider's prelude could not be read ({why}) and \
                         there is no earlier copy — the account's logic lanes will not run"
                    ),
                };
                Self::say(&mut state, line);
            }
        }
    }

    /// Say it once per distinct reason, not once per attempt.
    fn say(state: &mut State, line: String) {
        if state.said.as_deref() != Some(line.as_str()) {
            log::warn!("{line}");
            state.said = Some(line);
        }
    }

    /// Forget everything, on disk included.
    ///
    /// A new link may be a different account on a different deployment, whose
    /// prelude is its own. Keeping the old one would run one account's standard
    /// library against another's scripts.
    pub fn invalidate(&self, data_dir: &Path) {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *state = State::default();
        let _ = std::fs::remove_file(Self::path(data_dir));
    }
}

// ---------------------------------------------------------------------------
// What a lane is handed
// ---------------------------------------------------------------------------

/// What this desktop holds of the linked provider's prelude, owned so a
/// [`Prelude`] can borrow from it for the length of one dispatch or one run.
///
/// Resolved once per LANE and held for that lane's whole batch, rather than
/// consulted per job: a refresh landing mid-dispatch must not be able to run
/// half an event's scripts under one prelude and half under another.
pub enum Held {
    /// The provider declares no prelude. Every lane behaves as it did before
    /// this module existed.
    NotRequired,
    Ready(Arc<Profile>),
    /// The provider declares one and this desktop has not got it.
    Missing(String),
}

impl Held {
    /// Borrow it as the thing a batch takes.
    pub fn prelude(&self) -> Prelude<'_> {
        match self {
            Held::NotRequired => Prelude::None,
            Held::Ready(p) => Prelude::Profile(p.document()),
            Held::Missing(why) => Prelude::Missing(why),
        }
    }

    /// The same thing for a lane with NO other runtime behind it.
    ///
    /// The local trigger bindings are this desktop's own: if this desktop does
    /// not decide them, nobody does. So a missing prelude does not stop them —
    /// it costs the conditions that use the provider's helpers, where refusing
    /// would cost every condition there is. The opposite trade to the linked
    /// lanes, for the opposite reason.
    pub fn prelude_or_bare(&self) -> Prelude<'_> {
        match self {
            Held::Missing(_) => Prelude::None,
            other => other.prelude(),
        }
    }

    pub fn profile(&self) -> Option<&Arc<Profile>> {
        match self {
            Held::Ready(p) => Some(p),
            _ => None,
        }
    }
}

/// Resolve the linked provider's prelude for one dispatch or one run.
///
/// A descriptor with no `scriptProfile` never reaches the network and never
/// refuses anything.
pub fn resolve(account: &LinkedAccount, data_dir: &Path) -> Held {
    let Some(spec) = super::descriptor::find(data_dir, &account.connector_id)
        .and_then(|d| d.script_profile)
    else {
        return Held::NotRequired;
    };
    match ProfileCache::global().load(account, &spec, data_dir) {
        Some(profile) => Held::Ready(profile),
        None => Held::Missing(format!(
            "this provider publishes the standard library its scripts are written against ({}) \
             and this desktop could not read it, so nothing was run — every helper the library \
             defines would have thrown. The provider's own runtime is told this desktop has no \
             engine while that is true, and runs the work there instead",
            spec.path
        )),
    }
}

/// What one GET produced.
enum Fetched {
    NotModified,
    Document { document: Value, etag: Option<String> },
}

fn fetch(
    account: &LinkedAccount,
    spec: &ScriptProfileSpec,
    etag: Option<&str>,
) -> Result<Fetched, String> {
    let http = reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .map_err(|e| format!("could not build the script-profile client: {e}"))?;
    let mut request = http
        .get(super::oauth::join(&account.base_url, &spec.path))
        .bearer_auth(&account.credential);
    if let Some(tag) = etag {
        request = request.header(reqwest::header::IF_NONE_MATCH, tag);
    }
    let resp = request
        .send()
        .map_err(|e| format!("could not reach the script-profile lane: {e}"))?;
    let status = resp.status();
    if status.as_u16() == 304 {
        return Ok(Fetched::NotModified);
    }
    let tag = resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    if !status.is_success() {
        let body: Value = resp.json().unwrap_or(Value::Null);
        let message = body
            .get("message")
            .or_else(|| body.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the script profile");
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "this desktop's key may not read the account's script profile ({message}) — link again"
            ));
        }
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }
    let raw = resp
        .bytes()
        .map_err(|e| format!("the script profile could not be read off the socket: {e}"))?;
    if raw.len() > MAX_PROFILE_BYTES {
        return Err(format!(
            "the script profile is {} bytes and this desktop reads at most {MAX_PROFILE_BYTES}",
            raw.len()
        ));
    }
    let document: Value = serde_json::from_slice(&raw)
        .map_err(|e| format!("the script-profile lane returned something that is not JSON: {e}"))?;
    Ok(Fetched::Document { document, etag: tag })
}

/// The disk copy, verified on the way in.
fn read_disk(data_dir: &Path) -> Option<Profile> {
    let path = ProfileCache::path(data_dir);
    let raw = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<Value>(&raw).map_err(|e| e.to_string()).and_then(verify) {
        Ok(profile) => Some(profile),
        Err(why) => {
            // Refused, and removed: a file that cannot be used is a file that
            // would be re-read and re-refused on every start.
            log::warn!("script profile: the saved copy at {} was refused ({why})", path.display());
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

/// Keep it across restarts, temp-and-rename.
///
/// Worth a file because the alternative is a window on every start in which the
/// linked lanes refuse and the provider is told this desktop has no engine — the
/// work goes back to the browser, correctly but pointlessly, for as long as the
/// first fetch takes.
fn write_disk(data_dir: &Path, profile: &Profile) {
    let path = ProfileCache::path(data_dir);
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::warn!("script profile: could not create {}: {e}", parent.display());
            return;
        }
    }
    let tmp = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp, profile.document.to_string()) {
        log::warn!("script profile: could not write {}: {e}", tmp.display());
        return;
    }
    // Windows will not rename over an existing file.
    let _ = std::fs::remove_file(&path);
    if let Err(e) = std::fs::rename(&tmp, &path) {
        log::warn!("script profile: could not save {}: {e}", path.display());
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    super::restrict_to_owner(&path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc(preamble: &str) -> Value {
        json!({ "v": 1, "preamble": preamble, "preambleSha256": sha256_hex(preamble) })
    }

    #[test]
    fn a_good_document_is_taken_with_its_figures() {
        let p = verify(json!({
            "v": 1,
            "preamble": "var validators = { email: function (v) { return true; } };",
            "preambleSha256": sha256_hex("var validators = { email: function (v) { return true; } };"),
            "instructionSteps": 200_000_000u64,
            "python": { "contract": "c/1", "files": { "a.py": "" }, "entry": "main", "call": "run" },
        }))
        .expect("a well-formed profile is taken");
        assert_eq!(p.instruction_steps(), Some(200_000_000));
        // The python block travels untouched: this side never reads it, and a
        // rebuilt copy is something the two ends could disagree about.
        assert_eq!(p.document()["python"]["contract"], "c/1");
        assert!(p.request_bytes() > 0);
    }

    #[test]
    fn a_digest_that_does_not_match_its_preamble_is_refused() {
        // The whole reason this check exists: a server that serves a preamble
        // it did not build. Both figures must be in the message.
        let bad = json!({
            "v": 1,
            "preamble": "var validators = {};",
            "preambleSha256": "0".repeat(64),
        });
        let why = verify(bad).expect_err("a mismatched digest must refuse the document");
        assert!(why.contains("does not match its own digest"), "{why}");
        assert!(why.contains(&sha256_hex("var validators = {};")), "the computed digest: {why}");
        assert!(why.contains("NOT used"), "{why}");
    }

    #[test]
    fn one_changed_byte_is_enough() {
        let mut d = doc("var sum = function () { return 1; };");
        d["preamble"] = json!("var sum = function () { return 2; };");
        assert!(verify(d).is_err(), "the digest is over the bytes, not the shape");
    }

    #[test]
    fn a_preamble_that_shadows_the_engine_the_envelope_or_a_lane_is_refused() {
        for name in ["host", "window", "__emit", "__replies", "event"] {
            for kw in ["var", "let", "const", "function", "class"] {
                let src = format!("{kw} {name} = 1;\n");
                let Err(why) = verify(doc(&src)) else {
                    panic!("a top-level `{kw} {name}` must be refused");
                };
                assert!(why.contains(name), "{why}");
                assert!(why.contains("NOT used"), "{why}");
            }
        }
    }

    #[test]
    fn the_published_prelude_declares_nothing_this_desktop_binds() {
        // The names the live provider's prelude declares, as data. If one of them
        // ever collided, every batch would be refused and no lane would run.
        let src = "function __isArr(a) { return true; }\nvar validators = {};\nvar format = {};\n\
                   var compliance = {};\nvar finance = {};\nvar safety = {};\nfunction isEmpty(v) { return false; }\n\
                   function isNotEmpty(v) { return true; }\nfunction contains(a, b) { return false; }\n\
                   function sum(a) { return 0; }\nfunction avg(a) { return 0; }\nfunction count(a) { return 0; }\n";
        assert_eq!(preamble_collision(src), None);
    }

    #[test]
    fn the_scan_refuses_exactly_what_the_envelope_refuses_no_more_and_no_less() {
        // This scan has to agree with `detectPreambleCollision` in the envelope,
        // which is a line scan with no parser: a declaration ANYWHERE at the
        // start of a line counts, indented or not. Being cleverer here would be
        // worse than being wrong — a profile this desktop accepted and the CLI
        // refused would fail every batch of every event, blamed on the request.
        assert_eq!(preamble_collision("var host = 1;\n").as_deref(), Some("host"));
        assert_eq!(
            preamble_collision("function f() {\n  var host = 1;\n}\n").as_deref(),
            Some("host"),
            "the envelope's scan cannot see nesting either, and the two must agree"
        );
        // A name inside a STRING is not a declaration on either side: the line
        // does not open with one.
        assert_eq!(preamble_collision("var msg = \"var host = 1\";\n"), None);
        // Nor is a property, an assignment or a call.
        assert_eq!(preamble_collision("globalThis.host = 1;\nhost = 2;\nhostile();\n"), None);
        assert_eq!(preamble_collision("var hostname = 1;\n"), None, "a prefix is not a name");
    }

    #[test]
    fn the_vendored_global_lists_match_the_typescript_they_were_copied_from() {
        // Two lists that must not drift: the envelope refuses a preamble on ITS
        // list, and a name missing from this one is a batch refused at the far
        // end on every single event, with the refusal blamed on the request.
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("ui")
            .join("vendor")
            .join("oaiy-core")
            .join("src");
        for (file, konst, ours) in [
            ("zipp-executor.ts", "ZIPP_PREAMBLE_GLOBALS", ZIPP_PREAMBLE_GLOBALS),
            ("zipp-script.ts", "SCRIPT_ENVELOPE_GLOBALS", SCRIPT_ENVELOPE_GLOBALS),
        ] {
            let Ok(source) = std::fs::read_to_string(root.join(file)) else {
                eprintln!("no {file} beside this crate — skipping");
                continue;
            };
            let after = source
                .split(&format!("export const {konst}: readonly string[] = ["))
                .nth(1)
                .unwrap_or_else(|| panic!("{konst} is no longer declared in {file}"));
            let block = after.split("];").next().unwrap();
            let mut names: Vec<String> = Vec::new();
            for part in block.split(',') {
                let part = part.trim();
                // Comment lines inside the list are not names.
                let part = part.lines().map(str::trim).find(|l| l.starts_with('\'')).unwrap_or(part);
                if let Some(name) = part.trim().strip_prefix('\'').and_then(|s| s.split('\'').next()) {
                    if !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
            }
            let mine: Vec<String> = ours.iter().map(|s| s.to_string()).collect();
            assert_eq!(names, mine, "{konst} in {file} has drifted from the copy in this module");
        }
    }

    #[test]
    fn the_document_is_carried_verbatim_and_never_rebuilt() {
        // Including keys this side has no opinion about. The CLI refuses an
        // unknown key, so anything added here would be a request refused.
        let served = json!({
            "v": 1,
            "preamble": "var a = 1;",
            "preambleSha256": sha256_hex("var a = 1;"),
            "instructionSteps": 5,
            "hooks": { "bindingCondition": { "prepare": "__p" } },
            "python": {
                "contract": "formlogic-python/1",
                "files": { "formlogic.py": "x = 1\n" },
                "entry": "main",
                "call": "__run__",
                "modes": [{
                    "name": "condition", "files": { "main.py": "" }, "block": "logic_block.py",
                    "before": "from formlogic import *\n", "after": "", "lineOffset": 1,
                }],
            },
        });
        let p = verify(served.clone()).expect("taken");
        assert_eq!(p.document(), &served, "not one byte of the provider's document is ours to change");
    }

    #[test]
    fn a_document_that_is_not_v1_or_carries_no_preamble_is_refused() {
        assert!(verify(json!({ "v": 2, "preamble": "", "preambleSha256": sha256_hex("") })).is_err());
        assert!(verify(json!({ "v": 1, "preambleSha256": sha256_hex("") })).is_err());
        assert!(verify(json!({ "v": 1, "preamble": "" })).is_err());
        assert!(verify(json!("not an object")).is_err());
    }

    #[test]
    fn a_saved_copy_is_verified_on_the_way_in_and_removed_when_it_fails() {
        let dir = std::env::temp_dir().join(format!("oaiy-profile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("link")).unwrap();
        let path = ProfileCache::path(&dir);

        let good = doc("var validators = {};");
        std::fs::write(&path, good.to_string()).unwrap();
        assert!(read_disk(&dir).is_some(), "a good saved copy is used");

        // Tampered after it was written — which is exactly the case a file
        // under <data> cannot be trusted about.
        std::fs::write(&path, json!({
            "v": 1,
            "preamble": "var host = 1; /* smuggled */",
            "preambleSha256": sha256_hex("var validators = {};"),
        }).to_string()).unwrap();
        assert!(read_disk(&dir).is_none(), "a tampered saved copy is refused");
        assert!(!path.exists(), "and it is not left there to be refused again on every start");
        let _ = std::fs::remove_dir_all(&dir);
    }


    // -----------------------------------------------------------------------
    // The cache, against a real socket
    // -----------------------------------------------------------------------

    /// One reply a stub provider will give, in order.
    struct Reply {
        status: &'static str,
        body: String,
        etag: Option<String>,
    }

    fn ok(body: &Value, etag: &str) -> Reply {
        Reply { status: "200 OK", body: body.to_string(), etag: Some(format!("\"{etag}\"")) }
    }

    fn unavailable() -> Reply {
        Reply {
            status: "503 Service Unavailable",
            body: r#"{"error":"script_profile_unavailable","message":"not generated"}"#.to_string(),
            etag: None,
        }
    }

    fn not_modified(etag: &str) -> Reply {
        Reply { status: "304 Not Modified", body: String::new(), etag: Some(format!("\"{etag}\"")) }
    }

    /// A provider that answers `replies` in order and records every request.
    fn stub(replies: Vec<Reply>) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let mut replies = replies.into_iter();
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let Ok(n) = stream.read(&mut buf) else { return };
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    if String::from_utf8_lossy(&raw).contains("\r\n\r\n") {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&raw).to_string();
                let _ = tx.send(text);
                let Some(reply) = replies.next() else { return };
                let mut head = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                    reply.status,
                    reply.body.len()
                );
                if let Some(tag) = &reply.etag {
                    head.push_str(&format!("ETag: {tag}\r\n"));
                }
                head.push_str("\r\n");
                let _ = write!(stream, "{head}{}", reply.body);
                let _ = stream.flush();
            }
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    fn account(base: String) -> LinkedAccount {
        LinkedAccount {
            connector_id: "stub".into(),
            base_url: base,
            credential: "flk_secret".into(),
            account_id: None,
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("oaiy-test".into()),
        }
    }

    fn spec() -> ScriptProfileSpec {
        ScriptProfileSpec { path: "/api/v1/script-profile".into(), ttl_seconds: 300 }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oaiy-profile-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("link")).unwrap();
        dir
    }

    #[test]
    fn the_profile_is_fetched_with_the_link_credential_and_kept() {
        let d = doc("var helper = 1;");
        let (base, rx) = stub(vec![ok(&d, "abc123")]);
        let dir = scratch("fetch");
        let cache = ProfileCache::new();

        let held = cache.load(&account(base), &spec(), &dir).expect("the profile is taken");
        assert_eq!(held.document(), &d);

        let request = rx.recv().unwrap();
        assert!(request.starts_with("GET /api/v1/script-profile "), "{request}");
        // The same read gate as every other lane: a bearer, not a public route.
        assert!(request.contains("authorization: Bearer flk_secret"), "{request}");
        // Nothing to revalidate against yet.
        assert!(!request.to_lowercase().contains("if-none-match"), "{request}");

        // Kept in memory: a second read costs no request at all.
        assert!(cache.current().is_some());
        // …and on disk, so the next start of the app is ready before its first
        // fetch rather than refusing the linked lanes until one lands.
        assert_eq!(
            serde_json::from_str::<Value>(&std::fs::read_to_string(ProfileCache::path(&dir)).unwrap()).unwrap(),
            d,
            "the saved copy is the document, byte for byte"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_304_keeps_the_copy_already_held_and_refreshes_nothing_else() {
        // The whole reason the ETag is stored. Without it every poll would pull
        // the provider's whole standard library down again, and a 304 that was
        // not understood would look like a failure.
        let d = doc("var helper = 1;");
        let (base, rx) = stub(vec![ok(&d, "abc123"), not_modified("abc123")]);
        let dir = scratch("304");
        let account = account(base);
        let mut spec = spec();
        spec.ttl_seconds = 1;
        let cache = ProfileCache::new();

        assert!(cache.load(&account, &spec, &dir).is_some());
        let first = rx.recv().unwrap();
        assert!(!first.to_lowercase().contains("if-none-match"), "{first}");

        // Past the TTL, so the next poll revalidates.
        std::thread::sleep(Duration::from_millis(1100));
        cache.poll(&account, &spec, &dir);
        let second = rx.recv().unwrap();
        assert!(
            second.contains("if-none-match: \"abc123\""),
            "the provider's own tag, echoed verbatim: {second}"
        );

        let held = cache.current().expect("a 304 means KEEP, not drop");
        assert_eq!(held.document(), &d);
        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn inside_the_ttl_a_poll_asks_the_provider_nothing_at_all() {
        // The heartbeat calls this every five seconds for the life of the app.
        // Without the TTL that is a request — and, on a document this size, a
        // download — every five seconds, for ever, per linked desktop.
        let d = doc("var helper = 1;");
        let (base, rx) = stub(vec![ok(&d, "abc123")]);
        let dir = scratch("ttl");
        let account = account(base);
        let cache = ProfileCache::new();

        cache.poll(&account, &spec(), &dir);
        assert!(rx.recv().unwrap().starts_with("GET "));
        for _ in 0..5 {
            cache.poll(&account, &spec(), &dir);
        }
        assert!(rx.try_recv().is_err(), "one fetch, then the TTL answers");
        assert!(cache.current().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_deployment_with_no_artifact_leaves_the_lanes_without_one() {
        // 503 `script_profile_unavailable`. Not fatal to the process, and not
        // silently "carry on": there is nothing to run the provider's scripts
        // with, so `resolve` reports Missing and the lanes refuse.
        let (base, rx) = stub(vec![unavailable()]);
        let dir = scratch("503");
        let cache = ProfileCache::new();

        assert!(cache.load(&account(base), &spec(), &dir).is_none());
        assert!(rx.recv().unwrap().starts_with("GET /api/v1/script-profile "));
        assert!(!ProfileCache::path(&dir).exists(), "nothing is saved from a refusal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_provider_that_goes_down_does_not_take_the_copy_already_held_with_it() {
        // A prelude does not rot. Dropping it because one revalidation failed
        // would turn a network blip into the very state this module exists to
        // prevent — and would hand the work back to the browser for no reason.
        let d = doc("var helper = 1;");
        let (base, _rx) = stub(vec![ok(&d, "abc123")]);
        let dir = scratch("down");
        let account = account(base);
        let mut spec = spec();
        spec.ttl_seconds = 1;
        let cache = ProfileCache::new();
        assert!(cache.load(&account, &spec, &dir).is_some());

        // The stub answers once and then closes: every later request fails.
        std::thread::sleep(Duration::from_millis(1100));
        cache.poll(&account, &spec, &dir);
        assert_eq!(
            cache.current().expect("still held").document(),
            &d,
            "a failed refresh keeps the last good copy"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_provider_serving_a_document_that_fails_its_own_digest_cannot_take_a_good_one_away() {
        // The security case, end to end. A server that starts serving a preamble
        // it did not build must not be able to replace the one this desktop
        // verified — and must not be able to run it either.
        let good = doc("var helper = 1;");
        let tampered = json!({
            "v": 1,
            "preamble": "var helper = 1; globalThis.stolen = true;",
            "preambleSha256": sha256_hex("var helper = 1;"),
        });
        let (base, _rx) = stub(vec![ok(&good, "t1"), ok(&tampered, "t2")]);
        let dir = scratch("tampered");
        let account = account(base);
        let mut spec = spec();
        spec.ttl_seconds = 1;
        let cache = ProfileCache::new();
        assert!(cache.load(&account, &spec, &dir).is_some());

        std::thread::sleep(Duration::from_millis(1100));
        cache.poll(&account, &spec, &dir);
        let held = cache.current().expect("the good copy is still held");
        assert_eq!(held.document(), &good, "the tampered document was refused, not taken");
        assert!(
            !held.document()["preamble"].as_str().unwrap().contains("stolen"),
            "and nothing of it reached the cache"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&std::fs::read_to_string(ProfileCache::path(&dir)).unwrap()).unwrap(),
            good,
            "nor the disk"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_saved_copy_is_ready_before_the_first_fetch_of_the_next_start() {
        // A restart, simulated exactly: the same data dir, a brand-new cache.
        // Without this the linked lanes refuse — and the beat says "no engine" —
        // for as long as the first fetch takes, on every single launch.
        let d = doc("var helper = 1;");
        let dir = scratch("restart");
        std::fs::write(ProfileCache::path(&dir), d.to_string()).unwrap();

        // A port nothing is listening on: any fetch at all would fail.
        let cache = ProfileCache::new();
        let held = cache
            .load(&account("http://127.0.0.1:9".into()), &spec(), &dir)
            .expect("the saved copy answers before the network is touched");
        assert_eq!(held.document(), &d);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_provider_that_is_down_is_not_retried_on_every_single_event() {
        // This runs on the event thread when a lane has to fetch for itself, so
        // a provider that is down would otherwise cost EVERY event a full HTTP
        // timeout — and the events that pile up behind it are dropped.
        let (base, rx) = stub(vec![unavailable(), unavailable()]);
        let dir = scratch("retry");
        let account = account(base);
        let cache = ProfileCache::new();
        assert!(cache.load(&account, &spec(), &dir).is_none());
        assert!(cache.load(&account, &spec(), &dir).is_none());
        assert!(cache.load(&account, &spec(), &dir).is_none());
        assert!(rx.recv().unwrap().starts_with("GET "));
        assert!(
            rx.try_recv().is_err(),
            "one attempt, then the retry gate — not one per caller"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unlinking_forgets_the_providers_prelude_on_disk_as_well() {
        // The next link may be another account on another deployment. Running
        // one account's standard library inside another's scripts is worse than
        // having none at all.
        let dir = scratch("unlink");
        std::fs::write(ProfileCache::path(&dir), doc("var helper = 1;").to_string()).unwrap();
        let cache = ProfileCache::new();
        assert!(cache
            .load(&account("http://127.0.0.1:9".into()), &spec(), &dir)
            .is_some());
        cache.invalidate(&dir);
        assert!(cache.current().is_none());
        assert!(!ProfileCache::path(&dir).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_provider_name_is_written_into_this_module() {
        let source = include_str!("script_profile.rs").replace("\r\n", "\n");
        let code_only = source.split("#[cfg(test)]\nmod tests").next().unwrap().to_string();
        for forbidden in ["formlogic", "FormLogic", "validators", "logic-engine", "__flBinding"] {
            assert!(
                !code_only.contains(forbidden),
                "{forbidden:?} must not appear outside the tests"
            );
        }
    }
}
