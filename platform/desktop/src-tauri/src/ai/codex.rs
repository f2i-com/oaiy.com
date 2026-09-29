//! Managed Codex app-server adapter — the ChatGPT connector.
//!
//! This is NOT an editable provider with an API key. It runs the official
//! `codex` CLI as a supervised child in `app-server` mode and speaks
//! line-delimited JSON-RPC over its stdio. **The child owns ChatGPT OAuth**: the
//! login flow, the refresh tokens and the credential file all live inside a
//! dedicated `CODEX_HOME`, and OAIY only ever passes login start/cancel/logout
//! through and reads back account metadata. No token enters this process, and
//! raw JSON-RPC never crosses the HTTP boundary.
//!
//! Scope note vs. the FormLogic reference this is adapted from: OAIY does not
//! (yet) apply the Windows EFS + protected-DACL hardening to `CODEX_HOME`, so
//! the credential file is protected by normal user-profile permissions, the same
//! posture as OAIY's other on-device secrets. That is a real difference and is
//! surfaced in the status payload rather than hidden.

use super::chat_tools;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// The provider id this agent answers to, in the sources union and in
/// `/api/ai/providers/<id>/v1/chat/completions`.
///
/// This is the GENERIC route: it accepts whatever model the caller names. The
/// fixed live-call aliases below pin their own, and exist for a different
/// reason — see [`LiveCallAlias`].
pub const CODEX_PROVIDER_ID: &str = "openai-codex-agent";

/// Fixed ChatGPT routes for taking a live phone call.
///
/// A call cannot wait on a reasoning model: the caller hears the silence. Each
/// alias therefore pins a model AND a reasoning effort, rather than letting the
/// caller choose and discovering mid-call that they chose badly.
///
/// They are separate provider IDS rather than parameters because the Aokie
/// plugin recognises a live-call route BY ITS URL, and applies a policy it
/// cannot apply to a generic one: it forces the matching model and refuses to
/// send caller audio. A parameter on the generic route would be invisible to
/// that check.
///
/// Ported from FormLogic Desktop so the same plugin, pointed at either host,
/// gets the same four routes under the same names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveCallAlias {
    ReasoningNone,
    ReasoningLow,
    LunaReasoningLow,
    LunaReasoningLowFast,
}

pub const LIVE_CALL_MODEL: &str = "gpt-5.5";
pub const LUNA_LIVE_CALL_MODEL: &str = "gpt-5.6-luna";

impl LiveCallAlias {
    /// The provider id, i.e. the `<id>` in the gateway path.
    pub fn id(self) -> &'static str {
        match self {
            Self::ReasoningNone => "openai-codex-agent-none",
            Self::ReasoningLow => "openai-codex-agent-low",
            Self::LunaReasoningLow => "openai-codex-agent-luna-low",
            Self::LunaReasoningLowFast => "openai-codex-agent-luna-low-fast",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        [
            Self::ReasoningNone,
            Self::ReasoningLow,
            Self::LunaReasoningLow,
            Self::LunaReasoningLowFast,
        ]
        .into_iter()
        .find(|a| a.id() == id)
    }

    pub fn model(self) -> &'static str {
        match self {
            Self::ReasoningNone | Self::ReasoningLow => LIVE_CALL_MODEL,
            Self::LunaReasoningLow | Self::LunaReasoningLowFast => LUNA_LIVE_CALL_MODEL,
        }
    }

    pub fn reasoning_effort(self) -> &'static str {
        match self {
            Self::ReasoningNone => "none",
            Self::ReasoningLow | Self::LunaReasoningLow | Self::LunaReasoningLowFast => "low",
        }
    }

    /// Luna's fast route asks for priority service; the rest take the default.
    pub fn service_tier(self) -> Option<&'static str> {
        match self {
            Self::LunaReasoningLowFast => Some("priority"),
            _ => None,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::ReasoningNone => "ChatGPT / Codex — GPT-5.5, reasoning off",
            Self::ReasoningLow => "ChatGPT / Codex — GPT-5.5, low reasoning",
            Self::LunaReasoningLow => "ChatGPT / Codex — GPT-5.6 Luna, low reasoning",
            Self::LunaReasoningLowFast => {
                "ChatGPT / Codex — GPT-5.6 Luna, low reasoning, Fast mode"
            }
        }
    }

    pub fn all() -> [Self; 4] {
        [
            Self::ReasoningNone,
            Self::ReasoningLow,
            Self::LunaReasoningLow,
            Self::LunaReasoningLowFast,
        ]
    }
}

const RPC_TIMEOUT: Duration = Duration::from_secs(60);
/// A turn can take a while; the caller's HTTP timeout is the real bound.
const TURN_TIMEOUT: Duration = Duration::from_secs(180);

/// Hosts a Codex login URL is allowed to point at. A login URL is handed to the
/// user's browser, so an unexpected host would be a redirect to somewhere we did
/// not intend.
const LOGIN_HOSTS: [&str; 3] = ["auth.openai.com", "chatgpt.com", "platform.openai.com"];

#[derive(Debug)]
pub enum CodexError {
    /// The `codex` CLI is not installed / not runnable.
    Unavailable(String),
    /// Signed out — the caller should start a login.
    NotAuthenticated,
    /// The child answered with a JSON-RPC error, or misbehaved.
    Rpc(String),
}

impl CodexError {
    pub fn code(&self) -> &'static str {
        match self {
            CodexError::Unavailable(_) => "codex_unavailable",
            CodexError::NotAuthenticated => "codex_not_authenticated",
            CodexError::Rpc(_) => "codex_error",
        }
    }
    pub fn message(&self) -> String {
        match self {
            CodexError::Unavailable(m) | CodexError::Rpc(m) => m.clone(),
            CodexError::NotAuthenticated => {
                "not signed in to ChatGPT — start a login from OAIY Desktop → Providers".into()
            }
        }
    }
}

/// How a session reaches its app-server: the two pipes it speaks over, and the
/// child behind them. In production that is the `codex` CLI ([`spawn_codex`]);
/// the tests connect an in-process fake speaking the same protocol, with no
/// child at all.
struct Transport {
    to_server: Box<dyn Write + Send>,
    from_server: Box<dyn Read + Send>,
    child: Option<Child>,
}

/// Starts an app-server for a `CODEX_HOME`.
type Connect = Arc<dyn Fn(&Path) -> Result<Transport, CodexError> + Send + Sync>;

/// The real app-server: the `codex` CLI, in `app-server` mode, on stdio.
fn spawn_codex(codex_home: &Path) -> Result<Transport, CodexError> {
    // The ChatGPT credential file the child writes lives in here, so where this
    // process makes the folder it makes it owner-only (see `secret_file`).
    crate::secret_file::create_private_dir(codex_home)
        .map_err(|e| CodexError::Unavailable(format!("cannot create CODEX_HOME: {e}")))?;

    let mut cmd = base_command();
    cmd.arg("app-server");
    // Nothing of this process's environment reaches the child except an
    // explicit allow-list: no PATH games, no OPENAI_API_KEY, no proxy vars.
    cmd.env_clear();
    for key in env_allow_list() {
        if let Ok(v) = std::env::var(key) {
            cmd.env(key, v);
        }
    }
    cmd.env("CODEX_HOME", codex_home);
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| CodexError::Unavailable(format!("cannot run the codex CLI: {e}")))?;
    // On Windows this child is `cmd /c codex`, so killing the tracked pid
    // reaps the shell and leaves the real agent running with a live
    // CODEX_HOME session. Job membership is inherited, so the whole tree
    // goes together.
    crate::services::job_object::adopt(child.id());
    let stdin: ChildStdin = child.stdin.take().ok_or_else(|| CodexError::Unavailable("no stdin".into()))?;
    let stdout = child.stdout.take().ok_or_else(|| CodexError::Unavailable("no stdout".into()))?;
    Ok(Transport { to_server: Box::new(stdin), from_server: Box::new(stdout), child: Some(child) })
}

/// Where the reader thread delivers what the app-server sends.
#[derive(Default)]
struct Routes {
    /// Request id → whoever is waiting for that reply.
    replies: HashMap<i64, Sender<Value>>,
    /// Thread id → the turn reading that thread's notifications.
    threads: HashMap<String, Sender<Value>>,
    /// The app-server's stdout has closed: nothing more will come.
    closed: bool,
}

/// One live app-server and its RPC plumbing, shared by every caller at once.
///
/// Nothing here is held for longer than it takes to write a line or update a
/// map. A caller waits on a channel of its OWN, for its own reply or its own
/// thread's notifications, so a long turn holds up nobody else's.
struct Session {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Box<dyn Write + Send>>,
    next_id: AtomicI64,
    routes: Arc<Mutex<Routes>>,
}

impl Session {
    fn start(transport: Transport) -> Self {
        let Transport { to_server, from_server, child } = transport;
        let routes: Arc<Mutex<Routes>> = Arc::default();
        let reading = routes.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(from_server).lines() {
                let Ok(line) = line else { break };
                let Ok(msg) = serde_json::from_str::<Value>(line.trim()) else { continue };
                deliver(&reading, msg);
            }
            // The app-server is gone. Dropping every waiter's sender wakes it
            // now, with an error, rather than when its timeout runs out.
            let mut routes = lock(&reading);
            routes.closed = true;
            routes.replies.clear();
            routes.threads.clear();
        });
        Session { child: Mutex::new(child), stdin: Mutex::new(to_server), next_id: AtomicI64::new(0), routes }
    }

    fn write(&self, msg: Value) -> Result<(), CodexError> {
        let mut stdin = lock(&self.stdin);
        writeln!(stdin, "{msg}").map_err(|e| CodexError::Rpc(format!("write failed: {e}")))?;
        stdin.flush().ok();
        Ok(())
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), CodexError> {
        self.write(json!({ "jsonrpc": "2.0", "method": method, "params": params }))
    }

    fn call(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, CodexError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = channel();
        {
            let mut routes = lock(&self.routes);
            if routes.closed {
                return Err(exited());
            }
            // Before the request is written, so even an instant reply finds it.
            routes.replies.insert(id, tx);
        }
        if let Err(e) = self.write(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })) {
            lock(&self.routes).replies.remove(&id);
            return Err(e);
        }
        let msg = match rx.recv_timeout(timeout) {
            Ok(msg) => msg,
            Err(RecvTimeoutError::Timeout) => {
                lock(&self.routes).replies.remove(&id);
                return Err(CodexError::Rpc(format!("{method} timed out")));
            }
            Err(RecvTimeoutError::Disconnected) => return Err(exited()),
        };
        if let Some(err) = msg.get("error") {
            return Err(CodexError::Rpc(
                err.get("message").and_then(Value::as_str).unwrap_or("codex error").to_string(),
            ));
        }
        Ok(msg.get("result").cloned().unwrap_or(Value::Null))
    }

    /// `thread`'s notifications from now on, until the listener is dropped.
    fn listen(&self, thread: &str) -> Result<ThreadListener<'_>, CodexError> {
        let (tx, rx) = channel();
        let mut routes = lock(&self.routes);
        if routes.closed {
            return Err(exited());
        }
        routes.threads.insert(thread.to_string(), tx);
        Ok(ThreadListener { session: self, thread: thread.to_string(), notes: rx })
    }

    /// One turn: `thread/start` and `turn/start` as `request` says, and the
    /// agent's whole message out, each fragment handed to `on_delta` as it
    /// arrives.
    fn turn(&self, request: &TurnRequest, on_delta: &mut dyn FnMut(&str)) -> Result<String, CodexError> {
        let thread = self.call("thread/start", request.thread.clone(), RPC_TIMEOUT)?;
        let thread_id = thread_id_of(&thread)
            .ok_or_else(|| CodexError::Rpc("codex did not return a thread id".into()))?;

        // Listening BEFORE the turn starts, so none of its notifications can
        // arrive unheard.
        let listener = self.listen(&thread_id)?;
        let mut turn = request.turn.clone();
        turn["threadId"] = Value::String(thread_id);
        self.call("turn/start", turn, TURN_TIMEOUT)?;

        // Collect the agent's message from this thread's notifications, each
        // taken off the channel exactly once (see `fold_turn_notes`).
        let deadline = Instant::now() + TURN_TIMEOUT;
        let mut out = String::new();
        loop {
            let note = match listener.notes.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(note) => note,
                // Out of time: whatever arrived is the answer, as it always was.
                Err(RecvTimeoutError::Timeout) => break,
                // The app-server died mid-turn: this is not an answer.
                Err(RecvTimeoutError::Disconnected) => return Err(exited()),
            };
            let before = out.len();
            let done = fold_turn_notes(&mut out, std::slice::from_ref(&note));
            // Whatever this note appended, verbatim — so a streaming caller
            // sees exactly the text the buffered answer will contain and the
            // two can never drift apart.
            if out.len() > before {
                on_delta(&out[before..]);
            }
            if done {
                break;
            }
        }
        if out.is_empty() {
            return Err(CodexError::Rpc("codex returned no output for this turn".into()));
        }
        Ok(out)
    }

    /// The app-server is still there to answer.
    fn alive(&self) -> bool {
        let closed = lock(&self.routes).closed;
        !closed && lock(&self.child).as_mut().map_or(true, |c| c.try_wait().ok().flatten().is_none())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(child) = lock(&self.child).as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// One turn's hold on its thread's notifications. Dropped, it lets them go,
/// so a turn that ended — or gave up — leaves nothing to pile up.
struct ThreadListener<'a> {
    session: &'a Session,
    thread: String,
    notes: Receiver<Value>,
}

impl Drop for ThreadListener<'_> {
    fn drop(&mut self) {
        lock(&self.session.routes).threads.remove(&self.thread);
    }
}

/// Hands one message from the app-server to whoever is waiting for it.
///
/// Never blocks: every channel is unbounded, so a turn slow to read its own
/// notifications cannot hold up another's.
fn deliver(routes: &Mutex<Routes>, msg: Value) {
    let mut routes = lock(routes);
    match msg.get("id").and_then(Value::as_i64) {
        // A reply to something we asked. One whose caller gave up has no
        // waiter any more, and is dropped.
        Some(id) if msg.get("method").is_none() => {
            if let Some(waiter) = routes.replies.remove(&id) {
                let _ = waiter.send(msg);
            }
        }
        // A server→client REQUEST. We expose no tools, so refusing is
        // the correct answer (and prevents the child from stalling).
        Some(_) => {}
        // A notification, to the turn on its thread. One for no thread
        // (account news, rate limits) or for a thread nobody is reading any
        // more is dropped.
        None => {
            let thread = msg.pointer("/params/threadId").and_then(Value::as_str);
            if let Some(listener) = thread.and_then(|t| routes.threads.get(t)) {
                let _ = listener.send(msg);
            }
        }
    }
}

fn exited() -> CodexError {
    CodexError::Rpc("the codex app-server exited".into())
}

/// A lock that outlives a panic elsewhere: nothing here is left half-changed
/// by one, and the connector must keep answering.
fn lock<T: ?Sized>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A first-come, first-served line: one holder at a time, in the order they
/// asked. `std`'s `Mutex` promises no order, and two turns of one call taken
/// out of order would answer the caller's second sentence before the first.
#[derive(Default)]
struct Lane {
    tickets: Mutex<Tickets>,
    next_up: Condvar,
}

#[derive(Default)]
struct Tickets {
    issued: u64,
    serving: u64,
}

impl Lane {
    /// Waits for this caller's turn; it lasts until the result is dropped.
    fn enter(&self) -> InLane<'_> {
        let mut tickets = lock(&self.tickets);
        let mine = tickets.issued;
        tickets.issued += 1;
        while tickets.serving != mine {
            tickets = self.next_up.wait(tickets).unwrap_or_else(PoisonError::into_inner);
        }
        InLane(self)
    }

    /// In the lane now: the holder and those waiting behind it.
    #[cfg(test)]
    fn queued(&self) -> u64 {
        let tickets = lock(&self.tickets);
        tickets.issued - tickets.serving
    }
}

struct InLane<'a>(&'a Lane);

impl Drop for InLane<'_> {
    fn drop(&mut self) {
        lock(&self.0.tickets).serving += 1;
        self.0.next_up.notify_all();
    }
}

fn base_command() -> Command {
    // `codex` ships as a shell shim on Windows (codex.cmd), which only resolves
    // through the command processor.
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.arg("/c").arg("codex");
        // `cmd` is a console program and this app has no console, so without
        // this Windows opens a visible one — once per status poll, which during
        // a pending sign-in is every 3 seconds.
        crate::HiddenCommand::pipe_hidden(&mut c);
        c
    }
    #[cfg(not(windows))]
    {
        Command::new("codex")
    }
}

fn env_allow_list() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["TEMP", "TMP", "SystemRoot", "WINDIR", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "PATH"]
    }
    #[cfg(not(windows))]
    {
        &["HOME", "TMPDIR", "PATH", "LANG", "LC_ALL", "XDG_RUNTIME_DIR"]
    }
}

/// Accept only an https login URL on a known OpenAI host.
fn safe_login_url(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    if raw.len() > 4096 {
        return None;
    }
    let url = reqwest::Url::parse(raw).ok()?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    let host = url.host_str()?;
    LOGIN_HOSTS.contains(&host).then(|| raw.to_string())
}

/// The managed agent: one app-server child, started lazily and reused, with
/// turns running on it side by side.
///
/// # A call never waits behind a setup turn
///
/// The app-server multiplexes. Each turn runs on a thread of its own,
/// `turn/start` answers at once while the turn carries on, and every
/// notification a turn sends (each delta, `item/completed`, `turn/completed`)
/// names its `threadId`. So the one child can answer a live call while a long
/// project or setup turn is still going. The [`Session`] routes each reply to
/// the request that asked for it, by JSON-RPC id, and each notification to the
/// turn on its thread, and holds nothing while a turn runs. It used to hold one
/// lock for the whole turn, and a caller heard silence until the setup turn
/// had finished.
///
/// What still takes turns is decided here, in two first-come, first-served
/// [`Lane`]s:
///
/// - The live-call aliases' turns go one at a time, in the order they came, so
///   two turns of one call never overtake each other. The routes carry no call
///   id, so calls share the lane: two calls at once take turns with each other,
///   never with anything else.
/// - Every other turn (project, setup, runner, sms, task, and the tunnel's
///   chat) goes one at a time in its own lane, as every turn did before. The
///   account sees at most one of those plus one call turn at once.
///
/// Sign-in, status, sign-out and the model list take no lane. They are single
/// requests, answered beside whatever turns are running.
///
/// # Why not a second child for calls
///
/// A second app-server on the same `CODEX_HOME` was the other way to keep
/// calls apart, and it is the riskier one. Each child keeps its own copy of the
/// ChatGPT credentials in memory, and writes `auth.json` itself when it
/// refreshes them. The refresh token ROTATES: once one child has refreshed, the
/// other's copy is spent, and Codex's answer to a spent token is "your refresh
/// token was already used — please log out and sign in again". Codex 0.144
/// re-reads `auth.json` before refreshing, which narrows that race, but no lock
/// against another process is evident in the shipped binary. Two children
/// would also mean two login flows to keep apart, and a sign-out through one
/// leaving the other signed in until it was restarted. One child has one copy
/// of the credentials, one login flow and one sign-out, and nothing new to stop
/// on shutdown.
pub struct CodexAgent {
    codex_home: PathBuf,
    connect: Connect,
    session: Mutex<Option<Arc<Session>>>,
    /// Live-call turns: one at a time, in the order they came.
    call_lane: Lane,
    /// Every other turn: one at a time, in the order they came.
    turn_lane: Lane,
}

pub type CodexHandle = Arc<CodexAgent>;

pub fn new_handle(data_dir: &Path) -> CodexHandle {
    CodexAgent::with_connect(data_dir.join("ai").join("codex-home"), Arc::new(spawn_codex))
}

/// For tests of the routes that ask what ChatGPT can do: the CLI is never there,
/// so nothing is launched (a developer's machine may well have a real one).
#[cfg(test)]
pub(crate) fn absent_for_tests() -> CodexHandle {
    CodexAgent::with_connect(
        PathBuf::from("no-codex-home"),
        Arc::new(|_home: &Path| -> Result<Transport, CodexError> {
            Err(CodexError::Unavailable("the codex CLI is not installed (a test)".into()))
        }),
    )
}

impl CodexAgent {
    fn with_connect(codex_home: PathBuf, connect: Connect) -> CodexHandle {
        Arc::new(CodexAgent {
            codex_home,
            connect,
            session: Mutex::new(None),
            call_lane: Lane::default(),
            turn_lane: Lane::default(),
        })
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexStatus {
    /// The CLI is installed and the app server answered.
    pub available: bool,
    /// A ChatGPT account is signed in.
    pub connected: bool,
    pub email: Option<String>,
    pub plan_type: Option<String>,
    pub account_type: Option<String>,
    /// Why it isn't available, when it isn't.
    pub detail: Option<String>,
}

impl CodexAgent {
    /// Run `f` against a live, initialized session, starting one if needed.
    ///
    /// The session is shared: `f` runs beside any other caller's, holding no
    /// lock of this agent's while it does.
    fn with_session<T>(&self, f: impl FnOnce(&Session) -> Result<T, CodexError>) -> Result<T, CodexError> {
        let session = self.session()?;
        f(&session)
    }

    /// The live, initialized app-server, starting one if there is none.
    fn session(&self) -> Result<Arc<Session>, CodexError> {
        let mut slot = lock(&self.session);
        if let Some(s) = slot.as_ref().filter(|s| s.alive()) {
            return Ok(s.clone());
        }
        // A dead child must not be reused: drop it so this call respawns. A
        // turn still holding it has already been told it exited.
        *slot = None;
        let s = Arc::new(Session::start((self.connect)(&self.codex_home)?));
        s.call(
            "initialize",
            json!({
                "clientInfo": { "name": "oaiy-desktop", "title": "OAIY Desktop",
                                "version": env!("CARGO_PKG_VERSION") },
                // Required, not optional. `thread/start` carries
                // `runtimeWorkspaceRoots` — the field that pins a turn to
                // NO workspace — and the runtime refuses that parameter
                // outright unless this capability was negotiated here.
                // Without it every chat fails with
                // "requires experimentalApi capability", and dropping the
                // parameter instead would hand the agent a default
                // workspace, which is the opposite of what it is for.
                "capabilities": { "experimentalApi": true }
            }),
            RPC_TIMEOUT,
        )?;
        s.notify("initialized", json!({}))?;
        *slot = Some(s.clone());
        Ok(s)
    }

    /// Sign-in state + whether the CLI is usable at all. Never errors: the panel
    /// wants to render "not installed" as a state, not a failure.
    pub fn status(&self) -> CodexStatus {
        match self.with_session(|s| s.call("account/read", json!({ "refreshToken": false }), RPC_TIMEOUT)) {
            Ok(v) => {
                // `requiresOpenaiAuth` is ALWAYS true and is not the sign-in
                // fact — the presence of a chatgpt account object is.
                let acct = v.get("account").filter(|a| !a.is_null());
                let account_type = acct
                    .and_then(|a| a.get("accountType").or_else(|| a.get("type")))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                CodexStatus {
                    available: true,
                    connected: acct.is_some(),
                    email: acct.and_then(|a| a.get("email")).and_then(Value::as_str).map(str::to_string),
                    plan_type: acct
                        .and_then(|a| a.get("planType").or_else(|| a.get("plan")))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    account_type,
                    detail: None,
                }
            }
            Err(e) => CodexStatus {
                available: !matches!(e, CodexError::Unavailable(_)),
                connected: false,
                email: None,
                plan_type: None,
                account_type: None,
                detail: Some(e.message()),
            },
        }
    }

    /// Begin a ChatGPT sign-in. Returns the URL (and device code, when the CLI
    /// chooses the device flow) for the user to complete in a browser.
    pub fn start_login(&self, device_code: bool) -> Result<Value, CodexError> {
        let kind = if device_code { "chatgptDeviceCode" } else { "chatgpt" };
        let v = self.with_session(|s| {
            s.call("account/login/start", json!({ "type": kind }), RPC_TIMEOUT)
        })?;
        // Only hand back fields we validated — never the raw RPC result.
        Ok(json!({
            "loginId": v.get("loginId").and_then(Value::as_str),
            "authUrl": safe_login_url(v.get("authUrl").and_then(Value::as_str)),
            "verificationUrl": safe_login_url(v.get("verificationUrl").and_then(Value::as_str)),
            "userCode": v.get("userCode").and_then(Value::as_str),
        }))
    }

    pub fn cancel_login(&self, login_id: Option<&str>) -> Result<(), CodexError> {
        let params = login_id.map(|id| json!({ "loginId": id })).unwrap_or_else(|| json!({}));
        self.with_session(|s| s.call("account/login/cancel", params, RPC_TIMEOUT)).map(|_| ())
    }

    pub fn logout(&self) -> Result<(), CodexError> {
        self.with_session(|s| s.call("account/logout", json!({}), RPC_TIMEOUT)).map(|_| ())
    }

    /// Models this account can use, in the OpenAI `{object:"list",data:[…]}` shape.
    pub fn models(&self) -> Result<Value, CodexError> {
        let v = self.with_session(|s| s.call("model/list", json!({}), RPC_TIMEOUT))?;
        Ok(json!({ "object": "list", "data": model_catalog(&v) }))
    }

    /// Run one ephemeral turn and return an OpenAI-shaped chat completion.
    ///
    /// The thread is started with every tool surface refused — no file, command,
    /// browser, MCP or network access — because this is a text completion for a
    /// flow, not an agent session with a workspace.
    pub fn chat(&self, body: &Value) -> Result<Value, CodexError> {
        self.chat_as(body, None)
    }

    /// Run a turn, optionally pinned to a live-call alias.
    ///
    /// When `alias` is set its model/effort/tier WIN over anything in the body.
    /// That is the point of a fixed route: a caller that could talk the
    /// reasoning-off route into a reasoning model would reintroduce exactly the
    /// mid-call latency the alias exists to prevent.
    pub fn chat_as(&self, body: &Value, alias: Option<LiveCallAlias>) -> Result<Value, CodexError> {
        self.chat_streaming(body, alias, |_| {})
    }

    /// [`chat_as`], calling `on_delta` with each fragment as the model produces
    /// it — for a caller that can show the answer arriving rather than waiting
    /// for all of it.
    ///
    /// The callback runs on this blocking thread, so it must not block for long
    /// itself; the tunnel hands fragments to an async sender and returns.
    /// The buffered completion is still returned, so a caller that ignores the
    /// callback behaves exactly as before.
    ///
    /// A request's `tools` are honoured by prompted tool use (see
    /// [`complete_with`]) on the generic route, and on a live-call alias when
    /// the request brings them; while the reply could still be a tool call its
    /// fragments are held back, so a streaming caller never shows the
    /// machinery. The buffered completion is the authority either way.
    pub fn chat_streaming(
        &self,
        body: &Value,
        alias: Option<LiveCallAlias>,
        on_delta: impl FnMut(&str),
    ) -> Result<Value, CodexError> {
        complete_with(body, alias, on_delta, |request, emit| self.run_turn(request, emit))
    }

    /// One turn on the child, once its lane lets it (see [`CodexAgent`]): a
    /// live call's turn waits only for the call's turns before it, and never
    /// for anything else.
    fn run_turn(&self, request: &TurnRequest, on_delta: &mut dyn FnMut(&str)) -> Result<String, CodexError> {
        let lane = if request.live_call { &self.call_lane } else { &self.turn_lane };
        let _in_lane = lane.enter();
        self.with_session(|s| s.turn(request, on_delta))
    }
}

/// What the Agent is told its tools are, ahead of the catalogue.
const TOOLS_INTRO: &str = "You can use these tools in this conversation.";

/// One turn's worth of work, decided before the child is involved.
struct TurnPlan {
    prompt: String,
    model: Option<String>,
    /// The live-call alias the turn is pinned to, if any.
    alias: Option<LiveCallAlias>,
    /// The tools the prompt offers: empty on a plain turn, and on a live-call
    /// alias whose request brings none.
    tools: Vec<chat_tools::Tool>,
}

/// What one turn sends the child: `thread/start`'s params, and `turn/start`'s
/// (the thread id is added once the thread exists).
pub(super) struct TurnRequest {
    pub(super) thread: Value,
    pub(super) turn: Value,
    /// A live-call alias's turn, which takes the call lane.
    pub(super) live_call: bool,
}

impl TurnRequest {
    fn for_plan(plan: &TurnPlan) -> Self {
        // Refuse everything a turn could otherwise reach.
        let mut thread = json!({
            "approvalPolicy": "never",
            "dynamicTools": [],
            "environments": [],
            "runtimeWorkspaceRoots": [],
            "allowProviderModelFallback": false,
        });
        if let Some(m) = &plan.model {
            thread["model"] = Value::String(m.clone());
        }
        // `input` is a SEQUENCE of typed items, not one map. A map is
        // refused with "invalid type: map, expected a sequence" — a runtime
        // message with nothing in it naming this line.
        let mut turn = json!({ "input": [{ "type": "text", "text": plan.prompt }] });
        if let Some(a) = plan.alias {
            // Named on the TURN as well as the thread: the effort is a
            // per-turn setting, and a thread-level model alone would leave
            // it at the account default.
            turn["model"] = Value::String(a.model().to_string());
            turn["effort"] = Value::String(a.reasoning_effort().to_string());
            if let Some(tier) = a.service_tier() {
                turn["serviceTier"] = Value::String(tier.to_string());
            }
        }
        Self { thread, turn, live_call: plan.alias.is_some() }
    }
}

/// A whole chat completion, with the turn itself handed in: `run` takes the
/// turn's request (thread and turn params, the prompt among them) and returns
/// the agent's message, passing fragments to the callback as they arrive. In
/// production that is the child; in the tests it is a fake codex. Everything
/// around the turn is here — what the prompt says, which tools it offers, the
/// model and effort it pins, and whether the reply is an answer or a tool call
/// — so the tests exercise exactly what ships.
///
/// Tools are honoured by PROMPTED tool use, as the tunnel does for FormLogic's
/// chat: a Codex turn has no place to put a schema, so the catalogue is taught
/// in [`chat_tools::preamble_with_intro`] and the reply read back with
/// [`chat_tools::parse_prompted`] — one well-formed fenced call, or the answer.
pub(super) fn complete_with(
    body: &Value,
    alias: Option<LiveCallAlias>,
    mut on_delta: impl FnMut(&str),
    run: impl FnOnce(&TurnRequest, &mut dyn FnMut(&str)) -> Result<String, CodexError>,
) -> Result<Value, CodexError> {
    let plan = plan_turn(body, alias)?;
    let mut held = HeldDeltas::new(!plan.tools.is_empty());
    let text = run(&TurnRequest::for_plan(&plan), &mut |d: &str| held.pass(d, &mut on_delta))?;
    Ok(finish_turn(&plan, text))
}

/// Whether a request brings tools: a non-empty `tools` array.
///
/// What decides how a live-call alias answers. Aokie, which the aliases exist
/// for, never sends tools, and gets exactly what it always got; an agent loop
/// taking a call sends them, and gets prompted tools and a stream.
pub(super) fn brings_tools(body: &Value) -> bool {
    body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty())
}

/// The prompt, model and tools for one request.
///
/// A live-call alias whose request brings no tools keeps exactly the prompt it
/// always had: its callers depend on what it does now. Everything else — the
/// generic route, and an alias an agent loop sends tools to — adds the tool
/// traffic of an agent loop, and mirrors the tunnel's switch: tools offered,
/// the preamble goes first; no tools but results already in the conversation,
/// the instruction to answer from them; neither, the plain prompt unchanged.
/// An alias pins its model (and its effort, on the turn) either way.
fn plan_turn(body: &Value, alias: Option<LiveCallAlias>) -> Result<TurnPlan, CodexError> {
    let empty = || CodexError::Rpc("the request carried no message content".into());
    if let Some(a) = alias.filter(|_| !brings_tools(body)) {
        let prompt = flatten_prompt(body);
        if prompt.trim().is_empty() {
            return Err(empty());
        }
        return Ok(TurnPlan { prompt, model: Some(a.model().to_string()), alias: Some(a), tools: Vec::new() });
    }
    let model = match alias {
        Some(a) => Some(a.model().to_string()),
        None => body.get("model").and_then(Value::as_str).map(str::to_string),
    };
    let conversation = render_messages(body);
    if conversation.trim().is_empty() {
        return Err(empty());
    }
    let tools = offered_tools(body);
    let prompt = if !tools.is_empty() {
        format!("{}\n\n{conversation}", chat_tools::preamble_with_intro(TOOLS_INTRO, &tools))
    } else if carries_tool_traffic(body) {
        format!("{}\n\n{conversation}", chat_tools::plain_answer_instruction())
    } else {
        conversation
    };
    Ok(TurnPlan { prompt, model, alias, tools })
}

/// The request's OpenAI `tools`, as the catalogue the preamble teaches.
///
/// `tool_choice: "none"` offers none. Any other choice is taken as `auto`: a
/// prompted model cannot be made to call a tool, only offered one. Entries of a
/// type other than `function` are skipped rather than taught wrongly.
fn offered_tools(body: &Value) -> Vec<chat_tools::Tool> {
    if body.get("tool_choice").and_then(Value::as_str) == Some("none") {
        return Vec::new();
    }
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    tools
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str).map_or(true, |ty| ty == "function"))
        .filter_map(|t| {
            let f = t.get("function")?;
            let name = f.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty())?;
            Some(chat_tools::Tool {
                name: name.to_string(),
                description: f.get("description").and_then(Value::as_str).unwrap_or_default().to_string(),
                input_schema: f
                    .get("parameters")
                    .filter(|p| p.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" })),
            })
        })
        .collect()
}

/// Whether the conversation already holds tool calls or their results.
fn carries_tool_traffic(body: &Value) -> bool {
    body.get("messages").and_then(Value::as_array).is_some_and(|messages| {
        messages.iter().any(|m| {
            m.get("role").and_then(Value::as_str) == Some("tool")
                || m.get("tool_calls").and_then(Value::as_array).is_some_and(|c| !c.is_empty())
        })
    })
}

/// The completion for a turn's reply: a tool call when the reply is exactly
/// the taught fenced block naming an offered tool with an object for its input,
/// and otherwise the answer, as it has always been.
///
/// The call goes out in the OpenAI shape an agent loop reads: an id of its own
/// (a prompted reply carries none), the arguments as a JSON STRING, no content,
/// and `finish_reason: "tool_calls"`.
fn finish_turn(plan: &TurnPlan, text: String) -> Value {
    let call = if plan.tools.is_empty() {
        None
    } else {
        chat_tools::parse_prompted(&text, &plan.tools).filter(|c| c.input.is_object())
    };
    let (message, finish_reason) = match call {
        Some(call) => (
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                    "type": "function",
                    "function": { "name": call.name, "arguments": call.input.to_string() },
                }],
            }),
            "tool_calls",
        ),
        None => (json!({ "role": "assistant", "content": text }), "stop"),
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    json!({
        "object": "chat.completion",
        "created": created,
        "model": plan.model.clone().unwrap_or_else(|| CODEX_PROVIDER_ID.to_string()),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
    })
}

/// Passes a reply's fragments on, except while they could still be the start
/// of a prompted tool call. A call's fenced block shown as it is written would
/// show the reader the machinery instead of the result; once the reply can
/// only be an answer, everything held is released and the rest flows. What
/// was held and never released is still in the buffered completion.
struct HeldDeltas {
    holding: bool,
    held: String,
}

impl HeldDeltas {
    fn new(tools_offered: bool) -> Self {
        Self { holding: tools_offered, held: String::new() }
    }

    fn pass(&mut self, delta: &str, out: &mut impl FnMut(&str)) {
        if !self.holding {
            out(delta);
            return;
        }
        self.held.push_str(delta);
        if !chat_tools::may_be_prompted_call(&self.held) {
            self.holding = false;
            out(&std::mem::take(&mut self.held));
        }
    }
}

/// The OpenAI-shaped `data` rows for a `model/list` result.
///
/// Two things this gets right that are easy to get wrong, and that both fail as
/// an EMPTY dropdown rather than as an error:
///
/// - the page is at `data`. Reading `items`/`models` found nothing, so the
///   catalogue came back empty from a call that succeeded.
/// - the usable name is `model` (`gpt-5.5`), not the catalogue's own `id`. A
///   client sends this string back as the model to run, and the catalogue id is
///   not one Codex accepts.
///
/// Only the first page: `nextCursor` is not followed, so a very large account
/// catalogue would be truncated rather than paged.
fn model_catalog(result: &Value) -> Vec<Value> {
    let entries = result
        .get("data")
        .or_else(|| result.get("items"))
        .or_else(|| result.get("models"))
        .and_then(Value::as_array);
    let Some(entries) = entries else {
        return Vec::new();
    };
    let mut seen = std::collections::HashSet::new();
    entries
        .iter()
        .filter_map(|m| {
            let id = m
                .get("model")
                .or_else(|| m.get("id"))
                .or_else(|| m.get("slug"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && s.len() <= 256)?;
            if !seen.insert(id.to_string()) {
                return None;
            }
            let mut row = json!({ "id": id, "object": "model" });
            if let Some(name) = m
                .get("displayName")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                row["displayName"] = json!(name);
            }
            // The model a turn that names none runs on: what "ChatGPT's own
            // default" means to a picker offering the choice.
            if m.get("isDefault").and_then(Value::as_bool) == Some(true) {
                row["isDefault"] = json!(true);
            }
            Some(row)
        })
        .collect()
}

/// Fold one batch of turn notifications into the answer. Returns whether the
/// turn finished.
///
/// Each note must be folded EXACTLY ONCE — the turn takes each off its
/// thread's channel for that reason. Folding an overlapping range appends every
/// delta again, and the duplicate reads as the model repeating itself: a poll
/// that re-read from a fixed mark once turned "OAIY tunnel works." into
/// "OAIYOAIY tunnel worksOAIY tunnel works.".
fn fold_turn_notes(out: &mut String, notes: &[Value]) -> bool {
    let mut done = false;
    for n in notes {
        let method = n.get("method").and_then(Value::as_str).unwrap_or("");
        let p = n.get("params").unwrap_or(&Value::Null);
        match method {
            "item/agentMessage/delta" => {
                if let Some(d) = p.get("delta").and_then(Value::as_str) {
                    out.push_str(d);
                }
            }
            // The whole message, for a runtime that sends no deltas. Only when
            // nothing streamed, or it would be appended after the deltas that
            // already spelled it out.
            "item/completed" => {
                if out.is_empty() {
                    if let Some(t) = p.pointer("/item/text").and_then(Value::as_str) {
                        out.push_str(t);
                    }
                }
            }
            "turn/completed" => done = true,
            _ => {}
        }
    }
    done
}

/// The thread id out of a `thread/start` result.
///
/// It arrives NESTED, as `{"thread": {"id": …}}`. Reading a flat `threadId`
/// first was a real bug: the call SUCCEEDED, the id came back None, and every
/// ChatGPT turn failed with "codex did not return a thread id" — which reads as
/// a fault in Codex rather than in how its answer was parsed. The flat
/// spellings are kept after it, as tolerance for another runtime version.
fn thread_id_of(result: &Value) -> Option<String> {
    result
        .pointer("/thread/id")
        .or_else(|| result.get("threadId"))
        .or_else(|| result.get("id"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// Collapse an OpenAI `messages` array into the single prompt a turn takes,
/// keeping role labels so a system instruction still reads as one.
///
/// A live-call alias's prompt when its request brings no tools, unchanged.
/// Every other turn renders with [`render_messages`], which is this plus the
/// tool traffic of an agent loop.
fn flatten_prompt(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = content_text(m.get("content"));
        if content.trim().is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        match role {
            "system" => out.push_str(&format!("[instructions]\n{content}")),
            "assistant" => out.push_str(&format!("[assistant]\n{content}")),
            _ => out.push_str(&content),
        }
    }
    out
}

/// A message's text: a plain string, or the text parts of the array form SDKs
/// emit.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// The generic route's prompt: [`flatten_prompt`]'s rendering, plus what an
/// agent loop carries between turns — each call the model made, written as the
/// very fenced block the preamble teaches, and each result as the
/// `tool_result` line it is told to expect.
///
/// Without them a model that asked for a tool would never learn what came
/// back, and would ask again. A conversation with neither renders exactly as
/// [`flatten_prompt`] does, so the tunnel's tool-free turns are unchanged.
fn render_messages(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    // A `tool` message names its call by id; the name is on the call.
    let mut names: HashMap<String, String> = HashMap::new();
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = content_text(m.get("content"));
        let block = match role {
            "assistant" => {
                let mut parts: Vec<String> = Vec::new();
                if !content.trim().is_empty() {
                    parts.push(content);
                }
                for call in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                    let Some(name) = call.pointer("/function/name").and_then(Value::as_str) else {
                        continue;
                    };
                    if let Some(id) = call.get("id").and_then(Value::as_str) {
                        names.insert(id.to_string(), name.to_string());
                    }
                    parts.push(chat_tools::prompted_call_block(name, &arguments_of(call)));
                }
                if parts.is_empty() {
                    continue;
                }
                format!("[assistant]\n{}", parts.join("\n\n"))
            }
            // Always kept, even empty: a tool that returned nothing still ran.
            "tool" => {
                let name = m
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .and_then(|id| names.get(id))
                    .map(String::as_str)
                    .or_else(|| m.get("name").and_then(Value::as_str))
                    .unwrap_or("tool");
                chat_tools::tool_result_line(name, content)
            }
            _ if content.trim().is_empty() => continue,
            "system" => format!("[instructions]\n{content}"),
            _ => content,
        };
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&block);
    }
    out
}

/// A call's arguments as a value: OpenAI sends them as a JSON STRING. One that
/// does not parse is kept as the text it was, so the model sees what it wrote.
fn arguments_of(call: &Value) -> Value {
    match call.pointer("/function/arguments") {
        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        Some(v) if !v.is_null() => v.clone(),
        _ => json!({}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_catalogue_is_read_from_the_page_codex_returns() {
        // Live shape. Reading `items`/`models` found nothing, so the catalogue
        // came back EMPTY from a call that succeeded — an empty dropdown with
        // no error anywhere.
        let real = json!({
            "data": [
                { "id": "cat-1", "model": "gpt-5.5", "displayName": "GPT-5.5", "isDefault": true },
                { "id": "cat-2", "model": "gpt-5.6-luna", "displayName": "Luna" },
                { "id": "cat-3", "model": "gpt-5.5" },
                { "id": "legacy-shape" },
                { "displayName": "nameless" },
                {}
            ],
            "nextCursor": null,
        });
        let rows = model_catalog(&real);
        // The usable name is `model`, not the catalogue's own id: a client
        // sends this string back as the model to run.
        assert_eq!(rows[0]["id"], "gpt-5.5");
        assert_eq!(rows[0]["displayName"], "GPT-5.5");
        // The model a turn naming none runs on, for a picker offering "ChatGPT's own default".
        assert_eq!(rows[0]["isDefault"], true);
        assert!(rows[1].get("isDefault").is_none(), "only the default is marked: {rows:?}");
        assert_eq!(rows[1]["id"], "gpt-5.6-luna");
        // `id` is the fallback for a runtime that names models differently…
        assert_eq!(rows[2]["id"], "legacy-shape");
        // …and an entry with no usable name at all is skipped, not blank.
        assert_eq!(rows.len(), 3, "one duplicate and two nameless dropped: {rows:?}");
        assert!(rows.iter().all(|r| r["id"].as_str().is_some_and(|s| !s.is_empty())));
        // A body with no page at all is an empty catalogue, not a panic.
        assert!(model_catalog(&json!({ "oops": true })).is_empty());
    }

    #[test]
    fn a_streamed_answer_is_assembled_once_not_once_per_poll() {
        // The bug this pins, seen live: the poll loop re-read the notification
        // list from a FIXED mark every 50 ms, so "OAIY tunnel works." came back
        // as "OAIYOAIY tunnel worksOAIY tunnel works." — which reads as the
        // model repeating itself, not as a cursor that never advanced.
        let delta = |t: &str| json!({ "method": "item/agentMessage/delta", "params": { "delta": t } });
        let notes = vec![delta("OAIY"), delta(" tunnel"), delta(" works.")];

        // Folded in batches, as the loop does, each note exactly once.
        let mut out = String::new();
        let mut cursor = 0usize;
        let mut done = false;
        while cursor < notes.len() {
            let batch = &notes[cursor..(cursor + 2).min(notes.len())];
            cursor += batch.len();
            done = fold_turn_notes(&mut out, batch) || done;
        }
        assert_eq!(out, "OAIY tunnel works.");
        assert!(!done, "no turn/completed arrived");

        // …and the shape of the old bug: an overlapping range duplicates.
        let mut doubled = String::new();
        fold_turn_notes(&mut doubled, &notes);
        fold_turn_notes(&mut doubled, &notes);
        assert_eq!(doubled, "OAIY tunnel works.OAIY tunnel works.");
    }

    #[test]
    fn a_runtime_that_sends_no_deltas_still_yields_the_message_once() {
        let completed = json!({
            "method": "item/completed",
            "params": { "item": { "text": "the whole answer" } },
        });
        let mut out = String::new();
        let done = fold_turn_notes(
            &mut out,
            &[completed.clone(), json!({ "method": "turn/completed" })],
        );
        assert_eq!(out, "the whole answer");
        assert!(done);

        // But it must NOT be appended after deltas that already spelled it out.
        let mut streamed = String::from("the whole answer");
        fold_turn_notes(&mut streamed, &[completed]);
        assert_eq!(streamed, "the whole answer");
    }

    #[test]
    fn a_turn_carries_its_prompt_as_a_sequence_of_typed_items() {
        // Codex refuses a bare map with "invalid type: map, expected a
        // sequence" — a runtime message that names nothing in this file. The
        // shape is pinned here so a rewrite cannot quietly go back to a map.
        let plan = plan_turn(&json!({ "messages": [{ "role": "user", "content": "hello" }] }), None).unwrap();
        let request = TurnRequest::for_plan(&plan);
        let input = request.turn["input"].as_array().expect("input must be a sequence");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "text");
        assert_eq!(input[0]["text"], "hello");
        // The generic route leaves the model and effort to the thread and the account.
        assert!(request.turn.get("effort").is_none() && request.turn.get("model").is_none());
        assert_eq!(request.thread["approvalPolicy"], "never");
        assert_eq!(request.thread["dynamicTools"], json!([]));
        assert_eq!(request.thread["runtimeWorkspaceRoots"], json!([]));
        // …and takes the lane of ordinary turns; every alias, tools or none, takes the call lane.
        assert!(!request.live_call);
        let with_tools = json!({ "messages": [{ "role": "user", "content": "hello" }], "tools": weather_tools() });
        for alias in LiveCallAlias::all() {
            for body in [json!({ "messages": [{ "role": "user", "content": "hello" }] }), with_tools.clone()] {
                assert!(TurnRequest::for_plan(&plan_turn(&body, Some(alias)).unwrap()).live_call, "{alias:?}");
            }
        }
        assert!(!TurnRequest::for_plan(&plan_turn(&with_tools, None).unwrap()).live_call);
    }

    #[test]
    fn the_thread_id_is_read_from_where_codex_actually_puts_it() {
        // The shape a live `thread/start` returns. Reading a flat `threadId`
        // first made every ChatGPT turn fail with "codex did not return a
        // thread id" — a message that blames Codex for a parsing mistake.
        let real = json!({
            "thread": { "id": "01996b2e-0000-7000-8000-0123456789ab" },
            "cwd": "C:/Users/User/AppData/Roaming/com.oaiy/codex",
            "approvalPolicy": "never",
            "model": "gpt-5.5",
            "runtimeWorkspaceRoots": [],
        });
        assert_eq!(
            thread_id_of(&real).as_deref(),
            Some("01996b2e-0000-7000-8000-0123456789ab")
        );
        // Tolerated alternatives, and the shapes that must yield nothing rather
        // than a bogus id that then fails at turn/start instead.
        assert_eq!(thread_id_of(&json!({ "threadId": "t1" })).as_deref(), Some("t1"));
        assert_eq!(thread_id_of(&json!({ "id": "t2" })).as_deref(), Some("t2"));
        assert!(thread_id_of(&json!({ "thread": { "id": "" } })).is_none());
        assert!(thread_id_of(&json!({ "thread": {} })).is_none());
        assert!(thread_id_of(&json!({})).is_none());
    }

    #[test]
    fn only_https_openai_login_urls_are_accepted() {
        assert!(safe_login_url(Some("https://auth.openai.com/x?y=1")).is_some());
        assert!(safe_login_url(Some("https://chatgpt.com/device")).is_some());
        assert!(safe_login_url(Some("http://auth.openai.com/x")).is_none(), "http rejected");
        assert!(safe_login_url(Some("https://evil.example/x")).is_none(), "unknown host rejected");
        assert!(safe_login_url(Some("https://u:p@auth.openai.com/x")).is_none(), "creds rejected");
        assert!(safe_login_url(None).is_none());
    }

    #[test]
    fn prompt_flattening_keeps_system_and_array_content() {
        let body = json!({ "messages": [
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": [{ "type": "text", "text": "2+2?" }] },
        ]});
        let p = flatten_prompt(&body);
        assert!(p.contains("[instructions]"), "{p}");
        assert!(p.contains("be terse"));
        assert!(p.contains("2+2?"), "array content must not be dropped: {p}");
    }

    #[test]
    fn empty_or_contentless_bodies_flatten_to_nothing() {
        assert_eq!(flatten_prompt(&json!({})), "");
        assert_eq!(flatten_prompt(&json!({ "messages": [{ "role": "user", "content": "" }] })), "");
    }

    #[test]
    fn the_environment_allow_list_excludes_credentials() {
        let allow = env_allow_list();
        assert!(!allow.contains(&"OPENAI_API_KEY"), "an API key must never reach the child");
        assert!(!allow.iter().any(|k| k.to_ascii_lowercase().contains("proxy")));
    }

    #[test]
    fn the_live_call_aliases_match_the_ids_the_plugin_recognises() {
        // The Aokie plugin matches a live-call route BY URL, against these exact
        // ids. A rename here silently stops it applying the live-call policy —
        // it would treat the route as an ordinary local provider.
        assert_eq!(LiveCallAlias::ReasoningNone.id(), "openai-codex-agent-none");
        assert_eq!(LiveCallAlias::ReasoningLow.id(), "openai-codex-agent-low");
        assert_eq!(LiveCallAlias::LunaReasoningLow.id(), "openai-codex-agent-luna-low");
        assert_eq!(
            LiveCallAlias::LunaReasoningLowFast.id(),
            "openai-codex-agent-luna-low-fast"
        );
        // …and none of them collides with the generic route, which has its own
        // behaviour (caller-chosen model, no forced effort).
        assert!(LiveCallAlias::from_id(CODEX_PROVIDER_ID).is_none());
        assert!(LiveCallAlias::from_id("openai-codex-agent-nonesuch").is_none());
        for alias in LiveCallAlias::all() {
            assert_eq!(LiveCallAlias::from_id(alias.id()), Some(alias));
        }
    }

    #[test]
    fn every_alias_pins_a_model_and_an_effort() {
        // The whole reason these exist: a call cannot wait on a reasoning model,
        // so neither field may be left to the account default.
        for alias in LiveCallAlias::all() {
            assert!(!alias.model().is_empty(), "{:?}", alias);
            assert!(
                matches!(alias.reasoning_effort(), "none" | "low"),
                "{:?} pins {:?}, which is not a live-call effort",
                alias,
                alias.reasoning_effort()
            );
        }
        assert_eq!(LiveCallAlias::ReasoningNone.reasoning_effort(), "none");
        assert_eq!(LiveCallAlias::ReasoningNone.model(), LIVE_CALL_MODEL);
        assert_eq!(LiveCallAlias::LunaReasoningLow.model(), LUNA_LIVE_CALL_MODEL);
    }

    // ---- ChatGPT with tools, against a fake codex ----

    /// The tools an agent loop sends, in the OpenAI shape.
    fn weather_tools() -> Value {
        json!([
            { "type": "function", "function": {
                "name": "get_weather",
                "description": "The weather in a city",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] },
            } },
            { "type": "function", "function": { "name": "end_call", "description": "Hang up" } },
        ])
    }

    /// What one turn against a fake codex produced: the completion, what the
    /// fake was sent (the prompt, the thread's model, the turn's params), and
    /// what a streaming caller was shown.
    struct Faked {
        completion: Value,
        prompt: String,
        model: Option<String>,
        turn: Value,
        streamed: String,
    }

    /// A fake codex: answers the turn with `reply`, cut into small fragments
    /// and folded through the same notification handling as the real child,
    /// batch by batch, so a streaming caller sees what it would see live.
    fn fake_codex(body: &Value, alias: Option<LiveCallAlias>, reply: &str) -> Faked {
        let mut prompt = String::new();
        let mut model = None;
        let mut turn = Value::Null;
        let mut streamed = String::new();
        let completion = complete_with(body, alias, |d| streamed.push_str(d), |request, emit| {
            prompt = request.turn.pointer("/input/0/text").and_then(Value::as_str).unwrap_or_default().to_string();
            model = request.thread.get("model").and_then(Value::as_str).map(str::to_string);
            turn = request.turn.clone();
            let chars: Vec<char> = reply.chars().collect();
            let mut notes: Vec<Value> = chars
                .chunks(5)
                .map(|c| json!({ "method": "item/agentMessage/delta", "params": { "delta": c.iter().collect::<String>() } }))
                .collect();
            notes.push(json!({ "method": "turn/completed" }));
            let mut out = String::new();
            for batch in notes.chunks(2) {
                let before = out.len();
                fold_turn_notes(&mut out, batch);
                if out.len() > before {
                    emit(&out[before..]);
                }
            }
            Ok(out)
        })
        .expect("the fake turn answers");
        Faked { completion, prompt, model, turn, streamed }
    }

    fn ask(tools: Value) -> Value {
        json!({
            "model": "gpt-5.5",
            "stream": true,
            "messages": [
                { "role": "system", "content": "You are the front desk." },
                { "role": "user", "content": "What is the weather in Perth?" },
            ],
            "tools": tools,
        })
    }

    #[test]
    fn with_tools_offered_a_plain_answer_is_the_answer_and_streams() {
        let reply = "It is sunny in Perth today.";
        let f = fake_codex(&ask(weather_tools()), None, reply);
        let choice = &f.completion["choices"][0];
        assert_eq!(choice["finish_reason"], "stop");
        assert_eq!(choice["message"]["content"], reply);
        assert!(choice["message"].get("tool_calls").is_none());
        assert_eq!(f.completion["model"], "gpt-5.5");
        assert_eq!(f.model.as_deref(), Some("gpt-5.5"));
        // It could not be a call once it began "It is", so it streamed — all of it.
        assert_eq!(f.streamed, reply);
        // The preamble went first, teaching the catalogue and the one fence the parser reads.
        assert!(f.prompt.starts_with(TOOLS_INTRO), "{}", f.prompt);
        assert!(f.prompt.contains("- get_weather: The weather in a city"), "{}", f.prompt);
        assert!(f.prompt.contains("\"required\":[\"city\"]"), "the schema is taught: {}", f.prompt);
        assert!(f.prompt.contains("```tool_call"), "{}", f.prompt);
        assert!(f.prompt.contains("[instructions]\nYou are the front desk."), "{}", f.prompt);
        assert!(f.prompt.ends_with("What is the weather in Perth?"), "{}", f.prompt);
    }

    #[test]
    fn a_fenced_reply_comes_back_as_an_openai_tool_call() {
        let reply = "```tool_call\n{\"tool\":\"get_weather\",\"input\":{\"city\":\"Perth\"}}\n```";
        let f = fake_codex(&ask(weather_tools()), None, reply);
        let choice = &f.completion["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        let message = &choice["message"];
        assert_eq!(message["role"], "assistant");
        assert!(message["content"].is_null(), "a call carries no text: {message}");
        let calls = message["tool_calls"].as_array().expect("tool_calls");
        assert_eq!(calls.len(), 1);
        let id = calls[0]["id"].as_str().unwrap();
        assert!(id.starts_with("call_") && id.len() > "call_".len(), "{id}");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // Arguments go out as a JSON STRING, as every OpenAI client parses them.
        let args = calls[0]["function"]["arguments"].as_str().expect("arguments is a string");
        assert_eq!(serde_json::from_str::<Value>(args).unwrap(), json!({ "city": "Perth" }));
        // …and nothing of the fenced block reached a streaming caller.
        assert_eq!(f.streamed, "", "the machinery must not be shown");
        // Each call gets an id of its own.
        let again = fake_codex(&ask(weather_tools()), None, reply);
        assert_ne!(again.completion["choices"][0]["message"]["tool_calls"][0]["id"], calls[0]["id"]);
        // A call with no input is a call with an empty object.
        let bare = fake_codex(&ask(weather_tools()), None, "```tool_call\n{\"tool\":\"end_call\"}\n```");
        assert_eq!(bare.completion["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{}");
    }

    #[test]
    fn a_malformed_call_falls_back_to_the_text() {
        for reply in [
            "```tool_call\n{\"tool\":\"get_weather\",\"input\":{\"city\":}\n```", // not JSON
            "```tool_call\n{\"tool\":\"delete_everything\",\"input\":{}}\n```",  // not offered
            "```tool_call\n{\"tool\":\"get_weather\",\"input\":\"Perth\"}\n```",  // input not an object
            "```tool_call\n{\"tool\":\"get_weather\",\"input\":{\"city\":\"Perth\"}}", // never closed
        ] {
            let f = fake_codex(&ask(weather_tools()), None, reply);
            let choice = &f.completion["choices"][0];
            assert_eq!(choice["finish_reason"], "stop", "{reply}");
            assert!(choice["message"].get("tool_calls").is_none(), "{reply}");
            // The whole reply is the answer: held back while it looked like a
            // call, it is in the completion for the caller to show.
            assert_eq!(choice["message"]["content"], reply);
            assert_eq!(f.streamed, "", "{reply}");
        }
        // Prose before the fence is an answer from its first word, so it streams.
        let prose = "Sure! ```tool_call\n{\"tool\":\"get_weather\",\"input\":{}}\n```";
        let f = fake_codex(&ask(weather_tools()), None, prose);
        assert_eq!(f.completion["choices"][0]["finish_reason"], "stop");
        assert_eq!(f.streamed, prose);
    }

    #[test]
    fn a_tool_result_is_carried_into_the_next_turn() {
        let body = json!({
            "messages": [
                { "role": "system", "content": "You are the front desk." },
                { "role": "user", "content": "What is the weather in Perth?" },
                { "role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_abc", "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Perth\"}" },
                }] },
                // No name: it is found by the call's id.
                { "role": "tool", "tool_call_id": "call_abc", "content": "{\"sky\":\"sunny\",\"high\":24}" },
            ],
            "tools": weather_tools(),
        });
        let f = fake_codex(&body, None, "Sunny, with a high of 24.");
        assert_eq!(f.completion["choices"][0]["finish_reason"], "stop");
        assert_eq!(f.completion["choices"][0]["message"]["content"], "Sunny, with a high of 24.");
        // No model named: Codex's own default runs, and the completion says whose it was.
        assert_eq!(f.model, None);
        assert_eq!(f.completion["model"], CODEX_PROVIDER_ID);

        // The model sees what it called, in the form it was taught…
        let call = "[assistant]\n```tool_call\n{\"tool\":\"get_weather\",\"input\":{\"city\":\"Perth\"}}\n```";
        let call_at = f.prompt.find(call).unwrap_or_else(|| panic!("the earlier call is replayed: {}", f.prompt));
        // …and what came back, as the line the preamble told it to expect, after the call.
        let result = "tool_result get_weather: {\"sky\":\"sunny\",\"high\":24}";
        let result_at = f.prompt.find(result).unwrap_or_else(|| panic!("the result is carried: {}", f.prompt));
        assert!(call_at < result_at);
        assert!(f.prompt.ends_with(result), "the result is the latest thing it reads: {}", f.prompt);
        // A replayed call parses back as the call it was, so the history agrees with the preamble.
        let replayed = &call["[assistant]\n".len()..];
        let tools = offered_tools(&body);
        assert_eq!(chat_tools::parse_prompted(replayed, &tools).map(|c| c.input), Some(json!({ "city": "Perth" })));

        // A second call, and its result, in a longer loop: each one is carried, in order.
        let mut longer = body.clone();
        let messages = longer["messages"].as_array_mut().unwrap();
        messages.push(json!({ "role": "assistant", "content": "Checking tomorrow too.", "tool_calls": [{
            "id": "call_def", "type": "function",
            "function": { "name": "get_weather", "arguments": "{\"city\":\"Perth\",\"day\":\"tomorrow\"}" },
        }] }));
        messages.push(json!({ "role": "tool", "tool_call_id": "call_def", "content": "rain" }));
        let f = fake_codex(&longer, None, "Sunny today, rain tomorrow.");
        let first = f.prompt.find("tool_result get_weather: {\"sky\"").unwrap();
        let said = f.prompt.find("[assistant]\nChecking tomorrow too.\n\n```tool_call").unwrap();
        let second = f.prompt.find("tool_result get_weather: rain").unwrap();
        assert!(first < said && said < second, "{}", f.prompt);
    }

    #[test]
    fn with_tools_withdrawn_the_model_is_asked_to_answer_from_the_results() {
        // The tunnel's switch: no tools this turn, but results already in the
        // conversation — the instruction to answer from them, not the preamble.
        let mut body = ask(weather_tools());
        body["tool_choice"] = json!("none");
        body["messages"].as_array_mut().unwrap().push(json!({ "role": "tool", "name": "get_weather", "content": "sunny" }));
        let plan = plan_turn(&body, None).unwrap();
        assert!(plan.tools.is_empty());
        assert!(plan.prompt.starts_with(chat_tools::plain_answer_instruction()), "{}", plan.prompt);
        assert!(plan.prompt.ends_with("tool_result get_weather: sunny"), "{}", plan.prompt);
        // A fenced reply with no tools offered is just text.
        let f = fake_codex(&body, None, "```tool_call\n{\"tool\":\"get_weather\",\"input\":{}}\n```");
        assert_eq!(f.completion["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn only_function_tools_with_a_name_are_taught() {
        let body = json!({ "tools": [
            { "type": "function", "function": { "name": "a" } },
            { "function": { "name": "no_type", "parameters": "not a schema" } },
            { "type": "custom", "custom": { "name": "grammar_tool" } },
            { "type": "function", "function": { "name": "  " } },
            { "type": "function" },
        ]});
        let tools = offered_tools(&body);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a", "no_type"]);
        assert_eq!(tools[1].input_schema, json!({ "type": "object" }), "a bad schema is replaced, not taught");
    }

    #[test]
    fn a_conversation_without_tools_renders_exactly_as_before() {
        // The tunnel hands tool-free bodies to this route's code (its plain
        // chat, and its own prompted loop as one user message); a drift here
        // would change what FormLogic's chat sends ChatGPT.
        for body in [
            json!({ "messages": [
                { "role": "system", "content": "be terse" },
                { "role": "user", "content": [{ "type": "text", "text": "2+2?" }] },
                { "role": "assistant", "content": "4" },
                { "role": "user", "content": "" },
                { "role": "user", "content": "and 3+3?" },
            ]}),
            json!({ "model": "gpt-5.5", "messages": [{ "role": "user", "content": "You can use tools…\n\n```tool_call\n{}\n```\n\nuser: hi\n\ntool_result x: 1" }] }),
            json!({ "messages": [{ "role": "developer", "content": "d" }, { "role": "function", "content": "f" }] }),
        ] {
            assert_eq!(render_messages(&body), flatten_prompt(&body), "{body}");
            let plan = plan_turn(&body, None).unwrap();
            assert_eq!(plan.prompt, flatten_prompt(&body));
            assert!(plan.tools.is_empty());
        }
        // …and its completion is the one it always was.
        let f = fake_codex(&json!({ "messages": [{ "role": "user", "content": "hi" }] }), None, "hello");
        assert_eq!(f.completion["choices"][0], json!({ "index": 0, "message": { "role": "assistant", "content": "hello" }, "finish_reason": "stop" }));
        assert_eq!(f.streamed, "hello");
        assert!(plan_turn(&json!({ "messages": [], "tools": weather_tools() }), None).is_err(), "tools alone are no conversation");
    }

    /// The turn params an alias pins, whatever the request says.
    fn assert_pinned(f: &Faked, alias: LiveCallAlias) {
        assert_eq!(f.model.as_deref(), Some(alias.model()), "{alias:?}: the thread's model");
        assert_eq!(f.turn["model"], alias.model(), "{alias:?}: the turn's model");
        assert_eq!(f.turn["effort"], alias.reasoning_effort(), "{alias:?}: the turn's effort");
        assert_eq!(f.turn.get("serviceTier").and_then(Value::as_str), alias.service_tier(), "{alias:?}");
        assert_eq!(f.completion["model"], alias.model(), "{alias:?}");
    }

    #[test]
    fn a_live_call_alias_without_tools_answers_exactly_as_before() {
        // Aokie, which the aliases exist for, never sends tools: it gets the
        // prompt, the pinned model and effort and the plain answer it always
        // had, whatever else the request carries.
        let conversation = json!([
            { "role": "system", "content": "You are the front desk." },
            { "role": "user", "content": "What is the weather in Perth?" },
            { "role": "assistant", "content": null, "tool_calls": [{ "id": "x", "type": "function", "function": { "name": "get_weather", "arguments": "{}" } }] },
            { "role": "tool", "tool_call_id": "x", "content": "sunny" },
        ]);
        // The prompt the aliases have always sent for it, byte for byte.
        let before = "[instructions]\nYou are the front desk.\n\nWhat is the weather in Perth?\n\nsunny";
        for body in [
            json!({ "model": "anything", "stream": true, "messages": conversation }),
            json!({ "stream": false, "messages": conversation, "tools": [] }),
            json!({ "messages": conversation, "tools": null }),
        ] {
            assert!(!brings_tools(&body), "{body}");
            for alias in LiveCallAlias::all() {
                let plan = plan_turn(&body, Some(alias)).unwrap();
                assert_eq!(plan.prompt, before, "{alias:?}");
                assert_eq!(plan.prompt, flatten_prompt(&body));
                assert!(plan.tools.is_empty(), "{alias:?}");
                let reply = "```tool_call\n{\"tool\":\"end_call\",\"input\":{}}\n```";
                let f = fake_codex(&body, Some(alias), reply);
                assert_eq!(f.prompt, before, "{alias:?}: the prompt sent is the one it always was");
                assert_pinned(&f, alias);
                assert_eq!(f.completion["choices"][0], json!({ "index": 0, "message": { "role": "assistant", "content": reply }, "finish_reason": "stop" }));
                assert_eq!(f.streamed, reply, "nothing held back");
            }
        }
    }

    #[test]
    fn a_live_call_alias_with_tools_uses_them_and_keeps_its_model_and_effort() {
        // An agent loop taking a phone call sends tools: prompted tool use as
        // on the generic route, with the alias's model and effort still pinned
        // over whatever model the request names.
        let mut body = ask(weather_tools());
        body["model"] = json!("a-reasoning-model");
        assert!(brings_tools(&body));
        let call = "```tool_call\n{\"tool\":\"get_weather\",\"input\":{\"city\":\"Perth\"}}\n```";
        for alias in LiveCallAlias::all() {
            let f = fake_codex(&body, Some(alias), call);
            assert_pinned(&f, alias);
            assert!(f.prompt.starts_with(TOOLS_INTRO), "{alias:?}: {}", f.prompt);
            let choice = &f.completion["choices"][0];
            assert_eq!(choice["finish_reason"], "tool_calls", "{alias:?}");
            assert_eq!(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather");
            assert_eq!(choice["message"]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Perth\"}");
            assert_eq!(f.streamed, "", "{alias:?}: the call is not shown");

            // …and with the result back, the answer streams as it is written.
            let mut next = body.clone();
            let messages = next["messages"].as_array_mut().unwrap();
            messages.push(json!({ "role": "assistant", "content": null, "tool_calls": choice["message"]["tool_calls"].clone() }));
            messages.push(json!({ "role": "tool", "tool_call_id": choice["message"]["tool_calls"][0]["id"].clone(), "content": "sunny" }));
            let f = fake_codex(&next, Some(alias), "Sunny in Perth.");
            assert_pinned(&f, alias);
            assert!(f.prompt.ends_with("tool_result get_weather: sunny"), "{}", f.prompt);
            assert_eq!(f.completion["choices"][0]["finish_reason"], "stop");
            assert_eq!(f.streamed, "Sunny in Perth.");
        }
    }

    #[test]
    fn only_the_fast_luna_route_asks_for_priority_service() {
        // Requesting priority on every route would spend the account's priority
        // budget on calls that did not ask for it.
        assert_eq!(LiveCallAlias::LunaReasoningLowFast.service_tier(), Some("priority"));
        assert_eq!(LiveCallAlias::ReasoningNone.service_tier(), None);
        assert_eq!(LiveCallAlias::ReasoningLow.service_tier(), None);
        assert_eq!(LiveCallAlias::LunaReasoningLow.service_tier(), None);
    }

    // ---- The connector against a fake app-server, over in-memory pipes ----
    //
    // `fake_codex` above stands in for the whole turn, so it cannot see how the
    // connector shares its child. This fake is the app-server instead: the
    // connector's own session, reader and turn loop talk to it in the same
    // line-delimited JSON-RPC as the real `codex app-server`.

    /// How long a test waits for something that should happen at once.
    const WAIT: Duration = Duration::from_secs(10);

    /// The writing end of an in-memory pipe: what it writes arrives, in order,
    /// at its reader. Dropping it is the end of the file, as a child's closed
    /// stdout is.
    struct PipeWriter(Sender<Vec<u8>>);

    struct PipeReader {
        rx: Receiver<Vec<u8>>,
        buf: Vec<u8>,
        at: usize,
    }

    fn pipe() -> (PipeWriter, PipeReader) {
        let (tx, rx) = channel();
        (PipeWriter(tx), PipeReader { rx, buf: Vec::new(), at: 0 })
    }

    impl Write for PipeWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.send(bytes.to_vec()).map_err(|_| std::io::ErrorKind::BrokenPipe)?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Read for PipeReader {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            while self.at >= self.buf.len() {
                match self.rx.recv() {
                    Ok(bytes) => {
                        self.buf = bytes;
                        self.at = 0;
                    }
                    Err(_) => return Ok(0),
                }
            }
            let n = out.len().min(self.buf.len() - self.at);
            out[..n].copy_from_slice(&self.buf[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    /// The fake's side of the connector's stdout. `None` once it has "died".
    type Wire = Arc<Mutex<Option<PipeWriter>>>;

    fn send(out: &Wire, msg: Value) {
        if let Some(w) = out.lock().unwrap().as_mut() {
            let _ = writeln!(w, "{msg}");
        }
    }

    /// What the fake app-server has seen and been told, shared with the test.
    #[derive(Default)]
    struct FakeState {
        /// App-servers started: one per child the connector would have run.
        connects: usize,
        signed_in: bool,
        login_starts: usize,
        /// Each `turn/start`, in the order it arrived: its prompt, and whether
        /// it was a live-call turn (an alias pins the effort on the turn).
        started: Vec<(String, bool)>,
        /// Each turn that finished, in order: its prompt.
        finished: Vec<String>,
        /// Live-call turns in progress now, and the most there ever were.
        calls_running: usize,
        most_calls_running: usize,
        /// The names a `hold:<name>` turn waits on, once released.
        released: std::collections::HashSet<String>,
        next_thread: u64,
    }

    /// An in-process `codex app-server`: answers what the connector asks, and
    /// runs each turn on a thread of its own, as the real one does.
    ///
    /// A turn answers "You said: <the prompt's last line>", in small fragments.
    /// A prompt carrying `hold:<name>` says its first fragment and then waits
    /// for [`FakeCodex::release`] — a long setup turn, held open.
    #[derive(Default)]
    struct FakeCodex {
        state: Mutex<FakeState>,
        changed: std::sync::Condvar,
        /// The latest app-server's side of the connector's stdout.
        wire: Mutex<Option<Wire>>,
    }

    impl FakeCodex {
        /// A connector whose app-server is this fake.
        fn agent(self: &Arc<Self>) -> CodexHandle {
            let fake = self.clone();
            CodexAgent::with_connect(PathBuf::from("fake-codex-home"), Arc::new(move |_home: &Path| Ok::<_, CodexError>(fake.serve())))
        }

        /// A new app-server, on a new pair of pipes.
        fn serve(self: &Arc<Self>) -> Transport {
            let (to_server, requests) = pipe();
            let (to_client, from_server) = pipe();
            let out: Wire = Arc::new(Mutex::new(Some(to_client)));
            *self.wire.lock().unwrap() = Some(out.clone());
            self.update(|s| s.connects += 1);
            let fake = self.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(requests).lines() {
                    let Ok(line) = line else { break };
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                    fake.answer(&msg, &out);
                }
            });
            Transport { to_server: Box::new(to_server), from_server: Box::new(from_server), child: None }
        }

        fn answer(self: &Arc<Self>, msg: &Value, out: &Wire) {
            // A notification (`initialized`) wants no answer.
            let Some(id) = msg.get("id").cloned() else { return };
            let params = &msg["params"];
            let result = match msg["method"].as_str().unwrap_or_default() {
                "initialize" => json!({ "userAgent": "fake-codex" }),
                "account/read" => {
                    let account = if self.state.lock().unwrap().signed_in {
                        json!({ "type": "chatgpt", "email": "caller@example.com", "planType": "plus" })
                    } else {
                        Value::Null
                    };
                    json!({ "account": account, "requiresOpenaiAuth": true })
                }
                "account/login/start" => {
                    // As if the person finished in the browser at once.
                    self.update(|s| {
                        s.login_starts += 1;
                        s.signed_in = true;
                    });
                    json!({ "type": "chatgpt", "loginId": "login-1", "authUrl": "https://auth.openai.com/oauth/authorize?fake=1" })
                }
                "account/logout" => {
                    self.update(|s| s.signed_in = false);
                    json!({})
                }
                "model/list" => json!({ "data": [{ "id": "m1", "model": "gpt-5.5", "isDefault": true }] }),
                "thread/start" => {
                    let thread = format!("thread-{}", self.update(|s| {
                        s.next_thread += 1;
                        s.next_thread
                    }));
                    // Routed by `thread.id`, not `threadId`, as the real one sends it.
                    send(out, json!({ "method": "thread/started", "params": { "thread": { "id": thread } } }));
                    json!({ "thread": { "id": thread }, "model": params.get("model").cloned().unwrap_or(json!("gpt-5.5")) })
                }
                "turn/start" => {
                    let thread = params["threadId"].as_str().unwrap_or_default().to_string();
                    let prompt = params.pointer("/input/0/text").and_then(Value::as_str).unwrap_or_default().to_string();
                    let call = params.get("effort").is_some();
                    let turn = format!("turn-of-{thread}");
                    self.update(|s| {
                        s.started.push((prompt.clone(), call));
                        if call {
                            s.calls_running += 1;
                            s.most_calls_running = s.most_calls_running.max(s.calls_running);
                        }
                    });
                    // Answered at once; the turn itself runs on.
                    send(out, json!({ "id": id, "result": { "turn": { "id": turn, "status": "inProgress", "items": [] } } }));
                    let (fake, out) = (self.clone(), out.clone());
                    std::thread::spawn(move || fake.run_turn(&out, &thread, &turn, &prompt, call));
                    return;
                }
                other => {
                    send(out, json!({ "id": id, "error": { "code": -32601, "message": format!("the fake has no {other}") } }));
                    return;
                }
            };
            send(out, json!({ "id": id, "result": result }));
        }

        fn run_turn(&self, out: &Wire, thread: &str, turn: &str, prompt: &str, call: bool) {
            let reply = format!("You said: {}", prompt.lines().last().unwrap_or_default());
            let chars: Vec<char> = reply.chars().collect();
            let fragments: Vec<String> = chars.chunks(4).map(|c| c.iter().collect()).collect();
            let hold = prompt
                .split_whitespace()
                .find_map(|w| w.strip_prefix("hold:"))
                .map(str::to_string);
            // Something no turn reads, for no thread in particular.
            send(out, json!({ "method": "account/rateLimits/updated", "params": { "rateLimits": {} } }));
            send(out, json!({ "method": "turn/started", "params": { "threadId": thread, "turn": { "id": turn, "status": "inProgress" } } }));
            for (i, fragment) in fragments.iter().enumerate() {
                send(out, json!({ "method": "item/agentMessage/delta", "params": {
                    "threadId": thread, "turnId": turn, "itemId": "msg", "delta": fragment,
                } }));
                if let (0, Some(name)) = (i, &hold) {
                    self.wait_for(|s| s.released.contains(name), Duration::from_secs(60));
                }
            }
            send(out, json!({ "method": "item/completed", "params": {
                "threadId": thread, "turnId": turn, "item": { "type": "agentMessage", "id": "msg", "text": reply },
            } }));
            self.update(|s| {
                s.finished.push(prompt.to_string());
                if call {
                    s.calls_running -= 1;
                }
            });
            send(out, json!({ "method": "turn/completed", "params": { "threadId": thread, "turn": { "id": turn, "status": "completed" } } }));
        }

        fn update<T>(&self, f: impl FnOnce(&mut FakeState) -> T) -> T {
            let t = f(&mut self.state.lock().unwrap());
            self.changed.notify_all();
            t
        }

        /// Waits until `until` holds, for at most `within`. Whether it did.
        fn wait_for(&self, until: impl Fn(&FakeState) -> bool, within: Duration) -> bool {
            let deadline = Instant::now() + within;
            let mut s = self.state.lock().unwrap();
            while !until(&s) {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                s = self.changed.wait_timeout(s, left).unwrap().0;
            }
            true
        }

        fn release(&self, name: &str) {
            self.update(|s| {
                s.released.insert(name.to_string());
            });
        }

        /// The latest app-server dies: its stdout closes, mid-turn or not.
        fn die(&self) {
            if let Some(wire) = self.wire.lock().unwrap().take() {
                wire.lock().unwrap().take();
            }
        }
    }

    /// Waits until `until` holds, for at most [`WAIT`]. Whether it did.
    fn eventually(until: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + WAIT;
        while !until() {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    /// `agent` answering `text` on a thread of its own, as the routes answer an
    /// HTTP request; the answer (or the error's message) arrives on the receiver.
    fn ask_in_background(agent: &CodexHandle, text: &str, alias: Option<LiveCallAlias>) -> Receiver<Result<String, String>> {
        let (tx, rx) = channel();
        let agent = agent.clone();
        let body = json!({ "messages": [{ "role": "user", "content": text }] });
        std::thread::spawn(move || {
            let answer = agent
                .chat_as(&body, alias)
                .map(|c| c["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string())
                .map_err(|e| e.message());
            let _ = tx.send(answer);
        });
        rx
    }

    fn answered(rx: &Receiver<Result<String, String>>) -> String {
        rx.recv_timeout(WAIT).expect("answered in time").expect("answered without an error")
    }

    #[test]
    fn turns_are_answered_by_the_app_server_over_its_pipes() {
        let fake = Arc::new(FakeCodex::default());
        let agent = fake.agent();
        assert_eq!(answered(&ask_in_background(&agent, "Set up OAIY", None)), "You said: Set up OAIY");
        let call = ask_in_background(&agent, "Hello, who is this?", Some(LiveCallAlias::ReasoningNone));
        assert_eq!(answered(&call), "You said: Hello, who is this?");
        let s = fake.state.lock().unwrap();
        assert_eq!(s.started, [("Set up OAIY".to_string(), false), ("Hello, who is this?".to_string(), true)]);
        assert_eq!(s.connects, 1, "one app-server serves every turn");
    }

    #[test]
    fn sign_in_status_and_sign_out_share_the_one_app_server() {
        let fake = Arc::new(FakeCodex::default());
        let agent = fake.agent();
        let before = agent.status();
        assert!(before.available && !before.connected, "{before:?}");

        let login = agent.start_login(false).expect("the login starts");
        assert_eq!(login["loginId"], "login-1");
        assert_eq!(login["authUrl"], "https://auth.openai.com/oauth/authorize?fake=1");
        let signed_in = agent.status();
        assert!(signed_in.connected, "{signed_in:?}");
        assert_eq!(signed_in.email.as_deref(), Some("caller@example.com"));
        assert_eq!(signed_in.plan_type.as_deref(), Some("plus"));
        assert_eq!(agent.models().expect("models")["data"][0]["id"], "gpt-5.5");
        assert_eq!(answered(&ask_in_background(&agent, "Hi", Some(LiveCallAlias::LunaReasoningLowFast))), "You said: Hi");

        agent.logout().expect("signed out");
        assert!(!agent.status().connected);
        let s = fake.state.lock().unwrap();
        assert_eq!((s.connects, s.login_starts), (1, 1), "one app-server, one login flow");
    }

    #[test]
    fn a_live_call_is_answered_while_a_long_turn_holds_the_app_server() {
        let fake = Arc::new(FakeCodex::default());
        let agent = fake.agent();
        // A long setup turn: it says its first words, then holds.
        let setup = ask_in_background(&agent, "Set up OAIY hold:setup", None);
        assert!(fake.wait_for(|s| s.started.len() == 1, WAIT), "the setup turn started");

        // A caller speaks meanwhile, and the panel asks whether ChatGPT is signed in.
        let call = ask_in_background(&agent, "Hello, is anyone there?", Some(LiveCallAlias::ReasoningNone));
        let heard = call.recv_timeout(WAIT);
        let (tx, status) = channel();
        let asking = agent.clone();
        std::thread::spawn(move || {
            let _ = tx.send(asking.status());
        });
        let status = status.recv_timeout(WAIT);
        let finished_while_held = fake.state.lock().unwrap().finished.clone();
        // Before any assertion, so a failure never leaves the turn held.
        fake.release("setup");

        let heard = heard.expect("the caller waited behind the setup turn");
        assert_eq!(heard.expect("the call was answered"), "You said: Hello, is anyone there?");
        let status = status.expect("the status waited behind the setup turn");
        assert!(status.available && !status.connected, "{status:?}");
        assert_eq!(finished_while_held, ["Hello, is anyone there?"], "the setup turn was still held");

        // The setup's first words went out before the call's and the rest after
        // them; each answer is its own, whole, with nothing of the other's.
        assert_eq!(answered(&setup), "You said: Set up OAIY hold:setup");
        let s = fake.state.lock().unwrap();
        assert_eq!(s.started, [("Set up OAIY hold:setup".to_string(), false), ("Hello, is anyone there?".to_string(), true)]);
        assert_eq!(s.connects, 1, "one app-server took both turns");
    }

    #[test]
    fn the_turns_of_a_call_are_taken_one_at_a_time_in_the_order_they_came() {
        let fake = Arc::new(FakeCodex::default());
        let agent = fake.agent();
        let call = Some(LiveCallAlias::ReasoningNone);
        // The caller's first sentence is still being answered…
        let first = ask_in_background(&agent, "first hold:first", call);
        assert!(fake.wait_for(|s| s.started.len() == 1, WAIT), "the first turn started");
        // …when the second and the third arrive, each once the one before is in line.
        let second = ask_in_background(&agent, "second", call);
        assert!(eventually(|| agent.call_lane.queued() == 2), "the second is in line");
        let third = ask_in_background(&agent, "third", call);
        assert!(eventually(|| agent.call_lane.queued() == 3), "the third is in line");
        // A setup turn meanwhile waits for none of them, either.
        let setup = answered(&ask_in_background(&agent, "Set up OAIY", None));
        let calls_started_while_held = fake.state.lock().unwrap().started.iter().filter(|(_, c)| *c).count();
        fake.release("first");

        assert_eq!(setup, "You said: Set up OAIY");
        assert_eq!(calls_started_while_held, 1, "no later turn of the call started before the first finished");
        assert_eq!(answered(&first), "You said: first hold:first");
        assert_eq!(answered(&second), "You said: second");
        assert_eq!(answered(&third), "You said: third");
        let s = fake.state.lock().unwrap();
        fn calls<'a>(turns: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
            turns.filter(|p| *p != "Set up OAIY").collect()
        }
        let order = ["first hold:first", "second", "third"];
        assert_eq!(calls(s.started.iter().map(|(p, _)| p.as_str())), order, "started in the order they came");
        assert_eq!(calls(s.finished.iter().map(String::as_str)), order, "answered in the order they came");
        assert_eq!(s.most_calls_running, 1, "never two of the call's turns at once");
        assert_eq!(agent.call_lane.queued(), 0);
    }

    #[test]
    fn an_app_server_that_dies_fails_its_turn_at_once_and_is_replaced() {
        let fake = Arc::new(FakeCodex::default());
        let agent = fake.agent();
        let held = ask_in_background(&agent, "Set up OAIY hold:forever", None);
        assert!(fake.wait_for(|s| s.started.len() == 1, WAIT), "the turn started");
        fake.die();
        // Told now, not when its three minutes run out; and its first words
        // are not passed off as the answer.
        let failed = held.recv_timeout(WAIT).expect("the turn was told at once");
        fake.release("forever");
        let err = failed.expect_err("a turn cut short is not an answer");
        assert!(err.contains("exited"), "{err}");

        // The next request, a call's, starts a new app-server and is answered.
        let call = ask_in_background(&agent, "Hello?", Some(LiveCallAlias::ReasoningNone));
        assert_eq!(answered(&call), "You said: Hello?");
        assert!(agent.status().available);
        assert_eq!(fake.state.lock().unwrap().connects, 2);
    }
}
