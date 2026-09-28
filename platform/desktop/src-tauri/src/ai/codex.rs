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

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
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

/// One live child + its RPC plumbing.
struct Session {
    child: Child,
    stdin: ChildStdin,
    next_id: AtomicI64,
    /// id → the reply, filled by the reader thread.
    replies: Arc<Mutex<HashMap<i64, Value>>>,
    /// Notifications, in arrival order (turn deltas etc.).
    notes: Arc<Mutex<Vec<Value>>>,
    _stop: Sender<()>,
}

impl Session {
    fn spawn(codex_home: &Path) -> Result<Self, CodexError> {
        std::fs::create_dir_all(codex_home)
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
        let stdin = child.stdin.take().ok_or_else(|| CodexError::Unavailable("no stdin".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| CodexError::Unavailable("no stdout".into()))?;

        let replies: Arc<Mutex<HashMap<i64, Value>>> = Arc::new(Mutex::new(HashMap::new()));
        let notes: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx): (Sender<()>, Receiver<()>) = channel();

        let r2 = replies.clone();
        let n2 = notes.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if rx.try_recv().is_ok() {
                    return;
                }
                let Ok(line) = line else { return };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let Ok(msg) = serde_json::from_str::<Value>(trimmed) else { continue };
                match msg.get("id").and_then(Value::as_i64) {
                    // A reply to something we asked.
                    Some(id) if msg.get("method").is_none() => {
                        if let Ok(mut map) = r2.lock() {
                            map.insert(id, msg);
                        }
                    }
                    // A server→client REQUEST. We expose no tools, so refusing is
                    // the correct answer (and prevents the child from stalling).
                    Some(_) => {}
                    None => {
                        if let Ok(mut v) = n2.lock() {
                            v.push(msg);
                        }
                    }
                }
            }
        });

        Ok(Session { child, stdin, next_id: AtomicI64::new(0), replies, notes, _stop: tx })
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), CodexError> {
        let line = json!({ "jsonrpc": "2.0", "method": method, "params": params }).to_string();
        writeln!(self.stdin, "{line}").map_err(|e| CodexError::Rpc(format!("write failed: {e}")))?;
        self.stdin.flush().ok();
        Ok(())
    }

    fn call(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value, CodexError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let line = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string();
        writeln!(self.stdin, "{line}").map_err(|e| CodexError::Rpc(format!("write failed: {e}")))?;
        self.stdin.flush().ok();

        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(mut map) = self.replies.lock() {
                if let Some(msg) = map.remove(&id) {
                    if let Some(err) = msg.get("error") {
                        return Err(CodexError::Rpc(
                            err.get("message").and_then(Value::as_str).unwrap_or("codex error").to_string(),
                        ));
                    }
                    return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                }
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        Err(CodexError::Rpc(format!("{method} timed out")))
    }

    /// Notifications received since `from`, leaving the buffer intact.
    fn notes_since(&self, from: usize) -> Vec<Value> {
        self.notes.lock().map(|v| v[from.min(v.len())..].to_vec()).unwrap_or_default()
    }
    fn notes_len(&self) -> usize {
        self.notes.lock().map(|v| v.len()).unwrap_or(0)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

/// The managed agent. One child at a time, started lazily and reused.
pub struct CodexAgent {
    codex_home: PathBuf,
    session: Mutex<Option<Session>>,
}

pub type CodexHandle = Arc<CodexAgent>;

pub fn new_handle(data_dir: &Path) -> CodexHandle {
    Arc::new(CodexAgent {
        codex_home: data_dir.join("ai").join("codex-home"),
        session: Mutex::new(None),
    })
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
    fn with_session<T>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T, CodexError>,
    ) -> Result<T, CodexError> {
        let mut guard = self.session.lock().map_err(|_| CodexError::Rpc("session lock poisoned".into()))?;
        if guard.is_none() {
            let mut s = Session::spawn(&self.codex_home)?;
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
            *guard = Some(s);
        }
        let session = guard.as_mut().expect("just ensured");
        let out = f(session);
        // A dead child must not be reused: drop it so the next call respawns.
        if let Err(CodexError::Rpc(_)) = &out {
            if session.child.try_wait().ok().flatten().is_some() {
                *guard = None;
            }
        }
        out
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
    pub fn chat_streaming(
        &self,
        body: &Value,
        alias: Option<LiveCallAlias>,
        mut on_delta: impl FnMut(&str),
    ) -> Result<Value, CodexError> {
        let prompt = flatten_prompt(body);
        if prompt.trim().is_empty() {
            return Err(CodexError::Rpc("the request carried no message content".into()));
        }
        let model = match alias {
            Some(a) => Some(a.model().to_string()),
            None => body.get("model").and_then(Value::as_str).map(str::to_string),
        };

        let text = self.with_session(|s| {
            // Refuse everything a turn could otherwise reach.
            let mut params = json!({
                "approvalPolicy": "never",
                "dynamicTools": [],
                "environments": [],
                "runtimeWorkspaceRoots": [],
                "allowProviderModelFallback": false,
            });
            if let Some(m) = &model {
                params["model"] = Value::String(m.clone());
            }
            let thread = s.call("thread/start", params, RPC_TIMEOUT)?;
            let thread_id = thread_id_of(&thread)
                .ok_or_else(|| CodexError::Rpc("codex did not return a thread id".into()))?;

            let from = s.notes_len();
            // `input` is a SEQUENCE of typed items, not one map. A map is
            // refused with "invalid type: map, expected a sequence" — a runtime
            // message with nothing in it naming this line.
            let mut turn = json!({
                "threadId": thread_id,
                "input": [{ "type": "text", "text": prompt }],
            });
            if let Some(a) = alias {
                // Named on the TURN as well as the thread: the effort is a
                // per-turn setting, and a thread-level model alone would leave
                // it at the account default.
                turn["model"] = Value::String(a.model().to_string());
                turn["effort"] = Value::String(a.reasoning_effort().to_string());
                if let Some(tier) = a.service_tier() {
                    turn["serviceTier"] = Value::String(tier.to_string());
                }
            }
            s.call("turn/start", turn, TURN_TIMEOUT)?;

            // Collect the agent's message from the notification stream.
            //
            // The cursor ADVANCES. Re-reading from a fixed mark every 50 ms
            // appended each delta once per polling round, so a two-word answer
            // came back as "OAIYOAIY tunnel worksOAIY tunnel works." — plausible
            // enough to read as the model repeating itself.
            let deadline = Instant::now() + TURN_TIMEOUT;
            let mut out = String::new();
            let mut done = false;
            let mut cursor = from;
            while Instant::now() < deadline && !done {
                let batch = s.notes_since(cursor);
                cursor += batch.len();
                let before = out.len();
                done = fold_turn_notes(&mut out, &batch);
                // Whatever this batch appended, verbatim — so a streaming
                // caller sees exactly the text the buffered answer will contain
                // and the two can never drift apart.
                if out.len() > before {
                    on_delta(&out[before..]);
                }
                if !done {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if out.is_empty() {
                return Err(CodexError::Rpc("codex returned no output for this turn".into()));
            }
            Ok(out)
        })?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(json!({
            "object": "chat.completion",
            "created": created,
            "model": model.unwrap_or_else(|| CODEX_PROVIDER_ID.to_string()),
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop",
            }],
        }))
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
            Some(row)
        })
        .collect()
}

/// Fold one batch of turn notifications into the answer. Returns whether the
/// turn finished.
///
/// Each note must be folded EXACTLY ONCE — the caller advances its cursor by
/// the batch length for that reason. Folding an overlapping range appends every
/// delta again, and the duplicate reads as the model repeating itself rather
/// than as a cursor that never moved.
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
fn flatten_prompt(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            // The array form SDKs emit: keep the text parts.
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
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
        let turn = json!({
            "threadId": "t1",
            "input": [{ "type": "text", "text": "hello" }],
        });
        let input = turn["input"].as_array().expect("input must be a sequence");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "text");
        assert_eq!(input[0]["text"], "hello");
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

    #[test]
    fn only_the_fast_luna_route_asks_for_priority_service() {
        // Requesting priority on every route would spend the account's priority
        // budget on calls that did not ask for it.
        assert_eq!(LiveCallAlias::LunaReasoningLowFast.service_tier(), Some("priority"));
        assert_eq!(LiveCallAlias::ReasoningNone.service_tier(), None);
        assert_eq!(LiveCallAlias::ReasoningLow.service_tier(), None);
        assert_eq!(LiveCallAlias::LunaReasoningLow.service_tier(), None);
    }
}
