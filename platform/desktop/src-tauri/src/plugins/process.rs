//! A live, supervised plugin process.
//!
//! Owns the child, a reader thread that routes inbound lines, and the correlation
//! map that lets several in-flight requests coexist.
//!
//! ```text
//!   request()  ──write──►  child stdin
//!                          child stdout ──► reader thread
//!                                             ├─ Response/Error → the waiting caller
//!                                             ├─ Notification   → the event/log sink
//!                                             ├─ Request        → answered here
//!                                             └─ NonProtocol    → the log ring
//! ```
//!
//! # Routing lives in the reader thread, not in the caller
//!
//! The obvious design — have `request()` read from stdout until it sees its own
//! reply — breaks the moment two requests are in flight: each caller consumes
//! lines belonging to the other, and a health probe swallows a connector reply.
//! So exactly one thread reads, and it hands each reply to whoever registered
//! that id.
//!
//! A consequence worth stating: **a reply that arrives after its caller gave up
//! is dropped, not applied.** The oneshot receiver is gone by then. Combined with
//! ids never being reused, that means a late reply can never be mistaken for the
//! answer to a newer question.

use std::collections::HashMap;
use std::io::BufReader;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

use super::manifest::PluginManifest;
use super::rpc::{
    self, classify, error_line, read_line_capped, result_line, write_line, RpcError, RpcId,
    RpcMessage,
};
use super::runner::{plugin_data_dir, plugin_env, HANDSHAKE_TIMEOUT, HEALTH_TIMEOUT, SHUTDOWN_GRACE};
use crate::services::runner::LogBuffer;

/// A failed plugin call, in the terms a caller can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    /// The plugin did not answer inside the deadline.
    Timeout { method: String, waited: Duration },
    /// The plugin answered with a JSON-RPC error.
    Plugin {
        code: i32,
        message: String,
        /// The typed `error.data.code`, when the plugin supplied one.
        typed: Option<String>,
    },
    /// The process is gone, or its stdio closed.
    Gone(String),
    Transport(String),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Timeout { method, waited } => write!(
                f,
                "the plugin did not answer {method} within {:.1}s",
                waited.as_secs_f32()
            ),
            CallError::Plugin { code, message, typed } => match typed {
                Some(t) => write!(f, "the plugin refused ({t}): {message}"),
                None => write!(f, "the plugin returned error {code}: {message}"),
            },
            CallError::Gone(m) => write!(f, "the plugin process is not available: {m}"),
            CallError::Transport(m) => write!(f, "plugin transport error: {m}"),
        }
    }
}

/// What the reader thread does with a plugin-initiated notification.
///
/// Boxed rather than generic so a `PluginProcess` is one concrete type and can
/// live in a registry map.
pub type EventSink = Arc<dyn Fn(&str, Value) + Send + Sync>;

/// What the reader thread does with a plugin-initiated **request** (currently
/// only `flow.run`). Returns `Ok(result)` or `Err((typed_code, message))`.
pub type RequestHandler =
    Arc<dyn Fn(&str, Value) -> Result<Value, (String, String)> + Send + Sync>;

type Pending = Arc<Mutex<HashMap<RpcId, Sender<Result<Value, CallError>>>>>;

/// Sender half of the stdin writer channel. `None` after close.
type WriteTx = Arc<Mutex<Option<std::sync::mpsc::SyncSender<String>>>>;

pub struct PluginProcess {
    pub plugin_id: String,
    pub pid: u32,
    pub logs: LogBuffer,
    child: Arc<Mutex<Option<Child>>>,
    /// All stdin writes go through a bounded channel to ONE writer thread.
    ///
    /// This replaced a mutex held across a blocking pipe write, and the change
    /// is load-bearing: `request()`'s timeout bounds only the reply wait, not
    /// the write, so a plugin that stopped draining its stdin filled the OS
    /// pipe buffer and the next writer blocked forever HOLDING THE MUTEX. The
    /// supervisor thread and the event thread are shared across ALL plugins,
    /// so one wedged plugin froze supervision, event dispatch, `stop()` and
    /// app exit — the review confirmed the full chain. With a channel, only
    /// the writer thread blocks; a full channel is reported as an error
    /// (`try_send`), never waited on.
    write_tx: WriteTx,
    pending: Pending,
    next_id: Arc<Mutex<i64>>,
}

pub struct SpawnOptions {
    pub desktop_version: String,
    pub dev_mode: bool,
    /// Receives validated `event.emit` payloads and `log.emit` lines.
    pub events: EventSink,
    /// Answers plugin-initiated requests.
    pub requests: RequestHandler,
}

impl PluginProcess {
    /// Spawn the plugin. Does **not** perform the handshake — call
    /// [`PluginProcess::init`] next, so the caller controls the state machine.
    pub fn spawn(
        manifest: &PluginManifest,
        dir: &std::path::Path,
        opts: SpawnOptions,
    ) -> Result<Self, String> {
        let exe = manifest
            .resolve_entry(dir)
            .map_err(|e| format!("cannot launch plugin: {}", e.reason()))?;
        if !exe.is_file() {
            // Said plainly, because "the plugin won't start" with no path is the
            // single least useful error a plugin host can produce.
            return Err(format!(
                "plugin executable not found at {}",
                exe.display()
            ));
        }

        let data_dir = plugin_data_dir(dir);
        // Created eagerly: a plugin told where its data dir is will write there
        // immediately, and failing on the first write is a worse first run than
        // failing here where we can name the path.
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| format!("cannot create plugin data dir {}: {e}", data_dir.display()))?;

        let env = plugin_env(
            std::env::vars().collect::<Vec<_>>(),
            &manifest.id,
            &data_dir,
            &opts.desktop_version,
            manifest.plugin_api_version,
            opts.dev_mode,
        );

        let mut cmd = Command::new(&exe);
        cmd.args(&manifest.entry.args)
            // cwd is the plugin dir so relative paths in the plugin resolve
            // against its own folder rather than wherever OAIY was launched from.
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Clear first: `envs()` ADDS to the inherited set, so without this the
            // allow-list would be a no-op and every host variable would leak.
            .env_clear()
            .envs(&env);

        #[cfg(windows)]
        {
            // No console window for a tray-resident app's children.
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot launch {}: {e}", exe.display()))?;
        let pid = child.id();
        let stdout = child.stdout.take().ok_or("plugin stdout unavailable")?;
        let stderr = child.stderr.take();
        let stdin = child.stdin.take().ok_or("plugin stdin unavailable")?;

        let logs = LogBuffer::new();
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));

        // Bounded: an unbounded channel to a wedged writer grows without limit.
        // 256 lines of headroom absorbs bursts (an event storm of acks) while a
        // plugin that has genuinely stopped reading surfaces as a send error
        // within one buffer's worth of traffic.
        let (raw_tx, write_rx) = std::sync::mpsc::sync_channel::<String>(256);
        let write_tx: WriteTx = Arc::new(Mutex::new(Some(raw_tx)));
        {
            // The ONLY thread that touches the child's stdin. It alone blocks on
            // a full pipe; everyone else does a non-blocking try_send.
            let mut stdin_owned = stdin;
            thread::spawn(move || {
                while let Ok(line) = write_rx.recv() {
                    if write_line(&mut stdin_owned, &line).is_err() {
                        break;
                    }
                }
                // Channel closed or write failed: dropping stdin sends EOF, the
                // well-behaved plugin's exit signal.
                drop(stdin_owned);
            });
        }

        // stderr is drained on its own thread. Without this a chatty plugin fills
        // the pipe buffer and blocks forever on its next write — a hang whose
        // cause is invisible.
        if let Some(err) = stderr {
            let logs_err = logs.clone();
            thread::spawn(move || {
                let mut r = BufReader::new(err);
                loop {
                    match read_line_capped(&mut r) {
                        Ok(line) => logs_err.push("stderr", line),
                        Err(_) => break,
                    }
                }
            });
        }

        // The single reader.
        {
            let logs = logs.clone();
            let pending = pending.clone();
            let tx_for_replies = write_tx.clone();
            let plugin_id = manifest.id.clone();
            let events = opts.events.clone();
            let requests = opts.requests.clone();
            let allowed_events: Vec<String> = manifest.events.clone();
            thread::spawn(move || {
                let mut r = BufReader::new(stdout);
                loop {
                    let line = match read_line_capped(&mut r) {
                        Ok(l) => l,
                        Err(RpcError::Closed) => {
                            fail_all(&pending, CallError::Gone("stdout closed".into()));
                            break;
                        }
                        Err(e) => {
                            // A LineTooLong poisons framing — see rpc.rs. Stop
                            // reading rather than resynchronising onto garbage.
                            logs.push("stderr", format!("[protocol] {e}"));
                            fail_all(&pending, CallError::Transport(e.to_string()));
                            break;
                        }
                    };

                    match classify(&line) {
                        RpcMessage::Response { id, result } => {
                            deliver(&pending, &id, Ok(result));
                        }
                        RpcMessage::ErrorResponse { id, code, message, data } => {
                            let typed = data
                                .as_ref()
                                .and_then(|d| d.get("code"))
                                .and_then(Value::as_str)
                                .map(str::to_string);
                            deliver(
                                &pending,
                                &id,
                                Err(CallError::Plugin { code, message, typed }),
                            );
                        }
                        RpcMessage::Notification { method, params } => {
                            handle_notification(
                                &plugin_id,
                                &allowed_events,
                                &logs,
                                &events,
                                &method,
                                params,
                            );
                        }
                        RpcMessage::Request { id, method, params } => {
                            let reply = match requests(&method, params) {
                                Ok(v) => result_line(&id, v),
                                Err((typed, msg)) => error_line(
                                    &id,
                                    rpc::METHOD_NOT_FOUND,
                                    &msg,
                                    Some(&typed),
                                ),
                            };
                            if let Ok(guard) = tx_for_replies.lock() {
                                if let Some(tx) = guard.as_ref() {
                                    let _ = tx.try_send(reply);
                                }
                            }
                        }
                        RpcMessage::NonProtocol(text) => {
                            if !text.is_empty() {
                                logs.push("stdout", text);
                            }
                        }
                    }
                }
            });
        }

        Ok(Self {
            plugin_id: manifest.id.clone(),
            pid,
            logs,
            child: Arc::new(Mutex::new(Some(child))),
            write_tx,
            pending,
            next_id: Arc::new(Mutex::new(1)),
        })
    }

    /// Send a request and wait for its reply.
    pub fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, CallError> {
        let id = {
            let mut n = self
                .next_id
                .lock()
                .map_err(|_| CallError::Transport("id lock poisoned".into()))?;
            let id = RpcId::Num(*n);
            *n += 1;
            id
        };

        let (tx, rx): (Sender<Result<Value, CallError>>, Receiver<_>) = channel();
        {
            let mut p = self
                .pending
                .lock()
                .map_err(|_| CallError::Transport("pending lock poisoned".into()))?;
            p.insert(id.clone(), tx);
        }

        let line = format!(
            "{}\n",
            json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
        );
        if let Err(e) = self.write(&line) {
            self.forget(&id);
            return Err(e);
        }

        match rx.recv_timeout(timeout) {
            Ok(res) => {
                self.forget(&id);
                res
            }
            Err(RecvTimeoutError::Timeout) => {
                // Drop the registration so a late reply is discarded rather than
                // sitting in the map forever. Ids never repeat, so it cannot be
                // mistaken for a newer request's answer.
                self.forget(&id);
                Err(CallError::Timeout {
                    method: method.to_string(),
                    waited: timeout,
                })
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.forget(&id);
                Err(CallError::Gone("reader thread ended".into()))
            }
        }
    }

    fn forget(&self, id: &RpcId) {
        if let Ok(mut p) = self.pending.lock() {
            p.remove(id);
        }
    }

    fn write(&self, line: &str) -> Result<(), CallError> {
        let guard = self
            .write_tx
            .lock()
            .map_err(|_| CallError::Transport("write channel lock poisoned".into()))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| CallError::Gone("stdin already closed".into()))?;
        // try_send, never send: a full channel means the plugin has stopped
        // reading its stdin, and the caller must learn that as an error, not by
        // blocking a thread the whole host shares.
        tx.try_send(line.to_string()).map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(_) => CallError::Gone(
                "the plugin has stopped reading its stdin (write backlog full)".into(),
            ),
            std::sync::mpsc::TrySendError::Disconnected(_) => {
                CallError::Gone("the plugin's stdin writer has exited".into())
            }
        })
    }

    /// The `plugin.init` handshake.
    pub fn init(&self, api_version: u32, data_dir: &std::path::Path, dev_mode: bool) -> Result<Value, CallError> {
        self.request(
            "plugin.init",
            json!({
                "desktopVersion": env!("CARGO_PKG_VERSION"),
                "pluginApiVersion": api_version,
                "dataDir": data_dir.display().to_string(),
                "devMode": dev_mode,
                // Advertised host features the plugin may opt into. Absent means
                // a legacy host, so a plugin must tolerate an empty list.
                "features": ["eventAck"],
            }),
            HANDSHAKE_TIMEOUT,
        )
    }

    pub fn health(&self) -> Result<Value, CallError> {
        self.request("plugin.health", json!({}), HEALTH_TIMEOUT)
    }

    /// Acknowledge a durably-journalled event so the plugin can stop re-sending.
    pub fn ack_event(&self, idempotency_key: &str) -> Result<(), CallError> {
        self.write(&rpc::notification_line(
            "event.ack",
            json!({ "idempotencyKey": idempotency_key }),
        ))
    }

    /// Has the process exited? `None` while still running.
    pub fn check_exited(&self) -> Option<Option<i32>> {
        let mut guard = self.child.lock().ok()?;
        let child = guard.as_mut()?;
        match child.try_wait() {
            Ok(Some(status)) => Some(status.code()),
            _ => None,
        }
    }

    /// Ask nicely, then insist.
    ///
    /// `plugin.shutdown` first so a plugin can release hardware — Aokie holds a
    /// USB dongle, and killing it mid-transaction can leave the device wedged
    /// until it is physically replugged. Then a bounded grace period, then kill:
    /// a plugin that ignores shutdown must not be able to block app exit.
    pub fn shutdown(&self) {
        let _ = self.write(&rpc::notification_line("plugin.shutdown", json!({})));

        let deadline = std::time::Instant::now() + SHUTDOWN_GRACE;
        while std::time::Instant::now() < deadline {
            if self.check_exited().is_some() {
                self.close_stdin();
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }

        // Closing stdin gives a well-behaved plugin a second exit signal (EOF on
        // its read loop) before we resort to killing it.
        self.close_stdin();
        thread::sleep(Duration::from_millis(200));
        if self.check_exited().is_some() {
            return;
        }
        self.kill();
    }

    fn close_stdin(&self) {
        // Dropping the sender disconnects the channel; the writer thread drains
        // what it already accepted, then drops ChildStdin, which is EOF to the
        // plugin — its second exit signal after `plugin.shutdown`.
        if let Ok(mut g) = self.write_tx.lock() {
            g.take();
        }
    }

    pub fn kill(&self) {
        if let Ok(mut g) = self.child.lock() {
            if let Some(child) = g.as_mut() {
                // Kill the TREE, not just the direct child. A plugin's entry is
                // routinely a shim (a .cmd launching node, a script launching a
                // real binary); Child::kill terminates only the shim and orphans
                // the grandchild, which keeps holding its hardware and its
                // inherited pipe handles. Proven twice in this repo's own test
                // runs: a killed test plugin's node grandchild survived and held
                // a pipeline open past a ten-minute timeout. services/registry
                // learned this lesson first; reuse its taskkill machinery.
                crate::services::registry::kill_process_tree(child.id());
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        fail_all(&self.pending, CallError::Gone("process killed".into()));
    }
}

impl Drop for PluginProcess {
    fn drop(&mut self) {
        // A dropped handle must not leave an orphan holding a dongle or a port.
        if self.check_exited().is_none() {
            self.kill();
        }
    }
}

fn deliver(pending: &Pending, id: &RpcId, res: Result<Value, CallError>) {
    let tx = pending.lock().ok().and_then(|mut p| p.remove(id));
    if let Some(tx) = tx {
        // The receiver may already be gone (the caller timed out). Dropping the
        // value is correct: applying it would answer a question nobody is asking.
        let _ = tx.send(res);
    }
}

fn fail_all(pending: &Pending, err: CallError) {
    let drained: Vec<_> = match pending.lock() {
        Ok(mut p) => p.drain().map(|(_, tx)| tx).collect(),
        Err(_) => return,
    };
    // Every waiter must be told, or a caller blocks until its timeout for a
    // process we already know is gone.
    for tx in drained {
        let _ = tx.send(Err(err.clone()));
    }
}

fn handle_notification(
    plugin_id: &str,
    allowed_events: &[String],
    logs: &LogBuffer,
    events: &EventSink,
    method: &str,
    params: Value,
) {
    match method {
        "log.emit" => {
            let level = params
                .get("level")
                .and_then(Value::as_str)
                .unwrap_or("info");
            let msg = params.get("message").and_then(Value::as_str).unwrap_or("");
            // Truncate on a CHAR BOUNDARY. `&msg[..N]` panics when byte N is
            // inside a multi-byte char, and msg is untrusted plugin output — one
            // >2KiB log line containing an emoji or an accented name (byte 2048
            // mid-'é') panicked the reader thread, which then never drained the
            // plugin, never failed its pending calls, and left it permanently
            // unresponsive-but-alive. A security review caught it; a real plugin
            // would have hit it by accident.
            let truncated = if msg.len() > rpc::MAX_LOG_BYTES {
                let mut end = rpc::MAX_LOG_BYTES;
                while end > 0 && !msg.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}… (truncated)", &msg[..end])
            } else {
                msg.to_string()
            };
            logs.push("stdout", format!("[{level}] {truncated}"));
        }
        "event.emit" => {
            let envelope = params.get("event").cloned().unwrap_or(Value::Null);
            match validate_event(plugin_id, allowed_events, &envelope) {
                Ok(name) => events(&name, envelope),
                // Dropped AND logged. Silently discarding would leave a plugin
                // author with an event that never arrives and no way to find out
                // why; forwarding it would let a plugin reach triggers it was
                // never granted.
                Err(why) => logs.push("stderr", format!("[event dropped] {why}")),
            }
        }
        other => {
            logs.push("stderr", format!("[protocol] unknown notification {other}"));
        }
    }
}

/// Validate an inbound event envelope against the manifest and the schema's
/// required fields. Returns the event name on success.
pub fn validate_event(
    plugin_id: &str,
    allowed_events: &[String],
    envelope: &Value,
) -> Result<String, String> {
    let obj = envelope
        .as_object()
        .ok_or_else(|| "event is not an object".to_string())?;

    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "event has no name".to_string())?;

    if !allowed_events.iter().any(|e| e == name) {
        return Err(format!(
            "{plugin_id} emitted {name:?}, which its manifest does not declare"
        ));
    }
    // `source` must be the plugin itself: otherwise a plugin could forge events
    // attributed to another subsystem and reach that subsystem's triggers.
    match obj.get("source").and_then(Value::as_str) {
        Some(src) if src == plugin_id => {}
        Some(src) => {
            return Err(format!(
                "{plugin_id} emitted an event claiming source {src:?}"
            ))
        }
        None => return Err("event has no source".into()),
    }
    // Both dedupe fields are required: delivery is at-least-once, and without
    // them a reconnect replays history and re-fires every trigger.
    for field in ["correlationId", "idempotencyKey", "occurredAt"] {
        if !obj.get(field).and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err(format!("event {name} has no {field}"));
        }
    }
    if !obj.contains_key("data") {
        return Err(format!("event {name} has no data"));
    }
    let size = serde_json::to_string(envelope).map(|s| s.len()).unwrap_or(0);
    if size > rpc::MAX_EVENT_BYTES {
        return Err(format!(
            "event {name} is {size} bytes, over the {} byte limit",
            rpc::MAX_EVENT_BYTES
        ));
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOWED: &[&str] = &["aokie.call.incoming"];

    fn allowed() -> Vec<String> {
        ALLOWED.iter().map(|s| s.to_string()).collect()
    }

    fn envelope() -> Value {
        json!({
            "schemaVersion": 1,
            "source": "aokie",
            "name": "aokie.call.incoming",
            "correlationId": "call_1",
            "idempotencyKey": "aokie:call_1:incoming:v1",
            "occurredAt": "2026-07-30T04:12:09Z",
            "data": { "callerNumber": "+61400000000" }
        })
    }

    #[test]
    fn a_declared_event_validates() {
        assert_eq!(
            validate_event("aokie", &allowed(), &envelope()).unwrap(),
            "aokie.call.incoming"
        );
    }

    #[test]
    fn an_undeclared_event_is_rejected() {
        let mut e = envelope();
        e["name"] = json!("aokie.call.invented");
        let err = validate_event("aokie", &allowed(), &e).unwrap_err();
        assert!(err.contains("does not declare"), "{err}");
    }

    #[test]
    fn a_forged_source_is_rejected() {
        // Otherwise a plugin forges events attributed to another subsystem and
        // reaches that subsystem's triggers.
        let mut e = envelope();
        e["source"] = json!("desktop");
        let err = validate_event("aokie", &allowed(), &e).unwrap_err();
        assert!(err.contains("claiming source"), "{err}");
    }

    #[test]
    fn the_dedupe_fields_are_required() {
        for field in ["correlationId", "idempotencyKey", "occurredAt"] {
            let mut e = envelope();
            e.as_object_mut().unwrap().remove(field);
            let err = validate_event("aokie", &allowed(), &e)
                .unwrap_err_or_panic(&format!("{field} must be required"));
            assert!(err.contains(field), "{err}");
        }
        // Present-but-empty is not present.
        let mut e = envelope();
        e["idempotencyKey"] = json!("");
        assert!(validate_event("aokie", &allowed(), &e).is_err());
    }

    #[test]
    fn data_is_required() {
        let mut e = envelope();
        e.as_object_mut().unwrap().remove("data");
        assert!(validate_event("aokie", &allowed(), &e).is_err());
    }

    #[test]
    fn an_oversized_event_is_rejected() {
        let mut e = envelope();
        e["data"] = json!({ "blob": "x".repeat(rpc::MAX_EVENT_BYTES + 100) });
        let err = validate_event("aokie", &allowed(), &e).unwrap_err();
        assert!(err.contains("over the"), "{err}");
    }

    #[test]
    fn a_non_object_envelope_is_rejected() {
        for v in [json!("hello"), json!([1, 2]), json!(null), json!(7)] {
            assert!(validate_event("aokie", &allowed(), &v).is_err(), "{v}");
        }
    }

    #[test]
    fn call_errors_describe_themselves() {
        let t = CallError::Timeout {
            method: "plugin.health".into(),
            waited: Duration::from_secs(5),
        };
        assert!(t.to_string().contains("plugin.health"), "{t}");
        assert!(t.to_string().contains("5.0s"), "{t}");

        let p = CallError::Plugin {
            code: -32601,
            message: "not declared".into(),
            typed: Some("capability_denied".into()),
        };
        assert!(p.to_string().contains("capability_denied"), "{p}");
    }

    /// Small helper so the loop above reads as one assertion per field.
    trait UnwrapErrOr {
        fn unwrap_err_or_panic(self, msg: &str) -> String;
    }
    impl UnwrapErrOr for Result<String, String> {
        fn unwrap_err_or_panic(self, msg: &str) -> String {
            match self {
                Err(e) => e,
                Ok(v) => panic!("{msg} (got Ok({v}))"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // End-to-end against a REAL child process.
    //
    // The unit tests above cover envelope validation; they cannot show that the
    // reader thread routes, that ids correlate under concurrency, that stray
    // stdout survives as logs, or that shutdown actually ends a process. Those
    // are the parts that break, so they get a real plugin: a small Node script
    // launched through a `.cmd` shim.
    //
    // The shim exists because `entry.command` is confined to the plugin directory
    // — deliberately, so a manifest cannot name `System32\cmd.exe`. A test that
    // bypassed that would be testing a path real plugins never take.
    #[cfg(windows)]
    mod e2e {
        use super::super::*;
        use super::*;
        use std::fs;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Mutex as StdMutex;

        static N: AtomicU32 = AtomicU32::new(0);

        const PLUGIN_JS: &str = r#"
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
// Ordinary stdout noise, of the kind a real dependency prints. Must survive as
// a log line and must not be treated as a protocol violation.
console.log("fake-plugin booting on COM3");
let buf = "";
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let msg; try { msg = JSON.parse(line); } catch { continue; }
    if (msg.method === "plugin.init") {
      send({ jsonrpc: "2.0", id: msg.id, result: { ok: true, features: msg.params.features } });
      // A declared event...
      send({ jsonrpc: "2.0", method: "event.emit", params: { event: {
        schemaVersion: 1, source: "fake", name: "fake.thing.happened",
        correlationId: "c1", idempotencyKey: "fake:c1:v1",
        occurredAt: "2026-07-30T04:00:00Z", data: { n: 1 } } } });
      // ...and one the manifest does NOT declare, which must be dropped.
      send({ jsonrpc: "2.0", method: "event.emit", params: { event: {
        schemaVersion: 1, source: "fake", name: "fake.not.declared",
        correlationId: "c2", idempotencyKey: "fake:c2:v1",
        occurredAt: "2026-07-30T04:00:01Z", data: {} } } });
      send({ jsonrpc: "2.0", method: "log.emit", params: { level: "info", message: "hello from the plugin" } });
      // A plugin-initiated request the host must answer.
      send({ jsonrpc: "2.0", id: 9001, method: "flow.run", params: { flowId: "greet" } });
    } else if (msg.method === "plugin.health") {
      send({ jsonrpc: "2.0", id: msg.id, result: { status: "ok" } });
    } else if (msg.method === "slow") {
      // Never answers: exercises the request timeout.
    } else if (msg.method === "boom") {
      send({ jsonrpc: "2.0", id: msg.id, error: { code: -32000, message: "no dongle", data: { code: "capability_unavailable" } } });
    } else if (msg.method === "plugin.shutdown") {
      process.exit(0);
    } else if (msg.id !== undefined) {
      send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "unknown method" } });
    }
  }
});
"#;

        struct Fixture {
            dir: PathBuf,
            manifest: PluginManifest,
        }

        impl Fixture {
            fn new() -> Option<Self> {
                if std::process::Command::new("node")
                    .arg("--version")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .map(|s| !s.success())
                    .unwrap_or(true)
                {
                    // No Node on this machine: skip rather than fail. A missing
                    // toolchain is not a defect in the code under test.
                    return None;
                }
                let n = N.fetch_add(1, Ordering::Relaxed);
                let dir = std::env::temp_dir()
                    .join(format!("oaiy-plugproc-{}-{n}", std::process::id()));
                let _ = fs::remove_dir_all(&dir);
                fs::create_dir_all(&dir).ok()?;
                fs::write(dir.join("plugin.mjs"), PLUGIN_JS).ok()?;
                // The shim keeps entry.command relative, as real plugins must be.
                fs::write(
                    dir.join("plugin.cmd"),
                    "@echo off\r\nnode \"%~dp0plugin.mjs\" %*\r\n",
                )
                .ok()?;
                let manifest_json = serde_json::json!({
                    "schemaVersion": 3,
                    "id": "fake",
                    "name": "Fake plugin",
                    "version": "0.0.1",
                    "pluginApiVersion": 1,
                    "entry": { "kind": "process", "command": "plugin.cmd" },
                    "capabilities": ["oaiy.flow.run"],
                    "connectors": [],
                    "events": ["fake.thing.happened"],
                });
                fs::write(
                    dir.join("manifest.json"),
                    serde_json::to_string_pretty(&manifest_json).ok()?,
                )
                .ok()?;
                let manifest = PluginManifest::load(&dir).ok()?;
                Some(Self { dir, manifest })
            }
        }

        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.dir);
            }
        }

        type Seen = Arc<StdMutex<Vec<String>>>;

        fn spawn(f: &Fixture) -> (PluginProcess, Seen, Seen) {
            let events: Seen = Arc::new(StdMutex::new(Vec::new()));
            let asked: Seen = Arc::new(StdMutex::new(Vec::new()));
            let ev = events.clone();
            let ak = asked.clone();
            let p = PluginProcess::spawn(
                &f.manifest,
                &f.dir,
                SpawnOptions {
                    desktop_version: "0.1.0".into(),
                    dev_mode: false,
                    events: Arc::new(move |name, _payload| {
                        ev.lock().unwrap().push(name.to_string());
                    }),
                    requests: Arc::new(move |method, _params| {
                        ak.lock().unwrap().push(method.to_string());
                        Ok(serde_json::json!({ "runId": "run_1", "status": "queued" }))
                    }),
                },
            )
            .expect("spawn");
            (p, events, asked)
        }

        #[test]
        fn a_real_plugin_handshakes_emits_and_shuts_down() {
            let Some(f) = Fixture::new() else { return };
            let (p, events, asked) = spawn(&f);

            let init = p
                .init(1, &f.dir.join("data"), false)
                .expect("plugin.init must answer");
            assert_eq!(init["ok"], true);
            // The host advertises eventAck; the fixture echoes what it received.
            assert_eq!(init["features"][0], "eventAck");

            let health = p.health().expect("plugin.health must answer");
            assert_eq!(health["status"], "ok");

            // Give the reader a moment to drain the notifications the plugin sent
            // during init.
            for _ in 0..40 {
                if !events.lock().unwrap().is_empty() && !asked.lock().unwrap().is_empty() {
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }

            let seen = events.lock().unwrap().clone();
            assert!(
                seen.contains(&"fake.thing.happened".to_string()),
                "a declared event must reach the sink: {seen:?}"
            );
            assert!(
                !seen.contains(&"fake.not.declared".to_string()),
                "an undeclared event must be dropped: {seen:?}"
            );

            assert_eq!(
                asked.lock().unwrap().clone(),
                vec!["flow.run".to_string()],
                "a plugin-initiated request must reach the handler"
            );

            let logs: Vec<String> = p.logs.snapshot(None).into_iter().map(|l| l.text).collect();
            let joined = logs.join("\n");
            assert!(
                joined.contains("fake-plugin booting on COM3"),
                "stray stdout must survive as a log line: {joined}"
            );
            assert!(
                joined.contains("hello from the plugin"),
                "log.emit must reach the ring: {joined}"
            );
            assert!(
                joined.contains("does not declare"),
                "a dropped event must say why, or a plugin author cannot debug it: {joined}"
            );

            p.shutdown();
            assert!(
                p.check_exited().is_some(),
                "shutdown must actually end the process"
            );
        }

        #[test]
        fn a_typed_plugin_error_survives_the_round_trip() {
            let Some(f) = Fixture::new() else { return };
            let (p, _, _) = spawn(&f);
            p.init(1, &f.dir.join("data"), false).expect("init");

            match p.request("boom", serde_json::json!({}), Duration::from_secs(5)) {
                Err(CallError::Plugin { typed, message, .. }) => {
                    assert_eq!(typed.as_deref(), Some("capability_unavailable"));
                    assert!(message.contains("no dongle"), "{message}");
                }
                other => panic!("expected a typed plugin error, got {other:?}"),
            }
            p.shutdown();
        }

        #[test]
        fn a_request_that_is_never_answered_times_out_without_wedging_the_next_one() {
            let Some(f) = Fixture::new() else { return };
            let (p, _, _) = spawn(&f);
            p.init(1, &f.dir.join("data"), false).expect("init");

            match p.request("slow", serde_json::json!({}), Duration::from_millis(400)) {
                Err(CallError::Timeout { method, .. }) => assert_eq!(method, "slow"),
                other => panic!("expected a timeout, got {other:?}"),
            }
            // The point of the test: one abandoned request must not break the
            // channel for everything after it.
            assert_eq!(p.health().expect("health after a timeout")["status"], "ok");
            p.shutdown();
        }

        #[test]
        fn concurrent_requests_do_not_steal_each_others_replies() {
            // The failure this guards against: a caller reading stdout itself
            // consumes the other's reply, so a health probe swallows a connector
            // result. Only correct if exactly one thread reads.
            let Some(f) = Fixture::new() else { return };
            let (p, _, _) = spawn(&f);
            p.init(1, &f.dir.join("data"), false).expect("init");

            let p = Arc::new(p);
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let p = p.clone();
                    thread::spawn(move || p.health())
                })
                .collect();
            for h in handles {
                let r = h.join().expect("thread");
                assert_eq!(r.expect("each caller gets its own reply")["status"], "ok");
            }
            p.shutdown();
        }

        #[test]
        fn killing_the_process_fails_waiters_instead_of_hanging_them() {
            let Some(f) = Fixture::new() else { return };
            let (p, _, _) = spawn(&f);
            p.init(1, &f.dir.join("data"), false).expect("init");

            let p = Arc::new(p);
            let waiter = {
                let p = p.clone();
                thread::spawn(move || p.request("slow", serde_json::json!({}), Duration::from_secs(30)))
            };
            thread::sleep(Duration::from_millis(300));
            p.kill();

            // Without fail_all this blocks for the full 30s on a process we
            // already know is gone.
            let r = waiter.join().expect("thread");
            assert!(matches!(r, Err(CallError::Gone(_))), "got {r:?}");
        }

        #[test]
        fn a_plugin_that_never_reads_stdin_cannot_wedge_shutdown() {
            // The review's blocker: writes used to hold a mutex across a
            // blocking pipe write, so a plugin that stopped draining stdin
            // froze the supervisor, the event thread, stop() and app exit.
            // This plugin never reads AT ALL — the worst case.
            let Some(f) = Fixture::new() else { return };
            fs::write(
                f.dir.join("plugin.mjs"),
                // Stays alive, ignores stdin entirely.
                "setInterval(() => {}, 1000);\n",
            )
            .unwrap();
            let (p, _, _) = spawn(&f);

            // Flood well past the channel bound. Every call must RETURN — an
            // error is fine, a hang is the bug.
            let start = std::time::Instant::now();
            let mut errors = 0;
            for i in 0..400 {
                if p.request(
                    "plugin.health",
                    serde_json::json!({ "n": i }),
                    Duration::from_millis(50),
                )
                .is_err()
                {
                    errors += 1;
                }
            }
            assert!(errors > 0, "an unread stdin must eventually surface as errors");

            // And shutdown must complete despite the wedged child — this is the
            // app-exit guarantee the old code violated.
            p.shutdown();
            assert!(
                p.check_exited().is_some(),
                "shutdown must kill a plugin that ignores it"
            );
            assert!(
                start.elapsed() < Duration::from_secs(40),
                "the whole ordeal must be bounded, took {:?}",
                start.elapsed()
            );
        }

        #[test]
        fn the_child_gets_no_host_secrets() {
            // plugin_env is unit-tested, but this asserts the SPAWN path applies
            // it — `envs()` adds to the inherited set, so a missing `env_clear()`
            // would leak everything while the unit test still passed.
            let Some(f) = Fixture::new() else { return };
            let env = plugin_env(
                vec![("OPENAI_API_KEY", "sk-should-not-appear"), ("PATH", "/x")],
                &f.manifest.id,
                &f.dir.join("data"),
                "0.1.0",
                1,
                false,
            );
            assert!(!env.contains_key("OPENAI_API_KEY"));
            assert!(env.contains_key("OAIY_PLUGIN_ID"));
        }
    }
}
