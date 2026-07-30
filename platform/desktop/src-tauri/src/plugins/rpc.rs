//! JSON-RPC 2.0 over newline-delimited stdio.
//!
//! One JSON object per line, `\n`-terminated, UTF-8, **no `Content-Length`
//! framing**. This module is the transport-agnostic core: it classifies inbound
//! lines and builds outbound ones. Process spawning lives in `runner`, so
//! everything here is testable without a child process.
//!
//! # Non-protocol output is expected, not exceptional
//!
//! A plugin writes to stdout by accident all the time — a stray `print`, a
//! dependency's banner, a panic message. Treating that as a protocol violation
//! and killing the plugin would make a debug `println!` fatal. So an
//! unparseable line becomes [`RpcMessage::NonProtocol`] and is routed to the log
//! ring instead, and the plugin keeps running.
//!
//! The inverse mistake matters more: **a non-protocol line must never be
//! silently discarded**. When a plugin dies mid-handshake, the reason is almost
//! always on stdout — a missing DLL, a bad path, a stack trace — and dropping it
//! leaves the user with "plugin crashed" and nothing else.
//!
//! # The line cap is a denial-of-service bound
//!
//! Reading an unbounded line means a plugin emitting a gigabyte with no newline
//! takes the host's memory with it. [`MAX_LINE_BYTES`] caps it; an over-long line
//! is reported as [`RpcError::LineTooLong`] and the connection is poisoned,
//! because a truncated line cannot be resynchronised — the remainder would parse
//! as a fresh message and the plugin's framing is already untrustworthy.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};

/// Maximum bytes in one protocol line, excluding the newline. 1 MiB.
pub const MAX_LINE_BYTES: usize = 1024 * 1024;

/// Maximum bytes in an event payload the host will forward, per the SDK contract.
pub const MAX_EVENT_BYTES: usize = 64 * 1024;

/// Maximum bytes of a single `log.emit` message. A plugin that logs a megabyte
/// per line fills the ring with one entry and hides everything before it.
pub const MAX_LOG_BYTES: usize = 2 * 1024;

// --- standard JSON-RPC 2.0 error codes ------------------------------------
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

/// A JSON-RPC id. The spec permits a number, a string, or null.
///
/// Modelled rather than assumed-numeric: a plugin written against the spec may
/// legitimately use string ids, and coercing them to numbers would silently
/// break correlation — a reply that never matches its request, presenting as a
/// timeout.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RpcId {
    Num(i64),
    Str(String),
}

/// A classified inbound line.
#[derive(Debug, Clone, PartialEq)]
pub enum RpcMessage {
    /// Has an `id` and a `method`: the peer wants an answer.
    Request {
        id: RpcId,
        method: String,
        params: Value,
    },
    /// Has a `method`, no `id`: fire-and-forget. Replying would be a protocol
    /// error, so the type makes it impossible to confuse with a Request.
    Notification { method: String, params: Value },
    /// A successful reply to something we sent.
    Response { id: RpcId, result: Value },
    /// A failure reply to something we sent.
    ErrorResponse {
        id: RpcId,
        code: i32,
        message: String,
        data: Option<Value>,
    },
    /// Not JSON-RPC at all. Goes to the log ring; never dropped.
    NonProtocol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcError {
    /// A line exceeded [`MAX_LINE_BYTES`]. Unrecoverable: the stream cannot be
    /// resynchronised because the tail would parse as a new message.
    LineTooLong { bytes: usize },
    Io(String),
    /// The peer's stdin/stdout closed.
    Closed,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RpcError::LineTooLong { bytes } => write!(
                f,
                "plugin sent a {bytes}-byte line, over the {MAX_LINE_BYTES}-byte limit; \
                 the stream cannot be resynchronised"
            ),
            RpcError::Io(m) => write!(f, "plugin stdio error: {m}"),
            RpcError::Closed => write!(f, "plugin closed its stdio"),
        }
    }
}

/// Classify one line. Never fails — anything unrecognised is `NonProtocol`.
///
/// A line is treated as protocol only when `jsonrpc` is exactly `"2.0"`. Being
/// strict here is what lets a plugin's ordinary stdout noise coexist with the
/// protocol: a bare `{"hello": 1}` is log output, not a malformed request to
/// reject.
pub fn classify(line: &str) -> RpcMessage {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return RpcMessage::NonProtocol(String::new());
    }
    let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed) else {
        return RpcMessage::NonProtocol(trimmed.to_string());
    };
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return RpcMessage::NonProtocol(trimmed.to_string());
    }

    let id = obj.get("id").and_then(|v| match v {
        Value::Number(n) => n.as_i64().map(RpcId::Num),
        Value::String(s) => Some(RpcId::Str(s.clone())),
        _ => None, // null id — a response to an unidentifiable request
    });

    if let Some(method) = obj.get("method").and_then(Value::as_str) {
        let params = obj.get("params").cloned().unwrap_or(Value::Null);
        return match id {
            Some(id) => RpcMessage::Request {
                id,
                method: method.to_string(),
                params,
            },
            None => RpcMessage::Notification {
                method: method.to_string(),
                params,
            },
        };
    }

    // No method: it is a reply. Without a usable id we cannot correlate it, so it
    // becomes log output rather than being dropped — a `null`-id error reply is
    // usually the most informative thing a broken plugin ever sends.
    let Some(id) = id else {
        return RpcMessage::NonProtocol(trimmed.to_string());
    };

    if let Some(err) = obj.get("error").and_then(Value::as_object) {
        return RpcMessage::ErrorResponse {
            id,
            code: err
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or(INTERNAL_ERROR as i64) as i32,
            message: err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("(no message)")
                .to_string(),
            data: err.get("data").cloned(),
        };
    }

    RpcMessage::Response {
        id,
        // A result of `null` is a valid success in JSON-RPC — absent and null are
        // the same thing here, and both mean "it worked".
        result: obj.get("result").cloned().unwrap_or(Value::Null),
    }
}

/// Read one line, enforcing [`MAX_LINE_BYTES`].
///
/// Reads bytewise rather than using `read_line` so the cap applies to what has
/// been buffered so far, not to a line already fully in memory. `read_line` on a
/// peer that never sends `\n` allocates without bound.
pub fn read_line_capped<R: BufRead>(reader: &mut R) -> Result<String, RpcError> {
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    loop {
        let available = match reader.fill_buf() {
            Ok(b) if b.is_empty() => {
                return if buf.is_empty() {
                    Err(RpcError::Closed)
                } else {
                    Ok(String::from_utf8_lossy(&buf).into_owned())
                };
            }
            Ok(b) => b,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(RpcError::Io(e.to_string())),
        };

        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            if buf.len() + pos > MAX_LINE_BYTES {
                let bytes = buf.len() + pos;
                reader.consume(pos + 1);
                return Err(RpcError::LineTooLong { bytes });
            }
            buf.extend_from_slice(&available[..pos]);
            reader.consume(pos + 1);
            // Tolerate CRLF: a plugin on Windows writing with a text-mode stream
            // emits \r\n, and a trailing \r makes the JSON parse fail for a
            // reason nobody would guess from the error.
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            return Ok(String::from_utf8_lossy(&buf).into_owned());
        }

        let take = available.len();
        if buf.len() + take > MAX_LINE_BYTES {
            let bytes = buf.len() + take;
            reader.consume(take);
            return Err(RpcError::LineTooLong { bytes });
        }
        buf.extend_from_slice(available);
        reader.consume(take);
    }
}

/// Tracks outbound requests awaiting replies, and mints ids.
///
/// Ids start at 1 and only increase. Reusing an id would let a late reply to a
/// timed-out request satisfy a *newer* request — one call's result delivered as
/// another's, which is worse than a timeout because it looks like success.
#[derive(Debug, Default)]
pub struct RpcPeer {
    next_id: i64,
    pending: HashMap<RpcId, String>,
}

impl RpcPeer {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            pending: HashMap::new(),
        }
    }

    /// Build a request line and record it as pending.
    pub fn request(&mut self, method: &str, params: Value) -> (RpcId, String) {
        let id = RpcId::Num(self.next_id);
        self.next_id += 1;
        self.pending.insert(id.clone(), method.to_string());
        let line = format!(
            "{}\n",
            json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
        );
        (id, line)
    }

    /// Match a reply to its request, returning the method it answered.
    ///
    /// `None` means the peer replied to something we never sent, or replied
    /// twice. Either is a protocol violation worth logging rather than acting on.
    pub fn resolve(&mut self, id: &RpcId) -> Option<String> {
        self.pending.remove(id)
    }

    /// Give up on a request. Called on timeout so the map cannot grow without
    /// bound across a long-lived plugin's lifetime.
    pub fn abandon(&mut self, id: &RpcId) -> Option<String> {
        self.pending.remove(id)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn pending_methods(&self) -> Vec<String> {
        let mut v: Vec<String> = self.pending.values().cloned().collect();
        v.sort();
        v
    }
}

/// A notification line (no id — the peer must not reply).
pub fn notification_line(method: &str, params: Value) -> String {
    format!(
        "{}\n",
        json!({ "jsonrpc": "2.0", "method": method, "params": params })
    )
}

/// A success reply to a plugin-initiated request.
pub fn result_line(id: &RpcId, result: Value) -> String {
    format!("{}\n", json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

/// An error reply to a plugin-initiated request.
///
/// `data` carries the typed code from `protocol/v1/error.schema.json`, so a
/// plugin can branch on `capability_denied` vs `runtime_unavailable` without
/// parsing prose — the same reason the protocol's taxonomy is closed.
pub fn error_line(id: &RpcId, code: i32, message: &str, typed: Option<&str>) -> String {
    let mut err = json!({ "code": code, "message": message });
    if let Some(t) = typed {
        err["data"] = json!({ "code": t });
    }
    format!("{}\n", json!({ "jsonrpc": "2.0", "id": id, "error": err }))
}

/// Write a prepared line and flush.
///
/// The flush is not optional. A plugin's stdin is a pipe, and an unflushed
/// request sits in the host's buffer while the host waits for a reply that
/// cannot come — presenting as the plugin hanging.
pub fn write_line<W: Write>(w: &mut W, line: &str) -> Result<(), RpcError> {
    w.write_all(line.as_bytes())
        .map_err(|e| RpcError::Io(e.to_string()))?;
    w.flush().map_err(|e| RpcError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // --- classification ---------------------------------------------------

    #[test]
    fn a_request_has_an_id_and_a_method() {
        let m = classify(r#"{"jsonrpc":"2.0","id":7,"method":"flow.run","params":{"flowId":"f"}}"#);
        match m {
            RpcMessage::Request { id, method, params } => {
                assert_eq!(id, RpcId::Num(7));
                assert_eq!(method, "flow.run");
                assert_eq!(params["flowId"], "f");
            }
            other => panic!("expected Request, got {other:?}"),
        }
    }

    #[test]
    fn a_notification_has_no_id() {
        let m = classify(r#"{"jsonrpc":"2.0","method":"log.emit","params":{"level":"info"}}"#);
        assert!(
            matches!(m, RpcMessage::Notification { ref method, .. } if method == "log.emit"),
            "{m:?}"
        );
    }

    #[test]
    fn string_ids_are_preserved() {
        // Spec-legal. Coercing to a number would break correlation, presenting
        // as a timeout on a reply that did arrive.
        let m = classify(r#"{"jsonrpc":"2.0","id":"abc-1","method":"plugin.health"}"#);
        assert!(matches!(m, RpcMessage::Request { id: RpcId::Str(ref s), .. } if s == "abc-1"), "{m:?}");
    }

    #[test]
    fn a_success_reply_is_a_response() {
        let m = classify(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#);
        assert!(matches!(m, RpcMessage::Response { id: RpcId::Num(1), .. }), "{m:?}");
    }

    #[test]
    fn a_null_result_is_still_success() {
        // JSON-RPC allows it, and absent-vs-null both mean "it worked".
        let m = classify(r#"{"jsonrpc":"2.0","id":1,"result":null}"#);
        assert!(matches!(m, RpcMessage::Response { .. }), "{m:?}");
        let m = classify(r#"{"jsonrpc":"2.0","id":1}"#);
        assert!(matches!(m, RpcMessage::Response { .. }), "{m:?}");
    }

    #[test]
    fn an_error_reply_carries_its_typed_code() {
        let m = classify(
            r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"nope","data":{"code":"capability_denied"}}}"#,
        );
        match m {
            RpcMessage::ErrorResponse { id, code, message, data } => {
                assert_eq!(id, RpcId::Num(3));
                assert_eq!(code, METHOD_NOT_FOUND);
                assert_eq!(message, "nope");
                assert_eq!(data.unwrap()["code"], "capability_denied");
            }
            other => panic!("expected ErrorResponse, got {other:?}"),
        }
    }

    // --- non-protocol output ----------------------------------------------

    #[test]
    fn stray_stdout_becomes_log_output_not_an_error() {
        // A stray println! must not be fatal.
        for line in [
            "Listening on COM3",
            "  * Serving Flask app",
            "Traceback (most recent call last):",
            "{ not json",
            "[]",
            "42",
        ] {
            assert!(
                matches!(classify(line), RpcMessage::NonProtocol(_)),
                "{line:?} should be NonProtocol"
            );
        }
    }

    #[test]
    fn json_without_the_jsonrpc_marker_is_log_output() {
        // Strictness here is what lets ordinary JSON logging coexist with the
        // protocol instead of being rejected as a malformed request.
        let m = classify(r#"{"hello":1,"id":4,"method":"whatever"}"#);
        assert!(matches!(m, RpcMessage::NonProtocol(_)), "{m:?}");
    }

    #[test]
    fn a_wrong_jsonrpc_version_is_log_output() {
        let m = classify(r#"{"jsonrpc":"1.0","id":1,"method":"x"}"#);
        assert!(matches!(m, RpcMessage::NonProtocol(_)), "{m:?}");
    }

    #[test]
    fn a_null_id_reply_is_preserved_as_log_output() {
        // The most informative thing a broken plugin sends is often exactly this,
        // so it must not be dropped.
        let raw = r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}}"#;
        match classify(raw) {
            RpcMessage::NonProtocol(s) => assert!(s.contains("parse error"), "{s}"),
            other => panic!("expected NonProtocol, got {other:?}"),
        }
    }

    // --- the line cap -----------------------------------------------------

    #[test]
    fn a_normal_line_reads_back() {
        let mut c = Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1}\n".to_vec());
        assert_eq!(read_line_capped(&mut c).unwrap(), r#"{"jsonrpc":"2.0","id":1}"#);
    }

    #[test]
    fn crlf_is_tolerated() {
        // A Windows plugin using a text-mode stream emits \r\n; a trailing \r
        // makes the JSON parse fail for a reason nobody would guess.
        let mut c = Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1}\r\n".to_vec());
        let line = read_line_capped(&mut c).unwrap();
        assert!(!line.ends_with('\r'), "{line:?}");
        assert!(matches!(classify(&line), RpcMessage::Response { .. }));
    }

    #[test]
    fn an_oversized_line_is_refused_not_buffered() {
        let mut data = vec![b'x'; MAX_LINE_BYTES + 10];
        data.push(b'\n');
        let mut c = Cursor::new(data);
        match read_line_capped(&mut c) {
            Err(RpcError::LineTooLong { bytes }) => assert!(bytes > MAX_LINE_BYTES),
            other => panic!("expected LineTooLong, got {other:?}"),
        }
    }

    #[test]
    fn an_unterminated_flood_is_capped() {
        // No newline at all: `read_line` would allocate without bound.
        let mut c = Cursor::new(vec![b'y'; MAX_LINE_BYTES * 2]);
        assert!(matches!(
            read_line_capped(&mut c),
            Err(RpcError::LineTooLong { .. })
        ));
    }

    #[test]
    fn a_line_exactly_at_the_cap_is_accepted() {
        let mut data = vec![b'z'; MAX_LINE_BYTES];
        data.push(b'\n');
        let mut c = Cursor::new(data);
        assert_eq!(read_line_capped(&mut c).unwrap().len(), MAX_LINE_BYTES);
    }

    #[test]
    fn a_closed_stream_is_reported_as_closed() {
        let mut c = Cursor::new(Vec::new());
        assert_eq!(read_line_capped(&mut c).unwrap_err(), RpcError::Closed);
    }

    #[test]
    fn a_final_line_without_a_newline_is_still_returned() {
        // A plugin that exits right after writing its last line should not have
        // that line — often the reason it exited — thrown away.
        let mut c = Cursor::new(b"goodbye".to_vec());
        assert_eq!(read_line_capped(&mut c).unwrap(), "goodbye");
    }

    #[test]
    fn multiple_lines_read_in_order() {
        let mut c = Cursor::new(b"one\ntwo\nthree\n".to_vec());
        assert_eq!(read_line_capped(&mut c).unwrap(), "one");
        assert_eq!(read_line_capped(&mut c).unwrap(), "two");
        assert_eq!(read_line_capped(&mut c).unwrap(), "three");
        assert_eq!(read_line_capped(&mut c).unwrap_err(), RpcError::Closed);
    }

    // --- correlation ------------------------------------------------------

    #[test]
    fn a_reply_resolves_to_the_method_it_answered() {
        let mut p = RpcPeer::new();
        let (id, line) = p.request("plugin.init", json!({"dataDir": "/tmp"}));
        assert!(line.ends_with('\n'), "lines must be newline-terminated");
        assert!(line.contains("\"jsonrpc\":\"2.0\""));
        assert_eq!(p.pending_count(), 1);
        assert_eq!(p.resolve(&id).as_deref(), Some("plugin.init"));
        assert_eq!(p.pending_count(), 0);
    }

    #[test]
    fn an_unsolicited_reply_resolves_to_nothing() {
        let mut p = RpcPeer::new();
        assert!(p.resolve(&RpcId::Num(999)).is_none());
    }

    #[test]
    fn a_duplicate_reply_resolves_only_once() {
        let mut p = RpcPeer::new();
        let (id, _) = p.request("plugin.health", json!({}));
        assert!(p.resolve(&id).is_some());
        assert!(
            p.resolve(&id).is_none(),
            "a second reply to one request must not resolve again"
        );
    }

    #[test]
    fn ids_are_never_reused() {
        // Otherwise a late reply to a timed-out request satisfies a NEWER one:
        // one call's result delivered as another's, which looks like success.
        let mut p = RpcPeer::new();
        let (first, _) = p.request("a", json!({}));
        p.abandon(&first);
        let (second, _) = p.request("b", json!({}));
        assert_ne!(first, second);

        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let (id, _) = p.request("x", json!({}));
            assert!(seen.insert(id), "an id was reused");
        }
    }

    #[test]
    fn abandoning_bounds_the_pending_map() {
        let mut p = RpcPeer::new();
        let ids: Vec<_> = (0..50).map(|_| p.request("slow", json!({})).0).collect();
        assert_eq!(p.pending_count(), 50);
        for id in &ids {
            p.abandon(id);
        }
        assert_eq!(p.pending_count(), 0, "timeouts must not leak entries");
    }

    #[test]
    fn pending_methods_are_reportable() {
        // So a stuck plugin's diagnostics can say WHAT is stuck.
        let mut p = RpcPeer::new();
        p.request("plugin.health", json!({}));
        p.request("connector.request", json!({}));
        assert_eq!(
            p.pending_methods(),
            vec!["connector.request".to_string(), "plugin.health".to_string()]
        );
    }

    // --- outbound framing -------------------------------------------------

    #[test]
    fn a_notification_carries_no_id() {
        let line = notification_line("event.ack", json!({"idempotencyKey": "k"}));
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert!(v.get("id").is_none(), "a notification must not be answerable");
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn a_result_line_round_trips() {
        let line = result_line(&RpcId::Num(4), json!({"runId": "run_1", "status": "queued"}));
        match classify(line.trim()) {
            RpcMessage::Response { id, result } => {
                assert_eq!(id, RpcId::Num(4));
                assert_eq!(result["runId"], "run_1");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_error_line_carries_the_typed_code() {
        let line = error_line(
            &RpcId::Num(5),
            METHOD_NOT_FOUND,
            "this plugin did not declare oaiy.flow.run",
            Some("capability_denied"),
        );
        match classify(line.trim()) {
            RpcMessage::ErrorResponse { code, data, message, .. } => {
                assert_eq!(code, METHOD_NOT_FOUND);
                assert_eq!(data.unwrap()["code"], "capability_denied");
                assert!(message.contains("oaiy.flow.run"), "name the missing capability");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_error_line_without_a_typed_code_omits_data() {
        let line = error_line(&RpcId::Num(6), INVALID_PARAMS, "bad params", None);
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert!(v["error"].get("data").is_none());
    }

    #[test]
    fn write_line_flushes() {
        // An unflushed request sits in our buffer while we await a reply that
        // cannot come — presenting as the plugin hanging.
        struct CountingWriter {
            buf: Vec<u8>,
            flushes: usize,
        }
        impl Write for CountingWriter {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.buf.extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.flushes += 1;
                Ok(())
            }
        }
        let mut w = CountingWriter { buf: Vec::new(), flushes: 0 };
        write_line(&mut w, "hello\n").unwrap();
        assert_eq!(w.flushes, 1);
        assert_eq!(w.buf, b"hello\n");
    }

    #[test]
    fn a_broken_pipe_is_reported_not_panicked() {
        struct BrokenWriter;
        impl Write for BrokenWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "gone"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(matches!(
            write_line(&mut BrokenWriter, "x\n"),
            Err(RpcError::Io(_))
        ));
    }

    #[test]
    fn errors_describe_themselves_usefully() {
        let msg = RpcError::LineTooLong { bytes: 2_000_000 }.to_string();
        assert!(msg.contains("2000000"), "{msg}");
        assert!(msg.contains("resynchronised"), "say why it is fatal: {msg}");
    }
}
