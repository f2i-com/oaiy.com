//! The audit log: `<data>/control-log.jsonl`, one line per change-tool call,
//! `{at, tool, args, session, ok, summary}`.
//!
//! The arguments are kept as they were asked, with secret-looking ones
//! (keys, tokens, passwords) redacted and long ones shortened: a flow's whole
//! graph does not belong in a log a dashboard reads back. The file rolls at
//! [`MAX_BYTES`], keeping one previous file, and is read newest first.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Map, Value};

/// Roll at 2 MiB, keeping one previous file.
pub const MAX_BYTES: u64 = 2 * 1024 * 1024;
/// The most entries one read returns.
pub const MAX_READ: usize = 1000;
/// What a redacted value reads as.
pub const REDACTED: &str = "[redacted]";

/// Strings longer than this are shortened.
const MAX_STRING: usize = 200;
/// Lists longer than this are shortened.
const MAX_LIST: usize = 20;
/// Objects nested deeper than this are elided.
const MAX_DEPTH: usize = 5;
/// The arguments of one entry, at most, once shortened.
const MAX_ARGS_BYTES: usize = 4096;

/// The log file, one writer at a time.
pub struct Log {
    path: PathBuf,
    lock: Mutex<()>,
}

impl Log {
    pub fn new(path: PathBuf) -> Log {
        Log { path, lock: Mutex::new(()) }
    }

    fn previous(&self) -> PathBuf {
        self.path.with_extension("jsonl.1")
    }

    /// Add one entry. The arguments are redacted here, so nothing unredacted is ever written.
    pub fn append(&self, tool: &str, args: &Value, session: &str, ok: bool, summary: &str) {
        let entry = json!({
            "at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            "tool": tool,
            "args": compact_args(args),
            "session": session,
            "ok": ok,
            "summary": shorten(summary, 400),
        });
        let _one = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0) >= MAX_BYTES {
            let _ = std::fs::rename(&self.path, self.previous());
        }
        match std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            Ok(mut f) => {
                if let Err(e) = writeln!(f, "{entry}") {
                    log::warn!("control: the audit log could not be written: {e}");
                }
            }
            Err(e) => log::warn!("control: the audit log could not be opened: {e}"),
        }
    }

    /// The newest `limit` entries, newest first (from the previous file too when needed).
    pub fn read(&self, limit: usize) -> Vec<Value> {
        let _one = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for path in [self.path.clone(), self.previous()] {
            if out.len() >= limit {
                break;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            for line in text.lines().rev() {
                if out.len() >= limit {
                    break;
                }
                // A line cut short by a crash is skipped, not a reason to lose the rest.
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    out.push(v);
                }
            }
        }
        out
    }
}

/// Whether an argument's name says it holds a secret.
pub fn secret_name(name: &str) -> bool {
    let n: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    const WORDS: [&str; 11] = [
        "password", "passwd", "passphrase", "secret", "token", "apikey", "privatekey", "credential", "authorization", "cookie", "bearer",
    ];
    WORDS.iter().any(|w| n.contains(w)) || n.ends_with("key") || n == "pin" || n == "otp"
}

/// Whether a value looks like a secret whatever it is called.
pub fn secret_value(s: &str) -> bool {
    let s = s.trim();
    const PREFIXES: [&str; 8] = ["sk-", "bearer ", "ghp_", "gho_", "github_pat_", "hf_", "oaiypat_", "xox"];
    let lower = s.to_ascii_lowercase();
    PREFIXES.iter().any(|p| lower.starts_with(p) && s.len() > p.len() + 8)
}

/// `value` with secret-looking parts replaced by [`REDACTED`]. Nothing is shortened.
pub fn redact(value: &Value) -> Value {
    match value {
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), if secret_name(k) && !v.is_null() { Value::String(REDACTED.into()) } else { redact(v) }))
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(redact).collect()),
        Value::String(s) if secret_value(s) => Value::String(REDACTED.into()),
        other => other.clone(),
    }
}

fn shorten(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        s.to_string()
    } else {
        format!("{}… ({n} characters)", s.chars().take(max).collect::<String>())
    }
}

fn compact(value: &Value, depth: usize) -> Value {
    match value {
        Value::Object(o) if depth >= MAX_DEPTH => Value::String(format!("{{… {} fields}}", o.len())),
        Value::Array(a) if depth >= MAX_DEPTH => Value::String(format!("[… {} items]", a.len())),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), compact(v, depth + 1))).collect()),
        Value::Array(a) => {
            let mut out: Vec<Value> = a.iter().take(MAX_LIST).map(|v| compact(v, depth + 1)).collect();
            if a.len() > MAX_LIST {
                out.push(Value::String(format!("… {} more", a.len() - MAX_LIST)));
            }
            Value::Array(out)
        }
        Value::String(s) => Value::String(shorten(s, MAX_STRING)),
        other => other.clone(),
    }
}

/// The arguments as the log keeps them: redacted, then shortened; and when
/// still too long, only which arguments there were.
pub fn compact_args(args: &Value) -> Value {
    let short = compact(&redact(args), 0);
    if short.to_string().len() <= MAX_ARGS_BYTES {
        return short;
    }
    match short {
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, v) in o {
                let brief = match v {
                    Value::Object(inner) => Value::String(format!("{{… {} fields}}", inner.len())),
                    Value::Array(inner) => Value::String(format!("[… {} items]", inner.len())),
                    other => other,
                };
                out.insert(k, brief);
            }
            Value::Object(out)
        }
        other => Value::String(shorten(&other.to_string(), MAX_STRING)),
    }
}
