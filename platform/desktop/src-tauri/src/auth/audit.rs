//! The audit log and the noise log (design 4.12).
//!
//! Two files, so that anonymous traffic can never wash evidence out of the record of what happened:
//!
//! - `<data>/auth/audit.jsonl` records what a person did and what the server decided. Nothing an
//!   anonymous request can cause is written here. It rolls at 5 MiB and keeps 8 files.
//! - `<data>/auth/noise.jsonl` records anonymous and high-rate events, aggregated: one line per
//!   `(event, client address, minute)` with a count. It rolls at 2 MiB and keeps 2 files.
//!
//! Every line is redacted with the same functions as the control tools' log
//! (`control::audit::redact`); nothing secret is written (no token, password, code or CSRF value), the
//! user agent is cut to 120 characters and the client address is kept (it is the operator's own
//! server). The same events go to stderr as one structured line each when asked, so `journalctl`, a
//! log shipper or `fail2ban` can see a login attack.
//!
//! Other designs write through [`critical`] with a namespaced event (`relay.*`, `device.*`,
//! `vault.*`, `mobile.*`): this is the single writer.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use serde_json::{json, Value};

use super::clock::Clock;
use super::principal::Actor;
use crate::control::audit::{compact_args, redact};

pub const AUDIT_MAX_BYTES: u64 = 5 * 1024 * 1024;
/// `audit.jsonl` and seven rolled files.
pub const AUDIT_FILES: usize = 8;
pub const NOISE_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// `noise.jsonl` and one rolled file.
pub const NOISE_FILES: usize = 2;
/// The user agent is cut to this many characters.
pub const MAX_UA: usize = 120;
/// Most `(event, address, minute)` buckets held before the rest are counted together: a flood from
/// many rotating addresses cannot grow memory without bound.
pub const MAX_NOISE_BUCKETS: usize = 10_000;
/// The address bucket that takes the counts past [`MAX_NOISE_BUCKETS`].
pub const OVERFLOW_IP: &str = "*";
/// The most one read returns.
pub const MAX_READ: usize = 1000;

/// Which of the two files a read wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogFile {
    Audit,
    Noise,
}

/// Who and where, when the caller knows.
#[derive(Clone, Copy, Debug, Default)]
pub struct Context<'a> {
    pub ip: Option<&'a str>,
    pub host: Option<&'a str>,
    pub ua: Option<&'a str>,
}

/// A file that rolls: `name`, then `name.1` (the newest rolled) up to `name.<keep-1>`.
struct Rolling {
    path: PathBuf,
    max_bytes: u64,
    keep: usize,
}

impl Rolling {
    fn rolled(&self, n: usize) -> PathBuf {
        let mut name = self
            .path
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        name.push(format!(".{n}"));
        self.path.with_file_name(name)
    }

    /// Every file, newest first.
    fn files(&self) -> Vec<PathBuf> {
        let mut out = vec![self.path.clone()];
        out.extend((1..self.keep).map(|n| self.rolled(n)));
        out
    }

    fn roll(&self) {
        // Drop the oldest, shift the rest up, and the live file becomes `.1`.
        let _ = std::fs::remove_file(self.rolled(self.keep - 1));
        for n in (1..self.keep - 1).rev() {
            let _ = std::fs::rename(self.rolled(n), self.rolled(n + 1));
        }
        if self.keep > 1 {
            let _ = std::fs::rename(&self.path, self.rolled(1));
        } else {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn append(&self, line: &str) -> std::io::Result<()> {
        self.append_many(&[line.to_string()])
    }

    /// Append lines with one open of the file. The file rolls before the batch, so it can pass its
    /// size by one batch (which is bounded by what the noise aggregation holds).
    fn append_many(&self, lines: &[String]) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent() {
            crate::secret_file::create_private_dir(dir)?;
        }
        if std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0) >= self.max_bytes {
            self.roll();
        }
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        // A line cut short by a crash has no newline: start this one on a line of its own.
        let torn = ends_mid_line(&self.path);
        let mut file = options.open(&self.path)?;
        let mut out = std::io::BufWriter::new(&mut file);
        if torn {
            out.write_all(b"\n")?;
        }
        for line in lines {
            out.write_all(line.as_bytes())?;
            out.write_all(b"\n")?;
        }
        out.flush()
    }
}

/// Whether the file has bytes and the last one is not a newline.
fn ends_mid_line(path: &Path) -> bool {
    use std::io::{Read as _, Seek as _, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(len) = f.metadata().map(|m| m.len()) else {
        return false;
    };
    if len == 0 || f.seek(SeekFrom::End(-1)).is_err() {
        return false;
    }
    let mut last = [0u8; 1];
    f.read_exact(&mut last).is_ok() && last[0] != b'\n'
}

#[derive(Default)]
struct Bucket {
    count: u64,
    first_ms: u64,
    last_ms: u64,
    host: String,
}

/// What one parent has derived in the minute it is in: the derives after the first, counted.
struct DerivedTally {
    minute: u64,
    extra: u64,
    who: Actor,
    ip: Option<String>,
}

#[derive(Default)]
struct NoiseState {
    /// `(event, address, minute start)`.
    buckets: HashMap<(String, String, u64), Bucket>,
    /// The minute the newest event fell in: buckets of an earlier minute are closed.
    minute: u64,
}

/// The two logs of one data folder.
pub struct AuditLog {
    audit: Mutex<Rolling>,
    noise_file: Mutex<Rolling>,
    noise: Mutex<NoiseState>,
    /// By the id of the parent that derives.
    derived: Mutex<HashMap<String, DerivedTally>>,
    clock: Arc<dyn Clock>,
    stderr: bool,
}

fn iso(ms: u64) -> String {
    chrono::DateTime::from_timestamp_millis(ms as i64)
        .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_string())
}

fn cut(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// A field of a structured stderr line must not be able to add another: no whitespace, no `=`.
fn stderr_field(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_whitespace() || c == '=' || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .take(64)
        .collect()
}

impl AuditLog {
    /// The logs in `dir` (`<data>/auth`).
    pub fn open(dir: &Path, clock: Arc<dyn Clock>, stderr: bool) -> AuditLog {
        Self::with_limits(dir, clock, stderr, AUDIT_MAX_BYTES, NOISE_MAX_BYTES)
    }

    /// As [`AuditLog::open`], with other sizes at which the files roll (for the tests).
    pub fn with_limits(
        dir: &Path,
        clock: Arc<dyn Clock>,
        stderr: bool,
        audit_max: u64,
        noise_max: u64,
    ) -> AuditLog {
        AuditLog {
            audit: Mutex::new(Rolling {
                path: dir.join("audit.jsonl"),
                max_bytes: audit_max,
                keep: AUDIT_FILES,
            }),
            noise_file: Mutex::new(Rolling {
                path: dir.join("noise.jsonl"),
                max_bytes: noise_max,
                keep: NOISE_FILES,
            }),
            noise: Mutex::new(NoiseState::default()),
            derived: Mutex::new(HashMap::new()),
            clock,
            stderr,
        }
    }

    /// Record what a person did or what the server decided.
    pub fn critical(
        &self,
        event: &str,
        principal: Option<&Actor>,
        ctx: &Context<'_>,
        detail: Value,
    ) {
        let now = self.clock.now_ms();
        let line = json!({
            "at": iso(now),
            "event": event,
            "principal": principal,
            "ip": ctx.ip,
            "host": ctx.host,
            "ua": ctx.ua.map(|u| cut(u, MAX_UA)),
            "detail": compact_args(&redact(&detail)),
        });
        if self.stderr {
            eprintln!(
                "oaiy-audit event={} principal={} ip={}",
                stderr_field(event),
                stderr_field(principal.map_or("-", |p| p.id.as_str())),
                stderr_field(ctx.ip.unwrap_or("-"))
            );
        }
        let rolling = self.audit.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = rolling.append(&line.to_string()) {
            log::warn!("auth: the audit log could not be written: {e}");
        }
    }

    /// A derived credential was made (`credential.created`, derived), by `parent`. Thirty a minute for a
    /// day would be 43,200 lines a day and wash the record of what people did out of a log that keeps
    /// 40 MiB: the first derive of a parent in a minute is written in full, and the rest of that minute
    /// are counted and written as one `credential.derived_more` line when the minute is over (by the next
    /// derive of that parent, [`AuditLog::flush_closed`] or [`AuditLog::flush_noise`]).
    pub fn derived(&self, parent: &Actor, ctx: &Context<'_>, detail: Value) {
        let now = self.clock.now_ms();
        let minute = now - now % 60_000;
        let mut over = None;
        {
            let mut tallies = self.derived.lock().unwrap_or_else(|e| e.into_inner());
            match tallies.get_mut(&parent.id) {
                Some(t) if t.minute == minute => {
                    t.extra += 1;
                    return;
                }
                Some(t) => {
                    if t.extra > 0 {
                        over = Some((t.who.clone(), t.minute, t.extra, t.ip.clone()));
                    }
                    t.minute = minute;
                    t.extra = 0;
                    t.ip = ctx.ip.map(str::to_string);
                }
                None => {
                    tallies.insert(
                        parent.id.clone(),
                        DerivedTally {
                            minute,
                            extra: 0,
                            who: parent.clone(),
                            ip: ctx.ip.map(str::to_string),
                        },
                    );
                }
            }
        }
        if let Some((who, minute, count, ip)) = over {
            self.write_derived_more(&who, minute, count, ip.as_deref());
        }
        self.critical("credential.created", Some(parent), ctx, detail);
    }

    fn write_derived_more(&self, who: &Actor, minute: u64, count: u64, ip: Option<&str>) {
        self.critical(
            "credential.derived_more",
            Some(who),
            &Context {
                ip,
                host: None,
                ua: None,
            },
            json!({ "derived": true, "count": count, "minute": iso(minute) }),
        );
    }

    /// Write the counts of derives whose minute is over (`all`: every one, at shutdown).
    fn flush_derived(&self, all: bool) {
        let now = self.clock.now_ms();
        let minute = now - now % 60_000;
        let due: Vec<DerivedTally> = {
            let mut tallies = self.derived.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<String> = tallies
                .iter()
                .filter(|(_, t)| all || t.minute < minute)
                .map(|(k, _)| k.clone())
                .collect();
            keys.into_iter()
                .filter_map(|k| tallies.remove(&k))
                .collect()
        };
        for t in due {
            if t.extra > 0 {
                self.write_derived_more(&t.who, t.minute, t.extra, t.ip.as_deref());
            }
        }
    }

    /// Count an anonymous or high-rate event. One line per `(event, address, minute)` is written once
    /// the minute has passed (by a later event, or [`AuditLog::flush_noise`]).
    pub fn noise(&self, event: &str, ip: &str, host: &str) {
        let now = self.clock.now_ms();
        let minute = now - now % 60_000;
        let mut closed = Vec::new();
        {
            let mut state = self.noise.lock().unwrap_or_else(|e| e.into_inner());
            let overflowing = state.buckets.len() >= MAX_NOISE_BUCKETS;
            let key = (event.to_string(), ip.to_string(), minute);
            let key = if overflowing && !state.buckets.contains_key(&key) {
                (event.to_string(), OVERFLOW_IP.to_string(), minute)
            } else {
                key
            };
            let bucket = state.buckets.entry(key).or_insert_with(|| Bucket {
                first_ms: now,
                host: cut(host, 255),
                ..Bucket::default()
            });
            bucket.count += 1;
            bucket.last_ms = now;
            // Once a minute, not once an event: a flood must not cost a scan of every bucket each time.
            if minute > state.minute {
                state.minute = minute;
                let stale: Vec<_> = state
                    .buckets
                    .keys()
                    .filter(|(_, _, m)| *m < minute)
                    .cloned()
                    .collect();
                for k in stale {
                    if let Some(b) = state.buckets.remove(&k) {
                        closed.push((k, b));
                    }
                }
            }
        }
        self.write_noise(closed);
    }

    /// Write the buckets of minutes that are over (the periodic upkeep of a running server).
    pub fn flush_closed(&self) {
        self.flush_derived(false);
        let now = self.clock.now_ms();
        let minute = now - now % 60_000;
        let closed: Vec<_> = {
            let mut state = self.noise.lock().unwrap_or_else(|e| e.into_inner());
            let stale: Vec<_> = state
                .buckets
                .keys()
                .filter(|(_, _, m)| *m < minute)
                .cloned()
                .collect();
            stale
                .into_iter()
                .filter_map(|k| state.buckets.remove(&k).map(|b| (k, b)))
                .collect()
        };
        self.write_noise(closed);
    }

    /// Write every bucket, closed or not (at shutdown, and for the tests).
    pub fn flush_noise(&self) {
        self.flush_derived(true);
        let closed: Vec<_> = {
            let mut state = self.noise.lock().unwrap_or_else(|e| e.into_inner());
            state.buckets.drain().collect()
        };
        self.write_noise(closed);
    }

    fn write_noise(&self, mut closed: Vec<((String, String, u64), Bucket)>) {
        if closed.is_empty() {
            return;
        }
        closed.sort_by(|a, b| a.0.cmp(&b.0));
        let mut lines = Vec::with_capacity(closed.len());
        for ((event, ip, minute), b) in closed {
            let line = json!({
                "at": iso(minute),
                "event": event,
                "ip": ip,
                "host": b.host,
                "count": b.count,
                "first_ms": b.first_ms,
                "last_ms": b.last_ms,
            });
            if self.stderr {
                eprintln!(
                    "oaiy-audit event={} ip={} count={} window=60s",
                    stderr_field(&event),
                    stderr_field(&ip),
                    b.count
                );
            }
            lines.push(line.to_string());
        }
        let rolling = self.noise_file.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = rolling.append_many(&lines) {
            log::warn!("auth: the noise log could not be written: {e}");
        }
    }

    /// The newest lines first: at most `limit` (never over [`MAX_READ`]), only those older than
    /// `before_ms` when given, and only those of `principal` (an id) when given.
    pub fn read(
        &self,
        file: LogFile,
        limit: usize,
        before_ms: Option<u64>,
        principal: Option<&str>,
    ) -> Vec<Value> {
        let limit = limit.min(MAX_READ);
        let files = match file {
            LogFile::Audit => self.audit.lock().unwrap_or_else(|e| e.into_inner()).files(),
            LogFile::Noise => self
                .noise_file
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .files(),
        };
        let mut out = Vec::new();
        for path in files {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for line in text.lines().rev() {
                if out.len() >= limit {
                    return out;
                }
                // A line cut short by a crash is skipped, not a reason to lose the rest.
                let Ok(v) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                if let Some(before) = before_ms {
                    let at = v["at"]
                        .as_str()
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|t| t.timestamp_millis());
                    if !at.is_some_and(|t| (t as u64) < before) {
                        continue;
                    }
                }
                if principal.is_some_and(|p| v["principal"]["id"].as_str() != Some(p)) {
                    continue;
                }
                out.push(v);
            }
        }
        out
    }
}

// ---- the single writer other designs use ------------------------------------------------------

fn global() -> &'static RwLock<Option<Arc<AuditLog>>> {
    static GLOBAL: OnceLock<RwLock<Option<Arc<AuditLog>>>> = OnceLock::new();
    GLOBAL.get_or_init(|| RwLock::new(None))
}

/// Make `log` the one [`critical`] and [`noise`] write to.
pub fn use_log(log: Arc<AuditLog>) {
    *global().write().unwrap_or_else(|e| e.into_inner()) = Some(log);
}

/// Stop writing to the installed log (at shutdown, and between tests).
pub fn uninstall() {
    *global().write().unwrap_or_else(|e| e.into_inner()) = None;
}

fn installed() -> Option<Arc<AuditLog>> {
    global().read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// `audit::critical(event, principal, detail)`: the single writer of `audit.jsonl`. Without an
/// installed log (a build that has no auth folder) it writes nothing.
pub fn critical(event: &str, principal: Option<&Actor>, detail: Value) {
    if let Some(log) = installed() {
        log.critical(event, principal, &Context::default(), detail);
    }
}

/// `audit::noise(event, ip, host)`: the single writer of `noise.jsonl`.
pub fn noise(event: &str, ip: &str, host: &str) {
    if let Some(log) = installed() {
        log.noise(event, ip, host);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::clock::ManualClock;
    use crate::secret_file::testing::TempDir;

    const TOKEN: &str = "oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    const T0: u64 = 1_790_000_000_000;

    fn log(dir: &TempDir, clock: &Arc<ManualClock>) -> AuditLog {
        AuditLog::open(&dir.0.join("auth"), clock.clone(), false)
    }

    fn actor() -> Actor {
        Actor {
            id: "0123456789abcdef".into(),
            kind: "session".into(),
            label: "dashboard".into(),
        }
    }

    fn lines(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn a_critical_event_is_one_json_line_in_the_shape_the_design_gives() {
        let dir = TempDir::new("audit-shape");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        let ctx = Context {
            ip: Some("203.0.113.9"),
            host: Some("dash.example.com"),
            ua: Some("Mozilla/5.0 (X11; Linux x86_64)"),
        };
        log.critical("login.ok", Some(&actor()), &ctx, json!({ "device": true }));
        let got = lines(&dir.0.join("auth").join("audit.jsonl"));
        assert_eq!(got.len(), 1);
        let l = &got[0];
        assert_eq!(l["event"], "login.ok");
        assert_eq!(
            l["principal"],
            json!({ "id": "0123456789abcdef", "kind": "session", "label": "dashboard" })
        );
        assert_eq!(l["ip"], "203.0.113.9");
        assert_eq!(l["host"], "dash.example.com");
        assert_eq!(l["ua"], "Mozilla/5.0 (X11; Linux x86_64)");
        assert_eq!(l["detail"], json!({ "device": true }));
        assert_eq!(l["at"], "2026-09-21T14:13:20.000Z");
    }

    fn derive_actor(id: &str) -> Actor {
        Actor {
            id: id.into(),
            kind: "pat".into(),
            label: "a tool".into(),
        }
    }

    fn events(log_dir: &TempDir, name: &str) -> Vec<Value> {
        lines(&log_dir.0.join("auth").join("audit.jsonl"))
            .into_iter()
            .filter(|l| l["event"] == name)
            .collect()
    }

    #[test]
    fn f7_thirty_derives_in_a_minute_are_one_line_in_full_and_one_line_that_counts_the_rest() {
        let dir = TempDir::new("audit-derived");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        let parent = derive_actor("aaaaaaaaaaaaaaaa");
        let ctx = Context {
            ip: Some("203.0.113.9"),
            host: None,
            ua: None,
        };
        for i in 0..30 {
            log.derived(&parent, &ctx, json!({ "n": i }));
            clock.advance(1000);
        }
        // The first is written when it happens; the count waits for the minute to be over.
        assert_eq!(events(&dir, "credential.created").len(), 1);
        assert_eq!(events(&dir, "credential.derived_more").len(), 0);
        log.flush_closed();
        assert_eq!(
            events(&dir, "credential.derived_more").len(),
            0,
            "the minute is not over yet"
        );
        assert_eq!(
            events(&dir, "credential.created")[0]["detail"],
            json!({ "n": 0 })
        );
        clock.advance(60_000);
        log.flush_closed();
        let more = events(&dir, "credential.derived_more");
        assert_eq!(more.len(), 1);
        assert_eq!(more[0]["detail"]["count"], 29);
        assert_eq!(more[0]["principal"]["id"], "aaaaaaaaaaaaaaaa");
        assert_eq!(more[0]["ip"], "203.0.113.9");
        assert!(more[0]["detail"]["minute"]
            .as_str()
            .unwrap()
            .starts_with("2026-09-21T14:13:00"));
        // Nothing more is written for that minute, by the same flush again.
        log.flush_closed();
        assert_eq!(events(&dir, "credential.derived_more").len(), 1);
    }

    #[test]
    fn f7_six_hours_at_thirty_a_minute_write_at_most_two_lines_a_minute_and_not_ten_thousand() {
        let dir = TempDir::new("audit-derived-hours");
        let clock = Arc::new(ManualClock::new(T0 - T0 % 60_000));
        let log = log(&dir, &clock);
        let parent = derive_actor("bbbbbbbbbbbbbbbb");
        let ctx = Context::default();
        for _ in 0..360 {
            for _ in 0..30 {
                log.derived(&parent, &ctx, json!({}));
                clock.advance(2000);
            }
        }
        log.flush_noise();
        let created = events(&dir, "credential.created").len();
        let more = events(&dir, "credential.derived_more");
        assert_eq!(created, 360, "one in full a minute");
        assert_eq!(more.len(), 360, "and one count a minute");
        let counted: u64 = more
            .iter()
            .map(|l| l["detail"]["count"].as_u64().unwrap())
            .sum();
        assert_eq!(
            counted + created as u64,
            10_800,
            "every derive is in a line"
        );
    }

    #[test]
    fn f7_each_parent_has_its_own_line_and_a_lone_derive_has_no_count() {
        let dir = TempDir::new("audit-derived-parents");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        let (a, b) = (
            derive_actor("aaaaaaaaaaaaaaaa"),
            derive_actor("bbbbbbbbbbbbbbbb"),
        );
        let ctx = Context::default();
        log.derived(&a, &ctx, json!({}));
        log.derived(&b, &ctx, json!({}));
        log.derived(&a, &ctx, json!({}));
        // `b` derived once in that minute, `a` twice.
        log.flush_noise();
        assert_eq!(events(&dir, "credential.created").len(), 2);
        let more = events(&dir, "credential.derived_more");
        assert_eq!(more.len(), 1);
        assert_eq!(more[0]["principal"]["id"], "aaaaaaaaaaaaaaaa");
        assert_eq!(more[0]["detail"]["count"], 1);
        // The next minute's first derive of a parent that had a count writes the count of the last one.
        let log2 = AuditLog::open(&dir.0.join("auth2"), clock.clone(), false);
        for _ in 0..3 {
            log2.derived(&a, &ctx, json!({}));
        }
        clock.advance(60_000);
        log2.derived(&a, &ctx, json!({}));
        let text = lines(&dir.0.join("auth2").join("audit.jsonl"));
        let names: Vec<&str> = text.iter().map(|l| l["event"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "credential.created",
                "credential.derived_more",
                "credential.created"
            ]
        );
        assert_eq!(text[1]["detail"]["count"], 2);
    }

    #[test]
    fn nothing_secret_is_written_whatever_the_detail_holds() {
        // T25: audit redaction of each new prefix, csrf, cookies and passwords.
        let dir = TempDir::new("audit-redact");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        let ses = TOKEN.replace("oaiypat", "oaiyses");
        let dsk = TOKEN.replace("oaiypat", "oaiydsk");
        let run = TOKEN.replace("oaiypat", "oaiyrun");
        let con = TOKEN.replace("oaiypat", "oaiycon");
        let dev = TOKEN.replace("oaiypat", "oaiydev");
        log.critical(
            "credential.created",
            None,
            &Context::default(),
            json!({
                "token": TOKEN, "session": ses, "desk": dsk, "run": run, "console": con, "device": dev,
                "csrf": "A-wCyU91xe9jXHeH2QyQhZgv00DTWUwk1H9x_7RSRmg", "X-OAIY-CSRF": "abc", "cookie": "x=y",
                "password": "hunter2", "note": "kept",
                "nested": { "list": [ses, "fine"], "Authorization": "Bearer abcdefghijklmnop" },
            }),
        );
        let text = std::fs::read_to_string(dir.0.join("auth").join("audit.jsonl")).unwrap();
        for secret in [
            TOKEN,
            &ses,
            &dsk,
            &run,
            &con,
            &dev,
            "A-wCyU91xe9jXHeH2QyQhZgv00DTWUwk1H9x_7RSRmg",
            "hunter2",
            "abcdefghijklmnop",
            "x=y",
        ] {
            assert!(!text.contains(secret), "{secret} leaked into {text}");
        }
        assert!(
            text.contains("kept") && text.contains("fine"),
            "what is not secret stays"
        );
        assert!(text.contains("[redacted]"));
    }

    #[test]
    fn the_user_agent_is_cut_and_a_long_detail_is_shortened() {
        let dir = TempDir::new("audit-cut");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        log.critical(
            "login.ok",
            None,
            &Context {
                ua: Some(&"u".repeat(500)),
                ..Context::default()
            },
            json!({ "long": "x".repeat(5000) }),
        );
        let l = &lines(&dir.0.join("auth").join("audit.jsonl"))[0];
        assert_eq!(l["ua"].as_str().unwrap().len(), MAX_UA);
        assert!(l["detail"]["long"].as_str().unwrap().len() < 400);
    }

    #[test]
    fn noise_is_one_line_per_event_address_and_minute_with_a_count() {
        let dir = TempDir::new("noise-agg");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        for _ in 0..4 {
            log.noise("login.fail", "203.0.113.9", "dash.example.com");
            clock.advance(1_000);
        }
        log.noise("login.fail", "198.51.100.7", "dash.example.com");
        log.noise("setup.fail", "203.0.113.9", "dash.example.com");
        // Nothing is written while the minute is open.
        assert!(lines(&dir.0.join("auth").join("noise.jsonl")).is_empty());
        clock.advance(120_000);
        log.noise("login.fail", "203.0.113.9", "dash.example.com");
        let got = lines(&dir.0.join("auth").join("noise.jsonl"));
        assert_eq!(got.len(), 3, "{got:?}");
        let mine = got
            .iter()
            .find(|l| l["event"] == "login.fail" && l["ip"] == "203.0.113.9")
            .unwrap();
        assert_eq!(mine["count"], 4);
        assert_eq!(mine["host"], "dash.example.com");
        assert_eq!(mine["first_ms"], T0);
        assert_eq!(mine["last_ms"], T0 + 3_000);
        assert_eq!(
            got.iter().find(|l| l["ip"] == "198.51.100.7").unwrap()["count"],
            1
        );
        // The open minute is written at flush.
        log.flush_noise();
        assert_eq!(lines(&dir.0.join("auth").join("noise.jsonl")).len(), 4);
    }

    #[test]
    fn anonymous_traffic_never_touches_the_audit_file() {
        // The first draft's single 5 MiB file held about 80,000 lines and one client at 200 requests a
        // second replaced it in six minutes. Here 20,000 anonymous events leave audit.jsonl as it was.
        let dir = TempDir::new("noise-flood");
        let clock = Arc::new(ManualClock::new(T0));
        let log = AuditLog::with_limits(
            &dir.0.join("auth"),
            clock.clone(),
            false,
            AUDIT_MAX_BYTES,
            4096,
        );
        log.critical(
            "password.changed",
            Some(&actor()),
            &Context::default(),
            json!({}),
        );
        let audit = dir.0.join("auth").join("audit.jsonl");
        let before = std::fs::read(&audit).unwrap();
        for i in 0..20_000u32 {
            log.noise(
                "login.fail",
                &format!("203.0.{}.{}", (i / 250) % 250, i % 250),
                "dash.example.com",
            );
            if i % 100 == 0 {
                clock.advance(61_000);
            }
        }
        log.flush_noise();
        assert_eq!(
            std::fs::read(&audit).unwrap(),
            before,
            "audit.jsonl must be untouched by anonymous events"
        );
        assert!(!dir.0.join("auth").join("audit.jsonl.1").exists());
        assert!(dir.0.join("auth").join("noise.jsonl").exists());
    }

    #[test]
    fn a_flood_from_rotating_addresses_cannot_grow_memory_without_bound() {
        let dir = TempDir::new("noise-overflow");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        for i in 0..(MAX_NOISE_BUCKETS + 500) {
            log.noise("login.fail", &format!("ip-{i}"), "h");
        }
        let held = log.noise.lock().unwrap().buckets.len();
        assert!(held <= MAX_NOISE_BUCKETS + 1, "{held} buckets");
        log.flush_noise();
        let got = lines(&dir.0.join("auth").join("noise.jsonl"));
        let overflow = got
            .iter()
            .find(|l| l["ip"] == OVERFLOW_IP)
            .expect("the excess is counted together");
        assert_eq!(overflow["count"], 500);
        let total: u64 = got.iter().map(|l| l["count"].as_u64().unwrap()).sum();
        assert_eq!(total, (MAX_NOISE_BUCKETS + 500) as u64, "nothing is lost");
    }

    #[test]
    fn the_audit_file_rolls_at_its_size_and_keeps_eight_files() {
        let dir = TempDir::new("audit-roll");
        let clock = Arc::new(ManualClock::new(T0));
        let log = AuditLog::with_limits(
            &dir.0.join("auth"),
            clock.clone(),
            false,
            300,
            NOISE_MAX_BYTES,
        );
        for i in 0..200 {
            log.critical(
                "session.revoked",
                Some(&actor()),
                &Context::default(),
                json!({ "n": i }),
            );
        }
        let auth = dir.0.join("auth");
        let mut names: Vec<String> = std::fs::read_dir(&auth)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("audit"))
            .collect();
        names.sort();
        assert_eq!(names.len(), AUDIT_FILES, "{names:?}");
        assert!(
            names.contains(&"audit.jsonl".to_string())
                && names.contains(&"audit.jsonl.7".to_string())
        );
        // The newest are kept and they read back newest first across the files.
        let read = log.read(LogFile::Audit, 10, None, None);
        assert_eq!(read[0]["detail"]["n"], 199);
        assert_eq!(read[9]["detail"]["n"], 190);
        // No file grows past the roll size by more than the line that crossed it.
        for n in &names {
            assert!(
                std::fs::metadata(auth.join(n)).unwrap().len() < 300 + 300,
                "{n}"
            );
        }
    }

    #[test]
    fn the_noise_file_keeps_two_files() {
        let dir = TempDir::new("noise-roll");
        let clock = Arc::new(ManualClock::new(T0));
        let log = AuditLog::with_limits(
            &dir.0.join("auth"),
            clock.clone(),
            false,
            AUDIT_MAX_BYTES,
            400,
        );
        for i in 0..100 {
            log.noise("login.fail", &format!("203.0.113.{i}"), "h");
            if i % 10 == 9 {
                clock.advance(61_000);
            }
        }
        log.flush_noise();
        let auth = dir.0.join("auth");
        assert!(auth.join("noise.jsonl").exists() && auth.join("noise.jsonl.1").exists());
        assert!(
            !auth.join("noise.jsonl.2").exists(),
            "one rolled file, no more"
        );
    }

    #[test]
    fn a_read_is_newest_first_filtered_and_capped() {
        let dir = TempDir::new("audit-read");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        let other = Actor {
            id: "ffffffffffffffff".into(),
            kind: "pat".into(),
            label: "x".into(),
        };
        for i in 0..6u64 {
            let who = if i % 2 == 0 { actor() } else { other.clone() };
            log.critical(
                "credential.revoked",
                Some(&who),
                &Context::default(),
                json!({ "n": i }),
            );
            clock.advance(1_000);
        }
        let all = log.read(LogFile::Audit, 100, None, None);
        assert_eq!(
            all.iter()
                .map(|l| l["detail"]["n"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [5, 4, 3, 2, 1, 0]
        );
        assert_eq!(log.read(LogFile::Audit, 2, None, None).len(), 2);
        let mine = log.read(LogFile::Audit, 100, None, Some("0123456789abcdef"));
        assert_eq!(mine.len(), 3);
        let before = log.read(LogFile::Audit, 100, Some(T0 + 3_000), None);
        assert_eq!(
            before
                .iter()
                .map(|l| l["detail"]["n"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [2, 1, 0]
        );
        assert!(log.read(LogFile::Audit, usize::MAX, None, None).len() <= MAX_READ);
        assert!(log.read(LogFile::Noise, 10, None, None).is_empty());
    }

    #[test]
    fn a_line_cut_short_by_a_crash_does_not_hide_the_rest() {
        let dir = TempDir::new("audit-torn");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        log.critical("a", None, &Context::default(), json!({}));
        let path = dir.0.join("auth").join("audit.jsonl");
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b"{\"at\":\"2026-01-01T00:00:00.000Z\",\"event\":\"tor")
            .unwrap();
        drop(f);
        log.critical("b", None, &Context::default(), json!({}));
        let read = log.read(LogFile::Audit, 10, None, None);
        assert_eq!(
            read.iter()
                .map(|l| l["event"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
    }

    #[test]
    fn a_folder_that_cannot_be_written_costs_a_log_line_and_not_a_panic() {
        let dir = TempDir::new("audit-unwritable");
        // A file where the folder belongs.
        std::fs::write(dir.0.join("auth"), "in the way").unwrap();
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        log.critical("login.ok", None, &Context::default(), json!({}));
        log.noise("login.fail", "1.2.3.4", "h");
        log.flush_noise();
    }

    #[cfg(unix)]
    #[test]
    fn the_files_and_their_folder_are_private() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = TempDir::new("audit-mode");
        let clock = Arc::new(ManualClock::new(T0));
        let log = log(&dir, &clock);
        log.critical("login.ok", None, &Context::default(), json!({}));
        log.noise("login.fail", "1.2.3.4", "h");
        log.flush_noise();
        let mode = |p: PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(dir.0.join("auth")), 0o700);
        assert_eq!(mode(dir.0.join("auth").join("audit.jsonl")), 0o600);
        assert_eq!(mode(dir.0.join("auth").join("noise.jsonl")), 0o600);
    }

    #[test]
    fn a_structured_stderr_field_cannot_add_another() {
        assert_eq!(stderr_field("login.fail"), "login.fail");
        assert_eq!(stderr_field("a b=c\nd"), "a_b_c_d");
        assert_eq!(stderr_field(&"x".repeat(200)).len(), 64);
    }

    #[test]
    fn the_single_writer_other_designs_use_writes_to_the_installed_log_and_nothing_without_one() {
        let dir = TempDir::new("audit-global");
        let clock = Arc::new(ManualClock::new(T0));
        // Global state: this test is the only one that installs a log.
        uninstall();
        critical("relay.paired", None, json!({}));
        assert!(
            !dir.0.join("auth").join("audit.jsonl").exists(),
            "nothing installed, nothing written"
        );
        use_log(Arc::new(log(&dir, &clock)));
        critical(
            "relay.paired",
            Some(&actor()),
            json!({ "device": "d1", "token": TOKEN }),
        );
        noise("bearer.failed", "203.0.113.9", "h");
        let read = installed().unwrap().read(LogFile::Audit, 10, None, None);
        uninstall();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0]["event"], "relay.paired");
        assert!(!read[0].to_string().contains(TOKEN));
    }
}
