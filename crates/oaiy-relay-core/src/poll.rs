//! The rules of the poll loop of a native client, as one deterministic function (README section 5.1.1, rules P1 to P9; packages DK-03 and MOB-21a).
//!
//! **There is one decision, and the loop only acts on it.** [`decide`] takes what the client knows (its four counters, `info`, the answer to the last poll, the `since` it
//! carried, whether what was accepted could be written to the store, the jitter draw, the clock for a `Retry-After` date) and returns [`Decision`]: the outcome (one of
//! the six of P2), the pause before the next request, the new counters, the action and the reports of P7, and the `since` to carry next. It does no I/O, reads no clock and
//! draws no random number, so the same function runs in a tokio task on the desktop and in a foreground service on the phone, and
//! `fixtures/poll-client/poll-client.json` (135 cases, read independently by `verify_poll_client.py` and `.mjs`) is its test: `tests/poll_fixture.rs` runs every case.
//!
//! The words are the README's: the **outcomes** are `progress`, `superseded`, `idle`, `flow` (a `429`), `failure` and `stop` (and `proved`, the outcome of a good identity
//! proof, P9); the **actions** are `forget_credential`, `refresh_or_reenrol`, `update_client`, `report_defect`, `clear_epoch`, `cancel_own_polls` and
//! `report_relay_changed`; the **reports** are `unreachable`, `in_flight_defect`, `duplicate_credential`, `invalid_request` and `storage_failure`.
//!
//! Where this differs from the design text (4.16.5), the README and the table win: the poll's own timeout is `wait + 10` seconds (not `W + 15`), the jitter is
//! `base * (1 + 0.2 * u)` with `u` in `[0, 1)` (not full jitter), and a `429` is never a failure (it does not count towards `unreachable`).

use crate::json::{Json, Number, MAX_SAFE_INT};

/// P1: the soonest a client that cancels a running poll may start the next, after it started the one it cancels.
pub const REPLACE_MIN_MS: u64 = 250;
/// P6: a pause taken from a `Retry-After` is at least this many seconds (a client never spins).
pub const CLAMP_MIN_S: u64 = 1;
/// P6: and at most this many.
pub const CLAMP_MAX_S: u64 = 120;
/// P6: the largest `error.retryAfter` of a body that is read at all.
pub const RETRY_AFTER_BODY_MAX: i128 = 86_400;
/// P6: the most digits of a `Retry-After` in seconds.
pub const RETRY_AFTER_DIGITS_MAX: usize = 6;
/// P6: a pause has up to this fraction added, drawn uniformly.
pub const JITTER: f64 = 0.2;
/// P5: the cap of the pause after consecutive `429`s, in seconds.
pub const BACKOFF_429_CAP_S: u64 = 30;
/// P5: the cap of the pause after consecutive failures, in seconds.
pub const BACKOFF_FAILURE_CAP_S: u64 = 60;
/// P7: failures in a row after which the relay is reported unreachable.
pub const UNREACHABLE_AFTER: u32 = 3;
/// P4 and P7: the fifth `429` in a row that says `in_flight` is a defect to report once.
pub const IN_FLIGHT_DEFECT_AFTER: u32 = 5;
/// P3: the `retryAfter` of a refused hold when the answer has none or has one that is not an integer.
pub const REFUSED_HOLD_DEFAULT_S: u64 = 2;
/// P9: the identity proof is repeated at least this often while polling.
pub const PROOF_EVERY_S: u64 = 300;
/// P9: and after any pause of this many seconds or more.
pub const PROOF_AFTER_PAUSE_S: u64 = 60;
/// P1: a poll's own timeout is its `wait` plus this many seconds.
pub const POLL_TIMEOUT_EXTRA_S: u64 = 10;
/// P2: the largest `cursor` and `seq`: 2^53 - 1.
pub const MAX_SEQ: u64 = MAX_SAFE_INT;

/// The four numbers a client keeps for these rules (P8), besides two times that are the driver's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// Consecutive `429`s.
    pub n429: u32,
    /// Consecutive failures.
    pub n_fail: u32,
    /// Consecutive refused holds.
    pub n_refused: u32,
    /// Consecutive `400`s.
    pub n400: u32,
}

/// What the rules read of `GET /v1/info`: `wait.pollGapMs` and `wait.fallbackS`. A relay that says more than its own configuration allows is clamped to it (0 to 5000
/// ms; 1 to 60 s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollInfo {
    /// `info.wait.pollGapMs`: the pause after a poll that made no progress.
    pub poll_gap_ms: u64,
    /// `info.wait.fallbackS`: the cap of the pause after a refused hold.
    pub fallback_s: u64,
}

impl PollInfo {
    /// The values of a parsed `info`, clamped to what the relay's own configuration allows (`gap_ms` 0 to 5000, `fallback_s` 1 to 60).
    pub fn from_info(info: &crate::info::Info) -> PollInfo {
        PollInfo { poll_gap_ms: info.wait.poll_gap_ms.min(5000), fallback_s: info.wait.fallback_s.clamp(1, 60) }
    }
}

impl Default for PollInfo {
    /// The values the relay ships: 250 ms and 5 s.
    fn default() -> Self {
        PollInfo { poll_gap_ms: 250, fallback_s: 5 }
    }
}

/// The answer to a poll, as the rules see it.
#[derive(Debug, Clone, Copy)]
pub struct Answer<'a> {
    /// The HTTP status, or `None` when no response came back (refused, reset, TLS error, the poll's own timeout, a body that ended early).
    pub status: Option<u16>,
    /// The headers, names in lower case; the first of a repeated name is the one read.
    pub headers: &'a [(String, String)],
    /// The body parsed as JSON, or `None` when there was none or it was not JSON.
    pub body: Option<&'a Json>,
}

impl Answer<'_> {
    /// No response at all.
    pub fn none() -> Answer<'static> {
        Answer { status: None, headers: &[], body: None }
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

/// An integer as Python's `isinstance(x, int)` and JavaScript's `Number.isInteger` see it: an integer spelling of any size (saturated here) and nothing that has a fraction,
/// an exponent or is `-0`.
fn int_like(v: &Json) -> Option<i128> {
    match v {
        Json::Num(Number::Int(n)) => Some(*n),
        Json::Num(Number::Big(t)) => Some(if t.starts_with('-') { i128::MIN } else { i128::MAX }),
        _ => None,
    }
}

fn is_epoch(text: &str) -> bool {
    crate::ids::is_epoch(text)
}

/// P2: a `200` is **valid** when its body is a JSON object whose `items` is an array, whose `epoch` is a string of 11 characters of the base64url alphabet and whose
/// `cursor` is an integer from 0 to 2^53 - 1.
pub fn valid_200(body: Option<&Json>) -> bool {
    let Some(b) = body else { return false };
    b.is_object()
        && b.get("items").and_then(Json::as_array).is_some()
        && b.get_str("epoch").is_some_and(is_epoch)
        && b.get("cursor").and_then(int_like).is_some_and(|c| (0..=i128::from(MAX_SEQ)).contains(&c))
}

/// P2: **an item is accepted** when its `seq` is an integer above the `since` the poll carried and above the `seq` of the item accepted before it in the same answer. The
/// relay sends items in ascending order and never one that was acknowledged, so an item that is not is the relay's defect or a replay, and is dropped. Returns the indexes of
/// the accepted items.
pub fn accepted_items(items: &[Json], since: u64) -> Vec<usize> {
    let mut last = i128::from(since);
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if let Some(seq) = item.get("seq").and_then(int_like) {
            if last < seq && seq <= i128::from(MAX_SEQ) {
                out.push(i);
                last = seq;
            }
        }
    }
    out
}

/// What a valid `200` leads the client to adopt, and so to persist before it sends the poll that carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adoption {
    /// The `since` to carry next: the highest accepted `seq`, or a reset's `cursor`.
    pub since: u64,
    /// The relay's epoch of this answer, to be sent back byte for byte.
    pub epoch: String,
    /// True for a `reset: true` answer: the cursor is the server's (it may be lower) and "mailbox reset: in-flight items may be lost" is recorded.
    pub reset: bool,
    /// The indexes (into `items`) of the accepted items; none on a reset.
    pub accepted: Vec<usize>,
}

/// P2: what a valid `200` carries that the client must persist: `Some` for progress (an accepted item, or a reset), `None` otherwise (idle, superseded without an item, an
/// answer that is not valid, any other status). The driver calls this first, writes what it names to its store, and passes the result of the write to [`decide`].
pub fn assess(status: Option<u16>, body: Option<&Json>, since: u64) -> Option<Adoption> {
    if status != Some(200) || !valid_200(body) {
        return None;
    }
    let b = body?;
    let epoch = b.get_str("epoch")?.to_string();
    if b.get("reset").and_then(Json::as_bool) == Some(true) {
        let cursor = b.get("cursor").and_then(int_like)?;
        return Some(Adoption { since: u64::try_from(cursor).ok()?, epoch, reset: true, accepted: Vec::new() });
    }
    let accepted = accepted_items(b.get("items")?.as_array()?, since);
    let last = *accepted.last()?;
    let seq = b.get("items")?.as_array()?.get(last)?.get("seq").and_then(int_like)?;
    Some(Adoption { since: u64::try_from(seq).ok()?, epoch, reset: false, accepted })
}

/// P2: the six outcomes, and `proved` (P9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// A valid `200` with an accepted item or a reset.
    Progress,
    /// A `200` with `hold.superseded`: a newer poll of the device replaced this one.
    Superseded,
    /// Any other valid `200`.
    Idle,
    /// A `429` of any code, rule or body. Never a failure.
    Flow,
    /// No answer, `408`, `5xx`, a redirect, an invalid `200`, the first `400` of a run, a failed write, a `426` that does not make the client too old.
    Failure,
    /// Any other `4xx`, a second `400` in a row, a `426` that makes the client too old, an identity proof that does not verify. The loop ends.
    Stop,
    /// A verified identity proof (P9).
    Proved,
}

impl Outcome {
    /// The README's word.
    pub const fn as_str(self) -> &'static str {
        match self {
            Outcome::Progress => "progress",
            Outcome::Superseded => "superseded",
            Outcome::Idle => "idle",
            Outcome::Flow => "flow",
            Outcome::Failure => "failure",
            Outcome::Stop => "stop",
            Outcome::Proved => "proved",
        }
    }
}

/// P7: what a decision tells the loop to do besides pausing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// `401 revoked`: the device was revoked.
    ForgetCredential,
    /// Any other `401`: refresh the credential once if it can be, otherwise end the loop and ask the owner to enrol again.
    RefreshOrReenrol,
    /// A `426` that makes the client too old: tell the owner to update.
    UpdateClient,
    /// Any other `4xx`, or a second `400`: a defect of the client or a relay that is switched off; report it with the code.
    ReportDefect,
    /// The first `400`: the retry leaves out the stored epoch (an omitted epoch is no check).
    ClearEpoch,
    /// A `429` that says `in_flight`: cancel every poll of this client that is still in flight.
    CancelOwnPolls,
    /// An identity proof that does not verify: report the relay as "not who it was", keep the credential and do not send it.
    ReportRelayChanged,
}

impl Action {
    /// The README's word.
    pub const fn as_str(self) -> &'static str {
        match self {
            Action::ForgetCredential => "forget_credential",
            Action::RefreshOrReenrol => "refresh_or_reenrol",
            Action::UpdateClient => "update_client",
            Action::ReportDefect => "report_defect",
            Action::ClearEpoch => "clear_epoch",
            Action::CancelOwnPolls => "cancel_own_polls",
            Action::ReportRelayChanged => "report_relay_changed",
        }
    }
}

/// P7: what the client tells its user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Report {
    /// Three failures in a row (about seven seconds).
    Unreachable,
    /// The fifth `429` in a row says `in_flight`: the client has, or shares its credential with, more polls than P1 allows. Once.
    InFlightDefect,
    /// A `superseded` answer to a poll this client did not replace: another process polls with its credential.
    DuplicateCredential,
    /// The first `400` of a run.
    InvalidRequest,
    /// What was accepted could not be written to the client's store.
    StorageFailure,
}

impl Report {
    /// The README's word.
    pub const fn as_str(self) -> &'static str {
        match self {
            Report::Unreachable => "unreachable",
            Report::InFlightDefect => "in_flight_defect",
            Report::DuplicateCredential => "duplicate_credential",
            Report::InvalidRequest => "invalid_request",
            Report::StorageFailure => "storage_failure",
        }
    }
}

/// Everything [`decide`] reads.
#[derive(Debug, Clone, Copy)]
pub struct DecideInput<'a> {
    /// The counters before this answer (P8).
    pub counters: Counters,
    /// `info.wait.pollGapMs` and `fallbackS`.
    pub info: PollInfo,
    /// The answer to the poll.
    pub answer: Answer<'a>,
    /// The `since` the poll carried (0 when it carried none).
    pub since: u64,
    /// True when what the answer accepted was written to the store (or there was nothing to write). The write is made before the next poll and this is its result.
    pub persisted: bool,
    /// True when this client cancelled the poll that was answered `superseded` itself (P2): the newer poll is its loop and nothing follows. False when another process
    /// polls with its credential.
    pub we_replaced: bool,
    /// What re-reading `info` after a `426` found: that `minClient` is above this client's level.
    pub min_client_above_ours: bool,
    /// The client's own clock in Unix seconds, for a `Retry-After` that is a date when there is no readable `Date` header.
    pub now_epoch: Option<i64>,
    /// The jitter draw `u`, uniform from 0 up to but not including 1.
    pub u: f64,
}

/// The decision: what the loop does next.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// The outcome of the answer.
    pub outcome: Outcome,
    /// The pause before jitter, in seconds.
    pub base_s: f64,
    /// The pause after jitter: `base * (1 + 0.2 * u)`.
    pub pause_s: f64,
    /// The counters after this answer.
    pub counters: Counters,
    /// The action, if any.
    pub action: Option<Action>,
    /// The reports, if any.
    pub reports: Vec<Report>,
    /// The `since` to carry next: what was accepted, a reset's cursor, or what it was.
    pub since: u64,
}

fn jittered(base: f64, u: f64) -> f64 {
    let u = if u.is_finite() { u.clamp(0.0, 1.0 - f64::EPSILON) } else { 0.0 };
    base * (1.0 + JITTER * u)
}

/// P6: `x` limited to 1 to 120, so that 0 is 1 and 600 is 120.
pub fn clamp_pause(x: i128) -> u64 {
    x.clamp(i128::from(CLAMP_MIN_S), i128::from(CLAMP_MAX_S)) as u64
}

/// `2^(n-1)` limited to `cap`, without overflow: 1, 2, 4, 8 ... (`n` is the count of the outcome in a row after this answer; the first is 1).
fn doubling(n: u32, cap: u64) -> u64 {
    let shift = n.saturating_sub(1).min(32);
    (1u64 << shift).min(cap)
}

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/// Days from 1970-01-01 to the first of `month` (1 to 12) of `year`: the proleptic Gregorian calendar.
fn days_to_month_start(year: i64, month: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// P6: an IMF-fixdate (RFC 9110: `Sun, 06 Nov 1994 08:49:37 GMT`) as Unix seconds. Nothing else is a date: no other format, no zone but `GMT`, no trailing text. The weekday is
/// not checked against the date, and a day, hour, minute or second that is past its range is added as it stands (`calendar.timegm` does the same), as the two readers of the
/// table do.
pub fn parse_http_date(text: &str) -> Option<i64> {
    let t = text.trim_matches(|c| c == ' ' || c == '\t');
    let b = t.as_bytes();
    // `Sun, 06 Nov 1994 08:49:37 GMT`: 29 bytes of ASCII (which also keeps every slice below on a character boundary).
    if !t.is_ascii()
        || b.len() != 29
        || b[3] != b','
        || b[4] != b' '
        || b[7] != b' '
        || b[11] != b' '
        || b[16] != b' '
        || b[19] != b':'
        || b[22] != b':'
        || &t[25..] != " GMT"
    {
        return None;
    }
    if !["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].contains(&&t[..3]) {
        return None;
    }
    let num = |s: &str| -> Option<i64> {
        if !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) {
            s.parse().ok()
        } else {
            None
        }
    };
    let day = num(&t[5..7])?;
    let month = MONTHS.iter().position(|m| *m == &t[8..11])? as i64 + 1;
    let year = num(&t[12..16])?;
    let (hh, mm, ss) = (num(&t[17..19])?, num(&t[20..22])?, num(&t[23..25])?);
    Some((days_to_month_start(year, month) + day - 1) * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// P6: how long the relay asked to wait, in seconds, or `None`. The header first: after trimming spaces and tabs one to six decimal digits are that many seconds; an
/// HTTP-date is the seconds from the `Date` header of the same answer (from the client's own clock when that header is absent or unreadable) to it, and a date that has
/// passed is 0; anything else (empty, a sign, a fraction, text, seven digits) is no header. With no readable header `error.retryAfter` is read when it is an integer from 0
/// to 86,400.
pub fn retry_after(answer: &Answer<'_>, now_epoch: Option<i64>) -> Option<u64> {
    if let Some(v) = answer.header("retry-after") {
        let s = v.trim_matches(|c| c == ' ' || c == '\t');
        if (1..=RETRY_AFTER_DIGITS_MAX).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_digit()) {
            return s.parse().ok();
        }
        if let Some(t) = parse_http_date(s) {
            let reference = answer.header("date").and_then(parse_http_date).or(now_epoch);
            if let Some(r) = reference {
                return Some((t - r).max(0) as u64);
            }
        }
    }
    let r = answer.body?.get("error")?.get("retryAfter").and_then(int_like)?;
    if (0..=RETRY_AFTER_BODY_MAX).contains(&r) {
        Some(r as u64)
    } else {
        None
    }
}

/// P1: how long a client that cancels a running poll waits before it starts the next: until 250 ms have passed since it started the first.
pub fn replace_wait_ms(ms_since_last_start: u64) -> u64 {
    REPLACE_MIN_MS.saturating_sub(ms_since_last_start)
}

/// What the loop knows when it asks whether a proof is due (P9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofDue {
    /// The process has not proved the relay yet.
    pub process_start: bool,
    /// The network changed.
    pub network_changed: bool,
    /// The longest single pause since the last proof, in seconds.
    pub longest_pause_s: u64,
    /// Seconds since the last proof succeeded.
    pub seconds_since_proof: u64,
}

/// P9: before the first poll of a process, after any pause of 60 seconds or more, after a change of network, and at least every 300 seconds while polling.
pub fn proof_due(p: &ProofDue) -> bool {
    p.process_start || p.network_changed || p.longest_pause_s >= PROOF_AFTER_PAUSE_S || p.seconds_since_proof >= PROOF_EVERY_S
}

/// The result of asking for the identity proof (P9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofResult {
    /// The proof verified against the pinned key.
    Verified,
    /// No answer to the proof request (any answer that is not a proof: a failure like any other).
    NoAnswer,
    /// An answer that does not verify: a replayed body and signature, another key.
    Invalid,
}

fn decision(outcome: Outcome, base_s: f64, u: f64, counters: Counters, since: u64) -> Decision {
    Decision { outcome, base_s, pause_s: jittered(base_s, u), counters, action: None, reports: Vec::new(), since }
}

/// P9: what the answer to the identity proof leads to. A proof that verifies sets the time of the proof and clears the failure count (the relay answered), and leaves the
/// other counts: a proof is not an answer to a poll. No answer is a failure like any other. An answer that does not verify is a stop, `report_relay_changed`.
pub fn decide_proof(counters: Counters, result: ProofResult, u: f64, since: u64) -> Decision {
    match result {
        ProofResult::Verified => decision(Outcome::Proved, 0.0, u, Counters { n_fail: 0, ..counters }, since),
        ProofResult::NoAnswer => {
            let n = counters.n_fail + 1;
            let mut d =
                decision(Outcome::Failure, doubling(n, BACKOFF_FAILURE_CAP_S) as f64, u, Counters { n_fail: n, ..Counters::default() }, since);
            if n >= UNREACHABLE_AFTER {
                d.reports.push(Report::Unreachable);
            }
            d
        }
        ProofResult::Invalid => {
            let mut d = decision(Outcome::Stop, 0.0, u, counters, since);
            d.action = Some(Action::ReportRelayChanged);
            d
        }
    }
}

/// The rules P2 to P8 for the answer to one poll. See the module.
pub fn decide(input: &DecideInput<'_>) -> Decision {
    let DecideInput { counters, info, answer, since, persisted, we_replaced, min_client_above_ours, now_epoch, u } = *input;
    let status = answer.status;
    let body = answer.body;
    let cleared = Counters::default();

    // The first row of the table that fits is the answer: a valid 200 with an accepted item, or a reset, is progress whatever its hold says.
    if status == Some(200) && valid_200(body) {
        let b = body.unwrap_or(&Json::Null);
        let hold = b.get("hold").filter(|h| h.is_object());
        if let Some(adoption) = assess(status, body, since) {
            if !persisted {
                // What was accepted could not be written: nothing advances, and the failure is paced like any other. It never reports `unreachable` by itself: the relay answered.
                let n = counters.n_fail + 1;
                let mut d = decision(Outcome::Failure, doubling(n, BACKOFF_FAILURE_CAP_S) as f64, u, Counters { n_fail: n, ..cleared }, since);
                d.reports.push(Report::StorageFailure);
                return d;
            }
            return decision(Outcome::Progress, 0.0, u, cleared, adoption.since);
        }
        if hold.and_then(|h| h.get("superseded")).and_then(Json::as_bool) == Some(true) {
            if !we_replaced {
                let mut d = decision(Outcome::Superseded, info.poll_gap_ms as f64 / 1000.0, u, cleared, since);
                d.reports.push(Report::DuplicateCredential);
                return d;
            }
            return decision(Outcome::Superseded, 0.0, u, cleared, since);
        }
        if hold.and_then(|h| h.get("refused")).and_then(Json::as_bool) == Some(true) {
            let r = hold.and_then(|h| h.get("retryAfter")).and_then(int_like).map_or(REFUSED_HOLD_DEFAULT_S, clamp_pause);
            let k = counters.n_refused.min(30);
            let base = r.max(info.fallback_s.min(r.saturating_mul(1u64 << k)));
            return decision(Outcome::Idle, base as f64, u, Counters { n_refused: counters.n_refused + 1, ..cleared }, since);
        }
        return decision(Outcome::Idle, info.poll_gap_ms as f64 / 1000.0, u, cleared, since);
    }

    if status == Some(429) {
        let n = counters.n429 + 1;
        let d = clamp_pause(retry_after(&answer, now_epoch).map_or(1, i128::from));
        let base = d.max(doubling(n, BACKOFF_429_CAP_S));
        let mut out = decision(Outcome::Flow, base as f64, u, Counters { n429: n, ..cleared }, since);
        let rule = body.and_then(|b| b.get("error")).and_then(|e| e.get_str("rule"));
        if rule == Some("in_flight") {
            out.action = Some(Action::CancelOwnPolls);
            if n == IN_FLIGHT_DEFECT_AFTER {
                out.reports.push(Report::InFlightDefect);
            }
        }
        return out;
    }

    if status == Some(400) {
        if counters.n400 >= 1 {
            // A second 400 in a row can only be a defect of the client or a damaged store.
            let mut d = decision(Outcome::Stop, 0.0, u, counters, since);
            d.action = Some(Action::ReportDefect);
            return d;
        }
        let n = counters.n_fail + 1;
        let mut d = decision(Outcome::Failure, doubling(n, BACKOFF_FAILURE_CAP_S) as f64, u, Counters { n_fail: n, n400: 1, ..cleared }, since);
        d.action = Some(Action::ClearEpoch);
        d.reports.push(Report::InvalidRequest);
        return d;
    }

    if status == Some(426) && min_client_above_ours {
        let mut d = decision(Outcome::Stop, 0.0, u, counters, since);
        d.action = Some(Action::UpdateClient);
        return d;
    }

    if let Some(s) = status {
        if (400..=499).contains(&s) && !matches!(s, 408 | 426 | 429) {
            let code = body.and_then(|b| b.get("error")).and_then(|e| e.get_str("code"));
            let mut d = decision(Outcome::Stop, 0.0, u, counters, since);
            d.action = Some(if s == 401 {
                if code == Some("revoked") {
                    Action::ForgetCredential
                } else {
                    Action::RefreshOrReenrol
                }
            } else {
                Action::ReportDefect
            });
            return d;
        }
    }

    // Everything else is a failure: no answer, 408, 5xx, 1xx, 2xx that is not a valid 200, 3xx (never followed), an invalid 200, a 426 that is not ours to obey.
    let n = counters.n_fail + 1;
    let asked = if status.is_some() { retry_after(&answer, now_epoch) } else { None };
    let mut base = doubling(n, BACKOFF_FAILURE_CAP_S);
    if let Some(d) = asked {
        base = base.max(clamp_pause(i128::from(d)));
    }
    let mut d = decision(Outcome::Failure, base as f64, u, Counters { n_fail: n, ..cleared }, since);
    if n >= UNREACHABLE_AFTER {
        d.reports.push(Report::Unreachable);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn http_dates() {
        assert_eq!(parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"), Some(784_111_777));
        assert_eq!(parse_http_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(parse_http_date("Fri, 29 Feb 2008 12:00:00 GMT"), Some(1_204_286_400));
        assert_eq!(parse_http_date(" \tSun, 06 Nov 1994 08:49:37 GMT\t"), Some(784_111_777));
        for bad in [
            "Sunday, 06-Nov-94 08:49:37 GMT",
            "Sun Nov  6 08:49:37 1994",
            "Sun, 06 Nov 1994 08:49:37 UTC",
            "Sun, 06 Nov 1994 08:49:37 GMT x",
            "Sun, 6 Nov 1994 08:49:37 GMT",
            "Xyz, 06 Nov 1994 08:49:37 GMT",
            "Sun, 06 Nox 1994 08:49:37 GMT",
            "",
        ] {
            assert_eq!(parse_http_date(bad), None, "{bad}");
        }
    }

    #[test]
    fn retry_after_precedence_and_forms() {
        let none: Vec<(String, String)> = Vec::new();
        let body = crate::json::parse(br#"{"error":{"retryAfter":7}}"#).unwrap();
        let a = |h: &[(String, String)], b: Option<&Json>| retry_after(&Answer { status: Some(429), headers: h, body: b }, Some(1_000));
        assert_eq!(a(&headers(&[("retry-after", " 12\t")]), Some(&body)), Some(12));
        assert_eq!(a(&headers(&[("retry-after", "012")]), None), Some(12));
        assert_eq!(a(&headers(&[("retry-after", "1234567")]), Some(&body)), Some(7), "seven digits is no header, the body is read");
        assert_eq!(a(&headers(&[("retry-after", "-3")]), None), None);
        assert_eq!(a(&headers(&[("retry-after", "1.5")]), None), None);
        assert_eq!(a(&headers(&[("retry-after", "")]), None), None);
        assert_eq!(a(&none, Some(&body)), Some(7));
        assert_eq!(a(&none, None), None);
        // A date is read against the Date header of the answer, or the client's own clock; a date that has passed is 0.
        let date = "Sun, 06 Nov 1994 08:49:47 GMT";
        assert_eq!(a(&headers(&[("retry-after", date), ("date", "Sun, 06 Nov 1994 08:49:37 GMT")]), None), Some(10));
        assert_eq!(a(&headers(&[("retry-after", "Sun, 06 Nov 1994 08:49:00 GMT"), ("date", "Sun, 06 Nov 1994 08:49:37 GMT")]), None), Some(0));
        assert_eq!(retry_after(&Answer { status: Some(429), headers: &headers(&[("retry-after", date)]), body: None }, Some(784_111_777)), Some(10));
        assert_eq!(
            retry_after(
                &Answer { status: Some(429), headers: &headers(&[("retry-after", date), ("date", "garbage")]), body: None },
                Some(784_111_777)
            ),
            Some(10)
        );
        assert_eq!(
            retry_after(&Answer { status: Some(429), headers: &headers(&[("retry-after", date)]), body: None }, None),
            None,
            "no clock to read a date against"
        );
        // The body's value: an integer from 0 to 86,400 and nothing else.
        for (text, want) in [
            (r#"{"error":{"retryAfter":0}}"#, Some(0)),
            (r#"{"error":{"retryAfter":86400}}"#, Some(86_400)),
            (r#"{"error":{"retryAfter":86401}}"#, None),
            (r#"{"error":{"retryAfter":true}}"#, None),
            (r#"{"error":{"retryAfter":"7"}}"#, None),
            (r#"{"error":{"retryAfter":7.0}}"#, None),
            (r#"{"error":{"retryAfter":-1}}"#, None),
        ] {
            let b = crate::json::parse(text.as_bytes()).unwrap();
            assert_eq!(a(&none, Some(&b)), want, "{text}");
        }
    }

    #[test]
    fn clamp_and_doubling() {
        assert_eq!(
            [clamp_pause(-5), clamp_pause(0), clamp_pause(1), clamp_pause(120), clamp_pause(121), clamp_pause(i128::MAX)],
            [1, 1, 1, 120, 120, 120]
        );
        assert_eq!((1..=9).map(|n| doubling(n, 60)).collect::<Vec<_>>(), vec![1, 2, 4, 8, 16, 32, 60, 60, 60]);
        assert_eq!(doubling(u32::MAX, 30), 30);
        assert_eq!(doubling(0, 30), 1);
    }

    #[test]
    fn replace_and_proof_rules() {
        assert_eq!(
            [replace_wait_ms(0), replace_wait_ms(100), replace_wait_ms(249), replace_wait_ms(250), replace_wait_ms(9_000)],
            [250, 150, 1, 0, 0]
        );
        let p = ProofDue { process_start: false, network_changed: false, longest_pause_s: 0, seconds_since_proof: 0 };
        assert!(!proof_due(&p));
        assert!(proof_due(&ProofDue { process_start: true, ..p }));
        assert!(proof_due(&ProofDue { network_changed: true, ..p }));
        assert!(proof_due(&ProofDue { longest_pause_s: 60, ..p }) && !proof_due(&ProofDue { longest_pause_s: 59, ..p }));
        assert!(proof_due(&ProofDue { seconds_since_proof: 300, ..p }) && !proof_due(&ProofDue { seconds_since_proof: 299, ..p }));
    }

    #[test]
    fn an_item_is_accepted_only_above_since_and_above_the_one_before_it() {
        let items = crate::json::parse(br#"[{"seq":5},{"seq":5},{"seq":4},{"seq":9},{"seq":"x"},{"seq":10.0},{"seq":12},{"nope":1}]"#).unwrap();
        assert_eq!(accepted_items(items.as_array().unwrap(), 4), vec![0, 3, 6]);
        assert_eq!(accepted_items(items.as_array().unwrap(), 12), Vec::<usize>::new());
        let big = crate::json::parse(br#"[{"seq":9007199254740992},{"seq":9007199254740991}]"#).unwrap();
        assert_eq!(accepted_items(big.as_array().unwrap(), 0), vec![1], "2^53 is not a seq");
    }
}
