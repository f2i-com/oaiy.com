//! Small helpers: base64, ids, clocks, log rings, and JSON conveniences.

use oaiy_engine::json::Json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Today's date where the computer is (year, month, day): UTC's moved by the computer's offset from it
/// ([`utc_offset`]). A day's outputs share a folder named for it, which a picture made before midnight UTC dated the day
/// before for anyone east of Greenwich.
pub fn local_date() -> (i64, u32, u32) {
    utc_date((now() as i64 + utc_offset()).max(0) as u64)
}

/// The computer's offset from UTC in seconds, as its OS says it now (`date +%z`; on Windows, .NET's local time zone
/// through PowerShell), asked once an hour (an offset changes on the hour), else 0.
pub fn utc_offset() -> i64 {
    static ASKED: Mutex<Option<(u64, i64)>> = Mutex::new(None);
    let hour = now() / 3600;
    if let Some((at, offset)) = *ASKED.lock().unwrap_or_else(|p| p.into_inner()) {
        if at == hour {
            return offset;
        }
    }
    let offset = ask_utc_offset().unwrap_or(0);
    *ASKED.lock().unwrap_or_else(|p| p.into_inner()) = Some((hour, offset));
    offset
}

fn ask_utc_offset() -> Option<i64> {
    #[cfg(unix)]
    {
        // "+1100", "-0430"
        let out = oaiy_host_env::command("date").arg("+%z").output().ok()?;
        let said = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let (sign, digits) = said.split_at_checked(1)?;
        if !out.status.success() || digits.len() != 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let (h, m): (i64, i64) = (digits[..2].parse().ok()?, digits[2..].parse().ok()?);
        let seconds = h * 3600 + m * 60;
        match sign {
            "+" => Some(seconds),
            "-" => Some(-seconds),
            _ => None,
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let out = oaiy_host_env::command("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", "[int][TimeZoneInfo]::Local.GetUtcOffset([DateTime]::UtcNow).TotalSeconds"])
            .creation_flags(0x0800_0000)
            .output()
            .ok()?;
        let seconds: i64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        (out.status.success() && seconds.abs() <= 14 * 3600).then_some(seconds)
    }
    #[cfg(not(any(unix, windows)))]
    None
}

/// The UTC date of `secs` since 1970 (Howard Hinnant's civil-from-days).
pub fn utc_date(secs: u64) -> (i64, u32, u32) {
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m as u32, d as u32)
}

pub fn now_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// A process-unique, hard-to-guess id: `prefix` + 24 characters.
pub fn random_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(7, |d| d.as_nanos() as u64);
    let mut s = nanos ^ (std::process::id() as u64) << 32 ^ COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    // Mix in a stack address so two processes started in the same instant differ.
    let local = 0u8;
    s ^= std::ptr::addr_of!(local) as u64;
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut out = String::from(prefix);
    for _ in 0..24 {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        out.push(alphabet[(s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % alphabet.len()] as char);
    }
    out
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\r' | b'\n' | b' ' => continue,
            _ => return Err("invalid base64".into()),
        };
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// The last `cap` lines of a subprocess's (or the studio's own) log, numbered
/// so a poller can ask for what it has not seen.
/// Numbered, timestamped lines: `(number, unix millis, text)`.
type Lines = VecDeque<(u64, u64, String)>;

pub struct LogRing {
    /// The last number handed out, and the lines kept.
    lines: Mutex<(u64, Lines)>,
    cap: usize,
}

impl LogRing {
    pub fn new(cap: usize) -> LogRing {
        LogRing { lines: Mutex::new((0, VecDeque::new())), cap }
    }

    pub fn push(&self, line: impl Into<String>) {
        let mut g = self.lines.lock().unwrap_or_else(|p| p.into_inner());
        g.0 += 1;
        let n = g.0;
        g.1.push_back((n, now_millis(), line.into()));
        while g.1.len() > self.cap {
            g.1.pop_front();
        }
    }

    /// Lines numbered above `after`, oldest first.
    pub fn since(&self, after: u64) -> Json {
        let g = self.lines.lock().unwrap_or_else(|p| p.into_inner());
        Json::Arr(
            g.1.iter()
                .filter(|(n, _, _)| *n > after)
                .map(|(n, t, l)| Json::obj([("n", Json::Int(*n as i64)), ("t", Json::Int(*t as i64)), ("line", Json::str(l))]))
                .collect(),
        )
    }

    pub fn tail(&self, n: usize) -> String {
        let g = self.lines.lock().unwrap_or_else(|p| p.into_inner());
        let skip = g.1.len().saturating_sub(n);
        g.1.iter().skip(skip).map(|(_, _, l)| l.as_str()).collect::<Vec<_>>().join("\n")
    }
}

/// `v[key]` as a string, or `default`.
pub fn str_or<'a>(v: &'a Json, key: &str, default: &'a str) -> &'a str {
    v.get(key).and_then(Json::as_str).unwrap_or(default)
}

pub fn int_or(v: &Json, key: &str, default: i64) -> i64 {
    v.get(key).and_then(Json::as_i64).unwrap_or(default)
}

pub fn num_or(v: &Json, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Json::as_f64).unwrap_or(default)
}

pub fn bool_or(v: &Json, key: &str, default: bool) -> bool {
    v.get(key).and_then(Json::as_bool).unwrap_or(default)
}

/// Set (or add) `key` in an object.
pub fn set(v: &mut Json, key: &str, value: Json) {
    if let Json::Obj(fields) = v {
        match fields.iter_mut().find(|(k, _)| k == key) {
            Some((_, slot)) => *slot = value,
            None => fields.push((key.into(), value)),
        }
    }
}

/// An OpenAI-style error body.
pub fn error_json(message: &str, kind: &str, code: &str) -> Json {
    Json::obj([(
        "error",
        Json::obj([
            ("message", Json::str(message)),
            ("type", Json::str(kind)),
            ("param", Json::Null),
            ("code", Json::str(code)),
        ]),
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_every_padding_length() {
        for n in 0..10 {
            let data: Vec<u8> = (0..n as u8).map(|i| i.wrapping_mul(37).wrapping_add(200)).collect();
            assert_eq!(base64_decode(&base64_encode(&data)).unwrap(), data);
        }
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert!(base64_decode("TW$u").is_err());
    }

    #[test]
    fn ids_are_distinct_and_logs_page_by_number() {
        let a = random_id("job_");
        assert!(a.starts_with("job_") && a.len() == 28);
        assert_ne!(a, random_id("job_"));
        let log = LogRing::new(3);
        for i in 0..5 {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.since(0).len(), 3);
        assert_eq!(log.since(4).len(), 1);
        assert_eq!(log.tail(2), "line 3\nline 4");
    }

    #[test]
    fn dates_are_civil_and_today_is_within_a_day_of_utcs() {
        assert_eq!(utc_date(0), (1970, 1, 1));
        assert_eq!(utc_date(951_782_400), (2000, 2, 29));
        assert_eq!(utc_date(1_791_936_000), (2026, 10, 14));
        // (a local date is UTC's, or the day before or after it: no zone is further than 14 hours from it)
        let (here, utc) = (local_date(), utc_date(now()));
        let day = |(y, m, d): (i64, u32, u32)| y * 400 + i64::from(m) * 32 + i64::from(d);
        assert!((day(here) - day(utc)).abs() <= 400, "{here:?} against UTC's {utc:?}");
        assert!((1..=12).contains(&here.1) && (1..=31).contains(&here.2), "{here:?}");
    }
}
