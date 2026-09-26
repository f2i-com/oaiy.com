//! Small helpers: base64, ids, clocks, log rings, and JSON conveniences.

use nrob::json::Json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
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
}
