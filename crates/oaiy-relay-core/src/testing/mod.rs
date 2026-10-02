//! Test support, public (feature `testing`) so that the desktop's and the phone's own tests and the end-to-end harness can drive the client without PHP: a fake clock whose pauses
//! cost no time, a seeded random source, a scripted HTTP client that records what it was sent, and an in-process stub relay ([`stub`]).
//!
//! Nothing in here is for a product build, and nothing in here is a second implementation of the protocol that a test could be written against instead of the contract: the
//! stub exists so that the client can be exercised without a PHP host, and is itself checked against the real relay by the L4 test (`tests/relay_php.rs`).

pub mod server;
pub mod stub;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use crate::client::clock::{Clock, Rng};
use crate::client::http::{Cancel, HttpClient, HttpRequest, HttpResponse, TransportError};

/// A clock that moves only when it is slept on or advanced: a loop that pauses for an hour finishes at once, and the pauses it took are recorded.
pub struct FakeClock {
    inner: Mutex<(Duration, i64)>,
    sleeps: Mutex<Vec<Duration>>,
    yield_real: bool,
}

impl FakeClock {
    /// A clock at monotonic time zero and wall time `unix`.
    pub fn new(unix: i64) -> FakeClock {
        FakeClock { inner: Mutex::new((Duration::ZERO, unix)), sleeps: Mutex::new(Vec::new()), yield_real: true }
    }

    /// Moves both clocks forward.
    pub fn advance(&self, d: Duration) {
        if let Ok(mut i) = self.inner.lock() {
            i.0 += d;
            i.1 += d.as_secs() as i64;
        }
    }

    /// Steps the wall clock by `seconds` (the owner sets the PC's clock) without moving the monotonic clock.
    pub fn step_wall(&self, seconds: i64) {
        if let Ok(mut i) = self.inner.lock() {
            i.1 += seconds;
        }
    }

    /// Every pause the clock was asked to sleep, in order.
    pub fn sleeps(&self) -> Vec<Duration> {
        self.sleeps.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Forgets the recorded pauses.
    pub fn clear_sleeps(&self) {
        if let Ok(mut s) = self.sleeps.lock() {
            s.clear();
        }
    }
}

impl Clock for FakeClock {
    fn unix_now(&self) -> i64 {
        self.inner.lock().map(|i| i.1).unwrap_or(0)
    }

    fn monotonic(&self) -> Duration {
        self.inner.lock().map(|i| i.0).unwrap_or_default()
    }

    fn sleep(&self, duration: Duration, cancel: &Cancel) -> bool {
        if cancel.is_cancelled() {
            return false;
        }
        if let Ok(mut s) = self.sleeps.lock() {
            s.push(duration);
        }
        self.advance(duration);
        if self.yield_real {
            // A millisecond of real time, so that a loop whose pauses cost nothing does not starve the thread that serves it.
            std::thread::sleep(Duration::from_millis(1));
        }
        !cancel.is_cancelled()
    }
}

/// SplitMix64: a small seeded generator, so that a test that draws jitter or ids is repeatable from its seed.
pub struct SeededRng(AtomicU64);

impl SeededRng {
    /// A generator from `seed`.
    pub fn new(seed: u64) -> SeededRng {
        SeededRng(AtomicU64::new(seed))
    }

    fn next(&self) -> u64 {
        let s = self.0.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::SeqCst).wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl Rng for SeededRng {
    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let n = self.next().to_le_bytes();
            chunk.copy_from_slice(&n[..chunk.len()]);
        }
    }
}

/// One thing a [`ScriptedHttp`] was asked, kept for the test to look at.
#[derive(Debug, Clone)]
pub struct Logged {
    /// `GET` or `POST`.
    pub method: String,
    /// The URL.
    pub url: String,
    /// The headers sent (a test that looks for a token looks here).
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Option<String>,
}

type Script = Box<dyn FnMut(&HttpRequest) -> Result<HttpResponse, TransportError> + Send>;

/// An HTTP client that answers from a closure and records every request. The closure is the whole relay: a test says what each request gets.
pub struct ScriptedHttp {
    script: Mutex<Script>,
    log: Mutex<Vec<Logged>>,
}

impl ScriptedHttp {
    /// A client that answers with `script`.
    pub fn new(script: impl FnMut(&HttpRequest) -> Result<HttpResponse, TransportError> + Send + 'static) -> ScriptedHttp {
        ScriptedHttp { script: Mutex::new(Box::new(script)), log: Mutex::new(Vec::new()) }
    }

    /// Every request so far, in order.
    pub fn log(&self) -> Vec<Logged> {
        self.log.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// The number of requests so far.
    pub fn count(&self) -> usize {
        self.log().len()
    }
}

impl HttpClient for ScriptedHttp {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        if request.cancel.is_cancelled() {
            return Err(TransportError::Cancelled);
        }
        if let Ok(mut l) = self.log.lock() {
            l.push(Logged {
                method: request.method.as_str().to_string(),
                url: request.url.clone(),
                headers: request.headers.clone(),
                body: request.body.as_ref().map(|b| String::from_utf8_lossy(b).into_owned()),
            });
        }
        let mut script = self.script.lock().map_err(|_| TransportError::Other("poisoned".into()))?;
        (script)(request)
    }
}

/// A JSON response with the headers a real relay adds.
pub fn json_response(status: u16, extra_headers: &[(&str, &str)], body: &str, relay_time: i64) -> HttpResponse {
    let mut headers = vec![
        ("content-type".to_string(), "application/json; charset=utf-8".to_string()),
        ("x-oaiy-relay".to_string(), "oaiy-relay/1".to_string()),
        ("x-oaiy-time".to_string(), relay_time.to_string()),
        ("cache-control".to_string(), "no-store".to_string()),
    ];
    for (k, v) in extra_headers {
        headers.push(((*k).to_ascii_lowercase(), (*v).to_string()));
    }
    HttpResponse { status, headers, body: body.as_bytes().to_vec() }
}
