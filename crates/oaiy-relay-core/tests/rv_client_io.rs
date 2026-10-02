//! Review tests of the client / IO layer (reviewer fork F4). Each test is either a check that the claimed behaviour holds against a hostile peer (they pass), or the evidence
//! of a finding: a test named `finding_*` asserts the behaviour the layer should have, is `#[ignore]`d so a normal run stays green, and fails when run with `--ignored`
//! (the reason is in the ignore string). `evidence_*` tests print measurements (run with `--nocapture`) and assert only loose bounds.
//!
//! Everything binds `127.0.0.1` only.

#![allow(clippy::type_complexity, clippy::len_zero, clippy::manual_repeat_n)]

mod common;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use common::env::{env, quick, T0};
use oaiy_relay_core::client::loopback::LoopbackHttp;
use oaiy_relay_core::client::*;
use oaiy_relay_core::json;
use oaiy_relay_core::testing::stub::{StubConfig, StubRelay};
use oaiy_relay_core::testing::{json_response, ScriptedHttp};
use oaiy_relay_core::url::RelayUrl;

const T: Duration = Duration::from_secs(5);

fn request(url: &str, timeout: Duration, max: usize, cancel: &Cancel) -> HttpRequest {
    HttpRequest {
        method: Method::Get,
        url: url.to_string(),
        headers: vec![("Accept".into(), "application/json".into()), ("Authorization".into(), "Bearer secret-value".into())],
        body: None,
        timeout,
        max_response_bytes: max,
        cancel: cancel.clone(),
    }
}

/// A server that accepts one connection, reads the request head, and runs `then` on the stream (with a write timeout so that it ends when the client goes away).
fn raw_server(then: impl FnOnce(&mut TcpStream) + Send + 'static) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut buf = vec![0u8; 8192];
        let mut got = Vec::new();
        loop {
            let n = s.read(&mut buf).unwrap_or(0);
            got.extend_from_slice(&buf[..n]);
            if n == 0 || got.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        then(&mut s);
    });
    (format!("http://{addr}"), handle)
}

// ------------------------------------------------------------------------------------------------------------------------------ LoopbackHttp against a hostile server

/// F4-L1: a chunked body whose second chunk size is `ffffffffffffffff` overflows `decoded.len() + size` (debug: panic; release: wraps and slices out of range). A transport must
/// return an error for any bytes a peer sends.
#[test]
fn finding_chunk_size_overflow_is_an_error_and_not_a_panic() {
    let (url, _h) = raw_server(|s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\nffffffffffffffff\r\nxxxx");
        std::thread::sleep(Duration::from_millis(500));
    });
    let r = std::thread::spawn(move || LoopbackHttp.send(&request(&url, T, 1 << 20, &Cancel::new()))).join();
    match r {
        Err(_) => panic!("LoopbackHttp::send panicked on a hostile chunk size"),
        Ok(res) => assert!(res.is_err(), "{res:?}"),
    }
}

/// The same size as the first chunk is refused cleanly (decoded.len() is 0): the bound check works when nothing is decoded yet.
#[test]
fn a_huge_first_chunk_size_is_body_too_large() {
    let (url, _h) = raw_server(|s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nffffffffffffffff\r\nxxxx");
        std::thread::sleep(Duration::from_millis(300));
    });
    let r = LoopbackHttp.send(&request(&url, T, 1 << 20, &Cancel::new()));
    assert_eq!(r.unwrap_err(), TransportError::BodyTooLarge);
}

#[test]
fn a_10_mb_body_with_a_content_length_is_refused_at_once_and_a_1_gb_length_too() {
    for len in [10_000_000usize, 1_073_741_824usize] {
        let written = Arc::new(AtomicUsize::new(0));
        let w = written.clone();
        let (url, h) = raw_server(move |s| {
            let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {len}\r\n\r\n").as_bytes());
            let chunk = vec![b'x'; 16 * 1024];
            // Keep sending until the client goes away or 20 MB went out.
            while w.load(Ordering::SeqCst) < 20_000_000 {
                if s.write_all(&chunk).is_err() {
                    break;
                }
                w.fetch_add(chunk.len(), Ordering::SeqCst);
            }
        });
        let started = Instant::now();
        let r = LoopbackHttp.send(&request(&url, T, 1_114_112, &Cancel::new()));
        assert_eq!(r.unwrap_err(), TransportError::BodyTooLarge, "Content-Length {len}");
        assert!(started.elapsed() < Duration::from_secs(2));
        h.join().unwrap();
        // Refused on the declared length, before the body was read: the server could push only what the socket buffers hold.
        assert!(written.load(Ordering::SeqCst) < 8_000_000, "the server wrote {} bytes", written.load(Ordering::SeqCst));
    }
}

#[test]
fn a_body_with_no_length_that_never_ends_and_headers_that_never_end_are_bounded() {
    // No Content-Length, endless body.
    let (url, _h) = raw_server(|s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n");
        let chunk = vec![b'x'; 16 * 1024];
        while s.write_all(&chunk).is_ok() {}
    });
    let started = Instant::now();
    assert_eq!(LoopbackHttp.send(&request(&url, T, 1_114_112, &Cancel::new())).unwrap_err(), TransportError::BodyTooLarge);
    assert!(started.elapsed() < Duration::from_secs(3));
    // Endless header lines: no blank line ever.
    let (url, _h) = raw_server(|s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\n");
        let line = b"X-Pad: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n".repeat(200);
        while s.write_all(&line).is_ok() {}
    });
    let started = Instant::now();
    assert_eq!(LoopbackHttp.send(&request(&url, T, 1_114_112, &Cancel::new())).unwrap_err(), TransportError::BodyTooLarge);
    assert!(started.elapsed() < Duration::from_secs(3));
    // Endless one-byte chunks: bounded by the decoded size, in bounded time.
    let (url, _h) = raw_server(|s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n");
        let c = b"1\r\nx\r\n".repeat(4096);
        while s.write_all(&c).is_ok() {}
    });
    let started = Instant::now();
    let r = LoopbackHttp.send(&request(&url, T, 1_114_112, &Cancel::new()));
    println!("one-byte chunks: {:?} after {:?}", r.as_ref().map(|r| r.body.len()), started.elapsed());
    assert!(matches!(r, Err(TransportError::BodyTooLarge) | Err(TransportError::Timeout)), "{r:?}");
}

#[test]
fn a_slow_drip_cannot_hold_a_request_past_its_timeout() {
    // One byte every 20 ms of a response that would be valid if it ever finished: the total deadline, not a per-read timeout.
    let (url, _h) = raw_server(|s| {
        let head = b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n";
        for b in head.iter().chain(std::iter::repeat(&b'x').take(100_000)) {
            if s.write_all(&[*b]).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    let started = Instant::now();
    let r = LoopbackHttp.send(&request(&url, Duration::from_millis(1500), 1_114_112, &Cancel::new()));
    let took = started.elapsed();
    assert_eq!(r.unwrap_err(), TransportError::Timeout);
    assert!(took < Duration::from_millis(2200), "{took:?}");
    // The same drip in the header block.
    let (url, _h) = raw_server(|s| {
        for b in b"HTTP/1.1 200 OK\r\nX-A: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".iter() {
            if s.write_all(&[*b]).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(30));
        }
    });
    let started = Instant::now();
    let r = LoopbackHttp.send(&request(&url, Duration::from_millis(1200), 1_114_112, &Cancel::new()));
    assert_eq!(r.unwrap_err(), TransportError::Timeout);
    assert!(started.elapsed() < Duration::from_millis(2000));
}

#[test]
fn a_cancel_is_noticed_within_a_slice_while_the_server_drips() {
    let (url, _h) = raw_server(|s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n");
        for _ in 0..1000 {
            if s.write_all(b"x").is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    let cancel = Cancel::new();
    let c2 = cancel.clone();
    let at = Arc::new(Mutex::new(None));
    let a2 = at.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        *a2.lock().unwrap() = Some(Instant::now());
        c2.cancel();
    });
    let r = LoopbackHttp.send(&request(&url, T, 1_114_112, &cancel));
    let ended = Instant::now();
    assert_eq!(r.unwrap_err(), TransportError::Cancelled);
    let lag = ended.duration_since(at.lock().unwrap().unwrap());
    assert!(lag < Duration::from_millis(150), "cancel noticed after {lag:?}");
}

#[test]
fn odd_status_lines_and_length_headers_are_errors_or_plain_answers_never_panics() {
    for (wire, ok) in [
        (&b"HTTP/1.1 99999 X\r\nContent-Length: 0\r\n\r\n"[..], false),
        (b"HTTP/1.1\r\n\r\n", false),
        (b"\r\n\r\n", false),
        (b"HTTP/1.1 200\r\nContent-Length: 2\r\n\r\nok", true),
        (b"HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\nrest", true),
        (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 9\r\n\r\nok", true),
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\n\r\n2\r\nok\r\n0\r\n\r\n", true),
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n", false),
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2;x\r\nokXX0\r\n\r\n", true),
    ] {
        let (url, _h) = raw_server(move |s| {
            let _ = s.write_all(wire);
            std::thread::sleep(Duration::from_millis(100));
        });
        let r = std::thread::spawn(move || LoopbackHttp.send(&request(&url, Duration::from_millis(800), 1 << 20, &Cancel::new()))).join();
        let r = r.unwrap_or_else(|_| panic!("panicked on {:?}", String::from_utf8_lossy(wire)));
        println!(
            "{:<90} -> {}",
            String::from_utf8_lossy(wire).replace("\r\n", "|"),
            match &r {
                Ok(x) => format!("{} {:?}", x.status, String::from_utf8_lossy(&x.body)),
                Err(e) => format!("Err({e})"),
            }
        );
        if ok {
            // Either a clean answer or a clean error (a chunk with an extension and no CRLF after it is not valid).
            let _ = r;
        } else {
            assert!(r.is_err(), "{:?} should be an error", String::from_utf8_lossy(wire));
        }
    }
}

/// F4-L2 (low): the request line is built from the URL path with no check for CR or LF, so a path with a line break splits the request. The relay client only builds paths from
/// validated ids, so this needs a caller that passes an unvalidated id (`pair_decision`, `pair_reject` and `pair_burn` take `pid: &str` unchecked).
#[test]
fn finding_a_path_with_a_line_break_does_not_split_the_request() {
    let (listener_url, seen) = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = l.accept() {
                let mut buf = vec![0u8; 4096];
                let mut got = Vec::new();
                s.set_read_timeout(Some(Duration::from_millis(500))).ok();
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                    if got.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                let _ = tx.send(String::from_utf8_lossy(&got).into_owned());
            }
        });
        (format!("http://{addr}"), rx)
    };
    let evil = format!("{listener_url}/v1/x HTTP/1.1\r\nX-Injected: 1\r\nX-Rest:");
    let _ = LoopbackHttp.send(&request(&evil, Duration::from_secs(2), 1 << 20, &Cancel::new()));
    let got = seen.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
    assert!(!got.contains("X-Injected: 1\r\n"), "the injected header reached the server as a header: {got:?}");
}

#[test]
fn a_redirect_target_is_never_contacted_and_http_hosts_that_are_not_loopback_are_refused_before_a_socket() {
    // A listener that must stay untouched for every hostile URL below.
    let victim = TcpListener::bind("127.0.0.1:0").unwrap();
    victim.set_nonblocking(true).unwrap();
    let port = victim.local_addr().unwrap().port();
    let urls = [
        format!("http://127.0.0.1:{port}@evil.example/"),
        format!("http://evil.example@127.0.0.1:{port}/"),
        format!("http://localhost:{port}@evil.example/"),
        format!("http://localhost.evil.example:{port}/"),
        format!("http://127.0.0.1.evil.example:{port}/"),
        format!("http://LOCALHOST.:{port}/"),
        format!("http://127.0.0.1.:{port}/"),
        format!("http://0.0.0.0:{port}/"),
        format!("http://127.1:{port}/"),
        format!("http://2130706433:{port}/"),
        format!("http://[::1]:{port}/"),
        format!("http://127.0.0.2:{port}/"),
        format!("http://127.0.0.1:{port}#@evil.example/"),
        format!("http://127.0.0.1:{port}?@evil.example/"),
        format!("http://127.0.0.1\\@evil.example:{port}/"),
        format!("http://127.0.0.1:{port}x/"),
        format!("HTTP://127.0.0.1:{port}/"),
        format!(" http://127.0.0.1:{port}/"),
        format!("https://127.0.0.1:{port}/"),
        format!("http://127.0.0.1:{port}:{port}/"),
    ];
    for u in &urls {
        let r = LoopbackHttp.send(&request(u, Duration::from_millis(300), 1 << 20, &Cancel::new()));
        println!(
            "{u:<55} -> {}",
            match &r {
                Ok(x) => format!("ANSWERED {}", x.status),
                Err(e) => format!("Err({e})"),
            }
        );
        assert!(r.is_err(), "{u}");
    }
    std::thread::sleep(Duration::from_millis(100));
    assert!(victim.accept().is_err(), "a hostile URL made the client connect to the loopback listener");
}

// ------------------------------------------------------------------------------------------------------------------------------ URL verdicts

#[test]
fn evidence_relay_url_verdicts() {
    let inputs = [
        "https://relay.example.com",
        "https://RELAY.Example.COM",
        "https://relay.example.com.",
        "https://.",
        "https://..",
        "https://a..b",
        "https://-a",
        "https://a-",
        "https://127.1",
        "https://0x7f.1",
        "https://2130706433",
        "https://xn--e1afmkfd.xn--p1ai",
        "https://\u{43f}\u{440}\u{438}\u{43c}\u{435}\u{440}.\u{440}\u{444}",
        "https://\u{ff45}\u{ff58}\u{ff41}\u{ff4d}\u{ff50}\u{ff4c}\u{ff45}.com",
        "https://[::1]",
        "https://[fe80::1%25eth0]",
        "https://relay.example.com:00443",
        "https://relay.example.com:000443",
        "https://relay.example.com:+443",
        "https://relay.example.com:443:80",
        "https://relay.example.com:65535",
        "https://relay.example.com:65536",
        "https://relay.example.com:0",
        "https://relay.example.com:00000",
        "https://relay.example.com\\@evil.example",
        "https://evil.example\\.relay.example.com",
        "https://relay.example.com%2f@evil.example",
        "https://relay.example.com\t",
        "https://relay.example.com\n",
        " https://relay.example.com",
        "https://relay.example.com ",
        "https://user@relay.example.com",
        "https://relay.example.com:443/",
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "http://localhost",
        "http://LOCALHOST:80",
        "http://localhost.",
        "http://127.0.0.1.",
        "http://127.0.0.1.evil.example",
        "http://127.0.0.1@evil.example",
        "http://127.0.0.1:80@evil.example",
        "http://localhost.evil.example",
        "http://127.0.0.2",
        "http://0.0.0.0",
        "http://[::1]",
        "http://127.1",
        "http://2130706433",
        "http://localhost:80@evil.example",
        "http://evil.example#@127.0.0.1",
    ];
    let mut accepted_http_non_loopback = Vec::new();
    for t in inputs {
        let with = RelayUrl::parse_with(t, true);
        let without = RelayUrl::parse_with(t, false);
        println!(
            "{:<48} loopback-http: {:<40} product build: {}",
            format!("{t:?}"),
            match &with {
                Ok(u) => format!("OK {u}"),
                Err(_) => "refused".into(),
            },
            match &without {
                Ok(u) => format!("OK {u}"),
                Err(_) => "refused".into(),
            }
        );
        if let Ok(u) = &with {
            if !u.is_https() && !matches!(u.host(), "127.0.0.1" | "localhost") {
                accepted_http_non_loopback.push(t);
            }
        }
        if let Ok(u) = &without {
            assert!(u.is_https(), "a product build accepted plain http: {t}");
        }
    }
    assert!(accepted_http_non_loopback.is_empty(), "{accepted_http_non_loopback:?}");
    // Equality is on the parsed parts, and the scheme's default port is the same origin as none. (Changed with the fix: it used to assert that they differ.)
    assert_eq!(RelayUrl::parse("https://relay.example.com").unwrap(), RelayUrl::parse("https://relay.example.com:443").unwrap());
}

// ------------------------------------------------------------------------------------------------------------------------------ the poll loop against a hostile relay

/// A transport that answers every request but the polls from the stub (proof, enrolment) and the polls from a script.
struct Hybrid {
    stub: StubRelay,
    script: Mutex<Box<dyn FnMut(&HttpRequest) -> Result<HttpResponse, TransportError> + Send>>,
    polls: AtomicUsize,
    infos: AtomicUsize,
    sinces: Mutex<Vec<u64>>,
    /// The URL of every request that carried an Authorization header.
    authed_urls: Mutex<Vec<String>>,
}

impl Hybrid {
    fn new(stub: &StubRelay, script: impl FnMut(&HttpRequest) -> Result<HttpResponse, TransportError> + Send + 'static) -> Arc<Hybrid> {
        Arc::new(Hybrid {
            stub: stub.clone(),
            script: Mutex::new(Box::new(script)),
            polls: AtomicUsize::new(0),
            infos: AtomicUsize::new(0),
            sinces: Mutex::new(Vec::new()),
            authed_urls: Mutex::new(Vec::new()),
        })
    }
}

impl HttpClient for Hybrid {
    fn send(&self, r: &HttpRequest) -> Result<HttpResponse, TransportError> {
        if r.cancel.is_cancelled() {
            return Err(TransportError::Cancelled);
        }
        if r.header("authorization").is_some() {
            self.authed_urls.lock().unwrap().push(r.url.clone());
        }
        if r.url.contains("/v1/poll") {
            self.polls.fetch_add(1, Ordering::SeqCst);
            if let Some(s) = since_of(&r.url) {
                self.sinces.lock().unwrap().push(s);
            }
            (self.script.lock().unwrap())(r)
        } else {
            if r.url.contains("/v1/info") {
                self.infos.fetch_add(1, Ordering::SeqCst);
            }
            self.stub.handle(r)
        }
    }
}

fn since_of(url: &str) -> Option<u64> {
    url.split("since=").nth(1)?.split('&').next()?.parse().ok()
}

fn ok_body(epoch: &str, cursor: u64, items: &str) -> String {
    format!(r#"{{"v":1,"epoch":"{epoch}","cursor":{cursor},"items":[{items}],"more":false,"time":{T0},"hold":{{"granted":true}}}}"#)
}

fn client_over(e: &common::env::Env, http: Arc<dyn HttpClient>, clock: Arc<dyn Clock>) -> Arc<RelayClient> {
    Arc::new(RelayClient::new(
        RelayUrl::parse(&e.stub.public_url()).unwrap(),
        Some(e.stub.relay_thumbprint()),
        http,
        clock,
        Box::new(oaiy_relay_core::testing::SeededRng::new(41)),
        ClientConfig::default(),
    ))
}

struct Running {
    handle: PollHandle,
    sink: Arc<RecordingSink>,
    thread: std::thread::JoinHandle<(LoopEnd, Box<dyn std::any::Any + Send>)>,
}

fn run<S: PollStore + Send + 'static>(client: &Arc<RelayClient>, token: &oaiy_relay_core::ids::Token, store: S) -> Running {
    let sink = Arc::new(RecordingSink::new());
    let mut lp = PollLoop::new(client.clone(), token.clone(), store, sink.clone(), PollLoopConfig::default());
    let handle = lp.handle();
    let thread = std::thread::spawn(move || {
        let end = lp.run();
        (end, Box::new(lp.into_store()) as Box<dyn std::any::Any + Send>)
    });
    Running { handle, sink, thread }
}

impl Running {
    fn stop(self) -> (LoopEnd, Box<dyn std::any::Any + Send>) {
        self.handle.stop();
        self.thread.join().expect("loop thread")
    }
}

fn wait_for(what: &str, secs: u64, cond: impl Fn() -> bool) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(secs) {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("timed out waiting for {what}");
}

/// The measured rate of a loop whose relay says `pollGapMs: 0` and `wait.default: 0` and answers every poll at once with an empty 200. By README 5.1.1 P3 the pause after
/// idle is `pollGapMs`; the client has a floor of its own (`MIN_POLL_GAP_MS`, 250 ms, P3's default) and never polls faster than four a second whatever the relay advertises.
/// (Changed with the fix: it used to assert a tight loop, `n > 20`.)
#[test]
fn the_client_has_a_floor_under_a_poll_gap_of_zero() {
    let e = env(StubConfig { wait_default: 0, wait_max: 0, poll_gap_ms: 0, ..Default::default() });
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, move |_| Ok(json_response(200, &[], &ok_body(&epoch, 0, ""), T0)));
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, MemoryPollStore::new());
    std::thread::sleep(Duration::from_millis(500));
    let n = http.polls.load(Ordering::SeqCst);
    let (end, _) = running.stop();
    println!("polls in 0.5 s with pollGapMs 0 and wait 0: {n} ({end:?})");
    assert!(n <= 4, "a floor of 250 ms allows at most four polls in half a second, got {n}");
}

/// The same with the shipped numbers: 250 ms and a refused-hold style relay is paced at about four a second at most.
#[test]
fn the_shipped_poll_gap_paces_an_idle_loop_to_about_four_a_second() {
    let e = env(StubConfig { wait_default: 0, wait_max: 0, poll_gap_ms: 250, ..Default::default() });
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, move |_| Ok(json_response(200, &[], &ok_body(&epoch, 0, ""), T0)));
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, MemoryPollStore::new());
    std::thread::sleep(Duration::from_millis(2000));
    let n = http.polls.load(Ordering::SeqCst);
    let _ = running.stop();
    println!("polls in 2 s with pollGapMs 250: {n}");
    assert!((4..=9).contains(&n), "{n}");
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("rv-f4-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A relay that has a fresh 1 MiB batch of items for every poll, for ever: the client accepts, writes and acknowledges each batch with no pause and no cap, and the file store
/// keeps everything (nothing in the crate removes or limits `inbox.jsonl`).
#[test]
fn evidence_an_endless_stream_of_items_is_accepted_without_pause_and_without_a_cap() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let seq = Arc::new(AtomicUsize::new(0));
    let s2 = seq.clone();
    let body_text = "z".repeat(10_000);
    let http = Hybrid::new(&e.stub, move |r| {
        let since = since_of(&r.url).unwrap_or(0) as usize;
        let start = s2.load(Ordering::SeqCst).max(since);
        let items: Vec<String> = (1..=100)
            .map(|i| {
                format!(
                    r#"{{"seq":{},"id":"i{}","lane":"cmd","from":"relay","at":1,"exp":99999999999,"hdr":{{}},"body":"{}"}}"#,
                    start + i,
                    start + i,
                    body_text
                )
            })
            .collect();
        s2.store(start + 100, Ordering::SeqCst);
        Ok(json_response(200, &[], &ok_body(&epoch, (start + 100) as u64, &items.join(",")), T0))
    });
    let dir = scratch_dir("endless");
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, FilePollStore::new(&dir));
    let started = Instant::now();
    wait_for("30 one-MiB batches", 60, || http.polls.load(Ordering::SeqCst) >= 30);
    let took = started.elapsed();
    let (_, _) = running.stop();
    let size = std::fs::metadata(dir.join("inbox.jsonl")).map(|m| m.len()).unwrap_or(0);
    println!("30 batches in {took:?}; inbox.jsonl is {} MiB; no pause between batches", size / (1 << 20));
    assert!(size > 25 * (1 << 20));
    let _ = std::fs::remove_dir_all(&dir);
}

/// One answer of about 0.9 MiB that holds 60,000 items (`{"seq":N}`, each accepted although the client asked for `limit=32`): processed in linear time, all of them accepted.
#[test]
fn a_poll_answer_with_sixty_thousand_tiny_items_is_processed_quickly_and_all_accepted() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let items: Vec<String> = (1..=60_000).map(|i| format!(r#"{{"seq":{i}}}"#)).collect();
    let items = items.join(",");
    let sent = Arc::new(AtomicBool::new(false));
    let s2 = sent.clone();
    let http = Hybrid::new(&e.stub, move |r| {
        let since = since_of(&r.url).unwrap_or(0);
        if since == 0 && !s2.swap(true, Ordering::SeqCst) {
            Ok(json_response(200, &[], &ok_body(&epoch, 60_000, &items), T0))
        } else {
            Ok(json_response(200, &[], &ok_body(&epoch, since, ""), T0))
        }
    });
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, MemoryPollStore::new());
    let started = Instant::now();
    wait_for("the big answer to be accepted", 20, || running.sink.events().iter().any(|ev| matches!(ev, Event::Accepted { count: 60_000, .. })));
    println!("60,000 items accepted {:?} after the loop started", started.elapsed());
    let (_, store) = running.stop();
    let store = store.downcast::<MemoryPollStore>().unwrap();
    assert_eq!(store.accepted.len(), 60_000);
    assert_eq!(store.cursor().since, 60_000);
}

/// A relay that answers every poll `429` for ever: the pause reaches its cap (30 s with up to 20 percent jitter) and stays there; the counters do not wrap in any realistic run.
#[test]
fn an_endless_429_is_paced_at_the_cap_and_nothing_grows() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let http = Hybrid::new(&e.stub, |_| {
        Ok(json_response(429, &[("retry-after", "1")], r#"{"error":{"code":"rate_limited","message":"x","retryAfter":1}}"#, T0))
    });
    let clock = e.clock.clone();
    let client = client_over(&e, http.clone(), clock.clone());
    clock.clear_sleeps();
    let running = run(&client, &token, MemoryPollStore::new());
    wait_for("500 polls", 30, || http.polls.load(Ordering::SeqCst) >= 500);
    let (end, _) = running.stop();
    assert_eq!(end, LoopEnd::Cancelled);
    let sleeps = clock.sleeps();
    let max = sleeps.iter().map(|d| d.as_secs_f64()).fold(0.0, f64::max);
    let min_late = sleeps.iter().skip(20).map(|d| d.as_secs_f64()).fold(f64::MAX, f64::min);
    println!("{} pauses; longest {max:.2} s; shortest after the 20th {min_late:.2} s", sleeps.len());
    assert!(max < 36.001);
    assert!(min_late >= 30.0 - 1e-9, "the cap pause is 30 s: {min_late}");
}

/// stop() before run(): nothing is sent.
#[test]
fn a_stop_before_the_loop_starts_sends_nothing_and_a_second_stop_is_harmless() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.stub.clear_log();
    let client = client_over(&e, Arc::new(e.stub.clone()), e.clock.clone());
    let mut lp = PollLoop::new(client, token, MemoryPollStore::new(), Arc::new(RecordingSink::new()), PollLoopConfig::default());
    let h = lp.handle();
    h.stop();
    h.stop();
    h.network_changed();
    assert_eq!(lp.run(), LoopEnd::Cancelled);
    assert!(e.stub.log().is_empty(), "{:?}", e.stub.log().len());
}

/// A store whose persist blocks until released.
struct BlockingStore {
    entered: mpsc::Sender<()>,
    release: Arc<Mutex<mpsc::Receiver<()>>>,
    persisted: Arc<Mutex<Vec<u64>>>,
    cursor: PollCursor,
}

impl PollStore for BlockingStore {
    fn load(&mut self) -> Result<PollCursor, StoreError> {
        Ok(self.cursor.clone())
    }

    fn persist(&mut self, batch: &PersistBatch<'_>) -> Result<(), StoreError> {
        let _ = self.entered.send(());
        let _ = self.release.lock().unwrap().recv_timeout(Duration::from_secs(5));
        self.persisted.lock().unwrap().push(batch.since);
        self.cursor = PollCursor { since: batch.since, epoch: Some(batch.epoch.to_string()) };
        Ok(())
    }

    fn clear_epoch(&mut self) -> Result<(), StoreError> {
        self.cursor.epoch = None;
        Ok(())
    }
}

/// A stop that arrives while the store is writing: the write completes (the items are safe), the loop ends, and it does not send the acknowledging poll afterwards.
#[test]
fn a_stop_during_a_write_lets_the_write_finish_and_sends_no_further_poll() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, {
        let epoch = epoch.clone();
        move |r| {
            let since = since_of(&r.url).unwrap_or(0);
            let items = if since == 0 {
                r#"{"seq":1,"id":"a","lane":"cmd","from":"relay","at":1,"exp":99999999999,"hdr":{},"body":"b"}"#.to_string()
            } else {
                String::new()
            };
            Ok(json_response(200, &[], &ok_body(&epoch, since.max(1), &items), T0))
        }
    });
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let persisted = Arc::new(Mutex::new(Vec::new()));
    let store =
        BlockingStore { entered: entered_tx, release: Arc::new(Mutex::new(release_rx)), persisted: persisted.clone(), cursor: PollCursor::default() };
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, store);
    entered_rx.recv_timeout(Duration::from_secs(5)).expect("the store was entered");
    running.handle.stop();
    release_tx.send(()).unwrap();
    let (end, _) = running.thread.join().unwrap();
    assert_eq!(end, LoopEnd::Cancelled);
    assert_eq!(*persisted.lock().unwrap(), vec![1], "the write finished");
    assert_eq!(http.polls.load(Ordering::SeqCst), 1, "no poll was sent after the stop (so the item stays unacknowledged and comes again)");
}

/// A failed write moves nothing: the next poll carries the old `since`.
#[test]
fn after_a_failed_write_the_next_poll_carries_the_old_since() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, {
        let epoch = epoch.clone();
        move |r| {
            let since = since_of(&r.url).unwrap_or(0);
            let items = if since == 0 {
                r#"{"seq":1,"id":"a","lane":"cmd","from":"relay","at":1,"exp":99999999999,"hdr":{},"body":"b"}"#.to_string()
            } else {
                String::new()
            };
            Ok(json_response(200, &[], &ok_body(&epoch, since.max(1), &items), T0))
        }
    });
    let mut store = MemoryPollStore::new();
    store.fail_writes = 2;
    let client = client_over(&e, http.clone(), e.clock.clone());
    let running = run(&client, &token, store);
    wait_for("5 polls", 10, || http.polls.load(Ordering::SeqCst) >= 5);
    let (_, store) = running.stop();
    let sinces = http.sinces.lock().unwrap().clone();
    println!("since of each poll: {sinces:?}");
    assert_eq!(&sinces[..3], &[0, 0, 0], "two failed writes then a good one: the item is requested again each time");
    assert!(sinces[3..].iter().all(|s| *s == 1), "after the good write the poll acknowledges seq 1");
    let store = store.downcast::<MemoryPollStore>().unwrap();
    assert_eq!(store.accepted.len(), 1);
}

/// Many threads stopping and signalling network changes while the loop runs: no deadlock, it ends.
#[test]
fn a_storm_of_stop_and_network_changed_from_many_threads_ends_the_loop() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, move |_| Ok(json_response(200, &[], &ok_body(&epoch, 0, ""), T0)));
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, MemoryPollStore::new());
    let h = running.handle.clone();
    let done = Arc::new(AtomicBool::new(false));
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let (h, done) = (h.clone(), done.clone());
            std::thread::spawn(move || {
                let mut n = 0u32;
                while !done.load(Ordering::SeqCst) && n < 2000 {
                    if i == 0 && n == 1500 {
                        h.stop();
                    } else {
                        h.network_changed();
                    }
                    n += 1;
                    std::thread::sleep(Duration::from_micros(200));
                }
            })
        })
        .collect();
    let started = Instant::now();
    let (end, _) = running.thread.join().unwrap();
    done.store(true, Ordering::SeqCst);
    for t in threads {
        t.join().unwrap();
    }
    println!(
        "ended {end:?} after {:?}; /v1/info requests: {}; polls: {}",
        started.elapsed(),
        http.infos.load(Ordering::SeqCst),
        http.polls.load(Ordering::SeqCst)
    );
    assert_eq!(end, LoopEnd::Cancelled);
}

/// F4-P2 (medium): README 5.1.1 P1: a client that cancels a running poll "starts the new one no sooner than 250 ms after it started the one it cancels". The loop has the
/// code for it (`poll_loop.rs:258-263`) but it can never run: a network change is the only way the loop cancels a poll, and it is always followed by a proof (the iteration
/// that takes the flag proves and `continue`s), so by the time the poll section runs the flag is already taken and `network_changed` there is false. The replacement poll
/// then starts one proof round trip after the cancelled one, which is milliseconds.
#[test]
fn finding_a_replacement_poll_starts_no_sooner_than_250_ms_after_the_one_it_cancels() {
    let e = env(StubConfig { wait_default: 20, wait_max: 20, ..Default::default() });
    let (token, _) = e.enrol_desktop();
    let starts = Arc::new(Mutex::new(Vec::<Instant>::new()));
    let s2 = starts.clone();
    let http = Hybrid::new(&e.stub, move |r| {
        s2.lock().unwrap().push(Instant::now());
        // A held poll: it ends only when the client cancels it.
        while !r.cancel.is_cancelled() {
            std::thread::sleep(Duration::from_millis(2));
        }
        Err(TransportError::Cancelled)
    });
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, MemoryPollStore::new());
    wait_for("the first poll to be in flight", 5, || starts.lock().unwrap().len() >= 1);
    std::thread::sleep(Duration::from_millis(50));
    running.handle.network_changed();
    wait_for("the replacement poll", 5, || starts.lock().unwrap().len() >= 2);
    let _ = running.stop();
    let s = starts.lock().unwrap().clone();
    let gap = s[1].duration_since(s[0]);
    println!("the replacement poll started {gap:?} after the one it cancelled (this includes the 50 ms the test waited)");
    // The test waited 50 ms before cancelling, so a conforming client starts the next one at least 250 ms after the first started.
    assert!(gap >= Duration::from_millis(250), "{gap:?}");
}

/// A network change flood: each change calls for an identity proof, and the client debounces them: the loop waits for the network to be quiet for 100 ms (at most 1 s) before it
/// proves the relay on it. (Changed with the fix: it used to assert that 50 changes cost at least 25 proof requests.)
#[test]
fn a_flood_of_network_changes_is_debounced_into_a_few_proofs() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, move |_| Ok(json_response(200, &[], &ok_body(&epoch, 0, ""), T0)));
    let client = client_over(&e, http.clone(), Arc::new(SystemClock::new()));
    let running = run(&client, &token, MemoryPollStore::new());
    wait_for("the first proof and poll", 5, || http.polls.load(Ordering::SeqCst) >= 1);
    let before = http.infos.load(Ordering::SeqCst);
    for _ in 0..50 {
        running.handle.network_changed();
        std::thread::sleep(Duration::from_millis(30));
    }
    let after = http.infos.load(Ordering::SeqCst);
    let _ = running.stop();
    println!("50 network changes in 1.5 s -> {} proof requests", after - before);
    assert!((1..=4).contains(&(after - before)), "{} proof requests for 50 changes in 1.5 s", after - before);
}

/// F4-C1 (low): the epoch a store hands back is put into the query string as it is. `FilePollStore::load` checks it; a host's own `PollStore` (a database, a platform
/// preferences file) need not, and the loop does not.
#[test]
fn finding_an_epoch_from_a_host_store_is_checked_before_it_reaches_the_request_line() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let epoch = e.stub.epoch();
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let s2 = seen.clone();
    let http = Hybrid::new(&e.stub, move |r| {
        s2.lock().unwrap().push(r.url.clone());
        Ok(json_response(200, &[], &ok_body(&epoch, 0, ""), T0))
    });
    let client = client_over(&e, http.clone(), e.clock.clone());
    let running = run(&client, &token, MemoryPollStore::at(0, Some("x&since=99&wait=0 HTTP/1.1")));
    wait_for("a poll", 5, || http.polls.load(Ordering::SeqCst) >= 1);
    let _ = running.stop();
    let url = seen.lock().unwrap()[0].clone();
    assert!(!url.contains("since=99"), "the poll URL carries a damaged epoch as it is: {url}");
}

/// F4-P1 (low/medium): a `200` from `GET /v1/info` that is not a proof (an HTML maintenance page, a proxy's body) is not an answer that does not verify: it is paced and retried
/// like a `503` of the same host, and the loop ends only on a stop. (Changed with the fix: it used to assert that the loop ended with `report_relay_changed`.) An answer that claims
/// to be a proof (it has the headers) and does not verify still ends the loop.
#[test]
fn a_non_proof_200_for_info_is_paced_and_retried_like_a_503() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let page = ScriptedHttp::new(|_| {
        Ok(HttpResponse {
            status: 200,
            headers: vec![("content-type".into(), "text/html".into())],
            body: b"<html><body>This account has been suspended</body></html>".to_vec(),
        })
    });
    let client = client_over(&e, Arc::new(page), e.clock.clone());
    let mut lp = PollLoop::new(client, token.clone(), MemoryPollStore::new(), Arc::new(RecordingSink::new()), PollLoopConfig::default());
    let h = lp.handle();
    let t = std::thread::spawn(move || lp.run());
    std::thread::sleep(Duration::from_millis(300));
    h.stop();
    let end = t.join().unwrap();
    println!("a 200 HTML page for the proof: {end:?}");
    assert_eq!(end, LoopEnd::Cancelled, "the loop kept trying until it was stopped");
    // The same host answering 503 instead is a failure that is paced and retried (the loop would not end on its own).
    let down = ScriptedHttp::new(|_| Ok(json_response(503, &[("retry-after", "1")], "{}", T0)));
    let client = client_over(&e, Arc::new(down), e.clock.clone());
    let lp = PollLoop::new(client, token, MemoryPollStore::new(), Arc::new(RecordingSink::new()), PollLoopConfig::default());
    let h = lp.handle();
    let mut lp = lp;
    let t = std::thread::spawn(move || lp.run());
    std::thread::sleep(Duration::from_millis(300));
    h.stop();
    assert_eq!(t.join().unwrap(), LoopEnd::Cancelled);
}

/// A hold of 20 s over a real socket: `stop()` ends the loop within a few slices, not at the end of the hold.
#[test]
fn a_stop_ends_a_held_poll_over_a_real_socket_within_a_fraction_of_a_second() {
    let stub = StubRelay::new(StubConfig { wait_default: 20, wait_max: 20, ..Default::default() });
    let server = stub.serve_loopback().unwrap();
    let url = RelayUrl::parse(&server.url()).unwrap();
    let clock = Arc::new(SystemClock::new());
    let make = |seed: u64, pin: Option<String>| {
        Arc::new(RelayClient::new(
            url.clone(),
            pin,
            Arc::new(LoopbackHttp),
            clock.clone(),
            Box::new(oaiy_relay_core::testing::SeededRng::new(seed)),
            ClientConfig::default(),
        ))
    };
    let desktop = make(1, None);
    let key = oaiy_relay_core::enrol::EnrolmentKey::parse(&stub.mint_enrolment_key(oaiy_relay_core::enrol::Role::Desktop, 3600)).unwrap();
    let secrets = MemorySecretStore::new();
    let profiles = MemoryProfileStore::new();
    let ed = oaiy_relay_core::keys::Signer::generate().unwrap().verify_key();
    let x = oaiy_relay_core::keys::X25519Secret::generate().unwrap().public_key();
    enrol_and_store(&desktop, &key, "PC", &ed, &x, &secrets, &profiles, &Cancel::new()).unwrap();
    let token = oaiy_relay_core::ids::Token::parse(std::str::from_utf8(&secrets.get(SECRET_TOKEN).unwrap().unwrap()).unwrap()).unwrap();
    let running = run(&desktop, &token, MemoryPollStore::new());
    // The stub logs a request when it has answered it, so a poll that is being held is not in its log: give the loop time to prove and send it.
    std::thread::sleep(Duration::from_millis(1000));
    assert!(!stub.log().iter().any(|r| r.target.starts_with("/v1/poll")), "the poll is held, not answered");
    let started = Instant::now();
    let (end, _) = running.stop();
    let took = started.elapsed();
    println!("stop() of a held poll took {took:?}");
    assert_eq!(end, LoopEnd::Cancelled);
    assert!(took < Duration::from_millis(500), "{took:?}");
}

/// F4-D1 (low): `HttpResponse` derives `Debug`, which prints the whole body; the body of `POST /v1/enroll` and `POST /v1/tokens/rotate` holds the device token. (`HttpRequest`
/// prints no header value for this reason; its response twin does not.)
#[test]
fn finding_http_response_debug_does_not_print_a_token_body() {
    let r =
        HttpResponse { status: 201, headers: vec![], body: br#"{"token":"oaiyrt1.AAAAAAAAAAA.SECRETSECRETSECRETSECRETSECRETSECRETSECR"}"#.to_vec() };
    let printed = format!("{r:?}");
    // Vec<u8> prints as decimal numbers; look for the bytes of "SECRET" (83, 69, 67, 82, 69, 84).
    assert!(!printed.contains("83, 69, 67, 82, 69, 84"), "Debug of an HttpResponse prints the body bytes");
}

/// F4-D2 (low): `IceServer` derives `Debug`, and `MobileAdmission` and `PluginAdmission` derive it over their `ice_servers`, so `{:?}` of an admission prints the TURN credential
/// (the admission bearer itself is redacted). The README says no secret is in `Debug` output.
#[test]
fn finding_the_debug_of_an_ice_server_does_not_print_its_turn_credential() {
    let s = oaiy_relay_core::admission::IceServer {
        urls: vec!["turns:turn.example.com:443".into()],
        username: "1790000000:abc".into(),
        credential: "TURN-SECRET-CREDENTIAL".into(),
        expires_at: Some(1_790_000_000),
    };
    assert!(!format!("{s:?}").contains("TURN-SECRET-CREDENTIAL"));
}

/// What a failed cursor write does (a file another process holds without delete sharing, as a virus scanner or a backup tool can): the write fails, the temporary file is
/// removed, the loop paces it as a failure; and the items that were appended first are appended again by every re-delivery.
#[cfg(windows)]
#[test]
fn evidence_a_locked_cursor_file_fails_the_write_cleanly_and_the_items_are_appended_again_each_time() {
    use std::os::windows::fs::OpenOptionsExt;
    let dir = scratch_dir("locked");
    let mut store = FilePollStore::new(&dir);
    let raw = json::parse(br#"{"seq":1,"id":"a","lane":"cmd","from":"relay","at":1,"exp":99,"hdr":{},"body":"b"}"#).unwrap();
    let items = vec![AcceptedItem { seq: 1, item: Item::from_json(&raw), raw }];
    store.persist(&PersistBatch { items: &items, since: 1, epoch: "eVp54C0-EJY", reset: false }).unwrap();
    // FILE_SHARE_READ only: no FILE_SHARE_DELETE, so a rename over the file is refused.
    let held = std::fs::OpenOptions::new().read(true).share_mode(1).open(dir.join("cursor.json")).unwrap();
    for _ in 0..2 {
        let r = store.persist(&PersistBatch { items: &items, since: 2, epoch: "eVp54C0-EJY", reset: false });
        println!("persist with the cursor locked: {r:?}");
        assert!(r.is_err());
    }
    let leftovers: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    println!("files: {leftovers:?}");
    assert!(leftovers.iter().all(|n| !n.contains(".tmp")), "{leftovers:?}");
    drop(held);
    assert_eq!(store.load().unwrap().since, 1, "the cursor did not move");
    println!("inbox lines after two failed writes: {}", store.read_inbox().unwrap().len());
    assert_eq!(store.read_inbox().unwrap().len(), 3, "the batch was appended on every attempt");
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------------------------------------------------------------------ stores and clock

/// F4-S1: a crash in the middle of an append to `inbox.jsonl` leaves a partial last line with no newline. The cursor was not advanced, so the same items are delivered again and are
/// appended after the fragment: the first re-delivered item shares a line with the fragment, the line cannot be parsed, and `read_inbox` fails for the whole file.
#[test]
#[ignore = "F4-S1: a torn append is never repaired; the next append fuses with it (store.rs:553-558) and read_inbox fails for the whole inbox"]
fn finding_a_torn_inbox_line_does_not_corrupt_the_items_that_come_again() {
    let dir = scratch_dir("torn");
    let mut store = FilePollStore::new(&dir);
    // The crash: half of the first line is on disk.
    std::fs::write(dir.join("inbox.jsonl"), br#"{"seq":1,"id":"a","lane":"cmd","fro"#).unwrap();
    let raw = json::parse(br#"{"seq":1,"id":"a","lane":"cmd","from":"relay","at":1,"exp":99,"hdr":{},"body":"b"}"#).unwrap();
    let items = vec![AcceptedItem { seq: 1, item: Item::from_json(&raw), raw: raw.clone() }];
    store.persist(&PersistBatch { items: &items, since: 1, epoch: "eVp54C0-EJY", reset: false }).unwrap();
    let inbox = store.read_inbox();
    let _ = std::fs::remove_dir_all(&dir);
    let inbox = inbox.expect("the inbox is unreadable after a torn append was followed by a good one");
    assert!(inbox.iter().any(|i| i.get_str("id") == Some("a")), "the re-delivered item is in the inbox");
}

/// The inbox has no way to be drained or trimmed: the API is `persist` and `read_inbox`. (Evidence by construction; this test only appends and reads.)
#[test]
fn evidence_the_file_store_only_grows() {
    let dir = scratch_dir("grows");
    let mut store = FilePollStore::new(&dir);
    let raw = json::parse(br#"{"seq":1,"id":"a","lane":"cmd","from":"relay","at":1,"exp":99,"hdr":{},"body":"b"}"#).unwrap();
    let items = vec![AcceptedItem { seq: 1, item: Item::from_json(&raw), raw }];
    for i in 0..3u64 {
        store.persist(&PersistBatch { items: &items, since: i + 1, epoch: "eVp54C0-EJY", reset: false }).unwrap();
    }
    assert_eq!(store.read_inbox().unwrap().len(), 3, "the same item three times: no de-duplication, no trimming");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evidence_the_first_relay_time_sample_is_applied_at_once_whatever_it_says() {
    // `GET /v1/health` needs no proof and no credential, and every response is sampled.
    let clock = Arc::new(oaiy_relay_core::testing::FakeClock::new(T0));
    let lie = Arc::new(Mutex::new(T0 + 10 * 365 * 86_400));
    let l2 = lie.clone();
    let http = ScriptedHttp::new(move |_| Ok(json_response(200, &[], r#"{"ok":true,"time":1,"authHeaderSeen":false}"#, *l2.lock().unwrap())));
    let client = RelayClient::new(
        RelayUrl::parse("https://relay.stub.test").unwrap(),
        None,
        Arc::new(http),
        clock.clone(),
        Box::new(oaiy_relay_core::testing::SeededRng::new(1)),
        ClientConfig::default(),
    );
    client.health(&Cancel::new()).unwrap();
    let now = client.relay_now().unwrap();
    println!("after one response: relay_now is local + {} days; clock_mismatch = {}", (now - T0) / 86_400, client.clock_mismatch());
    assert!(now - T0 > 3000 * 86_400, "the first sample is applied in full");
    assert!(client.clock_mismatch());
    // A relay at the largest time the header grammar allows (16 digits) does not overflow anything.
    *lie.lock().unwrap() = 9_999_999_999_999_999;
    for _ in 0..8 {
        client.health(&Cancel::new()).unwrap();
    }
    assert!(client.relay_now().unwrap() > T0);
    // And a relay that goes back to the past.
    *lie.lock().unwrap() = 0;
    for _ in 0..8 {
        client.health(&Cancel::new()).unwrap();
    }
    assert!(client.relay_now().is_some());
}

/// An adapter that returns header names as the server wrote them (`X-OAIY-Proof`, as OkHttp and many HTTP/1 stacks do) and not in lower case, which the `HttpClient` contract
/// asks for and nothing checks.
struct MixedCase(StubRelay);

impl HttpClient for MixedCase {
    fn send(&self, r: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let mut resp = self.0.handle(r)?;
        for (k, _) in resp.headers.iter_mut() {
            *k = k
                .split('-')
                .map(|p| {
                    let mut c = p.chars();
                    c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join("-");
        }
        Ok(resp)
    }
}

/// F4-H1 (low/medium): `RelayClient` reads response headers by their lower-case names and trusts the adapter to have lower-cased them. An adapter that does not (an OkHttp
/// one that returns names as sent) makes `prove` fail with "no X-OAIY-Proof" - the client then reports the relay as "not who it was" and the loop ends - and silently loses
/// `Retry-After` and `X-OAIY-Time`. One `to_ascii_lowercase` in `exchange` would remove the failure mode.
#[test]
fn finding_headers_are_matched_whatever_the_adapter_did_with_their_case() {
    let e = env(quick());
    let client = client_over(&e, Arc::new(MixedCase(e.stub.clone())), e.clock.clone());
    match client.prove(&Cancel::new()) {
        Ok(_) => {}
        Err(err) => panic!("the proof of a real relay failed because of header name case: {err:?}"),
    }
}

/// F4-K1 (low/medium): `clock.rs` says the offset "is kept on the monotonic clock so that a step of the PC's wall clock does not move it", but `remote_now` is
/// `local_wall + offset`, so a step of the wall clock moves the relay time 1:1, and the offset (median of the last five samples, slewed 1 s a minute) takes
/// `step / 1 s/min` to follow: a step of one hour leaves the relay time an hour wrong for about sixty hours. `FakeClock::step_wall` exists for this and no test uses it.
#[test]
#[ignore = "F4-K1: relay_now follows a wall-clock step 1:1 and recovers at 1 s a minute (clock.rs:167-169); the doc comment says a step does not move it"]
fn finding_the_relay_time_does_not_follow_a_step_of_the_wall_clock() {
    let clock = Arc::new(oaiy_relay_core::testing::FakeClock::new(T0));
    let truth = {
        let c = clock.clone();
        move || T0 + c.monotonic().as_secs() as i64
    };
    let t2 = truth.clone();
    let http = ScriptedHttp::new(move |_| Ok(json_response(200, &[], r#"{"ok":true,"time":1,"authHeaderSeen":false}"#, t2())));
    let client = RelayClient::new(
        RelayUrl::parse("https://relay.stub.test").unwrap(),
        None,
        Arc::new(http),
        clock.clone(),
        Box::new(oaiy_relay_core::testing::SeededRng::new(1)),
        ClientConfig::default(),
    );
    client.health(&Cancel::new()).unwrap();
    clock.advance(Duration::from_secs(100));
    client.health(&Cancel::new()).unwrap();
    assert_eq!(client.relay_now().unwrap(), truth());
    // The owner (or NTP) sets the PC's clock forward by an hour.
    clock.step_wall(3600);
    let right_after = client.relay_now().unwrap() - truth();
    // Ten minutes later, with a sample every ten seconds.
    for _ in 0..60 {
        clock.advance(Duration::from_secs(10));
        client.health(&Cancel::new()).unwrap();
    }
    let later = client.relay_now().unwrap() - truth();
    println!("relay_now is wrong by {right_after} s at once after a 3600 s step, and still by {later} s ten minutes and sixty good samples later");
    assert!(right_after.abs() <= 5, "wrong by {right_after} s at once");
    assert!(later.abs() <= 5, "wrong by {later} s after ten minutes");
}

/// A response with a bearer-looking `Authorization` echo, a relay error body with secrets, and a `Debug` of every public client type: none of the token's secret half appears.
#[test]
fn the_token_never_appears_in_debug_output_errors_or_events() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let secret_half = token.expose().rsplit('.').next().unwrap().to_string();
    let epoch = e.stub.epoch();
    let http = Hybrid::new(&e.stub, move |_| Ok(json_response(500, &[], "boom", T0)));
    let client = client_over(&e, http.clone(), e.clock.clone());
    let running = run(&client, &token, MemoryPollStore::new());
    wait_for("5 polls", 10, || http.polls.load(Ordering::SeqCst) >= 5);
    let sink = running.sink.clone();
    let (end, _) = running.stop();
    let mut all = format!("{:?}{:?}{:?}{:?}", sink.events(), end, token, PollRequest { since: 1, epoch: Some(epoch), wait_s: 1, limit: 1 });
    all.push_str(&format!(
        "{:?}",
        HttpRequest {
            method: Method::Get,
            url: "x".into(),
            headers: vec![("Authorization".into(), format!("Bearer {}", token.expose()))],
            body: None,
            timeout: T,
            max_response_bytes: 1,
            cancel: Cancel::new()
        }
    ));
    assert!(!all.contains(&secret_half));
    // The request headers do carry it, and only to the pinned origin.
    let urls = http.authed_urls.lock().unwrap().clone();
    assert!(urls.len() >= 5);
    let origin = e.stub.public_url();
    assert!(urls.iter().all(|u| u.starts_with(&format!("{origin}/v1/"))), "{urls:?}");
}
