//! The loopback HTTP client (`loopback-http`): what it promises as a transport (no redirect, a body cap, a timeout, a cancel, loopback only), against raw sockets that behave badly,
//! and a whole client run (enrolment, the loop, posting) over a real socket against the stub relay.

mod common;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use oaiy_relay_core::client::loopback::LoopbackHttp;
use oaiy_relay_core::client::*;
use oaiy_relay_core::enrol::{EnrolmentKey, Role};
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::testing::stub::{StubConfig, StubRelay};
use oaiy_relay_core::testing::SeededRng;
use oaiy_relay_core::url::RelayUrl;

fn request(url: &str, method: Method, body: Option<&str>, timeout: Duration, max: usize, cancel: &Cancel) -> HttpRequest {
    HttpRequest {
        method,
        url: url.to_string(),
        headers: vec![("Accept".into(), "application/json".into()), ("Authorization".into(), "Bearer secret-value".into())],
        body: body.map(|b| b.as_bytes().to_vec()),
        timeout,
        max_response_bytes: max,
        cancel: cancel.clone(),
    }
}

/// A server that accepts one connection, reads the request head (and returns it) and runs `then` on the stream.
fn raw_server(then: impl FnOnce(&mut TcpStream) + Send + 'static) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
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
        String::from_utf8_lossy(&got).into_owned()
    });
    (format!("http://{addr}"), handle)
}

const T: Duration = Duration::from_secs(5);

#[test]
fn a_body_with_a_content_length_a_chunked_body_and_one_that_ends_with_the_connection_are_all_read() {
    let (url, h) = raw_server(|s| {
        s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-OAIY-Time: 7\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2\r\nContent-Length: 11\r\n\r\n{\"ok\":true}").unwrap();
    });
    let r = LoopbackHttp.send(&request(&format!("{url}/v1/health?x=1"), Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap();
    assert_eq!((r.status, r.body.as_slice()), (200, b"{\"ok\":true}".as_slice()));
    assert_eq!(r.header("x-oaiy-time"), Some("7"), "names are lower case");
    assert_eq!(r.headers.iter().filter(|(k, _)| k == "set-cookie").count(), 2, "repeated headers are kept");
    let sent = h.join().unwrap();
    assert!(sent.starts_with("GET /v1/health?x=1 HTTP/1.1\r\n"), "{sent}");
    assert!(
        sent.contains("Connection: close\r\n") && sent.contains("Authorization: Bearer secret-value\r\n") && sent.contains("Host: 127.0.0.1:"),
        "{sent}"
    );

    let (url, _) = raw_server(|s| {
        s.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\n\r\n").unwrap();
    });
    let r = LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap();
    assert_eq!(r.body, b"hello world");

    let (url, _) = raw_server(|s| {
        s.write_all(b"HTTP/1.0 200 OK\r\n\r\nuntil the connection closes").unwrap();
    });
    let r = LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap();
    assert_eq!(r.body, b"until the connection closes");

    let (url, _) = raw_server(|s| {
        s.write_all(b"HTTP/1.1 204 No Content\r\nX-A: b\r\n\r\n").unwrap();
    });
    let r = LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap();
    assert_eq!((r.status, r.body.len()), (204, 0));
}

#[test]
fn a_post_sends_its_body_with_a_content_length() {
    let (url, h) = raw_server(|s| {
        // Read the rest of the body, which arrives after the head.
        let mut rest = [0u8; 64];
        let _ = s.set_read_timeout(Some(Duration::from_millis(300)));
        let _ = s.read(&mut rest);
        s.write_all(b"HTTP/1.1 201 Created\r\nContent-Length: 0\r\n\r\n").unwrap();
    });
    let r = LoopbackHttp.send(&request(&format!("{url}/v1/enroll"), Method::Post, Some("{\"a\":1}"), T, 1 << 20, &Cancel::new())).unwrap();
    assert_eq!(r.status, 201);
    let sent = h.join().unwrap();
    assert!(sent.starts_with("POST /v1/enroll HTTP/1.1\r\n") && sent.contains("Content-Length: 7\r\n"), "{sent}");
}

#[test]
fn a_redirect_is_returned_and_never_followed_so_a_bearer_cannot_leave_its_origin() {
    // The first server redirects to a second one; if the client followed, the second would get a connection (and the bearer).
    let second = TcpListener::bind("127.0.0.1:0").unwrap();
    second.set_nonblocking(true).unwrap();
    let target = format!("http://{}/stolen", second.local_addr().unwrap());
    let (url, _) = raw_server(move |s| {
        s.write_all(format!("HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\n\r\n").as_bytes()).unwrap();
    });
    let r = LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap();
    assert_eq!(r.status, 302);
    assert!(r.header("location").is_some());
    std::thread::sleep(Duration::from_millis(100));
    assert!(second.accept().is_err(), "nothing connected to the redirect target");
}

#[test]
fn a_body_that_is_too_large_too_short_or_too_slow_is_no_response() {
    // Declared larger than the cap.
    let (url, _) = raw_server(|s| {
        s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5000\r\n\r\nxxxxx").unwrap();
    });
    assert_eq!(LoopbackHttp.send(&request(&url, Method::Get, None, T, 1000, &Cancel::new())).unwrap_err(), TransportError::BodyTooLarge);
    // Chunked beyond the cap.
    let (url, _) = raw_server(|s| {
        let chunk = "a".repeat(600);
        let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n258\r\n{chunk}\r\n258\r\n{chunk}\r\n0\r\n\r\n").as_bytes());
    });
    assert_eq!(LoopbackHttp.send(&request(&url, Method::Get, None, T, 1000, &Cancel::new())).unwrap_err(), TransportError::BodyTooLarge);
    // A body that ends before its length.
    let (url, _) = raw_server(|s| {
        s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort").unwrap();
    });
    assert_eq!(LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap_err(), TransportError::ShortBody);
    // A connection that is closed with nothing.
    let (url, _) = raw_server(|_| {});
    assert!(matches!(
        LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap_err(),
        TransportError::Reset | TransportError::ShortBody
    ));
    // A server that never answers: the request's own timeout.
    let (url, _) = raw_server(|_s| std::thread::sleep(Duration::from_millis(1500)));
    let started = Instant::now();
    assert_eq!(
        LoopbackHttp.send(&request(&url, Method::Get, None, Duration::from_millis(300), 1 << 20, &Cancel::new())).unwrap_err(),
        TransportError::Timeout
    );
    assert!(started.elapsed() < Duration::from_millis(1200));
}

#[test]
fn a_cancel_ends_a_request_that_is_waiting_and_a_refused_connection_is_refused() {
    let (url, _) = raw_server(|_s| std::thread::sleep(Duration::from_millis(2000)));
    let cancel = Cancel::new();
    let c2 = cancel.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        c2.cancel();
    });
    let started = Instant::now();
    assert_eq!(LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &cancel)).unwrap_err(), TransportError::Cancelled);
    assert!(started.elapsed() < Duration::from_millis(1000), "{:?}", started.elapsed());
    canceller.join().unwrap();
    // A cancel that is set before the request is made never connects.
    assert_eq!(LoopbackHttp.send(&request(&url, Method::Get, None, T, 1 << 20, &cancel)).unwrap_err(), TransportError::Cancelled);
    // A port nobody listens on.
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    assert_eq!(
        LoopbackHttp.send(&request(&format!("http://127.0.0.1:{port}/"), Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap_err(),
        TransportError::Refused
    );
}

#[test]
fn it_refuses_every_host_but_loopback_and_every_scheme_but_http_before_it_connects() {
    for url in [
        "http://10.255.255.1/v1/info",
        "http://example.com/",
        "http://127.0.0.1.evil.test/",
        "http://[::1]:8080/",
        "https://127.0.0.1/",
        "ftp://127.0.0.1/",
        "http://127.0.0.1@evil.test/",
    ] {
        let started = Instant::now();
        let err = LoopbackHttp.send(&request(url, Method::Get, None, T, 1 << 20, &Cancel::new())).unwrap_err();
        assert!(matches!(err, TransportError::Other(_) | TransportError::Dns), "{url}: {err:?}");
        assert!(started.elapsed() < Duration::from_millis(500), "{url} must not even try");
    }
    // A header with a line break is refused, so that a credential cannot smuggle a header.
    let mut r = request("http://127.0.0.1:9/", Method::Get, None, T, 1 << 20, &Cancel::new());
    r.headers.push(("X-A".into(), "b\r\nX-Evil: 1".into()));
    assert!(matches!(LoopbackHttp.send(&r).unwrap_err(), TransportError::Other(_) | TransportError::Refused));
}

#[test]
fn a_whole_client_run_over_a_real_socket_enrolment_the_loop_and_a_post() {
    let stub = StubRelay::new(StubConfig { wait_default: 1, wait_max: 1, ..Default::default() });
    let server = stub.serve_loopback().unwrap();
    let url = RelayUrl::parse(&server.url()).unwrap();
    let clock = Arc::new(SystemClock::new());
    let make = |seed: u64, pin: Option<String>| {
        Arc::new(RelayClient::new(url.clone(), pin, Arc::new(LoopbackHttp), clock.clone(), Box::new(SeededRng::new(seed)), ClientConfig::default()))
    };
    // Enrolment with no pin: the key's `f` is the pin.
    let desktop = make(1, None);
    let key = EnrolmentKey::parse(&stub.mint_enrolment_key(Role::Desktop, 3600)).unwrap();
    let secrets = MemorySecretStore::new();
    let profiles = MemoryProfileStore::new();
    let ed = Signer::generate().unwrap().verify_key();
    let x = X25519Secret::generate().unwrap().public_key();
    let profile = enrol_and_store(&desktop, &key, "PC", &ed, &x, &secrets, &profiles, &Cancel::new()).unwrap();
    let token = Token::parse(std::str::from_utf8(&secrets.get(SECRET_TOKEN).unwrap().unwrap()).unwrap()).unwrap();
    // The loop, held by the relay for up to a second, and a post from a provider that wakes it.
    let sink = Arc::new(RecordingSink::new());
    let mut lp = PollLoop::new(desktop.clone(), token, MemoryPollStore::new(), sink.clone(), PollLoopConfig::default());
    let handle = lp.handle();
    let thread = std::thread::spawn(move || {
        let end = lp.run();
        (end, lp.into_store())
    });
    let (pid, ptoken) = stub.create_device("provider", "FormLogic", None, None, &[], None);
    let _ = pid;
    let ptoken = Token::parse(&ptoken).unwrap();
    let provider = make(2, Some(stub.relay_thumbprint()));
    provider.prove(&Cancel::new()).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let posted = provider
        .post_items(
            &ptoken,
            &[PostItem {
                to: format!("dev:{}", profile.device_id),
                lane: "cmd".into(),
                id: "c1".into(),
                ttl: None,
                hdr: Hdr::new().ct("sealed1"),
                body: "over a socket".into(),
            }],
            &Cancel::new(),
        )
        .unwrap();
    assert_eq!(posted[0].status, PostStatus::Queued);
    let start = Instant::now();
    while !sink.events().iter().any(|e| matches!(e, Event::Accepted { .. })) && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    handle.stop();
    let (end, store) = thread.join().unwrap();
    assert_eq!(end, LoopEnd::Cancelled);
    assert_eq!(store.accepted.len(), 1);
    assert_eq!(store.accepted[0].item.as_ref().unwrap().body, "over a socket");
    // The relay clock was sampled from the socket's answers.
    assert!(desktop.relay_now().is_some());
}
