//! Talking to the provider: saying why a request did not get there, and the
//! clients the lanes keep between their requests.
//!
//! reqwest's own message is "error sending request for url (...)" whatever
//! happened, which reads the same for a computer with no internet, a server
//! that is down and one that is slow. The cause is further down the error's
//! chain; this finds it and says it in words a person can act on.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Why `e` did not reach the other end: "formlogic.com can't be reached: it
/// refused the connection".
pub fn unreachable(e: &reqwest::Error) -> String {
    let host = e.url().and_then(|u| u.host_str()).unwrap_or("the server").to_string();
    format!("{host} can't be reached: {}", cause(e))
}

fn cause(e: &reqwest::Error) -> &'static str {
    let mut chain = Vec::new();
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = source {
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            match io.kind() {
                std::io::ErrorKind::ConnectionRefused => return "it refused the connection (is it running?)",
                std::io::ErrorKind::TimedOut => return "it did not answer in time",
                std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted => return "the connection was dropped",
                _ => {}
            }
        }
        chain.push(err.to_string().to_ascii_lowercase());
        source = err.source();
    }
    let said = |words: &[&str]| chain.iter().any(|m| words.iter().any(|w| m.contains(w)));
    if e.is_timeout() || said(&["timed out", "timeout"]) {
        "it did not answer in time"
    } else if said(&["dns", "lookup address", "no such host", "name or service not known", "nodename nor servname"]) {
        "its address could not be looked up (is this computer offline?)"
    } else if said(&["refused", "actively refused"]) {
        "it refused the connection (is it running?)"
    } else if said(&["certificate", "tls", "handshake"]) {
        "the secure connection failed"
    } else if e.is_connect() {
        "no connection could be made"
    } else {
        "the request did not complete"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_closed_port_reads_as_refused_not_as_a_bare_request_error() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let e = reqwest::blocking::Client::new().get(format!("http://127.0.0.1:{port}/x")).send().unwrap_err();
        let said = unreachable(&e);
        assert!(said.starts_with("127.0.0.1 can't be reached: "), "{said}");
        assert!(said.contains("refused") || said.contains("no connection"), "{said}");
        assert!(!said.contains("error sending request"), "{said}");
    }
}

/// How long a provider that refused for too many requests (429) asks to be left:
/// its `Retry-After` in seconds, at most two minutes (a lane is not parked for
/// longer on a provider's word), or None when it did not say.
pub fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let secs: u64 = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?.trim().parse().ok()?;
    Some(std::time::Duration::from_secs(secs.min(120)))
}

/// How long a lane waits after a poll that came back with nothing, given how long
/// that poll took: at least `least` (the provider's own setting, half a second
/// unless its descriptor says otherwise), and enough that the lane's polls are
/// two seconds apart. A provider that cuts its long polls short (FormLogic under
/// `php -S` answers in a second) would otherwise get a request a second from each
/// lane, over its rate limit, when the desktop has nothing to do.
pub fn idle_pause(polled_for: std::time::Duration, least: std::time::Duration) -> std::time::Duration {
    const CYCLE: std::time::Duration = std::time::Duration::from_secs(2);
    CYCLE.saturating_sub(polled_for).max(least)
}

// ---- the clients the lanes keep -------------------------------------------------
//
// A lane used to build a client for every poll or beat: a thread, a TLS
// configuration and, with them, a new connection and a new TLS handshake to the
// provider each time. A long-poll lane did that every 25 seconds, the queued-run
// check every 3. Each lane now keeps ONE client between its requests, so the
// requests that follow one another (a poll, the claim it leads to, the report)
// share a connection.
//
// What a client may not carry: a credential. The bearer is put on each request by
// the code that makes it, never on a client, so a client that a lane keeps cannot
// take one lane's credential to another lane or provider, and neither can a
// connection it reuses (HTTP keeps no login on a connection).
//
// Timeouts come in two kinds, and they are not the same. One set on a request is a
// single deadline for the whole exchange, from the request to the last byte of its
// reply. One set on a blocking client bounds the wait for the reply and, on a
// budget of its own that starts when the lane goes to read it, the reading. A lane
// whose requests differ (a poll waits, a claim does not) has to set them on the
// requests, and reads a reply as soon as it has it; a lane that waits between the
// two, out a `Retry-After`, reads first, or keeps its timeout on its client.

/// How long a connection may sit unused before it is not used again, for a lane
/// that comes back within seconds (the long-poll lanes).
///
/// A provider's web server closes a connection nobody has used for a few seconds
/// (Apache's default is five), and a request sent in the moment one is being
/// closed fails with no way to tell whether the provider saw it. A connection
/// idle for longer than this is not sent a request (the pool looks at its age when
/// it takes one), which is sooner than any common server closes theirs, so a
/// connection is reused only while it is very likely still open. That narrows the
/// case and does not remove it: a request already written to a connection the
/// server closes in that moment fails, and is not retried; the lane's back-off
/// takes it from there.
///
/// The pool clears out what has gone stale once each period, so a connection
/// nobody came back to is held for between one and two of them. That is why a lane
/// that will not be back within one does not rely on this to let go of it: see
/// [`Keep`].
pub const POOL_IDLE: Duration = Duration::from_secs(4);

/// How long a connection may sit unused before it is not used again, for a lane
/// whose requests come in bursts a long way apart: enough for the next request of
/// a burst, and no more.
pub const BURST_IDLE: Duration = Duration::from_secs(1);

/// How a lane's requests are spaced, which is what decides how long a connection
/// it has made is worth holding open. A connection that no request comes back to
/// is not free: the provider's web server holds a worker or a slot for it (all the
/// more on Apache's prefork and worker models), and did so before only for as long
/// as a request took, because every request made a client and dropped it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// The next request follows within seconds: a long poll and the claim and report
    /// it leads to, then the next poll after a short pause. Held for [`POOL_IDLE`].
    Between,
    /// Requests come in bursts, and bursts are far apart: the queue check (a claim
    /// and the graph follow a queued run at once, and the next look is 20 seconds
    /// away), a calendar sync. Held for [`BURST_IDLE`].
    Burst,
    /// One request, then nothing for a good while: a heartbeat. Closed as soon as
    /// its reply has been read, as it was when the client went with the request; the
    /// client is kept, with its TLS configuration and session cache, for the next.
    Never,
}

/// How long a lane goes on with one client before it builds another.
///
/// A client reads the computer's proxy settings when it is built (a lane that
/// built one for every poll noticed a change within a poll), so a lane that
/// keeps one would go on with the old settings for good. Five minutes bounds
/// that on a lane that never fails; one that does start afresh at once.
const CLIENT_LIFETIME: Duration = Duration::from_secs(300);

/// A blocking client as every lane starts it: the settings any lane shares, and
/// nothing that belongs to a credential. A lane adds its own (the sealed flows
/// lane turns redirects off) and gives its requests their timeouts.
pub fn blocking_builder(keep: Keep) -> reqwest::blocking::ClientBuilder {
    let builder = reqwest::blocking::Client::builder();
    match keep {
        Keep::Between => builder.pool_idle_timeout(POOL_IDLE),
        Keep::Burst => builder.pool_idle_timeout(BURST_IDLE),
        Keep::Never => builder.pool_max_idle_per_host(0),
    }
}

/// [`blocking_builder`] for the async client of the AI tunnel.
pub fn async_builder(keep: Keep) -> reqwest::ClientBuilder {
    let builder = reqwest::Client::builder();
    match keep {
        Keep::Between => builder.pool_idle_timeout(POOL_IDLE),
        Keep::Burst => builder.pool_idle_timeout(BURST_IDLE),
        Keep::Never => builder.pool_max_idle_per_host(0),
    }
}

/// The client one lane keeps between its requests.
///
/// Cloning a client shares its connections, so [`LaneClient::get`] hands out a
/// clone and the lane's requests go down the same ones. A new client is built
/// when there is none, when the one held has served [`CLIENT_LIFETIME`], and
/// after [`LaneClient::start_afresh`] (a lane calls it when a cycle failed, so the
/// try after a failure is on a client that has not seen the trouble: as it was
/// when every try built its own).
pub struct LaneClient<C> {
    build: fn() -> Result<C, String>,
    lifetime: Duration,
    held: Mutex<Option<(Instant, C)>>,
}

impl<C: Clone> LaneClient<C> {
    /// A lane's client, built by `build` (which says in its error which lane
    /// could not build one).
    pub const fn new(build: fn() -> Result<C, String>) -> Self {
        Self::lasting(build, CLIENT_LIFETIME)
    }

    /// As [`LaneClient::new`], for a lifetime of its own (a test's).
    pub const fn lasting(build: fn() -> Result<C, String>, lifetime: Duration) -> Self {
        Self { build, lifetime, held: Mutex::new(None) }
    }

    /// The client to use now.
    pub fn get(&self) -> Result<C, String> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((built, client)) = held.as_ref() {
            if built.elapsed() < self.lifetime {
                return Ok(client.clone());
            }
        }
        let client = (self.build)()?;
        *held = Some((Instant::now(), client.clone()));
        Ok(client)
    }

    /// Let go of the client held: the next [`LaneClient::get`] builds another.
    pub fn start_afresh(&self) {
        *self.held.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The lane is about to wait `pause` before its next request. A pause of
    /// [`POOL_IDLE`] or more is longer than a connection is kept for, so the next
    /// request would not use the one this lane has open, and until the pool cleared
    /// it out it would be a worker of the provider's web server doing nothing: the
    /// client is let go of now, as it was when every poll made its own.
    ///
    /// Only the lane knows the pause, and only a long-poll lane's comes from its
    /// descriptor (a provider that cannot hold polls asks for a hold of a second and
    /// a pause of five, to be a request every six seconds), so it is the lane that
    /// says. Any shorter pause keeps the client and its connection.
    ///
    /// Clones of a client share its pool, so a lane that still holds one from its
    /// last request has to drop it first, or the connection stays open through the
    /// pause after all.
    pub fn rest(&self, pause: Duration) {
        if pause >= POOL_IDLE {
            self.start_afresh();
        }
    }
}

/// How much of a reply nobody wanted is read to let its connection go on.
const DRAIN_AT_MOST: u64 = 1024 * 1024;

/// Finish with a reply whose body the caller has no use for.
///
/// A connection goes back to be reused only once its reply has been read to the
/// end; one dropped part way is closed. The replies a lane ignores are small
/// (a claim, a completion, an empty acknowledgement), but not always small enough
/// to have come in with their headers, so they are read out here rather than left
/// to chance.
pub fn drain(mut response: reqwest::blocking::Response) {
    use std::io::Read;
    let _ = std::io::copy(&mut (&mut response).take(DRAIN_AT_MOST), &mut std::io::sink());
}

/// [`drain`] for the async client.
pub async fn drain_async(mut response: reqwest::Response) {
    let mut read = 0u64;
    while let Ok(Some(chunk)) = response.chunk().await {
        read += chunk.len() as u64;
        if read > DRAIN_AT_MOST {
            break;
        }
    }
}

#[cfg(test)]
mod idle_tests {
    use super::idle_pause;
    use std::time::Duration;

    #[test]
    fn a_providers_retry_after_is_taken_within_reason() {
        let mut h = reqwest::header::HeaderMap::new();
        assert_eq!(super::retry_after(&h), None);
        h.insert(reqwest::header::RETRY_AFTER, "17".parse().unwrap());
        assert_eq!(super::retry_after(&h), Some(Duration::from_secs(17)));
        h.insert(reqwest::header::RETRY_AFTER, "3600".parse().unwrap());
        assert_eq!(super::retry_after(&h), Some(Duration::from_secs(120)));
        // A date instead of seconds: not understood, so the lane's own back-off.
        h.insert(reqwest::header::RETRY_AFTER, "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap());
        assert_eq!(super::retry_after(&h), None);
    }

    /// What a descriptor that says nothing about the pause gets (500 ms).
    const DEFAULT_LEAST: Duration = Duration::from_millis(500);

    #[test]
    fn an_empty_poll_is_followed_by_enough_of_a_pause() {
        // A long poll that waited server-side: the least pause.
        assert_eq!(idle_pause(Duration::from_secs(25), DEFAULT_LEAST), Duration::from_millis(500));
        // One cut short at a second: the rest of the two seconds.
        assert_eq!(idle_pause(Duration::from_secs(1), DEFAULT_LEAST), Duration::from_secs(1));
        assert_eq!(idle_pause(Duration::ZERO, DEFAULT_LEAST), Duration::from_secs(2));
    }

    #[test]
    fn a_provider_that_sets_the_pause_gets_at_least_that_and_still_the_two_second_spacing() {
        let five = Duration::from_secs(5);
        // A long poll: the pause the provider asked for, and no less.
        assert_eq!(idle_pause(Duration::from_secs(25), five), five);
        // One cut short: the provider's pause is longer than the spacing needs.
        assert_eq!(idle_pause(Duration::from_secs(1), five), five);
        assert_eq!(idle_pause(Duration::ZERO, five), five);
        // A pause shorter than the spacing does not shorten it: a provider that
        // answers at once is still not asked more than every two seconds.
        let tenth = Duration::from_millis(100);
        assert_eq!(idle_pause(Duration::from_secs(1), tenth), Duration::from_secs(1));
        assert_eq!(idle_pause(Duration::ZERO, tenth), Duration::from_secs(2));
        // …and after a full hold it is the provider's own, however short.
        assert_eq!(idle_pause(Duration::from_secs(25), tenth), tenth);
    }
}

#[cfg(test)]
mod client_tests {
    use super::*;
    use crate::link::testkit::{Provider, Reply};
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn a_lane_keeps_its_client_until_it_is_old_or_it_starts_afresh() {
        static BUILT: AtomicU32 = AtomicU32::new(0);
        fn build() -> Result<u32, String> {
            Ok(BUILT.fetch_add(1, Ordering::SeqCst) + 1)
        }
        let lane = LaneClient::new(build);
        assert_eq!(lane.get(), Ok(1));
        assert_eq!(lane.get(), Ok(1), "the client is kept, not built for each request");
        assert_eq!(lane.get(), Ok(1));
        lane.start_afresh();
        assert_eq!(lane.get(), Ok(2), "after a failed cycle the next try is on a new one");
        assert_eq!(lane.get(), Ok(2));
        assert_eq!(BUILT.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_lane_that_will_be_quiet_for_longer_than_a_connection_is_kept_lets_go_of_its_client() {
        static BUILT: AtomicU32 = AtomicU32::new(0);
        fn build() -> Result<u32, String> {
            Ok(BUILT.fetch_add(1, Ordering::SeqCst) + 1)
        }
        let lane = LaneClient::new(build);
        assert_eq!(lane.get(), Ok(1));
        // A pause the connection outlasts: the client, and its connection, are kept.
        lane.rest(Duration::ZERO);
        lane.rest(POOL_IDLE - Duration::from_millis(1));
        assert_eq!(lane.get(), Ok(1));
        // One it would not: let go now, and the next request is on a new client.
        lane.rest(POOL_IDLE);
        assert_eq!(lane.get(), Ok(2));
        lane.rest(Duration::from_secs(5));
        assert_eq!(lane.get(), Ok(3));
    }

    #[test]
    fn a_client_that_has_served_its_time_is_replaced() {
        static BUILT: AtomicU32 = AtomicU32::new(0);
        fn build() -> Result<u32, String> {
            Ok(BUILT.fetch_add(1, Ordering::SeqCst) + 1)
        }
        // The proxy settings a client read when it was built are the reason for a
        // limit: a lane that keeps one for good would never see them change.
        let lane = LaneClient::lasting(build, Duration::from_millis(60));
        assert_eq!(lane.get(), Ok(1));
        assert_eq!(lane.get(), Ok(1));
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(lane.get(), Ok(2));
        assert_eq!(lane.get(), Ok(2));
    }

    #[test]
    fn a_client_that_cannot_be_built_says_so_each_time_and_is_built_once_it_can() {
        static TRIES: AtomicU32 = AtomicU32::new(0);
        fn build() -> Result<u32, String> {
            match TRIES.fetch_add(1, Ordering::SeqCst) {
                0 | 1 => Err("could not build the test client: no TLS".into()),
                n => Ok(n),
            }
        }
        let lane = LaneClient::new(build);
        assert_eq!(lane.get(), Err("could not build the test client: no TLS".into()));
        assert_eq!(lane.get(), Err("could not build the test client: no TLS".into()));
        assert_eq!(lane.get(), Ok(2));
        assert_eq!(lane.get(), Ok(2), "and once built it is kept");
    }

    #[test]
    fn requests_one_after_another_share_a_connection_even_when_their_replies_are_not_read() {
        // A reply nobody reads is what a claim or a completion is to the lane that
        // makes it. Dropped part way, it would close the connection under the next
        // request; this one is too big to have come in with its headers.
        let server = Provider::start(|req| {
            if req.target == "/big" {
                Reply::ok(&format!(r#"{{"pad":"{}"}}"#, "x".repeat(300_000)))
            } else {
                Reply::ok("{}")
            }
        });
        let http = blocking_builder(Keep::Between).build().unwrap();
        drain(http.get(format!("{}/big", server.base)).send().unwrap());
        drain(http.post(format!("{}/claim", server.base)).body("{}").send().unwrap());
        assert_eq!(http.get(format!("{}/small", server.base)).send().unwrap().text().unwrap(), "{}");
        assert_eq!(server.connections(), 1, "{:?}", server.lines());
        assert_eq!(server.lines(), ["GET /big", "POST /claim", "GET /small"]);
    }

    #[test]
    fn a_connection_left_unused_for_longer_than_the_pool_allows_is_not_sent_a_request() {
        // A provider's web server closes a connection nobody has used for a few
        // seconds, and a request sent in the moment it does fails with no telling
        // whether it was seen. Our side lets go first.
        let server = Provider::start(|_| Reply::ok("{}"));
        let http = blocking_builder(Keep::Between).build().unwrap();
        drain(http.get(format!("{}/a", server.base)).send().unwrap());
        drain(http.get(format!("{}/b", server.base)).send().unwrap());
        assert_eq!(server.connections(), 1, "a request straight after one uses its connection");
        std::thread::sleep(POOL_IDLE + Duration::from_millis(700));
        drain(http.get(format!("{}/c", server.base)).send().unwrap());
        assert_eq!(server.connections(), 2, "one idle for {POOL_IDLE:?} is let go, not reused");
    }

    #[test]
    fn a_lane_that_comes_back_within_seconds_finds_its_connection_after_its_pause() {
        // The long-poll lanes wait half a second after an empty poll, and up to two
        // when the provider cuts its holds short, then poll again.
        let server = Provider::start(|_| Reply::ok("{}"));
        let http = blocking_builder(Keep::Between).build().unwrap();
        drain(http.get(format!("{}/a", server.base)).send().unwrap());
        std::thread::sleep(Duration::from_millis(1500));
        drain(http.get(format!("{}/b", server.base)).send().unwrap());
        assert_eq!(server.connections(), 1, "{:?}", server.lines());
    }

    #[test]
    fn a_lane_whose_requests_are_far_apart_closes_each_connection_when_its_reply_is_read() {
        let server = Provider::start(|_| Reply::ok("{}"));
        let http = blocking_builder(Keep::Never).build().unwrap();
        drain(http.get(format!("{}/a", server.base)).send().unwrap());
        drain(http.get(format!("{}/b", server.base)).send().unwrap());
        assert_eq!(server.connections(), 2, "nothing is kept for the next request to use");
        for conn in 0..2 {
            let held = server.closed_after_reply(conn, Duration::from_secs(10));
            assert!(held < Duration::from_secs(2), "connection {conn} was held open {held:?} after its reply");
        }
    }

    #[test]
    fn a_lane_that_makes_its_requests_in_bursts_keeps_a_connection_for_a_burst_and_no_longer() {
        let server = Provider::start(|_| Reply::ok("{}"));
        let http = blocking_builder(Keep::Burst).build().unwrap();
        for path in ["/a", "/b", "/c"] {
            drain(http.get(format!("{}{path}", server.base)).send().unwrap());
        }
        assert_eq!(server.connections(), 1, "the requests of a burst share a connection");
        // Closed soon after the last, and well inside the time a lane that comes
        // straight back keeps one.
        let held = server.closed_after_reply(0, Duration::from_secs(10));
        assert!(held < POOL_IDLE - Duration::from_secs(1), "held open {held:?} after its last reply");
    }

    #[test]
    fn a_client_carries_no_credential_and_a_request_carries_only_what_it_was_given() {
        // The bearer is put on each request by the code that makes it. A client
        // that held one would take it to whichever lane, or provider, next used it.
        let server = Provider::start(|_| Reply::ok("{}"));
        let http = blocking_builder(Keep::Between).build().unwrap();
        drain(http.get(format!("{}/bare", server.base)).send().unwrap());
        drain(http.get(format!("{}/first", server.base)).bearer_auth("flk_first").send().unwrap());
        drain(http.get(format!("{}/second", server.base)).bearer_auth("flk_second").send().unwrap());
        drain(http.get(format!("{}/bare-again", server.base)).send().unwrap());
        let seen = server.requests();
        let bearers: Vec<Option<&str>> = seen.iter().map(|r| r.header("authorization")).collect();
        assert_eq!(bearers, [None, Some("Bearer flk_first"), Some("Bearer flk_second"), None]);
        assert!(seen.iter().all(|r| r.header("cookie").is_none()), "no cookie jar either");
        assert_eq!(server.connections(), 1, "all four rode one connection, each with its own credential");
    }

    #[test]
    fn a_request_has_the_timeout_it_was_given_not_the_clients() {
        // A lane whose requests differ in length (a poll waits longer than a claim)
        // gives each its own timeout, which is what lets one client serve them all.
        let server = Provider::start(|_| {
            std::thread::sleep(Duration::from_millis(1500));
            Reply::ok("{}")
        });
        let http = blocking_builder(Keep::Between).build().unwrap();
        let started = Instant::now();
        let e = http
            .get(format!("{}/slow", server.base))
            .timeout(Duration::from_millis(250))
            .send()
            .unwrap_err();
        assert!(e.is_timeout(), "{e}");
        assert!(started.elapsed() < Duration::from_millis(1200), "{:?}", started.elapsed());
    }

    #[test]
    fn a_requests_timeout_runs_to_the_end_of_its_reply_and_a_clients_does_not() {
        // Why a lane that waits between the headers of a reply and its body (out a
        // `Retry-After`) reads first, or has its timeout on its client: the same
        // wait, longer than the timeout, is fine to a client's and fatal to a request's.
        let server = Provider::start(|_| Reply::ok(r#"{"message":"read late"}"#));

        let on_the_client = blocking_builder(Keep::Between).timeout(Duration::from_millis(300)).build().unwrap();
        let reply = on_the_client.get(format!("{}/client", server.base)).send().unwrap();
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(reply.text().unwrap(), r#"{"message":"read late"}"#);

        let on_the_request = blocking_builder(Keep::Between).build().unwrap();
        let reply = on_the_request
            .get(format!("{}/request", server.base))
            .timeout(Duration::from_millis(300))
            .send()
            .unwrap();
        std::thread::sleep(Duration::from_millis(600));
        let read = reply.text();
        assert!(read.is_err(), "the deadline of the request was past: {read:?}");
    }

    #[test]
    fn no_lane_builds_a_client_of_its_own_for_each_poll() {
        // What this replaced, kept from coming back: a `Client::builder()` in a
        // lane is a thread, a TLS configuration and a new connection per request.
        // The lanes' clients come from `LaneClient`, which builds them from
        // `blocking_builder` / `async_builder` here.
        for (name, source) in [
            ("relay.rs", include_str!("relay.rs")),
            ("sealed_flows.rs", include_str!("sealed_flows.rs")),
            ("flow_runner.rs", include_str!("flow_runner.rs")),
            ("heartbeat.rs", include_str!("heartbeat.rs")),
            ("data_node.rs", include_str!("data_node.rs")),
            ("ai/tunnel.rs", include_str!("../ai/tunnel.rs")),
            ("calendar/sync.rs", include_str!("../calendar/sync.rs")),
        ] {
            // Everything before the module of tests (a lane may have a helper of
            // its own for them above it).
            let source = source.replace("\r\n", "\n");
            let code = source.split("#[cfg(test)]\nmod tests").next().unwrap();
            assert!(code.len() < source.len(), "{name}: the module of tests was not found");
            for forbidden in ["Client::builder()", "Client::new()"] {
                assert!(!code.contains(forbidden), "{name} builds its own client with {forbidden}");
            }
            assert!(code.contains("LaneClient"), "{name} keeps no client between its requests");
            assert!(code.contains("start_afresh"), "{name} does not start afresh after a failed cycle");
        }
    }
}
