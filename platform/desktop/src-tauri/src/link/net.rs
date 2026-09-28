//! Saying why a request to the provider did not get there.
//!
//! reqwest's own message is "error sending request for url (...)" whatever
//! happened, which reads the same for a computer with no internet, a server
//! that is down and one that is slow. The cause is further down the error's
//! chain; this finds it and says it in words a person can act on.

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

/// How long a lane waits after a poll that came back with nothing, given how long
/// that poll took: at least half a second, and enough that the lane's polls are
/// two seconds apart. A provider that cuts its long polls short (FormLogic under
/// `php -S` answers in a second) would otherwise get a request a second from each
/// lane, over its rate limit, when the desktop has nothing to do.
pub fn idle_pause(polled_for: std::time::Duration) -> std::time::Duration {
    const CYCLE: std::time::Duration = std::time::Duration::from_secs(2);
    const LEAST: std::time::Duration = std::time::Duration::from_millis(500);
    CYCLE.saturating_sub(polled_for).max(LEAST)
}

#[cfg(test)]
mod idle_tests {
    use super::idle_pause;
    use std::time::Duration;

    #[test]
    fn an_empty_poll_is_followed_by_enough_of_a_pause() {
        // A long poll that waited server-side: the least pause.
        assert_eq!(idle_pause(Duration::from_secs(25)), Duration::from_millis(500));
        // One cut short at a second: the rest of the two seconds.
        assert_eq!(idle_pause(Duration::from_secs(1)), Duration::from_secs(1));
        assert_eq!(idle_pause(Duration::ZERO), Duration::from_secs(2));
    }
}
