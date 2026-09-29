//! Reading the feed over the network.
//!
//! A plain GET of one small JSON document, from an address fixed in the build
//! ([`super::FEED_URL`]; see [`super::FeedSource`] for the one debug-build exception). It has a
//! timeout, follows redirects only to https addresses (GitHub sends `releases/latest/download/…`
//! through two of them), and reads at most [`MAX_FEED_BYTES`] of the body however it arrives: the
//! size is counted as the bytes stream in, not taken from a header. What comes back is parsed and
//! checked by [`super::feed`] and trusted for nothing else.

use std::time::Duration;

use futures_util::StreamExt as _;

use super::feed::{self, Feed, FeedError, MAX_FEED_BYTES};
use super::FeedSource;

/// The whole request: connecting, the redirects, and reading the body.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// A feed is at most this many redirects from its address.
const MAX_REDIRECTS: usize = 5;

/// Why the feed could not be read, in words for a person.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError {
    /// The address is not https (and is not a debug build's override).
    InsecureAddress,
    /// No answer: offline, DNS, TLS or a timeout.
    Unreachable,
    /// An answer that was not the feed.
    Status(u16),
    /// A redirect to an address that is not https, or too many of them.
    Redirect,
    /// The body was not a usable feed.
    Feed(FeedError),
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckError::InsecureAddress => write!(f, "The update address is not an https address, so it was not used."),
            CheckError::Unreachable => write!(f, "Could not reach the update server. Check the internet connection and try again."),
            CheckError::Status(404) => write!(f, "No update information is published at the release address yet."),
            CheckError::Status(code) => write!(f, "The update server answered {code} instead of the update information."),
            CheckError::Redirect => write!(f, "The update server sent the request to an address that is not encrypted (or sent it round too many times), so it was refused."),
            CheckError::Feed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CheckError {}

/// Read the feed at `source`, once.
pub async fn fetch(source: &FeedSource) -> Result<Feed, CheckError> {
    let url = reqwest::Url::parse(&source.url).map_err(|_| CheckError::InsecureAddress)?;
    if url.scheme() != "https" && !source.insecure {
        return Err(CheckError::InsecureAddress);
    }
    let insecure = source.insecure;
    let redirects = reqwest::redirect::Policy::custom(move |attempt| {
        if redirect_allowed(attempt.url().scheme(), attempt.previous().len(), insecure) {
            attempt.follow()
        } else {
            attempt.error("a redirect that is not allowed")
        }
    });
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .user_agent(concat!("OAIY-Desktop/", env!("CARGO_PKG_VERSION")))
        .redirect(redirects)
        .build()
        .map_err(|_| CheckError::Unreachable)?;
    let response = client.get(url).header("Accept", "application/json").send().await.map_err(classify)?;
    let status = response.status();
    if !status.is_success() || status.as_u16() == 204 {
        return Err(CheckError::Status(status.as_u16()));
    }
    if response.content_length().is_some_and(|n| n > MAX_FEED_BYTES as u64) {
        return Err(CheckError::Feed(FeedError::TooLarge));
    }
    let mut body: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(classify)?;
        if body.len() + chunk.len() > MAX_FEED_BYTES {
            return Err(CheckError::Feed(FeedError::TooLarge));
        }
        body.extend_from_slice(&chunk);
    }
    feed::parse(&body).map_err(CheckError::Feed)
}

/// Whether a redirect to `scheme`, after `previous` redirects already, may be followed: only to https, and only so far.
/// (`insecure` is a debug build's own feed override, which may be a local http server.)
fn redirect_allowed(scheme: &str, previous: usize, insecure: bool) -> bool {
    previous < MAX_REDIRECTS && (scheme == "https" || insecure)
}

fn classify(error: reqwest::Error) -> CheckError {
    if error.is_redirect() {
        CheckError::Redirect
    } else {
        CheckError::Unreachable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_redirect_is_followed_only_to_https_and_only_so_far() {
        for previous in 0..MAX_REDIRECTS {
            assert!(redirect_allowed("https", previous, false), "hop {previous}");
        }
        assert!(!redirect_allowed("https", MAX_REDIRECTS, false));
        assert!(!redirect_allowed("https", 50, false));
        for scheme in ["http", "ftp", "file", "data", ""] {
            assert!(!redirect_allowed(scheme, 0, false), "{scheme}");
        }
        // A debug build reading a local stub may follow http; never past the limit.
        assert!(redirect_allowed("http", 0, true));
        assert!(!redirect_allowed("http", MAX_REDIRECTS, true));
    }

    #[test]
    fn an_address_that_is_not_https_is_refused_before_anything_is_sent() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        for url in ["http://github.com/f2i-com/oaiy.com/releases/latest/download/latest.json", "ftp://example.com/latest.json", "file:///C:/latest.json", "not a url"] {
            let source = FeedSource { url: url.into(), insecure: false };
            assert_eq!(rt.block_on(fetch(&source)).unwrap_err(), CheckError::InsecureAddress, "{url}");
        }
    }

    #[test]
    fn every_failure_is_in_words_a_person_can_read() {
        for error in [
            CheckError::InsecureAddress,
            CheckError::Unreachable,
            CheckError::Status(404),
            CheckError::Status(503),
            CheckError::Redirect,
            CheckError::Feed(FeedError::TooLarge),
            CheckError::Feed(FeedError::Malformed { what: "it is not valid JSON".into() }),
        ] {
            let text = error.to_string();
            assert!(text.ends_with('.') && text.chars().next().unwrap().is_uppercase(), "{text}");
            assert!(!text.contains("reqwest") && !text.contains("hyper") && !text.contains("os error"), "{text}");
        }
        assert!(CheckError::Status(404).to_string().contains("published at the release address"));
        assert!(CheckError::Status(503).to_string().contains("503"));
    }
}
