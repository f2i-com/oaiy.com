//! Updates: OAIY finding out that a newer release exists, and (in the desktop) installing it.
//!
//! The policy, in one paragraph (docs/UPDATES.md has it whole): OAIY looks for a newer release on
//! GitHub (at start, then daily, and when asked), tells the owner, downloads it when asked, checks
//! its signature against the key inside this build, and installs it ONLY when the owner presses
//! "Restart to update" and nothing is in the way (a live call, a running task, a download, a media
//! job, an install, or an app that only just started). The headless `oaiy-server` never installs
//! anything: it only reports that a newer release exists.
//!
//! This module is compiled without the `gui` feature, so both binaries have it:
//!
//! - [`version`]: which version is newer (a lower or equal one never is);
//! - [`feed`]: the release feed (`latest.json`, Tauri's format), read defensively, and what it
//!   means for this version on this platform;
//! - [`check`]: reading it over https, with a size cap and https-only redirects;
//! - [`verify`]: the signature check that turns downloaded bytes into a [`VerifiedPackage`], the
//!   only thing that can be installed (it also holds the signature to the announced version, by the
//!   name of the file it was made for);
//! - [`target`]: which kind of installer each platform takes;
//! - [`kind`]: whether this install can replace itself (the NSIS setup.exe and the AppImage can; an MSI, a
//!   .deb, an .rpm and a development build cannot);
//! - [`blockers`]: what stops an install, in words;
//! - [`updater`]: the state everything above feeds, and the moves between states;
//! - [`install`]: the safe order of an install, and the stop it shares with quitting;
//! - [`routes`]: `GET /api/update/status` and `POST /api/update/check`.
//!
//! The desktop adds `update::gui` (the updater plugin: the handle on the release, the download
//! with progress, the hand-off to the installer, and the commands) behind the `gui` feature.

pub mod blockers;
pub mod check;
pub mod feed;
pub mod install;
pub mod kind;
pub mod phone;
pub mod routes;
pub mod target;
pub mod updater;
pub mod verify;
pub mod version;

#[cfg(feature = "gui")]
pub mod gui;

#[cfg(test)]
mod guards;

pub use updater::{Updater, UpdaterHandle};
pub use verify::VerifiedPackage;

/// The GitHub repository every update comes from: the feed, and every installer the feed may name, are this repository's releases and no
/// other's. (`make-latest-json.mjs` refuses to write a feed for another one, and a test holds this to tauri.conf.json's endpoint.)
pub const REPO: &str = "f2i-com/oaiy.com";

/// Where the feed is: fixed in the build (tauri.conf.json's `plugins.updater.endpoints` says the same).
pub const FEED_URL: &str = "https://github.com/f2i-com/oaiy.com/releases/latest/download/latest.json";

/// The environment variable a DEBUG build reads a feed address from (a local stub, in a test or a trial). A release build ignores it.
pub const FEED_OVERRIDE_VAR: &str = "OAIY_UPDATE_FEED";

/// Where the feed is read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedSource {
    pub url: String,
    /// The address may be plain http (and so may a redirect): only ever true for a debug build's own override.
    pub insecure: bool,
}

impl FeedSource {
    /// The release feed, https only.
    pub fn production() -> FeedSource {
        FeedSource { url: FEED_URL.to_string(), insecure: false }
    }

    /// The feed for a build: the release feed, unless this is a debug build and `override_url` names another.
    /// An address from a request or a setting never comes through here: the override is the process's own environment.
    pub fn resolve(override_url: Option<&str>, debug_build: bool) -> FeedSource {
        match override_url.map(str::trim).filter(|u| !u.is_empty()) {
            Some(url) if debug_build => FeedSource { url: url.to_string(), insecure: true },
            _ => FeedSource::production(),
        }
    }

    /// [`resolve`](Self::resolve) with this build's kind and the process's environment.
    pub fn from_env() -> FeedSource {
        FeedSource::resolve(std::env::var(FEED_OVERRIDE_VAR).ok().as_deref(), cfg!(debug_assertions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_feed_is_the_release_feed_and_the_same_address_tauri_conf_json_names() {
        let conf: serde_json::Value = serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        assert_eq!(conf["plugins"]["updater"]["endpoints"], serde_json::json!([FEED_URL]));
        assert_eq!(FEED_URL, format!("https://github.com/{REPO}/releases/latest/download/latest.json"), "the feed is the latest release of REPO");
        assert_eq!(FeedSource::production(), FeedSource { url: FEED_URL.into(), insecure: false });
    }

    #[test]
    fn an_override_is_honoured_only_by_a_debug_build() {
        let local = "http://127.0.0.1:9999/latest.json";
        assert_eq!(FeedSource::resolve(Some(local), true), FeedSource { url: local.into(), insecure: true });
        // A release build ignores it, whatever the environment says.
        assert_eq!(FeedSource::resolve(Some(local), false), FeedSource::production());
        assert_eq!(FeedSource::resolve(Some("https://evil.example/latest.json"), false), FeedSource::production());
    }

    #[test]
    fn no_override_or_an_empty_one_is_the_release_feed_even_in_a_debug_build() {
        assert_eq!(FeedSource::resolve(None, true), FeedSource::production());
        assert_eq!(FeedSource::resolve(Some(""), true), FeedSource::production());
        assert_eq!(FeedSource::resolve(Some("   "), true), FeedSource::production());
    }

    #[test]
    fn nothing_but_the_process_environment_can_name_a_feed() {
        // The only way in is FeedSource::resolve's argument; no route, command or setting takes an address.
        // (from_env is the one caller: it reads the variable, and only in a debug build.)
        std::env::remove_var(FEED_OVERRIDE_VAR);
        assert_eq!(FeedSource::from_env(), FeedSource::production());
    }
}
