//! The update feed: `latest.json`, in the format the Tauri updater reads.
//!
//! ```json
//! { "version": "0.2.0", "notes": "...", "pub_date": "2026-10-01T02:03:04Z",
//!   "platforms": { "windows-x86_64": { "signature": "...", "url": "https://..." },
//!                  "linux-x86_64":   { "signature": "...", "url": "https://..." } } }
//! ```
//!
//! It is written by `platform/scripts/make-latest-json.mjs` in the release job and served from
//! the release (`releases/latest/download/latest.json`). What is read here is UNTRUSTED text:
//! it is capped in size before it is parsed, every field is checked, and nothing in it is acted
//! on until it has been through [`evaluate`]. (The installer it points at is trusted only after
//! its signature verifies: see [`super::verify`].)

use std::collections::BTreeMap;

use chrono::{DateTime, FixedOffset};
use semver::Version;
use serde::Deserialize;

use super::target::Target;

/// The most a feed may be. The real one is a couple of kilobytes; more is not ours.
pub const MAX_FEED_BYTES: usize = 256 * 1024;

/// Why a feed was not used, in words for a person (they are shown as they are).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedError {
    TooLarge,
    /// Not JSON, or JSON of the wrong shape: `what` says which field.
    Malformed { what: String },
    BadVersion(String),
    BadDate(String),
    BadUrl { platform: String },
    NoPlatforms,
}

impl std::fmt::Display for FeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedError::TooLarge => write!(f, "The update information is larger than expected ({} KiB at most), so it was not used.", MAX_FEED_BYTES / 1024),
            FeedError::Malformed { what } => write!(f, "The update information could not be understood ({what}), so it was not used."),
            FeedError::BadVersion(v) => write!(f, "The update information names a version that is not one (\"{v}\"), so it was not used."),
            FeedError::BadDate(d) => write!(f, "The update information has a date that is not one (\"{d}\"), so it was not used."),
            FeedError::BadUrl { platform } => write!(f, "The update information has no usable address for {platform}, so it was not used."),
            FeedError::NoPlatforms => write!(f, "The update information lists no platform, so it was not used."),
        }
    }
}

impl std::error::Error for FeedError {}

/// What a feed says about one platform: where its installer is and the signature it must verify against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformEntry {
    pub url: String,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feed {
    pub version: Version,
    pub notes: String,
    /// RFC 3339, as the feed wrote it (re-written in UTC).
    pub pub_date: Option<String>,
    pub platforms: BTreeMap<String, PlatformEntry>,
}

#[derive(Deserialize)]
struct RawFeed {
    version: String,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    pub_date: Option<String>,
    platforms: BTreeMap<String, RawPlatform>,
}

#[derive(Deserialize)]
struct RawPlatform {
    url: String,
    signature: String,
}

/// Read a feed. `bytes` longer than [`MAX_FEED_BYTES`] are refused before anything is parsed.
pub fn parse(bytes: &[u8]) -> Result<Feed, FeedError> {
    if bytes.len() > MAX_FEED_BYTES {
        return Err(FeedError::TooLarge);
    }
    let raw: RawFeed = serde_json::from_slice(bytes).map_err(|e| FeedError::Malformed { what: describe(&e) })?;
    let version = super::version::parse(&raw.version).map_err(|_| FeedError::BadVersion(raw.version.clone()))?;
    let pub_date = match raw.pub_date.as_deref().map(str::trim).filter(|d| !d.is_empty()) {
        Some(text) => Some(DateTime::<FixedOffset>::parse_from_rfc3339(text).map_err(|_| FeedError::BadDate(text.to_string()))?.with_timezone(&chrono::Utc).to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        None => None,
    };
    if raw.platforms.is_empty() {
        return Err(FeedError::NoPlatforms);
    }
    let mut platforms = BTreeMap::new();
    for (name, entry) in raw.platforms {
        let url = entry.url.trim().to_string();
        if url.is_empty() || url::Url::parse(&url).is_err() || entry.signature.trim().is_empty() {
            return Err(FeedError::BadUrl { platform: name });
        }
        platforms.insert(name, PlatformEntry { url, signature: entry.signature.trim().to_string() });
    }
    Ok(Feed { version, notes: raw.notes.unwrap_or_default().trim().to_string(), pub_date, platforms })
}

/// A serde error in plain words: the field or the kind of value at fault, not the parser's position.
fn describe(error: &serde_json::Error) -> String {
    use serde_json::error::Category;
    match error.classify() {
        Category::Syntax | Category::Eof => "it is not valid JSON".to_string(),
        Category::Data => {
            let text = error.to_string();
            // `missing field `version`` / `invalid type: integer `5`, expected a string`
            match text.split(" at line ").next() {
                Some(head) if head.starts_with("missing field") => format!("it has no {}", head.trim_start_matches("missing field ").replace('`', "\"")),
                _ => "a field has the wrong kind of value".to_string(),
            }
        }
        Category::Io => "it could not be read".to_string(),
    }
}

/// The feed's key for a platform (`windows-x86_64`, `linux-x86_64`), or None: no release is built for it.
pub fn platform_key(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("windows", "x86_64") => Some("windows-x86_64"),
        ("linux", "x86_64") => Some("linux-x86_64"),
        _ => None,
    }
}

/// The platform this build runs on.
pub fn current_platform_key() -> Option<&'static str> {
    platform_key(std::env::consts::OS, std::env::consts::ARCH)
}

/// Where an installer may be downloaded from: this project's release assets on github.com.
const ASSET_HOST: &str = "github.com";
const ASSET_PATH_PREFIX: &str = "/f2i-com/oaiy.com/releases/download/";

/// Whether `url` is the installer of release `version` of this project on GitHub for `target`: over https, on github.com,
/// with nothing to redirect the eye (no credentials, no other port, no query), under the release's tag (`v0.2.0` or
/// `0.2.0`) and named EXACTLY as the release job names this platform's installer ([`Target::asset_name`]:
/// `oaiy-desktop-0.2.0-windows-x64-setup.exe` for the Windows setup, `oaiy-desktop-0.2.0-linux-x86_64.AppImage` for the AppImage).
///
/// Doors beside the signature. A feed that names another host is refused before anything is fetched from it. The file has
/// to be this platform's installer and no other file of the release (the Windows setup where the AppImage belongs would
/// be written over the AppImage by the Linux updater, which does not look at what it is given, and leave a program that
/// will not start), and it has to be of the announced version by its name. (What the address cannot prove is that the
/// bytes behind it ARE that release: a release can hold any file. That is the signature's part: see `verify`.)
pub fn check_asset_url(url: &str, version: &str, target: Target) -> Result<(), String> {
    let refuse = || Err(format!("The update points at an address that is not {} of OAIY {version} on GitHub ({url}), so it was refused.", target.what()));
    let Ok(parsed) = url::Url::parse(url) else { return refuse() };
    let plain = parsed.scheme() == "https" && parsed.host_str() == Some(ASSET_HOST) && parsed.port().is_none() && parsed.username().is_empty() && parsed.password().is_none() && parsed.query().is_none() && parsed.fragment().is_none();
    let Some(rest) = parsed.path().strip_prefix(ASSET_PATH_PREFIX) else { return refuse() };
    let named = match rest.split_once('/') {
        Some((tag, file)) => (tag == version || tag.strip_prefix('v') == Some(version)) && file == target.asset_name(version),
        None => false,
    };
    if plain && named && super::version::parse(version).is_ok() {
        Ok(())
    } else {
        refuse()
    }
}

/// A newer release, for this platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub version: String,
    pub notes: String,
    pub published_at: Option<String>,
    pub url: String,
    pub signature: String,
}

/// What a feed means for the running version on a platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing higher than the running version: the feed's version is lower or equal.
    UpToDate { latest: String },
    /// A higher version exists, and the feed has an installer for this platform.
    Available(Release),
    /// A higher version exists but there is no release for this platform (or this platform is not one that is built): not an error.
    NoUpdateForPlatform { latest: String },
}

/// What the feed means for `current`, on the platform whose key is `platform` (None: an unknown platform).
pub fn evaluate(feed: &Feed, current: &str, platform: Option<&str>) -> Verdict {
    let latest = feed.version.to_string();
    if !super::version::is_newer(current, &latest) {
        return Verdict::UpToDate { latest };
    }
    match platform.and_then(|key| feed.platforms.get(key)) {
        Some(entry) => Verdict::Available(Release { version: latest, notes: feed.notes.clone(), published_at: feed.pub_date.clone(), url: entry.url.clone(), signature: entry.signature.clone() }),
        None => Verdict::NoUpdateForPlatform { latest },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SIG: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZQpSVVI=";

    fn feed_json() -> serde_json::Value {
        json!({
            "version": "0.2.0",
            "notes": "Calls keep their audio.",
            "pub_date": "2026-10-01T02:03:04Z",
            "platforms": {
                "windows-x86_64": { "signature": SIG, "url": "https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/oaiy-desktop-0.2.0-windows-x64-setup.exe" },
                "linux-x86_64": { "signature": SIG, "url": "https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/oaiy-desktop-0.2.0-linux-x86_64.AppImage" }
            }
        })
    }

    fn parse_json(value: &serde_json::Value) -> Result<Feed, FeedError> {
        parse(value.to_string().as_bytes())
    }

    #[test]
    fn a_valid_feed_is_read_field_by_field() {
        let feed = parse_json(&feed_json()).unwrap();
        assert_eq!(feed.version, Version::new(0, 2, 0));
        assert_eq!(feed.notes, "Calls keep their audio.");
        assert_eq!(feed.pub_date.as_deref(), Some("2026-10-01T02:03:04Z"));
        assert_eq!(feed.platforms.len(), 2);
        assert!(feed.platforms["windows-x86_64"].url.ends_with("-setup.exe"));
        assert_eq!(feed.platforms["linux-x86_64"].signature, SIG);
    }

    #[test]
    fn notes_and_the_date_may_be_left_out_and_a_date_in_another_zone_is_kept_as_the_same_moment() {
        let mut value = feed_json();
        value.as_object_mut().unwrap().remove("notes");
        value.as_object_mut().unwrap().remove("pub_date");
        let feed = parse_json(&value).unwrap();
        assert_eq!((feed.notes.as_str(), feed.pub_date), ("", None));
        value["pub_date"] = json!("2026-10-01T12:00:00+10:00");
        assert_eq!(parse_json(&value).unwrap().pub_date.as_deref(), Some("2026-10-01T02:00:00Z"));
    }

    #[test]
    fn a_feed_with_a_missing_field_is_refused_and_says_which() {
        for field in ["version", "platforms"] {
            let mut value = feed_json();
            value.as_object_mut().unwrap().remove(field);
            let error = parse_json(&value).unwrap_err();
            assert!(matches!(&error, FeedError::Malformed { what } if what.contains(field)), "{field}: {error}");
        }
        for field in ["url", "signature"] {
            let mut value = feed_json();
            value["platforms"]["windows-x86_64"].as_object_mut().unwrap().remove(field);
            assert!(matches!(parse_json(&value), Err(FeedError::Malformed { what }) if what.contains(field)), "{field}");
        }
    }

    #[test]
    fn a_feed_with_a_field_of_the_wrong_type_is_refused() {
        for (field, wrong) in [("version", json!(5)), ("version", json!(null)), ("platforms", json!([])), ("platforms", json!("windows")), ("notes", json!(7)), ("pub_date", json!(20261001))] {
            let mut value = feed_json();
            value[field] = wrong.clone();
            assert!(matches!(parse_json(&value), Err(FeedError::Malformed { .. })), "{field} = {wrong}");
        }
        let mut value = feed_json();
        value["platforms"]["linux-x86_64"]["url"] = json!(12);
        assert!(matches!(parse_json(&value), Err(FeedError::Malformed { .. })));
        assert!(matches!(parse(b"[]"), Err(FeedError::Malformed { .. })));
        assert!(matches!(parse(b"\"latest\""), Err(FeedError::Malformed { .. })));
    }

    #[test]
    fn something_that_is_not_json_is_refused() {
        for text in [&b""[..], b"<html>404</html>", b"{\"version\": ", b"not json"] {
            assert!(matches!(parse(text), Err(FeedError::Malformed { what }) if what.contains("JSON")));
        }
    }

    #[test]
    fn a_version_that_is_not_one_and_a_date_that_is_not_one_are_refused() {
        for bad in ["latest", "", "1.2", "0.01.0"] {
            let mut value = feed_json();
            value["version"] = json!(bad);
            assert!(matches!(parse_json(&value), Err(FeedError::BadVersion(v)) if v == bad), "{bad}");
        }
        let mut value = feed_json();
        value["pub_date"] = json!("last Tuesday");
        assert!(matches!(parse_json(&value), Err(FeedError::BadDate(_))));
    }

    #[test]
    fn a_platform_with_no_usable_address_or_signature_is_refused_and_no_platforms_at_all_too() {
        for (field, bad) in [("url", ""), ("url", "not a url"), ("signature", ""), ("signature", "   ")] {
            let mut value = feed_json();
            value["platforms"]["windows-x86_64"][field] = json!(bad);
            assert!(matches!(parse_json(&value), Err(FeedError::BadUrl { platform }) if platform == "windows-x86_64"), "{field} {bad:?}");
        }
        let mut value = feed_json();
        value["platforms"] = json!({});
        assert_eq!(parse_json(&value), Err(FeedError::NoPlatforms));
    }

    #[test]
    fn an_oversized_feed_is_refused_before_it_is_parsed() {
        let mut value = feed_json();
        value["notes"] = json!("x".repeat(MAX_FEED_BYTES));
        let bytes = value.to_string().into_bytes();
        assert!(bytes.len() > MAX_FEED_BYTES);
        assert_eq!(parse(&bytes), Err(FeedError::TooLarge));
        // Not even valid JSON: it is the size that is refused.
        assert_eq!(parse(&vec![b'a'; MAX_FEED_BYTES + 1]), Err(FeedError::TooLarge));
        // At the limit it is read (and here, a feed padded to exactly the limit).
        let mut ok = feed_json();
        ok["notes"] = json!("");
        let pad = MAX_FEED_BYTES - ok.to_string().len();
        ok["notes"] = json!("x".repeat(pad));
        assert_eq!(ok.to_string().len(), MAX_FEED_BYTES);
        assert!(parse(ok.to_string().as_bytes()).is_ok());
    }

    #[test]
    fn a_platform_the_feed_does_not_list_is_not_an_error_when_the_feed_is_read() {
        let mut value = feed_json();
        value["platforms"] = json!({ "darwin-aarch64": { "signature": SIG, "url": "https://example.com/a.tar.gz" } });
        let feed = parse_json(&value).unwrap();
        assert_eq!(evaluate(&feed, "0.1.0", Some("windows-x86_64")), Verdict::NoUpdateForPlatform { latest: "0.2.0".into() });
    }

    #[test]
    fn platform_keys_are_the_feeds_and_an_unknown_platform_has_none() {
        assert_eq!(platform_key("windows", "x86_64"), Some("windows-x86_64"));
        assert_eq!(platform_key("linux", "x86_64"), Some("linux-x86_64"));
        for (os, arch) in [("macos", "aarch64"), ("macos", "x86_64"), ("windows", "aarch64"), ("linux", "aarch64"), ("freebsd", "x86_64"), ("", "")] {
            assert_eq!(platform_key(os, arch), None, "{os}-{arch}");
        }
    }

    #[test]
    fn a_higher_version_is_available_with_this_platforms_installer_and_signature() {
        let feed = parse_json(&feed_json()).unwrap();
        match evaluate(&feed, "0.1.0", Some("windows-x86_64")) {
            Verdict::Available(release) => {
                assert_eq!(release.version, "0.2.0");
                assert!(release.url.ends_with("windows-x64-setup.exe"));
                assert_eq!(release.signature, SIG);
                assert_eq!(release.notes, "Calls keep their audio.");
                assert_eq!(release.published_at.as_deref(), Some("2026-10-01T02:03:04Z"));
            }
            other => panic!("{other:?}"),
        }
        match evaluate(&feed, "0.1.0", Some("linux-x86_64")) {
            Verdict::Available(release) => assert!(release.url.ends_with("AppImage")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_lower_or_equal_version_is_up_to_date_whatever_the_platform() {
        let feed = parse_json(&feed_json()).unwrap();
        for platform in [Some("windows-x86_64"), Some("linux-x86_64"), None] {
            assert_eq!(evaluate(&feed, "0.2.0", platform), Verdict::UpToDate { latest: "0.2.0".into() });
            assert_eq!(evaluate(&feed, "0.3.0", platform), Verdict::UpToDate { latest: "0.2.0".into() });
            assert_eq!(evaluate(&feed, "0.2.0+build9", platform), Verdict::UpToDate { latest: "0.2.0".into() });
        }
        // A feed naming a pre-release of the running release offers nothing either.
        let mut value = feed_json();
        value["version"] = json!("0.2.0-rc.1");
        assert!(matches!(evaluate(&parse_json(&value).unwrap(), "0.2.0", Some("linux-x86_64")), Verdict::UpToDate { .. }));
    }

    #[test]
    fn a_higher_version_with_no_installer_for_an_unknown_platform_is_no_update_for_it() {
        let feed = parse_json(&feed_json()).unwrap();
        assert_eq!(evaluate(&feed, "0.1.0", None), Verdict::NoUpdateForPlatform { latest: "0.2.0".into() });
        assert_eq!(evaluate(&feed, "0.1.0", Some("darwin-aarch64")), Verdict::NoUpdateForPlatform { latest: "0.2.0".into() });
    }

    const WINDOWS: Target = Target::WindowsSetup;
    const LINUX: Target = Target::LinuxAppImage;

    #[test]
    fn an_installer_may_only_come_from_this_projects_releases_on_github_over_https() {
        let ok = "https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/oaiy-desktop-0.2.0-windows-x64-setup.exe";
        assert!(check_asset_url(ok, "0.2.0", WINDOWS).is_ok());
        assert!(check_asset_url("https://GitHub.com/f2i-com/oaiy.com/releases/download/0.2.0/oaiy-desktop-0.2.0-linux-x86_64.AppImage", "0.2.0", LINUX).is_ok());
        let file = "oaiy-desktop-0.2.0-windows-x64-setup.exe";
        for bad in [
            format!("http://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/{file}"),
            format!("https://github.com.evil.example/f2i-com/oaiy.com/releases/download/v0.2.0/{file}"),
            format!("https://evil.example/f2i-com/oaiy.com/releases/download/v0.2.0/{file}"),
            format!("https://github.com@evil.example/f2i-com/oaiy.com/releases/download/v0.2.0/{file}"),
            format!("https://user:pw@github.com/f2i-com/oaiy.com/releases/download/v0.2.0/{file}"),
            format!("https://github.com:8443/f2i-com/oaiy.com/releases/download/v0.2.0/{file}"),
            format!("https://github.com/someone-else/oaiy.com/releases/download/v0.2.0/{file}"),
            "https://github.com/f2i-com/oaiy.com/archive/main.zip".to_string(),
            "https://github.com/f2i-com/oaiy.com/releases/download/".to_string(),
            "https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/".to_string(),
            "https://github.com/f2i-com/oaiy.com/releases/download/../../../evil/x.exe".to_string(),
            format!("https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/{file}?token=1"),
            format!("https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/{file}#frag"),
            format!("https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/sub/{file}"),
            "file:///C:/x.exe".to_string(),
            "not a url".to_string(),
            String::new(),
        ] {
            assert!(check_asset_url(&bad, "0.2.0", WINDOWS).is_err(), "{bad}");
        }
        let refusal = check_asset_url("http://evil.example/x.exe", "0.2.0", WINDOWS).unwrap_err();
        assert!(refusal.contains("http://evil.example/x.exe") && refusal.contains("Windows installer"), "{refusal}");
    }

    #[test]
    fn each_platform_takes_its_own_installer_by_its_exact_name_and_no_other_file_of_the_release() {
        let base = "https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/";
        let (setup, appimage) = ("oaiy-desktop-0.2.0-windows-x64-setup.exe", "oaiy-desktop-0.2.0-linux-x86_64.AppImage");
        assert!(check_asset_url(&format!("{base}{setup}"), "0.2.0", WINDOWS).is_ok());
        assert!(check_asset_url(&format!("{base}{appimage}"), "0.2.0", LINUX).is_ok());
        // A Linux entry pointing at the Windows setup, and a Windows entry pointing at the AppImage: each is a genuine asset of the release.
        let mixed = check_asset_url(&format!("{base}{setup}"), "0.2.0", LINUX).unwrap_err();
        assert!(mixed.contains("Linux AppImage") && mixed.contains("refused"), "{mixed}");
        assert!(check_asset_url(&format!("{base}{appimage}"), "0.2.0", WINDOWS).is_err());
        // Nor any other file the release holds, under a name that starts or ends like an installer's.
        for other in [
            "oaiy-desktop-0.2.0-windows-x64.msi",
            "oaiy-desktop-0.2.0-linux-amd64.deb",
            "oaiy-desktop-0.2.0-linux-x86_64.rpm",
            "oaiy-desktop-0.2.0-windows-x64-setup.exe.sig",
            "oaiy-desktop-0.2.0-linux-x86_64.AppImage.sig",
            "oaiy-desktop-0.2.0-windows-x64-setup.exe.zip",
            "oaiy-desktop-0.2.0-windows-x64-setup.exe.evil.exe",
            "oaiy-desktop-0.2.0-rc.1-windows-x64-setup.exe",
            "oaiy-desktop-0.2.0-windows-x86-setup.exe",
            "oaiy-desktop-0.2.0-linux-aarch64.AppImage",
            "oaiy-desktop-0.2.0-.AppImage",
            "oaiy-server-0.2.0-linux-x86_64.tar.gz",
            "OAIY_0.2.0_x64-setup.exe",
            "latest.json",
            "SHA256SUMS.txt",
        ] {
            for target in [WINDOWS, LINUX] {
                assert!(check_asset_url(&format!("{base}{other}"), "0.2.0", target).is_err(), "{other} as {target:?}");
            }
        }
    }

    #[test]
    fn the_announced_version_has_to_be_the_one_in_the_installers_address() {
        // A feed pairing a higher number with the address of an older, genuinely signed release: a downgrade dressed as an update.
        let older = "https://github.com/f2i-com/oaiy.com/releases/download/v0.1.5/oaiy-desktop-0.1.5-windows-x64-setup.exe";
        assert!(check_asset_url(older, "0.1.5", WINDOWS).is_ok());
        for announced in ["9.9.9", "0.2.0", "0.1.50", "0.1"] {
            assert!(check_asset_url(older, announced, WINDOWS).is_err(), "announced {announced}");
        }
        // The tag and the file each have to say it.
        assert!(check_asset_url("https://github.com/f2i-com/oaiy.com/releases/download/v0.1.5/oaiy-desktop-9.9.9-windows-x64-setup.exe", "9.9.9", WINDOWS).is_err());
        assert!(check_asset_url("https://github.com/f2i-com/oaiy.com/releases/download/v9.9.9/oaiy-desktop-0.1.5-windows-x64-setup.exe", "9.9.9", WINDOWS).is_err());
        // The version itself has to be one.
        assert!(check_asset_url("https://github.com/f2i-com/oaiy.com/releases/download/vlatest/oaiy-desktop-latest-windows-x64-setup.exe", "latest", WINDOWS).is_err());
    }

    #[test]
    fn a_garbage_running_version_is_never_offered_an_update() {
        let feed = parse_json(&feed_json()).unwrap();
        assert!(matches!(evaluate(&feed, "not a version", Some("windows-x86_64")), Verdict::UpToDate { .. }));
    }
}
