//! The relay link's files, beside the provider link's and never inside it (design 4.16.2).
//!
//! `link/account.json` keeps the shape it has always had and is written exactly as it was, with no new
//! field: a build from before these files reads it and works, and `LinkedAccount` still refuses a field
//! it does not know. Everything the relay adds lives in `<data>/relay/`, one file for each thing, which a
//! build that does not know the folder ignores:
//!
//! - `relay.json`: the relay's address and identity, this desktop's device id and token at it, and what
//!   the calibration measured ([`RelayLink`]);
//! - `routes.json`: which of the lanes that can move use the relay and which the provider ([`Routes`]);
//! - `providers.json`: the provider keys this desktop pinned, with the last rotation serial accepted
//!   for each ([`ProviderPins`]).
//!
//! The files the client keeps while it runs (the cursor, the command ledger, the outbox) are its own
//! and come with it.
//!
//! Every file is replaced whole by [`crate::secret_file::write`]: private from its first byte, and a
//! reader sees the old file or the new one, never half of either. And every read says what it found.
//! A file that is there and cannot be used (a newer shape, a cut file, one another program holds) is
//! KEPT as it is and reported ([`RelayStore::problems`]); it is never read as "no relay", which is how
//! `.ok()` used to unlink a desktop without a word, and the next write puts it aside as `<name>.corrupt`
//! before it writes a new one, or is refused.
//!
//! Nothing here reaches the origins the HTTP API trusts (the linked provider's, and the other pages of
//! its allow-lists): a relay URL is kept in a type of its own that no rule of that API reads (design
//! 4.16.6). A test reads every file of this folder for the names of those rules.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::link::{put_aside, read_stored, LinkError};

/// The folder of these files, under the data folder.
const DIR: &str = "relay";
const RELAY_FILE: &str = "relay.json";
const ROUTES_FILE: &str = "routes.json";
const PROVIDERS_FILE: &str = "providers.json";

/// What this desktop holds from enrolling at a relay (`relay.json`).
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelayLink {
    /// The relay's address, as it was enrolled.
    pub relay_url: String,
    pub relay_id: String,
    /// The thumbprint of the relay's key, from the enrolment key: what its `info` answers are checked against.
    pub relay_thumbprint: String,
    /// This desktop's device id at the relay.
    pub device_id: String,
    /// This desktop's bearer at the relay. Never in a status, a log line or `Debug`.
    pub token: String,
    /// The name this desktop gave itself at the relay.
    pub name: String,
    pub enrolled_at: DateTime<Utc>,
    /// What "Test this relay" measured, as the relay reported it. Kept as it came: the client that runs
    /// the calibration gives it a type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<serde_json::Value>,
}

impl std::fmt::Debug for RelayLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelayLink")
            .field("relay_url", &self.relay_url)
            .field("relay_id", &self.relay_id)
            .field("device_id", &self.device_id)
            .field("token", &"<hidden>")
            .finish_non_exhaustive()
    }
}

/// Where a lane that can move goes (design 4.16.4). `Provider` is what every lane does today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Route {
    #[default]
    Provider,
    Relay,
    Off,
}

/// Which base each of the lanes that can move uses (`routes.json`). A lane that is not named, or a file
/// that is not there, is on the provider: nothing moves until the owner moves it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Routes {
    pub commands: Route,
    pub chat: Route,
    pub companion: Route,
}

/// A provider's keys as this desktop pinned them (design 4.11).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderPin {
    pub provider_id: String,
    /// The provider's signing key and its sealing key, base64url.
    pub ed25519: String,
    pub x25519: String,
    pub thumbprint: String,
    /// The serial of the last rotation statement accepted: a statement with no higher one is a replay.
    #[serde(default)]
    pub serial: u64,
    /// A key change that the pinned key did not sign: the provider's commands are refused until the owner
    /// confirms the new keys.
    #[serde(default)]
    pub suspended: bool,
    pub pinned_at: DateTime<Utc>,
}

/// The pinned provider keys (`providers.json`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ProviderPins {
    pub pins: Vec<ProviderPin>,
}

#[derive(Default)]
struct Held {
    relay: Option<RelayLink>,
    routes: Routes,
    pins: ProviderPins,
    /// Files that were there and could not be used, by their name under the data folder.
    problems: Vec<LinkError>,
}

/// The three files, read at `open` and replaced whole by each setter.
pub struct RelayStore {
    dir: PathBuf,
    held: Mutex<Held>,
}

impl RelayStore {
    /// Read `<data_dir>/relay/`. A file that is not there is the default; one that cannot be used is
    /// reported by [`RelayStore::problems`] and left as it is.
    pub fn open(data_dir: &Path) -> Self {
        let dir = data_dir.join(DIR);
        let mut held = Held::default();
        held.relay = load(&dir, RELAY_FILE, &mut held.problems);
        held.routes = load(&dir, ROUTES_FILE, &mut held.problems).unwrap_or_default();
        held.pins = load(&dir, PROVIDERS_FILE, &mut held.problems).unwrap_or_default();
        Self { dir, held: Mutex::new(held) }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The relay this desktop is enrolled at, if any. `None` is also what a file that could not be used
    /// reads as here: ask [`RelayStore::problems`] before telling anyone there is no relay.
    pub fn relay(&self) -> Option<RelayLink> {
        self.held().relay.clone()
    }

    pub fn routes(&self) -> Routes {
        self.held().routes.clone()
    }

    pub fn pins(&self) -> ProviderPins {
        self.held().pins.clone()
    }

    /// The files that could not be used, and why. Nothing in them is quoted.
    pub fn problems(&self) -> Vec<LinkError> {
        self.held().problems.clone()
    }

    pub fn set_relay(&self, link: RelayLink) -> Result<(), String> {
        let mut held = self.held();
        self.replace(&mut held.problems, RELAY_FILE, &link)?;
        held.relay = Some(link);
        Ok(())
    }

    pub fn set_routes(&self, routes: Routes) -> Result<(), String> {
        let mut held = self.held();
        self.replace(&mut held.problems, ROUTES_FILE, &routes)?;
        held.routes = routes;
        Ok(())
    }

    pub fn set_pins(&self, pins: ProviderPins) -> Result<(), String> {
        let mut held = self.held();
        self.replace(&mut held.problems, PROVIDERS_FILE, &pins)?;
        held.pins = pins;
        Ok(())
    }

    /// Forget the relay link: the file is removed first, and one that cannot be removed is an error and
    /// stays in force. A file that could not be used is put aside, not thrown away.
    pub fn forget_relay(&self) -> Result<(), String> {
        let mut held = self.held();
        let (path, shown) = (self.dir.join(RELAY_FILE), format!("{DIR}/{RELAY_FILE}"));
        if held.problems.iter().any(|p| p.file == shown) {
            put_aside(&path, &shown)?;
            held.problems.retain(|p| p.file != shown);
        } else {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("the relay link could not be forgotten: {shown} could not be removed ({e})")),
            }
        }
        held.relay = None;
        Ok(())
    }

    /// Write `value` as the file `name`, whole. A file that could not be used is put aside first, and when
    /// that fails nothing is written.
    fn replace<T: Serialize>(&self, problems: &mut Vec<LinkError>, name: &str, value: &T) -> Result<(), String> {
        let (path, shown) = (self.dir.join(name), format!("{DIR}/{name}"));
        let raw = serde_json::to_string_pretty(value).map_err(|e| format!("could not encode {shown}: {e}"))?;
        if problems.iter().any(|p| p.file == shown) {
            put_aside(&path, &shown)?;
            problems.retain(|p| p.file != shown);
        }
        crate::secret_file::write(&path, raw).map_err(|e| format!("could not save {shown}: {e}"))
    }
}

/// One of the files, or `None` for one that is not there or could not be used (which is added to `problems`).
fn load<T: serde::de::DeserializeOwned>(dir: &Path, name: &str, problems: &mut Vec<LinkError>) -> Option<T> {
    match read_stored(&dir.join(name), &format!("{DIR}/{name}")) {
        Ok(value) => value,
        Err(problem) => {
            problems.push(problem);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::{assert_private, assert_private_dir, TempDir};

    fn relay() -> RelayLink {
        RelayLink {
            relay_url: "https://relay.example.com".into(),
            relay_id: "rly-1".into(),
            relay_thumbprint: "t".repeat(43),
            device_id: "dev-1".into(),
            token: "oaiyrt1.TOPSECRETTOKEN".into(),
            name: "Reception PC".into(),
            enrolled_at: Utc::now(),
            calibration: Some(serde_json::json!({ "holdOk": true, "waitMax": 20 })),
        }
    }

    fn pins() -> ProviderPins {
        ProviderPins {
            pins: vec![ProviderPin {
                provider_id: "formlogic".into(),
                ed25519: "e".repeat(43),
                x25519: "x".repeat(43),
                thumbprint: "p".repeat(43),
                serial: 3,
                suspended: false,
                pinned_at: Utc::now(),
            }],
        }
    }

    #[test]
    fn the_three_files_round_trip_through_a_restart_and_are_private() {
        let dir = TempDir::new("relay-store");
        let store = RelayStore::open(&dir.0);
        assert_eq!((store.relay(), store.routes(), store.pins()), (None, Routes::default(), ProviderPins::default()));
        assert!(store.problems().is_empty());

        let (link, routes, held) = (relay(), Routes { commands: Route::Relay, chat: Route::Provider, companion: Route::Off }, pins());
        store.set_relay(link.clone()).unwrap();
        store.set_routes(routes.clone()).unwrap();
        store.set_pins(held.clone()).unwrap();

        let again = RelayStore::open(&dir.0);
        assert_eq!(again.relay(), Some(link));
        assert_eq!(again.routes(), routes);
        assert_eq!(again.pins(), held);
        assert!(again.problems().is_empty());

        // Beside the provider's folder, never in it, and each file private from its first byte.
        let folder = dir.0.join("relay");
        for name in ["relay.json", "routes.json", "providers.json"] {
            assert_private(&folder.join(name));
        }
        assert_private_dir(&folder);
        assert!(!dir.0.join("link").exists(), "nothing of the relay's is written under link/");
        let leftovers: Vec<_> = std::fs::read_dir(&folder).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".tmp")).collect();
        assert!(leftovers.is_empty(), "no staging copy of the token is left behind: {leftovers:?}");

        // Forgetting the relay takes its file and nothing else.
        again.forget_relay().unwrap();
        assert!(!folder.join("relay.json").exists() && folder.join("routes.json").exists());
        assert_eq!(RelayStore::open(&dir.0).relay(), None, "and it is durable");
        again.forget_relay().unwrap();
    }

    #[test]
    fn a_lane_that_is_not_named_is_on_the_provider() {
        let dir = TempDir::new("relay-routes");
        std::fs::create_dir_all(dir.0.join("relay")).unwrap();
        std::fs::write(dir.0.join("relay").join("routes.json"), r#"{"commands":"relay"}"#).unwrap();
        let store = RelayStore::open(&dir.0);
        assert_eq!(store.routes(), Routes { commands: Route::Relay, chat: Route::Provider, companion: Route::Provider });
        assert_eq!(Routes::default(), Routes { commands: Route::Provider, chat: Route::Provider, companion: Route::Provider });
    }

    #[test]
    fn a_file_that_cannot_be_used_is_kept_reported_and_put_aside_before_a_new_one_is_written() {
        let cases: [(&str, fn(&RelayStore) -> Result<(), String>); 3] = [
            ("relay.json", |s| s.set_relay(relay())),
            ("routes.json", |s| s.set_routes(Routes::default())),
            ("providers.json", |s| s.set_pins(pins())),
        ];
        for (name, set) in cases {
            let dir = TempDir::new("relay-unusable");
            let folder = dir.0.join("relay");
            std::fs::create_dir_all(&folder).unwrap();
            // JSON, and not what any of the three holds.
            let kept = b"[1, 2]";
            std::fs::write(folder.join(name), kept).unwrap();

            let store = RelayStore::open(&dir.0);
            let problems = store.problems();
            assert_eq!(problems.len(), 1, "{name}: {problems:?}");
            assert_eq!(problems[0].file, format!("relay/{name}"));
            assert!(problems[0].message.contains("It has not been changed"), "{name}: {}", problems[0].message);
            assert_eq!(std::fs::read(folder.join(name)).unwrap(), kept, "{name}: reading it left it as it was");

            set(&store).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(std::fs::read(folder.join(format!("{name}.corrupt"))).unwrap(), kept, "{name}: what was there is kept");
            assert!(store.problems().is_empty(), "{name}: dealt with");
            assert!(RelayStore::open(&dir.0).problems().is_empty(), "{name}: and the file now is one this build wrote");
        }
    }

    #[test]
    fn forgetting_a_relay_file_that_could_not_be_used_keeps_it_aside() {
        let dir = TempDir::new("relay-forget-bad");
        let folder = dir.0.join("relay");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("relay.json"), b"{ not json").unwrap();
        let store = RelayStore::open(&dir.0);
        assert_eq!(store.problems().len(), 1);
        store.forget_relay().unwrap();
        assert_eq!(std::fs::read(folder.join("relay.json.corrupt")).unwrap(), b"{ not json");
        assert!(store.problems().is_empty() && !folder.join("relay.json").exists());
    }

    #[test]
    fn a_relay_that_cannot_be_forgotten_says_so_and_stays() {
        let dir = TempDir::new("relay-stuck");
        let store = RelayStore::open(&dir.0);
        store.set_relay(relay()).unwrap();
        // A folder with something in it where the file was cannot be removed as a file, on any system.
        let file = dir.0.join("relay").join("relay.json");
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir_all(file.join("held")).unwrap();
        let error = store.forget_relay().unwrap_err();
        assert!(error.contains("could not be forgotten"), "{error}");
        assert!(store.relay().is_some(), "the link is still in force");
    }

    #[test]
    fn the_token_is_in_the_file_and_nowhere_else() {
        let link = relay();
        assert!(!format!("{link:?}").contains("TOPSECRETTOKEN"), "{link:?}");
        let dir = TempDir::new("relay-token");
        let store = RelayStore::open(&dir.0);
        store.set_relay(link).unwrap();
        let on_disk = std::fs::read_to_string(dir.0.join("relay").join("relay.json")).unwrap();
        assert!(on_disk.contains("TOPSECRETTOKEN"));
        // What is reported of a file that could not be used quotes none of it.
        std::fs::write(dir.0.join("relay").join("relay.json"), r#"{"relayUrl":"https://r","token":"oaiyrt1.TOPSECRETTOKEN","enrolledAt":"oaiyrt1.TOPSECRETTOKEN"}"#).unwrap();
        let problems = RelayStore::open(&dir.0).problems();
        assert!(!format!("{problems:?}").contains("TOPSECRETTOKEN"), "{problems:?}");
    }

    #[test]
    fn nothing_in_this_folder_reaches_the_origins_the_http_api_trusts() {
        // Design 4.16.6: a relay URL is kept in a type with no path into `is_allowed_origin` or
        // `is_allowed_origin_privileged`. The simplest way to keep that true is that nothing in this folder
        // so much as names what they read.
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("relay");
        let mut read = 0;
        for entry in std::fs::read_dir(&folder).unwrap() {
            let path = entry.unwrap().path();
            // This test names them; everything else must not.
            let code = crate::source_scan::production_code(&std::fs::read_to_string(&path).unwrap());
            for name in ["linked_origin", "set_linked_origin", "is_allowed_origin", "LINKED_ORIGIN"] {
                assert!(!code.contains(name), "{} names {name}", path.display());
            }
            read += 1;
        }
        assert!(read >= 2, "the folder was not read");
    }
}
