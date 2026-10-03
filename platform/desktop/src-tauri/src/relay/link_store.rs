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

use crate::link::{purge_asides, put_aside, put_aside_to_forget, read_stored, reread, LinkError, Reread};
use crate::secret_file::{Patience, Retry};

/// The folder of these files, under the data folder.
const DIR: &str = "relay";
const RELAY_FILE: &str = "relay.json";
const ROUTES_FILE: &str = "routes.json";
const PROVIDERS_FILE: &str = "providers.json";

/// What this desktop holds from enrolling at a relay (`relay.json`).
///
/// It is not `Serialize`: it holds the token, so that a status or a log line cannot be made of it by a derive. The
/// store writes a private copy of it, [`StoredRelayLink`], and no other code writes a token in plain.
#[derive(Clone, PartialEq, Eq)]
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
    pub calibration: Option<serde_json::Value>,
    /// Members of the file that this build does not know (a newer build wrote them), kept as they were so that
    /// rewriting the file here does not drop them. A member is kept as `serde_json` reads it: a number that is an
    /// integer of 64 bits or fewer, or a decimal that a `f64` holds, comes back as the same number; one beyond that (an
    /// integer over 64 bits, a decimal of more than 17 significant digits) comes back as the nearest `f64`, and the members
    /// of an object are written in sorted order. A name this build writes itself (see [`RelayStore::set_relay`]) is
    /// not one it does not know, and is refused here.
    pub other: serde_json::Map<String, serde_json::Value>,
}

/// The members `relay.json` has of its own: a name in [`RelayLink::other`] that is one of these would be written twice.
const RELAY_MEMBERS: &[&str] = &["relayUrl", "relayId", "relayThumbprint", "deviceId", "token", "name", "enrolledAt", "calibration"];
/// The same for `routes.json`, a pin of `providers.json` and `providers.json`.
const ROUTES_MEMBERS: &[&str] = &["commands", "chat", "companion"];
const PIN_MEMBERS: &[&str] = &["providerId", "ed25519", "x25519", "thumbprint", "serial", "suspended", "pinnedAt"];
const PINS_MEMBERS: &[&str] = &["pins"];

/// A member that this build does not know cannot have the name of one it writes: both would be written, the file would
/// have two members of that name (a token, say), and which of them is read back is not decided by anything.
fn refuse_own_names(file: &str, other: &serde_json::Map<String, serde_json::Value>, own: &[&str]) -> Result<(), String> {
    match other.keys().find(|name| own.contains(&name.as_str())) {
        Some(name) => Err(format!("could not save {file}: {name:?} is a member this build writes itself, and is not one it does not know")),
        None => Ok(()),
    }
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

/// `relay.json` as it is on disk: the one place a [`RelayLink`]'s token is written.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredRelayLink {
    relay_url: String,
    relay_id: String,
    relay_thumbprint: String,
    device_id: String,
    token: String,
    name: String,
    enrolled_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    calibration: Option<serde_json::Value>,
    #[serde(flatten)]
    other: serde_json::Map<String, serde_json::Value>,
}

impl From<&StoredRelayLink> for RelayLink {
    fn from(s: &StoredRelayLink) -> Self {
        Self {
            relay_url: s.relay_url.clone(),
            relay_id: s.relay_id.clone(),
            relay_thumbprint: s.relay_thumbprint.clone(),
            device_id: s.device_id.clone(),
            token: s.token.clone(),
            name: s.name.clone(),
            enrolled_at: s.enrolled_at,
            calibration: s.calibration.clone(),
            other: s.other.clone(),
        }
    }
}

impl From<&RelayLink> for StoredRelayLink {
    fn from(l: &RelayLink) -> Self {
        Self {
            relay_url: l.relay_url.clone(),
            relay_id: l.relay_id.clone(),
            relay_thumbprint: l.relay_thumbprint.clone(),
            device_id: l.device_id.clone(),
            token: l.token.clone(),
            name: l.name.clone(),
            enrolled_at: l.enrolled_at,
            calibration: l.calibration.clone(),
            other: l.other.clone(),
        }
    }
}

/// Where a lane that can move goes (design 4.16.4). `Provider` is what every lane does today.
///
/// The design (6.4) also has a mode `relay+provider`: the relay first, and the provider when the relay fails. It is a
/// fourth value that the package which routes lanes adds (DK-04). A build that does not know a value in `routes.json`
/// reports the file as unusable and keeps it ([`RelayStore::routes`]), so a newer file is never read as a
/// different routing.
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
    /// Members this build does not know, kept as they were (see [`RelayLink::other`]).
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
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
    /// Members this build does not know, kept as they were (see [`RelayLink::other`]).
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

/// The pinned provider keys (`providers.json`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ProviderPins {
    pub pins: Vec<ProviderPin>,
    /// Members this build does not know, kept as they were (see [`RelayLink::other`]).
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

/// One of the files: what was read, why it could not be, and when it is read again if another program had it.
struct Slot<T> {
    name: &'static str,
    value: Option<T>,
    error: Option<LinkError>,
    retry: Option<Retry>,
}

impl<T: serde::de::DeserializeOwned> Slot<T> {
    fn open(dir: &Path, name: &'static str) -> Self {
        let (value, error) = match read_stored::<T>(&dir.join(name), &format!("{DIR}/{name}")) {
            Ok(value) => (value, None),
            Err(e) => (None, Some(e)),
        };
        let retry = error.as_ref().filter(|e| e.busy).map(|_| Retry::began(&Patience::default()));
        Self { name, value, error, retry }
    }

    fn shown(&self) -> String {
        format!("{DIR}/{}", self.name)
    }

    /// Read the file again if it could not be read when it was read last, and it is time or `force` is set. Only reads.
    fn refresh(&mut self, dir: &Path, force: bool) {
        if self.error.is_none() {
            return;
        }
        match reread::<T>(&dir.join(self.name), &self.shown(), &mut self.retry, force) {
            Reread::Still => {}
            Reread::Read(value) => {
                self.value = value;
                self.error = None;
            }
            Reread::Unusable(e) => self.error = Some(e),
        }
    }

    /// The file is there and cannot be used: it is put aside before it is written over.
    fn make_way(&mut self, dir: &Path) -> Result<(), String> {
        self.refresh(dir, true);
        if self.error.is_some() {
            self.put_unusable_aside(dir)?;
        }
        Ok(())
    }

    /// Move the file that could not be used aside, and say nothing more of it. What decides is the read that came just before
    /// (see [`Slot::refresh`]): this does not read again, so that a caller's decision is made on one read and not on two that
    /// can differ.
    fn put_unusable_aside(&mut self, dir: &Path) -> Result<(), String> {
        put_aside(&dir.join(self.name), &self.shown())?;
        self.error = None;
        self.retry = None;
        Ok(())
    }

    /// [`Slot::put_unusable_aside`] for a file that is being forgotten and not written over: it is moved or it is an error, and
    /// never copied, because a copy beside an original that stays is a link that is not forgotten. The slot is left as it was
    /// (the file is still there, and is read again if another program held it) when the move cannot be made. `true` is that what
    /// was moved is kept aside, and `false` that it was a link after all (a program let go of it between the read and the move)
    /// and its copy was removed: see [`put_aside_to_forget`].
    fn forget_unusable(&mut self, dir: &Path) -> Result<bool, String> {
        let kept = put_aside_to_forget::<T>(&dir.join(self.name), &self.shown()).map_err(|why| format!("the relay link could not be forgotten: {why}. It is still stored here, and would be used again at the next start. Close whatever has it open and try again."))?;
        self.error = None;
        self.retry = None;
        Ok(kept)
    }

    fn get(&self) -> Result<Option<&T>, LinkError> {
        match &self.error {
            Some(e) => Err(e.clone()),
            None => Ok(self.value.as_ref()),
        }
    }
}

struct Held {
    relay: Slot<StoredRelayLink>,
    routes: Slot<Routes>,
    pins: Slot<ProviderPins>,
}

/// The three files, read at `open` and replaced whole by each setter.
///
/// A file that could not be used is never read as the default: [`RelayStore::relay`], [`RelayStore::routes`] and
/// [`RelayStore::pins`] answer it with the reason, so that code that decides from them (whether a provider's keys
/// are pinned already, which lanes go where) has to say what it does when it cannot know. A trust-on-first-use
/// check that read an unusable `providers.json` as "nothing pinned" would pin whatever is offered.
pub struct RelayStore {
    dir: PathBuf,
    held: Mutex<Held>,
}

impl RelayStore {
    /// Read `<data_dir>/relay/`. A file that is not there is the default; one that cannot be used is
    /// reported and left as it is, and one that another program holds is read again later.
    pub fn open(data_dir: &Path) -> Self {
        let dir = data_dir.join(DIR);
        let held = Held { relay: Slot::open(&dir, RELAY_FILE), routes: Slot::open(&dir, ROUTES_FILE), pins: Slot::open(&dir, PROVIDERS_FILE) };
        Self { dir, held: Mutex::new(held) }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The relay this desktop is enrolled at, `Ok(None)` if there is none, and `Err` if its file cannot be used.
    pub fn relay(&self) -> Result<Option<RelayLink>, LinkError> {
        let mut held = self.held();
        held.relay.refresh(&self.dir, false);
        held.relay.get().map(|stored| stored.map(RelayLink::from))
    }

    /// Which lane goes where: the default when there is no file, and `Err` if the file cannot be used.
    pub fn routes(&self) -> Result<Routes, LinkError> {
        let mut held = self.held();
        held.routes.refresh(&self.dir, false);
        held.routes.get().map(|r| r.cloned().unwrap_or_default())
    }

    /// The pinned provider keys: none when there is no file, and `Err` if the file cannot be used.
    pub fn pins(&self) -> Result<ProviderPins, LinkError> {
        let mut held = self.held();
        held.pins.refresh(&self.dir, false);
        held.pins.get().map(|p| p.cloned().unwrap_or_default())
    }

    /// The files that could not be used, and why. Nothing in them is quoted.
    pub fn problems(&self) -> Vec<LinkError> {
        let mut held = self.held();
        held.relay.refresh(&self.dir, false);
        held.routes.refresh(&self.dir, false);
        held.pins.refresh(&self.dir, false);
        [held.relay.error.clone(), held.routes.error.clone(), held.pins.error.clone()].into_iter().flatten().collect()
    }

    /// Replace `relay.json`. `link.other` cannot have a name that the file has of its own (`token` among them).
    pub fn set_relay(&self, link: RelayLink) -> Result<(), String> {
        refuse_own_names(&format!("{DIR}/{RELAY_FILE}"), &link.other, RELAY_MEMBERS)?;
        let mut held = self.held();
        let stored = StoredRelayLink::from(&link);
        self.write(&mut held.relay, &stored)?;
        held.relay.value = Some(stored);
        Ok(())
    }

    pub fn set_routes(&self, routes: Routes) -> Result<(), String> {
        refuse_own_names(&format!("{DIR}/{ROUTES_FILE}"), &routes.other, ROUTES_MEMBERS)?;
        let mut held = self.held();
        self.write(&mut held.routes, &routes)?;
        held.routes.value = Some(routes);
        Ok(())
    }

    pub fn set_pins(&self, pins: ProviderPins) -> Result<(), String> {
        let file = format!("{DIR}/{PROVIDERS_FILE}");
        refuse_own_names(&file, &pins.other, PINS_MEMBERS)?;
        for pin in &pins.pins {
            refuse_own_names(&file, &pin.other, PIN_MEMBERS)?;
        }
        let mut held = self.held();
        self.write(&mut held.pins, &pins)?;
        held.pins.value = Some(pins);
        Ok(())
    }

    /// Forget the relay link: the file is removed first, and one that cannot be removed is an error and stays in
    /// force. A file that could not be used is put aside, not thrown away, and one that cannot be moved aside is an
    /// error too (it is never copied and left where it is: the token would stay). The copies kept of earlier unusable files
    /// (`relay.json.corrupt`, ...) are removed with the link: they can hold its token.
    pub fn forget_relay(&self) -> Result<(), String> {
        let mut held = self.held();
        let (path, shown) = (self.dir.join(RELAY_FILE), held.relay.shown());
        held.relay.refresh(&self.dir, true);
        let mut kept_aside = false;
        if held.relay.error.is_some() {
            // The read that decided is the one that is acted on: a file that is let go of since is moved aside all the same, and
            // what was moved is read where it is: a link (the token is in it) is removed, and what cannot be read is kept.
            kept_aside = held.relay.forget_unusable(&self.dir)?;
        } else {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("the relay link could not be forgotten: {shown} could not be removed ({e})")),
            }
        }
        held.relay.value = None;
        if kept_aside {
            return Ok(());
        }
        purge_asides(&path).map_err(|left| format!("the relay link was forgotten, but {left} could not be removed: a copy that was kept of it can hold its token"))
    }

    /// Write `value` as the file of `slot`, whole. A file that could not be used is put aside first, and when
    /// that fails nothing is written.
    fn write<T: Serialize + serde::de::DeserializeOwned>(&self, slot: &mut Slot<T>, value: &T) -> Result<(), String> {
        let shown = slot.shown();
        let raw = serde_json::to_string_pretty(value).map_err(|e| format!("could not encode {shown}: {e}"))?;
        slot.make_way(&self.dir)?;
        crate::secret_file::write(&self.dir.join(slot.name), raw).map_err(|e| format!("could not save {shown}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::{assert_private, assert_private_dir, TempDir};
    use std::time::Duration;

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
            other: Default::default(),
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
                other: Default::default(),
            }],
            other: Default::default(),
        }
    }

    /// The store's files, made in a data folder of its own.
    fn folder(dir: &TempDir) -> PathBuf {
        let folder = dir.0.join("relay");
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// Say when a held file is read again: at once.
    fn due_now() -> Option<Retry> {
        Some(Retry::began(&Patience { start: vec![], later: vec![Duration::ZERO], window: Duration::ZERO }))
    }

    #[test]
    fn the_three_files_round_trip_through_a_restart_and_are_private() {
        let dir = TempDir::new("relay-store");
        let store = RelayStore::open(&dir.0);
        assert_eq!((store.relay().unwrap(), store.routes().unwrap(), store.pins().unwrap()), (None, Routes::default(), ProviderPins::default()));
        assert!(store.problems().is_empty());

        let (link, routes, held) = (relay(), Routes { commands: Route::Relay, chat: Route::Provider, companion: Route::Off, other: Default::default() }, pins());
        store.set_relay(link.clone()).unwrap();
        store.set_routes(routes.clone()).unwrap();
        store.set_pins(held.clone()).unwrap();
        assert_eq!(store.relay().unwrap(), Some(link.clone()), "as it was set");

        let again = RelayStore::open(&dir.0);
        assert_eq!(again.relay().unwrap(), Some(link));
        assert_eq!(again.routes().unwrap(), routes);
        assert_eq!(again.pins().unwrap(), held);
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
        assert_eq!(RelayStore::open(&dir.0).relay().unwrap(), None, "and it is durable");
        again.forget_relay().unwrap();
    }

    #[test]
    fn a_lane_that_is_not_named_is_on_the_provider() {
        let dir = TempDir::new("relay-routes");
        std::fs::write(folder(&dir).join("routes.json"), r#"{"commands":"relay"}"#).unwrap();
        let store = RelayStore::open(&dir.0);
        assert_eq!(store.routes().unwrap(), Routes { commands: Route::Relay, ..Routes::default() });
        assert_eq!(Routes::default(), Routes { commands: Route::Provider, chat: Route::Provider, companion: Route::Provider, other: Default::default() });
    }

    #[test]
    fn a_file_that_cannot_be_used_is_an_error_and_never_the_default() {
        // A trust-on-first-use check that read an unusable `providers.json` as "nothing pinned" would pin whatever
        // is offered; routing that read an unusable `routes.json` as the default would move lanes back to the
        // provider without a word. The answer is the reason, and what asks has to say what it does without one.
        for (name, unknown) in [("relay.json", "[1, 2]"), ("routes.json", r#"{"commands":"relay+provider"}"#), ("providers.json", r#"{"pins":"all of them"}"#)] {
            let dir = TempDir::new("relay-unusable-answer");
            std::fs::write(folder(&dir).join(name), unknown).unwrap();
            let store = RelayStore::open(&dir.0);
            let (relay, routes, pins) = (store.relay(), store.routes(), store.pins());
            let unusable = |what: &str, error: Option<&LinkError>| {
                let error = error.unwrap_or_else(|| panic!("{name}: {what} answered when its file cannot be used"));
                assert_eq!(error.file, format!("relay/{name}"));
            };
            match name {
                "relay.json" => unusable("relay()", relay.as_ref().err()),
                "routes.json" => unusable("routes()", routes.as_ref().err()),
                _ => unusable("pins()", pins.as_ref().err()),
            }
            // And the other two are as they were: one bad file is not all three.
            let answered = [name != "relay.json" && relay.is_ok(), name != "routes.json" && routes.is_ok(), name != "providers.json" && pins.is_ok()];
            assert_eq!(answered.into_iter().filter(|a| *a).count(), 2, "{name}");
            assert_eq!(std::fs::read_to_string(dir.0.join("relay").join(name)).unwrap(), unknown, "{name}: and it is as it was");
        }
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
            let folder = folder(&dir);
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
    fn what_a_newer_build_added_to_a_file_survives_this_one_rewriting_it() {
        // `deny_unknown_fields` would make a newer file unusable here; dropping what is not known on the way back out would
        // make a rewrite here a downgrade of the file. The members are kept as they were, at the top and in a pin.
        let dir = TempDir::new("relay-newer");
        let folder = folder(&dir);
        let relay_json = serde_json::json!({
            "relayUrl": "https://relay.example.com", "relayId": "rly-1", "relayThumbprint": "t", "deviceId": "dev-1", "token": "oaiyrt1.T", "name": "PC",
            "enrolledAt": "2026-09-01T00:00:00Z", "fromTheFuture": {"a": [1, 2]},
        });
        let routes_json = serde_json::json!({"commands": "relay", "later": true});
        let pins_json = serde_json::json!({"pins": [{
            "providerId": "p", "ed25519": "e", "x25519": "x", "thumbprint": "t", "serial": 1, "pinnedAt": "2026-09-01T00:00:00Z", "pinNote": "kept",
        }], "revision": 9});
        for (name, json) in [("relay.json", &relay_json), ("routes.json", &routes_json), ("providers.json", &pins_json)] {
            std::fs::write(folder.join(name), json.to_string()).unwrap();
        }
        let store = RelayStore::open(&dir.0);
        assert!(store.problems().is_empty(), "a newer member is not a reason to refuse the file: {:?}", store.problems());

        let mut link = store.relay().unwrap().unwrap();
        link.name = "Renamed".into();
        store.set_relay(link).unwrap();
        let mut routes = store.routes().unwrap();
        routes.chat = Route::Relay;
        store.set_routes(routes).unwrap();
        let mut held = store.pins().unwrap();
        held.pins[0].serial = 2;
        store.set_pins(held).unwrap();

        let read = |name: &str| serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(folder.join(name)).unwrap()).unwrap();
        assert_eq!((read("relay.json")["fromTheFuture"].clone(), read("relay.json")["name"].clone()), (serde_json::json!({"a": [1, 2]}), serde_json::json!("Renamed")));
        assert_eq!((read("routes.json")["later"].clone(), read("routes.json")["chat"].clone()), (serde_json::json!(true), serde_json::json!("relay")));
        assert_eq!((read("providers.json")["revision"].clone(), read("providers.json")["pins"][0]["pinNote"].clone(), read("providers.json")["pins"][0]["serial"].clone()), (serde_json::json!(9), serde_json::json!("kept"), serde_json::json!(2)));
    }

    #[test]
    fn what_a_newer_build_added_is_kept_as_far_as_json_values_keep_it_and_no_further() {
        // The limit that `RelayLink::other` says it has: a value is kept as `serde_json` reads it. What a newer build wrote
        // as a number of 64 bits or fewer, or as a decimal a `f64` holds, comes back as it was; an integer over 64 bits is
        // the nearest `f64`, and the members of an object come back sorted. If this changes (a `RawValue` that keeps the
        // text and the order), the doc on `other` changes with it.
        let dir = TempDir::new("relay-numbers");
        let folder = folder(&dir);
        std::fs::write(
            folder.join("routes.json"),
            r#"{"commands":"relay","u":18446744073709551615,"i":-9223372036854775808,"f":0.1,"big":18446744073709551616,"z":{"b":1,"a":2}}"#,
        )
        .unwrap();
        let store = RelayStore::open(&dir.0);
        let routes = store.routes().unwrap();
        store.set_routes(routes).unwrap();
        let text = std::fs::read_to_string(folder.join("routes.json")).unwrap();
        for kept in ["18446744073709551615", "-9223372036854775808", "0.1"] {
            assert!(text.contains(kept), "{kept} is kept: {text}");
        }
        assert!(!text.contains("18446744073709551616") && text.contains("1.8446744073709552e+19"), "an integer over 64 bits is the nearest f64: {text}");
        assert!(text.find("\"a\": 2").unwrap() < text.find("\"b\": 1").unwrap(), "the members of an object are sorted: {text}");
    }

    #[test]
    fn a_member_this_build_does_not_know_cannot_have_the_name_of_one_it_writes() {
        // Both would be written: a file with two `token` members, and a reader that takes the first or the last.
        let dir = TempDir::new("relay-own-names");
        let store = RelayStore::open(&dir.0);
        let first = relay();
        store.set_relay(first.clone()).unwrap();
        let before = std::fs::read(dir.0.join("relay").join("relay.json")).unwrap();

        for own in RELAY_MEMBERS {
            let mut link = first.clone();
            link.other.insert((*own).to_string(), serde_json::json!("x"));
            let refused = store.set_relay(link).unwrap_err();
            assert!(refused.contains(&format!("{own:?}")) && refused.contains("relay/relay.json"), "{own}: {refused}");
            assert!(!refused.contains("TOPSECRETTOKEN"), "{refused}");
        }
        assert_eq!(std::fs::read(dir.0.join("relay").join("relay.json")).unwrap(), before, "a refused one writes nothing");
        assert_eq!(store.relay().unwrap(), Some(first), "and changes nothing held");

        for own in ROUTES_MEMBERS {
            let mut routes = Routes::default();
            routes.other.insert((*own).to_string(), serde_json::json!(1));
            assert!(store.set_routes(routes).is_err(), "{own}");
        }
        for own in PINS_MEMBERS {
            let mut held = pins();
            held.other.insert((*own).to_string(), serde_json::json!(1));
            assert!(store.set_pins(held).is_err(), "{own}");
        }
        for own in PIN_MEMBERS {
            let mut held = pins();
            held.pins[0].other.insert((*own).to_string(), serde_json::json!(1));
            assert!(store.set_pins(held).is_err(), "{own}");
        }
        assert!(!dir.0.join("relay").join("routes.json").exists() && !dir.0.join("relay").join("providers.json").exists());

        // A name that is not one of the file's own is kept, as before.
        let mut link = store.relay().unwrap().unwrap();
        link.other.insert("fromTheFuture".into(), serde_json::json!(true));
        store.set_relay(link).unwrap();
        assert!(std::fs::read_to_string(dir.0.join("relay").join("relay.json")).unwrap().contains("fromTheFuture"));
    }

    #[test]
    fn the_names_that_are_refused_are_the_names_the_files_write() {
        // The lists are kept by hand next to the structs: a member added to a struct and not to its list would be a name that
        // can be written twice again. This is the file's own idea of its names, written out.
        fn names(value: &impl Serialize) -> Vec<String> {
            let mut names: Vec<String> = serde_json::to_value(value).unwrap().as_object().unwrap().keys().cloned().collect();
            names.sort();
            names
        }
        fn sorted(list: &[&str]) -> Vec<String> {
            let mut list: Vec<String> = list.iter().map(|s| s.to_string()).collect();
            list.sort();
            list
        }
        assert_eq!(names(&StoredRelayLink::from(&relay())), sorted(RELAY_MEMBERS));
        assert_eq!(names(&Routes::default()), sorted(ROUTES_MEMBERS));
        assert_eq!(names(&pins().pins[0]), sorted(PIN_MEMBERS));
        assert_eq!(names(&pins()), sorted(PINS_MEMBERS));
    }

    #[test]
    fn a_file_another_program_holds_is_read_again_and_a_file_that_is_fine_is_never_put_aside_for_it() {
        // A folder where the file belongs cannot be read, as a file a scanner holds cannot.
        let dir = TempDir::new("relay-held");
        let held = folder(&dir).join("routes.json");
        std::fs::create_dir(&held).unwrap();
        let store = RelayStore::open(&dir.0);
        assert!(store.routes().is_err(), "it could not be read");
        assert!(store.routes().is_err(), "and the next ask is not time yet to read it again");

        // It is let go of: it is a file, and a good one. The time comes, and it is read.
        std::fs::remove_dir(&held).unwrap();
        std::fs::write(&held, r#"{"commands":"relay"}"#).unwrap();
        assert!(store.routes().is_err(), "not yet time");
        store.held().routes.retry = due_now();
        assert_eq!(store.routes().unwrap().commands, Route::Relay, "read again when it was time");
        assert!(store.problems().is_empty());
        assert!(!dir.0.join("relay").join("routes.json.corrupt").exists(), "it was read, not moved aside");

        // A write that comes while it is still held does not take it for a bad file either, if it can be read by then.
        let other = TempDir::new("relay-held-write");
        let file = folder(&other).join("providers.json");
        std::fs::create_dir(&file).unwrap();
        let store = RelayStore::open(&other.0);
        assert!(store.pins().is_err());
        std::fs::remove_dir(&file).unwrap();
        std::fs::write(&file, serde_json::to_string(&pins()).unwrap()).unwrap();
        store.set_pins(ProviderPins::default()).unwrap();
        assert!(!other.0.join("relay").join("providers.json.corrupt").exists(), "a good file is not put aside for being written over");
        assert!(store.pins().unwrap().pins.is_empty());
    }

    #[test]
    fn forgetting_a_relay_file_that_could_not_be_used_keeps_it_aside() {
        let dir = TempDir::new("relay-forget-bad");
        let folder = folder(&dir);
        std::fs::write(folder.join("relay.json"), b"{ not json").unwrap();
        let store = RelayStore::open(&dir.0);
        assert_eq!(store.problems().len(), 1);
        store.forget_relay().unwrap();
        assert_eq!(std::fs::read(folder.join("relay.json.corrupt")).unwrap(), b"{ not json");
        assert!(store.problems().is_empty() && !folder.join("relay.json").exists());
    }

    #[test]
    fn forgetting_a_relay_file_that_is_held_reads_it_once_and_acts_on_that_read() {
        // The same as the provider link's: a read that finds the file unusable and a second one inside `make_way` that finds it
        // readable (a program let go of it between them) left `relay.json` and its token in the folder, with the link said
        // to be forgotten and nothing purged.
        let dir = TempDir::new("relay-forget-held");
        let folder = folder(&dir);
        std::fs::create_dir(folder.join("relay.json")).unwrap();
        let store = RelayStore::open(&dir.0);
        assert!(store.relay().is_err(), "held");

        let reads = crate::link::reads::so_far();
        store.forget_relay().unwrap();
        assert_eq!(crate::link::reads::so_far() - reads, 1, "the file is read once to decide, not again to act");
        assert!(folder.join("relay.json.corrupt").is_dir(), "what could not be read is kept");
        assert!(!folder.join("relay.json").exists());
        assert!(store.problems().is_empty() && store.relay().unwrap().is_none());
    }

    #[test]
    fn a_relay_file_that_cannot_be_moved_aside_is_not_forgotten_and_not_copied() {
        // As the provider link's: the token is in the file, and a copy beside an original that stays is a link that is not
        // forgotten. `forget_relay` answered Ok with `relay.json` still in the folder.
        let dir = TempDir::new("relay-forget-held-move");
        let folder = folder(&dir);
        std::fs::write(folder.join("relay.json"), b"{ not json").unwrap();
        let store = RelayStore::open(&dir.0);
        assert_eq!(store.problems().len(), 1);
        let held = crate::link::held_files::hold(&folder.join("relay.json"));

        let refused = store.forget_relay().unwrap_err();
        assert!(refused.starts_with("the relay link could not be forgotten: relay/relay.json could not be moved aside"), "{refused}");
        assert_eq!(std::fs::read(folder.join("relay.json")).unwrap(), b"{ not json", "the file is where it was");
        assert!(!folder.join("relay.json.corrupt").exists(), "and no copy of it was made");
        assert_eq!(store.problems().len(), 1, "and it is still said of it that it cannot be used");

        drop(held);
        store.forget_relay().unwrap();
        assert_eq!(std::fs::read(folder.join("relay.json.corrupt")).unwrap(), b"{ not json");
        assert!(!folder.join("relay.json").exists() && store.problems().is_empty());
    }

    #[test]
    fn a_relay_file_that_became_a_link_between_the_read_and_the_move_is_forgotten_and_its_copy_is_not_kept() {
        // As the provider link's: a program lets go of a held relay.json between the read that found it unusable and the move, and
        // what is moved is the relay link with its token in it.
        let dir = TempDir::new("relay-let-go-before-the-move");
        let folder = folder(&dir);
        let file = folder.join("relay.json");
        std::fs::create_dir(&file).unwrap();
        let store = RelayStore::open(&dir.0);
        assert!(store.relay().is_err(), "held");
        let (there, text) = (file.clone(), serde_json::to_string_pretty(&StoredRelayLink::from(&relay())).unwrap());
        crate::link::after_the_read::once(move || {
            std::fs::remove_dir(&there).unwrap();
            std::fs::write(&there, text).unwrap();
        });

        store.forget_relay().unwrap();
        assert!(!file.exists(), "the file is not where it was");
        for entry in std::fs::read_dir(&folder).unwrap().flatten() {
            let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
            assert!(!text.contains("TOPSECRETTOKEN"), "the token is in {:?}", entry.file_name());
        }
        assert!(store.problems().is_empty() && RelayStore::open(&dir.0).relay().unwrap().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn a_relay_file_whose_hold_changes_from_no_reading_to_no_moving_is_not_forgotten() {
        // The review's h6: the same change of hold between the read and the move as for the provider's file, with the same two
        // answers that are allowed: not forgotten with the file where it was, or forgotten with the token nowhere in the folder.
        use std::os::windows::fs::OpenOptionsExt;
        let dir = TempDir::new("relay-forget-hold-changes");
        let folder = folder(&dir);
        let file = folder.join("relay.json");
        RelayStore::open(&dir.0).set_relay(relay()).unwrap();
        let exclusive = std::fs::OpenOptions::new().read(true).share_mode(0).open(&file).unwrap();
        let store = RelayStore::open(&dir.0);
        assert!(store.relay().is_err());
        let swapper = {
            let file = file.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(60));
                drop(exclusive);
                // FILE_SHARE_READ: no delete, no rename. The store's own attempt to move the file has it open for a moment, so this is
                // tried again, and not at all if the file has been moved.
                let shared = (0..200).find_map(|_| match std::fs::OpenOptions::new().read(true).share_mode(1).open(&file) {
                    Ok(held) => Some(Some(held)),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(None),
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(2));
                        None
                    }
                });
                std::thread::sleep(Duration::from_millis(2500));
                drop(shared);
            })
        };
        let answer = store.forget_relay();
        let token_in_the_folder = || std::fs::read_dir(&folder).unwrap().flatten().any(|e| std::fs::read_to_string(e.path()).unwrap_or_default().contains("TOPSECRETTOKEN"));
        match &answer {
            Err(_) => {
                assert!(file.exists(), "not forgotten, and the token is where it was");
                assert!(!folder.join("relay.json.corrupt").exists(), "and nothing was copied");
            }
            Ok(()) => assert!(!file.exists() && !token_in_the_folder(), "forgotten, so the token is not in the folder"),
        }
        swapper.join().unwrap();
    }
    #[test]
    fn forgetting_the_relay_takes_the_copies_kept_of_it_and_no_others() {
        // A copy kept of an unusable `relay.json` can hold the token: forgetting the link is what the owner asked for.
        let dir = TempDir::new("relay-forget-copies");
        let folder = folder(&dir);
        let store = RelayStore::open(&dir.0);
        for earlier in ["relay.json.corrupt", "relay.json.corrupt.1", "relay.json.corrupt.12", "routes.json.corrupt", "relay.json.corrupt.old", "relay.json.bak"] {
            std::fs::write(folder.join(earlier), r#"{"token":"oaiyrt1.AN.OLD.ONE"}"#).unwrap();
        }
        store.set_relay(relay()).unwrap();
        store.forget_relay().unwrap();
        let mut left: Vec<String> = std::fs::read_dir(&folder).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, ["relay.json.bak", "relay.json.corrupt.old", "routes.json.corrupt"], "only the copies of relay.json that this store keeps");
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
        assert!(store.relay().unwrap().is_some(), "the link is still in force");
    }

    /// Whether a type can be serialised, asked of the type itself (a bound would not compile for one that cannot be): the
    /// method that is found first is the one for a type that implements `Serialize`, and the other is the fallback.
    struct Probe<T>(std::marker::PhantomData<T>);
    trait Yes {
        fn answer(&self) -> bool {
            true
        }
    }
    impl<T: Serialize> Yes for Probe<T> {}
    trait No {
        fn answer(&self) -> bool {
            false
        }
    }
    impl<T> No for &Probe<T> {}
    macro_rules! can_be_serialised {
        ($t:ty) => {
            (&Probe::<$t>(std::marker::PhantomData)).answer()
        };
    }
    #[test]
    fn the_token_is_in_the_file_and_nowhere_else() {
        // The type that holds it cannot be serialised (a status route that tried would not build), and what writes it
        // is the store's own private copy.
        assert!(can_be_serialised!(Routes), "the probe finds a type that can");
        assert!(!can_be_serialised!(RelayLink), "RelayLink holds the token: it must not derive Serialize");

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
        // Nor an unknown variant that is a key pasted where a route goes.
        std::fs::write(dir.0.join("relay").join("routes.json"), r#"{"commands":"oaiyrt1.TOPSECRETTOKEN"}"#).unwrap();
        let routes = RelayStore::open(&dir.0).routes().unwrap_err();
        assert!(!format!("{routes:?}").contains("TOPSECRETTOKEN"), "{routes:?}");
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