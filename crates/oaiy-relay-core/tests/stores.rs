//! The stores: the profile and the poll cursor on a file system, the rule that "never stored" and "could not read" are different, a crash between the items and the cursor, and the
//! keystore adapter over `oaiy-keystore` (the desktop's K1).

use std::fs;
use std::path::PathBuf;

use oaiy_relay_core::client::keystore::KeystoreSecrets;
use oaiy_relay_core::client::store::write_atomic;
use oaiy_relay_core::client::*;
use oaiy_relay_core::json;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::url::RelayUrl;

/// A scratch directory under the target directory's `tmp`, removed when the test ends (also when it fails).
struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.0
    }
}

impl AsRef<std::path::Path> for Scratch {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl From<&Scratch> for PathBuf {
    fn from(s: &Scratch) -> PathBuf {
        s.0.clone()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn scratch(name: &str) -> Scratch {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("stores-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

fn profile() -> RelayProfile {
    let desktop = Signer::generate().unwrap().verify_key();
    RelayProfile {
        kind: ProfileKind::Phone,
        relay: RelayUrl::parse("https://relay.example.com").unwrap(),
        relay_id: "rly-0NHS09TV1tfY2drb3N3e3w".into(),
        relay_thumbprint: "b7dKD2-DlMApkGljz-RJwDdNcyxwyFBpSmhz2cRBnaw".into(),
        device_id: "dev-sLGys7S1tre4ubq7vL2-vw".into(),
        name: "Test phone".into(),
        enrolled_at: 1_790_000_000,
        app_id: Some("aokie".into()),
        grants: vec!["state_read".into(), "rtc_signal".into()],
        peer: Some(PeerPin {
            desktop_connection_id: "dev-oKGio6SlpqeoqaqrrK2urw".into(),
            desktop_name: "Front desk PC".into(),
            desktop_endpoint: desktop,
            desktop_x25519: X25519Secret::generate().unwrap().public_key(),
            host_ed25519: Signer::generate().unwrap().verify_key(),
            host_x25519: X25519Secret::generate().unwrap().public_key(),
        }),
    }
}

#[test]
fn a_profile_round_trips_and_a_missing_file_is_none_but_a_damaged_one_is_an_error() {
    let dir = scratch("profile");
    let store = FileProfileStore::new(dir.join("relay.json"));
    assert_eq!(store.load().unwrap(), None, "never stored");
    let p = profile();
    store.save(&p).unwrap();
    assert_eq!(store.load().unwrap().unwrap(), p);
    // A desktop profile has no peer and no grants.
    let mut d = p.clone();
    d.kind = ProfileKind::Desktop;
    d.peer = None;
    d.app_id = None;
    d.grants.clear();
    store.save(&d).unwrap();
    assert_eq!(store.load().unwrap().unwrap(), d);
    // Damaged in every way that matters: an error, never "not enrolled".
    let path = dir.join("relay.json");
    let good = fs::read_to_string(&path).unwrap();
    for bad in [
        "".to_string(),
        "{".to_string(),
        good.replace("\"v\":1", "\"v\":2"),
        good.replace("\"kind\":\"desktop\"", "\"kind\":\"toaster\""),
        good.replace("rly-0NHS09TV1tfY2drb3N3e3w", "rly-short"),
        good.replace("https://relay.example.com", "http://relay.example.com"),
        good.replace("dev-sLGys7S1tre4ubq7vL2-vw", "nope"),
    ] {
        fs::write(&path, &bad).unwrap();
        assert!(store.load().is_err(), "{bad}");
    }
    store.clear().unwrap();
    assert_eq!(store.load().unwrap(), None);
    store.clear().unwrap();
}

#[test]
fn a_profile_with_a_key_of_small_order_in_its_pin_is_refused_on_load() {
    let dir = scratch("smallorder");
    let store = FileProfileStore::new(dir.join("relay.json"));
    let p = profile();
    store.save(&p).unwrap();
    let text = fs::read_to_string(dir.join("relay.json")).unwrap();
    let doc = json::parse(text.as_bytes()).unwrap();
    let x = doc.get("peer").unwrap().get_str("desktopX25519").unwrap().to_string();
    // The identity point of X25519 (u = 0): 43 characters of `A`.
    fs::write(dir.join("relay.json"), text.replace(&x, &"A".repeat(43))).unwrap();
    assert!(store.load().is_err());
}

#[test]
fn the_cursor_and_the_inbox_are_written_items_first_and_survive_a_restart() {
    let dir = scratch("cursor");
    let mut store = FilePollStore::new(&dir);
    assert_eq!(store.load().unwrap(), PollCursor::default());
    let raw = |seq: u64, id: &str| {
        json::parse(
            format!(
                r#"{{"seq":{seq},"id":"{id}","lane":"cmd","from":"prov-wMHCw8TFxsfIycrLzM3Ozw","at":1,"exp":99,"hdr":{{"ct":"sealed1"}},"body":"b"}}"#
            )
            .as_bytes(),
        )
        .unwrap()
    };
    let items = vec![
        AcceptedItem { seq: 1, item: Item::from_json(&raw(1, "a")), raw: raw(1, "a") },
        AcceptedItem { seq: 2, item: Item::from_json(&raw(2, "b")), raw: raw(2, "b") },
    ];
    store.persist(&PersistBatch { items: &items, since: 2, epoch: "eVp54C0-EJY", reset: false }).unwrap();
    let mut again = FilePollStore::new(&dir);
    assert_eq!(again.load().unwrap(), PollCursor { since: 2, epoch: Some("eVp54C0-EJY".into()) });
    let inbox = again.read_inbox().unwrap();
    assert_eq!(inbox.len(), 2);
    assert_eq!(inbox[1].get_str("id"), Some("b"));
    // A reset adopts a lower cursor, and says so in the file.
    again.persist(&PersistBatch { items: &[], since: 1, epoch: "AAAAAAAAAAA", reset: true }).unwrap();
    assert_eq!(again.load().unwrap().since, 1);
    assert!(fs::read_to_string(dir.join("cursor.json")).unwrap().contains("mailbox reset: in-flight items may be lost"));
    // clear_epoch keeps the cursor.
    again.clear_epoch().unwrap();
    assert_eq!(again.load().unwrap(), PollCursor { since: 1, epoch: None });
    // A cursor that is damaged is an error: the loop does not start and does not begin again from zero.
    fs::write(dir.join("cursor.json"), "{\"since\":-1}").unwrap();
    assert!(again.load().is_err());
    fs::write(dir.join("cursor.json"), "{\"since\":1,\"epoch\":\"short\"}").unwrap();
    assert!(again.load().is_err());
}

#[test]
fn a_crash_between_the_items_and_the_cursor_redelivers_and_loses_nothing() {
    // The items are flushed before the cursor is written. Make the cursor write fail (its path is a directory), so that the process "dies" between the two.
    let dir = scratch("crash");
    let mut store = FilePollStore::new(&dir);
    fs::create_dir_all(dir.join("cursor.json")).unwrap();
    let item = json::parse(br#"{"seq":6,"id":"x","lane":"cmd","from":"relay","at":1,"exp":99,"hdr":{},"body":"b"}"#).unwrap();
    let items = vec![AcceptedItem { seq: 6, item: Item::from_json(&item), raw: item }];
    assert!(store.persist(&PersistBatch { items: &items, since: 6, epoch: "eVp54C0-EJY", reset: false }).is_err(), "the cursor could not be written");
    fs::remove_dir_all(dir.join("cursor.json")).unwrap();
    let mut restarted = FilePollStore::new(&dir);
    assert_eq!(restarted.load().unwrap().since, 0, "the cursor was not advanced: the item will come again");
    assert_eq!(restarted.read_inbox().unwrap().len(), 1, "and what was written before it is still there");
}
#[test]
fn an_atomic_write_leaves_no_temporary_file_and_a_failed_one_leaves_the_old_content() {
    let dir = scratch("atomic");
    let path = dir.join("f.json");
    write_atomic(&path, b"one").unwrap();
    write_atomic(&path, b"two").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"two");
    let names: Vec<String> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names, vec!["f.json"], "no temporary file left");
    // A target that is a directory cannot be replaced: the write fails and the temporary file is removed.
    let blocked = dir.join("d");
    fs::create_dir_all(blocked.join("inner")).unwrap();
    assert!(write_atomic(&blocked, b"x").is_err());
    assert!(fs::read_dir(&dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().contains(".tmp")));
}

#[test]
fn the_keystore_adapter_keeps_the_keystores_rules() {
    let dir = scratch("keystore");
    let keys = oaiy_keystore::open_at(dir.join("keys"), oaiy_keystore::ProviderChoice::Auto);
    let Ok(keys) = keys else {
        eprintln!("SKIPPED: this machine has no keystore provider ({:?})", keys.err());
        return;
    };
    let secrets = KeystoreSecrets(keys);
    assert!(secrets.get(SECRET_TOKEN).unwrap().is_none(), "never stored is Ok(None)");
    secrets.put(SECRET_TOKEN, b"oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8").unwrap();
    assert_eq!(&secrets.get(SECRET_TOKEN).unwrap().unwrap()[..8], b"oaiyrt1.");
    secrets.put(SECRET_TOKEN, b"replaced").unwrap();
    assert_eq!(&*secrets.get(SECRET_TOKEN).unwrap().unwrap(), b"replaced");
    secrets.delete(SECRET_TOKEN).unwrap();
    secrets.delete(SECRET_TOKEN).unwrap();
    assert!(secrets.get(SECRET_TOKEN).unwrap().is_none());
    assert!(secrets.put("Not A Name", b"x").is_err(), "a name the keystore does not accept");
}

#[test]
fn memory_stores_say_never_stored_and_cannot_read_apart() {
    let secrets = MemorySecretStore::new();
    assert!(secrets.get("relay.token").unwrap().is_none());
    secrets.broken.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(secrets.get("relay.token").is_err(), "an error is not the same as never stored");
    assert!(secrets.put("relay.token", b"x").is_err());
    secrets.broken.store(false, std::sync::atomic::Ordering::SeqCst);
    secrets.put("relay.token", b"x").unwrap();
    assert!(secrets.get("relay.token").unwrap().is_some());
}
