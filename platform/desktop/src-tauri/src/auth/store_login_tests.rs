//! What the login asks of the credential store beyond what it was made for: a list of the records, a revocation
//! by a rule (one write for all of them), the time an elevation ends, and extra fields on a record.

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use super::presets::{App, Preset};
use super::scopes::ScopeSet;
use super::store::*;
use super::token::Kind;
use crate::secret_file::testing::TempDir;

const T0: u64 = 1_790_000_000_000;
const DAY: u64 = 24 * 60 * 60_000;

fn session() -> MintSpec {
    let mut s = MintSpec::new(Kind::Ses, "a browser", Preset::Owner.scopes(), DAY);
    s.app = Some(App::Dash);
    s.preset = Some(Preset::Owner);
    s
}

fn pat() -> MintSpec {
    MintSpec::new(
        Kind::Pat,
        "a tool",
        ScopeSet::of(&["system.read"]),
        30 * DAY,
    )
}

fn open(dir: &TempDir, clock: &Arc<ManualClock>) -> AuthStore {
    AuthStore::open(
        &dir.0.join("auth"),
        Host::Server,
        clock.clone(),
        Arc::new(SecureWriter),
        None,
    )
    .unwrap()
}

/// Counts the writes it is asked for.
struct Counting(AtomicUsize);

impl FileWriter for Counting {
    fn write(&self, _: &std::path::Path, _: &[u8]) -> io::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn the_records_are_listed_without_any_secret() {
    let clock = Arc::new(ManualClock::new(T0));
    let store = AuthStore::memory(clock);
    let a = store.mint(session()).unwrap();
    let b = store.mint(pat()).unwrap();
    let mut ids: Vec<String> = store.records().into_iter().map(|r| r.id).collect();
    ids.sort();
    let mut expected = vec![a.id.clone(), b.id.clone()];
    expected.sort();
    assert_eq!(ids, expected);
    let text = format!("{:?}", store.records());
    assert!(!text.contains(&a.token) && !text.contains(&b.token));
}

#[test]
fn a_revocation_by_a_rule_takes_what_the_rule_names_and_writes_once() {
    let clock = Arc::new(ManualClock::new(T0));
    let writer = Arc::new(Counting(AtomicUsize::new(0)));
    let dir = TempDir::new("revoke-where");
    let store = AuthStore::open(
        &dir.0.join("auth"),
        Host::Server,
        clock.clone(),
        writer.clone(),
        None,
    )
    .unwrap();
    let mut sessions = Vec::new();
    for _ in 0..5 {
        sessions.push(store.mint(session()).unwrap());
    }
    let tool = store.mint(pat()).unwrap();
    let keep = sessions[2].id.clone();
    let before = writer.0.load(Ordering::SeqCst);
    // Every session but one: the shape of "log out everywhere else".
    let revoked = store.revoke_where("password_changed", &|r| r.kind == Kind::Ses && r.id != keep);
    assert_eq!(revoked.len(), 4);
    assert!(!revoked.contains(&keep));
    assert_eq!(
        writer.0.load(Ordering::SeqCst),
        before + 1,
        "one write for four revocations"
    );
    // The reason is the one given, and the ones left are alive.
    for s in &sessions {
        let alive = store.authenticate(&s.token, None).is_ok();
        assert_eq!(alive, s.id == keep, "{}", s.id);
    }
    assert!(
        store.authenticate(&tool.token, None).is_ok(),
        "the token was not named"
    );
    let err = store.authenticate(&sessions[0].token, None).unwrap_err();
    assert_eq!(err.reason(), Some("password_changed"));
    // Nothing left to revoke by the same rule: no ids, no write.
    let again = store.revoke_where("password_changed", &|r| r.kind == Kind::Ses && r.id != keep);
    assert!(again.is_empty());
    assert_eq!(writer.0.load(Ordering::SeqCst), before + 1);
}

#[test]
fn a_revocation_by_a_rule_survives_a_restart() {
    let dir = TempDir::new("revoke-where-restart");
    let clock = Arc::new(ManualClock::new(T0));
    let store = open(&dir, &clock);
    let a = store.mint(session()).unwrap();
    let b = store.mint(session()).unwrap();
    store.revoke_where("revoked", &|r| r.id == a.id);
    drop(store);
    let store = open(&dir, &clock);
    assert!(store.authenticate(&a.token, None).is_err());
    assert!(store.authenticate(&b.token, None).is_ok());
}

#[test]
fn the_extra_fields_of_a_mint_are_in_the_file_and_come_back() {
    let dir = TempDir::new("mint-extra");
    let clock = Arc::new(ManualClock::new(T0));
    let store = open(&dir, &clock);
    let mut spec = session();
    spec.extra.insert(
        "login".into(),
        json!({ "ua": "curl/8", "ip": "203.0.113.9" }),
    );
    let s = store.mint(spec).unwrap();
    assert_eq!(
        store.record(&s.id).unwrap().extra["login"]["ip"],
        "203.0.113.9"
    );
    let file: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.0.join("auth").join("credentials.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(file["credentials"][0]["login"]["ua"], "curl/8");
    drop(store);
    let store = open(&dir, &clock);
    assert_eq!(store.record(&s.id).unwrap().extra["login"]["ua"], "curl/8");
}

#[test]
fn the_end_of_an_elevation_is_reported_and_a_revocation_ends_it() {
    let clock = Arc::new(ManualClock::new(T0));
    let store = AuthStore::memory(clock);
    let s = store.mint(session()).unwrap();
    assert_eq!(store.elevated_until(&s.id), None);
    store.set_elevated_until(&s.id, T0 + 600_000);
    assert_eq!(store.elevated_until(&s.id), Some(T0 + 600_000));
    assert!(store.authenticate(&s.token, None).unwrap().elevated);
    store.revoke(&s.id, "logged_out");
    assert_eq!(store.elevated_until(&s.id), None);
}
