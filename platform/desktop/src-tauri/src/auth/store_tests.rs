//! Tests of the credential store: the rules of design 4.1, the grants of 4.2.3, the chain of 4.1 and
//! the derivation of 4.6, each named after the rule it holds.

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use super::audit::{AuditLog, LogFile};
use super::chain::STATIC_PARENT;
use super::presets::{App, Preset};
use super::principal::{Principal, PrincipalKind};
use super::scopes::{self, ScopeSet};
use super::store::*;
use super::token::{self, Kind, MintError};
use crate::secret_file::testing::TempDir;

const T0: u64 = 1_790_000_000_000;
const MIN: u64 = 60_000;
const HOUR: u64 = 60 * MIN;
const DAY: u64 = 24 * HOUR;

// The design's known-answer vector (4.1).
const ID_BYTES: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
const KNOWN_TOKEN: &str = "oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
const KNOWN_HASH: &str = "ea866a757e4c38babfa8127cbe9a409d3e1f93a00ff1488ff735fcf917afffd0";

#[cfg(unix)]
const ENOSPC: i32 = 28;
#[cfg(windows)]
const ENOSPC: i32 = 112;

fn clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new(T0))
}

fn auth_dir(dir: &TempDir) -> PathBuf {
    dir.0.join("auth")
}

fn open_as(dir: &TempDir, clock: &Arc<ManualClock>, host: Host) -> Result<AuthStore, StoreError> {
    AuthStore::open(
        &auth_dir(dir),
        host,
        clock.clone(),
        Arc::new(SecureWriter),
        None,
    )
}

fn open(dir: &TempDir, clock: &Arc<ManualClock>) -> AuthStore {
    open_as(dir, clock, Host::Server).expect("the store opens")
}

fn memory(clock: &Arc<ManualClock>) -> AuthStore {
    AuthStore::memory(clock.clone())
}

fn native_pat(scopes: &[&str]) -> MintSpec {
    MintSpec::new(Kind::Pat, "a tool", ScopeSet::of(scopes), 30 * DAY)
}

fn browser_pat(origin: &str, scopes: &[&str]) -> MintSpec {
    let mut s = MintSpec::new(Kind::Pat, "a web app", ScopeSet::of(scopes), 30 * DAY);
    s.origins = vec![origin.to_string()];
    s
}

fn session(app: App, preset: Preset) -> MintSpec {
    let mut s = MintSpec::new(Kind::Ses, "a browser", preset.scopes(), DAY);
    s.app = Some(app);
    s.preset = Some(preset);
    s
}

fn desk(app: App, preset: Preset, origins: &[&str]) -> MintSpec {
    let mut s = MintSpec::new(Kind::Dsk, "a webview", preset.scopes(), DAY);
    s.app = Some(app);
    s.preset = Some(preset);
    s.origins = origins.iter().map(|o| o.to_string()).collect();
    s
}

fn child(parent: &str, scopes: &[&str]) -> MintSpec {
    let mut s = MintSpec::new(Kind::Run, "a child", ScopeSet::of(scopes), HOUR);
    s.parent = Some(parent.to_string());
    s
}

/// A random source that hands out these bytes for the id and then for the secret, over and over.
fn fixed_random(id: [u8; 8], secret: [u8; 32]) -> Random {
    let calls = AtomicUsize::new(0);
    Arc::new(move |buf: &mut [u8]| {
        let n = calls.fetch_add(1, Ordering::SeqCst);
        if n % 2 == 0 {
            buf.copy_from_slice(&id);
        } else {
            buf.copy_from_slice(&secret);
        }
        Ok(())
    })
}

fn known_secret() -> [u8; 32] {
    let mut s = [0u8; 32];
    for (i, b) in s.iter_mut().enumerate() {
        *b = i as u8;
    }
    s
}

/// A writer that can be made to fail as a full disk does.
struct Toggle {
    fail: AtomicBool,
    writes: AtomicUsize,
}

impl Toggle {
    fn new() -> Arc<Toggle> {
        Arc::new(Toggle {
            fail: AtomicBool::new(false),
            writes: AtomicUsize::new(0),
        })
    }
}

impl FileWriter for Toggle {
    fn write(&self, path: &std::path::Path, bytes: &[u8]) -> io::Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        if self.fail.load(Ordering::SeqCst) {
            return Err(io::Error::from_raw_os_error(ENOSPC));
        }
        SecureWriter.write(path, bytes)
    }
}

/// A writer that only counts (for the limits, which need hundreds of writes).
struct Counting(AtomicUsize);

impl FileWriter for Counting {
    fn write(&self, _: &std::path::Path, _: &[u8]) -> io::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn read_file(dir: &TempDir) -> Value {
    serde_json::from_str(&std::fs::read_to_string(auth_dir(dir).join("credentials.json")).unwrap())
        .unwrap()
}

fn write_file(dir: &TempDir, doc: &Value) {
    std::fs::create_dir_all(auth_dir(dir)).unwrap();
    std::fs::write(
        auth_dir(dir).join("credentials.json"),
        serde_json::to_string_pretty(doc).unwrap(),
    )
    .unwrap();
}

/// A record as a file holds it, valid at `T0`.
fn file_record(id: &str, kind: &str) -> Value {
    json!({
        "id": id, "kind": kind, "hash": "a".repeat(64), "label": "from a file", "product": null, "preset": null, "preset_ver": null,
        "scopes": ["ai.read"], "origin": null, "app": null, "parent": null,
        "created_ms": T0 - HOUR, "expires_ms": T0 + DAY, "idle_ms": null, "last_used_ms": null, "last_used_ip": null,
        "revoked_ms": null, "legacy": false, "cnf": null, "max_uses": null, "uses": 0, "created_by": null
    })
}

// ================================ tokens, hashes, the known answer ================================

#[test]
fn a_minted_token_matches_the_known_answer_vector_and_only_its_hash_is_kept() {
    let clock = clock();
    let store = memory(&clock);
    store.set_random(fixed_random(ID_BYTES, known_secret()));
    let minted = store.mint(native_pat(&["ai.read"])).unwrap();
    assert_eq!(minted.token, KNOWN_TOKEN);
    assert_eq!(minted.id, "0123456789abcdef");
    let record = store.record(&minted.id).unwrap();
    assert_eq!(record.hash, KNOWN_HASH);
    let shown = format!("{record:?} {minted:?}");
    assert!(
        !shown.contains("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"),
        "no secret in a Debug print: {shown}"
    );
}

#[test]
fn the_secret_is_never_on_disk_only_its_hash() {
    let dir = TempDir::new("store-nosecret");
    let clock = clock();
    let store = open(&dir, &clock);
    store.set_random(fixed_random(ID_BYTES, known_secret()));
    store.mint(native_pat(&["ai.read"])).unwrap();
    let text = std::fs::read_to_string(auth_dir(&dir).join("credentials.json")).unwrap();
    assert!(text.contains(KNOWN_HASH));
    assert!(
        !text.contains("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"),
        "the secret must not be written"
    );
    assert!(!text.contains(KNOWN_TOKEN));
}

#[test]
fn a_minted_credential_authenticates_with_exactly_its_scopes_and_binding() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store
        .mint(browser_pat(
            "https://formlogic.example",
            &["ai.read", "ai.use"],
        ))
        .unwrap();
    let p = store
        .authenticate(&minted.token, Some("203.0.113.9"))
        .unwrap();
    assert_eq!(p.kind, PrincipalKind::Pat);
    assert_eq!(p.id, minted.id);
    assert_eq!(p.scopes, ScopeSet::of(&["ai.read", "ai.use"]));
    assert_eq!(p.origins, ["https://formlogic.example"]);
    assert!(
        !p.elevated && !p.persisted,
        "a memory-only store persists nothing"
    );
    assert_eq!(p.expires_ms, Some(T0 + 30 * DAY));
    let record = store.record(&minted.id).unwrap();
    assert_eq!(record.last_used_ms, Some(T0));
    assert_eq!(record.last_used_ip.as_deref(), Some("203.0.113.9"));
}

#[test]
fn an_unknown_id_and_a_wrong_secret_are_the_same_answer() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(native_pat(&["ai.read"])).unwrap();
    let parsed = token::parse(&minted.token).unwrap();
    // Right id, wrong secret; wrong id, right secret; nothing right; the wrong kind on a right id.
    let wrong_secret = format!("oaiypat_{}_{}", parsed.id, "A".repeat(43));
    let wrong_id = format!("oaiypat_{}_{}", "f".repeat(16), parsed.secret);
    let wrong_kind = format!("oaiyses_{}_{}", parsed.id, parsed.secret);
    let other_kind = format!("oaiyrun_{}_{}", parsed.id, parsed.secret);
    for bad in [
        wrong_secret,
        wrong_id,
        wrong_kind,
        other_kind,
        "oaiypat_short".into(),
        String::new(),
        "x".repeat(200),
        format!("{}A", minted.token),
    ] {
        assert_eq!(
            store.authenticate(&bad, None),
            Err(AuthError::Invalid),
            "{bad}"
        );
    }
    assert_eq!(AuthError::Invalid.code(), "token_invalid");
    assert_eq!(AuthError::Invalid.reason(), None);
    assert!(
        store.authenticate(&minted.token, None).is_ok(),
        "and the right one still works"
    );
}

#[test]
fn a_token_that_is_one_character_off_anywhere_is_refused() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(native_pat(&["ai.read"])).unwrap();
    for i in 0..minted.token.len() {
        let mut bytes = minted.token.clone().into_bytes();
        bytes[i] = if bytes[i] == b'x' { b'y' } else { b'x' };
        let bad = String::from_utf8(bytes).unwrap();
        assert!(store.authenticate(&bad, None).is_err(), "position {i}");
    }
}

// ======================================= life and death ===========================================

#[test]
fn a_credential_is_refused_from_the_instant_it_expires() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(native_pat(&["ai.read"])).unwrap();
    clock.set(T0 + 30 * DAY - 1);
    assert!(store.authenticate(&minted.token, None).is_ok());
    clock.set(T0 + 30 * DAY);
    let err = store.authenticate(&minted.token, None).unwrap_err();
    assert_eq!(err, AuthError::Expired);
    assert_eq!(err.code(), "token_expired");
}

#[test]
fn a_revoked_credential_is_refused_and_says_so() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(native_pat(&["ai.read"])).unwrap();
    assert!(store.revoke(&minted.id, "revoked"));
    let err = store.authenticate(&minted.token, None).unwrap_err();
    assert_eq!(err, AuthError::Revoked { reason: "revoked" });
    assert_eq!(err.code(), "token_revoked");
    assert_eq!(err.reason(), Some("revoked"));
    assert!(
        !store.revoke(&minted.id, "revoked"),
        "revoking twice changes nothing"
    );
    assert!(
        !store.revoke("ffffffffffffffff", "revoked"),
        "nor does revoking what is not there"
    );
}

#[test]
fn a_session_says_why_it_ended() {
    let clock = clock();
    let store = memory(&clock);
    for (reason, want) in [
        ("logged_out", "logged_out"),
        ("password_changed", "password_changed"),
        ("upgrade", "upgrade"),
        ("revoked", "revoked"),
        ("anything else", "revoked"),
    ] {
        let minted = store.mint(session(App::Dash, Preset::Owner)).unwrap();
        store.revoke(&minted.id, reason);
        let err = store.authenticate(&minted.token, None).unwrap_err();
        assert_eq!(err, AuthError::SessionEnded { reason: want }, "{reason}");
        assert_eq!(err.code(), "session_expired");
    }
}

#[test]
fn a_session_idles_out_and_using_it_keeps_it_alive() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    // Used after seven hours: not idle for the 8-hour timeout.
    clock.advance(7 * HOUR);
    assert!(store.authenticate(&minted.token, None).is_ok());
    // Exactly at the timeout is still inside it, and that use starts the timer again.
    clock.advance(8 * HOUR);
    assert!(store.authenticate(&minted.token, None).is_ok());
    // Unused for 8 hours and a moment: idle, and still inside the absolute life of 24 hours.
    clock.advance(8 * HOUR + 1);
    assert_eq!(
        store.authenticate(&minted.token, None),
        Err(AuthError::SessionEnded { reason: "idle" })
    );
}

#[test]
fn a_session_ends_at_its_absolute_time_however_busy_it_is() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    for _ in 0..47 {
        clock.advance(30 * MIN);
        assert!(store.authenticate(&minted.token, None).is_ok());
    }
    clock.advance(30 * MIN);
    assert_eq!(
        store.authenticate(&minted.token, None),
        Err(AuthError::SessionEnded { reason: "absolute" })
    );
}

#[test]
fn a_session_older_than_the_owners_epoch_is_ended_with_the_reason_upgrade() {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    store.set_min_session_epoch(1);
    assert_eq!(
        store.authenticate(&minted.token, None),
        Err(AuthError::SessionEnded { reason: "upgrade" })
    );
    // A session made after the epoch moved is fine.
    let later = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    assert!(store.authenticate(&later.token, None).is_ok());
    // And a paired token is not a session.
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    assert!(store.authenticate(&pat.token, None).is_ok());
}

#[test]
fn a_clock_that_jumps_forward_expires_things_and_one_that_goes_back_does_not_make_a_session_immortal(
) {
    let clock = clock();
    let store = memory(&clock);
    let minted = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    // The clock jumps forward 100 hours, and the session is used during the jump...
    clock.advance(100 * HOUR);
    assert!(
        store.authenticate(&minted.token, None).is_err(),
        "past its absolute life"
    );
    // ...a fresh one used at the jumped time has a last use in the future once the clock returns.
    clock.set(T0);
    let fresh = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    clock.set(T0 + 50 * HOUR);
    let again = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    store.authenticate(&again.token, None).unwrap();
    clock.set(T0 + HOUR);
    store.maintain();
    assert_eq!(
        store.record(&again.id).unwrap().last_used_ms,
        Some(T0 + HOUR),
        "the future use is set to now"
    );
    // Now it idles out from there, not 50 hours later.
    clock.set(T0 + HOUR + 9 * HOUR);
    assert_eq!(
        store.authenticate(&again.token, None),
        Err(AuthError::SessionEnded { reason: "idle" })
    );
    let _ = fresh;
}

// ========================================== the chain ============================================

#[test]
fn a_child_is_valid_only_while_its_whole_parent_chain_is() {
    let clock = clock();
    let store = memory(&clock);
    let grand = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    let mut spec = session(App::Agent, Preset::Agent);
    spec.parent = Some(grand.id.clone());
    let parent = store.mint(spec).unwrap();
    let mut spec = MintSpec::new(Kind::Run, "derived", ScopeSet::of(&["ai.read"]), HOUR);
    spec.parent = Some(parent.id.clone());
    let leaf = store.mint(spec).unwrap();
    assert!(store.authenticate(&leaf.token, None).is_ok());
    // T36: the grandparent is revoked (a logout): the parent and the leaf are dead at the next use.
    store.revoke(&grand.id, "logged_out");
    assert_eq!(
        store.authenticate(&parent.token, None),
        Err(AuthError::SessionEnded {
            reason: "parent_ended"
        })
    );
    let err = store.authenticate(&leaf.token, None).unwrap_err();
    assert_eq!(
        err,
        AuthError::Revoked {
            reason: "parent_ended"
        }
    );
    assert_eq!(err.code(), "token_revoked");
}

#[test]
fn a_parent_that_expires_or_idles_out_ends_its_children() {
    let clock = clock();
    let store = memory(&clock);
    let parent = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    let mut spec = session(App::Agent, Preset::Agent);
    spec.parent = Some(parent.id.clone());
    // The child would idle out only after 30 hours, so it is the parent that ends it.
    spec.idle_ms = Some(30 * HOUR);
    spec.ttl_ms = 40 * HOUR;
    let kid = store.mint(spec).unwrap();
    // The child is used, the parent is not: using a child keeps the parent alive.
    clock.advance(7 * HOUR);
    assert!(store.authenticate(&kid.token, None).is_ok());
    clock.advance(7 * HOUR);
    assert!(
        store.authenticate(&kid.token, None).is_ok(),
        "the parent's idle timer was refreshed by its child"
    );
    assert!(store.authenticate(&parent.token, None).is_ok());
    // Nobody uses either for nine hours: the parent idles out and the child goes with it.
    clock.advance(9 * HOUR);
    assert_eq!(
        store.authenticate(&kid.token, None),
        Err(AuthError::SessionEnded {
            reason: "parent_ended"
        })
    );
}

#[test]
fn a_child_that_outlives_its_parent_record_is_dead() {
    let clock = clock();
    let store = memory(&clock);
    let parent = store.mint(native_pat(&["ai.read", "ai.use"])).unwrap();
    let kid = store.mint(child(&parent.id, &["ai.read"])).unwrap();
    assert!(store.authenticate(&kid.token, None).is_ok());
    // The parent's record goes (purged long after it expired): the child has no parent at all.
    clock.set(T0 + 30 * DAY + 8 * DAY);
    store.maintain();
    assert!(store.record(&parent.id).is_none());
    assert!(store.authenticate(&kid.token, None).is_err());
}

#[test]
fn a_child_cannot_be_made_from_a_parent_that_is_not_valid() {
    let clock = clock();
    let store = memory(&clock);
    let parent = store.mint(native_pat(&["ai.read"])).unwrap();
    store.revoke(&parent.id, "revoked");
    assert!(matches!(
        store.mint(child(&parent.id, &["ai.read"])),
        Err(MintFailure::ParentInvalid)
    ));
    assert!(matches!(
        store.mint(child("ffffffffffffffff", &["ai.read"])),
        Err(MintFailure::ParentInvalid)
    ));
    // The static token as a parent needs the static token to be configured.
    assert!(matches!(
        store.mint(child(STATIC_PARENT, &["ai.read"])),
        Err(MintFailure::ParentInvalid)
    ));
    store.set_static_present(true);
    let kid = store.mint(child(STATIC_PARENT, &["ai.read"])).unwrap();
    assert!(store.authenticate(&kid.token, None).is_ok());
    store.set_static_present(false);
    assert!(
        store.authenticate(&kid.token, None).is_err(),
        "the static token went away"
    );
}

// ============================================ limits =============================================

#[test]
fn at_most_512_persisted_credentials() {
    let dir = TempDir::new("store-512");
    let clock = clock();
    let store = AuthStore::open(
        &auth_dir(&dir),
        Host::Server,
        clock.clone(),
        Arc::new(Counting(AtomicUsize::new(0))),
        None,
    )
    .unwrap();
    for i in 0..MAX_PERSISTED {
        store
            .mint(native_pat(&["ai.read"]))
            .unwrap_or_else(|e| panic!("#{i}: {e}"));
    }
    assert!(matches!(
        store.mint(native_pat(&["ai.read"])),
        Err(MintFailure::TooManyCredentials)
    ));
    // A memory-only credential is not counted against it.
    let mut spec = desk(App::Agent, Preset::Agent, &["http://oaiy.localhost"]);
    spec.ttl_ms = HOUR;
    store.mint(spec).unwrap();
    // A revoked credential frees its slot.
    let any = store.live_count(Kind::Pat);
    assert_eq!(any, MAX_PERSISTED);
}

#[test]
fn a_revoked_or_expired_credential_frees_its_slot() {
    let clock = clock();
    let dir = TempDir::new("store-512-free");
    let store = AuthStore::open(
        &auth_dir(&dir),
        Host::Server,
        clock.clone(),
        Arc::new(Counting(AtomicUsize::new(0))),
        None,
    )
    .unwrap();
    let mut first = None;
    for _ in 0..MAX_PERSISTED {
        let m = store.mint(native_pat(&["ai.read"])).unwrap();
        first.get_or_insert(m.id);
    }
    assert!(store.mint(native_pat(&["ai.read"])).is_err());
    store.revoke(first.as_deref().unwrap(), "revoked");
    store
        .mint(native_pat(&["ai.read"]))
        .expect("one revoked, one free");
}

#[test]
fn at_most_64_live_children_per_parent() {
    let clock = clock();
    let store = memory(&clock);
    let parent = store.mint(native_pat(&["ai.read", "ai.use"])).unwrap();
    let mut kids = Vec::new();
    for i in 0..MAX_CHILDREN {
        kids.push(
            store
                .mint(child(&parent.id, &["ai.read"]))
                .unwrap_or_else(|e| panic!("#{i}: {e}")),
        );
    }
    assert!(matches!(
        store.mint(child(&parent.id, &["ai.read"])),
        Err(MintFailure::TooManyChildren)
    ));
    // Another parent has its own 64.
    let other = store.mint(native_pat(&["ai.read"])).unwrap();
    store.mint(child(&other.id, &["ai.read"])).unwrap();
    // A child that ends frees a place.
    store.revoke(&kids[0].id, "revoked");
    store.mint(child(&parent.id, &["ai.read"])).unwrap();
}

#[test]
fn at_most_32_live_sessions_and_the_one_used_longest_ago_makes_room() {
    let clock = clock();
    let store = memory(&clock);
    let mut sessions = Vec::new();
    for _ in 0..MAX_SESSIONS {
        sessions.push(store.mint(session(App::Dash, Preset::Owner)).unwrap());
        clock.advance(1_000);
    }
    // Use the first one, so that the second is the one used longest ago.
    store.authenticate(&sessions[0].token, None).unwrap();
    clock.advance(1_000);
    let newest = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    assert!(store.authenticate(&newest.token, None).is_ok());
    assert!(
        store.authenticate(&sessions[0].token, None).is_ok(),
        "the used one stays"
    );
    assert_eq!(
        store.authenticate(&sessions[1].token, None),
        Err(AuthError::SessionEnded { reason: "revoked" }),
        "the idle one made room"
    );
    assert!(store.authenticate(&sessions[2].token, None).is_ok());
    assert_eq!(store.live_count(Kind::Ses), MAX_SESSIONS);
}

#[test]
fn one_child_session_per_parent_and_app_a_new_handoff_revokes_the_previous() {
    let clock = clock();
    let store = memory(&clock);
    let parent = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    let handoff = |app: App, preset: Preset| {
        let mut spec = session(app, preset);
        spec.parent = Some(parent.id.clone());
        store.mint(spec).unwrap()
    };
    let agent1 = handoff(App::Agent, Preset::Agent);
    let flows1 = handoff(App::Flows, Preset::FlowsHost);
    let agent2 = handoff(App::Agent, Preset::Agent);
    assert!(
        store.authenticate(&agent1.token, None).is_err(),
        "rotated out"
    );
    assert!(store.authenticate(&agent2.token, None).is_ok());
    assert!(
        store.authenticate(&flows1.token, None).is_ok(),
        "another app's child is untouched"
    );
    assert_eq!(store.live_count(Kind::Ses), 3);
}

// ==================================== what a kind may carry ======================================

fn refused(store: &AuthStore, spec: MintSpec) -> String {
    match store.mint(spec) {
        Err(e) => e.to_string(),
        Ok(m) => panic!("was made: {m:?}"),
    }
}

#[test]
fn a_browser_bound_token_never_holds_a_dangerous_scope() {
    let clock = clock();
    let store = memory(&clock);
    for s in [
        "services.define",
        "runtimes.install",
        "plugins.install",
        "ai.admin",
        "flows.approve",
        "auth.manage",
        "system.update",
    ] {
        let why = refused(&store, browser_pat("https://app.example", &["ai.read", s]));
        assert!(why.contains(s), "{s}: {why}");
    }
}

#[test]
fn a_native_token_holds_at_most_the_five_dangerous_scopes_and_then_for_a_day() {
    let clock = clock();
    let store = memory(&clock);
    for s in [
        "services.define",
        "runtimes.install",
        "plugins.install",
        "ai.admin",
        "flows.approve",
    ] {
        let mut spec = native_pat(&["ai.read", s]);
        spec.ttl_ms = DAY;
        store.mint(spec).unwrap_or_else(|e| panic!("{s}: {e}"));
        let mut long = native_pat(&["ai.read", s]);
        long.ttl_ms = DAY + 1;
        assert!(
            matches!(store.mint(long), Err(MintFailure::TtlTooLong { max_ms }) if max_ms == DAY),
            "{s}"
        );
    }
    // The nine that no token can ever hold.
    for s in scopes::NEVER_ON_A_TOKEN {
        let mut spec = native_pat(&["ai.read", s]);
        spec.ttl_ms = HOUR;
        assert!(
            matches!(store.mint(spec), Err(MintFailure::ScopeNotGrantable(_))),
            "{s}"
        );
    }
}

#[test]
fn token_lifetimes_are_capped_by_kind() {
    let clock = clock();
    let store = memory(&clock);
    let mut ok = browser_pat("https://app.example", &["ai.read"]);
    ok.ttl_ms = 90 * DAY;
    store.mint(ok).unwrap();
    let mut long = browser_pat("https://app.example", &["ai.read"]);
    long.ttl_ms = 90 * DAY + 1;
    assert!(
        matches!(store.mint(long), Err(MintFailure::TtlTooLong { max_ms }) if max_ms == 90 * DAY)
    );
    let mut native = native_pat(&["ai.read"]);
    native.ttl_ms = 365 * DAY;
    store.mint(native).unwrap();
    let mut too_long = native_pat(&["ai.read"]);
    too_long.ttl_ms = 365 * DAY + 1;
    assert!(matches!(
        store.mint(too_long),
        Err(MintFailure::TtlTooLong { .. })
    ));
    let mut run = MintSpec::new(Kind::Run, "r", ScopeSet::of(&["ai.read"]), DAY);
    store.mint(run.clone()).unwrap();
    run.ttl_ms = DAY + 1;
    assert!(matches!(
        store.mint(run),
        Err(MintFailure::TtlTooLong { .. })
    ));
    let mut none = native_pat(&["ai.read"]);
    none.ttl_ms = 0;
    assert!(matches!(store.mint(none), Err(MintFailure::Invalid(_))));
}

#[test]
fn the_vault_ceremony_token_is_one_scope_browser_bound_five_minutes_and_one_use() {
    let clock = clock();
    let store = memory(&clock);
    let ceremony = |origin: Option<&str>, scopes: &[&str], ttl: u64, uses: Option<u32>| {
        let mut s = MintSpec::new(Kind::Pat, "ceremony", ScopeSet::of(scopes), ttl);
        s.origins = origin.map(|o| vec![o.to_string()]).unwrap_or_default();
        s.max_uses = uses;
        s
    };
    let made = store
        .mint(ceremony(
            Some("https://app.example"),
            &["vault.kt"],
            5 * MIN,
            Some(1),
        ))
        .unwrap();
    assert!(store.authenticate(&made.token, None).is_ok());
    for bad in [
        ceremony(None, &["vault.kt"], 5 * MIN, Some(1)),
        ceremony(
            Some("https://app.example"),
            &["vault.kt", "ai.read"],
            5 * MIN,
            Some(1),
        ),
        ceremony(
            Some("https://app.example"),
            &["vault.kt"],
            5 * MIN + 1,
            Some(1),
        ),
        ceremony(Some("https://app.example"), &["vault.kt"], 5 * MIN, None),
        ceremony(Some("https://app.example"), &["vault.kt"], 5 * MIN, Some(2)),
        ceremony(
            Some("https://app.example"),
            &["vault.read"],
            5 * MIN,
            Some(1),
        ),
    ] {
        assert!(store.mint(bad.clone()).is_err(), "{bad:?}");
    }
    // One use, then it is spent.
    assert!(store.spend(&made.id));
    assert!(
        store.authenticate(&made.token, None).is_err(),
        "revoked once used"
    );
    assert!(!store.spend(&made.id), "and there is nothing left to spend");
}

#[test]
fn a_session_holds_no_more_than_its_hosts_ceiling() {
    let clock = clock();
    let store = memory(&clock);
    // The Agent host: the agent preset and nothing more.
    let mut wide = session(App::Agent, Preset::Owner);
    wide.scopes = Preset::Owner.scopes();
    assert!(matches!(
        store.mint(wide),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
    let mut dangerous = session(App::Flows, Preset::FlowsHost);
    dangerous.scopes.insert("services.define");
    assert!(matches!(
        store.mint(dangerous),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
    // The dashboard host: everything (dangerous scopes then need a step-up at use).
    store.mint(session(App::Dash, Preset::Owner)).unwrap();
    // A session needs an app, and is not bound to an origin.
    let mut no_app = session(App::Dash, Preset::Owner);
    no_app.app = None;
    assert!(matches!(store.mint(no_app), Err(MintFailure::Invalid(_))));
    let mut origin = session(App::Dash, Preset::Owner);
    origin.origins = vec!["https://x.example".into()];
    assert!(matches!(store.mint(origin), Err(MintFailure::Invalid(_))));
    let mut long = session(App::Dash, Preset::Owner);
    long.ttl_ms = 30 * DAY + 1;
    assert!(matches!(
        store.mint(long),
        Err(MintFailure::TtlTooLong { .. })
    ));
}

#[test]
fn a_desk_credential_is_bound_to_its_webviews_origins_and_capped_by_its_role() {
    let clock = clock();
    let store = memory(&clock);
    let dash = store
        .mint(desk(
            App::Dash,
            Preset::Owner,
            &[
                "tauri://localhost",
                "http://tauri.localhost",
                "https://tauri.localhost",
            ],
        ))
        .unwrap();
    let p = store.authenticate(&dash.token, None).unwrap();
    assert_eq!(p.kind, PrincipalKind::Desk);
    assert_eq!(
        p.origins,
        [
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost"
        ]
    );
    assert!(p.elevated, "the dashboard's desk is always elevated");
    assert!(p.has("plugins.install") && p.has("flows.approve"));
    let agent = store
        .mint(desk(App::Agent, Preset::Agent, &["http://oaiy.localhost"]))
        .unwrap();
    let p = store.authenticate(&agent.token, None).unwrap();
    assert!(!p.elevated && !p.scopes.has_dangerous());
    assert!(
        matches!(
            store.mint(desk(App::Agent, Preset::Agent, &[])),
            Err(MintFailure::Invalid(_))
        ),
        "no origin, no desk credential"
    );
    let mut wide = desk(App::Flows, Preset::Flows, &["http://oaiyflows.localhost"]);
    wide.scopes = Preset::Owner.scopes();
    assert!(matches!(
        store.mint(wide),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
    // The flow editor's window holds the wider `flows` bundle, its host build the narrower one.
    let flows = store
        .mint(desk(
            App::Flows,
            Preset::Flows,
            &["http://oaiyflows.localhost"],
        ))
        .unwrap();
    assert!(store
        .authenticate(&flows.token, None)
        .unwrap()
        .has("connectors.use"));
    assert!(
        matches!(
            store.mint(session(App::Flows, Preset::Flows)),
            Err(MintFailure::ScopeNotGrantable(_))
        ),
        "a session on the flow host holds flows-host at most"
    );
}

#[test]
fn a_per_run_or_derived_credential_holds_no_dangerous_auth_or_reserved_scope() {
    let clock = clock();
    let store = memory(&clock);
    for s in [
        "services.define",
        "auth.read",
        "auth.revoke",
        "auth.manage",
        "vault.kt",
        "relay.read",
    ] {
        let mut spec = MintSpec::new(Kind::Run, "r", ScopeSet::of(&["ai.read", s]), HOUR);
        spec.parent = None;
        assert!(
            matches!(store.mint(spec), Err(MintFailure::ScopeNotGrantable(_))),
            "{s}"
        );
    }
}

#[test]
fn control_project_always_travels_with_control_read() {
    let clock = clock();
    let store = memory(&clock);
    assert!(matches!(
        store.mint(native_pat(&["control.project"])),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
    store
        .mint(native_pat(&["control.project", "control.read"]))
        .unwrap();
    store.mint(native_pat(&["control.read"])).unwrap();
}

#[test]
fn only_a_paired_token_holds_a_connector_scope_and_a_name_that_is_no_scope_is_refused() {
    let clock = clock();
    let store = memory(&clock);
    let mut pat = native_pat(&["connectors.use"]);
    pat.scopes.insert("connector.aokie.call.answer");
    let made = store.mint(pat).unwrap();
    assert!(store
        .authenticate(&made.token, None)
        .unwrap()
        .has("connector.aokie.call.answer"));
    let mut run = MintSpec::new(Kind::Run, "r", ScopeSet::of(&["connectors.use"]), HOUR);
    run.scopes.insert("connector.aokie.call.answer");
    assert!(matches!(
        store.mint(run),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
    let mut bogus = native_pat(&["ai.read"]);
    bogus.scopes = ScopeSet::parse_lenient(["ai.read", "ai.*"]);
    assert!(matches!(
        store.mint(bogus),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
    let mut bogus = native_pat(&["ai.read"]);
    bogus.scopes = ScopeSet::parse_lenient(["connector.Aokie.x"]);
    assert!(matches!(
        store.mint(bogus),
        Err(MintFailure::ScopeNotGrantable(_))
    ));
}

#[test]
fn an_origin_is_exact_lowercase_and_never_null() {
    let clock = clock();
    let store = memory(&clock);
    for bad in [
        "null",
        "NULL",
        "",
        " ",
        "https://APP.example",
        "https://app.example/",
        "https://app.example/path",
        "https://app.example?x=1",
        "https://user@app.example",
        "app.example",
        "ftp://app.example",
        "http://",
        "https://app.example#f",
        "https://ap p.example",
    ] {
        let why = refused(&store, browser_pat(bad, &["ai.read"]));
        assert!(why.contains("origin"), "{bad:?}: {why}");
    }
    for ok in [
        "https://app.example",
        "http://localhost:3000",
        "tauri://localhost",
        "http://oaiy.localhost",
        "https://formlogic.example:8443",
    ] {
        store
            .mint(browser_pat(ok, &["ai.read"]))
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    assert_eq!(
        canonical_origin("HTTPS://App.Example"),
        Some("https://app.example".into())
    );
    assert_eq!(canonical_origin("null"), None);
    let mut two = browser_pat("https://a.example", &["ai.read"]);
    two.origins.push("https://b.example".into());
    assert!(
        matches!(store.mint(two), Err(MintFailure::Invalid(_))),
        "a paired token has one origin"
    );
}

#[test]
fn the_device_cookie_is_not_made_here() {
    let clock = clock();
    let store = memory(&clock);
    assert!(matches!(
        store.mint(MintSpec::new(Kind::Dev, "d", ScopeSet::empty(), HOUR)),
        Err(MintFailure::Token(MintError::KindNotMintable(Kind::Dev)))
    ));
}

#[test]
fn without_randomness_no_credential_is_made_and_nothing_is_kept() {
    let dir = TempDir::new("store-norandom");
    let clock = clock();
    let store = open(&dir, &clock);
    store.set_random(Arc::new(|_: &mut [u8]| Err(MintError::NoRandomness)));
    assert!(matches!(
        store.mint(native_pat(&["ai.read"])),
        Err(MintFailure::Token(MintError::NoRandomness))
    ));
    assert_eq!(store.live_count(Kind::Pat), 0);
    assert!(
        !auth_dir(&dir).join("credentials.json").exists(),
        "and nothing was written"
    );
}

#[test]
fn a_repeated_id_is_not_used_twice() {
    let clock = clock();
    let store = memory(&clock);
    store.set_random(fixed_random(ID_BYTES, known_secret()));
    store.mint(native_pat(&["ai.read"])).unwrap();
    // The source keeps returning the same id: minting gives up rather than replacing the first.
    assert!(store.mint(native_pat(&["ai.read"])).is_err());
    assert_eq!(store.live_count(Kind::Pat), 1);
}

// ===================================== effective scopes ==========================================

#[test]
fn a_sessions_scopes_are_recomputed_from_the_running_tables_not_frozen_at_mint() {
    // T46: a live session across a preset change. The file holds a session made by an older release
    // whose `agent` preset lacked scopes the current one has, and which held one it no longer does.
    let dir = TempDir::new("store-recompute");
    let clock = clock();
    let mut rec = file_record("0123456789abcdef", "ses");
    rec["app"] = json!("agent");
    rec["preset"] = json!("agent");
    rec["scopes"] = json!(["plugins.read", "agent.tasks", "plugins.install"]);
    rec["idle_ms"] = json!(8 * HOUR);
    rec["hash"] = json!(KNOWN_HASH);
    write_file(&dir, &json!({ "v": 1, "credentials": [rec] }));
    let store = open(&dir, &clock);
    let p = store
        .authenticate(
            &format!(
                "oaiyses_0123456789abcdef_{}",
                "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
            ),
            None,
        )
        .unwrap();
    // The agent preset of this build, exactly: new scopes appear; nothing outside the ceiling stays.
    assert_eq!(p.scopes, Preset::Agent.scopes());
    assert!(
        p.has("runs.write") && p.has("contacts.write"),
        "scopes added since the session was made reach it at once"
    );
    assert!(
        !p.has("agent.tasks"),
        "a scope the preset no longer has is gone at once"
    );
    assert!(!p.has("plugins.install"));
}

#[test]
fn a_release_never_adds_a_dangerous_scope_to_a_live_session() {
    let dir = TempDir::new("store-nodanger");
    let clock = clock();
    let mut rec = file_record("0123456789abcdef", "ses");
    rec["app"] = json!("dash");
    rec["preset"] = json!("owner");
    // Made when `owner` had fewer dangerous scopes: it holds one of them and nothing else dangerous.
    rec["scopes"] = json!(["system.read", "services.define"]);
    rec["idle_ms"] = json!(8 * HOUR);
    rec["hash"] = json!(KNOWN_HASH);
    write_file(&dir, &json!({ "v": 1, "credentials": [rec] }));
    let store = open(&dir, &clock);
    let p = store
        .authenticate(KNOWN_TOKEN.replace("oaiypat", "oaiyses").as_str(), None)
        .unwrap();
    assert!(
        p.has("services.define"),
        "the dangerous scope it was made with stays"
    );
    assert!(
        !p.has("plugins.install") && !p.has("flows.approve"),
        "no dangerous scope it was not made with is added"
    );
    assert!(
        p.has("runs.write"),
        "the ordinary scopes of the owner preset are recomputed"
    );
    assert!(!p.elevated, "and a session is elevated only by a step-up");
}

#[test]
fn a_session_with_a_preset_this_build_does_not_know_keeps_what_it_had_and_nothing_more() {
    let dir = TempDir::new("store-unknown-preset");
    let clock = clock();
    let mut rec = file_record("0123456789abcdef", "ses");
    rec["app"] = json!("dash");
    rec["preset"] = json!("from-the-future");
    rec["scopes"] = json!(["system.read", "future.scope"]);
    rec["idle_ms"] = json!(8 * HOUR);
    rec["hash"] = json!(KNOWN_HASH);
    write_file(&dir, &json!({ "v": 1, "credentials": [rec] }));
    let store = open(&dir, &clock);
    let p = store
        .authenticate(KNOWN_TOKEN.replace("oaiypat", "oaiyses").as_str(), None)
        .unwrap();
    assert_eq!(p.scopes.core_names(), ["system.read"]);
    assert!(
        !p.has("future.scope") || !crate::auth::scopes::is_known("future.scope"),
        "an unknown scope matches no route"
    );
}

#[test]
fn a_paired_tokens_scopes_are_exactly_as_approved_and_an_edited_file_cannot_widen_them() {
    let dir = TempDir::new("store-pat-clamp");
    let clock = clock();
    let mut rec = file_record("0123456789abcdef", "pat");
    rec["hash"] = json!(KNOWN_HASH);
    // Someone edited the file: dangerous scopes on a browser-bound token, and one no token can hold.
    rec["origin"] = json!("https://app.example");
    rec["scopes"] = json!([
        "ai.read",
        "auth.manage",
        "services.define",
        "vault.admin",
        "relay.read",
        "connector.aokie.call.answer"
    ]);
    write_file(&dir, &json!({ "v": 1, "credentials": [rec] }));
    let store = open(&dir, &clock);
    let p = store.authenticate(KNOWN_TOKEN, None).unwrap();
    assert_eq!(p.scopes.names(), ["ai.read", "connector.aokie.call.answer"]);
    // A native token keeps the five it may hold and loses the nine it never may.
    let mut native = file_record("fedcba9876543210", "pat");
    native["hash"] = json!(token::secret_hash(&"B".repeat(43)));
    native["scopes"] = json!(["ai.read", "auth.manage", "services.define", "system.update"]);
    write_file(&dir, &json!({ "v": 1, "credentials": [native] }));
    drop(store);
    let store = open(&dir, &clock);
    let p = store
        .authenticate(
            &format!("oaiypat_fedcba9876543210_{}", "B".repeat(43)),
            None,
        )
        .unwrap();
    assert_eq!(p.scopes.names(), ["ai.read", "services.define"]);
}

// =============================================== derive ==========================================

fn principal_of(kind: PrincipalKind, id: &str, scopes: ScopeSet) -> Principal {
    Principal {
        id: id.into(),
        kind,
        label: "parent".into(),
        scopes,
        origins: Vec::new(),
        app: None,
        elevated: false,
        chain: Vec::new(),
        persisted: false,
        expires_ms: None,
        preset: None,
        legacy_import: false,
    }
}

fn request(scopes: &[&str]) -> DeriveRequest {
    DeriveRequest {
        scopes: ScopeSet::of(scopes),
        ttl_ms: None,
        label: "chatgpt".into(),
    }
}

#[test]
fn a_desk_pat_or_static_credential_can_derive_a_child_of_no_more_than_it_holds() {
    let clock = clock();
    let store = memory(&clock);
    store.set_static_present(true);
    let desk_made = store
        .mint(desk(App::Agent, Preset::Agent, &["http://oaiy.localhost"]))
        .unwrap();
    let desk_p = store.authenticate(&desk_made.token, None).unwrap();
    let pat_made = store
        .mint(native_pat(&[
            "ai.read",
            "ai.use",
            "control.read",
            "control.project",
        ]))
        .unwrap();
    let pat_p = store.authenticate(&pat_made.token, None).unwrap();
    let static_p = Principal::static_token();
    for parent in [&desk_p, &pat_p, &static_p] {
        let want: &[&str] = if parent.has("ai.use") {
            &["ai.read", "ai.use"]
        } else {
            &["runs.read"]
        };
        let minted = store
            .derive(parent, request(want))
            .unwrap_or_else(|e| panic!("{:?}: {e}", parent.kind));
        let child = store.authenticate(&minted.token, None).unwrap();
        assert_eq!(child.kind, PrincipalKind::Run);
        assert!(child.scopes.is_subset_of(&parent.scopes));
        assert_eq!(
            child.origins, parent.origins,
            "a child keeps its parent's origin binding"
        );
        assert_eq!(child.chain, std::slice::from_ref(&parent.id));
        assert!(!child.persisted);
    }
    // Asking for a scope the parent does not hold.
    assert!(matches!(
        store.derive(&pat_p, request(&["ai.read", "flows.write"])),
        Err(DeriveError::Refused(_))
    ));
    assert!(matches!(
        store.derive(&static_p, request(&["ai.admin"])),
        Err(DeriveError::Refused(_))
    ));
}

#[test]
fn a_session_cannot_derive_and_neither_can_a_derived_credential() {
    // A portable bearer minted from a session would undo HttpOnly.
    let clock = clock();
    let store = memory(&clock);
    let ses = store.mint(session(App::Agent, Preset::Agent)).unwrap();
    let ses_p = store.authenticate(&ses.token, None).unwrap();
    assert!(matches!(
        store.derive(&ses_p, request(&["ai.read"])),
        Err(DeriveError::Refused(_))
    ));
    store.set_static_present(true);
    let derived = store
        .derive(&Principal::static_token(), request(&["ai.read"]))
        .unwrap();
    let derived_p = store.authenticate(&derived.token, None).unwrap();
    assert!(matches!(
        store.derive(&derived_p, request(&["ai.read"])),
        Err(DeriveError::Refused(_))
    ));
    for kind in [
        PrincipalKind::Console,
        PrincipalKind::Legacy,
        PrincipalKind::Session,
        PrincipalKind::Run,
    ] {
        let p = principal_of(kind, "x", Preset::Owner.scopes());
        assert!(
            matches!(
                store.derive(&p, request(&["ai.read"])),
                Err(DeriveError::Refused(_))
            ),
            "{kind:?}"
        );
    }
}

#[test]
fn a_derived_credential_holds_no_dangerous_scope_and_no_auth_scope_even_if_the_parent_does() {
    let clock = clock();
    let store = memory(&clock);
    let dash = store
        .mint(desk(App::Dash, Preset::Owner, &["tauri://localhost"]))
        .unwrap();
    let dash_p = store.authenticate(&dash.token, None).unwrap();
    assert!(dash_p.has("auth.manage") && dash_p.has("auth.read") && dash_p.has("services.define"));
    for s in [
        "auth.read",
        "auth.revoke",
        "auth.manage",
        "services.define",
        "plugins.install",
        "flows.approve",
        "secrets.write",
        "system.restart",
        "vault.kt",
        "relay.manage",
    ] {
        assert!(
            matches!(
                store.derive(&dash_p, request(&["ai.read", s])),
                Err(DeriveError::Refused(_))
            ),
            "{s}"
        );
    }
    assert!(matches!(
        store.derive(
            &dash_p,
            DeriveRequest {
                scopes: ScopeSet::empty(),
                ttl_ms: None,
                label: "x".into()
            }
        ),
        Err(DeriveError::Mint(MintFailure::Invalid(_)))
    ));
    // A connector scope is not for a derived credential either.
    let mut with_connector = ScopeSet::of(&["ai.read"]);
    with_connector.insert("connector.aokie.call.answer");
    let mut parent = dash_p.clone();
    parent.scopes.insert("connector.aokie.call.answer");
    assert!(matches!(
        store.derive(
            &parent,
            DeriveRequest {
                scopes: with_connector,
                ttl_ms: None,
                label: "x".into()
            }
        ),
        Err(DeriveError::Refused(_))
    ));
    // The change level without the read level cannot be derived.
    assert!(matches!(
        store.derive(&dash_p, request(&["control.project"])),
        Err(DeriveError::Mint(MintFailure::ScopeNotGrantable(_)))
    ));
    store
        .derive(&dash_p, request(&["control.read", "control.project"]))
        .unwrap();
}

#[test]
fn a_derived_credentials_life_is_an_hour_a_day_at_most_and_never_beyond_its_parents() {
    let clock = clock();
    let store = memory(&clock);
    store.set_static_present(true);
    let s = Principal::static_token();
    let default = store.derive(&s, request(&["ai.read"])).unwrap();
    assert_eq!(default.expires_ms, T0 + HOUR, "an hour by default");
    let asked = |ttl: u64| {
        store
            .derive(
                &s,
                DeriveRequest {
                    scopes: ScopeSet::of(&["ai.read"]),
                    ttl_ms: Some(ttl),
                    label: "x".into(),
                },
            )
            .unwrap()
            .expires_ms
            - T0
    };
    assert_eq!(asked(10 * MIN), 10 * MIN);
    assert_eq!(asked(DAY), DAY);
    assert_eq!(asked(1000 * DAY), DAY, "a day at the most");
    // A pat with three hours left cannot give a child more.
    let mut pat_spec = native_pat(&["ai.read"]);
    pat_spec.ttl_ms = 3 * HOUR;
    let pat = store.mint(pat_spec).unwrap();
    let pat_p = store.authenticate(&pat.token, None).unwrap();
    assert_eq!(
        store
            .derive(
                &pat_p,
                DeriveRequest {
                    scopes: ScopeSet::of(&["ai.read"]),
                    ttl_ms: Some(DAY),
                    label: "x".into()
                }
            )
            .unwrap()
            .expires_ms
            - T0,
        3 * HOUR
    );
    clock.advance(2 * HOUR + 30 * MIN);
    assert_eq!(
        store
            .derive(&pat_p, request(&["ai.read"]))
            .unwrap()
            .expires_ms
            - clock_now(&clock),
        30 * MIN,
        "only what is left"
    );
    // A parent with no life left gives nothing.
    clock.advance(HOUR);
    assert!(matches!(
        store.derive(&pat_p, request(&["ai.read"])),
        Err(DeriveError::Refused(_))
    ));
}

fn clock_now(c: &Arc<ManualClock>) -> u64 {
    use super::store::Clock as _;
    c.now_ms()
}

#[test]
fn a_derived_credential_dies_with_its_parent() {
    let clock = clock();
    let store = memory(&clock);
    let pat = store.mint(native_pat(&["ai.read", "ai.use"])).unwrap();
    let pat_p = store.authenticate(&pat.token, None).unwrap();
    let kid = store.derive(&pat_p, request(&["ai.read"])).unwrap();
    assert!(store.authenticate(&kid.token, None).is_ok());
    store.revoke(&pat.id, "revoked");
    assert_eq!(
        store.authenticate(&kid.token, None),
        Err(AuthError::Revoked {
            reason: "parent_ended"
        })
    );
}

#[test]
fn at_most_30_derived_credentials_a_minute_per_credential_and_64_alive() {
    let clock = clock();
    let store = memory(&clock);
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    let pat_p = store.authenticate(&pat.token, None).unwrap();
    for i in 0..DERIVE_PER_MINUTE {
        store
            .derive(&pat_p, request(&["ai.read"]))
            .unwrap_or_else(|e| panic!("#{i}: {e}"));
        clock.advance(100);
    }
    match store.derive(&pat_p, request(&["ai.read"])) {
        Err(DeriveError::RateLimited { retry_after_s }) => {
            assert!((1..=60).contains(&retry_after_s), "{retry_after_s}")
        }
        other => panic!("{other:?}"),
    }
    // Another credential has its own allowance.
    let other = store.mint(native_pat(&["ai.read"])).unwrap();
    let other_p = store.authenticate(&other.token, None).unwrap();
    store.derive(&other_p, request(&["ai.read"])).unwrap();
    // A minute on, the first may derive again (30 more, then 4 a minute after that): 64 are alive, and
    // the 65th is refused by the cap on live children, not by the rate.
    clock.advance(MIN);
    for _ in 0..DERIVE_PER_MINUTE {
        store.derive(&pat_p, request(&["ai.read"])).unwrap();
    }
    clock.advance(MIN);
    for _ in 0..(MAX_CHILDREN - 2 * DERIVE_PER_MINUTE) {
        store.derive(&pat_p, request(&["ai.read"])).unwrap();
    }
    assert!(matches!(
        store.derive(&pat_p, request(&["ai.read"])),
        Err(DeriveError::Mint(MintFailure::TooManyChildren))
    ));
}

/// A small xorshift, so that the property test is the same run every time.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_scopes(rng: &mut Rng, density: u64) -> ScopeSet {
    let mut set = ScopeSet::empty();
    for s in &scopes::SCOPES {
        if rng.below(100) < density {
            set.insert(s.name);
        }
    }
    if rng.below(10) == 0 {
        set.insert("connector.aokie.call.answer");
    }
    set
}

#[test]
fn a_derived_credential_never_exceeds_its_parent_in_100000_random_cases() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let kinds = [
        PrincipalKind::Desk,
        PrincipalKind::Pat,
        PrincipalKind::Static,
        PrincipalKind::Session,
        PrincipalKind::Run,
        PrincipalKind::Console,
        PrincipalKind::Legacy,
    ];
    let clock = clock();
    let (mut allowed, mut refused_count) = (0u32, 0u32);
    for i in 0..100_000u32 {
        let kind = kinds[rng.below(kinds.len() as u64) as usize];
        let density = 5 + rng.below(90);
        let parent_scopes = random_scopes(&mut rng, density);
        // Requests: what the parent could give (30%), a piece of the parent (30%), the parent and one
        // more (20%), and anything at all (20%), so that both outcomes are common.
        let request_scopes = match rng.below(10) {
            0..=2 => parent_scopes.filtered(|n| {
                !scopes::is_dangerous(n)
                    && !scopes::is_auth_scope(n)
                    && !scopes::is_reserved(n)
                    && !scopes::is_resource_scope(n)
            }),
            3..=5 => {
                let keep = 20 + rng.below(70);
                let mut pick = Rng(rng.next());
                parent_scopes.filtered(|_| pick.below(100) < keep)
            }
            6..=7 => {
                let mut s = parent_scopes.clone();
                s.insert(scopes::SCOPES[rng.below(scopes::SCOPES.len() as u64) as usize].name);
                s
            }
            _ => {
                let density = 1 + rng.below(60);
                random_scopes(&mut rng, density)
            }
        };
        let store = memory(&clock);
        store.set_static_present(true);
        let id = if kind == PrincipalKind::Static {
            STATIC_PARENT.to_string()
        } else {
            format!("{i:016x}")
        };
        // A parent that exists in the store, as a desk or paired credential would.
        if matches!(kind, PrincipalKind::Desk | PrincipalKind::Pat) {
            let mut rec = Record::blank(
                &id,
                if kind == PrincipalKind::Desk {
                    Kind::Dsk
                } else {
                    Kind::Pat
                },
            );
            rec.created_ms = T0 - HOUR;
            rec.expires_ms = T0 + 100 * DAY;
            store.insert_for_tests(rec);
        }
        let mut parent = principal_of(kind, &id, parent_scopes.clone());
        parent.expires_ms = Some(T0 + 100 * DAY);
        let req = DeriveRequest {
            scopes: request_scopes.clone(),
            ttl_ms: Some(rng.below(3 * DAY)),
            label: "p".into(),
        };
        let forbidden = request_scopes.names().iter().any(|n| {
            scopes::is_dangerous(n)
                || scopes::is_auth_scope(n)
                || scopes::is_reserved(n)
                || scopes::is_resource_scope(n)
        });
        let expect_ok = matches!(
            kind,
            PrincipalKind::Desk | PrincipalKind::Pat | PrincipalKind::Static
        ) && !request_scopes.is_empty()
            && request_scopes.is_subset_of(&parent_scopes)
            && !forbidden
            && (!request_scopes.contains("control.project")
                || request_scopes.contains("control.read"));
        match store.derive(&parent, req) {
            Ok(minted) => {
                allowed += 1;
                assert!(
                    expect_ok,
                    "case {i}: derived where the rules say no ({kind:?})"
                );
                assert!(
                    minted.scopes.is_subset_of(&parent_scopes),
                    "case {i}: the child holds more than its parent"
                );
                assert!(!minted.scopes.has_dangerous(), "case {i}");
                assert!(
                    !minted
                        .scopes
                        .names()
                        .iter()
                        .any(|n| scopes::is_auth_scope(n)),
                    "case {i}"
                );
                assert!(
                    minted.expires_ms > T0 && minted.expires_ms - T0 <= DAY,
                    "case {i}: life {}",
                    minted.expires_ms - T0
                );
            }
            Err(e) => {
                refused_count += 1;
                assert!(
                    !expect_ok,
                    "case {i}: refused ({e}) where the rules say yes ({kind:?})"
                );
            }
        }
    }
    // The generator reached both outcomes many times.
    assert!(
        allowed > 1_000 && refused_count > 1_000,
        "{allowed} allowed, {refused_count} refused"
    );
}

// ===================================== files, atomicity, modes ===================================

#[test]
fn the_file_is_written_atomically_and_holds_the_version_and_no_staging_file_is_left() {
    let dir = TempDir::new("store-atomic");
    let clock = clock();
    let store = open(&dir, &clock);
    store.mint(native_pat(&["ai.read"])).unwrap();
    store.mint(native_pat(&["ai.use"])).unwrap();
    let doc = read_file(&dir);
    assert_eq!(doc["v"], 1);
    assert_eq!(doc["credentials"].as_array().unwrap().len(), 2);
    let mut names: Vec<String> = std::fs::read_dir(auth_dir(&dir))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [".lock", ".lock.pid", "credentials.json"],
        "no staging file is left beside it"
    );
}

#[cfg(unix)]
#[test]
fn the_folder_and_the_files_are_private() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = TempDir::new("store-modes");
    let clock = clock();
    let store = open(&dir, &clock);
    store.mint(native_pat(&["ai.read"])).unwrap();
    let mode = |p: PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(auth_dir(&dir)), 0o700);
    assert_eq!(mode(auth_dir(&dir).join("credentials.json")), 0o600);
    assert_eq!(mode(auth_dir(&dir).join(".lock")), 0o600);
}

#[test]
fn a_restart_keeps_persisted_credentials_and_drops_memory_only_ones() {
    let dir = TempDir::new("store-restart");
    let clock = clock();
    let store = open(&dir, &clock);
    let pat = store
        .mint(browser_pat("https://app.example", &["ai.read"]))
        .unwrap();
    let ses = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    let desk_made = store
        .mint(desk(App::Agent, Preset::Agent, &["http://oaiy.localhost"]))
        .unwrap();
    store.set_static_present(true);
    let run = store.mint(child(STATIC_PARENT, &["ai.read"])).unwrap();
    let con = store
        .mint(MintSpec::new(Kind::Con, "console", ScopeSet::all(), DAY))
        .unwrap();
    store.flush().unwrap();
    drop(store);
    let store = open(&dir, &clock);
    assert!(store.authenticate(&pat.token, None).is_ok());
    assert!(store.authenticate(&ses.token, None).is_ok());
    for (name, t) in [
        ("desk", &desk_made.token),
        ("run", &run.token),
        ("con", &con.token),
    ] {
        assert_eq!(
            store.authenticate(t, None),
            Err(AuthError::Invalid),
            "a {name} credential is gone with the process"
        );
    }
    let text = std::fs::read_to_string(auth_dir(&dir).join("credentials.json")).unwrap();
    assert!(
        !text.contains(&desk_made.id) && !text.contains(&run.id) && !text.contains(&con.id),
        "memory-only kinds are never written"
    );
}

#[test]
fn unknown_fields_are_kept_and_written_back() {
    let dir = TempDir::new("store-extra");
    let clock = clock();
    let mut rec = file_record("0123456789abcdef", "pat");
    rec["hash"] = json!(KNOWN_HASH);
    rec["future_field"] = json!({ "nested": [1, 2, 3] });
    write_file(
        &dir,
        &json!({ "v": 1, "credentials": [rec], "top_level_extra": "kept", "another": { "a": 1 } }),
    );
    let store = open(&dir, &clock);
    store.mint(native_pat(&["ai.read"])).unwrap();
    store.flush().unwrap();
    let doc = read_file(&dir);
    assert_eq!(doc["top_level_extra"], "kept");
    assert_eq!(doc["another"], json!({ "a": 1 }));
    let old = doc["credentials"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == "0123456789abcdef")
        .unwrap();
    assert_eq!(old["future_field"], json!({ "nested": [1, 2, 3] }));
    assert_eq!(doc["v"], 1);
}

#[test]
fn a_file_with_an_unknown_version_stops_the_server_and_leaves_the_desktop_with_an_empty_store_it_never_writes(
) {
    let dir = TempDir::new("store-newer");
    let clock = clock();
    let newer = json!({ "v": 2, "credentials": [{ "something": "new" }], "extra": 1 });
    write_file(&dir, &newer);
    let before = std::fs::read(auth_dir(&dir).join("credentials.json")).unwrap();
    // The server refuses to start, naming the file and both versions, with the exit code the unit does not restart.
    match open_as(&dir, &clock, Host::Server) {
        Err(e @ StoreError::UnknownVersion { .. }) => {
            let msg = e.to_string();
            assert!(
                msg.contains("credentials.json")
                    && msg.contains("version 2")
                    && msg.contains("version 1"),
                "{msg}"
            );
            assert_eq!(e.exit_code(), 78);
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    // The desktop starts with an empty store, says so, and never writes.
    let store = open_as(&dir, &clock, Host::Gui).expect("the desktop starts");
    assert!(!store.is_persistent());
    assert_eq!(store.notices().len(), 1);
    assert!(
        store.notices()[0].contains("newer OAIY"),
        "{:?}",
        store.notices()
    );
    let made = store.mint(native_pat(&["ai.read"])).unwrap();
    assert!(
        !store.authenticate(&made.token, None).unwrap().persisted,
        "a pairing made now lives in memory only"
    );
    store.flush().unwrap();
    assert_eq!(
        std::fs::read(auth_dir(&dir).join("credentials.json")).unwrap(),
        before,
        "the newer file is untouched"
    );
}

#[test]
fn an_unparsable_credentials_file_is_moved_aside_and_the_store_starts_empty_and_says_so() {
    let dir = TempDir::new("store-corrupt");
    let clock = clock();
    std::fs::create_dir_all(auth_dir(&dir)).unwrap();
    std::fs::write(
        auth_dir(&dir).join("credentials.json"),
        b"{ this is not json",
    )
    .unwrap();
    let audit = Arc::new(AuditLog::open(&auth_dir(&dir), clock.clone(), false));
    let store = AuthStore::open(
        &auth_dir(&dir),
        Host::Server,
        clock.clone(),
        Arc::new(SecureWriter),
        Some(audit.clone()),
    )
    .unwrap();
    assert_eq!(store.live_count(Kind::Pat), 0);
    let aside = auth_dir(&dir).join(format!("credentials.json.corrupt-{T0}"));
    assert_eq!(
        std::fs::read(&aside).unwrap(),
        b"{ this is not json",
        "kept for the operator"
    );
    assert!(!store.notices().is_empty());
    let events = audit.read(LogFile::Audit, 10, None, None);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["event"], "credentials.corrupt");
    // A new credential writes a fresh file.
    store.mint(native_pat(&["ai.read"])).unwrap();
    assert_eq!(read_file(&dir)["credentials"].as_array().unwrap().len(), 1);
}

#[test]
fn a_file_that_is_json_but_not_a_credential_file_is_corrupt_too() {
    for (tag, text) in [
        ("array", "[]"),
        ("no-version", r#"{"credentials":[]}"#),
        ("string-version", r#"{"v":"1","credentials":[]}"#),
        ("no-credentials", r#"{"v":1}"#),
        ("bad-record", r#"{"v":1,"credentials":[{"id":5}]}"#),
        (
            "unknown-kind",
            r#"{"v":1,"credentials":[{"id":"a","kind":"tok","hash":"x"}]}"#,
        ),
    ] {
        let dir = TempDir::new(&format!("store-corrupt-{tag}"));
        let clock = clock();
        std::fs::create_dir_all(auth_dir(&dir)).unwrap();
        std::fs::write(auth_dir(&dir).join("credentials.json"), text).unwrap();
        let store = open(&dir, &clock);
        assert_eq!(store.live_count(Kind::Pat), 0, "{tag}");
        assert!(
            auth_dir(&dir)
                .join(format!("credentials.json.corrupt-{T0}"))
                .exists(),
            "{tag}"
        );
    }
}

#[test]
fn a_file_with_the_same_id_twice_is_corrupt_not_first_wins() {
    let dir = TempDir::new("store-dup");
    let clock = clock();
    write_file(
        &dir,
        &json!({ "v": 1, "credentials": [file_record("0123456789abcdef", "pat"), file_record("0123456789abcdef", "pat")] }),
    );
    let store = open(&dir, &clock);
    assert_eq!(store.live_count(Kind::Pat), 0);
}

#[test]
fn only_a_missing_file_means_nothing_is_there_yet_any_other_read_error_names_the_file() {
    // ENOENT: an empty store, no complaint, and no file made by opening.
    let dir = TempDir::new("store-enoent");
    let clock = clock();
    let store = open(&dir, &clock);
    assert_eq!(store.live_count(Kind::Pat), 0);
    assert!(store.owner().is_none());
    assert!(store.notices().is_empty());
    assert!(!auth_dir(&dir).join("credentials.json").exists());
    drop(store);
    // Not ENOENT (here: a folder where the file belongs): a fatal error naming the file, never "corrupt".
    let dir = TempDir::new("store-eacces");
    std::fs::create_dir_all(auth_dir(&dir).join("credentials.json")).unwrap();
    match open_as(&dir, &clock, Host::Server) {
        Err(e @ StoreError::Unreadable { .. }) => {
            assert!(e.to_string().contains("credentials.json"), "{e}");
            assert_eq!(e.exit_code(), 78);
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    assert!(
        !auth_dir(&dir)
            .join(format!("credentials.json.corrupt-{T0}"))
            .exists(),
        "an unreadable file is not moved aside as corrupt"
    );
    // The same for the desktop: it does not start with a store that could not be read.
    assert!(matches!(
        open_as(&dir, &clock, Host::Gui),
        Err(StoreError::Unreadable { .. })
    ));
    let dir = TempDir::new("store-owner-eacces");
    std::fs::create_dir_all(auth_dir(&dir).join("owner.json")).unwrap();
    assert!(matches!(
        open_as(&dir, &clock, Host::Server),
        Err(StoreError::Unreadable { .. })
    ));
}

#[test]
fn owner_json_missing_means_no_owner_and_a_mangled_one_is_an_error_that_does_not_reopen_setup() {
    let clock = clock();
    let dir = TempDir::new("store-owner");
    std::fs::create_dir_all(auth_dir(&dir)).unwrap();
    std::fs::write(auth_dir(&dir).join("owner.json"), r#"{"v":1,"created_ms":1,"min_session_epoch":3,"password":"$argon2id$x","factors":["password"]}"#).unwrap();
    let store = open(&dir, &clock);
    let owner = store.owner().expect("an owner");
    assert_eq!(owner.min_session_epoch, 3);
    assert_eq!(
        owner.doc["factors"],
        json!(["password"]),
        "the rest of the file is kept as read"
    );
    assert_eq!(store.min_session_epoch(), 3);
    drop(store);
    for (tag, text) in [
        ("garbage", "not json"),
        ("no-version", r#"{"password":"x"}"#),
        ("empty", ""),
    ] {
        std::fs::write(auth_dir(&dir).join("owner.json"), text).unwrap();
        match open_as(&dir, &clock, Host::Server) {
            Err(e @ StoreError::OwnerUnparsable { .. }) => {
                assert!(e.to_string().contains("owner.json"), "{tag}: {e}");
                assert_eq!(e.exit_code(), 78);
            }
            other => panic!("{tag}: {:?}", other.map(|_| ())),
        }
        assert!(
            matches!(
                open_as(&dir, &clock, Host::Gui),
                Err(StoreError::OwnerUnparsable { .. })
            ),
            "{tag}: the desktop does not shrug it off either"
        );
        assert_eq!(
            std::fs::read_to_string(auth_dir(&dir).join("owner.json")).unwrap(),
            text,
            "{tag}: the file is left alone"
        );
    }
    std::fs::write(auth_dir(&dir).join("owner.json"), r#"{"v":9}"#).unwrap();
    assert!(matches!(
        open_as(&dir, &clock, Host::Server),
        Err(StoreError::UnknownVersion { .. })
    ));
}

#[test]
fn a_second_process_on_the_data_folder_is_refused() {
    let dir = TempDir::new("store-lock");
    let clock = clock();
    let first = open(&dir, &clock);
    match open_as(&dir, &clock, Host::Server) {
        Err(StoreError::Lock(e)) => assert!(e.to_string().contains("in use by process"), "{e}"),
        other => panic!("{:?}", other.map(|_| ())),
    }
    drop(first);
    open(&dir, &clock);
}

#[test]
fn what_expired_more_than_seven_days_ago_is_dropped_and_what_is_newer_is_kept() {
    let dir = TempDir::new("store-purge");
    let clock = clock();
    let mut old = file_record("aaaaaaaaaaaaaaaa", "pat");
    old["expires_ms"] = json!(T0 - 8 * DAY);
    let mut recent = file_record("bbbbbbbbbbbbbbbb", "pat");
    recent["expires_ms"] = json!(T0 - 6 * DAY);
    let mut revoked_old = file_record("cccccccccccccccc", "pat");
    revoked_old["revoked_ms"] = json!(T0 - 8 * DAY);
    let live = file_record("dddddddddddddddd", "pat");
    write_file(
        &dir,
        &json!({ "v": 1, "credentials": [old, recent, revoked_old, live] }),
    );
    let store = open(&dir, &clock);
    assert!(
        store.record("aaaaaaaaaaaaaaaa").is_none() && store.record("cccccccccccccccc").is_none()
    );
    assert!(
        store.record("bbbbbbbbbbbbbbbb").is_some() && store.record("dddddddddddddddd").is_some()
    );
    let ids: Vec<String> = read_file(&dir)["credentials"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids.len(), 2, "the purge is written: {ids:?}");
    // And hourly while running.
    clock.advance(2 * DAY);
    store.maintain();
    assert!(store.record("bbbbbbbbbbbbbbbb").is_none());
}

// ========================== memory is authoritative, writes are timely ============================

#[test]
fn nothing_on_disk_changes_a_live_credential() {
    // T42: a bogus credentials.json written while the server runs has no effect: no hot reload.
    let dir = TempDir::new("store-nohot");
    let clock = clock();
    let store = open(&dir, &clock);
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    store.revoke(&pat.id, "revoked");
    // Someone (a flow, say) rewrites the file: un-revokes it, and adds a credential of their own.
    let mut forged = file_record("0123456789abcdef", "pat");
    forged["hash"] = json!(KNOWN_HASH);
    let mut unrevoked = read_file(&dir);
    unrevoked["credentials"][0]["revoked_ms"] = Value::Null;
    unrevoked["credentials"]
        .as_array_mut()
        .unwrap()
        .push(forged);
    write_file(&dir, &unrevoked);
    assert_eq!(
        store.authenticate(&pat.token, None),
        Err(AuthError::Revoked { reason: "revoked" }),
        "the revocation stands"
    );
    assert_eq!(
        store.authenticate(KNOWN_TOKEN, None),
        Err(AuthError::Invalid),
        "the forged credential does not exist"
    );
    // The next write puts memory back on disk.
    store.flush().unwrap();
    let doc = read_file(&dir);
    assert_eq!(doc["credentials"].as_array().unwrap().len(), 1);
    assert!(doc["credentials"][0]["revoked_ms"].is_number());
}

#[test]
fn a_revocation_is_written_at_once_and_survives_a_restart() {
    let dir = TempDir::new("store-revoke-durable");
    let clock = clock();
    let store = open(&dir, &clock);
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    store.authenticate(&pat.token, None).unwrap();
    store.revoke(&pat.id, "revoked");
    // Without a flush call, without a minute passing: the file already says revoked.
    assert!(read_file(&dir)["credentials"][0]["revoked_ms"].is_number());
    drop(store);
    let store = open(&dir, &clock);
    assert_eq!(
        store.authenticate(&pat.token, None),
        Err(AuthError::Revoked { reason: "revoked" }),
        "nothing resurrects it"
    );
}

#[test]
fn last_used_is_flushed_at_most_once_a_minute_and_at_shutdown() {
    let dir = TempDir::new("store-flush-rate");
    let clock = clock();
    let writer = Toggle::new();
    let store = AuthStore::open(
        &auth_dir(&dir),
        Host::Server,
        clock.clone(),
        writer.clone(),
        None,
    )
    .unwrap();
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    let after_mint = writer.writes.load(Ordering::SeqCst);
    for _ in 0..200 {
        store.authenticate(&pat.token, Some("203.0.113.9")).unwrap();
        clock.advance(100);
        store.maintain();
    }
    assert_eq!(
        writer.writes.load(Ordering::SeqCst),
        after_mint,
        "20 seconds of requests write nothing"
    );
    clock.advance(MIN);
    store.maintain();
    assert_eq!(
        writer.writes.load(Ordering::SeqCst),
        after_mint + 1,
        "a minute later, one write"
    );
    store.maintain();
    assert_eq!(
        writer.writes.load(Ordering::SeqCst),
        after_mint + 1,
        "and not again until something changes"
    );
    // Shutdown flushes what is dirty, before the process exits.
    store
        .authenticate(&pat.token, Some("198.51.100.7"))
        .unwrap();
    store.flush().unwrap();
    let doc = read_file(&dir);
    assert_eq!(doc["credentials"][0]["last_used_ip"], "198.51.100.7");
    assert!(doc["credentials"][0]["last_used_ms"].is_number());
}

// ================================= a disk that will not take a write =============================

#[test]
fn signing_in_needs_no_write_a_full_disk_makes_a_session_that_lives_in_memory_and_says_so() {
    let dir = TempDir::new("store-full");
    let clock = clock();
    let writer = Toggle::new();
    let store = AuthStore::open(
        &auth_dir(&dir),
        Host::Server,
        clock.clone(),
        writer.clone(),
        None,
    )
    .unwrap();
    writer.fail.store(true, Ordering::SeqCst);
    let ses = store
        .mint(session(App::Dash, Preset::Owner))
        .expect("the login still succeeds");
    let p = store.authenticate(&ses.token, None).unwrap();
    assert!(!p.persisted, "`persisted: false` is visible");
    assert_eq!(store.storage(), Storage::Full);
    assert_eq!(store.storage().name(), "full");
    // Every other writer fails with store_unavailable and leaves nothing behind.
    assert!(matches!(
        store.mint(native_pat(&["ai.read"])),
        Err(MintFailure::StoreUnavailable(_))
    ));
    assert_eq!(
        store.live_count(Kind::Pat),
        0,
        "the failed pairing is not half-made"
    );
    // The disk recovers: the session goes to disk with the next write.
    writer.fail.store(false, Ordering::SeqCst);
    store.mint(native_pat(&["ai.read"])).unwrap();
    assert_eq!(store.storage(), Storage::Ok);
    let written: Vec<String> = read_file(&dir)["credentials"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        written.contains(&"ses".to_string()),
        "the memory-only session was written once the disk took writes: {written:?}"
    );
    assert!(store.authenticate(&ses.token, None).unwrap().persisted);
}

#[test]
fn a_revocation_survives_a_failed_write_in_memory() {
    let dir = TempDir::new("store-full-revoke");
    let clock = clock();
    let writer = Toggle::new();
    let store = AuthStore::open(
        &auth_dir(&dir),
        Host::Server,
        clock.clone(),
        writer.clone(),
        None,
    )
    .unwrap();
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    writer.fail.store(true, Ordering::SeqCst);
    assert!(store.revoke(&pat.id, "revoked"), "memory is authoritative");
    assert!(store.authenticate(&pat.token, None).is_err());
    assert_eq!(store.storage(), Storage::Full);
}

#[test]
fn is_storage_error_knows_a_full_or_read_only_disk() {
    assert!(is_storage_error(&io::Error::from_raw_os_error(ENOSPC)));
    assert!(!is_storage_error(&io::Error::other("nope")));
    assert!(!is_storage_error(&io::Error::from(
        io::ErrorKind::PermissionDenied
    )));
}

// ================================================ misc ===========================================

#[test]
fn using_a_child_refreshes_its_parents_idle_timer_at_most_once_a_minute_and_never_extends_an_absolute_time(
) {
    let clock = clock();
    let store = memory(&clock);
    let parent = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    let mut spec = session(App::Agent, Preset::Agent);
    spec.parent = Some(parent.id.clone());
    let kid = store.mint(spec).unwrap();
    store.authenticate(&kid.token, None).unwrap();
    let first = store.record(&parent.id).unwrap().last_used_ms;
    assert_eq!(first, Some(T0));
    clock.advance(30_000);
    store.authenticate(&kid.token, None).unwrap();
    assert_eq!(
        store.record(&parent.id).unwrap().last_used_ms,
        first,
        "not twice inside a minute"
    );
    clock.advance(31_000);
    store.authenticate(&kid.token, None).unwrap();
    assert_eq!(
        store.record(&parent.id).unwrap().last_used_ms,
        Some(T0 + 61_000)
    );
    assert_eq!(
        store.record(&parent.id).unwrap().expires_ms,
        T0 + DAY,
        "the absolute time is never extended"
    );
}

#[test]
fn elevation_is_a_window_on_a_session_and_nothing_else() {
    let clock = clock();
    let store = memory(&clock);
    let ses = store.mint(session(App::Dash, Preset::Owner)).unwrap();
    assert!(!store.authenticate(&ses.token, None).unwrap().elevated);
    store.set_elevated_until(&ses.id, T0 + 10 * MIN);
    assert!(store.authenticate(&ses.token, None).unwrap().elevated);
    clock.advance(10 * MIN);
    assert!(
        !store.authenticate(&ses.token, None).unwrap().elevated,
        "not extended by use"
    );
    // A token never elevates, whatever is set for its id.
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    store.set_elevated_until(&pat.id, T0 + DAY);
    assert!(!store.authenticate(&pat.token, None).unwrap().elevated);
    // The console is the OS user.
    let con = store
        .mint(MintSpec::new(Kind::Con, "console", ScopeSet::all(), DAY))
        .unwrap();
    let p = store.authenticate(&con.token, None).unwrap();
    assert!(p.elevated && p.kind == PrincipalKind::Console);
    // Revoking clears it.
    store.set_elevated_until(&ses.id, T0 + DAY);
    store.revoke(&ses.id, "logged_out");
    assert!(store.authenticate(&ses.token, None).is_err());
}

#[test]
fn the_allowed_origins_are_those_of_live_credentials_only() {
    let clock = clock();
    let store = memory(&clock);
    let a = store
        .mint(browser_pat("https://a.example", &["ai.read"]))
        .unwrap();
    let mut b = browser_pat("https://b.example", &["ai.read"]);
    b.ttl_ms = HOUR;
    store.mint(b).unwrap();
    store
        .mint(desk(
            App::Dash,
            Preset::Owner,
            &["tauri://localhost", "http://tauri.localhost"],
        ))
        .unwrap();
    store.mint(native_pat(&["ai.read"])).unwrap();
    let origins = |s: &AuthStore| s.allowed_origins().into_iter().collect::<Vec<_>>();
    assert_eq!(
        origins(&store),
        [
            "http://tauri.localhost",
            "https://a.example",
            "https://b.example",
            "tauri://localhost"
        ]
    );
    store.revoke(&a.id, "revoked");
    clock.advance(2 * HOUR);
    assert_eq!(
        origins(&store),
        ["http://tauri.localhost", "tauri://localhost"]
    );
    assert!(
        !store.allowed_origins().contains("null"),
        "null is never a member"
    );
}

#[test]
fn a_record_with_a_sender_constraint_or_no_expiry_or_a_dev_kind_never_authenticates() {
    let dir = TempDir::new("store-unusable");
    let clock = clock();
    let secret = "C".repeat(43);
    let hash = token::secret_hash(&secret);
    let mut cnf = file_record("1111111111111111", "pat");
    cnf["hash"] = json!(hash);
    cnf["cnf"] = json!({ "jkt": "abc" });
    let mut no_expiry = file_record("2222222222222222", "pat");
    no_expiry["hash"] = json!(hash);
    no_expiry.as_object_mut().unwrap().remove("expires_ms");
    let mut dev = file_record("3333333333333333", "dev");
    dev["hash"] = json!(hash);
    let mut good = file_record("4444444444444444", "pat");
    good["hash"] = json!(hash);
    write_file(
        &dir,
        &json!({ "v": 1, "credentials": [cnf, no_expiry, dev, good] }),
    );
    let store = open(&dir, &clock);
    let with = |id: &str, kind: &str| format!("oaiy{kind}_{id}_{secret}");
    assert_eq!(
        store.authenticate(&with("1111111111111111", "pat"), None),
        Err(AuthError::Invalid),
        "a constraint this build cannot honour"
    );
    assert!(
        store
            .authenticate(&with("2222222222222222", "pat"), None)
            .is_err(),
        "no expiry means expired"
    );
    assert_eq!(
        store.authenticate(&with("3333333333333333", "dev"), None),
        Err(AuthError::Invalid),
        "the device cookie is not a credential"
    );
    assert!(store
        .authenticate(&with("4444444444444444", "pat"), None)
        .is_ok());
}

#[test]
fn a_legacy_pairing_token_is_found_by_the_hash_of_the_whole_token_and_only_for_an_imported_record()
{
    let dir = TempDir::new("store-legacy");
    let clock = clock();
    let legacy = format!("oaiypat_{}", "0123456789abcdef".repeat(4));
    let mut rec = file_record("c1fffb8ad3bcf516", "pat");
    rec["hash"] = json!(token::legacy_hash(&legacy));
    rec["legacy"] = json!(true);
    rec["scopes"] = json!(["ai.read"]);
    write_file(&dir, &json!({ "v": 1, "credentials": [rec] }));
    let store = open(&dir, &clock);
    let p = store.authenticate(&legacy, None).unwrap();
    assert!(p.legacy_import && p.id == "c1fffb8ad3bcf516");
    // A token of the new grammar cannot reach a legacy record, whatever it says.
    let forged = format!("oaiypat_c1fffb8ad3bcf516_{}", "A".repeat(43));
    assert_eq!(store.authenticate(&forged, None), Err(AuthError::Invalid));
    // A legacy-shaped token nobody imported is nothing.
    let other = format!("oaiypat_{}", "fedcba9876543210".repeat(4));
    assert_eq!(store.authenticate(&other, None), Err(AuthError::Invalid));
    // And a record that is not marked legacy cannot be reached by a legacy-shaped token.
    let mut plain = file_record("5555555555555555", "pat");
    plain["hash"] = json!(token::legacy_hash(&other));
    write_file(&dir, &json!({ "v": 1, "credentials": [plain] }));
    drop(store);
    let store = open(&dir, &clock);
    assert_eq!(store.authenticate(&other, None), Err(AuthError::Invalid));
}

#[test]
fn a_memory_store_touches_no_disk() {
    let clock = clock();
    let store = memory(&clock);
    assert!(!store.is_persistent());
    store.mint(native_pat(&["ai.read"])).unwrap();
    store.flush().unwrap();
    assert_eq!(store.storage(), Storage::Ok);
}

#[test]
fn the_ip_recorded_is_cut_to_64_characters() {
    let clock = clock();
    let store = memory(&clock);
    let pat = store.mint(native_pat(&["ai.read"])).unwrap();
    store
        .authenticate(&pat.token, Some(&"9".repeat(500)))
        .unwrap();
    assert_eq!(
        store.record(&pat.id).unwrap().last_used_ip.unwrap().len(),
        64
    );
}

#[test]
fn the_error_messages_of_the_store_carry_no_secret() {
    let all = [
        MintFailure::TooManyCredentials.to_string(),
        MintFailure::TooManyChildren.to_string(),
        MintFailure::ScopeNotGrantable("x".into()).to_string(),
        MintFailure::TtlTooLong { max_ms: 5000 }.to_string(),
        MintFailure::ParentInvalid.to_string(),
        DeriveError::Refused("no").to_string(),
        DeriveError::RateLimited { retry_after_s: 3 }.to_string(),
    ];
    for m in all {
        assert!(!m.contains("oaiy"), "{m}");
    }
}
