//! Putting the guard together for a running server or desktop.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use super::audit::{self, AuditLog};
use super::bearer_throttle::ThrottleFile;
use super::clock::{Clock, SystemClock};
use super::exposure::Config;
use super::guard::{Guard, GuardConfig};
use super::mode::{validate_mode, AccessMode};
use super::store::{AuthStore, Host, SecureWriter, StoreError};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// What `serve` is told about the access model.
#[derive(Clone, Debug)]
pub struct AccessSettings {
    pub mode: AccessMode,
    /// A server that passed the startup rules (`auth::exposure`): what the guard, the login and the control
    /// switch are built from. The desktop has none, and reads its settings leniently.
    pub config: Option<Arc<Config>>,
}

impl AccessSettings {
    pub fn new(mode: AccessMode) -> Self {
        AccessSettings { mode, config: None }
    }

    /// Today's behaviour: the default until the flip.
    pub fn legacy() -> Self {
        AccessSettings::new(AccessMode::Legacy)
    }

    /// A server whose configuration passed the startup rules.
    pub fn for_server(config: Config) -> Self {
        AccessSettings {
            mode: config.mode,
            config: Some(Arc::new(config)),
        }
    }
}

impl Default for AccessSettings {
    fn default() -> Self {
        Self::legacy()
    }
}

/// Make the guard for a listener on `port`.
///
/// In `legacy` mode the store is memory-only: nothing is created, locked, read or written under the data
/// folder, so the mode changes nothing on the owner's machine (the one thing looked at is whether
/// `<data>/auth/owner.json` exists, because a login means `legacy` is refused). In `scoped` and `shadow` the
/// store opens `<data>/auth` (locked: a second process on the folder is refused), the audit and noise logs
/// start there, and a refusal (`ConfigRefusal`, `StoreError`) is an error the caller exits 78 on.
pub fn build_guard(
    settings: &AccessSettings,
    data_dir: &Path,
    port: u16,
    bind_all: bool,
    gui: bool,
    static_token: Option<String>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Arc<Guard>, BoxError> {
    let (config, warnings) = match &settings.config {
        Some(validated) => (GuardConfig::from_config(validated, gui), Vec::new()),
        None => GuardConfig::from_env(env, bind_all, gui, port),
    };
    for w in &warnings {
        log::warn!("auth: {w}");
    }
    let auth_dir = data_dir.join("auth");
    validate_mode(
        settings.mode,
        config.exposure,
        auth_dir.join("owner.json").exists(),
    )?;
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let (store, audit_log) = if settings.mode.is_enforcing() {
        let log = Arc::new(AuditLog::open(&auth_dir, clock.clone(), !gui));
        let store = AuthStore::open(
            &auth_dir,
            if gui { Host::Gui } else { Host::Server },
            clock.clone(),
            Arc::new(SecureWriter),
            Some(log.clone()),
        )
        .map_err(|e: StoreError| -> BoxError { Box::new(e) })?;
        audit::use_log(log.clone());
        (store, Some(log))
    } else {
        (AuthStore::memory(clock.clone()), None)
    };
    let exposure = config.exposure;
    let guard = Arc::new(Guard::new(
        settings.mode,
        config,
        Arc::new(store),
        static_token,
        audit_log,
        clock,
    ));
    log::info!(
        "auth: access mode {}, exposure {}",
        settings.mode.name(),
        exposure.name()
    );
    if settings.mode.is_enforcing() {
        // Blocks survive a restart and a kill: what the last run saved is loaded, and the upkeep saves it.
        let (file, saved) = ThrottleFile::open(&auth_dir.join("throttle.json"));
        guard.keep_throttle_in(file, saved.as_ref());
    }
    if let Some(log) = guard.audit() {
        log.critical(
            "startup",
            None,
            &Default::default(),
            serde_json::json!({
                "exposure": exposure.name(),
                "mode": settings.mode.name(),
                "port": port,
                "hosts": guard.config().hosts.describe(),
                "trustedProxies": guard.config().trusted.describe(),
                "proxyOnly": guard.config().proxy_only,
            }),
        );
    }
    remember(guard.clone());
    Ok(guard)
}

fn installed_slot() -> &'static Mutex<Option<Arc<Guard>>> {
    static SLOT: OnceLock<Mutex<Option<Arc<Guard>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Remember the running guard, so that shutdown can flush it.
pub fn remember(guard: Arc<Guard>) {
    *installed_slot().lock().unwrap_or_else(|e| e.into_inner()) = Some(guard);
}

/// Write what the running guard holds in memory to disk: the last-used times, and the noise counted so far.
/// Called at shutdown, before the process exits; harmless with nothing installed.
pub fn flush_installed() {
    let guard = installed_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(guard) = guard {
        flush_guard(&guard);
    }
}

/// What shutdown does to a guard: write the credential store, the throttle and the audit counts that are
/// still in memory.
pub fn flush_guard(guard: &Guard) {
    if let Err(e) = guard.store().flush() {
        log::warn!("auth: the credential store could not be written at shutdown: {e}");
    }
    // The web login's own state (its throttle goes into the throttle file with the failed-bearer throttle's, and
    // the owner's devices into `owner.json`): first, so that the write below has nothing left to add. The login
    // has an upkeep of its own while the server runs (`login::maintain_forever`).
    #[cfg(feature = "web")]
    if let Some(login) = guard.login() {
        login.flush();
    }
    guard.flush_throttle();
    if let Some(log) = guard.audit() {
        log.flush_noise();
    }
}

/// How often the upkeep of a running guard runs.
pub const MAINTAIN_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

/// Periodic upkeep of a running guard: expire, purge, write the last-used times once a minute, write the
/// noise of minutes that are over. Runs until the process ends, in every access mode: `legacy` has a
/// (memory-only) store too, and the routes it takes (`derive`) make credentials that must not pile up.
pub async fn maintain_forever(guard: Arc<Guard>) {
    maintain_every(guard, MAINTAIN_EVERY).await
}

/// [`maintain_forever`] at another pace (the tests').
pub async fn maintain_every(guard: Arc<Guard>, period: std::time::Duration) {
    log::info!(
        "auth: credential upkeep every {} s (access mode {})",
        period.as_secs(),
        guard.mode().name()
    );
    loop {
        tokio::time::sleep(period).await;
        maintain_once(&guard);
    }
}

/// One round of upkeep.
pub fn maintain_once(guard: &Guard) {
    guard.store().maintain();
    guard.flush_throttle();
    if let Some(log) = guard.audit() {
        log.flush_closed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::TempDir;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn legacy_mode_touches_nothing_under_the_data_folder() {
        let dir = TempDir::new("runtime-legacy");
        let guard = build_guard(
            &AccessSettings::legacy(),
            &dir.0,
            17972,
            false,
            true,
            Some("t".repeat(32)),
            &no_env,
        )
        .unwrap();
        assert_eq!(guard.mode(), AccessMode::Legacy);
        assert!(!guard.store().is_persistent());
        assert!(guard.audit().is_none());
        assert_eq!(
            std::fs::read_dir(&dir.0).unwrap().count(),
            0,
            "no folder, no lock, no file was made"
        );
        assert_eq!(guard.health_extras().access, "legacy");
        assert_eq!(guard.health_extras().storage, "ok");
        // With the upkeep running (it does in every mode) and a credential derived, still nothing on disk.
        guard
            .store()
            .derive(
                &super::super::principal::Principal::static_token(),
                super::super::store::DeriveRequest {
                    scopes: super::super::scopes::ScopeSet::of(&["system.read"]),
                    ttl_ms: Some(1000),
                    label: "x".into(),
                },
            )
            .unwrap();
        maintain_once(&guard);
        flush_guard(&guard);
        assert_eq!(
            std::fs::read_dir(&dir.0).unwrap().count(),
            0,
            "the upkeep of a legacy guard writes nothing either"
        );
    }

    #[test]
    fn shutdown_writes_the_last_used_times_the_store_kept_in_memory() {
        use super::super::scopes::ScopeSet;
        use super::super::store::MintSpec;
        use super::super::token::Kind;
        let dir = TempDir::new("runtime-shutdown");
        let guard = build_guard(
            &AccessSettings::new(AccessMode::Scoped),
            &dir.0,
            17972,
            false,
            false,
            None,
            &no_env,
        )
        .unwrap();
        let made = guard
            .store()
            .mint(MintSpec::new(
                Kind::Pat,
                "a tool",
                ScopeSet::of(&["system.read"]),
                30 * 24 * 3_600_000,
            ))
            .unwrap();
        guard
            .store()
            .authenticate(&made.token, Some("127.0.0.1"))
            .unwrap();
        let last_used = || {
            let text =
                std::fs::read_to_string(dir.0.join("auth").join("credentials.json")).unwrap();
            let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
            doc["credentials"][0]["last_used_ms"].clone()
        };
        assert!(last_used().is_null(), "kept in memory until it is written");
        flush_guard(&guard);
        assert!(last_used().is_u64(), "{}", last_used());
        audit::uninstall();
    }

    #[test]
    fn f9_the_startup_event_says_the_hosts_and_the_trusted_proxies() {
        let dir = TempDir::new("runtime-startup");
        let vars = |name: &str| match name {
            "OAIY_PUBLIC_URL" => Some("https://dash.example.com".to_string()),
            "OAIY_AGENT_URL" => Some("https://agent.example.com:8443".to_string()),
            "OAIY_TRUSTED_PROXIES" => Some("10.0.0.0/8, 192.0.2.7".to_string()),
            "OAIY_ALLOWED_HOSTS" => Some("nas.example:9000".to_string()),
            _ => None,
        };
        let _guard = build_guard(
            &AccessSettings::new(AccessMode::Scoped),
            &dir.0,
            17972,
            false,
            false,
            None,
            &vars,
        )
        .unwrap();
        let text = std::fs::read_to_string(dir.0.join("auth").join("audit.jsonl")).unwrap();
        let startup: serde_json::Value = text
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .find(|l| l["event"] == "startup")
            .expect("a startup event");
        let d = &startup["detail"];
        assert_eq!(d["mode"], "scoped");
        assert_eq!(d["exposure"], "proxied");
        assert_eq!(d["port"], 17972);
        let hosts: Vec<&str> = d["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h.as_str().unwrap())
            .collect();
        for want in [
            "dash.example.com",
            "agent.example.com:8443",
            "nas.example:9000",
        ] {
            assert!(hosts.contains(&want), "{want} in {hosts:?}");
        }
        assert!(
            hosts.iter().any(|h| h.starts_with("localhost")),
            "{hosts:?}"
        );
        assert_eq!(
            d["trustedProxies"],
            serde_json::json!(["10.0.0.0/8", "192.0.2.7/32"])
        );
        audit::uninstall();
    }

    #[test]
    fn f9_a_scoped_guard_keeps_its_throttle_in_the_data_folder_and_a_legacy_one_does_not() {
        use super::super::bearer_throttle::THROTTLE_FILE_VERSION;
        let dir = TempDir::new("runtime-throttle");
        // Saved by an earlier run: an address blocked for a while.
        let now = super::super::clock::SystemClock.now_ms();
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        std::fs::write(
            dir.0.join("auth").join("throttle.json"),
            serde_json::json!({
                "v": THROTTLE_FILE_VERSION,
                "bearer": { "203.0.113.9": { "failures": [], "blocked_until": now + 600_000, "last_block_end": now + 600_000, "level": 1, "last_seen": now } }
            })
            .to_string(),
        )
        .unwrap();
        let guard = build_guard(
            &AccessSettings::new(AccessMode::Scoped),
            &dir.0,
            17972,
            false,
            false,
            None,
            &no_env,
        )
        .unwrap();
        assert!(
            guard.throttle().blocked_for("203.0.113.9").is_some(),
            "loaded at start"
        );
        // A failure is saved by the upkeep, and the last one at shutdown.
        let file = dir.0.join("auth").join("throttle.json");
        guard.throttle().record_failure("198.51.100.4");
        maintain_once(&guard);
        let saved = std::fs::read_to_string(&file).unwrap();
        assert!(
            saved.contains("198.51.100.4") && saved.contains("203.0.113.9"),
            "{saved}"
        );
        guard.throttle().record_failure("198.51.100.5");
        flush_guard(&guard);
        let saved = std::fs::read_to_string(&file).unwrap();
        assert!(saved.contains("198.51.100.5"), "{saved}");
        audit::uninstall();
        // Legacy touches no disk, so it has no file to keep it in.
        let legacy_dir = TempDir::new("runtime-throttle-legacy");
        let legacy = build_guard(
            &AccessSettings::legacy(),
            &legacy_dir.0,
            17972,
            false,
            true,
            None,
            &no_env,
        )
        .unwrap();
        legacy.throttle().record_failure("198.51.100.4");
        maintain_once(&legacy);
        assert_eq!(std::fs::read_dir(&legacy_dir.0).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn f7_the_upkeep_runs_in_legacy_mode_too_and_drops_what_derive_made() {
        use super::super::clock::ManualClock;
        use super::super::principal::Principal;
        use super::super::scopes::ScopeSet;
        use super::super::store::{DeriveRequest, RUN_PURGE_AFTER_MS};
        use super::super::token::Kind;
        let clock = Arc::new(ManualClock::new(1_790_000_000_000));
        let (config, _) = GuardConfig::from_env(&no_env, false, false, 17972);
        let guard = Arc::new(Guard::new(
            AccessMode::Legacy,
            config,
            Arc::new(AuthStore::memory(clock.clone())),
            Some("t".repeat(32)),
            None,
            clock.clone(),
        ));
        for _ in 0..5 {
            guard
                .store()
                .derive(
                    &Principal::static_token(),
                    DeriveRequest {
                        scopes: ScopeSet::of(&["system.read"]),
                        ttl_ms: Some(1000),
                        label: "x".into(),
                    },
                )
                .unwrap();
            clock.advance(2000);
        }
        assert_eq!(guard.store().held_count(Kind::Run), 5);
        clock.advance(RUN_PURGE_AFTER_MS + 60_000);
        // The task the listener spawns in every mode, at a pace a test can wait for.
        let task = tokio::spawn(maintain_every(
            guard.clone(),
            std::time::Duration::from_millis(5),
        ));
        for _ in 0..400 {
            if guard.store().held_count(Kind::Run) == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        task.abort();
        assert_eq!(guard.store().held_count(Kind::Run), 0);
    }

    #[test]
    fn scoped_mode_opens_the_folder_locks_it_and_starts_the_logs() {
        let dir = TempDir::new("runtime-scoped");
        let guard = build_guard(
            &AccessSettings::new(AccessMode::Scoped),
            &dir.0,
            17972,
            false,
            false,
            None,
            &no_env,
        )
        .unwrap();
        assert!(guard.store().is_persistent());
        assert!(guard.audit().is_some());
        assert!(dir.0.join("auth").join(".lock").exists());
        assert!(
            dir.0.join("auth").join("audit.jsonl").exists(),
            "the startup event is written"
        );
        // A second process on the folder is refused.
        let second = build_guard(
            &AccessSettings::new(AccessMode::Scoped),
            &dir.0,
            17973,
            false,
            false,
            None,
            &no_env,
        );
        let err = second.err().expect("refused");
        assert!(err.to_string().contains("in use by process"), "{err}");
        assert!(err.downcast_ref::<StoreError>().is_some());
        audit::uninstall();
    }

    #[test]
    fn a_refused_mode_is_an_error_that_carries_the_exit_code() {
        let dir = TempDir::new("runtime-refused");
        let lan = |name: &str| {
            (name == "OAIY_PUBLIC_URL").then(|| "https://dash.example.com".to_string())
        };
        let err = build_guard(
            &AccessSettings::new(AccessMode::Shadow),
            &dir.0,
            17972,
            false,
            false,
            None,
            &lan,
        )
        .err()
        .expect("refused");
        let refusal = err
            .downcast_ref::<super::super::mode::ConfigRefusal>()
            .expect("a ConfigRefusal");
        assert_eq!(refusal.exit_code(), 78);
        assert!(!dir.0.join("auth").exists(), "and nothing was made");
        // An owner login means legacy is refused.
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        std::fs::write(dir.0.join("auth").join("owner.json"), "{}").unwrap();
        assert!(build_guard(
            &AccessSettings::legacy(),
            &dir.0,
            17972,
            false,
            false,
            None,
            &no_env
        )
        .is_err());
    }
}
