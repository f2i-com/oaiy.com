//! Putting the guard together for a running server or desktop.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use super::audit::{self, AuditLog};
use super::clock::{Clock, SystemClock};
use super::guard::{Guard, GuardConfig};
use super::mode::{validate_mode, AccessMode};
use super::store::{AuthStore, Host, SecureWriter, StoreError};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// What `serve` is told about the access model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessSettings {
    pub mode: AccessMode,
}

impl AccessSettings {
    pub fn new(mode: AccessMode) -> Self {
        AccessSettings { mode }
    }

    /// Today's behaviour: the default until the flip.
    pub fn legacy() -> Self {
        AccessSettings {
            mode: AccessMode::Legacy,
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
    let (config, warnings) = GuardConfig::from_env(env, bind_all, gui, port);
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
    if let Some(log) = guard.audit() {
        log.critical("startup", None, &Default::default(), serde_json::json!({ "exposure": exposure.name(), "mode": settings.mode.name(), "port": port }));
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
        if let Err(e) = guard.store().flush() {
            log::warn!("auth: the credential store could not be written at shutdown: {e}");
        }
        if let Some(log) = guard.audit() {
            log.flush_noise();
        }
    }
}

/// Periodic upkeep of a running guard: expire, purge, write the last-used times once a minute, write the
/// noise of minutes that are over. Runs until the process ends.
pub async fn maintain_forever(guard: Arc<Guard>) {
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        guard.store().maintain();
        if let Some(log) = guard.audit() {
            log.flush_closed();
        }
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
