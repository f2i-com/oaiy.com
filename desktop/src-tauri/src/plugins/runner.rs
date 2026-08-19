//! Plugin process supervision: spawn, handshake, health, restart, shutdown.
//!
//! The *policy* here is deliberately separated from the *plumbing*:
//! [`plugin_env`], [`restart_delay`] and [`HealthTracker`] are pure and directly
//! tested, because they encode the decisions that matter and a threaded
//! integration test would only exercise them incidentally.
//!
//! # The environment is an allow-list, not an inheritance
//!
//! [`plugin_env`] builds the child's environment from nothing. This is a security
//! control, not tidiness: the host process environment can hold a Hugging Face
//! token, provider API keys, proxy credentials, and whatever else the user's shell
//! exported. A plugin is untrusted code the user chose to run, and
//! `Command::new(..).spawn()` hands it every one of those by default.
//!
//! So the child gets a fixed set of `OAIY_*` values plus the handful of OS
//! variables a process genuinely cannot run without.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

/// How long the plugin has to answer `plugin.init` before it is killed.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Interval between `plugin.health` probes.
pub const HEALTH_INTERVAL: Duration = Duration::from_secs(10);
/// Per-probe deadline. Shorter than the interval so a slow probe cannot overlap
/// the next one and make "one plugin is slow" look like "the host is stuck".
pub const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);
/// Consecutive missed probes before a plugin is marked `Unhealthy`.
///
/// Three, not one: a single missed probe is usually a plugin busy on a real
/// call, and flapping a status badge teaches people to ignore it.
pub const HEALTH_MISS_LIMIT: u32 = 3;
/// Grace period after `plugin.shutdown` before the process is killed.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
/// Automatic restarts after a crash, before the plugin stays `Crashed`.
///
/// Bounded because a plugin that crashes on startup would otherwise respawn
/// forever, and an infinite restart loop is indistinguishable from a working
/// plugin right up until the disk fills with logs.
pub const MAX_RESTART_ATTEMPTS: u32 = 3;

/// Backoff before restart attempt `attempt` (1-based).
///
/// 1s, 4s, 16s. Exponential rather than fixed so a plugin that is going to fail
/// anyway stops consuming the machine, and capped by
/// [`MAX_RESTART_ATTEMPTS`] rather than by the delay growing forever.
pub fn restart_delay(attempt: u32) -> Duration {
    let secs = match attempt {
        0 | 1 => 1,
        2 => 4,
        _ => 16,
    };
    Duration::from_secs(secs)
}

/// Should we try again after a crash?
pub fn should_restart(attempts_so_far: u32) -> bool {
    attempts_so_far < MAX_RESTART_ATTEMPTS
}

/// OS variables a child process genuinely needs.
///
/// Everything else is dropped. `PATH` is included because a plugin that shells
/// out to a system tool cannot find it otherwise; the Windows entries are
/// required by the loader and by `std::env::temp_dir`.
#[cfg(windows)]
const OS_ENV_ALLOW_LIST: &[&str] = &[
    "PATH",
    "SystemRoot",
    "SystemDrive",
    "windir",
    "TEMP",
    "TMP",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
];

#[cfg(not(windows))]
const OS_ENV_ALLOW_LIST: &[&str] = &["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TZ"];

/// Names that must never reach a plugin even if they appear on the allow-list.
///
/// Belt and braces: the allow-list above is the control, but a future edit adding
/// something broad (`*_PROXY`, say) shouldn't be able to leak a credential
/// without tripping a test.
pub(crate) const NEVER_FORWARD: &[&str] = &[
    "OAIY_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "GITHUB_TOKEN",
];

/// Build the child environment from scratch.
///
/// `host_env` is the parent environment, passed in rather than read from the
/// process so this is testable.
pub fn plugin_env<I, K, V>(
    host_env: I,
    plugin_id: &str,
    plugin_data_dir: &Path,
    desktop_version: &str,
    plugin_api_version: u32,
    dev_mode: bool,
) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut env = BTreeMap::new();

    for (k, v) in host_env {
        let key = k.as_ref();
        if NEVER_FORWARD
            .iter()
            .any(|deny| deny.eq_ignore_ascii_case(key))
        {
            continue;
        }
        // Case-insensitive on Windows, where `Path` and `PATH` are the same
        // variable and a case-sensitive compare would drop it.
        if OS_ENV_ALLOW_LIST
            .iter()
            .any(|allow| allow.eq_ignore_ascii_case(key))
        {
            env.insert(key.to_string(), v.as_ref().to_string());
        }
    }

    env.insert("OAIY_PLUGIN_ID".into(), plugin_id.to_string());
    env.insert(
        "OAIY_PLUGIN_DATA_DIR".into(),
        plugin_data_dir.display().to_string(),
    );
    env.insert("OAIY_DESKTOP_VERSION".into(), desktop_version.to_string());
    env.insert(
        "OAIY_PLUGIN_API_VERSION".into(),
        plugin_api_version.to_string(),
    );
    if dev_mode {
        env.insert("OAIY_DEV_MODE".into(), "1".into());
    }
    env
}

/// Tracks consecutive health-probe misses.
#[derive(Debug, Default, Clone)]
pub struct HealthTracker {
    consecutive_misses: u32,
    last_detail: Option<String>,
}

/// What the supervisor should do after a probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthVerdict {
    /// Healthy, or not yet at the miss limit.
    Ok,
    /// A probe failed but the limit is not reached — no state change yet.
    Degraded { misses: u32 },
    /// At or past the limit: mark `Unhealthy`.
    Unhealthy { misses: u32, detail: String },
}

impl HealthTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// The plugin answered `{status: "ok"}`.
    ///
    /// Resets the counter, so misses must be *consecutive*. Counting cumulative
    /// misses would eventually mark every long-running plugin unhealthy.
    pub fn record_ok(&mut self) -> HealthVerdict {
        self.consecutive_misses = 0;
        self.last_detail = None;
        HealthVerdict::Ok
    }

    /// The probe timed out, errored, or the plugin answered `error`.
    pub fn record_miss(&mut self, detail: impl Into<String>) -> HealthVerdict {
        self.consecutive_misses += 1;
        let detail = detail.into();
        self.last_detail = Some(detail.clone());
        if self.consecutive_misses >= HEALTH_MISS_LIMIT {
            HealthVerdict::Unhealthy {
                misses: self.consecutive_misses,
                detail: format!(
                    "{} consecutive health probes failed. Last: {detail}",
                    self.consecutive_misses
                ),
            }
        } else {
            HealthVerdict::Degraded {
                misses: self.consecutive_misses,
            }
        }
    }

    /// The plugin answered `{status: "degraded"}` — it is alive and telling us
    /// something is wrong, which is not the same as failing to answer.
    ///
    /// Reported without incrementing the miss counter: a plugin honestly
    /// reporting a degraded dongle should not be killed for its honesty.
    pub fn record_self_reported_degraded(&mut self, detail: impl Into<String>) -> HealthVerdict {
        self.consecutive_misses = 0;
        let detail = detail.into();
        self.last_detail = Some(detail.clone());
        HealthVerdict::Unhealthy { misses: 0, detail }
    }

    pub fn misses(&self) -> u32 {
        self.consecutive_misses
    }

    pub fn last_detail(&self) -> Option<&str> {
        self.last_detail.as_deref()
    }
}

/// Per-plugin writable directory, beside the plugins root rather than inside
/// the plugin's own folder.
///
/// It used to be `<plugin>/data`, which put mutable runtime state INSIDE the
/// signed bundle. `package-signer verify` requires every listed file to match
/// its digest AND no unlisted file to exist, so the plugin's first settings
/// write invalidated its own signature permanently — there was no way to
/// re-sign it that survived the next write.
///
/// Uninstall still takes the data with it: [`super::install::uninstall`]
/// removes this directory alongside the bundle. That property is precisely why
/// the data lived inside the bundle before, so it is now preserved explicitly
/// rather than as a side effect — Aokie's state includes phone pairing keys,
/// and state that outlives its uninstall surprises people.
pub fn plugin_data_dir(plugin_dir: &Path) -> std::path::PathBuf {
    // <dataDir>/plugins/<id>  ->  <dataDir>/plugin-data/<id>
    match (
        plugin_dir.parent().and_then(|p| p.parent()),
        plugin_dir.file_name(),
    ) {
        (Some(data_root), Some(id)) => data_root.join("plugin-data").join(id),
        // Only reachable for a path with no grandparent, which the real layout
        // never produces. Staying inside the folder beats inventing a location
        // outside the tree the caller controls.
        _ => plugin_dir.join("data"),
    }
}

/// Move a pre-existing `<plugin>/data` to the new location, once.
///
/// Returning an error here would refuse to start a plugin whose data we could
/// not move, which is worse than starting it: the caller logs and continues,
/// and a plugin that finds an empty data dir re-pairs rather than breaking.
/// Only a genuine legacy directory is moved — if the new location already
/// exists, the old one is left alone rather than merged, so a half-migrated
/// state can be inspected instead of silently reconciled.
pub fn migrate_legacy_data_dir(plugin_dir: &Path) -> Result<bool, String> {
    let legacy = plugin_dir.join("data");
    if !legacy.is_dir() {
        return Ok(false);
    }
    let target = plugin_data_dir(plugin_dir);
    if target == legacy {
        return Ok(false);
    }
    if target.exists() {
        return Err(format!(
            "both {} and {} exist — leaving the legacy directory in place",
            legacy.display(),
            target.display()
        ));
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::rename(&legacy, &target)
        .map_err(|e| format!("cannot move {} to {}: {e}", legacy.display(), target.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_env() -> Vec<(&'static str, &'static str)> {
        vec![
            ("PATH", "/usr/bin"),
            ("OAIY_HF_TOKEN", "hf_secret_do_not_leak"),
            ("OPENAI_API_KEY", "sk-secret"),
            ("ANTHROPIC_API_KEY", "sk-ant-secret"),
            ("AWS_SECRET_ACCESS_KEY", "aws-secret"),
            ("GITHUB_TOKEN", "ghp_secret"),
            ("SOME_USER_SECRET", "hunter2"),
            ("DATABASE_URL", "postgres://user:pw@host/db"),
            ("SystemRoot", r"C:\Windows"),
            ("TEMP", r"C:\Temp"),
            ("HOME", "/home/u"),
        ]
    }

    fn env_for_test() -> BTreeMap<String, String> {
        plugin_env(
            host_env(),
            "aokie",
            Path::new(r"C:\plugins\aokie\data"),
            "0.1.0",
            1,
            false,
        )
    }

    // --- the environment allow-list: a security control -------------------

    #[test]
    fn the_host_environment_is_not_inherited() {
        let env = env_for_test();
        // These are the whole reason the allow-list exists.
        for leaked in [
            "OAIY_HF_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "SOME_USER_SECRET",
            "DATABASE_URL",
        ] {
            assert!(
                !env.contains_key(leaked),
                "{leaked} reached the plugin; a plugin is untrusted code"
            );
        }
    }

    #[test]
    fn no_value_from_a_denied_variable_appears_anywhere() {
        // Guards against a future edit forwarding a secret under a different key.
        let env = env_for_test();
        let joined = env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";");
        for secret in [
            "hf_secret_do_not_leak",
            "sk-secret",
            "sk-ant-secret",
            "aws-secret",
            "ghp_secret",
            "hunter2",
            "postgres://user:pw@host/db",
        ] {
            assert!(!joined.contains(secret), "{secret} leaked: {joined}");
        }
    }

    #[test]
    fn the_os_essentials_are_forwarded() {
        let env = env_for_test();
        // A plugin that shells out to a system tool needs PATH.
        assert_eq!(env.get("PATH").map(String::as_str), Some("/usr/bin"));
        #[cfg(windows)]
        {
            assert!(env.contains_key("SystemRoot"), "the Windows loader needs it");
            assert!(env.contains_key("TEMP"));
        }
        #[cfg(not(windows))]
        {
            assert!(env.contains_key("HOME"));
        }
    }

    #[test]
    fn path_is_matched_case_insensitively() {
        // On Windows `Path` and `PATH` are the same variable; a case-sensitive
        // compare would silently drop it and the plugin would fail to find tools.
        let env = plugin_env(
            vec![("Path", "/usr/bin")],
            "p",
            Path::new("/d"),
            "0.1.0",
            1,
            false,
        );
        assert_eq!(env.get("Path").map(String::as_str), Some("/usr/bin"));
    }

    #[test]
    fn a_denied_variable_is_dropped_whatever_its_case() {
        let env = plugin_env(
            vec![("openai_api_key", "sk-x"), ("Hf_Token", "hf-x")],
            "p",
            Path::new("/d"),
            "0.1.0",
            1,
            false,
        );
        assert!(env.values().all(|v| v != "sk-x" && v != "hf-x"), "{env:?}");
    }

    #[test]
    fn the_contract_variables_are_present() {
        let env = env_for_test();
        assert_eq!(env.get("OAIY_PLUGIN_ID").map(String::as_str), Some("aokie"));
        assert_eq!(
            env.get("OAIY_DESKTOP_VERSION").map(String::as_str),
            Some("0.1.0")
        );
        assert_eq!(
            env.get("OAIY_PLUGIN_API_VERSION").map(String::as_str),
            Some("1")
        );
        assert!(env
            .get("OAIY_PLUGIN_DATA_DIR")
            .is_some_and(|d| d.contains("aokie")));
    }

    #[test]
    fn dev_mode_is_absent_unless_asked_for() {
        assert!(!env_for_test().contains_key("OAIY_DEV_MODE"));
        let dev = plugin_env(host_env(), "aokie", Path::new("/d"), "0.1.0", 1, true);
        assert_eq!(dev.get("OAIY_DEV_MODE").map(String::as_str), Some("1"));
    }

    #[test]
    fn a_plugin_cannot_override_its_own_identity() {
        // The OAIY_* values are written AFTER the allow-list pass, so a host
        // environment carrying a stale OAIY_PLUGIN_ID cannot mislabel the child.
        let env = plugin_env(
            vec![("OAIY_PLUGIN_ID", "impostor"), ("PATH", "/usr/bin")],
            "aokie",
            Path::new("/d"),
            "0.1.0",
            1,
            false,
        );
        assert_eq!(env.get("OAIY_PLUGIN_ID").map(String::as_str), Some("aokie"));
    }

    #[test]
    fn the_environment_is_small() {
        // If this grows unexpectedly, something started inheriting again.
        let env = env_for_test();
        assert!(env.len() <= OS_ENV_ALLOW_LIST.len() + 5, "{env:?}");
    }

    // --- restart policy ---------------------------------------------------

    #[test]
    fn restart_backoff_grows() {
        assert_eq!(restart_delay(1), Duration::from_secs(1));
        assert_eq!(restart_delay(2), Duration::from_secs(4));
        assert_eq!(restart_delay(3), Duration::from_secs(16));
        // And is capped rather than growing without bound.
        assert_eq!(restart_delay(99), Duration::from_secs(16));
    }

    #[test]
    fn restarts_are_bounded() {
        // A plugin that crashes on startup must stop respawning; an infinite
        // restart loop looks like a working plugin until the disk fills.
        assert!(should_restart(0));
        assert!(should_restart(1));
        assert!(should_restart(2));
        assert!(!should_restart(MAX_RESTART_ATTEMPTS));
        assert!(!should_restart(MAX_RESTART_ATTEMPTS + 10));
    }

    // --- health policy ----------------------------------------------------

    #[test]
    fn one_missed_probe_does_not_flip_the_state() {
        // A single miss is usually a plugin busy on a real call; flapping a badge
        // teaches people to ignore it.
        let mut h = HealthTracker::new();
        assert_eq!(h.record_miss("timeout"), HealthVerdict::Degraded { misses: 1 });
        assert_eq!(h.record_miss("timeout"), HealthVerdict::Degraded { misses: 2 });
    }

    #[test]
    fn the_third_consecutive_miss_marks_unhealthy() {
        let mut h = HealthTracker::new();
        h.record_miss("timeout");
        h.record_miss("timeout");
        match h.record_miss("timeout after 5s") {
            HealthVerdict::Unhealthy { misses, detail } => {
                assert_eq!(misses, HEALTH_MISS_LIMIT);
                assert!(detail.contains("3 consecutive"), "{detail}");
                assert!(detail.contains("timeout after 5s"), "name the last cause: {detail}");
            }
            other => panic!("expected Unhealthy, got {other:?}"),
        }
    }

    #[test]
    fn misses_must_be_consecutive() {
        // Counting cumulatively would eventually mark every long-running plugin
        // unhealthy.
        let mut h = HealthTracker::new();
        h.record_miss("a");
        h.record_miss("b");
        assert_eq!(h.record_ok(), HealthVerdict::Ok);
        assert_eq!(h.misses(), 0);
        assert!(h.last_detail().is_none());
        assert_eq!(h.record_miss("c"), HealthVerdict::Degraded { misses: 1 });
    }

    #[test]
    fn a_self_reported_degraded_plugin_is_not_penalised() {
        // Answering "degraded" is not the same as failing to answer. A plugin
        // honestly reporting a bad dongle should not be killed for its honesty.
        let mut h = HealthTracker::new();
        match h.record_self_reported_degraded("dongle firmware out of date") {
            HealthVerdict::Unhealthy { misses, detail } => {
                assert_eq!(misses, 0, "it answered, so it missed nothing");
                assert!(detail.contains("firmware"), "{detail}");
            }
            other => panic!("expected Unhealthy, got {other:?}"),
        }
        assert_eq!(h.misses(), 0);
    }

    // --- timings ----------------------------------------------------------

    #[test]
    fn a_probe_cannot_outlive_its_interval() {
        // Otherwise a slow probe overlaps the next and "one plugin is slow"
        // starts looking like "the host is stuck".
        assert!(HEALTH_TIMEOUT < HEALTH_INTERVAL);
    }

    #[test]
    fn the_handshake_gets_longer_than_a_health_probe() {
        // First start does real work: opening a dongle, loading a model.
        assert!(HANDSHAKE_TIMEOUT > HEALTH_TIMEOUT);
    }

    #[test]
    fn shutdown_grace_is_shorter_than_the_handshake() {
        // Quitting should be quicker than starting; a long grace makes app exit
        // feel broken.
        assert!(SHUTDOWN_GRACE < HANDSHAKE_TIMEOUT);
    }

    // --- data dir ---------------------------------------------------------

    #[test]
    fn plugin_data_lives_outside_the_signed_bundle() {
        // Inside the bundle, the first settings write invalidated the plugin's
        // own package signature: verification demands every listed file match
        // AND no unlisted file exist.
        let dir = Path::new("/data/plugins/aokie");
        let data = plugin_data_dir(dir);
        assert!(
            !data.starts_with(dir),
            "{} must not be inside the bundle",
            data.display()
        );
        assert!(data.ends_with("aokie"), "{}", data.display());
        assert!(
            data.to_string_lossy().contains("plugin-data"),
            "{}",
            data.display()
        );
    }

    #[test]
    fn a_legacy_data_dir_is_moved_once_and_its_contents_survive() {
        // The live install's legacy dir holds phone pairing keys, so this has
        // to move them rather than start clean beside them.
        let base = std::env::temp_dir().join(format!("oaiy-mig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let plugin = base.join("plugins").join("aokie");
        std::fs::create_dir_all(plugin.join("data")).unwrap();
        std::fs::write(plugin.join("data").join("settings.json"), b"paired").unwrap();

        assert_eq!(migrate_legacy_data_dir(&plugin), Ok(true));
        let moved = plugin_data_dir(&plugin);
        assert_eq!(std::fs::read(moved.join("settings.json")).unwrap(), b"paired");
        assert!(!plugin.join("data").exists(), "legacy dir should be gone");

        // Idempotent: a second start must not error or re-move anything.
        assert_eq!(migrate_legacy_data_dir(&plugin), Ok(false));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn migration_refuses_to_merge_when_both_locations_exist() {
        // Merging two settings files silently picks a winner; surfacing the
        // conflict lets someone look at both.
        let base = std::env::temp_dir().join(format!("oaiy-mig2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let plugin = base.join("plugins").join("aokie");
        std::fs::create_dir_all(plugin.join("data")).unwrap();
        std::fs::create_dir_all(plugin_data_dir(&plugin)).unwrap();

        let err = migrate_legacy_data_dir(&plugin).unwrap_err();
        assert!(err.contains("both"), "{err}");
        assert!(plugin.join("data").exists(), "legacy dir must be preserved");
        let _ = std::fs::remove_dir_all(&base);
    }
}
