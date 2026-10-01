//! The access mode, the exposure of an install, and the refusals at startup (design 4.3 and 4.5.5).
//!
//! - `legacy`: today's guard for every route that existed before the access model. Routes the model
//!   adds are judged by the new guard in every mode. The default until the flip.
//! - `scoped`: the new guard everywhere.
//! - `shadow`: the rollback lever. Authentication is exactly `scoped`; only a non-dangerous route-scope
//!   mismatch is logged and allowed.
//!
//! **`scoped` must not be offered as a security boundary until flow approval (design ACC-05) exists.**
//! The design says a flow must be approved before it runs, but no approval is built: there is no
//! `POST /api/bridge/flows/:id/approve` handler, and a credential holding `flows.write` and `runs.write`
//! (the `agent` preset, among others) can store a flow and start it with nothing in between. A flow is
//! code that runs on this computer, so such a credential can do whatever a flow can, including writing
//! files that the route scopes above guard. The route table, the scopes and `scoped`'s refusals are
//! real and tested; this gap sits beside them. Until ACC-05 lands, `legacy` (the default) is the mode
//! the desktop runs, and a `scoped` install should be treated as hardening, not as isolation from a
//! credential that can run flows. `shadow` enforces only authentication and dangerous scopes, so the
//! newer scopes (`calls.settings`, `calls.manage`) are enforced in `scoped` only.
//!
//! The mode is read once at startup: `accessMode` in the desktop's config, `OAIY_ACCESS_MODE` for
//! `oaiy-server`.

/// The exit code of a refused configuration (`EX_CONFIG`): a shipped unit does not restart it.
pub use super::store::EX_CONFIG;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessMode {
    Legacy,
    Scoped,
    Shadow,
}

impl AccessMode {
    pub fn name(self) -> &'static str {
        match self {
            AccessMode::Legacy => "legacy",
            AccessMode::Scoped => "scoped",
            AccessMode::Shadow => "shadow",
        }
    }

    /// `legacy`, `scoped` or `shadow`, exactly (after trimming and lowercasing).
    pub fn parse(text: &str) -> Option<AccessMode> {
        match text.trim().to_ascii_lowercase().as_str() {
            "legacy" => Some(AccessMode::Legacy),
            "scoped" => Some(AccessMode::Scoped),
            "shadow" => Some(AccessMode::Shadow),
            _ => None,
        }
    }

    /// The desktop's config key: absent or empty is the default (`legacy`); anything else that is not a
    /// mode is `legacy` too, and the second answer says so (a typo must not stop the desktop, and must not
    /// be silent).
    pub fn from_config(value: Option<&str>) -> (AccessMode, Option<String>) {
        match value.map(str::trim).filter(|v| !v.is_empty()) {
            None => (AccessMode::Legacy, None),
            Some(v) => match AccessMode::parse(v) {
                Some(mode) => (mode, None),
                None => (
                    AccessMode::Legacy,
                    Some(format!(
                        "accessMode {v:?} is not legacy, scoped or shadow: using legacy"
                    )),
                ),
            },
        }
    }

    /// `OAIY_ACCESS_MODE` for the server: absent or empty is the default, and a value that is not a mode
    /// is a refusal to start (a typo must not leave a server on a mode nobody chose).
    pub fn from_env(value: Option<&str>) -> Result<AccessMode, ConfigRefusal> {
        match value.map(str::trim).filter(|v| !v.is_empty()) {
            None => Ok(AccessMode::Legacy),
            Some(v) => AccessMode::parse(v).ok_or_else(|| {
                ConfigRefusal(format!(
                    "OAIY_ACCESS_MODE={v:?} is not legacy, scoped or shadow"
                ))
            }),
        }
    }

    /// Whether the new guard judges every route (`scoped`, `shadow`), not only the routes the model adds.
    pub fn is_enforcing(self) -> bool {
        !matches!(self, AccessMode::Legacy)
    }
}

/// How an install can be reached, computed once and logged at start.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Exposure {
    /// Bound to loopback, no public URL.
    Local,
    /// `OAIY_PUBLIC_URL` is set: behind a reverse proxy.
    Proxied,
    /// Bound beyond loopback with no public URL.
    Lan,
}

impl Exposure {
    pub fn compute(bind_all: bool, public_url_set: bool) -> Exposure {
        if public_url_set {
            Exposure::Proxied
        } else if bind_all {
            Exposure::Lan
        } else {
            Exposure::Local
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Exposure::Local => "local",
            Exposure::Proxied => "proxied",
            Exposure::Lan => "lan",
        }
    }
}

/// A configuration OAIY refuses to start with. The message says what to change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigRefusal(pub String);

impl ConfigRefusal {
    pub fn exit_code(&self) -> i32 {
        EX_CONFIG
    }
}

impl std::fmt::Display for ConfigRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigRefusal {}

/// The refusals that concern the mode (design 4.5.5 rule 6):
///
/// - `shadow` on a proxied or LAN install: a valid `readonly` token must not become a `cli` token on the
///   internet.
/// - `legacy` when an owner login exists (`<data>/auth/owner.json`): a login means the install has moved
///   past Origin trust.
/// - `legacy` when the exposure is not local: Origin trust on an address other machines can reach. (Until
///   the startup rules of `auth::exposure` this was not refused, because a LAN server with a bearer token
///   was a supported install and had no login to move to; a LAN install now needs an owner login, made
///   with `oaiy-server auth init`, and runs `scoped`.)
pub fn validate_mode(
    mode: AccessMode,
    exposure: Exposure,
    owner_exists: bool,
) -> Result<(), ConfigRefusal> {
    match mode {
        AccessMode::Shadow if exposure != Exposure::Local => Err(ConfigRefusal(format!(
            "OAIY_ACCESS_MODE=shadow is refused on a {} install: it would let a token past a scope it lacks on a reachable address; use scoped",
            exposure.name()
        ))),
        AccessMode::Legacy if owner_exists => Err(ConfigRefusal(
            "OAIY_ACCESS_MODE=legacy is refused because an owner login exists (<data>/auth/owner.json): use scoped".into(),
        )),
        AccessMode::Legacy if exposure != Exposure::Local => Err(ConfigRefusal(format!(
            "OAIY_ACCESS_MODE=legacy is refused on a {} install: use scoped",
            exposure.name()
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mode_is_one_of_three_names() {
        for (text, mode) in [
            ("legacy", AccessMode::Legacy),
            ("scoped", AccessMode::Scoped),
            ("shadow", AccessMode::Shadow),
            (" Scoped\n", AccessMode::Scoped),
            ("SHADOW", AccessMode::Shadow),
        ] {
            assert_eq!(AccessMode::parse(text), Some(mode), "{text:?}");
            assert_eq!(AccessMode::parse(mode.name()), Some(mode));
        }
        for bad in [
            "",
            "scope",
            "on",
            "true",
            "strict",
            "legacy;scoped",
            "l egacy",
            "0",
        ] {
            assert_eq!(AccessMode::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_default_is_legacy_and_only_scoped_and_shadow_enforce() {
        assert_eq!(AccessMode::from_config(None), (AccessMode::Legacy, None));
        assert_eq!(
            AccessMode::from_config(Some("")),
            (AccessMode::Legacy, None)
        );
        assert_eq!(
            AccessMode::from_config(Some("  ")),
            (AccessMode::Legacy, None)
        );
        assert_eq!(
            AccessMode::from_config(Some("scoped")),
            (AccessMode::Scoped, None)
        );
        assert!(
            !AccessMode::Legacy.is_enforcing()
                && AccessMode::Scoped.is_enforcing()
                && AccessMode::Shadow.is_enforcing()
        );
    }

    #[test]
    fn the_desktop_keeps_running_on_a_mistyped_mode_and_says_so_and_the_server_refuses_to_start() {
        let (mode, warning) = AccessMode::from_config(Some("scopd"));
        assert_eq!(mode, AccessMode::Legacy);
        assert!(warning.unwrap().contains("scopd"));
        assert_eq!(AccessMode::from_env(None), Ok(AccessMode::Legacy));
        assert_eq!(AccessMode::from_env(Some("shadow")), Ok(AccessMode::Shadow));
        let refusal = AccessMode::from_env(Some("scopd")).unwrap_err();
        assert!(
            refusal.to_string().contains("OAIY_ACCESS_MODE")
                && refusal.to_string().contains("scopd")
        );
        assert_eq!(refusal.exit_code(), 78);
    }

    #[test]
    fn the_exposure_follows_the_bind_and_the_public_url() {
        assert_eq!(Exposure::compute(false, false), Exposure::Local);
        assert_eq!(Exposure::compute(true, false), Exposure::Lan);
        assert_eq!(Exposure::compute(false, true), Exposure::Proxied);
        assert_eq!(
            Exposure::compute(true, true),
            Exposure::Proxied,
            "a bind beyond loopback behind a proxy is a proxied install"
        );
    }

    #[test]
    fn shadow_is_refused_where_a_reachable_address_could_meet_it() {
        assert!(validate_mode(AccessMode::Shadow, Exposure::Local, false).is_ok());
        for exposure in [Exposure::Proxied, Exposure::Lan] {
            let why = validate_mode(AccessMode::Shadow, exposure, false).unwrap_err();
            assert!(
                why.to_string().contains("shadow") && why.to_string().contains(exposure.name()),
                "{why}"
            );
            assert_eq!(why.exit_code(), 78);
            assert!(validate_mode(AccessMode::Scoped, exposure, false).is_ok());
        }
    }

    #[test]
    fn legacy_is_refused_once_an_owner_login_exists() {
        assert!(validate_mode(AccessMode::Legacy, Exposure::Local, false).is_ok());
        assert!(validate_mode(AccessMode::Legacy, Exposure::Local, true).is_err());
        assert!(validate_mode(AccessMode::Scoped, Exposure::Local, true).is_ok());
        assert!(validate_mode(AccessMode::Shadow, Exposure::Local, true).is_ok());
    }

    /// Design 4.3 and 4.5.5 rule 6: `legacy` is refused off a local install. This test used to pin the
    /// opposite (a LAN server with a bearer token had no login to move to until `oaiy-server auth init`);
    /// ACC-14 flips it on purpose.
    #[test]
    fn legacy_is_refused_off_a_local_install_because_it_trusts_origins_other_machines_can_send() {
        assert!(validate_mode(AccessMode::Legacy, Exposure::Local, false).is_ok());
        for exposure in [Exposure::Proxied, Exposure::Lan] {
            let why = validate_mode(AccessMode::Legacy, exposure, false).unwrap_err();
            assert!(
                why.to_string().contains("legacy") && why.to_string().contains(exposure.name()),
                "{why}"
            );
            assert_eq!(why.exit_code(), 78);
            // With or without an owner, and the modes that judge every route are not refused for it.
            assert!(validate_mode(AccessMode::Legacy, exposure, true).is_err());
            assert!(validate_mode(AccessMode::Scoped, exposure, false).is_ok());
        }
    }
}
