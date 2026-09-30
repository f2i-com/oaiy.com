//! The principal: who a request is, once it has been authenticated.
//!
//! The pipeline never sees "a token", only a principal with a kind, a scope set, the origins it is
//! bound to, an elevation flag and a chain of parents. A principal holds no secret: not the token,
//! not its hash.

use super::presets::App;
use super::scopes::ScopeSet;

/// What kind of credential made the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrincipalKind {
    /// A cookie session.
    Session,
    /// A desktop webview's credential.
    Desk,
    /// A paired app or tool.
    Pat,
    /// A per-run or derived credential.
    Run,
    /// The console.
    Console,
    /// `OAIY_SERVER_TOKEN`: never more than the `cli` preset.
    Static,
    /// The owner, as the old guard sees them (`legacy` access mode only).
    Legacy,
}

impl PrincipalKind {
    /// The spelling the audit log and `whoami` use.
    pub fn name(self) -> &'static str {
        match self {
            PrincipalKind::Session => "session",
            PrincipalKind::Desk => "desk",
            PrincipalKind::Pat => "pat",
            PrincipalKind::Run => "run",
            PrincipalKind::Console => "con",
            PrincipalKind::Static => "static",
            PrincipalKind::Legacy => "legacy",
        }
    }

    /// `desk`, `run` and `con` credentials are worth something only on the machine they were made
    /// on: they are refused unless the peer is loopback and the request carries none of the
    /// forwarded headers.
    pub fn needs_direct_loopback(self) -> bool {
        matches!(
            self,
            PrincipalKind::Desk | PrincipalKind::Run | PrincipalKind::Console
        )
    }
}

/// The level of the control MCP a credential's scopes give (design 4.10.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ControlLevel {
    /// `/api/mcp` answers `403 insufficient_scope`.
    None,
    /// `control.read` only: the read tools.
    Read,
    /// `control.project`, always with `control.read`: the change tools too.
    Project,
}

/// Who is asking. Cloned into each request as an extension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    /// The credential's id (16 hex), or `static` / `legacy` for the two that are not stored.
    pub id: String,
    pub kind: PrincipalKind,
    pub label: String,
    /// Effective scopes, computed at authentication.
    pub scopes: ScopeSet,
    /// The origins the credential is bound to; empty means unbound.
    pub origins: Vec<String>,
    pub app: Option<App>,
    /// A dangerous scope is usable without a step-up.
    pub elevated: bool,
    /// The ids of its ancestors, nearest first.
    pub chain: Vec<String>,
    /// Survives a restart.
    pub persisted: bool,
    /// The credential's own expiry, if it has one.
    pub expires_ms: Option<u64>,
    /// The preset it was made from, if any.
    pub preset: Option<String>,
    /// Imported from a plaintext pairing file.
    pub legacy_import: bool,
}

impl Principal {
    /// Whether the principal holds exactly this scope.
    pub fn has(&self, scope: &str) -> bool {
        self.scopes.contains(scope)
    }

    pub fn control_level(&self) -> ControlLevel {
        if self.has("control.project") && self.has("control.read") {
            ControlLevel::Project
        } else if self.has("control.read") {
            ControlLevel::Read
        } else {
            ControlLevel::None
        }
    }

    /// Whether the credential is bound to the origins it was made for.
    pub fn is_bound(&self) -> bool {
        !self.origins.is_empty()
    }

    /// The owner as the old guard sees them. The old guard inserts it so that a handler that reads
    /// `Extension<Principal>` works in the `legacy` access mode; it carries every scope and is
    /// elevated, because in that mode any accepted credential is admin.
    pub fn legacy_owner() -> Principal {
        Principal {
            id: "legacy".into(),
            kind: PrincipalKind::Legacy,
            label: "legacy access".into(),
            scopes: ScopeSet::all(),
            origins: Vec::new(),
            app: None,
            elevated: true,
            chain: Vec::new(),
            persisted: false,
            expires_ms: None,
            preset: Some("owner".into()),
            legacy_import: false,
        }
    }

    /// The operator's environment token: the `cli` preset on every install, never a dangerous scope.
    pub fn static_token() -> Principal {
        Principal {
            id: STATIC_ID.into(),
            kind: PrincipalKind::Static,
            label: "OAIY_SERVER_TOKEN".into(),
            scopes: super::presets::Preset::Cli.scopes(),
            origins: Vec::new(),
            app: None,
            elevated: false,
            chain: Vec::new(),
            persisted: false,
            expires_ms: None,
            preset: Some("cli".into()),
            legacy_import: false,
        }
    }

    /// `{id, kind, label}`: what the audit log records about who did something.
    pub fn actor(&self) -> Actor {
        Actor {
            id: self.id.clone(),
            kind: self.kind.name().to_string(),
            label: self.label.clone(),
        }
    }
}

/// The id of the static token as a parent of a derived credential.
pub const STATIC_ID: &str = "static";

/// Who did something, for the audit log.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Actor {
    pub id: String,
    pub kind: String,
    pub label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_legacy_owner_holds_everything_and_is_elevated_and_holds_no_secret() {
        let p = Principal::legacy_owner();
        assert_eq!(p.scopes.len(), 56);
        assert!(p.elevated && p.has("plugins.install") && p.has("vault.kt"));
        assert_eq!(p.kind, PrincipalKind::Legacy);
        assert!(!p.is_bound());
        assert_eq!(p.control_level(), ControlLevel::Project);
        let shown = format!("{p:?}");
        assert!(
            !shown.contains("oaiy") || shown.contains("legacy"),
            "{shown}"
        );
    }

    #[test]
    fn the_static_token_is_the_cli_preset_and_never_more() {
        let p = Principal::static_token();
        assert_eq!(p.scopes.len(), 15);
        assert!(!p.scopes.has_dangerous());
        assert!(!p.elevated);
        assert!(p.has("runs.write") && !p.has("auth.read") && !p.has("flows.approve"));
        assert_eq!(p.control_level(), ControlLevel::None);
    }

    #[test]
    fn the_control_level_follows_the_scopes() {
        let mut p = Principal::static_token();
        p.scopes = ScopeSet::of(&["control.read"]);
        assert_eq!(p.control_level(), ControlLevel::Read);
        p.scopes = ScopeSet::of(&["control.read", "control.project"]);
        assert_eq!(p.control_level(), ControlLevel::Project);
        // The change level without the read level is no level at all.
        p.scopes = ScopeSet::of(&["control.project"]);
        assert_eq!(p.control_level(), ControlLevel::None);
        assert!(
            ControlLevel::None < ControlLevel::Read && ControlLevel::Read < ControlLevel::Project
        );
    }

    #[test]
    fn the_kinds_that_need_a_direct_loopback_peer_are_desk_run_and_console() {
        for k in [
            PrincipalKind::Desk,
            PrincipalKind::Run,
            PrincipalKind::Console,
        ] {
            assert!(k.needs_direct_loopback(), "{k:?}");
        }
        for k in [
            PrincipalKind::Session,
            PrincipalKind::Pat,
            PrincipalKind::Static,
            PrincipalKind::Legacy,
        ] {
            assert!(!k.needs_direct_loopback(), "{k:?}");
        }
        assert_eq!(PrincipalKind::Console.name(), "con");
    }

    #[test]
    fn the_actor_is_what_the_audit_log_names() {
        let a = Principal::static_token().actor();
        assert_eq!(
            (a.id.as_str(), a.kind.as_str(), a.label.as_str()),
            ("static", "static", "OAIY_SERVER_TOKEN")
        );
    }
}
