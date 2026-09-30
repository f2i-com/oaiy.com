//! Presets: the named bundles of scopes a person sees, and the ceiling of each app.
//!
//! Users and pairing requests see only these bundles, and a host's session ceiling is the same
//! bundle. The scope table stays the conformance artifact. A preset is expanded to exact scopes when
//! a credential is created (a `pat` keeps exactly what it was approved with); a session or desk
//! credential stores `(app, preset)` and its scopes are recomputed from these tables at every use.

use super::scopes::ScopeSet;

/// A named bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    /// All 54. The dashboard's `desk`, and a cookie session on the dashboard host (dangerous scopes
    /// then need step-up). Not grantable to a token.
    Owner,
    /// The Agent page.
    Agent,
    /// The flow editor's own window on the desktop.
    Flows,
    /// The flow editor host of a server (a cookie session).
    FlowsHost,
    /// The hosted editor (oaiy.com) paired to a desktop.
    FlowsWeb,
    /// FormLogic Web.
    Formlogic,
    /// The `oaiy` CLI and scripts, and the static token everywhere.
    Cli,
    /// A native token minted with elevation or on the console, 24 hours.
    CliAdmin,
    /// An external MCP client.
    Mcp,
    /// Dashboards and monitors.
    Readonly,
    /// The Aokie Companion on a LAN, interim.
    Companion,
    /// The per-run credential of an approved flow's CLI child.
    Run,
    /// The vault ceremony token.
    Ceremony,
}

pub const ALL_PRESETS: [Preset; 13] = [
    Preset::Owner,
    Preset::Agent,
    Preset::Flows,
    Preset::FlowsHost,
    Preset::FlowsWeb,
    Preset::Formlogic,
    Preset::Cli,
    Preset::CliAdmin,
    Preset::Mcp,
    Preset::Readonly,
    Preset::Companion,
    Preset::Run,
    Preset::Ceremony,
];

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Preset::Owner => "owner",
            Preset::Agent => "agent",
            Preset::Flows => "flows",
            Preset::FlowsHost => "flows-host",
            Preset::FlowsWeb => "flows-web",
            Preset::Formlogic => "formlogic",
            Preset::Cli => "cli",
            Preset::CliAdmin => "cli-admin",
            Preset::Mcp => "mcp",
            Preset::Readonly => "readonly",
            Preset::Companion => "companion",
            Preset::Run => "run",
            Preset::Ceremony => "ceremony",
        }
    }

    pub fn by_name(name: &str) -> Option<Preset> {
        ALL_PRESETS.iter().copied().find(|p| p.name() == name)
    }

    /// Bumped when a preset's bundle changes; a session or desk credential records the version it
    /// was made under so that Connections can say a release changed it.
    pub const VERSION: u32 = 1;

    /// Whether a pairing request or a token mint may name this preset (`owner` never; `run` is made
    /// by the host for a flow run, not asked for).
    pub fn grantable_on_request(self) -> bool {
        matches!(
            self,
            Preset::Agent
                | Preset::Flows
                | Preset::FlowsWeb
                | Preset::Formlogic
                | Preset::Cli
                | Preset::Mcp
                | Preset::Readonly
                | Preset::Companion
                | Preset::Ceremony
        )
    }

    /// The scopes of the bundle.
    pub fn scopes(self) -> ScopeSet {
        match self {
            Preset::Owner => ScopeSet::all(),
            Preset::Agent => ScopeSet::of(&[
                "plugins.read",
                "events.read",
                "flows.read",
                "flows.write",
                "runs.read",
                "runs.write",
                "connectors.use",
                "ai.read",
                "ai.use",
                "speech.use",
                "calls.read",
                "calls.write",
                "calendar.read",
                "calendar.write",
                "contacts.read",
                "contacts.write",
                "agent.read",
                "agent.serve",
                "control.read",
                "control.project",
            ]),
            Preset::Flows => ScopeSet::of(&[
                "services.read",
                "services.control",
                "logs.read",
                "models.read",
                "plugins.read",
                "flows.read",
                "flows.write",
                "runs.read",
                "runs.write",
                "connectors.use",
                "ai.read",
                "ai.use",
                "speech.use",
                "agent.tasks",
            ]),
            Preset::FlowsHost => ScopeSet::of(&[
                "services.read",
                "services.control",
                "models.read",
                "plugins.read",
                "flows.read",
                "flows.write",
                "runs.read",
                "runs.write",
                "ai.read",
                "ai.use",
                "speech.use",
            ]),
            Preset::FlowsWeb => ScopeSet::of(&[
                "services.read",
                "services.control",
                "models.read",
                "plugins.read",
                "ai.read",
                "ai.use",
                "speech.use",
            ]),
            Preset::Formlogic => ScopeSet::of(&[
                "ai.read",
                "ai.use",
                "events.read",
                "connectors.use",
                "plugins.read",
                "plugins.control",
                "services.read",
                "services.control",
            ]),
            Preset::Cli => cli_scopes(),
            Preset::CliAdmin => cli_scopes().union(&ScopeSet::of(&[
                "services.define",
                "runtimes.install",
                "plugins.install",
                "flows.approve",
            ])),
            Preset::Mcp => ScopeSet::of(&["control.read", "control.project"]),
            Preset::Readonly => ScopeSet::of(&[
                "system.read",
                "services.read",
                "models.read",
                "plugins.read",
                "flows.read",
                "runs.read",
            ]),
            Preset::Companion => ScopeSet::of(&[
                "events.read",
                "calls.read",
                "plugins.read",
                "connectors.use",
            ]),
            Preset::Run => ScopeSet::of(&[
                "ai.read",
                "ai.use",
                "connectors.use",
                "services.read",
                "services.control",
                "agent.tasks",
                "flows.read",
                "speech.use",
            ]),
            Preset::Ceremony => ScopeSet::of(&["vault.kt"]),
        }
    }
}

fn cli_scopes() -> ScopeSet {
    ScopeSet::of(&[
        "system.read",
        "logs.read",
        "services.read",
        "services.control",
        "models.read",
        "models.write",
        "plugins.read",
        "plugins.control",
        "flows.read",
        "flows.write",
        "runs.read",
        "runs.write",
        "ai.read",
        "ai.use",
        "events.read",
    ])
}

/// The three apps a cookie session or a desktop webview can be for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum App {
    /// The dashboard.
    Dash,
    Agent,
    Flows,
}

impl App {
    pub fn name(self) -> &'static str {
        match self {
            App::Dash => "dash",
            App::Agent => "agent",
            App::Flows => "flows",
        }
    }

    pub fn by_name(name: &str) -> Option<App> {
        [App::Dash, App::Agent, App::Flows]
            .into_iter()
            .find(|a| a.name() == name)
    }

    /// The most a cookie session on this app's host can hold: `owner` on the dashboard host,
    /// `agent` on the Agent host, `flows-host` on the flow editor host.
    pub fn session_ceiling(self) -> Preset {
        match self {
            App::Dash => Preset::Owner,
            App::Agent => Preset::Agent,
            App::Flows => Preset::FlowsHost,
        }
    }

    /// The most a desktop webview's `desk` credential can hold: the flow editor's window has the
    /// wider `flows` bundle (connectors, logs, asking the Agent) that its host build does not.
    pub fn desk_ceiling(self) -> Preset {
        match self {
            App::Dash => Preset::Owner,
            App::Agent => Preset::Agent,
            App::Flows => Preset::Flows,
        }
    }
}

/// What a relay principal (a phone acting through the owner-run relay) holds, by tier. No tier holds
/// a dangerous scope. Their connector authority is by tier at the connector gate, not by scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayTier {
    Read,
    CallControl,
    /// Only with a confirmed step-up key.
    Admin,
}

pub const ALL_TIERS: [RelayTier; 3] = [RelayTier::Read, RelayTier::CallControl, RelayTier::Admin];

impl RelayTier {
    pub fn name(self) -> &'static str {
        match self {
            RelayTier::Read => "read",
            RelayTier::CallControl => "call-control",
            RelayTier::Admin => "admin",
        }
    }

    pub fn scopes(self) -> ScopeSet {
        let read = ScopeSet::of(&[
            "plugins.read",
            "calls.read",
            "calendar.read",
            "contacts.read",
            "services.read",
            "link.read",
            "agent.read",
            "control.read",
            "connectors.use",
        ]);
        let call_control = read.union(&ScopeSet::of(&["calls.write", "agent.tasks"]));
        match self {
            RelayTier::Read => read,
            RelayTier::CallControl => call_control,
            RelayTier::Admin => call_control.union(&ScopeSet::of(&["control.project"])),
        }
    }
}

/// `control.project` is the change level of the control MCP and always travels with `control.read`
/// (a credential without the read level cannot reach `/api/mcp` at all).
pub fn control_project_needs_read(scopes: &ScopeSet) -> bool {
    !scopes.contains("control.project") || scopes.contains("control.read")
}

/// Every scope table in one list, for the exported file and the checks that run over all of them.
pub fn all_bundles() -> Vec<(String, ScopeSet)> {
    let mut out: Vec<(String, ScopeSet)> = ALL_PRESETS
        .iter()
        .map(|p| (p.name().to_string(), p.scopes()))
        .collect();
    out.extend(
        ALL_TIERS
            .iter()
            .map(|t| (format!("relay:{}", t.name()), t.scopes())),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::routes::{Class, ROUTES};
    use crate::auth::scopes::{is_dangerous, is_reserved, SCOPES};

    #[test]
    fn the_presets_have_the_sizes_the_design_gives() {
        let sizes: Vec<(&str, usize)> = ALL_PRESETS
            .iter()
            .map(|p| (p.name(), p.scopes().len()))
            .collect();
        assert_eq!(
            sizes,
            [
                ("owner", 54),
                ("agent", 20),
                ("flows", 14),
                ("flows-host", 11),
                ("flows-web", 7),
                ("formlogic", 8),
                ("cli", 15),
                ("cli-admin", 19),
                ("mcp", 2),
                ("readonly", 6),
                ("companion", 4),
                ("run", 8),
                ("ceremony", 1),
            ]
        );
    }

    #[test]
    fn only_owner_and_cli_admin_hold_a_dangerous_scope() {
        for p in ALL_PRESETS {
            let dangerous: Vec<String> = p
                .scopes()
                .names()
                .into_iter()
                .filter(|n| is_dangerous(n))
                .collect();
            match p {
                Preset::Owner => assert_eq!(dangerous.len(), 14),
                Preset::CliAdmin => {
                    assert_eq!(
                        dangerous,
                        [
                            "services.define",
                            "runtimes.install",
                            "plugins.install",
                            "flows.approve"
                        ]
                    );
                }
                _ => assert!(dangerous.is_empty(), "{} holds {dangerous:?}", p.name()),
            }
        }
        for t in ALL_TIERS {
            assert!(
                !t.scopes().has_dangerous(),
                "relay tier {} holds a dangerous scope",
                t.name()
            );
        }
    }

    #[test]
    fn no_preset_names_a_reserved_scope_but_owner_and_ceremony() {
        for p in ALL_PRESETS {
            let reserved: Vec<String> = p
                .scopes()
                .names()
                .into_iter()
                .filter(|n| is_reserved(n))
                .collect();
            match p {
                Preset::Owner => assert_eq!(reserved.len(), 6),
                Preset::Ceremony => assert_eq!(reserved, ["vault.kt"]),
                _ => assert!(reserved.is_empty(), "{} names {reserved:?}", p.name()),
            }
        }
    }

    #[test]
    fn control_project_always_travels_with_control_read() {
        for (name, scopes) in all_bundles() {
            assert!(
                control_project_needs_read(&scopes),
                "{name} has control.project without control.read"
            );
        }
        assert!(!control_project_needs_read(&ScopeSet::of(&[
            "control.project"
        ])));
        assert!(control_project_needs_read(&ScopeSet::of(&["control.read"])));
        assert!(control_project_needs_read(&ScopeSet::empty()));
    }

    #[test]
    fn the_relay_tiers_are_the_ones_the_mobile_design_gives_and_grow_by_one_step() {
        let (read, call, admin) = (
            RelayTier::Read.scopes(),
            RelayTier::CallControl.scopes(),
            RelayTier::Admin.scopes(),
        );
        assert_eq!((read.len(), call.len(), admin.len()), (9, 11, 12));
        assert!(read.is_subset_of(&call) && call.is_subset_of(&admin));
        assert_eq!(
            call.names()
                .into_iter()
                .filter(|n| !read.contains(n))
                .collect::<Vec<_>>(),
            ["calls.write", "agent.tasks"]
        );
        assert!(admin.contains("control.project") && !call.contains("control.project"));
    }

    #[test]
    fn a_preset_is_found_by_its_name_and_only_the_named_ones_can_be_asked_for() {
        for p in ALL_PRESETS {
            assert_eq!(Preset::by_name(p.name()), Some(p));
        }
        assert_eq!(Preset::by_name("admin"), None);
        assert_eq!(Preset::by_name("OWNER"), None);
        assert_eq!(Preset::by_name(""), None);
        let asked: Vec<&str> = ALL_PRESETS
            .iter()
            .filter(|p| p.grantable_on_request())
            .map(|p| p.name())
            .collect();
        assert_eq!(
            asked,
            [
                "agent",
                "flows",
                "flows-web",
                "formlogic",
                "cli",
                "mcp",
                "readonly",
                "companion",
                "ceremony"
            ]
        );
        assert!(
            !Preset::Owner.grantable_on_request()
                && !Preset::CliAdmin.grantable_on_request()
                && !Preset::Run.grantable_on_request()
        );
    }

    #[test]
    fn app_ceilings_are_the_design_s() {
        assert_eq!(App::Dash.session_ceiling(), Preset::Owner);
        assert_eq!(App::Agent.session_ceiling(), Preset::Agent);
        assert_eq!(App::Flows.session_ceiling(), Preset::FlowsHost);
        assert_eq!(
            App::Flows.desk_ceiling(),
            Preset::Flows,
            "the editor's own window has the wider bundle than its host build"
        );
        // The host build of the flow editor holds no dangerous scope and no connectors.
        assert!(!App::Flows
            .session_ceiling()
            .scopes()
            .contains("connectors.use"));
        for app in [App::Agent, App::Flows] {
            assert!(
                !app.session_ceiling().scopes().has_dangerous()
                    && !app.desk_ceiling().scopes().has_dangerous()
            );
        }
        assert_eq!(App::by_name("dash"), Some(App::Dash));
        assert_eq!(App::by_name("dashboard"), None);
    }

    /// How many of the routes that existed before the access model a preset can call: the routes that
    /// need no scope (public and any-credential) count for everyone.
    fn reach(scopes: &ScopeSet) -> usize {
        ROUTES
            .iter()
            .filter(|r| r.since == 1)
            .filter(|r| match r.class {
                Class::Public | Class::AnyCredential => true,
                Class::Scope(s) => scopes.contains(s),
                _ => false,
            })
            .count()
    }

    /// The design's coverage numbers (161 routes) plus what the code has gained since: the update
    /// status route (`system.read`), the update check (`services.control`: an action that reaches out to
    /// GitHub, so the `flows`, `flows-host`, `flows-web`, `formlogic` and `run` presets, which hold that
    /// scope, gain a route and `readonly` loses the one it had), the Agent's flush acknowledgement
    /// (`agent.serve`) and the plugin trust route (`plugins.install`), and the eleven routes of the
    /// receptionist's transfers and messages, which are `calls.read` (five: the messages, one message, the
    /// ring's settings, its preview and the rings going now) or `calls.write` (six): so `owner` and `agent`,
    /// which hold both, reach eleven more, and `companion` (the interim LAN preset), which holds
    /// `calls.read` alone, five more. No other preset holds either scope.
    #[test]
    fn what_each_preset_reaches_of_the_routes_that_existed() {
        let reaches: Vec<(&str, usize)> = ALL_PRESETS
            .iter()
            .map(|p| (p.name(), reach(&p.scopes())))
            .collect();
        assert_eq!(
            reaches,
            [
                ("owner", 165 + 11),
                ("agent", 78 + 11),
                ("flows", 64),
                ("flows-host", 54),
                ("flows-web", 38),
                ("formlogic", 36),
                ("cli", 75),
                ("cli-admin", 86),
                ("mcp", 7),
                ("readonly", 28),
                ("companion", 17 + 5),
                ("run", 32),
                ("ceremony", 4),
            ]
        );
    }

    #[test]
    fn the_routes_the_appendix_does_not_have_are_classified_on_purpose_and_a_monitor_reaches_only_the_read(
    ) {
        use crate::auth::routes::{lookup, Verb};
        let can = |preset: Preset, verb: Verb, pattern: &str| -> bool {
            match lookup(verb, pattern).unwrap().class {
                Class::Scope(s) => preset.scopes().contains(s),
                other => panic!("{other:?}"),
            }
        };
        // The update check asks GitHub for a release: an action. The monitor's token cannot; the operator's
        // (`cli`, the static token) and the dashboard's can, as `docs/UPDATES.md` shows.
        for (preset, expected) in [
            (Preset::Readonly, false),
            (Preset::Mcp, false),
            (Preset::Agent, false),
            (Preset::Cli, true),
            (Preset::CliAdmin, true),
            (Preset::Owner, true),
        ] {
            assert_eq!(
                can(preset, Verb::Post, "/api/update/check"),
                expected,
                "{}",
                preset.name()
            );
        }
        // Its status is a read: the monitor has it.
        assert!(can(Preset::Readonly, Verb::Get, "/api/update/status"));
        // The Agent page acknowledges "save your work" and nothing else of the updater's is its own.
        assert!(can(Preset::Agent, Verb::Post, "/api/update/agent-flushed"));
        assert!(!can(Preset::Cli, Verb::Post, "/api/update/agent-flushed"));
        assert!(!can(
            Preset::Readonly,
            Verb::Post,
            "/api/update/agent-flushed"
        ));
        // Trusting a plugin is trusting native code: dangerous, so a token of `cli` cannot and `cli-admin` can.
        assert!(!can(Preset::Cli, Verb::Post, "/api/plugins/:id/trust"));
        assert!(can(Preset::CliAdmin, Verb::Post, "/api/plugins/:id/trust"));
        assert!(!can(Preset::Agent, Verb::Post, "/api/plugins/:id/trust"));
    }

    #[test]
    fn every_scope_a_route_row_names_exists_and_every_scope_has_a_route_but_two() {
        let mut used = std::collections::BTreeSet::new();
        for r in ROUTES {
            if let Class::Scope(s) = r.class {
                assert!(
                    crate::auth::scopes::is_known(s),
                    "{} names {s}, which is not a scope",
                    r.key()
                );
                used.insert(s);
            }
        }
        // `control.project` is the change level of POST /api/mcp and `relay.manage` is the relay
        // design's to route: neither has a row of its own.
        for s in SCOPES.iter().map(|s| s.name) {
            if !used.contains(s) {
                assert!(
                    matches!(s, "control.project" | "relay.manage"),
                    "scope {s} has no route"
                );
            }
        }
    }

    #[test]
    fn a_preset_that_holds_a_scope_reaches_the_routes_of_it() {
        // Spot checks that the table and the presets meet: what the Agent's page calls.
        let agent = Preset::Agent.scopes();
        for (verb, pattern) in [
            (crate::auth::routes::Verb::Post, "/api/mcp"),
            (crate::auth::routes::Verb::Get, "/api/agent/events"),
            (crate::auth::routes::Verb::Post, "/api/bridge/leases/:name"),
            (crate::auth::routes::Verb::Post, "/api/update/agent-flushed"),
            (crate::auth::routes::Verb::Put, "/api/contacts/:number"),
        ] {
            let row = crate::auth::routes::lookup(verb, pattern).unwrap();
            match row.class {
                Class::Scope(s) => {
                    assert!(agent.contains(s), "the Agent lacks {s} for {}", row.key())
                }
                other => panic!("{other:?}"),
            }
        }
        // And what it must not call.
        let install =
            crate::auth::routes::lookup(crate::auth::routes::Verb::Post, "/api/plugins/install")
                .unwrap();
        assert_eq!(install.class, Class::Scope("plugins.install"));
        assert!(!agent.contains("plugins.install"));
    }
}
