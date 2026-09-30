//! The route table: every `(method, path)` the API answers, and what it takes to call it.
//!
//! This is the source of truth for the access model. The guard of a later step keys this table on
//! `(Method, MatchedPath)`: the route pattern axum records before an endpoint's layers run, so there
//! is no second matcher and no precedence code to keep in step with the router (`/api/plugins/install`
//! and `/api/plugins/:id` are different strings). A request whose route has no row here is refused
//! in `scoped` mode (`403 unclassified_route`): a route added without a row is unusable until someone
//! classifies it, and `cargo test` fails until they do (`route_coverage`).
//!
//! - [`Class::Public`]: no credential (health, capability discovery, the pairing bootstrap and the
//!   login routes). `OPTIONS` is public for every path.
//! - [`Class::AnyCredential`]: any live credential; the handler adds its own kind rules.
//! - [`Class::Console`]: the console credential only.
//! - [`Class::Session`]: a live cookie session with CSRF; the handler adds its own rule.
//! - [`Class::Desk`]: a desktop webview's credential, by role.
//! - [`Class::Scope`]: the credential must hold that exact scope.
//!
//! `since` is 1 for a route that existed before the access model and 2 for a route it adds or that
//! another design announced (a reserved row with no handler yet). `only` marks the rows of routes
//! that exist only in the headless build.
//!
//! Nothing consults this table yet in the `legacy` access mode for a route with `since: 1`: those
//! keep the guard they have always had.

use std::collections::HashMap;
use std::sync::OnceLock;

/// A method as the table spells it. `Head` is answered from the `Get` row and `Options` is public
/// for every path, so neither has a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Verb {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    /// Every method (`axum::routing::any`).
    Any,
}

impl Verb {
    pub fn as_str(self) -> &'static str {
        match self {
            Verb::Get => "GET",
            Verb::Post => "POST",
            Verb::Put => "PUT",
            Verb::Patch => "PATCH",
            Verb::Delete => "DELETE",
            Verb::Any => "ANY",
        }
    }

    /// The row a request's method is looked up under; `HEAD` uses the `GET` row. `None` for a method
    /// the API does not serve (and for `OPTIONS`, which [`route_class`] answers by itself).
    pub fn of_method(method: &axum::http::Method) -> Option<Verb> {
        use axum::http::Method;
        match *method {
            Method::GET | Method::HEAD => Some(Verb::Get),
            Method::POST => Some(Verb::Post),
            Method::PUT => Some(Verb::Put),
            Method::PATCH => Some(Verb::Patch),
            Method::DELETE => Some(Verb::Delete),
            _ => None,
        }
    }
}

/// Which desktop webview a `desk` credential was made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeskRole {
    Dashboard,
    Agent,
    Flows,
}

/// What a route takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Public,
    AnyCredential,
    Console,
    Session {
        elevate: bool,
    },
    Desk {
        roles: &'static [DeskRole],
    },
    Scope(&'static str),
    /// What [`route_class`] answers for a route with no row. No row in [`ROUTES`] has this class, and
    /// every match on a class must refuse it.
    Unclassified,
}

/// Where a route exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Only {
    /// In every build.
    Everywhere,
    /// Only in the headless build.
    Server,
}

/// One row of the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Route {
    pub method: Verb,
    /// The axum route pattern: `/api/services/:id/start`, `/api/ai/engine/gateway/*path`.
    pub pattern: &'static str,
    pub class: Class,
    /// 1 for a route that existed before the access model, 2 for one it (or another design) adds.
    pub since: u8,
    pub only: Only,
}

impl Route {
    /// `GET /api/config`: how the tests and the exported file name a row.
    pub fn key(&self) -> String {
        format!("{} {}", self.method.as_str(), self.pattern)
    }
}

const DESK_ANY: &[DeskRole] = &[DeskRole::Dashboard, DeskRole::Agent, DeskRole::Flows];

const fn row(method: Verb, pattern: &'static str, class: Class, since: u8, only: Only) -> Route {
    Route {
        method,
        pattern,
        class,
        since,
        only,
    }
}
/// A route that existed before the access model, open to anyone.
const fn public(method: Verb, pattern: &'static str) -> Route {
    row(method, pattern, Class::Public, 1, Only::Everywhere)
}
/// A route the access model adds, open to anyone.
const fn public_new(method: Verb, pattern: &'static str) -> Route {
    row(method, pattern, Class::Public, 2, Only::Everywhere)
}
const fn any_credential_new(method: Verb, pattern: &'static str) -> Route {
    row(method, pattern, Class::AnyCredential, 2, Only::Everywhere)
}
const fn console_new(method: Verb, pattern: &'static str) -> Route {
    row(method, pattern, Class::Console, 2, Only::Everywhere)
}
const fn desk_rsv(method: Verb, pattern: &'static str) -> Route {
    row(
        method,
        pattern,
        Class::Desk { roles: DESK_ANY },
        2,
        Only::Everywhere,
    )
}
const fn session_new(method: Verb, pattern: &'static str) -> Route {
    row(
        method,
        pattern,
        Class::Session { elevate: false },
        2,
        Only::Everywhere,
    )
}
/// A route that existed before the access model, taking `scope`.
const fn scope(method: Verb, pattern: &'static str, scope: &'static str) -> Route {
    row(method, pattern, Class::Scope(scope), 1, Only::Everywhere)
}
/// A route the access model adds, taking `scope`.
const fn scope_new(method: Verb, pattern: &'static str, scope: &'static str) -> Route {
    row(method, pattern, Class::Scope(scope), 2, Only::Everywhere)
}
/// A route another design announced, taking `scope`: classified here, built there.
const fn scope_rsv(method: Verb, pattern: &'static str, scope: &'static str) -> Route {
    row(method, pattern, Class::Scope(scope), 2, Only::Everywhere)
}
/// A route another design announced that exists only in the headless build.
const fn scope_rsv_server(method: Verb, pattern: &'static str, scope: &'static str) -> Route {
    row(method, pattern, Class::Scope(scope), 2, Only::Server)
}

/// Every route of the API. Grouped by scope, in the order of the design's Appendix B.
pub static ROUTES: &[Route] = &[
    // Public: no credential (the six /api/auth routes below carry no CORS headers in scoped mode).
    public(Verb::Get, "/api/health"),
    public(Verb::Get, "/api/bridge/capabilities"),
    public(Verb::Post, "/api/bridge/pairing"),
    public(Verb::Get, "/api/bridge/pairing/:id"),
    public_new(Verb::Get, "/api/auth/info"),
    public_new(Verb::Get, "/api/auth/session"),
    public_new(Verb::Post, "/api/auth/login"),
    public_new(Verb::Post, "/api/auth/setup"),
    public_new(Verb::Post, "/api/auth/link"),
    public_new(Verb::Post, "/api/auth/callback"),
    // Any live credential; the handler adds its own kind rules.
    any_credential_new(Verb::Get, "/api/auth/whoami"),
    any_credential_new(Verb::Post, "/api/auth/derive"),
    // The console credential only.
    console_new(Verb::Post, "/api/auth/console/reset-password"),
    console_new(Verb::Post, "/api/auth/console/setup-code"),
    console_new(Verb::Post, "/api/auth/console/session-link"),
    console_new(Verb::Post, "/api/auth/console/sessions/revoke-all"),
    console_new(Verb::Get, "/api/auth/console/status"),
    // Desk credentials only (the role is read from the credential).
    desk_rsv(Verb::Get, "/api/local-protection/webview-key"),
    // Cookie sessions with their own rule (4.7); no scope.
    session_new(Verb::Post, "/api/auth/logout"),
    session_new(Verb::Post, "/api/auth/elevate"),
    session_new(Verb::Post, "/api/auth/password"),
    session_new(Verb::Post, "/api/auth/handoff"),
    session_new(Verb::Post, "/api/auth/renew"),
    // system.read
    scope(Verb::Get, "/api/config", "system.read"),
    scope(Verb::Get, "/api/node", "system.read"),
    scope_new(Verb::Get, "/api/system/gpus", "system.read"),
    scope_new(Verb::Get, "/api/secrets/hf-token", "system.read"),
    scope(Verb::Get, "/api/update/status", "system.read"),
    scope(Verb::Post, "/api/update/check", "system.read"),
    // logs.read
    scope(Verb::Get, "/api/node/logs", "logs.read"),
    scope(Verb::Get, "/api/python/logs", "logs.read"),
    scope(Verb::Get, "/api/engines/logs", "logs.read"),
    scope(Verb::Get, "/api/control/desktop-log", "logs.read"),
    scope(Verb::Get, "/api/services/:id/logs", "logs.read"),
    scope(Verb::Get, "/api/plugins/:id/logs", "logs.read"),
    // services.read
    scope(Verb::Get, "/api/services", "services.read"),
    scope(Verb::Get, "/api/services/definitions", "services.read"),
    scope(Verb::Get, "/api/python", "services.read"),
    // services.control
    scope(
        Verb::Post,
        "/api/services/ensure-by-port",
        "services.control",
    ),
    scope(Verb::Post, "/api/services/:id/start", "services.control"),
    scope(Verb::Post, "/api/services/:id/stop", "services.control"),
    scope(Verb::Post, "/api/services/:id/repair", "services.control"),
    scope(
        Verb::Post,
        "/api/services/:id/autostart",
        "services.control",
    ),
    scope(Verb::Post, "/api/services/:id/install", "services.control"),
    scope(
        Verb::Post,
        "/api/services/:id/cancel-install",
        "services.control",
    ),
    scope_new(Verb::Put, "/api/services/:id/gpu", "services.control"),
    // services.define
    scope(Verb::Post, "/api/services", "services.define"),
    scope(Verb::Delete, "/api/services/:id", "services.define"),
    scope(Verb::Post, "/api/services/:id/uninstall", "services.define"),
    scope(Verb::Get, "/api/services/:id/export", "services.define"),
    // models.read
    scope(Verb::Get, "/api/models", "models.read"),
    scope(Verb::Get, "/api/models/catalog", "models.read"),
    scope(Verb::Get, "/api/models/downloads", "models.read"),
    scope(Verb::Get, "/api/engines", "models.read"),
    scope(Verb::Get, "/api/engines/catalog", "models.read"),
    scope(Verb::Get, "/api/engines/downloads", "models.read"),
    scope(Verb::Get, "/api/engines/recommendation", "models.read"),
    scope(Verb::Get, "/api/engines/defaults", "models.read"),
    // models.write
    scope(Verb::Post, "/api/models/download", "models.write"),
    scope(
        Verb::Post,
        "/api/models/downloads/:id/pause",
        "models.write",
    ),
    scope(
        Verb::Post,
        "/api/models/downloads/:id/resume",
        "models.write",
    ),
    scope(
        Verb::Post,
        "/api/models/downloads/:id/cancel",
        "models.write",
    ),
    scope(Verb::Delete, "/api/models/:name", "models.write"),
    scope(Verb::Post, "/api/engines/downloads", "models.write"),
    scope(Verb::Put, "/api/engines/defaults", "models.write"),
    scope(Verb::Post, "/api/engines/llm/:action", "models.write"),
    // runtimes.install
    scope(Verb::Post, "/api/python/install", "runtimes.install"),
    scope(Verb::Post, "/api/python/venvs", "runtimes.install"),
    scope(Verb::Delete, "/api/python/venvs/:name", "runtimes.install"),
    scope(Verb::Post, "/api/node/install", "runtimes.install"),
    // plugins.read
    scope(Verb::Get, "/api/plugins", "plugins.read"),
    scope(
        Verb::Get,
        "/api/plugins/:id/ui/:screen/*path",
        "plugins.read",
    ),
    scope(Verb::Get, "/api/modules", "plugins.read"),
    scope(Verb::Get, "/api/modules/events", "plugins.read"),
    scope(Verb::Get, "/api/bridge/status", "plugins.read"),
    // plugins.control
    scope(Verb::Post, "/api/plugins/:id/start", "plugins.control"),
    scope(Verb::Post, "/api/plugins/:id/stop", "plugins.control"),
    scope(Verb::Post, "/api/plugins/:id/enabled", "plugins.control"),
    // plugins.install
    scope(Verb::Post, "/api/plugins/install", "plugins.install"),
    scope(Verb::Delete, "/api/plugins/:id", "plugins.install"),
    scope(Verb::Post, "/api/plugins/:id/trust", "plugins.install"),
    // events.read
    scope(Verb::Get, "/api/bridge/events", "events.read"),
    scope(Verb::Get, "/api/bridge/deadletters", "events.read"),
    // flows.read
    scope(Verb::Get, "/api/bridge/flows", "flows.read"),
    scope(Verb::Get, "/api/bridge/flows/:id", "flows.read"),
    scope_new(Verb::Get, "/api/bridge/flows/:id/review", "flows.read"),
    scope(Verb::Get, "/api/bridge/triggers", "flows.read"),
    // flows.write
    scope(Verb::Put, "/api/bridge/flows/:id", "flows.write"),
    scope(Verb::Delete, "/api/bridge/flows/:id", "flows.write"),
    scope_new(
        Verb::Delete,
        "/api/bridge/flows/:id/approval",
        "flows.write",
    ),
    scope(Verb::Post, "/api/bridge/triggers", "flows.write"),
    scope(Verb::Delete, "/api/bridge/triggers/:id", "flows.write"),
    // flows.approve
    scope_new(Verb::Post, "/api/bridge/flows/:id/approve", "flows.approve"),
    // runs.read
    scope(Verb::Get, "/api/bridge/runs", "runs.read"),
    scope(Verb::Get, "/api/bridge/runs/:id", "runs.read"),
    // runs.write
    scope(Verb::Post, "/api/bridge/runs", "runs.write"),
    scope(Verb::Delete, "/api/bridge/runs", "runs.write"),
    scope(Verb::Post, "/api/bridge/runs/:id/claim", "runs.write"),
    scope(Verb::Post, "/api/bridge/runs/:id/finish", "runs.write"),
    scope(Verb::Post, "/api/bridge/runs/:id/cancel", "runs.write"),
    scope(
        Verb::Post,
        "/api/bridge/deadletters/:id/redrive",
        "runs.write",
    ),
    scope(Verb::Delete, "/api/bridge/deadletters/:id", "runs.write"),
    // connectors.use
    scope(
        Verb::Post,
        "/api/bridge/connectors/:id/request",
        "connectors.use",
    ),
    scope(
        Verb::Post,
        "/api/services/actions/:definition_id/:action_id/invoke",
        "connectors.use",
    ),
    // ai.read
    scope(Verb::Get, "/api/ai/sources", "ai.read"),
    scope(Verb::Get, "/api/ai/providers", "ai.read"),
    scope(Verb::Get, "/api/ai/v1/models", "ai.read"),
    scope(Verb::Get, "/api/ai/providers/:id/v1/models", "ai.read"),
    scope(Verb::Get, "/api/ai/codex/status", "ai.read"),
    scope(Verb::Get, "/api/ai/engine/services", "ai.read"),
    // ai.use
    scope(Verb::Post, "/api/ai/v1/chat/completions", "ai.use"),
    scope(
        Verb::Post,
        "/api/ai/providers/:id/v1/chat/completions",
        "ai.use",
    ),
    scope(Verb::Any, "/api/ai/engine/gateway/*path", "ai.use"),
    // ai.admin
    scope(Verb::Post, "/api/ai/providers", "ai.admin"),
    scope(Verb::Delete, "/api/ai/providers/:id", "ai.admin"),
    scope(Verb::Post, "/api/ai/providers/:id/key", "ai.admin"),
    scope(Verb::Post, "/api/ai/providers/:id/test", "ai.admin"),
    scope(Verb::Post, "/api/ai/codex/login", "ai.admin"),
    scope(Verb::Delete, "/api/ai/codex/login", "ai.admin"),
    scope(Verb::Post, "/api/ai/codex/logout", "ai.admin"),
    // speech.use
    scope(Verb::Post, "/api/voice/transcribe", "speech.use"),
    // calls.read
    scope(Verb::Get, "/api/voice/events", "calls.read"),
    scope(Verb::Get, "/api/voice/calls", "calls.read"),
    scope(Verb::Get, "/api/voice/voices", "calls.read"),
    scope(Verb::Get, "/api/voice/settings", "calls.read"),
    scope_rsv(Verb::Get, "/api/messages", "calls.read"),
    scope_rsv(Verb::Get, "/api/ring/settings", "calls.read"),
    // calls.write
    scope(Verb::Post, "/api/voice/calls/:id/say", "calls.write"),
    scope(Verb::Post, "/api/voice/calls/:id/tool", "calls.write"),
    scope(Verb::Post, "/api/voice/calls/:id/finish", "calls.write"),
    scope(Verb::Post, "/api/voice/calls/:id/hush", "calls.write"),
    scope(Verb::Put, "/api/voice/callers", "calls.write"),
    scope(Verb::Put, "/api/voice/settings", "calls.write"),
    scope(Verb::Post, "/api/voice/voices", "calls.write"),
    scope(Verb::Put, "/api/voice/voices/chosen", "calls.write"),
    scope(Verb::Delete, "/api/voice/voices/:name", "calls.write"),
    scope(Verb::Post, "/api/voice/voices/:name/try", "calls.write"),
    scope_rsv(Verb::Patch, "/api/messages/:id", "calls.write"),
    scope_rsv(Verb::Put, "/api/ring/settings", "calls.write"),
    // calendar.read
    scope(Verb::Get, "/api/calendar", "calendar.read"),
    scope(Verb::Get, "/api/calendar/free", "calendar.read"),
    scope(Verb::Get, "/api/calendar/sync", "calendar.read"),
    scope(Verb::Post, "/api/calendar/lookup", "calendar.read"),
    // calendar.write
    scope(Verb::Put, "/api/calendar/settings", "calendar.write"),
    scope(Verb::Post, "/api/calendar/appointments", "calendar.write"),
    scope(
        Verb::Patch,
        "/api/calendar/appointments/:id",
        "calendar.write",
    ),
    scope(
        Verb::Delete,
        "/api/calendar/appointments/:id",
        "calendar.write",
    ),
    scope(Verb::Post, "/api/calendar/sync", "calendar.write"),
    // contacts.read
    scope(Verb::Get, "/api/contacts", "contacts.read"),
    scope(Verb::Get, "/api/contacts/export.csv", "contacts.read"),
    scope(Verb::Get, "/api/contacts/:number", "contacts.read"),
    // contacts.write
    scope(Verb::Put, "/api/contacts/:number", "contacts.write"),
    scope(Verb::Delete, "/api/contacts/:number", "contacts.write"),
    scope(Verb::Post, "/api/contacts/:number/facts", "contacts.write"),
    scope(
        Verb::Delete,
        "/api/contacts/:number/facts/:index",
        "contacts.write",
    ),
    scope(Verb::Post, "/api/contacts/import", "contacts.write"),
    // agent.read
    scope(Verb::Get, "/api/agent/preferences", "agent.read"),
    scope_new(Verb::Get, "/api/agent/status", "agent.read"),
    // agent.tasks
    scope(Verb::Post, "/api/agent/tasks", "agent.tasks"),
    scope(Verb::Get, "/api/agent/tasks/:id", "agent.tasks"),
    // agent.serve
    scope(Verb::Get, "/api/agent/events", "agent.serve"),
    scope(Verb::Post, "/api/agent/tasks/:id/reply", "agent.serve"),
    scope(Verb::Post, "/api/bridge/leases/:name", "agent.serve"),
    scope_rsv(Verb::Get, "/api/remote-agent/events", "agent.serve"),
    scope_rsv(
        Verb::Post,
        "/api/remote-agent/turns/:id/events",
        "agent.serve",
    ),
    scope(Verb::Post, "/api/update/agent-flushed", "agent.serve"),
    // agent.settings
    scope(Verb::Put, "/api/agent/preferences", "agent.settings"),
    scope_new(Verb::Post, "/api/agent/intents", "agent.settings"),
    // setup.read
    scope(Verb::Get, "/api/setup", "setup.read"),
    scope(Verb::Get, "/api/setup/catalog", "setup.read"),
    scope(Verb::Get, "/api/setup/plugins/:id", "setup.read"),
    // setup.write
    scope(Verb::Put, "/api/setup", "setup.write"),
    scope(
        Verb::Post,
        "/api/setup/plugins/:id/steps/:step",
        "setup.write",
    ),
    scope(Verb::Post, "/api/setup/plugins/:id/finish", "setup.write"),
    scope(
        Verb::Post,
        "/api/setup/plugins/:id/check/:step",
        "setup.write",
    ),
    // control.read
    scope(Verb::Post, "/api/mcp", "control.read"),
    scope(Verb::Get, "/api/control/settings", "control.read"),
    scope(Verb::Get, "/api/control/log", "control.read"),
    // control.admin
    scope(Verb::Put, "/api/control/settings", "control.admin"),
    scope_new(Verb::Post, "/api/control/approvals/resume", "control.admin"),
    // link.read
    scope(Verb::Get, "/api/link", "link.read"),
    // link.manage
    scope(Verb::Delete, "/api/link", "link.manage"),
    scope(Verb::Post, "/api/link/start", "link.manage"),
    scope(Verb::Post, "/api/link/cancel", "link.manage"),
    // auth.read
    scope(Verb::Get, "/api/bridge/pairing", "auth.read"),
    scope(Verb::Get, "/api/bridge/pairings", "auth.read"),
    scope_new(Verb::Get, "/api/auth/credentials", "auth.read"),
    scope_new(Verb::Get, "/api/auth/sessions", "auth.read"),
    scope_new(Verb::Get, "/api/auth/audit", "auth.read"),
    scope_new(Verb::Get, "/api/auth/invites", "auth.read"),
    // auth.revoke
    scope(Verb::Post, "/api/bridge/pairing/:id/deny", "auth.revoke"),
    scope_new(Verb::Delete, "/api/bridge/pairing", "auth.revoke"),
    scope(Verb::Delete, "/api/bridge/pairings/:id", "auth.revoke"),
    scope_new(Verb::Delete, "/api/auth/credentials/:id", "auth.revoke"),
    scope_new(Verb::Delete, "/api/auth/sessions/:id", "auth.revoke"),
    scope_new(
        Verb::Post,
        "/api/auth/sessions/revoke-others",
        "auth.revoke",
    ),
    scope_new(Verb::Delete, "/api/auth/invites/:id", "auth.revoke"),
    // auth.manage
    scope(Verb::Post, "/api/bridge/pairing/:id/approve", "auth.manage"),
    scope_new(Verb::Post, "/api/auth/credentials", "auth.manage"),
    scope_new(Verb::Post, "/api/auth/invites", "auth.manage"),
    // companion.read
    scope(Verb::Get, "/api/companion/relay", "companion.read"),
    scope(
        Verb::Get,
        "/api/companion/:plugin/pairing",
        "companion.read",
    ),
    scope_rsv(Verb::Get, "/api/devices", "companion.read"),
    // companion.manage
    scope(Verb::Post, "/api/companion/relay", "companion.manage"),
    scope(Verb::Delete, "/api/companion/relay", "companion.manage"),
    scope(
        Verb::Post,
        "/api/companion/:plugin/pairing/offers",
        "companion.manage",
    ),
    scope(
        Verb::Post,
        "/api/companion/:plugin/pairing/responses",
        "companion.manage",
    ),
    scope(
        Verb::Post,
        "/api/companion/:plugin/pairing/approvals/:id/approve",
        "companion.manage",
    ),
    scope(
        Verb::Post,
        "/api/companion/:plugin/pairing/approvals/:id/deny",
        "companion.manage",
    ),
    scope(
        Verb::Delete,
        "/api/companion/:plugin/mobiles/:thumbprint",
        "companion.manage",
    ),
    scope(
        Verb::Post,
        "/api/companion/:plugin/identity/rotate",
        "companion.manage",
    ),
    scope_rsv(Verb::Patch, "/api/devices/:id", "companion.manage"),
    scope_rsv(Verb::Delete, "/api/devices/:id", "companion.manage"),
    // secrets.write
    scope_new(Verb::Put, "/api/secrets/hf-token", "secrets.write"),
    // system.update
    scope_rsv(Verb::Post, "/api/update/apply", "system.update"),
    // system.restart
    scope_new(Verb::Post, "/api/system/restart", "system.restart"),
    // ui.events
    scope_new(Verb::Get, "/api/ui/events", "ui.events"),
    // relay.read
    scope_rsv(Verb::Get, "/api/relay/status", "relay.read"),
    // vault.read
    scope_rsv(Verb::Get, "/api/local-protection", "vault.read"),
    scope_rsv(Verb::Get, "/api/vault/status", "vault.read"),
    scope_rsv(Verb::Get, "/api/archive/status", "vault.read"),
    scope_rsv(Verb::Get, "/api/backup", "vault.read"),
    scope_rsv(Verb::Get, "/api/backup/catalog", "vault.read"),
    scope_rsv(Verb::Get, "/api/backup/jobs/:id", "vault.read"),
    // vault.control
    scope_rsv(Verb::Put, "/api/backup/config", "vault.control"),
    scope_rsv(Verb::Post, "/api/backup/run", "vault.control"),
    scope_rsv(Verb::Post, "/api/archive/mode", "vault.control"),
    scope_rsv(Verb::Post, "/api/archive/pause", "vault.control"),
    scope_rsv(Verb::Post, "/api/archive/resume", "vault.control"),
    scope_rsv(Verb::Post, "/api/archive/redrive", "vault.control"),
    scope_rsv(
        Verb::Post,
        "/api/local-protection/webview-progress",
        "vault.control",
    ),
    // vault.admin
    scope_rsv_server(Verb::Put, "/api/vault", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/unlock", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/lock", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/disconnect", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/change-passphrase", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/recovery", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/recovery/retire-kit", "vault.admin"),
    scope_rsv_server(Verb::Get, "/api/vault/devices", "vault.admin"),
    scope_rsv_server(Verb::Get, "/api/vault/devices/:id", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/devices", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/devices/:id/revoke", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/vault/devices/:id/seen", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/archive/connect", "vault.admin"),
    scope_rsv_server(Verb::Get, "/api/archive/records", "vault.admin"),
    scope_rsv_server(Verb::Get, "/api/archive/records/:recordId", "vault.admin"),
    scope_rsv_server(Verb::Get, "/api/archive/dead-letters", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/archive/audit", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/archive/export", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/backup/restore", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/backup/rollback", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/backup/verify", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/backup/identity", "vault.admin"),
    scope_rsv_server(Verb::Delete, "/api/backup/catalog/:id", "vault.admin"),
    scope_rsv_server(Verb::Get, "/api/backup/webview-import", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/backup/webview/:jobId/part", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/local-protection/enable", "vault.admin"),
    scope_rsv_server(Verb::Post, "/api/local-protection/recover", "vault.admin"),
    scope_rsv_server(
        Verb::Post,
        "/api/local-protection/wipe-free-space",
        "vault.admin",
    ),
    // vault.kt
    scope_rsv(Verb::Post, "/api/vault/kt", "vault.kt"),
    scope_rsv(Verb::Get, "/api/vault/kt/:sid", "vault.kt"),
    scope_rsv(Verb::Get, "/api/vault/kt/:sid/msg", "vault.kt"),
    scope_rsv(Verb::Post, "/api/vault/kt/:sid/msg", "vault.kt"),
    scope_rsv(Verb::Post, "/api/vault/kt/:sid/approve", "vault.kt"),
    scope_rsv(Verb::Post, "/api/vault/kt/:sid/cancel", "vault.kt"),
];

type Index = HashMap<(Verb, &'static str), &'static Route>;

fn index() -> &'static Index {
    static INDEX: OnceLock<Index> = OnceLock::new();
    INDEX.get_or_init(|| ROUTES.iter().map(|r| ((r.method, r.pattern), r)).collect())
}

/// The row for `method` on the route `pattern` (as axum's `MatchedPath` spells it): an exact
/// `(method, pattern)` row, else the pattern's `Any` row. `None` when there is none.
pub fn lookup(method: Verb, pattern: &str) -> Option<&'static Route> {
    let idx = index();
    idx.get(&(method, pattern))
        .or_else(|| idx.get(&(Verb::Any, pattern)))
        .copied()
}

/// What a request for `method` on the matched route `matched_path` takes. `OPTIONS` is public for
/// every path (the CORS layer answers it); a method the API does not serve, and a route with no
/// row, are [`Class::Unclassified`], which every caller must refuse.
pub fn route_class(method: &axum::http::Method, matched_path: &str) -> Class {
    if method == axum::http::Method::OPTIONS {
        return Class::Public;
    }
    Verb::of_method(method)
        .and_then(|verb| lookup(verb, matched_path))
        .map_or(Class::Unclassified, |r| r.class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Method;

    #[test]
    fn a_row_is_found_by_its_method_and_matched_path() {
        assert_eq!(
            route_class(&Method::GET, "/api/config"),
            Class::Scope("system.read")
        );
        assert_eq!(
            route_class(&Method::POST, "/api/services/:id/start"),
            Class::Scope("services.control")
        );
        assert_eq!(
            route_class(&Method::DELETE, "/api/services/:id"),
            Class::Scope("services.define")
        );
        assert_eq!(route_class(&Method::GET, "/api/health"), Class::Public);
        assert_eq!(
            route_class(&Method::GET, "/api/auth/whoami"),
            Class::AnyCredential
        );
        assert_eq!(
            route_class(&Method::POST, "/api/auth/console/setup-code"),
            Class::Console
        );
        assert_eq!(
            route_class(&Method::POST, "/api/auth/logout"),
            Class::Session { elevate: false }
        );
    }

    #[test]
    fn the_method_is_part_of_the_key() {
        // The same path answers different things by method.
        assert_eq!(
            route_class(&Method::GET, "/api/services"),
            Class::Scope("services.read")
        );
        assert_eq!(
            route_class(&Method::POST, "/api/services"),
            Class::Scope("services.define")
        );
        assert_eq!(
            route_class(&Method::PUT, "/api/services"),
            Class::Unclassified
        );
        assert_eq!(
            route_class(&Method::GET, "/api/bridge/pairing"),
            Class::Scope("auth.read")
        );
        assert_eq!(
            route_class(&Method::POST, "/api/bridge/pairing"),
            Class::Public
        );
        assert_eq!(
            route_class(&Method::DELETE, "/api/bridge/pairing"),
            Class::Scope("auth.revoke")
        );
    }

    #[test]
    fn head_is_answered_from_the_get_row() {
        assert_eq!(
            route_class(&Method::HEAD, "/api/config"),
            route_class(&Method::GET, "/api/config")
        );
        assert_eq!(route_class(&Method::HEAD, "/api/health"), Class::Public);
        assert_eq!(
            route_class(&Method::HEAD, "/api/bridge/pairing/:id"),
            Class::Public
        );
        // No GET row, no HEAD row: a POST-only route is not reachable as HEAD.
        assert_eq!(
            route_class(&Method::POST, "/api/mcp"),
            Class::Scope("control.read")
        );
        assert_eq!(route_class(&Method::HEAD, "/api/mcp"), Class::Unclassified);
        assert_eq!(
            route_class(&Method::HEAD, "/api/bridge/runs/:id/claim"),
            Class::Unclassified
        );
    }

    #[test]
    fn options_is_public_for_every_path_even_an_unknown_one() {
        assert_eq!(route_class(&Method::OPTIONS, "/api/config"), Class::Public);
        assert_eq!(
            route_class(&Method::OPTIONS, "/api/no/such/route"),
            Class::Public
        );
    }

    #[test]
    fn a_wildcard_route_has_one_row_for_every_method() {
        let pattern = "/api/ai/engine/gateway/*path";
        for method in [
            Method::GET,
            Method::HEAD,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
        ] {
            assert_eq!(
                route_class(&method, pattern),
                Class::Scope("ai.use"),
                "{method}"
            );
        }
        // The pattern is the key: a concrete path is not a pattern.
        assert_eq!(
            route_class(&Method::GET, "/api/ai/engine/gateway/v1/images"),
            Class::Unclassified
        );
    }

    #[test]
    fn an_unknown_path_or_method_is_unclassified_which_callers_must_refuse() {
        assert_eq!(
            route_class(&Method::GET, "/api/no/such/route"),
            Class::Unclassified
        );
        assert_eq!(route_class(&Method::GET, ""), Class::Unclassified);
        // A concrete path is not a route pattern: only the pattern axum matched is a key.
        assert_eq!(
            route_class(&Method::GET, "/api/services/abc/logs"),
            Class::Unclassified
        );
        assert_eq!(
            route_class(&Method::TRACE, "/api/config"),
            Class::Unclassified
        );
        assert_eq!(
            route_class(&Method::CONNECT, "/api/config"),
            Class::Unclassified
        );
        assert_eq!(
            route_class(&Method::from_bytes(b"BREW").unwrap(), "/api/config"),
            Class::Unclassified
        );
    }

    #[test]
    fn the_static_routes_beat_their_parameter_siblings_because_they_are_different_patterns() {
        // `/api/plugins/install` and `/api/plugins/:id` are different `MatchedPath` strings.
        assert_eq!(
            route_class(&Method::POST, "/api/plugins/install"),
            Class::Scope("plugins.install")
        );
        assert_eq!(
            route_class(&Method::DELETE, "/api/plugins/:id"),
            Class::Scope("plugins.install")
        );
        assert_eq!(
            route_class(&Method::POST, "/api/plugins/:id/start"),
            Class::Scope("plugins.control")
        );
        assert_eq!(
            route_class(&Method::POST, "/api/services/ensure-by-port"),
            Class::Scope("services.control")
        );
        assert_eq!(
            route_class(&Method::POST, "/api/services/:id/start"),
            Class::Scope("services.control")
        );
    }

    #[test]
    fn no_row_is_a_duplicate_and_none_carries_the_unclassified_class() {
        let mut seen = std::collections::HashSet::new();
        for r in ROUTES {
            assert!(
                seen.insert((r.method, r.pattern)),
                "duplicate row {}",
                r.key()
            );
            assert_ne!(r.class, Class::Unclassified, "{} is unclassified", r.key());
            assert!(
                matches!(r.since, 1 | 2),
                "{} has since {}",
                r.key(),
                r.since
            );
            assert!(r.pattern.starts_with('/'), "{}", r.key());
        }
    }

    #[test]
    fn the_rows_that_hold_no_scope_are_the_ones_the_design_lists() {
        let mut classes: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
        for r in ROUTES {
            let name = match r.class {
                Class::Public => "public",
                Class::AnyCredential => "any-credential",
                Class::Console => "console",
                Class::Session { .. } => "session",
                Class::Desk { .. } => "desk",
                Class::Scope(_) => continue,
                Class::Unclassified => "unclassified",
            };
            classes.entry(name).or_default().push(r.key());
        }
        assert_eq!(
            classes["public"],
            [
                "GET /api/health",
                "GET /api/bridge/capabilities",
                "POST /api/bridge/pairing",
                "GET /api/bridge/pairing/:id",
                "GET /api/auth/info",
                "GET /api/auth/session",
                "POST /api/auth/login",
                "POST /api/auth/setup",
                "POST /api/auth/link",
                "POST /api/auth/callback",
            ]
        );
        assert_eq!(
            classes["any-credential"],
            ["GET /api/auth/whoami", "POST /api/auth/derive"]
        );
        assert_eq!(classes["console"].len(), 5);
        assert_eq!(classes["session"].len(), 5);
        assert_eq!(classes["desk"], ["GET /api/local-protection/webview-key"]);
        assert!(!classes.contains_key("unclassified"));
    }
}
