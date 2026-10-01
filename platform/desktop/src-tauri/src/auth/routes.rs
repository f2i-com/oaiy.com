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
    // The updater merged after the design counted its routes, so three of its rows, and the plugin trust
    // route, are not in Appendix B (`routes.golden.txt` lists them as its four differences). Each is
    // classified here on purpose, against what the guard that was always in front of it did:
    //
    // - `GET /api/update/status` is `system.read`: the design lists the update status there. The desktop's old
    //   guard kept it to OAIY's own pages and the token, because it says in words whether a call is live (the
    //   blockers of "Restart to update"); a `system.read` holder, a `readonly` monitor too, sees that now.
    // - `POST /api/update/check` (below, under `services.control`) makes OAIY ask GitHub for a newer release.
    // - `POST /api/update/agent-flushed` (under `agent.serve`) is the Agent page's answer to "save your work
    //   before an update".
    // - `POST /api/plugins/:id/trust` (under `plugins.install`) trusts native code.
    scope(Verb::Get, "/api/update/status", "system.read"),
    // The encrypted backup (`backup/routes.rs`) was built before the access model merged, and is not in Appendix B (the vault design
    // reserves other `/api/backup` routes, which stay as they are below). Eight rows, each classified on purpose, `since: 1`: the routes
    // exist, and `legacy` mode keeps the guard they were built with (the lines of `is_backup_path` in `http.rs`, which the frozen guard
    // copies, marked). No scope is dangerous, and none of the eight restores or overwrites a setting: making and restoring a backup are
    // commands of the dashboard's own window (`backup_create`, `backup_restore_stage`, ...), not routes, and the desktop applies a restore
    // itself, at the next start, from files the owner looked at and ticked.
    //
    // - `GET /api/backup/status` is `system.read`, beside `GET /api/update/status`: when the last backup was made, whether a restore waits
    //   for the next start and how the last one went. The dashboard's window reads it; a monitor may.
    // - The Agent page's hand-over of its own storage (seven routes, below under `agent.serve`) is the same kind of route as the page's
    //   answer to "save your work before an update": the desktop asks the page for its conversations and projects when the owner makes a
    //   backup, and hands the page what a restore left for it, and the page answers. Each takes a session secret that only the desktop and
    //   the page hold, so a credential with the scope alone can do nothing with them, and `agent.serve` is held by the `agent` preset and
    //   `owner` alone.
    scope(Verb::Get, "/api/backup/status", "system.read"),
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
    // Not in Appendix B. It makes OAIY send a request to GitHub (at most every 30 seconds), which is an action
    // and not a read: the old guard held it as a privileged route (OAIY's own window or the token), so a
    // `readonly` monitor's token, which the old model had no such thing as, must not be able to trigger it.
    // No scope names the updater's check (`system.update` is the apply, and dangerous), and a new scope is a
    // change to the design; `services.control` is the scope of the `cli` preset that keeps things installed
    // and that a monitor lacks, so `curl -X POST -H "Authorization: Bearer $OAIY_SERVER_TOKEN"
    // .../api/update/check` (docs/UPDATES.md) works as it did.
    scope(Verb::Post, "/api/update/check", "services.control"),
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
    // The receptionist's transfers and messages (`ring/routes.rs`, `messages/routes.rs`) were built after the
    // design counted its routes, before the access model merged: four of these rows are the ones the design
    // reserved (`since: 2`, no route yet) and are now `since: 1`, since the routes exist and the guard that has
    // always been in front of them (the `is_personal_path` lines in `http.rs`) is what `legacy` mode keeps; seven
    // are not in the design at all (`routes.golden.txt` marks them "added since the appendix"). What each is:
    //
    // - Reading the messages (callers' words and numbers), the rings going now (who is asking for the owner and
    //   what they said) and what a caller would get now is `calls.read`, the scope of the call events and the
    //   calls list: the same words and the same numbers, from the same callers.
    // - The receptionist's own `take_message` on its call is `calls.write`, beside `say`, `tool`, `finish` and
    //   `hush`: the Agent page answers calls, and this is how what a caller leaves is kept.
    // - Acting on what callers left and on the rings that are going (a message marked seen or handled, or deleted, a
    //   ring declined or a message offered in its place, a notice put away) is `calls.manage`, a scope of its own that
    //   the `owner` preset and the relay's call-control tier (a phone that may control a call, which marks messages and
    //   declines rings: mobile.md) hold, and the Agent page's `agent` preset does not: what a caller left is the
    //   owner's record, and a page that may be untrusted must not be able to clear it, or to decline the owner's
    //   rings. Deleting a message is no more dangerous than deleting a contact or an appointment (`contacts.write`,
    //   `calendar.write`), and none of these takes the call to the owner: only the Companion does that, and no route
    //   here accepts.
    // - The owner's transfer settings (`GET` and `PUT /api/ring/settings`: whether transfers are on at all, whom
    //   they may be put through to, the VIP numbers that are exempt from the limits and the quiet hours, which
    //   Companions ring) are `calls.settings`, a scope of their own that only the `owner` preset (and the relay's
    //   admin tier) holds. The design has them under `calls.read` and `calls.write`, which the `agent` preset holds
    //   (it is the Agent page, and a page that may be untrusted must not be able to turn transfers on, which are
    //   off until the owner does, or name a VIP number, or silence the owner's phones): the host's policy has to hold
    //   whatever the Agent page does, as `agent.settings` is off that preset for the same reason.
    scope(Verb::Get, "/api/messages", "calls.read"),
    scope(Verb::Get, "/api/messages/:id", "calls.read"),
    scope(Verb::Get, "/api/ring/preview", "calls.read"),
    scope(Verb::Get, "/api/ring/active", "calls.read"),
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
    scope(Verb::Post, "/api/voice/calls/:id/message", "calls.write"),
    // calls.manage
    scope(Verb::Patch, "/api/messages/:id", "calls.manage"),
    scope(Verb::Delete, "/api/messages/:id", "calls.manage"),
    scope(Verb::Post, "/api/ring/active/:id/respond", "calls.manage"),
    scope(Verb::Post, "/api/ring/notices/:id/dismiss", "calls.manage"),
    // calls.settings
    scope(Verb::Get, "/api/ring/settings", "calls.settings"),
    scope(Verb::Put, "/api/ring/settings", "calls.settings"),
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
    scope(Verb::Post, "/api/backup/agent/:id/part", "agent.serve"),
    scope(Verb::Post, "/api/backup/agent/:id/done", "agent.serve"),
    scope(Verb::Get, "/api/backup/agent-import", "agent.serve"),
    scope(
        Verb::Get,
        "/api/backup/agent-import/:id/part/:index",
        "agent.serve",
    ),
    scope(
        Verb::Post,
        "/api/backup/agent-import/:id/undo-part",
        "agent.serve",
    ),
    scope(
        Verb::Post,
        "/api/backup/agent-import/:id/undo-done",
        "agent.serve",
    ),
    scope(
        Verb::Post,
        "/api/backup/agent-import/:id/done",
        "agent.serve",
    ),
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

/// Whether any method of the route `pattern` has a row with `since: 1`: the route existed before the access
/// model, so the guard that was always in front of it stays in front of every method of it, including a
/// method the table adds later (`DELETE /api/bridge/pairing` next to the old `GET` and `POST`).
pub fn pattern_existed_before(pattern: &str) -> bool {
    static OLD: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    OLD.get_or_init(|| {
        ROUTES
            .iter()
            .filter(|r| r.since == 1)
            .map(|r| r.pattern)
            .collect()
    })
    .contains(pattern)
}

/// Whether every row of the route `pattern` is [`Class::Public`] (health, capability discovery, the login
/// routes): the one kind of route a bare `OPTIONS` (one that is not a CORS preflight) is passed on to, since
/// what it would learn there is public already. A pattern with any other row, and a pattern with no row, is
/// not.
pub fn pattern_is_public(pattern: &str) -> bool {
    static PUBLIC: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    PUBLIC
        .get_or_init(|| {
            ROUTES
                .iter()
                .filter(|r| r.class == Class::Public)
                .map(|r| r.pattern)
                .filter(|p| {
                    ROUTES
                        .iter()
                        .filter(|r| r.pattern == *p)
                        .all(|r| r.class == Class::Public)
                })
                .collect()
        })
        .contains(pattern)
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

    /// The design's table (Appendix B), one row per line, as `routes.golden.txt` holds it.
    const GOLDEN: &str = include_str!("routes.golden.txt");

    /// A row of the table in the golden file's words.
    fn golden_line(r: &Route) -> String {
        let class = match r.class {
            Class::Public => "public".to_string(),
            Class::AnyCredential => "any".to_string(),
            Class::Console => "console".to_string(),
            Class::Session { .. } => "session".to_string(),
            Class::Desk { .. } => "desk".to_string(),
            Class::Scope(s) => format!("scope:{s}"),
            Class::Unclassified => "unclassified".to_string(),
        };
        let only = match r.only {
            Only::Everywhere => "all",
            Only::Server => "server",
        };
        format!(
            "{} {} {class} {} {only}",
            r.method.as_str(),
            r.pattern,
            r.since
        )
    }

    /// The rows of the golden file, how many of them are marked as added since the appendix, how many as
    /// reserved by it and built since (`since` 1 here, 2 in the appendix), and which as having a scope that is
    /// not the appendix's.
    fn golden_rows() -> (Vec<String>, usize, usize, Vec<String>) {
        let mut rows = Vec::new();
        let (mut added, mut built) = (0, 0);
        let mut rescoped = Vec::new();
        for line in GOLDEN.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (row, note) = line.split_once('#').unwrap_or((line, ""));
            if note.contains("added since the appendix") {
                added += 1;
            }
            if note.contains("built since the appendix") {
                built += 1;
            }
            if note.contains("scope changed since the appendix") {
                rescoped.push(row.split_whitespace().take(2).collect::<Vec<_>>().join(" "));
            }
            rows.push(row.trim().to_string());
        }
        (rows, added, built, rescoped)
    }

    #[test]
    fn every_row_of_the_table_is_the_row_the_design_gives_and_nothing_else_is_there() {
        // The golden file is written from Appendix B of the design, not from `ROUTES`; the two are compared
        // exactly. A scope changed on a route (`POST /api/link/start` from `link.manage` to `link.read`),
        // a method, a `since`, a route added or one dropped fails here until the file says the same.
        let (mut want, added, built, rescoped) = golden_rows();
        let mut have: Vec<String> = ROUTES.iter().map(golden_line).collect();
        want.sort();
        have.sort();
        let missing: Vec<&String> = want.iter().filter(|w| !have.contains(w)).collect();
        let extra: Vec<&String> = have.iter().filter(|h| !want.contains(h)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "rows the design has and the table does not:\n{}\nrows the table has and the design does not:\n{}",
            missing
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            extra
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert_eq!(want, have);
        // 161 existing pairs, 41 new and 60 reserved (Appendix B), less the reserved `GET /api/update` that
        // was built under another name, plus the eleven routes the code gained since (four of the updater, and
        // seven of the receptionist's transfers and messages).
        assert_eq!(
            want.len(),
            161 + 41 + 60 - 1 + 4 + 7 + 8,
            "the rows of the design and its documented differences"
        );
        assert_eq!(
            added, 19,
            "the routes added since the appendix are the nineteen the file lists"
        );
        // Four rows the appendix reserves were built (the messages and the ring's settings): the scope is the
        // design's and the `since` is 1, since the routes exist.
        assert_eq!(built, 4, "the reserved rows that were built are the four the file lists");
        for key in [
            "GET /api/messages",
            "PATCH /api/messages/:id",
            "GET /api/ring/settings",
            "PUT /api/ring/settings",
        ] {
            let row = ROUTES.iter().find(|r| r.key() == key).unwrap_or_else(|| panic!("{key}"));
            assert_eq!(row.since, 1, "{key}: built, so no longer reserved");
        }
        // ...and the scope of three of them is not the design's: the owner's transfer settings are `calls.settings`, and marking a message is
        // `calls.manage`, which the `agent` preset holds neither of (see the comment on the rows in `routes.rs`).
        let mut rescoped = rescoped;
        rescoped.sort();
        assert_eq!(
            rescoped,
            ["GET /api/ring/settings", "PATCH /api/messages/:id", "PUT /api/ring/settings"],
            "the rows whose scope is not the appendix's are the ones the file lists"
        );
        let mut unique = want.clone();
        unique.dedup();
        assert_eq!(unique.len(), want.len(), "no row twice");
    }

    #[test]
    fn what_the_golden_file_does_not_say_of_a_row_is_pinned_here() {
        // The appendix says "own rule, 4.7" for a cookie session and "role read from the credential" for a
        // desk: the table holds no elevation on the five session routes and all three roles on the desk one.
        let sessions: Vec<&Route> = ROUTES
            .iter()
            .filter(|r| matches!(r.class, Class::Session { .. }))
            .collect();
        assert_eq!(sessions.len(), 5);
        assert!(sessions
            .iter()
            .all(|r| r.class == Class::Session { elevate: false }));
        let desk: Vec<&Route> = ROUTES
            .iter()
            .filter(|r| matches!(r.class, Class::Desk { .. }))
            .collect();
        assert_eq!(desk.len(), 1);
        assert_eq!(
            desk[0].class,
            Class::Desk {
                roles: &[DeskRole::Dashboard, DeskRole::Agent, DeskRole::Flows]
            }
        );
    }

    #[test]
    fn a_pattern_is_old_when_any_method_of_it_has_a_row_from_before_the_model() {
        // `/api/bridge/pairing` has the old `GET` and `POST` and the `DELETE` the table adds: the pattern is old.
        for old in [
            "/api/bridge/pairing",
            "/api/config",
            "/api/health",
            "/api/ai/engine/gateway/*path",
        ] {
            assert!(pattern_existed_before(old), "{old}");
        }
        // Patterns only the model adds (or another design announces), and paths that are not patterns.
        for new in [
            "/api/auth/info",
            "/api/auth/derive",
            "/api/system/gpus",
            "/api/services/:id/gpu",
            "/api/config/x",
            "",
        ] {
            assert!(!pattern_existed_before(new), "{new}");
        }
    }

    #[test]
    fn a_pattern_is_public_only_when_every_row_of_it_is() {
        for public in [
            "/api/health",
            "/api/bridge/capabilities",
            "/api/bridge/pairing/:id",
            "/api/auth/info",
            "/api/auth/login",
            "/api/auth/callback",
        ] {
            assert!(pattern_is_public(public), "{public}");
        }
        // Public `POST` next to a scoped `GET` and `DELETE`: not public altogether.
        for not_public in [
            "/api/bridge/pairing",
            "/api/services",
            "/api/auth/whoami",
            "/api/no/such/route",
            "",
        ] {
            assert!(!pattern_is_public(not_public), "{not_public}");
        }
    }
}
