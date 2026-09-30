//! The guard as it was before the access model: a frozen copy, for the differential test.
//!
//! Copied from `platform/desktop/src-tauri/src/http.rs` at commit 2ea1ee8 (`origin_guard` and everything it
//! decides with: the path predicates, the origin lists, the bearer comparison, `AuthConfig`), byte for byte
//! except that `AuthConfig`, its fields and `origin_guard` are `pub(super)` so the test can reach them.
//! Nothing else is changed. Do not edit it: the differential test in `legacy_neutrality.rs` exists to notice
//! when the live guard stops answering as this one does, and it can only do that against a copy that does not
//! move with it.

#![allow(dead_code)]

use axum::{
    extract::{Request, State},
    http::{
        header::{AUTHORIZATION, ORIGIN},
        Method, StatusCode,
    },
    middleware::Next,
    response::IntoResponse,
    Json,
};
/// Whether a browser `Origin` is allowed to drive state-changing endpoints.
/// The localhost bind keeps non-browser callers out; this stops a *web page*
/// the user happens to have open from issuing drive-by POST/DELETE requests
/// (which would otherwise be possible since CORS is permissive for reads).
/// True only when `origin`'s HOST is exactly a loopback name — NOT a prefix.
/// `origin.starts_with("http://localhost")` would also accept the attacker-owned
/// `http://localhost.evil.com`, so we parse the host and compare it exactly.
/// Port-agnostic; handles bracketed IPv6 (`http://[::1]:port`).
fn is_loopback_origin(origin: &str) -> bool {
    let rest = match origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    {
        Some(r) => r,
        None => return false,
    };
    let host = rest.split('/').next().unwrap_or(rest);
    if let Some(inner) = host.strip_prefix('[') {
        // Bracketed IPv6: take the part before ']'.
        return inner.split(']').next() == Some("::1");
    }
    // host[:port] — strip a trailing :port (none of our loopback names contain ':').
    let host = host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host);
    host == "localhost" || host == "127.0.0.1"
}

/// Whether `origin` is one of the pages OAIY shows in its own window (the
/// agent, `oaiy`; the flow editor, `oaiyflows`), served from its own schemes.
pub fn is_embedded_origin(origin: &str) -> bool {
    ["oaiy", "oaiyflows"].iter().any(|s| {
        origin == format!("{s}://localhost") || origin == format!("http://{s}.localhost") || origin == format!("https://{s}.localhost")
    })
}

/// `GET /api/update/status`: open on the headless server (like health), a restricted read on the desktop.
const UPDATE_STATUS_PATH: &str = "/api/update/status";

fn is_allowed_origin(origin: &str) -> bool {
    // Dev + locally-served oaiy-web (any loopback port).
    if is_loopback_origin(origin) {
        return true;
    }
    // OAIY's own pages in its window: the agent and the flow editor.
    if is_embedded_origin(origin) {
        return true;
    }
    // The provider this desktop is LINKED to. Linking is the user approving
    // that provider, and its web app is where they then expect to see and
    // control this machine. Derived from the link rather than hardcoded, so no
    // address is baked in and unlinking withdraws it.
    if crate::link::linked_origin().is_some_and(|o| o == origin) {
        return true;
    }
    // Tauri webview origins (in case OAIY Desktop's own UI ever calls over HTTP).
    if origin == "tauri://localhost"
        || origin == "http://tauri.localhost"
        || origin == "https://tauri.localhost"
    {
        return true;
    }
    // Production oaiy-web: https://oaiy.com and any subdomain (port-agnostic).
    if let Some(rest) = origin.strip_prefix("https://") {
        let host = rest.split('/').next().unwrap_or(rest);
        let host = host.split(':').next().unwrap_or(host);
        if host == "oaiy.com" || host.ends_with(".oaiy.com") {
            return true;
        }
    }
    false
}

/// Endpoints that DEFINE/INSTALL arbitrary code or DESTROY user data — i.e. the
/// real exec surface. A malicious local web page (any `http://localhost:<port>`)
/// must not be able to reach these: defining a service command and starting it
/// would be remote code execution. They get the stricter origin check below and
/// fail CLOSED on a missing `Origin`. Note: starting/stopping/installing an
/// ALREADY-DEFINED service (and ensure-by-port) stays on the broad allow-list —
/// those only run commands the user already added + reviewed, and the web app
/// relies on them.
fn is_privileged_path(method: &Method, path: &str) -> bool {
    match *method {
        Method::POST => {
            matches!(
                path,
                "/api/services"
                    | "/api/models/download"
                    | "/api/python/venvs"
                    | "/api/python/install"
                    | "/api/node/install"
                    // Starts a download of gigabytes into the engines' folder.
                    | "/api/engines/downloads"
                    // Makes OAIY ask GitHub for a newer release (at most every 30 seconds), and the Agent
                    // page's answer when it is asked to save its work before an update: a page the owner
                    // happens to have open must not do either. (Downloading and installing are commands of
                    // the dashboard's own window, not routes.)
                    | "/api/update/check"
                    | "/api/update/agent-flushed"
            ) || (path.starts_with("/api/services/") && path.ends_with("/uninstall"))
                || is_setup_path(path)
                || is_bridge_exec_path(path)
                || is_ai_exec_path(path)
                || is_personal_path(path)
                || is_control_path(path)
                || is_engine_control_path(path)
        }
        Method::PATCH => is_personal_path(path),
        // PUT is only used by the bridge (flow documents). A flow doc is
        // executable code the worker hands to the CLI, so it is exec surface.
        Method::PUT => {
            is_bridge_exec_path(path) || is_personal_path(path) || is_setup_path(path) || is_control_path(path) || is_engine_control_path(path)
        }
        Method::DELETE => {
            path.starts_with("/api/services/")
                || path.starts_with("/api/models/")
                || path.starts_with("/api/python/venvs/")
                // Uninstalling a plugin removes native code from disk.
                || path.starts_with("/api/plugins/")
                || is_bridge_exec_path(path)
                || is_ai_exec_path(path)
                || is_personal_path(path)
                || is_control_path(path)
        }
        _ => false,
    }
}

/// The control API (`control/`): the MCP server the Agent configures OAIY
/// with, the Agent's switch and the log of what it changed. Its tools reach
/// everything the other privileged routes do, so every change takes the
/// privileged gate and every read is a restricted read, like `/api/setup`.
fn is_control_path(path: &str) -> bool {
    path == "/api/mcp" || path == "/api/control" || path.starts_with("/api/control/")
}

/// Choosing the engines' models and starting or stopping their language
/// model (`control/engines.rs`): the engines' configuration, taken on the
/// privileged gate like their downloads.
fn is_engine_control_path(path: &str) -> bool {
    path == "/api/engines/defaults" || path.starts_with("/api/engines/llm/")
}

/// The setup wizard's record and its checks (`setup.rs`). Changing it is
/// privileged: accepting a plugin's capabilities is a trust act, and a check
/// runs one of the plugin's commands. Reading it is a restricted read.
fn is_setup_path(path: &str) -> bool {
    path == "/api/setup" || path.starts_with("/api/setup/")
}

/// Calls, the calendar, the contacts and flows' tasks for the agent: callers' numbers and words,
/// customers' names and appointments, the person's notes about the people who ring and what the
/// receptionist remembered, what a flow asks. Reading them is a restricted read; changing them
/// (speaking on a live call, booking or deleting an appointment, naming or importing contacts)
/// takes the privileged gate.
fn is_personal_path(path: &str) -> bool {
    path.starts_with("/api/voice/")
        || path == "/api/calendar"
        || path.starts_with("/api/calendar/")
        || path == "/api/contacts"
        || path.starts_with("/api/contacts/")
        || path.starts_with("/api/agent/")
}

/// Bridge + plugin routes that EXECUTE code, cause physical side effects, or
/// persist a foothold across restart. A security review found the entire bridge
/// surface was on the broad (loopback-permissive) allow-list: a local web page
/// could `PUT` a flow document and `POST /api/bridge/runs` to run it — remote
/// code execution of exactly the shape `is_privileged_path` already defends the
/// services routes against — or `POST` a connector command to send an SMS. So
/// these join the strict-origin set (OAIY's own webview / oaiy.com only; loopback
/// only in debug) and fail closed on a missing Origin. `/api/bridge/capabilities`
/// and `/api/health` are deliberately NOT here — they are the open discovery
/// handshake the protocol commits to, and carry no user data.
fn is_bridge_exec_path(path: &str) -> bool {
    path == "/api/bridge/runs"                       // reserve + execute a flow
        || path == "/api/bridge/triggers"            // create a persistent binding
        || path.starts_with("/api/bridge/runs/")     // claim / finish / cancel
        || path.starts_with("/api/bridge/flows/")    // define / delete a flow doc
        || path.starts_with("/api/bridge/triggers/") // delete a binding
        || path.starts_with("/api/bridge/connectors/") // physical side effects
        // Redrive re-dispatches a stored event through the ordinary trigger
        // path, so it RESERVES RUNS the worker then executes — the same power
        // as creating a run, and it was on the broad loopback allow-list.
        // DELETE on the same prefix destroys the only record that an event was
        // lost, which is not something a local page should be able to do either.
        || path.starts_with("/api/bridge/deadletters")
        // Companion device trust: these decide which phones may carry a live
        // call's audio, and rotation invalidates every existing pairing. A
        // local web page must not reach them just by being on loopback.
        || path.starts_with("/api/companion/")
        // Linking opens the user's browser and ends in a stored credential;
        // unlinking throws that credential away. Neither belongs to a local
        // page that happens to be on loopback.
        || path.starts_with("/api/link")
        // Pairing APPROVAL/denial is the user's trust act, and revoke unpairs an
        // app — only OAIY's own webview (or a token holder) may. But raising a
        // request (`POST /api/bridge/pairing`) and polling it
        // (`/api/bridge/pairing/<id>`) are OPEN — their whole job is to let an
        // untrusted consumer bootstrap, and neither grants anything without the
        // approval below. `pairings` (plural) is the granted-token surface.
        || path.ends_with("/approve")
        || path.ends_with("/deny")
        || path.starts_with("/api/bridge/pairings")
        // Installing a plugin installs NATIVE CODE this host will then supervise,
        // and removing one deletes it from disk — strictly more dangerous than
        // starting an already-reviewed plugin. A paired web page must never
        // reach either; only OAIY's own window or a token holder.
        || path == "/api/plugins/install"
        // Invoking a plugin-contributed action IS a connector command with
        // physical side effects (aokie.phone's call.dial places a real call), so
        // it takes the same gate as the raw connector route.
        || path.starts_with("/api/services/actions/")
        || (path.starts_with("/api/plugins/")
            && (path.ends_with("/start")
                || path.ends_with("/stop")
                || path.ends_with("/enabled")
                // Letting a package nobody signed run is the person's decision, and
                // only OAIY's own window (or a token holder) may make it for them.
                || path.ends_with("/trust")))
}

/// The AI gateway surface: provider CRUD + credential admin AND the chat/models
/// proxy that spends the stored key. ALL of it takes the exec-surface gate
/// (trusted origin OR bearer/paired token, fail-closed on a missing Origin) — an
/// anonymous local page must not reconfigure a provider, plant a key, or spend
/// the user's API credits. The path prefix covers `/api/ai/providers*` (config)
/// and `/api/ai/{v1,providers/:id/v1}/*` (the gateway). GET reads are handled by
/// `is_restricted_read_path`, not here.
fn is_ai_exec_path(path: &str) -> bool {
    path.starts_with("/api/ai/")
}

/// GET /api/services/:id/export returns the FULL ServiceTemplate — including `run.env` (which a
/// user-authored service may hold an API key in) and the verbatim install/helper script bodies.
/// It's the read-twin of the privileged `add_service` POST, so it's gated like a privileged read
/// (trusted origin or token) rather than left on the open GET surface.
fn is_export_path(path: &str) -> bool {
    path.starts_with("/api/services/") && path.ends_with("/export")
}

/// GET reads that expose process output / absolute paths (the OS username via the data-dir path)
/// and so must not be readable by an arbitrary cross-origin page: the logs endpoints + the config
/// snapshot. Gated on the broad allow-list (blocks only a remote cross-origin page; loopback dev
/// tools + the native CLI still pass).
fn is_restricted_read_path(path: &str) -> bool {
    path == "/api/config"
        || path == "/api/python/logs"
        || path == "/api/node"
        || path == "/api/node/logs"
        || (path.starts_with("/api/services/") && path.ends_with("/logs"))
        // Bridge/plugin reads carry real data an arbitrary remote page must not
        // scrape cross-origin: events hold plugin-supplied payloads (for Aokie,
        // caller phone numbers and message bodies), runs hold flow inputs and
        // outputs, and the plugins listing exposes each plugin's absolute `dir`
        // — the OS username, the exact leak `/api/config` is already gated for.
        // `capabilities` + `health` stay open (discovery); everything else under
        // these prefixes is gated to a non-remote origin.
        || path == "/api/plugins"
        // Which modules the plugins provide (the phone, the calendar) — same tier as /api/plugins.
        || path == "/api/modules"
        || path == "/api/modules/events"
        // Which plugins are installed and what they can do — same tier as /api/plugins.
        || path == "/api/services/definitions"
        // Readiness names plugins + queue depth: gated like the other bridge reads.
        || path == "/api/bridge/status"
        // A dead letter stores the WHOLE event envelope — the same
        // plugin-supplied payload `/api/bridge/events` is gated for, except
        // durable across restarts rather than a 500-entry ring.
        || path.starts_with("/api/bridge/deadletters")
        // Companion status names every trusted device and the desktop's own
        // thumbprint — the material an attacker would want in order to imitate
        // a pairing screen. Same tier as the other bridge reads.
        || path.starts_with("/api/companion/")
        // The link status names the provider, the account and the granted
        // scopes — an inventory of what this machine can reach.
        || path.starts_with("/api/link")
        || path == "/api/bridge/events"
        || path == "/api/bridge/runs"
        || path == "/api/bridge/flows"
        || path == "/api/bridge/triggers"
        || path.starts_with("/api/bridge/runs/")
        || (path.starts_with("/api/plugins/") && path.ends_with("/logs"))
        // Who is asking to pair, and which apps are paired, are the OAIY UI's to
        // see — not a remote page's. Note the exact match: the POLL route
        // `/api/bridge/pairing/<id>` is deliberately NOT here (a consumer must be
        // able to poll for its own token), only the listing `/api/bridge/pairing`.
        || path == "/api/bridge/pairing"
        || path == "/api/bridge/pairings"
        // The AI gateway's reads: sources union + provider listing (names, base
        // URLs, hasKey/enabled — no secret) and the models proxy. Not for an
        // arbitrary remote page; a paired token or a trusted origin passes.
        || path.starts_with("/api/ai/")
        || is_personal_path(path)
        // The Agent's model (engine or ChatGPT). Already under `/api/agent/`,
        // named so it stays gated if that prefix ever narrows.
        || path == "/api/agent/preferences"
        // Which models are loaded, the GPUs, the engines' address; their
        // catalog, the models chosen and the downloads.
        || path == "/api/engines"
        || path.starts_with("/api/engines/")
        // How far setup got, which plugins were chosen, what was accepted.
        || is_setup_path(path)
        // The MCP server (a GET is 405 behind this), the Agent's switch, and what it changed.
        || is_control_path(path)
}

/// Stricter allow-list for privileged endpoints: OAIY Desktop's OWN webview and
/// oaiy.com only — never an arbitrary localhost page. Loopback origins are
/// allowed in debug builds (the dev UI is served from a localhost port) but NOT
/// in a release build, which is what ships.
fn is_allowed_origin_privileged(origin: &str) -> bool {
    if origin == "tauri://localhost"
        || origin == "http://tauri.localhost"
        || origin == "https://tauri.localhost"
        || is_embedded_origin(origin)
    {
        return true;
    }
    if let Some(rest) = origin.strip_prefix("https://") {
        let host = rest.split('/').next().unwrap_or(rest);
        let host = host.split(':').next().unwrap_or(host);
        if host == "oaiy.com" || host.ends_with(".oaiy.com") {
            return true;
        }
    }
    #[cfg(debug_assertions)]
    if is_loopback_origin(origin) {
        return true;
    }
    false
}

/// Extract a `Bearer <token>` from the Authorization header, if present.
fn bearer_token(req: &Request) -> Option<String> {
    req.headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim().to_owned())
}

/// Compare the configured token to the supplied one without short-circuiting on
/// the first differing byte (so it can't be recovered prefix-by-prefix via
/// timing). Length still differs early — acceptable for a loopback secret.
fn token_eq(want: &str, got: &str) -> bool {
    let (w, g) = (want.as_bytes(), got.as_bytes());
    if w.len() != g.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..w.len() {
        diff |= w[i] ^ g[i];
    }
    diff == 0
}

/// Auth config for the origin guard: an optional bearer `token` (the only key
/// for privileged routes on a headless server that has one set) plus `gui_mode`,
/// which the GUI companion sets so its trusted webview still reaches privileged
/// routes via the origin allow-list even when a token is ALSO configured -- so
/// the CLI can drive OAIY Desktop without locking out its own UI.
#[derive(Clone)]
pub(super) struct AuthConfig {
    pub(super) token: Option<String>,
    pub(super) gui_mode: bool,
    /// Tokens minted by the pairing flow. A consumer that paired (the user
    /// approved it in the OAIY UI) presents one of these as its bearer — the
    /// production path for an untrusted-origin consumer like FormLogic Web. The
    /// guard checks it alongside the single configured `token`.
    pub(super) pairing: Option<crate::bridge::PairingHandle>,
}

impl AuthConfig {
    /// Does `presented` match the configured token, this process's own internal
    /// token, OR a live paired token?
    fn token_matches(&self, presented: &str) -> bool {
        // An empty bearer is never a match — a caller that sent nothing must not
        // pass because something on this side is also unset.
        if presented.is_empty() {
            return false;
        }
        if let Some(want) = self.token.as_deref() {
            if token_eq(want, presented) {
                return true;
            }
        }
        // The credential this process hands its own children (the CLI running a
        // flow). Same trust as the parent, by construction — it was spawned by
        // it — and it has no browser origin the guard could recognise instead.
        let internal = crate::internal_token();
        if !internal.is_empty() && token_eq(internal, presented) {
            return true;
        }
        if let Some(p) = &self.pairing {
            if let Ok(mgr) = p.lock() {
                return mgr.is_valid_token(presented);
            }
        }
        false
    }
}

/// Decide whether a privileged request is allowed. A matching bearer token
/// always passes. The trusted-origin allow-list is honored ONLY for the GUI
/// companion (gui_mode), which has a real, unspoofable webview origin. A headless
/// server has no webview — any local process can forge the `Origin` header — so it
/// trusts the token alone: headless WITH a token is token-only, and headless with
/// NO token has its privileged (command-defining / destructive) routes CLOSED (the
/// operator must set OAIY_SERVER_TOKEN to administer it; the CLI sends the bearer).
fn privileged_allowed(token_ok: bool, gui_mode: bool, _has_token: bool, origin_priv_ok: bool) -> bool {
    token_ok || (gui_mode && origin_priv_ok)
}

/// Gate mutating/exec requests (POST/PUT/DELETE/PATCH) on the `Origin` header.
/// Privileged (command-defining / destructive) paths require OAIY Desktop's own
/// origin and fail CLOSED on a missing Origin; other mutations keep the broad
/// loopback allow-list. GET reads and CORS preflight (OPTIONS) pass through.
pub(super) async fn origin_guard(
    State(auth): State<AuthConfig>,
    req: Request,
    next: Next,
) -> axum::response::Response {
    let m = req.method().clone();
    let path = req.uri().path();
    let public = m == Method::OPTIONS
        || (m == Method::POST && path == "/api/bridge/pairing")
        || ((m == Method::GET || m == Method::HEAD)
            && (path == "/api/health"
                // Which version runs and whether a newer release exists: what health half says already.
                // Open, so the headless server, which only reports a newer release, can be asked without a
                // token. (The desktop's listener does not take this exemption: see the restricted read below.)
                || path == UPDATE_STATUS_PATH
                || path == "/api/bridge/capabilities"
                || path.strip_prefix("/api/bridge/pairing/")
                    .is_some_and(|id| !id.is_empty() && !id.contains('/'))));
    // Default-deny on headless/network listeners, including new routes and
    // HEAD requests. Only discovery and user-approved pairing bootstrap are public.
    if !auth.gui_mode && !public
        && !bearer_token(&req).is_some_and(|got| auth.token_matches(&got))
    {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "authentication required"
        }))).into_response();
    }
    let mutating =
        m == Method::POST || m == Method::PUT || m == Method::DELETE || m == Method::PATCH;
    // A tiny public-mutation allow-list: raising a pairing request is a POST that
    // MUST be reachable from an untrusted origin — bootstrapping a token is its
    // whole purpose, and it grants nothing without the user's approval (a
    // separate, privileged act). Without this exemption the ordinary mutation
    // guard below would 403 FormLogic's very first call. Everything else stays
    // gated.
    if mutating && m == Method::POST && req.uri().path() == "/api/bridge/pairing" {
        return next.run(req).await;
    }
    if mutating {
        let privileged = is_privileged_path(&m, req.uri().path());
        let origin = req
            .headers()
            .get(ORIGIN)
            .and_then(|o| o.to_str().ok())
            .map(str::to_owned);
        // A configured bearer token lets a headless/non-browser admin client
        // (the CLI, oaiy-server tooling) perform privileged ops the origin
        // allow-list would otherwise block — there's no browser origin on a
        // server. Compared without per-byte short-circuit (token_eq).
        // A configured token OR a paired token satisfies auth.
        let token_ok = bearer_token(&req).is_some_and(|got| auth.token_matches(&got));
        let allowed = if privileged {
            let origin_priv_ok =
                matches!(origin.as_deref(), Some(o) if is_allowed_origin_privileged(o));
            privileged_allowed(token_ok, auth.gui_mode, auth.token.is_some(), origin_priv_ok)
        } else {
            // Origin is a browser CSRF check, never a headless credential.
            let origin_ok = match origin.as_deref() {
                Some(o) => is_allowed_origin(o),
                None => true, // native/CLI caller: no browser Origin to check
            };
            token_ok || (auth.gui_mode && origin_ok)
        };
        if !allowed {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": "origin not allowed" })),
            )
                .into_response();
        }
    } else if m == Method::GET || m == Method::HEAD {
        // A missing Origin proves nothing about a caller's location or authority.
        // Sensitive reads require a token outside the loopback GUI listener.
        let path = req.uri().path();
        let export_read = is_export_path(path);
        // The update status says whether a phone call is live (it is why "Restart to update" is off), so on the
        // desktop it is read like the calls themselves are: by OAIY's own pages or with the token, never by a page
        // the owner happens to have open. The headless server computes no such thing and answers it openly.
        let restricted_read = is_restricted_read_path(path) || (auth.gui_mode && path == UPDATE_STATUS_PATH);
        if export_read || restricted_read {
            let origin = req
                .headers()
                .get(ORIGIN)
                .and_then(|o| o.to_str().ok())
                .map(str::to_owned);
            let token_ok = bearer_token(&req).is_some_and(|got| auth.token_matches(&got));
            let origin_ok = origin.as_deref().is_some_and(|o| {
                if export_read { is_allowed_origin_privileged(o) } else { is_allowed_origin(o) }
            });
            let allowed = token_ok || (auth.gui_mode && origin_ok);
            if !allowed {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({ "error": "origin not allowed" })),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
}
