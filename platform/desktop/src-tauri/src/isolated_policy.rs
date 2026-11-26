//! Deliberately small surface of the opt-in local plugin qualification process.
//! This limits OAIY operations, not the native code of a plugin the owner trusts.

pub const REFUSAL: &str = "This operation is unavailable in an isolated local plugin qualification launch.";

/// `window_token` is the one command here that hands the page a credential: the dashboard's key for this process's
/// own API, which answers on the launch's reserved port only and behind [`guard`]. The isolated dashboard's origin is
/// `http://tauri.localhost` like the ordinary one's, a name a page on this computer's port 80 has too, so its API
/// takes the origin for the window only with that key (`http::tighten`). The launch context itself still carries a
/// port and nothing else (`isolated::Launch::initialization_script`).
pub fn ipc_allowed(command: &str) -> bool {
    matches!(command, "get_config" | "log_path" | "list_model_dirs" | "get_hf_token_status" | "hide_embedded" | "window_token")
}

pub fn http_allowed(method: &str, path: &str) -> bool {
    // Match canonical literal path components. Encoded separators, duplicate slashes and
    // dot components cannot turn a denied route into an allowed one after routing.
    if path.contains('%') || path.contains('\\') || path.contains("//") {
        return false;
    }
    let parts: Vec<_> = path.split('/').skip(1).collect();
    if parts.iter().any(|p| p.is_empty() || *p == "." || *p == "..") {
        return false;
    }
    if method == "GET" && matches!(path,
        "/api/health" | "/api/config" | "/api/plugins" |
        "/api/modules" | "/api/update/status") {
        return true;
    }
    match (method, parts.as_slice()) {
        ("POST", ["api", "plugins", "install"]) => true,
        ("POST", ["api", "plugins", _, "start" | "stop" | "trust" | "enabled"]) => true,
        ("GET", ["api", "plugins", _, "logs"]) => true,
        ("GET", ["api", "plugins", _, "ui", _, rest @ ..]) => !rest.is_empty(),
        ("POST", ["api", "bridge", "connectors", _, "request"]) => true,
        // The scoped handler still requires a trusted live plugin, its manifest
        // capability and an explicitly configured keyless loopback provider.
        ("GET", ["api", "plugins", _, "ai", "sources"]) => true,
        ("POST", ["api", "plugins", _, "ai", "complete" | "cancel"]) => true,
        _ => false,
    }
}

pub async fn guard(req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    use axum::response::IntoResponse;
    // One bounded startup diagnostic: confirms the bundled Windows dashboard
    // reached this process's API, without logging headers, bodies or credentials.
    if req.method() == axum::http::Method::GET && req.uri().path() == "/api/plugins"
        && req.headers().get("origin").and_then(|v| v.to_str().ok()) == Some("http://tauri.localhost") {
        static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!("isolated dashboard requested plugin inventory");
        }
    }
    if http_allowed(req.method().as_str(), req.uri().path()) {
        next.run(req).await
    } else {
        (axum::http::StatusCode::FORBIDDEN, axum::Json(serde_json::json!({
            "ok": false, "error": {"code": "isolated_capability_unavailable", "message": REFUSAL}
        }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::any, middleware};
    use tower::ServiceExt;

    #[tokio::test]
    async fn allowed_plugin_calls_reach_the_handler_and_other_capabilities_do_not() {
        let app = Router::new().fallback(any(|| async { "called" })).layer(middleware::from_fn(guard));
        for (method, path, expected) in [
            ("GET", "/api/health", 200), ("GET", "/api/plugins", 200),
            ("POST", "/api/plugins/install", 200), ("POST", "/api/plugins/example/trust", 200),
            ("POST", "/api/plugins/example/start", 200), ("POST", "/api/plugins/example/stop", 200),
            ("GET", "/api/plugins/example/ui/main/assets/app.js", 200),
            ("POST", "/api/bridge/connectors/example/request", 200),
            ("GET", "/api/plugins/example/ai/sources", 200),
            ("POST", "/api/plugins/example/ai/complete", 200),
            ("POST", "/api/plugins/example/ai/cancel", 200),
            ("POST", "/api/ai/providers/test/v1/chat/completions", 403),
            ("POST", "/api/ai/providers", 403),
            ("GET", "/api/ai/codex/status", 403), ("GET", "/api/engines/catalog", 403),
            ("POST", "/api/services/example/start", 403), ("GET", "/api/bridge/status", 403),
            ("POST", "/api/mcp", 403), ("POST", "/api/bridge/runs", 403),
            ("POST", "/api/setup/plugins/example/check/foo", 403), ("GET", "/api/link", 403),
            ("POST", "/api/update/check", 403), ("GET", "/api/unknown-future-route", 403),
            ("POST", "/api/plugins/example/start/extra", 403), ("DELETE", "/api/plugins/example", 403),
            ("POST", "/api/bridge/connectors/../request", 403),
            ("POST", "/api/bridge/connectors/a%2fb/request", 403),
        ] {
            let response = app.clone().oneshot(axum::http::Request::builder().method(method).uri(path).body(axum::body::Body::empty()).unwrap()).await.unwrap();
            assert_eq!(response.status().as_u16(), expected, "{method} {path}");
            if expected == 403 {
                let bytes = axum::body::to_bytes(response.into_body(), 2048).await.unwrap();
                let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["code"], "isolated_capability_unavailable");
            }
        }
    }

    /// The isolated dashboard's origin is `http://tauri.localhost` like the ordinary one's, which its API takes for the
    /// window only with the window's key: the command that hands the key over is one it may call (without it every page
    /// of the qualification window would be refused), and it is the only one of them that hands over a credential.
    #[test]
    fn the_isolated_dashboard_may_ask_for_its_own_windows_key() {
        assert!(ipc_allowed("window_token"));
        for near in ["window_token ", "Window_token", "window_tokens", "token", "internal_token", "get_hf_token", "set_hf_token"] {
            assert!(!ipc_allowed(near), "{near}");
        }
        // The command is one the application registers (a name allowed here and registered nowhere would be refused as
        // unknown, and the window left without its key).
        let lib = include_str!("lib.rs");
        assert!(lib.contains("async fn window_token(") && lib.contains("            window_token,"), "window_token is registered in the handler");
    }

    #[test]
    fn native_settings_and_execution_commands_cannot_bypass_http_refusals() {
        assert!(ipc_allowed("get_config"));
        for command in ["set_data_dir", "set_models_dir", "start_migration", "set_hf_token", "list_gpus",
            "show_embedded", "agent_intent", "flows_intent", "set_theme", "open_url", "open_path", "restart_app",
            "backup_restore_stage", "update_check", "update_install", "unknown"] {
            assert!(!ipc_allowed(command), "{command}");
        }
    }
}
