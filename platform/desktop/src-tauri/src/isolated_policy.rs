//! Deliberately small surface of the opt-in local plugin qualification process.
//! This limits OAIY operations, not the native code of a plugin the owner trusts.

pub const REFUSAL: &str = "This operation is unavailable in an isolated local plugin qualification launch.";

pub fn ipc_allowed(command: &str) -> bool {
    matches!(command, "get_config" | "log_path" | "list_model_dirs" | "get_hf_token_status" | "hide_embedded")
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

    #[test]
    fn native_settings_and_execution_commands_cannot_bypass_http_refusals() {
        assert!(ipc_allowed("get_config"));
        for command in ["set_data_dir", "set_models_dir", "start_migration", "set_hf_token", "list_gpus",
            "show_embedded", "agent_intent", "set_theme", "open_url", "open_path", "restart_app",
            "backup_restore_stage", "update_check", "update_install", "unknown"] {
            assert!(!ipc_allowed(command), "{command}");
        }
    }
}
