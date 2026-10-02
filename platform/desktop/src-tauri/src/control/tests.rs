use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Map, Value};
use tower::ServiceExt as _;

use super::audit;
use super::mcp::{INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR, PROTOCOL_VERSIONS};
use super::tools::{self, Kind};
use super::{Control, Navigator, Session, SWITCHED_OFF};

struct Sandbox(PathBuf);

impl Sandbox {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!("oaiy-control-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A control API whose tools call `desk` (none: an empty router, where every route is 404).
fn control_with(sb: &Sandbox, desk: Option<Router>, navigator: Option<Navigator>) -> (Control, Router) {
    let control = Control::with_navigator(&sb.0, navigator);
    let app = desk.unwrap_or_default().merge(super::router(control.clone()));
    control.set_router(app.clone());
    (control, app)
}

async fn send(app: &Router, method: Method, path: &str, headers: &[(&str, &str)], body: Option<String>) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(path);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let response = app.clone().oneshot(req.body(body.map_or_else(Body::empty, Body::from)).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// One JSON-RPC request to `/api/mcp`, as `session` (none: no header).
async fn rpc(app: &Router, session: Option<&str>, message: Value) -> (StatusCode, Value) {
    let mut headers = vec![];
    if let Some(s) = session {
        headers.push(("X-OAIY-Session", s));
    }
    send(app, Method::POST, "/api/mcp", &headers, Some(message.to_string())).await
}

async fn call(app: &Router, session: Option<&str>, tool: &str, args: Value) -> Value {
    let (status, body) = rpc(app, session, json!({ "jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": { "name": tool, "arguments": args } })).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], 7);
    body["result"].clone()
}

fn text(result: &Value) -> String {
    result["content"][0]["text"].as_str().unwrap_or_default().to_string()
}

fn is_error(result: &Value) -> bool {
    result["isError"] == Value::Bool(true)
}

async fn tool_names(app: &Router, session: Option<&str>) -> Vec<String> {
    let (_, body) = rpc(app, session, json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" })).await;
    body["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect()
}

// ---------------------------------------------------------------------------
// Who is asking
// ---------------------------------------------------------------------------

#[test]
fn the_session_header_decides_who_may_read_and_change() {
    for (header, session, read, change) in [
        (None, Session::Project, true, true),
        (Some("project"), Session::Project, true, true),
        (Some(" Setup "), Session::Setup, true, true),
        (Some("runner"), Session::Runner, true, false),
        (Some("call"), Session::Call, false, false),
        (Some("sms"), Session::Sms, false, false),
        (Some("task"), Session::Task, false, false),
        // A value this OAIY does not know, even an empty one, gets nothing.
        (Some(""), Session::Other, false, false),
        (Some("admin"), Session::Other, false, false),
    ] {
        let s = Session::from_header(header);
        assert_eq!(s, session, "{header:?}");
        assert_eq!((s.may_read(), s.may_change()), (read, change), "{header:?}");
    }
}

#[tokio::test]
async fn tools_list_offers_each_session_its_tools() {
    let sb = Sandbox::new("list");
    let (_, app) = control_with(&sb, None, None);
    let all: Vec<String> = tools::defs().iter().map(|d| d.name.to_string()).collect();
    let reads: Vec<String> = tools::defs().iter().filter(|d| d.kind == Kind::Read).map(|d| d.name.to_string()).collect();
    assert_eq!(tool_names(&app, None).await, all, "no header is project");
    assert_eq!(tool_names(&app, Some("project")).await, all);
    assert_eq!(tool_names(&app, Some("setup")).await, all);
    assert_eq!(tool_names(&app, Some("runner")).await, reads, "a runner reads only");
    for s in ["call", "sms", "task", "nobody"] {
        assert!(tool_names(&app, Some(s)).await.is_empty(), "{s} gets no tools");
    }
    // The contacts: a runner reads them; a caller or a texter is offered none, so no one on the
    // phone can have the receptionist read out, or change, what the person wrote about others.
    let runner = tool_names(&app, Some("runner")).await;
    for read in ["contacts_list", "contact_get"] {
        assert!(runner.iter().any(|t| t == read), "a runner reads {read}");
    }
    for change in ["contact_set", "contact_forget_fact", "ui_open"] {
        assert!(!runner.iter().any(|t| t == change), "a runner may not {change}");
        assert!(all.iter().any(|t| t == change), "a project may {change}");
    }
}

#[tokio::test]
async fn a_session_is_refused_the_tools_it_is_not_offered_and_a_change_attempt_is_logged() {
    let sb = Sandbox::new("refuse");
    let (control, app) = control_with(&sb, None, None);
    let r = call(&app, Some("runner"), "flow_delete", json!({ "id": "x" })).await;
    assert!(is_error(&r) && text(&r).contains("runner session may only read"), "{r}");
    let r = call(&app, Some("call"), "status", json!({})).await;
    assert!(is_error(&r) && text(&r).contains("not offered in call sessions"), "{r}");
    let r = call(&app, Some("sms"), "plugin_command", json!({ "pluginId": "aokie", "command": "sms.send" })).await;
    assert!(is_error(&r), "{r}");
    let log = control.audit().read(10);
    // Only the change attempts: a read is never logged.
    assert_eq!(log.len(), 2, "{log:?}");
    assert_eq!((log[0]["tool"].as_str(), log[0]["session"].as_str(), log[0]["ok"].as_bool()), (Some("plugin_command"), Some("sms"), Some(false)));
    assert_eq!((log[1]["tool"].as_str(), log[1]["session"].as_str()), (Some("flow_delete"), Some("runner")));
}

// ---------------------------------------------------------------------------
// The switch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_switch_is_on_by_default_and_off_refuses_changes_but_not_reads() {
    let sb = Sandbox::new("switch");
    let (control, app) = control_with(&sb, None, None);
    assert!(control.agent_may_change(), "no file: on");
    let (status, v) = send(&app, Method::GET, "/api/control/settings", &[], None).await;
    assert_eq!((status, v), (StatusCode::OK, json!({ "agentMayChange": true })));

    let (status, v) = send(&app, Method::PUT, "/api/control/settings", &[], Some(json!({ "agentMayChange": false }).to_string())).await;
    assert_eq!((status, v), (StatusCode::OK, json!({ "agentMayChange": false })));
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(sb.0.join(super::SETTINGS_FILE)).unwrap()).unwrap();
    assert_eq!(saved, json!({ "agentMayChange": false }));
    // A misspelt field is refused rather than read as "on".
    let (status, _) = send(&app, Method::PUT, "/api/control/settings", &[], Some(json!({ "agentMayChanges": true }).to_string())).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(!control.agent_may_change());

    let r = call(&app, None, "flow_delete", json!({ "id": "x" })).await;
    assert!(is_error(&r));
    assert_eq!(text(&r), SWITCHED_OFF);
    let r = call(&app, None, "setup_finish", json!({})).await;
    assert_eq!(text(&r), SWITCHED_OFF);
    // Reads still work (every route is 404 here, so the list is empty but answered).
    let r = call(&app, None, "status", json!({})).await;
    assert!(!is_error(&r), "{r}");
    let entry = &control.audit().read(1)[0];
    assert_eq!((entry["tool"].as_str(), entry["ok"].as_bool(), entry["summary"].as_str()), (Some("setup_finish"), Some(false), Some(SWITCHED_OFF)));

    control.set_agent_may_change(true).unwrap();
    assert!(control.agent_may_change());
}

#[test]
fn a_damaged_switch_file_is_off_and_an_empty_one_is_the_default() {
    let sb = Sandbox::new("switchfile");
    let control = Control::new(&sb.0);
    let file = sb.0.join(super::SETTINGS_FILE);
    std::fs::write(&file, "{ not json").unwrap();
    assert!(!control.agent_may_change(), "a file that cannot be read never switches the Agent on");
    std::fs::write(&file, r#"{"agentMayChange": "yes"}"#).unwrap();
    assert!(!control.agent_may_change());
    std::fs::write(&file, "{}").unwrap();
    assert!(control.agent_may_change(), "no field: the default");
    std::fs::write(&file, "\u{feff}{\"agentMayChange\": false}").unwrap();
    assert!(!control.agent_may_change(), "a byte-order mark is no reason to miss it");
}

/// Design 4.5.5: the Agent may change OAIY, when nobody has said, on a local install (as it always could) and not on
/// a proxied or lan one; what the person saved wins on all three, and a damaged file is off on all three.
#[tokio::test]
async fn t_agent_may_change_by_default_only_on_a_local_install_and_the_persons_word_wins_everywhere() {
    use crate::auth::mode::Exposure;
    for (exposure, default_on) in [
        (Exposure::Local, true),
        (Exposure::Proxied, false),
        (Exposure::Lan, false),
    ] {
        let sb = Sandbox::new("switch-exposure");
        let control = Control::with_exposure(&sb.0, exposure);
        let file = sb.0.join(super::SETTINGS_FILE);
        assert_eq!(control.agent_may_change(), default_on, "{exposure:?}: no file");
        // A file with no such field says nothing about it: `approveDangerous` will live there too.
        for body in ["{}", r#"{"approveDangerous":true}"#, r#"{"agentMayChange":null}"#] {
            std::fs::write(&file, body).unwrap();
            assert_eq!(control.agent_may_change(), default_on, "{exposure:?}: {body}");
        }
        // What the person saved, either way.
        for saved in [true, false] {
            std::fs::write(&file, json!({ "agentMayChange": saved }).to_string()).unwrap();
            assert_eq!(control.agent_may_change(), saved, "{exposure:?}: saved {saved}");
        }
        // A file that cannot be read, or says something that is not a switch, is off: never a default that is on.
        for damaged in ["{ not json", r#"{"agentMayChange":"yes"}"#, r#"{"agentMayChange":1}"#] {
            std::fs::write(&file, damaged).unwrap();
            assert!(!control.agent_may_change(), "{exposure:?}: {damaged}");
        }
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir(&file).unwrap();
        assert!(!control.agent_may_change(), "{exposure:?}: a folder where the file belongs");
        std::fs::remove_dir(&file).unwrap();
        // The switch's own routes and the change tools say the same.
        let app = super::router(control.clone());
        control.set_router(app.clone());
        let (status, v) = send(&app, Method::GET, "/api/control/settings", &[], None).await;
        assert_eq!((status, v), (StatusCode::OK, json!({ "agentMayChange": default_on })), "{exposure:?}");
        let r = call(&app, None, "setup_finish", json!({})).await;
        assert_eq!(text(&r) == SWITCHED_OFF, !default_on, "{exposure:?}: {r}");
        // The person turns it on, where it was off: the file says so from then on.
        let (status, v) = send(&app, Method::PUT, "/api/control/settings", &[], Some(json!({ "agentMayChange": true }).to_string())).await;
        assert_eq!((status, v), (StatusCode::OK, json!({ "agentMayChange": true })));
        assert!(control.agent_may_change());
    }
    // A local one built the old way is the same as `with_exposure(Local)`.
    let sb = Sandbox::new("switch-new");
    assert!(Control::new(&sb.0).agent_may_change());
}

// ---------------------------------------------------------------------------
// The audit log
// ---------------------------------------------------------------------------

#[test]
fn secrets_are_redacted_and_long_arguments_shortened() {
    let args = json!({
        "pluginId": "aokie",
        "apiKey": "abc",
        "settings": { "hfToken": "hf_abcdefghijklmnopqrstuvwxyz", "password": "p", "mode": "native", "list": [{ "secret": 1 }, { "fine": "ok" }] },
        "header": "Bearer abcdefghijklmnopqrstu",
        "note": "sk-proj-abcdefghijklmnopqrstuvwxyz",
        "idempotencyKey": "agent-1",
        "short": "sk-",
        "nothing": null,
    });
    let r = audit::redact(&args);
    assert_eq!(r["pluginId"], "aokie");
    for pointer in ["/apiKey", "/settings/hfToken", "/settings/password", "/settings/list/0/secret", "/header", "/note", "/idempotencyKey"] {
        assert_eq!(r.pointer(pointer).unwrap(), audit::REDACTED, "{pointer}");
    }
    assert_eq!(r["settings"]["mode"], "native");
    assert_eq!(r["settings"]["list"][1]["fine"], "ok");
    assert_eq!(r["short"], "sk-", "a bare prefix is not a secret");
    assert_eq!(r["nothing"], Value::Null);

    let long = "x".repeat(1000);
    let c = audit::compact_args(&json!({ "note": long, "list": (0..100).collect::<Vec<_>>() }));
    assert!(c["note"].as_str().unwrap().len() < 300 && c["note"].as_str().unwrap().contains("1000 characters"));
    assert_eq!(c["list"].as_array().unwrap().len(), 21);
    // A whole flow graph is only named.
    let nodes: Vec<Value> = (0..20).map(|i| json!({ "id": format!("n{i}"), "data": { "task": "y".repeat(190) } })).collect();
    let c = audit::compact_args(&json!({ "id": "big", "flow": { "name": "Big", "nodes": nodes } }));
    assert_eq!(c["id"], "big");
    assert_eq!(c["flow"], "{… 2 fields}");
    assert!(c.to_string().len() < 4096);
}

#[test]
fn the_log_is_read_newest_first_and_rolls() {
    let sb = Sandbox::new("audit");
    let log = audit::Log::new(sb.0.join(super::LOG_FILE));
    assert!(log.read(10).is_empty());
    log.append("flow_create", &json!({ "id": "a", "token": "t0p-s3cret" }), "project", true, "Stored the flow a.");
    log.append("flow_update", &json!({ "id": "a" }), "setup", false, "no");
    log.append("flow_delete", &json!({ "id": "a" }), "project", true, "Deleted flow a.");
    let entries = log.read(2);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["tool"], "flow_delete");
    assert_eq!(entries[1]["tool"], "flow_update");
    for key in ["at", "tool", "args", "session", "ok", "summary"] {
        assert!(entries[0].get(key).is_some(), "{key}");
    }
    let raw = std::fs::read_to_string(sb.0.join(super::LOG_FILE)).unwrap();
    assert!(!raw.contains("t0p-s3cret"), "nothing unredacted is ever written");
    assert_eq!(raw.lines().count(), 3);

    // Past its size it rolls, and a read still reaches the entries before.
    let big = "x".repeat(audit::MAX_BYTES as usize);
    std::fs::write(sb.0.join(super::LOG_FILE), format!("{}\n{big}\n", json!({ "tool": "old" }))).unwrap();
    log.append("flow_run", &json!({}), "project", true, "ran");
    assert!(sb.0.join("control-log.jsonl.1").is_file());
    let entries = log.read(5);
    assert_eq!(entries[0]["tool"], "flow_run");
    assert_eq!(entries[1]["tool"], "old", "the line that is not JSON is skipped, the rest kept");
}

#[tokio::test]
async fn the_log_route_answers_newest_first_with_a_limit() {
    let sb = Sandbox::new("logroute");
    let (control, app) = control_with(&sb, None, None);
    for i in 0..5 {
        control.audit().append(&format!("t{i}"), &json!({}), "project", true, "ok");
    }
    let (status, v) = send(&app, Method::GET, "/api/control/log?limit=3", &[], None).await;
    assert_eq!(status, StatusCode::OK);
    let tools: Vec<&str> = v["entries"].as_array().unwrap().iter().map(|e| e["tool"].as_str().unwrap()).collect();
    assert_eq!(tools, ["t4", "t3", "t2"]);
    let (_, v) = send(&app, Method::GET, "/api/control/log", &[], None).await;
    assert_eq!(v["entries"].as_array().unwrap().len(), 5);
}

// ---------------------------------------------------------------------------
// JSON-RPC
// ---------------------------------------------------------------------------

#[tokio::test]
async fn initialize_ping_and_the_protocol_version() {
    let sb = Sandbox::new("init");
    let (_, app) = control_with(&sb, None, None);
    let (status, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": "a", "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } } })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["id"], "a");
    assert_eq!(v["result"]["protocolVersion"], "2025-06-18", "a version it speaks is echoed");
    assert_eq!(v["result"]["serverInfo"]["name"], "oaiy");
    assert_eq!(v["result"]["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["result"]["capabilities"], json!({ "tools": {} }));
    let (_, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": 2, "method": "initialize", "params": { "protocolVersion": "1999-01-01" } })).await;
    assert_eq!(v["result"]["protocolVersion"], PROTOCOL_VERSIONS[0], "else its own");
    // Even a session with no tools may start and ping.
    let (_, v) = rpc(&app, Some("call"), json!({ "jsonrpc": "2.0", "id": 3, "method": "ping" })).await;
    assert_eq!(v["result"], json!({}));
    // The client's notification: accepted, nothing to answer.
    let (status, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).await;
    assert_eq!((status, v), (StatusCode::ACCEPTED, Value::Null));
}

#[tokio::test]
async fn json_rpc_errors_have_their_codes() {
    let sb = Sandbox::new("errors");
    let (_, app) = control_with(&sb, None, None);
    let code = |v: &Value| v["error"]["code"].as_i64();

    let (status, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": 1, "method": "resources/list" })).await;
    assert_eq!((status, code(&v)), (StatusCode::OK, Some(METHOD_NOT_FOUND)));
    assert_eq!(v["id"], 1);

    for params in [json!("x"), json!([1]), json!({ "arguments": {} }), json!({ "name": 5 }), json!({ "name": "status", "arguments": [] })] {
        let (_, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params })).await;
        assert_eq!(code(&v), Some(INVALID_PARAMS), "{params}");
    }
    let (_, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": 3, "method": "initialize", "params": [] })).await;
    assert_eq!(code(&v), Some(INVALID_PARAMS));
    let (_, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": { "cursor": 5 } })).await;
    assert_eq!(code(&v), Some(INVALID_PARAMS));

    // An unknown tool is a result the model can read, not a protocol error.
    let r = call(&app, None, "make_coffee", json!({})).await;
    assert!(is_error(&r) && text(&r).contains("no tool called \"make_coffee\""), "{r}");

    let (status, v) = send(&app, Method::POST, "/api/mcp", &[], Some("{ nope".into())).await;
    assert_eq!((status, code(&v)), (StatusCode::BAD_REQUEST, Some(PARSE_ERROR)));
    let (status, v) = rpc(&app, None, json!([{ "jsonrpc": "2.0", "id": 1, "method": "ping" }])).await;
    assert_eq!((status, code(&v)), (StatusCode::BAD_REQUEST, Some(INVALID_REQUEST)), "a batch is refused");
    let (_, v) = rpc(&app, None, json!({ "jsonrpc": "1.0", "id": 1, "method": "ping" })).await;
    assert_eq!(code(&v), Some(INVALID_REQUEST));
    let (_, v) = rpc(&app, None, json!({ "jsonrpc": "2.0", "id": true, "method": "ping" })).await;
    assert_eq!(code(&v), Some(INVALID_REQUEST));
    let (status, _) = send(&app, Method::GET, "/api/mcp", &[], None).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn arguments_are_checked_against_the_tool_and_the_error_names_the_fix() {
    let sb = Sandbox::new("args");
    let (control, app) = control_with(&sb, None, None);
    let r = call(&app, None, "flow_get", json!({})).await;
    assert!(is_error(&r) && text(&r).contains("needs \"id\""), "{r}");
    let r = call(&app, None, "flow_get", json!({ "id": 5 })).await;
    assert!(text(&r).contains("id must be a non-empty string"), "{r}");
    let r = call(&app, None, "flow_get", json!({ "id": "a", "flowId": "a" })).await;
    assert!(text(&r).contains("takes no argument \"flowId\": its arguments are id"), "{r}");
    let r = call(&app, None, "service_logs", json!({ "id": "a", "lines": 5000 })).await;
    assert!(text(&r).contains("from 1 to 500"), "{r}");
    let r = call(&app, None, "model_set_default", json!({ "group": "upscale", "model": "x" })).await;
    assert!(text(&r).contains("one of llm, image"), "{r}");
    // A change refused for its arguments is still a change attempt, and logged.
    assert_eq!(control.audit().read(1)[0]["tool"], "model_set_default");
}

// ---------------------------------------------------------------------------
// The declarations
// ---------------------------------------------------------------------------

#[test]
fn every_tool_is_declared_for_a_model() {
    let defs = tools::defs();
    let mut names: Vec<&str> = defs.iter().map(|d| d.name).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), defs.len(), "names are unique");
    for want in [
        "status", "models_list", "model_download_status", "model_set_default", "model_download", "engine_start", "engine_stop", "engine_restart",
        "ai_sources_list", "agent_model_set", "chatgpt_sign_in", "chatgpt_sign_out", "services_list", "service_logs", "service_install",
        "service_start", "service_stop", "service_uninstall", "plugins_list", "plugin_catalog", "plugin_settings_get", "plugin_setup_status",
        "plugin_install", "plugin_enable", "plugin_disable", "plugin_uninstall", "plugin_settings_set", "plugin_command", "plugin_setup_open",
        "plugin_setup_step_done", "flows_list", "flow_get", "flow_create", "flow_update", "flow_run", "flow_delete", "calendar_settings_get",
        "calendar_settings_set", "link_status", "link_sync_now", "setup_status", "setup_finish", "logs_tail",
        "contacts_list", "contact_get", "contact_set", "contact_forget_fact", "ui_open",
    ] {
        assert!(names.contains(&want), "{want} is declared");
    }
    let removals = ["chatgpt_sign_out", "service_uninstall", "plugin_uninstall", "flow_delete", "contact_forget_fact"];
    for d in defs {
        let l = d.listing();
        assert_eq!(l["inputSchema"]["type"], "object", "{}", d.name);
        assert!(!d.description.is_empty() && !d.description.contains('\n'), "{}", d.name);
        assert_eq!(l["annotations"]["readOnlyHint"], d.kind == Kind::Read, "{}", d.name);
        assert_eq!(l["annotations"]["destructiveHint"], removals.contains(&d.name), "{}", d.name);
        for req in l["inputSchema"]["required"].as_array().unwrap() {
            assert!(l["inputSchema"]["properties"].get(req.as_str().unwrap()).is_some(), "{} requires what it declares", d.name);
        }
    }
}

/// Arguments that pass a tool's schema.
fn valid_args(schema: &Value) -> Value {
    let mut out = Map::new();
    for req in schema["required"].as_array().unwrap() {
        let key = req.as_str().unwrap();
        let spec = &schema["properties"][key];
        let v = match (spec.get("enum"), spec["type"].as_str()) {
            (Some(e), _) => e[0].clone(),
            (_, Some("integer")) => json!(1),
            (_, Some("boolean")) => json!(true),
            (_, Some("object")) => json!({ "name": "x" }),
            (_, Some("array")) => json!([]),
            _ => json!("x"),
        };
        out.insert(key.into(), v);
    }
    Value::Object(out)
}

#[tokio::test]
async fn every_tool_runs_and_a_desktop_without_its_routes_is_explained() {
    // An OAIY with no routes at all: every tool must still answer, in words, and none may fall through.
    let sb = Sandbox::new("every");
    let (_, app) = control_with(&sb, None, None);
    for d in tools::defs() {
        let r = call(&app, None, d.name, valid_args(&d.schema)).await;
        assert!(r["content"][0]["text"].as_str().is_some_and(|t| !t.is_empty()), "{}: {r}", d.name);
        assert!(!text(&r).contains("has no handler"), "{}: {r}", d.name);
        assert!(r["structuredContent"].is_object(), "{}: {r}", d.name);
    }
    // The Agent's model preference belongs to a route this OAIY may not have yet: the answer says so.
    let r = call(&app, None, "agent_model_set", json!({ "source": "chatgpt" })).await;
    assert!(is_error(&r) && text(&r).contains("no PUT /api/agent/preferences"), "{r}");
    let r = call(&app, None, "agent_model_set", json!({ "source": "engine", "model": "gpt-5.5" })).await;
    assert!(is_error(&r) && text(&r).contains("chatgpt only"), "{r}");
    // Headless: no dashboard to show a page in, said honestly.
    let r = call(&app, None, "ui_open", json!({ "view": "plugins" })).await;
    assert!(is_error(&r) && text(&r).contains("headless"), "{r}");
    let r = call(&app, None, "logs_tail", json!({})).await;
    assert!(!is_error(&r) && text(&r).contains("headless"), "{r}");
}

// ---------------------------------------------------------------------------
// Through the desktop's own router
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_read_and_a_change_go_through_the_desktops_router_and_its_gate() {
    let sb = Sandbox::new("inprocess");
    let bridge = crate::build_bridge_state(sb.0.join("plugins"), sb.0.clone(), "device".into(), None);
    let control = Control::with_navigator(&sb.0, None);
    let routes = Router::new().merge(crate::bridge::bridge_router(bridge)).merge(super::router(control.clone()));
    // The real gate, as a headless server has it: nothing without a credential.
    let app = crate::http::guarded_for_tests(routes, Some("desk-token".into()), false);
    control.set_router(app.clone());
    let auth = [("Authorization", "Bearer desk-token")];

    // The gate is really there: the flows route refuses a stranger...
    let (status, _) = send(&app, Method::GET, "/api/bridge/flows", &[("Authorization", "Bearer wrong")], None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    // ...and /api/mcp itself.
    let (status, _) = send(&app, Method::POST, "/api/mcp", &[], Some(json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }).to_string())).await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let tool = |name: &'static str, args: Value| {
        let app = app.clone();
        async move {
            let (status, v) = send(&app, Method::POST, "/api/mcp", &auth, Some(json!({ "jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": { "name": name, "arguments": args } }).to_string())).await;
            assert_eq!(status, StatusCode::OK, "{v}");
            v["result"].clone()
        }
    };

    let r = tool("flows_list", json!({})).await;
    assert!(!is_error(&r), "{r}");
    assert_eq!(r["structuredContent"], json!({ "flows": [] }));

    let flow = json!({ "name": "Hello there", "nodes": [{ "id": "in", "type": "input_text" }], "edges": [] });
    let r = tool("flow_create", json!({ "flow": flow })).await;
    assert!(!is_error(&r), "{r}");
    assert_eq!(r["structuredContent"]["id"], "hello-there", "the id is made from its name");
    assert!(sb.0.join("flows").join("hello-there.json").is_file(), "stored by the desktop's own flows route");

    let r = tool("flow_get", json!({ "id": "hello-there" })).await;
    assert_eq!(r["structuredContent"]["flow"], flow);
    let r = tool("flow_create", json!({ "id": "hello-there", "flow": flow })).await;
    assert!(is_error(&r) && text(&r).contains("flow_update"), "{r}");
    let r = tool("flow_create", json!({ "flow": flow })).await;
    assert_eq!(r["structuredContent"]["id"], "hello-there-2", "a second of the same name gets its own id");
    let r = tool("flow_create", json!({ "id": "../evil", "flow": flow })).await;
    assert!(is_error(&r), "{r}");

    let r = tool("flow_update", json!({ "id": "hello-there", "flow": { "name": "Hello again", "nodes": [], "edges": [] } })).await;
    assert!(!is_error(&r), "{r}");
    let r = tool("flows_list", json!({})).await;
    assert_eq!(r["structuredContent"]["flows"], json!([{ "id": "hello-there", "name": "Hello again" }, { "id": "hello-there-2", "name": "Hello there" }]));
    let r = tool("flow_update", json!({ "id": "nope", "flow": {} })).await;
    assert!(is_error(&r) && text(&r).contains("flow_create"), "{r}");

    let r = tool("flow_delete", json!({ "id": "hello-there-2" })).await;
    assert!(!is_error(&r), "{r}");
    let r = tool("flow_get", json!({ "id": "hello-there-2" })).await;
    assert!(is_error(&r) && text(&r).contains("no flow"), "the route's own words: {r}");
    let r = tool("flow_delete", json!({ "id": "hello-there-2" })).await;
    assert!(is_error(&r), "{r}");

    // Only the changes are in the log, each with how it went.
    let log: Vec<(String, bool)> = control
        .audit()
        .read(20)
        .iter()
        .map(|e| (e["tool"].as_str().unwrap().to_string(), e["ok"].as_bool().unwrap()))
        .collect();
    let expect: Vec<(String, bool)> = [
        ("flow_delete", false),
        ("flow_delete", true),
        ("flow_update", false),
        ("flow_update", true),
        ("flow_create", false),
        ("flow_create", true),
        ("flow_create", false),
        ("flow_create", true),
    ]
    .iter()
    .map(|(t, ok)| (t.to_string(), *ok))
    .collect();
    assert_eq!(log, expect);
}

#[tokio::test]
async fn the_contact_tools_read_and_change_the_contacts_through_the_desktops_router() {
    use crate::voice::contacts::{routes, By, Store};
    let sb = Sandbox::new("contacts");
    let store = Store::at(sb.0.join("callers.json"));
    let control = Control::with_navigator(&sb.0, None);
    let app = crate::http::guarded_for_tests(Router::new().merge(routes::router(store.clone())).merge(super::router(control.clone())), Some("desk-token".into()), false);
    control.set_router(app.clone());
    let tool = |session: &'static str, name: &'static str, args: Value| {
        let app = app.clone();
        async move {
            let headers = [("Authorization", "Bearer desk-token"), ("X-OAIY-Session", session)];
            let (status, v) = send(&app, Method::POST, "/api/mcp", &headers, Some(json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": name, "arguments": args } }).to_string())).await;
            assert_eq!(status, StatusCode::OK, "{v}");
            v["result"].clone()
        }
    };

    // Named by the person's Agent: theirs, with one word for a name.
    let r = tool("project", "contact_set", json!({ "number": "+61 491 570 006", "name": "Liam", "notes": "Prefers texts" })).await;
    assert!(!is_error(&r), "{r}");
    assert_eq!(text(&r).lines().next(), Some("Saved Liam's name and notes."));
    let c = store.get("0491570006").unwrap();
    assert_eq!((c.name.as_str(), c.name_by, c.number.as_str()), ("Liam", Some(By::Owner), "+61 491 570 006"));
    store.add_fact("0491570006", "Has a dog called Max", By::Agent).unwrap();
    store.saw("+61400000001").unwrap();

    let r = tool("project", "contacts_list", json!({})).await;
    assert_eq!(r["structuredContent"]["total"], 2, "{r}");
    assert_eq!(r["structuredContent"]["contacts"][0], json!({ "key": "491570006", "number": "+61 491 570 006", "name": "Liam", "nameBy": "owner", "notes": "Prefers texts", "facts": 1 }));
    let r = tool("runner", "contacts_list", json!({ "q": "0491 570" })).await;
    assert_eq!(r["structuredContent"]["found"], 1, "a runner reads, and finds by a number written the local way: {r}");
    let r = tool("runner", "contact_get", json!({ "number": "0491570006" })).await;
    assert_eq!(r["structuredContent"]["facts"], json!([{ "index": 0, "text": "Has a dog called Max", "by": "agent", "at": store.get("0491570006").unwrap().facts[0].at }]));
    let r = tool("project", "contact_get", json!({ "number": "0499999999" })).await;
    assert!(is_error(&r) && text(&r).contains("no contact"), "the route's own words: {r}");

    // Forgotten; and a runner, a caller or a texter may not change them.
    let r = tool("runner", "contact_forget_fact", json!({ "number": "0491570006", "index": 0 })).await;
    assert!(is_error(&r) && text(&r).contains("runner session may only read"), "{r}");
    for s in ["call", "sms", "task"] {
        let r = tool(s, "contact_get", json!({ "number": "0491570006" })).await;
        assert!(is_error(&r) && text(&r).contains(&format!("not offered in {s} sessions")), "{r}");
    }
    let r = tool("project", "contact_forget_fact", json!({ "number": "+61491570006", "index": 0 })).await;
    assert!(!is_error(&r), "{r}");
    assert_eq!(text(&r).lines().next(), Some("Forgot \u{201c}Has a dog called Max\u{201d} about Liam."));
    assert!(store.get("0491570006").unwrap().facts.is_empty());
    let r = tool("project", "contact_forget_fact", json!({ "number": "0491570006", "index": 0 })).await;
    assert!(is_error(&r) && text(&r).contains("no fact 0"), "{r}");

    // Nothing to change, or a hidden number: said so.
    let r = tool("project", "contact_set", json!({ "number": "0491570006" })).await;
    assert!(is_error(&r) && text(&r).contains("Give name or notes"), "{r}");
    let r = tool("project", "contact_set", json!({ "number": "Private", "name": "Who" })).await;
    assert!(is_error(&r) && text(&r).contains("hidden"), "{r}");
    // An empty name clears it, and the receptionist may learn one again.
    let r = tool("setup", "contact_set", json!({ "number": "0491570006", "name": "" })).await;
    assert!(!is_error(&r), "{r}");
    assert_eq!(store.get("0491570006").unwrap().name_by, None);

    // The changes are in the log; the reads are not.
    let log: Vec<(String, String, bool)> = control
        .audit()
        .read(20)
        .iter()
        .map(|e| (e["tool"].as_str().unwrap().to_string(), e["session"].as_str().unwrap().to_string(), e["ok"].as_bool().unwrap()))
        .collect();
    let want: Vec<(String, String, bool)> = [
        ("contact_set", "setup", true),
        ("contact_set", "project", false),
        ("contact_set", "project", false),
        ("contact_forget_fact", "project", false),
        ("contact_forget_fact", "project", true),
        ("contact_forget_fact", "runner", false),
        ("contact_set", "project", true),
    ]
    .iter()
    .map(|(t, s, ok)| (t.to_string(), s.to_string(), *ok))
    .collect();
    assert_eq!(log, want);
}

#[tokio::test]
async fn the_calendar_settings_tools_carry_the_receptionists_name() {
    let sb = Sandbox::new("calendar");
    // The desktop's one calendar (another test may have opened it first: either will do, as only
    // the receptionist's name is looked at, and it is put back as it was).
    crate::calendar::init(&sb.0);
    let _on = crate::modules::test_gate::enable(&[crate::modules::CALENDAR]);
    let (_, app) = control_with(&sb, Some(crate::calendar::routes::router()), None);
    let get = || call(&app, None, "calendar_settings_get", json!({}));
    let before = get().await["structuredContent"]["settings"]["receptionist"].clone();

    // Only the name changes; the calendar keeps it trimmed, and says the name it goes by.
    let r = call(&app, None, "calendar_settings_set", json!({ "receptionist": "  Sam " })).await;
    assert!(!is_error(&r), "{r}");
    assert_eq!(text(&r).lines().next(), Some("Saved the calendar's receptionist."), "{r}");
    assert_eq!(r["structuredContent"]["settings"]["receptionist"], "Sam", "{r}");
    let r = get().await;
    assert_eq!((&r["structuredContent"]["settings"]["receptionist"], &r["structuredContent"]["receptionistName"]), (&json!("Sam"), &json!("Sam")), "{r}");

    // Longer than 40 characters: refused in the calendar's own words, and nothing changes.
    let r = call(&app, None, "calendar_settings_set", json!({ "receptionist": "A".repeat(41) })).await;
    assert!(is_error(&r) && text(&r).contains("at most 40 characters"), "{r}");
    assert_eq!(get().await["structuredContent"]["settings"]["receptionist"], "Sam");

    // Emptied: Aokie again.
    let r = call(&app, None, "calendar_settings_set", json!({ "receptionist": "" })).await;
    assert!(!is_error(&r), "{r}");
    let r = get().await;
    assert_eq!((&r["structuredContent"]["settings"]["receptionist"], &r["structuredContent"]["receptionistName"]), (&json!(""), &json!("Aokie")), "{r}");

    let r = call(&app, None, "calendar_settings_set", json!({ "receptionist": before })).await;
    assert!(!is_error(&r), "{r}");
}

// ---------------------------------------------------------------------------
// Showing a page
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_plugins_setup_step_is_shown_with_the_navigate_payload() {
    let sb = Sandbox::new("navigate");
    let shown: Arc<Mutex<Vec<Value>>> = Arc::default();
    let navigator: Navigator = {
        let shown = shown.clone();
        Arc::new(move |payload| {
            shown.lock().unwrap().push(payload);
            Ok(())
        })
    };
    let desk = Router::new()
        .route("/api/plugins", get(|| async { Json(json!({ "plugins": [{ "id": "demo", "state": "running", "manifest": { "name": "Demo" } }] })) }))
        // Its setup declares no permissions step: the host shows one first all the same.
        .route("/api/setup/plugins/demo", get(|| async { Json(json!({ "name": "Demo", "setup": { "version": 1, "steps": [{ "id": "pair", "kind": "screen" }] } })) }));
    let (_, app) = control_with(&sb, Some(desk), Some(navigator));

    let r = call(&app, Some("setup"), "plugin_setup_open", json!({ "pluginId": "demo", "stepId": "pair" })).await;
    assert!(!is_error(&r), "{r}");
    let r = call(&app, None, "plugin_setup_open", json!({ "pluginId": "demo" })).await;
    assert!(!is_error(&r), "{r}");
    let r = call(&app, None, "plugin_setup_open", json!({ "pluginId": "demo", "stepId": "fly" })).await;
    assert!(is_error(&r) && text(&r).contains("permissions, pair"), "{r}");
    let r = call(&app, None, "plugin_setup_open", json!({ "pluginId": "demo", "stepId": "permissions" })).await;
    assert!(!is_error(&r), "{r}");
    let r = call(&app, None, "plugin_setup_open", json!({ "pluginId": "ghost" })).await;
    assert!(is_error(&r) && text(&r).contains("demo"), "{r}");
    let r = call(&app, None, "ui_open", json!({ "view": "plugin:demo:home" })).await;
    assert!(!is_error(&r), "{r}");
    // Hours & Services is a page of its own (under the AI Receptionist in the dashboard).
    let r = call(&app, None, "ui_open", json!({ "view": "hours" })).await;
    assert!(!is_error(&r), "{r}");
    // The receptionist's Messages and Transfers pages too (the person is pointed to them: a message was kept, transfers are off).
    for view in ["messages", "transfers"] {
        let r = call(&app, None, "ui_open", json!({ "view": view })).await;
        assert!(!is_error(&r), "{view}: {r}");
    }
    let r = call(&app, None, "ui_open", json!({ "view": "../../etc" })).await;
    assert!(is_error(&r) && text(&r).contains("not a page"), "{r}");
    // Contacts, and one person's: their number written any way, sent as its key.
    let r = call(&app, None, "ui_open", json!({ "view": "contacts" })).await;
    assert!(!is_error(&r), "{r}");
    let r = call(&app, None, "ui_open", json!({ "view": "contacts", "contact": "+61 491 570 006" })).await;
    assert_eq!(r["structuredContent"], json!({ "view": "contacts", "contact": "491570006" }), "{r}");
    let r = call(&app, None, "ui_open", json!({ "view": "hours", "contact": "0491570006" })).await;
    assert!(is_error(&r) && text(&r).contains("goes with view contacts"), "{r}");
    let r = call(&app, None, "ui_open", json!({ "view": "contacts", "contact": "Private" })).await;
    assert!(is_error(&r) && text(&r).contains("not a phone number"), "{r}");
    assert_eq!(
        *shown.lock().unwrap(),
        vec![
            json!({ "view": "setup", "pluginId": "demo", "stepId": "pair" }),
            json!({ "view": "setup", "pluginId": "demo" }),
            json!({ "view": "setup", "pluginId": "demo", "stepId": "permissions" }),
            json!({ "view": "plugin:demo:home" }),
            json!({ "view": "hours" }),
            json!({ "view": "messages" }),
            json!({ "view": "transfers" }),
            json!({ "view": "contacts" }),
            json!({ "view": "contacts", "contact": "491570006" }),
        ]
    );
    // The person accepts what a plugin may do: the Agent shows it rather than recording it.
    let r = call(&app, None, "plugin_setup_step_done", json!({ "pluginId": "demo", "stepId": "permissions" })).await;
    assert!(is_error(&r) && text(&r).contains("plugin_setup_open"), "{r}");
}

/// The pages the Agent may ask to be shown are the dashboard's own: every view the dashboard can be navigated to (`NAV_VIEWS` in navigate.ts) is one the
/// tool names, except the one that is a setting of the Agent's own (`agent-settings`), and the tool's own `setup` is the dashboard's setup page. The tool's
/// description lists them, so a page the dashboard has and the description does not name is one the Agent cannot be asked to show.
#[test]
fn the_pages_the_agent_may_show_are_the_dashboards_own_and_the_description_names_each() {
    let navigate = include_str!("../../../src/navigate.ts");
    let list = navigate.split("export const NAV_VIEWS = [").nth(1).and_then(|rest| rest.split("] as const;").next()).expect("the dashboard's views are listed in navigate.ts");
    let dashboard: Vec<String> = regex::Regex::new(r"'([a-z-]+)',").unwrap().captures_iter(list).map(|c| c[1].to_string()).collect();
    assert!(dashboard.len() >= 17, "{dashboard:?}");
    for view in &dashboard {
        if view == "agent-settings" {
            continue;
        }
        assert!(super::tools::VIEWS.contains(&view.as_str()), "the dashboard has a {view} page that ui_open cannot show");
    }
    for view in super::tools::VIEWS {
        assert!(view == "setup" || dashboard.iter().any(|d| d == view), "ui_open names a {view} page the dashboard does not have");
    }
    let description = super::tools::defs().iter().find(|t| t.name == "ui_open").expect("ui_open").description;
    for view in super::tools::VIEWS {
        assert!(description.contains(view), "the description of ui_open does not name {view}");
    }
}

// ---------------------------------------------------------------------------
// The engines relay
// ---------------------------------------------------------------------------

/// A stand-in for the engines' control port: a configuration (with secrets
/// in it), a discovery document, and the language model's controls.
async fn fake_engines() -> (String, Arc<Mutex<Vec<Value>>>, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    let saved: Arc<Mutex<Vec<Value>>> = Arc::default();
    let config = || {
        json!({
            "gateway": { "api_key": "sk-gateway-secret" },
            "downloads": { "hf_token": "hf_secret" },
            "llm": { "default_model": "", "models": [{ "name": "qwen" }, { "name": "gemma" }] },
            "media": { "image": { "default_model": "", "models": { "flux": {} } }, "picture": { "background": "", "upscaler": "" } }
        })
    };
    let recorder = saved.clone();
    let app = Router::new()
        .route(
            "/api/config",
            get(move || async move { Json(config()) }).put(move |Json(body): Json<Value>| {
                let recorder = recorder.clone();
                async move {
                    let model = body["llm"]["default_model"].as_str().unwrap_or_default().to_string();
                    if !model.is_empty() && model != "qwen" && model != "gemma" {
                        return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("llm.default_model {model} is not one of the listed models") })));
                    }
                    recorder.lock().unwrap().push(body.clone());
                    (StatusCode::OK, Json(body))
                }
            }),
        )
        .route(
            "/api/discovery",
            get(|| async { Json(json!({ "defaults": { "llm": "qwen", "image": "" }, "models": { "llm": [{ "id": "qwen", "default": true }, { "id": "gemma" }], "image": [] } })) }),
        )
        .route("/api/llm/:action", post(|| async { Json(json!({ "state": "starting", "resident": null, "models": ["qwen"], "error": null, "command": "secret command line" })) }))
        .route("/api/logs", get(|| async { Json(json!({ "lines": [{ "n": 1, "t": 0, "line": "one" }, { "n": 2, "t": 0, "line": "two" }] })) }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ui = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (ui, saved, server)
}

#[tokio::test]
async fn choosing_a_model_writes_only_that_groups_default_and_answers_no_secret() {
    use super::engines::{defaults_at, llm_at, logs_at, set_default_at};
    let (ui, saved, server) = fake_engines().await;

    let v = defaults_at(Some(ui.clone())).await;
    assert_eq!(v["running"], true);
    assert_eq!(v["defaults"]["llm"], "qwen");
    assert_eq!(v["defaults"]["image"], Value::Null, "an empty default is no model");
    assert_eq!(v["models"]["llm"], json!(["qwen", "gemma"]));
    assert_eq!(v["models"]["upscale"], json!([]));

    let v = set_default_at(Some(ui.clone()), "llm", " gemma ").await.unwrap();
    assert_eq!(v, json!({ "group": "llm", "model": "gemma" }));
    assert!(!v.to_string().contains("secret"));
    let written = saved.lock().unwrap().last().unwrap().clone();
    assert_eq!(written["llm"]["default_model"], "gemma");
    assert_eq!(written["gateway"]["api_key"], "sk-gateway-secret", "the rest of the configuration goes back as it was");
    set_default_at(Some(ui.clone()), "image", "flux").await.unwrap();
    assert_eq!(saved.lock().unwrap().last().unwrap()["media"]["image"]["default_model"], "flux");

    let (status, why) = set_default_at(Some(ui.clone()), "llm", "ghost").await.unwrap_err();
    assert_eq!((status, why.as_str()), (400, "llm.default_model ghost is not one of the listed models"), "the engines' own refusal");
    assert_eq!(set_default_at(Some(ui.clone()), "upscale", "x").await.unwrap_err().0, 400);
    assert_eq!(set_default_at(Some(ui.clone()), "video3d", "x").await.unwrap_err().0, 400);
    assert_eq!(set_default_at(Some(ui.clone()), "llm", "").await.unwrap_err().0, 400);
    assert_eq!(set_default_at(None, "llm", "qwen").await.unwrap_err().0, 409);

    let v = llm_at(Some(ui.clone()), "start").await.unwrap();
    assert_eq!(v, json!({ "llm": { "state": "starting", "resident": null, "models": ["qwen"], "error": null } }));
    assert_eq!(llm_at(Some(ui.clone()), "explode").await.unwrap_err().0, 404);
    assert_eq!(logs_at(Some(ui.clone()), "llm", 1).await.unwrap(), json!({ "source": "llm", "lines": ["two"] }));
    assert_eq!(logs_at(Some(ui.clone()), "gpu", 1).await.unwrap_err().0, 400);
    assert_eq!(defaults_at(None).await["running"], false);
    server.abort();
}

// ---------------------------------------------------------------------------
// Whether a plugin is set up: the dashboard's rule
// ---------------------------------------------------------------------------

#[test]
fn set_up_by_checks_counts_the_shown_steps_with_a_done_check() {
    use super::tools::set_up_by_checks as by;
    assert_eq!(by(&[]), None, "no step to judge by");
    assert_eq!(by(&[(None, None), (Some(true), None)]), None, "no step has a done check");
    assert_eq!(by(&[(None, Some(true))]), Some(true));
    assert_eq!(by(&[(None, Some(true)), (None, Some(false))]), Some(false), "every shown one must pass");
    assert_eq!(by(&[(Some(false), Some(false)), (None, Some(true))]), Some(true), "a step that does not show is not counted");
    assert_eq!(by(&[(Some(false), Some(true))]), Some(false), "at least one must count");
    assert_eq!(by(&[(Some(true), Some(true)), (None, None)]), Some(true));
}

/// A desktop with four plugins, each with a setup of version 1:
/// - `live`: never finished here, but its `pair` step's done check passes (its
///   `extra` step does not show, and would fail): set up, by its checks;
/// - `off`: the same, but switched off: its checks are not run;
/// - `plain`: only a settings step, so nothing to judge live: needs setup;
/// - `old`: finished here: set up, by the record.
/// `pair_passes` decides whether live's pair check passes; every check asked is recorded.
fn setup_desk(pair_passes: Arc<Mutex<bool>>, asked: Arc<Mutex<Vec<String>>>) -> Router {
    use axum::extract::{Path, Query};
    use axum::routing::post;
    let steps = json!([
        { "id": "pair", "kind": "screen", "title": "Pair", "screen": "s", "view": "pair", "done": { "command": "phone.status", "path": "paired", "equals": true } },
        { "id": "extra", "kind": "screen", "title": "Extra", "screen": "s", "view": "x", "when": { "command": "phone.status", "path": "extra", "equals": true }, "done": { "command": "phone.status", "path": "x", "equals": true } },
    ]);
    let plugin = |id: &str, disabled: bool, steps: &Value| json!({ "id": id, "state": "running", "userDisabled": disabled, "manifest": { "name": id, "version": "1", "setup": { "version": 1, "title": "Set up", "steps": steps } } });
    let plugins = json!({ "plugins": [
        plugin("live", false, &steps),
        plugin("off", true, &steps),
        plugin("plain", false, &json!([{ "id": "prefs", "kind": "settings", "title": "Prefs", "fields": [{ "key": "a", "label": "A", "type": "text" }] }])),
        plugin("old", false, &steps),
    ] });
    let record = json!({ "firstRun": { "finished": true, "skipped": [], "chosenPlugins": [] }, "plugins": { "old": { "version": 1, "done": [], "skipped": [] } } });
    let detail = json!({ "pluginId": "live", "name": "live", "setup": { "version": 1, "title": "Set up", "steps": steps }, "capabilities": ["connector.live.phone.status"],
        "permissionsAccepted": false, "needsSetup": true, "state": { "version": 0, "done": [], "skipped": [] } });
    #[derive(serde::Deserialize)]
    struct Which {
        check: Option<String>,
    }
    Router::new()
        .route("/api/plugins", get(move || { let plugins = plugins.clone(); async move { Json(plugins) } }))
        .route("/api/setup", get(move || { let record = record.clone(); async move { Json(record) } }))
        .route("/api/setup/plugins/live", get(move || { let detail = detail.clone(); async move { Json(detail) } }))
        .route(
            "/api/setup/plugins/:id/check/:step",
            post(move |Path((id, step)): Path<(String, String)>, Query(q): Query<Which>| {
                let (pair_passes, asked) = (pair_passes.clone(), asked.clone());
                async move {
                    let which = q.check.unwrap_or_else(|| "done".into());
                    asked.lock().unwrap().push(format!("{id}/{step}/{which}"));
                    let passed = match (step.as_str(), which.as_str()) {
                        ("pair", "done") => *pair_passes.lock().unwrap(),
                        _ => false,
                    };
                    Json(json!({ "passed": passed, "detail": format!("{step} {which}: {passed}") }))
                }
            }),
        )
}

#[tokio::test]
async fn a_plugin_counts_as_set_up_when_its_shown_done_checks_pass_as_the_dashboard_judges_it() {
    let sb = Sandbox::new("liveset");
    let pair_passes = Arc::new(Mutex::new(true));
    let asked: Arc<Mutex<Vec<String>>> = Arc::default();
    let (_, app) = control_with(&sb, Some(setup_desk(pair_passes.clone(), asked.clone())), None);
    let needs = |rows: &Value| -> Vec<(String, Value, Value)> {
        rows.as_array().unwrap().iter().map(|r| (r["id"].as_str().unwrap().to_string(), r["needsSetup"].clone(), r.get("setUpBy").cloned().unwrap_or(Value::Null))).collect()
    };
    let expect = |live_needs: bool| {
        vec![
            ("live".to_string(), json!(live_needs), if live_needs { Value::Null } else { json!("checks") }),
            ("off".to_string(), json!(true), Value::Null),
            ("plain".to_string(), json!(true), Value::Null),
            ("old".to_string(), json!(false), json!("record")),
        ]
    };

    let r = call(&app, None, "plugins_list", json!({})).await;
    assert_eq!(needs(&r["structuredContent"]["plugins"]), expect(false), "{r}");
    // Checks ran only for the plugin that is on, has done checks, and was never finished here.
    assert!(asked.lock().unwrap().iter().all(|a| a.starts_with("live/")), "{:?}", asked.lock().unwrap());
    assert!(asked.lock().unwrap().contains(&"live/extra/when".to_string()));

    let r = call(&app, None, "status", json!({})).await;
    assert_eq!(needs(&r["structuredContent"]["plugins"]), expect(false));
    // The nudge list leaves out a plugin that is switched off, as the dashboard does.
    assert_eq!(r["structuredContent"]["setup"]["pluginsNeedingSetup"], json!(["plain"]));

    let r = call(&app, None, "setup_status", json!({})).await;
    assert_eq!(needs(&r["structuredContent"]["plugins"]), expect(false));

    let r = call(&app, None, "plugin_setup_status", json!({ "pluginId": "live" })).await;
    let v = &r["structuredContent"];
    assert_eq!((v["needsSetup"].clone(), v["setUpBy"].clone(), v["outstanding"].clone()), (json!(false), json!("checks"), json!([])), "{r}");
    assert_eq!(v["steps"][0]["id"], "permissions", "the host's permissions step comes first");
    assert_eq!(v["steps"][2]["applies"], false, "extra does not show");

    // The pair check fails now: it needs setup again, everywhere.
    *pair_passes.lock().unwrap() = false;
    let r = call(&app, None, "plugins_list", json!({})).await;
    assert_eq!(needs(&r["structuredContent"]["plugins"]), expect(true));
    let r = call(&app, None, "plugin_setup_status", json!({ "pluginId": "live" })).await;
    let v = &r["structuredContent"];
    assert_eq!((v["needsSetup"].clone(), v["setUpBy"].clone()), (json!(true), Value::Null));
    assert_eq!(v["outstanding"], json!(["permissions", "pair"]), "{r}");
}

#[tokio::test]
async fn a_hidden_setting_read_back_is_never_written_over_the_real_one() {
    use axum::extract::Path;
    use axum::routing::post;
    let sb = Sandbox::new("hidden");
    let sent: Arc<Mutex<Vec<Value>>> = Arc::default();
    let desk = {
        let sent = sent.clone();
        Router::new()
            .route(
                "/api/plugins",
                get(|| async {
                    Json(json!({ "plugins": [{ "id": "demo", "state": "running", "manifest": { "name": "Demo",
                        "connectors": [{ "id": "demo", "commands": ["settings.get", "settings.set"] }] } }] }))
                }),
            )
            .route(
                "/api/bridge/connectors/:id/request",
                post(move |Path(_id): Path<String>, Json(body): Json<Value>| {
                    let sent = sent.clone();
                    async move {
                        sent.lock().unwrap().push(body.clone());
                        let settings = json!({ "apiKey": "sk-live-abcdefghijklmnopqrstuvwxyz", "mood": "calm", "relay": { "token": "t0k3n", "host": "h" } });
                        Json(json!({ "ok": true, "result": { "ok": true, "data": { "settings": settings } } }))
                    }
                }),
            )
    };
    let (_, app) = control_with(&sb, Some(desk), None);

    let r = call(&app, None, "plugin_settings_get", json!({ "pluginId": "demo" })).await;
    let shown = r["structuredContent"]["settings"].clone();
    assert_eq!(shown, json!({ "apiKey": audit::REDACTED, "mood": "calm", "relay": { "token": audit::REDACTED, "host": "h" } }), "{r}");
    // A command's answer hides them too.
    let r = call(&app, None, "plugin_command", json!({ "pluginId": "demo", "command": "settings.get" })).await;
    assert!(!r.to_string().contains("sk-live") && !r.to_string().contains("t0k3n"), "{r}");

    // What was read, written back whole with one change: the hidden values are left out.
    let mut back = shown.clone();
    back["mood"] = json!("cheerful");
    let before = sent.lock().unwrap().len();
    let r = call(&app, None, "plugin_settings_set", json!({ "pluginId": "demo", "settings": back })).await;
    assert!(!is_error(&r) && text(&r).contains("apiKey, relay.token"), "{r}");
    let payload = sent.lock().unwrap()[before]["payload"].clone();
    assert_eq!(payload, json!({ "mood": "cheerful", "relay": { "host": "h" } }));

    // Only hidden values: nothing to send, and the model is told why.
    let before = sent.lock().unwrap().len();
    let r = call(&app, None, "plugin_settings_set", json!({ "pluginId": "demo", "settings": { "apiKey": audit::REDACTED } })).await;
    assert!(is_error(&r) && text(&r).contains("hidden"), "{r}");
    assert_eq!(sent.lock().unwrap().len(), before, "nothing reached the plugin");
}

#[tokio::test]
async fn setup_finish_after_the_hand_off_succeeds_without_changing_anything() {
    use axum::routing::put;
    let sb = Sandbox::new("finish");
    let finished = Arc::new(Mutex::new(false));
    let puts: Arc<Mutex<Vec<Value>>> = Arc::default();
    let desk = {
        let (f1, f2, p) = (finished.clone(), finished.clone(), puts.clone());
        Router::new().route(
            "/api/setup",
            get(move || {
                let f = f1.clone();
                async move { Json(json!({ "firstRun": { "finished": *f.lock().unwrap(), "position": "agent", "skipped": [], "chosenPlugins": ["aokie"], "migrated": null }, "plugins": {} })) }
            })
            .merge(put(move |Json(body): Json<Value>| {
                let (f, p) = (f2.clone(), p.clone());
                async move {
                    *f.lock().unwrap() = body["firstRun"]["finished"] == json!(true);
                    p.lock().unwrap().push(body.clone());
                    Json(body)
                }
            })),
        )
    };
    let (control, app) = control_with(&sb, Some(desk), None);
    let r = call(&app, Some("setup"), "setup_finish", json!({})).await;
    assert!(!is_error(&r), "{r}");
    let sent = puts.lock().unwrap().clone();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["firstRun"]["finished"], true);
    assert_eq!(sent[0]["firstRun"]["chosenPlugins"], json!(["aokie"]), "the rest of the record is kept");
    assert!(sent[0]["firstRun"].get("migrated").is_none(), "the desktop's own note is not sent back");
    // Already finished (as the dashboard's hand-off to the Agent records it): a success, and nothing sent.
    let r = call(&app, Some("setup"), "setup_finish", json!({})).await;
    assert!(!is_error(&r) && text(&r).contains("already"), "{r}");
    assert_eq!(puts.lock().unwrap().len(), 1);
    assert_eq!(control.audit().read(1)[0]["ok"], true);
}

#[test]
fn the_link_status_tool_says_why_a_stored_link_is_not_in_use() {
    // `linked: false` with a `linkError` is a link file that could not be read, or a link that could not be forgotten:
    // the Agent that is asked "is this linked?" must not answer "no" and stop there.
    let why = "link/account.json is not a link this version of OAIY understands (an unknown field, line 1, column 99). It has not been changed.";
    let unusable = json!({"linked": false, "linkError": {"file": "link/account.json", "message": why}});
    let summary = tools::link_summary(&unusable, Err("no sync".into()));
    assert_eq!(summary["linked"], json!(false));
    assert_eq!(summary["linkError"]["message"], json!(why));
    assert_eq!(summary["problem"], json!(why), "the one line the tool leads with");

    // A link that could not be forgotten is still linked, and says so.
    let stuck = json!({"linked": true, "connectorName": "FormLogic", "heartbeatError": "HTTP 500", "linkError": {"file": "link/account.json", "message": "the link could not be forgotten"}});
    let summary = tools::link_summary(&stuck, Err("no sync".into()));
    assert_eq!((summary["linked"].clone(), summary["problem"].clone()), (json!(true), json!("the link could not be forgotten")));

    // Without one, the problem is what it was: the heartbeat's, then the command lane's, else none; and there is no linkError.
    let beat = tools::link_summary(&json!({"linked": true, "heartbeatError": "HTTP 500", "relayError": "late"}), Err("no sync".into()));
    assert_eq!((beat["problem"].clone(), beat["linkError"].clone()), (json!("HTTP 500"), Value::Null));
    let lane = tools::link_summary(&json!({"linked": true, "relayError": "late"}), Err("no sync".into()));
    assert_eq!(lane["problem"], json!("late"));
    let fine = tools::link_summary(&json!({"linked": true, "linkError": null}), Err("no sync".into()));
    assert_eq!((fine["problem"].clone(), fine["linkError"].clone()), (Value::Null, Value::Null));
}