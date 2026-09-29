//! The control tools: their declarations (what `tools/list` answers) and what
//! each does, which is always a call to one or more of the desktop's own
//! routes, in-process (see [`Desk`]).

use std::sync::OnceLock;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use serde_json::{json, Map, Value};
use tower::ServiceExt as _;

use super::engines::GROUPS;
use super::{audit, Control, Session, SWITCHED_OFF};

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

/// Whether a tool reads, changes, or removes something.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Read,
    Change,
    /// A change that takes something away.
    Remove,
}

pub(crate) struct ToolDef {
    pub name: &'static str,
    pub title: &'static str,
    pub kind: Kind,
    pub description: &'static str,
    pub schema: Value,
}

impl ToolDef {
    pub fn changes(&self) -> bool {
        self.kind != Kind::Read
    }

    /// The tool as `tools/list` shows it.
    pub fn listing(&self) -> Value {
        json!({
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.schema,
            "annotations": {
                "title": self.title,
                "readOnlyHint": self.kind == Kind::Read,
                "destructiveHint": self.kind == Kind::Remove,
                "openWorldHint": false,
            },
        })
    }
}

/// An object schema with these properties, these required, and no others.
fn object(properties: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required, "additionalProperties": false })
}

fn none() -> Value {
    object(json!({}), &[])
}

fn plugin_id() -> Value {
    json!({ "type": "string", "minLength": 1, "description": "The plugin's id, as plugins_list shows it." })
}

fn service_id() -> Value {
    json!({ "type": "string", "minLength": 1, "description": "The service's id, as services_list shows it." })
}

fn flow_id() -> Value {
    json!({ "type": "string", "minLength": 1, "description": "The flow's id, as flows_list shows it." })
}

fn def(name: &'static str, title: &'static str, kind: Kind, description: &'static str, schema: Value) -> ToolDef {
    ToolDef { name, title, kind, description, schema }
}

/// Every tool, in the order `tools/list` gives them.
pub(crate) fn defs() -> &'static [ToolDef] {
    static DEFS: OnceLock<Vec<ToolDef>> = OnceLock::new();
    DEFS.get_or_init(|| {
        use Kind::{Change, Read, Remove};
        // The engine's groups, and the catalog's name for its picture tools.
        let mut groups: Vec<&str> = GROUPS.to_vec();
        groups.push("picture");
        let choosable: Vec<&str> = GROUPS.iter().copied().filter(|g| !matches!(*g, "background" | "upscale")).collect();
        vec![
            // Overview
            def("status", "OAIY at a glance", Read,
                "Overview of OAIY in one call: the engines (running, the model chosen per group, GPUs and memory), services, plugins (state, whether setup is needed), modules, ChatGPT sign-in, the Agent's model, the FormLogic link and sync, setup, and agentMayChange (whether the person lets the Agent change OAIY). Call it first.",
                none()),
            // Engines and models
            def("models_list", "Engine models", Read,
                "The engine's models per group: the model chosen, the models installed (the names model_set_default takes), and the catalog's entries with size, memory need (vramGb), recommended and installed (the catalog files background removal and upscaling under picture). Give group for one group only.",
                object(json!({ "group": { "type": "string", "enum": groups, "description": "One group only." } }), &[])),
            def("model_download_status", "Model downloads", Read,
                "How the engine's model downloads stand: each with its status, percent done, speed and any error.",
                none()),
            def("model_set_default", "Choose a model", Change,
                "Choose the model a group of the engine uses: model is one of the group's installed models from models_list (not a catalog id). The engines refuse a model the group does not have. Restart the language model (engine_restart) for an llm change to take effect now.",
                object(json!({
                    "group": { "type": "string", "enum": choosable, "description": "The engine group." },
                    "model": { "type": "string", "minLength": 1, "description": "An installed model of that group, by the name models_list shows." }
                }), &["group", "model"])),
            def("model_download", "Download a model", Change,
                "Start downloading a model from the engine's catalog into the engines' own folder (catalogId: a catalog entry's id from models_list); it is added to its group when done, and model_download_status follows it. Downloads are gigabytes: check size and memory need first.",
                object(json!({ "catalogId": { "type": "string", "minLength": 1, "description": "The catalog entry's id." } }), &["catalogId"])),
            def("engine_start", "Start the language model", Change,
                "Start the engine's language model (the model chosen for llm). Answers its state; loading takes a while, and status shows when it is ready.",
                none()),
            def("engine_stop", "Stop the language model", Change,
                "Stop the engine's language model, freeing its GPU memory. Calls, texts and flows that need it fail until it starts again.",
                none()),
            def("engine_restart", "Restart the language model", Change,
                "Restart the engine's language model, for example to load a model just chosen with model_set_default.",
                none()),
            // AI sources
            def("ai_sources_list", "AI sources", Read,
                "The AI sources flows and the Agent can use: OAIY's engine, configured providers and local AI services, ChatGPT (signed in or not, and as whom), and the Agent's own model preference.",
                none()),
            def("agent_model_set", "Choose the Agent's model", Change,
                "Choose what the Agent itself runs on: source engine (the model chosen in Engines) or chatgpt (the person's ChatGPT sign-in; model is optional, Codex's own default when left out). Live phone calls keep their own fast route either way.",
                object(json!({
                    "source": { "type": "string", "enum": ["engine", "chatgpt"], "description": "engine or chatgpt." },
                    "model": { "type": "string", "minLength": 1, "description": "chatgpt only: a model from Codex's catalogue." }
                }), &["source"])),
            def("chatgpt_sign_in", "Sign in to ChatGPT", Change,
                "Start signing OAIY in to ChatGPT (its Codex connector). Answers the address for the person to open and sign in at (with deviceCode, a code for them to type instead); ai_sources_list shows when they have.",
                object(json!({ "deviceCode": { "type": "boolean", "description": "Use the device-code sign-in (a code to type) instead of a browser redirect." } }), &[])),
            def("chatgpt_sign_out", "Sign out of ChatGPT", Remove,
                "Sign OAIY out of ChatGPT. Flows and an Agent running on ChatGPT stop working until someone signs in again.",
                none()),
            // Services
            def("services_list", "Services", Read,
                "OAIY's services (local programs such as the voice server and model servers): each with its status (running, stopped, installing, starting, errored), whether it is installed, and its port. Use the ids with the service tools.",
                none()),
            def("service_logs", "A service's log", Read,
                "The last lines a service printed, for why an install or a start failed.",
                object(json!({
                    "id": service_id(),
                    "lines": { "type": "integer", "minimum": 1, "maximum": 500, "description": "How many (default 50)." }
                }), &["id"])),
            def("service_install", "Install a service", Change,
                "Install a service (download and set up its files). It runs in the background: service_logs shows its progress and services_list when it is installed.",
                object(json!({ "id": service_id() }), &["id"])),
            def("service_start", "Start a service", Change,
                "Start an installed service.",
                object(json!({ "id": service_id() }), &["id"])),
            def("service_stop", "Stop a service", Change,
                "Stop a running service.",
                object(json!({ "id": service_id() }), &["id"])),
            def("service_uninstall", "Uninstall a service", Remove,
                "Remove a service's installed files (for a clean reinstall). Its definition stays, so service_install can put it back.",
                object(json!({ "id": service_id() }), &["id"])),
            // Plugins
            def("plugins_list", "Plugins", Read,
                "The installed plugins: id, name, version, state (running, stopped, crashed, disabled…), whether it is switched on, whether its setup is needed (setUpBy: record when finished here, checks when its live checks all pass, as the dashboard judges it), and its connectors' commands (for plugin_command).",
                none()),
            def("plugin_catalog", "Plugins to install", Read,
                "The plugins OAIY knows how to install, each with what it does and needs, whether it is installed, and the folder on this computer it installs from (source; null when the person has to choose the folder). Pass source to plugin_install.",
                none()),
            def("plugin_settings_get", "A plugin's settings", Read,
                "A plugin's settings, read through its own settings command: the values, and the fields its setup declares (key, label, type, options). Secret-looking values are hidden.",
                object(json!({ "pluginId": plugin_id() }), &["pluginId"])),
            def("plugin_setup_status", "A plugin's setup", Read,
                "A plugin's setup steps with their live state: recorded done or skipped, a screen step's done check, a requirements step's services and engine models, and whether a step applies; needsSetup as the dashboard judges it, and outstanding lists what is left. Steps of kind permissions, screen and host are the person's to do on screen (plugin_setup_open).",
                object(json!({ "pluginId": plugin_id() }), &["pluginId"])),
            def("plugin_install", "Install a plugin", Change,
                "Install (or update) a plugin from a folder or .tar.gz on this computer (source: a path, as plugin_catalog gives it; never a URL). A plugin is native code: install only what the person asked for. Then plugin_enable starts it and plugin_setup_status shows its setup.",
                object(json!({ "source": { "type": "string", "minLength": 1, "description": "A folder or .tar.gz path on this computer." } }), &["source"])),
            def("plugin_enable", "Switch a plugin on", Change,
                "Switch a plugin on and start it. Answers an error, saying why, when it does not start.",
                object(json!({ "id": plugin_id() }), &["id"])),
            def("plugin_disable", "Switch a plugin off", Change,
                "Switch a plugin off: it stops, and its commands, modules (such as the phone and the calendar) and events are gone until plugin_enable.",
                object(json!({ "id": plugin_id() }), &["id"])),
            def("plugin_restart", "Restart a plugin", Change,
                "Stop and start a plugin, for settings that apply only when it starts (plugin_settings_set says when).",
                object(json!({ "id": plugin_id() }), &["id"])),
            def("plugin_uninstall", "Uninstall a plugin", Remove,
                "Remove a plugin from this computer, with its data (for the phone plugin, its pairing with the phone).",
                object(json!({ "id": plugin_id() }), &["id"])),
            def("plugin_settings_set", "Change a plugin's settings", Change,
                "Change some of a plugin's settings through its own settings command: settings holds only the keys to change ({key: value}); plugin_settings_get shows the keys and allowed values, and a value it showed as [redacted] is left as it is. Answers whether they apply only after plugin_restart.",
                object(json!({
                    "pluginId": plugin_id(),
                    "settings": { "type": "object", "description": "The settings to change, by key." }
                }), &["pluginId", "settings"])),
            def("plugin_command", "Send a plugin a command", Change,
                "Send one of a plugin's declared connector commands (plugins_list lists them), with an optional payload, through the same gate as the dashboard, and answer what it said (secret-looking values hidden). Commands with effects in the world (a call, a text) really happen.",
                object(json!({
                    "pluginId": plugin_id(),
                    "command": { "type": "string", "minLength": 1, "description": "A command the plugin declares, e.g. phone.status." },
                    "payload": { "description": "The command's payload, when it takes one." }
                }), &["pluginId", "command"])),
            def("plugin_setup_open", "Show a plugin's setup", Change,
                "Show the person a plugin's setup in the OAIY dashboard, at a step if stepId is given: for what they must do on screen, such as accepting what the plugin may do or pairing a device.",
                object(json!({
                    "pluginId": plugin_id(),
                    "stepId": { "type": "string", "minLength": 1, "description": "A step id from plugin_setup_status." }
                }), &["pluginId"])),
            def("plugin_setup_step_done", "Record a setup step done", Change,
                "Record a plugin's setup step as done (not its permissions step, which the person accepts on screen).",
                object(json!({
                    "pluginId": plugin_id(),
                    "stepId": { "type": "string", "minLength": 1, "description": "A step id from plugin_setup_status." }
                }), &["pluginId", "stepId"])),
            def("plugin_setup_finish", "Finish a plugin's setup", Change,
                "Record a plugin's setup as finished once its steps are done, so OAIY stops asking. Refused until the person has accepted what the plugin may do.",
                object(json!({ "pluginId": plugin_id() }), &["pluginId"])),
            // Flows
            def("flows_list", "Flows", Read,
                "The flows stored on this desktop: id and name.",
                none()),
            def("flow_get", "A flow", Read,
                "One flow's document as stored ({name, nodes, edges}). Read one before flow_create or flow_update, to follow its shape.",
                object(json!({ "id": flow_id() }), &["id"])),
            def("flow_create", "Create a flow", Change,
                "Store a new flow: flow is its document ({name, nodes, edges}, as flow_get shows); id (letters, digits, - and _) is made from its name when left out. Refuses an id that exists: use flow_update for that.",
                object(json!({
                    "flow": { "type": "object", "description": "The flow's document." },
                    "id": { "type": "string", "minLength": 1, "description": "The id to store it under." }
                }), &["flow"])),
            def("flow_update", "Change a flow", Change,
                "Replace a stored flow's document with a new one (the whole document, as flow_get shows it).",
                object(json!({
                    "id": flow_id(),
                    "flow": { "type": "object", "description": "The flow's whole new document." }
                }), &["id", "flow"])),
            def("flow_run", "Run a flow", Change,
                "Run a stored flow with an optional input. By default waits for it (up to timeoutSeconds, 60) and answers its status and output; with wait false, answers the run at once.",
                object(json!({
                    "id": flow_id(),
                    "input": { "description": "The flow's input." },
                    "wait": { "type": "boolean", "description": "Wait for the result (default true)." },
                    "timeoutSeconds": { "type": "integer", "minimum": 1, "maximum": 600, "description": "How long the run may take (default 60)." }
                }), &["id"])),
            def("flow_delete", "Delete a flow", Remove,
                "Delete a stored flow.",
                object(json!({ "id": flow_id() }), &["id"])),
            // Calendar
            def("calendar_settings_get", "Calendar settings", Read,
                "The calendar's settings: business (its name), hours (seven days, Monday first, each a list of {open, close} like 09:00), services ({id, name, minutes, description, price}), slotMinutes, noticeMinutes, horizonDays, textConfirmations; available says whether a plugin provides the calendar.",
                none()),
            def("calendar_settings_set", "Change calendar settings", Change,
                "Change the calendar's settings: only the fields given change, in the shapes calendar_settings_get shows (hours: seven days, Monday first). Needs a plugin that provides the calendar.",
                object(json!({
                    "business": { "type": "string", "description": "The business's name, as the receptionist says it." },
                    "hours": { "type": "array", "description": "Seven days, Monday first, each a list of {open, close} (HH:MM); an empty list is closed." },
                    "services": { "type": "array", "description": "The services booked: {id, name, minutes, description, price}." },
                    "slotMinutes": { "type": "integer", "minimum": 1, "description": "The step between the times offered." },
                    "noticeMinutes": { "type": "integer", "minimum": 0, "description": "How soon from now a time may be offered." },
                    "horizonDays": { "type": "integer", "minimum": 1, "description": "How far ahead times are offered." },
                    "textConfirmations": { "type": "boolean", "description": "Text the person when their appointment is confirmed." }
                }), &[])),
            // FormLogic
            def("link_status", "FormLogic link", Read,
                "The FormLogic link (linked or not, the account and address) and how the calendar sync with it stands (state, last success, what is pending, any error).",
                none()),
            def("link_sync_now", "Sync with FormLogic now", Change,
                "Sync the calendar with FormLogic now and answer how it went. Needs the link and a plugin that provides the calendar.",
                none()),
            // Setup
            def("setup_status", "Setup", Read,
                "The first-run setup (finished or not, where it is, what was skipped and which plugins were chosen) and each installed plugin's setup (finished version, steps recorded, whether it needs setup).",
                none()),
            def("setup_finish", "Finish first-run setup", Change,
                "Record OAIY's first-run setup as finished, so the dashboard stops offering it.",
                none()),
            def("ui_open", "Show a dashboard page", Change,
                "Show the person a page of the OAIY dashboard: overview, agent, flows, calendar, engines, services, plugins, runs, models, python, providers, connections, settings, setup, or a plugin's page as plugin:<pluginId>:<navId>.",
                object(json!({ "view": { "type": "string", "minLength": 1, "description": "The page." } }), &["view"])),
            // Diagnostics
            def("logs_tail", "Logs", Read,
                "The end of a log: source desktop (the default, OAIY's own), service:<id>, plugin:<id>, or engines, llm or media (the engine's); lines default 100.",
                object(json!({
                    "source": { "type": "string", "minLength": 1, "description": "desktop, service:<id>, plugin:<id>, engines, llm or media." },
                    "lines": { "type": "integer", "minimum": 1, "maximum": 1000, "description": "How many (default 100)." }
                }), &[])),
        ]
    })
}

pub(crate) fn find(name: &str) -> Option<&'static ToolDef> {
    defs().iter().find(|d| d.name == name)
}

/// Whether `session` is offered `tool`.
pub(crate) fn offered(tool: &ToolDef, session: Session) -> bool {
    if tool.changes() { session.may_change() } else { session.may_read() }
}

/// The tools `session` is offered, as `tools/list` shows them.
pub(crate) fn list(session: Session) -> Vec<Value> {
    defs().iter().filter(|d| offered(d, session)).map(ToolDef::listing).collect()
}

// ---------------------------------------------------------------------------
// Calling a tool
// ---------------------------------------------------------------------------

/// What a tool did: a sentence for the model, and the JSON behind it.
pub(crate) struct Done {
    pub summary: String,
    pub data: Value,
}

fn done(summary: impl Into<String>, data: Value) -> Result<Done, String> {
    Ok(Done { summary: summary.into(), data })
}

/// A read's answer: the JSON only.
fn data(data: Value) -> Result<Done, String> {
    Ok(Done { summary: String::new(), data })
}

fn error_result(message: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": message }], "isError": true, "structuredContent": { "error": message } })
}

fn success_result(d: Done) -> Value {
    let data = match d.data {
        Value::Object(o) => Value::Object(o),
        Value::Null => json!({}),
        other => json!({ "result": other }),
    };
    let body = if data.as_object().is_some_and(|o| o.is_empty()) { String::new() } else { data.to_string() };
    let text = [d.summary.as_str(), body.as_str()].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join("\n");
    json!({ "content": [{ "type": "text", "text": if text.is_empty() { "Done.".to_string() } else { text } }], "structuredContent": data })
}

/// Check `args` against `tool`'s schema: no unknown names, the required ones
/// there, each of its type (and in its list, when it has one).
pub(crate) fn validate(tool: &ToolDef, args: &Map<String, Value>) -> Result<(), String> {
    let props = tool.schema.get("properties").and_then(Value::as_object).cloned().unwrap_or_default();
    let names = || if props.is_empty() { "none".to_string() } else { props.keys().cloned().collect::<Vec<_>>().join(", ") };
    for key in args.keys() {
        if !props.contains_key(key) {
            return Err(format!("{} takes no argument {key:?}: its arguments are {}.", tool.name, names()));
        }
    }
    for req in tool.schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
        if args.get(req).is_none_or(Value::is_null) {
            return Err(format!("{} needs {req:?}: its arguments are {}.", tool.name, names()));
        }
    }
    for (key, value) in args {
        if value.is_null() {
            continue;
        }
        let spec = &props[key];
        let ok = match spec.get("type").and_then(Value::as_str) {
            Some("string") => value.as_str().is_some_and(|s| spec.get("minLength").is_none() || !s.trim().is_empty()),
            Some("integer") => value.as_i64().is_some_and(|n| {
                spec.get("minimum").and_then(Value::as_i64).is_none_or(|m| n >= m) && spec.get("maximum").and_then(Value::as_i64).is_none_or(|m| n <= m)
            }),
            Some("boolean") => value.is_boolean(),
            Some("object") => value.is_object(),
            Some("array") => value.is_array(),
            _ => true,
        };
        if !ok {
            let want = match spec.get("type").and_then(Value::as_str) {
                Some("integer") => {
                    let (lo, hi) = (spec.get("minimum").and_then(Value::as_i64), spec.get("maximum").and_then(Value::as_i64));
                    match (lo, hi) {
                        (Some(lo), Some(hi)) => format!("a whole number from {lo} to {hi}"),
                        (Some(lo), None) => format!("a whole number, at least {lo}"),
                        _ => "a whole number".into(),
                    }
                }
                Some("string") => "a non-empty string".into(),
                Some(t) => format!("a {t}"),
                None => "a value".into(),
            };
            return Err(format!("{}: {key} must be {want}, not {value}.", tool.name));
        }
        if let Some(list) = spec.get("enum").and_then(Value::as_array) {
            if !list.contains(value) {
                let options: Vec<String> = list.iter().filter_map(Value::as_str).map(str::to_string).collect();
                return Err(format!("{}: {key} must be one of {}, not {value}.", tool.name, options.join(", ")));
            }
        }
    }
    Ok(())
}

/// Run tool `name` for `session`: the answer is always a `tools/call` result
/// (a failure is `isError: true`). Change tools are logged whatever happens.
pub(crate) async fn call(control: &Control, session: Session, name: &str, args: Map<String, Value>) -> Value {
    let Some(tool) = find(name) else {
        return error_result(&format!("There is no tool called {name:?}: tools/list names the tools there are."));
    };
    let args_value = Value::Object(args.clone());
    let log = |ok: bool, summary: &str| {
        if tool.changes() {
            control.audit().append(tool.name, &args_value, session.name(), ok, summary);
        }
    };
    if !offered(tool, session) {
        let why = match session {
            Session::Runner => format!("{} changes OAIY, and a runner session may only read it: change OAIY from a project or setup conversation.", tool.name),
            Session::Other => "OAIY's control tools are not offered to this session: the X-OAIY-Session header names project, setup, runner, call, sms or task.".to_string(),
            s => format!("OAIY's control tools are not offered in {} sessions.", s.name()),
        };
        log(false, &why);
        return error_result(&why);
    }
    if tool.changes() && !control.agent_may_change() {
        log(false, SWITCHED_OFF);
        return error_result(SWITCHED_OFF);
    }
    if let Err(why) = validate(tool, &args) {
        log(false, &why);
        return error_result(&why);
    }
    let Some(router) = control.router() else {
        let why = "OAIY is still starting: try again in a moment.";
        log(false, why);
        return error_result(why);
    };
    let desk = Desk { router };
    let outcome = run(&desk, control, tool.name, &Args(&args)).await;
    match outcome {
        Ok(d) => {
            log(true, if d.summary.is_empty() { "done" } else { &d.summary });
            success_result(d)
        }
        Err(why) => {
            log(false, &why);
            error_result(&why)
        }
    }
}

/// Arguments, already checked against the schema.
struct Args<'a>(&'a Map<String, Value>);

impl Args<'_> {
    fn str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
    }
    /// A required string (the schema checked it is there).
    fn req(&self, key: &str) -> &str {
        self.str(key).unwrap_or_default()
    }
    fn bool(&self, key: &str) -> Option<bool> {
        self.0.get(key).and_then(Value::as_bool)
    }
    fn int(&self, key: &str) -> Option<i64> {
        self.0.get(key).and_then(Value::as_i64)
    }
    fn value(&self, key: &str) -> Option<&Value> {
        self.0.get(key).filter(|v| !v.is_null())
    }
}

async fn run(d: &Desk, control: &Control, name: &str, a: &Args<'_>) -> Result<Done, String> {
    match name {
        "status" => status(d).await,
        "models_list" => models_list(d, a.str("group")).await,
        "model_download_status" => model_download_status(d).await,
        "model_set_default" => {
            let (group, model) = (a.req("group"), a.req("model"));
            d.put("/api/engines/defaults", json!({ "group": group, "model": model })).await?;
            let later = if group == "llm" { " It loads when the language model next starts (engine_restart to load it now)." } else { "" };
            done(format!("The {group} group now uses {model}.{later}"), json!({ "group": group, "model": model }))
        }
        "model_download" => {
            let id = a.req("catalogId");
            let v = d.post("/api/engines/downloads", json!({ "id": id })).await?;
            done(format!("Downloading {id}: model_download_status follows it."), json!({ "downloads": compact_downloads(&v) }))
        }
        "engine_start" | "engine_stop" | "engine_restart" => {
            let action = name.trim_start_matches("engine_");
            let v = d.post_empty(&format!("/api/engines/llm/{action}")).await?;
            let state = v.pointer("/llm/state").and_then(Value::as_str).unwrap_or("unknown");
            done(format!("The language model is {state}."), v)
        }
        "ai_sources_list" => ai_sources_list(d).await,
        "agent_model_set" => {
            let source = a.req("source");
            let model = a.str("model");
            if source == "engine" && model.is_some() {
                return Err("model goes with chatgpt only: on the engine the Agent uses the model chosen in Engines (model_set_default for llm).".into());
            }
            let mut pref = json!({ "source": source });
            if let Some(m) = model {
                pref["model"] = json!(m);
            }
            let v = d.put("/api/agent/preferences", json!({ "model": pref })).await?;
            let on = match (source, model) {
                ("chatgpt", Some(m)) => format!("ChatGPT ({m})"),
                ("chatgpt", None) => "ChatGPT (Codex's default model)".to_string(),
                _ => "the engine (the model chosen in Engines)".to_string(),
            };
            done(format!("The Agent now runs on {on}."), v)
        }
        "chatgpt_sign_in" => {
            let v = d.post("/api/ai/codex/login", json!({ "deviceCode": a.bool("deviceCode").unwrap_or(false) })).await?;
            let url = v.get("authUrl").and_then(Value::as_str).or_else(|| v.get("verificationUrl").and_then(Value::as_str));
            let code = v.get("userCode").and_then(Value::as_str);
            let summary = match (url, code) {
                (Some(u), Some(c)) => format!("Ask the person to open {u} and enter the code {c}, then sign in to ChatGPT."),
                (Some(u), None) => format!("Ask the person to open {u} and sign in to ChatGPT."),
                _ => "The sign-in started, but ChatGPT gave no address to open.".to_string(),
            };
            done(summary, json!({ "authUrl": v.get("authUrl"), "verificationUrl": v.get("verificationUrl"), "userCode": v.get("userCode") }))
        }
        "chatgpt_sign_out" => {
            d.post_empty("/api/ai/codex/logout").await?;
            done("OAIY is signed out of ChatGPT.", json!({}))
        }
        "services_list" => services_list(d).await,
        "service_logs" => {
            let id = a.req("id");
            let lines = a.int("lines").unwrap_or(50) as usize;
            service_exists(d, id).await?;
            data(json!({ "id": id, "lines": service_log_lines(d, id, lines).await? }))
        }
        "service_install" => {
            let id = a.req("id");
            service_exists(d, id).await?;
            d.post_empty(&format!("/api/services/{}/install", seg(id))).await?;
            done(format!("Installing {id}: service_logs shows its progress, and services_list shows it installed when it is."), json!({ "id": id }))
        }
        "service_start" | "service_stop" => {
            let id = a.req("id");
            service_exists(d, id).await?;
            let action = name.trim_start_matches("service_");
            d.post_empty(&format!("/api/services/{}/{action}", seg(id))).await?;
            done(format!("{} {id}.", if action == "start" { "Started" } else { "Stopped" }), json!({ "id": id }))
        }
        "service_uninstall" => {
            let id = a.req("id");
            service_exists(d, id).await?;
            let v = d.post_empty(&format!("/api/services/{}/uninstall", seg(id))).await?;
            done(format!("Removed {id}'s installed files: service_install puts them back."), json!({ "id": id, "removed": v.get("removed") }))
        }
        "plugins_list" => plugins_list(d).await,
        "plugin_catalog" => plugin_catalog(d).await,
        "plugin_settings_get" => plugin_settings_get(d, a.req("pluginId")).await,
        "plugin_setup_status" => plugin_setup_status(d, a.req("pluginId")).await,
        "plugin_install" => {
            let v = d.post("/api/plugins/install", json!({ "source": a.req("source") })).await?;
            let (id, name, version) = (s(&v, "id"), s(&v, "name"), s(&v, "version"));
            let setup = if v.get("setup").is_some_and(|x| !x.is_null()) { " It has a setup: plugin_setup_status shows its steps." } else { "" };
            done(format!("Installed {name} {version} ({id}); plugin_enable starts it.{setup}"), json!({ "id": id, "name": name, "version": version, "replaced": v.get("replaced"), "setup": v.get("setup") }))
        }
        "plugin_enable" => {
            let id = a.req("id");
            plugin_record(d, id).await?;
            d.post(&format!("/api/plugins/{}/enabled", seg(id)), json!({ "enabled": true })).await?;
            if let Err(why) = d.post_empty(&format!("/api/plugins/{}/start", seg(id))).await {
                return Err(format!("{id} is switched on, but it did not start: {why} (logs_tail with source plugin:{id} may say more)."));
            }
            let state = plugin_record(d, id).await.ok().and_then(|r| r.get("state").and_then(Value::as_str).map(str::to_string)).unwrap_or_else(|| "starting".into());
            done(format!("{id} is switched on and {state}."), json!({ "id": id, "state": state }))
        }
        "plugin_disable" => {
            let id = a.req("id");
            plugin_record(d, id).await?;
            let v = d.post(&format!("/api/plugins/{}/enabled", seg(id)), json!({ "enabled": false })).await?;
            done(format!("{id} is switched off: plugin_enable switches it back on."), json!({ "id": id, "state": v.get("state") }))
        }
        "plugin_restart" => {
            let id = a.req("id");
            plugin_record(d, id).await?;
            d.post_empty(&format!("/api/plugins/{}/stop", seg(id))).await?;
            d.post_empty(&format!("/api/plugins/{}/start", seg(id))).await.map_err(|why| format!("{id} stopped, but did not start again: {why}"))?;
            done(format!("{id} restarted."), json!({ "id": id }))
        }
        "plugin_uninstall" => {
            let id = a.req("id");
            plugin_record(d, id).await?;
            d.delete(&format!("/api/plugins/{}", seg(id))).await?;
            done(format!("Removed {id} and its data from this computer."), json!({ "id": id }))
        }
        "plugin_settings_set" => plugin_settings_set(d, a.req("pluginId"), a.value("settings").cloned().unwrap_or_else(|| json!({}))).await,
        "plugin_command" => {
            let (id, command) = (a.req("pluginId"), a.req("command"));
            let record = plugin_record(d, id).await?;
            let connector = connector_for(&record, command).ok_or_else(|| {
                format!("{} has no command {command:?}: its commands are {}.", plugin_name(&record), commands_of(&record).join(", "))
            })?;
            let result = connector_call(d, &connector, command, a.value("payload").cloned()).await?;
            // Secret-looking values stay hidden here too, as plugin_settings_get hides them.
            done(format!("{command} answered."), json!({ "result": trim(&audit::redact(&result)) }))
        }
        "plugin_setup_open" => {
            let id = a.req("pluginId");
            let step = a.str("stepId");
            let record = plugin_record(d, id).await?;
            // The permissions step is always there: the host shows it first, declared or not.
            if let Some(step) = step.filter(|s| *s != crate::setup::PERMISSIONS) {
                let detail = d.get(&format!("/api/setup/plugins/{}", seg(id))).await?;
                let mut steps = step_ids(&detail);
                if !steps.iter().any(|s| s == crate::setup::PERMISSIONS) {
                    steps.insert(0, crate::setup::PERMISSIONS.to_string());
                }
                if !steps.iter().any(|s| s == step) {
                    return Err(format!("{} has no setup step {step:?}: its steps are {}.", plugin_name(&record), steps.join(", ")));
                }
            }
            let mut payload = json!({ "view": "setup", "pluginId": id });
            if let Some(step) = step {
                payload["stepId"] = json!(step);
            }
            control.navigate(payload.clone())?;
            done(format!("The dashboard shows {}'s setup{}: the person does it there.", plugin_name(&record), step.map(|s| format!(" at the {s} step")).unwrap_or_default()), payload)
        }
        "plugin_setup_step_done" => {
            let (id, step) = (a.req("pluginId"), a.req("stepId"));
            if step == crate::setup::PERMISSIONS {
                return Err(format!("The person accepts what a plugin may do themselves: plugin_setup_open with pluginId {id} and stepId permissions shows it to them."));
            }
            d.post(&format!("/api/setup/plugins/{}/steps/{}", seg(id), seg(step)), json!({ "status": "done" })).await?;
            done(format!("{id}'s {step} step is recorded as done."), json!({ "pluginId": id, "stepId": step }))
        }
        "plugin_setup_finish" => {
            let id = a.req("pluginId");
            let v = d.post_empty(&format!("/api/setup/plugins/{}/finish", seg(id))).await?;
            let version = v.pointer(&format!("/plugins/{id}/version")).cloned().unwrap_or(Value::Null);
            done(format!("{id}'s setup is recorded as finished."), json!({ "pluginId": id, "version": version }))
        }
        "flows_list" => {
            let v = d.get("/api/bridge/flows").await?;
            let flows: Vec<Value> = arr(&v, "flows").iter().map(|f| json!({ "id": f.get("flowId"), "name": f.get("name") })).collect();
            data(json!({ "flows": flows }))
        }
        "flow_get" => {
            let id = a.req("id");
            let flow = d.get(&format!("/api/bridge/flows/{}", seg(id))).await?;
            let size = flow.to_string().len();
            if size > MAX_FLOW_BYTES {
                return Err(format!("{id} is {size} bytes, too large to read here: open it in the Flows page."));
            }
            data(json!({ "id": id, "flow": flow }))
        }
        "flow_create" => flow_create(d, a.str("id"), a.value("flow").cloned().unwrap_or_default()).await,
        "flow_update" => {
            let id = a.req("id");
            flow_ids(d).await?.iter().find(|f| *f == id).ok_or_else(|| format!("There is no flow {id:?}: flow_create stores a new one."))?;
            d.put_json_body(&format!("/api/bridge/flows/{}", seg(id)), a.value("flow").cloned().unwrap_or_default()).await?;
            done(format!("Saved flow {id}."), json!({ "id": id }))
        }
        "flow_run" => flow_run(d, a.req("id"), a.value("input").cloned(), a.bool("wait").unwrap_or(true), a.int("timeoutSeconds").unwrap_or(60) as u64).await,
        "flow_delete" => {
            let id = a.req("id");
            d.delete(&format!("/api/bridge/flows/{}", seg(id))).await?;
            done(format!("Deleted flow {id}."), json!({ "id": id }))
        }
        "calendar_settings_get" => {
            let v = d.get("/api/calendar").await?;
            data(json!({ "available": v.get("available"), "settings": v.get("settings") }))
        }
        "calendar_settings_set" => calendar_settings_set(d, a).await,
        "link_status" => link_status(d).await,
        "link_sync_now" => {
            let report = d.post_empty("/api/calendar/sync").await?;
            let state = report.get("state").and_then(Value::as_str).unwrap_or("unknown");
            let summary = match report.get("error").and_then(Value::as_str) {
                Some(e) => format!("The sync is {state}: {e}"),
                None => format!("The sync is {state}."),
            };
            done(summary, sync_summary(&report))
        }
        "setup_status" => setup_status(d).await,
        "setup_finish" => {
            let state = d.get("/api/setup").await?;
            // Already finished (the dashboard's hand-off to the Agent records it): nothing to change.
            if state.pointer("/firstRun/finished") == Some(&Value::Bool(true)) {
                return done("OAIY's first-run setup was already recorded as finished.", json!({ "finished": true }));
            }
            let mut first_run = state.get("firstRun").cloned().unwrap_or_else(|| json!({}));
            first_run["finished"] = json!(true);
            if let Some(o) = first_run.as_object_mut() {
                // The desktop's own note; it keeps it itself.
                o.remove("migrated");
            }
            d.put("/api/setup", json!({ "firstRun": first_run })).await?;
            done("OAIY's first-run setup is recorded as finished.", json!({ "finished": true }))
        }
        "ui_open" => {
            let view = a.req("view");
            if !view_ok(view) {
                return Err(format!("{view:?} is not a page: one of {}, or plugin:<pluginId>:<navId>.", VIEWS.join(", ")));
            }
            control.navigate(json!({ "view": view }))?;
            done(format!("The dashboard shows {view}."), json!({ "view": view }))
        }
        "logs_tail" => logs_tail(d, a.str("source").unwrap_or("desktop"), a.int("lines").unwrap_or(100) as usize).await,
        other => Err(format!("{other} has no handler: this is a bug in OAIY.")),
    }
}

// ---------------------------------------------------------------------------
// The desktop's routes, in-process
// ---------------------------------------------------------------------------

/// The largest answer a tool reads from a route.
const MAX_BODY: usize = 16 * 1024 * 1024;
/// How long a route may take, unless a tool says otherwise.
const WAIT: Duration = Duration::from_secs(60);
/// How long each part of `status` may take.
const QUICK: Duration = Duration::from_secs(8);
/// A flow document larger than this is not read back whole.
const MAX_FLOW_BYTES: usize = 200 * 1024;

/// The desktop's own router, called in-process with this process's bearer:
/// the same gate, validation and module checks as any other caller.
pub(crate) struct Desk {
    pub router: axum::Router,
}

pub(crate) struct Reply {
    pub status: StatusCode,
    /// The JSON answered; `Null` for an empty body, a string for one that is not JSON.
    pub body: Value,
}

impl Desk {
    pub async fn send(&self, method: Method, path: &str, body: Option<String>, wait: Duration) -> Result<Reply, String> {
        let mut request = Request::builder()
            .method(method.clone())
            .uri(path)
            .header(header::AUTHORIZATION, format!("Bearer {}", crate::internal_token()));
        let body = match body {
            Some(text) => {
                request = request.header(header::CONTENT_TYPE, "application/json");
                Body::from(text)
            }
            None => Body::empty(),
        };
        let request = request.body(body).map_err(|e| format!("the request for {method} {path} could not be made: {e}"))?;
        let response = match tokio::time::timeout(wait, self.router.clone().oneshot(request)).await {
            Ok(Ok(r)) => r,
            Ok(Err(never)) => match never {},
            Err(_) => return Err(format!("OAIY did not answer {method} {} within {} s.", bare(path), wait.as_secs())),
        };
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), MAX_BODY)
            .await
            .map_err(|e| format!("OAIY's answer to {method} {} could not be read: {e}", bare(path)))?;
        let body = if bytes.iter().all(u8::is_ascii_whitespace) {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).chars().take(2000).collect()))
        };
        Ok(Reply { status, body })
    }

    async fn expect(&self, method: Method, path: &str, body: Option<String>, wait: Duration) -> Result<Value, String> {
        let reply = self.send(method.clone(), path, body, wait).await?;
        if reply.status.is_success() {
            Ok(if reply.body.is_null() { json!({ "ok": true }) } else { reply.body })
        } else {
            Err(explain(&method, path, &reply))
        }
    }

    pub async fn get(&self, path: &str) -> Result<Value, String> {
        self.expect(Method::GET, path, None, WAIT).await
    }

    async fn get_within(&self, path: &str, wait: Duration) -> Result<Value, String> {
        self.expect(Method::GET, path, None, wait).await
    }

    pub async fn post(&self, path: &str, body: Value) -> Result<Value, String> {
        self.expect(Method::POST, path, Some(body.to_string()), WAIT).await
    }

    async fn post_within(&self, path: &str, body: Value, wait: Duration) -> Result<Value, String> {
        self.expect(Method::POST, path, Some(body.to_string()), wait).await
    }

    async fn post_empty(&self, path: &str) -> Result<Value, String> {
        self.expect(Method::POST, path, None, WAIT).await
    }

    async fn put(&self, path: &str, body: Value) -> Result<Value, String> {
        self.expect(Method::PUT, path, Some(body.to_string()), WAIT).await
    }

    /// A PUT whose body is a document stored as it is (a flow).
    async fn put_json_body(&self, path: &str, document: Value) -> Result<Value, String> {
        self.expect(Method::PUT, path, Some(document.to_string()), WAIT).await
    }

    async fn delete(&self, path: &str) -> Result<Value, String> {
        self.expect(Method::DELETE, path, None, WAIT).await
    }
}

/// `path` without its query.
fn bare(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

/// The error a route answered, in words: `{error: "…"}`, `{error: {code, message}}`, or plain text.
fn error_text(body: &Value) -> Option<String> {
    match body.get("error") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(e @ Value::Object(_)) => e.get("message").and_then(Value::as_str).map(str::to_string),
        _ => body
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| body.as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
    }
}

/// Why a route refused, for the model: the route's own words, or what its status means.
pub(crate) fn explain(method: &Method, path: &str, reply: &Reply) -> String {
    let what = format!("{method} {}", bare(path));
    match (reply.status.as_u16(), error_text(&reply.body)) {
        (404 | 405, None) => format!("This OAIY has no {what} (it answered {}): it is older than this tool, so OAIY needs updating first.", reply.status.as_u16()),
        (401 | 403, message) => format!("OAIY refused its own call to {what}{}.", message.map(|m| format!(": {m}")).unwrap_or_default()),
        (_, Some(message)) => message,
        (status, None) => format!("{what} answered {status}."),
    }
}

/// A path segment, percent-encoded.
fn seg(s: &str) -> String {
    const KEEP: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC.remove(b'-').remove(b'_').remove(b'.').remove(b'~');
    percent_encoding::utf8_percent_encode(s, KEEP).to_string()
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// A route's answer, or `{error}` where it failed: a part of an overview that does not sink the rest.
fn part(result: Result<Value, String>, f: impl FnOnce(Value) -> Value) -> Value {
    match result {
        Ok(v) => f(v),
        Err(e) => json!({ "error": e }),
    }
}

/// A plugin's answer kept to a readable size.
fn trim(v: &Value) -> Value {
    fn go(v: &Value, depth: usize) -> Value {
        match v {
            Value::String(s) if s.chars().count() > 2000 => Value::String(format!("{}… ({} characters)", s.chars().take(2000).collect::<String>(), s.chars().count())),
            Value::Array(a) if depth > 8 => Value::String(format!("[… {} items]", a.len())),
            Value::Object(o) if depth > 8 => Value::String(format!("{{… {} fields}}", o.len())),
            Value::Array(a) => {
                let mut out: Vec<Value> = a.iter().take(50).map(|x| go(x, depth + 1)).collect();
                if a.len() > 50 {
                    out.push(Value::String(format!("… {} more", a.len() - 50)));
                }
                Value::Array(out)
            }
            Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), go(x, depth + 1))).collect()),
            other => other.clone(),
        }
    }
    go(v, 0)
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

fn gb(mb: Option<f64>) -> Value {
    mb.map_or(Value::Null, |mb| json!((mb / 1024.0 * 10.0).round() / 10.0))
}

fn gpus(list: Option<&Value>) -> Value {
    let Some(list) = list.and_then(Value::as_array) else { return json!([]) };
    Value::Array(
        list.iter()
            .map(|g| {
                let total = g.get("memory_total_mb").and_then(Value::as_f64);
                let used = g.get("memory_used_mb").and_then(Value::as_f64);
                let free = g.get("memory_free_mb").and_then(Value::as_f64).or(total.zip(used).map(|(t, u)| t - u));
                json!({ "name": g.get("name"), "totalGb": gb(total), "freeGb": gb(free) })
            })
            .collect(),
    )
}

fn engines_summary(state: Value, chosen: Result<Value, String>) -> Value {
    if state.get("running") != Some(&Value::Bool(true)) {
        return json!({ "running": false, "note": super::engines::NOT_RUNNING });
    }
    json!({
        "running": true,
        "llm": {
            "state": state.pointer("/llm/state"),
            "resident": state.pointer("/llm/resident"),
            "models": state.pointer("/llm/models"),
        },
        "chosen": part(chosen, |c| c.get("defaults").cloned().unwrap_or_else(|| json!({ "error": c.get("error") }))),
        "gpus": gpus(state.get("gpus")),
    })
}

fn service_rows(v: &Value) -> Vec<Value> {
    arr(v, "services")
        .iter()
        .map(|s| {
            let mut row = json!({ "id": s.get("id"), "status": s.get("status"), "installed": s.get("installed") });
            if let Some(e) = s.get("error").filter(|e| !e.is_null()) {
                row["error"] = e.clone();
            }
            row
        })
        .collect()
}

/// The setup version a plugin declares, and the one last finished (0: never).
fn setup_versions(p: &Value, setup: Option<&Value>) -> (Option<u64>, u64) {
    let id = p.get("id").and_then(Value::as_str).unwrap_or_default();
    let declared = p.pointer("/manifest/setup/version").and_then(Value::as_u64);
    let finished = setup.and_then(|s| s.pointer(&format!("/plugins/{id}/version"))).and_then(Value::as_u64).unwrap_or(0);
    (declared, finished)
}

fn plugin_name(p: &Value) -> String {
    p.pointer("/manifest/name").and_then(Value::as_str).or_else(|| p.get("id").and_then(Value::as_str)).unwrap_or_default().to_string()
}

/// Whether a plugin needs its setup run, by the dashboard's rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Judgement {
    pub declared: bool,
    pub needs: bool,
    /// What says it is set up: `record` (its setup version was finished here) or `checks` (live).
    pub by: Option<&'static str>,
}

/// Set up by what is true now, as the dashboard judges it: every step with a
/// `done` check that shows (its `when`, if it has one, passes) passes it, and
/// there is at least one. `None` when no step has a `done` check. Each item
/// is a step's `when` outcome (none: always shown) and its `done` outcome
/// (none: no `done` check).
pub(crate) fn set_up_by_checks(steps: &[(Option<bool>, Option<bool>)]) -> Option<bool> {
    let mut counted = 0;
    let mut any = false;
    for (when, done) in steps {
        let Some(done) = done else { continue };
        any = true;
        if *when == Some(false) {
            continue;
        }
        if !done {
            return Some(false);
        }
        counted += 1;
    }
    any.then_some(counted > 0)
}

/// The steps of a plugin's declared setup that have a `done` check.
fn done_checked(p: &Value) -> Vec<Value> {
    p.pointer("/manifest/setup/steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|s| s.get("done").is_some_and(|d| !d.is_null()))
        .cloned()
        .collect()
}

/// Each plugin's setup judged as the dashboard does: finished here by its
/// setup version, or else by its live `done` checks. The checks run only for
/// the plugins that are switched on, declare some, and are not finished by the
/// record, so an overview stays cheap.
async fn judge_plugins(d: &Desk, plugins: &[Value], setup: Option<&Value>) -> std::collections::HashMap<String, Judgement> {
    let mut out = std::collections::HashMap::new();
    let mut to_check = Vec::new();
    for p in plugins {
        let id = s(p, "id");
        let (declared, finished) = setup_versions(p, setup);
        let judgement = match declared {
            None => Judgement { declared: false, needs: false, by: None },
            Some(v) if v <= finished => Judgement { declared: true, needs: false, by: Some("record") },
            Some(_) => {
                let steps = done_checked(p);
                if !steps.is_empty() && !p.get("userDisabled").and_then(Value::as_bool).unwrap_or(false) {
                    to_check.push((id.clone(), steps));
                }
                Judgement { declared: true, needs: true, by: None }
            }
        };
        out.insert(id, judgement);
    }
    let judged = futures_util::future::join_all(to_check.into_iter().map(|(id, steps)| async move {
        let outcomes = futures_util::future::join_all(steps.iter().map(|st| {
            let id = id.clone();
            async move {
                let step = s(st, "id");
                let when = if st.get("when").is_some_and(|w| !w.is_null()) { Some(passed(&setup_check(d, &id, &step, true).await)) } else { None };
                (when, Some(passed(&setup_check(d, &id, &step, false).await)))
            }
        }))
        .await;
        (id, set_up_by_checks(&outcomes))
    }))
    .await;
    for (id, live) in judged {
        if live == Some(true) {
            out.insert(id, Judgement { declared: true, needs: false, by: Some("checks") });
        }
    }
    out
}

fn passed(outcome: &Value) -> bool {
    outcome.get("passed") == Some(&Value::Bool(true))
}

fn plugin_row(p: &Value, judgement: Option<&Judgement>, with_commands: bool) -> Value {
    let mut row = json!({
        "id": p.get("id"),
        "name": plugin_name(p),
        "version": p.pointer("/manifest/version"),
        "state": p.get("state"),
        "enabled": !p.get("userDisabled").and_then(Value::as_bool).unwrap_or(false),
        "needsSetup": judgement.map(|j| j.needs),
    });
    if let Some(by) = judgement.and_then(|j| j.by) {
        row["setUpBy"] = json!(by);
    }
    if let Some(r) = p.get("reason").filter(|r| !r.is_null() && p.get("state").and_then(Value::as_str) != Some("running")) {
        row["reason"] = r.clone();
    }
    if with_commands {
        let connectors: Vec<Value> = p
            .pointer("/manifest/connectors")
            .and_then(Value::as_array)
            .map(|cs| cs.iter().map(|c| json!({ "id": c.get("id"), "commands": c.get("commands") })).collect())
            .unwrap_or_default();
        row["connectors"] = Value::Array(connectors);
    }
    row
}

async fn status(d: &Desk) -> Result<Done, String> {
    let (settings, engines, chosen, services, plugins, setup, modules, chatgpt, preference, link, sync) = tokio::join!(
        d.get_within("/api/control/settings", QUICK),
        d.get_within("/api/engines", QUICK),
        d.get_within("/api/engines/defaults", QUICK),
        d.get_within("/api/services", QUICK),
        d.get_within("/api/plugins", QUICK),
        d.get_within("/api/setup", QUICK),
        d.get_within("/api/modules", QUICK),
        d.get_within("/api/ai/codex/status", QUICK),
        d.get_within("/api/agent/preferences", QUICK),
        d.get_within("/api/link", QUICK),
        d.get_within("/api/calendar/sync", QUICK),
    );
    let judged = match &plugins {
        Ok(p) => judge_plugins(d, arr(p, "plugins"), setup.as_ref().ok()).await,
        Err(_) => Default::default(),
    };
    let plugins = part(plugins, |p| Value::Array(arr(&p, "plugins").iter().map(|r| plugin_row(r, judged.get(&s(r, "id")), false)).collect()));
    // As the dashboard nudges: the plugins switched on whose setup is not done.
    let needing: Vec<Value> = plugins
        .as_array()
        .map(|list| {
            list.iter()
                .filter(|p| p.get("needsSetup") == Some(&Value::Bool(true)) && p.get("enabled") == Some(&Value::Bool(true)))
                .filter_map(|p| p.get("id").cloned())
                .collect()
        })
        .unwrap_or_default();
    let setup = part(setup, |s| json!({ "firstRunFinished": s.pointer("/firstRun/finished"), "pluginsNeedingSetup": needing }));
    data(json!({
        "agentMayChange": part(settings, |s| s.get("agentMayChange").cloned().unwrap_or(Value::Null)),
        "engines": part(engines, |e| engines_summary(e, chosen)),
        "services": part(services, |v| Value::Array(service_rows(&v))),
        "plugins": plugins,
        "modules": part(modules, |m| {
            Value::Array(arr(&m, "modules").iter().map(|x| {
                let mut row = json!({ "id": x.get("id"), "enabled": x.get("enabled"), "provider": x.pointer("/provider/pluginId") });
                if let Some(r) = x.get("reason").filter(|r| !r.is_null()) {
                    row["reason"] = r.clone();
                }
                row
            }).collect())
        }),
        "chatgpt": part(chatgpt, |c| chatgpt_summary(&c)),
        "agentModel": part(preference, |p| p.get("model").cloned().unwrap_or(p)),
        "link": part(link, |l| link_summary(&l, sync)),
        "setup": setup,
    }))
}

fn chatgpt_summary(c: &Value) -> Value {
    let mut out = json!({ "available": c.get("available"), "signedIn": c.get("connected") });
    if let Some(e) = c.get("email").filter(|e| !e.is_null()) {
        out["account"] = e.clone();
    }
    if let Some(d) = c.get("detail").filter(|d| !d.is_null()) {
        out["detail"] = d.clone();
    }
    out
}

fn sync_summary(r: &Value) -> Value {
    json!({
        "state": r.get("state"),
        "lastSuccessAt": r.get("lastSuccessAt"),
        "pulled": r.get("pulled"),
        "pushed": r.get("pushed"),
        "removed": r.get("removed"),
        "pending": r.get("pending"),
        "error": r.get("error"),
    })
}

fn link_summary(l: &Value, sync: Result<Value, String>) -> Value {
    json!({
        "linked": l.get("linked"),
        "provider": l.get("connectorName").or_else(|| l.get("connectorId")),
        "address": l.get("baseUrl"),
        "account": l.get("accountName"),
        "linkedAt": l.get("linkedAt"),
        "problem": l.get("heartbeatError").or_else(|| l.get("relayError")),
        "calendarSync": part(sync, |r| sync_summary(&r)),
    })
}

async fn link_status(d: &Desk) -> Result<Done, String> {
    let (link, sync) = tokio::join!(d.get("/api/link"), d.get("/api/calendar/sync"));
    data(link_summary(&link?, sync))
}

// ---------------------------------------------------------------------------
// Engines and models
// ---------------------------------------------------------------------------

fn percent(d: &Value) -> Value {
    match (d.get("done").and_then(Value::as_f64), d.get("total").and_then(Value::as_f64)) {
        (Some(done), Some(total)) if total > 0.0 => json!((done / total * 100.0).round()),
        _ => Value::Null,
    }
}

fn compact_download(d: &Value) -> Value {
    json!({ "id": d.get("id"), "status": d.get("status"), "percent": percent(d), "file": d.get("file"), "speed": d.get("speed"), "error": d.get("error") })
}

fn compact_downloads(v: &Value) -> Vec<Value> {
    arr(v, "downloads").iter().map(compact_download).collect()
}

async fn model_download_status(d: &Desk) -> Result<Done, String> {
    let v = d.get("/api/engines/downloads").await?;
    if v.get("running") != Some(&Value::Bool(true)) {
        return data(json!({ "running": false, "note": v.get("error") }));
    }
    data(json!({ "running": true, "downloads": compact_downloads(&v) }))
}

async fn models_list(d: &Desk, only: Option<&str>) -> Result<Done, String> {
    let (catalog, chosen) = tokio::join!(d.get("/api/engines/catalog"), d.get("/api/engines/defaults"));
    let catalog = catalog?;
    if catalog.get("running") != Some(&Value::Bool(true)) {
        return data(json!({ "running": false, "note": catalog.get("error").cloned().unwrap_or_else(|| json!(super::engines::NOT_RUNNING)) }));
    }
    let chosen = chosen.unwrap_or_else(|e| json!({ "error": e }));
    let names: Map<String, Value> = arr(&catalog, "groups")
        .iter()
        .filter_map(|g| Some((g.get("id")?.as_str()?.to_string(), g.get("name").cloned().unwrap_or(Value::Null))))
        .collect();
    // The engine's groups, and any the catalog files models under besides (its `picture`:
    // background removal and upscaling, which the engine serves as background and upscale).
    let mut all: Vec<&str> = GROUPS.to_vec();
    for g in names.keys() {
        if !all.contains(&g.as_str()) {
            all.push(g.as_str());
        }
    }
    let groups: Vec<Value> = all
        .into_iter()
        .filter(|g| only.is_none_or(|o| o == *g))
        .map(|group| {
            let entries: Vec<Value> = arr(&catalog, "models")
                .iter()
                .filter(|m| m.get("group").and_then(Value::as_str) == Some(group))
                .map(|m| {
                    let mut e = json!({
                        "id": m.get("id"), "name": m.get("name"), "sizeGb": m.get("sizeGb"), "vramGb": m.get("vramGb"),
                        "recommended": m.get("recommended"), "installed": m.get("installed"),
                    });
                    if let Some(dl) = m.get("download").filter(|x| !x.is_null()) {
                        e["download"] = json!({ "status": dl.get("status"), "percent": percent(dl) });
                    }
                    if let Some(n) = m.get("needs").and_then(Value::as_array).filter(|n| !n.is_empty()) {
                        e["needs"] = Value::Array(n.clone());
                    }
                    e
                })
                .collect();
            json!({
                "group": group,
                "name": names.get(group),
                "chosen": chosen.pointer(&format!("/defaults/{group}")).cloned().or_else(|| catalog.pointer(&format!("/defaults/{group}")).cloned()),
                "installed": chosen.pointer(&format!("/models/{group}")),
                "catalog": entries,
            })
        })
        .collect();
    data(json!({ "running": true, "groups": groups }))
}

// ---------------------------------------------------------------------------
// AI sources, services
// ---------------------------------------------------------------------------

async fn ai_sources_list(d: &Desk) -> Result<Done, String> {
    let (sources, chatgpt, preference) = tokio::join!(d.get("/api/ai/sources"), d.get("/api/ai/codex/status"), d.get_within("/api/agent/preferences", QUICK));
    let sources = sources?;
    let list = sources.as_array().map(Vec::as_slice).unwrap_or_else(|| arr(&sources, "sources"));
    let rows: Vec<Value> = list
        .iter()
        .map(|x| {
            json!({
                "id": x.get("id"), "kind": x.get("kind"), "name": x.get("name"), "status": x.get("status"),
                "model": x.get("model"), "capabilities": x.get("capabilities"),
            })
        })
        .collect();
    data(json!({
        "sources": rows,
        "chatgpt": part(chatgpt, |c| chatgpt_summary(&c)),
        "agentModel": part(preference, |p| p.get("model").cloned().unwrap_or(p)),
    }))
}

async fn services_list(d: &Desk) -> Result<Done, String> {
    let v = d.get("/api/services").await?;
    let rows: Vec<Value> = arr(&v, "services")
        .iter()
        .map(|x| {
            let mut row = json!({
                "id": x.get("id"), "name": x.get("name"), "category": x.get("category"), "status": x.get("status"),
                "installed": x.get("installed"), "installable": x.get("installable"), "port": x.get("port"), "autostart": x.get("autostart"),
            });
            for key in ["error", "needsRepair"] {
                if let Some(e) = x.get(key).filter(|e| !e.is_null() && **e != Value::Bool(false)) {
                    row[key] = e.clone();
                }
            }
            row
        })
        .collect();
    let broken = arr(&v, "templateErrors").len();
    let mut out = json!({ "services": rows });
    if broken > 0 {
        out["definitionsThatFailedToLoad"] = json!(broken);
    }
    data(out)
}

async fn service_exists(d: &Desk, id: &str) -> Result<(), String> {
    let v = d.get("/api/services").await?;
    let ids: Vec<&str> = arr(&v, "services").iter().filter_map(|x| x.get("id").and_then(Value::as_str)).collect();
    if ids.contains(&id) {
        Ok(())
    } else {
        Err(format!("There is no service {id:?}: services_list shows them ({}).", ids.join(", ")))
    }
}

/// A log line as text, whichever shape it came in.
fn line_text(l: &Value) -> String {
    match l {
        Value::String(s) => s.clone(),
        Value::Object(o) => {
            let text = o.get("text").or_else(|| o.get("line")).or_else(|| o.get("message")).and_then(Value::as_str).unwrap_or_default();
            let at = o.get("timestamp").or_else(|| o.get("at")).and_then(Value::as_str).map(|t| t.get(11..19).unwrap_or(t).to_string());
            let stream = o.get("stream").or_else(|| o.get("level")).and_then(Value::as_str).filter(|s| *s != "stdout");
            [at, stream.map(str::to_string), Some(text.to_string())].into_iter().flatten().collect::<Vec<_>>().join(" ")
        }
        other => other.to_string(),
    }
}

async fn service_log_lines(d: &Desk, id: &str, lines: usize) -> Result<Vec<String>, String> {
    let v = d.get(&format!("/api/services/{}/logs?tail={lines}", seg(id))).await?;
    let list = v.as_array().map(Vec::as_slice).unwrap_or_else(|| arr(&v, "lines"));
    Ok(list.iter().map(line_text).collect())
}

async fn logs_tail(d: &Desk, source: &str, lines: usize) -> Result<Done, String> {
    if source == "desktop" {
        let v = d.get(&format!("/api/control/desktop-log?lines={lines}")).await?;
        if v.get("available") == Some(&Value::Bool(false)) {
            return done(s(&v, "reason"), json!({ "source": source, "lines": [] }));
        }
        return data(json!({ "source": source, "lines": v.get("lines") }));
    }
    if let Some(id) = source.strip_prefix("service:") {
        service_exists(d, id).await?;
        return data(json!({ "source": source, "lines": service_log_lines(d, id, lines).await? }));
    }
    if let Some(id) = source.strip_prefix("plugin:") {
        plugin_record(d, id).await?;
        let v = d.get(&format!("/api/plugins/{}/logs?tail={lines}", seg(id))).await?;
        let all: Vec<String> = arr(&v, "lines").iter().map(line_text).collect();
        let tail = all[all.len().saturating_sub(lines)..].to_vec();
        return data(json!({ "source": source, "lines": tail }));
    }
    if matches!(source, "engines" | "studio" | "llm" | "media") {
        let v = d.get(&format!("/api/engines/logs?source={source}&lines={lines}")).await?;
        return data(json!({ "source": source, "lines": v.get("lines") }));
    }
    Err(format!("{source:?} is not a log: desktop, service:<id>, plugin:<id>, engines, llm or media."))
}

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------

async fn plugins_list(d: &Desk) -> Result<Done, String> {
    let (plugins, setup) = tokio::join!(d.get("/api/plugins"), d.get("/api/setup"));
    let plugins = plugins?;
    let setup = setup.ok();
    let judged = judge_plugins(d, arr(&plugins, "plugins"), setup.as_ref()).await;
    let rows: Vec<Value> = arr(&plugins, "plugins").iter().map(|p| plugin_row(p, judged.get(&s(p, "id")), true)).collect();
    data(json!({ "plugins": rows }))
}

async fn plugin_catalog(d: &Desk) -> Result<Done, String> {
    let v = d.get("/api/setup/catalog").await?;
    let rows: Vec<Value> = arr(&v, "plugins")
        .iter()
        .map(|p| {
            json!({
                "id": p.get("id"), "name": p.get("name"), "plugin": p.get("plugin"), "publisher": p.get("publisher"),
                "description": p.get("description"), "provides": p.get("provides"), "needs": p.get("needs"),
                "installed": p.get("installed"), "installedVersion": p.get("installedVersion"),
                "source": p.pointer("/source/path"), "sourceNote": p.pointer("/source/note"),
            })
        })
        .collect();
    data(json!({ "plugins": rows }))
}

/// The installed plugin `id`'s record, as `/api/plugins` answers it.
async fn plugin_record(d: &Desk, id: &str) -> Result<Value, String> {
    let v = d.get("/api/plugins").await?;
    let list = arr(&v, "plugins");
    list.iter().find(|p| p.get("id").and_then(Value::as_str) == Some(id)).cloned().ok_or_else(|| {
        let ids: Vec<&str> = list.iter().filter_map(|p| p.get("id").and_then(Value::as_str)).collect();
        format!("No plugin {id:?} is installed: plugins_list shows them ({}).", if ids.is_empty() { "none".to_string() } else { ids.join(", ") })
    })
}

fn commands_of(record: &Value) -> Vec<String> {
    record
        .pointer("/manifest/connectors")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|c| c.get("commands").and_then(Value::as_array).cloned().unwrap_or_default())
        .filter_map(|c| c.as_str().map(str::to_string))
        .collect()
}

/// The connector that declares `command`, as the dashboard finds it.
fn connector_for(record: &Value, command: &str) -> Option<String> {
    record
        .pointer("/manifest/connectors")
        .and_then(Value::as_array)?
        .iter()
        .find(|c| c.get("commands").and_then(Value::as_array).is_some_and(|l| l.iter().any(|x| x.as_str() == Some(command))))
        .and_then(|c| c.get("id").and_then(Value::as_str))
        .map(str::to_string)
}

/// Send `command` through the gated connector route, with a key of its own
/// (the gate requires one for a journalled command), and take off the SDK
/// envelope as the dashboard does.
async fn connector_call(d: &Desk, connector: &str, command: &str, payload: Option<Value>) -> Result<Value, String> {
    let mut body = json!({ "command": command, "idempotencyKey": format!("agent-{}", uuid::Uuid::new_v4().simple()) });
    if let Some(p) = payload {
        body["payload"] = p;
    }
    let v = d.post(&format!("/api/bridge/connectors/{}/request", seg(connector)), body).await?;
    let result = v.get("result").cloned().unwrap_or(Value::Null);
    crate::setup::unwrap_reply(result).map_err(|why| format!("{command}: {why}"))
}

/// The first `settings` step of a plugin's setup, if it declares one.
fn settings_step(record: &Value) -> Option<Value> {
    record
        .pointer("/manifest/setup/steps")
        .and_then(Value::as_array)?
        .iter()
        .find(|s| s.get("kind").and_then(Value::as_str) == Some("settings"))
        .cloned()
}

async fn plugin_settings_get(d: &Desk, id: &str) -> Result<Done, String> {
    let record = plugin_record(d, id).await?;
    let step = settings_step(&record);
    // As the dashboard reads them: the step's `read`, else settings.get's `settings`.
    let (command, path) = match step.as_ref().and_then(|s| s.get("read")).filter(|r| !r.is_null()) {
        Some(read) => (s(read, "command"), read.get("path").and_then(Value::as_str).map(str::to_string)),
        None => ("settings.get".to_string(), Some("settings".to_string())),
    };
    let connector = connector_for(&record, &command)
        .ok_or_else(|| format!("{} has no settings OAIY can read: it declares no {command} command.", plugin_name(&record)))?;
    let answer = connector_call(d, &connector, &command, None).await?;
    let bag = match path.as_deref().filter(|p| !p.is_empty()) {
        Some(p) => crate::setup::lookup(&answer, p).cloned().unwrap_or(Value::Null),
        None => answer,
    };
    let fields = step.as_ref().and_then(|s| s.get("fields")).cloned().unwrap_or_else(|| json!([]));
    data(json!({ "pluginId": id, "fields": fields, "settings": trim(&audit::redact(&bag)) }))
}

/// `settings` without the values plugin_settings_get hid: sent back, they
/// would overwrite the plugin's real secret with the word itself. Answers the
/// keys left out.
pub(crate) fn without_hidden(settings: &mut Value) -> Vec<String> {
    fn go(v: &mut Value, at: &str, out: &mut Vec<String>) {
        if let Value::Object(o) = v {
            let hidden: Vec<String> = o.iter().filter(|(_, x)| x.as_str() == Some(audit::REDACTED)).map(|(k, _)| k.clone()).collect();
            for k in hidden {
                o.remove(&k);
                out.push(format!("{at}{k}"));
            }
            for (k, x) in o.iter_mut() {
                go(x, &format!("{at}{k}."), out);
            }
        }
    }
    let mut out = Vec::new();
    go(settings, "", &mut out);
    out
}

async fn plugin_settings_set(d: &Desk, id: &str, mut settings: Value) -> Result<Done, String> {
    let kept = without_hidden(&mut settings);
    let n = settings.as_object().map_or(0, Map::len);
    if n == 0 {
        if !kept.is_empty() {
            return Err(format!("{} were hidden when read and are left as they are: give a real value to change one.", kept.join(", ")));
        }
        return Err("settings names no setting to change: give {key: value} for each.".into());
    }
    let record = plugin_record(d, id).await?;
    let command = settings_step(&record)
        .and_then(|s| s.pointer("/write/command").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| "settings.set".to_string());
    let connector = connector_for(&record, &command)
        .ok_or_else(|| format!("{} has no settings OAIY can change: it declares no {command} command.", plugin_name(&record)))?;
    let answer = connector_call(d, &connector, &command, Some(settings)).await?;
    let later: Vec<String> = arr(&answer, "appliesAtReconnect").iter().filter_map(Value::as_str).map(str::to_string).collect();
    let mut summary = format!("Saved {n} setting{} of {}.", if n == 1 { "" } else { "s" }, plugin_name(&record));
    if !later.is_empty() {
        summary.push_str(&format!(" These take effect only when it starts again (plugin_restart): {}.", later.join(", ")));
    }
    if let Some(blocked) = answer.get("blocked").and_then(Value::as_str) {
        summary.push_str(&format!(" It is paused: {blocked}"));
    }
    if !kept.is_empty() {
        summary.push_str(&format!(" Left as they are (hidden when read): {}.", kept.join(", ")));
    }
    done(summary, json!({ "pluginId": id, "answer": trim(&audit::redact(&answer)) }))
}

fn step_ids(detail: &Value) -> Vec<String> {
    detail
        .pointer("/setup/steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| s.get("id").and_then(Value::as_str).map(str::to_string))
        .collect()
}

/// A setup check through the route the wizard uses (`?check=when` for whether a step applies).
async fn setup_check(d: &Desk, id: &str, step: &str, when: bool) -> Value {
    let path = format!("/api/setup/plugins/{}/check/{}{}", seg(id), seg(step), if when { "?check=when" } else { "" });
    match d.post_within(&path, json!({}), Duration::from_secs(8)).await {
        Ok(v) => json!({ "passed": v.get("passed"), "detail": v.get("detail") }),
        Err(e) => json!({ "passed": false, "detail": e }),
    }
}

async fn plugin_setup_status(d: &Desk, id: &str) -> Result<Done, String> {
    let detail = d.get(&format!("/api/setup/plugins/{}", seg(id))).await?;
    let name = s(&detail, "name");
    let Some(setup) = detail.get("setup").filter(|x| !x.is_null()) else {
        return done(format!("{name} declares no setup: there is nothing to set up."), json!({ "pluginId": id, "declaresSetup": false }));
    };
    let mut steps: Vec<Value> = arr(setup, "steps").to_vec();
    // The host always shows the permissions step first, declared or not.
    if !steps.iter().any(|st| st.get("kind").and_then(Value::as_str) == Some(crate::setup::PERMISSIONS)) {
        steps.insert(0, json!({ "id": crate::setup::PERMISSIONS, "kind": crate::setup::PERMISSIONS, "title": format!("What {name} may do") }));
    }
    let done_list: Vec<String> = detail.pointer("/state/done").and_then(Value::as_array).into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
    let skipped_list: Vec<String> = detail.pointer("/state/skipped").and_then(Value::as_array).into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
    let accepted = detail.get("permissionsAccepted").and_then(Value::as_bool).unwrap_or(false);
    let needs_requirements = steps.iter().any(|st| st.get("kind").and_then(Value::as_str) == Some("requirements"));
    let (services, chosen) = if needs_requirements {
        let (a, b) = tokio::join!(d.get_within("/api/services", QUICK), d.get_within("/api/engines/defaults", QUICK));
        (a.ok(), b.ok())
    } else {
        (None, None)
    };
    // Every step's live checks, at once.
    let checks = futures_util::future::join_all(steps.iter().map(|st| async move {
        let step_id = s(st, "id");
        let when = if st.get("when").is_some_and(|w| !w.is_null()) { Some(setup_check(d, id, &step_id, true).await) } else { None };
        let done_check = if st.get("kind").and_then(Value::as_str) == Some("screen") && st.get("done").is_some_and(|x| !x.is_null()) {
            Some(setup_check(d, id, &step_id, false).await)
        } else {
            None
        };
        (when, done_check)
    }))
    .await;
    // Set up by what is true now (the dashboard's rule), from the same answers.
    let live = set_up_by_checks(&checks.iter().map(|(w, c)| (w.as_ref().map(passed), c.as_ref().map(passed))).collect::<Vec<_>>());
    let mut outstanding = Vec::new();
    let mut rows = Vec::new();
    for (st, (when, done_check)) in steps.iter().zip(checks) {
        let step_id = s(st, "id");
        let kind = s(st, "kind");
        let applies = when.as_ref().is_none_or(|w| w.get("passed") == Some(&Value::Bool(true)));
        let recorded = if kind == crate::setup::PERMISSIONS {
            accepted.then_some("done")
        } else if done_list.contains(&step_id) {
            Some("done")
        } else if skipped_list.contains(&step_id) {
            Some("skipped")
        } else {
            None
        };
        // Accepting what a plugin may do, a screen of its own, a host step: the person's, on screen.
        let who = if matches!(kind.as_str(), "permissions" | "screen" | "host") { "person" } else { "agent" };
        let mut row = json!({
            "id": step_id, "kind": kind, "title": st.get("title"), "optional": st.get("optional").cloned().unwrap_or(Value::Bool(false)),
            "recorded": recorded, "applies": applies, "who": who,
        });
        let mut complete = recorded.is_some();
        if let Some(c) = &done_check {
            complete |= c.get("passed") == Some(&Value::Bool(true));
            row["check"] = c.clone();
        }
        if let Some(w) = &when {
            row["appliesBecause"] = w.get("detail").cloned().unwrap_or(Value::Null);
        }
        if kind == "requirements" {
            let reqs: Vec<Value> = arr(st, "requires")
                .iter()
                .map(|r| match r.get("kind").and_then(Value::as_str) {
                    Some("service") => {
                        let sid = s(r, "id");
                        let svc = services.as_ref().and_then(|v| arr(v, "services").iter().find(|x| x.get("id").and_then(Value::as_str) == Some(sid.as_str())).cloned());
                        let installed = svc.as_ref().and_then(|x| x.get("installed")).and_then(Value::as_bool).unwrap_or(false);
                        json!({ "kind": "service", "id": sid, "met": installed, "status": svc.as_ref().and_then(|x| x.get("status").cloned()), "why": r.get("why") })
                    }
                    Some("engineModel") => {
                        let group = s(r, "group");
                        let model = chosen.as_ref().and_then(|c| c.pointer(&format!("/defaults/{group}")).cloned()).filter(|m| !m.is_null());
                        json!({ "kind": "engineModel", "group": group, "met": model.is_some(), "chosen": model, "why": r.get("why") })
                    }
                    _ => r.clone(),
                })
                .collect();
            complete |= !reqs.is_empty() && reqs.iter().all(|r| r.get("met") == Some(&Value::Bool(true)));
            row["requires"] = Value::Array(reqs);
        }
        if kind == "settings" {
            row["fields"] = Value::Array(arr(st, "fields").iter().filter_map(|f| f.get("key").cloned()).collect());
        }
        if kind == crate::setup::PERMISSIONS {
            // What the person is asked to accept, for the Agent to explain before showing it.
            row["capabilities"] = detail.get("capabilities").cloned().unwrap_or_else(|| json!([]));
        }
        row["complete"] = json!(complete);
        if applies && !complete {
            outstanding.push(json!(step_id));
        }
        rows.push(row);
    }
    let finished = detail.pointer("/state/version").and_then(Value::as_u64).unwrap_or(0);
    let version = setup.get("version").and_then(Value::as_u64).unwrap_or(1);
    // As the dashboard judges it: finished here, or its shown done checks all pass now.
    let set_up_by = if version <= finished {
        Some("record")
    } else if live == Some(true) {
        Some("checks")
    } else {
        None
    };
    if set_up_by.is_some() {
        // Set up: nothing is left to do, though any step may still be changed.
        outstanding.clear();
    }
    data(json!({
        "pluginId": id,
        "name": name,
        "title": setup.get("title"),
        "version": version,
        "finishedVersion": finished,
        "needsSetup": set_up_by.is_none(),
        "setUpBy": set_up_by,
        "permissionsAccepted": accepted,
        "steps": rows,
        "outstanding": outstanding,
    }))
}

// ---------------------------------------------------------------------------
// Flows
// ---------------------------------------------------------------------------

async fn flow_ids(d: &Desk) -> Result<Vec<String>, String> {
    let v = d.get("/api/bridge/flows").await?;
    Ok(arr(&v, "flows").iter().filter_map(|f| f.get("flowId").and_then(Value::as_str).map(str::to_string)).collect())
}

/// An id made from a flow's name: lower case letters and digits, joined by `-`.
pub(crate) fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out: String = out.trim_end_matches('-').chars().take(60).collect();
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() { "flow".into() } else { out }
}

async fn flow_create(d: &Desk, id: Option<&str>, flow: Value) -> Result<Done, String> {
    let existing = flow_ids(d).await?;
    let name = flow.get("name").or_else(|| flow.get("title")).and_then(Value::as_str).unwrap_or_default().to_string();
    let id = match id {
        Some(id) => {
            if !crate::bridge::FlowStore::valid_id(id) {
                return Err(format!("{id:?} cannot be a flow id: use letters, digits, - and _ (at most 128)."));
            }
            if existing.iter().any(|e| e == id) {
                return Err(format!("A flow {id:?} exists: flow_update changes it, or choose another id."));
            }
            id.to_string()
        }
        None => {
            let base = slug(&name);
            let mut id = base.clone();
            let mut n = 2;
            while existing.contains(&id) {
                id = format!("{base}-{n}");
                n += 1;
            }
            id
        }
    };
    d.put_json_body(&format!("/api/bridge/flows/{}", seg(&id)), flow).await?;
    let shown = if name.is_empty() { id.clone() } else { format!("{id} ({name})") };
    done(format!("Stored the flow {shown}."), json!({ "id": id, "name": name }))
}

async fn flow_run(d: &Desk, id: &str, input: Option<Value>, wait: bool, timeout_s: u64) -> Result<Done, String> {
    if !flow_ids(d).await?.iter().any(|f| f == id) {
        return Err(format!("There is no flow {id:?}: flows_list shows them."));
    }
    let run = uuid::Uuid::new_v4().simple().to_string();
    let mut body = json!({
        "protocol": crate::http::BRIDGE_PROTOCOL,
        "caller": { "product": "oaiy-agent", "label": "the Agent" },
        "flowId": id,
        "mode": if wait { "sync" } else { "async" },
        "timeoutMs": timeout_s * 1000,
        "correlationId": format!("agent-{run}"),
        "idempotencyKey": format!("agent-{run}"),
    });
    if let Some(input) = input {
        body["input"] = input;
    }
    let r = d.post_within("/api/bridge/runs", body, Duration::from_secs(timeout_s + 20)).await?;
    let status = r.get("status").and_then(Value::as_str).unwrap_or("unknown").to_string();
    let error = r.get("error").filter(|e| !e.is_null()).map(|e| e.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| e.to_string()));
    let summary = match (status.as_str(), &error) {
        (_, Some(e)) => format!("The run of {id} {status}: {e}"),
        ("succeeded", None) => format!("The run of {id} succeeded."),
        (s, None) => format!("The run of {id} is {s}."),
    };
    done(summary, json!({ "runId": r.get("runId"), "status": status, "output": r.get("output").map(trim), "error": r.get("error") }))
}

// ---------------------------------------------------------------------------
// Calendar, setup, pages
// ---------------------------------------------------------------------------

async fn calendar_settings_set(d: &Desk, a: &Args<'_>) -> Result<Done, String> {
    const FIELDS: [&str; 7] = ["business", "hours", "services", "slotMinutes", "noticeMinutes", "horizonDays", "textConfirmations"];
    let given: Vec<&str> = FIELDS.iter().copied().filter(|k| a.value(k).is_some()).collect();
    if given.is_empty() {
        return Err(format!("Give at least one of {} to change.", FIELDS.join(", ")));
    }
    let current = d.get("/api/calendar").await?;
    let mut settings = current
        .get("settings")
        .cloned()
        .filter(Value::is_object)
        .ok_or_else(|| "The calendar answered no settings to change.".to_string())?;
    for key in &given {
        if let Some(v) = a.value(key) {
            settings[*key] = v.clone();
        }
    }
    let saved = d.put("/api/calendar/settings", settings).await?;
    done(format!("Saved the calendar's {}.", given.join(", ")), json!({ "settings": saved }))
}

async fn setup_status(d: &Desk) -> Result<Done, String> {
    let (state, plugins) = tokio::join!(d.get("/api/setup"), d.get("/api/plugins"));
    let state = state?;
    let rows: Vec<Value> = match plugins {
        Ok(p) => {
            let list = arr(&p, "plugins");
            let judged = judge_plugins(d, list, Some(&state)).await;
            list.iter()
                .map(|p| {
                    let id = s(p, "id");
                    let (_, finished) = setup_versions(p, Some(&state));
                    let j = judged.get(&id);
                    json!({
                        "id": id,
                        "declaresSetup": j.is_some_and(|j| j.declared),
                        "needsSetup": j.map(|j| j.needs),
                        "setUpBy": j.and_then(|j| j.by),
                        "finishedVersion": finished,
                        "done": state.pointer(&format!("/plugins/{id}/done")),
                        "skipped": state.pointer(&format!("/plugins/{id}/skipped")),
                    })
                })
                .collect()
        }
        Err(e) => vec![json!({ "error": e })],
    };
    let first = state.get("firstRun").cloned().unwrap_or_else(|| json!({}));
    data(json!({
        "firstRun": {
            "finished": first.get("finished"),
            "position": first.get("position"),
            "skipped": first.get("skipped"),
            "chosenPlugins": first.get("chosenPlugins"),
            "recordedFinishedBecause": first.get("migrated"),
        },
        "plugins": rows,
    }))
}

/// The dashboard's own pages, as it names them.
pub(crate) const VIEWS: [&str; 14] = [
    "overview", "agent", "flows", "calendar", "engines", "services", "plugins", "runs", "models", "python", "providers", "connections", "settings", "setup",
];

/// A page the dashboard has: one of [`VIEWS`], or `plugin:<pluginId>:<navId>`.
pub(crate) fn view_ok(view: &str) -> bool {
    if VIEWS.contains(&view) {
        return true;
    }
    let mut parts = view.split(':');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("plugin"), Some(id), Some(nav), None) => {
            crate::setup::valid_plugin_id(id) && !nav.is_empty() && nav.len() <= 64 && nav.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        }
        _ => false,
    }
}
