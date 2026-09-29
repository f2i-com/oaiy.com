//! The Agent's model: which AI the Agent app answers with, and what this
//! computer can run it on.
//!
//! - `GET`/`PUT /api/agent/preferences`: the person's choice, stored in
//!   `<data>/agent.json`.
//!   ```json
//!   { "model": { "source": "engine" } }
//!   { "model": { "source": "chatgpt", "model": "gpt-5.5" } }
//!   ```
//!   `engine` (the default) is OAIY's own engine, answering with the model
//!   chosen in Engines. `chatgpt` is the Codex connector's generic route; with
//!   no `model` it runs Codex's own default (its catalogue's `isDefault`).
//! - `GET /api/engines/recommendation`: whether the engine or ChatGPT suits
//!   this computer, from what Engines already reports (its GPUs, the model
//!   chosen there, its catalog's recommended language model) and whether
//!   ChatGPT is signed in. Nothing here names a model of its own.
//!
//! Both are restricted routes, gated in `http.rs` like the other `/api/agent/`
//! and `/api/engines/` routes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Map, Value};

use super::routes::{ai_error, AiState};

/// The file, in the data folder.
pub const FILE: &str = "agent.json";

/// Where the Agent's model comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Engine,
    Chatgpt,
}

impl Source {
    fn as_str(self) -> &'static str {
        match self {
            Source::Engine => "engine",
            Source::Chatgpt => "chatgpt",
        }
    }
}

/// The person's choice of model for the Agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelChoice {
    pub source: Source,
    /// ChatGPT's model; `None` is Codex's own default. Always `None` for the
    /// engine, which answers with the model chosen in Engines.
    pub model: Option<String>,
}

impl Default for ModelChoice {
    fn default() -> Self {
        Self { source: Source::Engine, model: None }
    }
}

impl ModelChoice {
    pub fn to_json(&self) -> Value {
        let mut m = json!({ "source": self.source.as_str() });
        if let Some(model) = &self.model {
            m["model"] = json!(model);
        }
        m
    }

    /// Read `{source, model?}`, refusing anything else with a message that
    /// names the fix (an MCP tool relays it to a model, which acts on it).
    pub fn parse(v: &Value) -> Result<Self, String> {
        let obj = v.as_object().ok_or("`model` must be an object: { \"source\": \"engine\" | \"chatgpt\", \"model\"?: string }")?;
        if let Some(key) = obj.keys().find(|k| *k != "source" && *k != "model") {
            return Err(format!("`model.{key}` is not a setting: only `source` and `model` are"));
        }
        let source = match obj.get("source").and_then(Value::as_str) {
            Some("engine") => Source::Engine,
            Some("chatgpt") => Source::Chatgpt,
            _ => return Err("`model.source` must be \"engine\" or \"chatgpt\"".into()),
        };
        let model = match obj.get("model") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => {
                let s = s.trim();
                if s.is_empty() {
                    return Err("`model.model` must not be empty: leave it out for ChatGPT's own default".into());
                }
                if s.len() > 256 || s.chars().any(char::is_control) {
                    return Err("`model.model` is not a model name".into());
                }
                Some(s.to_string())
            }
            Some(_) => return Err("`model.model` must be a string".into()),
        };
        if source == Source::Engine && model.is_some() {
            return Err("the engine answers with the model chosen in Engines: choose it there (model_set_default for the llm group), and send no `model` with \"engine\"".into());
        }
        Ok(Self { source, model })
    }
}

/// The preferences file.
pub struct Store {
    path: PathBuf,
    /// One write at a time: a read-modify-write that raced another would
    /// drop its change.
    write: Mutex<()>,
}

impl Store {
    pub fn open(data_dir: &Path) -> Self {
        Self { path: data_dir.join(FILE), write: Mutex::new(()) }
    }

    /// The whole document as stored: an object, or an empty one when there is
    /// no file (or one that is not an object).
    fn document(&self) -> Map<String, Value> {
        let Ok(text) = std::fs::read_to_string(&self.path) else {
            return Map::new();
        };
        // (A byte-order mark, as some editors write one, is not a reason to lose the choice.)
        match serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) {
            Ok(Value::Object(doc)) => doc,
            _ => {
                log::warn!("{} is not a JSON object: the Agent's model is the engine until it is chosen again", self.path.display());
                Map::new()
            }
        }
    }

    /// The Agent's model: the engine unless something valid was chosen.
    pub fn model(&self) -> ModelChoice {
        match self.document().get("model") {
            None => ModelChoice::default(),
            Some(v) => ModelChoice::parse(v).unwrap_or_else(|e| {
                log::warn!("{}: {e}; the Agent's model is the engine until it is chosen again", self.path.display());
                ModelChoice::default()
            }),
        }
    }

    /// Store `choice`, keeping anything else the file holds. Written whole to
    /// a temporary file and renamed over, so a crash leaves the old choice or
    /// the new one, never half of one.
    pub fn set_model(&self, choice: &ModelChoice) -> Result<(), String> {
        let _guard = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let mut doc = self.document();
        doc.insert("model".into(), choice.to_json());
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(&Value::Object(doc)).map_err(|e| e.to_string())?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, text.as_bytes())
            .and_then(|()| std::fs::rename(&tmp, &self.path))
            .map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("could not save {}: {e}", self.path.display())
            })
    }
}

/// `{ "model": {…} }`, as both routes answer.
fn preferences(store: &Store) -> Value {
    json!({ "model": store.model().to_json() })
}

/// The routes: the preference, and the recommendation.
pub fn router(state: AiState) -> Router {
    let data_dir = state.registry.lock().unwrap_or_else(|e| e.into_inner()).data_dir().to_path_buf();
    let store = Arc::new(Store::open(&data_dir));
    preferences_router(store).merge(
        Router::new()
            .route("/api/engines/recommendation", get(recommendation))
            .with_state(Recommend { codex: state.codex, data_dir }),
    )
}

fn preferences_router(store: Arc<Store>) -> Router {
    Router::new()
        .route("/api/agent/preferences", get(get_preferences).put(put_preferences))
        .with_state(store)
}

async fn get_preferences(State(store): State<Arc<Store>>) -> Response {
    Json(preferences(&store)).into_response()
}

async fn put_preferences(State(store): State<Arc<Store>>, body: Option<Json<Value>>) -> Response {
    let bad = |m: String| ai_error(StatusCode::BAD_REQUEST, "invalid_request", m);
    let Some(Json(body)) = body else {
        return bad("send JSON: { \"model\": { \"source\": \"engine\" | \"chatgpt\", \"model\"?: string } }".into());
    };
    let Some(obj) = body.as_object() else {
        return bad("send an object: { \"model\": { … } }".into());
    };
    if let Some(key) = obj.keys().find(|k| *k != "model") {
        return bad(format!("`{key}` is not a preference: only `model` is"));
    }
    let Some(model) = obj.get("model") else {
        return bad("`model` is missing: { \"model\": { \"source\": \"engine\" | \"chatgpt\" } }".into());
    };
    let choice = match ModelChoice::parse(model) {
        Ok(c) => c,
        Err(e) => return bad(e),
    };
    let saving = store.clone();
    match tokio::task::spawn_blocking(move || saving.set_model(&choice)).await {
        Ok(Ok(())) => Json(preferences(&store)).into_response(),
        Ok(Err(e)) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
        Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// The recommendation
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Recommend {
    codex: super::codex::CodexHandle,
    data_dir: PathBuf,
}

/// How long the recommendation waits to learn whether ChatGPT is signed in.
/// Asking starts the Codex child the first time, which can take a while; the
/// wizard showing this should not wait for it. (A ChatGPT turn in progress no
/// longer holds the question up: status is answered beside it.)
const SIGN_IN_BUDGET: Duration = Duration::from_secs(5);

/// The last answer, for when the budget runs out.
static LAST_SIGNED_IN: AtomicBool = AtomicBool::new(false);

async fn recommendation(State(st): State<Recommend>) -> Response {
    let codex = st.codex.clone();
    let asking = tokio::task::spawn_blocking(move || {
        let signed_in = codex.status().connected;
        LAST_SIGNED_IN.store(signed_in, Ordering::Release);
        signed_in
    });
    let signed_in = async move {
        match tokio::time::timeout(SIGN_IN_BUDGET, asking).await {
            Ok(Ok(signed_in)) => signed_in,
            // Still asking (it finishes on its own and is remembered): the last answer.
            _ => LAST_SIGNED_IN.load(Ordering::Acquire),
        }
    };
    let (mut answer, signed_in) = tokio::join!(recommendation_at(crate::http::engines_ui(), &st.data_dir), signed_in);
    answer["chatgpt"] = json!({ "signedIn": signed_in });
    Json(answer).into_response()
}

/// What this computer can run the Agent on, from the engines at `ui` (none:
/// they are not running) and the engines' configuration in `data_dir`.
///
/// Local is ok when a language model is already chosen in Engines, or when the
/// catalog's recommended language model states a memory need the largest GPU's
/// total memory meets. Otherwise ChatGPT is recommended; the engine stays a
/// choice. The `chatgpt` part is the caller's to fill in.
pub(crate) async fn recommendation_at(ui: Option<String>, data_dir: &Path) -> Value {
    let (catalog, system) = match ui.as_deref() {
        Some(ui) => {
            let (catalog, system) = tokio::join!(
                crate::http::engines_catalog_at(Some(ui.to_string())),
                crate::http::studio_json(ui, reqwest::Method::GET, "/api/system", None)
            );
            (catalog, system.ok())
        }
        None => (json!({ "running": false }), None),
    };
    let running = catalog.get("running").and_then(Value::as_bool) == Some(true);
    let gpus = system.as_ref().map(gpus_of).unwrap_or_default();

    // The model chosen in Engines: the running engines' discovery document
    // when it answered (its `defaults.llm`), else their configuration as it
    // would report it.
    let chosen = match catalog.pointer("/defaults/llm") {
        Some(v) if running => v.as_str().map(str::to_string),
        _ => chosen_in_config(data_dir),
    };
    // The catalog's recommended language model, as the catalog relay shows it.
    let suggested = catalog
        .get("models")
        .and_then(Value::as_array)
        .and_then(|models| {
            models.iter().find(|m| {
                m.get("group").and_then(Value::as_str) == Some("llm") && m.get("recommended").and_then(Value::as_bool) == Some(true)
            })
        })
        .cloned();

    let largest = gpus
        .iter()
        .filter_map(|g| Some((g.get("name").and_then(Value::as_str).unwrap_or("GPU").to_string(), g.get("totalGb")?.as_f64()?)))
        .fold(None::<(String, f64)>, |best, g| match best {
            Some(b) if b.1 >= g.1 => Some(b),
            _ => Some(g),
        });

    let (ok, reason) = judge(running, chosen.as_deref(), suggested.as_ref(), largest.as_ref());
    json!({
        "recommend": if ok { "engine" } else { "chatgpt" },
        "local": {
            "ok": ok,
            "reason": reason,
            "gpus": gpus,
            "chosen": chosen,
            "suggested": suggested,
        },
        "chatgpt": { "signedIn": false },
    })
}

/// The rule, and the sentence that says why.
fn judge(running: bool, chosen: Option<&str>, suggested: Option<&Value>, largest: Option<&(String, f64)>) -> (bool, String) {
    if let Some(model) = chosen {
        let later = if running { "" } else { " (the engines are not running now; it runs when they start)" };
        return (true, format!("A language model is chosen in Engines: {model}{later}."));
    }
    if !running {
        return (
            false,
            "The engines are not running, so this computer's GPUs and the catalog cannot be checked, and no language model is chosen in Engines.".into(),
        );
    }
    let Some(s) = suggested else {
        return (false, "The engines' catalog recommends no language model, and none is chosen in Engines.".into());
    };
    let name = s.get("name").and_then(Value::as_str).or_else(|| s.get("id").and_then(Value::as_str)).unwrap_or("the recommended language model");
    let Some(need) = s.get("vramGb").and_then(Value::as_f64).filter(|n| *n > 0.0) else {
        return (false, format!("The catalog's recommended language model, {name}, states no memory need, so it cannot be matched to this computer's GPUs."));
    };
    match largest {
        None => (false, format!("No GPU was found, and the catalog's recommended language model, {name}, needs {} GB of GPU memory.", gb_text(need))),
        Some((gpu, total)) if *total >= need => (
            true,
            format!("The largest GPU, {gpu}, has {} GB: enough for the catalog's recommended language model, {name}, which needs {} GB.", gb_text(*total), gb_text(need)),
        ),
        Some((gpu, total)) => (
            false,
            format!("The catalog's recommended language model, {name}, needs {} GB of GPU memory; the largest GPU, {gpu}, has {} GB.", gb_text(need), gb_text(*total)),
        ),
    }
}

/// The GPUs as Engines reports them (`/api/system`: nvidia-smi's MiB), in GB
/// to one place as its overview page shows them. The comparison with a
/// model's need uses these shown figures: an "8 GB" card reports 8188 MiB,
/// 7.996 GB, which the page shows as 8.0 — and which meets an 8 GB need.
fn gpus_of(system: &Value) -> Vec<Value> {
    system
        .get("gpus")
        .and_then(Value::as_array)
        .map(|gpus| {
            gpus.iter()
                .filter_map(|g| {
                    let total = g.get("memory_total_mb").and_then(Value::as_f64).filter(|t| *t > 0.0)?;
                    let used = g.get("memory_used_mb").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, total);
                    Some(json!({
                        "name": g.get("name").and_then(Value::as_str).unwrap_or("GPU"),
                        "totalGb": gb(total),
                        "freeGb": gb(total - used),
                    }))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn gb(mb: f64) -> f64 {
    (mb / 1024.0 * 10.0).round() / 10.0
}

fn gb_text(gb: f64) -> String {
    if gb.fract() == 0.0 {
        format!("{gb:.0}")
    } else {
        format!("{gb:.1}")
    }
}

/// The language model the engines' configuration chooses, as their discovery
/// document would report it: `llm.default_model`, else the first language
/// model not switched off.
fn chosen_in_config(data_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(data_dir.join("engines").join("oaiy-studio.json")).ok()?;
    let cfg: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).ok()?;
    let llm = cfg.get("llm")?;
    let named = llm.get("default_model").and_then(Value::as_str).map(str::trim).filter(|m| !m.is_empty());
    let first = || {
        llm.get("models")
            .and_then(Value::as_array)?
            .iter()
            .find(|m| m.get("enabled").and_then(Value::as_bool).unwrap_or(true))
            .and_then(|m| m.get("name").and_then(Value::as_str))
            .map(str::trim)
            .filter(|m| !m.is_empty())
    };
    named.or_else(first).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("oaiy-agent-model-{tag}-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
            std::fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // ---- the preference ----

    #[test]
    fn a_choice_is_engine_or_chatgpt_with_an_optional_model() {
        assert_eq!(ModelChoice::parse(&json!({ "source": "engine" })).unwrap(), ModelChoice::default());
        assert_eq!(
            ModelChoice::parse(&json!({ "source": "chatgpt", "model": " gpt-5.5 " })).unwrap(),
            ModelChoice { source: Source::Chatgpt, model: Some("gpt-5.5".into()) }
        );
        // No model is ChatGPT's own default; null says the same.
        for v in [json!({ "source": "chatgpt" }), json!({ "source": "chatgpt", "model": null })] {
            assert_eq!(ModelChoice::parse(&v).unwrap().model, None);
        }
        for bad in [
            json!("chatgpt"),
            json!({}),
            json!({ "source": "openai" }),
            json!({ "source": "ChatGPT" }),
            json!({ "source": "chatgpt", "model": "" }),
            json!({ "source": "chatgpt", "model": "   " }),
            json!({ "source": "chatgpt", "model": 5 }),
            json!({ "source": "chatgpt", "model": "a\nb" }),
            json!({ "source": "chatgpt", "model": "x".repeat(257) }),
            json!({ "source": "chatgpt", "effort": "high" }),
            // The engine's model is the one chosen in Engines, never one named here.
            json!({ "source": "engine", "model": "qwen3.5-9b" }),
        ] {
            assert!(ModelChoice::parse(&bad).is_err(), "{bad}");
        }
        let why = ModelChoice::parse(&json!({ "source": "engine", "model": "m" })).unwrap_err();
        assert!(why.contains("model_set_default"), "the refusal names the fix: {why}");
    }

    #[test]
    fn the_store_defaults_to_the_engine_and_keeps_a_choice() {
        let dir = Scratch::new("store");
        let store = Store::open(&dir.0);
        assert_eq!(store.model(), ModelChoice::default(), "no file: the engine");
        let chatgpt = ModelChoice { source: Source::Chatgpt, model: Some("gpt-5.5".into()) };
        store.set_model(&chatgpt).unwrap();
        assert_eq!(Store::open(&dir.0).model(), chatgpt, "kept across a restart");
        let on_disk: Value = serde_json::from_str(&std::fs::read_to_string(dir.0.join(FILE)).unwrap()).unwrap();
        assert_eq!(on_disk, json!({ "model": { "source": "chatgpt", "model": "gpt-5.5" } }));
        assert!(!dir.0.join("agent.json.tmp").exists(), "no temporary file is left");

        // Whatever else the file holds is kept.
        std::fs::write(dir.0.join(FILE), "\u{feff}{\"later\":{\"x\":1},\"model\":{\"source\":\"chatgpt\"}}").unwrap();
        assert_eq!(store.model(), ModelChoice { source: Source::Chatgpt, model: None }, "a byte-order mark is read through");
        store.set_model(&ModelChoice::default()).unwrap();
        let on_disk: Value = serde_json::from_str(&std::fs::read_to_string(dir.0.join(FILE)).unwrap()).unwrap();
        assert_eq!(on_disk, json!({ "later": { "x": 1 }, "model": { "source": "engine" } }));

        // A file that is not a choice is the engine, not an error.
        for junk in ["not json", "[1,2]", "{\"model\":{\"source\":\"llama\"}}"] {
            std::fs::write(dir.0.join(FILE), junk).unwrap();
            assert_eq!(store.model(), ModelChoice::default(), "{junk}");
        }
    }

    async fn serve(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        (base, tokio::spawn(async move { axum::serve(listener, router).await.unwrap() }))
    }

    #[tokio::test]
    async fn the_preference_round_trips_over_http() {
        let dir = Scratch::new("http");
        let (base, server) = serve(preferences_router(Arc::new(Store::open(&dir.0)))).await;
        let url = format!("{base}/api/agent/preferences");
        let http = reqwest::Client::new();

        let v: Value = http.get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(v, json!({ "model": { "source": "engine" } }));

        let put = http.put(&url).json(&json!({ "model": { "source": "chatgpt", "model": "gpt-5.5" } })).send().await.unwrap();
        assert_eq!(put.status(), 200);
        assert_eq!(put.json::<Value>().await.unwrap(), json!({ "model": { "source": "chatgpt", "model": "gpt-5.5" } }));
        let v: Value = http.get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(v["model"]["model"], "gpt-5.5");

        for bad in [
            json!({ "model": { "source": "nope" } }),
            json!({ "model": { "source": "chatgpt", "model": "" } }),
            json!({ "source": "chatgpt" }),
            json!({ "model": { "source": "engine" }, "extra": true }),
            json!([]),
        ] {
            let r = http.put(&url).json(&bad).send().await.unwrap();
            assert_eq!(r.status(), 400, "{bad}");
            let e: Value = r.json().await.unwrap();
            assert_eq!(e["error"]["code"], "invalid_request", "{bad}");
        }
        let r = http.put(&url).header("content-type", "application/json").body("{").send().await.unwrap();
        assert!(r.status().is_client_error());
        // A refused change changes nothing.
        let v: Value = http.get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(v, json!({ "model": { "source": "chatgpt", "model": "gpt-5.5" } }));

        let back = http.put(&url).json(&json!({ "model": { "source": "engine" } })).send().await.unwrap();
        assert_eq!(back.json::<Value>().await.unwrap(), json!({ "model": { "source": "engine" } }));
        server.abort();
    }

    // ---- the recommendation, against a fake studio ----

    /// A stand-in for the engines' control port: `/api/system`'s GPUs, the
    /// catalog (`/api/downloads`) and the discovery document's defaults.
    async fn fake_studio(gpus: Value, models: Value, chosen: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/api/system", get(move || { let gpus = gpus.clone(); async move { Json(json!({ "gpus": gpus, "ram": { "total_mb": 65536, "available_mb": 30000 } })) } }))
            .route("/api/downloads", get(move || { let models = models.clone(); async move { Json(json!({ "dir": "D:/models", "free": 1, "groups": [], "models": models })) } }))
            .route("/api/discovery", get(move || async move { Json(json!({ "defaults": { "llm": chosen, "image": "" } })) }));
        serve(app).await
    }

    fn catalog(vram: Value) -> Value {
        json!([
            { "id": "small-image", "group": "image", "name": "Small image", "vram_gb": 4, "recommended": true },
            { "id": "big-chat", "group": "llm", "name": "Big Chat", "size_gb": 9.5, "vram_gb": vram, "recommended": true, "installed": false },
            { "id": "tiny-chat", "group": "llm", "name": "Tiny Chat", "vram_gb": 2, "recommended": false },
        ])
    }

    fn rtx(name: &str, total_mb: i64, used_mb: i64) -> Value {
        json!({ "index": 0, "name": name, "memory_used_mb": used_mb, "memory_total_mb": total_mb, "utilization": 3, "temperature": 41 })
    }

    #[tokio::test]
    async fn a_gpu_that_meets_the_recommended_models_need_recommends_the_engine() {
        let dir = Scratch::new("fits");
        let (ui, server) = fake_studio(json!([rtx("RTX 3060", 12288, 1024), rtx("RTX 5090", 32607, 25849)]), catalog(json!(16)), "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["recommend"], "engine");
        assert_eq!(v["local"]["ok"], true);
        // The GPUs as the overview page shows them: GB to one place, from MiB.
        assert_eq!(v["local"]["gpus"][1], json!({ "name": "RTX 5090", "totalGb": 31.8, "freeGb": 6.6 }));
        assert_eq!(v["local"]["gpus"][0], json!({ "name": "RTX 3060", "totalGb": 12.0, "freeGb": 11.0 }));
        assert_eq!(v["local"]["chosen"], Value::Null);
        // The catalog's own entry, with its stated need: never a model of ours.
        assert_eq!(v["local"]["suggested"]["id"], "big-chat");
        assert_eq!(v["local"]["suggested"]["vramGb"], 16);
        let reason = v["local"]["reason"].as_str().unwrap();
        assert!(reason.contains("RTX 5090") && reason.contains("31.8 GB") && reason.contains("16 GB"), "{reason}");
        assert_eq!(v["chatgpt"], json!({ "signedIn": false }), "the route fills it in");
        server.abort();
    }

    #[tokio::test]
    async fn a_gpu_too_small_recommends_chatgpt_and_says_why() {
        let dir = Scratch::new("small");
        let (ui, server) = fake_studio(json!([rtx("RTX 3060", 12288, 0)]), catalog(json!(16)), "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["recommend"], "chatgpt");
        assert_eq!(v["local"]["ok"], false);
        assert_eq!(v["local"]["suggested"]["id"], "big-chat", "the engine stays a choice, with what it would take");
        let reason = v["local"]["reason"].as_str().unwrap();
        assert!(reason.contains("needs 16 GB") && reason.contains("RTX 3060") && reason.contains("12 GB"), "{reason}");
        server.abort();
    }

    #[tokio::test]
    async fn an_8_gb_card_meets_an_8_gb_need() {
        // nvidia-smi reports an "8 GB" card as 8188 MiB, 7.996 GB: shown as 8.0,
        // and compared as shown.
        let dir = Scratch::new("eight");
        let (ui, server) = fake_studio(json!([rtx("RTX 4060", 8188, 500)]), catalog(json!(8)), "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["local"]["gpus"][0]["totalGb"], 8.0);
        assert_eq!(v["recommend"], "engine", "{v}");
        server.abort();
    }

    #[tokio::test]
    async fn a_model_chosen_in_engines_is_enough_whatever_the_gpu() {
        let dir = Scratch::new("chosen");
        let (ui, server) = fake_studio(json!([]), catalog(json!(16)), "Qwen3.8-Flash-Next").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["recommend"], "engine");
        assert_eq!(v["local"]["chosen"], "Qwen3.8-Flash-Next");
        assert_eq!(v["local"]["gpus"], json!([]));
        assert!(v["local"]["reason"].as_str().unwrap().contains("Qwen3.8-Flash-Next"));
        server.abort();
    }

    #[tokio::test]
    async fn no_gpu_or_no_stated_need_or_no_recommended_model_is_chatgpt() {
        let dir = Scratch::new("none");
        let (ui, server) = fake_studio(json!([]), catalog(json!(16)), "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["recommend"], "chatgpt");
        assert!(v["local"]["reason"].as_str().unwrap().starts_with("No GPU was found"), "{v}");
        server.abort();

        let (ui, server) = fake_studio(json!([rtx("RTX 5090", 32607, 0)]), catalog(Value::Null), "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["recommend"], "chatgpt");
        assert!(v["local"]["reason"].as_str().unwrap().contains("states no memory need"), "{v}");
        server.abort();

        let only_image = json!([{ "id": "small-image", "group": "image", "vram_gb": 4, "recommended": true }]);
        let (ui, server) = fake_studio(json!([rtx("RTX 5090", 32607, 0)]), only_image, "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!((v["recommend"].as_str(), &v["local"]["suggested"]), (Some("chatgpt"), &Value::Null));
        assert!(v["local"]["reason"].as_str().unwrap().contains("recommends no language model"), "{v}");
        server.abort();
    }

    #[tokio::test]
    async fn with_the_engines_down_it_still_answers_and_says_why() {
        let dir = Scratch::new("down");
        for ui in [None, Some("http://127.0.0.1:9".to_string())] {
            let v = recommendation_at(ui.clone(), &dir.0).await;
            assert_eq!(v["recommend"], "chatgpt", "{ui:?}");
            assert_eq!(v["local"]["ok"], false);
            assert_eq!(v["local"]["gpus"], json!([]));
            assert_eq!(v["local"]["chosen"], Value::Null);
            assert_eq!(v["local"]["suggested"], Value::Null);
            assert!(v["local"]["reason"].as_str().unwrap().contains("engines are not running"), "{v}");
        }
        // …but a model chosen in their configuration still counts, as their
        // discovery would report it: the default, else the first one switched on.
        std::fs::create_dir_all(dir.0.join("engines")).unwrap();
        let config = dir.0.join("engines").join("oaiy-studio.json");
        std::fs::write(&config, "\u{feff}{\"llm\":{\"default_model\":\"\",\"models\":[{\"name\":\"Off\",\"enabled\":false},{\"name\":\"On\"}]}}").unwrap();
        let v = recommendation_at(None, &dir.0).await;
        assert_eq!((v["recommend"].as_str(), v["local"]["chosen"].as_str()), (Some("engine"), Some("On")));
        assert!(v["local"]["reason"].as_str().unwrap().contains("not running now"), "{v}");
        std::fs::write(&config, "{\"llm\":{\"default_model\":\"Named\",\"models\":[{\"name\":\"On\"}]}}").unwrap();
        assert_eq!(recommendation_at(None, &dir.0).await["local"]["chosen"], "Named");
        // A fresh install's configuration chooses nothing.
        std::fs::write(&config, "{\"llm\":{\"default_model\":\"\",\"models\":[]}}").unwrap();
        assert_eq!(recommendation_at(None, &dir.0).await["local"]["chosen"], Value::Null);
    }

    #[tokio::test]
    async fn running_engines_are_believed_over_their_configuration_file() {
        // The engines in use may keep their configuration elsewhere (a studio
        // beside a headless server): what they report wins.
        let dir = Scratch::new("believe");
        std::fs::create_dir_all(dir.0.join("engines")).unwrap();
        std::fs::write(dir.0.join("engines").join("oaiy-studio.json"), "{\"llm\":{\"default_model\":\"Stale\"}}").unwrap();
        let (ui, server) = fake_studio(json!([rtx("RTX 3060", 12288, 0)]), catalog(json!(16)), "").await;
        let v = recommendation_at(Some(ui), &dir.0).await;
        assert_eq!(v["local"]["chosen"], Value::Null, "{v}");
        assert_eq!(v["recommend"], "chatgpt");
        server.abort();
    }
}
