//! `oaiy-flows`: this desktop's flows as a connector, for callers that name commands rather than routes.
//!
//! A FormLogic automation run on this computer (its encrypted desktop relay, or a queued run) asks for things with
//! its `connector_request` node, which comes to `POST /api/bridge/connectors/:id/request`; so does a browser paired
//! with this OAIY, and any app that holds a token for it. Under this connector's id that route answers two commands
//! of its own, never a plugin's:
//!
//! * `flows.list`: the flows shared with apps, each `{flowId, name, description, inputs: [{name, default}]}`.
//! * `flow.run` `{flowId, inputs, timeoutSeconds?, pictures?, maxPictureBytes?}`: runs one of them as a run of the
//!   bridge's (it is in the run history like any other) and answers `{runId, status, output, error, pictures}`:
//!   `output` the flow's own answer, `pictures` the pictures it names with their bytes (see [`super::pictures`]),
//!   since its references are addresses on this computer that a caller elsewhere cannot read.
//!
//! Only a shared flow runs: one whose document says `"oaiyRemote": true`, which **Let apps run it** in the flow's
//! menu in Flows sets. A FormLogic automation is written on FormLogic's site, by whoever acts as the account's owner
//! there, and runs here; the flows it can start are the ones the person here chose, not every flow on the computer
//! (a flow can read files, drive a browser, run code).

use serde_json::{json, Value};
use std::time::{Duration, Instant};

use super::ledger::{ClaimOutcome, LedgerHandle, LineageRef, ReserveOutcome, RunRecord, RunRequest, Runtime};
use super::worker::{execute_run, FlowStore};
use super::pictures;

/// The connector's id.
pub const CONNECTOR: &str = "oaiy-flows";

/// Its commands.
pub const COMMANDS: [&str; 2] = ["flows.list", "flow.run"];

/// A run's budget when the caller gives none: an image flow's first run loads its models.
const DEFAULT_SECONDS: u64 = 300;

/// The pictures' base64 together, when the caller gives no budget: what fits in FormLogic's encrypted relay's
/// 192 KiB answer beside the flow's own.
pub const DEFAULT_PICTURE_BYTES: usize = 150_000;

/// The most a caller may ask for.
const MAX_PICTURE_BYTES: usize = 32 * 1024 * 1024;

/// A refusal: the HTTP status, the protocol's error code, and what to do.
pub type Refusal = (u16, &'static str, String);

/// Whether a flow's document shares it with apps.
pub fn shared(doc: &Value) -> bool {
    doc.get("oaiyRemote").and_then(Value::as_bool) == Some(true)
}

/// The inputs a flow takes from its caller: its inputs, each by its label (what the run's `inputs` are keyed by), with
/// the value it has when none is given and, for one that is not text, its kind (`file`: content, a picture as a data
/// URL or its address; `video`, `audio`: a path or an address; `folder`: a folder's path). Its file, video, audio and
/// folder inputs were left out: a run could not be given them (they took only their saved value), and now can.
pub fn inputs_of(doc: &Value) -> Vec<Value> {
    doc.get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|n| {
            let (kind, default_key) = match n.get("type").and_then(Value::as_str)? {
                "input_text" => (None, "value"),
                "input_file" => (Some("file"), "filePath"),
                "input_video" => (Some("video"), "filePath"),
                "input_audio" => (Some("audio"), "filePath"),
                "input_folder" => (Some("folder"), "path"),
                _ => return None,
            };
            let data = n.get("data");
            let name = data.and_then(|d| d.get("label")).and_then(Value::as_str).filter(|s| !s.trim().is_empty());
            let name = name.or_else(|| n.get("id").and_then(Value::as_str)).unwrap_or("input");
            // (a file input with no path saved has no default, not "")
            let saved = data.and_then(|d| d.get(default_key)).cloned().unwrap_or(Value::Null);
            let default = if kind.is_some() && saved.as_str() == Some("") { Value::Null } else { saved };
            let mut entry = json!({ "name": name, "default": default });
            if let Some(kind) = kind {
                entry["type"] = json!(kind);
            }
            Some(entry)
        })
        .collect()
}

fn document(flows: &FlowStore, id: &str) -> Option<Value> {
    flows.get(id).and_then(|text| serde_json::from_str(&text).ok())
}

/// The flows shared with apps.
pub fn list(flows: &FlowStore) -> Value {
    let shared: Vec<Value> = flows
        .list()
        .into_iter()
        .filter_map(|(id, _)| {
            let doc = document(flows, &id)?;
            shared(&doc).then(|| {
                json!({
                    "flowId": id,
                    "name": doc.get("name").and_then(Value::as_str).unwrap_or(&id),
                    "description": doc.get("description").and_then(Value::as_str).unwrap_or(""),
                    "inputs": inputs_of(&doc),
                })
            })
        })
        .collect();
    json!({ "flows": shared })
}

/// A command of this connector, answered.
pub fn answer(
    ledger: &LedgerHandle,
    flows: &FlowStore,
    node: Option<&crate::services::node_runtime::NodeHandle>,
    command: &str,
    payload: &Value,
    key: Option<&str>,
) -> Result<Value, Refusal> {
    match command {
        "flows.list" => Ok(list(flows)),
        "flow.run" => run(ledger, flows, node, payload, key, &pictures::fetch_local),
        other => Err((
            404,
            "capability_unavailable",
            format!("The {CONNECTOR} connector has no command {other:?}: it answers {}.", COMMANDS.join(" and ")),
        )),
    }
}

/// `flow.run`: the flow run (or, for a key it has seen, the run it started then), answered with its pictures.
pub fn run(
    ledger: &LedgerHandle,
    flows: &FlowStore,
    node: Option<&crate::services::node_runtime::NodeHandle>,
    payload: &Value,
    key: Option<&str>,
    fetch: &dyn Fn(&str) -> Result<Vec<u8>, String>,
) -> Result<Value, Refusal> {
    let bad = |message: String| (400, "invalid_request", message);
    let id = payload
        .get("flowId")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| bad("flow.run needs a flowId: flows.list names the flows shared with apps.".into()))?;
    if !FlowStore::valid_id(id) {
        return Err(bad(format!("{id:?} is not a flow id (letters, digits, - and _).")));
    }
    let doc = document(flows, id).ok_or_else(|| (404, "flow_not_found", format!("There is no flow {id:?} on this computer.")))?;
    if !shared(&doc) {
        return Err((
            403,
            "capability_denied",
            format!("The flow {id:?} is not shared with apps: in OAIY's Flows, choose Let apps run it in its menu."),
        ));
    }
    let inputs = payload.get("inputs").or_else(|| payload.get("input")).cloned().unwrap_or_else(|| json!({}));
    if !inputs.is_object() {
        return Err(bad("inputs must be an object, each input by its name (flows.list names them).".into()));
    }
    let seconds = payload.get("timeoutSeconds").and_then(Value::as_u64).unwrap_or(DEFAULT_SECONDS).clamp(1, 600);
    let budget = match payload.get("maxPictureBytes") {
        None | Some(Value::Null) => DEFAULT_PICTURE_BYTES,
        Some(v) => v.as_u64().map(|n| (n as usize).min(MAX_PICTURE_BYTES)).ok_or_else(|| bad("maxPictureBytes must be a number of bytes.".into()))?,
    };
    let with_pictures = match payload.get("pictures").and_then(Value::as_str) {
        None | Some("inline") => true,
        Some("none") => false,
        Some(other) => return Err(bad(format!("pictures is \"inline\" or \"none\", not {other:?}."))),
    };

    // One key for one asking: the caller's (a FormLogic node's is made from its run, node and payload, so a
    // redelivery is this same run), else one of its own.
    let key = key.filter(|k| !k.trim().is_empty()).map(str::to_string).unwrap_or_else(|| format!("oaiy-flows:{}", uuid::Uuid::new_v4().simple()));
    let request = RunRequest {
        caller_product: CONNECTOR.into(),
        flow_id: Some(id.to_string()),
        inline_graph: false,
        // (its pictures stored with the engines first, the run keeping their references)
        input: Some(pictures::referenced(inputs)),
        timeout_ms: Some(seconds * 1000),
        // Reserved for this connector to claim: the worker leaves `queued` runs alone.
        mode: "queued".into(),
        correlation_id: key.clone(),
        idempotency_key: format!("{CONNECTOR}:{key}"),
        lineage: LineageRef::default(),
        trigger_event: None,
        caller_label: None,
    };
    let reserved = {
        let mut l = ledger.lock().map_err(|_| (500, "internal", "the run ledger is unavailable".to_string()))?;
        l.reserve(&request)
    };
    let record = match reserved {
        ReserveOutcome::Reserved(rec) => {
            let claimed = {
                let mut l = ledger.lock().map_err(|_| (500, "internal", "the run ledger is unavailable".to_string()))?;
                l.claim(&rec.run_id, Runtime::Desktop, CONNECTOR)
            };
            match claimed {
                ClaimOutcome::Claimed(rec) => {
                    let (status, output, error) = execute_run(flows, node, ledger, &rec);
                    let mut l = ledger.lock().map_err(|_| (500, "internal", "the run ledger is unavailable".to_string()))?;
                    l.finish(&rec.run_id, status, output, error).map_err(|e| (500, "internal", e))?
                }
                _ => settled(ledger, &rec.run_id, seconds)?,
            }
        }
        ReserveOutcome::Duplicate(rec) => settled(ledger, &rec.run_id, seconds)?,
        ReserveOutcome::Refused { reason } => return Err((422, "invalid_request", format!("refused by a loop guard: {reason}"))),
    };
    Ok(reply(&record, with_pictures.then_some(budget), fetch))
}

/// A run another took on (or one this key started before), once it has ended, or as it is when `seconds` pass.
fn settled(ledger: &LedgerHandle, run_id: &str, seconds: u64) -> Result<RunRecord, Refusal> {
    let deadline = Instant::now() + Duration::from_secs(seconds + 5);
    loop {
        let rec = ledger
            .lock()
            .map_err(|_| (500, "internal", "the run ledger is unavailable".to_string()))?
            .get(run_id)
            .ok_or_else(|| (500, "internal", format!("the run {run_id} is gone")))?;
        if rec.status.is_terminal() || Instant::now() >= deadline {
            return Ok(rec);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// What the caller is answered: the run, the flow's own answer (the output node's value, inside the CLI's report),
/// and its pictures within `budget` (none asked: none).
fn reply(record: &RunRecord, budget: Option<usize>, fetch: &dyn Fn(&str) -> Result<Vec<u8>, String>) -> Value {
    let report = record.output.clone().unwrap_or(Value::Null);
    let output = match report.get("output") {
        Some(own) => own.clone(),
        None => report,
    };
    let pictures = budget.map(|b| pictures::gather(&output, b, fetch)).unwrap_or_default();
    json!({
        "runId": record.run_id,
        "status": record.status,
        "output": output,
        "error": record.error,
        "pictures": pictures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::ledger::{new_handle, RunStatus};

    fn store(name: &str) -> FlowStore {
        let dir = std::env::temp_dir().join(format!("oaiy-flow-connector-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        FlowStore::new(dir)
    }

    const FLOW: &str = r#"{"name": "Pictures", "description": "Makes one", "nodes": [
        {"id": "p", "type": "input_text", "data": {"label": "Prompt", "value": "a cat"}},
        {"id": "e", "type": "input_text", "data": {"label": "", "value": ""}},
        {"id": "x", "type": "output", "data": {}}], "edges": []}"#;

    /// Only the flows the person shared are listed or run, each with the inputs it takes.
    #[test]
    fn only_a_shared_flow_is_listed_or_run() {
        let flows = store("shared");
        flows.put("mine", FLOW).unwrap();
        let mut doc: Value = serde_json::from_str(FLOW).unwrap();
        doc["oaiyRemote"] = json!(true);
        flows.put("apps", &doc.to_string()).unwrap();
        let listed = list(&flows);
        assert_eq!(listed["flows"].as_array().unwrap().len(), 1);
        assert_eq!(listed["flows"][0]["flowId"], "apps");
        assert_eq!(listed["flows"][0]["inputs"], json!([{"name": "Prompt", "default": "a cat"}, {"name": "e", "default": ""}]));

        let l = new_handle();
        let never = |_: &str| -> Result<Vec<u8>, String> { panic!("nothing is read for a refused run") };
        let refused = run(&l, &flows, None, &json!({"flowId": "mine"}), Some("k1"), &never).unwrap_err();
        assert_eq!((refused.0, refused.1), (403, "capability_denied"));
        assert_eq!(run(&l, &flows, None, &json!({"flowId": "nope"}), Some("k2"), &never).unwrap_err().1, "flow_not_found");
        assert_eq!(run(&l, &flows, None, &json!({"flowId": "../x"}), Some("k3"), &never).unwrap_err().1, "invalid_request");
        assert_eq!(run(&l, &flows, None, &json!({"flowId": "apps", "inputs": [1]}), Some("k4"), &never).unwrap_err().1, "invalid_request");
        assert!(l.lock().unwrap().recent(10, &[]).is_empty(), "a refused run is never reserved");
        assert_eq!(answer(&l, &flows, None, "flow.delete", &json!({}), None).unwrap_err().1, "capability_unavailable");
    }

    /// A flow's file, video, audio and folder inputs are listed too, each with its kind and the path it has saved.
    #[test]
    fn every_kind_of_input_is_listed() {
        let doc = json!({ "nodes": [
            {"id": "t", "type": "input_text", "data": {"label": "Prompt", "value": "a cat"}},
            {"id": "f", "type": "input_file", "data": {"label": "File", "filePath": ""}},
            {"id": "v", "type": "input_video", "data": {"label": "Clip", "filePath": "/v.mp4"}},
            {"id": "a", "type": "input_audio", "data": {"filePath": "/a.wav"}},
            {"id": "d", "type": "input_folder", "data": {"path": "/notes"}},
            {"id": "x", "type": "template", "data": {}}] });
        assert_eq!(inputs_of(&doc), json!([
            {"name": "Prompt", "default": "a cat"},
            {"name": "File", "default": null, "type": "file"},
            {"name": "Clip", "default": "/v.mp4", "type": "video"},
            {"name": "a", "default": "/a.wav", "type": "audio"},
            {"name": "d", "default": "/notes", "type": "folder"},
        ]).as_array().unwrap().clone());
    }

    /// The answer is the flow's own (the output node's, inside the CLI's report), with its pictures' bytes.
    #[test]
    fn the_answer_is_the_flows_own_with_its_pictures() {
        let picture = "http://127.0.0.1:9/api/ai/engine/gateway/files/images/a.png";
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(4, 4).write_to(&mut png, image::ImageFormat::Png).unwrap();
        let png = png.into_inner();
        let l = new_handle();
        let request = RunRequest {
            caller_product: CONNECTOR.into(),
            flow_id: Some("apps".into()),
            inline_graph: false,
            input: None,
            timeout_ms: None,
            mode: "queued".into(),
            correlation_id: "c".into(),
            idempotency_key: "k".into(),
            lineage: LineageRef::default(),
            trigger_event: None,
            caller_label: None,
        };
        let ReserveOutcome::Reserved(rec) = l.lock().unwrap().reserve(&request) else { panic!("not reserved") };
        let ClaimOutcome::Claimed(rec) = l.lock().unwrap().claim(&rec.run_id, Runtime::Desktop, CONNECTOR) else { panic!("not claimed") };
        let output = json!({"success": true, "output": {"image": picture, "review": {"verdict": "pass"}}});
        let record = l.lock().unwrap().finish(&rec.run_id, RunStatus::Succeeded, Some(output), None).unwrap();
        let fetch = |r: &str| if r == picture { Ok(png.clone()) } else { Err("no".to_string()) };
        let got = reply(&record, Some(DEFAULT_PICTURE_BYTES), &fetch);
        assert_eq!(got["output"]["review"]["verdict"], "pass");
        assert_eq!(got["pictures"][0]["ref"], picture);
        assert_eq!(got["pictures"][0]["mimeType"], "image/png");
        assert_eq!(got["pictures"][0]["width"], 4);
        assert!(reply(&record, None, &fetch)["pictures"].as_array().unwrap().is_empty());
    }
}
