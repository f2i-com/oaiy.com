//! OAIY's own age model, for an age check a site sends over the AI lane (`verify: {task: "face-age", images}` on a
//! sealed chat turn): the turn's pictures it names are read by the media worker's `face-age` job, YuNet finding the
//! faces and MiVOLO v2 reading the age of the one face there should be. The worker loads both models for the job and
//! ends with it, so they are loaded for the check and gone before the language model answers the turn.
//!
//! What goes back is each picture's face count and, for one face, its age, and nothing else; no picture is written,
//! kept or logged here. The models are the files the engines' settings name (`media.faceage.model` and `.detector`,
//! which adding the Age check on the Models page fills); when they are not there the answer says so, and the turn
//! is answered by the language model as before.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const TASK: &str = "face-age";
/// As many pictures as the worker takes in a job.
const MAX_PICTURES: usize = 16;
/// Loading both models and reading a few pictures takes seconds on a CPU; past this the check did not happen.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The worker and the two model files.
#[derive(Debug, PartialEq)]
pub struct Files {
    pub worker: PathBuf,
    pub model: PathBuf,
    pub detector: PathBuf,
}

/// The engines' settings, as the Engines page saves them.
fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("oaiy-studio.json")
}

fn resolve(root: &Path, value: &str) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { root.join(p) }
}

/// The media worker: a path the settings give, else the program beside this one (in its `engines` folder or with
/// it), in `OAIY_ENGINES_DIR`, or beside the settings, as the engines find it.
fn worker(root: &Path, named: &str) -> Option<PathBuf> {
    if named.contains(['/', '\\']) {
        return Some(resolve(root, named)).filter(|p| p.is_file());
    }
    let file = if cfg!(windows) && !named.to_ascii_lowercase().ends_with(".exe") { format!("{named}.exe") } else { named.to_string() };
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("OAIY_ENGINES_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(dir.join("engines"));
        dirs.push(dir);
    }
    // The engines an installer carries (the engines write this path into the settings when they start; a studio
    // that was already running when this one started may not have).
    #[cfg(feature = "gui")]
    if let Some(dir) = crate::engines::bundled() {
        dirs.push(dir.to_path_buf());
    }
    dirs.push(root.to_path_buf());
    dirs.into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
}

/// Where the age check's files are, or why it cannot run here.
pub fn files(data_dir: &Path) -> Result<Files, String> {
    let path = settings_path(data_dir);
    let root = path.parent().unwrap_or(data_dir).to_path_buf();
    let settings: Value = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    let media = settings.get("media");
    let section = media.and_then(|m| m.get("faceage"));
    if section.and_then(|s| s.get("enabled")).and_then(Value::as_bool) == Some(false) {
        return Err("the age check is turned off in this OAIY's engine settings".into());
    }
    let file = |key: &str| {
        section
            .and_then(|s| s.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| resolve(&root, s))
            .filter(|p| p.is_file())
    };
    let missing = "this OAIY has no age model: add the Age check (MiVOLO v2 and YuNet) on its Models page";
    let model = file("model").ok_or(missing)?;
    let detector = file("detector").ok_or(missing)?;
    let named = media.and_then(|m| m.get("worker")).and_then(Value::as_str).unwrap_or("oaiy-media");
    let worker = worker(&root, named).ok_or("this OAIY has no media engine (oaiy-media) to run the age check")?;
    Ok(Files { worker, model, detector })
}

/// The pictures the check names: the turn's image parts, numbered from 0 in the order they come, as `data:` URLs
/// (a picture at an address is never fetched).
fn pictures(ask: &Value, messages: &[Value]) -> Result<Vec<String>, String> {
    let all: Vec<&str> = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
        .filter_map(|p| p.get("image_url").and_then(|u| u.get("url").or(Some(u))).and_then(Value::as_str))
        .collect();
    let wanted = ask.get("images").and_then(Value::as_array).ok_or("the age check names no pictures")?;
    if wanted.is_empty() || wanted.len() > MAX_PICTURES {
        return Err(format!("the age check takes 1 to {MAX_PICTURES} pictures"));
    }
    wanted
        .iter()
        .map(|i| {
            let i = i.as_u64().ok_or("a picture named by other than its number")? as usize;
            let url = all.get(i).ok_or(format!("the turn has no picture {i}"))?;
            if !url.starts_with("data:image/") {
                return Err("a picture that is not in the turn itself".to_string());
            }
            Ok(url.to_string())
        })
        .collect()
}

/// Run the worker's job and read its answer.
async fn run(files: &Files, pictures: Vec<String>) -> Result<Value, String> {
    use crate::HiddenCommand as _;
    use tokio::io::AsyncWriteExt as _;
    let request = serde_json::to_vec(&json!({
        "kind": TASK,
        "detector": files.detector,
        "model": files.model,
        "images": pictures,
    }))
    .map_err(|e| e.to_string())?;
    let mut std_cmd = std::process::Command::new(&files.worker);
    std_cmd.arg("--stdin").pipe_hidden();
    let mut cmd = tokio::process::Command::from(std_cmd);
    cmd.stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|e| format!("the media engine did not start: {e}"))?;
    let mut stdin = child.stdin.take().ok_or("the media engine took no input")?;
    stdin.write_all(&request).await.map_err(|e| format!("the media engine did not take the pictures: {e}"))?;
    drop(stdin);
    let out = tokio::time::timeout(TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| "the age check took too long".to_string())?
        .map_err(|e| format!("the media engine stopped: {e}"))?;
    let last_json = |bytes: &[u8]| -> Option<Value> {
        String::from_utf8_lossy(bytes).lines().rev().find_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
    };
    if !out.status.success() {
        let said = last_json(&out.stderr).and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string));
        return Err(said.unwrap_or_else(|| format!("the media engine ended with {}", out.status)));
    }
    let answer = last_json(&out.stdout).ok_or("the media engine's answer could not be read")?;
    let images: Vec<Value> = answer
        .get("images")
        .and_then(Value::as_array)
        .ok_or("the media engine's answer has no pictures")?
        .iter()
        .map(reading)
        .collect();
    Ok(json!({
        "task": TASK,
        "model": answer.get("model").cloned().unwrap_or(Value::Null),
        "detector": answer.get("detector").cloned().unwrap_or(Value::Null),
        "images": images,
    }))
}

/// One picture's reading as the worker gave it, held to what the check sends on: the face count, and for one face its
/// age and which way it points (yaw and pitch in degrees, from the detector's landmarks).
fn reading(i: &Value) -> Value {
    let mut one = json!({ "faces": i.get("faces").and_then(Value::as_u64).unwrap_or(0) });
    if let Some(age) = i.get("age").and_then(Value::as_f64).filter(|a| a.is_finite()) {
        one["age"] = json!(age);
    }
    for key in ["yaw", "pitch"] {
        if let Some(angle) = i.get(key).and_then(Value::as_f64).filter(|a| a.is_finite() && a.abs() <= 90.) {
            one[key] = json!(angle);
        }
    }
    one
}

/// The answer to a turn's `verify`, for its final frame: the readings, or why there are none.
pub async fn verify(data_dir: &Path, ask: &Value, messages: &[Value]) -> Value {
    let task = ask.get("task").and_then(Value::as_str).unwrap_or("");
    if task != TASK {
        return json!({ "task": task, "error": "unknown_task", "message": format!("this OAIY does not check {task:?}") });
    }
    let pictures = match pictures(ask, messages) {
        Ok(p) => p,
        Err(e) => return json!({ "task": TASK, "error": "invalid", "message": e }),
    };
    let files = match files(data_dir) {
        Ok(f) => f,
        Err(e) => return json!({ "task": TASK, "error": "not_installed", "message": e }),
    };
    match run(&files, pictures).await {
        Ok(v) => v,
        Err(e) => {
            log::warn!("age check: {e}");
            json!({ "task": TASK, "error": "failed", "message": e })
        }
    }
}

/// `frame` (a final frame's JSON) with `verification` in it.
pub fn with_verification(frame: Vec<u8>, verification: Value) -> Vec<u8> {
    match serde_json::from_slice::<Value>(&frame) {
        Ok(Value::Object(mut fields)) => {
            fields.insert("verification".into(), verification);
            serde_json::to_vec(&Value::Object(fields)).unwrap_or(frame)
        }
        _ => frame,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oaiy-faceage-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("engines")).unwrap();
        d
    }

    #[test]
    fn the_pictures_are_the_turns_own_by_number() {
        let messages = vec![
            json!({ "role": "system", "content": "describe" }),
            json!({ "role": "user", "content": [
                { "type": "text", "text": "look" },
                { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,AAAA" } },
                { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,BBBB" } },
                { "type": "image_url", "image_url": { "url": "https://example.com/c.jpg" } },
            ]}),
        ];
        assert_eq!(pictures(&json!({ "images": [1, 0] }), &messages).unwrap(), ["data:image/jpeg;base64,BBBB", "data:image/jpeg;base64,AAAA"]);
        assert!(pictures(&json!({ "images": [2] }), &messages).unwrap_err().contains("not in the turn"), "never fetched");
        assert!(pictures(&json!({ "images": [3] }), &messages).unwrap_err().contains("no picture 3"));
        assert!(pictures(&json!({ "images": [] }), &messages).is_err());
        assert!(pictures(&json!({}), &messages).is_err());
    }

    #[test]
    fn a_reading_keeps_the_count_the_age_and_the_angles_and_nothing_else() {
        let r = reading(&json!({ "faces": 1, "age": 31.5, "yaw": -12.3, "pitch": 4.0, "gender": "x", "box": [1, 2, 3, 4] }));
        assert_eq!(r, json!({ "faces": 1, "age": 31.5, "yaw": -12.3, "pitch": 4.0 }));
        assert_eq!(reading(&json!({ "faces": 2, "yaw": 400.0 })), json!({ "faces": 2 }), "an angle no head turns is dropped");
    }

    #[test]
    fn without_its_models_the_check_says_so() {
        let d = dir("none");
        assert!(files(&d).unwrap_err().contains("no age model"));
        std::fs::write(settings_path(&d), r#"{"media":{"faceage":{"enabled":false}}}"#).unwrap();
        assert!(files(&d).unwrap_err().contains("turned off"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_settings_name_the_files_beside_them_or_anywhere() {
        let d = dir("some");
        let engines = d.join("engines");
        std::fs::create_dir_all(engines.join("MiVOLO-v2")).unwrap();
        std::fs::write(engines.join("MiVOLO-v2").join("model.safetensors"), b"x").unwrap();
        let detector = d.join("yunet.onnx");
        std::fs::write(&detector, b"x").unwrap();
        let program = d.join(if cfg!(windows) { "worker.exe" } else { "worker" });
        std::fs::write(&program, b"x").unwrap();
        std::fs::write(
            settings_path(&d),
            serde_json::to_vec(&json!({ "media": { "worker": program, "faceage": { "model": "MiVOLO-v2/model.safetensors", "detector": detector } } })).unwrap(),
        )
        .unwrap();
        let f = files(&d).unwrap();
        assert_eq!(f, Files { worker: program, model: engines.join("MiVOLO-v2").join("model.safetensors"), detector });
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn an_unknown_task_or_no_model_is_an_answer_not_a_failed_turn() {
        let d = dir("answer");
        let v = verify(&d, &json!({ "task": "palm-reading", "images": [0] }), &[]).await;
        assert_eq!(v["error"], "unknown_task");
        let messages = vec![json!({ "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,AAAA" } }] })];
        let v = verify(&d, &json!({ "task": TASK, "images": [0] }), &messages).await;
        assert_eq!(v["error"], "not_installed");
        let frame = with_verification(br#"{"v":1,"type":"final","kind":"final"}"#.to_vec(), v);
        let frame: Value = serde_json::from_slice(&frame).unwrap();
        assert_eq!((frame["kind"].as_str(), frame["verification"]["task"].as_str()), (Some("final"), Some(TASK)));
        let _ = std::fs::remove_dir_all(&d);
    }
}
