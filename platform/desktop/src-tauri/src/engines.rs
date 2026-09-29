//! The engines (language models, pictures, video, speech, music, sound, 3D),
//! run by OAIY Desktop itself: `oaiy-studio`'s supervisor, in this process (it
//! is std-only and runs on its own threads beside tokio), which starts
//! `oaiy-llm-server` and `oaiy-media` as they are needed. Its gateway serves
//! 8080 (the agent's models, Aokie's language-model fallback) and its control
//! pages 7860 (the Engines page of the window).
//!
//! When a studio already serves this configuration, the desktop uses that one.
//! `OAIY_ENGINES` decides whether the desktop may start them itself: `launch`
//! (the default in a release build), `attach` (only use a running studio: the
//! default in a debug build, which restarts on every rebuild and must not take
//! the model down with it), or `off`.
//!
//! The configuration is `<data>/engines/oaiy-studio.json`; the engine
//! programs are found beside this program (installed), in `OAIY_ENGINES_DIR`,
//! or in the repository's release build.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

static RUNNING: Mutex<Option<oaiy_studio::Running>> = Mutex::new(None);
static UI_URL: OnceLock<String> = OnceLock::new();

/// How the desktop treats the engines (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Launch,
    Attach,
    Off,
}

pub fn mode() -> Mode {
    match std::env::var("OAIY_ENGINES").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
        "launch" => Mode::Launch,
        "attach" => Mode::Attach,
        "off" | "0" | "false" => Mode::Off,
        _ if cfg!(debug_assertions) => Mode::Attach,
        _ => Mode::Launch,
    }
}

/// The configuration's path in the data folder.
pub fn config_path(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("oaiy-studio.json")
}

fn exe(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

/// Where the engine programs are.
pub fn programs_dir() -> Option<PathBuf> {
    let has = |d: &Path| d.join(exe("oaiy-llm-server")).is_file() || d.join(exe("oaiy-media")).is_file();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("OAIY_ENGINES_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join("engines"));
        candidates.push(dir);
    }
    // A build from the repository: platform/desktop/src-tauri → the workspace's release build.
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../target/release"));
    candidates.into_iter().find(|d| has(d)).map(|d| std::path::absolute(&d).unwrap_or(d))
}

/// Start the engines, or find them running. Returns their control pages' address.
pub fn start(data_dir: &Path) -> Result<String, String> {
    start_with(data_dir, mode())
}

fn start_with(data_dir: &Path, m: Mode) -> Result<String, String> {
    if m == Mode::Off {
        return Err("the engines are off (OAIY_ENGINES=off)".into());
    }
    let config = config_path(data_dir);
    if let Some(dir) = config.parent() {
        // The studio's configuration holds its gateway key and the Hugging Face token: where
        // this makes the folder it is owner-only (unix), and the studio saves the file the same.
        crate::secret_file::create_private_dir(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
    }
    if let Some(programs) = programs_dir() {
        if oaiy_studio::use_programs_from(&config, &programs)? {
            log::info!("engines: the programs are in {}", programs.display());
        }
    }
    let args = oaiy_studio::Args { config: config.clone(), open: Some("none".into()), ui_port: None, port: None, start_llm: false };
    if let Some(url) = oaiy_studio::running_instance(&args) {
        log::info!("engines: using the ones already running at {url}");
        crate::http::set_engines_ui(&url);
        let _ = UI_URL.set(url.clone());
        return Ok(url);
    }
    if m == Mode::Attach {
        return Err("no engines are running for this configuration, and this build only uses running ones (OAIY_ENGINES=attach)".into());
    }
    let running = oaiy_studio::launch(&args, true)?;
    let url = running.ui_url.clone();
    log::info!("engines: started (control pages {url}, gateway {})", running.gateway_url);
    crate::http::set_engines_ui(&url);
    let _ = UI_URL.set(url.clone());
    *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = Some(running);
    Ok(url)
}

/// Where the engines' control pages are, once found or started.
pub fn ui_url() -> Option<String> {
    UI_URL.get().cloned()
}

/// The engines were started by this desktop (and so stop with it), rather than found already running.
pub fn started_here() -> bool {
    RUNNING.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// What the engines are doing that a restart would end: (media jobs running, catalog model downloads running).
/// Asked at their control pages, so it covers a studio this desktop found running as well as its own. Nothing
/// running, or no answer within two seconds, is (0, 0). The answer is kept for three seconds: the window asks often.
pub fn activity() -> (usize, usize) {
    static LAST: Mutex<Option<(std::time::Instant, (usize, usize))>> = Mutex::new(None);
    if let Some((at, answer)) = *LAST.lock().unwrap_or_else(|e| e.into_inner()) {
        if at.elapsed() < std::time::Duration::from_secs(3) {
            return answer;
        }
    }
    let Some(ui) = ui_url() else { return (0, 0) };
    // On a thread of its own: a blocking client must not be made (or dropped) on an async thread.
    let asked = std::thread::spawn(move || {
        let get = |path: &str| -> Option<serde_json::Value> {
            reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(2)).build().ok()?.get(format!("{ui}{path}")).send().ok()?.json().ok()
        };
        Some(parse_activity(&get("/api/state")?, &get("/api/downloads")?))
    })
    .join()
    .ok()
    .flatten()
    .unwrap_or((0, 0));
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some((std::time::Instant::now(), asked));
    asked
}

/// The engines' `/api/state` (its media jobs) and `/api/downloads` (its catalog downloads) as (media jobs running, downloads running).
fn parse_activity(state: &serde_json::Value, downloads: &serde_json::Value) -> (usize, usize) {
    let finished = |job: &serde_json::Value| matches!(job.get("status").and_then(|s| s.as_str()), Some("completed" | "failed" | "cancelled"));
    let media = state.pointer("/media/jobs").and_then(|j| j.as_array()).map_or(0, |jobs| jobs.iter().filter(|job| !finished(job)).count());
    // `busy` says the same in one word: a running job with no list (an older studio) still counts once.
    let media = if media == 0 && state.pointer("/media/busy").and_then(|b| b.as_bool()).unwrap_or(false) { 1 } else { media };
    let running = downloads
        .get("models")
        .and_then(|m| m.as_array())
        .map_or(0, |models| models.iter().filter(|m| m.pointer("/download/status").and_then(|s| s.as_str()).is_some_and(|s| matches!(s, "queued" | "downloading" | "adding"))).count());
    (media, running)
}

/// Stop the engines this desktop started (a studio that was already running is left running).
pub fn stop() {
    if let Some(running) = RUNNING.lock().unwrap_or_else(|e| e.into_inner()).take() {
        log::info!("engines: stopping");
        running.studio.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_configuration_lives_in_the_data_folder() {
        assert_eq!(config_path(Path::new("D:/data")), Path::new("D:/data").join("engines").join("oaiy-studio.json"));
    }

    #[test]
    fn started_by_the_desktop_they_serve_and_stop() {
        // A configuration of its own on ports the system picks, so it cannot meet a running studio.
        let data = std::env::temp_dir().join(format!("oaiy-engines-{}", std::process::id()));
        let config = config_path(&data);
        std::fs::create_dir_all(config.parent().unwrap()).unwrap();
        oaiy_studio::use_programs_from(&config, &data).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
        v["ui"]["port"] = 0.into();
        v["gateway"]["port"] = 0.into();
        std::fs::write(&config, serde_json::to_string_pretty(&v).unwrap()).unwrap();

        let ui = start_with(&data, Mode::Launch).unwrap();
        assert!(ui.starts_with("http://127.0.0.1:") && !ui.ends_with(":0"), "{ui}");
        assert_eq!(ui_url().as_deref(), Some(ui.as_str()));
        // The control pages answer while it runs. Read as an HTTP client does, up
        // to the reply's own length: on Windows the studio may reset the socket
        // once it has answered, and reading to the end then fails a good reply.
        let resp = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap()
            .get(format!("{ui}/api/state"))
            .send()
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let reply = resp.text().unwrap();
        assert!(reply.contains("config_path"), "{}", &reply[..reply.len().min(200)]);
        stop();
        assert!(RUNNING.lock().unwrap().is_none());
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn what_the_engines_are_doing_is_read_from_their_state_and_downloads() {
        use serde_json::json;
        let idle = json!({"media": {"busy": false, "jobs": []}});
        let none = json!({"models": [{"id": "a", "download": null}, {"id": "b"}]});
        assert_eq!(parse_activity(&idle, &none), (0, 0));
        // Jobs the studio lists that have not finished.
        let busy = json!({"media": {"busy": true, "jobs": [{"id": "j1", "status": "running"}, {"id": "j2", "status": "completed"}, {"id": "j3", "status": "queued"}, {"id": "j4", "status": "failed"}, {"id": "j5", "status": "cancelled"}]}});
        assert_eq!(parse_activity(&busy, &none).0, 2);
        // busy with no list still counts once; not busy with a list of finished jobs counts none.
        assert_eq!(parse_activity(&json!({"media": {"busy": true}}), &none).0, 1);
        assert_eq!(parse_activity(&json!({"media": {"busy": false, "jobs": [{"status": "completed"}]}}), &none).0, 0);
        // Downloads waiting or running count; finished, paused, failed and cancelled do not.
        let downloads = json!({"models": [
            {"download": {"status": "queued"}}, {"download": {"status": "downloading"}}, {"download": {"status": "adding"}},
            {"download": {"status": "done"}}, {"download": {"status": "paused"}}, {"download": {"status": "failed"}}, {"download": {"status": "cancelled"}}
        ]});
        assert_eq!(parse_activity(&idle, &downloads).1, 3);
        // Whatever else they answer is nothing.
        assert_eq!(parse_activity(&json!(null), &json!("nope")), (0, 0));
    }

    #[test]
    fn a_debug_build_only_uses_running_engines_unless_told() {
        // (The variable is read at call time; the default depends on the build.)
        if std::env::var_os("OAIY_ENGINES").is_none() {
            assert_eq!(mode(), if cfg!(debug_assertions) { Mode::Attach } else { Mode::Launch });
        }
    }
}
