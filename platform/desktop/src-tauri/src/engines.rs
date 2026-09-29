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
        std::fs::create_dir_all(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
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
    fn a_debug_build_only_uses_running_engines_unless_told() {
        // (The variable is read at call time; the default depends on the build.)
        if std::env::var_os("OAIY_ENGINES").is_none() {
            assert_eq!(mode(), if cfg!(debug_assertions) { Mode::Attach } else { Mode::Launch });
        }
    }
}
