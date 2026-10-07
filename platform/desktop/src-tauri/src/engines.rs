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
static BUNDLED: OnceLock<PathBuf> = OnceLock::new();

/// The engines an installer carries, `<resources>/resources/engines`: the portable language-model server
/// (`oaiy-llm-server-webgpu`, GGUF models on any graphics card through WebGPU, else the CPU), so an installed OAIY
/// runs a model with nothing else to install. The app sets it from its resource folder before the engines start.
pub fn set_bundled(dir: PathBuf) {
    let _ = BUNDLED.set(dir);
}

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
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("OAIY_ENGINES_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join("engines"));
        candidates.push(dir);
    }
    // The engine the installer carries: after one put beside the program (a build of one's own), which is still the
    // one used when it is there.
    if let Some(dir) = BUNDLED.get() {
        candidates.push(dir.clone());
    }
    // A build from the repository: platform/desktop/src-tauri → the workspace's release build.
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../target/release"));
    first_with_programs(candidates)
}

/// The first of `candidates` that holds an engine program: the language-model server, under either of its names, or
/// the media worker.
fn first_with_programs(candidates: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    let has = |d: &Path| ["oaiy-llm-server", "oaiy-llm-server-webgpu", "oaiy-media"].iter().any(|p| d.join(exe(p)).is_file());
    candidates.into_iter().find(|d| has(d)).map(|d| std::path::absolute(&d).unwrap_or(d))
}

/// Where an older OAIY Desktop kept the NVIDIA engine it downloaded: the CUDA build of the language-model server,
/// one folder a version.
fn old_cuda_engines(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("cuda")
}

/// There is one engine now, and it runs on any GPU. A studio that an older OAIY Desktop pointed at its NVIDIA engine
/// (`llm.server` under `<data>/engines/cuda/`) is taken off it, back to the default, which this install's engine
/// answers to; and that folder is removed. Left alone, an upgraded install would go on running an older release's
/// program with this release's command line.
fn retire_cuda_engine(config: &Path, data_dir: &Path) -> Result<(), String> {
    let root = old_cuda_engines(data_dir);
    let server = oaiy_studio::llm_server(config)?;
    if Path::new(&server).starts_with(&root) {
        oaiy_studio::set_llm_server(config, "oaiy-llm-server")?;
        log::info!("engines: the language-model server was an older version's NVIDIA engine ({server}); it is this install's engine again");
    }
    if root.is_dir() {
        std::fs::remove_dir_all(&root).map_err(|e| format!("could not remove the older NVIDIA engine in {}: {e}", root.display()))?;
        log::info!("engines: removed the older NVIDIA engine, {}", root.display());
    }
    Ok(())
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
    // The NVIDIA engine an older version fetched (the CUDA build, which is no more): set aside before the studio starts.
    if let Err(e) = retire_cuda_engine(&config, data_dir) {
        log::warn!("engines: {e}");
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

/// How long the engines have to answer a question about what they are doing.
const ACTIVITY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
/// How long the engines have to accept a connection, on its own clock: Windows takes a second or two to report a REFUSED connection (it
/// tries again before it gives up), and a studio that is gone must not be mistaken for one that is listening and not answering.
const ACTIVITY_CONNECT: std::time::Duration = std::time::Duration::from_secs(4);
/// How long an answer is kept for the status a window polls (an install always asks again), and how long "nothing is listening" is (it
/// costs the connection wait above every time it is asked).
const ACTIVITY_KEEP: std::time::Duration = std::time::Duration::from_secs(3);
const ACTIVITY_KEEP_GONE: std::time::Duration = std::time::Duration::from_secs(30);

/// When it was asked, how long it may be kept, and what was answered.
type Kept = Mutex<Option<(std::time::Instant, std::time::Duration, Result<(usize, usize), String>)>>;

/// What the engines are doing that a restart would end: (media jobs running, catalog model downloads running), or why that cannot be said.
///
/// Asked at their control pages, so it covers a studio this desktop found running as well as its own.
///
/// - Not started, or nothing listening at the address they were found at (the connection is refused: the studio is gone): `Ok((0, 0))`.
///   There is nothing there to interrupt.
/// - An answer OAIY can read: what it says.
/// - Anything else: `Err(why)`. It accepts no connection, gives no answer within two seconds, answers with an error status, or with
///   something that is not the state a studio gives. That is NOT idle: a studio that is busy enough not to answer is the one whose
///   work a restart would throw away, so an install waits.
///
/// An answer is kept for three seconds (the window asks often); `fresh` asks now, whatever was answered a moment ago. An install asks
/// with `fresh`.
pub fn activity(fresh: bool) -> Result<(usize, usize), String> {
    static LAST: Kept = Mutex::new(None);
    activity_at(ui_url().as_deref(), fresh, &LAST, ACTIVITY_TIMEOUT, ACTIVITY_CONNECT)
}

fn activity_at(ui: Option<&str>, fresh: bool, kept: &Kept, timeout: std::time::Duration, connect: std::time::Duration) -> Result<(usize, usize), String> {
    if !fresh {
        if let Some((at, keep, answer)) = kept.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            if at.elapsed() < *keep {
                return answer.clone();
            }
        }
    }
    let Some(ui) = ui else { return Ok((0, 0)) };
    let ui = ui.to_string();
    // On a thread of its own: a blocking client must not be made (or dropped) on an async thread.
    let (answer, gone) = std::thread::spawn(move || ask_engines(&ui, timeout, connect)).join().unwrap_or_else(|_| (Err("the question could not be asked".to_string()), false));
    let keep = if gone { ACTIVITY_KEEP_GONE } else { ACTIVITY_KEEP };
    *kept.lock().unwrap_or_else(|e| e.into_inner()) = Some((std::time::Instant::now(), keep, answer.clone()));
    answer
}

/// The answer, and whether it is because nothing listens there.
fn ask_engines(ui: &str, timeout: std::time::Duration, connect: std::time::Duration) -> (Result<(usize, usize), String>, bool) {
    // First: is anything listening? On its own clock (see ACTIVITY_CONNECT). Refused: it is gone, there is nothing to interrupt.
    match listening(ui, connect) {
        Ok(true) => {}
        Ok(false) => return (Ok((0, 0)), true),
        Err(why) => return (Err(why), false),
    }
    let client = match reqwest::blocking::Client::builder().timeout(timeout).build() {
        Ok(client) => client,
        Err(e) => return (Err(format!("the question could not be asked ({e})")), false),
    };
    let get = |path: &str| -> Result<serde_json::Value, String> {
        match client.get(format!("{ui}{path}")).send() {
            Ok(response) if response.status().is_success() => response.json().map_err(|_| format!("{path} did not answer with anything readable")),
            Ok(response) => Err(format!("{path} answered {}", response.status())),
            Err(e) if e.is_timeout() => Err(format!("no answer within {} s", timeout.as_secs_f32().max(0.1))),
            Err(_) => Err(format!("{path} could not be read")),
        }
    };
    let answer = get("/api/state").and_then(|state| get("/api/downloads").and_then(|downloads| parse_activity(&state, &downloads)));
    (answer, false)
}

/// Whether anything listens at `ui`: Ok(true) it accepts a connection, Ok(false) it is refused, Err(why) neither (no answer to the
/// connection within `wait`, or the address is not one).
fn listening(ui: &str, wait: std::time::Duration) -> Result<bool, String> {
    use std::net::ToSocketAddrs;
    let url = url::Url::parse(ui).map_err(|_| "its address is not one".to_string())?;
    let (Some(host), Some(port)) = (url.host_str(), url.port_or_known_default()) else { return Err("its address has no host or port".to_string()) };
    let addresses: Vec<std::net::SocketAddr> = (host, port).to_socket_addrs().map_err(|_| "its address could not be found".to_string())?.collect();
    let mut last = None;
    for address in &addresses {
        match std::net::TcpStream::connect_timeout(address, wait) {
            Ok(_) => return Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => return Ok(false),
            Err(e) => last = Some(e),
        }
    }
    match last {
        Some(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) => Err(format!("it accepted no connection within {} s", wait.as_secs_f32())),
        Some(e) => Err(format!("it could not be reached ({e})")),
        None => Err("its address could not be found".to_string()),
    }
}

/// The engines' `/api/state` (its media jobs) and `/api/downloads` (its catalog downloads) as (media jobs running, downloads running), or why
/// they cannot be read: a state with no media section, or downloads with no list, is not "nothing is running".
fn parse_activity(state: &serde_json::Value, downloads: &serde_json::Value) -> Result<(usize, usize), String> {
    let finished = |job: &serde_json::Value| matches!(job.get("status").and_then(|s| s.as_str()), Some("completed" | "failed" | "cancelled"));
    let Some(media_section) = state.get("media").filter(|m| m.is_object()) else { return Err("its state has no media section".to_string()) };
    let jobs = media_section.get("jobs").and_then(|j| j.as_array());
    let busy = media_section.get("busy").and_then(|b| b.as_bool());
    if jobs.is_none() && busy.is_none() {
        return Err("its media section says neither what is running nor whether it is busy".to_string());
    }
    let media = jobs.map_or(0, |jobs| jobs.iter().filter(|job| !finished(job)).count());
    // `busy` says the same in one word: a running job with no list (an older studio) still counts once.
    let media = if media == 0 && busy.unwrap_or(false) { 1 } else { media };
    let Some(models) = downloads.get("models").and_then(|m| m.as_array()) else { return Err("its downloads have no list".to_string()) };
    let running = models.iter().filter(|m| m.pointer("/download/status").and_then(|s| s.as_str()).is_some_and(|s| matches!(s, "queued" | "downloading" | "adding"))).count();
    Ok((media, running))
}

/// GGUF architectures the language-model engine loads (oaiy-llm-server's `load_gguf`): Qwen3.5, the dense families it
/// loads whole, and the MoE families whose experts it streams. A file of another is refused when it is added, rather
/// than chosen and failing on the first message.
const ENGINE_RUNS: [&str; 11] = ["qwen35", "llama", "mistral", "qwen2", "qwen3", "gemma3", "gemma3n", "gemma4", "qwen3moe", "qwen3vlmoe", "glm5next"];
/// Of those, the ones whose tool calls the engine reads (the Qwen3.5 and GLM formats): the Agent's tools need one.
/// The others chat.
const ENGINE_TOOLS: [&str; 2] = ["qwen35", "glm5next"];
/// The tensor types the engine decodes (`ggml_quants::is_supported`): the K-quants, the legacy ones, IQ4 and the
/// floats. The other IQ types (IQ1, IQ2, IQ3), the ternary ones and MXFP4 fail when the model loads.
const ENGINE_TYPES: [&str; 15] = ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "IQ4_XS", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"];

/// A language model file the person named, as typed or pasted (Windows' "Copy as path" quotes it): a whole path to a
/// `.gguf` file that is there; of a model split into files, the first.
pub fn model_file(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    let unquoted = ['"', '\''].iter().find_map(|q| trimmed.strip_prefix(*q).and_then(|s| s.strip_suffix(*q))).unwrap_or(trimmed).trim();
    if unquoted.is_empty() {
        return Err("Choose a model file first.".into());
    }
    let path = PathBuf::from(unquoted);
    if !path.is_absolute() {
        return Err(format!("{unquoted} is not a whole path: choose the file, or paste its full path."));
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !name.to_ascii_lowercase().ends_with(".gguf") {
        return Err(format!("{name} is not a GGUF file: the engine runs language models from a .gguf file."));
    }
    if let Some((part, of)) = split_part(&name) {
        if part != 1 {
            return Err(format!("{name} is part {part} of {of} of a model split into files: choose the first part, the one ending -00001-of-{of:05}.gguf."));
        }
    }
    if !path.is_file() {
        return Err(format!("{} is not there.", path.display()));
    }
    Ok(path)
}

/// `…-00002-of-00004.gguf`: the part and the count of a model split into files, as llama.cpp names them.
fn split_part(name: &str) -> Option<(u32, u32)> {
    let stem = name.get(..name.len().checked_sub(".gguf".len())?)?;
    let (rest, of) = stem.rsplit_once("-of-")?;
    let (_, part) = rest.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(part) || !digits(of) {
        return None;
    }
    Some((part.parse().ok()?, of.parse().ok()?))
}

/// What the engines detected in a file (`/api/detect`), checked: a language model, not a part of one or a picture
/// model, of an architecture the engine loads. Its architecture, and whether the Agent can use its tools with it.
fn runnable(detected: &serde_json::Value, name: &str) -> Result<(String, bool), String> {
    let text = |key: &str| detected.get(key).and_then(|v| v.as_str()).unwrap_or("");
    if text("section") != "llm" {
        return Err(if text("kind") == "vision_projector" {
            format!("{name} is a vision projector, a part of a model: choose the language model's own file beside it.")
        } else {
            format!("{name} is not a language model: {}.", text("summary"))
        });
    }
    let arch = text("architecture");
    if !ENGINE_RUNS.contains(&arch) {
        return Err(format!(
            "{name} is a {} model, which OAIY's engine does not run yet. It runs Qwen3.5, Qwen3, Qwen2, Llama, Mistral, Gemma 3 and 4, and GLM models.",
            if arch.is_empty() { "kind of" } else { arch }
        ));
    }
    let types: Vec<&str> = detected.get("tensorTypes").and_then(|t| t.as_array()).map(|t| t.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
    let unread: Vec<&str> = types.into_iter().filter(|t| !ENGINE_TYPES.contains(t)).collect();
    if !unread.is_empty() {
        return Err(format!(
            "{name} is quantized as {}, which OAIY's engine does not decode yet: choose a Q4_K_M, Q5_K_M, Q6_K or Q8_0 file of the model instead.",
            unread.join(", ")
        ));
    }
    Ok((arch.to_string(), ENGINE_TOOLS.contains(&arch)))
}

/// Add a language model file the person already has to the engines (`/api/models/add`), once they have said what it
/// is (`/api/detect`), so a file the engine cannot run is refused before anything is written. The engines choose it
/// when none is chosen. Its name in Engines, its architecture, and whether the Agent can use its tools with it.
pub async fn add_model_file(raw: &str) -> Result<serde_json::Value, String> {
    let path = model_file(raw)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ui = crate::http::engines_ui().ok_or("The engines are not running: they start with OAIY.")?;
    let body = serde_json::json!({ "path": path.to_string_lossy() });
    let detected = crate::http::studio_json(&ui, reqwest::Method::POST, "/api/detect", Some(body.clone())).await.map_err(|(_, e)| e)?;
    let (arch, tools) = runnable(&detected, &name)?;
    let added = crate::http::studio_json(&ui, reqwest::Method::POST, "/api/models/add", Some(body)).await.map_err(|(_, e)| e)?;
    log::info!("engines: added the language model {} ({arch})", path.display());
    Ok(serde_json::json!({ "name": added.get("name"), "architecture": arch, "tools": tools }))
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
    fn a_model_file_is_a_whole_path_to_a_gguf_that_is_there_quoted_or_not() {
        let dir = std::env::temp_dir().join(format!("oaiy-model-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("Qwen3.5-4B-Q4_K_M.gguf");
        std::fs::write(&file, b"GGUF").unwrap();
        let shown = file.display().to_string();
        // As picked, typed with spaces around it, or pasted from Windows' "Copy as path".
        for raw in [shown.clone(), format!("  {shown} "), format!("\"{shown}\""), format!("'{shown}'")] {
            assert_eq!(model_file(&raw).unwrap(), file, "{raw}");
        }
        assert!(model_file("  ").unwrap_err().contains("Choose a model file"));
        assert!(model_file("models/x.gguf").unwrap_err().contains("not a whole path"));
        let not_gguf = dir.join("model.safetensors");
        std::fs::write(&not_gguf, b"x").unwrap();
        assert!(model_file(&not_gguf.display().to_string()).unwrap_err().contains("not a GGUF file"));
        assert!(model_file(&dir.join("gone.gguf").display().to_string()).unwrap_err().contains("is not there"));
        // A model split into files: its first part, never another.
        let first = dir.join("Big-Q4_K_M-00001-of-00003.gguf");
        let second = dir.join("Big-Q4_K_M-00002-of-00003.gguf");
        std::fs::write(&first, b"GGUF").unwrap();
        std::fs::write(&second, b"GGUF").unwrap();
        assert_eq!(model_file(&first.display().to_string()).unwrap(), first);
        let refused = model_file(&second.display().to_string()).unwrap_err();
        assert!(refused.contains("part 2 of 3") && refused.contains("-00001-of-00003.gguf"), "{refused}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_parts_are_read_from_llama_cpps_names_only() {
        assert_eq!(split_part("m-00001-of-00004.gguf"), Some((1, 4)));
        assert_eq!(split_part("Qwen3-235B-Q4_K_M-00004-of-00004.gguf"), Some((4, 4)));
        assert_eq!(split_part("m.gguf"), None);
        assert_eq!(split_part("m-1-of-4.gguf"), None);
        assert_eq!(split_part("m-0000a-of-00004.gguf"), None);
    }

    #[test]
    fn only_a_language_model_the_engine_loads_is_added_and_the_agent_tools_follow_its_architecture() {
        let llm = |arch: &str| serde_json::json!({ "kind": "llm", "section": "llm", "summary": "x — GGUF LLM", "architecture": arch });
        assert_eq!(runnable(&llm("qwen35"), "a.gguf").unwrap(), ("qwen35".to_string(), true));
        assert_eq!(runnable(&llm("glm5next"), "a.gguf").unwrap(), ("glm5next".to_string(), true));
        for arch in ["llama", "mistral", "qwen2", "qwen3", "gemma3", "gemma3n", "gemma4", "qwen3moe"] {
            assert_eq!(runnable(&llm(arch), "a.gguf").unwrap(), (arch.to_string(), false), "{arch} chats only");
        }
        let phi = runnable(&llm("phi3"), "phi.gguf").unwrap_err();
        assert!(phi.contains("phi3 model") && phi.contains("does not run yet"), "{phi}");
        // A studio older than the architecture in /api/detect: refused, not guessed.
        assert!(runnable(&serde_json::json!({ "section": "llm" }), "a.gguf").is_err());
        let projector = serde_json::json!({ "kind": "vision_projector", "section": "component", "summary": "Vision projector" });
        assert!(runnable(&projector, "mmproj-F16.gguf").unwrap_err().contains("vision projector"));
        let image = serde_json::json!({ "kind": "image", "section": "image", "summary": "Qwen Image transformer" });
        assert!(runnable(&image, "qwen-image.gguf").unwrap_err().contains("not a language model: Qwen Image transformer"));
    }

    #[test]
    fn a_quantization_the_engine_does_not_decode_is_refused_and_the_quants_it_does_are_named() {
        let llm = |types: &[&str]| serde_json::json!({ "section": "llm", "architecture": "qwen35", "tensorTypes": types });
        // A K-quant, Q8_0, IQ4 and the floats: decoded.
        assert!(runnable(&llm(&["F32", "Q4_K", "Q6_K"]), "a.gguf").is_ok());
        assert!(runnable(&llm(&["F32", "Q8_0", "BF16", "IQ4_XS", "IQ4_NL", "Q2_K", "Q3_K", "Q5_K"]), "a.gguf").is_ok());
        // Unsloth's smaller "UD" quants mix in IQ2 and IQ3 tensors, which fail when the model loads.
        let iq = runnable(&llm(&["F32", "IQ3_XXS", "Q4_K", "IQ2_S"]), "Qwen3.8-27B-UD-IQ3_XXS.gguf").unwrap_err();
        assert!(iq.contains("quantized as IQ3_XXS, IQ2_S") && iq.contains("Q4_K_M, Q5_K_M, Q6_K or Q8_0"), "{iq}");
        assert!(runnable(&llm(&["MXFP4", "F32"]), "gpt-oss.gguf").unwrap_err().contains("MXFP4"));
        assert!(runnable(&llm(&["TQ1_0"]), "t.gguf").unwrap_err().contains("TQ1_0"));
        // A studio that does not list them: the architecture decides.
        assert!(runnable(&serde_json::json!({ "section": "llm", "architecture": "llama" }), "a.gguf").is_ok());
    }

    #[test]
    fn the_engine_an_installer_carries_is_found_and_a_build_beside_the_program_before_it_is_preferred() {
        let base = std::env::temp_dir().join(format!("oaiy-engine-dirs-{}", std::process::id()));
        let (empty, beside, bundled) = (base.join("empty"), base.join("beside"), base.join("bundled"));
        for d in [&empty, &beside, &bundled] {
            std::fs::create_dir_all(d).unwrap();
        }
        std::fs::write(bundled.join(exe("oaiy-llm-server-webgpu")), b"x").unwrap();
        // The bundled server alone is enough, under the name the installer stages it by.
        let found = first_with_programs([empty.clone(), bundled.clone()]).unwrap();
        assert!(found.ends_with("bundled"), "{}", found.display());
        // A build earlier in the list wins.
        std::fs::write(beside.join(exe("oaiy-llm-server")), b"x").unwrap();
        assert!(first_with_programs([empty.clone(), beside.clone(), bundled.clone()]).unwrap().ends_with("beside"));
        // Nothing anywhere: none.
        assert_eq!(first_with_programs([empty.clone()]), None);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_older_versions_nvidia_engine_is_set_aside_and_its_folder_removed() {
        let data = std::env::temp_dir().join(format!("oaiy-retire-cuda-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let config = config_path(&data);
        std::fs::create_dir_all(config.parent().unwrap()).unwrap();
        // An install that fetched the NVIDIA engine of two versions, and ran the later.
        let old = old_cuda_engines(&data);
        for v in ["0.1.0", "0.2.0"] {
            std::fs::create_dir_all(old.join(v)).unwrap();
            std::fs::write(old.join(v).join(exe("oaiy-llm-server")), b"x").unwrap();
        }
        oaiy_studio::set_llm_server(&config, &old.join("0.2.0").join(exe("oaiy-llm-server")).to_string_lossy()).unwrap();
        retire_cuda_engine(&config, &data).unwrap();
        assert_eq!(oaiy_studio::llm_server(&config).unwrap(), "oaiy-llm-server", "back to the default, which this install's engine answers to");
        assert!(!old.exists(), "the downloaded engines are removed");
        assert!(config.is_file(), "the configuration beside them is kept");
        // A server of the person's own choosing is theirs; and a start with nothing to set aside changes nothing.
        let own = data.join("mine").join(exe("oaiy-llm-server"));
        oaiy_studio::set_llm_server(&config, &own.to_string_lossy()).unwrap();
        retire_cuda_engine(&config, &data).unwrap();
        assert_eq!(oaiy_studio::llm_server(&config).unwrap(), own.to_string_lossy());
        // The folder gone but the setting still under it (removed by hand): the default again.
        oaiy_studio::set_llm_server(&config, &old.join("0.2.0").join(exe("oaiy-llm-server")).to_string_lossy()).unwrap();
        retire_cuda_engine(&config, &data).unwrap();
        assert_eq!(oaiy_studio::llm_server(&config).unwrap(), "oaiy-llm-server");
        let _ = std::fs::remove_dir_all(&data);
    }

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
        assert_eq!(parse_activity(&idle, &none), Ok((0, 0)));
        // Jobs the studio lists that have not finished.
        let busy = json!({"media": {"busy": true, "jobs": [{"id": "j1", "status": "running"}, {"id": "j2", "status": "completed"}, {"id": "j3", "status": "queued"}, {"id": "j4", "status": "failed"}, {"id": "j5", "status": "cancelled"}]}});
        assert_eq!(parse_activity(&busy, &none).unwrap().0, 2);
        // busy with no list still counts once; not busy with a list of finished jobs counts none.
        assert_eq!(parse_activity(&json!({"media": {"busy": true}}), &none).unwrap().0, 1);
        assert_eq!(parse_activity(&json!({"media": {"busy": false, "jobs": [{"status": "completed"}]}}), &none).unwrap().0, 0);
        // Downloads waiting or running count; finished, paused, failed and cancelled do not.
        let downloads = json!({"models": [
            {"download": {"status": "queued"}}, {"download": {"status": "downloading"}}, {"download": {"status": "adding"}},
            {"download": {"status": "done"}}, {"download": {"status": "paused"}}, {"download": {"status": "failed"}}, {"download": {"status": "cancelled"}}
        ]});
        assert_eq!(parse_activity(&idle, &downloads).unwrap().1, 3);
    }

    #[test]
    fn what_they_answer_that_is_not_the_state_of_a_studio_is_not_nothing_running() {
        use serde_json::json;
        let idle = json!({"media": {"busy": false, "jobs": []}});
        let none = json!({"models": []});
        for (state, downloads) in [
            (json!(null), none.clone()),
            (json!({}), none.clone()),
            (json!({"media": null}), none.clone()),
            (json!({"media": "busy"}), none.clone()),
            (json!({"media": {}}), none.clone()),
            (idle.clone(), json!("nope")),
            (idle.clone(), json!({})),
            (idle.clone(), json!({"models": null})),
        ] {
            assert!(parse_activity(&state, &downloads).is_err(), "{state} {downloads}");
        }
    }

    /// A stand-in for a studio's control pages on a port of its own: `answer` says what a path is answered with (a status and a body), or None: it takes
    /// the connection and never answers.
    struct Stub {
        url: String,
        asked: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    fn stub(answer: impl Fn(&str) -> Option<(u16, String)> + Send + Sync + 'static) -> Stub {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (asked2, answer) = (asked.clone(), std::sync::Arc::new(answer));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let (asked, answer) = (asked2.clone(), answer.clone());
                std::thread::spawn(move || {
                    let mut buffer = [0u8; 2048];
                    let n = stream.read(&mut buffer).unwrap_or(0);
                    let request = String::from_utf8_lossy(&buffer[..n]).to_string();
                    let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                    asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    match answer(&path) {
                        Some((status, body)) => {
                            let _ = write!(stream, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                        }
                        None => std::thread::sleep(std::time::Duration::from_secs(5)),
                    }
                });
            }
        });
        Stub { url, asked }
    }

    const QUICK: std::time::Duration = std::time::Duration::from_millis(400);
    /// Time for a refused connection to be reported: about two seconds on Windows.
    const CONNECT: std::time::Duration = std::time::Duration::from_secs(6);

    fn studio(media: &str) -> impl Fn(&str) -> Option<(u16, String)> + Send + Sync + 'static {
        let media = media.to_string();
        move |path| match path {
            "/api/state" => Some((200, format!("{{\"media\": {media}}}"))),
            "/api/downloads" => Some((200, "{\"models\": [{\"download\": {\"status\": \"downloading\"}}]}".to_string())),
            _ => Some((404, "{}".to_string())),
        }
    }

    #[test]
    fn a_studio_that_answers_is_read_and_one_that_is_not_there_is_nothing_running() {
        let kept: Kept = Mutex::new(None);
        // No control pages known at all: nothing to interrupt.
        assert_eq!(activity_at(None, true, &kept, QUICK, CONNECT), Ok((0, 0)));
        // One that answers.
        let s = stub(studio("{\"busy\": true, \"jobs\": [{\"status\": \"running\"}]}"));
        assert_eq!(activity_at(Some(&s.url), true, &kept, QUICK, CONNECT), Ok((1, 1)));
        // Nothing listening at the address they were found at (the connection is refused): the studio is gone, so there is no work to lose.
        let gone = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}", listener.local_addr().unwrap())
        };
        assert_eq!(activity_at(Some(&gone), true, &kept, QUICK, CONNECT), Ok((0, 0)));
        // ...and that is kept for longer than an answer (it costs the wait for the refusal to be reported), but never for an install.
        let (_, keep, _) = kept.lock().unwrap().clone().unwrap();
        assert_eq!(keep, ACTIVITY_KEEP_GONE);
    }

    #[test]
    fn a_studio_that_does_not_answer_or_answers_wrongly_is_not_idle_and_the_reason_is_in_words() {
        let kept: Kept = Mutex::new(None);
        // It takes the connection and never answers: the two-second wait ends, and that is not "nothing is running".
        let hung = stub(|_| None);
        let started = std::time::Instant::now();
        match activity_at(Some(&hung.url), true, &kept, QUICK, CONNECT) {
            Err(why) => assert!(why.contains("no answer within"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(4), "it gave up at its own deadline");
        for (what, answer) in [
            ("an error status", stub(|_| Some((500, "{}".to_string())))),
            ("something that is not JSON", stub(|_| Some((200, "<html>busy</html>".to_string())))),
            ("JSON that is not a state", stub(|_| Some((200, "{\"hello\": 1}".to_string())))),
        ] {
            assert!(activity_at(Some(&answer.url), true, &kept, QUICK, CONNECT).is_err(), "{what}");
        }
        // And through the blockers: the engines that cannot say block an install.
        let unknown = crate::update::blockers::EnginesState::Unknown("no answer within 0.4 s".to_string());
        let readings = crate::update::blockers::Readings { engines: unknown, ..Default::default() };
        let blockers = crate::update::blockers::compute(Some(&readings), std::time::Duration::from_secs(3600));
        assert!(blockers.iter().any(|b| b.code == "enginesUnknown"), "{blockers:?}");
    }

    #[test]
    fn an_answer_may_be_kept_for_the_status_and_an_install_asks_again() {
        let kept: Kept = Mutex::new(None);
        let busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let b = busy.clone();
        let s = stub(move |path| {
            let media = if b.load(std::sync::atomic::Ordering::SeqCst) { "{\"busy\": true, \"jobs\": [{\"status\": \"running\"}]}" } else { "{\"busy\": false, \"jobs\": []}" };
            studio(media)(path)
        });
        assert_eq!(activity_at(Some(&s.url), false, &kept, QUICK, CONNECT).unwrap().0, 1);
        let asked = s.asked.load(std::sync::atomic::Ordering::SeqCst);
        // The job ends. The polled status answers from what it kept; the decision to install asks again.
        busy.store(false, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(activity_at(Some(&s.url), false, &kept, QUICK, CONNECT).unwrap().0, 1, "kept for a few seconds");
        assert_eq!(s.asked.load(std::sync::atomic::Ordering::SeqCst), asked, "nothing was asked");
        assert_eq!(activity_at(Some(&s.url), true, &kept, QUICK, CONNECT).unwrap().0, 0, "fresh: the studio is asked again, whatever it said a moment ago");
        assert!(s.asked.load(std::sync::atomic::Ordering::SeqCst) > asked);
    }

    #[test]
    fn a_debug_build_only_uses_running_engines_unless_told() {
        // (The variable is read at call time; the default depends on the build.)
        if std::env::var_os("OAIY_ENGINES").is_none() {
            assert_eq!(mode(), if cfg!(debug_assertions) { Mode::Attach } else { Mode::Launch });
        }
    }
}
