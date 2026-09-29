//! The headless `oaiy-server`, run as the program it is.
//!
//! What the unit tests cannot reach is what `main` does with its environment and its signals. On
//! this build `OAIY_ENGINES_UI` was recorded there and the AI gateway ignored it, which no test of
//! the gateway's routes could see, and a server that was stopped left its plugins running.
//!
//! Each test starts its own server on a port the system picks, with a data folder, home folder
//! and environment of its own, and the voice gateway (a fixed port) switched off, so it can never
//! meet the desktop that may be running on this machine.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const TOKEN: &str = "integration-test-token";

/// A folder of the test's own, removed when the test ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("oaiy-headless-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A running `oaiy-server`. Killed when dropped, whatever the test did.
struct Server {
    child: Child,
    port: u16,
    stderr: PathBuf,
}

impl Server {
    /// Start it on a port the system picks, and wait until it answers.
    fn start(scratch: &Scratch, extra_env: &[(&str, String)]) -> Server {
        // A port the system picked, released a moment before the server takes it.
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut server = Server::spawn(scratch, extra_env, port);
        server.wait_until_up();
        server
    }

    /// Start it on `port` and leave it to come up (or not) by itself.
    fn spawn(scratch: &Scratch, extra_env: &[(&str, String)], port: u16) -> Server {
        let data = scratch.0.join("data");
        let stderr = scratch.0.join("server.stderr");

        let mut command = Command::new(env!("CARGO_BIN_EXE_oaiy-server"));
        // Nothing of this environment but what a program needs to run: a developer's shell may
        // hold the real server's token, and the server must not take its port or folders from it.
        command.env_clear();
        for name in ["PATH", "SystemRoot", "SYSTEMROOT", "windir", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "LANG"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .env("HOME", &scratch.0)
            .env("USERPROFILE", &scratch.0)
            .env("OAIY_DATA_DIR", &data)
            .env("OAIY_MODELS_DIR", data.join("models"))
            .env("OAIY_SERVER_PORT", port.to_string())
            .env("OAIY_SERVER_TOKEN", TOKEN)
            // A fixed port (17872) the running desktop may hold.
            .env("OAIY_VOICE_GATEWAY", "off");
        for (name, value) in extra_env {
            command.env(name, value);
        }
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()))
            .spawn()
            .expect("start oaiy-server");
        Server { child, port, stderr }
    }

    fn wait_until_up(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("oaiy-server exited early ({status}): {}", self.stderr_tail());
            }
            if self.healthy() {
                return;
            }
            assert!(Instant::now() < deadline, "oaiy-server did not come up: {}", self.stderr_tail());
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Does it answer its health route yet? (False, not a panic, while it is still starting.)
    fn healthy(&self) -> bool {
        let Ok(client) = reqwest::blocking::Client::builder().timeout(Duration::from_secs(5)).build() else { return false };
        client
            .get(format!("http://127.0.0.1:{}/api/health", self.port))
            .send()
            .is_ok_and(|r| r.status().as_u16() == 200)
    }

    fn stderr_tail(&self) -> String {
        let text = std::fs::read_to_string(&self.stderr).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        lines[lines.len().saturating_sub(15)..].join("\n")
    }

    fn call(&self, method: reqwest::Method, path: &str, body: Option<Value>, authorised: bool) -> (u16, Value) {
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
        let mut request = client.request(method, format!("http://127.0.0.1:{}{path}", self.port));
        if authorised {
            request = request.bearer_auth(TOKEN);
        }
        if let Some(body) = body {
            request = request.header("content-type", "application/json").body(body.to_string());
        }
        let response = request.send().unwrap_or_else(|e| panic!("{path}: {e}"));
        let status = response.status().as_u16();
        let text = response.text().unwrap_or_default();
        (status, serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    fn get(&self, path: &str, authorised: bool) -> (u16, Value) {
        self.call(reqwest::Method::GET, path, None, authorised)
    }

    fn post(&self, path: &str, body: Value) -> (u16, Value) {
        self.call(reqwest::Method::POST, path, Some(body), true)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A stand-in for the engines, on a port of its own: their control page (`/api/state`, which
/// names the gateway) and a gateway behind it that records what it is asked to chat about.
struct FakeEngines {
    ui: String,
    chats: Arc<Mutex<Vec<Value>>>,
}

fn fake_engines() -> FakeEngines {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let ui = format!("http://{}", listener.local_addr().unwrap());
    let chats: Arc<Mutex<Vec<Value>>> = Arc::default();
    let (gateway, recorded) = (format!("{ui}/gw"), chats.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (gateway, recorded) = (gateway.clone(), recorded.clone());
            std::thread::spawn(move || answer(stream, &gateway, &recorded));
        }
    });
    FakeEngines { ui, chats }
}

/// One HTTP exchange with the fake engines.
fn answer(mut stream: TcpStream, gateway: &str, chats: &Mutex<Vec<Value>>) {
    let mut received = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => received.extend_from_slice(&chunk[..n]),
        }
        if let Some(at) = received.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let head = String::from_utf8_lossy(&received[..head_end]).to_string();
    let mut request_line = head.lines().next().unwrap_or_default().split(' ');
    let (method, path) = (request_line.next().unwrap_or_default(), request_line.next().unwrap_or_default());
    let length: usize = head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
        .unwrap_or(0);
    while received.len() < head_end + length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => received.extend_from_slice(&chunk[..n]),
        }
    }
    let body: Value = serde_json::from_slice(&received[head_end..]).unwrap_or(Value::Null);

    let (status, answer) = match (method, path) {
        ("GET", "/api/state") => (200, json!({ "gateway_url": gateway })),
        ("GET", "/gw/v1/discovery") => (
            200,
            json!({
                "endpoints": [{ "name": "chat", "path": "/v1/chat/completions", "spec": "openai" }],
                "models": { "llm": [{ "id": "flash", "default": true, "files_present": true }] },
                "defaults": { "llm": "flash" },
            }),
        ),
        ("POST", "/gw/v1/chat/completions") => {
            chats.lock().unwrap().push(body);
            (
                200,
                json!({
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "choices": [{ "index": 0, "message": { "role": "assistant", "content": "hello from the engine" }, "finish_reason": "stop" }],
                }),
            )
        }
        _ => (404, json!({ "error": "not here" })),
    };
    let text = answer.to_string();
    let _ = write!(
        stream,
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
}

/// The headless build used to answer 503 here whatever it was told: its AI gateway asked a stub
/// that only the window's own engines could fill in.
#[test]
fn a_headless_server_told_where_the_engines_are_answers_through_them() {
    let scratch = Scratch::new("engines");
    let engines = fake_engines();
    let server = Server::start(&scratch, &[("OAIY_ENGINES_UI", engines.ui.clone())]);

    // The route the engines' own pages have always had here answers, as before.
    let (status, engines_state) = server.get("/api/engines", true);
    assert_eq!(status, 200, "{engines_state}");
    assert_eq!(engines_state["running"], true, "{engines_state}");

    // And so does the AI gateway's.
    let (status, listed) = server.get("/api/ai/engine/services", true);
    assert_eq!(status, 200, "{listed}");
    assert_eq!(listed["running"], true, "the engines' models are listed for flows: {listed}");
    assert_eq!(listed["services"][0]["id"], "engine:llm:flash", "{listed}");

    let request = json!({ "model": "a-model-nobody-chose", "messages": [{ "role": "user", "content": "hi" }] });
    let (status, answered) = server.post("/api/ai/providers/oaiy-engine/v1/chat/completions", request);
    assert_eq!(status, 200, "{answered}");
    assert_eq!(answered["choices"][0]["message"]["content"], "hello from the engine");
    assert_eq!(engines.chats.lock().unwrap()[0]["model"], "flash", "the model chosen in Engines answers");

    let (status, sources) = server.get("/api/ai/sources", true);
    assert_eq!(status, 200, "{sources}");
    assert!(
        sources["sources"].as_array().unwrap().iter().any(|s| s["providerId"] == "oaiy-engine"),
        "the engine is one of the sources a flow can pick: {sources}"
    );
}

/// Told nothing, the same server says the engines are not there, as it always did.
#[test]
fn a_headless_server_told_nothing_says_the_engines_are_not_running() {
    let scratch = Scratch::new("no-engines");
    let server = Server::start(&scratch, &[]);
    let (status, listed) = server.get("/api/ai/engine/services", true);
    assert_eq!(status, 200, "{listed}");
    assert_eq!(listed["running"], false, "{listed}");
    let request = json!({ "messages": [{ "role": "user", "content": "hi" }] });
    let (status, refused) = server.post("/api/ai/providers/oaiy-engine/v1/chat/completions", request);
    assert_eq!((status, refused["error"]["code"].as_str()), (503, Some("engine_unavailable")), "{refused}");
}

/// What the server writes is private where the OS lets us say so: the data folder it made
/// itself, the identity keys it mints on the way up, and a provider key somebody saves.
#[cfg(unix)]
#[test]
fn a_fresh_server_keeps_its_data_folder_keys_and_provider_store_to_its_owner() {
    use std::os::unix::fs::PermissionsExt as _;
    let scratch = Scratch::new("private");
    let server = Server::start(&scratch, &[]);
    let data = scratch.0.join("data");
    let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&data), 0o700, "the data folder the server made");

    // A provider's API key, saved over the API as the dashboard does.
    let (status, saved) = server.post(
        "/api/ai/providers",
        json!({ "id": "openai", "name": "OpenAI", "baseUrl": "https://api.openai.com" }),
    );
    assert_eq!(status, 200, "{saved}");
    let (status, _) = server.post("/api/ai/providers/openai/key", json!({ "key": "sk-not-a-real-key" }));
    assert_eq!(status, 204);
    let store = data.join("ai").join("providers.json");
    assert!(std::fs::read_to_string(&store).unwrap().contains("sk-not-a-real-key"), "the key is what the file is for");
    assert_eq!(mode(&store), 0o600, "providers.json is readable by others");
    assert_eq!(mode(store.parent().unwrap()), 0o700, "and so is the folder the server made for it");

    // The keys are minted on threads of their own as the server comes up.
    let keys = |data: &Path| -> Vec<PathBuf> {
        walk(data).into_iter().filter(|p| p.extension().is_some_and(|e| e == "key")).collect()
    };
    let deadline = Instant::now() + Duration::from_secs(15);
    while keys(&data).is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    let keys = keys(&data);
    assert!(!keys.is_empty(), "the server should have minted at least the node's signing key");
    for key in keys {
        assert_eq!(mode(&key), 0o600, "{} is readable by others", key.file_name().unwrap().to_string_lossy());
    }
}

#[cfg(unix)]
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found
}

/// A plugin that ignores every request to stop, and started a helper of its own: what a wedged
/// plugin launched through a shim looks like.
#[cfg(target_os = "linux")]
const WEDGED_PLUGIN: &str = r#"#!/bin/sh
sleep 300 &
echo $! > "$OAIY_PLUGIN_DATA_DIR/helper.pid"
echo $$ > "$OAIY_PLUGIN_DATA_DIR/plugin.pid"
while IFS= read -r line; do
  case "$line" in
    *'"method":"plugin.init"'*|*'"method":"plugin.health"'*)
      id=${line#*\"id\":}; id=${id%%,*}
      printf '{"jsonrpc":"2.0","id":%s,"result":{"status":"ok"}}\n' "$id"
      ;;
  esac
done
sleep 300
"#;

/// A data folder with the wedged plugin installed in it.
#[cfg(target_os = "linux")]
fn install_wedged_plugin(scratch: &Scratch) {
    use std::os::unix::fs::PermissionsExt as _;
    let plugin = scratch.0.join("data").join("plugins").join("fake");
    std::fs::create_dir_all(&plugin).unwrap();
    std::fs::write(
        plugin.join("manifest.json"),
        json!({
            "schemaVersion": 3, "id": "fake", "name": "Fake plugin", "version": "0.0.1", "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.sh" },
            "capabilities": ["oaiy.flow.run"], "connectors": [], "events": [],
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(plugin.join("plugin.sh"), WEDGED_PLUGIN).unwrap();
    std::fs::set_permissions(plugin.join("plugin.sh"), std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// The pids the wedged plugin wrote: itself, and the helper it started.
#[cfg(target_os = "linux")]
fn wedged_plugin_pids(scratch: &Scratch) -> (u32, u32) {
    let data = scratch.0.join("data").join("plugin-data").join("fake");
    let pid_of = |name: &str| -> u32 { std::fs::read_to_string(data.join(name)).unwrap().trim().parse().unwrap() };
    (pid_of("plugin.pid"), pid_of("helper.pid"))
}

/// Running, or as good as gone? A zombie only waits to be reaped.
#[cfg(target_os = "linux")]
fn alive(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat.rsplit(')').next().map(str::trim_start).is_some_and(|rest| !rest.starts_with('Z')),
        Err(_) => false,
    }
}

/// After the server has gone, the kernel may need a moment to finish with what was signalled.
#[cfg(target_os = "linux")]
fn gone_within(pids: &[u32], wait: Duration) -> bool {
    let deadline = Instant::now() + wait;
    while pids.iter().any(|p| alive(*p)) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    !pids.iter().any(|p| alive(*p))
}

/// A stopped server used to leave its plugins running (the app's own exit stopped them, this
/// binary's did not), and a plugin's own helpers as well, because a plugin shared the server's
/// process group and so could not be stopped as a tree.
#[cfg(target_os = "linux")]
#[test]
fn stopping_the_server_stops_its_plugins_and_what_they_started() {
    let scratch = Scratch::new("plugins");
    install_wedged_plugin(&scratch);
    let mut server = Server::start(&scratch, &[]);

    // Started at boot, and past its handshake.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, listed) = server.get("/api/plugins", true);
        if listed["plugins"].as_array().is_some_and(|p| p.iter().any(|p| p["id"] == "fake" && p["state"] == "running")) {
            break;
        }
        assert!(Instant::now() < deadline, "the plugin never came up: {listed} {}", server.stderr_tail());
        std::thread::sleep(Duration::from_millis(200));
    }
    let (plugin, helper) = wedged_plugin_pids(&scratch);
    assert!(alive(plugin) && alive(helper), "both run before the server is stopped");

    // What `systemctl stop` sends.
    let sent = Command::new("kill").args(["-TERM", &server.child.id().to_string()]).status().unwrap();
    assert!(sent.success());
    let deadline = Instant::now() + Duration::from_secs(60);
    while server.child.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "the server did not stop: {}", server.stderr_tail());
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(gone_within(&[plugin, helper], Duration::from_secs(15)), "the plugin or its helper outlived the server");
}

/// A server that cannot take its port used to exit with the plugins it had already started still
/// running: the port is found taken only when the API binds it, after the plugins are up.
#[cfg(target_os = "linux")]
#[test]
fn a_server_that_cannot_bind_its_port_stops_its_plugins_before_it_exits() {
    let scratch = Scratch::new("bind-fails");
    install_wedged_plugin(&scratch);
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port();
    let mut server = Server::spawn(&scratch, &[], port);

    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = server.child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "the server neither came up nor gave up");
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(status.code(), Some(1), "it gives up: {}", server.stderr_tail());

    // The plugin starts on a thread of its own while the API is being made ready. It was up
    // before the bind failed, or this run proved nothing.
    let pids = std::fs::read_to_string(scratch.0.join("data").join("plugin-data").join("fake").join("plugin.pid"));
    assert!(pids.is_ok(), "the plugin had not started when the bind failed, so this run proved nothing");
    let (plugin, helper) = wedged_plugin_pids(&scratch);
    assert!(gone_within(&[plugin, helper], Duration::from_secs(15)), "the plugin or its helper outlived the server");
    drop(taken);
}
