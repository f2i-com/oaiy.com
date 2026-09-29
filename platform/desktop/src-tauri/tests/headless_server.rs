//! The headless `oaiy-server`, run as the program it is.
//!
//! What the unit tests cannot reach is what `main` does with its environment. On this build
//! `OAIY_ENGINES_UI` was recorded there and the AI gateway ignored it, which no test of the
//! gateway's routes could see.
//!
//! Each test starts its own server on a port the system picks, with a data folder, home folder
//! and environment of its own, and the voice gateway (a fixed port) switched off, so it can never
//! meet the desktop that may be running on this machine.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
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
    fn start(scratch: &Scratch, extra_env: &[(&str, String)]) -> Server {
        // A port the system picked, released a moment before the server takes it.
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
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
        let mut server = Server { child, port, stderr };
        server.wait_until_up();
        server
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
