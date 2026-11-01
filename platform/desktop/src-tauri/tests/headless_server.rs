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

/// A token of the kind the server requires of OAIY_SERVER_TOKEN (`auth::token::check_static_token_shape`: 32 to 256
/// printable characters, no common pattern). One that has one stops the server with exit 78.
const TOKEN: &str = "vS9nr2Bbbn7lc1uHb8B9ewB1B7oVDU9m4fQqDLJ-M9c";

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
        Server::spawn_logging_to(scratch, extra_env, port, "server.stderr")
    }

    /// [`Server::spawn`], its stderr in the file `log` of the folder: a second server on one data folder has its own.
    fn spawn_logging_to(scratch: &Scratch, extra_env: &[(&str, String)], port: u16, log: &str) -> Server {
        let data = scratch.0.join("data");
        let stderr = scratch.0.join(log);

        let mut command = Command::new(env!("CARGO_BIN_EXE_oaiy-server"));
        // Nothing of this environment but what a program needs to run: a developer's shell may
        // hold the real server's token, and the server must not take its port or folders from it.
        command.env_clear();
        for name in ["SystemRoot", "SYSTEMROOT", "windir", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "LANG"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        // The system's own folders and no others: enough for `sh` and `sleep` (the plugin tests)
        // and `cmd`, and not the developer's `codex` CLI, which the AI sources route would launch.
        let path = if cfg!(windows) {
            let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
            format!(r"{root}\System32;{root}")
        } else {
            "/usr/bin:/bin".to_string()
        };
        command
            .env("PATH", path)
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
                "endpoints": [{ "name": "chat", "path": "/v1/chat/completions", "spec": "openai" }, { "name": "files", "path": "/files" }],
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
        ("POST", "/gw/files") => (200, json!({ "url": "/files/inputs/stored.png" })),
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

/// The folder `node` is in, when this machine has one: what a flow's CLI runs under.
fn node_dir() -> Option<PathBuf> {
    let name = if cfg!(windows) { "node.exe" } else { "node" };
    std::env::split_paths(&std::env::var_os("PATH")?).find(|dir| dir.join(name).is_file())
}

/// A stand-in for the CLI: a ready ZIPP engine to the server, and for a run what a flow's nodes do with what the server
/// gives its process. It writes a line to `runs.log` for each run, so a run that executes twice shows, and calls back
/// on its credential: the AI gateway, as an AI node does, and two routes no flow calls.
fn stub_cli(scratch: &Scratch) -> PathBuf {
    let script = scratch.0.join("stub-cli.mjs");
    let capabilities = json!({
        "version": "0.2.0",
        "protocols": { "run": 1, "script": 1, "profile": 1 },
        "engine": {
            "name": "zipp", "status": "ready",
            "release": "vF.I.X", "version": "F.I.X", "revision": "0123abcd",
            "wasmSha256": "ab".repeat(32), "languages": ["javascript"],
        },
        "run": { "languages": ["javascript"], "defaultInstructionSteps": 50, "maxInstructionSteps": 2000 },
    });
    let body = format!(
        r#"import fs from 'node:fs';
const a = process.argv.slice(2);
if (a[0] === 'capabilities') {{ process.stdout.write({capabilities}); process.exit(0); }}
if (a[0] !== 'run') process.exit(2);
fs.appendFileSync({log}, 'ran\n');
const inputs = fs.readFileSync(a[a.indexOf('--inputs') + 1], 'utf8');
fs.writeFileSync({kept}, inputs);
if (JSON.parse(inputs).hang) {{ await new Promise((go) => setTimeout(go, 60000)); process.exit(1); }}
const token = process.env.OAIY_SERVER_TOKEN;
const call = (method, path, body) => fetch(process.env.OAIY_SERVER_URL + path, {{
  method, headers: {{ authorization: `Bearer ${{token}}`, 'content-type': 'application/json' }}, body: body && JSON.stringify(body),
}});
const chat = await call('POST', '/api/ai/providers/oaiy-engine/v1/chat/completions', {{ messages: [{{ role: 'user', content: 'hi' }}] }});
const answer = await chat.json();
fs.writeFileSync(a[a.indexOf('-o') + 1], JSON.stringify({{
  success: chat.ok, engine: 'zipp', error: chat.ok ? undefined : JSON.stringify(answer), token, chat: chat.status,
  config: (await call('GET', '/api/config')).status,
  whoami: (await call('GET', '/api/auth/whoami')).status,
}}));
"#,
        capabilities = serde_json::to_string(&capabilities.to_string()).unwrap(),
        log = serde_json::to_string(&scratch.0.join("runs.log")).unwrap(),
        kept = serde_json::to_string(&scratch.0.join("last-inputs.json")).unwrap(),
    );
    std::fs::write(&script, body).unwrap();
    script
}

/// What a server needs to run flows on [`stub_cli`] in `scoped` mode (the web build's, and the one named for the build
/// without the login, which defaults to `legacy`), with `engines` for its AI. `None` on a machine with no `node`.
fn flow_env(scratch: &Scratch, engines: &FakeEngines) -> Option<Vec<(&'static str, String)>> {
    let Some(node) = node_dir() else {
        eprintln!("skipped: no node on this machine");
        return None;
    };
    let path = if cfg!(windows) {
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        format!(r"{root}\System32;{root};{}", node.display())
    } else {
        format!("/usr/bin:/bin:{}", node.display())
    };
    Some(vec![
        ("OAIY_ACCESS_MODE", "scoped".to_string()),
        ("OAIY_CLI", stub_cli(scratch).display().to_string()),
        ("OAIY_ENGINES_UI", engines.ui.clone()),
        ("PATH", path),
    ])
}

/// Store a flow for [`stub_cli`] to run.
fn store_flow(server: &Server) {
    let flow = json!({ "nodes": [{ "id": "n", "type": "logic_block", "position": { "x": 0, "y": 0 }, "data": { "code": "return 1;" } }], "edges": [] });
    let (status, answered) = server.call(reqwest::Method::PUT, "/api/bridge/flows/stub", Some(flow), true);
    assert!(status < 300, "{status} {answered}");
}

/// Reserve a run of that flow under `key`.
fn run_flow(server: &Server, key: &str) -> Value {
    run_flow_on(server, key, json!({}))
}

/// [`run_flow`], with inputs.
fn run_flow_on(server: &Server, key: &str, input: Value) -> Value {
    let request = json!({
        "protocol": "oaiy-bridge/1", "caller": { "product": "headless-test" }, "flowId": "stub", "input": input,
        "correlationId": key, "idempotencyKey": key, "timeoutMs": 60000, "mode": "async",
    });
    let (status, answered) = server.post("/api/bridge/runs", request);
    assert!(status < 300, "{status} {answered}");
    answered
}

/// Wait for a run to end, and answer what the ledger says of it.
fn finished(server: &Server, run: &Value) -> Value {
    let id = run["runId"].as_str().expect("a run id");
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let (_, now) = server.get(&format!("/api/bridge/runs/{id}"), true);
        if matches!(now["status"].as_str(), Some("succeeded" | "failed" | "cancelled" | "timed_out")) {
            return now;
        }
        assert!(Instant::now() < deadline, "the run did not end: {now} {}", server.stderr_tail());
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// `scoped` does not take the internal token, and a flow is a process the server starts: it used to be handed that
/// token, so every flow with an AI node failed at its first call with "Check your API key". The release's mode.
#[test]
fn a_flow_the_server_runs_calls_back_on_a_credential_of_its_own_that_ends_with_it() {
    let scratch = Scratch::new("flow-credential");
    let engines = fake_engines();
    let Some(env) = flow_env(&scratch, &engines) else { return };
    let server = Server::start(&scratch, &env);
    store_flow(&server);
    let run = finished(&server, &run_flow(&server, "a"));

    assert_eq!(run["status"], "succeeded", "{run}");
    assert_eq!(run["output"]["chat"], 200, "{run}");
    assert_eq!(engines.chats.lock().unwrap().len(), 1, "the AI node reached the engines");
    let token = run["output"]["token"].as_str().unwrap();
    assert!(token.starts_with("oaiyrun_"), "{token}");

    // It is what a flow's nodes need, no more: a route of a scope it does not hold, and one the access model added.
    assert_eq!(run["output"]["config"], 403, "{run}");
    assert_eq!(run["output"]["whoami"], 403, "{run}");

    // And it ended with the run.
    let refused = reqwest::blocking::Client::new()
        .get(format!("http://127.0.0.1:{}/api/ai/engine/services", server.port))
        .bearer_auth(token)
        .send()
        .unwrap();
    assert_eq!(refused.status().as_u16(), 401);
    assert_eq!(refused.json::<Value>().unwrap()["error"]["code"], "token_revoked");
    // Nothing of it is on the disk.
    let credentials = std::fs::read_to_string(scratch.0.join("data").join("auth").join("credentials.json")).unwrap_or_default();
    assert!(!credentials.contains("flow run"), "{credentials}");
}

/// A picture given inline goes to the engines before the run starts, from this process over loopback, on the same
/// internal token: in `scoped` the run failed there, before the flow was given anything.
#[test]
fn a_run_with_a_picture_hands_it_to_the_engines_on_a_credential_of_its_own() {
    let scratch = Scratch::new("flow-picture");
    let engines = fake_engines();
    let Some(env) = flow_env(&scratch, &engines) else { return };
    let server = Server::start(&scratch, &env);
    store_flow(&server);
    let picture = format!("data:image/png;base64,{}", "AAAA".repeat(24 * 1024));
    let run = finished(&server, &run_flow_on(&server, "picture", json!({ "picture": picture })));
    assert_eq!(run["status"], "succeeded", "{}", run["error"]);
    let inputs = std::fs::read_to_string(scratch.0.join("last-inputs.json")).unwrap();
    let reference = format!("http://127.0.0.1:{}/api/ai/engine/gateway/files/inputs/stored.png", server.port);
    assert_eq!(serde_json::from_str::<Value>(&inputs).unwrap()["picture"], reference.as_str());
}

/// A start that is refused used to do all that a start does first: its ledger replaced the running server's file by
/// rename, every run after that lived in memory alone, and the same idempotency key ran the flow again after a restart.
#[test]
fn a_refused_second_start_leaves_the_running_servers_ledger_and_idempotency_alone() {
    let scratch = Scratch::new("second-start");
    let engines = fake_engines();
    let Some(env) = flow_env(&scratch, &engines) else { return };
    let first = Server::start(&scratch, &env);
    store_flow(&first);
    assert_eq!(finished(&first, &run_flow(&first, "before"))["status"], "succeeded");

    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut second = Server::spawn_logging_to(&scratch, &env, port, "second.stderr");
    let deadline = Instant::now() + Duration::from_secs(60);
    let refused = loop {
        if let Some(status) = second.child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "the second server was not refused");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(refused.code(), Some(78), "{}", second.stderr_tail());
    let said = second.stderr_tail();
    assert!(said.contains("in use by process") && !said.contains("oaiy-server starting"), "{said}");

    // The run after it reaches the file, which the first server's handle still names.
    let after = finished(&first, &run_flow(&first, "after"));
    assert_eq!(after["status"], "succeeded");
    let id = after["runId"].as_str().unwrap();
    let ledger = std::fs::read_to_string(scratch.0.join("data").join("bridge").join("ledger.jsonl")).unwrap();
    assert!(ledger.contains(id), "the run is not in the ledger's file");

    // A restart still answers it, and the same key does not run the flow again.
    drop(first);
    let ran = || std::fs::read_to_string(scratch.0.join("runs.log")).unwrap().lines().count();
    assert_eq!(ran(), 2);
    let restarted = Server::start(&scratch, &env);
    let (status, kept) = restarted.get(&format!("/api/bridge/runs/{id}"), true);
    assert_eq!((status, kept["status"].as_str()), (200, Some("succeeded")), "{kept}");
    let repeat = run_flow(&restarted, "after");
    assert_eq!((repeat["runId"].as_str(), repeat["idempotent"].as_bool()), (Some(id), Some(true)), "{repeat}");
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(ran(), 2, "the flow ran again");
}

/// A run's folder was removed only by the run: the server killed in the middle of one left its inputs in the temp
/// folder for good. A start sweeps them, and only them: a folder a run on this same temp folder is using stays.
#[test]
fn what_a_killed_run_left_in_the_temp_folder_goes_at_the_next_start() {
    let scratch = Scratch::new("killed-run");
    let engines = fake_engines();
    let Some(mut env) = flow_env(&scratch, &engines) else { return };
    let temp = scratch.0.join("temp");
    std::fs::create_dir(&temp).unwrap();
    for name in ["TEMP", "TMP", "TMPDIR"] {
        env.push((name, temp.display().to_string()));
    }
    let leftovers = || std::fs::read_dir(&temp).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("oaiy-run-")).count();
    {
        let server = Server::start(&scratch, &env);
        store_flow(&server);
        run_flow_on(&server, "killed", json!({ "hang": true }));
        let deadline = Instant::now() + Duration::from_secs(60);
        while !scratch.0.join("runs.log").exists() {
            assert!(Instant::now() < deadline, "the run did not start: {}", server.stderr_tail());
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    assert_eq!(leftovers(), 1, "the kill left the run's folder");
    let server = Server::start(&scratch, &env);
    let deadline = Instant::now() + Duration::from_secs(30);
    while leftovers() != 0 {
        assert!(Instant::now() < deadline, "the start did not sweep it: {}", server.stderr_tail());
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// `--help`, `--version` and a mistyped word used to start a real server, on whatever folder the environment named.
#[test]
fn only_no_arguments_at_all_starts_the_server() {
    let scratch = Scratch::new("arguments");
    let data = scratch.0.join("data");
    // (a port that is taken: a server this does start stops by itself, and the data folder says it ran)
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_oaiy-server"))
            .args(args)
            .env("OAIY_DATA_DIR", &data)
            .env("OAIY_SERVER_PORT", taken.local_addr().unwrap().port().to_string())
            .env("OAIY_VOICE_GATEWAY", "off")
            .output()
            .unwrap()
    };
    let help = run(&["--help"]);
    assert_eq!(help.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&help.stdout).contains("oaiy-server check"));
    let version = run(&["--version"]);
    assert_eq!(version.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&version.stdout).trim(), format!("oaiy-server {}", env!("CARGO_PKG_VERSION")));
    for words in [&["status"][..], &["--helpp"], &["--version", "x"], &["-h"], &[""]] {
        let used = run(words);
        assert_eq!(used.status.code(), Some(2), "{words:?}");
        assert!(String::from_utf8_lossy(&used.stderr).contains("usage:"), "{words:?}");
    }
    assert!(!data.exists(), "none of these touched the data folder");
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

    // A provider's API key, saved over the API as the dashboard does. With the web login the static token is the `cli`
    // preset, which does not hold `ai.admin` (the provider keys are among the routes it loses, as the README says), so
    // there the save is refused; the store is written by the same code in both builds and checked in the one without it.
    let (status, saved) = server.post(
        "/api/ai/providers",
        json!({ "id": "openai", "name": "OpenAI", "baseUrl": "https://api.openai.com" }),
    );
    if cfg!(feature = "web") {
        assert_eq!((status, saved["required"].as_str()), (403, Some("ai.admin")), "{saved}");
    } else {
        assert_eq!(status, 200, "{saved}");
        let (status, _) = server.post("/api/ai/providers/openai/key", json!({ "key": "sk-not-a-real-key" }));
        assert_eq!(status, 204);
        let store = data.join("ai").join("providers.json");
        assert!(std::fs::read_to_string(&store).unwrap().contains("sk-not-a-real-key"), "the key is what the file is for");
        assert_eq!(mode(&store), 0o600, "providers.json is readable by others");
        assert_eq!(mode(store.parent().unwrap()), 0o700, "and so is the folder the server made for it");
    }

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

/// The server's word that it may run an unsigned plugin (the wedged one is): a debug build always may, a release
/// build (`cargo test --release`) only with `OAIY_PLUGIN_DEV_MODE=1`, as a developer's would.
#[cfg(target_os = "linux")]
fn unsigned_plugins() -> [(&'static str, String); 1] {
    [("OAIY_PLUGIN_DEV_MODE", "1".to_string())]
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
    let mut server = Server::start(&scratch, &unsigned_plugins());

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
    // The plugin starts on a thread of its own while the API is being made ready, and a release build can find the port
    // taken before that thread has started it: such a run proves nothing, and is made again (up to five times).
    for attempt in 1..=5 {
        let scratch = Scratch::new(&format!("bind-fails-{attempt}"));
        install_wedged_plugin(&scratch);
        let taken = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = taken.local_addr().unwrap().port();
        let mut server = Server::spawn(&scratch, &unsigned_plugins(), port);

        let deadline = Instant::now() + Duration::from_secs(90);
        let status = loop {
            if let Some(status) = server.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "the server neither came up nor gave up");
            std::thread::sleep(Duration::from_millis(100));
        };
        assert_eq!(status.code(), Some(1), "it gives up: {}", server.stderr_tail());

        // It was up before the bind failed, or this run proved nothing.
        if std::fs::read_to_string(scratch.0.join("data").join("plugin-data").join("fake").join("plugin.pid")).is_err() {
            drop(taken);
            continue;
        }
        let (plugin, helper) = wedged_plugin_pids(&scratch);
        assert!(gone_within(&[plugin, helper], Duration::from_secs(15)), "the plugin or its helper outlived the server");
        drop(taken);
        return;
    }
    // (a machine busy enough that the bind failed first five times over: inconclusive, not the server's fault)
    eprintln!("inconclusive: the plugin had not started when the bind failed in five runs");
}
