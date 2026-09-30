//! The route table and the access modes against the real router (design 4.3, "the boot test").
//!
//! The source walk in `auth::route_coverage` says every route has a row. This says the other half:
//! every row is *behind the guard*. It starts `oaiy-server` as the program it is, on a port the system
//! picks, and sends an anonymous request to every row of the table. A route merged after the guard's
//! `.layer(...)` (the comment at `http.rs` where the routers are merged warns about it) would answer
//! 200 or 405 to a stranger. In `legacy` mode every non-public row must be refused with 401, 403 or 421;
//! in `scoped` mode, with 401.
//!
//! The same program is also started with `OAIY_ACCESS_MODE` set to what it must refuse (a value that is
//! not a mode, `shadow` on a network address) and twice on one data folder, and must exit 78, the code
//! the shipped unit does not restart.
//!
//! The server runs with a data folder, home folder and environment of its own, the voice gateway (a
//! fixed port) switched off and a port the system picked, so it can never meet the desktop that may
//! be running on this machine.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use oaiy_desktop_lib::auth::routes::{Class, Route, Verb, ROUTES};
use serde_json::Value;

const TOKEN: &str = "integration-test-token";

/// A folder of the test's own, removed when the test ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("oaiy-access-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
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
    /// Start it and wait until it answers.
    fn start(scratch: &Scratch, extra_env: &[(&str, &str)]) -> Server {
        let mut server = Server::spawn(scratch, extra_env, "server");
        server.wait_until_up();
        server
    }

    /// Start it and leave it to come up (or not) by itself. `name` names its stderr file, so that two servers
    /// on one data folder can be told apart.
    fn spawn(scratch: &Scratch, extra_env: &[(&str, &str)], name: &str) -> Server {
        // A port the system picked, released a moment before the server takes it.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let data = scratch.data();
        let stderr = scratch.0.join(format!("{name}.stderr"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_oaiy-server"));
        // Nothing of this environment but what a program needs to run.
        command.env_clear();
        for name in [
            "SystemRoot",
            "SYSTEMROOT",
            "windir",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "LANG",
        ] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
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
        Server {
            child,
            port,
            stderr,
        }
    }

    fn wait_until_up(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "oaiy-server exited early ({status}): {}",
                    self.stderr_tail()
                );
            }
            let up = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .ok()
                .and_then(|c| {
                    c.get(format!("http://127.0.0.1:{}/api/health", self.port))
                        .send()
                        .ok()
                })
                .is_some_and(|r| r.status().as_u16() == 200);
            if up {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "oaiy-server did not come up: {}",
                self.stderr_tail()
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Wait for it to exit and return its exit code.
    fn exit_code(&mut self, within: Duration) -> Option<i32> {
        let deadline = Instant::now() + within;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status.code();
            }
            if Instant::now() > deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn stderr_tail(&self) -> String {
        let text = std::fs::read_to_string(&self.stderr).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        lines[lines.len().saturating_sub(15)..].join("\n")
    }

    /// One request, with an optional bearer, returning the status and the JSON body.
    fn call(&self, method: reqwest::Method, path: &str, token: Option<&str>) -> (u16, Value) {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let mut request = client
            .request(method, format!("http://127.0.0.1:{}{path}", self.port))
            .header("content-type", "application/json")
            .body("{}");
        if let Some(t) = token {
            request = request.bearer_auth(t);
        }
        let response = request.send().unwrap_or_else(|e| panic!("{path}: {e}"));
        let status = response.status().as_u16();
        (
            status,
            serde_json::from_str(&response.text().unwrap_or_default()).unwrap_or(Value::Null),
        )
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A concrete path for a route pattern: every `:param` and `*wildcard` segment becomes `x`.
fn concrete(pattern: &str) -> String {
    pattern
        .split('/')
        .map(|s| {
            if s.starts_with(':') || s.starts_with('*') {
                "x"
            } else {
                s
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// The methods a row is asked with.
fn methods(verb: Verb) -> Vec<reqwest::Method> {
    match verb {
        Verb::Get => vec![reqwest::Method::GET],
        Verb::Post => vec![reqwest::Method::POST],
        Verb::Put => vec![reqwest::Method::PUT],
        Verb::Patch => vec![reqwest::Method::PATCH],
        Verb::Delete => vec![reqwest::Method::DELETE],
        Verb::Any => vec![
            reqwest::Method::GET,
            reqwest::Method::POST,
            reqwest::Method::PUT,
            reqwest::Method::PATCH,
            reqwest::Method::DELETE,
        ],
    }
}

/// Routes the `legacy` guard has always left open on a headless server, though the table asks for a
/// scope: `GET /api/update/status` is "open like health" there (`http.rs`, `UPDATE_STATUS_PATH`).
/// The new guard does not keep this exemption; when it is on, the row is refused like the others.
const OPEN_IN_LEGACY: &[&str] = &["GET /api/update/status"];

/// What one anonymous request to a row got back.
struct Answer {
    row: &'static Route,
    method: reqwest::Method,
    status: u16,
}

impl Answer {
    fn name(&self) -> String {
        format!("{} {}", self.method, self.row.pattern)
    }
}

/// Send every row of the table to a server, anonymously, and return what came back for each.
fn probe_every_row(server: &Server) -> Vec<Answer> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let mut answers = Vec::new();
    for row in ROUTES {
        for method in methods(row.method) {
            let url = format!("http://127.0.0.1:{}{}", server.port, concrete(row.pattern));
            let response = client
                .request(method.clone(), &url)
                .header("content-type", "application/json")
                .body("{}")
                .send()
                .unwrap_or_else(|e| panic!("{method} {url}: {e}"));
            answers.push(Answer {
                row,
                method,
                status: response.status().as_u16(),
            });
        }
    }
    answers
}

#[test]
fn every_row_of_the_table_is_behind_the_guard_in_legacy_mode() {
    let scratch = Scratch::new("boot-legacy");
    let server = Server::start(&scratch, &[]);
    let answers = probe_every_row(&server);
    assert!(
        answers.len() > 250,
        "the probe covered {} requests",
        answers.len()
    );

    let mut refused_but_open = Vec::new();
    let mut leaked = Vec::new();
    for a in &answers {
        let refused = matches!(a.status, 401 | 403 | 421);
        let existed_and_open = a.row.class == Class::Public && a.row.since == 1;
        let legacy_open = OPEN_IN_LEGACY.contains(&a.name().as_str());
        // The one route of the model that exists and is public (`GET /api/auth/info`): a stranger is answered.
        let built_and_public =
            a.row.class == Class::Public && a.row.since == 2 && a.row.pattern == "/api/auth/info";
        if existed_and_open || legacy_open || built_and_public {
            // A public route answers a stranger (health 200, an unknown pairing id 404, ...).
            if refused {
                refused_but_open.push(format!("{} -> {}", a.name(), a.status));
            }
        } else if a.row.class == Class::Public {
            // A public route of the model that is built by a later step: no route, so the old guard answers.
            assert!(refused || a.status == 404, "{} -> {}", a.name(), a.status);
        } else if !refused {
            leaked.push(format!("{} -> {}", a.name(), a.status));
        }
    }
    assert!(
        leaked.is_empty(),
        "rows a stranger got past the guard (a route merged outside it?):\n{}",
        leaked.join("\n")
    );
    assert!(
        refused_but_open.is_empty(),
        "public rows the guard refused:\n{}",
        refused_but_open.join("\n")
    );
    // The exemption is real today, so the list above is not stale.
    for name in OPEN_IN_LEGACY {
        let a = answers.iter().find(|a| a.name() == *name).unwrap();
        assert_eq!(a.status, 200, "{name}");
    }
    // Nothing of the access model was written to the data folder: `legacy` changes nothing on disk.
    assert!(
        !scratch.data().join("auth").exists(),
        "legacy mode made <data>/auth"
    );
}

#[test]
fn a_legacy_server_answers_what_it_always_did_and_health_says_the_mode() {
    let scratch = Scratch::new("boot-legacy-health");
    let server = Server::start(&scratch, &[]);
    // The old refusal, byte for byte.
    let (status, body) = server.call(reqwest::Method::GET, "/api/config", None);
    assert_eq!(
        (status, body["error"].as_str()),
        (403, Some("authentication required"))
    );
    // The token opens it, as before.
    let (status, _) = server.call(reqwest::Method::GET, "/api/config", Some(TOKEN));
    assert_eq!(status, 200);
    // Health: the fields it always had, and two more; the version of the API is unchanged.
    let (status, health) = server.call(reqwest::Method::GET, "/api/health", None);
    assert_eq!(status, 200);
    assert_eq!(
        (
            health["apiVersion"].as_u64(),
            health["access"].as_str(),
            health["storage"].as_str()
        ),
        (Some(1), Some("legacy"), Some("ok"))
    );
    assert_eq!(health["product"], "oaiy-desktop");
    // The new routes exist and are authenticated: whoami answers the token as the `cli` preset, never as the owner.
    assert_eq!(
        server
            .call(reqwest::Method::GET, "/api/auth/whoami", None)
            .0,
        401
    );
    let (status, me) = server.call(reqwest::Method::GET, "/api/auth/whoami", Some(TOKEN));
    assert_eq!(
        (
            status,
            me["kind"].as_str(),
            me["scopes"].as_array().map(Vec::len)
        ),
        (200, Some("static"), Some(15))
    );
    assert!(!scratch.data().join("auth").exists());
}

#[test]
fn every_row_of_the_table_is_behind_the_guard_in_scoped_mode() {
    let scratch = Scratch::new("boot-scoped");
    let server = Server::start(&scratch, &[("OAIY_ACCESS_MODE", "scoped")]);
    let answers = probe_every_row(&server);
    let mut answered = Vec::new();
    for a in &answers {
        let public_and_built =
            a.row.class == Class::Public && (a.row.since == 1 || a.row.pattern == "/api/auth/info");
        if public_and_built {
            if matches!(a.status, 401 | 403 | 421) {
                answered.push(format!("public {} -> {}", a.name(), a.status));
            }
        } else if a.status != 401 {
            // Every other row, the update status included (no legacy exemption), is a plain 401 to a stranger.
            answered.push(format!("{} -> {}", a.name(), a.status));
        }
    }
    assert!(
        answered.is_empty(),
        "rows a stranger was not refused with 401:\n{}",
        answered.join("\n")
    );
    // A path that is not there is refused the same way: nothing tells a stranger what exists.
    assert_eq!(
        server
            .call(reqwest::Method::GET, "/api/no/such/route", None)
            .0,
        401
    );
    // The auth folder exists, locked, with the startup event and no credential file yet.
    let auth = scratch.data().join("auth");
    assert!(auth.join(".lock").exists() && auth.join("audit.jsonl").exists());
    let audit = std::fs::read_to_string(auth.join("audit.jsonl")).unwrap();
    assert!(
        audit.contains("\"startup\"") && audit.contains("scoped"),
        "{audit}"
    );
}

#[test]
fn a_scoped_server_knows_the_environment_token_as_the_cli_preset_and_nothing_more() {
    let scratch = Scratch::new("boot-scoped-token");
    let server = Server::start(&scratch, &[("OAIY_ACCESS_MODE", "scoped")]);
    let get = |path: &str, token: Option<&str>| server.call(reqwest::Method::GET, path, token);
    // A wrong token, and the old kinds of credential, are not credentials.
    assert_eq!(
        get("/api/config", Some("wrong")).1["error"]["code"],
        "token_invalid"
    );
    // The `cli` preset reaches what it holds.
    assert_eq!(get("/api/config", Some(TOKEN)).0, 200);
    assert_eq!(get("/api/services", Some(TOKEN)).0, 200);
    // And is refused, by name, what it does not: a dangerous scope, the auth routes.
    let (status, body) = server.call(reqwest::Method::POST, "/api/services", Some(TOKEN));
    assert_eq!(
        (
            status,
            body["error"]["code"].as_str(),
            body["required"].as_str()
        ),
        (403, Some("insufficient_scope"), Some("services.define"))
    );
    let (status, body) = get("/api/bridge/pairing", Some(TOKEN));
    assert_eq!(
        (status, body["required"].as_str()),
        (403, Some("auth.read"))
    );
    // whoami and derive: a child of at most what the token holds.
    let (status, me) = get("/api/auth/whoami", Some(TOKEN));
    assert_eq!(
        (status, me["kind"].as_str(), me["elevated"].as_bool()),
        (200, Some("static"), Some(false))
    );
    let client = reqwest::blocking::Client::new();
    let derived: Value = client
        .post(format!("http://127.0.0.1:{}/api/auth/derive", server.port))
        .bearer_auth(TOKEN)
        .header("content-type", "application/json")
        .body(r#"{"scopes":["services.read"],"ttlSeconds":600,"label":"boot test"}"#)
        .send()
        .unwrap()
        .json()
        .unwrap();
    let child = derived["token"].as_str().expect("a token").to_string();
    let (status, me) = get("/api/auth/whoami", Some(&child));
    assert_eq!(
        (
            status,
            me["kind"].as_str(),
            me["scopes"].as_array().map(Vec::len)
        ),
        (200, Some("run"), Some(1))
    );
    assert_eq!(get("/api/services", Some(&child)).0, 200);
    assert_eq!(
        get("/api/config", Some(&child)).0,
        403,
        "a child holds only what it was given"
    );
    // The mode is on health.
    let (_, health) = get("/api/health", None);
    assert_eq!(
        (health["access"].as_str(), health["apiVersion"].as_u64()),
        (Some("scoped"), Some(1))
    );
    // Nothing secret was written to the audit log: not the environment token, not the derived one.
    let audit = std::fs::read_to_string(scratch.data().join("auth").join("audit.jsonl")).unwrap();
    assert!(audit.contains("credential.created"), "{audit}");
    assert!(
        !audit.contains(TOKEN) && !audit.contains(&child),
        "no secret in the audit log"
    );
    let stderr = std::fs::read_to_string(&server.stderr).unwrap_or_default();
    assert!(
        !stderr.contains(TOKEN) && !stderr.contains(&child),
        "no secret on stderr"
    );
}

#[test]
fn a_scoped_server_sees_the_peer_and_the_host_of_each_request_from_the_socket() {
    // The guard needs the peer address (`into_make_service_with_connect_info`): without it a request
    // would look like one made by code in the process, and the address checks would be skipped.
    let scratch = Scratch::new("boot-scoped-address");
    let server = Server::start(&scratch, &[("OAIY_ACCESS_MODE", "scoped")]);
    let client = reqwest::blocking::Client::new();
    let get = |path: &str, headers: &[(&str, &str)]| {
        let mut request = client
            .get(format!("http://127.0.0.1:{}{path}", server.port))
            .bearer_auth(TOKEN);
        for (n, v) in headers {
            request = request.header(*n, *v);
        }
        let response = request.send().unwrap();
        let status = response.status().as_u16();
        (status, response.json::<Value>().unwrap_or(Value::Null))
    };
    // A forwarded header on a local install: the request came through something that is not configured.
    for header in ["x-forwarded-for", "x-real-ip", "via"] {
        let (status, body) = get("/api/config", &[(header, "203.0.113.9")]);
        assert_eq!(
            (status, body["error"]["code"].as_str()),
            (421, Some("proxy_detected")),
            "{header}"
        );
    }
    // A Host that is not this server's (DNS rebinding).
    let (status, body) = get("/api/config", &[("host", "evil.example:17972")]);
    assert_eq!(
        (status, body["error"]["code"].as_str()),
        (421, Some("misdirected_host"))
    );
    // `info` says what the server saw: the peer, and no proxy.
    let (status, info) = get("/api/auth/info", &[]);
    assert_eq!(
        (
            status,
            info["seen"]["clientIp"].as_str(),
            info["seen"]["viaTrustedProxy"].as_bool()
        ),
        (200, Some("127.0.0.1"), Some(false))
    );
    // A browser page on another origin holding the token is refused: an unbound token, with an Origin.
    let (status, body) = get("/api/config", &[("origin", "http://localhost:3000")]);
    assert_eq!(
        (status, body["error"]["code"].as_str()),
        (403, Some("origin_mismatch"))
    );
}

#[test]
fn a_bare_options_on_a_scoped_server_runs_no_handler_and_tells_no_path_from_another() {
    // The engine gateway is an `any` route: its handler runs for every method, and looks for the engine. A bare
    // `OPTIONS` (no Origin, no requested method) must not get that far, and its answer must be the same for a
    // path that exists and one that does not.
    let scratch = Scratch::new("boot-scoped-options");
    let server = Server::start(&scratch, &[("OAIY_ACCESS_MODE", "scoped")]);
    let client = reqwest::blocking::Client::new();
    let options = |path: &str, token: Option<&str>| {
        let mut request = client.request(
            reqwest::Method::OPTIONS,
            format!("http://127.0.0.1:{}{path}", server.port),
        );
        if let Some(t) = token {
            request = request.bearer_auth(t);
        }
        let response = request.send().unwrap();
        let status = response.status().as_u16();
        let mut headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter(|(n, _)| n.as_str() != "date")
            .map(|(n, v)| (n.as_str().to_owned(), v.to_str().unwrap_or("?").to_owned()))
            .collect();
        headers.sort();
        (status, headers, response.text().unwrap())
    };
    for token in [None, Some(TOKEN)] {
        let gateway = options("/api/ai/engine/gateway/x", token);
        assert_eq!(gateway.0, 204, "{gateway:?}");
        assert_eq!(gateway.2, "");
        for path in [
            "/api/services",
            "/api/no/such/route",
            "/api/ai/engine/gateway",
        ] {
            assert_eq!(options(path, token), gateway, "{path}");
        }
        assert!(
            gateway.1.iter().all(|(n, v)| n != "allow" || v.is_empty()),
            "{gateway:?}"
        );
    }
    // A public route is passed on to its router.
    assert_eq!(options("/api/health", None).0, 405);
}

#[test]
fn a_mode_that_is_not_one_stops_the_server_with_exit_78() {
    let scratch = Scratch::new("boot-badmode");
    let mut server = Server::spawn(&scratch, &[("OAIY_ACCESS_MODE", "scopd")], "server");
    assert_eq!(
        server.exit_code(Duration::from_secs(60)),
        Some(78),
        "{}",
        server.stderr_tail()
    );
    assert!(
        server.stderr_tail().contains("OAIY_ACCESS_MODE"),
        "{}",
        server.stderr_tail()
    );
}

#[test]
fn shadow_on_a_network_address_stops_the_server_with_exit_78() {
    let scratch = Scratch::new("boot-shadow-lan");
    let mut server = Server::spawn(
        &scratch,
        &[("OAIY_ACCESS_MODE", "shadow"), ("OAIY_SERVER_BIND", "lan")],
        "server",
    );
    assert_eq!(
        server.exit_code(Duration::from_secs(60)),
        Some(78),
        "{}",
        server.stderr_tail()
    );
    assert!(
        server.stderr_tail().contains("shadow"),
        "{}",
        server.stderr_tail()
    );
    // Nothing was opened on the way to refusing.
    assert!(!scratch.data().join("auth").exists());
}

#[test]
fn a_second_scoped_server_on_the_data_folder_is_refused_with_exit_78() {
    let scratch = Scratch::new("boot-lock");
    let _first = Server::start(&scratch, &[("OAIY_ACCESS_MODE", "scoped")]);
    let mut second = Server::spawn(&scratch, &[("OAIY_ACCESS_MODE", "scoped")], "second");
    assert_eq!(
        second.exit_code(Duration::from_secs(60)),
        Some(78),
        "{}",
        second.stderr_tail()
    );
    assert!(
        second.stderr_tail().contains("in use by process"),
        "{}",
        second.stderr_tail()
    );
}
