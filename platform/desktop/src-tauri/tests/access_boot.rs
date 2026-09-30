//! The route table against the real router (design 4.3, "the boot test").
//!
//! The source walk in `auth::route_coverage` says every route has a row. This says the other half:
//! every row is *behind the guard*. It starts `oaiy-server` as the program it is, on a port the system
//! picks, and sends an anonymous request to every row of the table. A route merged after the guard's
//! `.layer(...)` (the comment at `http.rs` where the routers are merged warns about it) would answer
//! 200 or 405 to a stranger; every non-public row must be refused with 401, 403 or 421.
//!
//! The server runs with a data folder, home folder and environment of its own, the voice gateway (a
//! fixed port) switched off and a port the system picked, so it can never meet the desktop that may
//! be running on this machine.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use oaiy_desktop_lib::auth::routes::{Class, Route, Verb, ROUTES};

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
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let data = scratch.0.join("data");
        let stderr = scratch.0.join("server.stderr");
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
        let mut server = Server {
            child,
            port,
            stderr,
        };
        server.wait_until_up();
        server
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

    fn stderr_tail(&self) -> String {
        let text = std::fs::read_to_string(&self.stderr).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        lines[lines.len().saturating_sub(15)..].join("\n")
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
        let meant_open = (a.row.class == Class::Public && a.row.since == 1)
            || OPEN_IN_LEGACY.contains(&a.name().as_str());
        if meant_open {
            // A public route answers a stranger (health 200, an unknown pairing id 404, ...).
            if refused {
                refused_but_open.push(format!("{} -> {}", a.name(), a.status));
            }
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
}
