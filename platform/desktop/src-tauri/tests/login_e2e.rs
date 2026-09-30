#![cfg(feature = "web")]
//! The web login of `oaiy-server` end to end: the real program, on a port the system picked, over a data folder of
//! its own, driven over real HTTP, and the console (`oaiy-server auth ...`) run as a real process against the running
//! server and against a stopped one.
//!
//! What only a real process shows: the banner on stderr (no secret in it), the console credential's files (0600,
//! rewritten at every start), the console reaching the running server through them, a console edit of the files of a
//! stopped server that the next start sees, the startup error of an unreadable owner file (exit 78, naming it),
//! and that no secret ends up on stderr or in the audit log.
//!
//! The server has a home folder, data folder and environment of its own, the voice gateway (a fixed port) off, and a
//! port the system picked, so it can never meet the desktop that may be running on this machine.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const PASSWORD: &str = "k7Qz!mV3#pW9xLd2 rn8Tb";
const NEW_PASSWORD: &str = "Hv4$wN6@cJ1&zX8 qs5Fe";
const OFFLINE_PASSWORD: &str = "Zt2%qP7^mB4*vC9 yh3Kd";

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("oaiy-login-e2e-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
    }

    fn auth(&self) -> PathBuf {
        self.data().join("auth")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The environment of a program of ours: nothing of this machine's but what it needs to run.
fn command(scratch: &Scratch, extra_env: &[(&str, &str)]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oaiy-server"));
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
        .env("OAIY_DATA_DIR", scratch.data())
        .env("OAIY_MODELS_DIR", scratch.data().join("models"))
        .env("OAIY_VOICE_GATEWAY", "off");
    for (name, value) in extra_env {
        command.env(name, value);
    }
    command
}

struct Server {
    child: Child,
    port: u16,
    stderr: PathBuf,
}

impl Server {
    fn spawn(scratch: &Scratch, extra_env: &[(&str, &str)], name: &str) -> Server {
        Server::spawn_on(scratch, extra_env, name, None)
    }

    /// On this port when one is given (a restart on the port it had), else on one the system picked.
    fn spawn_on(
        scratch: &Scratch,
        extra_env: &[(&str, &str)],
        name: &str,
        port: Option<u16>,
    ) -> Server {
        let port = port.unwrap_or_else(|| {
            TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        });
        let stderr = scratch.0.join(format!("{name}.stderr"));
        let child = command(scratch, extra_env)
            .env("OAIY_SERVER_PORT", port.to_string())
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

    fn start(scratch: &Scratch, name: &str) -> Server {
        let mut server = Server::spawn(scratch, &[], name);
        server.wait_until_up();
        server
    }

    fn wait_until_up(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "oaiy-server exited early ({status}): {}",
                    self.stderr_text()
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
            // The console's files are written once the listener has bound: wait for them too.
            if up && self.console_files_exist() {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "oaiy-server did not come up: {}",
                self.stderr_text()
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn console_files_exist(&self) -> bool {
        let dir = self.stderr.parent().unwrap().join("data").join("auth");
        dir.join("console.token").exists() && dir.join("console.json").exists()
    }

    fn stderr_text(&self) -> String {
        std::fs::read_to_string(&self.stderr).unwrap_or_default()
    }

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

    /// Stop it the hard way and wait: the lock is released with the process.
    fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// The dashboard's name on this loopback server, as a browser has it.
    fn dash_host(&self) -> String {
        format!("dash.oaiy.localhost:{}", self.port)
    }

    /// A request as the dashboard page makes it: the Host and Origin of the dashboard name, over the loopback socket.
    fn page(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
        cookie: Option<&str>,
        csrf: Option<&str>,
    ) -> Reply {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        let mut request = client
            .request(method, format!("http://127.0.0.1:{}{path}", self.port))
            .header("host", self.dash_host())
            .header("origin", format!("http://{}", self.dash_host()))
            .header("sec-fetch-site", "same-origin")
            .header("user-agent", "login-e2e");
        if let Some(c) = cookie {
            request = request.header("cookie", c);
        }
        if let Some(c) = csrf {
            request = request.header("x-oaiy-csrf", c);
        }
        if let Some(b) = body {
            request = request
                .header("content-type", "application/json")
                .body(b.to_string());
        }
        let response = request.send().unwrap_or_else(|e| panic!("{path}: {e}"));
        let status = response.status().as_u16();
        let cookies: Vec<String> = response
            .headers()
            .get_all("set-cookie")
            .iter()
            .map(|v| v.to_str().unwrap().to_string())
            .collect();
        let text = response.text().unwrap_or_default();
        Reply {
            status,
            cookies,
            body: serde_json::from_str(&text).unwrap_or(Value::Null),
            text,
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Reply {
    status: u16,
    cookies: Vec<String>,
    body: Value,
    text: String,
}

impl Reply {
    fn cookie_value(&self, name: &str) -> Option<String> {
        self.cookies.iter().find_map(|c| {
            c.strip_prefix(&format!("{name}="))
                .map(|rest| rest.split(';').next().unwrap_or("").to_string())
        })
    }
}

/// What the console printed.
struct Console {
    code: i32,
    out: String,
    err: String,
}

fn console(scratch: &Scratch, args: &[&str]) -> Console {
    let output = command(scratch, &[])
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("run the console");
    Console {
        code: output.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&output.stdout).into_owned(),
        err: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn password_file(scratch: &Scratch, password: &str) -> PathBuf {
    let path = scratch.0.join("pw.txt");
    std::fs::write(&path, format!("{password}\n")).unwrap();
    path
}

fn setup_code_of(c: &Console) -> String {
    c.out
        .lines()
        .find_map(|l| l.strip_prefix("Setup code: "))
        .unwrap_or_else(|| panic!("no code in: {} / {}", c.out, c.err))
        .trim()
        .to_string()
}

fn login(server: &Server, password: &str) -> Reply {
    server.page(
        reqwest::Method::POST,
        "/api/auth/login",
        Some(json!({ "password": password })),
        None,
        None,
    )
}

struct Browser {
    cookie: String,
    csrf: String,
}

fn browser(server: &Server, r: &Reply) -> Browser {
    let name = format!("oaiy_dash_{}", server.port);
    Browser {
        cookie: format!(
            "{name}={}",
            r.cookie_value(&name).expect("a session cookie")
        ),
        csrf: r.body["csrf"].as_str().expect("a csrf value").to_string(),
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn a_running_server_is_set_up_signed_in_to_and_administered_through_its_console() {
    let scratch = Scratch::new("run");
    let mut server = Server::start(&scratch, "server");
    let port = server.port;

    // The banner is on stderr, says what to run, and has no secret in it.
    let banner = server.stderr_text();
    assert!(
        banner.contains("no owner login yet") && banner.contains("oaiy-server auth setup-code"),
        "{banner}"
    );
    assert!(
        banner.contains(&format!("http://dash.oaiy.localhost:{port}/setup")),
        "{banner}"
    );

    // The console credential's files: 0600, a token that is the console's and a port that is this server's.
    let auth = scratch.auth();
    let token_text = read(&auth.join("console.token"));
    assert!(
        token_text.trim().starts_with("oaiycon_") && token_text.trim().len() == 68,
        "{token_text}"
    );
    let info: Value = serde_json::from_str(&read(&auth.join("console.json"))).unwrap();
    assert_eq!(info["port"], port);
    assert_eq!(info["pid"], server.child.id());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for f in ["console.token", "console.json", "owner.json"] {
            if auth.join(f).exists() {
                assert_eq!(
                    std::fs::metadata(auth.join(f))
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777,
                    0o600,
                    "{f}"
                );
            }
        }
    }

    // Before setup: login is 409 setup_required, and info says there is no login and no code.
    let r = login(&server, PASSWORD);
    assert_eq!(
        (r.status, r.body["error"]["code"].as_str()),
        (409, Some("setup_required")),
        "{}",
        r.text
    );
    let info = server
        .page(reqwest::Method::GET, "/api/auth/info", None, None, None)
        .body;
    assert_eq!(
        (
            info["loginConfigured"].as_bool(),
            info["setupCode"].as_str()
        ),
        (Some(false), Some("none"))
    );

    // The console makes the code through the running server, and prints it only there.
    let made = console(&scratch, &["auth", "setup-code"]);
    assert_eq!(made.code, 0, "{} / {}", made.out, made.err);
    assert!(made.out.contains("data folder:"), "{}", made.out);
    let code = setup_code_of(&made);
    assert_eq!(code.len(), 14);
    let info = server
        .page(reqwest::Method::GET, "/api/auth/info", None, None, None)
        .body;
    assert_eq!(info["setupCode"], "active");
    assert!(
        !server.stderr_text().contains(&code),
        "the code is never on the server's stderr"
    );

    // Setup in the browser's way: the code and a password; a wrong code first.
    let wrong = server.page(
        reqwest::Method::POST,
        "/api/auth/setup",
        Some(json!({ "code": "AAAA-AAAA-AAAA", "password": PASSWORD })),
        None,
        None,
    );
    assert_eq!(
        (wrong.status, wrong.body["attemptsLeft"].as_u64()),
        (401, Some(99))
    );
    let r = server.page(
        reqwest::Method::POST,
        "/api/auth/setup",
        Some(json!({ "code": code, "password": PASSWORD })),
        None,
        None,
    );
    assert_eq!(r.status, 201, "{}", r.text);
    let first = browser(&server, &r);
    let name = format!("oaiy_dash_{port}");
    assert!(
        r.cookies[0].starts_with(&format!("{name}=oaiyses_"))
            && r.cookies[0].ends_with("; Path=/; HttpOnly; SameSite=Strict"),
        "{:?}",
        r.cookies
    );

    // The session works, with its CSRF; without it a mutation does not.
    let read_config = server.page(
        reqwest::Method::GET,
        "/api/config",
        None,
        Some(&first.cookie),
        None,
    );
    assert_eq!(read_config.status, 200, "{}", read_config.text);
    let no_csrf = server.page(
        reqwest::Method::POST,
        "/api/auth/logout",
        None,
        Some(&first.cookie),
        None,
    );
    assert_eq!(
        (no_csrf.status, no_csrf.body["error"]["code"].as_str()),
        (403, Some("csrf"))
    );

    // The console reads the running server's status: no secret in it.
    let status = console(&scratch, &["auth", "status"]);
    assert_eq!(status.code, 0, "{} / {}", status.out, status.err);
    for want in ["server: running", "login configured: yes", "sessions: 1"] {
        assert!(status.out.contains(want), "{want} missing: {}", status.out);
    }
    assert!(!status.out.contains(&first.cookie) && !status.out.contains(&first.csrf));

    // A session link made by the console (the incident lifeline), spent in the browser's way.
    let link = console(&scratch, &["auth", "session-link"]);
    assert_eq!(link.code, 0, "{} / {}", link.out, link.err);
    let url = link
        .out
        .lines()
        .find(|l| l.trim().starts_with("http"))
        .unwrap()
        .trim()
        .to_string();
    let fragment = url
        .split_once("/auth/link#")
        .expect("the code is in the fragment")
        .1
        .to_string();
    assert_eq!(fragment.len(), 43);
    assert!(
        url.starts_with(&format!("http://dash.oaiy.localhost:{port}/auth/link#")),
        "{url}"
    );
    let spent = server.page(
        reqwest::Method::POST,
        "/api/auth/link",
        Some(json!({ "code": fragment })),
        None,
        None,
    );
    assert_eq!(spent.status, 200, "{}", spent.text);
    assert_eq!(
        spent.body["elevatedUntilMs"], 0,
        "a link is not an elevation"
    );
    let again = server.page(
        reqwest::Method::POST,
        "/api/auth/link",
        Some(json!({ "code": fragment })),
        None,
        None,
    );
    assert_eq!(again.status, 400);

    // Logout, and login again with the password: the device cookie made at setup routes it to the reserved lane.
    let out = server.page(
        reqwest::Method::POST,
        "/api/auth/logout",
        None,
        Some(&first.cookie),
        Some(&first.csrf),
    );
    assert_eq!(out.status, 204);
    let dead = server.page(
        reqwest::Method::GET,
        "/api/config",
        None,
        Some(&first.cookie),
        None,
    );
    assert_eq!(
        (dead.status, dead.body["reason"].as_str()),
        (401, Some("logged_out"))
    );
    let device = format!(
        "oaiy_dev_{port}={}",
        r.cookie_value(&format!("oaiy_dev_{port}"))
            .expect("a device cookie")
    );
    let second = {
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("http://127.0.0.1:{port}/api/auth/login"))
            .header("host", server.dash_host())
            .header("origin", format!("http://{}", server.dash_host()))
            .header("cookie", &device)
            .header("content-type", "application/json")
            .body(json!({ "password": PASSWORD }).to_string())
            .send()
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        assert!(
            response
                .headers()
                .get_all("set-cookie")
                .iter()
                .all(|c| !c.to_str().unwrap().starts_with("oaiy_dev_")),
            "a known device is not given another cookie"
        );
        let cookies: Vec<String> = response
            .headers()
            .get_all("set-cookie")
            .iter()
            .map(|v| v.to_str().unwrap().to_string())
            .collect();
        let body: Value = response.json().unwrap();
        Browser {
            cookie: cookies[0].split(';').next().unwrap().to_string(),
            csrf: body["csrf"].as_str().unwrap().to_string(),
        }
    };

    // The console resets the password on the running server: every session and device dies at once.
    let pw = password_file(&scratch, NEW_PASSWORD);
    let reset = console(
        &scratch,
        &[
            "auth",
            "reset-password",
            "--password-file",
            pw.to_str().unwrap(),
        ],
    );
    assert_eq!(reset.code, 0, "{} / {}", reset.out, reset.err);
    assert!(
        reset.out.contains("every session and device is revoked"),
        "{}",
        reset.out
    );
    assert!(!reset.out.contains(NEW_PASSWORD) && !reset.err.contains(NEW_PASSWORD));
    let after = server.page(
        reqwest::Method::GET,
        "/api/config",
        None,
        Some(&second.cookie),
        None,
    );
    assert_eq!(
        (after.status, after.body["reason"].as_str()),
        (401, Some("password_changed"))
    );
    assert_eq!(
        login(&server, PASSWORD).status,
        401,
        "the old password fails at once"
    );
    let fresh = login(&server, NEW_PASSWORD);
    assert_eq!(fresh.status, 200, "{}", fresh.text);

    // The canary sweep: no secret on the server's stderr, in its audit log, or in the noise log.
    let secrets = [
        PASSWORD,
        NEW_PASSWORD,
        code.as_str(),
        token_text.trim(),
        fragment.as_str(),
        first.csrf.as_str(),
    ];
    let stderr = server.stderr_text();
    let audit = read(&auth.join("audit.jsonl"));
    let noise = read(&auth.join("noise.jsonl"));
    for s in secrets {
        assert!(!stderr.contains(s), "stderr holds a secret: {s}");
        assert!(!audit.contains(s), "audit.jsonl holds a secret: {s}");
        assert!(!noise.contains(s), "noise.jsonl holds a secret: {s}");
    }
    for event in [
        "setup.ok",
        "login.ok",
        "logout",
        "link.issued",
        "link.used",
        "password.changed",
        "console.command",
    ] {
        assert!(audit.contains(event), "{event} not in the audit log");
    }
    assert!(audit.contains("\"command\":\"reset-password\"") || audit.contains("reset-password"));
    let _ = server.exit_code(Duration::from_millis(1));
}

#[test]
fn a_stopped_server_is_administered_under_the_lock_and_the_next_start_sees_it() {
    let scratch = Scratch::new("stopped");
    let server = Server::start(&scratch, "first");
    // Make the owner through the running server, then stop it the hard way.
    let made = console(&scratch, &["auth", "setup-code"]);
    let code = setup_code_of(&made);
    let r = server.page(
        reqwest::Method::POST,
        "/api/auth/setup",
        Some(json!({ "code": code, "password": PASSWORD })),
        None,
        None,
    );
    assert_eq!(r.status, 201, "{}", r.text);
    let session = browser(&server, &r);
    let port = server.port;
    server.kill();

    // The console finds the lock free, says the server is not running, and edits the files.
    let status = console(&scratch, &["auth", "status"]);
    assert_eq!(status.code, 0, "{} / {}", status.out, status.err);
    assert!(
        status.out.contains("not running") && status.out.contains("login configured: yes"),
        "{}",
        status.out
    );
    // A setup code is for the first owner only.
    let refused = console(&scratch, &["auth", "setup-code"]);
    assert_eq!(refused.code, 1);
    assert!(refused.err.contains("already exists"), "{}", refused.err);
    // A session link is the running server's.
    assert_eq!(console(&scratch, &["auth", "session-link"]).code, 1);
    // A password reset offline: revokes every session and device in the files.
    let pw = password_file(&scratch, OFFLINE_PASSWORD);
    let reset = console(
        &scratch,
        &[
            "auth",
            "reset-password",
            "--password-file",
            pw.to_str().unwrap(),
        ],
    );
    assert_eq!(reset.code, 0, "{} / {}", reset.out, reset.err);

    // The next start sees the new password and the revoked session.
    // On the same port, so that the old session's cookie has the name this server reads.
    let mut server = Server::spawn_on(&scratch, &[], "second", Some(port));
    server.wait_until_up();
    assert!(
        !server.stderr_text().contains("no owner login yet"),
        "an owner exists: no banner: {}",
        server.stderr_text()
    );
    let dead = server.page(
        reqwest::Method::GET,
        "/api/config",
        None,
        Some(&session.cookie),
        None,
    );
    assert_eq!(
        (dead.status, dead.body["reason"].as_str()),
        (401, Some("password_changed")),
        "the offline reset revoked the session: {}",
        dead.text
    );
    let old = server.page(
        reqwest::Method::POST,
        "/api/auth/login",
        Some(json!({ "password": PASSWORD })),
        None,
        None,
    );
    assert_eq!(old.status, 401);
    let new = server.page(
        reqwest::Method::POST,
        "/api/auth/login",
        Some(json!({ "password": OFFLINE_PASSWORD })),
        None,
        None,
    );
    assert_eq!(new.status, 200, "{}", new.text);
    let _ = server.exit_code(Duration::from_millis(1));
}

#[test]
fn a_console_for_a_folder_that_is_not_the_servers_makes_nothing_and_init_makes_the_owner_of_a_new_one(
) {
    let scratch = Scratch::new("init");
    for args in [
        &["auth", "setup-code"][..],
        &["auth", "status"],
        &["auth", "sessions", "revoke-all"],
        &["auth", "token", "list"],
    ] {
        let c = console(&scratch, args);
        assert_eq!(c.code, 1, "{args:?}: {} / {}", c.out, c.err);
        assert!(c.out.contains("data folder:"), "{args:?}: {}", c.out);
    }
    assert!(
        !scratch.data().exists(),
        "no console command but init makes a folder"
    );
    let pw = password_file(&scratch, PASSWORD);
    std::fs::create_dir_all(scratch.data()).unwrap();
    let init = console(
        &scratch,
        &["auth", "init", "--password-file", pw.to_str().unwrap()],
    );
    assert_eq!(init.code, 0, "{} / {}", init.out, init.err);
    assert!(scratch.auth().join("owner.json").exists());
    // The server that starts on it has an owner and no banner, and the password works.
    let mut server = Server::spawn(&scratch, &[], "server");
    server.wait_until_up();
    assert!(
        !server.stderr_text().contains("no owner login yet"),
        "{}",
        server.stderr_text()
    );
    assert_eq!(login(&server, PASSWORD).status, 200);
    // `init` again: refused without --force.
    let again = console(
        &scratch,
        &["auth", "init", "--password-file", pw.to_str().unwrap()],
    );
    assert_eq!(again.code, 1, "{} / {}", again.out, again.err);
    assert!(again.err.contains("already exists"), "{}", again.err);
    let _ = server.exit_code(Duration::from_millis(1));
}

#[test]
fn an_unreadable_or_mangled_owner_file_stops_the_server_with_exit_78_naming_it() {
    for (tag, make) in [("mangled", 0), ("directory", 1), ("newer", 2)] {
        let scratch = Scratch::new(&format!("bad-{tag}"));
        std::fs::create_dir_all(scratch.auth()).unwrap();
        let owner = scratch.auth().join("owner.json");
        match make {
            0 => std::fs::write(&owner, "{ not json").unwrap(),
            // A directory where the file belongs: a read error that is not "not there".
            1 => std::fs::create_dir(&owner).unwrap(),
            _ => std::fs::write(
                &owner,
                r#"{"v":9,"created_ms":1,"password_changed_ms":1,"password":"x"}"#,
            )
            .unwrap(),
        }
        let mut server = Server::spawn(&scratch, &[], "server");
        assert_eq!(
            server.exit_code(Duration::from_secs(60)),
            Some(78),
            "{tag}: {}",
            server.stderr_text()
        );
        assert!(
            server.stderr_text().contains("owner.json"),
            "{tag}: {}",
            server.stderr_text()
        );
        // Never setup-only: a mangled file does not reopen setup, and nothing was written over it.
        assert!(
            !server.stderr_text().contains("no owner login yet"),
            "{tag}"
        );
        if make == 0 {
            assert_eq!(read(&owner), "{ not json");
        }
        // `check` says the same, all at once, without changing anything.
        let checked = console(&scratch, &["check"]);
        assert_eq!(checked.code, 78, "{tag}: {} / {}", checked.out, checked.err);
        assert!(checked.err.contains("owner.json"), "{tag}: {}", checked.err);
    }
}

#[test]
fn a_login_allow_list_with_a_typo_stops_the_server_with_exit_78_and_check_says_the_same() {
    let scratch = Scratch::new("allow-typo");
    for list in ["203.0.113.0/24x", "203.0.113.0/24, oops"] {
        let mut server = Server::spawn(&scratch, &[("OAIY_LOGIN_ALLOW", list)], "server");
        assert_eq!(
            server.exit_code(Duration::from_secs(60)),
            Some(78),
            "{list:?}: {}",
            server.stderr_text()
        );
        let stderr = server.stderr_text();
        assert!(
            stderr.contains("OAIY_LOGIN_ALLOW") && stderr.contains("not an address or a network"),
            "{list:?}: {stderr}"
        );
        // Nothing listened, so nothing was open to the world for want of a restriction.
        let checked = command(&scratch, &[("OAIY_LOGIN_ALLOW", list)])
            .arg("check")
            .output()
            .unwrap();
        assert_eq!(checked.status.code(), Some(78), "{list:?}");
        assert!(
            String::from_utf8_lossy(&checked.stderr).contains("OAIY_LOGIN_ALLOW"),
            "{list:?}"
        );
    }
    // A list that is right starts the server, and the restriction is there.
    let mut server = Server::spawn(&scratch, &[("OAIY_LOGIN_ALLOW", "203.0.113.0/24")], "ok");
    server.wait_until_up();
    // The machine itself is not in the list: the sign-in is refused, before the password is looked at.
    let r = login(&server, PASSWORD);
    assert_eq!(
        (r.status, r.body["error"]["code"].as_str()),
        (403, Some("login_not_allowed")),
        "{}",
        r.text
    );
    let _ = server.exit_code(Duration::from_millis(1));
}

#[test]
fn check_is_ok_on_a_good_configuration_and_lists_every_violation_on_a_bad_one() {
    let scratch = Scratch::new("check");
    let ok = console(&scratch, &["check"]);
    assert_eq!(ok.code, 0, "{} / {}", ok.out, ok.err);
    assert!(ok.out.contains("check: ok"), "{}", ok.out);
    let bad = command(
        &scratch,
        &[
            ("OAIY_ACCESS_MODE", "legacy"),
            ("OAIY_SERVER_PORT", "99999"),
        ],
    )
    .args(["check"])
    .output()
    .unwrap();
    assert_eq!(bad.status.code(), Some(78));
    let err = String::from_utf8_lossy(&bad.stderr);
    assert!(
        err.contains("OAIY_ACCESS_MODE") && err.contains("OAIY_SERVER_PORT"),
        "{err}"
    );
}

/// The running server's console status as JSON, read the way the console reads it: the credential in its file, from the
/// machine itself, no Origin.
fn console_status(scratch: &Scratch, server: &Server) -> Value {
    let token = read(&scratch.auth().join("console.token"));
    let response = reqwest::blocking::Client::new()
        .get(format!(
            "http://127.0.0.1:{}/api/auth/console/status",
            server.port
        ))
        .bearer_auth(token.trim())
        .send()
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    response.json().unwrap()
}

/// A login from `from` behind a trusted proxy, sent whole and abandoned 12 ms later: the pass of Argon2 that it
/// started is still running (it takes far longer) when the client is gone.
fn hang_up_login(port: u16, from: &str) {
    use std::io::Write;
    let body = json!({ "password": "wrong wrong wrong" }).to_string();
    let request = format!(
        "POST /api/auth/login HTTP/1.1\r\nHost: dash.example.com\r\nOrigin: https://dash.example.com\r\n\
         Sec-Fetch-Site: same-origin\r\nX-Forwarded-For: {from}\r\nX-Forwarded-Proto: https\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    if let Ok(mut socket) = std::net::TcpStream::connect(("127.0.0.1", port)) {
        let _ = socket.write_all(request.as_bytes());
        std::thread::sleep(Duration::from_millis(12));
        // Dropped without reading a byte.
    }
}

#[test]
fn a_flood_of_logins_whose_clients_hang_up_is_bounded_and_counted() {
    let scratch = Scratch::new("hangup");
    let mut server = Server::spawn(
        &scratch,
        &[
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_TRUSTED_PROXIES", "127.0.0.1"),
        ],
        "server",
    );
    server.wait_until_up();
    let code = setup_code_of(&console(&scratch, &["auth", "setup-code"]));
    let proxied = |method: reqwest::Method, path: &str, body: Value, from: &str| {
        reqwest::blocking::Client::new()
            .request(method, format!("http://127.0.0.1:{}{path}", server.port))
            .header("host", "dash.example.com")
            .header("origin", "https://dash.example.com")
            .header("sec-fetch-site", "same-origin")
            .header("x-forwarded-for", from)
            .header("x-forwarded-proto", "https")
            .json(&body)
            .send()
            .unwrap()
    };
    let made = proxied(
        reqwest::Method::POST,
        "/api/auth/setup",
        json!({ "code": code, "password": PASSWORD }),
        "203.0.113.1",
    );
    assert_eq!(made.status().as_u16(), 201, "{}", made.text().unwrap());

    // Forty logins from forty addresses, each abandoned as soon as it is sent, about thirty a second.
    let port = server.port;
    let clients: Vec<_> = (0..40)
        .map(|i| {
            let client = std::thread::spawn(move || {
                hang_up_login(port, &format!("198.51.100.{}", 10 + i));
            });
            std::thread::sleep(Duration::from_millis(30));
            client
        })
        .collect();
    for client in clients {
        client.join().unwrap();
    }
    // Let the passes that are still running end.
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        let status = console_status(&scratch, &server);
        if status["verifications"]["running"] == 0 {
            std::thread::sleep(Duration::from_millis(500));
            let again = console_status(&scratch, &server);
            if again["verifications"]["running"] == 0 {
                break again;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the passes did not end: {status}"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    let v = &status["verifications"];
    let (started, most, bound) = (
        v["started"].as_u64().unwrap(),
        v["mostAtOnce"].as_u64().unwrap(),
        v["bound"].as_u64().unwrap(),
    );
    eprintln!(
        "hang-up flood: {v}, recent failures {}",
        status["recentFailures"]
    );
    assert_eq!(bound, 3);
    assert!(started >= 3, "the flood started passes: {v}");
    assert!(
        most <= bound,
        "{most} passes ran at once (the bound is {bound}): a client that hangs up must not free the bound: {v}"
    );
    // Every pass of the flood was a wrong password, and every one was counted though nobody was left to be told.
    assert_eq!(
        status["recentFailures"].as_u64(),
        Some(started),
        "the failures of clients that hung up are counted: {status}"
    );
    // The owner still gets in.
    let owner = proxied(
        reqwest::Method::POST,
        "/api/auth/login",
        json!({ "password": PASSWORD }),
        "203.0.113.77",
    );
    assert_eq!(owner.status().as_u16(), 200, "{}", owner.text().unwrap());
    let _ = server.exit_code(Duration::from_millis(1));
}

#[test]
fn a_running_server_that_the_console_cannot_read_the_files_of_is_not_touched() {
    let scratch = Scratch::new("cannot-read");
    let server = Server::start(&scratch, "server");
    // A console that cannot read the credential (its file is gone) does nothing and says which file.
    std::fs::remove_file(scratch.auth().join("console.token")).unwrap();
    let c = console(&scratch, &["auth", "setup-code"]);
    assert_eq!(c.code, 1, "{} / {}", c.out, c.err);
    assert!(
        c.err.contains("console.token") && c.err.contains("nothing was done"),
        "{}",
        c.err
    );
    let info = server
        .page(reqwest::Method::GET, "/api/auth/info", None, None, None)
        .body;
    assert_eq!(info["setupCode"], "none", "nothing was made");
    // A second server on the folder is refused (the lock), with the process that holds it.
    let mut second = Server::spawn(&scratch, &[], "second");
    assert_eq!(
        second.exit_code(Duration::from_secs(60)),
        Some(78),
        "{}",
        second.stderr_text()
    );
    assert!(
        second.stderr_text().contains("in use by process"),
        "{}",
        second.stderr_text()
    );
}
