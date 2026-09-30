//! Exposure and the startup rules of design 4.5.4 and 4.5.5 against the real program: `oaiy-server` on a port the
//! system picked, over a data folder of its own, in the shapes an operator can put it in (local, behind a proxy,
//! proxy-only, on a lan address), driven over real HTTP.
//!
//! What only a real process shows: the exit code of a refusal (78, which the shipped unit does not restart) and the
//! one line before it, `oaiy-server check` listing every violation, the banner on stderr, the address the listener
//! binds, a proxied server that starts with no owner and waits for one, and a proxied server that believes the
//! proxy it names (a peer that is another loopback address: 127.0.0.2 to 127.0.0.9) and no other.
//!
//! **A default run opens nothing on a network address.** Every server here listens on 127.0.0.1 (or another 127.x.y.z
//! address) and every client is one, so Windows Firewall has nothing to ask about. What genuinely needs a listener
//! that is not loopback (the `lan` bind, the proxy-only shape, and connecting to this machine's own network address
//! as a peer that is neither loopback nor the proxy) is the last group of tests below. Those are `#[ignore]`d and also
//! need `OAIY_TEST_LAN=1`, and each says what it is about to do before it does it:
//!
//! ```text
//! OAIY_TEST_LAN=1 cargo test --test access_exposure -- --ignored --nocapture
//! ```
//!
//! The same behaviours are covered without a socket by the guard tests (`auth::guard_tests`, `auth::login_tests`, which
//! send requests with a fake peer address and build the guard from a validated configuration) and by
//! `auth::exposure`'s tests of every rule; `scripts/e2e-exposure.mjs` has the same opt-in for its half.
//!
//! The server has a home folder, data folder and environment of its own, the voice gateway (a fixed port) off, and a
//! port the system picked, so it can never meet the desktop that may be running on this machine.

use std::io::Write as _;
use std::net::{IpAddr, TcpListener, UdpSocket};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// A token of the kind the server takes (`auth::token::check_static_token_shape`: worth 128 bits, no pattern).
const TOKEN: &str = "55Vg_eegIwyr7yMQ_Nh6euAXoVKz9nTnWw8vtNNtxeE";
#[cfg(feature = "web")]
const PASSWORD: &str = "k7Qz!mV3#pW9xLd2 rn8Tb";

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("oaiy-exposure-{tag}-{}", std::process::id()));
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

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A running `oaiy-server`. Killed when dropped.
struct Server {
    child: Child,
    port: u16,
    stderr: PathBuf,
}

impl Server {
    fn spawn(scratch: &Scratch, extra_env: &[(&str, &str)]) -> Server {
        let port = free_port();
        let stderr = scratch.0.join("server.stderr");
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

    fn start(scratch: &Scratch, extra_env: &[(&str, &str)]) -> Server {
        let mut server = Server::spawn(scratch, extra_env);
        server.wait_until_up();
        server
    }

    fn wait_until_up(&mut self) {
        self.wait_until_up_at("127.0.0.1");
    }

    /// Wait until `/api/health` answers at this address (a loopback one unless the test is an opt-in).
    fn wait_until_up_at(&mut self, at: &str) {
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
                    c.get(format!("http://{at}:{}/api/health", self.port))
                        .header("host", format!("{at}:{}", self.port))
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
                self.stderr_text()
            );
            std::thread::sleep(Duration::from_millis(100));
        }
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

    fn stderr_text(&self) -> String {
        std::fs::read_to_string(&self.stderr).unwrap_or_default()
    }

    /// One request to `at:port` with these headers. `Host` is what the headers say, else the address asked.
    fn ask(
        &self,
        at: &str,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
    ) -> (u16, Value, String) {
        self.ask_from(None, at, method, path, headers)
    }

    /// [`Server::ask`] from another loopback address of this machine (127.0.0.2 to 127.0.0.9): a different peer that
    /// is still this machine, which needs no network address and asks nothing of the firewall.
    fn ask_from(
        &self,
        from: Option<&str>,
        at: &str,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
    ) -> (u16, Value, String) {
        let mut builder = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none());
        if let Some(from) = from {
            let ip: IpAddr = from.parse().unwrap();
            assert!(
                ip.is_loopback(),
                "a default test asks from loopback addresses only: {from}"
            );
            builder = builder.local_address(ip);
        }
        let client = builder.build().unwrap();
        let mut request = client
            .request(
                reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
                format!("http://{at}:{}{path}", self.port),
            )
            .header("content-type", "application/json");
        if !headers.iter().any(|(n, _)| n.eq_ignore_ascii_case("host")) {
            request = request.header("host", format!("{at}:{}", self.port));
        }
        for (n, v) in headers {
            request = request.header(*n, *v);
        }
        if method == "POST" {
            request = request.body("{}");
        }
        let response = request
            .send()
            .unwrap_or_else(|e| panic!("{method} {at}{path}: {e}"));
        let status = response.status().as_u16();
        let text = response.text().unwrap_or_default();
        (
            status,
            serde_json::from_str(&text).unwrap_or(Value::Null),
            text,
        )
    }

    /// With the token and the machine's own name.
    fn as_cli(&self, method: &str, path: &str, headers: &[(&str, &str)]) -> (u16, Value, String) {
        let bearer = format!("Bearer {TOKEN}");
        let mut all = vec![("authorization", bearer.as_str())];
        all.extend_from_slice(headers);
        self.ask("127.0.0.1", method, path, &all)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn code(body: &Value) -> Option<&str> {
    body["error"]["code"].as_str()
}

/// What an opt-in test says before it does anything.
const LAN_NOTICE: &str = "this opens a socket on your LAN address (a listener on 0.0.0.0 and connections to this machine's network address) and Windows will ask for a firewall exception for the program that listens; every 'Allow' is a permanent inbound rule for that exe";

/// The gate of a test that needs a listener that is not loopback: it says what it is about to do, and does it only when
/// `OAIY_TEST_LAN=1` says the person running it means it (an `--ignored` or `--include-ignored` run of the crate's
/// other ignored tests must not open a socket on the network by accident). Written straight to stderr so that the
/// harness does not swallow it.
fn lan_opt_in(test: &str) -> bool {
    let mut err = std::io::stderr();
    let _ = writeln!(err, "\n{test}: {LAN_NOTICE}");
    if std::env::var("OAIY_TEST_LAN").as_deref() == Ok("1") {
        return true;
    }
    let _ = writeln!(err, "{test}: skipped: set OAIY_TEST_LAN=1 to run it");
    false
}

/// This machine's address on the network, if it has one: what the operating system would send from to reach
/// another machine (nothing is sent). `None` on a machine with no route. Only the opt-in tests call it.
fn lan_address() -> Option<IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.0.2.1:9").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
}

/// Whether the address is one a lan listener takes a bearer from (loopback, RFC 1918, link-local, CGNAT).
fn is_private(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || (o[0] == 100 && (64..=127).contains(&o[1]))
        }
        IpAddr::V6(v6) => {
            let first = v6.segments()[0];
            v6.is_loopback() || (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80
        }
    }
}

/// Make the owner login the way an operator does before a lan start: with the console, on a stopped server. (A build
/// without the web login has no console: it writes the file the rule looks for.)
fn make_owner(scratch: &Scratch) {
    #[cfg(feature = "web")]
    {
        let password_file = scratch.0.join("owner-password");
        std::fs::write(&password_file, PASSWORD).unwrap();
        let status = command(scratch, &[])
            // (`--new-folder`: the console makes no folder of its own unless told to, and this one has never run.)
            .args(["auth", "init", "--new-folder", "--password-file"])
            .arg(&password_file)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("run oaiy-server auth init");
        assert!(status.success(), "auth init: {status}");
    }
    #[cfg(not(feature = "web"))]
    {
        let auth = scratch.data().join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        std::fs::write(auth.join("owner.json"), r#"{"v":1}"#).unwrap();
    }
}

/// `oaiy-server check` with this environment: the exit code, stdout and stderr.
fn check(scratch: &Scratch, extra_env: &[(&str, &str)]) -> (Option<i32>, String, String) {
    let output = command(scratch, extra_env)
        .arg("check")
        .stdin(Stdio::null())
        .output()
        .expect("run oaiy-server check");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

// ==================================== rules 1 to 6: a refusal and its non-refusal ================================

/// A configuration the rules refuse, the server run with it exits 78 with one line that names `names`, makes nothing,
/// and a configuration next to it that they allow starts.
fn refused(tag: &str, env: &[(&str, &str)], names: &str) {
    let scratch = Scratch::new(tag);
    let mut server = Server::spawn(&scratch, env);
    assert_eq!(
        server.exit_code(Duration::from_secs(60)),
        Some(78),
        "{tag}: {}",
        server.stderr_text()
    );
    let said = server.stderr_text();
    assert!(said.contains(names), "{tag}: {said}");
    assert_eq!(
        said.lines()
            .filter(|l| l.starts_with("oaiy-server:"))
            .count(),
        1,
        "{tag}: one line: {said}"
    );
    assert!(!scratch.data().exists(), "{tag}: the data folder was made");
}

#[test]
fn t28_rule_1_a_bind_that_is_not_loopback_lan_or_an_address_is_refused_and_the_three_are_not() {
    for (i, bad) in [
        "bogus",
        "true",
        "localhost",
        "0.0.0.0/0",
        "127.0.0.1:8080",
        "lan;",
    ]
    .into_iter()
    .enumerate()
    {
        refused(
            &format!("r1-{i}"),
            &[("OAIY_SERVER_BIND", bad)],
            "OAIY_SERVER_BIND",
        );
    }
    for (i, good) in ["loopback", " Loopback ", "127.0.0.1"]
        .into_iter()
        .enumerate()
    {
        let scratch = Scratch::new(&format!("r1-ok-{i}"));
        let server = Server::start(
            &scratch,
            &[("OAIY_SERVER_BIND", good), ("OAIY_ACCESS_MODE", "scoped")],
        );
        assert!(
            server.stderr_text().contains("exposure local"),
            "{good}: {}",
            server.stderr_text()
        );
    }
}

/// An IP literal is an address to bind, and the listener is there and nowhere else. 127.0.0.2 is loopback, so this is
/// the real thing (a socket bound to the address the operator named) without a network address.
#[test]
fn t28_rule_1_the_listener_binds_the_address_it_was_told() {
    let scratch = Scratch::new("r1-bind-127-0-0-2");
    let mut server = Server::spawn(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "127.0.0.2"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
    );
    server.wait_until_up_at("127.0.0.2");
    let said = server.stderr_text();
    assert!(
        said.contains("exposure local")
            && said.contains(&format!("listening on 127.0.0.2:{}", server.port)),
        "{said}"
    );
    // (127.0.0.2 is not one of the loopback *names* a Host may carry, so a client that connects there says `localhost`.)
    let host = format!("localhost:{}", server.port);
    let (status, body, _) = server.ask("127.0.0.2", "GET", "/api/auth/info", &[("host", &host)]);
    assert_eq!(
        (status, body["scheme"].as_str()),
        (200, Some("oaiy-auth/1"))
    );
    let (status, body, _) = server.ask("127.0.0.2", "GET", "/api/auth/info", &[]);
    assert_eq!(
        (status, code(&body)),
        (421, Some("misdirected_host")),
        "the Host is the address, which is no name of this install"
    );
    // Not on the address it was not told: nothing is listening on 127.0.0.1 at that port.
    let elsewhere = std::net::SocketAddr::from(([127, 0, 0, 1], server.port));
    assert!(
        std::net::TcpStream::connect_timeout(&elsewhere, Duration::from_secs(2)).is_err(),
        "the server answers on 127.0.0.1 too"
    );
}

/// A lan bind needs an owner: refused without one (the static token is no stand-in for a login), and with one `check`
/// says what the install is. (Starting a lan listener is the opt-in test at the end of this file.)
#[test]
fn t28_rule_2_a_lan_bind_needs_an_owner_and_is_accepted_with_one() {
    refused(
        "r2",
        &[("OAIY_SERVER_BIND", "lan"), ("OAIY_ACCESS_MODE", "scoped")],
        "oaiy-server auth init",
    );
    refused(
        "r2-token",
        &[
            ("OAIY_SERVER_BIND", "0.0.0.0"),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_SERVER_TOKEN", TOKEN),
        ],
        "oaiy-server auth init",
    );
    refused(
        "r2-address",
        &[
            ("OAIY_SERVER_BIND", "192.168.1.5"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
        "oaiy-server auth init",
    );
    let scratch = Scratch::new("r2-ok");
    make_owner(&scratch);
    let (code, out, err) = check(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "lan"),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_SERVER_TOKEN", TOKEN),
        ],
    );
    assert_eq!(code, Some(0), "{err}");
    assert!(
        out.contains("exposure lan")
            && out.contains("bearer tokens only")
            && out.contains("0.0.0.0:17972")
            && out.contains("oaiy-server check: ok"),
        "{out}"
    );
}

#[test]
fn t28_rule_3_a_public_url_is_an_https_origin_with_no_path_and_no_loopback_host() {
    for (i, bad) in [
        "https://dash.example.com/some/path",
        "http://dash.example.com",
        "https://localhost",
        "https://dash.example.com?x=1",
    ]
    .into_iter()
    .enumerate()
    {
        refused(
            &format!("r3-{i}"),
            &[("OAIY_PUBLIC_URL", bad), ("OAIY_ACCESS_MODE", "scoped")],
            "OAIY_PUBLIC_URL",
        );
    }
    refused(
        "r3-same",
        &[
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_AGENT_URL", "https://dash.example.com:8443"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
        "same host",
    );
    let scratch = Scratch::new("r3-ok");
    let server = Server::start(
        &scratch,
        &[
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_AGENT_URL", "https://agent.example.com:8443"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
    );
    let said = server.stderr_text();
    assert!(
        said.contains("exposure proxied") && said.contains("agent https://agent.example.com:8443"),
        "{said}"
    );
}

#[test]
fn t28_rule_4_a_network_bind_behind_a_public_url_must_name_its_proxy() {
    refused(
        "r4",
        &[
            ("OAIY_SERVER_BIND", "0.0.0.0"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
        "OAIY_TRUSTED_PROXIES",
    );
    // With the proxy named it is accepted, with no owner: the proxied install waits in setup-only mode (rule 2 does not
    // apply to it). `check` says so without opening a socket; starting the proxy-only listener is the opt-in test below.
    let scratch = Scratch::new("r4-ok");
    let (code, out, err) = check(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "0.0.0.0"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_TRUSTED_PROXIES", "172.30.0.0/24"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
    );
    assert_eq!(code, Some(0), "{err}");
    assert!(
        out.contains("exposure proxied")
            && out.contains("proxy-only")
            && out.contains("172.30.0.0/24")
            && out.contains("direct_access_refused")
            && out.contains("oaiy-server check: ok"),
        "{out}"
    );
    assert!(!scratch.data().exists(), "check made a folder");
}

#[test]
fn t28_rule_5_a_weak_static_token_is_refused_and_a_good_one_is_the_cli_preset() {
    for (i, bad) in [
        "short",
        "change-me",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "has a space in it 0123456789abcdefghijk",
        // 40 different characters that count up, which the shape of design 4.1 took; and what a hand made up.
        "abcdefghijklmnopqrstuvwxyz0123456789ABCD",
        "verysecrettokenverysecrettoken1234",
    ]
    .into_iter()
    .enumerate()
    {
        refused(
            &format!("r5-{i}"),
            &[("OAIY_SERVER_TOKEN", bad), ("OAIY_ACCESS_MODE", "scoped")],
            "OAIY_SERVER_TOKEN",
        );
    }
    let scratch = Scratch::new("r5-ok");
    let server = Server::start(
        &scratch,
        &[("OAIY_SERVER_TOKEN", TOKEN), ("OAIY_ACCESS_MODE", "scoped")],
    );
    let (status, body, _) = server.as_cli("GET", "/api/auth/whoami", &[]);
    assert_eq!((status, body["kind"].as_str()), (200, Some("static")));
}

#[test]
fn t28_rule_6_the_modes_that_are_not_allowed_there_are_refused() {
    refused(
        "r6-shadow-proxied",
        &[
            ("OAIY_ACCESS_MODE", "shadow"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
        ],
        "shadow",
    );
    // Without the web login the default is legacy, which is refused off a local install.
    #[cfg(not(feature = "web"))]
    refused(
        "r6-legacy-proxied",
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        "legacy",
    );
    #[cfg(feature = "web")]
    refused("r6-legacy-web", &[("OAIY_ACCESS_MODE", "legacy")], "legacy");
}

#[test]
fn t28_check_lists_every_violation_at_once_and_a_good_configuration_says_what_it_is() {
    let scratch = Scratch::new("check-all");
    let (code, out, err) = check(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "bogus"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com/x"),
            ("OAIY_SERVER_TOKEN", "short"),
            ("OAIY_ACCESS_MODE", "scopd"),
            ("OAIY_SERVER_PORT", "0"),
            ("OAIY_TRUSTED_PROXIES", "nginx"),
        ],
    );
    assert_eq!(code, Some(78), "{err}");
    assert!(out.is_empty(), "{out}");
    // (Violations: a warning is a line of its own, such as the one of the flows a login can run on a proxied install.)
    let lines: Vec<&str> = err
        .lines()
        .filter(|l| l.starts_with("oaiy-server check:") && !l.contains(": warning: "))
        .collect();
    for name in [
        "OAIY_SERVER_BIND",
        "OAIY_PUBLIC_URL",
        "OAIY_SERVER_TOKEN",
        "OAIY_ACCESS_MODE",
        "OAIY_SERVER_PORT",
        "OAIY_TRUSTED_PROXIES",
    ] {
        assert!(
            lines.iter().any(|l| l.contains(name)),
            "{name} missing from:\n{err}"
        );
    }
    assert_eq!(lines.len(), 6, "one line each: {err}");
    // A lan bind with no owner and the weak token: both rules, in one run; nothing was made on the way.
    let (code, _, err) = check(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "lan"),
            ("OAIY_SERVER_TOKEN", "short"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ],
    );
    assert_eq!(code, Some(78));
    assert!(
        err.contains("oaiy-server auth init") && err.contains("OAIY_SERVER_TOKEN"),
        "{err}"
    );
    assert!(!scratch.data().exists(), "check made a folder");
    // A good one: exposure and ok, exit 0, and the warnings that are not refusals.
    let (code, out, err) = check(
        &scratch,
        &[
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_ALLOW_PUBLIC_PLAINTEXT", "1"),
        ],
    );
    assert_eq!(code, Some(0), "{err}");
    assert!(
        out.contains("exposure proxied") && out.contains("oaiy-server check: ok"),
        "{out}"
    );
    assert!(
        err.contains("warning") && err.contains("OAIY_ALLOW_PUBLIC_PLAINTEXT"),
        "{err}"
    );
    assert!(!scratch.data().exists(), "check made a folder");
}

// ==================================== a proxied server that starts with no owner ================================

/// What a proxy adds, from this machine (which the default trusts as a proxy).
const THROUGH_CADDY: [(&str, &str); 4] = [
    ("host", "dash.example.com"),
    ("x-forwarded-for", "203.0.113.9"),
    ("x-forwarded-proto", "https"),
    ("x-forwarded-host", "dash.example.com"),
];

/// A proxied install with no owner starts in setup-only mode and says so. `through_proxy`: ask as a proxy on this
/// machine would (the default trusts one); otherwise straight to the port, as the CLI does.
#[cfg(feature = "web")]
fn setup_only(tag: &str, extra: Vec<(&str, &str)>, through_proxy: bool) {
    {
        let scratch = Scratch::new(tag);
        let mut env = vec![
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_ACCESS_MODE", "scoped"),
        ];
        env.extend(extra.iter().copied());
        let server = Server::start(&scratch, &env);
        assert!(!scratch.data().join("auth").join("owner.json").exists());
        // The banner names the console command and holds no code.
        let said = server.stderr_text();
        assert!(
            said.contains("no owner login yet") && said.contains("oaiy-server auth setup-code"),
            "{tag}: {said}"
        );
        // Through the proxy this machine is (loopback is trusted unless the list names others), or straight from the
        // machine itself for the CLI: an anonymous caller is told to set up, not to sign in.
        let through: Vec<(&str, &str)> = if through_proxy {
            THROUGH_CADDY.to_vec()
        } else {
            // Proxy-only: the CLI's way, straight to the port from this machine, no forwarded header.
            vec![]
        };
        let (status, body, text) = server.ask("127.0.0.1", "GET", "/api/config", &through);
        assert_eq!(
            (status, code(&body)),
            (401, Some("setup_required")),
            "{tag}: {text}"
        );
        // A bearer is unaffected: the static token is not needed for that, and none is set: token_invalid.
        let stranger = format!("Bearer oaiypat_0123456789abcdef_{}", "A".repeat(43));
        let (status, body, _) = server.ask(
            "127.0.0.1",
            "GET",
            "/api/config",
            &[("authorization", &stranger)],
        );
        assert_eq!((status, code(&body)), (401, Some("token_invalid")), "{tag}");
        // Health is answered, and `info` says there is no login yet.
        let (status, _, _) = server.ask("127.0.0.1", "GET", "/api/health", &through);
        assert_eq!(status, 200, "{tag}");
        let (status, info, text) = server.ask("127.0.0.1", "GET", "/api/auth/info", &through);
        assert_eq!(status, 200, "{tag}: {text}");
        assert_eq!(info["loginConfigured"], false, "{tag}");
        if through_proxy {
            assert_eq!(
                (
                    info["seen"]["clientIp"].as_str(),
                    info["seen"]["proto"].as_str(),
                    info["secureChannel"].as_bool()
                ),
                (Some("203.0.113.9"), Some("https"), Some(true)),
                "the proxy's word is believed: {text}"
            );
        }
    }
}

#[cfg(feature = "web")]
#[test]
fn t45_a_proxied_server_with_no_owner_starts_in_setup_only_mode_and_says_so() {
    setup_only("setup-only-loopback", vec![], true);
}

/// The same, as a proxy-only server (bound beyond loopback, its proxy named). Opt-in: it listens on 0.0.0.0.
#[cfg(feature = "web")]
#[test]
#[ignore = "opt-in: opens a socket on the LAN address (OAIY_TEST_LAN=1 and --ignored); Windows asks for a firewall exception"]
fn t45_lan_optin_a_proxy_only_server_with_no_owner_starts_in_setup_only_mode() {
    if !lan_opt_in("t45_lan_optin_a_proxy_only_server_with_no_owner_starts_in_setup_only_mode") {
        return;
    }
    setup_only(
        "setup-only-proxy-only",
        vec![
            ("OAIY_SERVER_BIND", "0.0.0.0"),
            ("OAIY_TRUSTED_PROXIES", "192.0.2.1"),
        ],
        false,
    );
}

/// The proxy the operator names is believed and no other, and the same headers from a peer that is not named are worth
/// nothing (design 4.5.4): with the peers being other loopback addresses of this machine, so that nothing listens or
/// connects beyond loopback. (`OAIY_TRUSTED_PROXIES` replaces the default, which is 127.0.0.1 and ::1.)
#[test]
fn t45_a_proxied_server_believes_the_proxy_it_names_and_only_that_one() {
    let scratch = Scratch::new("proxy-named-loopback");
    let server = Server::start(
        &scratch,
        &[
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_TRUSTED_PROXIES", "127.0.0.5/32"),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_SERVER_TOKEN", TOKEN),
        ],
    );
    let seen = |from: &str, headers: &[(&str, &str)]| {
        let (status, info, text) =
            server.ask_from(Some(from), "127.0.0.1", "GET", "/api/auth/info", headers);
        assert_eq!(status, 200, "{from}: {text}");
        (
            info["seen"]["clientIp"].as_str().map(str::to_owned),
            info["seen"]["viaTrustedProxy"].as_bool(),
            info["secureChannel"].as_bool(),
        )
    };
    // The named proxy: its word is taken, and its channel is secure.
    assert_eq!(
        seen("127.0.0.5", &THROUGH_CADDY),
        (Some("203.0.113.9".into()), Some(true), Some(true))
    );
    // Another address of this machine, and this machine's usual one, with the same headers: not the proxy, so not believed.
    for from in ["127.0.0.6", "127.0.0.1"] {
        assert_eq!(
            seen(from, &THROUGH_CADDY),
            (Some(from.into()), Some(false), Some(false)),
            "{from}"
        );
    }
    // A wrong `X-Forwarded-Proto` from the proxy is its misconfiguration, said once in the log however often it is sent.
    let wrong: Vec<(&str, &str)> = THROUGH_CADDY
        .iter()
        .map(|(n, v)| {
            (
                *n,
                if *n == "x-forwarded-proto" {
                    "http"
                } else {
                    *v
                },
            )
        })
        .collect();
    for _ in 0..3 {
        let (status, body, _) = server.ask_from(
            Some("127.0.0.5"),
            "127.0.0.1",
            "GET",
            "/api/auth/info",
            &wrong,
        );
        assert_eq!((status, code(&body)), (400, Some("proxy_misconfigured")));
    }
    assert_eq!(
        server
            .stderr_text()
            .matches("X-Forwarded-Proto http")
            .count(),
        1,
        "warned once: {}",
        server.stderr_text()
    );
    // nginx's default `Host` (the address it proxies to) through the named proxy is not one of this server's names.
    let (status, body, _) = server.ask_from(
        Some("127.0.0.5"),
        "127.0.0.1",
        "GET",
        "/api/config",
        &[
            ("host", &format!("127.0.0.1:{}", server.port)),
            ("x-forwarded-for", "203.0.113.9"),
            ("authorization", &format!("Bearer {TOKEN}")),
        ],
    );
    assert_eq!((status, code(&body)), (421, Some("misdirected_host")));
    // A proxy that names no client: its own address stands for everyone behind it, and the log says so once.
    for _ in 0..3 {
        let (status, info, _) = server.ask_from(
            Some("127.0.0.5"),
            "127.0.0.1",
            "GET",
            "/api/auth/info",
            &[("host", "dash.example.com"), ("x-forwarded-proto", "https")],
        );
        assert_eq!(
            (status, info["seen"]["clientIp"].as_str()),
            (200, Some("127.0.0.5"))
        );
    }
    assert_eq!(
        server
            .stderr_text()
            .matches("forwarded a request with no X-Forwarded-For")
            .count(),
        1
    );
    // The CLI on this machine (no forwarded header) is answered, and the token is the `cli` preset.
    let (status, body, _) = server.as_cli("GET", "/api/auth/whoami", &[]);
    assert_eq!((status, body["kind"].as_str()), (200, Some("static")));
}

// ==================================== proxy-only: a connection that did not come through the proxy ==============
//
// The rest of this file needs a listener that is not loopback: the proxy-only shape is "bound beyond loopback behind a
// public URL with the proxy named", and the lan shape is a bind beyond loopback. Each of these is `#[ignore]`d and
// checks `OAIY_TEST_LAN=1` first (see the top of the file). `auth::guard_tests` covers the same branches in process.

#[test]
#[ignore = "opt-in: opens a socket on the LAN address (OAIY_TEST_LAN=1 and --ignored); Windows asks for a firewall exception"]
fn t45_lan_optin_a_proxy_only_server_refuses_what_did_not_come_through_its_proxy() {
    if !lan_opt_in("t45_lan_optin_a_proxy_only_server_refuses_what_did_not_come_through_its_proxy")
    {
        return;
    }
    let scratch = Scratch::new("proxy-only");
    let server = Server::start(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "0.0.0.0"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            // Nobody on this machine's network is the proxy.
            ("OAIY_TRUSTED_PROXIES", "192.0.2.1"),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_SERVER_TOKEN", TOKEN),
        ],
    );
    let said = server.stderr_text();
    assert!(
        said.contains("proxy-only") && said.contains("192.0.2.1"),
        "{said}"
    );
    // This machine, straight to the port: the CLI and `auth ...` (a loopback peer with no forwarded header).
    let (status, _, text) = server.as_cli("GET", "/api/config", &[]);
    assert_eq!(status, 200, "{text}");
    // With the headers a proxy adds, it is a proxy that was never named.
    let (status, body, _) =
        server.as_cli("GET", "/api/config", &[("x-forwarded-for", "203.0.113.9")]);
    assert_eq!((status, code(&body)), (403, Some("direct_access_refused")));
    let (status, body, _) = server.ask("127.0.0.1", "GET", "/api/config", &THROUGH_CADDY);
    assert_eq!((status, code(&body)), (403, Some("direct_access_refused")));
    // Health is a probe: it is answered whoever asks.
    let (status, _, _) = server.ask("127.0.0.1", "GET", "/api/health", &THROUGH_CADDY);
    assert_eq!(status, 200);
    // From this machine's address on the network: a peer that is not loopback and not the proxy.
    let Some(lan) = lan_address() else {
        eprintln!(
            "skipped the network half of the proxy-only test: this machine has no network address"
        );
        return;
    };
    let at = lan.to_string();
    let host_of = |ip: &str| {
        if ip.contains(':') {
            format!("[{ip}]:{}", server.port)
        } else {
            format!("{ip}:{}", server.port)
        }
    };
    let bearer = format!("Bearer {TOKEN}");
    let (status, body, text) = server.ask(
        &if lan.is_ipv6() {
            format!("[{at}]")
        } else {
            at.clone()
        },
        "GET",
        "/api/config",
        &[("host", &host_of(&at)), ("authorization", &bearer)],
    );
    assert_eq!(
        (status, code(&body)),
        (403, Some("direct_access_refused")),
        "{text}"
    );
    let (status, body, _) = server.ask(
        &if lan.is_ipv6() {
            format!("[{at}]")
        } else {
            at.clone()
        },
        "GET",
        "/api/config",
        &[
            ("host", "dash.example.com"),
            ("x-forwarded-for", "203.0.113.9"),
            ("x-forwarded-proto", "https"),
        ],
    );
    assert_eq!((status, code(&body)), (403, Some("direct_access_refused")));
    let (status, _, _) = server.ask(
        &if lan.is_ipv6() {
            format!("[{at}]")
        } else {
            at.clone()
        },
        "GET",
        "/api/health",
        &[("host", &host_of(&at))],
    );
    assert_eq!(status, 200, "a probe from the network address");
}

#[test]
#[ignore = "opt-in: opens a socket on the LAN address (OAIY_TEST_LAN=1 and --ignored); Windows asks for a firewall exception"]
fn t45_lan_optin_a_proxy_only_server_believes_the_proxy_it_names_and_only_that_one() {
    if !lan_opt_in(
        "t45_lan_optin_a_proxy_only_server_believes_the_proxy_it_names_and_only_that_one",
    ) {
        return;
    }
    let Some(lan) = lan_address().filter(|ip| ip.is_ipv4()) else {
        eprintln!("skipped: this machine has no IPv4 network address to be the proxy");
        return;
    };
    let scratch = Scratch::new("proxy-named");
    let named = format!("{lan}/32");
    let server = Server::start(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "0.0.0.0"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_TRUSTED_PROXIES", &named),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_SERVER_TOKEN", TOKEN),
        ],
    );
    // This machine's own network address is the proxy: its word is taken (what `seen` says), and its channel is secure.
    let (status, info, text) =
        server.ask(&lan.to_string(), "GET", "/api/auth/info", &THROUGH_CADDY);
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        (
            info["seen"]["clientIp"].as_str(),
            info["seen"]["viaTrustedProxy"].as_bool(),
            info["secureChannel"].as_bool()
        ),
        (Some("203.0.113.9"), Some(true), Some(true)),
        "{text}"
    );
    // A wrong `X-Forwarded-Proto` from the proxy is its misconfiguration.
    let wrong: Vec<(&str, &str)> = THROUGH_CADDY
        .iter()
        .map(|(n, v)| {
            (
                *n,
                if *n == "x-forwarded-proto" {
                    "http"
                } else {
                    *v
                },
            )
        })
        .collect();
    let (status, body, _) = server.ask(&lan.to_string(), "GET", "/api/auth/info", &wrong);
    assert_eq!((status, code(&body)), (400, Some("proxy_misconfigured")));
    assert!(
        server
            .stderr_text()
            .matches("X-Forwarded-Proto http")
            .count()
            == 1,
        "warned once: {}",
        server.stderr_text()
    );
    // nginx's default `Host` (the address it proxies to) is not one of this server's names.
    let (status, body, _) = server.ask(
        &lan.to_string(),
        "GET",
        "/api/config",
        &[
            ("host", &format!("127.0.0.1:{}", server.port)),
            ("x-forwarded-for", "203.0.113.9"),
            ("authorization", &format!("Bearer {TOKEN}")),
        ],
    );
    assert_eq!((status, code(&body)), (421, Some("misdirected_host")));
    // The same headers from this machine's loopback address are not from the proxy: it is not in the list, and it is
    // not a bare loopback request either (it carries a forwarded header).
    let (status, body, _) = server.ask("127.0.0.1", "GET", "/api/auth/info", &THROUGH_CADDY);
    assert_eq!((status, code(&body)), (403, Some("direct_access_refused")));
}

// ==================================== lan: bearer only, no cookies, no UI, no sign-in ===========================

#[cfg(feature = "web")]
#[test]
#[ignore = "opt-in: opens a socket on the LAN address (OAIY_TEST_LAN=1 and --ignored); Windows asks for a firewall exception"]
fn t45_lan_optin_a_lan_listener_refuses_the_sign_in_and_a_cookie_and_serves_no_ui_to_the_network() {
    if !lan_opt_in(
        "t45_lan_optin_a_lan_listener_refuses_the_sign_in_and_a_cookie_and_serves_no_ui_to_the_network",
    ) {
        return;
    }
    let Some(lan) = lan_address().filter(|ip| ip.is_ipv4()) else {
        eprintln!("skipped: this machine has no IPv4 network address to reach the listener at");
        return;
    };
    let scratch = Scratch::new("lan");
    make_owner(&scratch);
    let server = Server::start(
        &scratch,
        &[
            ("OAIY_SERVER_BIND", "lan"),
            ("OAIY_ACCESS_MODE", "scoped"),
            ("OAIY_SERVER_TOKEN", TOKEN),
        ],
    );
    // A lan bind with an owner starts, and says what it is.
    let said = server.stderr_text();
    assert!(
        said.contains("exposure lan") && said.contains("bearer tokens only"),
        "{said}"
    );
    let at = lan.to_string();
    // The sign-in, the setup and the link are refused on plain HTTP, and say what is needed.
    for path in ["/api/auth/login", "/api/auth/setup", "/api/auth/link"] {
        let (status, body, text) = server.ask(&at, "POST", path, &[]);
        assert_eq!(
            (status, code(&body)),
            (403, Some("secure_channel_required")),
            "{path}: {text}"
        );
    }
    // A cookie of ours is refused, not ignored.
    let session = format!("oaiy_dash_1=oaiyses_0123456789abcdef_{}", "A".repeat(43));
    let (status, body, _) = server.ask(&at, "GET", "/api/services", &[("cookie", &session)]);
    assert_eq!(
        (status, code(&body)),
        (403, Some("secure_channel_required"))
    );
    let (status, body, _) = server.ask(
        &at,
        "GET",
        "/api/services",
        &[("cookie", "__Host-oaiy_dash=x")],
    );
    assert_eq!(
        (status, code(&body)),
        (403, Some("secure_channel_required"))
    );
    // No cookie, no credential: 401 (an owner exists, so it is not setup_required).
    let (status, body, _) = server.ask(&at, "GET", "/api/services", &[]);
    assert_eq!((status, code(&body)), (401, Some("auth_required")));
    // No UI: nothing is served but the API.
    for path in ["/", "/index.html", "/apps/dash", "/login"] {
        let (status, _, text) = server.ask(&at, "GET", path, &[]);
        assert_eq!(status, 404, "{path}");
        assert!(!text.contains("<html"), "{path}");
    }
    // The bearer works from a private address (this machine's own, if it is one), and is refused from a public one.
    let (status, body, text) = server.ask(
        &at,
        "GET",
        "/api/auth/whoami",
        &[("authorization", &format!("Bearer {TOKEN}"))],
    );
    if is_private(lan) {
        assert_eq!(
            (status, body["kind"].as_str()),
            (200, Some("static")),
            "{text}"
        );
    } else {
        assert_eq!(
            (status, code(&body)),
            (403, Some("plaintext_from_public_address")),
            "{text}"
        );
    }
    // `info` says the channel is not secure and that there is an owner.
    let (status, info, _) = server.ask(&at, "GET", "/api/auth/info", &[]);
    assert_eq!(
        (
            status,
            info["secureChannel"].as_bool(),
            info["loginConfigured"].as_bool()
        ),
        (200, Some(false), Some(true))
    );
    // Forwarded headers are read from no one here (nobody is trusted), so they cannot make a client look private.
    let (status, body, _) = server.ask(
        &at,
        "GET",
        "/api/auth/whoami",
        &[
            ("authorization", &format!("Bearer {TOKEN}")),
            ("x-forwarded-for", "192.168.1.9"),
        ],
    );
    if !is_private(lan) {
        assert_eq!(
            (status, code(&body)),
            (403, Some("plaintext_from_public_address"))
        );
    }
    // From the machine itself under a loopback name, the same server is the CLI's, and the sign-in is not there.
    let (status, body, _) = server.as_cli("GET", "/api/auth/whoami", &[]);
    assert_eq!((status, body["kind"].as_str()), (200, Some("static")));
    let (status, _, _) = server.ask("127.0.0.1", "POST", "/api/auth/login", &[]);
    assert_eq!(status, 404);
}

/// A local server (loopback, no public URL) has no proxy: every forwarded header is refused, whatever it says, and a
/// request carrying none is the machine's own.
#[test]
fn t45_a_local_server_refuses_every_forwarded_header() {
    let scratch = Scratch::new("local");
    let server = Server::start(
        &scratch,
        &[("OAIY_ACCESS_MODE", "scoped"), ("OAIY_SERVER_TOKEN", TOKEN)],
    );
    for header in [
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "forwarded",
        "via",
        "x-real-ip",
        "cf-connecting-ip",
        "true-client-ip",
    ] {
        let (status, body, text) = server.as_cli("GET", "/api/config", &[(header, "203.0.113.9")]);
        assert_eq!(
            (status, code(&body)),
            (421, Some("proxy_detected")),
            "{header}: {text}"
        );
    }
    let (status, _, text) = server.as_cli("GET", "/api/config", &[]);
    assert_eq!(status, 200, "{text}");
    assert!(
        server.stderr_text().contains("OAIY_PUBLIC_URL"),
        "one line names the setting to change: {}",
        server.stderr_text()
    );
}
