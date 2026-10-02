//! The real PHP relay for the L4 tests: a scratch data directory, the installer run as the owner would run it, and a small fleet of `php -S` servers on loopback that share it.
//!
//! **Safety.** Every process binds `127.0.0.1` on a port the operating system chose (never one of the owner's: 3306, 7860, 8080, 9333, 17872, 17972, 17973), uses a data directory under
//! the cargo target directory and never the repository's own `platform/relay/data`, never the WAMP web root, and is killed by its own PID when the harness is dropped (also when a test
//! panics). Where PHP, or its `sodium` and `pdo_sqlite` extensions, are missing, the test is skipped and says so.
//!
//! **Why a fleet.** `php -S` on Windows is single threaded: a held poll blocks every other request (design 9.4). The relay keeps its holds in the shared database, so several servers on
//! one data directory behave like the workers of one host; [`FleetHttp`] sends each request to a server that is not serving one, as a load balancer in front of a pool does, while
//! the clients keep using the one public URL the relay was installed with.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use oaiy_relay_core::client::loopback::LoopbackHttp;
use oaiy_relay_core::client::*;

const FORBIDDEN_PORTS: [u16; 7] = [3306, 7860, 8080, 9333, 17872, 17972, 17973];

pub fn relay_root() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../platform/relay").canonicalize().expect("platform/relay");
    // PHP cannot open the verbatim `\\?\` form that Windows canonicalisation gives.
    match path.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(plain) => PathBuf::from(plain),
        None => path,
    }
}

/// The PHP binary, or why there is none.
pub fn find_php() -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("OAIY_PHP") {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(PathBuf::from(r"C:\wamp64\bin\php\php8.4.15\php.exe"));
    candidates.push(PathBuf::from("php"));
    for c in candidates {
        let ok = Command::new(&c).arg("-r").arg("echo extension_loaded('sodium') && extension_loaded('pdo_sqlite') ? 'ok' : 'missing';").output();
        match ok {
            Ok(o) if String::from_utf8_lossy(&o.stdout) == "ok" => return Ok(c),
            Ok(o) => return Err(format!("{} has no sodium or pdo_sqlite ({})", c.display(), String::from_utf8_lossy(&o.stdout))),
            Err(_) => continue,
        }
    }
    Err("no php binary found (set OAIY_PHP)".to_string())
}

fn free_port() -> u16 {
    for _ in 0..100 {
        let l = TcpListener::bind("127.0.0.1:0").expect("a free port");
        let port = l.local_addr().unwrap().port();
        drop(l);
        if port > 1024 && !FORBIDDEN_PORTS.contains(&port) {
            return port;
        }
    }
    panic!("no free port");
}

pub struct PhpRelay {
    pub php: PathBuf,
    pub dir: PathBuf,
    pub data: PathBuf,
    pub ports: Vec<u16>,
    servers: Vec<Option<Child>>,
    pub first_key: String,
    pub admin_token: String,
}

impl Drop for PhpRelay {
    fn drop(&mut self) {
        for c in self.servers.iter_mut().flatten() {
            let _ = c.kill();
            let _ = c.wait();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn php_command(php: &Path) -> Command {
    let mut c = Command::new(php);
    c.arg("-d").arg("xdebug.mode=off").arg("-d").arg("display_errors=1").arg("-d").arg("error_reporting=-1").arg("-d").arg("html_errors=0");
    c
}

impl PhpRelay {
    /// Installs a relay in a scratch directory and starts `workers` servers on it. `None` (and a note on stderr) when PHP is not usable.
    pub fn start(workers: usize, wait_max: u64, call_features: bool) -> Option<PhpRelay> {
        let php = match find_php() {
            Ok(p) => p,
            Err(why) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
        };
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("relay-php-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&dir);
        let data = dir.join("data");
        std::fs::create_dir_all(&dir).unwrap();
        let ports: Vec<u16> = (0..workers.max(1)).map(|_| free_port()).collect();
        let root = relay_root();
        let url = format!("http://127.0.0.1:{}", ports[0]);
        let out = php_command(&php)
            .arg(root.join("bin/install.php"))
            .arg(format!("--url={url}"))
            .arg("--no-probe")
            .arg("--yes")
            .arg(format!("--call-features={}", if call_features { "yes" } else { "no" }))
            .env("OAIY_RELAY_DATA", &data)
            .current_dir(&root)
            .output()
            .expect("php install");
        assert!(out.status.success(), "the installer failed: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        // A short wait, so that a held poll ends in a couple of seconds.
        let config_path = data.join("config.json");
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["wait"]["max"] = serde_json::json!(wait_max);
        std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        let first_key = std::fs::read_to_string(data.join("first-key.txt")).unwrap().trim().to_string();
        let admin_token = std::fs::read_to_string(data.join("admin-token.txt")).unwrap().trim().to_string();
        let mut relay = PhpRelay { php, dir, data, ports, servers: Vec::new(), first_key, admin_token };
        for i in 0..relay.ports.len() {
            let child = relay.spawn(relay.ports[i]);
            relay.servers.push(Some(child));
        }
        for &p in &relay.ports {
            relay.wait_listening(p);
        }
        Some(relay)
    }

    fn spawn(&self, port: u16) -> Child {
        let root = relay_root();
        php_command(&self.php)
            .arg("-S")
            .arg(format!("127.0.0.1:{port}"))
            .arg("-t")
            .arg(root.join("public"))
            .arg(root.join("tests/router.php"))
            .env("OAIY_TEST_DATA", &self.data)
            .env_remove("PHP_CLI_SERVER_WORKERS")
            .current_dir(root.join("public"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("php -S")
    }

    fn wait_listening(&self, port: u16) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            if std::net::TcpStream::connect_timeout(&format!("127.0.0.1:{port}").parse().unwrap(), Duration::from_millis(200)).is_ok() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("php -S did not start on {port}");
    }

    pub fn public_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.ports[0])
    }

    /// The relay's own command line (`bin/relay.php`), as the owner runs it, in this relay's data directory.
    pub fn cli(&self, args: &[&str]) -> String {
        let root = relay_root();
        let out = php_command(&self.php)
            .arg(root.join("bin/relay.php"))
            .args(args)
            .env("OAIY_RELAY_DATA", &self.data)
            .current_dir(&root)
            .output()
            .expect("php relay.php");
        assert!(out.status.success(), "relay.php {args:?}: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Kills every server by its own PID (an outage).
    pub fn stop_servers(&mut self) {
        for c in self.servers.iter_mut() {
            if let Some(mut child) = c.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    /// Starts the servers again on the same ports.
    pub fn start_servers(&mut self) {
        for i in 0..self.ports.len() {
            if self.servers[i].is_none() {
                let child = self.spawn(self.ports[i]);
                self.servers[i] = Some(child);
            }
        }
        for &p in &self.ports {
            self.wait_listening(p);
        }
    }

    /// An HTTP client that spreads requests over the fleet.
    pub fn http(&self) -> Arc<FleetHttp> {
        Arc::new(FleetHttp {
            ports: self.ports.clone(),
            busy: self.ports.iter().map(|_| AtomicBool::new(false)).collect(),
            next: AtomicUsize::new(0),
        })
    }
}

/// A load balancer over the fleet: a request goes to a server that is not serving one, and its URL is rewritten from the public port to that server's.
pub struct FleetHttp {
    ports: Vec<u16>,
    busy: Vec<AtomicBool>,
    next: AtomicUsize,
}

impl HttpClient for FleetHttp {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let n = self.ports.len();
        let public = format!("127.0.0.1:{}", self.ports[0]);
        let deadline = Instant::now() + Duration::from_secs(30);
        let index = loop {
            let start = self.next.fetch_add(1, Ordering::SeqCst);
            if let Some(i) =
                (0..n).map(|k| (start + k) % n).find(|i| self.busy[*i].compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok())
            {
                break i;
            }
            if request.cancel.is_cancelled() {
                return Err(TransportError::Cancelled);
            }
            if Instant::now() > deadline {
                return Err(TransportError::Timeout);
            }
            std::thread::sleep(Duration::from_millis(3));
        };
        let mut routed = request.clone();
        routed.url = request.url.replacen(&public, &format!("127.0.0.1:{}", self.ports[index]), 1);
        let result = LoopbackHttp.send(&routed);
        self.busy[index].store(false, Ordering::SeqCst);
        result
    }
}
