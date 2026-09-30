//! `oaiy-server auth ...`, `check` and `flows ...` (design 4.7.9): the command line of the console.
//!
//! ```text
//! oaiy-server auth init [--generate] [--password-file PATH] [--force]   create owner.json (asks twice, no echo)
//! oaiy-server auth setup-code                                            make a setup code (24 h, 100 wrong guesses)
//! oaiy-server auth reset-password [--generate] [--password-file PATH]    new password; revokes every session and device
//! oaiy-server auth session-link                                          a single-use link (5 minutes) that makes a session
//! oaiy-server auth token create --preset P [--scope S]... [--ttl 30d] [--label L] [--origin URL]
//! oaiy-server auth token list | revoke <id>
//! oaiy-server auth sessions revoke-all [--devices]
//! oaiy-server auth status              exposure, login configured, counts, slow mode and blocked addresses, disk (no secrets)
//! oaiy-server check                    validate the configuration and the data folder; exit 78 lists every violation
//! ```
//!
//! **How it runs.** The server is the only writer of its auth files while it runs and memory is authoritative, so
//! the console has two paths and tries them in order:
//!
//! 1. the exclusive lock on `<data>/auth/.lock`: if it is taken the server is running, so the command reads
//!    `console.json` and `console.token` and calls the console's routes on the loopback port with the console
//!    credential, and memory and file change together; a file that is missing or unreadable, or a port that does not
//!    answer, is a message naming the file, and nothing is done;
//! 2. if the lock is free the server is stopped, and the command edits the files under that lock.
//!
//! **Rules.** Every run prints the resolved data folder and the folder's owner; nothing creates a directory but
//! `auth init`; a console run as root against a folder that another user owns refuses and prints the `sudo -u <owner>`
//! form (a `sudo oaiy-server auth init` would otherwise make `/root/.oaiy-server` and report success); a permission
//! error on an auth file is an error naming the file, never "corrupt". `check` is read-only and takes no lock.
//!
//! **What is not here.** `flows list|approve` belongs to the flow authority (a later package) and says so. A running
//! server's credential routes (`auth token` against it) are the credentials API's (a later package): `token`
//! edits the files of a stopped server, and says so when the server is running.
//!
//! Exit codes: 0 done, 1 could not, 2 misused, 78 a refusal of the configuration or of a file (what the shipped unit
//! does not restart).

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use zeroize::Zeroizing;

use super::audit::{AuditLog, Context as AuditContext};
use super::clock::{Clock, SystemClock};
use super::console::{status_lines, INFO_FILE, TOKEN_FILE};
use super::lock::LockError;
use super::mode::{validate_mode, Exposure};
use super::owner::{self, CreateError, OwnerDoc};
use super::password::{Argon2Engine, PasswordEngine};
use super::policy::{self, Reason};
use super::presets::Preset;
use super::principal::Actor;
use super::scopes::ScopeSet;
use super::setup::{SetupCode, Status, VALID_MS};
use super::store::{AuthStore, Host, MintFailure, MintSpec, SecureWriter, StoreError};
use super::token::{self, Kind};

pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILED: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
/// `EX_CONFIG`: the shipped unit does not restart it.
pub const EXIT_CONFIG: i32 = 78;

const DAY_MS: u64 = 24 * 3_600_000;

/// What the command line runs against: the environment, the terminal and the user, so that a test can stand in for all.
pub struct Io<'a> {
    pub env: &'a dyn Fn(&str) -> Option<String>,
    pub out: &'a mut dyn Write,
    pub err: &'a mut dyn Write,
    /// Read a line without echoing it (a password).
    pub prompt: &'a mut dyn FnMut(&str) -> io::Result<String>,
    /// The effective user id: `None` where there is no such thing.
    pub euid: Option<u32>,
}

// ---- the commands -----------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Init {
        generate: bool,
        password_file: Option<PathBuf>,
        force: bool,
        /// Make the auth folder (and the data folder) if there is none: without it init only works in a folder
        /// a server has already made its own.
        new_folder: bool,
    },
    SetupCode,
    ResetPassword {
        generate: bool,
        password_file: Option<PathBuf>,
    },
    SessionLink,
    TokenCreate {
        preset: String,
        scopes: Vec<String>,
        ttl: Option<String>,
        label: Option<String>,
        origin: Option<String>,
    },
    TokenList,
    TokenRevoke {
        id: String,
    },
    SessionsRevokeAll {
        devices: bool,
    },
    Status,
    Check,
    Flows,
}

impl Command {
    /// The name the audit log keeps of it (never the arguments).
    fn name(&self) -> &'static str {
        match self {
            Command::Init { .. } => "init",
            Command::SetupCode => "setup-code",
            Command::ResetPassword { .. } => "reset-password",
            Command::SessionLink => "session-link",
            Command::TokenCreate { .. } => "token-create",
            Command::TokenList => "token-list",
            Command::TokenRevoke { .. } => "token-revoke",
            Command::SessionsRevokeAll { .. } => "sessions-revoke-all",
            Command::Status => "status",
            Command::Check => "check",
            Command::Flows => "flows",
        }
    }
}

pub const USAGE: &str = "usage:
  oaiy-server auth init [--generate] [--password-file PATH] [--force] [--new-folder]
  oaiy-server auth setup-code
  oaiy-server auth reset-password [--generate] [--password-file PATH]
  oaiy-server auth session-link
  oaiy-server auth token create --preset P [--scope S]... [--ttl 30d] [--label L] [--origin URL]
  oaiy-server auth token list | revoke <id>
  oaiy-server auth sessions revoke-all [--devices]
  oaiy-server auth status
  oaiy-server check
  oaiy-server flows list | approve <id> [--hash H]";

/// Take the arguments after the program name apart.
pub fn parse(args: &[String]) -> Result<Command, String> {
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    // A flag with a value, and a flag without one, out of what is left.
    struct Flags<'a> {
        rest: Vec<&'a str>,
    }
    impl<'a> Flags<'a> {
        fn has(&mut self, name: &str) -> bool {
            let before = self.rest.len();
            self.rest.retain(|w| *w != name);
            self.rest.len() != before
        }
        fn value(&mut self, name: &str) -> Result<Option<String>, String> {
            let Some(i) = self.rest.iter().position(|w| *w == name) else {
                return Ok(None);
            };
            self.rest.remove(i);
            if i >= self.rest.len() {
                return Err(format!("{name} needs a value"));
            }
            Ok(Some(self.rest.remove(i).to_string()))
        }
        fn values(&mut self, name: &str) -> Result<Vec<String>, String> {
            let mut out = Vec::new();
            while let Some(v) = self.value(name)? {
                out.push(v);
            }
            Ok(out)
        }
        fn none_left(&self) -> Result<(), String> {
            match self.rest.first() {
                None => Ok(()),
                Some(x) => Err(format!("unexpected argument {x:?}")),
            }
        }
    }
    let (head, tail) = match words.split_first() {
        Some((h, t)) => (*h, t.to_vec()),
        None => return Err("nothing to do".into()),
    };
    let mut f = Flags { rest: tail };
    let command = match head {
        "check" => Command::Check,
        "flows" => Command::Flows,
        "auth" => {
            let Some(sub) = (!f.rest.is_empty()).then(|| f.rest.remove(0)) else {
                return Err("auth needs a command".into());
            };
            match sub {
                "init" => Command::Init {
                    generate: f.has("--generate"),
                    force: f.has("--force"),
                    new_folder: f.has("--new-folder"),
                    password_file: f.value("--password-file")?.map(PathBuf::from),
                },
                "setup-code" => Command::SetupCode,
                "reset-password" => Command::ResetPassword {
                    generate: f.has("--generate"),
                    password_file: f.value("--password-file")?.map(PathBuf::from),
                },
                "session-link" => Command::SessionLink,
                "status" => Command::Status,
                "sessions" => {
                    if f.rest.first() != Some(&"revoke-all") {
                        return Err("sessions takes `revoke-all`".into());
                    }
                    f.rest.remove(0);
                    Command::SessionsRevokeAll {
                        devices: f.has("--devices"),
                    }
                }
                "token" => {
                    let Some(action) = (!f.rest.is_empty()).then(|| f.rest.remove(0)) else {
                        return Err("token takes create, list or revoke".into());
                    };
                    match action {
                        "create" => Command::TokenCreate {
                            preset: f
                                .value("--preset")?
                                .ok_or_else(|| "token create needs --preset".to_string())?,
                            scopes: f.values("--scope")?,
                            ttl: f.value("--ttl")?,
                            label: f.value("--label")?,
                            origin: f.value("--origin")?,
                        },
                        "list" => Command::TokenList,
                        "revoke" => {
                            if f.rest.is_empty() {
                                return Err("token revoke needs an id".into());
                            }
                            Command::TokenRevoke {
                                id: f.rest.remove(0).to_string(),
                            }
                        }
                        other => return Err(format!("token has no {other:?}")),
                    }
                }
                other => return Err(format!("auth has no command {other:?}")),
            }
        }
        other => return Err(format!("no such command {other:?}")),
    };
    // `flows` takes arguments this build does not read; every other command must have used all of its own.
    if !matches!(command, Command::Flows) {
        f.none_left()?;
    }
    Ok(command)
}

// ---- the data folder ---------------------------------------------------------------------------------

fn data_dir(env: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if let Some(d) = env("OAIY_DATA_DIR").filter(|d| !d.trim().is_empty()) {
        return Some(PathBuf::from(d));
    }
    let home = env("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| env("USERPROFILE").filter(|h| !h.is_empty()))?;
    Some(PathBuf::from(home).join(".oaiy-server"))
}

/// Who owns a folder: its user id and, when the password file names it, the user.
#[cfg(unix)]
fn folder_owner(path: &Path) -> Option<(u32, String)> {
    use std::os::unix::fs::MetadataExt;
    let uid = std::fs::metadata(path).ok()?.uid();
    let name = std::fs::read_to_string("/etc/passwd")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|l| {
                let mut f = l.split(':');
                let (name, _, id) = (f.next()?, f.next()?, f.next()?);
                (id.parse::<u32>().ok() == Some(uid)).then(|| name.to_string())
            })
        })
        // `sudo -u '#1000'` names a user by number.
        .unwrap_or_else(|| format!("#{uid}"));
    Some((uid, name))
}

#[cfg(not(unix))]
fn folder_owner(_path: &Path) -> Option<(u32, String)> {
    None
}

/// The effective user id of this process, where there is one.
#[cfg(unix)]
fn effective_uid() -> Option<u32> {
    let out = std::process::Command::new("id").arg("-u").output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[cfg(not(unix))]
fn effective_uid() -> Option<u32> {
    None
}

// ---- entry points ------------------------------------------------------------------------------------

/// The real thing: the process's environment and terminal.
pub fn run_process(args: &[String]) -> i32 {
    let env = |name: &str| std::env::var(name).ok();
    let mut out = io::stdout();
    let mut err = io::stderr();
    let mut prompt = |text: &str| rpassword::prompt_password(text);
    let mut io = Io {
        env: &env,
        out: &mut out,
        err: &mut err,
        prompt: &mut prompt,
        euid: effective_uid(),
    };
    run(args, &mut io)
}

/// Run one command. The exit code.
pub fn run(args: &[String], io: &mut Io<'_>) -> i32 {
    let command = match parse(args) {
        Ok(c) => c,
        Err(why) => {
            let _ = writeln!(io.err, "oaiy-server: {why}\n{USAGE}");
            return EXIT_USAGE;
        }
    };
    match command {
        Command::Flows => {
            let _ = writeln!(
                io.err,
                "oaiy-server flows: flow approvals are not part of this build (they arrive with the flow authority package)"
            );
            EXIT_USAGE
        }
        Command::Check => check_command(io),
        other => auth_command(other, io),
    }
}

fn say(io: &mut Io<'_>, text: impl AsRef<str>) {
    let _ = writeln!(io.out, "{}", text.as_ref());
}

fn fail(io: &mut Io<'_>, text: impl AsRef<str>) -> i32 {
    let _ = writeln!(io.err, "oaiy-server: {}", text.as_ref());
    EXIT_FAILED
}

fn refuse(io: &mut Io<'_>, text: impl AsRef<str>) -> i32 {
    let _ = writeln!(io.err, "oaiy-server: {}", text.as_ref());
    EXIT_CONFIG
}

// ---- `auth` ---------------------------------------------------------------------------------------------

/// What a run works against.
enum Way {
    /// The server is stopped: the files, under the lock this store holds.
    Stopped(Box<AuthStore>),
    /// The server is running: its console routes.
    Running(Client),
}

struct Client {
    base: String,
    token: String,
    http: reqwest::blocking::Client,
}

impl Client {
    /// Read `console.json` and `console.token`.
    fn connect(auth_dir: &Path) -> Result<Client, String> {
        let read = |name: &str| -> Result<String, String> {
            std::fs::read_to_string(auth_dir.join(name)).map_err(|e| {
                format!(
                    "the server is running, but {} cannot be read ({e}): the console cannot reach it, and nothing was done. Run as the server's user (see the folder's owner above), or stop the server and run the command again",
                    auth_dir.join(name).display()
                )
            })
        };
        let info: Value = serde_json::from_str(&read(INFO_FILE)?)
            .map_err(|e| format!("{} is not JSON: {e}", auth_dir.join(INFO_FILE).display()))?;
        let port = info["port"]
            .as_u64()
            .and_then(|p| u16::try_from(p).ok())
            .ok_or_else(|| format!("{} names no port", auth_dir.join(INFO_FILE).display()))?;
        let token = read(TOKEN_FILE)?.trim().to_string();
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .map_err(|e| format!("no HTTP client: {e}"))?;
        Ok(Client {
            base: format!("http://127.0.0.1:{port}"),
            token,
            http,
        })
    }

    /// One call: the status and the JSON body.
    fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<(u16, Value), String> {
        let mut request = self
            .http
            .request(method, format!("{}{path}", self.base))
            .bearer_auth(&self.token)
            .header("content-type", "application/json");
        request = request.body(body.unwrap_or_else(|| json!({})).to_string());
        let response = request.send().map_err(|e| {
            format!(
                "the server did not answer on {} ({e}): its console files may be stale; nothing was done",
                self.base
            )
        })?;
        let status = response.status().as_u16();
        let text = response.text().unwrap_or_default();
        Ok((status, serde_json::from_str(&text).unwrap_or(Value::Null)))
    }
}

/// What a refusal of the server says, in a line.
fn refusal(status: u16, body: &Value) -> String {
    let code = body["error"]["code"].as_str().unwrap_or("error");
    let message = body["error"]["message"].as_str().unwrap_or("");
    let reasons = body["reasons"]
        .as_array()
        .map(|r| {
            format!(
                " ({})",
                r.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .unwrap_or_default();
    format!("the server answered {status} {code}{reasons}: {message}")
}

fn dashboard_origin(env: &dyn Fn(&str) -> Option<String>, port: Option<u16>) -> String {
    env("OAIY_PUBLIC_URL")
        .map(|u| u.trim().trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| format!("http://dash.oaiy.localhost:{}", port.unwrap_or(17972)))
}

/// A console run as root against a folder that another user owns would make files owned by root in it (or, with no
/// OAIY_DATA_DIR, a folder of its own under root's home): refuse, and say how to run it.
fn root_refusal(euid: Option<u32>, who: &Option<(u32, String)>, data: &Path) -> Option<String> {
    match (euid, who) {
        (Some(0), Some((uid, name))) if *uid != 0 => Some(format!(
            "this is root, and {} belongs to {name}: run it as that user (`sudo -u {name} oaiy-server ...`), or files in it would be made by root",
            data.display()
        )),
        _ => None,
    }
}

fn auth_command(command: Command, io: &mut Io<'_>) -> i32 {
    let Some(data) = data_dir(io.env) else {
        return fail(
            io,
            "set OAIY_DATA_DIR: there is no HOME or USERPROFILE to derive a data folder from",
        );
    };
    let auth_dir = data.join("auth");
    // The data folder and who owns it, on every run: `sudo oaiy-server auth init` as root would otherwise make
    // /root/.oaiy-server and report success.
    let who = folder_owner(&data);
    match &who {
        Some((uid, name)) => say(
            io,
            format!(
                "data folder: {} (owned by {name}, uid {uid})",
                data.display()
            ),
        ),
        None => say(io, format!("data folder: {}", data.display())),
    }
    if let Some(why) = root_refusal(io.euid, &who, &data) {
        return fail(io, why);
    }
    let creating = matches!(command, Command::Init { .. });
    // init makes nothing that a running server has not made: a folder that is not there is the sign of a console
    // that is not looking at the server's data (the unit's environment differs from this one's), and an owner made
    // there would be reported as a success and be no use. A new install says so with --new-folder.
    if let Command::Init {
        new_folder: false, ..
    } = &command
    {
        if !auth_dir.is_dir() {
            return fail(
                io,
                format!(
                    "{} does not exist: a server that has run has made it, so this is not the data folder of a server that runs (a service reads OAIY_DATA_DIR from its environment file, /etc/oaiy/oaiy.env, and oaiyctl reads the same one). To make a data folder for a server that has not run yet, add --new-folder",
                    auth_dir.display()
                ),
            );
        }
    }
    if !creating && !data.is_dir() {
        return fail(
            io,
            format!(
                "{} is not a folder: this is not the server's data folder (set OAIY_DATA_DIR to it). Only `auth init` makes one",
                data.display()
            ),
        );
    }
    if !creating && !auth_dir.is_dir() {
        return fail(
            io,
            format!(
                "{} does not exist yet: the server makes it when it first runs, or `auth init` does. Only `auth init` makes a folder",
                auth_dir.display()
            ),
        );
    }
    if creating {
        if let Err(e) = crate::secret_file::create_private_dir(&auth_dir) {
            return fail(io, format!("cannot make {}: {e}", auth_dir.display()));
        }
    }
    // Which path: the lock says whether the server runs.
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let path = match AuthStore::open(
        &auth_dir,
        Host::Server,
        clock.clone(),
        Arc::new(SecureWriter),
        None,
    ) {
        Ok(store) => Way::Stopped(Box::new(store)),
        Err(StoreError::Lock(LockError::InUse { .. })) => match Client::connect(&auth_dir) {
            Ok(c) => Way::Running(c),
            Err(why) => return fail(io, why),
        },
        Err(e @ StoreError::Lock(LockError::Io { .. })) => return fail(io, e.to_string()),
        Err(e) => return refuse(io, e.to_string()),
    };
    match path {
        Way::Stopped(store) => stopped(command, &store, &auth_dir, &clock, io),
        Way::Running(client) => running(command, &client, &auth_dir, io),
    }
}

/// The audit line of a command run against the files.
fn audit_offline(auth_dir: &Path, clock: &Arc<dyn Clock>, command: &Command) {
    let log = AuditLog::open(auth_dir, clock.clone(), false);
    let actor = Actor {
        id: "console".into(),
        kind: "con".into(),
        label: "console".into(),
    };
    log.critical(
        "console.command",
        Some(&actor),
        &AuditContext::default(),
        json!({ "command": command.name(), "offline": true }),
    );
}

/// A new password from the terminal, a file or the generator. Judged by the policy of 4.7.2.
fn new_password(
    generate: bool,
    file: &Option<PathBuf>,
    io: &mut Io<'_>,
) -> Result<Zeroizing<String>, i32> {
    let hosts: Vec<String> = ["OAIY_PUBLIC_URL", "OAIY_AGENT_URL", "OAIY_FLOWS_URL"]
        .iter()
        .filter_map(|v| (io.env)(v))
        .map(|u| {
            u.trim_start_matches("https://")
                .trim_end_matches('/')
                .to_string()
        })
        .collect();
    let extra = policy::extra_inputs(hosts.iter().map(String::as_str));
    let reasons_text = |r: &[Reason]| r.iter().map(|x| x.code()).collect::<Vec<_>>().join(", ");
    if generate {
        let mut fill = |buf: &mut [u8]| token::os_random(buf);
        let phrase = policy::generate(&mut fill, &extra).map_err(|e| fail(io, e.to_string()))?;
        if let Err(r) = policy::judge(&phrase, &extra) {
            return Err(fail(
                io,
                format!(
                    "a generated passphrase was refused ({}): this is a bug",
                    reasons_text(&r)
                ),
            ));
        }
        say(
            io,
            "Generated password (write it down: it is not shown again):",
        );
        say(io, format!("  {}", phrase.as_str()));
        return Ok(phrase);
    }
    if let Some(path) = file {
        let text = std::fs::read_to_string(path)
            .map_err(|e| fail(io, format!("cannot read {}: {e}", path.display())))?;
        let text = Zeroizing::new(text);
        let trimmed = text.strip_suffix('\n').unwrap_or(&text);
        let trimmed = trimmed.strip_suffix('\r').unwrap_or(trimmed);
        let password = Zeroizing::new(trimmed.to_string());
        return match policy::judge(&password, &extra) {
            Ok(_) => Ok(password),
            Err(r) => Err(fail(
                io,
                format!("that password is not accepted ({})", reasons_text(&r)),
            )),
        };
    }
    for _ in 0..3 {
        let first = (io.prompt)("New password: ")
            .map_err(|e| fail(io, format!("cannot read a password: {e}")))?;
        let first = Zeroizing::new(first);
        let second =
            (io.prompt)("Again: ").map_err(|e| fail(io, format!("cannot read a password: {e}")))?;
        let second = Zeroizing::new(second);
        if !token::secrets_equal(first.as_bytes(), second.as_bytes()) {
            say(io, "The two do not match: try again.");
            continue;
        }
        match policy::judge(&first, &extra) {
            Ok(_) => return Ok(first),
            Err(r) => say(
                io,
                format!(
                    "That password is not accepted ({}): 16 to 128 characters that are not easy to guess. `--generate` makes one.",
                    reasons_text(&r)
                ),
            ),
        }
    }
    Err(fail(io, "no password was set"))
}

fn stopped(
    command: Command,
    store: &AuthStore,
    auth_dir: &Path,
    clock: &Arc<dyn Clock>,
    io: &mut Io<'_>,
) -> i32 {
    let has_owner = store.owner().is_some();
    let writer = Arc::new(SecureWriter);
    let exit = match &command {
        Command::SetupCode => {
            if has_owner {
                return fail(io, "an owner login already exists: a setup code is only for the first one. Use `auth reset-password`");
            }
            let setup = match SetupCode::open(auth_dir, clock.clone(), writer.clone()) {
                Ok(s) => s,
                Err(e) => return refuse(io, e.to_string()),
            };
            let random: super::store::Random = Arc::new(token::os_random);
            match setup.make(&random) {
                Ok(code) => {
                    let dash = dashboard_origin(io.env, None);
                    say(io, format!("Setup code: {code}"));
                    say(
                        io,
                        format!(
                            "Valid for 24 hours (or until used); 100 wrong guesses burn it. Open {dash}/setup and enter it with the password you choose."
                        ),
                    );
                    EXIT_OK
                }
                Err(e) => fail(io, e.to_string()),
            }
        }
        Command::Init { generate, password_file, force, .. } => {
            if has_owner && !*force {
                return fail(io, "an owner login already exists; `--force` makes a new password (revoking every session and device)");
            }
            match new_password(*generate, password_file, io) {
                Ok(password) => set_password_offline(store, auth_dir, clock, &password, has_owner, io),
                Err(code) => code,
            }
        }
        Command::ResetPassword { generate, password_file } => match new_password(*generate, password_file, io) {
            Ok(password) => set_password_offline(store, auth_dir, clock, &password, has_owner, io),
            Err(code) => code,
        },
        Command::SessionLink => fail(
            io,
            "a session link is made by the running server (it keeps only the hash, in memory): start the server and run this again",
        ),
        Command::SessionsRevokeAll { devices } => {
            let revoked = store.revoke_where("revoked", &|r| r.kind == Kind::Ses);
            let mut removed = 0;
            if *devices {
                if let Some(o) = store.owner() {
                    if let Ok(mut doc) = OwnerDoc::from_value(&o.doc) {
                        removed = doc.devices.len();
                        doc.devices.clear();
                        if let Err(e) = owner::write(writer.as_ref(), auth_dir, &doc) {
                            return fail(io, format!("owner.json could not be written: {e}"));
                        }
                    }
                }
            }
            if let Err(e) = store.flush() {
                return fail(io, format!("credentials.json could not be written: {e}"));
            }
            say(io, format!("revoked {} session(s) and {removed} device(s)", revoked.len()));
            EXIT_OK
        }
        Command::Status => {
            let now = clock.now_ms();
            let live = |kind: Kind| {
                store
                    .records()
                    .iter()
                    .filter(|r| r.kind == kind && r.revoked_ms.is_none() && now < r.expires_ms)
                    .count()
            };
            let devices = store
                .owner()
                .and_then(|o| OwnerDoc::from_value(&o.doc).ok())
                .map_or(0, |d| d.devices.len());
            let setup = SetupCode::open(auth_dir, clock.clone(), writer.clone())
                .map(|s| s.status())
                .unwrap_or(Status::None);
            say(io, "server: not running (the files are read as they are)");
            say(io, format!("login configured: {}", if has_owner { "yes" } else { "no" }));
            say(io, format!("setup code: {}", setup.name()));
            say(io, format!("sessions: {}, devices: {devices}, paired tokens: {}", live(Kind::Ses), live(Kind::Pat)));
            say(io, format!("storage: {}", store.storage().name()));
            EXIT_OK
        }
        Command::TokenList => {
            let now = clock.now_ms();
            let mut any = false;
            for r in store.records().into_iter().filter(|r| r.kind == Kind::Pat && r.revoked_ms.is_none() && now < r.expires_ms) {
                any = true;
                say(
                    io,
                    format!(
                        "{}  {}  preset {}  expires in {} days",
                        r.id,
                        r.label,
                        r.preset.as_deref().unwrap_or("-"),
                        (r.expires_ms - now) / DAY_MS
                    ),
                );
            }
            if !any {
                say(io, "no paired tokens");
            }
            EXIT_OK
        }
        Command::TokenRevoke { id } => {
            if store.revoke(id, "revoked") {
                let _ = store.flush();
                say(io, format!("revoked {id}"));
                EXIT_OK
            } else {
                fail(io, format!("no live credential {id}"))
            }
        }
        Command::TokenCreate { preset, scopes, ttl, label, origin } => {
            token_create(store, preset, scopes, ttl.as_deref(), label.as_deref(), origin.as_deref(), io)
        }
        Command::Check | Command::Flows => EXIT_USAGE,
    };
    if exit == EXIT_OK {
        audit_offline(auth_dir, clock, &command);
    }
    exit
}

/// Set the owner's password in the files: a new owner, or a changed password that revokes every session and device.
fn set_password_offline(
    store: &AuthStore,
    auth_dir: &Path,
    clock: &Arc<dyn Clock>,
    password: &str,
    has_owner: bool,
    io: &mut Io<'_>,
) -> i32 {
    let normalised = policy::normalise(password);
    let phc = match Argon2Engine::production().hash(normalised.as_bytes()) {
        Ok(p) => p,
        Err(e) => return fail(io, e.to_string()),
    };
    let writer = SecureWriter;
    if has_owner {
        let Some(o) = store.owner() else {
            return fail(io, "the owner file disappeared");
        };
        let mut doc = match OwnerDoc::from_value(&o.doc) {
            Ok(d) => d,
            Err(e) => return refuse(io, format!("owner.json cannot be read: {e}")),
        };
        doc.password = phc;
        doc.password_changed_ms = clock.now_ms();
        doc.devices.clear();
        if let Err(e) = owner::write(&writer, auth_dir, &doc) {
            return fail(io, format!("owner.json could not be written: {e}"));
        }
        let revoked = store.revoke_where("password_changed", &|r| r.kind == Kind::Ses);
        if let Err(e) = store.flush() {
            return fail(io, format!("credentials.json could not be written: {e}"));
        }
        say(
            io,
            format!(
                "password changed; {} session(s) and every device revoked",
                revoked.len()
            ),
        );
    } else {
        let doc = OwnerDoc::new(clock.now_ms(), phc);
        match owner::create_exclusive(&writer, auth_dir, &doc) {
            Ok(()) => {}
            Err(CreateError::Exists) => return fail(io, "an owner login was made meanwhile"),
            Err(CreateError::Io(e)) => {
                return fail(io, format!("owner.json could not be written: {e}"))
            }
        }
        // A setup code has nothing left to open.
        if let Ok(setup) = SetupCode::open(auth_dir, clock.clone(), Arc::new(SecureWriter)) {
            setup.consume();
        }
        say(
            io,
            "the owner login is made: sign in on the dashboard with that password",
        );
    }
    EXIT_OK
}

/// `--ttl 30d`, `12h`, `90m`, `600s`, or a bare number of seconds.
pub fn parse_ttl(text: &str) -> Option<u64> {
    let text = text.trim();
    let (digits, unit) = match text.char_indices().find(|(_, c)| !c.is_ascii_digit()) {
        Some((i, _)) => text.split_at(i),
        None => (text, "s"),
    };
    let n: u64 = digits.parse().ok().filter(|n| *n > 0)?;
    let unit_ms = match unit {
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => DAY_MS,
        _ => return None,
    };
    n.checked_mul(unit_ms)
}

fn token_create(
    store: &AuthStore,
    preset: &str,
    extra_scopes: &[String],
    ttl: Option<&str>,
    label: Option<&str>,
    origin: Option<&str>,
    io: &mut Io<'_>,
) -> i32 {
    let Some(p) =
        Preset::by_name(preset).filter(|p| p.grantable_on_request() || *p == Preset::CliAdmin)
    else {
        return fail(io, format!("{preset:?} is not a preset a token can hold (the tokens' presets: cli, cli-admin, readonly, mcp, agent, flows, flows-web, formlogic, companion)"));
    };
    let mut scopes = p.scopes();
    if !extra_scopes.is_empty() {
        match ScopeSet::parse(extra_scopes.iter().map(String::as_str)) {
            Ok(more) => scopes = scopes.union(&more),
            Err(e) => return fail(io, e.to_string()),
        }
    }
    // A dangerous scope is a 24-hour token; the rest live 90 days unless asked otherwise.
    let default_ttl = if scopes.has_dangerous() {
        DAY_MS
    } else {
        90 * DAY_MS
    };
    let ttl_ms = match ttl {
        Some(t) => match parse_ttl(t) {
            Some(ms) => ms,
            None => {
                return fail(
                    io,
                    format!("{t:?} is not a lifetime: use 30d, 12h, 90m or 600s"),
                )
            }
        },
        None => default_ttl,
    };
    let mut spec = MintSpec::new(Kind::Pat, label.unwrap_or("console token"), scopes, ttl_ms);
    spec.preset = Some(p);
    if let Some(o) = origin {
        match super::store::canonical_origin(o) {
            Some(c) => spec.origins = vec![c],
            None => return fail(io, format!("{o:?} is not an origin (scheme://host[:port])")),
        }
    }
    match store.mint(spec) {
        Ok(minted) => {
            let _ = store.flush();
            say(io, format!("id: {}", minted.id));
            say(io, format!("token (shown once): {}", minted.token));
            EXIT_OK
        }
        Err(MintFailure::ScopeNotGrantable(why)) => fail(io, why),
        Err(e) => fail(io, e.to_string()),
    }
}

/// The commands against a running server.
fn running(command: Command, client: &Client, auth_dir: &Path, io: &mut Io<'_>) -> i32 {
    use reqwest::Method;
    let port = client
        .base
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok());
    let outcome: Result<(u16, Value), String> = match &command {
        Command::SetupCode => client.call(Method::POST, "/api/auth/console/setup-code", None),
        Command::SessionLink => client.call(Method::POST, "/api/auth/console/session-link", None),
        Command::Status => client.call(Method::GET, "/api/auth/console/status", None),
        Command::SessionsRevokeAll { devices } => client.call(
            Method::POST,
            "/api/auth/console/sessions/revoke-all",
            Some(json!({ "devices": devices })),
        ),
        Command::Init { generate, password_file, force, .. } => {
            // `init` on a running server asks its status first: it refuses when an owner exists unless `--force`.
            match client.call(Method::GET, "/api/auth/console/status", None) {
                Ok((200, s)) if s["loginConfigured"] == true && !*force => {
                    return fail(io, "an owner login already exists; `--force` makes a new password (revoking every session and device)")
                }
                Ok((200, _)) => {}
                Ok((status, body)) => return fail(io, refusal(status, &body)),
                Err(why) => return fail(io, why),
            }
            match new_password(*generate, password_file, io) {
                Ok(password) => client.call(
                    Method::POST,
                    "/api/auth/console/reset-password",
                    Some(json!({ "password": password.as_str() })),
                ),
                Err(code) => return code,
            }
        }
        Command::ResetPassword { generate, password_file } => match new_password(*generate, password_file, io) {
            Ok(password) => client.call(
                Method::POST,
                "/api/auth/console/reset-password",
                Some(json!({ "password": password.as_str() })),
            ),
            Err(code) => return code,
        },
        Command::TokenCreate { .. } | Command::TokenList | Command::TokenRevoke { .. } => {
            return fail(
                io,
                "the server is running, and its credentials API is not part of this build: stop the server and run this again to edit its files",
            )
        }
        Command::Check | Command::Flows => return EXIT_USAGE,
    };
    let (status, body) = match outcome {
        Ok(x) => x,
        Err(why) => return fail(io, why),
    };
    if status != 200 {
        return fail(io, refusal(status, &body));
    }
    match &command {
        Command::SetupCode => {
            let dash = dashboard_origin(io.env, port);
            say(
                io,
                format!("Setup code: {}", body["code"].as_str().unwrap_or("?")),
            );
            say(
                io,
                format!(
                    "Valid for {} hours (or until used); {} wrong guesses burn it. Open {dash}/setup and enter it with the password you choose.",
                    VALID_MS / 3_600_000,
                    body["attemptsLeft"].as_u64().unwrap_or(0)
                ),
            );
        }
        Command::SessionLink => {
            say(
                io,
                format!(
                    "Session link (single use, {} minutes):",
                    body["expiresInSeconds"].as_u64().unwrap_or(0) / 60
                ),
            );
            say(io, format!("  {}", body["url"].as_str().unwrap_or("?")));
        }
        Command::Status => {
            say(io, "server: running");
            for line in status_lines(&body) {
                say(io, line);
            }
        }
        Command::SessionsRevokeAll { .. } => say(
            io,
            format!(
                "revoked {} session(s) and {} device(s)",
                body["sessions"].as_u64().unwrap_or(0),
                body["devices"].as_u64().unwrap_or(0)
            ),
        ),
        Command::Init { .. } | Command::ResetPassword { .. } => {
            if body["created"] == true {
                say(
                    io,
                    "the owner login is made: sign in on the dashboard with that password",
                );
            } else {
                say(io, "password changed; every session and device is revoked");
            }
        }
        _ => {}
    }
    let _ = auth_dir;
    EXIT_OK
}

// ---- `check` ------------------------------------------------------------------------------------------

/// What `check` found: the violations (a refusal to start: exit 78) and the warnings (things ignored, which a
/// person should know).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub violations: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validate the configuration and the data folder without changing or locking anything: the mode, the port, the
/// data folder and every file under `<data>/auth`. Every violation is listed, not only the first.
pub fn check(env: &dyn Fn(&str) -> Option<String>) -> Report {
    let mut report = Report::default();
    let get = |name: &str| {
        env(name)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    // The port.
    if let Some(p) = get("OAIY_SERVER_PORT") {
        if p.parse::<u16>().ok().filter(|n| *n != 0).is_none() {
            report
                .violations
                .push(format!("OAIY_SERVER_PORT={p:?} is not a port (1 to 65535)"));
        }
    }
    // The bind and the exposure.
    let bind_all = get("OAIY_SERVER_BIND").is_some_and(|b| b.eq_ignore_ascii_case("lan"));
    let exposure = Exposure::compute(bind_all, get("OAIY_PUBLIC_URL").is_some());
    // The data folder and the auth files.
    let data = data_dir(env);
    let mut owner_exists = false;
    match &data {
        None => report
            .violations
            .push("no data folder: set OAIY_DATA_DIR (there is no HOME or USERPROFILE)".into()),
        Some(dir) => {
            if dir.exists() && !dir.is_dir() {
                report
                    .violations
                    .push(format!("{} is not a folder", dir.display()));
            } else if dir.is_dir() {
                owner_exists = inspect_auth_dir(&dir.join("auth"), &mut report);
            }
        }
    }
    // The mode (a server with the web login defaults to scoped and refuses legacy).
    match super::login::server_mode(get("OAIY_ACCESS_MODE").as_deref()) {
        Ok(mode) => {
            if let Err(refusal) = validate_mode(mode, exposure, owner_exists) {
                report.violations.push(refusal.to_string());
            }
        }
        Err(refusal) => report.violations.push(refusal.to_string()),
    }
    // What the server reads leniently and ignores: worth a warning here.
    let (_, warnings) = super::guard::GuardConfig::from_env(env, bind_all, false, 17972);
    report.warnings.extend(warnings);
    // A list with a typo in it is refused, as the server refuses it: the restriction is a security setting.
    if let Some(list) = get("OAIY_LOGIN_ALLOW") {
        if let Err(refusal) = super::login::parse_login_allow(&list) {
            report.violations.push(refusal.to_string());
        }
    }
    // Flow authority is not here yet: a signed-in dashboard session (not elevated) can write and run flows.
    if exposure != Exposure::Local {
        report.warnings.push(format!(
            "flows: on a {} install a signed-in dashboard session can write and run flows without elevating (flows.write and runs.write are not dangerous scopes: flow authority is ACC-05, not built yet); do not put this install on a network with flows in use until it is (README, The web login)",
            exposure.name()
        ));
    }
    report
}

/// The files of `<data>/auth`, read the way the server reads them. Whether an owner exists.
fn inspect_auth_dir(auth: &Path, report: &mut Report) -> bool {
    if !auth.exists() {
        return false;
    }
    if !auth.is_dir() {
        report
            .violations
            .push(format!("{} is not a folder", auth.display()));
        return false;
    }
    let mut owner_exists = false;
    for name in [
        "owner.json",
        "credentials.json",
        "setup-code.json",
        "throttle.json",
    ] {
        let path = auth.join(name);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            // The throttle file is a memory of failures, not a credential: the server starts with no blocks.
            Err(e) if name == "throttle.json" => {
                report.warnings.push(format!(
                    "cannot read {}: {e}: the server starts with no blocks",
                    path.display()
                ));
                continue;
            }
            Err(e) => {
                report
                    .violations
                    .push(format!("cannot read {}: {e}", path.display()));
                continue;
            }
        };
        let doc: Result<Value, _> = serde_json::from_str(&text);
        match (name, doc) {
            ("owner.json", Err(e)) => report.violations.push(format!(
                "{} cannot be read ({e}): a mangled owner file must not reopen setup; restore it or run `oaiy-server auth init --force`",
                path.display()
            )),
            ("owner.json", Ok(v)) => {
                owner_exists = true;
                match v["v"].as_u64() {
                    // The server reads it as it does the login: a file it cannot make an owner of is a refusal to
                    // start (a hash it cannot use is not: that is a password that does not match).
                    Some(1) => {
                        if let Err(detail) = super::owner::OwnerDoc::from_value(&v) {
                            report.violations.push(format!(
                                "{} cannot be read as an owner ({detail}): a mangled owner file must not reopen setup; restore it or run `oaiy-server auth init --force`",
                                path.display()
                            ));
                        }
                    }
                    Some(other) => report.violations.push(format!(
                        "{} was written by a newer OAIY (version {other}; this one reads version 1): update OAIY",
                        path.display()
                    )),
                    None => report.violations.push(format!("{} has no version", path.display())),
                }
            }
            ("credentials.json", Err(e)) => report.warnings.push(format!(
                "{} cannot be read ({e}): the server moves it aside and starts with no paired apps",
                path.display()
            )),
            ("credentials.json", Ok(v)) => {
                if let Some(other) = v["v"].as_u64().filter(|n| *n != 1) {
                    report.violations.push(format!(
                        "{} was written by a newer OAIY (version {other}; this one reads version 1): update OAIY",
                        path.display()
                    ));
                }
            }
            (_, Err(e)) => report.warnings.push(format!("{} cannot be read ({e}): it is ignored", path.display())),
            _ => {}
        }
    }
    owner_exists
}

fn check_command(io: &mut Io<'_>) -> i32 {
    let report = check(io.env);
    for w in &report.warnings {
        let _ = writeln!(io.err, "oaiy-server check: warning: {w}");
    }
    if report.violations.is_empty() {
        say(io, "oaiy-server check: ok");
        return EXIT_OK;
    }
    for v in &report.violations {
        let _ = writeln!(io.err, "oaiy-server check: {v}");
    }
    EXIT_CONFIG
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::TempDir;
    use std::collections::BTreeMap;

    fn args(words: &str) -> Vec<String> {
        words.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn every_command_of_the_design_parses() {
        assert_eq!(
            parse(&args("auth init")).unwrap(),
            Command::Init {
                generate: false,
                password_file: None,
                force: false,
                new_folder: false
            }
        );
        assert_eq!(
            parse(&args(
                "auth init --generate --force --new-folder --password-file /tmp/pw"
            ))
            .unwrap(),
            Command::Init {
                generate: true,
                password_file: Some("/tmp/pw".into()),
                force: true,
                new_folder: true
            }
        );
        assert_eq!(parse(&args("auth setup-code")).unwrap(), Command::SetupCode);
        assert_eq!(
            parse(&args("auth reset-password --generate")).unwrap(),
            Command::ResetPassword {
                generate: true,
                password_file: None
            }
        );
        assert_eq!(
            parse(&args("auth session-link")).unwrap(),
            Command::SessionLink
        );
        assert_eq!(parse(&args("auth status")).unwrap(), Command::Status);
        assert_eq!(parse(&args("check")).unwrap(), Command::Check);
        assert_eq!(parse(&args("flows list")).unwrap(), Command::Flows);
        assert_eq!(
            parse(&args("flows approve abc --hash 00")).unwrap(),
            Command::Flows
        );
        assert_eq!(
            parse(&args("auth sessions revoke-all --devices")).unwrap(),
            Command::SessionsRevokeAll { devices: true }
        );
        assert_eq!(
            parse(&args("auth sessions revoke-all")).unwrap(),
            Command::SessionsRevokeAll { devices: false }
        );
        assert_eq!(
            parse(&args("auth token create --preset cli --scope ai.read --scope ai.use --ttl 30d --label ci --origin https://x.example")).unwrap(),
            Command::TokenCreate {
                preset: "cli".into(),
                scopes: vec!["ai.read".into(), "ai.use".into()],
                ttl: Some("30d".into()),
                label: Some("ci".into()),
                origin: Some("https://x.example".into()),
            }
        );
        assert_eq!(parse(&args("auth token list")).unwrap(), Command::TokenList);
        assert_eq!(
            parse(&args("auth token revoke 0123456789abcdef")).unwrap(),
            Command::TokenRevoke {
                id: "0123456789abcdef".into()
            }
        );
    }

    #[test]
    fn what_is_misused_is_told_and_never_guessed() {
        for bad in [
            "",
            "auth",
            "auth nonsense",
            "auth init --nonsense",
            "auth init extra",
            "auth setup-code now",
            "auth token",
            "auth token create",
            "auth token create --preset",
            "auth token revoke",
            "auth token frobnicate",
            "auth sessions",
            "auth sessions revoke-all --all",
            "auth init --password-file",
            "nonsense",
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_lifetime_is_a_number_and_a_unit() {
        assert_eq!(parse_ttl("30d"), Some(30 * DAY_MS));
        assert_eq!(parse_ttl("12h"), Some(12 * 3_600_000));
        assert_eq!(parse_ttl("90m"), Some(90 * 60_000));
        assert_eq!(parse_ttl("600s"), Some(600_000));
        assert_eq!(parse_ttl("600"), Some(600_000));
        for bad in [
            "",
            "d",
            "0d",
            "-1d",
            "3w",
            "1.5h",
            "99999999999999999999d",
            "1 d",
        ] {
            assert_eq!(parse_ttl(bad), None, "{bad:?}");
        }
    }

    fn env_of(pairs: &[(&str, String)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        move |n: &str| map.get(n).cloned()
    }

    fn check_in(dir: &TempDir, extra: &[(&str, &str)]) -> Report {
        let mut pairs: Vec<(&str, String)> = vec![("OAIY_DATA_DIR", dir.0.display().to_string())];
        pairs.extend(extra.iter().map(|(k, v)| (*k, v.to_string())));
        check(&env_of(&pairs))
    }

    #[test]
    fn check_passes_on_a_folder_that_does_not_exist_yet_and_on_an_empty_one() {
        let dir = TempDir::new("check-empty");
        let r = check_in(&dir, &[]);
        assert_eq!(r, Report::default(), "{r:?}");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        assert_eq!(check_in(&dir, &[]), Report::default());
    }

    #[test]
    fn check_lists_every_violation_at_once_and_reads_nothing_it_should_not() {
        let dir = TempDir::new("check-many");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        std::fs::write(auth.join("owner.json"), "{ not json").unwrap();
        std::fs::write(auth.join("credentials.json"), r#"{"v":9,"credentials":[]}"#).unwrap();
        // A directory where a file belongs: a read error that is not "not there".
        std::fs::create_dir(auth.join("setup-code.json")).unwrap();
        let r = check_in(
            &dir,
            &[("OAIY_SERVER_PORT", "70000"), ("OAIY_ACCESS_MODE", "scopd")],
        );
        let text = r.violations.join("\n");
        for part in [
            "OAIY_SERVER_PORT",
            "owner.json",
            "credentials.json",
            "setup-code.json",
            "OAIY_ACCESS_MODE",
        ] {
            assert!(text.contains(part), "{part} missing from:\n{text}");
        }
        assert_eq!(r.violations.len(), 5, "{text}");
        // Nothing was created, moved or locked.
        assert!(!auth.join(".lock").exists());
        assert!(std::fs::read_dir(&auth).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("corrupt")));
    }

    #[test]
    fn check_refuses_an_owner_file_of_another_version_and_one_with_no_version() {
        let dir = TempDir::new("check-version");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let write = |text: &str| std::fs::write(auth.join("owner.json"), text).unwrap();
        write(r#"{"v":1,"created_ms":1,"password_changed_ms":1,"password":"x"}"#);
        assert_eq!(check_in(&dir, &[]), Report::default());
        write(r#"{"v":2,"password":"x"}"#);
        let r = check_in(&dir, &[]);
        assert!(
            r.violations.len() == 1 && r.violations[0].contains("newer OAIY"),
            "{r:?}"
        );
        write(r#"{"password":"x"}"#);
        let r = check_in(&dir, &[]);
        assert!(
            r.violations.len() == 1 && r.violations[0].contains("no version"),
            "{r:?}"
        );
    }

    #[test]
    fn check_only_warns_of_a_throttle_file_it_cannot_read_because_the_server_starts_without_it() {
        let dir = TempDir::new("check-throttle");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        // A folder where the file belongs: a read error that is not "not there".
        std::fs::create_dir(auth.join("throttle.json")).unwrap();
        let r = check_in(&dir, &[]);
        assert!(r.violations.is_empty(), "{r:?}");
        assert!(
            r.warnings.iter().any(|w| w.contains("throttle.json")),
            "{r:?}"
        );
        // The same for `owner.json` is a refusal: the server would not start.
        std::fs::remove_dir(auth.join("throttle.json")).unwrap();
        std::fs::create_dir(auth.join("owner.json")).unwrap();
        let r = check_in(&dir, &[]);
        assert!(
            r.violations.iter().any(|v| v.contains("owner.json")),
            "{r:?}"
        );
    }

    #[test]
    fn check_reads_the_owner_file_as_the_server_does_and_refuses_what_the_server_would() {
        let dir = TempDir::new("check-owner-shape");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let write = |text: &str| std::fs::write(auth.join("owner.json"), text).unwrap();
        // What the server needs of it: a version it reads, and a password (a hash it cannot use is a wrong password, not
        // a refusal to start).
        write(r#"{"v":1,"created_ms":1,"password_changed_ms":1,"password":"x"}"#);
        assert_eq!(check_in(&dir, &[]), Report::default());
        for (text, word) in [
            (
                r#"{"v":1,"created_ms":1,"password_changed_ms":1}"#,
                "password",
            ),
            (
                r#"{"v":1,"created_ms":1,"password_changed_ms":1,"password":7}"#,
                "expected a string",
            ),
            (r#"[]"#, "owner.json"),
        ] {
            write(text);
            let r = check_in(&dir, &[]);
            assert_eq!(r.violations.len(), 1, "{text}: {r:?}");
            assert!(
                r.violations[0].contains("owner.json") && r.violations[0].contains(word),
                "{text}: {r:?}"
            );
        }
    }

    #[test]
    fn check_takes_a_port_from_1_to_65535_and_nothing_else() {
        let dir = TempDir::new("check-port");
        for good in ["1", "17972", "65535", " 8080 "] {
            let r = check_in(&dir, &[("OAIY_SERVER_PORT", good)]);
            assert!(r.violations.is_empty(), "{good:?}: {r:?}");
        }
        for bad in ["0", "65536", "70000", "-1", "80.5", "http", "0x50"] {
            let r = check_in(&dir, &[("OAIY_SERVER_PORT", bad)]);
            assert_eq!(r.violations.len(), 1, "{bad:?}: {r:?}");
            assert!(r.violations[0].contains("OAIY_SERVER_PORT"), "{bad:?}");
        }
    }

    #[test]
    fn check_says_what_the_mode_rules_say() {
        let dir = TempDir::new("check-mode");
        // legacy is refused by a server with the web login; shadow on a proxied install; an owner with legacy.
        let r = check_in(&dir, &[("OAIY_ACCESS_MODE", "legacy")]);
        assert!(r.violations.iter().any(|v| v.contains("legacy")), "{r:?}");
        let r = check_in(
            &dir,
            &[
                ("OAIY_ACCESS_MODE", "shadow"),
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ],
        );
        assert!(r.violations.iter().any(|v| v.contains("shadow")), "{r:?}");
        assert!(check_in(&dir, &[("OAIY_ACCESS_MODE", "shadow")])
            .violations
            .is_empty());
        assert!(
            check_in(&dir, &[("OAIY_PUBLIC_URL", "https://dash.example.com")])
                .violations
                .is_empty()
        );
    }

    #[test]
    fn check_warns_of_what_the_server_would_ignore_and_does_not_refuse_it() {
        let dir = TempDir::new("check-warn");
        let r = check_in(&dir, &[("OAIY_PUBLIC_URL", "http://not-https.example.com")]);
        assert!(r.violations.is_empty(), "{r:?}");
        let text = r.warnings.join("\n");
        assert!(text.contains("OAIY_PUBLIC_URL"), "{text}");
    }

    #[test]
    fn check_warns_that_a_login_on_a_network_can_write_and_run_flows_until_flow_authority_lands() {
        let dir = TempDir::new("check-flows");
        // On the machine alone: nothing to say.
        let r = check_in(&dir, &[]);
        assert!(!r.warnings.iter().any(|w| w.contains("flows")), "{r:?}");
        // Behind a proxy, and on a network address: the login is reachable from elsewhere.
        for extra in [
            &[("OAIY_PUBLIC_URL", "https://dash.example.com")][..],
            &[("OAIY_SERVER_BIND", "lan"), ("OAIY_SERVER_TOKEN", "x")][..],
        ] {
            let r = check_in(&dir, extra);
            let warning = r
                .warnings
                .iter()
                .find(|w| w.contains("flows"))
                .unwrap_or_else(|| panic!("{extra:?}: {r:?}"));
            assert!(
                warning.contains("write and run") && warning.contains("ACC-05"),
                "{warning}"
            );
            // A warning is not a refusal: the install starts, and the operator has been told.
            assert!(r.violations.is_empty(), "{extra:?}: {r:?}");
        }
    }

    #[test]
    fn check_refuses_a_login_allow_list_with_any_entry_that_is_not_an_address_or_a_network() {
        let dir = TempDir::new("check-allow");
        // A typo does not turn the restriction off: the whole list is refused, and every bad entry is named.
        for list in [
            "203.0.113.0/24x",
            "203.0.113.0/24, nonsense",
            "203.0.113.0/33",
            "203.0.113.256",
            "2001:db8::/129",
            "203.0.113.0/24;198.51.100.0/24",
            ",",
        ] {
            let r = check_in(&dir, &[("OAIY_LOGIN_ALLOW", list)]);
            assert_eq!(r.violations.len(), 1, "{list:?}: {r:?}");
            assert!(r.violations[0].contains("OAIY_LOGIN_ALLOW"), "{list:?}");
        }
        let r = check_in(
            &dir,
            &[("OAIY_LOGIN_ALLOW", "203.0.113.0/24, nonsense, 9.9.9.9/40")],
        );
        let text = r.violations.join("\n");
        assert!(
            text.contains("nonsense") && text.contains("9.9.9.9/40"),
            "{text}"
        );
        // What is right passes: v4 and v6, single addresses and networks, spaces and a trailing comma.
        for list in [
            "203.0.113.7",
            "203.0.113.0/24",
            "2001:db8::/32, 203.0.113.0/24,",
            " 198.51.100.1 ,203.0.113.0/24 ",
        ] {
            let r = check_in(&dir, &[("OAIY_LOGIN_ALLOW", list)]);
            assert!(r.violations.is_empty(), "{list:?}: {r:?}");
        }
        // Not set, or empty: no restriction, as before.
        assert!(check_in(&dir, &[]).violations.is_empty());
        assert!(check_in(&dir, &[("OAIY_LOGIN_ALLOW", "  ")])
            .violations
            .is_empty());
    }

    #[test]
    fn check_reports_a_folder_that_is_a_file() {
        let dir = TempDir::new("check-file");
        let file = dir.0.join("a-file");
        std::fs::write(&file, "x").unwrap();
        let r = check(&env_of(&[("OAIY_DATA_DIR", file.display().to_string())]));
        assert!(
            r.violations.iter().any(|v| v.contains("is not a folder")),
            "{r:?}"
        );
    }

    /// A terminal that answers with these lines.
    fn typed(lines: &[&str]) -> impl FnMut(&str) -> io::Result<String> {
        let mut queue: Vec<String> = lines.iter().rev().map(|s| s.to_string()).collect();
        move |_: &str| queue.pop().ok_or_else(|| io::Error::other("no more input"))
    }

    fn run_in(
        dir: &TempDir,
        words: &str,
        typed_lines: &[&str],
        euid: Option<u32>,
    ) -> (i32, String, String) {
        let env = env_of(&[("OAIY_DATA_DIR", dir.0.display().to_string())]);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut prompt = typed(typed_lines);
        let code = {
            let mut io = Io {
                env: &env,
                out: &mut out,
                err: &mut err,
                prompt: &mut prompt,
                euid,
            };
            run(&args(words), &mut io)
        };
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    const PASSWORD: &str = "k7Qz!mV3#pW9xLd2 rn8Tb";

    #[test]
    fn every_run_prints_the_data_folder_and_only_init_makes_one() {
        let dir = TempDir::new("cli-folder");
        let missing = TempDir(dir.0.join("nowhere"));
        for words in [
            "auth setup-code",
            "auth status",
            "auth sessions revoke-all",
            "auth token list",
        ] {
            let (code, out, err) = run_in(&missing, words, &[], None);
            assert_eq!(code, EXIT_FAILED, "{words}: {err}");
            assert!(
                out.contains(&format!("data folder: {}", missing.0.display())),
                "{out}"
            );
            assert!(err.contains("Only `auth init` makes"), "{err}");
            assert!(!missing.0.exists(), "{words} made a folder");
        }
        std::mem::forget(missing);
        // A data folder with no auth folder: the same message for the commands, and nothing made.
        let (code, _, err) = run_in(&dir, "auth setup-code", &[], None);
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("does not exist yet"), "{err}");
        assert!(!dir.0.join("auth").exists());
    }

    #[test]
    fn init_makes_the_folder_and_the_owner_from_a_password_file_and_refuses_a_second_without_force()
    {
        let dir = TempDir::new("cli-init");
        let pwfile = dir.0.join("pw.txt");
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(&pwfile, format!("{PASSWORD}\n")).unwrap();
        let (code, out, err) = run_in(
            &dir,
            &format!(
                "auth init --new-folder --password-file {}",
                pwfile.display()
            ),
            &[],
            None,
        );
        assert_eq!(code, EXIT_OK, "{out} / {err}");
        assert!(out.contains("owner login is made"), "{out}");
        let owner = std::fs::read_to_string(dir.0.join("auth").join("owner.json")).unwrap();
        assert!(owner.contains("$argon2id$") && !owner.contains(PASSWORD));
        // The output never says the password.
        assert!(!out.contains(PASSWORD) && !err.contains(PASSWORD));
        // A second init is refused, and `--force` makes a new password.
        let (code, _, err) = run_in(
            &dir,
            &format!(
                "auth init --new-folder --password-file {}",
                pwfile.display()
            ),
            &[],
            None,
        );
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("already exists"), "{err}");
        let (code, out, _) = run_in(
            &dir,
            &format!("auth init --force --password-file {}", pwfile.display()),
            &[],
            None,
        );
        assert_eq!(code, EXIT_OK, "{out}");
        assert!(out.contains("password changed"), "{out}");
        // The command is in the audit log by name.
        let audit = std::fs::read_to_string(dir.0.join("auth").join("audit.jsonl")).unwrap();
        assert!(
            audit.contains("console.command") && audit.contains("\"init\""),
            "{audit}"
        );
        assert!(!audit.contains(PASSWORD));
    }

    #[test]
    fn init_makes_no_folder_that_a_server_has_not_made_unless_it_is_told_to() {
        // A console that looks at the wrong folder (its environment is not the service's) must not make an owner
        // there and call it a success: with no auth folder, `init` refuses and says where the setting is.
        for (tag, make_data) in [("cli-init-nodata", false), ("cli-init-nodir", true)] {
            let dir = TempDir::new(tag);
            let pwfile =
                std::env::temp_dir().join(format!("oaiy-init-pw-{tag}-{}", std::process::id()));
            std::fs::write(&pwfile, format!("{PASSWORD}\n")).unwrap();
            if make_data {
                std::fs::create_dir_all(&dir.0).unwrap();
            } else {
                let _ = std::fs::remove_dir_all(&dir.0);
            }
            let (code, out, err) = run_in(
                &dir,
                &format!("auth init --password-file {}", pwfile.display()),
                &[],
                None,
            );
            assert_eq!(code, EXIT_FAILED, "{tag}: {out} / {err}");
            assert!(
                err.contains("--new-folder") && err.contains("OAIY_DATA_DIR"),
                "{tag}: {err}"
            );
            assert!(!dir.0.join("auth").exists(), "{tag}: a folder was made");
            // Told to, it makes the folder and the owner.
            let (code, out, err) = run_in(
                &dir,
                &format!(
                    "auth init --new-folder --password-file {}",
                    pwfile.display()
                ),
                &[],
                None,
            );
            assert_eq!(code, EXIT_OK, "{tag}: {out} / {err}");
            assert!(dir.0.join("auth").join("owner.json").exists(), "{tag}");
            // A folder that a server has made needs no flag: `--force` is the only thing a second one needs.
            let (code, _, err) = run_in(
                &dir,
                &format!("auth init --password-file {}", pwfile.display()),
                &[],
                None,
            );
            assert_eq!(code, EXIT_FAILED, "{tag}: {err}");
            assert!(err.contains("already exists"), "{tag}: {err}");
            let _ = std::fs::remove_file(&pwfile);
        }
    }

    #[test]
    fn a_weak_password_from_a_file_or_the_terminal_is_refused_with_its_reasons() {
        let dir = TempDir::new("cli-weak");
        std::fs::create_dir_all(&dir.0).unwrap();
        let pwfile = dir.0.join("pw.txt");
        std::fs::write(&pwfile, "short\n").unwrap();
        let (code, _, err) = run_in(
            &dir,
            &format!(
                "auth init --new-folder --password-file {}",
                pwfile.display()
            ),
            &[],
            None,
        );
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("too_short"), "{err}");
        assert!(!dir.0.join("auth").join("owner.json").exists());
        // The terminal: two that do not match, then two weak, then a good pair.
        let (code, out, _) = run_in(
            &dir,
            "auth init --new-folder",
            &[
                "one one one one one",
                "two",
                "short",
                "short",
                PASSWORD,
                PASSWORD,
            ],
            None,
        );
        assert_eq!(code, EXIT_OK, "{out}");
        assert!(
            out.contains("do not match") && out.contains("too_short"),
            "{out}"
        );
        // Three failures in a row: no password is set.
        let dir2 = TempDir::new("cli-weak2");
        std::fs::create_dir_all(&dir2.0).unwrap();
        let (code, _, err) = run_in(
            &dir2,
            "auth init --new-folder",
            &["a", "b", "a", "b", "a", "b"],
            None,
        );
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("no password was set"), "{err}");
    }

    #[test]
    fn a_generated_password_is_printed_once_and_is_the_password() {
        let dir = TempDir::new("cli-generate");
        std::fs::create_dir_all(&dir.0).unwrap();
        let (code, out, _) = run_in(&dir, "auth init --new-folder --generate", &[], None);
        assert_eq!(code, EXIT_OK, "{out}");
        let phrase = out
            .lines()
            .find(|l| l.starts_with("  "))
            .map(|l| l.trim().to_string())
            .expect("the passphrase is printed");
        assert_eq!(phrase.split('-').count(), 6, "{phrase}");
        // It opens the owner file: the stored hash verifies it.
        let owner: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.0.join("auth").join("owner.json")).unwrap(),
        )
        .unwrap();
        let stored = owner["password"].as_str().unwrap();
        assert!(matches!(
            Argon2Engine::production().verify(policy::normalise(&phrase).as_bytes(), stored),
            super::super::password::Verdict::Match { .. }
        ));
    }

    #[test]
    fn setup_code_on_a_stopped_server_writes_the_file_and_prints_the_code_only_on_the_console() {
        let dir = TempDir::new("cli-setup-code");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let (code, out, err) = run_in(&dir, "auth setup-code", &[], None);
        assert_eq!(code, EXIT_OK, "{err}");
        let shown = out
            .lines()
            .find_map(|l| l.strip_prefix("Setup code: "))
            .expect("a code")
            .to_string();
        assert_eq!(shown.len(), 14);
        let file = std::fs::read_to_string(dir.0.join("auth").join("setup-code.json")).unwrap();
        assert!(!file.contains(&shown) && !file.contains(&shown.replace('-', "")));
        let v: Value = serde_json::from_str(&file).unwrap();
        assert_eq!(
            v["hash"].as_str().unwrap(),
            super::super::setup::hash(&super::super::setup::normalise(&shown).unwrap())
        );
        assert_eq!(v["wrong_left"], 100);
        // The audit line names the command and holds no code.
        let audit = std::fs::read_to_string(dir.0.join("auth").join("audit.jsonl")).unwrap();
        assert!(audit.contains("setup-code") && !audit.contains(&shown));
        // With an owner, a setup code is refused.
        let pw = dir.0.join("pw.txt");
        std::fs::write(&pw, PASSWORD).unwrap();
        assert_eq!(
            run_in(
                &dir,
                &format!("auth init --new-folder --password-file {}", pw.display()),
                &[],
                None
            )
            .0,
            EXIT_OK
        );
        let (code, _, err) = run_in(&dir, "auth setup-code", &[], None);
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("already exists"), "{err}");
    }

    #[test]
    fn reset_password_on_a_stopped_server_revokes_every_session_and_device() {
        let dir = TempDir::new("cli-reset");
        std::fs::create_dir_all(&dir.0).unwrap();
        let pw = dir.0.join("pw.txt");
        std::fs::write(&pw, PASSWORD).unwrap();
        assert_eq!(
            run_in(
                &dir,
                &format!("auth init --new-folder --password-file {}", pw.display()),
                &[],
                None
            )
            .0,
            EXIT_OK
        );
        // A session and a device, as a server that ran and stopped would have left them.
        let auth = dir.0.join("auth");
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        let session = {
            let store = AuthStore::open(
                &auth,
                Host::Server,
                clock.clone(),
                Arc::new(SecureWriter),
                None,
            )
            .unwrap();
            let s =
                super::super::session::mint_session(&store, false, "203.0.113.9", "test").unwrap();
            store.flush().unwrap();
            s
        };
        let mut owner: OwnerDoc = OwnerDoc::from_value(
            &serde_json::from_str(&std::fs::read_to_string(auth.join("owner.json")).unwrap())
                .unwrap(),
        )
        .unwrap();
        let mut fill = |b: &mut [u8]| token::os_random(b);
        owner.devices.push(
            super::super::device::make(1, "203.0.113.9", &mut fill)
                .unwrap()
                .device,
        );
        owner::write(&SecureWriter, &auth, &owner).unwrap();
        std::fs::write(&pw, "Hv4$wN6@cJ1&zX8 qs5Fe").unwrap();
        let (code, out, err) = run_in(
            &dir,
            &format!("auth reset-password --password-file {}", pw.display()),
            &[],
            None,
        );
        assert_eq!(code, EXIT_OK, "{out} / {err}");
        assert!(
            out.contains("1 session(s) and every device revoked"),
            "{out}"
        );
        let store =
            AuthStore::open(&auth, Host::Server, clock, Arc::new(SecureWriter), None).unwrap();
        let err = store.authenticate(&session.token, None).unwrap_err();
        assert_eq!(err.reason(), Some("password_changed"));
        let after: Value =
            serde_json::from_str(&std::fs::read_to_string(auth.join("owner.json")).unwrap())
                .unwrap();
        assert_eq!(after["devices"].as_array().unwrap().len(), 0);
        assert!(after["password"]
            .as_str()
            .unwrap()
            .starts_with("$argon2id$"));
    }

    #[test]
    fn a_token_made_on_the_console_is_listed_and_revoked_and_holds_the_rules_of_a_token() {
        let dir = TempDir::new("cli-token");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let (code, out, err) = run_in(
            &dir,
            "auth token create --preset cli --ttl 30d --label ci",
            &[],
            None,
        );
        assert_eq!(code, EXIT_OK, "{err}");
        let id = out
            .lines()
            .find_map(|l| l.strip_prefix("id: "))
            .unwrap()
            .to_string();
        let tok = out
            .lines()
            .find_map(|l| l.strip_prefix("token (shown once): "))
            .unwrap()
            .to_string();
        assert!(tok.starts_with("oaiypat_") && tok.len() == 68);
        let (_, listed, _) = run_in(&dir, "auth token list", &[], None);
        assert!(
            listed.contains(&id) && listed.contains("ci") && !listed.contains(&tok),
            "{listed}"
        );
        // It is a real credential of the store.
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        {
            let store = AuthStore::open(
                &dir.0.join("auth"),
                Host::Server,
                clock,
                Arc::new(SecureWriter),
                None,
            )
            .unwrap();
            assert!(store.authenticate(&tok, None).is_ok());
        }
        // A dangerous preset is a 24-hour token; owner and unknown presets, and a lifetime over the maximum, are refused.
        let (code, out, _) = run_in(&dir, "auth token create --preset cli-admin", &[], None);
        assert_eq!(code, EXIT_OK, "{out}");
        let (code, out, _) = run_in(&dir, "auth token create --preset cli", &[], None);
        assert_eq!(code, EXIT_OK, "{out}");
        // The lifetimes, as the list rounds them down: 30 days as asked, one day for a dangerous preset (so none whole
        // days are left) and ninety for the rest.
        let (_, listed, _) = run_in(&dir, "auth token list", &[], None);
        for left in [
            "expires in 29 days",
            "expires in 0 days",
            "expires in 89 days",
        ] {
            assert_eq!(listed.matches(left).count(), 1, "{left}:\n{listed}");
        }
        for bad in [
            "auth token create --preset owner",
            "auth token create --preset nonsense",
            "auth token create --preset cli --scope nonsense.scope",
            "auth token create --preset cli --ttl 999d",
            "auth token create --preset cli-admin --ttl 3d",
            "auth token create --preset cli --origin not-an-origin",
        ] {
            let (code, _, _) = run_in(&dir, bad, &[], None);
            assert_eq!(code, EXIT_FAILED, "{bad}");
        }
        let (code, out, _) = run_in(&dir, &format!("auth token revoke {id}"), &[], None);
        assert_eq!(code, EXIT_OK, "{out}");
        assert_eq!(
            run_in(&dir, &format!("auth token revoke {id}"), &[], None).0,
            EXIT_FAILED
        );
    }

    #[test]
    fn the_console_run_as_root_against_another_users_folder_refuses_and_says_how_to_run_it() {
        let data = Path::new("/var/lib/oaiy");
        let service = Some((998, "oaiy".to_string()));
        let why = root_refusal(Some(0), &service, data).expect("root is refused");
        assert!(why.contains("sudo -u oaiy oaiy-server"), "{why}");
        assert!(
            why.contains("/var/lib/oaiy") || why.contains("var"),
            "{why}"
        );
        // A user known only by number is named the way sudo takes one.
        let numbered = Some((1234, "#1234".to_string()));
        assert!(root_refusal(Some(0), &numbered, data)
            .unwrap()
            .contains("sudo -u #1234"));
        // Not refused: root's own folder, the folder's owner, another user, no user ids at all (Windows), no owner known.
        assert_eq!(root_refusal(Some(0), &Some((0, "root".into())), data), None);
        assert_eq!(root_refusal(Some(998), &service, data), None);
        assert_eq!(root_refusal(Some(1000), &service, data), None);
        assert_eq!(root_refusal(None, &service, data), None);
        assert_eq!(root_refusal(Some(0), &None, data), None);
        // And it stops the command before anything is opened (a folder that exists here has an owner only on unix).
        let dir = TempDir::new("cli-root");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let (code, out, err) = run_in(&dir, "auth status", &[], Some(0));
        #[cfg(unix)]
        if folder_owner(&dir.0).is_some_and(|(uid, _)| uid != 0) {
            assert_eq!(code, EXIT_FAILED);
            assert!(
                err.contains("sudo -u") && out.contains("data folder:"),
                "{err}"
            );
            assert!(!dir.0.join("auth").join(".lock").exists());
            return;
        }
        let _ = (code, out, err);
    }

    #[test]
    fn session_link_is_the_running_servers_and_flows_is_not_in_this_build() {
        let dir = TempDir::new("cli-link");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let (code, _, err) = run_in(&dir, "auth session-link", &[], None);
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("running server"), "{err}");
        let (code, _, err) = run_in(&dir, "flows list", &[], None);
        assert_eq!(code, EXIT_USAGE);
        assert!(err.contains("flow authority"), "{err}");
        let (code, _, err) = run_in(&dir, "auth nonsense", &[], None);
        assert_eq!(code, EXIT_USAGE);
        assert!(err.contains("usage:"), "{err}");
    }

    #[test]
    fn a_running_server_that_cannot_be_reached_is_an_error_naming_the_file_and_changes_nothing() {
        // The lock is held (this process holds it, as a server would), and there are no console files.
        let dir = TempDir::new("cli-unreachable");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        let _held = AuthStore::open(
            &dir.0.join("auth"),
            Host::Server,
            clock,
            Arc::new(SecureWriter),
            None,
        )
        .unwrap();
        let (code, _, err) = run_in(&dir, "auth setup-code", &[], None);
        assert_eq!(code, EXIT_FAILED);
        assert!(
            err.contains("console.json") && err.contains("nothing was done"),
            "{err}"
        );
        assert!(!dir.0.join("auth").join("setup-code.json").exists());
        // Unreadable console files, and a port nobody answers on, are named too.
        std::fs::write(
            dir.0.join("auth").join("console.json"),
            r#"{"port":1,"pid":1,"started_ms":1}"#,
        )
        .unwrap();
        std::fs::write(
            dir.0.join("auth").join("console.token"),
            "oaiycon_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8\n",
        )
        .unwrap();
        let (code, _, err) = run_in(&dir, "auth setup-code", &[], None);
        assert_eq!(code, EXIT_FAILED);
        assert!(err.contains("did not answer"), "{err}");
    }

    #[test]
    fn a_refused_owner_file_is_exit_78_naming_it() {
        let dir = TempDir::new("cli-refused");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        std::fs::write(auth.join("owner.json"), "{ mangled").unwrap();
        let (code, _, err) = run_in(&dir, "auth status", &[], None);
        // 78 is EX_CONFIG of sysexits.h, the code a refusal to start has: the number itself is the rule.
        assert_eq!(code, 78);
        assert_eq!(EXIT_CONFIG, 78);
        assert!(err.contains("owner.json"), "{err}");
    }
}
