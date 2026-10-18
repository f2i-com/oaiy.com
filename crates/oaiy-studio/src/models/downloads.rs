//! Getting models: a catalog of the mainstream models OAIY runs
//! (`catalog.json`), downloaded from Hugging Face (and GitHub releases) with
//! the system's `curl`, each into a folder of its own under the chosen folder,
//! then added as the Models page adds them.
//!
//! One download runs at a time; the others wait their turn. A download can be
//! paused (curl stops, and the partial file stays beside its name as `.part`)
//! and resumed (curl carries on from there), also after the studio restarts;
//! cancelling deletes the partial file. A model whose licence has to be
//! accepted on huggingface.co first (gated) needs a Hugging Face token.
//!
//! A finished part leaves `.oaiy-<id>.json` in its folder, which is how the
//! catalog knows the model is there.
//!
//! Nothing this computer has already is downloaded again: a file Hugging Face
//! publishes a SHA-256 for is looked for among the model files here first
//! ([`library`]), and used where it is (linked into the model's folder); each
//! file downloaded is checked against it. [`Downloads::plan_of`] says, before a
//! download, what is here already and what is left to fetch.
use crate::util::str_or;
use crate::{
    config,
    models::{library, registry, sha256},
    Studio,
};
use oaiy_engine::json::Json;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

const CATALOG: &str = include_str!("catalog.json");

/// The catalog, parsed once.
pub fn catalog() -> &'static Json {
    static PARSED: OnceLock<Json> = OnceLock::new();
    PARSED.get_or_init(|| Json::parse(CATALOG.as_bytes()).expect("the built-in catalog is valid JSON"))
}

fn entries() -> &'static [Json] {
    catalog().get("models").and_then(Json::as_array).unwrap_or(&[])
}

fn entry(id: &str) -> Option<&'static Json> {
    entries().iter().find(|e| str_or(e, "id", "") == id)
}

fn strings(v: Option<&Json>) -> Vec<String> {
    v.and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(str::to_string).collect()
}

/// Does `path` (forward slashes) match `glob`: `*` within a folder, `**` across folders?
pub fn glob(pattern: &str, path: &str) -> bool {
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some(b'*') if p.get(1) == Some(&b'*') => {
                let rest = if p.get(2) == Some(&b'/') { &p[3..] } else { &p[2..] };
                (0..=s.len()).any(|i| go(rest, &s[i..])) || (p.get(2) == Some(&b'/') && go(&p[3..], s))
            }
            Some(b'*') => (0..=s.len()).take_while(|&i| i == 0 || s[i - 1] != b'/').any(|i| go(&p[1..], &s[i..])),
            Some(c) => s.first() == Some(c) && go(&p[1..], &s[1..]),
        }
    }
    go(pattern.as_bytes(), path.as_bytes())
}

/// Where downloads go: `downloads.dir` (relative to the studio's folder), else its `models` folder.
pub fn default_dir(studio: &Studio) -> PathBuf {
    let cfg = studio.config();
    let dir = cfg.get("downloads").map_or("", |d| str_or(d, "dir", "")).trim().to_string();
    config::resolve(&studio.root, if dir.is_empty() { "models" } else { &dir })
}

/// The Hugging Face token: one given with a download (the desktop's, from its Settings), else the studio's own.
fn token_for(cfg: &Json, given: Option<&str>) -> Option<String> {
    given.map(str::trim).filter(|t| !t.is_empty()).map(str::to_string).or_else(|| token(cfg))
}

fn token(cfg: &Json) -> Option<String> {
    let t = cfg.get("downloads").map_or("", |d| str_or(d, "hf_token", "")).trim().to_string();
    if !t.is_empty() {
        return Some(t);
    }
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
}

fn curl_program(cfg: &Json) -> String {
    let c = cfg.get("downloads").map_or("", |d| str_or(d, "curl", "")).trim().to_string();
    if c.is_empty() {
        "curl".into()
    } else {
        c
    }
}

fn command(program: &str) -> Command {
    // (curl with the computer's environment, where OAIY runs from an AppImage: the system's curl failed with its libraries)
    let mut c = oaiy_host_env::command(program);
    c.stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    c
}

/// A file to fetch: from where, to where, and how large it is when known.
#[derive(Clone, Debug)]
struct Planned {
    url: String,
    dest: PathBuf,
    size: Option<u64>,
    /// Sent the Hugging Face token (a file on huggingface.co).
    hub: bool,
    /// Its SHA-256 as Hugging Face publishes it (a large file's), lowercase hex.
    sha256: Option<String>,
}

/// `path` (forward slashes, from a catalog or the hub) under `base`, refusing anything that climbs out.
fn under(base: &Path, path: &str) -> Result<PathBuf, String> {
    let mut out = base.to_path_buf();
    for part in path.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." || part.contains(':') || part.contains('\\') {
            return Err(format!("unsafe path in the catalog: {path}"));
        }
        out.push(part);
    }
    Ok(out)
}

fn percent_encode(path: &str) -> String {
    let mut out = String::new();
    for b in path.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Runs curl to completion, returning its stdout (text).
fn curl_text(cfg: &Json, url: &str, auth: Option<&str>, head: bool) -> Result<String, String> {
    let mut c = command(&curl_program(cfg));
    c.args(["-sS", "-L", "--fail", "--connect-timeout", "30", "--max-time", "120", "-H", "User-Agent: oaiy-studio"]);
    if head {
        c.arg("-I");
    }
    if let Some(t) = auth {
        c.args(["-H", &format!("Authorization: Bearer {t}")]);
    }
    c.arg(url);
    let out = c.output().map_err(|e| format!("could not run curl ({e}): OAIY downloads models with the curl program, which Windows 10 and later, macOS and Linux include"))?;
    if !out.status.success() {
        return Err(curl_error(&String::from_utf8_lossy(&out.stderr), url));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// curl's complaint, in words a person can act on.
fn curl_error(stderr: &str, url: &str) -> String {
    let e = stderr.trim();
    let repo = url.strip_prefix("https://huggingface.co/").map(|r| r.trim_start_matches("api/models/")).and_then(|r| {
        let mut p = r.split('/');
        Some(format!("{}/{}", p.next()?, p.next()?))
    });
    if e.contains("401") || e.contains("403") {
        return match repo {
            Some(r) => format!("{r} is gated: sign in on huggingface.co, accept its licence at https://huggingface.co/{r}, then add a Hugging Face access token (read) in OAIY's Settings or on Get models"),
            None => format!("{url} refused the download ({e})"),
        };
    }
    if e.contains("404") {
        return format!("{url} was not found (the catalog may be older than the model's page)");
    }
    if e.contains("(23)") || e.to_ascii_lowercase().contains("no space") {
        return "writing the file failed: is the disk full?".into();
    }
    if e.contains("(6)") || e.contains("(7)") || e.contains("(28)") || e.contains("(35)") {
        return format!("could not reach {} ({e}); check the internet connection", url.split('/').take(3).collect::<Vec<_>>().join("/"));
    }
    if e.is_empty() {
        "curl failed".into()
    } else {
        e.to_string()
    }
}

/// The files a catalog entry's parts name, with their sizes.
fn plan(cfg: &Json, entry: &Json, dir: &Path, token: Option<&str>) -> Result<Vec<Planned>, String> {
    let mut out = Vec::new();
    for part in entry.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
        let into = under(dir, str_or(part, "into", ""))?;
        let repo = str_or(part, "repo", "");
        if !repo.is_empty() {
            let rev = str_or(part, "revision", "main");
            let list = curl_text(cfg, &format!("https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"), token, false)?;
            let tree = Json::parse(list.as_bytes()).map_err(|e| format!("{repo}: the file list did not read ({e})"))?;
            let (include, exclude) = (strings(part.get("include")), strings(part.get("exclude")));
            for f in tree.as_array().unwrap_or(&[]) {
                if str_or(f, "type", "") != "file" {
                    continue;
                }
                let path = str_or(f, "path", "");
                if !include.iter().any(|g| glob(g, path)) || exclude.iter().any(|g| glob(g, path) || path.rsplit('/').next().is_some_and(|n| glob(g, n))) {
                    continue;
                }
                let size = f.get("lfs").and_then(|l| l.get("size")).and_then(Json::as_i64).or_else(|| f.get("size").and_then(Json::as_i64)).map(|s| s as u64);
                // (a large file's SHA-256, as Hugging Face keeps it: what it is checked against, and found here by)
                let sha256 = f.get("lfs").and_then(|l| l.get("oid")).and_then(Json::as_str).map(|o| o.trim_start_matches("sha256:").to_ascii_lowercase()).filter(|o| o.len() == 64 && o.bytes().all(|b| b.is_ascii_hexdigit()));
                out.push(Planned { url: format!("https://huggingface.co/{repo}/resolve/{rev}/{}", percent_encode(path)), dest: under(&into, path)?, size, hub: true, sha256 });
            }
            // Every file the catalog names exactly must be there.
            for name in include.iter().filter(|g| !g.contains('*')) {
                if !out.iter().any(|p| p.dest == under(&into, name).unwrap_or_default()) && !["LICENSE", "LICENSE.md", "README.md", "NOTICE"].contains(&name.as_str()) {
                    return Err(format!("{repo} has no {name} any more (the catalog may be older than the model's page)"));
                }
            }
        } else {
            let url = str_or(part, "url", "");
            let file = str_or(part, "file", "");
            if url.is_empty() || file.is_empty() {
                return Err(format!("{}: a part names neither a repo nor a url and file", str_or(entry, "id", "")));
            }
            let size = part.get("size").and_then(Json::as_i64).map(|s| s as u64).or_else(|| head_size(cfg, url));
            out.push(Planned { url: url.to_string(), dest: under(&into, file)?, size, hub: false, sha256: part.get("sha256").and_then(Json::as_str).map(str::to_ascii_lowercase) });
        }
    }
    Ok(out)
}

fn head_size(cfg: &Json, url: &str) -> Option<u64> {
    let headers = curl_text(cfg, url, None, true).ok()?;
    headers.lines().filter_map(|l| l.split_once(':')).filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length")).filter_map(|(_, v)| v.trim().parse().ok()).last()
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

fn size_of(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// Free space on the drive holding `dir` (or its nearest existing parent), when it can be told.
pub fn free_space(dir: &Path) -> Option<u64> {
    let mut at = dir.to_path_buf();
    while !at.exists() {
        at = at.parent()?.to_path_buf();
    }
    #[cfg(windows)]
    {
        let s = at.to_string_lossy();
        let drive = s.get(..2).filter(|d| d.ends_with(':'))?;
        let out = command("powershell").args(["-NoProfile", "-NonInteractive", "-Command", &format!("(New-Object System.IO.DriveInfo('{drive}\\')).AvailableFreeSpace")]).output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(windows))]
    {
        let out = command("df").args(["-Pk"]).arg(&at).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let kb: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
        Some(kb * 1024)
    }
}

fn marker(dir: &Path, part: &Json, id: &str) -> Option<PathBuf> {
    under(dir, str_or(part, "into", "")).ok().map(|p| p.join(format!(".oaiy-{id}.json")))
}

/// Is the entry in `dir`: every part's marker written, or (downloaded some other
/// way) everything it adds there, with no partial file beside it?
fn installed(entry: &Json, dir: &Path) -> bool {
    let id = str_or(entry, "id", "");
    let parts = entry.get("parts").and_then(Json::as_array).unwrap_or(&[]);
    if !parts.is_empty() && parts.iter().all(|p| marker(dir, p, id).is_some_and(|m| m.is_file())) {
        return true;
    }
    let adds = strings(entry.get("add"));
    !adds.is_empty()
        && adds.iter().all(|a| under(dir, a).is_ok_and(|p| p.exists()))
        && !parts.iter().filter_map(|p| under(dir, str_or(p, "into", "")).ok()).any(|folder| has_part_file(&folder, 3))
}

/// Is part of the entry downloaded already (a partial file, or some parts done)?
fn partial(entry: &Json, dir: &Path) -> bool {
    let id = str_or(entry, "id", "");
    let parts = entry.get("parts").and_then(Json::as_array).unwrap_or(&[]);
    parts.iter().any(|p| marker(dir, p, id).is_some_and(|m| m.is_file()))
        || parts.iter().filter_map(|p| under(dir, str_or(p, "into", "")).ok()).any(|folder| has_part_file(&folder, 3))
}

fn has_part_file(dir: &Path, depth: usize) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    rd.flatten().any(|e| {
        let p = e.path();
        if p.is_dir() {
            depth > 0 && has_part_file(&p, depth - 1)
        } else {
            p.extension().is_some_and(|x| x == "part")
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Want {
    Run,
    Pause,
    Cancel,
}

#[derive(Clone, Debug)]
struct Download {
    id: String,
    dir: PathBuf,
    /// queued | downloading | paused | adding | done | failed | cancelled
    status: &'static str,
    done: u64,
    total: u64,
    file: String,
    files_done: usize,
    files_total: usize,
    /// Bytes a second, smoothed.
    speed: f64,
    error: Option<String>,
    added: Vec<String>,
    /// The bytes and files found on this computer already, used where they are rather than downloaded.
    reused: u64,
    reused_files: usize,
    /// The bytes fetched from the network's other computers (OAIY Network), and from which.
    lan: u64,
    lan_from: Vec<String>,
    /// The Hugging Face token given with it (never said back).
    token: Option<String>,
}

impl Download {
    fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::str(&self.id)),
            ("dir", Json::str(self.dir.to_string_lossy())),
            ("status", Json::str(self.status)),
            ("done", Json::Int(self.done as i64)),
            ("total", Json::Int(self.total as i64)),
            ("file", Json::str(&self.file)),
            ("files_done", Json::Int(self.files_done as i64)),
            ("files_total", Json::Int(self.files_total as i64)),
            ("speed", Json::Num((self.speed * 10.).round() / 10.)),
            ("error", self.error.as_ref().map_or(Json::Null, Json::str)),
            ("added", Json::Arr(self.added.iter().map(Json::str).collect())),
            ("reused", Json::Int(self.reused as i64)),
            ("reused_files", Json::Int(self.reused_files as i64)),
            ("lan", Json::Int(self.lan as i64)),
            ("lan_from", Json::Arr(self.lan_from.iter().map(Json::str).collect())),
        ])
    }
}

/// Model files on the other OAIY computers this one uses (OAIY Network), as the desktop that runs the studio reaches
/// them (`set_peer_files`): a file wanted is fetched from one of them, over the network, before Hugging Face is
/// asked; it is checked against its SHA-256 the same way.
pub trait PeerFiles: Send + Sync {
    /// For each of `files` (SHA-256, bytes), the name of a paired computer that has it (said from what it knows
    /// already: checked when it is fetched), or `None`.
    fn have(&self, files: &[(String, u64)]) -> Vec<Option<String>>;
    /// Fetch `sha256` (`size` bytes) into `part`, carrying on from what `part` holds, from a paired computer that has
    /// it: `Ok(Some(its name))` once `part` is whole, `Ok(None)` where none has it (or `stop` said to stop).
    /// `progress` is told the bytes in `part` as it grows; `stop` is asked between pieces.
    fn fetch(&self, sha256: &str, size: u64, part: &Path, progress: &dyn Fn(u64), stop: &dyn Fn() -> bool) -> Result<Option<String>, String>;
}

static PEER_FILES: RwLock<Option<Arc<dyn PeerFiles>>> = RwLock::new(None);

/// Where model files on the network's other computers are fetched from (`None`: from Hugging Face alone).
pub fn set_peer_files(peers: Option<Arc<dyn PeerFiles>>) {
    *PEER_FILES.write().unwrap_or_else(|p| p.into_inner()) = peers;
}

fn peer_files() -> Option<Arc<dyn PeerFiles>> {
    PEER_FILES.read().unwrap_or_else(|p| p.into_inner()).clone()
}

#[derive(Default)]
pub struct Downloads {
    list: Mutex<Vec<Download>>,
    want: Mutex<HashMap<String, Want>>,
    /// The curl fetching a file now.
    child: Mutex<Option<Child>>,
    wake: Condvar,
    worker: Mutex<bool>,
    free: Mutex<Option<(PathBuf, Instant, Option<u64>)>>,
}

impl Downloads {
    pub fn new() -> Self {
        Self::default()
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Download)) {
        let mut list = self.list.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(d) = list.iter_mut().find(|d| d.id == id) {
            f(d);
        }
    }

    fn wanted(&self, id: &str) -> Want {
        *self.want.lock().unwrap_or_else(|p| p.into_inner()).get(id).unwrap_or(&Want::Run)
    }

    fn stop_child(&self) {
        if let Some(c) = self.child.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            let _ = c.kill();
        }
    }

    fn free_cached(&self, dir: &Path) -> Option<u64> {
        let mut f = self.free.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((d, at, v)) = f.as_ref() {
            if d == dir && at.elapsed() < Duration::from_secs(15) {
                return *v;
            }
        }
        let v = free_space(dir);
        *f = Some((dir.to_path_buf(), Instant::now(), v));
        v
    }

    /// Queue `id` (and what it needs that is not there yet) into `dir`; `token`: a Hugging Face token for it (the
    /// desktop's), else the studio's own.
    pub fn start(&self, studio: &Arc<Studio>, id: &str, dir: Option<PathBuf>, token: Option<String>) -> Result<Json, String> {
        let e = entry(id).ok_or_else(|| format!("no model {id} in the catalog"))?;
        let dir = dir.filter(|d| !d.as_os_str().is_empty()).unwrap_or_else(|| default_dir(studio));
        if !dir.is_absolute() {
            return Err("the folder to save to must be a full path".into());
        }
        let mut ids: Vec<String> = strings(e.get("needs")).into_iter().filter(|n| entry(n).is_some_and(|x| !installed(x, &dir))).collect();
        ids.push(id.to_string());
        {
            let mut list = self.list.lock().unwrap_or_else(|p| p.into_inner());
            let mut want = self.want.lock().unwrap_or_else(|p| p.into_inner());
            for id in &ids {
                want.insert(id.clone(), Want::Run);
                match list.iter_mut().find(|d| &d.id == id) {
                    Some(d) if matches!(d.status, "queued" | "downloading" | "adding") => {}
                    Some(d) => {
                        d.status = "queued";
                        d.error = None;
                        d.dir = dir.clone();
                        d.token = token.clone().or(d.token.take());
                    }
                    None => list.push(Download {
                        id: id.clone(),
                        dir: dir.clone(),
                        status: "queued",
                        done: 0,
                        total: 0,
                        file: String::new(),
                        files_done: 0,
                        files_total: 0,
                        speed: 0.,
                        error: None,
                        added: Vec::new(),
                        reused: 0,
                        reused_files: 0,
                        lan: 0,
                        lan_from: Vec::new(),
                        token: token.clone(),
                    }),
                }
            }
        }
        self.ensure_worker(studio);
        self.wake.notify_all();
        Ok(self.state(studio))
    }

    /// Stop `id`, keeping what is downloaded so far.
    pub fn pause(&self, studio: &Studio, id: &str) -> Json {
        self.want.lock().unwrap_or_else(|p| p.into_inner()).insert(id.to_string(), Want::Pause);
        let active = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().any(|d| d.id == id && d.status == "downloading");
        if active {
            self.stop_child();
        } else {
            self.update(id, |d| {
                if d.status == "queued" {
                    d.status = "paused";
                }
            });
        }
        self.state(studio)
    }

    /// Stop `id` and delete its partial files.
    pub fn cancel(&self, studio: &Studio, id: &str) -> Json {
        self.want.lock().unwrap_or_else(|p| p.into_inner()).insert(id.to_string(), Want::Cancel);
        let (active, dir) = {
            let list = self.list.lock().unwrap_or_else(|p| p.into_inner());
            let d = list.iter().find(|d| d.id == id);
            (d.is_some_and(|d| d.status == "downloading"), d.map(|d| d.dir.clone()).unwrap_or_else(|| default_dir(studio)))
        };
        if active {
            self.stop_child();
        } else {
            remove_partials(id, &dir);
            self.update(id, |d| d.status = "cancelled");
        }
        self.state(studio)
    }

    /// Add a downloaded model to the model list again.
    pub fn add_again(&self, studio: &Arc<Studio>, id: &str, dir: Option<PathBuf>) -> Result<Json, String> {
        let e = entry(id).ok_or_else(|| format!("no model {id} in the catalog"))?;
        let dir = dir.unwrap_or_else(|| default_dir(studio));
        if !installed(e, &dir) {
            return Err(format!("{} is not downloaded in {}", str_or(e, "name", id), dir.display()));
        }
        let added = register(studio, e, &dir)?;
        studio.log.push(format!("added {} again from {}", str_or(e, "name", id), dir.display()));
        Ok(Json::obj([("added", Json::Arr(added.iter().map(Json::str).collect()))]))
    }

    /// The catalog with each model's state, the downloads, and where they go.
    pub fn state(&self, studio: &Studio) -> Json {
        let dir = default_dir(studio);
        let cfg = studio.config();
        let list = self.list.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let models: Vec<Json> = entries()
            .iter()
            .map(|e| {
                let id = str_or(e, "id", "");
                let mut o = match e {
                    Json::Obj(f) => f.clone(),
                    _ => Vec::new(),
                };
                o.retain(|(k, _)| k != "parts" && k != "add");
                let d = list.iter().find(|d| d.id == id);
                let where_ = d.map(|d| d.dir.clone()).unwrap_or_else(|| dir.clone());
                let have = installed(e, &where_);
                o.push(("installed".into(), Json::Bool(have)));
                // (a model all of whose parts are done is installed, not partly downloaded: each part's marker is
                // what says a part is done)
                o.push(("partial".into(), Json::Bool(d.is_none() && !have && partial(e, &where_))));
                o.push(("download".into(), d.map_or(Json::Null, Download::to_json)));
                Json::Obj(o)
            })
            .collect();
        Json::obj([
            ("dir", Json::str(dir.to_string_lossy())),
            ("free", self.free_cached(&dir).map_or(Json::Null, |v| Json::Int(v as i64))),
            ("token", Json::Bool(token(&cfg).is_some())),
            ("groups", catalog().get("groups").cloned().unwrap_or(Json::Arr(Vec::new()))),
            ("models", Json::Arr(models)),
        ])
    }

    /// What getting `id` takes, before it is got: each of its files here already (where it goes), on this computer
    /// elsewhere (another model's, another drive's: used where it is; `checked` once its SHA-256 has been read, else
    /// read when it is got), on a computer on the network (OAIY Network: fetched from it), or to download; and the
    /// bytes of each, the folder's free space, and the models it needs that are not here yet. Asks Hugging Face for
    /// the files' list (`token`: as for [`Downloads::start`]).
    pub fn plan_of(&self, studio: &Studio, id: &str, token: Option<String>) -> Result<Json, String> {
        let e = entry(id).ok_or_else(|| format!("no model {id} in the catalog"))?;
        let dir = default_dir(studio);
        let cfg = studio.config();
        let token = token_for(&cfg, token.as_deref());
        let files = plan(&cfg, e, &dir, token.as_deref())?;
        let here = library::scan(&library::roots(&studio.root, &cfg, &dir));
        // Each file: in place, on this computer elsewhere, on a computer on the network, or to download.
        let mut said: Vec<(&'static str, Option<String>, Option<bool>)> = files
            .iter()
            .map(|f| {
                if f.size.is_some_and(|s| size_of(&f.dest) == s) {
                    ("here", None, None)
                } else if let Some((from, checked)) = f.size.filter(|s| *s >= library::MIN_BYTES).and_then(|s| library::likely(&studio.root, &here, s, f.sha256.as_deref())) {
                    ("elsewhere", Some(from.to_string_lossy().into_owned()), Some(checked))
                } else {
                    ("download", None, None)
                }
            })
            .collect();
        if let Some(peers) = peer_files() {
            let asked: Vec<(usize, (String, u64))> = files
                .iter()
                .enumerate()
                .filter(|(i, f)| said[*i].0 == "download" && f.size.is_some_and(|s| s >= library::MIN_BYTES))
                .filter_map(|(i, f)| Some((i, (f.sha256.clone()?, f.size?))))
                .collect();
            if !asked.is_empty() {
                let answers = peers.have(&asked.iter().map(|(_, a)| a.clone()).collect::<Vec<_>>());
                for ((i, _), who) in asked.iter().zip(answers) {
                    if let Some(name) = who {
                        said[*i] = ("network", Some(name), None);
                    }
                }
            }
        }
        let (mut at_home, mut elsewhere, mut on_network, mut to_fetch) = (0u64, 0u64, 0u64, 0u64);
        let listed: Vec<Json> = files
            .iter()
            .zip(&said)
            .map(|(f, (status, from, checked))| {
                let size = f.size.unwrap_or(0);
                let shown = f.dest.strip_prefix(&dir).unwrap_or(&f.dest).to_string_lossy().replace('\\', "/");
                match *status {
                    "here" => at_home += size,
                    "elsewhere" => elsewhere += size,
                    "network" => on_network += size.saturating_sub(size_of(&part_path(&f.dest))),
                    _ => to_fetch += size.saturating_sub(size_of(&part_path(&f.dest))),
                }
                let mut o = vec![("path", Json::str(&shown)), ("size", f.size.map_or(Json::Null, |s| Json::Int(s as i64))), ("status", Json::str(*status))];
                if let Some(from) = from {
                    o.push(("from", Json::str(from)));
                }
                if let Some(c) = checked {
                    o.push(("checked", Json::Bool(*c)));
                }
                if let Some(sha) = &f.sha256 {
                    o.push(("sha256", Json::str(sha)));
                }
                Json::obj(o)
            })
            .collect();
        let needs: Vec<Json> = strings(e.get("needs")).into_iter().filter(|n| entry(n).is_some_and(|x| !installed(x, &dir))).map(Json::str).collect();
        Ok(Json::obj([
            ("id", Json::str(id)),
            ("name", Json::str(str_or(e, "name", id))),
            ("gated", Json::Bool(e.get("gated").and_then(Json::as_bool).unwrap_or(false))),
            ("token", Json::Bool(token.is_some())),
            ("dir", Json::str(dir.to_string_lossy())),
            ("free", self.free_cached(&dir).map_or(Json::Null, |v| Json::Int(v as i64))),
            ("here_bytes", Json::Int(at_home as i64)),
            ("elsewhere_bytes", Json::Int(elsewhere as i64)),
            ("network_bytes", Json::Int(on_network as i64)),
            ("download_bytes", Json::Int(to_fetch as i64)),
            ("needs", Json::Arr(needs)),
            ("files", Json::Arr(listed)),
        ]))
    }


    /// Where this computer keeps each of `files` (SHA-256, bytes) among its model files, for another computer on the
    /// network that wants it: `check` reads a file's SHA-256 where it is not known yet (else a file as large whose
    /// SHA-256 is not known yet is said, `checked` false). `{files: [{sha256, size, path, checked}]}`.
    pub fn find_files(&self, studio: &Studio, files: &[(String, u64)], check: bool) -> Json {
        let cfg = studio.config();
        let here = library::scan(&library::roots(&studio.root, &cfg, &default_dir(studio)));
        let found: Vec<Json> = files
            .iter()
            .map(|(sha, size)| {
                let at = if *size < library::MIN_BYTES {
                    None
                } else if check {
                    library::find(&studio.root, &here, *size, sha, |_| false).map(|p| (p, true))
                } else {
                    library::likely(&studio.root, &here, *size, Some(sha))
                };
                Json::obj([
                    ("sha256", Json::str(sha)),
                    ("size", Json::Int(*size as i64)),
                    ("path", at.as_ref().map_or(Json::Null, |(p, _)| Json::str(p.to_string_lossy()))),
                    ("checked", Json::Bool(at.is_some_and(|(_, c)| c))),
                ])
            })
            .collect();
        Json::obj([("files", Json::Arr(found))])
    }
}

impl Downloads {
    fn ensure_worker(&self, studio: &Arc<Studio>) {
        let mut started = self.worker.lock().unwrap_or_else(|p| p.into_inner());
        if *started {
            return;
        }
        *started = true;
        let studio = studio.clone();
        std::thread::spawn(move || loop {
            let next = {
                let mut list = studio.downloads.list.lock().unwrap_or_else(|p| p.into_inner());
                loop {
                    if let Some(d) = list.iter().find(|d| d.status == "queued") {
                        break d.id.clone();
                    }
                    list = studio.downloads.wake.wait(list).unwrap_or_else(|p| p.into_inner());
                }
            };
            studio.downloads.run(&studio, &next);
        });
    }

    fn run(&self, studio: &Arc<Studio>, id: &str) {
        let Some(e) = entry(id) else { return };
        let name = str_or(e, "name", id).to_string();
        let dir = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).map(|d| d.dir.clone()).unwrap_or_else(|| default_dir(studio));
        self.update(id, |d| {
            d.status = "downloading";
            d.file = "listing the files".into();
        });
        studio.log.push(format!("downloading {name} into {}", dir.display()));
        match self.fetch(studio, e, &dir) {
            Ok(Want::Run) => {
                self.update(id, |d| {
                    d.status = "adding";
                    d.file.clear();
                    d.speed = 0.;
                });
                match register(studio, e, &dir) {
                    Ok(added) => {
                        studio.log.push(format!("downloaded {name}: {}", if added.is_empty() { "nothing to add".into() } else { added.join(", ") }));
                        self.update(id, |d| {
                            d.status = "done";
                            d.added = added;
                        });
                    }
                    Err(err) => self.update(id, |d| {
                        d.status = "failed";
                        d.error = Some(format!("downloaded, but adding it failed: {err}"));
                    }),
                }
            }
            Ok(Want::Pause) => {
                studio.log.push(format!("paused {name}"));
                self.update(id, |d| {
                    d.status = "paused";
                    d.speed = 0.;
                });
            }
            Ok(Want::Cancel) => {
                remove_partials(id, &dir);
                studio.log.push(format!("cancelled {name}"));
                self.update(id, |d| {
                    d.status = "cancelled";
                    d.speed = 0.;
                    d.done = 0;
                });
            }
            Err(err) => {
                studio.log.push(format!("downloading {name} failed: {err}"));
                self.update(id, |d| {
                    d.status = "failed";
                    d.error = Some(err);
                    d.speed = 0.;
                });
            }
        }
    }

    /// Every file of the entry into `dir`: one this computer has already used where it is, the others downloaded and
    /// checked; stops early when paused or cancelled.
    fn fetch(&self, studio: &Arc<Studio>, e: &Json, dir: &Path) -> Result<Want, String> {
        let id = str_or(e, "id", "");
        let cfg = studio.config();
        let given = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).and_then(|d| d.token.clone());
        let files = plan(&cfg, e, dir, token_for(&cfg, given.as_deref()).as_deref())?;
        let total: u64 = files.iter().filter_map(|f| f.size).sum();
        let have = |f: &Planned| f.size.is_some_and(|s| size_of(&f.dest) == s) || (f.size.is_none() && f.dest.is_file());
        let mut done: u64 = files.iter().filter(|f| have(f)).filter_map(|f| f.size).sum();
        let partials: u64 = files.iter().filter(|f| !have(f)).map(|f| size_of(&part_path(&f.dest))).sum();
        // (the model files on this computer, looked through once: a file wanted that is one of them is not fetched)
        self.update(id, |d| d.file = "looking for its files on this computer".into());
        let here = library::scan(&library::roots(&studio.root, &cfg, dir));
        let found_here: u64 = files
            .iter()
            .filter(|f| !have(f))
            .filter_map(|f| f.size.filter(|s| *s >= library::MIN_BYTES && library::likely(&studio.root, &here, *s, f.sha256.as_deref()).is_some()))
            .sum();
        if let Some(free) = free_space(dir) {
            let need = total.saturating_sub(done + partials + found_here);
            if free < need + (1 << 30) {
                return Err(format!("{} needs {:.1} GB more on that drive and it has {:.1} GB free: choose another folder, or make room", str_or(e, "name", id), need as f64 / 1e9, free as f64 / 1e9));
            }
        }
        self.update(id, |d| {
            d.total = total;
            d.done = done;
            d.files_total = files.len();
            d.files_done = files.iter().filter(|f| have(f)).count();
        });
        for f in &files {
            if have(f) {
                continue;
            }
            let want = self.wanted(id);
            if want != Want::Run {
                return Ok(want);
            }
            let shown = f.dest.strip_prefix(dir).unwrap_or(&f.dest).to_string_lossy().replace('\\', "/");
            // Here already (another model's text encoder or VAE, a copy on another drive): used where it is.
            if let (Some(size), Some(sha)) = (f.size.filter(|s| *s >= library::MIN_BYTES), f.sha256.as_deref()) {
                if here.iter().any(|h| h.size == size) {
                    self.update(id, |d| d.file = format!("checking the copy of {shown} on this computer"));
                    let found = library::find(&studio.root, &here, size, sha, |_| self.wanted(id) != Want::Run);
                    let want = self.wanted(id);
                    if want != Want::Run {
                        return Ok(want);
                    }
                    if let Some(from) = found.filter(|from| *from != f.dest) {
                        let how = library::place(&from, &f.dest)?;
                        let said = match how {
                            library::Placed::Linked => "uses the file at",
                            library::Placed::Pointed => "links to the file at",
                            library::Placed::Copied => "copied the file at",
                        };
                        studio.log.push(format!("{shown}: on this computer already; {said} {}", from.display()));
                        done += size;
                        self.update(id, |d| {
                            d.done = done;
                            d.files_done += 1;
                            d.reused += size;
                            d.reused_files += 1;
                        });
                        continue;
                    }
                }
            }
            self.update(id, |d| d.file = shown.clone());
            if let Some(parent) = f.dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let part = part_path(&f.dest);
            // From a computer on the network that has it (OAIY Network) before Hugging Face: the same bytes, carried on
            // from what the partial file holds, and checked the same way below.
            let mut from_peer = None;
            if let (Some(size), Some(sha), Some(peers)) = (f.size.filter(|s| *s >= library::MIN_BYTES), f.sha256.as_deref(), peer_files()) {
                if size_of(&part) != size {
                    self.update(id, |d| d.file = format!("{shown}, from a computer on the network"));
                    let clock = Mutex::new((size_of(&part), Instant::now()));
                    let progress = |now: u64| {
                        let mut c = clock.lock().unwrap_or_else(|p| p.into_inner());
                        let dt = c.1.elapsed().as_secs_f64();
                        if dt >= 1.0 {
                            let rate = now.saturating_sub(c.0) as f64 / dt;
                            self.update(id, |d| d.speed = if d.speed == 0. { rate } else { d.speed * 0.7 + rate * 0.3 });
                            *c = (now, Instant::now());
                        }
                        self.update(id, |d| d.done = done + now);
                    };
                    match peers.fetch(sha, size, &part, &progress, &|| self.wanted(id) != Want::Run) {
                        Ok(Some(name)) => from_peer = Some(name),
                        Ok(None) => {}
                        Err(err) => studio.log.push(format!("{shown}: not from the network ({err}); from Hugging Face instead")),
                    }
                    let want = self.wanted(id);
                    if want != Want::Run {
                        return Ok(want);
                    }
                    self.update(id, |d| d.file = shown.clone());
                }
            }
            // A partial file that is whole already needs only its name.
            if !f.size.is_some_and(|s| size_of(&part) == s) {
                let want = self.curl(&cfg, id, f, &part, done)?;
                if want != Want::Run {
                    return Ok(want);
                }
            }
            if let Some(s) = f.size {
                let got = size_of(&part);
                if got != s {
                    return Err(format!("{shown}: got {got} bytes of {s}; resume the download to try again"));
                }
            }
            // Checked whole against the SHA-256 Hugging Face publishes for it.
            if let Some(sha) = &f.sha256 {
                self.update(id, |d| d.file = format!("checking {shown}"));
                let read = sha256::file(&part, |_| self.wanted(id) != Want::Run).map_err(|e| format!("{shown}: {e}"))?;
                let Some(got) = read else { return Ok(self.wanted(id)) };
                if got != *sha {
                    let _ = std::fs::remove_file(&part);
                    return Err(format!("{shown} did not download whole (its SHA-256 is not the one Hugging Face publishes): it was deleted; resume the download to fetch it again"));
                }
            }
            std::fs::rename(&part, &f.dest).map_err(|e| format!("{}: {e}", f.dest.display()))?;
            if let Some(sha) = &f.sha256 {
                library::remember(&studio.root, &f.dest, sha);
            }
            done += f.size.unwrap_or_else(|| size_of(&f.dest));
            if let Some(name) = &from_peer {
                studio.log.push(format!("{shown}: from {name}, over OAIY Network"));
            }
            self.update(id, |d| {
                d.done = done;
                d.files_done += 1;
                if let Some(name) = from_peer {
                    d.lan += f.size.unwrap_or(0);
                    if !d.lan_from.contains(&name) {
                        d.lan_from.push(name);
                    }
                }
            });
        }
        // Each part's marker: the model is here.
        for p in e.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
            if let Some(m) = marker(dir, p, id) {
                let names: Vec<Json> = files.iter().filter(|f| m.parent().is_some_and(|folder| f.dest.starts_with(folder))).map(|f| Json::str(f.dest.strip_prefix(dir).unwrap_or(&f.dest).to_string_lossy())).collect();
                let _ = std::fs::write(&m, Json::obj([("id", Json::str(id)), ("files", Json::Arr(names))]).to_json());
            }
        }
        Ok(Want::Run)
    }

    /// One file with curl into `part`, resuming it; progress as it grows.
    fn curl(&self, cfg: &Json, id: &str, f: &Planned, part: &Path, done_before: u64) -> Result<Want, String> {
        let mut c = command(&curl_program(cfg));
        c.args(["-sS", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "--connect-timeout", "30", "-C", "-", "-H", "User-Agent: oaiy-studio", "-o"]).arg(part);
        let given = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).and_then(|d| d.token.clone());
        if let (true, Some(t)) = (f.hub, token_for(cfg, given.as_deref())) {
            c.args(["-H", &format!("Authorization: Bearer {t}")]);
        }
        c.arg(&f.url).stdout(Stdio::null()).stderr(Stdio::piped());
        let child = c.spawn().map_err(|e| format!("could not run curl ({e}): OAIY downloads models with the curl program, which Windows 10 and later, macOS and Linux include"))?;
        *self.child.lock().unwrap_or_else(|p| p.into_inner()) = Some(child);
        let (mut last, mut at) = (size_of(part), Instant::now());
        let status = loop {
            std::thread::sleep(Duration::from_millis(400));
            let exited = {
                let mut guard = self.child.lock().unwrap_or_else(|p| p.into_inner());
                let c = guard.as_mut().expect("the running curl");
                c.try_wait().map_err(|e| e.to_string())?
            };
            let now = size_of(part);
            let dt = at.elapsed().as_secs_f64();
            if dt >= 1.0 {
                let rate = (now.saturating_sub(last)) as f64 / dt;
                self.update(id, |d| d.speed = if d.speed == 0. { rate } else { d.speed * 0.7 + rate * 0.3 });
                (last, at) = (now, Instant::now());
            }
            self.update(id, |d| d.done = done_before + now);
            if let Some(status) = exited {
                break status;
            }
            if self.wanted(id) != Want::Run {
                self.stop_child();
            }
        };
        let mut child = self.child.lock().unwrap_or_else(|p| p.into_inner()).take().expect("the running curl");
        let mut err = String::new();
        if let Some(mut s) = child.stderr.take() {
            let _ = s.read_to_string(&mut err);
        }
        let want = self.wanted(id);
        if want != Want::Run {
            return Ok(want);
        }
        if !status.success() {
            return Err(curl_error(&err, &f.url));
        }
        Ok(Want::Run)
    }
}

/// Deletes the partial files of `id`'s parts in `dir`.
fn remove_partials(id: &str, dir: &Path) {
    fn walk(p: &Path, depth: usize) {
        let Ok(rd) = std::fs::read_dir(p) else { return };
        for e in rd.flatten() {
            let q = e.path();
            if q.is_dir() && depth > 0 {
                walk(&q, depth - 1);
            } else if q.extension().is_some_and(|x| x == "part") {
                let _ = std::fs::remove_file(&q);
            }
        }
    }
    if let Some(e) = entry(id) {
        for p in e.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
            if let Ok(folder) = under(dir, str_or(p, "into", "")) {
                walk(&folder, 4);
            }
        }
    }
}

/// A model-list section in words.
fn section_name(section: &str) -> &str {
    match section {
        "llm" => "chat model",
        "image" => "image model",
        "video" => "video model",
        "speech" => "speech model",
        "music" => "music model",
        "sound" => "sound effects model",
        "model3d" => "3D model",
        other => other,
    }
}

/// Adds what the entry names to the model list, as the Models page's Add does: the first a model, each after it added
/// to that model (a part of it: a GGUF's vision projector, a Qwen Image model's base, text encoder and vision tower).
fn register(studio: &Arc<Studio>, e: &Json, dir: &Path) -> Result<Vec<String>, String> {
    let mut cfg = studio.config();
    let mut added = Vec::new();
    let mut first: Option<(&'static str, String)> = None;
    for rel in strings(e.get("add")) {
        let path = under(dir, &rel)?;
        // (a model whose file is named for nothing, `model.safetensors`, named as the catalog names it)
        let generic = first.is_none() && path.file_stem().is_some_and(|n| {
            let n = n.to_string_lossy().to_ascii_lowercase();
            n == "model" || n.starts_with("model-0") || n.starts_with("diffusion_pytorch_model")
        });
        let name = generic.then(|| str_or(e, "id", "").to_string());
        let result = registry::add(&mut cfg, &path, name.as_deref(), first.as_ref().map(|(s, n)| (*s, n.as_str())));
        if let (Ok((a, _)), None) = (&result, &first) {
            first = Some((a.section, a.name.clone()));
        }
        match result {
            Ok((a, _)) => {
                let said = match (a.section, a.name.as_str()) {
                    ("picture", "background") => "background removal".to_string(),
                    ("picture", _) => "upscaling".to_string(),
                    (section, name) => format!("{} {name}{}", section_name(section), if a.enabled { String::new() } else { format!(" (needs {})", a.missing.join(", ")) }),
                };
                // (a model and its parts said once: the parts were added to it, and it is said as it ends up)
                if let Some(at) = added.iter().position(|s: &String| s.split(" (needs ").next() == said.split(" (needs ").next()) {
                    added[at] = said;
                } else {
                    added.push(said);
                }
            }
            Err(err) => return Err(format!("{}: {err}", path.display())),
        }
    }
    studio.set_config(cfg)?;
    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_match_within_and_across_folders() {
        assert!(glob("ckpts/**", "ckpts/a/b.safetensors") && glob("ckpts/**", "ckpts/x.json"));
        assert!(glob("**", "a/b/c") && glob("*.gguf", "Qwen3-8B-Q4_K_M.gguf") && !glob("*.gguf", "sub/x.gguf"));
        assert!(glob("**_mv.*", "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.json") && !glob("**_mv.*", "ckpts/ss_flow_img_dit_1_3B_64_bf16.json"));
        assert!(glob("assets/**", "assets/logo.png") && !glob("assets/**", "model.safetensors"));
        assert!(glob("config.json", "config.json") && !glob("config.json", "sub/config.json"));
    }

    #[test]
    fn the_catalog_reads_and_every_entry_is_complete() {
        let groups: Vec<String> = catalog().get("groups").and_then(Json::as_array).unwrap().iter().map(|g| str_or(g, "id", "").to_string()).collect();
        let ids: Vec<&str> = entries().iter().map(|e| str_or(e, "id", "")).collect();
        assert!(ids.len() >= 10);
        for e in entries() {
            let id = str_or(e, "id", "");
            assert!(groups.contains(&str_or(e, "group", "").to_string()), "{id}: unknown group");
            for k in ["name", "about", "license"] {
                assert!(!str_or(e, k, "").is_empty(), "{id}: no {k}");
            }
            assert!(e.get("size_gb").and_then(Json::as_f64).is_some_and(|s| s > 0.), "{id}: no size");
            let parts = e.get("parts").and_then(Json::as_array).unwrap();
            assert!(!parts.is_empty(), "{id}: no parts");
            for p in parts {
                assert!(!str_or(p, "into", "").is_empty(), "{id}: a part has no folder");
                assert!(!str_or(p, "repo", "").is_empty() || (!str_or(p, "url", "").is_empty() && !str_or(p, "file", "").is_empty()), "{id}: a part names no source");
            }
            assert!(!strings(e.get("add")).is_empty(), "{id}: nothing to add");
            // Whether the Agent can use its tools with a language model: the engine reads tool calls in the
            // Qwen3.5 and GLM formats only, and the setup wizard labels each model by it.
            assert!(e.get("agent_tools").is_none_or(|t| matches!(t, Json::Bool(_))), "{id}: agent_tools is not a boolean");
            // One engine runs every model: an entry no longer says which of two.
            assert!(e.get("engines").is_none(), "{id}: engines (there is one engine)");
            for key in ["ram_gb", "gpu_count"] {
                assert!(e.get(key).is_none_or(|v| v.as_f64().is_some_and(|n| n > 0.)), "{id}: {key} is not a positive number");
            }
            for n in strings(e.get("needs")) {
                assert!(ids.contains(&n.as_str()), "{id} needs {n}, which is not in the catalog");
            }
        }
    }

    #[test]
    fn the_catalog_offers_official_models_and_plain_quantizations_of_them_only() {
        // Not a community fine-tune made to refuse less: an entry names its publisher's model, or a quantization of it.
        for e in entries() {
            let text = format!("{e:?}").to_lowercase();
            for word in ["uncensor", "abliterat", "disinhibit"] {
                assert!(!text.contains(word), "{}: names a {word}ed model", str_or(e, "id", ""));
            }
        }
        // GLM-5.3-Flash is unsloth's 4-bit dynamic GGUF of Z.ai's weights: its six shards and nothing else.
        let glm = entries().iter().find(|e| str_or(e, "id", "") == "glm-5.3-flash").expect("GLM-5.3-Flash is in the catalog");
        let part = &glm.get("parts").and_then(Json::as_array).unwrap()[0];
        assert_eq!(str_or(part, "repo", ""), "unsloth/GLM-5.3-Flash-GGUF");
        let include = strings(part.get("include"));
        for n in 1..=6 {
            assert!(include.iter().any(|g| glob(g, &format!("UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-0000{n}-of-00006.gguf"))));
        }
        for other in ["UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf", "mmproj-F16.gguf", "BF16/GLM-5.3-Flash-BF16-00001-of-00010.gguf"] {
            assert!(!include.iter().any(|g| glob(g, other)), "{other} is not downloaded");
        }
        assert_eq!(strings(glm.get("add")), ["GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf"]);
    }

    #[test]
    fn the_recommended_language_model_can_use_the_agents_tools() {
        // The setup wizard preselects it for the Agent, whose work is tool calls.
        let llms: Vec<&Json> = entries().iter().filter(|e| str_or(e, "group", "") == "llm").collect();
        let recommended: Vec<&&Json> = llms.iter().filter(|e| e.get("recommended").and_then(Json::as_bool) == Some(true)).collect();
        assert_eq!(recommended.len(), 1);
        assert_eq!(recommended[0].get("agent_tools").and_then(Json::as_bool), Some(true), "{}", str_or(recommended[0], "id", ""));
        // And the catalog offers one small enough for a modest GPU that can too.
        assert!(llms.iter().any(|e| e.get("agent_tools").and_then(Json::as_bool) == Some(true) && e.get("vram_gb").and_then(Json::as_f64).is_some_and(|v| v <= 4.)));
    }

    #[test]
    fn paths_stay_under_the_folder() {
        let base = Path::new("/models");
        assert_eq!(under(base, "Pixal3D/ckpts/a.json").unwrap(), base.join("Pixal3D").join("ckpts").join("a.json"));
        assert!(under(base, "../etc/passwd").is_err() && under(base, "C:/x").is_err());
        assert_eq!(percent_encode("a b/c+d.gguf"), "a%20b/c%2Bd.gguf");
    }

    /// A model all of whose parts are done (each part's marker written) is installed and not partly downloaded; one with
    /// a partial file alone is partly downloaded and not installed.
    #[test]
    fn a_model_whose_parts_are_done_is_installed_not_partial() {
        let dir = std::env::temp_dir().join(format!("oaiy-dl-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = crate::config::default_json();
        crate::util::set(crate::models::registry::obj_mut(&mut cfg, &["downloads"]).unwrap(), "dir", Json::str(dir.to_string_lossy()));
        let studio = crate::Studio::for_test(&dir, cfg);
        let of = |id: &str| -> (bool, bool) {
            let state = studio.downloads.state(&studio);
            let m = state.get("models").and_then(Json::as_array).unwrap().iter().find(|m| str_or(m, "id", "") == id).cloned().unwrap();
            (crate::util::bool_or(&m, "installed", false), crate::util::bool_or(&m, "partial", false))
        };
        let folder = dir.join("Qwen3.5-9B-GGUF");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("Qwen3.5-9B-Q4_K_M.gguf.part"), b"half").unwrap();
        assert_eq!(of("qwen3.5-9b"), (false, true), "a partial file alone");
        std::fs::remove_file(folder.join("Qwen3.5-9B-Q4_K_M.gguf.part")).unwrap();
        std::fs::write(folder.join("Qwen3.5-9B-Q4_K_M.gguf"), b"whole").unwrap();
        std::fs::write(folder.join(".oaiy-qwen3.5-9b.json"), br#"{"id":"qwen3.5-9b"}"#).unwrap();
        assert_eq!(of("qwen3.5-9b"), (true, false), "every part done");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A catalog entry's vision projector goes to the entry's own model (its first item), not to a chat model added
    /// before it that has none: Gemma 4's mmproj was Qwen3.5 9B's once.
    #[test]
    fn an_entrys_projector_goes_to_its_own_model() {
        let dir = std::env::temp_dir().join(format!("oaiy-dl-register-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("Other")).unwrap();
        std::fs::create_dir_all(dir.join("Gemma")).unwrap();
        let gguf = |path: &Path, arch: &str| {
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend(0u64.to_le_bytes());
            b.extend(1u64.to_le_bytes());
            b.extend(("general.architecture".len() as u64).to_le_bytes());
            b.extend(b"general.architecture");
            b.extend(8u32.to_le_bytes());
            b.extend((arch.len() as u64).to_le_bytes());
            b.extend(arch.as_bytes());
            std::fs::write(path, b).unwrap();
        };
        gguf(&dir.join("Other").join("other-Q4_K_M.gguf"), "qwen35");
        gguf(&dir.join("Gemma").join("gemma-Q4_K_M.gguf"), "gemma4");
        gguf(&dir.join("Gemma").join("mmproj-F16.gguf"), "clip");
        let mut cfg = crate::config::default_json();
        registry::add(&mut cfg, &dir.join("Other").join("other-Q4_K_M.gguf"), None, None).unwrap();
        let studio = Arc::new(crate::Studio::for_test(&dir, cfg));
        let entry = Json::obj([("id", Json::str("gemma")), ("add", Json::Arr(vec![Json::str("Gemma/gemma-Q4_K_M.gguf"), Json::str("Gemma/mmproj-F16.gguf")]))]);
        let said = register(&studio, &entry, &dir).unwrap();
        assert_eq!(said.len(), 1, "the model said once: {said:?}");
        let cfg = studio.config();
        let models = cfg.get("llm").and_then(|l| l.get("models")).and_then(Json::as_array).unwrap();
        let projector = |stem: &str| models.iter().find(|m| str_or(m, "path", "").contains(stem)).map(|m| str_or(m, "vision_projector", "").to_string()).unwrap();
        assert!(projector("gemma-Q4_K_M").ends_with("mmproj-F16.gguf"), "{models:?}");
        assert_eq!(projector("other-Q4_K_M"), "", "{models:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gated_and_missing_files_are_explained() {
        assert!(curl_error("curl: (22) The requested URL returned error: 401", "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/model.safetensors").contains("accept its licence at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"));
        assert!(curl_error("curl: (22) The requested URL returned error: 404", "https://huggingface.co/a/b/resolve/main/x").contains("not found"));
    }

    /// A file served once per request, from a listener of its own (as Hugging Face's CDN would): how many were asked.
    fn serve(body: Vec<u8>) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file", listener.local_addr().unwrap());
        let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = asked.clone();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && stream.read(&mut byte).is_ok_and(|n| n == 1) {
                    head.push(byte[0]);
                }
                let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = stream.write_all(&body);
            }
        });
        (url, asked)
    }

    /// A file this computer has already (another model's, another folder's) is used where it is, nothing fetched; one
    /// it has not is downloaded and checked against its SHA-256, and one that does not match is refused and deleted.
    #[test]
    fn nothing_here_already_is_downloaded_again_and_a_download_is_checked() {
        use std::sync::atomic::Ordering;
        let root = std::env::temp_dir().join(format!("oaiy-dl-reuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (dir, kept) = (root.join("models"), root.join("drive/qwen3-vl"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&kept).unwrap();
        let encoder: Vec<u8> = (0..library::MIN_BYTES as usize + 11).map(|i| (i * 7 % 253) as u8).collect();
        std::fs::write(kept.join("encoder.safetensors"), &encoder).unwrap();
        let sha_of = |b: &[u8]| {
            let mut h = sha256::Sha256::new();
            h.update(b);
            h.hex()
        };
        let mut cfg = crate::config::default_json();
        let named = Json::parse(format!(r#"{{"x":{{"text_encoder":"{}"}}}}"#, kept.join("encoder.safetensors").display()).as_bytes()).unwrap();
        let mut image = cfg.get("media").and_then(|m| m.get("image")).cloned().unwrap_or(Json::obj(Vec::<(&str, Json)>::new()));
        crate::util::set(&mut image, "models", named);
        let mut media = cfg.get("media").cloned().unwrap_or(Json::obj(Vec::<(&str, Json)>::new()));
        crate::util::set(&mut media, "image", image);
        crate::util::set(&mut cfg, "media", media);
        let studio = Arc::new(Studio::for_test(&root, cfg));
        let d = Downloads::new();
        let entry = |id: &str, url: &str, size: usize, sha: &str| {
            Json::parse(format!(r#"{{"id":"{id}","parts":[{{"url":"{url}","file":"text_encoder/model.safetensors","size":{size},"sha256":"{sha}","into":"{id}"}}]}}"#).as_bytes()).unwrap()
        };
        let start = |id: &str| {
            d.list.lock().unwrap().push(Download { id: id.into(), dir: dir.clone(), status: "downloading", done: 0, total: 0, file: String::new(), files_done: 0, files_total: 0, speed: 0., error: None, added: Vec::new(), reused: 0, reused_files: 0, lan: 0, lan_from: Vec::new(), token: None });
        };

        // here already, elsewhere: used where it is, the server never asked
        let (url, asked) = serve(encoder.clone());
        start("one");
        let e = entry("one", &url, encoder.len(), &sha_of(&encoder));
        assert_eq!(d.fetch(&studio, &e, &dir).unwrap(), Want::Run);
        assert_eq!(asked.load(Ordering::SeqCst), 0, "nothing fetched");
        assert_eq!(std::fs::read(dir.join("one/text_encoder/model.safetensors")).unwrap(), encoder);
        let one = d.list.lock().unwrap().iter().find(|x| x.id == "one").cloned().unwrap();
        assert_eq!((one.reused, one.reused_files), (encoder.len() as u64, 1));

        // not here: downloaded, checked, and known by its SHA-256 from then on
        let other: Vec<u8> = encoder.iter().map(|b| b.wrapping_add(1)).collect();
        let (url, asked) = serve(other.clone());
        start("two");
        assert_eq!(d.fetch(&studio, &entry("two", &url, other.len(), &sha_of(&other)), &dir).unwrap(), Want::Run);
        assert_eq!(asked.load(Ordering::SeqCst), 1);
        let got = dir.join("two/text_encoder/model.safetensors");
        assert_eq!(std::fs::read(&got).unwrap(), other);
        assert_eq!(library::known(&studio.root, &got), Some(sha_of(&other)));

        // a download that is not what Hugging Face says it is: refused, and not kept
        let (url, _) = serve(other.clone());
        start("three");
        let err = d.fetch(&studio, &entry("three", &url, other.len(), &"ab".repeat(32)), &dir).unwrap_err();
        assert!(err.contains("did not download whole"), "{err}");
        let three = dir.join("three/text_encoder/model.safetensors");
        assert!(!three.exists() && !part_path(&three).exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A computer on the network that has one file (by its SHA-256): what it sends, and how often it was asked.
    struct OnePeer {
        sha: String,
        body: Vec<u8>,
        sent: std::sync::atomic::AtomicUsize,
    }

    impl PeerFiles for OnePeer {
        fn have(&self, files: &[(String, u64)]) -> Vec<Option<String>> {
            files.iter().map(|(sha, _)| (*sha == self.sha).then(|| "Rig".to_string())).collect()
        }
        fn fetch(&self, sha256: &str, size: u64, part: &Path, progress: &dyn Fn(u64), _stop: &dyn Fn() -> bool) -> Result<Option<String>, String> {
            if sha256 != self.sha {
                return Ok(None);
            }
            self.sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // (carried on from what the partial file holds)
            let from = size_of(part) as usize;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(part).map_err(|e| e.to_string())?;
            std::io::Write::write_all(&mut f, &self.body[from..]).map_err(|e| e.to_string())?;
            progress(size);
            Ok(Some("Rig".into()))
        }
    }

    /// A file a computer on the network has is fetched from it, not from Hugging Face, carried on from a partial file,
    /// and checked the same way: one it sends wrong is refused. Others are downloaded as before.
    #[test]
    fn a_file_a_computer_on_the_network_has_comes_from_it() {
        use std::sync::atomic::Ordering;
        let root = std::env::temp_dir().join(format!("oaiy-dl-peer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("models");
        std::fs::create_dir_all(&dir).unwrap();
        let weights: Vec<u8> = (0..library::MIN_BYTES as usize + 3).map(|i| (i * 13 % 241) as u8).collect();
        let mut h = sha256::Sha256::new();
        h.update(&weights);
        let sha = h.hex();
        let studio = Arc::new(Studio::for_test(&root, crate::config::default_json()));
        let peer = Arc::new(OnePeer { sha: sha.clone(), body: weights.clone(), sent: Default::default() });
        set_peer_files(Some(peer.clone()));
        let (url, asked) = serve(weights.clone());
        let d = Downloads::new();
        d.list.lock().unwrap().push(Download { id: "net".into(), dir: dir.clone(), status: "downloading", done: 0, total: 0, file: String::new(), files_done: 0, files_total: 0, speed: 0., error: None, added: Vec::new(), reused: 0, reused_files: 0, lan: 0, lan_from: Vec::new(), token: None });
        let e = Json::parse(format!(r#"{{"id":"net","parts":[{{"url":"{url}","file":"model.safetensors","size":{},"sha256":"{sha}","into":"net"}}]}}"#, weights.len()).as_bytes()).unwrap();
        // half of it here already, from an earlier try
        std::fs::create_dir_all(dir.join("net")).unwrap();
        std::fs::write(part_path(&dir.join("net/model.safetensors")), &weights[..weights.len() / 2]).unwrap();
        let r = d.fetch(&studio, &e, &dir);
        set_peer_files(None);
        assert_eq!(r.unwrap(), Want::Run);
        assert_eq!((peer.sent.load(Ordering::SeqCst), asked.load(Ordering::SeqCst)), (1, 0), "from the computer on the network, not the internet");
        assert_eq!(std::fs::read(dir.join("net/model.safetensors")).unwrap(), weights);
        let net = d.list.lock().unwrap().iter().find(|x| x.id == "net").cloned().unwrap();
        assert_eq!((net.lan, net.lan_from.clone()), (weights.len() as u64, vec!["Rig".to_string()]));

        // what it sends wrong is refused, as a download would be
        let wrong = Arc::new(OnePeer { sha: sha.clone(), body: weights.iter().map(|b| b ^ 1).collect(), sent: Default::default() });
        set_peer_files(Some(wrong));
        d.list.lock().unwrap().push(Download { id: "bad".into(), dir: dir.clone(), status: "downloading", done: 0, total: 0, file: String::new(), files_done: 0, files_total: 0, speed: 0., error: None, added: Vec::new(), reused: 0, reused_files: 0, lan: 0, lan_from: Vec::new(), token: None });
        let bad = Json::parse(format!(r#"{{"id":"bad","parts":[{{"url":"{url}","file":"model.safetensors","size":{},"sha256":"{sha}","into":"bad"}}]}}"#, weights.len()).as_bytes()).unwrap();
        // (the first copy is in the library: moved aside, so this one is fetched)
        std::fs::remove_dir_all(dir.join("net")).unwrap();
        let err = d.fetch(&studio, &bad, &dir).unwrap_err();
        set_peer_files(None);
        assert!(err.contains("did not download whole"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// What getting a few catalog models takes on this computer, from its own settings (read, nothing written or
    /// started): OAIY_TEST_STUDIO_CONFIG=<its oaiy-studio.json> cargo test -p oaiy-studio plans_here -- --ignored --nocapture
    #[test]
    #[ignore]
    fn plans_here() {
        let Ok(path) = std::env::var("OAIY_TEST_STUDIO_CONFIG") else { return };
        let path = PathBuf::from(path);
        let cfg = Json::parse(&std::fs::read(&path).unwrap()).unwrap();
        let studio = Studio::for_test(path.parent().unwrap(), cfg);
        for id in std::env::var("OAIY_TEST_PLANS").unwrap_or_else(|_| "qwen3.5-9b,moss-image-sft-q4,flux-3-image-klein".into()).split(',') {
            let started = Instant::now();
            match studio.downloads.plan_of(&studio, id, None) {
                Ok(p) => {
                    let gb = |k: &str| p.get(k).and_then(Json::as_i64).unwrap_or(0) as f64 / 1e9;
                    println!("{id}: {:.2} GB to download, {:.2} GB elsewhere here, {:.2} GB in place ({:.1} s)", gb("download_bytes"), gb("elsewhere_bytes"), gb("here_bytes"), started.elapsed().as_secs_f64());
                    for f in p.get("files").and_then(Json::as_array).unwrap_or(&[]) {
                        println!("  {:<10} {:>8.2} GB  {}  {}", str_or(f, "status", ""), f.get("size").and_then(Json::as_i64).unwrap_or(0) as f64 / 1e9, str_or(f, "path", ""), str_or(f, "from", ""));
                    }
                }
                Err(e) => println!("{id}: {e}"),
            }
        }
    }

    /// A real download of a small model (BiRefNet's config) with curl, into a temp folder.
    /// cargo test --release -p oaiy-studio downloads -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_small_file_downloads_and_resumes() {
        let dir = std::env::temp_dir().join(format!("oaiy-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let e = Json::parse(br#"{"id":"t","parts":[{"repo":"ZhengPeng7/BiRefNet","include":["config.json","BiRefNet_config.py"],"into":"BiRefNet"}]}"#).unwrap();
        let cfg = crate::config::default_json();
        let files = plan(&cfg, &e, &dir, None).unwrap();
        assert_eq!(files.len(), 2);
        let d = Downloads::new();
        d.list.lock().unwrap().push(Download { id: "t".into(), dir: dir.clone(), status: "downloading", done: 0, total: 0, file: String::new(), files_done: 0, files_total: 0, speed: 0., error: None, added: Vec::new(), reused: 0, reused_files: 0, lan: 0, lan_from: Vec::new(), token: None });
        for f in &files {
            std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
            // Half a file first, as a pause leaves it; curl carries on from there.
            if let Some(s) = f.size {
                let whole = curl_text(&cfg, &f.url, None, false).unwrap();
                assert_eq!(whole.len() as u64, s);
                std::fs::write(part_path(&f.dest), &whole.as_bytes()[..(s / 2) as usize]).unwrap();
            }
            assert_eq!(d.curl(&cfg, "t", f, &part_path(&f.dest), 0).unwrap(), Want::Run);
            assert_eq!(Some(size_of(&part_path(&f.dest))), f.size);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
