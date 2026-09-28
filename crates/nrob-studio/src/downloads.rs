//! Getting models: a catalog of the mainstream models nrob runs
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
//! A finished part leaves `.nrob-<id>.json` in its folder, which is how the
//! catalog knows the model is there.
use crate::util::str_or;
use crate::{config, registry, Studio};
use nrob::json::Json;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
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
    let mut c = Command::new(program);
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
fn curl_text(cfg: &Json, url: &str, auth: bool, head: bool) -> Result<String, String> {
    let mut c = command(&curl_program(cfg));
    c.args(["-sS", "-L", "--fail", "--connect-timeout", "30", "--max-time", "120", "-H", "User-Agent: nrob-studio"]);
    if head {
        c.arg("-I");
    }
    if let (true, Some(t)) = (auth, token(cfg)) {
        c.args(["-H", &format!("Authorization: Bearer {t}")]);
    }
    c.arg(url);
    let out = c.output().map_err(|e| format!("could not run curl ({e}): nrob downloads models with the curl program, which Windows 10 and later, macOS and Linux include"))?;
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
            Some(r) => format!("{r} is gated: sign in on huggingface.co, accept its licence at https://huggingface.co/{r}, then add a Hugging Face access token (read) here"),
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
fn plan(cfg: &Json, entry: &Json, dir: &Path) -> Result<Vec<Planned>, String> {
    let mut out = Vec::new();
    for part in entry.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
        let into = under(dir, str_or(part, "into", ""))?;
        let repo = str_or(part, "repo", "");
        if !repo.is_empty() {
            let rev = str_or(part, "revision", "main");
            let list = curl_text(cfg, &format!("https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"), true, false)?;
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
                out.push(Planned { url: format!("https://huggingface.co/{repo}/resolve/{rev}/{}", percent_encode(path)), dest: under(&into, path)?, size, hub: true });
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
            out.push(Planned { url: url.to_string(), dest: under(&into, file)?, size, hub: false });
        }
    }
    Ok(out)
}

fn head_size(cfg: &Json, url: &str) -> Option<u64> {
    let headers = curl_text(cfg, url, false, true).ok()?;
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
    under(dir, str_or(part, "into", "")).ok().map(|p| p.join(format!(".nrob-{id}.json")))
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
        ])
    }
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

    /// Queue `id` (and what it needs that is not there yet) into `dir`.
    pub fn start(&self, studio: &Arc<Studio>, id: &str, dir: Option<PathBuf>) -> Result<Json, String> {
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
                    }
                    None => list.push(Download { id: id.clone(), dir: dir.clone(), status: "queued", done: 0, total: 0, file: String::new(), files_done: 0, files_total: 0, speed: 0., error: None, added: Vec::new() }),
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
                o.push(("installed".into(), Json::Bool(installed(e, &where_))));
                o.push(("partial".into(), Json::Bool(d.is_none() && partial(e, &where_))));
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

    /// Every file of the entry into `dir`; stops early when paused or cancelled.
    fn fetch(&self, studio: &Arc<Studio>, e: &Json, dir: &Path) -> Result<Want, String> {
        let id = str_or(e, "id", "");
        let cfg = studio.config();
        let files = plan(&cfg, e, dir)?;
        let total: u64 = files.iter().filter_map(|f| f.size).sum();
        let have = |f: &Planned| f.size.is_some_and(|s| size_of(&f.dest) == s) || (f.size.is_none() && f.dest.is_file());
        let mut done: u64 = files.iter().filter(|f| have(f)).filter_map(|f| f.size).sum();
        let partials: u64 = files.iter().filter(|f| !have(f)).map(|f| size_of(&part_path(&f.dest))).sum();
        if let Some(free) = free_space(dir) {
            let need = total.saturating_sub(done + partials);
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
            self.update(id, |d| d.file = shown.clone());
            if let Some(parent) = f.dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let part = part_path(&f.dest);
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
            std::fs::rename(&part, &f.dest).map_err(|e| format!("{}: {e}", f.dest.display()))?;
            done += f.size.unwrap_or_else(|| size_of(&f.dest));
            self.update(id, |d| {
                d.done = done;
                d.files_done += 1;
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
        c.args(["-sS", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "--connect-timeout", "30", "-C", "-", "-H", "User-Agent: nrob-studio", "-o"]).arg(part);
        if let (true, Some(t)) = (f.hub, token(cfg)) {
            c.args(["-H", &format!("Authorization: Bearer {t}")]);
        }
        c.arg(&f.url).stdout(Stdio::null()).stderr(Stdio::piped());
        let child = c.spawn().map_err(|e| format!("could not run curl ({e}): nrob downloads models with the curl program, which Windows 10 and later, macOS and Linux include"))?;
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

/// Adds what the entry names to the model list, as the Models page's Add does.
fn register(studio: &Arc<Studio>, e: &Json, dir: &Path) -> Result<Vec<String>, String> {
    let mut cfg = studio.config();
    let mut added = Vec::new();
    for rel in strings(e.get("add")) {
        let path = under(dir, &rel)?;
        match registry::add(&mut cfg, &path, None, None) {
            Ok((a, _)) => added.push(match (a.section, a.name.as_str()) {
                ("picture", "background") => "background removal".to_string(),
                ("picture", _) => "upscaling".to_string(),
                (section, name) => format!("{} {name}{}", section_name(section), if a.enabled { String::new() } else { format!(" (needs {})", a.missing.join(", ")) }),
            }),
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
            for n in strings(e.get("needs")) {
                assert!(ids.contains(&n.as_str()), "{id} needs {n}, which is not in the catalog");
            }
        }
    }

    #[test]
    fn paths_stay_under_the_folder() {
        let base = Path::new("/models");
        assert_eq!(under(base, "Pixal3D/ckpts/a.json").unwrap(), base.join("Pixal3D").join("ckpts").join("a.json"));
        assert!(under(base, "../etc/passwd").is_err() && under(base, "C:/x").is_err());
        assert_eq!(percent_encode("a b/c+d.gguf"), "a%20b/c%2Bd.gguf");
    }

    #[test]
    fn gated_and_missing_files_are_explained() {
        assert!(curl_error("curl: (22) The requested URL returned error: 401", "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/model.safetensors").contains("accept its licence at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"));
        assert!(curl_error("curl: (22) The requested URL returned error: 404", "https://huggingface.co/a/b/resolve/main/x").contains("not found"));
    }

    /// A real download of a small model (BiRefNet's config) with curl, into a temp folder.
    /// cargo test --release -p nrob-studio downloads -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_small_file_downloads_and_resumes() {
        let dir = std::env::temp_dir().join(format!("nrob-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let e = Json::parse(br#"{"id":"t","parts":[{"repo":"ZhengPeng7/BiRefNet","include":["config.json","BiRefNet_config.py"],"into":"BiRefNet"}]}"#).unwrap();
        let cfg = crate::config::default_json();
        let files = plan(&cfg, &e, &dir).unwrap();
        assert_eq!(files.len(), 2);
        let d = Downloads::new();
        d.list.lock().unwrap().push(Download { id: "t".into(), dir: dir.clone(), status: "downloading", done: 0, total: 0, file: String::new(), files_done: 0, files_total: 0, speed: 0., error: None, added: Vec::new() });
        for f in &files {
            std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
            // Half a file first, as a pause leaves it; curl carries on from there.
            if let Some(s) = f.size {
                let whole = curl_text(&cfg, &f.url, true, false).unwrap();
                assert_eq!(whole.len() as u64, s);
                std::fs::write(part_path(&f.dest), &whole.as_bytes()[..(s / 2) as usize]).unwrap();
            }
            assert_eq!(d.curl(&cfg, "t", f, &part_path(&f.dest), 0).unwrap(), Want::Run);
            assert_eq!(Some(size_of(&part_path(&f.dest))), f.size);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
