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
/// Refused where it has characters no token has ([`check_token`]).
fn token_for(cfg: &Json, given: Option<&str>) -> Result<Option<String>, String> {
    match given.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => check_token(t).map(|_| Some(t.to_string())),
        None => token(cfg),
    }
}

/// The studio's own token: the settings', else the environment's (`HF_TOKEN`).
fn token(cfg: &Json) -> Result<Option<String>, String> {
    let t = cfg.get("downloads").map_or("", |d| str_or(d, "hf_token", "")).trim().to_string();
    if !t.is_empty() {
        return check_token(&t).map(|_| Some(t));
    }
    let env = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().find_map(|k| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()));
    match env {
        Some(t) => check_token(&t).map(|_| Some(t)).map_err(|e| format!("HF_TOKEN: {e}")),
        None => Ok(None),
    }
}

/// Whether `t` reads as a Hugging Face token (`hf_` and letters and digits): letters, digits, `_`, `.` and `-` alone.
/// It is written into curl's configuration ([`token_to_curl`]), where a line break would begin an option of its own
/// (one that writes a file anywhere, say), so nothing else is taken.
pub(crate) fn check_token(t: &str) -> Result<(), String> {
    if t.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')) {
        Ok(())
    } else {
        Err("the Hugging Face token has characters no token has (only letters, digits, _ . and -): copy it again from huggingface.co/settings/tokens".into())
    }
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

const NO_CURL: &str = "OAIY downloads models with the curl program, which Windows 10 and later, macOS and Linux include";

/// The Hugging Face token said to curl on its standard input (`--config -`), never on its command line, which any
/// program on the computer can read in the list of processes: what to write to curl once it is started.
fn token_to_curl(c: &mut Command, token: Option<&str>) -> Option<String> {
    // (checked where it was given; one that is not a token is never written into curl's configuration)
    let t = token.filter(|t| check_token(t).is_ok())?;
    c.args(["--config", "-"]).stdin(Stdio::piped());
    Some(format!("header = \"Authorization: Bearer {}\"\n", t.replace('\\', "\\\\").replace('"', "\\\"")))
}

/// Give a started curl what [`token_to_curl`] asked to be written to it.
fn feed(child: &mut Child, said: Option<String>) {
    use std::io::Write;
    if let (Some(text), Some(mut stdin)) = (said, child.stdin.take()) {
        let _ = stdin.write_all(text.as_bytes());
    }
}

/// Runs curl to completion, returning its stdout (text). `headers`: `-I` (the headers alone) or `-i` (the headers,
/// then the body). `gated`: the model's licence is accepted first (said so where Hugging Face refuses).
fn curl_text(cfg: &Json, url: &str, auth: Option<&str>, headers: Option<&str>, gated: bool) -> Result<String, String> {
    let mut c = command(&curl_program(cfg));
    c.args(["-sS", "-L", "--fail", "--connect-timeout", "30", "--max-time", "120", "-H", "User-Agent: oaiy-studio"]);
    if let Some(flag) = headers {
        c.arg(flag);
    }
    let said = token_to_curl(&mut c, auth);
    c.arg(url).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| format!("could not run curl ({e}): {NO_CURL}"))?;
    feed(&mut child, said);
    let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(curl_error(&String::from_utf8_lossy(&out.stderr), url, gated));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The most pages of one file list read ([`hub_list`]): Hugging Face answers a large repository's in parts (some
/// thousand entries each), and a list that never ends is not followed for ever.
const LIST_PAGES: usize = 50;

/// Every entry of a list Hugging Face answers in pages (`/api/models/{repo}/tree/{rev}`): each page's, and the next
/// page's where its `Link` header names one (`rel="next"`; at the same site alone, as the token goes with it).
fn hub_list(cfg: &Json, url: &str, token: Option<&str>, gated: bool) -> Result<Vec<Json>, String> {
    let origin = origin_of(url);
    let mut all = Vec::new();
    let mut next = Some(url.to_string());
    let mut pages = 0;
    while let Some(at) = next.take() {
        pages += 1;
        if pages > LIST_PAGES {
            return Err(format!("{url}: the file list runs on past {LIST_PAGES} pages"));
        }
        let text = curl_text(cfg, &at, token, Some("-i"), gated)?;
        let (head, body) = split_head(&text);
        match Json::parse(body.trim().as_bytes()).map_err(|e| format!("{at}: the file list did not read ({e})"))? {
            Json::Arr(items) => all.extend(items),
            _ => return Err(format!("{at}: the file list did not read (not a list)")),
        }
        next = next_link(head, &origin);
    }
    Ok(all)
}

/// `scheme://host[:port]` of a URL.
fn origin_of(url: &str) -> String {
    let after = url.find("://").map_or(0, |i| i + 3);
    let end = url[after..].find(['/', '?', '#']).map_or(url.len(), |i| after + i);
    url[..end].to_string()
}

/// What `curl -i -L` printed, as the last response's headers and its body (a redirect's, and a `100 Continue`'s,
/// headers come before them).
fn split_head(text: &str) -> (&str, &str) {
    let (mut head, mut rest) = ("", text);
    while rest.starts_with("HTTP/") {
        let Some((end, gap)) = rest.find("\r\n\r\n").map(|i| (i, 4)).or_else(|| rest.find("\n\n").map(|i| (i, 2))) else {
            return (rest, "");
        };
        head = &rest[..end];
        rest = &rest[end + gap..];
    }
    (head, rest)
}

/// The next page a `Link` header names (`<url>; rel="next"`), where it is at `origin` (a path alone is taken as one
/// there).
fn next_link(head: &str, origin: &str) -> Option<String> {
    for line in head.lines() {
        let Some((name, value)) = line.split_once(':') else { continue };
        if !name.trim().eq_ignore_ascii_case("link") {
            continue;
        }
        for item in value.split('<').skip(1) {
            let Some((url, params)) = item.split_once('>') else { continue };
            let params = params.split('<').next().unwrap_or("");
            let next = params.split(';').any(|p| {
                let p = p.trim().trim_end_matches(',').trim();
                p.split_once('=').is_some_and(|(k, v)| k.trim().eq_ignore_ascii_case("rel") && v.trim().trim_matches('"').split_whitespace().any(|r| r.eq_ignore_ascii_case("next")))
            });
            if !next {
                continue;
            }
            let url = url.trim();
            let full = if url.starts_with('/') { format!("{origin}{url}") } else { url.to_string() };
            // (the token goes with each page: only the site the list began at is asked)
            let rest = full.strip_prefix(origin)?;
            return (rest.is_empty() || rest.starts_with(['/', '?'])).then_some(full);
        }
    }
    None
}

/// The HTTP status curl's `--fail` said it was refused with ("The requested URL returned error: 403").
fn refused_with(stderr: &str) -> Option<u16> {
    let at = stderr.find("returned error: ")? + "returned error: ".len();
    stderr[at..].chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok()
}

/// curl's complaint, in words a person can act on. `gated`: the model's licence is accepted on huggingface.co first.
fn curl_error(stderr: &str, url: &str, gated: bool) -> String {
    let e = stderr.trim();
    let repo = url.strip_prefix("https://huggingface.co/").map(|r| r.trim_start_matches("api/models/")).and_then(|r| {
        let mut p = r.split('/');
        Some(format!("{}/{}", p.next()?, p.next()?))
    });
    let host = url.split('/').take(3).collect::<Vec<_>>().join("/");
    // (curl's own error number, `curl: (28) …`: not any number in what it said, which may be a time or a size)
    let code = |n: u32| e.contains(&format!("curl: ({n})"));
    match refused_with(e) {
        Some(401 | 403) if gated => {
            return match repo {
                Some(r) => format!("{r} is gated: sign in on huggingface.co, accept its licence at https://huggingface.co/{r}, then add a Hugging Face access token (read) in OAIY's Settings or on Get models"),
                None => format!("{url} refused the download: its licence is accepted first, with a Hugging Face token"),
            };
        }
        Some(401) => return format!("{host} asked for a sign-in (HTTP 401): the model may be private; a Hugging Face token (read) in OAIY's Settings or on Get models may let it download"),
        Some(403) => return format!("{host} refused the download (HTTP 403): its download link may have expired; try again"),
        Some(404) => return format!("{url} was not found (the catalog may be older than the model's page)"),
        Some(n) => return format!("{host} answered HTTP {n}: try again later"),
        None => {}
    }
    if code(23) || e.to_ascii_lowercase().contains("no space") {
        return "writing the file failed: is the disk full?".into();
    }
    if code(28) {
        return format!("{host} stopped sending (curl timed out): resume the download to carry on");
    }
    if code(6) || code(7) || code(35) {
        return format!("could not reach {host} ({e}); check the internet connection");
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
            let gated = entry.get("gated").and_then(Json::as_bool).unwrap_or(false);
            let tree = hub_list(cfg, &format!("https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"), token, gated)?;
            let (include, exclude) = (strings(part.get("include")), strings(part.get("exclude")));
            for f in &tree {
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
    let headers = curl_text(cfg, url, None, Some("-I"), false).ok()?;
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

/// Whether planned file `f` is in place: as large as it should be, and not a file whose SHA-256 is known to be another
/// file's (one as large, but not it, is fetched again).
fn in_place(studio_root: &Path, f: &Planned) -> bool {
    match f.size {
        Some(s) => size_of(&f.dest) == s && !f.sha256.as_deref().is_some_and(|want| library::known(studio_root, &f.dest).is_some_and(|h| h != want)),
        None => f.dest.is_file(),
    }
}

fn marker(dir: &Path, part: &Json, id: &str) -> Option<PathBuf> {
    under(dir, str_or(part, "into", "")).ok().map(|p| p.join(format!(".oaiy-{id}.json")))
}

/// Whether the settings have a model of the catalog's `group` named `id` (as one added from the catalog is named, or a
/// person named it): a language model's `name`, a media model's key.
fn configured(cfg: &Json, group: &str, id: &str) -> bool {
    if id.is_empty() {
        return false;
    }
    match group {
        "llm" => cfg.get("llm").and_then(|l| l.get("models")).and_then(Json::as_array).unwrap_or(&[]).iter().any(|m| str_or(m, "name", "") == id),
        "picture" => false,
        section => cfg.get("media").and_then(|m| m.get(section)).and_then(|s| s.get("models")).and_then(|m| m.get(id)).is_some(),
    }
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
    /// Where each of its files goes (a file another download wants is not deleted with this one), and the files this
    /// download put there (deleted with it when it is cancelled, unless another model uses them).
    planned: Vec<PathBuf>,
    fetched: Vec<PathBuf>,
}

impl Download {
    fn new(id: &str, dir: &Path, token: Option<String>) -> Download {
        Download {
            id: id.to_string(),
            dir: dir.to_path_buf(),
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
            token,
            planned: Vec::new(),
            fetched: Vec::new(),
        }
    }

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
    /// The curl fetching a file now, and whether it was stopped here (a pause), not failed.
    child: Mutex<Option<Child>>,
    stopped: Mutex<bool>,
    wake: Condvar,
    worker: Mutex<bool>,
    free: Mutex<Option<(PathBuf, Instant, Option<u64>)>>,
    /// Stopped for good ([`Downloads::shutdown`]): nothing is started, resumed or added to the models after.
    closed: std::sync::atomic::AtomicBool,
    /// Held while a download is added to the models (the settings saved): a shutdown waits for it, and none is
    /// added after one.
    adding: Mutex<()>,
}

/// What a download asked of a studio that is stopping is told.
const STOPPING: &str = "the studio is stopping: downloads carry on from where they were at its next start";

impl Downloads {
    pub fn new() -> Self {
        Self::default()
    }

    fn closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Stop every download for good, as the studio stops (OAIY quits, updates, or puts a backup back and starts the
    /// studio again): each is paused where it is (its partial files kept: the next start resumes them), its curl
    /// killed and waited for, and nothing is started, resumed or added to the models after. Returns once a download
    /// being added to the models is added (its settings saved: none saved after this, over settings put back), and
    /// the worker has let go of its curl (a few seconds at most).
    pub fn shutdown(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        {
            let mut list = self.list.lock().unwrap_or_else(|p| p.into_inner());
            let mut want = self.want.lock().unwrap_or_else(|p| p.into_inner());
            for d in list.iter_mut().filter(|d| matches!(d.status, "queued" | "downloading")) {
                want.insert(d.id.clone(), Want::Pause);
                if d.status == "queued" {
                    d.status = "paused";
                }
            }
        }
        if let Some(c) = self.child.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            *self.stopped.lock().unwrap_or_else(|p| p.into_inner()) = true;
            let _ = c.kill();
            let _ = c.wait();
        }
        // (the worker, waiting for the next download, ends)
        self.wake.notify_all();
        drop(self.adding.lock().unwrap_or_else(|p| p.into_inner()));
        // (a curl started just before the pause was seen is killed by the worker within its next look, 0.4 s)
        let until = Instant::now() + Duration::from_secs(5);
        while Instant::now() < until && self.child.lock().unwrap_or_else(|p| p.into_inner()).is_some() {
            std::thread::sleep(Duration::from_millis(50));
        }
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

    /// What a step of `id`'s that was stopped part-way (a copy, a check) stopped for: paused or cancelled, and paused
    /// where it has been asked for again since (never `Run`: the step is not done, and its model is not whole; the
    /// download then has its turn again, [`Downloads::run`]).
    fn stopped_as(&self, id: &str) -> Want {
        match self.wanted(id) {
            Want::Run => Want::Pause,
            other => other,
        }
    }

    fn stop_child(&self) {
        if let Some(c) = self.child.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            *self.stopped.lock().unwrap_or_else(|p| p.into_inner()) = true;
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
        if self.closed() {
            return Err(STOPPING.into());
        }
        let e = entry(id).ok_or_else(|| format!("no model {id} in the catalog"))?;
        if let Some(t) = token.as_deref() {
            check_token(t.trim())?;
        }
        let dir = chosen_dir(studio, dir)?;
        let mut ids: Vec<String> = strings(e.get("needs")).into_iter().filter(|n| entry(n).is_some_and(|x| !installed(x, &dir))).collect();
        ids.push(id.to_string());
        {
            let mut list = self.list.lock().unwrap_or_else(|p| p.into_inner());
            // (again under the list's lock: a shutdown pauses what it finds queued under it)
            if self.closed() {
                return Err(STOPPING.into());
            }
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
                    None => list.push(Download::new(id, &dir, token.clone())),
                }
            }
        }
        // (a folder other than where downloads go, kept: what is partly downloaded there is found after a restart)
        if dir != default_dir(studio) {
            remember_dir(studio, &dir);
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

    /// Stop `id` and delete what it downloaded: its partial files, and the files it put in place, except any another
    /// model uses. `dir`: where it is partly downloaded, where no download of it is listed (after a restart).
    pub fn cancel(&self, studio: &Studio, id: &str, dir: Option<PathBuf>) -> Result<Json, String> {
        // (being added to the models: its files are the model's now, not a download's to take away)
        let listed = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).map(|d| (d.status, d.dir.clone()));
        match &listed {
            Some(("adding", _)) => return Err("it is being added to your models now: remove it under Models once it is there".into()),
            // (downloaded whole: its files are a model's, in the settings or to be added again, not a download's to
            // take away)
            Some(("done", _)) => return Err("it is downloaded: remove it under Models".into()),
            None if entry(id).is_some_and(|e| installed(e, &dir.clone().unwrap_or_else(|| default_dir(studio)))) => {
                return Err("it is downloaded: remove it under Models".into());
            }
            _ => {}
        }
        self.want.lock().unwrap_or_else(|p| p.into_inner()).insert(id.to_string(), Want::Cancel);
        let (active, dir) = {
            let list = self.list.lock().unwrap_or_else(|p| p.into_inner());
            let d = list.iter().find(|d| d.id == id);
            (d.is_some_and(|d| d.status == "downloading"), d.map(|d| d.dir.clone()).or(dir).unwrap_or_else(|| default_dir(studio)))
        };
        if active {
            self.stop_child();
        } else {
            self.discard(studio, id, &dir);
            self.update(id, |d| d.status = "cancelled");
        }
        Ok(self.state(studio))
    }

    /// What other models use: the files another download (listed, not cancelled) puts in place, and those another
    /// downloaded model's markers list, in `dir` and every other folder downloads went to (another drive's model may
    /// be a link to a file of this one's: [`library::place`]); and this model's own in those other folders.
    fn used_by_others(&self, studio: &Studio, id: &str, dir: &Path) -> std::collections::HashSet<PathBuf> {
        let (mut used, mut dirs): (std::collections::HashSet<PathBuf>, Vec<PathBuf>) = {
            let list = self.list.lock().unwrap_or_else(|p| p.into_inner());
            let used = list.iter().filter(|d| d.id != id && d.status != "cancelled").flat_map(|d| d.planned.iter().cloned()).collect();
            (used, list.iter().map(|d| d.dir.clone()).collect())
        };
        dirs.push(default_dir(studio));
        dirs.extend(remembered_dirs(studio));
        dirs.insert(0, dir.to_path_buf());
        let mut seen = Vec::new();
        for folder in dirs {
            if seen.contains(&folder) {
                continue;
            }
            seen.push(folder.clone());
            for other in entries() {
                let oid = str_or(other, "id", "");
                // (this model's own markers in `dir` are what is being taken away; elsewhere they are another copy's)
                if oid == id && folder == dir {
                    continue;
                }
                for part in other.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
                    let Some(m) = marker(&folder, part, oid).filter(|m| m.is_file()) else { continue };
                    let listed = std::fs::read(&m).ok().and_then(|b| Json::parse(&b).ok());
                    for f in listed.as_ref().and_then(|l| l.get("files")).and_then(Json::as_array).unwrap_or(&[]) {
                        if let Some(rel) = f.as_str() {
                            used.insert(under(&folder, &rel.replace('\\', "/")).unwrap_or_else(|_| folder.join(rel)));
                        }
                    }
                }
            }
        }
        used
    }

    /// Write down beside `dir` what download `id` has put in place so far (`.oaiy-<id>.fetched.json`), so a Discard after
    /// a restart, with no download of it listed, still knows what to take away.
    fn keep_fetched(&self, id: &str, dir: &Path) {
        let fetched = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).map(|d| d.fetched.clone()).unwrap_or_default();
        let _ = std::fs::write(fetched_record(dir, id), Json::Arr(fetched.iter().map(|f| Json::str(f.to_string_lossy())).collect()).to_json());
    }

    /// Delete what `id` downloaded into `dir`: its partial files and the files this download put in place (a link
    /// removed is the link alone), its markers, and the folders that leaves empty; never a file another model uses.
    fn discard(&self, studio: &Studio, id: &str, dir: &Path) {
        let used = self.used_by_others(studio, id, dir);
        // (what it put in place, as listed, else as its record beside the folder says: after a restart)
        let listed = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).map(|d| d.fetched.clone()).unwrap_or_default();
        let fetched = if listed.is_empty() { read_fetched(dir, id) } else { listed };
        // (a file the settings name, a model or a part of one, is the settings': never removed with a download)
        let named = settings_files(studio);
        remove_partials(id, dir, &used);
        // (each also as the file it is: a link another model has on another drive leads to this one's file, and a
        // file the settings name may be such a link)
        let real = |p: &Path| std::fs::canonicalize(p).ok();
        let used_real: std::collections::HashSet<PathBuf> = used.iter().filter_map(|p| real(p)).collect();
        let named_real: Vec<PathBuf> = named.iter().filter_map(|p| real(p)).collect();
        let kept = |f: &Path| {
            if used.contains(f) || named.iter().any(|n| f.starts_with(n)) {
                return true;
            }
            // (a link of its own goes as a link: only the file itself is another's to keep)
            let is_link = std::fs::symlink_metadata(f).is_ok_and(|m| m.file_type().is_symlink());
            !is_link && real(f).is_some_and(|r| used_real.contains(&r) || named_real.iter().any(|n| r.starts_with(n)))
        };
        for f in fetched.iter().filter(|f| !kept(f)) {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_file(fetched_record(dir, id));
        if let Some(e) = entry(id) {
            for part in e.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
                if let Some(m) = marker(dir, part, id) {
                    let _ = std::fs::remove_file(&m);
                }
                if let Ok(folder) = under(dir, str_or(part, "into", "")) {
                    remove_empty(&folder);
                }
            }
        }
        self.update(id, |d| {
            d.fetched.clear();
            d.done = 0;
        });
        library::forget_scan();
    }

    /// Add a downloaded model to the model list again.
    pub fn add_again(&self, studio: &Arc<Studio>, id: &str, dir: Option<PathBuf>) -> Result<Json, String> {
        let e = entry(id).ok_or_else(|| format!("no model {id} in the catalog"))?;
        let dir = dir.unwrap_or_else(|| default_dir(studio));
        if !installed(e, &dir) {
            return Err(format!("{} is not downloaded in {}", str_or(e, "name", id), dir.display()));
        }
        let _adding = self.adding.lock().unwrap_or_else(|p| p.into_inner());
        if self.closed() {
            return Err(STOPPING.into());
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
        let elsewhere = remembered_dirs(studio);
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
                // (with no download of it listed, where it is: where downloads go, else a folder they went to before)
                let where_ = d.map(|d| d.dir.clone()).unwrap_or_else(|| {
                    std::iter::once(dir.clone()).chain(elsewhere.iter().cloned()).find(|f| installed(e, f) || partial(e, f)).unwrap_or_else(|| dir.clone())
                });
                let have = installed(e, &where_);
                o.push(("installed".into(), Json::Bool(have)));
                // (a model all of whose parts are done is installed, not partly downloaded: each part's marker is
                // what says a part is done)
                o.push(("partial".into(), Json::Bool(d.is_none() && !have && partial(e, &where_))));
                if where_ != dir {
                    o.push(("at".into(), Json::str(where_.to_string_lossy())));
                }
                // (a model of that name in the settings already, added from wherever its files are: said, so its card
                // does not only offer to download it)
                o.push(("configured".into(), Json::Bool(!have && configured(&cfg, str_or(e, "group", ""), id))));
                o.push(("download".into(), d.map_or(Json::Null, Download::to_json)));
                Json::Obj(o)
            })
            .collect();
        Json::obj([
            ("dir", Json::str(dir.to_string_lossy())),
            ("free", self.free_cached(&dir).map_or(Json::Null, |v| Json::Int(v as i64))),
            ("token", Json::Bool(token(&cfg).is_ok_and(|t| t.is_some()))),
            ("groups", catalog().get("groups").cloned().unwrap_or(Json::Arr(Vec::new()))),
            ("models", Json::Arr(models)),
        ])
    }

    /// What getting `id` takes, before it is got: each of its files here already (where it goes), on this computer
    /// elsewhere (another model's, another drive's: used where it is; `checked` once its SHA-256 has been read, else
    /// read when it is got), on a computer on the network (OAIY Network: fetched from it), or to download; and the
    /// bytes of each, the folder's free space, and the models it needs that are not here yet. Asks Hugging Face for
    /// the files' list (`token`: as for [`Downloads::start`]).
    pub fn plan_of(&self, studio: &Studio, id: &str, token: Option<String>, dir: Option<PathBuf>) -> Result<Json, String> {
        let e = entry(id).ok_or_else(|| format!("no model {id} in the catalog"))?;
        // (the folder the download would go to: the one the page has in its box, saved or not, else where downloads go;
        // as Download takes it, `~` its home folder, refused where it is not a full path)
        let dir = chosen_dir(studio, dir)?;
        let cfg = studio.config();
        let token = token_for(&cfg, token.as_deref())?;
        let files = plan(&cfg, e, &dir, token.as_deref())?;
        let here = library::scan(&library::roots(&studio.root, &cfg, &dir));
        // Each file: in place, on this computer elsewhere, on a computer on the network, or to download.
        let mut said: Vec<(&'static str, Option<String>, Option<bool>)> = files
            .iter()
            .map(|f| {
                if in_place(&studio.root, f) {
                    ("here", None, None)
                } else if let Some((from, checked)) = f.size.filter(|s| *s >= library::MIN_BYTES).zip(f.sha256.as_deref()).and_then(|(s, sha)| library::likely(&studio.root, &here, s, sha)) {
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
        let (mut at_home, mut elsewhere, mut on_network, mut to_fetch, mut copied) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let listed: Vec<Json> = files
            .iter()
            .zip(&said)
            .map(|(f, (status, from, checked))| {
                let size = f.size.unwrap_or(0);
                let shown = f.dest.strip_prefix(&dir).unwrap_or(&f.dest).to_string_lossy().replace('\\', "/");
                // (a file here put in place by a copy, where no link can be made: it takes its room again)
                let copy = *status == "elsewhere" && from.as_deref().is_some_and(|from| library::needs_copy(Path::new(from), &f.dest));
                match *status {
                    "here" => at_home += size,
                    "elsewhere" => elsewhere += size,
                    "network" => on_network += size.saturating_sub(size_of(&part_path(&f.dest))),
                    _ => to_fetch += size.saturating_sub(size_of(&part_path(&f.dest))),
                }
                if copy {
                    copied += size;
                }
                let mut o = vec![("path", Json::str(&shown)), ("size", f.size.map_or(Json::Null, |s| Json::Int(s as i64))), ("status", Json::str(*status))];
                if let Some(from) = from {
                    o.push(("from", Json::str(from)));
                }
                if let Some(c) = checked {
                    o.push(("checked", Json::Bool(*c)));
                }
                if copy {
                    o.push(("copy", Json::Bool(true)));
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
            ("copy_bytes", Json::Int(copied as i64)),
            // (what it asks of the computer, from the catalog: the GPU memory, and the computer's own where it says)
            ("vram_gb", e.get("vram_gb").cloned().unwrap_or(Json::Null)),
            ("ram_gb", e.get("ram_gb").cloned().unwrap_or(Json::Null)),
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
                    library::likely(&studio.root, &here, *size, sha)
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
                    // (the studio stopping: the worker ends, and lets go of it)
                    if studio.downloads.closed() {
                        return;
                    }
                    if let Some(d) = list.iter().find(|d| d.status == "queued") {
                        break d.id.clone();
                    }
                    list = studio.downloads.wake.wait(list).unwrap_or_else(|p| p.into_inner());
                }
            };
            // (a download that panics, on a file that could not be read, fails alone: the next still run)
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| studio.downloads.run(&studio, &next)));
            if ran.is_err() {
                if let Some(mut c) = studio.downloads.child.lock().unwrap_or_else(|p| p.into_inner()).take() {
                    let _ = c.kill();
                }
                studio.downloads.update(&next, |d| {
                    d.status = "failed";
                    d.error = Some("the download stopped on an error of the studio's own: try again".into());
                    d.speed = 0.;
                });
            }
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
        let fetched = if self.closed() { Ok(Want::Pause) } else { self.fetch(studio, e, &dir) };
        // (paused or cancelled, then asked for again before the pause was seen: its turn again, not left paused, nor
        // cancelled once what it had is gone)
        let again = || !self.closed() && self.wanted(id) == Want::Run;
        match fetched {
            Ok(Want::Run) => {
                self.update(id, |d| {
                    d.status = "adding";
                    d.file.clear();
                    d.speed = 0.;
                });
                // (added under the lock a shutdown waits for, and not once it has begun: the settings saved now would
                // be saved over the ones a restore puts back, by a studio that is going)
                let adding = self.adding.lock().unwrap_or_else(|p| p.into_inner());
                if self.closed() {
                    drop(adding);
                    studio.log.push(format!("downloaded {name}; it is added to the models at the next start (Get models, Add again)"));
                    self.update(id, |d| d.status = "paused");
                    return;
                }
                let registered = register(studio, e, &dir);
                drop(adding);
                match registered {
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
                let resumed = again();
                if !resumed {
                    studio.log.push(format!("paused {name}"));
                }
                self.update(id, |d| {
                    d.status = if resumed { "queued" } else { "paused" };
                    d.speed = 0.;
                });
            }
            Ok(Want::Cancel) => {
                self.discard(studio, id, &dir);
                studio.log.push(format!("cancelled {name}"));
                let resumed = again();
                self.update(id, |d| {
                    d.status = if resumed { "queued" } else { "cancelled" };
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

    /// Every file of the entry into `dir`: one this computer has already used where it is, one a computer on the
    /// network has fetched from it, the others downloaded; each checked; stops early when paused or cancelled.
    fn fetch(&self, studio: &Arc<Studio>, e: &Json, dir: &Path) -> Result<Want, String> {
        let id = str_or(e, "id", "");
        let gated = e.get("gated").and_then(Json::as_bool).unwrap_or(false);
        let cfg = studio.config();
        let given = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).and_then(|d| d.token.clone());
        let files = plan(&cfg, e, dir, token_for(&cfg, given.as_deref())?.as_deref())?;
        self.update(id, |d| d.planned = files.iter().map(|f| f.dest.clone()).collect());
        let total: u64 = files.iter().filter_map(|f| f.size).sum();
        let have = |f: &Planned| in_place(&studio.root, f);
        let mut done: u64 = files.iter().filter(|f| have(f)).filter_map(|f| f.size).sum();
        let partials: u64 = files.iter().filter(|f| !have(f)).map(|f| size_of(&part_path(&f.dest))).sum();
        // (the model files on this computer, looked through once: a file wanted that is one of them is not fetched,
        // and takes no room again unless it has to be copied)
        self.update(id, |d| d.file = "looking for its files on this computer".into());
        let here = library::scan(&library::roots(&studio.root, &cfg, dir));
        let found_here: u64 = files
            .iter()
            .filter(|f| !have(f))
            .filter_map(|f| {
                let (size, sha) = (f.size.filter(|s| *s >= library::MIN_BYTES)?, f.sha256.as_deref()?);
                let (from, _) = library::likely(&studio.root, &here, size, sha)?;
                (!library::needs_copy(&from, &f.dest)).then_some(size)
            })
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
        let stop = || self.wanted(id) != Want::Run;
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
                    let found = library::find(&studio.root, &here, size, sha, |_| stop());
                    let want = self.wanted(id);
                    if want != Want::Run {
                        return Ok(want);
                    }
                    if let Some(from) = found.filter(|from| *from != f.dest) {
                        let base = done;
                        let mut copying = |n: u64| self.update(id, |d| d.done = base + n);
                        let Some(how) = library::place(&from, &f.dest, &mut copying, &stop)? else { return Ok(self.stopped_as(id)) };
                        let said = match how {
                            library::Placed::Linked => "uses the file at",
                            library::Placed::Pointed => "links to the file at",
                            library::Placed::Copied => "copied the file at",
                        };
                        studio.log.push(format!("{shown}: on this computer already; {said} {}", from.display()));
                        // (known by its SHA-256 where it is now too: not read again)
                        library::remember(&studio.root, &f.dest, sha);
                        library::forget_scan();
                        done += size;
                        self.update(id, |d| {
                            d.done = done;
                            d.files_done += 1;
                            d.reused += size;
                            d.reused_files += 1;
                            d.fetched.push(f.dest.clone());
                        });
                        self.keep_fetched(id, dir);
                        continue;
                    }
                }
            }
            self.update(id, |d| d.file = shown.clone());
            if let Some(parent) = f.dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let part = part_path(&f.dest);
            // A partial file larger than the file is not part of it (the file changed since, or another's bytes were
            // added to it): carried on from its end, it is refused every time, so it is begun again.
            if f.size.is_some_and(|s| size_of(&part) > s) {
                studio.log.push(format!("{shown}: the partial file is larger than the file ({} bytes of {}): downloading it again from the start", size_of(&part), f.size.unwrap_or(0)));
                std::fs::remove_file(&part).map_err(|e| format!("{}: {e}", part.display()))?;
            }
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
                    match peers.fetch(sha, size, &part, &progress, &stop) {
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
            loop {
                // A partial file that is whole already needs only its name.
                if !f.size.is_some_and(|s| size_of(&part) == s) {
                    let want = self.curl(&cfg, id, f, &part, done, gated)?;
                    if want != Want::Run {
                        return Ok(want);
                    }
                }
                if let Some(s) = f.size {
                    let got = size_of(&part);
                    if got > s {
                        // (so the next try begins it again, rather than be refused at its end as this one was)
                        let _ = std::fs::remove_file(&part);
                        return Err(format!("{shown}: got {got} bytes where the file has {s}: the partial file was deleted; resume the download to fetch it again"));
                    }
                    if got != s {
                        return Err(format!("{shown}: got {got} bytes of {s}; resume the download to try again"));
                    }
                }
                // Checked whole against the SHA-256 Hugging Face publishes for it.
                let Some(sha) = &f.sha256 else { break };
                self.update(id, |d| d.file = format!("checking {shown}"));
                let read = sha256::file(&part, |_| stop()).map_err(|e| format!("{shown}: {e}"))?;
                let Some(got) = read else { return Ok(self.stopped_as(id)) };
                if got == *sha {
                    break;
                }
                let _ = std::fs::remove_file(&part);
                // (what a computer on the network sent was not the file: from Hugging Face now, in this same run, and
                // not from that computer again)
                if let Some(name) = from_peer.take() {
                    studio.log.push(format!("{shown} from {name} was not the file (its SHA-256 is not the one Hugging Face publishes): from Hugging Face instead"));
                    continue;
                }
                return Err(format!("{shown} did not download whole (its SHA-256 is not the one Hugging Face publishes): it was deleted; resume the download to fetch it again"));
            }
            std::fs::rename(&part, &f.dest).map_err(|e| format!("{}: {e}", f.dest.display()))?;
            if let Some(sha) = &f.sha256 {
                library::remember(&studio.root, &f.dest, sha);
            }
            library::forget_scan();
            done += f.size.unwrap_or_else(|| size_of(&f.dest));
            if let Some(name) = &from_peer {
                studio.log.push(format!("{shown}: from {name}, over OAIY Network"));
            }
            self.update(id, |d| {
                d.done = done;
                d.files_done += 1;
                d.fetched.push(f.dest.clone());
                if let Some(name) = from_peer {
                    d.lan += f.size.unwrap_or(0);
                    if !d.lan_from.contains(&name) {
                        d.lan_from.push(name);
                    }
                }
            });
            self.keep_fetched(id, dir);
        }
        // Each part's marker: the model is here (and what it fetched is in them: no record of its own any more).
        for p in e.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
            if let Some(m) = marker(dir, p, id) {
                let names: Vec<Json> = files.iter().filter(|f| m.parent().is_some_and(|folder| f.dest.starts_with(folder))).map(|f| Json::str(f.dest.strip_prefix(dir).unwrap_or(&f.dest).to_string_lossy())).collect();
                let _ = std::fs::write(&m, Json::obj([("id", Json::str(id)), ("files", Json::Arr(names))]).to_json());
            }
        }
        let _ = std::fs::remove_file(fetched_record(dir, id));
        Ok(Want::Run)
    }

    /// One file with curl into `part`, resuming it; progress as it grows.
    fn curl(&self, cfg: &Json, id: &str, f: &Planned, part: &Path, done_before: u64, gated: bool) -> Result<Want, String> {
        loop {
            // (no curl begun for a studio that is stopping)
            if self.closed() {
                return Ok(Want::Pause);
            }
            let mut c = command(&curl_program(cfg));
            // (a connection that stops sending is given up after a minute below 1 KB/s, and `--retry` tries again,
            // rather than waiting on it for ever)
            c.args(["-sS", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60", "-C", "-", "-H", "User-Agent: oaiy-studio", "-o"]).arg(part);
            let given = self.list.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|d| d.id == id).and_then(|d| d.token.clone());
            let said = token_to_curl(&mut c, token_for(cfg, given.as_deref())?.as_deref().filter(|_| f.hub));
            c.arg(&f.url).stdout(Stdio::null()).stderr(Stdio::piped());
            let mut child = c.spawn().map_err(|e| format!("could not run curl ({e}): {NO_CURL}"))?;
            feed(&mut child, said);
            *self.stopped.lock().unwrap_or_else(|p| p.into_inner()) = false;
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
            // (stopped here, a pause, and resumed before it was seen: carried on, not a failure)
            if !status.success() && std::mem::take(&mut *self.stopped.lock().unwrap_or_else(|p| p.into_inner())) {
                continue;
            }
            if !status.success() {
                return Err(curl_error(&err, &f.url, gated));
            }
            return Ok(Want::Run);
        }
    }
}

/// Deletes the partial files of `id`'s parts in `dir`, except one of a file another model uses (`used`: two models
/// downloading into one folder, the language model and the picture model that has it as its text encoder).
fn remove_partials(id: &str, dir: &Path, used: &std::collections::HashSet<PathBuf>) {
    fn walk(p: &Path, depth: usize, used: &std::collections::HashSet<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(p) else { return };
        for e in rd.flatten() {
            let q = e.path();
            // (a folder of its own, not a link to one: a linked folder is another's, wherever it leads)
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_dir() {
                if depth > 0 {
                    walk(&q, depth - 1, used);
                }
            } else if kind.is_file() && q.extension().is_some_and(|x| x == "part") && !used.contains(&q.with_extension("")) {
                let _ = std::fs::remove_file(&q);
            }
        }
    }
    if let Some(e) = entry(id) {
        for p in e.get("parts").and_then(Json::as_array).unwrap_or(&[]) {
            if let Ok(folder) = under(dir, str_or(p, "into", "")) {
                walk(&folder, 4, used);
            }
        }
    }
}

/// The folder a download goes to, as the person wrote it (`~` their home folder), else where downloads go; refused
/// where it is not a full path.
fn chosen_dir(studio: &Studio, dir: Option<PathBuf>) -> Result<PathBuf, String> {
    let dir = dir
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| PathBuf::from(library::expand_home(&d.to_string_lossy())))
        .unwrap_or_else(|| default_dir(studio));
    if !dir.is_absolute() {
        return Err("the folder to save to must be a full path".into());
    }
    Ok(dir)
}

/// Where what download `id` put in place in `dir` is written down until it is done.
fn fetched_record(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!(".oaiy-{id}.fetched.json"))
}

/// What download `id` put in place in `dir`, as its record says (nothing outside `dir` is taken as its).
fn read_fetched(dir: &Path, id: &str) -> Vec<PathBuf> {
    std::fs::read(fetched_record(dir, id))
        .ok()
        .and_then(|b| Json::parse(&b).ok())
        .and_then(|j| j.as_array().map(|a| a.iter().filter_map(Json::as_str).map(PathBuf::from).filter(|p| p.starts_with(dir)).collect()))
        .unwrap_or_default()
}

/// The files and folders the settings' models name (a model, a part of one): a download never removes them.
fn settings_files(studio: &Studio) -> Vec<PathBuf> {
    let cfg = studio.config();
    let mut named = Vec::new();
    let mut collect = |v: &Json| library::paths_named(v, &mut named);
    if let Some(llm) = cfg.get("llm").and_then(|l| l.get("models")) {
        collect(llm);
    }
    for (_, section) in cfg.get("media").and_then(Json::as_object).unwrap_or(&[]) {
        if let Some(models) = section.get("models") {
            collect(models);
        }
    }
    named.into_iter().map(|n| config::resolve(&studio.root, &n)).filter(|p| p.components().count() > 1).collect()
}

/// Remove `dir` and the folders in it that hold nothing.
fn remove_empty(dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                remove_empty(&e.path());
            }
        }
    }
    let _ = std::fs::remove_dir(dir);
}

/// The folders downloads went to besides the one they go to now (`download-dirs.json` beside the settings): what is
/// partly downloaded there is found after a restart.
fn remembered_dirs(studio: &Studio) -> Vec<PathBuf> {
    std::fs::read(studio.root.join(DIRS_FILE))
        .ok()
        .and_then(|b| Json::parse(&b).ok())
        .and_then(|j| j.as_array().map(|a| a.iter().filter_map(Json::as_str).map(PathBuf::from).collect()))
        .unwrap_or_default()
}

fn remember_dir(studio: &Studio, dir: &Path) {
    let mut dirs = remembered_dirs(studio);
    dirs.retain(|d| d != dir);
    dirs.insert(0, dir.to_path_buf());
    dirs.truncate(8);
    let _ = std::fs::write(studio.root.join(DIRS_FILE), Json::Arr(dirs.iter().map(|d| Json::str(d.to_string_lossy())).collect()).to_json());
}

const DIRS_FILE: &str = "download-dirs.json";

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
    // (on the settings as they are now, under the save's lock: a save beside it neither loses this nor is lost)
    studio.update_config(|cfg| register_in(cfg, e, dir).map_err(|e| (400, e))).map(|(added, _)| added).map_err(|(_, e)| e)
}

fn register_in(cfg: &mut Json, e: &Json, dir: &Path) -> Result<Vec<String>, String> {
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
        let result = registry::add(cfg, &path, name.as_deref(), first.as_ref().map(|(s, n)| (*s, n.as_str())));
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
    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model already in the settings under the catalog's name, added from wherever its files are, is said to be so.
    #[test]
    fn a_model_in_the_settings_by_the_catalogs_name_is_said_to_be_there() {
        let cfg = Json::parse(br#"{"llm":{"models":[{"name":"qwen3.5-9b"}]},"media":{"image":{"models":{"qwen-image-2.1-turbo":{}}}}}"#).unwrap();
        assert!(configured(&cfg, "llm", "qwen3.5-9b"));
        assert!(configured(&cfg, "image", "qwen-image-2.1-turbo"));
        assert!(!configured(&cfg, "image", "flux-3-image-klein"));
        assert!(!configured(&cfg, "video", "qwen-image-2.1-turbo"), "another kind's model of that name is not it");
        assert!(!configured(&cfg, "llm", ""));
    }

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

    /// The Hugging Face token reaches curl on its standard input, never on its command line (which any program on the
    /// computer can read).
    #[test]
    fn the_token_is_not_on_curls_command_line() {
        let mut c = Command::new("curl");
        let said = token_to_curl(&mut c, Some("hf_secret")).unwrap();
        let args: Vec<String> = c.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(args, ["--config", "-"]);
        assert!(!args.iter().any(|a| a.contains("hf_secret")));
        assert_eq!(said, "header = \"Authorization: Bearer hf_secret\"\n");
        assert!(token_to_curl(&mut Command::new("curl"), None).is_none());
    }

    #[test]
    fn gated_and_missing_files_are_explained() {
        assert!(curl_error("curl: (22) The requested URL returned error: 401", "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/model.safetensors", true).contains("accept its licence at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"));
        assert!(curl_error("curl: (22) The requested URL returned error: 404", "https://huggingface.co/a/b/resolve/main/x", false).contains("not found"));
        // (a number in what curl said is not a status: a timeout of 120403 ms is not a refusal)
        let slow = curl_error("curl: (28) Operation timed out after 120403 milliseconds with 0 bytes received", "https://huggingface.co/a/b/resolve/main/x", true);
        assert!(slow.contains("stopped sending") && !slow.contains("gated"), "{slow}");
        // (a model that is not gated, refused: its link expired, not a licence to accept)
        let expired = curl_error("curl: (22) The requested URL returned error: 403", "https://cdn-lfs.hf.co/x", false);
        assert!(expired.contains("expired") && !expired.contains("gated"), "{expired}");
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
            d.list.lock().unwrap().push(Download { status: "downloading", ..Download::new(id, &dir, None) });
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
        d.list.lock().unwrap().push(Download { status: "downloading", ..Download::new("net", &dir, None) });
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

        // what it sends wrong is refused, and the file fetched from Hugging Face in the same run instead
        let wrong = Arc::new(OnePeer { sha: sha.clone(), body: weights.iter().map(|b| b ^ 1).collect(), sent: Default::default() });
        set_peer_files(Some(wrong.clone()));
        d.list.lock().unwrap().push(Download { status: "downloading", ..Download::new("bad", &dir, None) });
        let bad = Json::parse(format!(r#"{{"id":"bad","parts":[{{"url":"{url}","file":"model.safetensors","size":{},"sha256":"{sha}","into":"bad"}}]}}"#, weights.len()).as_bytes()).unwrap();
        // (the first copy is in the library: moved aside, so this one is fetched)
        std::fs::remove_dir_all(dir.join("net")).unwrap();
        library::forget_scan();
        let r = d.fetch(&studio, &bad, &dir);
        set_peer_files(None);
        assert_eq!(r.unwrap(), Want::Run);
        assert_eq!((wrong.sent.load(Ordering::SeqCst), asked.load(Ordering::SeqCst)), (1, 1), "asked of the computer once, then of the internet");
        assert_eq!(std::fs::read(dir.join("bad/model.safetensors")).unwrap(), weights);
        let bad = d.list.lock().unwrap().iter().find(|x| x.id == "bad").cloned().unwrap();
        assert_eq!(bad.lan, 0, "nothing counted as from the network");
        assert!(studio.log.since(0).to_json().contains("was not the file"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Cancelled, a download deletes what it put in place and its partial files: never a file another model uses,
    /// one this computer had before, or the file a link points to.
    #[test]
    fn a_cancelled_download_takes_away_what_it_fetched_and_nothing_else() {
        let root = std::env::temp_dir().join(format!("oaiy-dl-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("models");
        let kept = root.join("drive/kept.safetensors");
        std::fs::create_dir_all(kept.parent().unwrap()).unwrap();
        std::fs::write(&kept, b"weights kept elsewhere").unwrap();
        let mine = dir.join("Qwen3.5-9B-GGUF");
        std::fs::create_dir_all(&mine).unwrap();
        let (fetched, linked, before, other_part, own_part) = (mine.join("mine.gguf"), mine.join("linked.gguf"), mine.join("before.gguf"), mine.join("theirs.gguf.part"), mine.join("mine2.gguf.part"));
        std::fs::write(&fetched, b"downloaded").unwrap();
        std::fs::hard_link(&kept, &linked).unwrap();
        std::fs::write(&before, b"was here").unwrap();
        std::fs::write(&other_part, b"half").unwrap();
        std::fs::write(&own_part, b"half").unwrap();
        let d = Downloads::new();
        // this model (its catalog folder `Qwen3.5-9B-GGUF`), and another downloading into the same folder
        let id = "qwen3.5-9b";
        d.list.lock().unwrap().push(Download { status: "paused", fetched: vec![fetched.clone(), linked.clone()], ..Download::new(id, &dir, None) });
        d.list.lock().unwrap().push(Download { status: "downloading", planned: vec![mine.join("theirs.gguf")], ..Download::new("moss-image-sft-q4", &dir, None) });
        let studio = Studio::for_test(&root, crate::config::default_json());
        d.discard(&studio, id, &dir);
        assert!(!fetched.exists() && !linked.exists(), "what it put in place goes");
        assert!(kept.exists(), "the file a link was made to stays");
        assert!(before.exists(), "what was here before stays");
        assert!(other_part.exists(), "another download's partial file stays");
        assert!(!own_part.exists(), "its own partial file goes");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// After a restart (no download of it listed), Discard still takes away what the download put in place, as its
    /// record beside the folder says; never a file the settings' models name (the Qwen3.5-9B GGUF a picture model's
    /// text encoder is); and not while the download is being added to the models.
    #[test]
    fn a_discard_after_a_restart_knows_what_it_fetched_and_keeps_what_the_settings_use() {
        let root = std::env::temp_dir().join(format!("oaiy-dl-restart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("models");
        let folder = dir.join("MOSS-Image-GGUF");
        std::fs::create_dir_all(&folder).unwrap();
        let shared = dir.join("Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        let (own, outside) = (folder.join("MOSS-Image-SFT-Q4_K_M.gguf"), root.join("elsewhere.gguf"));
        for f in [&own, &shared, &outside] {
            std::fs::write(f, b"weights").unwrap();
        }
        // the settings name the shared GGUF (the language model): it stays
        let cfg = Json::parse(format!(r#"{{"llm":{{"models":[{{"name":"qwen3.5-9b","path":"{}"}}]}}}}"#, shared.display()).as_bytes()).unwrap();
        let mut full = crate::config::default_json();
        crate::util::set(&mut full, "llm", cfg.get("llm").cloned().unwrap());
        let studio = Studio::for_test(&root, full);
        // what a download had fetched, written down as it went (a record that names a file outside the folder is
        // not believed)
        let id = "moss-image-sft-q4";
        let d = Downloads::new();
        d.list.lock().unwrap().push(Download { status: "paused", fetched: vec![own.clone(), shared.clone(), outside.clone()], ..Download::new(id, &dir, None) });
        d.keep_fetched(id, &dir);
        assert!(fetched_record(&dir, id).is_file());
        // a restart: nothing listed; Discard reads the record
        let after = Downloads::new();
        after.cancel(&studio, id, Some(dir.clone())).unwrap();
        assert!(!own.exists(), "what it put in place goes");
        assert!(shared.exists(), "a file the settings name stays");
        assert!(outside.exists(), "nothing outside its folder is taken as its");
        assert!(!fetched_record(&dir, id).exists(), "its record goes with it");
        // being added to the models: not discarded
        after.list.lock().unwrap().push(Download { status: "adding", ..Download::new(id, &dir, None) });
        assert!(after.cancel(&studio, id, Some(dir.clone())).is_err());
        // a folder as the person wrote it: `~` their home folder; one that is not a full path refused
        assert!(chosen_dir(&studio, Some(PathBuf::from("models"))).is_err());
        if let Some(home) = std::env::var_os("HOME") {
            assert_eq!(chosen_dir(&studio, Some(PathBuf::from("~/Models"))).unwrap(), PathBuf::from(home).join("Models"));
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A Hugging Face token is letters, digits, `_`, `.` and `-`: anything else (a line break, which would begin an
    /// option of curl's own in its configuration) is refused where it is given, kept or used, and never reaches curl.
    #[test]
    fn a_token_with_other_characters_is_refused_and_never_reaches_curl() {
        assert!(check_token("hf_AbC123.x-y").is_ok() && check_token("").is_ok());
        for bad in ["hf_x\noutput = \"/tmp/x\"", "hf_\"x", "hf x", "hf_x\\", "hf_\rx"] {
            assert!(check_token(bad).is_err(), "{bad:?}");
            assert!(token_for(&Json::Null, Some(bad)).is_err(), "{bad:?}");
            assert!(token_to_curl(&mut Command::new("curl"), Some(bad)).is_none(), "{bad:?}");
        }
        assert_eq!(token_for(&Json::Null, Some(" hf_ok ")).unwrap().as_deref(), Some("hf_ok"));
        let mut cfg = crate::config::default_json();
        crate::util::set(crate::models::registry::obj_mut(&mut cfg, &["downloads"]).unwrap(), "hf_token", Json::str("hf_a\nb"));
        assert!(token_for(&cfg, None).is_err(), "the settings' own, refused where it is used");
        assert!(crate::config::validate(&cfg).unwrap_err().contains("downloads.hf_token"), "and where it is saved");
        crate::util::set(crate::models::registry::obj_mut(&mut cfg, &["downloads"]).unwrap(), "hf_token", Json::str("hf_fine"));
        assert!(crate::config::validate(&cfg).is_ok());
        assert_eq!(token_for(&cfg, None).unwrap().as_deref(), Some("hf_fine"));
        // and on a download asked for with one
        let root = std::env::temp_dir().join(format!("oaiy-dl-token-{}", std::process::id()));
        let studio = Arc::new(Studio::for_test(&root, crate::config::default_json()));
        assert!(studio.downloads.start(&studio, "qwen3.5-9b", None, Some("hf_x\nfoo".into())).unwrap_err().contains("Hugging Face token"));
        assert!(studio.downloads.list.lock().unwrap().is_empty(), "nothing queued");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A file list Hugging Face answers in pages is read whole: each next page its `Link` header names, at the same
    /// site alone.
    #[test]
    fn a_file_list_in_pages_is_read_whole_at_its_own_site_alone() {
        let head = "HTTP/1.1 302 Found\r\nLocation: /x\r\n\r\nHTTP/2 200\r\nlink: <https://huggingface.co/api/models/a/b/tree/main?recursive=true&cursor=Zz>; rel=\"next\"\r\n\r\n[{\"path\":\"a\"}]";
        let (h, body) = split_head(head);
        assert_eq!(body, "[{\"path\":\"a\"}]");
        assert_eq!(next_link(h, "https://huggingface.co").as_deref(), Some("https://huggingface.co/api/models/a/b/tree/main?recursive=true&cursor=Zz"));
        assert_eq!(next_link("Link: </api/x?cursor=1>; rel=next", "https://huggingface.co").as_deref(), Some("https://huggingface.co/api/x?cursor=1"));
        assert_eq!(next_link("Link: <https://huggingface.co.evil.example/x>; rel=\"next\"", "https://huggingface.co"), None, "another site");
        assert_eq!(next_link("Link: <https://huggingface.co/x>; rel=\"prev\"", "https://huggingface.co"), None);
        assert_eq!(next_link("Content-Type: application/json", "https://huggingface.co"), None);
        assert_eq!(origin_of("http://127.0.0.1:5/api/x?y"), "http://127.0.0.1:5");
        // two pages from a server of its own, with curl
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let next = format!("{base}/api/models/a/b/tree/main?recursive=true&cursor=2");
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten().take(2) {
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && stream.read(&mut byte).is_ok_and(|n| n == 1) {
                    head.push(byte[0]);
                }
                let (link, body) = if String::from_utf8_lossy(&head).contains("cursor=2") {
                    (String::new(), r#"[{"type":"file","path":"two"}]"#)
                } else {
                    (format!("Link: <{next}>; rel=\"next\"\r\n"), r#"[{"type":"file","path":"one"}]"#)
                };
                let _ = write!(stream, "HTTP/1.1 200 OK\r\n{link}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        let all = hub_list(&crate::config::default_json(), &format!("{base}/api/models/a/b/tree/main?recursive=true"), None, false).unwrap();
        let paths: Vec<&str> = all.iter().map(|f| str_or(f, "path", "")).collect();
        assert_eq!(paths, ["one", "two"]);
    }

    /// A stopped studio's downloads are paused where they are and their curl killed; none is started, resumed or added
    /// to the models after (the settings a restore put back are not saved over).
    #[cfg(unix)]
    #[test]
    fn a_shutdown_pauses_every_download_kills_its_curl_and_starts_none_after() {
        let root = std::env::temp_dir().join(format!("oaiy-dl-shutdown-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let studio = Arc::new(Studio::for_test(&root, crate::config::default_json()));
        let dir = root.join("models");
        let d = &studio.downloads;
        d.list.lock().unwrap().push(Download { status: "queued", ..Download::new("qwen3.5-9b", &dir, None) });
        d.list.lock().unwrap().push(Download { status: "downloading", ..Download::new("moss-image-sft-q4", &dir, None) });
        // a curl that would run on (a long sleep), and the worker's look at it, which takes it once it ends
        *d.child.lock().unwrap() = Some(Command::new("sleep").arg("30").spawn().unwrap());
        let downloads = Arc::clone(&studio);
        let worker = std::thread::spawn(move || loop {
            let mut slot = downloads.downloads.child.lock().unwrap();
            if let Some(Ok(Some(status))) = slot.as_mut().map(|c| c.try_wait()) {
                slot.take();
                return status;
            }
            drop(slot);
            std::thread::sleep(Duration::from_millis(20));
        });
        let started = Instant::now();
        d.shutdown();
        assert!(!worker.join().unwrap().success(), "killed");
        assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
        assert_eq!(d.list.lock().unwrap().iter().map(|x| x.status).collect::<Vec<_>>(), ["paused", "downloading"]);
        assert_eq!(d.wanted("moss-image-sft-q4"), Want::Pause, "the one under way stops at its next look");
        assert_eq!(d.start(&studio, "qwen3.5-9b", None, None).unwrap_err(), STOPPING);
        // the worker's turn after: nothing fetched, nothing added
        d.run(&studio, "moss-image-sft-q4");
        assert_eq!(d.list.lock().unwrap().iter().find(|x| x.id == "moss-image-sft-q4").map(|x| x.status), Some("paused"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A copy or a check stopped by a pause that was undone at once (Pause, then Resume) is not taken for the model
    /// done: it is not added to the models half-made, and has its turn again.
    #[test]
    fn a_step_stopped_by_a_pause_undone_at_once_is_not_the_model_done() {
        let d = Downloads::new();
        assert_eq!(d.stopped_as("x"), Want::Pause, "asked for again: paused, never done");
        d.want.lock().unwrap().insert("x".into(), Want::Cancel);
        assert_eq!(d.stopped_as("x"), Want::Cancel);
    }

    /// A partial file larger than its file is begun again, not carried on from its end (refused there every time).
    #[test]
    fn a_partial_file_larger_than_its_file_is_begun_again() {
        let root = std::env::temp_dir().join(format!("oaiy-dl-larger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("models");
        std::fs::create_dir_all(dir.join("big")).unwrap();
        let body: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let mut h = sha256::Sha256::new();
        h.update(&body);
        let (url, asked) = serve(body.clone());
        let studio = Arc::new(Studio::for_test(&root, crate::config::default_json()));
        let d = Downloads::new();
        d.list.lock().unwrap().push(Download { status: "downloading", ..Download::new("big", &dir, None) });
        let dest = dir.join("big/model.bin");
        std::fs::write(part_path(&dest), vec![9u8; body.len() + 10]).unwrap();
        let e = Json::parse(format!(r#"{{"id":"big","parts":[{{"url":"{url}","file":"model.bin","size":{},"sha256":"{}","into":"big"}}]}}"#, body.len(), h.hex()).as_bytes()).unwrap();
        assert_eq!(d.fetch(&studio, &e, &dir).unwrap(), Want::Run);
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A finished download is not cancelled (its files are a model's: removed under Models), listed or, after a
    /// restart, found whole in its folder; and the partial files a cancel deletes are never ones in a linked folder.
    #[test]
    fn a_finished_download_is_not_cancelled_and_a_linked_folder_is_not_looked_into() {
        let root = std::env::temp_dir().join(format!("oaiy-dl-finished-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("models");
        let folder = dir.join("Qwen3.5-9B-GGUF");
        std::fs::create_dir_all(&folder).unwrap();
        let gguf = folder.join("Qwen3.5-9B-Q4_K_M.gguf");
        std::fs::write(&gguf, b"weights").unwrap();
        std::fs::write(folder.join(".oaiy-qwen3.5-9b.json"), br#"{"id":"qwen3.5-9b","files":["Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"]}"#).unwrap();
        let studio = Studio::for_test(&root, crate::config::default_json());
        let d = Downloads::new();
        assert!(d.cancel(&studio, "qwen3.5-9b", Some(dir.clone())).unwrap_err().contains("downloaded"), "found whole after a restart");
        d.list.lock().unwrap().push(Download { status: "done", fetched: vec![gguf.clone()], ..Download::new("qwen3.5-9b", &dir, None) });
        assert!(d.cancel(&studio, "qwen3.5-9b", None).is_err(), "listed, done");
        assert!(gguf.exists());
        #[cfg(unix)]
        {
            let outside = root.join("elsewhere");
            std::fs::create_dir_all(&outside).unwrap();
            std::fs::write(outside.join("other.gguf.part"), b"half").unwrap();
            std::os::unix::fs::symlink(&outside, folder.join("linked")).unwrap();
            std::fs::write(folder.join("own.gguf.part"), b"half").unwrap();
            remove_partials("qwen3.5-9b", &dir, &Default::default());
            assert!(!folder.join("own.gguf.part").exists(), "its own goes");
            assert!(outside.join("other.gguf.part").exists(), "a linked folder's stays");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Across drives, a model that uses another's file is a link to it ([`library::place`]): a cancel of the first
    /// keeps the file the link leads to, wherever the other model's folder is.
    #[cfg(unix)]
    #[test]
    fn a_cancel_keeps_the_file_another_drives_model_links_to() {
        let root = std::env::temp_dir().join(format!("oaiy-dl-across-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (one, two) = (root.join("one"), root.join("two"));
        // MOSS-Image (its text encoder Qwen3.5-9B's GGUF) part-way in `one`
        let encoder = one.join("Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf");
        let own = one.join("MOSS-Image-GGUF/MOSS-Image-SFT-Q4_K_M.gguf");
        for f in [&encoder, &own] {
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, b"weights").unwrap();
        }
        // Qwen3.5-9B downloaded in `two` (where downloads go now), a link to that GGUF
        let linked = two.join("Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf");
        std::fs::create_dir_all(linked.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&encoder, &linked).unwrap();
        std::fs::write(two.join("Qwen3.5-9B-GGUF/.oaiy-qwen3.5-9b.json"), br#"{"id":"qwen3.5-9b","files":["Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"]}"#).unwrap();
        let mut cfg = crate::config::default_json();
        crate::util::set(crate::models::registry::obj_mut(&mut cfg, &["downloads"]).unwrap(), "dir", Json::str(two.to_string_lossy()));
        let studio = Studio::for_test(&root, cfg);
        let d = Downloads::new();
        let id = "moss-image-sft-q4";
        d.list.lock().unwrap().push(Download { status: "paused", fetched: vec![encoder.clone(), own.clone()], ..Download::new(id, &one, None) });
        d.cancel(&studio, id, None).unwrap();
        assert!(!own.exists(), "its own file goes");
        assert!(encoder.exists(), "the file another model's link leads to stays");
        assert_eq!(std::fs::read(&linked).unwrap(), b"weights");
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
            match studio.downloads.plan_of(&studio, id, None, None) {
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
        d.list.lock().unwrap().push(Download { status: "downloading", ..Download::new("t", &dir, None) });
        for f in &files {
            std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
            // Half a file first, as a pause leaves it; curl carries on from there.
            if let Some(s) = f.size {
                let whole = curl_text(&cfg, &f.url, None, None, false).unwrap();
                assert_eq!(whole.len() as u64, s);
                std::fs::write(part_path(&f.dest), &whole.as_bytes()[..(s / 2) as usize]).unwrap();
            }
            assert_eq!(d.curl(&cfg, "t", f, &part_path(&f.dest), 0, false).unwrap(), Want::Run);
            assert_eq!(Some(size_of(&part_path(&f.dest))), f.size);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
