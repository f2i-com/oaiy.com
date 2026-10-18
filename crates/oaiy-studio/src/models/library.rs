//! The model files already on this computer, so none is downloaded twice: a model's text encoder or VAE that another
//! model came with, or one the person keeps on another drive, is used where it is (linked into the new model's
//! folder) instead of fetched again.
//!
//! Files are looked for where downloads go, in every folder the configuration names a model or a part in, and in
//! `downloads.library` (more folders of the person's). A file is the one wanted when it is as large and its SHA-256 is
//! the one Hugging Face publishes for it; a file's SHA-256, once read, is kept beside the configuration
//! (`model-hashes.json`, by path, size and time changed), so it is read once.

use super::sha256;
use crate::config;
use crate::util::str_or;
use oaiy_engine::json::Json;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, UNIX_EPOCH};

/// Files smaller than this are fetched again rather than looked for (a tokenizer, a configuration): quick to
/// download, where a model's weights are far larger.
pub const MIN_BYTES: u64 = 8 << 20;
/// How deep a folder is looked through, how many model files are kept at most, how many entries are looked at in
/// all, and for how long: a folder named for a model that holds far more than models (a Documents folder) is looked
/// through partly, not for minutes.
const MAX_DEPTH: usize = 6;
const MAX_FILES: usize = 20_000;
const MAX_VISITS: usize = 100_000;
const MAX_TIME: Duration = Duration::from_secs(5);
/// Folders never looked into: a program's build or packages, not models.
const SKIPPED: [&str; 7] = ["node_modules", "target", "__pycache__", "site-packages", "venv", "Caches", "Trash"];
/// How long a look through is used again (a plan, then its download): files added since are not missed for longer.
const SCAN_REUSED: Duration = Duration::from_secs(15);
const HASHES_FILE: &str = "model-hashes.json";

/// A model file found here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    pub size: u64,
}

/// The person's home folder.
fn home() -> Option<PathBuf> {
    ["HOME", "USERPROFILE"].iter().find_map(|k| std::env::var_os(k)).map(PathBuf::from).filter(|p| p.is_absolute())
}

/// A folder too wide to look through for a model's sake: a drive (`C:\`, `/Volumes/T9`, `/mnt/x`), the filesystem's
/// root, the person's home folder or a folder above it. A model's own folder is narrower, wherever it is (`D:\Models`).
fn too_wide(p: &Path) -> bool {
    let normal = p.components().filter(|c| matches!(c, Component::Normal(_))).count();
    if normal == 0 || home().is_some_and(|h| h.starts_with(p)) {
        return true;
    }
    p.parent().is_some_and(|q| q == Path::new("/Volumes") || q == Path::new("/mnt")) || (p.starts_with("/media") && normal <= 3)
}

/// `~` at the start of a folder the person wrote, as their home folder.
fn expand_home(s: &str) -> String {
    match (s.strip_prefix('~'), home()) {
        (Some(rest), Some(h)) if rest.is_empty() || rest.starts_with(['/', '\\']) => format!("{}{rest}", h.display()),
        _ => s.to_string(),
    }
}

/// The folders model files are looked for in: `downloads`, the folder of each model and part the configuration names
/// (its `llm` and `media` models, never a whole drive or home folder: [`too_wide`]), and `downloads.library` (the
/// person's own, as they wrote them: `~` their home folder). One inside another is looked through once.
pub fn roots(studio_root: &Path, cfg: &Json, downloads: &Path) -> Vec<PathBuf> {
    let mut found = vec![downloads.to_path_buf()];
    let mut named = Vec::new();
    let mut strings = |v: &Json| collect_strings(v, &mut named);
    if let Some(llm) = cfg.get("llm").and_then(|l| l.get("models")) {
        strings(llm);
    }
    if let Some(media) = cfg.get("media").and_then(Json::as_object) {
        for (_, section) in media {
            if let Some(models) = section.get("models") {
                strings(models);
            }
        }
    }
    for s in named {
        let p = config::resolve(studio_root, &s);
        // (a model's folder, or the folder of a model's file: never a drive or a home folder named for another reason)
        let folder = if p.is_file() { p.parent().map(Path::to_path_buf) } else if p.is_dir() { Some(p) } else { None };
        if let Some(f) = folder.filter(|f| !too_wide(f)) {
            found.push(f);
        }
    }
    // (the person's own: looked through as they named them, a whole drive too, within the limits of a look)
    for extra in cfg.get("downloads").and_then(|d| d.get("library")).and_then(Json::as_array).unwrap_or(&[]) {
        if let Some(s) = extra.as_str().map(str::trim).filter(|s| !s.is_empty()) {
            let p = config::resolve(studio_root, &expand_home(s));
            if p.is_dir() {
                found.push(p);
            }
        }
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    found.sort_by_key(|p| p.components().count());
    for f in found {
        if !roots.iter().any(|r| f.starts_with(r)) {
            roots.push(f);
        }
    }
    roots
}

fn collect_strings(v: &Json, out: &mut Vec<String>) {
    match v {
        Json::Str(s) if s.contains('/') || s.contains('\\') => out.push(s.clone()),
        Json::Arr(items) => items.iter().for_each(|i| collect_strings(i, out)),
        Json::Obj(fields) => fields.iter().for_each(|(_, f)| collect_strings(f, out)),
        _ => {}
    }
}

/// The model files in `roots` ([`MIN_BYTES`] or larger; no partial downloads, hidden files, linked folders, or a
/// program's build and packages), within the limits of a look ([`MAX_VISITS`] entries, [`MAX_TIME`]); the same roots
/// looked through again within [`SCAN_REUSED`] answer what was found then.
pub fn scan(roots: &[PathBuf]) -> Vec<Found> {
    struct Look {
        out: Vec<Found>,
        visits: usize,
        until: Instant,
    }
    fn walk(dir: &Path, depth: usize, look: &mut Look) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            look.visits += 1;
            if look.out.len() >= MAX_FILES || look.visits >= MAX_VISITS || (look.visits % 256 == 0 && Instant::now() > look.until) {
                return;
            }
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_dir() {
                if depth > 0 && !SKIPPED.contains(&name.as_str()) {
                    walk(&p, depth - 1, look);
                }
            } else if p.extension().is_none_or(|x| x != "part") {
                // (a file, or a link to one: its size is the file's)
                if let Some(m) = std::fs::metadata(&p).ok().filter(|m| m.is_file() && m.len() >= MIN_BYTES) {
                    look.out.push(Found { path: p, size: m.len() });
                }
            }
        }
    }
    if let Some((r, at, found)) = LAST.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        if r == roots && at.elapsed() < SCAN_REUSED {
            return found.clone();
        }
    }
    let mut look = Look { out: Vec::new(), visits: 0, until: Instant::now() + MAX_TIME };
    for r in roots {
        walk(r, MAX_DEPTH, &mut look);
    }
    *LAST.lock().unwrap_or_else(|p| p.into_inner()) = Some((roots.to_vec(), Instant::now(), look.out.clone()));
    look.out
}

/// The last look through ([`scan`]): its roots, when, and what it found.
static LAST: Mutex<Option<(Vec<PathBuf>, Instant, Vec<Found>)>> = Mutex::new(None);

/// What a look through found is stale once a file is placed or removed: the next look reads again.
pub fn forget_scan() {
    *LAST.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// What is known of files' SHA-256s: by path, the size and time changed it was read at.
struct Hashes {
    file: PathBuf,
    known: HashMap<String, (u64, u128, String)>,
}

static HASHES: Mutex<Option<Hashes>> = Mutex::new(None);

fn changed(m: &std::fs::Metadata) -> u128 {
    m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_millis())
}

fn with_hashes<T>(studio_root: &Path, f: impl FnOnce(&mut Hashes) -> T) -> T {
    let mut guard = HASHES.lock().unwrap_or_else(|p| p.into_inner());
    let file = studio_root.join(HASHES_FILE);
    if guard.as_ref().is_none_or(|h| h.file != file) {
        let known = read_hashes(&file);
        *guard = Some(Hashes { file, known });
    }
    f(guard.as_mut().expect("loaded"))
}

/// The SHA-256s a hashes file keeps.
fn read_hashes(file: &Path) -> HashMap<String, (u64, u128, String)> {
    let mut known = HashMap::new();
    if let Some(Json::Obj(entries)) = std::fs::read(file).ok().and_then(|b| Json::parse(&b).ok()) {
        for (path, e) in entries {
            let size = e.get("size").and_then(Json::as_i64).unwrap_or(-1);
            let at = e.get("changed").and_then(Json::as_i64).unwrap_or(-1);
            let sha = str_or(&e, "sha256", "");
            if size >= 0 && at >= 0 && sha.len() == 64 {
                known.insert(path, (size as u64, at as u128, sha.to_string()));
            }
        }
    }
    known
}

impl Hashes {
    /// Written whole beside itself, then put in its place (a crash part-way leaves the one before, not half of
    /// one), and with what another studio using the same folder wrote meanwhile kept beside this one's.
    fn save(&mut self) {
        for (p, v) in read_hashes(&self.file) {
            self.known.entry(p).or_insert(v);
        }
        let entries: Vec<(String, Json)> = self
            .known
            .iter()
            .map(|(p, (size, at, sha))| (p.clone(), Json::obj([("size", Json::Int(*size as i64)), ("changed", Json::Int(*at as i64)), ("sha256", Json::str(sha))])))
            .collect();
        let beside = self.file.with_extension(format!("json.{}.tmp", std::process::id()));
        if std::fs::write(&beside, Json::Obj(entries).to_json()).is_ok() && std::fs::rename(&beside, &self.file).is_err() {
            let _ = std::fs::remove_file(&beside);
        }
    }
}

/// The SHA-256 of `path` known already (read before, the file unchanged since), without reading it.
pub fn known(studio_root: &Path, path: &Path) -> Option<String> {
    let m = std::fs::metadata(path).ok()?;
    with_hashes(studio_root, |h| h.known.get(&path.to_string_lossy().into_owned()).filter(|(s, at, _)| *s == m.len() && *at == changed(&m)).map(|(_, _, sha)| sha.clone()))
}

/// Keep `sha` as the SHA-256 of `path` (as it is now).
pub fn remember(studio_root: &Path, path: &Path, sha: &str) {
    let Ok(m) = std::fs::metadata(path) else { return };
    with_hashes(studio_root, |h| {
        h.known.insert(path.to_string_lossy().into_owned(), (m.len(), changed(&m), sha.to_string()));
        h.save();
    });
}

/// The SHA-256 of `path`: known already, or read now (`stop` asked as it is read; `true` gives up) and kept.
pub fn hash(studio_root: &Path, path: &Path, stop: impl FnMut(u64) -> bool) -> Option<String> {
    if let Some(sha) = known(studio_root, path) {
        return Some(sha);
    }
    let sha = sha256::file(path, stop).ok()??;
    remember(studio_root, path, &sha);
    Some(sha)
}

/// A file among `files` that is the one wanted: `size` bytes, and its SHA-256 `sha` (read where it is not known yet,
/// for the files of that size alone). `stop` as for [`hash`].
pub fn find(studio_root: &Path, files: &[Found], size: u64, sha: &str, mut stop: impl FnMut(u64) -> bool) -> Option<PathBuf> {
    let sha = sha.to_ascii_lowercase();
    for f in files.iter().filter(|f| f.size == size) {
        if hash(studio_root, &f.path, &mut stop).is_some_and(|h| h == sha) {
            return Some(f.path.clone());
        }
    }
    None
}

/// A file among `files` that may be the one wanted: as large, its SHA-256 the one wanted where it is known already,
/// else not read yet (`checked` false: read when the model is got). For saying what a model needs, quickly. Only a
/// file Hugging Face publishes a SHA-256 for is looked for: one without is downloaded (and a file as large is not it).
pub fn likely(studio_root: &Path, files: &[Found], size: u64, sha: &str) -> Option<(PathBuf, bool)> {
    let mut unchecked = None;
    for f in files.iter().filter(|f| f.size == size) {
        match known(studio_root, &f.path) {
            Some(h) if h.eq_ignore_ascii_case(sha) => return Some((f.path.clone(), true)),
            Some(_) => {}
            None => {
                if unchecked.is_none() {
                    unchecked = Some((f.path.clone(), false));
                }
            }
        }
    }
    unchecked
}

/// How a file found here is put where a model wants it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placed {
    /// The same file under a second name (one drive): no more room taken.
    Linked,
    /// A link to the file where it is (another drive): the model needs that drive there.
    Pointed,
    /// Copied (where neither can be made).
    Copied,
}

/// Whether placing the file at `from` where `to` names would copy it (taking its room again): on Windows, a file on
/// another drive (a hard link stays on one drive, and a link to a file needs Developer Mode); elsewhere a link is made.
pub fn needs_copy(from: &Path, to: &Path) -> bool {
    #[cfg(windows)]
    {
        let drive = |p: &Path| match std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).components().next() {
            Some(Component::Prefix(x)) => x.as_os_str().to_string_lossy().to_ascii_lowercase(),
            _ => String::new(),
        };
        let at = to.ancestors().find(|a| a.exists()).unwrap_or(to);
        drive(from) != drive(at)
    }
    #[cfg(not(windows))]
    {
        let _ = (from, to);
        false
    }
}

/// Put the file at `from` where `to` names, taking no more room where the system lets it. A copy (where no link can
/// be made) is made in pieces: `progress` told the bytes copied, `stop` asked between them (`None`: stopped, nothing
/// left behind).
pub fn place(from: &Path, to: &Path, progress: &mut dyn FnMut(u64), stop: &dyn Fn() -> bool) -> Result<Option<Placed>, String> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    if std::fs::symlink_metadata(to).is_ok() {
        std::fs::remove_file(to).map_err(|e| format!("{}: {e}", to.display()))?;
    }
    let from = std::fs::canonicalize(from).unwrap_or_else(|_| from.to_path_buf());
    if std::fs::hard_link(&from, to).is_ok() {
        return Ok(Some(Placed::Linked));
    }
    #[cfg(unix)]
    if std::os::unix::fs::symlink(&from, to).is_ok() {
        return Ok(Some(Placed::Pointed));
    }
    #[cfg(windows)]
    if std::os::windows::fs::symlink_file(&from, to).is_ok() {
        return Ok(Some(Placed::Pointed));
    }
    copy_in_pieces(&from, to, progress, stop).map(|done| done.then_some(Placed::Copied))
}

/// Copy `from` to `to` in pieces, beside it first (`.copying`) and put in place whole: `progress` told the bytes
/// copied, `stop` asked between pieces (`false`: stopped, nothing left behind).
fn copy_in_pieces(from: &Path, to: &Path, progress: &mut dyn FnMut(u64), stop: &dyn Fn() -> bool) -> Result<bool, String> {
    let copying = to.with_extension(format!("{}.copying", to.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()));
    let copied = (|| -> std::io::Result<bool> {
        use std::io::{Read, Write};
        let mut src = std::fs::File::open(from)?;
        let mut dst = std::fs::File::create(&copying)?;
        let mut buf = vec![0u8; 4 << 20];
        let mut n = 0u64;
        loop {
            let got = src.read(&mut buf)?;
            if got == 0 {
                break;
            }
            dst.write_all(&buf[..got])?;
            n += got as u64;
            progress(n);
            if stop() {
                return Ok(false);
            }
        }
        dst.flush()?;
        Ok(true)
    })();
    match copied {
        Ok(true) => std::fs::rename(&copying, to).map(|_| true).map_err(|e| format!("{}: {e}", to.display())),
        Ok(false) => {
            let _ = std::fs::remove_file(&copying);
            Ok(false)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&copying);
            Err(format!("{}: {e}", to.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_kept_elsewhere_is_found_by_its_size_and_sha256_and_linked_into_place() {
        let base = std::env::temp_dir().join(format!("oaiy-library-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (studio, downloads, kept) = (base.join("studio"), base.join("studio/models"), base.join("drive/flux-3/text_encoder"));
        for d in [&studio, &downloads, &kept] {
            std::fs::create_dir_all(d).unwrap();
        }
        let encoder: Vec<u8> = (0..MIN_BYTES as usize + 7).map(|i| (i % 251) as u8).collect();
        std::fs::write(kept.join("model.safetensors"), &encoder).unwrap();
        // another file as large, not the same
        let mut other = encoder.clone();
        other[0] ^= 1;
        std::fs::create_dir_all(downloads.join("old")).unwrap();
        std::fs::write(downloads.join("old/other.safetensors"), &other).unwrap();
        std::fs::write(downloads.join("old/small.json"), b"{}").unwrap();
        let mut h = sha256::Sha256::new();
        h.update(&encoder);
        let sha = h.hex();
        // the folder the configuration names a model's part in is looked through
        let cfg = Json::parse(format!(r#"{{"media":{{"image":{{"models":{{"flux":{{"text_encoder":"{}"}}}}}}}}}}"#, kept.display()).as_bytes()).unwrap();
        let roots = roots(&studio, &cfg, &downloads);
        assert_eq!(roots, vec![downloads.clone(), kept.clone()]);
        let files = scan(&roots);
        assert_eq!(files.len(), 2, "large files alone: {files:?}");
        // not read yet: as large, unchecked; then found by its SHA-256, the other as large passed over
        assert_eq!(likely(&studio, &files, encoder.len() as u64, &sha).map(|(_, c)| c), Some(false));
        let found = find(&studio, &files, encoder.len() as u64, &sha.to_uppercase(), |_| false).unwrap();
        assert_eq!(found, kept.join("model.safetensors"));
        assert_eq!(likely(&studio, &files, encoder.len() as u64, &sha), Some((found.clone(), true)), "known once read");
        assert!(studio.join(HASHES_FILE).is_file());
        assert!(find(&studio, &files, encoder.len() as u64, &"0".repeat(64), |_| false).is_none());
        // put where the new model wants it: the same file, no copy
        let to = downloads.join("flux-3-image-klein/text_encoder/model.safetensors");
        let placed = place(&found, &to, &mut |_| {}, &|| false).unwrap();
        assert!(placed.is_some_and(|p| p != Placed::Copied));
        assert_eq!(std::fs::read(&to).unwrap(), encoder);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A whole drive, the filesystem's root and the home folder (or above it) are not looked through for a model's
    /// sake; a model's own folder is, on any drive; `~` is the home folder in a folder the person wrote.
    #[test]
    fn a_drive_or_a_home_folder_is_too_wide_a_models_folder_is_not() {
        let home = home().unwrap();
        assert!(too_wide(Path::new("/")) && too_wide(&home) && too_wide(home.parent().unwrap()));
        assert!(too_wide(Path::new("/Volumes/T9")) && too_wide(Path::new("/mnt/models")));
        assert!(!too_wide(Path::new("/Volumes/T9/flux-3")) && !too_wide(&home.join("models")));
        assert_eq!(expand_home("~/models"), format!("{}/models", home.display()));
        assert_eq!(expand_home("/opt/~x"), "/opt/~x");
    }

    /// A copy (where no link can be made) goes in pieces, says how far it is, stops when asked, and leaves nothing
    /// half-made behind.
    #[test]
    fn a_copy_goes_in_pieces_and_stops_when_asked() {
        let base = std::env::temp_dir().join(format!("oaiy-library-copy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let (from, to) = (base.join("a.safetensors"), base.join("b.safetensors"));
        std::fs::write(&from, vec![7u8; 9 << 20]).unwrap();
        let mut seen = 0;
        assert_eq!(copy_in_pieces(&from, &to, &mut |n| seen = n, &|| false), Ok(true));
        assert_eq!((seen, std::fs::read(&to).unwrap().len()), (9 << 20, 9 << 20));
        std::fs::remove_file(&to).unwrap();
        assert_eq!(copy_in_pieces(&from, &to, &mut |_| {}, &|| true), Ok(false));
        assert!(!to.exists() && std::fs::read_dir(&base).unwrap().count() == 1, "nothing left behind");
        let _ = std::fs::remove_dir_all(&base);
    }
}
