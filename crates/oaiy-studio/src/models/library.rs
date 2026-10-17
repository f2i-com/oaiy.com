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
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

/// Files smaller than this are fetched again rather than looked for (a tokenizer, a configuration): quick to
/// download, where a model's weights are far larger.
pub const MIN_BYTES: u64 = 8 << 20;
/// How deep a folder is looked through, and how many files at most.
const MAX_DEPTH: usize = 6;
const MAX_FILES: usize = 20_000;
const HASHES_FILE: &str = "model-hashes.json";

/// A model file found here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    pub size: u64,
}

/// The folders model files are looked for in: `downloads`, the folder of each model and part the configuration names
/// (its `llm` and `media` models), and `downloads.library`. One inside another is looked through once.
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
    for extra in cfg.get("downloads").and_then(|d| d.get("library")).and_then(Json::as_array).unwrap_or(&[]) {
        if let Some(s) = extra.as_str().map(str::trim).filter(|s| !s.is_empty()) {
            named.push(s.to_string());
        }
    }
    for s in named {
        let p = config::resolve(studio_root, &s);
        // (a model's folder, or the folder of a model's file: never a drive or a home folder named for another reason)
        let folder = if p.is_file() { p.parent().map(Path::to_path_buf) } else if p.is_dir() { Some(p) } else { None };
        if let Some(f) = folder.filter(|f| f.components().count() > 3) {
            found.push(f);
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

/// The model files in `roots` ([`MIN_BYTES`] or larger; no partial downloads, hidden files or linked folders).
pub fn scan(roots: &[PathBuf]) -> Vec<Found> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<Found>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            if out.len() >= MAX_FILES {
                return;
            }
            let p = e.path();
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_dir() {
                if depth > 0 {
                    walk(&p, depth - 1, out);
                }
            } else if p.extension().is_none_or(|x| x != "part") {
                // (a file, or a link to one: its size is the file's)
                if let Some(m) = std::fs::metadata(&p).ok().filter(|m| m.is_file() && m.len() >= MIN_BYTES) {
                    out.push(Found { path: p, size: m.len() });
                }
            }
        }
    }
    let mut out = Vec::new();
    for r in roots {
        walk(r, MAX_DEPTH, &mut out);
    }
    out
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
        let mut known = HashMap::new();
        if let Some(Json::Obj(entries)) = std::fs::read(&file).ok().and_then(|b| Json::parse(&b).ok()) {
            for (path, e) in entries {
                let size = e.get("size").and_then(Json::as_i64).unwrap_or(-1);
                let at = e.get("changed").and_then(Json::as_i64).unwrap_or(-1);
                let sha = str_or(&e, "sha256", "");
                if size >= 0 && at >= 0 && sha.len() == 64 {
                    known.insert(path, (size as u64, at as u128, sha.to_string()));
                }
            }
        }
        *guard = Some(Hashes { file, known });
    }
    f(guard.as_mut().expect("loaded"))
}

impl Hashes {
    fn save(&self) {
        let entries: Vec<(String, Json)> = self
            .known
            .iter()
            .map(|(p, (size, at, sha))| (p.clone(), Json::obj([("size", Json::Int(*size as i64)), ("changed", Json::Int(*at as i64)), ("sha256", Json::str(sha))])))
            .collect();
        let _ = std::fs::write(&self.file, Json::Obj(entries).to_json());
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
/// else not read yet (`checked` false: read when the model is got). For saying what a model needs, quickly.
pub fn likely(studio_root: &Path, files: &[Found], size: u64, sha: Option<&str>) -> Option<(PathBuf, bool)> {
    let mut unchecked = None;
    for f in files.iter().filter(|f| f.size == size) {
        match (known(studio_root, &f.path), sha) {
            (Some(h), Some(want)) if h.eq_ignore_ascii_case(want) => return Some((f.path.clone(), true)),
            (Some(_), Some(_)) => {}
            _ => {
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

/// Put the file at `from` where `to` names, taking no more room where the system lets it.
pub fn place(from: &Path, to: &Path) -> Result<Placed, String> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    if std::fs::symlink_metadata(to).is_ok() {
        std::fs::remove_file(to).map_err(|e| format!("{}: {e}", to.display()))?;
    }
    let from = std::fs::canonicalize(from).unwrap_or_else(|_| from.to_path_buf());
    if std::fs::hard_link(&from, to).is_ok() {
        return Ok(Placed::Linked);
    }
    #[cfg(unix)]
    if std::os::unix::fs::symlink(&from, to).is_ok() {
        return Ok(Placed::Pointed);
    }
    #[cfg(windows)]
    if std::os::windows::fs::symlink_file(&from, to).is_ok() {
        return Ok(Placed::Pointed);
    }
    std::fs::copy(&from, to).map(|_| Placed::Copied).map_err(|e| format!("{}: {e}", to.display()))
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
        assert_eq!(likely(&studio, &files, encoder.len() as u64, Some(&sha)).map(|(_, c)| c), Some(false));
        let found = find(&studio, &files, encoder.len() as u64, &sha.to_uppercase(), |_| false).unwrap();
        assert_eq!(found, kept.join("model.safetensors"));
        assert_eq!(likely(&studio, &files, encoder.len() as u64, Some(&sha)), Some((found.clone(), true)), "known once read");
        assert!(studio.join(HASHES_FILE).is_file());
        assert!(find(&studio, &files, encoder.len() as u64, &"0".repeat(64), |_| false).is_none());
        // put where the new model wants it: the same file, no copy
        let to = downloads.join("flux-3-image-klein/text_encoder/model.safetensors");
        let placed = place(&found, &to).unwrap();
        assert_ne!(placed, Placed::Copied);
        assert_eq!(std::fs::read(&to).unwrap(), encoder);
        let _ = std::fs::remove_dir_all(&base);
    }
}
