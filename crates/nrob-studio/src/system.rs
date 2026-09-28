//! What the machine has: GPUs (through `nvidia-smi`), RAM, and a directory
//! listing for the UI's file picker.

use nrob::json::Json;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Probes spawn processes, so their answers are cached this long.
const GPU_TTL: Duration = Duration::from_secs(2);
const RAM_TTL: Duration = Duration::from_secs(5);

pub struct System {
    gpus: Mutex<Option<(Instant, Json)>>,
    ram: Mutex<Option<(Instant, Json)>>,
}

fn quiet(program: &str) -> Command {
    let mut c = Command::new(program);
    c.stdin(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

impl System {
    pub fn new() -> System {
        System { gpus: Mutex::new(None), ram: Mutex::new(None) }
    }

    fn cached(slot: &Mutex<Option<(Instant, Json)>>, ttl: Duration, probe: impl FnOnce() -> Json) -> Json {
        let mut g = slot.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((at, v)) = g.as_ref() {
            if at.elapsed() < ttl {
                return v.clone();
            }
        }
        let v = probe();
        *g = Some((Instant::now(), v.clone()));
        v
    }

    /// `[{index, name, memory_used_mb, memory_total_mb, utilization, temperature}]`,
    /// empty when there is no NVIDIA driver.
    pub fn gpus(&self) -> Json {
        Self::cached(&self.gpus, GPU_TTL, || {
            let out = quiet("nvidia-smi")
                .args(["--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
                .output();
            let Ok(out) = out else { return Json::Arr(Vec::new()) };
            Json::Arr(String::from_utf8_lossy(&out.stdout).lines().filter_map(parse_gpu).collect())
        })
    }

    /// `{total_mb, available_mb}` (nulls when the OS will not say).
    pub fn ram(&self) -> Json {
        Self::cached(&self.ram, RAM_TTL, || {
            let (total, available) = ram_mb().unwrap_or((0, 0));
            let n = |v: u64| if v == 0 { Json::Null } else { Json::Int(v as i64) };
            Json::obj([("total_mb", n(total)), ("available_mb", n(available))])
        })
    }
}

fn parse_gpu(line: &str) -> Option<Json> {
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    if f.len() < 6 {
        return None;
    }
    let n = |s: &str| s.parse::<i64>().map(Json::Int).unwrap_or(Json::Null);
    Some(Json::obj([
        ("index", n(f[0])),
        ("name", Json::str(f[1])),
        ("memory_used_mb", n(f[2])),
        ("memory_total_mb", n(f[3])),
        ("utilization", n(f[4])),
        ("temperature", n(f[5])),
    ]))
}

#[cfg(target_os = "linux")]
fn ram_mb() -> Option<(u64, u64)> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| {
        info.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok())
    };
    Some((field("MemTotal:")? / 1024, field("MemAvailable:")? / 1024))
}

#[cfg(windows)]
fn ram_mb() -> Option<(u64, u64)> {
    // std has no memory query; CIM answers in KiB.
    let out = quiet("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", "$o=Get-CimInstance Win32_OperatingSystem; \"$($o.TotalVisibleMemorySize) $($o.FreePhysicalMemory)\""])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.split_whitespace().filter_map(|v| v.parse::<u64>().ok());
    Some((it.next()? / 1024, it.next()? / 1024))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn ram_mb() -> Option<(u64, u64)> {
    None
}

/// Model-ish files the picker shows (folders are always shown).
const SHOWN: [&str; 6] = ["gguf", "safetensors", "json", "exe", "", "mp4"];

/// A directory listing for the file picker: folders first, then files that could
/// be models (or tokenizers, or ffmpeg). No path means the roots: drive letters
/// on Windows, `/` and home elsewhere.
pub fn browse(path: Option<&str>) -> Result<Json, String> {
    let Some(path) = path.map(str::trim).filter(|p| !p.is_empty()) else {
        return Ok(Json::obj([("path", Json::str("")), ("parent", Json::Null), ("entries", Json::Arr(roots()))]));
    };
    let dir = PathBuf::from(path);
    let read = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in read.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name.starts_with('$') {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            dirs.push((name.to_ascii_lowercase(), entry_json(&p, &name, true, 0)));
        } else {
            let ext = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let shown = SHOWN.contains(&ext.as_str()) && (ext != "exe" || name.to_ascii_lowercase().contains("ffmpeg")) && (!ext.is_empty() || name.contains("ffmpeg"));
            // NAF's weights, the one PyTorch file a model needs.
            if shown || name.eq_ignore_ascii_case(crate::detect::NAF_FILE) {
                files.push((name.to_ascii_lowercase(), entry_json(&p, &name, false, meta.len())));
            }
        }
    }
    dirs.sort_by(|a, b| a.0.cmp(&b.0));
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let parent = dir.parent().filter(|p| !p.as_os_str().is_empty()).map(|p| Json::str(p.to_string_lossy())).unwrap_or(Json::Str(String::new()));
    Ok(Json::obj([
        ("path", Json::str(dir.to_string_lossy())),
        ("parent", parent),
        ("entries", Json::Arr(dirs.into_iter().chain(files).map(|(_, e)| e).collect())),
    ]))
}

fn entry_json(p: &Path, name: &str, dir: bool, size: u64) -> Json {
    Json::obj([
        ("name", Json::str(name)),
        ("path", Json::str(p.to_string_lossy())),
        ("dir", Json::Bool(dir)),
        ("size", Json::Int(size as i64)),
    ])
}

fn roots() -> Vec<Json> {
    let mut out = Vec::new();
    #[cfg(windows)]
    for letter in b'A'..=b'Z' {
        let root = format!("{}:\\", letter as char);
        if Path::new(&root).exists() {
            out.push(entry_json(Path::new(&root), &root, true, 0));
        }
    }
    #[cfg(not(windows))]
    {
        out.push(entry_json(Path::new("/"), "/", true, 0));
        if let Some(home) = std::env::var_os("HOME") {
            out.push(entry_json(Path::new(&home), "~ (home)", true, 0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_lines_parse_and_listings_filter_to_models() {
        let g = parse_gpu("1, NVIDIA GeForce RTX 5090, 7456, 32607, 3, 41").unwrap();
        assert_eq!(g.get("memory_total_mb").and_then(Json::as_i64), Some(32607));
        assert_eq!(g.get("name").and_then(Json::as_str), Some("NVIDIA GeForce RTX 5090"));
        assert!(parse_gpu("garbage").is_none());
        let d = std::env::temp_dir().join(format!("nrob-studio-browse-{}", std::process::id()));
        std::fs::create_dir_all(d.join("sub")).unwrap();
        for f in ["a.gguf", "notes.txt", "b.safetensors", "naf_release.pth", "other.pth"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let listing = browse(Some(d.to_str().unwrap())).unwrap();
        let names: Vec<_> = listing.get("entries").unwrap().as_array().unwrap().iter().map(|e| e.get("name").unwrap().as_str().unwrap().to_string()).collect();
        assert_eq!(names, ["sub", "a.gguf", "b.safetensors", "naf_release.pth"]);
        assert!(browse(Some(d.join("missing").to_str().unwrap())).is_err());
        std::fs::remove_dir_all(d).unwrap();
        assert!(!browse(None).unwrap().get("entries").unwrap().as_array().unwrap().is_empty());
    }
}
