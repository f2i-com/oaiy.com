//! What the machine has: GPUs (through `nvidia-smi`; without an NVIDIA driver,
//! the display adapters the OS knows, AMD and Intel among them), RAM, and a
//! directory listing for the UI's file picker.

use oaiy_engine::json::Json;
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

    /// `[{index, name, memory_used_mb, memory_total_mb, utilization, temperature}]`
    /// from `nvidia-smi`. Without an NVIDIA driver, the display adapters the OS
    /// knows (an AMD or Intel GPU, which the portable engine runs on through
    /// WebGPU), with their `vendor` and total memory only: `index` is then their
    /// place in that list, not a CUDA device. Empty when neither says.
    pub fn gpus(&self) -> Json {
        Self::cached(&self.gpus, GPU_TTL, || {
            let out = quiet("nvidia-smi")
                .args(["--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
                .output();
            let nvidia: Vec<Json> = out.map(|o| String::from_utf8_lossy(&o.stdout).lines().filter_map(parse_gpu).collect()).unwrap_or_default();
            if !nvidia.is_empty() {
                return Json::Arr(nvidia);
            }
            // Adapters do not come and go: asked once.
            static OTHERS: std::sync::OnceLock<Vec<Json>> = std::sync::OnceLock::new();
            Json::Arr(OTHERS.get_or_init(other_gpus).clone())
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

/// A display adapter the OS knows, in the shape `nvidia-smi`'s give.
fn adapter(index: usize, name: &str, vendor: &str, total_mb: Option<u64>) -> Json {
    Json::obj([
        ("index", Json::Int(index as i64)),
        ("name", Json::str(name)),
        ("vendor", Json::str(vendor)),
        ("memory_used_mb", Json::Null),
        ("memory_total_mb", total_mb.map_or(Json::Null, |m| Json::Int(m as i64))),
        ("utilization", Json::Null),
        ("temperature", Json::Null),
    ])
}

/// A PCI vendor id's name ("1002" is AMD's).
fn vendor_of(id: &str) -> &'static str {
    match id.trim_start_matches("0x").to_ascii_lowercase().as_str() {
        "1002" | "1022" => "amd",
        "8086" => "intel",
        "10de" => "nvidia",
        _ => "other",
    }
}

/// Windows: the display class's adapters, `DriverDesc|HardwareInformation.qwMemorySize|MatchingDeviceId` a line, as
/// [`other_gpus`] reads them. The 64-bit memory size is the dedicated memory every vendor's driver writes (WMI's
/// `AdapterRAM` stops at 4 GB). Remote and basic display adapters, which have none, are left out.
#[cfg_attr(not(windows), allow(dead_code))]
fn parse_windows_adapters(text: &str) -> Vec<Json> {
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.trim().split('|').collect();
        let (Some(name), Some(bytes), Some(id)) = (f.first(), f.get(1), f.get(2)) else { continue };
        let Ok(bytes) = bytes.trim().parse::<u64>() else { continue };
        if bytes == 0 || name.trim().is_empty() || name.starts_with("Microsoft") {
            continue;
        }
        let vendor = id.to_ascii_lowercase().split("ven_").nth(1).map(|v| vendor_of(&v[..v.len().min(4)])).unwrap_or("other");
        out.push(adapter(out.len(), name.trim(), vendor, Some(bytes / (1024 * 1024))));
    }
    out
}

#[cfg(windows)]
fn other_gpus() -> Vec<Json> {
    let script = "Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')|$($_.MatchingDeviceId)\" }";
    let Ok(out) = quiet("powershell").args(["-NoProfile", "-NonInteractive", "-Command", script]).output() else { return Vec::new() };
    parse_windows_adapters(&String::from_utf8_lossy(&out.stdout))
}

/// Linux: the DRM cards' PCI vendors, and an AMD card's VRAM (amdgpu's `mem_info_vram_total`). Other vendors' memory
/// is not in sysfs in a form every driver writes, so it stays unknown.
#[cfg(target_os = "linux")]
fn other_gpus() -> Vec<Json> {
    let Ok(dir) = std::fs::read_dir("/sys/class/drm") else { return Vec::new() };
    let mut cards: Vec<PathBuf> = dir
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit())))
        .collect();
    cards.sort();
    let mut out = Vec::new();
    for card in cards {
        let read = |f: &str| std::fs::read_to_string(card.join("device").join(f)).ok().map(|s| s.trim().to_string());
        let Some(vendor) = read("vendor").map(|v| vendor_of(&v)) else { continue };
        let total = read("mem_info_vram_total").and_then(|v| v.parse::<u64>().ok()).filter(|b| *b > 0).map(|b| b / (1024 * 1024));
        let name = read("product_name").filter(|n| !n.is_empty()).unwrap_or_else(|| match vendor {
            "amd" => "AMD GPU".to_string(),
            "intel" => "Intel GPU".to_string(),
            "nvidia" => "NVIDIA GPU".to_string(),
            _ => "GPU".to_string(),
        });
        out.push(adapter(out.len(), &name, vendor, total));
    }
    out
}

#[cfg(not(any(windows, target_os = "linux")))]
fn other_gpus() -> Vec<Json> {
    Vec::new()
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
            // NAF's and Real-ESRGAN's weights, the PyTorch files a model needs.
            if shown || crate::detect::readable_pth(&name) {
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

    /// The adapters this computer's OS lists, read for real (`--ignored --nocapture` to see them): the reader that a
    /// computer without an NVIDIA driver relies on, run where it can be checked.
    #[test]
    #[ignore = "reads this computer's display adapters"]
    fn print_the_display_adapters_the_os_lists() {
        let gpus = other_gpus();
        for g in &gpus {
            eprintln!("{} | {} | {:?} MB", g.get("name").and_then(Json::as_str).unwrap_or(""), g.get("vendor").and_then(Json::as_str).unwrap_or(""), g.get("memory_total_mb").and_then(Json::as_i64));
        }
        if cfg!(windows) {
            assert!(!gpus.is_empty(), "a Windows computer lists at least one display adapter with memory");
        }
    }

    #[test]
    fn without_an_nvidia_driver_the_display_adapters_are_read_with_their_vendor_and_memory() {
        // As this computer's registry answers: the Ryzen's Radeon, two RTX 5090s, and remote adapters with no memory.
        let text = "AMD Radeon(TM) Graphics|2147483648|PCI\\VEN_1002&DEV_13C0&SUBSYS_88771043&REV_C9\r\n\
                    NVIDIA GeForce RTX 5090|34190917632|pci\\ven_10de&dev_2b85\r\n\
                    Intel(R) Arc(TM) A770 Graphics|17079205888|PCI\\VEN_8086&DEV_56A0\r\n\
                    Microsoft Remote Display Adapter||RdpIdd_IndirectDisplay\r\n\
                    Microsoft Basic Display Adapter|0|ROOT\\BasicDisplay\r\n";
        let gpus = parse_windows_adapters(text);
        let got: Vec<(i64, &str, &str, Option<i64>)> = gpus
            .iter()
            .map(|g| (g.get("index").and_then(Json::as_i64).unwrap(), g.get("name").and_then(Json::as_str).unwrap(), g.get("vendor").and_then(Json::as_str).unwrap(), g.get("memory_total_mb").and_then(Json::as_i64)))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, "AMD Radeon(TM) Graphics", "amd", Some(2048)),
                (1, "NVIDIA GeForce RTX 5090", "nvidia", Some(32607)),
                (2, "Intel(R) Arc(TM) A770 Graphics", "intel", Some(16288)),
            ]
        );
        // What nvidia-smi gives and these leave unknown.
        assert_eq!(gpus[0].get("utilization"), Some(&Json::Null));
        assert!(parse_windows_adapters("").is_empty());
        assert_eq!(vendor_of("0x1002"), "amd");
        assert_eq!(vendor_of("0x8086"), "intel");
    }

    #[test]
    fn nvidia_smi_lines_parse_and_listings_filter_to_models() {
        let g = parse_gpu("1, NVIDIA GeForce RTX 5090, 7456, 32607, 3, 41").unwrap();
        assert_eq!(g.get("memory_total_mb").and_then(Json::as_i64), Some(32607));
        assert_eq!(g.get("name").and_then(Json::as_str), Some("NVIDIA GeForce RTX 5090"));
        assert!(parse_gpu("garbage").is_none());
        let d = std::env::temp_dir().join(format!("oaiy-studio-browse-{}", std::process::id()));
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
