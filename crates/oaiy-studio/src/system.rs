//! What the machine has: GPUs (through `nvidia-smi`; without an NVIDIA driver,
//! the display adapters the OS knows, AMD and Intel among them, and an
//! Apple-silicon Mac's own, whose memory is the computer's, each with its load,
//! memory in use and temperature where the OS says them: `oaiy_gpu_stats`), RAM,
//! and a directory listing for the UI's file picker.

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
    let mut c = oaiy_host_env::command(program);
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
    /// knows (an AMD or Intel GPU, which the engine runs on through WebGPU), with
    /// their `vendor`, `pci` ids and total memory: `index` is then their place in
    /// that list, not nvidia-smi's. Each one's load, the memory it has in use and
    /// its temperature are read as it is now, where the OS says them
    /// (`oaiy_gpu_stats`: macOS, Linux and Windows). A Mac's GPU also says
    /// `unified`: its memory is the computer's. Empty when neither says.
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
            let mut gpus = OTHERS.get_or_init(other_gpus).clone();
            fill(&mut gpus, &oaiy_gpu_stats::now());
            Json::Arr(gpus)
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

/// A display adapter the OS knows, in the shape `nvidia-smi`'s give, with its PCI vendor and device ids
/// (`"1002:744c"`; null for an Apple-silicon Mac's GPU, which is on no PCI bus).
fn adapter(index: usize, name: &str, vendor: &str, pci: Option<(u32, u32)>, total_mb: Option<u64>) -> Json {
    Json::obj([
        ("index", Json::Int(index as i64)),
        ("name", Json::str(name)),
        ("vendor", Json::str(vendor)),
        ("pci", pci.map_or(Json::Null, |(v, d)| Json::str(format!("{v:04x}:{d:04x}")))),
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
        let id = id.to_ascii_lowercase();
        let field = |key: &str| id.split(key).nth(1).and_then(|v| v.get(..4)).and_then(|v| u32::from_str_radix(v, 16).ok());
        let vendor = id.split("ven_").nth(1).map(|v| vendor_of(&v[..v.len().min(4)])).unwrap_or("other");
        out.push(adapter(out.len(), name.trim(), vendor, field("ven_").zip(field("dev_")), Some(bytes / (1024 * 1024))));
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
        let Some(vendor_id) = read("vendor") else { continue };
        let vendor = vendor_of(&vendor_id);
        let hex = |v: String| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok();
        let pci = hex(vendor_id).zip(read("device").and_then(hex));
        let total = read("mem_info_vram_total").and_then(|v| v.parse::<u64>().ok()).filter(|b| *b > 0).map(|b| b / (1024 * 1024));
        let name = read("product_name").filter(|n| !n.is_empty()).unwrap_or_else(|| match vendor {
            "amd" => "AMD GPU".to_string(),
            "intel" => "Intel GPU".to_string(),
            "nvidia" => "NVIDIA GPU".to_string(),
            _ => "GPU".to_string(),
        });
        out.push(adapter(out.len(), &name, vendor, pci, total));
    }
    out
}

/// An Apple-silicon Mac's GPU, in the shape the others have: the chip's own name (its GPU has no other) and, for its
/// memory, the computer's, which is what it has. `unified` says so: a page shows it as sharing the RAM, and the
/// weights' share of it is the engine's own to pick ([`crate::llm::auto_webgpu_gb`] leaves it alone).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn apple_gpu(chip: &str, total_mb: Option<u64>) -> Json {
    let name = match chip.trim() {
        "" => "Apple GPU",
        chip => chip,
    };
    match adapter(0, name, "apple", None, total_mb) {
        Json::Obj(mut fields) => {
            fields.push(("unified".into(), Json::Bool(true)));
            Json::Obj(fields)
        }
        other => other,
    }
}

/// `g` (an adapter's) with what it is doing now, where that was read: its load (percent), the bytes of memory it has
/// in use and its temperature (°C).
/// Each adapter's readings now: those of the GPU with its PCI ids (the nth of those ids for the nth such adapter: two
/// cards alike), an Apple-silicon Mac's own GPU's for its `apple` one. An adapter no reading is for keeps its nulls.
fn fill(gpus: &mut [Json], readings: &[oaiy_gpu_stats::Gpu]) {
    let mut taken = vec![false; readings.len()];
    for g in gpus.iter_mut() {
        let pci = g.get("pci").and_then(Json::as_str).and_then(|p| {
            let (v, d) = p.split_once(':')?;
            Some((u32::from_str_radix(v, 16).ok()?, u32::from_str_radix(d, 16).ok()?))
        });
        let apple = g.get("vendor").and_then(Json::as_str) == Some("apple");
        let found = readings.iter().enumerate().position(|(i, r)| {
            !taken[i]
                && match pci {
                    Some((v, d)) => r.vendor_id == Some(v) && r.device_id == Some(d),
                    None => apple && r.vendor_id.is_none(),
                }
        });
        if let Some(i) = found {
            taken[i] = true;
            now(g, readings[i].load, readings[i].memory_in_use, readings[i].temperature);
        }
    }
}

fn now(g: &mut Json, load: Option<i64>, in_use: Option<i64>, temperature: Option<f64>) {
    let Json::Obj(fields) = g else { return };
    let said = [
        ("utilization", load.map(|p| Json::Int(p.clamp(0, 100)))),
        ("memory_used_mb", in_use.map(|b| Json::Int(b.max(0) >> 20))),
        ("temperature", temperature.map(|t| Json::Int(t.round() as i64))),
    ];
    for (name, value) in said {
        let Some(value) = value else { continue };
        match fields.iter_mut().find(|(k, _)| k == name) {
            Some((_, v)) => *v = value,
            None => fields.push((name.into(), value)),
        }
    }
}

/// What one of the programs every Mac has printed (`sysctl`, `vm_stat`), or None. By its full path: a program started
/// from the Finder has a short PATH.
#[cfg(target_os = "macos")]
fn said(program: &str, args: &[&str]) -> Option<String> {
    let out = quiet(program).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// macOS on Apple silicon: the chip's GPU, whose memory is the computer's. An Intel Mac's adapters are not read (its
/// GPUs have memory of their own that `sysctl` does not tell), so it lists none, as before.
#[cfg(target_os = "macos")]
fn other_gpus() -> Vec<Json> {
    if !cfg!(target_arch = "aarch64") {
        return Vec::new();
    }
    let chip = said("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
    vec![apple_gpu(&chip, mac_total_mb())]
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn other_gpus() -> Vec<Json> {
    Vec::new()
}

/// The megabytes a new allocation can have on a Mac, from `vm_stat`'s text: its free, speculative and inactive pages
/// (the last the file cache the system drops when asked), at the page size the first line says (16 KB on Apple
/// silicon). "Pages free" alone undercounts badly, as Linux's MemFree does.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_vm_stat(text: &str) -> Option<u64> {
    let page = text.lines().next()?.split("page size of ").nth(1)?.split_whitespace().next()?.parse::<u64>().ok()?;
    let pages = |name: &str| text.lines().find_map(|l| l.strip_prefix(name)).and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok());
    Some((pages("Pages free:")? + pages("Pages inactive:")? + pages("Pages speculative:").unwrap_or(0)) * page / (1024 * 1024))
}

/// A Mac's memory in megabytes, by itself: its GPU's memory is this, whether or not `vm_stat` can be read too.
#[cfg(target_os = "macos")]
fn mac_total_mb() -> Option<u64> {
    said("/usr/sbin/sysctl", &["-n", "hw.memsize"])?.trim().parse::<u64>().ok().map(|bytes| bytes / (1024 * 1024)).filter(|&mb| mb > 0)
}

#[cfg(target_os = "macos")]
fn ram_mb() -> Option<(u64, u64)> {
    let total = mac_total_mb()?;
    let available = parse_vm_stat(&said("/usr/bin/vm_stat", &[])?)?;
    Some((total, available.min(total)))
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

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
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
        // (a Python's own folder is the one hidden folder there is a reason to open: `.venv/bin/python3`)
        if (name.starts_with('.') && name != ".venv") || name.starts_with('$') {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            dirs.push((name.to_ascii_lowercase(), entry_json(&p, &name, true, 0)));
        } else {
            let ext = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let shown = SHOWN.contains(&ext.as_str()) && (ext != "exe" || name.to_ascii_lowercase().contains("ffmpeg")) && (!ext.is_empty() || name.contains("ffmpeg"));
            // A Python to run tinygrad with (`python3`, `python3.12`, `python.exe`: the eGPU's setting on a Mac).
            let python = name.to_ascii_lowercase().starts_with("python") && !name.to_ascii_lowercase().ends_with(".py");
            // NAF's and Real-ESRGAN's weights, the PyTorch files a model needs.
            if shown || python || crate::detect::readable_pth(&name) {
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
        let pci: Vec<Option<&str>> = gpus.iter().map(|g| g.get("pci").and_then(Json::as_str)).collect();
        assert_eq!(pci, vec![Some("1002:13c0"), Some("10de:2b85"), Some("8086:56a0")]);
        assert!(parse_windows_adapters("").is_empty());
        assert_eq!(vendor_of("0x1002"), "amd");
        assert_eq!(vendor_of("0x8086"), "intel");
    }

    #[test]
    fn an_apple_silicon_macs_gpu_is_the_chip_with_the_computers_memory() {
        let g = apple_gpu("Apple M4 Pro\n", Some(49152));
        assert_eq!(g.get("index").and_then(Json::as_i64), Some(0));
        assert_eq!(g.get("name").and_then(Json::as_str), Some("Apple M4 Pro"));
        assert_eq!(g.get("vendor").and_then(Json::as_str), Some("apple"));
        assert_eq!(g.get("memory_total_mb").and_then(Json::as_i64), Some(49152));
        assert_eq!(g.get("unified").and_then(Json::as_bool), Some(true));
        // what it uses is the computer's use, which the RAM's own row shows
        assert_eq!(g.get("memory_used_mb"), Some(&Json::Null));
        // a chip `sysctl` would not name, on a Mac whose memory could not be read
        let unknown = apple_gpu("", None);
        assert_eq!(unknown.get("name").and_then(Json::as_str), Some("Apple GPU"));
        assert_eq!(unknown.get("memory_total_mb"), Some(&Json::Null));
        // the others say nothing of it
        assert_eq!(adapter(0, "AMD Radeon(TM) Graphics", "amd", Some((0x1002, 0x13c0)), Some(2048)).get("unified"), None);
        assert_eq!(g.get("pci"), Some(&Json::Null));
    }

    /// What the GPU is doing now goes into its entry where it was read (`oaiy_gpu_stats`), and what was not
    /// read stays null.
    #[test]
    fn a_gpus_load_memory_in_use_and_temperature_are_its_own_now() {
        let said = |g: &Json| {
            let int = |k: &str| g.get(k).and_then(Json::as_i64);
            (int("utilization"), int("memory_used_mb"), int("temperature"))
        };
        let mut g = apple_gpu("Apple M5 Pro", Some(24576));
        now(&mut g, Some(61), Some(892_764_160), Some(34.65));
        assert_eq!(said(&g), (Some(61), Some(851), Some(35)));
        assert_eq!(g.get("unified").and_then(Json::as_bool), Some(true));
        // read again, in place; a load past 100 is 100
        now(&mut g, Some(140), None, None);
        assert_eq!(said(&g), (Some(100), Some(851), Some(35)));
        let mut quiet = apple_gpu("Apple M5 Pro", Some(24576));
        now(&mut quiet, None, None, None);
        assert_eq!((quiet.get("utilization"), quiet.get("temperature")), (Some(&Json::Null), Some(&Json::Null)));
    }

    /// Each display adapter gets the readings of the GPU with its PCI ids (`oaiy_gpu_stats`, on Windows and Linux; two
    /// cards alike, the first's to the first), an Apple-silicon Mac's GPU those of the one with none; an adapter with
    /// no reading keeps its nulls.
    #[test]
    fn each_adapter_gets_its_own_gpus_readings() {
        use oaiy_gpu_stats::Gpu;
        let said = |g: &Json| {
            let int = |k: &str| g.get(k).and_then(Json::as_i64);
            (int("utilization"), int("memory_used_mb"), int("temperature"))
        };
        let mut gpus = vec![
            adapter(0, "AMD Radeon RX 7900 XTX", "amd", Some((0x1002, 0x744c)), Some(24560)),
            adapter(1, "AMD Radeon RX 7900 XTX", "amd", Some((0x1002, 0x744c)), Some(24560)),
            adapter(2, "Intel(R) Arc(TM) A770 Graphics", "intel", Some((0x8086, 0x56a0)), Some(16288)),
            adapter(3, "Some GPU", "other", None, None),
        ];
        let readings = [
            Gpu { vendor_id: Some(0x8086), device_id: Some(0x56a0), load: None, memory_in_use: Some(1 << 30), temperature: Some(48.0) },
            Gpu { vendor_id: Some(0x1002), device_id: Some(0x744c), load: Some(97), memory_in_use: Some(20 << 30), temperature: Some(71.4) },
            Gpu { vendor_id: Some(0x1002), device_id: Some(0x744c), load: Some(3), memory_in_use: Some(512 << 20), temperature: Some(40.0) },
            Gpu { vendor_id: None, device_id: None, load: Some(50), memory_in_use: None, temperature: Some(60.0) },
        ];
        fill(&mut gpus, &readings);
        let got: Vec<_> = gpus.iter().map(said).collect();
        assert_eq!(got, vec![(Some(97), Some(20480), Some(71)), (Some(3), Some(512), Some(40)), (None, Some(1024), Some(48)), (None, None, None)]);
        // an Apple-silicon Mac's GPU: the reading with no PCI ids
        let mut mac = vec![apple_gpu("Apple M5 Pro", Some(49152))];
        fill(&mut mac, &readings[3..]);
        assert_eq!(said(&mac[0]), (Some(50), None, Some(60)));
        fill(&mut mac, &[]);
        assert_eq!(said(&mac[0]), (Some(50), None, Some(60)), "nothing read: as it was");
    }

    #[test]
    fn a_macs_available_memory_is_read_from_vm_stat() {
        // (the lines `vm_stat` prints, their counts made up: 16 KB pages, as Apple silicon's)
        let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                    Pages free:                               20000.\n\
                    Pages active:                            400000.\n\
                    Pages inactive:                          300000.\n\
                    Pages speculative:                         5000.\n\
                    Pages wired down:                        150000.\n\
                    Pages purgeable:                           1000.\n\
                    \"Translation faults\":                 123456789.\n\
                    Pages occupied by compressor:             30000.\n";
        assert_eq!(parse_vm_stat(text), Some((20000 + 300000 + 5000) * 16384 / (1024 * 1024)));
        assert_eq!(parse_vm_stat("Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 512.\nPages inactive: 256.\n"), Some(3));
        assert_eq!(parse_vm_stat(""), None);
        assert_eq!(parse_vm_stat("Pages free: 10.\nPages inactive: 5.\n"), None);
    }

    #[test]
    fn nvidia_smi_lines_parse_and_listings_filter_to_models() {
        let g = parse_gpu("1, NVIDIA GeForce RTX 5090, 7456, 32607, 3, 41").unwrap();
        assert_eq!(g.get("memory_total_mb").and_then(Json::as_i64), Some(32607));
        assert_eq!(g.get("name").and_then(Json::as_str), Some("NVIDIA GeForce RTX 5090"));
        assert!(parse_gpu("garbage").is_none());
        let d = std::env::temp_dir().join(format!("oaiy-studio-browse-{}", std::process::id()));
        std::fs::create_dir_all(d.join("sub")).unwrap();
        // (a Python's own folder is shown, other hidden folders are not; and a Python to run tinygrad with, by
        // any of its names, but not a Python file)
        std::fs::create_dir_all(d.join(".venv")).unwrap();
        std::fs::create_dir_all(d.join(".git")).unwrap();
        for f in ["a.gguf", "notes.txt", "b.safetensors", "naf_release.pth", "other.pth", "python3", "python3.12", "python_notes.py", "tool"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let listing = browse(Some(d.to_str().unwrap())).unwrap();
        let names: Vec<_> = listing.get("entries").unwrap().as_array().unwrap().iter().map(|e| e.get("name").unwrap().as_str().unwrap().to_string()).collect();
        assert_eq!(names, [".venv", "sub", "a.gguf", "b.safetensors", "naf_release.pth", "python3", "python3.12"]);
        assert!(browse(Some(d.join("missing").to_str().unwrap())).is_err());
        std::fs::remove_dir_all(d).unwrap();
        assert!(!browse(None).unwrap().get("entries").unwrap().as_array().unwrap().is_empty());
    }
}
