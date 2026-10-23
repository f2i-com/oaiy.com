//! What a computer has and how busy it is, as it tells the computers that use it (and they their Agent, to choose
//! where to run what): its processor (name, cores, how busy), its memory, and its GPUs (from its engines). Read with
//! the standard library, the system's own programs and, on Windows, its own calls, a few seconds apart at most.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// The processor's name, read once.
fn cpu_name() -> String {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        #[cfg(target_os = "macos")]
        let name = said("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"]);
        #[cfg(target_os = "linux")]
        let name = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|t| t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':')).map(|(_, v)| v.trim().to_string()));
        // (its marketing name, as Task Manager shows it: `PROCESSOR_IDENTIFIER` is its family and model numbers)
        #[cfg(windows)]
        let name = win::processor_name().or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok());
        #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
        let name: Option<String> = None;
        name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).unwrap_or_else(|| std::env::consts::ARCH.to_string())
    })
    .clone()
}

/// How busy the processor is, as a percent, where the system says it: on a Mac or Linux the last minute's load
/// (processes running or waiting) against its `cores`, on Windows its time not idle since it was last asked. Read
/// again a few seconds apart at most.
fn busy(cores: usize) -> Option<f64> {
    static LAST: Mutex<Option<(Instant, Option<f64>)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((at, v)) = *last {
        if at.elapsed() < Duration::from_secs(3) {
            return v;
        }
    }
    #[cfg(target_os = "macos")]
    let load: Option<f64> = said("/usr/sbin/sysctl", &["-n", "vm.loadavg"]).and_then(|t| t.trim().trim_start_matches('{').split_whitespace().next().and_then(|n| n.parse().ok()));
    #[cfg(target_os = "linux")]
    let load: Option<f64> = std::fs::read_to_string("/proc/loadavg").ok().and_then(|t| t.split_whitespace().next().and_then(|n| n.parse().ok()));
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let v = load.map(|l| l / cores.max(1) as f64 * 100.0);
    #[cfg(windows)]
    let v = {
        let _ = cores;
        win::busy()
    };
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    let v: Option<f64> = {
        let _ = cores;
        None
    };
    *last = Some((Instant::now(), v));
    v
}

/// Windows' own readings of the processor: its name, and how busy it is.
#[cfg(windows)]
mod win {
    use std::sync::Mutex;
    use std::time::Duration;

    use windows_sys::Win32::Foundation::FILETIME;

    /// The processor's marketing name ("Intel(R) Core(TM) i9-13900K"), as Windows keeps it in the registry:
    /// `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, `ProcessorNameString`.
    pub fn processor_name() -> Option<String> {
        use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};
        let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let (key, value) = (wide(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"), wide("ProcessorNameString"));
        let mut text = [0u16; 256];
        let mut bytes = (text.len() * 2) as u32;
        // SAFETY: the key and value names end in NUL, and `bytes` is the buffer's size in bytes, which the call writes
        // no further than (it answers an error for a longer value).
        let status = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), text.as_mut_ptr().cast(), &mut bytes) };
        if status != 0 {
            return None;
        }
        let len = (bytes as usize / 2).min(text.len());
        let name = String::from_utf16_lossy(&text[..len]).trim_end_matches('\0').trim().to_string();
        (!name.is_empty()).then_some(name)
    }

    /// How busy the processor was, as a percent: its time not idle between this reading of its times and the last
    /// (one a quarter of a second before, the first time).
    pub fn busy() -> Option<f64> {
        static LAST: Mutex<Option<(u64, u64)>> = Mutex::new(None);
        let mut last = LAST.lock().unwrap_or_else(|p| p.into_inner());
        let before = match *last {
            Some(b) => b,
            None => {
                let b = times()?;
                std::thread::sleep(Duration::from_millis(250));
                b
            }
        };
        let mut now = times()?;
        // (asked again before the system's times moved on, which they do in steps of about 16 ms: nothing to divide
        // by, so a quarter of a second more, as the first time)
        if now.1 == before.1 {
            std::thread::sleep(Duration::from_millis(250));
            now = times()?;
        }
        *last = Some(now);
        let (idle, all) = (now.0.saturating_sub(before.0), now.1.saturating_sub(before.1));
        (all > 0).then(|| (1.0 - idle as f64 / all as f64) * 100.0)
    }

    /// The processor's idle time and all its time (kernel, which counts idle, and user), in 100 ns, since it started.
    fn times() -> Option<(u64, u64)> {
        use windows_sys::Win32::System::Threading::GetSystemTimes;
        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut idle, mut kernel, mut user) = (zero, zero, zero);
        // SAFETY: three FILETIMEs of this frame's, written by the call.
        if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } == 0 {
            return None;
        }
        let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
        Some((ticks(idle), ticks(kernel) + ticks(user)))
    }
}

#[cfg(target_os = "macos")]
fn said(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// This computer's processor: `{name, cores, busy}` (busy as a percent, where the system says it: [`busy`]).
pub fn cpu() -> Value {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let busy = busy(cores).map(|b| b.clamp(0.0, 100.0).round() as i64);
    json!({ "name": cpu_name(), "cores": cores, "busy": busy })
}

/// What a computer tells the ones that use it of itself: its processor, its memory (its engines' reading of it) and
/// its GPUs (their memory, VRAM on a card of its own, how busy and how warm), as `{cpu, ram, gpus}`.
pub fn machine(system: &Value) -> Value {
    json!({
        "cpu": cpu(),
        "ram": system.get("ram").cloned().unwrap_or(Value::Null),
        "gpus": system.get("gpus").cloned().unwrap_or(json!([])),
    })
}

/// What this computer's engines' control page answers at `path` (`Null` where they do not answer).
pub(crate) async fn engines_get(net: &super::Network, path: &str) -> Value {
    let Some(ui) = crate::ai::engine_services::local_engines_ui(&net.inner().ai) else { return Value::Null };
    let Ok(client) = reqwest::Client::builder().timeout(Duration::from_secs(4)).build() else { return Value::Null };
    let Ok(r) = client.get(format!("{}{path}", ui.trim_end_matches('/'))).send().await else { return Value::Null };
    r.json::<Value>().await.unwrap_or(Value::Null)
}

/// What this computer's engines' control page answers `body` posted to `path` with, within `wait` (`Null` where they
/// do not answer).
pub(crate) async fn engines_post(net: &super::Network, path: &str, body: &Value, wait: Duration) -> Value {
    let Some(ui) = crate::ai::engine_services::local_engines_ui(&net.inner().ai) else { return Value::Null };
    let Ok(client) = reqwest::Client::builder().timeout(wait).build() else { return Value::Null };
    let Ok(r) = client.post(format!("{}{path}", ui.trim_end_matches('/'))).json(body).send().await else { return Value::Null };
    r.json::<Value>().await.unwrap_or(Value::Null)
}

/// In a line, what a computer has, as it says it on the LAN before it is paired (to choose which to pair with):
/// its GPUs and their memory, its memory, its cores, e.g. `RTX 4090 (24 GB VRAM) · 64 GB RAM · 32 cores`. Its GPUs
/// are its engines' system reading's and those their resources name besides (an eGPU's card, by its plugin).
pub fn summary(system: &Value, resources: &Value) -> String {
    let gb = |mb: &Value| mb.as_f64().filter(|m| *m > 0.0).map(|m| (m / 1024.0).round() as i64);
    let mut gpus: Vec<Value> = system.get("gpus").and_then(Value::as_array).cloned().unwrap_or_default();
    for d in resources.get("devices").and_then(Value::as_array).into_iter().flatten() {
        if !gpus.iter().any(|g| g.get("name") == d.get("name")) {
            gpus.push(d.clone());
        }
    }
    let mut parts: Vec<String> = gpus
        .iter()
        .filter_map(|g| {
            let name = g.get("name").and_then(Value::as_str)?.trim();
            // (a Mac's GPU shares the computer's memory: said as RAM, not again as the GPU's)
            let unified = g.get("unified").and_then(Value::as_bool) == Some(true);
            Some(match gb(&g["memory_total_mb"]).filter(|_| !unified) {
                Some(v) => format!("{name} ({v} GB VRAM)"),
                None => name.to_string(),
            })
        })
        .collect();
    if let Some(v) = system.pointer("/ram/total_mb").and_then(|m| gb(m)) {
        parts.push(format!("{v} GB RAM"));
    }
    parts.push(format!("{} cores", cpu()["cores"]));
    let line = parts.join(" · ");
    line.chars().filter(|c| !c.is_control()).take(160).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_computer_says_its_processor_and_takes_its_engines_reading_of_the_rest() {
        let c = cpu();
        assert!(c["cores"].as_u64().unwrap() >= 1 && !c["name"].as_str().unwrap().is_empty(), "{c}");
        if let Some(b) = c["busy"].as_i64() {
            assert!((0..=100).contains(&b), "{c}");
        }
        let m = machine(&json!({ "ram": { "total_mb": 16384, "available_mb": 4096 }, "gpus": [{ "name": "RTX 5090", "memory_total_mb": 32768 }] }));
        assert_eq!((m["ram"]["total_mb"].as_i64(), m["gpus"][0]["name"].as_str()), (Some(16384), Some("RTX 5090")));
        assert!(machine(&Value::Null)["gpus"].as_array().unwrap().is_empty(), "engines not answering: no GPUs said, the processor still");
        let cores = c["cores"].as_u64().unwrap();
        let card = json!({ "gpus": [{ "name": "NVIDIA GeForce RTX 4090", "memory_total_mb": 24564 }], "ram": { "total_mb": 65536 } });
        assert_eq!(summary(&card, &Value::Null), format!("NVIDIA GeForce RTX 4090 (24 GB VRAM) · 64 GB RAM · {cores} cores"));
        let mac = json!({ "gpus": [{ "name": "Apple M5 Pro", "memory_total_mb": 24576, "unified": true }], "ram": { "total_mb": 24576 } });
        assert_eq!(summary(&mac, &Value::Null), format!("Apple M5 Pro · 24 GB RAM · {cores} cores"));
        // (a Mac with an eGPU: its card from the engines' resources, the Mac's GPU said once)
        let egpu = json!({ "devices": [{ "name": "Apple M5 Pro", "unified": true }, { "name": "RTX 4090 (eGPU)", "memory_total_mb": 24564, "unified": false }] });
        assert_eq!(summary(&mac, &egpu), format!("Apple M5 Pro · RTX 4090 (eGPU) (24 GB VRAM) · 24 GB RAM · {cores} cores"));
        assert_eq!(summary(&Value::Null, &Value::Null), format!("{cores} cores"));
    }

    /// On Windows the processor is named as Windows names it to a person (not by its family and model numbers), and
    /// how busy it is is said.
    #[cfg(windows)]
    #[test]
    fn windows_names_its_processor_and_says_how_busy_it_is() {
        let name = win::processor_name().unwrap_or_default();
        assert!(!name.is_empty() && !name.contains(" Family "), "{name}");
        let busy = win::busy().unwrap();
        assert!((0.0..=100.0).contains(&busy), "{busy}");
    }
}
