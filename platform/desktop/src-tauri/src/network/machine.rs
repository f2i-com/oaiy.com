//! What a computer has and how busy it is, as it tells the computers that use it (and they their Agent, to choose
//! where to run what): its processor (name, cores, how busy), its memory, and its GPUs (from its engines). Read with
//! the standard library and the system's own programs, a few seconds apart at most.

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
        #[cfg(windows)]
        let name = std::env::var("PROCESSOR_IDENTIFIER").ok();
        #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
        let name: Option<String> = None;
        name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).unwrap_or_else(|| std::env::consts::ARCH.to_string())
    })
    .clone()
}

/// The last minute's load (processes running or waiting), where the system says it: a Mac's or Linux's.
fn load() -> Option<f64> {
    static LAST: Mutex<Option<(Instant, Option<f64>)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((at, v)) = *last {
        if at.elapsed() < Duration::from_secs(3) {
            return v;
        }
    }
    #[cfg(target_os = "macos")]
    let v = said("/usr/sbin/sysctl", &["-n", "vm.loadavg"]).and_then(|t| t.trim().trim_start_matches('{').split_whitespace().next().and_then(|n| n.parse().ok()));
    #[cfg(target_os = "linux")]
    let v = std::fs::read_to_string("/proc/loadavg").ok().and_then(|t| t.split_whitespace().next().and_then(|n| n.parse().ok()));
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let v: Option<f64> = None;
    *last = Some((Instant::now(), v));
    v
}

#[cfg(target_os = "macos")]
fn said(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// This computer's processor: `{name, cores, busy}` (busy: the last minute's load against its cores, as a percent,
/// where the system says it).
pub fn cpu() -> Value {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let busy = load().map(|l| ((l / cores as f64) * 100.0).clamp(0.0, 100.0).round() as i64);
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
}
