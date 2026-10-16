//! What every job shares, whatever it makes: the backend it asks for, the LoRA adapters it names, and the seeded
//! noise its latents start from.

use oaiy_engine::json::Json;
use std::path::PathBuf;

/// A job's `backend`, true for WebGPU: `webgpu` (any GPU wgpu reaches), which is also what a job naming none gets
/// in a build that has it, or `cpu` (Candle on the CPU: the reference, and slow). `cuda` is refused by name: a job
/// written for the CUDA build that was says so, rather than crawl through the CPU unasked.
pub(crate) fn backend_is_webgpu(j: &Json, what: &str) -> std::result::Result<bool, String> {
    match j.get("backend").and_then(Json::as_str) {
        None => Ok(cfg!(feature = "webgpu")),
        Some("webgpu") if cfg!(feature = "webgpu") => Ok(true),
        Some("webgpu") => Err(format!("{what}: this build has no WebGPU (the webgpu feature)")),
        Some("cpu") => Ok(false),
        Some("cuda") => Err(format!("{what}: there is no CUDA backend any more: leave backend out for WebGPU, or give cpu")),
        Some(other) => Err(format!("{what}: backend must be webgpu or cpu, not {other}")),
    }
}

/// `loras`: LoRA adapters as `[{"path": …, "strength": 0.8}]` (strength 1 when left out), or plain paths.
pub(crate) fn parse_loras(j: &Json) -> std::result::Result<Vec<(PathBuf, f64)>, String> {
    let Some(list) = j.get("loras") else { return Ok(Vec::new()) };
    if matches!(list, Json::Null) {
        return Ok(Vec::new());
    }
    list.as_array()
        .ok_or("loras must be an array of {path, strength}")?
        .iter()
        .map(|l| {
            let (path, strength) = match l {
                Json::Str(_) => (l.as_str(), None),
                _ => (l.get("path").and_then(Json::as_str), l.get("strength")),
            };
            let path = path.filter(|p| !p.trim().is_empty()).ok_or("each LoRA needs a path")?;
            let strength = match strength {
                None | Some(Json::Null) => 1.,
                Some(v) => v.as_f64().ok_or("a LoRA's strength must be a number")?,
            };
            Ok((PathBuf::from(path), strength))
        })
        .collect()
}

/// `n` normal samples (Box-Muller over SplitMix64) from `seed`: the same noise for the same seed, on every machine.
pub(crate) fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let radius = (-2. * uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * uniform();
        out.push((radius * angle.cos()) as f32);
        if out.len() < n {
            out.push((radius * angle.sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_jobs_backend_is_webgpu_unless_it_names_the_cpu() {
        let j = |b: &str| Json::parse(format!(r#"{{"backend":"{b}"}}"#).as_bytes()).unwrap();
        assert_eq!(backend_is_webgpu(&Json::parse(b"{}").unwrap(), "image"), Ok(cfg!(feature = "webgpu")), "none named: the GPU where the build has it");
        assert_eq!(backend_is_webgpu(&j("cpu"), "image"), Ok(false));
        if cfg!(feature = "webgpu") {
            assert_eq!(backend_is_webgpu(&j("webgpu"), "image"), Ok(true));
        }
        assert!(backend_is_webgpu(&j("cuda"), "image").unwrap_err().contains("no CUDA backend"));
        assert!(backend_is_webgpu(&j("vulkan"), "image").unwrap_err().contains("webgpu or cpu"));
    }

    #[test]
    fn seeds_are_repeatable_and_distinct() {
        assert_eq!(noise(7, 13), noise(7, 13));
        assert_ne!(noise(7, 13), noise(8, 13));
        assert!(noise(0, 10000).iter().all(|x| x.is_finite()));
    }
}
