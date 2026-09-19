//! Phase C1 gate: the GPU model against the oracle's golden files, with the
//! same method and bounds as the CPU model (`dsv41::golden`), then greedy
//! decoding token for token. Heavy, `#[ignore]`d:
//!
//!   cargo test -p dsv41-cuda --release --test gpu_model -- --ignored --nocapture --test-threads=1
//!
//! Environment: DSV41_MODEL, DSV41_GOLDEN_DIR, DSV41_GOLDEN_NAME (as in the
//! dsv41 tests) and DSV41_CUDA_DEVICES (comma-separated, default "1"; "1,0"
//! splits the layers across both cards).

use std::path::PathBuf;
use std::time::Instant;

use dsv41::golden::{apply_overrides, check, isolation, PhaseReport};
use dsv41::model::argmax;
use dsv41::safetensors::StIndex;
use dsv41_cuda::{GpuModel, GpuOptions};

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

fn setup() -> Option<(StIndex, GpuModel)> {
    let model_dir = env_path("DSV41_MODEL", r"E:\deepseek\model");
    let golden_dir = env_path("DSV41_GOLDEN_DIR", r"E:\deepseek\golden");
    let name = std::env::var("DSV41_GOLDEN_NAME").unwrap_or_else(|_| "golden.safetensors".into());
    let (golden, meta) = (golden_dir.join(&name), golden_dir.join("engram_meta.safetensors"));
    if !model_dir.join("config.json").exists() || !golden.exists() || !meta.exists() {
        eprintln!("skipping: checkpoint or golden files not found");
        return None;
    }
    let devices: Vec<usize> = std::env::var("DSV41_CUDA_DEVICES")
        .unwrap_or_else(|_| "1".into())
        .split(',')
        .map(|v| v.trim().parse().expect("DSV41_CUDA_DEVICES: comma-separated ordinals"))
        .collect();
    let g = StIndex::open_file(&golden).unwrap();
    let t = Instant::now();
    let vram_expert_bytes = std::env::var("DSV41_VRAM_EXPERT_GB").ok().and_then(|v| v.parse::<f64>().ok()).map(|gb| (gb * (1u64 << 30) as f64) as usize);
    // DSV41_CPU_THREADS: hybrid decode (VRAM misses on the CPU); with a small
    // DSV41_VRAM_EXPERT_GB most decode experts then take the CPU path
    let cpu_expert_threads = std::env::var("DSV41_CPU_THREADS").ok().and_then(|v| v.parse().ok());
    let opts = GpuOptions { devices: devices.clone(), max_seq: 1024, expert_cache_bytes: 24 << 30, direct_io: false, vram_expert_bytes, cpu_expert_threads };
    let mut model = match GpuModel::load(&model_dir, &meta, &opts) {
        Ok(m) => m,
        // only a missing device may skip; kernels that do not compile fail
        Err(e) if format!("{e}").contains("CompileError") => panic!("kernels do not compile: {e}"),
        Err(e) => {
            eprintln!("skipping: GPU model did not load ({e})");
            return None;
        }
    };
    apply_overrides(&mut model, &g).unwrap();
    eprintln!("{name}: GPU model on cuda:{devices:?} loaded in {:.1}s", t.elapsed().as_secs_f64());
    Some((g, model))
}

fn print(r: &PhaseReport) {
    for l in &r.layers {
        eprintln!(
            "  {} layer {:2}: out p50 {:.1e} p95 {:.1e} max {:.1e}  bf16-exact {:4.1}%  attn p95 {:.1e}  moe p95 {:.1e}  route flips {}",
            r.phase,
            l.layer,
            l.p50,
            l.p95,
            l.max,
            100.0 * l.bf16_exact,
            l.attn_p95,
            l.moe_p95,
            l.route_flips
        );
    }
    eprintln!(
        "  {}: {} of {} token-routes differ ({:.2}%); logits rel-L2 {:.2e}",
        r.phase,
        r.route_flips,
        r.routes,
        100.0 * r.route_flips as f64 / r.routes.max(1) as f64,
        r.logits_rel
    );
}

#[test]
#[ignore]
fn gpu_layers_in_isolation() {
    let Some((g, mut model)) = setup() else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let first: u32 = g.read_i64("generated_ids").unwrap()[0] as u32;
    let mut failures = Vec::new();
    for (phase, ids, start) in [("prefill", prompt.clone(), 0), ("decode0", vec![first], prompt.len())] {
        let t = Instant::now();
        let r = isolation(&mut model, &g, phase, &ids, start).unwrap();
        eprintln!("{phase}: {:.1}s", t.elapsed().as_secs_f64());
        print(&r);
        failures.extend(check(&r));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn gpu_greedy_matches_oracle() {
    let Some((g, mut model)) = setup() else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let expected: Vec<u32> = g.read_i64("generated_ids").unwrap().iter().map(|&v| v as u32).collect();
    let t = Instant::now();
    let mut got = vec![argmax(&model.forward(&prompt, 0).unwrap())];
    eprintln!("prefill of {} tokens: {:.1}s", prompt.len(), t.elapsed().as_secs_f64());
    let mut steps = Vec::new();
    while got.len() < expected.len() {
        let pos = prompt.len() + got.len() - 1;
        let t = Instant::now();
        let logits = model.forward(&got[got.len() - 1..], pos).unwrap();
        steps.push(t.elapsed().as_secs_f64());
        got.push(argmax(&logits));
    }
    eprintln!(
        "decode: {:.2}s/token mean over {} steps\n host cache {:?}",
        steps.iter().sum::<f64>() / steps.len().max(1) as f64,
        steps.len(),
        model.expert_cache().stats()
    );
    for (i, c) in model.device_caches().enumerate() {
        eprintln!(" VRAM cache {i} ({} slots) {:?}", c.slots(), c.stats);
    }
    eprintln!("generated {got:?}\n expected {expected:?}");
    assert_eq!(got, expected, "greedy tokens differ from the oracle");
}
