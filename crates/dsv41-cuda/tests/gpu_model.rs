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
    let opts = GpuOptions { devices: devices.clone(), max_seq: 1024, expert_cache_bytes: 24 << 30, direct_io: false, vram_expert_bytes, vram_headroom_bytes: 1 << 30, cpu_expert_threads, vision: false };
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

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*x as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// Greedy tokens after the prompt, continuing from the model's state.
fn greedy(model: &mut GpuModel, first: u32, pos: usize, n: usize) -> Vec<u32> {
    let mut got = vec![first];
    while got.len() < n {
        let logits = model.forward(&got[got.len() - 1..], pos + got.len() - 1).unwrap();
        got.push(argmax(&logits));
    }
    got
}

/// A prompt fed token by token ends in the same logits whether the tokens
/// before the last run the output head ([`GpuModel::forward`]) or skip it
/// ([`GpuModel::advance_with`], what the server does): the head leaves the
/// state alone. Bit for bit unless CPU experts share the decode (their sums
/// differ from the GPU's in the last bits, and which experts run where
/// depends on the cache state).
#[test]
#[ignore]
fn gpu_advance_matches_forward() {
    let Some((g, mut model)) = setup() else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let n = prompt.len().min(64);
    let mut with_head = Vec::new();
    for p in 0..n {
        with_head = model.forward(&prompt[p..p + 1], p).unwrap();
    }
    for p in 0..n - 1 {
        model.advance_with(&prompt[p..p + 1], p, &[]).unwrap();
    }
    let advanced = model.forward(&prompt[n - 1..n], n - 1).unwrap();
    let e = rel_l2(&advanced, &with_head);
    eprintln!("{n} tokens: advance_with vs forward, last logits rel-L2 {e:.2e}");
    if std::env::var("DSV41_CPU_THREADS").is_ok() {
        assert!(e < 1e-2, "rel-L2 {e}");
    } else {
        assert_eq!(advanced, with_head, "advance_with changed the state");
    }
}

/// Serving continues a conversation from where the cache stopped: the
/// prompt run as a prefill plus chunks continuing it (split at odd
/// positions, so compressor groups straddle chunks, and with a chunk longer
/// than the window ring) must give one prefill's logits and the oracle's
/// greedy tokens; a checkpoint restored mid-way must replay the same logits.
#[test]
#[ignore]
fn gpu_chunked_prefill_and_checkpoints() {
    let Some((g, mut model)) = setup() else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let expected: Vec<u32> = g.read_i64("generated_ids").unwrap().iter().map(|&v| v as u32).collect();
    let n = prompt.len();
    let whole = model.forward(&prompt, 0).unwrap();

    // every chunk at least 2 tokens: a 1-token forward is a decode step,
    // whose MoE path (CPU experts in hybrid mode) sums in another order
    let mut cuts: Vec<usize> = Vec::new();
    for c in [(n / 7).max(2) | 1, (n / 2) | 1, n - 2] {
        if c >= cuts.last().copied().unwrap_or(0) + 2 && c + 2 <= n {
            cuts.push(c);
        }
    }
    for cuts in [cuts.clone(), vec![3.min(n - 1)]] {
        let mut bounds = vec![0];
        bounds.extend(&cuts);
        bounds.push(n);
        bounds.dedup();
        let mut logits = Vec::new();
        let mut mid = None;
        for w in bounds.windows(2) {
            logits = model.forward(&prompt[w[0]..w[1]], w[0]).unwrap();
            if mid.is_none() && w[1] < n {
                mid = Some((w[1], model.checkpoint(w[1]).unwrap()));
            }
        }
        let rel = rel_l2(&whole, &logits);
        eprintln!("chunks at {bounds:?}: last logits rel-L2 vs one prefill {rel:.2e}, argmax {} vs {}", argmax(&logits), argmax(&whole));
        assert!(rel < 1e-3, "chunked prefill drifted: {rel}");

        // back to the mid-way checkpoint and forward again: the same logits
        let (at, ck) = mid.expect("a cut inside the prompt");
        model.restore(&ck).unwrap();
        let again = model.forward(&prompt[at..], at).unwrap();
        let rel2 = rel_l2(&logits, &again);
        eprintln!("  restored at {at}: rel-L2 {rel2:.2e}");
        assert!(rel2 < 1e-4, "restore did not replay: {rel2}");
    }

    // decode from the chunked state matches the oracle
    let full = model.checkpoint(n).unwrap();
    let got = greedy(&mut model, argmax(&whole), n, expected.len());
    eprintln!("generated {got:?}\n expected {expected:?}");
    assert_eq!(got, expected, "greedy tokens after a chunked prefill differ from the oracle");

    // and the end-of-prompt checkpoint replays the reply, after a detour
    model.restore(&full).unwrap();
    let _ = greedy(&mut model, 0, n, 4);
    model.restore(&full).unwrap();
    let replay = greedy(&mut model, argmax(&whole), n, expected.len());
    assert_eq!(replay, expected, "reply after restoring the prompt checkpoint differs");
}

/// The layer-by-layer prefill (every layer over the whole prompt, routed
/// experts fetched once) must give the logits of the same sub-chunks run
/// with `forward`, from position 0 and continuing a sequence, and the
/// oracle's greedy tokens after it.
#[test]
#[ignore]
fn gpu_layered_prefill_matches_chunks() {
    let Some((g, mut model)) = setup() else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let expected: Vec<u32> = g.read_i64("generated_ids").unwrap().iter().map(|&v| v as u32).collect();
    let n = prompt.len();
    let whole = model.forward(&prompt, 0).unwrap();
    for sub in [37usize, 64, n] {
        // sub-chunks of at least 2 tokens (a 1-token forward is a decode step)
        if n % sub == 1 {
            continue;
        }
        let t = Instant::now();
        let layered = model.prefill_layered(&prompt, 0, sub, None).unwrap();
        let rel = rel_l2(&whole, &layered);
        eprintln!("layered, sub-chunks of {sub}: {:.1}s, rel-L2 vs one prefill {rel:.2e}", t.elapsed().as_secs_f64());
        assert!(rel < 1e-3, "layered prefill drifted: {rel}");
    }
    // continuing: a prefix by forward, the rest layered
    let cut = (n / 3) | 1;
    if cut >= 2 && n - cut >= 2 {
        model.forward(&prompt[..cut], 0).unwrap();
        let rest = model.prefill_layered(&prompt[cut..], cut, 29, None).unwrap();
        let rel = rel_l2(&whole, &rest);
        eprintln!("prefix {cut} then layered: rel-L2 {rel:.2e}");
        assert!(rel < 1e-3, "layered continuation drifted: {rel}");
    }
    let got = greedy(&mut model, argmax(&whole), n, expected.len());
    eprintln!("generated {got:?}\n expected {expected:?}");
    assert_eq!(got, expected, "greedy tokens after a layered prefill differ from the oracle");
}
