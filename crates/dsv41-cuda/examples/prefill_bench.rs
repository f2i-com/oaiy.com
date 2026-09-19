//! Prefill throughput: tokenize a text (by default this repo's own Rust
//! sources, like the code a coding harness sends), then run it through the
//! model in chunks, timing each chunk and where the time went.
//!
//!   cargo run -p dsv41-cuda --release --example prefill_bench -- [tokens] [chunk] [devices] [text-file]
//!
//! Defaults: 4096 tokens in chunks of 1024 on cuda:1,0. Environment as for
//! the generate example (DSV41_MODEL, DSV41_GOLDEN_DIR, DSV41_RAM_GB,
//! DSV41_USAGE); DSV41_RUNS repeats the prompt (the later runs show warm
//! caches); DSV41_PROFILE adds a per-phase breakdown (with extra syncs);
//! DSV41_WAIT_WARM=1 lets the background RAM fill finish first (a server
//! that has been up a while); DSV41_HEADROOM_GB sets the VRAM kept for
//! activations (default 2, as nrob-server's).
//!
//! DSV41_LAYERED=1 runs layer by layer; DSV41_PASS=N splits that into passes
//! of N tokens, as nrob-server does a stretch longer than its layered_max.

use std::path::{Path, PathBuf};
use std::time::Instant;

use dsv41::tokenizer::Tokenizer;
use dsv41_cuda::{GpuModel, GpuOptions};

fn sources(dir: &Path, out: &mut String, limit: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if out.len() > limit {
            return;
        }
        if p.is_dir() && p.file_name().is_some_and(|n| n != "target") {
            sources(&p, out, limit);
        } else if p.extension().is_some_and(|e| e == "rs") {
            if let Ok(s) = std::fs::read_to_string(&p) {
                out.push_str(&format!("// file: {}\n{s}\n", p.display()));
            }
        }
    }
}

fn main() -> nrob::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_tokens: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let chunk: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let devices: Vec<usize> = args.get(3).map_or("1,0", String::as_str).split(',').map(|v| v.trim().parse().expect("device ordinal")).collect();
    let model_dir = std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| r"E:\deepseek\model".into());
    let golden = std::env::var_os("DSV41_GOLDEN_DIR").map(PathBuf::from).unwrap_or_else(|| r"E:\deepseek\golden".into());
    let ram_gb: usize = std::env::var("DSV41_RAM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(140);
    let runs: usize = std::env::var("DSV41_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);

    let tok = Tokenizer::load(&model_dir)?;
    let text = match args.get(4) {
        Some(f) => std::fs::read_to_string(f)?,
        None => {
            let mut s = String::new();
            sources(Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates dir"), &mut s, n_tokens * 8);
            s
        }
    };
    let t = Instant::now();
    let mut ids = tok.encode(&text);
    eprintln!("[tokenized {} bytes into {} tokens in {:.2}s]", text.len(), ids.len(), t.elapsed().as_secs_f64());
    ids.truncate(n_tokens);

    let t = Instant::now();
    let headroom_gb: f64 = std::env::var("DSV41_HEADROOM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let opts = GpuOptions { devices: devices.clone(), max_seq: ids.len() + 16, expert_cache_bytes: ram_gb << 30, direct_io: true, vram_expert_bytes: None, vram_headroom_bytes: (headroom_gb * (1u64 << 30) as f64) as usize, cpu_expert_threads: Some(24), vision: false };
    let mut model = GpuModel::load(&model_dir, &golden.join("engram_meta.safetensors"), &opts)?;
    eprintln!("[loaded in {:.1}s on cuda:{devices:?}]", t.elapsed().as_secs_f64());
    let usage = std::env::var_os("DSV41_USAGE").map(PathBuf::from).unwrap_or_else(|| r"E:\deepseek\expert_usage.txt".into());
    if usage.exists() && std::env::var("DSV41_USAGE").map_or(true, |v| v != "off") {
        let t = Instant::now();
        let (vram, queued) = model.warm(&usage, 4)?;
        eprintln!("[warmed {vram} experts into VRAM in {:.1}s; {queued} queued for RAM]", t.elapsed().as_secs_f64());
    }
    if std::env::var_os("DSV41_WAIT_WARM").is_some() {
        let t = Instant::now();
        while let Some((done, false)) = model.warming() {
            eprint!("\r[RAM fill: {done} records]");
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        eprintln!("\n[RAM fill finished in {:.0}s; RAM holds {} records]", t.elapsed().as_secs_f64(), model.expert_cache().len());
    }
    if std::env::var_os("DSV41_PROFILE").is_some() {
        model.enable_profile();
    }

    // DSV41_LAYERED=1: one layer-by-layer pass over all the tokens instead;
    // DSV41_STEP=1: token by token through the decode path, as a server
    // feeds a short prompt (DSV41_HEAD=1: with the output head every token)
    let layered = std::env::var_os("DSV41_LAYERED").is_some();
    let step = std::env::var_os("DSV41_STEP").is_some();
    let head = std::env::var_os("DSV41_HEAD").is_some();
    for run in 0..runs {
        if step {
            let t = Instant::now();
            for (p, id) in ids.iter().enumerate() {
                if head || p + 1 == ids.len() {
                    model.forward(&[*id], p)?;
                } else {
                    model.advance_with(&[*id], p, &[])?;
                }
            }
            let s = t.elapsed().as_secs_f64();
            eprintln!("[run {}: {} tokens one by one{} in {s:.1}s = {:.2} tok/s]", run + 1, ids.len(), if head { " (head every token)" } else { "" }, ids.len() as f64 / s);
            continue;
        }
        if layered {
            let pass: usize = std::env::var("DSV41_PASS").ok().and_then(|v| v.parse().ok()).unwrap_or(ids.len()).max(1);
            eprintln!("\n[run {} of {runs}: {} tokens layer by layer in passes of up to {pass}, attention sub-chunks of {chunk}]", run + 1, ids.len());
            let host0 = model.expert_cache().stats();
            let t = Instant::now();
            let mut at = 0;
            while at < ids.len() {
                let end = (at + pass).min(ids.len());
                let tp = Instant::now();
                model.prefill_layered(&ids[at..end], at, chunk, None)?;
                eprintln!("  pass {at}..{end}: {:.1}s", tp.elapsed().as_secs_f64());
                at = end;
            }
            let s = t.elapsed().as_secs_f64();
            let host = model.expert_cache().stats();
            eprintln!(
                "[run {}: {} tokens in {s:.1}s = {:.1} tok/s; SSD reads {} ({:.1} GB)]",
                run + 1,
                ids.len(),
                ids.len() as f64 / s,
                host.misses - host0.misses,
                (host.bytes_read - host0.bytes_read) as f64 / 1e9
            );
            let ps = model.pass_stats();
            eprintln!(
                "[experts used {}: resident at the start {} ({:.0}%), when their layer ran {} (VRAM {}); read from the drive {}]",
                ps.used,
                ps.resident_at_start,
                100.0 * ps.resident_at_start as f64 / ps.used.max(1) as f64,
                ps.resident_at_use,
                ps.vram_at_use,
                ps.reads
            );
            eprintln!(
                "[pass {:.1}s: routed experts {:.1}s (waiting for bytes {:.1}s, in uploads {:.1}s), Engram rows {:.1}s, the rest {:.1}s]",
                ps.total_s,
                ps.experts_s,
                ps.wait_s,
                ps.upload_s,
                ps.engram_s,
                ps.total_s - ps.experts_s - ps.engram_s
            );
            if let Some(p) = model.profile() {
                eprintln!("[routed experts: fetch {:.1}s (SSD/RAM reads and uploads), compute {:.1}s; the rest {:.1}s]", p.fetch, p.experts, s - p.fetch - p.experts);
            }
            continue;
        }
        eprintln!("\n[run {} of {runs}: {} tokens in chunks of {chunk}]", run + 1, ids.len());
        let total = Instant::now();
        let mut pos = 0;
        while pos < ids.len() {
            let end = (pos + chunk).min(ids.len());
            let host0 = model.expert_cache().stats();
            let dev0: Vec<_> = model.device_caches().map(|c| c.stats).collect();
            let t = Instant::now();
            model.forward(&ids[pos..end], pos)?;
            let s = t.elapsed().as_secs_f64();
            let host = model.expert_cache().stats();
            let uploads: u64 = model.device_caches().zip(&dev0).map(|(c, d0)| c.stats.misses - d0.misses).sum();
            eprintln!(
                "  chunk {pos:>6}..{end:<6} {s:6.1}s  {:6.1} tok/s  SSD reads {:5} ({:.1} GB)  VRAM uploads {uploads}",
                (end - pos) as f64 / s,
                host.misses - host0.misses,
                (host.bytes_read - host0.bytes_read) as f64 / 1e9,
            );
            pos = end;
        }
        let s = total.elapsed().as_secs_f64();
        eprintln!("[run {}: {} tokens in {s:.1}s = {:.1} tok/s]", run + 1, ids.len(), ids.len() as f64 / s);
        if let Some(p) = model.profile() {
            eprintln!("[profile so far: {p:?}]");
        }
    }
    Ok(())
}
