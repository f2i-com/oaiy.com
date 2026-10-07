//! Throughput micro-bench: prefill + decode tokens-per-second, median over runs.
//!
//! Usage:
//!   cargo run --release -p llama-rs --features cuda --example bench -- model.gguf cuda
//!   cargo run --release -p llama-rs --features cuda --example bench -- model.gguf cuda 256 128 5
//!   cargo run --release -p llama-rs --features cuda --example bench -- model.gguf cuda 128 64 3 --csv
//!
//! Positional args: `<model.gguf> [backend] [prefill_len] [decode_len] [runs] [--csv]`
//!   prefill_len  number of tokens fed to the prefill forward pass (default 128)
//!   decode_len   number of single-token decode steps per run         (default 64)
//!   runs         number of timed runs after a warmup run             (default 3)
//!   --csv        emit a single CSV line on stdout for piping to spreadsheets
//!
//! Sampling is replaced with argmax to keep the loop's perf representative of
//! the model's compute, not the sampler's. The prompt is a synthetic repeat of
//! BOS so sequence content doesn't influence cache locality between runs.
//!
//! Reports min / median / max for both phases so noisy CI machines are obvious.
//!
//! NB: with the per-call KV cache reallocation we use here, decode-step numbers
//! reflect "fresh cache after a long prefill" — i.e., realistic chat first-token
//! conditions, not steady-state long-context decode.

use std::env;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::Model;

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("CUDA backend not enabled. Build with `--features cuda`".into()),
        // "auto": try CUDA, fall back to CPU on failure (no driver, no GPU,
        // or build without --features cuda). Lets the user run on any machine
        // without changing the command line.
        "auto" => {
            {
                eprintln!("auto: built without --features cuda, using CPU");
                Ok(default_backend())
            }
        }
        other => Err(format!("unknown backend `{other}` (try `cpu`, `cuda`, or `auto`)").into()),
    }
}

fn argmax(logits: &ggml_rs::Tensor) -> u32 {
    // `last_logits` always returns CPU storage, so direct slice access is fine.
    let row = logits.data();
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i; }
    }
    best_i as u32
}

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

fn min_max(vals: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in vals { if v < lo { lo = v; } if v > hi { hi = v; } }
    (lo, hi)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw: Vec<String> = env::args().skip(1).collect();
    let csv = raw.iter().any(|a| a == "--csv");
    let mut args = raw.into_iter().filter(|a| a != "--csv");
    let path         = args.next().ok_or("usage: bench <model.gguf> [backend] [prefill_len] [decode_len] [runs] [--csv]")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());
    let prefill_len  = args.next().and_then(|s| s.parse().ok()).unwrap_or(128usize);
    let decode_len   = args.next().and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let runs         = args.next().and_then(|s| s.parse().ok()).unwrap_or(3usize);

    let backend = pick_backend(&backend_name)?;
    let vram_before = backend.vram_status();
    let load_t0 = Instant::now();
    let gguf  = GgufFile::open(&path)?;
    let model = Model::load(&gguf, backend)?;
    let load_ms = load_t0.elapsed().as_millis();
    let vram_after = model.backend().vram_status();

    let cfg = model.config();
    eprintln!(
        "loaded {:?} on {} backend (load {} ms); cfg = vocab={}, layers={}, heads={}, kv={}, hd={}",
        cfg.arch, model.backend().name(), load_ms,
        cfg.vocab_size, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
    );
    if let (Some((free_b, total)), Some((free_a, _))) = (vram_before, vram_after) {
        let used_mib = (free_b.saturating_sub(free_a)) / (1024 * 1024);
        let free_mib = free_a / (1024 * 1024);
        let total_mib = total / (1024 * 1024);
        eprintln!(
            "vram: {} MiB used by load, {} / {} MiB free after",
            used_mib, free_mib, total_mib,
        );
    }
    eprintln!("bench: prefill={} tok, decode={} tok, runs={} (+1 warmup)", prefill_len, decode_len, runs);

    // Synthetic prompt: BOS repeated. Avoids tokenizer encode cost in the loop
    // and keeps prefill length exact.
    let bos = model.tokenizer().bos().unwrap_or(1);
    let prompt: Vec<u32> = std::iter::repeat(bos).take(prefill_len).collect();

    let mut prefill_tps: Vec<f64> = Vec::with_capacity(runs);
    let mut decode_tps:  Vec<f64> = Vec::with_capacity(runs);

    for run in 0..(runs + 1) {
        let mut kv = model.new_kv_cache(prefill_len + decode_len + 16);

        let pre_t0 = Instant::now();
        let logits = model.forward(&prompt, &mut kv);
        let pre_s  = pre_t0.elapsed().as_secs_f64();
        let pre_tps = prefill_len as f64 / pre_s;

        let mut current = argmax(&model.last_logits(&logits));
        let dec_t0 = Instant::now();
        for _ in 0..decode_len {
            let logits = model.forward(&[current], &mut kv);
            current = argmax(&model.last_logits(&logits));
        }
        let dec_s = dec_t0.elapsed().as_secs_f64();
        let dec_tps = decode_len as f64 / dec_s;

        if run == 0 {
            eprintln!("warmup: prefill {:.1} tok/s, decode {:.1} tok/s", pre_tps, dec_tps);
        } else {
            eprintln!("run {}/{}: prefill {:.1} tok/s, decode {:.1} tok/s", run, runs, pre_tps, dec_tps);
            prefill_tps.push(pre_tps);
            decode_tps.push(dec_tps);
        }
    }

    let (pre_lo, pre_hi) = min_max(&prefill_tps);
    let (dec_lo, dec_hi) = min_max(&decode_tps);
    let pre_med = median(&mut prefill_tps);
    let dec_med = median(&mut decode_tps);

    if csv {
        // Header on stderr (so a `bench --csv … | head -1` pipeline still works
        // for the data line on stdout). Columns: model_path, arch, n_layers,
        // backend, prefill_tok, decode_tok, pre_min, pre_med, pre_max, dec_min, dec_med, dec_max
        eprintln!("# csv: model,arch,layers,backend,prefill_tok,decode_tok,pre_min,pre_med,pre_max,dec_min,dec_med,dec_max");
        println!(
            "{},{:?},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            path, cfg.arch, cfg.n_layers, model.backend().name(),
            prefill_len, decode_len,
            pre_lo, pre_med, pre_hi, dec_lo, dec_med, dec_hi,
        );
    } else {
        println!();
        println!("prefill ({} tok): min {:.1}  median {:.1}  max {:.1} tok/s", prefill_len, pre_lo, pre_med, pre_hi);
        println!("decode  ({} tok): min {:.1}  median {:.1}  max {:.1} tok/s", decode_len,  dec_lo, dec_med, dec_hi);
    }

    Ok(())
}
