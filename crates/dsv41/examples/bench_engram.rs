//! Time Engram row reads alone (no GPU): one decode step's rows for each
//! engram layer, over a stream of random tokens, repeated `passes` times
//! (the repeats find their rows in the page cache).
//! Usage: bench_engram MODEL_DIR ENGRAM_META [steps] [passes] [seed] [prompt]
//! (a new seed finds rows the page cache does not hold yet; "prompt" reads
//! every step's rows in one lookup per layer, as a prefill does)
use std::path::Path;
use std::time::Instant;

use dsv41::config::Config;
use dsv41::engram::{Engram, NgramHasher};
use dsv41::safetensors::StIndex;

fn main() -> nrob::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (dir, meta) = (Path::new(&args[1]), Path::new(&args[2]));
    let steps: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64);
    let passes: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let seed: u64 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(0x9E37_79B9);
    let prompt = args.get(6).is_some_and(|v| v == "prompt");
    let cfg = Config::load(dir)?;
    let idx = StIndex::open(dir)?;
    let engrams: Vec<Engram> = cfg.engram_layer_ids.iter().map(|&l| Engram::load(&idx, &cfg, l)).collect::<nrob::Result<_>>()?;
    let mut hasher = NgramHasher::load(meta, &cfg, steps + 1)?;
    let cols = hasher.cols();
    for pass in 0..passes {
        let mut s = seed.max(1);
        let ids: Vec<u32> = (0..steps)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 100_000) as u32 + 1000
            })
            .collect();
        let mut total = 0.0;
        if prompt {
            let hashes = hasher.forward(&ids, 0)?;
            let n = engrams.len();
            let t0 = Instant::now();
            for (k, eg) in engrams.iter().enumerate() {
                let hs: Vec<i64> = (0..steps).flat_map(|i| hashes[(i * n + k) * cols..(i * n + k + 1) * cols].iter().copied()).collect();
                std::hint::black_box(eg.rows(&hs, cfg.engram_head_dim)?);
            }
            total = t0.elapsed().as_secs_f64();
            println!("pass {}: a {steps}-token prompt's rows ({} engram layers x {cols}): {:.2}s", pass + 1, n, total);
            continue;
        }
        for (pos, &id) in ids.iter().enumerate() {
            let hashes = hasher.forward(&[id], pos)?;
            let t0 = Instant::now();
            for (k, eg) in engrams.iter().enumerate() {
                std::hint::black_box(eg.rows(&hashes[k * cols..(k + 1) * cols], cfg.engram_head_dim)?);
            }
            total += t0.elapsed().as_secs_f64();
        }
        println!("pass {}: {} engram layers x {cols} rows: {:.2} ms per decode step (mean over {steps})", pass + 1, engrams.len(), 1e3 * total / steps as f64);
    }
    Ok(())
}
