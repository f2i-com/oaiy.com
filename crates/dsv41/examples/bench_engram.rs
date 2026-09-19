//! Time Engram row reads alone (no GPU): one decode step's rows for each
//! engram layer, over a stream of random tokens.
//! Usage: bench_engram MODEL_DIR ENGRAM_META [steps]
use std::path::Path;
use std::time::Instant;

use dsv41::config::Config;
use dsv41::engram::{Engram, NgramHasher};
use dsv41::safetensors::StIndex;

fn main() -> nrob::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (dir, meta) = (Path::new(&args[1]), Path::new(&args[2]));
    let steps: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64);
    let cfg = Config::load(dir)?;
    let idx = StIndex::open(dir)?;
    let engrams: Vec<Engram> = cfg.engram_layer_ids.iter().map(|&l| Engram::load(&idx, &cfg, l)).collect::<nrob::Result<_>>()?;
    let mut hasher = NgramHasher::load(meta, &cfg, steps + 1)?;
    let cols = hasher.cols();
    let mut s = 0x9E37_79B9u64;
    let mut total = 0.0;
    for pos in 0..steps {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let id = (s % 100_000) as u32 + 1000;
        let hashes = hasher.forward(&[id], pos)?;
        let t0 = Instant::now();
        for (k, eg) in engrams.iter().enumerate() {
            std::hint::black_box(eg.rows(&hashes[k * cols..(k + 1) * cols], cfg.engram_head_dim)?);
        }
        total += t0.elapsed().as_secs_f64();
    }
    println!("{} engram layers x {cols} rows: {:.2} ms per decode step (mean over {steps})", engrams.len(), 1e3 * total / steps as f64);
    Ok(())
}
