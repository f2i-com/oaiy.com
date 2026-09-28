//! Fill-rate of a growing RAM tier: `readers` threads read expert records
//! into freshly allocated buffers that all stay alive (as the host cache
//! fills), reporting throughput every 250 records.
//! Usage: bench_fill MODEL_DIR records [readers]
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use dsv41::expert::{SafetensorsExpertStore, RECORD_BYTES};
use oaiy_engine::store::WeightStore;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let store = SafetensorsExpertStore::open_dir(Path::new(&args[1]), true).expect("open store");
    let n: usize = args[2].parse().unwrap();
    let readers: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4);
    // "mem": touch every page of each fresh buffer instead of reading into it
    let mem_only = args.get(4).is_some_and(|v| v == "mem");
    let keep: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::with_capacity(n));
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    let last = Mutex::new((0usize, 0.0f64));
    std::thread::scope(|s| {
        for _ in 0..readers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    return;
                }
                let (l, e) = ((i % 40) as u32, ((i / 40) % 384) as u32);
                let mut buf = vec![0u8; RECORD_BYTES];
                if mem_only {
                    for p in buf.chunks_mut(4096) {
                        p[0] = 1;
                    }
                } else {
                    store.fetch(l, e, &mut buf).expect("fetch");
                }
                let mut k = keep.lock().unwrap();
                k.push(buf);
                if k.len().is_multiple_of(250) {
                    let now = t0.elapsed().as_secs_f64();
                    let mut lp = last.lock().unwrap();
                    let gbs = ((k.len() - lp.0) * RECORD_BYTES) as f64 / (now - lp.1) / 1e9;
                    println!("{:5} records ({:5.1} GB) at {now:6.1}s: {gbs:.2} GB/s", k.len(), (k.len() * RECORD_BYTES) as f64 / 1e9);
                    *lp = (k.len(), now);
                }
            });
        }
    });
}
