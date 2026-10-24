//! Expert-record read throughput from the checkpoint at several queue depths
//! (concurrent readers), page cache bypassed as the RAM tier reads it.
//! Usage: bench_ssd MODEL_DIR [records per depth]
use std::path::Path;
use std::time::Instant;

use oaiy_engine::store::WeightStore;
use oaiy_llm_server::dsv41::expert::SafetensorsExpertStore;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: bench_ssd MODEL_DIR [records]");
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(32);
    let store = SafetensorsExpertStore::open_dir(Path::new(dir), true).expect("open store");
    // (a record is the checkpoint's: an fp4 one's, or an EXL3 one's three projections)
    let (bytes, (layers, experts)) = (store.record_bytes(), store.shape());
    println!("{layers} layers of {experts} experts, {:.1} MB a record", bytes as f64 / 1e6);
    let mut next = 0x2545_F491u64;
    for qd in [1usize, 2, 4, 8, 16] {
        // distinct pseudo-random experts per depth, so no run re-reads another's
        let keys: Vec<(u32, u32)> = (0..n)
            .map(|_| {
                next ^= next << 13;
                next ^= next >> 7;
                next ^= next << 17;
                ((next % layers as u64) as u32, ((next >> 8) % experts as u64) as u32)
            })
            .collect();
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for w in 0..qd {
                let (keys, store) = (&keys, &store);
                s.spawn(move || {
                    let mut buf = vec![0u8; bytes];
                    for &(l, e) in keys.iter().skip(w).step_by(qd) {
                        store.fetch(l, e, &mut buf).expect("fetch");
                    }
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        println!("qd {qd}: {n} records in {dt:.2}s = {:.2} GB/s, {:.1} ms/record", (n * bytes) as f64 / dt / 1e9, dt * 1e3 / n as f64);
    }
}
