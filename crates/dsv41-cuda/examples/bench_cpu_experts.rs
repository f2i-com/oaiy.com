//! Time `CpuExperts::forward` on synthetic records (distinct buffers, so the
//! pass streams from DRAM as a real RAM-tier hit does).
//! Usage: bench_cpu_experts [threads...] [hot] [portable] [noop]
use std::sync::Arc;
use std::time::Instant;

use dsv41::cpu_experts::CpuExperts;
use dsv41_cuda::cpu::row_kernel;
use dsv41::expert::{DIM, RECORD_BYTES};

fn main() {
    let threads: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    // "hot": reuse one record (18.8 MB stays in L3), separating compute from DRAM
    let hot = std::env::args().any(|a| a == "hot");
    let threads = if threads.is_empty() { vec![8, 16, 24, 32] } else { threads };
    // 48 records (900 MB) cycled so each call reads cold memory
    let recs: Vec<Arc<Vec<u8>>> = (0..48u8).map(|i| Arc::new(vec![i.wrapping_mul(37) | 0x11; RECORD_BYTES])).collect();
    let x: Vec<f32> = (0..DIM).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();
    for &t in &threads {
        fn noop(_: &[f32], _: &[u8], _: &[u8], _: usize, _: usize, _: &mut [f32]) {}
        let cpu = if std::env::args().any(|a| a == "portable") {
            CpuExperts::new(t)
        } else if std::env::args().any(|a| a == "noop") {
            CpuExperts::with_kernel(t, noop) // the pool's own cost: dispatch, barriers, assembly
        } else {
            CpuExperts::with_kernel(t, row_kernel())
        };
        for m in [1usize, 2, 6] {
            let w = vec![0.5f32; m];
            let mut next = 0;
            let _ = cpu.forward(&recs[..m], &w, &x, 10.0); // warm the pool
            let iters = 24 / m.min(24);
            let t0 = Instant::now();
            for _ in 0..iters {
                let batch: Vec<_> = (0..m).map(|k| Arc::clone(&recs[if hot { k } else { (next + k) % recs.len() }])).collect();
                next += m;
                std::hint::black_box(cpu.forward(&batch, &w, &x, 10.0));
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            println!("threads {t:2}, {m} experts: {:.2} ms/call, {:.2} ms/expert, {:.1} GB/s", dt * 1e3, dt * 1e3 / m as f64, (m * RECORD_BYTES) as f64 / dt / 1e9);
        }
    }
}
