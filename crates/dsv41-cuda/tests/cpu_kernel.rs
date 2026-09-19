//! The dispatched CPU row kernel (AVX-512 on the dev machine) against the
//! portable one and the expert oracle: bit for bit.

use std::sync::Arc;

use dsv41::cpu_experts::{fp4_rows, CpuExperts};
use dsv41::expert::{expert_forward, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1};
use dsv41::formats::to_bf16;

fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

fn record(seed: u64) -> Vec<u8> {
    let mut rec = bytes(RECORD_BYTES, seed);
    for r in [S1, S2, S3] {
        for (i, b) in rec[r].iter_mut().enumerate() {
            *b = 118 + ((i as u64 + seed) % 10) as u8;
        }
    }
    rec
}

fn activation(seed: u64) -> Vec<f32> {
    bytes(DIM, seed).iter().map(|&b| to_bf16((b as f32 - 127.5) / 16.0)).collect()
}

#[test]
fn row_kernel_matches_portable() {
    eprintln!("row kernel: {}", dsv41_cuda::cpu::row_kernel_name());
    let kernel = dsv41_cuda::cpu::row_kernel();
    let rec = record(5);
    let x = activation(6);
    // odd row starts and lengths exercise partial 16-row groups
    for (r0, len) in [(0, INTER), (7, 37), (2288, 16), (100, 1)] {
        let (mut a, mut b) = (vec![0.0f32; len], vec![0.0f32; len]);
        fp4_rows(&x, &rec[W1], &rec[S1], DIM, r0, &mut a);
        kernel(&x, &rec[W1], &rec[S1], DIM, r0, &mut b);
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "rows {r0}+{len}");
    }
}

#[test]
fn cpu_experts_with_kernel_match_oracle() {
    let recs: Vec<Arc<Vec<u8>>> = (0..3).map(|e| Arc::new(record(20 + e))).collect();
    let x = activation(9);
    let weights = [0.4f32, 1.1, 0.25];
    let cpu = CpuExperts::with_kernel(8, dsv41_cuda::cpu::row_kernel());
    let got = cpu.forward(&recs, &weights, &x, 10.0);
    for (e, rec) in recs.iter().enumerate() {
        let want = expert_forward(rec, &x, Some(weights[e]), 10.0);
        assert!(got[e].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "expert {e}");
    }
}
