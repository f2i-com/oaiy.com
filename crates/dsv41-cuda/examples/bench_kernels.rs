//! Decode-shape kernel timings (one token): microseconds per launch and the
//! weight bandwidth that implies. Weights rotate through enough copies to
//! exceed the 96 MB L2, as a real decode step (every layer's weights
//! different) sees them. Usage: bench_kernels [device]
use std::time::Instant;

use dsv41::expert::{DIM, INTER, RECORD_BYTES};
use dsv41_cuda::Gpu;

fn time(g: &Gpu, name: &str, bytes: usize, iters: usize, mut f: impl FnMut(usize)) {
    f(0);
    g.sync().unwrap();
    let t = Instant::now();
    for i in 0..iters {
        f(i);
    }
    g.sync().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{name:40} {us:8.1} us  {:7.0} GB/s", bytes as f64 / us / 1e3);
}

/// Copies of a `bytes`-sized weight needed to overflow L2.
fn copies(bytes: usize) -> usize {
    (300 << 20) / bytes.max(1) + 1
}

fn main() {
    let dev = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(1);
    let g = Gpu::new(dev).unwrap();
    let x = g.upload(&(0..32768).map(|i| ((i % 29) as f32 - 14.0) / 8.0).collect::<Vec<_>>()).unwrap();
    for (name, n, k) in [("wq_b 32768x1280", 32768, 1280), ("wo_b 5120x8192", 5120, 8192), ("shared 2304x5120", 2304, 5120), ("wq_a 1280x5120", 1280, 5120), ("wkv 512x5120", 512, 5120)] {
        let c = copies(n * k);
        let ws: Vec<_> = (0..c).map(|_| g.upload(&vec![0x38u8; n * k]).unwrap()).collect();
        let s = g.upload(&vec![127u8; n.div_ceil(32) * k.div_ceil(32)]).unwrap();
        let mut y = g.alloc::<f32>(n).unwrap();
        time(&g, &format!("gemv_fp8 + act_quant {name}"), n * k, 200, |i| {
            let xq = g.act_quant_fp8_to(&x.slice(..k)).unwrap();
            g.gemv_fp8(&xq.as_view(), &ws[i % c], &s, &mut y.slice_mut(..), n, k, 1, true).unwrap()
        });
        time(&g, &format!("gemv_fp8_token {name}"), n * k, 200, |i| {
            g.gemv_fp8_token(&x.slice(..k), &ws[i % c], &s, n, k, true, true).unwrap().unwrap();
        });
    }
    {
        let (n, k, groups) = (8192, 4096, 8);
        let c = copies(n * k * 2);
        let ws: Vec<_> = (0..c).map(|_| g.upload(&vec![0x3f80u16; n * k]).unwrap()).collect();
        let mut y = g.alloc::<f32>(n).unwrap();
        time(&g, "gemv_bf16 wo_a grouped 8192x4096", n * k * 2, 200, |i| {
            g.gemv_bf16(&x.slice(..), &ws[i % c], &mut y.slice_mut(..), n, k, 1, true, n / groups, groups * k).unwrap()
        });
    }
    {
        let recs: Vec<_> = (0..24).map(|_| g.upload(&vec![0x21u8; RECORD_BYTES]).unwrap()).collect();
        let tabs: Vec<_> = (0..4)
            .map(|q| {
                let ptrs: Vec<u64> = (0..6).map(|e| g.addr(&recs[q * 6 + e].as_view())).collect();
                g.upload(&Gpu::moe_table(&ptrs, &[0, 1, 2, 3, 4, 5], &[0.5; 6])).unwrap()
            })
            .collect();
        let mut h = g.alloc::<f32>(6 * INTER).unwrap();
        let mut outs = g.alloc::<f32>(6 * DIM).unwrap();
        time(&g, "moe_gate_up 6 experts", 6 * 2 * INTER * DIM / 2, 200, |i| g.moe_gate_up(&x.slice(..DIM), &tabs[i % 4], &mut h, 6, INTER, DIM, 10.0).unwrap());
        time(&g, "moe_down 6 experts", 6 * INTER * DIM / 2, 200, |i| g.moe_down(&h, &tabs[i % 4], &mut outs, 6, 6, INTER, DIM).unwrap());
    }
    {
        let n = 4 * DIM;
        let h = g.upload(&vec![0.25f32; n]).unwrap();
        let fnw = g.upload(&vec![0.01f32; 24 * n]).unwrap();
        let mut out = g.alloc::<f32>(25).unwrap();
        time(&g, "hc_project t=1", 24 * n * 4, 500, |_| g.hc_project(&h, &fnw, &mut out, n, 1).unwrap());
        let (base, scale) = (g.upload(&[0.1f32; 24]).unwrap(), g.upload(&[1.0f32; 3]).unwrap());
        let mut mix = g.alloc::<f32>(24).unwrap();
        time(&g, "hc_mix t=1", 0, 500, |_| g.hc_mix(&out, &base, &scale, &mut mix, 1, n, 1e-6, 20, 1e-6).unwrap());
    }
    {
        let mut xq = g.alloc::<f32>(DIM).unwrap();
        time(&g, "act_quant_fp8 5120 (launch floor)", 0, 500, |_| g.act_quant_fp8(&mut xq.slice_mut(..)).unwrap());
    }
}
