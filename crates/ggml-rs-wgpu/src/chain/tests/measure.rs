//! Timings and probes (ignored; run one with --ignored --nocapture).
use super::*;

/// What a chained one-row matmul costs on the GPU: 28 of one weight in a chain (a layer's worth of dispatches a
/// model's), at a 3B Llama's shapes, against the weight's bytes; and a chain of small ops alone. (The one-row kernel reads its weights at 170-280 GB/s,
/// whatever its lanes a row, its loads or its value array: a kernel of wide loads is what would change it.)
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_chained_matmuls() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    // a 3B Llama's attention and FFN, and Qwen3.8 27B's types: its FFN gate (Q3_K), qkv (Q5_K) and down (Q4_K)
    for (dtype, n, k, block, bytes) in [(GgmlType::Q4_K, 3072usize, 3072usize, 256usize, 144usize), (GgmlType::Q4_K, 16384, 3072, 256, 144), (GgmlType::Q6_K, 3072, 8192, 256, 210),
        (GgmlType::Q3_K, 17408, 5120, 256, 110), (GgmlType::Q5_K, 10240, 5120, 256, 176), (GgmlType::Q4_K, 5120, 17408, 256, 144)] {
        let nbytes = n * (k / block) * bytes;
        let mut next = rng(n as u32);
        let mut raw = vec![0u8; nbytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
        for m in [1usize, 2, 3, 4] {
            let (x, y) = (b.vec(m * k), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
            let reps = 28;
            let run = || {
                let mut rec = b.begin();
                for _ in 0..reps {
                    rec.matmul_rows(&w, &x, &y, m);
                }
                rec.read_range(&y, 0, 1);
                rec.finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let secs = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
            eprintln!("{dtype:?} [{n}, {k}] ({:.1} MB) x {m} rows: {:.1} us a matmul in a chain, {:.0} GB/s", nbytes as f64 / 1e6, secs * 1e6, nbytes as f64 / secs / 1e9);
        }
    }
    let (a, c) = (b.vec(3072), b.vec(3072));
    let run = || {
        let mut rec = b.begin();
        for _ in 0..28 * 8 {
            rec.add(&a, &c);
        }
        rec.read_range(&a, 0, 1);
        rec.finish();
    };
    run();
    let t = std::time::Instant::now();
    for _ in 0..5 {
        run();
    }
    eprintln!("an add of 3072 in a chain: {:.1} us", t.elapsed().as_secs_f64() / 5.0 / (28.0 * 8.0) * 1e6);
    // Qwen3.8 27B's gated delta net, a decode step's and a prompt chunk's, 48 layers' worth in a chain
    let (nv, nk, dim, kern) = (48usize, 16usize, 128usize, 4usize);
    let ch = 2 * nk * dim + nv * dim;
    for rows in [1usize, 6, 64] {
        let v = |n: usize| {
            let v = b.vec(n);
            DeviceChain::upload(&b, &v, &(0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>());
            v
        };
        let (qkv, z, ba, cw, a, dt, nm, conv, state) = (v(rows * ch), v(rows * nv * dim), v(rows * 2 * nv), v(ch * kern), v(nv), v(nv), v(dim), v((kern - 1) * ch), v(nv * dim * dim));
        let (co, out) = (b.vec(rows * ch), b.vec(rows * nv * dim));
        let d = DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: 0.088, eps: 1e-6, sigmoid_gate: false };
        for (what, conv_too) in [("conv", true), ("delta net", false)] {
            let run = || {
                let mut rec = b.begin();
                for _ in 0..48 {
                    if conv_too {
                        rec.ssm_conv(&qkv, &cw, &conv, &co, rows, ch, kern);
                    } else {
                        rec.delta_net(&co, &z, &ba, &a, &dt, &nm, &state, &out, d);
                    }
                }
                rec.read_range(&out, 0, 1);
                rec.finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            eprintln!("Qwen3.8 27B's {what} of {rows} rows: {:.1} us a layer", t.elapsed().as_secs_f64() / 5.0 / 48.0 * 1e6);
        }
    }
}

/// A prompt chunk's causal attention without tensor cores (`--ignored --nocapture`): the tiled kernel's against the
/// runs' for Qwen3.8 27B's chunk of 512 (24 heads of 256, 4 kv) and a 128-wide model's, the cache before it short
/// and long.
#[test]
#[ignore = "a measurement"]
fn measure_chunk_attention_f32() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    for (n_h, n_kv, hd, rows) in [(24usize, 4usize, 256usize, 512usize), (32, 8, 128, 512), (24, 4, 256, 128), (24, 4, 256, 64)] {
        for past in [0usize, 2048, 8192, 30000] {
            let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
            let mut next = rng((past + rows) as u32);
            let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
            DeviceChain::upload(&b, &qv, &(0..rows * qd).map(|_| next()).collect::<Vec<_>>());
            DeviceChain::upload(&b, &kv, &(0..kv_len * row).map(|_| next()).collect::<Vec<_>>());
            let scale = 1.0 / (hd as f32).sqrt();
            let time = |f: &dyn Fn(&mut Recorder)| {
                let mut rec = Recorder::new(&b);
                f(&mut rec);
                Box::new(rec).finish();
                let start = std::time::Instant::now();
                let mut rec = Recorder::new(&b);
                for _ in 0..4 {
                    f(&mut rec);
                }
                Box::new(rec).finish();
                start.elapsed().as_secs_f64() / 4.0
            };
            let out = b.vec(attention_runs_out_len(rows, n_h, hd, kv_len));
            let tiled = time(&|rec| rec.attention_rows_tiled(&qv, &kv, &out, rows, n_h, n_kv, hd, past, None, scale, false, 1 << 20));
            let runs = time(&|rec| rec.attention_rows_runs(&qv, &kv, &out, rows, n_h, n_kv, hd, past, None, scale, false));
            eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past}: tiled {:.2} ms, runs {:.2} ms", tiled * 1e3, runs * 1e3);
        }
    }
}

/// A prompt's attention without tensor cores (`--ignored --nocapture`): the tiled kernel's against the runs' at Qwen
/// Image's 1024x1024 (32 heads of 128, 4,096 queries over 4,200 positions), the tiled alone at LTX's 17,408.
#[test]
#[ignore = "a measurement"]
fn measure_tiled_attention() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    let (n_h, hd) = (32usize, 128usize);
    for (rows, kv_len, runs_too) in [(4096usize, 4200usize, true), (17408, 17408, false)] {
        let (qd, row) = (n_h * hd, 2 * n_h * hd);
        let mut next = rng(rows as u32);
        let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
        DeviceChain::upload(&b, &qv, &(0..rows * qd).map(|_| next()).collect::<Vec<_>>());
        DeviceChain::upload(&b, &kv, &(0..kv_len * row).map(|_| next()).collect::<Vec<_>>());
        let scale = 1.0 / (hd as f32).sqrt();
        let flops = 4.0 * rows as f64 * kv_len as f64 * qd as f64;
        let time = |f: &dyn Fn(&mut Recorder)| {
            let mut rec = Recorder::new(&b);
            f(&mut rec);
            Box::new(rec).finish();
            let start = std::time::Instant::now();
            let mut rec = Recorder::new(&b);
            for _ in 0..3 {
                f(&mut rec);
            }
            Box::new(rec).finish();
            start.elapsed().as_secs_f64() / 3.0
        };
        let out = b.vec(rows * qd);
        let tiled = time(&|rec| rec.attention_rows_f32_masked(&qv, &kv, &out, rows, n_h, n_h, hd, kv_len, None, scale, true));
        eprintln!("{rows} queries over {kv_len}: tiled {:.1} ms ({:.1} TFLOP/s)", tiled * 1e3, flops / tiled / 1e12);
        if runs_too {
            let out = b.vec(attention_runs_out_len(rows, n_h, hd, kv_len));
            let runs = time(&|rec| rec.attention_rows_runs(&qv, &kv, &out, rows, n_h, n_h, hd, kv_len, None, scale, true));
            eprintln!("{rows} queries over {kv_len}: runs {:.1} ms ({:.1} TFLOP/s)", runs * 1e3, flops / runs / 1e12);
        }
    }
}

/// A prompt's attention on the tensor cores (`--ignored --nocapture`): Qwen3.8 27B's (24 heads of 256, 4 kv) for a
/// chunk of 512 at several places, and what it does a second (its scores twice and its values once).
#[test]
#[ignore = "a measurement"]
fn measure_prompt_attention() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    // (OAIY_ATT_HD: heads that wide, as many more of them)
    let hd: usize = std::env::var("OAIY_ATT_HD").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let (n_h, n_kv) = (24 * 256 / hd, 4 * 256 / hd);
    // (OAIY_ATT_ROWS: the chunk that long)
    let rows: usize = std::env::var("OAIY_ATT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    for past in [0usize, 1024, 4096, 15360] {
        let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
        let mut next = rng(past as u32 + 5);
        let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
        DeviceChain::upload(&b, &qv, &(0..rows * qd).map(|_| next()).collect::<Vec<_>>());
        DeviceChain::upload(&b, &kv, &(0..kv_len * row).map(|_| next()).collect::<Vec<_>>());
        let out = b.vec(b.attention_rows_out_len(rows, n_h, hd, kv_len));
        let scale = 1.0 / (hd as f32).sqrt();
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..8 {
                assert!(rec.attention_rows_coop(&qv, &kv, &out, rows, n_h, n_kv, hd, past, None, scale));
            }
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
        // each query's keys up to its own: scores (twice) and values, 2 FLOPs a multiply-add
        let pairs: f64 = (0..rows).map(|r| (past + r + 1) as f64).sum();
        let flops = 3.0 * 2.0 * pairs * (n_h * hd) as f64;
        eprintln!("{rows} rows after {past}: {ms:.3} ms a layer ({:.0} TFLOPS)", flops / ms / 1e9);
    }
}

/// What a prompt's delta net takes (`--ignored --nocapture`): Qwen3.8 27B's (48 value heads on 16 key heads of 128)
/// for 512 tokens, as one kernel and in three passes.
#[test]
#[ignore = "a measurement"]
fn measure_delta_net_rows() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    let (nv, nk, dim, rows) = (48usize, 16usize, 128usize, 512usize);
    let ch = 2 * nk * dim + nv * dim;
    let mut next = rng(3);
    let mut up = |n: usize, s: f32| {
        let v = b.vec(n);
        DeviceChain::upload(&b, &v, &(0..n).map(|_| next() * s).collect::<Vec<f32>>());
        v
    };
    let (cv, z, ba, a, dt, nm, state, out) = (up(rows * ch, 1.0), up(rows * nv * dim, 1.0), up(rows * 2 * nv, 2.0), up(nv, 1.0), up(nv, 1.0), up(dim, 1.0), up(nv * dim * dim, 0.1), up(rows * nv * dim, 0.0));
    let d = DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: 1.0 / (dim as f32).sqrt(), eps: 1e-6, sigmoid_gate: false };
    let words = [nv as u32, nk as u32, dim as u32, dim as u32, rows as u32, d.scale_q.to_bits(), d.eps.to_bits(), 0];
    for three in [false, true] {
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..4 {
                if three {
                    rec.delta_net_rows(&cv, &z, &ba, &a, &dt, &nm, &state, &out, &d, &words);
                } else {
                    rec.dispatch_wide("chain-delta-net-128", &delta_net_one(dim), [buffer(&cv), buffer(&z), buffer(&ba), buffer(&a), buffer(&dt), buffer(&nm), buffer(&state), buffer(&out)], &words, (nv as u32, 1, 1));
                }
            }
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        eprintln!("{}: {:.3} ms a layer", if three { "three passes" } else { "one kernel" }, t.elapsed().as_secs_f64() / 12.0 * 1e3);
    }
}

/// Qwen3.8-Flash-Next's f16 matmuls of a chunk of 512 tokens (`--ignored --nocapture`): its hyper-connections' down
/// and up, its router and its delta nets' `ba`, through the f32 tiled kernel and on the tensor cores (split as
/// chosen and in 1 to 8).
#[test]
#[ignore = "a measurement"]
fn measure_f16_matmuls() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let m = 512usize;
    let mut r = rng(3);
    for (what, n, k) in [("hyper-connection down", 324usize, 10240usize), ("hyper-connection up", 10240, 324), ("router", 513, 2560), ("delta net ba", 96, 2560)] {
        let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
        let w16 = b.vec_f16(&w).expect("f16 values");
        let (x, y) = (b.vec(m * k), b.vec(m * n));
        DeviceChain::upload(&b, &x, &(0..m * k).map(|_| r()).collect::<Vec<_>>());
        let time = |f: &dyn Fn(&mut Recorder)| {
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    f(&mut rec);
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            t.elapsed().as_secs_f64() / 24.0 * 1e3
        };
        let ms = time(&|rec| rec.matmul_f16_tiled(&w16, n, k, &x, &y, m));
        let mut line = format!("{what} [{n}, {k}]: f32 tiled {ms:.3} ms ({:.1} TFLOPS); tensor cores", 2.0 * (m * n * k) as f64 / ms / 1e9);
        for split in [None, Some(1), Some(2), Some(4), Some(8)] {
            let ms = time(&|rec| {
                rec.wrote(buffer(&x));
                assert!(rec.matmul_f16_coop(&w16, n, k, &x, &y, m, split));
            });
            line += &format!(" {}: {ms:.3} ms ({:.0})", split.map_or("chosen".to_string(), |s| s.to_string()), 2.0 * (m * n * k) as f64 / ms / 1e9);
        }
        eprintln!("{line}");
    }
}

/// How far the tensor cores' sums (f16 a window of steps, then f32) are from the f32 kernel's (`--ignored
/// --nocapture`; OAIY_COOP_FOLD the window): the relative RMS error and the worst element's, Q3_K and Q6_K at the
/// 27B's widths, the tokens' rows as a model's (unit RMS, a few channels a hundred times the rest).
#[test]
#[ignore = "a measurement"]
fn measure_coop_fold_error() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    for (dtype, bytes, scales) in [(GgmlType::Q3_K, 110usize, &[108usize][..]), (GgmlType::Q6_K, 210, &[208][..])] {
        for k in [5120usize, 17408] {
            let (n, m) = (256usize, 256usize);
            let mut next = rng(k as u32 + bytes as u32);
            let mut raw = vec![0u8; n * (k / 256) * bytes];
            for v in raw.iter_mut() {
                *v = ((next() + 1.0) * 100.0) as u8;
            }
            for blk in raw.chunks_exact_mut(bytes) {
                for &at in scales {
                    let d = half::f16::from_f32(0.001 + (blk[(at + 4) % bytes] as f32) * 1e-5).to_bits().to_le_bytes();
                    blk[at] = d[0];
                    blk[at + 1] = d[1];
                }
            }
            let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
            let x: Vec<f32> = (0..m * k).map(|i| next() * 1.7 * if i % k % 997 == 3 { 100.0 } else { 1.0 }).collect();
            let (xv, y, yc) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &xv, &x);
            let mut rec = Recorder::new(&b);
            rec.matmul_rows_f32(&w, &xv, &y, m);
            rec.read(&y);
            let want = Box::new(rec).finish().pop().unwrap();
            let mut rec = Recorder::new(&b);
            assert!(rec.matmul_rows_coop(&w, &xv, &yc, m));
            rec.read(&yc);
            let got = Box::new(rec).finish().pop().unwrap();
            let err: f64 = got.iter().zip(&want).map(|(a, e)| ((*a - *e) as f64).powi(2)).sum::<f64>().sqrt();
            let norm: f64 = want.iter().map(|e| (*e as f64).powi(2)).sum::<f64>().sqrt();
            let rms = (norm * norm / want.len() as f64).sqrt();
            let worst = got.iter().zip(&want).map(|(a, e)| ((*a - *e) as f64).abs() / rms).fold(0.0, f64::max);
            eprintln!("{dtype:?} k {k}: relative error {:.2e}, the worst element's {:.2e} of the RMS ({rms:.3})", err / norm, worst);
        }
    }
}

/// What splitting a tensor-core matmul along k gains (`--ignored --nocapture`): Qwen3.8 27B's Q3_K matmuls of 512
/// tokens, each split as chosen and in 1 to 4.
#[test]
#[ignore = "a measurement"]
fn measure_coop_splits() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let m = 512usize;
    for (what, n, k) in [("FFN gate and up", 34816usize, 5120usize), ("FFN gate", 17408, 5120), ("FFN down", 5120, 17408), ("delta net qkv", 10240, 5120), ("delta net gate", 6144, 5120), ("attention q", 12288, 5120)] {
        let mut next = rng(7);
        // (OAIY_BENCH_DTYPE: the weights that type, Q3_K unless asked)
        let (dtype, block) = match std::env::var("OAIY_BENCH_DTYPE").as_deref() {
            Ok("Q4_K") => (GgmlType::Q4_K, 144),
            Ok("Q5_K") => (GgmlType::Q5_K, 176),
            Ok("Q6_K") => (GgmlType::Q6_K, 210),
            Ok("Q8_0") => (GgmlType::Q8_0, 272),
            _ => (GgmlType::Q3_K, 110),
        };
        let raw: Vec<u8> = (0..n * (k / 256) * block).map(|_| ((next() + 1.0) * 100.0) as u8).collect();
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
        let (x, y) = (b.vec(m * k), b.vec(m * n));
        let units = b.gpu.coop_units();
        let chosen = crate::shaders::coop_splits((n as u32).div_ceil(128) * (m as u32).div_ceil(128), units, (k / 32) as u32);
        // (OAIY_BENCH_SECONDS: the chosen split's matmul run that long, what it does each second of it: a card
        // under a power limit's rate at the limit, where the rest is a burst's)
        if let Some(secs) = std::env::var("OAIY_BENCH_SECONDS").ok().and_then(|v| v.parse::<f64>().ok()) {
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    rec.matmul_rows_coop_split(&w, &x, &y, m, None);
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let (t, mut done, mut said) = (std::time::Instant::now(), Vec::new(), String::new());
            while t.elapsed().as_secs_f64() < secs {
                run();
                done.push(t.elapsed().as_secs_f64());
            }
            for sec in 0..secs as usize {
                let runs = done.iter().filter(|&&at| at >= sec as f64 && at < sec as f64 + 1.0).count();
                said += &format!(" {:.0}", runs as f64 * 8.0 * 2.0 * (m * n * k) as f64 / 1e12);
            }
            eprintln!("{what} [{n}, {k}] for {secs} s, TFLOPS each second:{said}");
            continue;
        }
        let mut line = format!("{what} [{n}, {k}] (chosen {chosen}):");
        for split in [None, Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)] {
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    rec.matmul_rows_coop_split(&w, &x, &y, m, split);
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
            line += &format!(" {}: {ms:.3} ms ({:.0} TFLOPS)", split.map_or("chosen".to_string(), |s| s.to_string()), 2.0 * (m * n * k) as f64 / ms / 1e9);
        }
        eprintln!("{line}");
    }
}

/// Where [`crate::shaders::coop_tiled`]'s time goes (`--ignored --nocapture`): Qwen3.8 27B's FFN gate for 512 tokens,
/// the kernel as it is, its decode replaced by constant stores, by no stores, and its multiply-adds taken out.
#[test]
#[ignore = "a measurement"]
fn measure_coop_parts() {
    use ggml_rs::ChainRecorder;
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let (n, k, m) = (17408usize, 5120usize, 512usize);
    let mut next = rng(5);
    let mut raw = vec![0u8; n * (k / 256) * 110];
    for v in raw.iter_mut() {
        *v = ((next() + 1.0) * 100.0) as u8;
    }
    let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
    let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).unwrap();
    let (x16, y) = (b.vec(m * k / 2), b.vec(m * n));
    DeviceChain::upload(&b, &x16, &vec![f32::from_bits(0x3c003c00); m * k / 2]);
    let full = crate::shaders::coop_tiled(GgmlType::Q3_K).unwrap();
    let tail = "        } else {\n            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }\n        }";
    let step = &full[full.find("let at4 = buf").unwrap()..full.find(tail).unwrap() + tail.len()];
    let constants = full.replace(step, "let at4 = buf + lr * S4 + lh * 4u;\n        for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.5h); }");
    let nothing = full.replace(step, "");
    let mut no_mma = full.clone();
    for (c, a, bf) in [("c00", "a0", "b0f"), ("c01", "a0", "b1f"), ("c02", "a0", "b2f"), ("c03", "a0", "b3f"), ("c10", "a1", "b0f"), ("c11", "a1", "b1f"), ("c12", "a1", "b2f"), ("c13", "a1", "b3f")] {
        let mma = format!("{c} = coopMultiplyAdd({a}, {bf}, {c});");
        assert!(no_mma.contains(&mma), "{mma}");
        no_mma = no_mma.replace(&mma, "");
    }
    for (name, src) in [("bench-coop-full", full.clone()), ("bench-coop-constants", constants), ("bench-coop-nothing", nothing), ("bench-coop-no-mma", no_mma)] {
        let pipeline = b.gpu.named_pipeline(name, || src.clone());
        let (chunk, row0, rows) = &q.chunks[0];
        let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 1, 0];
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..4 {
                rec.dispatch_kept(&pipeline, chunk, buffer(&x16), buffer(&y), &words, (rows.div_ceil(128), (m as u32).div_ceil(128), 1));
            }
            rec.read_range(&y, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t.elapsed().as_secs_f64() / 12.0 * 1e3;
        eprintln!("{name}: {ms:.2} ms ({:.1} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
    }
}

/// The Q3_K tensor-core matmul and variants of it, each timed on FFN gate and up's shape (`--ignored --nocapture`):
/// as it is, and with a part of it taken out (results wrong, what that part costs): the step's barrier, its tokens'
/// loads, its decode, the f16 sums' folds.
#[test]
#[ignore = "a measurement"]
fn measure_coop_variants() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let (m, n, k) = (512usize, 34816usize, 5120usize);
    let mut next = rng(7);
    let raw: Vec<u8> = (0..n * (k / 256) * 110).map(|_| ((next() + 1.0) * 100.0) as u8).collect();
    let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
    let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("on the GPU");
    let x16 = b.vec(m * k / 2);
    let y = b.vec(m * n);
    let base = crate::shaders::coop_tiled(GgmlType::Q3_K).expect("Q3_K's kernel");
    let marked = crate::shaders::coop_tiled_marked(GgmlType::Q3_K).expect("Q3_K's kernel");
    // the loop's barrier (the prologue's kept)
    let step_end = "        xt[xa + 3u] = xr3;\n        workgroupBarrier();\n";
    let no_barrier = {
        let at = base.rfind(step_end).expect("the loop's barrier");
        format!("{}        xt[xa + 3u] = xr3;\n{}", &base[..at], &base[at + step_end.len()..])
    };
    // the loop's token loads (each step's from x16) as the first's
    let loads = "        let xb = xo + b * xs;\n        xr0 = x16[xb];\n        xr1 = x16[xb + 1u];\n        xr2 = x16[xb + 2u];\n        xr3 = x16[xb + 3u];\n";
    let no_x = {
        let at = base.rfind(loads).expect("the loop's loads");
        format!("{}{}", &base[..at], &base[at + loads.len()..])
    };
    let no_decode = {
        let mut out = String::new();
        let mut skipping = false;
        for line in marked.lines() {
            if line.contains("// DECODE BEGIN") {
                skipping = true;
            }
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
            if line.contains("// DECODE END") {
                skipping = false;
            }
        }
        out
    };
    let no_fold = base.replace("w0 += 32u", "w0 += 100000u").replace("w0 + 32u", "w0 + 100000u");
    for (name, src) in [("bench-coop-q3k", base.clone()), ("bench-coop-q3k-no-barrier", no_barrier), ("bench-coop-q3k-no-x-loads", no_x), ("bench-coop-q3k-no-decode", no_decode), ("bench-coop-q3k-no-fold", no_fold)] {
        let pipeline = b.gpu.named_pipeline(Box::leak(name.to_string().into_boxed_str()), || src.clone());
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..8 {
                for (chunk, row0, rows) in &q.chunks {
                    let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 1, 0];
                    rec.dispatch_kept(&pipeline, chunk, buffer(&x16), buffer(&y), &words, (rows.div_ceil(128), (m as u32).div_ceil(128), 1));
                }
            }
            rec.read_range(&y, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
        eprintln!("{name}: {ms:.3} ms ({:.0} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
    }
}

/// The tensor-core matmul's skeleton (no decode: the weights the first step's, the tokens loaded every step) in
/// shapes of a workgroup and its subgroups (`--ignored --nocapture`): what the loop alone runs at, f16 sums.
#[test]
#[ignore = "a measurement"]
fn measure_coop_skeletons() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let (m, n, k) = (512usize, 34816usize, 5120usize);
    let x16 = b.vec(m * k / 2);
    let w16 = b.vec(n * k / 2);
    let y = b.vec(m * n);
    // (warps down the rows, across the tokens; fragments each down, across): the tile 128 by 128 either way
    for (wr, wc, fr, fc) in [(4u32, 2u32, 2u32, 4u32), (2, 2, 4, 4), (2, 4, 4, 2), (4, 4, 2, 2)] {
        let threads = 32 * wr * wc;
        let mut src = String::new();
        src += "enable f16;\nenable wgpu_cooperative_matrix;\n";
        src += "struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, splits: u32, _pad1: u32, }\n";
        src += "@group(0) @binding(0) var<storage, read> w16: array<vec4<f16>>;\n@group(0) @binding(1) var<storage, read> x16: array<vec4<f16>>;\n@group(0) @binding(2) var<storage, read_write> y: array<f16>;\n@group(0) @binding(3) var<uniform> p: Params;\n";
        src += "const S4: u32 = 10u;\nconst BUF4: u32 = 1280u;\nvar<workgroup> wt: array<vec4<f16>, 2560>;\nvar<workgroup> xt: array<vec4<f16>, 2560>;\n";
        src += &format!("@compute @workgroup_size({threads})\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n");
        src += "    let r0 = wg.x * 128u;\n    let t0 = wg.y * 128u;\n    let sg = li / 32u;\n";
        src += &format!("    let sr = (sg % {wr}u) * {}u;\n    let st = (sg / {wr}u) * {}u;\n", 16 * fr, 16 * fc);
        // a thread's share of a step's 128 x 32 tokens (and, once, weights): `per` vec4s
        let per = 128 * 8 / threads;
        src += &format!("    let padded = ((p.m + 127u) / 128u) * 128u;\n    let xs = padded * 8u;\n");
        for r in 0..fr {
            for c in 0..fc {
                src += &format!("    var h{r}{c} = coop_mat16x16<f16, C>();\n");
            }
        }
        // the weights once (whatever is there), the first step's tokens
        src += &format!("    for (var e = li; e < 1024u; e += {threads}u) {{ let row = e / 8u; let q = e % 8u; wt[row * S4 + q] = w16[(r0 + row) * (p.k / 4u) + q]; wt[BUF4 + row * S4 + q] = w16[(r0 + row) * (p.k / 4u) + q]; xt[row * S4 + q] = x16[(t0 + row) * 8u + q]; }}\n");
        src += "    workgroupBarrier();\n    let all = p.k / 32u;\n";
        for v in 0..per {
            src += &format!("    var xr{v} = vec4<f16>();\n");
        }
        src += "    for (var b0 = 0u; b0 < all; b0++) {\n        let b = min(b0 + 1u, all - 1u);\n        let buf = ((b0 + 1u) % 2u) * BUF4;\n        let cur = (b0 % 2u) * BUF4;\n";
        for v in 0..per {
            src += &format!("        let e{v} = li + {}u;\n        xr{v} = x16[b * xs + (t0 + e{v} / 8u) * 8u + e{v} % 8u];\n", v * threads);
        }
        for kk in [0u32, 16] {
            src += "        {\n            let s10 = S4;\n";
            for r in 0..fr {
                src += &format!("            let ia{r} = cur + (sr + {}u) * S4 + {}u;\n            let a{r} = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia{r}], s10);\n", 16 * r, kk / 4);
            }
            for c in 0..fc {
                src += &format!("            let ib{c} = cur + (st + {}u) * S4 + {}u;\n            let b{c} = coopLoad<coop_mat16x16<f16, B>>(&xt[ib{c}], s10);\n", 16 * c, kk / 4);
            }
            for r in 0..fr {
                for c in 0..fc {
                    src += &format!("            h{r}{c} = coopMultiplyAdd(a{r}, b{c}, h{r}{c});\n");
                }
            }
            src += "        }\n";
        }
        for v in 0..per {
            src += &format!("        xt[buf + (e{v} / 8u) * S4 + e{v} % 8u] = xr{v};\n");
        }
        src += "        workgroupBarrier();\n    }\n";
        for r in 0..fr {
            for c in 0..fc {
                src += &format!("    {{\n        let o = (t0 + st + {}u) * p.n + r0 + sr + {}u;\n        let ns = p.n;\n        coopStore(h{r}{c}, &y[o], ns);\n    }}\n", 16 * c, 16 * r);
            }
        }
        src += "}\n";
        let name: &'static str = Box::leak(format!("bench-coop-skeleton-{wr}x{wc}-{fr}x{fc}").into_boxed_str());
        let pipeline = b.gpu.named_pipeline(name, || src.clone());
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..8 {
                let words = [k as u32, n as u32, m as u32, 0, n as u32, 0, 1, 0];
                rec.dispatch_kept(&pipeline, buffer(&w16), buffer(&x16), buffer(&y), &words, ((n as u32).div_ceil(128), (m as u32).div_ceil(128), 1));
            }
            rec.read_range(&y, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
        eprintln!("{name} ({threads} threads): {ms:.3} ms ({:.0} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
    }
}

/// How fast the host's bytes go up to each card and come back (`--ignored --nocapture`).
#[test]
#[ignore = "a measurement"]
fn measure_transfer_rates() {
    let Ok(b) = WgpuBackend::new(None) else { return };
    let others = b.others(None);
    for g in std::iter::once(&b).chain(&others) {
        let (up, down) = g.transfer_rates();
        let (again_up, again_down) = g.transfer_rates();
        eprintln!("{} at {}: up {up:.1} and {again_up:.1} GB/s, down {down:.1} and {again_down:.1} GB/s", g.adapter().name, g.adapter().pci_bus_id);
    }
}

/// What a dispatch costs of itself (`--ignored --nocapture`): 1,000 copies of 256 values, each reading what the
/// last wrote (a barrier between each two), and each into a vector of its own.
#[test]
#[ignore = "a measurement"]
fn measure_dispatch_overhead() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (x, y) = (b.vec(256), b.vec(256));
    let many: Vec<DeviceVec> = (0..1000).map(|_| b.vec(256)).collect();
    for dependent in [true, false] {
        let run = || {
            let mut rec = b.begin();
            rec.keep_groups(false);
            for i in 0..1000 {
                if dependent {
                    let (s, d) = if i % 2 == 0 { (&x, &y) } else { (&y, &x) };
                    rec.copy(s, 0, d, 0, 256);
                } else {
                    rec.copy(&x, 0, &many[i], 0, 256);
                }
            }
            rec.read_range(&x, 0, 1);
            rec.finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        eprintln!("1,000 copies, {}: {:.1} us a dispatch", if dependent { "each after the last" } else { "none after another" }, t.elapsed().as_secs_f64() / 3.0 / 1000.0 * 1e6);
    }
}

/// What a matmul's loop reaches on the tensor cores with both its tiles in the workgroup's memory (`--ignored
/// --nocapture`): a workgroup's tile of rows by tokens, its subgroups' shares of it, the k step, each step's
/// fragments loaded from the tiles (filled once) and multiplied, a barrier a step; 2 workgroups an SM of 170.
#[test]
#[ignore = "a measurement"]
fn measure_coop_tiles() {
    use ggml_rs::ChainRecorder;
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    // (rows, tokens of the workgroup's tile; subgroups down the rows, across the tokens; k a step)
    for (rows, tokens, wr, wt, ks) in [(128u32, 128u32, 4u32, 2u32, 64u32), (128, 128, 2, 2, 64), (128, 128, 2, 4, 64), (128, 128, 2, 2, 32), (256, 128, 4, 2, 32), (128, 256, 2, 4, 32), (128, 128, 4, 2, 32)] {
        let (fr, ft) = (rows / wr / 16, tokens / wt / 16);
        let threads = 32 * wr * wt;
        let stride4 = (ks + 8) / 4;
        let (a4, b4) = (rows * stride4, tokens * stride4);
        let mut body = String::new();
        for r in 0..fr {
            for t in 0..ft {
                body += &format!("    var c{r}_{t} = coop_mat16x16<f32, C>();\n");
            }
        }
        body += "    for (var it = 0u; it < p[0].x; it++) {\n        for (var kk = 0u; kk < KSu; kk += 16u) {\n            let s4 = STRIDE4u;\n";
        for r in 0..fr {
            body += &format!("            let ia{r} = (wr0 + {}u) * STRIDE4u + kk / 4u;\n            let a{r} = coopLoadT<coop_mat16x16<f16, A>>(&at[ia{r}], s4);\n", r * 16);
        }
        for t in 0..ft {
            body += &format!("            let ib{t} = (wt0 + {}u) * STRIDE4u + kk / 4u;\n            let b{t} = coopLoad<coop_mat16x16<f16, B>>(&bt[ib{t}], s4);\n", t * 16);
        }
        for r in 0..fr {
            for t in 0..ft {
                body += &format!("            c{r}_{t} = coopMultiplyAdd(a{r}, b{t}, c{r}_{t});\n");
            }
        }
        body += "        }\n        workgroupBarrier();\n    }\n";
        for r in 0..fr {
            for t in 0..ft {
                body += &format!("    {{\n        let o = ((wg.x * {threads}u / 32u + sg) * {} + {}u) * 256u;\n        coopStoreT(c{r}_{t}, &c[o], 16u);\n    }}\n", fr * ft, r * ft + t);
            }
        }
        let src = format!(
            "enable f16;\nenable wgpu_cooperative_matrix;\n@group(0) @binding(0) var<storage, read> a: array<f16>;\n@group(0) @binding(6) var<storage, read_write> c: array<f32>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\nvar<workgroup> at: array<vec4<f16>, {a4}>;\nvar<workgroup> bt: array<vec4<f16>, {b4}>;\n@compute @workgroup_size({threads})\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n    for (var i = li; i < {a4}u; i += {threads}u) {{ at[i] = vec4<f16>(f16(i % 7u) * 0.01h); }}\n    for (var i = li; i < {b4}u; i += {threads}u) {{ bt[i] = vec4<f16>(f16(i % 5u) * 0.01h); }}\n    workgroupBarrier();\n    let sg = li / 32u;\n    let wr0 = (sg % {wr}u) * {}u;\n    let wt0 = (sg / {wr}u) * {}u;\n{}}}\n",
            fr * 16,
            ft * 16,
            body.replace("KSu", &format!("{ks}u")).replace("STRIDE4u", &format!("{stride4}u"))
        );
        let groups = 340u32;
        let out = b.vec((groups * threads / 32 * fr * ft * 256) as usize);
        let a = b.vec(16);
        let iters = 4096u32 * 64 / ks;
        let name: &'static str = Box::leak(format!("bench-coop-tile-{rows}x{tokens}-{wr}x{wt}-{ks}").into_boxed_str());
        let run = || {
            let mut rec = Recorder::new(&b);
            let d = rec.gpu().dummy().clone();
            let drw = rec.gpu().dummy_rw().clone();
            rec.dispatch_wide(name, &src, [buffer(&a), &d, &d, &d, &d, &d, buffer(&out), &drw], &[iters], (groups, 1, 1));
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let secs = t.elapsed().as_secs_f64() / 3.0;
        let flops = groups as f64 * (rows * tokens) as f64 * (iters * ks) as f64 * 2.0;
        eprintln!("a tile of {rows}x{tokens}, subgroups {wr}x{wt} of {}x{} ({} fragments), k {ks} a step ({:.1} KB): {:.1} TFLOPS", fr * 16, ft * 16, fr * ft, (a4 + b4) as f64 * 8.0 / 1024.0, flops / secs / 1e12);
    }
}

/// The cooperative matrices each adapter offers through wgpu (`--ignored --nocapture`): their shapes and their
/// inputs' and sums' types. wgpu 30 names f32, f16, i32 and u32 only: a driver's 8-bit integer matrices (what
/// llama.cpp's CUDA backend multiplies K-quants with on these cards) are not among what it passes on.
#[test]
#[ignore = "a listing"]
fn list_cooperative_matrices() {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(desc.with_env());
    for adapter in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
        let info = adapter.get_info();
        let shapes = adapter.cooperative_matrix_properties();
        eprintln!("{} ({:?}): {} shapes", info.name, info.backend, shapes.len());
        for p in shapes.iter() {
            eprintln!("    {} x {} x {}: {:?} inputs, {:?} sums{}", p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, if p.saturating_accumulation { ", saturating" } else { "" });
        }
    }
}

/// What the tensor cores reach through cooperative matrices (`--ignored --nocapture`): each subgroup multiplying
/// 16x16 f16 fragments it holds into 8 accumulators, over and over (the arithmetic alone), its sums f32 and f16
/// (an RTX 5090: 244 and 485 TFLOPS).
#[test]
#[ignore = "a measurement"]
fn measure_cooperative_matrices() {
    use ggml_rs::ChainRecorder;
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    // the arithmetic alone, its sums f32 and f16
    for (acc, name) in [("f32", "bench-coop-alone"), ("f16", "bench-coop-alone-f16")] {
        let alone = format!(
            "enable f16;\nenable wgpu_cooperative_matrix;\n@group(0) @binding(0) var<storage, read> a: array<f16>;\n@group(0) @binding(6) var<storage, read_write> c: array<{acc}>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\n@compute @workgroup_size(128)\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n    let ma = coopLoadT<coop_mat16x16<f16, A>>(&a[0], 16u);\n    let mb = coopLoadT<coop_mat16x16<f16, B>>(&a[256], 16u);\n{}    for (var it = 0u; it < p[0].x; it++) {{\n{}    }}\n    let o = (wg.x * 4u + li / 32u) * 8u;\n{}}}\n",
            (0..8).map(|i| format!("    var c{i} = coop_mat16x16<{acc}, C>();\n")).collect::<String>(),
            (0..8).map(|i| format!("        c{i} = coopMultiplyAdd(ma, mb, c{i});\n")).collect::<String>(),
            (0..8).map(|i| format!("    coopStoreT(c{i}, &c[(o + {i}u) * 256u], 16u);\n")).collect::<String>()
        );
        let a = b.vec(256);
        DeviceChain::upload(&b, &a, &vec![f32::from_bits(0x3c003c00); 256]);
        let groups = 170 * 16;
        let out = b.vec(groups as usize * 4 * 8 * 256);
        let iters = 2048u32;
        let run = || {
            let mut rec = Recorder::new(&b);
            let d = rec.gpu().dummy().clone();
            let drw = rec.gpu().dummy_rw().clone();
            rec.dispatch_wide(name, &alone, [buffer(&a), &d, &d, &d, &d, &d, buffer(&out), &drw], &[iters], (groups, 1, 1));
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..5 {
            run();
        }
        let secs = t.elapsed().as_secs_f64() / 5.0;
        let flops = groups as f64 * 4.0 * 8.0 * iters as f64 * 2.0 * 4096.0;
        eprintln!("the arithmetic alone: {:.1} TFLOPS (f16 into {acc})", flops / secs / 1e12);
    }
}

/// What the GPU's arithmetic reaches in a kernel's registers (`--ignored --nocapture`): int8 dot products four at a
/// time (`dot4I8Packed`) against f32 multiply-adds, each thread 16 independent sums, a run's counts per second.
#[test]
#[ignore = "a measurement"]
fn measure_int8_and_f32_rates() {
    use ggml_rs::ChainRecorder;
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let body = |dp4a: bool| -> String {
        let (ty, zero, op) = if dp4a { ("i32", "0", "a{i} = a{i} + dot4I8Packed(u{i}, v);") } else { ("f32", "0.0", "a{i} = fma(f{i}, g, a{i});") };
        let decl: String = (0..16).map(|i| format!("    var a{i}: {ty} = {zero};\n    let u{i} = 0x01020304u + {i}u * 0x01010101u + t;\n    let f{i} = f32({i}) * 0.001 + f32(t) * 1e-7;\n")).collect();
        let ops: String = (0..16).map(|i| format!("        {}\n", op.replace("{i}", &i.to_string()))).collect();
        let sum: String = (0..16).map(|i| format!(" + f32(a{i})")).collect();
        format!(
            "@group(0) @binding(0) var<storage, read> unused: array<u32>;\n@group(0) @binding(6) var<storage, read_write> out: array<f32>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let t = id.x;\n{decl}    var v = 0x05060708u + t;\n    var g = 1.0001;\n    for (var it = 0u; it < p[0].x; it++) {{\n{ops}        v = v + 1u;\n        g = g * 0.99999;\n    }}\n    out[t] = 0.0{sum};\n}}\n"
        )
    };
    let out = b.vec(256 * 170 * 64);
    for dp4a in [true, false] {
        let src = body(dp4a);
        let name: &'static str = if dp4a { "bench-dp4a" } else { "bench-ffma" };
        let iters = 4096u32;
        let groups = 170 * 64;
        let run = || {
            let mut rec = crate::chain::Recorder::new(&b);
            let d = rec.gpu().dummy().clone();
            let drw = rec.gpu().dummy_rw().clone();
            rec.dispatch_wide(name, &src, [&d, &d, &d, &d, &d, &d, buffer(&out), &drw], &[iters], (groups, 1, 1));
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..5 {
            run();
        }
        let secs = t.elapsed().as_secs_f64() / 5.0;
        let ops = (groups as f64) * 256.0 * iters as f64 * 16.0;
        eprintln!("{}: {:.1} T a second ({:.1} T multiply-adds)", if dp4a { "dot4I8Packed" } else { "f32 fma" }, ops / secs / 1e12, ops * if dp4a { 4.0 } else { 1.0 } / secs / 1e12);
    }
}

/// Q3_K's int8 kernel against its f32 one (Qwen3.8 27B's FFN gate, [17408, 5120]): the same sums within int8's
/// rounding, and their times, 1 to 4 rows (`--ignored --nocapture`).
#[test]
#[ignore = "a measurement"]
fn measure_q8_matmuls() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    // the f16 scales' places in each type's block (d, and dmin where it has one)
    for (dtype, n, k, block, bytes, scales) in [(GgmlType::Q3_K, 17408usize, 5120usize, 256usize, 110usize, &[108usize][..]), (GgmlType::Q4_K, 5120, 17408, 256, 144, &[0, 2][..]), (GgmlType::Q5_K, 10240, 5120, 256, 176, &[0, 2][..]), (GgmlType::Q6_K, 5120, 6144, 256, 210, &[208][..]), (GgmlType::Q4_0, 17408, 5120, 32, 18, &[0][..])] {
    let nbytes = n * (k / block) * bytes;
    let mut next = rng(n as u32);
    let mut raw = vec![0u8; nbytes];
    for v in raw.iter_mut() {
        *v = ((next() + 1.0) * 100.0) as u8;
    }
    // sane block scales: small and finite
    for blk in raw.chunks_exact_mut(bytes) {
        for &at in scales {
            let d = half::f16::from_f32(0.01 + (blk[(at + 4) % bytes] as f32) * 1e-4).to_bits().to_le_bytes();
            blk[at] = d[0];
            blk[at + 1] = d[1];
        }
    }
    // eight matrices in turn (392 MB, past the L2's 96 MB), as a model's layers stream from memory
    let ws: Vec<_> = (0..8u8)
        .map(|i| {
            let mut r = raw.clone();
            for v in r.iter_mut().step_by(7) {
                *v = v.wrapping_add(i);
            }
            ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(r, vec![n, k], dtype))
        })
        .collect();
    let w = &ws[0];
    for m in [1usize, 2, 3, 4] {
        let (x, y, y8) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
        DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
        let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
        assert!(rq.matmul_rows_q8(w, &x, &y8, m));
        rq.read(&y8);
        let got = Box::new(rq).finish().pop().unwrap();
        let mut rec = b.begin();
        rec.matmul_rows(w, &x, &y, m);
        rec.read(&y);
        let want = rec.finish().pop().unwrap();
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        let worst = got.iter().zip(&want).map(|(a, e)| (a - e).abs() / scale).fold(0f32, f32::max);
        let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&got) * norm(&want));
        let reps = 28;
        let time = |q8: bool| {
            let run = || {
                let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
                for i in 0..reps {
                    let w = &ws[i % ws.len()];
                    if q8 {
                        rq.q8.clear();
                        rq.matmul_rows_q8(w, &x, &y8, m);
                    } else {
                        rq.matmul_rows(w, &x, &y, m);
                    }
                }
                rq.read_range(&y, 0, 1);
                Box::new(rq).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            t.elapsed().as_secs_f64() / 5.0 / reps as f64
        };
        let (f, q) = (time(false), time(true));
        eprintln!("{dtype:?} [{n}, {k}] x {m} rows: f32 {:.1} us ({:.0} GB/s), int8 {:.1} us ({:.0} GB/s; quantizing included); worst {worst:.2e} of the largest, cosine {cos:.6}", f * 1e6, nbytes as f64 / f / 1e9, q * 1e6, nbytes as f64 / q / 1e9);
        assert!(cos > 0.9999, "{dtype:?} x {m}: cosine {cos}");
    }
    }
}

/// A prompt's Q3_K matmul through the int8 tiled kernel against the f32 one (`--ignored --nocapture`): Qwen3.8 27B's
/// FFN gate [17408, 5120] and down [5120, 17408] for chunks of 512 and of 100 tokens, the results within int8's
/// rounding.
#[test]
#[ignore = "a measurement"]
fn measure_tiled_q8() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    for (n, k) in [(17408usize, 5120usize), (5120, 17408)] {
        let (block, bytes) = (256usize, 110usize);
        let mut next = rng(n as u32);
        let mut raw = vec![0u8; n * (k / block) * bytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        for blk in raw.chunks_exact_mut(bytes) {
            let d = half::f16::from_f32(0.01 + (blk[(108 + 4) % bytes] as f32) * 1e-4).to_bits().to_le_bytes();
            blk[108] = d[0];
            blk[109] = d[1];
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
        for m in [512usize, 100] {
            let (x, y, y8) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
            let mut rec = b.begin();
            rec.matmul_rows(&w, &x, &y, m);
            rec.read(&y);
            let want = rec.finish().pop().unwrap();
            let mut rq = Recorder::new(&b);
            assert!(rq.matmul_rows_tq8(&w, &x, &y8, m));
            rq.read(&y8);
            let got = Box::new(rq).finish().pop().unwrap();
            let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got) * norm(&want));
            let time = |q8: bool| {
                let run = || {
                    let mut rq = Recorder::new(&b);
                    for _ in 0..4 {
                        if q8 {
                            rq.q8.clear();
                            rq.matmul_rows_tq8(&w, &x, &y8, m);
                        } else {
                            rq.matmul_rows(&w, &x, &y, m);
                        }
                    }
                    rq.read_range(&y, 0, 1);
                    Box::new(rq).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                t.elapsed().as_secs_f64() / 12.0
            };
            let (f, q) = (time(false), time(true));
            let flops = 2.0 * (m * n * k) as f64;
            eprintln!("Q3_K [{n}, {k}] x {m}: f32 tiled {:.2} ms ({:.1} TFLOPS), int8 tiled {:.2} ms ({:.1}; quantizing included); cosine {cos:.6}", f * 1e3, flops / f / 1e12, q * 1e3, flops / q / 1e12);
            assert!(cos > 0.9999, "{n}x{k} of {m}: cosine {cos}");
            // the tensor cores
            let yc = b.vec(m * n);
            let mut rc = Recorder::new(&b);
            if rc.matmul_rows_coop(&w, &x, &yc, m) {
                rc.read(&yc);
                let got = Box::new(rc).finish().pop().unwrap();
                let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
                let cc = dot / (norm(&got) * norm(&want));
                let run = || {
                    let mut rc = Recorder::new(&b);
                    for _ in 0..4 {
                        rc.x16.clear();
                        rc.matmul_rows_coop(&w, &x, &yc, m);
                    }
                    rc.read_range(&yc, 0, 1);
                    Box::new(rc).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                let c = t.elapsed().as_secs_f64() / 12.0;
                eprintln!("    tensor cores {:.2} ms ({:.1} TFLOPS); cosine {cc:.6}", c * 1e3, flops / c / 1e12);
                assert!(cc > 0.9999, "{n}x{k} of {m} on the tensor cores: cosine {cc}");
            }
        }
    }
}

/// The one-row K-quant kernel's weight rows a lane, from memory (eight matrices in turn, past the L2): Qwen3.8 27B's
/// Q3_K FFN gate and Q4_K down, Q5_K qkv, Q6_K (`--ignored --nocapture`).
#[test]
#[ignore = "a measurement"]
fn measure_decode_rows_a_lane() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    for (dtype, n, k, bytes) in [(GgmlType::Q3_K, 17408usize, 5120usize, 112usize), (GgmlType::Q4_K, 5120, 17408, 144), (GgmlType::Q5_K, 10240, 5120, 176), (GgmlType::Q6_K, 5120, 6144, 210)] {
        let mut next = rng(n as u32 ^ k as u32);
        let raw_bytes = if dtype == GgmlType::Q3_K { 110 } else { bytes };
        let nbytes = n * (k / 256) * raw_bytes;
        let mut raw = vec![0u8; nbytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        let ws: Vec<_> = (0..8u8)
            .map(|i| {
                let mut r = raw.clone();
                for v in r.iter_mut().step_by(7) {
                    *v = v.wrapping_add(i);
                }
                ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(r, vec![n, k], dtype))
            })
            .collect();
        let (x, y) = (b.vec(k), b.vec(n));
        DeviceChain::upload(&b, &x, &(0..k).map(|_| next()).collect::<Vec<_>>());
        for (r, ks) in [(1u32, 1u32), (2, 1), (4, 1), (1, 2), (2, 2), (4, 2), (1, 4), (2, 4), (4, 4)] {
            let Some(src) = crate::shaders::rb_kernel_for_test(dtype, r, 1, ks) else { continue };
            let pipeline = b.gpu.named_pipeline(Box::leak(format!("test-rb-{dtype:?}-{r}-{ks}").into_boxed_str()), || src);
            let reps = 32;
            let run = || {
                let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
                for i in 0..reps {
                    let q = ws[i % ws.len()].device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).unwrap();
                    for (chunk, row0, rows) in &q.chunks {
                        let words = [k as u32, n as u32, 1, *row0, *rows, q.row_bytes as u32, 0, 0];
                        let groups = rows.div_ceil(4 * r);
                        rq.dispatch_kept(&pipeline, chunk, buffer(&x), buffer(&y), &words, (1, groups.min(65535), groups.div_ceil(65535)));
                    }
                }
                rq.read_range(&y, 0, 1);
                Box::new(rq).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let secs = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
            eprintln!("{dtype:?} [{n}, {k}] one row, {r} weight rows a lane, {ks} along k: {:.1} us, {:.0} GB/s", secs * 1e6, nbytes as f64 / secs / 1e9);
        }
    }
}

/// What reading memory reaches on this adapter through WebGPU: a kernel that sums 400 MB in vec4s, with 1, 2 and
/// 4 loads in flight a thread, at 4 and 8 warps a workgroup (`--ignored --nocapture`).
#[test]
#[ignore = "a measurement"]
fn measure_read_bandwidth() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    let len = 100usize << 20; // 400 MB of f32
    let src = b.vec(len);
    let out = b.vec(1 << 20);
    for (unroll, wg) in [(1u32, 128u32), (2, 128), (4, 128), (4, 256), (8, 256)] {
        let body = format!(
            r#"
@group(0) @binding(0) var<storage, read> s4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> unused: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let n4 = p[0].x;
    let threads = nw.x * {wg}u;
    var acc = vec4<f32>(0.0);
    var i = id.x;
    loop {{
        if (i + {last}u * threads >= n4) {{ break; }}
{loads}
        i += {unroll}u * threads;
    }}
    o[id.x % 1048576u] = acc.x + acc.y + acc.z + acc.w;
}}
"#,
            last = unroll - 1,
            loads = (0..unroll).map(|u| format!("        acc += s4[i + {u}u * threads];")).collect::<Vec<_>>().join("\n"),
        );
        let name: &'static str = Box::leak(format!("test-read-{unroll}-{wg}").into_boxed_str());
        let pipeline = b.gpu.named_pipeline(name, || body);
        let groups = 170 * 16;
        let run = || {
            let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
            for _ in 0..8 {
                rq.dispatch_kept(&pipeline, buffer(&src), buffer(&src), buffer(&out), &[(len / 4) as u32], (groups, 1, 1));
            }
            rq.read_range(&out, 0, 1);
            Box::new(rq).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let secs = t.elapsed().as_secs_f64() / 3.0 / 8.0;
        eprintln!("reading 400 MB, {unroll} loads in flight a thread, {wg} threads a workgroup: {:.0} GB/s", (len * 4) as f64 / secs / 1e9);
    }
}
