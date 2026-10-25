//! Timings and probes (ignored; run one with --ignored --nocapture).
use super::*;

/// A decode step's matmuls at DeepSeek-V4.1's GGUF shapes and types (its trunk's: the queries' low-rank and wide
/// projections, the grouped output's group and its second projection, the shared expert's, the head), each 28 in a
/// chain of one row, against its bytes.
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_deepseek_gguf_matmuls() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    for (dtype, n, k, block, bytes) in [
        (GgmlType::Q8_0, 1280usize, 5120usize, 32usize, 34usize),
        (GgmlType::Q5_K, 32768, 1280, 256, 176),
        (GgmlType::Q8_0, 512, 5120, 32, 34),
        (GgmlType::Q5_K, 1024, 4096, 256, 176),
        (GgmlType::Q6_K, 5120, 8192, 256, 210),
        (GgmlType::Q5_K, 2304, 5120, 256, 176),
        (GgmlType::Q8_0, 5120, 2304, 32, 34),
        (GgmlType::Q5_K, 129280, 5120, 256, 176),
    ] {
        let nbytes = n * (k / block) * bytes;
        let mut next = rng(n as u32);
        let mut raw = vec![0u8; nbytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
        let (x, y) = (b.vec(k), b.vec(n));
        DeviceChain::upload(&b, &x, &(0..k).map(|_| next()).collect::<Vec<_>>());
        let reps = if n > 100_000 { 4 } else { 28 };
        let run = || {
            let mut rec = b.begin();
            for _ in 0..reps {
                rec.matmul_rows(&w, &x, &y, 1);
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
        eprintln!("{dtype:?} [{n}, {k}] ({:.1} MB): {:.1} us a matmul in a chain, {:.0} GB/s", nbytes as f64 / 1e6, secs * 1e6, nbytes as f64 / secs / 1e9);
    }
}

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
    if !b.gpu.coop16() {
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
    // (OAIY_DN_HEADS=32: Qwen3.5 9B's 32 value heads; OAIY_DN_ROWS: a chunk that long)
    let nv: usize = std::env::var("OAIY_DN_HEADS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    let rows: usize = std::env::var("OAIY_DN_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let (nk, dim) = (16usize, 128usize);
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
    if !b.gpu.coop16() {
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
    if !b.gpu.coop16() {
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
    if !b.gpu.coop16() {
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
        let chosen = crate::quant::shaders::coop_splits((n as u32).div_ceil(128) * (m as u32).div_ceil(128), units, (k / 32) as u32);
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

/// Where [`crate::quant::shaders::coop_tiled`]'s time goes (`--ignored --nocapture`): Qwen3.8 27B's FFN gate for 512 tokens,
/// the kernel as it is, its decode replaced by constant stores, by no stores, and its multiply-adds taken out.
#[test]
#[ignore = "a measurement"]
fn measure_coop_parts() {
    use ggml_rs::ChainRecorder;
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.coop16() {
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
    let full = crate::quant::shaders::coop_tiled(GgmlType::Q3_K).unwrap();
    // (the loop's decode: its tile's place, then a block to the stores of its tokens)
    let loop_at = full.find("for (var b0 = w0").unwrap();
    let from = loop_at + full[loop_at..].find("let at4 = buf").unwrap();
    let step = &full[from..from + full[from..].find("        let xa = buf").unwrap()];
    let constants = full.replace(step, "let at4 = buf + lr * S4 + lh * 4u;\n        for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.5h); }\n");
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
    if !b.gpu.coop16() {
        return;
    }
    let (m, n, k) = (512usize, 34816usize, 5120usize);
    let mut next = rng(7);
    let raw: Vec<u8> = (0..n * (k / 256) * 110).map(|_| ((next() + 1.0) * 100.0) as u8).collect();
    let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
    let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("on the GPU");
    let x16 = b.vec(m * k / 2);
    let y = b.vec(m * n);
    let base = crate::quant::shaders::coop_tiled(GgmlType::Q3_K).expect("Q3_K's kernel");
    let marked = crate::quant::shaders::coop_tiled_marked(GgmlType::Q3_K).expect("Q3_K's kernel");
    // the loop's barrier (the prologue's kept)
    let step_end = "        xr3 = xn3;\n        workgroupBarrier();\n";
    let no_barrier = {
        let at = base.rfind(step_end).expect("the loop's barrier");
        format!("{}        xr3 = xn3;\n{}", &base[..at], &base[at + step_end.len()..])
    };
    // the loop's token loads (each step's from x16, two ahead) as the first's
    let loads = "        let xb = xo + b * xs;\n        xn0 = x16[xb];\n        xn1 = x16[xb + 1u];\n        xn2 = x16[xb + 2u];\n        xn3 = x16[xb + 3u];\n";
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
    if !b.gpu.coop16() {
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

/// What a replayed recording costs the host before its submission is in the queue (`--ignored --nocapture`): 100
/// dispatches (copies of 256 values, each into a vector of its own, their bind groups kept), recorded once and
/// replayed, each run's encoding and submit timed apart from its wait.
#[test]
#[ignore = "a measurement"]
fn measure_a_replays_encoding() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let x = b.vec(256);
    let many: Vec<DeviceVec> = (0..100).map(|_| b.vec(256)).collect();
    let mut first = crate::chain::Recorder::new(&b);
    first.keep_groups(true);
    for v in &many {
        first.copy(&x, 0, v, 0, 256);
    }
    let recipe = first.recipe().expect("a recording of dispatches alone");
    Box::new(first).launch().wait();
    let (mut launching, mut waiting) = (0.0, 0.0);
    let runs = 200;
    for _ in 0..runs {
        let mut rec = crate::chain::Recorder::new(&b);
        rec.keep_groups(true);
        rec.replay(recipe.clone());
        rec.read_range(&many[99], 0, 1);
        let t = std::time::Instant::now();
        let launched = Box::new(rec).launch();
        launching += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        launched.wait();
        waiting += t.elapsed().as_secs_f64();
    }
    eprintln!("100 dispatches replayed: encoded and submitted in {:.0} us ({:.2} us a dispatch), waited for {:.0} us", launching / runs as f64 * 1e6, launching / runs as f64 * 1e4, waiting / runs as f64 * 1e6);
}

/// What a matmul's loop reaches on the tensor cores with both its tiles in the workgroup's memory (`--ignored
/// --nocapture`): a workgroup's tile of rows by tokens, its subgroups' shares of it, the k step, each step's
/// fragments loaded from the tiles (filled once) and multiplied, a barrier a step; 2 workgroups an SM of 170.
#[test]
#[ignore = "a measurement"]
fn measure_coop_tiles() {
    use ggml_rs::ChainRecorder;
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.coop16() {
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
    if !b.gpu.coop16() {
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
        let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, joined: Vec::new(), places: Vec::new(), flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0, alike: false, settles: Vec::new(), moe_stage: None };
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
                let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, joined: Vec::new(), places: Vec::new(), flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0, alike: false, settles: Vec::new(), moe_stage: None };
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
            let Some(src) = crate::quant::shaders::rb_kernel_for_test(dtype, r, 1, ks) else { continue };
            let pipeline = b.gpu.named_pipeline(Box::leak(format!("test-rb-{dtype:?}-{r}-{ks}").into_boxed_str()), || src);
            let reps = 32;
            let run = || {
                let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, joined: Vec::new(), places: Vec::new(), flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0, alike: false, settles: Vec::new(), moe_stage: None };
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
    // (the rest a load at a time: the unrolled loop stops short of it, and every byte is counted)
    for (; i < n4; i += threads) {{ acc += s4[i]; }}
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
            let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), low_rank: [None, None, None], q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, joined: Vec::new(), places: Vec::new(), flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0, alike: false, settles: Vec::new(), moe_stage: None };
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

/// What a recording's dispatches cost at the least, 1 to 3,800 of them in one recording (a Flash-Next step is some
/// 1,250 dispatches and a check of four rows 1,650): copies of 256 values back and forth between two vectors (each
/// reads what the one before wrote, so none runs beside another), with the bind groups kept (a decode step's) and
/// made anew; then, kept, the same copy again and again (one pipeline and one bind group), copies from one vector
/// into 64 others in turn (none reads what another wrote), and a copy and an add in turn (two pipelines). Each
/// again held ([`ChainRecorder::hold`]: all of it encoded before any is submitted), its finish alone timed: the
/// submissions and the GPU's run of them, none of the recording (`--ignored --nocapture`). On an RTX 5090: some 4 us
/// a dispatch recorded as it runs (the CPU's recording and encoding: the wait at the end is next to nothing), and
/// held 1.9 to 2.8 us (a copy 2.8, an add 1).
#[test]
#[ignore = "a measurement"]
fn measure_the_dispatch_floor() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (x, y) = (b.vec(4096), b.vec(4096));
    let many: Vec<DeviceVec> = (0..64).map(|_| b.vec(4096)).collect();
    DeviceChain::upload(&b, &x, &vec![1.0; 4096]);
    for (what, keep) in [("back and forth", true), ("back and forth, groups made", false), ("the same copy", true), ("into 64 others", true), ("a copy and an add", true)] {
        for n in [1usize, 500, 1900, 3800] {
            let run = |held: bool| {
                let mut rec = Recorder::new(&b);
                ChainRecorder::keep_groups(&mut rec, keep);
                if held {
                    ChainRecorder::hold(&mut rec);
                }
                for i in 0..n {
                    match what {
                        "the same copy" => rec.copy(&x, 0, &y, 0, 256),
                        "into 64 others" => rec.copy(&x, 0, &many[i % 64], 0, 256),
                        "a copy and an add" if i % 2 == 1 => rec.add(&y, &x),
                        "a copy and an add" => rec.copy(&x, 0, &y, 0, 256),
                        _ if i % 2 == 0 => rec.copy(&x, 0, &y, 0, 256),
                        _ => rec.copy(&y, 0, &x, 0, 256),
                    }
                }
                rec.read_range(&x, 0, 1);
                let t = std::time::Instant::now();
                Box::new(rec).finish();
                t.elapsed().as_secs_f64() * 1e3
            };
            run(false);
            let _ = crate::profile::take_line();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run(false);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / 5.0;
            let line = crate::profile::take_line();
            run(true);
            let finish = (0..5).map(|_| run(true)).sum::<f64>() / 5.0;
            eprintln!("{n} dispatches, {what}: {ms:.2} ms a recording ({:.2} us a dispatch); held, its finish {finish:.2} ms ({:.2} us a dispatch); of 5: {line}", ms * 1e3 / n as f64, finish * 1e3 / n as f64);
        }
    }
}

/// What the system lets this process keep on the card (Vulkan's budget for its device-local heap), the card's size
/// and the weights' budget made of it (`--ignored --nocapture`): a model's weights, its caches and a prompt's scratch
/// past the first are moved off the card by the system, with no error and a prompt some three times as long.
#[test]
#[ignore = "a measurement"]
fn measure_the_cards_room() {
    let Ok(b) = WgpuBackend::new(None) else { return };
    let gib = |v: u64| v as f64 / (1u64 << 30) as f64;
    let (used, weights) = b.usage();
    eprintln!("{}: the system's budget for this process {:?} GiB (and its use of it), the weights' budget {:.2} GiB ({:.2} used)", b.adapter().name, b.memory_budget().map(|(budget, used)| (gib(budget), gib(used))), gib(weights), gib(used));
}

/// What an upload costs by its way (`--ignored --nocapture`): bytes of the host's written through a write's staging
/// memory on every core ([`crate::write_bytes`]) whole, by pieces of 64 MB waited for every fourth (as weights were
/// uploaded), and by pieces of 64, 32 and 16 MB each waited for; then as [`Gpu::write`] writes them and as
/// [`Gpu::write_with`] makes them in that staging in place. 64 MB to 1 GiB, each way three times (the best), the
/// upload waited for; and what each left there read back at its start, its end and its middle.
#[test]
#[ignore = "a measurement"]
fn measure_an_uploads_ways() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    for bytes in [64usize << 20, 128 << 20, 256 << 20, 512 << 20, 1 << 30] {
        let words: Vec<u32> = (0..bytes as u32 / 4).map(|i| i.wrapping_mul(2654435761).rotate_left(13) ^ 0x5bd1e995).collect();
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let fill = |at: usize, part: &mut [u8]| part.copy_from_slice(&data[at..at + part.len()]);
        eprintln!("{} MB:", bytes >> 20);
        // (a write's pieces: their size, and how many between waits)
        let ways: [(&str, usize, usize); 7] = [
            ("one write", bytes, 1),
            ("pieces of 64 MB, a wait every fourth", 64 << 20, 4),
            ("pieces of 64 MB, each waited for", 64 << 20, 1),
            ("pieces of 32 MB, each waited for", 32 << 20, 1),
            ("pieces of 16 MB, each waited for", 16 << 20, 1),
            ("as the backend writes", 0, 0),
            ("as the backend makes them in place", 0, 0),
        ];
        for (way, &(what, piece, every)) in ways.iter().enumerate() {
            let v = DeviceChain::vec(&b, bytes / 4);
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                match way {
                    0..=4 => {
                        for (i, part) in data.chunks(piece).enumerate() {
                            crate::write_bytes(b.gpu.queue(), buffer(&v), (i * piece) as u64, part);
                            if (i + 1) % every == 0 {
                                b.gpu.queue().submit([]);
                                b.gpu.wait(None);
                            }
                        }
                    }
                    5 => b.gpu.write(buffer(&v), 0, &data),
                    _ => assert!(b.gpu.write_with(buffer(&v), 0, bytes, &fill), "a write's staging memory"),
                }
                b.gpu.queue().submit([]);
                b.gpu.wait(None);
                best = best.min(t.elapsed().as_secs_f64());
            }
            let n = bytes / 4;
            for (from, len) in [(0usize, 1usize << 18), (n - (1 << 18), 1 << 18), ((n / 2).saturating_sub(1 << 17), 1 << 18)] {
                let mut rec = Recorder::new(&b);
                rec.read_range(&v, from, len);
                let got = Box::new(rec).finish().pop().expect("the words");
                assert!(got.iter().zip(&words[from..from + len]).all(|(g, w)| g.to_bits() == *w), "{what}: the words from {from}");
            }
            eprintln!("  {} MB {what}: {:.0} ms ({:.1} GB/s)", bytes >> 20, best * 1e3, bytes as f64 / best / 1e9);
        }
    }
}

/// What a kernel pays for a matrix it reads from the HOST's memory over the bus ([`WgpuBackend::host_vec_of_bytes`])
/// against one in the card's own (`--ignored --nocapture`): BF16 matrices of an expert's size (1.6 MB), a shared
/// expert's (6.5 MB) and a delta net's gate's (31 MB), one row, each dispatch another copy (256 MB of them in turn),
/// 100 dispatches in a held recording whose finish is timed. A card that has no room for all a model's experts could
/// read the ones it lacks so, with no word from the host between a layer's kernels.
#[test]
#[ignore = "a measurement"]
fn measure_a_matrix_in_the_hosts_memory() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    let mut r = rng(9);
    let times = 100usize;
    for (what, n, k) in [("an expert's size", 320usize, 2560usize), ("a shared expert's", 1280, 2560), ("a delta net's gate", 6144, 2560)] {
        let bytes: Vec<u8> = (0..n * k).flat_map(|_| (((r() * 0.05).to_bits() >> 16) as u16).to_le_bytes()).collect();
        let count = ((256usize << 20) / bytes.len()).clamp(8, 200);
        let Some(first) = b.host_vec_of_bytes(&bytes) else {
            eprintln!("no mappable storage buffers on this adapter");
            return;
        };
        let host: Vec<DeviceVec> = std::iter::once(first).chain((1..count).map(|_| b.host_vec_of_bytes(&bytes).expect("a host vector"))).collect();
        let card: Vec<DeviceVec> = (0..count).map(|_| b.vec_of_bytes(&bytes)).collect();
        let (x, y) = (b.vec(k), b.vec(n));
        DeviceChain::upload(&b, &x, &(0..k).map(|_| r()).collect::<Vec<_>>());
        let mut line = format!("{what} [{n}, {k}], {:.1} MB:", bytes.len() as f64 / 1e6);
        let mut sums = Vec::new();
        for (place, copies) in [("the card's", &card), ("the host's", &host), ("the card's again", &card), ("the host's again", &host)] {
            let at = std::cell::Cell::new(0usize);
            let run = || {
                let mut rec = Recorder::new(&b);
                ChainRecorder::hold(&mut rec);
                for _ in 0..times {
                    at.set(at.get() + 1);
                    rec.matmul_bf16_rows_f32(&copies[at.get() % copies.len()], n, k, &x, &y, 1);
                }
                rec.read_range(&y, 0, n);
                let t = std::time::Instant::now();
                let got = Box::new(rec).finish().pop().expect("the sums");
                (t.elapsed().as_secs_f64() * 1e6 / times as f64, got)
            };
            run();
            let (us, got) = (0..4).map(|_| run()).fold((f64::MAX, Vec::new()), |a, b| if b.0 < a.0 { b } else { a });
            line += &format!(" {place} {us:.1} us ({:.1} GB/s)", bytes.len() as f64 / us / 1e3);
            sums.push(got);
        }
        assert_eq!(sums[0], sums[1], "{what}: the host's memory's sums are the card's");
        eprintln!("{line}");
    }
}

/// What a step's row and a check's few cost Flash-Next's f16 matrices, a dispatch at a time (`--ignored --nocapture`):
/// its hyper-connections' down and up (alone and with their gates and mix), its router, its shared expert's gate and
/// up and its down, and a delta net's gate held as f16, each 200 times in a held recording whose finish is timed (the
/// GPU's run of them, none of the recording), for 1, 2, 4 and 8 rows. Each dispatch takes another copy of the matrix
/// (some 512 MB of them in turn, as a model's layers are each their own weights), then the same one every time (the
/// card's cache holds it: the kernel's own time, its weights' traffic apart). OAIY_F16_LANES sets the lanes a row.
#[test]
#[ignore = "a measurement"]
fn measure_a_few_rows_f16_matmuls() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    let mut r = rng(5);
    let times = 200usize;
    for (what, n, k) in [("hyper-connection down", 324usize, 10240usize), ("hyper-connection up", 10240, 324), ("router", 513, 2560), ("shared gate and up", 1280, 2560), ("shared down", 2560, 640), ("a delta net's gate", 6144, 2560)] {
        let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
        let copies: Vec<DeviceVec> = (0..((512usize << 20) / (n * k * 2)).clamp(8, 200)).map(|_| b.vec_f16(&w).expect("f16 values")).collect();
        let at = std::cell::Cell::new(0usize);
        let mut line = format!("{what} [{n}, {k}], {:.1} MB:", (n * k * 2) as f64 / 1e6);
        for (rows, cached) in [(1usize, false), (2, false), (4, false), (8, false), (1, true), (4, true)] {
            // the next copy, or the first every time
            let next = || {
                at.set(at.get() + 1);
                &copies[if cached { 0 } else { at.get() % copies.len() }]
            };
            let (x, y) = (b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &x, &(0..rows * k).map(|_| r()).collect::<Vec<_>>());
            let time = |f: &dyn Fn(&mut Recorder)| {
                let run = || {
                    let mut rec = Recorder::new(&b);
                    ChainRecorder::hold(&mut rec);
                    for _ in 0..times {
                        f(&mut rec);
                    }
                    rec.read_range(&y, 0, 1);
                    let t = std::time::Instant::now();
                    Box::new(rec).finish();
                    t.elapsed().as_secs_f64() * 1e6 / times as f64
                };
                run();
                (0..5).map(|_| run()).fold(f64::MAX, f64::min)
            };
            let us = time(&|rec| rec.matmul_f16_rows(next(), n, k, &x, &y, rows));
            line += &format!(" {rows} rows{} {us:.1} us", if cached { " of one copy" } else { "" });
            // the hyper-connections' fused kernels (4 streams of 2,560, a rank of 320 and 4 writes)
            if n == 324 {
                let post = b.vec(rows * 4);
                let us = time(&|rec| rec.hc_down_gates(next(), k, &x, &y, &post, rows, 320, 4, 4));
                line += &format!(" (with its gates {us:.1})");
            }
            if k == 324 {
                let (normed, out, logits) = (b.vec(rows * n), b.vec(rows * 2560), b.vec(rows * n));
                DeviceChain::upload(&b, &normed, &(0..rows * n).map(|_| r()).collect::<Vec<_>>());
                let us = time(&|rec| rec.hc_up_mix(next(), k, &x, &logits, &normed, &out, rows, 4, 2560));
                line += &format!(" (with its mix {us:.1})");
            }
        }
        eprintln!("{line}");
    }
}

/// A prompt's few rows (a conversation's next turn: 10 to 40 tokens) through each kernel that takes them, against as
/// many one-row steps (`--ignored --nocapture`): Q4_K at Qwen3.8 27B's widest and squarest.
#[test]
#[ignore = "a measurement"]
fn measure_few_rows() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    for (n, k) in [(17408usize, 5120usize), (5120, 5120)] {
        let (block, bytes) = (256usize, 144usize);
        let mut next = rng(n as u32);
        let mut raw = vec![0u8; n * (k / block) * bytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        for blk in raw.chunks_exact_mut(bytes) {
            let d = half::f16::from_f32(0.01 + (blk[4] as f32) * 1e-4).to_bits().to_le_bytes();
            let dmin = half::f16::from_f32(0.001 + (blk[5] as f32) * 1e-5).to_bits().to_le_bytes();
            blk[..4].copy_from_slice(&[d[0], d[1], dmin[0], dmin[1]]);
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q4_K));
        for m in [1usize, 4, 8, 12, 16, 20, 24, 32, 48] {
            let (x, y) = (b.vec(m * k), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
            let time = |how: u8| {
                let run = || {
                    let mut r = Recorder::new(&b);
                    for _ in 0..4 {
                        r.q8.clear();
                        r.x16.clear();
                        let ok = match how {
                            0 => r.matmul_rows_q8(&w, &x, &y, m),
                            1 => r.matmul_rows_coop(&w, &x, &y, m),
                            _ => {
                                r.matmul_rows_f32(&w, &x, &y, m);
                                true
                            }
                        };
                        if !ok {
                            return false;
                        }
                    }
                    r.read_range(&y, 0, 1);
                    Box::new(r).finish();
                    true
                };
                if !run() {
                    return f64::NAN;
                }
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                t.elapsed().as_secs_f64() / 12.0 * 1e3
            };
            let (q8, coop, f32) = (time(0), time(1), time(2));
            eprintln!("Q4_K [{n}, {k}] x {m:2}: int8 rows {q8:6.2} ms, tensor cores {coop:6.2} ms, f32 {f32:6.2} ms");
        }
    }
}

/// What a step's one row costs through the K-quants' int8 kernels at Qwen3.5 9B Q4_K_M's shapes (`--ignored
/// --nocapture`): its FFN's gate or up and down, its attention's and delta nets' squares, its delta nets' qkv, its
/// output; each weight alone (an NVIDIA A2's L2 is 2 MB, which none of them fits).
#[test]
#[ignore = "a measurement"]
fn measure_q8_rows_9b() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    let mut line = String::new();
    for (dtype, n, k, bytes) in [(GgmlType::Q4_K, 12288usize, 4096usize, 144usize), (GgmlType::Q4_K, 4096, 12288, 144), (GgmlType::Q4_K, 4096, 4096, 144), (GgmlType::Q5_K, 8192, 4096, 176), (GgmlType::Q5_K, 4096, 4096, 176), (GgmlType::Q6_K, 4096, 12288, 210), (GgmlType::Q6_K, 248320, 4096, 210)] {
        let raw: Vec<u8> = (0..n * (k / 256) * bytes).map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_shr(13) as u8 & 0x3b).collect();
        let nbytes = raw.len();
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
        // (OAIY_Q8_M: that many rows of x, a check's)
        let m: usize = std::env::var("OAIY_Q8_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let (x, y) = (b.vec(m * k), b.vec(m * n));
        let reps = 16;
        let run = || {
            let mut rq = Recorder::new(&b);
            for _ in 0..reps {
                rq.q8.clear();
                rq.matmul_rows_q8(&w, &x, &y, m);
            }
            rq.read_range(&y, 0, 1);
            Box::new(rq).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..4 {
            run();
        }
        let s = t.elapsed().as_secs_f64() / 4.0 / reps as f64;
        line += &format!("\n{dtype:?} [{n}, {k}]: {:.1} us, {:.0} GB/s", s * 1e6, nbytes as f64 / s / 1e9);
    }
    eprintln!("{line}");
}

/// A prompt's tensor-core matmul each way (`--ignored --nocapture`): [`crate::quant::shaders::coop_tiled`] (each subgroup
/// decodes, then multiplies) and [`crate::quant::shaders::coop_tiled_split`] (subgroups that decode apart from those that
/// multiply), on Qwen3.5 9B's FFN gate ([12288, 4096] by 512 tokens; OAIY_COOP_NK another `NxK`), each type's, and how
/// far apart their sums are (the worst element's, of their RMS: f16 windows' against f32 sums).
#[test]
#[ignore = "a measurement"]
fn measure_coop_kernels() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.coop16() || !b.gpu.subgroups32() {
        return;
    }
    let nk = std::env::var("OAIY_COOP_NK").unwrap_or_else(|_| "12288x4096".into());
    let (n, k): (usize, usize) = {
        let mut it = nk.split('x').map(|v| v.parse().expect("NxK"));
        (it.next().expect("N"), it.next().expect("K"))
    };
    let m: usize = std::env::var("OAIY_COOP_M").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let padded = m.div_ceil(128) * 128;
    let x16 = b.vec(padded * k / 2);
    let mut next = rng(11);
    let halves: Vec<f32> = (0..padded * k / 2).map(|_| f32::from_bits(half::f16::from_f32(next()).to_bits() as u32 | (half::f16::from_f32(next()).to_bits() as u32) << 16)).collect();
    DeviceChain::upload(&b, &x16, &halves);
    let y = b.vec(m * n);
    for (dtype, block, bytes) in [(GgmlType::Q4_K, 256usize, 144usize), (GgmlType::Q5_K, 256, 176), (GgmlType::Q6_K, 256, 210), (GgmlType::Q8_0, 32, 34), (GgmlType::Q3_K, 256, 110)] {
        // (bytes whose f16 scales are small and finite)
        let raw: Vec<u8> = (0..n * (k / block) * bytes).map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_shr(13) as u8 & 0x3b).collect();
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("on the GPU");
        let mut line = format!("{dtype:?} [{n}, {k}] by {m}:");
        let mut sums: Vec<Vec<f32>> = Vec::new();
        for (way, src) in [("together", crate::quant::shaders::coop_tiled(dtype).expect("its kernel")), ("apart", crate::quant::shaders::coop_tiled_split(dtype).expect("its kernel"))] {
            let name: &'static str = Box::leak(format!("bench-coop-{way}-{dtype:?}").into_boxed_str());
            let pipeline = b.gpu.named_pipeline(name, || src.clone());
            let dispatch = |rec: &mut Recorder| {
                for (chunk, row0, rows) in &q.chunks {
                    let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 1, 0];
                    rec.dispatch_kept(&pipeline, chunk, buffer(&x16), buffer(&y), &words, (rows.div_ceil(128), (m as u32).div_ceil(128), 1));
                }
            };
            let mut rec = Recorder::new(&b);
            dispatch(&mut rec);
            rec.read(&y);
            sums.push(Box::new(rec).finish().pop().expect("the sums"));
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    dispatch(&mut rec);
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
            line += &format!(" {way} {ms:.3} ms ({:.2} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
        }
        let rms = (sums[0].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / sums[0].len() as f64).sqrt();
        let worst = sums[1].iter().zip(&sums[0]).map(|(a, e)| ((a - e) as f64).abs()).fold(0.0, f64::max);
        eprintln!("{line}; apart's sums within {:.1e} of the RMS", worst / rms);
    }
}

/// A prompt's causal attention over a cache of halves each way (`--ignored --nocapture`): the one-pass kernel a (head,
/// 32 positions) a workgroup ([`attention_coop_one_rows`]) and the grouped one a (KV head, its query heads' positions)
/// ([`attention_coop_grouped`]), Qwen3.5 9B's heads (16 of 256 on 4; OAIY_ATT_NH, OAIY_ATT_NKV, OAIY_ATT_HD others) for
/// OAIY_ATT_ROWS (1,024) queries after OAIY_ATT_PAST (3,072), and how far apart their outputs are.
#[test]
#[ignore = "a measurement"]
fn measure_grouped_attention() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.coop16() {
        return;
    }
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n_h, n_kv, hd) = (env("OAIY_ATT_NH", 16), env("OAIY_ATT_NKV", 4), env("OAIY_ATT_HD", 256));
    let (rows, past) = (env("OAIY_ATT_ROWS", 1024), env("OAIY_ATT_PAST", 3072));
    let (kv_len, g) = (past + rows, n_h / n_kv);
    let (qs, row) = (n_h * hd, 2 * n_kv * hd);
    let mut next = rng(17);
    let mut halves = |n: usize| -> Vec<f32> { (0..n / 2).map(|_| f32::from_bits(half::f16::from_f32(next()).to_bits() as u32 | (half::f16::from_f32(next()).to_bits() as u32) << 16)).collect() };
    // the cache's rows as halves to 128 positions past its end, the queries' to 128 rows
    let (kvp, rp) = (kv_len.div_ceil(128) * 128, rows.div_ceil(128) * 128);
    let (kv, q16, out) = (b.vec(kvp * row / 2), b.vec(rp * qs / 2), b.vec(rp * qs));
    DeviceChain::upload(&b, &kv, &halves(kvp * row));
    DeviceChain::upload(&b, &q16, &halves(rp * qs));
    let words = [n_h as u32, n_kv as u32, past as u32, rows as u32, kv_len as u32, (1.0 / (hd as f32).sqrt()).to_bits(), 0, 0];
    let flops = 4.0 * (0..rows).map(|r| (past + r + 1) as f64).sum::<f64>() * qs as f64;
    let mut line = format!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} queries after {past}:");
    let mut outs: Vec<Vec<f32>> = Vec::new();
    for (way, src, grid) in [("one pass", attention_coop_one_rows(hd), (n_h as u32, rows.div_ceil(32) as u32, 1)), ("grouped", attention_coop_grouped(hd, g), (n_kv as u32, rows.div_ceil(64 / g) as u32, 1))] {
        let name: &'static str = Box::leak(format!("bench-attention-{way}-{hd}-{g}").into_boxed_str());
        let pipeline = b.gpu.named_pipeline(name, || src.clone());
        let mut rec = Recorder::new(&b);
        rec.dispatch_kept(&pipeline, buffer(&kv), buffer(&q16), buffer(&out), &words, grid);
        rec.read_range(&out, 0, rows * qs);
        outs.push(Box::new(rec).finish().pop().expect("the output"));
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..4 {
                rec.dispatch_kept(&pipeline, buffer(&kv), buffer(&q16), buffer(&out), &words, grid);
            }
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t.elapsed().as_secs_f64() / 12.0 * 1e3;
        line += &format!(" {way} {ms:.3} ms ({:.2} TFLOPS)", flops / ms / 1e9);
    }
    let rms = (outs[0].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / outs[0].len() as f64).sqrt();
    let worst = outs[1].iter().zip(&outs[0]).map(|(a, e)| ((a - e) as f64).abs()).fold(0.0, f64::max);
    eprintln!("{line}; grouped's within {:.1e} of the RMS", worst / rms);
}

/// A decode step's attention over a cache of halves each way (`--ignored --nocapture`): a thread a key
/// ([`attention_part_group_halved`]) and a subgroup's lanes a key ([`attention_part_group_lanes`]), Qwen3.5 9B's heads
/// (16 of 256 on 4; OAIY_ATT_NH, OAIY_ATT_NKV, OAIY_ATT_HD others) over OAIY_ATT_LEN positions (20,000), the parts and
/// their join; the cache's bytes a second, and how far apart the two are.
#[test]
#[ignore = "a measurement"]
fn measure_step_attention() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.subgroups32() {
        return;
    }
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n_h, n_kv, hd) = (env("OAIY_ATT_NH", 16), env("OAIY_ATT_NKV", 4), env("OAIY_ATT_HD", 256));
    let kv_len = env("OAIY_ATT_LEN", 20000);
    let (g, row, cap) = (n_h / n_kv, 2 * n_kv * hd, kv_len + 8);
    let mut next = rng(23);
    let q: Vec<f32> = (0..n_h * hd).map(|_| next()).collect();
    let halves: Vec<f32> = (0..cap * row / 2).map(|_| f32::from_bits(half::f16::from_f32(next()).to_bits() as u32 | (half::f16::from_f32(next()).to_bits() as u32) << 16)).collect();
    let (qv, kv) = (b.vec(n_h * hd), b.vec(cap * row / 2));
    DeviceChain::upload(&b, &qv, &q);
    DeviceChain::upload(&b, &kv, &halves);
    let runs = kv_len.div_ceil(SPLIT);
    let out = b.vec(n_h * hd + n_h * runs * (hd + 2));
    let scale = 1.0 / (hd as f32).sqrt();
    let mut line = format!("{n_h} heads ({n_kv} kv) {hd} wide over {kv_len}:");
    let mut outs: Vec<Vec<f32>> = Vec::new();
    for (way, src) in [("a thread a key", attention_part_group_halved(g)), ("lanes a key", attention_part_group_lanes(g, hd)), ("a key a pass", attention_part_group_lanes_by(g, hd, 32, 1))] {
        let name: &'static str = Box::leak(format!("bench-step-attention-{way}-{g}-{hd}").into_boxed_str());
        let part = b.gpu.named_pipeline(name, || src.clone());
        let record = |rec: &mut Recorder| {
            let params = rec.uniform(&[n_h as u32, n_kv as u32, hd as u32, kv_len as u32, 0, runs as u32, scale.to_bits()]);
            rec.dispatch(&part, buffer(&kv), buffer(&qv), buffer(&out), &params, (n_kv as u32, runs as u32, 1));
            let join = rec.named("chain-attention-join", ATTENTION_JOIN);
            rec.dispatch(&join, buffer(&kv), buffer(&qv), buffer(&out), &params, (n_h as u32, 1, 1));
        };
        let mut rec = Recorder::new(&b);
        record(&mut rec);
        rec.read_range(&out, 0, n_h * hd);
        outs.push(Box::new(rec).finish().pop().expect("the output"));
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..16 {
                record(&mut rec);
            }
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let s = t.elapsed().as_secs_f64() / 48.0;
        line += &format!(" {way} {:.0} us ({:.0} GB/s)", s * 1e6, (kv_len * row * 2) as f64 / s / 1e9);
    }
    {
        let run = || {
            let mut rec = Recorder::new(&b);
            for _ in 0..16 {
                let params = rec.uniform(&[n_h as u32, n_kv as u32, hd as u32, kv_len as u32, 0, runs as u32, scale.to_bits()]);
                let join = rec.named("chain-attention-join", ATTENTION_JOIN);
                rec.dispatch(&join, buffer(&kv), buffer(&qv), buffer(&out), &params, (n_h as u32, 1, 1));
            }
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        line += &format!(" join alone {:.0} us", t.elapsed().as_secs_f64() / 48.0 * 1e6);
    }
    let top = outs[0].iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = outs[1].iter().zip(&outs[0]).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
    eprintln!("{line}; lanes' within {:.1e} of the largest", worst / top);
}

/// What moving bytes between the host and the card costs by each way there is (`--ignored --nocapture`), for a device
/// with read-mappable storage buffers: 256 MB read back; 64 MB written by the CPU into a buffer made mapped (read-
/// mappable: wgpu's Vulkan backend gives it the host's memory) and into one mapped for writing (read- and
/// write-mappable: the card's memory, through its BAR), and into the host's own memory for scale; 1 GiB copied by a
/// kernel from each (its threads' reads many in flight); 1 GiB by the queue's write ([`Gpu::write`]); 1 GiB copied on
/// the GPU from staging in the host's memory. An NVIDIA A2 in a Thunderbolt 3 enclosure on an Ice Lake laptop: back
/// 0.8 GB/s; the CPU's writes into the host's buffer 0.07 (made in 333 ms, filled at 0.21: a buffer mapped at creation
/// that is not write-mappable is written through wgpu's staging memory, in the card's BAR), into the card's 0.2, into
/// its own memory 19; the kernel's copy from the host's buffer 2.2, from the card's 65; the GPU's copy 0.08 (its
/// staging made mapped too). The card's BAR, where wgpu's Vulkan backend puts every write-mappable buffer and the
/// queue's staging, takes the CPU's writes at 0.21 GB/s over the enclosure's link, where Vulkan's host-visible types
/// of the host's memory take them at 19 and 20: the queue's write was 0.21 (Qwen3.5 9B's 5.1 GB of weights in 27 s),
/// and through the host's memory ([`Gpu::host_staging`]) it is 2.1 (5.0 s).
#[test]
#[ignore = "a measurement"]
fn measure_upload_paths() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    let gpu = &b.gpu;
    if !gpu.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
        return;
    }
    let len = 1usize << 30;
    let piece = 64usize << 20;
    let data: Vec<u8> = (0..len).map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_shr(13) as u8).collect();
    let dst = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("measure-dst"), size: len as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let fill = |base: usize, from: &[u8]| {
        let each = from.len().div_ceil(8);
        std::thread::scope(|s| {
            for (i, part) in from.chunks(each).enumerate() {
                // SAFETY: each thread its own bytes of memory mapped for writing (or the host's own) past this scope
                s.spawn(move || unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), (base + i * each) as *mut u8, part.len()) });
            }
        });
    };
    let gb = |bytes: usize, secs: f64| bytes as f64 / secs / 1e9;
    let t = std::time::Instant::now();
    let got = gpu.read(&dst, 256 << 20);
    eprintln!("256 MB read back: {:.2} GB/s", gb(got.len(), t.elapsed().as_secs_f64()));
    // the host's buffer: made mapped, filled (twice: the first touch of its pages and the second apart), unmapped
    let t = std::time::Instant::now();
    let in_host = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("measure-host"), size: piece as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
    let made = t.elapsed().as_secs_f64();
    let (first, second) = {
        let mut view = in_host.slice(..).get_mapped_range_mut().expect("mapped at creation");
        let mut whole = view.slice(..);
        let at = whole.as_raw_ptr().cast::<u8>().as_ptr() as usize;
        let t = std::time::Instant::now();
        fill(at, &data[..piece]);
        let first = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        fill(at, &data[piece..2 * piece]);
        (first, t.elapsed().as_secs_f64())
    };
    let t = std::time::Instant::now();
    in_host.unmap();
    eprintln!("the host's 64 MB buffer: made in {:.0} ms, filled at {:.2} GB/s then {:.2}, unmapped in {:.0} ms", made * 1e3, gb(piece, first), gb(piece, second), t.elapsed().as_secs_f64() * 1e3);
    let host_fill = gb(piece, made + first + second);
    // the card's: mapped for writing, filled (the best of three)
    let in_card = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("measure-card"), size: piece as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let mut best = f64::MAX;
    for _ in 0..3 {
        in_card.slice(..).map_async(wgpu::MapMode::Write, |_| {});
        gpu.wait(None);
        let t = std::time::Instant::now();
        {
            let mut view = in_card.slice(..).get_mapped_range_mut().expect("mapped for writing");
            let mut whole = view.slice(..);
            fill(whole.as_raw_ptr().cast::<u8>().as_ptr() as usize, &data[..piece]);
        }
        best = best.min(t.elapsed().as_secs_f64());
        in_card.unmap();
    }
    let mut ram = vec![0u8; piece];
    fill(ram.as_mut_ptr() as usize, &data[..piece]);
    let t = std::time::Instant::now();
    fill(ram.as_mut_ptr() as usize, &data[..piece]);
    eprintln!("64 MB written by the CPU: into the host's buffer {host_fill:.2} GB/s, into the card's {:.2}, into the host's own memory {:.1}", gb(piece, best), gb(piece, t.elapsed().as_secs_f64()));
    const COPY: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> unused: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let threads = nw.x * 256u;
    for (var i = id.x; i < p[0].x; i += threads) { dst[p[0].y + i] = src[i]; }
}
"#;
    let pipeline = gpu.named_pipeline("measure-host-copy", || COPY.to_string());
    let dstv = DeviceVec { len: len / 4, inner: std::sync::Arc::new(dst.clone()) };
    for (what, from) in [("the host's buffer", &in_host), ("the card's", &in_card)] {
        let fromv = DeviceVec { len: piece / 4, inner: std::sync::Arc::new(from.clone()) };
        let t = std::time::Instant::now();
        let mut rec = Recorder::new(&b);
        for i in 0..len / piece {
            rec.dispatch_kept(&pipeline, buffer(&fromv), buffer(&fromv), buffer(&dstv), &[(piece / 4) as u32, (i * piece / 4) as u32], (640, 1, 1));
        }
        rec.read_range(&dstv, 0, 1);
        Box::new(rec).finish();
        eprintln!("1 GiB copied by a kernel from {what}: {:.2} GB/s", gb(len, t.elapsed().as_secs_f64()));
    }
    let t = std::time::Instant::now();
    gpu.write(&dst, 0, &data);
    gpu.queue().submit([]);
    gpu.wait(None);
    eprintln!("1 GiB by the queue's write: {:.2} GB/s", gb(len, t.elapsed().as_secs_f64()));
    let t = std::time::Instant::now();
    for (i, part) in data.chunks(piece).enumerate() {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("measure-staging"), size: part.len() as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true });
        {
            let mut view = staging.slice(..).get_mapped_range_mut().expect("mapped at creation");
            let mut whole = view.slice(..);
            fill(whole.as_raw_ptr().cast::<u8>().as_ptr() as usize, part);
        }
        staging.unmap();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&staging, 0, &dst, (i * piece) as u64, part.len() as u64);
        gpu.queue().submit([enc.finish()]);
        gpu.wait(None);
    }
    eprintln!("1 GiB copied on the GPU from staging in the host's memory: {:.2} GB/s", gb(len, t.elapsed().as_secs_f64()));
}

/// A wide single head's full attention without tensor cores (`--ignored --nocapture`): a VAE's mid-block attention,
/// `OAIY_ATT_ROWS` (16,384: SDXL's 1024 by 1024 picture's latent) positions of one head `OAIY_ATT_HD` (512) wide, by
/// the runs' kernel and by the tiled one.
#[test]
#[ignore = "a measurement"]
fn measure_wide_head_attention() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (rows, hd) = (env("OAIY_ATT_ROWS", 16384), env("OAIY_ATT_HD", 512));
    let mut next = rng(5);
    let q: Vec<f32> = (0..rows * hd).map(|_| next()).collect();
    let cache: Vec<f32> = (0..rows * 2 * hd).map(|_| next()).collect();
    let (qv, kv) = (b.vec(rows * hd), b.vec(rows * 2 * hd));
    DeviceChain::upload(&b, &qv, &q);
    DeviceChain::upload(&b, &kv, &cache);
    let out = b.vec(attention_runs_out_len(rows, 1, hd, rows));
    let scale = 1.0 / (hd as f32).sqrt();
    let flops = 4.0 * (rows * rows * hd) as f64;
    for tiled in [false, true] {
        let run = || {
            let mut rec = Recorder::new(&b);
            if tiled {
                rec.attention_rows_tiled(&qv, &kv, &out, rows, 1, 1, hd, rows, None, scale, true, rows);
            } else {
                rec.attention_rows_runs(&qv, &kv, &out, rows, 1, 1, hd, rows, None, scale, true);
            }
            rec.read_range(&out, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let t = std::time::Instant::now();
        run();
        let s = t.elapsed().as_secs_f64();
        eprintln!("{rows} positions of a head {hd} wide, {}: {:.1} ms ({:.2} TFLOPS)", if tiled { "tiled" } else { "runs" }, s * 1e3, flops / s / 1e12);
    }
}

/// A 3x3 convolution on the tensor cores at a VAE decoder's shapes (`--ignored --nocapture`; Qwen Image 2.1's at a
/// 1024x1024 picture: 1,152 channels at 128x128, 256x256's 576, 512x512's 288), against the f16 matmul of the same work
/// (its pixels the rows, nine taps' channels the inputs): what its implicit im2col costs. An NVIDIA A2 at 1,200 MHz:
/// the convolutions 4.05, 3.77 and 3.21 TFLOPS in [`crate::quant::shaders::coop_conv`]'s kernel, 4.41, 4.15 and 3.66 split
/// ([`crate::quant::shaders::coop_split_f16`]); the f16 matmul 4.47 and 4.98 at the first (where the Q4_K split kernel gives
/// 6.9: f16 weights are four times the bytes). `OAIY_COOP_SPLIT=0` for the first kernels.
#[test]
#[ignore = "a measurement"]
fn measure_conv3x3() {
    let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
    if !b.gpu.coop16() {
        return;
    }
    let mut next = rng(3);
    for (h, w, cin, cout) in [(128usize, 128usize, 1152usize, 1152usize), (256, 256, 576, 576), (512, 512, 288, 288)] {
        let m = h * w;
        let weights: Vec<f32> = (0..cout * cin * 9).map(|_| next() * 0.05).collect();
        let wv = DeviceChain::conv_weights(&b, &weights, cout, cin, 3).expect("conv weights");
        let bias = b.vec(cout);
        let (x, y) = (b.vec(m * cin), b.vec(m * cout));
        DeviceChain::upload(&b, &x, &(0..m * cin).map(|_| next()).collect::<Vec<f32>>());
        let k = 9 * cin;
        let w16 = DeviceChain::vec_f16_rounded(&b, &(0..cout * k).map(|_| next() * 0.05).collect::<Vec<f32>>()).unwrap();
        // (the matmul's rows the im2col's: 5.4 GB at 1024x1024's 144 channels, past a buffer; its first 256 rows'
        // worth of pixels timed and scaled there)
        let mm_rows = m.min((1usize << 28) / k);
        let x2 = b.vec(mm_rows * k);
        let flops = 2.0 * (m * cout * k) as f64;
        let time = |conv: bool| {
            let run = || {
                let mut rec = Recorder::new(&b);
                if conv {
                    rec.conv_rows(&wv, &bias, cout, cin, 3, &x, h, w, &y);
                } else {
                    rec.matmul_f16_rows(&w16, cout, k, &x2, &y, mm_rows);
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            t.elapsed().as_secs_f64() / 3.0
        };
        let (c, mm) = (time(true), time(false) * m as f64 / mm_rows as f64);
        eprintln!("{h}x{w}, {cin} to {cout}: the convolution {:.1} ms ({:.2} TFLOPS), the matmul {:.1} ms ({:.2})", c * 1e3, flops / c / 1e12, mm * 1e3, flops / mm / 1e12);
    }
}
