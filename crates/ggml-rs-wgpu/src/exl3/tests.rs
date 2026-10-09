//! `exl3`'s tests.

use super::*;
use ggml_rs::exl3::mul1;

/// An independent packing oracle: each weight's bits written MSB first into its tile's stream, and the dense
/// matrix they decode to (the same construction as ggml-rs-cuda's EXL3 test, which checks the CUDA kernels).
fn oracle(tw: usize, k: usize, n: usize) -> (Exl3Data, Vec<f32>) {
    let nw = tw / 2;
    let mut words = vec![0u32; k / 16 * n / 16 * nw];
    let mut dense = vec![0f32; k * n];
    for kt in 0..k / 16 {
        for nt in 0..n / 16 {
            let mut stream = vec![];
            for i in 0..256 {
                let bits = tw / 16 + if tw % 16 == 8 { i % 2 } else { 0 };
                let v = (i * 17 + kt * 43 + nt * 11 + 3) as u32 & ((1 << bits) - 1);
                for bit in (0..bits).rev() {
                    stream.push((v >> bit) & 1);
                }
            }
            let base = (kt * (n / 16) + nt) * nw;
            for (i, &bit) in stream.iter().enumerate() {
                words[base + i / 32] |= bit << (31 - i % 32);
            }
            let mut end = 0;
            for i in 0..256 {
                end += tw / 16 + if tw % 16 == 8 { i % 2 } else { 0 };
                let mut code = 0;
                for j in 0..16 {
                    code = (code << 1) | stream[(end + stream.len() - 16 + j) % stream.len()];
                }
                let lane = i / 8;
                let j = i % 8;
                let r = (lane % 4) * 2 + (j % 2) + if j & 2 != 0 { 8 } else { 0 };
                let c = lane / 4 + if j & 4 != 0 { 8 } else { 0 };
                dense[(kt * 16 + r) * n + nt * 16 + c] = mul1(code);
            }
        }
    }
    let data = Exl3Data {
        words,
        suh: (0..k).map(|i| if i % 3 == 0 { -0.5 } else { 0.5 }).collect(),
        svh: (0..n).map(|i| if i % 5 == 0 { -0.25 } else { 0.25 }).collect(),
        tile_words: tw,
        input_map: (0..k as u32).rev().collect(),
        output_map: (0..n as u32).rev().collect(),
    };
    (data, dense)
}

fn inputs(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k).map(|i| ((i % k * 7 + i / k * 13) % 23) as f32 / 32.0 - 0.25).collect()
}

/// The projection the oracle's dense matrix gives, step by step as exllamav3 rounds.
fn expected(data: &Exl3Data, dense: &[f32], x: &[f32]) -> Vec<f32> {
    let (k, n) = (data.suh.len(), data.svh.len());
    let mut out = vec![];
    for row in x.chunks_exact(k) {
        let mut xh: Vec<f32> = data.input_map.iter().enumerate().map(|(i, &j)| half(row[j as usize]) * data.suh[i]).collect();
        had(&mut xh);
        for v in &mut xh {
            *v = half(*v * ISQRT128);
        }
        let mut y: Vec<f32> = (0..n).map(|j| half((0..k).map(|i| xh[i] * dense[i * n + j]).sum())).collect();
        had(&mut y);
        out.extend(data.output_map.iter().map(|&j| half(y[j as usize] * ISQRT128 * data.svh[j as usize])));
    }
    out
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 0.003, "{what} [{i}]: {a} != {e}");
    }
}

const RATES: [usize; 11] = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128];

#[test]
fn the_cpu_projection_matches_the_independent_oracle_at_every_rate() {
    for tw in RATES {
        let (k, n) = (128, 256);
        let (data, dense) = oracle(tw, k, n);
        for i in 0..k {
            for j in 0..n {
                assert_eq!(data.value(i, j), dense[i * n + j], "the oracle and Exl3Data::value disagree: tw={tw} k={i} n={j}");
                let (w0, w1, sh) = positions(tw)[(i % 16) * 16 + j % 16];
                let base = ((i / 16) * (n / 16) + j / 16) * (tw / 2);
                let pair = ((data.words[base + w0] as u64) << 32) | data.words[base + w1] as u64;
                assert_eq!(decode((pair >> sh) as u32 & 0xffff), dense[i * n + j], "decode: tw={tw} k={i} n={j}");
            }
        }
        for rows in [1, 5] {
            let x = inputs(rows, k);
            let want = expected(&data, &dense, &x);
            let cpu = Exl3Cpu::new(Exl3Data { words: data.words.clone(), suh: data.suh.clone(), svh: data.svh.clone(), tile_words: tw, input_map: data.input_map.clone(), output_map: data.output_map.clone() }).unwrap();
            let y = cpu.linear(&Tensor::from_vec(x, vec![rows, k]));
            assert_eq!(y.shape(), &[rows, n]);
            close(y.data(), &want, &format!("cpu tw={tw} rows={rows}"));
        }
    }
}

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(Some(1 << 30)) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping WebGPU EXL3 tests: {e}");
            None
        }
    }
}

#[test]
fn the_gpu_projection_matches_the_independent_oracle_at_every_rate() {
    let Some(b) = backend() else { return };
    for tw in RATES {
        let (k, n) = (128, 256);
        let (data, dense) = oracle(tw, k, n);
        // More rows than one pass takes (32), and fewer.
        for rows in [1, 5, 37] {
            let x = inputs(rows, k);
            let want = expected(&data, &dense, &x);
            let w = b.exl3(Exl3Data { words: data.words.clone(), suh: data.suh.clone(), svh: data.svh.clone(), tile_words: tw, input_map: data.input_map.clone(), output_map: data.output_map.clone() }).unwrap();
            assert!(format!("{w:?}").starts_with("Exl3Gpu"), "on the GPU: {w:?}");
            let y = w.linear(&Tensor::from_vec(x, vec![rows, k]));
            close(y.data(), &want, &format!("gpu tw={tw} rows={rows}"));
        }
    }
}

#[test]
fn a_projection_over_several_buffers_and_splits_agrees_with_the_cpu() {
    let Some(b) = backend() else { return };
    // A model-like width, buffers of 3 tile rows (as a small binding limit would make them), split work.
    let (k, n, tw) = (1024, 512, 48);
    let mut seed = 13_234_567u32;
    let words: Vec<u32> = (0..k / 16 * n / 16 * (tw / 2))
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        })
        .collect();
    let data = || Exl3Data {
        words: words.clone(),
        suh: (0..k).map(|i| if i % 3 == 0 { -0.25 } else { 0.25 }).collect(),
        svh: vec![0.125; n],
        tile_words: tw,
        input_map: (0..k as u32).rev().collect(),
        output_map: (0..n as u32).rev().collect(),
    };
    let gpu = b.exl3_with(data(), Some(3)).unwrap();
    assert!(format!("{gpu:?}").contains("chunk(s)") && !format!("{gpu:?}").contains(" 1 chunk"), "{gpu:?}");
    let cpu = exl3_cpu(data()).unwrap();
    let x = Tensor::from_vec((0..3 * k).map(|i| ((i * 17 % 73) as f32 - 36.0) / 37.0).collect(), vec![3, k]);
    let (g, c) = (gpu.linear(&x), cpu.linear(&x));
    close(g.data(), c.data(), "gpu vs cpu");
}

fn random_exl3(k: usize, n: usize, tw: usize, seed: u32) -> Exl3Data {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    };
    let words = (0..k / 16 * n / 16 * (tw / 2)).map(|_| next()).collect();
    let mut scale = |len: usize, mag: f32| (0..len).map(|_| if next() & 1 == 0 { mag } else { -mag }).collect::<Vec<f32>>();
    Exl3Data { suh: scale(k, 0.25), svh: scale(n, 0.125), words, tile_words: tw, input_map: (0..k as u32).collect(), output_map: (0..n as u32).collect() }
}

/// `count` experts plus the shared one, each `[gate, up, down]`.
fn experts(count: usize, hidden: usize, ff: usize, tw: usize) -> Vec<[Exl3Data; 3]> {
    (0..=count as u32).map(|e| [random_exl3(hidden, ff, tw, 11 + e * 3), random_exl3(hidden, ff, tw, 12 + e * 3), random_exl3(ff, hidden, tw, 13 + e * 3)]).collect()
}

/// The MoE as its definition reads, from standalone projections: each row's top-k routed experts by logit (ties to
/// the lower index), softmax-weighted, plus the shared expert by the sigmoid of its gate.
fn reference(experts: Vec<[Exl3Data; 3]>, x: &[f32], logits: &[f32], top_k: usize) -> Vec<f32> {
    let p: Vec<[Exl3Cpu; 3]> = experts.into_iter().map(|[g, u, d]| [Exl3Cpu::new(g).unwrap(), Exl3Cpu::new(u).unwrap(), Exl3Cpu::new(d).unwrap()]).collect();
    let (h, width) = (p[0][0].t.k, p.len());
    let mut out = vec![];
    for (r, row) in x.chunks_exact(h).enumerate() {
        let l = &logits[r * width..(r + 1) * width];
        let mut idx: Vec<usize> = (0..width - 1).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let top = &idx[..top_k];
        let z: f32 = top.iter().map(|&e| (l[e] - l[top[0]]).exp()).sum();
        let mut picks: Vec<(usize, f32)> = top.iter().map(|&e| (e, (l[e] - l[top[0]]).exp() / z)).collect();
        picks.push((width - 1, 1.0 / (1.0 + (-l[width - 1]).exp())));
        let mut acc = vec![0f32; h];
        let xr = Tensor::from_vec(row.to_vec(), vec![1, h]);
        for (e, w) in picks {
            let g = p[e][0].linear(&xr);
            let u = p[e][1].linear(&xr);
            let hid: Vec<f32> = g.data().iter().zip(u.data()).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let f = hid.len();
            let d = p[e][2].linear(&Tensor::from_vec(hid, vec![1, f]));
            for (a, v) in acc.iter_mut().zip(d.data()) {
                *a += w * v;
            }
        }
        out.extend(acc);
    }
    out
}

#[test]
fn a_moe_layer_routes_and_mixes_as_its_definition_on_the_gpu_on_the_cpu_and_split_between_them() {
    let (count, hidden, ff, top_k) = (8, 256, 128, 3);
    for tw in [48, 80] {
        for rows in [1, 7] {
            let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
            let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29) as f32 - 14.0) / 7.0).collect();
            let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
            let xt = Tensor::from_vec(x.clone(), vec![rows, hidden]);
            let lt = Tensor::from_vec(logits.clone(), vec![rows, count + 1]);
            let cpu = exl3_experts_cpu(experts(count, hidden, ff, tw)).unwrap();
            close(cpu.forward(&xt, &lt, top_k).data(), &want, &format!("cpu moe tw={tw} rows={rows}"));
            let Some(b) = backend() else { continue };
            // the routed experts as groups (one shape and bitrate), their transforms on the GPU
            let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
            assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
            close(gpu.forward(&xt, &lt, top_k).data(), &want, &format!("grouped moe tw={tw} rows={rows}"));
            if rows == 1 {
                // chained, a step's kept scratch, twice
                let assign = vec![route(&logits, top_k)];
                let (xd, out) = (b.vec(hidden), b.vec(hidden));
                DeviceChain::upload(&b, &xd, &x);
                for _ in 0..2 {
                    let mut rec = b.begin();
                    rec.moe_rows(gpu.as_ref(), &xd, &out, &assign);
                    rec.read(&out);
                    close(&rec.finish().pop().unwrap(), &want, &format!("chained moe tw={tw}"));
                }
            }
            drop(gpu);
            // projections a dispatch each, as a layer whose experts differ in shape or bitrate goes
            std::env::set_var("OAIY_EXL3_UNGROUPED", "1");
            let host = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
            std::env::remove_var("OAIY_EXL3_UNGROUPED");
            assert!(format!("{host:?}").contains(&format!("{} of {} projections on the GPU", 3 * (count + 1), 3 * (count + 1))), "{host:?}");
            close(host.forward(&xt, &lt, top_k).data(), &want, &format!("gpu moe tw={tw} rows={rows}"));
            drop(host);
            // A budget for some of the experts: the rest decode on the CPU, and the layer is the same.
            let one = (hidden * ff * tw / 128) as u64;
            let Ok(small) = WgpuBackend::new(Some(one * 10)) else { continue };
            let split = small.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
            assert!(format!("{split:?}").contains("10 of 27 projections on the GPU"), "{split:?}");
            close(split.forward(&xt, &lt, top_k).data(), &want, &format!("split moe tw={tw} rows={rows}"));
        }
    }
}

#[test]
fn routing_takes_the_top_k_ties_to_the_lower_index_and_the_shared_expert_by_its_gate() {
    let r = route(&[1.0, 3.0, 3.0, 2.0, 0.0], 2);
    assert_eq!(r.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 4]);
    assert!((r[0].1 - 0.5).abs() < 1e-6 && (r[1].1 - 0.5).abs() < 1e-6, "{r:?}");
    assert!((r[2].1 - 0.5).abs() < 1e-6, "sigmoid(0) for the shared expert: {r:?}");
    let r = route(&[0.0, 0.0, 0.0, 5.0], 1);
    assert_eq!(r[0], (0, 1.0), "a three-way tie goes to the lowest index");
}

#[test]
fn the_cpu_projection_spreads_any_width_over_any_threads() {
    // 40 tile columns over 32 threads: two each, so the last threads have none (it panicked on a prompt's last row).
    let (k, n, tw) = (256, 640, 48);
    let data = random_exl3(k, n, tw, 7);
    let x: Vec<f32> = (0..9 * k).map(|i| ((i * 13 % 37) as f32 - 18.0) / 20.0).collect();
    let cpu = Exl3Cpu::new(data).unwrap();
    let mut xh = vec![0f32; 9 * k];
    for (row, out) in x.chunks_exact(k).zip(xh.chunks_exact_mut(k)) {
        cpu.t.pre(row, out);
    }
    let one = cpu.matmul(&xh, 9, 1);
    for threads in [2, 3, 7, 32, 64] {
        let many = cpu.matmul(&xh, 9, threads);
        close(&many, &one, &format!("{threads} threads"));
    }
}

/// How long a decode step's projection takes, and how much of it is waiting for the GPU (`--ignored --nocapture`).
#[test]
#[ignore = "timing"]
fn time_a_decode_projection() {
    let Some(b) = backend() else { return };
    for (k, n) in [(2560, 10240), (6144, 2560), (2560, 640)] {
        let w = b.exl3(random_exl3(k, n, 48, 5)).unwrap();
        let x = Tensor::from_vec(vec![0.1; k], vec![1, k]);
        for _ in 0..5 {
            w.linear(&x);
        }
        let t = std::time::Instant::now();
        for _ in 0..50 {
            w.linear(&x);
        }
        let each = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
        // The same with nothing to compute: an empty submit and wait.
        let t = std::time::Instant::now();
        for _ in 0..50 {
            b.gpu.queue().submit([]);
            b.gpu.wait(None);
        }
        let idle = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
        // As in a model: the host computes between calls (here 3 ms of spinning), so the GPU waits idle between them.
        let spin = |ms: f64| {
            let t = std::time::Instant::now();
            while t.elapsed().as_secs_f64() * 1000.0 < ms {
                std::hint::spin_loop();
            }
        };
        let mut gapped = 0.0;
        for _ in 0..50 {
            spin(3.0);
            let t = std::time::Instant::now();
            w.linear(&x);
            gapped += t.elapsed().as_secs_f64() * 1000.0;
        }
        // And with every other core busy, as a model's host threads keep them.
        let stop = std::sync::atomic::AtomicBool::new(false);
        let busy = std::thread::scope(|s| {
            for _ in 1..std::thread::available_parallelism().map_or(4, |n| n.get()) {
                s.spawn(|| {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                });
            }
            let t = std::time::Instant::now();
            for _ in 0..50 {
                w.linear(&x);
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            t.elapsed().as_secs_f64() * 1000.0 / 50.0
        });
        eprintln!("{k}x{n}: {each:.3} ms a call back to back; {:.3} ms after 3 ms of host work; {busy:.3} ms with every core busy; an empty submit and wait {idle:.3} ms", gapped / 50.0);
    }
}

#[test]
fn a_weight_beyond_the_budget_runs_on_the_cpu_and_the_budget_is_returned() {
    let Ok(b) = WgpuBackend::new(Some(4096)) else { return };
    let (data, _) = oracle(48, 128, 256);
    let w = b.exl3(data).unwrap();
    assert!(format!("{w:?}").starts_with("Exl3Cpu"), "{w:?}");
    assert_eq!(b.usage().0, 0);
    let Some(b) = backend() else { return };
    let (data, _) = oracle(48, 128, 256);
    let w = b.exl3(data).unwrap();
    assert!(b.usage().0 > 0);
    drop(w);
    assert_eq!(b.usage().0, 0, "dropping the weight returns its bytes");
}

/// A projection chained on the GPU, its transforms there too (the maps gathered, the Hadamard transforms and their
/// f16 roundings), gives the projection's own answer (its transforms on the host) bit for bit: one row (a step's,
/// its scratch kept), a few rows (a check of drafts: each row as one row alone), and a prompt's rows as its passes
/// take them (32 at a time, a last lone row as one row), with and without maps, at 3 and 5 bits.
#[test]
fn a_chained_projection_matches_the_projection() {
    let Some(b) = backend() else { return };
    let (k, n) = (512usize, 384usize);
    // (every rate a checkpoint may hold: a step's kernel is written for each, and at 7 bits a weight and more a
    // lane's last codes lie in its third and fourth words)
    assert_eq!(RATES, [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128]);
    for (tw, maps) in RATES.into_iter().zip([false, false, true, false, false, false, true, true, false, true, false]) {
        let mut data = random_exl3(k, n, tw, 77 + tw as u32);
        if maps {
            data.input_map = (0..k as u32).map(|i| (i * 7 + 3) % k as u32).collect();
            data.output_map = (0..n as u32).rev().collect();
        }
        let w = b.exl3(data).unwrap();
        assert!(b.holds_exl3(w.as_ref()), "the adapter holds it");
        for rows in [1usize, 2, 3, 5, 8, 9, 32, 33, 70, 129] {
            let xs: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect();
            let want = if (2..=FEW_MAX).contains(&rows) {
                let each: Vec<f32> = xs.chunks_exact(k).flat_map(|r| w.linear(&Tensor::from_vec(r.to_vec(), vec![1, k])).data().to_vec()).collect();
                Tensor::from_vec(each, vec![rows, n])
            } else {
                w.linear(&Tensor::from_vec(xs.clone(), vec![rows, k]))
            };
            let (x, y) = (b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &x, &xs);
            // twice: the second from the kept scratch
            for _ in 0..2 {
                let mut rec = b.begin();
                rec.exl3_rows(w.as_ref(), &x, &y, rows);
                rec.read(&y);
                let got = rec.finish().pop().unwrap();
                if rows > FEW_MAX && coop_on(&b.gpu) {
                    // a prompt's rows on the tensor cores: a matmul's sums (the same products, f16 exactly)
                    let dot: f64 = got.iter().zip(want.data()).map(|(a, e)| *a as f64 * *e as f64).sum();
                    let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                    let cos = dot / (norm(&got) * norm(want.data()));
                    let top = want.data().iter().fold(0f32, |m, v| m.max(v.abs()));
                    let worst = got.iter().zip(want.data()).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
                    assert!(cos > 0.99999 && worst <= 2e-3 * top, "tw={tw} maps={maps} rows={rows}: cosine {cos}, worst {worst} of {top}");
                    continue;
                }
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&got), bits(want.data()), "tw={tw} maps={maps} rows={rows}");
            }
        }
    }
}
/// A chained projection's time for a few rows (`--ignored --nocapture`): Qwen3.8-Flash-Next's delta nets' qkv, z and
/// output, its attention's q, and an expert's gate and down, at 3 bits, matrices in turn past the L2 as a model's
/// layers are.
#[test]
#[ignore = "a measurement"]
fn measure_chained_projections() {
    let Some(b) = backend() else { return };
    for (k, n, count) in [(2560usize, 10240usize, 8usize), (2560, 6144, 12), (6144, 2560, 12), (2560, 12288, 8), (2560, 640, 96), (640, 2560, 96)] {
        let ws: Vec<_> = (0..count as u32).map(|i| b.exl3(random_exl3(k, n, 48, 900 + i)).unwrap()).collect();
        let bytes = (k / 16 * n / 16 * 24 * 4) as f64;
        let mut line = format!("[{n}, {k}]:");
        let mut one = 0.0;
        for rows in [1usize, 2, 3, 4, 8] {
            let (x, y) = (b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &x, &(0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
            let reps = 4 * count;
            let run = || {
                let mut rec = b.begin();
                for i in 0..reps {
                    rec.exl3_rows(ws[i % count].as_ref(), &x, &y, rows);
                }
                rec.read_range(&y, 0, 1);
                rec.finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let each = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
            if rows == 1 {
                one = each;
            }
            line += &format!(" {rows} rows {:.1} us ({:.2}x, {:.0} GB/s);", each * 1e6, each / one, bytes / each / 1e9);
        }
        eprintln!("{line}");
    }
}

/// The few-rows kernel's cost by how many of a block's places hold jobs, against the one-job kernel's
/// (`--ignored --nocapture`): an expert's gate (640 x 2560) and a delta net's qkv (10240 x 2560) at 3 bits, matrices
/// in turn past the L2; and a dispatch of empty blocks.
#[test]
#[ignore = "a measurement"]
fn measure_few_blocks() {
    let Some(b) = backend() else { return };
    for (k, n, count) in [(2560usize, 640usize, 96usize), (2560, 10240, 8)] {
        let ws: Vec<_> = (0..count as u32).map(|i| b.exl3(random_exl3(k, n, 48, 700 + i)).unwrap()).collect();
        let g: Vec<&Exl3Gpu> = ws.iter().map(|w| w.as_any().unwrap().downcast_ref::<Exl3Gpu>().unwrap()).collect();
        let splits = g[0].single_chunk().unwrap().1;
        let (xh, part) = (b.vec(8 * k), b.vec(8 * splits as usize * n));
        DeviceChain::upload(&b, &xh, &(0..8 * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
        let jobs = u32_vec(&b, &(0..8u32).flat_map(|r| [0, r]).collect::<Vec<_>>());
        let ntiles = (n / 16) as u32;
        let reps = 4 * count;
        let time = |name: &'static str, body: &str, order: Option<&DeviceVec>, blocks: u32| {
            let run = || {
                use ggml_rs::ChainRecorder;
                let mut rec = crate::chain::Recorder::new(&b);
                let d = rec.gpu().dummy().clone();
                let drw = rec.gpu().dummy_rw().clone();
                let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().unwrap().clone();
                for i in 0..reps {
                    let (words, _) = g[i % count].single_chunk().unwrap();
                    let ob = order.map(buf).unwrap_or_else(|| d.clone());
                    rec.dispatch_wide(name, body, [words, &buf(&xh), &buf(&jobs), &ob, &d, &d, &buf(&part), &drw], &[n as u32, k as u32, 48, splits, 0, 0], (ntiles, 1, blocks * splits));
                }
                rec.read_range(&part, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            t.elapsed().as_secs_f64() / 5.0 / reps as f64 * 1e6
        };
        let one = time("exl3-mm", &chain_shader("mm"), None, 1);
        let mut line = format!("[{n}, {k}]: a job {one:.1} us;");
        for (rows, used) in [(4usize, 1usize), (4, 2), (4, 4), (2, 1), (2, 2), (8, 1)] {
            let order: Vec<u32> = (0..rows as u32).map(|r| if (r as usize) < used { r } else { NONE }).collect();
            let ov = u32_vec(&b, &order);
            line += &format!(" few-{rows} with {used} {:.1} us;", time(few_kernel(rows, 48).0, few_kernel(rows, 48).1, Some(&ov), 1));
        }
        let empty = u32_vec(&b, &vec![NONE; 4 * 40]);
        line += &format!(" 40 empty blocks of 4 {:.1} us", time(few_kernel(4, 48).0, few_kernel(4, 48).1, Some(&empty), 40));
        eprintln!("{line}");
    }
}

/// A check's experts as the GPU takes them in one pass (`--ignored --nocapture`): 4 rows of 10 of 128 experts
/// (Qwen3.8-Flash-Next's 2560 x 640 at 3 bits), their gate and up jobs a job a workgroup set, or grouped by matrix
/// in blocks of 4 (no empty blocks), as the rows share none, some or all of them.
#[test]
#[ignore = "a measurement"]
fn measure_grouped_check_experts() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, rows, k) = (128usize, 2560usize, 640usize, 4usize, 10usize);
    let moe = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
    let g = moe.as_any().unwrap().downcast_ref::<Exl3MoeGrouped>().unwrap();
    let st = g.scratch(&mut |n| b.vec(n), rows, k, true);
    let x = b.vec(rows * hidden);
    DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
    for (what, shared) in [("none shared", 0usize), ("3 of 10 shared", 3), ("all shared", 10)] {
        // row r's experts: the shared ones, then its own
        let picks: Vec<Vec<usize>> = (0..rows).map(|r| (0..k).map(|j| if j < shared { j } else { 10 + r * 25 + j }).collect()).collect();
        let jobs: Vec<u32> = picks.iter().enumerate().flat_map(|(r, p)| p.iter().flat_map(move |&e| [2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32])).collect();
        upload_u32(&b, &st.jobs_gu, &jobs);
        let n = jobs.len() / 2;
        // blocks of a matrix's jobs, as GROUP makes them
        let mut seen: Vec<(u32, Vec<u32>)> = Vec::new();
        for q in 0..n {
            let m = jobs[2 * q];
            match seen.iter_mut().find(|(mm, _)| *mm == m) {
                Some((_, list)) => list.push(q as u32),
                None => seen.push((m, vec![q as u32])),
            }
        }
        let order: Vec<u32> = seen.iter().flat_map(|(_, list)| (0..rows).map(|i| list.get(i).copied().unwrap_or(NONE))).collect();
        upload_u32(&b, &st.order_gu, &order);
        let blocks = seen.len();
        let time = |few: bool| {
            let run = || {
                use ggml_rs::ChainRecorder;
                let mut rec = crate::chain::Recorder::new(&b);
                for _ in 0..48 {
                    let o = if few { Order::Few(&st.order_gu, blocks, rows) } else { Order::Jobs };
                    g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, o, &st.xh_gu, &st.part_gu, &st.out_gu);
                }
                rec.read_range(&st.out_gu, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            t.elapsed().as_secs_f64() / 5.0 / 48.0 * 1e6
        };
        eprintln!("{what}: {n} jobs, {blocks} matrices: a job each {:.1} us a pass, grouped {:.1} us", time(false), time(true));
    }
}

/// What a prompt's grouped experts' gate and up matmul takes on the tensor cores (`--ignored --nocapture`, with
/// OAIY_CHAIN_PROFILE for each kernel's GPU time): 128 of Qwen3.8-Flash-Next's experts (2,560 by 640, 3 bits), 10,
/// 20 or 40 jobs each (a chunk of 512, 1,024 or 2,048 rows over its 512), in blocks of 16 to 128; then the same
/// blocks over 8 matrices (their words in the L2 throughout). On an RTX 5090 a block's time is its decode's, its
/// rows nearly free: 10 jobs in blocks of 32 0.47 ms, 20 0.51; but blocks of 64 or 128 0.83 and 0.94 for the same
/// blocks (their registers leave a workgroup an SM where 32's leave two: 32's padded to one an SM is 0.80), and the
/// words in the L2 0.42 where 0.50. Neither skipping a block's empty fragments' multiply-adds nor the decode's
/// conversion done in bits changed it.
#[test]
#[ignore = "a measurement"]
fn measure_prompt_expert_blocks() {
    let Some(b) = backend() else { return };
    if !coop_on(&b.gpu) {
        return;
    }
    let (count, hidden, ff, tw, top_k) = (128usize, 2560usize, 640usize, 48usize, 10usize);
    let moe = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    let g = moe.as_any().unwrap().downcast_ref::<Exl3MoeGrouped>().unwrap();
    for per in [10usize, 20, 40] {
        let rows = per * count / top_k;
        let st = g.scratch(&mut |n| b.vec(n), rows, top_k, false);
        let x = b.vec(rows * hidden);
        DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
        // row r's experts r k to r k + k - 1 (of the count, around): `per` rows each
        let jobs: Vec<u32> = (0..rows)
            .flat_map(|r| {
                (0..top_k).flat_map(move |j| {
                    let e = ((r * top_k + j) % count) as u32;
                    [2 * e, r as u32, 2 * e + 1, r as u32]
                })
            })
            .collect();
        upload_u32(&b, &st.jobs_gu, &jobs);
        let n = jobs.len() / 2;
        for bs in [16usize, 32, 64, 128] {
            let order = many_order(&jobs, bs);
            let ob = b.vec(order.len());
            upload_u32(&b, &ob, &order);
            let blocks = order.len() / bs;
            let run = || {
                let mut rec = crate::chain::Recorder::new(&b);
                for _ in 0..8 {
                    g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, Order::Many(&ob, blocks, bs), &st.xh_gu, &st.part_gu, &st.out_gu);
                }
                use ggml_rs::ChainRecorder;
                rec.read_range(&st.out_gu, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let _ = crate::profile::take_kernels();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
            let k = crate::profile::take_kernels();
            let mm: f64 = k.iter().filter(|e| e.0.starts_with("exl3-coop")).map(|e| e.1).sum::<f64>() / 24.0;
            eprintln!("{per} jobs an expert ({rows} rows), blocks of {bs} ({blocks}): a pass {ms:.3} ms, its matmul {mm:.3} ms on the GPU");
        }
    }
    // the same blocks (256 of 10 jobs in places for 32) over 4 experts' matrices (their words in the L2 throughout)
    let (rows, bs, blocks, per) = (256usize, 32usize, 256usize, 10usize);
    let st = g.scratch(&mut |n| b.vec(n), rows, top_k, false);
    let x = b.vec(rows * hidden);
    DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
    for (what, spread) in [("256 matrices", 256u32), ("8 matrices", 8)] {
        let jobs: Vec<u32> = (0..blocks * per).flat_map(|j| [((j / per) as u32) % spread, (j % rows) as u32]).collect();
        upload_u32(&b, &st.jobs_gu, &jobs);
        let order: Vec<u32> = (0..blocks).flat_map(|blk| (0..bs).map(move |i| if i < per { (blk * per + i) as u32 } else { NONE })).collect();
        let ob = b.vec(order.len());
        upload_u32(&b, &ob, &order);
        let n = blocks * per;
        let run = || {
            let mut rec = crate::chain::Recorder::new(&b);
            for _ in 0..8 {
                g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, Order::Many(&ob, blocks, bs), &st.xh_gu, &st.part_gu, &st.out_gu);
            }
            use ggml_rs::ChainRecorder;
            rec.read_range(&st.out_gu, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let _ = crate::profile::take_kernels();
        for _ in 0..3 {
            run();
        }
        let k = crate::profile::take_kernels();
        let mm: f64 = k.iter().filter(|e| e.0.starts_with("exl3-coop")).map(|e| e.1).sum::<f64>() / 24.0;
        eprintln!("{blocks} blocks of {per} jobs over {what}: the matmul {mm:.3} ms on the GPU");
    }
}

/// A prompt's rows through grouped experts, each expert's in blocks of 32 (a tile decoded once a block): an expert
/// with more rows than a block (two blocks), one with a single row, the rest a few each, as the definition gives.
#[test]
fn grouped_experts_take_a_prompts_rows_in_blocks_as_the_definition() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, rows) = (6, 256, 128, 2, 40);
    // expert 0 in the top two of rows 0..35, expert 5 only in row 39's, the others by a spread
    let logits: Vec<f32> = (0..rows)
        .flat_map(|r| {
            (0..=count).map(move |e| match e {
                0 if r < 35 => 5.0,
                5 if r == 39 => 5.0,
                5 => -5.0,
                e if e == count => 0.3,
                e => ((r * 7 + e * 3) % 11) as f32 / 11.0,
            })
        })
        .collect();
    let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
    let on = |e: usize| assign.iter().filter(|a| a[..top_k].iter().any(|p| p.0 == e)).count();
    assert!(on(0) > 32 && on(5) == 1, "expert 0 on {} rows, expert 5 on {}", on(0), on(5));
    let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    for tw in [48, 80] {
        let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
        let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
        assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
        let got = gpu.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k);
        close(got.data(), &want, &format!("grouped moe of a prompt tw={tw}"));
    }
}

#[test]
fn a_prompts_jobs_go_by_matrix_in_blocks_of_32() {
    // matrix 1 on 33 jobs, matrix 0 on 2, matrix 7 on 1: in their list's order within a matrix
    let mut jobs = vec![];
    for j in 0..36u32 {
        let m = match j {
            3 | 20 => 0,
            35 => 7,
            _ => 1,
        };
        jobs.extend([m, j]);
    }
    let order = many_order(&jobs, 32);
    assert_eq!(order.len(), 4 * 32, "a block for matrix 0, two for matrix 1, one for matrix 7");
    assert_eq!(&order[..3], &[3, 20, NONE]);
    assert!(order[2..32].iter().all(|&j| j == NONE));
    let ones: Vec<u32> = (0..35).filter(|&j| j != 3 && j != 20).collect();
    assert_eq!(&order[32..64], &ones[..32]);
    assert_eq!(&order[64..65], &ones[32..]);
    assert!(order[65..96].iter().all(|&j| j == NONE));
    assert_eq!(order[96], 35);
    assert!(order[97..].iter().all(|&j| j == NONE));
}

/// A prompt's experts routed and grouped by expert on the GPU (where the tensor cores take a prompt's blocks) give
/// what routing and grouping them on the host gives, and the MoE's definition: rows more than a check's, experts
/// on more rows than a block's 16 and on none, the sums into the streams as well.
#[test]
fn a_prompts_experts_routed_and_grouped_on_the_gpu_are_the_hosts() {
    let Some(b) = backend() else { return };
    if !coop_on(&b.gpu) {
        return;
    }
    let (count, hidden, ff, top_k) = (40, 256, 128, 6);
    let gpu = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
    for rows in [9usize, 40, 130] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        // expert 3 on every row, expert 39 on none, the rest spread
        let ls: Vec<f32> = (0..rows * (count + 1))
            .map(|i| match i % (count + 1) {
                3 => 4.0,
                39 => -9.0,
                e => (((i / (count + 1)) * 7 + e * 13) % 31) as f32 / 10.0 - 1.5,
            })
            .collect();
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
        let (xr, lr, on_gpu, on_host) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        let mut rec = b.begin();
        rec.keep_groups(false);
        assert!(rec.moe_routed(gpu.as_ref(), &xr, &on_gpu, &lr, top_k, rows), "{rows} rows route on the GPU");
        rec.moe_rows(gpu.as_ref(), &xr, &on_host, &assign);
        rec.read(&on_gpu);
        rec.read(&on_host);
        let mut got = rec.finish();
        let (h, g) = (got.pop().unwrap(), got.pop().unwrap());
        let scale = h.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        for (i, (a, e)) in g.iter().zip(&h).enumerate() {
            assert!((a - e).abs() <= 1e-5 * scale, "{rows} rows [{i}]: {a} against {e}");
        }
        close(&g, &reference(experts(count, hidden, ff, 48), &xs, &ls, top_k), &format!("{rows} rows routed on the GPU against the definition"));
    }
}

/// A step's experts routed on the GPU from the router's logits give what routing on the host gives: the same
/// experts (ties to the lower index; logits with few distinct values, so ties are common), their weights within an
/// ulp of exp; and the MoE's definition.
#[test]
fn experts_routed_on_the_gpu_are_the_hosts() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k) = (40, 256, 128, 6);
    let gpu = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    let xd = b.vec(hidden);
    DeviceChain::upload(&b, &xd, &x);
    let (on_host, on_gpu, ld) = (b.vec(hidden), b.vec(hidden), b.vec(count + 1));
    let mut seed = 7u32;
    for case in 0..12 {
        let levels = [4u32, 1 << 16][case % 2];
        let logits: Vec<f32> = (0..=count)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) % levels) as f32 / levels as f32 * 6.0 - 3.0
            })
            .collect();
        DeviceChain::upload(&b, &ld, &logits);
        for keep in [true, false] {
            let mut rec = b.begin();
            rec.keep_groups(keep);
            rec.moe_rows(gpu.as_ref(), &xd, &on_host, &[route(&logits, top_k)]);
            assert!(rec.moe_routed(gpu.as_ref(), &xd, &on_gpu, &ld, top_k, 1), "the grouped experts route on the GPU");
            rec.read(&on_host);
            rec.read(&on_gpu);
            let mut got = rec.finish();
            let (g, h) = (got.pop().unwrap(), got.pop().unwrap());
            let scale = h.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for (i, (a, e)) in g.iter().zip(&h).enumerate() {
                assert!((a - e).abs() <= 1e-5 * scale, "case {case} keep {keep} [{i}]: {a} against {e}");
            }
            if case == 0 {
                close(&g, &reference(experts(count, hidden, ff, 48), &x, &logits, top_k), "routed on the GPU against the definition");
            }
        }
    }
    // a check's few rows, each routed on its own: each row's sums its own step's
    for rows in [2usize, 3, 5] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        let ls: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 89) as f32 - 44.0) / 15.0).collect();
        let (xr, lr, yr) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        for keep in [true, false] {
            let mut rec = b.begin();
            rec.keep_groups(keep);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &yr, &lr, top_k, rows), "{rows} rows route on the GPU");
            rec.read(&yr);
            let got = rec.finish().pop().unwrap();
            for r in 0..rows {
                let (x1, l1, y1) = (b.vec(hidden), b.vec(count + 1), b.vec(hidden));
                DeviceChain::upload(&b, &x1, &xs[r * hidden..(r + 1) * hidden]);
                DeviceChain::upload(&b, &l1, &ls[r * (count + 1)..(r + 1) * (count + 1)]);
                let mut rec = b.begin();
                rec.keep_groups(keep);
                assert!(rec.moe_routed(gpu.as_ref(), &x1, &y1, &l1, top_k, 1));
                rec.read(&y1);
                let want = rec.finish().pop().unwrap();
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&got[r * hidden..(r + 1) * hidden]), bits(&want), "{rows} rows keep {keep}: row {r}");
            }
            // the sums added to each row's streams as a site's write-back adds them: the same bits as the sums,
            // then the write-back
            let streams = 3;
            let base: Vec<f32> = (0..rows * streams * hidden).map(|i| ((i * 13 % 47) as f32 - 23.0) / 9.0).collect();
            let post: Vec<f32> = (0..rows * streams).map(|i| (i as f32 - 2.0) / 3.0).collect();
            let (xa, xb, pd, sums) = (b.vec(base.len()), b.vec(base.len()), b.vec(post.len()), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xa, &base);
            DeviceChain::upload(&b, &xb, &base);
            DeviceChain::upload(&b, &pd, &post);
            let mut rec = b.begin();
            rec.keep_groups(keep);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &sums, &lr, top_k, rows));
            rec.stream_apply(&xa, &sums, &pd, rows, streams, hidden);
            assert!(rec.moe_routed_into(gpu.as_ref(), &xr, &xb, &pd, &lr, top_k, rows, streams));
            rec.read(&xa);
            rec.read(&xb);
            let mut got = rec.finish();
            let (into, apart) = (got.pop().unwrap(), got.pop().unwrap());
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
            assert_eq!(bits(&into), bits(&apart), "{rows} rows keep {keep}: into the streams");
        }
    }
}

/// A LoRA adapter's pairs for a layer of `count` routed experts and the shared one (the last), as `Experts::low_rank`
/// takes them: the experts `with` have one of `rank` beside a projection `k -> n` (a slot each, in that order), values
/// a seed makes, large beside the made experts' own outputs (so a wrong update shows).
fn lora_pairs(count: usize, with: &[usize], rank: usize, k: usize, n: usize, seed: u32) -> Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize)> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) % 2001) as f32 / 1000.0 - 1.0
    };
    let mut slot_of = vec![u32::MAX; count + 1];
    for (slot, &e) in with.iter().enumerate() {
        slot_of[e] = slot as u32;
    }
    let a = (0..with.len() * rank * k).map(|_| next() * 0.3).collect();
    let b = (0..with.len() * n * rank).map(|_| next() * 2.0).collect();
    Some((slot_of, a, b, rank))
}

type Pairs = [Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize)>; 3];

/// [`reference`] with low-rank updates beside the experts' gate, up and down projections (`lora[which]`): each
/// projection's output plus `b (a x)` of its input, the sums in f64.
fn reference_adapted(experts: Vec<[Exl3Data; 3]>, lora: &Pairs, x: &[f32], logits: &[f32], top_k: usize) -> Vec<f32> {
    let p: Vec<[Exl3Cpu; 3]> = experts.into_iter().map(|[g, u, d]| [Exl3Cpu::new(g).unwrap(), Exl3Cpu::new(u).unwrap(), Exl3Cpu::new(d).unwrap()]).collect();
    let (h, width) = (p[0][0].t.k, p.len());
    let update = |which: usize, e: usize, x: &[f32], y: &mut [f32]| {
        let Some((slot_of, a, b, rank)) = &lora[which] else { return };
        if slot_of[e] == u32::MAX {
            return;
        }
        let (s, k, n) = (slot_of[e] as usize, x.len(), y.len());
        let low: Vec<f64> = (0..*rank).map(|r| (0..k).map(|c| a[(s * rank + r) * k + c] as f64 * x[c] as f64).sum()).collect();
        for (i, yv) in y.iter_mut().enumerate() {
            *yv += (0..*rank).map(|r| b[(s * n + i) * rank + r] as f64 * low[r]).sum::<f64>() as f32;
        }
    };
    let mut out = vec![];
    for (r, row) in x.chunks_exact(h).enumerate() {
        let xr = Tensor::from_vec(row.to_vec(), vec![1, h]);
        let mut acc = vec![0f32; h];
        for (e, w) in route(&logits[r * width..(r + 1) * width], top_k) {
            let mut g = p[e][0].linear(&xr).data().to_vec();
            update(0, e, row, &mut g);
            let mut u = p[e][1].linear(&xr).data().to_vec();
            update(1, e, row, &mut u);
            let hid: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let f = hid.len();
            let mut d = p[e][2].linear(&Tensor::from_vec(hid.clone(), vec![1, f])).data().to_vec();
            update(2, e, &hid, &mut d);
            for (a, v) in acc.iter_mut().zip(&d) {
                *a += w * v;
            }
        }
        out.extend(acc);
    }
    out
}

/// A LoRA adapter's low-rank updates beside a layer's experts (`Experts::low_rank`) are the definition's sums: the
/// routed experts' in their groups' kernels and the shared expert's beside its projections on the GPU, and on the
/// host's path with its projections on the CPU or a dispatch each. A step's (its kept scratch, twice), a check's few
/// rows routed on the GPU (each row its own step's, to the bit), a prompt's rows by blocks; a gate and an up update of
/// different ranks in one group; experts with no update beside ones with; and an update whose B is zeros leaves the
/// layer as it was. Made experts and made pairs: 256 wide, 8 or 40 experts.
#[test]
fn a_low_rank_update_beside_the_experts_is_its_definition_on_the_gpu_and_the_host() {
    let (count, hidden, ff, top_k, tw) = (8, 256, 128, 3, 48);
    let lora: Pairs = [
        lora_pairs(count, &[2, count], 2, hidden, ff, 5),
        lora_pairs(count, &[2, 5], 4, hidden, ff, 6),
        lora_pairs(count, &[1, 3, 4, 6, count], 2, ff, hidden, 7),
    ];
    let none: Pairs = [None, None, None];
    // (by the sums' size: the made updates take them to tens, in terms that partly cancel, and the shared expert's
    // update goes through f16, its pairs as they are held and a prompt's rows as the tensor cores read them: a part in
    // 2,000 of a term at most. An update left out, or another expert's, is wrong by the sums' own size.)
    let near = |actual: &[f32], expected: &[f32], what: &str| {
        assert_eq!(actual.len(), expected.len(), "{what}");
        let size = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 0.003 + 1e-3 * size, "{what} [{i}]: {a} != {e} (sums up to {size})");
        }
    };
    let adapt = |e: &mut Box<dyn ggml_rs::exl3::Experts>, lora: &Pairs| {
        for (which, l) in lora.iter().enumerate() {
            let (s, a, b, r) = l.as_ref().unwrap();
            e.low_rank(which, s, a, b, *r).unwrap();
        }
    };
    for rows in [1usize, 7] {
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29) as f32 - 14.0) / 7.0).collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &x, &logits, top_k);
        let plain = reference_adapted(experts(count, hidden, ff, tw), &none, &x, &logits, top_k);
        close(&plain, &reference(experts(count, hidden, ff, tw), &x, &logits, top_k), "the definition with no update");
        let moved = want.iter().zip(&plain).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let size = plain.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(moved > 10.0 * size.max(0.01), "the updates are most of the layer's sums: they move them by {moved} at most, where the sums without are up to {size}");
        let xt = Tensor::from_vec(x.clone(), vec![rows, hidden]);
        let lt = Tensor::from_vec(logits.clone(), vec![rows, count + 1]);
        let mut cpu = exl3_experts_cpu(experts(count, hidden, ff, tw)).unwrap();
        adapt(&mut cpu, &lora);
        near(cpu.forward(&xt, &lt, top_k).data(), &want, &format!("adapted cpu moe rows={rows}"));
        let Some(b) = backend() else { continue };
        let mut gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
        assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
        adapt(&mut gpu, &lora);
        near(gpu.forward(&xt, &lt, top_k).data(), &want, &format!("adapted grouped moe rows={rows}"));
        if rows == 1 {
            let assign = vec![route(&logits, top_k)];
            let (xd, out) = (b.vec(hidden), b.vec(hidden));
            DeviceChain::upload(&b, &xd, &x);
            for _ in 0..2 {
                let mut rec = b.begin();
                rec.moe_rows(gpu.as_ref(), &xd, &out, &assign);
                rec.read(&out);
                near(&rec.finish().pop().unwrap(), &want, "adapted chained moe");
            }
        }
        drop(gpu);
        // each projection a dispatch of its own on the GPU, the sums the host's
        let each: Vec<[Proj; 3]> = experts(count, hidden, ff, tw).into_iter().map(|[g, u, d]| [b.proj(g, 0).unwrap(), b.proj(u, 0).unwrap(), b.proj(d, 0).unwrap()]).collect();
        let mut host: Box<dyn ggml_rs::exl3::Experts> = Box::new(Exl3MoeHost::new(each, Some((Arc::clone(&b.gpu), Arc::clone(&b.serial)))).unwrap());
        adapt(&mut host, &lora);
        near(host.forward(&xt, &lt, top_k).data(), &want, &format!("adapted gpu moe rows={rows}"));
    }
    let Some(b) = backend() else { return };
    // routed on the GPU: a step, and a check's rows, each row its own step's bits; kept scratch and not
    let (count, top_k) = (40, 6);
    let lora: Pairs = [
        lora_pairs(count, &[3, 7, 21, count], 2, hidden, ff, 15),
        lora_pairs(count, &[3, 8], 4, hidden, ff, 16),
        lora_pairs(count, &[0, 3, 5, 9, 17, 22, 31, 39, count], 2, ff, hidden, 17),
    ];
    let mut gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    adapt(&mut gpu, &lora);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    let width = count + 1;
    for rows in [1usize, 2, 5] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        // (expert 3, which has every update, on every row)
        let ls: Vec<f32> = (0..rows * width).map(|i| if i % width == 3 { 4.0 } else { ((i * 53 % 89) as f32 - 44.0) / 15.0 }).collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &xs, &ls, top_k);
        let (xr, lr, yr) = (b.vec(rows * hidden), b.vec(rows * width), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        for keep in [true, false] {
            let began = std::time::Instant::now();
            let mut rec = b.begin();
            rec.keep_groups(keep);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &yr, &lr, top_k, rows), "{rows} rows route on the GPU");
            rec.read(&yr);
            let got = rec.finish().pop().unwrap();
            eprintln!("    adapted experts routed on the GPU: {rows} rows, keep {keep}: {:.2} ms", began.elapsed().as_secs_f64() * 1e3);
            near(&got, &want, &format!("adapted {rows} rows routed on the GPU, keep {keep}"));
            for r in 0..rows {
                let (x1, l1, y1) = (b.vec(hidden), b.vec(width), b.vec(hidden));
                DeviceChain::upload(&b, &x1, &xs[r * hidden..(r + 1) * hidden]);
                DeviceChain::upload(&b, &l1, &ls[r * width..(r + 1) * width]);
                let mut rec = b.begin();
                rec.keep_groups(keep);
                assert!(rec.moe_routed(gpu.as_ref(), &x1, &y1, &l1, top_k, 1));
                rec.read(&y1);
                assert_eq!(bits(&got[r * hidden..(r + 1) * hidden]), bits(&rec.finish().pop().unwrap()), "adapted {rows} rows keep {keep}: row {r} is its step");
            }
        }
    }
    // a prompt's rows: routed and grouped by expert on the GPU where the tensor cores take them, and the host's routing
    for rows in [9usize, 40, 130] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        let ls: Vec<f32> = (0..rows * width)
            .map(|i| match i % width {
                3 => 4.0,
                39 => -9.0,
                e => (((i / width) * 7 + e * 13) % 31) as f32 / 10.0 - 1.5,
            })
            .collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &xs, &ls, top_k);
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * width..(r + 1) * width], top_k)).collect();
        let (xr, lr, on_gpu, on_host) = (b.vec(rows * hidden), b.vec(rows * width), b.vec(rows * hidden), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        let began = std::time::Instant::now();
        let mut rec = b.begin();
        rec.keep_groups(false);
        let routed = coop_on(&b.gpu) && rec.moe_routed(gpu.as_ref(), &xr, &on_gpu, &lr, top_k, rows);
        rec.moe_rows(gpu.as_ref(), &xr, &on_host, &assign);
        if routed {
            rec.read(&on_gpu);
        }
        rec.read(&on_host);
        let mut got = rec.finish();
        eprintln!("    adapted experts of a prompt's {rows} rows (routed on the GPU: {routed}): {:.2} ms", began.elapsed().as_secs_f64() * 1e3);
        near(&got.pop().unwrap(), &want, &format!("adapted {rows} rows routed on the host"));
        if routed {
            near(&got.pop().unwrap(), &want, &format!("adapted {rows} rows routed on the GPU"));
        }
    }
    // an update whose B is zeros adds nothing
    let (slot_of, a, bm, rank) = lora[2].clone().unwrap();
    let mut zeros = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    zeros.low_rank(2, &slot_of, &a, &vec![0.0; bm.len()], rank).unwrap();
    let plain = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    let xs: Vec<f32> = (0..3 * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
    let ls: Vec<f32> = (0..3 * width).map(|i| ((i * 53 % 89) as f32 - 44.0) / 15.0).collect();
    let (xt, lt) = (Tensor::from_vec(xs, vec![3, hidden]), Tensor::from_vec(ls, vec![3, width]));
    assert_eq!(zeros.forward(&xt, &lt, top_k).data(), plain.forward(&xt, &lt, top_k).data(), "an update of zeros");
    // and an update the experts cannot take is refused, not dropped
    assert!(zeros.low_rank(2, &slot_of[..count], &a, &bm, rank).is_err(), "pairs for another count of experts");
    assert!(zeros.low_rank(2, &slot_of, &a, &bm, rank).is_err(), "a second update of the same matrices");
}
