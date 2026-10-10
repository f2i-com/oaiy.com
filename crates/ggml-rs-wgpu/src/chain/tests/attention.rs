//! Attention's kernels against the CPU's and against each other: a prompt's, a step's, QSA's and a diffusion's.
use super::*;

/// A prompt's RoPE, its rows stored into a cache, and its causal attention over the cache (with and without a
/// window) give the CPU backend's answer: 37 queries after 300 positions (two runs of 256).
#[test]
fn a_prompts_rope_store_and_attention_match_the_cpus() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    let (n_h, n_kv, hd, past, rows) = (8usize, 2usize, 64usize, 300usize, 37usize);
    let (qd, kvd, cap) = (n_h * hd, n_kv * hd, 512usize);
    let mut next = rng(91);
    let ks: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
    let vs: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
    let q: Vec<f32> = (0..rows * qd).map(|_| next()).collect();
    let k: Vec<f32> = (0..rows * kvd).map(|_| next()).collect();
    let v: Vec<f32> = (0..rows * kvd).map(|_| next()).collect();
    let theta = 10000.0f32;
    let positions: Vec<u32> = (past..past + rows).map(|p| p as u32).collect();
    let table: Vec<f32> = positions
        .iter()
        .flat_map(|&pos| (0..hd / 2).flat_map(move |j| {
            let (s, c) = (pos as f32 * theta.powf(-2.0 * j as f32 / hd as f32)).sin_cos();
            [s, c]
        }))
        .collect();
    for window in [None, Some(100)] {
        let mut qc = ggml_rs::Tensor::from_vec(q.clone(), vec![rows, n_h, hd]);
        let mut kc = ggml_rs::Tensor::from_vec(k.clone(), vec![rows, n_kv, hd]);
        ggml_rs::Backend::rope(&cpu, &mut qc, &positions, hd, ggml_rs::RopeType::NeoX, theta, None);
        ggml_rs::Backend::rope(&cpu, &mut kc, &positions, hd, ggml_rs::RopeType::NeoX, theta, None);
        let (mut kcache, mut vcache) = (ks.clone(), vs.clone());
        kcache[past * kvd..(past + rows) * kvd].copy_from_slice(kc.data());
        vcache[past * kvd..(past + rows) * kvd].copy_from_slice(&v);
        let want = ggml_rs::Backend::attention(
            &cpu,
            &qc,
            &ggml_rs::Tensor::from_vec(kcache, vec![cap, n_kv, hd]),
            &ggml_rs::Tensor::from_vec(vcache, vec![cap, n_kv, hd]),
            past + rows,
            0.125,
            past,
            window,
        );
        let mut interleaved = Vec::with_capacity(cap * 2 * kvd);
        for t in 0..cap {
            interleaved.extend_from_slice(&ks[t * kvd..(t + 1) * kvd]);
            interleaved.extend_from_slice(&vs[t * kvd..(t + 1) * kvd]);
        }
        let (qv, kv_, vv, tab, cache) = (b.vec(rows * qd), b.vec(rows * kvd), b.vec(rows * kvd), b.vec(rows * hd), b.vec(cap * 2 * kvd));
        let out = b.vec(b.attention_rows_out_len(rows, n_h, hd, past + rows));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv_, &k);
        DeviceChain::upload(&b, &vv, &v);
        DeviceChain::upload(&b, &tab, &table);
        DeviceChain::upload(&b, &cache, &interleaved);
        // the f32 kernels' (the tensor cores' are checked against them)
        let mut rec = Recorder::new(&b);
        rec.rope_rows(&qv, rows, n_h, hd, &tab, true);
        rec.rope_rows(&kv_, rows, n_kv, hd, &tab, true);
        rec.store_rows(&kv_, &cache, rows, kvd, past, 2 * kvd, 0);
        rec.store_rows(&vv, &cache, rows, kvd, past, 2 * kvd, kvd);
        rec.attention_rows_f32(&qv, &cache, &out, rows, n_h, n_kv, hd, past, window, 0.125);
        rec.read_range(&out, 0, rows * qd);
        rec.read_range(&cache, past * 2 * kvd, kvd);
        let got = Box::new(rec).finish();
        close(&got[1], &kc.data()[..kvd], "the first stored key");
        close(&got[0], want.data(), &format!("a prompt's attention, window {window:?}"));
    }
}

/// QSA's attention of a prompt's rows on the tensor cores gives the f32 kernel's within f16's rounding: each
/// query's kept blocks (of 4 positions) a spread of its visible ones, every visible one where they are no more
/// than it keeps, its tail block's positions too; GQA, heads 128 and 256 wide, rows a tile's multiple and not.
#[test]
fn qsa_attention_on_the_tensor_cores_is_the_f32_kernels() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    if !b.gpu.coop16() {
        return;
    }
    let ratio = 4usize;
    for (n_h, n_kv, hd, first, rows, keep) in [(8usize, 2usize, 128usize, 300usize, 37usize, 24usize), (16, 2, 256, 2600, 64, 512), (16, 2, 256, 1000, 100, 2048)] {
        let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, first + rows);
        let mut next = rng((qd + first + keep) as u32);
        let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
        let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
        // a query's kept blocks: every visible one where it keeps as many, else a spread of them (ascending)
        let mut list = vec![0f32; rows * keep];
        for r in 0..rows {
            let visible = (first + r + 1) / ratio;
            let count = visible.min(keep);
            // (a spread, strictly ascending and below the visible: from the first on even rows, the last on odd)
            for i in 0..count {
                let blk = if visible <= keep { i } else if r % 2 == 0 { i * visible / count } else { visible - 1 - (count - 1 - i) * visible / count };
                list[r * keep + i] = f32::from_bits(blk as u32);
            }
        }
        let (qv, kv, lv) = (b.vec(rows * qd), b.vec(kv_len * row), b.vec(rows * keep));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        DeviceChain::upload(&b, &lv, &list);
        let scale = 1.0 / (hd as f32).sqrt();
        let len = b.qsa_attention_out_len(rows, n_h, hd, keep, ratio).max(rows.div_ceil(32) * 32 * qd);
        let (want, got) = (b.vec(len), b.vec(len));
        let mut rec = Recorder::new(&b);
        rec.qsa_attention_f32(&qv, &kv, &lv, &want, rows, n_h, n_kv, hd, first, ratio, keep, scale);
        assert!(rec.qsa_attention_coop(&qv, &kv, &lv, &got, rows, n_h, n_kv, hd, first, ratio, keep, scale), "on the tensor cores");
        rec.read_range(&want, 0, rows * qd);
        rec.read_range(&got, 0, rows * qd);
        let r = Box::new(rec).finish();
        let (want, got) = (&r[0], &r[1]);
        let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(got) * norm(want));
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {first}, keeping {keep}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}");
        assert!(cos > 0.99999 && worst <= 4e-3 * top, "{rows} rows after {first} keeping {keep}: cosine {cos}, worst {worst} of {top}");
    }
}

/// A step's attention with a KV head's query heads together gives what a workgroup a head gives: groups of 2, 3, 4
/// and 6, heads 64, 128 and 256 wide, a window's start, a run short of its 256, a single position; and a KV head's
/// twelve in two sixes (Flash-Next's).
#[test]
fn a_steps_attention_by_groups_is_the_heads() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    for (n_h, n_kv, hd, lo, kv_len) in [(24usize, 4usize, 256usize, 0usize, 3000usize), (8, 4, 64, 0, 700), (24, 8, 128, 100, 1111), (4, 1, 256, 0, 513), (16, 4, 128, 40, 41), (12, 2, 256, 0, 1), (24, 2, 256, 0, 2052), (24, 2, 256, 0, 300)] {
        let (qd, row, cap) = (n_h * hd, 2 * n_kv * hd, kv_len + 3);
        let mut next = rng((qd + kv_len) as u32);
        let q: Vec<f32> = (0..qd).map(|_| next() * 2.0).collect();
        let cache: Vec<f32> = (0..cap * row).map(|_| next() * 2.0).collect();
        let (qv, kv) = (b.vec(qd), b.vec(cap * row));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        let scale = 1.0 / (hd as f32).sqrt();
        let len = b.attention_out_len(n_h, hd, cap);
        let (want, got) = (b.vec(len), b.vec(len));
        assert!(attention_group_of(n_h / n_kv, hd, b.gpu.limits.max_compute_workgroup_storage_size, 0).is_some(), "{n_h} heads of {hd} over {n_kv}: by groups");
        let mut rec = Recorder::new(&b);
        rec.attention_by(&qv, &kv, &want, n_h, n_kv, hd, lo, kv_len, cap, scale, false);
        rec.attention_by(&qv, &kv, &got, n_h, n_kv, hd, lo, kv_len, cap, scale, true);
        rec.read_range(&want, 0, qd);
        rec.read_range(&got, 0, qd);
        let r = Box::new(rec).finish();
        let (want, got) = (&r[0], &r[1]);
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(top > 0.0 && worst <= 2e-5 * top, "{n_h} heads ({n_kv} kv) {hd} wide over {lo}..{kv_len}: worst {worst} of {top}");
    }
}

/// QSA's attention with a KV head's query heads together gives what a workgroup a head gives, for a step's row and a
/// check's few: Flash-Next's 24 heads over 2 of 256 in sixes past its dense span (blocks of 16, 128 kept), and smaller
/// shapes whose queries keep every block they see, a spread of them, and have a tail block or none. And with every
/// block kept a query's is the dense step's attention by groups, bit for bit.
#[test]
fn qsas_attention_by_groups_is_the_heads() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    for (n_h, n_kv, hd, first, rows, keep, ratio) in [(8usize, 2usize, 64usize, 300usize, 3usize, 24usize, 4usize), (24, 4, 128, 1000, 5, 64, 8), (24, 2, 256, 700, 2, 128, 16), (24, 2, 256, 4095, 1, 128, 16), (24, 2, 256, 4090, 4, 128, 16), (24, 2, 256, 4100, 8, 128, 16)] {
        let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, first + rows);
        let mut next = rng((qd + first + keep) as u32);
        let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
        let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
        // a query's kept blocks: every visible one where it keeps as many, else a spread of them (ascending)
        let mut list = vec![0f32; rows * keep];
        for r in 0..rows {
            let visible = (first + r + 1) / ratio;
            let count = visible.min(keep);
            for i in 0..count {
                let blk = if visible <= keep { i } else if r % 2 == 0 { i * visible / count } else { visible - 1 - (count - 1 - i) * visible / count };
                list[r * keep + i] = f32::from_bits(blk as u32);
            }
        }
        let (qv, kv, lv) = (b.vec(rows * qd), b.vec(kv_len * row), b.vec(rows * keep));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        DeviceChain::upload(&b, &lv, &list);
        let scale = 1.0 / (hd as f32).sqrt();
        assert!(attention_group_of(n_h / n_kv, hd, b.gpu.limits.max_compute_workgroup_storage_size, 1024).is_some(), "{n_h} heads of {hd} over {n_kv}: by groups");
        let len = b.qsa_attention_out_len(rows, n_h, hd, keep, ratio);
        let (want, got) = (b.vec(len), b.vec(len));
        let at = std::time::Instant::now();
        let mut rec = Recorder::new(&b);
        rec.qsa_attention_f32_by(&qv, &kv, &lv, &want, rows, n_h, n_kv, hd, first, ratio, keep, scale, false);
        rec.qsa_attention_f32_by(&qv, &kv, &lv, &got, rows, n_h, n_kv, hd, first, ratio, keep, scale, true);
        rec.read_range(&want, 0, rows * qd);
        rec.read_range(&got, 0, rows * qd);
        let r = Box::new(rec).finish();
        // (the small shapes come first: a kernel that is slow there stops the test before the model's)
        let took = at.elapsed().as_secs_f64();
        assert!(took < 5.0, "QSA's attention by groups took {took:.1} s: stop and look");
        let (want, got) = (&r[0], &r[1]);
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {first}, keeping {keep} of {ratio}: worst {worst:.2e} of {top:.2}, {:.0} ms with the pipelines", took * 1e3);
        assert!(top > 0.0 && worst <= 2e-5 * top, "{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {first}: worst {worst} of {top}");
    }
    // every block kept (37 positions, blocks of 4, 40 kept): the dense step's attention by groups, bit for bit
    let (n_h, n_kv, hd, at, ratio, keep) = (24usize, 2usize, 256usize, 37usize, 4usize, 40usize);
    let (qd, row, total) = (n_h * hd, 2 * n_kv * hd, at + 1);
    let mut next = rng(77);
    let (qv, kv, lv) = (b.vec(qd), b.vec(total * row), b.vec(keep));
    DeviceChain::upload(&b, &qv, &(0..qd).map(|_| next() * 2.0).collect::<Vec<_>>());
    DeviceChain::upload(&b, &kv, &(0..total * row).map(|_| next() * 2.0).collect::<Vec<_>>());
    DeviceChain::upload(&b, &lv, &(0..keep).map(|i| f32::from_bits(i as u32)).collect::<Vec<_>>());
    let scale = 1.0 / (hd as f32).sqrt();
    let (dense, sparse) = (b.vec(b.attention_out_len(n_h, hd, total)), b.vec(b.qsa_attention_out_len(1, n_h, hd, keep, ratio)));
    let mut rec = Recorder::new(&b);
    rec.attention_by(&qv, &kv, &dense, n_h, n_kv, hd, 0, total, total, scale, true);
    rec.qsa_attention_f32_by(&qv, &kv, &lv, &sparse, 1, n_h, n_kv, hd, at, ratio, keep, scale, true);
    rec.read_range(&dense, 0, qd);
    rec.read_range(&sparse, 0, qd);
    let r = Box::new(rec).finish();
    assert!(r[0].iter().any(|v| *v != 0.0), "values to compare");
    assert_eq!(r[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), r[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "every block kept: the dense attention by groups");
}

/// A step's attention over its cache's f16 halves gives what the f32 cache gives, to f16's rounding: the halves
/// made in two goes (rows added since the first), groups of 2, 3 and 6, heads 64, 128 and 256 wide.
#[test]
fn a_steps_attention_over_halves_is_the_f32_caches() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    for (n_h, n_kv, hd, lo, kv_len) in [(24usize, 4usize, 256usize, 0usize, 3000usize), (8, 4, 64, 0, 700), (24, 8, 128, 100, 1111)] {
        if !b.attention_halves(n_h, n_kv, hd) {
            return;
        }
        let (qd, row, cap) = (n_h * hd, 2 * n_kv * hd, kv_len + 3);
        let mut next = rng((qd + kv_len) as u32);
        let q: Vec<f32> = (0..qd).map(|_| next() * 2.0).collect();
        let cache: Vec<f32> = (0..cap * row).map(|_| next() * 2.0).collect();
        let (qv, kv, half) = (b.vec(qd), b.vec(cap * row), b.vec(cap * row / 2));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        let scale = 1.0 / (hd as f32).sqrt();
        let len = b.attention_out_len(n_h, hd, cap);
        let (want, got) = (b.vec(len), b.vec(len));
        let mut rec = Recorder::new(&b);
        rec.attention_by(&qv, &kv, &want, n_h, n_kv, hd, lo, kv_len, cap, scale, false);
        let first = kv_len / 3;
        rec.halve(&kv, &half, 0, first * row);
        rec.halve(&kv, &half, first * row, (kv_len - first) * row);
        rec.attention_halved(&qv, &half, &got, n_h, n_kv, hd, lo, kv_len, cap, scale);
        rec.read_range(&want, 0, qd);
        rec.read_range(&got, 0, qd);
        let r = Box::new(rec).finish();
        let (want, got) = (&r[0], &r[1]);
        let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(got) * norm(want));
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        eprintln!("{n_h} heads ({n_kv} kv) {hd} wide over {lo}..{kv_len}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}");
        assert!(cos > 0.99999 && worst <= 4e-3 * top, "{n_h} heads ({n_kv} kv) {hd} wide over {lo}..{kv_len}: cosine {cos}, worst {worst} of {top}");
    }
}

/// A few rows' attention with its parts' sums four at a time gives what a sum a load gives: a check's few rows
/// over a long cache, causal from its start and past it, windowed, full, GQA, heads 64, 128 and 256 wide.
#[test]
fn a_few_rows_parts_in_fours_are_the_parts() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    for (n_h, n_kv, hd, past, rows, window, full) in [
        (24usize, 4usize, 256usize, 3000usize, 4usize, None, false),
        (8, 2, 64, 300, 1, None, false),
        (16, 2, 128, 0, 5, None, false),
        (8, 8, 128, 500, 3, Some(200usize), false),
        (4, 1, 256, 513, 7, None, true),
        (6, 3, 128, 40, 9, Some(17), false),
    ] {
        let kv_len = if full { past } else { past + rows };
        let (qd, row) = (n_h * hd, 2 * n_kv * hd);
        let mut next = rng((qd + past + rows) as u32);
        let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
        let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
        let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        let scale = 1.0 / (hd as f32).sqrt();
        let len = attention_runs_out_len(rows, n_h, hd, kv_len);
        let (want, got) = (b.vec(len), b.vec(len));
        let mut rec = Recorder::new(&b);
        rec.attention_rows_runs_by(&qv, &kv, &want, rows, n_h, n_kv, hd, past, window, scale, full, false);
        rec.attention_rows_runs_by(&qv, &kv, &got, rows, n_h, n_kv, hd, past, window, scale, full, true);
        rec.read_range(&want, 0, rows * qd);
        rec.read_range(&got, 0, rows * qd);
        let r = Box::new(rec).finish();
        let (want, got) = (&r[0], &r[1]);
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(top > 0.0 && worst <= 2e-5 * top, "{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past} (window {window:?}, full {full}): worst {worst} of {top}");
    }
}

/// A prompt's attention in one pass (tiled) gives the runs' kernel's: causal from the cache's start and past it,
/// windowed, full (every query over every position: more positions than queries, and as many), GQA, heads 64,
/// 128 and 256 wide, rows a tile's multiple and not, chunks of a tile or two and one of them all.
#[test]
fn a_tiled_attention_is_the_runs_kernels() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    for (n_h, n_kv, hd, past, rows, window, full, chunk) in [
        (8usize, 2usize, 64usize, 300usize, 37usize, None, false, 64usize),
        (16, 2, 128, 0, 100, None, false, 64),
        (24, 4, 256, 214, 64, None, false, 32),
        (8, 8, 128, 500, 130, Some(200usize), false, 64),
        (8, 8, 128, 333, 333, None, true, 128),
        (4, 4, 64, 777, 100, None, true, 64),
        (4, 1, 256, 513, 70, None, true, 32),
        (6, 3, 128, 40, 300, Some(17), false, 1 << 20),
    ] {
        let kv_len = if full { past } else { past + rows };
        let (qd, row) = (n_h * hd, 2 * n_kv * hd);
        let mut next = rng((qd + past + rows) as u32);
        let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
        let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
        let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        let scale = 1.0 / (hd as f32).sqrt();
        let (want, got) = (b.vec(attention_runs_out_len(rows, n_h, hd, kv_len)), b.vec(rows * qd));
        let mut rec = Recorder::new(&b);
        rec.attention_rows_runs(&qv, &kv, &want, rows, n_h, n_kv, hd, past, window, scale, full);
        rec.attention_rows_tiled(&qv, &kv, &got, rows, n_h, n_kv, hd, past, window, scale, full, chunk);
        rec.read_range(&want, 0, rows * qd);
        rec.read(&got);
        let r = Box::new(rec).finish();
        let (want, got) = (&r[0], &r[1]);
        let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(got) * norm(want));
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows, past {past}, window {window:?}, full {full}, chunk {chunk}: cosine {cos:.8}, worst {worst:.2e} of {top:.2}");
        assert!(cos > 0.999999 && worst <= 2e-5 * top, "{n_h} heads {hd} wide, {rows} rows, past {past}: cosine {cos}, worst {worst} of {top}");
    }
}

/// A prompt's attention on the tensor cores gives the f32 kernels' within f16's rounding: GQA, heads 64, 128 and
/// 256 wide, from the cache's start and past it, the rows a tile's multiple and not, its scores spread and peaked.
#[test]
fn a_prompts_attention_on_the_tensor_cores_is_the_f32_kernels() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    if !b.gpu.coop16() {
        return;
    }
    // (each as it is, then with keys far past the rest planted)
    let shapes = [(8usize, 2usize, 64usize, 300usize, 37usize, 1.0f32), (16, 2, 128, 0, 100, 2.0), (24, 4, 256, 214, 64, 1.0), (24, 4, 256, 1000, 150, 3.0)];
    for (planted, (n_h, n_kv, hd, past, rows, spread)) in [false, true].into_iter().flat_map(|p| shapes.into_iter().map(move |s| (p, s))) {
        let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
        let mut next = rng((qd + past) as u32);
        let q: Vec<f32> = (0..rows * qd).map(|_| next() * spread).collect();
        let mut cache: Vec<f32> = (0..kv_len * row).map(|_| next() * spread).collect();
        // (three keys late in the cache made long: a score a dozen or more past a query's largest before it, or as far
        // under: the one-pass kernel's reference moved and its sums scaled down; f16 keeps so large a score to a
        // hundredth, so the answers agree less closely)
        if planted {
            for at in [kv_len / 3, kv_len / 2 + 7, kv_len - 9] {
                for v in &mut cache[at * row..at * row + n_kv * hd] {
                    *v *= 36.0 / (spread * spread);
                }
            }
        }
        let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv, &cache);
        let scale = 1.0 / (hd as f32).sqrt();
        let len = b.attention_rows_out_len(rows, n_h, hd, kv_len);
        let (want, got) = (b.vec(len), b.vec(len));
        let mut rec = Recorder::new(&b);
        rec.attention_rows_f32(&qv, &kv, &want, rows, n_h, n_kv, hd, past, None, scale);
        assert!(rec.attention_rows_coop(&qv, &kv, &got, rows, n_h, n_kv, hd, past, None, scale), "on the tensor cores");
        rec.read_range(&want, 0, rows * qd);
        rec.read_range(&got, 0, rows * qd);
        let r = Box::new(rec).finish();
        let (want, got) = (&r[0], &r[1]);
        let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(got) * norm(want));
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past}{}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}", if planted { ", keys planted" } else { "" });
        let (least, most) = if planted { (0.9999, 3e-2) } else { (0.99999, 4e-3) };
        assert!(cos > least && worst <= most * top, "{n_h} heads {hd} wide, {rows} rows after {past}: cosine {cos}, worst {worst} of {top}");
    }
}

/// RoPE, a store into a cache and attention over it give the CPU backend's answer (GQA): a few positions in, and
/// past 512 (three runs of the split attention put together).
#[test]
fn rope_store_and_attention_match_the_cpus() {
    for (cap, past) in [(16usize, 9usize), (640, 530)] {
        rope_store_and_attention_case(cap, past);
    }
}

fn rope_store_and_attention_case(cap: usize, past: usize) {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    let (n_h, n_kv, hd) = (8usize, 2usize, 64usize);
    let (qd, kvd) = (n_h * hd, n_kv * hd);
    let mut next = rng(77);
    // the cache's earlier rows, the new token's q, k and v
    let ks: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
    let vs: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
    let q: Vec<f32> = (0..qd).map(|_| next()).collect();
    let k: Vec<f32> = (0..kvd).map(|_| next()).collect();
    let v: Vec<f32> = (0..kvd).map(|_| next()).collect();
    let theta = 10000.0f32;
    let table: Vec<f32> = (0..hd / 2)
        .flat_map(|j| {
            let (s, c) = (past as f32 * theta.powf(-2.0 * j as f32 / hd as f32)).sin_cos();
            [s, c]
        })
        .collect();
    for neox in [false, true] {
        let rope_type = if neox { ggml_rs::RopeType::NeoX } else { ggml_rs::RopeType::Normal };
        // CPU
        let mut qc = ggml_rs::Tensor::from_vec(q.clone(), vec![1, n_h, hd]);
        let mut kc = ggml_rs::Tensor::from_vec(k.clone(), vec![1, n_kv, hd]);
        ggml_rs::Backend::rope(&cpu, &mut qc, &[past as u32], hd, rope_type, theta, None);
        ggml_rs::Backend::rope(&cpu, &mut kc, &[past as u32], hd, rope_type, theta, None);
        let mut kcache = ks.clone();
        let mut vcache = vs.clone();
        kcache[past * kvd..(past + 1) * kvd].copy_from_slice(kc.data());
        vcache[past * kvd..(past + 1) * kvd].copy_from_slice(&v);
        let want = ggml_rs::Backend::attention(
            &cpu,
            &qc,
            &ggml_rs::Tensor::from_vec(kcache, vec![cap, n_kv, hd]),
            &ggml_rs::Tensor::from_vec(vcache, vec![cap, n_kv, hd]),
            past + 1,
            0.125,
            past,
            None,
        );
        // the chain: the cache interleaved a row at a time (K then V)
        let mut interleaved = Vec::with_capacity(cap * 2 * kvd);
        for t in 0..cap {
            interleaved.extend_from_slice(&ks[t * kvd..(t + 1) * kvd]);
            interleaved.extend_from_slice(&vs[t * kvd..(t + 1) * kvd]);
        }
        let (qv, kv_, vv, tab, cache, out) = (b.vec(qd), b.vec(kvd), b.vec(kvd), b.vec(hd), b.vec(cap * 2 * kvd), b.vec(b.attention_out_len(n_h, hd, cap)));
        DeviceChain::upload(&b, &qv, &q);
        DeviceChain::upload(&b, &kv_, &k);
        DeviceChain::upload(&b, &vv, &v);
        DeviceChain::upload(&b, &tab, &table);
        DeviceChain::upload(&b, &cache, &interleaved);
        let mut rec = b.begin();
        rec.rope(&qv, n_h, hd, &tab, neox);
        rec.rope(&kv_, n_kv, hd, &tab, neox);
        rec.store(&kv_, &cache, past * 2 * kvd);
        rec.store(&vv, &cache, past * 2 * kvd + kvd);
        rec.attention(&qv, &cache, &out, n_h, n_kv, hd, 0, past + 1, cap, 0.125);
        rec.read_range(&out, 0, qd);
        rec.read_range(&cache, past * 2 * kvd, kvd);
        rec.read(&qv);
        let got = rec.finish();
        close(&got[2], qc.data(), "the rotated query");
        close(&got[1], kc.data(), "the stored key");
        close(&got[0], want.data(), "the attention");
    }
}

/// Full (unmasked) attention as the host computes it: every query over all of a text prefix's positions and the
/// queries' own (Qwen-Image's image tokens: 32 heads of 128, here 4), on the tensor cores (a ragged 70 queries
/// over 23 + 70) and in f32 (the same, and 5 queries), and a causal one beside it unchanged.
#[test]
fn full_attention_is_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(91);
    for hd in [128usize, 64] {
        let (n_h, scale) = (4usize, 1.0 / (hd as f32).sqrt());
        // (queries over a prefix and themselves, and over fewer positions than they are: a video's over a text's; and with
        // keys late in the cache made long, a score far past a query's largest before it: the one-pass kernels' reference
        // moved and their sums scaled down, f16 keeping so large a score to a hundredth)
        for (rows, kv_len, planted) in [(70usize, 93usize, false), (5, 14, false), (70, 23, false), (70, 150, true)] {
            let q: Vec<f32> = (0..rows * n_h * hd).map(|_| r()).collect();
            let mut kv: Vec<f32> = (0..kv_len * 2 * n_h * hd).map(|_| r()).collect();
            let row = 2 * n_h * hd;
            if planted {
                for at in [kv_len / 2 + 7, kv_len - 9] {
                    for v in &mut kv[at * row..at * row + n_h * hd] {
                        *v *= 36.0;
                    }
                }
            }
            let mut want = vec![0f64; rows * n_h * hd];
            for s in 0..rows {
                for h in 0..n_h {
                    let qv = &q[(s * n_h + h) * hd..(s * n_h + h + 1) * hd];
                    let scores: Vec<f64> = (0..kv_len).map(|t| (0..hd).map(|d| qv[d] as f64 * kv[t * row + h * hd + d] as f64).sum::<f64>() * scale as f64).collect();
                    let m = scores.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = scores.iter().map(|v| (v - m).exp()).collect();
                    let l: f64 = e.iter().sum();
                    for d in 0..hd {
                        want[(s * n_h + h) * hd + d] = (0..kv_len).map(|t| e[t] * kv[t * row + n_h * hd + h * hd + d] as f64).sum::<f64>() / l;
                    }
                }
            }
            let (qd, kvd) = (b.vec(q.len()), b.vec(kv.len()));
            DeviceChain::upload(&b, &qd, &q);
            DeviceChain::upload(&b, &kvd, &kv);
            for coop in [true, false] {
                if coop && (rows < 16 || !b.gpu.coop16()) {
                    continue;
                }
                let out = b.vec(b.attention_rows_full_out_len(rows, n_h, hd, kv_len).max(b.attention_rows_out_len(rows.div_ceil(32) * 32, n_h, hd, kv_len)));
                let mut rec = Recorder::new(&b);
                if coop {
                    assert!(rec.attention_rows_coop_masked(&qd, &kvd, &out, rows, n_h, n_h, hd, kv_len, None, scale, true));
                } else {
                    rec.attention_rows_f32_masked(&qd, &kvd, &out, rows, n_h, n_h, hd, kv_len, None, scale, true);
                }
                rec.read_range(&out, 0, rows * n_h * hd);
                let got = Box::new(rec).finish().pop().unwrap();
                let tol = if planted && coop { 2e-2 } else if coop { 2e-3 } else { 1e-5 };
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!((*g as f64 - w).abs() <= tol, "{} {rows} queries over {kv_len} [{i}]: {g} against {w}", if coop { "tensor cores" } else { "f32" });
                }
            }
        }
    }
}

/// QSA chained gives the host's: the pooled block keys bit for bit, the block scores within rounding, each query's
/// chosen blocks the same, and its attention over them and its tail within rounding (Qwen3.8-Flash-Next's 4 index
/// heads of 128, 24 heads of 256 over 2 kv heads, blocks of 4, 64 kept of 750); the selection of 512 of 4096 blocks
/// and of 3001; and where a query keeps every block, the dense decode attention's bits.
#[test]
fn qsa_chained_is_the_hosts() {
    use ggml_rs::Backend;
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    // (its selection takes 32 KB of a workgroup's memory: none where a workgroup has less)
    if DeviceChain::qsa_attention_out_len(&b, 1, 1, 64, 1, 4) == 0 {
        return;
    }
    let cpu = ggml_rs::CpuBackend::new();
    let mut r = rng(29);
    let (ratio, d, heads, keep, first, rows) = (4usize, 128usize, 4usize, 64usize, 2998usize, 3usize);
    let (nh, nkv, hd) = (24usize, 2usize, 256usize);
    let total = first + rows;
    let nb = total / ratio;
    let raw: Vec<f32> = (0..total * d).map(|_| r()).collect();
    let t = |v: &[f32], shape: Vec<usize>| ggml_rs::Tensor::from_vec(v.to_vec(), shape);
    let up = |v: &[f32]| {
        let x = b.vec(v.len());
        DeviceChain::upload(&b, &x, v);
        x
    };
    // the pool
    let (rawd, pooled) = (up(&raw), b.vec(nb * d));
    let mut rec = b.begin();
    rec.qsa_pool(&rawd, &pooled, nb, ratio, d);
    rec.read(&pooled);
    let got_pool = rec.finish().pop().unwrap();
    let want_pool = cpu.qsa_pool(&t(&raw, vec![total, d]), nb, ratio);
    assert_eq!(got_pool.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want_pool.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "the pooled keys");
    // the scores, the selection, the attention
    let q: Vec<f32> = (0..rows * heads * d).map(|_| r()).collect();
    let qa: Vec<f32> = (0..rows * nh * hd).map(|_| r()).collect();
    let (k, v): (Vec<f32>, Vec<f32>) = ((0..total * nkv * hd).map(|_| r()).collect(), (0..total * nkv * hd).map(|_| r()).collect());
    let mut kv = Vec::with_capacity(total * 2 * nkv * hd);
    for p in 0..total {
        kv.extend_from_slice(&k[p * nkv * hd..(p + 1) * nkv * hd]);
        kv.extend_from_slice(&v[p * nkv * hd..(p + 1) * nkv * hd]);
    }
    let scale = 1.0 / (d as f32).sqrt();
    let ascale = 1.0 / (hd as f32).sqrt();
    let (qd, scores, list, qad, kvd) = (up(&q), b.vec(rows * nb), b.vec(rows * keep), up(&qa), up(&kv));
    let out = b.vec(DeviceChain::qsa_attention_out_len(&b, rows, nh, hd, keep, ratio));
    let mut rec = b.begin();
    rec.qsa_scores(&qd, &pooled, &scores, rows, heads, d, nb, first, ratio, scale);
    rec.qsa_select(&scores, &list, rows, nb, first, ratio, keep);
    rec.qsa_attention(&qad, &kvd, &list, &out, rows, nh, nkv, hd, first, ratio, keep, ascale);
    rec.read(&scores);
    rec.read(&list);
    rec.read_range(&out, 0, rows * nh * hd);
    let mut got = rec.finish();
    let (got_out, got_list, got_scores) = (got.pop().unwrap(), got.pop().unwrap(), got.pop().unwrap());
    let want_scores = cpu.qsa_block_scores(&t(&q, vec![rows, heads, d]), &want_pool.reshape(vec![nb, d]).unwrap(), first, ratio, scale);
    for (i, (g, w)) in got_scores.iter().zip(want_scores.data()).enumerate() {
        assert!(g == w || (g - w).abs() <= 1e-5 * w.abs().max(1.0), "score {i}: {g} against {w}");
    }
    let want_sel = cpu.qsa_select(&want_scores, first, ratio, keep);
    let width = keep * ratio + ratio;
    for row in 0..rows {
        let mut want: Vec<u32> = want_sel.data()[row * width..row * width + keep * ratio].iter().step_by(ratio).map(|&v| v as u32 / ratio as u32).collect();
        want.sort();
        let got_row: Vec<u32> = got_list[row * keep..(row + 1) * keep].iter().map(|v| v.to_bits()).collect();
        assert_eq!(got_row, want, "row {row}'s blocks");
    }
    let want_out = cpu.sparse_attention(&t(&qa, vec![rows, nh, hd]), &t(&k, vec![total, nkv, hd]), &t(&v, vec![total, nkv, hd]), &want_sel, ascale);
    let scale_out = want_out.data().iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    for (i, (g, w)) in got_out.iter().zip(want_out.data()).enumerate() {
        assert!((g - w).abs() <= 1e-5 * scale_out, "attention {i}: {g} against {w}");
    }
    // the selection at its limits: 4096 blocks (the sort filling the workgroup's memory) and an odd count
    for nb in [4096usize, 3001] {
        let first = nb * ratio - 2;
        let sc: Vec<f32> = (0..2 * nb).map(|_| r().abs() * 4.0).collect();
        let (sd, ld) = (up(&sc), b.vec(2 * keep));
        let mut rec = b.begin();
        rec.qsa_select(&sd, &ld, 2, nb, first, ratio, keep);
        rec.read(&ld);
        let got = rec.finish().pop().unwrap();
        // what a selection must be (the host's picks among equal scores are its own): `keep` of the blocks the
        // row sees, ascending, none dropped above one kept, and of equals the lower kept first
        for row in 0..2 {
            let visible = ((first + row + 1) / ratio).min(nb);
            let got_row: Vec<usize> = got[row * keep..(row + 1) * keep].iter().map(|v| v.to_bits() as usize).collect();
            assert!(got_row.windows(2).all(|w| w[0] < w[1]) && got_row.iter().all(|&j| j < visible), "{nb} blocks: row {row}'s ascending, seen");
            let kept: std::collections::BTreeSet<usize> = got_row.iter().copied().collect();
            let s_row = &sc[row * nb..row * nb + visible];
            let worst_kept = kept.iter().map(|&j| s_row[j]).fold(f32::INFINITY, f32::min);
            for j in (0..visible).filter(|j| !kept.contains(j)) {
                assert!(s_row[j] <= worst_kept, "{nb} blocks: row {row} dropped {j} ({}) above a kept {worst_kept}", s_row[j]);
                if s_row[j] == worst_kept {
                    assert!(kept.iter().filter(|&&k| s_row[k] == worst_kept).all(|&k| k < j), "{nb} blocks: row {row}: of equals the lower first");
                }
            }
        }
    }
    // every block kept: the dense decode attention, bit for bit
    let (few, at) = (40usize, 37usize);
    let mut rec = b.begin();
    let q1 = up(&qa[..nh * hd]);
    let (dense, sparse, l2, s2) = (b.vec(DeviceChain::attention_out_len(&b, nh, hd, total)), b.vec(DeviceChain::qsa_attention_out_len(&b, 1, nh, hd, few, ratio)), b.vec(few), b.vec(few));
    rec.qsa_scores(&q1, &pooled, &s2, 1, 1, d, at / ratio, at, ratio, scale);
    let _ = (heads, &l2);
    rec.qsa_select(&s2, &l2, 1, at / ratio, at, ratio, few);
    rec.qsa_attention(&q1, &kvd, &l2, &sparse, 1, nh, nkv, hd, at, ratio, few, ascale);
    rec.attention(&q1, &kvd, &dense, nh, nkv, hd, 0, at + 1, total, ascale);
    rec.read_range(&sparse, 0, nh * hd);
    rec.read_range(&dense, 0, nh * hd);
    let mut got = rec.finish();
    let (dn, sp) = (got.pop().unwrap(), got.pop().unwrap());
    assert_eq!(sp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), dn.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "every block kept: the dense attention");
}

/// QSA's selection by ranks (a step's query, a check's few) keeps the blocks the sort keeps, written the same way:
/// scores with many equals (a tie to the lower block), queries that see fewer blocks than they keep, as many, and
/// more; 37, 256 and 1,024 blocks.
#[test]
fn qsas_ranked_selection_is_the_sorted_one() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut next = rng(71);
    for (nb, keep, ratio, first, rows) in [(37usize, 12usize, 4usize, 40usize, 4usize), (256, 128, 16, 4000, 4), (256, 128, 16, 4090, 1), (1024, 128, 16, 16300, 8), (64, 128, 16, 1000, 3)] {
        // (a few distinct values, negative ones too: equals are common)
        let scores: Vec<f32> = (0..rows * nb).map(|_| (next() * 6.0).round() / 3.0).collect();
        let sd = b.vec(rows * nb);
        DeviceChain::upload(&b, &sd, &scores);
        let (ranked, sorted) = (b.vec(rows * keep), b.vec(rows * keep));
        let dd = b.gpu.dummy().clone();
        let drw = b.gpu.dummy_rw().clone();
        let mut rec = Recorder::new(&b);
        ChainRecorder::qsa_select(&mut rec, &sd, &ranked, rows, nb, first, ratio, keep);
        rec.dispatch_wide("chain-qsa-select", QSA_SELECT, [buffer(&sd), &dd, &dd, &dd, &dd, &dd, buffer(&sorted), &drw], &[rows as u32, nb as u32, first as u32, ratio as u32, keep as u32], (rows as u32, 1, 1));
        rec.read(&ranked);
        rec.read(&sorted);
        let got = Box::new(rec).finish();
        for r in 0..rows {
            let count = ((first + r + 1) / ratio).min(nb).min(keep);
            let (a, s) = (&got[0][r * keep..r * keep + count], &got[1][r * keep..r * keep + count]);
            assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{nb} blocks keeping {keep}, row {r} of {rows}");
            assert!(count == 0 || a.windows(2).all(|w| w[0].to_bits() < w[1].to_bits()), "{nb} blocks: row {r}'s ascending");
        }
    }
}
