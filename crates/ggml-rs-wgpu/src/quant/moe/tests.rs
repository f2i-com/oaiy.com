//! `quant::moe`'s tests.

use super::*;
use ggml_rs::exl3::route;

fn rng(mut seed: u64) -> impl FnMut() -> f32 {
    move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
    }
}

/// `matrices` of `rows` by `cols` in Q2_0: every code at random, each block a scale of its own (some negative).
fn random_q2_0(next: &mut impl FnMut() -> f32, matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(matrices * rows * cols / 64 * 18);
    for _ in 0..matrices * rows * cols / 64 {
        let d = (0.015 + 0.01 * next()) * if next() > 0.8 { -1.0 } else { 1.0 };
        bytes.extend(half::f16::from_f32(d).to_bits().to_le_bytes());
        bytes.extend((0..16).map(|_| ((next() + 1.0) * 127.99) as u8));
    }
    bytes
}

/// `matrices` of `rows` by `cols` in a grid type (IQ2_S, IQ2_XXS or IQ1_M): every index, sign and scale at random,
/// each block's own scale a small f16 (IQ1_M's in the top nibbles of its four words of scales).
fn random_grid(next: &mut impl FnMut() -> f32, t: GgmlType, matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
    let blocks = matrices * rows * cols / 256;
    let mut bytes = Vec::with_capacity(blocks * t.type_size());
    for _ in 0..blocks {
        let d = half::f16::from_f32(0.004 + 0.002 * next()).to_bits();
        if t == GgmlType::IQ1_M {
            bytes.extend((0..48).map(|_| ((next() + 1.0) * 127.99) as u8));
            for i in 0..4 {
                let scales = ((next() + 1.0) * 2047.99) as u16;
                bytes.extend((((d >> (4 * i)) & 15) << 12 | scales).to_le_bytes());
            }
        } else {
            bytes.extend(d.to_le_bytes());
            bytes.extend((0..t.type_size() - 2).map(|_| ((next() + 1.0) * 127.99) as u8));
        }
    }
    bytes
}

/// `matrices` of `rows` by `cols` in a K-quant (Q4_K, Q5_K or Q6_K): every nibble, high bit and scale at random, each
/// block's own scales small f16s (Q6_K's sixteen int8 ones, some negative).
fn random_k(next: &mut impl FnMut() -> f32, t: GgmlType, matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
    let blocks = matrices * rows * cols / 256;
    let mut bytes = Vec::with_capacity(blocks * t.type_size());
    let byte = |next: &mut dyn FnMut() -> f32| ((next() + 1.0) * 127.99) as u8;
    for _ in 0..blocks {
        if t == GgmlType::Q4_K || t == GgmlType::Q5_K {
            bytes.extend(half::f16::from_f32(0.0004 + 0.0002 * next()).to_bits().to_le_bytes());
            bytes.extend(half::f16::from_f32(0.0004 + 0.0002 * next()).to_bits().to_le_bytes());
            bytes.extend((0..t.type_size() - 4).map(|_| byte(&mut *next)));
        } else {
            bytes.extend((0..208).map(|_| byte(&mut *next)));
            bytes.extend(half::f16::from_f32(0.0004 + 0.0002 * next()).to_bits().to_le_bytes());
        }
    }
    bytes
}

/// Experts at random in K-quants, as a Q4_K_M GGUF keeps them: gate and up `gu`, down `down`.
pub(crate) fn k_experts(seed: u64, hidden: usize, ff: usize, count: usize, gu: GgmlType, down: GgmlType) -> QuantExpertsData {
    let mut next = rng(seed);
    let gate = random_k(&mut next, gu, count, ff, hidden);
    let up = random_k(&mut next, gu, count, ff, hidden);
    let dn = random_k(&mut next, down, count, hidden, ff);
    let mut dense = |n: usize| -> Vec<f32> { (0..n).map(|_| half::f16::from_f32(0.03 * next()).to_f32()).collect() };
    let shared = [dense(ff * hidden), dense(ff * hidden), dense(hidden * ff)];
    QuantExpertsData { hidden, ff, experts: count, gate: (gu, gate), up: (gu, up), down: (down, dn), shared: Some(shared), routing: Routing::Softmax }
}

/// Experts at random: the routed ones Q2_0; the shared one's values f16's where `half` (as a Q2_0 matrix's are).
pub(crate) fn experts(seed: u64, hidden: usize, ff: usize, count: usize, half: bool) -> QuantExpertsData {
    experts_of(seed, hidden, ff, count, half, GgmlType::Q2_0)
}

/// [`experts`] with the routed ones' gate and up matrices of type `gu` (their down ones Q2_0, as the GSQ-RCO
/// IQ2_XS file has them).
pub(crate) fn experts_of(seed: u64, hidden: usize, ff: usize, count: usize, half: bool, gu: GgmlType) -> QuantExpertsData {
    let mut next = rng(seed);
    if gu != GgmlType::Q2_0 {
        let gate = random_grid(&mut next, gu, count, ff, hidden);
        let up = random_grid(&mut next, gu, count, ff, hidden);
        let down = random_q2_0(&mut next, count, hidden, ff);
        let mut dense = |n: usize| -> Vec<f32> { (0..n).map(|_| if half { half::f16::from_f32(0.03 * next()).to_f32() } else { 0.03 * next() }).collect() };
        let shared = [dense(ff * hidden), dense(ff * hidden), dense(hidden * ff)];
        return QuantExpertsData { hidden, ff, experts: count, gate: (gu, gate), up: (gu, up), down: (GgmlType::Q2_0, down), shared: Some(shared), routing: Routing::Softmax };
    }
    let gate = random_q2_0(&mut next, count, ff, hidden);
    let up = random_q2_0(&mut next, count, ff, hidden);
    let down = random_q2_0(&mut next, count, hidden, ff);
    let mut dense = |n: usize| -> Vec<f32> { (0..n).map(|_| if half { half::f16::from_f32(0.03 * next()).to_f32() } else { 0.03 * next() }).collect() };
    let shared = [dense(ff * hidden), dense(ff * hidden), dense(hidden * ff)];
    QuantExpertsData { hidden, ff, experts: count, gate: (GgmlType::Q2_0, gate), up: (GgmlType::Q2_0, up), down: (GgmlType::Q2_0, down), shared: Some(shared), routing: Routing::Softmax }
}

pub(crate) fn copy(d: &QuantExpertsData) -> QuantExpertsData {
    QuantExpertsData { hidden: d.hidden, ff: d.ff, experts: d.experts, gate: d.gate.clone(), up: d.up.clone(), down: d.down.clone(), shared: d.shared.clone(), routing: d.routing.clone() }
}

/// `rows` rows of inputs and of router logits (no two alike).
pub(crate) fn inputs(seed: u64, rows: usize, hidden: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut next = rng(seed);
    ((0..rows * hidden).map(|_| next()).collect(), (0..rows * (count + 1)).map(|_| 3.0 * next()).collect())
}

/// The worst difference between `got` and `want`, over `want`'s RMS.
pub(crate) fn worst(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    got.iter().zip(want).map(|(g, w)| (*g as f64 - *w as f64).abs()).fold(0f64, f64::max) / rms.max(1e-30)
}

/// The host's experts are the definition's: each row's top `k` routed experts by logit, softmax-weighted among
/// themselves, and the shared one by its gate's sigmoid, each `down(silu(gate x) * up x)`, from every tensor
/// dequantised whole and summed in three loops.
#[test]
fn the_hosts_experts_are_three_plain_loops() {
    let (hidden, ff, count, top_k, rows) = (256usize, 128usize, 12usize, 3usize, 5usize);
    let data = experts(11, hidden, ff, count, false);
    let (x, logits) = inputs(12, rows, hidden, count);
    let whole = |(t, bytes): &(GgmlType, Vec<u8>), n: usize| {
        let mut out = vec![0f32; n];
        ggml_quants::dequantize(*t, bytes, &mut out).unwrap();
        out
    };
    let (g, u, d) = (whole(&data.gate, count * ff * hidden), whole(&data.up, count * ff * hidden), whole(&data.down, count * hidden * ff));
    let mut want = vec![0f64; rows * hidden];
    for r in 0..rows {
        let l = &logits[r * (count + 1)..(r + 1) * (count + 1)];
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let top = &order[..top_k];
        let sum: f64 = top.iter().map(|&e| (l[e] as f64).exp()).sum();
        let mut parts: Vec<(f64, &[f32], &[f32], &[f32])> = top.iter().map(|&e| ((l[e] as f64).exp() / sum, &g[e * ff * hidden..(e + 1) * ff * hidden], &u[e * ff * hidden..(e + 1) * ff * hidden], &d[e * hidden * ff..(e + 1) * hidden * ff])).collect();
        let sh = data.shared.as_ref().expect("a shared expert");
        parts.push((1.0 / (1.0 + (-l[count] as f64).exp()), &sh[0], &sh[1], &sh[2]));
        for (w, g, u, d) in parts {
            let act: Vec<f64> = (0..ff)
                .map(|j| {
                    let dot = |m: &[f32]| (0..hidden).map(|c| m[j * hidden + c] as f64 * x[r * hidden + c] as f64).sum::<f64>();
                    let (a, b) = (dot(g), dot(u));
                    a / (1.0 + (-a).exp()) * b
                })
                .collect();
            for i in 0..hidden {
                want[r * hidden + i] += w * (0..ff).map(|j| d[i * ff + j] as f64 * act[j]).sum::<f64>();
            }
        }
    }
    let want: Vec<f32> = want.iter().map(|v| *v as f32).collect();
    let host = quant_experts_cpu(data).unwrap();
    let got = host.forward(&Tensor::from_vec(x, vec![rows, hidden]), &Tensor::from_vec(logits, vec![rows, count + 1]), top_k).to_host();
    let e = worst(got.data(), &want);
    eprintln!("the host's experts against three loops: the worst error {e:.2e} of the RMS");
    assert!(e < 1e-4, "the host's experts: {e}");
}

/// The GPU's experts' sums by each of its ways in: routed on the host (`Experts::forward`, `moe_rows`), routed on
/// the GPU (`moe_routed`), and those added to the streams (`moe_routed_into`; four streams, their sums the row's
/// weights' times it). None where the device routes no such rows.
#[allow(clippy::type_complexity)]
fn gpu_sums(b: &WgpuBackend, moe: &dyn Experts, x: &[f32], logits: &[f32], rows: usize, hidden: usize, count: usize, top_k: usize) -> (Vec<f32>, Option<Vec<f32>>, Option<(Vec<f32>, Vec<f32>)>) {
    let hosted = moe.forward(&Tensor::from_vec(x.to_vec(), vec![rows, hidden]), &Tensor::from_vec(logits.to_vec(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
    let streams = 4;
    let (xd, ld, out, xs, post) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden), b.vec(rows * streams * hidden), b.vec(rows * streams));
    DeviceChain::upload(b, &xd, x);
    DeviceChain::upload(b, &ld, logits);
    let posts: Vec<f32> = (0..rows * streams).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
    let before: Vec<f32> = (0..rows * streams * hidden).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
    DeviceChain::upload(b, &post, &posts);
    DeviceChain::upload(b, &xs, &before);
    let mut rec = b.begin();
    rec.keep_groups(false);
    if !rec.moe_routed(moe, &xd, &out, &ld, top_k, rows) {
        return (hosted, None, None);
    }
    rec.read(&out);
    assert!(rec.moe_routed_into(moe, &xd, &xs, &post, &ld, top_k, rows, streams));
    rec.read(&xs);
    let mut read = rec.finish();
    let after = read.pop().unwrap();
    let routed = read.pop().unwrap();
    // what the streams should hold, from the routed sums
    let want: Vec<f32> = (0..rows * streams * hidden).map(|i| before[i] + posts[i / hidden] * routed[(i / (streams * hidden)) * hidden + i % hidden]).collect();
    (hosted, Some(routed), Some((after, want)))
}

/// The GPU's Q2_0 experts are the host's: a small layer (hidden 256, 12 experts, 3 a row) and Flash-Next's shape
/// (2560 by 640, 24 experts, 10 a row), for a step's one row, a check's three, and a prompt's 40, 70 and 512, routed on
/// the host and on the GPU, in f32 and (where the adapter has them) on the tensor cores, whose inputs are f16's.
#[test]
fn the_gpus_q2_0_experts_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    for (hidden, ff, count, top_k) in [(256usize, 128usize, 12usize, 3usize), (2560, 640, 24, 10)] {
        for half in [true, false] {
            let data = experts(31 + hidden as u64, hidden, ff, count, half);
            let host = quant_experts_cpu(copy(&data)).unwrap();
            let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("room for the experts");
            f32s.coop = false;
            let cores = QuantMoe::try_new(&b, data, 0).ok().expect("room for the experts");
            for rows in [1usize, 3, 40, 70, 512] {
                let (x, logits) = inputs(7 + rows as u64, rows, hidden, count);
                let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
                for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
                    if !moe.coop && what != "f32" {
                        continue;
                    }
                    let (hosted, routed, into) = gpu_sums(&b, moe, &x, &logits, rows, hidden, count, top_k);
                    let e = worst(&hosted, &want);
                    let mut line = format!("{hidden} by {ff}, {count} experts ({} shared), {rows} rows, {what}: routed on the host {e:.2e}", if half { "an f16" } else { "an f32" });
                    // f32's sums to their rounding; the tensor cores' inputs are rounded to f16 (a block of 16
                    // jobs: the host's routing of any rows but a step's one, the GPU's of a prompt's), as the
                    // shared expert's f16 matmul rounds a prompt's
                    let shared = half && rows > 8 && coop_on(&b.gpu);
                    let bound = |cores: bool| if cores || shared { 6e-3 } else { 2e-5 };
                    assert!(e < bound(moe.coop && rows > 1), "{line}");
                    if let Some(routed) = routed {
                        let e = worst(&routed, &want);
                        line += &format!(", on the GPU {e:.2e}");
                        assert!(e < bound(moe.coop && rows > FEW_MAX), "{line}");
                        let (after, expect) = into.unwrap();
                        let e = worst(&after, &expect);
                        line += &format!(", into the streams {e:.2e}");
                        assert!(e < 1e-4, "{line}");
                    }
                    eprintln!("{line}");
                }
            }
        }
    }
}

/// A layer a card holds some of whose experts is the layer it holds all of, bit for bit: a small layer (hidden
/// 256, 12 experts, 3 a row) with 5 slots against the same all on the card, Q2_0 and IQ2_S, over passes of a
/// step's row, a check's three and a prompt's 40 and 70, in f32 and on the tensor cores, by every way in; and
/// its slots took in experts the passes used. Each pass's time is printed (new kernels: a slow one a stop sign).
#[test]
fn a_cards_share_of_a_layers_experts_is_all_of_them() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !b.host_weights() {
        eprintln!("this device's kernels do not read the host's memory");
        return;
    }
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    let (hidden, ff, count, top_k) = (256usize, 128usize, 12usize, 3usize);
    for gu in [GgmlType::Q2_0, GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
        for cores in [false, true] {
            if cores && !coop_on(&b.gpu) {
                continue;
            }
            let data = experts_of(77, hidden, ff, count, true, gu);
            let mut whole = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("room for the experts");
            let mut part = QuantMoe::make(&b, data, 0, Some(5)).ok().expect("room for five of the experts");
            (whole.coop, part.coop) = (cores, cores);
            let cache = Arc::clone(part.cache.as_ref().expect("a layer of 12 experts in 5 slots is part-held"));
            assert!(whole.cache.is_none());
            for (pass, rows) in [1usize, 3, 1, 40, 70, 1, 3, 40, 1].into_iter().enumerate() {
                let (x, logits) = inputs(100 + pass as u64, rows, hidden, count);
                let want = gpu_sums(&b, &whole, &x, &logits, rows, hidden, count, top_k);
                let clock = std::time::Instant::now();
                let got = gpu_sums(&b, &part, &x, &logits, rows, hidden, count, top_k);
                let ms = clock.elapsed().as_secs_f64() * 1e3;
                let what = format!("{gu:?}, {} pass {pass} of {rows} rows ({ms:.0} ms)", if cores { "the tensor cores," } else { "f32," });
                assert_eq!(bits(&got.0), bits(&want.0), "{what}: routed on the host");
                assert_eq!(got.1.as_deref().map(bits), want.1.as_deref().map(bits), "{what}: routed on the GPU");
                assert_eq!(got.2.as_ref().map(|(a, _)| bits(a)), want.2.as_ref().map(|(a, _)| bits(a)), "{what}: into the streams");
            }
            let (read, brought) = (cache.counts[0].load(Ordering::Relaxed), cache.counts[1].load(Ordering::Relaxed));
            // (the slots as the card left them: each still one expert's)
            let mut held: Vec<u32> = cache.host.lock().unwrap().map.iter().copied().filter(|s| *s != MISS).collect();
            held.sort_unstable();
            eprintln!("{gu:?}, {}: {read} experts read from the host's memory, {brought} of them taken into the card's {} slots", if cores { "the tensor cores" } else { "f32" }, held.len());
            assert!(read > 0 && brought > 0 && held == [0, 1, 2, 3, 4], "{gu:?}: the card's slots took in what the passes used: {held:?}");
        }
    }
}

/// A prompt's experts that take few of its rows go through the few rows' kernel, whose sums are f32's: a layer
/// of 192 experts (3 a row) gives each a row or two of a prompt's 40 or 70, and routed on the GPU where it has
/// the tensor cores the layer is the host's to f32's rounding (the shared expert f32 too), where a block of the
/// tensor cores' rounds its inputs to f16 (some 1e-3 of the RMS).
#[test]
fn a_prompts_experts_of_few_rows_are_f32s() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !coop_on(&b.gpu) {
        return;
    }
    let (hidden, ff, count, top_k) = (256usize, 128usize, 192usize, 3usize);
    for gu in [GgmlType::Q2_0, GgmlType::IQ2_S] {
        let data = experts_of(91, hidden, ff, count, false, gu);
        let host = quant_experts_cpu(copy(&data)).unwrap();
        let moe = QuantMoe::try_new(&b, data, 0).ok().expect("room for the experts");
        assert!(moe.coop, "the tensor cores take this device's prompts");
        for rows in [40usize, 70] {
            let (x, logits) = inputs(17 + rows as u64, rows, hidden, count);
            let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
            let (_, routed, into) = gpu_sums(&b, &moe, &x, &logits, rows, hidden, count, top_k);
            let e = worst(&routed.expect("a prompt's rows routed on the GPU"), &want);
            let (after, expect) = into.expect("and into the streams");
            eprintln!("{gu:?} gate and up, {count} experts, {rows} rows: routed on the GPU {e:.2e}, into the streams {:.2e}", worst(&after, &expect));
            assert!(e < 5e-5, "{gu:?}, {rows} rows: {e:.2e}");
        }
    }
}

/// What the grid types' kernels take for granted of ggml's tables: a grid's bytes are one of four values at most,
/// so an entry is two bits a weight (each grid given back from its codes and values), and IQ2_XXS's sign pattern
/// `i` is `i` with an eighth sign that makes its set bits even (the kernel computes it).
#[test]
fn the_grids_are_two_bits_a_weight() {
    for kind in [Kind::IQ2_S, Kind::IQ2_XXS, Kind::IQ2_XS, Kind::IQ1_M] {
        let (grid, signed) = kind.grid().unwrap();
        let (codes, levels) = grid_codes(grid, signed);
        assert_eq!(codes.len(), grid.len());
        for (e, c) in grid.iter().zip(&codes) {
            for (j, b) in e.to_le_bytes().iter().enumerate() {
                let want = if signed { *b as i8 as f32 } else { *b as f32 };
                assert_eq!(levels[((c >> (2 * j)) & 3) as usize], want, "{kind:?}: weight {j} of entry {e:#x}");
            }
        }
    }
    for (i, s) in ggml_quants::iq_tables::KSIGNS_IQ2XS.iter().enumerate() {
        assert_eq!(*s as u32, i as u32 | ((i as u32).count_ones() & 1) << 7, "sign pattern {i}");
    }
}

/// Experts in slots ([`QuantSlots`]) are the host's, for layers of DeepSeek-V4.1's GGUF's mixed types (its gate's and
/// up's type, its down's: 1-bit beside 2-bit grid types), slots written by the queue and in place: three experts'
/// records (each its three matrices' blocks one after another, as a GGUF's expert record holds them) put in, a step's
/// jobs of one row and a prompt's of 40 rows made, each against the definition (each matrix decoded whole by
/// `ggml_quants`; the gate's and up's sums rounded to bf16, clamped at the limit, `silu(gate) * up` rounded to bf16,
/// then down) in f64; a slot copied into another set's gives its expert's outputs to the bit.
#[test]
fn experts_in_slots_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (hidden, ff, limit) = (512usize, 256usize, 1.5f32);
    let bf16 = |v: f64| half::bf16::from_f32(v as f32).to_f32() as f64;
    for (types, mapped) in [([GgmlType::IQ2_XS, GgmlType::IQ1_M], true), ([GgmlType::IQ2_XXS, GgmlType::IQ2_XS], false), ([GgmlType::IQ1_M, GgmlType::IQ2_XXS], true)] {
        let mut next = rng(5 + types[0] as u64);
        let experts = 3usize;
        let bytes = |t: GgmlType, n: usize, k: usize| n * k / 256 * t.type_size();
        let (gb, db) = (bytes(types[0], ff, hidden), bytes(types[1], hidden, ff));
        let layout = QuantLayout {
            bytes: 2 * gb + db,
            proj: [
                QuantProj { at: 0, dtype: types[0], k: hidden, n: ff },
                QuantProj { at: gb, dtype: types[0], k: hidden, n: ff },
                QuantProj { at: 2 * gb, dtype: types[1], k: ff, n: hidden },
            ],
        };
        let records: Vec<Vec<u8>> = (0..experts).map(|_| [random_grid(&mut next, types[0], 2, ff, hidden), random_grid(&mut next, types[1], 1, hidden, ff)].concat()).collect();
        let slots = if mapped { b.quant_slots_mapped(4, hidden, ff, types) } else { b.quant_slots(4, hidden, ff, types) }.expect("room for four slots");
        assert!(slots.fits(&layout));
        let refs: Vec<&[u8]> = records.iter().map(|r| &r[..]).collect();
        slots.put_many(1, &refs, &layout);
        // the definition's outputs of expert `e` (slot e + 1) for rows `x`
        let want = |e: usize, x: &[f32]| -> Vec<f32> {
            let whole = |p: &QuantProj| {
                let mut out = vec![0f32; p.n * p.k];
                ggml_quants::dequantize(p.dtype, &records[e][p.at..p.at + bytes(p.dtype, p.n, p.k)], &mut out).unwrap();
                out
            };
            let (g, u, d) = (whole(&layout.proj[0]), whole(&layout.proj[1]), whole(&layout.proj[2]));
            x.chunks_exact(hidden)
                .flat_map(|row| {
                    let dot = |m: &[f32], j: usize, v: &[f64]| (0..v.len()).map(|c| m[j * v.len() + c] as f64 * v[c]).sum::<f64>();
                    let xr: Vec<f64> = row.iter().map(|&v| v as f64).collect();
                    let act: Vec<f64> = (0..ff)
                        .map(|j| {
                            let (gs, us) = (bf16(dot(&g, j, &xr)).min(limit as f64), bf16(dot(&u, j, &xr)).clamp(-limit as f64, limit as f64));
                            bf16(gs / (1.0 + (-gs).exp()) * us)
                        })
                        .collect();
                    (0..hidden).map(|i| dot(&d, i, &act) as f32).collect::<Vec<_>>()
                })
                .collect()
        };
        let x: Vec<f32> = (0..40 * hidden).map(|_| next()).collect();
        // a step's: each expert on its own row; a prompt's: one expert on 40 rows, another on one
        let step: Vec<(usize, &[f32])> = (0..experts).map(|e| (e + 1, &x[e * hidden..(e + 1) * hidden])).collect();
        let got = slots.forward(&step, limit);
        for e in 0..experts {
            let err = worst(&got[e], &want(e, step[e].1));
            eprintln!("{types:?}{}: a step's expert {e} {err:.2e}", if mapped { ", written in place" } else { "" });
            assert!(err < 2e-2, "{types:?}: a step's expert {e}: {err}");
        }
        let prompt: Vec<(usize, &[f32])> = vec![(2, &x[..]), (3, &x[..hidden])];
        let got = slots.forward(&prompt, limit);
        let (e0, e1) = (worst(&got[0], &want(1, &x)), worst(&got[1], &want(2, &x[..hidden])));
        eprintln!("{types:?}: a prompt's 40 rows {e0:.2e}, its expert of one row {e1:.2e}");
        assert!(e0 < 3e-2 && e1 < 2e-2, "{types:?}: a prompt's: {e0}, {e1}");
        // (a step's row the same bits beside a prompt's rows as alone)
        let alone = slots.forward(&[(3, &x[..hidden])], limit);
        assert_eq!(got[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), alone[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{types:?}: a row's output hung on its company");
        // a slot copied into another set's
        let other = b.quant_slots(2, hidden, ff, types).expect("room for two slots");
        slots.copy_slot_marked(2, &other, 1, &[]);
        let copied = other.forward(&[(1, step[1].1)], limit);
        let here = slots.forward(&[(2, step[1].1)], limit);
        assert_eq!(copied[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), here[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{types:?}: a copied slot");
    }
}

/// What a decode step's experts in slots ([`QuantSlots`]) take at DeepSeek-V4.1's shape (5120 to 2304), each pair of
/// types its 2-bit GGUF has most: six experts' one-row jobs in one call, against their bytes.
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_experts_in_slots() {
    let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
    let (hidden, ff, limit) = (5120usize, 2304usize, 10.0f32);
    for types in [[GgmlType::IQ1_M, GgmlType::IQ1_M], [GgmlType::IQ2_XXS, GgmlType::IQ1_M], [GgmlType::IQ2_XS, GgmlType::IQ1_M], [GgmlType::IQ2_XXS, GgmlType::IQ2_XS]] {
        let mut next = rng(41);
        let gb = ff * hidden / 256 * types[0].type_size();
        let db = hidden * ff / 256 * types[1].type_size();
        let layout = QuantLayout {
            bytes: 2 * gb + db,
            proj: [QuantProj { at: 0, dtype: types[0], k: hidden, n: ff }, QuantProj { at: gb, dtype: types[0], k: hidden, n: ff }, QuantProj { at: 2 * gb, dtype: types[1], k: ff, n: hidden }],
        };
        let slots = b.quant_slots_mapped(64, hidden, ff, types).expect("room for 64 slots");
        let record = [random_grid(&mut next, types[0], 2, ff, hidden), random_grid(&mut next, types[1], 1, hidden, ff)].concat();
        let records: Vec<&[u8]> = (0..64).map(|_| &record[..]).collect();
        let putting = std::time::Instant::now();
        slots.put_many(0, &records, &layout);
        let put = putting.elapsed().as_secs_f64() / 64.0;
        let x: Vec<f32> = (0..6 * hidden).map(|_| next()).collect();
        // six experts in slots far apart (none in another's cache lines)
        let jobs: Vec<(usize, &[f32])> = (0..6).map(|j| (j * 10, &x[j * hidden..(j + 1) * hidden])).collect();
        slots.forward(&jobs, limit);
        let _ = crate::profile::take_kernels();
        let reps = 50;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            slots.forward(&jobs, limit);
        }
        let secs = t.elapsed().as_secs_f64() / reps as f64;
        let bytes = 6.0 * layout.bytes as f64;
        eprintln!("{types:?}: six experts' step in {:.0} us a call ({:.0} GB/s of their weights; a record put in at {:.2} ms)", secs * 1e6, bytes / secs / 1e9, put * 1e3);
        // (OAIY_CHAIN_PROFILE: each kernel's own time on the GPU)
        for (name, ms, n) in crate::profile::take_kernels() {
            eprintln!("    {name}: {:.1} us a dispatch", ms * 1e3 / n as f64);
        }
    }
}

/// A GGUF's matrices kept together ([`QuantGroup`]) give each part's product in one pass of a device program's
/// [`crate::exl3::ProgramStep::MulGroup`]: DeepSeek-V4.1's `wo_a` shape (eight Q5_K slices of 1024 by 4096, each on its
/// own row of the heads' outputs) against the definition (each part decoded whole by `ggml_quants`) in f64.
#[test]
fn a_quant_groups_parts_are_each_the_definitions() {
    use crate::exl3::{run_program, GroupMat, ProgramStep};
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (parts, n, k) = (8usize, 1024usize, 4096usize);
    let mut next = rng(33);
    let mats: Vec<Vec<u8>> = (0..parts).map(|_| random_k(&mut next, GgmlType::Q5_K, 1, n, k)).collect();
    let refs: Vec<&[u8]> = mats.iter().map(|m| &m[..]).collect();
    let group = b.quant_group(GgmlType::Q5_K, &refs, n, k).expect("room for the group");
    assert_eq!((group.kn(), group.count()), ((k, n), parts));
    let x: Vec<f32> = (0..parts * k).map(|_| next()).collect();
    let jobs: Vec<(u32, u32)> = (0..parts as u32).map(|g| (g, g)).collect();
    let got = run_program(&b, &[ProgramStep::Input(&x), ProgramStep::MulGroup { g: GroupMat::Quant(&group), x: 0, jobs: &jobs }], &[(1, 0, parts * n)]).expect("the program runs").pop().unwrap();
    for (g, m) in mats.iter().enumerate() {
        let mut w = vec![0f32; n * k];
        ggml_quants::dequantize(GgmlType::Q5_K, m, &mut w).unwrap();
        let want: Vec<f32> = (0..n).map(|r| (0..k).map(|c| w[r * k + c] as f64 * x[g * k + c] as f64).sum::<f64>() as f32).collect();
        let e = worst(&got[g * n..(g + 1) * n], &want);
        assert!(e < 5e-5, "part {g}: {e:.2e} of the RMS");
    }
}

/// A device program's held experts in slots of a GGUF's ([`crate::exl3::ProgramStep::Candidates`], then `Held` of
/// [`QuantSlots`]) are the `k` best of the router's scores, those the table holds made as the slots make them alone (to
/// the bit) and those it does not skipped (their rows as they were).
#[test]
fn a_programs_held_quant_experts_are_the_slots_own() {
    use crate::exl3::{run_program, HeldSlots, ProgramStep};
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (hidden, ff, limit, n, k) = (512usize, 256usize, 1.5f32, 16usize, 4usize);
    let types = [GgmlType::IQ2_XXS, GgmlType::IQ1_M];
    let mut next = rng(21);
    let gb = ff * hidden / 256 * types[0].type_size();
    let layout = QuantLayout {
        bytes: 2 * gb + hidden * ff / 256 * types[1].type_size(),
        proj: [QuantProj { at: 0, dtype: types[0], k: hidden, n: ff }, QuantProj { at: gb, dtype: types[0], k: hidden, n: ff }, QuantProj { at: 2 * gb, dtype: types[1], k: ff, n: hidden }],
    };
    let slots = b.quant_slots(4, hidden, ff, types).expect("room for four slots");
    // experts 3, 7 and 11 of the layer in slots 2, 0 and 1
    let held = [(3usize, 2u32), (7, 0), (11, 1)];
    for &(_, s) in &held {
        let record = [random_grid(&mut next, types[0], 2, ff, hidden), random_grid(&mut next, types[1], 1, hidden, ff)].concat();
        slots.put(s as usize, &record, &layout);
    }
    let mut words = vec![u32::MAX; n];
    for &(e, s) in &held {
        words[e] = s;
    }
    let table = b.vec_of_bytes(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
    let x: Vec<f32> = (0..hidden).map(|_| next()).collect();
    // logits that put experts 7, 5, 3 and 12 first by score (the bias moving 12 above 11), 11 fifth
    let mut logits = vec![-3.0f32; n];
    for (e, v) in [(7, 4.0), (5, 3.0), (3, 2.0), (11, 1.5), (12, 1.0)] {
        logits[e] = v;
    }
    let mut bias = vec![0.0f32; n];
    bias[12] = 0.2;
    let steps = [
        ProgramStep::Input(&logits),
        ProgramStep::Input(&x),
        ProgramStep::Candidates { logits: 0, bias: &bias, table: &table, k },
        ProgramStep::Held { slots: HeldSlots::Quant(&slots), x: 1, cands: 2, limit },
    ];
    let got = run_program(&b, &steps, &[(2, 0, 4 * k), (3, 0, k * hidden)]).expect("the program runs");
    let bits: Vec<u32> = got[0].iter().map(|v| v.to_bits()).collect();
    assert_eq!(&bits[2 * k..3 * k], &[7, 5, 3, 12], "the candidates, best first");
    assert_eq!(&bits[3 * k..], &[0, u32::MAX, 2, u32::MAX], "their slots");
    for (r, slot) in [(0usize, 0usize), (2, 2)] {
        let alone = slots.forward(&[(slot, &x[..])], limit).pop().expect("one output");
        let mine = &got[1][r * hidden..(r + 1) * hidden];
        assert!(mine.iter().zip(&alone).all(|(a, b)| a.to_bits() == b.to_bits()), "candidate {r}: the slots' own output");
    }
}

/// The GPU's experts whose gate and up matrices are a grid type's (IQ2_S, IQ2_XXS, IQ1_M; their down ones Q2_0: the
/// GSQ-RCO IQ2_XS file's layers) are the host's reference: a small layer (hidden 256, 12 experts, 3 a row) for a
/// step's row, a check's three and a prompt's 40 and 70, routed on the host and on the GPU, in f32 and (where the
/// adapter has them) on the tensor cores, whose weights and inputs are rounded to f16; with
/// OAIY_GRID_EXPERTS_FULL Flash-Next's shape too (2560 by 640, 24 experts, 10 a row, and 512 rows). Each case's
/// time is printed: these kernels read their grids from a buffer, and a slow one is a stop sign.
#[test]
fn the_gpus_grid_experts_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let full = std::env::var_os("OAIY_GRID_EXPERTS_FULL").is_some();
    let shapes: &[(usize, usize, usize, usize)] = if full { &[(256, 128, 12, 3), (2560, 640, 24, 10)] } else { &[(256, 128, 12, 3)] };
    for gu in [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ2_XS, GgmlType::IQ1_M] {
        for &(hidden, ff, count, top_k) in shapes {
            let data = experts_of(53 + hidden as u64, hidden, ff, count, true, gu);
            let host = quant_experts_cpu(copy(&data)).unwrap();
            let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("the grid type's experts on the GPU");
            f32s.coop = false;
            let cores = QuantMoe::try_new(&b, data, 0).ok().expect("the grid type's experts on the GPU");
            let all_rows: &[usize] = if full && hidden > 256 { &[1, 3, 40, 512] } else { &[1, 3, 40, 70] };
            for &rows in all_rows {
                let (x, logits) = inputs(7 + rows as u64, rows, hidden, count);
                let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
                for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
                    if !moe.coop && what != "f32" {
                        continue;
                    }
                    let clock = std::time::Instant::now();
                    let (hosted, routed, into) = gpu_sums(&b, moe, &x, &logits, rows, hidden, count, top_k);
                    let ms = clock.elapsed().as_secs_f64() * 1e3;
                    let e = worst(&hosted, &want);
                    let mut line = format!("{gu:?} gate and up, {hidden} by {ff}, {count} experts, {rows} rows, {what} ({ms:.0} ms): routed on the host {e:.2e}");
                    // f32's sums to their rounding; the tensor cores' weights and inputs are rounded to f16, as
                    // the shared expert's f16 matmul rounds a prompt's
                    let shared = rows > 8 && coop_on(&b.gpu);
                    let bound = |cores: bool| if cores || shared { 8e-3 } else { 5e-5 };
                    assert!(e < bound(moe.coop && rows > 1), "{line}");
                    if let Some(routed) = routed {
                        let e = worst(&routed, &want);
                        line += &format!(", on the GPU {e:.2e}");
                        assert!(e < bound(moe.coop && rows > FEW_MAX), "{line}");
                        let (after, expect) = into.unwrap();
                        let e = worst(&after, &expect);
                        line += &format!(", into the streams {e:.2e}");
                        assert!(e < 1e-4, "{line}");
                    }
                    eprintln!("{line}");
                }
            }
        }
    }
}

/// Q4_K and Q6_K experts (a Q4_K_M GGUF's: Qwen3.5-MoE's and MiniCPM-V 4.7 35B-A3B's) on the GPU are the host's:
/// the f32 kernel's sums and the tensor cores', routed on the host and on the GPU.
#[test]
fn the_gpus_k_quant_experts_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (hidden, ff, count, top_k) = (512, 256, 12, 3);
    for (gu, down) in [(GgmlType::Q4_K, GgmlType::Q4_K), (GgmlType::Q4_K, GgmlType::Q6_K), (GgmlType::Q6_K, GgmlType::Q6_K), (GgmlType::Q5_K, GgmlType::Q5_K)] {
        let data = k_experts(91, hidden, ff, count, gu, down);
        let host = quant_experts_cpu(copy(&data)).unwrap();
        let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("K-quant experts on the GPU");
        f32s.coop = false;
        let cores = QuantMoe::try_new(&b, data, 0).ok().expect("K-quant experts on the GPU");
        for rows in [1, 3, 40, 70] {
            let (x, logits) = inputs(17 + rows as u64, rows, hidden, count);
            let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
            for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
                if !moe.coop && what != "f32" {
                    continue;
                }
                let (hosted, routed, _) = gpu_sums(&b, moe, &x, &logits, rows, hidden, count, top_k);
                let e = worst(&hosted, &want);
                let mut line = format!("{gu:?} gate and up, {down:?} down, {rows} rows, {what}: routed on the host {e:.2e}");
                let shared = rows > 8 && coop_on(&b.gpu);
                let bound = |cores: bool| if cores || shared { 8e-3 } else { 5e-5 };
                assert!(e < bound(moe.coop && rows > 1), "{line}");
                if let Some(routed) = routed {
                    let e = worst(&routed, &want);
                    line += &format!(", on the GPU {e:.2e}");
                    assert!(e < bound(moe.coop && rows > FEW_MAX), "{line}");
                }
                eprintln!("{line}");
            }
        }
    }
}

/// LFM2's experts (`Routing::Sigmoid`, no shared expert; Q4_K gate and up, Q6_K down, as LFM2.5-24B-A2B's Q4_K_M
/// GGUF keeps them) on the GPU are the host's, routed on the host and on the GPU; and the host's routing is
/// transformers' `Lfm2MoeTopKRouter`: the top k by sigmoid plus bias, weighted by the sigmoids alone over their sum.
#[test]
fn lfm2s_sigmoid_routed_experts_on_the_gpu_are_the_hosts() {
    let (hidden, ff, count, top_k) = (512, 256, 16, 4);
    let bias: Vec<f32> = (0..count).map(|e| ((e * 7) % 5) as f32 * 0.15 - 0.3).collect();
    let routing = Routing::Sigmoid { bias: bias.clone(), scale: 1.0 };
    let (_, logits) = inputs(95, 64, hidden, count);
    let mut moved = 0;
    for l in logits.chunks_exact(count + 1) {
        let score: Vec<f64> = l[..count].iter().map(|&v| 1.0 / (1.0 + (-v as f64).exp())).collect();
        let by = |key: &dyn Fn(usize) -> f64| {
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap().then(a.cmp(&b)));
            order.truncate(top_k);
            order
        };
        let top = by(&|e| score[e] + bias[e] as f64);
        moved += usize::from(top != by(&|e| score[e]));
        let sum: f64 = top.iter().map(|&e| score[e]).sum::<f64>() + 1e-6;
        let got = routing.route(l, top_k);
        assert_eq!(got.len(), top_k + 1);
        assert_eq!(got[top_k], (count, 0.0), "no shared expert");
        for (&(e, w), &want) in got.iter().zip(&top) {
            assert_eq!(e, want);
            assert!((w as f64 - score[e] / sum).abs() < 1e-6, "expert {e}: {w} where {}", score[e] / sum);
        }
    }
    assert!(moved > 0, "the bias changes some row's picks");
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut data = k_experts(93, hidden, ff, count, GgmlType::Q4_K, GgmlType::Q6_K);
    data.shared = None;
    data.routing = routing;
    let host = quant_experts_cpu(copy(&data)).unwrap();
    let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("LFM2's experts on the GPU");
    f32s.coop = false;
    let cores = QuantMoe::try_new(&b, data, 0).ok().expect("LFM2's experts on the GPU");
    for rows in [1, 3, 40, 70] {
        let (x, logits) = inputs(97 + rows as u64, rows, hidden, count);
        let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
        for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
            if !moe.coop && what != "f32" {
                continue;
            }
            let (hosted, routed, into) = gpu_sums(&b, moe, &x, &logits, rows, hidden, count, top_k);
            let e = worst(&hosted, &want);
            let mut line = format!("LFM2's experts, {rows} rows, {what}: routed on the host {e:.2e}");
            let shared = rows > 8 && coop_on(&b.gpu);
            let bound = |cores: bool| if cores || shared { 8e-3 } else { 5e-5 };
            assert!(e < bound(moe.coop && rows > 1), "{line}");
            if let Some(routed) = routed {
                let e = worst(&routed, &want);
                line += &format!(", on the GPU {e:.2e}");
                assert!(e < bound(moe.coop && rows > FEW_MAX), "{line}");
                let (after, want) = into.expect("the streams' sums");
                let e = worst(&after, &want);
                line += &format!(", into streams {e:.2e}");
                assert!(e < 1e-5, "{line}");
            }
            eprintln!("{line}");
        }
    }
}

/// The tensor cores' matmul is the f32 kernel's where the inputs are f16's (a Q2_0 weight is one exactly, so the
/// tiles lose nothing): a group's jobs in blocks of 16, 32, 64 and 128 (some places unused, some blocks none)
/// against a job each, for the gate and up group and the down one.
#[test]
fn the_tensor_cores_matmul_is_f32s_on_f16_inputs() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !coop_on(&b.gpu) {
        return;
    }
    let (hidden, ff, count) = (2560usize, 640usize, 6usize);
    let moe = QuantMoe::try_new(&b, experts(41, hidden, ff, count, true), 0).ok().expect("room for the experts");
    let mut next = rng(42);
    for (g, matrices) in [(&moe.gu, 2 * count), (&moe.down, count)] {
        // 150 jobs over the group's matrices (unevenly), each on one of 40 input rows
        let inputs = 40usize;
        let x: Vec<f32> = (0..inputs * g.k).map(|_| half::f16::from_f32(2.0 * next()).to_f32()).collect();
        let jobs: Vec<u32> = (0..150u32).flat_map(|j| [(j * j % 7 + j % 3) % matrices as u32, (j * 11) % inputs as u32]).collect();
        let count = jobs.len() / 2;
        let (xd, jd) = (b.vec(x.len()), crate::exl3::u32_vec(&b, &jobs));
        DeviceChain::upload(&b, &xd, &x);
        let run = |order: Option<(Vec<u32>, usize)>| -> Vec<f32> {
            let y = b.vec(count * g.n);
            let mut rec = crate::chain::Recorder::new(&b);
            rec.keep_groups(false);
            match &order {
                Some((o, rows)) => {
                    let ov = crate::exl3::u32_vec(&b, o);
                    moe.group_pass(&mut rec, g, &xd, &jd, count, Order::Blocks(&ov, o.len() / rows, *rows), &y);
                }
                None => moe.group_pass(&mut rec, g, &xd, &jd, count, Order::Jobs, &y),
            }
            rec.read(&y);
            Box::new(rec).finish().pop().unwrap()
        };
        let want = run(None);
        for rows in [16usize, 32, 64, 128] {
            let mut order = many_order(&jobs, rows);
            // (a block of none between the others, as a GPU's grouping leaves them)
            order.splice(rows..rows, vec![crate::exl3::NONE; rows]);
            let got = run(Some((order, rows)));
            let e = worst(&got, &want);
            eprintln!("[{}, {}], blocks of {rows} on the tensor cores against a job each in f32: the worst error {e:.2e} of the RMS", g.n, g.k);
            // (the sums of 2,560 products in another order)
            assert!(e < 1e-4, "blocks of {rows}: {e}");
        }
        // and the f32 kernel's blocks of 2 to 8
        for rows in 2..=FEW_MAX {
            let got = run(Some((many_order(&jobs, rows), rows)));
            let e = worst(&got, &want);
            assert!(e < 2e-6, "[{}, {}], blocks of {rows} in f32: {e}", g.n, g.k);
        }
    }
}

/// A step's experts with their bind groups kept are the same sums a second time, and a layer that goes gives its
/// bytes back to the budget.
#[test]
fn a_kept_step_repeats_and_a_layer_gives_its_bytes_back() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (hidden, ff, count, top_k) = (256usize, 128usize, 12usize, 3usize);
    let used = b.used.load(Ordering::Relaxed);
    let data = experts(5, hidden, ff, count, true);
    let host = quant_experts_cpu(copy(&data)).unwrap();
    let moe = b.quant_experts(data).unwrap();
    assert!(DeviceChain::holds_experts(&b, moe.as_ref()) && b.used.load(Ordering::Relaxed) > used, "the experts on the GPU, counted");
    let (xd, ld, out) = (b.vec(hidden), b.vec(count + 1), b.vec(hidden));
    for step in 0..3u64 {
        let (x, logits) = inputs(90 + step, 1, hidden, count);
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &ld, &logits);
        let mut rec = b.begin();
        assert!(rec.moe_routed(moe.as_ref(), &xd, &out, &ld, top_k, 1));
        rec.read(&out);
        let got = rec.finish().pop().unwrap();
        let want = host.forward(&Tensor::from_vec(x, vec![1, hidden]), &Tensor::from_vec(logits, vec![1, count + 1]), top_k).to_host();
        let e = worst(&got, want.data());
        assert!(e < 2e-5, "step {step}: {e}");
    }
    drop(moe);
    assert_eq!(b.used.load(Ordering::Relaxed), used, "the layer's bytes given back");
    // experts of a type the kernels do not decode run on the host
    let mut other = experts(6, hidden, ff, count, true);
    let q4 = |n: usize| (ggml_quants::GgmlType::Q4_0, vec![0u8; n / 32 * 18]);
    (other.gate, other.up, other.down) = (q4(count * ff * hidden), q4(count * ff * hidden), q4(count * hidden * ff));
    let hosted = b.quant_experts(other).unwrap();
    assert!(!DeviceChain::holds_experts(&b, hosted.as_ref()), "Q4_0 experts are the host's");
}

/// The experts' kernels' time (`--ignored --nocapture`): Flash-Next's layer (512 experts of 2560 by 640 in Q2_0,
/// 10 a row), a prompt's 512 rows in f32 and on the tensor cores, routed on the host and on the GPU, and a step's
/// one row.
#[test]
#[ignore = "a measurement"]
fn measure_q2_0_experts() {
    let Ok(b) = WgpuBackend::new(Some(3 << 30)) else { return };
    let (hidden, ff, count, top_k) = (2560usize, 640usize, 512usize, 10usize);
    let data = experts(77, hidden, ff, count, true);
    let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("room for the experts");
    f32s.coop = false;
    let cores = QuantMoe::try_new(&b, data, 0).ok().expect("room for the experts");
    for rows in [512usize, 1] {
        let (x, logits) = inputs(3, rows, hidden, count);
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
        let (xd, ld, out) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &ld, &logits);
        for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
            if what != "f32" && (!moe.coop || rows == 1) {
                continue;
            }
            for device in [false, true] {
                let reps = if rows == 1 { 200 } else { 8 };
                let run = || {
                    let mut rec = b.begin();
                    rec.keep_groups(rows == 1);
                    for _ in 0..reps {
                        if device {
                            if !rec.moe_routed(moe, &xd, &out, &ld, top_k, rows) {
                                return false;
                            }
                        } else {
                            rec.moe_rows(moe, &xd, &out, &assign);
                        }
                    }
                    rec.read_range(&out, 0, 1);
                    rec.finish();
                    true
                };
                if !run() {
                    eprintln!("{rows} rows, {what}, routed on the GPU: not this device's");
                    continue;
                }
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                let each = t.elapsed().as_secs_f64() / 3.0 / reps as f64;
                let flops = 2.0 * (rows * top_k) as f64 * (3 * hidden * ff) as f64;
                eprintln!("{rows} rows of {top_k} of {count} experts, {what}, routed on the {}: {:.3} ms a layer ({:.1} TFLOPS)", if device { "GPU" } else { "host" }, each * 1e3, flops / each / 1e12);
            }
        }
    }
}
