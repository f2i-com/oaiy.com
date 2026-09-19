//! Every kernel against its CPU oracle in `dsv41` (itself validated against
//! the reference model). Quantizers, RoPE and the hyper-connection mixes are
//! bit-exact by construction (same operations, --fmad=false); reductions
//! (GEMV, norms, attention) differ only in summation order and are held to
//! one or two bf16 steps.
//!
//! Needs a CUDA device; skips when none is reachable. DSV41_CUDA_DEVICE
//! picks the card (default 1, the better-linked one on the dev machine).

use dsv41::attention::{index_scores, pool, sparse_attend};
use dsv41::expert::{expert_forward_batch, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1, W2, W3};
use dsv41::formats::{fake_quant_fp4_inplace, fake_quant_fp8, to_bf16, Fp4Scale};
use dsv41::hc::{self, HcParams, HC, MIX};
use dsv41::linear::{Out, Weight};
use dsv41::ops::{rmsnorm, Rope};
use dsv41_cuda::gpu::Pos;
use dsv41_cuda::Gpu;

fn gpu() -> Option<Gpu> {
    let dev = std::env::var("DSV41_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    match Gpu::new(dev) {
        Ok(g) => Some(g),
        // a kernel that does not compile is a failure, not a missing device
        Err(e) if format!("{e}").contains("CompileError") => panic!("kernels do not compile: {e}"),
        Err(e) => {
            eprintln!("skipping: no CUDA device {dev} ({e})");
            None
        }
    }
}

/// Deterministic values spanning many magnitudes, rounded to bf16.
fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f32 / (1u64 << 53) as f32; // [0, 1)
            let mag = (spread * (u * 2.0 - 1.0)).exp2();
            let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
            to_bf16(sign * mag * (0.5 + u))
        })
        .collect()
}

fn bytes(n: usize, seed: u64, f: impl Fn(u8) -> u8) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f((s >> 24) as u8)
        })
        .collect()
}

/// Assert |a - b| <= `ulps` bf16 steps of max(|a|, |b|), elementwise, plus
/// an absolute floor of 1e-3 of the vector's largest value: a sum that
/// cancels to near zero can land several of its own (tiny) bf16 steps away
/// when the summation order changes, without any value being wrong.
fn close(what: &str, got: &[f32], want: &[f32], ulps: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let floor = want.iter().fold(0.0f32, |m, v| m.max(v.abs())) * 1e-3;
    let (mut worst, mut exact) = (0.0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let tol = ulps * a.abs().max(b.abs()) * 2f32.powi(-8) + floor + 1e-30;
        let d = (a - b).abs();
        worst = worst.max(d / tol);
        exact += usize::from(a.to_bits() == b.to_bits());
        assert!(d <= tol, "{what}[{i}]: gpu {a} vs cpu {b}");
    }
    eprintln!("{what}: {:.1}% bit-exact, worst {:.2} of the allowed {ulps} bf16 steps", 100.0 * exact as f64 / got.len() as f64, worst * ulps);
}

fn exact(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(a.to_bits() == b.to_bits(), "{what}[{i}]: gpu {a} vs cpu {b}");
    }
    eprintln!("{what}: bit-exact over {} values", got.len());
}

#[test]
fn quantizers_are_bit_exact() {
    let Some(g) = gpu() else { return };
    let x = values(64 * 1024, 1, 24.0);
    let mut d = g.upload(&x).unwrap();
    g.act_quant_fp8(&mut d.slice_mut(..)).unwrap();
    exact("act_quant_fp8", &g.download(&d).unwrap(), &fake_quant_fp8(&x, 32));
    let src = g.upload(&x).unwrap();
    exact("act_quant_fp8_to", &g.download(&g.act_quant_fp8_to(&src.as_view()).unwrap()).unwrap(), &fake_quant_fp8(&x, 32));

    for (block, kind) in [(32, Fp4Scale::E8M0), (16, Fp4Scale::E4M3)] {
        let mut want = x.clone();
        fake_quant_fp4_inplace(&mut want, block, kind);
        let mut d = g.upload(&x).unwrap();
        g.act_quant_fp4(&mut d.slice_mut(..), block, kind == Fp4Scale::E4M3).unwrap();
        exact(&format!("act_quant_fp4 {kind:?}/{block}"), &g.download(&d).unwrap(), &want);
    }
}

#[test]
fn gemv_matches_linear() {
    let Some(g) = gpu() else { return };
    let (n, k, t) = (96, 320, 3);
    let x = values(t * k, 2, 4.0);

    // fp8: never the NaN codes 0x7f / 0xff
    let w = bytes(n * k, 3, |b| if b & 0x7f == 0x7f { b ^ 1 } else { b });
    let s = bytes(n.div_ceil(32) * k.div_ceil(32), 4, |b| 118 + b % 12);
    let cpu = Weight::Fp8 { w: w.clone(), s: s.clone(), n, k }.forward(&x, t, Out::Bf16);
    let mut xq = g.upload(&x).unwrap();
    g.act_quant_fp8(&mut xq.slice_mut(..)).unwrap();
    let mut y = g.zeros::<f32>(t * n).unwrap();
    g.gemv_fp8(&xq.as_view(), &g.upload(&w).unwrap(), &g.upload(&s).unwrap(), &mut y.slice_mut(..), n, k, t, true).unwrap();
    close("gemv_fp8", &g.download(&y).unwrap(), &cpu, 2.0);
    // the shared expert's gate/up/SwiGLU in one launch = two gemv_fp8 + swiglu
    {
        let (ns, ks) = (64, 320);
        let (wa, wb) = (bytes(ns * ks, 40, |b| if b & 0x7f == 0x7f { b ^ 1 } else { b }), bytes(ns * ks, 41, |b| if b & 0x7f == 0x7f { b ^ 1 } else { b }));
        let (sa, sb) = (bytes(ns.div_ceil(32) * ks / 32, 42, |b| 118 + b % 12), bytes(ns.div_ceil(32) * ks / 32, 43, |b| 118 + b % 12));
        let (wad, wbd, sad, sbd) = (g.upload(&wa).unwrap(), g.upload(&wb).unwrap(), g.upload(&sa).unwrap(), g.upload(&sb).unwrap());
        let mut xq1 = g.upload(&x[..ks]).unwrap();
        g.act_quant_fp8(&mut xq1.slice_mut(..)).unwrap();
        let (mut gate, mut up) = (g.zeros::<f32>(ns).unwrap(), g.zeros::<f32>(ns).unwrap());
        g.gemv_fp8(&xq1.as_view(), &wad, &sad, &mut gate.slice_mut(..), ns, ks, 1, true).unwrap();
        g.gemv_fp8(&xq1.as_view(), &wbd, &sbd, &mut up.slice_mut(..), ns, ks, 1, true).unwrap();
        let mut want = g.zeros::<f32>(ns).unwrap();
        g.swiglu(&gate, &up, None, &mut want, ns, 1, 10.0).unwrap();
        let got = g.shared_gate_up(&xq1.as_view(), &wad, &sad, &wbd, &sbd, ns, ks, 10.0).unwrap().expect("shape fits");
        exact("shared_gate_up", &g.download(&got).unwrap(), &g.download(&want).unwrap());
    }

    // the fused one-token path: quantize in shared memory, then the same sums
    let (wd, sd) = (g.upload(&w).unwrap(), g.upload(&s).unwrap());
    let two_launch = g.download(&y).unwrap();
    for i in 0..t {
        let xi = g.upload(&x[i * k..(i + 1) * k]).unwrap();
        let yi = g.gemv_fp8_token(&xi.as_view(), &wd, &sd, n, k, true, true).unwrap().expect("shape fits");
        exact(&format!("gemv_fp8_token token {i}"), &g.download(&yi).unwrap(), &two_launch[i * n..(i + 1) * n]);
    }

    // bf16, both output kinds, and block-diagonal groups (wo_a)
    let wb: Vec<u16> = values(n * k, 5, 3.0).iter().map(|v| dsv41::formats::f32_to_bf16(*v)).collect();
    let wbd = g.upload(&wb).unwrap();
    let xd = g.upload(&x).unwrap();
    for out in [Out::Bf16, Out::F32] {
        let cpu = Weight::Bf16 { w: wb.clone(), n, k }.forward(&x, t, out);
        let mut y = g.zeros::<f32>(t * n).unwrap();
        g.gemv_bf16(&xd.as_view(), &wbd, &mut y.slice_mut(..), n, k, t, out == Out::Bf16, 0, 0).unwrap();
        close(&format!("gemv_bf16 {out:?}"), &g.download(&y).unwrap(), &cpu, if out == Out::Bf16 { 2.0 } else { 0.05 });
    }
    let groups = 4; // x rows are groups * k wide, W rows split into 4 groups of n/4
    let xg = values(t * groups * k, 6, 4.0);
    let mut cpu = vec![0.0f32; t * n];
    let wfull = Weight::Bf16 { w: wb.clone(), n, k };
    for grp in 0..groups {
        let xs: Vec<f32> = (0..t).flat_map(|i| xg[(i * groups + grp) * k..(i * groups + grp + 1) * k].iter().copied()).collect();
        let ys = wfull.forward_rows(&xs, t, grp * n / groups..(grp + 1) * n / groups, Out::Bf16);
        for i in 0..t {
            cpu[i * n + grp * n / groups..i * n + (grp + 1) * n / groups].copy_from_slice(&ys[i * n / groups..(i + 1) * n / groups]);
        }
    }
    let mut y = g.zeros::<f32>(t * n).unwrap();
    g.gemv_bf16(&g.upload(&xg).unwrap().as_view(), &wbd, &mut y.slice_mut(..), n, k, t, true, n / groups, groups * k).unwrap();
    close("gemv_bf16 grouped", &g.download(&y).unwrap(), &cpu, 2.0);

    // fp8 weights dequantized on the fly give the bf16 expansion's results, bit for bit
    let (n8, k8) = (64, 256);
    let w8 = bytes(n8 * k8, 30, |b| if b & 0x7f == 0x7f { b ^ 1 } else { b });
    let s8 = bytes(n8.div_ceil(32) * k8.div_ceil(32), 31, |b| 118 + b % 12);
    let expanded: Vec<u16> = (0..n8 * k8)
        .map(|i| {
            let (r, c) = (i / k8, i % k8);
            dsv41::formats::f32_to_bf16(dsv41::formats::fp8_e4m3_to_f32(w8[i]) * dsv41::formats::e8m0_to_f32(s8[(r / 32) * k8.div_ceil(32) + c / 32]))
        })
        .collect();
    let g8 = 4usize; // 4 groups of 16 rows, x rows 4 * k8 wide
    let x8 = values(t * g8 * k8, 32, 3.0);
    let xd8 = g.upload(&x8).unwrap();
    let mut want = g.zeros::<f32>(t * n8).unwrap();
    g.gemv_bf16(&xd8.as_view(), &g.upload(&expanded).unwrap(), &mut want.slice_mut(..), n8, k8, t, true, n8 / g8, g8 * k8).unwrap();
    let mut got = g.zeros::<f32>(t * n8).unwrap();
    g.gemv_fp8w(&xd8.as_view(), &g.upload(&w8).unwrap(), &g.upload(&s8).unwrap(), &mut got.slice_mut(..), n8, k8, t, true, n8 / g8, g8 * k8).unwrap();
    exact("gemv_fp8w grouped", &g.download(&got).unwrap(), &g.download(&want).unwrap());

    let wf = values(n * k, 7, 2.0);
    let cpu = Weight::F32 { w: wf.clone(), n, k }.forward(&x, t, Out::F32);
    let mut y = g.zeros::<f32>(t * n).unwrap();
    g.gemv_f32(&xd.as_view(), &g.upload(&wf).unwrap(), &mut y.slice_mut(..), n, k, t).unwrap();
    close("gemv_f32", &g.download(&y).unwrap(), &cpu, 0.05);
}

#[test]
fn expert_matches_cpu() {
    let Some(g) = gpu() else { return };
    let t = 3;
    // a synthetic record: random nibbles, scales around 2^-4
    let mut rec = bytes(RECORD_BYTES, 8, |b| b);
    for r in [S1, S2, S3] {
        for (i, b) in rec[r].iter_mut().enumerate() {
            *b = 121 + (i % 5) as u8;
        }
    }
    let x = values(t * DIM, 9, 3.0);
    let route = [0.4f32, 1.1, 0.25];
    let cpu = expert_forward_batch(&rec, &x, Some(&route), 10.0);

    let recd = g.upload(&rec).unwrap();
    let mut xq = g.upload(&x).unwrap();
    g.act_quant_fp8(&mut xq.slice_mut(..)).unwrap();
    let (mut gate, mut up) = (g.zeros::<f32>(t * INTER).unwrap(), g.zeros::<f32>(t * INTER).unwrap());
    g.gemv_fp4(&xq.as_view(), &recd.slice(W1), &recd.slice(S1), &mut gate.slice_mut(..), INTER, DIM, t, true).unwrap();
    g.gemv_fp4(&xq.as_view(), &recd.slice(W3), &recd.slice(S3), &mut up.slice_mut(..), INTER, DIM, t, true).unwrap();
    let mut h = g.zeros::<f32>(t * INTER).unwrap();
    g.swiglu(&gate, &up, Some(&g.upload(&route).unwrap()), &mut h, INTER, t, 10.0).unwrap();
    g.act_quant_fp8(&mut h.slice_mut(..)).unwrap();
    let mut y = g.zeros::<f32>(t * DIM).unwrap();
    g.gemv_fp4(&h.as_view(), &recd.slice(W2), &recd.slice(S2), &mut y.slice_mut(..), DIM, INTER, t, true).unwrap();
    // an fp8 step (2^-4) upstream can move an output by more than a bf16 step,
    // so this one is held on the whole vector
    let (got, want) = (g.download(&y).unwrap(), cpu);
    let num: f64 = got.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt();
    eprintln!("expert (fp4 gate/up, swiglu, fp8 requant, fp4 down): rel-L2 {rel:.2e}");
    assert!(rel < 5e-3, "expert rel-L2 {rel}");
}

/// Decode's grouped path: every routed expert of a token in one launch per
/// stage, reading records by address, then the expert-order sum plus the
/// shared output with one bf16 round (as `Moe::forward`).
#[test]
fn grouped_experts_match_cpu() {
    let Some(g) = gpu() else { return };
    let nexp = 3;
    let recs: Vec<Vec<u8>> = (0..nexp)
        .map(|e| {
            let mut rec = bytes(RECORD_BYTES, 20 + e as u64, |b| b);
            for r in [S1, S2, S3] {
                for (i, b) in rec[r].iter_mut().enumerate() {
                    *b = 120 + ((i + e) % 6) as u8;
                }
            }
            rec
        })
        .collect();
    let x = values(DIM, 30, 3.0);
    let route = [0.4f32, 1.1, 0.25];
    let shared = values(DIM, 31, 2.0);
    let mut want = vec![0.0f32; DIM];
    for (rec, &w) in recs.iter().zip(&route) {
        for (acc, v) in want.iter_mut().zip(expert_forward_batch(rec, &x, Some(&[w]), 10.0)) {
            *acc += v;
        }
    }
    for (acc, v) in want.iter_mut().zip(&shared) {
        *acc = to_bf16(*acc + v);
    }

    let dev: Vec<_> = recs.iter().map(|r| g.upload(r).unwrap()).collect();
    let ptrs: Vec<u64> = dev.iter().map(|d| g.addr(&d.as_view())).collect();
    // table rows in reverse: moe_down must honour them
    let rows: Vec<usize> = (0..nexp).rev().collect();
    let tab = g.upload(&dsv41_cuda::Gpu::moe_table(&ptrs, &rows, &route)).unwrap();
    let mut xq = g.upload(&x).unwrap();
    g.act_quant_fp8(&mut xq.slice_mut(..)).unwrap();
    let mut h = g.zeros::<f32>(nexp * INTER).unwrap();
    g.moe_gate_up(&xq.as_view(), &tab, &mut h, nexp, INTER, DIM, 10.0).unwrap();
    g.act_quant_fp8(&mut h.slice_mut(..)).unwrap();
    let mut outs = g.zeros::<f32>(nexp * DIM).unwrap();
    g.moe_down(&h, &tab, &mut outs, nexp, nexp, INTER, DIM).unwrap();
    // the reduction sums rows in order, so restore expert order first
    let mut host = g.download(&outs).unwrap();
    host = host.chunks_exact(DIM).rev().flatten().copied().collect();
    let outs = g.upload(&host).unwrap();
    let mut y = g.zeros::<f32>(DIM).unwrap();
    g.moe_reduce(&outs, &g.upload(&shared).unwrap(), &mut y, nexp, DIM).unwrap();

    let got = g.download(&y).unwrap();
    let num: f64 = got.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt();
    eprintln!("grouped experts (gate_up, requant, down, reduce + shared): rel-L2 {rel:.2e}");
    assert!(rel < 5e-3, "grouped experts rel-L2 {rel}");
}

#[test]
fn norms_rope_and_mixes() {
    let Some(g) = gpu() else { return };
    let (t, d) = (3, 5120);
    let x = values(t * d, 10, 6.0);
    let w = values(d, 11, 1.0);
    let mut y = g.zeros::<f32>(t * d).unwrap();
    g.rmsnorm(&g.upload(&x).unwrap().as_view(), &g.upload(&w).unwrap(), &mut y.slice_mut(..), t, d, 1e-20).unwrap();
    close("rmsnorm", &g.download(&y).unwrap(), &rmsnorm(&x, &w, 1e-20), 2.0);

    // hc_pre + rmsnorm fused: bit for bit the two launches
    {
        let hx = values(t * HC * d, 33, 3.0);
        let mixv = values(t * 24, 34, 1.0);
        let (hd_, md, wd) = (g.upload(&hx).unwrap(), g.upload(&mixv).unwrap(), g.upload(&w).unwrap());
        let mut xpre = g.zeros::<f32>(t * d).unwrap();
        g.hc_pre(&hd_, &md.as_view(), &mut xpre, d, t, 24).unwrap();
        let mut two = g.zeros::<f32>(t * d).unwrap();
        g.rmsnorm(&xpre.as_view(), &wd, &mut two.slice_mut(..), t, d, 1e-6).unwrap();
        let mut one = g.zeros::<f32>(t * d).unwrap();
        g.hc_pre_norm(&hd_, &md.as_view(), 24, &wd, &mut one, d, t, 1e-6).unwrap();
        exact("hc_pre_norm", &g.download(&one).unwrap(), &g.download(&two).unwrap());
    }
    // kv_finish = rmsnorm, rope, act_quant, copy into the slot
    {
        let hd = 512;
        let rope = Rope::new(64, 300, 65536, 160000.0, 16.0, 32.0, 1.0);
        let (cos, sin) = rope.tables();
        let (cd, sd) = (g.upload(cos).unwrap(), g.upload(sin).unwrap());
        let kv0 = values(hd, 35, 3.0);
        let wn = values(hd, 36, 1.0);
        let (kd, wd) = (g.upload(&kv0).unwrap(), g.upload(&wn).unwrap());
        let mut a = g.zeros::<f32>(hd).unwrap();
        g.rmsnorm(&kd.as_view(), &wd, &mut a.slice_mut(..), 1, hd, 1e-6).unwrap();
        g.rope(&mut a.slice_mut(..), &cd, &sd, Pos::Linear { base: 77, per: 1 }, 1, hd, hd - 64, 32, false).unwrap();
        g.act_quant_fp8(&mut a.slice_mut(..)).unwrap();
        let mut win = g.zeros::<f32>(3 * hd).unwrap();
        g.kv_finish(&kd.as_view(), &wd, &cd, &sd, 77, 32, &mut win.slice_mut(hd..2 * hd), hd, 1e-6).unwrap();
        exact("kv_finish", &g.download(&win).unwrap()[hd..2 * hd], &g.download(&a).unwrap());
    }

    // RoPE: identical tables and operations -> bit-exact, both directions
    let rope = Rope::new(64, 300, 65536, 160000.0, 16.0, 32.0, 1.0);
    let (cos, sin) = rope.tables();
    let (cd, sd) = (g.upload(cos).unwrap(), g.upload(sin).unwrap());
    let (rows, stride, off) = (5usize, 512usize, 448usize);
    let pos: Vec<i32> = vec![0, 7, 130, 255, 299];
    let base = values(rows * stride, 12, 3.0);
    for inverse in [false, true] {
        let mut want = base.clone();
        for (r, &p) in pos.iter().enumerate() {
            rope.apply(&mut want[r * stride + off..r * stride + off + 64], p as usize, inverse);
        }
        let mut xd = g.upload(&base).unwrap();
        g.rope(&mut xd.slice_mut(..), &cd, &sd, Pos::Rows(&g.upload(&pos).unwrap()), rows, stride, off, 32, inverse).unwrap();
        exact(&format!("rope inverse={inverse}"), &g.download(&xd).unwrap(), &want);
    }
    // linear positions (decode: heads of one token share base + row / per)
    let (base_pos, per) = (130usize, 2usize);
    let mut want = base.clone();
    for r in 0..rows {
        rope.apply(&mut want[r * stride + off..r * stride + off + 64], base_pos + r / per, false);
    }
    let mut xd = g.upload(&base).unwrap();
    g.rope(&mut xd.slice_mut(..), &cd, &sd, Pos::Linear { base: base_pos, per }, rows, stride, off, 32, false).unwrap();
    exact("rope linear positions", &g.download(&xd).unwrap(), &want);

    // hyper-connections: projections on the device, the tail on the host
    let n = HC * d;
    let p = HcParams::new(values(MIX * n, 13, 1.0), values(MIX, 14, 0.5), vec![0.6, 0.8, 1.2]);
    let hx = values(t * n, 15, 5.0);
    let mut out = g.zeros::<f32>(t * 25).unwrap();
    g.hc_project(&g.upload(&hx).unwrap(), &g.upload(p.projection()).unwrap(), &mut out, n, t).unwrap();
    let out = g.download(&out).unwrap();
    for i in 0..t {
        let proj: [f32; MIX] = std::array::from_fn(|j| out[i * 25 + j]);
        let got = hc::mixes_from_projection(&proj, out[i * 25 + 24], n, &p, 1e-20, 20, 1e-6);
        let want = hc::mixes(&hx[i * n..(i + 1) * n], &p, 1e-20, 20, 1e-6);
        let flat = |m: &hc::Mix| -> Vec<f32> { m.pre.iter().chain(&m.post).chain(m.comb.iter().flatten()).copied().collect() };
        close(&format!("hc mixes token {i}"), &flat(&got), &flat(&want), 0.05);
    }

    // hc_mix: the host tail of the mixes, on the device, from the same projections
    let (base, scale) = p.base_and_scale();
    let mut mixd = g.zeros::<f32>(t * MIX).unwrap();
    let projd = g.upload(&out).unwrap();
    g.hc_mix(&projd, &g.upload(base).unwrap(), &g.upload(scale).unwrap(), &mut mixd, t, n, 1e-20, 20, 1e-6).unwrap();
    let got_mix = g.download(&mixd).unwrap();
    let flat = |m: &hc::Mix| -> Vec<f32> { m.pre.iter().chain(&m.post).chain(m.comb.iter().flatten()).copied().collect() };
    let want_mix: Vec<f32> = (0..t)
        .flat_map(|i| {
            let proj: [f32; MIX] = std::array::from_fn(|j| out[i * 25 + j]);
            flat(&hc::mixes_from_projection(&proj, out[i * 25 + 24], n, &p, 1e-20, 20, 1e-6))
        })
        .collect();
    close("hc_mix", &got_mix, &want_mix, 0.05);

    // hc_project_mix: both in one launch, twice (the counters must reset)
    {
        let (hxd, pd) = (g.upload(&hx).unwrap(), g.upload(p.projection()).unwrap());
        let (bd, sd) = (g.upload(base).unwrap(), g.upload(scale).unwrap());
        let mut counter = g.zeros::<u32>(t).unwrap();
        for round in 0..2 {
            let (mut projf, mut mixf) = (g.zeros::<f32>(t * 25).unwrap(), g.zeros::<f32>(t * MIX).unwrap());
            g.hc_project_mix(&hxd, &pd, &mut projf, &bd, &sd, &mut mixf, &mut counter, t, n, 1e-20, 20, 1e-6).unwrap();
            exact(&format!("hc_project_mix round {round}"), &g.download(&mixf).unwrap(), &got_mix);
        }
        assert!(g.download(&counter).unwrap().iter().all(|&c| c == 0), "counters reset");
    }

    // hc_pre / hc_post read the device mix layout (pre, post, comb per token);
    // same sequential sums as the CPU -> bit-exact
    let mixes: Vec<hc::Mix> = (0..t)
        .map(|i| {
            let m = &got_mix[i * MIX..(i + 1) * MIX];
            hc::Mix {
                pre: std::array::from_fn(|j| m[j]),
                post: std::array::from_fn(|j| m[HC + j]),
                comb: std::array::from_fn(|a| std::array::from_fn(|b| m[2 * HC + a * HC + b])),
            }
        })
        .collect();
    let mut y = g.zeros::<f32>(t * d).unwrap();
    g.hc_pre(&g.upload(&hx).unwrap(), &mixd.as_view(), &mut y, d, t, MIX).unwrap();
    let want: Vec<f32> = (0..t).flat_map(|i| hc::pre(&hx[i * n..(i + 1) * n], &mixes[i].pre)).collect();
    exact("hc_pre", &g.download(&y).unwrap(), &want);
    let o = values(t * d, 16, 3.0);
    let mut y = g.zeros::<f32>(t * n).unwrap();
    g.hc_post(&g.upload(&o).unwrap(), &g.upload(&hx).unwrap(), &mixd, &mut y, d, t).unwrap();
    let want: Vec<f32> = (0..t).flat_map(|i| hc::post(&o[i * d..(i + 1) * d], &hx[i * n..(i + 1) * n], &mixes[i])).collect();
    exact("hc_post", &g.download(&y).unwrap(), &want);
}

#[test]
fn attention_pieces() {
    let Some(g) = gpu() else { return };
    let (t, nh, hd, nkv) = (2, 8, 512, 40);
    let q = values(t * nh * hd, 17, 2.0);
    let kv = values(nkv * hd, 18, 2.0);
    let sink = values(nh, 19, 1.0);
    let nidx = 24;
    let idx: Vec<i32> = (0..t * nidx).map(|i| if i % 7 == 3 { -1 } else { ((i * 13) % nkv) as i32 }).collect();
    let scale = (hd as f32).powf(-0.5);
    let mut want = vec![0.0f32; t * nh * hd];
    for i in 0..t {
        for h in 0..nh {
            let o = &mut want[(i * nh + h) * hd..(i * nh + h + 1) * hd];
            sparse_attend(&q[(i * nh + h) * hd..(i * nh + h + 1) * hd], &kv, hd, &idx[i * nidx..(i + 1) * nidx], sink[h], scale, o);
        }
    }
    let mut out = g.zeros::<f32>(t * nh * hd).unwrap();
    g.sparse_attn(&g.upload(&q).unwrap(), &g.upload(&kv).unwrap().as_view(), nkv, &g.upload(&kv).unwrap().as_view(), &g.upload(&idx).unwrap(), &g.upload(&sink).unwrap(), &mut out, t, nh, hd, nidx, scale, None)
        .unwrap();
    close("sparse_attn", &g.download(&out).unwrap(), &want, 3.0);
    // the same rows split over two buffers (window part, compressed part)
    let split = 17;
    let (ka, kb) = (g.upload(&kv[..split * hd]).unwrap(), g.upload(&kv[split * hd..]).unwrap());
    let mut out2 = g.zeros::<f32>(t * nh * hd).unwrap();
    g.sparse_attn(&g.upload(&q).unwrap(), &ka.as_view(), split, &kb.as_view(), &g.upload(&idx).unwrap(), &g.upload(&sink).unwrap(), &mut out2, t, nh, hd, nidx, scale, None)
        .unwrap();
    exact("sparse_attn over two buffers", &g.download(&out2).unwrap(), &g.download(&out).unwrap());
    // with the inverse rope fused in: = sparse_attn, then rope(inverse)
    {
        let rope = Rope::new(64, 300, 65536, 160000.0, 16.0, 32.0, 1.0);
        let (cos, sin) = rope.tables();
        let (cd, sd) = (g.upload(cos).unwrap(), g.upload(sin).unwrap());
        let pos0 = 123;
        let mut want = g.download(&out).unwrap();
        for r in 0..t * nh {
            rope.apply(&mut want[r * hd + hd - 64..(r + 1) * hd], pos0 + r / nh, true);
        }
        let mut out3 = g.zeros::<f32>(t * nh * hd).unwrap();
        g.sparse_attn(&g.upload(&q).unwrap(), &ka.as_view(), split, &kb.as_view(), &g.upload(&idx).unwrap(), &g.upload(&sink).unwrap(), &mut out3, t, nh, hd, nidx, scale, Some((&cd, &sd, 32, pos0)))
            .unwrap();
        exact("sparse_attn with fused inverse rope", &g.download(&out3).unwrap(), &want);
    }

    // indexer scores (bf16 at every step: allow a couple of steps)
    let (inh, ihd, nkeys) = (4, 128, 30);
    let iq = values(t * inh * ihd, 20, 1.0);
    let keys = values(nkeys * ihd, 21, 1.0);
    let w = values(t * inh, 22, 1.0);
    let want = index_scores(&iq, &keys, &w, inh, ihd);
    let kd = g.upload(&keys).unwrap();
    let mut s = g.zeros::<f32>(t * nkeys).unwrap();
    g.index_scores(&g.upload(&iq).unwrap(), &kd.as_view(), &g.upload(&w).unwrap(), &mut s, inh, ihd, nkeys, t, 1.0).unwrap();
    let got = g.download(&s).unwrap();
    let tol: f32 = want.iter().map(|v| v.abs()).fold(0.0, f32::max) * 2f32.powi(-6);
    let (mut worst, mut exact) = (0.0f32, 0);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() <= tol, "index_scores: gpu {a} vs cpu {b}");
        worst = worst.max((a - b).abs());
        exact += usize::from(a.to_bits() == b.to_bits());
    }
    eprintln!("index_scores: {exact} of {} bit-exact, worst {worst:.2e} (allowed {tol:.2e})", got.len());

    // compressor pooling (ratio 2), rounded to bf16 like the model does
    let (groups, r, phd) = (3, 2, 512);
    let pkv = values(groups * r * phd, 23, 2.0);
    let psc = values(groups * r * phd, 24, 2.0);
    let want: Vec<f32> = (0..groups).flat_map(|gr| pool(&pkv[gr * r * phd..(gr + 1) * r * phd], &psc[gr * r * phd..(gr + 1) * r * phd], r, phd)).map(to_bf16).collect();
    let mut o = g.zeros::<f32>(groups * phd).unwrap();
    g.compress_pool(&g.upload(&pkv).unwrap().as_view(), &g.upload(&psc).unwrap().as_view(), &mut o, groups, r, phd).unwrap();
    close("compress_pool", &g.download(&o).unwrap(), &want, 1.0);

    // engram gate
    let d = 5120;
    let h = values(t * HC * d, 25, 4.0);
    let ekv = values(t * (HC + 1) * d, 26, 2.0);
    let qk = values(HC * d, 27, 0.5);
    let want = dsv41::engram::gate(&h, &ekv, &qk, d, 1e-20);
    let mut o = g.zeros::<f32>(t * HC * d).unwrap();
    g.engram_gate(&g.upload(&h).unwrap(), &g.upload(&ekv).unwrap(), &g.upload(&qk).unwrap(), &mut o, d, t, 1e-20, (d as f32).powf(-0.5)).unwrap();
    close("engram_gate", &g.download(&o).unwrap(), &want, 1.0);
}

/// Launch-ahead: the reduction is queued before the CPU rows exist and must
/// wait on the device for them, then sum exactly as moe_reduce does.
#[test]
fn moe_reduce_waits_for_the_cpu_handoff() {
    let Some(g) = gpu() else { return };
    let nexp = 5;
    let outs = values(nexp * DIM, 40, 2.0);
    let shared = values(DIM, 41, 2.0);
    let host: Vec<Vec<f32>> = (0..nexp).map(|e| values(DIM, 50 + e as u64, 2.0)).collect();
    let mask = 0b01010u32; // rows 1 and 3 come from the CPU
    let mut merged = outs.clone();
    for r in [1, 3] {
        merged[r * DIM..(r + 1) * DIM].copy_from_slice(&host[r]);
    }
    let mut want = g.zeros::<f32>(DIM).unwrap();
    g.moe_reduce(&g.upload(&merged).unwrap(), &g.upload(&shared).unwrap(), &mut want, nexp, DIM).unwrap();
    let want = g.download(&want).unwrap();

    let handoff = std::sync::Arc::new(dsv41_cuda::handoff::Handoff::new(&g, 6, DIM).unwrap());
    let (od, sd) = (g.upload(&outs).unwrap(), g.upload(&shared).unwrap());
    for seq in 1..=2u32 {
        let mut y = g.zeros::<f32>(DIM).unwrap();
        g.moe_reduce_host(&od, &sd, &mut y, nexp, DIM, &handoff, mask, seq).unwrap();
        let (h, rows) = (std::sync::Arc::clone(&handoff), host.clone());
        let writer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            for r in [1, 3] {
                h.write_row(r, &rows[r]);
            }
            h.release(seq);
        });
        exact(&format!("moe_reduce_host (seq {seq})"), &g.download(&y).unwrap(), &want);
        writer.join().unwrap();
    }
}

/// The device publishes into host-mapped memory; the host spins on the flag.
#[test]
fn publish_reaches_the_host() {
    let Some(g) = gpu() else { return };
    let inbox = dsv41_cuda::handoff::Inbox::new(&g, 384 + DIM).unwrap();
    for seq in 1..=3u32 {
        let a = values(384, 60 + seq as u64, 2.0);
        let b = values(DIM, 70 + seq as u64, 2.0);
        let (ad, bd) = (g.upload(&a).unwrap(), g.upload(&b).unwrap());
        g.publish(&ad.as_view(), &bd.as_view(), &inbox, seq).unwrap();
        inbox.wait(&g, seq).unwrap();
        let (mut ga, mut gb) = (vec![0.0f32; 384], vec![0.0f32; DIM]);
        inbox.read(0, &mut ga);
        inbox.read(384, &mut gb);
        exact(&format!("publish a (seq {seq})"), &ga, &a);
        exact(&format!("publish b (seq {seq})"), &gb, &b);
    }
}
