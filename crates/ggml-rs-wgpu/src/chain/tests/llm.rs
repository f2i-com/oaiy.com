//! A language model's ops against the CPU's: a chained step, a delta net, the small ops, the hyper-connections and the n-gram layer.
use super::*;

/// A chain gives the CPU backend's answer: RMSNorm, a quantized matmul, the fused SwiGLU and an add, read back.
#[test]
fn a_chain_matches_the_cpus_ops() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    let (k, ff) = (256usize, 64usize);
    let mut next = rng(0x1234_5679);
    // a Q8_0 weight [2ff, k]: blocks of a scale and 32 int8s
    let mut bytes = vec![0u8; 2 * ff * (k / 32) * 34];
    for blk in bytes.chunks_mut(34) {
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        for v in blk[2..].iter_mut() {
            *v = ((next() + 1.0) * 127.0) as u8;
        }
    }
    let wq = ggml_rs::QuantizedTensor::from_bytes_cpu(bytes, vec![2 * ff, k], GgmlType::Q8_0);
    let wq = ggml_rs::Backend::to_device_quant(&b, wq);
    assert!(DeviceChain::holds(&b, &wq));
    let xv: Vec<f32> = (0..k).map(|_| next()).collect();
    let nw: Vec<f32> = (0..k).map(|_| next() * 0.5 + 1.0).collect();
    let res: Vec<f32> = (0..ff).map(|_| next()).collect();
    // CPU: rmsnorm, matmul, silu-mul split, add
    let xn = ggml_rs::Backend::rmsnorm(&cpu, &ggml_rs::Tensor::from_vec(xv.clone(), vec![1, k]), &ggml_rs::Tensor::from_vec(nw.clone(), vec![k]), 1e-5);
    let gu = ggml_rs::Backend::linear_q(&cpu, &xn, &ggml_rs::QuantizedTensor::from_bytes_cpu(wq.to_host().bytes().to_vec(), vec![2 * ff, k], GgmlType::Q8_0));
    let act = ggml_rs::Backend::silu_mul_split(&cpu, &gu, ff);
    let want: Vec<f32> = act.data().iter().zip(&res).map(|(a, r)| a + r).collect();
    // the chain
    let (x, n, xnd, gud, actd, acc) = (b.vec(k), b.vec(k), b.vec(k), b.vec(2 * ff), b.vec(ff), b.vec(ff));
    DeviceChain::upload(&b, &x, &xv);
    DeviceChain::upload(&b, &n, &nw);
    DeviceChain::upload(&b, &acc, &res);
    let mut rec = b.begin();
    rec.rmsnorm(&x, &n, &xnd, 1e-5);
    rec.matmul(&wq, &xnd, &gud);
    rec.silu_mul_split(&gud, &actd);
    rec.add(&acc, &actd);
    rec.read(&acc);
    rec.read(&xnd);
    let got = rec.finish();
    assert_eq!(got.len(), 2);
    close(&got[1], xn.data(), "rmsnorm");
    close(&got[0], &want, "the chain");
}

/// A gated delta net's conv and recurrence give the host's answer (`Backend::delta_net_step`, the CPU's): the
/// output, the conv state and the recurrent state after a decode step and after a run of tokens, from states the
/// host made, at a small shape and at Qwen3.8 27B's (48 value heads on 16 key heads of 128); runs of a few tokens
/// (one kernel) and of a prompt's (three passes).
#[test]
fn a_delta_net_matches_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    for (nv, nk, dim, kern, rows, sigmoid) in [(8usize, 4usize, 32usize, 4usize, 1usize, false), (8, 4, 32, 4, 7, true), (48, 16, 128, 4, 1, false), (48, 16, 128, 4, 5, true), (8, 4, 32, 4, 37, false), (48, 16, 128, 4, 100, true), (32, 16, 128, 4, 64, false)] {
        let ch = 2 * nk * dim + nv * dim;
        let mut next = rng((nv * 31 + rows) as u32);
        let mut vals = |n: usize, s: f32| (0..n).map(|_| next() * s).collect::<Vec<f32>>();
        let qkv = vals(rows * ch, 1.0);
        let z = vals(rows * nv * dim, 1.0);
        let ba = vals(rows * 2 * nv, 2.0);
        let cw = vals(ch * kern, 0.5);
        let a = vals(nv, 1.0).iter().map(|v| -v.abs() - 0.1).collect::<Vec<f32>>();
        let dt = vals(nv, 1.0);
        let nm = vals(dim, 1.0).iter().map(|v| v + 1.0).collect::<Vec<f32>>();
        let conv0 = vals((kern - 1) * ch, 1.0);
        let state0 = vals(nv * dim * dim, 0.1);
        let (scale, eps) = (1.0 / (dim as f32).sqrt(), 1e-6f32);
        let t = |d: &[f32], shape: Vec<usize>| ggml_rs::Tensor::from_vec(d.to_vec(), shape);
        let mut conv_c = t(&conv0, vec![kern - 1, ch]);
        let mut state_c = t(&state0, vec![nv, dim, dim]);
        // the host's step, or the backend's (its state kept on the GPU between calls)
        let step = |be: &dyn ggml_rs::Backend, conv: &mut ggml_rs::Tensor, state: &mut ggml_rs::Tensor| {
            let (q, zz, bb, ww) = (t(&qkv, vec![rows, ch]), t(&z, vec![rows, nv * dim]), t(&ba, vec![rows, 2 * nv]), t(&cw, vec![ch, kern]));
            let (aa, dd, nn) = (t(&a, vec![nv]), t(&dt, vec![nv]), t(&nm, vec![dim]));
            if sigmoid {
                be.delta_net_step_sigmoid(&q, &zz, &bb, &ww, &aa, &dd, &nn, conv, state, rows, nv, nk, dim, dim, nv / nk, scale, eps)
            } else {
                be.delta_net_step(&q, &zz, &bb, &ww, &aa, &dd, &nn, conv, state, rows, nv, nk, dim, dim, nv / nk, scale, eps)
            }
        };
        let want = step(&cpu, &mut conv_c, &mut state_c);
        // twice each way: the second from the first's state
        let (mut conv_g, mut state_g) = (t(&conv0, vec![kern - 1, ch]), t(&state0, vec![nv, dim, dim]));
        let (mut conv_h, mut state_h) = (t(&conv0, vec![kern - 1, ch]), t(&state0, vec![nv, dim, dim]));
        for _ in 0..2 {
            let got = step(&b, &mut conv_g, &mut state_g);
            let host = step(&cpu, &mut conv_h, &mut state_h);
            close(got.data(), host.data(), &format!("{nv} heads, {rows} rows: the backend's step"));
            assert!(state_g.is_device() && conv_g.is_device(), "the state stays on the GPU");
        }
        close(state_g.to_host().data(), state_h.data(), &format!("{nv} heads, {rows} rows: the backend's state"));
        close(conv_g.to_host().data(), conv_h.data(), &format!("{nv} heads, {rows} rows: the backend's conv"));
        let up = |d: &[f32]| {
            let v = b.vec(d.len());
            DeviceChain::upload(&b, &v, d);
            v
        };
        let (qkv_d, z_d, ba_d, cw_d, a_d, dt_d, nm_d, conv_d) = (up(&qkv), up(&z), up(&ba), up(&cw), up(&a), up(&dt), up(&nm), up(&conv0));
        let state_d = up(&state0);
        let (conv_out, out) = (b.vec(rows * ch), b.vec(rows * nv * dim));
        let mut rec = b.begin();
        rec.ssm_conv(&qkv_d, &cw_d, &conv_d, &conv_out, rows, ch, kern);
        rec.delta_net(&conv_out, &z_d, &ba_d, &a_d, &dt_d, &nm_d, &state_d, &out, DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: scale, eps, sigmoid_gate: sigmoid });
        rec.read(&out);
        rec.read(&conv_d);
        let got = rec.finish();
        let what = format!("{nv} heads of {dim}, {rows} rows");
        close(&got[0], want.data(), &format!("{what}: the output"));
        close(&got[1], conv_c.data(), &format!("{what}: the conv state"));
        // the recurrent state, through the alias the host would read it by
        let state = b.alias(&state_d, vec![nv, dim, dim]);
        close(state.to_host().data(), state_c.data(), &format!("{what}: the state"));
        assert!(b.aliased(&state).is_some_and(|v| Arc::ptr_eq(&v.inner, &state_d.inner)));
    }
}

/// An alias reads its vector as it is when read, and a clone of it is a copy that does not follow the vector.
#[test]
fn an_alias_reads_its_vector_and_a_clone_is_a_copy() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let v = b.vec(8);
    DeviceChain::upload(&b, &v, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let t = b.alias(&v, vec![2, 2, 2]);
    assert_eq!(t.to_host().data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let copy = t.clone();
    DeviceChain::upload(&b, &v, &[0.0; 8]);
    assert_eq!(t.to_host().data(), &[0.0; 8]);
    assert_eq!(copy.to_host().data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert!(b.aliased(&copy).is_some_and(|c| !Arc::ptr_eq(&c.inner, &v.inner)));
    assert!(b.aliased(&ggml_rs::Tensor::zeros(vec![8])).is_none());
    b.zero(&b.aliased(&copy).unwrap());
    assert_eq!(copy.to_host().data(), &[0.0; 8]);
}

/// Qwen3.5's partial RoPE, its q/gate split, gate and SwiGLU of two weights, and an f32 matmul give the CPU's
/// answers.
#[test]
fn qwen35s_small_ops_match_the_cpus() {
    use ggml_rs::Backend;
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    let (rows, heads, hd, rot, theta) = (3usize, 4usize, 32usize, 8usize, 1e7f32);
    let mut next = rng(77);
    let x: Vec<f32> = (0..rows * heads * 2 * hd).map(|_| next()).collect();
    let up = |d: &[f32]| {
        let v = b.vec(d.len());
        DeviceChain::upload(&b, &v, d);
        v
    };
    let xd = up(&x);
    let (q, g) = (b.vec(rows * heads * hd), b.vec(rows * heads * hd));
    let past = 17usize;
    let table: Vec<f32> = (past..past + rows)
        .flat_map(|pos| (0..rot / 2).flat_map(move |k| {
            let (s, c) = (pos as f32 * theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
            [s, c]
        }))
        .collect();
    let td = up(&table);
    let gated = b.vec(rows * heads * hd);
    let (gate_v, up_v, act) = (up(&x[..64]), up(&x[64..128]), b.vec(64));
    let (n, k) = (5usize, 48usize);
    let wf: Vec<f32> = (0..n * k).map(|_| next()).collect();
    let (wd, xf, yf) = (up(&wf), up(&x[..2 * k]), b.vec(2 * n));
    let mut rec = b.begin();
    rec.copy_cols(&xd, &q, rows * heads, hd, 2 * hd, 0);
    rec.copy_cols(&xd, &g, rows * heads, hd, 2 * hd, hd);
    rec.rope_partial_rows(&q, rows, heads, hd, rot, &td);
    rec.mul_sigmoid(&q, &g, &gated, rows * heads * hd);
    rec.silu_mul(&gate_v, &up_v, &act, 64);
    rec.matmul_f32_rows(&wd, n, k, &xf, &yf, 2);
    for v in [&q, &g, &gated, &act, &yf] {
        rec.read(v);
    }
    let got = rec.finish();
    let (mut qh, mut gh) = (Vec::new(), Vec::new());
    for r in x.chunks_exact(2 * hd) {
        qh.extend_from_slice(&r[..hd]);
        gh.extend_from_slice(&r[hd..]);
    }
    let mut qt = ggml_rs::Tensor::from_vec(qh, vec![rows, heads, hd]);
    let positions: Vec<u32> = (past..past + rows).map(|p| p as u32).collect();
    cpu.rope_partial_neox(&mut qt, &positions, hd, rot, theta);
    close(&got[0], qt.data(), "the partially rotated query");
    close(&got[1], &gh, "the gate half");
    let mut gt = qt.clone();
    cpu.mul_sigmoid_inplace(&mut gt, &ggml_rs::Tensor::from_vec(gh, vec![rows, heads, hd]));
    close(&got[2], gt.data(), "the gated output");
    let sw = cpu.silu_mul(&ggml_rs::Tensor::from_vec(x[..64].to_vec(), vec![64]), &ggml_rs::Tensor::from_vec(x[64..128].to_vec(), vec![64]));
    close(&got[3], sw.data(), "the SwiGLU");
    let mut want = Vec::new();
    for r in 0..2 {
        for o in 0..n {
            want.push((0..k).map(|i| wf[o * k + i] * x[r * k + i]).sum::<f32>());
        }
    }
    close(&got[4], &want, "the f32 matmul");
}

/// Flash-Next's hyper-connection ops give the host's: the per-stream norm, the gates, the mix, the write-back, and
/// a weighted term read from the device.
#[test]
fn hyper_connection_ops_match_the_hosts() {
    use ggml_rs::Backend;
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    let (rows, streams, d, rank, writes) = (3usize, 4usize, 64usize, 8usize, 4usize);
    let mut next = rng(5);
    let mut vals = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();
    let x = vals(rows * streams * d);
    let w = vals(streams * d);
    let t0 = vals(rows * (rank + writes));
    let logits = vals(rows * streams * d);
    let y = vals(rows * d);
    let post0 = vals(rows * streams);
    let up = |v: &[f32]| {
        let dv = b.vec(v.len());
        DeviceChain::upload(&b, &dv, v);
        dv
    };
    let t = |v: &[f32], shape: Vec<usize>| ggml_rs::Tensor::from_vec(v.to_vec(), shape);
    let (xd, wd, td, ld, yd, pd) = (up(&x), up(&w), up(&t0), up(&logits), up(&y), up(&post0));
    let (normed, post, mixed) = (b.vec(rows * streams * d), b.vec(rows * writes), b.vec(rows * d));
    let acc = up(&y);
    let weights = up(&[0.5, -1.25, 2.0]);
    let mut rec = b.begin();
    rec.rmsnorm_streams(&xd, &wd, &normed, rows, streams, 1e-6);
    rec.hc_gates(&td, &post, rows, rank, writes, streams);
    rec.hc_mix(&ld, &normed, &mixed, rows, streams, d);
    rec.stream_apply(&xd, &yd, &pd, rows, streams, d);
    rec.axpy_at(&acc, &yd, &weights, 1, rows * d);
    for v in [&normed, &td, &post, &mixed, &xd, &acc] {
        rec.read(v);
    }
    let got = rec.finish();
    let want_normed = cpu.hc_norm(&t(&x, vec![rows, streams * d]), &t(&w, vec![streams * d]), streams, 1e-6);
    close(&got[0], want_normed.data(), "the per-stream norm");
    let mut tt = t(&t0, vec![rows, rank + writes]);
    let want_post = cpu.hc_gates(&mut tt, rank, writes, streams);
    close(&got[1], tt.data(), "the gates' input");
    close(&got[2], want_post.data(), "the write weights");
    let want_mix = cpu.hc_mix(&t(&logits, vec![rows, streams * d]), &want_normed, streams);
    close(&got[3], want_mix.data(), "the mix");
    let mut xs = t(&x, vec![rows, streams * d]);
    cpu.stream_apply(&mut xs, &t(&y, vec![rows, d]), &t(&post0, vec![rows, streams]), streams);
    close(&got[4], xs.data(), "the write-back");
    let want_acc: Vec<f32> = y.iter().map(|v| v + -1.25 * v).collect();
    close(&got[5], &want_acc, "the weighted term");
}
/// A copy past a grid dimension's 65,535 workgroups (17 million values, from one offset to another), every value
/// where it belongs.
#[test]
fn a_long_copy_lands_every_value() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let len = 17_000_000usize;
    let src: Vec<f32> = (0..len + 3).map(|i| i as f32).collect();
    let (sd, dd) = (b.vec(src.len()), b.vec(len + 5));
    DeviceChain::upload(&b, &sd, &src);
    let mut rec = b.begin();
    rec.copy(&sd, 3, &dd, 5, len);
    rec.read(&dd);
    let got = rec.finish().pop().unwrap();
    for i in [0usize, 1, 16_776_959, 16_776_960, 16_776_961, len - 1] {
        assert_eq!(got[5 + i], (3 + i) as f32, "value {i}");
    }
    assert_eq!(&got[..5], &[0.; 5]);
}

/// A draft's token from logits on the device is the host's: the first of equal largest, and the sum of the
/// exponentials against it within rounding, over a vocabulary's 248,320 and a few.
#[test]
fn a_drafts_token_is_the_first_largest_logit_and_its_share() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(17);
    for (n, peak, tie) in [(248_320usize, 151_000usize, Some(200_000usize)), (5, 3, None), (300, 7, Some(6))] {
        let mut x: Vec<f32> = (0..n).map(|_| r() * 8.0).collect();
        x[peak] = 30.0;
        if let Some(t) = tie {
            x[t] = 30.0;
        }
        let (xd, out) = (b.vec(n), b.vec(3));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.argmax_softmax(&xd, &out);
        rec.read(&out);
        let got = rec.finish().pop().unwrap();
        let first = tie.map_or(peak, |t| t.min(peak));
        let total: f64 = x.iter().map(|&v| ((v - 30.0) as f64).exp()).sum();
        assert_eq!(got[0].to_bits() as usize, first, "{n}: the first largest");
        assert_eq!(got[1], 30.0);
        assert!(((got[2] as f64) - total).abs() <= 1e-5 * total, "{n}: {} against {total}", got[2]);
    }
}

/// Several rows' tokens are each row's first largest logit and its share: rows of a vocabulary's 248,320 and of a
/// few, a row's peak in its first and its last workgroup of 4,096, equals within a row.
#[test]
fn rows_tokens_are_each_rows_first_largest_logit() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(23);
    for (n, rows) in [(248_320usize, 4usize), (248_320, 8), (300, 3), (5, 2), (4096, 5)] {
        let mut x: Vec<f32> = (0..rows * n).map(|_| r() * 8.0).collect();
        let mut firsts = Vec::new();
        for row in 0..rows {
            // (a peak somewhere else in every row, and its equal after it in every other)
            let peak = (row * 7919 + 3) % n;
            x[row * n + peak] = 30.0;
            if row % 2 == 1 && peak + 1 < n {
                x[row * n + n - 1] = 30.0;
            }
            firsts.push(peak);
        }
        let (xd, out) = (b.vec(rows * n), b.vec(4 * rows));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.argmax_rows(&xd, rows, n, &out);
        rec.read(&out);
        let got = rec.finish().pop().unwrap();
        for row in 0..rows {
            let total: f64 = x[row * n..(row + 1) * n].iter().map(|&v| ((v - 30.0) as f64).exp()).sum();
            assert_eq!(got[4 * row].to_bits() as usize, firsts[row], "{rows} rows of {n}: row {row}'s first largest");
            assert_eq!(got[4 * row + 1], 30.0);
            assert!(((got[4 * row + 2] as f64) - total).abs() <= 1e-5 * total, "{rows} rows of {n}: row {row}'s {} against {total}", got[4 * row + 2]);
        }
    }
}

/// An n-gram layer's gate and conv chained give the CPU's: two rows (a prompt's) then one (a step's), its window
/// carried from the first to the second.
#[test]
fn an_ngram_layers_gate_and_conv_match_the_cpus() {
    use ggml_rs::Backend;
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let cpu = ggml_rs::CpuBackend::new();
    let mut r = rng(23);
    let (streams, d, kernel, dil) = (4usize, 320usize, 4usize, 3usize);
    let width = streams * d;
    let state = (kernel - 1) * dil;
    let mut v = |n: usize| (0..n).map(|_| r()).collect::<Vec<f32>>();
    let (nk, nq, nc, wc) = (v(width), v(width), v(width), v(width * kernel));
    let (dnk, dnq, dnc, dwc) = (b.vec(width), b.vec(width), b.vec(width), b.vec(width * kernel));
    for (dv, h) in [(&dnk, &nk), (&dnq, &nq), (&dnc, &nc), (&dwc, &wc)] {
        DeviceChain::upload(&b, dv, h);
    }
    let mut window = Tensor::from_vec(v(state * width), vec![state, width]);
    let dwin = b.vec(state * width);
    DeviceChain::upload(&b, &dwin, window.data());
    for rows in [2usize, 1] {
        let (key, x, value) = (v(rows * width), v(rows * width), v(rows * d));
        let t = |h: &[f32], shape: Vec<usize>| Tensor::from_vec(h.to_vec(), shape);
        let (gated, conv_in) = cpu.ple_gate(&t(&key, vec![rows, width]), &t(&x, vec![rows, width]), &t(&value, vec![rows, d]), &t(&nk, vec![width]), &t(&nq, vec![width]), &t(&nc, vec![width]), streams, 1e-6);
        let mut want = t(&x, vec![rows, width]);
        cpu.ple_conv(&mut want, &gated, &conv_in, &mut window, &t(&wc, vec![width, kernel]), kernel, dil);
        let (dk, dx, dval, dg, dci) = (b.vec(rows * width), b.vec(rows * width), b.vec(rows * d), b.vec(rows * width), b.vec(rows * width));
        DeviceChain::upload(&b, &dk, &key);
        DeviceChain::upload(&b, &dx, &x);
        DeviceChain::upload(&b, &dval, &value);
        let mut rec = b.begin();
        rec.ple_gate(&dk, &dx, &dval, &dnk, &dnq, &dnc, &dg, &dci, rows, streams, d, 1e-6);
        rec.read(&dg);
        rec.read(&dci);
        rec.ple_conv(&dx, &dg, &dci, &dwin, &dwc, rows, width, kernel, dil);
        rec.read(&dx);
        rec.read(&dwin);
        let mut got = rec.finish();
        let (win_got, x_got, ci_got, g_got) = (got.pop().unwrap(), got.pop().unwrap(), got.pop().unwrap(), got.pop().unwrap());
        close(&g_got, gated.data(), "gated");
        // conv_in is rounded to f16: an f16 step apart where the sums round differently
        for (i, (a, e)) in ci_got.iter().zip(conv_in.data()).enumerate() {
            assert!((a - e).abs() <= 2e-3 * e.abs().max(1.0), "conv_in [{i}]: {a} against {e}");
        }
        for (i, (a, e)) in x_got.iter().zip(want.data()).enumerate() {
            assert!((a - e).abs() <= 1e-2 * e.abs().max(1.0), "x [{i}] of {rows} rows: {a} against {e}");
        }
        close(&win_got, window.data(), "the window");
        // the next run's window is the CPU's (as a step after a prompt starts from the prompt's)
        DeviceChain::upload(&b, &dwin, window.data());
    }
}

/// A residual's add and the next norm in one dispatch give the two ops' answer; the vec4 norm gives the scalar
/// one's (within an f32 step: its sums run in another order).
#[test]
fn an_add_and_norm_in_one_are_the_two() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(31);
    for (rows, n) in [(1usize, 5120usize), (3, 256), (2, 20)] {
        let (x0, y0, w0): (Vec<f32>, Vec<f32>, Vec<f32>) = ((0..rows * n).map(|_| r()).collect(), (0..rows * n).map(|_| r()).collect(), (0..n).map(|_| r()).collect());
        let up = |v: &[f32]| {
            let d = b.vec(v.len());
            DeviceChain::upload(&b, &d, v);
            d
        };
        let (xa, xb, y, w, oa, ob) = (up(&x0), up(&x0), up(&y0), up(&w0), b.vec(rows * n), b.vec(rows * n));
        let mut rec = b.begin();
        rec.add(&xa, &y);
        rec.rmsnorm_rows(&xa, &w, &oa, rows, 1e-6);
        rec.add_rmsnorm_rows(&xb, &y, &w, &ob, rows, 1e-6);
        for v in [&xa, &xb, &oa, &ob] {
            rec.read(v);
        }
        let got = rec.finish();
        assert_eq!(got[0], got[1], "the sums alike, {rows} rows of {n}");
        // the expected norm on the host
        let want: Vec<f32> = got[0].chunks_exact(n).flat_map(|row| {
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-6).sqrt();
            row.iter().zip(&w0).map(move |(v, g)| v * inv * g).collect::<Vec<_>>()
        }).collect();
        close(&got[2], &want, "the two ops");
        close(&got[3], &want, "in one");
    }
}

/// A hyper-connection's fused projections are the ops they replace, bit for bit: the down matrix with its gates
/// against the f16 matmul then the gates, the up matrix with its mix against the matmul then the mix, for a step's
/// row and a check's few, at a small shape and at Flash-Next's (four streams of 2,560, a rank of 320 and four
/// writes).
#[test]
fn a_hyper_connections_fused_projections_are_their_two_ops() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut next = rng(41);
    for (streams, d, rank, writes) in [(4usize, 64usize, 12usize, 4usize), (2, 96, 20, 4), (4, 2560, 320, 4)] {
        let (width, n) = (streams * d, rank + writes);
        let f16s = |len: usize, next: &mut dyn FnMut() -> f32| -> Vec<f32> { (0..len).map(|_| half::f16::from_f32(0.05 * next()).to_f32()).collect() };
        let (down, up) = (f16s(n * width, &mut next), f16s(width * n, &mut next));
        let (dw, uw) = (DeviceChain::vec_f16(&b, &down).unwrap(), DeviceChain::vec_f16(&b, &up).unwrap());
        for rows in [1usize, 3, 8] {
            let normed: Vec<f32> = (0..rows * width).map(|_| next()).collect();
            let nd = b.vec(rows * width);
            DeviceChain::upload(&b, &nd, &normed);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
            // the two ops each
            let (t0, post0, logits0, out0) = (b.vec(rows * n), b.vec(rows * writes), b.vec(rows * width), b.vec(rows * d));
            let mut rec = b.begin();
            rec.matmul_f16_rows(&dw, n, width, &nd, &t0, rows);
            rec.hc_gates(&t0, &post0, rows, rank, writes, streams);
            rec.matmul_f16_rows(&uw, width, n, &t0, &logits0, rows);
            rec.hc_mix(&logits0, &nd, &out0, rows, streams, d);
            rec.read(&t0);
            rec.read(&post0);
            rec.read(&out0);
            let want = rec.finish();
            // the fused ones
            let (t1, post1, logits1, out1) = (b.vec(rows * n), b.vec(rows * writes), b.vec(rows * width), b.vec(rows * d));
            let at = std::time::Instant::now();
            let mut rec = b.begin();
            rec.hc_down_gates(&dw, width, &nd, &t1, &post1, rows, rank, writes, streams);
            rec.hc_up_mix(&uw, n, &t1, &logits1, &nd, &out1, rows, streams, d);
            rec.read(&t1);
            rec.read(&post1);
            rec.read(&out1);
            let got = rec.finish();
            // (the small shapes come first: a kernel that is slow there stops the test before the model's shape)
            let took = at.elapsed().as_secs_f64();
            println!("{streams} streams of {d}, {rows} rows: the fused projections in {:.1} ms (with their pipelines the first time)", took * 1e3);
            assert!(took < 5.0, "a hyper-connection's fused projections took {took:.1} s: stop and look");
            for (what, (g, w), len) in [("the gated projection", (&got[0], &want[0]), rows * n), ("the write weights", (&got[1], &want[1]), rows * writes), ("the mix", (&got[2], &want[2]), rows * d)] {
                assert!(w[..len].iter().any(|v| *v != 0.0), "{what}: values to compare");
                assert_eq!(bits(&g[..len]), bits(&w[..len]), "{streams} streams of {d}, {rows} rows: {what}");
            }
        }
    }
}
