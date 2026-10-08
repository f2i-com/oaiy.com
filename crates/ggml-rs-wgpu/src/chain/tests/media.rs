//! The picture, video and sound models' ops against the host's.
use super::*;

/// A diffusion transformer's ops as the host computes them: the modulated layer norm (with and without a shift,
/// rows of 4,096 and of 100), the gated residual (its gate's tanh or not), GELU, and a BF16 weight's f16 rounding
/// (its tiny values to the nearest f16, a value past f16's range refused).
#[test]
fn a_diffusion_transformers_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(77);
    for (rows, n) in [(37usize, 4096usize), (5, 100)] {
        let x: Vec<f32> = (0..rows * n).map(|_| r() * 3.0 + 0.5).collect();
        let mods: Vec<f32> = (0..4 * n).map(|_| r()).collect();
        let (xd, md, out) = (b.vec(rows * n), b.vec(4 * n), b.vec(rows * n));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &md, &mods);
        for shift in [None, Some(3 * n)] {
            let mut rec = b.begin();
            rec.layernorm_mod_rows(&xd, &out, rows, n, &md, n, shift, 1e-6);
            rec.read(&out);
            let got = rec.finish().pop().unwrap();
            for row in 0..rows {
                let v = &x[row * n..(row + 1) * n];
                let mean = v.iter().map(|&a| a as f64).sum::<f64>() / n as f64;
                let var = v.iter().map(|&a| (a as f64 - mean).powi(2)).sum::<f64>() / n as f64;
                for i in 0..n {
                    let want = (v[i] as f64 - mean) / (var + 1e-6).sqrt() * (1.0 + mods[n + i] as f64) + shift.map_or(0.0, |s| mods[s + i] as f64);
                    let g = got[row * n + i] as f64;
                    assert!((g - want).abs() <= 1e-4 * (1.0 + want.abs()), "layer norm row {row} [{i}] shift {shift:?}: {g} against {want}");
                }
            }
        }
        for tanh in [true, false] {
            let y: Vec<f32> = (0..rows * n).map(|_| r()).collect();
            let yd = b.vec(rows * n);
            DeviceChain::upload(&b, &yd, &y);
            DeviceChain::upload(&b, &xd, &x);
            let mut rec = b.begin();
            rec.add_gated_rows(&xd, &yd, rows, n, &md, 2 * n, tanh);
            rec.read(&xd);
            let got = rec.finish().pop().unwrap();
            for (i, g) in got.iter().enumerate() {
                let gate = mods[2 * n + i % n];
                let want = x[i] + y[i] * if tanh { gate.tanh() } else { gate };
                assert!((g - want).abs() <= 1e-5 * (1.0 + want.abs()), "gated residual [{i}] tanh {tanh}: {g} against {want}");
            }
        }
    }
    let x: Vec<f32> = (0..1000).map(|_| r() * 6.0).collect();
    let (xd, yd) = (b.vec(1000), b.vec(1000));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.gelu(&xd, &yd, 1000);
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for (g, &v) in got.iter().zip(&x) {
        let want = 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v * v * v)).tanh());
        assert!((g - want).abs() <= 1e-5 * (1.0 + want.abs()), "gelu({v}): {g} against {want}");
    }
    // a BF16 weight's tiny values to the nearest f16; a value past f16's range refused
    let w = [1.5f32, -3.0e-7, 7.0e-9, 0.333_333_34, 65504.0, -2.0];
    let wd = b.vec_f16_rounded(&w).expect("values within f16's range");
    let mut rec = b.begin();
    rec.read(&wd);
    let words = rec.finish().pop().unwrap();
    let back: Vec<f32> = words.iter().flat_map(|v| [half::f16::from_bits(v.to_bits() as u16).to_f32(), half::f16::from_bits((v.to_bits() >> 16) as u16).to_f32()]).collect();
    assert_eq!(back, w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect::<Vec<_>>());
    assert!(b.vec_f16_rounded(&[1.0, 70000.0]).is_none(), "past f16's range");
    assert!(b.vec_f16_rounded(&[1.0, f32::NAN]).is_none(), "not a number");
}

/// A sparse decoder's gathers as the host makes them: rows picked by an index (from an offset into it, a missing one
/// zeros), and each channel repeated and added.
#[test]
fn gathered_and_repeated_rows_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(149);
    let (src, c, rows, first, repeat) = (37usize, 12usize, 50usize, 7usize, 3usize);
    let x: Vec<f32> = (0..src * c).map(|_| r()).collect();
    // (every source row, and the missing one, some twice)
    let index: Vec<u32> = (0..first + rows).map(|i| ((i * 13) % (src + 1)) as u32).collect();
    let (xd, id, out) = (b.vec(x.len()), b.vec(index.len()), b.vec(rows * c));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &id, &index.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>());
    let acc: Vec<f32> = (0..rows * c * repeat).map(|_| r()).collect();
    let ad = b.vec(acc.len());
    DeviceChain::upload(&b, &ad, &acc);
    let mut rec = b.begin();
    rec.gather_rows(&xd, &id, &out, rows, c, first, src);
    rec.repeat_cols_add_rows(&out, &ad, rows, c, repeat);
    rec.read(&out);
    rec.read(&ad);
    let got = rec.finish();
    for i in 0..rows * c {
        let s = index[first + i / c] as usize;
        let want = if s < src { x[s * c + i % c] } else { 0. };
        assert_eq!(got[0][i], want, "gathered [{i}]");
    }
    for i in 0..rows * c * repeat {
        let w = c * repeat;
        let want = acc[i] + got[0][(i / w) * c + (i % w) / repeat];
        assert_eq!(got[1][i], want, "repeated [{i}]");
    }
}

/// The speech codec's ops as the host computes them: a causal 1-D convolution (its padding all before: 7 taps 3
/// apart), a depthwise causal one, SnakeBeta, and a clamp.
#[test]
fn a_speech_codecs_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(137);
    let (cin, cout, k, dil, len) = (24usize, 20usize, 7usize, 3usize, 40usize);
    let wt: Vec<f32> = (0..cout * cin * k).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
    let x: Vec<f32> = (0..len * cin).map(|_| r()).collect();
    let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
    let wd = b.conv1d_weights(&wt, cout, cin, k).expect("the weights");
    let (xd, yd, bd) = (b.vec(x.len()), b.vec(len * cout), b.vec(cout));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &bd, &bias);
    let mut rec = b.begin();
    rec.conv1d_padded_rows(&wd, &bd, cout, cin, k, dil, (k - 1) * dil, &xd, len, &yd);
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for t in 0..len {
        for co in 0..cout {
            let mut want = bias[co] as f64;
            for c in 0..cin {
                for j in 0..k {
                    let s = t as isize + (j * dil) as isize - ((k - 1) * dil) as isize;
                    if s >= 0 {
                        want += wt[(co * cin + c) * k + j] as f64 * x[s as usize * cin + c] as f64;
                    }
                }
            }
            let g = got[t * cout + co] as f64;
            assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "causal conv at {t}, {co}: {g} against {want}");
        }
    }
    let (c, k, len) = (12usize, 7usize, 30usize);
    let w: Vec<f32> = (0..c * k).map(|_| r()).collect();
    let bias: Vec<f32> = (0..c).map(|_| r()).collect();
    let x: Vec<f32> = (0..len * c).map(|_| r()).collect();
    let (wd, bd, xd, yd) = (b.vec(w.len()), b.vec(c), b.vec(x.len()), b.vec(x.len()));
    DeviceChain::upload(&b, &wd, &w);
    DeviceChain::upload(&b, &bd, &bias);
    DeviceChain::upload(&b, &xd, &x);
    let freq: Vec<f32> = (0..c).map(|_| (r() * 0.5).exp()).collect();
    let scale: Vec<f32> = (0..c).map(|_| 1.0 / ((r() * 0.5).exp() + 1e-9)).collect();
    let (fd, sd, td) = (b.vec(c), b.vec(c), b.vec(x.len()));
    DeviceChain::upload(&b, &fd, &freq);
    DeviceChain::upload(&b, &sd, &scale);
    DeviceChain::upload(&b, &td, &x.iter().map(|v| v * 3.0).collect::<Vec<_>>());
    let mut rec = b.begin();
    rec.depthwise_causal_conv1d_rows(&wd, &bd, c, k, &xd, len, &yd);
    rec.read(&yd);
    rec.snake_beta_rows(&xd, &fd, &sd, len, c);
    rec.read(&xd);
    rec.clamp_in_place(&td, len * c, -1.0, 1.0);
    rec.read(&td);
    let got = rec.finish();
    for t in 0..len {
        for ch in 0..c {
            let mut want = bias[ch];
            for j in 0..k {
                let s = t as isize + j as isize - k as isize + 1;
                if s >= 0 {
                    want += w[ch * k + j] * x[s as usize * c + ch];
                }
            }
            let i = t * c + ch;
            assert!((got[0][i] - want).abs() <= 1e-5 * (1.0 + want.abs()), "depthwise conv at {t}, {ch}: {} against {want}", got[0][i]);
            let v = x[i];
            let snake = v + scale[ch] * (freq[ch] * v).sin().powi(2);
            assert!((got[1][i] - snake).abs() <= 1e-5 * (1.0 + snake.abs()), "SnakeBeta at {t}, {ch}: {} against {snake}", got[1][i]);
            assert_eq!(got[2][i], (v * 3.0).clamp(-1.0, 1.0), "a clamp at {t}, {ch}");
        }
    }
}

/// A DAC decoder's ops as the host computes them: 1-D convolutions (7 taps 3 apart, 1 tap; channels not of 32),
/// transposed ones (stride 4, and an odd stride with output padding: DAC's `ceil(s / 2)` padding, `s % 2` extra),
/// Snake, and tanh.
#[test]
fn a_dacs_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(131);
    for (cin, cout, k, dil, len) in [(20usize, 36usize, 7usize, 3usize, 50usize), (64, 40, 7, 1, 70), (48, 8, 1, 1, 33)] {
        let wt: Vec<f32> = (0..cout * cin * k).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
        let x: Vec<f32> = (0..len * cin).map(|_| r()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let wd = b.conv1d_weights(&wt, cout, cin, k).expect("the weights");
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(len * cout), b.vec(cout));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv1d_rows(&wd, &bd, cout, cin, k, dil, &xd, len, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        let pad = (k / 2 * dil) as isize;
        for t in 0..len {
            for co in 0..cout {
                let mut want = bias[co] as f64;
                for c in 0..cin {
                    for j in 0..k {
                        let s = t as isize + (j * dil) as isize - pad;
                        if s >= 0 && (s as usize) < len {
                            want += wt[(co * cin + c) * k + j] as f64 * x[s as usize * cin + c] as f64;
                        }
                    }
                }
                let g = got[t * cout + co] as f64;
                assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "1-D conv of {k} taps {dil} apart at {t}, {co}: {g} against {want}");
            }
        }
    }
    for (cin, cout, stride, len) in [(16usize, 12usize, 4usize, 9usize), (8, 4, 3, 7)] {
        let (k, pad, out_pad) = (2 * stride, stride.div_ceil(2), stride % 2);
        // PyTorch's [cin, cout, k], and the kernel's [cout, k, cin]
        let wt: Vec<f32> = (0..cin * cout * k).map(|_| r() * 0.3).collect();
        let packed: Vec<f32> = (0..cout * k * cin).map(|i| { let (co, j, c) = (i / (k * cin), (i / cin) % k, i % cin); wt[(c * cout + co) * k + j] }).collect();
        let x: Vec<f32> = (0..len * cin).map(|_| r()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let out = (len - 1) * stride - 2 * pad + k + out_pad;
        let mut want = vec![0f64; out * cout];
        for (o, row) in want.chunks_mut(cout).enumerate() {
            for (co, v) in row.iter_mut().enumerate() {
                *v = bias[co] as f64;
                for i in 0..len {
                    for j in 0..k {
                        if i * stride + j == o + pad {
                            for c in 0..cin {
                                *v += wt[(c * cout + co) * k + j] as f64 * x[i * cin + c] as f64;
                            }
                        }
                    }
                }
            }
        }
        let (wd, xd, bd, yd) = (b.vec(packed.len()), b.vec(x.len()), b.vec(cout), b.vec(out * cout));
        DeviceChain::upload(&b, &wd, &packed);
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv_transpose1d_rows(&wd, &bd, cout, cin, k, stride, pad, &xd, len, out, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((*g as f64 - w).abs() <= 1e-4 * (1.0 + w.abs()), "transposed 1-D conv, stride {stride}, [{i}]: {g} against {w}");
        }
    }
    let (rows, c) = (11usize, 6usize);
    let x: Vec<f32> = (0..rows * c).map(|_| r() * 3.0).collect();
    let alpha: Vec<f32> = (0..c).map(|_| r() + 1.5).collect();
    let (xd, ad, td) = (b.vec(x.len()), b.vec(c), b.vec(x.len()));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &ad, &alpha);
    DeviceChain::upload(&b, &td, &x);
    let mut rec = b.begin();
    rec.snake_rows(&xd, &ad, rows, c);
    rec.tanh_in_place(&td, rows * c);
    rec.read(&xd);
    rec.read(&td);
    let got = rec.finish();
    for (i, &v) in x.iter().enumerate() {
        let a = alpha[i % c];
        let snake = v + (a * v).sin().powi(2) / (a + 1e-9);
        assert!((got[0][i] - snake).abs() <= 1e-5 * (1.0 + snake.abs()), "Snake [{i}]: {} against {snake}", got[0][i]);
        assert!((got[1][i] - v.tanh()).abs() <= 1e-6, "tanh [{i}]: {} against {}", got[1][i], v.tanh());
    }
}

/// BiRefNet's ops as the host computes them: Swin's windows there and back (shifted and not, padded), its window
/// attention (relative bias, the shifted windows' mask), a bilinear resize with the corners aligned, a picture as
/// patches, a modulated deformable convolution's taps, a column's mean, a broadcast, and a 7x7 convolution.
#[test]
fn birefnets_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(127);
    let close = |got: &[f32], want: &[f32], what: &str| {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{what} [{i}]: {g} against {w}");
        }
    };
    // windows of 4 over 6x7 tokens of 5 (padded to 8x8), shifted by 2 and not
    let (h, w, c, win) = (6usize, 7usize, 5usize, 4usize);
    let (hp, wp) = (8usize, 8usize);
    let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
    let xd = b.vec(x.len());
    DeviceChain::upload(&b, &xd, &x);
    for shift in [0usize, 2] {
        let mut want = vec![0f32; hp * wp * c];
        for (i, v) in want.iter_mut().enumerate() {
            let (ch, tok) = (i % c, i / c);
            let (wdx, t) = (tok / (win * win), tok % (win * win));
            let py = ((wdx / (wp / win)) * win + t / win + shift) % hp;
            let px = ((wdx % (wp / win)) * win + t % win + shift) % wp;
            if py < h && px < w {
                *v = x[(py * w + px) * c + ch];
            }
        }
        let (wd, acc) = (b.vec(hp * wp * c), b.vec(h * w * c));
        DeviceChain::upload(&b, &acc, &vec![1.0; h * w * c]);
        let mut rec = b.begin();
        rec.window_rows(&xd, &wd, h, w, c, win, shift);
        rec.read(&wd);
        rec.unwindow_add_rows(&wd, &acc, h, w, c, win, shift);
        rec.read(&acc);
        let got = rec.finish();
        close(&got[0], &want, &format!("windows shifted {shift}"));
        close(&got[1], &x.iter().map(|v| v + 1.0).collect::<Vec<_>>(), &format!("windows back shifted {shift}"));
    }
    // window attention: 2 heads of 32 over those windows' tokens, shifted and not
    let heads = 2usize;
    let cc = heads * 32;
    let n = win * win;
    let qkv: Vec<f32> = (0..hp * wp * 3 * cc).map(|_| r()).collect();
    let table: Vec<f32> = (0..(2 * win - 1) * (2 * win - 1) * heads).map(|_| r()).collect();
    let (qd, td, od) = (b.vec(qkv.len()), b.vec(table.len()), b.vec(hp * wp * cc));
    DeviceChain::upload(&b, &qd, &qkv);
    DeviceChain::upload(&b, &td, &table);
    let scale = 1.0 / 32f32.sqrt();
    for shift in [0usize, 2] {
        let region = |i: usize, len: usize| if i < len - win { 0 } else if i < len - shift { 1 } else { 2 };
        let mut want = vec![0f32; hp * wp * cc];
        for wdx in 0..(hp / win) * (wp / win) {
            let (gy, gx) = ((wdx / (wp / win)) * win, (wdx % (wp / win)) * win);
            for head in 0..heads {
                for qi in 0..n {
                    let q = &qkv[(wdx * n + qi) * 3 * cc + head * 32..][..32];
                    let scores: Vec<f32> = (0..n)
                        .map(|kj| {
                            let k = &qkv[(wdx * n + kj) * 3 * cc + cc + head * 32..][..32];
                            let mut s: f32 = q.iter().zip(k).map(|(a, b)| a * scale * b).sum();
                            s += table[((qi / win + win - 1 - kj / win) * (2 * win - 1) + qi % win + win - 1 - kj % win) * heads + head];
                            if shift > 0 && region(gy + qi / win, hp) * 3 + region(gx + qi % win, wp) != region(gy + kj / win, hp) * 3 + region(gx + kj % win, wp) {
                                s -= 100.0;
                            }
                            s
                        })
                        .collect();
                    let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
                    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                    let l: f32 = e.iter().sum();
                    for d in 0..32 {
                        want[(wdx * n + qi) * cc + head * 32 + d] = (0..n).map(|kj| e[kj] * qkv[(wdx * n + kj) * 3 * cc + 2 * cc + head * 32 + d]).sum::<f32>() / l;
                    }
                }
            }
        }
        let mut rec = b.begin();
        rec.window_attention(&qd, &td, &od, h, w, heads, win, shift, scale);
        rec.read(&od);
        close(&rec.finish().pop().unwrap(), &want, &format!("window attention shifted {shift}"));
    }
    // a bilinear resize, corners aligned
    let (h, w, c, oh, ow) = (5usize, 7usize, 3usize, 9usize, 4usize);
    let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
    let at = |n_in: usize, n_out: usize, o: usize| {
        let src = if n_out > 1 { (n_in - 1) as f32 / (n_out - 1) as f32 * o as f32 } else { 0.0 };
        let i0 = (src as usize).min(n_in - 1);
        (i0, (i0 + 1).min(n_in - 1), src - i0 as f32)
    };
    let want: Vec<f32> = (0..oh * ow * c)
        .map(|i| {
            let (ch, ox, oy) = (i % c, (i / c) % ow, i / (c * ow));
            let ((y0, y1, fy), (x0, x1, fx)) = (at(h, oh, oy), at(w, ow, ox));
            let v = |y: usize, xx: usize| x[(y * w + xx) * c + ch];
            (1.0 - fy) * ((1.0 - fx) * v(y0, x0) + fx * v(y0, x1)) + fy * ((1.0 - fx) * v(y1, x0) + fx * v(y1, x1))
        })
        .collect();
    let (xd, yd) = (b.vec(x.len()), b.vec(want.len()));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.resize_bilinear_rows(&xd, &yd, h, w, c, oh, ow);
    rec.read(&yd);
    close(&rec.finish().pop().unwrap(), &want, "a bilinear resize");
    // a picture of 8 as patches of 4
    let (s, c, size) = (8usize, 3usize, 4usize);
    let g = s / size;
    let x: Vec<f32> = (0..s * s * c).map(|_| r()).collect();
    let want: Vec<f32> = (0..s * s * c)
        .map(|i| {
            let (oc, pix) = (i % (c * g * g), i / (c * g * g));
            let (ch, hg, wg) = (oc / (g * g), (oc / g) % g, oc % g);
            x[((hg * size + pix / size) * s + wg * size + pix % size) * c + ch]
        })
        .collect();
    let (xd, yd) = (b.vec(x.len()), b.vec(want.len()));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.blocks_to_channels_rows(&xd, &yd, s, c, size);
    rec.read(&yd);
    close(&rec.finish().pop().unwrap(), &want, "patches");
    // a deformable 3x3's taps for pixels 5..30 of a 6x7 image of 4 channels
    let (h, w, c, k, first, pixels) = (6usize, 7usize, 4usize, 3usize, 5usize, 25usize);
    let kk = k * k;
    let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
    let offs: Vec<f32> = (0..h * w * 2 * kk).map(|_| r() * 2.5).collect();
    let mods: Vec<f32> = (0..h * w * kk).map(|_| r() + 1.0).collect();
    let mut want = vec![0f32; pixels * kk * c];
    for pi in 0..pixels {
        let pix = first + pi;
        for t in 0..kk {
            let y = (pix / w) as f32 - 1.0 + (t / k) as f32 + offs[pix * 2 * kk + 2 * t];
            let xx = (pix % w) as f32 - 1.0 + (t % k) as f32 + offs[pix * 2 * kk + 2 * t + 1];
            if y <= -1.0 || y >= h as f32 || xx <= -1.0 || xx >= w as f32 {
                continue;
            }
            let (y0, x0) = (y.floor(), xx.floor());
            let (ly, lx) = (y - y0, xx - x0);
            for ch in 0..c {
                let mut v = 0.0;
                for (dy, dx, wgt) in [(0i64, 0i64, (1.0 - ly) * (1.0 - lx)), (0, 1, (1.0 - ly) * lx), (1, 0, ly * (1.0 - lx)), (1, 1, ly * lx)] {
                    let (yy, xc) = (y0 as i64 + dy, x0 as i64 + dx);
                    if yy >= 0 && xc >= 0 && (yy as usize) < h && (xc as usize) < w {
                        v += wgt * x[(yy as usize * w + xc as usize) * c + ch];
                    }
                }
                want[(pi * kk + t) * c + ch] = v * mods[pix * kk + t];
            }
        }
    }
    let (xd, od, md, yd) = (b.vec(x.len()), b.vec(offs.len()), b.vec(mods.len()), b.vec(want.len()));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &od, &offs);
    DeviceChain::upload(&b, &md, &mods);
    let mut rec = b.begin();
    rec.deform_im2col_rows(&xd, &od, &md, &yd, h, w, c, k, first, pixels);
    rec.read(&yd);
    close(&rec.finish().pop().unwrap(), &want, "a deformable convolution's taps");
    // a column's mean, and broadcast into rows 6 apart at 2
    let (rows, c) = (37usize, 3usize);
    let x: Vec<f32> = (0..rows * c).map(|_| r()).collect();
    let mean: Vec<f32> = (0..c).map(|ch| (0..rows).map(|row| x[row * c + ch]).sum::<f32>() / rows as f32).collect();
    let (xd, md, bd) = (b.vec(x.len()), b.vec(c), b.vec(rows * 6));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.mean_rows(&xd, &md, rows, c);
    rec.broadcast_rows(&md, &bd, rows, c, 6, 2);
    rec.read(&md);
    rec.read(&bd);
    let got = rec.finish();
    close(&got[0], &mean, "a mean");
    for row in 0..rows {
        close(&got[1][row * 6 + 2..row * 6 + 5], &mean, "a broadcast");
    }
    // a 7x7 convolution (16 channels to 8, 9x10 pixels)
    let (cin, cout, h, w) = (16usize, 8usize, 9usize, 10usize);
    let wt: Vec<f32> = (0..cout * cin * 49).map(|_| half::f16::from_f32(r() * 0.1).to_f32()).collect();
    let x: Vec<f32> = (0..h * w * cin).map(|_| half::f16::from_f32(r()).to_f32()).collect();
    let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
    let wd = b.conv_weights(&wt, cout, cin, 7).expect("the weights");
    let (xd, yd, bd) = (b.vec(x.len()), b.vec(h * w * cout), b.vec(cout));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &bd, &bias);
    let mut rec = b.begin();
    rec.conv_rows(&wd, &bd, cout, cin, 7, &xd, h, w, &yd);
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for py in 0..h {
        for px in 0..w {
            for co in 0..cout {
                let mut want = bias[co] as f64;
                for ch in 0..cin {
                    for ky in 0..7 {
                        for kx in 0..7 {
                            let (iy, ix) = (py as isize + ky as isize - 3, px as isize + kx as isize - 3);
                            if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                want += wt[((co * cin + ch) * 7 + ky) * 7 + kx] as f64 * x[(iy as usize * w + ix as usize) * cin + ch] as f64;
                            }
                        }
                    }
                }
                let g = got[(py * w + px) * cout + co] as f64;
                assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "7x7 conv at ({py}, {px}) channel {co}: {g} against {want}");
            }
        }
    }
}

/// Real-ESRGAN's ops as the host computes them: a 3x3 convolution of a concatenation's leading channels (its
/// pixels' values further apart than it reads), and a leaky ReLU, apart and in place.
#[test]
fn real_esrgans_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(97);
    for (cin, xs, cout, h, w) in [(64usize, 192usize, 32usize, 9usize, 13usize), (160, 192, 64, 7, 5), (3, 3, 64, 6, 11)] {
        let wt: Vec<f32> = (0..cout * cin * 9).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
        let x: Vec<f32> = (0..h * w * xs).map(|_| half::f16::from_f32(r()).to_f32()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let wd = b.conv_weights(&wt, cout, cin, 3).expect("the weights");
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(h * w * cout), b.vec(cout));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv_rows_strided(&wd, &bd, cout, cin, 3, &xd, xs, h, w, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for py in 0..h {
            for px in 0..w {
                for co in 0..cout {
                    let mut want = bias[co] as f64;
                    for c in 0..cin {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let (iy, ix) = (py as isize + ky as isize - 1, px as isize + kx as isize - 1);
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                    want += wt[((co * cin + c) * 3 + ky) * 3 + kx] as f64 * x[(iy as usize * w + ix as usize) * xs + c] as f64;
                                }
                            }
                        }
                    }
                    let g = got[(py * w + px) * cout + co] as f64;
                    assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "3x3 conv of {cin} of {xs} channels to {cout} at ({py}, {px}) channel {co}: {g} against {want}");
                }
            }
        }
    }
    let x: Vec<f32> = (0..1000).map(|_| r() * 4.0).collect();
    let want: Vec<f32> = x.iter().map(|&v| if v > 0.0 { v } else { 0.2 * v }).collect();
    let (xd, yd) = (b.vec(1000), b.vec(1000));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.leaky_relu(&xd, &yd, 1000, 0.2);
    rec.read(&yd);
    rec.leaky_relu(&xd, &xd, 1000, 0.2);
    rec.read(&xd);
    let got = rec.finish();
    for (i, w) in want.iter().enumerate() {
        assert!((got[0][i] - w).abs() <= 1e-6 * (1.0 + w.abs()), "a leaky ReLU [{i}]: {} against {w}", got[0][i]);
        assert!((got[1][i] - w).abs() <= 1e-6 * (1.0 + w.abs()), "a leaky ReLU in place [{i}]: {} against {w}", got[1][i]);
    }
}

/// A VAE's ops as the host computes them: 3x3 and 1x1 convolutions on the tensor cores with their bias (channels
/// not of 32 and of 32, an image not of the tile, inputs past f16's range: a VAE's reach 230,000), the nearest
/// upsampling, and Wan's shuffled shortcut (one frame and two).
#[test]
fn a_vaes_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(31);
    for (cin, cout, h, w, k, big) in [(20usize, 36usize, 9usize, 13usize, 3usize, 1f32), (64, 160, 16, 16, 3, 1.0), (144, 4, 5, 7, 3, 1.0), (48, 40, 11, 6, 1, 1.0), (32, 24, 7, 9, 3, 300_000.0), (40, 8, 6, 5, 1, 300_000.0)] {
        let taps = k * k;
        let wt: Vec<f32> = (0..cout * cin * taps).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
        let x: Vec<f32> = (0..h * w * cin).map(|_| half::f16::from_f32(r()).to_f32() * big).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let Some(wd) = b.conv_weights(&wt, cout, cin, k) else { return };
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(h * w * cout), b.vec(cout));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv_rows(&wd, &bd, cout, cin, k, &xd, h, w, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        let off = (k / 2) as isize;
        for py in 0..h {
            for px in 0..w {
                for co in 0..cout {
                    let mut want = bias[co] as f64;
                    for c in 0..cin {
                        for ky in 0..k {
                            for kx in 0..k {
                                let (iy, ix) = (py as isize + ky as isize - off, px as isize + kx as isize - off);
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                    want += wt[((co * cin + c) * k + ky) * k + kx] as f64 * x[(iy as usize * w + ix as usize) * cin + c] as f64;
                                }
                            }
                        }
                    }
                    let g = got[(py * w + px) * cout + co] as f64;
                    assert!((g - want).abs() <= 1e-3 * (big as f64 + want.abs()), "{k}x{k} conv {cin}->{cout} (inputs to {big}) at ({py}, {px}) channel {co}: {g} against {want}");
                }
            }
        }
    }
    let (h, w, c) = (3usize, 5usize, 6usize);
    let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
    let (xd, yd) = (b.vec(x.len()), b.vec(4 * x.len()));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.upsample2x_rows(&xd, &yd, h, w, c);
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for oy in 0..2 * h {
        for ox in 0..2 * w {
            for ch in 0..c {
                assert_eq!(got[(oy * 2 * w + ox) * c + ch], x[((oy / 2) * w + ox / 2) * c + ch], "upsampled ({oy}, {ox}) {ch}");
            }
        }
    }
    for (cin, cout, ft) in [(8usize, 8usize, 2usize), (8, 4, 1), (12, 6, 2)] {
        let repeats = cout * ft * 4 / cin;
        let x: Vec<f32> = (0..h * w * cin).map(|_| r()).collect();
        let base: Vec<f32> = (0..4 * h * w * cout).map(|_| r()).collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(base.len()));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &yd, &base);
        let mut rec = b.begin();
        rec.shuffle_up_add_rows(&xd, &yd, h, w, cin, cout, ft);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        // as the host's: repeat each channel, view as (cout, ft, 2, 2), keep the last frame, shuffle into pixels
        for co in 0..cout {
            for a in 0..2 {
                for bb in 0..2 {
                    for y in 0..h {
                        for xx in 0..w {
                            let e = ((co * ft + ft - 1) * 2 + a) * 2 + bb;
                            let ci = e / repeats;
                            let o = ((2 * y + a) * 2 * w + 2 * xx + bb) * cout + co;
                            assert_eq!(got[o], base[o] + x[(y * w + xx) * cin + ci], "shuffled {cin}->{cout} ft {ft} at {o}");
                        }
                    }
                }
            }
        }
    }
}

/// Wan's encoder's downsampling ops as the host computes them: the odd rows' odd columns, and the shortcut's
/// shuffled means (time slots before the last zero; a spatial factor of 2 and of 1).
#[test]
fn a_vae_encoders_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(83);
    let (h, w, c) = (6usize, 8usize, 5usize);
    let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
    let (xd, yd) = (b.vec(x.len()), b.vec(x.len() / 4));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.subsample2x_rows(&xd, &yd, h, w, c);
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for oy in 0..h / 2 {
        for ox in 0..w / 2 {
            for ch in 0..c {
                assert_eq!(got[(oy * w / 2 + ox) * c + ch], x[((2 * oy + 1) * w + 2 * ox + 1) * c + ch], "subsampled ({oy}, {ox}) {ch}");
            }
        }
    }
    for (cin, cout, ft, fs) in [(4usize, 8usize, 2usize, 2usize), (4, 16, 1, 2), (6, 6, 1, 1), (8, 4, 2, 2)] {
        let x: Vec<f32> = (0..h * w * cin).map(|_| r()).collect();
        let (oh, ow) = (h / fs, w / fs);
        let base: Vec<f32> = (0..oh * ow * cout).map(|_| r()).collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(base.len()));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &yd, &base);
        let mut rec = b.begin();
        rec.shuffle_down_mean_add_rows(&xd, &yd, h, w, cin, cout, ft, fs);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        let g = cin * ft * fs * fs / cout;
        for oy in 0..oh {
            for ox in 0..ow {
                for co in 0..cout {
                    let mut s = 0.0f64;
                    for e in co * g..(co + 1) * g {
                        let (c, t, fy, fx) = (e / (ft * fs * fs), (e / (fs * fs)) % ft, (e / fs) % fs, e % fs);
                        if t == ft - 1 {
                            s += x[((oy * fs + fy) * w + ox * fs + fx) * cin + c] as f64;
                        }
                    }
                    let want = base[(oy * ow + ox) * cout + co] as f64 + s / g as f64;
                    let g = got[(oy * ow + ox) * cout + co] as f64;
                    assert!((g - want).abs() < 1e-5, "shuffled mean {cin}->{cout} ({ft}, {fs}) at ({oy}, {ox}) {co}: {g} against {want}");
                }
            }
        }
    }
}

/// LTX's encoder's single-image packing as the host computes it: space to depth (time slots one frame's; space,
/// time and both) and the shortcut's group means.
#[test]
fn ltxs_encoder_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(89);
    let (h, w, c) = (4usize, 6usize, 3usize);
    let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
    let xd = b.vec(x.len());
    DeviceChain::upload(&b, &xd, &x);
    for (st, sh, sw) in [(1usize, 2usize, 2usize), (2, 1, 1), (2, 2, 2)] {
        let vol = st * sh * sw;
        let (oh, ow) = (h / sh, w / sw);
        let yd = b.vec(oh * ow * c * vol);
        let mut rec = b.begin();
        rec.space_to_depth_rows(&xd, &yd, h, w, c, st, sh, sw);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for oy in 0..oh {
            for ox in 0..ow {
                for ch in 0..c {
                    for t in 0..st {
                        for fy in 0..sh {
                            for fx in 0..sw {
                                let e = ((ch * st + t) * sh + fy) * sw + fx;
                                assert_eq!(got[(oy * ow + ox) * c * vol + e], x[((oy * sh + fy) * w + ox * sw + fx) * c + ch], "space to depth ({st}, {sh}, {sw}) at ({oy}, {ox}) {e}");
                            }
                        }
                    }
                }
            }
        }
    }
    let (rows, cin, cout) = (7usize, 12usize, 4usize);
    let x: Vec<f32> = (0..rows * cin).map(|_| r()).collect();
    let base: Vec<f32> = (0..rows * cout).map(|_| r()).collect();
    let (xd, yd) = (b.vec(x.len()), b.vec(base.len()));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &yd, &base);
    let mut rec = b.begin();
    rec.group_mean_add_rows(&xd, &yd, rows, cin, cout);
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for row in 0..rows {
        for co in 0..cout {
            let g = cin / cout;
            let want = base[row * cout + co] as f64 + (0..g).map(|e| x[row * cin + co * g + e] as f64).sum::<f64>() / g as f64;
            assert!((got[row * cout + co] as f64 - want).abs() < 1e-5, "group mean row {row} {co}");
        }
    }
}

/// The exact GELU as the host's (erf by its series in f64), across a range of inputs.
#[test]
fn an_exact_gelu_is_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    // erf by its Maclaurin series (in f64 near enough to |x| 4.25; past it within 2e-9 of 1)
    let erf = |x: f64| -> f64 {
        if x.abs() > 4.25 {
            return x.signum();
        }
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        while term.abs() > 1e-17 * sum.abs().max(1e-300) && n < 400.0 {
            n += 1.0;
            term *= -x * x / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    };
    let x: Vec<f32> = (0..2001).map(|i| (i as f32 - 1000.0) / 125.0).collect();
    let (xd, yd) = (b.vec(x.len()), b.vec(x.len()));
    DeviceChain::upload(&b, &xd, &x);
    let mut rec = b.begin();
    rec.gelu_erf(&xd, &yd, x.len());
    rec.read(&yd);
    let got = rec.finish().pop().unwrap();
    for (v, g) in x.iter().zip(&got) {
        let want = 0.5 * *v as f64 * (1.0 + erf(*v as f64 / std::f64::consts::SQRT_2));
        assert!((*g as f64 - want).abs() <= 1e-6 * (1.0 + want.abs()), "gelu({v}): {g} against {want}");
    }
}

/// LTX's block ops as the host computes them: the modulated norm over the RMS and with no norm (an affine), the
/// split rotary (each head its own table), and the heads' gate.
#[test]
fn ltxs_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(53);
    let (rows, n) = (7usize, 96usize);
    let x: Vec<f32> = (0..rows * n).map(|_| r() * 2.0 + 0.3).collect();
    let mods: Vec<f32> = (0..3 * n).map(|_| r()).collect();
    let (xd, md, out) = (b.vec(x.len()), b.vec(mods.len()), b.vec(x.len()));
    DeviceChain::upload(&b, &xd, &x);
    DeviceChain::upload(&b, &md, &mods);
    for norm in [ggml_rs::RowNorm::Rms, ggml_rs::RowNorm::None] {
        let mut rec = b.begin();
        rec.norm_mod_rows(&xd, &out, rows, n, &md, n, Some(2 * n), norm, 1e-6);
        rec.read(&out);
        let got = rec.finish().pop().unwrap();
        for row in 0..rows {
            let v = &x[row * n..(row + 1) * n];
            let inv = match norm {
                ggml_rs::RowNorm::Rms => 1.0 / (v.iter().map(|&a| (a as f64).powi(2)).sum::<f64>() / n as f64 + 1e-6).sqrt(),
                _ => 1.0,
            };
            for i in 0..n {
                let want = v[i] as f64 * inv * (1.0 + mods[n + i] as f64) + mods[2 * n + i] as f64;
                let g = got[row * n + i] as f64;
                assert!((g - want).abs() <= 1e-5 * (1.0 + want.abs()), "{norm:?} row {row} [{i}]: {g} against {want}");
            }
        }
    }
    // clean rows (the first two and the last) by a second modulation, three sets on
    let clean = ggml_rs::CleanRows { before: 2, from: rows - 1, offset: 3 * n };
    let both: Vec<f32> = (0..6 * n).map(|_| r()).collect();
    let bd = b.vec(both.len());
    DeviceChain::upload(&b, &bd, &both);
    let acc = b.vec(x.len());
    DeviceChain::upload(&b, &acc, &x);
    let mut rec = b.begin();
    rec.norm_mod_rows_clean(&xd, &out, rows, n, &bd, n, Some(2 * n), ggml_rs::RowNorm::Rms, 1e-6, clean);
    rec.add_gated_rows_clean(&acc, &xd, rows, n, &bd, 0, false, clean);
    rec.read(&out);
    rec.read(&acc);
    let mut reads = rec.finish();
    let (gated, normed) = (reads.pop().unwrap(), reads.pop().unwrap());
    for row in 0..rows {
        let set = if row < 2 || row >= rows - 1 { 3 * n } else { 0 };
        let v = &x[row * n..(row + 1) * n];
        let inv = 1.0 / (v.iter().map(|&a| (a as f64).powi(2)).sum::<f64>() / n as f64 + 1e-6).sqrt();
        for i in 0..n {
            let want = v[i] as f64 * inv * (1.0 + both[set + n + i] as f64) + both[set + 2 * n + i] as f64;
            assert!((normed[row * n + i] as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()), "clean rows' norm, row {row} [{i}]");
            let want = v[i] as f64 * (1.0 + both[set + i] as f64);
            assert!((gated[row * n + i] as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()), "clean rows' gate, row {row} [{i}]");
        }
    }
    // the split rotary: (row, head) its own table, each head's halves paired
    let (heads, hd) = (3usize, 8usize);
    let q: Vec<f32> = (0..rows * heads * hd).map(|_| r()).collect();
    let table: Vec<f32> = (0..rows * heads * hd / 2).flat_map(|i| { let a = i as f32 * 0.37; [a.sin(), a.cos()] }).collect();
    let (qd, td) = (b.vec(q.len()), b.vec(table.len()));
    DeviceChain::upload(&b, &qd, &q);
    DeviceChain::upload(&b, &td, &table);
    let mut rec = b.begin();
    rec.rope_split_rows(&qd, rows, heads, hd, &td);
    rec.read(&qd);
    let got = rec.finish().pop().unwrap();
    for row in 0..rows {
        for h in 0..heads {
            let base = (row * heads + h) * hd;
            for k in 0..hd / 2 {
                let (s, c) = (table[(row * heads + h) * hd + 2 * k], table[(row * heads + h) * hd + 2 * k + 1]);
                let (a, bb) = (q[base + k], q[base + k + hd / 2]);
                assert!((got[base + k] - (a * c - bb * s)).abs() < 1e-5 && (got[base + k + hd / 2] - (a * s + bb * c)).abs() < 1e-5, "rotary row {row} head {h} pair {k}");
            }
        }
    }
    // the heads' gate
    let logits: Vec<f32> = (0..rows * heads).map(|_| r() * 3.0).collect();
    let ld = b.vec(logits.len());
    DeviceChain::upload(&b, &ld, &logits);
    DeviceChain::upload(&b, &qd, &q);
    let mut rec = b.begin();
    rec.head_gate_rows(&qd, &ld, rows, heads, hd);
    rec.read(&qd);
    let got = rec.finish().pop().unwrap();
    for (i, g) in got.iter().enumerate() {
        let want = q[i] * 2.0 / (1.0 + (-logits[i / hd]).exp());
        assert!((g - want).abs() < 1e-5, "gate [{i}]: {g} against {want}");
    }
}

/// LTX's VAE ops as the host computes them: a 3x3x3 convolution on the tensor cores (the first and last frames
/// repeated past the clip's ends, zeros past each frame's edge; one frame and several, channels not of 32, inputs
/// past f16's range) with its bias, and depth to space (time, space and both; the first frame dropped).
#[test]
fn ltxs_vae_ops_are_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(61);
    for (cin, cout, frames, h, w, big) in [(20usize, 36usize, 3usize, 5usize, 7usize, 1f32), (64, 40, 1, 4, 6, 1.0), (32, 8, 4, 3, 5, 300_000.0)] {
        let wt: Vec<f32> = (0..cout * cin * 27).map(|_| half::f16::from_f32(r() * 0.1).to_f32()).collect();
        let x: Vec<f32> = (0..frames * h * w * cin).map(|_| half::f16::from_f32(r()).to_f32() * big).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let Some(wd) = b.conv3d_weights(&wt, cout, cin) else { return };
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(frames * h * w * cout), b.vec(cout));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv3d_rows(&wd, &bd, cout, cin, &xd, frames, h, w, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for t in 0..frames {
            for py in 0..h {
                for px in 0..w {
                    for co in 0..cout {
                        let mut want = bias[co] as f64;
                        for c in 0..cin {
                            for kt in 0..3 {
                                let it = (t as isize + kt as isize - 1).clamp(0, frames as isize - 1) as usize;
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let (iy, ix) = (py as isize + ky as isize - 1, px as isize + kx as isize - 1);
                                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                            want += wt[(((co * cin + c) * 3 + kt) * 3 + ky) * 3 + kx] as f64 * x[((it * h + iy as usize) * w + ix as usize) * cin + c] as f64;
                                        }
                                    }
                                }
                            }
                        }
                        let g = got[((t * h + py) * w + px) * cout + co] as f64;
                        assert!((g - want).abs() <= 1e-3 * (big as f64 + want.abs()), "3D conv {cin}->{cout} at ({t}, {py}, {px}) channel {co}: {g} against {want}");
                    }
                }
            }
        }
    }
    for (st, sh, sw, drop) in [(2usize, 2usize, 2usize, 1usize), (2, 1, 1, 1), (1, 2, 2, 0)] {
        let (frames, h, w, c) = (3usize, 2usize, 3usize, 5usize);
        let vol = st * sh * sw;
        let x: Vec<f32> = (0..frames * h * w * c * vol).map(|_| r()).collect();
        let ot = frames * st - drop;
        let out_len = ot * h * sh * w * sw * c;
        let (xd, yd) = (b.vec(x.len()), b.vec(out_len));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.depth_to_space_rows(&xd, &yd, frames, h, w, c, st, sh, sw, drop);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for o in 0..ot {
            for oy in 0..h * sh {
                for ox in 0..w * sw {
                    for ch in 0..c {
                        let t = o + drop;
                        let (d, i, yy, j, xx, k) = (t / st, t % st, oy / sh, oy % sh, ox / sw, ox % sw);
                        let want = x[((d * h + yy) * w + xx) * c * vol + ch * vol + i * sh * sw + j * sw + k];
                        assert_eq!(got[((o * h * sh + oy) * w * sw + ox) * c + ch], want, "depth to space ({st}, {sh}, {sw}) at ({o}, {oy}, {ox}) {ch}");
                    }
                }
            }
        }
    }
}

/// A UNet's ops are their formulas: a group norm of an image's rows (its groups' means and variances over every
/// pixel, a large offset on a group kept), with and without its SiLU; GEGLU's gate times the exact GELU; and the
/// even pixels of an image.
#[test]
fn a_unets_ops_are_their_formulas() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut next = rng(77);
    // the group norm: 300 pixels (two chunks of 256, the second short) of 64 channels in 32 groups
    for (pixels, c, groups) in [(300usize, 64usize, 32usize), (16usize, 320, 32), (1024, 96, 8)] {
        let cpg = c / groups;
        // (a group's values about an offset of its own, one of them large)
        let x: Vec<f32> = (0..pixels * c).map(|i| next() * (1.0 + (i % c / cpg) as f32 * 0.1) + if (i % c) / cpg == 3 { 250.0 } else { (i % c / cpg) as f32 * 0.5 }).collect();
        let (weight, bias): (Vec<f32>, Vec<f32>) = ((0..c).map(|_| 1.0 + 0.3 * next()).collect(), (0..c).map(|_| 0.2 * next()).collect());
        for silu in [false, true] {
            let eps = 1e-5f32;
            let (xv, wv, bv, ov, sv) = (b.vec(pixels * c), b.vec(c), b.vec(c), b.vec(pixels * c), b.vec(groups * (pixels.div_ceil(256) + 1) * 2));
            DeviceChain::upload(&b, &xv, &x);
            DeviceChain::upload(&b, &wv, &weight);
            DeviceChain::upload(&b, &bv, &bias);
            let mut rec = Recorder::new(&b);
            rec.group_norm_rows(&xv, &wv, &bv, &ov, &sv, pixels, c, groups, eps, silu);
            rec.read(&ov);
            let got = Box::new(rec).finish().pop().unwrap();
            let mut worst = 0f64;
            for g in 0..groups {
                let values: Vec<f64> = (0..pixels).flat_map(|px| (0..cpg).map(move |i| (px, i))).map(|(px, i)| x[px * c + g * cpg + i] as f64).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                for px in 0..pixels {
                    for i in 0..cpg {
                        let ch = g * cpg + i;
                        let v = (x[px * c + ch] as f64 - mean) / (var + eps as f64).sqrt() * weight[ch] as f64 + bias[ch] as f64;
                        let want = if silu { v / (1.0 + (-v).exp()) } else { v };
                        worst = worst.max((got[px * c + ch] as f64 - want).abs());
                    }
                }
            }
            eprintln!("a group norm of {pixels} pixels of {c} in {groups} groups{}: the worst error {worst:.2e}", if silu { ", through SiLU" } else { "" });
            assert!(worst < 2e-4, "{pixels} pixels of {c}: {worst}");
        }
    }
    // GEGLU
    let (rows, ff) = (7usize, 300usize);
    let fused: Vec<f32> = (0..rows * 2 * ff).map(|_| 3.0 * next()).collect();
    let (fv, ov) = (b.vec(rows * 2 * ff), b.vec(rows * ff));
    DeviceChain::upload(&b, &fv, &fused);
    let mut rec = Recorder::new(&b);
    rec.geglu_rows(&fv, &ov, rows, ff);
    rec.read(&ov);
    let got = Box::new(rec).finish().pop().unwrap();
    let erf = |v: f64| {
        // (Abramowitz and Stegun 7.1.26 is the kernel's; the series here to 1e-12)
        let (mut sum, mut term) = (v, v);
        for n in 1..60 {
            term *= -v * v / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    };
    let worst = (0..rows * ff).map(|i| {
        let (r, j) = (i / ff, i % ff);
        let (gate, value) = (fused[r * 2 * ff + j] as f64, fused[r * 2 * ff + ff + j] as f64);
        (got[i] as f64 - gate * 0.5 * value * (1.0 + erf(value / std::f64::consts::SQRT_2))).abs()
    }).fold(0f64, f64::max);
    eprintln!("GEGLU of {rows} rows of {ff}: the worst error {worst:.2e}");
    assert!(worst < 1e-5, "GEGLU: {worst}");
    // the even pixels
    let (h, w, c) = (6usize, 10usize, 5usize);
    let x: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
    let (xv, ov) = (b.vec(h * w * c), b.vec(h * w * c / 4));
    DeviceChain::upload(&b, &xv, &x);
    let mut rec = Recorder::new(&b);
    rec.subsample2x_even_rows(&xv, &ov, h, w, c);
    rec.read(&ov);
    let got = Box::new(rec).finish().pop().unwrap();
    for oy in 0..h / 2 {
        for ox in 0..w / 2 {
            for ch in 0..c {
                assert_eq!(got[(oy * (w / 2) + ox) * c + ch], x[(2 * oy * w + 2 * ox) * c + ch], "pixel ({oy}, {ox}), channel {ch}");
            }
        }
    }
}

/// Normalised attention guidance's mix is the reference's formula (`ltx::transformer::nag_mix`): rows whose
/// guided output is within tau of the plain one, rows scaled back to it, and alpha's blend.
#[test]
fn nag_mix_is_the_formula() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (rows, width) = (37usize, 1028usize);
    let mut next = rng(41);
    let pos: Vec<f32> = (0..rows * width).map(|_| next()).collect();
    // (every third row's negative far from it: its guided output past tau, scaled back)
    let neg: Vec<f32> = (0..rows * width).map(|i| if (i / width) % 3 == 0 { -3.0 * next() } else { pos[i] + 0.05 * next() }).collect();
    for (scale, tau, alpha) in [(11.0f32, 2.5f32, 0.25f32), (3.0, 2.5, 1.0), (1.0, 2.5, 1.0)] {
        let (pv, nv) = (b.vec(rows * width), b.vec(rows * width));
        DeviceChain::upload(&b, &pv, &pos);
        DeviceChain::upload(&b, &nv, &neg);
        let mut rec = Recorder::new(&b);
        rec.nag_mix(&pv, &nv, rows, width, scale, tau, alpha);
        rec.read(&pv);
        let got = Box::new(rec).finish().pop().unwrap();
        let (mut worst, mut scaled) = (0f32, 0);
        for r in 0..rows {
            let (p, n) = (&pos[r * width..(r + 1) * width], &neg[r * width..(r + 1) * width]);
            let guided: Vec<f32> = p.iter().zip(n).map(|(a, b)| a * scale - b * (scale - 1.0)).collect();
            let l1 = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
            let factor = (tau as f64 * (l1(p) + 1e-6) / l1(&guided)).clamp(0.0, 1.0) as f32;
            scaled += (factor < 1.0) as usize;
            for c in 0..width {
                let want = guided[c] * factor * alpha + p[c] * (1.0 - alpha);
                worst = worst.max((got[r * width + c] - want).abs() / want.abs().max(1.0));
            }
        }
        eprintln!("scale {scale}, tau {tau}, alpha {alpha}: {scaled} of {rows} rows scaled back, the worst error {worst:.2e}");
        assert!(worst < 2e-5, "scale {scale}: {worst}");
        assert!(scale == 1.0 || scaled > 0, "some rows are scaled back");
    }
}

/// SnakeBeta without aliasing is PyTorch's steps written out on the host: the steps padded by their ends, a transposed
/// convolution of stride 2 (times 2, cropped to twice the steps), SnakeBeta, the ends padded again, the low-pass at
/// stride 2; twelve taps each as BigVGAN's, and an odd low-pass; one step, a few, and more than a workgroup's.
#[test]
fn snake_beta_without_aliasing_is_the_steps_written_out() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut next = rng(91);
    for (len, c, ku, kd) in [(1usize, 4usize, 12usize, 12usize), (7, 3, 12, 12), (300, 8, 12, 12), (40, 5, 8, 7)] {
        let x: Vec<f32> = (0..len * c).map(|_| next()).collect();
        let filters: Vec<f32> = (0..ku + kd).map(|_| 0.2 * next()).collect();
        let freq: Vec<f32> = (0..c).map(|_| 1.0 + next().abs() * 3.0).collect();
        let scale: Vec<f32> = (0..c).map(|_| 0.2 + next().abs()).collect();
        // the host's
        let mut want = vec![0f32; len * c];
        for ch in 0..c {
            let pad = ku / 2 - 1;
            let xp: Vec<f64> = (0..len + 2 * pad).map(|i| x[(i as isize - pad as isize).clamp(0, len as isize - 1) as usize * c + ch] as f64).collect();
            let mut full = vec![0f64; (xp.len() - 1) * 2 + ku];
            for (i, v) in xp.iter().enumerate() {
                for j in 0..ku {
                    full[2 * i + j] += v * filters[j] as f64;
                }
            }
            let (left, right) = (pad * 2 + (ku - 2) / 2, pad * 2 + (ku - 1) / 2);
            let up: Vec<f64> = full[left..full.len() - right].iter().map(|v| 2.0 * v).collect();
            assert_eq!(up.len(), 2 * len);
            let z: Vec<f64> = up.iter().map(|v| v + scale[ch] as f64 * (freq[ch] as f64 * v).sin().powi(2)).collect();
            let (l, r) = (kd / 2 - usize::from(kd % 2 == 0), kd / 2);
            let zp: Vec<f64> = (0..z.len() + l + r).map(|i| z[(i as isize - l as isize).clamp(0, z.len() as isize - 1) as usize]).collect();
            for n in 0..len {
                want[n * c + ch] = (0..kd).map(|j| zp[2 * n + j] * filters[ku + j] as f64).sum::<f64>() as f32;
            }
        }
        let (xd, fd, qd, sd, mid, y) = (b.vec(len * c), b.vec(ku + kd), b.vec(c), b.vec(c), b.vec(2 * len * c), b.vec(len * c));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &fd, &filters);
        DeviceChain::upload(&b, &qd, &freq);
        DeviceChain::upload(&b, &sd, &scale);
        let mut rec = b.begin();
        rec.snake_beta_alias_rows(&xd, &fd, ku, kd, &qd, &sd, len, c, &mid, &y);
        rec.read(&y);
        let got = rec.finish().pop().unwrap();
        let big = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let worst = got.iter().zip(&want).map(|(a, e)| (a - e).abs()).fold(0f32, f32::max);
        assert!(big > 0.0 && worst <= 2e-5 * big.max(1.0), "{len} steps of {c}, {ku} and {kd} taps: {worst} off of a largest {big}");
    }
}

/// A convolution's weights packed from their F16 bytes are the words the same values give as f32: 1x1 and 3x3,
/// channels a multiple of 32 and not; and an infinity among them is refused.
#[test]
fn a_convolutions_f16_bytes_pack_as_its_f32_values() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut next = rng(57);
    for (cout, cin, k) in [(5usize, 3usize, 3usize), (8, 32, 1), (4, 40, 3), (2, 64, 7)] {
        let halves: Vec<u16> = (0..cout * cin * k * k).map(|_| half::f16::from_f32(next() * 3.0).to_bits()).collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let values: Vec<f32> = halves.iter().map(|h| half::f16::from_bits(*h).to_f32()).collect();
        let (got, want) = (b.conv_weights_halves(&bytes, cout, cin, k).expect("finite halves"), DeviceChain::conv_weights(&b, &values, cout, cin, k).expect("values in f16's range"));
        assert_eq!(got.len, want.len, "{cout}x{cin}x{k}");
        let read = |v: &DeviceVec| {
            let mut rec = b.begin();
            rec.read(v);
            rec.finish().pop().unwrap().iter().map(|w| w.to_bits()).collect::<Vec<u32>>()
        };
        assert_eq!(read(&got), read(&want), "{cout}x{cin}x{k}");
        let mut bad = bytes.clone();
        (bad[2], bad[3]) = (0x00, 0x7c);
        assert!(b.conv_weights_halves(&bad, cout, cin, k).is_none(), "an infinity");
    }
}
