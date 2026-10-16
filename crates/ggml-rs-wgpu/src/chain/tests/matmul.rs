//! The matmuls against the host's: f32 and f16 matrices, the tensor cores' kernels and their splits, W4A8, NVFP4, a LoRA's merge.
use super::*;

/// A prompt's f32 matmul (64x64 tiles, `k` split where the tiles are few) gives the CPU's sums: shapes off the tiles'
/// edges, a long `k` over few outputs (Qwen3.8-Flash-Next's hyper-connections' and router's), a short one over many;
/// twice, the second from the pool's scratch as the first left it.
#[test]
fn a_prompts_f32_matmul_tiles_and_splits_as_the_cpu() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(91);
    for (n, k, rows) in [(5usize, 37usize, 3usize), (70, 100, 65), (324, 10240, 70), (513, 2560, 129), (1030, 324, 64), (96, 5120, 4), (96, 5120, 2), (10, 64, 8)] {
        let w: Vec<f32> = (0..n * k).map(|_| r()).collect();
        let x: Vec<f32> = (0..rows * k).map(|_| r()).collect();
        let want: Vec<f32> = (0..rows).flat_map(|i| (0..n).map(|o| (0..k).map(|j| w[o * k + j] as f64 * x[i * k + j] as f64).sum::<f64>() as f32).collect::<Vec<_>>()).collect();
        let (wd, xd, yd) = (b.vec(n * k), b.vec(rows * k), b.vec(rows * n));
        DeviceChain::upload(&b, &wd, &w);
        DeviceChain::upload(&b, &xd, &x);
        for _ in 0..2 {
            let mut rec = b.begin();
            rec.keep_groups(false);
            rec.matmul_f32_rows(&wd, n, k, &xd, &yd, rows);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let scale = (k as f32).sqrt();
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert!((g - e).abs() <= 1e-4 * scale, "[{n}, {k}] of {rows} rows [{i}]: {g} against {e}");
            }
        }
    }
}

/// ComfyUI's W4A8 decoded on the device as the host decodes it: codes of a random codebook, FP8 group scales (a
/// subnormal among them), rows' scales; not rotated, and rotated in groups of 16 and 256.
#[test]
fn w4a8s_decode_is_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(113);
    let e4m3 = |v: u8| -> f32 {
        let (e, m) = (((v >> 3) & 15) as i32, (v & 7) as f32);
        let x = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
        if v & 128 != 0 { -x } else { x }
    };
    for (rows, cols, rotation) in [(5usize, 48usize, 0usize), (3, 64, 16), (70, 512, 256)] {
        let codes: Vec<u8> = (0..rows * cols / 2).map(|_| ((r() + 1.0) * 127.9) as u8).collect();
        // (scales up to e4m3's 2^3 or so, a subnormal at the first)
        let rel: Vec<u8> = (0..rows * cols / 16).map(|i| if i == 0 { 3 } else { 0x30 + ((r() + 1.0) * 15.9) as u8 }).collect();
        let channel: Vec<f32> = (0..rows).map(|_| r() * 0.01).collect();
        let book: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * (1.0 + 0.1 * r())).collect();
        // the host's: the values, then each group rotated
        let mut want: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let (row, col) = (i / cols, i % cols);
                let byte = codes[row * cols / 2 + col / 2];
                let code = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                (book[code as usize] * e4m3(rel[row * cols / 16 + col / 16])).round_ties_even().clamp(-127.0, 127.0) * channel[row]
            })
            .collect();
        if rotation > 0 {
            const H: [[f32; 4]; 4] = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
            let entry = |mut a: usize, mut c: usize| {
                let mut v = 1.0 / (rotation as f32).sqrt();
                while a != 0 || c != 0 {
                    v *= H[a % 4][c % 4];
                    a /= 4;
                    c /= 4;
                }
                v
            };
            for group in want.chunks_mut(rotation) {
                let x = group.to_vec();
                for (c, y) in group.iter_mut().enumerate() {
                    *y = (0..rotation).map(|a| x[a] * entry(a, c)).sum();
                }
            }
        }
        let words = |bytes: &[u8]| -> Vec<f32> { bytes.chunks(4).map(|c| { let mut w = [0u8; 4]; w[..c.len()].copy_from_slice(c); f32::from_bits(u32::from_le_bytes(w)) }).collect() };
        let (cw, rw) = (words(&codes), words(&rel));
        let (cd, rd, chd, bd, out) = (b.vec(cw.len()), b.vec(rw.len()), b.vec(rows), b.vec(16), b.vec(rows * cols / 2));
        DeviceChain::upload(&b, &cd, &cw);
        DeviceChain::upload(&b, &rd, &rw);
        DeviceChain::upload(&b, &chd, &channel);
        DeviceChain::upload(&b, &bd, &book);
        let mut rec = b.begin();
        rec.w4a8_f16(&cd, &rd, &chd, &bd, rows, cols, rotation, &out);
        rec.read(&out);
        let got: Vec<f32> = rec.finish().pop().unwrap().iter().flat_map(|w| [half::f16::from_bits(w.to_bits() as u16).to_f32(), half::f16::from_bits((w.to_bits() >> 16) as u16).to_f32()]).collect();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-3 * w.abs() + 1e-6, "{rows}x{cols} rotated {rotation}: [{i}] {g} against {w}");
        }
    }
}

/// An NVFP4 matmul on the tensor cores as the host computes it: E2M1 pairs high nibble first, an E4M3 scale a block
/// of 16 (every byte but the NaNs), the tensor's own scale and a bias after the sums; rows of a tile and not.
#[test]
fn an_nvfp4_matmul_is_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let e2m1 = [0f64, 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.];
    let e4m3 = |v: u8| -> f64 {
        let (s, e, m) = (if v & 0x80 != 0 { -1.0 } else { 1.0 }, ((v >> 3) & 15) as i32, (v & 7) as f64);
        s * if e == 0 { m / 8.0 * 2f64.powi(-6) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) }
    };
    let mut seed = 0x1234_5678u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for (n, k, rows, global) in [(160usize, 256usize, 130usize, 0.0123f32), (64, 2048, 9, 3.5), (300, 128, 128, 1.0)] {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| next() as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|_| loop { let v = (next() % 0x78) as u8 | ((next() & 1) as u8) << 7; if v & 0x7f != 0x7f { break v; } }).collect();
        let x: Vec<f32> = (0..rows * k).map(|_| half::f16::from_f32((next() % 2001) as f32 / 1000.0 - 1.0).to_f32()).collect();
        let bias: Vec<f32> = (0..n).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let Some((wd, sd)) = b.nvfp4_weights(&packed, &scales, global, n, k) else { return };
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(rows * n), b.vec(n));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.matmul_nvfp4_rows(&wd, &sd, &bd, n, k, &xd, &yd, rows);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        let weight = |o: usize, j: usize| -> f64 {
            let byte = packed[o * k / 2 + j / 2];
            let code = if j % 2 == 0 { byte >> 4 } else { byte & 15 };
            e2m1[code as usize] * e4m3(scales[o * k / 16 + j / 16]) * global as f64
        };
        for r in 0..rows {
            for o in 0..n {
                let want = bias[o] as f64 + (0..k).map(|j| weight(o, j) * x[r * k + j] as f64).sum::<f64>();
                let mag: f64 = (0..k).map(|j| (weight(o, j) * x[r * k + j] as f64).abs()).sum();
                let g = got[r * n + o] as f64;
                assert!((g - want).abs() <= 1e-5 * mag + 1e-5, "[{n}, {k}] of {rows} rows, row {r} output {o}: {g} against {want}");
            }
        }
    }
}

/// A LoRA's merge on the device: `B A` by the f16 matmul (`A` transposed its weight, `B`'s rows its tokens) added
/// into an f16 matrix, each sum rounded to f16, as the host's merge rounds it.
#[test]
fn a_lora_merges_into_an_f16_matrix_as_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(71);
    let (n, k, rank) = (40usize, 96usize, 32usize);
    let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r() * 0.1).to_f32()).collect();
    let a: Vec<f32> = (0..rank * k).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
    let bm: Vec<f32> = (0..n * rank).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
    let at: Vec<f32> = (0..k * rank).map(|i| a[(i % rank) * k + i / rank]).collect();
    let (Some(wd), Some(ad)) = (b.vec_f16(&w), b.vec_f16(&at)) else { return };
    let (bd, dd) = (b.vec(bm.len()), b.vec(n * k));
    DeviceChain::upload(&b, &bd, &bm);
    let mut rec = b.begin();
    rec.matmul_f16_rows(&ad, k, rank, &bd, &dd, n);
    rec.add_f16(&wd, &dd, n * k);
    rec.read(&wd);
    let words = rec.finish().pop().unwrap();
    for i in 0..n {
        for j in 0..k {
            let delta: f64 = (0..rank).map(|q| bm[i * rank + q] as f64 * a[q * k + j] as f64).sum();
            let want = half::f16::from_f64(w[i * k + j] as f64 + delta).to_f64();
            let word = words[(i * k + j) / 2].to_bits();
            let got = half::f16::from_bits(if (i * k + j) % 2 == 0 { word as u16 } else { (word >> 16) as u16 }).to_f64();
            assert!((got - want).abs() <= 2e-3 * want.abs() + 1e-4, "({i}, {j}): {got} against {want}");
        }
    }
}

/// A matrix of f16 values held as f16 (two to a word) multiplies as it does held as f32: a step's row and a prompt's
/// within rounding (the f16 kernel sums an output's products by its lanes; a prompt's on the tensor cores within
/// f16's: its tokens f16, its sums f16 a window); a check's few rows are each a step's bit for bit (summed the same
/// way whatever the rows); a matrix not all f16 values is not made.
#[test]
fn an_f16_matrix_multiplies_as_its_f32_one() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(57);
    assert!(b.vec_f16(&[0.1, 0.5]).is_none(), "0.1 is no f16");
    for (n, k, rows) in [(324usize, 10240usize, 1usize), (513, 2560, 1), (1030, 324, 1), (70, 100, 1), (324, 10240, 70), (1030, 324, 65), (513, 2560, 300), (96, 2560, 512), (10240, 324, 130)] {
        let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r()).to_f32()).collect();
        let x: Vec<f32> = (0..rows * k).map(|_| r()).collect();
        let (w32, xd, y32, y16) = (b.vec(n * k), b.vec(rows * k), b.vec(rows * n), b.vec(rows * n));
        DeviceChain::upload(&b, &w32, &w);
        DeviceChain::upload(&b, &xd, &x);
        let w16 = b.vec_f16(&w).expect("f16 values");
        let mut rec = b.begin();
        rec.matmul_f32_rows(&w32, n, k, &xd, &y32, rows);
        rec.matmul_f16_rows(&w16, n, k, &xd, &y16, rows);
        rec.read(&y32);
        rec.read(&y16);
        let mut got = rec.finish();
        let (h, f) = (got.pop().unwrap(), got.pop().unwrap());
        if rows == 1 {
            // a check of three rows, the step's row its last: that row's sums the step's own
            let x3: Vec<f32> = (0..2 * k).map(|_| r()).chain(x.iter().copied()).collect();
            let (x3d, y3) = (b.vec(3 * k), b.vec(3 * n));
            DeviceChain::upload(&b, &x3d, &x3);
            let mut rec = b.begin();
            rec.matmul_f16_rows(&w16, n, k, &x3d, &y3, 3);
            rec.read(&y3);
            let three = rec.finish().pop().unwrap();
            assert_eq!(three[2 * n..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "[{n}, {k}]: a check's row is a step's");
        }
        if rows > 8 && b.gpu.coop16() {
            // on the tensor cores: the tokens rounded to f16, the sums f16 a window
            let rms = (f.iter().map(|e| (*e as f64).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
            let err = (h.iter().zip(&f).map(|(a, e)| ((*a - *e) as f64).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
            let worst = h.iter().zip(&f).map(|(a, e)| ((*a - *e) as f64).abs()).fold(0.0, f64::max);
            assert!(err < 2e-3 * rms && worst < 1.5e-2 * rms, "[{n}, {k}] of {rows} rows: RMS error {err:.3e}, the worst {worst:.3e}, of an RMS {rms:.3}");
        } else {
            let scale = (k as f32).sqrt();
            for (i, (a, e)) in h.iter().zip(&f).enumerate() {
                assert!((a - e).abs() <= 1e-4 * scale, "[{n}, {k}] of {rows} rows [{i}]: {a} against {e}");
            }
        }
    }
}

/// The tensor cores through WGSL's cooperative matrices (where the adapter has them): one subgroup's 16x16x16 f16
/// multiply into an f32 accumulator, the host's sums; the configurations the adapter reports, printed.
#[test]
fn a_cooperative_matrix_multiplies_as_the_host() {
    use ggml_rs::ChainRecorder;
    let b = match WgpuBackend::new(Some(1 << 30)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no adapter: {e}");
            return;
        }
    };
    eprintln!("cooperative matrices of {}, f16 {}", b.gpu.coop_tile(), b.gpu.device.features().contains(wgpu::Features::SHADER_F16));
    // (its probe is a 16 x 16 fragment's)
    if !b.gpu.coop16() {
        return;
    }
    const PROBE: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> bm: array<f16>;
@group(0) @binding(6) var<storage, read_write> c: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(32)
fn main() {
    let ma = coopLoadT<coop_mat16x16<f16, A>>(&a[0], 16u);
    let mb = coopLoadT<coop_mat16x16<f16, B>>(&bm[0], 16u);
    var mc = coop_mat16x16<f32, C>();
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(mc, &c[0], 16u);
}
"#;
    let a: Vec<f32> = (0..256).map(|i| ((i * 7 % 23) as f32 - 11.0) / 8.0).collect();
    let bv: Vec<f32> = (0..256).map(|i| ((i * 5 % 19) as f32 - 9.0) / 4.0).collect();
    let pack = |v: &[f32]| -> Vec<f32> { v.chunks_exact(2).map(|p| f32::from_bits(half::f16::from_f32(p[0]).to_bits() as u32 | (half::f16::from_f32(p[1]).to_bits() as u32) << 16)).collect() };
    let up = |v: &[f32]| {
        let x = b.vec(v.len());
        DeviceChain::upload(&b, &x, v);
        x
    };
    let (ad, bd, cd) = (up(&pack(&a)), up(&pack(&bv)), b.vec(256));
    let mut rec = Recorder::new(&b);
    let d = rec.gpu().dummy().clone();
    let drw = rec.gpu().dummy_rw().clone();
    rec.dispatch_wide("test-coop", PROBE, [buffer(&ad), buffer(&bd), &d, &d, &d, &d, buffer(&cd), &drw], &[0], (1, 1, 1));
    rec.read(&cd);
    let got = Box::new(rec).finish().pop().unwrap();
    for i in 0..16 {
        for j in 0..16 {
            let want: f32 = (0..16).map(|k| a[i * 16 + k] * bv[k * 16 + j]).sum();
            assert!((got[i * 16 + j] - want).abs() < 1e-3, "c[{i}, {j}]: {} against {want}", got[i * 16 + j]);
        }
    }
}

/// IQ4_XS's kernel for a step's row and a check's few gives the generic f32 kernel's sums (the same weights and the
/// same f32 products, added in another order): every count of rows it takes, weight rows off a workgroup's eight, a
/// width of one lane's turn and of three, every six-bit scale and every nibble among the blocks.
#[test]
fn iq4_xs_against_a_few_rows_is_the_generic_kernels() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    for (n, k) in [(37usize, 512usize), (64, 2560)] {
        let mut next = rng(n as u32 + k as u32);
        let mut raw = vec![0u8; n * (k / 256) * 136];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 127.9) as u8;
        }
        for blk in raw.chunks_exact_mut(136) {
            let d = half::f16::from_f32(0.001 + (blk[9] as f32) * 1e-5).to_bits().to_le_bytes();
            (blk[0], blk[1]) = (d[0], d[1]);
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::IQ4_XS));
        for m in 1..=crate::quant::shaders::IQ4_FEW_MAX {
            let (x, y, yf) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
            let mut rec = Recorder::new(&b);
            rec.matmul_rows_f32(&w, &x, &y, m);
            rec.read(&y);
            let want = Box::new(rec).finish().pop().unwrap();
            let mut rec = Recorder::new(&b);
            assert!(rec.matmul_rows_iq4_xs(&w, &x, &yf, m), "IQ4_XS [{n}, {k}] of {m} rows");
            rec.read(&yf);
            let got = Box::new(rec).finish().pop().unwrap();
            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let worst = got.iter().zip(&want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            assert!(scale > 0.0 && worst <= scale * 2e-5, "IQ4_XS [{n}, {k}] of {m} rows: {worst} off of a largest {scale}");
        }
    }
}

/// IQ4_XS's kernel from int8 activations (a check of drafts' rows) gives the f32 one's sums within the rounding of a
/// row's values to int8 (each 32 by its own scale): every count of rows, weight rows off a workgroup's eight, a width
/// of one lane's turn and of three; and exactly what the weights give against rows that are int8 already (every value
/// a whole number up to 127: the int8 kernel's sums are then the f32 kernel's but for the order they are added in).
#[test]
fn iq4_xs_from_int8_rows_is_the_f32_kernels() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    for (n, k) in [(37usize, 512usize), (64, 2560)] {
        let mut next = rng(n as u32 + 3 * k as u32);
        let mut raw = vec![0u8; n * (k / 256) * 136];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 127.9) as u8;
        }
        for blk in raw.chunks_exact_mut(136) {
            let d = half::f16::from_f32(0.001 + (blk[9] as f32) * 1e-5).to_bits().to_le_bytes();
            (blk[0], blk[1]) = (d[0], d[1]);
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::IQ4_XS));
        for m in 1..=crate::quant::shaders::IQ4_FEW_MAX {
            for whole in [false, true] {
                let (x, y, yq) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
                // (whole: each 32 has a 127 or a -127, so its scale is one and its values are their own int8)
                let xs: Vec<f32> = (0..m * k).map(|i| if !whole { next() } else if i % 32 == 5 { if next() > 0.0 { 127.0 } else { -127.0 } } else { (next() * 127.0).round() }).collect();
                DeviceChain::upload(&b, &x, &xs);
                let mut rec = Recorder::new(&b);
                assert!(rec.matmul_rows_iq4_xs(&w, &x, &y, m));
                rec.read(&y);
                let want = Box::new(rec).finish().pop().unwrap();
                let mut rec = Recorder::new(&b);
                assert!(rec.matmul_rows_iq4_xs_q8(&w, &x, &yq, m), "IQ4_XS [{n}, {k}] of {m} rows from int8");
                rec.read(&yq);
                let got = Box::new(rec).finish().pop().unwrap();
                let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
                let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (norm(&got) * norm(&want));
                let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let worst = got.iter().zip(&want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
                if whole {
                    assert!(scale > 0.0 && worst <= scale * 2e-5, "IQ4_XS [{n}, {k}] of {m} whole rows: {worst} off of a largest {scale}");
                } else {
                    assert!(cos > 0.9999, "IQ4_XS [{n}, {k}] of {m} rows from int8: cosine {cos}");
                }
            }
        }
    }
}

/// The tensor-core matmuls (where the device has them) give their f32 tiled kernels' sums within f16's rounding:
/// Q3_K, Q4_K, Q5_K, Q6_K and Q8_0, a tile's worth of tokens and a tile and a bit (the edge), rows off the tile.
#[test]
fn the_tensor_core_matmuls_are_the_f32_ones() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    // (Metal's 8 x 8 kernels as well as the 16 x 16 ones)
    if b.gpu.coop_tile() == 0 {
        return;
    }
    // the f16 scales' places in each type's block (d, and dmin where it has one)
    for (dtype, bytes, scales) in [(GgmlType::Q3_K, 110usize, &[108usize][..]), (GgmlType::Q4_K, 144, &[0, 2][..]), (GgmlType::Q5_K, 176, &[0, 2][..]), (GgmlType::Q6_K, 210, &[208][..]), (GgmlType::Q8_0, 34, &[0][..])] {
        for k in [512usize, 2048] {
        let n = 200usize;
        let mut next = rng(n as u32 + bytes as u32 + k as u32);
        let mut raw = vec![0u8; n * (k / if dtype == GgmlType::Q8_0 { 32 } else { 256 }) * bytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        for blk in raw.chunks_exact_mut(bytes) {
            for &at in scales {
                let d = half::f16::from_f32(0.01 + (blk[(at + 4) % bytes] as f32) * 1e-4).to_bits().to_le_bytes();
                blk[at] = d[0];
                blk[at + 1] = d[1];
            }
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
        for m in [128usize, 150] {
            let (x, y, yc) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
            // the f32 tiled kernel's sums (the int8 and tensor-core paths bypassed)
            let mut rec = Recorder::new(&b);
            rec.matmul_rows_f32(&w, &x, &y, m);
            rec.read(&y);
            let want = Box::new(rec).finish().pop().unwrap();
            let mut rec = Recorder::new(&b);
            assert!(rec.matmul_rows_coop(&w, &x, &yc, m), "{dtype:?} on the tensor cores");
            rec.read(&yc);
            let got = Box::new(rec).finish().pop().unwrap();
            let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got) * norm(&want));
            assert!(cos > 0.99999, "{dtype:?} [{n}, {k}] of {m}: cosine {cos}");
        }
        }
    }
}

/// The GPU's count of its units (SMs) is some (`--nocapture` shows it), and a matmul's split along k is the one its
/// waves of a workgroup a unit are fullest at for what the splits' sums cost: Qwen3.8 27B's matmuls of 512 tokens
/// on an RTX 5090's 170 SMs (as [`measure_coop_splits`] finds them), and none empty.
#[test]
fn a_tensor_core_matmul_is_split_to_fill_the_gpu() {
    use crate::quant::shaders::coop_splits;
    // its FFN's gate and up (1088 tiles, k 5120 in 160 steps), down (160 tiles, k 17408), a delta net's gate (192),
    // attention's q (384), k and v (32 each), output (160, k 6144)
    assert_eq!(coop_splits(1088, 170, 160), 1);
    assert_eq!(coop_splits(160, 170, 544), 1);
    assert_eq!(coop_splits(192, 170, 160), 3);
    assert_eq!(coop_splits(384, 170, 160), 2);
    assert_eq!(coop_splits(32, 170, 160), 5);
    assert_eq!(coop_splits(160, 170, 192), 1);
    // splits of 8 steps or more
    assert_eq!(coop_splits(1, 170, 32), 4);
    assert_eq!(coop_splits(1, 170, 18), 2);
    assert_eq!(coop_splits(1, 170, 15), 1);
    assert_eq!(coop_splits(1, 1, 160), 1);
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if b.gpu.coop_tile() != 0 {
        let units = b.gpu.coop_units();
        eprintln!("{units} units (SMs)");
        assert!(units >= 1);
    }
}

/// A few rows of an f16 matrix (a check of drafts) give each row's one-row sums bit for bit: long rows (the
/// hyper-connections' down matrices, the router) and short ones (their up matrices), 2 to 8 rows.
#[test]
fn a_few_rows_of_an_f16_matrix_are_each_row_alone() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut r = rng(61);
    for (n, k) in [(324usize, 10240usize), (513, 2560), (10240, 324), (70, 100)] {
        let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r()).to_f32()).collect();
        let w16 = b.vec_f16(&w).expect("f16 values");
        for rows in 2..=8usize {
            let x: Vec<f32> = (0..rows * k).map(|_| r()).collect();
            let (xd, yd) = (b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &xd, &x);
            let mut rec = b.begin();
            rec.matmul_f16_rows(&w16, n, k, &xd, &yd, rows);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let mut want = Vec::new();
            for row in x.chunks_exact(k) {
                let (x1, y1) = (b.vec(k), b.vec(n));
                DeviceChain::upload(&b, &x1, row);
                let mut rec = b.begin();
                rec.matmul_f16_rows(&w16, n, k, &x1, &y1, 1);
                rec.read(&y1);
                want.extend(rec.finish().pop().unwrap());
            }
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "[{n}, {k}] of {rows} rows");
        }
    }
}

/// A BF16 matrix from its bytes gives what the same values as f16 give, bit for bit (the same kernels, the weights
/// widened by a shift where two f16 were unpacked): a step's row and a check's few (the lanes' kernel) and a prompt's
/// rows (the tiled one, split along k and not). A few rows of a width that is not the lanes' kernel's take the tiled
/// kernel as BF16 and the narrow one as f16, another order of sums: close there.
#[test]
fn bf16_matmuls_are_the_f16_ones() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut next = rng(61);
    for (n, k, rows) in [(24usize, 64usize, 1usize), (24, 64, 5), (40, 2560, 8), (300, 1024, 40), (70, 38, 3), (70, 38, 100)] {
        // values BF16 holds (an f32's upper sixteen bits), which f16 holds too at this size
        let values: Vec<f32> = (0..n * k).map(|_| f32::from_bits((0.05 * next()).to_bits() & 0xffff_0000)).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let (w16, wb) = (DeviceChain::vec_f16(&b, &values).expect("f16 values"), b.vec_of_bytes(&bytes));
        let x: Vec<f32> = (0..rows * k).map(|_| next()).collect();
        let (xd, y16, yb) = (b.vec(rows * k), b.vec(rows * n), b.vec(rows * n));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.matmul_f16_rows_f32(&w16, n, k, &xd, &y16, rows);
        rec.matmul_bf16_rows_f32(&wb, n, k, &xd, &yb, rows);
        rec.read(&y16);
        rec.read(&yb);
        let got = rec.finish();
        assert!(got[0].iter().any(|v| *v != 0.0), "values to compare");
        if rows > 8 || k % 4 == 0 {
            assert_eq!(got[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "[{n}, {k}] of {rows} rows");
        } else {
            close(&got[1], &got[0], "a few rows of a narrow BF16 matrix");
        }
    }
}

/// Rows quantized to int8 by a subgroup's lanes together ([`crate::quant::shaders::QUANT_Q8_LANES`], where subgroups are 32
/// lanes) are a thread a block's ([`crate::quant::shaders::QUANT_Q8`]) bit for bit: their words, scales and halves' sums, rows
/// of 4,096 and 12,288 and a few of 96 (a last subgroup part full), a block of zeros and channels a hundred times the
/// rest among them.
#[test]
fn rows_quantized_by_lanes_are_a_thread_a_blocks() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !b.gpu.subgroups32() {
        return;
    }
    let mut next = rng(29);
    for (m, k) in [(1usize, 4096usize), (1, 12288), (3, 96), (5, 4096)] {
        let mut x: Vec<f32> = (0..m * k).map(|i| next() * if i % 997 == 3 { 100.0 } else { 1.0 }).collect();
        x[32..64].fill(0.0);
        let xv = b.vec(m * k);
        DeviceChain::upload(&b, &xv, &x);
        let (len, xs_at) = crate::quant::shaders::q8_len(m, k);
        let blocks = (m * k / 32) as u32;
        let mut got = Vec::new();
        for (name, src, groups) in [("test-q8-thread", crate::quant::shaders::QUANT_Q8, blocks.div_ceil(256)), ("test-q8-lanes", crate::quant::shaders::QUANT_Q8_LANES, (blocks * 8).div_ceil(256))] {
            let out = b.vec(len);
            DeviceChain::upload(&b, &out, &vec![f32::from_bits(0xdead_beef); len]);
            let pipeline = b.gpu.named_pipeline(name, || src.to_string());
            let mut rec = Recorder::new(&b);
            let d = rec.gpu().dummy().clone();
            rec.dispatch_kept(&pipeline, &d, buffer(&xv), buffer(&out), &[k as u32, m as u32, xs_at as u32], (groups, 1, 1));
            rec.read(&out);
            got.push(Box::new(rec).finish().pop().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
        }
        assert_eq!(got[1], got[0], "{m} rows of {k}: the lanes' words are the thread's");
    }
}

/// Q8_0's one-row kernel from int8 activations (where a step's row takes it) gives the f32 one's sums within the
/// rounding of the row's values to int8, and exactly against a row that is int8 already (every value a whole number up
/// to 127: the sums then the f32 kernel's but for their order): Qwen3.5 9B's delta nets' [64, 4096], a matrix of rows
/// off a workgroup's 4, and a row of one lane's turn.
#[test]
fn q8_0_from_an_int8_row_is_the_f32_kernels() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !b.gpu.int8_rows() {
        return;
    }
    for (n, k) in [(64usize, 4096usize), (37, 2048), (5, 1024)] {
        let mut next = rng(n as u32 + 7 * k as u32);
        let mut raw = vec![0u8; n * (k / 32) * 34];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 127.9) as u8;
        }
        for blk in raw.chunks_exact_mut(34) {
            let d = half::f16::from_f32(0.001 + (blk[9] as f32) * 1e-5).to_bits().to_le_bytes();
            (blk[0], blk[1]) = (d[0], d[1]);
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q8_0));
        // (a few rows' kernel too: 2 to 16 rows against the f32 kernel's, past a workgroup's 8)
        for m in [2usize, 3, 8, 9, 13, 16] {
            let (x, y, yq) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            let xs: Vec<f32> = (0..m * k).map(|_| next()).collect();
            DeviceChain::upload(&b, &x, &xs);
            let mut rec = Recorder::new(&b);
            rec.matmul_rows_f32(&w, &x, &y, m);
            rec.read(&y);
            let want = Box::new(rec).finish().pop().unwrap();
            let mut rec = Recorder::new(&b);
            assert!(rec.matmul_rows_q8(&w, &x, &yq, m), "Q8_0 [{n}, {k}] of {m} rows from int8");
            rec.read(&yq);
            let got = Box::new(rec).finish().pop().unwrap();
            for r in 0..m {
                let (g, e) = (&got[r * n..(r + 1) * n], &want[r * n..(r + 1) * n]);
                let dot: f64 = g.iter().zip(e).map(|(a, e)| *a as f64 * *e as f64).sum();
                let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (norm(g) * norm(e));
                assert!(cos > 0.9999, "Q8_0 [{n}, {k}] row {r} of {m} from int8: cosine {cos}");
            }
        }
        for whole in [false, true] {
            let (x, y, yq) = (b.vec(k), b.vec(n), b.vec(n));
            let xs: Vec<f32> = (0..k).map(|i| if !whole { next() } else if i % 32 == 5 { if next() > 0.0 { 127.0 } else { -127.0 } } else { (next() * 127.0).round() }).collect();
            DeviceChain::upload(&b, &x, &xs);
            let mut rec = Recorder::new(&b);
            rec.matmul_rows_f32(&w, &x, &y, 1);
            rec.read(&y);
            let want = Box::new(rec).finish().pop().unwrap();
            let mut rec = Recorder::new(&b);
            assert!(rec.matmul_rows_q8(&w, &x, &yq, 1), "Q8_0 [{n}, {k}] from int8");
            rec.read(&yq);
            let got = Box::new(rec).finish().pop().unwrap();
            let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got) * norm(&want));
            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let worst = got.iter().zip(&want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            if whole {
                assert!(scale > 0.0 && worst <= scale * 2e-5, "Q8_0 [{n}, {k}] of a whole row: {worst} off of a largest {scale}");
            } else {
                assert!(cos > 0.9999, "Q8_0 [{n}, {k}] from int8: cosine {cos}");
            }
        }
    }
}

/// A weight past the budget gives the card's own result ([`crate::quant::stream`]): from its file (its bytes written into
/// the slots before each matmul, the default) and from the host's memory (read there for a step's rows, copied into
/// the slots for a prompt's), for a step's row, a few rows and a prompt's, Q3_K's blocks padded on the card among
/// the types.
#[test]
fn a_weight_past_the_budget_is_the_cards() {
    let Ok(card) = WgpuBackend::new(Some(1 << 30)) else { return };
    let Ok(small) = WgpuBackend::new(Some(1)) else { return };
    // (each type's scale fields small and finite, the rest of its bytes any)
    for (dtype, n, k, block, scales) in [
        (GgmlType::Q4_K, 300usize, 512usize, 144usize, &[0usize, 2][..]),
        (GgmlType::Q3_K, 70, 256, 110, &[108][..]),
        (GgmlType::Q6_K, 64, 768, 210, &[208][..]),
        (GgmlType::Q8_0, 100, 128, 34, &[0][..]),
    ] {
        let elems = if dtype == GgmlType::Q8_0 { 32 } else { 256 };
        let mut next = rng(n as u32 * 7 + k as u32);
        let mut raw: Vec<u8> = (0..n * (k / elems) * block).map(|_| ((next() + 1.0) * 127.9) as u8).collect();
        for blk in raw.chunks_exact_mut(block) {
            for &at in scales {
                let d = half::f16::from_f32(0.002 + (blk[(at + 5) % block] as f32) * 1e-5).to_bits().to_le_bytes();
                (blk[at], blk[at + 1]) = (d[0], d[1]);
            }
        }
        let host = || ggml_rs::QuantizedTensor::from_bytes_cpu(raw.clone(), vec![n, k], dtype);
        let on_card = ggml_rs::Backend::to_device_quant(&card, host());
        let from_file = ggml_rs::Backend::to_device_quant(&small, host());
        assert!(card.holds(&on_card) && small.holds(&from_file), "{dtype:?}: both held");
        let q = from_file.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a streamed weight");
        assert!(q.file.is_some() && small.filed() >= q.nbytes as u64 && small.usage().0 == 0, "{dtype:?}: past the budget, in its file");
        let bytes = crate::quant::stream::card_bytes(&host(), crate::quant::shaders::padded_block(dtype)).into_owned();
        let row_bytes = bytes.len() / n;
        let pinned = small.pin_to_host(&bytes, row_bytes, dtype, &[n, k]);
        for m in [1usize, 3, 40] {
            let xs: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let run = |b: &WgpuBackend, w: &ggml_rs::QuantizedTensor| {
                let (x, y) = (b.vec(m * k), b.vec(m * n));
                DeviceChain::upload(b, &x, &xs);
                let mut rec = Recorder::new(b);
                ChainRecorder::matmul_rows(&mut rec, w, &x, &y, m);
                rec.read(&y);
                Box::new(rec).finish().pop().unwrap()
            };
            // The same in a held recording, as Flash-Next's next card's layers are recorded: an op before the matmul
            // reads an input written once the recording is made, which the weight's write from its file must not
            // send ahead of it.
            let held = |b: &WgpuBackend, w: &ggml_rs::QuantizedTensor| {
                let (x_in, x, y) = (b.vec(m * k), b.vec(m * k), b.vec(m * n));
                DeviceChain::upload(b, &x_in, &vec![0.0; m * k]);
                let mut rec = Recorder::new(b);
                ChainRecorder::hold(&mut rec);
                ChainRecorder::copy(&mut rec, &x_in, 0, &x, 0, m * k);
                ChainRecorder::matmul_rows(&mut rec, w, &x, &y, m);
                DeviceChain::upload(b, &x_in, &xs);
                rec.read(&y);
                Box::new(rec).finish().pop().unwrap()
            };
            let want = run(&card, &on_card);
            let mut got = vec![("file", run(&small, &from_file)), ("file, held", held(&small, &from_file))];
            if let Some(p) = &pinned {
                got.push(("host", run(&small, p)));
            }
            for (tier, g) in got {
                let dot: f64 = g.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
                let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (norm(&g) * norm(&want));
                assert!(g.iter().all(|v| v.is_finite()) && cos > 0.9999, "{dtype:?} [{n}, {k}], {m} rows, from the {tier}: cosine {cos}");
            }
        }
    }
}

/// Q8_0 weights of a prompt's rows on the tensor cores give the f32 kernel's sums: Qwen Image 2.1's widths (3,072 in,
/// 3,072 and 12,288 out) at rows that split the matmul along `k` and rows that do not.
#[test]
fn q8_0_on_the_tensor_cores_is_the_f32_kernels() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !b.gpu.coop16() {
        return;
    }
    for (n, k) in [(3072usize, 3072usize), (12288, 3072)] {
        let mut next = rng((n + k) as u32);
        let xs32: Vec<f32> = (0..n * k).map(|_| next() * 0.05).collect();
        let mut raw = vec![0u8; n * k / 32 * 34];
        ggml_quants::q8_0::quantize(&xs32, &mut raw);
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q8_0));
        for m in [1usize, 2, 9, 14, 17, 24, 25, 768, 790, 3072, 4096, 4120] {
            let xs: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let (x, y, yf) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &x, &xs);
            let mut rec = Recorder::new(&b);
            ChainRecorder::matmul_rows(&mut rec, &w, &x, &y, m);
            rec.matmul_rows_f32(&w, &x, &yf, m);
            rec.read(&y);
            rec.read(&yf);
            let r = Box::new(rec).finish();
            let (got, want) = (&r[0], &r[1]);
            let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(got) * norm(want));
            eprintln!("Q8_0 [{n}, {k}] by {m} rows: cosine {cos:.6}");
            assert!(got.iter().all(|v| v.is_finite()) && cos > 0.9999, "Q8_0 [{n}, {k}] by {m} rows: cosine {cos}");
        }
    }
}

/// An f16 matrix of few outputs (a LoRA's down factor: 128 of them over 3,072) by a prompt's rows on the tensor cores
/// gives the exact kernel's sums, at rows few enough that the matmul is split along `k` and many enough that it is
/// not; and its up factor (3,072 or 12,288 outputs over 128).
#[test]
fn a_lora_factors_f16_matmul_is_the_exact_kernels() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    if !b.gpu.coop16() {
        return;
    }
    for (n, k) in [(128usize, 3072usize), (3072, 128), (12288, 128)] {
        let mut next = rng((n * 3 + k) as u32);
        let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(next() * 0.05).to_f32()).collect();
        let w16 = b.vec_f16(&w).expect("f16 values");
        for m in (1usize..=40).chain([300, 768, 1024, 3072, 4096]) {
            let xs: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let (x, y, yf) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &x, &xs);
            let mut rec = Recorder::new(&b);
            ChainRecorder::matmul_f16_rows(&mut rec, &w16, n, k, &x, &y, m);
            ChainRecorder::matmul_f16_rows_f32(&mut rec, &w16, n, k, &x, &yf, m);
            rec.read(&y);
            rec.read(&yf);
            let r = Box::new(rec).finish();
            let (got, want) = (&r[0], &r[1]);
            let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(got) * norm(want));
            eprintln!("f16 [{n}, {k}] by {m} rows: cosine {cos:.6}");
            assert!(got.iter().all(|v| v.is_finite()) && cos > 0.9999, "f16 [{n}, {k}] by {m} rows: cosine {cos}");
        }
    }
}

/// A few rows with a few channels far larger than the rest (a text encoder's states, as Qwen Image's prefix takes
/// them): what the few rows' kernels give against the f32 kernel's, Q8_0 and Q4_K.
#[test]
#[ignore = "a measurement: prints the cosines"]
fn measure_few_rows_with_large_channels() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (n, k) = (3072usize, 3072usize);
    let mut next = rng(77);
    let xs32: Vec<f32> = (0..n * k).map(|_| next() * 0.05).collect();
    let mut q8 = vec![0u8; n * k / 32 * 34];
    ggml_quants::q8_0::quantize(&xs32, &mut q8);
    let wq8 = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(q8, vec![n, k], GgmlType::Q8_0));
    for m in [14usize, 30] {
        let xs: Vec<f32> = (0..m * k).map(|i| next() * if i % k % 997 == 5 { 300.0 } else { 3.0 }).collect();
        let (x, y, yf) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
        DeviceChain::upload(&b, &x, &xs);
        let mut rec = Recorder::new(&b);
        ChainRecorder::matmul_rows(&mut rec, &wq8, &x, &y, m);
        rec.matmul_rows_f32(&wq8, &x, &yf, m);
        rec.read(&y);
        rec.read(&yf);
        let r = Box::new(rec).finish();
        let (got, want) = (&r[0], &r[1]);
        let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        eprintln!("Q8_0 by {m} rows with large channels: cosine {:.6}, finite {}", dot / (norm(got) * norm(want)), got.iter().all(|v| v.is_finite()));
    }
}

/// Q8_0 weights at Qwen Image 2.1's widths (4,096 and 12,288) give the host's sums (dequantized, in f64) for a text
/// prefix's few rows and a picture's many, whichever kernel takes them.
#[test]
fn q8_0_at_a_pictures_widths_is_the_hosts() {
    let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
    for (n, k) in [(4096usize, 4096usize), (12288, 4096), (4096, 12288)] {
        let mut next = rng((n * 7 + k) as u32);
        let w32: Vec<f32> = (0..n * k).map(|_| next() * 0.05).collect();
        let mut raw = vec![0u8; n * k / 32 * 34];
        ggml_quants::q8_0::quantize(&w32, &mut raw);
        let mut deq = vec![0f32; n * k];
        ggml_quants::dequantize(GgmlType::Q8_0, &raw, &mut deq).unwrap();
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q8_0));
        for m in [1usize, 5, 14, 24, 25, 40, 300] {
            let xs: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let (x, y) = (b.vec(m * k), b.vec(m * n));
            DeviceChain::upload(&b, &x, &xs);
            let mut rec = Recorder::new(&b);
            ChainRecorder::matmul_rows(&mut rec, &w, &x, &y, m);
            rec.read(&y);
            let got = Box::new(rec).finish().pop().unwrap();
            // a few of the outputs against the host's
            let mut worst = 0f64;
            let mut scale = 0f64;
            for r in [0, m / 2, m - 1] {
                for o in [0, 1, n / 3, n - 1] {
                    let want: f64 = (0..k).map(|i| deq[o * k + i] as f64 * xs[r * k + i] as f64).sum();
                    worst = worst.max((got[r * n + o] as f64 - want).abs());
                    scale = scale.max(want.abs());
                }
            }
            eprintln!("Q8_0 [{n}, {k}] by {m} rows: worst {worst:.4} of {scale:.3}");
            assert!(worst <= 2e-2 * scale.max(1.0), "Q8_0 [{n}, {k}] by {m} rows: {worst} off of {scale}");
        }
    }
}

/// A prompt's EXL3 rows recorded after scratch the device wrote and let go of are the rows recorded alone: the tables
/// the host writes as it records (the rows' order, their jobs) take none of that scratch, whose writes, recorded before
/// them, would run after the host's and over them (a vision tower's wide attention let go of its heads' sums, and
/// Qwen3.8-Flash-Next read every picture as noise). On the tensor cores and off them, a last lone row and none.
#[test]
fn a_prompts_exl3_tables_are_not_written_over_by_scratch_let_go_of() {
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let (k, n, tw) = (512usize, 384usize, 48usize);
    let mut s = 91u32;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    };
    let words = (0..k / 16 * n / 16 * (tw / 2)).map(|_| next()).collect();
    let mut sign = |len: usize, mag: f32| (0..len).map(|_| if next() & 1 == 0 { mag } else { -mag }).collect::<Vec<f32>>();
    let data = ggml_rs::exl3::Exl3Data { suh: sign(k, 0.25), svh: sign(n, 0.125), words, tile_words: tw, input_map: (0..k as u32).collect(), output_map: (0..n as u32).collect() };
    let w = b.exl3(data).unwrap();
    assert!(b.holds_exl3(w.as_ref()), "the adapter holds it");
    let zeros = b.vec(4096);
    DeviceChain::upload(&b, &zeros, &[0f32; 4096]);
    for rows in [70usize, 129] {
        let xs: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect();
        let (x, y) = (b.vec(rows * k), b.vec(rows * n));
        DeviceChain::upload(&b, &x, &xs);
        let run = |dirty: bool| {
            let mut rec = Recorder::new(&b);
            if dirty {
                // a scratch of each size these rows' tables take (256 bytes to 16 KiB), zeroed by the device, let go of
                for len in (6..=12).map(|p| 1usize << p) {
                    let v = rec.scratch(len);
                    rec.copy(&zeros, 0, &v, 0, len);
                    rec.done_with(&v);
                }
            }
            rec.exl3_rows(w.as_ref(), &x, &y, rows);
            rec.read(&y);
            Box::new(rec).finish().pop().unwrap()
        };
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        assert_eq!(bits(run(true)), bits(run(false)), "{rows} rows");
    }
}
