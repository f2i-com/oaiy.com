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
        if rows > 8 && b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
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
    let features = b.gpu.device.features();
    eprintln!("cooperative matrices {}, f16 {}", features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX), features.contains(wgpu::Features::SHADER_F16));
    if !features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16) {
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
        for m in 1..=crate::shaders::IQ4_FEW_MAX {
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
        for m in 1..=crate::shaders::IQ4_FEW_MAX {
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
    if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
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
    use crate::shaders::coop_splits;
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
    if b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
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
