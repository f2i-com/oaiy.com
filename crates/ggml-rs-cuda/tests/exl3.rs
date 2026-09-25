//! Independent bit-by-bit packing oracle versus native CUDA EXL3 paths.
use ggml_rs::Backend;
use ggml_rs::{
    exl3::{mul1, Exl3Data, PackedLinear},
    Tensor,
};
use ggml_rs_cuda::{exl3::Exl3Matrix, CudaBackend};
use std::sync::Arc;

#[test]
#[ignore = "manual CUDA projection timing"]
fn benchmark_packed_decode_shapes() {
    let b = Arc::new(CudaBackend::new(0).unwrap());
    for (k, n, tw) in [
        (5120, 1024, 48),
        (5120, 6144, 48),
        (5120, 17408, 48),
        (17408, 5120, 56),
        (5120, 248320, 96),
    ] {
        let weights = Exl3Data {
            words: vec![0xdeadbeef; k * n / 256 * (tw / 2)],
            suh: vec![1.0; k],
            svh: vec![1.0; n],
            tile_words: tw,
            input_map: (0..k as u32).collect(),
            output_map: (0..n as u32).collect(),
        };
        let w = Exl3Matrix::upload(b.clone(), weights).unwrap();
        let x = b.to_device(Tensor::from_vec(vec![0.125; k], vec![1, k]));
        for splits in [1, 2, 4, 8, 16, 32] {
            for _ in 0..3 {
                let _ = w.linear_with_splits(&x, splits);
            }
            b.synchronize();
            let now = std::time::Instant::now();
            for _ in 0..50 {
                let _ = w.linear_with_splits(&x, splits);
            }
            b.synchronize();
            eprintln!(
                "EXL3 K={k} N={n} tw={tw} splits={splits}: {:.3} ms",
                now.elapsed().as_secs_f64() * 1000.0 / 50.0
            );
        }
    }
}

fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

#[test]
fn tiled_decode_matches_scalar_layout_at_model_width() {
    let Ok(b) = CudaBackend::new(0) else { return };
    let b = Arc::new(b);
    let (k, n) = (5120, 512);
    for tw in [32, 48, 56, 64, 96] {
        let mut seed = 13234567u32;
        let words = (0..k * n / 256 * (tw / 2))
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed
            })
            .collect();
        let weights = Exl3Data {
            words,
            suh: (0..k)
                .map(|i| if i % 3 == 0 { -0.25 } else { 0.25 })
                .collect(),
            svh: vec![0.125; n],
            tile_words: tw,
            input_map: (0..k as u32).rev().collect(),
            output_map: (0..n as u32).rev().collect(),
        };
        let w = Exl3Matrix::upload(b.clone(), weights).unwrap();
        let x = b.to_device(Tensor::from_vec(
            (0..k)
                .map(|i| ((i * 17 % 73) as f32 - 36.0) / 37.0)
                .collect(),
            vec![1, k],
        ));
        let expected = w.linear_reference(&x).to_host();
        for splits in [1, 2, 4, 8, 16, 32] {
            let actual = w.linear_with_splits(&x, splits).to_host();
            let error = actual
                .data()
                .iter()
                .zip(expected.data())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(error <= 0.003, "tw={tw} splits={splits}: {error}");
        }
    }
}
fn had(x: &mut [f32]) {
    for block in x.chunks_exact_mut(128) {
        for s in [1, 2, 4, 8, 16, 32, 64] {
            for i in 0..128 {
                if i & s == 0 {
                    let (a, b) = (block[i], block[i + s]);
                    block[i] = a + b;
                    block[i + s] = a - b;
                }
            }
        }
        for v in block {
            *v *= 1.0 / 128f32.sqrt();
        }
    }
}
#[test]
fn packed_cuda_matches_independent_oracle_all_supported_rates() {
    let b = match CudaBackend::new(0) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("[skipping] CUDA init failed: {e}");
            return;
        }
    };
    for tw in [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128] {
        let (k, n) = (128, 256);
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
            suh: (0..k)
                .map(|i| if i % 3 == 0 { -0.5 } else { 0.5 })
                .collect(),
            svh: (0..n)
                .map(|i| if i % 5 == 0 { -0.25 } else { 0.25 })
                .collect(),
            tile_words: tw,
            input_map: (0..k as u32).rev().collect(),
            output_map: (0..n as u32).rev().collect(),
        };
        data.validate().unwrap();
        for i in 0..k {
            for j in 0..n {
                assert_eq!(data.value(i, j), dense[i * n + j], "tw={tw} k={i} n={j}");
            }
        }
        let mut samples = vec![];
        let mut expected = vec![];
        for row in 0..5 {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 7 + row * 13) % 23) as f32 / 32.0 - 0.25)
                .collect();
            samples.extend_from_slice(&x);
            let mut xh: Vec<_> = data
                .input_map
                .iter()
                .enumerate()
                .map(|(i, &j)| half(x[j as usize]) * data.suh[i])
                .collect();
            had(&mut xh);
            for v in &mut xh {
                *v = half(*v);
            }
            let mut y: Vec<f32> = (0..n)
                .map(|j| half((0..k).map(|i| xh[i] * dense[i * n + j]).sum()))
                .collect();
            had(&mut y);
            expected.extend(
                data.output_map
                    .iter()
                    .map(|&j| half(y[j as usize] * data.svh[j as usize])),
            );
        }
        let w = Exl3Matrix::upload(b.clone(), data).unwrap();
        for rows in [1, 5] {
            let x = Tensor::from_vec(samples[..rows * k].to_vec(), vec![rows, k]);
            let y = w.linear(&x).to_host();
            for (i, (&a, &e)) in y.data().iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() < 0.003,
                    "tw={tw} rows={rows} i={i}: {a} != {e}"
                );
            }
        }
    }
}
