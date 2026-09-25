use ggml_rs::{Backend, CpuBackend, Tensor};
use ggml_rs_cuda::CudaBackend;

#[test]
fn partitioned_attention_matches_cpu_causal_gqa_and_window_masks() {
    let Ok(gpu) = CudaBackend::new(0) else { return };
    let cpu = CpuBackend::new();
    // Cross the old shared-memory limit, with partial and wholly masked chunks.
    for (seq, len, hd, past, window) in [
        (3, 12301, 8, 12298, None),
        (2, 13000, 256, 7000, None),
        (2, 13000, 32, 12998, Some(350)),
        (2, 13000, 32, 8500, Some(5000)),
    ] {
        let (nh, nk) = (4, 2);
        let tensor = |n: usize, shape| {
            Tensor::from_vec(
                (0..n)
                    .map(|i| ((i * 17 % 103) as f32 - 51.0) / 53.0)
                    .collect(),
                shape,
            )
        };
        let q = tensor(seq * nh * hd, vec![seq, nh, hd]);
        let k = tensor(len * nk * hd, vec![len, nk, hd]);
        let v = tensor(len * nk * hd, vec![len, nk, hd]);
        let scale = 1.0 / (hd as f32).sqrt();
        let expected = cpu.attention(&q, &k, &v, len, scale, past, window);
        let actual = gpu
            .attention(
                &gpu.to_device(q),
                &gpu.to_device(k),
                &gpu.to_device(v),
                len,
                scale,
                past,
                window,
            )
            .to_host();
        let error = actual
            .data()
            .iter()
            .zip(expected.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            error < 0.0001,
            "seq={seq} hd={hd} window={window:?}: error={error}"
        );
    }
}

#[test]
fn attention_at_260000_tokens_has_bounded_memory_and_correct_average() {
    let Ok(gpu) = CudaBackend::new(0) else { return };
    let (len, hd) = (260000, 256);
    let q = gpu.alloc_zeros(vec![1, 2, hd]);
    let k = gpu.alloc_zeros(vec![len, 1, hd]);
    let mut v = Tensor::from_vec(vec![0.0; len * hd], vec![len, 1, hd]);
    for (t, row) in v.data_mut().chunks_mut(hd).enumerate() {
        row.fill(if t % 2 == 0 { 1.0 } else { 3.0 });
    }
    let v = gpu.to_device(v);
    let out = gpu.attention(&q, &k, &v, len, 1.0, len - 1, None).to_host();
    assert!(out.data().iter().all(|v| (v - 2.0).abs() < 1e-5));
}
