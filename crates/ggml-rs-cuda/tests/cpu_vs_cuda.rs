//! Validate every CUDA kernel against the CPU backend on small inputs.
//!
//! These tests require a working NVIDIA GPU + driver + CUDA toolkit. They're
//! skipped (with `eprintln!`) when no device is available.

use ggml_quants::GgmlType;
use ggml_rs::{CpuBackend, Backend, QuantizedTensor, RopeType, Tensor};
use ggml_rs_cuda::CudaBackend;

use half::f16;

fn try_cuda() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("[skipping] CUDA init failed: {e}");
            None
        }
    }
}

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    let diff = (a - b).abs();
    diff < eps || diff < eps * (a.abs() + b.abs())
}

fn assert_tensors_close(actual: &Tensor, expected: &Tensor, eps: f32) {
    assert_eq!(actual.shape(), expected.shape(),
               "shape mismatch: {:?} vs {:?}", actual.shape(), expected.shape());
    let actual_h = actual.to_host();
    let expected_h = expected.to_host();
    let max_diff = actual_h.data().iter().zip(expected_h.data().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let any_bad = actual_h.data().iter().zip(expected_h.data().iter())
        .any(|(a, b)| !approx_eq(*a, *b, eps));
    assert!(!any_bad, "tensors differ; max_diff={max_diff}; first 8 actual={:?}, expected={:?}",
            &actual_h.data()[..actual_h.numel().min(8)],
            &expected_h.data()[..expected_h.numel().min(8)]);
}

fn deterministic_floats(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        out.push(((z >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * scale);
    }
    out
}

#[test]
fn linear_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    // x: [3, 8]; w: [5, 8]; out: [3, 5]
    let x = Tensor::from_vec(deterministic_floats(24, 1.0, 1), vec![3, 8]);
    let w = Tensor::from_vec(deterministic_floats(40, 1.0, 2), vec![5, 8]);

    let cpu_y = cpu.linear(&x, &w);
    let cuda_y = cuda.linear(&x, &w);
    assert_tensors_close(&cuda_y, &cpu_y, 1e-4);
}

#[test]
fn rmsnorm_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let x = Tensor::from_vec(deterministic_floats(64, 2.0, 11), vec![4, 16]);
    let w = Tensor::from_vec(deterministic_floats(16, 0.5, 12), vec![16]);

    let cpu_y = cpu.rmsnorm(&x, &w, 1e-5);
    let cuda_y = cuda.rmsnorm(&x, &w, 1e-5);
    assert_tensors_close(&cuda_y, &cpu_y, 1e-4);
}

#[test]
fn softmax_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut a = Tensor::from_vec(deterministic_floats(40, 4.0, 21), vec![5, 8]);
    let mut b = cuda.to_device(a.clone());

    cpu.softmax_last(&mut a);
    cuda.softmax_last(&mut b);
    assert_tensors_close(&b, &a, 1e-4);
}

#[test]
fn silu_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    let x = Tensor::from_vec(deterministic_floats(33, 3.0, 31), vec![33]);
    assert_tensors_close(&cuda.silu(&x), &cpu.silu(&x), 1e-5);
}

#[test]
fn gelu_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    let x = Tensor::from_vec(deterministic_floats(33, 3.0, 32), vec![33]);
    assert_tensors_close(&cuda.gelu_approx(&x), &cpu.gelu_approx(&x), 1e-5);
}

#[test]
fn add_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    let mut a = Tensor::from_vec(deterministic_floats(20, 1.0, 41), vec![20]);
    let mut b = cuda.to_device(a.clone());
    let y = Tensor::from_vec(deterministic_floats(20, 1.0, 42), vec![20]);
    cpu.add_inplace(&mut a, &y);
    cuda.add_inplace(&mut b, &y);
    assert_tensors_close(&b, &a, 1e-5);
}

#[test]
fn mul_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    let mut a = Tensor::from_vec(deterministic_floats(20, 1.0, 51), vec![20]);
    let mut b = cuda.to_device(a.clone());
    let y = Tensor::from_vec(deterministic_floats(20, 1.0, 52), vec![20]);
    cpu.mul_inplace(&mut a, &y);
    cuda.mul_inplace(&mut b, &y);
    assert_tensors_close(&b, &a, 1e-5);
}

#[test]
fn rope_neox_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut a = Tensor::from_vec(deterministic_floats(2 * 4 * 8, 1.0, 61), vec![2, 4, 8]);
    let mut b = cuda.to_device(a.clone());
    let pos = vec![0u32, 5];
    cpu.rope(&mut a, &pos, 8, RopeType::NeoX, 10000.0, None);
    cuda.rope(&mut b, &pos, 8, RopeType::NeoX, 10000.0, None);
    assert_tensors_close(&b, &a, 1e-3);
}

#[test]
fn rope_normal_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut a = Tensor::from_vec(deterministic_floats(2 * 4 * 8, 1.0, 71), vec![2, 4, 8]);
    let mut b = cuda.to_device(a.clone());
    let pos = vec![0u32, 5];
    cpu.rope(&mut a, &pos, 8, RopeType::Normal, 10000.0, None);
    cuda.rope(&mut b, &pos, 8, RopeType::Normal, 10000.0, None);
    assert_tensors_close(&b, &a, 1e-3);
}

#[test]
fn rope_partial_neox_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    // 2 tokens, 4 heads, head_dim=16, rotated_dim=8 (only first 8 dims rotated).
    let mut a = Tensor::from_vec(deterministic_floats(2 * 4 * 16, 1.0, 91), vec![2, 4, 16]);
    let mut b = cuda.to_device(a.clone());
    let pos = vec![0u32, 7];
    cpu.rope_partial_neox(&mut a, &pos, 16, 8, 10000.0);
    cuda.rope_partial_neox(&mut b, &pos, 16, 8, 10000.0);
    assert_tensors_close(&b, &a, 1e-3);
}

#[test]
fn repeat_kv_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    let x = Tensor::from_vec(deterministic_floats(2 * 3 * 4, 1.0, 81), vec![2, 3, 4]);
    let cpu_y = cpu.repeat_kv(&x, 3);
    let cuda_y = cuda.repeat_kv(&x, 3);
    assert_tensors_close(&cuda_y, &cpu_y, 1e-6);
}

/// Build a Q8_0 weight tensor [N, K] from a deterministic float matrix.
/// Returns (QuantizedTensor on CPU, equivalent F32 Tensor for reference).
fn make_q8_0_weight(n: usize, k: usize, seed: u64) -> (QuantizedTensor, Tensor) {
    assert_eq!(k % 32, 0, "Q8_0 requires K%32==0");
    let elems_per_block = 32usize;
    let bytes_per_block = 34usize;
    let n_blocks_per_row = k / elems_per_block;
    let mut bytes = Vec::with_capacity(n * n_blocks_per_row * bytes_per_block);
    let mut dense = Vec::with_capacity(n * k);

    let mut state = seed;
    let mut rand = || -> f32 {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0
    };

    for _ in 0..n {
        for _ in 0..n_blocks_per_row {
            // Pick a scale.
            let raw_max = (0..elems_per_block).map(|_| rand()).map(|v| v.abs())
                .fold(0.0f32, f32::max).max(1e-6);
            let d = raw_max / 127.0;
            let d_f16 = f16::from_f32(d);
            let d_back = d_f16.to_f32();
            bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());

            // Quantize 32 values.
            let mut row_block = Vec::with_capacity(32);
            for _ in 0..elems_per_block {
                let v = rand();
                let q = ((v / d_back).round().clamp(-128.0, 127.0)) as i8;
                row_block.push(q);
                dense.push(q as f32 * d_back);
            }
            for q in row_block { bytes.push(q as u8); }
        }
    }

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q8_0);
    let dense = Tensor::from_vec(dense, vec![n, k]);
    (qt, dense)
}

#[test]
fn linear_q8_0_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 64usize;
    let n = 5usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 1001), vec![m, k]);
    let (qt, dense) = make_q8_0_weight(n, k, 1002);

    let cpu_dense_y = cpu.linear(&x, &dense);
    let cuda_q_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_q_y, &cpu_dense_y, 1e-3);
}

#[test]
fn linear_q8_0_with_persistent_weight() {
    // Upload weight once to device, then run linear_q multiple times.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 3usize;
    let k = 96usize;
    let n = 7usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 2001), vec![m, k]);
    let (qt_cpu, dense) = make_q8_0_weight(n, k, 2002);
    let qt_dev = cuda.upload_quantized(qt_cpu.bytes(), qt_cpu.shape().to_vec(), qt_cpu.dtype());

    let cpu_dense_y = cpu.linear(&x, &dense);
    let cuda_q_y = cuda.linear_q(&x, &qt_dev);

    assert_tensors_close(&cuda_q_y, &cpu_dense_y, 1e-3);
}

/// Build random Q4_K bytes for a [N, K] weight. We don't need them to come
/// from a real quantizer — we just need the bytes to dequantize consistently
/// (which they will, since dequantization is deterministic from the bytes).
fn make_q4_k_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "Q4_K requires K%256==0");
    let bytes_per_block = 144;
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    // Set scale + dmin to small positive f16 values per block so dequant doesn't
    // produce huge numbers that overflow.
    let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off+2].copy_from_slice(&small_d);
            bytes[off+2..off+4].copy_from_slice(&small_d);
        }
    }
    bytes
}

/// Same idea as `make_q4_k_random_bytes` but for Q6_K (210-byte super-blocks).
fn make_q6_k_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "Q6_K requires K%256==0");
    let bytes_per_block = 210;
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    // Force a small positive `d` per block so dequant doesn't blow up.
    let small_d = f16::from_f32(0.005).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            // d lives at bytes [208..210] of each block.
            bytes[off + 208..off + 210].copy_from_slice(&small_d);
            // Bound scales[16] (offset 192..208) to small magnitudes.
            for j in 0..16 { bytes[off + 192 + j] = (bytes[off + 192 + j] & 0x07) as u8; }
        }
    }
    bytes
}

#[test]
fn linear_q6_k_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 256usize;
    let n = 4usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 4001), vec![m, k]);
    let bytes = make_q6_k_random_bytes(n, k, 4002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q6_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q6_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

#[test]
fn linear_q4_k_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 256usize;
    let n = 4usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 3001), vec![m, k]);

    let bytes = make_q4_k_random_bytes(n, k, 3002);

    // Reference: dequantize bytes to F32 weight + dense linear.
    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q4_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    // Test: linear_q on the quantized bytes.
    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q4_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    // Relax tolerance — randomly-generated Q4_K bytes can produce big values.
    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

#[test]
fn linear_q4_k_gemv_coop_matches_dense_linear() {
    // Specifically exercises the M=1, K%1024==0 coop GEMV fast path. K=4096
    // matches the embedding dim of the Qwen3.5 / Llama / Gemma decoder paths.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 4096usize;
    let n = 64usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 4001), vec![m, k]);

    let bytes = make_q4_k_random_bytes(n, k, 4002);
    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q4_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q4_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    // Q4_K dequant + matmul accumulates rounding over 4096-wide dot products,
    // and the coop kernel's order-of-summation differs from the per-thread
    // reference; allow a slightly looser tolerance.
    assert_tensors_close(&cuda_y, &cpu_y, 1e-2);
}

#[test]
fn linear_f32_gemv_coop_matches_cpu() {
    // Specifically exercises the M=1 + K%32==0 dense F32 coop fast path
    // (used by Qwen3.6 27B's F32 ssm_ba matmul, K=5120 N=96).
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 5120usize;
    let n = 96usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 11001), vec![m, k]);
    let w = Tensor::from_vec(deterministic_floats(n * k, 0.5, 11002), vec![n, k]);

    let cpu_y  = cpu.linear(&x, &w);
    let cuda_y = cuda.linear(&x, &w);

    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

#[test]
fn linear_q5_0_gemv_coop_matches_dense_linear() {
    // K=1152 deliberately matches Gemma 3-1B's hidden_dim, where attn_q/k +
    // ffn_gate/up are all Q5_0 with K not divisible by 1024 (so the K-quants
    // coop kernels never apply — the element-parallel coop path is the only
    // win available for these matmuls).
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 1152usize;
    let n = 64usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 7001), vec![m, k]);

    let bytes = make_q5_0_random_bytes(n, k, 7002);
    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q5_0, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q5_0);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-2);
}

#[test]
fn linear_q5_1_gemv_coop_matches_dense_linear() {
    // Gemma 3n e2b is 50% Q5_1; K=2048 matches its hidden_dim.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 2048usize;
    let n = 64usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 9001), vec![m, k]);

    let bytes = make_q5_1_random_bytes(n, k, 9002);
    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q5_1, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q5_1);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-2);
}

#[test]
fn linear_q8_0_gemv_coop_matches_dense_linear() {
    // K=1152 again — Gemma 3-1B's attn_v is Q8_0 at this dim. Validates that
    // the element-parallel coop kernel handles the same non-1024-aligned K.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 1152usize;
    let n = 64usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 8001), vec![m, k]);

    let (qt, dense) = make_q8_0_weight(n, k, 8002);
    let cpu_y = cpu.linear(&x, &dense);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-2);
}

#[test]
fn linear_q5_k_gemv_coop_matches_dense_linear() {
    // Specifically exercises the M=1, K%1024==0 Q5_K coop GEMV fast path. K=4096
    // covers the Qwen3.5-9B / Llama / Gemma decoder shapes where Q5_K appears.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 4096usize;
    let n = 64usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 6001), vec![m, k]);

    let bytes = make_q5_k_random_bytes(n, k, 6002);
    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q5_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q5_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-2);
}

#[test]
fn linear_q6_k_gemv_coop_matches_dense_linear() {
    // Specifically exercises the M=1, K%256==0 Q6_K coop GEMV fast path.
    // K=2048 matches Llama-3.2-1B's hidden_dim (lm_head is [128256, 2048] Q6_K).
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 1usize;
    let k = 2048usize;
    let n = 64usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 5001), vec![m, k]);

    let bytes = make_q6_k_random_bytes(n, k, 5002);
    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q6_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q6_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    // Looser tolerance for the coop kernel — the warp-reduce changes the
    // order of summation vs the per-thread reference, and Q6_K-on-random-bytes
    // can produce big-magnitude values.
    assert_tensors_close(&cuda_y, &cpu_y, 1e-2);
}

#[test]
fn argmax_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    let x = Tensor::from_vec(deterministic_floats(50, 2.0, 91), vec![5, 10]);
    assert_eq!(cuda.argmax_last(&x), cpu.argmax_last(&x));
}

/// Random Q2_K bytes for a [N, K] weight. 84 bytes per super-block.
fn make_q2_k_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "Q2_K requires K%256==0");
    let bytes_per_block = 84;
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d   = f16::from_f32(0.005).to_bits().to_le_bytes();
    let small_min = f16::from_f32(0.005).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            // d at [80..82], dmin at [82..84]
            bytes[off + 80..off + 82].copy_from_slice(&small_d);
            bytes[off + 82..off + 84].copy_from_slice(&small_min);
        }
    }
    bytes
}

#[test]
fn linear_q2_k_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 256usize;
    let n = 4usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 9001), vec![m, k]);
    let bytes = make_q2_k_random_bytes(n, k, 9002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q2_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q2_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

/// Random Q3_K bytes for a [N, K] weight. 110 bytes per super-block.
fn make_q3_k_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "Q3_K requires K%256==0");
    let bytes_per_block = 110;
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.005).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            // d at [108..110]
            bytes[off + 108..off + 110].copy_from_slice(&small_d);
        }
    }
    bytes
}

#[test]
fn linear_q3_k_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 256usize;
    let n = 4usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 8001), vec![m, k]);
    let bytes = make_q3_k_random_bytes(n, k, 8002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q3_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q3_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

/// Random Q5_K bytes for a [N, K] weight. 176 bytes per super-block.
fn make_q5_k_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "Q5_K requires K%256==0");
    let bytes_per_block = 176;
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
            bytes[off + 2..off + 4].copy_from_slice(&small_d);
        }
    }
    bytes
}

#[test]
fn linear_q5_k_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 256usize;
    let n = 4usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 5001), vec![m, k]);
    let bytes = make_q5_k_random_bytes(n, k, 5002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q5_K, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q5_K);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

/// Random IQ4_XS bytes for a [N, K] weight. 136 bytes per 256-elem super-block.
fn make_iq4_xs_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "IQ4_XS requires K%256==0");
    let bytes_per_block = 136;
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.005).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
        }
    }
    bytes
}

#[test]
fn linear_iq4_xs_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 256usize;
    let n = 4usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 13001), vec![m, k]);
    let bytes = make_iq4_xs_random_bytes(n, k, 13002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::IQ4_XS, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::IQ4_XS);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 5e-3);
}

/// Random IQ4_NL bytes for a [N, K] weight. 18 bytes per 32-elem block.
fn make_iq4_nl_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0, "IQ4_NL requires K%32==0");
    let bytes_per_block = 18;
    let blocks_per_row = k / 32;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.005).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
        }
    }
    bytes
}

#[test]
fn linear_iq4_nl_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 64usize;
    let n = 5usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 12001), vec![m, k]);
    let bytes = make_iq4_nl_random_bytes(n, k, 12002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::IQ4_NL, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::IQ4_NL);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-3);
}

/// Random Q4_0 bytes for a [N, K] weight. 18 bytes per 32-elem block.
fn make_q4_0_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q4_0 requires K%32==0");
    let bytes_per_block = 18;
    let blocks_per_row = k / 32;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
        }
    }
    bytes
}

#[test]
fn linear_q4_0_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 64usize;
    let n = 5usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 10001), vec![m, k]);
    let bytes = make_q4_0_random_bytes(n, k, 10002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q4_0, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q4_0);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-3);
}

/// Random Q4_1 bytes for a [N, K] weight. 20 bytes per 32-elem block.
fn make_q4_1_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q4_1 requires K%32==0");
    let bytes_per_block = 20;
    let blocks_per_row = k / 32;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
    let small_m = f16::from_f32(-0.05).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
            bytes[off + 2..off + 4].copy_from_slice(&small_m);
        }
    }
    bytes
}

#[test]
fn linear_q4_1_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 64usize;
    let n = 5usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 11001), vec![m, k]);
    let bytes = make_q4_1_random_bytes(n, k, 11002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q4_1, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q4_1);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-3);
}

/// Random Q5_0 bytes for a [N, K] weight. 22 bytes per 32-elem block.
fn make_q5_0_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q5_0 requires K%32==0");
    let bytes_per_block = 22;
    let blocks_per_row = k / 32;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
        }
    }
    bytes
}

#[test]
fn linear_q5_0_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 64usize;
    let n = 5usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 6001), vec![m, k]);
    let bytes = make_q5_0_random_bytes(n, k, 6002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q5_0, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q5_0);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-3);
}

/// Random Q5_1 bytes for a [N, K] weight. 24 bytes per 32-elem block.
fn make_q5_1_random_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q5_1 requires K%32==0");
    let bytes_per_block = 24;
    let blocks_per_row = k / 32;
    let total = n * blocks_per_row * bytes_per_block;
    let mut bytes = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
    let small_m = f16::from_f32(-0.1).to_bits().to_le_bytes();
    for nrow in 0..n {
        for blk in 0..blocks_per_row {
            let off = (nrow * blocks_per_row + blk) * bytes_per_block;
            bytes[off..off + 2].copy_from_slice(&small_d);
            bytes[off + 2..off + 4].copy_from_slice(&small_m);
        }
    }
    bytes
}

#[test]
fn linear_q5_1_matches_dense_linear() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let m = 2usize;
    let k = 64usize;
    let n = 5usize;
    let x = Tensor::from_vec(deterministic_floats(m * k, 1.0, 7001), vec![m, k]);
    let bytes = make_q5_1_random_bytes(n, k, 7002);

    let mut dense = vec![0.0f32; n * k];
    ggml_quants::dequantize(GgmlType::Q5_1, &bytes, &mut dense).unwrap();
    let dense_t = Tensor::from_vec(dense, vec![n, k]);
    let cpu_y = cpu.linear(&x, &dense_t);

    let qt = QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], GgmlType::Q5_1);
    let cuda_y = cuda.linear_q(&x, &qt);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-3);
}

/// Sliding-window attention: with a tiny window (4) over an 8-position cache,
/// CUDA fused attention must agree with CPU reference. This exercises the
/// `min_t` branch of the kernel and the host-side mask in the trait default.
#[test]
fn attention_sliding_window_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let seq = 2usize;
    let n_h_q = 2usize;
    let n_h_kv = 1usize;
    let hd = 8usize;
    let kv_len = 8usize;
    let max_kv_len = 16usize;
    let past = kv_len - seq;          // current step's queries are at positions [past, past+seq).
    let scale = 1.0 / (hd as f32).sqrt();
    let sw = Some(4usize);

    let q  = Tensor::from_vec(deterministic_floats(seq * n_h_q * hd, 1.0, 20001), vec![seq, n_h_q, hd]);
    let mut k_buf = vec![0.0f32; max_kv_len * n_h_kv * hd];
    let mut v_buf = vec![0.0f32; max_kv_len * n_h_kv * hd];
    let k_data = deterministic_floats(kv_len * n_h_kv * hd, 1.0, 20002);
    let v_data = deterministic_floats(kv_len * n_h_kv * hd, 1.0, 20003);
    k_buf[..k_data.len()].copy_from_slice(&k_data);
    v_buf[..v_data.len()].copy_from_slice(&v_data);
    let k_buffer = Tensor::from_vec(k_buf, vec![max_kv_len, n_h_kv, hd]);
    let v_buffer = Tensor::from_vec(v_buf, vec![max_kv_len, n_h_kv, hd]);

    let cpu_y = cpu.attention(&q, &k_buffer, &v_buffer, kv_len, scale, past, sw);

    let q_dev = cuda.to_device(q.clone());
    let k_dev = cuda.to_device(k_buffer.clone());
    let v_dev = cuda.to_device(v_buffer.clone());
    let cuda_y = cuda.attention(&q_dev, &k_dev, &v_dev, kv_len, scale, past, sw);

    assert_tensors_close(&cuda_y, &cpu_y, 1e-4);

    // Also verify that disabling sliding window (None) still matches.
    let cpu_full = cpu.attention(&q, &k_buffer, &v_buffer, kv_len, scale, past, None);
    let cuda_full = cuda.attention(&q_dev, &k_dev, &v_dev, kv_len, scale, past, None);
    assert_tensors_close(&cuda_full, &cpu_full, 1e-4);

    // And that the SWA result genuinely differs from the full-attention result
    // (otherwise the mask isn't doing anything).
    let any_differ = cpu_y.to_host().data().iter().zip(cpu_full.to_host().data().iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(any_differ, "SWA and full attention produced identical outputs — mask not active");
}

#[test]
fn rmsnorm_no_scale_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let x = Tensor::from_vec(deterministic_floats(4 * 16, 1.0, 30001), vec![4, 16]);
    let cpu_y = cpu.rmsnorm_no_scale(&x, 1e-5);
    let cuda_y = cuda.rmsnorm_no_scale(&cuda.to_device(x), 1e-5);
    assert_tensors_close(&cuda_y, &cpu_y, 1e-4);
}

#[test]
fn mul_inplace_broadcast_last_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut x_cpu = Tensor::from_vec(deterministic_floats(4 * 16, 1.0, 30002), vec![4, 16]);
    let w = Tensor::from_vec(deterministic_floats(16, 1.0, 30003), vec![16]);
    let mut x_cuda = cuda.to_device(x_cpu.clone());
    cpu.mul_inplace_broadcast_last(&mut x_cpu, &w);
    cuda.mul_inplace_broadcast_last(&mut x_cuda, &cuda.to_device(w));
    assert_tensors_close(&x_cuda, &x_cpu, 1e-5);
}

#[test]
fn add_to_axis0_range_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut dst_cpu = Tensor::from_vec(deterministic_floats(5 * 4 * 8, 1.0, 30004), vec![5, 4, 8]);
    let src = Tensor::from_vec(deterministic_floats(4 * 8, 1.0, 30005), vec![4, 8]);
    let mut dst_cuda = cuda.to_device(dst_cpu.clone());
    cpu.add_to_axis0_range(&mut dst_cpu, 1, 3, &src);
    cuda.add_to_axis0_range(&mut dst_cuda, 1, 3, &cuda.to_device(src));
    assert_tensors_close(&dst_cuda, &dst_cpu, 1e-5);
}

#[test]
fn gaussian_topk_inplace_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut x_cpu = Tensor::from_vec(deterministic_floats(4 * 32, 1.0, 30006), vec![4, 32]);
    let mut x_cuda = cuda.to_device(x_cpu.clone());
    let mult = 1.6448f32; // Φ⁻¹(0.95) — the Gemma 3n value for 95% sparsity.
    cpu.gaussian_topk_inplace(&mut x_cpu, mult);
    cuda.gaussian_topk_inplace(&mut x_cuda, mult);
    assert_tensors_close(&x_cuda, &x_cpu, 1e-4);
}

#[test]
fn altup_predict_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let n_alt = 4usize;
    let seq = 3usize;
    let hidden = 8usize;
    let streams = Tensor::from_vec(
        deterministic_floats(n_alt * seq * hidden, 1.0, 30007),
        vec![n_alt, seq, hidden],
    );
    let coefs = Tensor::from_vec(
        deterministic_floats(seq * n_alt * n_alt, 0.1, 30008),
        vec![seq, n_alt * n_alt],
    );
    let cpu_y = cpu.altup_predict(&streams, &coefs, n_alt);
    let cuda_y = cuda.altup_predict(&cuda.to_device(streams), &cuda.to_device(coefs), n_alt);
    assert_tensors_close(&cuda_y, &cpu_y, 1e-4);
}

#[test]
fn altup_correct_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let n_alt = 4usize;
    let seq = 3usize;
    let hidden = 8usize;
    let active_idx = 0usize;
    let preds = Tensor::from_vec(
        deterministic_floats(n_alt * seq * hidden, 1.0, 30009),
        vec![n_alt, seq, hidden],
    );
    let activated = Tensor::from_vec(
        deterministic_floats(seq * hidden, 1.0, 30010),
        vec![seq, hidden],
    );
    let coefs = Tensor::from_vec(
        deterministic_floats(seq * n_alt, 0.1, 30011),
        vec![seq, n_alt],
    );
    let cpu_y = cpu.altup_correct(&preds, &activated, &coefs, n_alt, active_idx);
    let cuda_y = cuda.altup_correct(
        &cuda.to_device(preds),
        &cuda.to_device(activated),
        &cuda.to_device(coefs),
        n_alt, active_idx,
    );
    assert_tensors_close(&cuda_y, &cpu_y, 1e-4);
}

#[test]
fn tanh_inplace_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut x_cpu = Tensor::from_vec(deterministic_floats(64, 2.0, 30012), vec![64]);
    let mut x_cuda = cuda.to_device(x_cpu.clone());
    cpu.tanh_inplace(&mut x_cpu);
    cuda.tanh_inplace(&mut x_cuda);
    assert_tensors_close(&x_cuda, &x_cpu, 1e-5);
}

#[test]
fn slice_axis1_2d_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let src = Tensor::from_vec(deterministic_floats(4 * 6 * 8, 1.0, 30013), vec![4, 6, 8]);
    let cpu_y = cpu.slice_axis1_2d(&src, 3);
    let cuda_y = cuda.slice_axis1_2d(&cuda.to_device(src), 3);
    assert_tensors_close(&cuda_y, &cpu_y, 1e-6);
}

#[test]
fn mul_scalar_inplace_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let mut x_cpu = Tensor::from_vec(deterministic_floats(64, 1.0, 30014), vec![64]);
    let mut x_cuda = cuda.to_device(x_cpu.clone());
    cpu.mul_scalar_inplace(&mut x_cpu, 2.5);
    cuda.mul_scalar_inplace(&mut x_cuda, 2.5);
    assert_tensors_close(&x_cuda, &x_cpu, 1e-6);
}

#[test]
fn delta_net_decode_step_matches_cpu() {
    // Tiny but representative of Qwen3.5: num_v_heads=4, num_k_heads=2 (so
    // v_per_k=2), head_v_dim=head_k_dim=32, conv_kernel=4. Total conv_dim
    // = 2*num_k_heads*head_k_dim + num_v_heads*head_v_dim = 128+128 = 256.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let num_v_heads = 4;
    let num_k_heads = 2;
    let head_v_dim  = 32;
    let head_k_dim  = 32;
    let v_per_k     = num_v_heads / num_k_heads;
    let conv_kernel = 4;
    let conv_dim    = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    let scale_q     = 1.0 / (head_v_dim as f32).sqrt();
    let eps         = 1e-6f32;

    let mqkv  = Tensor::from_vec(deterministic_floats(conv_dim, 0.5, 50001), vec![conv_dim]);
    let z     = Tensor::from_vec(deterministic_floats(num_v_heads * head_v_dim, 0.5, 50002),
                                 vec![num_v_heads * head_v_dim]);
    let ba    = Tensor::from_vec(deterministic_floats(2 * num_v_heads, 0.3, 50003),
                                 vec![2 * num_v_heads]);
    let cw    = Tensor::from_vec(deterministic_floats(conv_dim * conv_kernel, 0.2, 50004),
                                 vec![conv_dim, conv_kernel]);
    let sa    = Tensor::from_vec(deterministic_floats(num_v_heads, 0.5, 50005),
                                 vec![num_v_heads]);
    let dt    = Tensor::from_vec(deterministic_floats(num_v_heads, 0.1, 50006),
                                 vec![num_v_heads]);
    let nm    = Tensor::from_vec(deterministic_floats(head_v_dim, 0.4, 50007),
                                 vec![head_v_dim]);

    // Initial state: zeros
    let conv_state_cpu = Tensor::zeros(vec![conv_kernel - 1, conv_dim]);
    let state_cpu      = Tensor::zeros(vec![num_v_heads, head_v_dim, head_v_dim]);

    let mut conv_state_a = conv_state_cpu.clone();
    let mut state_a      = state_cpu.clone();
    let out_cpu = cpu.delta_net_step(
        &mqkv, &z, &ba, &cw, &sa, &dt, &nm,
        &mut conv_state_a, &mut state_a,
        1, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps,
    );

    let mut conv_state_b = cuda.to_device(conv_state_cpu);
    let mut state_b      = cuda.to_device(state_cpu);
    let out_cuda = cuda.delta_net_step(
        &cuda.to_device(mqkv),
        &cuda.to_device(z),
        &cuda.to_device(ba),
        &cuda.to_device(cw),
        &cuda.to_device(sa),
        &cuda.to_device(dt),
        &cuda.to_device(nm),
        &mut conv_state_b, &mut state_b,
        1, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps,
    );

    assert_tensors_close(&out_cuda, &out_cpu, 1e-4);
    assert_tensors_close(&conv_state_b, &conv_state_a, 1e-4);
    assert_tensors_close(&state_b, &state_a, 1e-4);
}

#[test]
fn delta_net_decode_step_full_size_matches_cpu() {
    // Production Qwen3.5-9B sizes: num_v_heads=32, num_k_heads=16, v_per_k=2,
    // head_v_dim=head_k_dim=128, conv_kernel=4 → conv_dim=8192. Verifies the
    // CUDA kernel at the size used in real inference (and exercises the 128-
    // thread reduction in shared memory).
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let num_v_heads = 32;
    let num_k_heads = 16;
    let head_v_dim  = 128;
    let head_k_dim  = 128;
    let v_per_k     = num_v_heads / num_k_heads;
    let conv_kernel = 4;
    let conv_dim    = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    let scale_q     = 1.0 / (head_v_dim as f32).sqrt();
    let eps         = 1e-6f32;

    let mqkv  = Tensor::from_vec(deterministic_floats(conv_dim, 0.5, 60001), vec![conv_dim]);
    let z     = Tensor::from_vec(deterministic_floats(num_v_heads * head_v_dim, 0.5, 60002),
                                 vec![num_v_heads * head_v_dim]);
    let ba    = Tensor::from_vec(deterministic_floats(2 * num_v_heads, 0.3, 60003),
                                 vec![2 * num_v_heads]);
    let cw    = Tensor::from_vec(deterministic_floats(conv_dim * conv_kernel, 0.2, 60004),
                                 vec![conv_dim, conv_kernel]);
    let sa    = Tensor::from_vec(deterministic_floats(num_v_heads, 0.5, 60005),
                                 vec![num_v_heads]);
    let dt    = Tensor::from_vec(deterministic_floats(num_v_heads, 0.1, 60006),
                                 vec![num_v_heads]);
    let nm    = Tensor::from_vec(deterministic_floats(head_v_dim, 0.4, 60007),
                                 vec![head_v_dim]);

    let conv_state_cpu = Tensor::zeros(vec![conv_kernel - 1, conv_dim]);
    let state_cpu      = Tensor::zeros(vec![num_v_heads, head_v_dim, head_v_dim]);

    let mut conv_state_a = conv_state_cpu.clone();
    let mut state_a      = state_cpu.clone();
    let out_cpu = cpu.delta_net_step(
        &mqkv, &z, &ba, &cw, &sa, &dt, &nm,
        &mut conv_state_a, &mut state_a,
        1, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps,
    );

    let mut conv_state_b = cuda.to_device(conv_state_cpu);
    let mut state_b      = cuda.to_device(state_cpu);
    let out_cuda = cuda.delta_net_step(
        &cuda.to_device(mqkv),
        &cuda.to_device(z),
        &cuda.to_device(ba),
        &cuda.to_device(cw),
        &cuda.to_device(sa),
        &cuda.to_device(dt),
        &cuda.to_device(nm),
        &mut conv_state_b, &mut state_b,
        1, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps,
    );

    assert_tensors_close(&out_cuda, &out_cpu, 1e-3);
    assert_tensors_close(&conv_state_b, &conv_state_a, 1e-4);
    assert_tensors_close(&state_b, &state_a, 1e-3);
}

#[test]
fn split_q_and_gate_matches_cpu() {
    // Qwen3.5 attention layer joint Q+gate split: per-head [q | gate] layout
    // gets split into two [seq, n_heads * head_dim] tensors. Tested at the
    // 9B production sizes (seq=3, n_heads=16, head_dim=256).
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let seq      = 3;
    let n_heads  = 16;
    let head_dim = 256;
    let total = seq * n_heads * 2 * head_dim;

    let q_full = Tensor::from_vec(deterministic_floats(total, 1.0, 80001),
                                  vec![seq, n_heads * 2 * head_dim]);
    let (q_cpu, g_cpu) = cpu.split_q_and_gate(&q_full, n_heads, head_dim);
    let (q_cuda, g_cuda) = cuda.split_q_and_gate(&cuda.to_device(q_full), n_heads, head_dim);

    assert_eq!(q_cuda.shape(), &[seq, n_heads * head_dim]);
    assert_eq!(g_cuda.shape(), &[seq, n_heads * head_dim]);
    assert_tensors_close(&q_cuda, &q_cpu, 1e-6);
    assert_tensors_close(&g_cuda, &g_cpu, 1e-6);
}

#[test]
fn delta_net_step_prefill_seq_matches_cpu() {
    // Multi-token prefill path (seq=4): exercises the per-token loop in both
    // CPU + CUDA delta_net_step impls. Both must update conv_state and state
    // sequentially across the seq tokens and produce identical outputs.
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let seq         = 4;
    let num_v_heads = 8;
    let num_k_heads = 4;
    let head_v_dim  = 32;
    let head_k_dim  = 32;
    let v_per_k     = num_v_heads / num_k_heads;
    let conv_kernel = 4;
    let conv_dim    = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    let scale_q     = 1.0 / (head_v_dim as f32).sqrt();
    let eps         = 1e-6f32;

    let mqkv  = Tensor::from_vec(deterministic_floats(seq * conv_dim, 0.5, 70001),
                                 vec![seq, conv_dim]);
    let z     = Tensor::from_vec(deterministic_floats(seq * num_v_heads * head_v_dim, 0.5, 70002),
                                 vec![seq, num_v_heads * head_v_dim]);
    let ba    = Tensor::from_vec(deterministic_floats(seq * 2 * num_v_heads, 0.3, 70003),
                                 vec![seq, 2 * num_v_heads]);
    let cw    = Tensor::from_vec(deterministic_floats(conv_dim * conv_kernel, 0.2, 70004),
                                 vec![conv_dim, conv_kernel]);
    let sa    = Tensor::from_vec(deterministic_floats(num_v_heads, 0.5, 70005),
                                 vec![num_v_heads]);
    let dt    = Tensor::from_vec(deterministic_floats(num_v_heads, 0.1, 70006),
                                 vec![num_v_heads]);
    let nm    = Tensor::from_vec(deterministic_floats(head_v_dim, 0.4, 70007),
                                 vec![head_v_dim]);

    let conv_state_cpu = Tensor::zeros(vec![conv_kernel - 1, conv_dim]);
    let state_cpu      = Tensor::zeros(vec![num_v_heads, head_v_dim, head_v_dim]);

    let mut conv_state_a = conv_state_cpu.clone();
    let mut state_a      = state_cpu.clone();
    let out_cpu = cpu.delta_net_step(
        &mqkv, &z, &ba, &cw, &sa, &dt, &nm,
        &mut conv_state_a, &mut state_a,
        seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps,
    );

    let mut conv_state_b = cuda.to_device(conv_state_cpu);
    let mut state_b      = cuda.to_device(state_cpu);
    let out_cuda = cuda.delta_net_step(
        &cuda.to_device(mqkv),
        &cuda.to_device(z),
        &cuda.to_device(ba),
        &cuda.to_device(cw),
        &cuda.to_device(sa),
        &cuda.to_device(dt),
        &cuda.to_device(nm),
        &mut conv_state_b, &mut state_b,
        seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps,
    );

    assert_eq!(out_cuda.shape(), &[seq, num_v_heads * head_v_dim]);
    assert_tensors_close(&out_cuda, &out_cpu, 1e-3);
    assert_tensors_close(&conv_state_b, &conv_state_a, 1e-4);
    assert_tensors_close(&state_b, &state_a, 1e-3);
}

// VENDORED-LOCAL: GLM-5.3-Flash clamped SwiGLU.
/// Both clamp orders, against the CPU backend. glm5next needs both: the text FFN
/// clamps the activation, the vision tower clamps the pre-activation, and values
/// astride the limit are where they disagree.
/// VENDORED-LOCAL: GLM-5.3-Flash. The batched GEMV against the host default, at
/// the released absorbed-MLA shapes and a few awkward ones.
///
/// `k_b` is `[64, 512, 256]` and `v_b` is `[64, 256, 512]`; the small cases check
/// an `m` that is not a multiple of the 8 rows a block covers, and a `k` shorter
/// than one warp.
#[test]
fn batched_gemv_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let host = CpuBackend::new();

    for &(b, m, k) in &[
        (64usize, 512usize, 256usize),
        (64, 256, 512),
        (3, 13, 7),
        (1, 1, 1),
        (5, 9, 32),
    ] {
        let w: Vec<f32> = (0..b * m * k)
            .map(|i| (((i * 37) % 97) as f32 - 48.0) / 48.0)
            .collect();
        let x: Vec<f32> = (0..b * k)
            .map(|i| (((i * 53) % 89) as f32 - 44.0) / 44.0)
            .collect();

        let wt = Tensor::from_vec(w, vec![b, m, k]);
        let xt = Tensor::from_vec(x, vec![b, k]);

        let want = host.to_host(host.batched_gemv(
            &host.to_device(wt.clone()),
            &host.to_device(xt.clone()),
            b,
            m,
            k,
        ));
        let got = cuda.to_host(cuda.batched_gemv(
            &cuda.to_device(wt),
            &cuda.to_device(xt),
            b,
            m,
            k,
        ));
        assert_eq!(want.data().len(), b * m);
        assert_eq!(got.data().len(), b * m);

        let scale = want
            .data()
            .iter()
            .fold(0.0f32, |a, v| a.max(v.abs()))
            .max(1.0);
        for (i, (p, q)) in want.data().iter().zip(got.data()).enumerate() {
            assert!(
                (p - q).abs() <= 2e-5 * scale,
                "[{b},{m},{k}] row {i}: host {p} vs cuda {q}"
            );
        }
        // and not trivially zero
        assert!(
            want.data().iter().any(|v| v.abs() > 1e-3),
            "[{b},{m},{k}] all zero"
        );
    }
}

#[test]
fn swiglu_clamped_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    let n = 1024usize;
    // Spread well past the limit on both sides so the clamp actually bites.
    let gate: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 3.0).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 61) as f32 - 30.0) / 2.0).collect();

    for &limit in &[10.0f32, 2.0, 0.0] {
        for &after in &[true, false] {
            let gc = cuda.to_device(Tensor::from_vec(gate.clone(), vec![1, n]));
            let uc = cuda.to_device(Tensor::from_vec(up.clone(), vec![1, n]));
            let got = cuda.swiglu_clamped(&gc, &uc, limit, after);

            let gh = Tensor::from_vec(gate.clone(), vec![1, n]);
            let uh = Tensor::from_vec(up.clone(), vec![1, n]);
            let want = cpu.swiglu_clamped(&gh, &uh, limit, after);

            assert_tensors_close(&got, &want, 1e-5);
        }
    }

    // And the two orders must genuinely differ where the gate exceeds the limit,
    // or the flag is not doing anything.
    let gc = cuda.to_device(Tensor::from_vec(vec![5.0f32; 8], vec![1, 8]));
    let uc = cuda.to_device(Tensor::from_vec(vec![1.0f32; 8], vec![1, 8]));
    let a = cuda.swiglu_clamped(&gc, &uc, 2.0, true).to_host();
    let b = cuda.swiglu_clamped(&gc, &uc, 2.0, false).to_host();
    assert!(
        (a.data()[0] - b.data()[0]).abs() > 0.2,
        "the clamp order must matter: {} vs {}",
        a.data()[0],
        b.data()[0]
    );
    // after_silu: silu(5) = 4.967 -> clamped to 2.0
    assert!((a.data()[0] - 2.0).abs() < 1e-4, "got {}", a.data()[0]);
    // pre-activation: silu(clamp(5)=2) = 1.7616
    assert!((b.data()[0] - 1.7616).abs() < 1e-3, "got {}", b.data()[0]);
}
