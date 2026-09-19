//! Differential tests for MOE-01 (on-device top-k + softmax routing) and
//! MOE-02 (grouped gate_up+act / down+scale / reduce kernels), plus the
//! SAMPLE-01 argmax contract.
//!
//! These tests require a working NVIDIA GPU + driver + CUDA toolkit. They're
//! skipped (with `eprintln!`) when no device is available.

use ggml_quants::GgmlType;
use ggml_rs::{Backend, CpuBackend, QuantizedTensor, Tensor};
use ggml_rs_cuda::{quant_device_ptr, CudaBackend, MoeDevicePlan};

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
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    let actual_h = actual.to_host();
    let expected_h = expected.to_host();
    let max_diff = actual_h
        .data()
        .iter()
        .zip(expected_h.data().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let any_bad = actual_h
        .data()
        .iter()
        .zip(expected_h.data().iter())
        .any(|(a, b)| !approx_eq(*a, *b, eps));
    assert!(
        !any_bad,
        "tensors differ; max_diff={max_diff}; first 8 actual={:?}, expected={:?}",
        &actual_h.data()[..actual_h.numel().min(8)],
        &expected_h.data()[..expected_h.numel().min(8)]
    );
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

fn splitmix_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        bytes.push((z & 0xFF) as u8);
    }
    bytes
}

/// Random packed bytes for a [N, K] weight of `dtype`, with scales forced
/// small so dequantized values stay tame. The bytes don't need to come from
/// a real quantizer — dequantization is deterministic from the bytes, which
/// is all the kernels read.
fn make_random_weight_bytes(dtype: GgmlType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    match dtype {
        GgmlType::Q4_K => {
            assert_eq!(k % 256, 0, "Q4_K requires K%256==0");
            let bpr = k / 256;
            let mut bytes = splitmix_bytes(n * bpr * 144, seed);
            let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
            for row in 0..n {
                for blk in 0..bpr {
                    let off = (row * bpr + blk) * 144;
                    bytes[off..off + 2].copy_from_slice(&small_d);
                    bytes[off + 2..off + 4].copy_from_slice(&small_d);
                }
            }
            bytes
        }
        GgmlType::Q6_K => {
            assert_eq!(k % 256, 0, "Q6_K requires K%256==0");
            let bpr = k / 256;
            let mut bytes = splitmix_bytes(n * bpr * 210, seed);
            let small_d = f16::from_f32(0.005).to_bits().to_le_bytes();
            for row in 0..n {
                for blk in 0..bpr {
                    let off = (row * bpr + blk) * 210;
                    bytes[off + 208..off + 210].copy_from_slice(&small_d);
                    for j in 0..16 {
                        bytes[off + 192 + j] &= 0x07;
                    }
                }
            }
            bytes
        }
        GgmlType::Q8_0 => {
            assert_eq!(k % 32, 0, "Q8_0 requires K%32==0");
            let bpr = k / 32;
            let mut bytes = splitmix_bytes(n * bpr * 34, seed);
            let small_d = f16::from_f32(0.01).to_bits().to_le_bytes();
            for row in 0..n {
                for blk in 0..bpr {
                    let off = (row * bpr + blk) * 34;
                    bytes[off..off + 2].copy_from_slice(&small_d);
                }
            }
            bytes
        }
        other => panic!("test fixture has no byte maker for {other:?}"),
    }
}

// ----- MOE-01: router differential ------------------------------------------

fn check_router(cuda: &CudaBackend, cpu: &CpuBackend, seq: usize, n_experts: usize, top_k: usize, seed: u64) {
    let logits = deterministic_floats(seq * n_experts, 4.0, seed);
    let logits_dev = cuda.to_device(Tensor::from_vec(logits.clone(), vec![seq, n_experts]));
    let logits_cpu = Tensor::from_vec(logits, vec![seq, n_experts]);

    let k = top_k.min(n_experts);
    let (ids_g, w_g) = cuda.moe_route_topk(&logits_dev, top_k);
    let (ids_c, w_c) = cpu.moe_route_topk(&logits_cpu, top_k);

    assert_eq!(ids_g.len(), seq * k);
    // Top-k SET and ORDER must match the CPU reference exactly (random
    // logits make exact f32 ties a non-issue).
    assert_eq!(ids_g, ids_c, "routed expert ids differ (seq={seq}, n_experts={n_experts}, k={k})");
    for (i, (a, b)) in w_g.iter().zip(w_c.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6,
            "routing weight {i} differs: gpu={a} cpu={b} (>1e-6)"
        );
    }
}

#[test]
fn moe_router_matches_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();
    // Qwen3-30B shape: 128 experts, top-8, multi-token (prefill) rows.
    check_router(&cuda, &cpu, 3, 128, 8, 9001);
    // Mixtral shape: 8 experts, top-2.
    check_router(&cuda, &cpu, 1, 8, 2, 9002);
    // top_k clamp: k > n_experts routes every expert.
    check_router(&cuda, &cpu, 2, 8, 16, 9003);
    // Single expert.
    check_router(&cuda, &cpu, 1, 1, 1, 9004);
}

// ----- MOE-02: grouped kernels vs reference composition ---------------------

struct ExpertSet {
    gu: Vec<QuantizedTensor>,
    dn: Vec<QuantizedTensor>,
    gu_ptrs: Vec<u64>,
    dn_ptrs: Vec<u64>,
}

fn build_experts(
    cuda: &CudaBackend,
    n_experts: usize,
    hidden: usize,
    ff: usize,
    gu_dt: GgmlType,
    dn_dt: GgmlType,
    seed: u64,
) -> ExpertSet {
    let mut gu = Vec::new();
    let mut dn = Vec::new();
    let mut gu_ptrs = Vec::new();
    let mut dn_ptrs = Vec::new();
    for e in 0..n_experts {
        let gu_bytes = make_random_weight_bytes(gu_dt, 2 * ff, hidden, seed + e as u64);
        let dn_bytes = make_random_weight_bytes(dn_dt, hidden, ff, seed + 1000 + e as u64);
        let gu_t = cuda.upload_quantized(&gu_bytes, vec![2 * ff, hidden], gu_dt);
        let dn_t = cuda.upload_quantized(&dn_bytes, vec![hidden, ff], dn_dt);
        gu_ptrs.push(quant_device_ptr(&gu_t).expect("gu on device"));
        dn_ptrs.push(quant_device_ptr(&dn_t).expect("dn on device"));
        gu.push(gu_t);
        dn.push(dn_t);
    }
    ExpertSet { gu, dn, gu_ptrs, dn_ptrs }
}

/// Reference composition: the pre-MOE-02 per-expert op chain (linear_q →
/// act_mul_split → linear_q → add_to_axis0_range_scaled), driven by the same
/// device routing the grouped call used.
#[allow(clippy::too_many_arguments)]
fn reference_moe(
    cuda: &CudaBackend,
    experts: &ExpertSet,
    x: &Tensor,
    ids: &[u32],
    weights: &[f32],
    ff: usize,
    hidden: usize,
    use_gelu: bool,
    scales: Option<&[f32]>,
) -> Tensor {
    let mut out = cuda.alloc_zeros(vec![1, hidden]);
    for (slot, &e) in ids.iter().enumerate() {
        let fused = cuda.linear_q(x, &experts.gu[e as usize]);
        let act = if use_gelu {
            cuda.gelu_approx_mul_split(&fused, ff)
        } else {
            cuda.silu_mul_split(&fused, ff)
        };
        let eo = cuda.linear_q(&act, &experts.dn[e as usize]);
        let w = match scales {
            Some(s) => weights[slot] * s[e as usize],
            None => weights[slot],
        };
        cuda.add_to_axis0_range_scaled(&mut out, 0, 1, &eo, w);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check_grouped(
    cuda: &CudaBackend,
    hidden: usize,
    ff: usize,
    gu_dt: GgmlType,
    dn_dt: GgmlType,
    use_gelu: bool,
    with_scales: bool,
    eps: f32,
    seed: u64,
) {
    let n_experts = 6;
    let top_k = 4;
    let experts = build_experts(cuda, n_experts, hidden, ff, gu_dt, dn_dt, seed);
    let x = cuda.to_device(Tensor::from_vec(
        deterministic_floats(hidden, 1.0, seed + 5000),
        vec![1, hidden],
    ));
    let logits = cuda.to_device(Tensor::from_vec(
        deterministic_floats(n_experts, 4.0, seed + 6000),
        vec![1, n_experts],
    ));
    let routing = cuda.moe_route_device(&logits, top_k).expect("route on device");
    let scales: Option<Vec<f32>> =
        with_scales.then(|| deterministic_floats(n_experts, 1.0, seed + 7000));

    let plan = MoeDevicePlan::new(
        cuda,
        &experts.gu_ptrs,
        &experts.dn_ptrs,
        scales.as_deref(),
        gu_dt,
        dn_dt,
        ff,
        hidden,
        use_gelu,
    );
    let grouped = cuda.moe_grouped_ffn(&x, &plan, &routing);

    let ids = routing.ids_to_host();
    let weights = routing.weights_to_host();
    let reference = reference_moe(
        cuda, &experts, &x, &ids, &weights, ff, hidden, use_gelu, scales.as_deref(),
    );

    assert_tensors_close(&grouped, &reference, eps);
}

#[test]
fn moe_grouped_matches_reference_contiguous() {
    let Some(cuda) = try_cuda() else { return };
    // Every inner dim here keeps the reference linear_q on its coop GEMV,
    // which the grouped kernels replicate exactly — the intermediates are
    // bit-identical. The residual difference is last-ulp FMA contraction in
    // the final accumulate: the reference's add_to_axis0_range_scaled fuses
    // `dst += src*scale` into one rounding, the grouped path rounds the
    // product into the per-slot partial first. With these adversarial
    // random weights (heavy cancellation) that's a few ulps of the largest
    // partial, ~1e-4 absolute.
    //
    // Qwen3-30B mix: Q4_K gate_up + Q6_K down (some layers).
    check_grouped(&cuda, 1024, 1024, GgmlType::Q4_K, GgmlType::Q6_K, false, false, 1e-3, 8101);
    // Mixtral: Q4_K everywhere.
    check_grouped(&cuda, 1024, 1024, GgmlType::Q4_K, GgmlType::Q4_K, false, false, 1e-3, 8102);
    // Q8_0 experts (dense-quant MoE dumps).
    check_grouped(&cuda, 1024, 1024, GgmlType::Q8_0, GgmlType::Q8_0, false, false, 1e-3, 8103);
}

#[test]
fn moe_grouped_matches_reference_strided_down() {
    let Some(cuda) = try_cuda() else { return };
    // Qwen3-30B geometry (scaled down): hidden % 1024 == 0 keeps gate_up
    // bit-exact, but the down inner dim (768) puts the Q4_K grouped kernel
    // on its deterministic strided partition — small reassociation vs the
    // reference's serial per-thread kernel, hence the looser tolerance.
    check_grouped(&cuda, 1024, 768, GgmlType::Q4_K, GgmlType::Q4_K, false, false, 2e-3, 8201);
    // Same geometry with a Q6_K down stays coop on both sides (only the
    // accumulate's FMA contraction differs — see the contiguous test).
    check_grouped(&cuda, 1024, 768, GgmlType::Q4_K, GgmlType::Q6_K, false, false, 1e-3, 8202);
}

#[test]
fn moe_grouped_matches_reference_gelu_scales() {
    let Some(cuda) = try_cuda() else { return };
    // Gemma 4 MoE flavor: GeGLU activation + per-expert down scales.
    check_grouped(&cuda, 1024, 1024, GgmlType::Q4_K, GgmlType::Q4_K, true, true, 1e-3, 8301);
}

// ----- SAMPLE-01: argmax contract ---------------------------------------------

#[test]
fn argmax_last_row_and_ties_match_cpu() {
    let Some(cuda) = try_cuda() else { return };
    let cpu = CpuBackend::new();

    // [2, 97] with a deliberate exact tie in each row (earliest index must
    // win on both backends, matching the CPU greedy sampler).
    let mut data = deterministic_floats(2 * 97, 2.0, 9501);
    data[10] = 5.0;
    data[40] = 5.0; // tie in row 0 — earliest (10) wins
    data[97 + 3] = 7.0;
    data[97 + 80] = 7.0; // tie in row 1 — earliest (3) wins
    let x_cpu = Tensor::from_vec(data.clone(), vec![2, 97]);
    let x_dev = cuda.to_device(Tensor::from_vec(data, vec![2, 97]));

    assert_eq!(cuda.argmax_last(&x_dev), cpu.argmax_last(&x_cpu));
    assert_eq!(cuda.argmax_last(&x_dev), vec![10, 3]);
}
