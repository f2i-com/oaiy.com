//! The delta-net step on the host (`Backend`'s default, which the CPU and WebGPU backends take) against the CUDA
//! kernels, both gates: silu (Qwen3.5) and sigmoid (Qwen3.8-Flash-Next), over several tokens so the conv and
//! recurrent state carry from one to the next.
use ggml_rs::{Backend, CpuBackend, Tensor};
use ggml_rs_cuda::CudaBackend;

fn noise(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.to_host().data().iter().zip(b.to_host().data()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn the_host_delta_net_step_matches_the_cuda_kernels_for_both_gates() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("[skipping] no CUDA device");
        return;
    };
    let cpu = CpuBackend::new();
    let (seq, nk, nv, kd, vd, kernel) = (3, 2, 4, 128, 128, 4);
    let conv_dim = 2 * nk * kd + nv * vd;
    let t = |v: Vec<f32>, shape: Vec<usize>| Tensor::from_vec(v, shape);
    let qkv = t(noise(seq * conv_dim, 1, 2.0), vec![seq, conv_dim]);
    let z = t(noise(seq * nv * vd, 2, 3.0), vec![seq, nv * vd]);
    let ba = t(noise(seq * 2 * nv, 3, 2.0), vec![seq, 2 * nv]);
    let conv_w = t(noise(conv_dim * kernel, 4, 1.0), vec![conv_dim, kernel]);
    let a = t(noise(nv, 5, 1.0).iter().map(|v| -v.abs() - 0.1).collect(), vec![nv]);
    let dt = t(noise(nv, 6, 1.0), vec![nv]);
    let norm = t(noise(vd, 7, 0.5).iter().map(|v| 1.0 + v).collect(), vec![vd]);
    let conv0 = t(noise((kernel - 1) * conv_dim, 8, 1.0), vec![kernel - 1, conv_dim]);
    let state0 = t(noise(nv * vd * vd, 9, 0.2), vec![nv, vd, vd]);
    let scale = 1.0 / (kd as f32).sqrt();
    for sigmoid in [false, true] {
        let run = |b: &dyn Backend| {
            let mut conv = b.to_device(conv0.clone());
            let mut state = b.to_device(state0.clone());
            let args = (b.to_device(qkv.clone()), b.to_device(z.clone()), b.to_device(ba.clone()), b.to_device(conv_w.clone()), b.to_device(a.clone()), b.to_device(dt.clone()), b.to_device(norm.clone()));
            let out = if sigmoid {
                b.delta_net_step_sigmoid(&args.0, &args.1, &args.2, &args.3, &args.4, &args.5, &args.6, &mut conv, &mut state, seq, nv, nk, vd, kd, nv / nk, scale, 1e-6)
            } else {
                b.delta_net_step(&args.0, &args.1, &args.2, &args.3, &args.4, &args.5, &args.6, &mut conv, &mut state, seq, nv, nk, vd, kd, nv / nk, scale, 1e-6)
            };
            (out.to_host(), conv.to_host(), state.to_host())
        };
        let (host, device) = (run(&cpu), run(&cuda));
        let gate = if sigmoid { "sigmoid" } else { "silu" };
        assert!(max_diff(&host.0, &device.0) < 1e-3, "{gate}: output differs by {}", max_diff(&host.0, &device.0));
        assert!(max_diff(&host.1, &device.1) < 1e-5, "{gate}: conv state differs by {}", max_diff(&host.1, &device.1));
        assert!(max_diff(&host.2, &device.2) < 1e-3, "{gate}: recurrent state differs by {}", max_diff(&host.2, &device.2));
        assert!(host.0.data().iter().any(|v| v.abs() > 1e-3), "{gate}: the output is not trivially zero");
    }
}
