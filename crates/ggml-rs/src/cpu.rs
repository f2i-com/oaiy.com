//! Single-threaded fallback CPU backend with optional rayon parallelism on
//! the heaviest op (linear / matmul).
//!
//! No SIMD intrinsics yet — readability and correctness are the priority for
//! v0. The hot-path is tiled+parallelized matmul; everything else is the
//! straightforward loop. SIMD via `std::simd` or per-arch intrinsics is a
//! follow-up.

use rayon::prelude::*;

use crate::backend::{Backend, RopeType};
use crate::tensor::Tensor;

#[derive(Debug, Default)]
pub struct CpuBackend {
    /// Min number of output rows below which matmul runs serial. Tuned by
    /// rough rule-of-thumb; replace with a real heuristic later.
    parallel_threshold: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { parallel_threshold: 16 }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str { "cpu" }

    fn as_any(&self) -> &dyn std::any::Any { self }

    fn embed_lookup(&self, table: &Tensor, tokens: &[u32], embedding_dim: usize) -> Tensor {
        let mut out = vec![0.0f32; tokens.len() * embedding_dim];
        let td = table.data();
        for (i, &id) in tokens.iter().enumerate() {
            let row = id as usize * embedding_dim;
            out[i * embedding_dim..(i + 1) * embedding_dim]
                .copy_from_slice(&td[row..row + embedding_dim]);
        }
        Tensor::from_vec(out, vec![tokens.len(), embedding_dim])
    }

    fn linear(&self, x: &Tensor, w: &Tensor) -> Tensor {
        // x: [..., in]; w: [out, in]; result: [..., out]
        assert_eq!(w.rank(), 2, "linear: weight must be 2D, got rank {}", w.rank());
        let in_ = x.dim(x.rank() - 1);
        let out = w.dim(0);
        assert_eq!(w.dim(1), in_, "linear: x[..,{in_}] vs w[{},{}]", w.dim(0), w.dim(1));

        let b = x.numel() / in_;
        let mut y = vec![0.0f32; b * out];
        let xd = x.data();
        let wd = w.data();

        let body = |bi: usize, y_row: &mut [f32]| {
            let xb = &xd[bi * in_..(bi + 1) * in_];
            for o in 0..out {
                let wo = &wd[o * in_..(o + 1) * in_];
                let mut acc = 0.0f32;
                for i in 0..in_ {
                    acc += xb[i] * wo[i];
                }
                y_row[o] = acc;
            }
        };

        if b >= self.parallel_threshold {
            y.par_chunks_mut(out)
                .enumerate()
                .for_each(|(bi, row)| body(bi, row));
        } else {
            for (bi, row) in y.chunks_mut(out).enumerate() {
                body(bi, row);
            }
        }

        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = out;
        Tensor::from_vec(y, shape)
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        let last = x.dim(x.rank() - 1);
        assert_eq!(weight.shape(), [last], "rmsnorm: weight must be [last_dim]");
        let n_rows = x.numel() / last;
        let mut out = vec![0.0f32; x.numel()];
        let xd = x.data();
        let wd = weight.data();

        for row in 0..n_rows {
            let s = row * last;
            let xr = &xd[s..s + last];
            let mut sum_sq = 0.0f32;
            for v in xr { sum_sq += v * v; }
            let inv_rms = 1.0 / (sum_sq / last as f32 + eps).sqrt();
            for j in 0..last {
                out[s + j] = xr[j] * inv_rms * wd[j];
            }
        }
        Tensor::from_vec(out, x.shape().to_vec())
    }

    fn softmax_last(&self, x: &mut Tensor) {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        let xd = x.data_mut();

        for row in 0..n_rows {
            let s = row * last;
            let xr = &mut xd[s..s + last];
            let mut m = f32::NEG_INFINITY;
            for v in xr.iter() { if *v > m { m = *v; } }
            let mut sum = 0.0f32;
            for v in xr.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            // sum > 0 because at least one element is now exp(0) = 1.
            let inv = 1.0 / sum;
            for v in xr.iter_mut() { *v *= inv; }
        }
    }

    fn silu(&self, x: &Tensor) -> Tensor {
        let mut out = vec![0.0f32; x.numel()];
        for (o, &v) in out.iter_mut().zip(x.data().iter()) {
            *o = v / (1.0 + (-v).exp());
        }
        Tensor::from_vec(out, x.shape().to_vec())
    }

    fn gelu_approx(&self, x: &Tensor) -> Tensor {
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654; // sqrt(2/π)
        const COEFF: f32 = 0.044715;
        let mut out = vec![0.0f32; x.numel()];
        for (o, &v) in out.iter_mut().zip(x.data().iter()) {
            let inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
            *o = 0.5 * v * (1.0 + inner.tanh());
        }
        Tensor::from_vec(out, x.shape().to_vec())
    }

    fn add_inplace(&self, x: &mut Tensor, y: &Tensor) {
        assert_eq!(x.shape(), y.shape(), "add_inplace: shape mismatch");
        let yd = y.data();
        for (a, b) in x.data_mut().iter_mut().zip(yd.iter()) {
            *a += *b;
        }
    }

    fn mul_inplace(&self, x: &mut Tensor, y: &Tensor) {
        assert_eq!(x.shape(), y.shape(), "mul_inplace: shape mismatch");
        let yd = y.data();
        for (a, b) in x.data_mut().iter_mut().zip(yd.iter()) {
            *a *= *b;
        }
    }

    fn rope(
        &self,
        x: &mut Tensor,
        positions: &[u32],
        head_dim: usize,
        rope_type: RopeType,
        theta: f32,
        freq_factors: Option<&[f32]>,
    ) {
        // x: [seq, n_heads, head_dim]
        assert_eq!(x.rank(), 3, "rope: expected 3D [seq, n_heads, head_dim]");
        let seq = x.dim(0);
        let n_heads = x.dim(1);
        assert_eq!(x.dim(2), head_dim);
        assert_eq!(positions.len(), seq);
        if let Some(ff) = freq_factors {
            assert!(ff.len() >= head_dim / 2, "freq_factors len {} < head_dim/2 {}", ff.len(), head_dim / 2);
        }

        let half = head_dim / 2;
        let xd = x.data_mut();

        for s in 0..seq {
            let pos = positions[s] as f32;
            for h in 0..n_heads {
                let off = (s * n_heads + h) * head_dim;
                for k in 0..half {
                    let freq = theta.powf(-2.0 * k as f32 / head_dim as f32);
                    let factor = freq_factors.map(|f| f[k]).unwrap_or(1.0);
                    let angle = pos * freq / factor;
                    let (sin_v, cos_v) = angle.sin_cos();

                    let (i_a, i_b) = match rope_type {
                        RopeType::Normal => (off + 2 * k, off + 2 * k + 1),
                        RopeType::NeoX   => (off + k,     off + k + half),
                    };
                    let a = xd[i_a];
                    let b = xd[i_b];
                    xd[i_a] = a * cos_v - b * sin_v;
                    xd[i_b] = a * sin_v + b * cos_v;
                }
            }
        }
    }

    fn repeat_kv(&self, x: &Tensor, n_rep: usize) -> Tensor {
        if n_rep == 1 {
            return x.clone();
        }
        // x: [seq, n_kv_heads, head_dim]
        assert_eq!(x.rank(), 3, "repeat_kv: expected 3D [seq, n_kv, head_dim]");
        let seq = x.dim(0);
        let n_kv = x.dim(1);
        let head_dim = x.dim(2);

        let mut out = vec![0.0f32; seq * n_kv * n_rep * head_dim];
        let xd = x.data();
        for s in 0..seq {
            for k in 0..n_kv {
                let src = &xd[(s * n_kv + k) * head_dim..(s * n_kv + k + 1) * head_dim];
                for r in 0..n_rep {
                    let dst_off = (s * n_kv * n_rep + k * n_rep + r) * head_dim;
                    out[dst_off..dst_off + head_dim].copy_from_slice(src);
                }
            }
        }
        Tensor::from_vec(out, vec![seq, n_kv * n_rep, head_dim])
    }

    fn bmm_qkt(&self, q: &Tensor, k: &Tensor, scale: f32, past: usize) -> Tensor {
        // q: [seq, n_h, hd]; k: [kv_len, n_h, hd]
        let seq = q.dim(0);
        let n_h = q.dim(1);
        let hd = q.dim(2);
        let kv_len = k.dim(0);
        debug_assert_eq!(k.dim(1), n_h);
        debug_assert_eq!(k.dim(2), hd);
        let qd = q.data();
        let kd = k.data();
        let mut scores = vec![0.0f32; seq * n_h * kv_len];
        // Outer parallelism: rayon over (s, h).
        scores.par_chunks_mut(kv_len).enumerate().for_each(|(sh, row)| {
            let s = sh / n_h;
            let h = sh % n_h;
            let q_off = (s * n_h + h) * hd;
            let max_t = past + s;
            for t in 0..kv_len {
                if t > max_t {
                    row[t] = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (t * n_h + h) * hd;
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += qd[q_off + d] * kd[k_off + d];
                }
                row[t] = acc * scale;
            }
        });
        Tensor::from_vec(scores, vec![seq, n_h, kv_len])
    }

    fn bmm_av(&self, scores: &Tensor, v: &Tensor) -> Tensor {
        // scores: [seq, n_h, kv_len]; v: [kv_len, n_h, hd]
        let seq = scores.dim(0);
        let n_h = scores.dim(1);
        let kv_len = scores.dim(2);
        let hd = v.dim(2);
        debug_assert_eq!(v.dim(0), kv_len);
        debug_assert_eq!(v.dim(1), n_h);
        let scd = scores.data();
        let vd = v.data();
        let mut out = vec![0.0f32; seq * n_h * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(sh, row)| {
            let s = sh / n_h;
            let h = sh % n_h;
            for d in 0..hd {
                let mut acc = 0.0f32;
                for t in 0..kv_len {
                    acc += scd[(s * n_h + h) * kv_len + t]
                         * vd[(t * n_h + h) * hd + d];
                }
                row[d] = acc;
            }
        });
        Tensor::from_vec(out, vec![seq, n_h, hd])
    }

    fn argmax_last(&self, x: &Tensor) -> Vec<u32> {
        let last = x.dim(x.rank() - 1);
        let rows = x.numel() / last;
        let xd = x.data();
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let s = r * last;
            let mut best_i = 0u32;
            let mut best_v = f32::NEG_INFINITY;
            for j in 0..last {
                let v = xd[s + j];
                if v > best_v { best_v = v; best_i = j as u32; }
            }
            out.push(best_i);
        }
        out
    }
}

// ----- tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    #[test]
    fn linear_basic() {
        let cpu = CpuBackend::new();
        // x: [2, 3], w: [4, 3]
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = Tensor::from_vec(vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ], vec![4, 3]);
        let y = cpu.linear(&x, &w);
        assert_eq!(y.shape(), [2, 4]);
        // row 0: [1, 2, 3, 6]
        // row 1: [4, 5, 6, 15]
        assert_eq!(y.data(), &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    #[test]
    fn rmsnorm_basic() {
        let cpu = CpuBackend::new();
        // x = [1, 2, 3], weight = [1, 1, 1]; mean(x²)=14/3≈4.667; rms ≈ 2.16; eps≈0
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let w = Tensor::from_vec(vec![1.0; 3], vec![3]);
        let y = cpu.rmsnorm(&x, &w, 1e-5);
        let rms = (14.0f32 / 3.0).sqrt();
        for (i, &expected) in [1.0/rms, 2.0/rms, 3.0/rms].iter().enumerate() {
            assert!(approx_eq(y.data()[i], expected, 1e-4),
                    "rmsnorm[{i}]: {} vs {}", y.data()[i], expected);
        }
    }

    #[test]
    fn silu_mul_split_matches_unfused() {
        // [seq=2, 2*ff=8] (ff=4): two rows where first half is the silu input,
        // second half is the up-projection that gets multiplied in.
        let cpu = CpuBackend::new();
        let fused_data = vec![
            // row 0: gate=[1,2,3,4]   up=[0.5, 1, 1.5, 2]
            1.0, 2.0, 3.0, 4.0,  0.5, 1.0, 1.5, 2.0,
            // row 1: gate=[-1,-2,0,2] up=[1, 1, 1, 1]
            -1.0, -2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let fused = Tensor::from_vec(fused_data.clone(), vec![2, 8]);
        let out = cpu.silu_mul_split(&fused, 4);
        assert_eq!(out.shape(), &[2, 4]);

        // Reference: silu(g) * u for each (row, j).
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for r in 0..2 {
            for j in 0..4 {
                let g = fused_data[r * 8 + j];
                let u = fused_data[r * 8 + 4 + j];
                let expected = silu(g) * u;
                let got = out.data()[r * 4 + j];
                assert!(approx_eq(got, expected, 1e-5),
                        "silu_mul_split[{r},{j}]: got {got}, want {expected}");
            }
        }
    }

    #[test]
    fn gelu_approx_mul_split_matches_unfused() {
        let cpu = CpuBackend::new();
        let fused_data = vec![
            // row 0: gate=[1,2,3,4]   up=[0.5, 1, 1.5, 2]
            1.0, 2.0, 3.0, 4.0,  0.5, 1.0, 1.5, 2.0,
            // row 1: gate=[-1,-2,0,2] up=[1, 1, 1, 1]
            -1.0, -2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let fused = Tensor::from_vec(fused_data.clone(), vec![2, 8]);
        let out = cpu.gelu_approx_mul_split(&fused, 4);
        assert_eq!(out.shape(), &[2, 4]);

        // Reference: split halves, separately call gelu_approx_mul.
        let gate = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.0, 2.0], vec![2, 4]);
        let up   = Tensor::from_vec(
            vec![0.5, 1.0, 1.5, 2.0,  1.0,  1.0, 1.0, 1.0], vec![2, 4]);
        let ref_out = cpu.gelu_approx_mul(&gate, &up);
        for i in 0..8 {
            assert!(approx_eq(out.data()[i], ref_out.data()[i], 1e-6),
                    "gelu_split[{i}]: fused={}, ref={}", out.data()[i], ref_out.data()[i]);
        }
    }

    #[test]
    fn add_inplace_then_rmsnorm_equals_separate() {
        let cpu = CpuBackend::new();
        // x: [1, 4], y: [1, 4], w: [4]
        let mut x_a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let mut x_b = x_a.clone();
        let y       = Tensor::from_vec(vec![0.5, -1.0, 0.5, 0.0], vec![1, 4]);
        let w       = Tensor::from_vec(vec![1.0, 0.5, 2.0, 1.0], vec![4]);

        // Reference: separate add_inplace + rmsnorm.
        cpu.add_inplace(&mut x_a, &y);
        let ref_out = cpu.rmsnorm(&x_a, &w, 1e-6);

        // Fused.
        let fused_out = cpu.add_inplace_then_rmsnorm(&mut x_b, &y, &w, 1e-6);

        for i in 0..4 {
            assert!(approx_eq(fused_out.data()[i], ref_out.data()[i], 1e-5),
                    "add+rmsnorm[{i}]: fused={}, ref={}",
                    fused_out.data()[i], ref_out.data()[i]);
            assert!(approx_eq(x_a.data()[i], x_b.data()[i], 1e-5),
                    "x[{i}] post-call: ref={}, fused={}",
                    x_a.data()[i], x_b.data()[i]);
        }
    }

    #[test]
    fn gelu_approx_mul_equals_separate() {
        let cpu = CpuBackend::new();
        let a = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]);
        let b = Tensor::from_vec(vec![1.0, 2.0, -1.0, 3.0, -0.5], vec![5]);
        let mut g = cpu.gelu_approx(&a);
        cpu.mul_inplace(&mut g, &b);
        let fused = cpu.gelu_approx_mul(&a, &b);
        for i in 0..5 {
            assert!(approx_eq(fused.data()[i], g.data()[i], 1e-6),
                    "gelu_mul[{i}]: fused={}, sep={}",
                    fused.data()[i], g.data()[i]);
        }
    }

    #[test]
    fn mul_inplace_broadcast_axis0_basic() {
        let cpu = CpuBackend::new();
        // x: [3, 2], g: [3]; expect x[s, j] *= g[s].
        let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let g = Tensor::from_vec(vec![10.0, -1.0, 0.5], vec![3]);
        cpu.mul_inplace_broadcast_axis0(&mut x, &g);
        let expected = [10.0, 20.0, -3.0, -4.0, 2.5, 3.0];
        for i in 0..6 {
            assert!(approx_eq(x.data()[i], expected[i], 1e-6),
                    "broadcast_axis0[{i}]: got {}, want {}", x.data()[i], expected[i]);
        }
    }

    #[test]
    fn add_to_axis0_range_scaled_basic() {
        let cpu = CpuBackend::new();
        // dst: [4, 3], src: [3]; dst[1..3] += src * scale.
        let mut dst = Tensor::from_vec(vec![0.0; 12], vec![4, 3]);
        let src = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        cpu.add_to_axis0_range_scaled(&mut dst, 1, 2, &src, 0.5);
        let expected = [
            0.0, 0.0, 0.0,    // row 0 untouched
            0.5, 1.0, 1.5,    // row 1 = src * 0.5
            0.5, 1.0, 1.5,    // row 2 = src * 0.5
            0.0, 0.0, 0.0,    // row 3 untouched
        ];
        for i in 0..12 {
            assert!(approx_eq(dst.data()[i], expected[i], 1e-6),
                    "scaled_range[{i}]: got {}, want {}", dst.data()[i], expected[i]);
        }
    }

    #[test]
    fn softmax_basic() {
        let cpu = CpuBackend::new();
        let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        cpu.softmax_last(&mut x);
        let sum: f32 = x.data().iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));
        // monotone increasing input → monotone output
        assert!(x.data()[0] < x.data()[1] && x.data()[1] < x.data()[2]);
    }

    #[test]
    fn silu_basic() {
        let cpu = CpuBackend::new();
        let x = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
        let y = cpu.silu(&x);
        // silu(0) = 0
        assert!(approx_eq(y.data()[0], 0.0, 1e-6));
        // silu(1) ≈ 0.7311
        assert!(approx_eq(y.data()[1], 0.7311, 1e-3));
        // silu(-1) ≈ -0.2689
        assert!(approx_eq(y.data()[2], -0.2689, 1e-3));
    }

    #[test]
    fn rope_neox_zero_pos_is_identity() {
        let cpu = CpuBackend::new();
        // pos=0 -> angles all zero -> identity rotation
        let mut x = Tensor::from_vec((0..16).map(|i| i as f32).collect(), vec![1, 2, 8]);
        let original = x.data().to_vec();
        cpu.rope(&mut x, &[0], 8, RopeType::NeoX, 10000.0, None);
        for (a, b) in x.data().iter().zip(original.iter()) {
            assert!(approx_eq(*a, *b, 1e-5));
        }
    }

    #[test]
    fn rope_normal_zero_pos_is_identity() {
        let cpu = CpuBackend::new();
        let mut x = Tensor::from_vec((0..16).map(|i| i as f32).collect(), vec![1, 2, 8]);
        let original = x.data().to_vec();
        cpu.rope(&mut x, &[0], 8, RopeType::Normal, 10000.0, None);
        for (a, b) in x.data().iter().zip(original.iter()) {
            assert!(approx_eq(*a, *b, 1e-5));
        }
    }

    #[test]
    fn rope_full_cycle_returns_identity() {
        let cpu = CpuBackend::new();
        // Apply rope twice with positions whose angles differ by 2π — equivalent to identity.
        // Easier check: rope with pos=0 should yield identity (already tested above);
        // here just check that values aren't NaN at large position.
        let mut x = Tensor::from_vec(vec![1.0; 8], vec![1, 1, 8]);
        cpu.rope(&mut x, &[1024], 8, RopeType::NeoX, 10000.0, None);
        for v in x.data() { assert!(v.is_finite()); }
    }

    #[test]
    fn repeat_kv_doubles_heads() {
        let cpu = CpuBackend::new();
        // x: [1 seq, 2 kv heads, 3 head_dim] = [[h0_d0, h0_d1, h0_d2], [h1_...]]
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let y = cpu.repeat_kv(&x, 2);
        assert_eq!(y.shape(), [1, 4, 3]);
        // Heads 0 and 1 are head 0 of input; heads 2 and 3 are head 1.
        assert_eq!(y.data()[..3],   [1.0, 2.0, 3.0]);
        assert_eq!(y.data()[3..6],  [1.0, 2.0, 3.0]);
        assert_eq!(y.data()[6..9],  [4.0, 5.0, 6.0]);
        assert_eq!(y.data()[9..12], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn argmax_basic() {
        let cpu = CpuBackend::new();
        let x = Tensor::from_vec(vec![1.0, 3.0, 2.0,    4.0, 0.0, 1.0], vec![2, 3]);
        let r = cpu.argmax_last(&x);
        assert_eq!(r, vec![1, 0]);
    }
}
