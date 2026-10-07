//! Which row kernel `dsv41::cpu_experts` runs on this machine.
//!
//! `dsv41` defines an AVX-512 kernel as a safe `#[target_feature]` function but cannot call it: the crate forbids
//! `unsafe`, and calling such a function from code without those features is `unsafe` (the CPU must have them). The
//! calls live here, each behind a runtime feature check, and nowhere else. They were the CUDA engine's crate's while
//! there was one; with it gone the routed experts on the CPU ran the portable kernel on every machine until a caller
//! asked this crate ([`row_kernel`]) and told the model (`dsv41::model::Model::set_expert_row_kernel`).

use dsv41::cpu_experts::{fp4_rows, RowKernel};

#[cfg(target_arch = "x86_64")]
fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw")
}

/// The fastest row kernel this CPU supports; every choice gives the same bits.
pub fn row_kernel() -> RowKernel {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        return fp4_rows_avx512;
    }
    fp4_rows
}

/// Name of [`row_kernel`]'s choice, for logs.
pub fn row_kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        return "avx512";
    }
    "portable"
}

#[cfg(target_arch = "x86_64")]
fn fp4_rows_avx512(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
    // SAFETY: a `#[target_feature(enable = "avx512f,avx512bw")]` function may only run on a CPU with those features.
    // This wrapper is private and only handed out by `row_kernel` after `is_x86_feature_detected!` confirmed both;
    // the callee's body is safe code (bounds-checked slices, value-only intrinsics), so nothing else is assumed.
    unsafe { dsv41::cpu_experts::avx512::fp4_rows(x, w, s, k, r0, out) }
}

/// [`row_kernel`] for ternary records (`dsv41::ternary`).
pub fn ternary_row_kernel() -> RowKernel {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        return ternary_avx512;
    }
    dsv41::ternary::rows
}

#[cfg(target_arch = "x86_64")]
fn ternary_avx512(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
    // SAFETY: as `fp4_rows_avx512`: this private wrapper is returned only after both CPU features were checked.
    unsafe { dsv41::cpu_experts::avx512::ternary_rows(x, w, s, k, r0, out) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsv41::cpu_experts::CpuExperts;
    use dsv41::expert::{expert_forward, expert_forward_batch, expert_forward_rows, DIM, RECORD_BYTES, S1, S2, S3};
    use dsv41::formats::to_bf16;
    use std::sync::Arc;

    fn record(seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut rec: Vec<u8> = (0..RECORD_BYTES)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect();
        for r in [S1, S2, S3] {
            for (i, b) in rec[r].iter_mut().enumerate() {
                *b = 120 + ((i as u64 + seed) % 6) as u8;
            }
        }
        rec
    }

    fn activation(rows: usize, mut s: u64) -> Vec<f32> {
        (0..rows * DIM)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                to_bf16(((s >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 8.0)
            })
            .collect()
    }

    /// This CPU's kernel gives the portable one's bits: a decode step's experts through the pool, and a prompt's
    /// expert of a few rows a row at a time.
    #[test]
    fn this_cpus_kernel_gives_the_portable_kernels_bits() {
        eprintln!("this CPU's row kernel: {}", row_kernel_name());
        let recs: Vec<Arc<Vec<u8>>> = (0..3).map(|e| Arc::new(record(40 + e))).collect();
        let x = activation(1, 7);
        let weights = [0.4f32, 1.1, 0.25];
        for threads in [1, 5, 16] {
            let got = CpuExperts::with_kernel(threads, row_kernel()).forward(&recs, &weights, &x, 10.0);
            for (e, rec) in recs.iter().enumerate() {
                let want = expert_forward(rec, &x, Some(weights[e]), 10.0);
                assert!(got[e].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "expert {e}, {threads} threads");
            }
        }
        let x3 = activation(3, 11);
        let want = expert_forward_batch(&recs[0], &x3, Some(&weights), 10.0);
        let got = expert_forward_rows(row_kernel(), &recs[0], &x3, Some(&weights), 10.0);
        assert!(got.len() == want.len() && got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "a prompt's rows");
    }
}
