//! The CPU half of hybrid decode: which row kernel `dsv41::cpu_experts`
//! runs on this machine.
//!
//! `dsv41` defines an AVX-512 kernel as a safe `#[target_feature]`
//! function but cannot call it (the crate forbids `unsafe`, and calling such
//! a function from code without those features is `unsafe`). The call lives
//! here, behind a runtime feature check.

use dsv41::cpu_experts::{fp4_rows, RowKernel};

/// The fastest row kernel this CPU supports; every choice gives the same bits.
pub fn row_kernel() -> RowKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") {
        return fp4_rows_avx512;
    }
    fp4_rows
}

/// Name of [`row_kernel`]'s choice, for logs.
pub fn row_kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") {
        return "avx512";
    }
    "portable"
}

#[cfg(target_arch = "x86_64")]
fn fp4_rows_avx512(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
    // SAFETY: a `#[target_feature(enable = "avx512f,avx512bw")]` function may
    // only run on a CPU with those features. This wrapper is private and only
    // handed out by `row_kernel` after `is_x86_feature_detected!` confirmed
    // both; the callee's body is safe code (bounds-checked slices, value-only
    // intrinsics), so nothing else is assumed.
    unsafe { dsv41::cpu_experts::avx512::fp4_rows(x, w, s, k, r0, out) }
}
