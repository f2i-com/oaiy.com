//! Which row kernel `cpu_experts` runs on this machine.
//!
//! [`cpu_experts::avx512`](crate::cpu_experts::avx512) is written as safe `#[target_feature]` functions, and calling
//! one from code that does not have those features is `unsafe`: the CPU must have them. The calls are here, each
//! behind `is_x86_feature_detected!`, and nowhere else in the crate (it denies `unsafe`; this module alone allows
//! it). They were the CUDA engine's crate's while there was one; without them the routed experts on the CPU ran the
//! portable kernel on every machine.
#![allow(unsafe_code)]

use crate::cpu_experts::{fp4_rows, RowKernel};

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
    unsafe { crate::cpu_experts::avx512::fp4_rows(x, w, s, k, r0, out) }
}

/// [`row_kernel`] for ternary records ([`crate::ternary`]).
pub fn ternary_row_kernel() -> RowKernel {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        return ternary_avx512;
    }
    crate::ternary::rows
}

#[cfg(target_arch = "x86_64")]
fn ternary_avx512(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
    // SAFETY: as `fp4_rows_avx512`: this private wrapper is returned only after both CPU features were checked.
    unsafe { crate::cpu_experts::avx512::ternary_rows(x, w, s, k, r0, out) }
}
