//! The chain's kernels against the CPU's ops and the host's formulas, by what they compute; the timings apart.
use super::*;
use ggml_quants::GgmlType;

fn rng(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 2001) as f32 / 1000.0 - 1.0
    }
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 1e-4 * scale, "{what} [{i}]: {x} against {y}");
    }
}

mod attention;
mod llm;
mod matmul;
mod measure;
mod media;
