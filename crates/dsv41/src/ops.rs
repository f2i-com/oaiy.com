//! Elementwise and per-row ops with the reference's dtype behaviour: compute
//! in f32, round to bf16 exactly where the reference's tensors are bf16.

use crate::formats::to_bf16;

/// Reference `RMSNorm`: `bf16(w * (x * rsqrt(mean(x^2) + eps)))` per row of `d`.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let var = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let r = 1.0 / (var + eps).sqrt();
        out.extend(row.iter().zip(w).map(|(v, g)| to_bf16(g * (v * r))));
    }
    out
}

/// torch `F.softplus` (beta 1, threshold 20).
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Round every element through bf16 in place.
pub fn round_bf16(x: &mut [f32]) {
    for v in x {
        *v = to_bf16(*v);
    }
}

/// Rotary tables for one attention flavour: `cos/sin[pos][i]` for the
/// `rope_head_dim / 2` frequency pairs, built as the reference
/// `precompute_freqs_cis` (YaRN when `original_seq_len > 0`).
pub struct Rope {
    pub half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(dim: usize, max_pos: usize, original_seq_len: usize, base: f32, factor: f32, beta_fast: f32, beta_slow: f32) -> Rope {
        let half = dim / 2;
        let mut freqs: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32)).collect();
        if original_seq_len > 0 {
            let corrected = |rot: f64| {
                dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * (base as f64).ln())
            };
            let low = corrected(beta_fast as f64).floor().max(0.0);
            let high = corrected(beta_slow as f64).ceil().min(dim as f64 - 1.0);
            let span = (high - low).max(1e-3) as f32;
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = *f / factor * (1.0 - smooth) + *f * smooth;
            }
        }
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for p in 0..max_pos {
            for &f in &freqs {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Rope { half, cos, sin }
    }

    pub fn max_pos(&self) -> usize {
        self.cos.len() / self.half
    }

    /// `(cos, sin)`, each `[max_pos][half]` — for uploading to a device.
    pub fn tables(&self) -> (&[f32], &[f32]) {
        (&self.cos, &self.sin)
    }

    /// Rotate the adjacent pairs of `x` (length `2 * half`) to position
    /// `pos`, or back from it (`inverse`: conjugate), rounding to bf16 like
    /// the reference's in-place copy into a bf16 tensor.
    pub fn apply(&self, x: &mut [f32], pos: usize, inverse: bool) {
        let (c, s) = (&self.cos[pos * self.half..(pos + 1) * self.half], &self.sin[pos * self.half..(pos + 1) * self.half]);
        for i in 0..self.half {
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            let (ci, si) = (c[i], if inverse { -s[i] } else { s[i] });
            x[2 * i] = to_bf16(a * ci - b * si);
            x[2 * i + 1] = to_bf16(a * si + b * ci);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_inverse_undoes_rotation_up_to_bf16() {
        let r = Rope::new(64, 16, 0, 10000.0, 1.0, 32.0, 1.0);
        let orig: Vec<f32> = (0..64).map(|i| to_bf16((i as f32 - 31.5) / 8.0)).collect();
        let mut x = orig.clone();
        r.apply(&mut x, 7, false);
        r.apply(&mut x, 7, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() <= b.abs() * 1e-2 + 1e-2, "{a} vs {b}");
        }
        // position 0 is the identity
        let mut y = orig.clone();
        r.apply(&mut y, 0, false);
        assert_eq!(y, orig);
    }

    #[test]
    fn yarn_leaves_high_frequencies_alone() {
        let plain = Rope::new(64, 2, 0, 160000.0, 16.0, 32.0, 1.0);
        let yarn = Rope::new(64, 2, 65536, 160000.0, 16.0, 32.0, 1.0);
        // pair 0 is the fastest-rotating: inside the "keep" band
        assert_eq!(plain.sin[plain.half], yarn.sin[yarn.half]);
        // the slowest pair is divided by the factor
        assert!(yarn.sin[2 * yarn.half - 1] < plain.sin[2 * plain.half - 1]);
    }
}
