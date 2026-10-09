//! The small functions of the pass: the activations, a matrix by a vector on the host, the norms, and the
//! clamped SwiGLU.

use super::*;

pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `out = w @ x` for `w` of `[out_dim, in_dim]` row-major.
pub(crate) fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) -> Result<()> {
    let n = x.len();
    if w.len() != out.len() * n {
        return Err(LlamaError::Config(format!(
            "forward: matvec weight is {} for [{}, {n}]",
            w.len(),
            out.len()
        )));
    }
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    Ok(())
}

/// RMSNorm with a learned scale, in place.
pub(super) fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) -> Result<()> {
    if w.len() != x.len() {
        return Err(LlamaError::Config(format!(
            "forward: rms_norm weight is {} for a {}-wide vector",
            w.len(),
            x.len()
        )));
    }
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (v, &s) in x.iter_mut().zip(w) {
        *v = *v * inv * s;
    }
    Ok(())
}

/// LayerNorm with scale **and bias** — the indexer's `k_norm`, and the only
/// non-RMS norm in this architecture.
pub(super) fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) -> Result<()> {
    if w.len() != x.len() || b.len() != x.len() {
        return Err(LlamaError::Config(format!(
            "forward: layer_norm weight/bias are {} / {} for a {}-wide vector",
            w.len(),
            b.len(),
            x.len()
        )));
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for ((v, &s), &bb) in x.iter_mut().zip(w).zip(b) {
        *v = (*v - mean) * inv * s + bb;
    }
    Ok(())
}

/// L2-normalise in place: `x / sqrt(sum(x^2) + eps)`. The KDA path applies this
/// per head at the reference's own hardcoded `1e-6`, not the model's norm eps.
pub(super) fn l2_norm(x: &mut [f32], eps: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The clamped SwiGLU, per `build_ffn`'s generic branch:
/// `clamp(silu(gate), -inf, L) * clamp(up, -L, L)`.
///
/// `limit <= 1e-6` disables the clamp, matching the reference's `limit > eps`
/// guard, and gives a plain SwiGLU.
///
/// Note the clamp is on the **activated** gate, after the SiLU, and is
/// one-sided; `up` is clamped symmetrically.
pub fn swiglu_clamped(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || out.len() != gate.len() {
        return Err(LlamaError::Config(format!(
            "forward: swiglu widths {} / {} / {}",
            gate.len(),
            up.len(),
            out.len()
        )));
    }
    let clamped = limit > 1e-6;
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        let mut a = silu(g);
        let mut b = u;
        if clamped {
            if a > limit {
                a = limit;
            }
            b = b.clamp(-limit, limit);
        }
        *o = a * b;
    }
    Ok(())
}
