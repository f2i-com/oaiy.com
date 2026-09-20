// VENDORED-LOCAL: whole module. GLM-5.3-Flash KDA recurrence.
//! Kimi Delta Attention: the delta-rule recurrence glm5next's 34 linear
//! attention layers run, with a **low-rank per-channel** decay.
//!
//! Reference: llama.cpp `llm_build_delta_net_base::build_delta_net_autoregressive`
//! (`src/models/delta-net-base.cpp`), reached from
//! `glm5next.cpp::build_kda_layer` via `build_recurrent_attn`. That function is
//! the *definition*; `build_delta_net_chunking` is a prefill optimisation
//! llama.cpp validates against it, and is deliberately not ported here.
//!
//! ## What makes this KDA rather than gated-delta-net
//!
//! ggml runs one function for both and tells them apart by the decay's width:
//! `const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k)`. Gated-delta-net
//! (Qwen 3.5, `qwen35.rs`) has **one decay scalar per head**; KDA has **one per
//! channel**, produced by the low-rank chain `ssm_f_a` (`[n_embd, head_dim]`) →
//! `ssm_f_b` (`[head_dim, d_inner]`). So the state update carries a diagonal
//! where GDN carries a scalar, and `qwen35`'s `delta_net_step` cannot be reused.
//!
//! ## Axis convention — read this before changing any index
//!
//! Two axes, named here by the role they play in the recurrence rather than as
//! "key" and "value", because the reference's own comments use `S_k` and `S_v`
//! interchangeably (they are equal, and `GGML_ASSERT(S_k == S_v)` sits right
//! above the loose comment):
//!
//!   * **`kq`** — the axis `k` and `q` index.
//!   * **`vo`** — the axis `v` and the output index.
//!
//! State here is `s[head][vo][kq]`, row-major, matching the `[n_head, head_dim,
//! head_dim]` shape `KvCache::ssm_state` already uses for `qwen35`. ggml's
//! autoregressive path holds the **transpose** of this internally (`k`/`q`
//! broadcast on its dim0, `v`/`g`/`o` live on its dim1); the arithmetic is the
//! same, only the physical layout differs. Do not "fix" one to look like the
//! other.
//!
//! **The decay is on `vo`.** In ggml, `g` reshapes to `[1, S, H, n_seqs]` and so
//! multiplies along dim1 — the same axis `v` and the output live on, not the one
//! `k` and `q` index. In this module's layout that makes it a **per-row** scale.
//! Derivation, from `build_delta_net_autoregressive`:
//!
//! ```text
//! sk  = sum_rows(s * k)      // k broadcasts on dim0 => sk survives on dim1
//! d   = v - transpose(sk)    // v's axis == sk's surviving axis == dim1
//! s   = s * g                // g is [1, S, ...] => also dim1
//! ```
//!
//! [`decay_axis_is_the_output_axis`] pins this; if a greedy-token comparison
//! against llama.cpp ever disagrees, that test is where the one-line fix goes.
//!
//! ## The step
//!
//! Per head, per token, with `S = kda.head_dim` (128):
//!
//! ```text
//! s[i][j] *= exp(g[i])                      // g is LOG-decay, exp'd HERE
//! sk[i]    = sum_j s[i][j] * k[j]
//! d[i]     = (v[i] - sk[i]) * beta          // beta is one scalar per head
//! s[i][j] += d[i] * k[j]                    // outer product
//! o[i]     = sum_j s[i][j] * q[j] / sqrt(S) // q scaled HERE
//! ```
//!
//! Two boundary conventions kept identical to the reference so the surrounding
//! pieces compose:
//!   * `g` arrives in **log space** — `build_kda_layer` produces
//!     `gate_lower_bound * sigmoid(-(ssm_a * (f_b(f_a(x)) + dt_bias)))`, which
//!     is in `(-5, 0)` for glm5next, and the `exp` happens inside this step.
//!   * `q` and `k` arrive **L2-normed** (at the reference's own hardcoded
//!     `1e-6`, not the model's norm eps) and `q` is **unscaled**; the
//!     `1/sqrt(S)` is applied inside, as `ggml_scale(q, scale)` does.
//!
//! What is *not* here, deliberately: the depthwise conv over `q‖k‖v`, the
//! `f`/`g`/`beta` projections, the `ssm_norm` + sigmoid output gate, and the
//! `wo` projection. Those are separate stages in `build_kda_layer`; keeping them
//! out leaves this function testable as pure arithmetic.
//!
//! **Wired into `forward`: no.** This is piece (1)'s reference implementation.

use crate::{LlamaError, Result};

/// One KDA step for every head of a single token.
///
/// Slice widths, all `n_head * head_dim` except `beta` (`n_head`) and `state`
/// (`n_head * head_dim * head_dim`):
///
///   * `state` — `s[head][vo][kq]`, updated in place.
///   * `q`, `k` — L2-normed, `q` unscaled.
///   * `v`
///   * `g_log` — log-decay per `vo` channel. For a gated-delta-net-style scalar
///     decay, repeat the head's value across its channels.
///   * `beta` — one per head, already sigmoid'd.
///   * `out` — `o[head][vo]`, overwritten.
#[allow(clippy::too_many_arguments)]
pub fn step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_log: &[f32],
    beta: &[f32],
    n_head: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    let hd = head_dim;
    let want = n_head * hd;
    if n_head == 0 || hd == 0 {
        return Err(LlamaError::Config(
            "kda: n_head and head_dim must be > 0".into(),
        ));
    }
    for (name, got) in [("q", q.len()), ("k", k.len()), ("v", v.len()), ("g_log", g_log.len()), ("out", out.len())] {
        if got != want {
            return Err(LlamaError::Config(format!(
                "kda: {name} is {got} wide, expected n_head*head_dim = {want}"
            )));
        }
    }
    if beta.len() != n_head {
        return Err(LlamaError::Config(format!(
            "kda: beta is {} wide, expected n_head = {n_head}",
            beta.len()
        )));
    }
    if state.len() != n_head * hd * hd {
        return Err(LlamaError::Config(format!(
            "kda: state is {} wide, expected n_head*head_dim^2 = {}",
            state.len(),
            n_head * hd * hd
        )));
    }

    // The reference scales q by 1/sqrt(S_k) before the recurrence.
    let scale = 1.0 / (hd as f32).sqrt();

    let mut d = vec![0.0f32; hd];

    for h in 0..n_head {
        let hs = &mut state[h * hd * hd..(h + 1) * hd * hd];
        let qh = &q[h * hd..(h + 1) * hd];
        let kh = &k[h * hd..(h + 1) * hd];
        let vh = &v[h * hd..(h + 1) * hd];
        let gh = &g_log[h * hd..(h + 1) * hd];
        let bh = beta[h];

        for i in 0..hd {
            // Per-row decay: g is on the vo axis. See the module docs.
            let decay = gh[i].exp();
            let row = &mut hs[i * hd..(i + 1) * hd];
            let mut acc = 0.0f32;
            for (rj, &kj) in row.iter_mut().zip(kh.iter()) {
                *rj *= decay;
                acc += *rj * kj;
            }
            // The delta rule: write only the part of v the state does not
            // already predict. With beta = 1 and a repeated k this is exactly
            // zero, which is the property `delta_rule_is_idempotent_on_a_repeat`
            // checks.
            d[i] = (vh[i] - acc) * bh;
        }

        for i in 0..hd {
            let row = &mut hs[i * hd..(i + 1) * hd];
            let di = d[i];
            for (rj, &kj) in row.iter_mut().zip(kh.iter()) {
                *rj += di * kj;
            }
            let mut acc = 0.0f32;
            for (&rj, &qj) in row.iter().zip(qh.iter()) {
                acc += rj * qj;
            }
            out[h * hd + i] = acc * scale;
        }
    }
    Ok(())
}

/// Convenience wrapper for a gated-delta-net-style **scalar** decay: broadcasts
/// one log-decay per head across that head's channels and calls [`step`]. Used
/// by the reduction test, and the shape a non-KDA delta-net layer would take.
#[allow(clippy::too_many_arguments)]
pub fn step_scalar_decay(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_log_per_head: &[f32],
    beta: &[f32],
    n_head: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    if g_log_per_head.len() != n_head {
        return Err(LlamaError::Config(format!(
            "kda: scalar g_log is {} wide, expected n_head = {n_head}",
            g_log_per_head.len()
        )));
    }
    let mut g = Vec::with_capacity(n_head * head_dim);
    for &gh in g_log_per_head {
        g.extend(std::iter::repeat(gh).take(head_dim));
    }
    step(state, q, k, v, &g, beta, n_head, head_dim, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &mut [f32]) {
        let inv = 1.0 / v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }

    /// The orientation this module must have, written the way
    /// `Backend::delta_net_step`'s host fallback writes it (see
    /// `ggml-rs/src/backend.rs`, the per-head block): state rows are the `vo`
    /// axis, `k` and `q` index the columns, decay is a per-head scalar.
    ///
    /// That code was verified greedy-identical to llama.cpp for Qwen 3.5, so it
    /// is the orientation of record. This pins that generalising its scalar
    /// decay to a per-channel one changed nothing else.
    fn gdn_reference(
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g_t: f32,
        beta: f32,
        head_dim: usize,
        out: &mut [f32],
    ) {
        let hd = head_dim;
        for s in state.iter_mut() {
            *s *= g_t;
        }
        let mut kv_mem = vec![0.0f32; hd];
        for i in 0..hd {
            let row = &state[i * hd..(i + 1) * hd];
            kv_mem[i] = (0..hd).map(|j| row[j] * k[j]).sum();
        }
        let mut delta = vec![0.0f32; hd];
        for i in 0..hd {
            delta[i] = (v[i] - kv_mem[i]) * beta;
        }
        for i in 0..hd {
            let row = &mut state[i * hd..(i + 1) * hd];
            let di = delta[i];
            for j in 0..hd {
                row[j] += di * k[j];
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..hd {
            let row = &state[i * hd..(i + 1) * hd];
            out[i] = (0..hd).map(|j| row[j] * q[j]).sum::<f32>() * scale;
        }
    }

    /// With a decay that is constant across a head's channels, KDA must reduce
    /// exactly to gated-delta-net. This isolates the one genuinely new thing
    /// (per-channel decay) from the shared structure.
    #[test]
    fn reduces_to_gated_delta_net_when_the_decay_is_per_head() {
        let hd = 8;
        let n_head = 2;
        let mut q: Vec<f32> = (0..n_head * hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut k: Vec<f32> = (0..n_head * hd).map(|i| (i as f32 * 0.71).cos()).collect();
        let v: Vec<f32> = (0..n_head * hd).map(|i| (i as f32 * 0.13).sin() * 2.0).collect();
        for h in 0..n_head {
            l2(&mut q[h * hd..(h + 1) * hd]);
            l2(&mut k[h * hd..(h + 1) * hd]);
        }
        let g_per_head = vec![-0.35f32, -1.2];
        let beta = vec![0.6f32, 0.9];

        let init: Vec<f32> = (0..n_head * hd * hd)
            .map(|i| ((i % 17) as f32 - 8.0) / 20.0)
            .collect();

        let mut mine = init.clone();
        let mut got = vec![0.0f32; n_head * hd];
        step_scalar_decay(&mut mine, &q, &k, &v, &g_per_head, &beta, n_head, hd, &mut got)
            .expect("kda step");

        let mut theirs = init.clone();
        let mut want = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            gdn_reference(
                &mut theirs[h * hd * hd..(h + 1) * hd * hd],
                &q[h * hd..(h + 1) * hd],
                &k[h * hd..(h + 1) * hd],
                &v[h * hd..(h + 1) * hd],
                g_per_head[h].exp(),
                beta[h],
                hd,
                &mut want[h * hd..(h + 1) * hd],
            );
        }

        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "out[{i}]: {a} vs {b}");
        }
        for (i, (a, b)) in mine.iter().zip(theirs.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "state[{i}]: {a} vs {b}");
        }
    }

    /// The decay is on the **output** axis, so a killed channel must clear a
    /// state **row**, leaving the other rows intact. `beta = 0` suppresses the
    /// write so only the decay is observed.
    ///
    /// If a real-weights comparison against llama.cpp ever disagrees, flip the
    /// index here and in [`step`] together.
    #[test]
    fn decay_axis_is_the_output_axis() {
        let hd = 4;
        let n_head = 1;
        let dead = 2usize;

        let mut state = vec![1.0f32; hd * hd];
        let mut g = vec![0.0f32; hd];
        g[dead] = -1000.0; // exp -> 0

        let q = vec![0.25f32; hd];
        let k = vec![0.5f32; hd];
        let v = vec![7.0f32; hd];
        let beta = vec![0.0f32]; // no write, decay only
        let mut out = vec![0.0f32; hd];

        step(&mut state, &q, &k, &v, &g, &beta, n_head, hd, &mut out).expect("step");

        for j in 0..hd {
            assert_eq!(state[dead * hd + j], 0.0, "row {dead} col {j} must be cleared");
        }
        for i in (0..hd).filter(|&i| i != dead) {
            for j in 0..hd {
                assert!(
                    (state[i * hd + j] - 1.0).abs() < 1e-6,
                    "row {i} col {j} must survive, got {}",
                    state[i * hd + j]
                );
            }
        }
    }

    /// From a zero state with `g = 0` and `beta = 1`, one step stores `v` keyed
    /// by `k`, and the read-out is `(k.q)/sqrt(S) * v`. Pins the outer-product
    /// orientation and the q scale together.
    #[test]
    fn zero_state_single_step_is_the_scaled_outer_product() {
        let hd = 8;
        let mut q: Vec<f32> = (0..hd).map(|i| i as f32 + 1.0).collect();
        let mut k: Vec<f32> = (0..hd).map(|i| i as f32 * 0.5 - 1.0).collect();
        l2(&mut q);
        l2(&mut k);
        let v: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.3).sin() * 3.0).collect();

        let mut state = vec![0.0f32; hd * hd];
        let mut out = vec![0.0f32; hd];
        step(&mut state, &q, &k, &v, &vec![0.0; hd], &[1.0], 1, hd, &mut out).expect("step");

        let kq: f32 = k.iter().zip(&q).map(|(a, b)| a * b).sum();
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..hd {
            let want = kq * v[i] * scale;
            assert!((out[i] - want).abs() < 1e-5, "out[{i}]: {} vs {want}", out[i]);
        }
    }

    /// The defining property of the delta rule: re-presenting a key the state
    /// already predicts writes nothing. With `beta = 1`, `g = 0` and an L2-normed
    /// `k`, the second step's delta is exactly zero.
    #[test]
    fn delta_rule_is_idempotent_on_a_repeat() {
        let hd = 8;
        let mut q: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.9).cos()).collect();
        let mut k: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.4 + 0.2).sin()).collect();
        l2(&mut q);
        l2(&mut k);
        let v: Vec<f32> = (0..hd).map(|i| (i as f32 - 3.0) * 1.5).collect();

        let mut state = vec![0.0f32; hd * hd];
        let mut out = vec![0.0f32; hd];
        step(&mut state, &q, &k, &v, &vec![0.0; hd], &[1.0], 1, hd, &mut out).expect("step 1");
        let after_first = state.clone();

        step(&mut state, &q, &k, &v, &vec![0.0; hd], &[1.0], 1, hd, &mut out).expect("step 2");
        for (i, (a, b)) in state.iter().zip(after_first.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "state[{i}] changed on a repeat: {a} vs {b}"
            );
        }
    }

    /// A per-channel decay must actually differ from any per-head scalar —
    /// otherwise the new kernel is not doing anything the old one could not.
    #[test]
    fn per_channel_decay_is_not_expressible_as_a_scalar() {
        let hd = 4;
        let state0: Vec<f32> = (0..hd * hd).map(|i| (i as f32 + 1.0) / 10.0).collect();
        let q = vec![0.5f32; hd];
        let k = vec![0.5f32; hd];
        let v = vec![1.0f32; hd];
        let beta = vec![0.7f32];

        let g_channels = vec![-0.1f32, -0.5, -1.0, -2.0];
        let mut a = state0.clone();
        let mut oa = vec![0.0f32; hd];
        step(&mut a, &q, &k, &v, &g_channels, &beta, 1, hd, &mut oa).expect("per-channel");

        // No single scalar reproduces it.
        for cand in [-0.1f32, -0.5, -0.9, -1.0, -2.0] {
            let mut b = state0.clone();
            let mut ob = vec![0.0f32; hd];
            step_scalar_decay(&mut b, &q, &k, &v, &[cand], &beta, 1, hd, &mut ob)
                .expect("scalar");
            let diff: f32 = oa.iter().zip(&ob).map(|(x, y)| (x - y).abs()).sum();
            assert!(diff > 1e-4, "scalar decay {cand} reproduced the per-channel result");
        }
    }

    #[test]
    fn rejects_mismatched_widths() {
        let hd = 4;
        let mut state = vec![0.0f32; hd * hd];
        let ok = vec![0.0f32; hd];
        let mut out = vec![0.0f32; hd];

        assert!(step(&mut state, &ok[..2], &ok, &ok, &ok, &[1.0], 1, hd, &mut out).is_err());
        assert!(step(&mut state, &ok, &ok, &ok, &ok, &[1.0, 1.0], 1, hd, &mut out).is_err());
        assert!(step(&mut vec![0.0; 3], &ok, &ok, &ok, &ok, &[1.0], 1, hd, &mut out).is_err());
        assert!(step(&mut state, &ok, &ok, &ok, &ok, &[1.0], 0, hd, &mut out).is_err());
        assert!(step_scalar_decay(&mut state, &ok, &ok, &ok, &[0.0, 0.0], &[1.0], 1, hd, &mut out).is_err());
    }

    /// Released geometry: 64 heads of 128. Just checks it runs and stays finite
    /// at the real widths.
    #[test]
    fn released_geometry_stays_finite() {
        let (n_head, hd) = (64, 128);
        let mut q: Vec<f32> = (0..n_head * hd).map(|i| ((i % 31) as f32 - 15.0) / 15.0).collect();
        let mut k: Vec<f32> = (0..n_head * hd).map(|i| ((i % 23) as f32 - 11.0) / 11.0).collect();
        let v: Vec<f32> = (0..n_head * hd).map(|i| ((i % 19) as f32 - 9.0) / 9.0).collect();
        for h in 0..n_head {
            l2(&mut q[h * hd..(h + 1) * hd]);
            l2(&mut k[h * hd..(h + 1) * hd]);
        }
        // glm5next's gate range: lower_bound -5 * sigmoid(..) => (-5, 0).
        let g: Vec<f32> = (0..n_head * hd).map(|i| -5.0 * (((i % 7) as f32) / 7.0)).collect();
        let beta = vec![0.5f32; n_head];

        let mut state = vec![0.0f32; n_head * hd * hd];
        let mut out = vec![0.0f32; n_head * hd];
        for _ in 0..4 {
            step(&mut state, &q, &k, &v, &g, &beta, n_head, hd, &mut out).expect("step");
        }
        assert!(out.iter().all(|x| x.is_finite()), "output must stay finite");
        assert!(state.iter().all(|x| x.is_finite()), "state must stay finite");
        assert!(out.iter().any(|&x| x != 0.0), "output must not be all zeros");
    }
}
