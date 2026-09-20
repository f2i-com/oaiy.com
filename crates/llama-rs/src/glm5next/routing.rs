// VENDORED-LOCAL: whole module. GLM-5.3-Flash MoE routing.
//! glm5next expert routing: sigmoid gating with a DeepSeek-V3-style selection
//! bias (`noaux_tc`).
//!
//! Reference: llama.cpp `llm_graph_context::build_moe_ffn`
//! (`src/llama-graph.cpp`), the `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` branch,
//! as called from `glm5next.cpp::build_layer_ffn`. The published weights set
//! `expert_gating_func = 2` (sigmoid), `expert_weights_norm = true`,
//! `expert_weights_scale = 2.5`, `expert_count = 288`, `expert_used_count = 8`
//! and `expert_group_count = 1` (so no group-limited routing).
//!
//! The order of operations is the whole content of this module, and getting it
//! wrong produces plausible output that only diverges a few layers later:
//!
//! ```text
//! probs  = sigmoid(logits)            // [n_expert]
//! sel    = probs + exp_probs_b        // bias affects SELECTION ONLY
//! ids    = top_k(sel)
//! w      = probs[ids]                 // gathered UNBIASED
//! w     /= sum(w)                     // iff expert_weights_norm
//! w     *= expert_weights_scale
//! ```
//!
//! Two traps:
//!   * **The bias never reaches the weights.** llama.cpp keeps `probs` and
//!     `selection_probs` as separate tensors for exactly this reason
//!     ("leave probs unbiased as it's later used to get expert weights"). An
//!     implementation that biases once and reuses the result is wrong in a way
//!     no shape check catches.
//!   * **`norm_w` normalises sigmoid values, not a softmax.** The gathered
//!     weights do not sum to 1 beforehand, so the division is a real
//!     renormalisation rather than a no-op. After it, the weights sum to
//!     `expert_weights_scale`.
//!
//! This differs from DeepSeek-V4.1's routing in `dsv41::moe`, which is
//! `sqrt(softplus(...))` — llama.cpp's separate
//! `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`. Do not share code between
//! them; they are different gating functions.
//!
//! **Wired into `forward`: no.** `Glm5NextModel::forward` is still the pending
//! stub; this is the routing half of piece (4), tested standalone.

/// Sigmoid, in f32. The reference runs this on device in f32 too.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One token's routing decision.
///
/// `logits` is the router's `[n_expert]` row, `probs_bias` the optional
/// `exp_probs_b` of the same width. Returns `top_k` expert ids in descending
/// selection order together with their weights.
///
/// Ties on the selection score resolve to the **lower expert id**, matching
/// `ggml_argsort_top_k`'s stable descending order on the CPU backend. With 288
/// experts and f32 sigmoid this is rare, but it must be deterministic or a
/// streamed run can route differently from a resident one.
pub fn route_token(
    logits: &[f32],
    probs_bias: Option<&[f32]>,
    top_k: usize,
    norm_w: bool,
    w_scale: f32,
) -> (Vec<u32>, Vec<f32>) {
    let n_expert = logits.len();
    let k = top_k.min(n_expert);

    // probs stays unbiased: it is what the weights are gathered from.
    let probs: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();

    // The bias exists only to move the top-k boundary.
    let mut order: Vec<u32> = (0..n_expert as u32).collect();
    let sel: Vec<f32> = match probs_bias {
        Some(b) => probs.iter().zip(b).map(|(p, bb)| p + bb).collect(),
        None => probs.clone(),
    };
    // NaN sorts last so a poisoned logit cannot win a slot. Mapping it to
    // -inf is load-bearing: `partial_cmp` returns None for NaN, and falling
    // back to Equal would let index order promote NaN to the first slot.
    let key = |e: u32| -> f32 {
        let v = sel[e as usize];
        if v.is_nan() {
            f32::NEG_INFINITY
        } else {
            v
        }
    };
    order.sort_by(|&a, &b| {
        key(b)
            .partial_cmp(&key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);

    let mut weights: Vec<f32> = order.iter().map(|&e| probs[e as usize]).collect();

    if norm_w {
        let sum: f32 = weights.iter().sum();
        // A zero sum would need every selected sigmoid to underflow; leaving the
        // weights untouched is better than emitting NaNs into the residual.
        if sum != 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }
    if w_scale != 1.0 {
        for w in weights.iter_mut() {
            *w *= w_scale;
        }
    }

    (order, weights)
}

/// Batch form matching `moe::moe_forward_with_logits`'s mailbox layout:
/// `logits` is `[seq, n_expert]` row-major, and the returned vectors are
/// `[seq, top_k]` flattened, so token `t`'s slice is `t*top_k..(t+1)*top_k`.
pub fn route_rows(
    logits: &[f32],
    n_expert: usize,
    probs_bias: Option<&[f32]>,
    top_k: usize,
    norm_w: bool,
    w_scale: f32,
) -> crate::Result<(Vec<u32>, Vec<f32>)> {
    if n_expert == 0 || top_k == 0 {
        return Err(crate::LlamaError::Config(
            "glm5next routing: n_expert and top_k must be > 0".into(),
        ));
    }
    if logits.len() % n_expert != 0 {
        return Err(crate::LlamaError::Config(format!(
            "glm5next routing: {} logits is not a multiple of n_expert {n_expert}",
            logits.len()
        )));
    }
    if let Some(b) = probs_bias {
        if b.len() != n_expert {
            return Err(crate::LlamaError::Config(format!(
                "glm5next routing: exp_probs_b has {} entries, n_expert is {n_expert}",
                b.len()
            )));
        }
    }
    let seq = logits.len() / n_expert;
    let k = top_k.min(n_expert);
    let mut ids = Vec::with_capacity(seq * k);
    let mut ws = Vec::with_capacity(seq * k);
    for t in 0..seq {
        let row = &logits[t * n_expert..(t + 1) * n_expert];
        let (i, w) = route_token(row, probs_bias, top_k, norm_w, w_scale);
        ids.extend_from_slice(&i);
        ws.extend_from_slice(&w);
    }
    Ok((ids, ws))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// glm5next's own settings.
    const SCALE: f32 = 2.5;

    #[test]
    fn weights_are_unbiased_sigmoids_normalised_to_the_scale() {
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        let (ids, w) = route_token(&logits, None, 2, true, SCALE);
        assert_eq!(ids, vec![0, 1]);

        let (p0, p1) = (sigmoid(2.0), sigmoid(1.0));
        let sum = p0 + p1;
        assert!((w[0] - p0 / sum * SCALE).abs() < 1e-6);
        assert!((w[1] - p1 / sum * SCALE).abs() < 1e-6);
        // After norm + scale the weights sum to exactly the scale.
        assert!((w.iter().sum::<f32>() - SCALE).abs() < 1e-5);
    }

    /// The whole point of `exp_probs_b`: it moves the top-k boundary while the
    /// returned weight stays the expert's *unbiased* sigmoid.
    #[test]
    fn bias_flips_selection_but_not_the_weight() {
        let logits = [2.0f32, 1.0, -4.0, -1.0];
        // Expert 2 has the smallest sigmoid by far, but a large bias.
        let bias = [0.0f32, 0.0, 10.0, 0.0];

        let (unbiased, _) = route_token(&logits, None, 2, false, 1.0);
        assert_eq!(unbiased, vec![0, 1], "without bias, the top two logits win");

        let (ids, w) = route_token(&logits, Some(&bias), 2, false, 1.0);
        assert_eq!(ids[0], 2, "the bias must win expert 2 a slot");
        assert!(ids.contains(&0));

        // ... and its weight is sigmoid(-4.0), NOT sigmoid(-4.0) + 10.
        let want = sigmoid(-4.0);
        let got = w[ids.iter().position(|&e| e == 2).unwrap()];
        assert!(
            (got - want).abs() < 1e-7,
            "weight {got} should be the unbiased sigmoid {want}"
        );
        assert!(got < 0.02, "a biased selection must not inherit the bias");
    }

    /// `norm_w` divides by a sum that is not 1, unlike a softmax.
    #[test]
    fn norm_is_a_real_renormalisation() {
        let logits = [3.0f32, 3.0, 3.0, 3.0];
        let (_, raw) = route_token(&logits, None, 4, false, 1.0);
        let raw_sum: f32 = raw.iter().sum();
        assert!(
            (raw_sum - 1.0).abs() > 0.5,
            "four sigmoid(3) values sum to {raw_sum}, nowhere near 1"
        );

        let (_, normed) = route_token(&logits, None, 4, true, 1.0);
        assert!((normed.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ties_resolve_to_the_lower_expert_id() {
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let (ids, _) = route_token(&logits, None, 2, true, 1.0);
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn nan_logits_do_not_win_a_slot() {
        let logits = [f32::NAN, 1.0, 0.5, 0.0];
        let (ids, _) = route_token(&logits, None, 2, false, 1.0);
        assert_eq!(ids, vec![1, 2], "NaN must sort last");
    }

    #[test]
    fn top_k_is_clamped_to_the_expert_count() {
        let logits = [1.0f32, 0.5];
        let (ids, w) = route_token(&logits, None, 8, true, SCALE);
        assert_eq!(ids.len(), 2);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn route_rows_lays_out_the_mailbox_per_token() {
        // 3 tokens, 4 experts; each row's argmax is a different expert.
        let logits = [
            3.0f32, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0,
        ];
        let (ids, w) = route_rows(&logits, 4, None, 2, true, SCALE).expect("route");
        assert_eq!(ids.len(), 6);
        assert_eq!(w.len(), 6);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[2], 1);
        assert_eq!(ids[4], 2);
        for t in 0..3 {
            let s: f32 = w[t * 2..(t + 1) * 2].iter().sum();
            assert!((s - SCALE).abs() < 1e-5, "token {t} weights sum to {s}");
        }
    }

    #[test]
    fn route_rows_rejects_a_mismatched_bias_or_ragged_logits() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        assert!(route_rows(&logits, 3, None, 2, true, 1.0).is_err());
        let bias = [0.0f32; 3];
        assert!(route_rows(&logits, 4, Some(&bias), 2, true, 1.0).is_err());
        assert!(route_rows(&logits, 0, None, 2, true, 1.0).is_err());
        assert!(route_rows(&logits, 4, None, 0, true, 1.0).is_err());
    }

    /// The published model's own numbers, as a smoke test on realistic widths.
    #[test]
    fn released_geometry_routes_top_8_of_288() {
        let n_expert = 288;
        let logits: Vec<f32> = (0..n_expert)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 20.0)
            .collect();
        let bias: Vec<f32> = (0..n_expert)
            .map(|i| ((i * 13 % 7) as f32 - 3.0) / 100.0)
            .collect();
        let (ids, w) = route_token(&logits, Some(&bias), 8, true, SCALE);
        assert_eq!(ids.len(), 8);
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 8, "no expert may be selected twice");
        assert!((w.iter().sum::<f32>() - SCALE).abs() < 1e-4);
        assert!(w.iter().all(|&x| x > 0.0 && x.is_finite()));
    }
}
