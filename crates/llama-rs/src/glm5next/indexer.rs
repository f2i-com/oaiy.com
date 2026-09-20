// VENDORED-LOCAL: whole module. GLM-5.3-Flash sparse "lightning indexer".
//! The indexer that decides which cached cells a full-attention layer attends
//! to: it compresses each complete pool of `kpool` cells into one pooled key,
//! scores those pooled keys against the query, and takes the top
//! [`crate::glm5next::kpool::select_k`] pools.
//!
//! Reference: llama.cpp `llama_model_glm5next::graph::build_indexer`
//! (PR #27754, pinned at `86ebfef`). The pool geometry, masks and per-pool bias
//! this module consumes live in [`crate::glm5next::kpool`].
//!
//! ## Stage 1 — the pooled key ([`pooled_key`])
//!
//! Each cell contributes two independently projected vectors, both cached:
//! `indexer.attn_k` → the **key** (through a LayerNorm **with bias**, unlike
//! every RMSNorm elsewhere in this arch), and `indexer_compressor_gate` → the
//! **gate**. The reference is explicit that the gate is "a SECOND, INDEPENDENT
//! projection, not a reuse of the key".
//!
//! Pooling is then a **per-channel** weighted average over the pool's slots:
//!
//! ```text
//! for each channel ch:
//!     p[slot] = softmax_over_slots(gate[slot][ch] + ape[slot][ch])
//!     pooled[ch] = sum_slot p[slot] * key[slot][ch]
//! ```
//!
//! Two details that are easy to lose:
//!   * **The softmax is over the slot axis, per channel** — not over channels.
//!     The reference permutes the slot axis to dim 0 precisely so
//!     `ggml_soft_max` reduces over it. Different channels may therefore favour
//!     different slots of the same pool, which [`pooling_is_independent_per_channel`]
//!     pins.
//!   * **`ape` is added pre-softmax.** `indexer_compressor_ape` is
//!     `[d_idx, kpool]` in the GGUF, indexed by `slot = position % kpool` — see
//!     [`crate::glm5next::kpool::slot_of`]. It is a positional encoding over
//!     slots, so adding it after the softmax would make it a plain scale.
//!
//! ## Stage 2 — the scores ([`pool_scores`])
//!
//! ```text
//! dot[h][p] = sum_ch iq[h][ch] * pooled_key[p][ch]
//! score[p]  = sum_h relu(dot[h][p]) * w[h] / sqrt(d_idx * n_ihead)
//! score[p] += pool_bias[p]
//! ```
//!
//! `iq` is `indexer.attn_q_b @ q_a_norm(x)`, reshaped to `n_ihead` heads of
//! `d_idx`. `w` is `indexer.proj @ x`, one weight per indexer head.
//!
//! **The ReLU sits between the per-head dot and the head weighting.** The
//! reference says so in as many words ("either side differs"), and it is
//! observable rather than cosmetic because `w` is **sign-unconstrained**: a
//! negative dot against a negative head weight contributes `0` here, but would
//! contribute a *positive* score if the ReLU were applied after the weighting.
//! [`relu_precedes_the_head_weighting`] is that test.
//!
//! The reference also forces F32 accumulation on `w`
//! (`ggml_prec_set_acc(w, GGML_PREC_F32)`) because bf16 swaps near-tied pools.
//! This module is f32 throughout, so that is satisfied by construction — but a
//! future device kernel must not quietly drop to f16 here.
//!
//! No RoPE anywhere: `n_rot()` is 0 for the whole text tower.
//!
//! ## Stage 3 — selection
//!
//! [`select_pools`] takes the top `k` **pools** (never cells — ReLU ties span
//! pool bounds, and a cell-level cut would take partial pools), then
//! [`crate::glm5next::kpool::expand_pools`] turns them into cell indices.
//! [`candidates`] is the two together.
//!
//! **Wired into `forward`: no.** This closes the indexer half of piece (3); the
//! absorbed MLA that consumes the candidate set is still to do.

use super::kpool;
use crate::{LlamaError, Result};

/// Build one pool's pooled key.
///
/// All three inputs are `[kpool, d_idx]` row-major — `d_idx` contiguous, which
/// is how the loader presents `indexer_compressor_ape` (GGUF `[d_idx, kpool]`
/// reversed). `keys` and `gates` are the pool's member rows in **slot order**,
/// i.e. member `s` is the cell whose `position % kpool == s`.
pub fn pooled_key(
    keys: &[f32],
    gates: &[f32],
    ape: &[f32],
    kpool: usize,
    d_idx: usize,
    out: &mut [f32],
) -> Result<()> {
    if kpool == 0 || d_idx == 0 {
        return Err(LlamaError::Config(
            "indexer: kpool and d_idx must be > 0".into(),
        ));
    }
    let want = kpool * d_idx;
    for (name, got) in [("keys", keys.len()), ("gates", gates.len()), ("ape", ape.len())] {
        if got != want {
            return Err(LlamaError::Config(format!(
                "indexer: {name} is {got} wide, expected kpool*d_idx = {want}"
            )));
        }
    }
    if out.len() != d_idx {
        return Err(LlamaError::Config(format!(
            "indexer: pooled key out is {} wide, expected d_idx = {d_idx}",
            out.len()
        )));
    }

    // One softmax per channel, over the slot axis. Done channel-major so each
    // reduction is over a strided slice of `kpool` values.
    let mut logits = vec![0.0f32; kpool];
    for (ch, o) in out.iter_mut().enumerate() {
        let mut max = f32::NEG_INFINITY;
        for s in 0..kpool {
            let l = gates[s * d_idx + ch] + ape[s * d_idx + ch];
            logits[s] = l;
            if l > max {
                max = l;
            }
        }
        let mut sum = 0.0f32;
        for l in logits.iter_mut() {
            *l = (*l - max).exp();
            sum += *l;
        }
        let inv = 1.0 / sum;
        let mut acc = 0.0f32;
        for s in 0..kpool {
            acc += logits[s] * inv * keys[s * d_idx + ch];
        }
        *o = acc;
    }
    Ok(())
}

/// The scale the reference applies to the head weights:
/// `1 / sqrt(d_idx * n_ihead)`. For the released model that is `1/64`.
pub fn score_scale(d_idx: usize, n_ihead: usize) -> f32 {
    1.0 / ((d_idx * n_ihead) as f32).sqrt()
}

/// Score every pool for one query.
///
/// * `iq` — `[n_ihead, d_idx]` row-major, the indexer queries.
/// * `w` — `[n_ihead]` **raw** head weights; the `1/sqrt(d_idx * n_ihead)`
///   scale is applied here so a caller cannot forget it. Values may be negative.
/// * `pool_keys` — `[n_pools, d_idx]` row-major pooled keys.
/// * `bias` — optional `[n_pools]` per-pool bias from
///   [`kpool::pool_bias_row`]; `-inf` entries make a pool unselectable.
/// * `out` — `[n_pools]`.
pub fn pool_scores(
    iq: &[f32],
    w: &[f32],
    pool_keys: &[f32],
    bias: Option<&[f32]>,
    n_ihead: usize,
    d_idx: usize,
    n_pools: usize,
    out: &mut [f32],
) -> Result<()> {
    if n_ihead == 0 || d_idx == 0 {
        return Err(LlamaError::Config(
            "indexer: n_ihead and d_idx must be > 0".into(),
        ));
    }
    if iq.len() != n_ihead * d_idx {
        return Err(LlamaError::Config(format!(
            "indexer: iq is {} wide, expected n_ihead*d_idx = {}",
            iq.len(),
            n_ihead * d_idx
        )));
    }
    if w.len() != n_ihead {
        return Err(LlamaError::Config(format!(
            "indexer: w is {} wide, expected n_ihead = {n_ihead}",
            w.len()
        )));
    }
    if pool_keys.len() != n_pools * d_idx {
        return Err(LlamaError::Config(format!(
            "indexer: pool_keys is {} wide, expected n_pools*d_idx = {}",
            pool_keys.len(),
            n_pools * d_idx
        )));
    }
    if out.len() != n_pools {
        return Err(LlamaError::Config(format!(
            "indexer: out is {} wide, expected n_pools = {n_pools}",
            out.len()
        )));
    }
    if let Some(b) = bias {
        if b.len() != n_pools {
            return Err(LlamaError::Config(format!(
                "indexer: bias is {} wide, expected n_pools = {n_pools}",
                b.len()
            )));
        }
    }

    let scale = score_scale(d_idx, n_ihead);

    for (p, o) in out.iter_mut().enumerate() {
        let pk = &pool_keys[p * d_idx..(p + 1) * d_idx];
        let mut acc = 0.0f32;
        for h in 0..n_ihead {
            let qh = &iq[h * d_idx..(h + 1) * d_idx];
            let dot: f32 = qh.iter().zip(pk).map(|(a, b)| a * b).sum();
            // The ReLU is HERE, before the head weight. `w[h]` may be negative,
            // so moving it after the multiply changes the result.
            acc += dot.max(0.0) * w[h];
        }
        *o = acc * scale;
    }
    // Added after the scale, as the reference does.
    if let Some(b) = bias {
        for (o, &bp) in out.iter_mut().zip(b) {
            *o += bp;
        }
    }
    Ok(())
}

/// Take the `k` highest-scoring pools, descending. Ties resolve to the **lower
/// pool index**, matching [`super::routing::route_token`] and
/// `ggml_argsort_top_k`'s stable order. A `-inf` score (a masked pool) sorts
/// last and is only returned if `k` exceeds the number of live pools.
pub fn select_pools(scores: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    let key = |p: u32| -> f32 {
        let v = scores[p as usize];
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
    order.truncate(k.min(scores.len()));
    order
}

/// Score, select and expand in one call: the candidate cell set for one query.
///
/// Pools whose bias is `-inf` are dropped rather than returned with a `-inf`
/// score, so a short history cannot smuggle in unselectable pools when
/// `select_k` exceeds the number of visible ones.
///
/// The returned cells are in **pool-score order, not position order**, because
/// [`select_pools`] sorts by score. That is fine for the only intended consumer,
/// [`super::mla::attn_mask`], which scatters them into a mask. Anything that
/// compares this against llama.cpp's `top_k` tensor, or uses it as a gather
/// order, must sort first.
#[allow(clippy::too_many_arguments)]
pub fn candidates(
    iq: &[f32],
    w: &[f32],
    pool_keys: &[f32],
    bias: &[f32],
    n_ihead: usize,
    d_idx: usize,
    len: usize,
    kpool_r: usize,
    indexer_top_k: usize,
) -> Result<Vec<u32>> {
    let n_pools = kpool::n_pools(len, kpool_r);
    let mut scores = vec![0.0f32; n_pools];
    pool_scores(
        iq,
        w,
        pool_keys,
        Some(&bias[..n_pools]),
        n_ihead,
        d_idx,
        n_pools,
        &mut scores,
    )?;

    let k = kpool::select_k(n_pools, indexer_top_k, kpool_r)?;
    let picked: Vec<u32> = select_pools(&scores, k)
        .into_iter()
        .filter(|&p| scores[p as usize].is_finite())
        .collect();

    let mut cells = Vec::with_capacity(picked.len() * kpool_r);
    kpool::expand_pools(&picked, len, kpool_r, &mut cells);
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KPOOL: usize = 4;
    const TOP_K: usize = 2048;

    /// With equal gates and no positional encoding, pooling is a plain mean of
    /// the member keys, per channel.
    #[test]
    fn uniform_gates_pool_to_the_mean() {
        let (r, d) = (KPOOL, 3);
        let keys = vec![
            1.0f32, 10.0, 100.0, // slot 0
            2.0, 20.0, 200.0, // slot 1
            3.0, 30.0, 300.0, // slot 2
            4.0, 40.0, 400.0, // slot 3
        ];
        let gates = vec![0.0f32; r * d];
        let ape = vec![0.0f32; r * d];
        let mut out = vec![0.0f32; d];
        pooled_key(&keys, &gates, &ape, r, d, &mut out).unwrap();

        assert!((out[0] - 2.5).abs() < 1e-5);
        assert!((out[1] - 25.0).abs() < 1e-4);
        assert!((out[2] - 250.0).abs() < 1e-3);
    }

    /// A dominant gate on one slot selects that slot's key.
    #[test]
    fn a_dominant_gate_selects_its_slot() {
        let (r, d) = (KPOOL, 2);
        let keys = vec![
            1.0f32, -1.0, //
            2.0, -2.0, //
            3.0, -3.0, //
            4.0, -4.0,
        ];
        let mut gates = vec![0.0f32; r * d];
        // Slot 2 dominates, in both channels.
        gates[2 * d] = 40.0;
        gates[2 * d + 1] = 40.0;
        let ape = vec![0.0f32; r * d];
        let mut out = vec![0.0f32; d];
        pooled_key(&keys, &gates, &ape, r, d, &mut out).unwrap();

        assert!((out[0] - 3.0).abs() < 1e-4, "got {}", out[0]);
        assert!((out[1] + 3.0).abs() < 1e-4, "got {}", out[1]);
    }

    /// The softmax is over slots **per channel**, so one channel can take slot 0
    /// while another takes slot 3. A softmax over channels could not do this,
    /// and neither could a single shared slot weighting.
    #[test]
    fn pooling_is_independent_per_channel() {
        let (r, d) = (KPOOL, 2);
        let keys = vec![
            7.0f32, 70.0, //
            0.0, 0.0, //
            0.0, 0.0, //
            9.0, 90.0,
        ];
        let mut gates = vec![0.0f32; r * d];
        gates[0 * d + 0] = 40.0; // channel 0 -> slot 0
        gates[3 * d + 1] = 40.0; // channel 1 -> slot 3
        let ape = vec![0.0f32; r * d];
        let mut out = vec![0.0f32; d];
        pooled_key(&keys, &gates, &ape, r, d, &mut out).unwrap();

        assert!((out[0] - 7.0).abs() < 1e-3, "channel 0 -> slot 0, got {}", out[0]);
        assert!((out[1] - 90.0).abs() < 1e-2, "channel 1 -> slot 3, got {}", out[1]);
    }

    /// `ape` is added **before** the softmax, so it can move the slot weighting.
    /// Applied afterwards it could only scale the result.
    #[test]
    fn ape_is_added_before_the_softmax() {
        let (r, d) = (KPOOL, 1);
        let keys = vec![1.0f32, 2.0, 3.0, 4.0];
        let gates = vec![0.0f32; r * d];

        let mut flat = vec![0.0f32; d];
        pooled_key(&keys, &gates, &vec![0.0; r * d], r, d, &mut flat).unwrap();
        assert!((flat[0] - 2.5).abs() < 1e-5);

        // An APE favouring slot 3 must pull the pooled key toward 4.0.
        let mut ape = vec![0.0f32; r * d];
        ape[3] = 40.0;
        let mut biased = vec![0.0f32; d];
        pooled_key(&keys, &gates, &ape, r, d, &mut biased).unwrap();
        assert!(
            (biased[0] - 4.0).abs() < 1e-4,
            "ape must reweight the slots, got {}",
            biased[0]
        );
    }

    /// The money test for stage 2. One head, a negative dot and a negative head
    /// weight: the ReLU before the weighting gives 0, after it gives a positive
    /// score. The reference puts it before.
    #[test]
    fn relu_precedes_the_head_weighting() {
        let (n_ihead, d, n_pools) = (1, 2, 1);
        // iq . pool_key = -1
        let iq = vec![1.0f32, 0.0];
        let pool_keys = vec![-1.0f32, 0.0];
        let w = vec![-1.0f32];

        let mut out = vec![0.0f32; n_pools];
        pool_scores(&iq, &w, &pool_keys, None, n_ihead, d, n_pools, &mut out).unwrap();

        assert_eq!(
            out[0], 0.0,
            "relu(-1) * -1 = 0; getting a positive score means the relu moved \
             after the head weighting"
        );

        // Sanity: a positive dot with a negative weight does go negative, so the
        // zero above is the ReLU and not a sign bug.
        let pos_keys = vec![1.0f32, 0.0];
        pool_scores(&iq, &w, &pos_keys, None, n_ihead, d, n_pools, &mut out).unwrap();
        assert!(out[0] < 0.0, "sign-unconstrained weights must stay signed");
    }

    #[test]
    fn scores_apply_the_head_scale() {
        let (n_ihead, d, n_pools) = (2, 4, 1);
        // Each head's dot is 1.0.
        let iq = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let pool_keys = vec![1.0f32, 0.0, 0.0, 0.0];
        let w = vec![1.0f32, 1.0];

        let mut out = vec![0.0f32; n_pools];
        pool_scores(&iq, &w, &pool_keys, None, n_ihead, d, n_pools, &mut out).unwrap();

        let want = 2.0 * score_scale(d, n_ihead);
        assert!((out[0] - want).abs() < 1e-6, "got {} want {want}", out[0]);
        // Released geometry: 1/sqrt(128*32) = 1/64.
        assert!((score_scale(128, 32) - 1.0 / 64.0).abs() < 1e-7);
    }

    #[test]
    fn bias_makes_a_pool_unselectable() {
        let (n_ihead, d, n_pools) = (1, 1, 3);
        let iq = vec![1.0f32];
        let pool_keys = vec![3.0f32, 2.0, 1.0];
        let w = vec![1.0f32];
        // Mask the best-scoring pool.
        let bias = vec![f32::NEG_INFINITY, 0.0, 0.0];

        let mut out = vec![0.0f32; n_pools];
        pool_scores(
            &iq, &w, &pool_keys, Some(&bias), n_ihead, d, n_pools, &mut out,
        )
        .unwrap();
        assert!(out[0].is_infinite() && out[0] < 0.0);

        let picked = select_pools(&out, 2);
        assert_eq!(picked, vec![1, 2], "a masked pool must not win a slot");
    }

    #[test]
    fn select_pools_is_descending_with_low_index_ties() {
        let scores = vec![1.0f32, 3.0, 3.0, 0.5];
        assert_eq!(select_pools(&scores, 3), vec![1, 2, 0]);
        // Clamped to what exists.
        assert_eq!(select_pools(&scores, 99).len(), 4);
        // NaN sorts last.
        let nan = vec![f32::NAN, 1.0, 2.0];
        assert_eq!(select_pools(&nan, 2), vec![2, 1]);
    }

    /// End to end against the cache geometry: every returned cell must be an
    /// allowed candidate for that query, and the count must respect `top_k`.
    #[test]
    fn candidates_stay_inside_the_cand_mask() {
        let (n_ihead, d) = (2, 4);
        let kp = KPOOL;

        for len in [16usize, 33, 64] {
            for q in [0usize, 7, 8, len - 1] {
                if q >= len {
                    continue;
                }
                let n_pools = kpool::n_pools(len, kp);

                // Arbitrary but deterministic pooled keys and query.
                let pool_keys: Vec<f32> = (0..n_pools * d)
                    .map(|i| ((i * 31 % 17) as f32 - 8.0) / 8.0)
                    .collect();
                let iq: Vec<f32> = (0..n_ihead * d)
                    .map(|i| ((i * 13 % 11) as f32 - 5.0) / 5.0)
                    .collect();
                let w = vec![0.7f32, -0.3];

                let mut bias = vec![0.0f32; n_pools];
                kpool::pool_bias_row(q, len, kp, &mut bias).unwrap();

                let cells =
                    candidates(&iq, &w, &pool_keys, &bias, n_ihead, d, len, kp, TOP_K).unwrap();

                let (mut sel, mut cand) = (vec![0.0f32; len], vec![0.0f32; len]);
                kpool::masks_row(q, len, kp, &mut sel, &mut cand).unwrap();

                for &c in &cells {
                    assert_eq!(
                        cand[c as usize], 0.0,
                        "len={len} q={q}: cell {c} is not a candidate"
                    );
                }
                assert!(cells.len() <= TOP_K);
                let mut u = cells.clone();
                u.sort_unstable();
                u.dedup();
                assert_eq!(u.len(), cells.len(), "pools must not overlap");

                // Every fully-visible complete pool is admissible, so with a
                // short history the selection covers all of them.
                let visible = kpool::n_visible_pools(q, kp).min(kpool::n_complete_pools(len, kp));
                if visible <= TOP_K / kp {
                    assert_eq!(
                        cells.len(),
                        visible * kp,
                        "len={len} q={q}: all {visible} visible pools should be taken"
                    );
                }
            }
        }
    }

    /// A query with no complete visible pool yields no candidates at all — the
    /// tail is attended densely instead, via `sel_mask`.
    #[test]
    fn no_visible_pool_yields_no_candidates() {
        let (n_ihead, d, kp) = (1, 2, KPOOL);
        let len = 8;
        let q = 2; // tail_start = 0, bo_vis = 0
        assert_eq!(kpool::n_visible_pools(q, kp), 0);

        let n_pools = kpool::n_pools(len, kp);
        let mut bias = vec![0.0f32; n_pools];
        kpool::pool_bias_row(q, len, kp, &mut bias).unwrap();
        assert!(bias.iter().all(|b| b.is_infinite()));

        let pool_keys = vec![1.0f32; n_pools * d];
        let iq = vec![1.0f32, 1.0];
        let w = vec![1.0f32];
        let cells = candidates(&iq, &w, &pool_keys, &bias, n_ihead, d, len, kp, TOP_K).unwrap();
        assert!(cells.is_empty(), "nothing is selectable yet, got {cells:?}");
    }

    #[test]
    fn rejects_mismatched_widths() {
        let ok = vec![0.0f32; 4];
        let mut out1 = vec![0.0f32; 1];
        assert!(pooled_key(&ok, &ok, &ok, 2, 2, &mut vec![0.0; 3]).is_err());
        assert!(pooled_key(&ok[..2], &ok, &ok, 2, 2, &mut vec![0.0; 2]).is_err());
        assert!(pooled_key(&ok, &ok, &ok, 0, 2, &mut vec![0.0; 2]).is_err());

        assert!(pool_scores(&ok, &[1.0], &ok, None, 1, 2, 1, &mut out1).is_err(), "iq width");
        assert!(pool_scores(&ok[..2], &[1.0, 1.0], &ok, None, 1, 2, 1, &mut out1).is_err(), "w width");
        assert!(
            pool_scores(&ok[..2], &[1.0], &ok[..2], Some(&ok[..3]), 1, 2, 1, &mut out1).is_err(),
            "bias width"
        );
    }

    /// Released geometry smoke: 32 indexer heads of 128, a 4096-cell history.
    #[test]
    fn released_geometry_scores_and_selects() {
        let (n_ihead, d, kp) = (32, 128, KPOOL);
        let len = 4096;
        let q = len - 1;
        let n_pools = kpool::n_pools(len, kp);
        assert_eq!(n_pools, 1024);

        let pool_keys: Vec<f32> = (0..n_pools * d)
            .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
            .collect();
        let iq: Vec<f32> = (0..n_ihead * d)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect();
        let w: Vec<f32> = (0..n_ihead).map(|h| ((h % 5) as f32 - 2.0) / 2.0).collect();

        let mut bias = vec![0.0f32; n_pools];
        kpool::pool_bias_row(q, len, kp, &mut bias).unwrap();

        let cells = candidates(&iq, &w, &pool_keys, &bias, n_ihead, d, len, kp, TOP_K).unwrap();
        // 1024 pools, select_k = 512, so exactly top_k cells come back.
        assert_eq!(kpool::select_k(n_pools, TOP_K, kp).unwrap(), 512);
        assert_eq!(cells.len(), TOP_K);
        assert!(cells.iter().all(|&c| (c as usize) < len));
    }
}
