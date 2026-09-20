// VENDORED-LOCAL: whole module. GLM-5.3-Flash absorbed MLA.
//! Multi-head latent attention in **absorbed** form, over the sparse candidate
//! set the indexer picked.
//!
//! Reference: llama.cpp `llama_model_glm5next::graph::build_dsa_layer` and
//! `llm_graph_context::build_attn_sparse` (PR #27754, pinned at `86ebfef`).
//! The same absorbed formulation `deepseek2` / `glm-dsa` use.
//!
//! ## Why absorbed
//!
//! The naive form projects the cached latent up to per-head keys and values,
//! which needs a V cache as wide as `n_head * v_head_dim`. The absorbed form
//! instead pushes `wk_b` onto the *query* and `wv_b` onto the *output*, so the
//! cache holds only the `kv_lora_rank`-wide latent — 512 floats per cell instead
//! of 16384 — and attention becomes MQA over a single head of keys, with **V the
//! same latent row as K**.
//!
//! ```text
//! qr      = RMSNorm(wq_a @ x)                  // [q_lora_rank]
//! q[h]    = (wq_b @ qr)[h]                     // [qk_head_dim] per head
//! q_abs[h]= wk_b[h] @ q[h]                     // [kv_lora_rank]  <- absorbed
//! latent  = RMSNorm(wkv_a_mqa @ x)             // [kv_lora_rank], the cached row
//! s[h][t] = (q_abs[h] . latent[t]) * kq_scale
//! p[h]    = softmax(s[h] + mask)
//! c[h]    = sum_t p[h][t] * latent[t]          // [kv_lora_rank]
//! o[h]    = wv_b[h] @ c[h]                     // [v_head_dim]    <- absorbed
//! out     = wo @ concat_h(o[h])                // [n_embd]
//! ```
//!
//! ## Two things to get right
//!
//! **`kq_scale` is over the MLA head size, not the absorbed width.** The
//! reference is explicit: `1/sqrt(n_embd_head_k_mla)` = `1/sqrt(256)` = `1/16`,
//! **not** `1/sqrt(kv_lora_rank)`. The dot product runs over 512 absorbed
//! channels but the scale stays at the 256-wide head. See [`kq_scale`].
//!
//! **No RoPE.** `build_dsa_layer` asserts `hparams.n_rot() == 0`. DeepSeek-V4.1's
//! `dsv41::attention` rotates its queries and then undoes the rotation; porting
//! that here would be wrong.
//!
//! ## The attention mask ([`attn_mask`])
//!
//! `build_attn_sparse` composes four things additively:
//!
//! ```text
//! mask = dup(sel_mask)                  // the dense tail, granted 0
//! mask = set_rows(mask, zeros, top_k)   // scatter 0 into the selected cells
//! mask = mask + cand_mask               // bound to legal candidates
//! mask = mask + kq_mask                 // empty / future cells stay masked
//! ```
//!
//! The scatter writes a **constant 0, never the cell's own bias** — the
//! reference warns that scattering `-inf` "would ERASE a zero granted to the
//! tail". So selection is purely additive: it can grant a cell access, never
//! revoke it. Adding `cand_mask` afterwards is the bound that keeps a spurious
//! selection out, and adding `kq_mask` is "load bearing: keeps an empty, future
//! or foreign-sequence cell masked whatever top-k said".
//!
//! Net effect, which is what this module computes for the sparse path:
//! `(tail OR selected) AND candidate AND causal`.
//!
//! **There is a second path.** When the indexer does not score — capacity at or
//! below `n_select`, per [`kpool::indexer_scores`] — `build_dsa_layer` passes no
//! `top_k` and falls through to plain `build_attn`, which applies only
//! `kq_mask`: dense causal attention that never consults `sel_mask` or
//! `cand_mask`. [`attn_mask`] takes `None` for that, and it is **not** the same
//! as `Some(&[])`.
//!
//! **Wired into `forward`: no.** With this, piece (3)'s pieces all exist; the
//! layer loop that calls them is still to come.

use super::kpool;
use crate::{LlamaError, Result};

/// `1 / sqrt(qk_head_dim)` — the MLA head size, **not** the absorbed
/// `kv_lora_rank` width the dot product actually runs over.
pub fn kq_scale(qk_head_dim: usize) -> f32 {
    1.0 / (qk_head_dim as f32).sqrt()
}

/// Absorb `wk_b` into the query: `q_abs[h] = wk_b[h] @ q[h]`, mapping each
/// head's `qk_head_dim` query into the `kv_lora_rank` latent space.
///
/// * `wk_b` — `[n_head, kv_lora, qk_head]` row-major, as the loader presents
///   GGUF `attn_k_b` (`[qk_head, kv_lora, n_head]` reversed).
/// * `q` — `[n_head, qk_head]`.
/// * `out` — `[n_head, kv_lora]`.
pub fn absorb_query(
    wk_b: &[f32],
    q: &[f32],
    n_head: usize,
    kv_lora: usize,
    qk_head: usize,
    out: &mut [f32],
) -> Result<()> {
    if n_head == 0 || kv_lora == 0 || qk_head == 0 {
        return Err(LlamaError::Config(
            "mla: n_head, kv_lora and qk_head must be > 0".into(),
        ));
    }
    if wk_b.len() != n_head * kv_lora * qk_head {
        return Err(LlamaError::Config(format!(
            "mla: wk_b is {} wide, expected n_head*kv_lora*qk_head = {}",
            wk_b.len(),
            n_head * kv_lora * qk_head
        )));
    }
    if q.len() != n_head * qk_head {
        return Err(LlamaError::Config(format!(
            "mla: q is {} wide, expected n_head*qk_head = {}",
            q.len(),
            n_head * qk_head
        )));
    }
    if out.len() != n_head * kv_lora {
        return Err(LlamaError::Config(format!(
            "mla: out is {} wide, expected n_head*kv_lora = {}",
            out.len(),
            n_head * kv_lora
        )));
    }

    for h in 0..n_head {
        let w = &wk_b[h * kv_lora * qk_head..(h + 1) * kv_lora * qk_head];
        let qh = &q[h * qk_head..(h + 1) * qk_head];
        let oh = &mut out[h * kv_lora..(h + 1) * kv_lora];
        for (c, o) in oh.iter_mut().enumerate() {
            let row = &w[c * qk_head..(c + 1) * qk_head];
            *o = row.iter().zip(qh).map(|(a, b)| a * b).sum();
        }
    }
    Ok(())
}

/// Build one query's additive attention mask over `len` cells, as `0.0` /
/// `-inf`.
///
/// `selected` distinguishes the two paths `build_dsa_layer` takes, which are
/// **not** the same and must not be confused:
///
///   * **`None` — the dense path.** When the indexer does not score
///     ([`kpool::indexer_scores`] is false for the cache capacity), the
///     reference passes no `top_k` and falls through to plain `build_attn` with
///     only `kq_mask`: **every** cell at or before `q` is attended. It does not
///     consult `sel_mask` or `cand_mask` at all.
///   * **`Some(cells)` — the sparse path.**
///     `(tail OR selected) AND candidate AND causal`, per
///     `build_attn_sparse`. `Some(&[])` is legitimate and is not the same as
///     `None`: with no complete visible pool the tail already spans every
///     visible cell, so the result is dense anyway — but it gets there through
///     the masks rather than around them.
///
/// Getting this wrong is silent. A `Some(&[])` on a short context whose `q`
/// lands on a pool boundary yields an entirely masked row, and attention would
/// return zeros rather than a dense result.
pub fn attn_mask(
    q: usize,
    len: usize,
    kpool_r: usize,
    selected: Option<&[u32]>,
    out: &mut [f32],
) -> Result<()> {
    if out.len() < len {
        return Err(LlamaError::Config(format!(
            "mla: mask is {} wide, need at least len = {len}",
            out.len()
        )));
    }
    let Some(selected) = selected else {
        // The dense path: plain causal, exactly what `build_attn` applies.
        for (c, o) in out.iter_mut().enumerate() {
            *o = if c < len && c <= q {
                0.0
            } else {
                f32::NEG_INFINITY
            };
        }
        return Ok(());
    };

    let mut sel = vec![0.0f32; len];
    let mut cand = vec![0.0f32; len];
    kpool::masks_row(q, len, kpool_r, &mut sel, &mut cand)?;

    // Start from the tail's grants, then let selection add to them. A scatter
    // must never write the cell's own bias here, or it would erase a grant the
    // tail already made.
    for (c, o) in out.iter_mut().enumerate() {
        *o = if c < len { sel[c] } else { f32::NEG_INFINITY };
    }
    for &c in selected {
        let c = c as usize;
        if c < len {
            out[c] = 0.0;
        }
    }
    // The bound: a selection outside the candidate set, or past the causal
    // boundary, is revoked here.
    for c in 0..len {
        if cand[c].is_infinite() {
            out[c] = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Absorbed MLA attention for one query against the cached latents.
///
/// * `q_abs` — `[n_head, kv_lora]` from [`absorb_query`].
/// * `latents` — `[len, kv_lora]`, the cached rows. Serves as **both** K and V.
/// * `mask` — `[len]` additive, from [`attn_mask`].
/// * `wv_b` — `[n_head, v_head, kv_lora]` row-major, as the loader presents
///   GGUF `attn_v_b` (`[kv_lora, v_head, n_head]` reversed).
/// * `out` — `[n_head * v_head]`, the concatenated per-head outputs, ready for
///   `wo`.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q_abs: &[f32],
    latents: &[f32],
    mask: &[f32],
    wv_b: &[f32],
    scale: f32,
    n_head: usize,
    kv_lora: usize,
    v_head: usize,
    len: usize,
    out: &mut [f32],
) -> Result<()> {
    if n_head == 0 || kv_lora == 0 || v_head == 0 {
        return Err(LlamaError::Config(
            "mla: n_head, kv_lora and v_head must be > 0".into(),
        ));
    }
    if q_abs.len() != n_head * kv_lora {
        return Err(LlamaError::Config(format!(
            "mla: q_abs is {} wide, expected n_head*kv_lora = {}",
            q_abs.len(),
            n_head * kv_lora
        )));
    }
    if latents.len() < len * kv_lora {
        return Err(LlamaError::Config(format!(
            "mla: latents is {} wide, need len*kv_lora = {}",
            latents.len(),
            len * kv_lora
        )));
    }
    if mask.len() < len {
        return Err(LlamaError::Config(format!(
            "mla: mask is {} wide, need len = {len}",
            mask.len()
        )));
    }
    if wv_b.len() != n_head * v_head * kv_lora {
        return Err(LlamaError::Config(format!(
            "mla: wv_b is {} wide, expected n_head*v_head*kv_lora = {}",
            wv_b.len(),
            n_head * v_head * kv_lora
        )));
    }
    if out.len() != n_head * v_head {
        return Err(LlamaError::Config(format!(
            "mla: out is {} wide, expected n_head*v_head = {}",
            out.len(),
            n_head * v_head
        )));
    }

    let mut p = vec![0.0f32; len];
    let mut ctx = vec![0.0f32; kv_lora];

    for h in 0..n_head {
        let qh = &q_abs[h * kv_lora..(h + 1) * kv_lora];

        // Scores against the single head of latent keys (absorbed MLA is MQA).
        let mut max = f32::NEG_INFINITY;
        for t in 0..len {
            let m = mask[t];
            // Any non-finite entry counts as masked. Catching +inf and NaN here
            // too keeps `exp(s - max)` from becoming NaN on a malformed mask.
            if !m.is_finite() {
                p[t] = f32::NEG_INFINITY;
                continue;
            }
            let row = &latents[t * kv_lora..(t + 1) * kv_lora];
            let s = qh.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() * scale + m;
            p[t] = s;
            if s > max {
                max = s;
            }
        }

        // A query with nothing visible contributes nothing rather than NaNs.
        if !max.is_finite() {
            out[h * v_head..(h + 1) * v_head].fill(0.0);
            continue;
        }

        let mut sum = 0.0f32;
        for s in p.iter_mut() {
            *s = if s.is_finite() { (*s - max).exp() } else { 0.0 };
            sum += *s;
        }
        let inv = 1.0 / sum;

        // V is the same latent row as K, so the context is in latent space.
        ctx.fill(0.0);
        for t in 0..len {
            let w = p[t] * inv;
            if w == 0.0 {
                continue;
            }
            let row = &latents[t * kv_lora..(t + 1) * kv_lora];
            for (c, &r) in ctx.iter_mut().zip(row) {
                *c += w * r;
            }
        }

        // Absorb wv_b on the way out: latent -> v_head_dim.
        let w = &wv_b[h * v_head * kv_lora..(h + 1) * v_head * kv_lora];
        let oh = &mut out[h * v_head..(h + 1) * v_head];
        for (e, o) in oh.iter_mut().enumerate() {
            let row = &w[e * kv_lora..(e + 1) * kv_lora];
            *o = row.iter().zip(ctx.iter()).map(|(a, b)| a * b).sum();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scale is over the MLA head size, not the absorbed latent width.
    #[test]
    fn scale_is_over_the_mla_head_not_the_latent() {
        // Released geometry: key_length_mla = 256, kv_lora_rank = 512.
        assert!((kq_scale(256) - 1.0 / 16.0).abs() < 1e-7);
        assert!(
            (kq_scale(256) - 1.0 / (512f32).sqrt()).abs() > 1e-3,
            "1/sqrt(256) must differ from 1/sqrt(512)"
        );
    }

    #[test]
    fn absorb_query_is_a_per_head_matvec() {
        let (n_head, kv_lora, qk_head) = (2, 3, 2);
        // head 0: identity-ish; head 1: doubles.
        let wk_b = vec![
            // head 0, rows of [qk_head]
            1.0f32, 0.0, //
            0.0, 1.0, //
            1.0, 1.0, //
            // head 1
            2.0, 0.0, //
            0.0, 2.0, //
            2.0, 2.0,
        ];
        let q = vec![3.0f32, 5.0, /* head 1 */ 1.0, 2.0];
        let mut out = vec![0.0f32; n_head * kv_lora];
        absorb_query(&wk_b, &q, n_head, kv_lora, qk_head, &mut out).unwrap();

        assert_eq!(&out[0..3], &[3.0, 5.0, 8.0]);
        assert_eq!(&out[3..6], &[2.0, 4.0, 6.0]);
    }

    /// With exactly one cell visible, softmax is 1 and the output is that
    /// latent pushed through `wv_b`.
    #[test]
    fn a_single_visible_cell_returns_its_projected_latent() {
        let (n_head, kv_lora, v_head, len) = (1, 2, 3, 3);
        let q_abs = vec![1.0f32, 1.0];
        let latents = vec![
            9.0f32, 9.0, // cell 0 - masked
            2.0, 4.0, // cell 1 - visible
            7.0, 7.0, // cell 2 - masked
        ];
        let mask = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        // wv_b rows: sum, first, second
        let wv_b = vec![
            1.0f32, 1.0, //
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let mut out = vec![0.0f32; n_head * v_head];
        attend(
            &q_abs, &latents, &mask, &wv_b, kq_scale(4), n_head, kv_lora, v_head, len, &mut out,
        )
        .unwrap();

        assert!((out[0] - 6.0).abs() < 1e-5, "sum of the latent");
        assert!((out[1] - 2.0).abs() < 1e-5);
        assert!((out[2] - 4.0).abs() < 1e-5);
    }

    /// A masked cell must not contribute however large its score would be.
    #[test]
    fn masked_cells_cannot_contribute() {
        let (n_head, kv_lora, v_head, len) = (1, 1, 1, 2);
        let q_abs = vec![1.0f32];
        // Cell 0 would dominate any softmax.
        let latents = vec![1000.0f32, 1.0];
        let wv_b = vec![1.0f32];

        let mut open = vec![0.0f32; v_head];
        attend(
            &q_abs,
            &latents,
            &[0.0, 0.0],
            &wv_b,
            1.0,
            n_head,
            kv_lora,
            v_head,
            len,
            &mut open,
        )
        .unwrap();
        assert!((open[0] - 1000.0).abs() < 1e-3, "unmasked, cell 0 wins");

        let mut masked = vec![0.0f32; v_head];
        attend(
            &q_abs,
            &latents,
            &[f32::NEG_INFINITY, 0.0],
            &wv_b,
            1.0,
            n_head,
            kv_lora,
            v_head,
            len,
            &mut masked,
        )
        .unwrap();
        assert!((masked[0] - 1.0).abs() < 1e-5, "masked, only cell 1 remains");
    }

    /// An entirely masked row yields zeros, not NaNs.
    #[test]
    fn a_fully_masked_query_yields_zeros() {
        let mut out = vec![7.0f32; 2];
        attend(
            &[1.0, 1.0],
            &[1.0, 1.0, 2.0, 2.0],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY],
            &[1.0, 0.0, 0.0, 1.0],
            1.0,
            1,
            2,
            2,
            2,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    /// Selection grants access and never revokes it — the reference's warning
    /// about scattering the cell's bias instead of a constant zero.
    #[test]
    fn selection_only_grants_never_revokes_the_tail() {
        let (len, kp, q) = (9usize, 4usize, 8usize);
        // tail_start = 8, so cell 8 is the tail; pools 0-1 are candidates.
        let mut tail_only = vec![0.0f32; len];
        attn_mask(q, len, kp, Some(&[]), &mut tail_only).unwrap();
        assert_eq!(tail_only[8], 0.0, "the tail is granted with no selection");
        for c in 0..8 {
            assert!(tail_only[c].is_infinite(), "cell {c} not selected yet");
        }

        // Selecting pool 0's cells must not disturb the tail's grant.
        let mut with_sel = vec![0.0f32; len];
        attn_mask(q, len, kp, Some(&[0, 1, 2, 3]), &mut with_sel).unwrap();
        assert_eq!(with_sel[8], 0.0, "the tail's grant must survive the scatter");
        for c in 0..4 {
            assert_eq!(with_sel[c], 0.0, "cell {c} was selected");
        }
        for c in 4..8 {
            assert!(with_sel[c].is_infinite(), "cell {c} was not selected");
        }
    }

    /// `cand_mask` is the bound: a cell selected but not a legal candidate stays
    /// masked. Here cell 5 is in the future, so no selection may admit it.
    #[test]
    fn the_candidate_bound_revokes_a_spurious_selection() {
        let (len, kp, q) = (8usize, 4usize, 3usize);
        let mut mask = vec![0.0f32; len];
        // Try to select cells 4..8, all beyond the causal boundary.
        attn_mask(q, len, kp, Some(&[4, 5, 6, 7]), &mut mask).unwrap();
        for c in 4..8 {
            assert!(
                mask[c].is_infinite(),
                "cell {c} is in the future and must stay masked"
            );
        }
        // Pool 0 is complete and visible at q = 3, so those remain reachable.
        let mut ok = vec![0.0f32; len];
        attn_mask(q, len, kp, Some(&[0, 1, 2, 3]), &mut ok).unwrap();
        for c in 0..4 {
            assert_eq!(ok[c], 0.0, "cell {c} is a legal candidate");
        }
    }

    /// The dense path attends to **every** visible cell. It is not the same as
    /// the sparse path with an empty selection, and conflating them is silent:
    /// at `q = 99, kpool = 4` the tail is empty, so `Some(&[])` grants nothing
    /// at all and attention would return zeros.
    #[test]
    fn the_dense_path_is_not_an_empty_selection() {
        let (len, kp, q) = (100usize, 4usize, 99usize);
        // A 128-cell cache never reaches the indexer's scoring threshold.
        assert!(!kpool::indexer_scores(128, 2048, kp));

        let mut dense = vec![0.0f32; len];
        attn_mask(q, len, kp, None, &mut dense).unwrap();
        for c in 0..len {
            assert_eq!(dense[c], 0.0, "cell {c} must be attended on the dense path");
        }

        let mut empty_sel = vec![0.0f32; len];
        attn_mask(q, len, kp, Some(&[]), &mut empty_sel).unwrap();
        assert!(
            empty_sel.iter().all(|x| x.is_infinite()),
            "q lands on a pool boundary, so the tail is empty and nothing is              granted -- which is exactly why the dense path needs None"
        );
    }

    #[test]
    fn the_dense_path_is_still_causal() {
        let mut m = vec![0.0f32; 16];
        attn_mask(5, 10, 4, None, &mut m).unwrap();
        for c in 0..=5 {
            assert_eq!(m[c], 0.0, "cell {c} is at or before q");
        }
        for c in 6..16 {
            assert!(m[c].is_infinite(), "cell {c} is in the future or slack");
        }
    }

    /// A `+inf` or NaN mask entry must be treated as masked rather than
    /// producing NaN out of `exp(s - max)`.
    #[test]
    fn a_malformed_mask_entry_cannot_poison_the_output() {
        for bad in [f32::INFINITY, f32::NAN] {
            let mut out = vec![0.0f32; 1];
            attend(
                &[1.0],
                &[5.0, 1.0],
                &[bad, 0.0],
                &[1.0],
                1.0,
                1,
                1,
                1,
                2,
                &mut out,
            )
            .unwrap();
            assert!(out[0].is_finite(), "{bad} poisoned the output");
            assert!((out[0] - 1.0).abs() < 1e-5, "the bad cell must be dropped");
        }
    }

    #[test]
    fn mask_slack_past_len_is_masked() {
        let mut mask = vec![0.0f32; 16];
        attn_mask(8, 9, 4, Some(&[0, 1]), &mut mask).unwrap();
        for c in 9..16 {
            assert!(mask[c].is_infinite(), "slack cell {c}");
        }
    }

    #[test]
    fn rejects_mismatched_widths() {
        let ok = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 2];
        assert!(absorb_query(&ok, &ok, 2, 1, 2, &mut vec![0.0; 1]).is_err());
        assert!(absorb_query(&ok[..2], &ok, 1, 1, 2, &mut out).is_err());
        assert!(absorb_query(&ok, &ok, 0, 1, 2, &mut out).is_err());
        assert!(attn_mask(0, 8, 4, Some(&[]), &mut vec![0.0; 4]).is_err());
        assert!(attend(&ok, &ok, &ok, &ok, 1.0, 9, 1, 1, 1, &mut out).is_err());
    }

    /// Released geometry, end to end through the indexer's candidate set.
    #[test]
    fn released_geometry_attends_over_the_candidate_set() {
        let (n_head, kv_lora, qk_head, v_head) = (4, 512, 256, 256);
        let (len, kp) = (64usize, 4usize);
        let q_pos = len - 1;

        // Deterministic pseudo-weights; magnitudes kept small so the softmax
        // does not saturate.
        let wk_b: Vec<f32> = (0..n_head * kv_lora * qk_head)
            .map(|i| (((i % 97) as f32) - 48.0) / 4800.0)
            .collect();
        let q: Vec<f32> = (0..n_head * qk_head)
            .map(|i| (((i % 31) as f32) - 15.0) / 15.0)
            .collect();
        let mut q_abs = vec![0.0f32; n_head * kv_lora];
        absorb_query(&wk_b, &q, n_head, kv_lora, qk_head, &mut q_abs).unwrap();

        let latents: Vec<f32> = (0..len * kv_lora)
            .map(|i| (((i % 53) as f32) - 26.0) / 260.0)
            .collect();
        let wv_b: Vec<f32> = (0..n_head * v_head * kv_lora)
            .map(|i| (((i % 41) as f32) - 20.0) / 2000.0)
            .collect();

        // Take only TWO of the 16 visible pools, so most cells stay masked and
        // the sparse path is genuinely exercised. Selecting every visible pool
        // would make the mask all-zeros and test nothing.
        let n_pools = kpool::n_pools(len, kp);
        assert_eq!(n_pools, 16);
        let bo_vis = kpool::n_visible_pools(q_pos, kp).min(kpool::n_complete_pools(len, kp));
        assert_eq!(bo_vis, 16, "the whole history is poolable at this q");

        let mut cells = Vec::new();
        kpool::expand_pools(&[0, 1], len, kp, &mut cells);
        assert_eq!(cells.len(), 2 * kp);

        let mut mask = vec![0.0f32; len];
        attn_mask(q_pos, len, kp, Some(&cells), &mut mask).unwrap();
        let open = mask.iter().filter(|m| **m == 0.0).count();
        assert_eq!(open, 2 * kp, "only the two selected pools may be open");

        let run = |m: &[f32]| {
            let mut o = vec![0.0f32; n_head * v_head];
            attend(
                &q_abs, &latents, m, &wv_b, kq_scale(qk_head), n_head, kv_lora,
                v_head, len, &mut o,
            )
            .unwrap();
            o
        };

        let sparse = run(&mask);
        assert!(sparse.iter().all(|x| x.is_finite()), "output must stay finite");
        assert!(sparse.iter().any(|&x| x != 0.0), "output must not be all zeros");

        // A dense run over the same latents must differ, or the mask did nothing.
        let mut dense = vec![0.0f32; len];
        attn_mask(q_pos, len, kp, None, &mut dense).unwrap();
        let full = run(&dense);
        let diff: f32 = sparse.iter().zip(&full).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-4,
            "masking 56 of 64 cells must change the output, got diff {diff}"
        );
    }
}
