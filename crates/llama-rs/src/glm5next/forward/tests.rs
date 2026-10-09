use super::*;
use crate::glm5next::LayerKind;

/// Deterministic pseudo-random weights, small enough not to saturate.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i * 2654435761 + seed * 40503) % 1009;
            (h as f32 - 504.0) / 5040.0
        })
        .collect()
}

/// A tiny model with the real *structure*: a dense KDA block, a MoE KDA
/// block and a MoE MLA block, driven by a per-layer kind array.
fn shape() -> Shape {
    let kinds = vec![LayerKind::Kda, LayerKind::Kda, LayerKind::Mla];
    Shape {
        n_embd: 8,
        n_vocab: 11,
        n_layer: 3,
        n_head: 2,
        kda_head_dim: 4,
        d_conv: 4,
        q_lora: 6,
        kv_lora: 4,
        qk_head: 4,
        v_head: 4,
        d_idx: 4,
        n_ihead: 2,
        kpool: 2,
        // n_select = 3, and select_k = 1 pool, so with 4+ tokens the
        // indexer genuinely leaves cells masked rather than selecting
        // everything visible.
        indexer_top_k: 2,
        n_expert: 4,
        n_expert_used: 2,
        n_ff_exp: 6,
        n_ff_shexp: 6,
        n_ff_dense: 10,
        n_dense_lead: 1,
        layer_kinds: kinds,
        hc_count: 4,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        norm_eps: 1e-6,
        kda_gate_lower_bound: -5.0,
        expert_weights_norm: true,
        expert_weights_scale: 2.5,
        swiglu_clamp_exp: vec![10.0; 3],
        swiglu_clamp_shexp: vec![10.0; 3],
        // Above n_select (3), so the MLA layer scores.
        max_len: 32,
    }
}

/// An owning [`ExpertFfn`] for the fixture: one per MoE layer, so each
/// carries a single layer and is addressed with `ord = 0`.
struct TestExperts {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    per: usize,
    ff: usize,
    n_embd: usize,
}

impl ExpertFfn for TestExperts {
    fn apply(
        &self,
        _ord: usize,
        e: usize,
        x: &[f32],
        limit: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let off = e * self.per;
        let (mut g, mut u, mut h) = (
            vec![0.0f32; self.ff],
            vec![0.0f32; self.ff],
            vec![0.0f32; self.ff],
        );
        matvec(&self.gate[off..off + self.per], x, &mut g)?;
        matvec(&self.up[off..off + self.per], x, &mut u)?;
        swiglu_clamped(&g, &u, limit, &mut h)?;
        matvec(&self.down[off..off + self.per], &h, out)?;
        let _ = self.n_embd;
        Ok(())
    }
}

/// Owns the buffers a [`ModelW`] borrows.
struct Owned {
    tok_embd: Vec<f32>,
    output_norm: Vec<f32>,
    output: Vec<f32>,
    per_layer: Vec<std::collections::BTreeMap<&'static str, Vec<f32>>>,
    /// `None` for the dense leading blocks.
    experts: Vec<Option<TestExperts>>,
}

fn weights(sh: &Shape) -> Owned {
    let mut experts: Vec<Option<TestExperts>> = Vec::new();
    let di = sh.d_inner();
    let hd = sh.kda_head_dim;
    let e = sh.n_embd;
    let mut per_layer = Vec::new();
    for il in 0..sh.n_layer {
        let mut m: std::collections::BTreeMap<&'static str, Vec<f32>> = Default::default();
        let s = il * 17 + 1;
        m.insert("attn_norm", vec![1.0; e]);
        m.insert("ffn_norm", vec![1.0; e]);
        for (k, n) in [
            ("hc_attn_fn", hc::MIX * sh.hc_count * e),
            ("hc_ffn_fn", hc::MIX * sh.hc_count * e),
        ] {
            m.insert(k, fill(n, s + k.len()));
        }
        m.insert("hc_attn_base", vec![0.0; hc::MIX]);
        m.insert("hc_ffn_base", vec![0.0; hc::MIX]);
        m.insert("hc_attn_scale", vec![1.0, 1.0, 1.0]);
        m.insert("hc_ffn_scale", vec![1.0, 1.0, 1.0]);

        match sh.layer_kinds[il] {
            LayerKind::Kda => {
                for (k, n) in [
                    ("q", di * e),
                    ("k", di * e),
                    ("v", di * e),
                    ("f_a", hd * e),
                    ("f_b", di * hd),
                    ("g_a", hd * e),
                    ("g_b", di * hd),
                    ("beta", sh.n_head * e),
                    ("attn_out", e * di),
                ] {
                    m.insert(k, fill(n, s + k.len() * 3));
                }
                for k in ["conv_q", "conv_k", "conv_v"] {
                    m.insert(k, fill(di * sh.d_conv, s + k.len() * 5));
                }
                m.insert("a", vec![-1.0; sh.n_head]);
                m.insert("dt_bias", vec![0.0; di]);
                m.insert("o_norm", vec![1.0; hd]);
            }
            LayerKind::Mla => {
                for (k, n) in [
                    ("q_a", sh.q_lora * e),
                    ("q_b", sh.n_head * sh.qk_head * sh.q_lora),
                    ("kv_a_mqa", sh.kv_lora * e),
                    ("k_b", sh.n_head * sh.kv_lora * sh.qk_head),
                    ("v_b", sh.n_head * sh.v_head * sh.kv_lora),
                    ("attn_out", e * sh.n_head * sh.v_head),
                    ("idx_attn_k", sh.d_idx * e),
                    ("idx_attn_q_b", sh.n_ihead * sh.d_idx * sh.q_lora),
                    ("idx_proj", sh.n_ihead * e),
                    ("idx_gate", sh.d_idx * e),
                    ("idx_ape", sh.kpool * sh.d_idx),
                ] {
                    m.insert(k, fill(n, s + k.len() * 7));
                }
                m.insert("q_a_norm", vec![1.0; sh.q_lora]);
                m.insert("kv_a_norm", vec![1.0; sh.kv_lora]);
                m.insert("idx_k_norm", vec![1.0; sh.d_idx]);
                m.insert("idx_k_norm_b", vec![0.0; sh.d_idx]);
            }
        }

        if il < sh.n_dense_lead {
            m.insert("ffn_gate", fill(sh.n_ff_dense * e, s + 101));
            m.insert("ffn_up", fill(sh.n_ff_dense * e, s + 103));
            m.insert("ffn_down", fill(e * sh.n_ff_dense, s + 107));
        } else {
            m.insert("router", fill(sh.n_expert * e, s + 109));
            m.insert("probs_b", vec![0.0; sh.n_expert]);
            experts.push(Some(TestExperts {
                gate: fill(sh.n_expert * sh.n_ff_exp * e, s + 113),
                up: fill(sh.n_expert * sh.n_ff_exp * e, s + 127),
                down: fill(sh.n_expert * e * sh.n_ff_exp, s + 131),
                per: sh.n_ff_exp * e,
                ff: sh.n_ff_exp,
                n_embd: e,
            }));
            m.insert("sh_gate", fill(sh.n_ff_shexp * e, s + 137));
            m.insert("sh_up", fill(sh.n_ff_shexp * e, s + 139));
            m.insert("sh_down", fill(e * sh.n_ff_shexp, s + 149));
        }
        if il < sh.n_dense_lead {
            experts.push(None);
        }
        per_layer.push(m);
    }
    Owned {
        tok_embd: fill(sh.n_vocab * sh.n_embd, 3),
        output_norm: vec![1.0; sh.n_embd],
        output: fill(sh.n_vocab * sh.n_embd, 5),
        per_layer,
        experts,
    }
}

fn model<'a>(sh: &Shape, o: &'a Owned) -> ModelW<'a> {
    let g = |il: usize, k: &str| -> &'a [f32] { o.per_layer[il][k].as_slice() };
    let mh = |il: usize, k: &str| -> Mat<'a> { Mat::Host(o.per_layer[il][k].as_slice()) };
    let mut layers = Vec::new();
    for il in 0..sh.n_layer {
        let attn = match sh.layer_kinds[il] {
            LayerKind::Kda => AttnW::Kda(KdaW {
                qk: Pair::Split(mh(il, "q"), mh(il, "k")),
                v: mh(il, "v"),
                conv_q: g(il, "conv_q"),
                conv_k: g(il, "conv_k"),
                conv_v: g(il, "conv_v"),
                fga: Pair::Split(mh(il, "f_a"), mh(il, "g_a")),
                f_b: mh(il, "f_b"),
                g_b: mh(il, "g_b"),
                beta: mh(il, "beta"),
                a: g(il, "a"),
                dt_bias: g(il, "dt_bias"),
                o_norm: g(il, "o_norm"),
                out: mh(il, "attn_out"),
            }),
            LayerKind::Mla => AttnW::Mla(MlaW {
                q_a: mh(il, "q_a"),
                q_a_norm: g(il, "q_a_norm"),
                q_b: mh(il, "q_b"),
                kv_a_mqa: mh(il, "kv_a_mqa"),
                kv_a_norm: g(il, "kv_a_norm"),
                k_b: Bat::Host(g(il, "k_b")),
                v_b: Bat::Host(g(il, "v_b")),
                out: mh(il, "attn_out"),
                indexer: IndexerW {
                    attn_k: mh(il, "idx_attn_k"),
                    attn_q_b: mh(il, "idx_attn_q_b"),
                    k_norm: g(il, "idx_k_norm"),
                    k_norm_bias: g(il, "idx_k_norm_b"),
                    proj: mh(il, "idx_proj"),
                    comp_gate: mh(il, "idx_gate"),
                    comp_ape: g(il, "idx_ape"),
                },
            }),
        };
        let ffn = if il < sh.n_dense_lead {
            FfnW::Dense {
                gate: mh(il, "ffn_gate"),
                up: mh(il, "ffn_up"),
                down: mh(il, "ffn_down"),
            }
        } else {
            FfnW::Moe(MoeW {
                router: mh(il, "router"),
                probs_b: g(il, "probs_b"),
                experts: o.experts[il]
                    .as_ref()
                    .expect("a MoE layer needs an expert source"),
                ord: 0,
                sh_gate_up: Pair::Split(mh(il, "sh_gate"), mh(il, "sh_up")),
                sh_down: mh(il, "sh_down"),
            })
        };
        layers.push(LayerW {
            attn_norm: g(il, "attn_norm"),
            ffn_norm: g(il, "ffn_norm"),
            hc_attn: HcW {
                fn_: mh(il, "hc_attn_fn"),
                base: g(il, "hc_attn_base"),
                scale: g(il, "hc_attn_scale"),
            },
            hc_ffn: HcW {
                fn_: mh(il, "hc_ffn_fn"),
                base: g(il, "hc_ffn_base"),
                scale: g(il, "hc_ffn_scale"),
            },
            attn,
            ffn,
        });
    }
    ModelW {
        tok_embd: &o.tok_embd,
        output_norm: &o.output_norm,
        output: Mat::Host(&o.output),
        layers,
    }
}

#[test]
fn a_prompt_produces_finite_varied_logits() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let mut st = State::new(&sh).unwrap();

    let tokens = [1u32, 4, 2, 7, 0, 3];
    let logits = forward_prompt(&sh, &m, &mut st, &tokens).unwrap();

    assert_eq!(logits.len(), sh.n_vocab);
    assert!(logits.iter().all(|x| x.is_finite()), "logits must be finite");
    let first = logits[0];
    assert!(
        logits.iter().any(|&x| (x - first).abs() > 1e-6),
        "logits must not be uniform: {logits:?}"
    );
    assert_eq!(st.len, tokens.len());
}

/// A chunk must give exactly what the one-token loop gives.
///
/// Bit-identical, not close: the batched pass reorders nothing that the
/// arithmetic depends on. It walks the layers once instead of once a token, but
/// within a layer the attention still runs token by token in position order
/// (KDA is a recurrence, MLA appends to the latent cache), and the FFN's default
/// `apply_batch` is the per-token loop. A real implementation may reassociate
/// and is held to a tolerance on the released weights instead; this is the gate
/// on the scaffolding -- the state threading, the per-token hyper-connection
/// mixes, the position arithmetic.
#[test]
fn a_chunk_matches_the_one_token_loop() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let tokens = [1u32, 4, 2, 7, 0, 3];

    let mut seq = State::new(&sh).unwrap();
    let want = forward_prompt(&sh, &m, &mut seq, &tokens).unwrap();

    let mut bat = State::new(&sh).unwrap();
    let got = forward_chunk(&sh, &m, &mut bat, &tokens).unwrap();

    assert_eq!(got.len(), want.len());
    assert_eq!(got, want, "a chunk diverged from the one-token loop");
    assert_eq!(bat.len, seq.len, "the chunk left the state at a different length");
}

/// Chunk boundaries must not matter: the state carries across them.
#[test]
fn chunks_of_different_sizes_agree() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let tokens = [1u32, 4, 2, 7, 0, 3, 5, 2];

    let mut whole = State::new(&sh).unwrap();
    let want = forward_chunk(&sh, &m, &mut whole, &tokens).unwrap();

    // The same tokens in two chunks, then in three.
    for splits in [vec![5usize, 3], vec![2, 2, 4], vec![1, 6, 1]] {
        let mut st = State::new(&sh).unwrap();
        let mut at = 0usize;
        let mut last = Vec::new();
        for len in &splits {
            last = forward_chunk(&sh, &m, &mut st, &tokens[at..at + len]).unwrap();
            at += len;
        }
        assert_eq!(at, tokens.len());
        assert_eq!(last, want, "chunking as {splits:?} changed the answer");
        assert_eq!(st.len, whole.len);
    }
}

/// And a chunk of one is the one-token path.
#[test]
fn a_chunk_of_one_is_a_token() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);

    let mut a = State::new(&sh).unwrap();
    let want = forward_token(&sh, &m, &mut a, 3).unwrap();
    let mut b = State::new(&sh).unwrap();
    let got = forward_chunk(&sh, &m, &mut b, &[3]).unwrap();
    assert_eq!(got, want);
}

/// A restored snapshot continues a sequence exactly as the original would.
///
/// This is the gate on the prompt cache: the point of keeping a prompt's state
/// is that the tokens after it come out the same, so what is checked is not the
/// snapshot's bytes but the next token's logits.
#[test]
fn a_restored_snapshot_continues_the_sequence() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let prompt = [1u32, 4, 2, 7];
    let after = [3u32, 5];

    // Straight through: prompt, then two more tokens.
    let mut direct = State::new(&sh).unwrap();
    forward_prompt(&sh, &m, &mut direct, &prompt).unwrap();
    let mut want = Vec::new();
    for &t in &after {
        want = forward_token(&sh, &m, &mut direct, t).unwrap();
    }

    // The same, but the prompt's state came off a snapshot.
    let mut taken = State::new(&sh).unwrap();
    forward_prompt(&sh, &m, &mut taken, &prompt).unwrap();
    let snap = taken.snapshot(sh.kv_lora).unwrap();
    assert_eq!(snap.len, prompt.len());

    let mut fresh = State::new(&sh).unwrap();
    fresh.restore(&snap, sh.kv_lora).unwrap();
    assert_eq!(fresh.len, prompt.len());
    let mut got = Vec::new();
    for &t in &after {
        got = forward_token(&sh, &m, &mut fresh, t).unwrap();
    }

    assert_eq!(got, want, "a restored state diverged from one built in place");
}

/// Restoring over a longer state must not leave its rows behind.
#[test]
fn restoring_a_shorter_state_clears_what_was_there() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);

    // A short snapshot.
    let mut short = State::new(&sh).unwrap();
    forward_prompt(&sh, &m, &mut short, &[1u32, 4]).unwrap();
    let snap = short.snapshot(sh.kv_lora).unwrap();

    // A state that has seen more, then restored back to the short one.
    let mut long = State::new(&sh).unwrap();
    forward_prompt(&sh, &m, &mut long, &[7u32, 0, 3, 5, 2, 6]).unwrap();
    long.restore(&snap, sh.kv_lora).unwrap();

    // It must now behave exactly like the short one.
    let mut a = long;
    let mut b = State::new(&sh).unwrap();
    b.restore(&snap, sh.kv_lora).unwrap();
    assert_eq!(
        forward_token(&sh, &m, &mut a, 3).unwrap(),
        forward_token(&sh, &m, &mut b, 3).unwrap(),
        "rows from the longer sequence survived the restore"
    );
}

/// The wire format round-trips, and refuses damage.
#[test]
fn a_snapshot_survives_encoding() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let mut st = State::new(&sh).unwrap();
    forward_prompt(&sh, &m, &mut st, &[1u32, 4, 2]).unwrap();
    let snap = st.snapshot(sh.kv_lora).unwrap();

    let mut bytes = Vec::new();
    snap.encode(&mut bytes);
    assert_eq!(bytes.len(), snap.encoded_len(), "encoded_len disagrees with encode");
    let back = StateSnapshot::decode(&bytes).expect("decode");
    assert_eq!(back, snap);

    // A truncated state is rejected rather than half-read.
    assert!(StateSnapshot::decode(&bytes[..bytes.len() - 4]).is_err());
    // So are trailing bytes, which would mean a format mismatch.
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(StateSnapshot::decode(&extra).is_err());
}

#[test]
fn the_same_prompt_is_deterministic() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let tokens = [2u32, 2, 5, 1];

    let mut a = State::new(&sh).unwrap();
    let la = forward_prompt(&sh, &m, &mut a, &tokens).unwrap();
    let mut b = State::new(&sh).unwrap();
    let lb = forward_prompt(&sh, &m, &mut b, &tokens).unwrap();
    assert_eq!(la, lb);

    // And a reset returns the state to its opening condition.
    a.reset();
    let lc = forward_prompt(&sh, &m, &mut a, &tokens).unwrap();
    assert_eq!(la, lc);
}

/// State must actually accumulate: the second occurrence of a token cannot
/// produce the same logits as the first, or attention and the recurrence are
/// not seeing history.
#[test]
fn history_changes_the_output_for_a_repeated_token() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let mut st = State::new(&sh).unwrap();

    let first = forward_token(&sh, &m, &mut st, 3).unwrap();
    let second = forward_token(&sh, &m, &mut st, 3).unwrap();
    let diff: f32 = first.iter().zip(&second).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "history must change the output, diff {diff}");
}

/// The caches advance as the geometry says: one latent row and one indexer
/// cell per token, and a pooled key every `kpool` tokens.
#[test]
fn caches_advance_with_the_geometry() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let mut st = State::new(&sh).unwrap();

    for t in 0..6u32 {
        forward_token(&sh, &m, &mut st, t % sh.n_vocab as u32).unwrap();
    }
    assert_eq!(st.len, 6);
    assert_eq!(kpool::n_complete_pools(st.len, sh.kpool), 3);

    // Every completed pool has a non-zero pooled key; the next one does not.
    let kc = st.kpool_cache();
    for p in 0..3 {
        let pooled = kc.pooled(0, p, sh.kpool).unwrap();
        assert!(
            pooled.iter().any(|&x| x != 0.0),
            "pool {p} should have been pooled"
        );
    }
}

/// A shape whose `n_select` exceeds the cache capacity, so the indexer never
/// scores and the MLA layer takes the plain causal path. Only `indexer_top_k`
/// changes, so it shares the sparse fixture's weights exactly.
fn dense_shape() -> Shape {
    let mut sh = shape();
    sh.indexer_top_k = 64;
    assert!(!kpool::indexer_scores(sh.max_len, sh.indexer_top_k, sh.kpool));
    sh
}

/// Below the indexer's threshold the MLA layer must still run, and still see
/// history, via dense causal attention.
#[test]
fn a_small_cache_takes_the_dense_attention_path() {
    let sh = dense_shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let mut st = State::new(&sh).unwrap();
    let first = forward_token(&sh, &m, &mut st, 1).unwrap();
    forward_prompt(&sh, &m, &mut st, &[2, 3, 4, 5]).unwrap();
    assert_eq!(st.len, 5);
    // The dense path is not a no-op: a later token differs from the first.
    let later = forward_token(&sh, &m, &mut st, 1).unwrap();
    let diff: f32 = first.iter().zip(&later).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "dense attention must accumulate history");
}

/// The sparse path must actually mask something. With `select_k = 1` of 3
/// visible pools it attends to 2 of 6 cells, so its logits must differ from
/// the dense run over the same weights and the same prompt.
#[test]
fn the_sparse_path_differs_from_the_dense_path() {
    let sparse = shape();
    let dense = dense_shape();
    assert_eq!(kpool::n_select(sparse.indexer_top_k, sparse.kpool), 3);
    assert!(kpool::indexer_scores(sparse.max_len, sparse.indexer_top_k, sparse.kpool));
    // 3 complete pools at len 6, but only one may be selected.
    assert_eq!(kpool::select_k(3, sparse.indexer_top_k, sparse.kpool).unwrap(), 1);

    let o = weights(&sparse);
    let tokens = [1u32, 2, 3, 4, 0, 6];

    let ms = model(&sparse, &o);
    let mut ss = State::new(&sparse).unwrap();
    let sparse_logits = forward_prompt(&sparse, &ms, &mut ss, &tokens).unwrap();

    let md = model(&dense, &o);
    let mut sd = State::new(&dense).unwrap();
    let dense_logits = forward_prompt(&dense, &md, &mut sd, &tokens).unwrap();

    assert!(sparse_logits.iter().all(|x| x.is_finite()));
    assert!(dense_logits.iter().all(|x| x.is_finite()));
    let diff: f32 = sparse_logits
        .iter()
        .zip(&dense_logits)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-7,
        "masking 4 of 6 cells must change the logits, diff {diff}"
    );
}

/// `hc::init` then `hc::mean` is the identity, so a layer whose sublayers
/// output zero and whose mixer is the identity leaves the stream alone. This
/// pins that the loop's HC wrapping is a residual path, not a replacement.
#[test]
fn mean_of_the_initial_stream_is_the_embedding() {
    let embd = vec![0.25f32, -1.5, 3.0, 0.0];
    let s = hc::init(&embd);
    assert_eq!(hc::mean(&s), embd);
}

#[test]
fn clamped_swiglu_matches_the_reference_branch() {
    // limit applies to silu(gate) as an UPPER bound and to up symmetrically.
    let gate = vec![40.0f32, -40.0, 1.0];
    let up = vec![100.0f32, -100.0, 2.0];
    let mut out = vec![0.0f32; 3];
    swiglu_clamped(&gate, &up, 10.0, &mut out).unwrap();

    // silu(40) ~ 40 -> clamped to 10; up 100 -> clamped to 10.
    assert!((out[0] - 100.0).abs() < 1e-3, "got {}", out[0]);
    // silu(-40) ~ 0, below the limit, untouched; up -100 -> -10.
    assert!(out[1].abs() < 1e-3, "got {}", out[1]);
    // Well inside the limit: a plain SwiGLU.
    assert!((out[2] - silu(1.0) * 2.0).abs() < 1e-6);

    // A zero limit disables the clamp.
    let mut raw = vec![0.0f32; 3];
    swiglu_clamped(&gate, &up, 0.0, &mut raw).unwrap();
    assert!(raw[0] > 1000.0, "unclamped, got {}", raw[0]);
}

#[test]
fn rejects_a_bad_token_or_an_overfull_cache() {
    let sh = shape();
    let o = weights(&sh);
    let m = model(&sh, &o);
    let mut st = State::new(&sh).unwrap();
    assert!(forward_token(&sh, &m, &mut st, 999).is_err(), "bad token");
    assert!(forward_prompt(&sh, &m, &mut st, &[]).is_err(), "empty prompt");

    let mut tiny = shape();
    tiny.max_len = 2;
    let mut st2 = State::new(&tiny).unwrap();
    let m2 = model(&tiny, &o);
    forward_token(&tiny, &m2, &mut st2, 1).unwrap();
    forward_token(&tiny, &m2, &mut st2, 1).unwrap();
    assert!(
        forward_token(&tiny, &m2, &mut st2, 1).is_err(),
        "past capacity"
    );
}
