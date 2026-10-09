//! Drafting on the chain: the prediction layer's block and cache, a round of drafts, and the checks of drafted tokens.

use super::*;

impl Qwen35Chain {
    /// Whether the chain drafts and checks tokens (the model's multi-token-prediction layer on the device).
    pub(crate) fn drafts(&self, m: &Qwen35Model) -> bool {
        std::env::var_os("OAIY_NO_CHAIN").is_none() && self.state(m).is_some_and(|st| st.spec.is_some())
    }

    /// A check of `rows` tokens (`embeds` `[rows, d]`: the token sampled, then its drafts; at most [`SPEC_ROWS`]) after
    /// what `kv` holds: every row's logits (`[1, vocab]` each). The run is kept undoable ([`Self::rollback`]).
    pub(crate) fn check(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache) -> Option<Vec<Tensor>> {
        if !self.drafts(m) || rows == 0 || rows > SPEC_ROWS || embeds.numel() != rows * m.config.embedding_dim || kv.len + rows > kv.max_len {
            return None;
        }
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        let h = embeds.to_host();
        Some(self.run(m, st, chain, h.data(), rows, kv, true))
    }

    /// Undo a check's rows past its first `keep` (the sampled token and the drafts accepted): each delta net's state
    /// and conv window as they were before it, its first `keep` rows run through them again, and the caches cut back.
    pub(crate) fn rollback(&self, m: &Qwen35Model, kv: &mut KvCache, rows: usize, keep: usize) {
        let (Some(st), Some(chain)) = (self.state(m), m.backend.chain()) else { return };
        let Some(sp) = &st.spec else { return };
        assert!(keep >= 1 && keep <= rows && rows <= kv.len, "a rollback of {rows} rows to {keep}");
        if keep == rows {
            return;
        }
        let s = st.dims;
        let delta = DeltaNet { rows: keep, v_heads: s.nv, k_heads: s.nk, k_dim: s.dk, v_dim: s.dv, scale_q: 1.0 / (s.dv as f32).sqrt(), eps: m.config.rms_eps, sigmoid_gate: false };
        let mut rec = chain.begin();
        for (slot, &l) in st.ssm_layers.iter().enumerate() {
            let (Some(state), Some(conv)) = (kv.ssm_state[l].as_ref().and_then(|t| chain.aliased(t)), kv.ssm_conv[l].as_ref().and_then(|t| chain.aliased(t))) else {
                unreachable!("a check left layer {l}'s state the chain's")
            };
            let Mixer::Ssm { conv_w, a, dt, norm, .. } = &st.layers[l].mixer else { unreachable!("layer {l} is a delta net") };
            let (bs, bc) = &sp.backups[slot];
            let (qkv, ba) = &sp.inputs[slot];
            rec.copy(bs, 0, &state, 0, state.len);
            rec.copy(bc, 0, &conv, 0, conv.len);
            rec.ssm_conv(qkv, conv_w, &conv, &sp.scratch_conv, keep, s.ch, s.kern);
            // the outputs are not wanted: the check's were the accepted rows' already
            rec.delta_net(&sp.scratch_conv, &sp.scratch_conv, ba, a, dt, norm, &state, &sp.scratch_core, delta);
        }
        rec.finish();
        kv.len -= rows - keep;
        let mut at = sp.hid_at.lock().unwrap_or_else(|p| p.into_inner());
        at.1 = at.1.min(keep);
    }

    /// The prediction layer's cache at a prompt's chunk (`t` rows at `past`, its embeddings `emb`, its hidden states
    /// after the output norm `hidden`): its entries for the chunk's positions but the last (whose next token the chunk
    /// does not have), each from the row's hidden state and the next row's token, where its entries run unbroken from
    /// the cache's start up to the chunk (else its entries start over at the chunk). Where they run up to the position
    /// before it, and the run before kept its hidden state there (the chunk before's last row), that position's entry
    /// too: its next token is the chunk's first.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp_prompt(&self, m: &Qwen35Model, st: &State, sp: &Spec, chain: &dyn DeviceChain, rec: &mut dyn ggml_rs::ChainRecorder, emb: &[f32], hidden: &DeviceVec, t: usize, past: usize, owner: u64) {
        let s = st.dims;
        let mtp = m.mtp.as_ref().expect("a drafting chain's model has its layer");
        let mut g = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
        mtp_reserve(chain, &mut g, &s, past + t);
        let (hid_at, hid_rows) = *sp.hid_at.lock().unwrap_or_else(|p| p.into_inner());
        let carry = past > 0 && g.owner == owner && g.start == 0 && g.valid + 1 == past && hid_rows > 0 && hid_at + hid_rows == past;
        if !carry && (g.owner != owner || g.valid != past || g.start != 0) {
            // a chunk's attention reaches back to the cache's start: entries from 0, or none
            if past != 0 {
                g.owner = 0;
                g.valid = 0;
                return;
            }
            g.start = 0;
        }
        g.owner = owner;
        let lead = usize::from(carry);
        let rows = t - 1 + lead;
        if rows == 0 {
            return;
        }
        let at = past - lead;
        // the next tokens' embeddings and the hidden states (the kept one carried, then the chunk's rows but its last),
        // in a prompt's own vectors
        let v = |n: usize| chain.vec(n);
        let (e, en, hn, cat, x, xn) = (v(rows * s.d), v(rows * s.d), v(rows * s.d), v(rows * 2 * s.d), v(rows * s.d), v(rows * s.d));
        chain.upload(&e, &emb[(1 - lead) * s.d..t * s.d]);
        let eps = m.config.rms_eps;
        let hid = if carry {
            let h = v(rows * s.d);
            rec.copy(&sp.hid, (hid_rows - 1) * s.d, &h, 0, s.d);
            rec.copy(hidden, 0, &h, s.d, (t - 1) * s.d);
            h
        } else {
            first(hidden, rows * s.d)
        };
        rec.rmsnorm_rows(&e, &sp.enorm, &en, rows, eps);
        rec.rmsnorm_rows(&hid, &sp.hnorm, &hn, rows, eps);
        // (each row's two halves side by side: two dispatches, where a copy a half a row was 2,046 a chunk of 1,024,
        // each with its parameters to make)
        rec.store_rows(&en, &cat, rows, s.d, 0, 2 * s.d, 0);
        rec.store_rows(&hn, &cat, rows, s.d, 0, 2 * s.d, s.d);
        rec.matmul_rows(quant(&mtp.eh_proj), &cat, &x, rows);
        // its K and V into its cache (as its block makes them: the rest of the block, whose output nothing reads
        // here, not run)
        let Qwen35Block::Attention { attn_k, attn_v, .. } = &mtp.block else { unreachable!("the prediction layer attends") };
        let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
        let (k, vv, kn, table) = (v(rows * kvd), v(rows * kvd), v(rows * kvd), v(rows * s.rot));
        chain.upload(&table, &rope_table(m.config.rope_theta, s.rot, at, rows));
        rec.rmsnorm_rows(&x, &sp.attn_norm, &xn, rows, eps);
        rec.matmul_rows(quant(attn_k), &xn, &k, rows);
        rec.matmul_rows(quant(attn_v), &xn, &vv, rows);
        rec.rmsnorm_rows(&k, &sp.k_norm, &kn, rows * s.n_kv, eps);
        rec.rope_partial_rows(&kn, rows, s.n_kv, s.hd, s.rot, &table);
        rec.store_rows(&kn, &g.layer, rows, kvd, at, row, 0);
        rec.store_rows(&vv, &g.layer, rows, kvd, at, row, kvd);
        g.valid = at + rows;
    }

    /// Draft up to `k` tokens after `next` (the token sampled for position `kv.len`): the prediction layer's entries
    /// caught up to it first (each position from the trunk's hidden state there, kept from the last run, and the token
    /// after, `tokens` the tokens at positions `hidden's first + 1..=kv.len`, ending with `next`), its last giving the
    /// first draft; each further draft from the layer's own output and the draft before. A draft the layer gives less
    /// than [`DRAFT_MIN_P`] ends them (none where the first is such): a check's rows cost, and an unlikely draft is
    /// seldom taken. None where the chain does not draft or the hidden states it needs are not kept.
    pub(crate) fn draft(&self, m: &Qwen35Model, kv: &KvCache, tokens: &[u32], k: usize) -> Option<Vec<u32>> {
        if !self.drafts(m) || k == 0 {
            return None;
        }
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        let sp = st.spec.as_ref()?;
        let s = st.dims;
        let mtp = m.mtp.as_ref()?;
        let n = kv.len;
        let (hid_at, hid_rows) = *sp.hid_at.lock().unwrap_or_else(|p| p.into_inner());
        // the rows the hidden states cover, up to the trunk's last position
        if hid_rows == 0 || hid_at + hid_rows != n || tokens.len() < hid_rows {
            return None;
        }
        let tokens = &tokens[tokens.len() - hid_rows..];
        let mut g = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
        mtp_reserve(chain, &mut g, &s, n + k + 1);
        // entries from the hidden states' first row on; before it, the run of true ones if it reaches it, else none
        if g.owner != kv.id || g.valid < hid_at || g.valid > n {
            g.start = hid_at;
        }
        g.owner = kv.id;
        let eps = m.config.rms_eps;
        let wk = &sp.work;
        let mut drafts = Vec::with_capacity(k);
        // pass 0: the caught-up rows (the hidden states kept); then a row a draft (the layer's own output)
        for pass in 0..k {
            let (rows, at) = if pass == 0 { (hid_rows, hid_at) } else { (1, n + pass - 1) };
            let next_tokens: Vec<u32> = if pass == 0 { tokens.to_vec() } else { vec![drafts[pass - 1]] };
            let e = m.embed_text(&next_tokens).to_host();
            chain.upload(&wk.e, e.data());
            chain.upload(&wk.table, &rope_table(m.config.rope_theta, s.rot, at, rows));
            let (ev, hv, env, hnv, cat) = (first(&wk.e, rows * s.d), first(&wk.h, rows * s.d), first(&wk.en, rows * s.d), first(&wk.hn, rows * s.d), first(&wk.cat, rows * 2 * s.d));
            let mut rec = chain.begin();
            if pass == 0 {
                rec.copy(&sp.hid, 0, &hv, 0, rows * s.d);
            } else {
                rec.copy(&wk.last, 0, &hv, 0, s.d);
            }
            rec.rmsnorm_rows(&ev, &sp.enorm, &env, rows, eps);
            rec.rmsnorm_rows(&hv, &sp.hnorm, &hnv, rows, eps);
            rec.store_rows(&env, &cat, rows, s.d, 0, 2 * s.d, 0);
            rec.store_rows(&hnv, &cat, rows, s.d, 0, 2 * s.d, s.d);
            let w = MtpRows::of(wk, &s, rows);
            rec.matmul_rows(quant(&mtp.eh_proj), &cat, &w.x, rows);
            mtp_block(m, mtp, sp, &s, &mut *rec, &g, &w, rows, at, Some((&wk.q1, g.start)));
            // the head of the last row: the next draft; the layer's own output of it, the next pass's hidden state
            rec.copy(&w.x, (rows - 1) * s.d, &wk.last, 0, s.d);
            let xn1 = first(&wk.xn, s.d);
            rec.rmsnorm_rows(&wk.last, &sp.head_norm, &xn1, 1, eps);
            rec.matmul(quant(&m.output), &xn1, &wk.logits);
            // the draft and its probability under the layer (the largest logit's share of their exponentials)
            rec.argmax_softmax(&wk.logits, &wk.best);
            rec.read(&wk.best);
            let got = rec.finish().pop().expect("the draft");
            let (best, total) = (got[0].to_bits(), got[2] as f64);
            if pass == 0 {
                // the layer's entries are its own up to the trunk's last position, the draft taken or not
                g.valid = n;
            }
            if 1.0 / total < DRAFT_MIN_P {
                break;
            }
            drafts.push(best);
        }
        Some(drafts)
    }
}

/// The prediction layer's vectors for a pass's rows.
struct MtpRows {
    x: DeviceVec,
    xn: DeviceVec,
    qfull: DeviceVec,
    q: DeviceVec,
    gate: DeviceVec,
    k: DeviceVec,
    vv: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    att: DeviceVec,
    gated: DeviceVec,
    proj: DeviceVec,
    ffa: DeviceVec,
    ffb: DeviceVec,
    act: DeviceVec,
    table: DeviceVec,
}

impl MtpRows {
    /// A pass's `rows` of the kept vectors.
    fn of(wk: &MtpWork, s: &Dims, rows: usize) -> MtpRows {
        let qh = s.n_h * s.hd;
        MtpRows {
            x: first(&wk.x, rows * s.d),
            xn: first(&wk.xn, rows * s.d),
            qfull: first(&wk.qfull, rows * 2 * qh),
            q: first(&wk.q, rows * qh),
            gate: first(&wk.gate, rows * qh),
            k: first(&wk.k, rows * s.n_kv * s.hd),
            vv: first(&wk.v, rows * s.n_kv * s.hd),
            qn: first(&wk.qn, rows * qh),
            kn: first(&wk.kn, rows * s.n_kv * s.hd),
            att: first(&wk.att, rows * qh),
            gated: first(&wk.gated, rows * qh),
            proj: first(&wk.proj, rows * s.d),
            ffa: first(&wk.ffa, rows * 2 * s.ff),
            ffb: first(&wk.ffb, rows * s.ff),
            act: first(&wk.act, rows * s.ff),
            table: first(&wk.table, rows * s.rot),
        }
    }
}

/// Room in the prediction layer's cache for `needed` rows (its rows kept).
fn mtp_reserve(chain: &dyn DeviceChain, g: &mut MtpKv, s: &Dims, needed: usize) {
    if g.cap >= needed {
        return;
    }
    let row = 2 * s.n_kv * s.hd;
    let cap = needed.next_power_of_two().max(256);
    g.layer = if g.cap == 0 { chain.vec(cap * row) } else { chain.resize(&g.layer, cap * row) };
    g.out = chain.vec(chain.attention_out_len(s.n_h, s.hd, cap));
    g.cap = cap;
}

/// The prediction layer's block on `rows` rows of `w.x` at positions `at..at + rows` (its RoPE table in `w.table`):
/// its attention over its cache (each row over the positions before it from `lo`, a row at a time, where `one` gives
/// a query's vector and `lo`; else a prompt's from the cache's start), then its FFN, `w.x` the block's output.
#[allow(clippy::too_many_arguments)]
fn mtp_block(m: &Qwen35Model, mtp: &crate::qwen35::Qwen35Mtp, sp: &Spec, s: &Dims, rec: &mut dyn ggml_rs::ChainRecorder, g: &MtpKv, w: &MtpRows, rows: usize, at: usize, one: Option<(&DeviceVec, usize)>) {
    let Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, .. } = &mtp.block else { unreachable!("the prediction layer attends") };
    let eps = m.config.rms_eps;
    let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
    let scale = 1.0 / (s.hd as f32).sqrt();
    let t = rows;
    rec.rmsnorm_rows(&w.x, &sp.attn_norm, &w.xn, t, eps);
    rec.matmul_rows(quant(attn_q), &w.xn, &w.qfull, t);
    rec.matmul_rows(quant(attn_k), &w.xn, &w.k, t);
    rec.matmul_rows(quant(attn_v), &w.xn, &w.vv, t);
    rec.copy_cols(&w.qfull, &w.q, t * s.n_h, s.hd, 2 * s.hd, 0);
    rec.copy_cols(&w.qfull, &w.gate, t * s.n_h, s.hd, 2 * s.hd, s.hd);
    rec.rmsnorm_rows(&w.q, &sp.q_norm, &w.qn, t * s.n_h, eps);
    rec.rmsnorm_rows(&w.k, &sp.k_norm, &w.kn, t * s.n_kv, eps);
    rec.rope_partial_rows(&w.qn, t, s.n_h, s.hd, s.rot, &w.table);
    rec.rope_partial_rows(&w.kn, t, s.n_kv, s.hd, s.rot, &w.table);
    rec.store_rows(&w.kn, &g.layer, t, kvd, at, row, 0);
    rec.store_rows(&w.vv, &g.layer, t, kvd, at, row, kvd);
    let qh = s.n_h * s.hd;
    match one {
        Some((q1, lo)) => {
            for r in 0..t {
                rec.copy(&w.qn, r * qh, q1, 0, qh);
                rec.attention(q1, &g.layer, &g.out, s.n_h, s.n_kv, s.hd, lo.min(at + r), at + r + 1, g.cap, scale);
                rec.copy(&g.out, 0, &w.att, r * qh, qh);
            }
        }
        None => rec.attention_rows(&w.qn, &g.layer, &w.att, t, s.n_h, s.n_kv, s.hd, at, None, scale),
    }
    rec.mul_sigmoid(&w.att, &w.gate, &w.gated, t * qh);
    rec.matmul_rows(quant(attn_output), &w.gated, &w.proj, t);
    rec.add_rmsnorm_rows(&w.x, &w.proj, &sp.post_norm, &w.xn, t, eps);
    match ffn_pair {
        FfnPair::Fused(gu) => {
            rec.matmul_rows(quant(gu), &w.xn, &w.ffa, t);
            rec.silu_mul_split_rows(&w.ffa, &w.act, t);
        }
        FfnPair::Split { gate, up } => {
            rec.matmul_rows(quant(gate), &w.xn, &w.ffa, t);
            rec.matmul_rows(quant(up), &w.xn, &w.ffb, t);
            rec.silu_mul(&w.ffa, &w.ffb, &w.act, t * s.ff);
        }
    }
    rec.matmul_rows(quant(ffn_down), &w.act, &w.proj, t);
    rec.add(&w.x, &w.proj);
}
