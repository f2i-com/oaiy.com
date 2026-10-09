//! A prompt's chunks run together down the chain, and the states kept from inside a run.

use super::*;

/// The most rows of a chunk after the first state it keeps from inside ([`FlashNext::forward_chunks_tapped`]): those
/// rows' recurrences go through vectors of their own this long (a prompt's tail: the assistant's header and its last
/// token).
pub(crate) const TAP_ROWS: usize = 64;

/// A state a run keeps from inside it: each delta net's state and conv window (a pair a slot, on the layer's device)
/// and the n-gram layer's window as they are once the run's first `row` rows are through them, copied there as the
/// run goes.
pub(super) struct Tap {
    pub(super) row: usize,
    pub(super) gdn: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    pub(super) window: Option<ggml_rs::DeviceVec>,
}

/// The vectors a tapped run's later rows' recurrences go through on a device ([`TAP_ROWS`] of them): a delta net's
/// conv input and output, gate, beta-alpha and output; the n-gram layer's streams, gate and conv input.
pub(super) struct TapSeg {
    pub(super) qkv: ggml_rs::DeviceVec,
    pub(super) conv: ggml_rs::DeviceVec,
    pub(super) z: ggml_rs::DeviceVec,
    pub(super) ba: ggml_rs::DeviceVec,
    pub(super) core: ggml_rs::DeviceVec,
    pub(super) x: ggml_rs::DeviceVec,
    pub(super) gated: ggml_rs::DeviceVec,
    pub(super) conv_in: ggml_rs::DeviceVec,
}

/// `v`'s first `len` elements as a vector of their own (the same buffer).
pub(super) fn first_of(v: &ggml_rs::DeviceVec, len: usize) -> ggml_rs::DeviceVec {
    assert!(len <= v.len, "{len} of a vector of {}", v.len);
    ggml_rs::DeviceVec { len, inner: Arc::clone(&v.inner) }
}

/// How much of a chained run [`FlashNext::run_part`] makes: all of it, a prompt chunk's first device's part, or the
/// rest of a chunk parked after that.
pub(super) enum Stage<'a> {
    Whole,
    First,
    Rest(Box<Parked<'a>>),
}

/// What [`FlashNext::run_part`] left: a run whose last device is running, or a chunk parked after its first device's
/// part.
pub(super) enum Went<'a> {
    Run(ChainedRun<'a>),
    Parked(Box<Parked<'a>>),
}

/// A prompt's chunk whose first device's layers are recorded and gone: its positions, its embeddings (the prediction
/// layer's), its own vectors on every device, its first device's recording (its streams and rows read at the
/// handoff), and the layer its next device's begin at.
pub(super) struct Parked<'a> {
    pub(super) past: usize,
    pub(super) t: usize,
    pub(super) e: Tensor,
    pub(super) devs: Arc<Vec<ChainDev>>,
    pub(super) ple: Option<Arc<PleVecs>>,
    pub(super) attn_rows: Vec<Option<ggml_rs::DeviceVec>>,
    pub(super) qsa: Arc<Vec<Option<QsaVecs>>>,
    pub(super) keys: Vec<Option<ggml_rs::DeviceVec>>,
    pub(super) handoffs: Vec<(Box<dyn ggml_rs::ChainRecorder + 'a>, usize, Vec<usize>)>,
    pub(super) at: usize,
    /// The states the chunk keeps from inside it (their vectors, and what the caller is given of them).
    pub(super) kept: Vec<Tap>,
    pub(super) tapped: Vec<llama_rs::Tapped>,
}

/// A chained run whose last device is running: its recording (the run's reads its attention layers' rows, then
/// the logits) and where its rows go in the host's cache.
pub(super) struct ChainedRun<'a> {
    pub(super) rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    pub(super) attn_reads: Vec<usize>,
    pub(super) at: usize,
    pub(super) t: usize,
    /// The states it keeps from inside (theirs once the run has run)
    pub(super) tapped: Vec<llama_rs::Tapped>,
}

impl ChainedRun<'_> {
    /// Its last device waited for: its attention layers' rows into the host's cache, the logits.
    pub(super) fn finish(self, fnx: &FlashNext, kv: &mut KvCache) -> Vec<f32> {
        let mut got = self.rec.finish().into_iter();
        for &a in &self.attn_reads {
            let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
            fnx.cache_rows(kv, a, self.at, self.t, kvrows, raw);
        }
        kv.dirty_from = usize::MAX;
        got.next().expect("the logits")
    }
}

impl FlashNext {
    /// The devices its layers are over.
    pub fn devices_len(&self) -> usize {
        self.devices.len()
    }

    /// The rows a prompt's chunk has at most: 512, or 1,024 on one card that holds a part of its experts
    /// (OAIY_FN_ROWS: as given, 64 to 1,024). A chunk's experts' weights are decoded once a block of their rows, and
    /// a chunk of 512 gives each of the 512 experts some 10 rows of a block's 32, so a chunk of 1,024 takes less of
    /// the GPUs a token (its kernels 298 ms where two of 512 take 342). It is not the faster for that over two cards
    /// (2,148 tokens run together 679 to 691 ms in chunks of 1,024 where 602 to 612 in 512s: fewer chunks one behind
    /// the other), and with 24 GB of weights a card its vectors do not fit two RTX 5090s' 32 GB (out of memory at
    /// the server's second prompt); past 1,638 rows a kernel's grid is wider than a dispatch may be.
    ///
    /// One card that holds a part of its experts copies those a chunk uses and it lacks from the host's memory, once
    /// a chunk, so half as many chunks copy fewer (Strata's 4,086 tokens: 16,000 to 22,000 experts where 20,000 to
    /// 26,000, the prompt in 2.5 to 2.8 s in most launches and 3.0 to 3.5 in the others, where chunks of 512 are 3.0
    /// to 3.4 s). Their vectors and scratch are 0.7 GiB more, which the card's share of its experts leaves room for
    /// (`flashnext_gguf_on`).
    pub fn prompt_rows(&self) -> usize {
        static ASKED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let asked = *ASKED.get_or_init(|| std::env::var("OAIY_FN_ROWS").ok().and_then(|v| v.parse().ok()).filter(|n| (64..=1024).contains(n)));
        asked.unwrap_or(if self.devices.len() == 1 && self.layers.iter().any(|l| l.moe.experts.part_held()) { 1024 } else { 512 })
    }

    /// The layers whose experts run on the host (no GPU had room for them).
    pub fn host_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.moe.experts.on_host()).count()
    }

    /// Whether the chain drafts tokens (its multi-token-prediction layer loaded and chained).
    pub fn drafts(&self) -> bool {
        std::env::var_os("OAIY_NO_CHAIN").is_none() && self.chain_state().is_some_and(|c| c.mtp.is_some())
    }

    /// QSA's vectors on `c` for runs of up to `rows` rows over a cache of `cap` positions.
    pub(super) fn qsa_vecs(&self, c: &dyn ggml_rs::DeviceChain, rows: usize, cap: usize) -> QsaVecs {
        let cfg = &self.config;
        let (ratio, id, keep) = (cfg.index_ratio, cfg.index_dim, cfg.index_budget / cfg.index_ratio);
        let blocks = cap / ratio;
        let v = |n: usize| c.vec(n.max(1));
        let table = v(blocks * cfg.rope_dim);
        c.upload(&table, &self.rope_table_of((0..blocks).map(|j| j * ratio)));
        QsaVecs {
            rows,
            iq: v(rows * cfg.index_heads * id),
            iqn: v(rows * cfg.index_heads * id),
            pooled: v(blocks * id),
            pooledn: v(blocks * id),
            table,
            scores: v(rows * blocks),
            list: v(rows * keep),
            out: v(c.qsa_attention_out_len(rows, cfg.heads, cfg.head_dim, keep, ratio)),
        }
    }

    /// The prediction layer's cache with room for `len` positions (grown as the trunk's copy is).
    pub(super) fn mtp_reserve<'a>(c: &dyn ggml_rs::DeviceChain, g: &'a mut Option<MtpKv>, cfg: &Config, len: usize) -> &'a mut MtpKv {
        let row = 2 * cfg.kv_heads * cfg.head_dim;
        let g = g.get_or_insert_with(|| MtpKv { layer: c.vec(1), cap: 0, out: c.vec(1), owner: 0, start: 0, valid: 0 });
        if g.cap < len {
            let cap = len.next_power_of_two().max(256);
            g.layer = if g.cap == 0 { c.vec(cap * row) } else { c.resize(&g.layer, cap * row) };
            g.out = c.vec(c.attention_out_len(cfg.heads, cfg.head_dim, cap));
            g.cap = cap;
        }
        g
    }

    /// The partial RoPE's sines and cosines at positions `at..at + rows`.
    pub(super) fn rope_table(&self, at: usize, rows: usize) -> Vec<f32> {
        self.rope_table_of(at..at + rows)
    }

    /// The partial RoPE's sines and cosines at `positions`.
    pub(super) fn rope_table_of(&self, positions: impl Iterator<Item = usize>) -> Vec<f32> {
        let (cfg, rot) = (&self.config, self.config.rope_dim);
        positions
            .flat_map(|pos| {
                (0..rot / 2).flat_map(move |k| {
                    let (sn, cs) = (pos as f32 * cfg.rope_theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                    [sn, cs]
                })
            })
            .collect()
    }

    /// The prediction layer's cache at a prompt's chunk (`t` rows at `past`, its embeddings `e` and its streams after
    /// the trunk's last layer `x`): its K and V for the chunk's positions but the last, each from the row's streams and
    /// the next row's token, where its entries run unbroken from the cache's start up to the chunk (else none: the
    /// next draft's start them over).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp_prompt(&self, mp: &FnMtp, mc: &MtpChain, m: &mut ChainMut, c: &dyn ggml_rs::DeviceChain, rec: &mut dyn ggml_rs::ChainRecorder, e: &Tensor, x: &ggml_rs::DeviceVec, t: usize, past: usize, owner: u64) {
        let cfg = &self.config;
        let (h, s, eps) = (cfg.hidden, cfg.streams, cfg.eps);
        let (nkv, hd) = (cfg.kv_heads, cfg.head_dim);
        let (kvd, row) = (nkv * hd, 2 * nkv * hd);
        let g = Self::mtp_reserve(c, &mut m.mtp_kv, cfg, past + t);
        if g.owner != owner || g.valid != past || g.start != 0 {
            if past != 0 {
                g.owner = 0;
                g.valid = 0;
                return;
            }
            g.start = 0;
        }
        g.owner = owner;
        let rows = t - 1;
        if rows == 0 {
            g.valid = past;
            return;
        }
        let (Some(fc_e), Some(fc_h), Some(kw), Some(vw)) = (chain_packed(&mp.fc_e), chain_packed(&mp.fc_h), chain_packed(&mp.k), chain_packed(&mp.v)) else { return };
        let v = |n: usize| c.vec(n);
        let (ev, en, e2, hn, xs, normed, tt, post, logits, mixed, kk, vv, kn, table) =
            (v(rows * h), v(rows * h), v(rows * h), v(rows * s * h), v(rows * s * h), v(rows * s * h), v(rows * (self.chain_state().map_or(0, |st| st.rank) + s)), v(rows * s), v(rows * s * h), v(rows * h), v(rows * kvd), v(rows * kvd), v(rows * kvd), v(rows * cfg.rope_dim));
        c.upload(&ev, &e.data()[h..t * h]);
        c.upload(&table, &self.rope_table(past, rows));
        rec.rmsnorm_rows(&ev, &mc.enorm, &en, rows, eps);
        rec.exl3_rows(fc_e, &en, &e2, rows);
        let xv = ggml_rs::DeviceVec { len: rows * s * h, inner: Arc::clone(&x.inner) };
        rec.rmsnorm_rows(&xv, &mc.hnorm, &hn, rows, eps);
        rec.exl3_rows(fc_h, &hn, &xs, rows * s);
        let ones = v(rows * s);
        c.upload(&ones, &vec![1.0; rows * s]);
        rec.stream_apply(&xs, &e2, &ones, rows, s, h);
        let hcv = &mc.attn_hc;
        rec.rmsnorm_streams(&xs, &hcv.norm, &normed, rows, s, eps);
        hcv.project(&mut *rec, &normed, &tt, &post, &logits, &mixed, rows, s, h);
        rec.exl3_rows(kw, &mixed, &kk, rows);
        rec.exl3_rows(vw, &mixed, &vv, rows);
        rec.rmsnorm_rows(&kk, &mc.k_norm, &kn, rows * nkv, eps);
        rec.rope_partial_rows(&kn, rows, nkv, hd, cfg.rope_dim, &table);
        rec.store_rows(&kn, &g.layer, rows, kvd, past, row, 0);
        rec.store_rows(&vv, &g.layer, rows, kvd, past, row, kvd);
        g.valid = past + rows;
    }

    /// Draft up to `k` tokens after `next` (the token sampled for position `kv.len`): the prediction layer's entries
    /// caught up to it first (each position from the trunk's streams there, kept from the last run, and the token after;
    /// `tokens` the tokens at positions up to `kv.len`, ending with `next`), its last row giving the first draft; each
    /// further draft from the layer's own output streams and the draft before. A draft the layer gives less than
    /// [`DRAFT_MIN_P`] ends them (none where the first is such). None where the chain does not draft or the streams it
    /// needs are not kept.
    pub fn draft(&self, kv: &KvCache, tokens: &[u32], k: usize) -> Option<Vec<u32>> {
        self.draft_above(kv, tokens, k, DRAFT_MIN_P)
    }

    /// [`Self::draft`], a draft the layer gives less than `min_p` ending them.
    pub(crate) fn draft_above(&self, kv: &KvCache, tokens: &[u32], k: usize, min_p: f64) -> Option<Vec<u32>> {
        use ggml_rs::ChainRecorder;
        if !self.drafts() || k == 0 {
            return None;
        }
        let (st, mp) = (self.chain_state()?, self.mtp.as_ref()?);
        let mc = st.mtp.as_ref()?;
        let last = self.devices.len() - 1;
        let c = self.devices[last].chain()?;
        let cfg = &self.config;
        let (h, s, eps) = (cfg.hidden, cfg.streams, cfg.eps);
        let (nh, nkv, hd, rot) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.rope_dim);
        let (kvd, row, qh) = (nkv * hd, 2 * nkv * hd, nh * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let n = kv.len;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        let (hid_at, hid_rows) = m.mtp_hid;
        // the rows the streams cover, up to the trunk's last position
        if hid_rows == 0 || hid_at + hid_rows != n || tokens.len() < hid_rows {
            return None;
        }
        let tokens = &tokens[tokens.len() - hid_rows..];
        let g = Self::mtp_reserve(c, &mut m.mtp_kv, cfg, n + k + 1);
        // entries from the streams' first row on; before it, the run of true ones if it reaches it, else none
        if g.owner != kv.id || g.valid < hid_at || g.valid > n {
            g.start = hid_at;
        }
        g.owner = kv.id;
        let dev = |rows: usize| {
            mc.devs[rows - 1].get_or_init(|| {
                let v = |n: usize| c.vec(rows * n);
                MtpDev { dv: chain_dev(c, cfg, st.rank, rows, 1), e: v(h), en: v(h), e2: v(h), hin: v(s * h), hn: v(s * h), att: v(qh), q1: c.vec(qh) }
            })
        };
        let one = dev(1);
        fn packed(w: &Weight) -> &dyn PackedLinear {
            chain_packed(w).expect("a chained layer's matrix")
        }
        let mut drafts = Vec::with_capacity(k);
        // pass 0: the caught-up rows (the streams kept); then a row a draft (the layer's own output)
        for pass in 0..k {
            let (rows, at) = if pass == 0 { (hid_rows, hid_at) } else { (1, n + pass - 1) };
            let md = dev(rows);
            let dv = &md.dv;
            let next: Vec<u32> = if pass == 0 { tokens.to_vec() } else { vec![drafts[pass - 1]] };
            c.upload(&md.e, self.embed_text(&next).ok()?.data());
            c.upload(&dv.table, &self.rope_table(at, rows));
            let mut rec = c.begin();
            rec.keep_groups(true);
            // (a GGUF's head from int8 activations, as the trunk's rows take it: half a draft's GPU time in f32, and
            // a draft is only what a check then takes or refuses)
            rec.rows_alike(true);
            if pass == 0 {
                rec.copy(&mc.hid, 0, &md.hin, 0, rows * s * h);
            } else {
                rec.copy(&one.dv.x, 0, &md.hin, 0, s * h);
            }
            // the inputs: the embedding's projection in every stream of the streams' own
            rec.rmsnorm_rows(&md.e, &mc.enorm, &md.en, rows, eps);
            rec.exl3_rows(packed(&mp.fc_e), &md.en, &md.e2, rows);
            rec.rmsnorm_rows(&md.hin, &mc.hnorm, &md.hn, rows, eps);
            rec.exl3_rows(packed(&mp.fc_h), &md.hn, &dv.x, rows * s);
            rec.stream_apply(&dv.x, &md.e2, &mc.ones, rows, s, h);
            let hc = |rec: &mut dyn ChainRecorder, hcv: &HcVecs, pending: Option<(&ggml_rs::DeviceVec, &ggml_rs::DeviceVec)>, post: &ggml_rs::DeviceVec, out: &ggml_rs::DeviceVec, dv: &ChainDev, rows: usize| {
                if let Some((y, p)) = pending {
                    rec.stream_apply(&dv.x, y, p, rows, s, h);
                }
                rec.rmsnorm_streams(&dv.x, &hcv.norm, &dv.normed, rows, s, eps);
                hcv.project(&mut *rec, &dv.normed, &dv.t, post, &dv.logits, out, rows, s, h);
            };
            // attention over its own cache, a row at a time from its entries' start
            hc(&mut *rec, &mc.attn_hc, None, &dv.post, &dv.y_in, dv, rows);
            rec.exl3_rows(packed(&mp.q), &dv.y_in, &dv.qfull, rows);
            rec.exl3_rows(packed(&mp.k), &dv.y_in, &dv.k, rows);
            rec.exl3_rows(packed(&mp.v), &dv.y_in, &dv.v, rows);
            rec.copy_cols(&dv.qfull, &dv.q, rows * nh, hd, 2 * hd, 0);
            rec.copy_cols(&dv.qfull, &dv.gate, rows * nh, hd, 2 * hd, hd);
            rec.rmsnorm_rows(&dv.q, &mc.q_norm, &dv.qn, rows * nh, eps);
            rec.rmsnorm_rows(&dv.k, &mc.k_norm, &dv.kn, rows * nkv, eps);
            rec.rope_partial_rows(&dv.qn, rows, nh, hd, rot, &dv.table);
            rec.rope_partial_rows(&dv.kn, rows, nkv, hd, rot, &dv.table);
            rec.store_rows(&dv.kn, &g.layer, rows, kvd, at, row, 0);
            rec.store_rows(&dv.v, &g.layer, rows, kvd, at, row, kvd);
            for r in 0..rows {
                rec.copy(&dv.qn, r * qh, &md.q1, 0, qh);
                rec.attention(&md.q1, &g.layer, &g.out, nh, nkv, hd, g.start.min(at + r), at + r + 1, g.cap, scale);
                rec.copy(&g.out, 0, &md.att, r * qh, qh);
            }
            rec.mul_sigmoid(&md.att, &dv.gate, &dv.gated, rows * qh);
            rec.exl3_rows(packed(&mp.o), &dv.gated, &dv.y_out, rows);
            // the experts, their write the layer's output streams
            hc(&mut *rec, &mc.mlp_hc, Some((&dv.y_out, &dv.post)), &dv.post2, &dv.y2_in, dv, rows);
            mc.router.mul(&mut *rec, cfg.experts + 1, h, &dv.y2_in, &dv.router, rows);
            assert!(rec.moe_routed_into(mp.experts.as_ref(), &dv.y2_in, &dv.x, &dv.post2, &dv.router, cfg.top_k, rows, s), "the prediction layer's experts route on their GPU");
            // the last row's streams (the next pass's input), collapsed, then the head: the next draft
            if rows > 1 {
                rec.copy(&dv.x, (rows - 1) * s * h, &one.dv.x, 0, s * h);
            }
            hc(&mut *rec, &mc.mixer, None, &one.dv.post, &one.dv.mixed, &one.dv, 1);
            rec.exl3_rows(chain_packed(&self.head)?, &one.dv.mixed, &one.dv.head, 1);
            // the draft and its probability under the layer (the largest logit's share of their exponentials)
            rec.argmax_softmax(&one.dv.head, &mc.best);
            rec.read(&mc.best);
            let got = rec.finish().pop().expect("the draft");
            let (best, total) = (got[0].to_bits(), got[2] as f64);
            if pass == 0 {
                // the layer's entries are true ones up to the trunk's last position, the draft taken or not
                g.valid = n;
            }
            if 1.0 / total < min_p {
                break;
            }
            drafts.push(best);
        }
        Some(drafts)
    }
}
