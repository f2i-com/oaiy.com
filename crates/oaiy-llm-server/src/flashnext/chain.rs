//! Flash-Next chained on the GPUs: a decode step, a prompt's chunk, drafts and their checks, each a recording.

use super::*;

impl FlashNext {
    /// The chain made and its kernels compiled before a first request would wait on them (its matrices packed and
    /// uploaded, some 30 pipelines built): a short prompt and a step on a cache of their own; drafting, a draft and
    /// checks of 2 to 4 rows, each undone. False where nothing is chained.
    pub fn warm_up(&self) -> bool {
        if self.chain_state().is_none() {
            return false;
        }
        let Ok(tokens) = self.tokenizer.encode("The river town kept its market on the north bank.", false) else { return false };
        if tokens.len() < 4 {
            return false;
        }
        let mut kv = self.new_kv_cache(tokens.len() + 8);
        let step = |tokens: &[u32], kv: &mut KvCache| self.embed_text(tokens).and_then(|e| self.forward(tokens, &e, kv, None)).is_ok();
        let mut warmed = step(&tokens, &mut kv) && step(&tokens[..1], &mut kv);
        if warmed && self.drafts() {
            // every draft made, however unlikely: the layer's passes all run
            warmed = self.draft_above(&kv, &tokens[1..2], 3, 0.0).is_some();
            for rows in 2..=4 {
                warmed &= self.check(&tokens[..rows], &mut kv).is_some();
                self.rollback(&mut kv, 1);
            }
        }
        self.chain.get().and_then(|c| c.as_ref()).inspect(|c| c.runs.store(0, std::sync::atomic::Ordering::Relaxed));
        warmed
    }

    /// Steps the chain has taken (a test's check that it ran).
    #[cfg(test)]
    pub(crate) fn chain_runs(&self) -> usize {
        self.chain.get().and_then(|c| c.as_ref()).map_or(0, |c| c.runs.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// `tokens` (a decode step's one, or a prompt's chunk; their embeddings `embeds`), chained, if the devices can: the
    /// last row's logits. See [`Self::run_chained`].
    pub(super) fn forward_chained(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache) -> Option<Tensor> {
        let logits = self.run_chained(tokens, embeds, kv, false)?;
        Some(Tensor::from_vec(logits, vec![1, self.config.vocab]))
    }

    /// A check of `tokens` (the token sampled, then its drafts; 2 to [`CHECK_ROWS`] of them) after what `kv` holds:
    /// every row's logits (`[1, vocab]` each), each row as a step would give it. The run is kept undoable
    /// ([`Self::rollback`]). None where it cannot be chained.
    pub fn check(&self, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<Tensor>> {
        if !(2..=CHECK_ROWS).contains(&tokens.len()) || kv.len + tokens.len() > kv.max_len {
            return None;
        }
        let embeds = self.embed_text(tokens).ok()?;
        let logits = self.run_chained(tokens, &embeds, kv, true)?;
        Some(logits.chunks_exact(self.config.vocab).map(|r| Tensor::from_vec(r.to_vec(), vec![1, self.config.vocab])).collect())
    }

    /// [`Self::check`] for a request that samples greedily: each row's token alone, its largest logit's (the first of
    /// equals, as greedy sampling takes it), picked on the GPU. A check's rows of logits are a megabyte each to read
    /// back, for a token each. None as [`Self::check`] (nothing run).
    pub fn check_picks(&self, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<u32>> {
        if !(2..=CHECK_ROWS).contains(&tokens.len()) || kv.len + tokens.len() > kv.max_len {
            return None;
        }
        let embeds = self.embed_text(tokens).ok()?;
        self.pick.store(true, std::sync::atomic::Ordering::Relaxed);
        let got = self.run_chained(tokens, &embeds, kv, true);
        self.pick.store(false, std::sync::atomic::Ordering::Relaxed);
        Some(got?.chunks_exact(4).take(tokens.len()).map(|row| row[0].to_bits()).collect())
    }

    /// A decode step's token for a request that samples greedily, as [`Self::check_picks`]: `token` (its embedding
    /// `embeds`) run after what `kv` holds, the next token picked on the GPU. None where the step cannot be chained
    /// (nothing run: [`Self::forward`] is the caller's).
    pub fn step_pick(&self, token: u32, embeds: &Tensor, kv: &mut KvCache) -> Option<u32> {
        if profile::on() || self.prompt_rows() == 0 {
            return None;
        }
        self.pick.store(true, std::sync::atomic::Ordering::Relaxed);
        let got = self.run_chained(&[token], embeds, kv, false);
        self.pick.store(false, std::sync::atomic::Ordering::Relaxed);
        got.map(|row| row[0].to_bits())
    }

    /// Undo the last check's rows past its first `keep` (the token sampled and the drafts accepted): each delta net's
    /// state and conv window and the n-gram layer's window as they were before it, its first `keep` rows run through
    /// them again, the n-gram history its tokens', and the cache cut back.
    pub fn rollback(&self, kv: &mut KvCache, keep: usize) {
        use ggml_rs::DeltaNet;
        let Some(st) = self.chain_state() else { return };
        let Some(chains) = self.devices.iter().map(|b| b.chain()).collect::<Option<Vec<_>>>() else { return };
        let cfg = &self.config;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        let Some(u) = m.undo.as_mut() else { return };
        let rows = u.tokens.len();
        assert!(keep >= 1 && keep <= rows && rows <= kv.len, "a rollback of {rows} rows to {keep}");
        if keep == rows {
            return;
        }
        let few = st.few[rows - 2].get().expect("the check's vectors");
        let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
        let dn = DeltaNet { rows: keep, v_heads: cfg.nv, k_heads: cfg.nk, k_dim: cfg.kd, v_dim: cfg.vd, scale_q: 1.0 / (cfg.vd as f32).sqrt(), eps: cfg.eps, sigmoid_gate: true };
        let ple_device = self.layers[cfg.ple_layer].device;
        let slot = ple_slot(cfg);
        for (d, c) in chains.iter().enumerate() {
            let dv = &few.devs[d];
            let mut rec = c.begin();
            rec.keep_groups(true);
            for (i, &l) in st.gdn.iter().enumerate().filter(|&(_, &l)| self.layers[l].device == d) {
                let (Some(state), Some(conv)) = (kv.ssm_state[l].as_ref().and_then(|t| c.aliased(t)), kv.ssm_conv[l].as_ref().and_then(|t| c.aliased(t))) else {
                    unreachable!("a check left layer {l}'s state the chain's")
                };
                let ChainMixer::Gdn { conv: w, a, dt, norm, .. } = &st.layers[l].mixer else { unreachable!("layer {l} is a delta net") };
                let (bs, bc) = &u.backups[i];
                let (qkv, ba) = &u.inputs[i];
                rec.copy(bs, 0, &state, 0, state.len);
                rec.copy(bc, 0, &conv, 0, conv.len);
                rec.ssm_conv(qkv, w, &conv, &dv.conv, keep, conv_dim, cfg.conv);
                // the outputs are not wanted: the check's were the kept rows' already
                rec.delta_net(&dv.conv, &dv.z, ba, a, dt, norm, &state, &dv.core, dn);
            }
            if d == ple_device {
                if let (Some(backup), Some((p, _)), Some(v), Some(window)) = (&u.window, &st.ple, &few.ple, kv.ssm_conv[slot].as_ref().and_then(|t| c.aliased(t))) {
                    // the window as the kept rows leave it (the conv's sums into scratch)
                    rec.copy(backup, 0, &window, 0, window.len);
                    rec.ple_conv(&dv.logits, &dv.normed, &v.conv_in, &window, &p.conv, keep, cfg.streams * cfg.hidden, cfg.ple_kernel, cfg.ngram);
                }
            }
            // (nothing of it is read back, and each device's next run goes to its queue behind it: not waited for,
            // where the thread parked for the first card's undoing and then the second's)
            rec.send();
        }
        // the n-gram history: the one before the check, then its kept tokens
        let ctx = cfg.ngram - 1;
        let mut history: Vec<f32> = match &u.history {
            Some(t) => t.to_host().data().to_vec(),
            None => vec![cfg.ple_eos as f32; ctx],
        };
        history.extend(u.tokens[..keep].iter().map(|&t| t as f32));
        kv.ssm_state[slot] = Some(Tensor::from_vec(history[history.len() - ctx..].to_vec(), vec![ctx]));
        kv.len -= rows - keep;
        u.tokens.truncate(keep);
        if m.mtp_hid.0 + m.mtp_hid.1 == kv.len + rows - keep {
            m.mtp_hid.1 = m.mtp_hid.1.min(keep);
        }
    }

    /// `tokens` (a decode step's one, a check's few, or a prompt's chunk; their embeddings `embeds`), chained, if the
    /// devices can: one submit a layer, the layer's work all on its GPU (the previous layer's experts as the host routed
    /// them, the hyper-connections' write-back, norm, gates and mix, the delta net or the attention, the router), only
    /// the router's logits coming back (and an attention layer's K, V and indexer key for the host's cache); the n-gram
    /// features before their layer and the hand-over between devices through the host. A step's and a few rows' (up to
    /// [`CHECK_ROWS`]) experts are routed on their GPU, a device's layers one submit, their vectors kept. The last row's
    /// logits; every row's for a check (`check`: undoable, see [`Self::rollback`]). None leaves the run to `forward`'s
    /// own path: past the dense span (QSA's sparse attention), or with images.
    pub(super) fn run_chained(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool) -> Option<Vec<f32>> {
        let run = self.run_begin(tokens, embeds, kv, check, &mut None, None, &[])?;
        Some(run.finish(self, kv))
    }

    /// A prompt's chunks in turn, chained, each chunk's first devices' layers run as the last device runs the chunk
    /// before's (`kv` then holds them all): the last chunk's logits, or None where a chunk cannot be chained (the
    /// chunks before it run, `done` of each said; the rest the caller's). `done(i)` once chunk `i` has gone to the
    /// GPUs (its K and V in the host's cache once the next has).
    pub fn forward_chunks(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize)) -> Option<Tensor> {
        self.forward_chunks_tapped(chunks, kv, done, &[]).0
    }

    /// Whether a prompt's chunks of [`Self::prompt_rows`] (`rows` in all, after what `kv` holds) can keep their
    /// recurrent states from inside them once each of `taps` rows is in ([`Self::forward_chunks_tapped`]): chained,
    /// the n-gram layer too (its window a device's vector), and each chunk's rows after the first state it keeps few
    /// ([`TAP_ROWS`]: their recurrences go through vectors of their own). OAIY_NO_TAPS: never.
    pub fn can_tap(&self, rows: usize, kv: &KvCache, taps: &[usize]) -> bool {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || std::env::var_os("OAIY_NO_TAPS").is_some() || profile::on() || rows == 0 || taps.is_empty() {
            return false;
        }
        let most = self.prompt_rows();
        self.chain_state().is_some_and(|st| st.ple.is_some())
            && self.devices.iter().all(|b| b.chain().is_some())
            && (kv.len + rows) / self.config.index_ratio <= 4096
            && taps.windows(2).all(|w| w[0] < w[1])
            && taps[0] > 0
            && taps[taps.len() - 1] <= rows
            && (0..rows).step_by(most).all(|at| {
                let end = (at + most).min(rows);
                taps.iter().find(|&&p| p > at && p <= end).map_or(true, |&p| end - p <= TAP_ROWS)
            })
    }

    /// [`Self::forward_chunks`] with the delta nets' states and conv windows and the n-gram layer's window and history
    /// as they are once each of `taps` rows of the chunks is in (ascending, counted through the chunks): what a
    /// checkpoint there holds, the run not stopping for it. A prompt's last two, before the assistant's header and
    /// before its last token, each ended a run, and a run of few rows costs what a chunk of 64 does (its weights are
    /// decoded once whatever its rows): a follow-up turn's 21 new tokens 102 ms, then 42 and 18 for those two. The
    /// states of the chunks that ran (all of them where the logits are given).
    pub fn forward_chunks_tapped(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize]) -> (Option<Tensor>, Vec<llama_rs::Tapped>) {
        let mut tapped: Vec<llama_rs::Tapped> = Vec::new();
        // each chunk's first row among the chunks'
        let starts: Vec<usize> = chunks.iter().scan(0, |at, (t, _)| { let a = *at; *at += t.len(); Some(a) }).collect();
        let logits = self.chunks_run(chunks, kv, done, taps, &starts, &mut tapped);
        (logits, tapped)
    }

    pub(super) fn chunks_run(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize], starts: &[usize], tapped: &mut Vec<llama_rs::Tapped>) -> Option<Tensor> {
        // each chunk's n-gram features (random reads of a 32 GB table: some 25 ms a chunk of 512) read on a thread of
        // their own, a chunk ahead of the GPUs
        let ctx = self.config.ngram - 1;
        let mut history = self.ple_history(kv);
        let histories: Vec<Vec<i64>> = chunks
            .iter()
            .map(|(t, _)| {
                history.extend(t.iter().map(|&v| v as i64));
                let h = history.clone();
                history.drain(..history.len() - ctx);
                h
            })
            .collect();
        std::thread::scope(|sc| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Option<Vec<f32>>>(1);
            let histories = &histories;
            sc.spawn(move || {
                for h in histories {
                    if tx.send(self.ngram_embedding(h).ok()).is_err() {
                        break;
                    }
                }
            });
            let mut pending: Option<ChainedRun<'_>> = None;
            let mut last = None;
            // Over two devices each chunk in two parts, its first device's as that device still runs the chunk
            // before's (OAIY_FN_IN_TURN: a chunk whole at a time): where the n-gram layer is chained, the experts
            // routed on their GPU, the layers one device's then the other's, and each device has room for another
            // chunk's vectors.
            let ahead = std::env::var_os("OAIY_FN_IN_TURN").is_none()
                && self.devices.len() == 2
                && self.chain_state().is_some_and(|st| st.ple.is_some() && st.layers.iter().all(|l| !l.host))
                && self.config.experts <= 1024
                && self.config.top_k <= 32
                && std::env::var_os("OAIY_HOST_ROUTE").is_none()
                && self.layers.windows(2).filter(|w| w[0].device != w[1].device).count() == 1
                && self.devices.iter().all(|b| b.chain().is_some_and(|c| c.has_room(1 << 30)));
            let mut parked: Option<(usize, Box<Parked<'_>>)> = None;
            // Each device then has a chunk's work behind the one it runs, with no gap: its pieces two at a time on its
            // queue, each encoded at its turn, until the chunks are in. Without that a card under a power limit ran a
            // tenth as fast for seconds at a time (15,037 tokens in 6.6 to 13.3 s where 5.3; the chunks whole, with
            // their gaps, 5.9 to 7.7).
            struct Fed<'a>(Vec<&'a dyn ggml_rs::DeviceChain>);
            impl Drop for Fed<'_> {
                fn drop(&mut self) {
                    for c in &self.0 {
                        c.pieces_in_flight_at_most(0);
                    }
                }
            }
            let fed = Fed(if ahead { self.devices.iter().filter_map(|b| b.chain()).collect() } else { Vec::new() });
            for c in &fed.0 {
                c.pieces_in_flight_at_most(2);
            }
            let said = std::env::var_os("OAIY_FN_LOG").is_some();
            let began = std::time::Instant::now();
            for (i, (tokens, embeds)) in chunks.iter().enumerate() {
                let t0 = began.elapsed().as_secs_f64() * 1e3;
                let ple = rx.recv().ok().flatten();
                let t1 = began.elapsed().as_secs_f64() * 1e3;
                let had = pending.is_some();
                let fits = tokens.len() <= self.prompt_rows() && !profile::on() && ple.is_some();
                // (the states this chunk keeps: by its own rows)
                let local: Vec<usize> = taps.iter().filter(|&&p| p > starts[i] && p <= starts[i] + tokens.len()).map(|&p| p - starts[i]).collect();
                // (the chunk before's rest goes after this chunk's first part, or before a chunk that goes whole)
                if ahead && fits && tokens.len() > CHECK_ROWS {
                    match self.run_part(tokens, embeds, kv, false, &mut pending, ple, Stage::First, &local) {
                        Some(Went::Parked(p)) => {
                            let mut before = parked.replace((i, p));
                            self.run_rest(&mut before, &mut pending, embeds, kv, done, tapped);
                            if said {
                                eprintln!("  fn chunk {i}: at {t0:.0} ms, features waited {:.0}, its first part and the chunk before's rest in {:.0}", t1 - t0, began.elapsed().as_secs_f64() * 1e3 - t1);
                            }
                            continue;
                        }
                        Some(Went::Run(mut run)) => {
                            // (no second device's part after all: as a chunk whole)
                            self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                            tapped.append(&mut run.tapped);
                            if let Some(p) = pending.replace(run) {
                                p.finish(self, kv);
                            }
                            done(i);
                            continue;
                        }
                        None => {
                            self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                            if let Some(p) = pending.take() {
                                p.finish(self, kv);
                            }
                            return None;
                        }
                    }
                }
                self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                let run = if fits { self.run_begin(tokens, embeds, kv, false, &mut pending, ple, &local) } else { None };
                if said {
                    eprintln!("  fn chunk {i}: at {t0:.0} ms, features waited {:.0}, begun in {:.0} (the one before {})", t1 - t0, began.elapsed().as_secs_f64() * 1e3 - t1, if had && pending.is_none() { "finished inside" } else { "left" });
                }
                let Some(mut run) = run else {
                    if let Some(p) = pending.take() {
                        p.finish(self, kv);
                    }
                    return None;
                };
                tapped.append(&mut run.tapped);
                // (one the run did not take: a chain on one device)
                if let Some(p) = pending.replace(run) {
                    p.finish(self, kv);
                }
                done(i);
            }
            if let Some(&(_, embeds)) = chunks.last() {
                self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
            }
            if let Some(p) = pending.take() {
                last = Some(p.finish(self, kv));
            }
            last.map(|l| Tensor::from_vec(l, vec![1, self.config.vocab]))
        })
    }

    /// A parked chunk's rest ([`Stage::Rest`]; `embeds` any chunk's, not read): its second device's layers recorded
    /// and gone behind the run before's, which is then finished, this chunk's run `pending` in its place.
    pub(super) fn run_rest<'a>(&'a self, parked: &mut Option<(usize, Box<Parked<'a>>)>, pending: &mut Option<ChainedRun<'a>>, embeds: &Tensor, kv: &mut KvCache, done: &mut dyn FnMut(usize), tapped: &mut Vec<llama_rs::Tapped>) {
        if let Some((j, q)) = parked.take() {
            let Some(Went::Run(mut run)) = self.run_part(&[], embeds, kv, false, pending, None, Stage::Rest(q), &[]) else { panic!("a chunk's second device's part") };
            tapped.append(&mut run.tapped);
            if let Some(p) = pending.replace(run) {
                p.finish(self, kv);
            }
            done(j);
        }
    }

    /// An attention layer's rows of a chained run (`t` of them from `at`, K then V each, and its indexer keys) into
    /// the host's cache.
    pub(super) fn cache_rows(&self, kv: &mut KvCache, i: usize, at: usize, t: usize, kvrows: Vec<f32>, raw: Vec<f32>) {
        let Mixer::Attn(a) = &self.layers[i].mixer else { unreachable!("layer {i} attends") };
        let b = self.devices[self.layers[i].device].as_ref();
        let id = self.config.index_dim;
        let len = kv.len;
        kv.len = at;
        kv.append(b, a.index_slot, &Tensor::from_vec(raw.clone(), vec![t, 1, id]), &Tensor::from_vec(raw, vec![t, 1, id]));
        kv.append_rows(b, i, &kvrows, t);
        kv.len = len;
    }

    /// [`Self::run_chained`] up to its last device's wait: every device's work gone (the last's running), `kv`
    /// committed; `prev` (a chunk's run before this one's, its last device still running) finished as the next
    /// device's layers are recorded, so the device holds one chunk's scratch at a time.
    pub(super) fn run_begin<'a>(&'a self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool, prev: &mut Option<ChainedRun<'a>>, ple: Option<Vec<f32>>, taps: &[usize]) -> Option<ChainedRun<'a>> {
        match self.run_part(tokens, embeds, kv, check, prev, ple, Stage::Whole, taps)? {
            Went::Run(run) => Some(run),
            Went::Parked(_) => unreachable!("a whole run stops at no device"),
        }
    }
}
