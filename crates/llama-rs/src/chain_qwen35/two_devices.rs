//! The chain over two devices: a prompt's chunks down the first's layers as the second runs the chunk before's.

use super::*;

/// A split prompt's run ([`Qwen35Chain::forward_split`]): what its chunks share.
struct SplitRun<'a> {
    chained: &'a Qwen35Chain,
    m: &'a Qwen35Model,
    st: &'a State,
    sp: &'a Split,
    chain: &'a dyn DeviceChain,
    chain1: &'a dyn DeviceChain,
    emb: &'a [f32],
    /// each chunk's first row (of the prompt's) and rows; the cache's rows before the prompt
    chunks: &'a [(usize, usize)],
    past0: usize,
    /// the first's copy of the attention cache (its buffers, and rows of room), and the second's of its layers
    kv0: (&'a [DeviceVec], usize),
    kv1: (&'a [DeviceVec], usize),
    states: &'a [(DeviceVec, DeviceVec)],
    states1: &'a [(DeviceVec, DeviceVec)],
    /// the first's attention slots of the layers it runs, and the layer of each of the second's slots
    slots0: Vec<usize>,
    layers1: &'a [usize],
    owner: u64,
    /// The rows of the run after which the recurrent states are kept (ascending), those states as its chunks record
    /// them (a device's layers' at its part of a chunk), and the cache's layers.
    taps: &'a [usize],
    kept: std::cell::RefCell<Vec<Tapped>>,
    layers_n: usize,
    /// Whether the second's recurrent states go up from the first's (else they are zero: a new conversation's)
    states_up: bool,
    /// OAIY_SPLIT_LOG: when each step began and ended (ms from the run's start), for the log
    log: Option<(std::time::Instant, std::cell::RefCell<Vec<String>>)>,
}

/// A split run's chunk gone to the first device: its recording (its reads the residual stream and the last layer's
/// output, at the first chunk the second's recurrent states, then the first's layers' K and V rows).
struct FirstRun<'a> {
    rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    i: usize,
}

/// A split run's chunk gone to the second device: its recording (its reads the last chunk's logits, the hidden states
/// where a prediction layer's cache takes them, the second's layers' K and V rows, at the last chunk its recurrent
/// states).
struct SecondRun<'a> {
    rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    i: usize,
}

impl<'a> SplitRun<'a> {
    /// The time since the run began, ms (for the log).
    fn now(&self) -> f64 {
        self.log.as_ref().map_or(0.0, |(t, _)| t.elapsed().as_secs_f64() * 1e3)
    }

    fn note(&self, what: String) {
        if let Some((_, l)) = &self.log {
            l.borrow_mut().push(what);
        }
    }

    /// The states chunk `i` keeps on one device (`c`, whose recurrent slots are `states`): for each of the run's kept
    /// rows in the chunk a pair of vectors for each slot whose layer the device runs (`layer_of`: that layer, in the
    /// cache's order), the run's kept states given them (theirs once that part of the chunk has run).
    fn tap(&self, i: usize, c: &dyn DeviceChain, states: &[(DeviceVec, DeviceVec)], layer_of: &dyn Fn(usize) -> Option<usize>) -> Vec<Tap> {
        let s = self.st.dims;
        let (at, t) = self.chunks[i];
        let mut kept = self.kept.borrow_mut();
        let mut spare: Option<DeviceVec> = None;
        self.taps
            .iter()
            .filter(|&&p| p > at && p <= at + t)
            .map(|&p| {
                let abs = self.past0 + p;
                if !kept.iter().any(|k| k.at == abs) {
                    let none = || (0..self.layers_n).map(|_| None).collect::<Vec<Option<Tensor>>>();
                    kept.push(Tapped { at: abs, states: none(), convs: none() });
                }
                let entry = kept.iter_mut().find(|k| k.at == abs).expect("the kept state");
                let vecs = states
                    .iter()
                    .enumerate()
                    .map(|(slot, (state, conv))| match layer_of(slot) {
                        Some(l) => {
                            let (sv, cv) = (c.vec(state.len), c.vec(conv.len));
                            entry.states[l] = Some(c.alias(&sv, vec![s.nv, s.dv, s.dk]));
                            entry.convs[l] = Some(c.alias(&cv, vec![s.kern - 1, s.ch]));
                            (sv, cv)
                        }
                        // (a slot whose layer the other device runs: never copied to)
                        None => {
                            let v = spare.get_or_insert_with(|| c.vec(1)).clone();
                            (v.clone(), v)
                        }
                    })
                    .collect();
                Tap { row: p - at, vecs }
            })
            .collect()
    }

    /// Chunk `i`'s layers before the split's, gone to the first device after the prediction layer's work for the
    /// chunks whose hidden states are back (`due`).
    fn first(&self, i: usize, due: &mut Vec<(usize, Vec<f32>)>) -> FirstRun<'a> {
        let t0 = self.now();
        let (s, cfg) = (self.st.dims, &self.m.config);
        let (at, t) = self.chunks[i];
        let pos = self.past0 + at;
        let row = 2 * s.n_kv * s.hd;
        let c = self.chain;
        let w = Work::new(c, &s, t);
        let attn = c.vec(c.attention_rows_out_len(t, s.n_h, s.hd, pos + t));
        c.upload(&w.table, &rope_table(cfg.rope_theta, s.rot, pos, t));
        c.upload(&w.x, &self.emb[at * s.d..(at + t) * s.d]);
        let mut rec = c.begin();
        rec.keep_groups(false);
        for (j, hidden) in due.drain(..) {
            self.mtp(&mut *rec, j, &hidden);
        }
        let mut added = false;
        let kept = self.tap(i, c, self.states, &|j| Some(self.st.ssm_layers[j]).filter(|&l| l < self.sp.from));
        let seg = kept.first().filter(|tap| tap.row < t).map(|_| self.st.seg.get_or_init(|| Seg::new(c, &s)));
        let bound = Bound { kvl: self.kv0.0, half: None, one: None, cap: self.kv0.1, states: self.states, w: &w, attn: &attn, taps: &kept, seg };
        let from = self.sp.from;
        record_layers(&mut *rec, &s, cfg.rms_eps, self.m.blocks[..from].iter().map(LayerW::of).zip(&self.st.layers[..from]), &bound, t, pos, None, &mut added);
        // to the second: the residual stream and the last layer's output (their add fused with its first layer's norm)
        rec.read(&w.x);
        rec.read(&w.proj);
        if i == 0 && self.states_up {
            for &j in &self.sp.ssm_slots {
                rec.read(&self.states[j].0);
                rec.read(&self.states[j].1);
            }
        }
        for &a in &self.slots0 {
            rec.read_range(&self.kv0.0[a], pos * row, t * row);
        }
        rec.flush();
        self.note(format!("first {i}: recorded {t0:.1}..{:.1}", self.now()));
        FirstRun { rec, i }
    }

    /// Chunk `f`'s layers from the split's on and the head, recorded on the second device as the first runs its first
    /// ones, gone once those have run and their output is up (their K and V rows into the host's cache).
    fn hand_off(&self, f: FirstRun<'a>, kv: &mut KvCache) -> SecondRun<'a> {
        let t0 = self.now();
        let (s, cfg) = (self.st.dims, &self.m.config);
        let i = f.i;
        let (at, t) = self.chunks[i];
        let pos = self.past0 + at;
        let last = i + 1 == self.chunks.len();
        let row = 2 * s.n_kv * s.hd;
        let c = self.chain1;
        let w = Work::new(c, &s, t);
        let attn = c.vec(c.attention_rows_out_len(t, s.n_h, s.hd, pos + t));
        c.upload(&w.table, &rope_table(cfg.rope_theta, s.rot, pos, t));
        let mut rec = c.begin();
        rec.keep_groups(false);
        rec.hold();
        let mut added = true;
        let kept = self.tap(i, c, self.states1, &|k| Some(self.st.ssm_layers[self.sp.ssm_slots[k]]));
        let seg = kept.first().filter(|tap| tap.row < t).map(|_| self.sp.seg.get_or_init(|| Seg::new(c, &s)));
        let bound = Bound { kvl: self.kv1.0, half: None, one: None, cap: self.kv1.1, states: self.states1, w: &w, attn: &attn, taps: &kept, seg };
        record_layers(&mut *rec, &s, cfg.rms_eps, self.sp.weights.iter().map(SplitW::view).zip(&self.sp.layers), &bound, t, pos, None, &mut added);
        rec.add_rmsnorm_rows(&w.x, &w.proj, &self.sp.output_norm, &w.xn, t, cfg.rms_eps);
        if last {
            rec.copy(&w.xn, (t - 1) * s.d, &w.last, 0, s.d);
            rec.matmul(&self.sp.output, &w.last, &self.sp.logits);
            rec.read(&self.sp.logits);
        }
        if self.st.spec.is_some() {
            rec.read(&w.xn);
        }
        for a in 0..self.layers1.len() {
            rec.read_range(&self.kv1.0[a], pos * row, t * row);
        }
        if last {
            for (state, conv) in self.states1 {
                rec.read(state);
                rec.read(conv);
            }
        }
        // the first's part done: its output up, the second's going, then the first's layers' rows into the host's cache
        let t1 = self.now();
        let mut got = f.rec.finish().into_iter();
        let t2 = self.now();
        c.upload(&w.x, &got.next().expect("the residual stream"));
        c.upload(&w.proj, &got.next().expect("the last layer's output"));
        if i == 0 && self.states_up {
            for (state, conv) in self.states1 {
                c.upload(state, &got.next().expect("a recurrent state"));
                c.upload(conv, &got.next().expect("its conv window"));
            }
        }
        rec.flush();
        let t3 = self.now();
        for &a in &self.slots0 {
            cache_rows(&*self.m.backend, kv, self.st.attention_layers[a], pos, t, &got.next().expect("a layer's K and V"));
        }
        self.note(format!("hand off {i}: recorded {t0:.1}..{t1:.1}, the first's done {t2:.1}, the second's gone {t3:.1}, stored {:.1}", self.now()));
        SecondRun { rec, i }
    }

    /// Chunk `r` done on the second device: its layers' K and V rows into the host's cache and the first's copy, its
    /// hidden states due for the prediction layer's cache, at the last chunk its logits and the recurrent states back
    /// on the first.
    fn finish(&self, r: SecondRun<'a>, kv: &mut KvCache, due: &mut Vec<(usize, Vec<f32>)>, logits: &mut Option<Vec<f32>>) {
        let s = self.st.dims;
        let (at, t) = self.chunks[r.i];
        let pos = self.past0 + at;
        let last = r.i + 1 == self.chunks.len();
        let row = 2 * s.n_kv * s.hd;
        let t0 = self.now();
        let mut got = r.rec.finish().into_iter();
        let t1 = self.now();
        if last {
            *logits = got.next();
        }
        if self.st.spec.is_some() {
            due.push((r.i, got.next().expect("the hidden states")));
        }
        for (k, &l) in self.layers1.iter().enumerate() {
            let rows = got.next().expect("a layer's K and V");
            self.chain.upload_at(&self.kv0.0[self.sp.attention_slots[k]], pos * row, &rows);
            cache_rows(&*self.m.backend, kv, l, pos, t, &rows);
        }
        if last {
            for &j in &self.sp.ssm_slots {
                self.chain.upload(&self.states[j].0, &got.next().expect("a recurrent state"));
                self.chain.upload(&self.states[j].1, &got.next().expect("its conv window"));
            }
        }
        self.note(format!("finish {}: waited {t0:.1}..{t1:.1}, stored {:.1}", r.i, self.now()));
    }

    /// The prediction layer's cache at chunk `j` (its hidden states after the output norm back from the second), and
    /// the chunk's last hidden state kept: as a run on the first device alone does.
    fn mtp(&self, rec: &mut dyn ggml_rs::ChainRecorder, j: usize, hidden: &[f32]) {
        let Some(spec) = &self.st.spec else { return };
        let s = self.st.dims;
        let (at, t) = self.chunks[j];
        let pos = self.past0 + at;
        let hv = self.chain.vec(t * s.d);
        self.chain.upload(&hv, hidden);
        if t > 1 {
            self.chained.mtp_prompt(self.m, self.st, spec, self.chain, rec, &self.emb[at * s.d..(at + t) * s.d], &hv, t, pos, self.owner);
        }
        rec.copy(&hv, (t - 1) * s.d, &spec.hid, 0, s.d);
        *spec.hid_at.lock().unwrap_or_else(|p| p.into_inner()) = (pos + t - 1, 1);
    }

    /// The prediction layer's work for the chunks `due`, on the first device (which has run its last chunk), gone.
    fn mtp_only(&self, due: &mut Vec<(usize, Vec<f32>)>) -> Box<dyn ggml_rs::ChainRecorder + 'a> {
        let mut rec = self.chain.begin();
        rec.keep_groups(false);
        for (j, hidden) in due.drain(..) {
            self.mtp(&mut *rec, j, &hidden);
        }
        rec.flush();
        rec
    }
}

impl Qwen35Chain {
    /// A second device for prompts: a prompt's chunks run over both ([`Self::forward`]), the layers from the middle
    /// on and the head copied there at the first prompt that can use them (steps and checks stay on the model's own).
    pub fn split_onto(&self, backend: Arc<dyn Backend>) {
        let _ = self.second.set(backend);
    }

    /// The second device's share of a prompt's layers, made at the first prompt that can use it: None where there
    /// is no second device, no room on it for them, or prompts are not split (`split_off`, OAIY_NO_SPLIT).
    pub(super) fn split(&self, m: &Qwen35Model, st: &State) -> Option<&Split> {
        if self.split_off.load(Ordering::Relaxed) || std::env::var_os("OAIY_NO_SPLIT").is_some() {
            return None;
        }
        self.split
            .get_or_init(|| {
                let chain = self.second.get()?.chain()?;
                let s = st.dims;
                let n = m.blocks.len();
                // half the layers on each, unless asked otherwise (OAIY_SPLIT_AT: the second's first layer)
                let from = std::env::var("OAIY_SPLIT_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(n / 2);
                if from == 0 || from >= n {
                    return None;
                }
                // (a GGUF's alone: a packed model's prompts stay on the device its projections were made on)
                let Weight::Quant(head) = &m.output else { return None };
                let output = chain.copy_weight(head)?;
                let weights = m.blocks[from..].iter().map(|b| SplitW::copy(chain, LayerW::of(b))).collect::<Option<Vec<_>>>()?;
                let (mut attention_slots, mut ssm_slots) = (Vec::new(), Vec::new());
                let layers = (from..n)
                    .map(|l| {
                        let slot = match &st.layers[l].mixer {
                            Mixer::Attention { slot, .. } => {
                                attention_slots.push(*slot);
                                attention_slots.len() - 1
                            }
                            Mixer::Ssm { slot, .. } => {
                                ssm_slots.push(*slot);
                                ssm_slots.len() - 1
                            }
                        };
                        layer_vecs(chain, &m.blocks[l], slot)
                    })
                    .collect();
                let pool = Pool { states: ssm_slots.iter().map(|_| chain.vec(s.nv * s.dk * s.dv)).collect(), convs: ssm_slots.iter().map(|_| chain.vec((s.kern - 1) * s.ch)).collect() };
                Some(Split {
                    from,
                    weights,
                    layers,
                    attention_slots,
                    ssm_slots,
                    output_norm: upload_tensor(chain, &m.output_norm),
                    output,
                    logits: chain.vec(s.vocab),
                    kv: Mutex::new(SplitKv { kv: Kv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0, half: Vec::new(), halved: 0, refused: 0 }, upto: 0 }),
                    pool,
                    seg: Default::default(),
                })
            })
            .as_ref()
    }

    /// The second device's share of a prompt's layers made (their weights copied there) and its kernels compiled: a
    /// prompt of two chunks over both devices, on a cache of its own (`tokens` repeated). False where prompts do not
    /// split.
    pub(crate) fn warm_split(&self, m: &Qwen35Model, tokens: &[u32]) -> bool {
        let (Some(st), Some(chain)) = (self.state(m), m.backend.chain()) else { return false };
        let (Some(sp), Some(chain1)) = (self.split(m, st), self.second.get().and_then(|b| b.chain())) else { return false };
        let rows = 2 * 64;
        let tokens: Vec<u32> = tokens.iter().copied().cycle().take(rows).collect();
        if tokens.len() < rows {
            return false;
        }
        let e = m.embed_text(&tokens).to_host();
        let mut kv = KvCache::new(&*m.backend, m.config.n_layers, rows + 16, m.config.n_kv_heads, m.config.head_dim);
        self.forward_split(m, st, sp, chain, chain1, e.data(), &[(0, 64), (64, 64)], &mut kv, &[]);
        true
    }

    /// Rows of `kv` from `past` on written on the first device alone (a step, a check, a prompt there): the second's
    /// copy of its layers' cache holds the cache's up to there at most (and none the host wrote since).
    pub(super) fn split_written(&self, kv: &KvCache, past: usize) {
        if let Some(Some(sp)) = self.split.get() {
            let mut g1 = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
            if g1.kv.owner == kv.id {
                g1.upto = g1.upto.min(kv.dirty_from).min(past);
            }
        }
    }

    /// [`Self::forward`]'s `chunks` (each its first row of the prompt's and its rows) over two devices: each chunk's
    /// layers before the split's on the first as the second runs the chunk before's from there on, and the head. The
    /// second's copy of its layers' attention cache is brought up to the prompt first, and its recurrent states are
    /// the first's (read back at the first chunk's handoff), the first's again after (steps run there); its layers'
    /// K and V rows go into the host's cache and the first's copy. With a prediction layer each chunk's hidden states
    /// come back to the first for the layer's cache. The last chunk's logits.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_split(&self, m: &Qwen35Model, st: &State, sp: &Split, chain: &dyn DeviceChain, chain1: &dyn DeviceChain, emb: &[f32], chunks: &[(usize, usize)], kv: &mut KvCache, taps: &[usize]) -> (Tensor, Vec<Tapped>) {
        let s = st.dims;
        let past0 = kv.len;
        let end = past0 + chunks.iter().map(|c| c.1).sum::<usize>();
        // the first's copy of the cache (every layer's) up to the prompt, and room for all of it
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        // (the prompt's rows are written from here on: the halves' end there)
        g.halved = g.halved.min(past0);
        reserve(chain, &mut g, &s, st.attention_layers.len(), end.max(kv.expect.min(kv.max_len)));
        sync(chain, &mut g, &s, &st.attention_layers, kv, past0);
        // the second's, of its layers: the rows it lacks (the host's since, and those the first wrote)
        let mut g1 = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
        let layers1: Vec<usize> = sp.attention_slots.iter().map(|&a| st.attention_layers[a]).collect();
        reserve(chain1, &mut g1.kv, &s, layers1.len(), end.max(kv.expect.min(kv.max_len)));
        let from1 = if g1.kv.owner == kv.id { g1.upto.min(kv.dirty_from).min(past0) } else { 0 };
        upload_rows(chain1, &g1.kv, &s, &layers1, kv, from1, past0);
        g1.kv.owner = kv.id;
        kv.dirty_from = usize::MAX;
        // the recurrent states the cache holds, as the first's vectors; the second's sent up from them, or zero for a
        // new conversation's
        let fresh = sp.ssm_slots.iter().all(|&j| kv.ssm_state[st.ssm_layers[j]].is_none() && kv.ssm_conv[st.ssm_layers[j]].is_none());
        let mut pool = st.pool.lock().unwrap_or_else(|p| p.into_inner());
        let states: Vec<(DeviceVec, DeviceVec)> = st
            .ssm_layers
            .iter()
            .enumerate()
            .map(|(i, &l)| (adopt(chain, &mut pool.states[i], &mut kv.ssm_state[l], vec![s.nv, s.dv, s.dk]), adopt(chain, &mut pool.convs[i], &mut kv.ssm_conv[l], vec![s.kern - 1, s.ch])))
            .collect();
        drop(pool);
        let states1: Vec<(DeviceVec, DeviceVec)> = sp.pool.states.iter().cloned().zip(sp.pool.convs.iter().cloned()).collect();
        if fresh {
            for (state, conv) in &states1 {
                chain1.zero(state);
                chain1.zero(conv);
            }
        }
        let n = chunks.len();
        let mut logits = None;
        let tapped;
        {
            let run = SplitRun {
                chained: self,
                m,
                st,
                sp,
                chain,
                chain1,
                emb,
                chunks,
                past0,
                kv0: (&g.layers, g.cap),
                kv1: (&g1.kv.layers, g1.kv.cap),
                states: &states,
                states1: &states1,
                slots0: (0..st.attention_layers.len()).filter(|&a| st.attention_layers[a] < sp.from).collect(),
                layers1: &layers1,
                owner: kv.id,
                taps,
                kept: Default::default(),
                layers_n: kv.ssm_state.len(),
                states_up: !fresh,
                log: std::env::var_os("OAIY_SPLIT_LOG").map(|_| (std::time::Instant::now(), Default::default())),
            };
            // each chunk's first layers gone as the first runs the chunk before's, whose last ones then go to the
            // second as it runs the one before that
            let (mut firsts, mut seconds) = (VecDeque::new(), VecDeque::new());
            // the chunks whose hidden states are back for the prediction layer's cache, and its work on the first
            // after that one's last chunk
            let mut due = Vec::new();
            let mut tail = Vec::new();
            for i in 0..n {
                firsts.push_back(run.first(i, &mut due));
                if i >= 1 {
                    seconds.push_back(run.hand_off(firsts.pop_front().expect("a chunk"), kv));
                }
                if i >= 2 {
                    run.finish(seconds.pop_front().expect("a chunk"), kv, &mut due, &mut logits);
                }
            }
            seconds.push_back(run.hand_off(firsts.pop_front().expect("a chunk"), kv));
            while let Some(r) = seconds.pop_front() {
                run.finish(r, kv, &mut due, &mut logits);
                if !due.is_empty() {
                    tail.push(run.mtp_only(&mut due));
                }
            }
            for r in tail {
                r.finish();
            }
            if let Some((t, l)) = &run.log {
                eprintln!("split run of {n} chunks, {:.1} ms:\n  {}", t.elapsed().as_secs_f64() * 1e3, l.borrow().join("\n  "));
            }
            tapped = run.kept.take();
        }
        g1.upto = end;
        kv.len = end;
        kv.dirty_from = usize::MAX;
        self.runs.fetch_add(n, Ordering::Relaxed);
        (Tensor::from_vec(logits.expect("the last chunk's logits"), vec![1, s.vocab]), tapped)
    }
}
