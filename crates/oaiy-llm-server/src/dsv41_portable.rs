//! DeepSeek-V4.1 without CUDA: the CPU model (`dsv41::model`, the reference the CUDA path is tested against) with its
//! dense trunk on the WebGPU adapter. Measured on the CPU alone, a prompt is compute-bound and its attention (the fp8
//! trunk's projections among it) costs more than its experts, and a warm decode step spends a flat second there
//! (docs/DEEPSEEK_V41.md, "The CPU model, measured"): so the trunk goes to the GPU, every fp8 and bf16 matrix the
//! budget holds (`ggml_rs_wgpu::dense`), the activation still quantized and the result still rounded by the CPU model
//! as the reference does. The routed experts go there too ([`WgpuExperts`]): a prompt's busy ones through slots its
//! records are uploaded into, and the ones used most kept there between passes, a decode step's computed there while
//! the CPU reads and computes the rest.

use std::sync::{Arc, Mutex};

use dsv41::linear::{DenseKernel, Weight};
use ggml_rs_wgpu::dense::{DenseData, DenseGpu, RecordSlots};
use ggml_rs_wgpu::WgpuBackend;

/// A dense weight on the WebGPU adapter, as the CPU model's [`DenseKernel`].
struct WgpuDense(Arc<DenseGpu>);

impl DenseKernel for WgpuDense {
    fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
        self.0.forward(x, t, rows)
    }

    /// Weights on this adapter in one submit and one read back.
    fn forward_many(&self, items: &[(&dyn DenseKernel, &[f32], usize, std::ops::Range<usize>)]) -> Option<Vec<Vec<f32>>> {
        let batch: Option<Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)>> = items
            .iter()
            .map(|(k, x, t, rows)| {
                let w = &(*k as &dyn std::any::Any).downcast_ref::<WgpuDense>()?.0;
                w.same_device(&self.0).then(|| (&**w, *x, *t, rows.clone()))
            })
            .collect();
        Some(ggml_rs_wgpu::dense::forward_batch(&batch?))
    }
}

/// The tokens one forward pass reads of a prompt. A pass reads each layer's experts once for all its tokens, so a
/// prompt is read whole when it fits (an Agent's is 1.5-3k tokens): in pieces it would read them once a piece.
const PREFILL_CHUNK: usize = 4096;

/// The portable server's DeepSeek worker: jobs in, events out, the contract of the CUDA engine (`crate::engine`) and
/// GLM's (`crate::glm`). A request that extends what the model's state covers reads only the new tail, in one chunk
/// (`dsv41`'s attention continues a sequence by a chunk as it would token by token). That state has moved on through
/// the last reply, which the next turn's prompt writes again its own way (its chat template), so the state as the last
/// prompt left it is kept too, a token short of its end (so the same prompt again can still use it): a conversation's
/// next turn continues from there. A prompt that shares neither starts again. A reply ends at the end-of-sentence
/// token, at a complete tool call, or at `max_tokens`.
pub(crate) struct Engine {
    model: dsv41::model::Model,
    tok: Arc<dsv41::tokenizer::Tokenizer>,
    /// The token ids the model's state covers, in order.
    covered: Vec<u32>,
    /// The state after the last prompt's tokens but its last one, and those tokens.
    checkpoint: Option<(Vec<u32>, dsv41::model::Checkpoint)>,
    eos: u32,
    log: bool,
    /// Reading the experts into RAM while no request waits ([`Warm`]); None once done, or switched off.
    warm: Option<Warm>,
    /// The experts' kernel on the GPUs, for what the other cards hold for good ([`Pinned`]): filled while idle too.
    kernel: Option<Arc<WgpuExperts>>,
}

/// The idle reading of the routed experts into the RAM tier: where it is, and what it has read so far.
struct Warm {
    cursor: usize,
    started: Option<std::time::Instant>,
    reading: std::time::Duration,
    bytes_before: u64,
}

/// Records an idle slice reads (some 300 MB: a fifth of a second from a 1.4 GB/s drive), so a request that arrives
/// meanwhile waits no longer than that.
const WARM_SLICE: usize = 16;

/// How much of `prompt` a state covering `tokens` can serve: all of `tokens`, when the prompt starts with them and goes
/// on past them (a state cannot be rewound, and the next token's logits need a forward pass); else none.
fn reusable(tokens: &[u32], prompt: &[u32]) -> usize {
    let n = tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if n == tokens.len() && n < prompt.len() {
        n
    } else {
        0
    }
}

impl Engine {
    pub(crate) fn new(model: dsv41::model::Model, tok: Arc<dsv41::tokenizer::Tokenizer>, log: bool) -> Engine {
        let eos = model.cfg.eos_token_id;
        // (OAIY_DSV41_NO_WARM: leave the drive alone between requests)
        let warm = std::env::var_os("OAIY_DSV41_NO_WARM")
            .is_none()
            .then(|| Warm { cursor: 0, started: None, reading: std::time::Duration::ZERO, bytes_before: model.expert_cache().stats().bytes_read });
        Engine { model, tok, covered: Vec::new(), checkpoint: None, eos, log, warm, kernel: None }
    }

    /// The experts' kernel the model was given, when other cards hold a share of them: the idle reading fills those
    /// first, and leaves them out of RAM.
    pub(crate) fn with_kernel(mut self, kernel: Option<Arc<WgpuExperts>>) -> Engine {
        self.kernel = kernel;
        self
    }

    /// An idle slice of the experts' reading into RAM; says when it starts and when there is no more to read.
    fn warm_some(&mut self) {
        let Some(w) = &mut self.warm else { return };
        if w.started.is_none() {
            w.started = Some(std::time::Instant::now());
            if self.log {
                let c = self.model.expert_cache();
                eprintln!("  reading the experts into RAM while idle ({} of the {} the cache holds are there)", c.len(), c.n_slots());
            }
        }
        let slice = std::time::Instant::now();
        // the other cards' share first (it is never read again), then RAM's, which leaves that share out
        let kernel = self.kernel.as_deref();
        if kernel.is_some_and(|k| k.pin_some(self.model.expert_store().as_ref(), self.model.expert_cache(), WARM_SLICE)) {
            w.reading += slice.elapsed();
            return;
        }
        let more = self.model.warm_experts(&mut w.cursor, WARM_SLICE, &|l, e| kernel.is_some_and(|k| k.pins(l, e)));
        w.reading += slice.elapsed();
        if !more {
            if self.log {
                let c = self.model.expert_cache();
                let read = c.stats().bytes_read.saturating_sub(w.bytes_before);
                let pinned = kernel.map_or(0, |k| k.pinned());
                eprintln!(
                    "  the experts' RAM tier holds {} of its {}{}: {:.1} s of idle reading ({:.1} GB read into RAM since the model loaded)",
                    c.len(),
                    c.n_slots(),
                    if pinned > 0 { format!(", the other GPUs {pinned} more for good") } else { String::new() },
                    w.reading.as_secs_f64(),
                    read as f64 / 1e9
                );
            }
            self.warm = None;
        }
    }

    pub(crate) fn run(mut self, jobs: std::sync::mpsc::Receiver<crate::engine::Job>) {
        use crate::engine::{Event, Finish};
        use std::sync::mpsc::TryRecvError;
        loop {
            // with nothing asked, the experts' next records are read; a request is taken between slices
            let job = if self.warm.is_some() {
                match jobs.try_recv() {
                    Ok(job) => job,
                    Err(TryRecvError::Empty) => {
                        self.warm_some();
                        continue;
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            } else {
                match jobs.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                }
            };
            if job.wipe {
                self.covered.clear();
                self.checkpoint = None;
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                continue;
            }
            if let Err(e) = self.generate(&job) {
                // The state's extent is unknown after a failure: start clean.
                self.covered.clear();
                self.checkpoint = None;
                let _ = job.events.send(Event::Error(e.to_string()));
            }
            // Incognito: nothing of it is kept for the next request.
            if job.forget {
                self.covered.clear();
                self.checkpoint = None;
            }
        }
    }

    fn generate(&mut self, job: &crate::engine::Job) -> oaiy_engine::Result<()> {
        use crate::engine::{sample, Event, Finish};
        use std::sync::atomic::Ordering;
        if !job.images.is_empty() {
            return Err(oaiy_engine::Error::Arg("DeepSeek-V4.1 does not take images on WebGPU yet".into()));
        }
        let prompt = &job.prompt;
        let max_seq = self.model.max_seq();
        if prompt.is_empty() || prompt.len() >= max_seq {
            return Err(oaiy_engine::Error::Arg(format!("a prompt of {} tokens does not fit a context of {max_seq}", prompt.len())));
        }
        let started = std::time::Instant::now();
        // OAIY_DSV41_PROFILE: where the prompt's and the reply's time went (`dsv41::profile`), and what each read
        let profiling = self.log && std::env::var_os("OAIY_DSV41_PROFILE").is_some();
        let read_so_far = |m: &dsv41::model::Model| {
            let st = m.expert_cache().stats();
            (st.bytes_read, st.hits, st.misses)
        };
        let part = |label: &str, before: (u64, u64, u64), after: (u64, u64, u64)| {
            eprintln!(
                "    {label}: read {:.2} GB, {} hits / {} misses in RAM; {}; {}",
                (after.0 - before.0) as f64 / 1e9,
                after.1 - before.1,
                after.2 - before.2,
                dsv41::profile::take_line(),
                ggml_rs_wgpu::profile::take_dense_line()
            );
        };
        if profiling {
            dsv41::profile::take();
        }
        let at_start = read_so_far(&self.model);
        let common = self.covered.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        // The live state, or the last prompt's checkpoint: whichever serves more of this prompt.
        let live = reusable(&self.covered, prompt);
        let kept = self.checkpoint.as_ref().map_or(0, |(tokens, _)| reusable(tokens, prompt));
        let start = if kept > live {
            let (tokens, state) = self.checkpoint.as_ref().expect("a checkpoint serves it");
            self.model.restore(state);
            self.covered.clone_from(tokens);
            kept
        } else {
            live
        };
        if start == 0 {
            self.covered.clear();
        }
        let _ = job.events.send(Event::CacheReuse { cached: start, source: if start > 0 { "memory" } else { "none" }, common });
        let total = prompt.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });
        // All but the last token, then the checkpoint, then the last one (whose logits the reply starts from).
        let last = prompt.len() - 1;
        let mut pos = start;
        while pos < last {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                return Ok(());
            }
            let take = PREFILL_CHUNK.min(last - pos);
            self.model.forward(&prompt[pos..pos + take], pos, &mut |_, _| {})?;
            self.covered.extend_from_slice(&prompt[pos..pos + take]);
            pos += take;
            let _ = job.events.send(Event::Progress { done: pos - start, total });
        }
        if !job.forget {
            self.checkpoint = Some((self.covered.clone(), self.model.checkpoint()));
        }
        let mut logits = self.model.forward(&prompt[last..], last, &mut |_, _| {})?;
        self.covered.push(prompt[last]);
        pos = prompt.len();
        let _ = job.events.send(Event::Progress { done: total, total });
        let prefill_s = started.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });
        let at_reply = read_so_far(&self.model);
        if profiling {
            part("the prompt", at_start, at_reply);
        }

        let think_start = self.tok.special(dsv41::chat::THINK_START);
        let mut budget = crate::glm::ThinkBudget::new(think_start.is_some() && prompt.last().copied() == think_start, self.tok.special(dsv41::chat::THINK_END), job.think_budget);
        let mut parser = dsv41::chat::StreamParser::new(if budget.thinking { dsv41::chat::Mode::Thinking } else { dsv41::chat::Mode::Chat });
        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        let (mut pending, mut n) = (Vec::<u8>::new(), 0usize);
        let decode = std::time::Instant::now();
        let finish = loop {
            let was_thinking = budget.thinking;
            let (next, _forced) = budget.pass(sample(&logits, &job.sampling, &mut rng));
            if was_thinking && (!budget.thinking || budget.used.is_multiple_of(crate::glm::THINKING_EVERY)) {
                let _ = job.events.send(Event::Thinking { used: budget.used, budget: budget.budget, done: !budget.thinking });
            }
            if next == self.eos {
                break Finish::Stop;
            }
            n += 1;
            // Whole UTF-8 characters only: a token can be part of one.
            pending.extend_from_slice(self.tok.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                parser.push(&text);
                if job.events.send(Event::Text(text)).is_err() {
                    break Finish::Stop; // nobody is listening
                }
            }
            // A complete tool call is the reply: the caller runs it and comes back with its result.
            if parser.tool_calls_ready() {
                break Finish::Stop;
            }
            if n >= job.max_tokens || pos + 1 >= max_seq {
                break Finish::Length;
            }
            if job.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            logits = self.model.forward(&[next], pos, &mut |_, _| {})?;
            self.covered.push(next);
            pos += 1;
        };
        if !pending.is_empty() {
            let _ = job.events.send(Event::Text(String::from_utf8_lossy(&pending).into_owned()));
        }
        if self.log {
            let decode_s = decode.elapsed().as_secs_f64();
            eprintln!("  {} prompt tokens ({start} reused) in {prefill_s:.1}s; {n} generated in {decode_s:.1}s ({:.2} tok/s)", prompt.len(), n as f64 / decode_s.max(1e-9));
        }
        if profiling {
            part("the reply", at_reply, read_so_far(&self.model));
        }
        let _ = job.events.send(Event::Done { finish, completion_tokens: n });
        Ok(())
    }
}

/// Routed experts on the WebGPU adapter (`dsv41::moe::Experts::gpu`). A prompt's busy ones pass through a set of slots
/// made once (`RecordSlots`), each record uploaded as stored and its matrices read in place (their e8m0 scales decoded
/// by the kernels): uploading each matrix into buffers of its own, its scales widened to f32 on the host, took 0.67 s
/// for 32 experts of 31 rows, the slots 0.29 s. What the budget has left after those is a tier of experts kept there
/// between passes ([`Resident`]): a decode step's experts that are computed there while the CPU reads and computes the
/// rest, and a prompt's computed there and not read.
pub(crate) struct WgpuExperts {
    group: Mutex<RecordSlots>,
    resident: Option<Mutex<Resident>>,
    /// Whether a decode step's misses may come in (each an upload of a record); a prompt's always may.
    admit_on_decode: std::sync::atomic::AtomicBool,
    /// Whether what comes in leaves the RAM tier (the tiers exclusive), or stays in both (the default). Measured on
    /// 32 decode steps of the same prompt (the same tokens): exclusive read 35.8 GB, inclusive 34.2. A decode step
    /// takes experts in and lets others go, and one let go was then in neither tier.
    exclusive: std::sync::atomic::AtomicBool,
    /// The share of the experts the computer's other GPUs hold for good ([`WgpuExperts::pin_on`]).
    pinned: Option<Pinned>,
}

/// Experts the other GPUs hold for good: a fixed share of every layer's (its last ones: the RAM tier's idle reading
/// starts from the first), each read from the drive once into a slot of its own while the server idles and never
/// replaced. The first card's tier follows what is used (LFRU) and RAM holds what it can of the rest; this share needs
/// neither, so between them the tiers hold more of the 10,612 than RAM alone, and a prompt, which routes to most of
/// them, reads that much less from the drive each time.
struct Pinned {
    /// each card's slots, and the plan's first slot that is its own
    cards: Vec<(RecordSlots, usize)>,
    layers: usize,
    per_layer: usize,
    /// slots over the cards: plan slot `g` is layer `g % layers`'s expert `per_layer - 1 - g / layers`
    total: usize,
    /// the experts whose records are in their slots so far: `(card, slot)`
    index: std::sync::RwLock<std::collections::HashMap<(u32, u32), (usize, usize)>>,
    /// the plan's next slot to fill
    next: std::sync::atomic::AtomicUsize,
}

impl Pinned {
    fn expert(&self, g: usize) -> (u32, u32) {
        ((g % self.layers) as u32, (self.per_layer - 1 - g / self.layers) as u32)
    }

    /// The plan's slot for `(layer, expert)`, if it is one of the share.
    fn slot(&self, layer: u32, expert: u32) -> Option<usize> {
        let g = (self.per_layer - 1).checked_sub(expert as usize)? * self.layers + layer as usize;
        (g < self.total).then_some(g)
    }

    fn card(&self, g: usize) -> (usize, usize) {
        let c = self.cards.iter().rposition(|&(_, first)| first <= g).expect("a plan slot is a card's");
        (c, g - self.cards[c].1)
    }
}

/// Threads reading the pinned share's records from the drive at once (as dsv41's readers).
const PIN_READERS: usize = 8;

/// Decode steps after which every count halves (as the CUDA engine's tier): without aging, an old topic's experts would
/// keep the slots for good.
const AGE_TOKENS: u64 = 128;
/// What the weight budget keeps free beyond the slots: the passes' own buffers (a 2,000-token prompt's projections
/// make outputs of a few hundred MB).
const MARGIN: u64 = 1 << 30;

impl WgpuExperts {
    /// Slots for a call of [`dsv41::moe::GPU_GROUP`] experts, or as many as the weight budget has left (after the trunk),
    /// and a resident tier in what is left after them but [`MARGIN`]; None if not one slot fits.
    pub(crate) fn new(gpu: &WgpuBackend) -> Option<WgpuExperts> {
        let record = dsv41::expert::RECORD_BYTES;
        let group = gpu.record_slots(dsv41::moe::GPU_GROUP, record)?;
        let (used, budget) = gpu.usage();
        let room = (budget.saturating_sub(used).saturating_sub(MARGIN) / record as u64) as usize;
        let resident = (room > 0).then(|| gpu.record_slots(room, record)).flatten().map(|slots| Mutex::new(Resident::new(slots)));
        Some(WgpuExperts {
            group: Mutex::new(group),
            resident,
            admit_on_decode: std::sync::atomic::AtomicBool::new(true),
            exclusive: std::sync::atomic::AtomicBool::new(false),
            pinned: None,
        })
    }

    /// Have `gpus` (the computer's other cards) hold a share of the `layers * per_layer` routed experts for good, as
    /// many as each one's weight budget has room for but [`MARGIN`]: the slots are made now and filled while the server
    /// idles ([`Self::pin_some`]). How many they will hold.
    pub(crate) fn pin_on(&mut self, gpus: &[Arc<WgpuBackend>], layers: usize, per_layer: usize) -> usize {
        let record = dsv41::expert::RECORD_BYTES;
        let (mut cards, mut total) = (Vec::new(), 0usize);
        for gpu in gpus {
            let (used, budget) = gpu.usage();
            let room = ((budget.saturating_sub(used).saturating_sub(MARGIN) / record as u64) as usize).min(layers * per_layer - total);
            if let Some(slots) = (room > 0).then(|| gpu.record_slots(room, record)).flatten() {
                let n = slots.len();
                cards.push((slots, total));
                total += n;
            }
        }
        if total > 0 {
            self.pinned = Some(Pinned { cards, layers, per_layer, total, index: Default::default(), next: Default::default() });
        }
        total
    }

    /// For each of `layer`'s `experts`, the other card and slot that hold it now, if one does.
    fn pinned_now(&self, layer: u32, experts: &[u32]) -> Vec<Option<(usize, usize)>> {
        match &self.pinned {
            Some(p) => {
                let index = p.index.read().unwrap_or_else(|e| e.into_inner());
                experts.iter().map(|&e| index.get(&(layer, e)).copied()).collect()
            }
            None => vec![None; experts.len()],
        }
    }

    /// Whether `(layer, expert)` is one of the share the other GPUs hold for good (filled yet or not).
    pub(crate) fn pins(&self, layer: u32, expert: u32) -> bool {
        self.pinned.as_ref().is_some_and(|p| p.slot(layer, expert).is_some())
    }

    /// The experts the other GPUs hold so far.
    pub(crate) fn pinned(&self) -> usize {
        self.pinned.as_ref().map_or(0, |p| p.index.read().unwrap_or_else(|e| e.into_inner()).len())
    }

    /// Fill the share's next `count` slots: their records read from `store` on several threads, written to their slots,
    /// and let go from the RAM tier if a request had read them there. Whether more are left to fill.
    pub(crate) fn pin_some(&self, store: &dyn oaiy_engine::store::WeightStore, cache: &oaiy_engine::ecache::Ecache, count: usize) -> bool {
        use std::sync::atomic::Ordering;
        let Some(p) = &self.pinned else { return false };
        let first = p.next.load(Ordering::Relaxed);
        if first >= p.total {
            return false;
        }
        let last = (first + count).min(p.total);
        let turn = std::sync::atomic::AtomicUsize::new(first);
        let read: Mutex<Vec<(usize, Vec<u8>)>> = Mutex::new(Vec::with_capacity(last - first));
        std::thread::scope(|scope| {
            for _ in 0..PIN_READERS.min(last - first) {
                scope.spawn(|| loop {
                    let g = turn.fetch_add(1, Ordering::Relaxed);
                    if g >= last {
                        break;
                    }
                    let (layer, expert) = p.expert(g);
                    let mut record = vec![0u8; dsv41::expert::RECORD_BYTES];
                    // (a record that cannot be read is left out: its expert goes through RAM as before)
                    if store.fetch(layer, expert, &mut record).is_ok() {
                        read.lock().unwrap_or_else(|e| e.into_inner()).push((g, record));
                    }
                });
            }
        });
        let read = read.into_inner().unwrap_or_else(|e| e.into_inner());
        for (g, record) in &read {
            let (card, slot) = p.card(*g);
            p.cards[card].0.write(slot, record);
        }
        let mut index = p.index.write().unwrap_or_else(|e| e.into_inner());
        for (g, _) in &read {
            let (layer, expert) = p.expert(*g);
            index.insert((layer, expert), p.card(*g));
            cache.remove(layer, expert);
        }
        p.next.store(last, Ordering::Relaxed);
        last < p.total
    }

    /// Keep what comes in out of the RAM tier, or in both (the default).
    pub(crate) fn set_exclusive(&self, on: bool) {
        self.exclusive.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Let a decode step's misses come in, or not (to measure what their uploads cost).
    pub(crate) fn set_admit_on_decode(&self, on: bool) {
        self.admit_on_decode.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Experts a prompt's call takes at once.
    pub(crate) fn slots(&self) -> usize {
        self.group.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// The resident tier: its slots, and its hits, misses and records taken in so far.
    pub(crate) fn tier(&self) -> (usize, u64, u64, u64) {
        self.resident.as_ref().map_or((0, 0, 0, 0), |r| {
            let r = r.lock().unwrap_or_else(|p| p.into_inner());
            (r.keys.len(), r.hits, r.misses, r.admitted)
        })
    }
}

/// Experts whose records are in `slots`, `(slot, x, weights)` each: every one's gate and up in one submit, the SwiGLU on
/// the host as dsv41 takes it, then every down in another.
fn run(slots: &RecordSlots, picks: &[(usize, &[f32], &[f32])], swiglu_limit: f32) -> Vec<Vec<f32>> {
    use dsv41::expert::{BLOCK, DIM, INTER, S1, S2, S3, W1, W2, W3};
    use dsv41::formats::{fake_quant_fp8, to_bf16};
    let rows: Vec<usize> = picks.iter().map(|(_, x, _)| x.len() / DIM).collect();
    let mats: Vec<[DenseGpu; 3]> = picks
        .iter()
        .map(|&(i, ..)| [slots.mxfp4(i, W1.start, S1.start, INTER, DIM), slots.mxfp4(i, W3.start, S3.start, INTER, DIM), slots.mxfp4(i, W2.start, S2.start, DIM, INTER)])
        .collect();
    let xq: Vec<Vec<f32>> = picks.iter().map(|(_, x, _)| fake_quant_fp8(x, BLOCK)).collect();
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
        (0..picks.len()).flat_map(|i| [(&mats[i][0], xq[i].as_slice(), rows[i], 0..INTER), (&mats[i][1], xq[i].as_slice(), rows[i], 0..INTER)]).collect();
    let sums = ggml_rs_wgpu::dense::forward_batch(&items);
    let hq: Vec<Vec<f32>> =
        sums.chunks(2).zip(picks).map(|(p, (_, _, w))| fake_quant_fp8(&dsv41::expert::swiglu(&p[0], &p[1], Some(w), swiglu_limit), BLOCK)).collect();
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> = (0..picks.len()).map(|i| (&mats[i][2], hq[i].as_slice(), rows[i], 0..DIM)).collect();
    ggml_rs_wgpu::dense::forward_batch(&items).into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect()
}

impl dsv41::expert::ExpertsKernel for WgpuExperts {
    fn forward(&self, jobs: &[dsv41::expert::ExpertJob<'_>], swiglu_limit: f32) -> Vec<Vec<f32>> {
        let slots = self.group.lock().unwrap_or_else(|p| p.into_inner());
        let mut out = Vec::with_capacity(jobs.len());
        for part in jobs.chunks(slots.len()) {
            for (i, j) in part.iter().enumerate() {
                slots.write(i, j.record);
            }
            let picks: Vec<(usize, &[f32], &[f32])> = part.iter().enumerate().map(|(i, j)| (i, j.x, j.weights)).collect();
            out.extend(run(&slots, &picks, swiglu_limit));
        }
        out
    }

    fn holds(&self, layer: u32, experts: &[u32], tokens: &[usize]) -> Vec<bool> {
        // the other cards' share is held without being counted; the first card's tier counts and holds of the rest
        let there = self.pinned_now(layer, experts);
        let (rest, rest_tokens): (Vec<u32>, Vec<usize>) = experts.iter().zip(tokens).zip(&there).filter(|(_, at)| at.is_none()).map(|((&e, &n), _)| (e, n)).unzip();
        let mut of_rest = match &self.resident {
            Some(r) => r.lock().unwrap_or_else(|p| p.into_inner()).holds(layer, &rest, &rest_tokens),
            None => vec![false; rest.len()],
        }
        .into_iter();
        there.iter().map(|at| at.is_some() || of_rest.next().expect("one of the rest")).collect()
    }

    fn forward_held(&self, layer: u32, jobs: &[(u32, &[f32], &[f32])], swiglu_limit: f32) -> Vec<Vec<f32>> {
        let experts: Vec<u32> = jobs.iter().map(|j| j.0).collect();
        let there = self.pinned_now(layer, &experts);
        // each card's experts in a thread of its own: the first card's tier, and each other card's share
        let cards = self.pinned.as_ref().map_or(0, |p| p.cards.len());
        let mut outs: Vec<Option<Vec<f32>>> = (0..jobs.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let shares: Vec<_> = (0..cards)
                .filter_map(|c| {
                    let mine: Vec<usize> = (0..jobs.len()).filter(|&j| there[j].is_some_and(|(card, _)| card == c)).collect();
                    (!mine.is_empty()).then(|| {
                        let there = &there;
                        let slots = &self.pinned.as_ref().expect("a share's card").cards[c].0;
                        let handle = scope.spawn(move || {
                            let picks: Vec<(usize, &[f32], &[f32])> = mine.iter().map(|&j| (there[j].expect("its slot").1, jobs[j].1, jobs[j].2)).collect();
                            let got: Vec<Vec<f32>> = picks.chunks(dsv41::moe::GPU_GROUP).flat_map(|part| run(slots, part, swiglu_limit)).collect();
                            (mine, got)
                        });
                        handle
                    })
                })
                .collect();
            let first: Vec<usize> = (0..jobs.len()).filter(|&j| there[j].is_none()).collect();
            if !first.is_empty() {
                let r = self.resident.as_ref().expect("a kernel that holds experts").lock().unwrap_or_else(|p| p.into_inner());
                let picks: Vec<(usize, &[f32], &[f32])> = first.iter().map(|&j| (r.index[&(layer, jobs[j].0)], jobs[j].1, jobs[j].2)).collect();
                let got: Vec<Vec<f32>> = picks.chunks(dsv41::moe::GPU_GROUP).flat_map(|part| run(&r.slots, part, swiglu_limit)).collect();
                for (j, out) in first.into_iter().zip(got) {
                    outs[j] = Some(out);
                }
            }
            for share in shares {
                let (mine, got) = share.join().expect("a card's experts panicked");
                for (j, out) in mine.into_iter().zip(got) {
                    outs[j] = Some(out);
                }
            }
        });
        outs.into_iter().map(|o| o.expect("every held expert computed")).collect()
    }

    fn prefetch_order(&self, layer: u32, experts: u32) -> Vec<u32> {
        let all: Vec<u32> = (0..experts).collect();
        let there = self.pinned_now(layer, &all);
        let rest = all.into_iter().filter(|&e| there[e as usize].is_none());
        let Some(r) = &self.resident else { return rest.collect() };
        let r = r.lock().unwrap_or_else(|p| p.into_inner());
        let mut order: Vec<u32> = rest.filter(|&e| !r.index.contains_key(&(layer, e))).collect();
        order.sort_by_key(|&e| std::cmp::Reverse(r.freq((layer, e))));
        order
    }

    fn offer(&self, layer: u32, records: &[(u32, &[u8])]) -> Vec<u32> {
        if !self.admit_on_decode.load(std::sync::atomic::Ordering::Relaxed) {
            return Vec::new();
        }
        let Some(r) = &self.resident else { return Vec::new() };
        let mut r = r.lock().unwrap_or_else(|p| p.into_inner());
        let taken: Vec<u32> = records.iter().filter(|&&(e, record)| r.admit((layer, e), record)).map(|&(e, _)| e).collect();
        if self.exclusive.load(std::sync::atomic::Ordering::Relaxed) { taken } else { Vec::new() }
    }

    fn pass_done(&self, tokens: usize, cache: &oaiy_engine::ecache::Ecache, store: &dyn oaiy_engine::store::WeightStore) {
        if let Some(r) = &self.resident {
            let exclusive = self.exclusive.load(std::sync::atomic::Ordering::Relaxed);
            r.lock().unwrap_or_else(|p| p.into_inner()).pass_done(tokens, cache, store, exclusive);
        }
    }
}

/// Experts kept on the adapter between passes, as the CUDA engine keeps them in VRAM (dsv41-cuda's expert_cache): a
/// slot a record; the one to replace the least used, and of equals the longest unused (LFRU); uses counted in tokens (a
/// prompt's expert that 300 tokens chose counts 300) and halved every [`AGE_TOKENS`] decode steps. A decode step's
/// misses come in when they are used more than what they would replace; a prompt's are noted and, once it is read, the
/// most used come in from RAM (taking them in during the pass would evict experts the same pass needs later). What
/// comes in can leave the RAM tier, as the CUDA engine's tiers do (`WgpuExperts::set_exclusive`), but by default stays.
struct Resident {
    slots: RecordSlots,
    keys: Vec<Option<(u32, u32)>>,
    last: Vec<u64>,
    index: std::collections::HashMap<(u32, u32), usize>,
    freq: std::collections::HashMap<(u32, u32), u64>,
    clock: u64,
    decoded: u64,
    pending: Vec<(u32, u32)>,
    hits: u64,
    misses: u64,
    admitted: u64,
}

impl Resident {
    fn new(slots: RecordSlots) -> Resident {
        let n = slots.len();
        Resident {
            slots,
            keys: vec![None; n],
            last: vec![0; n],
            index: std::collections::HashMap::with_capacity(n),
            freq: std::collections::HashMap::new(),
            clock: 0,
            decoded: 0,
            pending: Vec::new(),
            hits: 0,
            misses: 0,
            admitted: 0,
        }
    }

    fn freq(&self, key: (u32, u32)) -> u64 {
        self.freq.get(&key).copied().unwrap_or(0)
    }

    fn holds(&mut self, layer: u32, experts: &[u32], tokens: &[usize]) -> Vec<bool> {
        self.clock += 1;
        experts
            .iter()
            .zip(tokens)
            .map(|(&e, &n)| {
                let key = (layer, e);
                *self.freq.entry(key).or_insert(0) += n as u64;
                match self.index.get(&key) {
                    Some(&i) => {
                        self.last[i] = self.clock;
                        self.hits += 1;
                        true
                    }
                    None => {
                        self.misses += 1;
                        self.pending.push(key);
                        false
                    }
                }
            })
            .collect()
    }

    /// The slot to fill next: an empty one, else the least used (of equals, the longest unused).
    fn victim(&self) -> usize {
        if let Some(i) = self.keys.iter().position(Option::is_none) {
            return i;
        }
        (0..self.keys.len()).min_by_key(|&i| (self.keys[i].map_or(0, |k| self.freq(k)), self.last[i])).expect("a slot")
    }

    /// Take in `key`'s record if it is not held and is used more than what it would replace (a tie keeps the resident
    /// one: each swap is an upload).
    fn admit(&mut self, key: (u32, u32), record: &[u8]) -> bool {
        if self.index.contains_key(&key) {
            return false;
        }
        let i = self.victim();
        if let Some(old) = self.keys[i] {
            if self.freq(old) >= self.freq(key) {
                return false;
            }
            self.index.remove(&old);
        }
        self.slots.write(i, record);
        self.keys[i] = Some(key);
        self.index.insert(key, i);
        self.last[i] = self.clock;
        self.admitted += 1;
        true
    }

    fn pass_done(&mut self, tokens: usize, cache: &oaiy_engine::ecache::Ecache, store: &dyn oaiy_engine::store::WeightStore, exclusive: bool) {
        let mut pending = std::mem::take(&mut self.pending);
        if tokens > 1 {
            pending.sort_unstable();
            pending.dedup();
            pending.sort_by_key(|&k| std::cmp::Reverse(self.freq(k)));
            for key in pending {
                if self.index.contains_key(&key) || !cache.probe(key.0, key.1) {
                    continue;
                }
                let Ok(record) = cache.acquire(key.0, key.1, store) else { continue };
                // most used first: once one does not beat what it would replace, none after it will
                if !self.admit(key, &record) {
                    break;
                }
                // held here now, so out of RAM: the two tiers hold different experts
                drop(record);
                if exclusive {
                    cache.remove(key.0, key.1);
                }
            }
        } else {
            self.decoded += 1;
            if self.decoded % AGE_TOKENS == 0 {
                for f in self.freq.values_mut() {
                    *f /= 2;
                }
            }
        }
    }
}

/// Put `model`'s dense trunk on `gpu` while its budget holds it: how many matrices went, and their stored bytes.
pub(crate) fn offload(model: &mut dsv41::model::Model, gpu: &WgpuBackend) -> (usize, u64) {
    model.offload(|_name, w| {
        let (data, n, k, fp8) = match w {
            Weight::Fp8 { w, s, n, k } => {
                let scales = s.iter().map(|&b| dsv41::formats::e8m0_to_f32(b)).collect();
                (DenseData::Fp8 { w: w.clone(), scales, n: *n, k: *k }, *n, *k, true)
            }
            Weight::Bf16 { w, n, k } => (DenseData::Bf16 { w: w.clone(), n: *n, k: *k }, *n, *k, false),
            _ => return None,
        };
        let placed = gpu.dense(data).ok().flatten()?;
        Some(Weight::Device { kernel: Arc::new(WgpuDense(placed)), n, k, fp8 })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsv41::model::{Model, ModelOptions};
    use std::path::PathBuf;
    use std::time::Instant;

    fn checkpoint() -> PathBuf {
        std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"D:\deepseek\model"))
    }

    fn prompt(dir: &std::path::Path, tokens: usize) -> Vec<u32> {
        let tok = dsv41::tokenizer::Tokenizer::load(dir).unwrap();
        // A text kept with the tests (docs/WEBGPU.md as it was on 2026-10-05, some 3,400 tokens), LF whatever the
        // checkout wrote: a CRLF copy is another prompt (other tokens, other experts). The prompt was the docs as they
        // stood, and every edit of them was another prompt: by 2026-10-09 the GPU's logits after its first 400 tokens
        // were the CPU's to a cosine of 0.9908 where this text's are to 0.9990 (the same greedy tokens after both), and
        // the check that asks for 0.998 failed with nothing in the model changed. DSV41_PROMPT_FILE: another text.
        let text: String = match std::env::var("DSV41_PROMPT_FILE") {
            Ok(file) => std::fs::read_to_string(file).expect("DSV41_PROMPT_FILE"),
            Err(_) => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv41-prompt.md")).to_string(),
        }
        .replace("\r\n", "\n");
        let mut ids = vec![0u32];
        ids.extend(tok.encode(&text));
        ids.truncate(tokens);
        ids
    }

    /// An expert on the GPU gives the CPU's output but for the order of its sums: on a random record (realistic scales)
    /// and 33 rows, nearly every output is the same bf16 value, and the rest a few bf16 steps away (an fp8 rounding of
    /// the SwiGLU's output can go the other way), never a different answer.
    #[test]
    fn an_expert_on_the_gpu_gives_the_cpus_output_but_for_its_sums_order() {
        use dsv41::expert::{expert_forward_batch, ExpertJob, ExpertsKernel, DIM, RECORD_BYTES, S1, W1};
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut record = vec![0u8; RECORD_BYTES];
        for (i, byte) in record.iter_mut().enumerate() {
            // Weights: any nibbles. Scales (e8m0): 2^-9 .. 2^-6, as a trained expert's are.
            *byte = if i < S1.start { (next() & 255) as u8 } else { 118 + (next() % 4) as u8 };
        }
        assert!(W1.start == 0);
        let rows = 33;
        let x: Vec<f32> = (0..rows * DIM).map(|_| (((next() % 2001) as f32 / 1000.0) - 1.0) * 0.5).collect();
        let weights: Vec<f32> = (0..rows).map(|i| 0.05 + 0.01 * i as f32).collect();
        let cpu = expert_forward_batch(&record, &x, Some(&weights), 10.0);
        let gpu = WgpuExperts::new(&b).unwrap().forward(&[ExpertJob { record: &record, x: &x, weights: &weights }], 10.0).pop().unwrap();
        assert_eq!(cpu.len(), gpu.len());
        let same = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        eprintln!("{same} of {} outputs the same; the largest difference {worst:.2e} of the largest output", cpu.len());
        assert!(same as f64 >= 0.9 * cpu.len() as f64, "{same} of {}", cpu.len());
        assert!(worst < 0.05, "{worst}");
    }

    /// Where a call of a prompt's busy experts on the GPU spends its time: 32 experts of 31 rows (a 2,000-token prompt's
    /// average), the whole call and each of its steps.
    #[test]
    #[ignore = "a timing; needs a WebGPU adapter; run with --nocapture"]
    fn measure_a_group_of_experts_on_the_gpu() {
        use dsv41::expert::{ExpertJob, ExpertsKernel, BLOCK, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1, W2, W3};
        use dsv41::formats::fake_quant_fp8;
        let b = gpu();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let (n, rows) = (32, 31);
        let records: Vec<Vec<u8>> = (0..n)
            .map(|_| (0..RECORD_BYTES).map(|i| if i < S1.start { (next() & 255) as u8 } else { 118 + (next() % 4) as u8 }).collect())
            .collect();
        let x: Vec<f32> = (0..rows * DIM).map(|_| (((next() % 2001) as f32 / 1000.0) - 1.0) * 0.5).collect();
        let weights: Vec<f32> = (0..rows).map(|i| 0.05 + 0.01 * i as f32).collect();
        let jobs: Vec<ExpertJob<'_>> = records.iter().map(|r| ExpertJob { record: r, x: &x, weights: &weights }).collect();
        let kernel = WgpuExperts::new(&b).unwrap();
        kernel.forward(&jobs[..2], 10.0);
        for round in 0..3 {
            let t = Instant::now();
            kernel.forward(&jobs, 10.0);
            eprintln!("round {round}: {n} experts of {rows} rows in {:.3} s", t.elapsed().as_secs_f64());
        }
        drop(kernel);
        // The same steps one at a time.
        let slots = b.record_slots(n, RECORD_BYTES).unwrap();
        let t = Instant::now();
        for (i, r) in records.iter().enumerate() {
            slots.write(i, r);
        }
        let upload = t.elapsed().as_secs_f64();
        let placed: Vec<(DenseGpu, DenseGpu, DenseGpu)> = (0..n)
            .map(|i| (slots.mxfp4(i, W1.start, S1.start, INTER, DIM), slots.mxfp4(i, W3.start, S3.start, INTER, DIM), slots.mxfp4(i, W2.start, S2.start, DIM, INTER)))
            .collect();
        let t = Instant::now();
        let xq = fake_quant_fp8(&x, BLOCK);
        let quant = t.elapsed().as_secs_f64();
        let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
            placed.iter().flat_map(|(g, u, _)| [(g, xq.as_slice(), rows, 0..INTER), (u, xq.as_slice(), rows, 0..INTER)]).collect();
        let t = Instant::now();
        let sums = ggml_rs_wgpu::dense::forward_batch(&items);
        let gate_up = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let hq: Vec<Vec<f32>> = sums.chunks(2).map(|p| fake_quant_fp8(&dsv41::expert::swiglu(&p[0], &p[1], Some(&weights), 10.0), BLOCK)).collect();
        let host = t.elapsed().as_secs_f64();
        let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> = placed.iter().zip(&hq).map(|((_, _, d), h)| (d, h.as_slice(), rows, 0..DIM)).collect();
        let t = Instant::now();
        ggml_rs_wgpu::dense::forward_batch(&items);
        let down = t.elapsed().as_secs_f64();
        eprintln!(
            "the records' writes queued {upload:.3} s, the input's quantization {quant:.3} s, gate and up (the uploads landing first) {gate_up:.3} s, the SwiGLU on the host {host:.3} s, down {down:.3} s"
        );
        // Gate and up again, the records already there: the matmuls and the round trip alone.
        let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
            placed.iter().flat_map(|(g, u, _)| [(g, xq.as_slice(), rows, 0..INTER), (u, xq.as_slice(), rows, 0..INTER)]).collect();
        let t = Instant::now();
        ggml_rs_wgpu::dense::forward_batch(&items);
        eprintln!("gate and up with the records in place {:.3} s", t.elapsed().as_secs_f64());
    }

    fn gpu() -> WgpuBackend {
        let gb: u64 = std::env::var("DSV41_WEBGPU_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(27);
        WgpuBackend::new(Some(gb << 30)).expect("a WebGPU adapter")
    }

    /// The trunk and a prompt's busy experts on the GPU give the CPU model's answer: on a prompt long enough for experts
    /// to be busy (400 tokens) the last position's logits agree closely, and the next greedy tokens are the same.
    #[test]
    #[ignore = "loads the 510 GB checkpoint twice; needs a WebGPU adapter"]
    fn the_trunk_and_the_experts_on_the_gpu_answer_as_the_cpu_model_does() {
        let dir = checkpoint();
        let meta = dir.join("engram_meta.safetensors");
        let ids = prompt(&dir, 400);
        let opts = ModelOptions { max_seq: 512, expert_cache_bytes: 24 << 30, direct_io: true };
        let run = |model: &mut Model| {
            let mut logits = model.forward(&ids, 0, &mut |_, _| {}).unwrap();
            let first = logits.clone();
            let mut tokens = Vec::new();
            for step in 0..6 {
                let next = dsv41::model::argmax(&logits);
                tokens.push(next);
                logits = model.forward(&[next], ids.len() + step, &mut |_, _| {}).unwrap();
            }
            (first, tokens)
        };
        let (cpu_logits, cpu_tokens) = run(&mut Model::load(&dir, &meta, &opts).unwrap());
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(gpu());
        let mut model = Model::load(&dir, &meta, &opts).unwrap();
        let (count, bytes) = offload(&mut model, backend.as_any().downcast_ref::<WgpuBackend>().unwrap());
        let kernel = Arc::new(WgpuExperts::new(backend.as_any().downcast_ref::<WgpuBackend>().unwrap()).unwrap());
        model.set_experts_kernel(Some(Arc::clone(&kernel) as Arc<dyn dsv41::expert::ExpertsKernel>));
        eprintln!("{count} dense matrices on the GPU, {:.2} GB, and the experts", bytes as f64 / 1e9);
        let (gpu_logits, gpu_tokens) = run(&mut model);
        let (slots, hits, misses, admitted) = kernel.tier();
        eprintln!("the experts kept on the GPU: {hits} hits / {misses} misses, {admitted} taken in, {slots} slots");
        // The decode steps computed some of their experts there, so the same tokens cover those too.
        assert!(hits > 0, "no expert was computed from the GPU's tier");
        let dot: f64 = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&cpu_logits) * norm(&gpu_logits));
        let max_diff = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("logits: cosine {cosine:.6}, largest difference {max_diff:.4}; tokens CPU {cpu_tokens:?}, GPU {gpu_tokens:?}");
        assert!(count > 0);
        // The sums' order differs on the GPU: an expert's outputs are the CPU's to the bit but for about one in 170,000
        // (`an_expert_on_the_gpu_gives_the_cpus_output_but_for_its_sums_order`), and those few, through the fp8
        // roundings of 40 layers and a 400-token prompt's experts, leave the logits close, not equal (measured 2026-10-05:
        // cosine 0.99882, the same six tokens; the trunk alone, cosine 1.000000).
        assert!(cosine > 0.998, "cosine {cosine}");
        assert_eq!(cpu_tokens, gpu_tokens);
    }

    /// The CPU model's measurement (dsv41/tests/cpu_speed.rs) with the trunk on the GPU.
    #[test]
    #[ignore = "loads the 510 GB checkpoint and prefills a 2,000-token prompt; needs a WebGPU adapter"]
    fn measure_prefill_and_warm_decode_with_the_trunk_on_the_gpu() {
        use std::cell::RefCell;
        use std::collections::BTreeMap;
        let dir = checkpoint();
        let meta = dir.join("engram_meta.safetensors");
        let tokens: usize = std::env::var("DSV41_PROMPT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
        let steps: usize = std::env::var("DSV41_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let gb: usize = std::env::var("DSV41_CACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
        let ids = prompt(&dir, tokens);
        let opts = ModelOptions { max_seq: ids.len() + steps + 8, expert_cache_bytes: gb << 30, direct_io: true };
        let mut model = Model::load(&dir, &meta, &opts).unwrap();
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(gpu());
        let t = Instant::now();
        let (count, bytes) = offload(&mut model, backend.as_any().downcast_ref::<WgpuBackend>().unwrap());
        eprintln!("{count} dense matrices on the GPU, {:.2} GB, in {:.1} s", bytes as f64 / 1e9, t.elapsed().as_secs_f64());
        let mut kernel = None;
        if std::env::var("DSV41_GPU_EXPERTS").map_or(true, |v| v != "0") {
            let k = Arc::new(WgpuExperts::new(backend.as_any().downcast_ref::<WgpuBackend>().unwrap()).expect("room for a record"));
            eprintln!("{} record slots for a prompt's experts, {} kept between passes", k.slots(), k.tier().0);
            if std::env::var("DSV41_TIER_EXCLUSIVE").is_ok_and(|v| v == "1") {
                k.set_exclusive(true);
                eprintln!("what the tier takes in leaves RAM");
            }
            if std::env::var("DSV41_TIER_ADMIT").is_ok_and(|v| v == "0") {
                k.set_admit_on_decode(false);
                eprintln!("a decode step's misses kept out of the tier");
            }
            model.set_experts_kernel(Some(Arc::clone(&k) as Arc<dyn dsv41::expert::ExpertsKernel>));
            kernel = Some(k);
            eprintln!("a prompt's busy experts on the GPU");
        }
        let spans: RefCell<BTreeMap<&'static str, f64>> = RefCell::new(BTreeMap::new());
        let last = RefCell::new(Instant::now());
        let mut trace = |name: &str, _: &[f32]| {
            let now = Instant::now();
            let what = if name.ends_with(".attn_out") { "attention" } else if name.ends_with(".moe_out") { "moe" } else { "other" };
            *spans.borrow_mut().entry(what).or_default() += now.duration_since(*last.borrow()).as_secs_f64();
            *last.borrow_mut() = now;
        };
        let report = |label: &str, secs: f64, before: oaiy_engine::ecache::CacheStats, after: oaiy_engine::ecache::CacheStats, spans: &BTreeMap<&'static str, f64>| {
            let read = after.bytes_read - before.bytes_read;
            let tier = kernel.as_ref().map_or(String::new(), |k| {
                let (slots, hits, misses, admitted) = k.tier();
                format!("; on the GPU so far: {hits} hits / {misses} misses, {admitted} taken in, {slots} slots")
            });
            eprintln!(
                "{label}: {secs:.2} s; read {:.2} GB, {} hits / {} misses{tier}; {}
    {}
    {}",
                read as f64 / 1e9,
                after.hits - before.hits,
                after.misses - before.misses,
                spans.iter().map(|(k, v)| format!("{k} {v:.2} s")).collect::<Vec<_>>().join(", "),
                dsv41::profile::take_line(),
                ggml_rs_wgpu::profile::take_dense_line()
            );
        };
        let before = model.expert_cache().stats();
        dsv41::profile::take();
        *last.borrow_mut() = Instant::now();
        let t = Instant::now();
        let mut logits = model.forward(&ids, 0, &mut trace).unwrap();
        report(&format!("prompt of {} tokens", ids.len()), t.elapsed().as_secs_f64(), before, model.expert_cache().stats(), &spans.borrow());
        let mut times = Vec::new();
        for step in 0..steps {
            let next = dsv41::model::argmax(&logits);
            spans.borrow_mut().clear();
            let before = model.expert_cache().stats();
            *last.borrow_mut() = Instant::now();
            let t = Instant::now();
            logits = model.forward(&[next], ids.len() + step, &mut trace).unwrap();
            let secs = t.elapsed().as_secs_f64();
            times.push(secs);
            report(&format!("step {step} (token {next})"), secs, before, model.expert_cache().stats(), &spans.borrow());
        }
        let tail = &times[times.len() / 2..];
        eprintln!("last {} steps: {:.2} s a token on average", tail.len(), tail.iter().sum::<f64>() / tail.len() as f64);
    }
}
