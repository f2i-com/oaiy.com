//! GLM-5.3-Flash served through the same job/event contract as
//! the DeepSeek engine.
//!
//! [`engine::Engine`](crate::engines::job::Engine) is built around `dsv41-cuda`'s
//! `GpuModel`: safetensors on disk, Engram tables, layered prefill, checkpoints,
//! image spans. None of that applies to a GGUF opened through `crate::llm::Model`,
//! and rewriting that engine to abstract over both would put the working DeepSeek
//! path at risk for no gain.
//!
//! It does not need abstracting, because the seam is already there. `api.rs` and
//! `http.rs` contain no reference to `Engine` at all: the engine runs on its own
//! thread, takes [`Job`]s off a channel and sends [`Event`]s back. So this is a
//! sibling rather than a modification — same channel, same events, different model
//! behind it — and [`crate::models`] swaps which one is running.
//!
//! # Why the state is kept between requests
//!
//! GLM prefills at about 6 tokens a second on this machine, so re-reading a
//! 4,000-token conversation on every turn would cost eleven minutes. It does not
//! have to: `Glm5NextModel` carries its own recurrent state and continues from
//! wherever the last `forward` left off, so a turn that *extends* the previous
//! prompt only has to read the new tail.
//!
//! This engine therefore remembers which token ids its state covers. A prompt that
//! extends that run forwards only the suffix; one that diverges resets and starts
//! again. That is the same idea as the DeepSeek engine's checkpoint ladder, minus
//! the ladder: there is one live state rather than a set to rewind to, because
//! glm5next has no cheap snapshot.

use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use crate::llm::{KvCache, Model};
use oaiy_engine::Result;
use tokenizer::Tokenizer;

use crate::engines::job::{sample, Event, Finish, Job};

/// A reply's reasoning budget, as the DeepSeek engine does it: count the tokens
/// spent inside the `<think>` block the prompt opened and, at the budget, put the
/// closing tag in place of the next one so a model going in circles stops and
/// answers.
pub(crate) struct ThinkBudget {
    pub(crate) thinking: bool,
    end: Option<u32>,
    pub(crate) used: usize,
    pub(crate) budget: Option<usize>,
}

impl ThinkBudget {
    pub(crate) fn new(thinking: bool, end: Option<u32>, budget: Option<usize>) -> Self {
        Self { thinking, end, used: 0, budget }
    }

    /// `(token, forced)` — `forced` when the budget replaced the sampled token.
    pub(crate) fn pass(&mut self, next: u32) -> (u32, bool) {
        if !self.thinking {
            return (next, false);
        }
        if Some(next) == self.end {
            self.thinking = false;
            return (next, false);
        }
        self.used += 1;
        match (self.budget, self.end) {
            (Some(b), Some(end)) if self.used >= b => {
                self.thinking = false;
                (end, true)
            }
            _ => (next, false),
        }
    }
}

/// How often a long reasoning block reports progress, in tokens.
pub(crate) const THINKING_EVERY: usize = 32;

/// A dense model's prompt chunk ([`GlmEngine::prefill_step`]).
const DENSE_PREFILL_CHUNK: usize = 512;

pub struct GlmEngine {
    model: Model,
    tok: Arc<Tokenizer>,
    /// The `KvCache` the generic API takes. glm5next keeps its real state inside
    /// the model and uses this only to signal "new sequence" and to report the
    /// position back, which is what `Glm5NextModel::forward` documents.
    kv: KvCache,
    /// The token ids the model's state currently covers, in order.
    covered: Vec<u32>,
    /// `<think>` / `</think>`, when the tokenizer has them.
    think_start: Option<u32>,
    think_end: Option<u32>,
    eos: Vec<u32>,
    /// The tier counters as of the last report, so each one covers its own request.
    tiers: TierCounters,
    /// GLM-5.3-Flash. Prompt states kept between runs.
    ///
    /// A prompt is the expensive part of a request here -- 1740 tokens measured at
    /// 585 s, because MLA's attention is O(n) a token -- and a harness sends the
    /// same system prompt every time. Without this the daemon re-reads it on every
    /// start.
    pub disk: Option<crate::engines::disk::DiskCache<crate::llm::arch::glm5next::forward::StateSnapshot>>,
    /// The request running is incognito: keep no prompt state of it.
    forgetting: bool,
    /// The incognito session whose prompt state is held (in memory only), if any.
    private_session: Option<String>,
    pub warn: bool,
    pub log: bool,
    /// Gemma 4's vision tower, where its projector is loaded, and the soft tokens of the last pictures it read (by
    /// their bytes' hash): a conversation sends its pictures again with each turn.
    vision: Option<crate::llm::MmProj>,
    /// GLM-5.3 Flash's vision tower, where its projector is loaded.
    glm_vision: Option<Arc<GlmTower>>,
    soft: std::collections::VecDeque<(u64, Arc<Vec<f32>>)>,
}

/// Pictures' soft tokens kept for the turns after.
const SOFT_KEPT: usize = 4;

/// GLM-5.3 Flash's vision tower (`glm5next` mmproj), its weights f32 on the host.
pub struct GlmTower {
    pub shape: crate::llm::arch::glm5next::vision::VisionShape,
    pub weights: crate::llm::arch::glm5next::mmproj::VisionWeights,
}

impl GlmTower {
    /// A picture's tokens (`[n, proj_dim]`) from its pixels as `glm5next::vision::preprocess_within` laid them out.
    pub fn read(&self, pixels: &[f32], width: usize, height: usize) -> crate::llm::Result<Vec<f32>> {
        use crate::llm::arch::glm5next::vision;
        let sh = &self.shape;
        let rope = vision::MRope::for_grid(sh.d_head(), crate::llm::arch::glm5next::mmproj::ROPE_THETA, width / sh.patch, height / sh.patch, sh.n_merge)?;
        vision::encode_image(sh, &self.weights.view(), &rope, pixels, width, height)
    }
}

/// A snapshot of the expert tiers' cumulative counters, for reporting deltas.
#[derive(Default, Clone, Copy)]
struct TierCounters {
    drive_bytes: u64,
    ram_hits: u64,
    ram_misses: u64,
    vram_hits: u64,
    vram_misses: u64,
    h2d_bytes: u64,
    waits_free: u64,
    waits_pending: u64,
    cpu_records: u64,
    cpu_secs: f64,
    grouped_calls: u64,
    grouped_slots: u64,
    grouped_fell: u64,
}

impl TierCounters {
    /// This snapshot less an earlier one. Saturating, because the CPU tier's
    /// counters can be reset from elsewhere (a test does).
    fn since(&self, then: &Self) -> Self {
        Self {
            drive_bytes: self.drive_bytes.saturating_sub(then.drive_bytes),
            ram_hits: self.ram_hits.saturating_sub(then.ram_hits),
            ram_misses: self.ram_misses.saturating_sub(then.ram_misses),
            vram_hits: self.vram_hits.saturating_sub(then.vram_hits),
            vram_misses: self.vram_misses.saturating_sub(then.vram_misses),
            h2d_bytes: self.h2d_bytes.saturating_sub(then.h2d_bytes),
            waits_free: self.waits_free.saturating_sub(then.waits_free),
            waits_pending: self.waits_pending.saturating_sub(then.waits_pending),
            cpu_records: self.cpu_records.saturating_sub(then.cpu_records),
            cpu_secs: (self.cpu_secs - then.cpu_secs).max(0.0),
            grouped_calls: self.grouped_calls.saturating_sub(then.grouped_calls),
            grouped_slots: self.grouped_slots.saturating_sub(then.grouped_slots),
            grouped_fell: self.grouped_fell.saturating_sub(then.grouped_fell),
        }
    }
}

impl GlmEngine {
    /// `max_len` bounds the state the model was opened with.
    pub fn new(model: Model, tok: Arc<Tokenizer>, max_len: usize, log: bool) -> Self {
        let kv = model.new_kv_cache(max_len);
        // The model's own markers (GLM's for GLM, `<|eot_id|>` for Llama, `<end_of_turn>` for Gemma, `<|im_end|>`
        // for Qwen): `chat_stop_tokens` gives the strings; only the ids matter here, and a missing one simply means
        // that feature is off.
        let think_start = tok.token_id("<think>");
        let think_end = tok.token_id("</think>");
        let mut eos: Vec<u32> = crate::llm::chat_stop_tokens(&model.config().arch)
            .iter()
            .filter_map(|s| tok.token_id(s))
            .collect();
        if let Some(e) = tok.eos() {
            if !eos.contains(&e) {
                eos.push(e);
            }
        }
        Self {
            model,
            tok,
            kv,
            covered: Vec::new(),
            think_start,
            think_end,
            eos,
            tiers: TierCounters::default(),
            disk: None,
            forgetting: false,
            private_session: None,
            warn: log,
            log,
            vision: None,
            glm_vision: None,
            soft: Default::default(),
        }
    }

    /// With GLM-5.3 Flash's vision tower: a job's pictures are read by it.
    pub fn with_glm_vision(mut self, tower: Option<Arc<GlmTower>>) -> Self {
        self.glm_vision = tower;
        self
    }

    /// With Gemma 4's vision tower: a job's pictures are read by it.
    pub fn with_vision(mut self, vision: Option<crate::llm::MmProj>) -> Self {
        self.vision = vision;
        self
    }

    /// A model whose state is its attention cache alone, which a prompt that diverges partway can keep the shared
    /// part of (the dense families): GLM-5.3-Flash's MLA state and the MoE models' are not rewound.
    fn rewinds(&self) -> bool {
        matches!(self.model, Model::Llama(_) | Model::Qwen3(_) | Model::Gemma3(_) | Model::Gemma3n(_) | Model::Gemma4(_))
    }

    /// How many leading tokens of `prompt` the state already covers.
    ///
    /// Only a prefix counts: the state is a recurrence, so there is no way to
    /// rewind part of it. A prompt that diverges anywhere has to start over.
    fn reusable(&mut self, prompt: &[u32]) -> usize {
        let n = self
            .covered
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        // Only if the whole covered run matches: a shorter match means the state
        // holds tokens this prompt does not have.
        if n == self.covered.len() && n < prompt.len() {
            n
        } else if self.rewinds() && n > 0 {
            // a cache keeps its first rows as they are: the shared part (all but the prompt's last
            // token, which the reply's logits come from), the rest dropped. A conversation's next turn re-reads only
            // what is new since its last reply (the reply as the template writes it back may differ from as it came).
            let n = n.min(prompt.len() - 1);
            self.covered.truncate(n);
            self.kv.len = n;
            self.kv.dirty_from = self.kv.dirty_from.min(n);
            n
        } else {
            0
        }
    }

    /// The longest prompt state on disk that covers more of `prompt` than `here`,
    /// restored; the position the prompt then starts from.
    ///
    /// Only a prefix is any use: the state is a recurrence, so there is no rewinding
    /// part of it -- the same reason `reusable` insists on a whole-prefix match.
    fn restore_from_disk(&mut self, prompt: &[u32], here: usize) -> Option<usize> {
        let keys: Vec<u64> = prompt.iter().map(|&t| u64::from(t)).collect();
        let disk = self.disk.as_mut()?;
        // Never the whole prompt: a state covering all of it would leave nothing to
        // run, and the caller needs one forward pass to have logits to sample from.
        let t = std::time::Instant::now();
        let (keys, snap) = disk.load_best(&keys, prompt.len().saturating_sub(1), here, self.warn)?;
        let len = keys.len();
        let Model::Glm5Next(g) = &self.model else { return None };
        if let Err(e) = g.restore_state(&snap) {
            if self.warn {
                eprintln!("oaiy-llm-server: a prompt state on disk did not fit this model: {e}");
            }
            // The state is now of unknown length: start clean rather than guess.
            self.reset();
            return None;
        }
        self.covered = keys.iter().map(|&k| k as u32).collect();
        self.kv.len = self.covered.len();
        if self.log {
            eprintln!(
                "  prompt state for {len} tokens loaded from disk in {:.2}s",
                t.elapsed().as_secs_f64()
            );
        }
        Some(len)
    }

    /// Keep the state after the first `pos` tokens, unless it is there already.
    ///
    /// `base` marks where a conversation's first user message begins -- the system
    /// prompt and tools, the part every request shares and the one worth keeping
    /// longest.
    fn persist(&mut self, pos: usize, base: bool) {
        if self.forgetting {
            return;
        }
        let Some(disk) = self.disk.as_mut() else { return };
        let keys: Vec<u64> = self.covered[..pos].iter().map(|&t| u64::from(t)).collect();
        if disk.has(&keys) {
            return;
        }
        let Model::Glm5Next(g) = &self.model else { return };
        match g.snapshot_state() {
            Ok(snap) if snap.len == pos => disk.save(keys, snap, base),
            Ok(snap) => {
                if self.warn {
                    eprintln!(
                        "oaiy-llm-server: not keeping a prompt state for {pos} tokens; the \
                         model is at {}",
                        snap.len
                    );
                }
            }
            Err(e) => {
                if self.warn {
                    eprintln!("oaiy-llm-server: keeping the prompt state failed: {e}");
                }
            }
        }
    }

    pub fn run(mut self, jobs: Receiver<Job>) {
        for job in jobs {
            // The end of an incognito session: wipe what is held of it.
            if job.wipe {
                if self.private_session.is_some() && self.private_session == job.session { self.reset(); }
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                continue;
            }
            // A session's private state is reused by that session only.
            if self.private_session.is_some() && !(job.forget && job.session.is_some() && job.session == self.private_session) {
                self.reset();
            }
            self.forgetting = job.forget;
            let r = self.generate(&job);
            if let Err(e) = r {
                // The state's position is unknown after a failure: start clean.
                self.reset();
                let _ = job.events.send(Event::Error(e.to_string()));
            }
            if job.forget {
                // An incognito session keeps its state (in memory only) for its next request.
                if job.session.is_some() { self.private_session = job.session.clone(); } else { self.reset(); }
            }
        }
    }

    fn reset(&mut self) {
        self.private_session = None;
        if let Model::Glm5Next(g) = &self.model {
            g.reset();
        }
        self.kv.len = 0;
        self.covered.clear();
    }

    /// The soft tokens of each of a job's pictures (`[n, d]` each), by Gemma 4's vision tower, or as kept from a
    /// turn before.
    fn pictures(&mut self, job: &Job) -> Result<Vec<Arc<Vec<f32>>>> {
        let mut out = Vec::with_capacity(job.images.len());
        for (i, image) in job.images.iter().enumerate() {
            if let Some((_, t)) = self.soft.iter().find(|(h, _)| *h == image.hash) {
                out.push(Arc::clone(t));
                continue;
            }
            if let (Some(tower), crate::engines::job::ImagePrep::Glm { pixels, width, height, tokens }) = (&self.glm_vision, &image.prep) {
                let t = std::time::Instant::now();
                let soft = tower.read(pixels, *width, *height).map_err(|e| oaiy_engine::Error::Arg(format!("image {}: {e}", i + 1)))?;
                if soft.len() != tokens * self.model.config().embedding_dim {
                    return Err(oaiy_engine::Error::Arg(format!("image {}: the vision tower gave {} values, not {tokens} tokens", i + 1, soft.len())));
                }
                if self.log {
                    eprintln!("  picture {} ({width}x{height}, {tokens} tokens) read in {:.2}s", i + 1, t.elapsed().as_secs_f64());
                }
                let soft = Arc::new(soft);
                self.soft.push_back((image.hash, Arc::clone(&soft)));
                if self.soft.len() > SOFT_KEPT {
                    self.soft.pop_front();
                }
                out.push(soft);
                continue;
            }
            let (Some(crate::llm::MmProj::Gemma4V(mm)), crate::engines::job::ImagePrep::Gemma { pixels, tokens }) = (&self.vision, &image.prep) else {
                return Err(oaiy_engine::Error::Arg("this model reads text only: no vision projector is loaded for it".into()));
            };
            let t = std::time::Instant::now();
            let soft = mm.forward(pixels).map_err(|e| oaiy_engine::Error::Arg(format!("image {}: {e}", i + 1)))?;
            let soft = if soft.is_device() { soft.to_host() } else { soft };
            if soft.dim(0) != *tokens || soft.dim(1) != self.model.config().embedding_dim {
                return Err(oaiy_engine::Error::Arg(format!("image {}: the vision tower gave {:?}, not {tokens} tokens", i + 1, soft.shape())));
            }
            if self.log {
                eprintln!("  picture {} ({}x{}, {tokens} tokens) read in {:.2}s", i + 1, pixels.dim(2), pixels.dim(1), t.elapsed().as_secs_f64());
            }
            let soft = Arc::new(soft.data().to_vec());
            self.soft.push_back((image.hash, Arc::clone(&soft)));
            if self.soft.len() > SOFT_KEPT {
                self.soft.pop_front();
            }
            out.push(soft);
        }
        Ok(out)
    }

    fn generate(&mut self, job: &Job) -> Result<()> {
        if !job.images.is_empty() && self.vision.is_none() && self.glm_vision.is_none() {
            return Err(oaiy_engine::Error::Arg("this model reads text only: no vision projector is loaded for it".into()));
        }
        if job.prompt.is_empty() {
            return Err(oaiy_engine::Error::Arg("glm5next: empty prompt".into()));
        }
        let pictures = self.pictures(job)?;
        // The prompt as the state is kept by: each picture's soft tokens keyed by its bytes (a token past the
        // vocabulary's ids each), so another picture in the same place is another prompt.
        let keyed: Vec<u32> = {
            let mut k = job.prompt.clone();
            for image in &job.images {
                for (offset, id) in k[image.start..image.start + image.prep.n_tokens()].iter_mut().enumerate() {
                    *id = 0x8000_0000 | ((image.hash.rotate_left(17) ^ offset as u64) as u32 & 0x7fff_ffff);
                }
            }
            k
        };
        let prompt = &keyed;
        let tokens = &job.prompt;

        let started = std::time::Instant::now();
        let common = self.covered.iter().zip(prompt).take_while(|(a,b)| a == b).count();
        let mut start = self.reusable(prompt);
        let mut source = if start > 0 { "memory" } else { "none" };
        // Disk may hold more of this prompt than the state does -- a fresh daemon's
        // state holds none of it, and the system prompt is the same every time.
        if let Some(len) = job.images.is_empty().then(|| self.restore_from_disk(prompt, start)).flatten() {
            start = len;
            source = "disk";
        } else if start == 0 {
            self.reset();
        }
        let _ = job.events.send(Event::CacheReuse { cached: start, source, common });
        let total = prompt.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });

        // Prefill the tail in chunks, reporting after each: a long prompt is still
        // tens of seconds, and a caller with no progress cannot tell that from a
        // hang.
        //
        // GLM-5.3-Flash. A chunk, not a token. This loop used to
        // hand `forward` one token at a time so it could report between them, which
        // meant the batched prefill inside it never engaged -- `forward` only
        // chunks what it is given, and it was given one. A chunk of 64 is one
        // batched pass and one progress event, which is frequent enough to watch.
        let mut logits;
        let mut pos = start;
        let step = self.prefill_step();
        // All but the last token, then the state is kept, then the last one.
        //
        // The state is kept one token short on purpose. A restore has to leave at
        // least one token to run -- the logits a reply is sampled from come from a
        // forward pass, and a state covering the whole prompt has already consumed
        // it -- which is why `DiskCache::best` will not return an entry as long as
        // the prompt. Saved at full length, an entry could never serve the prompt it
        // came from, only a longer one that starts with it. Saved a token short it
        // serves both.
        let keep_at = prompt.len() - 1;
        while pos < keep_at {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done {
                    finish: Finish::Stop,
                    completion_tokens: 0,
                });
                return Ok(());
            }
            let take = step.min(keep_at - pos);
            self.forward_chunk(tokens, pos, take, job, &pictures)?;
            self.covered.extend_from_slice(&prompt[pos..pos + take]);
            pos += take;
            let _ = job.events.send(Event::Progress {
                done: pos - start,
                total,
            });
        }
        // `base` marks a prompt read from nothing: the system prompt every later
        // request shares, and so the entry worth protecting from eviction.
        if keep_at > start && job.images.is_empty() {
            self.persist(keep_at, start == 0);
        }
        if job.cancel.load(Ordering::Relaxed) {
            let _ = job.events.send(Event::Done {
                finish: Finish::Stop,
                completion_tokens: 0,
            });
            return Ok(());
        }
        logits = self.forward_chunk(tokens, keep_at, prompt.len() - keep_at, job, &pictures)?;
        self.covered.extend_from_slice(&prompt[keep_at..]);
        let _ = job.events.send(Event::Progress { done: total, total });
        let prefill_s = started.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });

        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        let (mut pending, mut n) = (Vec::new(), 0usize);
        let opened = self.think_start.is_some() && prompt.last().copied() == self.think_start;
        let mut budget = ThinkBudget::new(opened, self.think_end, job.think_budget);
        let max_seq = self.model.config().context_length.min(self.kv.max_len);

        let decode = std::time::Instant::now();
        let finish = loop {
            let was_thinking = budget.thinking;
            let (next, forced) = budget.pass(sample(&logits, &job.sampling, &mut rng));
            if forced && self.log {
                eprintln!(
                    "  reasoning ended at its budget ({} tokens)",
                    job.think_budget.unwrap_or(0)
                );
            }
            if was_thinking && (!budget.thinking || budget.used.is_multiple_of(THINKING_EVERY)) {
                let _ = job.events.send(Event::Thinking {
                    used: budget.used,
                    budget: budget.budget,
                    done: !budget.thinking,
                });
            }
            if self.eos.contains(&next) {
                break Finish::Stop;
            }
            n += 1;

            // Decode the run so far and send only what is new: a single token can be
            // half a UTF-8 sequence, and this tokenizer hands back strings rather
            // than bytes, so the delta of the decoded run is what is safe to emit.
            pending.push(next);
            let text = self.tok.decode(&pending);
            if !text.contains('\u{FFFD}') {
                pending.clear();
                if job.events.send(Event::Text(text)).is_err() {
                    break Finish::Stop; // nobody is listening
                }
            }

            if n >= job.max_tokens || pos + 1 >= max_seq {
                break Finish::Length;
            }
            if job.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            logits = self.forward_one(next)?;
            self.covered.push(next);
            pos += 1;
        };
        if !pending.is_empty() {
            let _ = job
                .events
                .send(Event::Text(self.tok.decode(&pending).replace('\u{FFFD}', "")));
        }
        let decode_s = decode.elapsed().as_secs_f64();
        #[cfg(feature = "webgpu")]
        if self.log && ggml_rs_wgpu::profile::chain_on() {
            let kernels = ggml_rs_wgpu::profile::take_kernels();
            eprintln!("  kernels: the request's {:.1} ms of the GPU's in {} dispatches (each timed in a pass of its own)", kernels.iter().map(|k| k.1).sum::<f64>(), kernels.iter().map(|k| k.2).sum::<u64>());
            for (name, ms, count) in kernels.iter().take(32) {
                eprintln!("    {name:<34} {ms:>9.2} ms {count:>7}");
            }
        }
        #[cfg(feature = "webgpu")]
        if self.log && std::env::var_os("OAIY_DECODE_LOG").is_some() {
            eprintln!("  {}", ggml_rs_wgpu::profile::take_line());
        }
        if self.log {
            eprintln!(
                "  {} prompt tokens ({start} reused) in {prefill_s:.1}s; {n} generated in \
                 {decode_s:.1}s ({:.1} tok/s)",
                prompt.len(),
                n as f64 / decode_s.max(1e-9)
            );
        }
        // GLM5_PROF=1 breaks the request down by forward phase, so the thing being
        // optimised is measured on the path that serves rather than in a test. The
        // counters are process-global and cover prefill and decode together, which
        // is why they are reset per request.
        if std::env::var("GLM5_PROF").ok().as_deref() == Some("1") {
            use crate::llm::arch::glm5next::forward::prof;
            let passes = (prompt.len() - start + n).max(1) as f64;
            eprintln!("  profile over {passes:.0} forward passes, ms a pass:");
            for (name, c) in prof::all() {
                eprintln!("    {name:<30} {:7.2}", prof::ms(c) / passes);
            }
            for (name, c) in prof::inner() {
                eprintln!("    {name:<30} {:7.2}", prof::ms(c) / passes);
            }
            eprintln!("    {:<30} {:7.2}", "counted total", prof::total_ms() / passes);
            prof::reset();
            self.tier_report(passes);
        }
        let _ = job.events.send(Event::Done {
            finish,
            completion_tokens: n,
        });
        Ok(())
    }

    /// Where a token's expert work went: the tiers, the drive, and how much of it
    /// the grouped dispatch took.
    ///
    /// The phase timings say the FFN costs 70 ms a token; they cannot say whether
    /// that is VRAM reads, PCIe uploads, CPU experts the GPU waited on, or the
    /// drive. These counters can, and they are the difference between optimising
    /// the right thing and optimising the thing that was easy to reach.
    ///
    /// Every counter here is cumulative and process-global, so this reports the
    /// **delta** over the request. Dividing the running totals instead reads as a
    /// per-token figure and is not one: it says 166 grouped layer-calls a pass for
    /// a model with 42 MoE layers.
    fn tier_report(&mut self, _passes: f64) {}

    fn forward_one(&mut self, id: u32) -> Result<Vec<f32>> {
        self.forward_many(&[id])
    }

    /// Several tokens in one call, so `forward` can batch them. Returns the last
    /// one's logits, which is all a prompt needs.
    /// Tokens a prompt's chunk holds. A model whose experts stream (GLM-5.3-Flash and the other MoE models) reads
    /// each layer's experts once a chunk and reports progress a chunk; a dense one reads every weight once a chunk, and
    /// on WebGPU each projection is a submit and a read back, so its chunks are bigger: a 3B Llama's 2,000-token prompt
    /// took 17.5 s in chunks of 64.
    fn prefill_step(&self) -> usize {
        match &self.model {
            Model::Llama(_) | Model::Qwen3(_) | Model::Gemma3(_) | Model::Gemma3n(_) | Model::Gemma4(_) => DENSE_PREFILL_CHUNK,
            _ => crate::llm::arch::glm5next::forward::prefill_chunk().max(1),
        }
    }

    /// `tokens[at..at + len]`, with the soft tokens of any of the job's `pictures` that fall in them in place of
    /// their embeddings: the last one's logits.
    fn forward_chunk(&mut self, tokens: &[u32], at: usize, len: usize, job: &Job, pictures: &[Arc<Vec<f32>>]) -> Result<Vec<f32>> {
        let ids = &tokens[at..at + len];
        let d = self.model.config().embedding_dim;
        let soft: Vec<(usize, &[f32])> = job
            .images
            .iter()
            .zip(pictures)
            .filter_map(|(image, rows)| {
                // the part of the picture's span within the chunk
                let (lo, hi) = (image.start.max(at), (image.start + image.prep.n_tokens()).min(at + len));
                (lo < hi).then(|| (lo - at, &rows[(lo - image.start) * d..(hi - image.start) * d]))
            })
            .collect();
        if soft.is_empty() {
            return self.forward_many(ids);
        }
        let t = self
            .model
            .forward_spliced(ids, &soft, &mut self.kv)
            .ok_or_else(|| oaiy_engine::Error::Arg("this model takes no pictures".into()))?;
        Ok(if t.device_storage().is_some() { t.to_host().data().to_vec() } else { t.data().to_vec() })
    }

    fn forward_many(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.model.forward(ids, &mut self.kv);
        // A dense model on a CUDA card leaves its logits on the card; the streamed MoE models hand them back on the host.
        if t.device_storage().is_some() {
            return Ok(t.to_host().data().to_vec());
        }
        Ok(t.data().to_vec())
    }
}
