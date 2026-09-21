//! VENDORED-LOCAL: GLM-5.3-Flash served through the same job/event contract as
//! the DeepSeek engine.
//!
//! [`engine::Engine`](crate::engine::Engine) is built around `dsv41-cuda`'s
//! `GpuModel`: safetensors on disk, Engram tables, layered prefill, checkpoints,
//! image spans. None of that applies to a GGUF opened through `llama_rs::Model`,
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

use llama_rs::{KvCache, Model};
use nrob::Result;
use tokenizer::Tokenizer;

use crate::engine::{sample, Event, Finish, Job};

/// A reply's reasoning budget, as the DeepSeek engine does it: count the tokens
/// spent inside the `<think>` block the prompt opened and, at the budget, put the
/// closing tag in place of the next one so a model going in circles stops and
/// answers.
struct ThinkBudget {
    thinking: bool,
    end: Option<u32>,
    used: usize,
    budget: Option<usize>,
}

impl ThinkBudget {
    fn new(thinking: bool, end: Option<u32>, budget: Option<usize>) -> Self {
        Self { thinking, end, used: 0, budget }
    }

    /// `(token, forced)` — `forced` when the budget replaced the sampled token.
    fn pass(&mut self, next: u32) -> (u32, bool) {
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
const THINKING_EVERY: usize = 32;

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
    pub log: bool,
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
        // GLM's markers. `chat_stop_tokens` gives the strings; only the ids matter
        // here, and a missing one simply means that feature is off.
        let think_start = tok.token_id("<think>");
        let think_end = tok.token_id("</think>");
        let mut eos: Vec<u32> = llama_rs::chat_stop_tokens(&llama_rs::Architecture::Glm5Next)
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
            log,
        }
    }

    /// How many leading tokens of `prompt` the state already covers.
    ///
    /// Only a prefix counts: the state is a recurrence, so there is no way to
    /// rewind part of it. A prompt that diverges anywhere has to start over.
    fn reusable(&self, prompt: &[u32]) -> usize {
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
        } else {
            0
        }
    }

    pub fn run(mut self, jobs: Receiver<Job>) {
        for job in jobs {
            let r = self.generate(&job);
            if let Err(e) = r {
                // The state's position is unknown after a failure: start clean.
                self.reset();
                let _ = job.events.send(Event::Error(e.to_string()));
            }
        }
    }

    fn reset(&mut self) {
        if let Model::Glm5Next(g) = &self.model {
            g.reset();
        }
        self.kv.len = 0;
        self.covered.clear();
    }

    fn generate(&mut self, job: &Job) -> Result<()> {
        if !job.images.is_empty() {
            return Err(nrob::Error::Arg(
                "glm5next: this server does not take images yet; the vision tower is \
                 implemented in llama-rs but not wired through the generic forward"
                    .into(),
            ));
        }
        let prompt = &job.prompt;
        if prompt.is_empty() {
            return Err(nrob::Error::Arg("glm5next: empty prompt".into()));
        }

        let started = std::time::Instant::now();
        let start = self.reusable(prompt);
        if start == 0 {
            self.reset();
        }
        let total = prompt.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });

        // Prefill the tail in chunks, reporting after each: a long prompt is still
        // tens of seconds, and a caller with no progress cannot tell that from a
        // hang.
        //
        // VENDORED-LOCAL: GLM-5.3-Flash. A chunk, not a token. This loop used to
        // hand `forward` one token at a time so it could report between them, which
        // meant the batched prefill inside it never engaged -- `forward` only
        // chunks what it is given, and it was given one. A chunk of 64 is one
        // batched pass and one progress event, which is frequent enough to watch.
        let mut logits;
        let mut pos = start;
        let step = llama_rs::glm5next::forward::prefill_chunk().max(1);
        loop {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done {
                    finish: Finish::Stop,
                    completion_tokens: 0,
                });
                return Ok(());
            }
            let take = step.min(prompt.len() - pos);
            logits = self.forward_many(&prompt[pos..pos + take])?;
            self.covered.extend_from_slice(&prompt[pos..pos + take]);
            pos += take;
            let _ = job.events.send(Event::Progress {
                done: pos - start,
                total,
            });
            if pos == prompt.len() {
                break;
            }
        }
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
            use llama_rs::glm5next::forward::prof;
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
            // Host<->device round trips. A D2H of something the GPU just wrote is a
            // synchronisation, and the cards measured 2-18% busy with their memory
            // controllers at 0-6% -- starved, not slow. This says by how much.
            let (dh, dhb, hd, hdb) = ggml_rs_cuda::xfer::get();
            eprintln!(
                "    {:<30} {:7.1} D2H ({:.2} MB), {:.1} H2D ({:.2} MB)",
                "round trips a pass",
                dh as f64 / passes,
                dhb as f64 / passes / 1e6,
                hd as f64 / passes,
                hdb as f64 / passes / 1e6,
            );
            ggml_rs_cuda::xfer::reset();
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
    fn tier_report(&mut self, passes: f64) {
        let llama_rs::Model::Glm5Next(g) = &self.model else { return };
        let Some((cache, shards, cpu, grouped)) = g.tier_stats() else { return };
        let now = TierCounters {
            drive_bytes: cache.bytes_read,
            ram_hits: cache.hits,
            ram_misses: cache.misses,
            vram_hits: shards.iter().map(|s| s.hits).sum(),
            vram_misses: shards.iter().map(|s| s.misses).sum(),
            h2d_bytes: shards.iter().map(|s| s.h2d_bytes).sum(),
            waits_free: shards.iter().map(|s| s.waits_free).sum(),
            waits_pending: shards.iter().map(|s| s.waits_pending).sum(),
            cpu_records: cpu.0,
            cpu_secs: cpu.2,
            grouped_calls: grouped.0,
            grouped_slots: grouped.1,
            grouped_fell: grouped.2,
        };
        let d = now.since(&self.tiers);
        self.tiers = now;

        let look = (d.vram_hits + d.vram_misses).max(1);
        eprintln!(
            "  VRAM tier: {:.1}% of {:.1} lookups a pass hit, {:.1} MB uploaded a pass",
            100.0 * d.vram_hits as f64 / look as f64,
            (d.vram_hits + d.vram_misses) as f64 / passes,
            d.h2d_bytes as f64 / passes / 1e6,
        );
        // Whether the async H2D actually hid: a `waits_free` is a transfer that
        // had finished before its expert was needed, a `waits_pending` one the
        // compute stream had to be ordered behind. The second kind is the only
        // way the upload bytes reach the clock.
        let waits = (d.waits_free + d.waits_pending).max(1);
        eprintln!(
            "  uploads: {:.1}% had landed before they were needed ({:.1} of {:.1} a pass stalled)",
            100.0 * d.waits_free as f64 / waits as f64,
            d.waits_pending as f64 / passes,
            (d.waits_free + d.waits_pending) as f64 / passes,
        );
        eprintln!(
            "  CPU tier: {:.1} records a pass, {:.1} ms a pass",
            d.cpu_records as f64 / passes,
            d.cpu_secs * 1e3 / passes,
        );
        eprintln!(
            "  grouped: {:.1} of {:.1} layer-calls a pass, {:.1} slots each",
            d.grouped_calls as f64 / passes,
            (d.grouped_calls + d.grouped_fell) as f64 / passes,
            d.grouped_slots as f64 / d.grouped_calls.max(1) as f64,
        );
        // The drive. `bytes_read` is what the RAM cache had to go and get; on a warm
        // request it does not move at all, which is the point of a 140 GB cache.
        let ram_look = (d.ram_hits + d.ram_misses).max(1);
        eprintln!(
            "  drive: {:.1} MB over the request ({:.1} MB a pass, {:.1}% of {} RAM lookups missed)",
            d.drive_bytes as f64 / 1e6,
            d.drive_bytes as f64 / passes / 1e6,
            100.0 * d.ram_misses as f64 / ram_look as f64,
            d.ram_hits + d.ram_misses,
        );
    }

    fn forward_one(&mut self, id: u32) -> Result<Vec<f32>> {
        self.forward_many(&[id])
    }

    /// Several tokens in one call, so `forward` can batch them. Returns the last
    /// one's logits, which is all a prompt needs.
    fn forward_many(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.model.forward(ids, &mut self.kv);
        Ok(t.data().to_vec())
    }
}
