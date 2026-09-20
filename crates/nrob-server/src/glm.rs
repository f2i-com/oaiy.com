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
    pub log: bool,
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

        // Prefill the tail, a token at a time, reporting as it goes: at ~6 tok/s a
        // long prompt is minutes, and a caller with no progress cannot tell the
        // difference between that and a hang.
        let mut logits;
        let mut pos = start;
        loop {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done {
                    finish: Finish::Stop,
                    completion_tokens: 0,
                });
                return Ok(());
            }
            logits = self.forward_one(prompt[pos])?;
            self.covered.push(prompt[pos]);
            pos += 1;
            if pos.is_multiple_of(16) || pos == prompt.len() {
                let _ = job.events.send(Event::Progress {
                    done: pos - start,
                    total,
                });
            }
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
        }
        let _ = job.events.send(Event::Done {
            finish,
            completion_tokens: n,
        });
        Ok(())
    }

    fn forward_one(&mut self, id: u32) -> Result<Vec<f32>> {
        let t = self.model.forward(&[id], &mut self.kv);
        Ok(t.data().to_vec())
    }
}
