//! The model worker: one thread owns the GPU model and runs requests one at
//! a time, reusing whatever prefix of the new prompt the model has already
//! seen.
//!
//! A coding harness resends the whole conversation every turn, and the
//! prompt is the slow part: a stretch of a few hundred tokens already
//! touches nearly all 15,360 experts, and a 192 GB machine holds only about
//! two thirds of them in RAM and VRAM, the rest on the SSD. So:
//!
//! - the worker keeps the token sequence its state covers, plus checkpoints
//!   (every 256 prompt tokens, at the end of every layered pass, and at the
//!   first and last user turn): a request that extends the last
//!   conversation continues from the live state, and one that diverges
//!   earlier (a re-rendered reply, an edited history, a new chat with the
//!   same system prompt) restores the latest checkpoint before the
//!   divergence and runs only the rest;
//! - with a prompt cache directory, states also go to disk
//!   ([`crate::disk`]): where a conversation's first user message begins,
//!   after each layered pass, where the newest user message begins, and at
//!   the end of each prompt. A prompt that disk covers more of than the
//!   worker does starts from there: a new process does not read the system
//!   prompt again, nor a conversation it resumes, and a long read stopped
//!   part-way carries on where it stopped;
//! - what is left runs token by token through the decode path when short
//!   (its experts mostly sit in VRAM and RAM already), and layer by layer
//!   when long ([`GpuModel::prefill_layered`]: every expert read once for
//!   the whole stretch).
//!
//! Images run through the vision tower when the prompt reaches them (a
//! cached prefix keeps its images' state); every image position carries the
//! same token id, so the prefix cache compares image content, not ids.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use dsv41::tokenizer::Tokenizer;
use dsv41::vision::Prepared;
use dsv41_cuda::{Checkpoint, GpuModel, ImageSpan};

#[derive(Clone, Debug)]
pub struct Sampling {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = no limit (up to [`CANDIDATES`]).
    pub top_k: usize,
    pub seed: u64,
}

/// Tokens considered when sampling (the tail beyond is negligible mass and
/// a full sort of the 129k vocabulary per token is not).
const CANDIDATES: usize = 512;

/// An image of a prompt, sized for the tower; its span starts at `start`.
pub struct JobImage {
    pub start: usize,
    pub prep: Prepared,
    /// Of the encoded bytes: the prefix cache's notion of "the same image".
    pub hash: u64,
}

pub struct Job {
    pub tool_precision: bool,
    pub prompt: Vec<u32>,
    pub images: Vec<JobImage>,
    pub max_tokens: usize,
    /// Reasoning tokens the reply may spend before it has to answer (when
    /// the prompt opens a `<think>` block); `None`: no limit.
    pub think_budget: Option<usize>,
    pub sampling: Sampling,
    /// Set by the request side to stop early (a stop string, a closed
    /// connection).
    pub cancel: Arc<AtomicBool>,
    pub events: Sender<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// End of sequence (or cancelled).
    Stop,
    /// `max_tokens` or the context limit.
    Length,
}

pub enum Event {
    /// How far the prompt has got: `done` of the `total` tokens the prefix
    /// cache did not have (sent as it goes, the first time with `done` 0).
    Progress { done: usize, total: usize },
    /// The prompt is in: `cached` of its tokens came from the prefix cache.
    Prefilled { cached: usize },
    /// How far the reply's reasoning has got: `used` tokens of its `budget`
    /// (`None`: no limit); `done` once it has closed (sent every
    /// [`THINKING_EVERY`] tokens, and at the end).
    Thinking { used: usize, budget: Option<usize>, done: bool },
    /// More reply text (whole UTF-8 characters).
    Text(String),
    Done { finish: Finish, completion_tokens: usize },
    Error(String),
}

pub struct Engine {
    pub model: GpuModel,
    pub tok: Arc<Tokenizer>,
    pub eos: u32,
    /// Attention sub-chunk of the layered prefill (tokens).
    pub chunk: usize,
    /// Prompt stretches this short run one token at a time instead.
    pub step_below: usize,
    /// Longer stretches run layer by layer, at most this many tokens a pass.
    pub layered_max: usize,
    pub max_checkpoints: usize,
    pub usage: Option<PathBuf>,
    /// Log each request's timings.
    pub log: bool,
    /// Report failures that have no request to go to (on by default).
    pub warn: bool,
    /// Prompt states kept on disk between runs.
    pub disk: Option<crate::disk::DiskCache<dsv41_cuda::Snapshot>>,
    /// What the state covers, one key per position: the token id, or for
    /// image positions a key from the image's content and the offset.
    tokens: Vec<u64>,
    pub tool_experts: bool,
    pub repetition_guard: bool,
    request_number: u64,
    checkpoints: Vec<Checkpoint>,
}

/// Holds a reply's reasoning to its budget: counts the tokens sampled
/// inside the `<think>` block the prompt opens and, once they reach the
/// budget, puts `</think>` in place of the next one, so a model going round
/// in circles (re-deriving what it has already checked) stops and acts.
struct ThinkBudget {
    end: u32,
    /// Still inside the `<think>` block.
    thinking: bool,
    /// Reasoning tokens so far.
    used: usize,
    budget: Option<usize>,
}

/// Reasoning progress goes to the client every this many tokens.
const THINKING_EVERY: usize = 16;

impl ThinkBudget {
    fn new(opens_thinking: bool, end: Option<u32>, budget: Option<usize>) -> ThinkBudget {
        match end {
            Some(end) if opens_thinking => ThinkBudget { end, thinking: true, used: 0, budget },
            _ => ThinkBudget { end: 0, thinking: false, used: 0, budget: None },
        }
    }

    /// The token to take in place of `sampled`, and whether the budget
    /// forced it.
    fn pass(&mut self, sampled: u32) -> (u32, bool) {
        if !self.thinking {
            return (sampled, false);
        }
        if sampled == self.end {
            self.thinking = false;
            return (sampled, false);
        }
        if self.budget.is_some_and(|b| self.used >= b) {
            self.thinking = false;
            return (self.end, true);
        }
        self.used += 1;
        (sampled, false)
    }
}

/// How much of a `left`-token stretch the next layered pass takes: all of
/// it, or an equal share when it needs more than one pass of at most
/// `max` tokens.
fn pass_len(left: usize, max: usize) -> usize {
    left.div_ceil(left.div_ceil(max.max(1)))
}

/// Prefix-cache keys of a prompt: token ids, image positions keyed by
/// their image (ids stay below 2^32, image keys above).
fn prompt_keys(prompt: &[u32], images: &[JobImage]) -> Vec<u64> {
    let mut keys: Vec<u64> = prompt.iter().map(|&t| u64::from(t)).collect();
    for img in images {
        for (k, key) in keys[img.start..img.start + img.prep.n_tokens()].iter_mut().enumerate() {
            *key = (img.hash.rotate_left(17) ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1 << 63;
        }
    }
    keys
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(model: GpuModel, tok: Arc<Tokenizer>, chunk: usize, step_below: usize, layered_max: usize, max_checkpoints: usize, usage: Option<PathBuf>, log: bool) -> Engine {
        let eos = model.cfg.eos_token_id;
        Engine {
            model,
            tok,
            eos,
            chunk: chunk.max(2),
            step_below,
            layered_max: layered_max.max(2),
            max_checkpoints,
            usage,
            log,
            warn: true,
            disk: None,
            tokens: Vec::new(),
            tool_experts: false, repetition_guard: true,
            request_number: 0,
            checkpoints: Vec::new(),
        }
    }

    fn switch_phase(&mut self, original: bool) -> nrob::Result<()> {
        if !self.tool_experts { return Ok(()); }
        let t = Instant::now();
        if self.model.select_tool_experts(original)? && self.log {
            eprintln!("expert-only phase switch: {} in {:.3}s; trunk and both expert caches retained",
                if original {"mxfp4"} else {"w2g128"}, t.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Serve jobs until every sender is gone.
    pub fn run(mut self, jobs: Receiver<Job>) {
        for job in jobs {
            // the background cache fill waits for idle time
            self.model.set_busy(true);
            self.request_number += 1;
            let result = self.switch_phase(false).and_then(|_| self.generate(&job));
            // Every saved prompt checkpoint used ternary. Never reuse the generated
            // mixed-precision suffix as if it had been evaluated in ternary.
            let cleanup = if self.tool_experts {
                self.switch_phase(false).and_then(|_| self.rollback(job.prompt.len()))
            } else { Ok(()) };
            let result = result.and(cleanup).and(self.model.flush_route_log());
            self.model.set_busy(false);
            if let Err(e) = result {
                // the state is unknown after a failure: start clean
                self.tokens.clear();
                self.checkpoints.clear();
                let _ = job.events.send(Event::Error(e.to_string()));
            }
            if let Some(path) = &self.usage {
                if let (Err(e), true) = (self.model.save_usage(path), self.warn) {
                    eprintln!("nrob-server: saving the expert usage profile failed: {e}");
                }
            }
        }
    }

    /// Where the prompt can start: the live state if it is a prefix of the
    /// prompt, else the latest checkpoint inside the common prefix, else 0.
    /// At least one prompt token is always left to run (its logits start
    /// the reply).
    fn resume(&mut self, prompt: &[u64]) -> nrob::Result<usize> {
        let common = self.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let limit = common.min(prompt.len().saturating_sub(1));
        // what this process has (the live state, or a checkpoint), unless
        // disk has more
        let here = if self.tokens.len() <= limit {
            self.tokens.len()
        } else {
            self.checkpoints.iter().rev().find(|c| c.pos() <= limit).map_or(0, |c| c.pos())
        };
        if let Some(pos) = self.restore_from_disk(prompt, here)? {
            return Ok(pos);
        }
        if self.tokens.len() <= limit {
            return Ok(self.tokens.len());
        }
        let at = self.checkpoints.iter().rposition(|c| c.pos() <= limit);
        match at {
            Some(i) => {
                let pos = self.checkpoints[i].pos();
                self.model.restore(&self.checkpoints[i])?;
                // later checkpoints belong to the old continuation
                self.checkpoints.truncate(i + 1);
                self.tokens.truncate(pos);
                Ok(pos)
            }
            None => {
                self.checkpoints.clear();
                self.tokens.clear();
                Ok(0)
            }
        }
    }

    /// Load the longest prompt state on disk that covers more of `prompt`
    /// than `here`: where the prompt then starts.
    fn restore_from_disk(&mut self, prompt: &[u64], here: usize) -> nrob::Result<Option<usize>> {
        let Some(disk) = self.disk.as_mut() else {
            return Ok(None);
        };
        let Some((i, len)) = disk.best(prompt, prompt.len().saturating_sub(1)).filter(|&(_, len)| len > here) else {
            return Ok(None);
        };
        let t = Instant::now();
        let (keys, snap) = match disk.load(i) {
            Ok(got) => got,
            Err(e) => {
                if self.warn {
                    eprintln!("nrob-server: a prompt state on disk could not be read: {e}");
                }
                return Ok(None);
            }
        };
        let ids: Vec<u32> = keys.iter().map(|&k| k as u32).collect();
        self.model.restore_snapshot(&snap, &ids)?;
        self.tokens = keys;
        self.checkpoints = vec![self.model.checkpoint(len)?];
        if self.log {
            eprintln!("  prompt state for {len} tokens loaded from disk in {:.2}s", t.elapsed().as_secs_f64());
        }
        Ok(Some(len))
    }

    /// Keep the state after the first `pos` tokens on disk (`base`: where a
    /// conversation's first user message begins), unless it is there
    /// already or the prompt has images.
    fn persist(&mut self, pos: usize, base: bool) {
        let Some(disk) = self.disk.as_mut() else {
            return;
        };
        let keys = &self.tokens[..pos];
        if keys.iter().any(|k| k >> 63 == 1) || disk.has(keys) {
            return;
        }
        match self.model.snapshot(pos) {
            Ok(snap) => disk.save(keys.to_vec(), snap, base),
            Err(e) => {
                if self.warn {
                    eprintln!("nrob-server: keeping the prompt state failed: {e}");
                }
            }
        }
    }

    /// The state past `pos` is half-written: restore the latest checkpoint at
    /// or before it (or forget everything when there is none).
    fn rollback(&mut self, pos: usize) -> nrob::Result<()> {
        match self.checkpoints.iter().rposition(|c| c.pos() <= pos) {
            Some(i) => {
                let at = self.checkpoints[i].pos();
                self.model.restore(&self.checkpoints[i])?;
                self.checkpoints.truncate(i + 1);
                self.tokens.truncate(at);
            }
            None => {
                self.checkpoints.clear();
                self.tokens.clear();
            }
        }
        Ok(())
    }

    fn save_checkpoint(&mut self, pos: usize) -> nrob::Result<()> {
        if self.max_checkpoints == 0 || self.checkpoints.last().is_some_and(|c| c.pos() >= pos) {
            return Ok(());
        }
        self.checkpoints.push(self.model.checkpoint(pos)?);
        if self.checkpoints.len() > self.max_checkpoints {
            // thin out: drop the checkpoint closest to its predecessor, so
            // the rest stay spread over the sequence (the latest is kept)
            let n = self.checkpoints.len();
            let gap = |i: usize| self.checkpoints[i].pos() - if i == 0 { 0 } else { self.checkpoints[i - 1].pos() };
            let drop = (0..n - 1).min_by_key(|&i| gap(i)).unwrap_or(0);
            self.checkpoints.remove(drop);
        }
        Ok(())
    }

    fn generate(&mut self, job: &Job) -> nrob::Result<()> {
        let prompt = &job.prompt;
        if prompt.is_empty() {
            return Err(nrob::Error::Arg("empty prompt".into()));
        }
        self.model.trace_phase(self.request_number, "prompt", None, "");
        let started = Instant::now();
        let keys = prompt_keys(prompt, &job.images);
        let start = self.resume(&keys)?;
        // the images the uncached part reaches, through the vision tower
        let mut spans = Vec::new();
        for img in job.images.iter().filter(|i| i.start + i.prep.n_tokens() > start) {
            let t = Instant::now();
            spans.push(ImageSpan { start: img.start, rows: self.model.encode_image(&img.prep)? });
            if self.log {
                let (h, w) = (img.prep.n_vit_h * 14, img.prep.n_vit_w * 14);
                eprintln!("  image at {} ({w}x{h}, {} tokens) encoded in {:.2}s", img.start, img.prep.n_tokens(), t.elapsed().as_secs_f64());
            }
        }
        // stretches end at the first and the last user turn too, so a
        // checkpoint lands where the next request is likely to diverge (a new
        // chat with the same system prompt; the latest turn re-rendered)
        let user = self.tok.special("<｜User｜>");
        let turns: Vec<usize> = (start + 1..prompt.len()).filter(|&p| Some(prompt[p]) == user).collect();
        // where the conversation's first user message begins (the system
        // prompt and tools before it), if this prompt still has to run it
        let first_turn = prompt.iter().position(|&t| Some(t) == user).filter(|&p| p > start);
        // where the newest user message begins: the conversation before it
        let last_turn = turns.last().copied();
        let mut ends: Vec<usize> = [turns.first(), turns.last()].into_iter().flatten().copied().collect();
        ends.push(prompt.len());
        ends.dedup();
        let mut pos = start;
        let mut logits = Vec::new();
        let total = prompt.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });
        let mut reported = Instant::now();
        for stop in ends {
            while pos < stop {
                if job.cancel.load(Ordering::Relaxed) {
                    let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                    return Ok(());
                }
                let t = Instant::now();
                let left = stop - pos;
                let (end, how) = if left <= self.step_below {
                    // a short stretch: one token at a time through the decode
                    // path, whose experts mostly sit in VRAM and RAM
                    let end = pos + left.min(256);
                    for p in pos..end {
                        if job.cancel.load(Ordering::Relaxed) {
                            // the state is whole up to p: keep it
                            let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                            return Ok(());
                        }
                        // only the prompt's last token needs logits
                        if p + 1 == prompt.len() {
                            logits = self.model.forward_with(&prompt[p..p + 1], p, &spans)?;
                        } else {
                            self.model.advance_with(&prompt[p..p + 1], p, &spans)?;
                        }
                        self.tokens.push(keys[p]);
                        if reported.elapsed().as_secs_f64() >= 0.5 {
                            let _ = job.events.send(Event::Progress { done: p + 1 - start, total });
                            reported = Instant::now();
                        }
                    }
                    (end, "token by token")
                } else {
                    // a long one layer by layer: every expert is read once for
                    // the whole stretch (a chunk of a few hundred tokens
                    // already touches nearly all of them); one longer than a
                    // pass splits into equal passes (not full ones and a short
                    // rest, which would go token by token)
                    let end = pos + pass_len(left, self.layered_max);
                    // a clean state to come back to if the pass fails
                    self.save_checkpoint(pos)?;
                    // every layer takes the whole stretch a step further
                    let (events, before, len) = (job.events.clone(), pos - start, end - pos);
                    self.model.set_layer_progress(Some(Box::new(move |layers_done, layers| {
                        let _ = events.send(Event::Progress { done: before + len * layers_done / layers, total });
                    })));
                    let pass = self.model.prefill_layered_with(&prompt[pos..end], pos, self.chunk, Some(&job.cancel), &spans);
                    self.model.set_layer_progress(None);
                    match pass {
                        Ok(l) => logits = l,
                        Err(_) if job.cancel.load(Ordering::Relaxed) => {
                            // abandoned part-way: back to the last clean state
                            self.rollback(pos)?;
                            let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                            return Ok(());
                        }
                        // a pass too long for the VRAM left for activations:
                        // back to its start, and shorter passes from now on
                        Err(e) if e.to_string().contains("OUT_OF_MEMORY") && (end - pos) / 2 > self.step_below => {
                            self.rollback(pos)?;
                            self.layered_max = (end - pos) / 2;
                            if self.warn {
                                eprintln!("nrob-server: a {}-token layered pass ran out of VRAM; passes of up to {} tokens from now on", end - pos, self.layered_max);
                            }
                            continue;
                        }
                        Err(e) => return Err(e),
                    }
                    self.tokens.extend_from_slice(&keys[pos..end]);
                    (end, "layered")
                };
                if self.log {
                    let s = t.elapsed().as_secs_f64();
                    eprintln!("  prompt {pos}..{end} {how}: {s:.1}s ({:.1} tok/s)", (end - pos) as f64 / s);
                }
                pos = end;
                self.save_checkpoint(pos)?;
                if Some(pos) == first_turn {
                    // the system prompt and tools: where the next process's
                    // chats start
                    self.persist(pos, true);
                } else if how == "layered" || Some(pos) == last_turn {
                    // a long read keeps what it has read so far (a restart,
                    // or a request stopped part-way, carries on from here),
                    // and the conversation up to its newest message is what
                    // a resumed session starts from (read-aheads included)
                    self.persist(pos, false);
                }
            }
        }
        // (a one-token request reads a prompt ahead of its turn: the end of
        // that prompt is not worth keeping)
        if job.max_tokens > 1 && prompt.len() > start {
            self.persist(prompt.len(), false);
        }
        let prefill_s = started.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });

        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        let (mut pending, mut n) = (Vec::new(), 0usize);
        let think_start = self.tok.special(dsv41::chat::THINK_START);
        let mut budget = ThinkBudget::new(think_start.is_some() && prompt.last().copied() == think_start, self.tok.special(dsv41::chat::THINK_END), job.think_budget);
        let decode = Instant::now();
        let mut phase = crate::tool_phase::ToolPhase::default();
        let mut repetition = crate::repetition::Guard::default();
        let phase_enabled = self.tool_experts && job.tool_precision;
        let finish = loop {
            let was_thinking = budget.thinking;
            let (next, forced) = budget.pass(sample(&logits, &job.sampling, &mut rng));
            if forced && self.log {
                eprintln!("  reasoning ended at its budget ({} tokens)", job.think_budget.unwrap_or(0));
            }
            if was_thinking && (!budget.thinking || budget.used.is_multiple_of(THINKING_EVERY)) {
                let _ = job.events.send(Event::Thinking { used: budget.used, budget: budget.budget, done: !budget.thinking });
            }
            if next == self.eos {
                break Finish::Stop;
            }
            n += 1;
            pending.extend_from_slice(self.tok.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                phase.push(&text); // also identifies payloads exempt from the prose repetition guard
                if job.events.send(Event::Text(text)).is_err() {
                    break Finish::Stop; // nobody is listening
                }
            }
            if self.repetition_guard && repetition.push(next, phase.original()) {
                return Err(nrob::Error::Format(format!("{}: stopped repeated prose/reasoning blocks; generation did not complete. No tool from this incomplete reply was executed. Start a fresh turn or select the original model; automatic retry is disabled.", crate::repetition::CODE)));
            }
            if n >= job.max_tokens || pos + 1 >= self.model.max_seq() {
                break Finish::Length;
            }
            if job.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            if phase_enabled { self.switch_phase(phase.original())?; }
            self.model.trace_phase(self.request_number,
                if phase_enabled && phase.original() {"tool_call"} else {"reply"},
                Some(next), &String::from_utf8_lossy(self.tok.token_bytes(next)));
            logits = self.model.forward(&[next], pos)?;
            self.tokens.push(u64::from(next));
            pos += 1;
        };
        if !pending.is_empty() {
            let _ = job.events.send(Event::Text(String::from_utf8_lossy(&pending).into_owned()));
        }
        let decode_s = decode.elapsed().as_secs_f64();
        if self.log {
            eprintln!(
                "  {} prompt tokens ({start} cached) in {prefill_s:.1}s; {n} generated in {decode_s:.1}s ({:.1} tok/s)",
                prompt.len(),
                n as f64 / decode_s.max(1e-9)
            );
        }
        let _ = job.events.send(Event::Done { finish, completion_tokens: n });
        Ok(())
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pick the next token: greedy at temperature 0, else temperature, top-k
/// and top-p (nucleus) sampling over the leading candidates.
pub fn sample(logits: &[f32], s: &Sampling, rng: &mut u64) -> u32 {
    let greedy = || {
        logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
    };
    if s.temperature <= 0.0 || logits.is_empty() {
        return greedy();
    }
    let k = if s.top_k > 0 { s.top_k.min(CANDIDATES) } else { CANDIDATES }.min(logits.len());
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let desc = |a: &u32, b: &u32| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap_or(std::cmp::Ordering::Equal);
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, desc);
        idx.truncate(k);
    }
    idx.sort_unstable_by(desc);
    let top = logits[idx[0] as usize];
    let mut probs: Vec<f64> = idx.iter().map(|&i| (((logits[i as usize] - top) / s.temperature) as f64).exp()).collect();
    let total: f64 = probs.iter().sum();
    let mut keep = probs.len();
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p / total;
            if acc >= s.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
    }
    probs.truncate(keep);
    let total: f64 = probs.iter().sum();
    let mut r = (splitmix(rng) >> 11) as f64 / (1u64 << 53) as f64 * total;
    for (i, p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[keep - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_stretch_splits_into_equal_layered_passes() {
        assert_eq!(pass_len(5_000, 8_192), 5_000);
        assert_eq!(pass_len(8_192, 8_192), 8_192);
        // 16,844 tokens: three passes of ~5,615, not 8,192 + 8,192 + 460
        // (whose rest would run token by token)
        assert_eq!(pass_len(16_844, 8_192), 5_615);
        let mut left = 16_844;
        let mut passes = Vec::new();
        while left > 0 {
            let n = pass_len(left, 8_192);
            passes.push(n);
            left -= n;
        }
        assert_eq!(passes, [5_615, 5_615, 5_614]);
    }

    #[test]
    fn reasoning_is_closed_at_its_budget() {
        const END: u32 = 9;
        let mut b = ThinkBudget::new(true, Some(END), Some(2));
        let got: Vec<(u32, bool)> = [1, 2, 3, 4].iter().map(|&t| b.pass(t)).collect();
        assert_eq!(got, [(1, false), (2, false), (END, true), (4, false)]);
        // a reply that closes its reasoning itself is left alone
        let mut b = ThinkBudget::new(true, Some(END), Some(2));
        assert_eq!([1, END, 3, 4].map(|t| b.pass(t).0), [1, END, 3, 4]);
        // no budget, or a prompt that does not open reasoning: nothing forced
        let mut b = ThinkBudget::new(true, Some(END), None);
        assert_eq!([1, 2, 3].map(|t| b.pass(t).0), [1, 2, 3]);
        let mut b = ThinkBudget::new(false, Some(END), Some(0));
        assert_eq!([1, 2].map(|t| b.pass(t).0), [1, 2]);
    }

    #[test]
    fn sampling_respects_temperature_top_k_and_top_p() {
        let logits = vec![0.0, 5.0, 4.0, -1.0, 4.9];
        let mut rng = 1;
        let greedy = Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0 };
        assert_eq!(sample(&logits, &greedy, &mut rng), 1);
        let top1 = Sampling { temperature: 1.0, top_p: 1.0, top_k: 1, seed: 0 };
        for _ in 0..50 {
            assert_eq!(sample(&logits, &top1, &mut rng), 1);
        }
        // top-p 0.5 keeps the best two (5.0, 4.9: ~0.47 then ~0.9 of the mass)
        let nucleus = Sampling { temperature: 1.0, top_p: 0.5, top_k: 0, seed: 0 };
        let mut seen = [0usize; 5];
        for _ in 0..2000 {
            seen[sample(&logits, &nucleus, &mut rng) as usize] += 1;
        }
        assert!(seen[1] > 0 && seen[4] > 0 && seen[0] + seen[2] + seen[3] == 0, "{seen:?}");
    }
}
