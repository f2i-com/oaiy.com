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
    pub reasoning_repeat_penalty: f32,
    pub reasoning_repeat_last_n: usize,
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
    pub prep: ImagePrep,
    /// Of the encoded bytes: the prefix cache's notion of "the same image".
    pub hash: u64,
}

pub enum ImagePrep {
    Deepseek(Prepared),
    Qwen { pixels: ggml_rs::Tensor, side: usize },
}
impl ImagePrep {
    pub fn n_tokens(&self) -> usize {
        match self { Self::Deepseek(p) => p.n_tokens(), Self::Qwen { side, .. } => side * side }
    }
    fn deepseek(&self) -> nrob::Result<&Prepared> {
        match self { Self::Deepseek(p) => Ok(p), _ => Err(nrob::Error::Arg("Qwen image sent to DeepSeek worker".into())) }
    }
}

pub struct Job {
    pub tools: Vec<nrob::json::Json>,
    pub tool_precision: bool,
    pub observer_context: String,
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
    /// Sent before processing the uncached tail; no prompt text is exposed.
    CacheReuse { cached: usize, source: &'static str, common: usize },
    /// How far the reply's reasoning has got: `used` tokens of its `budget`
    /// (`None`: no limit); `done` once it has closed (sent every
    /// [`THINKING_EVERY`] tokens, and at the end).
    Thinking { used: usize, budget: Option<usize>, done: bool },
    /// More reply text (whole UTF-8 characters).
    Text(String),
    Observer(String),
    ReasoningPreview(String),
    ContentPreview(String),
    ObserverDelta { text: String, thinking: bool, start: bool },
    ToolPreview { text: String, start: bool },
    ObserverProgress { action: String, comment: String },
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
    pub observer: Option<crate::observer::Observer>,
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
    repetition: Option<crate::repetition::ShortReasoningGuard>,
}

/// Reasoning progress goes to the client every this many tokens.
const THINKING_EVERY: usize = 16;

impl ThinkBudget {
    fn new(opens_thinking: bool, end: Option<u32>, budget: Option<usize>) -> ThinkBudget {
        match end {
            Some(end) => ThinkBudget { end, thinking: opens_thinking, used: 0, budget, repetition: None },
            _ => ThinkBudget { end: 0, thinking: false, used: 0, budget: None, repetition: None },
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
        if self.budget.is_some_and(|b| self.used >= b)
            || self.repetition.as_mut().is_some_and(|guard| guard.push(sampled))
        {
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
            tool_experts: false, repetition_guard: true, observer: None,
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
        for mut job in jobs {
            // the background cache fill waits for idle time
            self.model.set_busy(true);
            self.request_number += 1;
            let result = self.switch_phase(false).and_then(|_| {
                if job.tool_precision && blocking_observer_review(&job.observer_context, self.model.uses_ternary_experts()) {
                    if self.observer.is_none() {
                        return Err(nrob::Error::Arg("observer_review_failed: blocking review requested but no observer is loaded; no tool executed".into()));
                    }
                    self.reviewed_generate(&mut job)
                }
                else { self.generate(&job, &mut Vec::new(), 0, false, None, "") }
            });
            // Every saved prompt checkpoint used ternary. Never reuse the generated
            // mixed-precision suffix as if it had been evaluated in ternary.
            let cleanup = if self.tool_experts {
                self.switch_phase(false).and_then(|_| self.rollback(job.prompt.len()))
            } else { Ok(()) };
            let result = result.and(cleanup).and(self.model.flush_route_log());
            self.model.set_busy(false);
            if let Err(e) = result {
                if self.warn { eprintln!("nrob-server: request {} failed: {e}", self.request_number); }
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
    fn resume(&mut self, prompt: &[u64]) -> nrob::Result<(usize, &'static str, usize)> {
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
            return Ok((pos, "disk", common));
        }
        if self.tokens.len() <= limit {
            return Ok((self.tokens.len(), if self.tokens.is_empty() { "none" } else { "memory" }, common));
        }
        let at = self.checkpoints.iter().rposition(|c| c.pos() <= limit);
        match at {
            Some(i) => {
                let pos = self.checkpoints[i].pos();
                self.model.restore(&self.checkpoints[i])?;
                // later checkpoints belong to the old continuation
                self.checkpoints.truncate(i + 1);
                self.tokens.truncate(pos);
                Ok((pos, "checkpoint", common))
            }
            None => {
                self.checkpoints.clear();
                self.tokens.clear();
                Ok((0, "none", common))
            }
        }
    }

    /// Load the longest prompt state on disk that covers more of `prompt`
    /// than `here`: where the prompt then starts.
    fn restore_from_disk(&mut self, prompt: &[u64], here: usize) -> nrob::Result<Option<usize>> {
        let Some(disk) = self.disk.as_mut() else {
            return Ok(None);
        };
        let t = Instant::now();
        let Some((keys, snap)) = disk.load_best(prompt, prompt.len().saturating_sub(1), here, self.warn) else {
            return Ok(None);
        };
        let len = keys.len();
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


    /// Capture unapproved output while continuing to forward progress/comments.
    /// Display previews stream immediately; executable/accepted events stay buffered.
    fn capture(&mut self, job: &mut Job, q4: usize, comments: bool, fresh: Option<&std::collections::BTreeSet<String>>, tool_prefix: &str) -> nrob::Result<(Vec<u32>, Vec<Event>)> {
        let (tx, rx) = std::sync::mpsc::channel();
        let destination = std::mem::replace(&mut job.events, tx);
        let forward = destination.clone();
        let cancel = Arc::clone(&job.cancel);
        let preview_mode = if job.prompt.last().copied() == self.tok.special(dsv41::chat::THINK_START) { dsv41::chat::Mode::Thinking } else { dsv41::chat::Mode::Chat };
        let preview_prefix = tool_prefix.to_owned();
        let relay = std::thread::spawn(move || {
            hold_observer_events(rx, forward, cancel, Some(preview_mode), &preview_prefix)
        });
        let mut ids = Vec::new();
        let result = self.generate(job, &mut ids, q4, comments, fresh, tool_prefix);
        drop(std::mem::replace(&mut job.events, destination));
        let (kept, size) = relay.join().map_err(|_| nrob::Error::Format("observer event relay failed".into()))?;
        result?;
        if size > 1024 * 1024 { return Err(nrob::Error::Format("observer draft exceeded 1 MiB; withheld".into())); }
        if job.cancel.load(Ordering::Relaxed) { return Err(nrob::Error::Arg("observer draft cancelled; withheld".into())); }
        Ok((ids, kept))
    }

    /// Refresh guidance before the unexecuted envelope, preserving exact generated
    /// IDs through the selected header. Resume arguments, never regenerate the batch.
    fn capture_with_tool_context(&mut self, job: &mut Job) -> nrob::Result<(Vec<u32>, Vec<Event>)> {
        let context = nrob::json::Json::parse(job.observer_context.as_bytes()).ok();
        if context.as_ref().and_then(|c| c.get("tool_context")).and_then(nrob::json::Json::as_bool) != Some(true) {
            return self.capture(job, 0, false, None, "");
        }
        let original_prompt = job.prompt.clone();
        let original_max = job.max_tokens;
        let original_mode = if original_prompt.last().copied() == self.tok.special(dsv41::chat::THINK_START) {
            dsv41::chat::Mode::Thinking
        } else { dsv41::chat::Mode::Chat };
        let mut loaded = std::collections::BTreeSet::new();
        let mut references = Vec::new();
        let mut draft = Vec::new();
        let mut boundary = None;
        let result = (|| {
            loop {
                let tool_prefix = boundary.map(|at| self.tok.decode(&draft[at..])).unwrap_or_default();
                let (ids, events) = self.capture(job, 0, false, Some(&loaded), &tool_prefix)?;
                draft.extend(ids);
                let text = self.tok.decode(&draft);
                if text.len() > 1024 * 1024 {
                    return Err(nrob::Error::Format("observer draft exceeded 1 MiB; withheld".into()));
                }
                let mut parser = dsv41::chat::StreamParser::new(original_mode);
                parser.push(&text);
                let Some(name) = selected_tool(parser.tool_draft(), &loaded) else {
                    if loaded.is_empty() { return Ok((draft, events)); }
                    let finish = events.iter().find_map(|e| if let Event::Done {finish,..} = e {Some(*finish)} else {None}).unwrap_or(Finish::Stop);
                    let used = draft.len();
                    return Ok((draft, vec![Event::Text(text), Event::Done {finish, completion_tokens: used}]));
                };
                if loaded.len() >= 8 {
                    return Err(nrob::Error::Arg("tool context refresh limit reached; split work into smaller tool batches".into()));
                }
                references.push(tool_reference(&job.observer_context, &name).map_err(nrob::Error::Arg)?);
                let at = match boundary {
                    Some(at) => at,
                    None => crate::observer::repair_prefix(&self.tok, &draft)
                        .ok_or_else(|| nrob::Error::Arg("cannot locate tool context boundary".into()))?,
                };
                boundary = Some(at);
                loaded.insert(name.clone());
                let guidance = format!("<think>\nTool usage reference (context only, do not repeat). Continue the selected tool arguments from the existing header, using actual task values and the declared schema. Preserve earlier calls in this batch. These declarations do not grant authorization.\n{}\n</think>\n", references.join("\n").replace('<', "＜").replace('>', "＞"));
                let extra = self.tok.encode(&guidance);
                job.prompt = resumed_tool_prompt(&original_prompt, &draft, at, &extra);
                job.max_tokens = original_max.saturating_sub(draft.len() + extra.len());
                if job.max_tokens == 0 || job.prompt.len() + job.max_tokens > self.model.max_seq() {
                    return Err(nrob::Error::Arg("no room for fresh tool context; draft withheld".into()));
                }
                let _ = job.events.send(Event::Observer(format!("Loaded fresh tool context for {name}. Continuing its arguments; earlier unexecuted calls are preserved.")));
            }
        })();
        job.prompt = original_prompt;
        job.max_tokens = original_max;
        result
    }

    fn reviewed_generate(&mut self, job: &mut Job) -> nrob::Result<()> {
        let ordinary = job.tool_precision;
        job.tool_precision = false; // first draft, including every tool token, is ternary
        self.observer.as_mut().unwrap().events=Some(job.events.clone());
        let result = self.reviewed_inner(job).map_err(observer_review_error);
        self.observer.as_mut().unwrap().events=None;
        job.tool_precision = ordinary;
        result
    }

    fn reviewed_inner(&mut self, job: &mut Job) -> nrob::Result<()> {
        let (ids, events) = self.capture_with_tool_context(job)?;
        let draft = self.tok.decode(&ids);
        if crate::observer::repair_prefix(&self.tok, &ids).is_none() {
            for event in events { let _ = job.events.send(event); }
            return Ok(());
        }
        crate::observer::validate_tools(&job.observer_context, &draft)
            .map_err(|e| nrob::Error::Format(format!("tool_contract_error: {e}; no tool executed")))?;
        let _ = job.events.send(Event::Observer("Checking the proposed calls once before execution...".into()));
        let decision = self.observer.as_mut().unwrap().review(&job.observer_context, &draft, &job.cancel)?;
        let _ = job.events.send(Event::Observer(decision.comment.clone()));
        release_reviewed_events(&decision, events, &job.events, &job.cancel)
    }

    fn generate(&mut self, job: &Job, generated: &mut Vec<u32>, q4_tokens: usize, comments: bool, fresh: Option<&std::collections::BTreeSet<String>>, tool_prefix: &str) -> nrob::Result<()> {
        let prompt = &job.prompt;
        if prompt.is_empty() {
            return Err(nrob::Error::Arg("empty prompt".into()));
        }
        self.model.trace_phase(self.request_number, "prompt", None, "");
        let started = Instant::now();
        let keys = prompt_keys(prompt, &job.images);
        let (start, source, common) = self.resume(&keys)?;
        let _ = job.events.send(Event::CacheReuse { cached: start, source, common });
        if self.log {
            eprintln!("  prompt cache: {start}/{} reused from {source}; {} to process; {common} tokens match prior in-memory history", prompt.len(), prompt.len() - start);
        }
        // the images the uncached part reaches, through the vision tower
        let mut spans = Vec::new();
        for img in job.images.iter().filter(|i| i.start + i.prep.n_tokens() > start) {
            let t = Instant::now();
            spans.push(ImageSpan { start: img.start, rows: self.model.encode_image(img.prep.deepseek()?)? });
            if self.log {
                let prep = img.prep.deepseek()?;
                let (h, w) = (prep.n_vit_h * 14, prep.n_vit_w * 14);
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
                            // the state is whole up to p: keep it, including on disk.
                            self.save_checkpoint(p)?;
                            self.persist(p, false);
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
                } else {
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
        if self.repetition_guard { budget.repetition = Some(Default::default()); }
        let mut channel = dsv41::chat::StreamParser::new(if budget.thinking {
            dsv41::chat::Mode::Thinking
        } else { dsv41::chat::Mode::Chat });
        let decode = Instant::now();
        let mut phase = crate::tool_phase::ToolPhase::default();
        let mut complete_tools = dsv41::chat::StreamParser::new(if budget.thinking { dsv41::chat::Mode::Thinking } else { dsv41::chat::Mode::Chat });
        // The continuation starts inside a tool envelope already in the prompt.
        // Seed every parser/controller without redisplaying or dispatching it.
        channel.push(tool_prefix);
        phase.push(tool_prefix);
        complete_tools.push(tool_prefix);
        let mut reasoning_history = std::collections::VecDeque::new();
        let mut commented = false;
        let mut reasoning_commented = false;
        let mut repetition = crate::repetition::Guard::default();
        let mut prose = crate::repetition::ProseGuard::default();
        let phase_enabled = self.tool_experts && job.tool_precision;
        let finish = loop {
            let was_thinking = budget.thinking;
            if budget.thinking && !phase.original() {
                penalize_reasoning(&mut logits, &reasoning_history, job.sampling.reasoning_repeat_penalty);
            } else { reasoning_history.clear(); }
            let (next, forced) = budget.pass(sample(&logits, &job.sampling, &mut rng));
            if was_thinking && !phase.original() && !forced && !self.tok.is_special(next) {
                reasoning_history.push_back(next);
                while reasoning_history.len() > job.sampling.reasoning_repeat_last_n { reasoning_history.pop_front(); }
            }
            if forced && self.log {
                eprintln!("  reasoning ended by budget/repetition control after {} tokens", budget.used);
            }
            if next == self.eos {
                break Finish::Stop;
            }
            n += 1;
            generated.push(next);
            pending.extend_from_slice(self.tok.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            let mut prose_repeated = false;
            if valid > 0 {
                let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                complete_tools.push(&text);
                let was_tool = phase.original();
                phase.push(&text); // also identifies payloads exempt from the prose repetition guard
                if !was_tool && !phase.original() {
                    if forced { channel.end_reasoning(); }
                    for delta in channel.push(&text) {
                        match delta {
                            dsv41::chat::Delta::Content(text) => {
                                if self.repetition_guard { prose_repeated |= prose.push(&text); }
                            }
                            dsv41::chat::Delta::Reasoning(_) => prose.clear(),
                        }
                    }
                    budget.thinking = channel.is_reasoning();
                    if !was_thinking && budget.thinking && self.repetition_guard {
                        budget.repetition = Some(Default::default());
                    }
                } else { budget.thinking = false; prose.clear(); }
                if was_thinking != budget.thinking || budget.thinking && budget.used.is_multiple_of(THINKING_EVERY) {
                    let _ = job.events.send(Event::Thinking { used: budget.used, budget: budget.budget, done: !budget.thinking });
                }
                if job.events.send(Event::Text(text)).is_err() {
                    break Finish::Stop; // nobody is listening
                }
            }
            // Stop an entire completed call envelope before generating post-tool
            // prose. Observer capture withholds output, so the API cannot cancel
            // generation at this boundary for us.
            if fresh.is_some_and(|loaded|selected_tool(complete_tools.tool_draft(),loaded).is_some()) {break Finish::Stop;}
            if complete_tools.tool_calls_ready() { break Finish::Stop; }
            if self.repetition_guard && (prose_repeated || repetition.push(next, phase.original())) {
                return Err(nrob::Error::Format(format!("{}: stopped repeated prose/reasoning blocks; generation did not complete. No tool from this incomplete reply was executed. Start a fresh turn or select the original model; automatic retry is disabled.", crate::repetition::CODE)));
            }
            let mut advice_to_inject = None;
            // A single consultation at a complete reasoning line. Never inject
            // into tool arguments or a partial UTF-8/token-markup fragment.
            let tail = self.tok.token_bytes(next);
            if comments && !reasoning_commented && safe_reasoning_consultation(
                budget.thinking,phase.original(),pending.is_empty(),tail,budget.used) {
                reasoning_commented = true;
                if let Some(observer) = self.observer.as_mut() {
                    let _ = job.events.send(Event::Observer("Helping DeepSeek with its current reasoning...".into()));
                    match observer.advise_reasoning(&job.observer_context,&self.tok.decode(generated),&job.cancel) {
                        Ok(note) => { advice_to_inject = Some(note); }
                        Err(_) => { let _ = job.events.send(Event::Observer("Reasoning advice unavailable; DeepSeek is continuing.".into())); }
                    }
                }
            }
            // One provisional note at a natural tool-writing checkpoint. Both
            // models stay resident, but compute is scheduled sequentially.
            if comments && !commented && n >= 512 && phase.original() {
                commented = true;
                let draft = self.tok.decode(generated);
                if let Some(observer) = self.observer.as_mut() {
                    let _ = job.events.send(Event::Observer("Looking over the unfinished tool draft...".into()));
                    match observer.comment(&job.observer_context, &draft, &job.cancel) {
                        Ok(note) => { let _ = job.events.send(Event::Observer(format!("Draft note (provisional; automatically resuming DeepSeek): {note}"))); }
                        Err(_) => { let _ = job.events.send(Event::Observer("Draft note unavailable; completed tools still require review.".into())); }
                    }
                }
            }
            if n >= job.max_tokens || pos + 1 >= self.model.max_seq() {
                break Finish::Length;
            }
            if job.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            if q4_tokens > 0 { self.switch_phase(observer_q4_forward(n, q4_tokens))?; }
            else if phase_enabled { self.switch_phase(phase.original())?; }
            self.model.trace_phase(self.request_number,
                if q4_tokens > 0 && n <= q4_tokens {"observer_q4_window"} else if phase_enabled && phase.original() {"tool_call"} else {"reply"},
                Some(next), &String::from_utf8_lossy(self.tok.token_bytes(next)));
            logits = self.model.forward(&[next], pos)?;
            self.tokens.push(u64::from(next));
            pos += 1;
            // Only single-precision state is valid for a later prompt. Save
            // committed forwards (never the last sampled, unconsumed token).
            if cache_generated_state(self.tool_experts, pos, prompt.len(), false) {
                self.save_checkpoint(pos)?;
                self.persist(pos, false);
            }
            if let Some(note) = advice_to_inject {
                let advice = reasoning_advice(&note);
                let advice_ids = self.tok.encode(&advice);
                let remaining = job.max_tokens.saturating_sub(n)
                    .min(self.model.max_seq().saturating_sub(pos + 1))
                    .min(budget.budget.map_or(usize::MAX,|b|b.saturating_sub(budget.used)));
                if advice_ids.len().saturating_add(32) <= remaining {
                    for &id in &advice_ids {
                        if job.cancel.load(Ordering::Relaxed) { return Err(nrob::Error::Arg("cancelled while handing off observer advice".into())); }
                        self.model.trace_phase(self.request_number,"observer_reasoning_advice",Some(id),&String::from_utf8_lossy(self.tok.token_bytes(id)));
                        logits = self.model.forward(&[id],pos)?;
                        self.tokens.push(u64::from(id)); pos += 1;
                    }
                    n += advice_ids.len();
                    budget.used += advice_ids.len();
                    generated.extend_from_slice(&advice_ids);
                    phase.push(&advice);
                    let _ = channel.push(&advice);
                    let _ = job.events.send(Event::Text(advice));
                    let _ = job.events.send(Event::Observer(format!("Advice delivered to DeepSeek; continuing automatically: {note}")));
                } else {
                    let _ = job.events.send(Event::Observer(format!("Advice saved for the next turn (current reasoning budget is nearly full): {note}")));
                }
            }
        };
        if cache_generated_state(self.tool_experts, pos, prompt.len(), true) {
            self.save_checkpoint(pos)?;
            self.persist(pos, false);
        }
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

// A main-model repetition stop is not a reviewer failure. Keep its original
// code/message so the client can explain the cause without suggesting Qwen failed.
fn observer_review_error(error: nrob::Error) -> nrob::Error {
    if error.to_string().contains(crate::repetition::CODE) {
        error
    } else {
        nrob::Error::Format(format!("{}: {error}", crate::observer::CODE))
    }
}

#[test]
fn reviewed_generation_preserves_repetition_failure_without_blaming_observer() {
    let error = nrob::Error::Format(format!("{}: stopped repeated prose; no tool executed", crate::repetition::CODE));
    let original = error.to_string();
    let reported = observer_review_error(error).to_string();
    assert_eq!(reported, original);
    assert!(!reported.contains(crate::observer::CODE));
    let rejected = observer_review_error(nrob::Error::Format("observer rejected the repair".into())).to_string();
    assert!(rejected.contains(crate::observer::CODE));
}

// Automatic review follows the loaded expert representation, never a model name.
fn blocking_observer_review(context: &str, ternary: bool) -> bool {
    let context = nrob::json::Json::parse(context.as_bytes()).ok();
    match context.as_ref().and_then(|c| c.get("review_policy")).and_then(nrob::json::Json::as_str) {
        Some("blocking") => true,
        Some("off") => false,
        Some("auto") => ternary,
        _ => false,
    }
}

#[test]
fn automatic_observer_review_is_for_ternary_and_explicit_modes_override_it() {
    for context in [r#"{"review_policy":"auto"}"#] {
        assert!(!blocking_observer_review(context, false));
        assert!(blocking_observer_review(context, true));
    }
    for ternary in [false, true] {
        assert!(!blocking_observer_review("{}", ternary));
        assert!(blocking_observer_review(r#"{"review_policy":"blocking"}"#, ternary));
        assert!(!blocking_observer_review(r#"{"review_policy":"off"}"#, ternary));
    }
}

fn resumed_tool_prompt(original: &[u32], draft: &[u32], boundary: usize, guidance: &[u32]) -> Vec<u32> {
    let mut prompt = original.to_vec();
    prompt.extend_from_slice(&draft[..boundary]);
    prompt.extend_from_slice(guidance);
    prompt.extend_from_slice(&draft[boundary..]);
    prompt
}

fn selected_tool(draft:&str,loaded:&std::collections::BTreeSet<String>)->Option<String> {
    for marker in ["<｜DSML｜ invoke name=\"","<｜DSML｜invoke name=\""] {
        for rest in draft.split(marker).skip(1) {
            let (name,tail)=rest.split_once('"')?;
            if tail.starts_with('>') && !name.is_empty() && !loaded.contains(name) {return Some(name.into());}
        }
    }
    None
}
fn tool_reference(context:&str,name:&str)->std::result::Result<String,String> {
    let context=nrob::json::Json::parse(context.as_bytes()).map_err(|e|e.to_string())?;
    let function=context.get("tools").and_then(nrob::json::Json::as_array).into_iter().flatten()
        .filter_map(|t|t.get("function")).find(|f|f.get("name").and_then(nrob::json::Json::as_str)==Some(name))
        .ok_or_else(||format!("unknown selected tool {name}; no context or execution available"))?;
    let reference=function.to_json();
    if reference.len()>16*1024 {return Err(format!("tool reference for {name} exceeds 16 KiB context-refresh limit"));}
    let no_arguments=function.get("parameters").is_some_and(|schema|schema.get("properties")==Some(&nrob::json::Json::Obj(vec![])) && schema.get("additionalProperties")==Some(&nrob::json::Json::Bool(false)));
    Ok(if no_arguments {format!("{reference}\nThis is a zero-argument tool. Emit an empty invoke body with no parameter elements. The JSON object {{}} is not a parameter name. No args element is needed.")} else {reference})
}
#[test]
fn tool_context_is_selected_at_complete_header_and_comes_from_current_catalog() {
    let mut loaded=std::collections::BTreeSet::new();
    assert!(selected_tool("<｜DSML｜ invoke name=\"write_f",&loaded).is_none());
    assert_eq!(selected_tool("<｜DSML｜ invoke name=\"write_file\">",&loaded).as_deref(),Some("write_file"));
    loaded.insert("write_file".into());
    assert!(selected_tool("<｜DSML｜ invoke name=\"write_file\">",&loaded).is_none());
    let context=r#"{"tools":[{"function":{"name":"write_file","description":"Write relative to workspace","parameters":{"type":"object","required":["path","content"]}}}]}"#;
    let reference=tool_reference(context,"write_file").unwrap();
    assert!(reference.contains("path"));assert!(reference.contains("content"));assert!(reference.contains("relative to workspace"));
    assert!(tool_reference(context,"unknown").is_err());
}

fn hold_observer_events(rx: Receiver<Event>, forward: Sender<Event>, cancel: Arc<AtomicBool>, mode: Option<dsv41::chat::Mode>, tool_prefix: &str) -> (Vec<Event>, usize) {
            let mut preview = mode.map(dsv41::chat::StreamParser::new);
            let mut kept = Vec::new();
            let mut size = 0usize;
            let mut tool_shown = 0usize;
            if let Some(parser) = preview.as_mut() {
                parser.push(tool_prefix);
                tool_shown = parser.tool_draft().len();
            }
            for event in rx {
                match event {
                    Event::Text(ref text) => {
                        if let Some(parser) = preview.as_mut() {
                            for delta in parser.push(text) {
                                let event = match delta {
                                    dsv41::chat::Delta::Reasoning(text) => Event::ReasoningPreview(text),
                                    dsv41::chat::Delta::Content(text) => Event::ContentPreview(text),
                                };
                                let _ = forward.send(event);
                            }
                            let draft = parser.tool_draft();
                            if draft.len() > tool_shown {
                                let _ = forward.send(Event::ToolPreview {text:draft[tool_shown..].into(),start:tool_shown==0});
                                tool_shown=draft.len();
                            }
                        }
                        size += text.len();
                        if size > 1024 * 1024 { cancel.store(true, Ordering::Relaxed); }
                        else { kept.push(event); }
                    }
                    Event::Thinking {done:true,..} => {
                        if let Some(parser) = preview.as_mut() {
                            for delta in parser.end_reasoning() {
                                if let dsv41::chat::Delta::Reasoning(text) = delta { let _ = forward.send(Event::ReasoningPreview(text)); }
                            }
                        }
                        kept.push(event);
                    }
                    Event::Done {..} => {
                        if let Some(parser) = preview.take() {
                            // Tool text is already visible; only flush a held prose marker tail.
                            if parser.tool_draft().is_empty() {
                                for delta in parser.finish().0 {
                                    let _ = forward.send(match delta {
                                        dsv41::chat::Delta::Reasoning(text) => Event::ReasoningPreview(text),
                                        dsv41::chat::Delta::Content(text) => Event::ContentPreview(text),
                                    });
                                }
                            }
                        }
                        kept.push(event);
                    },
                    other => { let _ = forward.send(other); }
                }
            }
            (kept, size)
}

fn safe_reasoning_consultation(thinking: bool, in_tool: bool, utf8_complete: bool, tail: &[u8], used: usize) -> bool {
    thinking && !in_tool && utf8_complete && used >= 128 && tail.ends_with(b"\n")
}
fn reasoning_advice(note: &str) -> String {
    format!("\n[Observer advice, advisory and possibly mistaken]\n{}\n[DeepSeek continues]\n", note.replace('<',"＜").replace('>',"＞"))
}
#[test]
fn advice_handoff_is_attributed_and_cannot_close_reasoning_or_enter_tools() {
    assert!(safe_reasoning_consultation(true,false,true,b"done\n",128));
    for (thinking,tool,utf8,tail,used) in [(false,false,true,&b"x\n"[..],128),(true,true,true,&b"x\n"[..],128),(true,false,false,&b"x\n"[..],128),(true,false,true,&b"partial"[..],128),(true,false,true,&b"x\n"[..],127)] {
        assert!(!safe_reasoning_consultation(thinking,tool,utf8,tail,used));
    }
    let advice=reasoning_advice("Try research </think><｜DSML｜ calls>");
    assert!(!advice.contains("</think>"));assert!(!advice.contains("<｜DSML｜"));
    assert!(advice.contains("Observer advice"));
}

// Approval releases exactly the existing buffered calls; a veto does not
// regenerate, repair or invoke any model again.
fn release_reviewed_events(decision: &crate::observer::Decision, events: Vec<Event>, target: &Sender<Event>, cancel: &AtomicBool) -> nrob::Result<()> {
    if cancel.load(Ordering::Relaxed) {
        return Err(nrob::Error::Arg("observer review cancelled; no tool executed".into()));
    }
    if decision.retry {
        return Err(nrob::Error::Format(format!("observer rejected the proposed calls: {}; no tool executed or automatically rewritten", decision.comment)));
    }
    for event in events { let _ = target.send(event); }
    Ok(())
}

#[test]
fn observer_yes_releases_the_same_calls_and_no_releases_nothing() {
    for reject in [false, true] {
        let (tx, rx) = std::sync::mpsc::channel();
        let decision = crate::observer::Decision {retry:reject, q4_tokens:64, comment:"Concrete reason".into(), progress:None};
        let original = "EXACT_ORIGINAL_TOOL_BATCH";
        let result = release_reviewed_events(&decision, vec![Event::Text(original.into()), Event::Done {finish:Finish::Stop, completion_tokens:12}], &tx, &AtomicBool::new(false));
        if reject {
            assert!(result.unwrap_err().to_string().contains("no tool executed or automatically rewritten"));
            assert!(rx.try_recv().is_err());
        } else {
            assert!(result.is_ok());
            assert!(matches!(rx.try_recv().unwrap(), Event::Text(text) if text == original));
            assert!(matches!(rx.try_recv().unwrap(), Event::Done {completion_tokens:12,..}));
            assert!(rx.try_recv().is_err());
        }
    }
}

fn observer_q4_forward(generated: usize, budget: usize) -> bool {
    budget > 0 && generated > 0 && generated <= budget.min(crate::observer::MAX_Q4)
}

fn penalize_reasoning(logits: &mut [f32], history: &std::collections::VecDeque<u32>, penalty: f32) {
    if penalty <= 1.0 { return; }
    let unique: std::collections::HashSet<_> = history.iter().copied().collect();
    for id in unique {
        if let Some(value) = logits.get_mut(id as usize) {
            *value = if *value < 0.0 { *value * penalty } else { *value / penalty };
        }
    }
}
#[test]
fn repetition_penalty_is_sign_aware_once_per_recent_token_and_disabled_at_one() {
    let history = std::collections::VecDeque::from(vec![0, 0, 1]);
    let original = vec![2.0, -2.0, 1.5];
    let mut logits = original.clone();
    penalize_reasoning(&mut logits, &history, 1.0); assert_eq!(logits, original);
    penalize_reasoning(&mut logits, &history, 2.0); assert_eq!(logits, vec![1.0,-4.0,1.5]);
}
#[test]
fn observer_streams_all_draft_channels_without_releasing_executable_events() {
    use dsv41::chat::Mode;
    let input="first λ</think>Writing now.<think>recheck</think><｜DSML｜ calls><｜DSML｜ invoke name=\"write_file\"><｜DSML｜ parameter name=\"content\" string=\"true\">CODE λ</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    let (tx,rx)=std::sync::mpsc::channel(); let (out,visible)=std::sync::mpsc::channel();
    for ch in input.chars() { tx.send(Event::Text(ch.to_string())).unwrap(); }
    drop(tx);
    let (held,_) = hold_observer_events(rx,out,Arc::new(AtomicBool::new(false)),Some(Mode::Thinking), "");
    assert_eq!(held.len(),input.chars().count());
    let (mut reasoning,mut content,mut tools)=(String::new(),String::new(),String::new());
    let mut starts=0;
    for event in visible.try_iter() { match event {
        Event::ReasoningPreview(s)=>reasoning.push_str(&s),
        Event::ContentPreview(s)=>content.push_str(&s),
        Event::ToolPreview{text,start}=>{starts+=usize::from(start);tools.push_str(&text);},
        _=>panic!("only display previews may escape before review"),
    }}
    assert_eq!(reasoning,"first λrecheck"); assert_eq!(content,"Writing now.");
    assert_eq!(tools,input[input.find("<｜DSML｜ calls>").unwrap()..]);assert_eq!(starts,1);
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
    #[ignore = "requires NROB_TOKENIZER_DIR; CPU tokenizer only, no model inference"]
    fn short_tool_intention_loop_with_the_real_tokenizer() {
        let dir = std::env::var("NROB_TOKENIZER_DIR").expect("set NROB_TOKENIZER_DIR");
        let tok = Tokenizer::load(std::path::Path::new(&dir)).unwrap();
        let start = tok.special(dsv41::chat::THINK_START).unwrap();
        let end = tok.special(dsv41::chat::THINK_END).unwrap();
        assert_eq!(tok.encode(dsv41::chat::THINK_START), vec![start]);
        let tokens = tok.encode(&"Let's call web_search.\n\nLet's call.\n\n".repeat(20));
        let mut b = ThinkBudget::new(true, Some(end), Some(2048));
        b.repetition = Some(Default::default());
        let forced = tokens.iter().position(|&t| b.pass(t) == (end, true));
        assert!(forced.is_some_and(|i| i < 376), "must close the repeated intention loop");
        println!("real tokenizer: reasoning closed at generated token {}", forced.unwrap() + 1);
    }

    #[test]
    fn short_repetition_closes_thinking_once_without_touching_the_answer() {
        const END: u32 = 1000;
        let mut b = ThinkBudget::new(true, Some(END), Some(2048));
        b.repetition = Some(Default::default());
        for i in 0..71 { assert_eq!(b.pass(i % 9), (i % 9, false)); }
        assert_eq!(b.pass(8), (END, true));
        assert!(!b.thinking);
        for _ in 0..100 { assert_eq!(b.pass(8), (8, false)); }
        let mut disabled = ThinkBudget::new(true, Some(END), None);
        for _ in 0..100 { assert_eq!(disabled.pass(8), (8, false)); }
    }

    #[test]
    fn reopened_thinking_uses_the_same_total_budget_and_end_token() {
        const END: u32 = 1000;
        let mut b = ThinkBudget::new(false, Some(END), Some(2));
        b.thinking = true; // parser sees a later <think>
        assert_eq!(b.pass(7), (7, false));
        assert_eq!(b.pass(END), (END, false));
        assert_eq!(b.pass(8), (8, false)); // answer tokens do not count
        b.thinking = true;
        assert_eq!(b.pass(9), (9, false));
        assert_eq!(b.pass(10), (END, true));
    }

    #[test]
    fn observer_withholds_all_draft_text_and_completion() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (out, visible) = std::sync::mpsc::channel();
        tx.send(Event::Text("<tool draft>".into())).unwrap();
        tx.send(Event::Observer("a provisional note".into())).unwrap();
        tx.send(Event::Done { finish: Finish::Stop, completion_tokens: 3 }).unwrap();
        drop(tx);
        let (held, _) = hold_observer_events(rx, out, Arc::new(AtomicBool::new(false)), None, "");
        assert_eq!(held.len(), 2);
        let shown: Vec<_> = visible.try_iter().collect();
        assert_eq!(shown.len(), 1);
        assert!(matches!(&shown[0], Event::Observer(_)));
    }

    #[test]
    fn observer_does_not_close_reasoning_ahead_of_buffered_text() {
        let (tx,rx)=std::sync::mpsc::channel();
        let (out,visible)=std::sync::mpsc::channel();
        tx.send(Event::Text("initial thought".into())).unwrap();
        tx.send(Event::Thinking {used:10,budget:Some(20),done:true}).unwrap();
        tx.send(Event::Text("</think>answer".into())).unwrap();
        drop(tx);
        let (held,_)=hold_observer_events(rx,out,Arc::new(AtomicBool::new(false)),None, "");
        assert_eq!(visible.try_iter().count(),0);
        let mut parser=dsv41::chat::StreamParser::new(dsv41::chat::Mode::Thinking);
        let mut deltas=Vec::new();
        for event in held {
            match event {
                Event::Text(text)=>deltas.extend(parser.push(&text)),
                Event::Thinking {done:true,..}=>deltas.extend(parser.end_reasoning()),
                _=>{},
            }
        }
        deltas.extend(parser.finish().0);
        assert!(deltas.iter().any(|d|matches!(d,dsv41::chat::Delta::Reasoning(t) if t.contains("initial thought"))));
        assert!(!deltas.iter().any(|d|matches!(d,dsv41::chat::Delta::Content(t) if t.contains("initial thought"))));
    }

    #[test]
    fn observer_q4_window_is_bounded_and_returns_to_ternary() {
        for budget in [0, 1, 12, 64, 256, 1000] {
            let active = (0..1200).filter(|&n|observer_q4_forward(n,budget)).count();
            assert_eq!(active, budget.min(crate::observer::MAX_Q4));
            assert!(!observer_q4_forward(0,budget));
            assert!(!observer_q4_forward(257,budget));
        }
    }

    #[test]
    fn tool_completion_boundary_waits_for_the_whole_parallel_envelope() {
        let mut parser=dsv41::chat::StreamParser::new(dsv41::chat::Mode::Thinking);
        parser.push("Inspect first.</think><｜DSML｜ calls><｜DSML｜ invoke name=\"workspace_info\"></｜DSML｜ invoke>");
        assert!(!parser.tool_calls_ready());
        parser.push("<｜DSML｜ invoke name=\"list_files\"><｜DSML｜ parameter name=\"path\" string=\"true\">.</｜DSML｜ parameter></｜DSML｜ invoke>");
        assert!(!parser.tool_calls_ready());
        parser.push("</｜DSML｜ calls>");
        assert!(parser.tool_calls_ready());
        assert_eq!(parser.finish().1.len(),2);
    }
    #[test]
    fn sampling_respects_temperature_top_k_and_top_p() {
        let logits = vec![0.0, 5.0, 4.0, -1.0, 4.9];
        let mut rng = 1;
        let greedy = Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0, reasoning_repeat_penalty: 1.0, reasoning_repeat_last_n: 256 };
        assert_eq!(sample(&logits, &greedy, &mut rng), 1);
        let top1 = Sampling { temperature: 1.0, top_p: 1.0, top_k: 1, seed: 0, reasoning_repeat_penalty: 1.0, reasoning_repeat_last_n: 256 };
        for _ in 0..50 {
            assert_eq!(sample(&logits, &top1, &mut rng), 1);
        }
        // top-p 0.5 keeps the best two (5.0, 4.9: ~0.47 then ~0.9 of the mass)
        let nucleus = Sampling { temperature: 1.0, top_p: 0.5, top_k: 0, seed: 0, reasoning_repeat_penalty: 1.0, reasoning_repeat_last_n: 256 };
        let mut seen = [0usize; 5];
        for _ in 0..2000 {
            seen[sample(&logits, &nucleus, &mut rng) as usize] += 1;
        }
        assert!(seen[1] > 0 && seen[4] > 0 && seen[0] + seen[2] + seen[3] == 0, "{seen:?}");
    }
}

#[test]
fn refreshed_parallel_tool_preview_continues_without_repeating_or_executing_prefix() {
    let prefix = "<｜DSML｜ calls><｜DSML｜ invoke name=\"workspace_info\"></｜DSML｜ invoke><｜DSML｜ invoke name=\"list_files\">";
    let suffix = "<｜DSML｜ parameter name=\"path\" string=\"true\">λ</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    let (tx, rx) = std::sync::mpsc::channel();
    let (out, visible) = std::sync::mpsc::channel();
    for ch in suffix.chars() { tx.send(Event::Text(ch.to_string())).unwrap(); }
    tx.send(Event::Done {finish: Finish::Stop, completion_tokens: 1}).unwrap();
    drop(tx);
    let (held, _) = hold_observer_events(rx, out, Arc::new(AtomicBool::new(false)), Some(dsv41::chat::Mode::Chat), prefix);
    let mut shown = String::new();
    for event in visible.try_iter() {
        match event {
            Event::ToolPreview {text, start} => { assert!(!start); shown.push_str(&text); }
            _ => panic!("continuation must stay in a display-only tool draft"),
        }
    }
    assert_eq!(shown, suffix);
    assert!(matches!(held.last(), Some(Event::Done {..})));
    let mut parser = dsv41::chat::StreamParser::new(dsv41::chat::Mode::Chat);
    parser.push(prefix);
    assert!(!parser.tool_calls_ready());
    parser.push(suffix);
    assert!(parser.tool_calls_ready());
    let calls = parser.finish().1;
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "workspace_info");
    assert_eq!(calls[1].name, "list_files");
    let arguments = nrob::json::Json::parse(calls[1].arguments.as_bytes()).unwrap();
    assert_eq!(arguments.get("path").and_then(nrob::json::Json::as_str), Some("λ"));
}

#[test]
#[ignore = "requires NROB_TOKENIZER_DIR; CPU tokenizer only, no model inference"]
fn refreshed_parallel_tool_prompt_preserves_real_token_ids_and_excludes_guidance_from_output() {
    let dir = std::env::var("NROB_TOKENIZER_DIR").expect("set NROB_TOKENIZER_DIR");
    let tok = Tokenizer::load(std::path::Path::new(&dir)).unwrap();
    let original = tok.encode("<｜Assistant｜><think>");
    let mut draft = tok.encode("Inspect first.</think>Inspecting.\n\n<｜DSML｜ calls><｜DSML｜ invoke name=\"workspace_info\">");
    let boundary = crate::observer::repair_prefix(&tok, &draft).unwrap();
    let guidance = tok.encode("<think>PRIVATE_REFERENCE_ONLY</think>\n");
    let first = resumed_tool_prompt(&original, &draft, boundary, &guidance);
    assert!(first.ends_with(&draft[boundary..]));
    draft.extend(tok.encode("</｜DSML｜ invoke><｜DSML｜ invoke name=\"list_files\">"));
    let second = resumed_tool_prompt(&original, &draft, boundary, &guidance);
    assert!(second.ends_with(&draft[boundary..]));
    assert_eq!(&second[..original.len()+boundary], &[original.clone(), draft[..boundary].to_vec()].concat());
    assert_eq!(tok.decode(&second).matches("PRIVATE_REFERENCE_ONLY").count(), 1);
    let mut parser = dsv41::chat::StreamParser::new(dsv41::chat::Mode::Chat);
    parser.push(&tok.decode(&draft[boundary..]));
    assert!(!parser.tool_calls_ready());
    let suffix = "<｜DSML｜ parameter name=\"path\" string=\"true\">.</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    parser.push(suffix);
    assert!(parser.tool_calls_ready());
    assert_eq!(parser.finish().1.len(), 2);
    draft.extend(tok.encode(suffix));
    assert!(!tok.decode(&draft).contains("PRIVATE_REFERENCE_ONLY"));
}

// Bound device snapshot overhead to one checkpoint per 256 committed reply
// tokens, plus the final prefix. Hybrid expert state must never leak into a
// ternary prompt cache. A changed prompt still requires exact prefix matching.
fn cache_generated_state(hybrid: bool, pos: usize, prompt_len: usize, final_prefix: bool) -> bool {
    !hybrid && pos > prompt_len && (final_prefix || (pos - prompt_len).is_multiple_of(256))
}

#[test]
fn generated_cache_uses_only_committed_single_precision_prefixes() {
    assert!(!cache_generated_state(false, 100, 100, true));
    assert!(!cache_generated_state(false, 355, 100, false));
    assert!(cache_generated_state(false, 356, 100, false));
    assert!(cache_generated_state(false, 110, 100, true));
    assert!(!cache_generated_state(true, 356, 100, true));
}
