//! The server's DeepSeek worker: requests, the state they continue, and what it does while idle.

use super::*;

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
    /// The experts' kernel on the GPUs, for what the other cards hold ([`Pinned`]): filled while idle too.
    kernel: Option<Arc<WgpuExperts>>,
    /// Experts the idle rebalance has moved since it last said so.
    rebalanced: usize,
    /// Whether the drive may be read while no request waits (the idle reading, the rebalance).
    idle: bool,
    /// The usage profile's file (`--usage`): the experts' counts of uses, written after each request and read at the
    /// next start.
    usage: Option<std::path::PathBuf>,
    /// The order a profile gave the idle reading (every expert, the most used first); None without one (by number).
    order: Option<Vec<(u32, u32)>>,
    /// The RAM tier's ceiling as configured (`--ram-gb`; 0: none but the memory free), and when the tier was last
    /// fitted to the memory free ([`Engine::fit_ram`]).
    ram_gb: usize,
    fitted: Option<std::time::Instant>,
}

/// How often an idle server looks at the memory the computer has free ([`Engine::fit_ram`]).
const RAM_FIT_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

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
/// Experts an idle slice of the rebalance moves (each a record read from the drive and one uploaded: as long a wait).
const REBALANCE_SLICE: usize = 4;

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
        // (OAIY_DSV41_NO_WARM: leave the drive alone between requests: neither the idle reading nor the rebalance,
        // which reads a record for each expert a card gives up)
        let idle = std::env::var_os("OAIY_DSV41_NO_WARM").is_none();
        let warm = idle.then(|| Warm { cursor: 0, started: None, reading: std::time::Duration::ZERO, bytes_before: model.expert_cache().stats().bytes_read });
        Engine { model, tok, covered: Vec::new(), checkpoint: None, eos, log, warm, kernel: None, rebalanced: 0, idle, usage: None, order: None, ram_gb: 0, fitted: None }
    }

    /// The experts' kernel the model was given, when other cards hold a share of them: the idle reading fills those
    /// first, and leaves them out of RAM.
    pub(crate) fn with_kernel(mut self, kernel: Option<Arc<WgpuExperts>>) -> Engine {
        self.kernel = kernel;
        self
    }

    /// Keep a usage profile at `usage` (written after each request), and read the experts while idle in `order` (what
    /// the profile read at load gave: the most used first; None: there was none yet, and they are read by number).
    pub(crate) fn with_usage(mut self, usage: Option<std::path::PathBuf>, order: Option<Vec<(u32, u32)>>) -> Engine {
        self.usage = usage;
        self.order = order;
        self
    }

    /// The RAM tier's configured ceiling (`--ram-gb`; 0: none), for [`Self::fit_ram`].
    pub(crate) fn with_ram_ceiling(mut self, ram_gb: usize) -> Engine {
        self.ram_gb = ram_gb;
        self
    }

    /// The RAM tier fitted to the memory the computer has free now, by the rule it was sized by at load (four fifths
    /// of what is free, the tier's own records counted free, and no more than `--ram-gb` where that is set). The
    /// budget was what was free at the moment the model loaded and stayed that for good: a server started while
    /// another program was letting its memory go (the one it replaces, as measured: 78 GiB where 141 a minute later)
    /// kept half a tier all session. More room is taken when there is 2% more of it, and the idle reading goes on
    /// into it; records are let go when the rule gives a tenth less (another program wants the memory: held on to, it
    /// would be paged out and read back from a slower place than the checkpoint's drive).
    fn fit_ram(&mut self) {
        if !self.idle || self.fitted.is_some_and(|at| at.elapsed() < RAM_FIT_EVERY) {
            return;
        }
        self.fitted = Some(std::time::Instant::now());
        let Some((free, _)) = ggml_rs_wgpu::host_memory() else { return };
        let cache = self.model.expert_cache();
        let (now, held) = (cache.budget_bytes() as u64, (cache.len() * cache.rec_bytes()) as u64);
        let want = crate::host_cache_budget(self.ram_gb, Some(free as u64 + held));
        if want <= now + now / 50 && want >= now - now / 10 {
            return;
        }
        let slots = cache.set_budget(want as usize);
        if self.log {
            eprintln!(
                "  the experts' RAM tier is {:.1} GiB now ({slots} experts; {:.1} GiB before): {:.1} GiB of memory are free beside what it holds",
                want as f64 / (1u64 << 30) as f64,
                now as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64
            );
        }
        // more room: the idle reading takes it up again where it had finished, in the order of the counts as they are
        // now (the most used of what no tier holds first), not the order it started with
        if want > now && self.warm.is_none() {
            self.order = Some(self.model.expert_uses().order());
            self.warm = Some(Warm { cursor: 0, started: None, reading: std::time::Duration::ZERO, bytes_before: self.model.expert_cache().stats().bytes_read });
        }
    }

    /// The experts' counts of uses to the usage profile's file, if one is kept (a failure said once a request, not
    /// fatal: the profile is a start-up's head start, nothing an answer depends on).
    fn save_usage(&self) {
        let Some(path) = &self.usage else { return };
        if let Err(e) = write_usage(path, self.model.expert_uses()) {
            if self.log {
                eprintln!("  the usage profile was not written to {}: {e}", path.display());
            }
        }
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
        let elsewhere = |l: u32, e: u32| kernel.is_some_and(|k| k.elsewhere(l, e));
        let more = match &self.order {
            Some(order) => self.model.warm_experts_in(order, &mut w.cursor, WARM_SLICE, &elsewhere),
            None => self.model.warm_experts(&mut w.cursor, WARM_SLICE, &elsewhere),
        };
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

    /// An idle slice of the cards' rebalance ([`WgpuExperts::rebalance`]): whether it moved any (false: there is
    /// nothing more to move, said once with what was moved since the last request).
    fn rebalance_some(&mut self) -> bool {
        if !self.idle {
            return false;
        }
        let Some(k) = self.kernel.as_deref() else { return false };
        let moved = k.rebalance(self.model.expert_store().as_ref(), self.model.expert_cache(), self.model.expert_uses(), REBALANCE_SLICE);
        self.rebalanced += moved;
        if moved == 0 && self.rebalanced > 0 {
            if self.log {
                eprintln!("  {} experts moved onto the GPUs while idle, in place of their least used ({} so far)", self.rebalanced, k.moved());
            }
            self.rebalanced = 0;
        }
        moved > 0
    }

    pub(crate) fn run(mut self, jobs: std::sync::mpsc::Receiver<crate::engine::Job>) {
        use crate::engine::{Event, Finish};
        use std::sync::mpsc::TryRecvError;
        loop {
            self.fit_ram();
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
                // then, with nothing asked, the cards take RAM's most used experts in place of their least used
                match jobs.try_recv() {
                    Ok(job) => job,
                    Err(TryRecvError::Empty) => {
                        if self.rebalance_some() {
                            continue;
                        }
                        match jobs.recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        }
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            };
            if job.wipe {
                self.covered.clear();
                self.checkpoint = None;
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                continue;
            }
            // (an incognito request leaves the experts' counts as it found them: the profile keeps nothing of it)
            let before = job.forget.then(|| self.model.expert_uses().counts());
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
            match before {
                Some(counts) => drop(self.model.expert_uses().seed(&counts)),
                None => self.save_usage(),
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
        let kernel = self.kernel.clone();
        let read_so_far = move |m: &dsv41::model::Model| {
            let st = m.expert_cache().stats();
            let tier = kernel.as_ref().map_or((0, 0, 0, 0), |k| {
                let (_, hits, misses, admitted) = k.tier();
                (hits, misses, admitted, k.share_hits())
            });
            (st.bytes_read, st.hits, st.misses, tier)
        };
        type Read = (u64, u64, u64, (u64, u64, u64, u64));
        let part = |label: &str, before: Read, after: Read| {
            eprintln!(
                "    {label}: read {:.2} GB, {} hits / {} misses in RAM; the first card's tier {} hits / {} misses, {} taken in, the other cards' share {} hits; {}; {}",
                (after.0 - before.0) as f64 / 1e9,
                after.1 - before.1,
                after.2 - before.2,
                after.3 .0 - before.3 .0,
                after.3 .1 - before.3 .1,
                after.3 .2 - before.3 .2,
                after.3 .3 - before.3 .3,
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
