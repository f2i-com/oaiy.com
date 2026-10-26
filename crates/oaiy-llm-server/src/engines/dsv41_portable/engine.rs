//! The server's DeepSeek worker: requests, the state they continue, and what it does while idle.

use super::*;

/// The tokens one forward pass reads of a prompt. A pass reads each layer's experts once for all its tokens, so a
/// prompt is read whole when it fits: in pieces it would read them once a piece.
const PREFILL_CHUNK: usize = 4096;

/// The fewest tokens a prompt's system turn has for its state to be kept at its end ([`Engine`]'s `system`): a state
/// is every layer's caches whole (some 1 GB at a 65,536-token context), and a shorter turn is read again in seconds.
const SYSTEM_KEPT_FROM: usize = 256;

/// The portable server's DeepSeek worker: jobs in, events out, the contract of the CUDA engine (`crate::engines::job`) and
/// GLM's (`crate::engines::glm`). A request that extends what the model's state covers reads only the new tail, in one chunk
/// (`dsv41`'s attention continues a sequence by a chunk as it would token by token). That state has moved on through
/// the last reply, which the next turn's prompt writes again its own way (its chat template), so the state as the last
/// prompt left it is kept too, a token short of its end (so the same prompt again can still use it): a conversation's
/// next turn continues from there. The state at the end of a conversation's first long system turn is kept as well: a
/// new conversation that starts with the same one (an agent's instructions and tools, some 19-21k tokens, the same at
/// each of its runs) reads on from there. A prompt that shares none of them starts again. A reply ends at the
/// end-of-sentence token, at a complete tool call, or at `max_tokens`.
pub(crate) struct Engine {
    model: crate::dsv41::model::Model,
    tok: Arc<crate::dsv41::tokenizer::Tokenizer>,
    /// The token ids the model's state covers, in order.
    covered: Vec<u32>,
    /// The state after the last prompt's tokens but its last one, and those tokens.
    checkpoint: Option<(Vec<u32>, crate::dsv41::model::Checkpoint)>,
    /// The state at the end of a prompt's system turn ([`system_end`]), those tokens, and the drafter's keys then.
    system: Option<(Vec<u32>, Arc<crate::dsv41::model::Checkpoint>, Option<Vec<crate::dsv41::attention::AttnState>>)>,
    /// Prompt states kept on disk (`--prompt-cache`): each system turn's state, so a later process reads on from it
    /// (none where there is a drafter, whose keys it does not keep).
    disk: Option<crate::engines::disk::DiskCache<super::saved::SavedState>>,
    /// The tokens that open a user turn (`<｜User｜>`, which ends the system turn) and an assistant's.
    user: Option<u32>,
    assistant: Option<u32>,
    eos: u32,
    log: bool,
    /// Reading the experts into RAM while no request waits ([`Warm`]); None once done, or switched off.
    warm: Option<Warm>,
    /// The experts' kernel on the GPUs, for what the other cards hold ([`Pinned`]): filled while idle too.
    kernel: Option<Arc<WgpuExperts>>,
    /// An EXL3 checkpoint's experts' kernel on the GPUs: its slots filled while idle, before RAM is.
    exl3: Option<Arc<WgpuExl3Experts>>,
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
    /// DSpark's drafter, where the checkpoint has one and it is used: each decode pass checks its drafts.
    drafter: Option<crate::dsv41::dspark::Drafter>,
    /// Its keys of the trunk as they were at the checkpoint.
    drafter_keys: Option<Vec<crate::dsv41::attention::AttnState>>,
    /// Drafts a pass checks, no more than `draft_max` ([`Engine::drafts_to_check`]): those the drafter is `draft_min` sure
    /// of all together (the product of their confidences, up to each) where that is set; else as many as make the most
    /// tokens for the time, a greedy reply's by [`DRAFT_ROW_COST`], another's by what passes have taken of late.
    draft_min: Option<f32>,
    draft_max: usize,
    timing: Timing,
    /// Drafted passes in a row that checked none of their drafts, and passes still to make without drafting
    /// ([`DRAFT_PAUSE`]).
    unsure: usize,
    pause: usize,
    /// An incognito request's state (`job.forget`), kept in memory alone (never on disk) for the next incognito request of
    /// its session (or, where it named none, of none): its session, and when it was last used. Wiped when any other
    /// request comes, when its session ends (`job.wipe`), or when it has gone [`crate::engines::job::INCOGNITO_KEPT`] unused.
    private: Option<(Option<String>, std::time::Instant)>,
    /// Its vision tower, where it reads pictures ([`super::vision::Vision`]).
    vision: Option<super::vision::Vision>,
}

/// What a check's row past the first is taken to cost, a step (with its draft) being 1, where a greedy reply's pass
/// picks how many drafts to check ([`Engine::drafts_to_check`]): by the drafter's word and this alone, so a prompt's
/// reply is the same whenever it is asked (a pass's rows are summed apart from a step's, so how the drafts were checked
/// can tip a near tie). Measured on the EXL3 checkpoint here: a step 120-220 ms, each row more 70-110, a check of four
/// some 3.5 steps (its experts that no card holds; three fifths of a step before the step's own calls were cut).
const DRAFT_ROW_COST: f64 = 0.75;
/// Drafted passes in a row that check none of their drafts before drafting pauses ([`DRAFT_PAUSE`] passes), and the
/// pause: the drafter is unsure of what comes, and its own pass is a fifth of a step's.
const DRAFT_UNSURE: usize = 3;
const DRAFT_PAUSE: usize = 6;

/// What a decode pass of each number of rows has taken of late, and a draft: averages weighted to the recent, for
/// [`Engine::drafts_to_check`].
#[derive(Default)]
struct Timing {
    /// By rows (from 1): (ms, passes).
    check: Vec<(f64, u32)>,
    draft: (f64, u32),
}

/// The weight a pass's time has in [`Timing`]'s averages.
const TIMING_NEW: f64 = 0.1;

/// `v` (ms) into an average weighted to the recent.
fn fold(avg: &mut (f64, u32), v: f64) {
    avg.0 = if avg.1 == 0 { v } else { avg.0 + TIMING_NEW * (v - avg.0) };
    avg.1 += 1;
}

impl Timing {
    fn passed(&mut self, rows: usize, check: std::time::Duration) {
        if self.check.len() <= rows {
            self.check.resize(rows + 1, (0.0, 0));
        }
        fold(&mut self.check[rows], check.as_secs_f64() * 1e3);
    }

    fn drafted(&mut self, draft: std::time::Duration) {
        fold(&mut self.draft, draft.as_secs_f64() * 1e3);
    }

    /// A pass of `rows` rows' time: as measured; else (never yet) on the line through those that were (from one alone,
    /// a tenth more a row), so that what is not tried yet is weighed too; None before any pass.
    fn check_ms(&self, rows: usize) -> Option<f64> {
        if let Some(c) = self.check.get(rows).filter(|c| c.1 > 0) {
            return Some(c.0);
        }
        let known: Vec<(f64, f64)> = self.check.iter().enumerate().filter(|(_, c)| c.1 > 0).map(|(r, c)| (r as f64, c.0)).collect();
        let n = known.len() as f64;
        let (mr, mt) = (known.iter().map(|k| k.0).sum::<f64>() / n, known.iter().map(|k| k.1).sum::<f64>() / n);
        let slope = match known.len() {
            0 => return None,
            1 => mt / 10.0,
            _ => {
                let (sxy, sxx) = known.iter().fold((0.0, 0.0), |(sxy, sxx), &(r, t)| (sxy + (r - mr) * (t - mt), sxx + (r - mr) * (r - mr)));
                sxy / sxx
            }
        };
        Some((mt + slope * (rows as f64 - mr)).max(0.0))
    }

    fn line(&self) -> String {
        let checks: Vec<String> = self.check.iter().enumerate().filter(|(_, c)| c.1 > 0).map(|(r, c)| format!("{r}: {:.0} ms ({})", c.0, c.1)).collect();
        format!("a draft {:.1} ms, a check of rows {}", self.draft.0, checks.join(", "))
    }
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

/// Where `prompt`'s system turn ends (its first `user` token), when its state is worth keeping there: a conversation's
/// first prompt (no `assistant` turn but the one its reply opens), a turn of [`SYSTEM_KEPT_FROM`] tokens or more, past
/// `start` (what a state already serves) and before `last` (the token the reply's logits come from). A conversation's
/// later prompts keep none: an agent's run that read a guide into its instructions would leave the state of those
/// instructions in place of the ones its next run starts with.
fn system_end(prompt: &[u32], user: Option<u32>, assistant: Option<u32>, start: usize, last: usize) -> Option<usize> {
    let end = prompt.iter().position(|&t| Some(t) == user)?;
    let first = prompt.iter().filter(|&&t| Some(t) == assistant).count() <= 1;
    (first && end >= SYSTEM_KEPT_FROM && end > start && end < last).then_some(end)
}

impl Engine {
    pub(crate) fn new(model: crate::dsv41::model::Model, tok: Arc<crate::dsv41::tokenizer::Tokenizer>, log: bool) -> Engine {
        let eos = model.cfg.eos_token_id;
        let (user, assistant) = (tok.special(crate::dsv41::chat::USER), tok.special(crate::dsv41::chat::ASSISTANT));
        // (OAIY_DSV41_NO_WARM: leave the drive alone between requests: neither the idle reading nor the rebalance,
        // which reads a record for each expert a card gives up)
        let idle = std::env::var_os("OAIY_DSV41_NO_WARM").is_none();
        let warm = idle.then(|| Warm { cursor: 0, started: None, reading: std::time::Duration::ZERO, bytes_before: model.expert_cache().stats().bytes_read });
        Engine {
            model,
            tok,
            covered: Vec::new(),
            checkpoint: None,
            system: None,
            disk: None,
            user,
            assistant,
            eos,
            log,
            warm,
            kernel: None,
            exl3: None,
            rebalanced: 0,
            idle,
            usage: None,
            order: None,
            ram_gb: 0,
            fitted: None,
            drafter: None,
            drafter_keys: None,
            draft_min: None,
            draft_max: 0,
            timing: Timing::default(),
            unsure: 0,
            pause: 0,
            private: None,
            vision: None,
        }
    }

    /// Nothing kept of the prompts read: the state, the last prompt's checkpoint and the system turn's.
    fn forget_state(&mut self) {
        self.covered.clear();
        self.checkpoint = None;
        self.system = None;
        self.private = None;
        if let Some(v) = &mut self.vision {
            v.forget();
        }
    }

    /// Its vision tower: a prompt's pictures read through it ([`super::vision::Vision`]).
    pub(crate) fn with_vision(mut self, vision: Option<super::vision::Vision>) -> Engine {
        self.vision = vision;
        self
    }

    /// DSpark's drafter: each decode pass checks up to `max` of its drafts, those it is `min` sure of together where
    /// that is given, else as many as pay ([`Engine::drafts_to_check`]).
    pub(crate) fn with_drafter(mut self, drafter: Option<crate::dsv41::dspark::Drafter>, min: Option<f32>, max: usize) -> Engine {
        self.draft_max = drafter.as_ref().map_or(0, |d| max.min(d.block()));
        self.drafter = drafter;
        self.draft_min = min;
        self
    }

    /// How many of a draft whose drafts the drafter is `confidence` sure of (each given those before) a pass checks: of
    /// up to `draft_max`, those it is `draft_min` sure of together where that is set; else the number that makes the
    /// most tokens for the time, a pass's tokens expected (the one chosen, and each draft by the product of the
    /// confidences up to it: DSpark's confidence is its estimate that the model takes it) over its time: a greedy
    /// reply's by [`DRAFT_ROW_COST`] a row, another's a draft's and a check of that many rows as they have taken of late
    /// (a check of more rows reads more experts, so what pays is the machine's and the subject's).
    fn drafts_to_check(&self, confidence: &[f32], greedy: bool) -> usize {
        let most = confidence.len().min(self.draft_max);
        if let Some(min) = self.draft_min {
            let mut sure = 1.0f32;
            return confidence[..most]
                .iter()
                .take_while(|&&c| {
                    sure *= c;
                    sure >= min
                })
                .count();
        }
        let (mut best, mut best_rate) = (0, f64::MIN);
        let (mut expected, mut sure) = (1.0f64, 1.0f64);
        for k in 0..=most {
            if k > 0 {
                sure *= f64::from(confidence[k - 1]);
                expected += sure;
            }
            let time = if greedy {
                1.0 + DRAFT_ROW_COST * k as f64
            } else {
                // (before any pass: all of them, which a pass then measures)
                let Some(check) = self.timing.check_ms(k + 1) else { return most };
                self.timing.draft.0 + check
            };
            let rate = expected / time.max(1e-3);
            if rate > best_rate {
                (best, best_rate) = (k, rate);
            }
        }
        best
    }

    /// An EXL3 checkpoint's experts' kernel the model was given: the idle reading fills its slots first (in the
    /// profile's order where there is one), and leaves what they hold out of RAM.
    pub(crate) fn with_exl3(mut self, exl3: Option<Arc<WgpuExl3Experts>>) -> Engine {
        self.exl3 = exl3;
        self
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

    /// Prompt states kept on disk in `disk` ([`super::saved::SavedState`]).
    pub(crate) fn with_disk(mut self, disk: Option<crate::engines::disk::DiskCache<super::saved::SavedState>>) -> Engine {
        self.disk = disk;
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
        // (an EXL3 checkpoint's cards filled while idle only in a usage profile's order: by number they would hold a
        // quarter of what a step asks for by chance, and for good; without one, a decode step's reads fill them)
        let exl3 = self.exl3.as_deref();
        let order = self.order.as_deref();
        if order.is_some() && exl3.is_some_and(|k| k.fill_some(self.model.expert_store().as_ref(), self.model.expert_cache(), order, WARM_SLICE)) {
            w.reading += slice.elapsed();
            return;
        }
        let elsewhere = |l: u32, e: u32| kernel.is_some_and(|k| k.elsewhere(l, e)) || exl3.is_some_and(|k| k.elsewhere(l, e));
        let more = match &self.order {
            Some(order) => self.model.warm_experts_in(order, &mut w.cursor, WARM_SLICE, &elsewhere),
            None => self.model.warm_experts(&mut w.cursor, WARM_SLICE, &elsewhere),
        };
        w.reading += slice.elapsed();
        if !more {
            if self.log {
                let c = self.model.expert_cache();
                let read = c.stats().bytes_read.saturating_sub(w.bytes_before);
                let pinned = kernel.map_or(0, |k| k.pinned()) + exl3.map_or(0, |k| k.tier().1);
                eprintln!(
                    "  the experts' RAM tier holds {} of its {}{}: {:.1} s of idle reading ({:.1} GB read into RAM since the model loaded)",
                    c.len(),
                    c.n_slots(),
                    if pinned > 0 { format!(", the {}GPUs {pinned} more for good", if exl3.is_some() { "" } else { "other " }) } else { String::new() },
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

    pub(crate) fn run(mut self, jobs: std::sync::mpsc::Receiver<crate::engines::job::Job>) {
        use crate::engines::job::{Event, Finish};
        use std::sync::mpsc::TryRecvError;
        loop {
            self.fit_ram();
            // (an incognito request's state unused for long enough: wiped)
            if self.private.as_ref().is_some_and(|(_, at)| at.elapsed() >= crate::engines::job::INCOGNITO_KEPT) {
                self.forget_state();
            }
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
                        // (waiting no longer than an incognito state is kept: it is wiped on time)
                        match self.private.as_ref().map(|(_, at)| crate::engines::job::INCOGNITO_KEPT.saturating_sub(at.elapsed())) {
                            Some(left) => match jobs.recv_timeout(left) {
                                Ok(job) => job,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                            },
                            None => match jobs.recv() {
                                Ok(job) => job,
                                Err(_) => break,
                            },
                        }
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            };
            // The end of an incognito session: what is kept of it is wiped.
            if job.wipe {
                if self.private.as_ref().is_some_and(|(session, _)| *session == job.session) {
                    self.forget_state();
                }
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                continue;
            }
            // An incognito request's state is read on from by the next incognito request of its session alone.
            if self.private.as_ref().is_some_and(|(session, _)| !serves(session, job.forget, &job.session)) {
                self.forget_state();
            }
            // (an incognito request leaves the experts' counts as it found them: the profile keeps nothing of it)
            let before = job.forget.then(|| self.model.expert_uses().counts());
            if let Err(e) = self.generate(&job) {
                // The state's extent is unknown after a failure: start clean.
                self.forget_state();
                let _ = job.events.send(Event::Error(e.to_string()));
            }
            // Incognito: kept in memory alone, for its session's next request ([`crate::engines::job::INCOGNITO_KEPT`] at most unused).
            if job.forget {
                self.private = Some((job.session.clone(), std::time::Instant::now()));
            }
            match before {
                Some(counts) => drop(self.model.expert_uses().seed(&counts)),
                None => self.save_usage(),
            }
        }
    }

    /// `ids` at `pos..` through the model, the last row's logits; the drafter, where there is one, given its keys of
    /// their rows (it reads the layers `targets` of the trunk).
    fn read(&mut self, ids: &[u32], pos: usize, targets: &[usize]) -> oaiy_engine::Result<Vec<f32>> {
        let Some(d) = self.drafter.as_mut() else { return self.model.forward(ids, pos, &mut |_, _| {}) };
        // (its keys of the last window's rows are all it keeps)
        let from = ids.len().saturating_sub(d.config().window_size);
        let (logits, features) = self.model.forward_rows(ids, pos, targets, false, from)?;
        d.observe(&features, ids.len() - from, pos + from);
        Ok(logits)
    }

    fn generate(&mut self, job: &crate::engines::job::Job) -> oaiy_engine::Result<()> {
        use crate::engines::job::{sample, Event, Finish};
        use std::sync::atomic::Ordering;
        // A prompt's pictures: each its span of rows from the tower (its ViT and aligner), the same picture's kept from
        // when it was last read; and the prompt as states know it, its pictures' tokens told apart by their bytes
        // (`super::vision::keyed`), so a state is read on from only for the same pictures.
        let spans = match (&mut self.vision, job.images.is_empty()) {
            (_, true) => Vec::new(),
            (None, false) => return Err(oaiy_engine::Error::Arg("this DeepSeek-V4.1 reads no pictures here: no vision tower is loaded for it (a GGUF's comes from its checkpoint's folder, --vision-projector NAME=DIR)".into())),
            (Some(vision), false) => {
                let encoding = std::time::Instant::now();
                let spans = job.images.iter().map(|img| Ok(crate::dsv41::model::ImageSpan { start: img.start, rows: vision.span(img.prep.deepseek()?, img.hash) })).collect::<oaiy_engine::Result<Vec<_>>>()?;
                if self.log {
                    eprintln!("  {} picture{} through the vision tower in {:.1}s ({} tokens)", spans.len(), if spans.len() == 1 { "" } else { "s" }, encoding.elapsed().as_secs_f64(), job.images.iter().map(|i| i.prep.n_tokens()).sum::<usize>());
                }
                spans
            }
        };
        self.model.set_images(spans);
        let keyed = (!job.images.is_empty()).then(|| super::vision::keyed(&job.prompt, &job.images));
        let keys: &[u32] = keyed.as_deref().unwrap_or(&job.prompt);
        let prompt = &job.prompt;
        let max_seq = self.model.max_seq();
        if prompt.is_empty() || prompt.len() >= max_seq {
            return Err(oaiy_engine::Error::Arg(format!("a prompt of {} tokens does not fit a context of {max_seq}", prompt.len())));
        }
        let started = std::time::Instant::now();
        // OAIY_DSV41_PROFILE: where the prompt's and the reply's time went (`crate::dsv41::profile`), and what each read
        let profiling = self.log && std::env::var_os("OAIY_DSV41_PROFILE").is_some();
        let (kernel, exl3) = (self.kernel.clone(), self.exl3.clone());
        let exl3_waits = self.exl3.clone();
        // (an EXL3 checkpoint's cards: their experts computed from slots as the first card's tier's hits, those put
        // through the scratch slots as its misses)
        let read_so_far = move |m: &crate::dsv41::model::Model| {
            let st = m.expert_cache().stats();
            let tier = match (&kernel, &exl3) {
                (Some(k), _) => {
                    let (_, hits, misses, admitted) = k.tier();
                    (hits, misses, admitted, k.share_hits())
                }
                // (the other cards' share: what was computed from slots on a card past the first)
                (None, Some(k)) => {
                    let (held, scratch, taken, kept) = k.counts();
                    (held, scratch, taken + kept, k.held_past_first())
                }
                (None, None) => (0, 0, 0, 0),
            };
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
                crate::dsv41::profile::take_line(),
                ggml_rs_wgpu::profile::take_dense_line()
            );
            // (OAIY_CHAIN_PROFILE: the chained recordings' kernels on the GPU, the most first; OAIY_PIECE_STAMPS: their
            // pieces' busy time against their span)
            let kernels: Vec<String> = ggml_rs_wgpu::profile::take_kernels().into_iter().take(40).map(|(name, ms, n)| format!("{name} {ms:.0} ms ({n})")).collect();
            if !kernels.is_empty() {
                eprintln!("    {label}'s kernels: {}", kernels.join(", "));
            }
            let (busy, span, pieces) = ggml_rs_wgpu::profile::take_pieces();
            let chains = ggml_rs_wgpu::profile::take_line();
            if pieces > 0 {
                eprintln!("    {label}'s pieces: {pieces}, busy {busy:.0} ms of a {span:.0} ms span; {chains}");
            } else {
                eprintln!("    {label}'s chains: {chains}");
            }
            // (each card's held-experts calls as the host found them once it had made the rest of a layer's experts)
            if let Some(k) = &exl3_waits {
                let [a, b] = k.take_held_waits();
                eprintln!(
                    "    {label}'s held experts by card, once the host had the rest: the first's {} of {} calls still running, waited for {:.0} us on average; the second's {} of {}, {:.0} us",
                    a.2, a.1, a.0, b.2, b.1, b.0
                );
                let [p, q] = k.take_puts();
                eprintln!(
                    "    {label}'s puts into scratch slots by card: the first's {} at {:.0} us each ({:.1} GB/s), {} more held there still; the second's {} at {:.0} us ({:.1} GB/s), {} more",
                    p.0, p.1, p.2, p.3, q.0, q.1, q.2, q.3
                );
            }
        };
        if profiling {
            crate::dsv41::profile::take();
        }
        let at_start = read_so_far(&self.model);
        let common = self.covered.iter().zip(keys).take_while(|(a, b)| a == b).count();
        // The live state, the last prompt's checkpoint or the state at the end of a system turn: whichever serves more
        // of this prompt.
        let live = reusable(&self.covered, keys);
        let kept = self.checkpoint.as_ref().map_or(0, |(tokens, _)| reusable(tokens, keys));
        // (a state on disk that serves more of it than these, a system turn's an earlier process kept: read in as the
        // system turn's)
        let mut from_disk = false;
        if let (Some(disk), None, false) = (self.disk.as_mut(), &self.drafter, job.forget) {
            let held = live.max(kept).max(self.system.as_ref().map_or(0, |(tokens, ..)| reusable(tokens, keys)));
            let keys: Vec<u64> = keys.iter().map(|&t| u64::from(t)).collect();
            if let Some((keys, saved)) = disk.load_best(&keys, prompt.len() - 1, held, self.log).filter(|(_, saved)| self.model.fits(&saved.state)) {
                self.system = Some((keys.iter().map(|&k| k as u32).collect(), saved.state, None));
                from_disk = true;
            }
        }
        let system = self.system.as_ref().map_or(0, |(tokens, ..)| reusable(tokens, keys));
        let (start, source) = if kept > live && kept >= system {
            let (tokens, state) = self.checkpoint.as_ref().expect("a checkpoint serves it");
            self.model.restore(state);
            self.covered.clone_from(tokens);
            if let (Some(d), Some(keys)) = (self.drafter.as_mut(), &self.drafter_keys) {
                d.set_keys(keys);
            }
            (kept, "memory")
        } else if system > live {
            let (tokens, state, keys) = self.system.as_ref().expect("a system turn's state serves it");
            self.model.restore(state);
            self.covered.clone_from(tokens);
            if let (Some(d), Some(keys)) = (self.drafter.as_mut(), keys) {
                d.set_keys(keys);
            }
            (system, if from_disk { "disk" } else { "memory" })
        } else {
            (live, if live > 0 { "memory" } else { "none" })
        };
        if start == 0 {
            self.covered.clear();
        }
        let _ = job.events.send(Event::CacheReuse { cached: start, source, common });
        let total = prompt.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });
        // All but the last token, then the checkpoint, then the last one (whose logits the reply starts from). Where there
        // is a drafter, each pass's rows give it its keys of the trunk.
        let targets: Vec<usize> = self.drafter.as_ref().map(|d| d.targets().to_vec()).unwrap_or_default();
        let last = prompt.len() - 1;
        // (a conversation's first long system turn read up to its end first, its state kept there: incognito, in memory
        // alone, as its other states are)
        let system_end = system_end(prompt, self.user, self.assistant, start, last);
        let mut pos = start;
        while pos < last {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                return Ok(());
            }
            let take = PREFILL_CHUNK.min(last - pos).min(system_end.filter(|&end| end > pos).map_or(usize::MAX, |end| end - pos));
            self.read(&prompt[pos..pos + take], pos, &targets)?;
            self.covered.extend_from_slice(&keys[pos..pos + take]);
            pos += take;
            let _ = job.events.send(Event::Progress { done: pos - start, total });
            if system_end == Some(pos) {
                let state = Arc::new(self.model.checkpoint());
                // (and on disk, for a later process: written by the cache's own thread, the state shared; never an
                // incognito request's)
                if let (Some(disk), None, false) = (self.disk.as_mut(), &self.drafter, job.forget) {
                    let keys: Vec<u64> = self.covered.iter().map(|&t| u64::from(t)).collect();
                    let saved = super::saved::SavedState::new(pos, Arc::clone(&state));
                    if !disk.has(&keys) && disk.takes(crate::engines::disk::PromptState::encoded_len(&saved) as u64) {
                        disk.save(keys, saved, true);
                    }
                }
                self.system = Some((self.covered.clone(), state, self.drafter.as_ref().map(|d| d.keys())));
            }
        }
        // (an incognito request's too: in memory alone, wiped as its state is)
        self.checkpoint = Some((self.covered.clone(), self.model.checkpoint()));
        self.drafter_keys = self.drafter.as_ref().map(|d| d.keys());
        let mut logits = self.read(&prompt[last..], last, &targets)?;
        self.covered.push(keys[last]);
        // (the pictures' rows read: none kept for the reply's steps)
        self.model.set_images(Vec::new());
        pos = prompt.len();
        let _ = job.events.send(Event::Progress { done: total, total });
        let prefill_s = started.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });
        let at_reply = read_so_far(&self.model);
        if profiling {
            part("the prompt", at_start, at_reply);
        }

        let think_start = self.tok.special(crate::dsv41::chat::THINK_START);
        let mut budget = crate::engines::glm::ThinkBudget::new(think_start.is_some() && prompt.last().copied() == think_start, self.tok.special(crate::dsv41::chat::THINK_END), job.think_budget);
        let mut parser = crate::dsv41::chat::StreamParser::new(if budget.thinking { crate::dsv41::chat::Mode::Thinking } else { crate::dsv41::chat::Mode::Chat });
        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        // (the tokens given out so far, read beside `give`, which counts them)
        let (mut pending, n) = (Vec::<u8>::new(), std::cell::Cell::new(0usize));
        // The token chosen after `logits` (the thinking budget's say included).
        let mut choose = |logits: &[f32]| -> u32 {
            let was_thinking = budget.thinking;
            let (next, _forced) = budget.pass(sample(logits, &job.sampling, &mut rng));
            if was_thinking && (!budget.thinking || budget.used.is_multiple_of(crate::engines::glm::THINKING_EVERY)) {
                let _ = job.events.send(Event::Thinking { used: budget.used, budget: budget.budget, done: !budget.thinking });
            }
            next
        };
        // `next` given out, to be at `pos`: how the reply ends there, if it does.
        let (tok, eos) = (Arc::clone(&self.tok), self.eos);
        let mut give = |next: u32, pos: usize| -> Option<Finish> {
            if next == eos {
                return Some(Finish::Stop);
            }
            n.set(n.get() + 1);
            // Whole UTF-8 characters only: a token can be part of one.
            pending.extend_from_slice(tok.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                parser.push(&text);
                if job.events.send(Event::Text(text)).is_err() {
                    return Some(Finish::Stop); // nobody is listening
                }
            }
            // A complete tool call is the reply: the caller runs it and comes back with its result.
            if parser.tool_calls_ready() {
                return Some(Finish::Stop);
            }
            if n.get() >= job.max_tokens || pos + 1 >= max_seq {
                return Some(Finish::Length);
            }
            if job.cancel.load(Ordering::Relaxed) {
                return Some(Finish::Stop);
            }
            None
        };
        let decode = std::time::Instant::now();
        let (mut passes, mut drafted, mut taken) = (0usize, 0usize, 0usize);
        // (a reply's drafting as its own tokens have it, whatever the last reply's did)
        (self.unsure, self.pause) = (0, 0);
        let mut next = choose(&logits);
        let finish = loop {
            if let Some(finish) = give(next, pos) {
                break finish;
            }
            // Drafts after it, where there is a drafter: as many as it is confident enough of and the reply has room for.
            let room = (max_seq - pos - 1).min(job.max_tokens.saturating_sub(n.get()));
            let drafting = std::time::Instant::now();
            let paused = self.pause > 0;
            self.pause = self.pause.saturating_sub(1);
            let drafts = match self.drafter.as_mut() {
                Some(d) if room > 0 && !paused => {
                    let draft = d.draft(self.model.embed(), self.model.head(), next, pos)?;
                    let k = self.drafts_to_check(&draft.confidence, job.sampling.temperature <= 0.0).min(room);
                    // (unsure of every draft some passes in a row: a few passes without drafting)
                    self.unsure = if k == 0 { self.unsure + 1 } else { 0 };
                    if self.unsure >= DRAFT_UNSURE {
                        (self.unsure, self.pause) = (0, DRAFT_PAUSE);
                    }
                    draft.tokens[..k].to_vec()
                }
                _ => Vec::new(),
            };
            let drafted_in = drafting.elapsed();
            if self.drafter.is_none() {
                logits = self.model.forward(&[next], pos, &mut |_, _| {})?;
                self.covered.push(next);
                pos += 1;
                next = choose(&logits);
                continue;
            }
            // The token and its drafts in one pass, every row's choice made: the drafts stand up to the first the
            // model chose otherwise, whose choice is the next token; the caches are rewound past the rest.
            let rows: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
            let checking = std::time::Instant::now();
            let saved = self.model.save(pos, rows.len());
            let (all, features) = self.model.forward_rows(&rows, pos, &targets, true, 0)?;
            let checked_in = checking.elapsed();
            self.timing.passed(rows.len(), checked_in);
            let vocab = all.len() / rows.len();
            let mut kept = 1;
            let mut ended = None;
            next = choose(&all[..vocab]);
            while kept <= drafts.len() && next == drafts[kept - 1] {
                kept += 1;
                if let Some(finish) = give(next, pos + kept - 1) {
                    ended = Some(finish);
                    break;
                }
                next = choose(&all[(kept - 1) * vocab..kept * vocab]);
            }
            self.model.rewind(&saved, kept);
            if let Some(d) = self.drafter.as_mut() {
                d.observe(&features[..kept * features.len() / rows.len()], kept, pos);
            }
            self.covered.extend_from_slice(&rows[..kept]);
            pos += kept;
            if !paused {
                self.timing.drafted(drafted_in);
                passes += 1;
                drafted += drafts.len();
                taken += kept - 1;
            }
            if let Some(finish) = ended {
                break finish;
            }
        };
        if passes > 0 && self.log {
            eprintln!("  drafts: {taken} of {drafted} taken in {passes} passes ({:.2} tokens a pass); {}", (passes + taken) as f64 / passes as f64, self.timing.line());
        }
        if !pending.is_empty() {
            let _ = job.events.send(Event::Text(String::from_utf8_lossy(&pending).into_owned()));
        }
        if self.log {
            let decode_s = decode.elapsed().as_secs_f64();
            let n = n.get();
            eprintln!("  {} prompt tokens ({start} reused) in {prefill_s:.1}s; {n} generated in {decode_s:.1}s ({:.2} tok/s)", prompt.len(), n as f64 / decode_s.max(1e-9));
        }
        if profiling {
            part("the reply", at_reply, read_so_far(&self.model));
        }
        let _ = job.events.send(Event::Done { finish, completion_tokens: n.get() });
        Ok(())
    }
}

/// Whether the state an incognito request left (`held`: its session, None where it named none) may serve a request
/// (`forget`: incognito; `session`): the next incognito request of the same session alone.
fn serves(held: &Option<String>, forget: bool, session: &Option<String>) -> bool {
    forget && held == session
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_incognito_state_serves_its_own_sessions_next_incognito_request_alone() {
        let (a, b) = (Some("project-a".to_string()), Some("project-b".to_string()));
        assert!(serves(&a, true, &a), "the session's next step reads on");
        assert!(serves(&None, true, &None), "an incognito request that named no session, the next that names none");
        assert!(!serves(&a, true, &b), "another session's");
        assert!(!serves(&a, true, &None), "one with no session");
        assert!(!serves(&None, true, &a));
        assert!(!serves(&a, false, &a), "a request that is not incognito");
        assert!(!serves(&None, false, &None));
    }

    const USER: u32 = 9;
    const ASSISTANT: u32 = 8;

    /// A conversation's first prompt: `system` tokens, then a user turn (`USER`) of `rest`, then the reply's opening
    /// (`ASSISTANT`).
    fn prompt(system: usize, rest: usize) -> Vec<u32> {
        let mut p = vec![1; system];
        p.push(USER);
        p.extend(std::iter::repeat_n(2, rest));
        p.push(ASSISTANT);
        p
    }

    fn end(p: &[u32], start: usize) -> Option<usize> {
        system_end(p, Some(USER), Some(ASSISTANT), start, p.len() - 1)
    }

    #[test]
    fn a_long_system_turns_state_is_kept_at_its_end() {
        let p = prompt(20000, 40);
        assert_eq!(end(&p, 0), Some(20000), "at the user turn's token");
        assert_eq!(end(&p, 20000), None, "not where a state already serves it");
        assert_eq!(end(&p, 20010), None);
        assert_eq!(end(&prompt(100, 40), 0), None, "a short turn is read again");
        let mut bare = vec![1; 20000];
        bare.push(USER);
        assert_eq!(end(&bare, 0), None, "not the token the reply's logits come from");
        assert_eq!(system_end(&p, None, Some(ASSISTANT), 0, p.len() - 1), None, "a tokenizer without the token");
        assert_eq!(end(&[vec![1; 20000], vec![ASSISTANT]].concat(), 0), None, "a prompt with no user turn");
    }

    #[test]
    fn a_conversations_later_prompts_keep_no_system_state() {
        // (its first step's reply and a tool's result, then the next step's opening: and the same with instructions a
        // guide grew, which must not take the place of the ones a new run starts with)
        let mut later = prompt(20000, 40);
        later.extend([5, 5, USER, 2, 2, ASSISTANT]);
        assert_eq!(end(&later, 0), None);
        let mut grown = vec![1; 22000];
        grown.extend_from_slice(&later[20000..]);
        assert_eq!(end(&grown, 0), None);
    }

    #[test]
    fn a_new_conversation_with_the_same_system_turn_reads_on_from_its_state() {
        let (first, second) = (prompt(20000, 40), prompt(20000, 3));
        let kept = first[..20000].to_vec();
        assert_eq!(reusable(&kept, &second), 20000);
        // (the last prompt's checkpoint, a conversation of its own, serves none of it)
        assert_eq!(reusable(&first[..first.len() - 1], &second), 0);
    }
}
