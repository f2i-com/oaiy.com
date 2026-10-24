//! An EXL3 checkpoint's routed experts on the WebGPU adapters ([`WgpuExl3Experts`]): slots on every card the model may
//! use (`ggml_rs_wgpu::exl3::Exl3Slots`, a pool for each bitrate its layers have: DeepSeek-V4.1's EXL3 quantizations
//! keep a few layers at 2 bits where the rest are 3), each slot an expert put in from its record as the drive or the
//! RAM tier gives it. What the cards hold they compute. Every other expert a pass asks for goes through scratch slots
//! on every card, its record written into the card's memory in place by the host's threads (`Exl3Slots::put_many`),
//! the cards given them by how fast each has taken them in (a card on four lanes at half the rate of one on eight).
//!
//! What the cards hold follows what is used: every expert's uses are counted (halved every
//! [`crate::dsv41::moe::USES_AGE_STEPS`] decode steps, as the model's own counts are), and an expert that went through
//! a scratch slot is kept in a free slot of its card, or in place of the least used expert there when it is used more,
//! by a copy of the card's own memory (`Exl3Slots::copy_slot`), its upload already made. A usage profile fills the
//! slots while the server idles, the most used first.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;

use ggml_rs_wgpu::exl3::{Exl3Slots, ProjAt, RecordLayout};

use super::*;
use crate::dsv41::expert::{Exl3Proj, ExpertFormat, ExpertJob, ExpertsKernel};
use crate::dsv41::formats::to_bf16;

/// Scratch slots a bitrate has on each card: a pass's experts not held go through them, a prompt's a group
/// ([`crate::dsv41::moe::GPU_GROUP`]) at a time.
const SCRATCH: usize = crate::dsv41::moe::GPU_GROUP;

/// What the weight budget keeps free on a card beyond the slots: a prompt's passes' own buffers.
const MARGIN: u64 = 1 << 30;

/// The rate a card is taken to put records into its scratch slots at before it has (bytes a microsecond: 10 GB/s).
const RATE_GUESS: f64 = 10_000.0;

/// Uses an expert needs before it takes the place of one the cards hold (a free slot takes any): one use is a step's
/// chance, and swapping on it would trade one passing expert for another.
const KEEP_USES: u32 = 2;

/// Threads reading the records an idle fill puts on the cards: a drive serves a queue of reads several times faster than
/// one at a time.
const FILL_READERS: usize = 8;

/// Slots of one bitrate on one card.
struct Pool {
    slots: Exl3Slots,
    /// The bitrate's words a tile: the gate's and up's, the down's.
    tw: [usize; 2],
    /// Scratch (a pass's experts pass through), or held experts.
    scratch: bool,
    /// Which of the cards it is on.
    card: usize,
    /// One call in the scratch slots at a time.
    busy: Mutex<()>,
}

/// Where the held experts are: each one's `(pool, slot)`, each pool's slots' experts, and each pool's free slots.
struct Placement {
    at: HashMap<(u32, u32), (usize, usize)>,
    keys: Vec<Vec<Option<(u32, u32)>>>,
    free: Vec<usize>,
}

/// What the tier has done since the model loaded: experts computed from their slots, experts put through the scratch
/// slots, records taken into a free slot from the RAM tier or the drive, and experts kept from a scratch slot.
#[derive(Default)]
struct Counts {
    held: AtomicU64,
    scratch: AtomicU64,
    taken: AtomicU64,
    kept: AtomicU64,
}

pub(crate) struct WgpuExl3Experts {
    /// Each layer's record layout.
    layouts: Vec<RecordLayout>,
    pools: Vec<Pool>,
    placed: RwLock<Placement>,
    /// Every expert's uses of late (`layer * per_layer + expert`), and the calls since they were last halved.
    uses: Vec<AtomicU32>,
    per_layer: usize,
    calls: AtomicU64,
    /// Each card's rate of putting records into its scratch slots, as measured (bytes a microsecond, f64 bits).
    rates: Vec<AtomicU64>,
    /// Where the idle fill ([`WgpuExl3Experts::fill_some`]) is in its order.
    cursor: Mutex<usize>,
    counts: Counts,
    /// Experts kept from a scratch slot since the caller last asked ([`ExpertsKernel::offer`]): RAM may let them go.
    kept: Mutex<Vec<(u32, u32)>>,
}

fn record_layout(l: &[Exl3Proj; 3], bytes: usize) -> RecordLayout {
    let at = |p: &Exl3Proj| ProjAt { suh: p.suh, svh: p.svh, mul1: p.mul1, trellis: p.trellis, k: p.k, n: p.n, tile_words: p.tile_words };
    RecordLayout { bytes, proj: [at(&l[0]), at(&l[1]), at(&l[2])] }
}

/// A slot's bytes on a card for `hidden -> ff` experts at `tw` words a tile (as `Exl3Slots` counts them).
fn slot_bytes(hidden: usize, ff: usize, tw: [usize; 2]) -> u64 {
    let mwords = |k: usize, n: usize, tw: usize| k / 16 * n / 16 * (tw / 2);
    (2 * mwords(hidden, ff, tw[0]) * 4 + mwords(ff, hidden, tw[1]) * 4 + 3 * (hidden + ff) * 4) as u64
}

/// `y`'s rows (`[rows, DIM]`, an expert's output) weighted by their routes' `weights` and rounded to bf16, as the
/// CPU's experts give theirs.
fn weighed(mut y: Vec<f32>, weights: &[f32]) -> Vec<f32> {
    let d = crate::dsv41::expert::DIM;
    for (row, &w) in y.chunks_exact_mut(d).zip(weights) {
        for v in row {
            *v = to_bf16(*v * w);
        }
    }
    y
}

/// Rows of an expert's job that go as jobs of a row each ([`apart`]): a draft's check's few.
const ROWS_APART: usize = 8;

/// `(slot, x)` as the slots' jobs: a job of each row where `x` (rows `width` wide) is a few rows ([`ROWS_APART`] or
/// fewer), so each row's sums are a decode step's (the kernel of a row, its splits), and a check's experts give what a
/// step's would; else one job of them all (a prompt's, each tile decoded once for a block of its rows). A few rows went
/// a prompt's way before: a block of 32 row places for one or two rows, unsplit, its sums another order than a step's.
fn apart(slot: usize, x: &[f32], width: usize) -> Vec<(usize, &[f32])> {
    if pieces(x, width) > 1 {
        x.chunks_exact(width).map(|r| (slot, r)).collect()
    } else {
        vec![(slot, x)]
    }
}

/// How many jobs [`apart`] makes of `x`.
fn pieces(x: &[f32], width: usize) -> usize {
    let rows = x.len() / width;
    if (2..=ROWS_APART).contains(&rows) {
        rows
    } else {
        1
    }
}

/// A layout's projections as the CPU's decoder takes them.
fn layout_proj(l: &RecordLayout) -> [Exl3Proj; 3] {
    l.proj.map(|p| Exl3Proj { suh: p.suh, svh: p.svh, mul1: p.mul1, trellis: p.trellis, k: p.k, n: p.n, tile_words: p.tile_words })
}

impl WgpuExl3Experts {
    /// Slots on `cards` (the model's first card first, the trunk on it already) for a model of `per_layer` experts a
    /// layer: scratch slots on each, then on every card what its weight budget has left but [`MARGIN`], shared by the
    /// bitrates by how many layers each has, in pools of what a binding holds. None if `format` is not EXL3's or not
    /// one slot fits.
    pub(crate) fn new(format: &ExpertFormat, cards: &[Arc<WgpuBackend>], per_layer: usize) -> Option<WgpuExl3Experts> {
        Self::new_within(format, cards, per_layer, None)
    }

    /// [`Self::new`] with no more than `most` slots for held experts, the first card's room taken first (DSpark's
    /// drafter's 384 experts, all held, before the trunk's take the rest).
    pub(crate) fn new_within(format: &ExpertFormat, cards: &[Arc<WgpuBackend>], per_layer: usize, most: Option<usize>) -> Option<WgpuExl3Experts> {
        let ExpertFormat::Exl3 { layers, bytes } = format else { return None };
        let mut left = most.unwrap_or(usize::MAX);
        let first = layers.first()?;
        let (hidden, ff) = (first[0].k, first[0].n);
        let layouts: Vec<RecordLayout> = layers.iter().map(|l| record_layout(l, *bytes)).collect();
        let mut classes: Vec<([usize; 2], usize)> = Vec::new();
        for l in layers {
            let tw = [l[0].tile_words, l[2].tile_words];
            match classes.iter_mut().find(|(t, _)| *t == tw) {
                Some(c) => c.1 += 1,
                None => classes.push((tw, 1)),
            }
        }
        let mut pools = Vec::new();
        // (a tier that holds all of its experts puts few through scratch slots: a few of them)
        let scratch = if most.is_some() { 4 } else { SCRATCH };
        for (c, card) in cards.iter().enumerate() {
            for &(tw, _) in &classes {
                if let Some(slots) = card.exl3_slots_mapped(scratch, hidden, ff, tw) {
                    pools.push(Pool { slots, tw, scratch: true, card: c, busy: Mutex::new(()) });
                }
            }
        }
        for (c, card) in cards.iter().enumerate() {
            let (used, budget) = card.usage();
            let room = budget.saturating_sub(used).saturating_sub(MARGIN);
            for &(tw, n) in &classes {
                let per = slot_bytes(hidden, ff, tw);
                let share = most.map_or(usize::MAX, |most| most * n / layers.len());
                let mut want = ((room as f64 * n as f64 / layers.len() as f64 / per as f64) as usize).min(share).min(left);
                while want > 0 {
                    let Some(slots) = card.exl3_slots(want, hidden, ff, tw) else { break };
                    want = want.saturating_sub(slots.slots());
                    left = left.saturating_sub(slots.slots());
                    pools.push(Pool { slots, tw, scratch: false, card: c, busy: Mutex::new(()) });
                }
            }
        }
        if !pools.iter().any(|p| !p.scratch) {
            return None;
        }
        let keys: Vec<Vec<Option<(u32, u32)>>> = pools.iter().map(|p| if p.scratch { Vec::new() } else { vec![None; p.slots.slots()] }).collect();
        let free = keys.iter().map(Vec::len).collect();
        let placed = RwLock::new(Placement { at: HashMap::new(), keys, free });
        let uses = (0..layers.len() * per_layer).map(|_| AtomicU32::new(0)).collect();
        let rates = cards.iter().map(|_| AtomicU64::new(RATE_GUESS.to_bits())).collect();
        Some(WgpuExl3Experts { layouts, pools, placed, uses, per_layer, calls: AtomicU64::new(0), rates, cursor: Mutex::new(0), counts: Counts::default(), kept: Mutex::new(Vec::new()) })
    }

    /// Start every expert's uses from `counts` (a usage profile's, the model's own counts as an earlier run left
    /// them: `layer * per_layer + expert`), so the experts a profile put on the cards are not taken as unused and
    /// given up to the first that passes twice. Nothing changed if they are not this model's shape.
    pub(crate) fn seed_uses(&self, counts: &[u32]) {
        if counts.len() == self.uses.len() {
            for (n, &c) in self.uses.iter().zip(counts) {
                n.store(c, Ordering::Relaxed);
            }
        }
    }

    fn placement(&self) -> std::sync::RwLockReadGuard<'_, Placement> {
        self.placed.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Slots for held experts, and those filled so far.
    pub(crate) fn tier(&self) -> (usize, usize) {
        let total = self.pools.iter().filter(|p| !p.scratch).map(|p| p.slots.slots()).sum();
        (total, self.placement().at.len())
    }

    /// Since the model loaded: experts computed from their slots, experts put through the scratch slots, records taken
    /// into a free slot (from RAM or the drive), experts kept from a scratch slot.
    pub(crate) fn counts(&self) -> (u64, u64, u64, u64) {
        let c = &self.counts;
        (c.held.load(Ordering::Relaxed), c.scratch.load(Ordering::Relaxed), c.taken.load(Ordering::Relaxed), c.kept.load(Ordering::Relaxed))
    }

    /// The bitrate of `layer`'s experts.
    fn tw(&self, layer: u32) -> [usize; 2] {
        let l = &self.layouts[layer as usize].proj;
        [l[0].tile_words, l[2].tile_words]
    }

    fn uses_of(&self, layer: u32, expert: u32) -> u32 {
        self.uses.get(layer as usize * self.per_layer + expert as usize).map_or(0, |n| n.load(Ordering::Relaxed))
    }

    /// A free slot of `layer`'s bitrate (on `card`, where one is named): `(pool, slot)`.
    fn free_slot(&self, placed: &Placement, layer: u32, card: Option<usize>) -> Option<(usize, usize)> {
        let tw = self.tw(layer);
        let p = (0..self.pools.len()).find(|&p| !self.pools[p].scratch && self.pools[p].tw == tw && card.is_none_or(|c| self.pools[p].card == c) && placed.free[p] > 0)?;
        Some((p, placed.keys[p].iter().position(Option::is_none).expect("a pool with a free slot has one")))
    }

    /// `layer`'s `expert` into a free slot of its bitrate, if there is one and it is not held already: whether it was.
    pub(crate) fn take(&self, layer: u32, expert: u32, record: &[u8]) -> bool {
        let mut placed = self.placed.write().unwrap_or_else(|p| p.into_inner());
        if placed.at.contains_key(&(layer, expert)) {
            return false;
        }
        let Some((p, s)) = self.free_slot(&placed, layer, None) else { return false };
        // Written into a scratch slot of its card in place and copied from there by the card: the queue's writes went
        // through staging memory at some 3 GB/s, where the host's threads write a card's memory at 11 to 22.
        let (card, tw, layout) = (self.pools[p].card, self.pools[p].tw, &self.layouts[layer as usize]);
        match self.pools.iter().find(|q| q.scratch && q.card == card && q.tw == tw) {
            Some(q) => {
                let _one = q.busy.lock().unwrap_or_else(|e| e.into_inner());
                q.slots.put_many(0, &[record], layout);
                q.slots.copy_slot(0, &self.pools[p].slots, s);
            }
            None => self.pools[p].slots.put(s, record, layout),
        }
        placed.keys[p][s] = Some((layer, expert));
        placed.free[p] -= 1;
        placed.at.insert((layer, expert), (p, s));
        self.counts.taken.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Keep `layer`'s `expert`, in slot `slot` of scratch pool `from` now, on that card: in a free slot of its bitrate
    /// there, or in place of the least used of another layer's experts there (this layer's may be being computed) when
    /// it is used more and [`KEEP_USES`] times or more. By a copy of the card's memory.
    fn keep(&self, layer: u32, expert: u32, from: (usize, usize)) {
        let uses = self.uses_of(layer, expert);
        let card = self.pools[from.0].card;
        let tw = self.tw(layer);
        let mut placed = self.placed.write().unwrap_or_else(|p| p.into_inner());
        if placed.at.contains_key(&(layer, expert)) {
            return;
        }
        let at = match self.free_slot(&placed, layer, Some(card)) {
            Some(at) => at,
            None if uses < KEEP_USES => return,
            None => {
                let mut least: Option<(usize, usize, u32)> = None;
                for p in (0..self.pools.len()).filter(|&p| !self.pools[p].scratch && self.pools[p].tw == tw && self.pools[p].card == card) {
                    for (s, key) in placed.keys[p].iter().enumerate() {
                        let Some((l, e)) = *key else { continue };
                        let n = self.uses_of(l, e);
                        if l != layer && least.is_none_or(|(.., m)| n < m) {
                            least = Some((p, s, n));
                        }
                    }
                }
                match least {
                    Some((p, s, n)) if n < uses => (p, s),
                    _ => return,
                }
            }
        };
        self.pools[from.0].slots.copy_slot(from.1, &self.pools[at.0].slots, at.1);
        match placed.keys[at.0][at.1].replace((layer, expert)) {
            Some(gone) => drop(placed.at.remove(&gone)),
            None => placed.free[at.0] -= 1,
        }
        placed.at.insert((layer, expert), at);
        self.counts.kept.fetch_add(1, Ordering::Relaxed);
        self.kept.lock().unwrap_or_else(|p| p.into_inner()).push((layer, expert));
    }

    /// Whether a free slot is left for `layer`'s bitrate.
    pub(crate) fn has_room(&self, layer: u32) -> bool {
        self.free_slot(&self.placement(), layer, None).is_some()
    }

    /// Whether the cards hold `layer`'s `expert`: the idle reading's question, which leaves those out of RAM (an expert
    /// in one place).
    pub(crate) fn elsewhere(&self, layer: u32, expert: u32) -> bool {
        self.placement().at.contains_key(&(layer, expert))
    }

    /// Fill up to `count` more free slots while the server idles, in `order` (a usage profile's, the most used first;
    /// None: every layer's expert `e` before any layer's `e + 1`) from where the last call stopped: each record taken
    /// from the RAM tier where a request read it there (and let go from it), else read from `store` on several
    /// threads. Whether any slot is left to fill.
    pub(crate) fn fill_some(&self, store: &dyn oaiy_engine::store::WeightStore, cache: &oaiy_engine::ecache::Ecache, order: Option<&[(u32, u32)]>, count: usize) -> bool {
        let (layers, per_layer) = store.shape();
        let total = layers as usize * per_layer as usize;
        let mut picks = Vec::with_capacity(count);
        {
            let mut cursor = self.cursor.lock().unwrap_or_else(|p| p.into_inner());
            while *cursor < total && picks.len() < count {
                let (l, e) = order.and_then(|o| o.get(*cursor).copied()).unwrap_or(((*cursor % layers as usize) as u32, (*cursor / layers as usize) as u32));
                *cursor += 1;
                if !self.elsewhere(l, e) && self.has_room(l) {
                    picks.push((l, e));
                }
            }
            if picks.is_empty() {
                return false;
            }
        }
        let (in_ram, to_read): (Vec<(u32, u32)>, Vec<(u32, u32)>) = picks.into_iter().partition(|&(l, e)| cache.probe(l, e));
        for (l, e) in in_ram {
            if let Ok(lease) = cache.acquire(l, e, store) {
                let taken = self.take(l, e, &lease);
                drop(lease);
                if taken {
                    cache.remove(l, e);
                }
            }
        }
        let turn = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..FILL_READERS.min(to_read.len()) {
                scope.spawn(|| {
                    let mut record = vec![0u8; store.record_bytes()];
                    while let Some(&(l, e)) = to_read.get(turn.fetch_add(1, Ordering::Relaxed)) {
                        if store.fetch(l, e, &mut record).is_ok() {
                            self.take(l, e, &record);
                        }
                    }
                });
            }
        });
        let cursor = *self.cursor.lock().unwrap_or_else(|p| p.into_inner());
        cursor < total && self.placement().free.iter().any(|&n| n > 0)
    }

    /// `card`'s rate of putting records into its scratch slots (bytes a microsecond).
    fn rate(&self, card: usize) -> f64 {
        f64::from_bits(self.rates[card].load(Ordering::Relaxed))
    }

    /// `jobs` (of `list`) through scratch pool `p`, a group of its slots at a time: each one's output, weighted. Each
    /// expert is then kept on the card where it is used enough ([`Self::keep`]).
    fn through_scratch(&self, layer: u32, p: usize, jobs: &[ExpertJob<'_>], list: &[usize], limit: f32) -> Vec<(usize, Vec<f32>)> {
        let (pool, layout) = (&self.pools[p], &self.layouts[layer as usize]);
        let _one = pool.busy.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(list.len());
        for group in list.chunks(pool.slots.slots()) {
            let uploading = std::time::Instant::now();
            let records: Vec<&[u8]> = group.iter().map(|&i| jobs[i].record).collect();
            pool.slots.put_many(0, &records, layout);
            let took = uploading.elapsed().as_secs_f64() * 1e6;
            crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertUpload, uploading);
            // (the card's rate, a fifth from this put)
            let seen = (records.len() * layout.bytes) as f64 / took.max(1.0);
            self.rates[pool.card].store((self.rate(pool.card) * 0.8 + seen * 0.2).to_bits(), Ordering::Relaxed);
            let width = layout.proj[0].k;
            let these: Vec<(usize, &[f32])> = group.iter().enumerate().flat_map(|(s, &i)| apart(s, jobs[i].x, width)).collect();
            let mut ys = pool.slots.forward(&these, limit).into_iter();
            for &i in group {
                let n = pieces(jobs[i].x, width);
                out.push((i, weighed(ys.by_ref().take(n).flatten().collect(), jobs[i].weights)));
            }
            for (s, &i) in group.iter().enumerate() {
                self.keep(layer, jobs[i].expert, (p, s));
            }
        }
        out
    }
}

impl ExpertsKernel for WgpuExl3Experts {
    /// Experts the cards do not hold: their records through the scratch slots of every card, each card given them by
    /// its rate of putting them in (each job to the card that would be done with it first), the cards at once.
    fn forward(&self, layer: u32, jobs: &[ExpertJob<'_>], swiglu_limit: f32) -> Vec<Vec<f32>> {
        let tw = self.tw(layer);
        let layout = &self.layouts[layer as usize];
        let scratch: Vec<usize> = (0..self.pools.len()).filter(|&p| self.pools[p].scratch && self.pools[p].tw == tw).collect();
        if scratch.is_empty() {
            // (no scratch slots of this bitrate: on the CPU)
            return jobs.iter().map(|j| crate::dsv41::exl3_experts::forward_rows(j.record, &layout_proj(layout), j.x, Some(j.weights), swiglu_limit).unwrap_or_default()).collect();
        }
        self.counts.scratch.fetch_add(jobs.len() as u64, Ordering::Relaxed);
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); scratch.len()];
        let mut load = vec![0f64; scratch.len()];
        for i in 0..jobs.len() {
            let k = (0..scratch.len())
                .min_by(|&a, &b| {
                    let done = |k: usize| (load[k] + layout.bytes as f64) / self.rate(self.pools[scratch[k]].card);
                    done(a).total_cmp(&done(b))
                })
                .expect("a scratch pool");
            lists[k].push(i);
            load[k] += layout.bytes as f64;
        }
        let busy: Vec<(usize, &Vec<usize>)> = scratch.iter().copied().zip(&lists).filter(|(_, l)| !l.is_empty()).collect();
        let made: Vec<(usize, Vec<f32>)> = match busy.as_slice() {
            [] => Vec::new(),
            [(p, list)] => self.through_scratch(layer, *p, jobs, list, swiglu_limit),
            [(p, list), rest @ ..] => std::thread::scope(|scope| {
                let handles: Vec<_> = rest.iter().map(|&(q, l)| scope.spawn(move || self.through_scratch(layer, q, jobs, l, swiglu_limit))).collect();
                let mut made = self.through_scratch(layer, *p, jobs, list, swiglu_limit);
                for h in handles {
                    made.extend(h.join().expect("an EXL3 experts' card panicked"));
                }
                made
            }),
        };
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); jobs.len()];
        for (i, y) in made {
            out[i] = y;
        }
        out
    }

    /// Which of `experts` the cards hold; each counted as used (the counts halved every
    /// [`crate::dsv41::moe::USES_AGE_STEPS`] steps of every layer).
    fn holds(&self, layer: u32, experts: &[u32], tokens: &[usize]) -> Vec<bool> {
        let _ = tokens;
        for &e in experts {
            if let Some(n) = self.uses.get(layer as usize * self.per_layer + e as usize) {
                n.fetch_add(1, Ordering::Relaxed);
            }
        }
        let every = crate::dsv41::moe::USES_AGE_STEPS * self.layouts.len() as u64;
        if (self.calls.fetch_add(1, Ordering::Relaxed) + 1) % every == 0 {
            for n in &self.uses {
                n.store(n.load(Ordering::Relaxed) / 2, Ordering::Relaxed);
            }
        }
        let placed = self.placement();
        experts.iter().map(|&e| placed.at.contains_key(&(layer, e))).collect()
    }

    /// The held experts' jobs: each card's pools in one call (one recording and one read back), the cards at once.
    fn forward_held(&self, layer: u32, jobs: &[(u32, &[f32], &[f32])], swiglu_limit: f32) -> Vec<Vec<f32>> {
        self.begin_held(layer, jobs, swiglu_limit)()
    }

    /// [`Self::forward_held`] begun: each card's call recorded and submitted here, one after another, and waited for
    /// by the closure, so the cards and what the caller does meanwhile (the reads of the experts they do not hold, and
    /// their calls) run at once with no thread started (a thread for the held experts and one for each card past the
    /// first, a layer).
    fn begin_held<'a>(&'a self, layer: u32, jobs: &[(u32, &[f32], &[f32])], swiglu_limit: f32) -> Box<dyn FnOnce() -> Vec<Vec<f32>> + 'a> {
        let beginning = std::time::Instant::now();
        self.counts.held.fetch_add(jobs.len() as u64, Ordering::Relaxed);
        let places: Vec<(usize, usize)> = {
            let placed = self.placement();
            jobs.iter().map(|(e, ..)| *placed.at.get(&(layer, *e)).expect("a held expert")).collect()
        };
        // each card's jobs, by pool: (card, [(pool, [job])])
        let mut by_card: Vec<(usize, Vec<(usize, Vec<usize>)>)> = Vec::new();
        for (i, &(p, _)) in places.iter().enumerate() {
            let card = self.pools[p].card;
            let at = by_card.iter().position(|(c, _)| *c == card).unwrap_or_else(|| {
                by_card.push((card, Vec::new()));
                by_card.len() - 1
            });
            let pools = &mut by_card[at].1;
            match pools.iter_mut().find(|(q, _)| *q == p) {
                Some((_, list)) => list.push(i),
                None => pools.push((p, vec![i])),
            }
        }
        let width = self.layouts[layer as usize].proj[0].k;
        let begun: Vec<(Vec<Vec<(usize, usize)>>, ggml_rs_wgpu::exl3::SlotsBegun<'a>)> = by_card
            .into_iter()
            .map(|(_, pools)| {
                let these: Vec<Vec<(usize, &[f32])>> = pools.iter().map(|(_, list)| list.iter().flat_map(|&i| apart(places[i].1, jobs[i].1, width)).collect()).collect();
                let parts: Vec<(&'a Exl3Slots, &[(usize, &[f32])])> = pools.iter().zip(&these).map(|((p, _), t)| (&self.pools[*p].slots, &t[..])).collect();
                let call = Exl3Slots::begin_many(&parts, swiglu_limit);
                // (each pool's jobs, and how many of the call's each became)
                (pools.into_iter().map(|(_, list)| list.into_iter().map(|i| (i, pieces(jobs[i].1, width))).collect()).collect(), call)
            })
            .collect();
        let weights: Vec<Vec<f32>> = jobs.iter().map(|j| j.2.to_vec()).collect();
        let begun_in = beginning.elapsed();
        Box::new(move || {
            let finishing = std::time::Instant::now();
            let mut out: Vec<Vec<f32>> = vec![Vec::new(); weights.len()];
            for (lists, call) in begun {
                for (list, ys) in lists.into_iter().zip(call.finish()) {
                    let mut ys = ys.into_iter();
                    for (i, n) in list {
                        out[i] = weighed(ys.by_ref().take(n).flatten().collect(), &weights[i]);
                    }
                }
            }
            crate::dsv41::profile::add_spent(crate::dsv41::profile::Part::ExpertHeld, begun_in + finishing.elapsed());
            out
        })
    }

    fn prefetch_order(&self, layer: u32, experts: u32) -> Vec<u32> {
        let placed = self.placement();
        let mut order: Vec<u32> = (0..experts).filter(|&e| !placed.at.contains_key(&(layer, e))).collect();
        order.sort_by_key(|&e| std::cmp::Reverse(self.uses_of(layer, e)));
        order
    }

    /// Of `records` (a decode step's, read for experts the cards did not hold and since put through the scratch slots),
    /// those the cards kept: RAM lets them go (an expert in one place). Nothing is taken in here: an expert is kept
    /// from its scratch slot by a copy of the card's memory, where a slot written from the record again was a second
    /// upload of it.
    fn offer(&self, layer: u32, records: &[(u32, &[u8])]) -> Vec<u32> {
        let mut kept = self.kept.lock().unwrap_or_else(|p| p.into_inner());
        let mine: Vec<u32> = records.iter().map(|&(e, _)| e).filter(|&e| kept.contains(&(layer, e))).collect();
        kept.retain(|&(l, e)| l != layer || !mine.contains(&e));
        // (what a prompt kept is not asked for: let go of what has waited past a few layers' worth)
        if kept.len() > 4 * SCRATCH {
            kept.clear();
        }
        mine
    }
}
