//! The routed experts on the WebGPU adapters: the slots a prompt's pass through, the other cards' share, and a
//! decode step's computed where they are held.

use super::resident::Resident;
use super::*;

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
    /// The share of the experts the computer's other GPUs hold ([`WgpuExperts::pin_on`]).
    pinned: Option<Pinned>,
    /// The experts the first card's tier is filled with at start-up, where a usage profile says which are used most
    /// ([`WgpuExperts::start_from`]): read from the drive while the server idles, before the other cards' share.
    /// Empty without a profile: the tier then takes in what the first requests use.
    first: Vec<(u32, u32)>,
    first_next: std::sync::atomic::AtomicUsize,
    /// Experts the idle rebalance has moved onto the cards so far.
    moved: std::sync::atomic::AtomicU64,
}

/// Where a card holds an expert: the first card's tier's slot, or another card's and its slot.
#[derive(Clone, Copy)]
enum Place {
    First(usize),
    Share(usize, usize),
}

/// How much more an expert in RAM must have been used than the one a card would give up for it: twice as much and
/// [`REBALANCE_MARGIN`] more. Each move is an upload and a record read back from the drive, so two experts used about
/// as much do not change places back and forth.
const REBALANCE_MARGIN: u32 = 4;

/// Experts the other GPUs hold for good: a fixed share of every layer's (its last ones: the RAM tier's idle reading
/// starts from the first), each read from the drive once into a slot of its own while the server idles and never
/// replaced. The first card's tier follows what is used (LFRU) and RAM holds what it can of the rest; this share needs
/// neither, so between them the tiers hold more of the 10,612 than RAM alone, and a prompt, which routes to most of
/// them, reads that much less from the drive each time.
struct Pinned {
    /// each card's slots, and the plan's first slot that is its own
    cards: Vec<(RecordSlots, usize)>,
    /// slots over the cards, and the expert each is filled with at start-up: layer `g % layers`'s expert
    /// `per_layer - 1 - g / layers` for slot `g`, or a usage profile's ([`WgpuExperts::start_from`])
    total: usize,
    plan: Vec<(u32, u32)>,
    slot_of: std::collections::HashMap<(u32, u32), usize>,
    /// the experts whose records are in their slots so far: `(card, slot)`
    index: std::sync::RwLock<std::collections::HashMap<(u32, u32), (usize, usize)>>,
    /// the plan's next slot to fill
    next: std::sync::atomic::AtomicUsize,
    /// uses of the share's experts so far (for the profile)
    hits: std::sync::atomic::AtomicU64,
}

impl Pinned {
    fn expert(&self, g: usize) -> (u32, u32) {
        self.plan[g]
    }

    /// The plan's slot for `(layer, expert)`, if it is one of the share.
    fn slot(&self, layer: u32, expert: u32) -> Option<usize> {
        self.slot_of.get(&(layer, expert)).copied()
    }

    fn card(&self, g: usize) -> (usize, usize) {
        let c = self.cards.iter().rposition(|&(_, first)| first <= g).expect("a plan slot is a card's");
        (c, g - self.cards[c].1)
    }
}

/// Threads reading the pinned share's records from the drive at once (as dsv41's readers).
const PIN_READERS: usize = 8;

/// `keys`' records read from `store` on [`PIN_READERS`] threads, each with `slot(i)` for the `i`-th key. A record that
/// cannot be read is left out (its expert goes through RAM as any other).
fn read_records(store: &dyn oaiy_engine::store::WeightStore, keys: &[(u32, u32)], slot: &(dyn Fn(usize) -> usize + Sync)) -> Vec<(usize, Vec<u8>)> {
    use std::sync::atomic::Ordering;
    let turn = std::sync::atomic::AtomicUsize::new(0);
    let read: Mutex<Vec<(usize, Vec<u8>)>> = Mutex::new(Vec::with_capacity(keys.len()));
    std::thread::scope(|scope| {
        for _ in 0..PIN_READERS.min(keys.len()) {
            scope.spawn(|| loop {
                let i = turn.fetch_add(1, Ordering::Relaxed);
                let Some(&(layer, expert)) = keys.get(i) else { break };
                let mut record = vec![0u8; dsv41::expert::RECORD_BYTES];
                if store.fetch(layer, expert, &mut record).is_ok() {
                    read.lock().unwrap_or_else(|e| e.into_inner()).push((slot(i), record));
                }
            });
        }
    });
    let mut read = read.into_inner().unwrap_or_else(|e| e.into_inner());
    read.sort_by_key(|(g, _)| *g);
    read
}

/// What the weight budget keeps free beyond the slots: the passes' own buffers (a 2,000-token prompt's projections
/// make outputs of a few hundred MB).
pub(super) const MARGIN: u64 = 1 << 30;

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
            first: Vec::new(),
            first_next: std::sync::atomic::AtomicUsize::new(0),
            moved: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Start from a usage profile's order (every expert once, the most used first: [`dsv41::moe::Uses::order`]): the
    /// first card's tier is filled with the first of them and the other cards' share with the next, where without a
    /// profile the tier waits for the first requests and the share is each layer's last experts. Before the model
    /// serves anything.
    pub(crate) fn start_from(&mut self, order: &[(u32, u32)]) {
        let tier = self.resident.as_ref().map_or(0, |r| r.lock().unwrap_or_else(|p| p.into_inner()).keys.len()).min(order.len());
        self.first = order[..tier].to_vec();
        if let Some(p) = &mut self.pinned {
            let share = &order[tier..(tier + p.total).min(order.len())];
            if share.len() == p.total {
                p.plan = share.to_vec();
                p.slot_of = share.iter().enumerate().map(|(g, &key)| (key, g)).collect();
            }
        }
    }

    /// While the server idles: up to `count` of the most used experts RAM holds go onto the cards, each in place of a
    /// card's least used (or into a slot the first card has free), where it has been used twice as much and
    /// [`REBALANCE_MARGIN`] more; the displaced expert's record is read from the drive into the place RAM has free
    /// again. How many moved (0: the cards hold what is used most, as far as the counts say).
    ///
    /// Only where the tiers are exclusive (an expert in one place): the cards then hold the experts a session uses
    /// most, whichever they turn out to be, and RAM the next most. The other cards' share starts as a fixed one (each
    /// layer's last experts, read at start-up before anything is known of their use), and the first card's as what
    /// the first prompts routed to; neither changes on a request's own path, where a swap is an upload of 18.9 MB and
    /// an expert left in no tier.
    pub(crate) fn rebalance(&self, store: &dyn oaiy_engine::store::WeightStore, cache: &oaiy_engine::ecache::Ecache, uses: &dsv41::moe::Uses, count: usize) -> usize {
        use std::sync::atomic::Ordering;
        if !self.exclusive.load(Ordering::Relaxed) || count == 0 {
            return 0;
        }
        // what the cards hold, least used first (a free slot of the first card before any), and RAM's most used
        let (mut held, mut wanted) = {
            let (counts, per_layer) = (uses.counts(), uses.shape().1);
            let of = |key: &(u32, u32)| counts.get(key.0 as usize * per_layer + key.1 as usize).copied().unwrap_or(0);
            let mut held: Vec<(u32, Option<(u32, u32)>, Place)> = Vec::new();
            if let Some(r) = &self.resident {
                let r = r.lock().unwrap_or_else(|p| p.into_inner());
                held.extend(r.keys.iter().enumerate().map(|(i, key)| (key.as_ref().map_or(0, of), *key, Place::First(i))));
            }
            if let Some(p) = &self.pinned {
                let index = p.index.read().unwrap_or_else(|e| e.into_inner());
                held.extend(index.iter().map(|(key, &(card, slot))| (of(key), Some(*key), Place::Share(card, slot))));
            }
            let on_a_card: std::collections::HashSet<(u32, u32)> = held.iter().filter_map(|h| h.1).collect();
            let mut wanted: Vec<(u32, (u32, u32))> = counts
                .iter()
                .enumerate()
                .map(|(i, &n)| (n, ((i / per_layer.max(1)) as u32, (i % per_layer.max(1)) as u32)))
                .filter(|(n, key)| *n > REBALANCE_MARGIN && !on_a_card.contains(key))
                .collect();
            (held, {
                wanted.sort_unstable_by(|a, b| b.cmp(a));
                wanted
            })
        };
        held.sort_unstable_by_key(|h| (h.1.is_some(), h.0, h.1));
        wanted.retain(|&(_, (layer, expert))| cache.probe(layer, expert));
        let mut moved = 0;
        for (&(n, key), &(least, old, place)) in wanted.iter().zip(&held) {
            if moved == count || (old.is_some() && n < 2 * least + REBALANCE_MARGIN) {
                break;
            }
            // the displaced one's record first: if the drive will not give it, nothing changes
            let back = match old {
                Some((layer, expert)) => {
                    let mut record = vec![0u8; dsv41::expert::RECORD_BYTES];
                    if store.fetch(layer, expert, &mut record).is_err() {
                        continue;
                    }
                    Some(record)
                }
                None => None,
            };
            let Ok(record) = cache.acquire(key.0, key.1, store) else { continue };
            match place {
                Place::First(i) => {
                    let mut r = self.resident.as_ref().expect("the first card's slot").lock().unwrap_or_else(|p| p.into_inner());
                    r.put(i, key, &record);
                }
                Place::Share(card, slot) => {
                    let p = self.pinned.as_ref().expect("another card's slot");
                    p.cards[card].0.write(slot, &record);
                    let mut index = p.index.write().unwrap_or_else(|e| e.into_inner());
                    if let Some(old) = old {
                        index.remove(&old);
                    }
                    index.insert(key, (card, slot));
                }
            }
            drop(record);
            cache.remove(key.0, key.1);
            if let (Some((layer, expert)), Some(record)) = (old, back) {
                // (with the count it has: it comes to RAM as one of its more used, not as a newcomer)
                let _ = cache.admit_owned(layer, expert, record);
                cache.credit(layer, expert, least as u64);
            }
            moved += 1;
        }
        self.moved.fetch_add(moved as u64, Ordering::Relaxed);
        moved
    }

    /// Experts the idle rebalance has moved onto the cards so far.
    pub(crate) fn moved(&self) -> u64 {
        self.moved.load(std::sync::atomic::Ordering::Relaxed)
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
            let plan: Vec<(u32, u32)> = (0..total).map(|g| ((g % layers) as u32, (per_layer - 1 - g / layers) as u32)).collect();
            let slot_of = plan.iter().enumerate().map(|(g, &key)| (key, g)).collect();
            self.pinned = Some(Pinned { cards, total, plan, slot_of, index: Default::default(), next: Default::default(), hits: Default::default() });
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

    /// Whether a card holds `(layer, expert)` now, or a slot of the start-up plan still waits to be filled with it:
    /// the idle reading's question, which leaves those out of RAM (an expert in one place). By what the cards hold,
    /// not by the plan: after a [`Self::rebalance`] the plan's experts may be in RAM and others on the cards, and a
    /// reading begun again (RAM grown: `Engine::fit_ram`) that asked the plan read the cards' experts into RAM a
    /// second time.
    pub(crate) fn elsewhere(&self, layer: u32, expert: u32) -> bool {
        use std::sync::atomic::Ordering;
        let key = (layer, expert);
        if self.resident.as_ref().is_some_and(|r| r.lock().unwrap_or_else(|p| p.into_inner()).index.contains_key(&key)) {
            return true;
        }
        if let Some(p) = &self.pinned {
            if p.index.read().unwrap_or_else(|e| e.into_inner()).contains_key(&key) || p.slot(layer, expert).is_some_and(|g| g >= p.next.load(Ordering::Relaxed)) {
                return true;
            }
        }
        self.first[self.first_next.load(Ordering::Relaxed).min(self.first.len())..].contains(&key)
    }

    /// Uses of the other GPUs' share so far.
    pub(crate) fn share_hits(&self) -> u64 {
        self.pinned.as_ref().map_or(0, |p| p.hits.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The experts the other GPUs hold so far.
    pub(crate) fn pinned(&self) -> usize {
        self.pinned.as_ref().map_or(0, |p| p.index.read().unwrap_or_else(|e| e.into_inner()).len())
    }

    /// Fill the share's next `count` slots: their records read from `store` on several threads, written to their slots,
    /// and let go from the RAM tier if a request had read them there. Whether more are left to fill.
    pub(crate) fn pin_some(&self, store: &dyn oaiy_engine::store::WeightStore, cache: &oaiy_engine::ecache::Ecache, count: usize) -> bool {
        use std::sync::atomic::Ordering;
        // the first card's tier first, where a usage profile says what it should hold: the most used of all
        let at = self.first_next.load(Ordering::Relaxed);
        if at < self.first.len() {
            let last = (at + count).min(self.first.len());
            let read = read_records(store, &self.first[at..last], &|i| at + i);
            let mut r = self.resident.as_ref().expect("a tier to fill").lock().unwrap_or_else(|p| p.into_inner());
            for (g, record) in &read {
                let key = self.first[*g];
                if let (false, Some(i)) = (r.index.contains_key(&key), r.keys.iter().position(Option::is_none)) {
                    r.put(i, key, record);
                    cache.remove(key.0, key.1);
                }
            }
            self.first_next.store(last, Ordering::Relaxed);
            return last < self.first.len() || self.pinned.as_ref().is_some_and(|p| p.next.load(Ordering::Relaxed) < p.total);
        }
        let Some(p) = &self.pinned else { return false };
        let first = p.next.load(Ordering::Relaxed);
        if first >= p.total {
            return false;
        }
        let last = (first + count).min(p.total);
        let read = read_records(store, &p.plan[first..last], &|i| first + i);
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

    /// Keep what comes in out of the RAM tier (an expert in one place: the first card's tier then takes in only what
    /// a free slot holds as requests go, and changes by [`Self::rebalance`] while the server idles), or in both (as
    /// made: the tier replaces its least used as it goes).
    pub(crate) fn set_exclusive(&self, on: bool) {
        self.exclusive.store(on, std::sync::atomic::Ordering::Relaxed);
        if let Some(r) = &self.resident {
            r.lock().unwrap_or_else(|p| p.into_inner()).replaces = !on;
        }
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

/// Experts whose records are in `slots`, `(slot, x, weights)` each. A decode step's (one row each): all of it in one
/// submit, the SwiGLU between an expert's projections made on the device ([`ggml_rs_wgpu::dense::forward_units`]). A
/// prompt's rows: every one's gate and up in one submit, the SwiGLU on the host as dsv41 takes it, then every down in
/// another; a step's went that way too, two round trips a card a layer, and once the cards held most of a step's
/// experts that was what the step waited for.
fn run(slots: &RecordSlots, picks: &[(usize, &[f32], &[f32])], swiglu_limit: f32) -> Vec<Vec<f32>> {
    use dsv41::expert::{BLOCK, DIM, INTER, S1, S2, S3, W1, W2, W3};
    use dsv41::formats::{fake_quant_fp8, to_bf16};
    let rows: Vec<usize> = picks.iter().map(|(_, x, _)| x.len() / DIM).collect();
    let mats: Vec<[DenseGpu; 3]> = picks
        .iter()
        .map(|&(i, ..)| [slots.mxfp4(i, W1.start, S1.start, INTER, DIM), slots.mxfp4(i, W3.start, S3.start, INTER, DIM), slots.mxfp4(i, W2.start, S2.start, DIM, INTER)])
        .collect();
    // Each input quantized once, however many experts take it (a decode step's all take the token's one row: quantized
    // for each, it was most of the host's part of the call, and each copy an upload of its own).
    let mut quantized: Vec<(*const f32, usize, Vec<f32>)> = Vec::new();
    for (_, x, _) in picks {
        if !quantized.iter().any(|(at, len, _)| (*at, *len) == (x.as_ptr(), x.len())) {
            quantized.push((x.as_ptr(), x.len(), fake_quant_fp8(x, BLOCK)));
        }
    }
    let xq: Vec<&[f32]> = picks.iter().map(|(_, x, _)| &quantized.iter().find(|(at, len, _)| (*at, *len) == (x.as_ptr(), x.len())).expect("quantized above").2[..]).collect();
    if rows.iter().all(|&r| r == 1) {
        let units: Vec<ggml_rs_wgpu::dense::Unit<'_>> = (0..picks.len())
            .map(|i| ggml_rs_wgpu::dense::Unit { gate: &mats[i][0], up: &mats[i][1], down: &mats[i][2], x: xq[i], weight: picks[i].2[0] })
            .collect();
        if let Some((downs, _)) = ggml_rs_wgpu::dense::forward_units(&units, &[], swiglu_limit) {
            return downs.into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect();
        }
    }
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
        (0..picks.len()).flat_map(|i| [(&mats[i][0], xq[i], rows[i], 0..INTER), (&mats[i][1], xq[i], rows[i], 0..INTER)]).collect();
    let sums = ggml_rs_wgpu::dense::forward_batch(&items);
    let hq: Vec<Vec<f32>> =
        sums.chunks(2).zip(picks).map(|(p, (_, _, w))| fake_quant_fp8(&dsv41::expert::swiglu(&p[0], &p[1], Some(w), swiglu_limit), BLOCK)).collect();
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> = (0..picks.len()).map(|i| (&mats[i][2], hq[i].as_slice(), rows[i], 0..DIM)).collect();
    ggml_rs_wgpu::dense::forward_batch(&items).into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect()
}

/// A decode step's experts on one card, begun ([`begin`]): the call submitted and its results pending, or, where it
/// did not go through the device's arena, made already.
enum Begun {
    Pending(ggml_rs_wgpu::dense::Pending),
    Made(Vec<Vec<f32>>),
}

impl Begun {
    fn finish(self) -> Vec<Vec<f32>> {
        match self {
            Begun::Pending(pending) => pending.finish().0.into_iter().map(|y| y.into_iter().map(dsv41::formats::to_bf16).collect()).collect(),
            Begun::Made(outs) => outs,
        }
    }
}

/// [`run`] begun for a decode step's experts (one row each, a group of them at most): submitted to their card in one
/// call ([`ggml_rs_wgpu::dense::begin_units`]) and left for [`Begun::finish`], so the caller's own work meanwhile is
/// beside the card's.
fn begin(slots: &RecordSlots, picks: &[(usize, &[f32], &[f32])], swiglu_limit: f32) -> Begun {
    use dsv41::expert::{BLOCK, DIM, INTER, S1, S2, S3, W1, W2, W3};
    if picks.len() <= dsv41::moe::GPU_GROUP && picks.iter().all(|(_, x, w)| x.len() == DIM && w.len() == 1) {
        let mats: Vec<[DenseGpu; 3]> = picks
            .iter()
            .map(|&(i, ..)| [slots.mxfp4(i, W1.start, S1.start, INTER, DIM), slots.mxfp4(i, W3.start, S3.start, INTER, DIM), slots.mxfp4(i, W2.start, S2.start, DIM, INTER)])
            .collect();
        // (each input quantized once, however many experts take it: a step's all take the token's one row)
        let mut quantized: Vec<(*const f32, Vec<f32>)> = Vec::new();
        for (_, x, _) in picks {
            if !quantized.iter().any(|(at, _)| *at == x.as_ptr()) {
                quantized.push((x.as_ptr(), dsv41::formats::fake_quant_fp8(x, BLOCK)));
            }
        }
        let units: Vec<ggml_rs_wgpu::dense::Unit<'_>> = picks
            .iter()
            .zip(&mats)
            .map(|((_, x, w), m)| ggml_rs_wgpu::dense::Unit { gate: &m[0], up: &m[1], down: &m[2], x: &quantized.iter().find(|(at, _)| *at == x.as_ptr()).expect("quantized above").1, weight: w[0] })
            .collect();
        if let Some(pending) = ggml_rs_wgpu::dense::begin_units(&units, &[], swiglu_limit) {
            return Begun::Pending(pending);
        }
    }
    Begun::Made(picks.chunks(dsv41::moe::GPU_GROUP).flat_map(|part| run(slots, part, swiglu_limit)).collect())
}

impl dsv41::expert::ExpertsKernel for WgpuExperts {
    /// Each card's share of a decode step's held experts submitted in turn (the other cards', then the first card's
    /// tier), and all read back by the closure: the cards work beside each other and beside whatever the caller does
    /// before it asks, and no thread is started for any of them.
    fn begin_held<'a>(&'a self, layer: u32, jobs: &[(u32, &[f32], &[f32])], swiglu_limit: f32) -> Box<dyn FnOnce() -> Vec<Vec<f32>> + 'a> {
        let experts: Vec<u32> = jobs.iter().map(|j| j.0).collect();
        let there = self.pinned_now(layer, &experts);
        let cards = self.pinned.as_ref().map_or(0, |p| p.cards.len());
        let mut begun: Vec<(Vec<usize>, Begun)> = Vec::new();
        for c in 0..cards {
            let mine: Vec<usize> = (0..jobs.len()).filter(|&j| there[j].is_some_and(|(card, _)| card == c)).collect();
            if !mine.is_empty() {
                let slots = &self.pinned.as_ref().expect("a share's card").cards[c].0;
                let picks: Vec<(usize, &[f32], &[f32])> = mine.iter().map(|&j| (there[j].expect("its slot").1, jobs[j].1, jobs[j].2)).collect();
                begun.push((mine, begin(slots, &picks, swiglu_limit)));
            }
        }
        let first: Vec<usize> = (0..jobs.len()).filter(|&j| there[j].is_none()).collect();
        if !first.is_empty() {
            let r = self.resident.as_ref().expect("a kernel that holds experts").lock().unwrap_or_else(|p| p.into_inner());
            let picks: Vec<(usize, &[f32], &[f32])> = first.iter().map(|&j| (r.index[&(layer, jobs[j].0)], jobs[j].1, jobs[j].2)).collect();
            let b = begin(&r.slots, &picks, swiglu_limit);
            begun.push((first, b));
        }
        let n = jobs.len();
        Box::new(move || {
            let mut outs: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
            for (mine, b) in begun {
                for (j, out) in mine.into_iter().zip(b.finish()) {
                    outs[j] = Some(out);
                }
            }
            outs.into_iter().map(|o| o.expect("every held expert computed")).collect()
        })
    }

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
        // the other cards' share is held without being counted there; the first card's tier counts and holds of the rest
        let there = self.pinned_now(layer, experts);
        if let Some(p) = &self.pinned {
            p.hits.fetch_add(there.iter().flatten().count() as u64, std::sync::atomic::Ordering::Relaxed);
        }
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
