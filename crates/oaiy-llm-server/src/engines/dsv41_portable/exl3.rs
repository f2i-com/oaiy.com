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
//!
//! A GGUF checkpoint's experts are held so too, in `ggml_rs_wgpu::quant::moe::QuantSlots` (a pool for each pair of
//! types its layers have: a layer's gate's and up's, its down's), each slot an expert's blocks as the quantized
//! kernels read them. Its misses go through the scratch slots alone: the host's decoding sums otherwise than the
//! cards', and which side made an expert would show in the reply.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;

use ggml_rs_wgpu::exl3::{Exl3Slots, ProjAt, RecordLayout, SlotsBegun};
use ggml_rs_wgpu::quant::moe::{QuantLayout, QuantProj, QuantSlots};

use super::*;
use crate::dsv41::expert::{Exl3Proj, ExpertFormat, ExpertJob, ExpertsKernel, GgufProj};
use crate::dsv41::formats::to_bf16;

/// Scratch slots a bitrate has on each card: a pass's experts not held go through them, a prompt's a group
/// ([`crate::dsv41::moe::GPU_GROUP`]) at a time.
const SCRATCH: usize = crate::dsv41::moe::GPU_GROUP;

/// Scratch slots a GGUF's pair of types has on each card: a decode step's misses of a layer (six experts at most) at
/// once, a prompt's a few at a time. Its layers have several pairs (eight in a 2-bit file), each its own scratch.
const GGUF_SCRATCH: usize = 8;

/// What the weight budget keeps free on a card beyond the slots: a prompt's passes' own buffers.
const MARGIN: u64 = 1 << 30;

/// The rate a card is taken to put records into its scratch slots at before it has (bytes a microsecond: 10 GB/s).
const RATE_GUESS: f64 = 10_000.0;

/// An expert on the host's cores, and a scratch call after its puts, before either has been timed (us): sixteen idle
/// cores of AVX-512 make one in some 500 us; a call after a put waited some 1,000 for the card to take the writes in.
const HOST_US_GUESS: f64 = 600.0;
const CALL_US_GUESS: f64 = 1_000.0;

/// `v` (f64 bits) a fifth of the way to `x`.
fn ease(v: &AtomicU64, x: f64) {
    let old = f64::from_bits(v.load(Ordering::Relaxed));
    v.store((old * 0.8 + x * 0.2).to_bits(), Ordering::Relaxed);
}

/// Uses an expert needs before it takes the place of one the cards hold (a free slot takes any): one use is a step's
/// chance, and swapping on it would trade one passing expert for another.
const KEEP_USES: u32 = 2;

/// Threads reading the records an idle fill puts on the cards: a drive serves a queue of reads several times faster than
/// one at a time.
const FILL_READERS: usize = 8;

/// A pool's slots: an EXL3 checkpoint's experts, or a GGUF's.
enum Slots {
    Exl3(Exl3Slots),
    Quant(QuantSlots),
}

/// A layer's record layout, as its slots take it.
enum Layout {
    Exl3(RecordLayout),
    Quant(QuantLayout),
}

impl Layout {
    /// The bytes of a record its layer's expert takes (a GGUF's: its own three matrices', where a record is the
    /// largest layer's).
    fn bytes(&self) -> usize {
        match self {
            Layout::Exl3(l) => l.bytes,
            Layout::Quant(l) => l.bytes,
        }
    }

    /// Its experts' `(hidden, ff)`.
    fn shape(&self) -> (usize, usize) {
        match self {
            Layout::Exl3(l) => (l.proj[0].k, l.proj[0].n),
            Layout::Quant(l) => (l.proj[0].k, l.proj[0].n),
        }
    }

    /// Its class: the pools that may hold its experts are of it (an EXL3 layer's words a tile, the gate's and up's
    /// and the down's; a GGUF layer's types so).
    fn class(&self) -> [usize; 2] {
        match self {
            Layout::Exl3(l) => [l.proj[0].tile_words, l.proj[2].tile_words],
            Layout::Quant(l) => [l.proj[0].dtype as usize, l.proj[2].dtype as usize],
        }
    }

    /// A slot's bytes on a card for its experts.
    fn slot_bytes(&self) -> u64 {
        let ((hidden, ff), class) = (self.shape(), self.class());
        match self {
            Layout::Exl3(_) => slot_bytes(hidden, ff, class),
            Layout::Quant(l) => ggml_rs_wgpu::WgpuBackend::quant_slot_bytes(hidden, ff, [l.proj[0].dtype, l.proj[2].dtype]).unwrap_or(u64::MAX),
        }
    }

    /// Up to `n` slots of its class on `card` (`mapped`: the host writes them in place), as many as there is room for.
    fn slots_on(&self, card: &WgpuBackend, n: usize, mapped: bool) -> Option<Slots> {
        let ((hidden, ff), class) = (self.shape(), self.class());
        match self {
            Layout::Exl3(_) if mapped => card.exl3_slots_mapped(n, hidden, ff, class).map(Slots::Exl3),
            Layout::Exl3(_) => card.exl3_slots(n, hidden, ff, class).map(Slots::Exl3),
            Layout::Quant(l) if mapped => card.quant_slots_mapped(n, hidden, ff, [l.proj[0].dtype, l.proj[2].dtype]).map(Slots::Quant),
            Layout::Quant(l) => card.quant_slots(n, hidden, ff, [l.proj[0].dtype, l.proj[2].dtype]).map(Slots::Quant),
        }
    }
}

impl Slots {
    fn slots(&self) -> usize {
        match self {
            Slots::Exl3(s) => s.slots(),
            Slots::Quant(s) => s.slots(),
        }
    }

    fn put(&self, slot: usize, record: &[u8], layout: &Layout) {
        match (self, layout) {
            (Slots::Exl3(s), Layout::Exl3(l)) => s.put(slot, record, l),
            (Slots::Quant(s), Layout::Quant(l)) => s.put(slot, record, l),
            _ => unreachable!("a layer's slots are of its checkpoint's kind"),
        }
    }

    fn put_many(&self, first: usize, records: &[&[u8]], layout: &Layout) {
        match (self, layout) {
            (Slots::Exl3(s), Layout::Exl3(l)) => s.put_many(first, records, l),
            (Slots::Quant(s), Layout::Quant(l)) => s.put_many(first, records, l),
            _ => unreachable!("a layer's slots are of its checkpoint's kind"),
        }
    }

    fn copy_slot_marked(&self, from: usize, to: &Slots, at: usize, marks: &[(&ggml_rs::DeviceVec, usize, u32)]) {
        match (self, to) {
            (Slots::Exl3(s), Slots::Exl3(t)) => s.copy_slot_marked(from, t, at, marks),
            (Slots::Quant(s), Slots::Quant(t)) => s.copy_slot_marked(from, t, at, marks),
            _ => unreachable!("a copy between slots of one kind"),
        }
    }

    fn forward(&self, jobs: &[(usize, &[f32])], limit: f32) -> Vec<Vec<f32>> {
        match self {
            Slots::Exl3(s) => s.forward(jobs, limit),
            Slots::Quant(s) => s.forward(jobs, limit),
        }
    }

    /// `parts` (sets of slots on one card, all of one kind) begun in one recording.
    fn begin_many<'a>(parts: &[(&'a Slots, &[(usize, &[f32])])], limit: f32) -> SlotsBegun<'a> {
        if parts.iter().all(|(s, _)| matches!(s, Slots::Quant(_))) {
            let these: Vec<(&'a QuantSlots, &[(usize, &[f32])])> = parts.iter().map(|&(s, j)| (if let Slots::Quant(s) = s { s } else { unreachable!("matched") }, j)).collect();
            return QuantSlots::begin_many(&these, limit);
        }
        let these: Vec<(&'a Exl3Slots, &[(usize, &[f32])])> = parts.iter().map(|&(s, j)| (if let Slots::Exl3(s) = s { s } else { unreachable!("a card's slots are of one kind") }, j)).collect();
        Exl3Slots::begin_many(&these, limit)
    }
}

/// Slots of one class (a bitrate, or a GGUF's pair of types) on one card.
struct Pool {
    slots: Slots,
    /// Its class ([`Layout::class`]).
    class: [usize; 2],
    /// Scratch (a pass's experts pass through), or held experts.
    scratch: bool,
    /// Which of the cards it is on.
    card: usize,
    /// One call in the scratch slots at a time, and what they hold.
    busy: Mutex<Scratch>,
}

/// What a scratch pool's slots hold: each one's expert (`(layer, expert)`, as its last put left it) and when it was
/// last used, by the pool's own count of uses. An expert a pass finds there still is used as it is, no put again.
#[derive(Default)]
struct Scratch {
    keys: Vec<Option<(u32, u32)>>,
    used: Vec<u64>,
    uses: u64,
}

impl Scratch {
    fn new(slots: usize) -> Scratch {
        Scratch { keys: vec![None; slots], used: vec![0; slots], uses: 0 }
    }

    /// Slot `s` holds `key` now, used now.
    fn mark(&mut self, s: usize, key: (u32, u32)) {
        self.uses += 1;
        self.keys[s] = Some(key);
        self.used[s] = self.uses;
    }

    /// The slot that holds `key`, where one does (and OAIY_DSV41_SCRATCH_REUSE=0 does not say to put every expert
    /// again).
    fn find(&self, key: (u32, u32)) -> Option<usize> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("OAIY_DSV41_SCRATCH_REUSE").as_deref() == Ok("0")) {
            return None;
        }
        self.keys.iter().position(|k| *k == Some(key))
    }

    /// The slot used longest ago.
    fn oldest(&self) -> usize {
        (0..self.used.len()).min_by_key(|&s| self.used[s]).expect("a scratch pool has slots")
    }
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
    /// Experts a step made on the host where the cards did not hold them ([`ExpertsKernel::forward_here`]).
    host: AtomicU64,
    /// Of `held`, those on a card past the first (the trunk's).
    held_past_first: AtomicU64,
    /// Each card's held-experts calls (the first two cards), as the host found them once it had made the rest of the
    /// layer's experts: the time it then waited for their outputs (ns), how many calls, and how many it found running.
    held_wait: [[AtomicU64; 3]; 2],
    /// Each card's puts into its scratch slots (the first two cards): their time (ns), how many, their bytes, and the
    /// experts its scratch slots held still (no put).
    puts: [[AtomicU64; 4]; 2],
}

pub(crate) struct WgpuExl3Experts {
    /// Each layer's record layout.
    layouts: Vec<Layout>,
    pools: Vec<Pool>,
    placed: RwLock<Placement>,
    /// Every expert's uses of late (`layer * per_layer + expert`), and the calls since they were last halved.
    uses: Vec<AtomicU32>,
    per_layer: usize,
    calls: AtomicU64,
    /// Each card's rate of putting records into its scratch slots, as measured (bytes a microsecond, f64 bits).
    rates: Vec<AtomicU64>,
    /// Each card's best such rate of late (f64 bits: the best seen, let go of slowly): what a pass's misses are shared
    /// by ([`ExpertsKernel::forward`]).
    peaks: Vec<AtomicU64>,
    /// What a step's experts the cards do not hold have taken of late (us, f64 bits): an expert made on the host's
    /// cores, and a scratch call after its records' puts (the card's taking the writes in among it): how
    /// [`ExpertsKernel::forward_here`] shares a layer's between the two.
    host_us: AtomicU64,
    call_us: AtomicU64,
    /// Where the idle fill ([`WgpuExl3Experts::fill_some`]) is in its order.
    cursor: Mutex<usize>,
    counts: Counts,
    /// Experts kept from a scratch slot since the caller last asked ([`ExpertsKernel::offer`]): RAM may let them go.
    kept: Mutex<Vec<(u32, u32)>>,
    /// Experts let go of to keep others since the caller last asked ([`ExpertsKernel::evicted`]): RAM let them go when
    /// they came here, so it reads them back.
    evicted: Mutex<Vec<(u32, u32)>>,
    /// Where a step's experts made on the host go to be kept ([`ExpertsKernel::consider`]): a thread of its own
    /// ([`WgpuExl3Experts::start_promoter`]), so a record's upload is off the steps' path.
    promoter: Mutex<Option<std::sync::mpsc::Sender<Promotion>>>,
    /// Promotions handed over and not yet made: past [`PROMOTIONS_PENDING`], more are not handed over.
    pending: std::sync::atomic::AtomicUsize,
    /// Experts the promoter kept since the caller last asked ([`ExpertsKernel::taken`]): RAM lets them go.
    promoted: Mutex<Vec<(u32, u32)>>,
    /// Each layer's card, where the trunk is split over the cards ([`super::trunk::offload_split`]): a layer's held
    /// experts are on its card alone, beside its weights. None: anywhere.
    home: Option<Vec<usize>>,
    /// The cards.
    cards: Vec<Arc<WgpuBackend>>,
    /// Where the router's call makes a layer's held experts itself ([`ExpertsKernel::fused`]): each layer's pool (the
    /// one on its card that holds them all) and its table of where they are (each expert's slot there, `u32::MAX` for
    /// none: the card's own copy, written as they move, in the queue's order). Empty where not.
    fused: Vec<Option<(usize, ggml_rs::DeviceVec)>>,
}

/// A slot table's mark for an expert its pool does not hold.
const NO_SLOT: u32 = u32::MAX;

/// What the promoter is handed: a step's expert made on the host, `(layer, expert, record)`.
type Promotion = (u32, u32, Arc<Vec<u8>>);

/// Promotions queued at most: a reply on a new subject misses more than a card takes in, and those past this wait
/// for their next use.
const PROMOTIONS_PENDING: usize = 16;

/// Threads the promoter writes a record into a card's memory on (its own: the host's cores are the steps' experts').
const PROMOTE_THREADS: usize = 4;

fn record_layout(l: &[Exl3Proj; 3], bytes: usize) -> RecordLayout {
    let at = |p: &Exl3Proj| ProjAt { suh: p.suh, svh: p.svh, mul1: p.mul1, trellis: p.trellis, k: p.k, n: p.n, tile_words: p.tile_words };
    RecordLayout { bytes, proj: [at(&l[0]), at(&l[1]), at(&l[2])] }
}

/// A GGUF layer's record layout as its slots take it: its own three matrices' bytes (a record is the largest layer's).
fn quant_layout(l: &[GgufProj; 3]) -> QuantLayout {
    let at = |p: &GgufProj| QuantProj { at: p.at, dtype: p.ggml(), k: p.k, n: p.n };
    QuantLayout { bytes: l.iter().map(|p| p.at + p.bytes()).max().unwrap_or(0), proj: [at(&l[0]), at(&l[1]), at(&l[2])] }
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

/// A layout's projections as the CPU's decoder takes them.
fn layout_proj(l: &RecordLayout) -> [Exl3Proj; 3] {
    l.proj.map(|p| Exl3Proj { suh: p.suh, svh: p.svh, mul1: p.mul1, trellis: p.trellis, k: p.k, n: p.n, tile_words: p.tile_words })
}

/// A GGUF layout's matrices as the CPU's decoders take them.
fn gguf_proj(l: &QuantLayout) -> [GgufProj; 3] {
    l.proj.map(|p| GgufProj { dtype: p.dtype as u32, at: p.at, k: p.k, n: p.n })
}

impl WgpuExl3Experts {
    /// Slots on `cards` (the model's first card first, the trunk on it already) for a model of `per_layer` experts a
    /// layer: scratch slots on each, then on every card what its weight budget has left but [`MARGIN`], shared by the
    /// classes (bitrates, or a GGUF's pairs of types) by how many layers each has, in pools of what a binding holds.
    /// None if `format` is neither EXL3's nor a GGUF's, or not one slot fits.
    pub(crate) fn new(format: &ExpertFormat, cards: &[Arc<WgpuBackend>], per_layer: usize) -> Option<WgpuExl3Experts> {
        Self::new_within(format, cards, per_layer, None, None)
    }

    /// [`Self::new`] for a trunk split over `cards`, `home` each layer's card: a card's slots are for its layers'
    /// experts (by how many of its layers each bitrate has), and a layer's experts are held on its card alone.
    pub(crate) fn new_split(format: &ExpertFormat, cards: &[Arc<WgpuBackend>], per_layer: usize, home: Vec<usize>) -> Option<WgpuExl3Experts> {
        Self::new_within(format, cards, per_layer, None, Some(home))
    }

    /// [`Self::new`] with no more than `most` slots for held experts, the first card's room taken first (DSpark's
    /// drafter's 384 experts, all held, before the trunk's take the rest), and each layer's experts on its card of
    /// `home` where given.
    pub(crate) fn new_within(format: &ExpertFormat, cards: &[Arc<WgpuBackend>], per_layer: usize, most: Option<usize>, home: Option<Vec<usize>>) -> Option<WgpuExl3Experts> {
        let (layouts, gguf): (Vec<Layout>, bool) = match format {
            ExpertFormat::Exl3 { layers, bytes } => (layers.iter().map(|l| Layout::Exl3(record_layout(l, *bytes))).collect(), false),
            ExpertFormat::Gguf { layers, .. } => (layers.iter().map(|l| Layout::Quant(quant_layout(l))).collect(), true),
            ExpertFormat::Fp4 => return None,
        };
        let mut left = most.unwrap_or(usize::MAX);
        layouts.first()?;
        // each class's layers (on each card, where the layers have cards), and a layer of it
        let mut classes: Vec<([usize; 2], usize, usize)> = Vec::new();
        for (li, l) in layouts.iter().enumerate() {
            let class = l.class();
            match classes.iter_mut().find(|(t, ..)| *t == class) {
                Some(c) => c.1 += 1,
                None => classes.push((class, 1, li)),
            }
        }
        let on_card = |c: usize, class: [usize; 2]| -> (usize, usize) {
            match &home {
                Some(h) => (layouts.iter().zip(h).filter(|(l, &hc)| hc == c && l.class() == class).count(), h.iter().filter(|&&hc| hc == c).count()),
                None => (classes.iter().find(|(t, ..)| *t == class).map_or(0, |c| c.1), layouts.len()),
            }
        };
        let mut pools = Vec::new();
        // (a tier that holds all of its experts puts few through scratch slots: a few of them)
        let scratch = if most.is_some() { 4 } else if gguf { GGUF_SCRATCH } else { SCRATCH };
        for (c, card) in cards.iter().enumerate() {
            for &(class, _, li) in &classes {
                if let Some(slots) = layouts[li].slots_on(card, scratch, true) {
                    let held = Scratch::new(slots.slots());
                    pools.push(Pool { slots, class, scratch: true, card: c, busy: Mutex::new(held) });
                }
            }
        }
        for (c, card) in cards.iter().enumerate() {
            let (used, budget) = card.usage();
            let room = budget.saturating_sub(used).saturating_sub(MARGIN);
            for &(class, _, li) in &classes {
                let (n, of) = on_card(c, class);
                if n == 0 {
                    continue;
                }
                let per = layouts[li].slot_bytes();
                let share = most.map_or(usize::MAX, |most| most * n / layouts.len());
                let mut want = ((room as f64 * n as f64 / of as f64 / per as f64) as usize).min(share).min(left);
                while want > 0 {
                    let Some(slots) = layouts[li].slots_on(card, want, false) else { break };
                    want = want.saturating_sub(slots.slots());
                    left = left.saturating_sub(slots.slots());
                    pools.push(Pool { slots, class, scratch: false, card: c, busy: Mutex::new(Scratch::default()) });
                }
            }
        }
        if !pools.iter().any(|p| !p.scratch) {
            return None;
        }
        // Each layer given a pool of its trunk's card (its own card where the trunk is split, else the first) to hold its
        // experts in first, the pools of a bitrate on a card shared by its layers in turn, so the router's call makes
        // those there (a pass of the held experts' kernels takes one pool); the rest go where there is room, as before.
        // (OAIY_DSV41_FUSED=0: none so; a drafter's experts, all held, neither)
        let fused: Vec<Option<(usize, ggml_rs::DeviceVec)>> = match &home {
            _ if std::env::var("OAIY_DSV41_FUSED").as_deref() == Ok("0") || most.is_some() || cards.is_empty() => Vec::new(),
            h => {
                let mut turn: HashMap<(usize, [usize; 2]), usize> = HashMap::new();
                layouts
                    .iter()
                    .enumerate()
                    .map(|(li, l)| (l, h.as_ref().map_or(0, |h| h[li])))
                    .map(|(l, c)| {
                        let class = l.class();
                        let mine: Vec<usize> = (0..pools.len()).filter(|&p| !pools[p].scratch && pools[p].card == c && pools[p].class == class).collect();
                        let at = turn.entry((c, class)).or_insert(0);
                        let p = *mine.get(*at % mine.len().max(1))?;
                        *at += 1;
                        let table = cards[c].vec_of_bytes(&vec![0xffu8; per_layer * 4]);
                        Some((p, table))
                    })
                    .collect()
            }
        };
        let keys: Vec<Vec<Option<(u32, u32)>>> = pools.iter().map(|p| if p.scratch { Vec::new() } else { vec![None; p.slots.slots()] }).collect();
        let free = keys.iter().map(Vec::len).collect();
        let placed = RwLock::new(Placement { at: HashMap::new(), keys, free });
        let uses = (0..layouts.len() * per_layer).map(|_| AtomicU32::new(0)).collect();
        let rates = cards.iter().map(|_| AtomicU64::new(RATE_GUESS.to_bits())).collect();
        let peaks = cards.iter().map(|_| AtomicU64::new(RATE_GUESS.to_bits())).collect();
        Some(WgpuExl3Experts { layouts, pools, placed, uses, per_layer, calls: AtomicU64::new(0), rates, peaks, host_us: AtomicU64::new(HOST_US_GUESS.to_bits()), call_us: AtomicU64::new(CALL_US_GUESS.to_bits()), cursor: Mutex::new(0), counts: Counts::default(),
            kept: Mutex::new(Vec::new()),
            evicted: Mutex::new(Vec::new()),
            promoter: Mutex::new(None),
            pending: std::sync::atomic::AtomicUsize::new(0),
            promoted: Mutex::new(Vec::new()),
            home,
            cards: cards.to_vec(),
            fused,
        })
    }

    /// `layer`'s card, where the layers have cards.
    fn home_of(&self, layer: u32) -> Option<usize> {
        self.home.as_ref().and_then(|h| h.get(layer as usize).copied())
    }

    /// Whether held pool `p` may hold `layer`'s experts: of its bitrate, on its card where it has one.
    fn may_hold(&self, p: usize, layer: u32) -> bool {
        let pool = &self.pools[p];
        !pool.scratch && pool.class == self.class(layer) && self.home_of(layer).is_none_or(|h| pool.card == h)
    }

    /// The pool the router's call makes `layer`'s held experts from, where it has one ([`Self::fused_parts`]): the one
    /// its experts go to first, the card's other pools taking what that has no room for (those made in a call of their
    /// own, as before).
    fn own_pool(&self, layer: u32) -> Option<usize> {
        self.fused.get(layer as usize).and_then(Option::as_ref).map(|(p, _)| *p)
    }

    /// `layer`'s pool and slot table where the router's call makes its held experts ([`ExpertsKernel::fused`]).
    pub(crate) fn fused_parts(&self, layer: u32) -> Option<(ggml_rs_wgpu::exl3::HeldSlots<'_>, &ggml_rs::DeviceVec)> {
        let (p, table) = self.fused.get(layer as usize)?.as_ref()?;
        match &self.pools[*p].slots {
            Slots::Exl3(s) => Some((ggml_rs_wgpu::exl3::HeldSlots::Exl3(s), table)),
            Slots::Quant(s) => Some((ggml_rs_wgpu::exl3::HeldSlots::Quant(s), table)),
        }
    }

    /// `layer`'s `expert` written into its slot table as in `slot` (or [`NO_SLOT`]), where it has one.
    fn mark(&self, layer: u32, expert: u32, slot: u32) {
        if let Some((p, table)) = self.fused.get(layer as usize).and_then(Option::as_ref) {
            self.cards[self.pools[*p].card].write_words(table, expert as usize, &[slot]);
        }
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

    /// Since the model loaded: experts computed from their slots, experts the cards did not hold (put through the
    /// scratch slots, or made on the host), records taken into a free slot (from RAM or the drive), experts kept from
    /// a scratch slot.
    pub(crate) fn counts(&self) -> (u64, u64, u64, u64) {
        let c = &self.counts;
        let missed = c.scratch.load(Ordering::Relaxed) + c.host.load(Ordering::Relaxed);
        (c.held.load(Ordering::Relaxed), missed, c.taken.load(Ordering::Relaxed), c.kept.load(Ordering::Relaxed))
    }

    /// Of the experts computed from their slots since the model loaded, those on a card past the first.
    pub(crate) fn held_past_first(&self) -> u64 {
        self.counts.held_past_first.load(Ordering::Relaxed)
    }

    /// The first two cards' held-experts calls since the last ask, as the host found them once it had made the rest of
    /// a layer's experts: the mean time it then waited for each card's (us), how many calls, and how many still
    /// running; reset.
    /// The first two cards' puts into their scratch slots since the last ask: how many, their mean time (us), their
    /// rate (GB/s), and the experts the slots held still; reset.
    pub(crate) fn take_puts(&self) -> [(u64, f64, f64, u64); 2] {
        self.counts.puts.each_ref().map(|[ns, n, bytes, reused]| {
            let (ns, n, bytes) = (ns.swap(0, Ordering::Relaxed), n.swap(0, Ordering::Relaxed), bytes.swap(0, Ordering::Relaxed));
            (n, ns as f64 / 1e3 / n.max(1) as f64, bytes as f64 / ns.max(1) as f64, reused.swap(0, Ordering::Relaxed))
        })
    }

    pub(crate) fn take_held_waits(&self) -> [(f64, u64, u64); 2] {
        self.counts.held_wait.each_ref().map(|[ns, n, late]| {
            let (ns, n, late) = (ns.swap(0, Ordering::Relaxed), n.swap(0, Ordering::Relaxed), late.swap(0, Ordering::Relaxed));
            (ns as f64 / 1e3 / n.max(1) as f64, n, late)
        })
    }

    /// The class of `layer`'s experts ([`Layout::class`]).
    fn class(&self, layer: u32) -> [usize; 2] {
        self.layouts[layer as usize].class()
    }

    fn uses_of(&self, layer: u32, expert: u32) -> u32 {
        self.uses.get(layer as usize * self.per_layer + expert as usize).map_or(0, |n| n.load(Ordering::Relaxed))
    }

    /// A free slot of `layer`'s bitrate (on `card`, where one is named; on the layer's own card, where it has one, and
    /// none on another): `(pool, slot)`.
    fn free_slot(&self, placed: &Placement, layer: u32, card: Option<usize>) -> Option<(usize, usize)> {
        // (the layer's own pool first, where the router's call makes its experts)
        let fits = |p: usize| self.may_hold(p, layer) && card.is_none_or(|c| self.pools[p].card == c) && placed.free[p] > 0;
        let p = self.own_pool(layer).filter(|&p| fits(p)).or_else(|| (0..self.pools.len()).find(|&p| fits(p)))?;
        Some((p, placed.keys[p].iter().position(Option::is_none).expect("a pool with a free slot has one")))
    }

    /// The slot tables' marks of `expert` of `layer` coming into slot `at` of pool `pool` in place of `gone` (where
    /// that is the layer's own pool, its table has them): what [`Exl3Slots::copy_slot_marked`] writes with the copy.
    fn marks(&self, layer: u32, expert: u32, (pool, at): (usize, usize), gone: Option<(u32, u32)>) -> Vec<(&ggml_rs::DeviceVec, usize, u32)> {
        let table = |l: u32| self.fused.get(l as usize).and_then(Option::as_ref).filter(|(p, _)| *p == pool).map(|(_, t)| t);
        let mut marks = Vec::new();
        if let Some((gl, ge)) = gone {
            if let Some(t) = table(gl) {
                marks.push((t, ge as usize, NO_SLOT));
            }
        }
        if let Some(t) = table(layer) {
            marks.push((t, expert as usize, at as u32));
        }
        marks
    }

    /// `layer`'s `expert` into a free slot of its class, if there is one and it is not held already: whether it was.
    pub(crate) fn take(&self, layer: u32, expert: u32, record: &[u8]) -> bool {
        let mut placed = self.placed.write().unwrap_or_else(|p| p.into_inner());
        if placed.at.contains_key(&(layer, expert)) {
            return false;
        }
        let Some((p, s)) = self.free_slot(&placed, layer, None) else { return false };
        // Written into a scratch slot of its card in place and copied from there by the card: the queue's writes went
        // through staging memory at some 3 GB/s, where the host's threads write a card's memory at 11 to 22.
        let (card, class, layout) = (self.pools[p].card, self.pools[p].class, &self.layouts[layer as usize]);
        match self.pools.iter().find(|q| q.scratch && q.card == card && q.class == class) {
            Some(q) => {
                let mut held = q.busy.lock().unwrap_or_else(|e| e.into_inner());
                let at = held.oldest();
                q.slots.put_many(at, &[record], layout);
                held.mark(at, (layer, expert));
                q.slots.copy_slot_marked(at, &self.pools[p].slots, s, &self.marks(layer, expert, (p, s), None));
            }
            None => {
                self.pools[p].slots.put(s, record, layout);
                if self.own_pool(layer) == Some(p) {
                    self.mark(layer, expert, s as u32);
                }
            }
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
        // (a layer with a card of its own keeps its experts there alone)
        if self.home_of(layer).is_some_and(|h| h != card) {
            return;
        }
        let mut placed = self.placed.write().unwrap_or_else(|p| p.into_inner());
        if placed.at.contains_key(&(layer, expert)) {
            return;
        }
        let at = match self.free_slot(&placed, layer, Some(card)) {
            Some(at) => at,
            None if uses < KEEP_USES => return,
            None => {
                let mut least: Option<(usize, usize, u32)> = None;
                for p in (0..self.pools.len()).filter(|&p| self.may_hold(p, layer) && self.pools[p].card == card) {
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
        let marks = self.marks(layer, expert, at, placed.keys[at.0][at.1]);
        self.pools[from.0].slots.copy_slot_marked(from.1, &self.pools[at.0].slots, at.1, &marks);
        match placed.keys[at.0][at.1].replace((layer, expert)) {
            Some(gone) => {
                placed.at.remove(&gone);
                self.evicted.lock().unwrap_or_else(|p| p.into_inner()).push(gone);
            }
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

    /// Start the thread that keeps on the cards what steps made on the host and use enough ([`ExpertsKernel::
    /// consider`]): it holds these weakly, and ends when they are dropped (its channel closed with them).
    pub(crate) fn start_promoter(self: &Arc<Self>) {
        let (tx, rx) = std::sync::mpsc::channel::<Promotion>();
        let me = Arc::downgrade(self);
        let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(PROMOTE_THREADS).thread_name(|i| format!("oaiy-expert-promote-{i}")).build() else { return };
        let started = std::thread::Builder::new().name("oaiy-expert-promote".into()).spawn(move || {
            while let Ok((layer, e, record)) = rx.recv() {
                let Some(k) = me.upgrade() else { break };
                pool.install(|| k.promote(layer, e, &record));
                k.pending.fetch_sub(1, Ordering::Relaxed);
            }
        });
        if started.is_ok() {
            *self.promoter.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
        }
    }

    /// Keep `layer`'s `expert` (`record`, a step made it on the host) on the card it does the most good on: one with a
    /// free slot of its bitrate, else the one whose least used of another layer's experts is used least, where that
    /// is used less than this and this [`KEEP_USES`] times or more. Its record written into that card's scratch slot
    /// and copied from there by the card ([`Self::keep`]), as a step's scratch slots' experts are kept.
    fn promote(&self, layer: u32, expert: u32, record: &[u8]) {
        let (uses, class) = (self.uses_of(layer, expert), self.class(layer));
        let card = {
            let placed = self.placement();
            if placed.at.contains_key(&(layer, expert)) {
                return;
            }
            match self.free_slot(&placed, layer, None) {
                Some((p, _)) => self.pools[p].card,
                None if uses < KEEP_USES => return,
                None => {
                    let mut least: Option<(usize, u32)> = None;
                    for p in (0..self.pools.len()).filter(|&p| self.may_hold(p, layer)) {
                        for &(l, e) in placed.keys[p].iter().flatten() {
                            let n = self.uses_of(l, e);
                            if l != layer && least.is_none_or(|(_, m)| n < m) {
                                least = Some((self.pools[p].card, n));
                            }
                        }
                    }
                    match least {
                        Some((card, n)) if n < uses => card,
                        _ => return,
                    }
                }
            }
        };
        let Some(q) = (0..self.pools.len()).find(|&q| self.pools[q].scratch && self.pools[q].card == card && self.pools[q].class == class) else { return };
        let mut held = self.pools[q].busy.lock().unwrap_or_else(|e| e.into_inner());
        let at = held.oldest();
        self.pools[q].slots.put_many(at, &[record], &self.layouts[layer as usize]);
        held.mark(at, (layer, expert));
        self.keep(layer, expert, (q, at));
        if self.elsewhere(layer, expert) {
            self.promoted.lock().unwrap_or_else(|p| p.into_inner()).push((layer, expert));
        }
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

    /// `card`'s best rate of late ([`Self::peaks`]): a few slow puts (a card waking from a low power state took them in
    /// at a sixth of its rate) do not send every miss to the other cards after them, where its rate then stayed low,
    /// none put there to raise it (OAIY_DSV41_SPLIT_BY_PEAK=0: shared by the rates of late, as before).
    fn share_rate(&self, card: usize) -> f64 {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("OAIY_DSV41_SPLIT_BY_PEAK").as_deref() == Ok("0")) {
            return self.rate(card);
        }
        f64::from_bits(self.peaks[card].load(Ordering::Relaxed))
    }

    /// How many of `layer`'s `misses` (experts no card holds) a step sends through the cards' scratch slots while the
    /// host's cores make the rest: the share whose cards' time (a call, and each record's put at the fastest card's
    /// rate) and the host's (each expert's) end soonest, the fewer to the cards where two end alike, and one at least
    /// here (so the host's time is known as it changes). 0 where this bitrate has no scratch slots.
    fn share_to_cards(&self, layer: u32, misses: usize) -> usize {
        let class = self.class(layer);
        let Some(rate) = self.pools.iter().filter(|p| p.scratch && p.class == class).map(|p| self.rate(p.card)).reduce(f64::max) else { return 0 };
        let (host, call) = (f64::from_bits(self.host_us.load(Ordering::Relaxed)), f64::from_bits(self.call_us.load(Ordering::Relaxed)));
        let put = self.layouts[layer as usize].bytes() as f64 / rate.max(1.0);
        let ends = |k: usize| ((misses - k) as f64 * host).max(if k > 0 { call + k as f64 * put } else { 0.0 });
        (0..misses).min_by(|&a, &b| ends(a).total_cmp(&ends(b))).unwrap_or(0)
    }

    /// `jobs` (of `list`) through scratch pool `p`, a group of its slots at a time: each one's output, weighted. An
    /// expert the pool's slots hold still (a recent pass put it there) is made from its slot as it is, the rest put into
    /// the slots used longest ago (runs of slots side by side in one put). Each expert is then kept on the card where
    /// it is used enough ([`Self::keep`]).
    fn through_scratch(&self, layer: u32, p: usize, jobs: &[ExpertJob<'_>], list: &[usize], limit: f32) -> Vec<(usize, Vec<f32>)> {
        let (pool, layout) = (&self.pools[p], &self.layouts[layer as usize]);
        let mut held = pool.busy.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(list.len());
        for group in list.chunks(pool.slots.slots()) {
            // each job's slot: the one that holds its expert, else one of the oldest, in slot order
            let mut at: Vec<Option<usize>> = group.iter().map(|&i| held.find((layer, jobs[i].expert))).collect();
            let mut free: Vec<usize> = (0..held.keys.len()).filter(|s| !at.contains(&Some(*s))).collect();
            free.sort_by_key(|&s| held.used[s]);
            let mut fresh: Vec<usize> = free[..at.iter().filter(|a| a.is_none()).count()].to_vec();
            fresh.sort_unstable();
            let mut fresh = fresh.into_iter();
            for a in at.iter_mut().filter(|a| a.is_none()) {
                *a = fresh.next();
            }
            let at: Vec<usize> = at.into_iter().map(|a| a.expect("a slot for every job: a group is no more than the slots")).collect();
            let mut puts: Vec<(usize, &[u8])> = group.iter().zip(&at).filter(|&(&i, _)| held.find((layer, jobs[i].expert)).is_none()).map(|(&i, &s)| (s, jobs[i].record)).collect();
            puts.sort_by_key(|&(s, _)| s);
            if !puts.is_empty() {
                let uploading = std::time::Instant::now();
                for run in puts.chunk_by(|a, b| b.0 == a.0 + 1) {
                    let records: Vec<&[u8]> = run.iter().map(|&(_, r)| r).collect();
                    pool.slots.put_many(run[0].0, &records, layout);
                }
                let took = uploading.elapsed().as_secs_f64() * 1e6;
                crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertUpload, uploading);
                if let Some([ns, n, bytes, _]) = self.counts.puts.get(pool.card) {
                    ns.fetch_add((took * 1e3) as u64, Ordering::Relaxed);
                    n.fetch_add(puts.len() as u64, Ordering::Relaxed);
                    bytes.fetch_add((puts.len() * layout.bytes()) as u64, Ordering::Relaxed);
                }
                // (the card's rate, a fifth from this put)
                let seen = (puts.len() * layout.bytes()) as f64 / took.max(1.0);
                self.rates[pool.card].store((self.rate(pool.card) * 0.8 + seen * 0.2).to_bits(), Ordering::Relaxed);
                // (its best of late: the best seen, a 200th let go of at each put)
                let peak = f64::from_bits(self.peaks[pool.card].load(Ordering::Relaxed));
                self.peaks[pool.card].store((peak * 0.995).max(seen).to_bits(), Ordering::Relaxed);
            }
            if let Some([.., reused]) = self.counts.puts.get(pool.card) {
                reused.fetch_add((group.len() - puts.len()) as u64, Ordering::Relaxed);
            }
            for (&i, &s) in group.iter().zip(&at) {
                held.mark(s, (layer, jobs[i].expert));
            }
            let these: Vec<(usize, &[f32])> = group.iter().zip(&at).map(|(&i, &s)| (s, jobs[i].x)).collect();
            let calling = std::time::Instant::now();
            let ys = pool.slots.forward(&these, limit);
            ease(&self.call_us, calling.elapsed().as_secs_f64() * 1e6);
            for (y, &i) in ys.into_iter().zip(group) {
                out.push((i, weighed(y, jobs[i].weights)));
            }
            for (&i, &s) in group.iter().zip(&at) {
                self.keep(layer, jobs[i].expert, (p, s));
            }
        }
        out
    }

    /// Which of `scratch` (scratch pools) holds `layer`'s `expert` in a slot still, where one does.
    fn scratch_holding(&self, scratch: &[usize], layer: u32, expert: u32) -> Option<usize> {
        (0..scratch.len()).find(|&k| self.pools[scratch[k]].busy.lock().unwrap_or_else(|e| e.into_inner()).find((layer, expert)).is_some())
    }
}

impl ExpertsKernel for WgpuExl3Experts {
    /// Experts the cards do not hold: their records through the scratch slots of every card, each card given them by
    /// its rate of putting them in (each job to the card that would be done with it first), the cards at once.
    fn forward(&self, layer: u32, jobs: &[ExpertJob<'_>], swiglu_limit: f32) -> Vec<Vec<f32>> {
        let class = self.class(layer);
        let layout = &self.layouts[layer as usize];
        let scratch: Vec<usize> = (0..self.pools.len()).filter(|&p| self.pools[p].scratch && self.pools[p].class == class).collect();
        if scratch.is_empty() {
            // (no scratch slots of this class: on the CPU)
            return jobs
                .iter()
                .map(|j| {
                    match layout {
                        Layout::Exl3(l) => crate::dsv41::exl3_experts::forward_rows(j.record, &layout_proj(l), j.x, Some(j.weights), swiglu_limit),
                        Layout::Quant(l) => crate::dsv41::gguf_experts::forward_rows(j.record, &gguf_proj(l), j.x, Some(j.weights), swiglu_limit),
                    }
                    .unwrap_or_default()
                })
                .collect();
        }
        self.counts.scratch.fetch_add(jobs.len() as u64, Ordering::Relaxed);
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); scratch.len()];
        let mut load = vec![0f64; scratch.len()];
        // (an expert a pool's slots hold still to that pool, no put)
        for i in 0..jobs.len() {
            if let Some(k) = self.scratch_holding(&scratch, layer, jobs[i].expert) {
                lists[k].push(i);
                continue;
            }
            let k = (0..scratch.len())
                .min_by(|&a, &b| {
                    let done = |k: usize| (load[k] + layout.bytes() as f64) / self.share_rate(self.pools[scratch[k]].card);
                    done(a).total_cmp(&done(b))
                })
                .expect("a scratch pool");
            lists[k].push(i);
            load[k] += layout.bytes() as f64;
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
        // (the placement held until the cards' calls are submitted: a slot the promoter gives another expert is copied
        // into in the queue's order, so a call submitted before the copy reads the expert it found there)
        let placed = self.placement();
        let places: Vec<(usize, usize)> = jobs.iter().map(|(e, ..)| *placed.at.get(&(layer, *e)).expect("a held expert")).collect();
        self.counts.held_past_first.fetch_add(places.iter().filter(|(p, _)| self.pools[*p].card > 0).count() as u64, Ordering::Relaxed);
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
        let begun: Vec<(usize, Vec<Vec<usize>>, SlotsBegun<'a>)> = by_card
            .into_iter()
            .map(|(card, pools)| {
                let these: Vec<Vec<(usize, &[f32])>> = pools.iter().map(|(_, list)| list.iter().map(|&i| (places[i].1, jobs[i].1)).collect()).collect();
                let parts: Vec<(&'a Slots, &[(usize, &[f32])])> = pools.iter().zip(&these).map(|((p, _), t)| (&self.pools[*p].slots, &t[..])).collect();
                let call = Slots::begin_many(&parts, swiglu_limit);
                (card, pools.into_iter().map(|(_, list)| list).collect(), call)
            })
            .collect();
        drop(placed);
        let weights: Vec<Vec<f32>> = jobs.iter().map(|j| j.2.to_vec()).collect();
        let begun_in = beginning.elapsed();
        Box::new(move || {
            let finishing = std::time::Instant::now();
            let mut out: Vec<Vec<f32>> = vec![Vec::new(); weights.len()];
            // (which cards the host waits for, now that it has made the rest: each found running, and the wait from
            // here to its outputs)
            let running: Vec<bool> = begun.iter().map(|(_, _, call)| !call.done()).collect();
            for ((card, lists, call), late) in begun.into_iter().zip(running) {
                let ys = call.finish();
                if let Some([ns, n, l]) = self.counts.held_wait.get(card) {
                    ns.fetch_add(finishing.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    n.fetch_add(1, Ordering::Relaxed);
                    l.fetch_add(late as u64, Ordering::Relaxed);
                }
                for (list, ys) in lists.into_iter().zip(ys) {
                    for (i, y) in list.into_iter().zip(ys) {
                        out[i] = weighed(y, &weights[i]);
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

    fn evicted(&self) -> Vec<(u32, u32)> {
        std::mem::take(&mut *self.evicted.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// On the host's cores where it has the vectors to make them there quickly (`ggml_rs_wgpu::exl3::step_host_fast`:
    /// some half a millisecond an expert, where a card took a record in at one to two and a half), as the cards'
    /// slots make them to the bit (`step_experts_host`), so a reply is the same whichever tier held what.
    fn forward_here(&self, layer: u32, jobs: &[ExpertJob<'_>], swiglu_limit: f32) -> Option<Vec<Vec<f32>>> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // (OAIY_DSV41_HOST_EXPERTS=0: through the cards' scratch slots, as before)
        if *OFF.get_or_init(|| std::env::var("OAIY_DSV41_HOST_EXPERTS").as_deref() == Ok("0")) || !ggml_rs_wgpu::exl3::step_host_fast() {
            return None;
        }
        // (a GGUF's misses through the scratch slots alone: the host's decoding sums otherwise than the cards')
        let Layout::Exl3(layout) = &self.layouts[layer as usize] else { return None };
        let d = crate::dsv41::expert::DIM;
        if jobs.iter().any(|j| j.x.len() != d || j.weights.len() != 1) {
            return None;
        }
        // Some of a layer's misses through the cards' scratch slots, which take a record in over their links while the
        // host's cores make the rest: as many as are done no later than those, by what each has taken of late (the
        // same bits either way). On idle cores an expert here took some half a millisecond and a card's a millisecond
        // and more, so a few misses stay here; another program on the cores made one here two, and a card's then paid.
        // (OAIY_DSV41_HYBRID=0: all of them here)
        static HYBRID: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let hybrid = *HYBRID.get_or_init(|| std::env::var("OAIY_DSV41_HYBRID").as_deref() != Ok("0"));
        let to_cards = if hybrid { self.share_to_cards(layer, jobs.len()) } else { 0 };
        let (theirs, mine) = jobs.split_at(to_cards);
        self.counts.host.fetch_add(mine.len() as u64, Ordering::Relaxed);
        let these: Vec<(&[u8], &[f32])> = mine.iter().map(|j| (j.record, j.x)).collect();
        let (made_there, made_here) = std::thread::scope(|scope| {
            let there = (!theirs.is_empty()).then(|| scope.spawn(|| self.forward(layer, theirs, swiglu_limit)));
            let making = std::time::Instant::now();
            let here = ggml_rs_wgpu::exl3::step_experts_host(&these, layout, swiglu_limit, true);
            if !these.is_empty() {
                ease(&self.host_us, making.elapsed().as_secs_f64() * 1e6 / these.len() as f64);
            }
            (there.map(|h| h.join().expect("the cards' share of a step's experts panicked")).unwrap_or_default(), here)
        });
        Some(made_there.into_iter().chain(made_here.into_iter().zip(mine).map(|(y, j)| weighed(y, j.weights))).collect())
    }

    fn consider(&self, layer: u32, records: &[(u32, Arc<Vec<u8>>)]) {
        let Some(tx) = self.promoter.lock().unwrap_or_else(|p| p.into_inner()).clone() else { return };
        for (e, record) in records {
            if self.pending.load(Ordering::Relaxed) >= PROMOTIONS_PENDING {
                break;
            }
            if self.elsewhere(layer, *e) || (self.uses_of(layer, *e) < KEEP_USES && !self.has_room(layer)) {
                continue;
            }
            self.pending.fetch_add(1, Ordering::Relaxed);
            if tx.send((layer, *e, Arc::clone(record))).is_err() {
                self.pending.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    fn taken(&self) -> Vec<(u32, u32)> {
        std::mem::take(&mut *self.promoted.lock().unwrap_or_else(|p| p.into_inner()))
    }

    fn keeps(&self, layer: u32, expert: u32) -> bool {
        self.elsewhere(layer, expert)
    }

    fn fused(&self, layer: u32) -> Option<&dyn std::any::Any> {
        self.fused_parts(layer).map(|_| self as &dyn std::any::Any)
    }

    fn in_fused(&self, layer: u32, expert: u32) -> bool {
        let own = self.own_pool(layer);
        own.is_some() && self.placement().at.get(&(layer, expert)).map(|&(p, _)| p) == own
    }

    fn took_fused(&self, layer: u32, n: usize) {
        self.counts.held.fetch_add(n as u64, Ordering::Relaxed);
        if self.home_of(layer).is_some_and(|c| c > 0) {
            self.counts.held_past_first.fetch_add(n as u64, Ordering::Relaxed);
        }
    }
}
