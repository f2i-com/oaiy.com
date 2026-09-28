//! The VRAM tier of the expert hierarchy (Phase C2): a byte-budgeted pool of
//! fixed record slots carved out of one device allocation, keyed by
//! `(layer, expert)`, evicting by LFRU (frequency first, recency as the
//! tiebreak — the policy the route-trace simulation measured at ~76% warm
//! hits for ~1,200 slots per GPU on a 247-token document).
//!
//! A hit costs zero PCIe bytes: the kernels read the slot in place. A miss
//! takes a record from OAIY's host cache (RAM tier, or the SSD behind it)
//! and uploads it into the victim slot on the compute stream, so stream
//! order alone guarantees the upload lands before the kernels that read it,
//! and that no kernel still reading a victim's bytes is overtaken by the
//! upload that replaces them (both happen on one stream, in issue order).
//! Prefill's own misses are the exception: they cross on the copy stream,
//! into two staging slots, so an upload runs while the last expert computes.
//! Events stand in for stream order there (see `get_prefill`).
//!
//! Frequencies count tokens: a prefill expert used by 300 prompt tokens
//! counts 300, as 300 decode steps would. During a prefill, misses go
//! through staging slots ([`DeviceExpertCache::get_prefill`]); when it
//! ends, the ones used most move in where they beat what they would replace
//! ([`DeviceExpertCache::admit_pending`]), from RAM. (Admitting every miss
//! as it came evicted, measured, each upload before its next use and much
//! of the warm set; admitting mid-pass by frequency still evicted experts
//! the same pass needed later; admitting none cost the reply some speed.)

use std::collections::HashMap;

use cudarc::driver::{CudaEvent, CudaSlice, CudaView};
use oaiy_engine::ecache::Ecache;
use oaiy_engine::store::WeightStore;
use oaiy_engine::Result;

use crate::gpu::Gpu;

/// Staging slots a prefill hands out in turn. Two is enough to overlap:
/// the next expert's bytes land in one while this one's kernels read the
/// other.
const STAGE_SLOTS: usize = 2;

/// Every frequency halves after this many decode tokens: LFU without aging
/// keeps an old topic's experts in VRAM forever, and a new topic's never
/// collect enough uses to displace them. (Counted in tokens, not accesses:
/// a long prefill touches nearly every expert and would age the counts away.)
const AGE_TOKENS: u64 = 128;

/// Largest frequency a saved usage profile seeds: the profile picks which
/// experts start in VRAM, but should not shield them from this run's usage
/// for thousands of tokens.
const SEED_CAP: u64 = 64;

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Bytes uploaded over PCIe for misses.
    pub bytes_uploaded: u64,
    /// Misses left out of VRAM by [`DeviceExpertCache::worth_admitting`]
    /// (served elsewhere, e.g. on the CPU).
    pub declined: u64,
    /// Prefill: wall seconds waiting for a missing expert's bytes (RAM, or
    /// the drive through the readers), and in the host-to-device copy calls
    /// (which also wait for the stream's earlier work). Host clocks, no syncs.
    pub wait_s: f64,
    pub upload_s: f64,
}

struct Slot {
    key: Option<(u32, u32)>,
    freq: u64,
    last: u64,
    /// Batch that last used this slot; slots of the current batch are pinned.
    batch: u64,
}

pub struct DeviceExpertCache {
    pool: CudaSlice<u8>,
    /// Records for prefill's misses, handed out in turn. Two, so the next
    /// expert's upload crosses while this one's kernels run; with one, the
    /// upload had to wait for them.
    stage: Vec<CudaSlice<u8>>,
    next_stage: usize,
    /// Per stage slot: the bytes are in place, and the kernels that read
    /// them are done - the second only once it has been recorded.
    filled: Vec<CudaEvent>,
    used: Vec<CudaEvent>,
    used_valid: Vec<bool>,
    /// The slot handed out last. Its readers are issued by the time the
    /// caller asks for another expert, which is when `used` is recorded.
    live: Option<usize>,
    record: usize,
    slots: Vec<Slot>,
    index: HashMap<(u32, u32), usize>,
    clock: u64,
    /// Current batch (see [`begin_batch`](Self::begin_batch)).
    batch: u64,
    /// Access counts survive eviction, so a returning hot expert is not
    /// judged as new (a cheap frequency sketch; ~15k entries at most).
    freq: HashMap<(u32, u32), u64>,
    /// Decode tokens seen (for aging).
    tokens: u64,
    /// Prefill misses since the last [`admit_pending`](Self::admit_pending).
    pending: Vec<(u32, u32)>,
    pub stats: DeviceCacheStats,
}

impl DeviceExpertCache {
    /// `slots` records of `record` bytes each, allocated up front.
    pub fn new(g: &Gpu, slots: usize, record: usize) -> Result<DeviceExpertCache> {
        Ok(DeviceExpertCache {
            pool: g.zeros::<u8>(slots.max(1) * record)?,
            stage: (0..STAGE_SLOTS).map(|_| g.zeros::<u8>(record)).collect::<Result<Vec<_>>>()?,
            next_stage: 0,
            filled: (0..STAGE_SLOTS).map(|_| g.new_event()).collect::<Result<Vec<_>>>()?,
            used: (0..STAGE_SLOTS).map(|_| g.new_event()).collect::<Result<Vec<_>>>()?,
            used_valid: vec![false; STAGE_SLOTS],
            live: None,
            record,
            slots: (0..slots).map(|_| Slot { key: None, freq: 0, last: 0, batch: 0 }).collect(),
            index: HashMap::with_capacity(slots),
            clock: 0,
            batch: 1,
            freq: HashMap::new(),
            tokens: 0,
            pending: Vec::new(),
            stats: DeviceCacheStats::default(),
        })
    }

    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Every `(layer, expert)` this cache has been asked for, with how often:
    /// the device sees each access (a VRAM hit never reaches the host
    /// cache), so this is the true usage profile of its layers.
    pub fn usage(&self) -> impl Iterator<Item = ((u32, u32), u64)> + '_ {
        self.freq.iter().map(|(&k, &f)| (k, f))
    }

    /// Start `(layer, expert)`'s frequency at `freq` (a warm start from a
    /// saved profile), so recorded-hot experts are not the first victims.
    pub fn seed(&mut self, layer: u32, expert: u32, freq: u64) {
        let freq = freq.min(SEED_CAP);
        let f = self.freq.entry((layer, expert)).or_insert(0);
        *f = (*f).max(freq);
        if let Some(&i) = self.index.get(&(layer, expert)) {
            self.slots[i].freq = *f;
        }
    }

    /// How often `(layer, expert)` has been used here (token-weighted, aged,
    /// seeded from the usage profile).
    pub fn freq(&self, layer: u32, expert: u32) -> u64 {
        self.freq.get(&(layer, expert)).copied().unwrap_or(0)
    }

    pub fn contains(&self, layer: u32, expert: u32) -> bool {
        self.index.contains_key(&(layer, expert))
    }

    /// Start a batch: every slot [`get`](Self::get) returns from here until
    /// the next `begin_batch` is pinned (never chosen as a victim), so a
    /// grouped kernel can read all of a layer's experts at once without a
    /// later fetch in the same batch overwriting an earlier one's slot.
    pub fn begin_batch(&mut self) {
        self.batch += 1;
    }

    /// The device bytes of expert `(layer, expert)`, uploading it on a miss.
    pub fn get(&mut self, g: &Gpu, layer: u32, expert: u32, host: &Ecache, store: &dyn WeightStore) -> Result<CudaView<'_, u8>> {
        if let Some(i) = self.touch(layer, expert) {
            return Ok(self.view(i));
        }
        let lease = host.acquire(layer, expert, store)?;
        let i = self.place(g, (layer, expert), &lease)?;
        Ok(self.view(i))
    }

    /// Prefill's access for an expert `tokens` prompt tokens use: the
    /// resident slot on a hit; on a miss the record from the host cache (or
    /// the drive) in the staging slot, for this one use, noted for
    /// [`admit_pending`](Self::admit_pending). The staging slot is reused by
    /// the next call: stream order puts that upload after the kernels issued
    /// on this one.
    pub fn get_prefill(&mut self, g: &Gpu, layer: u32, expert: u32, tokens: u64, host: &Ecache, store: &dyn WeightStore) -> Result<CudaView<'_, u8>> {
        // asking again means the kernels reading the slot handed out last
        // are issued: mark where they end, for whoever takes that slot next
        if let Some(k) = self.live.take() {
            g.record_compute(&self.used[k])?;
            self.used_valid[k] = true;
        }
        if let Some(i) = self.touch_by(layer, expert, tokens) {
            return Ok(self.view(i));
        }
        self.pending.push((layer, expert));
        let t = std::time::Instant::now();
        let lease = host.acquire(layer, expert, store)?;
        let t1 = std::time::Instant::now();
        let k = self.next_stage;
        self.next_stage = (k + 1) % self.stage.len();
        let after = self.used_valid[k].then_some(&self.used[k]);
        g.write_staged_async(&lease, &mut self.stage[k].slice_mut(..), after, &self.filled[k])?;
        // the kernels this expert is about to get read those bytes
        g.compute_wait(&self.filled[k])?;
        self.live = Some(k);
        self.stats.wait_s += (t1 - t).as_secs_f64();
        self.stats.upload_s += t1.elapsed().as_secs_f64();
        self.stats.bytes_uploaded += self.record as u64;
        Ok(self.stage[k].slice(..))
    }

    /// After a prefill: move the experts it missed that it used most into
    /// VRAM, each while its token-weighted frequency beats the victim's, and
    /// only from RAM (nothing is read from the drive here). Returns how many.
    pub fn admit_pending(&mut self, g: &Gpu, host: &Ecache, store: &dyn WeightStore) -> Result<usize> {
        let mut keys = std::mem::take(&mut self.pending);
        keys.sort_unstable();
        keys.dedup();
        keys.sort_by_key(|k| std::cmp::Reverse(self.freq.get(k).copied().unwrap_or(0)));
        self.begin_batch();
        let mut admitted = 0;
        for key in keys {
            if self.index.contains_key(&key) || !host.probe(key.0, key.1) {
                continue;
            }
            // most-used first: once one does not beat the victim, none will
            if !self.beats_victim(self.freq.get(&key).copied().unwrap_or(0)) {
                break;
            }
            let lease = host.acquire(key.0, key.1, store)?;
            self.place(g, key, &lease)?;
            admitted += 1;
        }
        Ok(admitted)
    }

    /// Decode-time lookup that never uploads: the slot on a hit (pinned for
    /// this batch), `None` on a miss. Counts the access either way, so the
    /// frequencies [`worth_admitting`](Self::worth_admitting) compares
    /// include uses served elsewhere.
    pub fn lookup(&mut self, layer: u32, expert: u32) -> Option<CudaView<'_, u8>> {
        self.touch(layer, expert).map(|i| self.view(i))
    }

    /// Whether uploading `(layer, expert)` (just missed) would displace
    /// nothing, or only an expert used less often: LFRU admission. A tie
    /// keeps the resident one, so equally-cold experts do not churn the
    /// cache (each swap is a PCIe copy). Counts a refusal in `declined`.
    pub fn worth_admitting(&mut self, layer: u32, expert: u32) -> bool {
        let freq = self.freq.get(&(layer, expert)).copied().unwrap_or(0);
        let ok = self.beats_victim(freq);
        if !ok {
            self.stats.declined += 1;
        }
        ok
    }

    /// Whether a record used `freq` times would displace nothing, or only a
    /// record used less.
    fn beats_victim(&self, freq: u64) -> bool {
        match self.victim() {
            None => false,
            Some(i) => self.slots[i].key.is_none() || self.slots[i].freq < freq,
        }
    }

    /// Upload a record the caller already holds (a miss it decided to admit).
    pub fn insert(&mut self, g: &Gpu, layer: u32, expert: u32, record: &[u8]) -> Result<CudaView<'_, u8>> {
        let i = self.place(g, (layer, expert), record)?;
        Ok(self.view(i))
    }

    fn view(&self, i: usize) -> CudaView<'_, u8> {
        self.pool.slice(i * self.record..(i + 1) * self.record)
    }

    /// One decode token has run; every [`AGE_TOKENS`] halve the frequencies.
    pub fn tick_token(&mut self) {
        self.tokens += 1;
        if self.tokens.is_multiple_of(AGE_TOKENS) {
            self.age();
        }
    }

    /// Halve every frequency.
    fn age(&mut self) {
        self.freq.retain(|_, f| {
            *f /= 2;
            *f > 0
        });
        for s in &mut self.slots {
            s.freq /= 2;
        }
    }

    /// Count an access; on a hit refresh and pin the slot and return it.
    fn touch(&mut self, layer: u32, expert: u32) -> Option<usize> {
        self.touch_by(layer, expert, 1)
    }

    /// [`touch`](Self::touch) for an access by `tokens` tokens at once.
    fn touch_by(&mut self, layer: u32, expert: u32, tokens: u64) -> Option<usize> {
        self.clock += 1;
        let key = (layer, expert);
        let f = self.freq.entry(key).or_insert(0);
        *f += tokens.max(1);
        let freq = *f;
        match self.index.get(&key) {
            Some(&i) => {
                self.stats.hits += 1;
                let s = &mut self.slots[i];
                s.freq = freq;
                s.last = self.clock;
                s.batch = self.batch;
                Some(i)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// A free slot, else the least-frequently (then least-recently) used one
    /// not pinned by the current batch.
    fn victim(&self) -> Option<usize> {
        if let Some(i) = self.slots.iter().position(|s| s.key.is_none()) {
            return Some(i);
        }
        (0..self.slots.len()).filter(|&i| self.slots[i].batch != self.batch).min_by_key(|&i| (self.slots[i].freq, self.slots[i].last))
    }

    /// Upload `record` for `key` into the victim slot.
    fn place(&mut self, g: &Gpu, key: (u32, u32), record: &[u8]) -> Result<usize> {
        let i = self
            .victim()
            .ok_or_else(|| oaiy_engine::Error::Arg("every VRAM expert slot is pinned by this batch; the cache is smaller than one layer's experts".into()))?;
        if let Some(old) = self.slots[i].key.take() {
            self.index.remove(&old);
            self.stats.evictions += 1;
        }
        g.write(record, &mut self.pool.slice_mut(i * self.record..(i + 1) * self.record))?;
        self.stats.bytes_uploaded += self.record as u64;
        let freq = self.freq.get(&key).copied().unwrap_or(1);
        self.slots[i] = Slot { key: Some(key), freq, last: self.clock, batch: self.batch };
        self.index.insert(key, i);
        Ok(i)
    }
}
