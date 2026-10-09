//! The first card's tier of experts kept between passes.

use super::*;

/// Decode steps after which every count halves (as the CUDA engine's tier): without aging, an old topic's experts would
/// keep the slots for good.
const AGE_TOKENS: u64 = 128;

/// Experts kept on the adapter between passes, as the CUDA engine keeps them in VRAM (dsv41-cuda's expert_cache): a
/// slot a record; the one to replace the least used, and of equals the longest unused (LFRU); uses counted in tokens (a
/// prompt's expert that 300 tokens chose counts 300) and halved every [`AGE_TOKENS`] decode steps. A decode step's
/// misses come in when they are used more than what they would replace; a prompt's are noted and, once it is read, the
/// most used come in from RAM (taking them in during the pass would evict experts the same pass needs later). What
/// comes in can leave the RAM tier, as the CUDA engine's tiers do (`WgpuExperts::set_exclusive`), but by default stays.
pub(super) struct Resident {
    pub(super) slots: RecordSlots,
    pub(super) keys: Vec<Option<(u32, u32)>>,
    pub(super) last: Vec<u64>,
    pub(super) index: std::collections::HashMap<(u32, u32), usize>,
    pub(super) freq: std::collections::HashMap<(u32, u32), u64>,
    pub(super) clock: u64,
    pub(super) decoded: u64,
    pub(super) pending: Vec<(u32, u32)>,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) admitted: u64,
    /// Whether a full tier replaces its least used with what is used more as requests go (the tiers inclusive: what
    /// it lets go is still in RAM), or changes only by [`WgpuExperts::rebalance`] (exclusive: what it let go on a
    /// request's path would be in no tier).
    pub(super) replaces: bool,
}

impl Resident {
    pub(super) fn new(slots: RecordSlots) -> Resident {
        let n = slots.len();
        Resident {
            slots,
            keys: vec![None; n],
            last: vec![0; n],
            index: std::collections::HashMap::with_capacity(n),
            freq: std::collections::HashMap::new(),
            clock: 0,
            decoded: 0,
            pending: Vec::new(),
            hits: 0,
            misses: 0,
            admitted: 0,
            replaces: true,
        }
    }

    pub(super) fn freq(&self, key: (u32, u32)) -> u64 {
        self.freq.get(&key).copied().unwrap_or(0)
    }

    pub(super) fn holds(&mut self, layer: u32, experts: &[u32], tokens: &[usize]) -> Vec<bool> {
        self.clock += 1;
        experts
            .iter()
            .zip(tokens)
            .map(|(&e, &n)| {
                let key = (layer, e);
                *self.freq.entry(key).or_insert(0) += n as u64;
                match self.index.get(&key) {
                    Some(&i) => {
                        self.last[i] = self.clock;
                        self.hits += 1;
                        true
                    }
                    None => {
                        self.misses += 1;
                        self.pending.push(key);
                        false
                    }
                }
            })
            .collect()
    }

    /// The slot to fill next: an empty one, else (where the tier replaces as it goes) the least used, of equals the
    /// longest unused.
    pub(super) fn victim(&self) -> Option<usize> {
        if let Some(i) = self.keys.iter().position(Option::is_none) {
            return Some(i);
        }
        if !self.replaces {
            return None;
        }
        (0..self.keys.len()).min_by_key(|&i| (self.keys[i].map_or(0, |k| self.freq(k)), self.last[i]))
    }

    /// Take in `key`'s record if it is not held and a slot is free, or (where the tier replaces as it goes) it is used
    /// more than what it would replace (a tie keeps the resident one: each swap is an upload).
    pub(super) fn admit(&mut self, key: (u32, u32), record: &[u8]) -> bool {
        if self.index.contains_key(&key) {
            return false;
        }
        let Some(i) = self.victim() else { return false };
        if let Some(old) = self.keys[i] {
            if self.freq(old) >= self.freq(key) {
                return false;
            }
        }
        self.put(i, key, record);
        true
    }

    /// `key`'s record into slot `i`, in place of what it held.
    pub(super) fn put(&mut self, i: usize, key: (u32, u32), record: &[u8]) {
        if let Some(old) = self.keys[i] {
            self.index.remove(&old);
        }
        self.slots.write(i, record);
        self.keys[i] = Some(key);
        self.index.insert(key, i);
        self.last[i] = self.clock;
        self.admitted += 1;
    }

    pub(super) fn pass_done(&mut self, tokens: usize, cache: &oaiy_engine::ecache::Ecache, store: &dyn oaiy_engine::store::WeightStore, exclusive: bool) {
        let mut pending = std::mem::take(&mut self.pending);
        if tokens > 1 {
            pending.sort_unstable();
            pending.dedup();
            pending.sort_by_key(|&k| std::cmp::Reverse(self.freq(k)));
            for key in pending {
                if self.index.contains_key(&key) || !cache.probe(key.0, key.1) {
                    continue;
                }
                let Ok(record) = cache.acquire(key.0, key.1, store) else { continue };
                // most used first: once one does not beat what it would replace, none after it will
                if !self.admit(key, &record) {
                    break;
                }
                // held here now, so out of RAM: the two tiers hold different experts
                drop(record);
                if exclusive {
                    cache.remove(key.0, key.1);
                }
            }
        } else {
            self.decoded += 1;
            if self.decoded % AGE_TOKENS == 0 {
                for f in self.freq.values_mut() {
                    *f /= 2;
                }
            }
        }
    }
}
