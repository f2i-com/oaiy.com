//! The RAM tier: a bounded cache of expert records in front of a
//! [`WeightStore`].
//!
//! A big MoE model does not fit in RAM (DeepSeek-V4.1-Flash is 510 GB), so
//! its experts live on the NVMe and this cache decides how often a token
//! pays for a read. It holds at most `budget / record_size` records.
//!
//! **Policy.** Which record to drop is chosen from a random draw of
//! [`SAMPLES`] resident records: the one used least often, and of equals the
//! one used longest ago ([`CachePolicy::Lfru`]); or just the one used
//! longest ago ([`CachePolicy::Lru`]). Routing in a MoE is skewed towards
//! popular experts, which rewards frequency; the random draw keeps a newly
//! read record from being the certain next victim, so a record gets the
//! chance to prove itself before the old favourites crowd it out. When the
//! cache holds no more records than the draw size, every record is
//! considered.
//!
//! **Passes.** A prefill runs the layers in order and uses each layer's
//! experts once, so a record it reads is the least used one around, and
//! plain LFRU evicts, for it, a record of a layer the pass has yet to reach:
//! the pass eats its own future (measured on DeepSeek-V4.1: a full RAM tier
//! served 2.6K of the 10.2K records it held that a 5.6K-token prompt
//! needed). [`Ecache::set_scan_layer`] tells the cache where a pass is;
//! victims then come from the layers below it first.
//!
//! **Aging.** The counts are accesses since a record was read, so a record a
//! long conversation used thousands of times would outrank, for good, the ones
//! the next conversation uses. [`Ecache::decay`] halves every count; its owner
//! calls it as its tokens go by (dsv41 every few replies' worth), and the cache
//! then holds what has been used of late. [`Ecache::credit`] gives a record a
//! count it earned elsewhere (in a tier that has just handed it back), where
//! it would arrive as a newcomer and be among the first to go.
//!
//! **Leases.** [`Ecache::acquire`] hands out a [`HostLease`], a shared
//! handle to the record's immutable bytes. A hit copies nothing, and a
//! record with a live lease is never chosen for eviction, so a caller can
//! compute from it without holding any lock. If every record is leased or
//! still being read, a miss reads around the cache instead of waiting.
//!
//! **Concurrency.** One mutex guards the bookkeeping and is never held
//! during a read. A miss inserts a placeholder, reads with the lock
//! released, then publishes the bytes; any other thread asking for the same
//! record meanwhile waits for that read rather than issuing its own.
//!
//! **Counters.** `hits` and `misses` count demand accesses; `bytes_read` is
//! what the misses cost from the store and `bytes_hit` what the hits saved.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::error::{Error, Result};
use crate::store::WeightStore;
use crate::types::CachePolicy;

/// Resident records drawn when choosing what to evict.
pub const SAMPLES: usize = 16;

/// Evicted buffers kept for reuse by the next reads. A fresh record-sized
/// allocation is zeroed and faulted in page by page; reusing one skips that.
const SPARE_BUFFERS: usize = 4;

type Key = (u32, u32);

/// A shared, read-only handle to one cached record. Dereferences to the
/// record bytes, clones cheaply, and keeps the bytes alive (and the record
/// in the cache) for as long as it exists.
#[derive(Clone)]
pub struct HostLease(Arc<Vec<u8>>);

impl HostLease {
    /// The bytes as an `Arc`, for handing to another thread. It pins the
    /// record exactly as the lease does.
    pub fn to_arc(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.0)
    }
}

impl Deref for HostLease {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for HostLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostLease({} bytes)", self.0.len())
    }
}

/// Counters since the cache was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Demand accesses served from RAM.
    pub hits: u64,
    /// Demand accesses (and admitted records) that had to be read.
    pub misses: u64,
    /// Bytes read from the store for those misses.
    pub bytes_read: u64,
    /// Records dropped to make room.
    pub evictions: u64,
    /// Bytes served from RAM by the hits.
    pub bytes_hit: u64,
}

struct Entry {
    /// `None` while the record is being read.
    bytes: Option<Arc<Vec<u8>>>,
    /// Accesses since the record was read.
    uses: u64,
    /// Tick of the latest access.
    last: u64,
    /// Index in `Book::resident` once the bytes are in.
    slot: usize,
}

impl Entry {
    fn loaded(&self) -> bool {
        self.bytes.is_some()
    }

    /// Loaded and not leased: only the cache holds the bytes.
    fn evictable(&self) -> bool {
        self.bytes.as_ref().is_some_and(|b| Arc::strong_count(b) == 1)
    }
}

/// Everything the mutex guards.
struct Book {
    entries: HashMap<Key, Entry>,
    /// Keys whose bytes are in, for picking eviction candidates.
    resident: Vec<Key>,
    tick: u64,
    rng: u64,
    stats: CacheStats,
    spare: Vec<Vec<u8>>,
    /// A pass through the layers is at this one (see `set_scan_layer`).
    scan: Option<u32>,
}

impl Book {
    fn rand(&mut self) -> u64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// The record to drop, if any resident record is not leased.
    fn victim(&mut self, policy: CachePolicy) -> Option<Key> {
        let n = self.resident.len();
        let rank = |e: &Entry| match policy {
            CachePolicy::Lfru => (e.uses, e.last),
            CachePolicy::Lru => (0, e.last),
        };
        let lowest = |book: &Book, keys: &mut dyn Iterator<Item = Key>| {
            keys.filter_map(|k| {
                let e = &book.entries[&k];
                e.evictable().then(|| (rank(e), k))
            })
            .min()
            .map(|(_, k)| k)
        };
        if let Some(floor) = self.scan {
            // the layers below the pass are done with until the next pass;
            // the rest are about to be used (a full scan: a pass evicts a few
            // thousand times, against minutes of reading)
            let mut done = self.resident.iter().copied().filter(|k| k.0 < floor);
            if let Some(k) = lowest(self, &mut done) {
                return Some(k);
            }
        }
        if n > SAMPLES {
            let mut draw = [(0, 0); SAMPLES];
            for d in &mut draw {
                let i = (self.rand() % n as u64) as usize;
                *d = self.resident[i];
            }
            if let Some(k) = lowest(self, &mut draw.into_iter()) {
                return Some(k);
            }
        }
        // a small cache, or a draw that found only leased records
        lowest(self, &mut self.resident.iter().copied())
    }

    fn evict(&mut self, k: Key) {
        self.drop_entry(k);
        self.stats.evictions += 1;
    }

    /// Take a resident record out, keeping its buffer for reuse.
    fn drop_entry(&mut self, k: Key) {
        let e = self.entries.remove(&k).expect("resident key is cached");
        self.resident.swap_remove(e.slot);
        if let Some(&moved) = self.resident.get(e.slot) {
            self.entries.get_mut(&moved).expect("resident key is cached").slot = e.slot;
        }
        if let Some(Ok(buf)) = e.bytes.map(Arc::try_unwrap) {
            if self.spare.len() < SPARE_BUFFERS {
                self.spare.push(buf);
            }
        }
    }

    /// Free a place for one more record; false when every record is leased
    /// or still being read.
    fn make_room(&mut self, capacity: usize, policy: CachePolicy) -> bool {
        if self.entries.len() < capacity {
            return true;
        }
        match self.victim(policy) {
            Some(k) => {
                self.evict(k);
                true
            }
            None => false,
        }
    }

    fn publish(&mut self, k: Key, bytes: Arc<Vec<u8>>) {
        let slot = self.resident.len();
        self.resident.push(k);
        let e = self.entries.get_mut(&k).expect("placeholder is cached");
        e.bytes = Some(bytes);
        e.slot = slot;
    }

    fn count_miss(&mut self, rec: usize) {
        self.stats.misses += 1;
        self.stats.bytes_read += rec as u64;
    }
}

/// The expert cache. Every method takes `&self`; share it between threads
/// behind an `Arc`.
pub struct Ecache {
    book: Mutex<Book>,
    /// Signalled whenever a read finishes, successfully or not.
    read_done: Condvar,
    rec_bytes: usize,
    /// The budget and the records it holds: set when the cache is made, and again by [`Ecache::set_budget`].
    budget_bytes: std::sync::atomic::AtomicUsize,
    capacity: std::sync::atomic::AtomicUsize,
    policy: CachePolicy,
}

impl Ecache {
    /// A cache of `budget_bytes / rec_bytes` records. A budget below one
    /// record disables caching: every access reads from the store.
    pub fn new(budget_bytes: usize, rec_bytes: usize, policy: CachePolicy) -> Ecache {
        let capacity = if rec_bytes == 0 { 0 } else { budget_bytes / rec_bytes };
        Ecache {
            book: Mutex::new(Book {
                entries: HashMap::with_capacity(capacity),
                resident: Vec::with_capacity(capacity),
                tick: 0,
                rng: 0x9E37_79B9_7F4A_7C15,
                stats: CacheStats::default(),
                spare: Vec::new(),
                scan: None,
            }),
            read_done: Condvar::new(),
            rec_bytes,
            budget_bytes: std::sync::atomic::AtomicUsize::new(budget_bytes),
            capacity: std::sync::atomic::AtomicUsize::new(capacity),
            policy,
        }
    }

    /// Change the budget to `budget_bytes`: more records may then be held, or, where it is less than what is held,
    /// records are let go (as a miss lets them go: the least used first) until what is left fits or every record
    /// left is leased. A cache is sized from the memory free when its model loads; this is for when that changes
    /// (another program's memory come free, or wanted). The records it holds now.
    pub fn set_budget(&self, budget_bytes: usize) -> usize {
        use std::sync::atomic::Ordering;
        let capacity = if self.rec_bytes == 0 { 0 } else { budget_bytes / self.rec_bytes };
        let mut g = self.book();
        self.budget_bytes.store(budget_bytes, Ordering::Relaxed);
        self.capacity.store(capacity, Ordering::Relaxed);
        while g.entries.len() > capacity {
            let Some(k) = g.victim(self.policy) else { break };
            g.evict(k);
        }
        capacity
    }

    fn book(&self) -> MutexGuard<'_, Book> {
        // Nothing in `Book` is left half-updated across a panic point, so a
        // poisoned lock is still consistent.
        self.book.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn wait<'a>(&self, g: MutexGuard<'a, Book>) -> MutexGuard<'a, Book> {
        self.read_done.wait(g).unwrap_or_else(|p| p.into_inner())
    }

    /// Records the budget holds.
    pub fn n_slots(&self) -> usize {
        self.capacity.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Bytes per record.
    pub fn rec_bytes(&self) -> usize {
        self.rec_bytes
    }

    /// The budget the cache was sized from.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    /// False when the budget buys no records and every access reads.
    pub fn is_enabled(&self) -> bool {
        self.n_slots() > 0
    }

    /// Records resident right now.
    pub fn len(&self) -> usize {
        self.book().resident.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> CacheStats {
        self.book().stats
    }

    /// Share of demand accesses served from RAM (0 before any access).
    pub fn hit_rate(&self) -> f64 {
        let s = self.stats();
        let total = s.hits + s.misses;
        if total == 0 {
            0.0
        } else {
            s.hits as f64 / total as f64
        }
    }

    /// Whether `(layer, expert)` is resident. Counts nothing and changes no
    /// ranking; a record still being read reports false.
    pub fn probe(&self, layer: u32, expert: u32) -> bool {
        self.book().entries.get(&(layer, expert)).is_some_and(Entry::loaded)
    }

    /// Drop `(layer, expert)` to free its place (another tier holds it now,
    /// say). A leased or still-loading record stays. Returns whether it was
    /// dropped. Not counted as an eviction.
    pub fn remove(&self, layer: u32, expert: u32) -> bool {
        let mut g = self.book();
        let key = (layer, expert);
        if !g.entries.get(&key).is_some_and(Entry::evictable) {
            return false;
        }
        g.drop_entry(key);
        true
    }

    /// Copy the record into `dst` (at least [`rec_bytes`](Self::rec_bytes)
    /// long), reading it through `store` on a miss.
    pub fn get(&self, layer: u32, expert: u32, store: &dyn WeightStore, dst: &mut [u8]) -> Result<()> {
        if dst.len() < self.rec_bytes {
            return Err(Error::Arg(format!(
                "expert cache: {}-byte buffer for a {}-byte record",
                dst.len(),
                self.rec_bytes
            )));
        }
        let lease = self.acquire(layer, expert, store)?;
        dst[..self.rec_bytes].copy_from_slice(&lease);
        Ok(())
    }

    /// A pass through the layers in order (a prefill) is at `layer`; `None`
    /// when it ends. Meanwhile room is made from the layers below it first,
    /// which the pass is done with, instead of from the records it is about
    /// to use; the usual policy applies when none of those can go.
    pub fn set_scan_layer(&self, layer: Option<u32>) {
        self.book().scan = layer;
    }

    /// The record for `(layer, expert)`, read through `store` on a miss.
    pub fn acquire(&self, layer: u32, expert: u32, store: &dyn WeightStore) -> Result<HostLease> {
        let key = (layer, expert);
        let mut g = self.book();
        g.tick += 1;
        let now = g.tick;
        loop {
            match g.entries.get_mut(&key) {
                Some(e) if e.loaded() => {
                    e.uses += 1;
                    e.last = now;
                    let lease = HostLease(Arc::clone(e.bytes.as_ref().expect("loaded")));
                    g.stats.hits += 1;
                    g.stats.bytes_hit += self.rec_bytes as u64;
                    return Ok(lease);
                }
                // another thread is reading it: wait, then look again (the
                // read may have failed, or the record been evicted already)
                Some(_) => g = self.wait(g),
                None => break,
            }
        }

        g.count_miss(self.rec_bytes);
        if !self.is_enabled() || !g.make_room(self.n_slots(), self.policy) {
            drop(g);
            return self.read_around(layer, expert, store);
        }
        g.entries.insert(key, Entry { bytes: None, uses: 1, last: now, slot: usize::MAX });
        let mut buf = g.spare.pop().unwrap_or_else(|| vec![0; self.rec_bytes]);
        drop(g);

        let read = store.fetch(layer, expert, &mut buf);

        let mut g = self.book();
        let out = match read {
            Ok(()) => {
                let bytes = Arc::new(buf);
                g.publish(key, Arc::clone(&bytes));
                Ok(HostLease(bytes))
            }
            Err(e) => {
                g.entries.remove(&key);
                if g.spare.len() < SPARE_BUFFERS {
                    g.spare.push(buf);
                }
                Err(e)
            }
        };
        drop(g);
        self.read_done.notify_all();
        out
    }

    /// A read that bypasses the cache (already counted as a miss).
    fn read_around(&self, layer: u32, expert: u32, store: &dyn WeightStore) -> Result<HostLease> {
        let mut buf = vec![0; self.rec_bytes];
        store.fetch(layer, expert, &mut buf)?;
        Ok(HostLease(Arc::new(buf)))
    }

    /// Halve every record's count of accesses (see the module's "Aging").
    pub fn decay(&self) {
        for e in self.book().entries.values_mut() {
            e.uses /= 2;
        }
    }

    /// Raise `(layer, expert)`'s count of accesses to `uses` if it is resident and has fewer: a record that comes
    /// with a history. Whether it was resident.
    pub fn credit(&self, layer: u32, expert: u32, uses: u64) -> bool {
        match self.book().entries.get_mut(&(layer, expert)) {
            Some(e) if e.loaded() => {
                e.uses = e.uses.max(uses);
                true
            }
            _ => false,
        }
    }

    /// Insert a record the caller read itself (a batched read, say),
    /// taking ownership of the buffer. Counted as a miss, since it cost a
    /// read. If the record is already resident, or arrives meanwhile
    /// through a concurrent read, that copy is kept (the bytes are the
    /// same). A disabled or fully leased cache counts the read and keeps
    /// nothing.
    pub fn admit_owned(&self, layer: u32, expert: u32, record: Vec<u8>) -> Result<()> {
        if record.len() != self.rec_bytes {
            return Err(Error::Arg(format!(
                "expert cache: admitted record is {} bytes, records are {}",
                record.len(),
                self.rec_bytes
            )));
        }
        let key = (layer, expert);
        let mut g = self.book();
        g.tick += 1;
        let now = g.tick;
        g.count_miss(self.rec_bytes);
        if !self.is_enabled() {
            return Ok(());
        }
        loop {
            match g.entries.get_mut(&key) {
                Some(e) if e.loaded() => {
                    e.last = now;
                    return Ok(());
                }
                Some(_) => g = self.wait(g),
                None => break,
            }
        }
        if g.make_room(self.n_slots(), self.policy) {
            g.entries.insert(key, Entry { bytes: None, uses: 1, last: now, slot: usize::MAX });
            g.publish(key, Arc::new(record));
        }
        Ok(())
    }
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    struct Store;
    impl WeightStore for Store {
        fn record_bytes(&self) -> usize {
            4
        }
        fn shape(&self) -> (u32, u32) {
            (8, 8)
        }
        fn fetch(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()> {
            dst.copy_from_slice(&[layer as u8, expert as u8, 0, 0]);
            Ok(())
        }
    }

    /// A pass over layers 0..4, three experts each, through a full cache
    /// holding two of each layer's: with the scan hint room comes from the
    /// layers done, so only layer 0's miss (nothing is done yet) costs a
    /// record the pass needs later; without it, every miss does.
    #[test]
    fn a_pass_keeps_the_records_it_is_about_to_use() {
        let run = |hint: bool| {
            let c = Ecache::new(8 * 4, 4, CachePolicy::Lfru);
            // resident before the pass: two experts of every layer
            for l in 0..4 {
                for e in 0..2 {
                    c.acquire(l, e, &Store).unwrap();
                }
            }
            let before = c.stats();
            for l in 0..4 {
                c.set_scan_layer(hint.then_some(l));
                for e in 0..3 {
                    c.acquire(l, e, &Store).unwrap();
                }
            }
            c.set_scan_layer(None);
            c.stats().hits - before.hits
        };
        assert_eq!(run(true), 7);
        assert_eq!(run(false), 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records whose bytes name their key; counts fetches; can be told to
    /// fail one key.
    struct Store {
        rec: usize,
        reads: AtomicUsize,
        fail: Option<Key>,
        delay_ms: u64,
    }

    impl Store {
        fn new(rec: usize) -> Store {
            Store { rec, reads: AtomicUsize::new(0), fail: None, delay_ms: 0 }
        }
        fn record(&self, l: u32, e: u32) -> Vec<u8> {
            (0..self.rec).map(|i| (l as usize * 31 + e as usize * 7 + i) as u8).collect()
        }
        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    impl WeightStore for Store {
        fn record_bytes(&self) -> usize {
            self.rec
        }
        fn shape(&self) -> (u32, u32) {
            (64, 64)
        }
        fn fetch(&self, l: u32, e: u32, dst: &mut [u8]) -> Result<()> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if self.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            }
            if self.fail == Some((l, e)) {
                return Err(Error::Format(format!("bad record ({l}, {e})")));
            }
            dst.copy_from_slice(&self.record(l, e));
            Ok(())
        }
    }

    /// A budget set anew: a cache of two holds a third record once its budget is three records', and lets go of its
    /// least used when the budget is one record's again.
    #[test]
    fn a_budget_set_anew_makes_room_or_lets_records_go() {
        let s = Store::new(32);
        let c = Ecache::new(64, 32, CachePolicy::Lfru);
        for e in 0..2 {
            c.acquire(0, e, &s).unwrap();
        }
        c.acquire(0, 1, &s).unwrap();
        assert_eq!((c.n_slots(), c.len()), (2, 2));
        assert_eq!(c.set_budget(96), 3);
        c.acquire(0, 2, &s).unwrap();
        c.acquire(0, 2, &s).unwrap();
        assert_eq!((c.n_slots(), c.len(), c.budget_bytes()), (3, 3, 96));
        assert_eq!(c.set_budget(40), 1);
        assert_eq!(c.len(), 1);
        assert!(c.probe(0, 1) || c.probe(0, 2), "one of the two used twice is what is left");
        assert!(!c.probe(0, 0), "the one used once went first");
    }

    /// A record's count decides whether it stays: of two records in a cache of two, the one used ten times is
    /// kept when a third arrives, until the counts are aged away and the other is used; and a record credited
    /// with a count it earned elsewhere is kept as if it had been used that often here.
    #[test]
    fn aged_counts_let_go_of_an_old_favourite_and_a_credited_record_stays() {
        let s = Store::new(32);
        let c = Ecache::new(64, 32, CachePolicy::Lfru);
        for _ in 0..10 {
            c.acquire(0, 0, &s).unwrap();
        }
        c.acquire(0, 1, &s).unwrap();
        c.acquire(0, 2, &s).unwrap();
        assert!(c.probe(0, 0) && !c.probe(0, 1) && c.probe(0, 2), "the favourite stays, the other of the two goes");
        // four halvings: ten uses are none, and two uses of the newcomer outrank them
        for _ in 0..4 {
            c.decay();
        }
        c.acquire(0, 2, &s).unwrap();
        c.acquire(0, 2, &s).unwrap();
        c.acquire(0, 3, &s).unwrap();
        assert!(!c.probe(0, 0) && c.probe(0, 2) && c.probe(0, 3), "the old favourite goes once its count has aged away");
        // 3 arrived last with one use; credited with nine it outranks 2's two
        assert!(c.credit(0, 3, 9) && !c.credit(0, 0, 9));
        c.acquire(0, 4, &s).unwrap();
        assert!(c.probe(0, 3) && !c.probe(0, 2) && c.probe(0, 4));
    }

    #[test]
    fn disabled_cache_reads_every_time() {
        let s = Store::new(32);
        let c = Ecache::new(16, 32, CachePolicy::Lfru);
        assert!(!c.is_enabled());
        for _ in 0..3 {
            assert_eq!(&*c.acquire(1, 2, &s).unwrap(), &s.record(1, 2)[..]);
        }
        let st = c.stats();
        assert_eq!((st.hits, st.misses, st.bytes_read), (0, 3, 96));
        assert_eq!(s.reads(), 3);
        assert!(!c.probe(1, 2));
    }

    #[test]
    fn second_access_hits_without_reading() {
        let s = Store::new(64);
        let c = Ecache::new(4 * 64, 64, CachePolicy::Lfru);
        let mut dst = vec![0u8; 64];
        c.get(3, 9, &s, &mut dst).unwrap();
        assert_eq!(dst, s.record(3, 9));
        assert!(c.probe(3, 9));
        c.get(3, 9, &s, &mut dst).unwrap();
        assert_eq!(dst, s.record(3, 9));
        assert_eq!(s.reads(), 1);
        let st = c.stats();
        assert_eq!((st.hits, st.misses, st.bytes_hit, st.bytes_read), (1, 1, 64, 64));
        assert!((c.hit_rate() - 0.5).abs() < 1e-12);
        assert!(c.get(3, 9, &s, &mut [0u8; 8]).is_err(), "short buffer accepted");
    }

    #[test]
    fn never_holds_more_than_the_budget() {
        let s = Store::new(16);
        let c = Ecache::new(5 * 16 + 7, 16, CachePolicy::Lfru);
        assert_eq!(c.n_slots(), 5);
        for e in 0..40 {
            c.acquire(0, e, &s).unwrap();
            assert!(c.len() <= 5);
        }
        assert_eq!(c.len(), 5);
        assert_eq!(c.stats().evictions, 35);
    }

    #[test]
    fn lfru_keeps_the_popular_record() {
        let s = Store::new(8);
        let c = Ecache::new(3 * 8, 8, CachePolicy::Lfru);
        for _ in 0..5 {
            c.acquire(0, 0, &s).unwrap(); // popular
        }
        for e in 1..20 {
            c.acquire(0, e, &s).unwrap(); // a stream of one-offs
        }
        assert!(c.probe(0, 0), "frequency did not protect the popular record");
    }

    #[test]
    fn lru_drops_the_oldest() {
        let s = Store::new(8);
        let c = Ecache::new(2 * 8, 8, CachePolicy::Lru);
        for _ in 0..5 {
            c.acquire(0, 0, &s).unwrap();
        }
        c.acquire(0, 1, &s).unwrap();
        c.acquire(0, 2, &s).unwrap(); // evicts (0, 0): least recent
        assert!(!c.probe(0, 0));
        assert!(c.probe(0, 1) && c.probe(0, 2));
    }

    #[test]
    fn leased_records_are_never_evicted() {
        let s = Store::new(8);
        let c = Ecache::new(2 * 8, 8, CachePolicy::Lfru);
        let a = c.acquire(0, 0, &s).unwrap();
        let b = c.acquire(0, 1, &s).unwrap();
        // both leased: a third record reads around the cache
        let x = c.acquire(0, 2, &s).unwrap();
        assert_eq!(&*x, &s.record(0, 2)[..]);
        assert!(c.probe(0, 0) && c.probe(0, 1) && !c.probe(0, 2));
        // an Arc taken from a lease pins just the same
        let pinned = a.to_arc();
        drop(a);
        c.acquire(0, 3, &s).unwrap();
        assert!(c.probe(0, 0) && !c.probe(0, 3));
        // released: now (0, 0) can go, and (0, 1) is still leased
        drop(pinned);
        c.acquire(0, 3, &s).unwrap();
        assert!(!c.probe(0, 0) && c.probe(0, 3));
        assert_eq!(&*b, &s.record(0, 1)[..]);
    }

    #[test]
    fn remove_frees_a_place_but_not_a_leased_record() {
        let s = Store::new(8);
        let c = Ecache::new(2 * 8, 8, CachePolicy::Lfru);
        c.acquire(0, 0, &s).unwrap();
        let held = c.acquire(0, 1, &s).unwrap();
        assert!(!c.remove(0, 1), "a leased record was removed");
        assert!(c.remove(0, 0));
        assert!(!c.remove(0, 0), "removed twice");
        assert_eq!((c.len(), c.stats().evictions), (1, 0));
        // the freed place takes a new record without evicting
        c.acquire(0, 2, &s).unwrap();
        assert!(c.probe(0, 1) && c.probe(0, 2));
        drop(held);
    }

    #[test]
    fn failed_reads_are_not_cached() {
        let mut s = Store::new(8);
        s.fail = Some((2, 2));
        let c = Ecache::new(4 * 8, 8, CachePolicy::Lfru);
        assert!(c.acquire(2, 2, &s).is_err());
        assert!(!c.probe(2, 2));
        assert!(c.acquire(2, 2, &s).is_err(), "a failed read is retried");
        assert_eq!(s.reads(), 2);
        assert_eq!(c.len(), 0);
        c.acquire(2, 3, &s).unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn admitted_records_serve_hits() {
        let s = Store::new(8);
        let c = Ecache::new(4 * 8, 8, CachePolicy::Lfru);
        c.admit_owned(1, 1, s.record(1, 1)).unwrap();
        assert!(c.probe(1, 1));
        assert_eq!(&*c.acquire(1, 1, &s).unwrap(), &s.record(1, 1)[..]);
        assert_eq!(s.reads(), 0);
        // a second admit keeps the resident copy but counts its read
        c.admit_owned(1, 1, vec![0xEE; 8]).unwrap();
        assert_eq!(&*c.acquire(1, 1, &s).unwrap(), &s.record(1, 1)[..]);
        let st = c.stats();
        assert_eq!((st.misses, st.hits), (2, 2));
        assert!(c.admit_owned(1, 2, vec![0; 7]).is_err(), "wrong size accepted");
        // disabled: counted, not kept
        let off = Ecache::new(0, 8, CachePolicy::Lfru);
        off.admit_owned(0, 0, vec![0; 8]).unwrap();
        assert_eq!(off.stats().misses, 1);
        assert!(!off.probe(0, 0));
    }

    #[test]
    fn concurrent_misses_on_one_record_read_it_once() {
        let mut s = Store::new(4096);
        s.delay_ms = 30;
        let c = Ecache::new(8 * 4096, 4096, CachePolicy::Lfru);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let r = c.acquire(5, 6, &s).unwrap();
                    assert_eq!(&*r, &s.record(5, 6)[..]);
                });
            }
        });
        assert_eq!(s.reads(), 1);
        let st = c.stats();
        assert_eq!((st.misses, st.hits), (1, 7));
    }

    #[test]
    fn many_threads_many_records_stay_consistent() {
        let s = Store::new(256);
        let c = Ecache::new(24 * 256, 256, CachePolicy::Lfru);
        std::thread::scope(|scope| {
            for t in 0..8u32 {
                let (c, s) = (&c, &s);
                scope.spawn(move || {
                    for i in 0..500u32 {
                        let e = (i * 7 + t * 13) % 60;
                        let r = c.acquire(e % 3, e, s).unwrap();
                        assert_eq!(&*r, &s.record(e % 3, e)[..]);
                    }
                });
            }
        });
        let st = c.stats();
        assert_eq!(st.hits + st.misses, 8 * 500);
        assert!(c.len() <= 24);
    }
}
