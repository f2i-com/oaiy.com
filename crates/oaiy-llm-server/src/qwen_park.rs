//! Conversations the Qwen engine has set aside in host RAM.
//!
//! Qwen3.8-Flash-Next holds one conversation's state (`kv`, `covered`, the recurrent
//! checkpoints) and keeps nothing on disk. Two conversations that take turns on it (a runner
//! and a call's sub-agent, sharing little more than a system prompt) each read their whole
//! prompt again at every switch: 36 s for a 21,000-token prompt that costs 0.3 to 0.7 s when
//! the engine has it to itself. Here a state the engine is about to lose is copied to host RAM
//! first (about 60 KB a token), and comes back when a later prompt continues it: a switch costs
//! the copies instead of the reading (the copies are RAM at PCIe speed, the reading is the model).
//!
//! Only when the engine displaces a state, never after every turn; and never a private one.
//! A state comes back as the same bytes it left as, so a prompt that continues it reads on
//! from it exactly as it would have from the live cache. What decides is [`reusable`]: the
//! tokens a state would not read again, by the same rules `QwenEngine::generate` starts from.
use crate::qwen_cache::RecurrentSnapshot;
use ggml_rs::Tensor;
use llama_rs::KvCache;
use std::time::Instant;

/// A recurrent checkpoint the engine keeps with its state: the tokens it covers, its snapshot,
/// and whether it is the base one (the system prompt).
pub(crate) type Checkpoint = (Vec<u64>, RecurrentSnapshot, bool);

/// A state shorter than this is not kept or fetched back: reading it again is quick (1,024
/// tokens are under 2 s at the engine's 600 tokens a second) and it takes budget a longer one
/// would use. One of exactly this many is kept.
pub(crate) const MIN_TOKENS: usize = 1024;

/// A stashed state is brought back only when it saves this many more tokens than the live one.
pub(crate) const MIN_GAIN: usize = 1024;

fn common_prefix(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// How many leading tokens of `keys` a state that covers `covered` (with recurrent checkpoints
/// at `checkpoints`) would not read again: what `QwenEngine::generate` starts from. The state
/// reads on from where it ends when the prompt continues it exactly (and leaves a token to run
/// for the logits), else goes back to the longest checkpoint the prompt shares, if one lies past
/// that, else starts over.
pub(crate) fn reusable<'a>(covered: &[u64], checkpoints: impl IntoIterator<Item = &'a [u64]>, keys: &[u64]) -> usize {
    let common = common_prefix(covered, keys);
    let mut start = if common == covered.len() && common < keys.len() { common } else { 0 };
    for saved in checkpoints {
        if saved.len() <= common && saved.len() > start && saved.len() < keys.len() && keys.starts_with(saved) {
            start = saved.len();
        }
    }
    start
}

/// Whether a state a prompt is about to displace is worth keeping: at least [`MIN_TOKENS`], and
/// mostly unrelated to that prompt. A prompt that shares half of it or more is the same
/// conversation branching, which the checkpoints already hold.
pub(crate) fn worth_parking(held: usize, common: usize) -> bool {
    held >= MIN_TOKENS && common * 2 < held
}

/// What decides which stashed state comes back: its tokens, where its checkpoints lie, and when
/// it was set aside (a counter).
pub(crate) struct View<'a> {
    pub keys: &'a [u64],
    pub checkpoints: Vec<&'a [u64]>,
    pub used: u64,
}

/// The stashed state to bring back for `keys`, as (index, tokens it would not read again), when
/// it saves at least [`MIN_TOKENS`] and beats the live state's own `live` by [`MIN_GAIN`]. The
/// one that saves most wins; of equals, the one set aside last.
pub(crate) fn choose(views: &[View<'_>], live: usize, keys: &[u64]) -> Option<(usize, usize)> {
    views.iter().enumerate()
        .map(|(i, v)| (i, reusable(v.keys, v.checkpoints.iter().copied(), keys), v.used))
        .max_by_key(|&(_, reuse, used)| (reuse, used))
        .map(|(i, reuse, _)| (i, reuse))
        .filter(|&(_, reuse)| reuse >= MIN_TOKENS && reuse >= live.saturating_add(MIN_GAIN))
}

/// One slot of a cache. Qwen3.8-Flash-Next has a slot for each layer, one for the n-gram layer
/// and one for each full-attention layer's indexer keys, each with its own heads and width.
struct Slot {
    heads: usize,
    dim: usize,
    /// K and V rows `[0..len)`, when the slot holds rows.
    rows: Option<(Tensor, Tensor)>,
    ssm_state: Option<Tensor>,
    ssm_conv: Option<Tensor>,
}

/// A full copy of a cache in host RAM, whatever its shape. Unlike `Snapshot` it needs to know
/// nothing of the model (which layers attend, the recurrent shapes): it takes every slot as it
/// finds it, from the slot's own device.
pub(crate) struct ParkedState {
    len: usize,
    max_len: usize,
    slots: Vec<Slot>,
}

/// The cache's slots, all with a backend of their own (a lazy cache, as the model builds it):
/// how many there are.
fn layout(kv: &KvCache) -> Result<usize, String> {
    let n = kv.k.len();
    if kv.v.len() != n || kv.ssm_state.len() != n || kv.ssm_conv.len() != n
        || kv.n_kv_heads_per_layer.len() != n || kv.head_dims.len() != n || kv.layer_backends.len() != n {
        return Err("the cache's slots do not line up".into());
    }
    Ok(n)
}

/// Whether slot `i` holds a row for each of the first `kv.len` tokens. A slot nothing appends
/// to (a recurrent layer's: only its recurrent state is used) keeps the small buffer it began
/// with, which says nothing once the sequence is longer than that.
fn holds_rows(kv: &KvCache, i: usize) -> bool {
    kv.k[i].dim(0) >= kv.len && kv.v[i].dim(0) >= kv.len
}

impl ParkedState {
    /// What parking `kv` would hold, in bytes, before any of it is copied.
    pub(crate) fn estimate(kv: &KvCache) -> Result<usize, String> {
        let n = layout(kv)?;
        Ok(4 * (0..n).map(|i| {
            let rows = if holds_rows(kv, i) { 2 * kv.len * kv.n_kv_heads_per_layer[i] * kv.head_dims[i] } else { 0 };
            rows + kv.ssm_state[i].as_ref().map_or(0, Tensor::numel) + kv.ssm_conv[i].as_ref().map_or(0, Tensor::numel)
        }).sum::<usize>())
    }

    /// Copy `kv` to the host: every slot's rows `[0..len)`, read from the slot's own device,
    /// and its recurrent tensors. Reads only: `kv` is as it was, whatever happens.
    pub(crate) fn capture(kv: &KvCache) -> Result<Self, String> {
        let n = layout(kv)?;
        if kv.len == 0 || kv.len > kv.max_len {
            return Err("no tokens to park".into());
        }
        let mut slots = Vec::with_capacity(n);
        for i in 0..n {
            let backend = kv.layer_backends[i].as_ref();
            let (heads, dim) = (kv.n_kv_heads_per_layer[i], kv.head_dims[i]);
            let rows = if holds_rows(kv, i) {
                if kv.k[i].shape()[1..] != [heads, dim] || kv.v[i].shape()[1..] != [heads, dim] {
                    return Err(format!("slot {i}: its buffers are not [rows, {heads}, {dim}]"));
                }
                Some((backend.slice_axis0(&kv.k[i], kv.len).to_host(), backend.slice_axis0(&kv.v[i], kv.len).to_host()))
            } else {
                None
            };
            slots.push(Slot { heads, dim, rows, ssm_state: kv.ssm_state[i].as_ref().map(Tensor::to_host), ssm_conv: kv.ssm_conv[i].as_ref().map(Tensor::to_host) });
        }
        Ok(Self { len: kv.len, max_len: kv.max_len, slots })
    }

    pub(crate) fn bytes(&self) -> usize {
        4 * self.slots.iter().map(|s| {
            s.rows.iter().map(|(k, v)| k.numel() + v.numel()).sum::<usize>()
                + s.ssm_state.iter().chain(&s.ssm_conv).map(Tensor::numel).sum::<usize>()
        }).sum::<usize>()
    }

    /// Whether this fits `kv`: the same slots, heads, widths and context, and every tensor the
    /// shape it should be. Reads only.
    pub(crate) fn validate(&self, kv: &KvCache) -> Result<(), String> {
        let n = layout(kv)?;
        if self.slots.len() != n {
            return Err(format!("{} slots parked, the cache has {n}", self.slots.len()));
        }
        if self.max_len != kv.max_len || self.len == 0 || self.len > kv.max_len {
            return Err("the cache's context length is not the one parked from".into());
        }
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.heads != kv.n_kv_heads_per_layer[i] || slot.dim != kv.head_dims[i] {
                return Err(format!("slot {i}: {} heads of {} parked, the cache has {} of {}", slot.heads, slot.dim, kv.n_kv_heads_per_layer[i], kv.head_dims[i]));
            }
            if let Some((k, v)) = &slot.rows {
                let want = [self.len, slot.heads, slot.dim];
                if k.shape() != want || v.shape() != want || !k.is_cpu() || !v.is_cpu() {
                    return Err(format!("slot {i}: parked rows are not {want:?} on the host"));
                }
            }
            for (parked, live) in [(&slot.ssm_state, &kv.ssm_state[i]), (&slot.ssm_conv, &kv.ssm_conv[i])] {
                let Some(parked) = parked else { continue };
                if !parked.is_cpu() || live.as_ref().is_some_and(|live| live.shape() != parked.shape()) {
                    return Err(format!("slot {i}: a recurrent tensor does not fit the cache"));
                }
            }
        }
        Ok(())
    }

    /// Put this state into `kv`, which it replaces. Checked against `kv` before anything is
    /// touched: an `Err` leaves it as it was. After that the cache is being rewritten, and a
    /// panic (an allocation the device refuses) leaves it half done: the caller resets it.
    ///
    /// Each slot grows to the length through `reserve_layer`, the path lazy growth takes, so the
    /// buffers are the ones the model would have grown. Nothing else is keyed to them: a decode
    /// step's graph is captured again from the buffers at hand at every step (and the old graph
    /// updated to its addresses), see `FlashNext::forward`.
    pub(crate) fn restore(mut self, kv: &mut KvCache) -> Result<(), String> {
        self.validate(kv)?;
        #[cfg(test)]
        {
            if fixtures::INJECT_PANIC.with(|f| f.get()) {
                panic!("injected: the restore failed after the cache was reset");
            }
        }
        kv.reset();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let backend = kv.layer_backends[i].clone();
            if let Some((k, v)) = slot.rows.take() {
                kv.reserve_layer(backend.as_ref(), i, self.len);
                backend.copy_axis0_into(&mut kv.k[i], 0, &backend.to_device(k));
                backend.copy_axis0_into(&mut kv.v[i], 0, &backend.to_device(v));
            }
            kv.ssm_state[i] = slot.ssm_state.take().map(|t| backend.to_device(t));
            kv.ssm_conv[i] = slot.ssm_conv.take().map(|t| backend.to_device(t));
        }
        kv.len = self.len;
        Ok(())
    }
}

/// A conversation set aside: the tokens it covers, its cache, and the recurrent checkpoints that
/// went with it (so that it can go back to a message boundary as it could when it was live).
struct Entry {
    keys: Vec<u64>,
    state: ParkedState,
    checkpoints: Vec<Checkpoint>,
    bytes: usize,
    /// When it was set aside (a counter): the least recently used goes first.
    used: u64,
}

impl Entry {
    fn new(keys: Vec<u64>, state: ParkedState, checkpoints: Vec<Checkpoint>) -> Self {
        let bytes = state.bytes() + 8 * keys.len() + checkpoints.iter().map(|(k, s, _)| s.bytes() + 8 * k.len()).sum::<usize>();
        Self { keys, state, checkpoints, bytes, used: 0 }
    }
}

/// What [`Stash::insert`] did.
#[derive(Debug, PartialEq, Eq)]
enum Stashed {
    Kept { evicted: usize },
    /// Bigger than the whole budget (or the stash is off): nothing was kept.
    Refused,
}

/// The states set aside, within a byte budget. The one set aside longest ago goes first.
#[derive(Default)]
struct Stash {
    entries: Vec<Entry>,
    budget: usize,
    held: usize,
    clock: u64,
}

impl Stash {
    fn insert(&mut self, mut entry: Entry) -> Stashed {
        if self.budget == 0 || entry.bytes > self.budget {
            return Stashed::Refused;
        }
        // The same tokens again: the newer copy replaces the older.
        let mut freed = 0;
        self.entries.retain(|e| if e.keys == entry.keys { freed += e.bytes; false } else { true });
        self.held -= freed;
        let mut evicted = 0;
        while self.held + entry.bytes > self.budget {
            let Some(oldest) = (0..self.entries.len()).min_by_key(|&i| self.entries[i].used) else { break };
            self.take(oldest);
            evicted += 1;
        }
        self.clock += 1;
        entry.used = self.clock;
        self.held += entry.bytes;
        self.entries.push(entry);
        Stashed::Kept { evicted }
    }

    /// Take entry `index` out (its bytes no longer count).
    fn take(&mut self, index: usize) -> Entry {
        let entry = self.entries.remove(index);
        self.held -= entry.bytes;
        entry
    }

    fn views(&self) -> Vec<View<'_>> {
        self.entries.iter().map(|e| View { keys: &e.keys, checkpoints: e.checkpoints.iter().map(|(k, _, _)| k.as_slice()).collect(), used: e.used }).collect()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.held = 0;
    }

    /// The bytes held are those of the entries, and within the budget.
    #[cfg(test)]
    fn consistent(&self) -> bool {
        self.held == self.entries.iter().map(|e| e.bytes).sum::<usize>() && self.held <= self.budget
    }
}

/// The engine's live state, lent for a swap: its cache, the tokens it covers and its recurrent
/// checkpoints. `private`: it holds an incognito session's prompt, which is never set aside.
pub(crate) struct Held<'a> {
    pub kv: &'a mut KvCache,
    pub covered: &'a mut Vec<u64>,
    pub checkpoints: &'a mut Vec<Checkpoint>,
    pub private: bool,
}

/// Setting conversations aside and bringing them back. Off (a budget of 0) it does nothing: that
/// is how the Qwen3.5 hybrid, which has its disk states, keeps its behaviour as it was.
pub(crate) struct Parking {
    stash: Stash,
    log: bool,
}

impl Parking {
    pub(crate) fn new(budget: usize, log: bool) -> Self {
        Self { stash: Stash { budget, ..Stash::default() }, log }
    }

    pub(crate) fn set_budget(&mut self, bytes: usize) {
        self.stash.budget = bytes;
        self.stash.clear();
    }

    pub(crate) fn enabled(&self) -> bool {
        self.stash.budget > 0
    }

    /// Forget every state set aside (an incognito request or session ends: the engine holds
    /// nothing afterwards, as it never did).
    pub(crate) fn clear(&mut self) {
        self.stash.clear();
    }

    /// Copy the live state to the host and stash it, when it is worth keeping: it is big enough,
    /// it is not a private session's, and its cache and tokens agree. The cache is as it was.
    /// `covered` and `checkpoints` have gone into the stash (and are empty), so that nothing
    /// claims a cache the caller is about to rewrite; or, with `copy`, the stash has copies and
    /// they stay (the prompt that displaces the state reads on from one of its checkpoints).
    fn park(&mut self, held: &mut Held<'_>, copy: bool) -> bool {
        let tokens = held.covered.len();
        if held.private || tokens < MIN_TOKENS || held.kv.len != tokens {
            return false;
        }
        match ParkedState::estimate(held.kv) {
            Ok(bytes) if bytes <= self.stash.budget => {}
            Ok(bytes) => {
                if self.log { eprintln!("  Qwen park skipped: {tokens} tokens ({:.0} MB) do not fit the {:.0} MB budget", bytes as f64 / 1e6, self.stash.budget as f64 / 1e6); }
                return false;
            }
            Err(e) => {
                if self.log { eprintln!("  Qwen park skipped: {e}"); }
                return false;
            }
        }
        let clock = Instant::now();
        let state = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ParkedState::capture(held.kv))) {
            Ok(Ok(state)) => state,
            Ok(Err(e)) => {
                if self.log { eprintln!("  Qwen park skipped: {e}"); }
                return false;
            }
            Err(_) => {
                if self.log { eprintln!("  Qwen park skipped: copying the cache failed"); }
                return false;
            }
        };
        let entry = if copy {
            Entry::new(held.covered.clone(), state, held.checkpoints.clone())
        } else {
            Entry::new(std::mem::take(held.covered), state, std::mem::take(held.checkpoints))
        };
        let bytes = entry.bytes;
        let outcome = self.stash.insert(entry);
        if self.log {
            let secs = clock.elapsed().as_secs_f64();
            match outcome {
                Stashed::Kept { evicted } => eprintln!("  Qwen park: stashed {tokens} tokens ({:.0} MB) in {secs:.3}s; {} states, {:.0} MB held{}",
                    bytes as f64 / 1e6, self.stash.entries.len(), self.stash.held as f64 / 1e6,
                    if evicted > 0 { format!("; {evicted} older dropped") } else { String::new() }),
                Stashed::Refused => eprintln!("  Qwen park skipped: {tokens} tokens ({:.0} MB) do not fit the budget", bytes as f64 / 1e6),
            }
        }
        true
    }

    /// Before the cache logic: when a stashed state is what `keys` continues, and clearly better
    /// than the live one, set the live one aside and bring the stashed one back. `true`: the cache
    /// now holds the stashed state, with its tokens and checkpoints, and the engine goes on from
    /// there as it always does. `false`: the engine goes on as it did before this existed (the
    /// live state, or none, and the prompt read again); at worst a state that was set aside is
    /// lost.
    pub(crate) fn swap_in(&mut self, held: Held<'_>, keys: &[u64]) -> bool {
        if !self.enabled() || self.stash.entries.is_empty() {
            return false;
        }
        let Held { kv, covered, checkpoints, private } = held;
        let live = reusable(covered.as_slice(), checkpoints.iter().map(|(k, _, _)| k.as_slice()), keys);
        let Some((index, _)) = choose(&self.stash.views(), live, keys) else { return false };
        if let Err(e) = self.stash.entries[index].state.validate(kv) {
            // It can never fit this cache: no use keeping it.
            let gone = self.stash.take(index);
            if self.log { eprintln!("  Qwen restore skipped: {e}; dropped the {} tokens it held", gone.keys.len()); }
            return false;
        }
        // Out of the stash first: the room it held is the live state's.
        let Entry { keys: entry_keys, state, checkpoints: entry_checkpoints, .. } = self.stash.take(index);
        self.park(&mut Held { kv: &mut *kv, covered: &mut *covered, checkpoints: &mut *checkpoints, private }, false);
        // While the cache is rewritten, nothing claims what it holds.
        covered.clear();
        checkpoints.clear();
        let (tokens, bytes) = (entry_keys.len(), state.bytes());
        let clock = Instant::now();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.restore(&mut *kv))) {
            Ok(Ok(())) => {
                *covered = entry_keys;
                *checkpoints = entry_checkpoints;
                if self.log { eprintln!("  Qwen restore: {tokens} tokens ({:.0} MB) in {:.3}s", bytes as f64 / 1e6, clock.elapsed().as_secs_f64()); }
                true
            }
            failed => {
                kv.reset();
                if self.log {
                    let why = match failed { Ok(Err(e)) => e, _ => "the device refused".into() };
                    eprintln!("  Qwen restore failed ({why}): reading the prompt again");
                }
                false
            }
        }
    }

    /// The engine is about to read `keys` over its live state, which they share less than half
    /// of: it is thrown away, or rolled back to an early checkpoint and written over. Set it
    /// aside first, so that the prompt that comes back for it finds it. This is the last moment
    /// the state's checkpoints are whole: the engine drops the ones `keys` do not share right
    /// after, so it is asked before that, and not from the place where the state is reset.
    ///
    /// When `keys` read on from one of its checkpoints (the system prompt they share ends at a
    /// message boundary) the engine still needs them, and the stash gets copies.
    pub(crate) fn park_displaced(&mut self, mut held: Held<'_>, keys: &[u64]) {
        if !self.enabled() || !worth_parking(held.covered.len(), common_prefix(held.covered, keys)) {
            return;
        }
        let reads_on = reusable(held.covered.as_slice(), held.checkpoints.iter().map(|(k, _, _)| k.as_slice()), keys) > 0;
        self.park(&mut held, reads_on);
    }

    /// The states set aside and the bytes they hold.
    #[cfg(test)]
    pub(crate) fn held(&self) -> (usize, usize) {
        (self.stash.entries.len(), self.stash.held)
    }

    #[cfg(test)]
    pub(crate) fn keys_held(&self) -> Vec<Vec<u64>> {
        self.stash.entries.iter().map(|e| e.keys.clone()).collect()
    }
}

/// Host RAM the engine may set conversations aside in, from `--park-gb`: never more than half
/// of what is free (0 when the machine will not say and nothing is asked).
pub(crate) fn budget(gb: f64, free: Option<u64>) -> usize {
    if !gb.is_finite() || gb <= 0.0 {
        return 0;
    }
    let asked = (gb * 1e9) as u64;
    free.map_or(asked, |free| asked.min(free / 2)) as usize
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use ggml_rs::{Backend, CpuBackend};
    use std::cell::Cell;
    use std::sync::Arc;

    thread_local! {
        /// Makes `ParkedState::restore` panic once the cache is reset (this thread only).
        pub(crate) static INJECT_PANIC: Cell<bool> = const { Cell::new(false) };
    }

    /// A deterministic float stream with awkward values in it (signed zeros, subnormals, big and
    /// tiny magnitudes): a copy that rounds or flushes anything shows.
    pub(crate) struct Floats(pub u64);
    impl Floats {
        pub(crate) fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            match (self.0 >> 60) % 8 {
                0 => -0.0,
                1 => f32::from_bits(1),
                2 => 3.0e38,
                3 => -1.0e-30,
                _ => f32::from_bits((self.0 >> 20) as u32 & 0x7f7f_ffff | (self.0 as u32 & 0x8000_0000)),
            }
        }
        pub(crate) fn tensor(&mut self, shape: Vec<usize>) -> Tensor {
            let n = shape.iter().product();
            Tensor::from_vec((0..n).map(|_| self.next()).collect(), shape)
        }
    }

    pub(crate) fn bits(t: &Tensor) -> Vec<u32> {
        t.data().iter().map(|v| v.to_bits()).collect()
    }

    /// The shape of Qwen3.8-Flash-Next's cache, small: an attention layer (2 heads of 4), a
    /// recurrent layer (a slot of 1 it never uses, a state and a conv window), another attention
    /// layer, the n-gram slot (a window and the last ids) and an indexer slot (1 head of 3), the
    /// slots on two "devices".
    pub(crate) fn kv_like_flash(max_len: usize) -> KvCache {
        let (a, b): (Arc<dyn Backend>, Arc<dyn Backend>) = (Arc::new(CpuBackend::new()), Arc::new(CpuBackend::new()));
        KvCache::new_lazy_per_layer_kv(vec![a.clone(), a, b.clone(), b.clone(), b], max_len, &[2, 1, 2, 1, 1], &[4, 1, 4, 1, 3])
    }

    /// The slots that take rows: the attention layers and the indexer.
    pub(crate) const ROWS: [(usize, usize, usize); 3] = [(0, 2, 4), (2, 2, 4), (4, 1, 3)];

    /// `n` tokens appended to the slots that take rows, and the recurrent tensors set.
    pub(crate) fn fill(kv: &mut KvCache, f: &mut Floats, n: usize) {
        for (slot, heads, dim) in ROWS {
            let backend = kv.layer_backends[slot].clone();
            let (k, v) = (f.tensor(vec![n, heads, dim]), f.tensor(vec![n, heads, dim]));
            kv.append(backend.as_ref(), slot, &k, &v);
        }
        kv.commit(n);
        kv.ssm_state[1] = Some(f.tensor(vec![2, 3, 3]));
        kv.ssm_conv[1] = Some(f.tensor(vec![3, 7]));
        kv.ssm_state[3] = Some(f.tensor(vec![2]));
        kv.ssm_conv[3] = Some(f.tensor(vec![4, 5]));
    }

    /// Everything a cache means: its length, the rows below it of the slots that take rows, and
    /// every recurrent tensor, as bits.
    pub(crate) fn dump(kv: &KvCache) -> Vec<Option<Vec<u32>>> {
        let mut out = vec![Some(vec![kv.len as u32])];
        for (slot, heads, dim) in ROWS {
            let n = kv.len * heads * dim;
            out.push(Some(bits(&kv.k[slot])[..n].to_vec()));
            out.push(Some(bits(&kv.v[slot])[..n].to_vec()));
        }
        for i in 0..kv.k.len() {
            out.push(kv.ssm_state[i].as_ref().map(bits));
            out.push(kv.ssm_conv[i].as_ref().map(bits));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::fixtures::*;
    use ggml_rs::{Backend, CpuBackend};
    use std::sync::Arc;

    /// A stash entry of `tokens` tokens that are all `id`.
    fn entry(tokens: usize, id: u64) -> Entry {
        let mut kv = kv_like_flash(4096);
        fill(&mut kv, &mut Floats(id), tokens);
        Entry::new(vec![id; tokens], ParkedState::capture(&kv).unwrap(), Vec::new())
    }

    #[test]
    fn a_lazily_grown_multi_slot_cache_round_trips_bit_for_bit() {
        let mut f = Floats(7);
        let mut kv = kv_like_flash(4096);
        fill(&mut kv, &mut f, 200);
        fill(&mut kv, &mut f, 700);
        assert_eq!(kv.k[0].dim(0), 1024, "grown past its 256 rows");
        assert_eq!(kv.k[1].dim(0), 256, "the recurrent slot never grows");
        let want = dump(&kv);
        let parked = ParkedState::capture(&kv).unwrap();
        // The attention slots' 900 x 2 x 4 and the indexer's 900 x 1 x 3, K and V, and the recurrent tensors.
        assert_eq!(parked.bytes(), 4 * (2 * 2 * 900 * 8 + 2 * 900 * 3 + 18 + 21 + 2 + 20));
        assert_eq!(ParkedState::estimate(&kv).unwrap(), parked.bytes());
        assert!(parked.slots[1].rows.is_none() && parked.slots[3].rows.is_none(), "a recurrent slot has no rows once the sequence outgrew its buffer");
        assert_eq!(dump(&kv), want, "capturing reads only");

        // A cache that moved on to other content: more tokens, other recurrent tensors, a reset.
        let mut moved = kv_like_flash(4096);
        fill(&mut moved, &mut f, 900);
        fill(&mut moved, &mut f, 300);
        moved.reset();
        fill(&mut moved, &mut f, 50);
        assert_ne!(dump(&moved), want);
        ParkedState::capture(&kv).unwrap().restore(&mut moved).unwrap();
        assert_eq!(dump(&moved), want);

        // A fresh cache, whose buffers grow to hold it, and one already grown larger.
        let mut fresh = kv_like_flash(4096);
        parked.restore(&mut fresh).unwrap();
        assert_eq!(dump(&fresh), want);
        assert_eq!(fresh.k[0].dim(0), 1024);
        let mut larger = kv_like_flash(4096);
        fill(&mut larger, &mut Floats(5), 1500);
        ParkedState::capture(&kv).unwrap().restore(&mut larger).unwrap();
        assert_eq!(dump(&larger), want);
        assert_eq!(larger.k[0].dim(0), 2048, "buffers do not shrink");

        // It goes on as the original does: the same token appended to both reads back the same.
        let (k, v) = (f.tensor(vec![1, 2, 4]), f.tensor(vec![1, 2, 4]));
        for cache in [&mut kv, &mut larger] {
            let backend = cache.layer_backends[0].clone();
            cache.append(backend.as_ref(), 0, &k, &v);
            cache.commit(1);
        }
        assert_eq!(bits(&kv.k[0])[..901 * 8], bits(&larger.k[0])[..901 * 8]);
    }

    #[test]
    fn a_short_sequence_keeps_the_recurrent_slots_rows_too() {
        let mut kv = kv_like_flash(1024);
        fill(&mut kv, &mut Floats(3), 40);
        let parked = ParkedState::capture(&kv).unwrap();
        assert!(parked.slots[1].rows.is_some(), "the recurrent slot's 256-row buffer still covers 40 tokens");
        let mut other = kv_like_flash(1024);
        parked.restore(&mut other).unwrap();
        assert_eq!(dump(&other), dump(&kv));
    }

    #[test]
    fn a_cache_it_cannot_describe_is_not_parked() {
        let backend = CpuBackend::new();
        let eager = KvCache::new(&backend, 2, 16, 1, 2);
        assert!(ParkedState::capture(&eager).is_err() && ParkedState::estimate(&eager).is_err());
        assert!(ParkedState::capture(&kv_like_flash(64)).is_err(), "nothing in it");
    }

    #[test]
    fn a_state_that_does_not_fit_is_refused_before_the_cache_is_touched() {
        let make = || {
            let mut kv = kv_like_flash(2048);
            fill(&mut kv, &mut Floats(11), 300);
            ParkedState::capture(&kv).unwrap()
        };
        let one: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let lazy = |slots: usize, max_len: usize, heads: &[usize], dims: &[usize]| KvCache::new_lazy_per_layer_kv(vec![one.clone(); slots], max_len, heads, dims);
        let mut victims = vec![
            ("fewer slots", lazy(4, 2048, &[2, 1, 2, 1], &[4, 1, 4, 1])),
            ("other heads", lazy(5, 2048, &[2, 1, 3, 1, 1], &[4, 1, 4, 1, 3])),
            ("other width", lazy(5, 2048, &[2, 1, 2, 1, 1], &[4, 1, 5, 1, 3])),
            ("other context", lazy(5, 1024, &[2, 1, 2, 1, 1], &[4, 1, 4, 1, 3])),
            ("other recurrent shape", lazy(5, 2048, &[2, 1, 2, 1, 1], &[4, 1, 4, 1, 3])),
        ];
        victims[4].1.ssm_state[1] = Some(Tensor::from_vec(vec![1.0; 9], vec![9]));
        for (name, kv) in &mut victims {
            kv.len = 2;
            kv.k[0].data_mut()[..4].fill(8.0);
            let before = (kv.len, (0..kv.k.len()).map(|i| (bits(&kv.k[i]), kv.ssm_state[i].as_ref().map(bits))).collect::<Vec<_>>());
            assert!(make().validate(kv).is_err(), "{name}");
            assert!(make().restore(kv).is_err(), "{name}");
            let after = (kv.len, (0..kv.k.len()).map(|i| (bits(&kv.k[i]), kv.ssm_state[i].as_ref().map(bits))).collect::<Vec<_>>());
            assert!(before == after, "{name}: the live cache was touched");
        }
        // The same layout is accepted.
        let mut same = kv_like_flash(2048);
        assert!(make().validate(&same).is_ok());
        make().restore(&mut same).unwrap();
        assert_eq!(same.len, 300);
    }

    #[test]
    fn the_stash_drops_the_least_recently_set_aside_and_stays_within_its_budget() {
        let size = entry(100, 1).bytes;
        let mut stash = Stash { budget: size * 5 / 2, ..Stash::default() };
        for id in 1..=2 {
            assert_eq!(stash.insert(entry(100, id)), Stashed::Kept { evicted: 0 });
        }
        assert_eq!(stash.insert(entry(100, 3)), Stashed::Kept { evicted: 1 });
        let ids = |s: &Stash| s.entries.iter().map(|e| e.keys[0]).collect::<Vec<_>>();
        assert_eq!(ids(&stash), [2, 3]);
        assert!(stash.consistent());
        // The same tokens again replace the old copy, which is then the newest.
        assert_eq!(stash.insert(entry(100, 2)), Stashed::Kept { evicted: 0 });
        assert_eq!(ids(&stash), [3, 2]);
        assert!(stash.consistent());
        // A bigger one pushes out as many as it needs.
        assert_eq!(stash.insert(entry(220, 4)), Stashed::Kept { evicted: 2 });
        assert_eq!(ids(&stash), [4]);
        // More than the whole budget is refused and changes nothing.
        let before = stash.held;
        assert_eq!(stash.insert(entry(300, 5)), Stashed::Refused);
        assert_eq!((ids(&stash), stash.held), (vec![4], before));
        // Taking an entry out gives its bytes back.
        let taken = stash.take(0);
        assert_eq!((taken.keys[0], stash.held), (4, 0));
        stash.insert(entry(100, 6));
        stash.clear();
        assert!(stash.entries.is_empty() && stash.held == 0);
        // Off: nothing is kept.
        assert_eq!(Stash::default().insert(entry(10, 7)), Stashed::Refused);
    }

    fn view<'a>(keys: &'a [u64], checkpoints: Vec<&'a [u64]>, used: u64) -> View<'a> {
        View { keys, checkpoints, used }
    }

    #[test]
    fn the_stashed_state_with_the_longest_shared_prefix_comes_back() {
        let lineage = |id: u64, n: usize| -> Vec<u64> { (0..n).map(|i| if i < 520 { i as u64 } else { id * 1_000_000 + i as u64 }).collect() };
        let (a, b, c) = (lineage(1, 21_000), lineage(2, 5_000), lineage(3, 3_000));
        let mut prompt = a.clone();
        prompt.extend(10_000_000..10_000_300);
        let views = [view(&b, vec![], 5), view(&a, vec![], 3), view(&c, vec![], 9)];
        assert_eq!(choose(&views, 0, &prompt), Some((1, 21_000)));
        // The preamble alone (520 tokens) is not worth a swap.
        let other = lineage(4, 6_000);
        assert_eq!(choose(&views, 0, &other), None);
        // A prompt that diverges from a state before it ends starts over, unless a checkpoint lies before the fork.
        let mut forked = a[..15_000].to_vec();
        forked.extend(20_000_000..20_000_100);
        assert_eq!(choose(&[view(&a, vec![], 3)], 0, &forked), None);
        assert_eq!(choose(&[view(&a, vec![&a[..520], &a[..12_000]], 3)], 0, &forked), Some((0, 12_000)));
        assert_eq!(choose(&[view(&a, vec![&a[..520], &a[..12_000], &a[..16_000]], 3)], 0, &forked), Some((0, 12_000)), "a checkpoint past the fork does not count");
    }

    #[test]
    fn a_swap_needs_a_state_that_saves_something_the_live_one_does_not() {
        let a: Vec<u64> = (0..6_000).collect();
        let mut prompt = a.clone();
        prompt.push(99_999);
        let views = [view(&a, vec![], 1)];
        // At least MIN_TOKENS...
        assert_eq!(choose(&[view(&a[..1_023], vec![], 1)], 0, &prompt), None);
        assert_eq!(choose(&[view(&a[..1_024], vec![], 1)], 0, &prompt), Some((0, 1_024)));
        // ...and MIN_GAIN more than the live state.
        assert_eq!(choose(&views, 0, &prompt), Some((0, 6_000)));
        assert_eq!(choose(&views, 4_976, &prompt), Some((0, 6_000)));
        assert_eq!(choose(&views, 4_977, &prompt), None);
        assert_eq!(choose(&views, 6_000, &prompt), None);
        // Of two that save the same, the one set aside last.
        let twin = a.clone();
        assert_eq!(choose(&[view(&a, vec![], 1), view(&twin, vec![], 2), view(&a, vec![], 0)], 0, &prompt), Some((1, 6_000)));
        // A retry of exactly what a state holds leaves a token to run only through a checkpoint.
        assert_eq!(choose(&views, 0, &a), None);
        assert_eq!(choose(&[view(&a, vec![&a[..5_999]], 1)], 0, &a), Some((0, 5_999)));
        // Nothing stashed.
        assert_eq!(choose(&[], 0, &prompt), None);
    }

    /// The old cache logic of `QwenEngine::generate`, as it was written there.
    fn what_generate_starts_from(covered: &[u64], checkpoints: &[Vec<u64>], keys: &[u64]) -> usize {
        let common = covered.iter().zip(keys).take_while(|(a, b)| a == b).count();
        let mut checkpoints: Vec<_> = checkpoints.to_vec();
        checkpoints.retain(|saved| saved.len() <= common && keys.starts_with(saved));
        let mut start = if common == covered.len() && common < keys.len() { common } else { 0 };
        if let Some(saved) = checkpoints.iter().filter(|saved| saved.len() > start && saved.len() < keys.len() && keys.starts_with(saved)).max_by_key(|saved| saved.len()) {
            start = saved.len();
        }
        start
    }

    #[test]
    fn what_a_state_would_not_read_again_is_what_generate_starts_from() {
        let mut seed = 12345u64;
        let mut roll = move |n: u64| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) % n };
        for _ in 0..30_000 {
            let covered: Vec<u64> = (0..roll(14)).map(|_| roll(3)).collect();
            // The prompt: the state's tokens cut short or whole, then something else (or nothing).
            let keep = roll(covered.len() as u64 + 1) as usize;
            let mut keys = covered[..keep.min(covered.len())].to_vec();
            keys.extend((0..roll(8)).map(|_| roll(3)));
            // The engine's checkpoints are prefixes of what it covers; a few others too, which it would drop.
            let mut checkpoints: Vec<Vec<u64>> = (0..roll(4)).map(|_| covered[..roll(covered.len() as u64 + 1) as usize].to_vec()).collect();
            if roll(4) == 0 { checkpoints.push((0..roll(6)).map(|_| roll(3)).collect()); }
            let want = what_generate_starts_from(&covered, &checkpoints, &keys);
            let got = reusable(&covered, checkpoints.iter().map(Vec::as_slice), &keys);
            assert_eq!(got, want, "covered {covered:?} checkpoints {checkpoints:?} keys {keys:?}");
        }
    }

    #[test]
    fn a_state_is_worth_parking_when_big_and_mostly_unrelated() {
        assert!(worth_parking(21_000, 520));
        assert!(worth_parking(1_024, 0));
        assert!(!worth_parking(1_023, 0), "small");
        assert!(!worth_parking(21_000, 10_500), "half of it shared: the same conversation branching");
        assert!(worth_parking(21_000, 10_499));
        assert!(!worth_parking(0, 0));
    }

    #[test]
    fn the_budget_follows_the_flag_and_never_asks_more_than_half_of_what_is_free() {
        let gb = 1_000_000_000u64;
        assert_eq!(budget(8.0, Some(150 * gb)), 8 * gb as usize);
        assert_eq!(budget(8.0, Some(10 * gb)), 5 * gb as usize);
        assert_eq!(budget(8.0, None), 8 * gb as usize);
        assert_eq!(budget(0.0, Some(150 * gb)), 0);
        assert_eq!(budget(-1.0, None), 0);
        assert_eq!(budget(f64::NAN, None), 0);
        assert_eq!(budget(0.5, Some(0)), 0);
    }

    /// The live state of an engine, in small: `n` tokens of lineage `id`, set as the engine would hold them.
    struct Rig {
        kv: KvCache,
        covered: Vec<u64>,
        checkpoints: Vec<Checkpoint>,
        parking: Parking,
        private: bool,
    }

    impl Rig {
        /// A budget of `thousands` thousand tokens' worth of states (0: off).
        fn new(thousands: usize) -> Self {
            let thousand = entry(1_000, 0).bytes;
            Self { kv: kv_like_flash(8192), covered: Vec::new(), checkpoints: Vec::new(), parking: Parking::new(thousand * thousands, false), private: false }
        }
        /// The engine holds `n` tokens of lineage `id` (their keys are `id * 10^6 + position`).
        fn hold(&mut self, id: u64, n: usize) {
            self.kv.reset();
            fill(&mut self.kv, &mut Floats(id), n);
            self.covered = (0..n as u64).map(|i| id * 1_000_000 + i).collect();
            self.checkpoints = vec![(self.covered[..n / 2].to_vec(), RecurrentSnapshot::capture(&self.kv), true)];
        }
        fn swap(&mut self, keys: &[u64]) -> bool {
            self.parking.swap_in(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: self.private }, keys)
        }
        fn discard(&mut self, keys: &[u64]) {
            self.parking.park_displaced(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: self.private }, keys)
        }
    }

    fn lineage(id: u64, n: usize) -> Vec<u64> {
        (0..n as u64).map(|i| id * 1_000_000 + i).collect()
    }

    #[test]
    fn a_swap_sets_the_live_state_aside_and_brings_the_stashed_one_back_whole() {
        let mut rig = Rig::new(20);
        rig.hold(1, 3_000);
        let first = (dump(&rig.kv), rig.covered.clone(), rig.checkpoints.iter().map(|(k, _, b)| (k.clone(), *b)).collect::<Vec<_>>());
        // Another conversation displaces it...
        let other = lineage(2, 1_500);
        rig.discard(&other);
        assert_eq!(rig.parking.held().0, 1);
        assert!(rig.covered.is_empty() && rig.checkpoints.is_empty(), "what was set aside no longer claims the cache");
        rig.hold(2, 1_500);
        let second = dump(&rig.kv);
        // ...and a prompt that continues the first brings it back, setting the second aside.
        let mut prompt = lineage(1, 3_000);
        prompt.extend(7_000_000..7_000_040);
        assert!(rig.swap(&prompt));
        assert_eq!((dump(&rig.kv), rig.covered.clone(), rig.checkpoints.iter().map(|(k, _, b)| (k.clone(), *b)).collect::<Vec<_>>()), first);
        assert_eq!(rig.parking.keys_held(), [lineage(2, 1_500)]);
        assert!(rig.parking.stash.consistent());
        // The second comes back from there the same way, bit for bit.
        let mut prompt = lineage(2, 1_500);
        prompt.push(8_000_000);
        assert!(rig.swap(&prompt));
        assert_eq!(dump(&rig.kv), second);
        assert_eq!(rig.parking.keys_held(), [lineage(1, 3_000)]);
    }

    #[test]
    fn a_swap_leaves_a_live_state_the_prompt_continues_alone() {
        let mut rig = Rig::new(20);
        rig.hold(1, 3_000);
        rig.discard(&lineage(2, 10));
        rig.hold(2, 2_000);
        let live = dump(&rig.kv);
        // Continues the live state: nothing to fetch, nothing parked.
        let mut prompt = lineage(2, 2_000);
        prompt.push(1);
        assert!(!rig.swap(&prompt));
        assert_eq!(dump(&rig.kv), live);
        assert_eq!(rig.parking.held().0, 1);
        // Continues the stashed one, but only for 900 tokens: too few.
        let mut short = lineage(1, 900);
        short.push(5);
        assert!(!rig.swap(&short));
        assert_eq!(dump(&rig.kv), live);
        assert_eq!(rig.covered, lineage(2, 2_000));
        // A prompt that continues neither.
        assert!(!rig.swap(&lineage(3, 5_000)));
        assert_eq!(rig.parking.held().0, 1);
    }

    #[test]
    fn private_states_are_never_set_aside() {
        let mut rig = Rig::new(20);
        rig.hold(1, 3_000);
        rig.private = true;
        rig.discard(&lineage(2, 100));
        assert_eq!(rig.parking.held().0, 0, "a private session's state is not parked when discarded");
        assert_eq!(rig.covered.len(), 3_000, "and nothing else changes");
        // Nor when a stashed state displaces it.
        rig.private = false;
        rig.hold(2, 2_000);
        rig.discard(&lineage(3, 5));
        rig.hold(3, 1_500);
        rig.private = true;
        let mut prompt = lineage(2, 2_000);
        prompt.push(1);
        assert!(rig.swap(&prompt));
        assert_eq!(rig.covered, lineage(2, 2_000));
        assert_eq!(rig.parking.held().0, 0, "the private state went nowhere");
        assert!(rig.parking.keys_held().is_empty());
    }

    #[test]
    fn small_and_related_states_are_not_set_aside() {
        let mut rig = Rig::new(20);
        rig.hold(1, 1_000);
        rig.discard(&lineage(2, 100));
        assert_eq!(rig.parking.held().0, 0, "under MIN_TOKENS");
        rig.hold(1, 4_000);
        let mut branch = lineage(1, 2_500);
        branch.push(9);
        rig.discard(&branch);
        assert_eq!(rig.parking.held().0, 0, "a branch of the same conversation: the checkpoints hold it");
        let mut far = lineage(1, 1_500);
        far.push(9);
        rig.discard(&far);
        assert_eq!(rig.parking.held().0, 1);
    }

    #[test]
    fn a_prompt_that_reads_on_from_a_checkpoint_of_the_live_state_gets_a_copy_set_aside() {
        let mut rig = Rig::new(20);
        rig.hold(1, 4_000);
        // The state also has a checkpoint at 800 (the end of a system prompt both conversations share).
        rig.checkpoints.insert(0, (rig.covered[..800].to_vec(), RecurrentSnapshot::capture(&rig.kv), true));
        let live = (dump(&rig.kv), rig.covered.clone(), rig.checkpoints.iter().map(|(k, _, _)| k.len()).collect::<Vec<_>>());
        let mut prompt = lineage(1, 1_500);
        prompt.push(5);
        rig.discard(&prompt);
        // The engine rolls back to the checkpoint at 800 and still needs what it holds: nothing moved...
        assert_eq!((dump(&rig.kv), rig.covered.clone(), rig.checkpoints.iter().map(|(k, _, _)| k.len()).collect::<Vec<_>>()), live);
        // ...and the stash has the state whole, checkpoints and all.
        assert_eq!(rig.parking.keys_held(), [lineage(1, 4_000)]);
        assert_eq!(rig.parking.stash.entries[0].checkpoints.iter().map(|(k, _, _)| k.len()).collect::<Vec<_>>(), [800, 2_000]);
        assert!(rig.parking.stash.consistent());
        // With nothing of the prompt to read on from, the state itself goes into the stash.
        let mut rig = Rig::new(20);
        rig.hold(1, 4_000);
        let mut prompt = lineage(1, 700);
        prompt.push(5);
        rig.discard(&prompt);
        assert!(rig.covered.is_empty() && rig.checkpoints.is_empty());
        assert_eq!(rig.parking.stash.entries[0].checkpoints.iter().map(|(k, _, _)| k.len()).collect::<Vec<_>>(), [2_000]);
    }

    #[test]
    fn a_state_that_does_not_fit_the_budget_is_not_copied() {
        let mut rig = Rig::new(1);
        rig.parking.set_budget(1_000);
        rig.hold(1, 3_000);
        rig.discard(&lineage(2, 10));
        assert_eq!(rig.parking.held(), (0, 0));
    }

    #[test]
    fn a_failed_restore_falls_back_to_reading_the_prompt_and_leaves_the_stash_sound() {
        let mut rig = Rig::new(20);
        rig.hold(1, 3_000);
        rig.discard(&lineage(2, 10));
        rig.hold(2, 2_000);
        let mut prompt = lineage(1, 3_000);
        prompt.push(1);
        fixtures::INJECT_PANIC.with(|f| f.set(true));
        let swapped = rig.swap(&prompt);
        fixtures::INJECT_PANIC.with(|f| f.set(false));
        assert!(!swapped);
        // A cache that claims nothing and holds nothing; the live state is in the stash, the failed one gone.
        assert!(rig.covered.is_empty() && rig.checkpoints.is_empty());
        assert_eq!(rig.kv.len, 0);
        assert!(rig.kv.ssm_state.iter().chain(&rig.kv.ssm_conv).all(Option::is_none));
        assert_eq!(rig.parking.keys_held(), [lineage(2, 2_000)]);
        assert!(rig.parking.stash.consistent());
        // And it still works: the live state comes back for a prompt that continues it.
        let mut again = lineage(2, 2_000);
        again.push(1);
        assert!(rig.swap(&again));
        assert_eq!(rig.covered, lineage(2, 2_000));
        assert!(rig.parking.stash.consistent());
    }

    #[test]
    fn a_stashed_state_that_cannot_fit_the_cache_is_dropped_and_the_engine_goes_on() {
        let mut rig = Rig::new(20);
        rig.hold(1, 3_000);
        rig.discard(&lineage(2, 10));
        // The engine's cache is rebuilt for another context length: the stashed state cannot fit it.
        rig.kv = kv_like_flash(4096);
        rig.covered = lineage(2, 10);
        let mut prompt = lineage(1, 3_000);
        prompt.push(1);
        assert!(!rig.swap(&prompt));
        assert_eq!(rig.parking.held(), (0, 0));
        assert_eq!(rig.covered, lineage(2, 10), "the live state was not touched");
    }

    #[test]
    fn off_it_does_nothing_and_clearing_forgets_everything() {
        let mut rig = Rig::new(0);
        rig.hold(1, 3_000);
        let before = dump(&rig.kv);
        rig.discard(&lineage(2, 10));
        assert!(!rig.swap(&lineage(1, 3_000)));
        assert_eq!((dump(&rig.kv), rig.covered.len(), rig.parking.held()), (before, 3_000, (0, 0)));
        assert!(!rig.parking.enabled());

        let mut rig = Rig::new(20);
        rig.hold(1, 3_000);
        rig.discard(&lineage(2, 10));
        assert_eq!(rig.parking.held().0, 1);
        rig.parking.clear();
        assert_eq!(rig.parking.held(), (0, 0));
        let mut prompt = lineage(1, 3_000);
        prompt.push(1);
        rig.hold(3, 1_500);
        assert!(!rig.swap(&prompt), "what was cleared does not come back");
        // Changing the budget starts over as well.
        rig.parking.set_budget(0);
        assert!(!rig.parking.enabled());
    }

}
