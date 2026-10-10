//! Prompt states kept on disk, so that a new process starts where an
//! earlier one left off: the system prompt a harness sends every time, and
//! the conversations it may resume.
//!
//! An entry is a [`Snapshot`] of the model's state after a prompt prefix,
//! with that prefix's keys. Image prompts are kept only by engines whose
//! namespace also identifies the vision weights and preprocessing. The engine saves one where a conversation's first user
//! message begins (the system prompt and tools), and for the conversation
//! after each layered pass, where its newest user message begins and at the
//! end of each prompt; earlier prefixes remain reusable when a rendered turn changes its suffix. When a
//! prompt comes that the model's own state and checkpoints cover less of
//! than an entry does, the entry is loaded instead.
//!
//! Files are written by a thread of their own (a temporary name, then
//! renamed), and the least recently used go when the entries pass the size
//! budget. Each file carries the model's fingerprint: entries of another
//! model (or another format) in the same directory are left alone.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, atomic::{AtomicU8, Ordering}};
use std::time::SystemTime;

/// VENDORED-LOCAL: GLM-5.3-Flash. What a state has to do to live on disk.
///
/// This file was written around `dsv41_cuda::Snapshot`. GLM-5.3-Flash keeps a
/// different state -- a delta-rule recurrence and a latent cache rather than window
/// rings and compressed rows -- but it wants exactly this: the prefix a harness
/// sends every time, kept so a later process does not read it again. So the cache is
/// generic over the state and both models' snapshots implement this.
pub trait PromptState: Send + 'static {
    /// Tokens the state covers. Checked against the entry's key count on load,
    /// which is what catches a file from another build.
    fn pos(&self) -> usize;
    /// Size of what `encode` will write.
    fn encoded_len(&self) -> usize;
    /// Append the state's bytes.
    fn encode(&self, out: &mut Vec<u8>);
    /// Override for large states to avoid a second full in-memory copy.
    fn write_to(&self, out: &mut dyn Write) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(self.encoded_len());
        self.encode(&mut bytes);
        out.write_all(&bytes)
    }
    /// Read back what `encode` wrote.
    fn decode(bytes: &[u8]) -> oaiy_engine::Result<Self>
    where
        Self: Sized;
    fn read_from(input: &mut dyn Read, length: u64) -> oaiy_engine::Result<Self> where Self: Sized {
        let mut bytes = Vec::new();
        input.take(length).read_to_end(&mut bytes)?;
        Self::decode(&bytes)
    }
}

impl PromptState for llama_rs::glm5next::forward::StateSnapshot {
    fn pos(&self) -> usize {
        self.len
    }
    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
    fn encode(&self, out: &mut Vec<u8>) {
        self.encode(out)
    }
    fn decode(bytes: &[u8]) -> oaiy_engine::Result<Self> {
        Self::decode(bytes).map_err(|e| oaiy_engine::Error::Format(e.to_string()))
    }
}

const MAGIC: &[u8; 8] = b"OAIYPRM1";
const EXT: &str = "oaiystate";

struct Entry {
    path: PathBuf,
    keys: Vec<u64>,
    bytes: u64,
    used: SystemTime,
    /// A system prompt's (not replaced by the conversations that follow).
    base: bool,
    // 0: writer pending; 1: atomically published; 2: failed (may retry).
    ready: Arc<AtomicU8>,
}

enum Job<S> {
    Write { path: PathBuf, header: Vec<u8>, snap: S, ready: Arc<AtomicU8> },
    Delete(PathBuf),
}

pub struct DiskCache<S: PromptState> {
    dir: PathBuf,
    fingerprint: u64,
    budget: u64,
    entries: Vec<Entry>,
    writer: SyncSender<Job<S>>,
}

/// FNV-1a, for file names and fingerprints (not security).
pub fn fnv(bytes: &[u8], seed: u64) -> u64 {
    bytes.iter().fold(seed ^ 0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

fn keys_hash(keys: &[u64]) -> u64 {
    keys.iter().fold(0, |h, k| fnv(&k.to_le_bytes(), h))
}

/// A file's header: magic, fingerprint, kind, keys.
fn header(fingerprint: u64, base: bool, keys: &[u64]) -> Vec<u8> {
    let mut h = Vec::with_capacity(25 + keys.len() * 8);
    h.extend(MAGIC);
    h.extend(fingerprint.to_le_bytes());
    h.push(u8::from(base));
    h.extend((keys.len() as u64).to_le_bytes());
    for k in keys {
        h.extend(k.to_le_bytes());
    }
    h
}

/// Read a header: (base, keys, header length), if the file is this model's.
fn read_header(r: &mut impl Read, fingerprint: u64) -> Option<(bool, Vec<u64>, u64)> {
    let mut head = [0u8; 25];
    r.read_exact(&mut head).ok()?;
    if &head[..8] != MAGIC || u64::from_le_bytes(head[8..16].try_into().ok()?) != fingerprint {
        return None;
    }
    let n = u64::from_le_bytes(head[17..25].try_into().ok()?) as usize;
    if n > 1 << 24 {
        return None;
    }
    let mut raw = vec![0u8; n * 8];
    r.read_exact(&mut raw).ok()?;
    let keys = raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
    Some((head[16] == 1, keys, 25 + raw.len() as u64))
}

/// The namespace a model's states are kept under on this computer: on a Mac its fingerprint moved on once.
///
/// A Mac's states from before the GPU's kernels staged their tiles a scalar a thread (`lane_writes` in ggml-rs-wgpu
/// says why) were made by kernels that lost three values in four. The first Mac's Agent, given the fixed engine, read
/// 10,340 tokens of its system prompt from such a state and answered "angangang": the fingerprint names the model's
/// file, not the engine that made the state. Elsewhere the kernels were right, and the namespace is the fingerprint.
fn namespace(fingerprint: u64) -> u64 {
    if cfg!(target_os = "macos") {
        fnv(b"metal-tiles-a-scalar-a-thread", fingerprint)
    } else {
        fingerprint
    }
}

impl<S: PromptState> DiskCache<S> {
    /// Open (or make) the directory and index this model's entries. This model's states from before its
    /// [`namespace`] moved on are deleted: they are wrong, and as another namespace's no budget would ever take them.
    pub fn open(dir: &Path, fingerprint: u64, budget_bytes: u64) -> oaiy_engine::Result<DiskCache<S>> {
        let (before, fingerprint) = (fingerprint, namespace(fingerprint));
        fs::create_dir_all(dir)?;
        let mut entries = Vec::new();
        for item in fs::read_dir(dir)?.flatten() {
            let path = item.path();
            if path.extension().and_then(|e| e.to_str()) != Some(EXT) {
                continue;
            }
            let Ok(mut file) = File::open(&path) else { continue };
            let Some((base, keys, _)) = read_header(&mut file, fingerprint) else {
                if before != fingerprint && File::open(&path).is_ok_and(|mut f| read_header(&mut f, before).is_some()) {
                    let _ = fs::remove_file(&path);
                }
                continue;
            };
            let meta = item.metadata()?;
            entries.push(Entry { path, keys, bytes: meta.len(), used: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), base, ready: Arc::new(AtomicU8::new(1)) });
        }
        // Backpressure bounds retained snapshots if storage falls behind inference.
        let (writer, jobs) = sync_channel::<Job<S>>(1);
        std::thread::Builder::new()
            .name("prompt cache".into())
            .spawn(move || {
                for job in jobs {
                    match job {
                        Job::Write { path, header, snap, ready } => {
                            let tmp = path.with_extension("tmp");
                            let wrote = File::create(&tmp).and_then(|mut f| {
                                f.write_all(&header)?;
                                snap.write_to(&mut f)
                            }).and_then(|()| fs::rename(&tmp, &path));
                            if let Err(error) = &wrote {
                                eprintln!("oaiy-llm-server: writing prompt checkpoint {} failed: {error}", path.display());
                                let _ = fs::remove_file(&tmp);
                            }
                            ready.store(if wrote.is_ok() { 1 } else { 2 }, Ordering::Release);
                        }
                        Job::Delete(path) => {
                            let _ = fs::remove_file(path);
                        }
                    }
                }
            })
            .map_err(oaiy_engine::Error::Io)?;
        Ok(DiskCache { dir: dir.to_path_buf(), fingerprint, budget: budget_bytes, entries, writer })
    }

    /// Entries this model has on disk.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The longest entry that is a prefix of `prompt` no longer than
    /// `limit`: its index and length.
    pub fn best(&self, prompt: &[u64], limit: usize) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.ready.load(Ordering::Acquire) == 1 && e.keys.len() <= limit && prompt.starts_with(&e.keys))
            .max_by_key(|(_, e)| e.keys.len())
            .map(|(i, e)| (i, e.keys.len()))
    }

    /// Read entry `i`: its keys and snapshot. A damaged or missing file is
    /// forgotten.
    pub fn load(&mut self, i: usize) -> oaiy_engine::Result<(Vec<u64>, S)> {
        let read = || -> oaiy_engine::Result<(Vec<u64>, S)> {
            let mut file = File::open(&self.entries[i].path)?;
            let (_, keys, header_len) = read_header(&mut file, self.fingerprint).ok_or_else(|| oaiy_engine::Error::Format("not a prompt state of this model".into()))?;
            let remaining = file.metadata()?.len().saturating_sub(header_len);
            let snap = S::read_from(&mut file, remaining)?;
            if snap.pos() != keys.len() {
                return Err(oaiy_engine::Error::Format("damaged prompt state".into()));
            }
            Ok((keys, snap))
        };
        match read() {
            Ok(got) => {
                let now = SystemTime::now();
                self.entries[i].used = now;
                if let Ok(f) = File::options().write(true).open(&self.entries[i].path) {
                    let _ = f.set_modified(now);
                }
                Ok(got)
            }
            Err(e) => {
                let gone = self.entries.remove(i);
                let _ = self.writer.send(Job::Delete(gone.path));
                Err(e)
            }
        }
    }

    /// Try successively shorter exact prefixes if a published file is damaged
    /// or was removed externally. Pending writes are never selected or deleted.
    pub fn load_best(&mut self, prompt: &[u64], limit: usize, after: usize, warn: bool) -> Option<(Vec<u64>, S)> {
        while let Some((i, _)) = self.best(prompt, limit).filter(|&(_, len)| len > after) {
            match self.load(i) {
                Ok(state) => return Some(state),
                Err(e) if warn => eprintln!("oaiy-llm-server: skipping unavailable prompt checkpoint: {e}"),
                Err(_) => {},
            }
        }
        None
    }

    /// Whether an entry holds exactly these keys.
    pub fn has(&self, keys: &[u64]) -> bool {
        self.entries.iter().any(|e| e.ready.load(Ordering::Acquire) != 2 && e.keys == keys)
    }

    /// Keep `snap`, the state after `keys` (`base`: a system prompt's).
    /// Prefix entries remain useful for branches and re-rendered tool messages;
    /// least recently used entries go when the byte budget is exceeded.
    pub fn save(&mut self, keys: Vec<u64>, snap: S, base: bool) {
        let path = self.dir.join(format!("{:016x}-{:016x}.{EXT}", self.fingerprint, keys_hash(&keys)));
        let header = header(self.fingerprint, base, &keys);
        let bytes = (header.len() + snap.encoded_len()) as u64;
        // Preserve prefix checkpoints: rendering a completed reply/tool call can
        // change its suffix, making an older prefix the longest usable state.
        // The existing byte-budget LRU bounds storage. Deduplicate exact keys.
        if self.has(&keys) { return; }
        self.entries.retain(|e| e.ready.load(Ordering::Acquire) != 2);
        let ready = Arc::new(AtomicU8::new(0));
        if self.writer.send(Job::Write { path: path.clone(), header, snap, ready: ready.clone() }).is_err() { return; }
        self.entries.push(Entry { path, keys, bytes, used: SystemTime::now(), base, ready });
        while self.entries.iter().map(|e| e.bytes).sum::<u64>() > self.budget && self.entries.len() > 1 {
            let oldest = (0..self.entries.len() - 1).min_by_key(|&i| (self.entries[i].base, self.entries[i].used)).expect("two entries or more");
            let gone = self.entries.remove(oldest);
            let _ = self.writer.send(Job::Delete(gone.path));
        }
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use std::sync::{Mutex, mpsc::{Receiver, Sender, channel}};
    struct State {
        pos: usize,
        gate: Option<(Sender<()>, Mutex<Receiver<()>>)>,
    }
    impl PromptState for State {
        fn pos(&self) -> usize { self.pos }
        fn encoded_len(&self) -> usize { 8 }
        fn encode(&self, out: &mut Vec<u8>) {
            if let Some((started, release)) = &self.gate {
                let _ = started.send(());
                let _ = release.lock().unwrap().recv();
            }
            out.extend((self.pos as u64).to_le_bytes());
        }
        fn decode(bytes: &[u8]) -> oaiy_engine::Result<Self> {
            let b: [u8; 8] = bytes.try_into().map_err(|_| oaiy_engine::Error::Format("bad fixture".into()))?;
            Ok(Self { pos: u64::from_le_bytes(b) as usize, gate: None })
        }
    }
    fn state(pos: usize) -> State { State { pos, gate: None } }
    fn directory(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oaiy-cache-{name}-{}-{}", std::process::id(), SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
    fn settled(c: &DiskCache<State>) {
        for _ in 0..500 {
            if c.entries.iter().all(|e| e.ready.load(Ordering::Acquire) != 0) { return; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("cache writer did not finish");
    }

    #[test]
    fn pending_write_keeps_older_checkpoint_and_is_not_deleted() {
        let dir = directory("pending");
        let mut cache = DiskCache::open(&dir, 1, 10000).unwrap();
        cache.save(vec![1,2], state(2), true);
        settled(&cache);
        let (started, rx) = channel();
        let (release, gate) = channel();
        cache.save(vec![1,2,3,4], State { pos: 4, gate: Some((started, Mutex::new(gate))) }, false);
        rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert!(cache.has(&[1,2,3,4])); // suppress duplicate queued snapshots
        assert_eq!(cache.load_best(&[1,2,3,4,5], 4, 0, false).unwrap().0, [1,2]);
        assert_eq!(cache.len(), 2); // no deletion of the unpublished file
        release.send(()).unwrap();
        settled(&cache);
        assert_eq!(cache.load_best(&[1,2,3,4,5], 4, 0, false).unwrap().0, [1,2,3,4]);
        let mut reopened: DiskCache<State> = DiskCache::open(&dir, 1, 10000).unwrap();
        assert_eq!(reopened.load_best(&[1,2,3,4,5], 4, 0, false).unwrap().1.pos(), 4);
        drop(reopened); drop(cache);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn damaged_and_missing_longest_prefix_fall_back_without_reprocessing_all_history() {
        let dir = directory("fallback");
        let mut cache = DiskCache::open(&dir, 1, 10000).unwrap();
        for n in [2,4,6] { cache.save((1..=n).collect(), state(n as usize), false); }
        settled(&cache);
        fs::write(&cache.entries[2].path, b"broken").unwrap();
        assert_eq!(cache.load_best(&[1,2,3,4,5,6,7], 6, 0, false).unwrap().0, [1,2,3,4]);
        fs::remove_file(&cache.entries[1].path).unwrap();
        assert_eq!(cache.load_best(&[1,2,3,4,5,6,7], 6, 0, false).unwrap().0, [1,2]);
        assert!(cache.load_best(&[1,2,3], 2, 2, false).is_none()); // never replace a better live state
        assert!(cache.load_best(&[9,2,3], 2, 0, false).is_none()); // never reuse a different prefix
        drop(cache);
        let _ = fs::remove_dir_all(dir);
    }

    /// A Mac's states written before the namespace moved on are not read, and go; another model's stay. Elsewhere the
    /// namespace is the fingerprint, and the same file is read.
    #[test]
    fn a_macs_states_from_before_the_scalar_tiles_are_not_read() {
        let dir = directory("namespace");
        let write = |fingerprint: u64, keys: &[u64]| {
            let path = dir.join(format!("{fingerprint:016x}-{:016x}.{EXT}", keys_hash(keys)));
            let mut bytes = header(fingerprint, false, keys);
            state(keys.len()).encode(&mut bytes);
            fs::write(&path, bytes).unwrap();
            path
        };
        let old = write(1, &[1, 2]);
        let other = write(7, &[1, 2]);
        let mut cache: DiskCache<State> = DiskCache::open(&dir, 1, 10000).unwrap();
        if cfg!(target_os = "macos") {
            assert_eq!(cache.len(), 0, "a state from before is not this namespace's");
            assert!(cache.load_best(&[1, 2, 3], 2, 0, false).is_none());
            assert!(!old.exists(), "and it is deleted");
        } else {
            assert_eq!(cache.load_best(&[1, 2, 3], 2, 0, false).unwrap().0, [1, 2]);
        }
        assert!(other.exists(), "another model's state is left alone");
        drop(cache);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_publication_can_be_saved_again() {
        let dir = directory("retry");
        let mut cache = DiskCache::open(&dir, 1, 10000).unwrap();
        let path = dir.join(format!("{:016x}-{:016x}.{EXT}", namespace(1), keys_hash(&[1,2])));
        fs::create_dir(&path).unwrap(); // atomic rename fails against this directory
        cache.save(vec![1,2], state(2), false);
        settled(&cache);
        assert!(!cache.has(&[1,2]));
        assert!(cache.best(&[1,2,3], 2).is_none());
        fs::remove_dir(path).unwrap();
        cache.save(vec![1,2], state(2), false);
        settled(&cache);
        assert_eq!(cache.load_best(&[1,2,3], 2, 0, false).unwrap().0, [1,2]);
        drop(cache);
        fs::remove_dir_all(dir).unwrap();
    }
}
