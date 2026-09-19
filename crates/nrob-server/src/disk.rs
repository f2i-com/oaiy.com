//! Prompt states kept on disk, so that a new process starts where an
//! earlier one left off: the system prompt a harness sends every time, and
//! the conversations it may resume.
//!
//! An entry is a [`Snapshot`] of the model's state after a prompt prefix,
//! with that prefix's keys (see `engine::prompt_keys`; prompts with images
//! are not kept). The engine saves one where a conversation's first user
//! message begins (the system prompt and tools), and for the conversation
//! after each layered pass, where its newest user message begins and at the
//! end of each prompt; a newer entry replaces its conversation's shorter
//! ones. When a
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
use std::sync::mpsc::{channel, Sender};
use std::time::SystemTime;

use dsv41_cuda::Snapshot;

const MAGIC: &[u8; 8] = b"NROBPRM1";
const EXT: &str = "nrobstate";

struct Entry {
    path: PathBuf,
    keys: Vec<u64>,
    bytes: u64,
    used: SystemTime,
    /// A system prompt's (not replaced by the conversations that follow).
    base: bool,
}

enum Job {
    Write { path: PathBuf, header: Vec<u8>, snap: Snapshot },
    Delete(PathBuf),
}

pub struct DiskCache {
    dir: PathBuf,
    fingerprint: u64,
    budget: u64,
    entries: Vec<Entry>,
    writer: Sender<Job>,
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

impl DiskCache {
    /// Open (or make) the directory and index this model's entries.
    pub fn open(dir: &Path, fingerprint: u64, budget_bytes: u64) -> nrob::Result<DiskCache> {
        fs::create_dir_all(dir)?;
        let mut entries = Vec::new();
        for item in fs::read_dir(dir)?.flatten() {
            let path = item.path();
            if path.extension().and_then(|e| e.to_str()) != Some(EXT) {
                continue;
            }
            let Ok(mut file) = File::open(&path) else { continue };
            let Some((base, keys, _)) = read_header(&mut file, fingerprint) else { continue };
            let meta = item.metadata()?;
            entries.push(Entry { path, keys, bytes: meta.len(), used: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), base });
        }
        let (writer, jobs) = channel::<Job>();
        std::thread::Builder::new()
            .name("prompt cache".into())
            .spawn(move || {
                for job in jobs {
                    match job {
                        Job::Write { path, header, snap } => {
                            let tmp = path.with_extension("tmp");
                            let mut bytes = header;
                            snap.encode(&mut bytes);
                            let wrote = File::create(&tmp).and_then(|mut f| f.write_all(&bytes)).and_then(|()| fs::rename(&tmp, &path));
                            if wrote.is_err() {
                                let _ = fs::remove_file(&tmp);
                            }
                        }
                        Job::Delete(path) => {
                            let _ = fs::remove_file(path);
                        }
                    }
                }
            })
            .map_err(nrob::Error::Io)?;
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
            .filter(|(_, e)| e.keys.len() <= limit && prompt.starts_with(&e.keys))
            .max_by_key(|(_, e)| e.keys.len())
            .map(|(i, e)| (i, e.keys.len()))
    }

    /// Read entry `i`: its keys and snapshot. A damaged or missing file is
    /// forgotten.
    pub fn load(&mut self, i: usize) -> nrob::Result<(Vec<u64>, Snapshot)> {
        let read = || -> nrob::Result<(Vec<u64>, Snapshot)> {
            let mut file = File::open(&self.entries[i].path)?;
            let (_, keys, _) = read_header(&mut file, self.fingerprint).ok_or_else(|| nrob::Error::Format("not a prompt state of this model".into()))?;
            let mut rest = Vec::new();
            file.read_to_end(&mut rest)?;
            let snap = Snapshot::decode(&rest)?;
            if snap.pos() != keys.len() {
                return Err(nrob::Error::Format("damaged prompt state".into()));
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

    /// Whether an entry holds exactly these keys.
    pub fn has(&self, keys: &[u64]) -> bool {
        self.entries.iter().any(|e| e.keys == keys)
    }

    /// Keep `snap`, the state after `keys` (`base`: a system prompt's).
    /// Shorter entries of the same conversation go, and then the least
    /// recently used while the entries pass the budget.
    pub fn save(&mut self, keys: Vec<u64>, snap: Snapshot, base: bool) {
        let path = self.dir.join(format!("{:016x}-{:016x}.{EXT}", self.fingerprint, keys_hash(&keys)));
        let header = header(self.fingerprint, base, &keys);
        let bytes = (header.len() + snap.encoded_len()) as u64;
        // a conversation's newest state covers its earlier ones
        let superseded: Vec<usize> = (0..self.entries.len()).filter(|&i| !self.entries[i].base && !base && keys.starts_with(&self.entries[i].keys)).collect();
        for i in superseded.into_iter().rev() {
            let gone = self.entries.remove(i);
            let _ = self.writer.send(Job::Delete(gone.path));
        }
        let _ = self.writer.send(Job::Write { path: path.clone(), header, snap });
        self.entries.push(Entry { path, keys, bytes, used: SystemTime::now(), base });
        while self.entries.iter().map(|e| e.bytes).sum::<u64>() > self.budget && self.entries.len() > 1 {
            let oldest = (0..self.entries.len() - 1).min_by_key(|&i| self.entries[i].used).expect("two entries or more");
            let gone = self.entries.remove(oldest);
            let _ = self.writer.send(Job::Delete(gone.path));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A snapshot of `pos` tokens with no layers (what the cache stores
    /// does not matter here).
    fn snap(pos: usize) -> Snapshot {
        let mut bytes = (pos as u64).to_le_bytes().to_vec();
        bytes.extend(0u32.to_le_bytes());
        Snapshot::decode(&bytes).unwrap()
    }

    fn written(dir: &Path, n: usize) -> bool {
        for _ in 0..200 {
            let files = fs::read_dir(dir).unwrap().flatten().filter(|f| f.path().extension().and_then(|e| e.to_str()) == Some(EXT)).count();
            if files == n {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn states_outlive_the_process_and_a_conversation_keeps_its_newest() {
        let dir = std::env::temp_dir().join(format!("nrob-prompt-cache-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut cache = DiskCache::open(&dir, 7, 1 << 30).unwrap();
        cache.save(vec![1, 2, 3], snap(3), true);
        cache.save(vec![1, 2, 3, 4, 5], snap(5), false);
        cache.save(vec![1, 2, 3, 4, 5, 6, 7], snap(7), false);
        // the system prompt's stays; the conversation keeps its newest
        assert_eq!(cache.len(), 2);
        assert!(cache.has(&[1, 2, 3]) && !cache.has(&[1, 2, 3, 4, 5]));
        assert_eq!(cache.best(&[1, 2, 3, 4, 5, 6, 7, 8], 7).map(|b| b.1), Some(7));
        assert_eq!(cache.best(&[1, 2, 3, 9], 3).map(|b| b.1), Some(3));
        assert_eq!(cache.best(&[1, 2, 3, 4, 5, 6, 7], 6).map(|b| b.1), Some(3));
        assert_eq!(cache.best(&[2, 3], 1), None);
        assert!(written(&dir, 2));

        // another process of the same model finds them; another model does not
        let mut again = DiskCache::open(&dir, 7, 1 << 30).unwrap();
        assert_eq!(again.len(), 2);
        let (i, len) = again.best(&[1, 2, 3, 4, 5, 6, 7, 8], 7).unwrap();
        let (keys, state) = again.load(i).unwrap();
        assert_eq!((keys, state.pos(), len), (vec![1, 2, 3, 4, 5, 6, 7], 7, 7));
        assert_eq!(DiskCache::open(&dir, 8, 1 << 30).unwrap().len(), 0);

        // over the budget the least recently used go
        let mut small = DiskCache::open(&dir, 7, 150).unwrap();
        small.save(vec![9; 8], snap(8), true);
        assert_eq!(small.len(), 1);
        assert!(small.has(&[9; 8]));
        let _ = fs::remove_dir_all(&dir);
    }
}
