//! Small, disposable prompt-conditioning cache. Published weights are never copied.
use super::Request;
use candle_core::{DType, Device, Result, Tensor};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

/// Contexts are 1024 tokens of the stream's width, stored as F32.
const TOKENS: usize = 1024;
/// A prompt with audio keeps two entries (video and audio).
const ENTRIES: usize = 16;

fn hash(value: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

pub struct PromptCache {
    directory: PathBuf,
    path: PathBuf,
    key: String,
    width: usize,
}
impl PromptCache {
    /// The video stream's connector output (1024 x 4096).
    pub fn new(r: &Request) -> Option<Self> {
        Self::stream(r, "nrob-ltx-conditioning-v2", 4096)
    }
    /// The audio stream's connector output (1024 x 2048).
    pub fn audio(r: &Request) -> Option<Self> {
        Self::stream(r, "nrob-ltx-audio-conditioning-v1", 2048)
    }
    fn stream(r: &Request, version: &str, width: usize) -> Option<Self> {
        let directory = r.cache_dir.clone()?;
        let mut key = format!("{version}\n{}\n{}\n", r.model, r.prompt);
        for path in [&r.transformer, &r.text_encoder]
            .into_iter()
            .chain(r.tokenizer.iter())
        {
            let meta = std::fs::metadata(path).ok()?;
            let modified = meta
                .modified()
                .ok()?
                .duration_since(UNIX_EPOCH)
                .ok()?
                .as_nanos();
            key.push_str(&format!(
                "{:?}:{}:{modified}\n",
                path.canonicalize().ok()?,
                meta.len()
            ));
        }
        let path = directory.join(format!("{:016x}.ltx-context", hash(key.as_bytes())));
        Some(Self {
            directory,
            path,
            key,
            width,
        })
    }
    fn bytes(&self) -> usize {
        TOKENS * self.width * 4
    }
    pub fn load(&self, dev: &Device) -> Option<Tensor> {
        let mut f = std::fs::File::open(&self.path).ok()?;
        if f.metadata().ok()?.len() != (16 + self.key.len() + self.bytes()) as u64 {
            return None;
        }
        let mut header = [0u8; 16];
        f.read_exact(&mut header).ok()?;
        let length = u64::from_le_bytes(header[..8].try_into().ok()?);
        if length != self.key.len() as u64 {
            return None;
        }
        let expected = u64::from_le_bytes(header[8..].try_into().ok()?);
        let mut key = vec![0u8; self.key.len()];
        f.read_exact(&mut key).ok()?;
        if key != self.key.as_bytes() {
            return None;
        }
        let mut data = vec![0u8; self.bytes()];
        f.read_exact(&mut data).ok()?;
        if hash(&data) != expected {
            return None;
        }
        let tensor = Tensor::from_raw_buffer(&data, DType::F32, &[1, TOKENS, self.width], dev)
            .ok()?
            .to_dtype(DType::BF16)
            .ok()?;
        // Best effort LRU; a read-only cache is still useful.
        if let Ok(file) = std::fs::OpenOptions::new().write(true).open(&self.path) {
            let _ = file.set_modified(std::time::SystemTime::now());
        }
        Some(tensor)
    }
    pub fn save(&self, context: &Tensor) -> Result<()> {
        std::fs::create_dir_all(&self.directory)?;
        prune(&self.directory, &self.path)?;
        let values = context
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if values.len() * 4 != self.bytes() || values.iter().any(|v| !v.is_finite()) {
            candle_core::bail!("invalid prompt context cache shape or values");
        }
        let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let temp = self
            .path
            .with_extension(format!("{}.tmp", std::process::id()));
        let result = (|| -> std::io::Result<()> {
            let mut f = std::fs::File::create(&temp)?;
            f.write_all(&(self.key.len() as u64).to_le_bytes())?;
            f.write_all(&hash(&data).to_le_bytes())?;
            f.write_all(self.key.as_bytes())?;
            f.write_all(&data)?;
            drop(f);
            std::fs::rename(&temp, &self.path)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result?;
        Ok(())
    }
}
fn prune(directory: &Path, current: &Path) -> std::io::Result<()> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path == current || path.extension().and_then(|s| s.to_str()) != Some("ltx-context") {
            continue;
        }
        if entry.file_type()?.is_file() {
            entries.push((entry.metadata()?.modified()?, path));
        }
    }
    entries.sort_by_key(|(time, _)| *time);
    let remove = entries.len().saturating_sub(ENTRIES - 1);
    for (_, path) in entries.into_iter().take(remove) {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_checks_key_integrity_and_evicts_only_own_entries() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "nrob-ltx-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root)?;
        let cache = PromptCache {
            directory: root.clone(),
            path: root.join("current.ltx-context"),
            key: "exact prompt and model identity".into(),
            width: 4096,
        };
        let tensor = Tensor::ones((1, 1024, 4096), DType::F32, &Device::Cpu)?;
        cache.save(&tensor)?;
        assert!(cache.load(&Device::Cpu).is_some());
        let wrong = PromptCache {
            directory: root.clone(),
            path: cache.path.clone(),
            key: "different prompt/model stamp!!".into(),
            width: 4096,
        };
        assert!(wrong.load(&Device::Cpu).is_none());
        let mut data = std::fs::read(&cache.path)?;
        *data.last_mut().unwrap() ^= 1;
        std::fs::write(&cache.path, data)?;
        assert!(cache.load(&Device::Cpu).is_none());
        for i in 0..20 {
            std::fs::write(root.join(format!("{i}.ltx-context")), b"old")?;
        }
        std::fs::write(root.join("keep.txt"), b"user file")?;
        prune(&root, &cache.path)?;
        assert_eq!(
            std::fs::read_dir(&root)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("ltx-context"))
                .count(),
            ENTRIES
        );
        assert!(root.join("keep.txt").is_file());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
