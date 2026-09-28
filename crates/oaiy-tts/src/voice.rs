//! A reusable voice: the transcript and codec codes of its reference clip,
//! and its speaker embedding. A few kilobytes; no audio is needed to use it.
use crate::talker::AUDIO_CODES;
use oaiy_engine::json::Json;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Voice {
    pub name: String,
    pub description: String,
    pub language: String,
    pub ref_text: String,
    pub ref_codes: Vec<[u32; 16]>,
    /// The speaker embedding: 2048 values for the 1.7B Base, 1024 for the
    /// 0.6B; none for a voice made with Breeze TTS 2 (which needs none).
    pub speaker: Vec<f32>,
}

impl Voice {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("oaiy_voice", Json::Int(1)),
            ("name", Json::str(&self.name)),
            ("description", Json::str(&self.description)),
            ("language", Json::str(&self.language)),
            ("ref_text", Json::str(&self.ref_text)),
            ("ref_codes", Json::Arr(self.ref_codes.iter().map(|f| Json::Arr(f.iter().map(|&c| Json::Int(c as i64)).collect())).collect())),
            ("speaker", Json::Arr(self.speaker.iter().map(|&v| Json::Num(v as f64)).collect())),
        ])
    }

    pub fn from_json(j: &Json) -> Result<Self, String> {
        if j.get("oaiy_voice").and_then(Json::as_i64) != Some(1) {
            return Err("not an OAIY voice file".into());
        }
        let s = |k: &str| j.get(k).and_then(Json::as_str).unwrap_or_default().to_string();
        let ref_codes = j
            .get("ref_codes")
            .and_then(Json::as_array)
            .ok_or("voice: missing ref_codes")?
            .iter()
            .map(|f| {
                let v: Vec<u32> = f.as_array().unwrap_or(&[]).iter().filter_map(|c| c.as_i64()).map(|c| c as u32).collect();
                <[u32; 16]>::try_from(v).map_err(|_| "voice: every frame needs 16 codes".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let speaker: Vec<f32> = j.get("speaker").and_then(Json::as_array).ok_or("voice: missing speaker")?.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
        if !matches!(speaker.len(), 0 | 1024 | 2048) || ref_codes.is_empty() || ref_codes.iter().flatten().any(|&c| c >= AUDIO_CODES) {
            return Err("voice: needs valid reference codes (and a 1024- or 2048-value speaker embedding, or none)".into());
        }
        Ok(Self { name: s("name"), description: s("description"), language: s("language"), ref_text: s("ref_text"), ref_codes, speaker })
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Written aside and renamed, so a reader never sees half a file.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, self.to_json().to_json())?;
        std::fs::rename(&tmp, path)
    }

    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("voice file {}: {e}", path.display()))?;
        Self::from_json(&Json::parse(&bytes).map_err(|e| format!("voice file {}: {e}", path.display()))?)
    }
}

/// 64-bit FNV-1a, continued from `h`.
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Where a voice made from `audio` (the file's bytes) and `transcript` for a
/// talker `width` wide is kept: the same clip, words and model size give the
/// same file.
pub fn cache_path(dir: &Path, audio: &[u8], transcript: &str, width: usize) -> PathBuf {
    let mut h = fnv(0xcbf2_9ce4_8422_2325, &(audio.len() as u64).to_le_bytes());
    h = fnv(h, audio);
    h = fnv(h, &[0xff]);
    h = fnv(h, transcript.trim().as_bytes());
    h = fnv(h, &(width as u64).to_le_bytes());
    dir.join(format!("voice-{h:016x}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(speaker: usize) -> Voice {
        Voice {
            name: "n".into(),
            description: String::new(),
            language: "auto".into(),
            ref_text: "Hello there.".into(),
            ref_codes: vec![[1; 16], [2047; 16]],
            speaker: (0..speaker).map(|i| i as f32 * 0.25).collect(),
        }
    }

    #[test]
    fn voices_round_trip_through_their_files() {
        let dir = std::env::temp_dir().join(format!("oaiy-tts-voice-{}", std::process::id()));
        for n in [1024, 2048, 0] {
            let v = voice(n);
            let path = dir.join(format!("v{n}.json"));
            v.save(&path).unwrap();
            assert_eq!(Voice::open(&path).unwrap(), v);
        }
        let mut bad = voice(1024);
        bad.ref_codes[0][3] = 2048;
        assert!(Voice::from_json(&bad.to_json()).is_err());
        assert!(Voice::from_json(&voice(512).to_json()).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_cache_key_follows_the_clip_the_words_and_the_model_size() {
        let dir = Path::new("c");
        let a = cache_path(dir, b"clip", "Hello.", 1024);
        assert_eq!(a, cache_path(dir, b"clip", " Hello. ", 1024));
        assert_ne!(a, cache_path(dir, b"clip2", "Hello.", 1024));
        assert_ne!(a, cache_path(dir, b"clip", "Hello!", 1024));
        assert_ne!(a, cache_path(dir, b"clip", "Hello.", 2048));
    }
}
