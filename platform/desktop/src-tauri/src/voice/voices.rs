//! The voice calls are answered in.
//!
//! A voice is a clip of someone speaking (WAV, MP3, ...) in `<data>/voices`,
//! named by its file name; the one chosen is in `<data>/voices/chosen`. OAIY's
//! speech server (the `oaiy-voice` service, which reads this folder) makes a
//! voice from a clip the first time it speaks in it, hearing what the clip
//! says itself when nothing is written beside it (`NAME.txt`). A folder with
//! no clips is given OAIY's own receptionist voice.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The audio a clip may be.
pub const CLIP_EXTENSIONS: [&str; 8] = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "opus", "aac"];
/// The largest clip taken (a clip is cut to 30 seconds anyway).
pub const MAX_CLIP_BYTES: usize = 20 * 1024 * 1024;

/// OAIY's own voice: a warm receptionist, made with Qwen3-TTS VoiceDesign.
const RECEPTIONIST: (&str, &[u8], &str) = (
    "receptionist",
    include_bytes!("../../resources/voices/receptionist.wav"),
    include_str!("../../resources/voices/receptionist.txt"),
);

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Where the voices are: `<data>/voices`, made (with OAIY's own voice) if it has none.
pub fn init(data_dir: &Path) {
    let dir = data_dir.join("voices");
    if let Err(e) = seed(&dir) {
        log::warn!("the voices folder {}: {e}", dir.display());
    }
    let _ = DIR.set(dir);
}

fn seed(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    if list_in(dir).is_empty() {
        let (name, clip, words) = RECEPTIONIST;
        std::fs::write(dir.join(format!("{name}.wav")), clip)?;
        std::fs::write(dir.join(format!("{name}.txt")), words.trim())?;
    }
    Ok(())
}

pub fn dir() -> Option<&'static Path> {
    DIR.get().map(PathBuf::as_path)
}

/// A voice in the folder.
#[derive(Clone, Debug, serde::Serialize, PartialEq)]
pub struct VoiceClip {
    pub name: String,
    pub file: String,
    pub bytes: u64,
    /// What it says is written beside it (else the speech server hears it).
    pub written: bool,
}

fn is_clip(p: &Path) -> bool {
    p.is_file() && p.extension().is_some_and(|e| CLIP_EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

fn list_in(dir: &Path) -> Vec<VoiceClip> {
    let mut clips: Vec<VoiceClip> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_clip(p))
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some(VoiceClip { written: p.with_extension("txt").is_file(), bytes: std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), file: p.file_name()?.to_string_lossy().into_owned(), name })
        })
        .collect();
    clips.sort_by_key(|c| c.name.to_lowercase());
    clips
}

/// The voices, by name.
pub fn list() -> Vec<VoiceClip> {
    dir().map(list_in).unwrap_or_default()
}

/// The voice calls are answered in: the one chosen while its clip is there,
/// else the first there is.
pub fn chosen() -> Option<String> {
    let dir = dir()?;
    chosen_in(dir)
}

fn chosen_in(dir: &Path) -> Option<String> {
    let clips = list_in(dir);
    let picked = std::fs::read_to_string(dir.join("chosen")).ok().map(|s| s.trim().to_string());
    picked
        .and_then(|p| clips.iter().find(|c| c.name.eq_ignore_ascii_case(&p)).map(|c| c.name.clone()))
        .or_else(|| clips.first().map(|c| c.name.clone()))
}

/// A name a clip can be kept under: letters, digits, `-` and `_` (spaces become `-`).
pub fn clean_name(name: &str) -> Result<String, String> {
    let cleaned: String = name.trim().chars().map(|c| if c.is_whitespace() { '-' } else { c }).filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')).take(48).collect();
    let cleaned = cleaned.trim_matches('-').to_string();
    if cleaned.is_empty() || cleaned.eq_ignore_ascii_case("chosen") {
        return Err("give the voice a name (letters, digits, - or _)".into());
    }
    Ok(cleaned)
}

/// Choose the voice calls are answered in.
pub fn choose(name: &str) -> Result<String, String> {
    let dir = dir().ok_or("the voices folder is not set up")?;
    let clip = list_in(dir).into_iter().find(|c| c.name.eq_ignore_ascii_case(name.trim())).ok_or_else(|| format!("no voice called {name:?}"))?;
    std::fs::write(dir.join("chosen"), &clip.name).map_err(|e| e.to_string())?;
    Ok(clip.name)
}

/// Keep a clip as the voice `name` (replacing one of that name), and what it says when given.
pub fn add(name: &str, extension: &str, clip: &[u8], words: Option<&str>) -> Result<VoiceClip, String> {
    let dir = dir().ok_or("the voices folder is not set up")?;
    add_in(dir, name, extension, clip, words)
}

fn add_in(dir: &Path, name: &str, extension: &str, clip: &[u8], words: Option<&str>) -> Result<VoiceClip, String> {
    let name = clean_name(name)?;
    let extension = extension.trim().trim_start_matches('.').to_ascii_lowercase();
    if !CLIP_EXTENSIONS.contains(&extension.as_str()) {
        return Err(format!("a voice clip is {} audio, not .{extension}", CLIP_EXTENSIONS.join(", ")));
    }
    if clip.len() < 1024 {
        return Err("that clip is empty".into());
    }
    if clip.len() > MAX_CLIP_BYTES {
        return Err(format!("that clip is over {} MB: 5 to 30 seconds of speech is best", MAX_CLIP_BYTES / (1024 * 1024)));
    }
    remove_in(dir, &name);
    let file = dir.join(format!("{name}.{extension}"));
    std::fs::write(&file, clip).map_err(|e| e.to_string())?;
    if let Some(words) = words.map(str::trim).filter(|w| !w.is_empty()) {
        std::fs::write(dir.join(format!("{name}.txt")), words).map_err(|e| e.to_string())?;
    }
    list_in(dir).into_iter().find(|c| c.name == name).ok_or_else(|| "the clip was not kept".to_string())
}

/// Remove a voice (its clip, and what was written beside it).
pub fn remove(name: &str) -> Result<(), String> {
    let dir = dir().ok_or("the voices folder is not set up")?;
    let name = clean_name(name)?;
    if !list_in(dir).iter().any(|c| c.name.eq_ignore_ascii_case(&name)) {
        return Err(format!("no voice called {name:?}"));
    }
    remove_in(dir, &name);
    Ok(())
}

fn remove_in(dir: &Path, name: &str) {
    for c in list_in(dir).into_iter().filter(|c| c.name.eq_ignore_ascii_case(name)) {
        let _ = std::fs::remove_file(dir.join(&c.file));
        let _ = std::fs::remove_file(dir.join(format!("{}.txt", c.name)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oaiy-desktop-voices-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_new_folder_gets_oaiy_s_own_voice_and_it_is_chosen() {
        let dir = folder("seed");
        seed(&dir).unwrap();
        let clips = list_in(&dir);
        assert_eq!(clips.len(), 1);
        assert_eq!((clips[0].name.as_str(), clips[0].written), ("receptionist", true));
        assert_eq!(chosen_in(&dir).as_deref(), Some("receptionist"));
        // Seeded once: a folder with a voice is left as it is.
        std::fs::remove_file(dir.join("receptionist.txt")).unwrap();
        seed(&dir).unwrap();
        assert!(!list_in(&dir)[0].written);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_clip_is_kept_by_name_replacing_one_of_that_name_and_can_be_chosen() {
        let dir = folder("add");
        std::fs::create_dir_all(&dir).unwrap();
        let clip = vec![1u8; 4096];
        let v = add_in(&dir, "  Front desk ", "MP3", &clip, Some("Hi there.")).unwrap();
        assert_eq!((v.name.as_str(), v.file.as_str(), v.written), ("Front-desk", "Front-desk.mp3", true));
        let again = add_in(&dir, "Front-desk", ".wav", &clip, None).unwrap();
        assert_eq!((again.file.as_str(), again.written), ("Front-desk.wav", false), "the new clip's words are heard anew");
        assert_eq!(list_in(&dir).len(), 1);
        std::fs::write(dir.join("chosen"), "front-desk").unwrap();
        assert_eq!(chosen_in(&dir).as_deref(), Some("Front-desk"));
        std::fs::write(dir.join("chosen"), "gone").unwrap();
        assert_eq!(chosen_in(&dir).as_deref(), Some("Front-desk"), "a chosen voice that went falls back to one there is");
        assert!(add_in(&dir, "x", "exe", &clip, None).is_err());
        assert!(add_in(&dir, "x", "wav", &[0; 10], None).is_err());
        assert!(add_in(&dir, "../..", "wav", &clip, None).is_err());
        assert!(add_in(&dir, "chosen", "wav", &clip, None).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn names_are_cleaned() {
        assert_eq!(clean_name("My voice (2)").unwrap(), "My-voice-2");
        assert_eq!(clean_name("..\\evil/../x").unwrap(), "evilx");
        assert!(clean_name("   ").is_err());
    }
}
