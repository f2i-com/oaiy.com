//! Text to speech: Qwen3-TTS on a GPU through WebGPU (`oaiy-media`'s
//! `tts::realtime`), on a thread of its own, in voices made from clips.
//!
//! A voice is a clip of someone speaking (WAV, MP3, anything FFmpeg reads) in
//! the voices folder, named by its file name: `phone.mp3` is the voice
//! "phone". What the clip says comes from `phone.txt` beside it, or a saved
//! voice's `phone.json` (`ref_text`), or else this server's own
//! speech-to-text, which hears exactly the audio the voice is made from. So a
//! clip alone is enough. Voices are made once and kept (the engine's cache,
//! keyed by the clip's bytes and words), and a changed clip makes a new one.
//!
//! The engine keeps the thread that loaded it. Lines are spoken one at a
//! time, in turn; each streams 16-bit PCM at 24 kHz as it is made, and stops
//! when its listener goes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use oaiy_media::tts::realtime::WgpuTts as Tts;
use oaiy_tts::Voice;

/// What speech-to-text gives the voices: the text of mono audio at a rate.
pub type Transcribe = Arc<dyn Fn(&[f32], usize) -> Result<String, String> + Send + Sync>;

/// The audio a voice clip may be.
const CLIP_EXTENSIONS: [&str; 8] = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "opus", "aac"];

/// The voices folder and the voice lines are spoken in unless a request names one.
#[derive(Clone, Debug, Default)]
pub struct Voices {
    pub dir: Option<PathBuf>,
    /// A voice's name in the folder, or a clip's path.
    pub default: Option<String>,
}

impl Voices {
    /// The voices in the folder, by name.
    pub fn list(&self) -> Vec<String> {
        let Some(dir) = &self.dir else { return Vec::new() };
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_clip(p))
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .collect();
        names.sort_by_key(|n| n.to_lowercase());
        names.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        names
    }

    /// The clip for a voice: one named in the folder, else the default.
    pub fn clip(&self, name: Option<&str>) -> Result<PathBuf, String> {
        let name = name.map(str::trim).filter(|n| !n.is_empty());
        if let Some(name) = name {
            // Names only: a request cannot reach a file outside the folder.
            if name.contains(['/', '\\', ':']) || name.starts_with('.') {
                return Err(format!("voice {name:?}: a voice is named by its clip's name in the voices folder"));
            }
            return self.named(name).ok_or_else(|| format!("no voice called {name:?} (the voices: {})", self.list().join(", ")));
        }
        let Some(default) = self.default.as_deref() else {
            // Nothing named: a clip called default, else the first there is.
            return self.named("default").or_else(|| self.list().first().and_then(|n| self.named(n))).ok_or_else(|| "no voice: pass --voice, or put a clip in the voices folder".to_string());
        };
        let as_path = Path::new(default);
        if as_path.is_file() {
            return Ok(as_path.to_path_buf());
        }
        self.named(default).ok_or_else(|| format!("the default voice {default:?} is neither a clip nor a voice in the voices folder"))
    }

    fn named(&self, name: &str) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_clip(p) && p.file_stem().is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case(name)))
            .collect();
        // Two clips for one name: the newest is the one meant.
        found.sort_by_key(|p| std::cmp::Reverse(modified(p)));
        found.into_iter().next()
    }
}

fn is_clip(p: &Path) -> bool {
    p.is_file() && p.extension().is_some_and(|e| CLIP_EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

fn modified(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// What a clip says, when it is written down beside it: `NAME.txt`, or a
/// saved voice's `NAME.json` (`ref_text`).
pub fn written_transcript(clip: &Path) -> Option<String> {
    if let Ok(text) = std::fs::read_to_string(clip.with_extension("txt")) {
        let text = text.trim();
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }
    let json = std::fs::read(clip.with_extension("json")).ok()?;
    let j = oaiy_engine::json::Json::parse(&json).ok()?;
    j.get("ref_text").and_then(|t| t.as_str()).map(str::trim).filter(|t| !t.is_empty()).map(str::to_string)
}

/// How a line went.
#[derive(Clone, Debug)]
pub struct Spoken {
    pub audio_seconds: f64,
    pub first_chunk_seconds: Option<f64>,
    pub total_seconds: f64,
    pub cancelled: bool,
}

struct Job {
    text: String,
    voice: Option<String>,
    cancel: Arc<AtomicBool>,
    chunks: mpsc::Sender<Vec<i16>>,
    done: mpsc::Sender<Result<Spoken, String>>,
}

/// The engine, on its thread.
pub struct Speaker {
    jobs: mpsc::Sender<Job>,
    pub model: String,
    pub device: String,
    pub voices: Voices,
    remembered: Mutex<Vec<(Line, Arc<Vec<i16>>)>>,
}

/// A short line in a voice, as [`Speaker`] keeps what it said: its clip (and the clip's time and size: a changed clip
/// is a new voice) and its words.
#[derive(Clone, PartialEq, Eq)]
struct Line {
    clip: PathBuf,
    stamp: (Option<SystemTime>, u64),
    text: String,
}

/// Lines this short are kept once said, and said again from memory at once: a call's fillers ("Okay —"), its
/// greeting, "One moment, let me check." come up in call after call, and made again each time they held the engine
/// just as the reply's first sentence wanted it. A line made again is the same audio (its sampling is seeded the
/// same), so nothing is heard differently.
const REMEMBER_CHARS: usize = 160;
/// The most recent lines kept: some 10 seconds of 24 kHz audio at most each, under 25 MB in all.
const REMEMBERED: usize = 48;
/// Samples handed out at a time from a kept line (0.2 s).
const RECALL_PIECE: usize = 4_800;

/// How the engine is started.
pub struct Setup {
    pub model_dir: PathBuf,
    /// The GPU it runs on, as nvidia-smi counts them (the WebGPU backend's number too).
    pub gpu: usize,
    pub voices: Voices,
    /// FFmpeg, for clips that are not plain WAV.
    pub ffmpeg: PathBuf,
    /// Where voices made from clips are kept.
    pub cache: Option<PathBuf>,
    pub transcribe: Option<Transcribe>,
}

impl Speaker {
    /// Load the engine on its thread (and make the default voice there, so the
    /// first line does not wait for it). Returns once it is ready.
    pub fn start(setup: Setup) -> Result<Self, String> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (ready_tx, ready) = mpsc::channel::<Result<String, String>>();
        let model = setup.model_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "qwen3-tts".into());
        let voices = setup.voices.clone();
        let gpu = setup.gpu;
        std::thread::Builder::new()
            .name("oaiy-voice tts".into())
            .spawn(move || engine(setup, inbox, ready_tx))
            .map_err(|e| format!("the speech thread: {e}"))?;
        let loaded = ready.recv().map_err(|_| "the speech engine stopped while loading".to_string())??;
        eprintln!("[oaiy-voice] {loaded}");
        Ok(Self { jobs, model, device: format!("webgpu:{gpu}"), voices, remembered: Mutex::default() })
    }

    /// The line `text` in `voice` is, where it is short enough to keep.
    fn line(&self, text: &str, voice: Option<&str>) -> Option<Line> {
        let text = text.trim();
        if text.is_empty() || text.chars().count() > REMEMBER_CHARS {
            return None;
        }
        let clip = self.voices.clip(voice).ok()?;
        let stamp = (modified(&clip), std::fs::metadata(&clip).map(|m| m.len()).unwrap_or(0));
        Some(Line { clip, stamp, text: text.to_string() })
    }

    /// A kept line's audio (now the most recent).
    fn recall(&self, line: &Line) -> Option<Arc<Vec<i16>>> {
        let mut kept = self.remembered.lock().unwrap_or_else(|p| p.into_inner());
        let at = kept.iter().position(|(l, _)| l == line)?;
        let entry = kept.remove(at);
        let pcm = Arc::clone(&entry.1);
        kept.push(entry);
        Some(pcm)
    }

    fn remember(&self, line: Line, pcm: Vec<i16>) {
        let mut kept = self.remembered.lock().unwrap_or_else(|p| p.into_inner());
        kept.retain(|(l, _)| l != &line);
        kept.push((line, Arc::new(pcm)));
        let over = kept.len().saturating_sub(REMEMBERED);
        kept.drain(..over);
    }

    /// Speak `text` in `voice` (the default when `None`), handing each piece of
    /// 24 kHz PCM to `on_chunk` as it is made. `on_chunk` returning false (its
    /// listener went), or `cancel` set, stops the line.
    pub fn say(&self, text: &str, voice: Option<&str>, cancel: Arc<AtomicBool>, on_chunk: &mut dyn FnMut(&[i16]) -> bool) -> Result<Spoken, String> {
        let (chunks, pcm) = mpsc::channel();
        let (done, result) = mpsc::channel();
        self.jobs
            .send(Job { text: text.to_string(), voice: voice.map(str::to_string), cancel: Arc::clone(&cancel), chunks, done })
            .map_err(|_| "the speech engine has stopped".to_string())?;
        // The pieces end when the engine drops its sender (the line is done).
        for piece in pcm {
            if !cancel.load(Ordering::Relaxed) && !on_chunk(&piece) {
                cancel.store(true, Ordering::Relaxed);
            }
        }
        result.recv().map_err(|_| "the speech engine stopped mid-line".to_string())?
    }
}

impl crate::server::TextToSpeech for Speaker {
    fn model(&self) -> String {
        self.model.clone()
    }

    fn voices(&self) -> (Vec<String>, Option<String>) {
        let default = self.voices.clip(None).ok().and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()));
        (self.voices.list(), default)
    }

    fn check_voice(&self, voice: Option<&str>) -> Result<(), String> {
        self.voices.clip(voice).map(|_| ())
    }

    fn speak(&self, text: &str, voice: Option<&str>, cancel: Arc<AtomicBool>, on_chunk: &mut dyn FnMut(&[i16]) -> bool) -> Result<(), String> {
        let line = self.line(text, voice);
        if let Some(pcm) = line.as_ref().and_then(|l| self.recall(l)) {
            for piece in pcm.chunks(RECALL_PIECE) {
                if cancel.load(Ordering::Relaxed) || !on_chunk(piece) {
                    break;
                }
            }
            eprintln!("[oaiy-voice] {:.2} s of speech from memory (said before)", pcm.len() as f64 / 24_000.);
            return Ok(());
        }
        let mut heard: Option<Vec<i16>> = line.is_some().then(Vec::new);
        let spoken = self.say(text, voice, cancel, &mut |pcm| {
            if let Some(h) = heard.as_mut() {
                h.extend_from_slice(pcm);
            }
            on_chunk(pcm)
        })?;
        if !spoken.cancelled {
            eprintln!(
                "[oaiy-voice] {:.2} s of speech in {:.2} s (first audio {:.0} ms)",
                spoken.audio_seconds,
                spoken.total_seconds,
                spoken.first_chunk_seconds.unwrap_or(0.) * 1e3
            );
            if let (Some(line), Some(pcm)) = (line, heard.filter(|h| !h.is_empty())) {
                self.remember(line, pcm);
            }
        }
        Ok(())
    }
}

/// The engine's thread: load, then speak each job in turn.
fn engine(setup: Setup, inbox: mpsc::Receiver<Job>, ready: mpsc::Sender<Result<String, String>>) {
    let started = Instant::now();
    let loaded = Tts::load(&setup.model_dir, setup.gpu).map_err(|e| format!("{}: {e}", setup.model_dir.display()));
    let mut tts = match loaded {
        Ok(t) => t,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    tts.ffmpeg = setup.ffmpeg.clone();
    if setup.cache.is_some() {
        tts.voice_cache = setup.cache.clone();
    }
    let mut voices = VoiceShelf { made: HashMap::new(), transcribe: setup.transcribe.clone() };
    // The default voice, made now (a missing one is only a warning: a request may name one).
    let default = match setup.voices.clip(None).and_then(|clip| voices.voice(&mut tts, &clip)) {
        Ok(v) => format!(", voice {:?}", v.name),
        Err(e) => {
            eprintln!("[oaiy-voice] no default voice yet: {e}");
            String::new()
        }
    };
    let _ = ready.send(Ok(format!("loaded {} on GPU {}, {} through WebGPU, in {:.1} s{default}", setup.model_dir.display(), setup.gpu, tts.adapter(), started.elapsed().as_secs_f64())));
    for job in inbox {
        let result = setup.voices.clip(job.voice.as_deref()).and_then(|clip| voices.voice(&mut tts, &clip)).and_then(|voice| {
            let chunks = job.chunks.clone();
            let cancel = Arc::clone(&job.cancel);
            let report = tts
                .speak(&job.text, &voice, &job.cancel, |pcm| {
                    if chunks.send(pcm.to_vec()).is_err() {
                        cancel.store(true, Ordering::Relaxed);
                    }
                })
                .map_err(|e| e.to_string())?;
            Ok(Spoken {
                audio_seconds: report.audio_seconds,
                first_chunk_seconds: report.first_chunk_seconds,
                total_seconds: report.total_seconds,
                cancelled: report.finish == oaiy_tts::Finish::Cancelled,
            })
        });
        drop(job.chunks);
        let _ = job.done.send(result);
    }
}

/// Voices made so far, by clip (a changed clip is made again).
struct VoiceShelf {
    made: HashMap<PathBuf, (Option<SystemTime>, u64, Voice)>,
    transcribe: Option<Transcribe>,
}

impl VoiceShelf {
    fn voice(&mut self, tts: &mut Tts, clip: &Path) -> Result<Voice, String> {
        let stamp = (modified(clip), std::fs::metadata(clip).map(|m| m.len()).unwrap_or(0));
        if let Some((m, len, v)) = self.made.get(clip) {
            if (*m, *len) == stamp {
                return Ok(v.clone());
            }
        }
        let started = Instant::now();
        let transcript = match written_transcript(clip) {
            Some(t) => t,
            None => {
                let transcribe = self.transcribe.as_ref().ok_or_else(|| format!("{}: what the clip says is not written beside it ({}), and this server has no speech-to-text to hear it (run it with --mode both)", clip.display(), clip.with_extension("txt").display()))?;
                // Heard from exactly the audio the voice is made from.
                let audio = oaiy_tts::audio::read_clip(clip, &tts.ffmpeg).map_err(|e| e.to_string())?;
                let heard = transcribe(&audio, 24_000)?;
                if heard.trim().is_empty() {
                    return Err(format!("{}: no words were heard in the clip", clip.display()));
                }
                heard
            }
        };
        let mut voice = tts.voice_from_audio(clip, Some(&transcript)).map_err(|e| format!("{}: {e}", clip.display()))?;
        voice.name = clip.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        eprintln!("[oaiy-voice] voice {:?} ready in {:.2} s: {:?}", voice.name, started.elapsed().as_secs_f64(), transcript);
        self.made.insert(clip.to_path_buf(), (stamp.0, stamp.1, voice.clone()));
        Ok(voice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(test: &str, files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oaiy-voice-voices-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in files {
            std::fs::write(dir.join(name), body).unwrap();
        }
        dir
    }

    #[test]
    fn a_short_line_said_once_is_kept_and_a_changed_clip_is_a_new_voice() {
        let dir = folder("kept", &[("phone.wav", "RIFF")]);
        let (jobs, _inbox) = mpsc::channel();
        let speaker = Speaker { jobs, model: "m".into(), device: "webgpu:0".into(), voices: Voices { dir: Some(dir.clone()), default: Some("phone".into()) }, remembered: Mutex::default() };
        let okay = speaker.line(" Okay — ", None).expect("a short line");
        assert!(speaker.recall(&okay).is_none(), "nothing said yet");
        speaker.remember(okay.clone(), vec![1, 2, 3]);
        assert_eq!(speaker.recall(&speaker.line("Okay —", Some("phone")).unwrap()).as_deref(), Some(&vec![1, 2, 3]));
        // a long line is not kept, nor one in no voice there is
        assert!(speaker.line(&"word ".repeat(40), None).is_none());
        assert!(speaker.line("Okay —", Some("nobody")).is_none());
        // the clip changed: another voice, nothing kept for it
        std::fs::write(dir.join("phone.wav"), "RIFF and more").unwrap();
        assert!(speaker.recall(&speaker.line("Okay —", None).unwrap()).is_none());
        // the most recent lines are kept, the oldest go
        for i in 0..REMEMBERED + 2 {
            speaker.remember(speaker.line(&format!("line {i}"), None).unwrap(), vec![i as i16]);
        }
        assert!(speaker.recall(&speaker.line("line 0", None).unwrap()).is_none());
        assert_eq!(speaker.recall(&speaker.line(&format!("line {}", REMEMBERED + 1), None).unwrap()).as_deref(), Some(&vec![(REMEMBERED + 1) as i16]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn voices_are_clips_in_the_folder_by_name() {
        let dir = folder("by-name", &[("Phone.mp3", "x"), ("narrator.wav", "x"), ("narrator.json", r#"{"ref_text":" Hello there. "}"#), ("notes.txt", "x"), ("phone.txt", "Hi, you have reached us.")]);
        let v = Voices { dir: Some(dir.clone()), default: Some("phone".into()) };
        assert_eq!(v.list(), vec!["narrator", "Phone"]);
        assert_eq!(v.clip(None).unwrap(), dir.join("Phone.mp3"));
        assert_eq!(v.clip(Some("NARRATOR")).unwrap(), dir.join("narrator.wav"));
        assert_eq!(written_transcript(&dir.join("narrator.wav")).as_deref(), Some("Hello there."));
        assert_eq!(written_transcript(&dir.join("Phone.mp3")).as_deref(), Some("Hi, you have reached us."));
        assert!(v.clip(Some("nobody")).unwrap_err().contains("narrator, Phone"));
        for outside in ["../x", "C:/x", "a\\b", ".hidden"] {
            assert!(v.clip(Some(outside)).is_err(), "{outside}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn with_no_default_the_clip_called_default_is_used_else_the_first() {
        let dir = folder("default", &[("default.wav", "x"), ("other.wav", "x"), ("x.mp3", "x"), ("y.mp3", "x"), ("z", "x")]);
        assert_eq!(Voices { dir: Some(dir.clone()), default: None }.clip(None).unwrap(), dir.join("default.wav"));
        std::fs::remove_file(dir.join("default.wav")).unwrap();
        assert_eq!(Voices { dir: Some(dir.clone()), default: None }.clip(None).unwrap(), dir.join("other.wav"));
        assert!(Voices { dir: Some(dir.clone()), default: Some("missing".into()) }.clip(None).is_err());
        assert!(Voices::default().clip(None).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
