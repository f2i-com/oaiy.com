//! Speech: OpenAI-style text to speech (`/v1/audio/speech`) and saved voices
//! (`/v1/audio/voices`), run by the worker's Qwen3-TTS or Breeze TTS 2.
//!
//! A Qwen3-TTS speech model entry (`media.speech.models.<name>`) names the
//! VoiceDesign folder (`design`: speaks in a voice described in words) and the
//! Base folder (`base`: speaks in a saved voice). A Breeze TTS 2 entry names its
//! folder (`breeze`), which does both. A saved voice is made from a description
//! once, and is then used by name wherever a voice is asked for (a voice Qwen3-TTS
//! made, Breeze speaks too; one Breeze made, only Breeze speaks).
use crate::config;
use crate::util::{bool_or, str_or};
use nrob::json::Json;
use std::path::{Path, PathBuf};

/// OpenAI's built-in voice names, as descriptions VoiceDesign can speak in.
pub const STOCK_VOICES: [(&str, &str); 13] = [
    ("alloy", "A neutral, balanced adult voice, clear and even, with a calm conversational tone."),
    ("ash", "A warm, mature male voice, relaxed and confident."),
    ("ballad", "A gentle, expressive male voice with a soft, melodic delivery."),
    ("cedar", "A deep, grounded male voice, steady and reassuring."),
    ("coral", "A bright, friendly female voice, upbeat and approachable."),
    ("echo", "A clear, resonant male voice with a measured, steady pace."),
    ("fable", "A lively, expressive storyteller's voice with a light British accent."),
    ("marin", "A clear, warm female voice, natural and professional."),
    ("nova", "A young, energetic female voice, crisp and cheerful."),
    ("onyx", "A deep, rich male voice, calm and authoritative."),
    ("sage", "A calm, thoughtful female voice, soft-spoken and wise."),
    ("shimmer", "A soft, airy female voice, warm and soothing."),
    ("verse", "A versatile, expressive male voice with natural warmth."),
];

/// Output formats and their content types (OpenAI's list). `pcm` is raw
/// 16-bit little-endian mono at 24 kHz.
pub const FORMATS: [(&str, &str); 6] = [
    ("mp3", "audio/mpeg"),
    ("opus", "audio/ogg"),
    ("aac", "audio/aac"),
    ("flac", "audio/flac"),
    ("wav", "audio/wav"),
    ("pcm", "audio/pcm"),
];

/// Where saved voices live (`media.speech.voices_dir`, beside the studio).
pub fn voices_dir(cfg: &Json, root: &Path) -> PathBuf {
    let dir = cfg.get("media").and_then(|m| m.get("speech")).map_or("voices", |s| str_or(s, "voices_dir", "voices"));
    config::resolve(root, if dir.trim().is_empty() { "voices" } else { dir })
}

/// A voice name: letters, digits, spaces, `-` and `_`, up to 64 characters.
pub fn valid_name(name: &str) -> bool {
    let n = name.trim();
    !n.is_empty() && n.len() <= 64 && n.chars().all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_')) && !n.starts_with(['-', ' '])
}

/// The file stem for a voice: lower-case, spaces as `-`.
fn stem(name: &str) -> String {
    name.trim().to_lowercase().replace(' ', "-")
}

pub fn voice_file(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.json", stem(name)))
}

pub fn sample_file(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.wav", stem(name)))
}

/// A saved voice's public description (never the embedding or codes).
fn summary(v: &Json) -> Json {
    let name = str_or(v, "name", "");
    Json::obj([
        ("id", Json::str(name)),
        ("object", Json::str("voice")),
        ("name", Json::str(name)),
        ("description", Json::str(str_or(v, "description", ""))),
        ("language", Json::str(str_or(v, "language", "auto"))),
        ("created_at", v.get("created_at").cloned().unwrap_or(Json::Null)),
        ("sample_text", Json::str(str_or(v, "ref_text", ""))),
    ])
}

/// Every saved voice, sorted by name.
pub fn list(dir: &Path) -> Vec<Json> {
    let mut out: Vec<Json> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter_map(|p| Json::parse(&std::fs::read(&p).ok()?).ok())
        .filter(|v| v.get("nrob_voice").and_then(Json::as_i64) == Some(1))
        .map(|v| summary(&v))
        .collect();
    out.sort_by_key(|a| str_or(a, "name", "").to_lowercase());
    out
}

pub fn get(dir: &Path, name: &str) -> Option<Json> {
    if !valid_name(name) {
        return None;
    }
    let v = Json::parse(&std::fs::read(voice_file(dir, name)).ok()?).ok()?;
    Some(summary(&v))
}

pub fn remove(dir: &Path, name: &str) -> bool {
    if !valid_name(name) {
        return false;
    }
    let existed = std::fs::remove_file(voice_file(dir, name)).is_ok();
    let _ = std::fs::remove_file(sample_file(dir, name));
    existed
}

/// Keep a voice the worker made: its file (with name and time) and its sample
/// clip move into the voices folder.
pub fn keep(dir: &Path, name: &str, worker_voice: &Path, clip: &Path) -> Result<Json, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut v = Json::parse(&std::fs::read(worker_voice).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    crate::util::set(&mut v, "name", Json::str(name.trim()));
    crate::util::set(&mut v, "created_at", Json::Int(crate::util::now() as i64));
    std::fs::write(voice_file(dir, name), v.to_json()).map_err(|e| e.to_string())?;
    let _ = std::fs::copy(clip, sample_file(dir, name));
    let _ = std::fs::remove_file(worker_voice);
    let _ = std::fs::remove_file(clip);
    Ok(summary(&v))
}

/// A voice the worker made, handed back rather than kept: the voice (for
/// `voice` in later requests) with its sample clip inside it as base64 WAV
/// (`sample`). The worker's files are removed.
pub fn hand_back(name: &str, worker_voice: &Path, clip: &Path) -> Result<Json, String> {
    let mut v = Json::parse(&std::fs::read(worker_voice).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    crate::util::set(&mut v, "name", Json::str(name.trim()));
    let sample = std::fs::read(clip).map_err(|e| e.to_string())?;
    let mut out = summary(&v);
    crate::util::set(&mut out, "voice", v);
    crate::util::set(&mut out, "sample", Json::obj([("format", Json::str("wav")), ("data", Json::str(crate::util::base64_encode(&sample)))]));
    let _ = std::fs::remove_file(worker_voice);
    let _ = std::fs::remove_file(clip);
    Ok(out)
}

/// A voice sent with the request (`voice` as the object a voice design handed
/// back, with its `ref_codes` and `speaker`), rather than named.
pub fn inline_voice(body: &Json) -> Option<&Json> {
    body.get("voice").filter(|v| v.get("ref_codes").is_some() && v.get("speaker").is_some())
}

/// An inline voice's sample clip (base64 WAV), written into `dir` for the worker.
pub fn inline_sample(voice: &Json, dir: &Path) -> Result<Option<PathBuf>, String> {
    let Some(data) = voice.get("sample").and_then(|s| s.get("data")).and_then(Json::as_str) else { return Ok(None) };
    let bytes = crate::util::base64_decode(data)?;
    if bytes.len() > 16 << 20 {
        return Err("a voice sample is limited to 16 MiB".into());
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.wav", crate::util::random_id("voice_sample_")));
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    Ok(Some(path))
}

fn section(cfg: &Json) -> Result<&Json, String> {
    let s = cfg.get("media").and_then(|m| m.get("speech")).ok_or("no speech section")?;
    if !bool_or(s, "enabled", true) {
        return Err("speech is disabled".into());
    }
    Ok(s)
}

fn folder(root: &Path, model: &Json, key: &str) -> Option<String> {
    let v = str_or(model, key, "").trim();
    (!v.is_empty()).then(|| config::resolve(root, v).to_string_lossy().into_owned())
}

/// The voice a request names: `"alloy"`, `"my voice"`, or `{"id": ...}`.
fn voice_name(body: &Json) -> Option<String> {
    match body.get("voice") {
        Some(Json::Str(s)) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Some(v @ Json::Obj(_)) => v.get("id").or_else(|| v.get("name")).and_then(Json::as_str).map(|s| s.trim().to_string()),
        _ => None,
    }
}

fn language(body: &Json, fallback: &str) -> Result<String, String> {
    let l = body.get("language").and_then(Json::as_str).unwrap_or(fallback).trim().to_lowercase();
    if l.is_empty() || l.len() > 32 || !l.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err("language must be auto or a language name (english, chinese, ...)".into());
    }
    Ok(l)
}

fn seed(body: &Json) -> Result<i64, String> {
    match body.get("seed") {
        None | Some(Json::Null) => Ok((crate::util::now_millis() % 2_147_483_647) as i64),
        Some(v) => v.as_i64().filter(|n| *n >= 0).ok_or_else(|| "seed must be a nonnegative integer".into()),
    }
}

/// A speech request as the worker's, with the model name, what the voice is
/// (for the job list), and roughly how many 80 ms frames it will take.
pub fn speech_request(cfg: &Json, root: &Path, output_dir: &Path, body: &Json) -> Result<(Json, String, String, usize), String> {
    let section = section(cfg)?;
    let (name, model) = crate::media::pick_model(section, body, "speech")?;
    let input = body.get("input").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty()).ok_or("input must be the text to speak")?;
    if input.len() > 20_000 {
        return Err("input is limited to 20000 bytes".into());
    }
    let media = cfg.get("media").ok_or("no media section")?;
    let mut f: Vec<(String, Json)> = vec![
        ("kind".into(), Json::str("speech")),
        ("text".into(), Json::str(input)),
        // For the job list (the worker ignores it).
        ("prompt".into(), Json::str(input)),
        ("seed".into(), Json::Int(seed(body)?)),
        ("device".into(), Json::Int(crate::util::int_or(media, "device", 0))),
        ("output_dir".into(), Json::str(output_dir.to_string_lossy())),
    ];
    if let Some(s) = body.get("max_seconds").and_then(Json::as_f64) {
        f.push(("max_seconds".into(), Json::Num(s.clamp(1.0, 600.0))));
    }
    let asked = voice_name(body);
    let dir = voices_dir(cfg, root);
    // A voice sent inline is written beside the job's output for the worker,
    // without its sample (which only a talking video uses).
    let inline = match inline_voice(body) {
        Some(v) => {
            let mut v = v.clone();
            if let Json::Obj(fields) = &mut v {
                fields.retain(|(k, _)| k != "sample");
            }
            std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
            let path = output_dir.join(format!("{}.json", crate::util::random_id("voice_")));
            std::fs::write(&path, v.to_json()).map_err(|e| e.to_string())?;
            Some(path)
        }
        None => None,
    };
    let saved = inline.or_else(|| asked.as_deref().filter(|n| valid_name(n)).map(|n| voice_file(&dir, n)).filter(|p| p.is_file()));
    // Breeze TTS 2: one folder speaks saved and described voices.
    if let Some(breeze) = folder(root, model, "breeze") {
        f.push(("model_dir".into(), Json::str(breeze)));
        let label = match saved {
            Some(path) => {
                f.push(("voice_file".into(), Json::str(path.to_string_lossy())));
                asked.unwrap_or_default()
            }
            None => {
                let instructions = body.get("instructions").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty());
                let stock = asked.as_deref().map(str::to_lowercase).and_then(|n| STOCK_VOICES.iter().find(|(k, _)| *k == n).map(|(_, d)| *d));
                if instructions.is_none() && stock.is_none() {
                    if let Some(n) = &asked {
                        return Err(format!("unknown voice {n:?}: not a saved voice or an OpenAI voice name; describe one in `instructions`"));
                    }
                }
                let description = instructions.or(stock).unwrap_or(STOCK_VOICES[0].1);
                if description.len() > 4000 {
                    return Err("instructions are limited to 4000 bytes".into());
                }
                f.push(("instructions".into(), Json::str(description)));
                // Guidance makes Breeze follow a description closely (its authors use 4).
                let scale = body.get("cfg_scale").and_then(Json::as_f64).or_else(|| model.get("cfg_scale").and_then(Json::as_f64)).unwrap_or(4.0);
                if !(0.0..=20.0).contains(&scale) {
                    return Err("cfg_scale must be 0 to 20".into());
                }
                f.push(("cfg_scale".into(), Json::Num(scale)));
                asked.filter(|_| instructions.is_none()).unwrap_or_else(|| "described".into())
            }
        };
        let frames = (input.chars().count() as f64 / 14.0 * 12.5).ceil().max(12.0) as usize;
        return Ok((Json::Obj(f), name, label, frames));
    }
    let label = match saved {
        Some(path) => {
            // A saved voice: the Base model, prompted with the voice.
            let base = folder(root, model, "base").ok_or_else(|| format!("speech model {name} has no Base model folder, which saved voices need"))?;
            let voice = Json::parse(&std::fs::read(&path).map_err(|e| e.to_string())?).ok();
            // A voice Breeze TTS 2 made has no speaker embedding, which Qwen3-TTS needs.
            if voice.as_ref().and_then(|v| v.get("speaker")).and_then(Json::as_array).is_some_and(|s| s.is_empty()) {
                return Err(format!("this voice was made with Breeze TTS 2, and speech model {name} is Qwen3-TTS; speak it with a Breeze TTS 2 model"));
            }
            let saved_language = voice.map(|v| str_or(&v, "language", "auto").to_string()).unwrap_or_else(|| "auto".into());
            f.push(("model_dir".into(), Json::str(base)));
            f.push(("voice_file".into(), Json::str(path.to_string_lossy())));
            f.push(("language".into(), Json::str(language(body, &saved_language)?)));
            asked.unwrap_or_default()
        }
        None => {
            // A voice in words: `instructions`, or a stock voice's description.
            let design = folder(root, model, "design").ok_or_else(|| format!("speech model {name} has no VoiceDesign folder, which described voices need"))?;
            let instructions = body.get("instructions").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty());
            let stock = asked.as_deref().map(str::to_lowercase).and_then(|n| STOCK_VOICES.iter().find(|(k, _)| *k == n).map(|(_, d)| *d));
            if instructions.is_none() && stock.is_none() {
                if let Some(n) = &asked {
                    return Err(format!("unknown voice {n:?}: not a saved voice or an OpenAI voice name; describe one in `instructions`"));
                }
            }
            let description = instructions.or(stock).unwrap_or(STOCK_VOICES[0].1);
            if description.len() > 4000 {
                return Err("instructions are limited to 4000 bytes".into());
            }
            f.push(("model_dir".into(), Json::str(design)));
            f.push(("instructions".into(), Json::str(description)));
            f.push(("language".into(), Json::str(language(body, "auto")?)));
            asked.filter(|_| instructions.is_none()).unwrap_or_else(|| "described".into())
        }
    };
    // About 14 characters a second of speech, 12.5 frames a second.
    let frames = (input.chars().count() as f64 / 14.0 * 12.5).ceil().max(12.0) as usize;
    Ok((Json::Obj(f), name, label, frames))
}

/// A request to design and save a voice: `name`, `description`, and
/// optionally `sample_text` (what the voice reads to capture itself),
/// `language`, `seed`.
pub fn voice_request(cfg: &Json, root: &Path, output_dir: &Path, body: &Json) -> Result<(Json, String, String), String> {
    let section = section(cfg)?;
    let (name, model) = crate::media::pick_model(section, body, "speech")?;
    let voice = body.get("name").and_then(Json::as_str).map(str::trim).unwrap_or("");
    if !valid_name(voice) {
        return Err("name must be 1-64 letters, digits, spaces, - or _".into());
    }
    let description = body.get("description").or_else(|| body.get("instructions")).and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty()).ok_or("description must say what the voice sounds like")?;
    // Breeze TTS 2 designs and keeps a voice by itself.
    let (design, base) = match folder(root, model, "breeze") {
        Some(b) => (b.clone(), b),
        None => (
            folder(root, model, "design").ok_or_else(|| format!("speech model {name} has no VoiceDesign folder"))?,
            folder(root, model, "base").ok_or_else(|| format!("speech model {name} has no Base model folder, which saving a voice needs"))?,
        ),
    };
    let media = cfg.get("media").ok_or("no media section")?;
    let mut f: Vec<(String, Json)> = vec![
        ("kind".into(), Json::str("voice")),
        ("prompt".into(), Json::str(description)),
        ("design_model_dir".into(), Json::str(design)),
        ("base_model_dir".into(), Json::str(base)),
        ("name".into(), Json::str(voice)),
        ("description".into(), Json::str(description)),
        ("language".into(), Json::str(language(body, "auto")?)),
        ("seed".into(), Json::Int(seed(body)?)),
        ("device".into(), Json::Int(crate::util::int_or(media, "device", 0))),
        ("output_dir".into(), Json::str(output_dir.to_string_lossy())),
    ];
    if let Some(t) = body.get("sample_text").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty()) {
        f.push(("sample_text".into(), Json::str(t)));
    }
    Ok((Json::Obj(f), name, voice.to_string()))
}

/// FFmpeg arguments turning the worker's 24 kHz WAV into `format`, at `speed`
/// (OpenAI allows 0.25-4; atempo takes 0.5-2 per stage, so it is chained).
pub fn convert_args(format: &str, speed: f64) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    if (speed - 1.0).abs() > 1e-6 {
        let mut stages = Vec::new();
        let mut s = speed;
        while s > 2.0 {
            stages.push("atempo=2.0".to_string());
            s /= 2.0;
        }
        while s < 0.5 {
            stages.push("atempo=0.5".to_string());
            s /= 0.5;
        }
        stages.push(format!("atempo={s:.6}"));
        args.extend(["-filter:a".into(), stages.join(",")]);
    }
    let codec: &[&str] = match format {
        "mp3" => &["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
        "opus" => &["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
        "aac" => &["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
        "flac" => &["-c:a", "flac", "-f", "flac"],
        "pcm" => &["-f", "s16le", "-ar", "24000", "-ac", "1"],
        _ => &["-f", "wav"],
    };
    args.extend(codec.iter().map(|s| s.to_string()));
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_names_speed_and_formats() {
        assert!(valid_name("Narrator 2") && valid_name("my_voice-1"));
        assert!(!valid_name("") && !valid_name("../x") && !valid_name("a/b") && !valid_name(&"x".repeat(65)));
        assert_eq!(stem("Narrator 2"), "narrator-2");
        assert_eq!(convert_args("wav", 1.0), vec!["-f", "wav"]);
        let fast = convert_args("mp3", 4.0).join(" ");
        assert!(fast.contains("atempo=2.0,atempo=2.000000") && fast.contains("libmp3lame"));
        assert!(convert_args("pcm", 0.25).join(" ").contains("atempo=0.5,atempo=0.500000"));
        assert!(FORMATS.iter().all(|(f, _)| !convert_args(f, 1.0).is_empty()));
    }

    #[test]
    fn a_breeze_model_speaks_described_and_saved_voices_and_designs_them() {
        let dir = std::env::temp_dir().join(format!("nrob-voices-breeze-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("breeze")).unwrap();
        std::fs::create_dir_all(dir.join("voices")).unwrap();
        std::fs::write(dir.join("voices").join("chloe.json"), r#"{"nrob_voice":1,"name":"Chloe","ref_text":"hi","ref_codes":[[1]],"speaker":[]}"#).unwrap();
        let cfg = Json::parse(format!(r#"{{"media":{{"speech":{{"models":{{"breeze":{{"breeze":"{0}/breeze"}}}},"voices_dir":"{0}/voices"}}}}}}"#, dir.to_string_lossy().replace('\\', "/")).as_bytes()).unwrap();
        let out = dir.join("out");
        // Described: the Breeze folder, its instruction, guided at 4 unless asked otherwise.
        let body = Json::obj([("input", Json::str("Hello.")), ("instructions", Json::str("a sly old fox"))]);
        let (request, name, label, _) = speech_request(&cfg, &dir, &out, &body).unwrap();
        assert_eq!((name.as_str(), label.as_str()), ("breeze", "described"));
        assert!(str_or(&request, "model_dir", "").ends_with("breeze"));
        assert_eq!(str_or(&request, "instructions", ""), "a sly old fox");
        assert_eq!(request.get("cfg_scale").and_then(Json::as_f64), Some(4.0));
        // A saved voice by name.
        let body = Json::obj([("input", Json::str("Hello.")), ("voice", Json::str("Chloe"))]);
        let (request, _, label, _) = speech_request(&cfg, &dir, &out, &body).unwrap();
        assert_eq!(label, "Chloe");
        assert!(str_or(&request, "voice_file", "").ends_with("chloe.json"));
        assert!(request.get("cfg_scale").is_none());
        // Qwen3-TTS can't speak a voice Breeze made (it has no speaker embedding).
        let qwen = Json::parse(format!(r#"{{"media":{{"speech":{{"models":{{"qwen":{{"design":"{0}/breeze","base":"{0}/breeze"}}}},"voices_dir":"{0}/voices"}}}}}}"#, dir.to_string_lossy().replace('\\', "/")).as_bytes()).unwrap();
        let body = Json::obj([("input", Json::str("Hello.")), ("voice", Json::str("Chloe"))]);
        assert!(speech_request(&qwen, &dir, &out, &body).unwrap_err().contains("Breeze"));
        // Designing a voice: Breeze on its own.
        let body = Json::obj([("name", Json::str("Fox")), ("description", Json::str("a sly old fox"))]);
        let (request, _, _) = voice_request(&cfg, &dir, &out, &body).unwrap();
        assert!(str_or(&request, "design_model_dir", "").ends_with("breeze") && str_or(&request, "base_model_dir", "").ends_with("breeze"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn voices_are_kept_listed_and_removed() {
        let dir = std::env::temp_dir().join(format!("nrob-voices-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let worker = dir.join("w.voice.json");
        std::fs::write(&worker, r#"{"nrob_voice":1,"name":"x","description":"bright","language":"english","ref_text":"hi","ref_codes":[],"speaker":[]}"#).unwrap();
        let clip = dir.join("w.wav");
        std::fs::write(&clip, b"RIFF").unwrap();
        let kept = keep(&dir, "Narrator 2", &worker, &clip).unwrap();
        assert_eq!(str_or(&kept, "name", ""), "Narrator 2");
        assert!(voice_file(&dir, "narrator 2").is_file() && sample_file(&dir, "Narrator 2").is_file());
        assert_eq!(list(&dir).len(), 1);
        assert!(get(&dir, "Narrator 2").is_some());
        assert!(remove(&dir, "Narrator 2"));
        assert!(list(&dir).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_voice_handed_back_is_spoken_inline_and_nothing_is_kept() {
        let dir = std::env::temp_dir().join(format!("nrob-voices-inline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("base")).unwrap();
        std::fs::create_dir_all(dir.join("design")).unwrap();
        let worker = dir.join("w.voice.json");
        std::fs::write(&worker, r#"{"nrob_voice":1,"name":"x","language":"english","ref_text":"hi","ref_codes":[[1,2]],"speaker":[0.5]}"#).unwrap();
        let clip = dir.join("w.wav");
        std::fs::write(&clip, b"RIFFdata").unwrap();
        // Handed back: the voice with its sample inside, and the worker's files gone.
        let handed = hand_back("Gary", &worker, &clip).unwrap();
        assert!(!worker.exists() && !clip.exists() && list(&dir).is_empty());
        let voice = handed.get("voice").unwrap().clone();
        assert_eq!(str_or(&voice, "name", ""), "Gary");
        assert_eq!(handed.get("sample").and_then(|s| s.get("data")).and_then(Json::as_str), Some(crate::util::base64_encode(b"RIFFdata").as_str()));
        // Spoken inline: the Base model, prompted with the voice written for the worker (without its sample).
        let cfg = Json::parse(format!(r#"{{"media":{{"speech":{{"models":{{"tts":{{"base":"{0}/base","design":"{0}/design"}}}},"voices_dir":"{0}/voices"}}}}}}"#, dir.to_string_lossy().replace('\\', "/")).as_bytes()).unwrap();
        let mut with_sample = voice.clone();
        crate::util::set(&mut with_sample, "sample", handed.get("sample").unwrap().clone());
        let out = dir.join("out");
        let body = Json::obj([("input", Json::str("We are closing.")), ("voice", with_sample.clone())]);
        let (request, _, label, _) = speech_request(&cfg, &dir, &out, &body).unwrap();
        assert!(str_or(&request, "model_dir", "").ends_with("base"));
        let written = Json::parse(&std::fs::read(str_or(&request, "voice_file", "")).unwrap()).unwrap();
        assert!(written.get("ref_codes").is_some() && written.get("sample").is_none());
        assert_eq!(label, "Gary");
        // Its sample is a talking video's reference voice.
        let sample = inline_sample(&with_sample, &out).unwrap().unwrap();
        assert_eq!(std::fs::read(sample).unwrap(), b"RIFFdata");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
