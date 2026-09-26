//! Songs from lyrics and a description (MiniMax Music 3): requests for the
//! worker, from the OpenAI-style APIs (`/v1/audio/music` jobs, and
//! `/v1/audio/speech` with a music model, as the official server takes it).
use crate::config;
use crate::util::{bool_or, int_or, str_or};
use nrob::json::Json;
use std::path::Path;

/// Language-model frames per second of audio.
pub const FRAMES_PER_SECOND: f64 = 25.0;
pub const DEFAULT_SECONDS: f64 = 60.0;
/// The model's limit: 9000 frames.
pub const MAX_SECONDS: f64 = 360.0;
/// Lyrics for a song without words (the model always takes lyrics).
pub const INSTRUMENTAL: &str = "[Intro]\n(instrumental)";
pub const QUANTS: [&str; 4] = ["q4_k", "q5_k", "q6_k", "q8_0"];

fn section(cfg: &Json) -> Result<&Json, String> {
    let s = cfg.get("media").and_then(|m| m.get("music")).ok_or("no music section")?;
    if !bool_or(s, "enabled", true) {
        return Err("music is disabled".into());
    }
    Ok(s)
}

/// Whether a `/v1/audio/speech` request names a music model: one of
/// `media.music.models`, or any name with "music" in it (the official
/// server's `MiniMaxAI/MiniMax-Music3`) while one is configured.
pub fn names_music_model(cfg: &Json, body: &Json) -> bool {
    let Some(asked) = body.get("model").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty()) else { return false };
    let Some(models) = cfg.get("media").and_then(|m| m.get("music")).and_then(|s| s.get("models")) else { return false };
    models.get(asked).is_some() || (asked.to_lowercase().contains("music") && models.members().next().is_some())
}

/// The music model a request names, or the default.
fn pick<'a>(section: &'a Json, body: &Json) -> Result<(String, &'a Json), String> {
    match crate::media::pick_model(section, body, "music") {
        Ok(found) => Ok(found),
        // "MiniMaxAI/MiniMax-Music3" and the like mean the default.
        Err(e) if body.get("model").and_then(Json::as_str).is_some_and(|m| m.to_lowercase().contains("music")) => {
            let mut plain = body.clone();
            crate::util::set(&mut plain, "model", Json::Null);
            crate::media::pick_model(section, &plain, "music").map_err(|_| e)
        }
        Err(e) => Err(e),
    }
}

fn text<'a>(body: &'a Json, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| body.get(k).and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty()))
}

/// Seconds asked for: `duration`, `seconds` (a number or OpenAI's string),
/// `max_seconds`, or the official server's `max_new_tokens` (frames).
fn seconds(body: &Json, model: &Json) -> Result<f64, String> {
    let number = |k: &str| -> Option<f64> {
        match body.get(k)? {
            Json::Str(s) => s.trim().parse().ok(),
            v => v.as_f64(),
        }
    };
    let s = number("duration")
        .or_else(|| number("seconds"))
        .or_else(|| number("max_seconds"))
        .or_else(|| number("max_new_tokens").map(|f| f / FRAMES_PER_SECOND))
        .or_else(|| model.get("duration").and_then(Json::as_f64))
        .unwrap_or(DEFAULT_SECONDS);
    if !(1.0..=MAX_SECONDS).contains(&s) {
        return Err(format!("duration must be 1 to {MAX_SECONDS} seconds"));
    }
    Ok(s)
}

fn seed(body: &Json) -> Result<i64, String> {
    match body.get("seed") {
        None | Some(Json::Null) => Ok((crate::util::now_millis() % 2_147_483_647) as i64),
        Some(v) => v.as_i64().filter(|n| *n >= 0).ok_or_else(|| "seed must be a nonnegative integer".into()),
    }
}

/// A song request as the worker's, with the model name, a label for the job
/// list, the most frames it may take and its length cap in seconds.
///
/// The description comes from `prompt` (or `instructions`, `description`);
/// the lyrics from `lyrics` (or `input`). `instrumental: true` without
/// lyrics asks for a song without words.
pub fn music_request(cfg: &Json, root: &Path, output_dir: &Path, body: &Json) -> Result<(Json, String, String, usize, f64), String> {
    let section = section(cfg)?;
    let (name, model) = pick(section, body)?;
    let path = str_or(model, "path", "").trim();
    if path.is_empty() {
        return Err(format!("music model {name} has no MiniMax-Music3 folder"));
    }
    let instrumental = body.get("instrumental").and_then(Json::as_bool) == Some(true);
    let mut description = text(body, &["prompt", "instructions", "description", "style"]).ok_or("prompt must describe the music (style, mood, vocals, instruments, tempo)")?.to_string();
    let lyrics = match text(body, &["lyrics", "input"]) {
        Some(l) => l.to_string(),
        None if instrumental => {
            if !description.to_lowercase().contains("instrumental") {
                description.push_str(" An instrumental piece with no vocals.");
            }
            INSTRUMENTAL.to_string()
        }
        None => return Err("lyrics are required (put structure tags such as [Verse] on their own lines), or send instrumental: true".into()),
    };
    if description.len() > 20_000 || lyrics.len() > 20_000 {
        return Err("prompt and lyrics are limited to 20000 bytes each".into());
    }
    let secs = seconds(body, model)?;
    let steps = body.get("steps").and_then(Json::as_i64).unwrap_or_else(|| int_or(model, "steps", 30));
    if !(1..=200).contains(&steps) {
        return Err("steps must be 1 to 200".into());
    }
    let guidance = body.get("guidance").and_then(Json::as_f64).or_else(|| model.get("guidance").and_then(Json::as_f64)).unwrap_or(1.7);
    if !(0.0..=20.0).contains(&guidance) {
        return Err("guidance must be 0 to 20".into());
    }
    let media = cfg.get("media").ok_or("no media section")?;
    let mut f: Vec<(String, Json)> = vec![
        ("kind".into(), Json::str("music")),
        ("model_dir".into(), Json::str(config::resolve(root, path).to_string_lossy())),
        ("prompt".into(), Json::str(&description)),
        ("lyrics".into(), Json::str(&lyrics)),
        ("seed".into(), Json::Int(seed(body)?)),
        ("device".into(), Json::Int(int_or(media, "device", 0))),
        ("output_dir".into(), Json::str(output_dir.to_string_lossy())),
        ("max_seconds".into(), Json::Num(secs)),
        ("steps".into(), Json::Int(steps)),
        ("guidance".into(), Json::Num(guidance)),
        ("precision".into(), Json::str(str_or(model, "precision", "bf16"))),
    ];
    let lm = str_or(model, "language_model", "").trim();
    if !lm.is_empty() {
        let file = config::resolve(root, lm);
        if !file.is_file() {
            return Err(format!("music model {name}: its smaller language model {} is missing", file.display()));
        }
        f.push(("language_model".into(), Json::str(file.to_string_lossy())));
    }
    f.extend(crate::media::residency(body, section, model)?);
    let label = if instrumental && text(body, &["lyrics", "input"]).is_none() { "instrumental".to_string() } else { "song".to_string() };
    Ok((Json::Obj(f), name, label, (secs * FRAMES_PER_SECOND).round() as usize, secs))
}

/// A request to make a music model's smaller language model: the worker
/// request and where it will be written.
pub fn quantize_request(cfg: &Json, root: &Path, name: &str, quant: &str) -> Result<(Json, std::path::PathBuf), String> {
    let section = cfg.get("media").and_then(|m| m.get("music")).ok_or("no music section")?;
    let model = section.get("models").and_then(|m| m.get(name)).ok_or_else(|| format!("no music model {name}"))?;
    if !QUANTS.contains(&quant) {
        return Err(format!("quant must be one of {}", QUANTS.join(", ")));
    }
    let dir = config::resolve(root, str_or(model, "path", "").trim());
    if !dir.join("language_model").is_dir() {
        return Err(format!("music model {name}: {} has no language_model folder", dir.display()));
    }
    let out = dir.join(format!("language_model-{quant}.gguf"));
    let request = Json::obj([
        ("kind", Json::str("music_quantize")),
        ("model_dir", Json::str(dir.to_string_lossy())),
        ("quant", Json::str(quant)),
        ("output", Json::str(out.to_string_lossy())),
        ("prompt", Json::str(format!("smaller language model ({quant})"))),
    ]);
    Ok((request, out))
}

/// FFmpeg arguments turning a 44.1 kHz stereo WAV into `format`. PCM stays
/// 44.1 kHz stereo (16-bit little-endian, interleaved).
pub fn convert_args(format: &str, speed: f64) -> Vec<String> {
    let mut args: Vec<String> = crate::speech::convert_args("wav", speed);
    args.truncate(args.len().saturating_sub(2));
    let codec: &[&str] = match format {
        "mp3" => &["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
        "opus" => &["-c:a", "libopus", "-b:a", "128k", "-f", "ogg"],
        "aac" => &["-c:a", "aac", "-b:a", "192k", "-f", "adts"],
        "flac" => &["-c:a", "flac", "-f", "flac"],
        "pcm" => &["-f", "s16le"],
        _ => &["-f", "wav"],
    };
    args.extend(codec.iter().map(|s| s.to_string()));
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Json {
        let mut c = config::default_json();
        let models = Json::parse(br#"{"minimax-music3":{"path":"/m/MiniMax-Music3","steps":24,"enabled":true}}"#).unwrap();
        let music = c.get("media").unwrap().get("music").unwrap().clone();
        let mut music = music;
        crate::util::set(&mut music, "models", models);
        crate::util::set(&mut music, "default_model", Json::str("minimax-music3"));
        let mut media = c.get("media").unwrap().clone();
        crate::util::set(&mut media, "music", music);
        crate::util::set(&mut c, "media", media);
        c
    }
    fn body(s: &str) -> Json {
        Json::parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn requests_take_openai_and_official_shapes() {
        let c = cfg();
        let out = Path::new("/out");
        let (r, name, label, frames, secs) = music_request(&c, Path::new("/root"), out, &body(r#"{"prompt":"lo-fi","lyrics":"[Verse]\nhi","duration":30,"seed":3}"#)).unwrap();
        assert_eq!((name.as_str(), label.as_str(), frames, secs), ("minimax-music3", "song", 750, 30.0));
        assert_eq!(str_or(&r, "kind", ""), "music");
        assert_eq!(r.get("steps").and_then(Json::as_i64), Some(24));
        assert_eq!(str_or(&r, "memory", ""), "auto");
        // The official server's speech shape: input = lyrics, instructions = style, frames.
        let (r, ..) = music_request(&c, Path::new("/root"), out, &body(r#"{"model":"MiniMaxAI/MiniMax-Music3","input":"[Chorus]\nla","instructions":"rock","max_new_tokens":250}"#)).unwrap();
        assert_eq!((str_or(&r, "lyrics", ""), str_or(&r, "prompt", "")), ("[Chorus]\nla", "rock"));
        assert_eq!(r.get("max_seconds").and_then(Json::as_f64), Some(10.0));
        // Instrumental without lyrics.
        let (r, _, label, ..) = music_request(&c, Path::new("/root"), out, &body(r#"{"prompt":"ambient pads","instrumental":true,"seconds":"8"}"#)).unwrap();
        assert_eq!((str_or(&r, "lyrics", ""), label.as_str()), (INSTRUMENTAL, "instrumental"));
        assert!(str_or(&r, "prompt", "").ends_with("no vocals."));
        assert!(music_request(&c, Path::new("/root"), out, &body(r#"{"prompt":"x"}"#)).unwrap_err().contains("lyrics"));
        assert!(music_request(&c, Path::new("/root"), out, &body(r#"{"prompt":"x","lyrics":"y","duration":400}"#)).is_err());
        assert!(names_music_model(&c, &body(r#"{"model":"minimax-music3"}"#)));
        assert!(names_music_model(&c, &body(r#"{"model":"MiniMaxAI/MiniMax-Music3"}"#)));
        assert!(!names_music_model(&c, &body(r#"{"model":"tts-1"}"#)));
    }

    #[test]
    fn music_formats_keep_stereo() {
        assert_eq!(convert_args("pcm", 1.0), vec!["-f", "s16le"]);
        assert!(convert_args("mp3", 1.0).join(" ").contains("192k"));
        assert!(convert_args("wav", 2.0).join(" ").starts_with("-filter:a atempo="));
        // No such folder here, and no such format.
        assert!(quantize_request(&cfg(), Path::new("/root"), "minimax-music3", "q4_k").unwrap_err().contains("language_model"));
        assert!(quantize_request(&cfg(), Path::new("/root"), "minimax-music3", "q3").unwrap_err().contains("quant"));
    }
}
