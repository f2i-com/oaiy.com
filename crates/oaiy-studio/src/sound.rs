//! Sound effects from a description (MOSS-SoundEffect v2.0): requests for the
//! worker, from `/v1/audio/sound_effects` jobs.
use crate::config;
use crate::util::{bool_or, int_or, str_or};
use oaiy_engine::json::Json;
use std::path::Path;

pub const DEFAULT_SECONDS: f64 = 10.0;
/// The model's limit (it always works on a 30-second window).
pub const MAX_SECONDS: f64 = 30.0;
pub const SAMPLE_RATE: i64 = 48_000;

fn section(cfg: &Json) -> Result<&Json, String> {
    let s = cfg.get("media").and_then(|m| m.get("sound")).ok_or("no sound section")?;
    if !bool_or(s, "enabled", true) {
        return Err("sound effects are disabled".into());
    }
    Ok(s)
}

fn number(body: &Json, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| match body.get(k)? {
        Json::Str(s) => s.trim().parse().ok(),
        v => v.as_f64(),
    })
}

/// A sound effect request as the worker's, with the model name, a label for the
/// job list and its length in seconds.
///
/// The description comes from `prompt` (or `input`, `text`, `description`); the
/// length from `seconds` (or `duration`, `duration_seconds`).
pub fn sound_request(cfg: &Json, root: &Path, output_dir: &Path, body: &Json) -> Result<(Json, String, String, f64), String> {
    let section = section(cfg)?;
    let (name, model) = crate::media::pick_model(section, body, "sound")?;
    let path = str_or(model, "path", "").trim();
    if path.is_empty() {
        return Err(format!("sound model {name} has no MOSS-SoundEffect folder"));
    }
    let prompt = ["prompt", "input", "text", "description"]
        .iter()
        .find_map(|k| body.get(k).and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty()))
        .ok_or("prompt must describe the sound (what makes it, where, how it sounds)")?;
    if prompt.len() > 4000 {
        return Err("the prompt is limited to 4000 bytes".into());
    }
    let seconds = number(body, &["seconds", "duration", "duration_seconds"]).or_else(|| model.get("seconds").and_then(Json::as_f64)).unwrap_or(DEFAULT_SECONDS);
    if !(0.5..=MAX_SECONDS).contains(&seconds) {
        return Err(format!("seconds must be 0.5 to {MAX_SECONDS}"));
    }
    let steps = body.get("steps").and_then(Json::as_i64).unwrap_or_else(|| int_or(model, "steps", 100));
    if !(1..=200).contains(&steps) {
        return Err("steps must be 1 to 200".into());
    }
    let cfg_scale = number(body, &["cfg_scale", "guidance"]).or_else(|| model.get("cfg_scale").and_then(Json::as_f64)).unwrap_or(4.0);
    if !(0.0..=20.0).contains(&cfg_scale) {
        return Err("cfg_scale must be 0 to 20".into());
    }
    let seed = match body.get("seed") {
        None | Some(Json::Null) => (crate::util::now_millis() % 2_147_483_647) as i64,
        Some(v) => v.as_i64().filter(|n| *n >= 0).ok_or("seed must be a nonnegative integer")?,
    };
    let negative = body.get("negative_prompt").and_then(Json::as_str).unwrap_or("").trim();
    let media = cfg.get("media").ok_or("no media section")?;
    let request = Json::Obj(vec![
        ("kind".into(), Json::str("sound")),
        ("model_dir".into(), Json::str(config::resolve(root, path).to_string_lossy())),
        ("prompt".into(), Json::str(prompt)),
        ("negative_prompt".into(), Json::str(negative)),
        ("seconds".into(), Json::Num(seconds)),
        ("steps".into(), Json::Int(steps)),
        ("cfg_scale".into(), Json::Num(cfg_scale)),
        ("seed".into(), Json::Int(seed)),
        ("device".into(), Json::Int(int_or(media, "device", 0))),
        ("output_dir".into(), Json::str(output_dir.to_string_lossy())),
    ]);
    Ok((request, name, format!("{seconds:.1} s"), seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Json {
        let mut c = config::default_json();
        let Json::Obj(top) = &mut c else { unreachable!() };
        let (_, media) = top.iter_mut().find(|(k, _)| k == "media").unwrap();
        let Json::Obj(media) = media else { unreachable!() };
        let (_, sound) = media.iter_mut().find(|(k, _)| k == "sound").unwrap();
        crate::util::set(sound, "models", Json::obj([("moss", Json::obj([("path", Json::str("models/moss"))]))]));
        c
    }

    #[test]
    fn a_description_and_a_length_make_a_worker_request() {
        let body = Json::parse(br#"{"prompt":"  rain on a tin roof ","seconds":"4.5","seed":7}"#).unwrap();
        let (r, name, label, secs) = sound_request(&cfg(), Path::new("/root"), Path::new("/out"), &body).unwrap();
        assert_eq!((name.as_str(), label.as_str(), secs), ("moss", "4.5 s", 4.5));
        assert_eq!(str_or(&r, "kind", ""), "sound");
        assert_eq!(str_or(&r, "prompt", ""), "rain on a tin roof");
        assert_eq!((int_or(&r, "steps", 0), int_or(&r, "seed", 0)), (100, 7));
        assert!(str_or(&r, "model_dir", "").ends_with("moss"));
    }

    #[test]
    fn bad_requests_are_refused() {
        for body in [r#"{}"#, r#"{"prompt":"x","seconds":31}"#, r#"{"prompt":"x","steps":0}"#, r#"{"prompt":"x","seed":-1}"#] {
            assert!(sound_request(&cfg(), Path::new("/"), Path::new("/o"), &Json::parse(body.as_bytes()).unwrap()).is_err(), "{body}");
        }
    }
}
