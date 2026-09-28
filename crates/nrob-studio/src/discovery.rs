//! What this studio offers, as one JSON document for companion apps:
//! `GET /v1/discovery` (a configurable route) and `GET /.well-known/nrob.json`
//! (fixed, so an app can probe any host:port).
//!
//! It lists every enabled route with its full URL, method, dialect and the
//! paths below it, the models each serves (with defaults and capabilities), and
//! how to authenticate. Without the API key (when one is set) it says only that
//! a key is needed, so model names do not leak to whoever can reach the port.

use crate::util::{bool_or, int_or, str_or};
use crate::Studio;
use nrob::json::Json;

fn sub(method: &str, path: String, what: &str) -> Json {
    Json::obj([("method", Json::str(method)), ("path", Json::str(path)), ("description", Json::str(what))])
}

/// The operations a route serves, beyond its primary path.
fn operations(target: &str, spec: &str, path: &str) -> Vec<Json> {
    match (target, spec) {
        ("videos", "openai") => vec![
            sub("POST", path.into(), "create a video job (JSON: prompt, model, seconds, size, input_reference (start frame), end_image, speech {text, voice}: on a model with lip_sync the speech is generated with the picture in that voice, lips in sync; input_audio (with its transcript) is kept exactly and followed (soundtrack_mode inpaint or frozen; lip_sync: voice to speak it again in that voice instead, off to follow speech as a soundtrack))"),
            sub("GET", path.into(), "list video jobs"),
            sub("GET", format!("{path}/{{id}}"), "poll a job: status queued | in_progress | completed | failed, progress 0-100"),
            sub("GET", format!("{path}/{{id}}/content"), "download the MP4; ?variant=thumbnail for its first frame (PNG)"),
            sub("DELETE", format!("{path}/{{id}}"), "forget a job (files stay on disk)"),
        ],
        ("images" | "videos", "nrob") => vec![
            sub("POST", path.into(), "queue a job: 202 {id, status_url}"),
            sub("GET", format!("{path}/status"), "the latest job, or ?id="),
            sub("POST", format!("{path}/cancel"), "cancel the running job"),
        ],
        ("files", _) => vec![sub("GET", format!("{path}/{{file}}"), "a generated image or video (URLs come back from image and video replies)")],
        ("speech", _) => vec![sub(
            "POST",
            path.into(),
            "JSON: input, voice (a saved voice's name, or an OpenAI voice name), instructions (describe any voice), response_format mp3|opus|aac|flac|wav|pcm, speed 0.25-4, language, seed; returns the audio. With a music model: input is the lyrics and instructions the style (a whole song comes back)",
        )],
        ("music", _) => vec![
            sub("POST", path.into(), "create a song job (JSON: prompt (the style), lyrics or instrumental: true, duration 1-360 s, seed, steps)"),
            sub("GET", path.into(), "list song jobs"),
            sub("GET", format!("{path}/{{id}}"), "poll a job: status queued | in_progress | completed | failed, progress 0-100"),
            sub("GET", format!("{path}/{{id}}/content"), "download the song: 44.1 kHz stereo WAV, or ?format=mp3|opus|aac|flac|pcm"),
            sub("POST", format!("{path}/{{id}}/cancel"), "stop a job"),
            sub("DELETE", format!("{path}/{{id}}"), "forget a job (files stay on disk)"),
        ],
        ("sound", _) => vec![
            sub("POST", path.into(), "create a sound effect job (JSON: prompt (describe the sound), seconds 0.5-30 (default 10), seed, steps, cfg_scale)"),
            sub("GET", path.into(), "list sound effect jobs"),
            sub("GET", format!("{path}/{{id}}"), "poll a job: status queued | in_progress | completed | failed, progress 0-100"),
            sub("GET", format!("{path}/{{id}}/content"), "download the sound: 48 kHz mono WAV, or ?format=mp3|opus|aac|flac|pcm"),
            sub("POST", format!("{path}/{{id}}/cancel"), "stop a job"),
            sub("DELETE", format!("{path}/{{id}}"), "forget a job (files stay on disk)"),
        ],
        ("voices", _) => vec![
            sub("GET", path.into(), "list saved voices"),
            sub("POST", path.into(), "design and save a voice (JSON: name, description, sample_text?, language?, seed?)"),
            sub("GET", format!("{path}/{{name}}"), "one voice"),
            sub("GET", format!("{path}/{{name}}/sample"), "the voice's sample clip (WAV)"),
            sub("DELETE", format!("{path}/{{name}}"), "delete a voice"),
        ],
        _ => Vec::new(),
    }
}

fn describe(target: &str) -> &'static str {
    match target {
        "chat" => "chat completions, streamed as server-sent events with stream: true",
        "completions" => "raw text completions",
        "models" => "model list; each entry has type llm | image | video | speech | music | sound",
        "images" => "text-to-image; add images: [{image_url}] to edit",
        "edits" => "image edits: multipart (image, prompt) or JSON with images",
        "videos" => "text/image-to-video",
        "speech" => "text to speech in a saved or described voice (OpenAI audio.speech)",
        "voices" => "saved voices: design one from a description, then speak in it by name",
        "music" => "songs with vocals and instruments from lyrics and a description, as asynchronous jobs",
        "sound" => "sound effects (ambience, creatures, machines, actions) from a description, up to 30 seconds, as asynchronous jobs",
        "files" => "generated media",
        "health" => "liveness, no key needed",
        "discovery" => "this document",
        _ => "",
    }
}

fn names(section: Option<&Json>) -> Vec<(String, Json)> {
    section
        .and_then(|s| s.get("models"))
        .map(|m| m.members().filter(|(_, v)| bool_or(v, "enabled", true)).map(|(k, v)| (k.to_string(), v.clone())).collect())
        .unwrap_or_default()
}

/// The document. `authorized`: the caller may see models and routes.
pub fn document(studio: &Studio, base: &str, authorized: bool) -> Json {
    let cfg = studio.config();
    let gateway = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let key_required = !str_or(&gateway, "api_key", "").is_empty();
    let base = base.trim_end_matches('/');
    let mut doc = vec![
        ("service".to_string(), Json::str("nrob-studio")),
        ("version".into(), Json::str(env!("CARGO_PKG_VERSION"))),
        ("base_url".into(), Json::str(base)),
        ("openai_base_url".into(), Json::str(format!("{base}/v1"))),
        (
            "auth".into(),
            Json::obj([
                ("type", Json::str(if key_required { "bearer" } else { "none" })),
                ("required", Json::Bool(key_required)),
                ("header", Json::str("Authorization: Bearer <key>")),
            ]),
        ),
    ];
    if key_required && !authorized {
        doc.push(("note".into(), Json::str("send the API key to see the endpoints and models")));
        return Json::Obj(doc);
    }

    let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
    let media = cfg.get("media").cloned().unwrap_or(Json::Null);
    let status = studio.llm.status();
    let loaded: Vec<String> = if status.get("state").and_then(Json::as_str) == Some("ready") {
        status.get("models").and_then(Json::as_array).unwrap_or(&[]).iter().take(1).filter_map(Json::as_str).map(String::from).collect()
    } else {
        Vec::new()
    };
    let llm_default = {
        let d = str_or(&llm, "default_model", "");
        let first = llm.get("models").and_then(Json::as_array).and_then(|m| m.iter().find(|m| bool_or(m, "enabled", true))).map(|m| str_or(m, "name", "").to_string());
        if d.is_empty() { first.unwrap_or_default() } else { d.to_string() }
    };
    let llm_models: Vec<Json> = llm
        .get("models")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter(|m| bool_or(m, "enabled", true))
        .map(|m| {
            let name = str_or(m, "name", "");
            Json::obj([
                ("id", Json::str(name)),
                ("default", Json::Bool(name == llm_default)),
                ("loaded", Json::Bool(loaded.iter().any(|l| l == name))),
                ("vision", Json::Bool(!str_or(m, "vision_projector", "").is_empty())),
            ])
        })
        .collect();
    let section = |kind: &str| media.get(kind).filter(|s| bool_or(s, "enabled", true));
    let image = section("image");
    // What a request without `model` gets: the default, else the first enabled.
    let effective = |section: Option<&Json>| {
        let d = section.map_or("", |s| str_or(s, "default_model", "")).to_string();
        if d.is_empty() { names(section).first().map(|(n, _)| n.clone()).unwrap_or_default() } else { d }
    };
    let image_default = effective(image);
    let image_models: Vec<Json> = names(image)
        .into_iter()
        .map(|(name, m)| {
            let sdxl = str_or(&m, "architecture", "qwen-image") == "sdxl";
            Json::obj([
                ("id", Json::str(&name)),
                ("default", Json::Bool(name == image_default)),
                ("architecture", Json::str(if sdxl { "sdxl" } else { "qwen-image" })),
                ("edits", Json::Bool(!sdxl)),
                ("max_references", Json::Int(if sdxl { 0 } else { 3 })),
                ("size_step", Json::Int(if sdxl { 64 } else { 32 })),
                ("default_size", Json::str(format!("{}x{}", int_or(&m, "width", 1024), int_or(&m, "height", 1024)))),
                ("negative_prompt", Json::Bool(sdxl)),
            ])
        })
        .collect();
    let video = section("video");
    let video_default = effective(video);
    let fps = video.map_or(24, |s| int_or(s, "fps", 24));
    let video_models: Vec<Json> = names(video)
        .into_iter()
        .map(|(name, m)| {
            Json::obj([
                ("id", Json::str(&name)),
                ("default", Json::Bool(name == video_default)),
                ("family", Json::str(str_or(&m, "family", &name))),
                ("fps", Json::Int(fps)),
                ("max_frames", Json::Int(121)),
                ("max_seconds", Json::Num(120.0 / fps as f64)),
                ("max_side", Json::Int(1024)),
                ("start_image", Json::Bool(true)),
                // Speech generated with the picture, lips in sync, in a saved voice (ID-LoRA).
                ("lip_sync", Json::Bool(!str_or(&m, "id_lora", "").trim().is_empty())),
            ])
        })
        .collect();
    let speech = section("speech");
    let speech_default = effective(speech);
    let voices = crate::speech::list(&crate::speech::voices_dir(&cfg, &studio.root));
    let speech_models: Vec<Json> = names(speech)
        .into_iter()
        .map(|(name, m)| {
            let breeze = !str_or(&m, "breeze", "").is_empty();
            Json::obj([
                ("id", Json::str(&name)),
                ("default", Json::Bool(name == speech_default)),
                ("engine", Json::str(if breeze { "breeze-tts-2" } else { "qwen3-tts" })),
                ("described_voices", Json::Bool(breeze || !str_or(&m, "design", "").is_empty())),
                ("saved_voices", Json::Bool(breeze || !str_or(&m, "base", "").is_empty())),
                // Breeze TTS 2's weights and what they make are for research and non-commercial use.
                ("license", Json::str(if breeze { "research and non-commercial" } else { "apache-2.0" })),
                ("sample_rate", Json::Int(24_000)),
            ])
        })
        .collect();
    let music = section("music");
    let music_default = effective(music);
    let music_models: Vec<Json> = names(music)
        .into_iter()
        .map(|(name, m)| {
            Json::obj([
                ("id", Json::str(&name)),
                ("default", Json::Bool(name == music_default)),
                ("max_seconds", Json::Num(crate::music::MAX_SECONDS)),
                ("sample_rate", Json::Int(44_100)),
                ("channels", Json::Int(2)),
                ("quantized", Json::Bool(!str_or(&m, "language_model", "").is_empty())),
            ])
        })
        .collect();
    let sound = section("sound");
    let sound_default = effective(sound);
    let sound_models: Vec<Json> = names(sound)
        .into_iter()
        .map(|(name, _)| {
            Json::obj([
                ("id", Json::str(&name)),
                ("default", Json::Bool(name == sound_default)),
                ("max_seconds", Json::Num(crate::sound::MAX_SECONDS)),
                ("sample_rate", Json::Int(crate::sound::SAMPLE_RATE)),
                ("channels", Json::Int(1)),
            ])
        })
        .collect();
    let models_for = |target: &str| -> Vec<Json> {
        let ids = |list: &[Json]| list.iter().filter_map(|m| m.get("id").cloned()).collect::<Vec<_>>();
        match target {
            "chat" | "completions" => ids(&llm_models),
            "images" => ids(&image_models),
            "edits" => image_models.iter().filter(|m| m.get("edits") == Some(&Json::Bool(true))).filter_map(|m| m.get("id").cloned()).collect(),
            "videos" => ids(&video_models),
            // The speech endpoint also sings, given a music model.
            "speech" => ids(&speech_models).into_iter().chain(ids(&music_models)).collect(),
            "voices" => ids(&speech_models),
            "music" => ids(&music_models),
            "sound" => ids(&sound_models),
            _ => Vec::new(),
        }
    };

    let routes = gateway.get("routes").and_then(Json::as_array).unwrap_or(&[]);
    let endpoints: Vec<Json> = routes
        .iter()
        .filter(|r| bool_or(r, "enabled", true))
        .map(|r| {
            let (path, target, spec) = (str_or(r, "path", ""), str_or(r, "target", ""), str_or(r, "spec", "openai"));
            let method = if target == "videos" && spec == "openai" { "POST" } else { str_or(r, "method", "POST") };
            let mut e = vec![
                ("name".to_string(), Json::str(target)),
                ("method".into(), Json::str(method.to_ascii_uppercase())),
                ("path".into(), Json::str(path)),
                ("url".into(), Json::str(format!("{base}{path}"))),
                ("spec".into(), Json::str(spec)),
                ("description".into(), Json::str(describe(target))),
            ];
            if target == "chat" {
                e.push(("streaming".into(), Json::Bool(true)));
            }
            let ops = operations(target, spec, path);
            if !ops.is_empty() {
                e.push(("operations".into(), Json::Arr(ops)));
            }
            let models = models_for(target);
            if target == "speech" || target == "voices" {
                e.push(("voices".into(), Json::Arr(voices.iter().filter_map(|v| v.get("name").cloned()).collect())));
            }
            if !models.is_empty() || matches!(target, "chat" | "completions" | "images" | "edits" | "videos" | "speech" | "music" | "sound") {
                e.push(("models".into(), Json::Arr(models)));
            }
            Json::Obj(e)
        })
        .collect();

    doc.extend([
        ("endpoints".into(), Json::Arr(endpoints)),
        (
            "models".into(),
            Json::obj([("llm", Json::Arr(llm_models)), ("image", Json::Arr(image_models)), ("video", Json::Arr(video_models)), ("speech", Json::Arr(speech_models)), ("music", Json::Arr(music_models)), ("sound", Json::Arr(sound_models))]),
        ),
        (
            "defaults".into(),
            Json::obj([
                ("llm", Json::str(&llm_default)),
                ("image", Json::str(&image_default)),
                ("video", Json::str(&video_default)),
                ("speech", Json::str(&speech_default)),
                ("music", Json::str(&music_default)),
                ("sound", Json::str(&sound_default)),
            ]),
        ),
        (
            "voices".into(),
            Json::obj([
                ("saved", Json::Arr(voices.clone())),
                ("openai_names", Json::Arr(crate::speech::STOCK_VOICES.iter().map(|(n, _)| Json::str(*n)).collect())),
            ]),
        ),
        (
            "llm".into(),
            Json::obj([
                ("state", status.get("state").cloned().unwrap_or(Json::Null)),
                ("runs_on", status.get("runs_on").cloned().unwrap_or(Json::Null)),
                // What the loaded model was opened with; before it loads, the setting
                // (0, the model's maximum, is not known until then).
                ("context_tokens", status.get("context_tokens").filter(|c| !matches!(c, Json::Null)).cloned()
                    .unwrap_or_else(|| match int_or(&llm, "ctx", 0) { 0 => Json::Null, n => Json::Int(n) })),
                ("starts_on_demand", Json::Bool(true)),
            ]),
        ),
        ("media".into(), Json::obj([("busy", Json::Bool(studio.media.busy()))])),
        (
            "incognito".into(),
            Json::obj([
                ("always", Json::Bool(cfg.get("privacy").is_some_and(|p| bool_or(p, "incognito", false)))),
                ("per_request_header", Json::str("X-NROB-Incognito: 1")),
                ("per_request_body_field", Json::str("incognito")),
                ("keeps", Json::str("nothing: no prompt cache, no job history or logs; media files are deleted once returned inline, or after 10 minutes (images) / 30 minutes (videos) when fetched by URL")),
            ]),
        ),
    ]);
    Json::Obj(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_and_job_routes_list_their_operations() {
        let ops = operations("videos", "openai", "/v1/videos");
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[3].get("path").and_then(Json::as_str), Some("/v1/videos/{id}/content"));
        assert_eq!(operations("images", "nrob", "/nrob/images")[1].get("path").and_then(Json::as_str), Some("/nrob/images/status"));
        assert!(operations("chat", "openai", "/v1/chat/completions").is_empty());
    }
}
