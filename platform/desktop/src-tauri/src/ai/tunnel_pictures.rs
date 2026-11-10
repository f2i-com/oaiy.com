//! The `image` kind of the sealed AI lane: a picture a provider's site asks this computer for (FormLogic's App AI: a
//! logo an app on the site makes for someone), made by OAIY's own picture engine and nothing else.
//!
//! The request, inside the sealed envelope (the field names are the provider's contract, and binding):
//!
//! ```text
//! { "v": 1, "kind": "image", "requestId": "...",
//!   "model": "<an image model of OAIY Engines, named: there is no default>",
//!   "prompt": "<at most 4,000 characters>", "size": "1024x1024", "seed": 12345 | null,
//!   "reference": "data:image/jpeg;base64,..." | null, "incognito": true }
//! ```
//!
//! is answered with frames of the lane's own kinds:
//!
//! ```text
//! { "v": 1, "type": "progress", "kind": "progress", "progress": "drawing" }   about every 15 s while it is drawn
//! { "v": 1, "type": "final", "kind": "final", "picture": { "mimeType", "data", "width", "height", "seed", "model" } }
//! { "v": 1, "type": "final", "kind": "final", "error": { "code", "message" } }
//! ```
//!
//! The last frame is always a `final` one, as a chat turn's is, with the picture or why there is none (a code of
//! [`CODES`]). The progress frames say the picture is still being drawn: the site lets a picture it has heard nothing
//! of go after five minutes, and can tell a slow picture from a dead one sooner. The picture is a PNG or a baseline
//! JPEG, never anything else: the site's apps read those two.
//!
//! Three rules shape it:
//!
//! 1. **The engines' picture route and nothing else.** The request goes to the engines' OpenAI images route, as their
//!    discovery names it, with the fields [`make`] builds and no others: never a tool, a flow or a provider of the
//!    store, whatever the request carries. One that carries a tool grant or a flow is refused rather than run without
//!    them, so a site that sends one learns it was not used.
//! 2. **No default model.** The engines answer a request without a model, and one naming OpenAI's own models
//!    (`dall-e-3`), with their default. The site names its model, and gets that one or an error, never another.
//! 3. **Incognito, always.** The engines keep nothing of the request or the picture: the job runs in their incognito
//!    folder, which goes with it. Their answer then carries no record of the seed, so the seed is chosen here when the
//!    request leaves it to this computer, and the reply says which.
//!
//! A picture is made in a slot of its own beside the lane's chat turns (`tunnel::AiTunnel::start_picture`), one at a
//! time on a computer, so a chat turn never waits for a picture.

use std::io::Cursor;
use std::time::Duration;

use base64::Engine as _;
use serde_json::{json, Value};

/// The request kind on the lane's row and in the sealed body.
pub const KIND: &str = "image";

/// The engines know no such model as the one named, or none at all.
pub const MODEL_UNKNOWN: &str = "model_unknown";
/// The model named is set up in Engines and out of use: switched off (or pictures are), or its files not downloaded.
pub const MODEL_DISABLED: &str = "model_disabled";
/// The engines are making as many pictures as they will queue: try again later.
pub const BUSY: &str = "busy";
/// The request itself is wrong: a field missing or out of range, a reference that is not a picture.
pub const BAD_REQUEST: &str = "bad_request";
/// The reference is larger than the lane takes, or the picture made cannot be fitted into one frame.
pub const TOO_LARGE: &str = "too_large";
/// The engines are not running, or failed making the picture.
pub const ENGINE_FAILED: &str = "engine_failed";
/// The picture was not made within [`PICTURE_TIMEOUT`].
pub const TIMEOUT: &str = "timeout";
/// Every code a picture's error frame carries: the site maps each to a message and what to do.
pub const CODES: [&str; 7] = [MODEL_DISABLED, MODEL_UNKNOWN, BUSY, BAD_REQUEST, TOO_LARGE, ENGINE_FAILED, TIMEOUT];

/// The sizes a site may ask for: square and the two 4:3 shapes, at sizes every architecture the engines run takes
/// (multiples of 64), which keeps a picture's cost to the computer within what its owner agreed to lend.
pub const SIZES: [&str; 5] = ["512x512", "768x768", "1024x1024", "1024x768", "768x1024"];

/// The longest prompt, in characters (not bytes: a prompt in Greek is as long as one in English). The engines take
/// 16 KiB, which 4,000 characters of four bytes each still fit.
pub const MAX_PROMPT_CHARS: usize = 4_000;

/// The longest model name, as the lane takes one for chat.
const MAX_MODEL_CHARS: usize = 256;

/// The largest seed a site may name: the largest whole number JavaScript holds exactly, so the seed the reply reports
/// is the one the site reads. (The engines take up to 2^62.)
pub const MAX_SEED: u64 = (1 << 53) - 1;

/// The largest reference picture. The whole sealed request is at most 256 KiB (the provider's envelope limit), so a
/// larger one never arrives; said here too, so that it is this lane's own limit and answered as `too_large`.
pub const MAX_REFERENCE_BYTES: usize = 256 * 1024;

/// The longest side a reference picture may have: the engines scale one to about a megapixel, and a larger header is
/// more likely a mistake (or a picture made to take memory) than a logo to edit.
const MAX_REFERENCE_SIDE: u32 = 8192;

/// How long a picture may take, from the request to the engines to their answer. A picture of FLUX 3 Klein takes 20
/// to 90 seconds on a card that has its model loaded, and loading it takes a minute more. The provider reaps a
/// claimed request it has heard nothing of for five minutes, so the answer, `timeout` if it must be, comes before
/// that. When this passes, the request to the engines is dropped, and the engines stop the job of a caller that has
/// gone.
pub const PICTURE_TIMEOUT: Duration = Duration::from_secs(240);

/// How often a picture being drawn says so ([`progress_frame`]): well inside the five minutes after which the site
/// lets a silent picture go, and often enough for it to tell a slow picture from a dead one long before that. The
/// site's answer to one is also where it says the picture was cancelled (contract §0.13), so this is how soon a
/// cancelled picture stops being drawn: five seconds at most (it was fifteen), twelve small requests a minute.
pub const PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// The most of the engines' answer read: one picture, as base64, with room to spare.
const MAX_ENGINE_REPLY_BYTES: u64 = 32 * 1024 * 1024;

/// The most a frame of the lane carries, before it is sealed.
const MAX_FRAME_BYTES: usize = crate::ai::e2e::MAX_ENVELOPE_PLAINTEXT_BYTES;

/// What is kept free of a frame for the reckoning of its size to be wrong by.
const FRAME_SLACK: usize = 64;

/// A refused or failed picture: one of [`CODES`] and a short message in plain English for the site to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureError {
    code: &'static str,
    message: String,
}

impl PictureError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

fn bad(message: impl Into<String>) -> PictureError {
    PictureError::new(BAD_REQUEST, message)
}

fn failed(message: impl Into<String>) -> PictureError {
    PictureError::new(ENGINE_FAILED, message)
}

/// A picture request as the site sent it, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureRequest {
    /// The image model of OAIY Engines that makes it, by its id there.
    pub model: String,
    pub prompt: String,
    /// One of [`SIZES`].
    pub size: &'static str,
    /// The seed the site named; none leaves it to this computer, which picks one and says which.
    pub seed: Option<u64>,
    /// The picture to edit, as a data URL whose type is what its bytes are, whatever the site called them.
    pub reference: Option<String>,
}

impl PictureRequest {
    /// The request in `body`, the sealed JSON, whose row named `row_provider`, or why it cannot be made.
    ///
    /// The row's `kind` and `providerId` are plaintext a backend could rewrite: the sealed kind must be this one, and a
    /// sealed provider, where there is one, the row's. (A picture is made by this computer's engines whichever provider
    /// a row names, so the provider is checked for a request that says one and not required: the contract has none.)
    pub fn from_sealed(row_provider: Option<&str>, body: &Value) -> Result<Self, PictureError> {
        if !body.is_object() {
            return Err(bad("the sealed request is not a JSON object"));
        }
        match body.get("v") {
            None | Some(Value::Null) => {}
            Some(v) if v.as_u64() == Some(1) => {}
            Some(_) => return Err(bad("the request is of a version this computer does not read (v is 1)")),
        }
        if body.get("kind").and_then(Value::as_str) != Some(KIND) {
            return Err(bad("the sealed request is not a picture request, whatever its row says"));
        }
        if let Some(sealed) = body.get("providerId").filter(|p| !p.is_null()) {
            if sealed.as_str() != Some(row_provider.unwrap_or("")) {
                return Err(bad("the sealed request does not match its row"));
            }
        }
        for key in ["toolGrant", "tools", "flow", "flowId", "flows"] {
            if body.get(key).is_some_and(|v| !v.is_null()) {
                return Err(bad(format!(
                    "a picture request takes no tools or flows ({key}): it runs OAIY's picture engine and nothing else"
                )));
            }
        }

        let model = match body.get("model") {
            Some(Value::String(m)) if !m.trim().is_empty() => m.as_str(),
            Some(Value::String(_)) | None | Some(Value::Null) => {
                return Err(bad("the request names no picture model: name one of this computer's (there is no default)"))
            }
            Some(_) => return Err(bad("model must be a string")),
        };
        if model.chars().count() > MAX_MODEL_CHARS || model.chars().any(char::is_control) {
            return Err(bad("the picture model's name is malformed"));
        }

        let prompt = match body.get("prompt") {
            Some(Value::String(p)) if !p.trim().is_empty() => p.as_str(),
            Some(Value::String(_)) | None | Some(Value::Null) => return Err(bad("the request has no prompt")),
            Some(_) => return Err(bad("prompt must be a string")),
        };
        if prompt.chars().count() > MAX_PROMPT_CHARS {
            return Err(bad(format!("the prompt is longer than {MAX_PROMPT_CHARS} characters")));
        }

        let size = match body.get("size").and_then(Value::as_str) {
            Some(asked) => SIZES.iter().copied().find(|s| *s == asked),
            None => None,
        }
        .ok_or_else(|| bad(format!("size must be one of {}", SIZES.join(", "))))?;

        let seed = match body.get("seed") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_u64()
                    .filter(|s| *s <= MAX_SEED)
                    .ok_or_else(|| bad(format!("seed must be a whole number from 0 to {MAX_SEED}, or null")))?,
            ),
        };

        let reference = match body.get("reference") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if s.is_empty() => None,
            Some(Value::String(s)) => Some(reference_data_url(s)?),
            Some(Value::Array(_)) => return Err(bad("reference takes one picture at most, as a data: URL")),
            Some(_) => return Err(bad("reference must be a picture, as a data: URL")),
        };

        Ok(Self { model: model.to_string(), prompt: prompt.to_string(), size, seed, reference })
    }
}

/// The reference picture `text` names, as a data URL of the type its bytes are: a site that called a PNG a JPEG would
/// have the engines write it with the wrong extension, for their worker to misread.
fn reference_data_url(text: &str) -> Result<String, PictureError> {
    let rest = text
        .strip_prefix("data:")
        .ok_or_else(|| bad("reference must be a data: URL (data:image/jpeg;base64,...): this computer fetches nothing"))?;
    let (head, b64) = rest.split_once(',').ok_or_else(|| bad("reference is not a data: URL"))?;
    let declared = head
        .strip_suffix(";base64")
        .ok_or_else(|| bad("reference must be base64 (data:image/...;base64,...)"))?
        .to_ascii_lowercase();
    if !matches!(declared.as_str(), "image/png" | "image/jpeg" | "image/jpg" | "image/webp") {
        return Err(bad("reference must be a PNG, JPEG or WebP picture"));
    }
    let too_large = || {
        PictureError::new(TOO_LARGE, format!("the reference picture is larger than {} KiB", MAX_REFERENCE_BYTES / 1024))
    };
    // (counted before decoding, so a reference larger than the lane takes is not decoded to be refused)
    if b64.len() / 4 * 3 > MAX_REFERENCE_BYTES + 3 {
        return Err(too_large());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|_| bad("reference is not valid base64"))?;
    if bytes.len() > MAX_REFERENCE_BYTES {
        return Err(too_large());
    }
    let (format, mime) = match image::guess_format(&bytes) {
        Ok(image::ImageFormat::Png) => (image::ImageFormat::Png, "image/png"),
        Ok(image::ImageFormat::Jpeg) => (image::ImageFormat::Jpeg, "image/jpeg"),
        Ok(image::ImageFormat::WebP) => (image::ImageFormat::WebP, "image/webp"),
        _ => return Err(bad("reference is not a PNG, JPEG or WebP picture")),
    };
    // Its header alone: the engines decode it, and this need only know it is a picture of a sensible size.
    let (width, height) = image::ImageReader::with_format(Cursor::new(&bytes), format)
        .into_dimensions()
        .map_err(|_| bad("the reference picture could not be read"))?;
    if width == 0 || height == 0 || width > MAX_REFERENCE_SIDE || height > MAX_REFERENCE_SIDE {
        return Err(bad(format!("the reference picture's sides must be from 1 to {MAX_REFERENCE_SIDE} pixels")));
    }
    Ok(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&bytes)))
}

/// Whether the engines would answer a request naming `model`, when they have no model of that name, with their
/// default model: they take OpenAI's own model names for theirs (as `oaiy-studio`'s `media::pick_model` does), so
/// an OpenAI client works unchanged. A site's request never gets their default, so such a name is unknown here.
fn stands_in_for_default(model: &str) -> bool {
    model.starts_with("dall-e")
        || model.starts_with("gpt-image")
        || model.starts_with("sora")
        || model.starts_with("tts-")
        || model.ends_with("-tts")
}

/// A seed for a request that leaves it to this computer: as the engines pick one, below 2^31.
fn pick_seed() -> u64 {
    let mut bytes = [0u8; 4];
    if getrandom::getrandom(&mut bytes).is_err() {
        // (no randomness from the system: the clock, which is what the engines themselves would use)
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        bytes = nanos.to_le_bytes();
    }
    u64::from(u32::from_le_bytes(bytes) & 0x7FFF_FFFF)
}

/// The picture `request` asks for, from the engines whose control pages are at `engines_ui`, as the lane's terminal
/// frame (the plaintext to seal): `{picture: {...}}`. `wait` is how long the engines have ([`PICTURE_TIMEOUT`]).
pub async fn make(engines_ui: Option<String>, request: &PictureRequest, wait: Duration) -> Result<Vec<u8>, PictureError> {
    let (gateway, discovery) = crate::ai::routes::engine_discovery_at(engines_ui)
        .await
        .map_err(|e| failed(format!("OAIY's engines cannot make pictures now: {e}")))?;
    check_model(&discovery, request)?;
    let path = crate::ai::engine_services::endpoint_path(&discovery, "images")
        .filter(|p| crate::ai::engine_services::plain_path(p))
        .ok_or_else(|| failed("OAIY's engines have no picture route switched on (their OpenAI images route)"))?;
    let url = crate::ai::engine_services::exact_url(&gateway, &path, "")
        .ok_or_else(|| failed("OAIY's engines name a picture route this computer cannot call"))?;

    let seed = request.seed.unwrap_or_else(pick_seed);
    // Every field the engines are sent, and no other: what the site sent besides these goes nowhere.
    let mut body = json!({
        "model": request.model,
        "prompt": request.prompt,
        "size": request.size,
        "seed": seed,
        "n": 1,
        "response_format": "b64_json",
        "incognito": true,
    });
    if let Some(reference) = &request.reference {
        body["images"] = json!([reference]);
    }

    // On this computer, so never through a proxy, and a redirect is never followed: the engines answer here or not.
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| failed(format!("this computer could not call its engines: {e}")))?;
    let asked = async {
        let response = client
            .post(url)
            // (incognito twice over: the body's field and the header, which the engines read alike)
            .header("x-oaiy-incognito", "1")
            .json(&body)
            .send()
            .await
            .map_err(|e| failed(format!("OAIY's engines did not answer: {}", crate::link::net::unreachable(&e))))?;
        let status = response.status().as_u16();
        let bytes = crate::link::net::read_body_async(response, MAX_ENGINE_REPLY_BYTES)
            .await
            .map_err(|e| failed(format!("OAIY's engines' answer could not be read: {e}")))?;
        Ok::<_, PictureError>((status, bytes))
    };
    let (status, bytes) = tokio::time::timeout(wait, asked).await.map_err(|_| {
        PictureError::new(TIMEOUT, format!("the picture was not made within {} seconds", wait.as_secs()))
    })??;

    let answer: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    if !(200..300).contains(&status) {
        let message = answer
            .pointer("/error/message")
            .and_then(Value::as_str)
            .or_else(|| answer.get("error").and_then(Value::as_str))
            .unwrap_or("no reason given");
        return Err(engine_refusal(status, message));
    }
    let b64 = answer
        .pointer("/data/0/b64_json")
        .and_then(Value::as_str)
        .ok_or_else(|| failed("OAIY's engines answered with no picture"))?;
    let picture = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|_| failed("OAIY's engines answered with a picture that is not valid base64"))?;
    // The model the engines say made it, which is the one named (the name was checked against theirs above).
    let model = answer
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .unwrap_or(&request.model)
        .to_string();
    // Decoding and the JPEG ladder are CPU work of a second or two at worst, on the blocking pool: the lane's own
    // thread serves the chat turns meanwhile.
    tokio::task::spawn_blocking(move || picture_frame(&picture, seed, &model))
        .await
        .map_err(|e| failed(format!("the picture could not be prepared: {e}")))?
}

/// The model `request` names, as the engines' discovery lists it: refused here, before a job is made, when the
/// engines would answer it with their default, when its files are not on this computer, or when it cannot edit and a
/// reference was sent. A model discovery does not list is either switched off or unknown, and the engines say which.
fn check_model(discovery: &Value, request: &PictureRequest) -> Result<(), PictureError> {
    let model = request.model.as_str();
    let listed = discovery
        .pointer("/models/image")
        .and_then(Value::as_array)
        .and_then(|models| models.iter().find(|m| m.get("id").and_then(Value::as_str) == Some(model)));
    let Some(entry) = listed else {
        if stands_in_for_default(model) {
            return Err(PictureError::new(
                MODEL_UNKNOWN,
                format!("OAIY Engines on this computer have no picture model {model:?}"),
            ));
        }
        return Ok(());
    };
    // (a model that cannot run until its files are downloaded is out of use as a switched-off one is, and for the
    // same person to put right: the site warns its administrator of both)
    if entry.get("files_present").and_then(Value::as_bool) == Some(false) {
        let missing: Vec<&str> = entry
            .get("missing_files")
            .and_then(Value::as_array)
            .map(|m| m.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let which = if missing.is_empty() { String::new() } else { format!(" (missing: {})", missing.join(", ")) };
        return Err(PictureError::new(
            MODEL_DISABLED,
            format!(
                "the picture model {model:?} is set up in OAIY Engines, but its files are not on this computer{which}: \
                 download them under Models"
            ),
        ));
    }
    let edits = entry.get("edits").and_then(Value::as_bool) != Some(false)
        && entry.get("max_references").and_then(Value::as_u64) != Some(0);
    if request.reference.is_some() && !edits {
        return Err(bad(format!("the picture model {model:?} makes pictures from text alone: it cannot edit a reference picture")));
    }
    Ok(())
}

/// The contract's code for the engines' refusal of a picture request: their HTTP `status` and the `message` they gave.
pub fn engine_refusal(status: u16, message: &str) -> PictureError {
    // (what `oaiy-studio` says, `media::pick_model` and `media::image_request`: "image model <id> is disabled", "image
    // generation is disabled", "unknown image model \"<id>\"", "no image model is configured; add one under Models")
    let disabled = (message.starts_with("image model ") && message.ends_with(" is disabled"))
        || message.starts_with("image generation is disabled");
    let unknown = message.starts_with("unknown image model") || message.starts_with("no image model is configured");
    let code = match status {
        400 if disabled => MODEL_DISABLED,
        400 if unknown => MODEL_UNKNOWN,
        400 | 422 => BAD_REQUEST,
        413 => TOO_LARGE,
        // ("the media queue is full (N jobs waiting); try again soon")
        429 | 503 => BUSY,
        // (the job still running when the engines' own wait ran out)
        408 | 504 => TIMEOUT,
        _ => ENGINE_FAILED,
    };
    let message: String = message.chars().take(300).collect();
    PictureError::new(code, format!("OAIY's engines: {message}"))
}

/// `bytes`, the picture the engines made, as the lane's terminal frame: within the most a frame carries, as it is
/// when it is a PNG or a JPEG that fits, else as a JPEG that does (a lower quality first, then smaller:
/// `bridge::pictures`' ladder).
pub fn picture_frame(bytes: &[u8], seed: u64, model: &str) -> Result<Vec<u8>, PictureError> {
    let picture = image::load_from_memory(bytes)
        .map_err(|e| failed(format!("OAIY's engines answered with a picture that could not be read ({e})")))?;
    let frame = |mime: &str, data: &str, width: u32, height: u32| {
        json!({
            "v": 1,
            "type": "final",
            "kind": "final",
            "picture": { "mimeType": mime, "data": data, "width": width, "height": height, "seed": seed, "model": model },
        })
    };
    // The frame without the picture, at the picture's own size (one made smaller has no more digits in its sides).
    let around = serde_json::to_vec(&frame("image/jpeg", "", picture.width(), picture.height()))
        .map(|f| f.len())
        .unwrap_or(MAX_FRAME_BYTES);
    let room = MAX_FRAME_BYTES.saturating_sub(around + FRAME_SLACK) / 4 * 3;
    let fitted = match image::guess_format(bytes) {
        Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg) => crate::bridge::pictures::fit_decoded(bytes, &picture, room),
        // (the site's apps read PNG and baseline JPEG alone: a picture in any other format goes as a JPEG)
        _ => crate::bridge::pictures::jpeg_within(&picture, room),
    }
    .map_err(|e| PictureError::new(TOO_LARGE, format!("the picture does not fit in one frame of the lane: {e}")))?;
    let data = base64::engine::general_purpose::STANDARD.encode(&fitted.bytes);
    let out = serde_json::to_vec(&frame(fitted.mime, &data, fitted.width, fitted.height))
        .map_err(|e| failed(format!("the picture's frame could not be written: {e}")))?;
    if out.len() > MAX_FRAME_BYTES {
        return Err(PictureError::new(TOO_LARGE, "the picture does not fit in one frame of the lane"));
    }
    Ok(out)
}

/// A picture's refusal or failure as the lane's terminal frame: a `final` frame, as the picture would have been,
/// carrying the contract's `{error: {code, message}}`.
pub fn error_frame(code: &str, message: &str) -> Value {
    json!({
        "v": 1,
        "type": "final",
        "kind": "final",
        "error": { "code": contract_code(code), "message": message },
    })
}

/// The frame a picture being drawn sends about every [`PROGRESS_EVERY`]. Its kind is its own: a reader that knows
/// only chat's kinds skips it, and the site's notes that the picture is still coming (each frame posted is news of
/// the request to it).
pub fn progress_frame() -> Value {
    json!({ "v": 1, "type": "progress", "kind": "progress", "progress": "drawing" })
}

/// A code of the lane's own as one of [`CODES`]: an envelope that would not open or a request without one is the
/// request's fault (`bad_request`); anything else that went wrong here, this computer's (`engine_failed`).
pub fn contract_code(code: &str) -> &'static str {
    if let Some(known) = CODES.iter().copied().find(|c| *c == code) {
        return known;
    }
    match code {
        "sealed_envelope_invalid" | "invalid_request" => BAD_REQUEST,
        _ => ENGINE_FAILED,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::{extract::State, http::HeaderMap, routing::{get, post}, Json, Router};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// A picture with detail all over, so its PNG is large and a JPEG of it is not trivially small.
    pub(crate) fn detailed_png(width: u32, height: u32) -> Vec<u8> {
        let mut state = 0x2545_F491_u32;
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state & 0x3F) as u8;
            image::Rgb([((x * 3) % 256) as u8 ^ noise, ((y * 5) % 256) as u8 ^ noise, ((x + y) % 256) as u8 ^ (noise >> 1)])
        });
        let mut out = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img).write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    /// A small flat picture, whose PNG fits a frame as it is.
    pub(crate) fn small_png() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(64, 48, image::Rgb([200, 30, 30]));
        let mut out = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img).write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    fn data_url(mime: &str, bytes: &[u8]) -> String {
        format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    /// A request as the contract writes one.
    pub(crate) fn sealed(fields: Value) -> Value {
        let mut body = json!({
            "v": 1,
            "kind": "image",
            "requestId": "fl-ai-job-1",
            "model": "flux-3-klein",
            "prompt": "a red fox, flat vector logo",
            "size": "1024x1024",
            "seed": 12345,
            "reference": null,
            "incognito": true,
        });
        for (k, v) in fields.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    }

    fn refused(fields: Value) -> PictureError {
        PictureRequest::from_sealed(Some("oaiy-engine"), &sealed(fields)).unwrap_err()
    }

    #[test]
    fn a_request_as_the_contract_writes_it_is_taken() {
        let asked = PictureRequest::from_sealed(Some("oaiy-engine"), &sealed(json!({}))).unwrap();
        assert_eq!(
            asked,
            PictureRequest {
                model: "flux-3-klein".into(),
                prompt: "a red fox, flat vector logo".into(),
                size: "1024x1024",
                seed: Some(12345),
                reference: None,
            }
        );
        // No v (the contract's example has none), no provider (it has none either), no seed, an empty reference.
        let mut bare = sealed(json!({ "seed": null, "reference": "" }));
        bare.as_object_mut().unwrap().remove("v");
        let asked = PictureRequest::from_sealed(None, &bare).unwrap();
        assert_eq!((asked.seed, asked.reference), (None, None));
        // `incognito: false` is not a way out of incognito: it is not read at all.
        assert!(PictureRequest::from_sealed(None, &sealed(json!({ "incognito": false }))).is_ok());
    }

    #[test]
    fn the_model_is_named_and_there_is_no_default() {
        for missing in [json!({ "model": null }), json!({ "model": "" }), json!({ "model": "   " })] {
            let e = refused(missing.clone());
            assert_eq!(e.code(), BAD_REQUEST, "{missing}");
            assert!(e.message().contains("no default"), "{}", e.message());
        }
        let mut none = sealed(json!({}));
        none.as_object_mut().unwrap().remove("model");
        assert_eq!(PictureRequest::from_sealed(None, &none).unwrap_err().code(), BAD_REQUEST);
        for malformed in [json!({ "model": 7 }), json!({ "model": "a\nb" }), json!({ "model": "m".repeat(257) })] {
            assert_eq!(refused(malformed.clone()).code(), BAD_REQUEST, "{malformed}");
        }
        // A name with spaces, as the engines' own are ("Qwen Image 2.1 Uncensored"), is taken as it is.
        let spaced = PictureRequest::from_sealed(None, &sealed(json!({ "model": "Qwen Image 2.1" }))).unwrap();
        assert_eq!(spaced.model, "Qwen Image 2.1");
    }

    #[test]
    fn the_prompt_is_at_most_four_thousand_characters() {
        assert!(PictureRequest::from_sealed(None, &sealed(json!({ "prompt": "x".repeat(4_000) }))).is_ok());
        // Characters, not bytes: 4,000 of two bytes each are taken.
        assert!(PictureRequest::from_sealed(None, &sealed(json!({ "prompt": "é".repeat(4_000) }))).is_ok());
        let long = refused(json!({ "prompt": "x".repeat(4_001) }));
        assert_eq!(long.code(), BAD_REQUEST);
        assert!(long.message().contains("4000"), "{}", long.message());
        for empty in [json!({ "prompt": "" }), json!({ "prompt": " \n " }), json!({ "prompt": null }), json!({ "prompt": ["a"] })] {
            assert_eq!(refused(empty.clone()).code(), BAD_REQUEST, "{empty}");
        }
    }

    #[test]
    fn the_size_comes_from_the_allow_list() {
        for size in SIZES {
            let asked = PictureRequest::from_sealed(None, &sealed(json!({ "size": size }))).unwrap();
            assert_eq!(asked.size, size);
        }
        for size in [json!("1024x1025"), json!("2048x2048"), json!("1024X1024"), json!(" 1024x1024"), json!("auto"), json!(1024), json!(null)] {
            let e = refused(json!({ "size": size.clone() }));
            assert_eq!(e.code(), BAD_REQUEST, "{size}");
            assert!(e.message().contains("512x512"), "{}", e.message());
        }
        let mut none = sealed(json!({}));
        none.as_object_mut().unwrap().remove("size");
        assert_eq!(PictureRequest::from_sealed(None, &none).unwrap_err().code(), BAD_REQUEST);
    }

    #[test]
    fn the_seed_is_optional_and_a_whole_number() {
        assert_eq!(PictureRequest::from_sealed(None, &sealed(json!({ "seed": 0 }))).unwrap().seed, Some(0));
        assert_eq!(PictureRequest::from_sealed(None, &sealed(json!({ "seed": MAX_SEED }))).unwrap().seed, Some(MAX_SEED));
        for seed in [json!(-1), json!(1.5), json!("12"), json!(MAX_SEED + 1), json!(true)] {
            assert_eq!(refused(json!({ "seed": seed.clone() })).code(), BAD_REQUEST, "{seed}");
        }
        let picked: Vec<u64> = (0..20).map(|_| pick_seed()).collect();
        assert!(picked.iter().all(|s| *s < 1 << 31), "{picked:?}");
        assert!(picked.windows(2).any(|w| w[0] != w[1]), "seeds are picked, not fixed: {picked:?}");
    }

    #[test]
    fn a_reference_is_one_picture_as_a_data_url() {
        let png = small_png();
        let asked = PictureRequest::from_sealed(None, &sealed(json!({ "reference": data_url("image/png", &png) }))).unwrap();
        assert_eq!(asked.reference.as_deref(), Some(data_url("image/png", &png).as_str()));
        // Called a JPEG, it is a PNG: it goes to the engines as what it is.
        let relabelled = PictureRequest::from_sealed(None, &sealed(json!({ "reference": data_url("image/jpeg", &png) }))).unwrap();
        assert_eq!(relabelled.reference.as_deref(), Some(data_url("image/png", &png).as_str()));
        let jpeg = {
            let mut out = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80)
                .encode_image(&image::RgbImage::from_pixel(32, 32, image::Rgb([1, 2, 3])))
                .unwrap();
            out
        };
        let asked = PictureRequest::from_sealed(None, &sealed(json!({ "reference": data_url("image/jpeg", &jpeg) }))).unwrap();
        assert!(asked.reference.unwrap().starts_with("data:image/jpeg;base64,"));

        for (why, reference) in [
            ("two pictures", json!([data_url("image/png", &png), data_url("image/png", &png)])),
            ("an address to fetch", json!("https://example.com/logo.png")),
            ("not a picture", json!(data_url("image/png", b"not a picture at all"))),
            ("a text file", json!(data_url("text/plain", b"hello"))),
            ("not base64", json!("data:image/png;base64,***")),
            ("not base64-encoded", json!("data:image/png,rawbytes")),
            ("an object", json!({ "url": data_url("image/png", &png) })),
        ] {
            assert_eq!(refused(json!({ "reference": reference })).code(), BAD_REQUEST, "{why}");
        }
        // Larger than the lane takes: too large, and refused before it is decoded.
        let big = vec![0u8; MAX_REFERENCE_BYTES + 1];
        assert_eq!(refused(json!({ "reference": data_url("image/png", &big) })).code(), TOO_LARGE);
    }

    #[test]
    fn a_request_with_a_tool_grant_or_a_flow_is_refused_not_run_without_them() {
        for (key, value) in [("toolGrant", json!("grant-1")), ("tools", json!([{ "name": "x" }])), ("flow", json!("f1")), ("flowId", json!("f1"))] {
            let e = refused(json!({ key: value }));
            assert_eq!(e.code(), BAD_REQUEST, "{key}");
            assert!(e.message().contains("no tools or flows"), "{}", e.message());
        }
        // Null is no grant.
        assert!(PictureRequest::from_sealed(None, &sealed(json!({ "toolGrant": null, "flow": null }))).is_ok());
    }

    #[test]
    fn a_row_cannot_turn_another_sealed_request_into_a_picture() {
        assert_eq!(refused(json!({ "kind": "chat" })).code(), BAD_REQUEST);
        let mut no_kind = sealed(json!({}));
        no_kind.as_object_mut().unwrap().remove("kind");
        assert_eq!(PictureRequest::from_sealed(None, &no_kind).unwrap_err().code(), BAD_REQUEST);
        // A sealed provider must be the row's; one the request does not name is not asked for.
        assert!(PictureRequest::from_sealed(Some("oaiy-engine"), &sealed(json!({ "providerId": "oaiy-engine" }))).is_ok());
        assert_eq!(
            PictureRequest::from_sealed(Some("openai"), &sealed(json!({ "providerId": "oaiy-engine" }))).unwrap_err().code(),
            BAD_REQUEST
        );
        assert_eq!(refused(json!({ "v": 2 })).code(), BAD_REQUEST);
        assert_eq!(PictureRequest::from_sealed(None, &json!([1, 2])).unwrap_err().code(), BAD_REQUEST);
    }

    #[test]
    fn the_engines_refusals_map_to_the_contracts_codes() {
        // (what oaiy-studio answers, word for word)
        for (status, message, code) in [
            (400, "image model qwen-image-2.1 is disabled", MODEL_DISABLED),
            (400, "image model Qwen Image 2.1 Uncensored is disabled", MODEL_DISABLED),
            (400, "image generation is disabled", MODEL_DISABLED),
            (400, "unknown image model \"flux-9\"", MODEL_UNKNOWN),
            (400, "no image model is configured; add one under Models", MODEL_UNKNOWN),
            (400, "width and height must be between 256 and 2048", BAD_REQUEST),
            (400, "red-sdxl is an SDXL model, which cannot edit images; pick a Qwen Image model", BAD_REQUEST),
            (413, "request body too large", TOO_LARGE),
            (503, "the media queue is full (8 jobs waiting); try again soon", BUSY),
            (504, "image job in_progress", TIMEOUT),
            (500, "missing FLUX 3 transformer", ENGINE_FAILED),
            (401, "missing or wrong API key", ENGINE_FAILED),
            (403, "requests from https://x are not allowed", ENGINE_FAILED),
        ] {
            let e = engine_refusal(status, message);
            assert_eq!(e.code(), code, "{status} {message}");
            assert!(e.message().contains(message), "the engines' own words are kept: {}", e.message());
        }
        // A message that only mentions "disabled" in passing is not a disabled model.
        assert_eq!(engine_refusal(400, "the LoRA named is disabled in this request: x").code(), BAD_REQUEST);
        assert!(engine_refusal(500, &"x".repeat(5_000)).message().len() < 400, "a short message");
    }

    #[test]
    fn a_large_picture_is_made_a_jpeg_that_fits_one_frame() {
        let png = detailed_png(1024, 1024);
        assert!(png.len() > MAX_FRAME_BYTES, "the test picture does not fit as it is: {} bytes", png.len());
        let frame = picture_frame(&png, 77, "flux-3-klein").unwrap();
        assert!(frame.len() <= MAX_FRAME_BYTES, "{} bytes", frame.len());
        let frame: Value = serde_json::from_slice(&frame).unwrap();
        let picture = &frame["picture"];
        assert_eq!((frame["kind"].as_str(), frame["type"].as_str()), (Some("final"), Some("final")), "chat's terminal kind");
        assert_eq!(picture["mimeType"], "image/jpeg");
        assert_eq!((picture["seed"].as_u64(), picture["model"].as_str()), (Some(77), Some("flux-3-klein")));
        let bytes = base64::engine::general_purpose::STANDARD.decode(picture["data"].as_str().unwrap()).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((u64::from(decoded.width()), u64::from(decoded.height())), (picture["width"].as_u64().unwrap(), picture["height"].as_u64().unwrap()));
        assert!(decoded.width() <= 1024 && decoded.width() >= 256, "{}", decoded.width());

        // A small one goes as it is: the engines' PNG.
        let small = small_png();
        let frame: Value = serde_json::from_slice(&picture_frame(&small, 1, "m").unwrap()).unwrap();
        assert_eq!(frame["picture"]["mimeType"], "image/png");
        assert_eq!((frame["picture"]["width"].as_u64(), frame["picture"]["height"].as_u64()), (Some(64), Some(48)));
        assert_eq!(base64::engine::general_purpose::STANDARD.decode(frame["picture"]["data"].as_str().unwrap()).unwrap(), small);

        // Not a picture: the engines failed, not the request.
        assert_eq!(picture_frame(b"not a picture", 1, "m").unwrap_err().code(), ENGINE_FAILED);
    }

    #[test]
    fn a_picture_in_another_format_goes_as_a_baseline_jpeg() {
        // The site's apps read PNG and baseline JPEG alone: a WebP the engines made, small as it is, is not sent as one.
        let mut webp = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(40, 30, image::Rgb([10, 200, 90])))
            .write_to(&mut webp, image::ImageFormat::WebP)
            .unwrap();
        let frame: Value = serde_json::from_slice(&picture_frame(webp.get_ref(), 3, "m").unwrap()).unwrap();
        assert_eq!(frame["picture"]["mimeType"], "image/jpeg");
        let bytes = base64::engine::general_purpose::STANDARD.decode(frame["picture"]["data"].as_str().unwrap()).unwrap();
        assert_eq!(image::guess_format(&bytes).unwrap(), image::ImageFormat::Jpeg);
        assert_eq!(start_of_frame(&bytes), Some(0xC0), "a baseline JPEG (SOF0), not a progressive one (SOF2)");
        // And the ladder's JPEG of a large picture is baseline too.
        let frame: Value = serde_json::from_slice(&picture_frame(&detailed_png(1024, 1024), 3, "m").unwrap()).unwrap();
        let bytes = base64::engine::general_purpose::STANDARD.decode(frame["picture"]["data"].as_str().unwrap()).unwrap();
        assert_eq!(start_of_frame(&bytes), Some(0xC0));
    }

    /// The start-of-frame marker of a JPEG, found by walking its segments: 0xC0 baseline, 0xC2 progressive.
    fn start_of_frame(jpeg: &[u8]) -> Option<u8> {
        let mut at = 2; // past the start of image
        while at + 4 <= jpeg.len() && jpeg[at] == 0xFF {
            let marker = jpeg[at + 1];
            if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
                return Some(marker);
            }
            at += 2 + usize::from(u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]));
        }
        None
    }

    #[test]
    fn the_progress_frame_is_a_kind_of_its_own() {
        assert_eq!(progress_frame(), json!({ "v": 1, "type": "progress", "kind": "progress", "progress": "drawing" }));
    }

    #[test]
    fn an_error_comes_in_the_final_frame_in_the_contracts_shape() {
        let frame = error_frame(MODEL_DISABLED, "switched off");
        assert_eq!(
            frame,
            json!({ "v": 1, "type": "final", "kind": "final", "error": { "code": "model_disabled", "message": "switched off" } })
        );
        // The lane's own codes become the contract's.
        assert_eq!(error_frame("sealed_envelope_invalid", "x")["error"]["code"], "bad_request");
        assert_eq!(error_frame("invalid_request", "x")["error"]["code"], "bad_request");
        assert_eq!(error_frame("upstream_error", "x")["error"]["code"], "engine_failed");
        for code in CODES {
            assert_eq!(contract_code(code), code);
        }
    }

    // ---- the engines, stood in for ----

    /// What the stand-in engines have and do.
    #[derive(Clone)]
    pub(crate) struct Setup {
        /// The image models discovery lists (the switched-on ones), as discovery writes each.
        pub image_models: Vec<Value>,
        /// Image models set up and switched off: the images route refuses them as the engines do.
        pub disabled: Vec<String>,
        /// The language model chosen in Engines ("" for none).
        pub llm: String,
        /// The picture the images route answers with.
        pub picture: Vec<u8>,
        /// What the images route refuses every request with, if anything (a full queue's 503).
        pub refuse_with: Option<(u16, String)>,
        /// How long the images route takes.
        pub delay: Duration,
        /// How long the chat route takes.
        pub chat_delay: Duration,
        /// What the images route waits for before it answers (it gives up after 20 seconds).
        pub hold: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
        /// What the chat route waits for before it answers, the same way.
        pub chat_hold: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
        /// The chat route answers as a stream (`text/event-stream`), in this many pieces with this long before each,
        /// as a model writing its answer does; none: one whole answer.
        pub chat_stream: Option<(usize, Duration)>,
    }

    impl Default for Setup {
        fn default() -> Self {
            Self {
                image_models: vec![
                    json!({ "id": "flux-3-klein", "default": true, "architecture": "flux3-klein", "edits": true, "max_references": 10, "files_present": true, "missing_files": [] }),
                    json!({ "id": "red-sdxl", "default": false, "architecture": "sdxl", "edits": false, "max_references": 0, "files_present": true, "missing_files": [] }),
                    json!({ "id": "not-downloaded", "default": false, "architecture": "flux3-klein", "edits": true, "max_references": 10, "files_present": false, "missing_files": ["transformer", "vae"] }),
                ],
                disabled: vec!["qwen-image-2.1".into()],
                llm: "flash".into(),
                picture: small_png(),
                refuse_with: None,
                delay: Duration::ZERO,
                chat_delay: Duration::ZERO,
                hold: None,
                chat_hold: None,
                chat_stream: None,
            }
        }
    }

    /// One request the stand-in engines were sent.
    #[derive(Clone, Debug)]
    pub(crate) struct Asked {
        pub route: &'static str,
        pub incognito_header: Option<String>,
        pub body: Value,
    }

    /// Engines standing in for OAIY's: their control page (`/api/state`), and behind it a gateway with discovery, the
    /// model list, chat and the OpenAI images route, which answers as `oaiy-studio`'s does and records what it is sent.
    pub(crate) struct FakeEngines {
        pub ui: String,
        pub asked: Arc<Mutex<Vec<Asked>>>,
        /// Pictures whose caller hung up before they were answered: `oaiy-studio` cancels such a job.
        pub abandoned: Arc<AtomicUsize>,
        /// The pieces of streamed chat answers written so far ([`Setup::chat_stream`]): a caller that hangs up
        /// stops the writing, as it stops `oaiy-studio`'s language model.
        pub chat_pieces: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }

    impl FakeEngines {
        pub(crate) fn pictures_asked(&self) -> Vec<Asked> {
            self.asked.lock().unwrap().iter().filter(|a| a.route == "images").cloned().collect()
        }

        pub(crate) fn chats_asked(&self) -> Vec<Asked> {
            self.asked.lock().unwrap().iter().filter(|a| a.route == "chat").cloned().collect()
        }
    }

    /// Counts a picture as abandoned when its request is dropped before it is answered (its caller hung up).
    struct Abandoned {
        count: Arc<AtomicUsize>,
        answered: bool,
    }

    impl Drop for Abandoned {
        fn drop(&mut self) {
            if !self.answered {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl Drop for FakeEngines {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    #[derive(Clone)]
    struct Shared {
        setup: Setup,
        gateway: String,
        asked: Arc<Mutex<Vec<Asked>>>,
        abandoned: Arc<AtomicUsize>,
        chat_pieces: Arc<AtomicUsize>,
    }

    /// A chat answer as a stream of `pieces` chunks in OpenAI's shape, `gap` before each, then `[DONE]`. Each piece
    /// written is counted; a caller that hangs up ends the stream, and the count with it.
    fn streamed_answer(pieces: usize, gap: Duration, written: Arc<AtomicUsize>) -> axum::response::Response {
        use axum::response::IntoResponse;
        let stream = futures_util::stream::unfold(0usize, move |i| {
            let written = written.clone();
            async move {
                if i > pieces {
                    return None;
                }
                tokio::time::sleep(gap).await;
                let event = if i == pieces {
                    "data: [DONE]\n\n".to_string()
                } else {
                    written.fetch_add(1, Ordering::SeqCst);
                    format!("data: {}\n\n", json!({ "id": "cmpl-1", "object": "chat.completion.chunk", "choices": [{ "index": 0, "delta": { "content": format!("word{i} ") } }] }))
                };
                Some((Ok::<_, std::convert::Infallible>(event), i + 1))
            }
        });
        ([(axum::http::header::CONTENT_TYPE, "text/event-stream")], axum::body::Body::from_stream(stream)).into_response()
    }

    pub(crate) async fn fake_engines(setup: Setup) -> FakeEngines {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ui = format!("http://{}", listener.local_addr().unwrap());
        let asked: Arc<Mutex<Vec<Asked>>> = Arc::default();
        let (abandoned, chat_pieces): (Arc<AtomicUsize>, Arc<AtomicUsize>) = Default::default();
        let shared = Shared { setup, gateway: format!("{ui}/gw"), asked: asked.clone(), abandoned: abandoned.clone(), chat_pieces: chat_pieces.clone() };
        let app = Router::new()
            .route("/api/state", get(|State(s): State<Shared>| async move { Json(json!({ "gateway_url": s.gateway })) }))
            .route(
                "/gw/v1/discovery",
                get(|State(s): State<Shared>| async move {
                    let llm: Vec<Value> = if s.setup.llm.is_empty() { Vec::new() } else { vec![json!({ "id": s.setup.llm, "default": true, "files_present": true })] };
                    Json(json!({
                        "endpoints": [
                            { "name": "chat", "path": "/v1/chat/completions", "spec": "openai" },
                            { "name": "images", "path": "/oaiy/images", "spec": "oaiy" },
                            { "name": "images", "path": "/v1/images/generations", "spec": "openai" },
                        ],
                        "models": { "llm": llm, "image": s.setup.image_models },
                        "defaults": { "llm": s.setup.llm, "image": "flux-3-klein" },
                    }))
                }),
            )
            .route(
                "/gw/v1/models",
                get(|State(s): State<Shared>| async move {
                    let mut data: Vec<Value> = Vec::new();
                    if !s.setup.llm.is_empty() {
                        data.push(json!({ "id": s.setup.llm, "object": "model", "owned_by": "oaiy" }));
                    }
                    for m in &s.setup.image_models {
                        data.push(json!({ "id": m["id"], "object": "model", "owned_by": "oaiy-studio", "type": "image", "loaded": false }));
                    }
                    Json(json!({ "object": "list", "data": data }))
                }),
            )
            .route(
                "/gw/v1/chat/completions",
                post(|State(s): State<Shared>, Json(body): Json<Value>| async move {
                    use axum::response::IntoResponse;
                    s.asked.lock().unwrap().push(Asked { route: "chat", incognito_header: None, body });
                    tokio::time::sleep(s.setup.chat_delay).await;
                    if let Some(hold) = s.setup.chat_hold.clone() {
                        let until = std::time::Instant::now() + Duration::from_secs(20);
                        while !hold() && std::time::Instant::now() < until {
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                    }
                    if let Some((pieces, gap)) = s.setup.chat_stream {
                        return streamed_answer(pieces, gap, s.chat_pieces.clone());
                    }
                    Json(json!({
                        "id": "cmpl-1",
                        "object": "chat.completion",
                        "choices": [{ "index": 0, "message": { "role": "assistant", "content": "hello from the engine" }, "finish_reason": "stop" }],
                    }))
                    .into_response()
                }),
            )
            .route("/gw/v1/images/generations", post(images))
            .with_state(shared);
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        FakeEngines { ui, asked, abandoned, chat_pieces, server }
    }

    /// The images route, as `oaiy-studio` answers: a model it has switched off or does not know is refused; one of
    /// OpenAI's names gets its default model (which a site's request must never reach); else the picture.
    async fn images(State(s): State<Shared>, headers: HeaderMap, Json(body): Json<Value>) -> axum::response::Response {
        use axum::response::IntoResponse;
        let incognito_header = headers.get("x-oaiy-incognito").and_then(|v| v.to_str().ok()).map(str::to_string);
        s.asked.lock().unwrap().push(Asked { route: "images", incognito_header, body: body.clone() });
        let refuse = |status: u16, message: String| {
            (axum::http::StatusCode::from_u16(status).unwrap(), Json(json!({ "error": { "message": message, "type": "invalid_request_error", "param": null, "code": "invalid_request" } })))
                .into_response()
        };
        if let Some((status, message)) = s.setup.refuse_with.clone() {
            return refuse(status, message);
        }
        let asked = body["model"].as_str().unwrap_or_default().to_string();
        let model = if s.setup.disabled.contains(&asked) {
            return refuse(400, format!("image model {asked} is disabled"));
        } else if s.setup.image_models.iter().any(|m| m["id"] == asked) {
            asked
        } else if stands_in_for_default(&asked) {
            "flux-3-klein".to_string()
        } else {
            return refuse(400, format!("unknown image model {asked:?}"));
        };
        // (dropped with this request when its caller hangs up while the picture is being made)
        let mut making = Abandoned { count: s.abandoned.clone(), answered: false };
        tokio::time::sleep(s.setup.delay).await;
        if let Some(hold) = s.setup.hold.clone() {
            let until = std::time::Instant::now() + Duration::from_secs(20);
            while !hold() && std::time::Instant::now() < until {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        making.answered = true;
        Json(json!({
            "created": 1,
            "data": [{ "b64_json": base64::engine::general_purpose::STANDARD.encode(&s.setup.picture), "revised_prompt": body["prompt"] }],
            "output_format": "png",
            "size": body["size"],
            "model": model,
        }))
        .into_response()
    }

    fn request(model: &str) -> PictureRequest {
        PictureRequest { model: model.into(), prompt: "a fox".into(), size: "768x1024", seed: Some(9), reference: None }
    }

    #[tokio::test]
    async fn a_picture_is_asked_of_the_engines_incognito_with_the_fields_named_and_no_others() {
        let engines = fake_engines(Setup::default()).await;
        let png = small_png();
        let mut asked = request("flux-3-klein");
        asked.reference = Some(data_url("image/png", &png));
        let frame: Value = serde_json::from_slice(&make(Some(engines.ui.clone()), &asked, PICTURE_TIMEOUT).await.unwrap()).unwrap();
        assert_eq!(frame["picture"]["model"], "flux-3-klein");
        assert_eq!(frame["picture"]["seed"], 9);
        assert_eq!(frame["picture"]["mimeType"], "image/png");

        let sent = engines.pictures_asked();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].incognito_header.as_deref(), Some("1"));
        assert_eq!(
            sent[0].body,
            json!({
                "model": "flux-3-klein",
                "prompt": "a fox",
                "size": "768x1024",
                "seed": 9,
                "n": 1,
                "response_format": "b64_json",
                "incognito": true,
                "images": [data_url("image/png", &png)],
            })
        );
    }

    #[tokio::test]
    async fn a_seed_left_to_this_computer_is_picked_sent_and_reported() {
        let engines = fake_engines(Setup::default()).await;
        let mut asked = request("flux-3-klein");
        asked.seed = None;
        let frame: Value = serde_json::from_slice(&make(Some(engines.ui.clone()), &asked, PICTURE_TIMEOUT).await.unwrap()).unwrap();
        let sent = engines.pictures_asked();
        let seed = sent[0].body["seed"].as_u64().expect("a seed is always sent: the engines' answer says none in incognito");
        assert_eq!(frame["picture"]["seed"].as_u64(), Some(seed));
        assert!(sent[0].body.get("images").is_none(), "no reference, no images");
    }

    #[tokio::test]
    async fn a_model_switched_off_or_unknown_is_said_so_by_the_engines() {
        let engines = fake_engines(Setup::default()).await;
        let disabled = make(Some(engines.ui.clone()), &request("qwen-image-2.1"), PICTURE_TIMEOUT).await.unwrap_err();
        assert_eq!(disabled.code(), MODEL_DISABLED, "{}", disabled.message());
        assert!(disabled.message().contains("is disabled"), "{}", disabled.message());
        let unknown = make(Some(engines.ui.clone()), &request("flux-9"), PICTURE_TIMEOUT).await.unwrap_err();
        assert_eq!(unknown.code(), MODEL_UNKNOWN, "{}", unknown.message());
    }

    #[tokio::test]
    async fn no_request_reaches_the_engines_default_model() {
        // The engines would answer "dall-e-3" with their default; this computer never asks them.
        let engines = fake_engines(Setup::default()).await;
        for name in ["dall-e-3", "gpt-image-1", "sora-2"] {
            let e = make(Some(engines.ui.clone()), &request(name), PICTURE_TIMEOUT).await.unwrap_err();
            assert_eq!(e.code(), MODEL_UNKNOWN, "{name}");
        }
        assert!(engines.pictures_asked().is_empty(), "{:?}", engines.pictures_asked());
    }

    #[tokio::test]
    async fn what_cannot_run_is_refused_before_the_engines_make_a_job() {
        let engines = fake_engines(Setup::default()).await;
        let missing = make(Some(engines.ui.clone()), &request("not-downloaded"), PICTURE_TIMEOUT).await.unwrap_err();
        assert_eq!(missing.code(), MODEL_DISABLED, "out of use until its files are there: {}", missing.message());
        assert!(missing.message().contains("transformer, vae") && missing.message().contains("Models"), "{}", missing.message());
        let mut edit = request("red-sdxl");
        edit.reference = Some(data_url("image/png", &small_png()));
        let no_edits = make(Some(engines.ui.clone()), &edit, PICTURE_TIMEOUT).await.unwrap_err();
        assert_eq!(no_edits.code(), BAD_REQUEST);
        assert!(engines.pictures_asked().is_empty());
        // Without a reference, the same model makes a picture.
        assert!(make(Some(engines.ui.clone()), &request("red-sdxl"), PICTURE_TIMEOUT).await.is_ok());
    }

    #[tokio::test]
    async fn engines_that_are_not_running_busy_or_slow_are_each_said_so() {
        let none = make(None, &request("flux-3-klein"), PICTURE_TIMEOUT).await.unwrap_err();
        assert_eq!(none.code(), ENGINE_FAILED);
        assert!(none.message().contains("not running"), "{}", none.message());

        let busy = fake_engines(Setup { refuse_with: Some((503, "the media queue is full (8 jobs waiting); try again soon".into())), ..Setup::default() }).await;
        assert_eq!(make(Some(busy.ui.clone()), &request("flux-3-klein"), PICTURE_TIMEOUT).await.unwrap_err().code(), BUSY);

        let slow = fake_engines(Setup { delay: Duration::from_secs(5), ..Setup::default() }).await;
        let started = std::time::Instant::now();
        let late = make(Some(slow.ui.clone()), &request("flux-3-klein"), Duration::from_millis(400)).await.unwrap_err();
        assert_eq!(late.code(), TIMEOUT, "{}", late.message());
        assert!(started.elapsed() < Duration::from_secs(4), "{:?}", started.elapsed());
    }

    #[tokio::test]
    async fn a_large_picture_from_the_engines_comes_back_within_one_frame() {
        let engines = fake_engines(Setup { picture: detailed_png(1024, 1024), ..Setup::default() }).await;
        let frame = make(Some(engines.ui.clone()), &request("flux-3-klein"), PICTURE_TIMEOUT).await.unwrap();
        assert!(frame.len() <= MAX_FRAME_BYTES, "{}", frame.len());
        let frame: Value = serde_json::from_slice(&frame).unwrap();
        assert_eq!(frame["picture"]["mimeType"], "image/jpeg");
    }
}
