//! A flow's pictures, for a caller that cannot reach them where they are.
//!
//! A flow answers with references to the pictures it made: the engine's files, as this desktop's own addresses
//! (`http://127.0.0.1:<port>/api/ai/engine/gateway/files/images/….png`), which a caller on this computer reads with
//! its credential. A caller elsewhere cannot: a FormLogic automation's answer goes back through FormLogic's relay,
//! and an MCP client is handed content, not addresses. So the pictures go with the answer: each once, in the order
//! the answer names them (its top level first), as base64, within a budget. A picture larger than what is left is
//! made a JPEG that fits (a lower quality first, then smaller); one that cannot fit is named as left out, with why.

use base64::Engine as _;
use image::{codecs::jpeg::JpegEncoder, imageops::FilterType, ImageFormat};
use serde_json::{json, Value};
use std::time::Duration;

/// Where the engine's files are on this desktop.
pub const FILES_PATH: &str = "/api/ai/engine/gateway/files/";

/// Below this many bytes left, no picture is tried: a JPEG that small says little.
const MIN_BYTES: usize = 8 * 1024;

/// A picture is made no smaller than this on its longer side to fit.
const MIN_SIDE: u32 = 256;

/// The JPEG qualities tried, best first, at each size.
const QUALITIES: [u8; 7] = [90, 82, 74, 66, 58, 50, 42];

/// The port and path of a reference to a picture of the engine's on this computer, or `None` for anything else (a
/// picture elsewhere, another file, any other text).
pub fn locate(reference: &str) -> Option<(Option<u16>, &str)> {
    let rest = reference.strip_prefix("http://")?;
    let at = rest.find('/')?;
    let (authority, path) = rest.split_at(at);
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !port.contains(']') => (host, Some(port.parse::<u16>().ok()?)),
        _ => (authority, None),
    };
    if !matches!(host, "127.0.0.1" | "localhost" | "[::1]") {
        return None;
    }
    let file = path.strip_prefix(FILES_PATH)?;
    let file = file.split(['?', '#']).next()?;
    if file.split('/').any(|part| part.is_empty() || part == "." || part == "..") {
        return None;
    }
    let ext = file.rsplit_once('.')?.1.to_ascii_lowercase();
    matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp").then_some((port, path))
}

/// The pictures an answer names, each once: its top level first, then what is under it, in the order each level
/// has them.
pub fn references(answer: &Value) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    let mut level = vec![answer];
    while !level.is_empty() {
        let mut next = Vec::new();
        for value in level {
            match value {
                Value::String(s) if locate(s).is_some() && !found.contains(s) => found.push(s.clone()),
                Value::Array(items) => next.extend(items.iter()),
                Value::Object(fields) => next.extend(fields.values()),
                _ => {}
            }
        }
        level = next;
    }
    found
}

/// A picture as it is handed over.
#[derive(Debug)]
pub struct Fitted {
    pub bytes: Vec<u8>,
    pub mime: &'static str,
    pub width: u32,
    pub height: u32,
    /// Whether it was made a JPEG (and maybe smaller) to fit.
    pub reencoded: bool,
}

/// `bytes`, a picture, in at most `max` bytes: as it is when it fits, else as a JPEG that does.
pub fn fit(bytes: &[u8], max: usize) -> Result<Fitted, String> {
    let picture = image::load_from_memory(bytes).map_err(|e| format!("it is not a picture this can read ({e})"))?;
    let (width, height) = (picture.width(), picture.height());
    if bytes.len() <= max {
        let mime = match image::guess_format(bytes) {
            Ok(ImageFormat::Png) => "image/png",
            Ok(ImageFormat::Jpeg) => "image/jpeg",
            Ok(ImageFormat::WebP) => "image/webp",
            _ => "application/octet-stream",
        };
        return Ok(Fitted { bytes: bytes.to_vec(), mime, width, height, reencoded: false });
    }
    let mut current = picture.to_rgb8();
    loop {
        for quality in QUALITIES {
            let mut out = Vec::new();
            JpegEncoder::new_with_quality(&mut out, quality)
                .encode_image(&current)
                .map_err(|e| format!("it could not be made a JPEG ({e})"))?;
            if out.len() <= max {
                let (width, height) = current.dimensions();
                return Ok(Fitted { bytes: out, mime: "image/jpeg", width, height, reencoded: true });
            }
        }
        let (w, h) = current.dimensions();
        if w.max(h) <= MIN_SIDE {
            return Err(format!("it does not fit in {max} bytes even as a small JPEG"));
        }
        let (w, h) = ((w as f32 * 0.75).round().max(1.0) as u32, (h as f32 * 0.75).round().max(1.0) as u32);
        current = image::imageops::resize(&current, w, h, FilterType::Triangle);
    }
}

/// The base64 of `n` bytes is this long.
fn base64_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

/// The pictures `answer` names, each `{ref, included, mimeType, data, width, height, bytes, reencoded}` (`data` its
/// base64) while their base64 together stays within `budget` characters, else `{ref, included: false, reason}`.
/// `fetch` reads a reference's bytes.
pub fn gather(answer: &Value, budget: usize, fetch: &dyn Fn(&str) -> Result<Vec<u8>, String>) -> Vec<Value> {
    let mut left = budget;
    references(answer)
        .into_iter()
        .map(|reference| {
            let skipped = |reason: String| json!({ "ref": reference, "included": false, "reason": reason });
            let room = left / 4 * 3;
            if room < MIN_BYTES {
                return skipped("no room was left for it within maxPictureBytes".into());
            }
            let bytes = match fetch(&reference) {
                Ok(bytes) => bytes,
                Err(e) => return skipped(e),
            };
            match fit(&bytes, room) {
                Ok(f) => {
                    left -= base64_len(f.bytes.len());
                    json!({
                        "ref": reference,
                        "included": true,
                        "mimeType": f.mime,
                        "width": f.width,
                        "height": f.height,
                        "bytes": f.bytes.len(),
                        "reencoded": f.reencoded,
                        "data": base64::engine::general_purpose::STANDARD.encode(&f.bytes),
                    })
                }
                Err(e) => skipped(e),
            }
        })
        .collect()
}

/// A picture input this large or larger, as a data URL, goes to a flow as a reference instead.
pub const BY_REFERENCE_FROM: usize = 64 * 1024;

/// The bytes and type of a `data:image/…;base64,…` URL.
pub fn data_url(text: &str) -> Option<(Vec<u8>, &str)> {
    let rest = text.strip_prefix("data:")?;
    let (mime, b64) = rest.split_once(";base64,")?;
    if !matches!(mime, "image/png" | "image/jpeg" | "image/jpg" | "image/webp") {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim()).ok()?;
    Some((bytes, mime))
}

/// A flow's inputs with each picture given inline (a data URL of [`BY_REFERENCE_FROM`] or more, at the top level)
/// stored with the engines first and replaced by its reference: a flow hands a picture from step to step by its
/// address, and a few MB of one inline is more than one step's call may carry (4 MiB). `upload` stores one and
/// answers its reference.
pub fn by_reference(inputs: &mut Value, upload: &dyn Fn(&[u8], &str) -> Result<String, String>) -> Result<(), String> {
    let Some(fields) = inputs.as_object_mut() else { return Ok(()) };
    for (name, value) in fields.iter_mut() {
        let Some(text) = value.as_str().filter(|t| t.len() >= BY_REFERENCE_FROM) else { continue };
        let Some((bytes, mime)) = data_url(text) else { continue };
        let reference = upload(&bytes, mime).map_err(|e| format!("the picture in {name} could not be handed to the engines: {e}"))?;
        *value = Value::String(reference);
    }
    Ok(())
}

/// Store a picture with this desktop's engines (`POST` to their files' route) and answer its reference, as this
/// desktop's own address of it (as a picture the engines made is named).
pub fn upload_local(bytes: &[u8], mime: &str) -> Result<String, String> {
    let base = format!("http://127.0.0.1:{}", crate::api_port());
    let response = reqwest::blocking::Client::new()
        .post(format!("{base}{}", FILES_PATH.trim_end_matches('/')))
        .bearer_auth(crate::internal_token())
        .header(reqwest::header::CONTENT_TYPE, mime)
        .body(bytes.to_vec())
        .timeout(Duration::from_secs(60))
        .send()
        .map_err(|e| format!("the engines could not be reached ({e})"))?;
    let status = response.status();
    let body: Value = response.json().unwrap_or(Value::Null);
    if !status.is_success() {
        let why = body.pointer("/error/message").and_then(Value::as_str).unwrap_or("no reason given");
        return Err(format!("the engines answered HTTP {} ({why})", status.as_u16()));
    }
    let url = body.get("url").and_then(Value::as_str).ok_or("the engines answered no address for it")?;
    // `/files/inputs/…` on the engines is `/api/ai/engine/gateway/files/inputs/…` here.
    let file = url.strip_prefix("/files/").ok_or("the engines answered an address that is not one of their files")?;
    Ok(format!("{base}{FILES_PATH}{file}"))
}

/// A reference's bytes, read from this desktop's own engine route with its own credential: a reference to this
/// desktop's port alone (another port of this computer is another program's).
pub fn fetch_local(reference: &str) -> Result<Vec<u8>, String> {
    const MAX: u64 = 64 * 1024 * 1024;
    let (port, path) = locate(reference).ok_or("it is not a picture of this desktop's engines")?;
    if port != Some(crate::api_port()) {
        return Err("it is a picture at another address than this desktop's".into());
    }
    let response = reqwest::blocking::Client::new()
        .get(format!("http://127.0.0.1:{}{path}", crate::api_port()))
        .bearer_auth(crate::internal_token())
        .timeout(Duration::from_secs(30))
        .send()
        .map_err(|e| format!("it could not be read ({e})"))?;
    if !response.status().is_success() {
        return Err(format!("reading it answered HTTP {}", response.status().as_u16()));
    }
    let mut bytes = Vec::new();
    use std::io::Read;
    response.take(MAX + 1).read_to_end(&mut bytes).map_err(|e| format!("it could not be read ({e})"))?;
    if bytes.len() as u64 > MAX {
        return Err("it is larger than 64 MiB".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        // A picture with detail, so a JPEG of it is not trivially small.
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x ^ y) % 256) as u8]));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img).write_to(&mut out, ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn only_this_computers_engine_pictures_are_references() {
        let ok = "http://127.0.0.1:17972/api/ai/engine/gateway/files/images/2026-10-17/a.png";
        assert_eq!(locate(ok), Some((Some(17972), "/api/ai/engine/gateway/files/images/2026-10-17/a.png")));
        assert!(locate("http://localhost:1/api/ai/engine/gateway/files/x.JPG?v=1").is_some());
        for not in [
            "https://127.0.0.1:17972/api/ai/engine/gateway/files/a.png",
            "http://example.com/api/ai/engine/gateway/files/a.png",
            "http://127.0.0.1:17972/api/ai/engine/gateway/files/a.wav",
            "http://127.0.0.1:17972/api/ai/engine/gateway/files/../secret.png",
            "http://127.0.0.1:17972/files/a.png",
            "a.png",
        ] {
            assert_eq!(locate(not), None, "{not}");
        }
    }

    /// The answer's top level first (the final picture before the ones under `pictures`), each picture once.
    #[test]
    fn references_come_shallow_first_and_once() {
        let a = "http://127.0.0.1:1/api/ai/engine/gateway/files/images/a.png";
        let b = "http://127.0.0.1:1/api/ai/engine/gateway/files/images/b.png";
        let answer = json!({"pictures": {"made": {"file": b}, "edited": {"file": a}}, "image": a, "original": b, "prompt": "x"});
        assert_eq!(references(&answer), vec![a.to_string(), b.to_string()]);
    }

    #[test]
    fn a_picture_that_fits_goes_as_it_is_and_one_that_does_not_as_a_jpeg_that_does() {
        let bytes = png(512, 512);
        let same = fit(&bytes, bytes.len()).unwrap();
        assert!(!same.reencoded && same.mime == "image/png" && same.bytes == bytes);
        let small = fit(&bytes, 40_000).unwrap();
        assert!(small.reencoded && small.mime == "image/jpeg" && small.bytes.len() <= 40_000, "{} bytes", small.bytes.len());
        assert!(fit(&bytes, 500).is_err());
        assert!(fit(b"not a picture", 1_000_000).is_err());
    }

    /// A large picture given inline goes to the flow as the reference the engines answer; text, a small picture and
    /// anything that is not a picture are left as they are.
    #[test]
    fn a_large_picture_input_goes_by_reference() {
        let picture = png(400, 400);
        assert!(picture.len() >= BY_REFERENCE_FROM, "the test picture is large enough: {}", picture.len());
        let inline = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&picture));
        let small = "data:image/png;base64,iVBORw0KGgo=";
        let mut inputs = json!({"Picture": inline, "Prompt": "a cat", "Small": small, "Other": format!("data:text/plain;base64,{}", "QUJD".repeat(30_000))});
        let seen = std::cell::RefCell::new(Vec::new());
        let upload = |bytes: &[u8], mime: &str| {
            seen.borrow_mut().push((bytes.len(), mime.to_string()));
            Ok("http://127.0.0.1:1/api/ai/engine/gateway/files/inputs/ref_a.png".to_string())
        };
        by_reference(&mut inputs, &upload).unwrap();
        assert_eq!(inputs["Picture"], "http://127.0.0.1:1/api/ai/engine/gateway/files/inputs/ref_a.png");
        assert_eq!((inputs["Prompt"].as_str(), inputs["Small"].as_str()), (Some("a cat"), Some(small)));
        assert!(inputs["Other"].as_str().unwrap().starts_with("data:text/plain"));
        assert_eq!(*seen.borrow(), vec![(picture.len(), "image/png".to_string())]);
        let mut again = json!({"Picture": inline});
        let err = by_reference(&mut again, &|_: &[u8], _: &str| Err("engines stopped".to_string())).unwrap_err();
        assert!(err.contains("Picture") && err.contains("engines stopped"), "{err}");
    }

    #[test]
    fn the_budget_holds_the_base64_of_every_picture_together() {
        let a = "http://127.0.0.1:1/api/ai/engine/gateway/files/a.png";
        let b = "http://127.0.0.1:1/api/ai/engine/gateway/files/b.png";
        let bytes = png(640, 640);
        let fetch = |_: &str| Ok(bytes.clone());
        let got = gather(&json!({"image": a, "original": b}), 120_000, &fetch);
        let total: usize = got.iter().filter_map(|p| p.get("data").and_then(Value::as_str)).map(str::len).sum();
        assert!(total <= 120_000, "{total}");
        assert_eq!(got[0]["ref"], a);
        assert_eq!(got[0]["included"], true);
        assert!(got[0]["data"].as_str().unwrap().len() > 40_000, "the first picture takes the room it needs");
        let missing = gather(&json!({"image": a}), 120_000, &|_: &str| Err("gone".into()));
        assert_eq!(missing[0], json!({"ref": a, "included": false, "reason": "gone"}));
    }
}
