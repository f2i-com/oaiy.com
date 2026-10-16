//! Picture tools: a picture's background removed (BiRefNet), or the picture made
//! two or four times larger (Real-ESRGAN), as jobs for the worker from
//! `/v1/images/background_removal` and `/v1/images/upscale`.
//!
//! `media.picture` names the models: `background` (a BiRefNet folder) and
//! `upscaler` (Real-ESRGAN x4plus's `.pth`). Adding either on the Models page
//! fills it (and a 3D model's helpers).
use crate::config;
use crate::util::{bool_or, str_or};
use oaiy_engine::json::Json;
use std::path::Path;

/// The largest picture `upscale` takes, in pixels (as the worker's limit).
pub const MAX_UPSCALE_PIXELS: u64 = 4 << 20;

/// What a picture job does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    RemoveBackground,
    Upscale,
}

impl Op {
    /// The model's name in discovery and on jobs.
    pub fn model(self) -> &'static str {
        match self {
            Op::RemoveBackground => "birefnet",
            Op::Upscale => "real-esrgan-x4plus",
        }
    }

    /// The field of `media.picture` that names its weights.
    pub fn field(self) -> &'static str {
        match self {
            Op::RemoveBackground => "background",
            Op::Upscale => "upscaler",
        }
    }
}

/// Is the tool configured (and the section on)?
pub fn ready(cfg: &Json, op: Op) -> bool {
    cfg.get("media").and_then(|m| m.get("picture")).is_some_and(|s| bool_or(s, "enabled", true) && !str_or(s, op.field(), "").trim().is_empty())
}

/// A picture job's request for the worker, with its model name and a label for the job list.
/// The picture comes from `image` (or `input_reference`): a `data:` URL, or a local path from this
/// machine; `upscale` takes `scale` 2 or 4 (default 4).
pub fn picture_request(cfg: &Json, root: &Path, output_dir: &Path, body: &Json, allow_local: bool, op: Op) -> Result<(Json, String, String), String> {
    let section = cfg.get("media").and_then(|m| m.get("picture")).ok_or("no picture tools section")?;
    if !bool_or(section, "enabled", true) {
        return Err("picture tools are disabled".into());
    }
    let weights = str_or(section, op.field(), "").trim().to_string();
    if weights.is_empty() {
        return Err(match op {
            Op::RemoveBackground => "background removal needs BiRefNet: add its folder on the Models page".into(),
            Op::Upscale => "upscaling needs Real-ESRGAN: add RealESRGAN_x4plus.pth on the Models page".into(),
        });
    }
    let image = match body.get("image").or_else(|| body.get("input_reference")) {
        Some(v) => crate::media::reference_image(v, output_dir, allow_local)?.ok_or("image is required: the picture (a data: URL)")?,
        None => return Err("image is required: the picture (a data: URL)".into()),
    };
    let scale = match body.get("scale") {
        None | Some(Json::Null) => 4,
        Some(v) => v.as_i64().filter(|s| *s == 2 || *s == 4).ok_or("scale must be 2 or 4")?,
    };
    let mut f = vec![
        ("kind".into(), Json::str("picture")),
        ("op".into(), Json::str(if op == Op::Upscale { "upscale" } else { "remove_background" })),
        ("model".into(), Json::str(config::resolve(root, &weights).to_string_lossy())),
        ("image".into(), Json::str(&image)),
        ("device".into(), Json::Int(cfg.get("media").map_or(0, |m| crate::util::int_or(m, "device", 0)))),
        ("output_dir".into(), Json::str(output_dir.to_string_lossy())),
    ];
    let label = match op {
        Op::RemoveBackground => "background removed".to_string(),
        Op::Upscale => {
            f.push(("scale".into(), Json::Int(scale)));
            format!("{scale}× larger")
        }
    };
    Ok((Json::Obj(f), op.model().to_string(), label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::int_or;

    #[test]
    fn a_picture_request_names_its_model_and_takes_the_picture() {
        let dir = std::env::temp_dir().join(format!("oaiy-picture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Json::parse(br#"{"media":{"device":1,"picture":{"background":"/m/BiRefNet","upscaler":""}}}"#).unwrap();
        let png = crate::util::base64_encode(b"\x89PNG\r\n");
        let body = Json::parse(format!(r#"{{"image":"data:image/png;base64,{png}"}}"#).as_bytes()).unwrap();
        let (r, model, label) = picture_request(&cfg, &dir, &dir, &body, false, Op::RemoveBackground).unwrap();
        assert_eq!((model.as_str(), label.as_str()), ("birefnet", "background removed"));
        assert_eq!((str_or(&r, "kind", ""), str_or(&r, "op", ""), int_or(&r, "device", 0)), ("picture", "remove_background", 1));
        assert!(str_or(&r, "model", "").ends_with("BiRefNet") && str_or(&r, "image", "").ends_with(".png"));
        assert!(ready(&cfg, Op::RemoveBackground) && !ready(&cfg, Op::Upscale));
        // Upscaling is not configured here; a bad scale is refused before that matters.
        assert!(picture_request(&cfg, &dir, &dir, &body, false, Op::Upscale).unwrap_err().contains("Real-ESRGAN"));
        let cfg = Json::parse(br#"{"media":{"picture":{"upscaler":"/m/RealESRGAN_x4plus.pth"}}}"#).unwrap();
        let two = Json::parse(format!(r#"{{"image":"data:image/png;base64,{png}","scale":2}}"#).as_bytes()).unwrap();
        let (r, _, label) = picture_request(&cfg, &dir, &dir, &two, false, Op::Upscale).unwrap();
        assert_eq!((int_or(&r, "scale", 0), label.as_str()), (2, "2× larger"));
        let three = Json::parse(format!(r#"{{"image":"data:image/png;base64,{png}","scale":3}}"#).as_bytes()).unwrap();
        assert!(picture_request(&cfg, &dir, &dir, &three, false, Op::Upscale).is_err());
        assert!(picture_request(&cfg, &dir, &dir, &Json::parse(b"{}").unwrap(), false, Op::Upscale).unwrap_err().contains("image"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
