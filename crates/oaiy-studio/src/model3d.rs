//! 3D models from a picture (Pixal3D): requests for the worker, from `/v1/3d/models` jobs.
//!
//! A model entry (`media.model3d.models.<name>`) names the Pixal3D folder
//! (`path`), DINOv3 ViT-L/16 (`dino`) and NAF's weights (`naf`), and may name
//! BiRefNet (`matte`: cuts the object out of any background) and Real-ESRGAN
//! x4plus's weights (`upscaler`: enlarges a small picture first).
use crate::config;
use crate::util::{bool_or, int_or, str_or};
use oaiy_engine::json::Json;
use std::path::Path;

pub const DEFAULT_FACES: i64 = 200_000;

fn section(cfg: &Json) -> Result<&Json, String> {
    let s = cfg.get("media").and_then(|m| m.get("model3d")).ok_or("no 3D model section")?;
    if !bool_or(s, "enabled", true) {
        return Err("3D models are disabled".into());
    }
    Ok(s)
}

fn number(body: &Json, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| match body.get(k)? {
        Json::Str(s) => s.trim().parse().ok(),
        v => v.as_f64(),
    })
}

/// A 3D model request as the worker's, with the model name and a label for the job list.
///
/// The picture comes from `image` (or `input_reference`): a `data:` URL, or a local path
/// from this machine; `resolution` is 1024 (default) or 1536; `faces` is the
/// simplified mesh's triangle budget; `fov_degrees` the camera the picture was taken with;
/// `texture_size` the baked textures' size (2048, or 0 for vertex colours only).
pub fn model3d_request(cfg: &Json, root: &Path, output_dir: &Path, body: &Json, allow_local: bool) -> Result<(Json, String, String), String> {
    let section = section(cfg)?;
    let (name, model) = crate::media::pick_model(section, body, "3D")?;
    let folder = |key: &str, what: &str| -> Result<String, String> {
        let p = str_or(model, key, "").trim().to_string();
        if p.is_empty() {
            return Err(format!("3D model {name} has no {what} ({key}); add it on the Models page"));
        }
        Ok(config::resolve(root, &p).to_string_lossy().into_owned())
    };
    let path = folder("path", "Pixal3D folder")?;
    let dino = folder("dino", "DINOv3 ViT-L/16 folder")?;
    let naf = folder("naf", "NAF weights (naf_release.pth)")?;
    let helper = |key: &str| -> Option<String> {
        let p = str_or(model, key, "").trim().to_string();
        (!p.is_empty()).then(|| config::resolve(root, &p).to_string_lossy().into_owned())
    };
    let image = match body.get("image").or_else(|| body.get("input_reference")) {
        Some(v) => crate::media::reference_image(v, output_dir, allow_local)?.ok_or("image is required: the picture of the object")?,
        None => return Err("image is required: the picture of the object (a data: URL)".into()),
    };
    let resolution = number(body, &["resolution"]).map(|v| v as i64).unwrap_or_else(|| int_or(model, "resolution", 1024));
    if resolution != 1024 && resolution != 1536 {
        return Err("resolution must be 1024 or 1536".into());
    }
    let faces = number(body, &["faces", "max_faces"]).map(|v| v as i64).unwrap_or_else(|| int_or(model, "faces", DEFAULT_FACES));
    if !(1_000..=2_000_000).contains(&faces) {
        return Err("faces must be 1000 to 2000000".into());
    }
    let fov = number(body, &["fov_degrees", "fov"]).or_else(|| model.get("fov_degrees").and_then(Json::as_f64)).unwrap_or(30.);
    if !(5.0..=120.0).contains(&fov) {
        return Err("fov_degrees must be 5 to 120".into());
    }
    let seed = match body.get("seed") {
        None | Some(Json::Null) => (crate::util::now_millis() % 2_147_483_647) as i64,
        Some(v) => v.as_i64().filter(|n| *n >= 0).ok_or("seed must be a nonnegative integer")?,
    };
    let mut f = vec![
        ("kind".into(), Json::str("model3d")),
        ("model_dir".into(), Json::str(&path)),
        ("dino_dir".into(), Json::str(&dino)),
        ("naf".into(), Json::str(&naf)),
        ("image".into(), Json::str(&image)),
        ("resolution".into(), Json::Int(resolution)),
        ("faces".into(), Json::Int(faces)),
        ("fov_degrees".into(), Json::Num(fov)),
        ("seed".into(), Json::Int(seed)),
        ("device".into(), Json::Int(int_or(cfg.get("media").unwrap_or(&Json::Null), "device", 0))),
        ("output_dir".into(), Json::str(output_dir.to_string_lossy())),
    ];
    // The baked textures' size, or 0 for vertex colours only.
    let texture_size = number(body, &["texture_size"]).map(|v| v as i64).unwrap_or_else(|| int_or(model, "texture_size", 2048));
    if ![0, 512, 1024, 2048, 4096].contains(&texture_size) {
        return Err("texture_size must be 0 (vertex colours only), 512, 1024, 2048 or 4096".into());
    }
    f.push(("texture_size".into(), Json::Int(texture_size)));
    for key in ["matte", "upscaler"] {
        if let Some(p) = helper(key) {
            f.push((key.into(), Json::str(&p)));
        }
    }
    if let Some(steps) = body.get("steps").and_then(Json::as_i64) {
        if !(1..=50).contains(&steps) {
            return Err("steps must be 1 to 50".into());
        }
        f.push(("steps".into(), Json::Int(steps)));
    }
    Ok((Json::Obj(f), name, format!("{resolution}, {} faces", faces / 1000 * 1000)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_names_every_part_and_takes_the_picture() {
        let dir = std::env::temp_dir().join(format!("oaiy-3d-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Json::parse(br#"{"media":{"device":1,"model3d":{"default_model":"pixal3d","models":{"pixal3d":{"path":"/m/Pixal3D","dino":"/m/dino","naf":"/m/naf.pth","matte":"/m/BiRefNet"}}}}}"#).unwrap();
        let png = crate::util::base64_encode(b"\x89PNG\r\n");
        let body = Json::parse(format!(r#"{{"image":"data:image/png;base64,{png}","seed":5,"faces":50000}}"#).as_bytes()).unwrap();
        let (r, name, label) = model3d_request(&cfg, &dir, &dir, &body, false).unwrap();
        assert_eq!(name, "pixal3d");
        assert_eq!(label, "1024, 50000 faces");
        assert_eq!(str_or(&r, "kind", ""), "model3d");
        assert!(str_or(&r, "image", "").ends_with(".png"));
        assert_eq!((int_or(&r, "seed", 0), int_or(&r, "device", 0), int_or(&r, "faces", 0)), (5, 1, 50000));
        // Every part is needed, and a picture.
        let missing = Json::parse(br#"{"media":{"model3d":{"models":{"p":{"path":"/m/Pixal3D"}}}}}"#).unwrap();
        assert!(model3d_request(&missing, &dir, &dir, &body, false).unwrap_err().contains("DINOv3"));
        assert!(model3d_request(&cfg, &dir, &dir, &Json::parse(b"{}").unwrap(), false).unwrap_err().contains("image"));
        assert!(model3d_request(&cfg, &dir, &dir, &Json::parse(format!(r#"{{"image":"data:image/png;base64,{png}","resolution":2048}}"#).as_bytes()).unwrap(), false).is_err());
        assert_eq!(int_or(&r, "texture_size", 0), 2048);
        // BiRefNet goes with it; Real-ESRGAN, not configured, does not.
        assert!(str_or(&r, "matte", "").ends_with("BiRefNet") && r.get("upscaler").is_none());
        assert!(model3d_request(&cfg, &dir, &dir, &Json::parse(format!(r#"{{"image":"data:image/png;base64,{png}","texture_size":3000}}"#).as_bytes()).unwrap(), false).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
