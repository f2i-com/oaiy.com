//! A person's age from camera pictures, for an age check: a face detector (YuNet) finds the faces, and an age model
//! (MiVOLO v2) reads the one face there should be, with the rest of the picture as the body (the face blacked out, as
//! MiVOLO cuts a face out of its body crop). One job a process: the models are loaded for it and gone when it ends.
//!
//! The job (`kind: face-age`) takes the two models' files and the pictures, inline as base64; it answers, for each
//! picture, how many faces were found in front of the camera and, when there is exactly one, its age and which way it points (`pose`, by
//! geometry from the detector's landmarks: no model judges it). Nothing else of a picture leaves it
//! (the model's gender output is not read), and nothing of one is written anywhere.
pub mod mivolo;
pub mod onnx;
pub mod pose;
pub mod prep;
pub mod yunet;

#[cfg(test)]
mod tests;

use candle_core::{Device, Result, Tensor};
use oaiy_engine::json::Json;
use prep::{PixelBox, Rgb};
use std::path::PathBuf;

/// At most this many pictures a job, each at most this many pixels on a side. A job's request is at most 2 MiB as a whole
/// (main.rs), which is what holds a picture's encoded size.
pub const MAX_PICTURES: usize = 16;
const MAX_SIDE: usize = 4096;
/// A picture larger than this on its longer side is read at this size. The detector keeps every layer of its pass, and a
/// flat 4096-pixel picture, a few hundred KB as a JPEG, took 1.9 GB of the computer's memory to read (2048: 1.2 GB); at
/// this size a picture takes a few hundred MB whatever it was sent as. A check's frames are 360 pixels wide, and a face in
/// front of the camera is large enough at this size for both models.
pub const READ_SIDE: usize = 1024;

pub struct Request {
    pub detector: PathBuf,
    pub model: PathBuf,
    pub pictures: Vec<Vec<u8>>,
    /// The detector's threshold (CHECK_SCORE_THRESHOLD when not given).
    pub score_threshold: f32,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let path = |key: &str| -> std::result::Result<PathBuf, String> {
            let p = PathBuf::from(j.get(key).and_then(Json::as_str).ok_or(format!("face-age: no {key}"))?);
            if p.is_file() { Ok(p) } else { Err(format!("face-age: the {key} file {} is not there", p.display())) }
        };
        let images = j.get("images").and_then(Json::as_array).ok_or("face-age: no images")?;
        if images.is_empty() || images.len() > MAX_PICTURES {
            return Err(format!("face-age: 1 to {MAX_PICTURES} pictures a job"));
        }
        let pictures = images
            .iter()
            .map(|v| {
                let text = v.as_str().ok_or("face-age: a picture that is not base64 text")?;
                let text = text.split_once(";base64,").map_or(text, |(_, b)| b);
                base64(text).ok_or_else(|| "face-age: a picture that is not base64".to_string())
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        let score_threshold = j.get("score_threshold").and_then(Json::as_f64).map(|v| v as f32).unwrap_or(CHECK_SCORE_THRESHOLD);
        if !(0.0..=1.0).contains(&score_threshold) {
            return Err("face-age: score_threshold is 0 to 1".into());
        }
        Ok(Self { detector: path("detector")?, model: path("model")?, pictures, score_threshold })
    }
}

/// Standard base64 (padding optional, white space ignored).
pub fn base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// The detector's threshold for a check's pictures, below OpenCV's 0.9: they are small webcam frames (360 pixels wide
/// to fit a relay turn), often dim, and at 0.9 a real face in them can go unseen. Faces in the background are left out by
/// size instead (in_front).
pub const CHECK_SCORE_THRESHOLD: f32 = 0.8;

/// A face counts when it is at least this share of the largest face's area: a photo on the wall behind or someone far
/// back is not the person in front of the camera, whom the check is about. Two people side by side both count.
pub const MIN_FACE_SHARE: f32 = 0.2;

/// The faces in front of the camera, of those found.
pub fn in_front(found: Vec<yunet::Face>) -> Vec<yunet::Face> {
    let largest = found.iter().map(|f| f.w * f.h).fold(0f32, f32::max);
    found.into_iter().filter(|f| f.w * f.h >= largest * MIN_FACE_SHARE).collect()
}

/// A found face's box, in the picture's pixels.
fn face_box(face: &yunet::Face, img: &Rgb) -> PixelBox {
    PixelBox {
        x1: face.x.max(0.) as usize,
        y1: face.y.max(0.) as usize,
        x2: ((face.x + face.w).max(0.) as usize).min(img.width),
        y2: ((face.y + face.h).max(0.) as usize).min(img.height),
    }
}

/// The picture with every face found blacked out, those left out of the count too: the model's whole-picture input is
/// for the one person's body, and a parent's face behind a child must not be read into the child's age.
fn without_faces(img: &Rgb, found: &[yunet::Face]) -> Rgb {
    // One copy, blacked out in place: a copy a face was thousands of copies for a picture of thousands of faces.
    let mut out = img.clone();
    for face in found {
        out.black_out(face_box(face, img));
    }
    out
}

/// A picture as it is read: no larger than READ_SIDE on its longer side.
pub fn to_read_size(img: Rgb) -> Rgb {
    let longer = img.width.max(img.height);
    if longer <= READ_SIDE {
        return img;
    }
    let scale = READ_SIDE as f64 / longer as f64;
    let (w, h) = (((img.width as f64 * scale).round() as usize).max(1), ((img.height as f64 * scale).round() as usize).max(1));
    img.resize_linear(w, h)
}

/// What one picture showed.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub faces: usize,
    pub age: Option<f64>,
    /// Which way the one face points (from its landmarks), when there is exactly one.
    pub pose: Option<pose::Pose>,
}

/// Faces and ages for decoded pictures, with the models loaded.
pub fn read(detector: &yunet::YuNet, model: &mivolo::MiVolo, pictures: &[Rgb], score_threshold: f32) -> Result<Vec<Reading>> {
    let dev = model.device().clone();
    let mut readings = Vec::with_capacity(pictures.len());
    let (mut faces, mut bodies, mut which) = (Vec::new(), Vec::new(), Vec::new());
    for (i, img) in pictures.iter().enumerate() {
        let all = detector.detect(img, score_threshold, yunet::NMS_THRESHOLD, yunet::TOP_K)?;
        let found = in_front(all.clone());
        readings.push(Reading { faces: found.len(), age: None, pose: None });
        if let [face] = found.as_slice() {
            readings[i].pose = pose::pose(&face.landmarks);
            let b = face_box(face, img);
            if let Some(crop) = img.crop(b) {
                faces.push(crop.letterbox(mivolo::SIDE).normalised(&dev)?);
                bodies.push(without_faces(img, &all).letterbox(mivolo::SIDE).normalised(&dev)?);
                which.push(i);
            }
        }
    }
    if !which.is_empty() {
        let ages = model.ages(&Tensor::cat(&faces, 0)?, &Tensor::cat(&bodies, 0)?)?;
        for (i, age) in which.into_iter().zip(ages) {
            readings[i].age = Some(age);
        }
    }
    Ok(readings)
}

/// The `face-age` job: loads both models, reads every picture, and answers
/// `{detector, model, images: [{faces, age?}], seconds}`.
pub fn run(r: &Request) -> Result<Json> {
    let started = std::time::Instant::now();
    let pictures = r
        .pictures
        .iter()
        .map(|bytes| {
            let img = Rgb::decode(bytes).map_err(candle_core::Error::Msg)?;
            if img.width > MAX_SIDE || img.height > MAX_SIDE {
                candle_core::bail!("face-age: a picture larger than {MAX_SIDE} pixels on a side");
            }
            Ok(to_read_size(img))
        })
        .collect::<Result<Vec<_>>>()?;
    let dev = Device::Cpu;
    let detector = yunet::YuNet::load(&r.detector, &dev)?;
    let model = mivolo::MiVolo::load(&r.model, &dev)?;
    let readings = read(&detector, &model, &pictures, r.score_threshold)?;
    let images = readings
        .into_iter()
        .map(|reading| {
            let mut fields = vec![("faces".to_string(), Json::Int(reading.faces as i64))];
            if let Some(age) = reading.age {
                fields.push(("age".to_string(), Json::Num((age * 10.).round() / 10.)));
            }
            if let Some(p) = reading.pose {
                fields.push(("yaw".to_string(), Json::Num((p.yaw * 10.).round() / 10.)));
                fields.push(("pitch".to_string(), Json::Num((p.pitch * 10.).round() / 10.)));
            }
            Json::Obj(fields)
        })
        .collect();
    Ok(Json::obj([
        ("detector", Json::str("yunet-2023mar")),
        ("model", Json::str("mivolo-v2")),
        ("images", Json::Arr(images)),
        ("seconds", Json::Num((started.elapsed().as_secs_f64() * 100.).round() / 100.)),
    ]))
}
