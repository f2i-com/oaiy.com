//! The port checked against the authors' own code: `OAIY_FACEAGE_MIVOLO` names MiVOLO v2's `model.safetensors` and
//! `OAIY_FACEAGE_REF` a folder of reference tensors made by running their code on the same weights (see
//! `docs/FACE_AGE.md`). Without them these tests say so and pass.
use super::mivolo::MiVolo;
use super::prep::{PixelBox, Rgb};
use candle_core::{Device, Tensor};
use std::path::PathBuf;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("OAIY_FACEAGE_MIVOLO").map(PathBuf::from)?;
    let reference = std::env::var_os("OAIY_FACEAGE_REF").map(PathBuf::from)?;
    (model.is_file() && reference.is_dir()).then_some((model, reference))
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a.len(), b.len(), "sizes differ");
    a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0., f32::max)
}

#[test]
fn mivolo_matches_the_authors_code_part_by_part() {
    let Some((model, reference)) = paths() else {
        eprintln!("skipped: set OAIY_FACEAGE_MIVOLO and OAIY_FACEAGE_REF");
        return;
    };
    let dev = Device::Cpu;
    let net = MiVolo::load(&model, &dev).unwrap();
    let r = candle_core::safetensors::load(reference.join("mivolo_ref.safetensors"), &dev).unwrap();
    let input = &r["input"];
    let mut trace = super::mivolo::Trace::default();
    let head = net.head(&input.narrow(1, 0, 3).unwrap(), &input.narrow(1, 3, 3).unwrap(), Some(&mut trace)).unwrap();
    for (name, got) in &trace.parts {
        let want = &r[name.as_str()];
        let scale = want.abs().unwrap().max_keepdim(0).unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1.);
        let diff = max_diff(got, want);
        eprintln!("{name}: max difference {diff:.2e} (largest value {scale:.2})");
        assert!(diff / scale < 2e-3, "{name} differs by {diff}");
    }
    let diff = max_diff(&head, &r["head"]);
    assert!(diff < 1e-3, "the head differs by {diff}");
    let age = net.ages(&input.narrow(1, 0, 3).unwrap(), &input.narrow(1, 3, 3).unwrap()).unwrap()[0];
    let want = r["age"].flatten_all().unwrap().to_vec1::<f32>().unwrap()[0] as f64;
    eprintln!("age {age:.3}, reference {want:.3}");
    assert!((age - want).abs() < 0.05);
}

#[test]
fn mivolo_preprocessing_and_age_match_the_authors_code() {
    let Some((model, reference)) = paths() else {
        eprintln!("skipped: set OAIY_FACEAGE_MIVOLO and OAIY_FACEAGE_REF");
        return;
    };
    let dev = Device::Cpu;
    let raw = std::fs::read(reference.join("pre_image_rgb.raw")).unwrap();
    let image = Rgb::new(640, 480, raw).unwrap();
    let face_box = PixelBox { x1: 250, y1: 120, x2: 410, y2: 330 };
    let face = image.crop(face_box).unwrap().letterbox(384).normalised(&dev).unwrap();
    let body = image.without(face_box).letterbox(384).normalised(&dev).unwrap();
    let r = candle_core::safetensors::load(reference.join("mivolo_pre.safetensors"), &dev).unwrap();
    // OpenCV's resize works in fixed point: a pixel may be one level off, which is 1/255 over the spread here.
    let (df, db) = (max_diff(&face, &r["faces"]), max_diff(&body, &r["bodies"]));
    eprintln!("face input differs by at most {df:.4}, body by {db:.4}");
    assert!(df < 0.02 && db < 0.02);
    let net = MiVolo::load(&model, &dev).unwrap();
    let age = net.ages(&face, &body).unwrap()[0];
    let head = r["head"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = head[2] as f64 * 122. + 61.;
    eprintln!("age {age:.3}, reference {want:.3}");
    assert!((age - want).abs() < 0.5, "{age} against {want}");
}
