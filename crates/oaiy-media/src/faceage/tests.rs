//! The port checked against the authors' own code: `OAIY_FACEAGE_MIVOLO` names MiVOLO v2's `model.safetensors` and
//! `OAIY_FACEAGE_REF` a folder of reference tensors made by running their code on the same weights (see
//! `tools/faceage`). Without them these tests say so and pass.
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

fn encode64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { A[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

#[test]
fn every_face_found_is_blacked_out_of_the_body_picture() {
    let img = Rgb { width: 4, height: 4, data: vec![200; 48] };
    let face = |x: f32, y: f32| super::yunet::Face { x, y, w: 1., h: 1., landmarks: [0.; 10], score: 0.95 };
    let out = super::without_faces(&img, &[face(0., 0.), face(3., 3.)]);
    let pixel = |x: usize, y: usize| out.data[(y * 4 + x) * 3];
    assert_eq!((pixel(0, 0), pixel(3, 3)), (0, 0), "the one in front and the one behind");
    assert_eq!(pixel(1, 1), 200, "and nothing else");
}

#[test]
fn a_large_picture_is_read_at_the_read_size_and_a_check_frame_as_it_is() {
    let picture = |w: usize, h: usize| Rgb { width: w, height: h, data: vec![90; w * h * 3] };
    let large = super::to_read_size(picture(4096, 2048));
    assert_eq!((large.width, large.height), (super::READ_SIDE, super::READ_SIDE / 2));
    assert_eq!(large.data.len(), large.width * large.height * 3);
    let tall = super::to_read_size(picture(1000, 3000));
    assert_eq!((tall.width, tall.height), (341, super::READ_SIDE));
    // A check's frames are 360 pixels wide: read as they came, pixel for pixel.
    assert_eq!(super::to_read_size(picture(360, 270)), picture(360, 270));
    assert_eq!(super::to_read_size(picture(super::READ_SIDE, 10)), picture(super::READ_SIDE, 10));
}

#[test]
fn a_picture_that_says_it_is_too_large_is_refused_from_its_header() {
    // A flat PNG a few KB long that is 5000 pixels wide: refused as too large, not decoded and measured after.
    let mut png = Vec::new();
    image::GrayImage::from_pixel(5000, 8, image::Luma([90]))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    assert!(png.len() < 10_000);
    assert_eq!(Rgb::decode(&png).unwrap_err(), format!("a picture larger than {} pixels on a side", super::MAX_SIDE));
    // At the limit it is read, and as RGB.
    let mut ok = Vec::new();
    image::GrayImage::from_pixel(super::MAX_SIDE as u32, 2, image::Luma([90]))
        .write_to(&mut std::io::Cursor::new(&mut ok), image::ImageFormat::Png)
        .unwrap();
    let read = Rgb::decode(&ok).unwrap();
    assert_eq!((read.width, read.height, read.data.len()), (super::MAX_SIDE, 2, super::MAX_SIDE * 2 * 3));
}

#[test]
fn a_picture_shrunk_to_the_read_size_keeps_what_falls_between_the_pixels_read() {
    // A dark picture with a one-pixel bright line at a column a two-tap shrink by four never reads (4n: it reads 4n+1, 4n+2).
    let (w, h) = (4096, 64);
    let mut img = Rgb { width: w, height: h, data: vec![10; w * h * 3] };
    for y in 0..h {
        img.data[(y * w + 1000) * 3..(y * w + 1001) * 3].fill(250);
    }
    let read = super::to_read_size(img.clone());
    assert_eq!((read.width, read.height), (super::READ_SIDE, 16));
    let at = |x: usize, y: usize| read.data[(y * read.width + x) * 3];
    // The line's quarter of its pixel's area: (250 + 3 × 10) / 4.
    assert_eq!((0..16).map(|y| at(250, y)).collect::<Vec<_>>(), vec![70; 16], "the line is still there");
    assert_eq!((at(249, 0), at(251, 0), at(0, 15)), (10, 10, 10), "and nothing beside it changed");
    // Brightness kept: the mean of the picture is the mean of what is read.
    let mean = |p: &Rgb| p.data.iter().map(|&v| f64::from(v)).sum::<f64>() / p.data.len() as f64;
    assert!((mean(&img) - mean(&read)).abs() < 0.5, "{} against {}", mean(&img), mean(&read));
    // A size that does not divide: every pixel read whole or in part, the flat picture flat.
    let odd = Rgb { width: 3001, height: 7, data: vec![123; 3001 * 7 * 3] }.resize_area(1024, 2);
    assert!(odd.data.iter().all(|&v| v == 123));
}

#[test]
fn a_box_is_blacked_out_in_place_and_held_to_the_picture() {
    let mut img = Rgb { width: 4, height: 3, data: vec![200; 36] };
    img.black_out(super::prep::PixelBox { x1: 2, y1: 1, x2: 99, y2: 99 });
    let pixel = |img: &Rgb, x: usize, y: usize| img.data[(y * 4 + x) * 3];
    assert_eq!((pixel(&img, 2, 1), pixel(&img, 3, 2), pixel(&img, 1, 1), pixel(&img, 2, 0)), (0, 0, 200, 200));
    img.black_out(super::prep::PixelBox { x1: 9, y1: 0, x2: 12, y2: 3 });
    assert_eq!(img.data.iter().filter(|&&v| v == 0).count(), 2 * 2 * 3, "a box beside the picture changes nothing");
}

#[test]
fn only_the_faces_in_front_of_the_camera_count() {
    let face = |w: f32, h: f32| super::yunet::Face { x: 0., y: 0., w, h, landmarks: [0.; 10], score: 0.95 };
    let sizes = |faces: Vec<super::yunet::Face>| super::in_front(faces).iter().map(|f| f.w).collect::<Vec<_>>();
    // A webcam frame: the person, a photo on the wall behind (a tenth of the area), someone far back.
    assert_eq!(sizes(vec![face(120., 140.), face(38., 44.), face(30., 36.)]), [120.], "the background is left out");
    // Two people side by side are two faces, as they should be.
    assert_eq!(sizes(vec![face(120., 140.), face(100., 120.)]), [120., 100.]);
    assert!(super::in_front(Vec::new()).is_empty());
    assert!(super::CHECK_SCORE_THRESHOLD < super::yunet::SCORE_THRESHOLD, "a check's small frames are read less strictly than OpenCV's default");
}

#[test]
fn base64_reads_what_it_is_given() {
    for sample in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar", &[0, 255, 128, 7, 9]] {
        assert_eq!(super::base64(&encode64(sample)).unwrap(), sample);
    }
    assert_eq!(super::base64("Zm9v\nYmFy").unwrap(), b"foobar");
    assert!(super::base64("Zm9v!").is_none());
}

#[test]
fn a_job_is_refused_without_its_models_or_with_too_many_pictures() {
    use oaiy_engine::json::Json;
    let dir = std::env::temp_dir();
    let missing = Json::obj([("kind", Json::str("face-age")), ("detector", Json::str("no-such-file.onnx")), ("model", Json::str("no-such.safetensors")), ("images", Json::Arr(vec![Json::str(encode64(b"x"))]))]);
    assert!(super::Request::parse(&missing).err().unwrap().contains("not there"));
    let exe = std::env::current_exe().unwrap();
    let many = Json::obj([
        ("detector", Json::str(exe.to_string_lossy())),
        ("model", Json::str(exe.to_string_lossy())),
        ("images", Json::Arr((0..17).map(|_| Json::str(encode64(b"x"))).collect())),
    ]);
    assert!(super::Request::parse(&many).err().unwrap().contains("1 to 16"));
    let none = Json::obj([("detector", Json::str(dir.to_string_lossy())), ("images", Json::Arr(vec![]))]);
    assert!(super::Request::parse(&none).is_err());
}

#[test]
fn the_job_reads_a_picture_with_no_face_as_no_face() {
    let (Some((model, reference)), Some((detector, _))) = (paths(), yunet_paths()) else {
        eprintln!("skipped: set OAIY_FACEAGE_MIVOLO, OAIY_FACEAGE_YUNET and OAIY_FACEAGE_REF");
        return;
    };
    use oaiy_engine::json::Json;
    let raw = std::fs::read(reference.join("pre_image_rgb.raw")).unwrap();
    let mut png = Vec::new();
    image::RgbImage::from_raw(640, 480, raw).unwrap().write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let request = Json::obj([
        ("kind", Json::str("face-age")),
        ("detector", Json::str(detector.to_string_lossy())),
        ("model", Json::str(model.to_string_lossy())),
        ("images", Json::Arr(vec![Json::str(format!("data:image/png;base64,{}", encode64(&png))), Json::str(encode64(&png))])),
    ]);
    let r = super::Request::parse(&request).unwrap();
    let answer = super::run(&r).unwrap();
    eprintln!("{}", answer.to_json());
    let images = answer.get("images").and_then(Json::as_array).unwrap();
    assert_eq!(images.len(), 2);
    for image in images {
        assert_eq!(image.get("faces").and_then(Json::as_f64), Some(0.));
        assert!(image.get("age").is_none());
    }
    // Only what is said here leaves the job: no gender, no box, no picture.
    let text = answer.to_json();
    for word in ["gender", "male", "box", "base64", "landmark"] {
        assert!(!text.contains(word), "{word} in {text}");
    }
}

#[test]
fn the_job_reads_one_faces_age_as_the_authors_code_does_two_pictures_at_once() {
    let (Some((model, reference)), Some((detector, _))) = (paths(), yunet_paths()) else {
        eprintln!("skipped: set OAIY_FACEAGE_MIVOLO, OAIY_FACEAGE_YUNET and OAIY_FACEAGE_REF");
        return;
    };
    use oaiy_engine::json::Json;
    // The test picture has no face, but at 0.13 YuNet keeps exactly one candidate there (OpenCV's too: 0.144, the next
    // 0.117), so the job's one-face path runs: the box, the crop, the body with it blacked out, MiVOLO on a batch.
    let raw = std::fs::read(reference.join("pre_image_rgb.raw")).unwrap();
    let mut png = Vec::new();
    image::RgbImage::from_raw(640, 480, raw).unwrap().write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let request = Json::obj([
        ("kind", Json::str("face-age")),
        ("detector", Json::str(detector.to_string_lossy())),
        ("model", Json::str(model.to_string_lossy())),
        ("images", Json::Arr(vec![Json::str(encode64(&png)), Json::str(encode64(&png))])),
        ("score_threshold", Json::Num(0.13)),
    ]);
    let answer = super::run(&super::Request::parse(&request).unwrap()).unwrap();
    eprintln!("{}", answer.to_json());
    let images = answer.get("images").and_then(Json::as_array).unwrap();
    let ages: Vec<f64> = images
        .iter()
        .map(|i| {
            assert_eq!(i.get("faces").and_then(Json::as_f64), Some(1.));
            i.get("age").and_then(Json::as_f64).expect("an age for one face")
        })
        .collect();
    assert_eq!(ages[0], ages[1], "the same picture twice in one batch reads the same");
    let r = candle_core::safetensors::load(reference.join("mivolo_job.safetensors"), &Device::Cpu).unwrap();
    let want = r["age"].flatten_all().unwrap().to_vec1::<f32>().unwrap()[0] as f64;
    eprintln!("age {} against the authors' code's {want:.2}", ages[0]);
    assert!((ages[0] - want).abs() < 0.5, "{} against {want}", ages[0]);
}

fn yunet_paths() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("OAIY_FACEAGE_YUNET").map(PathBuf::from)?;
    let reference = std::env::var_os("OAIY_FACEAGE_REF").map(PathBuf::from)?;
    (model.is_file() && reference.is_dir()).then_some((model, reference))
}

#[test]
fn yunet_runs_the_published_graph_as_onnxruntime_does() {
    let Some((model, reference)) = yunet_paths() else {
        eprintln!("skipped: set OAIY_FACEAGE_YUNET and OAIY_FACEAGE_REF");
        return;
    };
    let dev = Device::Cpu;
    let net = super::yunet::YuNet::load(&model, &dev).unwrap();
    let r = candle_core::safetensors::load(reference.join("yunet_raw.safetensors"), &dev).unwrap();
    let got = net.raw(&r["input"]).unwrap();
    for (name, want) in r.iter().filter(|(n, _)| n.as_str() != "input") {
        let diff = max_diff(&got[name], want);
        eprintln!("{name}: max difference {diff:.2e}");
        assert!(diff < 1e-3, "{name} differs by {diff}");
    }
}

#[test]
fn yunet_finds_what_opencv_finds() {
    let Some((model, reference)) = yunet_paths() else {
        eprintln!("skipped: set OAIY_FACEAGE_YUNET and OAIY_FACEAGE_REF");
        return;
    };
    let dev = Device::Cpu;
    let net = super::yunet::YuNet::load(&model, &dev).unwrap();
    let image = Rgb::new(640, 480, std::fs::read(reference.join("pre_image_rgb.raw")).unwrap()).unwrap();
    let lines = std::fs::read_to_string(reference.join("yunet_cv2.txt")).unwrap();
    for threshold in [0.01f32, 0.05] {
        let want: Vec<Vec<f32>> = lines
            .lines()
            .map(|l| l.split(' ').map(|v| v.parse::<f32>().unwrap()).collect::<Vec<_>>())
            .filter(|v| (v[0] - threshold).abs() < 1e-6)
            .map(|v| v[1..].to_vec())
            .collect();
        let got = net.detect(&image, threshold, super::yunet::NMS_THRESHOLD, super::yunet::TOP_K).unwrap();
        eprintln!("threshold {threshold}: OpenCV {} faces, the port {}", want.len(), got.len());
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            let mine = [g.x, g.y, g.w, g.h];
            for k in 0..4 {
                assert!((mine[k] - w[k]).abs() < 0.5, "box {mine:?} against {:?}", &w[..4]);
            }
            for k in 0..10 {
                assert!((g.landmarks[k] - w[4 + k]).abs() < 0.5);
            }
            assert!((g.score - w[14]).abs() < 1e-3, "score {} against {}", g.score, w[14]);
        }
    }
    // The picture has no face: nothing at OpenCV's own threshold.
    assert!(net.detect(&image, super::yunet::SCORE_THRESHOLD, super::yunet::NMS_THRESHOLD, super::yunet::TOP_K).unwrap().is_empty());
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
