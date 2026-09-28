//! oaiy-image against Pillow: every test image decodes to the pixels
//! `Image.open(f).convert("RGB")` gives, and `ImageOps.pad` of the
//! preprocessing images matches Pillow's. The files come from
//! tools/dsv41/vision_golden.py; the tests skip when they are absent.

use std::path::PathBuf;

use oaiy_engine::json::Json;

fn golden_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("OAIY_VISION_GOLDEN").unwrap_or_else(|_| r"E:\deepseek\golden\vision".into()));
    dir.join("decode").join("manifest.json").exists().then_some(dir)
}

fn manifest(path: PathBuf) -> Vec<Json> {
    let src = std::fs::read(path).expect("manifest");
    match Json::parse(&src).expect("manifest json") {
        Json::Arr(v) => v,
        _ => panic!("manifest is not a list"),
    }
}

fn field<'a>(e: &'a Json, k: &str) -> &'a Json {
    e.get(k).unwrap_or_else(|| panic!("manifest entry has no {k}"))
}

fn num(e: &Json, k: &str) -> usize {
    field(e, k).as_i64().expect("integer") as usize
}

/// (differing bytes, largest difference)
fn diff(a: &[u8], b: &[u8]) -> (usize, u8) {
    a.iter().zip(b).filter(|(x, y)| x != y).fold((0, 0), |(n, m), (x, y)| (n + 1, m.max(x.abs_diff(*y))))
}

#[test]
fn decodes_like_pillow() {
    let Some(dir) = golden_dir() else {
        eprintln!("skipping: no vision golden files");
        return;
    };
    let d = dir.join("decode");
    let mut failed = Vec::new();
    let cases = manifest(d.join("manifest.json"));
    for e in &cases {
        let file = field(e, "file").as_str().expect("file");
        let want = std::fs::read(d.join(format!("{file}.rgb"))).expect("rgb");
        let img = match oaiy_image::decode(&std::fs::read(d.join(file)).expect("image")) {
            Ok(i) => i,
            Err(err) => {
                failed.push(format!("{file}: {err}"));
                continue;
            }
        };
        if (img.width, img.height) != (num(e, "width"), num(e, "height")) {
            failed.push(format!("{file}: size {}x{}", img.width, img.height));
            continue;
        }
        let (n, max) = diff(&img.rgb, &want);
        if n > 0 {
            failed.push(format!("{file}: {n} of {} bytes differ (max {max})", want.len()));
        }
    }
    assert!(failed.is_empty(), "{} of {} images differ from Pillow:\n{}", failed.len(), cases.len(), failed.join("\n"));
    assert!(cases.len() > 100, "only {} cases", cases.len());
}

#[test]
fn pads_like_pillow() {
    let Some(dir) = golden_dir() else {
        eprintln!("skipping: no vision golden files");
        return;
    };
    let d = dir.join("preprocess");
    for e in manifest(d.join("manifest.json")) {
        let name = field(&e, "name").as_str().expect("name");
        let file = field(&e, "file").as_str().expect("file");
        let img = oaiy_image::decode(&std::fs::read(d.join(file)).expect("image")).expect("decode");
        let (w, h) = (num(&e, "best_width"), num(&e, "best_height"));
        let padded = oaiy_image::resize::pad(&img, w, h, [127; 3]);
        let want = std::fs::read(d.join(format!("{name}.padded"))).expect("padded");
        let (n, max) = diff(&padded.rgb, &want);
        assert_eq!(n, 0, "{name}: {n} of {} bytes differ from ImageOps.pad (max {max})", want.len());
    }
}

/// Truncated and corrupted files return an error or some image, never a
/// panic (overflow checks included when run in debug).
#[test]
fn survives_corrupt_files() {
    let Some(dir) = golden_dir() else {
        eprintln!("skipping: no vision golden files");
        return;
    };
    let d = dir.join("decode");
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for e in manifest(d.join("manifest.json")) {
        let file = field(&e, "file").as_str().expect("file");
        let data = std::fs::read(d.join(file)).expect("image");
        for cut in [1, 2, 8, 20, data.len() / 3, data.len() / 2, data.len() - 3, data.len() - 1] {
            let _ = oaiy_image::decode(&data[..cut.min(data.len())]);
        }
        for _ in 0..40 {
            let mut bad = data.clone();
            for _ in 0..1 + rand() % 4 {
                let i = (rand() % bad.len() as u64) as usize;
                bad[i] ^= 1 << (rand() % 8);
            }
            let _ = oaiy_image::decode(&bad);
        }
    }
}
