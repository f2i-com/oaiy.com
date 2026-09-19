//! Vision preprocessing and the CPU tower against the reference
//! (tools/dsv41/vision_golden.py writes the files; tests skip without them).
//!
//! The tower comparison reads the 1.9 GB of vision weights and runs a ViT on
//! the CPU, so it is ignored by default:
//!   cargo test -p dsv41 --release --test vision_golden -- --ignored --nocapture

use std::path::PathBuf;

use dsv41::config::{Config, VisionConfig};
use dsv41::safetensors::StIndex;
use dsv41::vision::{self, VisionTower};
use nrob::json::Json;

fn golden() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("NROB_VISION_GOLDEN").unwrap_or_else(|_| r"E:\deepseek\golden\vision".into()));
    dir.join("plan.json").exists().then_some(dir)
}

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("DSV41_MODEL").unwrap_or_else(|_| r"E:\deepseek\model".into()))
}

fn config() -> Option<Config> {
    Config::load(&model_dir()).ok()
}

/// The config.json values, for when only the golden files are around.
fn vision_config() -> VisionConfig {
    config().and_then(|c| c.vision).unwrap_or(VisionConfig {
        n_layers: 32,
        dim: 1024,
        n_heads: 16,
        inter_dim: 2816,
        patch_size: 14,
        rope_theta: 10000.0,
        downsample: 3,
        max_tokens: 1024,
        min_pixels: 544 * 544,
        max_wh_ratio: None,
    })
}

fn json(path: PathBuf) -> Json {
    Json::parse(&std::fs::read(path).expect("golden file")).expect("json")
}

fn n(v: &Json) -> usize {
    v.as_i64().expect("integer") as usize
}

#[test]
fn plan_matches_the_reference_sweep() {
    let Some(dir) = golden() else { return };
    let cfg = vision_config();
    let Json::Arr(rows) = json(dir.join("plan.json")) else { panic!("plan.json is not a list") };
    let mut bad = Vec::new();
    for r in &rows {
        let Json::Arr(v) = r else { panic!("row") };
        let (w, h) = (n(&v[0]), n(&v[1]));
        let want = vision::Plan { n_llm_h: n(&v[2]), n_llm_w: n(&v[3]), best_h: n(&v[4]), best_w: n(&v[5]) };
        let got = vision::plan_image_grid(w, h, &cfg);
        if got != want {
            bad.push(format!("{w}x{h}: {got:?} want {want:?}"));
        }
    }
    assert!(bad.is_empty(), "{} of {} sizes differ:\n{}", bad.len(), rows.len(), bad[..bad.len().min(20)].join("\n"));
    assert!(rows.len() > 5000);
}

#[test]
fn patches_match_the_reference_bit_for_bit() {
    let Some(dir) = golden() else { return };
    let cfg = vision_config();
    let d = dir.join("preprocess");
    let Json::Arr(items) = json(d.join("manifest.json")) else { panic!("manifest") };
    for e in &items {
        let name = e.get("name").and_then(Json::as_str).expect("name");
        let file = e.get("file").and_then(Json::as_str).expect("file");
        let p = vision::load_image(&std::fs::read(d.join(file)).expect("image"), &cfg).expect("prepare");
        let dims = (p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w);
        let want_dims = (n(e.get("n_vit_h").unwrap()), n(e.get("n_vit_w").unwrap()), n(e.get("n_llm_h").unwrap()), n(e.get("n_llm_w").unwrap()));
        assert_eq!(dims, want_dims, "{name}: grid");
        let want: Vec<u16> = std::fs::read(d.join(format!("{name}.patches"))).expect("patches").chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let got = vision::bf16_bits(&p.patches);
        assert_eq!(got.len(), want.len(), "{name}: patch count");
        let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{name}: {diff} of {} patch values differ", want.len());
    }
}

/// Relative L2 distance.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += f64::from(x - y).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
#[ignore = "reads the vision weights and runs a ViT on the CPU"]
fn cpu_tower_matches_the_reference() {
    let Some(dir) = golden() else { return };
    let Some(cfg) = config() else { return };
    let t = std::time::Instant::now();
    let tower = VisionTower::load(&model_dir(), &cfg).expect("tower");
    eprintln!("loaded the tower in {:.1}s", t.elapsed().as_secs_f64());
    let feats = StIndex::open_file(&dir.join("features.safetensors")).expect("features");
    let names: Vec<String> = std::env::var("NROB_VISION_IMAGES").unwrap_or_else(|_| "mascot".into()).split(',').map(String::from).collect();
    for name in names {
        let patches = feats.read_f32(&format!("{name}.patches")).expect("patches");
        let dims = feats.read_i64(&format!("{name}.dims")).expect("dims");
        let [n_vit_h, n_vit_w, n_llm_h, n_llm_w] = dims[..] else { panic!("dims") };
        let img = vision::Prepared { patches, n_vit_h: n_vit_h as usize, n_vit_w: n_vit_w as usize, n_llm_h: n_llm_h as usize, n_llm_w: n_llm_w as usize };
        let n = img.n_patches();
        let t = std::time::Instant::now();
        let x = tower.patch_embed(&img);
        let e = rel_l2(&x, &feats.read_f32(&format!("{name}.patch_embed")).unwrap());
        eprintln!("{name}: patch_embed rel-L2 {e:.2e}");
        assert!(e < 5e-3, "{name}: patch_embed {e}");
        let (cos, sin) = vision::rope_tables(img.n_vit_h, img.n_vit_w, tower.cfg.head_dim() / 2, tower.cfg.rope_theta);
        // block 0 from the reference's own input isolates the block
        let ref_x = feats.read_f32(&format!("{name}.patch_embed")).unwrap();
        let b0 = tower.block(0, &ref_x, n, &cos, &sin);
        let e = rel_l2(&b0, &feats.read_f32(&format!("{name}.block0")).unwrap());
        eprintln!("{name}: block 0 rel-L2 {e:.2e}");
        assert!(e < 1e-2, "{name}: block 0 {e}");
        if std::env::var("NROB_VISION_BLOCKS").is_ok() {
            // each block from the reference's own input: where drift comes from
            let mut prev = ref_x.clone();
            for i in 0..tower.cfg.n_layers {
                let Ok(want) = feats.read_f32(&format!("{name}.block{i:02}.out")) else { break };
                let got = tower.block(i, &prev, n, &cos, &sin);
                let rms = (want.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
                let maxabs = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                eprintln!("{name}: block {i:2} rel-L2 {:.2e}  (rms {rms:.1}, max {maxabs:.0})", rel_l2(&got, &want));
                prev = want;
            }
        }
        let vit = tower.vit(&img);
        let e = rel_l2(&vit, &feats.read_f32(&format!("{name}.vit")).unwrap());
        eprintln!("{name}: ViT rel-L2 {e:.2e} ({:.1}s)", t.elapsed().as_secs_f64());
        // bf16 noise compounds through the blocks that grow outlier
        // activations (block 12 on): torch's own CPU bf16 run is 3.2% from
        // this GPU golden, exact fp32 4.8%; each block alone is ~1e-3
        assert!(e < 1e-1, "{name}: vit {e}");
        // the aligner from the reference's ViT output isolates it
        let ref_vit = feats.read_f32(&format!("{name}.vit")).unwrap();
        let al = tower.align(&ref_vit, &img);
        let e = rel_l2(&al, &feats.read_f32(&format!("{name}.aligner")).unwrap());
        eprintln!("{name}: aligner rel-L2 {e:.2e}");
        assert!(e < 1e-2, "{name}: aligner {e}");
        let full = tower.align(&vit, &img);
        let e = rel_l2(&full, &feats.read_f32(&format!("{name}.aligner")).unwrap());
        eprintln!("{name}: end to end rel-L2 {e:.2e}");
        // torch: CPU bf16 1.8%, fp32 2.8% from the golden
        assert!(e < 6e-2, "{name}: end to end {e}");
    }
}
