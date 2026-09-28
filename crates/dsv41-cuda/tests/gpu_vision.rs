//! The GPU vision tower against the reference's features
//! (tools/dsv41/vision_golden.py) and each ViT block against the CPU tower:
//!   cargo test -p dsv41-cuda --release --test gpu_vision -- --ignored --nocapture
//! (`DSV41_CUDA_DEVICES=1` picks the device; `OAIY_VISION_IMAGES` the images.)

use std::path::PathBuf;
use std::time::Instant;

use dsv41::config::Config;
use dsv41::safetensors::StIndex;
use dsv41::vision::{self, VisionTower};
use dsv41_cuda::{Gpu, GpuVision};

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("DSV41_MODEL").unwrap_or_else(|_| r"E:\deepseek\model".into()))
}

fn golden() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("OAIY_VISION_GOLDEN").unwrap_or_else(|_| r"E:\deepseek\golden\vision".into()));
    dir.join("features.safetensors").exists().then_some(dir)
}

fn device() -> usize {
    std::env::var("DSV41_CUDA_DEVICES").ok().and_then(|v| v.split(',').next().and_then(|d| d.trim().parse().ok())).unwrap_or(0)
}

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
#[ignore = "needs a GPU, the checkpoint and the vision golden files"]
fn gpu_tower_matches_the_reference() {
    let Some(dir) = golden() else { return };
    let cfg = Config::load(&model_dir()).expect("config");
    let g = Gpu::new(device()).expect("gpu");
    let idx = StIndex::open(&model_dir()).expect("index");
    let t = Instant::now();
    let tower = GpuVision::load(&g, &idx, &cfg).expect("tower");
    eprintln!("loaded {:.0} MB of vision weights in {:.1}s", tower.bytes() as f64 / 1e6, t.elapsed().as_secs_f64());
    let cpu = VisionTower::load(&model_dir(), &cfg).expect("cpu tower");
    let feats = StIndex::open_file(&dir.join("features.safetensors")).expect("features");
    let names = std::env::var("OAIY_VISION_IMAGES").unwrap_or_else(|_| "mascot,wide,odd,tiny".into());
    for name in names.split(',') {
        let patches = feats.read_f32(&format!("{name}.patches")).expect("patches");
        let dims = feats.read_i64(&format!("{name}.dims")).expect("dims");
        let [vh, vw, lh, lw] = dims[..] else { panic!("dims") };
        let img = vision::Prepared { patches, n_vit_h: vh as usize, n_vit_w: vw as usize, n_llm_h: lh as usize, n_llm_w: lw as usize };
        let n = img.n_patches();
        let want = |k: &str| feats.read_f32(&format!("{name}.{k}")).expect("golden tensor");

        let e = rel_l2(&g.download(&tower.patch_embed(&g, &img).unwrap()).unwrap(), &want("patch_embed"));
        eprintln!("{name}: patch_embed rel-L2 {e:.2e}");
        assert!(e < 1e-3, "{name}: patch_embed {e}");

        // block 0 from the reference's input, against the reference and the CPU tower
        let x = want("patch_embed");
        let rope = tower.rope(&g, &img).unwrap();
        let b0 = g.download(&tower.block(&g, 0, &g.upload(&x).unwrap(), n, &rope).unwrap()).unwrap();
        let (c, s) = vision::rope_tables(img.n_vit_h, img.n_vit_w, cfg.vision.as_ref().unwrap().head_dim() / 2, 10000.0);
        let b0_cpu = cpu.block(0, &x, n, &c, &s);
        let (e_ref, e_cpu) = (rel_l2(&b0, &want("block0")), rel_l2(&b0, &b0_cpu));
        eprintln!("{name}: block 0 rel-L2 {e_ref:.2e} to the reference, {e_cpu:.2e} to the CPU tower");
        assert!(e_ref < 1e-2 && e_cpu < 1e-2, "{name}: block 0");

        g.sync().unwrap();
        let t = Instant::now();
        let vit = g.download(&tower.vit(&g, &img).unwrap()).unwrap();
        let vit_s = t.elapsed().as_secs_f64();
        let e = rel_l2(&vit, &want("vit"));
        eprintln!("{name}: ViT rel-L2 {e:.2e} ({n} patches in {:.0} ms)", vit_s * 1e3);
        // as the CPU tower: bf16 noise compounds past block 12's outliers
        assert!(e < 1e-1, "{name}: vit {e}");
        let al = tower.align(&g, &want("vit"), &img).unwrap();
        let e = rel_l2(&al, &want("aligner"));
        eprintln!("{name}: aligner rel-L2 {e:.2e}");
        assert!(e < 1e-2, "{name}: aligner {e}");
        let t = Instant::now();
        let full = tower.encode(&g, &img).unwrap();
        let enc_s = t.elapsed().as_secs_f64();
        let e = rel_l2(&full, &want("aligner"));
        eprintln!("{name}: end to end rel-L2 {e:.2e} (encode {:.0} ms)", enc_s * 1e3);
        assert!(e < 6e-2, "{name}: end to end {e}");
        let span = vision::span_rows(&img, &full, &tower.start, &tower.newline, &tower.end);
        assert_eq!(span.len(), img.n_tokens() * cfg.dim);
    }
}
