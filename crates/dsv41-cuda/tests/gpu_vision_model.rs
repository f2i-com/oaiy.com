//! Images through the whole model, against the oracle's image golden
//! (`tools/dsv41/oracle.py --image E:\deepseek\model\mascot.png
//! --prompt "Describe this image in one sentence." --golden-name
//! golden_image.safetensors`). Heavy, `#[ignore]`d:
//!
//!   cargo test -p dsv41-cuda --release --test gpu_vision_model -- --ignored --nocapture --test-threads=1
//!
//! Environment as tests/gpu_model.rs, plus DSV41_IMAGE (the golden's image).
//!
//! What can be held to the oracle: every layer in isolation (teacher
//! forcing) matches as tightly as text does. A free-running image prompt
//! cannot: one bf16 ulp on 1% of the image rows moves this model's own
//! final logits by ~25% (`gpu_image_prompt_sensitivity`) — router near-ties
//! among the image tokens — which is as far as it lands from the oracle. So
//! the whole-prompt checks are "within that band, same top candidates",
//! and chunked / layered runs must agree with one prefill bit for bit.

use std::path::PathBuf;
use std::time::Instant;

use dsv41::golden::{isolation, PhaseReport};
use dsv41::model::argmax;
use dsv41::safetensors::StIndex;
use dsv41::vision;
use dsv41_cuda::{GpuModel, GpuOptions, ImageSpan};

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

fn setup() -> Option<(StIndex, GpuModel)> {
    let model_dir = env_path("DSV41_MODEL", r"E:\deepseek\model");
    let golden_dir = env_path("DSV41_GOLDEN_DIR", r"E:\deepseek\golden");
    let name = std::env::var("DSV41_GOLDEN_NAME").unwrap_or_else(|_| "golden_image.safetensors".into());
    let (golden, meta) = (golden_dir.join(&name), golden_dir.join("engram_meta.safetensors"));
    if !model_dir.join("config.json").exists() || !golden.exists() || !meta.exists() {
        eprintln!("skipping: checkpoint or golden files not found");
        return None;
    }
    let devices: Vec<usize> = std::env::var("DSV41_CUDA_DEVICES")
        .unwrap_or_else(|_| "1".into())
        .split(',')
        .map(|v| v.trim().parse().expect("DSV41_CUDA_DEVICES: comma-separated ordinals"))
        .collect();
    let opts = GpuOptions {
        devices: devices.clone(),
        max_seq: 1024,
        expert_cache_bytes: 24 << 30,
        direct_io: false,
        vram_expert_bytes: None,
        vram_headroom_bytes: 1 << 30,
        cpu_expert_threads: None,
        vision: true,
        residual_on_device: None,
    };
    let t = Instant::now();
    let model = GpuModel::load(&model_dir, &meta, &opts).expect("GPU model");
    assert!(model.has_vision(), "the vision tower did not load");
    eprintln!("{name}: GPU model with vision on cuda:{devices:?} loaded in {:.1}s", t.elapsed().as_secs_f64());
    Some((StIndex::open_file(&golden).unwrap(), model))
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// The image spans of a prompt: each run of image-typed positions (types
/// >= 0), with its rows taken from `emb` (`[positions][dim]`).
fn spans(types: &[i64], emb: &[f32], d: usize) -> Vec<ImageSpan> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < types.len() {
        if types[i] < 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < types.len() && types[i] >= 0 && !(i > start && types[i] == vision::IMAGE_START as i64) {
            i += 1;
        }
        out.push(ImageSpan { start, rows: emb[start * d..i * d].to_vec() });
    }
    out
}

fn print(r: &PhaseReport) {
    for l in &r.layers {
        eprintln!(
            "  layer {:2}: out p50 {:.1e} p95 {:.1e} max {:.1e}  bf16-exact {:4.1}%  attn p95 {:.1e}  moe p95 {:.1e}  route flips {}",
            l.layer, l.p50, l.p95, l.max, 100.0 * l.bf16_exact, l.attn_p95, l.moe_p95, l.route_flips
        );
    }
    eprintln!("  {} of {} token-routes differ; logits rel-L2 {:.2e}", r.route_flips, r.routes, r.logits_rel);
}

fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}

/// How sensitive this prompt is: a 0.1% nudge to the image rows against
/// the distance to the oracle.
#[test]
#[ignore]
fn gpu_image_prompt_sensitivity() {
    let Some((g, mut model)) = setup() else { return };
    let d = model.cfg.dim;
    let ids: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let types = g.read_i64("prompt_types").unwrap();
    let ref_logits = g.read_f32("prefill.logits").unwrap();
    let sp = spans(&types, &g.read_f32("prefill.embed_vl").unwrap(), d);
    let base = model.forward_with(&ids, 0, &sp).unwrap();
    eprintln!("oracle top-5 {:?}\nours   top-5 {:?}", top(&ref_logits, 5), top(&base, 5));
    eprintln!("ours vs oracle: logits rel-L2 {:.2e}", rel_l2(&base, &ref_logits));
    for (seed, scale) in [(1u64, 0.1f32), (2, 0.1), (3, 0.01)] {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut nudged = sp.clone();
        for v in &mut nudged[0].rows {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            // one bf16 ulp up or down on a `scale` share of the values
            if ((st >> 40) as f32 / (1u64 << 24) as f32) < scale {
                let b = dsv41::formats::f32_to_bf16(*v);
                *v = dsv41::formats::bf16_to_f32(if st & 1 == 0 { b.wrapping_add(1) } else { b.wrapping_sub(1) });
            }
        }
        let l = model.forward_with(&ids, 0, &nudged).unwrap();
        eprintln!("one bf16 ulp on {:.0}% of the image values:", 100.0 * scale); eprintln!("   logits rel-L2 {:.2e} to ours, {:.2e} to the oracle; top-3 {:?}", rel_l2(&l, &base), rel_l2(&l, &ref_logits), top(&l, 3));
    }
}

/// Each layer from the oracle's own input (teacher forcing), with the
/// reference's image rows: where the image path departs, if it does.
#[test]
#[ignore]
fn gpu_image_layers_in_isolation() {
    let Some((g, mut model)) = setup() else { return };
    let d = model.cfg.dim;
    let ids: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let types = g.read_i64("prompt_types").unwrap();
    model.set_trace_images(spans(&types, &g.read_f32("prefill.embed_vl").unwrap(), d));
    let r = isolation(&mut model, &g, "prefill", &ids, 0).unwrap();
    print(&r);
    for l in &r.layers {
        assert!(l.p95 < 3e-2, "layer {}: out p95 {:.2e}", l.layer, l.p95);
    }
    assert!((r.route_flips as f64) < 0.01 * r.routes as f64, "{} of {} routes differ", r.route_flips, r.routes);
    assert!(r.logits_rel < 2e-2, "logits rel-L2 {:.2e}", r.logits_rel);

    if std::env::var("NROB_FREE_RUN").is_err() {
        return;
    }
    // free-running: how the difference grows, text and image tokens apart
    use dsv41::model::Backbone;
    let mut trace = std::collections::HashMap::new();
    model.forward_traced(&ids, 0, None, &mut |k, v| {
        trace.insert(k.to_string(), v.to_vec());
    })
    .unwrap();
    let w = 4 * d;
    for l in 0..3 {
        for name in ["attn_out", "moe_out"] {
            let key = format!("layer{l:02}.{name}");
            let (Some(have), Ok(want)) = (trace.get(&key), g.read_f32(&format!("prefill.{key}"))) else { continue };
            let per: Vec<(bool, f64, f64)> = (0..ids.len())
                .map(|i| {
                    let (a, b) = (&have[i * d..(i + 1) * d], &want[i * d..(i + 1) * d]);
                    (types[i] >= 0, rel_l2(a, b), b.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt())
                })
                .collect();
            for img in [false, true] {
                let v: Vec<&(bool, f64, f64)> = per.iter().filter(|p| p.0 == img).collect();
                let mean = v.iter().map(|p| p.1).sum::<f64>() / v.len().max(1) as f64;
                let max = v.iter().map(|p| p.1).fold(0.0, f64::max);
                let norm = v.iter().map(|p| p.2).sum::<f64>() / v.len().max(1) as f64;
                eprintln!("  layer {l} {name} {}: mean rel-L2 {mean:.2e} max {max:.2e} mean norm {norm:.1}", if img { "image" } else { "text " });
            }
            if l == 0 && name == "moe_out" {
                let worst: Vec<String> = per.iter().enumerate().filter(|(_, p)| p.1 > 3e-2).take(12).map(|(i, p)| format!("{i}:{:.1e}", p.1)).collect();
                eprintln!("    worst tokens: {worst:?}");
            }
        }
    }
    for l in 0..model.cfg.n_layers {
        let (have, want) = (&trace[&format!("layer{l:02}.out")], g.read_f32(&format!("prefill.layer{l:02}.out")).unwrap());
        let part = |img: bool| {
            let (a, b): (Vec<f32>, Vec<f32>) = (0..ids.len())
                .filter(|&i| (types[i] >= 0) == img)
                .flat_map(|i| have[i * w..(i + 1) * w].iter().copied().zip(want[i * w..(i + 1) * w].iter().copied()))
                .unzip();
            rel_l2(&a, &b)
        };
        let last = rel_l2(&have[(ids.len() - 1) * w..], &want[(ids.len() - 1) * w..]);
        eprintln!("  free run layer {l:2}: text rel-L2 {:.2e}  image {:.2e}  last token {last:.2e}", part(false), part(true));
    }
}

fn greedy(model: &mut GpuModel, first: u32, pos: usize, n: usize) -> Vec<u32> {
    let mut got = vec![first];
    while got.len() < n {
        let logits = model.forward(&got[got.len() - 1..], pos + got.len() - 1).unwrap();
        got.push(argmax(&logits));
    }
    got
}

#[test]
#[ignore]
fn gpu_image_prompt_matches_oracle() {
    let Some((g, mut model)) = setup() else { return };
    let d = model.cfg.dim;
    let ids: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let types = g.read_i64("prompt_types").unwrap();
    let expected: Vec<u32> = g.read_i64("generated_ids").unwrap().iter().map(|&v| v as u32).collect();
    let ref_logits = g.read_f32("prefill.logits").unwrap();
    let ref_emb = g.read_f32("prefill.embed_vl").unwrap();
    let n = ids.len();

    // 1. the reference's own image rows: the LLM side alone
    let ref_spans = spans(&types, &ref_emb, d);
    eprintln!("prompt: {n} tokens, image spans at {:?}", ref_spans.iter().map(|s| (s.start, s.rows.len() / d)).collect::<Vec<_>>());
    let t = Instant::now();
    let logits = model.forward_with(&ids, 0, &ref_spans).unwrap();
    let e = rel_l2(&logits, &ref_logits);
    eprintln!("reference image rows: logits rel-L2 {e:.2e}, prefill {:.1}s", t.elapsed().as_secs_f64());
    let got = greedy(&mut model, argmax(&logits), n, expected.len());
    eprintln!(" generated {got:?}\n expected  {expected:?}");
    // within the prompt's own noise band (see the module comment), and the
    // two leading candidates agree as a set
    assert!(e < 4e-1, "logits rel-L2 {e}");
    let set = |l: &[f32]| {
        let mut t: Vec<u32> = top(l, 2).into_iter().map(|p| p.0).collect();
        t.sort_unstable();
        t
    };
    assert_eq!(set(&logits), set(&ref_logits), "leading candidates differ");

    // 2. our own pipeline: decode, preprocess, the GPU tower
    let image = std::fs::read(env_path("DSV41_IMAGE", r"E:\deepseek\model\mascot.png")).unwrap();
    let prep = vision::load_image(&image, model.cfg.vision.as_ref().unwrap()).unwrap();
    let t = Instant::now();
    let rows = model.encode_image(&prep).unwrap();
    eprintln!("encoded the image ({} patches, {} tokens) in {:.0} ms", prep.n_patches(), prep.n_tokens(), t.elapsed().as_secs_f64() * 1e3);
    let span = &ref_spans[0];
    assert_eq!(rows.len(), span.rows.len(), "span length");
    let e_rows = rel_l2(&rows, &span.rows);
    eprintln!("our image rows vs the reference's: rel-L2 {e_rows:.2e}");
    assert!(e_rows < 6e-2, "image rows {e_rows}");
    let ours = vec![ImageSpan { start: span.start, rows }];
    let logits = model.forward_with(&ids, 0, &ours).unwrap();
    let e = rel_l2(&logits, &ref_logits);
    let got = greedy(&mut model, argmax(&logits), n, expected.len());
    eprintln!("our pipeline: logits rel-L2 {e:.2e}\n generated {got:?}");
    assert!(e < 4e-1, "logits rel-L2 {e}");
    assert!(top(&ref_logits, 2).iter().any(|p| p.0 == got[0]), "first token outside the oracle's top two");

    // 3. a split inside the image span: two chunks give one prefill's logits
    let whole = model.forward_with(&ids, 0, &ours).unwrap();
    let cut = span.start + 57;
    model.forward_with(&ids[..cut], 0, &ours).unwrap();
    let split = model.forward_with(&ids[cut..], cut, &ours).unwrap();
    assert_eq!(whole, split, "chunks split inside the image differ from one prefill");

    // 4. layer by layer, in sub-chunks that cut the image, as the same chunks one by one
    let sub = 50;
    let layered = model.prefill_layered_with(&ids, 0, sub, None, &ours).unwrap();
    let mut chunked = Vec::new();
    for a in (0..n).step_by(sub) {
        let b = (a + sub).min(n);
        chunked = model.forward_with(&ids[a..b], a, &ours).unwrap();
    }
    assert_eq!(layered, chunked, "layered prefill with an image differs from its chunks");
    eprintln!("chunked and layered prefills with the image agree bit for bit");
}
