//! Vision-language demo for Qwen3.6 (27B + the upcoming MoE variants). Mirrors
//! `gemma4_vlm.rs` but uses Qwen3-VL's tower (LayerNorm + fused QKV + GeLU MLP +
//! post-LN, then 2x2 spatial-merge MLP head) and chat template.
//!
//! Image splice: the chat content uses `<|vision_start|><|image_pad|><|vision_end|>`
//! per Qwen3's HF template; we expand the single `<|image_pad|>` (id 248056) into
//! 576 soft-token rows produced by the vision tower (768x768 input → 2304 patches
//! → 576 after 2x2 spatial merge).
//!
//! Usage:
//!   cargo run --release --features cuda -p llama-rs --example qwen36_vlm -- \
//!       models/Qwen3.6-27B-Q4_K_M.gguf \
//!       models/qwen3.6-mmproj-f16.gguf \
//!       models/test_image.jpg \
//!       "What's in this image?" \
//!       cuda

use std::env;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{
    preprocess_image, sampler::Sampler, MmProj, Model, SampleParams, VisionConfig,
};

/// `<|image_pad|>` token id in the Qwen3.6 vocabulary (verified via probe_image_tokens).
const QWEN36_IMAGE_PAD_TOKEN_ID: u32 = 248056;

fn pick_backend(name: &str) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Arc::new(ggml_rs_cuda::CudaBackend::new(0)?)),
        #[cfg(feature = "cuda")]
        s if s.starts_with("cuda:") => {
            let idx: usize = s[5..].parse().map_err(|e| format!("bad cuda device: {e}"))?;
            Ok(Arc::new(ggml_rs_cuda::CudaBackend::new(idx)?))
        }
        #[cfg(not(feature = "cuda"))]
        s if s.starts_with("cuda:") => Err(format!("CUDA not enabled; got `{s}`. Build with --features cuda").into()),
        #[cfg(not(feature = "cuda"))]
        "cuda" => Err("CUDA backend not enabled. Build with `--features cuda`".into()),
        other => Err(format!("unknown backend `{other}`").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_path  = args.next().ok_or("usage: qwen36_vlm <model.gguf> <mmproj.gguf> <image> <prompt> [backend]")?;
    let mmproj_path = args.next().ok_or("missing mmproj path")?;
    let image_path  = args.next().ok_or("missing image path")?;
    let prompt      = args.next().ok_or("missing prompt")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;
    eprintln!("Loading Qwen3.6 model ...");
    let t0 = Instant::now();
    let model_gguf = GgufFile::open(&model_path)?;
    let model = Model::load(&model_gguf, backend.clone())?;
    let qwen35 = match &model {
        Model::Qwen35(m) => m,
        _ => return Err("this demo only handles the Qwen3.5/3.6 architecture; \
                         pass a Qwen3.6-* GGUF as the first argument".into()),
    };
    eprintln!("  model loaded in {:.2?}", t0.elapsed());

    eprintln!("Loading mmproj ...");
    let t1 = Instant::now();
    let mmproj_gguf = GgufFile::open(&mmproj_path)?;
    let mmproj = MmProj::from_gguf(&mmproj_gguf, backend.clone())?;
    let kind_label = match &mmproj {
        MmProj::Qwen3Vl(_) => "Qwen3-VL (LN + fused QKV + GeLU MLP + post-LN) + 2x2 spatial-merge MLP head",
        MmProj::SigLip(_)  => return Err("expected a Qwen3-VL mmproj; got SigLIP".into()),
        MmProj::Gemma4V(_) => return Err("expected a Qwen3-VL mmproj; got Gemma 4 V".into()),
    };
    eprintln!("  mmproj loaded in {:.2?} ({} blocks, {})",
              t1.elapsed(), mmproj.n_blocks(), kind_label);

    // ----- 1. Preprocess image ---------------------------------------------
    eprintln!("Preprocessing image ...");
    let cfg = mmproj.config();
    let vc = VisionConfig {
        image_size: cfg.image_size,
        patch_size: cfg.patch_size,
        mean:       cfg.mean,
        std:        cfg.std,
    };
    let image = preprocess_image(image_path.as_ref(), &vc)?;
    eprintln!("  image: {:?}", image.shape());

    // ----- 2. Vision tower → soft tokens ----------------------------------
    eprintln!("Running vision tower (one-shot) ...");
    let t2 = Instant::now();
    let soft_tokens = mmproj.forward(&image)?;
    eprintln!("  soft-token shape {:?} in {:.2?}", soft_tokens.shape(), t2.elapsed());

    // ----- 3. Build chat template with <|image_pad|> placeholder ----------
    // Qwen3 chat template: `<|im_start|>user\n<image content>{prompt}<|im_end|>\n<|im_start|>assistant\n`
    // The image content is HF's `<|vision_start|><|image_pad|><|vision_end|>` —
    // splicing happens at the single `<|image_pad|>` position.
    let chat = format!(
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let (tokens, embeds) = qwen35.embed_with_vision_at_placeholder(
        &chat, &soft_tokens, QWEN36_IMAGE_PAD_TOKEN_ID, false /* Qwen has no BOS */,
    )?;
    eprintln!("  spliced sequence: {} tokens (text + {} vision soft tokens)",
              tokens.len(), soft_tokens.dim(0));

    // ----- 4. Prefill -----------------------------------------------------
    let mut kv = model.new_kv_cache(8192);
    eprintln!("Prefill ...");
    let t3 = Instant::now();
    let start = tokens.iter().position(|&t| t == QWEN36_IMAGE_PAD_TOKEN_ID).unwrap();
    let side = (soft_tokens.dim(0) as f64).sqrt() as usize;
    let mut positions = Vec::new();
    positions.extend((0..start).map(|p| [p as u32; 3]));
    positions.extend((0..side*side).map(|p| [start as u32, (start+p/side) as u32, (start+p%side) as u32]));
    let next = start + side;
    positions.extend((0..tokens.len()-start-side*side).map(|p| [(next+p) as u32; 3]));
    let mut next_position = positions.last().unwrap().iter().copied().max().unwrap() + 1;
    let logits = qwen35.forward_embeds_positions(&embeds, tokens.len(), &mut kv, Some(&positions))?;
    let dt_prefill = t3.elapsed();
    let n_prompt = tokens.len();
    eprintln!("  prefill: {} tokens in {:.2?} ({:.0} tok/s)",
              n_prompt, dt_prefill, n_prompt as f64 / dt_prefill.as_secs_f64());

    // Sample first token.
    let logits_host = model.last_logits(&logits);
    let params = SampleParams {
        temperature:    0.7,
        top_k:          Some(40),
        top_p:          Some(0.9),
        min_p:          Some(0.05),
        repeat_penalty: Some(1.1),
        repeat_last_n:  64,
        ..Default::default()
    };
    let mut sampler = Sampler::new(params);
    let mut tok = sampler.sample(&logits_host);

    // Stop on `<|im_end|>` end-of-turn marker.
    let stop_ids = qwen35.tokenizer.encode("<|im_end|>", false).unwrap_or_default();
    eprintln!();
    eprintln!("--- Response ---");

    // ----- 5. Decode loop -------------------------------------------------
    let max_new = 512;
    let t4 = Instant::now();
    let mut produced = 0usize;
    loop {
        if stop_ids.contains(&tok) { break; }
        let piece = qwen35.tokenizer.decode(&[tok]);
        print!("{piece}");
        std::io::stdout().flush()?;
        sampler.observe(tok);
        produced += 1;
        if produced >= max_new { break; }

        let e = qwen35.embed_text(&[tok]);
        let logits = qwen35.forward_embeds_positions(&e, 1, &mut kv, Some(&[[next_position;3]]))?;
        next_position += 1;
        let logits_host = model.last_logits(&logits);
        tok = sampler.sample(&logits_host);
    }
    let dt_decode = t4.elapsed();
    println!();
    eprintln!();
    eprintln!("--- Stats ---");
    eprintln!("  decoded {} tokens in {:.2?} ({:.0} tok/s)",
              produced, dt_decode, produced as f64 / dt_decode.as_secs_f64());

    Ok(())
}
