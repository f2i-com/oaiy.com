//! Vision-language demo for Qwen3-VL-30B-A3B (qwen3moe arch + Qwen3-VL mmproj).
//! Mirrors `qwen36_vlm.rs` (qwen35 backbone) but routes the LM stack through
//! `Qwen3MoeModel` instead. Uses the same `qwen3vl_merger` mmproj loader from
//! `mmproj.rs`, so the vision side is identical.
//!
//! Usage:
//!   cargo run --release --features cuda -p llama-rs --example qwen3_moe_vlm -- \
//!       models/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf \
//!       models/Qwen3-VL-30B-A3B-mmproj-F16.gguf \
//!       test_image.png \
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

/// Encode `<|image_pad|>` to recover its id from the model's tokenizer at
/// runtime. Qwen3-VL = 151655; Qwen3.6-VL = 248056. We don't hardcode either.
fn image_pad_token_id(tok: &tokenizer::Tokenizer) -> Result<u32, Box<dyn std::error::Error>> {
    let ids = tok.encode("<|image_pad|>", false)?;
    if ids.len() != 1 {
        return Err(format!("expected `<|image_pad|>` to encode to 1 token, got {ids:?}").into());
    }
    Ok(ids[0])
}

fn pick_backend(name: &str) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("CUDA backend not enabled. Build with `--features cuda`".into()),
        s if s.starts_with("cuda:") => Err(format!("CUDA not enabled; got `{s}`").into()),
        other => Err(format!("unknown backend `{other}`").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_path  = args.next().ok_or("usage: qwen3_moe_vlm <model.gguf> <mmproj.gguf> <image> <prompt> [backend]")?;
    let mmproj_path = args.next().ok_or("missing mmproj path")?;
    let image_path  = args.next().ok_or("missing image path")?;
    let prompt      = args.next().ok_or("missing prompt")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;
    eprintln!("Loading Qwen3-MoE model ...");
    let t0 = Instant::now();
    let model_gguf = GgufFile::open(&model_path)?;
    let model = Model::load(&model_gguf, backend.clone())?;
    let qwen3moe = match &model {
        Model::Qwen3Moe(m) => m,
        _ => return Err("expected a qwen3moe-arch GGUF as the first argument".into()),
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

    eprintln!("Running vision tower (one-shot) ...");
    let t2 = Instant::now();
    let soft_tokens = mmproj.forward(&image)?;
    eprintln!("  soft-token shape {:?} in {:.2?}", soft_tokens.shape(), t2.elapsed());

    let chat = format!(
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let pad_id = image_pad_token_id(&qwen3moe.tokenizer)?;
    let (tokens, embeds) = qwen3moe.embed_with_vision_at_placeholder(
        &chat, &soft_tokens, pad_id, false,
    )?;
    eprintln!("  spliced sequence: {} tokens (text + {} vision soft tokens)",
              tokens.len(), soft_tokens.dim(0));

    let mut kv = model.new_kv_cache(8192);
    eprintln!("Prefill ...");
    let t3 = Instant::now();
    let logits = qwen3moe.forward_embeds(&embeds, tokens.len(), &mut kv);
    let dt_prefill = t3.elapsed();
    let n_prompt = tokens.len();
    eprintln!("  prefill: {} tokens in {:.2?} ({:.0} tok/s)",
              n_prompt, dt_prefill, n_prompt as f64 / dt_prefill.as_secs_f64());

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

    let stop_ids = qwen3moe.tokenizer.encode("<|im_end|>", false).unwrap_or_default();
    eprintln!();
    eprintln!("--- Response ---");

    let max_new = 512;
    let t4 = Instant::now();
    let mut produced = 0usize;
    loop {
        if stop_ids.contains(&tok) { break; }
        let piece = qwen3moe.tokenizer.decode(&[tok]);
        print!("{piece}");
        std::io::stdout().flush()?;
        sampler.observe(tok);
        produced += 1;
        if produced >= max_new { break; }

        let logits = qwen3moe.forward(&[tok], &mut kv);
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
