//! Vision-language demo for Gemma 4 (E2B / E4B). **WORK IN PROGRESS** — the
//! vision tower runs cleanly and produces correct-shape soft tokens; the
//! splice + LM-side prefill hit an `ILLEGAL_ADDRESS` error on the CUDA backend
//! that was (never resolved; the CPU backend is what runs it now). Mirrors `gemma_vlm.rs`
//! (Gemma 3) but uses:
//!   * Gemma 4's bespoke vision tower (RMSNorm + SwiGLU + sandwich norms)
//!   * Gemma 4's chat template (`<|turn>user\n{prompt}<turn|>\n<|turn>model\n`)
//!   * Single `<|image|>` placeholder token (id 258880) expanded to N=196
//!     soft-token rows at runtime (vs Gemma 3's `<start_of_image>` ... `<end_of_image>`)
//!   * `Gemma4Model::embed_with_vision_at_placeholder` + `forward_embeds`
//!     (PLE-aware — placeholder positions get `<|image|>`'s PLE)
//!
//! Usage:
//!   cargo run --release -p llama-rs --example gemma4_vlm -- \
//!       models/gemma-4-e2b-q4_k_m.gguf \
//!       models/mmproj-gemma-4-e2b-f16.gguf \
//!       models/test_image.jpg \
//!       "What's in this image?" \
//!       cpu

use std::env;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{
    preprocess_image, sampler::Sampler, MmProj, Model, SampleParams, VisionConfig,
};

/// `<|image|>` token id in the Gemma 4 vocabulary, found via probe_image_tokens.
const GEMMA4_IMAGE_TOKEN_ID: u32 = 258880;

fn pick_backend(name: &str) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        s if s.starts_with("cuda:") => Err(format!("there is no CUDA backend any more: the GPU is WebGPU (oaiy-llm --webgpu, or the server); got `{s}`").into()),
        "cuda" => Err("there is no CUDA backend any more: the GPU is WebGPU (oaiy-llm --webgpu, or the server)".into()),
        other => Err(format!("unknown backend `{other}`").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_path  = args.next().ok_or("usage: gemma4_vlm <model.gguf> <mmproj.gguf> <image> <prompt> [backend]")?;
    let mmproj_path = args.next().ok_or("missing mmproj path")?;
    let image_path  = args.next().ok_or("missing image path")?;
    let prompt      = args.next().ok_or("missing prompt")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;
    eprintln!("Loading Gemma 4 model ...");
    let t0 = Instant::now();
    let model_gguf = GgufFile::open(&model_path)?;
    let model = Model::load(&model_gguf, backend.clone())?;
    let gemma4 = match &model {
        Model::Gemma4(m) => m,
        _ => return Err("this demo only handles the Gemma 4 architecture; \
                         pass a gemma-4-* GGUF as the first argument".into()),
    };
    eprintln!("  model loaded in {:.2?}", t0.elapsed());

    eprintln!("Loading mmproj ...");
    let t1 = Instant::now();
    let mmproj_gguf = GgufFile::open(&mmproj_path)?;
    let mmproj = MmProj::from_gguf(&mmproj_gguf, backend.clone())?;
    let kind_label = match &mmproj {
        MmProj::Gemma4V(_) => "Gemma 4 V (RMSNorm + SwiGLU + sandwich) + linear projector",
        MmProj::SigLip(_)  => return Err("expected a Gemma 4 V mmproj; got SigLIP \
                                         (you probably want examples/gemma_vlm.rs instead)".into()),
        // VENDORED-LOCAL: upstream match was not updated when MmProj::Qwen3Vl
        // was added to the enum; this example only handles Gemma 4 V mmprojs.
        MmProj::Qwen3Vl(_) => return Err("expected a Gemma 4 V mmproj; got Qwen3-VL".into()),
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

    // ----- 3. Build chat template with <|image|> placeholder --------------
    // Gemma 4 chat template uses `<|turn>{role}\n{content}<turn|>\n` blocks.
    // Image content is a single `<|image|>` token, which we'll expand to 196
    // soft-token rows at the splice step.
    let chat = format!("<|turn>user\n<|image|>{prompt}<turn|>\n<|turn>model\n");
    let (tokens, embeds) = gemma4.embed_with_vision_at_placeholder(
        &chat, &soft_tokens, GEMMA4_IMAGE_TOKEN_ID, true /* add BOS */,
    )?;
    eprintln!("  spliced sequence: {} tokens (text + {} vision soft tokens)",
              tokens.len(), soft_tokens.dim(0));

    // ----- 4. Prefill -----------------------------------------------------
    let mut kv = model.new_kv_cache(8192);
    eprintln!("Prefill ...");
    let t3 = Instant::now();
    let logits = gemma4.forward_embeds(&embeds, &tokens, &mut kv);
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

    // Stop on Gemma 4's `<turn|>` end-of-turn marker.
    let stop_ids = gemma4.tokenizer.encode("<turn|>", false).unwrap_or_default();
    eprintln!();
    eprintln!("--- Response ---");

    // ----- 5. Decode loop -------------------------------------------------
    let max_new = 512;
    let t4 = Instant::now();
    let mut produced = 0usize;
    loop {
        if stop_ids.contains(&tok) { break; }
        let piece = gemma4.tokenizer.decode(&[tok]);
        print!("{piece}");
        std::io::stdout().flush()?;
        sampler.observe(tok);
        produced += 1;
        if produced >= max_new { break; }

        let logits = gemma4.forward(&[tok], &mut kv);
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
