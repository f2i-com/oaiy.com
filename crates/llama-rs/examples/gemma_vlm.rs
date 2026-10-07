//! Vision-language demo: ask Gemma 3 4B about an image.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example gemma_vlm -- \
//!       models/gemma-3-4b-it-q4_k_m.gguf \
//!       models/mmproj-gemma-3-4b-f16.gguf \
//!       path/to/image.jpg \
//!       "What's in this image?"
//!
//! Pipeline (per turn):
//!   1. Preprocess image       (vision::preprocess_image)
//!   2. Run vision tower       (MmProj::forward → [256, 2560] soft tokens)
//!   3. Tokenize text spans    (around the image placeholder)
//!   4. Splice embeddings      (Gemma3Model::embed_with_vision)
//!   5. Forward                (Gemma3Model::forward_embeds → logits)
//!   6. Sample + decode loop   (manual; mirrors generate.rs without prefill)
//!
//! Cost note: vision tower runs once (~90s on host CPU for 896×896 SigLIP-400M).
//! Text decode is the same as text-only Gemma 3 4B. The vision tower runs on
//! the LM's backend
//! (one-shot per image, host fallback paths inside MmProj::forward).

use std::env;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{
    preprocess_image, sampler::Sampler, MmProj, Model,
    SampleParams, VisionConfig,
};

fn pick_backend(name: &str) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("there is no CUDA backend any more: the GPU is WebGPU (oaiy-llm --webgpu, or the server)".into()),
        other => Err(format!("unknown backend `{other}` (try `cpu`)").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_path  = args.next().ok_or("usage: gemma_vlm <model.gguf> <mmproj.gguf> <image> <prompt> [backend]")?;
    let mmproj_path = args.next().ok_or("missing mmproj path")?;
    let image_path  = args.next().ok_or("missing image path")?;
    let prompt      = args.next().ok_or("missing prompt")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;
    eprintln!("Loading model ...");
    let t0 = Instant::now();
    let model_gguf = GgufFile::open(&model_path)?;
    let model = Model::load(&model_gguf, backend.clone())?;
    let gemma3 = match &model {
        Model::Gemma3(m) => m,
        _ => return Err("this demo only handles the Gemma 3 architecture; \
                         pass a gemma-3-* GGUF as the first argument".into()),
    };
    eprintln!("  model loaded in {:.2?}", t0.elapsed());

    eprintln!("Loading mmproj ...");
    let t1 = Instant::now();
    let mmproj_gguf = GgufFile::open(&mmproj_path)?;
    let mmproj = MmProj::from_gguf(&mmproj_gguf, backend.clone())?;
    let kind_label = match &mmproj {
        MmProj::SigLip(m) => match &m.projector {
            llama_rs::Projector::Gemma(_) => "SigLIP + Gemma projector",
            llama_rs::Projector::Mlp(_)   => "SigLIP + Mlp projector",
        }
        MmProj::Gemma4V(_) => "Gemma 4 V (RMSNorm + SwiGLU + sandwich) + single-linear projector",
        // VENDORED-LOCAL: upstream match was not updated when MmProj::Qwen3Vl
        // was added to the enum; this example only handles Gemma mmprojs.
        MmProj::Qwen3Vl(_) => return Err("expected a Gemma mmproj; got Qwen3-VL".into()),
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

    // ----- 2. Run vision tower → soft tokens -------------------------------
    eprintln!("Running vision tower (one-shot, may be slow on first run) ...");
    let t2 = Instant::now();
    let soft_tokens = mmproj.forward(&image)?;
    eprintln!("  soft-token shape {:?} in {:.2?}", soft_tokens.shape(), t2.elapsed());

    // ----- 3. Build chat template with image placeholder -------------------
    // Gemma 3 multimodal chat template:
    //   <start_of_turn>user\n<start_of_image>{soft_tokens}<end_of_image>{prompt}<end_of_turn>\n
    //   <start_of_turn>model\n
    // We tokenize the pre-image and post-image strings separately so we can
    // splice the soft tokens in at the right embedding-stream position.
    let pre_text  = format!("<start_of_turn>user\n<start_of_image>");
    let post_text = format!("<end_of_image>{prompt}<end_of_turn>\n<start_of_turn>model\n");
    let pre_tokens  = gemma3.tokenizer.encode(&pre_text,  true /* add BOS */)?;
    let post_tokens = gemma3.tokenizer.encode(&post_text, false)?;
    let n_soft = soft_tokens.dim(0);
    eprintln!("Prompt: {} pre-text + {} soft + {} post-text = {} tokens",
              pre_tokens.len(), n_soft, post_tokens.len(),
              pre_tokens.len() + n_soft + post_tokens.len());

    // ----- 4. Splice embeddings --------------------------------------------
    let embeds = gemma3.embed_with_vision(&pre_tokens, &soft_tokens, &post_tokens);

    // ----- 5. Prefill --------------------------------------------------------
    let mut kv = model.new_kv_cache(8192);
    eprintln!("Prefill ...");
    let t3 = Instant::now();
    let logits = gemma3.forward_embeds(&embeds, &mut kv);
    let dt_prefill = t3.elapsed();
    let n_prompt = embeds.dim(0);
    // forward_embeds already calls kv.commit() internally — same as forward().
    eprintln!("  prefill: {} tokens in {:.2?} ({:.0} tok/s)",
              n_prompt, dt_prefill, n_prompt as f64 / dt_prefill.as_secs_f64());

    // Sample first token from prefill output.
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

    // Stop tokens: Gemma 3's `<end_of_turn>` marker.
    let stop_text = "<end_of_turn>";
    let stop_ids = gemma3.tokenizer.encode(stop_text, false).unwrap_or_default();
    eprintln!();
    eprintln!("--- Response ---");

    // ----- 6. Decode loop --------------------------------------------------
    let max_new = 512;
    let t4 = Instant::now();
    let mut produced = 0usize;
    loop {
        if stop_ids.contains(&tok) { break; }
        let piece = gemma3.tokenizer.decode(&[tok]);
        print!("{piece}");
        std::io::stdout().flush()?;
        sampler.observe(tok);
        produced += 1;
        if produced >= max_new { break; }

        // Single-token forward.
        let logits = gemma3.forward(&[tok], &mut kv);
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
