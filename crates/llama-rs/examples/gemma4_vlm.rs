//! Vision-language demo for Gemma 4 (E2B / E4B): its vision tower (`Gemma4VMmProj`, chained on a GPU's device by
//! `chain_gemma4v`), the picture's soft tokens spliced into the prompt in place of its `<|image|>` markers
//! (`Gemma4Model::forward_spliced`, whose PLE reads the padding's row there, as Transformers does), and Gemma 4's chat
//! template (`<|turn>user\n<|image>` + one `<|image|>` a soft token + `<image|>{prompt}<turn|>\n<|turn>model\n`).
//!
//! The tower runs on a backend with a device chain only (WebGPU: OAIY's server, `oaiy-llm-server-webgpu`, reads
//! pictures with Gemma 4 when given its mmproj with `--vision-projector`); on the CPU backend this demo stops at the
//! tower with that error.
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
use llama_rs::{sampler::Sampler, MmProj, Model, SampleParams};

/// The soft tokens a picture is at most (Gemma 4's image processor's default).
const SOFT_TOKENS: usize = 280;

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
        MmProj::MiniCpmV(_) => return Err("expected a Gemma 4 V mmproj; got MiniCPM-V".into()),
    };
    eprintln!("  mmproj loaded in {:.2?} ({} blocks, {})",
              t1.elapsed(), mmproj.n_blocks(), kind_label);

    // ----- 1. Preprocess image ---------------------------------------------
    eprintln!("Preprocessing image ...");
    let image = llama_rs::vision::gemma4_image(&std::fs::read(&image_path)?, mmproj.config().patch_size, SOFT_TOKENS)?;
    eprintln!("  image: {:?}", image.shape());

    // ----- 2. Vision tower → soft tokens ----------------------------------
    eprintln!("Running vision tower (one-shot) ...");
    let t2 = Instant::now();
    let soft_tokens = mmproj.forward(&image)?.to_host();
    eprintln!("  soft-token shape {:?} in {:.2?}", soft_tokens.shape(), t2.elapsed());

    // ----- 3. The prompt: the picture's soft tokens between `<|image>` and `<image|>` -------------
    let tok_id = |t: &str| gemma4.tokenizer.token_id(t).ok_or(format!("the tokenizer lacks {t}"));
    let (open, image_id, close) = (tok_id("<|image>")?, tok_id("<|image|>")?, tok_id("<image|>")?);
    let mut tokens = gemma4.tokenizer.encode("<|turn>user\n", true)?;
    tokens.push(open);
    let at = tokens.len();
    tokens.extend(std::iter::repeat_n(image_id, soft_tokens.dim(0)));
    tokens.push(close);
    tokens.extend(gemma4.tokenizer.encode(&format!("{prompt}<turn|>\n<|turn>model\n"), false)?);
    eprintln!("  spliced sequence: {} tokens (text + {} vision soft tokens)", tokens.len(), soft_tokens.dim(0));

    // ----- 4. Prefill -----------------------------------------------------
    let mut kv = model.new_kv_cache(8192);
    eprintln!("Prefill ...");
    let t3 = Instant::now();
    let logits = gemma4.forward_spliced(&tokens, &[(at, soft_tokens.data())], &mut kv);
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
