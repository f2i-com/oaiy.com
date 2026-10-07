//! Two-turn chat using `Model::generate_with_kv` to reuse the KV cache across
//! turns. Demonstrates the prefill speedup: turn 2 only re-encodes the delta
//! (assistant response from turn 1 + new user turn opener) instead of the
//! whole conversation.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example chat_kv_reuse -- model.gguf cpu

use std::env;
use std::io::Write;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{apply_chat_template, ChatMessage, Model, SampleParams};

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("there is no CUDA backend any more: the GPU is WebGPU (oaiy-llm --webgpu, or the server)".into()),
        other => Err(format!("unknown backend `{other}` (try `cpu`)").into()),
    }
}

fn run_turn(
    model: &Model,
    prompt_ids: &[u32],
    kv: &mut llama_rs::KvCache,
    label: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let params = SampleParams {
        temperature: 0.7,
        top_k: Some(40),
        top_p: Some(0.9),
        min_p: Some(0.05),
        repeat_penalty: Some(1.1),
        repeat_last_n: 64,
        ..Default::default()
    };

    let pre_t0 = Instant::now();
    let iter = model.generate_with_kv(prompt_ids, params, 256, kv);
    let pre_ms = pre_t0.elapsed().as_millis();
    eprintln!("[{label}] prefill: {pre_ms} ms for {} new tokens", prompt_ids.len());

    print!("[{label}] Assistant: ");
    std::io::stdout().flush()?;
    let dec_t0 = Instant::now();
    let mut response = String::new();
    let mut produced = 0usize;
    for tok in iter {
        let piece = model.tokenizer().decode(&[tok]);
        response.push_str(&piece);
        print!("{piece}");
        std::io::stdout().flush()?;
        produced += 1;
    }
    println!();
    let dec_ms = dec_t0.elapsed().as_millis();
    eprintln!("[{label}] decode: {dec_ms} ms, {produced} tokens");
    Ok(response)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: chat_kv_reuse <model.gguf> [backend]")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());
    let backend = pick_backend(&backend_name)?;

    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, backend)?;
    eprintln!("loaded {:?} on {}", model.config().arch, model.backend().name());

    let mut kv = model.new_kv_cache(4096);

    // ---- Turn 1: full conversation prefill (system + user_1) ----
    let mut history = vec![
        ChatMessage::system("You are a concise geography assistant. Answer in one short sentence."),
        ChatMessage::user("What is the capital of France?"),
    ];
    let template_t1 = apply_chat_template(&model.config().arch, &history, true);
    let prompt_t1 = model.tokenizer().encode(&template_t1, false)?;
    let response_t1 = run_turn(&model, &prompt_t1, &mut kv, "turn 1")?;
    history.push(ChatMessage::assistant(response_t1.clone()));

    // ---- Turn 2: only encode the delta (assistant_t1 + user_2) ----
    history.push(ChatMessage::user("And what is the capital of Japan?"));
    let template_t2 = apply_chat_template(&model.config().arch, &history, true);
    // The delta is the suffix of template_t2 that starts after template_t1
    // (which the model has already prefilled into kv). The assistant's
    // response text was sampled token-by-token, but we re-tokenise the joined
    // text so subsequent prefill stays consistent — BPE merges may differ
    // across turn boundaries otherwise.
    let prefix_len = template_t1.len() + response_t1.len();
    let delta = if template_t2.len() >= prefix_len && template_t2.starts_with(&template_t1) {
        &template_t2[prefix_len..]
    } else {
        // Mismatch — fall back to a full re-prefill (shouldn't happen for
        // arch templates that are append-only, but defensive).
        eprintln!("[turn 2] template mismatch — falling back to full prefill");
        kv.reset();
        &template_t2[..]
    };
    let prompt_t2 = model.tokenizer().encode(delta, false)?;
    let _response_t2 = run_turn(&model, &prompt_t2, &mut kv, "turn 2")?;

    eprintln!("\n(KV cache reused across turns — turn 2's prefill skipped the prior conversation.)");
    Ok(())
}
