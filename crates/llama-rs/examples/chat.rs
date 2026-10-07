//! Single-turn chat: format a system + user message with the architecture's
//! chat template, run inference, stream tokens.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example chat -- model.gguf "your question"
//!   cargo run --release -p llama-rs --example chat -- model.gguf "your question" cpu

use std::env;
use std::io::Write;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{ChatMessage, Model, SampleParams};

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("there is no CUDA backend any more: the GPU is WebGPU (oaiy-llm --webgpu, or the server)".into()),
        other => Err(format!("unknown backend `{other}` (try `cpu`)").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: chat <path.gguf> [user_msg] [backend]")?;
    let user_msg = args.next().unwrap_or_else(|| "Hello! Briefly tell me what you can do.".to_string());
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;
    let load_t0 = Instant::now();
    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, backend)?;
    let load_ms = load_t0.elapsed().as_millis();
    eprintln!("loaded {:?} on {} ({} ms)", model.config().arch, model.backend().name(), load_ms);

    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Answer concisely."),
        ChatMessage::user(&user_msg),
    ];
    let prompt_ids = model.encode_chat(&messages, true)?;
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let params = SampleParams {
        temperature:    0.7,
        top_k:          Some(40),
        top_p:          Some(0.9),
        min_p:          Some(0.05),
        repeat_penalty: Some(1.1),
        repeat_last_n:  64,
        ..Default::default()
    };

    let pre_t0 = Instant::now();
    let mut iter = model.generate(&prompt_ids, params, 256);
    let pre_ms = pre_t0.elapsed().as_millis();
    eprintln!("prefill: {pre_ms} ms ({:.1} tok/s)",
              prompt_ids.len() as f32 * 1000.0 / pre_ms.max(1) as f32);

    print!("Assistant: ");
    std::io::stdout().flush()?;

    let dec_t0 = Instant::now();
    for token in &mut iter {
        let piece = model.tokenizer().decode(&[token]);
        print!("{piece}");
        std::io::stdout().flush()?;
    }
    let dec_ms = dec_t0.elapsed().as_millis();
    let produced = iter.produced();
    println!();
    eprintln!("decode: {dec_ms} ms, {} tokens ({:.1} tok/s)",
              produced,
              produced as f32 * 1000.0 / dec_ms.max(1) as f32);
    Ok(())
}
