//! One-shot completion demo. Showcases the high-level `Model::ask*` APIs that
//! wrap `generate() + decode` for batch / scripting use cases where streaming
//! isn't needed.
//!
//! Three modes:
//!   * `raw`  — `Model::ask(prompt, max_new)`: no chat template, just prompt → text
//!   * `chat` — `Model::ask_chat(messages, max_new)`: chat-templated, returns text
//!   * `json` — `Model::ask_json(messages, max_new)`: JSON-constrained via the FSM
//!
//! Usage:
//!   cargo run --release -p llama-rs --example ask -- model.gguf cpu chat "Capital of Japan?"
//!   cargo run --release -p llama-rs --example ask -- model.gguf cpu json "Describe a fictional cat as JSON: name, age, indoor (bool)"

use std::env;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{ChatMessage, Model};

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("there is no CUDA backend any more: the GPU is WebGPU (oaiy-llm --webgpu, or the server)".into()),
        "auto" => Ok(default_backend()),
        other => Err(format!("unknown backend `{other}` (try `cpu` or `auto`)").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: ask <model.gguf> [backend] [mode] [prompt]")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());
    let mode = args.next().unwrap_or_else(|| "chat".to_string());
    let prompt = args.next()
        .unwrap_or_else(|| "What is the capital of Japan? Answer in one word.".to_string());

    let backend = pick_backend(&backend_name)?;
    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, backend)?;
    eprintln!("loaded {:?} on {}", model.config().arch, model.backend().name());
    eprintln!("mode: {mode}");
    eprintln!("prompt: {prompt}");
    eprintln!("---");

    let response = match mode.as_str() {
        "raw" => model.ask(&prompt, 256)?,
        "chat" => {
            let msgs = vec![
                ChatMessage::system("You are a helpful, concise assistant. Reply directly without preamble."),
                ChatMessage::user(prompt),
            ];
            model.ask_chat(&msgs, 256)?
        }
        "json" => {
            let msgs = vec![
                ChatMessage::system("Respond only with a JSON value. No prose, no markdown fences."),
                ChatMessage::user(prompt),
            ];
            model.ask_json(&msgs, 256)?
        }
        other => return Err(format!("unknown mode `{other}` (try raw/chat/json)").into()),
    };
    println!("{response}");
    Ok(())
}
