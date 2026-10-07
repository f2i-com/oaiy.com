//! Structured-output demo. Two modes:
//!   * Default: char-class filter (`with_token_filter`) — fast to set up,
//!     no formal guarantee, but well-trained models tend to fall into JSON
//!     shape on their own when their non-JSON tokens are masked out.
//!   * `--strict`: full JSON FSM (`with_json_object`) — output is *guaranteed*
//!     to parse as a flat `{"key": value, ...}` object with string/integer
//!     values. Auto-stops on the closing `}`.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example json_mode -- model.gguf cpu
//!   cargo run --release -p llama-rs --example json_mode -- model.gguf cpu --strict
//!   cargo run --release -p llama-rs --example json_mode -- model.gguf cpu --strict "List 3 primes"

use std::env;
use std::io::Write;

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

/// Bytes allowed in the body of a JSON value: punctuation, ASCII text, digits,
/// whitespace. Excludes characters that English prose uses but JSON does not
/// (e.g. `?`, `!`, `'`, `(`, `)`).
fn is_json_byte(b: u8) -> bool {
    matches!(b,
        b'{' | b'}' | b'[' | b']' | b':' | b',' | b'"' | b'\\'
        | b' ' | b'\t' | b'\n' | b'\r'
        | b'0'..=b'9'
        | b'a'..=b'z' | b'A'..=b'Z'
        | b'.' | b'-' | b'+' | b'_'
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw: Vec<String> = env::args().skip(1).collect();
    let strict = raw.iter().any(|a| a == "--strict");
    let mut args = raw.into_iter().filter(|a| a != "--strict");
    let path = args.next().ok_or("usage: json_mode <model.gguf> [backend] [--strict] [user_prompt]")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());
    let user_prompt = args.next().unwrap_or_else(|| {
        "Respond with a JSON object containing the keys 'name' and 'age', \
         describing a fictional person. Output JSON only, no prose.".to_string()
    });

    let backend = pick_backend(&backend_name)?;
    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, backend)?;

    eprintln!("loaded {:?} on {} backend",
              model.config().arch, model.backend().name());
    eprintln!("user prompt: {user_prompt}");
    eprintln!("---");

    let messages = vec![
        ChatMessage::system("You are a JSON-only responder. Reply with only a JSON value."),
        ChatMessage::user(user_prompt),
    ];
    let prompt_text = apply_chat_template(&model.config().arch, &messages, true);
    let prompt_ids = model.tokenizer().encode(&prompt_text, false)?;

    let params = SampleParams {
        temperature: 0.3,
        top_k: Some(40),
        top_p: Some(0.95),
        min_p: Some(0.05),
        repeat_penalty: Some(1.05),
        repeat_last_n: 64,
        ..Default::default()
    };

    let base = model.generate(&prompt_ids, params, 256)
        .no_repeat_ngram_size(4)
        .min_new_tokens(4);
    let iter = if strict {
        // Strict FSM: guaranteed parseable JSON, auto-stops on closing `}`.
        eprintln!("[strict mode: with_json_object FSM]");
        base.with_json_object()
    } else {
        // Char-class filter: looser, faster, "JSON-ish" output.
        eprintln!("[soft mode: with_token_filter + stop_on_balanced]");
        base.with_token_filter(|_id, text| text.bytes().all(is_json_byte))
            .stop_on_balanced(b'{', b'}')
    };

    for tok in iter {
        let piece = model.tokenizer().decode(&[tok]);
        print!("{piece}");
        std::io::stdout().flush()?;
    }
    println!();
    Ok(())
}
