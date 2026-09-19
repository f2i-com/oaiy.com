//! Dump tokens 0..N from a GGUF tokenizer along with their type. Helps debug
//! tokenizer issues when a model's chat markers don't tokenize as expected.

use std::env;

use ggml_rs::default_backend;
use gguf::{Array, GgufFile};
use llama_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path  = env::args().nth(1).ok_or("usage: dump_special_tokens <path.gguf> [N]")?;
    let limit = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(150usize);

    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, default_backend())?;
    let tok = model.tokenizer();

    // Try to read token_type from metadata
    let types: Option<Vec<i32>> = gguf.metadata()
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_array())
        .and_then(|a| if let Array::I32(v) = a { Some(v.clone()) } else { None });

    if let Some(s) = gguf.metadata().get("tokenizer.chat_template").and_then(|v| v.as_str()) {
        println!("--- chat_template ({} bytes) ---", s.len());
        // Print only the parts that look like role/turn markers to keep output readable.
        for (i, line) in s.lines().enumerate() {
            if line.contains("turn") || line.contains("|>") || line.contains("<|")
                || line.to_lowercase().contains("role") || line.to_lowercase().contains("user")
                || line.to_lowercase().contains("assistant") || line.to_lowercase().contains("model")
            {
                println!("  L{i:>4}: {line}");
            }
        }
        println!();
    }

    println!("First {limit} tokens (id, type, literal):");
    for id in 0..limit.min(tok.vocab_size()) {
        let s = tok.token(id as u32).unwrap_or("<?>");
        let t = types.as_ref().and_then(|v| v.get(id).copied()).unwrap_or(-1);
        let kind = match t {
            1 => "NORMAL",
            2 => "UNKNOWN",
            3 => "CONTROL",
            4 => "USER_DEFINED",
            5 => "UNUSED",
            6 => "BYTE",
            _ => "OTHER",
        };
        println!("  [{id:>5}] type={t} ({kind:<13}) {s:?}");
    }
    Ok(())
}
