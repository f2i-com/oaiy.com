//! Probe a GGUF tokenizer for image-related markers. Useful when porting a new
//! multimodal architecture: tells you whether tokens like `<|image|>` or
//! `<start_of_image>` are recognised as single tokens (a sign they're
//! "special" / chat-template markers) or get byte-shredded.
//!
//! Usage: `cargo run --release -p llama-rs --example probe_image_tokens -- model.gguf`

use std::env;

use gguf::GgufFile;
use llama_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: probe_image_tokens <model.gguf>")?;
    let g = GgufFile::open(&path)?;
    let backend = ggml_rs::default_backend();
    let model = Model::load(&g, backend)?;
    let tok = model.tokenizer();

    let candidates = [
        "<|image|>",
        "<image_soft_token>",
        "<start_of_image>",
        "<end_of_image>",
        "<|image_soft_token|>",
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|object_ref_start|>",
        "<|vision_pad|>",
    ];
    println!("== Encoding common image markers ==");
    for s in candidates {
        match tok.encode(s, false) {
            Ok(ids) => {
                let decoded: String = ids.iter().map(|&i| tok.decode(&[i])).collect::<Vec<_>>().join("|");
                println!("  {s:30} -> {} tokens: {:?}  (decoded: {decoded:?})", ids.len(), ids);
            }
            Err(e) => println!("  {s:30} -> ERR: {e}"),
        }
    }

    println!();
    println!("== Probing token ids near vocab edges ==");
    let vocab_size = tok.vocab_size() as u32;
    // Show last 10 tokens (often where special markers live).
    let tail_start = vocab_size.saturating_sub(10);
    for id in tail_start..vocab_size {
        let s = tok.decode(&[id]);
        println!("  id {id:6} -> {s:?}");
    }

    Ok(())
}
