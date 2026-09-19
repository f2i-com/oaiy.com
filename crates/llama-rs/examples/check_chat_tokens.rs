//! Diagnostic: print how a model's tokenizer encodes the chat-template special
//! markers. Useful for debugging chat-template/stop-token issues.
//!
//! Usage: cargo run --release -p llama-rs --example check_chat_tokens -- model.gguf

use std::env;

use ggml_rs::default_backend;
use gguf::GgufFile;
use llama_rs::{apply_chat_template, chat_stop_tokens, ChatMessage, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: check_chat_tokens <path.gguf>")?;
    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, default_backend())?;
    let tok = model.tokenizer();
    let arch = &model.config().arch;

    println!("== arch: {arch:?} ==");
    println!("== tokenizer: {:?} ==", tok.model());
    println!("== EOS id: {:?} (string: {:?}) ==",
             tok.eos(), tok.eos().and_then(|i| tok.token(i)));
    println!("== BOS id: {:?} (string: {:?}) ==",
             tok.bos(), tok.bos().and_then(|i| tok.token(i)));

    println!();
    println!("== special tokens lookup ==");
    let special_strings: &[&str] = match arch {
        llama_rs::Architecture::Gemma3 | llama_rs::Architecture::Gemma3n =>
            &["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>"],
        llama_rs::Architecture::Qwen3 | llama_rs::Architecture::Qwen2 =>
            &["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"],
        llama_rs::Architecture::Llama =>
            &["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        _ => &[],
    };
    for s in special_strings {
        println!("  {s:30} -> id={:?}", tok.token_id(s));
    }

    println!();
    println!("== chat_stop_tokens ==");
    for s in chat_stop_tokens(arch) {
        println!("  {s:30} -> id={:?}", tok.token_id(s));
    }

    println!();
    let msgs = [
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Capital of France?"),
    ];
    let prompt = apply_chat_template(arch, &msgs, true);
    println!("== rendered template ({} bytes) ==", prompt.len());
    println!("{prompt:?}");
    println!();
    let ids = tok.encode(&prompt, false)?;
    println!("== encoded to {} tokens ==", ids.len());
    for i in &ids {
        println!("  {:6} = {:?}", i, tok.token(*i));
    }

    Ok(())
}
