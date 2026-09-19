//! Print the token-by-token breakdown of a chat-template-formatted prompt.
//! Useful for debugging whether special chat markers (<|im_end|>, <end_of_turn>, …)
//! are being mapped to their proper special-token ids vs being byte-shredded.

use std::env;

use ggml_rs::default_backend;
use gguf::GgufFile;
use llama_rs::{apply_chat_template, ChatMessage, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: debug_chat_tokens <path.gguf>")?;
    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, default_backend())?;
    let arch = &model.config().arch;

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hi"),
    ];
    let prompt = apply_chat_template(arch, &messages, true);

    println!("Architecture: {arch:?}");
    println!("Vocab size:   {}", model.tokenizer().vocab_size());
    println!("BOS:          {:?}", model.tokenizer().bos());
    println!("EOS:          {:?}", model.tokenizer().eos());
    println!();
    println!("--- Raw template ({} bytes) ---", prompt.len());
    for line in prompt.lines() {
        println!("  {line:?}");
    }
    println!();

    let ids = model.tokenizer().encode(&prompt, false)?;
    println!("--- Tokens ({} total) ---", ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let piece = model.tokenizer().token(id).unwrap_or("<?>");
        println!("  [{i:3}] {id:>6}  {piece:?}");
    }

    println!();
    println!("--- Stop-token resolution ---");
    for marker in &["<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|im_start|>", "<bos>", "<start_of_turn>"] {
        match model.tokenizer().token_id(marker) {
            Some(id) => println!("  {:<20} -> id {id}", marker),
            None     => println!("  {:<20} -> NOT FOUND in vocab", marker),
        }
    }

    Ok(())
}
