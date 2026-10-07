//! Interactive multi-turn chat REPL with conversation memory.
//!
//! Usage:
//!   cargo run --release --features cuda -p llama-rs --example chat_repl -- model.gguf cuda
//!
//! Slash commands:
//!   /clear      reset conversation history
//!   /sys <text> set the system prompt and reset history
//!   /history    print the current message history
//!   /tokens     show prompt + decode token counts for the last turn
//!   /quit       exit
//!
//! KV-cache is kept across turns. Each turn we re-encode the full history
//! with the architecture's chat template, then feed only the suffix beyond what
//! the cache already contains. Falls back to full re-prefill on `/clear`,
//! `/sys`, or in the rare case where assistant-side decode/encode round-trips
//! don't match exactly.

use std::env;
use std::io::{self, BufRead, Write};

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{apply_chat_template, ChatMessage, Model, SampleParams};

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("CUDA backend not enabled. Build with `--features cuda`".into()),
        "auto" => {
            { eprintln!("auto: built without --features cuda, using CPU"); Ok(default_backend()) }
        }
        other => Err(format!("unknown backend `{other}` (try `cpu`, `cuda`, or `auto`)").into()),
    }
}

fn print_help() {
    eprintln!("Commands:");
    eprintln!("  /clear           reset conversation");
    eprintln!("  /sys <prompt>    set system prompt and reset");
    eprintln!("  /history         show messages so far");
    eprintln!("  /tokens          report last turn's prompt + decode tokens");
    eprintln!("  /quit            exit");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: chat_repl <path.gguf> [backend]")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;
    let gguf = GgufFile::open(&path)?;
    let n_params: u64 = gguf.tensors().iter().map(|t| t.numel()).sum();
    let model = Model::load(&gguf, backend)?;
    let arch = &model.config().arch;
    eprintln!("loaded {arch:?} on {} ({} params)",
              model.backend().name(),
              human_count(n_params));
    eprintln!("type /help for commands. Send an empty line to skip an empty turn.");
    eprintln!();

    // Per-arch chat stop tokens are registered automatically by `generate_with_kv`.
    let mut history: Vec<ChatMessage> = vec![ChatMessage::system(
        "You are a helpful, concise assistant. Reply directly without preamble.",
    )];
    let stdin = io::stdin();
    let mut last_prompt_tokens: usize = 0;
    let mut last_cached_tokens: usize = 0;
    let mut last_decode_tokens: usize = 0;

    // KV cache + bookkeeping for incremental prompt extension. We keep the
    // most recently rendered chat-template string and the assistant's textual
    // response, so each new turn we can compute the trailer string (the chat
    // markers that close the prior turn and open the new one) and tokenize
    // *just* that trailer rather than re-tokenizing the whole history. This
    // matters because BPE merges can re-arrange across the boundary between
    // the cached prefix and the sampled response, which would otherwise force
    // a full re-prefill on every turn.
    let mut kv = model.new_kv_cache(8192);
    let mut kv_tokens: Vec<u32> = Vec::new();
    let mut prior_template_str = String::new();
    let mut prior_response_str = String::new();

    loop {
        eprint!("> ");
        io::stderr().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        // Slash commands.
        if let Some(rest) = trimmed.strip_prefix('/') {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let cmd = parts.next().unwrap_or("");
            let arg = parts.next().unwrap_or("").trim();
            match cmd {
                "quit" | "exit" | "q" => break,
                "help" | "h" | "?" => { print_help(); continue; }
                "clear" => {
                    let sys = history.first().cloned();
                    history.clear();
                    if let Some(s) = sys { history.push(s); }
                    kv.reset();
                    kv_tokens.clear();
                    prior_template_str.clear();
                    prior_response_str.clear();
                    eprintln!("[history cleared]");
                    continue;
                }
                "sys" => {
                    if arg.is_empty() {
                        eprintln!("[usage: /sys <new system prompt>]");
                    } else {
                        history.clear();
                        history.push(ChatMessage::system(arg.to_string()));
                        kv.reset();
                        kv_tokens.clear();
                        prior_template_str.clear();
                        prior_response_str.clear();
                        eprintln!("[system prompt set; history reset]");
                    }
                    continue;
                }
                "history" => {
                    for (i, m) in history.iter().enumerate() {
                        eprintln!("  [{i}] {:?}: {}", m.role, m.content);
                    }
                    continue;
                }
                "tokens" => {
                    eprintln!(
                        "[last turn: {} prompt ({} cached, {} new) + {} decode]",
                        last_prompt_tokens,
                        last_cached_tokens,
                        last_prompt_tokens.saturating_sub(last_cached_tokens),
                        last_decode_tokens,
                    );
                    continue;
                }
                other => {
                    eprintln!("[unknown command /{other}; try /help]");
                    continue;
                }
            }
        }

        // Append user turn and render the new chat template.
        history.push(ChatMessage::user(trimmed.to_string()));
        let new_template_str = apply_chat_template(arch, &history, true);

        // We expect new_template_str to be a strict suffix-extension of
        // (prior_template_str + prior_response_str): the chat templates are
        // append-only, and the sampled response text is what filled the open
        // assistant turn. If that invariant holds we can tokenize just the
        // trailer; otherwise we full-reset (Qwen3 with its `<think>` wrapper
        // around the assistant opener trips this on every turn — correct, just
        // slower).
        let cached_str_len = prior_template_str.len() + prior_response_str.len();
        let trailer: String = if cached_str_len > 0
            && new_template_str.len() >= cached_str_len
            && new_template_str.starts_with(&prior_template_str)
            && new_template_str[prior_template_str.len()..].starts_with(&prior_response_str)
        {
            new_template_str[cached_str_len..].to_string()
        } else {
            // Mismatch (or first turn) — full reset.
            if !kv_tokens.is_empty() {
                kv.reset();
                kv_tokens.clear();
            }
            new_template_str.clone()
        };

        // Tokenize the trailer with add_bos=false. The trailer for the first
        // turn includes the architecture's BOS marker as a literal string that
        // the tokenizer recognises; for subsequent turns it starts with a
        // special "end of turn" token, so BPE stays local to the trailer.
        let trailer_ids = model.tokenizer().encode(&trailer, false)?;
        last_prompt_tokens = kv_tokens.len() + trailer_ids.len();
        last_cached_tokens = kv_tokens.len();

        // High-level generate path: hands the borrowed `&mut kv` straight to
        // the iterator. Seed sampler history with prior turns' tokens so the
        // repetition penalty considers the full conversation, not just the
        // latest trailer. Stop tokens (per-arch chat markers) are registered
        // automatically by `generate_with_kv`.
        let params = SampleParams {
            temperature:    0.7,
            top_k:          Some(40),
            top_p:          Some(0.9),
            min_p:          Some(0.05),
            repeat_penalty: Some(1.1),
            repeat_last_n:  64,
            ..Default::default()
        };
        let max_new = 1024;
        let cached_history: Vec<u32> = kv_tokens.clone();
        kv_tokens.extend_from_slice(&trailer_ids);

        let mut response = String::new();
        let mut produced = 0usize;
        {
            let iter = model
                .generate_with_kv(&trailer_ids, params, max_new, &mut kv)
                .with_history(&cached_history);
            for tok in iter {
                let piece = model.tokenizer().decode(&[tok]);
                response.push_str(&piece);
                print!("{piece}");
                io::stdout().flush()?;
                kv_tokens.push(tok);
                produced += 1;
            }
        }
        last_decode_tokens = produced;

        println!();
        // Stash for the next turn's incremental-prompt computation. The chat
        // template's open-assistant marker is in `new_template_str`; the model
        // filled it with `response`.
        prior_template_str = new_template_str;
        prior_response_str = response.clone();
        history.push(ChatMessage::assistant(response));
    }

    Ok(())
}

/// Render an integer parameter count as e.g. "1.2B", "135M", "3.8B".
fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
    else if n >= 1_000 { format!("{}K", n / 1_000) }
    else { n.to_string() }
}
