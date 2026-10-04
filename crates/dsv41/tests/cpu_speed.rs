//! How fast the CPU model decodes on the real checkpoint (`DSV41_MODEL`, its `engram_meta.safetensors` beside it):
//! the measure of whether DeepSeek-V4.1 can be served without CUDA. `#[ignore]`d; run it with
//! `cargo test --release -p dsv41 --test cpu_speed -- --ignored --nocapture`.
use dsv41::model::{Model, ModelOptions};
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore = "loads the 510 GB checkpoint and decodes on the CPU"]
fn time_cpu_decode_steps() {
    let dir = std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"D:\deepseek\model"));
    let meta = dir.join("engram_meta.safetensors");
    let gb: usize = std::env::var("DSV41_CACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
    let opts = ModelOptions { max_seq: 256, expert_cache_bytes: gb << 30, direct_io: true };
    let t = Instant::now();
    let mut model = Model::load(&dir, &meta, &opts).unwrap();
    eprintln!("loaded in {:.1} s (expert cache {gb} GB)", t.elapsed().as_secs_f64());
    // A short prompt of ordinary token ids, then greedy steps.
    let prompt: Vec<u32> = vec![0, 19923, 14, 1035, 2580, 4194, 13];
    let t = Instant::now();
    let mut logits = model.forward(&prompt, 0, &mut |_, _| {}).unwrap();
    eprintln!("prompt of {} tokens in {:.1} s", prompt.len(), t.elapsed().as_secs_f64());
    let vocab = model.cfg.vocab_size;
    let mut pos = prompt.len();
    for step in 0..8 {
        let last = &logits[logits.len() - vocab..];
        let next = last.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
        let t = Instant::now();
        logits = model.forward(&[next], pos, &mut |_, _| {}).unwrap();
        pos += 1;
        eprintln!("step {step}: token {next}, {:.2} s", t.elapsed().as_secs_f64());
    }
}

/// An Agent's prompt is its system prompt and tool schemas, 1.5-3k tokens: what a first request would wait for. Real
/// text (this repository's docs), so the routing spreads over the experts as a prompt's does; one forward, each layer's
/// experts read once.
#[test]
#[ignore = "loads the 510 GB checkpoint and prefills on the CPU"]
fn time_cpu_prefill_of_an_agent_sized_prompt() {
    let dir = std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"D:\deepseek\model"));
    let meta = dir.join("engram_meta.safetensors");
    let gb: usize = std::env::var("DSV41_CACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
    let tokens: usize = std::env::var("DSV41_PROMPT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let tok = dsv41::tokenizer::Tokenizer::load(&dir).unwrap();
    let docs = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs");
    let text: String = ["WEBGPU.md", "STUDIO.md", "FLASHNEXT.md", "ORCASAQ.md"].iter().filter_map(|f| std::fs::read_to_string(format!("{docs}/{f}")).ok()).collect();
    let mut prompt = vec![0u32];
    prompt.extend(tok.encode(&text));
    assert!(prompt.len() >= tokens, "only {} tokens of text", prompt.len());
    prompt.truncate(tokens);
    let opts = ModelOptions { max_seq: tokens + 64, expert_cache_bytes: gb << 30, direct_io: true };
    let t = Instant::now();
    let mut model = Model::load(&dir, &meta, &opts).unwrap();
    eprintln!("loaded in {:.1} s (expert cache {gb} GB)", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let logits = model.forward(&prompt, 0, &mut |_, _| {}).unwrap();
    let secs = t.elapsed().as_secs_f64();
    eprintln!("prompt of {} tokens in {secs:.1} s ({:.1} tokens a second)", prompt.len(), prompt.len() as f64 / secs);
    let next = dsv41::model::argmax(&logits);
    let t = Instant::now();
    model.forward(&[next], prompt.len(), &mut |_, _| {}).unwrap();
    eprintln!("then a step: token {next}, {:.2} s", t.elapsed().as_secs_f64());
}
