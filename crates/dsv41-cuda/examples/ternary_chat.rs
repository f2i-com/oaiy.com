//! Small single-prompt runner using the same memory/device settings as full_trial.
use std::path::Path;
use std::io::Write;
use std::time::Instant;
use oaiy_engine::json::Json as J;
use dsv41::{chat, detok::Detokenizer, tokenizer::Tokenizer, model::argmax};
use dsv41_cuda::{GpuModel, GpuOptions};

fn main() -> oaiy_engine::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let source = Path::new(args.get(1).expect("MODEL_DIR"));
    let prompt = args.get(2).expect("PROMPT");
    let max_new: usize = args.get(3).map(|v|v.parse().expect("integer MAX_NEW_TOKENS")).unwrap_or(32);
    assert!((1..=256).contains(&max_new));
    let tokenizer = Tokenizer::load(source)?;
    let detok = Detokenizer::load(source)?;
    let user = J::obj([("role",J::str("user")),("content",J::str(prompt))]);
    let ids = tokenizer.encode(&chat::encode(&[user], &chat::Options::default())?.prompt);
    assert!(ids.len()+max_new<=512, "Prompt plus completion exceeds 512-token context");
    let opts = GpuOptions { devices:vec![0,1], max_seq:512, expert_cache_bytes:140<<30,
        direct_io:true, vram_expert_bytes:Some(16<<30), vram_headroom_bytes:1<<30,
        cpu_expert_threads:Some(24), vision:false, residual_on_device:None };
    let mut model = GpuModel::load(source, &source.join("engram_meta.safetensors"), &opts)?;
    let t = Instant::now();
    let logits = model.forward(&ids,0)?;
    assert!(logits.iter().all(|v|v.is_finite()));
    eprintln!("Prefill: {:.3}s",t.elapsed().as_secs_f64());
    let mut token = argmax(&logits);
    let mut bytes = Vec::new();
    let mut times = Vec::new();
    for step in 0..max_new {
        bytes.extend_from_slice(detok.bytes(token));
        if token==model.cfg.eos_token_id || step+1==max_new {break;}
        let t = Instant::now();
        let logits = model.forward(&[token],ids.len()+step)?;
        assert!(logits.iter().all(|v|v.is_finite()));
        times.push(t.elapsed().as_secs_f64());
        token=argmax(&logits);
    }
    println!("{}",String::from_utf8_lossy(&bytes));
    std::io::stdout().flush()?;
    let seconds:f64=times.iter().sum();
    eprintln!("Decode: {} forwards, {:.3}s, {:.3} forwards/s",times.len(),seconds,times.len() as f64/seconds.max(1e-30));
    Ok(())
}
