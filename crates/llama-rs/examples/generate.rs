//! Load a GGUF, run inference, stream tokens to stdout.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example generate -- path/to/model.gguf "Hello, world"
//!   cargo run --release -p llama-rs --features cuda --example generate -- path/to/model.gguf "Hi" cuda

use std::env;
use std::io::Write;
use std::time::Instant;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{Model, SampleParams, Sampler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path   = args.next().ok_or("usage: generate <path.gguf> [prompt] [backend]")?;
    let prompt = args.next().unwrap_or_else(|| "Hello".to_string());
    let backend_name = args.next().unwrap_or_else(|| "cpu".to_string());

    let backend = pick_backend(&backend_name)?;

    let load_t0 = Instant::now();
    let gguf = GgufFile::open(&path)?;
    let model = Model::load(&gguf, backend)?;
    let load_ms = load_t0.elapsed().as_millis();

    let cfg = model.config();
    eprintln!(
        "loaded {:?} on {} backend; cfg = vocab={}, layers={}, heads={}, kv={}, hd={}, ctx={} ({} ms)",
        cfg.arch,
        model.backend().name(),
        cfg.vocab_size,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.context_length,
        load_ms,
    );

    let mut kv = model.new_kv_cache(2048);
    let mut sampler = Sampler::new(SampleParams {
        temperature:    0.8,
        top_k:          Some(40),
        top_p:          Some(0.95),
        min_p:          Some(0.05),
        repeat_penalty: Some(1.1),
        repeat_last_n:  64,
        ..Default::default()
    });

    let prompt_ids = model.tokenizer().encode(&prompt, true)?;
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // Seed sampler history with the prompt so repetition penalty considers it.
    for &t in &prompt_ids {
        sampler.observe(t);
    }

    let pre_t0 = Instant::now();
    let logits = model.forward(&prompt_ids, &mut kv);
    let pre_ms = pre_t0.elapsed().as_millis();
    eprintln!("prefill: {pre_ms} ms ({:.1} tok/s)",
              prompt_ids.len() as f32 * 1000.0 / pre_ms.max(1) as f32);

    print!("{prompt}");
    std::io::stdout().flush()?;

    let mut current = sampler.sample(&model.last_logits(&logits));
    let max_new = 64;
    // For benchmarking, set BENCH=1 to ignore EOS and run to max_new tokens.
    let bench = std::env::var("BENCH").is_ok();
    let dec_t0 = Instant::now();
    let mut produced = 0u32;
    for _ in 0..max_new {
        if !bench {
            if let Some(eos) = model.tokenizer().eos() {
                if current == eos { break; }
            }
        }
        let piece = model.tokenizer().decode(&[current]);
        print!("{piece}");
        std::io::stdout().flush()?;
        produced += 1;

        let logits = model.forward(&[current], &mut kv);
        current = sampler.sample(&model.last_logits(&logits));
    }
    let dec_ms = dec_t0.elapsed().as_millis();
    println!();
    eprintln!("decode: {dec_ms} ms, {} tokens ({:.1} tok/s)",
              produced,
              produced as f32 * 1000.0 / dec_ms.max(1) as f32);

    Ok(())
}

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(std::sync::Arc::new(ggml_rs_cuda::CudaBackend::new(0)?)),
        #[cfg(not(feature = "cuda"))]
        "cuda" => Err("CUDA backend not enabled. Build with `--features cuda`".into()),
        "auto" => {
            #[cfg(feature = "cuda")]
            { match ggml_rs_cuda::CudaBackend::new(0) {
                Ok(b)  => { eprintln!("auto: CUDA available, using cuda:0"); Ok(std::sync::Arc::new(b)) }
                Err(e) => { eprintln!("auto: CUDA unavailable ({e:?}), falling back to CPU"); Ok(default_backend()) }
            } }
            #[cfg(not(feature = "cuda"))]
            { eprintln!("auto: built without --features cuda, using CPU"); Ok(default_backend()) }
        }
        other => Err(format!("unknown backend `{other}` (try `cpu`, `cuda`, or `auto`)").into()),
    }
}
