//! VENDORED-LOCAL: Wave D diagnostic — print greedy token ids for a prompt,
//! so token-id-exactness regressions can be checked against known baselines
//! (e.g. Qwen3-30B-A3B starts `[12095, 13, 15920, ...]` on the bench prompt).
//!
//! Usage:
//!   cargo run --release -p llama-rs --features cuda --example greedy_ids -- \
//!     path/to/model.gguf "prompt" [max_new] [cuda|cpu]
//!   ... -- path/to/model.gguf stream [max_new]    (streamed experts: 4G RAM + 8G VRAM cache on CUDA)

use std::env;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{Model, SampleParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path   = args.next().ok_or("usage: greedy_ids <path.gguf> [prompt] [max_new] [backend]")?;
    let prompt = args.next().unwrap_or_else(|| {
        "Write a C function that parses a JSON array of integers and explain its edge cases."
            .to_string()
    });
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let backend_name = args.next().unwrap_or_else(|| "cuda".to_string());

    let model = if prompt == "stream" {
        // Streaming mode: greedy_ids <model.gguf> stream [max_new]
        #[cfg(feature = "cuda")]
        {
            let cuda = std::sync::Arc::new(ggml_rs_cuda::CudaBackend::new(0)?);
            let m = Model::open_streaming(
                &path,
                cuda.clone(),
                4u64 << 30, // same 4G RAM budget as the streaming bench
            )?;
            if let Some(shared) = m.stream_shared() {
                shared
                    .enable_device_cache(cuda, 8usize << 30)
                    .map_err(|e| format!("device cache: {e}"))?;
            }
            m
        }
        #[cfg(not(feature = "cuda"))]
        {
            Model::open_streaming(&path, default_backend(), 4u64 << 30)?
        }
    } else {
        let backend: std::sync::Arc<dyn Backend> = match backend_name.as_str() {
            "cpu" => default_backend(),
            #[cfg(feature = "cuda")]
            "cuda" => std::sync::Arc::new(ggml_rs_cuda::CudaBackend::new(0)?),
            other => return Err(format!("unknown backend '{other}'").into()),
        };
        let gguf = GgufFile::open(&path)?;
        Model::load(&gguf, backend)?
    };
    let prompt = if prompt == "stream" {
        "Write a C function that parses a JSON array of integers and explain its edge cases."
            .to_string()
    } else {
        prompt
    };
    let ids = model.tokenizer().encode(&prompt, true)?;
    let out: Vec<u32> = model.generate(&ids, SampleParams::greedy(), max_new).collect();
    if let Some(e) = model.expert_stream_error() {
        return Err(format!("expert stream: {e}").into());
    }
    println!("{out:?}");
    Ok(())
}
