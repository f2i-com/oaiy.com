use nrob::json::Json;
use std::io::{Read, Write};
fn main() {
    if let Err(e) = run() {
        eprintln!(
            "{}",
            Json::obj([("error", Json::str(e.to_string()))]).to_json()
        );
        std::process::exit(1);
    }
}
fn run() -> candle_core::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--help") {
        println!("nrob-diffusion --request request.json | --stdin\nNative Rust Qwen Image 2.1 and LTX video. JSON: base, transformer, adapter (optional), prompt or prompts, output_dir, n, width, height, steps, seed, device, cfg.\nTurbo: 6 steps by default (4 supported), CFG=1. Base: 40 steps, CFG=6.\nVideo: kind=video, model=ltx-2.3|ltx-2.5|sulphur-2, transformer, text_encoder, tokenizer (Gemma 3), vae, prompt, output_dir. Eight steps; memory=auto|gpu|ram|ssd, ram_gb, vram_gb.");
        return Ok(());
    }
    let bytes = match args.first().map(String::as_str) {
        Some("--stdin") => {
            let mut b = Vec::new();
            std::io::stdin()
                .take(2 * 1024 * 1024 + 1)
                .read_to_end(&mut b)?;
            b
        }
        Some("--request") if args.len() == 2 => std::fs::read(&args[1])?,
        _ => candle_core::bail!("use --request request.json or --stdin (see --help)"),
    };
    if bytes.len() > 2 * 1024 * 1024 {
        candle_core::bail!("request exceeds 2 MiB");
    }
    let j = Json::parse(&bytes).map_err(candle_core::Error::wrap)?;
    if j.get("kind").and_then(Json::as_str) == Some("video") {
        let r = nrob_diffusion::ltx::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = nrob_diffusion::ltx::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    let r = nrob_diffusion::pipeline::Request::parse(&j).map_err(candle_core::Error::Msg)?;
    configure_cache(&r.output)?;
    let result = nrob_diffusion::pipeline::generate(&r, |event| {
        eprintln!("{}", event.to_json());
    })?;
    writeln!(std::io::stdout(), "{}", result.to_json())?;
    Ok(())
}

fn configure_cache(output: &std::path::Path) -> candle_core::Result<()> {
    #[cfg(feature = "cuda")]
    {
        // Configure the driver before CUDA or tokenizer threads are started.
        // A writable cache avoids recompiling PTX on every worker invocation.
        if std::env::var_os("CUDA_CACHE_PATH").is_none() {
            let cache = output.join(".cuda-cache");
            std::fs::create_dir_all(&cache)?;
            std::env::set_var("CUDA_CACHE_PATH", cache.canonicalize()?);
        }
        if std::env::var_os("CUDA_CACHE_MAXSIZE").is_none() {
            std::env::set_var("CUDA_CACHE_MAXSIZE", "1073741824");
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = output;
    Ok(())
}
