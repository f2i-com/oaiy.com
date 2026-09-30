use oaiy_engine::json::Json;
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
        println!("oaiy-media --request request.json | --stdin\nNative Rust Qwen Image 2.1, SDXL and LTX video. JSON: base, transformer, adapter (optional), prompt or prompts, output_dir, n, width, height, steps, seed, device, cfg.\nTurbo: 6 steps by default (4 supported), CFG=1. Base: 40 steps, CFG=6.\nSDXL: architecture=sdxl, checkpoint, tokenizer (CLIP tokenizer.json), prompt, negative_prompt, output_dir. Defaults: 1024x1024, 16 steps, CFG=2.5, DPM++ 2M Karras, clip_skip=1 (penultimate); no turbo or reference images.\nVideo: kind=video, model=ltx-2.3|ltx-2.5|sulphur-2, transformer, text_encoder, tokenizer (Gemma 3), vae, prompt, output_dir, optional image (starting frame), end_image (final-frame guidance) and cache_dir (bounded prompt cache). Eight steps; memory=auto|gpu|ram|ssd, ram_gb, vram_gb.");
        println!("Klein: architecture=flux2-klein-4b, transformer (original BFL safetensors), text_encoder (Qwen3-4B), tokenizer, vae (Flux2), prompt, output_dir. Distilled: 4 steps, CFG=1. Style adapters: loras=[{{path,strength}}]. Native text-to-image; dimensions multiple of 16; no Qwen turbo, negative prompts or references. Explicit variant=base for Klein base 4B weights.");
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
    if let Some(a) = j.get("architecture") {
        if !a.as_str().is_some_and(|s| ["qwen-image", "sdxl", "flux2-klein-4b"].contains(&s)) {
            candle_core::bail!("unsupported architecture; use qwen-image, sdxl or flux2-klein-4b");
        }
    }
    if j.get("architecture").and_then(Json::as_str) == Some("flux2-klein-4b") {
        let r = oaiy_media::klein::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::klein::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("architecture").and_then(Json::as_str) == Some("sdxl") {
        let r = oaiy_media::sdxl::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::sdxl::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("voice") {
        let r = oaiy_media::tts::DesignRequest::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::tts::design_voice(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("music_quantize") {
        let r = oaiy_media::music::quant::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        let result = oaiy_media::music::quant::run(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("music") {
        let r = oaiy_media::music::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::music::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("picture") {
        let r = oaiy_media::picture::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        let result = oaiy_media::picture::run(&r, |event| eprintln!("{}", event.to_json()))?;
        println!("{}", result.to_json());
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("model3d") {
        let r = oaiy_media::model3d::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        let result = oaiy_media::model3d::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        println!("{}", result.to_json());
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("sound") {
        let r = oaiy_media::sound::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::sound::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("speech") {
        let r = oaiy_media::tts::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::tts::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    if j.get("kind").and_then(Json::as_str) == Some("video") {
        let r = oaiy_media::ltx::Request::parse(&j).map_err(candle_core::Error::Msg)?;
        configure_cache(&r.output)?;
        let result = oaiy_media::ltx::generate(&r, |event| eprintln!("{}", event.to_json()))?;
        writeln!(std::io::stdout(), "{}", result.to_json())?;
        return Ok(());
    }
    let r = oaiy_media::pipeline::Request::parse(&j).map_err(candle_core::Error::Msg)?;
    configure_cache(&r.output)?;
    let result = oaiy_media::pipeline::generate(&r, |event| {
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
            // A plain absolute path: the driver ignores `\?\`-prefixed
            // (canonical Windows) paths and would JIT every kernel again.
            std::env::set_var("CUDA_CACHE_PATH", std::path::absolute(&cache)?);
        }
        if std::env::var_os("CUDA_CACHE_MAXSIZE").is_none() {
            // The driver's maximum (4 GiB): one GPU's worth of kernels for
            // every model the worker runs.
            std::env::set_var("CUDA_CACHE_MAXSIZE", "4294967296");
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = output;
    Ok(())
}
