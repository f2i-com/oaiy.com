use oaiy_engine::json::Json;
use std::io::{Read, Write};
fn main() {
    // a panic's message the last line too, as an error's (the server keeps the last: else the backtrace's note)
    let report = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        report(info);
        let message = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("a panic");
        let at = info.location().map_or(String::new(), |l| format!(" at {}:{}", l.file(), l.line()));
        eprintln!("{}", Json::obj([("error", Json::str(format!("{message}{at}")))]).to_json());
    }));
    // (one job a process: a GPU driver hung on a lost device ends it, the job failed rather than never done)
    #[cfg(feature = "webgpu")]
    ggml_rs_wgpu::end_process_on_hang();
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
    if args.first().is_some_and(|a| a == "--serve") {
        return serve();
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

/// `oaiy-media --serve`: Qwen Image jobs one after another, each a line of JSON on standard input, each one's answer a
/// line on standard output (its result, or `{"error": ...}`), its events on standard error as a job's are. The models a
/// job loads are kept for the next that names the same files (`pipeline::Kept`): a worker that OAIY keeps while
/// pictures come one after another. It ends when its standard input does.
fn serve() -> candle_core::Result<()> {
    use std::io::BufRead;
    let mut kept = oaiy_media::pipeline::Kept::default();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = (|| -> candle_core::Result<Json> {
            let j = Json::parse(line.as_bytes()).map_err(candle_core::Error::wrap)?;
            if j.get("architecture").is_some_and(|a| a.as_str() != Some("qwen-image")) || j.get("kind").is_some() {
                candle_core::bail!("a worker that serves takes Qwen Image jobs");
            }
            let r = oaiy_media::pipeline::Request::parse(&j).map_err(candle_core::Error::Msg)?;
            configure_cache(&r.output)?;
            oaiy_media::pipeline::generate_kept(&r, Some(&mut kept), |event| eprintln!("{}", event.to_json()))
        })();
        let answer = answer.unwrap_or_else(|e| {
            // (what it had is let go: a job that failed may have left a model half made)
            kept = oaiy_media::pipeline::Kept::default();
            Json::obj([("error", Json::str(e.to_string()))])
        });
        let mut out = std::io::stdout().lock();
        writeln!(out, "{}", answer.to_json())?;
        out.flush()?;
    }
    Ok(())
}

fn configure_cache(output: &std::path::Path) -> candle_core::Result<()> {
    let _ = output;
    Ok(())
}
