//! Where the time goes: the talker's frames and the codec's chunks, timed
//! apart (with the device synchronized around each part).
//!
//! cargo run --release -p oaiy-tts --features flash-attn --example bench -- \
//!     --voice clip.wav --transcript "..." --device 1 [--frames 60]
use oaiy_tts::codec::{CodecDecoder, CodecStream};
use oaiy_tts::talker::Talker;
use oaiy_tts::{text, Sampling, Tts};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut device, mut voice, mut transcript, mut frames) = (PathBuf::from("E:/models/Qwen3-TTS-12Hz-0.6B-Base"), 0usize, None, None, 60usize);
    let mut text_arg = "Sure, I can help with that. Your appointment is booked for Tuesday at three in the afternoon, and you will get a text to confirm it.".to_string();
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = value()?.into(),
            "--device" => device = value()?.parse()?,
            "--voice" => voice = Some(PathBuf::from(value()?)),
            "--transcript" => transcript = Some(value()?),
            "--frames" => frames = value()?.parse()?,
            "--text" => text_arg = value()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let dev = oaiy_tts::cuda(device)?;
    let voice = {
        let mut tts = Tts::load(&model, &dev)?;
        tts.voice_from_audio(&voice.ok_or("--voice is required")?, transcript.as_deref())?
    };
    let mut talker = Talker::load(&model, &dev)?;
    let codec = CodecDecoder::load(&model.join("speech_tokenizer").join("model.safetensors"), &dev)?;
    let tok = text::tokenizer(&model)?;
    let text_ids = text::encode(&tok, &text_arg)?;
    let ref_ids = text::encode(&tok, &voice.ref_text)?;
    let sync = || dev.synchronize();
    for round in 0..4 {
        // Rounds alternate where the draws happen: on the device, then on the host.
        let on_device = round % 2 == 0;
        sync()?;
        let t = Instant::now();
        let (prefill, trailing) = talker.prefill_clone(&text_ids, &ref_ids, &voice, None)?;
        let mut g = talker.start(&prefill, Some(trailing), Sampling { on_device, ..Sampling::default() })?;
        sync()?;
        let prefill_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut made = Vec::new();
        for _ in 0..frames {
            match talker.next_frame(&mut g)? {
                Some(f) => made.push(f),
                None => break,
            }
        }
        sync()?;
        let per_frame = t.elapsed().as_secs_f64() * 1e3 / made.len().max(1) as f64;
        println!("round {round} (draws on the {}): prefill ({} positions) {prefill_ms:.1} ms; {} frames at {per_frame:.2} ms a frame ({:.1} frames/s)", if on_device { "device" } else { "host" }, prefill.dim(1)?, made.len(), 1e3 / per_frame);
        for chunk in [1usize, 2, 4, 8] {
            let mut stream = CodecStream::primed(25, &voice.ref_codes);
            sync()?;
            let t = Instant::now();
            let mut n = 0;
            for c in made.chunks(chunk).take(8) {
                stream.push(&codec, c)?;
                n += 1;
            }
            sync()?;
            println!("  codec, {chunk} frames a chunk behind 25: {:.2} ms a chunk", t.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
    }
    Ok(())
}
