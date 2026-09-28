//! Speak a line in a cloned voice and report how fast it came.
//!
//! cargo run --release -p oaiy-tts --features flash-attn --example speak -- \
//!     --voice clip.mp3 --transcript "What the clip says." \
//!     --text "Hello! How can I help you today?" --out out.wav --device 1
//!
//! Options: --model DIR (default E:/models/Qwen3-TTS-12Hz-0.6B-Base),
//! --device N (CUDA ordinal, default 0), --transcript-file PATH, --repeat N
//! (speak the line N times; the first run warms the GPU up), --seed N,
//! --language NAME, --chunk N (frames per chunk after the first),
//! --ffmpeg PATH, --no-cache (make the voice again).
use oaiy_tts::{SpeakOptions, Tts};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut device, mut voice, mut transcript, mut text, mut out) = (PathBuf::from("E:/models/Qwen3-TTS-12Hz-0.6B-Base"), 0usize, None, None, None, PathBuf::from("out.wav"));
    let (mut repeat, mut seed, mut language, mut chunk, mut ffmpeg, mut cache) = (1usize, 0u64, None, None, None, true);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = value()?.into(),
            "--device" => device = value()?.parse().map_err(|e| format!("--device: {e}"))?,
            "--voice" => voice = Some(PathBuf::from(value()?)),
            "--transcript" => transcript = Some(value()?),
            "--transcript-file" => transcript = Some(std::fs::read_to_string(value()?).map_err(|e| e.to_string())?),
            "--text" => text = Some(value()?),
            "--out" => out = value()?.into(),
            "--repeat" => repeat = value()?.parse().map_err(|e| format!("--repeat: {e}"))?,
            "--seed" => seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--language" => language = Some(value()?),
            "--chunk" => chunk = Some(value()?.parse().map_err(|e| format!("--chunk: {e}"))?),
            "--ffmpeg" => ffmpeg = Some(PathBuf::from(value()?)),
            "--no-cache" => cache = false,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let voice_path = voice.ok_or("--voice CLIP (an mp3 or wav of the voice) is required")?;
    let text = text.ok_or("--text is required")?;
    let dev = oaiy_tts::cuda(device).map_err(|e| e.to_string())?;
    let free_before = oaiy_tts::device_memory(&dev).map(|m| m.0);
    let t = Instant::now();
    let mut tts = Tts::load(&model, &dev).map_err(|e| e.to_string())?;
    println!("loaded {} (width {}) in {:.2} s: {:.0} MB of weights on the device", model.display(), tts.width(), t.elapsed().as_secs_f64(), tts.weight_bytes() as f64 / 1e6);
    if let Some(f) = ffmpeg {
        tts.ffmpeg = f;
    }
    if !cache {
        tts.voice_cache = None;
    }
    let mut options = SpeakOptions { sampling: oaiy_tts::Sampling { seed, ..Default::default() }, ..Default::default() };
    if let Some(l) = language {
        options.language = l;
    }
    if let Some(c) = chunk {
        options.chunk_frames = c;
    }
    tts.options = options;
    let t = Instant::now();
    let voice = tts.voice_from_audio(&voice_path, transcript.as_deref()).map_err(|e| e.to_string())?;
    println!("voice from {}: {} reference frames ({:.1} s), in {:.2} s", voice_path.display(), voice.ref_codes.len(), voice.ref_codes.len() as f64 / 12.5, t.elapsed().as_secs_f64());
    let cancel = AtomicBool::new(false);
    let mut pcm: Vec<i16> = Vec::new();
    for run in 0..repeat.max(1) {
        pcm.clear();
        let mut chunks = 0;
        let report = tts
            .speak(&text, &voice, &cancel, |c| {
                chunks += 1;
                pcm.extend_from_slice(c);
            })
            .map_err(|e| e.to_string())?;
        println!(
            "run {}: {:?}, {} frames, {:.2} s of audio in {} chunks; prefill {:.0} ms, first audio {:.0} ms, total {:.2} s, real-time factor {:.3}",
            run + 1,
            report.finish,
            report.frames,
            report.audio_seconds,
            chunks,
            report.prefill_seconds * 1e3,
            report.first_chunk_seconds.unwrap_or(f64::NAN) * 1e3,
            report.total_seconds,
            report.real_time_factor()
        );
    }
    if let (Some(before), Some((after, _))) = (free_before, oaiy_tts::device_memory(&dev)) {
        println!("device memory in use by this process: {:.0} MB", before.saturating_sub(after) as f64 / 1e6);
    }
    std::fs::write(&out, oaiy_tts::audio::wav_bytes(&pcm, oaiy_tts::audio::SAMPLE_RATE)).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("wrote {}", out.display());
    Ok(())
}
