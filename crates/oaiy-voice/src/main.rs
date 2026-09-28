//! `oaiy-voice`: the resident speech server, or `oaiy-voice transcribe` to
//! transcribe WAV files from the command line (and time it).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{DType, Device};
use oaiy_voice::audio;
use oaiy_voice::cli::{self, Args, DeviceChoice, Precision};
use oaiy_voice::server::{self, Server, SharedStt, SpeechToText, TextToSpeech};
use oaiy_voice::stt::Transcriber;
use oaiy_voice::tts::{Setup, Speaker, Transcribe, Voices};

struct Parakeet(Transcriber);

impl SpeechToText for Parakeet {
    fn sample_rate(&self) -> usize {
        self.0.sample_rate()
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        let t = self.0.transcribe(samples).map_err(|e| e.to_string())?;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "[oaiy-voice] {:.2} s audio: features {:.1} ms, encoder {:.1} ms, decoder {:.1} ms ({} joint trips), RTF {:.4}",
            t.timings.audio.as_secs_f64(),
            ms(t.timings.features),
            ms(t.timings.encoder),
            ms(t.timings.decoder),
            t.timings.joint_trips,
            t.timings.rtf()
        );
        Ok(t.text)
    }
}

/// The CUDA GPU with the most free memory, and how much: speech takes what
/// is spare there first, and the engines that load later size themselves
/// around it.
fn roomiest_gpu() -> Option<(usize, u64)> {
    if !candle_core::utils::cuda_is_available() {
        return None;
    }
    let mut best: Option<(usize, u64)> = None;
    for n in 0..16 {
        let Ok(dev) = Device::new_cuda(n) else { break };
        if let Some((free, _)) = oaiy_tts::device_memory(&dev) {
            if best.is_none_or(|(_, most)| free > most) {
                best = Some((n, free));
            }
        }
    }
    best
}

/// The GPU speech runs on (`None`: the CPU).
fn pick_gpu(choice: DeviceChoice) -> Option<usize> {
    match choice {
        DeviceChoice::Cpu => None,
        DeviceChoice::Cuda(n) => Some(n),
        DeviceChoice::Auto => roomiest_gpu().map(|(n, free)| {
            eprintln!("[oaiy-voice] cuda:{n} has the most free memory ({:.1} GB): speech runs there", free as f64 / 1e9);
            n
        }),
    }
}

fn device_for(gpu: Option<usize>) -> Result<Device, String> {
    match gpu {
        None => Ok(Device::Cpu),
        Some(n) => Device::new_cuda(n).map_err(|e| format!("cuda:{n}: {e}")),
    }
}

fn pick_dtype(p: Precision, dev: &Device) -> DType {
    // TF32 products are a process-wide cuBLAS setting; this process runs one model.
    candle_core::cuda::set_gemm_reduced_precision_f32(p == Precision::Tf32 && dev.is_cuda());
    if !dev.is_cuda() {
        // Candle's CPU kernels are fastest in f32; half precision there only loses.
        if matches!(p, Precision::F16 | Precision::Bf16) {
            eprintln!("[oaiy-voice] the CPU runs in f32 (half precision is for the GPU)");
        }
        return DType::F32;
    }
    match p {
        Precision::Auto | Precision::F16 => DType::F16,
        Precision::F32 | Precision::Tf32 => DType::F32,
        Precision::Bf16 => DType::BF16,
    }
}

fn describe(dev: &Device, dtype: DType) -> String {
    let d = match dev {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => match dev.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            _ => "cuda".to_string(),
        },
        _ => format!("{:?}", dev.location()),
    };
    let tf32 = dtype == DType::F32 && dev.is_cuda() && candle_core::cuda::gemm_reduced_precision_f32();
    format!("{d} {}", if tf32 { "tf32" } else { dtype.as_str() })
}

fn load(path: &Path, gpu: Option<usize>, precision: Precision) -> Result<Transcriber, String> {
    let dev = device_for(gpu)?;
    let dtype = pick_dtype(precision, &dev);
    let started = Instant::now();
    let t = Transcriber::load(path, &dev, dtype).map_err(|e| format!("{}: {e}", path.display()))?;
    // One pass over a second of quiet, so the first caller does not wait
    // for the GPU's libraries and kernels to load.
    let quiet: Vec<f32> = (0..t.sample_rate()).map(|i| 1e-3 * ((i as f32) * 0.37).sin()).collect();
    t.transcribe(&quiet).map_err(|e| format!("{}: warm-up: {e}", path.display()))?;
    eprintln!("[oaiy-voice] loaded {} on {} in {:.1} s", t.name, describe(&dev, dtype), started.elapsed().as_secs_f64());
    Ok(t)
}

fn serve(args: Args) -> Result<(), String> {
    let gpu = pick_gpu(args.device);
    let (stt, id, device) = if args.mode.stt() {
        let path = args.stt_model.clone().ok_or("no speech-to-text model: pass --stt-model-dir (a Parakeet .nemo, a model.safetensors or a folder holding one)")?;
        let t = load(&args.find_model(&path), gpu, args.dtype)?;
        let (id, device) = (t.name.clone(), describe(t.device(), t.dtype()));
        let shared: SharedStt = Arc::new(Mutex::new(Box::new(Parakeet(t)) as Box<dyn SpeechToText>));
        (Some(shared), id, device)
    } else {
        (None, String::new(), "none".to_string())
    };
    let tts = if args.mode.tts() {
        let path = args.tts_model.clone().ok_or("no text-to-speech model: pass --tts-model-dir (a Qwen3-TTS Base folder)")?;
        let gpu = gpu.ok_or("text-to-speech runs on a CUDA GPU: there is none here, or this build has no CUDA (build with --features cuda)")?;
        // A voice clip with nothing written beside it is heard by this server's own ears.
        let transcribe = stt.clone().map(|stt| {
            Arc::new(move |samples: &[f32], rate: usize| {
                let mut stt = stt.lock().unwrap_or_else(|p| p.into_inner());
                let at = stt.sample_rate();
                stt.transcribe(&audio::resample(samples, rate, at))
            }) as Transcribe
        });
        let voices = Voices { dir: args.voices_dir.clone(), default: args.voice.clone() };
        let cache = args.voices_dir.as_ref().map(|d| d.join(".made"));
        let speaker = Speaker::start(Setup { model_dir: args.find_model(&path), gpu, voices, ffmpeg: args.ffmpeg.clone(), cache, transcribe })?;
        Some(Arc::new(speaker) as Arc<dyn TextToSpeech>)
    } else {
        None
    };
    server::run(Server::new(args.mode, stt, id, device, tts), &args.host, args.port)
}

/// `transcribe --model PATH [--device D] [--dtype T] [--repeat N] FILE...`
fn transcribe(argv: Vec<String>) -> Result<(), String> {
    let defaults = cli::parse(Vec::new())?;
    let (mut model, mut device, mut dtype, mut repeat, mut files) = (None, defaults.device, defaults.dtype, 1usize, Vec::new());
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--model" | "--stt-model-dir" => model = Some(PathBuf::from(val(&a)?)),
            "--device" => device = DeviceChoice::parse(&val(&a)?)?,
            "--dtype" => dtype = Precision::parse(&val(&a)?)?,
            "--repeat" => repeat = val(&a)?.parse().map_err(|_| "--repeat: a number")?,
            f if f.starts_with("--") => return Err(format!("unknown argument {f:?}\n{}", cli::USAGE)),
            f => files.push(PathBuf::from(f)),
        }
    }
    let model = model.ok_or("transcribe needs --model")?;
    let t = load(&model, pick_gpu(device), dtype)?;
    for f in files {
        let bytes = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        let wav = audio::parse_wav(&bytes).map_err(|e| format!("{}: {e}", f.display()))?;
        let samples = audio::resample(&wav.samples, wav.sample_rate as usize, t.sample_rate());
        let mut best: Option<oaiy_voice::stt::Transcript> = None;
        for _ in 0..repeat.max(1) {
            let r = t.transcribe(&samples).map_err(|e| e.to_string())?;
            if best.as_ref().is_none_or(|b| r.timings.total() < b.timings.total()) {
                best = Some(r);
            }
        }
        let r = best.expect("at least one run");
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        println!("{}\t{}", f.display(), r.text);
        eprintln!(
            "  {:.2} s: features {:.1} ms, encoder {:.1} ms, decoder {:.1} ms ({} joint trips, {} tokens), RTF {:.4}",
            r.timings.audio.as_secs_f64(),
            ms(r.timings.features),
            ms(r.timings.encoder),
            ms(r.timings.decoder),
            r.timings.joint_trips,
            r.tokens.len(),
            r.timings.rtf()
        );
    }
    Ok(())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let result = if argv.first().is_some_and(|a| a == "transcribe") {
        transcribe(argv.into_iter().skip(1).collect())
    } else {
        cli::parse(argv).and_then(serve)
    };
    if let Err(e) = result {
        eprintln!("[oaiy-voice] {e}");
        std::process::exit(1);
    }
}
