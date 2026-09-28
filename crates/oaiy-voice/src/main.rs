//! `oaiy-voice`: the resident speech server, or `oaiy-voice transcribe` to
//! transcribe WAV files from the command line (and time it).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use oaiy_voice::cli::{self, Args, DeviceChoice, Precision};
use oaiy_voice::server::{self, Server, SpeechToText};
use oaiy_voice::stt::Transcriber;
use oaiy_voice::audio;

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

fn pick_device(choice: DeviceChoice) -> Result<Device, String> {
    match choice {
        DeviceChoice::Cpu => Ok(Device::Cpu),
        DeviceChoice::Cuda(n) => Device::new_cuda(n).map_err(|e| format!("cuda:{n}: {e}")),
        DeviceChoice::Auto => {
            if candle_core::utils::cuda_is_available() {
                if let Ok(d) = Device::new_cuda(0) {
                    return Ok(d);
                }
            }
            Ok(Device::Cpu)
        }
    }
}

fn pick_dtype(p: Precision, dev: &Device) -> DType {
    match p {
        Precision::Auto if dev.is_cuda() => DType::F16,
        Precision::Auto | Precision::F32 => DType::F32,
        Precision::F16 => DType::F16,
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
    format!("{d} {}", dtype.as_str())
}

fn load(path: &Path, device: DeviceChoice, precision: Precision) -> Result<Transcriber, String> {
    let dev = pick_device(device)?;
    let dtype = pick_dtype(precision, &dev);
    let started = Instant::now();
    let t = Transcriber::load(path, &dev, dtype).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("[oaiy-voice] loaded {} on {} in {:.1} s", t.name, describe(&dev, dtype), started.elapsed().as_secs_f64());
    Ok(t)
}

fn serve(args: Args) -> Result<(), String> {
    let (stt, id, device) = if args.mode.stt() {
        let path = args.stt_model.clone().ok_or("no speech-to-text model: pass --stt-model-dir (a Parakeet .nemo, a model.safetensors or a folder holding one)")?;
        let t = load(&path, args.device, args.dtype)?;
        let (id, device) = (t.name.clone(), describe(t.device(), t.dtype()));
        (Some(Box::new(Parakeet(t)) as Box<dyn SpeechToText>), id, device)
    } else {
        (None, String::new(), "none".to_string())
    };
    if args.mode.tts() {
        eprintln!("[oaiy-voice] text-to-speech is not in this build: /v1/audio/speech answers 501");
    }
    server::run(Server::new(args.mode, stt, id, device), &args.host, args.port)
}

/// `transcribe --model PATH [--device D] [--dtype T] [--repeat N] FILE...`
fn transcribe(argv: Vec<String>) -> Result<(), String> {
    let (mut model, mut device, mut dtype, mut repeat, mut files) = (None, DeviceChoice::Auto, Precision::Auto, 1usize, Vec::new());
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
    let t = load(&model, device, dtype)?;
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
