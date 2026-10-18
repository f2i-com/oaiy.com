//! `oaiy-voice`: the resident speech server, or `oaiy-voice transcribe` to
//! transcribe WAV files from the command line (and time it).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// The GPU with the most free memory, and how much, as nvidia-smi counts them (which is how the WebGPU backend
/// numbers them too): speech takes what is spare there first, and the engines that load later size themselves
/// around it. None where nvidia-smi does not answer (a card of another make: the first GPU then).
fn roomiest_gpu() -> Option<(usize, u64)> {
    free_by_smi().and_then(|rows| rows.into_iter().max_by_key(|&(_, free)| free))
}

/// How long nvidia-smi is given: a driver that hangs (a card lost, a driver mid-update) held the start for good.
const SMI_WITHIN: Duration = Duration::from_secs(5);

/// Free memory per GPU, from nvidia-smi (in its order, the PCI bus's).
fn free_by_smi() -> Option<Vec<(usize, u64)>> {
    let mut command = std::process::Command::new("nvidia-smi");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console window for the child.
        command.creation_flags(0x0800_0000);
    }
    command.args(["--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]);
    let out = output_within(&mut command, SMI_WITHIN)?;
    let rows = parse_free(&out);
    (!rows.is_empty()).then_some(rows)
}

/// What a program prints, where it ends well within `within`; one that does not is ended (and nothing is had of it).
fn output_within(command: &mut std::process::Command, within: Duration) -> Option<String> {
    use std::io::Read as _;
    use std::process::Stdio;
    let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut stdout = child.stdout.take()?;
    // (read beside the wait, so a program that prints more than a pipe holds is not stopped by it)
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let until = Instant::now() + within;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < until => std::thread::sleep(Duration::from_millis(20)),
            _ => break None,
        }
    };
    let Some(status) = status else {
        eprintln!("[oaiy-voice] nvidia-smi did not answer within {} s: the first GPU is used", within.as_secs());
        let _ = child.kill();
        let _ = child.wait();
        return None;
    };
    let text = reader.join().ok()?;
    status.success().then_some(text)
}

/// `index, free MiB` lines, as bytes.
fn parse_free(text: &str) -> Vec<(usize, u64)> {
    text.lines()
        .filter_map(|line| {
            let (index, free) = line.split_once(',')?;
            Some((index.trim().parse().ok()?, free.trim().parse::<u64>().ok()? * 1024 * 1024))
        })
        .collect()
}

/// The one GPU `CUDA_VISIBLE_DEVICES` names, if it names one by number: how OAIY Desktop's GPU picker assigns a
/// service its card (in nvidia-smi's order, the WebGPU backend's too).
fn pinned_gpu() -> Option<usize> {
    std::env::var("CUDA_VISIBLE_DEVICES").ok().and_then(|v| v.trim().parse().ok())
}

/// The GPU text to speech runs on, through WebGPU (`None`: `--device cpu`, which leaves it none).
fn pick_gpu(choice: DeviceChoice) -> Option<usize> {
    match choice {
        DeviceChoice::Cpu => None,
        DeviceChoice::Gpu(n) => Some(n),
        DeviceChoice::Auto => Some(match (pinned_gpu(), roomiest_gpu()) {
            // (the desktop's GPU picker pins a service by CUDA_VISIBLE_DEVICES, which WebGPU does not read: read here)
            (Some(n), _) => {
                eprintln!("[oaiy-voice] pinned to GPU {n} (CUDA_VISIBLE_DEVICES): speech is made there");
                n
            }
            (None, Some((n, free))) => {
                eprintln!("[oaiy-voice] GPU {n} has the most free memory ({:.1} GB): speech is made there", free as f64 / 1e9);
                n
            }
            (None, None) => 0,
        }),
    }
}

/// Speech to text runs on the CPU (Candle's; its GPU path was CUDA's), in f32: Candle's CPU kernels are fastest so,
/// and half precision there only loses.
fn pick_dtype(p: Precision) -> DType {
    if matches!(p, Precision::F16 | Precision::Bf16 | Precision::Tf32) {
        eprintln!("[oaiy-voice] speech to text runs on the CPU in f32 (the other precisions were the CUDA build's)");
    }
    DType::F32
}

fn describe(dtype: DType) -> String {
    format!("cpu {}", dtype.as_str())
}

/// The speech-to-text model, on the CPU.
fn load(path: &Path, precision: Precision) -> Result<Transcriber, String> {
    let dev = Device::Cpu;
    let dtype = pick_dtype(precision);
    let started = Instant::now();
    let t = Transcriber::load(path, &dev, dtype).map_err(|e| format!("{}: {e}", path.display()))?;
    // One pass over a second of quiet, so the first caller does not wait for the first use's set-up.
    let quiet: Vec<f32> = (0..t.sample_rate()).map(|i| 1e-3 * ((i as f32) * 0.37).sin()).collect();
    t.transcribe(&quiet).map_err(|e| format!("{}: warm-up: {e}", path.display()))?;
    eprintln!("[oaiy-voice] loaded {} on {} in {:.1} s", t.name, describe(dtype), started.elapsed().as_secs_f64());
    Ok(t)
}

fn serve(args: Args, key: Option<String>) -> Result<(), String> {
    let (stt, id, device) = if args.mode.stt() {
        let path = args.stt_model.clone().ok_or("no speech-to-text model: pass --stt-model-dir (a Parakeet .nemo, a model.safetensors or a folder holding one)")?;
        let t = load(&args.find_model(&path), args.dtype)?;
        let (id, device) = (t.name.clone(), describe(t.dtype()));
        let shared: SharedStt = Arc::new(Mutex::new(Box::new(Parakeet(t)) as Box<dyn SpeechToText>));
        (Some(shared), id, device)
    } else {
        (None, String::new(), "none".to_string())
    };
    let tts = if args.mode.tts() {
        let path = args.tts_model.clone().ok_or("no text-to-speech model: pass --tts-model-dir (a Qwen3-TTS Base folder)")?;
        let gpu = pick_gpu(args.device).ok_or("text-to-speech runs on a GPU, through WebGPU: --device cpu leaves it none (--mode stt is speech-to-text alone)")?;
        // A voice clip with nothing written beside it is heard by this server's own ears (resampled before they are
        // waited on).
        let transcribe = stt.clone().map(|stt| {
            let at = stt.lock().unwrap_or_else(|p| p.into_inner()).sample_rate();
            Arc::new(move |samples: &[f32], rate: usize| {
                let samples = audio::resample(samples, rate, at);
                stt.lock().unwrap_or_else(|p| p.into_inner()).transcribe(&samples)
            }) as Transcribe
        });
        let voices = Voices { dir: args.voices_dir.clone(), default: args.voice.clone() };
        let cache = args.voices_dir.as_ref().map(|d| d.join(".made"));
        let speaker = Speaker::start(Setup { model_dir: args.find_model(&path), gpu, voices, ffmpeg: args.ffmpeg.clone(), cache, transcribe })?;
        Some(Arc::new(speaker) as Arc<dyn TextToSpeech>)
    } else {
        None
    };
    server::run(Server::new(args.mode, stt, id, device, tts).with_key(key), &args.host, args.port)
}

/// The key requests must carry, from `OAIY_VOICE_API_KEY` (out of sight of other users' `ps`, as a flag would not be),
/// taken out of the environment so the programs this one starts (FFmpeg) are not handed it. Read first thing, before
/// any thread runs.
fn api_key() -> Option<String> {
    let key = std::env::var("OAIY_VOICE_API_KEY").ok().map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    std::env::remove_var("OAIY_VOICE_API_KEY");
    key
}

/// `--watch-stdin`: the server ends when stdin reaches its end (its supervisor, which holds the other end, has gone).
fn watch_stdin() {
    std::thread::spawn(|| {
        let mut sink = [0u8; 256];
        let mut stdin = std::io::stdin();
        while std::io::Read::read(&mut stdin, &mut sink).is_ok_and(|n| n > 0) {}
        eprintln!("[oaiy-voice] stdin closed: its supervisor has gone, so the server ends");
        std::process::exit(0);
    });
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
    if matches!(device, DeviceChoice::Gpu(_)) {
        eprintln!("[oaiy-voice] speech to text runs on the CPU: --device names the GPU text to speech is made on");
    }
    let t = load(&model, dtype)?;
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
    // GPUs numbered as nvidia-smi numbers them (by PCI bus), so the one it
    // reports with the most free memory is the one CUDA opens.
    if std::env::var_os("CUDA_DEVICE_ORDER").is_none() {
        std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    }
    let key = api_key();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let result = if argv.first().is_some_and(|a| a == "transcribe") {
        transcribe(argv.into_iter().skip(1).collect())
    } else {
        cli::parse(argv).and_then(|args| {
            if args.watch_stdin {
                watch_stdin();
            }
            serve(args, key)
        })
    };
    if let Err(e) = result {
        eprintln!("[oaiy-voice] {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_free;

    #[test]
    fn nvidia_smi_s_free_memory_is_read_per_gpu() {
        assert_eq!(parse_free("0, 11074\n1, 32168\n"), vec![(0, 11_074 * 1024 * 1024), (1, 32_168 * 1024 * 1024)]);
        assert_eq!(parse_free("No devices were found\n"), vec![]);
    }
}
