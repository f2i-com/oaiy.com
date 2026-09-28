//! The command line. The server takes the flags OAIY Desktop's `aokie-stt`
//! and `aokie-tts` service templates pass to `aokie-voice-server.exe`
//! (`--mode`, `--port`, `--stt-model-dir`, `--tts-model-dir`), accepts
//! Aokie's engine flags where they name what this server does, and adds
//! `--device` and `--dtype`. Those two can also come from the environment
//! (`OAIY_VOICE_DEVICE`, `OAIY_VOICE_DTYPE`), so a service template can set
//! them without changing its arguments; a flag wins over the environment.
//!
//! For speech it adds `--voices-dir` (voice clips by name), `--voice` (the
//! default: a name there, or a clip) and `--ffmpeg`. A model given by name
//! rather than path is looked for in `--model-dirs` (`OAIY_MODEL_DIRS`), the
//! model folders joined by the system's path separator.

use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Stt,
    Tts,
    Both,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Stt => "stt",
            Mode::Tts => "tts",
            Mode::Both => "both",
        }
    }

    pub fn stt(self) -> bool {
        matches!(self, Mode::Stt | Mode::Both)
    }

    pub fn tts(self) -> bool {
        matches!(self, Mode::Tts | Mode::Both)
    }

    /// Aokie's ports when `--port` is absent: 17921 (stt), 17922 (tts), 17920 (both).
    pub fn default_port(self) -> u16 {
        match self {
            Mode::Stt => 17_921,
            Mode::Tts => 17_922,
            Mode::Both => 17_920,
        }
    }
}

/// Where the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The CUDA GPU with the most free memory when this build has CUDA and
    /// one is there, else the CPU.
    Auto,
    Cpu,
    Cuda(usize),
}

impl DeviceChoice {
    pub fn parse(v: &str) -> Result<Self, String> {
        let v = v.trim().to_ascii_lowercase();
        match v.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda(0)),
            _ => {
                let n = v.strip_prefix("cuda:").or_else(|| v.strip_prefix("gpu:")).ok_or_else(|| format!("--device {v:?}: expected auto, cpu, cuda or cuda:N"))?;
                n.parse().map(Self::Cuda).map_err(|_| format!("--device {v:?}: bad GPU index"))
            }
        }
    }
}

/// Matrix-product precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// f16 on a GPU, f32 on the CPU.
    Auto,
    F32,
    /// f32 weights with TF32 tensor-core products (GPU only; f32 on the CPU).
    Tf32,
    F16,
    Bf16,
}

impl Precision {
    pub fn parse(v: &str) -> Result<Self, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "tf32" => Ok(Self::Tf32),
            "f16" | "fp16" | "float16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            other => Err(format!("--dtype {other:?}: expected auto, f32, tf32, f16 or bf16")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Args {
    pub mode: Mode,
    pub port: u16,
    pub host: String,
    pub stt_model: Option<PathBuf>,
    pub tts_model: Option<PathBuf>,
    pub device: DeviceChoice,
    pub dtype: Precision,
    /// Voice clips, by name.
    pub voices_dir: Option<PathBuf>,
    /// The default voice: a name in `voices_dir`, or a clip.
    pub voice: Option<String>,
    pub ffmpeg: PathBuf,
    /// Where a model given by name is looked for.
    pub model_dirs: Vec<PathBuf>,
}

impl Args {
    /// A model's folder or file: the path as given when it is there, else the
    /// first of the model folders that has it by that name.
    pub fn find_model(&self, given: &std::path::Path) -> PathBuf {
        if given.exists() || given.is_absolute() {
            return given.to_path_buf();
        }
        self.model_dirs.iter().map(|d| d.join(given)).find(|p| p.exists()).unwrap_or_else(|| given.to_path_buf())
    }
}

pub const USAGE: &str = "usage: oaiy-voice [--mode stt|tts|both] [--port N] [--host ADDR]
                  [--stt-model-dir PATH | --model PATH] [--tts-model-dir PATH]
                  [--device auto|cpu|cuda|cuda:N] [--dtype auto|f32|tf32|f16|bf16]
                  [--voices-dir DIR] [--voice NAME|CLIP] [--ffmpeg PATH] [--model-dirs DIRS]
       oaiy-voice transcribe --model PATH [--device ...] [--dtype ...] [--repeat N] FILE.wav...

PATH is a Parakeet .nemo, a model.safetensors (with config.json and
tokenizer.json beside it), or a folder holding either; the TTS model is a
Qwen3-TTS Base folder. A model named without a path is looked for in the
model folders. A voice is a clip (WAV, MP3...) in DIR, by its file name.";

/// Parse the server's flags (`--flag value` or `--flag=value`), with the
/// process environment for what they leave out.
pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    parse_with_env(args, |k| std::env::var(k).ok())
}

/// [`parse`] with the environment given (tests).
pub fn parse_with_env<I: IntoIterator<Item = String>>(args: I, env: impl Fn(&str) -> Option<String>) -> Result<Args, String> {
    let argv: Vec<String> = args.into_iter().collect();
    let mut mode = None;
    let mut port = None;
    let mut host = "127.0.0.1".to_string();
    let (mut stt_model, mut tts_model) = (None, None);
    let mut device = match env("OAIY_VOICE_DEVICE").filter(|v| !v.trim().is_empty()) {
        Some(v) => DeviceChoice::parse(&v).map_err(|e| format!("OAIY_VOICE_DEVICE: {e}"))?,
        None => DeviceChoice::Auto,
    };
    let mut dtype = match env("OAIY_VOICE_DTYPE").filter(|v| !v.trim().is_empty()) {
        Some(v) => Precision::parse(&v).map_err(|e| format!("OAIY_VOICE_DTYPE: {e}"))?,
        None => Precision::Auto,
    };
    let (mut voices_dir, mut voice) = (None, None);
    let mut ffmpeg = PathBuf::from("ffmpeg");
    let split = |v: &str| std::env::split_paths(v).filter(|p| !p.as_os_str().is_empty()).collect::<Vec<PathBuf>>();
    let mut model_dirs = env("OAIY_MODEL_DIRS").map(|v| split(&v)).unwrap_or_default();
    let mut i = 0;
    while i < argv.len() {
        let (flag, inline) = match argv[i].split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (argv[i].clone(), None),
        };
        let mut value = || -> Result<String, String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            i += 1;
            argv.get(i).cloned().ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--mode" => {
                mode = Some(match value()?.to_ascii_lowercase().as_str() {
                    "stt" => Mode::Stt,
                    "tts" => Mode::Tts,
                    "both" => Mode::Both,
                    other => return Err(format!("--mode {other:?}: expected stt, tts or both")),
                })
            }
            "--port" => port = Some(value()?.parse::<u16>().map_err(|_| "--port: expected 1..65535".to_string()).and_then(|p| if p == 0 { Err("--port: expected 1..65535".to_string()) } else { Ok(p) })?),
            "--host" => host = value()?,
            "--stt-model-dir" | "--model" => stt_model = Some(PathBuf::from(value()?)),
            "--tts-model-dir" => tts_model = Some(PathBuf::from(value()?)),
            "--device" => device = DeviceChoice::parse(&value()?)?,
            "--dtype" => dtype = Precision::parse(&value()?)?,
            "--voices-dir" => voices_dir = Some(PathBuf::from(value()?)),
            "--voice" => voice = Some(value()?).filter(|v| !v.trim().is_empty()),
            "--ffmpeg" => ffmpeg = PathBuf::from(value()?),
            "--model-dirs" => model_dirs = split(&value()?),
            // Aokie's engine choice: only Parakeet is served here.
            "--stt-engine" => {
                let v = value()?;
                if !v.eq_ignore_ascii_case("parakeet") {
                    return Err(format!("--stt-engine {v:?}: this server runs parakeet"));
                }
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
        i += 1;
    }
    let mode = mode.unwrap_or(Mode::Both);
    Ok(Args { mode, port: port.unwrap_or(mode.default_port()), host, stt_model, tts_model, device, dtype, voices_dir, voice, ffmpeg, model_dirs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(a: &[&str]) -> Result<Args, String> {
        parse_with_env(a.iter().map(|s| s.to_string()), |_| None)
    }

    #[test]
    fn desktop_template_flags() {
        let a = p(&["--mode", "stt", "--port", "8781", "--stt-model-dir", "C:/models/aokie/parakeet"]).unwrap();
        assert_eq!((a.mode, a.port), (Mode::Stt, 8781));
        assert_eq!(a.stt_model, Some(PathBuf::from("C:/models/aokie/parakeet")));
        assert_eq!((a.device, a.dtype, a.host.as_str()), (DeviceChoice::Auto, Precision::Auto, "127.0.0.1"));
        let a = p(&["--mode=tts", "--tts-model-dir=C:/x"]).unwrap();
        assert_eq!((a.mode, a.port), (Mode::Tts, 17_922));
    }

    #[test]
    fn device_and_dtype() {
        let a = p(&["--device", "cuda:1", "--dtype", "bf16", "--model", "m.nemo"]).unwrap();
        assert_eq!((a.device, a.dtype), (DeviceChoice::Cuda(1), Precision::Bf16));
        assert_eq!(DeviceChoice::parse("cuda").unwrap(), DeviceChoice::Cuda(0));
        assert_eq!(DeviceChoice::parse("CPU").unwrap(), DeviceChoice::Cpu);
        assert!(DeviceChoice::parse("cuda:x").is_err());
        assert!(Precision::parse("int8").is_err());
    }

    #[test]
    fn environment_sets_device_and_dtype_unless_a_flag_does() {
        let env = |k: &str| match k {
            "OAIY_VOICE_DEVICE" => Some("cuda:1".to_string()),
            "OAIY_VOICE_DTYPE" => Some("f32".to_string()),
            _ => None,
        };
        let a = parse_with_env(["--mode", "stt"].map(String::from), env).unwrap();
        assert_eq!((a.device, a.dtype), (DeviceChoice::Cuda(1), Precision::F32));
        let a = parse_with_env(["--device", "cpu"].map(String::from), env).unwrap();
        assert_eq!(a.device, DeviceChoice::Cpu);
        assert!(parse_with_env(Vec::<String>::new(), |k| (k == "OAIY_VOICE_DEVICE").then(|| "tpu".to_string())).is_err());
    }

    #[test]
    fn speech_flags_and_models_by_name() {
        let roots = std::env::join_paths([std::env::temp_dir().join("oaiy-voice-none"), std::env::temp_dir()]).unwrap().into_string().unwrap();
        let a = p(&["--mode", "both", "--voices-dir", "V", "--voice", "phone", "--ffmpeg", "C:/ff/ffmpeg.exe", "--model-dirs", &roots]).unwrap();
        assert_eq!((a.voices_dir.clone(), a.voice.as_deref(), a.ffmpeg.clone()), (Some(PathBuf::from("V")), Some("phone"), PathBuf::from("C:/ff/ffmpeg.exe")));
        assert_eq!(a.model_dirs.len(), 2);
        let name = format!("oaiy-voice-model-{}", std::process::id());
        std::fs::create_dir_all(std::env::temp_dir().join(&name)).unwrap();
        assert_eq!(a.find_model(std::path::Path::new(&name)), std::env::temp_dir().join(&name));
        assert_eq!(a.find_model(std::path::Path::new("not-here")), PathBuf::from("not-here"));
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(&name));
        let env = parse_with_env(Vec::<String>::new(), |k| (k == "OAIY_MODEL_DIRS").then(|| roots.clone())).unwrap();
        assert_eq!(env.model_dirs.len(), 2);
        assert_eq!(p(&["--voice", " "]).unwrap().voice, None);
    }

    #[test]
    fn defaults_and_errors() {
        let a = p(&[]).unwrap();
        assert_eq!((a.mode, a.port), (Mode::Both, 17_920));
        assert!(p(&["--port"]).is_err());
        assert!(p(&["--port", "0"]).is_err());
        assert!(p(&["--mode", "phone"]).is_err());
        assert!(p(&["--frobnicate"]).is_err());
        assert!(p(&["--stt-engine", "whisper"]).is_err());
        assert!(p(&["--stt-engine", "parakeet"]).is_ok());
    }
}
