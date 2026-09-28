//! Parity with NeMo: the mel features, the encoder output and the greedy
//! tokens of real checkpoints against fixtures NeMo made on the CPU in f32.
//!
//! The fixtures are not in the repository (a few MB per model):
//! `tools/make_clips.py` writes the clips and `tools/nemo_fixtures.py` the
//! fixtures (its docstring has the Python setup). Each test is skipped
//! unless `OAIY_VOICE_FIXTURES` names the fixtures folder (with
//! `<model>/index.json`, `<clip>.mel.f32`, `<clip>.enc.f32`, and the clips
//! in `../clips/`) and the weights are under `OAIY_VOICE_MODELS` (default
//! `E:\models`). Run them in release:
//! `cargo test --release -p oaiy-voice parity -- --nocapture`; the GPU ones
//! also need a CUDA build and `OAIY_VOICE_CUDA=<gpu index>`.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use oaiy_engine::json::Json;

use super::encoder::Precision;
use super::Transcriber;
use crate::audio;

fn fixtures() -> Option<PathBuf> {
    std::env::var_os("OAIY_VOICE_FIXTURES").map(PathBuf::from).filter(|p| p.is_dir())
}

fn models() -> PathBuf {
    std::env::var_os("OAIY_VOICE_MODELS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"E:\models"))
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

struct Outcome {
    mel: f32,
    enc: f32,
    tokens_equal: usize,
    clips: usize,
}

/// Compare one model against its fixtures; `None` when they are not there.
fn check(fixture: &str, weights: &Path, dev: &Device, precision: Precision, check_features: bool) -> Option<Outcome> {
    let root = fixtures()?;
    let dir = root.join(fixture);
    let index = std::fs::read(dir.join("index.json")).ok()?;
    if !weights.exists() {
        eprintln!("{fixture}: {} not found, skipped", weights.display());
        return None;
    }
    let index = Json::parse(&index).unwrap();
    let t = Transcriber::load_with(weights, dev, precision).unwrap();
    let mut out = Outcome { mel: 0.0, enc: 0.0, tokens_equal: 0, clips: 0 };
    for (clip, want) in index.members() {
        let wav = audio::parse_wav(&std::fs::read(root.join("..").join("clips").join(format!("{clip}.wav"))).unwrap()).unwrap();
        assert_eq!(wav.sample_rate, 16_000);
        let f = t.features(&wav.samples);
        let mel_shape: Vec<usize> = want.get("mel").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect();
        assert_eq!(vec![f.n_mels, f.frames], mel_shape, "{clip}: mel shape");
        let mel = max_abs(&f.data, &read_f32(&dir.join(format!("{clip}.mel.f32"))));
        if check_features {
            assert!(mel < 1e-3, "{fixture} {clip}: mel differs by {mel}");
        }
        let enc = t.encode(&f).unwrap();
        let enc_shape: Vec<usize> = want.get("enc").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect();
        assert_eq!(enc.dims(), &enc_shape[..], "{clip}: encoder shape");
        let got: Vec<f32> = enc.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let e = max_abs(&got, &read_f32(&dir.join(format!("{clip}.enc.f32"))));
        let (tokens, _) = t.decode(&enc).unwrap();
        let want_tokens: Vec<u32> = want.get("tokens").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as u32).collect();
        let text = t.detokenize(&tokens);
        let want_text = want.get("text").and_then(Json::as_str).unwrap_or("");
        let same = tokens == want_tokens;
        eprintln!("{fixture} {clip}: mel {mel:.2e}, encoder {e:.2e}, tokens {}{}", if same { "equal" } else { "DIFFER" }, if same { String::new() } else { format!("\n  ours:   {text:?}\n  nemo:   {want_text:?}") });
        if same {
            assert_eq!(text, want_text.trim(), "{clip}: detokenized text");
        }
        out.mel = out.mel.max(mel);
        out.enc = out.enc.max(e);
        out.tokens_equal += same as usize;
        out.clips += 1;
    }
    Some(out)
}

fn report(name: &str, o: Option<Outcome>, enc_limit: f32, all_tokens: bool) {
    let Some(o) = o else {
        eprintln!("{name}: fixtures or weights not found, skipped");
        return;
    };
    eprintln!("{name}: {} clips, mel max |diff| {:.2e}, encoder max |diff| {:.2e}, tokens equal on {}/{}", o.clips, o.mel, o.enc, o.tokens_equal, o.clips);
    assert!(o.enc < enc_limit, "{name}: encoder differs by {}", o.enc);
    if all_tokens {
        assert_eq!(o.tokens_equal, o.clips, "{name}: transcripts differ");
    }
}

#[test]
fn parity_v2_nemo_cpu() {
    let w = models().join("parakeet-tdt-0.6b-v2").join("parakeet-tdt-0.6b-v2.nemo");
    report("v2", check("v2", &w, &Device::Cpu, Precision::f32(), true), 1e-3, true);
}

#[test]
fn parity_unified_nemo_cpu() {
    let w = models().join("parakeet-unified-en-0.6b").join("parakeet-unified-en-0.6b.nemo");
    report("unified", check("unified", &w, &Device::Cpu, Precision::f32(), true), 1e-3, true);
}

#[test]
fn parity_v3_nemo_cpu() {
    let w = models().join("parakeet-tdt-0.6b-v3").join("parakeet-tdt-0.6b-v3.nemo");
    report("v3 .nemo", check("v3", &w, &Device::Cpu, Precision::f32(), true), 1e-3, true);
}

#[test]
fn parity_v3_safetensors_cpu() {
    let w = models().join("parakeet-tdt-0.6b-v3").join("model.safetensors");
    report("v3 safetensors", check("v3", &w, &Device::Cpu, Precision::f32(), true), 1e-3, true);
}

#[test]
fn parity_ultra_safetensors_cpu() {
    let w = models().join("parakeet-ultra");
    report("ultra", check("ultra", &w, &Device::Cpu, Precision::f32(), true), 1e-3, true);
}

/// The GPU named by `OAIY_VOICE_CUDA` (an index), when this build has CUDA.
fn cuda() -> Option<Device> {
    let n: usize = std::env::var("OAIY_VOICE_CUDA").ok()?.parse().ok()?;
    Device::new_cuda(n).ok()
}

fn gpu_report(name: &str, fixture: &str, weights: PathBuf, dtype: DType, all_tokens: bool) {
    gpu_report_with(name, fixture, weights, Precision { weights: dtype, act: dtype }, all_tokens)
}

fn gpu_report_with(name: &str, fixture: &str, weights: PathBuf, precision: Precision, all_tokens: bool) {
    let Some(dev) = cuda() else {
        eprintln!("{name}: OAIY_VOICE_CUDA not set (or no CUDA), skipped");
        return;
    };
    // Half precision moves the encoder by about 1e-2; with f16 the transcripts
    // must not move. bf16 keeps 8 bits of mantissa and is only reported.
    report(name, check(fixture, &weights, &dev, precision, true), 0.25, all_tokens);
}

#[test]
fn parity_v2_cuda_f16() {
    gpu_report("v2 cuda f16", "v2", models().join("parakeet-tdt-0.6b-v2").join("parakeet-tdt-0.6b-v2.nemo"), DType::F16, true);
}

#[test]
fn parity_v2_cuda_bf16() {
    gpu_report("v2 cuda bf16", "v2", models().join("parakeet-tdt-0.6b-v2").join("parakeet-tdt-0.6b-v2.nemo"), DType::BF16, false);
}

#[test]
fn parity_v2_cuda_f32() {
    gpu_report("v2 cuda f32", "v2", models().join("parakeet-tdt-0.6b-v2").join("parakeet-tdt-0.6b-v2.nemo"), DType::F32, true);
}

#[test]
fn parity_unified_cuda_f16() {
    gpu_report("unified cuda f16", "unified", models().join("parakeet-unified-en-0.6b").join("parakeet-unified-en-0.6b.nemo"), DType::F16, true);
}

#[test]
fn parity_ultra_cuda_f16() {
    gpu_report("ultra cuda f16", "ultra", models().join("parakeet-ultra"), DType::F16, true);
}

#[test]
fn parity_v2_cuda_f16_mixed() {
    let p = Precision { weights: DType::F16, act: DType::F32 };
    gpu_report_with("v2 cuda f16 weights, f32 activations", "v2", models().join("parakeet-tdt-0.6b-v2").join("parakeet-tdt-0.6b-v2.nemo"), p, true);
}

#[test]
fn parity_v3_cuda_f16() {
    gpu_report("v3 cuda f16", "v3", models().join("parakeet-tdt-0.6b-v3").join("model.safetensors"), DType::F16, true);
}
