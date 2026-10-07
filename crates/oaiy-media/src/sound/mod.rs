//! Native Rust MOSS-SoundEffect v2.0: sound effects from a description, up to 30
//! seconds at 48 kHz. A Qwen3-1.7B text encoder reads the prompt (with its
//! duration, as trained), a 1.3B DiT (Wan 2.1 style, 1-D) turns noise into DAC
//! latents by flow matching with classifier-free guidance, and the DAC decoder
//! makes the audio. As the reference does, it always denoises the full 30 s and
//! keeps the length asked for.
//!
//! `backend` "webgpu" runs it on WebGPU ([`crate::sound_wgpu`]: any GPU), as a worker built with WebGPU and without CUDA
//! does by default (Candle's needs CUDA: its models BF16).
pub mod dac;
pub mod dit;
pub mod pth;

use crate::ltx::store::Store;
use crate::tts::model::{Cache, Decoder};
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The text encoder's input is padded to this many tokens (and cut there).
pub(crate) const TEXT_LEN: usize = 512;

#[derive(Clone, Debug)]
pub struct Request {
    /// The pipeline folder: model_index.json, text_encoder/, tokenizer/,
    /// transformer/, vae/.
    pub model_dir: PathBuf,
    pub prompt: String,
    pub negative_prompt: String,
    pub seconds: f64,
    pub steps: usize,
    pub cfg: f64,
    pub shift: f64,
    pub seed: u64,
    pub device: usize,
    pub output: PathBuf,
    /// Testing: start from this noise (F32 little-endian, (128, frames)) instead of the seed's.
    pub noise_file: Option<PathBuf>,
    /// On WebGPU.
    pub webgpu: bool,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::to_owned);
        let f = |k: &str, d: f64| j.get(k).and_then(Json::as_f64).unwrap_or(d);
        let r = Self {
            model_dir: s("model_dir").filter(|p| !p.trim().is_empty()).ok_or("sound: missing model_dir")?.into(),
            prompt: s("prompt").map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).ok_or("sound: describe the sound")?,
            negative_prompt: s("negative_prompt").unwrap_or_default(),
            seconds: f("seconds", 10.),
            steps: j.get("steps").and_then(Json::as_i64).unwrap_or(100).max(1) as usize,
            cfg: f("cfg_scale", 4.),
            shift: f("shift", 5.),
            seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0).max(0) as u64,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            output: s("output_dir").ok_or("sound: missing output_dir")?.into(),
            noise_file: s("noise_file").map(PathBuf::from),
            webgpu: match s("backend").as_deref() {
                Some("webgpu") => true,
                Some("cuda" | "cpu") => false,
                Some(other) => return Err(format!("sound: backend must be webgpu, cuda or cpu, not {other}")),
                None => cfg!(feature = "webgpu"),
            },
        };
        if r.webgpu && !cfg!(feature = "webgpu") {
            return Err("sound: this build has no WebGPU (the webgpu feature)".into());
        }
        if r.prompt.len() > 4000 || r.negative_prompt.len() > 4000 {
            return Err("sound: the prompt is limited to 4000 bytes".into());
        }
        if !(0.1..=30.0).contains(&r.seconds) {
            return Err("sound: seconds must be 0.1..30".into());
        }
        if !(1..=500).contains(&r.steps) || !(0.0..=20.0).contains(&r.cfg) || !(0.1..=20.0).contains(&r.shift) {
            return Err("sound: steps must be 1..500, cfg_scale 0..20 and shift 0.1..20".into());
        }
        Ok(r)
    }
}

pub(crate) fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([("stage", Json::str(stage)), ("current", Json::Int(current as i64)), ("total", Json::Int(total as i64))])
}

fn device(index: usize) -> Result<Device> {
    // The models run in BF16, which the CPU backend cannot multiply.
    {
        let _ = index;
        candle_core::bail!("sound effects need a GPU: this oaiy-media was built without CUDA (build it with --features cuda or flash-attn)")
    }
}

/// Seeded standard normal noise (splitmix64, Box-Muller), rounded to BF16 as the
/// reference's noise is.
pub(crate) fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (uniform(), uniform());
        let r = (-2. * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out.push(half::bf16::from_f64(r * a.cos()).to_f32());
        out.push(half::bf16::from_f64(r * a.sin()).to_f32());
    }
    out.truncate(n);
    out
}

/// The prompter's whitespace clean: runs of whitespace to one space, trimmed.
pub(crate) fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Qwen3-1.7B's final (normed) hidden states for each prompt, zero-padded to
/// `TEXT_LEN` rows: (1, TEXT_LEN, hidden) BF16. An empty prompt is all zeros.
pub(crate) fn encode_texts(dir: &Path, prompts: &[&str], dev: &Device) -> Result<Vec<Tensor>> {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json")).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;
    let te = dir.join("text_encoder");
    let cfg = Json::parse(&std::fs::read(te.join("config.json"))?).map_err(candle_core::Error::wrap)?;
    let n = |k: &str| cfg.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| candle_core::Error::Msg(format!("text encoder config: no {k}")));
    let hidden = n("hidden_size")?;
    let mut store = Store::open(&te, 0)?;
    let decoder = Decoder::load(
        &mut store,
        "model",
        n("num_hidden_layers")?,
        n("num_attention_heads")?,
        n("num_key_value_heads")?,
        n("head_dim")?,
        cfg.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6),
        TEXT_LEN,
        cfg.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
        dev,
    )?;
    let mut out = Vec::new();
    for p in prompts {
        let text = clean(p);
        let mut ids = tok.encode(text.as_str(), true).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?.get_ids().to_vec();
        ids.truncate(TEXT_LEN);
        if ids.is_empty() {
            out.push(Tensor::zeros((1, TEXT_LEN, hidden), DType::BF16, dev)?);
            continue;
        }
        let emb = store.rows("model.embed_tokens.weight", &ids, dev)?.to_dtype(DType::BF16)?.unsqueeze(0)?;
        let h = decoder.forward(&emb, &mut Cache::new(decoder.layers()))?;
        let pad = TEXT_LEN - ids.len();
        out.push(if pad == 0 { h } else { Tensor::cat(&[h, Tensor::zeros((1, pad, hidden), DType::BF16, dev)?], 1)? });
    }
    Ok(out)
}

/// The flow-matching noise levels: `steps` from 1 toward 0 (the last one short of
/// it), shifted toward noise, then 0.
pub(crate) fn sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let mut s: Vec<f64> = (0..steps).map(|i| 1. - i as f64 / steps as f64).map(|x| shift * x / (1. + (shift - 1.) * x)).collect();
    s.push(0.);
    s
}

pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    #[cfg(feature = "webgpu")]
    if r.webgpu {
        return crate::sound_wgpu::generate(r, report);
    }
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    let dev = device(r.device)?;
    let index = Json::parse(&std::fs::read(r.model_dir.join("model_index.json"))?).map_err(candle_core::Error::wrap)?;
    let full_seconds = index.get("max_inference_seconds").and_then(Json::as_i64).unwrap_or(30) as f64;
    if r.seconds > full_seconds {
        candle_core::bail!("sound: at most {full_seconds} seconds");
    }
    // The prompt carries its duration, as in training.
    let seconds = (r.seconds * 10.).round() / 10.;
    let prompt = format!("{} duration: {seconds:.1}s", r.prompt.trim());
    report(event("encoding_prompt", 0, 1));
    let texts = encode_texts(&r.model_dir, &[&prompt, &r.negative_prompt], &dev)?;
    let load_started = Instant::now();
    report(event("loading_sound_model", 0, 1));
    let dac = dac::Dac::load(&r.model_dir.join("vae").join("vae_128d_48k.pth"), &dev)?;
    let dit = dit::Dit::load(&r.model_dir.join("transformer"), &dev)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let (positive, negative) = (dit.context(&texts[0])?, dit.context(&texts[1])?);
    drop(texts);
    // The whole window's latents: (1, 128, frames).
    let frames = (dac.cfg.sample_rate as f64 * full_seconds) as usize / dac.cfg.hop;
    let channels = dit.cfg.in_dim;
    let values = match &r.noise_file {
        Some(p) => {
            let bytes = std::fs::read(p)?;
            if bytes.len() != channels * frames * 4 {
                candle_core::bail!("sound: {} holds {} bytes, not {channels} x {frames} F32", p.display(), bytes.len());
            }
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        None => noise(r.seed, channels * frames),
    };
    let mut x = Tensor::from_vec(values, (1, channels, frames), &dev)?;
    let s = sigmas(r.steps, r.shift);
    let denoise_started = Instant::now();
    for i in 0..r.steps {
        report(event("generating_sound", i, r.steps));
        let t = s[i] * 1000.;
        let v = if r.cfg == 1.0 {
            dit.forward(&x, t, &[&positive])?
        } else {
            // Both texts in one batch.
            let both = dit.forward(&Tensor::cat(&[&x, &x], 0)?, t, &[&positive, &negative])?;
            let (pos, neg) = (both.narrow(0, 0, 1)?, both.narrow(0, 1, 1)?);
            (&neg + ((pos - &neg)? * r.cfg)?)?
        };
        x = (x + (v * (s[i + 1] - s[i]))?)?;
    }
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    drop(positive);
    drop(negative);
    drop(dit);
    report(event("decoding_sound", 0, 1));
    let decode_started = Instant::now();
    // Only the frames asked for are decoded, with a margin past them so that the
    // convolutions near the cut see what they would in the whole window.
    let keep = ((seconds * dac.cfg.sample_rate as f64) as usize).div_ceil(dac.cfg.hop);
    let margin = 64;
    let decoded = dac.decode(&x.narrow(2, 0, (keep + margin).min(frames))?)?;
    let samples = (seconds * dac.cfg.sample_rate as f64) as usize;
    let audio: Vec<f32> = decoded.flatten_all()?.narrow(0, 0, samples)?.to_vec1()?;
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("sound-{stamp}-{}.wav", r.seed));
    crate::tts::write_wav(&path, &audio, dac.cfg.sample_rate)?;
    report(event("generating_sound", r.steps, r.steps));
    Ok(Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(dac.cfg.sample_rate as i64)),
        ("duration", Json::Num(audio.len() as f64 / dac.cfg.sample_rate as f64)),
        ("prompt", Json::str(&prompt)),
        ("steps", Json::Int(r.steps as i64)),
        ("cfg_scale", Json::Num(r.cfg)),
        ("load_seconds", Json::Num(load_seconds)),
        ("denoise_seconds", Json::Num(denoise_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
    ]))
}
