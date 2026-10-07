//! MiniMax Music 3: complete songs (vocals and instruments, 44.1 kHz
//! stereo) from lyrics and a music description, natively.
//!
//! Two stages run one after the other, so only one is ever in memory:
//! the autoregressive stage ([`lm`]: an 8B language model and a depth
//! decoder write 25 frames a second) and the acoustic stage ([`acoustic`]:
//! a flow-matching transformer and the Flow-VAE decoder render them).
//!
//! `backend` "webgpu" runs both on WebGPU ([`crate::music_lm_wgpu`], [`crate::music_wgpu`]: any GPU), as a worker
//! built with WebGPU and without CUDA does by default (Candle's language model is BF16, which its CPU cannot multiply).
pub mod acoustic;
pub mod lm;
pub mod quant;

use crate::residency::Budget;
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Language-model frames per second of audio (24000 / 960).
pub const FRAMES_PER_SECOND: f64 = 25.;
pub const MAX_FRAMES: usize = 9_000;
pub const MAX_PROMPT_TOKENS: usize = 5_000;

/// Python's `str.splitlines()` line boundaries.
fn split_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            out.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = chars.peek() {
                    chars.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// `(?<!\*)\*([^*\n]+)\*(?!\*)` -> the inner text (Rust's regex has no
/// look-around).
fn strip_single_stars(line: &str) -> String {
    let c: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '*' && (i == 0 || c[i - 1] != '*') {
            let mut j = i + 1;
            while j < c.len() && c[j] != '*' && c[j] != '\n' {
                j += 1;
            }
            if j > i + 1 && j < c.len() && c[j] == '*' && c.get(j + 1) != Some(&'*') {
                out.extend(&c[i + 1..j]);
                i = j + 1;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// The description as the model was trained on it: `<|tag value|>` becomes
/// `tag is value`, Markdown residue goes, blank lines collapse.
pub fn clean_caption(caption: &str) -> String {
    use regex::Regex;
    let tag = Regex::new(r"<\|([^|]*)\|>").unwrap();
    let text = tag.replace_all(caption, |m: &regex::Captures| {
        let inner = m[1].trim();
        match inner.split_once(char::is_whitespace) {
            Some((a, b)) => format!("{a} is {}", b.trim_start()),
            None => inner.to_string(),
        }
    });
    let heading = Regex::new(r"^\s{0,3}#{1,6}\s+").unwrap();
    let bullet = Regex::new(r"^\s*[*+-]\s+").unwrap();
    let star_bullet = Regex::new(r"^\s*\*\s+").unwrap();
    let bold = Regex::new(r"\*\*([^*]+)\*\*").unwrap();
    let lines: Vec<String> = split_lines(&text)
        .into_iter()
        .map(|line| {
            let mut line = heading.replace(line, "").into_owned();
            line = bullet.replace(&line, "").into_owned();
            line = star_bullet.replace(&line, "").into_owned();
            while line.contains("**") {
                let updated = bold.replace_all(&line, "$1").into_owned();
                if updated == line {
                    break;
                }
                line = updated;
            }
            strip_single_stars(&line).trim_end().to_string()
        })
        .collect();
    let text = lines.join("\n");
    let rule = Regex::new(r"(?m)^\s*[-*_]{3,}\s*$").unwrap();
    let text = rule.replace_all(&text, "").replace("• ", "").replace("    ", "");
    Regex::new(r"\n{2,}").unwrap().replace_all(&text, "\n").into_owned()
}

/// Lyrics as the model expects them: structure tags alone on their lines
/// (text after a leading tag is dropped), tags lower-cased, `[start]` first.
pub fn normalize_lyrics(lyrics: &str) -> String {
    use regex::Regex;
    let leading = Regex::new(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)").unwrap();
    let text = lyrics
        .split('\n')
        .map(|line| match leading.captures(line) {
            Some(m) => m[1].trim().to_string(),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text = text.replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n");
    let text = Regex::new(r"\[([^\]]+)\]").unwrap().replace_all(&text, |m: &regex::Captures| format!("[{}]", m[1].to_lowercase()));
    format!("[start]\n{text}")
}

pub fn prompt_text(caption: &str, lyrics: &str) -> String {
    format!(
        "<|im_start|><|caption_start|>{}<|caption_end|><|lyrics_start|>{}<|lyrics_end|><|im_end|><|audio_start|>",
        clean_caption(caption),
        normalize_lyrics(lyrics)
    )
}

/// The conditional prompt and its unconditional twin (every token but the
/// first and the last two masked).
pub fn prompt_ids(dir: &Path, caption: &str, lyrics: &str) -> Result<[Vec<u32>; 2]> {
    let path = dir.join("tokenizer").join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&path).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
    let ids = tok.encode(prompt_text(caption, lyrics), false).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?.get_ids().to_vec();
    if ids.len() > MAX_PROMPT_TOKENS {
        candle_core::bail!("the lyrics and description come to {} tokens; the limit is {MAX_PROMPT_TOKENS}", ids.len());
    }
    if ids.len() < 3 {
        candle_core::bail!("the prompt is too short");
    }
    let mut uncond = ids.clone();
    let n = uncond.len();
    for id in &mut uncond[1..n - 2] {
        *id = lm::AUDIO_CFG;
    }
    Ok([ids, uncond])
}

pub struct Request {
    /// The MiniMax-Music3 folder (language_model/, rvq_depth_decoder/,
    /// condition_encoder/, transformer/, vocoder/, tokenizer/).
    pub model_dir: PathBuf,
    /// The music description (style, mood, vocals, instruments, tempo...).
    pub prompt: String,
    pub lyrics: String,
    pub output: PathBuf,
    pub seed: u64,
    pub device: usize,
    /// Upper bound on the length; the song may end sooner.
    pub max_seconds: f64,
    pub steps: usize,
    pub guidance: f64,
    /// The transformer's precision: BF16 (the official setting) or F32.
    pub dtype: DType,
    pub greedy: bool,
    pub budget: Budget,
    /// A quantized language model (see [`quant`]) instead of the BF16 one.
    pub language_model: Option<PathBuf>,
    /// On WebGPU (the transformer's weights f16 there, whatever `dtype`).
    pub webgpu: bool,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::to_owned);
        let r = Self {
            model_dir: s("model_dir").filter(|p| !p.trim().is_empty()).ok_or("music: missing model_dir")?.into(),
            prompt: s("prompt").filter(|t| !t.trim().is_empty()).ok_or("music: describe the music (prompt)")?,
            lyrics: s("lyrics").filter(|t| !t.trim().is_empty()).ok_or("music: lyrics must not be empty (use \"[Intro]\\n(instrumental)\" for an instrumental)")?,
            output: s("output_dir").ok_or("music: missing output_dir")?.into(),
            seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0).max(0) as u64,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            max_seconds: j.get("max_seconds").and_then(Json::as_f64).unwrap_or(60.),
            steps: j.get("steps").and_then(Json::as_i64).unwrap_or(30) as usize,
            guidance: j.get("guidance").and_then(Json::as_f64).unwrap_or(1.7),
            dtype: match s("precision").as_deref() {
                None | Some("bf16") => DType::BF16,
                Some("f32") => DType::F32,
                Some(o) => return Err(format!("music: precision must be bf16 or f32, not {o}")),
            },
            greedy: j.get("greedy").and_then(Json::as_bool).unwrap_or(false),
            budget: Budget::parse(j)?,
            language_model: s("language_model").filter(|p| !p.trim().is_empty()).map(PathBuf::from),
            webgpu: crate::pipeline::backend_is_webgpu(j, "music")?,
        };
        if r.webgpu && !cfg!(feature = "webgpu") {
            return Err("music: this build has no WebGPU (the webgpu feature)".into());
        }
        if r.prompt.len() > 20_000 || r.lyrics.len() > 20_000 {
            return Err("music: prompt and lyrics are limited to 20000 bytes each".into());
        }
        if !(1.0..=MAX_FRAMES as f64 / FRAMES_PER_SECOND).contains(&r.max_seconds) {
            return Err(format!("music: max_seconds must be 1..{}", MAX_FRAMES as f64 / FRAMES_PER_SECOND));
        }
        if !(1..=200).contains(&r.steps) || !(0.0..=20.0).contains(&r.guidance) {
            return Err("music: steps must be 1..200 and guidance 0..20".into());
        }
        Ok(r)
    }
}

fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([("stage", Json::str(stage)), ("current", Json::Int(current as i64)), ("total", Json::Int(total as i64))])
}

/// 16-bit PCM stereo WAV from (2, samples).
pub fn write_stereo_wav(path: &Path, left: &[f32], right: &[f32], rate: usize) -> std::io::Result<()> {
    let q = |s: f32| ((s.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes();
    let mut data = Vec::with_capacity(left.len() * 4);
    for (l, r) in left.iter().zip(right) {
        data.extend_from_slice(&q(*l));
        data.extend_from_slice(&q(*r));
    }
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&(rate as u32).to_le_bytes());
    out.extend_from_slice(&(rate as u32 * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    std::fs::write(path, out)
}

fn device(index: usize) -> Result<Device> {
    let _ = index;
    Ok(Device::Cpu)
}

/// Window `k`'s starting noise, (1, 128, len), from the seed alone.
pub(crate) fn noise(seed: u64, k: usize, len: usize) -> Result<Tensor> {
    let mut rng = lm::Rng::new(seed ^ 0xA5A5_0000_0000_0000 ^ (k as u64).wrapping_mul(0x9E37_79B9));
    Tensor::from_vec(rng.normals(acoustic::LATENT_CHANNELS * len), (1, acoustic::LATENT_CHANNELS, len), &Device::Cpu)
}

pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    #[cfg(feature = "webgpu")]
    if r.webgpu {
        return generate_webgpu(r, report);
    }
    let dev = device(r.device)?;
    let ids = prompt_ids(&r.model_dir, &r.prompt, &r.lyrics)?;
    let max_frames = ((r.max_seconds * FRAMES_PER_SECOND) as usize).clamp(1, MAX_FRAMES);

    // Stage 1: the language model writes the song, frame by frame.
    let lm_dir = r.model_dir.join("language_model");
    let layers = 36;
    report(event("loading_music_model", 0, layers));
    // The depth decoder and the output rows load first; the layers leave room
    // for the song's KV cache and the prompt's activations.
    let depth = lm::Depth::load(&r.model_dir.join("rvq_depth_decoder"), &dev)?;
    let kv = lm::kv_bytes(ids[0].len() + max_frames + 2);
    let budget = r.budget.with_headroom(kv + (3 << 29));
    let mut model = lm::Lm::load(&lm_dir, r.language_model.as_deref(), &budget, &dev, |i| report(event("loading_music_model", i, layers)))?;
    let lm_report = model.report();
    let load_seconds = started.elapsed().as_secs_f64();
    let compose_started = Instant::now();
    let mut rng = lm::Rng::new(r.seed);
    let frames = lm::generate(&mut model, &depth, &ids, max_frames, if r.greedy { None } else { Some(&mut rng) }, |n| {
        report(event("composing", n, max_frames));
        Ok(())
    })?;
    let compose_seconds = compose_started.elapsed().as_secs_f64();
    drop(depth);
    drop(model);
    if frames.is_empty() {
        candle_core::bail!("the model ended the song before it began; try another seed or description");
    }
    let hidden = Tensor::cat(&frames.iter().map(|f| &f.hidden).collect::<Vec<_>>(), 0)?.unsqueeze(0)?;

    // Stage 2: render the frames to audio.
    let render_started = Instant::now();
    let blocks = 36;
    report(event("loading_renderer", 0, blocks));
    let mut acoustic = acoustic::Acoustic {
        condition: acoustic::ConditionEncoder::load(&r.model_dir.join("condition_encoder"), &dev)?,
        // Room for a window's decode (the vocoder runs in F32 at 44.1 kHz).
        transformer: acoustic::Transformer::load(&r.model_dir.join("transformer"), r.dtype, &r.budget.with_headroom(3 << 30), &dev, |i| report(event("loading_renderer", i, blocks)))?,
        vocoder: acoustic::Vocoder::load(&r.model_dir.join("vocoder"), &dev)?,
        steps: r.steps,
        guidance: r.guidance,
    };
    let dit_report = acoustic.transformer.report();
    let seed = r.seed;
    let audio = acoustic.generate(&hidden, |k, len| noise(seed, k, len), |done, total| report(event("rendering", done, total)))?;
    drop(acoustic);
    let render_seconds = render_started.elapsed().as_secs_f64();
    let audio = audio.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("music-{stamp}-{}.wav", r.seed));
    write_stereo_wav(&path, &audio[0], &audio[1], acoustic::SAMPLE_RATE)?;
    let seconds = audio[0].len() as f64 / acoustic::SAMPLE_RATE as f64;
    let result = Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(acoustic::SAMPLE_RATE as i64)),
        ("channels", Json::Int(2)),
        ("frames", Json::Int(frames.len() as i64)),
        ("duration", Json::Num(seconds)),
        ("finish_reason", Json::str(if frames.len() >= max_frames { "length" } else { "stop" })),
        ("prompt_tokens", Json::Int(ids[0].len() as i64)),
        ("load_seconds", Json::Num(load_seconds)),
        ("compose_seconds", Json::Num(compose_seconds)),
        ("render_seconds", Json::Num(render_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
        // Where each stage's weights lived (the studio shows the renderer's).
        ("residency", Json::obj([("language_model", lm_report), ("transformer", dit_report)])),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}

/// [`generate`] on WebGPU: the language model and depth decoder ([`crate::music_lm_wgpu`]), then the renderer
/// ([`crate::music_wgpu`]), one after the other on the same device.
#[cfg(feature = "webgpu")]
fn generate_webgpu(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    use crate::music_lm_wgpu::{WgpuDepth, WgpuMusicLm};
    let started = Instant::now();
    let gpu = ggml_rs_wgpu::WgpuBackend::nth(r.device, None).map_err(|e| candle_core::Error::Msg(format!("music on WebGPU: {e}")))?;
    let ids = prompt_ids(&r.model_dir, &r.prompt, &r.lyrics)?;
    let max_frames = ((r.max_seconds * FRAMES_PER_SECOND) as usize).clamp(1, MAX_FRAMES);

    // Stage 1: the language model writes the song, frame by frame.
    let layers = 36;
    report(event("loading_music_model", 0, layers));
    let depth = WgpuDepth::load(&r.model_dir.join("rvq_depth_decoder"), &gpu)?;
    let positions = ids[0].len() + max_frames + 2;
    let f16 = r.language_model.is_none() && crate::music_lm_wgpu::lm_f16(&gpu, positions);
    let mut model = WgpuMusicLm::load(&r.model_dir.join("language_model"), r.language_model.as_deref(), positions, &gpu, |i| report(event("loading_music_model", i, layers)))?;
    let load_seconds = started.elapsed().as_secs_f64();
    let compose_started = Instant::now();
    let mut rng = lm::Rng::new(r.seed);
    let frames = crate::music_lm_wgpu::generate(&mut model, &depth, &ids, max_frames, if r.greedy { None } else { Some(&mut rng) }, |n| {
        report(event("composing", n, max_frames));
        Ok(())
    })?;
    let compose_seconds = compose_started.elapsed().as_secs_f64();
    drop(depth);
    drop(model);
    if frames.is_empty() {
        candle_core::bail!("the model ended the song before it began; try another seed or description");
    }
    let hidden: Vec<f32> = frames.iter().flat_map(|f| f.hidden.iter().copied()).collect();

    // Stage 2: render the frames to audio.
    let render_started = Instant::now();
    let blocks = 36;
    report(event("loading_renderer", 0, blocks));
    let acoustic = crate::music_wgpu::WgpuAcoustic::load(&r.model_dir, &gpu, r.steps, r.guidance, |i| report(event("loading_renderer", i, blocks)))?;
    let seed = r.seed;
    let audio = acoustic.generate(&hidden, frames.len(), |k, len| noise(seed, k, len)?.flatten_all()?.to_vec1::<f32>(), |done, total| report(event("rendering", done, total)))?;
    drop(acoustic);
    let render_seconds = render_started.elapsed().as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("music-{stamp}-{}.wav", r.seed));
    write_stereo_wav(&path, &audio[0], &audio[1], acoustic::SAMPLE_RATE)?;
    let seconds = audio[0].len() as f64 / acoustic::SAMPLE_RATE as f64;
    let result = Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(acoustic::SAMPLE_RATE as i64)),
        ("channels", Json::Int(2)),
        ("frames", Json::Int(frames.len() as i64)),
        ("duration", Json::Num(seconds)),
        ("finish_reason", Json::str(if frames.len() >= max_frames { "length" } else { "stop" })),
        ("prompt_tokens", Json::Int(ids[0].len() as i64)),
        ("load_seconds", Json::Num(load_seconds)),
        ("compose_seconds", Json::Num(compose_seconds)),
        ("render_seconds", Json::Num(render_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
        ("backend", Json::str("webgpu")),
        ("language_model_weights", Json::str(if r.language_model.is_some() { "quantized file" } else if f16 { "f16" } else { "q8_0" })),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captions_are_cleaned_like_the_reference() {
        assert_eq!(clean_caption("<|bpm 96|> and <|key C major|>"), "bpm is 96 and key is C major");
        assert_eq!(clean_caption("## Genre\n- **acoustic** pop\n\n\n* soft *piano*  \n---\nend"), "Genre\nacoustic pop\nsoft piano\nend");
        assert_eq!(clean_caption("a **b *c* d** e"), "a **b c d** e");
        assert_eq!(clean_caption("  *a* ** b ** *c*d* e"), "  a  b  cd* e");
        assert_eq!(clean_caption("keep **this* odd"), "keep **this* odd");
        assert_eq!(clean_caption("x\r\ny\u{2028}z"), "x\ny\nz");
        assert_eq!(clean_caption("• one    two"), "onetwo");
    }

    #[test]
    fn lyrics_are_normalized_like_the_reference() {
        assert_eq!(normalize_lyrics("[Verse] dropped words\nline one\n[Chorus][Hook]\nsing ^ again [Bridge] x"), "[start]\n[verse]\nline one\n[chorus][hook]\nsing\nagain\n[bridge]\nx");
        assert_eq!(normalize_lyrics("[Intro]\n(instrumental)"), "[start]\n[intro]\n(instrumental)");
        assert_eq!(normalize_lyrics("  [Verse 1]  [Pre-Chorus]\nABC [X]y"), "[start]\n[verse 1]\n\n[pre-chorus]\nABC\n[x]y");
    }

    #[test]
    fn stereo_wav_interleaves_channels() {
        let path = std::env::temp_dir().join(format!("oaiy-music-{}.wav", std::process::id()));
        write_stereo_wav(&path, &[1., 0.], &[-1., 0.5], 44_100).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!(&b[22..24], &2u16.to_le_bytes());
        assert_eq!(&b[44..52], &[0xFF, 0x7F, 0x01, 0x80, 0, 0, 0x00, 0x40]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prompt_matches_the_reference_tokens() -> Result<()> {
        let Some((g, m)) = acoustic::golden::dirs() else { return Ok(()) };
        let p = Json::parse(&std::fs::read(g.join("prompt.json"))?).map_err(candle_core::Error::wrap)?;
        let (caption, lyrics) = (p.get("prompt").and_then(Json::as_str).unwrap(), p.get("lyrics").and_then(Json::as_str).unwrap());
        assert_eq!(prompt_text(caption, lyrics), p.get("text").and_then(Json::as_str).unwrap());
        let ids = prompt_ids(&m, caption, lyrics)?;
        let expected: Vec<u32> = acoustic::golden::ints(&g, "ids.i32").into_iter().map(|v| v as u32).collect();
        assert_eq!(ids[0], expected);
        assert_eq!(ids[1][0], expected[0]);
        assert!(ids[1][1..ids[1].len() - 2].iter().all(|&i| i == lm::AUDIO_CFG));
        assert_eq!(&ids[1][ids[1].len() - 2..], &expected[expected.len() - 2..]);
        Ok(())
    }
}
