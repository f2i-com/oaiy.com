//! Native Rust Qwen3-TTS (12 Hz): text to speech in a voice described in
//! words (VoiceDesign). The talker writes one frame of 16 codebook ids per
//! 80 ms; the speech codec turns frames into 24 kHz audio.
pub mod codec;
pub mod model;

use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor};
use model::{Cache, Decoder, Linear};
use nrob::json::Json;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const ASSISTANT: u32 = 77091;
const USER: u32 = 872;
const NEWLINE: u32 = 198;
const TTS_PAD: u32 = 151671;
const TTS_BOS: u32 = 151672;
const TTS_EOS: u32 = 151673;
const CODEC_PAD: u32 = 2148;
const CODEC_BOS: u32 = 2149;
const CODEC_EOS: u32 = 2150;
const CODEC_THINK: u32 = 2154;
const CODEC_NOTHINK: u32 = 2155;
const CODEC_THINK_BOS: u32 = 2156;
const CODEC_THINK_EOS: u32 = 2157;
/// Codec ids at or above this are control tokens, never sampled (EOS aside).
const AUDIO_CODES: u32 = 2048;
const FRAMES_PER_SECOND: f64 = 12.5;

#[derive(Clone, Debug)]
pub struct Request {
    /// The model folder (config.json, model.safetensors, vocab.json,
    /// merges.txt, speech_tokenizer/).
    pub model_dir: PathBuf,
    pub text: String,
    /// The voice, described in words (VoiceDesign).
    pub instructions: String,
    /// `auto` or a language the model knows (english, chinese, ...).
    pub language: String,
    pub output: PathBuf,
    pub seed: u64,
    pub device: usize,
    pub max_seconds: f64,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    /// Argmax instead of sampling (for tests; it tends to loop on long text).
    pub greedy: bool,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::to_owned);
        let f = |k: &str, d: f64| j.get(k).and_then(Json::as_f64).unwrap_or(d);
        let r = Self {
            model_dir: s("model_dir").filter(|p| !p.trim().is_empty()).ok_or("speech: missing model_dir")?.into(),
            text: s("text").filter(|t| !t.trim().is_empty()).ok_or("speech: text must not be empty")?,
            instructions: s("instructions").unwrap_or_default(),
            language: s("language").unwrap_or_else(|| "auto".into()).to_lowercase(),
            output: s("output_dir").ok_or("speech: missing output_dir")?.into(),
            seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0).max(0) as u64,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            max_seconds: f("max_seconds", 120.),
            temperature: f("temperature", 0.9),
            top_k: j.get("top_k").and_then(Json::as_i64).unwrap_or(50).max(1) as usize,
            top_p: f("top_p", 1.0),
            repetition_penalty: f("repetition_penalty", 1.05),
            greedy: j.get("greedy").and_then(Json::as_bool).unwrap_or(false),
        };
        if r.text.len() > 20_000 || r.instructions.len() > 4_000 {
            return Err("speech: text is limited to 20000 bytes and instructions to 4000".into());
        }
        if !(1.0..=600.0).contains(&r.max_seconds) {
            return Err("speech: max_seconds must be 1..600".into());
        }
        if !(0.05..=2.0).contains(&r.temperature) || !(0.0..=1.0).contains(&r.top_p) || r.top_p == 0.0 {
            return Err("speech: temperature must be 0.05..2 and top_p in (0, 1]".into());
        }
        Ok(r)
    }
}

/// Qwen2 byte-level BPE from `vocab.json` and `merges.txt`.
pub fn tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer> {
    use tokenizers::pre_tokenizers::{byte_level::ByteLevel, sequence::Sequence, split::Split, split::SplitPattern};
    let (vocab, merges) = (dir.join("vocab.json"), dir.join("merges.txt"));
    let bpe = tokenizers::models::bpe::BPE::from_file(&vocab.to_string_lossy(), &merges.to_string_lossy())
        .build()
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;
    let mut tok = tokenizers::Tokenizer::new(bpe);
    tok.with_normalizer(Some(tokenizers::normalizers::unicode::NFC));
    const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(SplitPattern::Regex(PATTERN.into()), tokenizers::SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;
    tok.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), ByteLevel::new(false, false, false).into()])));
    Ok(tok)
}

fn encode(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(text, false).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?.get_ids().to_vec())
}

/// A seeded generator (splitmix64) for sampling.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x2545_f491_4f6c_dd1d)
    }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Temperature, top-k, top-p, then a draw (or the argmax).
fn sample(logits: &[f32], temperature: f64, top_k: usize, top_p: f64, greedy: bool, rng: &mut Rng) -> u32 {
    let mut order: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if greedy || order.len() == 1 {
        return order[0] as u32;
    }
    order.truncate(top_k.max(1));
    let top = logits[order[0]] as f64;
    let weights: Vec<f64> = order.iter().map(|&i| ((logits[i] as f64 - top) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut kept = order.len();
    if top_p < 1.0 {
        let mut acc = 0.;
        for (n, w) in weights.iter().enumerate() {
            acc += w / total;
            if acc >= top_p {
                kept = n + 1;
                break;
            }
        }
    }
    let total: f64 = weights[..kept].iter().sum();
    let mut target = rng.next() * total;
    for (n, w) in weights[..kept].iter().enumerate() {
        target -= w;
        if target <= 0. {
            return order[n] as u32;
        }
    }
    order[kept - 1] as u32
}

pub struct Tts {
    talker: Decoder,
    predictor: Decoder,
    codec_embedding: Tensor,
    predictor_embeddings: Vec<Tensor>,
    predictor_heads: Vec<Linear>,
    codec_head: Linear,
    to_predictor: Linear,
    text_fc1: Linear,
    text_fc2: Linear,
    store: Store,
    config: Json,
    dev: Device,
}

impl Tts {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let config = Json::parse(&std::fs::read(dir.join("config.json"))?).map_err(candle_core::Error::wrap)?;
        let tc = config.get("talker_config").ok_or_else(|| candle_core::Error::Msg("config lacks talker_config".into()))?;
        let cp = tc.get("code_predictor_config").ok_or_else(|| candle_core::Error::Msg("config lacks code_predictor_config".into()))?;
        let int = |c: &Json, k: &str| c.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| candle_core::Error::Msg(format!("config lacks {k}")));
        let theta = |c: &Json| c.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6);
        let mut store = Store::open(&dir.join("model.safetensors"), 0)?;
        let talker = Decoder::load(&mut store, "talker.model", int(tc, "num_hidden_layers")?, int(tc, "num_attention_heads")?, int(tc, "num_key_value_heads")?, int(tc, "head_dim")?, theta(tc), 16384, 1e-6, dev)?;
        let predictor = Decoder::load(&mut store, "talker.code_predictor.model", int(cp, "num_hidden_layers")?, int(cp, "num_attention_heads")?, int(cp, "num_key_value_heads")?, int(cp, "head_dim")?, theta(cp), 32, 1e-6, dev)?;
        let groups = int(tc, "num_code_groups")?;
        let mut predictor_embeddings = Vec::new();
        let mut predictor_heads = Vec::new();
        for i in 0..groups - 1 {
            predictor_embeddings.push(store.tensor(&format!("talker.code_predictor.model.codec_embedding.{i}.weight"), dev, false)?);
            predictor_heads.push(Linear::load(&mut store, &format!("talker.code_predictor.lm_head.{i}"), dev)?);
        }
        Ok(Self {
            talker,
            predictor,
            codec_embedding: store.tensor("talker.model.codec_embedding.weight", dev, false)?,
            predictor_embeddings,
            predictor_heads,
            codec_head: Linear::load(&mut store, "talker.codec_head", dev)?,
            to_predictor: Linear::load(&mut store, "talker.code_predictor.small_to_mtp_projection", dev)?,
            text_fc1: Linear::load(&mut store, "talker.text_projection.linear_fc1", dev)?,
            text_fc2: Linear::load(&mut store, "talker.text_projection.linear_fc2", dev)?,
            store,
            config,
            dev: dev.clone(),
        })
    }

    /// T(ids): text embedding rows (read from the file, not uploaded whole),
    /// through the text projection. (1, n, 2048).
    fn text(&mut self, ids: &[u32]) -> Result<Tensor> {
        let rows = self.store.rows("talker.model.text_embedding.weight", ids, &self.dev)?.unsqueeze(0)?;
        self.text_fc2.forward(&self.text_fc1.forward(&rows)?.silu()?)
    }

    /// C(ids): codec embedding rows. (1, n, 2048).
    fn codec(&self, ids: &[u32]) -> Result<Tensor> {
        self.codec_embedding.index_select(&Tensor::new(ids, &self.dev)?, 0)?.unsqueeze(0)
    }

    fn language_id(&self, language: &str) -> Result<Option<u32>> {
        if language == "auto" {
            return Ok(None);
        }
        let ids = self.config.get("talker_config").and_then(|t| t.get("codec_language_id"));
        ids.and_then(|m| m.get(language)).and_then(Json::as_i64).map(|v| Some(v as u32)).ok_or_else(|| {
            candle_core::Error::Msg(format!("speech: unknown language {language}; use auto or one of the model's languages"))
        })
    }

    /// The talker's prefill (non-streaming VoiceDesign layout): the voice
    /// description, the assistant role, the codec prefix, every text token
    /// over a codec pad, text end, then codec BOS.
    pub fn prefill(&mut self, text_ids: &[u32], instruct_ids: Option<&[u32]>, language: Option<u32>) -> Result<Tensor> {
        let mut parts = Vec::new();
        if let Some(ids) = instruct_ids {
            let mut full = vec![IM_START, USER, NEWLINE];
            full.extend_from_slice(ids);
            full.extend_from_slice(&[IM_END, NEWLINE]);
            parts.push(self.text(&full)?);
        }
        parts.push(self.text(&[IM_START, ASSISTANT, NEWLINE])?);
        let prefix: Vec<u32> = match language {
            Some(l) => vec![CODEC_THINK, CODEC_THINK_BOS, l, CODEC_THINK_EOS, CODEC_PAD],
            None => vec![CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS, CODEC_PAD],
        };
        let specials = self.text(&[TTS_PAD, TTS_BOS, TTS_EOS])?;
        let (pad, bos, eos) = (specials.narrow(1, 0, 1)?, specials.narrow(1, 1, 1)?, specials.narrow(1, 2, 1)?);
        let n = prefix.len();
        let text_side = Tensor::cat(&[pad.broadcast_as((1, n - 1, pad.dim(2)?))?.contiguous()?, bos.clone()], 1)?;
        parts.push((text_side + self.codec(&prefix)?)?);
        let words = Tensor::cat(&[self.text(text_ids)?, eos.clone()], 1)?;
        parts.push((&words + self.codec(&vec![CODEC_PAD; text_ids.len() + 1])?)?);
        parts.push((&pad + self.codec(&[CODEC_BOS])?)?);
        Tensor::cat(&parts, 1)
    }

    /// The text term added to every generated step (non-streaming: tts_pad).
    fn pad_embedding(&mut self) -> Result<Tensor> {
        self.text(&[TTS_PAD])
    }

    /// Codebooks 1..15 for a frame whose codebook 0 is `c0`, from the talker's
    /// hidden state `hidden` (1, 1, 2048).
    fn predict_rest(&self, hidden: &Tensor, c0: u32, r: &Request, rng: &mut Rng) -> Result<[u32; 16]> {
        let mut frame = [0u32; 16];
        frame[0] = c0;
        let mut cache = Cache::new(self.predictor.layers());
        let x = Tensor::cat(&[hidden, &self.codec(&[c0])?], 1)?;
        let mut h = self.predictor.forward(&self.to_predictor.forward(&x)?, &mut cache)?;
        for (i, head) in self.predictor_heads.iter().enumerate() {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let logits = head.forward(&last)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let code = sample(&logits, r.temperature, r.top_k, 1.0, r.greedy, rng);
            frame[i + 1] = code;
            if i + 1 < self.predictor_heads.len() {
                let e = self.predictor_embeddings[i].index_select(&Tensor::new(&[code], &self.dev)?, 0)?.unsqueeze(0)?;
                h = self.predictor.forward(&self.to_predictor.forward(&e)?, &mut cache)?;
            }
        }
        Ok(frame)
    }

    /// The next talker input: the frame's 16 codec embeddings summed, plus
    /// the text term.
    fn frame_embedding(&self, frame: &[u32; 16], text: &Tensor) -> Result<Tensor> {
        let mut e = self.codec(&frame[..1])?;
        for (i, emb) in self.predictor_embeddings.iter().enumerate() {
            e = (e + emb.index_select(&Tensor::new(&frame[i + 1..i + 2], &self.dev)?, 0)?.unsqueeze(0)?)?;
        }
        e + text
    }

    /// Generate frames until the codec EOS (or `max_frames`).
    pub fn frames(&mut self, prefill: &Tensor, r: &Request, max_frames: usize, mut progress: impl FnMut(usize)) -> Result<Vec<[u32; 16]>> {
        let mut rng = Rng::new(r.seed);
        let pad = self.pad_embedding()?;
        let mut cache = Cache::new(self.talker.layers());
        let h = self.talker.forward(prefill, &mut cache)?;
        let mut hidden = h.narrow(1, h.dim(1)? - 1, 1)?;
        let mut frames: Vec<[u32; 16]> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        while frames.len() < max_frames {
            let mut logits = self.codec_head.forward(&hidden)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            // As the reference's processors: repetition penalty over earlier
            // first codes, no EOS for the first 2 frames, control ids
            // suppressed (EOS aside); then temperature, top-k, top-p.
            for &id in &seen {
                let l: &mut f32 = &mut logits[id as usize];
                *l = if *l < 0. { *l * r.repetition_penalty as f32 } else { *l / r.repetition_penalty as f32 };
            }
            if frames.len() < 2 {
                logits[CODEC_EOS as usize] = f32::NEG_INFINITY;
            }
            for (id, l) in logits.iter_mut().enumerate().skip(AUDIO_CODES as usize) {
                if id as u32 != CODEC_EOS {
                    *l = f32::NEG_INFINITY;
                }
            }
            let c0 = sample(&logits, r.temperature, r.top_k, r.top_p, r.greedy, &mut rng);
            if c0 == CODEC_EOS {
                break;
            }
            seen.insert(c0);
            let frame = self.predict_rest(&hidden, c0, r, &mut rng)?;
            frames.push(frame);
            progress(frames.len());
            hidden = self.talker.forward(&self.frame_embedding(&frame, &pad)?, &mut cache)?;
        }
        Ok(frames)
    }
}

fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([("stage", Json::str(stage)), ("current", Json::Int(current as i64)), ("total", Json::Int(total as i64))])
}

/// 16-bit PCM mono WAV.
pub fn write_wav(path: &Path, samples: &[f32], rate: usize) -> std::io::Result<()> {
    let data: Vec<u8> = samples.iter().flat_map(|s| ((s.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes()).collect();
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(rate as u32).to_le_bytes());
    out.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    std::fs::write(path, out)
}

fn device(index: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(index)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = index;
        Ok(Device::Cpu)
    }
}

pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    let dev = device(r.device)?;
    report(event("loading_speech_model", 0, 1));
    let tok = tokenizer(&r.model_dir)?;
    let mut tts = Tts::load(&r.model_dir, &dev)?;
    let language = tts.language_id(&r.language)?;
    let text_ids = encode(&tok, &r.text)?;
    let instruct_ids = if r.instructions.trim().is_empty() { None } else { Some(encode(&tok, &r.instructions)?) };
    let prefill = tts.prefill(&text_ids, instruct_ids.as_deref(), language)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let max_frames = (r.max_seconds * FRAMES_PER_SECOND).ceil() as usize;
    let speak_started = Instant::now();
    let frames = tts.frames(&prefill, r, max_frames, |n| report(event("speaking", n, max_frames)))?;
    let speak_seconds = speak_started.elapsed().as_secs_f64();
    drop(tts);
    report(event("decoding_speech", 0, 1));
    let decode_started = Instant::now();
    let codec = codec::CodecDecoder::load(&r.model_dir.join("speech_tokenizer").join("model.safetensors"), &dev)?;
    let samples = if frames.is_empty() { Vec::new() } else { codec.decode(&frames)? };
    drop(codec);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("speech-{stamp}-{}.wav", r.seed));
    write_wav(&path, &samples, codec::SAMPLE_RATE)?;
    let audio_seconds = samples.len() as f64 / codec::SAMPLE_RATE as f64;
    let result = Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(codec::SAMPLE_RATE as i64)),
        ("frames", Json::Int(frames.len() as i64)),
        ("duration", Json::Num(audio_seconds)),
        ("finish_reason", Json::str(if frames.len() >= max_frames { "length" } else { "stop" })),
        ("load_seconds", Json::Num(load_seconds)),
        ("speak_seconds", Json::Num(speak_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual.to_dtype(DType::F32)?;
        let error = (&actual - expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        Ok(error / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
    }

    #[test]
    fn sampling_respects_top_k_and_greedy() {
        let logits = [0.0f32, 5.0, 4.9, f32::NEG_INFINITY, -3.0];
        let mut rng = Rng::new(7);
        assert_eq!(sample(&logits, 0.9, 50, 1.0, true, &mut rng), 1);
        for _ in 0..200 {
            let s = sample(&logits, 1.0, 2, 1.0, false, &mut rng);
            assert!(s == 1 || s == 2, "top-2 only, got {s}");
        }
    }

    #[test]
    #[ignore = "requires the Qwen3-TTS VoiceDesign model and reference dumps; NROB_TTS_GOLDEN, NROB_TTS_MODEL"]
    fn talker_matches_reference() -> Result<()> {
        let root = PathBuf::from(std::env::var("NROB_TTS_GOLDEN").map_err(candle_core::Error::wrap)?);
        let model = PathBuf::from(std::env::var("NROB_TTS_MODEL").map_err(candle_core::Error::wrap)?);
        let dev = Device::new_cuda(std::env::var("NROB_TTS_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let ints = |name: &str| -> Result<Vec<u32>> {
            Ok(std::fs::read(root.join(name))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect())
        };
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let meta = Json::parse(&std::fs::read(root.join("meta.json"))?).map_err(candle_core::Error::wrap)?;
        let text = meta.get("text").and_then(Json::as_str).unwrap_or_default();
        let instruct = meta.get("instruct").and_then(Json::as_str).unwrap_or_default();
        let tok = tokenizer(&model)?;
        let (input_ids, instruct_ids) = (ints("input_ids.i32")?, ints("instruct_ids.i32")?);
        let text_ids = encode(&tok, text)?;
        let inst_ids = encode(&tok, instruct)?;
        assert_eq!(text_ids, input_ids[3..input_ids.len() - 5], "text token ids");
        assert_eq!(inst_ids, instruct_ids[3..instruct_ids.len() - 2], "instruct token ids");
        let mut tts = Tts::load(&model, &dev)?;
        let lang = tts.language_id("english")?;
        let prefill = tts.prefill(&text_ids, Some(&inst_ids), lang)?;
        let n = prefill.dim(1)?;
        let e = relative(&prefill.squeeze(0)?, &read("talker_prefill_in.f32", &[n, 2048])?)?;
        println!("prefill embeddings ({n} positions): relative RMS error {e}");
        assert!(e < 0.01, "prefill {e}");
        let mut cache = Cache::new(tts.talker.layers());
        let out = tts.talker.forward(&prefill, &mut cache)?;
        let e = relative(&out.squeeze(0)?, &read("talker_prefill_out.f32", &[n, 2048])?)?;
        println!("talker prefill output: relative RMS error {e}");
        assert!(e < 0.03, "talker prefill {e}");
        // Greedy frames: the reference's first frames, exactly.
        let greedy = ints("greedy_codes.i32")?;
        let r = Request {
            model_dir: model.clone(),
            text: text.into(),
            instructions: instruct.into(),
            language: "english".into(),
            output: root.clone(),
            seed: 0,
            device: 0,
            max_seconds: 10.,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            greedy: true,
        };
        let frames = tts.frames(&prefill, &r, 4, |_| {})?;
        for (i, f) in frames.iter().enumerate() {
            let want = &greedy[i * 16..(i + 1) * 16];
            println!("frame {i}: ours {:?}\n         ref  {:?}", f, want);
        }
        assert_eq!(frames[0][0], greedy[0], "first codebook-0 id");
        let matching = frames.iter().enumerate().filter(|(i, f)| f[..] == greedy[i * 16..(i + 1) * 16]).count();
        println!("{matching} of {} greedy frames identical", frames.len());
        assert!(matching >= 2, "greedy frames diverge immediately");
        Ok(())
    }
}
