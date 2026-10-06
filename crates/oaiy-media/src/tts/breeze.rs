//! Native Rust Breeze TTS 2: text to speech in a plain voice, a voice described in words
//! (an instruction), or a saved voice (its reference clip's codes and transcript).
//!
//! A T5Gemma2 encoder reads each text segment on its own (bidirectional, 26 layers); a
//! linear projection puts it in the backbone's space, beside the reference clip's frames
//! (each frame the sum of its 16 codebooks' embeddings). The backbone (Qwen3, 28 layers)
//! writes one frame's first code at a time (a code of 2051 ends the speech); a small depth
//! decoder (Llama, 12 layers) fills in the other 15 from the backbone's state. With a
//! guidance scale other than 1, both also run on the prompt without its instruction (or
//! without the voice's clip) and are pushed away from it. Qwen3-TTS's 12 Hz codec, which
//! Breeze ships as `audio_tokenizer/`, turns the frames into 24 kHz audio.
use super::model::{Cache, Decoder, Linear};
use super::{codec, device, event, sample, trim_leading_silence, write_wav, Request, Rng, Voice};
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor, D};
use oaiy_engine::json::Json;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const NUM_CODEBOOKS: usize = 16;
/// Audio codes are 0..2048; 2048..2051 are reserved (2050 pads); the backbone's class 2051
/// ends the speech.
const CODEBOOK_SIZE: usize = 2048;
const AUDIO_VOCAB: usize = 2051;
const EOS: usize = 2051;
const AUDIO_EOS_CODE: u32 = 0;
const FRAMES_PER_SECOND: f64 = 12.5;
/// The reference's limit: 1500 frames (two minutes).
const MAX_FRAMES: usize = 1500;

/// Whether `dir` is a Breeze TTS 2 checkpoint.
pub fn is_breeze(dir: &Path) -> bool {
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| Json::parse(&b).ok())
        .is_some_and(|c| c.get("model_type").and_then(Json::as_str) == Some("breeze"))
}

/// Gemma-style RMS norm: `x / rms(x) * (1 + w)`, in F32.
fn gemma_rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let n = f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)?;
    n.broadcast_mul(&(w.to_dtype(DType::F32)? + 1.)?)?.to_dtype(x.dtype())
}

struct TeLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    pre_attn: Tensor,
    post_attn: Tensor,
    pre_ff: Tensor,
    post_ff: Tensor,
    gate_up: Linear,
    down: Linear,
    sliding: bool,
}

/// T5Gemma2's text encoder: Gemma 3 layers, attention both ways (sliding layers see 512
/// tokens around each one), RoPE at theta 10000 (sliding) or 1e6 over 8 (full).
struct TextEncoder {
    layers: Vec<TeLayer>,
    eoi: Tensor,
    eoi_id: u32,
    norm: Tensor,
    scale: f64,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
    eps: f64,
    local: Vec<f64>,
    global: Vec<f64>,
}

impl TextEncoder {
    fn load(store: &mut Store, cfg: &Json, dev: &Device) -> Result<Self> {
        let n = |k: &str| cfg.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| candle_core::Error::Msg(format!("Breeze text encoder config: no {k}")));
        let (hidden, heads, kv_heads, head_dim) = (n("hidden_size")?, n("num_attention_heads")?, n("num_key_value_heads")?, n("head_dim")?);
        let types: Vec<bool> = cfg.get("layer_types").and_then(Json::as_array).ok_or_else(|| candle_core::Error::Msg("Breeze text encoder: no layer_types".into()))?
            .iter().map(|t| t.as_str() == Some("sliding_attention")).collect();
        let rope = |kind: &str, theta: f64| -> Vec<f64> {
            let p = cfg.get("rope_parameters").and_then(|r| r.get(kind));
            let base = p.and_then(|p| p.get("rope_theta")).and_then(Json::as_f64).unwrap_or(theta);
            let factor = match p.and_then(|p| p.get("rope_type")).and_then(Json::as_str) {
                Some("linear") => p.and_then(|p| p.get("factor")).and_then(Json::as_f64).unwrap_or(1.),
                _ => 1.,
            };
            (0..head_dim / 2).map(|i| (1. / base.powf(2. * i as f64 / head_dim as f64)) as f32 as f64 / factor).collect()
        };
        let mut layers = Vec::with_capacity(types.len());
        for (i, &sliding) in types.iter().enumerate() {
            let p = format!("text_encoder.layers.{i}");
            let lin = |store: &mut Store, name: &str| Linear::load(store, &format!("{p}.{name}"), dev);
            let t = |store: &mut Store, name: &str| store.tensor(&format!("{p}.{name}"), dev, false);
            layers.push(TeLayer {
                q: lin(store, "self_attn.q_proj")?,
                k: lin(store, "self_attn.k_proj")?,
                v: lin(store, "self_attn.v_proj")?,
                o: lin(store, "self_attn.o_proj")?,
                q_norm: t(store, "self_attn.q_norm.weight")?,
                k_norm: t(store, "self_attn.k_norm.weight")?,
                pre_attn: t(store, "pre_self_attn_layernorm.weight")?,
                post_attn: t(store, "post_self_attn_layernorm.weight")?,
                pre_ff: t(store, "pre_feedforward_layernorm.weight")?,
                post_ff: t(store, "post_feedforward_layernorm.weight")?,
                gate_up: Linear::fused(store, &[format!("{p}.mlp.gate_proj"), format!("{p}.mlp.up_proj")], dev)?,
                down: lin(store, "mlp.down_proj")?,
                sliding,
            });
        }
        Ok(Self {
            layers,
            eoi: store.tensor("text_encoder.embed_tokens.eoi_embedding", dev, false)?,
            eoi_id: cfg.get("eoi_token_index").and_then(Json::as_i64).unwrap_or(256_000) as u32,
            norm: store.tensor("text_encoder.norm.weight", dev, false)?,
            // The embedding scale, as the reference applies it: rounded to BF16.
            scale: half::bf16::from_f64((hidden as f64).sqrt()).to_f64(),
            heads,
            kv_heads,
            head_dim,
            window: cfg.get("sliding_window").and_then(Json::as_i64).unwrap_or(512) as usize,
            eps: cfg.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6),
            local: rope("sliding_attention", 10_000.),
            global: rope("full_attention", 1_000_000.),
        })
    }

    /// One text segment's final states, (T, hidden) BF16.
    fn forward(&self, store: &mut Store, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let t = ids.len();
        let (h, kv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let mut x = (store.rows("text_encoder.embed_tokens.weight", ids, dev)?.to_dtype(DType::BF16)? * self.scale)?;
        if ids.contains(&self.eoi_id) {
            let rows: Vec<Tensor> = ids.iter().enumerate().map(|(i, &id)| if id == self.eoi_id { self.eoi.unsqueeze(0) } else { x.narrow(0, i, 1) }).collect::<Result<_>>()?;
            x = Tensor::cat(&rows, 0)?;
        }
        let table = |inv: &[f64]| -> Result<(Tensor, Tensor)> {
            let f: Vec<f32> = (0..t).flat_map(|p| inv.iter().map(move |f| p as f32 * *f as f32)).collect();
            let f = Tensor::from_vec(f, (t, hd / 2), dev)?;
            Ok((f.cos()?.to_dtype(DType::BF16)?, f.sin()?.to_dtype(DType::BF16)?))
        };
        let (local, global) = (table(&self.local)?, table(&self.global)?);
        // Sliding layers: a query sees keys fewer than 256 before it and 257 after.
        let (left, right) = (self.window.div_ceil(2), self.window / 2 + 1);
        let mask: Vec<f32> = (0..t).flat_map(|q| (0..t).map(move |k| {
            let d = q as i64 - k as i64;
            if (d >= 0 && (d as usize) < left) || (d < 0 && ((-d) as usize) < right) { 0. } else { f32::NEG_INFINITY }
        })).collect();
        let mask = Tensor::from_vec(mask, (t, t), dev)?;
        let mut x = x.unsqueeze(0)?;
        for l in &self.layers {
            let n = gemma_rms(&x, &l.pre_attn, self.eps)?;
            let q = gemma_rms(&l.q.forward(&n)?.reshape((1, t, h, hd))?, &l.q_norm, self.eps)?;
            let k = gemma_rms(&l.k.forward(&n)?.reshape((1, t, kv, hd))?, &l.k_norm, self.eps)?;
            let v = l.v.forward(&n)?.reshape((1, t, kv, hd))?;
            let (cos, sin) = if l.sliding { &local } else { &global };
            let q = candle_nn::rotary_emb::rope_thd(&q.contiguous()?, cos, sin)?;
            let k = candle_nn::rotary_emb::rope_thd(&k.contiguous()?, cos, sin)?;
            // (1, heads, T, D) in F32; the one key/value head serves every query head.
            let qh = q.transpose(1, 2)?.to_dtype(DType::F32)?;
            let kh = k.transpose(1, 2)?.to_dtype(DType::F32)?.broadcast_as((1, h, t, hd))?.contiguous()?;
            let vh = v.transpose(1, 2)?.to_dtype(DType::F32)?.broadcast_as((1, h, t, hd))?.contiguous()?;
            let mut scores = (qh.contiguous()?.matmul(&kh.t()?)? / (hd as f64).sqrt())?;
            if l.sliding && t > left.min(right) {
                scores = scores.broadcast_add(&mask)?;
            }
            let a = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vh)?.transpose(1, 2)?.to_dtype(DType::BF16)?.reshape((1, t, h * hd))?;
            x = (x + gemma_rms(&l.o.forward(&a)?, &l.post_attn, self.eps)?)?;
            let n = gemma_rms(&x, &l.pre_ff, self.eps)?;
            let gu = l.gate_up.forward(&n)?;
            let inner = gu.dim(2)? / 2;
            let m = (gu.narrow(2, 0, inner)?.gelu()? * gu.narrow(2, inner, inner)?)?;
            x = (x + gemma_rms(&l.down.forward(&m)?, &l.post_ff, self.eps)?)?;
        }
        gemma_rms(&x, &self.norm, self.eps)?.squeeze(0)
    }
}

/// Llama 3's RoPE frequencies: the long wavelengths slowed by `factor`, the middle ones
/// blended.
fn llama3_inv_freq(head_dim: usize, theta: f64, factor: f64, low: f64, high: f64, original: f64) -> Vec<f64> {
    let (low_wave, high_wave) = (original / low, original / high);
    (0..head_dim / 2)
        .map(|i| {
            let f = (1. / theta.powf(2. * i as f64 / head_dim as f64)) as f32 as f64;
            let wave = 2. * std::f64::consts::PI / f;
            let scaled = if wave > low_wave { f / factor } else { f };
            if wave >= high_wave && wave <= low_wave {
                let smooth = (original / wave - low) / (high - low);
                (1. - smooth) * scaled / factor + smooth * scaled
            } else {
                scaled
            }
        })
        .collect()
}

/// Everything but the text encoder and the codec.
struct Breeze {
    backbone: Decoder,
    depth: Decoder,
    /// (16 * 2051, 2048): each codebook's code embeddings, one after another.
    audio_embed: Tensor,
    lm_head: Tensor,
    depth_in: Linear,
    /// (15, 1024, 2051) F32.
    depth_heads: Tensor,
    text_proj: Linear,
}

/// One guidance branch's state: its caches and the backbone's last state.
struct Branch {
    backbone: Cache,
    hidden: Tensor,
}

impl Breeze {
    fn load(store: &mut Store, cfg: &Json, dev: &Device) -> Result<Self> {
        let sub = |k: &str| cfg.get(k).cloned().unwrap_or(Json::Null);
        let n = |c: &Json, k: &str| c.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| candle_core::Error::Msg(format!("Breeze config: no {k}")));
        let b = sub("backbone_config");
        let backbone = Decoder::load(
            store,
            "backbone_model",
            n(&b, "num_hidden_layers")?,
            n(&b, "num_attention_heads")?,
            n(&b, "num_key_value_heads")?,
            n(&b, "head_dim")?,
            b.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6),
            8192,
            b.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            dev,
        )?;
        let d = sub("depth_decoder_config");
        let hd = n(&d, "head_dim")?;
        let rs = d.get("rope_scaling").cloned().unwrap_or(Json::Null);
        let f = |k: &str, v: f64| rs.get(k).and_then(Json::as_f64).unwrap_or(v);
        let inv = llama3_inv_freq(hd, d.get("rope_theta").and_then(Json::as_f64).unwrap_or(500_000.), f("factor", 32.), f("low_freq_factor", 1.), f("high_freq_factor", 4.), f("original_max_position_embeddings", 16.));
        let depth = Decoder::load_with(
            store,
            "depth_decoder.model",
            n(&d, "num_hidden_layers")?,
            n(&d, "num_attention_heads")?,
            n(&d, "num_key_value_heads")?,
            hd,
            &inv,
            NUM_CODEBOOKS + 2,
            d.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5) as f32,
            false,
            dev,
        )?;
        Ok(Self {
            backbone,
            depth,
            audio_embed: store.tensor("depth_decoder.model.embed_tokens.weight", dev, false)?,
            lm_head: store.tensor_f32("lm_head.weight", dev)?,
            depth_in: Linear::load(store, "depth_decoder.model.inputs_embeds_projector", dev)?,
            depth_heads: store.tensor_f32("depth_decoder.codebooks_head.weight", dev)?,
            text_proj: Linear::load(store, "text_encoder_proj", dev)?,
        })
    }

    /// Frames as the backbone reads them: each the sum of its codebooks' embeddings, (F, 2048).
    fn frames_embedding(&self, frames: &[[u32; 16]], dev: &Device) -> Result<Tensor> {
        let ids: Vec<u32> = frames.iter().flat_map(|f| f.iter().enumerate().map(|(c, &code)| code + (c * AUDIO_VOCAB) as u32)).collect();
        let ids = Tensor::from_vec(ids, frames.len() * NUM_CODEBOOKS, dev)?;
        self.audio_embed.index_select(&ids, 0)?.reshape((frames.len(), NUM_CODEBOOKS, ()))?.to_dtype(DType::F32)?.sum(1)?.to_dtype(DType::BF16)
    }

    /// The backbone's logits for its last state, (2052,) F32.
    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.to_dtype(DType::F32)?.reshape((1, ()))?.matmul(&self.lm_head.t()?)?.squeeze(0)
    }

    /// Run the backbone over `x` (1, T, 2048) in `branch`, keeping its last state.
    fn step(&self, branch: &mut Branch, x: &Tensor) -> Result<()> {
        let h = self.backbone.forward(x, &mut branch.backbone)?;
        branch.hidden = h.narrow(1, h.dim(1)? - 1, 1)?;
        Ok(())
    }
}

/// Guided logits: `uncond + s (cond - uncond)`, or `cond` alone.
fn guide(cond: Tensor, uncond: Option<Tensor>, scale: f64) -> Result<Tensor> {
    match uncond {
        Some(u) => &u + ((cond - &u)? * scale)?,
        None => Ok(cond),
    }
}

/// A prompt's segments: text (encoded), or a clip's frames and then its end.
enum Segment {
    Text(String),
    Clip(Vec<[u32; 16]>),
}

/// The prompt's input embeddings, (1, T, 2048).
fn prompt(model: &Breeze, te: &TextEncoder, store: &mut Store, tok: &tokenizers::Tokenizer, segments: &[Segment], dev: &Device) -> Result<Tensor> {
    let mut parts = Vec::new();
    for s in segments {
        match s {
            Segment::Text(text) => {
                // Each segment starts with <bos>, as the reference tokenizes it.
                let ids = tok.encode(text.as_str(), true).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?.get_ids().to_vec();
                let states = te.forward(store, &ids, dev)?;
                parts.push(model.text_proj.forward(&states)?);
            }
            Segment::Clip(frames) => {
                let mut all = frames.clone();
                all.push([AUDIO_EOS_CODE; NUM_CODEBOOKS]);
                parts.push(model.frames_embedding(&all, dev)?);
            }
        }
    }
    Tensor::cat(&parts, 0)?.unsqueeze(0)
}

/// Speak `r.text`: in `r.voice` when given (its clip and transcript lead the prompt),
/// directed by `r.instructions` when given.
pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    let (samples, frames, load_seconds, speak_seconds) = speak(r, &mut report)?;
    let decode_started = Instant::now();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("speech-{stamp}-{}.wav", r.seed));
    write_wav(&path, &samples, codec::SAMPLE_RATE)?;
    let max_frames = ((r.max_seconds * FRAMES_PER_SECOND).ceil() as usize).min(MAX_FRAMES);
    Ok(Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(codec::SAMPLE_RATE as i64)),
        ("frames", Json::Int(frames as i64)),
        ("duration", Json::Num(samples.len() as f64 / codec::SAMPLE_RATE as f64)),
        ("finish_reason", Json::str(if frames >= max_frames { "length" } else { "stop" })),
        ("load_seconds", Json::Num(load_seconds)),
        ("speak_seconds", Json::Num(speak_seconds)),
        ("decode_seconds", Json::Num(decode_started.elapsed().as_secs_f64())),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
        ("engine", Json::str("breeze-tts-2")),
    ]))
}

/// The audio (24 kHz), how many frames, and how long loading and speaking took.
fn speak(r: &Request, report: &mut impl FnMut(Json)) -> Result<(Vec<f32>, usize, f64, f64)> {
    let dev = device(r.device)?;
    report(event("loading_speech_model", 0, 1));
    let load_started = Instant::now();
    let cfg = Json::parse(&std::fs::read(r.model_dir.join("config.json"))?).map_err(candle_core::Error::wrap)?;
    let tok = tokenizers::Tokenizer::from_file(r.model_dir.join("tokenizer.json")).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;
    let mut store = Store::open(&r.model_dir, 0)?;
    let te = TextEncoder::load(&mut store, cfg.get("text_encoder_config").unwrap_or(&Json::Null), &dev)?;
    let model = Breeze::load(&mut store, &cfg, &dev)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    // The prompt: the voice's clip and transcript first, then the line (its instruction
    // before it). Without guidance, one branch; with it, a second without the instruction
    // (or, lacking one, without the voice).
    let speaker = "[S0]";
    let instruction = r.instructions.trim();
    let line = |with_instruction: bool| {
        if with_instruction && !instruction.is_empty() {
            format!("{speaker}<ins_bos>{instruction}<ins_eos>{}", r.text)
        } else {
            format!("{speaker}{}", r.text)
        }
    };
    let with_voice = |line: String| -> Vec<Segment> {
        match &r.voice {
            Some(v) => vec![Segment::Text(format!("{speaker}{}", v.ref_text)), Segment::Clip(v.ref_codes.clone()), Segment::Text(line)],
            None => vec![Segment::Text(line)],
        }
    };
    let scale = r.cfg_scale.unwrap_or(1.0);
    let main = with_voice(line(true));
    // Guidance steers toward the instruction: without one there is nothing to steer.
    let negative = if scale != 1.0 && !instruction.is_empty() { Some(with_voice(line(false))) } else { None };
    let speak_started = Instant::now();
    let mut branches = Vec::new();
    for segments in std::iter::once(&main).chain(negative.as_ref()) {
        let x = prompt(&model, &te, &mut store, &tok, segments, &dev)?;
        let mut b = Branch { backbone: Cache::new(model.backbone.layers()), hidden: Tensor::zeros((1, 1, 1), DType::BF16, &dev)? };
        model.step(&mut b, &x)?;
        branches.push(b);
    }
    drop(te);
    let guided = branches.len() > 1;
    let max_frames = ((r.max_seconds * FRAMES_PER_SECOND).ceil() as usize).min(MAX_FRAMES);
    let penalty = if r.repetition_penalty_set { r.repetition_penalty } else { 1.1 };
    let mut rng = Rng::new(r.seed);
    let mut history: Vec<usize> = Vec::new();
    let mut frames: Vec<[u32; 16]> = Vec::new();
    while frames.len() < max_frames {
        // The frame's first code, from the backbone.
        let logits = guide(model.logits(&branches[0].hidden)?, if guided { Some(model.logits(&branches[1].hidden)?) } else { None }, scale)?;
        let mut l: Vec<f32> = logits.to_vec1()?;
        for &t in history.iter().collect::<std::collections::BTreeSet<_>>() {
            l[t] = if l[t] > 0. { l[t] / penalty as f32 } else { l[t] * penalty as f32 };
        }
        for v in &mut l[CODEBOOK_SIZE..EOS] {
            *v = f32::NEG_INFINITY;
        }
        let first = sample(&l, r.temperature, r.top_k, r.top_p, r.greedy, &mut rng) as usize;
        if first == EOS {
            break;
        }
        // The other codes, from the depth decoder: its first input is the backbone's state.
        let mut frame = [0u32; 16];
        frame[0] = first as u32;
        let mut caches: Vec<Cache> = branches.iter().map(|_| Cache::new(model.depth.layers())).collect();
        let mut inputs: Vec<Tensor> = branches
            .iter()
            .map(|b| -> Result<Tensor> {
                let code = model.audio_embed.narrow(0, first, 1)?.unsqueeze(0)?;
                Tensor::cat(&[b.hidden.clone(), code], 1)
            })
            .collect::<Result<_>>()?;
        for c in 1..NUM_CODEBOOKS {
            let mut outs = Vec::new();
            for (cache, x) in caches.iter_mut().zip(&inputs) {
                let h = model.depth.forward(&model.depth_in.forward(x)?, cache)?;
                let h = h.narrow(1, h.dim(1)? - 1, 1)?.to_dtype(DType::F32)?.reshape((1, ()))?;
                outs.push(h.matmul(&model.depth_heads.get(c - 1)?)?.squeeze(0)?);
            }
            let mut outs = outs.into_iter();
            let cond = outs.next().expect("a branch");
            let mut l: Vec<f32> = guide(cond, outs.next(), scale)?.to_vec1()?;
            for v in &mut l[CODEBOOK_SIZE..] {
                *v = f32::NEG_INFINITY;
            }
            let code = sample(&l, r.temperature, r.top_k, r.top_p, r.greedy, &mut rng);
            frame[c] = code;
            let e = model.audio_embed.narrow(0, code as usize + c * AUDIO_VOCAB, 1)?.unsqueeze(0)?;
            inputs = inputs.iter().map(|_| e.clone()).collect();
        }
        history.push(first);
        frames.push(frame);
        report(event("speaking", frames.len(), max_frames));
        let x = model.frames_embedding(&[frame], &dev)?.unsqueeze(0)?;
        for b in &mut branches {
            model.step(b, &x)?;
        }
    }
    let speak_seconds = speak_started.elapsed().as_secs_f64();
    drop(model);
    drop(store);
    report(event("decoding_speech", 0, 1));
    let samples = if frames.is_empty() {
        Vec::new()
    } else {
        let codec = codec::CodecDecoder::load(&r.model_dir.join("audio_tokenizer").join("model.safetensors"), &dev)?;
        trim_leading_silence(codec.decode(&frames)?, codec::SAMPLE_RATE)
    };
    Ok((samples, frames.len(), load_seconds, speak_seconds))
}

/// A voice made with Breeze: it speaks `sample` as `description` says, and the clip's codes
/// (and the sample, its transcript) are the voice. Breeze speaks in it again; the Qwen3-TTS
/// Base model cannot (it has no speaker embedding).
pub fn design(dir: &Path, name: &str, description: &str, sample_text: &str, language: &str, seed: u64, device_index: usize, output: &Path) -> Result<(Voice, Vec<f32>)> {
    let r = Request {
        model_dir: dir.to_path_buf(),
        text: sample_text.to_string(),
        instructions: description.to_string(),
        language: language.to_string(),
        output: output.to_path_buf(),
        seed,
        device: device_index,
        max_seconds: 30.,
        temperature: 0.9,
        top_k: 50,
        top_p: 1.0,
        repetition_penalty: 1.1,
        repetition_penalty_set: true,
        cfg_scale: Some(4.0),
        greedy: false,
        voice: None,
        webgpu: false,
    };
    let (clip, frames, _, _) = speak(&r, &mut |_| {})?;
    if frames < 12 {
        candle_core::bail!("the voice sample came out too short; try a longer sample text or another seed");
    }
    let dev = device(device_index)?;
    let ref_codes = super::clone::SpeechEncoder::load(&dir.join("audio_tokenizer").join("model.safetensors"), &dev)?.encode(&clip)?;
    Ok((Voice { name: name.to_string(), description: description.to_string(), language: language.to_string(), ref_text: sample_text.to_string(), ref_codes, speaker: Vec::new() }, clip))
}
