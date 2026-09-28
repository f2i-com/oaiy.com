//! The Qwen3-TTS talker: text (and a voice) in, one frame of 16 codec ids per
//! 80 ms out. The talker picks each frame's first codebook; the code
//! predictor, a small decoder of its own, fills in the other 15.
//!
//! The 1.7B and 0.6B checkpoints differ only in width (2048 and 1024) and in
//! the projection between talker and predictor, which the 0.6B does not need
//! (both are 1024 wide).
use crate::model::{Cache, Decoder, Linear};
use crate::sampling::{gumbel, pick, sample, Rng};
use crate::voice::Voice;
use crate::weights::{tensor_bytes, TensorSource, Weights};
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::path::Path;

pub const IM_START: u32 = 151644;
pub const IM_END: u32 = 151645;
pub const ASSISTANT: u32 = 77091;
pub const USER: u32 = 872;
pub const NEWLINE: u32 = 198;
pub const TTS_PAD: u32 = 151671;
pub const TTS_BOS: u32 = 151672;
pub const TTS_EOS: u32 = 151673;
pub const CODEC_PAD: u32 = 2148;
pub const CODEC_BOS: u32 = 2149;
pub const CODEC_EOS: u32 = 2150;
pub const CODEC_THINK: u32 = 2154;
pub const CODEC_NOTHINK: u32 = 2155;
pub const CODEC_THINK_BOS: u32 = 2156;
pub const CODEC_THINK_EOS: u32 = 2157;
/// Codec ids at or above this are control tokens, never sampled (EOS aside).
pub const AUDIO_CODES: u32 = 2048;
pub const FRAMES_PER_SECOND: f64 = 12.5;
/// Positions the talker's RoPE tables cover (about 20 minutes of frames).
const MAX_POSITIONS: usize = 16384;

/// How frames are drawn. The defaults are the checkpoints'
/// `generation_config.json` (the same for the 0.6B and 1.7B Base).
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    /// The code predictor's own (`subtalker_*`).
    pub sub_temperature: f64,
    pub sub_top_k: usize,
    /// Argmax instead of sampling (for tests; it tends to loop on long text).
    pub greedy: bool,
    pub seed: u64,
    /// On a GPU, draw there (one host round trip a frame instead of 16).
    /// Top-p below 1 always draws on the host.
    pub on_device: bool,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.9, top_k: 50, top_p: 1.0, repetition_penalty: 1.05, sub_temperature: 0.9, sub_top_k: 50, greedy: false, seed: 0, on_device: true }
    }
}

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

pub struct Talker {
    talker: Decoder,
    predictor: Decoder,
    codec_embedding: Tensor,
    predictor_embeddings: Vec<Tensor>,
    predictor_heads: Vec<Linear>,
    codec_head: Linear,
    /// The 1.7B's projection from the talker's width to the predictor's.
    to_predictor: Option<Linear>,
    text_fc1: Linear,
    text_fc2: Linear,
    /// The text embedding table stays in the file: only a prompt's rows are read.
    text_rows: Box<dyn TensorSource + Send>,
    config: Json,
    hidden: usize,
    /// Added to the first codebook's logits on the device: control ids out
    /// (EOS aside), and for the first two frames EOS too.
    allowed: Tensor,
    allowed_early: Tensor,
    /// 0..vocab, for marking drawn ids.
    ids: Tensor,
    dev: Device,
}

impl Talker {
    /// The talker of a Qwen3-TTS model folder (`config.json`, `model.safetensors`).
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let config = Json::parse(&std::fs::read(dir.join("config.json")).map_err(|e| msg(format!("{}: {e}", dir.join("config.json").display())))?).map_err(candle_core::Error::wrap)?;
        let weights = Weights::open(&dir.join("model.safetensors"))?;
        Self::from_source(Box::new(weights), config, dev)
    }

    pub fn from_source(mut store: Box<dyn TensorSource + Send>, config: Json, dev: &Device) -> Result<Self> {
        let tc = config.get("talker_config").ok_or_else(|| msg("config lacks talker_config"))?;
        let cp = tc.get("code_predictor_config").ok_or_else(|| msg("config lacks code_predictor_config"))?;
        let int = |c: &Json, k: &str| c.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| msg(format!("config lacks {k}")));
        let theta = |c: &Json| c.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6);
        let s = store.as_mut();
        let talker = Decoder::load(s, "talker.model", int(tc, "num_hidden_layers")?, int(tc, "num_attention_heads")?, int(tc, "num_key_value_heads")?, int(tc, "head_dim")?, theta(tc), MAX_POSITIONS, 1e-6, dev)?;
        let predictor = Decoder::load(s, "talker.code_predictor.model", int(cp, "num_hidden_layers")?, int(cp, "num_attention_heads")?, int(cp, "num_key_value_heads")?, int(cp, "head_dim")?, theta(cp), 32, 1e-6, dev)?;
        let groups = int(tc, "num_code_groups")?;
        if groups != 16 {
            candle_core::bail!("this talker has {groups} codebooks; Qwen3-TTS 12 Hz has 16");
        }
        let mut predictor_embeddings = Vec::new();
        let mut predictor_heads = Vec::new();
        for i in 0..groups - 1 {
            predictor_embeddings.push(s.load(&format!("talker.code_predictor.model.codec_embedding.{i}.weight"), dev)?);
            predictor_heads.push(Linear::load(s, &format!("talker.code_predictor.lm_head.{i}"), dev)?);
        }
        let projection = "talker.code_predictor.small_to_mtp_projection";
        let to_predictor = if s.has(&format!("{projection}.weight")) { Some(Linear::load(s, projection, dev)?) } else { None };
        let codec_embedding = s.load("talker.model.codec_embedding.weight", dev)?;
        let (vocab, hidden) = codec_embedding.dims2()?;
        let mask = |eos: bool| -> Result<Tensor> {
            let m: Vec<f32> = (0..vocab as u32).map(|i| if i < AUDIO_CODES || (eos && i == CODEC_EOS) { 0. } else { f32::NEG_INFINITY }).collect();
            Tensor::from_vec(m, (1, vocab), dev)
        };
        Ok(Self {
            allowed: mask(true)?,
            allowed_early: mask(false)?,
            ids: Tensor::arange(0u32, vocab as u32, dev)?.unsqueeze(0)?,
            talker,
            predictor,
            codec_embedding,
            predictor_embeddings,
            predictor_heads,
            codec_head: Linear::load(s, "talker.codec_head", dev)?,
            to_predictor,
            text_fc1: Linear::load(s, "talker.text_projection.linear_fc1", dev)?,
            text_fc2: Linear::load(s, "talker.text_projection.linear_fc2", dev)?,
            text_rows: store,
            config,
            hidden,
            dev: dev.clone(),
        })
    }

    /// The talker's width: 2048 (1.7B) or 1024 (0.6B), and so the length of
    /// the speaker embeddings it takes.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The talker's decoder itself (for checks against the reference).
    pub fn decoder(&self) -> &Decoder {
        &self.talker
    }

    /// Device bytes held by the weights.
    pub fn bytes(&self) -> u64 {
        let linears = self.predictor_heads.iter().chain([&self.codec_head, &self.text_fc1, &self.text_fc2]).chain(self.to_predictor.as_ref());
        self.talker.bytes() + self.predictor.bytes() + tensor_bytes(self.predictor_embeddings.iter().chain([&self.codec_embedding])) + linears.map(Linear::bytes).sum::<u64>()
    }

    /// T(ids): text embedding rows through the text projection. (1, n, hidden).
    pub fn text(&mut self, ids: &[u32]) -> Result<Tensor> {
        let rows = self.text_rows.load_rows("talker.model.text_embedding.weight", ids, &self.dev)?.unsqueeze(0)?;
        self.text_fc2.forward(&self.text_fc1.forward(&rows)?.silu()?)
    }

    /// C(ids): codec embedding rows. (1, n, hidden).
    fn codec(&self, ids: &[u32]) -> Result<Tensor> {
        self.codec_embedding.index_select(&Tensor::new(ids, &self.dev)?, 0)?.unsqueeze(0)
    }

    /// `auto` (None) or a language the model knows (english, chinese, ...).
    pub fn language_id(&self, language: &str) -> Result<Option<u32>> {
        let language = language.trim().to_lowercase();
        if language.is_empty() || language == "auto" {
            return Ok(None);
        }
        let ids = self.config.get("talker_config").and_then(|t| t.get("codec_language_id"));
        ids.and_then(|m| m.get(&language))
            .and_then(Json::as_i64)
            .map(|v| Some(v as u32))
            .ok_or_else(|| msg(format!("speech: unknown language {language}; use auto or one of the model's languages")))
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

    /// S(frames): each frame's 16 codec embeddings summed, (1, n, hidden).
    fn frames_embedding(&self, frames: &[[u32; 16]]) -> Result<Tensor> {
        let ids = |q: usize| Tensor::from_vec(frames.iter().map(|f| f[q]).collect::<Vec<u32>>(), frames.len(), &self.dev);
        let mut e = self.codec_embedding.index_select(&ids(0)?, 0)?;
        for (i, emb) in self.predictor_embeddings.iter().enumerate() {
            e = (e + emb.index_select(&ids(i + 1)?, 0)?)?;
        }
        e.unsqueeze(0)
    }

    /// The Base talker's prefill for a cloned voice (in-context, streaming
    /// text): the codec prefix with the speaker embedding in it, then the
    /// reference transcript and the new text over codec BOS and the reference
    /// clip's frames. Text that outlasts the clip's frames is returned to be
    /// fed one token per generated frame.
    pub fn prefill_clone(&mut self, text_ids: &[u32], ref_ids: &[u32], voice: &Voice, language: Option<u32>) -> Result<(Tensor, Tensor)> {
        if voice.speaker.len() != self.hidden {
            candle_core::bail!(
                "this voice's speaker embedding has {} values but this model takes {}: it was made with another Qwen3-TTS size; make it again from its clip",
                voice.speaker.len(),
                self.hidden
            );
        }
        let mut parts = vec![self.text(&[IM_START, ASSISTANT, NEWLINE])?];
        let prefix: Vec<u32> = match language {
            Some(l) => vec![CODEC_THINK, CODEC_THINK_BOS, l, CODEC_THINK_EOS],
            None => vec![CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS],
        };
        let specials = self.text(&[TTS_PAD, TTS_BOS, TTS_EOS])?;
        let (pad, bos, eos) = (specials.narrow(1, 0, 1)?, specials.narrow(1, 1, 1)?, specials.narrow(1, 2, 1)?);
        let speaker = Tensor::from_slice(&voice.speaker, (1, 1, voice.speaker.len()), &self.dev)?.to_dtype(DType::BF16)?;
        let codec_side = Tensor::cat(&[self.codec(&prefix)?, speaker, self.codec(&[CODEC_PAD])?], 1)?;
        let n = codec_side.dim(1)?;
        let text_side = Tensor::cat(&[pad.broadcast_as((1, n - 1, pad.dim(2)?))?.contiguous()?, bos], 1)?;
        parts.push((text_side + codec_side)?);
        let mut all_ids = ref_ids.to_vec();
        all_ids.extend_from_slice(text_ids);
        let text = Tensor::cat(&[self.text(&all_ids)?, eos], 1)?;
        let codec = Tensor::cat(&[self.codec(&[CODEC_BOS])?, self.frames_embedding(&voice.ref_codes)?], 1)?;
        let (lt, lc) = (text.dim(1)?, codec.dim(1)?);
        let trailing = if lt > lc {
            parts.push((text.narrow(1, 0, lc)? + &codec)?);
            text.narrow(1, lc, lt - lc)?
        } else {
            let padded = Tensor::cat(&[text, pad.broadcast_as((1, lc - lt, pad.dim(2)?))?.contiguous()?], 1)?;
            parts.push((padded + &codec)?);
            pad
        };
        Ok((Tensor::cat(&parts, 1)?, trailing))
    }

    /// Run the prefill; frames then come one at a time from [`Talker::next_frame`].
    /// `trailing`: text still to be read, one position per generated frame
    /// (then tts_pad); `None` for the non-streaming layout.
    pub fn start(&mut self, prefill: &Tensor, trailing: Option<Tensor>, sampling: Sampling) -> Result<Generation> {
        let pad = self.text(&[TTS_PAD])?;
        let mut cache = Cache::new(self.talker.layers());
        let h = self.talker.forward(prefill, &mut cache)?;
        let hidden = h.narrow(1, h.dim(1)? - 1, 1)?;
        Ok(Generation {
            cache,
            predictor_cache: Cache::new(self.predictor.layers()),
            hidden,
            trailing_len: trailing.as_ref().map(|t| t.dim(1)).transpose()?.unwrap_or(0),
            trailing,
            pad,
            seen: vec![false; AUDIO_CODES as usize],
            seen_on_device: Tensor::zeros(self.ids.shape(), DType::U8, &self.dev)?,
            rng: Rng::new(sampling.seed),
            sampling,
            frames: 0,
            done: false,
        })
    }

    /// The next frame, or `None` once the talker has said all (codec EOS).
    pub fn next_frame(&self, g: &mut Generation) -> Result<Option<[u32; 16]>> {
        if g.done {
            return Ok(None);
        }
        // On a GPU every draw stays there (top-p needs the host).
        if g.sampling.on_device && !self.dev.is_cpu() && g.sampling.top_p >= 1.0 {
            return self.next_frame_on_device(g);
        }
        let s = &g.sampling;
        let mut logits = self.codec_head.forward(&g.hidden)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        // As the reference's processors: repetition penalty over earlier
        // first codes, no EOS for the first 2 frames, control ids
        // suppressed (EOS aside); then temperature, top-k, top-p.
        let penalty = s.repetition_penalty as f32;
        if penalty != 1.0 {
            for (l, _) in logits.iter_mut().zip(&g.seen).filter(|(_, &seen)| seen) {
                *l = if *l < 0. { *l * penalty } else { *l / penalty };
            }
        }
        if g.frames < 2 {
            logits[CODEC_EOS as usize] = f32::NEG_INFINITY;
        }
        for (id, l) in logits.iter_mut().enumerate().skip(AUDIO_CODES as usize) {
            if id as u32 != CODEC_EOS {
                *l = f32::NEG_INFINITY;
            }
        }
        let c0 = sample(&logits, s.temperature, s.top_k, s.top_p, s.greedy, &mut g.rng);
        if c0 == CODEC_EOS {
            g.done = true;
            return Ok(None);
        }
        g.seen[c0 as usize] = true;
        let frame = self.predict_rest(&g.hidden, c0, s.sub_temperature, s.sub_top_k, s.greedy, &mut g.rng, &mut g.predictor_cache)?;
        let step = g.frames;
        g.frames += 1;
        let text = match &g.trailing {
            Some(t) if step < g.trailing_len => t.narrow(1, step, 1)?,
            _ => g.pad.clone(),
        };
        g.hidden = self.talker.forward(&self.frame_embedding(&frame, &text)?, &mut g.cache)?;
        Ok(Some(frame))
    }

    /// As the host path, with the draws on the device: the frame's 16 ids come
    /// back in one read, and its next talker input is summed from the
    /// embeddings the predictor already looked up.
    fn next_frame_on_device(&self, g: &mut Generation) -> Result<Option<[u32; 16]>> {
        let s = g.sampling.clone();
        let width = s.top_k.max(s.sub_top_k).max(1);
        let noise = Tensor::from_vec(gumbel(&mut g.rng, 16 * width), (16, width), &self.dev)?;
        let vocab = self.ids.dim(1)?;
        let mut logits = self.codec_head.forward(&g.hidden)?.reshape((1, vocab))?.to_dtype(DType::F32)?;
        if s.repetition_penalty != 1.0 {
            let p = s.repetition_penalty;
            let penalized = logits.ge(0f64)?.where_cond(&logits.affine(1. / p, 0.)?, &logits.affine(p, 0.)?)?;
            logits = g.seen_on_device.where_cond(&penalized, &logits)?;
        }
        let logits = logits.broadcast_add(if g.frames < 2 { &self.allowed_early } else { &self.allowed })?;
        let c0 = pick(&logits, s.top_k, s.temperature, &noise.narrow(0, 0, 1)?, s.greedy)?;
        let e0 = self.codec_embedding.index_select(&c0, 0)?.unsqueeze(0)?;
        let mut codes = vec![c0.clone()];
        let mut embeddings = vec![e0.clone()];
        let cache = &mut g.predictor_cache;
        cache.reset();
        let mut h = self.predictor.forward(&self.project(&Tensor::cat(&[&g.hidden, &e0], 1)?)?, cache)?;
        let heads = self.predictor_heads.len();
        for (i, head) in self.predictor_heads.iter().enumerate() {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let l = head.forward(&last)?.flatten_all()?.to_dtype(DType::F32)?;
            let l = l.reshape((1, l.elem_count()))?;
            let code = pick(&l, s.sub_top_k, s.sub_temperature, &noise.narrow(0, i + 1, 1)?, s.greedy)?;
            let e = self.predictor_embeddings[i].index_select(&code, 0)?.unsqueeze(0)?;
            if i + 1 < heads {
                h = self.predictor.forward(&self.project(&e)?, cache)?;
            }
            codes.push(code);
            embeddings.push(e);
        }
        let ids = Tensor::cat(&codes, 0)?.to_vec1::<u32>()?;
        if ids[0] == CODEC_EOS {
            g.done = true;
            return Ok(None);
        }
        let frame: [u32; 16] = ids.try_into().map_err(|_| msg("a frame needs 16 codes"))?;
        let drawn = self.ids.broadcast_eq(&c0.reshape((1, 1))?)?;
        g.seen_on_device = g.seen_on_device.maximum(&drawn)?;
        let step = g.frames;
        g.frames += 1;
        let text = match &g.trailing {
            Some(t) if step < g.trailing_len => t.narrow(1, step, 1)?,
            _ => g.pad.clone(),
        };
        let summed = Tensor::cat(&embeddings, 1)?.sum_keepdim(1)?;
        g.hidden = self.talker.forward(&(summed + text)?, &mut g.cache)?;
        Ok(Some(frame))
    }

    fn project(&self, x: &Tensor) -> Result<Tensor> {
        match &self.to_predictor {
            Some(p) => p.forward(x),
            None => Ok(x.clone()),
        }
    }

    /// Codebooks 1..15 for a frame whose codebook 0 is `c0`, from the talker's
    /// hidden state `hidden` (1, 1, hidden).
    #[allow(clippy::too_many_arguments)]
    fn predict_rest(&self, hidden: &Tensor, c0: u32, temperature: f64, top_k: usize, greedy: bool, rng: &mut Rng, cache: &mut Cache) -> Result<[u32; 16]> {
        let mut frame = [0u32; 16];
        frame[0] = c0;
        cache.reset();
        let x = Tensor::cat(&[hidden, &self.codec(&[c0])?], 1)?;
        let mut h = self.predictor.forward(&self.project(&x)?, cache)?;
        for (i, head) in self.predictor_heads.iter().enumerate() {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let logits = head.forward(&last)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let code = sample(&logits, temperature, top_k, 1.0, greedy, rng);
            frame[i + 1] = code;
            if i + 1 < self.predictor_heads.len() {
                let e = self.predictor_embeddings[i].index_select(&Tensor::new(&[code], &self.dev)?, 0)?.unsqueeze(0)?;
                h = self.predictor.forward(&self.project(&e)?, cache)?;
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

    /// Every frame until the codec EOS (or `max_frames`), for callers that
    /// want the whole line at once.
    pub fn frames(&mut self, prefill: &Tensor, trailing: Option<Tensor>, sampling: Sampling, max_frames: usize, mut progress: impl FnMut(usize)) -> Result<Vec<[u32; 16]>> {
        let mut g = self.start(prefill, trailing, sampling)?;
        let mut frames = Vec::new();
        while frames.len() < max_frames {
            match self.next_frame(&mut g)? {
                Some(f) => frames.push(f),
                None => break,
            }
            progress(frames.len());
        }
        Ok(frames)
    }
}

/// One line being spoken: the talker's cache and where it is in the text.
pub struct Generation {
    cache: Cache,
    /// The code predictor's, started over every frame (its buffers kept).
    predictor_cache: Cache,
    hidden: Tensor,
    trailing: Option<Tensor>,
    trailing_len: usize,
    pad: Tensor,
    /// First codes drawn so far (for the repetition penalty), on the host
    /// and on the device.
    seen: Vec<bool>,
    seen_on_device: Tensor,
    rng: Rng,
    sampling: Sampling,
    frames: usize,
    done: bool,
}

impl Generation {
    /// Frames drawn so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Text positions still to be read after the prefill.
    pub fn trailing_len(&self) -> usize {
        self.trailing_len
    }
}
