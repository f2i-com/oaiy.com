//! Qwen3-TTS's talker on WebGPU (a speech job's `backend` "webgpu": any GPU; Candle's needs CUDA, its models BF16):
//! [`oaiy_tts::talker`]'s host path, its two Qwen3 decoders ([`crate::qwen3_wgpu`], the talker's keys and values kept
//! for the line, the code predictor's from each frame's start), each draw on the host as the reference's processors
//! make it (the repetition penalty over earlier first codes, no end for the first two frames, control ids out; then
//! temperature, top-k and top-p), the embeddings' rows on the host.
use crate::ltx::store::Store;
use crate::qwen3_wgpu::{mat, vector, Mat, WgpuQwen3};
use candle_core::{Device, Result};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs_wgpu::WgpuBackend;
use oaiy_engine::json::Json;
use oaiy_tts::sampling::{sample, Rng};
use oaiy_tts::talker::{Sampling, ASSISTANT, AUDIO_CODES, CODEC_BOS, CODEC_EOS, CODEC_NOTHINK, CODEC_PAD, CODEC_THINK, CODEC_THINK_BOS, CODEC_THINK_EOS, IM_END, IM_START, NEWLINE, TTS_BOS, TTS_EOS, TTS_PAD, USER};
use oaiy_tts::voice::Voice;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("speech on WebGPU: {e}"))
}

/// A linear layer: its f16 weight and bias.
struct Lin {
    m: Mat,
    b: Option<DeviceVec>,
}

impl Lin {
    fn load(store: &mut Store, gpu: &WgpuBackend, prefix: &str) -> Result<Self> {
        let bias = format!("{prefix}.bias");
        let b = if store.index.get(&bias).is_some() { Some(vector(store, gpu, &bias)?) } else { None };
        Ok(Self { m: mat(store, gpu, &format!("{prefix}.weight"))?, b })
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows(&self.m.w, self.m.n, self.m.k, x, y, rows);
        if let Some(b) = &self.b {
            r.add_bias_rows(y, b, rows, self.m.n);
        }
    }
}

/// A table's rows on the host (`[rows, width]`).
struct Table {
    values: Vec<f32>,
    width: usize,
}

impl Table {
    fn load(store: &mut Store, key: &str) -> Result<Self> {
        let t = store.tensor_f32(key, &Device::Cpu)?;
        let width = t.dims2()?.1;
        Ok(Self { values: t.flatten_all()?.to_vec1::<f32>()?, width })
    }

    fn row(&self, id: u32) -> &[f32] {
        &self.values[id as usize * self.width..(id as usize + 1) * self.width]
    }
}

pub struct WgpuTalker {
    gpu: WgpuBackend,
    talker: WgpuQwen3,
    predictor: WgpuQwen3,
    codec_embedding: Table,
    predictor_embeddings: Vec<Table>,
    codec_head: Lin,
    predictor_heads: Vec<Lin>,
    /// The 1.7B's projection from the talker's width to the predictor's.
    to_predictor: Option<Lin>,
    text_fc1: Lin,
    text_fc2: Lin,
    /// The text embedding table stays in the file: only a prompt's rows are read.
    store: Store,
    config: Json,
    hidden: usize,
    predictor_hidden: usize,
}

impl WgpuTalker {
    /// The talker of a Qwen3-TTS model folder (`config.json`, `model.safetensors`) on WebGPU device `device`.
    pub fn load(dir: &Path, device: usize) -> Result<Self> {
        let config = Json::parse(&std::fs::read(dir.join("config.json"))?).map_err(candle_core::Error::wrap)?;
        let gpu = WgpuBackend::nth(device, None).map_err(err)?;
        let tc = config.get("talker_config").ok_or_else(|| err("config lacks talker_config"))?;
        let cp = tc.get("code_predictor_config").ok_or_else(|| err("config lacks code_predictor_config"))?;
        let int = |c: &Json, k: &str| c.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| err(format!("config lacks {k}")));
        let theta = |c: &Json| c.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6);
        if int(tc, "num_code_groups")? != 16 {
            candle_core::bail!("this talker's codebooks are not Qwen3-TTS 12 Hz's 16");
        }
        let mut store = Store::open(&dir.join("model.safetensors"), 0)?;
        let s = &mut store;
        let talker = WgpuQwen3::load(s, &gpu, "talker.model", int(tc, "num_hidden_layers")?, int(tc, "num_attention_heads")?, int(tc, "num_key_value_heads")?, int(tc, "head_dim")?, theta(tc), 1e-6)?;
        let predictor = WgpuQwen3::load(s, &gpu, "talker.code_predictor.model", int(cp, "num_hidden_layers")?, int(cp, "num_attention_heads")?, int(cp, "num_key_value_heads")?, int(cp, "head_dim")?, theta(cp), 1e-6)?;
        let mut predictor_embeddings = Vec::new();
        let mut predictor_heads = Vec::new();
        for i in 0..15 {
            predictor_embeddings.push(Table::load(s, &format!("talker.code_predictor.model.codec_embedding.{i}.weight"))?);
            predictor_heads.push(Lin::load(s, &gpu, &format!("talker.code_predictor.lm_head.{i}"))?);
        }
        let projection = "talker.code_predictor.small_to_mtp_projection";
        let to_predictor = if s.index.get(&format!("{projection}.weight")).is_some() { Some(Lin::load(s, &gpu, projection)?) } else { None };
        let codec_embedding = Table::load(s, "talker.model.codec_embedding.weight")?;
        let (hidden, predictor_hidden) = (talker.hidden(), predictor.hidden());
        Ok(Self {
            codec_head: Lin::load(s, &gpu, "talker.codec_head")?,
            text_fc1: Lin::load(s, &gpu, "talker.text_projection.linear_fc1")?,
            text_fc2: Lin::load(s, &gpu, "talker.text_projection.linear_fc2")?,
            talker,
            predictor,
            codec_embedding,
            predictor_embeddings,
            predictor_heads,
            to_predictor,
            store,
            config,
            hidden,
            predictor_hidden,
            gpu,
        })
    }

    /// The talker's width (and so a speaker embedding's length).
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The device it runs on (the codec's after it).
    pub fn gpu(&self) -> &WgpuBackend {
        &self.gpu
    }

    /// T(ids): text embedding rows through the text projection (`[n, hidden]`, on the host).
    fn text(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let g = &self.gpu;
        let rows = self.store.rows("talker.model.text_embedding.weight", ids, &Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
        let width = rows.dims2()?.1;
        let n = ids.len();
        let x = g.vec(n * width);
        g.upload(&x, &rows.flatten_all()?.to_vec1::<f32>()?);
        let (a, b, y) = (g.vec(n * self.text_fc1.m.n), g.vec(n * self.text_fc1.m.n), g.vec(n * self.hidden));
        let mut rec = g.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        self.text_fc1.run(r, &x, &a, n);
        // SiLU: a times its own sigmoid
        r.mul_sigmoid(&a, &a, &b, n * self.text_fc1.m.n);
        self.text_fc2.run(r, &b, &y, n);
        r.read(&y);
        rec.finish().pop().ok_or_else(|| err("the text rows were not read"))
    }

    /// C(ids): codec embedding rows (on the host).
    fn codec(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter().flat_map(|&i| self.codec_embedding.row(i).iter().copied()).collect()
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
            .ok_or_else(|| err(format!("speech: unknown language {language}; use auto or one of the model's languages")))
    }

    /// [`oaiy_tts::talker::Talker::prefill`]'s rows (`[n, hidden]`).
    pub fn prefill(&mut self, text_ids: &[u32], instruct_ids: Option<&[u32]>, language: Option<u32>) -> Result<Vec<f32>> {
        let h = self.hidden;
        let mut rows = Vec::new();
        if let Some(ids) = instruct_ids {
            let mut full = vec![IM_START, USER, NEWLINE];
            full.extend_from_slice(ids);
            full.extend_from_slice(&[IM_END, NEWLINE]);
            rows.extend(self.text(&full)?);
        }
        rows.extend(self.text(&[IM_START, ASSISTANT, NEWLINE])?);
        let prefix: Vec<u32> = match language {
            Some(l) => vec![CODEC_THINK, CODEC_THINK_BOS, l, CODEC_THINK_EOS, CODEC_PAD],
            None => vec![CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS, CODEC_PAD],
        };
        let specials = self.text(&[TTS_PAD, TTS_BOS, TTS_EOS])?;
        let (pad, bos, eos) = (&specials[..h], &specials[h..2 * h], &specials[2 * h..]);
        let codec = self.codec(&prefix);
        for (i, c) in codec.chunks(h).enumerate() {
            let t = if i + 1 < prefix.len() { pad } else { bos };
            rows.extend(t.iter().zip(c).map(|(a, b)| a + b));
        }
        let mut words = self.text(text_ids)?;
        words.extend_from_slice(eos);
        let pads = self.codec(&vec![CODEC_PAD; text_ids.len() + 1]);
        rows.extend(words.iter().zip(&pads).map(|(a, b)| a + b));
        rows.extend(pad.iter().zip(self.codec(&[CODEC_BOS])).map(|(a, b)| a + b));
        Ok(rows)
    }

    /// S(frames): each frame's 16 codec embeddings summed (`[n, hidden]`, on the host).
    fn frames_embedding(&self, frames: &[[u32; 16]]) -> Vec<f32> {
        frames
            .iter()
            .flat_map(|f| {
                let mut e = self.codec_embedding.row(f[0]).to_vec();
                for (i, t) in self.predictor_embeddings.iter().enumerate() {
                    e.iter_mut().zip(t.row(f[i + 1])).for_each(|(a, b)| *a += b);
                }
                e
            })
            .collect()
    }

    /// [`oaiy_tts::talker::Talker::prefill_clone`]'s rows, and the text that outlasts the clip's frames (`[n, hidden]`
    /// each).
    pub fn prefill_clone(&mut self, text_ids: &[u32], ref_ids: &[u32], voice: &Voice, language: Option<u32>) -> Result<(Vec<f32>, Vec<f32>)> {
        let h = self.hidden;
        if voice.speaker.len() != h {
            candle_core::bail!("this voice's speaker embedding has {} values but this model takes {h}: it was made with another Qwen3-TTS size; make it again from its clip", voice.speaker.len());
        }
        let mut rows = self.text(&[IM_START, ASSISTANT, NEWLINE])?;
        let prefix: Vec<u32> = match language {
            Some(l) => vec![CODEC_THINK, CODEC_THINK_BOS, l, CODEC_THINK_EOS],
            None => vec![CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS],
        };
        let specials = self.text(&[TTS_PAD, TTS_BOS, TTS_EOS])?;
        let (pad, bos, eos) = (specials[..h].to_vec(), specials[h..2 * h].to_vec(), specials[2 * h..].to_vec());
        let mut codec_side = self.codec(&prefix);
        codec_side.extend_from_slice(&voice.speaker);
        codec_side.extend(self.codec(&[CODEC_PAD]));
        let n = codec_side.len() / h;
        for (i, c) in codec_side.chunks(h).enumerate() {
            let t = if i + 1 < n { &pad } else { &bos };
            rows.extend(t.iter().zip(c).map(|(a, b)| a + b));
        }
        let mut all_ids = ref_ids.to_vec();
        all_ids.extend_from_slice(text_ids);
        let mut text = self.text(&all_ids)?;
        text.extend_from_slice(&eos);
        let mut codec = self.codec(&[CODEC_BOS]);
        codec.extend(self.frames_embedding(&voice.ref_codes));
        let (lt, lc) = (text.len() / h, codec.len() / h);
        let trailing = if lt > lc {
            rows.extend(text[..lc * h].iter().zip(&codec).map(|(a, b)| a + b));
            text[lc * h..].to_vec()
        } else {
            let mut padded = text;
            for _ in lt..lc {
                padded.extend_from_slice(&pad);
            }
            rows.extend(padded.iter().zip(&codec).map(|(a, b)| a + b));
            pad
        };
        Ok((rows, trailing))
    }

    /// Every frame until the codec's end (or `max_frames`), as [`oaiy_tts::talker::Talker::frames`] on the host's draws.
    pub fn frames(&mut self, prefill: &[f32], trailing: Option<Vec<f32>>, sampling: Sampling, max_frames: usize, mut progress: impl FnMut(usize)) -> Result<Vec<[u32; 16]>> {
        let (h, ph) = (self.hidden, self.predictor_hidden);
        let pad = self.text(&[TTS_PAD])?;
        let g = &self.gpu;
        let n = prefill.len() / h;
        let trailing_len = trailing.as_ref().map_or(0, |t| t.len() / h);
        let mut cache = self.talker.cache(g, n + max_frames + 1);
        let mut pcache = self.predictor.cache(g, 17);
        let vocab = self.codec_head.m.n;
        let mut rng = Rng::new(sampling.seed);
        let mut seen = vec![false; AUDIO_CODES as usize];
        // the talker's step: its last row's state and the first codebook's logits
        let talker_step = |this: &Self, cache: &mut crate::qwen3_wgpu::Cache, x: &[f32]| -> Result<(Vec<f32>, Vec<f32>)> {
            let t = x.len() / h;
            let (xd, states, logits) = (g.vec(x.len()), g.vec(t * h), g.vec(t * vocab));
            g.upload(&xd, x);
            let mut rec = g.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            this.talker.step(g, r, &xd, t, cache, &states);
            this.codec_head.run(r, &states, &logits, t);
            r.read_range(&states, (t - 1) * h, h);
            r.read_range(&logits, (t - 1) * vocab, vocab);
            let mut out = rec.finish();
            let l = out.pop().ok_or_else(|| err("the logits were not read"))?;
            Ok((out.pop().ok_or_else(|| err("the state was not read"))?, l))
        };
        // the predictor's step: its last row's logits from head `i` (its inputs the talker's width, projected to its own)
        let predictor_step = |this: &Self, pcache: &mut crate::qwen3_wgpu::Cache, x: &[f32], i: usize| -> Result<Vec<f32>> {
            let t = x.len() / h;
            let head = &this.predictor_heads[i];
            let (xd, xp, states, logits) = (g.vec(x.len()), g.vec(t * ph), g.vec(t * ph), g.vec(t * head.m.n));
            g.upload(&xd, x);
            let mut rec = g.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            let input = match &this.to_predictor {
                Some(p) => {
                    p.run(r, &xd, &xp, t);
                    &xp
                }
                None => &xd,
            };
            this.predictor.step(g, r, input, t, pcache, &states);
            head.run(r, &states, &logits, t);
            r.read_range(&logits, (t - 1) * head.m.n, head.m.n);
            rec.finish().pop().ok_or_else(|| err("the logits were not read"))
        };
        let (mut hidden, mut logits) = talker_step(self, &mut cache, prefill)?;
        let mut frames = Vec::new();
        while frames.len() < max_frames {
            let s = &sampling;
            let penalty = s.repetition_penalty as f32;
            if penalty != 1.0 {
                for (l, _) in logits.iter_mut().zip(&seen).filter(|(_, &seen)| seen) {
                    *l = if *l < 0. { *l * penalty } else { *l / penalty };
                }
            }
            if frames.len() < 2 {
                logits[CODEC_EOS as usize] = f32::NEG_INFINITY;
            }
            for (id, l) in logits.iter_mut().enumerate().skip(AUDIO_CODES as usize) {
                if id as u32 != CODEC_EOS {
                    *l = f32::NEG_INFINITY;
                }
            }
            let c0 = sample(&logits, s.temperature, s.top_k, s.top_p, s.greedy, &mut rng);
            if c0 == CODEC_EOS {
                break;
            }
            seen[c0 as usize] = true;
            // the other 15 codes, the predictor's from this frame's start
            pcache.len = 0;
            let mut frame = [0u32; 16];
            frame[0] = c0;
            let mut x = hidden.clone();
            x.extend_from_slice(self.codec_embedding.row(c0));
            for i in 0..15 {
                let l = predictor_step(self, &mut pcache, &x, i)?;
                let code = sample(&l, s.sub_temperature, s.sub_top_k, 1.0, s.greedy, &mut rng);
                frame[i + 1] = code;
                x = self.predictor_embeddings[i].row(code).to_vec();
            }
            let step = frames.len();
            frames.push(frame);
            progress(frames.len());
            if frames.len() >= max_frames {
                break;
            }
            // the next input: the frame's embeddings summed, plus the text's
            let mut e = self.frames_embedding(&[frame]);
            let text = match &trailing {
                Some(t) if step < trailing_len => &t[step * h..(step + 1) * h],
                _ => &pad[..],
            };
            e.iter_mut().zip(text).for_each(|(a, b)| *a += b);
            (hidden, logits) = talker_step(self, &mut cache, &e)?;
        }
        Ok(frames)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// The talker's decoder's states for a prefill on WebGPU against Candle's (`--ignored --nocapture`, `webgpu cuda`),
    /// and the first logits' best codes.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs Qwen3-TTS and CUDA"]
    fn the_webgpu_talkers_states_are_candles() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let tok = crate::tts::tokenizer(&dir)?;
        let text_ids = oaiy_tts::text::encode(&tok, "Hello there, this is a short test of speech on the GPU.")?;
        let dev = Device::new_cuda(0)?;
        let mut ct = oaiy_tts::talker::Talker::load(&dir, &dev)?;
        let lang = ct.language_id("english")?;
        let prefill = ct.prefill(&text_ids, None, lang)?;
        let mut cache = oaiy_tts::model::Cache::new(ct.decoder().layers());
        let cs = ct.decoder().forward(&prefill, &mut cache)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let pf = prefill.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        drop(ct);
        let wt = WgpuTalker::load(&dir, 1)?;
        let (g, h) = (&wt.gpu, wt.hidden);
        let n = pf.len() / h;
        let mut wc = wt.talker.cache(g, n);
        let (xd, sd) = (g.vec(pf.len()), g.vec(pf.len()));
        g.upload(&xd, &pf);
        let mut rec = g.begin();
        wt.talker.step(g, rec.as_mut(), &xd, n, &mut wc, &sd);
        rec.read(&sd);
        let ws = rec.finish().pop().unwrap();
        for (i, (a, b)) in cs.chunks(h).zip(ws.chunks(h)).enumerate() {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let nn = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            if i < 3 || i + 3 >= n {
                eprintln!("row {i}: cosine {:.6}, norms {:.3} and {:.3}", dot / (nn(a) * nn(b)).max(1e-30), nn(a), nn(b));
            }
        }
        Ok(())
    }

    /// The talker on WebGPU against Candle's on CUDA, both greedy (`--ignored --nocapture`, built with `webgpu cuda`;
    /// OAIY_TTS the model folder, the 0.6B Base by default): the prefill's rows, and how many frames agree.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs Qwen3-TTS and CUDA"]
    fn the_webgpu_talker_is_candles() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let tok = crate::tts::tokenizer(&dir)?;
        let text_ids = oaiy_tts::text::encode(&tok, "Hello there, this is a short test of speech on the GPU.")?;
        let greedy = Sampling { greedy: true, ..Sampling::default() };
        let mut ct = oaiy_tts::talker::Talker::load(&dir, &Device::new_cuda(0)?)?;
        let lang = ct.language_id("english")?;
        let cp = ct.prefill(&text_ids, None, lang)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let t = std::time::Instant::now();
        let prefill = ct.prefill(&text_ids, None, lang)?;
        let cframes = ct.frames(&prefill, None, greedy.clone(), 60, |_| {})?;
        eprintln!("Candle: {} frames in {:.2} s", cframes.len(), t.elapsed().as_secs_f64());
        drop(ct);
        let mut wt = WgpuTalker::load(&dir, 1)?;
        let wp = wt.prefill(&text_ids, None, lang)?;
        let h = wt.hidden();
        let worst = cp.chunks(h).zip(wp.chunks(h)).map(|(a, b)| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b)).max(1e-30)
        }).fold(1f64, f64::min);
        eprintln!("prefill: {} rows, the worst's cosine {worst:.6}", cp.len() / h);
        assert!(cp.len() == wp.len() && worst > 0.999);
        let t = std::time::Instant::now();
        let wframes = wt.frames(&wp, None, greedy, 60, |_| {})?;
        eprintln!("WebGPU: {} frames in {:.2} s", wframes.len(), t.elapsed().as_secs_f64());
        for i in 0..3 {
            eprintln!("frame {i}: Candle {:?}", cframes[i]);
            eprintln!("frame {i}: WebGPU {:?}", wframes[i]);
        }
        let same = cframes.iter().zip(&wframes).take_while(|(a, b)| a == b).count();
        let first_codes = cframes.iter().zip(&wframes).filter(|(a, b)| a[0] == b[0]).count();
        eprintln!("frames alike from the start: {same} of {}; first codes alike {first_codes}", cframes.len().min(wframes.len()));
        assert!(same >= 5, "the first frames differ");
        Ok(())
    }
}

#[cfg(test)]
mod golden {
    use super::*;

    fn cosines(a: &[f32], b: &[f32], width: usize) -> f64 {
        a.chunks(width).zip(b.chunks(width)).map(|(x, y)| {
            let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
            let n = |v: &[f32]| v.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(x) * n(y)).max(1e-30)
        }).fold(1f64, f64::min)
    }

    /// The talker on WebGPU against the official implementation's dumps (`--ignored --nocapture`; OAIY_TTS_GOLDEN, else
    /// E:/deepseek/nrob/target/qwen-tts-golden; OAIY_TTS_MODEL, else the 1.7B VoiceDesign): the prefill's rows, the
    /// talker's states for them, the code predictor's for its first input, and the first greedy frames.
    #[test]
    #[ignore = "needs Qwen3-TTS VoiceDesign and the reference's dumps"]
    fn the_webgpu_talker_is_the_references() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("OAIY_TTS_GOLDEN").unwrap_or_else(|_| "E:/deepseek/nrob/target/qwen-tts-golden".into()));
        let model = std::path::PathBuf::from(std::env::var("OAIY_TTS_MODEL").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign".into()));
        let f32s = |name: &str| -> Vec<f32> { std::fs::read(root.join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
        let ints = |name: &str| -> Vec<u32> { std::fs::read(root.join(name)).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect() };
        let meta = Json::parse(&std::fs::read(root.join("meta.json"))?).map_err(candle_core::Error::wrap)?;
        let text = meta.get("text").and_then(Json::as_str).unwrap_or_default();
        let instruct = meta.get("instruct").and_then(Json::as_str).unwrap_or_default();
        let tok = crate::tts::tokenizer(&model)?;
        let (text_ids, inst_ids) = (oaiy_tts::text::encode(&tok, text)?, oaiy_tts::text::encode(&tok, instruct)?);
        let mut wt = WgpuTalker::load(&model, 1)?;
        let lang = wt.language_id("english")?;
        let (h, ph) = (wt.hidden, wt.predictor_hidden);
        // the prefill's rows
        let prefill = wt.prefill(&text_ids, Some(&inst_ids), lang)?;
        let want = f32s("talker_prefill_in.f32");
        eprintln!("prefill rows: {} and {}, the worst's cosine {:.6}", prefill.len() / h, want.len() / h, cosines(&prefill, &want, h));
        // the talker's states for the reference's rows
        let g = &wt.gpu;
        let n = want.len() / h;
        let mut cache = wt.talker.cache(g, n);
        let (xd, sd) = (g.vec(want.len()), g.vec(want.len()));
        g.upload(&xd, &want);
        let mut rec = g.begin();
        wt.talker.step(g, rec.as_mut(), &xd, n, &mut cache, &sd);
        rec.read(&sd);
        let got = rec.finish().pop().unwrap();
        eprintln!("talker states: the worst row's cosine {:.6}", cosines(&got, &f32s("talker_prefill_out.f32"), h));
        // the predictor's for its first input
        let cin = f32s("cp_prefill_in.f32");
        let mut pc = wt.predictor.cache(g, 17);
        let (xd, sd) = (g.vec(cin.len()), g.vec(cin.len()));
        g.upload(&xd, &cin);
        let mut rec = g.begin();
        wt.predictor.step(g, rec.as_mut(), &xd, cin.len() / ph, &mut pc, &sd);
        rec.read(&sd);
        let got = rec.finish().pop().unwrap();
        eprintln!("predictor states: the worst row's cosine {:.6}", cosines(&got, &f32s("cp_prefill_out.f32"), ph));
        // the first greedy frames
        let frames = wt.frames(&prefill, None, Sampling { greedy: true, ..Sampling::default() }, 6, |_| {})?;
        let golden = ints("greedy_codes.i32");
        for (i, f) in frames.iter().enumerate() {
            eprintln!("frame {i}: {:?}\n   want: {:?}", f, &golden[i * 16..(i + 1) * 16]);
        }
        let same = frames.iter().enumerate().take_while(|(i, f)| f[..] == golden[i * 16..(i + 1) * 16]).count();
        eprintln!("greedy frames alike from the start: {same}");
        // (as Candle's test of the same dumps: the reference ran in BF16, and greedy draws part after a frame or two)
        assert!(frames[0][0] == golden[0] && same >= 2, "greedy frames diverge at once");
        Ok(())
    }
}
