//! Speech to text with NVIDIA's Parakeet models (a FastConformer encoder and
//! a transducer head) on Candle.
//!
//! Supported checkpoints: `nvidia/parakeet-tdt-0.6b-v2` and `-v3` and
//! `nvidia/parakeet-unified-en-0.6b` as `.nemo` files, and the Hugging Face
//! `model.safetensors` form of v3 and of `moondream/parakeet-ultra` (v3
//! retrained; its VAD head is not used here).

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod mel;
pub mod nemo;
pub mod vocab;
pub mod weights;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};

use config::{Head, ModelConfig};
use decoder::{greedy_rnnt, greedy_tdt, Decoder, DeviceScorer};
use encoder::{Encoder, Precision};
use mel::{Features, MelFrontEnd};
use vocab::Vocab;
use weights::Weights;

pub type Result<T> = candle_core::Result<T>;

pub(crate) fn bad(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// How many encoder frames the joint scores per device round trip.
const JOINT_BLOCK: usize = 8;

/// The longest audio transcribed in one pass, in seconds. Full attention
/// and the subsampling's gathers grow with the length (300 s takes about
/// 6 GB of GPU memory, 90 s about 1.5 GB); longer audio is cut where it is
/// quietest into pieces no longer than this.
pub const MAX_PASS_SECONDS: f64 = 90.0;

/// A cut falls in the quietest 100 ms of the last this many seconds before
/// the limit.
const CUT_SEARCH_SECONDS: f64 = 10.0;

/// Pieces of the audio no longer than `max` samples, each cut at the
/// quietest `frame` samples (by energy, on a `frame / 10` grid) of the last
/// `search` samples before the limit.
pub fn quiet_cuts(samples: &[f32], max: usize, search: usize, frame: usize) -> Vec<(usize, usize)> {
    let n = samples.len();
    let (search, frame) = (search.min(max / 2).max(1), frame.max(1));
    let hop = (frame / 10).max(1);
    let mut pieces = Vec::new();
    let mut start = 0;
    while n - start > max {
        let end = start + max;
        let from = end - search;
        let mut best = (f64::INFINITY, end);
        let mut at = from;
        while at + frame <= end {
            let e: f64 = samples[at..at + frame].iter().map(|&x| (x as f64) * (x as f64)).sum();
            if e < best.0 {
                best = (e, at + frame / 2);
            }
            at += hop;
        }
        pieces.push((start, best.1));
        start = best.1;
    }
    pieces.push((start, n));
    pieces
}

/// The files of one model: its weights and where its configuration and
/// tokenizer are.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub weights: PathBuf,
    /// `config.json` beside `model.safetensors` (none for a `.nemo`).
    pub config: Option<PathBuf>,
    pub processor: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
}

impl ModelFiles {
    /// A `.nemo` file, a `model.safetensors` file, or a folder holding either
    /// (a `.nemo` is preferred when both are there).
    pub fn find(path: &Path) -> Result<Self> {
        let is = |p: &Path, ext: &str| p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext));
        if path.is_file() {
            if is(path, "nemo") {
                return Ok(Self { weights: path.to_path_buf(), config: None, processor: None, tokenizer: None });
            }
            if is(path, "safetensors") {
                let dir = path.parent().unwrap_or(Path::new("."));
                return Ok(Self::beside(path.to_path_buf(), dir));
            }
            return Err(bad("not a .nemo or .safetensors file"));
        }
        if !path.is_dir() {
            return Err(bad("no such file or folder"));
        }
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        if let Some(nemo) = entries.iter().find(|p| p.is_file() && is(p, "nemo")) {
            return Ok(Self { weights: nemo.clone(), config: None, processor: None, tokenizer: None });
        }
        let st = path.join("model.safetensors");
        if st.is_file() {
            return Ok(Self::beside(st, path));
        }
        if let Some(st) = entries.iter().find(|p| p.is_file() && is(p, "safetensors")) {
            return Ok(Self::beside(st.clone(), path));
        }
        if entries.iter().any(|p| is(p, "onnx")) {
            return Err(bad("this folder holds ONNX files (an export for ONNX Runtime, as Aokie's voice server uses); oaiy-voice reads the published checkpoint instead: a .nemo (nvidia/parakeet-unified-en-0.6b, nvidia/parakeet-tdt-0.6b-v2 or -v3) or a model.safetensors with config.json and tokenizer.json (nvidia/parakeet-tdt-0.6b-v3, moondream/parakeet-ultra)"));
        }
        Err(bad("no Parakeet .nemo, or model.safetensors with config.json and tokenizer.json, in this folder"))
    }

    fn beside(weights: PathBuf, dir: &Path) -> Self {
        let opt = |n: &str| Some(dir.join(n)).filter(|p| p.is_file());
        Self { weights, config: opt("config.json"), processor: opt("processor_config.json"), tokenizer: opt("tokenizer.json") }
    }
}

/// Where the time of one transcription went.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub audio: Duration,
    pub features: Duration,
    pub encoder: Duration,
    pub decoder: Duration,
    /// Joint evaluations that went to the device and back.
    pub joint_trips: usize,
}

impl Timings {
    pub fn total(&self) -> Duration {
        self.features + self.encoder + self.decoder
    }

    /// Processing time over audio time.
    pub fn rtf(&self) -> f64 {
        self.total().as_secs_f64() / self.audio.as_secs_f64().max(1e-9)
    }
}

#[derive(Clone, Debug)]
pub struct Transcript {
    pub text: String,
    pub tokens: Vec<u32>,
    pub timings: Timings,
}

pub struct Transcriber {
    pub config: ModelConfig,
    mel: MelFrontEnd,
    encoder: Encoder,
    decoder: Decoder,
    vocab: Vocab,
    dev: Device,
    dtype: DType,
    /// A short name for logs and `/v1/models`.
    pub name: String,
}

impl Transcriber {
    /// Load a model (see [`ModelFiles::find`]) onto `dev`, with its matrix
    /// products in `dtype` (f32, f16 or bf16). Half precision runs the whole
    /// encoder in it; the prediction network and joint stay f32.
    pub fn load(path: &Path, dev: &Device, dtype: DType) -> Result<Self> {
        Self::load_with(path, dev, Precision { weights: dtype, act: dtype })
    }

    /// Like [`load`](Self::load), with the encoder's dtypes chosen apart.
    pub fn load_with(path: &Path, dev: &Device, precision: Precision) -> Result<Self> {
        let dtype = precision.weights;
        let files = ModelFiles::find(path)?;
        let mut w = Weights::open(&files.weights)?;
        let (config, vocab) = match &w {
            Weights::Nemo(_) => {
                let yaml = w.nemo_member("model_config.yaml")?.ok_or_else(|| bad("the .nemo has no model_config.yaml"))?;
                let config = ModelConfig::from_nemo_yaml(&String::from_utf8_lossy(&yaml))?;
                let sp = w.nemo_member("tokenizer.model")?.ok_or_else(|| bad("the .nemo has no tokenizer.model"))?;
                (config, Vocab::from_sentencepiece(&sp)?)
            }
            Weights::Safetensors(_) => {
                let config = files.config.as_ref().ok_or_else(|| bad(format!("no config.json beside {}", files.weights.display())))?;
                let processor = files.processor.as_ref().map(std::fs::read).transpose()?;
                let config = ModelConfig::from_hf(&std::fs::read(config)?, processor.as_deref())?;
                let tok = files.tokenizer.as_ref().ok_or_else(|| bad(format!("no tokenizer.json beside {}", files.weights.display())))?;
                (config, Vocab::from_tokenizer_json(&std::fs::read(tok)?)?)
            }
        };
        if config.att_context != [-1, -1] {
            return Err(bad(format!("limited attention context {:?} is not supported", config.att_context)));
        }
        let encoder = Encoder::load(&mut w, config.mel.n_mels, config.xscaling, precision, dev)?;
        let decoder = Decoder::load(&mut w, encoder.shape.d_model, dev)?;
        let extra = match &config.head {
            Head::Tdt { durations } => durations.len(),
            Head::Rnnt => 0,
        };
        if decoder.shape.outputs != decoder.shape.tokens + extra {
            return Err(bad(format!("the joint has {} outputs for {} tokens and {extra} durations", decoder.shape.outputs, decoder.shape.tokens)));
        }
        if vocab.len() + 1 < decoder.shape.tokens {
            return Err(bad(format!("the tokenizer has {} pieces for {} tokens", vocab.len(), decoder.shape.tokens - 1)));
        }
        let name = model_name(&files.weights);
        Ok(Self { mel: MelFrontEnd::new(config.mel.clone()), config, encoder, decoder, vocab, dev: dev.clone(), dtype, name })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn sample_rate(&self) -> usize {
        self.config.mel.sample_rate
    }

    /// The mel features of 16 kHz audio.
    pub fn features(&self, samples: &[f32]) -> Features {
        self.mel.features(samples)
    }

    /// The encoder's output `(frames, d_model)` for features, as f32.
    pub fn encode(&self, f: &Features) -> Result<Tensor> {
        let x = Tensor::from_vec(f.data.clone(), (f.n_mels, f.frames), &Device::Cpu)?.to_device(&self.dev)?;
        self.encoder.forward(&x)?.to_dtype(DType::F32)
    }

    /// The encoder after its first `n` layers (parity tests).
    pub fn encode_layers(&self, f: &Features, n: usize) -> Result<Tensor> {
        let x = Tensor::from_vec(f.data.clone(), (f.n_mels, f.frames), &Device::Cpu)?.to_device(&self.dev)?;
        self.encoder.forward_layers(&x, n)?.to_dtype(DType::F32)
    }

    /// Greedy decoding of encoder output; returns the tokens and the number
    /// of joint round trips.
    pub fn decode(&self, enc: &Tensor) -> Result<(Vec<u32>, usize)> {
        let frames = enc.dim(0)?;
        let blank = self.decoder.blank();
        let mut scorer = DeviceScorer::new(&self.decoder, enc, self.decoder.shape.tokens, JOINT_BLOCK)?;
        let tokens = match &self.config.head {
            Head::Rnnt => greedy_rnnt(frames, blank, self.config.max_symbols, &mut scorer)?,
            Head::Tdt { durations } => greedy_tdt(frames, blank, durations, self.config.max_symbols, &mut scorer)?,
        };
        Ok((tokens, scorer.trips))
    }

    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.vocab.decode(tokens)
    }

    /// Transcribe mono audio at the model's sample rate, in [-1, 1].
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        let rate = self.sample_rate() as f64;
        let max = (MAX_PASS_SECONDS * rate) as usize;
        if samples.len() <= max {
            return self.transcribe_pass(samples);
        }
        let mut tokens = Vec::new();
        let mut timings = Timings::default();
        for (a, b) in quiet_cuts(samples, max, (CUT_SEARCH_SECONDS * rate) as usize, (0.1 * rate) as usize) {
            let t = self.transcribe_pass(&samples[a..b])?;
            tokens.extend(t.tokens);
            timings.audio += t.timings.audio;
            timings.features += t.timings.features;
            timings.encoder += t.timings.encoder;
            timings.decoder += t.timings.decoder;
            timings.joint_trips += t.timings.joint_trips;
        }
        // Each piece begins with a word, whose first token carries the space.
        Ok(Transcript { text: self.detokenize(&tokens), tokens, timings })
    }

    /// One pass of the model over at most [`MAX_PASS_SECONDS`] of audio.
    fn transcribe_pass(&self, samples: &[f32]) -> Result<Transcript> {
        let mut timings = Timings { audio: Duration::from_secs_f64(samples.len() as f64 / self.sample_rate() as f64), ..Default::default() };
        let started = Instant::now();
        let f = self.features(samples);
        timings.features = started.elapsed();
        // Too short for one encoder frame: nothing was said.
        if Encoder::out_frames(f.frames) < 1 || f.frames < 2 {
            return Ok(Transcript { text: String::new(), tokens: Vec::new(), timings });
        }
        let started = Instant::now();
        let enc = self.encode(&f)?;
        // Wait for the device, so the time is the encoder's.
        self.dev.synchronize()?;
        timings.encoder = started.elapsed();
        let started = Instant::now();
        let (tokens, trips) = self.decode(&enc)?;
        timings.decoder = started.elapsed();
        timings.joint_trips = trips;
        Ok(Transcript { text: self.detokenize(&tokens), tokens, timings })
    }
}

/// A model's short name: the `.nemo`'s stem, or the folder of a safetensors file.
fn model_name(weights: &Path) -> String {
    let stem = if weights.extension().is_some_and(|e| e.eq_ignore_ascii_case("nemo")) { weights.file_stem() } else { weights.parent().and_then(|p| p.file_name()) };
    stem.map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "parakeet".into())
}

#[cfg(test)]
mod cut_tests {
    use super::quiet_cuts;

    #[test]
    fn short_audio_is_one_piece() {
        assert_eq!(quiet_cuts(&[0.5; 100], 100, 20, 4), vec![(0, 100)]);
        assert_eq!(quiet_cuts(&[], 100, 20, 4), vec![(0, 0)]);
    }

    #[test]
    fn long_audio_is_cut_in_its_quiet_parts() {
        // Loud everywhere except two quiet stretches, at 85..90 and 170..175.
        let mut x = vec![0.5f32; 260];
        x[85..90].fill(0.0);
        x[170..175].fill(0.0);
        let pieces = quiet_cuts(&x, 100, 30, 4);
        assert_eq!(pieces.len(), 3, "{pieces:?}");
        assert_eq!(pieces[0].0, 0);
        assert!((85..90).contains(&pieces[0].1), "{pieces:?}");
        assert!((170..175).contains(&pieces[1].1), "{pieces:?}");
        assert_eq!(pieces[2].1, 260);
        for w in pieces.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        assert!(pieces.iter().all(|(a, b)| b - a <= 100));
    }
}
