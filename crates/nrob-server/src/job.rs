//! What a request hands a model worker and what comes back: the job, its
//! sampling settings, and the events a reply streams as. Shared by every
//! engine (DeepSeek on CUDA, GGUF through llama-rs on CUDA, WebGPU or CPU), so
//! it depends on no GPU crate.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use dsv41::vision::Prepared;

#[derive(Clone, Debug)]
pub struct Sampling {
    /// 0 = greedy.
    pub temperature: f32,
    pub reasoning_repeat_penalty: f32,
    pub reasoning_repeat_last_n: usize,
    pub top_p: f32,
    /// 0 = no limit (up to [`CANDIDATES`]).
    pub top_k: usize,
    pub seed: u64,
}

/// Tokens considered when sampling (the tail beyond is negligible mass and
/// a full sort of the 129k vocabulary per token is not).
const CANDIDATES: usize = 512;

/// An image of a prompt, sized for the tower; its span starts at `start`.
pub struct JobImage {
    pub start: usize,
    pub prep: ImagePrep,
    /// Of the encoded bytes: the prefix cache's notion of "the same image".
    pub hash: u64,
}

pub enum ImagePrep {
    Deepseek(Prepared),
    Qwen { pixels: ggml_rs::Tensor, side: usize },
}
impl ImagePrep {
    pub fn n_tokens(&self) -> usize {
        match self { Self::Deepseek(p) => p.n_tokens(), Self::Qwen { side, .. } => side * side }
    }
    pub(crate) fn deepseek(&self) -> nrob::Result<&Prepared> {
        match self { Self::Deepseek(p) => Ok(p), _ => Err(nrob::Error::Arg("Qwen image sent to DeepSeek worker".into())) }
    }
}

pub struct Job {
    pub tools: Vec<nrob::json::Json>,
    pub tool_precision: bool,
    pub observer_context: String,
    pub prompt: Vec<u32>,
    pub images: Vec<JobImage>,
    pub max_tokens: usize,
    /// Reasoning tokens the reply may spend before it has to answer (when
    /// the prompt opens a `<think>` block); `None`: no limit.
    pub think_budget: Option<usize>,
    pub sampling: Sampling,
    /// Set by the request side to stop early (a stop string, a closed
    /// connection).
    pub cancel: Arc<AtomicBool>,
    pub events: Sender<Event>,
    /// Incognito: keep nothing of this request. No prompt state goes to disk,
    /// and when it ends the engine drops its live state and checkpoints, so
    /// the next request cannot reuse (or reveal) this one's prefix.
    pub forget: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// End of sequence (or cancelled).
    Stop,
    /// `max_tokens` or the context limit.
    Length,
}

pub enum Event {
    /// How far the prompt has got: `done` of the `total` tokens the prefix
    /// cache did not have (sent as it goes, the first time with `done` 0).
    Progress { done: usize, total: usize },
    /// The prompt is in: `cached` of its tokens came from the prefix cache.
    Prefilled { cached: usize },
    /// Sent before processing the uncached tail; no prompt text is exposed.
    CacheReuse { cached: usize, source: &'static str, common: usize },
    /// How far the reply's reasoning has got: `used` tokens of its `budget`
    /// (`None`: no limit); `done` once it has closed (sent every
    /// [`THINKING_EVERY`] tokens, and at the end).
    Thinking { used: usize, budget: Option<usize>, done: bool },
    /// More reply text (whole UTF-8 characters).
    Text(String),
    Observer(String),
    ReasoningPreview(String),
    ContentPreview(String),
    ObserverDelta { text: String, thinking: bool, start: bool },
    ToolPreview { text: String, start: bool },
    ObserverProgress { action: String, comment: String },
    Done { finish: Finish, completion_tokens: usize },
    Error(String),
}

pub(crate) fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pick the next token: greedy at temperature 0, else temperature, top-k
/// and top-p (nucleus) sampling over the leading candidates.
pub fn sample(logits: &[f32], s: &Sampling, rng: &mut u64) -> u32 {
    let greedy = || {
        logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
    };
    if s.temperature <= 0.0 || logits.is_empty() {
        return greedy();
    }
    let k = if s.top_k > 0 { s.top_k.min(CANDIDATES) } else { CANDIDATES }.min(logits.len());
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let desc = |a: &u32, b: &u32| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap_or(std::cmp::Ordering::Equal);
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, desc);
        idx.truncate(k);
    }
    idx.sort_unstable_by(desc);
    let top = logits[idx[0] as usize];
    let mut probs: Vec<f64> = idx.iter().map(|&i| (((logits[i as usize] - top) / s.temperature) as f64).exp()).collect();
    let total: f64 = probs.iter().sum();
    let mut keep = probs.len();
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p / total;
            if acc >= s.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
    }
    probs.truncate(keep);
    let total: f64 = probs.iter().sum();
    let mut r = (splitmix(rng) >> 11) as f64 / (1u64 << 53) as f64 * total;
    for (i, p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[keep - 1]
}
