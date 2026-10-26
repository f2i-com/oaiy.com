//! What a request hands a model worker and what comes back: the job, its
//! sampling settings, and the events a reply streams as. Shared by every
//! engine (DeepSeek on CUDA, GGUF through `llm` on CUDA, WebGPU or CPU), so
//! it depends on no GPU crate.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::dsv41::vision::Prepared;

#[derive(Clone, Debug)]
pub struct Sampling {
    /// 0 = greedy.
    pub temperature: f32,
    pub reasoning_repeat_penalty: f32,
    pub reasoning_repeat_last_n: usize,
    /// On the reply's prose (not its thinking, not inside a tool call, where
    /// code rightly repeats): a repeat penalty over the last `repeat_last_n`
    /// tokens (1 = off), and OpenAI's presence and frequency penalties.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
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
    /// Gemma 4's: the picture as its tower takes it (`[3, h, w]`), and its soft tokens.
    Gemma { pixels: ggml_rs::Tensor, tokens: usize },
    /// GLM-5.3 Flash's: the picture as its tower takes it (`[3, height, width]`, normalised), and its tokens.
    Glm { pixels: Vec<f32>, width: usize, height: usize, tokens: usize },
    /// One tile of a MiniCPM-V 4.7 picture (its overview or a slice: `[3, h, w]`, normalised), its patch grid
    /// (`size`: rows, columns) and the patches a side each of its tokens is (`merge`: 4, or 2 in 4x).
    MiniCpmV { pixels: ggml_rs::Tensor, size: (usize, usize), merge: usize },
}
impl ImagePrep {
    pub fn n_tokens(&self) -> usize {
        match self {
            Self::Deepseek(p) => p.n_tokens(),
            Self::Qwen { side, .. } => side * side,
            Self::Gemma { tokens, .. } | Self::Glm { tokens, .. } => *tokens,
            Self::MiniCpmV { size, merge, .. } => crate::llm::vision::mmproj::minicpmv::tile_tokens(*size, *merge),
        }
    }
    pub(crate) fn deepseek(&self) -> oaiy_engine::Result<&Prepared> {
        match self { Self::Deepseek(p) => Ok(p), _ => Err(oaiy_engine::Error::Arg("Qwen image sent to DeepSeek worker".into())) }
    }
}

pub struct Job {
    pub tools: Vec<oaiy_engine::json::Json>,
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
    /// Incognito: keep nothing of this request past [`INCOGNITO_KEPT`]. No prompt
    /// state goes to disk; its live state stays in memory alone for the next
    /// incognito request of its session (of none, where it named none), and is
    /// wiped when any other request comes, its session ends, or it has gone
    /// [`INCOGNITO_KEPT`] unused.
    pub forget: bool,
    /// An incognito session this request belongs to: its prompt state stays in
    /// memory (never on disk) for the session's next request, and is wiped when a
    /// request from anywhere else comes, the session ends, or [`INCOGNITO_KEPT`]
    /// passes unused.
    pub session: Option<String>,
    /// Not a request: wipe the session's state (the end of an incognito session).
    pub wipe: bool,
}

/// How long an incognito request's state is kept in memory unused before it is wiped: an agent's steps, a minute or
/// two apart, read on from the step before (wiped after each request, every step read its whole prompt again: a
/// 2,600-token prompt a minute on DeepSeek-V4.1).
pub const INCOGNITO_KEPT: std::time::Duration = std::time::Duration::from_secs(600);

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

/// Whether `s` asks for any of the reply's penalties ([`penalize_reply`]).
pub fn penalizes(s: &Sampling) -> bool {
    s.repeat_penalty > 1.0 || s.presence_penalty != 0.0 || s.frequency_penalty != 0.0
}

/// The reply's penalties over its recent prose (`history`, its last `repeat_last_n` tokens of it): repeat (sign-aware,
/// once a token), presence (once a token) and frequency (each time). The engines apply them to the reply's prose
/// alone: not its thinking, never inside a tool call, where code rightly repeats.
pub fn penalize_reply(logits: &mut [f32], history: &std::collections::VecDeque<u32>, s: &Sampling) {
    if !penalizes(s) {
        return;
    }
    let mut counts: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for &id in history {
        *counts.entry(id).or_insert(0.0) += 1.0;
    }
    for (id, count) in counts {
        if let Some(value) = logits.get_mut(id as usize) {
            if s.repeat_penalty > 1.0 {
                *value = if *value < 0.0 { *value * s.repeat_penalty } else { *value / s.repeat_penalty };
            }
            *value -= s.presence_penalty + s.frequency_penalty * count;
        }
    }
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


#[cfg(test)]
mod penalty_tests {
    use super::*;

    /// The reply's penalties push its recent tokens down (repeat sign-aware, once a token; presence once; frequency
    /// each time) and are off unless asked for.
    #[test]
    fn reply_penalties_push_recent_tokens_down_and_are_off_by_default() {
        let history = std::collections::VecDeque::from(vec![0, 0, 1]);
        let original = vec![2.0, -2.0, 1.5];
        let base = Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, seed: 0, reasoning_repeat_penalty: 1.0, reasoning_repeat_last_n: 256, repeat_penalty: 1.0, repeat_last_n: 256, presence_penalty: 0.0, frequency_penalty: 0.0 };
        let mut logits = original.clone();
        penalize_reply(&mut logits, &history, &base);
        assert_eq!(logits, original);
        penalize_reply(&mut logits, &history, &Sampling { repeat_penalty: 2.0, ..base.clone() });
        assert_eq!(logits, vec![1.0, -4.0, 1.5]);
        let mut logits = original.clone();
        penalize_reply(&mut logits, &history, &Sampling { presence_penalty: 0.5, frequency_penalty: 0.25, ..base });
        assert_eq!(logits, vec![2.0 - 0.5 - 0.5, -2.0 - 0.5 - 0.25, 1.5]);
    }
}
