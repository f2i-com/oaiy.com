//! the configured models, and swapping between them.
//!
//! The server used to load one model for the life of the process. It can now hold
//! a list and switch on demand, which is what a client asking for a different
//! `model` means: the running engine is dropped — freeing its VRAM and its host
//! expert cache — and the requested one is loaded in its place.
//!
//! # Why dropping is enough
//!
//! Each engine runs on its own thread with `for job in jobs`, so closing the job
//! channel ends the loop, the thread returns, and the engine and its model are
//! dropped with it. That releases the device allocations and the expert cache
//! without this module knowing anything about either. [`Models::activate`] joins
//! the thread before loading the replacement, so the two models are never resident
//! at once — which matters when each is ~190 GB across VRAM, RAM and the drive.
//!
//! # What differs per model, and what does not
//!
//! Very little differs. `api.rs` and `http.rs` never mention an engine: a request
//! becomes a [`Job`] on a channel and the reply is a stream of `Event`s. Of the
//! model-specific surface, `api.rs` touches exactly two things — turning messages
//! into a prompt string, and turning that string into tokens — so [`Flavour`]
//! carries those and everything else is shared, including the SSE plumbing, the
//! sampling defaults, the stop handling, and the `<think>` stream parser (GLM and
//! DeepSeek both use `<think>` / `</think>`, so it needs no variant).

use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use oaiy_engine::{Error, Result};

use crate::engines::job::Job;
use crate::{api, engines::disk, engines::glm, Options, STATE_FORMAT};

pub(crate) fn needs_tool_precision(body: &oaiy_engine::json::Json) -> bool {
    use oaiy_engine::json::Json;
    let nonempty=|v: Option<&Json>|v.and_then(Json::as_array).is_some_and(|a|!a.is_empty());
    (body.get("tool_choice").and_then(Json::as_str)!=Some("none") && nonempty(body.get("tools"))) ||
        body.get("messages").and_then(Json::as_array).is_some_and(|messages|messages.iter().any(|m|
            matches!(m.get("role").and_then(Json::as_str),Some("tool"|"function")) ||
            nonempty(m.get("tool_calls")) || nonempty(m.get("tools"))))
}

/// Which runtime serves a model, decided by what is on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Qwen hybrid with original EXL3 trellis safetensors.
    OrcaSaq,
    /// Qwen3.8-Flash-Next (`qwen4_exp`) with EXL3 trellis safetensors.
    FlashNext,
    /// `dsv41-cuda`: a safetensors checkpoint directory with Engram tables.
    Deepseek,
    /// `llm`: a GGUF file, or a directory holding one.
    Gguf,
}

impl Kind {
    /// Recognize EXL3 Qwen checkpoints by config, GGUF by extension, then
    /// fall back to the existing DeepSeek checkpoint loader.
    pub fn detect(path: &Path) -> Kind {
        if orcasaq_checkpoint(path) { return Kind::OrcaSaq; }
        if flashnext_checkpoint(path) { return Kind::FlashNext; }
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            return Kind::Gguf;
        }
        if path.is_dir() {
            let has_gguf = std::fs::read_dir(path).ok().is_some_and(|d| {
                d.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
                })
            });
            if has_gguf {
                return Kind::Gguf;
            }
        }
        Kind::Deepseek
    }

    /// The first shard of a split GGUF in `dir`, or `dir` itself when it is a file.
    ///
    /// `gguf`'s reader follows `-00002-of-00005` onwards from whichever shard it is
    /// given, so naming the lowest one is enough.
    fn gguf_path(path: &Path) -> Result<PathBuf> {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        let mut shards: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(Error::Io)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
            })
            .collect();
        shards.sort();
        shards.into_iter().next().ok_or_else(|| {
            Error::Arg(format!("no .gguf in {}", path.display()))
        })
    }
}

/// One configured model: the name a client asks for, and where it is.
#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub path: PathBuf,
    pub kind: Kind,
}

/// The model-specific half of turning a request into tokens.
pub enum Flavour {
    Deepseek(Arc<crate::dsv41::tokenizer::Tokenizer>),
    /// GLM-5.3 Flash (and the other streamed MoE GGUFs): the marker a picture stands as in its prompt where it reads
    /// pictures (`<|begin_of_image|><|image|><|end_of_image|>`, with its vision projector).
    Gguf(Arc<tokenizer::Tokenizer>, Option<&'static str>),
    /// Qwen3.5 (and Flash-Next): its pictures as Qwen3.5's own, or as MiniCPM-V 4.7's (a Qwen3.5 model with its
    /// projector).
    Qwen(Arc<tokenizer::Tokenizer>, crate::engines::qwen::Pictures),
    /// LFM2 (`lfm2moe`, served as a Qwen3.5 hybrid): its own template and Python-list calls (`crate::engines::lfm2`).
    Lfm2(Arc<tokenizer::Tokenizer>),
    /// A dense GGUF (Llama, Mistral, Qwen2, Qwen3, Gemma 3 / 3n / 4): its own architecture's template, whether its
    /// own template opens the reply on an empty thought channel (see `dense_prompt`), and the marker a picture
    /// stands as in its prompt where it reads pictures (Gemma 4's `<|image|>`, with its vision projector).
    Dense(Arc<tokenizer::Tokenizer>, crate::llm::Architecture, bool, Option<&'static str>),
}

impl Flavour {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Deepseek(t) => t.encode(text),
            // `llm` applies BOS through the template, so not again here.
            Self::Gguf(t, _) | Self::Qwen(t, _) | Self::Lfm2(t) | Self::Dense(t, ..) => t.encode(text, false).unwrap_or_default(),
        }
    }

    /// The prompt for a chat request, in this model's own template.
    ///
    /// Both templates put the reply inside a `<think>` block the prompt opens, so
    /// the reply parser downstream needs no variant.
    pub fn chat_prompt(
        &self,
        msgs: &[oaiy_engine::json::Json],
        opts: &crate::dsv41::chat::Options,
    ) -> std::result::Result<crate::dsv41::chat::Encoded, String> {
        match self {
            Self::Deepseek(_) => {
                crate::dsv41::chat::encode(msgs, opts).map_err(|e| e.to_string())
            }
            Self::Qwen(_, style) => crate::engines::qwen::chat_prompt_as(msgs, opts, *style),
            Self::Lfm2(_) => crate::engines::lfm2::chat_prompt(msgs, opts),
            Self::Gguf(_, marker) => {
                // The same messages through `llm`'s GLM template. `reasoning
                // effort` maps onto the three the reference Jinja accepts; the
                // DeepSeek options carry a 1-100 scale, so it is banded.
                let effort = match opts.effort {
                    0..=33 => crate::llm::ReasoningEffort::Low,
                    34..=66 => crate::llm::ReasoningEffort::High,
                    _ => crate::llm::ReasoningEffort::Max,
                };
                let (out, images) = match marker {
                    Some(_) => dense_messages(msgs, *marker)?,
                    None => (llama_messages(msgs)?, Vec::new()),
                };
                // The mode decides which generation prompt: `Thinking` opens
                // `<think>` for the model to reason in, `Chat` pre-fills
                // `<think></think>` so it answers directly. Ignoring this was worth
                // the difference between "Paris." and two hundred tokens of
                // deliberation that never finished.
                let thinking = opts.mode == crate::dsv41::chat::Mode::Thinking;
                Ok(crate::dsv41::chat::Encoded {
                    prompt: crate::llm::glm5next_template_full(&out, true, effort, thinking),
                    images,
                })
            }
            Self::Dense(_, arch, empty_thought, marker) => {
                let (messages, images) = dense_messages(msgs, *marker)?;
                Ok(crate::dsv41::chat::Encoded {
                    prompt: dense_prompt(arch, &messages, opts.mode == crate::dsv41::chat::Mode::Thinking, *empty_thought),
                    images,
                })
            }
        }
    }
}

/// A chat request's messages as `llm`'s templates take them: the text of each, by role. Content may be a string
/// or a list of parts (OpenAI's `[{"type":"text","text":…}]`), whose texts are joined as DeepSeek's encoder joins
/// them; these models read no images, so a part that is one is refused rather than dropped.
fn llama_messages(msgs: &[oaiy_engine::json::Json]) -> std::result::Result<Vec<crate::llm::ChatMessage>, String> {
    dense_messages(msgs, None).map(|(m, _)| m)
}

/// [`llama_messages`] for a model that reads pictures where `marker` is its prompt's stand-in for one (Gemma 4's
/// `<|image|>`): a message with pictures is its parts in turn, each text trimmed and each picture its marker, as
/// Gemma 4's own template writes them; the pictures' parts (`{"url": …}`, or the part) in the order they come. A
/// marker written in a message's text is refused (it would stand for a picture nobody sent).
fn dense_messages(msgs: &[oaiy_engine::json::Json], marker: Option<&str>) -> std::result::Result<(Vec<crate::llm::ChatMessage>, Vec<oaiy_engine::json::Json>), String> {
    use oaiy_engine::json::Json;
    let mut images = Vec::new();
    let forged = |s: &str| marker.is_some_and(|m| s.contains(m));
    let messages = msgs.iter()
        .enumerate()
        .map(|(i, m)| {
            let content = match m.get("content") {
                None | Some(Json::Null) => String::new(),
                Some(Json::Str(s)) if forged(s) => return Err(format!("message {}: send pictures as image_url parts, not as {}", i + 1, marker.unwrap_or_default())),
                Some(Json::Str(s)) => s.clone(),
                Some(Json::Arr(parts)) => {
                    let pictures = parts.iter().any(|p| matches!(p.get("type").and_then(Json::as_str), Some("image_url" | "input_image" | "image")));
                    let mut texts = Vec::with_capacity(parts.len());
                    for p in parts {
                        match p.get("type").and_then(Json::as_str) {
                            Some("text" | "input_text") => {
                                let text = p.get("text").and_then(Json::as_str).unwrap_or_default();
                                if forged(text) {
                                    return Err(format!("message {}: send pictures as image_url parts, not as {}", i + 1, marker.unwrap_or_default()));
                                }
                                texts.push(if pictures { text.trim() } else { text });
                            }
                            Some("image_url" | "input_image" | "image") => {
                                let Some(marker) = marker else {
                                    return Err(format!("this model reads text only, and message {} holds an image", i + 1));
                                };
                                images.push(match p.get("image_url") {
                                    Some(url) => Json::obj([("url", url.get("url").unwrap_or(url).clone())]),
                                    None => p.clone(),
                                });
                                texts.push(marker);
                            }
                            other => return Err(format!("message {}: unsupported content part {other:?}", i + 1)),
                        }
                    }
                    texts.join(if pictures { "" } else { "\n\n" })
                }
                Some(_) => return Err(format!("message {}: content must be text or a list of parts", i + 1)),
            };
            let role = match m.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                "system" => crate::llm::Role::System,
                "assistant" => crate::llm::Role::Assistant,
                _ => crate::llm::Role::User,
            };
            Ok(crate::llm::ChatMessage { role, content })
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    Ok((messages, images))
}

/// A dense model's prompt, in its own architecture's template.
///
/// Qwen3 reasons when asked: ChatML with `<think>` opened, as the GLM prompt opens it, so the reply's reasoning is
/// told apart by the same token. Every other dense family, and Qwen3 in a plain chat (its template pre-fills an empty
/// think block, as the official one does), answers directly.
///
/// Gemma 4 reasons in a channel of its own (`<|channel>thought … <channel|>`), which the reasoning split does not
/// read, so it always answers directly: its reply opens on an empty thought channel when its own template opens it so
/// with thinking off (`empty_thought`, read from the GGUF by `opens_empty_thought`). The 26B-A4B and 31B templates do,
/// and without it the 26B-A4B wrote its channel's name into the answer ("thought The capital of France is Paris.");
/// the E2B and E4B templates do not, and with it the E2B reasoned aloud in its answer.
fn dense_prompt(arch: &crate::llm::Architecture, msgs: &[crate::llm::ChatMessage], thinking: bool, empty_thought: bool) -> String {
    if thinking && *arch == crate::llm::Architecture::Qwen3 {
        let mut p = crate::llm::apply_chat_template(&crate::llm::Architecture::Qwen2, msgs, true);
        p.push_str("<think>");
        return p;
    }
    let mut p = crate::llm::apply_chat_template(arch, msgs, true);
    if empty_thought && *arch == crate::llm::Architecture::Gemma4 {
        p.push_str("<|channel>thought\n<channel|>");
    }
    p
}

/// Whether a GGUF's chat template writes an empty Gemma 4 thought channel (as one string, its newline a Jinja escape
/// or a newline): the 26B-A4B's and 31B's do after `<|turn>model`, when thinking is off. The E2B's names the channel
/// only around earlier reasoning (`'<|channel>thought\n' + thinking_text + '\n<channel|>'`), which this does not match.
fn opens_empty_thought(template: &str) -> bool {
    template.contains(r"<|channel>thought\n<channel|>") || template.contains("<|channel>thought\n<channel|>")
}

/// Where a GGUF model's weights are once it is loaded on a GPU, for the log: `on_gpu` bytes there and `left` that had
/// no room within the GPU's share (`WgpuBackend::left_on_host`). Those are multiplied by the CPU straight from the
/// file as it is mapped: nothing of them is copied, the system reads them from the drive as they are used and keeps
/// them in memory while it has room, and where it has not (`free`: what is free now) it reads them again at each
/// token. So a model larger than the computer's memory still answers, at the drive's speed.
#[cfg_attr(not(feature = "webgpu"), allow(dead_code))]
/// What a model whose attention cache is sized by its context is served with when nobody has said: these models
/// name maximums of a quarter of a million tokens, and a cache for that is tens of gigabytes.
pub(crate) const DENSE_CTX: usize = 16384;

/// The tokens a picture is at most for GLM-5.3 Flash (its image processor allows 8,000): each is a token of prompt,
/// and its experts stream from the drive, so a picture costs what that much text does.
const GLM_IMAGE_TOKENS: usize = 576;

/// The soft tokens a picture is at most for Gemma 4 (its image processor's default, of 70, 140, 280, 560 or 1,120):
/// 2,520 patches of 16 pixels, a square picture 768 x 768.
pub(crate) const GEMMA_SOFT_TOKENS: usize = 280;

/// The context of a model whose attention cache is sized by it (Qwen3.5 and 3.8, and the dense families): `ctx`,
/// the server's setting (0: the model's own maximum), never more than `model_max`.
///
/// A number the person gave (`--ctx N`, Engines' "Context") is what they get: the memory it takes is theirs to
/// spend (about 64 KiB a token for a 9B Qwen3.5 and 128 KiB for a 27B, for each copy of the cache, and a GPU's
/// chained runs keep a second). It was held to [`DENSE_CTX`] whatever was asked, so the setting could lower the
/// context and never raise it. With no number given (the default, and `auto`) it is held to [`DENSE_CTX`] as before.
pub(crate) fn dense_context(ctx: usize, asked: bool, model_max: usize) -> usize {
    let most = if ctx == 0 { model_max } else { ctx.min(model_max) };
    if asked && ctx != 0 { most } else { most.min(DENSE_CTX) }
}

/// What a model whose cache is sized by its context is served with when nobody has said and its GPUs have the room
/// for that cache: 65,536 positions (the model's own maximum at most). OAIY's Agent, its instructions and OAIY's
/// tools beside them, takes some 19,000 tokens before a word of the conversation: in [`DENSE_CTX`]'s 16,384 it
/// refused every request, and the person asked for "like 64000 by default".
pub(crate) const AUTO_CTX: usize = 65536;

/// The context nobody gave: from `most` (the model's maximum at most) down in steps of 4,096 to the first that `fits`
/// says the model's GPUs have room for, never below `base` ([`dense_context`]'s), and `base` itself where `fits`
/// cannot say (a device that gives no figure for its memory, the CPU). With whether it could say.
pub(crate) fn auto_context(base: usize, model_max: usize, most: usize, fits: impl Fn(usize) -> Option<bool>) -> (usize, bool) {
    let mut n = most.min(model_max);
    while n > base {
        match fits(n) {
            Some(true) => return (n, true),
            Some(false) => n = n.saturating_sub(4096),
            None => return (base, false),
        }
    }
    (base, n < most.min(model_max))
}

/// a vision tower made on its GPU now and the host's copy of its matrices let go (`MmProj::
/// keep_on_device`: one copy of the tower, the GPU's, not the GPU's and the host's): what the load's line says of it.
pub(crate) fn on_device_only(mm: &mut crate::llm::MmProj) -> String {
    match mm.keep_on_device() {
        0 => String::new(),
        bytes => format!(" (its tower on the GPU alone: {:.1} GB of the host's copy let go)", bytes as f64 / 1e9),
    }
}

pub(crate) fn placement(name: &str, on_gpu: u64, streamed: u64, filed: u64, left: u64, free: Option<u64>) -> String {
    let gb = |bytes: u64| bytes as f64 / 1e9;
    if left == 0 && streamed == 0 && filed == 0 {
        return format!("{name}: its weights are all on the GPU ({:.1} GB)", gb(on_gpu));
    }
    // (past the GPU's share: in the host's memory for the GPU to read over the bus, else in the file and written to
    // the GPU as they are used: `ggml_rs_wgpu::stream`)
    if left == 0 {
        let mut parts = Vec::new();
        if streamed > 0 {
            parts.push(format!("{:.1} GB in this computer's memory, which the GPU reads them from", gb(streamed)));
        }
        if filed > 0 {
            parts.push(format!("{:.1} GB in the model's file, sent to the GPU as they are used (what the system has not kept in memory read from the drive)", gb(filed)));
        }
        return format!(
            "{name}: {:.1} GB of its weights are on the GPU and {:.1} GB had no room there: {} (each of their bytes over the bus once a token, and once a prompt's chunk)",
            gb(on_gpu),
            gb(streamed + filed),
            parts.join(", ")
        );
    }
    let on_gpu = on_gpu + streamed;
    let memory = match free {
        Some(free) if left > free => format!(
            "; {:.1} GB of memory are free for them, so what does not stay there is read from the drive again at each token (slow: a file that fits the GPU's share, or a larger share with --webgpu-gb, answers faster)",
            gb(free)
        ),
        Some(free) => format!("; they fit the {:.1} GB of memory that are free", gb(free)),
        None => String::new(),
    };
    format!(
        "{name}: {:.1} GB of its weights are on the GPU and {:.1} GB had no room there: those run on the CPU straight from the file, which the system reads from the drive as they are used{memory}",
        gb(on_gpu),
        gb(left)
    )
}

#[test]
fn a_models_weights_are_said_to_be_on_the_gpu_or_read_from_the_file() {
    let gb = 1_000_000_000u64;
    assert_eq!(placement("m", 5_700_000_000, 0, 0, 0, Some(9 * gb)), "m: its weights are all on the GPU (5.7 GB)");
    let streamed = placement("m", 10 * gb, 5 * gb, 2 * gb, 0, Some(6 * gb));
    assert!(streamed.starts_with("m: 10.0 GB of its weights are on the GPU and 7.0 GB had no room there: 5.0 GB in this computer's memory") && streamed.contains("2.0 GB in the model's file"), "{streamed}");
    let part = placement("m", 12 * gb, 0, 0, 4 * gb, Some(6 * gb));
    assert!(part.starts_with("m: 12.0 GB of its weights are on the GPU and 4.0 GB had no room there") && part.ends_with("; they fit the 6.0 GB of memory that are free"), "{part}");
    let short = placement("m", 12 * gb, 0, 0, 10 * gb, Some(6 * gb));
    assert!(short.contains("6.0 GB of memory are free for them") && short.contains("read from the drive again at each token"), "{short}");
    assert!(placement("m", 12 * gb, 0, 0, 10 * gb, None).ends_with("as they are used"));
}

/// Where a model's weights are, given what names it: a checkpoint that is a folder (EXL3, safetensors) may be named by
/// the `config.json` in it (a host that takes a model to be one file can then name it), and is that file's folder.
pub(crate) fn checkpoint(path: &Path) -> PathBuf {
    match (path.file_name().and_then(|n| n.to_str()), path.parent()) {
        (Some("config.json"), Some(folder)) if path.is_file() => folder.to_path_buf(),
        _ => path.to_path_buf(),
    }
}

#[test]
fn a_checkpoint_named_by_its_config_is_its_folder() {
    let dir = std::env::temp_dir().join(format!("oaiy-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    assert_eq!(checkpoint(&dir.join("config.json")), dir);
    assert_eq!(checkpoint(&dir), dir);
    // (a file of another name, or a config.json that is not there, is left as it is named)
    assert_eq!(checkpoint(&dir.join("model.gguf")), dir.join("model.gguf"));
    assert_eq!(checkpoint(&dir.join("none").join("config.json")), dir.join("none").join("config.json"));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The architecture of a dense GGUF, which is loaded whole (see `load_gguf`); None for Qwen3.5 (a path of its
/// own), the MoE families expert streaming serves, and what `llm` does not know.
fn dense_arch(gguf: &gguf::GgufFile) -> Option<crate::llm::Architecture> {
    use crate::llm::Architecture as A;
    let name = gguf.get_str("general.architecture").ok()?;
    let experts = gguf.get_u64(&format!("{name}.expert_count")).unwrap_or(0) > 0;
    match A::from_str(name) {
        arch @ (A::Llama | A::Mistral | A::Qwen2) if !experts => Some(arch),
        arch @ (A::Qwen3 | A::Gemma3 | A::Gemma3n | A::Gemma4) => Some(arch),
        _ => None,
    }
}

/// A model that is loaded and serving.
pub struct Active {
    pub jobs: Sender<Job>,
    pub cfg: Arc<api::Config>,
    pub flavour: Arc<Flavour>,
}

struct Live {
    name: String,
    jobs: Sender<Job>,
    thread: JoinHandle<()>,
    cfg: Arc<api::Config>,
    flavour: Arc<Flavour>,
}

/// The configured models, and whichever one is currently loaded.
pub struct Models {
    image_config: Option<crate::media::Config>,
    image_route: Mutex<Option<String>>,
    specs: Vec<Spec>,
    default_name: String,
    opts: Options,
    loopback: bool,
    live: Mutex<Option<Live>>,
    timings: Mutex<std::collections::BTreeMap<String, oaiy_engine::json::Json>>,
}

impl Models {
    pub fn new(opts: Options, loopback: bool) -> Result<Models> {
        let mut image_config = opts.image_config.as_deref().map(crate::media::Config::read).transpose().map_err(Error::Arg)?;
        if let Some(root) = &opts.media_output_root {
            if !root.is_absolute() { return Err(Error::Arg("media output root must be absolute".into())); }
            if let Some(config) = &mut image_config { config.output_root = root.clone(); }
        }
        let first = checkpoint(&opts.model);
        let mut specs = vec![Spec { name: opts.name.clone(), kind: Kind::detect(&first), path: first }];
        for (name, path) in &opts.extra_models {
            if specs.iter().any(|s| &s.name == name) {
                return Err(Error::Arg(format!("two models are both called {name}")));
            }
            let path = checkpoint(path);
            specs.push(Spec { name: name.clone(), kind: Kind::detect(&path), path });
        }
        if let Some(c) = &image_config {
            if let Some(spec) = specs.iter().find(|s|s.name==c.controller_name) {
                if spec.path != c.controller_path {return Err(Error::Arg("image controller name already points at different weights".into()));}
            } else {specs.push(Spec{name:c.controller_name.clone(),path:c.controller_path.clone(),kind:Kind::Gguf});}
        }
        for name in opts.ternary_experts.keys().chain(opts.tool_expert_sources.keys()) {
            if !specs.iter().any(|s| &s.name == name && s.kind == Kind::Deepseek) {
                return Err(Error::Arg(format!("expert source {name} must name a configured DeepSeek model")));
            }
        }
        for name in opts.lora_strengths.keys() {
            if !opts.lora_adapters.contains_key(name) {
                return Err(Error::Arg(format!("LoRA strength for {name}, which has no LoRA")));
            }
        }
        for name in opts.lora_adapters.keys() {
            if !specs.iter().any(|s| &s.name==name && matches!(s.kind, Kind::OrcaSaq | Kind::FlashNext)) {
                return Err(Error::Arg(format!("LoRA {name} must name a configured OrcaSAQ or Flash-Next model")));
            }
        }
        for name in opts.vision_projectors.keys() {
            if !specs.iter().any(|s| &s.name==name && matches!(s.kind,Kind::Gguf | Kind::OrcaSaq | Kind::FlashNext)) {
                return Err(Error::Arg(format!("vision projector {name} must name a configured GGUF, OrcaSAQ or Flash-Next model")));
            }
        }
        for name in opts.tool_expert_sources.keys().chain(opts.tools_experts.iter().map(|_| &opts.name)) {
            if !opts.ternary_experts.contains_key(name) {
                return Err(Error::Arg(format!("tool expert source {name} requires a ternary expert source")));
            }
        }
        Ok(Models {
            image_config,
            image_route: Mutex::new(None),
            default_name: specs[0].name.clone(),
            specs,
            opts,
            loopback,
            live: Mutex::new(None),
            timings: Mutex::new(Default::default()),
        })
    }

    fn expert_sources(&self, name: &str) -> (Option<&Path>, Option<&Path>) {
        let ternary = self.opts.ternary_experts.get(name).map(PathBuf::as_path);
        let tools = self.opts.tool_expert_sources.get(name).map(PathBuf::as_path)
            .or_else(|| if name == self.default_name { self.opts.tools_experts.as_deref() } else { None });
        (ternary, tools)
    }

    pub fn names(&self) -> Vec<String> {
        self.specs.iter().map(|s| s.name.clone()).collect()
    }

    pub fn default_name(&self) -> &str {
        &self.default_name
    }

    /// Measurements survive model unloading; asking for status never loads one.
    pub fn status(&self) -> oaiy_engine::json::Json {
        use oaiy_engine::json::Json;
        let loaded = self.loaded();
        let timings = self.timings.lock().unwrap_or_else(|p|p.into_inner());
        let models = self.specs.iter().map(|s| Json::obj([
            ("model",Json::str(&s.name)),
            ("loaded",Json::Bool(loaded.as_deref()==Some(&s.name))),
            ("vision_configured",Json::Bool(self.opts.vision && (s.kind==Kind::Deepseek || self.opts.vision_projectors.contains_key(&s.name)))),
            ("timing",timings.get(&s.name).cloned().unwrap_or(Json::Null)),
        ])).collect();
        Json::obj([("models",Json::Arr(models)),("resident_model_limit",Json::Int(1)),
            ("routing_note",Json::str("Model names are allowlisted. Changing model unloads the current worker and its GPU state before loading the replacement. Compare load time plus uncached prompt and output time; keep short related actions on the same model. Measurements are observed, not guarantees."))])
    }

    pub fn record_request(&self, name: &str, prompt: usize, cached: usize, output: usize, prefill_secs: f64, decode_secs: f64) {
        use oaiy_engine::json::Json;
        let mut timings=self.timings.lock().unwrap_or_else(|p|p.into_inner());
        let entry=timings.entry(name.into()).or_insert_with(||Json::Obj(vec![]));
        if let Json::Obj(fields)=entry {
            fields.retain(|(k,_)|k=="load_seconds");
            fields.extend([
                ("uncached_prompt_tokens".into(),Json::Int(prompt.saturating_sub(cached) as i64)),
                ("prefill_seconds_including_queue_and_vision".into(),Json::Num(prefill_secs)),
                ("prefill_tokens_per_second".into(),Json::Num(prompt.saturating_sub(cached) as f64/prefill_secs.max(0.001))),
                ("output_tokens".into(),Json::Int(output as i64)),
                ("decode_seconds".into(),Json::Num(decode_secs)),
                ("decode_tokens_per_second".into(),Json::Num(output as f64/decode_secs.max(0.001))),
            ]);
        }
    }

    /// The loaded model's context in tokens, as it was opened.
    pub fn loaded_context(&self) -> Option<(String, usize)> {
        self.live
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|l| (l.name.clone(), l.cfg.max_seq))
    }

    /// Which model is loaded right now, if any.
    pub fn loaded(&self) -> Option<String> {
        self.live
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|l| l.name.clone())
    }

    /// Make `want` the running model and hand back what a request needs.
    ///
    /// Already running: this is a lock and three clones. Otherwise the current
    /// engine is shut down and joined *before* the replacement loads, so the two
    /// models are never resident together.
    pub fn activate(&self, want: Option<&str>) -> std::result::Result<Active, String> {
        if let Some(want)=want {if !self.specs.iter().any(|s|s.name==want){return Err(format!("no configured model called {want}"));}}
        let route = self.image_route.lock().unwrap_or_else(|p|p.into_inner());
        // During image work all chat turns stay on the controller. The media
        // supervisor clears this route after its worker has released memory.
        self.activate_inner(route.as_deref().or(want))
    }

    pub fn image_config(&self) -> Option<&crate::media::Config> { self.image_config.as_ref() }

    pub fn begin_images(&self) -> std::result::Result<(), String> {
        let cfg=self.image_config.as_ref().ok_or("image generation is not configured")?;
        let mut route=self.image_route.lock().unwrap_or_else(|p|p.into_inner());
        // Only publish the route after a successful unload-and-load handoff.
        drop(self.activate_inner(Some(&cfg.controller_name))?);
        *route=Some(cfg.controller_name.clone());
        Ok(())
    }

    pub fn release_images(&self) { *self.image_route.lock().unwrap_or_else(|p|p.into_inner())=None; }

    fn activate_inner(&self, want: Option<&str>) -> std::result::Result<Active, String> {
        let want = want.unwrap_or(&self.default_name);
        let spec = self
            .specs
            .iter()
            .find(|s| s.name == want)
            .ok_or_else(|| {
                format!(
                    "no model called {want}; this server has {}",
                    self.names().join(", ")
                )
            })?
            .clone();

        let mut live = self.live.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(l) = live.as_ref() {
            if l.name == spec.name {
                return Ok(Active {
                    jobs: l.jobs.clone(),
                    cfg: Arc::clone(&l.cfg),
                    flavour: Arc::clone(&l.flavour),
                });
            }
        }

        // Unload first. Dropping the sender ends the engine's `for job in jobs`
        // loop; joining waits for the model to be dropped with the thread, which
        // is what frees the VRAM and the expert cache.
        if let Some(l) = live.take() {
            self.say(format!("unloading {}", l.name));
            drop(l.jobs);
            let _ = l.thread.join();
            self.say(format!("{} unloaded", l.name));
        }

        let t = std::time::Instant::now();
        self.say(format!("loading {} from {}", spec.name, spec.path.display()));
        let next = match spec.kind {
            // OrcaSAQ's EXL3 projections on any GPU through WebGPU (else the CPU).
            #[cfg(feature = "webgpu")]
            Kind::OrcaSaq => self.load_orcasaq_portable(&spec),
            // Flash-Next's EXL3 matrices and experts too.
            #[cfg(feature = "webgpu")]
            Kind::FlashNext => self.load_flashnext_portable(&spec),
            #[cfg(not(feature = "webgpu"))]
            Kind::OrcaSaq | Kind::FlashNext => Err(Error::Arg(format!("{} is an EXL3 checkpoint, which needs the CUDA or the WebGPU build", spec.name))),
            // DeepSeek-V4.1's CPU model with its dense trunk on any GPU through WebGPU (else the CPU).
            #[cfg(feature = "webgpu")]
            Kind::Deepseek => self.load_deepseek_portable(&spec),
            #[cfg(not(feature = "webgpu"))]
            Kind::Deepseek => Err(Error::Arg(format!(
                "{} is a DeepSeek checkpoint, which needs the CUDA or the WebGPU build; this build serves GGUF models",
                spec.name
            ))),
            Kind::Gguf => self.load_gguf(&spec),
        }
        .map_err(|e| format!("loading {}: {e}", spec.name))?;
        {
            use oaiy_engine::json::Json;
            let mut timings=self.timings.lock().unwrap_or_else(|p|p.into_inner());
            let entry=timings.entry(spec.name.clone()).or_insert_with(||Json::Obj(vec![]));
            if let Json::Obj(fields)=entry { fields.retain(|(k,_)| k!="load_seconds"); fields.push(("load_seconds".into(),Json::Num(t.elapsed().as_secs_f64()))); }
        }
        self.say(format!(
            "{} ready in {:.1}s",
            spec.name,
            t.elapsed().as_secs_f64()
        ));

        let active = Active {
            jobs: next.jobs.clone(),
            cfg: Arc::clone(&next.cfg),
            flavour: Arc::clone(&next.flavour),
        };
        *live = Some(next);
        Ok(active)
    }

    /// The context a model is served with: `--ctx`, or under `--ctx auto` (0)
    /// the model's own maximum -- never more than the model allows.
    fn context(&self, model_max: usize) -> usize {
        if self.opts.ctx == 0 { model_max } else { self.opts.ctx.min(model_max) }
    }

    /// The context of a model whose attention cache is sized by it (Qwen3.5 and 3.8, and the dense families).
    fn dense_context(&self, model_max: usize) -> usize {
        dense_context(self.opts.ctx, self.opts.ctx_asked, model_max)
    }

    /// The bytes a GPU keeps free beside a cache the default's context makes (`--vram-headroom-gb`: for what a prompt's
    /// run makes over its chunk's rows, which the cache's count does not hold).
    fn activations_spare(&self) -> u64 {
        (self.opts.headroom_gb.max(0.0) * 1e9) as u64
    }

    /// The context of a model loaded with `base` ([`Self::dense_context`]'s) where nobody gave one: [`auto_context`]
    /// up to `most`, its devices' room for the cache said by `fits` (taken after its weights and its warm-up, before a
    /// request makes any of the larger cache), [`Self::activations_spare`] left beside it. A number the person gave
    /// (`--ctx N`) is theirs whatever the room.
    fn auto_context(&self, name: &str, base: usize, model_max: usize, most: usize, fits: impl Fn(usize) -> Option<bool>) -> usize {
        if self.opts.ctx_asked && self.opts.ctx != 0 {
            return base;
        }
        let (n, said) = auto_context(base, model_max, most, fits);
        if n > base {
            self.say(format!("{name}: its context is {n} positions, what its GPUs have room for its cache in ({} at most where nobody sets one)", most.min(model_max)));
        } else if said {
            self.say(format!("{name}: its GPUs have room for its cache at {base} positions, not more: its context is {base}"));
        }
        n
    }

    fn say(&self, msg: String) {
        if !self.opts.silent {
            eprintln!("{msg}");
        }
    }

    fn base_cfg(&self, spec: &Spec, max_seq: usize) -> api::Config {
        let o = &self.opts;
        api::Config {
            model_name: spec.name.clone(),
            api_key: o.api_key.clone(),
            max_seq,
            thinking: o.thinking,
            effort: o.effort,
            max_tokens: o.max_tokens,
            temperature: o.temperature,
            top_p: o.top_p,
            repeat_penalty: o.repeat_penalty,
            vision: None,
            qwen_vision: None,
            minicpmv: None,
            gemma_vision: None,
            glm_vision: None,
            image_token_id: 0,
            local_images: o.local_images.unwrap_or(self.loopback),
        }
    }

    /// Qwen3.8-Flash-Next without CUDA: its EXL3 matrices and experts on the WebGPU adapter while the weight budget
    /// holds them (the first layers' experts), the rest decoded on the CPU, everything else on the host. No PEFT
    /// adapters and no vision tower (both CUDA's); conversations set aside in host RAM as on CUDA.
    #[cfg(feature = "webgpu")]
    fn load_flashnext_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        // Every adapter for the alias, applied together, each at its strength (else the alias's).
        let default_strength = o.lora_strengths.get(&spec.name).copied().unwrap_or(1.0);
        let base = crate::engines::flashnext::lora_base(&spec.path)?;
        let adapters = o
            .lora_adapters
            .get(&spec.name)
            .map(|list| list.iter().map(|(path, strength)| crate::engines::lora::Adapter::open_for(path, &base).map(|a| a.with_strength(strength.unwrap_or(default_strength)))).collect::<Result<Vec<_>>>())
            .transpose()?
            .unwrap_or_default();
        let devices = self.devices_of(spec);
        let picked = crate::engines::backend::open(o, &devices)?;
        self.say(format!("{} runs on {} (EXL3 experts decoded in the matmul, a layer's in two batches)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        // The other GPUs it may use (the rest of its devices; every other discrete one where none is named) take a
        // share of the layers, a whole layer each (its experts too): two 32 GB cards hold all of its 46 GB of
        // experts, where the CPU decoded what one could not (most of a step).
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = wgpu.map(|b| crate::engines::backend::others(o, &devices, b)).unwrap_or_default().into_iter().map(Arc::new).collect();
        let gpus: Vec<&ggml_rs_wgpu::WgpuBackend> = wgpu.into_iter().chain(others.iter().map(|g| g.as_ref())).collect();
        let backends: Vec<Arc<dyn ggml_rs::Backend>> =
            std::iter::once(Arc::clone(&picked.backend)).chain(others.iter().map(|g| Arc::clone(g) as Arc<dyn ggml_rs::Backend>)).collect();
        type Make<'a> = Box<dyn Fn(ggml_rs::exl3::Exl3Data) -> std::result::Result<Arc<dyn ggml_rs::exl3::PackedLinear>, String> + Send + Sync + 'a>;
        let packed = |device: usize| -> Make<'_> {
            match gpus.get(device).copied() {
                Some(b) => Box::new(move |d| b.exl3(d)),
                None => Box::new(ggml_rs_wgpu::exl3::exl3_cpu),
            }
        };
        // A device's experts take what its layers' attention, delta-net and head matrices leave of its budget.
        let reserve = crate::engines::flashnext::dense_exl3_bytes(&spec.path)? / backends.len() as u64 + (1 << 30);
        let experts = |device: usize, _layer: &str, list: Vec<[ggml_rs::exl3::Exl3Data; 3]>| -> Result<Box<dyn ggml_rs::exl3::Experts>> {
            match gpus.get(device) {
                Some(b) => b.exl3_experts_leaving(list, reserve),
                None => ggml_rs_wgpu::exl3::exl3_experts_cpu(list),
            }
            .map_err(Error::Arg)
        };
        let tower_backend = backends.last().map(Arc::clone);
        // Its routed experts read as records, as rows are routed to them, where the GPUs and the memory that is free
        // hold too little of them to keep them all (a 24 GB card beside a 24 GB Mac: 46.7 GB of them).
        // OAIY_FN_TIERED=1 always, 0 never.
        let expert_bytes = crate::engines::flashnext::routed_expert_bytes(&spec.path)?;
        let gpu_room: u64 = gpus.iter().map(|g| g.usage().1.saturating_sub(reserve)).sum();
        let ram_free = ggml_rs_wgpu::host_memory().map_or(0, |(free, _)| free as u64);
        let tiered = match std::env::var("OAIY_FN_TIERED").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => !gpus.is_empty() && expert_bytes > gpu_room + ram_free / 4 * 3,
        };
        // Each layer's share of what the budget leaves (less the reserve, the attention's cache for the context, and
        // some 1.2 GB for the chain's vectors and a prompt chunk's scratch) as slots of experts the GPU holds, taken in
        // as they are used (OAIY_FN_SLOTS: as many a layer).
        let slots_asked: Option<usize> = std::env::var("OAIY_FN_SLOTS").ok().and_then(|v| v.parse().ok());
        let fc = crate::engines::flashnext::Config::read(&spec.path)?;
        // (the cache for the context where the GPU keeps it, and no more: past `resident_rows` its rows are the host's,
        // and the GPU holds one attention layer's of them at most, for a prompt)
        let resident = self.dense_context(fc.context_length).min(crate::engines::flashnext::resident_rows());
        let cache_bytes = (resident * fc.attention.iter().filter(|&&a| a).count() * (2 * fc.kv_heads * fc.head_dim + fc.index_dim) * 4) as u64;
        let kept_free = reserve + cache_bytes + (12 << 30) / 10;
        let tiered_make = |device: usize, _layer: &str, layout: ggml_rs_wgpu::exl3::RecordLayout, source: Arc<dyn ggml_rs_wgpu::exl3::ExpertSource>, shared: [ggml_rs::exl3::Exl3Data; 3]| -> Result<Box<dyn ggml_rs::exl3::Experts>> {
            let g = gpus.get(device).ok_or_else(|| Error::Arg("Flash-Next's experts read as records need a GPU for the rest of the model".into()))?;
            let layer_bytes = (source.experts() * layout.bytes) as u64;
            let layers = (expert_bytes / layer_bytes.max(1)).max(1) as usize;
            let slot_bytes: usize = layout.proj.iter().map(|p| p.k / 16 * (p.n / 16) * p.tile_words * 2 + (p.k + p.n) * 4).sum();
            let room = g.usage().1.saturating_sub(kept_free);
            let slots = slots_asked.unwrap_or((room / layers as u64 / slot_bytes.max(1) as u64) as usize);
            g.exl3_experts_tiered(layout, source, shared, slots, reserve).map_err(Error::Arg)
        };
        if tiered {
            self.say(format!("{}: its {:.1} GB of routed experts read as records as they are routed to (the GPUs' room {:.1} GB, {:.1} GB of memory free)", spec.name, expert_bytes as f64 / 1e9, gpu_room as f64 / 1e9, ram_free as f64 / 1e9));
        }
        let loaded = if tiered {
            crate::engines::flashnext::load_portable_tiered(&spec.path, backends, &adapters, &packed, &experts, &tiered_make)
        } else {
            crate::engines::flashnext::load_portable_with(&spec.path, backends, &adapters, &packed, &experts, o.mtp.contains(&spec.name))
        };
        let model = loaded.map_err(|e| match adapters.is_empty() {
            true => e,
            // (an adapter's target that no matrix of the model took, or experts that take no update)
            false => Error::Arg(format!("{}: {e} (a LoRA is applied beside the attention, dense and expert projections of the model; this one names something else, or experts that take none)", spec.name)),
        })?;
        for (adapter, (path, strength)) in adapters.iter().zip(o.lora_adapters.get(&spec.name).into_iter().flatten()) {
            self.say(format!("Flash-Next: loaded LoRA {} for {} projections (strength {})", path.display(), adapter.len(), strength.unwrap_or(default_strength)));
        }
        let warm = std::time::Instant::now();
        if model.warm_up() {
            self.say(format!("{}: its chained steps ready in {:.1} s{}", spec.name, warm.elapsed().as_secs_f64(), if model.drafts() { ", drafting with its MTP layer" } else { "" }));
        }
        if o.mtp.contains(&spec.name) && !model.drafts() {
            self.say(format!("{}: its MTP layer is not all on a GPU, so it does not draft", spec.name));
        }
        if tiered {
            let (held, layers, slots, experts) = model.rest_layers();
            self.say(format!("{}: {held} of its {layers} layers' experts partly on the GPU ({slots} slots for its {experts} experts, {} a layer, taken as they are used), the rest read and run on the host", spec.name, slots / layers.max(1)));
        }
        for (d, g) in gpus.iter().enumerate() {
            let (used, budget) = g.usage();
            self.say(format!("{}: {:.1} GB of EXL3 weights on GPU {d} ({} at {}, budget {:.0} GB), the rest on the CPU", spec.name, used as f64 / 1e9, g.adapter().name, g.adapter().pci_bus_id, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer.clone());
        // Bound as the other portable models are, and made longer where nobody set it and the GPUs have room for
        // their copies of the cache (on one GPU at most what it keeps its copy for: past that a request's cache is
        // the host's, and slower, which is not the default's to choose); not where its experts are read as records,
        // whose slots are sized around the cache at `resident`.
        let base = self.dense_context(model.config.context_length);
        let most = if gpus.len() > 1 { AUTO_CTX } else { AUTO_CTX.min(crate::engines::flashnext::resident_rows()) };
        let max_seq = if tiered { base } else { self.auto_context(&spec.name, base, model.config.context_length, most, |n| model.cache_fits(n, self.activations_spare())) };
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Flash-Next tokenizer lacks image_pad".into()))?;
        // Its vision tower, where the alias is given one: on the last GPU (the first carries the head's side of the
        // text model), the matrices the checkpoint stores as EXL3 packed there. The checkpoint's own folder, or a
        // GGUF projector (an mmproj file, as llama.cpp's GGUF of the model comes with: its matrices in f16).
        let vision_path = o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten();
        let projector = match (vision_path, tower_backend) {
            (Some(path), Some(backend)) if path.is_file() => {
                let file = gguf::GgufFile::open(path).map_err(|e| Error::Arg(format!("{}: {e}", path.display())))?;
                let mut mm = crate::llm::MmProj::from_gguf(&file, backend).map_err(|e| Error::Arg(format!("{}: {e}", path.display())))?;
                let width = match &mm {
                    crate::llm::MmProj::Qwen3Vl(q) => Some(q.projector.mm2.shape()[0]),
                    _ => None,
                };
                if width != Some(model.config.hidden) {
                    return Err(Error::Arg(format!(
                        "{}: {} is not a projector for this model (a Qwen vision tower into {} wide, where it {})",
                        spec.name,
                        path.display(),
                        model.config.hidden,
                        width.map_or("is not a Qwen one".to_string(), |w| format!("is into {w}"))
                    )));
                }
                cfg.qwen_vision = Some(mm.config().clone());
                self.say(format!("{}: vision projector loaded from {} (576 tokens an image){}", spec.name, path.display(), on_device_only(&mut mm)));
                Some(mm)
            }
            (Some(path), Some(backend)) => {
                let last = gpus.last().copied();
                let make = |d: ggml_rs::exl3::Exl3Data| match last {
                    Some(g) => g.exl3(d),
                    None => ggml_rs_wgpu::exl3::exl3_cpu(d),
                };
                let mm = crate::engines::qwen::vision::load(path, backend, model.config.hidden, Some(&make))?;
                cfg.qwen_vision = Some(mm.config().clone());
                self.say(format!("{}: Qwen vision tower loaded (576 tokens an image)", spec.name));
                Some(mm)
            }
            _ => None,
        };
        let (jobs, rx) = std::sync::mpsc::channel();
        let park = crate::engines::qwen::park::budget(o.park_gb, ggml_rs_wgpu::host_memory().map(|(free, _)| free as u64));
        let mut e = crate::engines::qwen::QwenEngine::new(crate::engines::qwen::Hybrid::Flash(Box::new(model)), projector, max_seq, !o.quiet && !o.silent).park_up_to(park).repetition_guard(o.repetition_guard);
        // Its prompt states on the disk (a whole state of its cache at a prompt's boundaries): a conversation, or a
        // system prompt, is not read again by the next process to load the model. The states are this checkpoint's
        // under these adapters and no other's.
        if let Some(dir) = &o.prompt_cache {
            let mut fp = disk::fnv(b"flashnext-exl3-state-v1", 0);
            fp = disk::fnv(spec.path.as_os_str().as_encoded_bytes(), fp);
            let mut files: Vec<_> = std::fs::read_dir(&spec.path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors" || x == "json"))
                .collect();
            files.sort();
            for p in files {
                let meta = std::fs::metadata(&p)?;
                fp = disk::fnv(p.as_os_str().as_encoded_bytes(), fp);
                fp = disk::fnv(&meta.len().to_le_bytes(), fp);
            }
            for adapter in &adapters {
                fp = disk::fnv(&adapter.fingerprint.to_le_bytes(), fp);
            }
            match disk::DiskCache::open(dir, fp, (o.prompt_cache_gb * 1e9) as u64) {
                Ok(cache) => e.disk = Some(cache),
                Err(err) => self.say(format!("{}: prompt cache unavailable: {err}", spec.name)),
            }
        }
        let thread = std::thread::Builder::new().name("flashnext-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok, crate::engines::qwen::Pictures::Qwen)) })
    }

    /// DeepSeek-V4.1 without CUDA: the CPU model (`crate::dsv41::model`, the reference the CUDA path is tested against), its
    /// experts streamed through the host cache from RAM and the drive, and its dense trunk (attention projections,
    /// shared experts, router, head: about 9.7 GB) on the WebGPU adapter while the budget holds it. No images and no
    /// observer (both CUDA's); a conversation's next turn continues the state rather than reading it all again.
    #[cfg(feature = "webgpu")]
    fn load_deepseek_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let tok = Arc::new(crate::dsv41::tokenizer::Tokenizer::load(&spec.path)?);
        // As the CUDA build: the server's default context unless one is asked for.
        let max_seq = if o.ctx == 0 { crate::DEFAULT_CTX } else { o.ctx };
        let engram_meta = o.engram_meta.clone().unwrap_or_else(|| spec.path.join("engram_meta.safetensors"));
        let opts = crate::dsv41::model::ModelOptions { max_seq, expert_cache_bytes: o.expert_cache_bytes() as usize, direct_io: true };
        self.say(format!("expert host cache: {:.2} GiB", opts.expert_cache_bytes as f64 / (1u64 << 30) as f64));
        let mut model = crate::dsv41::model::Model::load(&spec.path, &engram_meta, &opts)?;
        // The routed experts on the CPU through this CPU's fastest kernel (the same bits as the portable one).
        model.set_expert_row_kernel(crate::dsv41::simd::row_kernel());
        model.set_expert_tokens_kernel(crate::dsv41::simd::tokens_kernel());
        self.say(format!("experts on the CPU: the {} kernel", crate::dsv41::simd::row_kernel_name()));
        // Its usage profile (`--usage`), if an earlier run left one: the experts' counts of uses, and with them the
        // order the tiers are filled in at this start (the most used on the cards, the next in RAM), where without
        // one they are filled by number and find their order as they are used.
        let order = o.usage.as_deref().filter(|p| crate::engines::dsv41_portable::read_usage(p, model.expert_uses())).map(|p| {
            self.say(format!("the experts' usage profile read from {}: the most used are read first", p.display()));
            model.expert_uses().order()
        });
        let devices = self.devices_of(spec);
        let picked = crate::engines::backend::open(o, &devices)?;
        let mut kernel = None;
        match picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
            Some(b) => {
                let (count, bytes) = crate::engines::dsv41_portable::offload(&mut model, b);
                // A prompt's busy experts there too, through record slots in what the budget has left after the trunk,
                // and the experts used most kept there in what is left after those (a decode step's computed there while
                // the CPU reads and computes the rest).
                // (a card of tinygrad's is the engine's alone: its experts take half of what the default budget leaves
                // for other uses of a card; a budget that was asked for is kept to)
                let more = if o.webgpu_gb.is_none() && b.adapter().pci_bus_id == "tinygpu" { crate::engines::dsv41_portable::ALONE } else { 0 };
                let experts = match crate::engines::dsv41_portable::WgpuExperts::new(b, more) {
                    Some(mut k) => {
                        let (n, kept) = (k.slots(), k.tier().0);
                        // The other GPUs it may use (the rest of its devices; every other discrete one where none is
                        // named; OAIY_NO_SPLIT: none) hold a share of the experts for good.
                        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = crate::engines::backend::others(o, &devices, b).into_iter().map(Arc::new).collect();
                        let pinned = k.pin_on(&others, model.cfg.n_layers, model.cfg.n_routed_experts);
                        // An expert in one place: the 15,360 are 290 GB, and what a card holds RAM need not. The cards
                        // then change what they hold only while the server idles (OAIY_DSV41_INCLUSIVE: as before, the
                        // first card's tier replacing as it goes and RAM holding its experts too).
                        k.set_exclusive(std::env::var_os("OAIY_DSV41_INCLUSIVE").is_none());
                        if let Some(order) = &order {
                            k.start_from(order);
                        }
                        let k = Arc::new(k);
                        model.set_experts_kernel(Some(Arc::clone(&k) as Arc<dyn crate::dsv41::expert::ExpertsKernel>));
                        kernel = Some(k);
                        let share = if pinned > 0 { format!(", {pinned} more on the other GPU{} (read while idle)", if others.len() > 1 { "s" } else { "" }) } else { String::new() };
                        format!("a prompt's busy experts ({n} at a time) and {kept} experts kept there between requests{share}; the rest on the CPU")
                    }
                    None => "no room left there for experts; they run on the CPU".into(),
                };
                self.say(format!("{} runs on {}: {count} dense matrices ({:.1} GB) there, {experts}", spec.name, picked.label, bytes as f64 / 1e9));
            }
            None => self.say(format!("{} runs on the CPU", spec.name)),
        }
        let mut cfg = self.base_cfg(spec, max_seq);
        // Its own placeholder (as the CUDA build): the default, token 0, is DeepSeek's start of sequence.
        cfg.image_token_id = model.cfg.image_token_id;
        let (jobs, rx) = std::sync::mpsc::channel();
        let e = crate::engines::dsv41_portable::Engine::new(model, Arc::clone(&tok), !o.quiet && !o.silent).with_kernel(kernel).with_usage(o.usage.clone(), order).with_ram_ceiling(o.ram_gb);
        let thread = std::thread::Builder::new().name("deepseek-model".into()).spawn(move || e.run(rx)).map_err(Error::Io)?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Deepseek(tok)) })
    }

    /// The GPUs `spec` may use, by number: its own list (`--devices-for`), else the server's (`--devices`); empty
    /// where neither names any (the first adapter, and every other discrete GPU for a model that spreads).
    fn devices_of(&self, spec: &Spec) -> Vec<usize> {
        self.opts.model_devices.get(&spec.name).cloned().unwrap_or_else(|| self.opts.devices.clone())
    }

    /// OrcaSAQ: its packed EXL3 projections on the WebGPU adapter while the weight budget holds them, the rest
    /// decoded on the CPU, everything else on the host as in any portable model; its PEFT adapters beside the
    /// projections they adapt, and its vision tower where it is given one. Its prompt states are kept under a
    /// fingerprint of their own, which its adapters are part of.
    #[cfg(feature = "webgpu")]
    fn load_orcasaq_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        // Every adapter for the alias, applied together, each at its strength (else the alias's).
        let default_strength = o.lora_strengths.get(&spec.name).copied().unwrap_or(1.0);
        let adapters = o
            .lora_adapters
            .get(&spec.name)
            .map(|list| list.iter().map(|(path, strength)| crate::engines::lora::Adapter::open(path).map(|a| a.with_strength(strength.unwrap_or(default_strength)))).collect::<Result<Vec<_>>>())
            .transpose()?
            .unwrap_or_default();
        let picked = crate::engines::backend::open(o, &self.devices_of(spec))?;
        self.say(format!("{} runs on {} (EXL3, decoded in the matmul)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        let packed = |data: ggml_rs::exl3::Exl3Data| match wgpu {
            Some(b) => b.exl3(data),
            None => ggml_rs_wgpu::exl3::exl3_cpu(data),
        };
        let model = crate::engines::orcasaq::load_portable_with(&spec.path, Arc::clone(&picked.backend), &packed, &adapters)?;
        for (adapter, (path, strength)) in adapters.iter().zip(o.lora_adapters.get(&spec.name).into_iter().flatten()) {
            self.say(format!("OrcaSAQ: loaded LoRA {} for {} text projections (strength {})", path.display(), adapter.len(), strength.unwrap_or(default_strength)));
        }
        if let Some((used, budget)) = wgpu.map(|b| b.usage()) {
            self.say(format!("{}: {:.1} GB of EXL3 weights on the GPU (budget {:.0} GB)", spec.name, used as f64 / 1e9, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer().clone());
        // Bound as the other portable models are, and made longer where nobody set it and the GPU has room for the
        // chain's copy of the cache (as the GGUF's: `Qwen35Chain::context_fits`).
        let base = self.dense_context(model.config().context_length);
        let max_seq = match &model {
            crate::llm::Model::Qwen35(m) => self.auto_context(&spec.name, base, model.config().context_length, AUTO_CTX, |n| m.chain.context_fits(m, n, self.activations_spare())),
            _ => base,
        };
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Orca tokenizer lacks image_pad".into()))?;
        // Its vision tower, where the alias is given one (the checkpoint's `vision` folder): on the model's backend.
        let vision_path = o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten();
        let projector = match vision_path {
            Some(path) => {
                let mm = crate::engines::qwen::vision::load(path, Arc::clone(&picked.backend), model.config().embedding_dim, None)?;
                cfg.qwen_vision = Some(mm.config().clone());
                self.say(format!("{}: original Qwen vision tower loaded (576 tokens an image)", spec.name));
                Some(mm)
            }
            None => None,
        };
        let (jobs, rx) = std::sync::mpsc::channel();
        let mut e = crate::engines::qwen::QwenEngine::new(model, projector, max_seq, !o.quiet && !o.silent).repetition_guard(o.repetition_guard);
        e.image_disk_cache = vision_path.is_some();
        if let Some(dir) = &o.prompt_cache {
            let mut fp = disk::fnv(b"orcasaq2-exl3-qwen-state-portable-v1", 0);
            fp = disk::fnv(spec.path.as_os_str().as_encoded_bytes(), fp);
            let mut files: Vec<_> = std::fs::read_dir(&spec.path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors" || x == "json"))
                .collect();
            files.sort();
            for p in files {
                let meta = std::fs::metadata(&p)?;
                fp = disk::fnv(p.as_os_str().as_encoded_bytes(), fp);
                fp = disk::fnv(&meta.len().to_le_bytes(), fp);
            }
            // (a state made under one set of adapters belongs to another model under another)
            for adapter in &adapters {
                fp = disk::fnv(&adapter.fingerprint.to_le_bytes(), fp);
            }
            match disk::DiskCache::open(dir, fp, (o.prompt_cache_gb * 1e9) as u64) {
                Ok(cache) => e.disk = Some(cache),
                Err(err) => self.say(format!("OrcaSAQ prompt cache unavailable: {err}")),
            }
        }
        let thread = std::thread::Builder::new().name("orcasaq-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok, crate::engines::qwen::Pictures::Qwen)) })
    }

    /// Qwen3.8-Flash-Next from a GGUF (`qwen4exp`: the GSQ-RCO files) on WebGPU: its layers over every discrete GPU, a
    /// layer's experts on its device in the file's type while the budget holds them (the rest on the host), the dense
    /// matrices in theirs. No adapters, no vision tower and no multi-token-prediction layer (the GGUFs carry none).
    #[cfg(feature = "webgpu")]
    fn load_flashnext_gguf(&self, spec: &Spec, path: &Path) -> Result<Live> {
        let o = &self.opts;
        if o.lora_adapters.contains_key(&spec.name) {
            return Err(Error::Arg(format!("{}: LoRA adapters are not applied to a GGUF", spec.name)));
        }
        let devices = self.devices_of(spec);
        let picked = crate::engines::backend::open(o, &devices)?;
        self.say(format!("{} runs on {} (a GGUF: each matrix in its file's type)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        // (its other GPUs: the rest of its devices, or every other discrete one where none is named; OAIY_NO_SPLIT:
        // the one GPU, as for the dense models: what does not fit there runs on the host)
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = wgpu.map(|b| crate::engines::backend::others(o, &devices, b)).unwrap_or_default().into_iter().map(Arc::new).collect();
        let gpus: Vec<&ggml_rs_wgpu::WgpuBackend> = wgpu.into_iter().chain(others.iter().map(|g| g.as_ref())).collect();
        let backends: Vec<Arc<dyn ggml_rs::Backend>> =
            std::iter::once(Arc::clone(&picked.backend)).chain(others.iter().map(|g| Arc::clone(g) as Arc<dyn ggml_rs::Backend>)).collect();
        // (its draft layer: the GGUFs carry none, so from the same model's EXL3 checkpoint where one is named)
        let drafting = o.mtp.contains(&spec.name);
        let mtp_from = o.mtp_from.get(&spec.name).filter(|_| drafting);
        if drafting && mtp_from.is_none() {
            self.say(format!("{}: its GGUF has no MTP layer; name the model's EXL3 checkpoint with --mtp-from {}=DIR for it to draft", spec.name, spec.name));
        }
        let model = flashnext_gguf_on(path, &gpus, backends, mtp_from.map(|p| p.as_path()))?;
        let warm = std::time::Instant::now();
        if model.warm_up() {
            self.say(format!("{}: its chained steps ready in {:.1} s{}", spec.name, warm.elapsed().as_secs_f64(), if model.drafts() { ", drafting with its MTP layer" } else { "" }));
        }
        let on_host = model.host_layers();
        if on_host > 0 {
            self.say(format!("{}: {on_host} of its {} layers' experts run on the host (no room on the GPU for them, or no kernel there for their type)", spec.name, model.config.layers));
        }
        if mtp_from.is_some() && !model.drafts() {
            let why = if on_host > 0 { "a check of drafts costs the host's experts more than the steps it saves" } else { "its MTP layer is not all on a GPU" };
            self.say(format!("{}: it does not draft: {why}", spec.name));
        }
        for (d, g) in gpus.iter().enumerate() {
            let (used, budget) = g.usage();
            self.say(format!("{}: {:.1} GB of its weights on GPU {d} ({} at {}, budget {:.0} GB), the rest on the CPU", spec.name, used as f64 / 1e9, g.adapter().name, g.adapter().pci_bus_id, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer.clone());
        // (made longer where nobody set it and the GPUs have room, as the EXL3 checkpoint's: `load_flashnext_portable`)
        let base = self.dense_context(model.config.context_length);
        let most = if gpus.len() > 1 { AUTO_CTX } else { AUTO_CTX.min(crate::engines::flashnext::resident_rows()) };
        let max_seq = self.auto_context(&spec.name, base, model.config.context_length, most, |n| model.cache_fits(n, self.activations_spare()));
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Flash-Next tokenizer lacks image_pad".into()))?;
        let (jobs, rx) = std::sync::mpsc::channel();
        let park = crate::engines::qwen::park::budget(o.park_gb, ggml_rs_wgpu::host_memory().map(|(free, _)| free as u64));
        let e = crate::engines::qwen::QwenEngine::new(crate::engines::qwen::Hybrid::Flash(Box::new(model)), None, max_seq, !o.quiet && !o.silent).park_up_to(park).repetition_guard(o.repetition_guard);
        let thread = std::thread::Builder::new().name("flashnext-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok, crate::engines::qwen::Pictures::Qwen)) })
    }

    /// Says where a GGUF model's weights are once it is loaded on a GPU ([`placement`]).
    fn say_placement(&self, name: &str, backend: &dyn ggml_rs::Backend) {
        #[cfg(feature = "webgpu")]
        if let Some(b) = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
            self.say(placement(name, b.usage().0, b.streamed(), b.filed(), b.left_on_host(), ggml_rs_wgpu::host_memory().map(|(free, _)| free as u64)));
        }
        #[cfg(not(feature = "webgpu"))]
        let _ = (name, backend);
    }

    // ---------------------------------------------------------------- GGUF
    /// A Qwen3.5-MoE GGUF (`qwen35moe`), or an LFM2-MoE one (`lfm2moe`), as a Qwen3.5 hybrid with routed experts: each layer's experts held by the GPU's
    /// kernels (`QuantMoe`: the K-quants, the GSQ-RCO types) where its budget has room, else run on the host; a chained
    /// run's pieces 64 rows at most where the GPU groups no more of a prompt's rows by expert (no tensor cores).
    fn qwen35_moe(&self, name: &str, gguf: &gguf::GgufFile, backend: &Arc<dyn ggml_rs::Backend>) -> Result<crate::llm::Model> {
        #[cfg(feature = "webgpu")]
        {
            let b = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>().ok_or_else(|| Error::Arg(format!("{name}: a MoE model's experts need a GPU (WebGPU)")))?;
            let on_host = std::sync::atomic::AtomicUsize::new(0);
            let make = |_: usize, e: crate::llm::ExpertBlocks| -> crate::llm::Result<Box<dyn ggml_rs::exl3::Experts>> {
                let data = ggml_rs_wgpu::quant_moe::QuantExpertsData { hidden: e.hidden, ff: e.ff, experts: e.experts, gate: e.gate, up: e.up, down: e.down, shared: e.shared, routing: e.routing };
                let made = b.quant_experts(data).map_err(crate::llm::LlamaError::Config)?;
                if made.on_host() {
                    on_host.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(made)
            };
            let mut m = crate::llm::Qwen35Model::from_gguf_with(gguf, Arc::clone(backend), Some(&make)).map_err(|e| Error::Arg(e.to_string()))?;
            let cores = b.f16_tensor_cores() && std::env::var_os("OAIY_QUANT_MOE_NO_COOP").is_none();
            m.moe_rows = if cores { 0 } else { 64 };
            let host = on_host.into_inner();
            if host > 0 {
                self.say(format!("{name}: {host} of its {} layers' experts run on the host (no room for them on the GPU)", m.blocks.len()));
            }
            Ok(crate::llm::Model::Qwen35(m))
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (gguf, backend);
            Err(Error::Arg(format!("{name}: a Qwen3.5-MoE model's experts need a GPU (the webgpu build)")))
        }
    }

    fn load_gguf(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let path = Kind::gguf_path(&spec.path)?;
        #[cfg(feature = "webgpu")]
        if crate::engines::flashnext::gguf_file::detect(&path) {
            return self.load_flashnext_gguf(spec, &path);
        }
        // The GPUs it may use: an image model's controller the one its configuration names, else the model's own
        // list or the server's; the first carries it.
        let devices = self.image_config.as_ref().filter(|c| c.controller_name == spec.name).map(|c| vec![c.controller_device]).unwrap_or_else(|| self.devices_of(spec));
        let picked = crate::engines::backend::open(o, &devices)?;
        self.say(format!("{} runs on {}", spec.name, picked.label));
        let backend: Arc<dyn ggml_rs::Backend> = Arc::clone(&picked.backend);
        let gguf = gguf::GgufFile::open(&path).map_err(|e| Error::Arg(e.to_string()))?;
        // (Qwen3.5-MoE, `qwen35moe`: Qwen3.6-35B-A3B's and MiniCPM-V 4.7 35B-A3B's architecture, the Qwen3.5 hybrid's
        // layers with routed experts, which a GPU's kernels hold; and LFM2-MoE, `lfm2moe`, a hybrid of short convs and
        // attention with experts, served as one: LFM2.5-24B-A2B's)
        let lfm2 = gguf.get_str("general.architecture").ok() == Some("lfm2moe");
        let moe = gguf.get_str("general.architecture").ok() == Some("qwen35moe") || lfm2;
        if gguf.get_str("general.architecture").ok() == Some("qwen35") || moe {
            // With a second GPU, a prompt's later layers go to it (copies of their weights and the head): each
            // chunk's first layers run on one as the other runs the chunk before's. The card the host's bytes reach
            // faster carries the model (it moves the more: the cache's rows, the steps), the other the later layers.
            // (Not a MoE model's: its experts are its card's.)
            #[allow(unused_mut)]
            let (mut backend, mut helper): (Arc<dyn ggml_rs::Backend>, Option<Arc<dyn ggml_rs::Backend>>) = (backend, None);
            #[cfg(feature = "webgpu")]
            if let Some(b) = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>().filter(|_| !moe) {
                if let Some(other) = crate::engines::backend::others(o, &devices, b).into_iter().next() {
                    let (mine, theirs) = (b.transfer_rates().0, other.transfer_rates().0);
                    let (bus, other_bus) = (b.adapter().pci_bus_id.clone(), other.adapter().pci_bus_id.clone());
                    let other: Arc<dyn ggml_rs::Backend> = Arc::new(other);
                    let swap = theirs > mine * 1.2;
                    let (main_at, main_rate, helper_at, helper_rate) = if swap { (other_bus, theirs, bus, mine) } else { (bus, mine, other_bus, theirs) };
                    self.say(format!("{}: the GPU at {main_at} carries it ({main_rate:.0} GB/s from the host), the one at {helper_at} a prompt's later layers ({helper_rate:.0} GB/s)", spec.name));
                    helper = Some(if swap { std::mem::replace(&mut backend, other) } else { other });
                }
            }
            let mut model = if moe { self.qwen35_moe(&spec.name, &gguf, &backend)? } else { crate::llm::Model::load(&gguf, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))? };
            // Its multi-token-prediction layer, where asked for: it drafts tokens a chained run checks.
            if o.mtp.contains(&spec.name) {
                if let crate::llm::Model::Qwen35(m) = &mut model {
                    match m.load_mtp(&gguf) {
                        Ok(Some(layer)) => {
                            m.mtp = Some(layer);
                            self.say(format!("{}: drafts tokens with its multi-token-prediction layer (--mtp)", spec.name));
                        }
                        Ok(None) => self.say(format!("{}: --mtp asked for, but its GGUF has no multi-token-prediction layer", spec.name)),
                        Err(e) => self.say(format!("{}: its multi-token-prediction layer did not load ({e})", spec.name)),
                    }
                }
            }
            if let (crate::llm::Model::Qwen35(m), Some(h)) = (&model, helper) {
                m.chain.split_onto(h);
            }
            // its chained runs' kernels compiled before the first request waits on them
            if let crate::llm::Model::Qwen35(m) = &model {
                let warm = std::time::Instant::now();
                if m.warm_up() {
                    self.say(format!("{}: its chained runs ready in {:.1} s", spec.name, warm.elapsed().as_secs_f64()));
                }
            }
            self.say_placement(&spec.name, backend.as_ref());
            let tok = Arc::new(model.tokenizer().clone());
            // Dense Qwen fits one card; bound the initial KV allocation. On a card that says what it holds only by
            // refusing what it has no room for (tinygrad's), the context is what the chain's cache has room for there
            // beside a prompt's work: a longer prompt is then refused as one past any context is, where it failed
            // halfway through for want of room (and OAIY answered it with the Mac's copy of the model).
            // (its vision tower first, where it has one: on the card it is 1.1 GB of the room the cache is measured in,
            // so a Mac runs it on its own GPU: `backend::for_vision`)
            let projector = if o.vision {
                o.vision_projectors.get(&spec.name).map(|p| {
                    let file = gguf::GgufFile::open(p).map_err(|e| Error::Arg(e.to_string()))?;
                    let (tower, said) = crate::engines::backend::for_vision(&backend);
                    if let Some(said) = said { self.say(format!("{}: {said}", spec.name)); }
                    let mut mm = crate::llm::MmProj::from_gguf(&file, tower).map_err(|e| Error::Arg(e.to_string()))?;
                    let width = model.config().embedding_dim;
                    // (Qwen3-VL's tower, or MiniCPM-V 4.7's: a Qwen3.5 model of its own)
                    if !matches!(&mm, crate::llm::MmProj::Qwen3Vl(q) if q.projector.mm2.shape()[0] == width)
                        && !matches!(&mm, crate::llm::MmProj::MiniCpmV(m) if m.lm_width() == width) {
                        return Err(Error::Arg("Qwen vision projector does not match the text model".into()));
                    }
                    self.say(format!("{}: vision projector loaded from {}{}", spec.name, p.display(), on_device_only(&mut mm)));
                    Ok(mm)
                }).transpose()?
            } else { None };
            let mut max_seq = self.dense_context(model.config().context_length);
            if let crate::llm::Model::Qwen35(m) = &model {
                // (longer where nobody set it and a GPU that says what it holds has the room: `context_fits`; a card
                // that says so only by refusing measures it below, for the context as it is)
                max_seq = self.auto_context(&spec.name, max_seq, model.config().context_length, AUTO_CTX, |n| m.chain.context_fits(m, n, self.activations_spare()));
                match m.chain.context_room(m, max_seq) {
                    Some(0) => self.say(format!("{}: the GPU has no room for its attention cache at 4,096 positions beside its weights", spec.name)),
                    Some(room) if room < max_seq => {
                        self.say(format!("{}: the GPU holds its attention cache for {room} positions beside its weights and a prompt's work: its context here is {room}, not {max_seq}", spec.name));
                        max_seq = room;
                    }
                    Some(_) => self.say(format!("{}: the GPU holds its attention cache for all {max_seq} positions", spec.name)),
                    None => {}
                }
            }
            let mut cfg = self.base_cfg(spec, max_seq);
            // (LFM2 reads no pictures: its own picture token stands in, never met)
            cfg.image_token_id = tok.token_id("<|image_pad|>").or_else(|| lfm2.then(|| tok.token_id("<image>")).flatten()).ok_or_else(|| Error::Arg("Qwen tokenizer lacks image_pad".into()))?;
            cfg.qwen_vision = projector.as_ref().map(|p|p.config().clone());
            // MiniCPM-V 4.7: its markers' ids from its own tokenizer
            if let Some(crate::llm::MmProj::MiniCpmV(m)) = &projector {
                let id = |s: &str| tok.token_id(s).ok_or_else(|| Error::Arg(format!("MiniCPM-V's tokenizer lacks {s}")));
                let newline = match tok.encode("\n", false).unwrap_or_default().as_slice() {
                    &[t] => t,
                    other => return Err(Error::Arg(format!("MiniCPM-V's tokenizer makes a newline {other:?}"))),
                };
                let markers = crate::llm::vision::mmproj::minicpmv::Markers { image_start: id("<image>")?, image_end: id("</image>")?, slice_start: id("<slice>")?, slice_end: id("</slice>")?, newline };
                cfg.minicpmv = Some(crate::engines::qwen::MiniCpmV { markers, merge: m.merge, max_slices: m.max_slices, mean: m.config.mean, std: m.config.std });
                self.say(format!("{}: MiniCPM-V 4.7 pictures ({}x: up to {} slices)", spec.name, m.merge * m.merge, m.max_slices));
            }
            let pictures = if cfg.minicpmv.is_some() { crate::engines::qwen::Pictures::MiniCpmV } else { crate::engines::qwen::Pictures::Qwen };
            let (jobs,rx) = std::sync::mpsc::channel();
            let calls = if lfm2 { crate::engines::qwen::Calls::Lfm2 } else { crate::engines::qwen::Calls::Qwen };
            let mut e = crate::engines::qwen::QwenEngine::new(model,projector,max_seq,!o.quiet && !o.silent).repetition_guard(o.repetition_guard).minicpmv(cfg.minicpmv.clone()).calls(calls);
            if let Some(dir) = &o.prompt_cache {
                // Separate format/implementation namespace; replacing weights
                // in place also invalidates states via size and modification time.
                let mut fingerprint = disk::fnv(b"qwen35-hybrid-state-v1", 0);
                fingerprint = disk::fnv(path.as_os_str().as_encoded_bytes(), fingerprint);
                let metadata = std::fs::metadata(&path)?;
                fingerprint = disk::fnv(&metadata.len().to_le_bytes(), fingerprint);
                if let Ok(modified) = metadata.modified().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)) {
                    fingerprint = disk::fnv(&modified.as_nanos().to_le_bytes(), fingerprint);
                }
                match disk::DiskCache::open(dir, fingerprint, (o.prompt_cache_gb * 1e9) as u64) {
                    Ok(cache) => {
                        self.say(format!("{} Qwen prompt states on disk in {}", cache.len(), dir.display()));
                        e.disk = Some(cache);
                    }
                    Err(err) => self.say(format!("Qwen prompt states are not kept ({}: {err})", dir.display())),
                }
            }
            let thread = std::thread::Builder::new().name("qwen-model".into()).spawn(move || e.run(rx)).map_err(Error::Io)?;
            let flavour = if lfm2 { Flavour::Lfm2(tok) } else { Flavour::Qwen(tok, pictures) };
            return Ok(Live {name:spec.name.clone(),jobs,thread,cfg:Arc::new(cfg),flavour:Arc::new(flavour)});
        }
        // Dense models (Llama, Mistral and Qwen2 without experts, Qwen3, Gemma 3, 3n and 4) are loaded whole onto
        // the backend, as Qwen3.5 is above, and answer in their own template with their own stop tokens. Expert
        // streaming below is for the MoE families only and refused every dense model, so a Llama or Gemma GGUF (most of
        // what people keep, and most of the catalog) could not be served at all. They chat; tool calls are read in the
        // GLM and Qwen3.5 formats only, so the Agent's tools need one of those.
        if let Some(arch) = dense_arch(&gguf) {
            let model = crate::llm::Model::load(&gguf, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))?;
            self.say_placement(&spec.name, backend.as_ref());
            let empty_thought = gguf.get_str("tokenizer.chat_template").is_ok_and(opens_empty_thought);
            drop(gguf);
            let tok = Arc::new(model.tokenizer().clone());
            // Bound the initial KV allocation, as for dense Qwen3.5: a full context of an 8B model is gigabytes.
            let max_seq = self.dense_context(model.config().context_length);
            let mut cfg = self.base_cfg(spec, max_seq);
            // Gemma 4 reads pictures through its mmproj's vision tower (on the GPU); another dense model's projector
            // is not read.
            let projector = match o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten() {
                Some(p) if arch == crate::llm::Architecture::Gemma4 => {
                    let file = gguf::GgufFile::open(p).map_err(|e| Error::Arg(format!("{}: {e}", p.display())))?;
                    let mm = crate::llm::MmProj::from_gguf(&file, Arc::clone(&backend)).map_err(|e| Error::Arg(format!("{}: {e}", p.display())))?;
                    let crate::llm::MmProj::Gemma4V(g) = &mm else {
                        return Err(Error::Arg(format!("{}: not a Gemma 4 vision projector", p.display())));
                    };
                    if g.projector.input_projection.shape()[0] != model.config().embedding_dim {
                        return Err(Error::Arg(format!("{}: its vision tower is another Gemma 4's (its width is not this model's)", p.display())));
                    }
                    let ids = ["<|image|>", "<|image>", "<image|>"].map(|t| tok.token_id(t));
                    let [Some(image), Some(open), Some(close)] = ids else {
                        return Err(Error::Arg("this Gemma 4's tokenizer lacks its picture tokens".into()));
                    };
                    cfg.image_token_id = image;
                    cfg.gemma_vision = Some(api::GemmaVision { patch: mm.config().patch_size, max_soft_tokens: GEMMA_SOFT_TOKENS, open, close });
                    self.say(format!("{}: vision projector loaded from {}", spec.name, p.display()));
                    Some(mm)
                }
                Some(p) => {
                    self.say(format!("{}: its vision projector ({}) is not read: of the dense models only Gemma 4 reads pictures here", spec.name, p.display()));
                    None
                }
                None => None,
            };
            let marker = projector.is_some().then_some("<|image|>");
            let (jobs, rx) = std::sync::mpsc::channel();
            let e = glm::GlmEngine::new(model, Arc::clone(&tok), max_seq, !o.quiet && !o.silent).with_vision(projector);
            let thread = std::thread::Builder::new().name("model".into()).spawn(move || e.run(rx)).map_err(Error::Io)?;
            return Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Dense(tok, arch, empty_thought, marker)) });
        }
        drop(gguf);
        let budget = o.expert_cache_bytes();
        if o.ram_gb == 0 {
            self.say(format!(
                "expert cache: {:.0} GB of host RAM (80% of what is free; --ram-gb pins it)",
                budget as f64 / 1e9
            ));
        }
        #[allow(unused_mut)]
        let mut model = crate::llm::Model::open_streaming(&path, backend, budget)
            .map_err(|e| Error::Arg(e.to_string()))?;

        let model = model;
        let tok = Arc::new(model.tokenizer().clone());

        // The state the model was opened with bounds the context, whatever was
        // asked for.
        let max_seq = match &model {
            crate::llm::Model::Glm5Next(g) => self.context(g.max_len()),
            _ => self.context(model.config().context_length),
        };
        let mut cfg = self.base_cfg(spec, max_seq);
        // GLM-5.3 Flash reads pictures through its mmproj's vision tower (on the host, a picture's tokens then spliced
        // in where its `<|image|>` tokens stand); another streamed model's projector is not read.
        let mut marker = None;
        let mut tower = None;
        if let Some(p) = o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten() {
            if matches!(model, crate::llm::Model::Glm5Next(_)) {
                let t = std::time::Instant::now();
                let (shape, weights) = crate::llm::arch::glm5next::mmproj::VisionWeights::open(p).map_err(|e| Error::Arg(format!("{}: {e}", p.display())))?;
                if shape.proj_dim != model.config().embedding_dim {
                    return Err(Error::Arg(format!("{}: its vision tower gives {}-wide tokens, not this model's {}", p.display(), shape.proj_dim, model.config().embedding_dim)));
                }
                let ids = ["<|image|>", "<|begin_of_image|>", "<|end_of_image|>"].map(|t| tok.token_id(t));
                let [Some(image), Some(_), Some(_)] = ids else {
                    return Err(Error::Arg("this GLM's tokenizer lacks its picture tokens".into()));
                };
                cfg.image_token_id = image;
                cfg.glm_vision = Some(api::GlmVision { shape: shape.clone(), max_tokens: GLM_IMAGE_TOKENS });
                marker = Some("<|begin_of_image|><|image|><|end_of_image|>");
                self.say(format!("{}: vision projector loaded from {} in {:.1}s ({:.1} GB; at most {GLM_IMAGE_TOKENS} tokens a picture)", spec.name, p.display(), t.elapsed().as_secs_f64(), weights.size_bytes() as f64 / 1e9));
                tower = Some(Arc::new(glm::GlmTower { shape, weights }));
            } else {
                self.say(format!("{}: its vision projector ({}) is not read: of the models whose experts stream from their file only GLM-5.3 Flash reads pictures here", spec.name, p.display()));
            }
        }

        let (jobs, rx) = std::sync::mpsc::channel();
        let mut e = glm::GlmEngine::new(model, Arc::clone(&tok), max_seq, !o.quiet && !o.silent).with_glm_vision(tower);
        // GLM-5.3-Flash. Prompt states on disk, as DeepSeek has.
        //
        // This matters more here than it does there: MLA's attention is O(n) a
        // token, so a 1740-token prompt measured at 585 s, and a harness sends the
        // same system prompt on every request. Without this a daemon re-reads it
        // from nothing every time it starts.
        //
        // The fingerprint is what stops a state being restored into weights it did
        // not come from. DeepSeek hashes config.json and the weight index; a GGUF
        // has neither, and hashing 193 GB of shards is not an option, so it is the
        // shard names and sizes plus the state format -- which moves if a shard is
        // replaced or the file re-quantised, and is cheap to compute.
        if let Some(dir) = &o.prompt_cache {
            let mut fingerprint = disk::fnv(&[STATE_FORMAT], 0);
            fingerprint = disk::fnv(path.as_os_str().as_encoded_bytes(), fingerprint);
            if let Ok(entries) = std::fs::read_dir(path.parent().unwrap_or(&path)) {
                let mut shards: Vec<(String, u64)> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().is_some_and(|x| x == "gguf"))
                    .filter_map(|e| {
                        let len = e.metadata().ok()?.len();
                        Some((e.file_name().to_string_lossy().into_owned(), len))
                    })
                    .collect();
                shards.sort();
                for (name, len) in shards {
                    fingerprint = disk::fnv(name.as_bytes(), fingerprint);
                    fingerprint = disk::fnv(&len.to_le_bytes(), fingerprint);
                }
            }
            match disk::DiskCache::open(dir, fingerprint, (o.prompt_cache_gb * 1e9) as u64) {
                Ok(cache) => {
                    self.say(format!(
                        "{} prompt states on disk in {}",
                        cache.len(),
                        dir.display()
                    ));
                    e.disk = Some(cache);
                }
                Err(err) => {
                    self.say(format!("prompt states are not kept ({}: {err})", dir.display()))
                }
            }
        }
        let thread = std::thread::Builder::new()
            .name("model".into())
            .spawn(move || e.run(rx))
            .map_err(Error::Io)?;

        Ok(Live {
            name: spec.name.clone(),
            jobs,
            thread,
            cfg: Arc::new(cfg),
            flavour: Arc::new(Flavour::Gguf(tok, marker)),
        })
    }
}

/// A hybrid model starts with its ternary expert bank selected. Its Q4 bank
/// can still fill on demand, but the usage profile should warm the bank that
/// serves prompt and prose turns immediately after startup.
/// An OrcaSAQ checkpoint: Qwen3.5 quantized to EXL3 (see `orcasaq`). Read here
/// rather than there so a build without CUDA still recognises one and says why
/// it cannot serve it.
fn orcasaq_checkpoint(path: &Path) -> bool {
    std::fs::read(path.join("config.json")).ok().and_then(|b| oaiy_engine::json::Json::parse(&b).ok()).is_some_and(|c| {
        c.get("model_type").and_then(oaiy_engine::json::Json::as_str) == Some("qwen3_5")
            && c.get("quantization_config")
                .and_then(|q| q.get("quant_method"))
                .and_then(oaiy_engine::json::Json::as_str)
                == Some("exl3")
    })
}

/// A Qwen3.8-Flash-Next checkpoint (`qwen4_exp`, EXL3). Read here so a build without CUDA
/// still recognises one and says why it cannot serve it.
fn flashnext_checkpoint(path: &Path) -> bool {
    std::fs::read(path.join("config.json")).ok().and_then(|b| oaiy_engine::json::Json::parse(&b).ok()).is_some_and(|c| {
        c.get("model_type").and_then(oaiy_engine::json::Json::as_str) == Some("qwen4_exp")
            && c.get("quantization_config")
                .and_then(|q| q.get("quant_method"))
                .and_then(oaiy_engine::json::Json::as_str)
                == Some("exl3")
    })
}

fn startup_warm_profile(usage: Option<&Path>, tool_experts: bool) -> Option<(&Path, &'static str)> {
    usage.filter(|p| p.is_file())
        .map(|p| (p, if tool_experts { "ternary" } else { "model" }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_context_someone_asked_for_is_served_and_the_default_is_as_it_was() {
        const QWEN: usize = 262_144; // what a Qwen3.5 file names as its maximum
        // Nobody said (the binary's default, and `--ctx auto`): held to DENSE_CTX, as before.
        assert_eq!(dense_context(crate::DEFAULT_CTX, false, QWEN), DENSE_CTX);
        assert_eq!(dense_context(0, false, QWEN), DENSE_CTX);
        assert_eq!(dense_context(0, true, QWEN), DENSE_CTX, "auto is nobody's number, however it was written");
        // A number the person gave is served: more than the default, or less.
        assert_eq!(dense_context(64_000, true, QWEN), 64_000);
        assert_eq!(dense_context(65_536, true, QWEN), 65_536);
        assert_eq!(dense_context(4096, true, QWEN), 4096);
        assert_eq!(dense_context(4096, false, QWEN), 4096, "a smaller default is still the bound");
        // Never more than the model allows, asked or not.
        assert_eq!(dense_context(64_000, true, 8192), 8192);
        assert_eq!(dense_context(0, false, 8192), 8192);
    }

    #[test]
    fn the_context_nobody_set_is_as_long_as_the_gpus_have_room_for() {
        const QWEN: usize = 262_144;
        // Room for all of it: the default's most, the model's maximum at most.
        assert_eq!(auto_context(DENSE_CTX, QWEN, AUTO_CTX, |_| Some(true)), (AUTO_CTX, true));
        assert_eq!(auto_context(DENSE_CTX, 40_960, AUTO_CTX, |_| Some(true)), (40_960, true));
        // Room for less: down in steps of 4,096 to what fits.
        let asked = std::cell::RefCell::new(Vec::new());
        let fits = |n: usize| {
            asked.borrow_mut().push(n);
            Some(n <= 50_000)
        };
        assert_eq!(auto_context(DENSE_CTX, QWEN, AUTO_CTX, fits), (49_152, true));
        assert_eq!(asked.borrow().as_slice(), &[65_536, 61_440, 57_344, 53_248, 49_152]);
        // Room for none of it past the old default: the old default, said.
        assert_eq!(auto_context(DENSE_CTX, QWEN, AUTO_CTX, |_| Some(false)), (DENSE_CTX, true));
        // Devices that say nothing of their memory (the CPU, a card that only refuses): the old default, as before.
        assert_eq!(auto_context(DENSE_CTX, QWEN, AUTO_CTX, |_| None), (DENSE_CTX, false));
        // A most of its own (one GPU's copy of Flash-Next's cache), and a model whose maximum is the old default's or less.
        assert_eq!(auto_context(DENSE_CTX, QWEN, 32_768, |_| Some(true)), (32_768, true));
        assert_eq!(auto_context(8192, 8192, AUTO_CTX, |_| Some(true)), (8192, false));
        assert_eq!(auto_context(DENSE_CTX, DENSE_CTX, AUTO_CTX, |_| Some(true)), (DENSE_CTX, false));
    }

    #[test]
    fn orcasaq_vision_configuration_is_accepted_and_scoped() {
        let root=std::env::temp_dir().join(format!("oaiy-orca-vision-config-{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"),br#"{"model_type":"qwen3_5","quantization_config":{"quant_method":"exl3"}}"#).unwrap();
        let mut o=Options::default();
        o.name="orca".into(); o.model=root.clone();
        o.vision_projectors.insert("orca".into(),root.join("vision"));
        let models=Models::new(o.clone(),true).unwrap();
        assert_eq!(models.specs[0].kind,Kind::OrcaSaq);
        o.vision_projectors.insert("unknown".into(),root.join("other"));
        assert!(Models::new(o,true).is_err());
        std::fs::remove_file(root.join("config.json")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn hybrid_startup_warms_its_ternary_bank_from_an_existing_profile() {
        let path = std::env::temp_dir().join(format!("oaiy-warm-{}.txt", std::process::id()));
        std::fs::write(&path, b"# profile\n20 1 5\n").unwrap();
        assert_eq!(startup_warm_profile(Some(&path), true), Some((path.as_path(), "ternary")));
        assert_eq!(startup_warm_profile(Some(&path), false), Some((path.as_path(), "model")));
        std::fs::remove_file(&path).unwrap();
        assert!(startup_warm_profile(Some(&path), true).is_none());
    }

    #[test]
    fn expert_sources_are_scoped_to_the_requested_model() {
        let mut o = Options::default();
        o.extra_models = vec![("original".into(), "source".into()), ("ternary".into(), "copy".into())];
        o.ternary_experts.insert("ternary".into(), "packed".into());
        o.tool_expert_sources.insert("ternary".into(), "source".into());
        let models = Models::new(o.clone(), true).unwrap();
        assert_eq!(models.expert_sources("original"), (None, None));
        assert_eq!(models.expert_sources(&o.name), (None, None));
        assert_eq!(models.expert_sources("ternary"), (Some(Path::new("packed")), Some(Path::new("source"))));
        o.ternary_experts.clear();
        assert!(Models::new(o, true).is_err());
    }

    #[test]
    fn tool_phase_policy_requires_tools_or_tool_history() {
        use oaiy_engine::json::Json;
        for body in [r#"{"tools":[{}]}"#, r#"{"messages":[{"role":"tool","content":"ok"}]}"#,
            r#"{"messages":[{"role":"assistant","tool_calls":[{}]}]}"#] {
            assert!(needs_tool_precision(&Json::parse(body.as_bytes()).unwrap()));
        }
        for body in [r#"{}"#, r#"{"tools":[]}"#, r#"{"tools":[{}],"tool_choice":"none"}"#] {
            assert!(!needs_tool_precision(&Json::parse(body.as_bytes()).unwrap()));
        }
    }

    /// The runtime is chosen by what is on disk, not by configuration, so a
    /// mis-set `kind` cannot send a GGUF to the DeepSeek loader.
    #[test]
    fn kind_comes_from_the_path() {
        let d = std::env::temp_dir().join("oaiy-kind-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("gguf")).unwrap();
        std::fs::create_dir_all(d.join("checkpoint")).unwrap();
        std::fs::write(d.join("gguf").join("m-00001-of-00002.gguf"), b"x").unwrap();
        std::fs::write(d.join("gguf").join("m-00002-of-00002.gguf"), b"x").unwrap();
        std::fs::write(d.join("checkpoint").join("config.json"), b"{}").unwrap();

        assert_eq!(Kind::detect(&d.join("gguf")), Kind::Gguf);
        assert_eq!(Kind::detect(&d.join("gguf").join("m-00001-of-00002.gguf")), Kind::Gguf);
        assert_eq!(Kind::detect(&d.join("checkpoint")), Kind::Deepseek);
        assert_eq!(Kind::detect(Path::new("nothing-here")), Kind::Deepseek);

        // A split GGUF is opened at its lowest shard; the reader follows the rest.
        let first = Kind::gguf_path(&d.join("gguf")).unwrap();
        assert!(first.ends_with("m-00001-of-00002.gguf"), "got {}", first.display());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Two models may not share a name, or `activate` could not tell them apart.
    #[test]
    fn duplicate_names_are_refused() {
        let mut o = Options::default();
        o.name = "same".into();
        o.extra_models = vec![("same".into(), PathBuf::from("elsewhere"))];
        let err = Models::new(o, true).map(|_| ()).unwrap_err().to_string();
        assert!(err.contains("both called same"), "got {err}");
    }
}

fn available_devices(requested: &[usize], count: usize) -> Result<Vec<usize>> {
    if count == 0 { return Err(oaiy_engine::Error::Arg("DeepSeek currently requires a CUDA GPU for its trunk and state".into())); }
    let mut devices: Vec<usize> = requested.iter().copied().filter(|&d|d<count).collect();
    devices.dedup();
    if devices.is_empty() { devices.extend(0..count.min(2)); }
    Ok(devices)
}
#[test]
fn gpu_selection_adapts_to_single_device() {
    assert_eq!(available_devices(&[0,1],1).unwrap(),vec![0]);
    assert_eq!(available_devices(&[],2).unwrap(),vec![0,1]);
    assert_eq!(available_devices(&[1],1).unwrap(),vec![0]);
    assert!(available_devices(&[],0).is_err());
}

/// Where a dense GGUF model's decode step goes on WebGPU: a 3B Llama, a prompt then steps, each with the backend's
/// counters (`ggml_rs_wgpu::profile`).
/// Qwen3.8-Flash-Next from the GGUF at `path` over `gpus` (the devices `backends` are, in order): each quantized matrix
/// and each matrix of floats a chain reads packed on its layer's device, a layer's experts there while the device's
/// budget holds them beside its share of the dense matrices (and the prediction layer, where one is asked for). A
/// device with no room for all its layers' experts holds the same share of each layer's where its kernels read the
/// host's memory (`quant_experts_cached`: one RTX 5090 holds 313 of each of the IQ2_XS file's layers' 512), else
/// whole layers' experts are on the host (a chain runs those between the layer's submits).
#[cfg(feature = "webgpu")]
pub(crate) fn flashnext_gguf_on(path: &Path, gpus: &[&ggml_rs_wgpu::WgpuBackend], backends: Vec<Arc<dyn ggml_rs::Backend>>, mtp_from: Option<&Path>) -> Result<crate::engines::flashnext::FlashNext> {
    use crate::engines::flashnext::gguf_file::{dense_bytes, experts_layout, load, ExpertBlocks};
    let on = |device: usize| gpus.get(device).copied().ok_or_else(|| "no GPU to hold it".to_string());
    let quant = |device: usize, w: ggml_rs::QuantizedTensor| on(device)?.quant_linear(w);
    let float = |device: usize, values: Vec<f32>, n: usize, k: usize| on(device)?.half_linear(values, n, k);
    // The prediction layer (an EXL3 checkpoint's, its experts some 1 GB at 3 bits a weight, on the last device) only
    // where the GPUs have room for every layer's experts beside it: where some run on the host a check's rows each
    // cost those experts a row, and drafting saved nothing (one RTX 5090, 300 tokens: 6.0 to 6.5 s drafting one to
    // three deep where 5.75 to 6.07 without), so its room is the experts'. OAIY_DRAFT_OVER_HOST_LAYERS asks for it all
    // the same (OAIY_DRAFTS is the drafts' depth, whatever loads).
    let mtp_bytes: u64 = 5 << 28;
    let room: u64 = gpus.iter().map(|g| g.usage().1.saturating_sub(g.usage().0)).sum();
    let whole = std::fs::metadata(path).map_or(0, |m| m.len());
    // A card with no room for all its layers' experts holds the same share of each layer's, the ones last used, and
    // its kernels read the others from the host's memory over the bus (`quant_experts_cached`): no layer's experts
    // on the host's cores, so no trip to the host between a layer's router and its experts, and it drafts. Where
    // the cards' kernels read the host's memory and that has room for every routed expert within 80% of what is
    // free (they are all there, whichever the card holds); OAIY_EXPERT_CACHE=0: as before, whole layers' experts on
    // the host's cores. OAIY_EXPERT_SLOTS=N: N of each layer's on every card whatever its room (a check's: cards
    // that hold them all answer the same with a part held).
    let (routed_bytes, layers) = experts_layout(path)?;
    let ram = ggml_rs_wgpu::host_memory().map_or(0, |(free, _)| free as u64);
    let part = !gpus.is_empty() && gpus.iter().all(|g| g.host_weights()) && routed_bytes <= ram / 10 * 8 && std::env::var("OAIY_EXPERT_CACHE").map_or(true, |v| v != "0");
    let mtp_from = mtp_from.filter(|_| part || whole + mtp_bytes + ((gpus.len() as u64) << 30) <= room || std::env::var_os("OAIY_DRAFT_OVER_HOST_LAYERS").is_some());
    let reserve = dense_bytes(path)? / backends.len().max(1) as u64 + (1 << 30) + if mtp_from.is_some() { mtp_bytes } else { 0 };
    // Each device's experts by an allowance of their own: its budget less that reserve, counted as they are placed
    // (the layers load four at a time, each with its dense matrices: the device's own count of its weights already
    // holds the dense ones loaded so far, and the reserve taken off that left a card 3.8 GB short of its budget).
    // (OAIY_EXPERTS_ON_HOST: no allowance, every layer's routed experts on the host: for studying what a model routes
    // to, with OAIY_HOST_ROUTE_LOG)
    let none = std::env::var_os("OAIY_EXPERTS_ON_HOST").is_some();
    let allowance: Vec<u64> = gpus.iter().map(|g| if none { 0 } else { g.usage().1.saturating_sub(g.usage().0).saturating_sub(reserve) }).collect();
    let placed: Vec<std::sync::atomic::AtomicU64> = gpus.iter().map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    // (a device's share of each of its layers' experts, where it holds only some: the slots and the experts, for
    // the load's line)
    let shares: Vec<std::sync::atomic::AtomicUsize> = gpus.iter().map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
    let experts = |device: usize, e: ExpertBlocks| -> Result<Box<dyn ggml_rs::exl3::Experts>> {
        use std::sync::atomic::Ordering::Relaxed;
        let data = ggml_rs_wgpu::quant_moe::QuantExpertsData { hidden: e.hidden, ff: e.ff, experts: e.experts, gate: e.gate, up: e.up, down: e.down, shared: Some(e.shared), routing: ggml_rs::exl3::Routing::Softmax };
        match gpus.get(device) {
            Some(b) => {
                let shared = (3 * data.hidden * data.ff * 4) as u64;
                let bytes = (data.gate.1.len() + data.up.1.len() + data.down.1.len()) as u64 + shared;
                // (the device's layers' routed experts, and its allowance less their shared ones: the share of each
                // layer's it has room for)
                let (here, routed) = (layers.div_ceil(gpus.len()) as u64, routed_bytes / gpus.len() as u64);
                // (a card short of room for them all: half a gigabyte of it left for a prompt's scratch of the experts
                // it reads from the host's memory, and where it is the only card three quarters more for its chunks
                // of 1,024 rows: `FlashNext::prompt_rows`)
                let asked: Option<usize> = std::env::var("OAIY_EXPERT_SLOTS").ok().and_then(|v| v.parse().ok());
                let fits = asked.is_none() && allowance[device] >= here * shared + routed;
                let scratch: u64 = if gpus.len() == 1 { 5 << 28 } else { 1 << 29 };
                let share = allowance[device].saturating_sub(here * shared + scratch) as f64 / routed.max(1) as f64;
                if part && !fits {
                    let slots = asked.unwrap_or((data.experts as f64 * share * 0.98) as usize);
                    if slots >= 64 {
                        shares[device].store(slots.min(data.experts) * 1024 + data.experts.min(1023), Relaxed);
                        b.quant_experts_cached(data, slots)
                    } else {
                        ggml_rs_wgpu::quant_host::quant_experts_host_beside(b, data)
                    }
                } else if placed[device].fetch_add(bytes, Relaxed) + bytes <= allowance[device] {
                    // (experts the GPU's kernels do not decode, or the card had no room for after all, are the
                    // host's: their bytes are not the card's)
                    let made = b.quant_experts(data);
                    if made.as_ref().is_ok_and(|e| e.on_host()) {
                        placed[device].fetch_sub(bytes, Relaxed);
                    }
                    made
                } else {
                    placed[device].fetch_sub(bytes, Relaxed);
                    ggml_rs_wgpu::quant_host::quant_experts_host_beside(b, data)
                }
            }
            None => ggml_rs_wgpu::quant_host::quant_experts_host(data),
        }
        .map_err(Error::Arg)
    };
    let mut model = load(path, backends, &quant, &float, &experts)?;
    for (device, share) in shares.iter().enumerate() {
        let share = share.load(std::sync::atomic::Ordering::Relaxed);
        if share > 0 {
            eprintln!("GPU {device} holds {} of each layer's {} experts (the ones last used); its kernels read the others from the host's memory", share / 1024, share % 1024);
        }
    }
    // its multi-token-prediction layer, an EXL3 checkpoint's (`mtp_from`): on the last device, as that loader puts it
    if let Some(exl3) = mtp_from {
        type Make<'a> = Box<dyn Fn(ggml_rs::exl3::Exl3Data) -> std::result::Result<Arc<dyn ggml_rs::exl3::PackedLinear>, String> + Send + Sync + 'a>;
        let packed = |device: usize| -> Make<'_> {
            match gpus.get(device).copied() {
                Some(b) => Box::new(move |d| b.exl3(d)),
                None => Box::new(ggml_rs_wgpu::exl3::exl3_cpu),
            }
        };
        let layer = |device: usize, _layer: &str, list: Vec<[ggml_rs::exl3::Exl3Data; 3]>| -> Result<Box<dyn ggml_rs::exl3::Experts>> {
            match gpus.get(device) {
                Some(b) => b.exl3_experts(list),
                None => ggml_rs_wgpu::exl3::exl3_experts_cpu(list),
            }
            .map_err(Error::Arg)
        };
        if !model.attach_mtp(exl3, &packed, &layer)? {
            return Err(Error::Arg(format!("{} has no MTP layer", exl3.display())));
        }
    }
    Ok(model)
}

#[cfg(all(test, feature = "webgpu"))]
mod dense_webgpu_timing;

#[cfg(test)]
mod dense_tests {
    use super::*;
    use crate::llm::Architecture as A;

    fn msgs() -> Vec<crate::llm::ChatMessage> {
        llama_messages(&[
            oaiy_engine::json::Json::parse(br#"{"role":"system","content":"Be brief."}"#).unwrap(),
            oaiy_engine::json::Json::parse(br#"{"role":"user","content":"Capital of France?"}"#).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn each_dense_family_is_asked_in_its_own_template_and_its_turn_is_left_open() {
        let llama = dense_prompt(&A::Llama, &msgs(), false, false);
        assert!(llama.contains("<|start_header_id|>user<|end_header_id|>") && llama.contains("Capital of France?"), "{llama}");
        assert!(llama.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"), "{llama:?}");
        let gemma = dense_prompt(&A::Gemma3, &msgs(), false, false);
        assert!(gemma.contains("<start_of_turn>user") && gemma.trim_end().ends_with("<start_of_turn>model"), "{gemma:?}");
        let mistral = dense_prompt(&A::Mistral, &msgs(), false, false);
        assert!(mistral.contains("[INST]") && mistral.contains("Capital of France?"), "{mistral:?}");
        for prompt in [&llama, &gemma, &mistral] {
            assert!(!prompt.contains("<think>"), "no family but Qwen3 opens reasoning: {prompt:?}");
        }
    }

    #[test]
    fn qwen3_answers_directly_in_a_chat_and_opens_its_reasoning_only_when_asked_to_think() {
        let chat = dense_prompt(&A::Qwen3, &msgs(), false, false);
        assert!(chat.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{chat:?}");
        let thinking = dense_prompt(&A::Qwen3, &msgs(), true, false);
        assert!(thinking.ends_with("<|im_start|>assistant\n<think>"), "{thinking:?}");
        // Thinking is Qwen3's alone: another family asked to think still answers in its own template.
        assert_eq!(dense_prompt(&A::Llama, &msgs(), true, false), dense_prompt(&A::Llama, &msgs(), false, false));
    }

    #[test]
    fn gemma4_opens_its_reply_on_an_empty_thought_channel_only_where_its_template_does() {
        for thinking in [false, true] {
            let p = dense_prompt(&A::Gemma4, &msgs(), thinking, true);
            assert!(p.contains("<|turn>user\nCapital of France?<turn|>"), "{p:?}");
            assert!(p.ends_with("<|turn>model\n<|channel>thought\n<channel|>"), "{p:?}");
            let p = dense_prompt(&A::Gemma4, &msgs(), thinking, false);
            assert!(p.ends_with("<|turn>model\n"), "{p:?}");
        }
        // Only Gemma 4 has the channel.
        assert_eq!(dense_prompt(&A::Gemma3, &msgs(), false, true), dense_prompt(&A::Gemma3, &msgs(), false, false));
    }

    #[test]
    fn the_empty_thought_channel_is_read_from_the_template_not_from_where_it_wraps_earlier_reasoning() {
        // The 26B-A4B's generation prompt, and the E2B's (which names the channel only around earlier reasoning).
        let big = r"{{- '<|turn>model\n' -}}{%- if not enable_thinking | default(false) -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}";
        let small = r"{{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}{%- if add_generation_prompt -%}{{- '<|turn>model\n' -}}{%- endif -%}";
        assert!(opens_empty_thought(big));
        assert!(!opens_empty_thought(small));
        assert!(opens_empty_thought("<|channel>thought\n<channel|>"));
    }

    #[test]
    fn a_message_keeps_its_role_and_its_text() {
        let m = msgs();
        assert!(matches!(m[0].role, crate::llm::Role::System) && m[0].content == "Be brief.");
        assert!(matches!(m[1].role, crate::llm::Role::User) && m[1].content == "Capital of France?");
    }

    #[test]
    fn a_message_of_parts_keeps_their_text_and_an_image_is_refused_not_dropped() {
        let parse = |s: &[u8]| oaiy_engine::json::Json::parse(s).unwrap();
        let m = llama_messages(&[parse(br#"{"role":"user","content":[{"type":"text","text":"Capital"},{"type":"text","text":"of France?"}]}"#)]).unwrap();
        assert_eq!(m[0].content, "Capital\n\nof France?");
        let m = llama_messages(&[parse(br#"{"role":"assistant","content":null}"#)]).unwrap();
        assert!(matches!(m[0].role, crate::llm::Role::Assistant) && m[0].content.is_empty());
        let e = llama_messages(&[
            parse(br#"{"role":"user","content":"look"}"#),
            parse(br#"{"role":"tool","content":[{"type":"text","text":"screen"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}"#),
        ])
        .unwrap_err();
        assert!(e.contains("text only") && e.contains("message 2"), "{e}");
    }

    #[test]
    fn a_gemma_picture_is_its_marker_in_place_among_the_trimmed_texts_and_a_written_marker_is_refused() {
        let parse = |s: &[u8]| oaiy_engine::json::Json::parse(s).unwrap();
        let (m, images) = dense_messages(&[
            parse(br#"{"role":"user","content":[{"type":"text","text":"First: "},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":" what is it?"}]}"#),
            parse(br#"{"role":"user","content":[{"type":"image_url","image_url":"data:image/png;base64,AQ=="},{"type":"text","text":"and this"}]}"#),
            parse(br#"{"role":"user","content":[{"type":"text","text":"plain"},{"type":"text","text":"parts"}]}"#),
        ], Some("<|image|>"))
        .unwrap();
        assert_eq!(m[0].content, "First:<|image|>what is it?");
        assert_eq!(m[1].content, "<|image|>and this");
        // a message without pictures is joined as before
        assert_eq!(m[2].content, "plain\n\nparts");
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].get("url").and_then(|u| u.as_str()), Some("data:image/png;base64,AA=="));
        assert_eq!(images[1].get("url").and_then(|u| u.as_str()), Some("data:image/png;base64,AQ=="));
        for forged in [&br#"{"role":"user","content":"look <|image|>"}"#[..], br#"{"role":"user","content":[{"type":"text","text":"<|image|>"}]}"#] {
            let e = dense_messages(&[parse(forged)], Some("<|image|>")).unwrap_err();
            assert!(e.contains("image_url parts"), "{e}");
        }
        // a model without a marker reads the same text as text
        assert_eq!(dense_messages(&[parse(br#"{"role":"user","content":"look <|image|>"}"#)], None).unwrap().0[0].content, "look <|image|>");
    }

    #[test]
    fn a_gemma_picture_becomes_its_soft_tokens_between_its_open_and_close() {
        use oaiy_engine::json::Json;
        // a 1 x 1 picture: Gemma 4's processor makes it 768 x 768, 16 x 16 tokens of 3 x 3 patches of 16 pixels
        let png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
        let gv = crate::api::GemmaVision { patch: 16, max_soft_tokens: GEMMA_SOFT_TOKENS, open: 7, close: 8 };
        let rec = Json::obj([("url", Json::Str(png.into()))]);
        let (ids, images) = crate::api::gemma_images(std::slice::from_ref(&rec), vec![1, 2, 9, 3], 9, &gv, false, 4096).unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].start, 3);
        assert_eq!(images[0].prep.n_tokens(), 256);
        assert_eq!(&ids[..3], &[1, 2, 7]);
        assert!(ids[3..259].iter().all(|&t| t == 9));
        assert_eq!(&ids[259..], &[8, 3]);
        let crate::engines::job::ImagePrep::Gemma { pixels, .. } = &images[0].prep else { panic!("a Gemma picture") };
        assert_eq!(pixels.shape(), &[3, 768, 768]);
        // one marker a picture, and the context holds them
        assert!(crate::api::gemma_images(&[], vec![1, 9], 9, &gv, false, 4096).is_err());
        let Err(e) = crate::api::gemma_images(std::slice::from_ref(&rec), vec![1, 9], 9, &gv, false, 200) else { panic!("past the context") };
        assert!(e.contains("context"), "{e}");
    }
}
