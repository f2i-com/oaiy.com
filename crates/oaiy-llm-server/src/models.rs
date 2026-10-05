//! VENDORED-LOCAL: the configured models, and swapping between them.
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

use crate::engine::Job;
use crate::{api, disk, glm, Options, STATE_FORMAT};
#[cfg(feature = "cuda")]
use crate::engine;

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
    /// `llama-rs`: a GGUF file, or a directory holding one.
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
    Deepseek(Arc<dsv41::tokenizer::Tokenizer>),
    Gguf(Arc<tokenizer::Tokenizer>),
    Qwen(Arc<tokenizer::Tokenizer>),
    /// A dense GGUF (Llama, Mistral, Qwen2, Qwen3, Gemma 3 / 3n / 4): its own architecture's template, and whether
    /// its own template opens the reply on an empty thought channel (see `dense_prompt`).
    Dense(Arc<tokenizer::Tokenizer>, llama_rs::Architecture, bool),
}

impl Flavour {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Deepseek(t) => t.encode(text),
            // llama-rs applies BOS through the template, so not again here.
            Self::Gguf(t) | Self::Qwen(t) | Self::Dense(t, ..) => t.encode(text, false).unwrap_or_default(),
        }
    }

    /// The prompt for a chat request, in this model's own template.
    ///
    /// Both templates put the reply inside a `<think>` block the prompt opens, so
    /// the reply parser downstream needs no variant.
    pub fn chat_prompt(
        &self,
        msgs: &[oaiy_engine::json::Json],
        opts: &dsv41::chat::Options,
    ) -> std::result::Result<dsv41::chat::Encoded, String> {
        match self {
            Self::Deepseek(_) => {
                dsv41::chat::encode(msgs, opts).map_err(|e| e.to_string())
            }
            Self::Qwen(_) => crate::qwen::chat_prompt(msgs, opts),
            Self::Gguf(_) => {
                // The same messages through llama-rs's GLM template. `reasoning
                // effort` maps onto the three the reference Jinja accepts; the
                // DeepSeek options carry a 1-100 scale, so it is banded.
                let effort = match opts.effort {
                    0..=33 => llama_rs::ReasoningEffort::Low,
                    34..=66 => llama_rs::ReasoningEffort::High,
                    _ => llama_rs::ReasoningEffort::Max,
                };
                let out = llama_messages(msgs)?;
                // The mode decides which generation prompt: `Thinking` opens
                // `<think>` for the model to reason in, `Chat` pre-fills
                // `<think></think>` so it answers directly. Ignoring this was worth
                // the difference between "Paris." and two hundred tokens of
                // deliberation that never finished.
                let thinking = opts.mode == dsv41::chat::Mode::Thinking;
                Ok(dsv41::chat::Encoded {
                    prompt: llama_rs::glm5next_template_full(&out, true, effort, thinking),
                    images: Vec::new(),
                })
            }
            Self::Dense(_, arch, empty_thought) => Ok(dsv41::chat::Encoded {
                prompt: dense_prompt(arch, &llama_messages(msgs)?, opts.mode == dsv41::chat::Mode::Thinking, *empty_thought),
                images: Vec::new(),
            }),
        }
    }
}

/// A chat request's messages as llama-rs's templates take them: the text of each, by role. Content may be a string
/// or a list of parts (OpenAI's `[{"type":"text","text":…}]`), whose texts are joined as DeepSeek's encoder joins
/// them; these models read no images, so a part that is one is refused rather than dropped.
fn llama_messages(msgs: &[oaiy_engine::json::Json]) -> std::result::Result<Vec<llama_rs::ChatMessage>, String> {
    use oaiy_engine::json::Json;
    msgs.iter()
        .enumerate()
        .map(|(i, m)| {
            let content = match m.get("content") {
                None | Some(Json::Null) => String::new(),
                Some(Json::Str(s)) => s.clone(),
                Some(Json::Arr(parts)) => {
                    let mut texts = Vec::with_capacity(parts.len());
                    for p in parts {
                        match p.get("type").and_then(Json::as_str) {
                            Some("text" | "input_text") => texts.push(p.get("text").and_then(Json::as_str).unwrap_or_default()),
                            Some("image_url" | "input_image" | "image") => {
                                return Err(format!("this model reads text only, and message {} holds an image", i + 1));
                            }
                            other => return Err(format!("message {}: unsupported content part {other:?}", i + 1)),
                        }
                    }
                    texts.join("\n\n")
                }
                Some(_) => return Err(format!("message {}: content must be text or a list of parts", i + 1)),
            };
            let role = match m.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                "system" => llama_rs::Role::System,
                "assistant" => llama_rs::Role::Assistant,
                _ => llama_rs::Role::User,
            };
            Ok(llama_rs::ChatMessage { role, content })
        })
        .collect()
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
fn dense_prompt(arch: &llama_rs::Architecture, msgs: &[llama_rs::ChatMessage], thinking: bool, empty_thought: bool) -> String {
    if thinking && *arch == llama_rs::Architecture::Qwen3 {
        let mut p = llama_rs::apply_chat_template(&llama_rs::Architecture::Qwen2, msgs, true);
        p.push_str("<think>");
        return p;
    }
    let mut p = llama_rs::apply_chat_template(arch, msgs, true);
    if empty_thought && *arch == llama_rs::Architecture::Gemma4 {
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

/// The architecture of a dense GGUF, which is loaded whole (see `load_gguf`); None for Qwen3.5 (a path of its
/// own), the MoE families expert streaming serves, and what llama-rs does not know.
fn dense_arch(gguf: &gguf::GgufFile) -> Option<llama_rs::Architecture> {
    use llama_rs::Architecture as A;
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
    image_config: Option<crate::images::Config>,
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
        let mut image_config = opts.image_config.as_deref().map(crate::images::Config::read).transpose().map_err(Error::Arg)?;
        if let Some(root) = &opts.media_output_root {
            if !root.is_absolute() { return Err(Error::Arg("media output root must be absolute".into())); }
            if let Some(config) = &mut image_config { config.output_root = root.clone(); }
        }
        let mut specs = vec![Spec {
            name: opts.name.clone(),
            path: opts.model.clone(),
            kind: Kind::detect(&opts.model),
        }];
        for (name, path) in &opts.extra_models {
            if specs.iter().any(|s| &s.name == name) {
                return Err(Error::Arg(format!("two models are both called {name}")));
            }
            specs.push(Spec {
                name: name.clone(),
                path: path.clone(),
                kind: Kind::detect(path),
            });
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

    pub fn image_config(&self) -> Option<&crate::images::Config> { self.image_config.as_ref() }

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
            // What it freed goes back to the driver, not the allocator's pool: the next
            // model may need the whole GPU.
            #[cfg(feature = "cuda")]
            ggml_rs_cuda::release_unused_memory();
            self.say(format!("{} unloaded", l.name));
        }

        let t = std::time::Instant::now();
        self.say(format!("loading {} from {}", spec.name, spec.path.display()));
        let next = match spec.kind {
            #[cfg(feature = "cuda")]
            Kind::Deepseek => self.load_deepseek(&spec),
            #[cfg(feature = "cuda")]
            Kind::OrcaSaq => self.load_orcasaq(&spec),
            #[cfg(feature = "cuda")]
            Kind::FlashNext => self.load_flashnext(&spec),
            // OrcaSAQ's EXL3 projections on any GPU through WebGPU (else the CPU).
            #[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
            Kind::OrcaSaq => self.load_orcasaq_portable(&spec),
            // Flash-Next's EXL3 matrices and experts too.
            #[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
            Kind::FlashNext => self.load_flashnext_portable(&spec),
            #[cfg(all(not(feature = "cuda"), not(feature = "webgpu")))]
            Kind::OrcaSaq | Kind::FlashNext => Err(Error::Arg(format!("{} is an EXL3 checkpoint, which needs the CUDA or the WebGPU build", spec.name))),
            // DeepSeek-V4.1's CPU model with its dense trunk on any GPU through WebGPU (else the CPU).
            #[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
            Kind::Deepseek => self.load_deepseek_portable(&spec),
            #[cfg(all(not(feature = "cuda"), not(feature = "webgpu")))]
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
            image_token_id: 0,
            local_images: o.local_images.unwrap_or(self.loopback),
        }
    }

    // ---------------------------------------------------------------- DeepSeek
    #[cfg(feature = "cuda")]
    fn load_deepseek(&self, spec: &Spec) -> Result<Live> {
        use dsv41_cuda::{GpuModel, GpuOptions};

        let o = &self.opts;
        let count = dsv41_cuda::gpu::device_count()?;
        let devices = available_devices(&o.devices, count)?;
        if !o.devices.is_empty() && devices != o.devices {
            self.say(format!("available GPU selection: {:?} (requested {:?})", devices, o.devices));
        }
        let observer_device = if o.observer_device < count { o.observer_device }
            else { *devices.last().ok_or_else(||oaiy_engine::Error::Arg("no CUDA device available".into()))? };
        // Load the reviewer FIRST, so DeepSeek sizes its caches from the VRAM
        // genuinely remaining. Both stay resident; model routing is unchanged.
        let observer = o.observer_model.as_deref().map(|path| {
            self.say(format!("loading resident observer {} on GPU {}", path.display(), observer_device));
            crate::observer::Observer::load_shared(path, observer_device, o.observer_vram_gb, devices.contains(&observer_device))
        }).transpose()?;
        let tok = Arc::new(dsv41::tokenizer::Tokenizer::load(&spec.path)?);
        // DeepSeek sizes its caches up front and names no maximum of its own,
        // so `--ctx auto` keeps it at the server's default.
        let max_seq = if o.ctx == 0 { crate::DEFAULT_CTX } else { o.ctx };
        let gopts = GpuOptions {
            devices,
            max_seq,
            expert_cache_bytes: o.expert_cache_bytes() as usize,
            direct_io: true,
            vram_expert_bytes: None,
            vram_headroom_bytes: (o.headroom_gb * (1u64 << 30) as f64) as usize,
            cpu_expert_threads: o.cpu_threads,
            vision: o.vision,
            residual_on_device: None,
        };
        self.say(format!("expert host cache: {:.2} GiB effective (configured ceiling {} GiB; capped by available RAM)",gopts.expert_cache_bytes as f64 / (1u64<<30) as f64,o.ram_gb));
        let engram_meta = o
            .engram_meta
            .clone()
            .unwrap_or_else(|| spec.path.join("engram_meta.safetensors"));
        let (ternary_source, tool_source) = self.expert_sources(&spec.name);
        let mut model = GpuModel::load_with_expert_source(&spec.path, &engram_meta, &gopts, ternary_source)?;
        let tool_experts = tool_source.is_some();
        if let Some(source) = tool_source { model.enable_tool_experts(source)?; }
        if let Some(path) = &o.expert_trace { model.enable_route_log(path)?; }
        if let Some((path, bank)) = startup_warm_profile(o.usage.as_deref(), tool_experts) {
            let (vram, queued) = model.warm(path, 4)?;
            self.say(format!(
                "warmed {vram} {bank} experts into VRAM; {queued} more loading into RAM in the background"
            ));
        }

        let mut cfg = self.base_cfg(spec, max_seq);
        if model.has_vision() {
            cfg.vision = model.cfg.vision.clone();
            self.say("vision tower loaded; chat requests may carry images".into());
        }
        cfg.image_token_id = model.cfg.image_token_id;

        let (jobs, rx) = std::sync::mpsc::channel();
        let request_log = !o.quiet && !o.silent;
        let mut e = engine::Engine::new(
            model,
            Arc::clone(&tok),
            o.chunk,
            o.step_below,
            o.layered_max,
            o.checkpoints,
            o.usage.clone(),
            request_log,
        );
        e.warn = !o.silent;
        e.tool_experts = tool_experts;
        e.observer = observer;
        e.repetition_guard = o.repetition_guard;
        if e.observer.is_some() {
            self.say("Observer available: review off by default; opt in to a single approval gate with oaiy_observer_review=blocking".into());
        } else if tool_experts {
            self.say("DSML boundary precision: ternary prompt/prose; MXFP4 tool payload; 80/20 expert cache budgets; trunk retained".into());
        }
        if let Some(dir) = &o.prompt_cache {
            // Different expert banks must never share prompt states, including
            // two variants built from the same retained tensor index.
            let identity = format!("{:?}|{:?}|{:?}|ctx={}|dsml-v1", spec.path, ternary_source, tool_source, max_seq);
            let precision = disk::fnv(identity.as_bytes(), 0);
            let isolated = dir.join(format!("model-{precision:016x}"));
            let dir = &isolated;
            // States belong to this model (its config and weight map) and to this
            // state format.
            let mut fingerprint = disk::fnv(&[STATE_FORMAT], precision);
            for name in ["config.json", "model.safetensors.index.json"] {
                fingerprint =
                    disk::fnv(&std::fs::read(spec.path.join(name)).unwrap_or_default(), fingerprint);
            }
            if tool_experts { fingerprint = disk::fnv(b"dsml-boundary-v1-ternary-prompts", fingerprint); }
            match disk::DiskCache::open(dir, fingerprint, (o.prompt_cache_gb * 1e9) as u64) {
                Ok(cache) => {
                    self.say(format!("{} prompt states on disk in {}", cache.len(), dir.display()));
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
            flavour: Arc::new(Flavour::Deepseek(tok)),
        })
    }

    /// Qwen3.8-Flash-Next: its layers split over the configured GPUs (it needs about 50 GB).
    #[cfg(feature = "cuda")]
    fn load_flashnext(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let configured = o.model_devices.get(&spec.name).unwrap_or(&o.devices);
        // None configured (a model added from the catalog): the first two GPUs, as DeepSeek takes them. It needs about
        // 50 GB, more than one card holds, and on GPU 0 alone it could not load.
        let devices = if configured.is_empty() { available_devices(&[], dsv41_cuda::gpu::device_count()?)? } else { configured.clone() };
        self.say(format!("loading {}: Qwen3.8-Flash-Next, native EXL3 over {} GPU(s)", spec.name, devices.len()));
        let devices = &devices;
        // Every adapter for the alias, applied together, each at its strength (else the alias's).
        let default_strength = o.lora_strengths.get(&spec.name).copied().unwrap_or(1.0);
        let base = crate::flashnext::lora_base(&spec.path)?;
        let adapters = o.lora_adapters.get(&spec.name).map(|list| list.iter()
            .map(|(path, strength)| crate::lora::Adapter::open_for(path, &base).map(|a| a.with_strength(strength.unwrap_or(default_strength))))
            .collect::<Result<Vec<_>>>()).transpose()?.unwrap_or_default();
        let model = crate::flashnext::load_with_adapters(&spec.path, devices, &adapters)?;
        for (adapter, (path, strength)) in adapters.iter().zip(o.lora_adapters.get(&spec.name).into_iter().flatten()) {
            self.say(format!("Flash-Next: loaded LoRA {} for {} projections (strength {})", path.display(), adapter.len(), strength.unwrap_or(default_strength)));
        }
        let tok = Arc::new(model.tokenizer.clone());
        let max_seq = self.context(model.config.context_length);
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Flash-Next tokenizer lacks image_pad".into()))?;
        // Its vision tower: Qwen3.5's, stored as EXL3, on the last GPU.
        let vision_path = o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten();
        let projector = if let Some(path) = vision_path {
            let cuda = model.cudas.last().expect("a GPU").clone();
            let mm = crate::qwen_vision::load(path, cuda.clone(), model.config.hidden, Some(cuda))?;
            cfg.qwen_vision = Some(mm.config().clone());
            self.say("Flash-Next: Qwen vision tower loaded (576 tokens/image)".into());
            Some(mm)
        } else { None };
        let (jobs, rx) = std::sync::mpsc::channel();
        // Flash-Next keeps one conversation on the GPUs and nothing on disk: the conversations it
        // sets aside (a runner and its call's sub-agent taking turns) wait in host RAM, within
        // --park-gb and never more than half of what is free.
        let park = crate::qwen_park::budget(o.park_gb, ggml_rs_cuda::host_memory().map(|(free, _)| free as u64));
        if park > 0 {
            self.say(format!("Flash-Next: conversations it sets aside wait in host RAM, up to {:.1} GB (--park-gb)", park as f64 / 1e9));
        }
        let e = crate::qwen::QwenEngine::new(crate::qwen::Hybrid::Flash(Box::new(model)), projector, max_seq, !o.quiet && !o.silent).park_up_to(park);
        let thread = std::thread::Builder::new().name("flashnext-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok)) })
    }

    #[cfg(feature = "cuda")]
    fn load_orcasaq(&self, spec: &Spec) -> Result<Live> {
        let o=&self.opts;
        self.say(format!("loading {}: native EXL3 mixed precision",spec.name));
        // Every adapter for the alias, applied together, each at its strength (else the alias's).
        let default_strength=o.lora_strengths.get(&spec.name).copied().unwrap_or(1.0);
        let adapters=o.lora_adapters.get(&spec.name).map(|list| list.iter().map(|(path,strength)| crate::lora::Adapter::open(path).map(|a|a.with_strength(strength.unwrap_or(default_strength)))).collect::<Result<Vec<_>>>()).transpose()?.unwrap_or_default();
        let model=if adapters.is_empty() {crate::orcasaq::load(&spec.path,&o.devices)?} else {crate::orcasaq::load_with_adapter(&spec.path,&o.devices,&adapters)?};
        for (adapter,(path,strength)) in adapters.iter().zip(o.lora_adapters.get(&spec.name).into_iter().flatten()) {
            self.say(format!("OrcaSAQ: loaded LoRA {} for {} text projections (strength {})",path.display(),adapter.len(),strength.unwrap_or(default_strength)));
        }
        let tok=Arc::new(model.tokenizer().clone());
        let max_seq=self.context(model.config().context_length);
        let mut cfg=self.base_cfg(spec,max_seq);
        cfg.image_token_id=tok.token_id("<|image_pad|>").ok_or_else(||Error::Arg("Orca tokenizer lacks image_pad".into()))?;
        let vision_path=o.vision.then(||o.vision_projectors.get(&spec.name)).flatten();
        let projector=if let Some(path)=vision_path {
            let llama_rs::Model::Qwen35(m)=&model else { return Err(Error::Arg("Orca requires Qwen hybrid runtime".into())); };
            // Use the last configured cache device: the first already holds
            // the text weights. Image embeddings cross back through host RAM.
            let backend=m.cache_backends.last().unwrap_or(&m.backend).clone();
            let mm=crate::qwen_vision::load(path,backend,model.config().embedding_dim,None)?;
            cfg.qwen_vision=Some(mm.config().clone());
            self.say("OrcaSAQ: original Qwen vision tower loaded (576 tokens/image)".into());
            Some(mm)
        } else { None };
        let (jobs,rx)=std::sync::mpsc::channel();
        let mut e=crate::qwen::QwenEngine::new(model,projector,max_seq,!o.quiet && !o.silent);
        e.image_disk_cache=vision_path.is_some();
        if let Some(dir)=&o.prompt_cache {
            let mut fp=disk::fnv(b"orcasaq2-exl3-qwen-state-v1",0);
            fp=disk::fnv(spec.path.as_os_str().as_encoded_bytes(),fp);
            for adapter in &adapters {fp=disk::fnv(&adapter.fingerprint.to_le_bytes(),fp);}
            let mut files:Vec<_>=std::fs::read_dir(&spec.path)?.filter_map(|e|e.ok().map(|e|e.path())).filter(|p|
                p.extension().is_some_and(|x|x=="safetensors" || x=="json")).collect();
            if let Some(path)=vision_path {
                // Include tower weights/config and preprocessing semantics:
                // a state from a different vision pipeline is never reusable.
                fp=disk::fnv(b"qwen38-vision-letterbox768-erfgelu-v1",fp);
                files.extend(std::fs::read_dir(path)?.filter_map(|e|e.ok().map(|e|e.path())).filter(|p|
                    p.extension().is_some_and(|x|x=="safetensors" || x=="json")));
            }
            files.sort();
            for p in files {
                let meta=std::fs::metadata(&p)?;
                fp=disk::fnv(p.as_os_str().as_encoded_bytes(),fp);
                fp=disk::fnv(&meta.len().to_le_bytes(),fp);
                if let Ok(t)=meta.modified().and_then(|t|t.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)) {fp=disk::fnv(&t.as_nanos().to_le_bytes(),fp);}
            }
            match disk::DiskCache::open(dir,fp,(o.prompt_cache_gb*1e9)as u64){
                Ok(cache)=>{self.say(format!("OrcaSAQ prompt cache: {} entries in {}",cache.len(),dir.display())); e.disk=Some(cache);},
                Err(err)=>self.say(format!("OrcaSAQ prompt cache unavailable: {err}")),
            }
        }
        let thread=std::thread::Builder::new().name("orcasaq-model".into()).spawn(move||e.run(rx))?;
        Ok(Live{name:spec.name.clone(),jobs,thread,cfg:Arc::new(cfg),flavour:Arc::new(Flavour::Qwen(tok))})
    }

    /// Qwen3.8-Flash-Next without CUDA: its EXL3 matrices and experts on the WebGPU adapter while the weight budget
    /// holds them (the first layers' experts), the rest decoded on the CPU, everything else on the host. No PEFT
    /// adapters and no vision tower (both CUDA's); conversations set aside in host RAM as on CUDA.
    #[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
    fn load_flashnext_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        if o.lora_adapters.contains_key(&spec.name) {
            return Err(Error::Arg(format!("{}: LoRA adapters need the CUDA build", spec.name)));
        }
        let picked = crate::backend::open(o, &o.devices)?;
        self.say(format!("{} runs on {} (EXL3 experts decoded in the matmul, a layer's in two batches)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        // The computer's other discrete GPUs take a share of the layers, a whole layer each (its experts too): two
        // 32 GB cards hold all of its 46 GB of experts, where the CPU decoded what one could not (most of a step).
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = wgpu.map(|b| b.others(o.webgpu_gb.map(|g| g << 30))).unwrap_or_default().into_iter().map(Arc::new).collect();
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
        let reserve = crate::flashnext::dense_exl3_bytes(&spec.path)? / backends.len() as u64 + (1 << 30);
        let experts = |device: usize, _layer: &str, list: Vec<[ggml_rs::exl3::Exl3Data; 3]>| -> Result<Box<dyn ggml_rs::exl3::Experts>> {
            match gpus.get(device) {
                Some(b) => b.exl3_experts_leaving(list, reserve),
                None => ggml_rs_wgpu::exl3::exl3_experts_cpu(list),
            }
            .map_err(Error::Arg)
        };
        let model = crate::flashnext::load_portable(&spec.path, backends, &packed, &experts)?;
        let warm = std::time::Instant::now();
        if model.warm_up() {
            self.say(format!("{}: its chained steps ready in {:.1} s", spec.name, warm.elapsed().as_secs_f64()));
        }
        for (d, g) in gpus.iter().enumerate() {
            let (used, budget) = g.usage();
            self.say(format!("{}: {:.1} GB of EXL3 weights on GPU {d} ({} at {}, budget {:.0} GB), the rest on the CPU", spec.name, used as f64 / 1e9, g.adapter().name, g.adapter().pci_bus_id, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer.clone());
        // The cache is on the host: bound it as the other portable models are.
        let max_seq = self.context(model.config.context_length).min(16384);
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Flash-Next tokenizer lacks image_pad".into()))?;
        let (jobs, rx) = std::sync::mpsc::channel();
        let park = crate::qwen_park::budget(o.park_gb, ggml_rs_wgpu::host_memory().map(|(free, _)| free as u64));
        let e = crate::qwen::QwenEngine::new(crate::qwen::Hybrid::Flash(Box::new(model)), None, max_seq, !o.quiet && !o.silent).park_up_to(park);
        let thread = std::thread::Builder::new().name("flashnext-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok)) })
    }

    /// DeepSeek-V4.1 without CUDA: the CPU model (`dsv41::model`, the reference the CUDA path is tested against), its
    /// experts streamed through the host cache from RAM and the drive, and its dense trunk (attention projections,
    /// shared experts, router, head: about 9.7 GB) on the WebGPU adapter while the budget holds it. No images and no
    /// observer (both CUDA's); a conversation's next turn continues the state rather than reading it all again.
    #[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
    fn load_deepseek_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let tok = Arc::new(dsv41::tokenizer::Tokenizer::load(&spec.path)?);
        // As the CUDA build: the server's default context unless one is asked for.
        let max_seq = if o.ctx == 0 { crate::DEFAULT_CTX } else { o.ctx };
        let engram_meta = o.engram_meta.clone().unwrap_or_else(|| spec.path.join("engram_meta.safetensors"));
        let opts = dsv41::model::ModelOptions { max_seq, expert_cache_bytes: o.expert_cache_bytes() as usize, direct_io: true };
        self.say(format!("expert host cache: {:.2} GiB", opts.expert_cache_bytes as f64 / (1u64 << 30) as f64));
        let mut model = dsv41::model::Model::load(&spec.path, &engram_meta, &opts)?;
        let picked = crate::backend::open(o, &o.devices)?;
        match picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
            Some(b) => {
                let (count, bytes) = crate::dsv41_portable::offload(&mut model, b);
                // A prompt's busy experts there too, through record slots in what the budget has left after the trunk,
                // and the experts used most kept there in what is left after those (a decode step's computed there while
                // the CPU reads and computes the rest).
                let experts = match crate::dsv41_portable::WgpuExperts::new(b) {
                    Some(k) => {
                        let (n, kept) = (k.slots(), k.tier().0);
                        model.set_experts_kernel(Some(Arc::new(k)));
                        format!("a prompt's busy experts ({n} at a time) and {kept} experts kept there between requests; the rest on the CPU")
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
        let e = crate::dsv41_portable::Engine::new(model, Arc::clone(&tok), !o.quiet && !o.silent);
        let thread = std::thread::Builder::new().name("deepseek-model".into()).spawn(move || e.run(rx)).map_err(Error::Io)?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Deepseek(tok)) })
    }

    /// OrcaSAQ without CUDA: its packed EXL3 projections on the WebGPU adapter while the weight budget holds them,
    /// the rest decoded on the CPU, everything else on the host as in any portable model. No PEFT adapters and no
    /// vision tower (both CUDA's); its prompt states are kept as on CUDA, under a fingerprint of their own.
    #[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
    fn load_orcasaq_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        if o.lora_adapters.contains_key(&spec.name) {
            return Err(Error::Arg(format!("{}: LoRA adapters need the CUDA build", spec.name)));
        }
        let picked = crate::backend::open(o, &o.devices)?;
        self.say(format!("{} runs on {} (EXL3, decoded in the matmul)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        let packed = |data: ggml_rs::exl3::Exl3Data| match wgpu {
            Some(b) => b.exl3(data),
            None => ggml_rs_wgpu::exl3::exl3_cpu(data),
        };
        let model = crate::orcasaq::load_portable(&spec.path, Arc::clone(&picked.backend), &packed)?;
        if let Some((used, budget)) = wgpu.map(|b| b.usage()) {
            self.say(format!("{}: {:.1} GB of EXL3 weights on the GPU (budget {:.0} GB)", spec.name, used as f64 / 1e9, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer().clone());
        // The cache is on the host: bound it as the other portable models are.
        let max_seq = self.context(model.config().context_length).min(16384);
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Orca tokenizer lacks image_pad".into()))?;
        let (jobs, rx) = std::sync::mpsc::channel();
        let mut e = crate::qwen::QwenEngine::new(model, None, max_seq, !o.quiet && !o.silent);
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
            match disk::DiskCache::open(dir, fp, (o.prompt_cache_gb * 1e9) as u64) {
                Ok(cache) => e.disk = Some(cache),
                Err(err) => self.say(format!("OrcaSAQ prompt cache unavailable: {err}")),
            }
        }
        let thread = std::thread::Builder::new().name("orcasaq-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok)) })
    }

    // ---------------------------------------------------------------- GGUF
    fn load_gguf(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let path = Kind::gguf_path(&spec.path)?;
        // Every card `--devices` names, in that order: the first carries the trunk
        // and the rest take a share of the MoE layers. `open_cards` hands back the
        // same instances the expert tier will use, so the trunk and card 0's expert
        // shard share one CUDA context rather than opening a second on that device.
        let devices = self.image_config.as_ref().filter(|c|c.controller_name==spec.name).map(|c|vec![c.controller_device]).unwrap_or_else(||o.devices.clone());
        // CUDA cards in a CUDA build; WebGPU or the CPU without one.
        let picked = crate::backend::open(o, &devices)?;
        self.say(format!("{} runs on {}", spec.name, picked.label));
        let backend: Arc<dyn ggml_rs::Backend> = Arc::clone(&picked.backend);
        let gguf = gguf::GgufFile::open(&path).map_err(|e| Error::Arg(e.to_string()))?;
        if gguf.get_str("general.architecture").ok() == Some("qwen35") {
            let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))?;
            let tok = Arc::new(model.tokenizer().clone());
            // Dense Qwen fits one card; bound the initial KV allocation.
            let max_seq = self.context(model.config().context_length).min(16384);
            let mut cfg = self.base_cfg(spec, max_seq);
            cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Qwen tokenizer lacks image_pad".into()))?;
            let projector = if o.vision {
                o.vision_projectors.get(&spec.name).map(|p| {
                    let file = gguf::GgufFile::open(p).map_err(|e| Error::Arg(e.to_string()))?;
                    let mm = llama_rs::MmProj::from_gguf(&file, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))?;
                    if !matches!(&mm, llama_rs::MmProj::Qwen3Vl(q) if q.projector.mm2.shape()[0] == model.config().embedding_dim) {
                        return Err(Error::Arg("Qwen vision projector does not match the text model".into()));
                    }
                    Ok(mm)
                }).transpose()?
            } else { None };
            cfg.qwen_vision = projector.as_ref().map(|p|p.config().clone());
            let (jobs,rx) = std::sync::mpsc::channel();
            let mut e = crate::qwen::QwenEngine::new(model,projector,max_seq,!o.quiet && !o.silent);
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
            return Ok(Live {name:spec.name.clone(),jobs,thread,cfg:Arc::new(cfg),flavour:Arc::new(Flavour::Qwen(tok))});
        }
        // Dense models (Llama, Mistral and Qwen2 without experts, Qwen3, Gemma 3, 3n and 4) are loaded whole onto
        // the backend, as Qwen3.5 is above, and answer in their own template with their own stop tokens. Expert
        // streaming below is for the MoE families only and refused every dense model, so a Llama or Gemma GGUF (most of
        // what people keep, and most of the catalog) could not be served at all. They chat; tool calls are read in the
        // GLM and Qwen3.5 formats only, so the Agent's tools need one of those.
        if let Some(arch) = dense_arch(&gguf) {
            let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))?;
            let empty_thought = gguf.get_str("tokenizer.chat_template").is_ok_and(opens_empty_thought);
            drop(gguf);
            let tok = Arc::new(model.tokenizer().clone());
            // Bound the initial KV allocation, as for dense Qwen3.5: a full context of an 8B model is gigabytes.
            let max_seq = self.context(model.config().context_length).min(16384);
            let cfg = self.base_cfg(spec, max_seq);
            let (jobs, rx) = std::sync::mpsc::channel();
            let e = glm::GlmEngine::new(model, Arc::clone(&tok), max_seq, !o.quiet && !o.silent);
            let thread = std::thread::Builder::new().name("model".into()).spawn(move || e.run(rx)).map_err(Error::Io)?;
            return Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Dense(tok, arch, empty_thought)) });
        }
        drop(gguf);
        let budget = o.expert_cache_bytes();
        if o.ram_gb == 0 {
            self.say(format!(
                "expert cache: {:.0} GB of host RAM (80% of what is free; --ram-gb pins it)",
                budget as f64 / 1e9
            ));
        }
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut model = llama_rs::Model::open_streaming(&path, backend, budget)
            .map_err(|e| Error::Arg(e.to_string()))?;

        // The expert hierarchy. `open_streaming` leaves a streamed model with its
        // RAM cache only -- no VRAM expert cache, one card, no CPU tier -- which for
        // GLM-5.3-Flash is 0.686 s a token against 0.139 tiered, because then every
        // routed expert of every token crosses PCIe. Nothing else served here needs
        // asking, so the arch decides.
        #[cfg(feature = "cuda")]
        if let (llama_rs::Model::Glm5Next(g), Some(cards)) = (&mut model, picked.cards) {
            g.enable_tiering(cards, 0).map_err(|e| Error::Arg(e.to_string()))?;
            if !o.quiet && !o.silent {
                let (budgets, layers) = g.tier_layout();
                let gb: Vec<String> =
                    budgets.iter().map(|b| format!("{:.1}", *b as f64 / 1e9)).collect();
                eprintln!(
                    "  expert tier: {} card(s), VRAM {} GB, MoE layers {:?}",
                    budgets.len(),
                    gb.join("+"),
                    layers
                );
            }
        }
        let model = model;
        let tok = Arc::new(model.tokenizer().clone());

        // The state the model was opened with bounds the context, whatever was
        // asked for.
        let max_seq = match &model {
            llama_rs::Model::Glm5Next(g) => self.context(g.max_len()),
            _ => self.context(model.config().context_length),
        };
        let cfg = self.base_cfg(spec, max_seq);

        let (jobs, rx) = std::sync::mpsc::channel();
        let mut e = glm::GlmEngine::new(model, Arc::clone(&tok), max_seq, !o.quiet && !o.silent);
        // VENDORED-LOCAL: GLM-5.3-Flash. Prompt states on disk, as DeepSeek has.
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
            flavour: Arc::new(Flavour::Gguf(tok)),
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
#[cfg(all(test, feature = "webgpu", not(feature = "cuda")))]
mod dense_webgpu_timing {
    #[test]
    #[ignore = "a timing; needs a WebGPU adapter and E:\\models\\llama-3.2-3b-q4_k_m.gguf; run with --nocapture"]
    fn measure_a_dense_models_decode_on_webgpu() {
        use std::sync::Arc;
        use std::time::Instant;
        let path = std::env::var("DENSE_MODEL").unwrap_or_else(|_| r"E:\models\llama-3.2-3b-q4_k_m.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(Some(8 << 30)) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        let mut kv = model.new_kv_cache(4096);
        let n: u32 = std::env::var("DENSE_PROMPT").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let prompt: Vec<u32> = (0..n).map(|i| 1000 + (i * 7919) % 20000).collect();
        ggml_rs_wgpu::profile::take_line();
        let t = Instant::now();
        let mut logits = model.forward(&prompt, &mut kv);
        eprintln!("prompt of {} tokens: {:.3} s; {}", prompt.len(), t.elapsed().as_secs_f64(), ggml_rs_wgpu::profile::take_line());
        let mut next = model.last_logits(&logits).data().iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let t = Instant::now();
        let steps = 32;
        for _ in 0..steps {
            logits = model.forward(&[next], &mut kv);
            next = model.last_logits(&logits).data().iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        }
        let secs = t.elapsed().as_secs_f64();
        eprintln!("{steps} decode steps: {:.1} ms a step; {}", secs * 1e3 / steps as f64, ggml_rs_wgpu::profile::take_line());
    }

    /// A decode step chained on the GPU (a Llama's, a Qwen3's or a Gemma 3's: DENSE_MODEL) answers as the host path does: from the
    /// same prompt, 64 greedy steps each way give the same tokens, every step's logits close (cosine 0.9999 or more).
    #[test]
    #[ignore = "needs a WebGPU adapter and the 3B Llama GGUF (E:/models/llama-3.2-3b-q4_k_m.gguf, or DENSE_MODEL)"]
    fn a_chained_decode_step_answers_as_the_host_path() {
        use std::sync::Arc;
        let path = std::env::var("DENSE_MODEL").unwrap_or_else(|_| r"E:\models\llama-3.2-3b-q4_k_m.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(Some(8 << 30)) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        // the host path, op by op (a Llama's or a Qwen3's), and the model's own (chained where it can be)
        let host = |t: &[u32], kv: &mut llama_rs::KvCache| match &model {
            llama_rs::Model::Llama(m) => m.forward_host(t, kv),
            llama_rs::Model::Qwen3(m) => m.forward_host(t, kv),
            llama_rs::Model::Gemma3(m) => m.forward_host(t, kv),
            _ => panic!("a Llama, a Qwen3 or a Gemma 3"),
        };
        let n: u32 = std::env::var("DENSE_PROMPT").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let prompt: Vec<u32> = (0..n).map(|i| 1000 + (i * 7919) % 20000).collect();
        let argmax = |l: &ggml_rs::Tensor| l.data().iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let run = |chained: bool| {
            let mut kv = model.new_kv_cache(prompt.len() + 80);
            // the prompt chained too, in one chunk, its logits compared with the rest
            let l = if chained { model.forward(&prompt, &mut kv) } else { host(&prompt, &mut kv) };
            let first = model.last_logits(&l).data().to_vec();
            let mut next = argmax(&model.last_logits(&l));
            let (mut tokens, mut all) = (vec![next], vec![first]);
            for _ in 0..64 {
                let l = if chained { model.forward(&[next], &mut kv) } else { host(&[next], &mut kv) };
                let l = model.last_logits(&l).data().to_vec();
                next = l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
                tokens.push(next);
                all.push(l);
            }
            (tokens, all)
        };
        let t = std::time::Instant::now();
        let (host_tokens, host_logits) = run(false);
        let host_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let (chain_tokens, chain_logits) = run(true);
        let chain_s = t.elapsed().as_secs_f64();
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        let worst = host_logits.iter().zip(&chain_logits).map(|(a, b)| cosine(a, b)).fold(1.0f64, f64::min);
        eprintln!("prompt {n}: host {host_s:.2} s, chained {chain_s:.2} s for 64 steps; worst logits cosine {worst:.6}");
        assert_eq!(host_tokens, chain_tokens);
        assert!(worst >= 0.9999, "{worst}");
    }

    /// Qwen3.8-Flash-Next chained on the GPUs (its layers over every discrete one) answers as its own path does: the
    /// prompt as one chunk, then step by step on the same tokens (the host path's greedy ones): each step's logits
    /// close, the same greedy token (bar near-ties), and the chain ran. FLASHNEXT_MODEL: the EXL3 checkpoint.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next and its checkpoint (FLASHNEXT_MODEL)"]
    fn a_chained_flashnext_step_answers_as_its_own_path() {
        use std::sync::Arc;
        let path = std::env::var("FLASHNEXT_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-Flash-Next\exl3-3.05bpw".into());
        let Ok(b0) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = b0.others(None).into_iter().map(Arc::new).collect();
        let b0 = Arc::new(b0);
        let gpus: Vec<&ggml_rs_wgpu::WgpuBackend> = std::iter::once(b0.as_ref()).chain(others.iter().map(|g| g.as_ref())).collect();
        let backends: Vec<Arc<dyn ggml_rs::Backend>> = std::iter::once(Arc::clone(&b0) as Arc<dyn ggml_rs::Backend>).chain(others.iter().map(|g| Arc::clone(g) as Arc<dyn ggml_rs::Backend>)).collect();
        type Make<'a> = Box<dyn Fn(ggml_rs::exl3::Exl3Data) -> std::result::Result<Arc<dyn ggml_rs::exl3::PackedLinear>, String> + Send + Sync + 'a>;
        let packed = |device: usize| -> Make<'_> {
            let b = gpus[device];
            Box::new(move |d| b.exl3(d))
        };
        let p = std::path::Path::new(&path);
        let reserve = crate::flashnext::dense_exl3_bytes(p).unwrap() / backends.len() as u64 + (1 << 30);
        let experts = |device: usize, _layer: &str, list: Vec<[ggml_rs::exl3::Exl3Data; 3]>| -> oaiy_engine::Result<Box<dyn ggml_rs::exl3::Experts>> {
            gpus[device].exl3_experts_leaving(list, reserve).map_err(oaiy_engine::Error::Arg)
        };
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts).unwrap();
        let steps: usize = std::env::var("FLASHNEXT_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        let prompt: Vec<u32> = model.tokenizer.encode("Write a short story about a cat called Moss who lives on a boat.", false).unwrap();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        let (mut kh, mut kc) = (model.new_kv_cache(prompt.len() + steps + 64), model.new_kv_cache(prompt.len() + steps + 64));
        let e = model.embed_text(&prompt).unwrap();
        let t = std::time::Instant::now();
        let lh = model.forward_host(&prompt, &e, &mut kh, None).unwrap();
        let ph = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let lc = model.forward(&prompt, &e, &mut kc, None).unwrap();
        let pc = t.elapsed().as_secs_f64();
        let prompt_cos = cosine(lh.data(), lc.data());
        eprintln!("the prompt's {} tokens: host {:.0} ms, chained {:.0} ms; logits cosine {prompt_cos:.6}, the same greedy token {}", prompt.len(), ph * 1e3, pc * 1e3, argmax(lh.data()) == argmax(lc.data()));
        assert!(prompt_cos > 0.99, "{prompt_cos}");
        let mut next = argmax(lh.data());
        let (mut worst, mut same, mut th, mut tc) = (1.0f64, 0usize, 0f64, 0f64);
        for _ in 0..steps {
            let e = model.embed_text(&[next]).unwrap();
            let t = std::time::Instant::now();
            let host = model.forward_host(&[next], &e, &mut kh, None).unwrap();
            th += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let chained = model.forward(&[next], &e, &mut kc, None).unwrap();
            tc += t.elapsed().as_secs_f64();
            worst = worst.min(cosine(host.data(), chained.data()));
            same += (argmax(host.data()) == argmax(chained.data())) as usize;
            next = argmax(host.data());
        }
        let runs = model.chain_runs();
        eprintln!("{steps} steps: host {:.1} ms a step, chained {:.1} ms ({runs} chained); worst logits cosine {worst:.6}; the same greedy token {same} of {steps}", th * 1e3 / steps as f64, tc * 1e3 / steps as f64);
        assert_eq!(runs, steps + 1, "the prompt and every step chained");
        assert!(worst > 0.99, "{worst}");
        // a later chunk: after the chained steps (the state the chain's, the cache's rows its own), and after a prompt
        // the host path ran (its state adopted, the cache's rows the host's), as a turn after a checkpoint's restore
        let more: Vec<u32> = model.tokenizer.encode(" Now tell me what Moss likes to eat, and where on the boat Moss sleeps when it rains.", false).unwrap();
        let e = model.embed_text(&more).unwrap();
        let lh = model.forward_host(&more, &e, &mut kh, None).unwrap();
        let lc = model.forward(&more, &e, &mut kc, None).unwrap();
        let after_steps = cosine(lh.data(), lc.data());
        let (mut kh2, mut kc2) = (model.new_kv_cache(prompt.len() + more.len() + 8), model.new_kv_cache(prompt.len() + more.len() + 8));
        let e1 = model.embed_text(&prompt).unwrap();
        let _ = model.forward_host(&prompt, &e1, &mut kh2, None).unwrap();
        let _ = model.forward_host(&prompt, &e1, &mut kc2, None).unwrap();
        let lh2 = model.forward_host(&more, &e, &mut kh2, None).unwrap();
        let lc2 = model.forward(&more, &e, &mut kc2, None).unwrap();
        let after_host = cosine(lh2.data(), lc2.data());
        eprintln!("a later chunk of {}: after the chained steps logits cosine {after_steps:.6} (the same greedy token {}), after the host's prompt {after_host:.6} ({})", more.len(), argmax(lh.data()) == argmax(lc.data()), argmax(lh2.data()) == argmax(lc2.data()));
        assert_eq!(model.chain_runs(), steps + 3, "both chunks chained");
        assert!(after_steps > 0.99 && after_host > 0.99, "{after_steps} {after_host}");
    }

    /// Where a chained Qwen3.8-Flash-Next run's time goes (FLASHNEXT_MODEL): a prompt's chunk of 512 and decode steps,
    /// each kernel's GPU time with OAIY_CHAIN_PROFILE set (`ggml_rs_wgpu::profile::take_kernels`).
    #[test]
    #[ignore = "a timing; needs WebGPU adapters with room for Qwen3.8-Flash-Next (FLASHNEXT_MODEL); run with --nocapture"]
    fn measure_a_chained_flashnext() {
        use std::sync::Arc;
        use std::time::Instant;
        let path = std::env::var("FLASHNEXT_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-Flash-Next\exl3-3.05bpw".into());
        let Ok(b0) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = b0.others(None).into_iter().map(Arc::new).collect();
        let b0 = Arc::new(b0);
        let gpus: Vec<&ggml_rs_wgpu::WgpuBackend> = std::iter::once(b0.as_ref()).chain(others.iter().map(|g| g.as_ref())).collect();
        let backends: Vec<Arc<dyn ggml_rs::Backend>> = std::iter::once(Arc::clone(&b0) as Arc<dyn ggml_rs::Backend>).chain(others.iter().map(|g| Arc::clone(g) as Arc<dyn ggml_rs::Backend>)).collect();
        type Make<'a> = Box<dyn Fn(ggml_rs::exl3::Exl3Data) -> std::result::Result<Arc<dyn ggml_rs::exl3::PackedLinear>, String> + Send + Sync + 'a>;
        let packed = |device: usize| -> Make<'_> {
            let b = gpus[device];
            Box::new(move |d| b.exl3(d))
        };
        let p = std::path::Path::new(&path);
        let reserve = crate::flashnext::dense_exl3_bytes(p).unwrap() / backends.len() as u64 + (1 << 30);
        let experts = |device: usize, _layer: &str, list: Vec<[ggml_rs::exl3::Exl3Data; 3]>| -> oaiy_engine::Result<Box<dyn ggml_rs::exl3::Experts>> {
            gpus[device].exl3_experts_leaving(list, reserve).map_err(oaiy_engine::Error::Arg)
        };
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts).unwrap();
        let tokens: Vec<u32> = (0..1100u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let mut kv = model.new_kv_cache(2048);
        let report = |what: &str, wall: f64| {
            let k = ggml_rs_wgpu::profile::take_kernels();
            let gpu: f64 = k.iter().map(|e| e.1).sum();
            eprintln!("{what}: {wall:.1} ms, of it the GPU's kernels {gpu:.1} ms; {}", ggml_rs_wgpu::profile::take_line());
            for (name, ms, n) in k.iter().take(14) {
                eprintln!("  {name:<28} {ms:>9.2} ms {n:>6}");
            }
        };
        let mut at = 0;
        for (i, n) in [512usize, 512, 64].into_iter().enumerate() {
            let e = model.embed_text(&tokens[at..at + n]).unwrap();
            let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
            let t = Instant::now();
            let _ = model.forward(&tokens[at..at + n], &e, &mut kv, None).unwrap();
            report(&format!("chunk {i} of {n} at {at}"), t.elapsed().as_secs_f64() * 1e3);
            at += n;
        }
        let mut next = 1234u32;
        for step in 0..4 {
            let e = model.embed_text(&[next]).unwrap();
            let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
            let t = Instant::now();
            let l = model.forward(&[next], &e, &mut kv, None).unwrap();
            if step == 3 {
                report("a decode step", t.elapsed().as_secs_f64() * 1e3);
            }
            next = l.data().iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        }
        assert_eq!(model.chain_runs(), 7, "every run chained");
    }

    /// Where a chained Qwen3.5 run's time goes (QWEN35_MODEL): prompt chunks of a few tokens as the server's
    /// checkpoints cut them, a checkpoint's read of the recurrent state, decode steps, and a chunk of 512.
    #[test]
    #[ignore = "a timing; needs a WebGPU adapter and a Qwen3.5 GGUF; run with --nocapture"]
    fn measure_a_chained_qwen35() {
        use std::sync::Arc;
        use std::time::Instant;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let mut kv = model.new_kv_cache(4096);
        let tokens: Vec<u32> = (0..2048u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let mut at = 0;
        let mut forward = |n: usize, kv: &mut llama_rs::KvCache| {
            let t = Instant::now();
            let e = m.embed_text(&tokens[at..at + n]);
            let l = m.forward_embeds_positions(&e, n, kv, None).unwrap();
            at += n;
            let _ = l.to_host();
            t.elapsed().as_secs_f64() * 1e3
        };
        let capture = |kv: &llama_rs::KvCache| {
            let t = Instant::now();
            let n: usize = kv.ssm_state.iter().chain(&kv.ssm_conv).flatten().map(|s| s.to_host().numel()).sum();
            (t.elapsed().as_secs_f64() * 1e3, n * 4)
        };
        for (label, n) in [("first 26", 26usize), ("then 6", 6), ("then 1", 1), ("then 22", 22), ("then 6", 6), ("then 1", 1)] {
            let ms = forward(n, &mut kv);
            eprintln!("{label} tokens: {ms:.1} ms");
            if n > 1 {
                let (ms, bytes) = capture(&kv);
                eprintln!("  a checkpoint's read of the state ({:.0} MB): {ms:.1} ms", bytes as f64 / 1e6);
            }
        }
        let steps = 32;
        let t = Instant::now();
        for _ in 0..steps {
            forward(1, &mut kv);
        }
        eprintln!("{steps} decode steps: {:.2} ms a step", t.elapsed().as_secs_f64() * 1e3 / steps as f64);
        for n in [512usize, 512] {
            let past = kv.len;
            eprintln!("a chunk of {n} at {past}: {:.1} ms", forward(n, &mut kv));
        }
    }

    /// Qwen3.5's hybrid chained on the GPU (QWEN35_MODEL: the 9B, or Qwen3.8 27B) answers as its host path does: the
    /// same prompt (QWEN35_PROMPT tokens, in chunks of 512 as the server sends them) then 64 greedy steps each way
    /// give the same tokens, every step's logits close; and the chain did run.
    #[test]
    #[ignore = "needs a WebGPU adapter and a Qwen3.5 GGUF (E:/models/Qwen3.5-9B-Q4_K_M.gguf, or QWEN35_MODEL)"]
    fn a_chained_qwen35_run_answers_as_the_host_path() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.5-9B-Q4_K_M.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let n: u32 = std::env::var("QWEN35_PROMPT").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let steps: usize = std::env::var("QWEN35_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let prompt: Vec<u32> = (0..n).map(|i| 1000 + (i * 7919) % 20000).collect();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let run = |chained: bool| {
            let mut kv = model.new_kv_cache(prompt.len() + steps + 16);
            let forward = |t: &[u32], kv: &mut llama_rs::KvCache| {
                let e = m.embed_text(t);
                let l = if chained { m.forward_embeds_positions(&e, t.len(), kv, None) } else { m.forward_embeds_host(&e, t.len(), kv, None) };
                l.unwrap().to_host().data().to_vec()
            };
            let mut l = Vec::new();
            for chunk in prompt.chunks(512) {
                l = forward(chunk, &mut kv);
            }
            let mut next = argmax(&l);
            let (mut tokens, mut all) = (vec![next], vec![l]);
            for _ in 0..steps {
                let l = forward(&[next], &mut kv);
                next = argmax(&l);
                tokens.push(next);
                all.push(l);
            }
            (tokens, all)
        };
        let t = std::time::Instant::now();
        let (host_tokens, host_logits) = run(false);
        let host_s = t.elapsed().as_secs_f64();
        assert_eq!(m.chain.runs.load(Ordering::Relaxed), 0, "the host path never chains");
        let t = std::time::Instant::now();
        let (chain_tokens, chain_logits) = run(true);
        let chain_s = t.elapsed().as_secs_f64();
        let runs = m.chain.runs.load(Ordering::Relaxed);
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        let each: Vec<f64> = host_logits.iter().zip(&chain_logits).map(|(a, b)| cosine(a, b)).collect();
        let worst = each.iter().copied().fold(1.0f64, f64::min);
        let first_diff = host_tokens.iter().zip(&chain_tokens).position(|(a, b)| a != b);
        eprintln!("prompt {n}: host {host_s:.2} s, chained {chain_s:.2} s ({runs} runs) for {steps} steps; prompt logits cosine {:.6}, worst {worst:.6}; first differing token {first_diff:?}", each[0]);
        assert!(runs > prompt.len().div_ceil(512), "the chain ran ({runs} runs)");
        assert_eq!(host_tokens, chain_tokens);
        assert!(worst >= 0.9999, "{worst}");
    }
}

#[cfg(test)]
mod dense_tests {
    use super::*;
    use llama_rs::Architecture as A;

    fn msgs() -> Vec<llama_rs::ChatMessage> {
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
        assert!(matches!(m[0].role, llama_rs::Role::System) && m[0].content == "Be brief.");
        assert!(matches!(m[1].role, llama_rs::Role::User) && m[1].content == "Capital of France?");
    }

    #[test]
    fn a_message_of_parts_keeps_their_text_and_an_image_is_refused_not_dropped() {
        let parse = |s: &[u8]| oaiy_engine::json::Json::parse(s).unwrap();
        let m = llama_messages(&[parse(br#"{"role":"user","content":[{"type":"text","text":"Capital"},{"type":"text","text":"of France?"}]}"#)]).unwrap();
        assert_eq!(m[0].content, "Capital\n\nof France?");
        let m = llama_messages(&[parse(br#"{"role":"assistant","content":null}"#)]).unwrap();
        assert!(matches!(m[0].role, llama_rs::Role::Assistant) && m[0].content.is_empty());
        let e = llama_messages(&[
            parse(br#"{"role":"user","content":"look"}"#),
            parse(br#"{"role":"tool","content":[{"type":"text","text":"screen"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}"#),
        ])
        .unwrap_err();
        assert!(e.contains("text only") && e.contains("message 2"), "{e}");
    }
}
