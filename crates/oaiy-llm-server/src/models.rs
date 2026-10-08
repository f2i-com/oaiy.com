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

    /// Qwen3.8-Flash-Next without CUDA: its EXL3 matrices and experts on the WebGPU adapter while the weight budget
    /// holds them (the first layers' experts), the rest decoded on the CPU, everything else on the host. No PEFT
    /// adapters and no vision tower (both CUDA's); conversations set aside in host RAM as on CUDA.
    #[cfg(feature = "webgpu")]
    fn load_flashnext_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        // Every adapter for the alias, applied together, each at its strength (else the alias's).
        let default_strength = o.lora_strengths.get(&spec.name).copied().unwrap_or(1.0);
        let base = crate::flashnext::lora_base(&spec.path)?;
        let adapters = o
            .lora_adapters
            .get(&spec.name)
            .map(|list| list.iter().map(|(path, strength)| crate::lora::Adapter::open_for(path, &base).map(|a| a.with_strength(strength.unwrap_or(default_strength)))).collect::<Result<Vec<_>>>())
            .transpose()?
            .unwrap_or_default();
        let devices = self.devices_of(spec);
        let picked = crate::backend::open(o, &devices)?;
        self.say(format!("{} runs on {} (EXL3 experts decoded in the matmul, a layer's in two batches)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        // The other GPUs it may use (the rest of its devices; every other discrete one where none is named) take a
        // share of the layers, a whole layer each (its experts too): two 32 GB cards hold all of its 46 GB of
        // experts, where the CPU decoded what one could not (most of a step).
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = wgpu.map(|b| crate::backend::others(o, &devices, b)).unwrap_or_default().into_iter().map(Arc::new).collect();
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
        let tower_backend = backends.last().map(Arc::clone);
        let model = crate::flashnext::load_portable_with(&spec.path, backends, &adapters, &packed, &experts, o.mtp.contains(&spec.name)).map_err(|e| match adapters.is_empty() {
            true => e,
            // (an adapter's target nothing took: the routed experts' are not applied on WebGPU yet)
            false => Error::Arg(format!("{}: {e} (a LoRA that adapts the routed experts is not applied on WebGPU yet; one for the attention and dense projections is)", spec.name)),
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
        for (d, g) in gpus.iter().enumerate() {
            let (used, budget) = g.usage();
            self.say(format!("{}: {:.1} GB of EXL3 weights on GPU {d} ({} at {}, budget {:.0} GB), the rest on the CPU", spec.name, used as f64 / 1e9, g.adapter().name, g.adapter().pci_bus_id, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer.clone());
        // The cache is on the host: bound it as the other portable models are.
        let max_seq = self.context(model.config.context_length).min(16384);
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Flash-Next tokenizer lacks image_pad".into()))?;
        // Its vision tower, where the alias is given one: on the last GPU (the first carries the head's side of the
        // text model), the matrices the checkpoint stores as EXL3 packed there.
        let vision_path = o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten();
        let projector = match (vision_path, tower_backend) {
            (Some(path), Some(backend)) => {
                let last = gpus.last().copied();
                let make = |d: ggml_rs::exl3::Exl3Data| match last {
                    Some(g) => g.exl3(d),
                    None => ggml_rs_wgpu::exl3::exl3_cpu(d),
                };
                let mm = crate::qwen_vision::load(path, backend, model.config.hidden, Some(&make))?;
                cfg.qwen_vision = Some(mm.config().clone());
                self.say(format!("{}: Qwen vision tower loaded (576 tokens an image)", spec.name));
                Some(mm)
            }
            _ => None,
        };
        let (jobs, rx) = std::sync::mpsc::channel();
        let park = crate::qwen_park::budget(o.park_gb, ggml_rs_wgpu::host_memory().map(|(free, _)| free as u64));
        let e = crate::qwen::QwenEngine::new(crate::qwen::Hybrid::Flash(Box::new(model)), projector, max_seq, !o.quiet && !o.silent).park_up_to(park);
        let thread = std::thread::Builder::new().name("flashnext-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok)) })
    }

    /// DeepSeek-V4.1 without CUDA: the CPU model (`dsv41::model`, the reference the CUDA path is tested against), its
    /// experts streamed through the host cache from RAM and the drive, and its dense trunk (attention projections,
    /// shared experts, router, head: about 9.7 GB) on the WebGPU adapter while the budget holds it. No images and no
    /// observer (both CUDA's); a conversation's next turn continues the state rather than reading it all again.
    #[cfg(feature = "webgpu")]
    fn load_deepseek_portable(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let tok = Arc::new(dsv41::tokenizer::Tokenizer::load(&spec.path)?);
        // As the CUDA build: the server's default context unless one is asked for.
        let max_seq = if o.ctx == 0 { crate::DEFAULT_CTX } else { o.ctx };
        let engram_meta = o.engram_meta.clone().unwrap_or_else(|| spec.path.join("engram_meta.safetensors"));
        let opts = dsv41::model::ModelOptions { max_seq, expert_cache_bytes: o.expert_cache_bytes() as usize, direct_io: true };
        self.say(format!("expert host cache: {:.2} GiB", opts.expert_cache_bytes as f64 / (1u64 << 30) as f64));
        let mut model = dsv41::model::Model::load(&spec.path, &engram_meta, &opts)?;
        // The routed experts on the CPU through this CPU's fastest kernel (the same bits as the portable one).
        model.set_expert_row_kernel(dsv41_simd::row_kernel());
        self.say(format!("experts on the CPU: the {} kernel", dsv41_simd::row_kernel_name()));
        // Its usage profile (`--usage`), if an earlier run left one: the experts' counts of uses, and with them the
        // order the tiers are filled in at this start (the most used on the cards, the next in RAM), where without
        // one they are filled by number and find their order as they are used.
        let order = o.usage.as_deref().filter(|p| crate::dsv41_portable::read_usage(p, model.expert_uses())).map(|p| {
            self.say(format!("the experts' usage profile read from {}: the most used are read first", p.display()));
            model.expert_uses().order()
        });
        let devices = self.devices_of(spec);
        let picked = crate::backend::open(o, &devices)?;
        let mut kernel = None;
        match picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
            Some(b) => {
                let (count, bytes) = crate::dsv41_portable::offload(&mut model, b);
                // A prompt's busy experts there too, through record slots in what the budget has left after the trunk,
                // and the experts used most kept there in what is left after those (a decode step's computed there while
                // the CPU reads and computes the rest).
                let experts = match crate::dsv41_portable::WgpuExperts::new(b) {
                    Some(mut k) => {
                        let (n, kept) = (k.slots(), k.tier().0);
                        // The other GPUs it may use (the rest of its devices; every other discrete one where none is
                        // named; OAIY_NO_SPLIT: none) hold a share of the experts for good.
                        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = crate::backend::others(o, &devices, b).into_iter().map(Arc::new).collect();
                        let pinned = k.pin_on(&others, model.cfg.n_layers, model.cfg.n_routed_experts);
                        // An expert in one place: the 15,360 are 290 GB, and what a card holds RAM need not. The cards
                        // then change what they hold only while the server idles (OAIY_DSV41_INCLUSIVE: as before, the
                        // first card's tier replacing as it goes and RAM holding its experts too).
                        k.set_exclusive(std::env::var_os("OAIY_DSV41_INCLUSIVE").is_none());
                        if let Some(order) = &order {
                            k.start_from(order);
                        }
                        let k = Arc::new(k);
                        model.set_experts_kernel(Some(Arc::clone(&k) as Arc<dyn dsv41::expert::ExpertsKernel>));
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
        let e = crate::dsv41_portable::Engine::new(model, Arc::clone(&tok), !o.quiet && !o.silent).with_kernel(kernel).with_usage(o.usage.clone(), order).with_ram_ceiling(o.ram_gb);
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
            .map(|list| list.iter().map(|(path, strength)| crate::lora::Adapter::open(path).map(|a| a.with_strength(strength.unwrap_or(default_strength)))).collect::<Result<Vec<_>>>())
            .transpose()?
            .unwrap_or_default();
        let picked = crate::backend::open(o, &self.devices_of(spec))?;
        self.say(format!("{} runs on {} (EXL3, decoded in the matmul)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        let packed = |data: ggml_rs::exl3::Exl3Data| match wgpu {
            Some(b) => b.exl3(data),
            None => ggml_rs_wgpu::exl3::exl3_cpu(data),
        };
        let model = crate::orcasaq::load_portable_with(&spec.path, Arc::clone(&picked.backend), &packed, &adapters)?;
        for (adapter, (path, strength)) in adapters.iter().zip(o.lora_adapters.get(&spec.name).into_iter().flatten()) {
            self.say(format!("OrcaSAQ: loaded LoRA {} for {} text projections (strength {})", path.display(), adapter.len(), strength.unwrap_or(default_strength)));
        }
        if let Some((used, budget)) = wgpu.map(|b| b.usage()) {
            self.say(format!("{}: {:.1} GB of EXL3 weights on the GPU (budget {:.0} GB)", spec.name, used as f64 / 1e9, budget as f64 / 1e9));
        }
        let tok = Arc::new(model.tokenizer().clone());
        // The cache is on the host: bound it as the other portable models are.
        let max_seq = self.context(model.config().context_length).min(16384);
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Orca tokenizer lacks image_pad".into()))?;
        // Its vision tower, where the alias is given one (the checkpoint's `vision` folder): on the model's backend.
        let vision_path = o.vision.then(|| o.vision_projectors.get(&spec.name)).flatten();
        let projector = match vision_path {
            Some(path) => {
                let mm = crate::qwen_vision::load(path, Arc::clone(&picked.backend), model.config().embedding_dim, None)?;
                cfg.qwen_vision = Some(mm.config().clone());
                self.say(format!("{}: original Qwen vision tower loaded (576 tokens an image)", spec.name));
                Some(mm)
            }
            None => None,
        };
        let (jobs, rx) = std::sync::mpsc::channel();
        let mut e = crate::qwen::QwenEngine::new(model, projector, max_seq, !o.quiet && !o.silent);
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
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok)) })
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
        let picked = crate::backend::open(o, &devices)?;
        self.say(format!("{} runs on {} (a GGUF: each matrix in its file's type)", spec.name, picked.label));
        let wgpu = picked.backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>();
        // (its other GPUs: the rest of its devices, or every other discrete one where none is named; OAIY_NO_SPLIT:
        // the one GPU, as for the dense models: what does not fit there runs on the host)
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = wgpu.map(|b| crate::backend::others(o, &devices, b)).unwrap_or_default().into_iter().map(Arc::new).collect();
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
        let max_seq = self.context(model.config.context_length).min(16384);
        let mut cfg = self.base_cfg(spec, max_seq);
        cfg.image_token_id = tok.token_id("<|image_pad|>").ok_or_else(|| Error::Arg("Flash-Next tokenizer lacks image_pad".into()))?;
        let (jobs, rx) = std::sync::mpsc::channel();
        let park = crate::qwen_park::budget(o.park_gb, ggml_rs_wgpu::host_memory().map(|(free, _)| free as u64));
        let e = crate::qwen::QwenEngine::new(crate::qwen::Hybrid::Flash(Box::new(model)), None, max_seq, !o.quiet && !o.silent).park_up_to(park);
        let thread = std::thread::Builder::new().name("flashnext-model".into()).spawn(move || e.run(rx))?;
        Ok(Live { name: spec.name.clone(), jobs, thread, cfg: Arc::new(cfg), flavour: Arc::new(Flavour::Qwen(tok)) })
    }

    // ---------------------------------------------------------------- GGUF
    fn load_gguf(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let path = Kind::gguf_path(&spec.path)?;
        #[cfg(feature = "webgpu")]
        if crate::flashnext::gguf_file::detect(&path) {
            return self.load_flashnext_gguf(spec, &path);
        }
        // The GPUs it may use: an image model's controller the one its configuration names, else the model's own
        // list or the server's; the first carries it.
        let devices = self.image_config.as_ref().filter(|c| c.controller_name == spec.name).map(|c| vec![c.controller_device]).unwrap_or_else(|| self.devices_of(spec));
        let picked = crate::backend::open(o, &devices)?;
        self.say(format!("{} runs on {}", spec.name, picked.label));
        let backend: Arc<dyn ggml_rs::Backend> = Arc::clone(&picked.backend);
        let gguf = gguf::GgufFile::open(&path).map_err(|e| Error::Arg(e.to_string()))?;
        if gguf.get_str("general.architecture").ok() == Some("qwen35") {
            // With a second GPU, a prompt's later layers go to it (copies of their weights and the head): each
            // chunk's first layers run on one as the other runs the chunk before's. The card the host's bytes reach
            // faster carries the model (it moves the more: the cache's rows, the steps), the other the later layers.
            #[allow(unused_mut)]
            let (mut backend, mut helper): (Arc<dyn ggml_rs::Backend>, Option<Arc<dyn ggml_rs::Backend>>) = (backend, None);
            #[cfg(feature = "webgpu")]
            if let Some(b) = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
                if let Some(other) = crate::backend::others(o, &devices, b).into_iter().next() {
                    let (mine, theirs) = (b.transfer_rates().0, other.transfer_rates().0);
                    let (bus, other_bus) = (b.adapter().pci_bus_id.clone(), other.adapter().pci_bus_id.clone());
                    let other: Arc<dyn ggml_rs::Backend> = Arc::new(other);
                    let swap = theirs > mine * 1.2;
                    let (main_at, main_rate, helper_at, helper_rate) = if swap { (other_bus, theirs, bus, mine) } else { (bus, mine, other_bus, theirs) };
                    self.say(format!("{}: the GPU at {main_at} carries it ({main_rate:.0} GB/s from the host), the one at {helper_at} a prompt's later layers ({helper_rate:.0} GB/s)", spec.name));
                    helper = Some(if swap { std::mem::replace(&mut backend, other) } else { other });
                }
            }
            let mut model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))?;
            // Its multi-token-prediction layer, where asked for: it drafts tokens a chained run checks.
            if o.mtp.contains(&spec.name) {
                if let llama_rs::Model::Qwen35(m) = &mut model {
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
            if let (llama_rs::Model::Qwen35(m), Some(h)) = (&model, helper) {
                m.chain.split_onto(h);
            }
            // its chained runs' kernels compiled before the first request waits on them
            if let llama_rs::Model::Qwen35(m) = &model {
                let warm = std::time::Instant::now();
                if m.warm_up() {
                    self.say(format!("{}: its chained runs ready in {:.1} s", spec.name, warm.elapsed().as_secs_f64()));
                }
            }
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
        #[allow(unused_mut)]
        let mut model = llama_rs::Model::open_streaming(&path, backend, budget)
            .map_err(|e| Error::Arg(e.to_string()))?;

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
/// Qwen3.8-Flash-Next from the GGUF at `path` over `gpus` (the devices `backends` are, in order): each quantized matrix
/// and each matrix of floats a chain reads packed on its layer's device, a layer's experts there while the device's
/// budget holds them beside its share of the dense matrices (and the prediction layer, where one is asked for), else
/// on the host (a chain runs those between the layer's submits: one RTX 5090 holds some 35 of the Q2_0 file's 48
/// layers' experts).
#[cfg(feature = "webgpu")]
pub(crate) fn flashnext_gguf_on(path: &Path, gpus: &[&ggml_rs_wgpu::WgpuBackend], backends: Vec<Arc<dyn ggml_rs::Backend>>, mtp_from: Option<&Path>) -> Result<crate::flashnext::FlashNext> {
    use crate::flashnext::gguf_file::{dense_bytes, load, ExpertBlocks};
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
    let mtp_from = mtp_from.filter(|_| whole + mtp_bytes + ((gpus.len() as u64) << 30) <= room || std::env::var_os("OAIY_DRAFT_OVER_HOST_LAYERS").is_some());
    let reserve = dense_bytes(path)? / backends.len().max(1) as u64 + (1 << 30) + if mtp_from.is_some() { mtp_bytes } else { 0 };
    // Each device's experts by an allowance of their own: its budget less that reserve, counted as they are placed
    // (the layers load four at a time, each with its dense matrices: the device's own count of its weights already
    // holds the dense ones loaded so far, and the reserve taken off that left a card 3.8 GB short of its budget).
    let allowance: Vec<u64> = gpus.iter().map(|g| g.usage().1.saturating_sub(g.usage().0).saturating_sub(reserve)).collect();
    let placed: Vec<std::sync::atomic::AtomicU64> = gpus.iter().map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let experts = |device: usize, e: ExpertBlocks| -> Result<Box<dyn ggml_rs::exl3::Experts>> {
        use std::sync::atomic::Ordering::Relaxed;
        let data = ggml_rs_wgpu::quant_moe::QuantExpertsData { hidden: e.hidden, ff: e.ff, experts: e.experts, gate: e.gate, up: e.up, down: e.down, shared: e.shared };
        match gpus.get(device) {
            Some(b) => {
                let bytes = (data.gate.1.len() + data.up.1.len() + data.down.1.len() + 3 * data.hidden * data.ff * 4) as u64;
                if placed[device].fetch_add(bytes, Relaxed) + bytes <= allowance[device] {
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

    /// Qwen3.8-Flash-Next from a GGUF (FLASHNEXT_GGUF, its first shard) chained on the GPUs answers as its own host
    /// path does (the same weights through the host's matmuls and its experts' reference): the prompt as one chunk,
    /// then step by step on the host path's greedy tokens, each step's logits close; and what it writes reads as an
    /// answer (printed: a wrong tensor or layout writes noise). Then the same tokens again as checks of two to four
    /// rows (as drafting runs them): each row's logits close to its step's (a check's rows go through the int8
    /// kernels where the type has them, a step's through f32's). FLASHNEXT_STEPS: the steps (48).
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next and its GGUF (FLASHNEXT_GGUF)"]
    fn a_flashnext_gguf_chained_answers_as_its_own_path() {
        use std::sync::Arc;
        let Ok(path) = std::env::var("FLASHNEXT_GGUF") else { return };
        let Ok(b0) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = if std::env::var_os("OAIY_NO_SPLIT").is_some() { Vec::new() } else { b0.others(None).into_iter().map(Arc::new).collect() };
        let b0 = Arc::new(b0);
        let gpus: Vec<&ggml_rs_wgpu::WgpuBackend> = std::iter::once(b0.as_ref()).chain(others.iter().map(|g| g.as_ref())).collect();
        let backends: Vec<Arc<dyn ggml_rs::Backend>> = std::iter::once(Arc::clone(&b0) as Arc<dyn ggml_rs::Backend>).chain(others.iter().map(|g| Arc::clone(g) as Arc<dyn ggml_rs::Backend>)).collect();
        let clock = std::time::Instant::now();
        let model = super::flashnext_gguf_on(std::path::Path::new(&path), &gpus, backends, None).unwrap();
        eprintln!("loaded in {:.1} s over {} GPUs: {}", clock.elapsed().as_secs_f64(), gpus.len(), gpus.iter().map(|g| format!("{:.1} GB", g.usage().0 as f64 / 1e9)).collect::<Vec<_>>().join(", "));
        let steps: usize = std::env::var("FLASHNEXT_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        let text = "<|im_start|>user\nWrite a short story about a cat called Moss who lives on a boat.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let prompt: Vec<u32> = model.tokenizer.encode(text, false).unwrap();
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
        let mut next = argmax(lh.data());
        let (mut worst, mut same, mut th, mut tc) = (1.0f64, 0usize, 0f64, 0f64);
        let mut written = vec![next];
        // (each chained step's logits, for the checks after)
        let mut stepped: Vec<Vec<f32>> = Vec::with_capacity(steps);
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
            written.push(next);
            stepped.push(chained.data().to_vec());
        }
        let runs = model.chain_runs();
        eprintln!("{steps} steps: host {:.1} ms a step, chained {:.1} ms ({runs} chained); worst logits cosine {worst:.6}; the same greedy token {same} of {steps}", th * 1e3 / steps as f64, tc * 1e3 / steps as f64);
        eprintln!("it writes: {:?}", model.tokenizer.decode(&written));
        assert!(prompt_cos > 0.99, "{prompt_cos}");
        assert!(worst > 0.99, "{worst}");
        assert_eq!(runs, steps + 1, "the prompt and every step chained");
        // the same tokens as checks: row r of a check at token `at` against the step on token `at + r`
        let mut kk = model.new_kv_cache(prompt.len() + steps + 64);
        let _ = model.forward(&prompt, &e, &mut kk, None).unwrap();
        let (mut at, mut rows_seen, mut same, mut worst, mut sum) = (0usize, 0usize, 0usize, 1.0f64, 0f64);
        for rows in [4usize, 3, 2].into_iter().cycle() {
            if at + rows > steps {
                break;
            }
            let got = model.check(&written[at..at + rows], &mut kk).expect("a check chained");
            for (r, row) in got.iter().enumerate() {
                let want = &stepped[at + r];
                let c = cosine(row.data(), want);
                worst = worst.min(c);
                sum += c;
                same += (argmax(row.data()) == argmax(want)) as usize;
                rows_seen += 1;
            }
            at += rows;
        }
        // (a rounding that changes a row's tenth expert in some layer moves its logits more than the rounding does:
        // the steps against the host path are as far apart at worst)
        eprintln!("{rows_seen} rows in checks of 4, 3 and 2 against their steps: logits cosine {:.6} on average, {worst:.6} at worst; the same greedy token {same} of {rows_seen}", sum / rows_seen.max(1) as f64);
        assert!(rows_seen == 0 || worst > 0.99, "a check's rows against its steps': {worst}");
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
        // (FLASHNEXT_LORA: a PEFT adapter's folder, applied as the weights load: the chain runs each adapted
        // projection with its update beside it, and must answer as the host's path does with the same adapter)
        let adapters: Vec<crate::lora::Adapter> = std::env::var("FLASHNEXT_LORA")
            .ok()
            .map(|dir| vec![crate::lora::Adapter::open_for(std::path::Path::new(&dir), &crate::flashnext::lora_base(p).unwrap()).unwrap()])
            .unwrap_or_default();
        let model = crate::flashnext::load_portable_with(p, backends, &adapters, &packed, &experts, false).unwrap();
        if let Some(a) = adapters.first() {
            eprintln!("with a LoRA for {} projections", a.len());
        }
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

    /// Qwen3.8-Flash-Next's prompt chunks run together (each chunk's first device's layers as the last device runs the
    /// chunk before's) answer as chunks run one at a time: a prompt of 4 chunks of 512 and one of 100, the last logits
    /// and steps after it bit for bit (the caches the same), and how long each takes. FLASHNEXT_MODEL: the checkpoint.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next (FLASHNEXT_MODEL); run with --nocapture"]
    fn flashnext_chunks_run_together_answer_as_one_at_a_time() {
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
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, false).unwrap();
        let tokens: Vec<u32> = (0..2148u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let spans = [(0usize, 512usize), (512, 1024), (1024, 1536), (1536, 2048), (2048, 2148)];
        let embeds: Vec<ggml_rs::Tensor> = spans.iter().map(|&(a, z)| model.embed_text(&tokens[a..z]).unwrap()).collect();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        // twice each way (the first warms the kernels)
        let mut results = Vec::new();
        for round in 0..2 {
            for together in [false, true] {
                let mut kv = model.new_kv_cache(4096);
                let t = std::time::Instant::now();
                let last = if together {
                    let chunks: Vec<(&[u32], &ggml_rs::Tensor)> = spans.iter().zip(&embeds).map(|(&(a, z), e)| (&tokens[a..z], e)).collect();
                    let mut n = 0;
                    let l = model.forward_chunks(&chunks, &mut kv, &mut |_| n += 1).expect("the chunks chained");
                    assert_eq!(n, spans.len(), "every chunk said");
                    l
                } else {
                    let mut l = None;
                    for (&(a, z), e) in spans.iter().zip(&embeds) {
                        l = Some(model.forward(&tokens[a..z], e, &mut kv, None).unwrap());
                    }
                    l.unwrap()
                };
                let secs = t.elapsed().as_secs_f64();
                // steps after it: the caches the same
                let mut next = argmax(last.data());
                let mut steps = Vec::new();
                for _ in 0..8 {
                    let e = model.embed_text(&[next]).unwrap();
                    let l = model.forward(&[next], &e, &mut kv, None).unwrap();
                    next = argmax(l.data());
                    steps.push(bits(l.data()));
                }
                eprintln!("round {round}, {}: {} tokens in {:.0} ms ({:.0} tokens a second)", if together { "together" } else { "one at a time" }, tokens.len(), secs * 1e3, tokens.len() as f64 / secs);
                results.push((bits(last.data()), steps, kv.len));
            }
        }
        for pair in results.chunks(2) {
            assert_eq!(pair[0].2, pair[1].2, "the same rows in the caches");
            assert!(pair[0].0 == pair[1].0, "the prompt's logits bit for bit");
            assert!(pair[0].1 == pair[1].1, "the steps' logits after it bit for bit");
        }
    }

    /// Qwen3.8-Flash-Next chained past QSA's dense span (FLASHNEXT_MODEL) answers as its own path does: a prompt of some
    /// Qwen3.8-Flash-Next's prompt whose chunks keep its checkpoints' states from inside them (FLASHNEXT_MODEL;
    /// FLASHNEXT_LEN tokens, 1,300 unless asked) against the runs that stop where it keeps them, before the prompt's
    /// last 7 tokens and before its last: the delta nets' states and conv windows and the n-gram layer's window as
    /// close as those runs' (the same rows through the same kernels where the stopped run's last chunk is long; a
    /// run of few rows' kernels apart else), the n-gram layer's history the same ids, the logits as close; and a cache
    /// put back to either kept state and run on from there gives the run's own logits again.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next (FLASHNEXT_MODEL); run with --nocapture"]
    fn flashnext_chunks_keep_the_states_their_stops_would() {
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
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, false).unwrap();
        let n: usize = std::env::var("FLASHNEXT_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1300);
        let tokens: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let taps = [n - 7, n - 1];
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (norm(a) * norm(b)).max(1e-30)
        };
        // (the same token, or one as good to a tenth of a logit: these prompts' tokens are no text)
        let agree = |a: &[f32], b: &[f32]| argmax(a) == argmax(b) || (b[argmax(b)] - b[argmax(a)]).abs() < 0.1;
        let host = |ts: &[Option<ggml_rs::Tensor>]| -> Vec<Option<Vec<f32>>> { ts.iter().map(|t| t.as_ref().map(|t| t.to_host().data().to_vec())).collect() };
        // a prompt's rows as the server runs them: chunks of the model's own, together
        let most = model.prompt_rows();
        let run = |from: usize, to: usize, kv: &mut llama_rs::KvCache, taps: &[usize]| {
            let spans: Vec<(usize, usize)> = (from..to).step_by(most).map(|a| (a, (a + most).min(to))).collect();
            let embeds: Vec<ggml_rs::Tensor> = spans.iter().map(|&(a, z)| model.embed_text(&tokens[a..z]).unwrap()).collect();
            let chunks: Vec<(&[u32], &ggml_rs::Tensor)> = spans.iter().zip(&embeds).map(|(&(a, z), e)| (&tokens[a..z], e)).collect();
            let (logits, kept) = model.forward_chunks_tapped(&chunks, kv, &mut |_| {}, taps);
            (logits.expect("the chunks chained").to_host(), kept)
        };
        // (the kernels warmed)
        {
            let mut kv = model.new_kv_cache(4096);
            run(0, n - 7, &mut kv, &[]);
            run(n - 7, n, &mut kv, &[]);
        }
        // the runs that stop
        let mut kv = model.new_kv_cache(4096);
        let t = std::time::Instant::now();
        run(0, n - 7, &mut kv, &[]);
        let first = (host(&kv.ssm_state), host(&kv.ssm_conv));
        run(n - 7, n - 1, &mut kv, &[]);
        let second = (host(&kv.ssm_state), host(&kv.ssm_conv));
        let (stopped, _) = run(n - 1, n, &mut kv, &[]);
        let ms_stopped = t.elapsed().as_secs_f64() * 1e3;
        // the chunks that keep them
        let mut kv = model.new_kv_cache(4096);
        assert!(model.can_tap(n, &kv, &taps), "chunks of {n} rows keep their states after {taps:?}");
        let t = std::time::Instant::now();
        let (logits, kept) = run(0, n, &mut kv, &taps);
        let ms_kept = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(kept.iter().map(|k| k.at).collect::<Vec<_>>(), taps, "the states' places");
        assert_eq!(kv.len, n, "the cache's rows");
        // (a short stopped chunk's rows are through other kernels than a long one's)
        let before = (n - 7) % most;
        for (i, (tap, (states, convs))) in kept.iter().zip([&first, &second]).enumerate() {
            let (mut worst, mut count) = (1f64, 0);
            for (got, want) in host(&tap.states).iter().zip(states).chain(host(&tap.convs).iter().zip(convs)).map(|(g, w)| (g.as_ref(), w.as_ref())) {
                assert_eq!(got.is_some(), want.is_some(), "a slot's state where the cache has one");
                if let (Some(g), Some(w)) = (got, want) {
                    assert_eq!(g.len(), w.len(), "a state's size");
                    // (the n-gram layer's history: token ids, the same ones)
                    if g.len() < 16 {
                        assert_eq!(g, w, "the n-gram layer's history");
                    } else {
                        worst = worst.min(cosine(g, w));
                    }
                    count += 1;
                }
            }
            eprintln!("the states kept after {} rows against the run's that stops there: the worst cosine {worst:.7}, {count} of them", tap.at);
            assert!(worst > if i == 0 && (before == 0 || before >= 128) { 0.9999 } else { 0.98 }, "the states after {} rows: {worst}", tap.at);
        }
        // The same chunks keeping nothing: the same kernels over the same rows but the recurrences of the rows after a
        // kept state, which go a few rows at a time (their own kernels, where a chunk's rows go through the scan's):
        // the same logits to those kernels' rounding, which a prompt of a few tokens that are no text makes the most
        // of (28 of them: the delta nets' states the same to five places through 17 layers, 0.997 by the last).
        let mut whole_kv = model.new_kv_cache(4096);
        let (whole, _) = run(0, n, &mut whole_kv, &[]);
        let cw = cosine(logits.data(), whole.data());
        // (where the two differ, slot by slot: the states at the end)
        if std::env::var_os("FLASHNEXT_TAPS_DEBUG").is_some() {
            let (a, b) = ((host(&kv.ssm_state), host(&kv.ssm_conv)), (host(&whole_kv.ssm_state), host(&whole_kv.ssm_conv)));
            for (what, x, y) in [("state", &a.0, &b.0), ("conv", &a.1, &b.1)] {
                let line: Vec<String> = x.iter().zip(y).enumerate().filter_map(|(i, (g, w))| Some((i, g.as_ref()?, w.as_ref()?))).map(|(i, g, w)| format!("{i}:{:.5}", cosine(g, w))).collect();
                eprintln!("the end's {what}s, kept against not: {}", line.join(" "));
            }
        }
        let c = cosine(logits.data(), stopped.data());
        eprintln!("{n} tokens: the runs that stop {ms_stopped:.0} ms (this test reading their states to the host between them), the chunks that keep their states {ms_kept:.0} ms; the logits' cosine with the chunks that keep none {cw:.7}, with the runs that stop {c:.6} (the same token {})", argmax(logits.data()) == argmax(stopped.data()));
        assert!(cw > if before == 0 || before >= 128 { 0.9999 } else { 0.98 }, "the logits against the chunks that keep nothing: {cw}");
        // (the runs that stop are runs of few rows at the end, other kernels: a prompt of a few tokens is all such)
        assert!(c > 0.98 && (c < 0.999 || agree(logits.data(), stopped.data())), "the logits against the stopped runs': {c}");
        // a cache put back to a kept state runs on to the same logits
        for tap in kept.iter().rev() {
            for i in 0..kv.ssm_state.len() {
                let backend = kv.layer_backends[i].clone();
                kv.ssm_state[i] = tap.states[i].as_ref().map(|t| backend.to_device(t.to_host()));
                kv.ssm_conv[i] = tap.convs[i].as_ref().map(|t| backend.to_device(t.to_host()));
            }
            kv.len = tap.at;
            let (again, _) = run(tap.at, n, &mut kv, &[]);
            let c = cosine(again.data(), logits.data());
            eprintln!("put back to the state after {} rows and run on: the logits' cosine {c:.6}, the same token {}", tap.at, argmax(again.data()) == argmax(logits.data()));
            assert!(c > 0.98 && (c < 0.999 || agree(again.data(), logits.data())), "run on from the state after {} rows: {c}", tap.at);
        }
    }

    /// Qwen3.8-Flash-Next's prompt in chunks of 1,024 run together (FLASHNEXT_MODEL, with OAIY_FN_ROWS=1024: 512 is
    /// the most otherwise, and there is nothing to compare) answers as in chunks of 512 one at a time: the last logits to the kernels' rounding (the experts' blocks hold other rows, their sums in another
    /// order), the same token, and the steps after it the same tokens.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next (FLASHNEXT_MODEL); run with --nocapture"]
    fn flashnext_chunks_of_1024_answer_as_512s() {
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
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, false).unwrap();
        if model.prompt_rows() < 1024 {
            eprintln!("chunks of {} here: nothing to compare", model.prompt_rows());
            return;
        }
        let tokens: Vec<u32> = (0..2148u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let mut results = Vec::new();
        for (wide, spans) in [(false, vec![(0usize, 512usize), (512, 1024), (1024, 1536), (1536, 2048), (2048, 2148)]), (true, vec![(0, 1024), (1024, 2048), (2048, 2148)])] {
            // twice (the first warms the kernels)
            for round in 0..2 {
                let embeds: Vec<ggml_rs::Tensor> = spans.iter().map(|&(a, z)| model.embed_text(&tokens[a..z]).unwrap()).collect();
                let mut kv = model.new_kv_cache(4096);
                let t = std::time::Instant::now();
                let last = if wide {
                    let chunks: Vec<(&[u32], &ggml_rs::Tensor)> = spans.iter().zip(&embeds).map(|(&(a, z), e)| (&tokens[a..z], e)).collect();
                    model.forward_chunks(&chunks, &mut kv, &mut |_| {}).expect("the chunks chained")
                } else {
                    let mut l = None;
                    for (&(a, z), e) in spans.iter().zip(&embeds) {
                        l = Some(model.forward(&tokens[a..z], e, &mut kv, None).unwrap());
                    }
                    l.unwrap()
                };
                let secs = t.elapsed().as_secs_f64();
                let mut next = argmax(last.data());
                let mut picked = vec![next];
                for _ in 0..8 {
                    let e = model.embed_text(&[next]).unwrap();
                    let l = model.forward(&[next], &e, &mut kv, None).unwrap();
                    next = argmax(l.data());
                    picked.push(next);
                }
                eprintln!("{}, round {round}: {} tokens in {:.0} ms ({:.0} tokens a second)", if wide { "chunks of 1,024 together" } else { "chunks of 512 one at a time" }, tokens.len(), secs * 1e3, tokens.len() as f64 / secs);
                if round == 1 {
                    results.push((last.data().to_vec(), picked));
                }
            }
        }
        let (a, b) = (&results[0], &results[1]);
        let dot: f64 = a.0.iter().zip(&b.0).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&a.0) * norm(&b.0));
        let same = a.1.iter().zip(&b.1).take_while(|(x, y)| x == y).count();
        eprintln!("the last logits' cosine {cos:.6}; the same tokens for the first {same} of {}", a.1.len());
        assert!(cos > 0.999 && same >= 1, "chunks of 1,024 against 512s: cosine {cos}, {same} tokens the same");
    }

    /// 2,100 tokens in chunks of 512 (the last past 2,051 positions, its rows' blocks chosen on the GPU) against the
    /// host path's whole prompt, then steps on the host path's greedy tokens; each step's logits close and the same
    /// greedy token (bar near-ties), and a step's time there.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next (FLASHNEXT_MODEL); run with --nocapture"]
    fn a_chained_flashnext_attends_past_the_dense_span_as_its_own_path() {
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
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, false).unwrap();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        // a long prose of the market town, then a question about its start
        let para = "The river town kept its market on the north bank, where barges unloaded grain, wool and salt. Every spring the floods rose to the steps of the old customs house, and every summer the merchants rebuilt the stalls a little higher. The ferryman counted the seasons by the colour of the water. ";
        let mut text = String::from("<|im_start|>user\nThe ferryman's name was Aldous Penrose. ");
        while model.tokenizer.encode(&text, false).unwrap().len() < 2080 {
            text += para;
        }
        text += "\n\nWhat was the ferryman's name, and what did he count the seasons by?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let prompt: Vec<u32> = model.tokenizer.encode(&text, false).unwrap();
        let steps: usize = std::env::var("FLASHNEXT_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let (mut kh, mut kc) = (model.new_kv_cache(prompt.len() + steps + 8), model.new_kv_cache(prompt.len() + steps + 8));
        let t = std::time::Instant::now();
        let lh = model.forward_host(&prompt, &model.embed_text(&prompt).unwrap(), &mut kh, None).unwrap();
        let host_prompt = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let mut lc = None;
        for chunk in prompt.chunks(512) {
            lc = Some(model.forward(chunk, &model.embed_text(chunk).unwrap(), &mut kc, None).unwrap());
        }
        let chained_prompt = t.elapsed().as_secs_f64();
        let lc = lc.unwrap();
        let runs = model.chain_runs();
        eprintln!("a prompt of {} tokens: host {host_prompt:.1} s, chained {chained_prompt:.1} s ({runs} chunks chained); logits cosine {:.6}, the same greedy token {}", prompt.len(), cosine(lh.data(), lc.data()), argmax(lh.data()) == argmax(lc.data()));
        assert_eq!(runs, prompt.len().div_ceil(512), "every chunk chained");
        let mut next = argmax(lh.data());
        let (mut worst, mut same, mut th, mut tc) = (1.0f64, 0usize, 0f64, 0f64);
        let mut out = Vec::new();
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
            out.push(next);
        }
        eprintln!("{steps} steps past {} positions: host {:.1} ms a step, chained {:.1} ms; worst logits cosine {worst:.6}; the same greedy token {same} of {steps}", prompt.len(), th * 1e3 / steps as f64, tc * 1e3 / steps as f64);
        eprintln!("  {}", model.tokenizer.decode(&out).replace('\n', " "));
        assert_eq!(model.chain_runs(), runs + steps, "every step chained");
        assert!(worst > 0.99, "{worst}");
    }

    /// A chained Qwen3.8-Flash-Next check of a few tokens (FLASHNEXT_MODEL) gives each row's logits as steps on the same
    /// tokens do, bit for bit, and undone past its first rows leaves the state those rows would: a check of four of the
    /// steps' greedy tokens, rolled back to two, a check of the next three, a step, then checks of 2 to 4 rows each kept
    /// in part or whole (FLASHNEXT_CHECK_AT steps in; FLASHNEXT_DRAFT: with a draft before the first two); and what checks
    /// of 2 to 4 rows cost against a step.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next and its checkpoint (FLASHNEXT_MODEL); run with --nocapture"]
    fn a_flashnext_check_answers_as_its_steps_do() {
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
        // FLASHNEXT_DRAFT: with the prediction layer, a draft before each check (it changes nothing a check reads)
        let drafting = std::env::var_os("FLASHNEXT_DRAFT").is_some();
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, drafting).unwrap();
        let prompt: Vec<u32> = model.tokenizer.encode("Write a short story about a cat called Moss who lives on a boat.", false).unwrap();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        let step = |t: u32, kv: &mut llama_rs::KvCache| model.forward(&[t], &model.embed_text(&[t]).unwrap(), kv, None).unwrap();
        // the steps' greedy tokens g and their logits l (l[i] after g[i])
        let room = prompt.len() + 512 + std::env::var("FLASHNEXT_CHECK_AT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let (mut ks, mut kc) = (model.new_kv_cache(room), model.new_kv_cache(room));
        let e = model.embed_text(&prompt).unwrap();
        let mut g = vec![argmax(model.forward(&prompt, &e, &mut ks, None).unwrap().data())];
        let _ = model.forward(&prompt, &e, &mut kc, None).unwrap();
        // FLASHNEXT_CHECK_AT: that many greedy steps on both first
        let pre: usize = std::env::var("FLASHNEXT_CHECK_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for _ in 0..pre {
            let t = *g.last().unwrap();
            let _ = step(t, &mut kc);
            g = vec![argmax(step(t, &mut ks).data())];
        }
        let mut l: Vec<Vec<f32>> = Vec::new();
        for i in 0..40 {
            let out = step(g[i], &mut ks);
            g.push(argmax(out.data()));
            l.push(out.data().to_vec());
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let (mut rows_seen, mut exact) = (0, 0);
        let mut compare = |what: &str, rows: &[ggml_rs::Tensor], from: usize| {
            for (r, row) in rows.iter().enumerate() {
                let same_bits = bits(row.data()) == bits(&l[from + r]);
                rows_seen += 1;
                exact += same_bits as usize;
                if !same_bits {
                    eprintln!("{what} row {r}: cosine {:.6} against step {}, the same greedy token {}", cosine(row.data(), &l[from + r]), from + r, argmax(row.data()) == argmax(&l[from + r]));
                }
            }
        };
        if drafting {
            let _ = model.draft_above(&kc, &[g[0]], 3, 0.0);
        }
        let a = model.check(&g[0..4], &mut kc).expect("a check of four");
        compare("check A", &a, 0);
        model.rollback(&mut kc, 2);
        assert_eq!(kc.len, prompt.len() + pre + 2, "rolled back to its first two rows");
        if drafting {
            let _ = model.draft_above(&kc, &g[1..3], 3, 0.0);
        }
        let b = model.check(&g[2..5], &mut kc).expect("a check of three");
        compare("check B (after the rollback)", &b, 2);
        let after = step(g[5], &mut kc);
        assert_eq!(bits(after.data()), bits(&l[5]), "the step after the checks");
        // a run of checks as drafting makes them: (rows, kept), each from where the last left off
        let mut at = 6;
        for (rows, keep) in [(2usize, 2usize), (2, 1), (4, 4), (3, 2), (2, 2), (4, 1), (4, 3), (3, 3), (2, 1), (4, 4)] {
            let got = model.check(&g[at..at + rows], &mut kc).expect("a check");
            compare(&format!("check of {rows} keeping {keep} at {at}"), &got, at);
            model.rollback(&mut kc, keep);
            at += keep;
        }
        eprintln!("{exact} of {rows_seen} check rows a step's bit for bit");
        assert_eq!(exact, rows_seen, "every check row a step's");
        // what a check costs: each of 2, 3 and 4 rows, rolled back to its first, against a step
        let (mut kt, at) = (kc, g[6]);
        let time = |f: &mut dyn FnMut()| {
            f();
            let t = std::time::Instant::now();
            for _ in 0..8 {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / 8.0
        };
        let one = time(&mut || {
            let _ = step(at, &mut kt);
            kt.len -= 1;
        });
        let mut line = format!("a step {one:.1} ms;");
        for rows in 2..=4usize {
            let toks: Vec<u32> = (0..rows).map(|i| g[(6 + i) % g.len()]).collect();
            let ms = time(&mut || {
                let _ = model.check(&toks, &mut kt).unwrap();
                model.rollback(&mut kt, 1);
                kt.len -= 1;
            });
            line += &format!(" a check of {rows} {ms:.1} ms ({:.2} steps);", ms / one);
        }
        eprintln!("{line}");
    }

    /// Qwen3.8-Flash-Next drafting with its multi-token-prediction layer (FLASHNEXT_MODEL) answers as its steps do: greedy
    /// decoding as the server's loop drafts (three drafts a check, each taken while it is the token picked, the check's
    /// rows past it undone) gives the steps' tokens, every logits row it used a step's on the same tokens; and its time
    /// a token against steps'.
    #[test]
    #[ignore = "needs WebGPU adapters with room for Qwen3.8-Flash-Next and its layer (FLASHNEXT_MODEL); run with --nocapture"]
    fn a_drafting_flashnext_answers_as_its_steps_do() {
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
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, true).unwrap();
        assert!(model.warm_up() && model.drafts(), "the layer chained");
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        let gen: usize = std::env::var("FLASHNEXT_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(160);
        // FLASHNEXT_DRAFTS: drafts a check (the server's 3)
        let drafts_a_check: usize = std::env::var("FLASHNEXT_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        for prompt in [
            "<|im_start|>user\nList ten facts about the planet Mars, one a line.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>user\nWrite a short story about a cat called Moss who lives on a boat.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ] {
            let pt: Vec<u32> = model.tokenizer.encode(prompt, false).unwrap();
            let e = model.embed_text(&pt).unwrap();
            let step = |t: u32, kv: &mut llama_rs::KvCache| model.forward(&[t], &model.embed_text(&[t]).unwrap(), kv, None).unwrap();
            // plain steps
            let mut kv = model.new_kv_cache(pt.len() + gen + 16);
            let mut l = model.forward(&pt, &e, &mut kv, None).unwrap();
            let t0 = std::time::Instant::now();
            let mut plain = Vec::new();
            for _ in 0..gen {
                let next = argmax(l.data());
                plain.push(next);
                l = step(next, &mut kv);
            }
            let plain_s = t0.elapsed().as_secs_f64();
            // drafting, as the server's loop does it
            let mut kv = model.new_kv_cache(pt.len() + gen + 16);
            let mut logits = model.forward(&pt, &e, &mut kv, None).unwrap().data().to_vec();
            let mut seq = pt.clone();
            let mut used: Vec<Vec<f32>> = Vec::new();
            // where each logits row came from: a step (0), a check's first row (1), a check's row r + 1 (r + 2)
            let mut from: Vec<usize> = Vec::new();
            let mut src = 0usize;
            let mut pending: std::collections::VecDeque<(u32, ggml_rs::Tensor)> = Default::default();
            let (mut check_rows, mut checked, mut taken) = (0usize, 0usize, 0usize);
            let t0 = std::time::Instant::now();
            while seq.len() - pt.len() < gen {
                let next = argmax(&logits);
                used.push(logits.clone());
                from.push(src);
                let ran = pending.front().is_some_and(|(d, _)| *d == next);
                if !ran && !pending.is_empty() {
                    model.rollback(&mut kv, check_rows - pending.len());
                    pending.clear();
                }
                seq.push(next);
                if ran {
                    logits = pending.pop_front().unwrap().1.data().to_vec();
                    taken += 1;
                    src += 1;
                    continue;
                }
                let drafted = model.draft(&kv, &seq[seq.len().saturating_sub(crate::flashnext::CHECK_ROWS + 1)..], drafts_a_check).filter(|d| !d.is_empty());
                match drafted.and_then(|d| {
                    let rows: Vec<u32> = std::iter::once(next).chain(d.iter().copied()).collect();
                    Some((d, model.check(&rows, &mut kv)?))
                }) {
                    Some((d, mut rows)) => {
                        check_rows = rows.len();
                        checked += d.len();
                        logits = rows.remove(0).data().to_vec();
                        pending = d.into_iter().zip(rows).collect();
                        src = 1;
                    }
                    None => {
                        logits = step(next, &mut kv).data().to_vec();
                        src = 0;
                    }
                }
            }
            if !pending.is_empty() {
                model.rollback(&mut kv, check_rows - pending.len());
            }
            let draft_s = t0.elapsed().as_secs_f64();
            let out = &seq[pt.len()..];
            let same = out.iter().zip(&plain).take_while(|(a, b)| a == b).count();
            // the logits it used against steps on its own tokens
            let mut kv = model.new_kv_cache(pt.len() + gen + 16);
            let mut l = model.forward(&pt, &e, &mut kv, None).unwrap();
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
            let (mut worst, mut exact) = (1.0f64, 0usize);
            let mut each = Vec::new();
            for (i, &t) in out.iter().enumerate() {
                let c = cosine(l.data(), &used[i]);
                worst = worst.min(c);
                exact += (bits(l.data()) == bits(&used[i])) as usize;
                each.push((c, i, from[i]));
                l = step(t, &mut kv);
            }
            each.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if exact < out.len() {
                eprintln!("  the worst rows (cosine, token, from: 0 a step, 1 a check's first row, r + 1 its row r): {:?}", &each[..6.min(each.len())]);
            }
            eprintln!(
                "{gen} tokens: steps {:.1} ms a token, drafting {:.1} ms ({taken} of {checked} drafts taken); the same tokens for {same}; {exact} logits rows a step's bit for bit, the worst cosine {worst:.6}",
                plain_s * 1e3 / gen as f64,
                draft_s * 1e3 / gen as f64
            );
            eprintln!("  {}", model.tokenizer.decode(out).chars().take(160).collect::<String>().replace('\n', " "));
            assert_eq!(same, gen, "the same tokens");
            assert_eq!(exact, out.len(), "every logits row a step's, bit for bit");
        }
    }

    /// How often Qwen3.8-Flash-Next's multi-token-prediction layer (FLASHNEXT_MODEL) drafts what its trunk then picks:
    /// the trunk's greedy continuation of two prompts, then teacher-forced, the layer's three drafts at each position
    /// against the tokens after (each where the ones before it were), with the probability the layer gives each; what a
    /// draft costs, and the GPUs' memory with the layer.
    #[test]
    #[ignore = "a measurement; needs WebGPU adapters with room for Qwen3.8-Flash-Next and its layer; run with --nocapture"]
    fn measure_flashnext_mtp_acceptance() {
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
        let model = crate::flashnext::load_portable(p, backends, &packed, &experts, true).unwrap();
        for (d, g) in gpus.iter().enumerate() {
            let (used, budget) = g.usage();
            eprintln!("GPU {d}: {:.1} GB of {:.1} GB", used as f64 / 1e9, budget as f64 / 1e9);
        }
        assert!(model.drafts(), "the layer chained");
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let prompts = [
            "<|im_start|>user\nExplain how a refrigerator keeps food cold, in a few short paragraphs.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ];
        let gen = 96usize;
        let (mut hit, mut tried) = ([0usize; 3], [0usize; 3]);
        let (mut drafting, mut drafts_made) = (0f64, 0usize);
        for prompt in prompts {
            let pt: Vec<u32> = model.tokenizer.encode(prompt, false).unwrap();
            let mut kv = model.new_kv_cache(pt.len() + gen + 16);
            let mut l = model.forward(&pt, &model.embed_text(&pt).unwrap(), &mut kv, None).unwrap();
            let mut seq = pt.clone();
            // the trunk's greedy tokens, each step's drafts made before the next step
            let mut made: Vec<Vec<u32>> = Vec::new();
            for _ in 0..gen {
                let next = argmax(l.data());
                seq.push(next);
                let t = std::time::Instant::now();
                // every draft the layer makes, however unlikely
                let d = model.draft_above(&kv, &seq[seq.len() - 8..], 3, 0.0).expect("drafts after a step");
                drafting += t.elapsed().as_secs_f64();
                drafts_made += d.len();
                made.push(d);
                l = model.forward(&[next], &model.embed_text(&[next]).unwrap(), &mut kv, None).unwrap();
            }
            // OAIY_CHAIN_PROFILE: where a draft's time goes
            if ggml_rs_wgpu::profile::chain_on() {
                let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
                let t = std::time::Instant::now();
                let d = model.draft_above(&kv, &seq[seq.len() - 8..], 3, 0.0);
                let k = ggml_rs_wgpu::profile::take_kernels();
                let gpu: f64 = k.iter().map(|e| e.1).sum();
                eprintln!("a draft of {:?}: {:.1} ms, of it the GPU's kernels {gpu:.2} ms; {}", d.map(|d| d.len()), t.elapsed().as_secs_f64() * 1e3, ggml_rs_wgpu::profile::take_line());
                for (name, ms, n) in k.iter().take(12) {
                    eprintln!("  {name:<28} {ms:>9.3} ms {n:>6}");
                }
            }
            // drafts made with token i sampled guess tokens i + 1, i + 2, i + 3
            let out = &seq[pt.len()..];
            for (i, d) in made.iter().enumerate() {
                for (j, &dj) in d.iter().enumerate() {
                    if i + 1 + j >= out.len() {
                        break;
                    }
                    tried[j] += 1;
                    if dj != out[i + 1 + j] {
                        break;
                    }
                    hit[j] += 1;
                }
            }
            eprintln!("{}", model.tokenizer.decode(out).chars().take(200).collect::<String>().replace('\n', " "));
        }
        let rates: Vec<String> = (0..3).map(|j| format!("{:.2} ({} of {})", hit[j] as f64 / tried[j].max(1) as f64, hit[j], tried[j])).collect();
        eprintln!("drafts taken, by depth (each where the ones before it were): {}", rates.join(", "));
        eprintln!("a draft {:.2} ms", drafting * 1e3 / drafts_made.max(1) as f64);
        assert!(hit[0] * 2 > tried[0], "the first draft is the trunk's token at least half the time");
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
        // (OAIY_NO_SPLIT: one card, the layers it has no room for with their experts on the host)
        let others: Vec<Arc<ggml_rs_wgpu::WgpuBackend>> = if std::env::var_os("OAIY_NO_SPLIT").is_some() { Vec::new() } else { b0.others(None).into_iter().map(Arc::new).collect() };
        let b0 = Arc::new(b0);
        let gpus: Vec<&ggml_rs_wgpu::WgpuBackend> = std::iter::once(b0.as_ref()).chain(others.iter().map(|g| g.as_ref())).collect();
        let backends: Vec<Arc<dyn ggml_rs::Backend>> = std::iter::once(Arc::clone(&b0) as Arc<dyn ggml_rs::Backend>).chain(others.iter().map(|g| Arc::clone(g) as Arc<dyn ggml_rs::Backend>)).collect();
        type Make<'a> = Box<dyn Fn(ggml_rs::exl3::Exl3Data) -> std::result::Result<Arc<dyn ggml_rs::exl3::PackedLinear>, String> + Send + Sync + 'a>;
        let packed = |device: usize| -> Make<'_> {
            let b = gpus[device];
            Box::new(move |d| b.exl3(d))
        };
        let p = std::path::Path::new(&path);
        // (FLASHNEXT_GGUF: the model from that GGUF's first shard, where the EXL3 checkpoint's)
        let model = match std::env::var("FLASHNEXT_GGUF") {
            Ok(g) => super::flashnext_gguf_on(std::path::Path::new(&g), &gpus, backends, None).unwrap(),
            Err(_) => {
                let reserve = crate::flashnext::dense_exl3_bytes(p).unwrap() / backends.len() as u64 + (1 << 30);
                let experts = |device: usize, _layer: &str, list: Vec<[ggml_rs::exl3::Exl3Data; 3]>| -> oaiy_engine::Result<Box<dyn ggml_rs::exl3::Experts>> {
                    gpus[device].exl3_experts_leaving(list, reserve).map_err(oaiy_engine::Error::Arg)
                };
                crate::flashnext::load_portable(p, backends, &packed, &experts, false).unwrap()
            }
        };
        let tokens: Vec<u32> = (0..16_400u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let mut kv = model.new_kv_cache(16_384);
        let report = |what: &str, wall: f64| {
            let k = ggml_rs_wgpu::profile::take_kernels();
            let gpu: f64 = k.iter().map(|e| e.1).sum();
            let count: u64 = k.iter().map(|e| e.2 as u64).sum();
            eprintln!("{what}: {wall:.1} ms, of it the GPU's kernels {gpu:.1} ms in {count} dispatches; {}", ggml_rs_wgpu::profile::take_line());
            // OAIY_PROFILE_ALL: every kernel, not only the costliest
            for (name, ms, n) in k.iter().take(if std::env::var_os("OAIY_PROFILE_ALL").is_some() { usize::MAX } else { 14 }) {
                eprintln!("  {name:<28} {ms:>9.2} ms {n:>6}");
            }
        };
        let mut at = 0;
        // FLASHNEXT_PAST: chunks of 512 up to that many positions first (past QSA's dense span at 2,052)
        let past_span: usize = std::env::var("FLASHNEXT_PAST").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        // FLASHNEXT_CHUNK: the prompt's chunks that long (512 as the server sends them)
        let chunk: usize = std::env::var("FLASHNEXT_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let mut chunks: Vec<usize> = vec![chunk, chunk, 64];
        if past_span > 0 {
            chunks = std::iter::repeat_n(512, past_span / 512).chain([past_span % 512]).filter(|&n| n > 0).collect();
        }
        let chunk_count = chunks.len();
        for (i, n) in chunks.into_iter().enumerate() {
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
        // runs of a few rows (a prompt's last few: the last row's logits only): each a mean of 6 after one to warm
        let mut runs = chunk_count + 4;
        for rows in [1usize, 2, 3, 4, 8] {
            let mut wall = 0.0;
            for rep in 0..7 {
                let toks: Vec<u32> = (0..rows as u32).map(|i| 2000 + i * 31 + rep * 7).collect();
                let e = model.embed_text(&toks).unwrap();
                let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
                let t = Instant::now();
                let _ = model.forward(&toks, &e, &mut kv, None).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if rep > 0 {
                    wall += ms;
                }
                if rep == 6 && rows == 4 {
                    report("a run of 4 rows", ms);
                }
                runs += 1;
            }
            eprintln!("a run of {rows} rows: {:.1} ms", wall / 6.0);
        }
        // checks of drafts as the server makes them (`Flashnext::check`: every row's logits, the run kept undoable):
        // each a mean of 6 after one to warm
        for rows in 2..=crate::flashnext::CHECK_ROWS {
            let mut wall = 0.0;
            for rep in 0..7 {
                let toks: Vec<u32> = (0..rows as u32).map(|i| 3000 + i * 31 + rep * 7).collect();
                let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
                let t = Instant::now();
                let l = model.check(&toks, &mut kv).expect("a check chained");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                assert_eq!(l.len(), rows, "a row of logits for each of a check's");
                if rep > 0 {
                    wall += ms;
                }
                if rep == 6 && rows == 4.min(crate::flashnext::CHECK_ROWS) {
                    report(&format!("a check of {rows} rows"), ms);
                }
                runs += 1;
            }
            eprintln!("a check of {rows} rows: {:.1} ms", wall / 6.0);
        }
        assert_eq!(model.chain_runs(), runs, "every run chained");
    }

    /// How often a Qwen3.5 GGUF's multi-token-prediction layer (QWEN35_MODEL) drafts the token its trunk then picks:
    /// the trunk's greedy continuation of a prompt, then teacher-forced, the layer's argmax for each position against
    /// the token two on, for the ways its inputs could be wired (the trunk's hidden state before or after the output
    /// norm; the embedding joined first or second; the layer's rope position that of the hidden state or the next).
    #[test]
    #[ignore = "a measurement; needs a WebGPU adapter and a Qwen3.5 GGUF with an MTP layer; run with --nocapture"]
    fn measure_qwen35_mtp_acceptance() {
        use std::sync::Arc;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let mut model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        if let llama_rs::Model::Qwen35(m) = &mut model {
            m.mtp = m.load_mtp(&gguf).unwrap();
        }
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let Some(mtp) = m.mtp.as_ref() else { panic!("no MTP layer in {path}") };
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        let prompts = [
            "<|im_start|>user\nExplain how a refrigerator keeps food cold, in a few short paragraphs.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ];
        let gen = 96usize;
        let mut totals = std::collections::BTreeMap::<String, (usize, usize)>::new();
        for prompt in prompts {
            let p: Vec<u32> = m.tokenizer.encode(prompt, false).unwrap();
            // the trunk's greedy continuation (chained)
            let mut kv = model.new_kv_cache(p.len() + gen + 8);
            let mut seq = p.clone();
            let mut l = m.forward_embeds_positions(&m.embed_text(&p), p.len(), &mut kv, None).unwrap().to_host();
            for _ in 0..gen {
                let t = argmax(l.data());
                seq.push(t);
                l = m.forward_embeds_positions(&m.embed_text(&[t]), 1, &mut kv, None).unwrap().to_host();
            }
            // teacher-forced: every position's hidden state, before and after the output norm
            let n = seq.len();
            let mut kh = model.new_kv_cache(n + 8);
            let hidden = m.trunk_host(&m.embed_text(&seq), n, &mut kh, None).unwrap();
            let normed = ggml_rs::ops::rmsnorm(backend.as_ref(), &hidden, &m.output_norm, m.config.rms_eps);
            let rows = n - 1;
            let first = |t: &ggml_rs::Tensor| backend.slice_axis0_range(t, 0, rows);
            let next = m.embed_text(&seq[1..]);
            for (hname, h) in [("pre-norm", first(&hidden)), ("post-norm", first(&normed))] {
                for embed_first in [true, false] {
                    for offset in [0u32, 1] {
                        let mut mkv = llama_rs::KvCache::new(backend.as_ref(), 1, rows + 8, m.config.n_kv_heads, m.config.head_dim);
                        let positions: Vec<u32> = (0..rows as u32).map(|r| r + offset).collect();
                        let logits = m.mtp_logits_host(mtp, &h, &next, &mut mkv, &positions, embed_first).unwrap().to_host();
                        let v = logits.numel() / rows;
                        // row t drafts seq[t + 2]; count the continuation's
                        let (mut hit, mut all) = (0, 0);
                        for t in p.len() - 1..rows - 1 {
                            hit += (argmax(&logits.data()[t * v..(t + 1) * v]) == seq[t + 2]) as usize;
                            all += 1;
                        }
                        let key = format!("hidden {hname}, embedding {}, rope at {}", if embed_first { "first" } else { "second" }, if offset == 0 { "t" } else { "t + 1" });
                        let e = totals.entry(key).or_default();
                        e.0 += hit;
                        e.1 += all;
                    }
                }
            }
        }
        for (k, (hit, all)) in &totals {
            eprintln!("{k}: {hit} of {all} drafts the trunk's next token ({:.2})", *hit as f64 / *all as f64);
        }
    }

    /// A Qwen3.5 GGUF's multi-token-prediction layer drafting further on its own (QWEN35_MODEL): from the trunk's
    /// hidden state (after the output norm, the embedding joined first, as `measure_qwen35_mtp_acceptance` finds), a
    /// first draft, then each next from the layer's own output and the draft before; teacher-forced on the trunk's
    /// greedy continuation, how often draft `d` is the trunk's token where the drafts before it were (each depth a
    /// pass of its own over the sequence: an estimate, its cache that pass's).
    #[test]
    #[ignore = "a measurement; needs a WebGPU adapter and a Qwen3.5 GGUF with an MTP layer; run with --nocapture"]
    fn measure_qwen35_mtp_depth() {
        use std::sync::Arc;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let mut model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        if let llama_rs::Model::Qwen35(m) = &mut model {
            m.mtp = m.load_mtp(&gguf).unwrap();
        }
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let Some(mtp) = m.mtp.as_ref() else { panic!("no MTP layer in {path}") };
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        let prompts = [
            "<|im_start|>user\nExplain how a refrigerator keeps food cold, in a few short paragraphs.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>user\nList five tips for staying focused while studying, one line each.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ];
        let (gen, depth) = (96usize, 3usize);
        let mut hits = vec![(0usize, 0usize); depth];
        for prompt in prompts {
            let p: Vec<u32> = m.tokenizer.encode(prompt, false).unwrap();
            let mut kv = model.new_kv_cache(p.len() + gen + 8);
            let mut seq = p.clone();
            let mut l = m.forward_embeds_positions(&m.embed_text(&p), p.len(), &mut kv, None).unwrap().to_host();
            for _ in 0..gen {
                let t = argmax(l.data());
                seq.push(t);
                l = m.forward_embeds_positions(&m.embed_text(&[t]), 1, &mut kv, None).unwrap().to_host();
            }
            let n = seq.len();
            let mut kh = model.new_kv_cache(n + 8);
            let hidden = m.trunk_host(&m.embed_text(&seq), n, &mut kh, None).unwrap();
            let normed = ggml_rs::ops::rmsnorm(backend.as_ref(), &hidden, &m.output_norm, m.config.rms_eps);
            // depth d: rows t (positions t + d), the hidden state the depth before's (the trunk's at d = 0), the token
            // at t + d + 1; it drafts the token at t + d + 2
            let mut h = backend.slice_axis0_range(&normed, 0, n - 1);
            let mut drafts_ok: Vec<bool> = vec![true; n];
            for d in 0..depth {
                let rows = n - 1 - d;
                let h_rows = backend.slice_axis0_range(&h, 0, rows);
                let next = m.embed_text(&seq[d + 1..d + 1 + rows]);
                let mut mkv = llama_rs::KvCache::new(backend.as_ref(), 1, rows + 8, m.config.n_kv_heads, m.config.head_dim);
                let positions: Vec<u32> = (0..rows as u32).map(|r| r + d as u32).collect();
                let (out, logits) = m.mtp_host(mtp, &h_rows, &next, &mut mkv, &positions, true).unwrap();
                let logits = logits.to_host();
                let v = logits.numel() / rows;
                for t in p.len() - 1..rows.saturating_sub(1) {
                    if t + d + 2 >= n || !drafts_ok[t] {
                        continue;
                    }
                    let ok = argmax(&logits.data()[t * v..(t + 1) * v]) == seq[t + d + 2];
                    hits[d].0 += ok as usize;
                    hits[d].1 += 1;
                    drafts_ok[t] = ok;
                }
                h = out;
            }
        }
        for (d, (hit, all)) in hits.iter().enumerate() {
            eprintln!("draft {}: {hit} of {all} the trunk's token where the drafts before it were ({:.2})", d + 1, *hit as f64 / (*all).max(1) as f64);
        }
    }

    /// A Qwen3.5 GGUF with a multi-token-prediction layer (QWEN35_MODEL), greedy, drafting `k` tokens a step and
    /// checking them (QWEN35_DRAFTS, default 3): the tokens a step at a time give, drafted or not (bar near-ties: a
    /// check's attention sums as a prompt's does); how many drafts the checks take, and the time a token.
    #[test]
    #[ignore = "needs a WebGPU adapter and a Qwen3.5 GGUF with an MTP layer; run with --nocapture"]
    fn a_drafting_qwen35_answers_as_its_steps_do() {
        use std::sync::Arc;
        use std::time::Instant;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let k: usize = std::env::var("QWEN35_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        // the layer is loaded where asked for
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let mut model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        if let llama_rs::Model::Qwen35(m) = &mut model {
            m.mtp = m.load_mtp(&gguf).unwrap();
        }
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        // (QWEN35_PARAS: that many paragraphs of text before the question, some 60 tokens each: the steps and checks
        // deep in a context, their attention over the cache's f16 halves past 4,096 positions)
        let paras: usize = std::env::var("QWEN35_PARAS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let filler = "The river town kept its market on the north bank, where barges unloaded grain, wool and salt. Every spring the floods rose to the steps of the old customs house, and every summer the merchants rebuilt the stalls a little higher. The ferryman counted the seasons by the colour of the water. ".repeat(paras);
        let prompt: Vec<u32> = m.tokenizer.encode(&format!("<|im_start|>user\n{filler}Explain how a refrigerator keeps food cold, in a few short paragraphs.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), false).unwrap();
        eprintln!("a prompt of {} tokens", prompt.len());
        let gen = 160usize;
        // a step at a time
        let mut kv = model.new_kv_cache(prompt.len() + gen + 16);
        let mut l = m.forward_embeds_positions(&m.embed_text(&prompt), prompt.len(), &mut kv, None).unwrap().to_host();
        let (mut plain, clock) = (Vec::new(), Instant::now());
        for _ in 0..gen {
            let t = argmax(l.data());
            plain.push(t);
            l = m.forward_embeds_positions(&m.embed_text(&[t]), 1, &mut kv, None).unwrap().to_host();
        }
        let plain_s = clock.elapsed().as_secs_f64();
        assert!(m.drafts(), "the chain drafts with the model's MTP layer");
        // drafted and checked
        let mut kv = model.new_kv_cache(prompt.len() + gen + 16);
        let mut logits = m.forward_embeds_positions(&m.embed_text(&prompt), prompt.len(), &mut kv, None).unwrap().to_host();
        let mut history = prompt.clone();
        // the rows of logits the checks gave (the token they follow's place in `history`, the logits), to be held
        // against a step's at the same place afterwards (a check's rows run through the int8 kernels)
        let mut used: Vec<(usize, ggml_rs::Tensor)> = Vec::new();
        let (mut out, mut checks, mut accepted, clock) = (Vec::new(), 0usize, 0usize, Instant::now());
        let (mut t_draft, mut t_check, mut t_back) = (0f64, 0f64, 0f64);
        let mut check_kernels = std::collections::BTreeMap::<&'static str, (f64, u64)>::new();
        let _ = ggml_rs_wgpu::profile::take_kernels();
        while out.len() < gen {
            let next = argmax(logits.data());
            out.push(next);
            history.push(next);
            let c = Instant::now();
            let drafts = m.draft(&kv, &history, k).expect("drafts");
            t_draft += c.elapsed().as_secs_f64();
            if drafts.is_empty() {
                // none the layer is sure enough of: a step of the token alone
                logits = m.forward_embeds_positions(&m.embed_text(&[next]), 1, &mut kv, None).unwrap().to_host();
                continue;
            }
            let mut rows = vec![next];
            rows.extend(&drafts);
            let before = ggml_rs_wgpu::profile::take_kernels();
            let c = Instant::now();
            let checked = m.check(&rows, &mut kv).expect("a check");
            t_check += c.elapsed().as_secs_f64();
            for (name, ms, n) in ggml_rs_wgpu::profile::take_kernels() {
                let e = check_kernels.entry(name).or_insert((0.0, 0));
                e.0 += ms;
                e.1 += n;
            }
            drop(before);
            checks += 1;
            let mut j = 0;
            while j < drafts.len() && out.len() < gen && argmax(checked[j].data()) == drafts[j] {
                out.push(drafts[j]);
                history.push(drafts[j]);
                j += 1;
            }
            accepted += j;
            let c = Instant::now();
            m.rollback(&mut kv, rows.len(), j + 1);
            t_back += c.elapsed().as_secs_f64();
            logits = checked[j].clone();
            // row r follows `rows[r]`, at history's place len - 1 - (j - r)
            let at = history.len() - 1 - j;
            for (r, row) in checked.into_iter().take(j + 1).enumerate() {
                used.push((at + r, row));
            }
        }
        let spec_s = clock.elapsed().as_secs_f64();
        // a step at a time along the drafted tokens, each check's row against the step's logits there
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        let mut kv2 = model.new_kv_cache(prompt.len() + gen + 16);
        let _ = m.forward_embeds_positions(&m.embed_text(&prompt), prompt.len(), &mut kv2, None).unwrap();
        let mut worst = 1.0f64;
        let mut u = used.iter().peekable();
        for (place, &tok) in history.iter().enumerate().skip(prompt.len()) {
            let step = m.forward_embeds_positions(&m.embed_text(&[tok]), 1, &mut kv2, None).unwrap().to_host();
            while let Some((p, row)) = u.next_if(|(p, _)| *p == place) {
                let _ = p;
                worst = worst.min(cosine(step.data(), row.data()));
            }
        }
        eprintln!("a check's round: drafting {:.1} ms, the check {:.1} ms, rollbacks {:.1} ms", t_draft * 1e3 / checks as f64, t_check * 1e3 / checks as f64, t_back * 1e3 / checks as f64);
        let mut ck: Vec<_> = check_kernels.into_iter().collect();
        ck.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        for (name, (ms, c)) in ck.iter().take(12) {
            eprintln!("    check: {name:<28} {:>8.2} ms ({} dispatches)", ms / checks as f64, c / checks as u64);
        }
        out.truncate(gen);
        let same = plain.iter().zip(&out).take_while(|(a, b)| a == b).count();
        eprintln!("{gen} tokens: a step at a time {:.1} ms a token; drafting {k} {:.1} ms a token ({checks} checks, {accepted} drafts taken of {}, {:.2} a check); the same tokens for the first {same}; the checks' logits against the steps' at the same places: worst cosine {worst:.6}", plain_s * 1e3 / gen as f64, (spec_s - 0.0) * 1e3 / gen as f64, checks * k, accepted as f64 / checks as f64);
        eprintln!("plain:   {:?}", m.tokenizer.decode(&plain[..48.min(gen)]));
        eprintln!("drafted: {:?}", m.tokenizer.decode(&out[..48.min(gen)]));
        assert!(worst > 0.999, "a check's logits against a step's: cosine {worst}");
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
        // (a GGUF, or an EXL3 checkpoint's folder: OrcaSAQ's, its projections packed on the device)
        let model = if std::path::Path::new(&path).is_dir() {
            let gpu = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>().expect("the WebGPU backend");
            crate::orcasaq::load_portable(std::path::Path::new(&path), Arc::clone(&backend), &|d| gpu.exl3(d)).unwrap()
        } else {
            let gguf = gguf::GgufFile::open(&path).unwrap();
            llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap()
        };
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let mut kv = model.new_kv_cache(4096);
        let tokens: Vec<u32> = (0..2048u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let mut at = 0;
        let embed_ms = std::cell::Cell::new(0.0f64);
        let mut forward = |n: usize, kv: &mut llama_rs::KvCache| {
            let t = Instant::now();
            let e = m.embed_text(&tokens[at..at + n]);
            embed_ms.set(t.elapsed().as_secs_f64() * 1e3);
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
        let _ = ggml_rs_wgpu::profile::take_line();
        let t = Instant::now();
        for _ in 0..steps {
            forward(1, &mut kv);
        }
        eprintln!("{steps} decode steps: {:.2} ms a step; {}", t.elapsed().as_secs_f64() * 1e3 / steps as f64, ggml_rs_wgpu::profile::take_line());
        // what checking drafted tokens costs: a run of 2, 3, 4 and 5 rows against a step's one (each the best of 8)
        let _ = ggml_rs_wgpu::profile::take_kernels();
        for n in [1usize, 2, 3, 4, 5] {
            let best = (0..8).map(|_| forward(n, &mut kv)).fold(f64::MAX, f64::min);
            eprintln!("a run of {n} rows: {best:.2} ms");
            let k = ggml_rs_wgpu::profile::take_kernels();
            if !k.is_empty() {
                let all: f64 = k.iter().map(|e| e.1).sum();
                let count: u64 = k.iter().map(|e| e.2).sum();
                eprintln!("    every kernel: {:.2} ms a run ({} dispatches)", all / 8.0, count / 8);
                for (name, ms, c) in k.iter().take(if n == 1 { 24 } else { 6 }) {
                    eprintln!("    {name:<28} {:>8.2} ms a run ({} dispatches)", ms / 8.0, c / 8);
                }
            }
        }
        for n in [512usize, 512] {
            let past = kv.len;
            let _ = ggml_rs_wgpu::profile::take_kernels();
            let _ = ggml_rs_wgpu::profile::take_line();
            let _ = ggml_rs_wgpu::profile::take_pieces();
            eprintln!("a chunk of {n} at {past}: {:.1} ms", forward(n, &mut kv));
            eprintln!("    its embedding {:.1} ms; {}", embed_ms.get(), ggml_rs_wgpu::profile::take_line());
            if ggml_rs_wgpu::profile::pieces_on() {
                let (busy, span, pieces) = ggml_rs_wgpu::profile::take_pieces();
                eprintln!("    its {pieces} pieces on the GPU {busy:.1} ms over {span:.1} ms");
            }
            let k = ggml_rs_wgpu::profile::take_kernels();
            if !k.is_empty() {
                let all: f64 = k.iter().map(|e| e.1).sum();
                let count: u64 = k.iter().map(|e| e.2).sum();
                eprintln!("    every kernel: {all:.1} ms ({count} dispatches)");
                for (name, ms, c) in k.iter().take(16) {
                    eprintln!("    {name:<28} {ms:>8.2} ms ({c} dispatches)");
                }
            }
        }
    }

    /// A Qwen3.5 hybrid deep in a long context on one GPU (QWEN35_MODEL; QWEN35_PAST positions, 15,360 unless given):
    /// each chunk of 512's time on the way there, then a chunk and decode steps there with each kernel's GPU time
    /// (OAIY_CHAIN_PROFILE). QWEN35_FILLS: the way there that many times, each from an empty cache (a slow chunk the
    /// context's, or the card's after so many seconds of work?), QWEN35_PAUSE seconds idle before each but the first.
    #[test]
    #[ignore = "a timing; needs a WebGPU adapter and a Qwen3.5 GGUF; run with --nocapture"]
    fn measure_a_long_qwen35() {
        use std::sync::Arc;
        use std::time::Instant;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let past: usize = std::env::var("QWEN35_PAST").ok().and_then(|v| v.parse().ok()).unwrap_or(15_360);
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let fills: usize = std::env::var("QWEN35_FILLS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let pause: f64 = std::env::var("QWEN35_PAUSE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.);
        // (QWEN35_CALL: the tokens a call on the way there, 512 unless given; the server's are 4,096)
        let call: usize = std::env::var("QWEN35_CALL").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        // (QWEN35_CHUNK: the rows of the chunk measured there, 512 unless given; one card's prompt chunks are 1,024)
        let chunk: usize = std::env::var("QWEN35_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let mut kv = model.new_kv_cache(past + chunk + 512);
        let tokens: Vec<u32> = (0..(past + chunk + 512) as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let at = std::cell::Cell::new(0usize);
        let forward = |n: usize, kv: &mut llama_rs::KvCache| {
            let t = Instant::now();
            let e = m.embed_text(&tokens[at.get()..at.get() + n]);
            let l = m.forward_embeds_positions(&e, n, kv, None).unwrap();
            at.set(at.get() + n);
            let _ = l.to_host();
            t.elapsed().as_secs_f64() * 1e3
        };
        let kernels = |runs: f64| {
            let k = ggml_rs_wgpu::profile::take_kernels();
            if !k.is_empty() {
                let all: f64 = k.iter().map(|e| e.1).sum();
                let count: u64 = k.iter().map(|e| e.2).sum();
                eprintln!("    every kernel: {:.2} ms ({} dispatches)", all / runs, count as f64 / runs);
                for (name, ms, c) in k.iter().take(14) {
                    eprintln!("    {name:<32} {:>8.2} ms ({} dispatches)", ms / runs, *c as f64 / runs);
                }
            }
        };
        for fill in 0..fills {
            if fill > 0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(pause));
                kv = model.new_kv_cache(past + chunk + 512);
                at.set(0);
            }
            let started = Instant::now();
            let mut line = Vec::new();
            for i in 0..past / call {
                line.push(format!("{:.0}", forward(call, &mut kv)));
                if i % 6 == 5 || i + 1 == past / call {
                    eprintln!("chunks to {}: {} ms", (i + 1) * call, line.join(" "));
                    line.clear();
                }
            }
            // (the rows short of a whole call too: the fill is `past` tokens whatever a call's are; it printed `past`
            // for the whole calls' tokens, 12,288 of 15,360 with calls of 4,096)
            if past % call > 0 {
                eprintln!("the last {}: {:.0} ms", past % call, forward(past % call, &mut kv));
            }
            eprintln!("{} tokens in {:.2} s", kv.len, started.elapsed().as_secs_f64());
        }
        let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
        let (here, ms) = (kv.len, forward(chunk, &mut kv));
        eprintln!("a chunk of {chunk} at {here}: {ms:.1} ms; {}", ggml_rs_wgpu::profile::take_line());
        kernels(1.);
        let steps = 16;
        // (a first step apart: deep in a long cache it makes the cache's f16 halves, once)
        let first = forward(1, &mut kv);
        let _ = (ggml_rs_wgpu::profile::take_kernels(), ggml_rs_wgpu::profile::take_line());
        eprintln!("the first step at {}: {first:.1} ms", kv.len - 1);
        let t = Instant::now();
        for _ in 0..steps {
            forward(1, &mut kv);
        }
        eprintln!("{steps} decode steps at {}: {:.2} ms a step; {}", kv.len, t.elapsed().as_secs_f64() * 1e3 / steps as f64, ggml_rs_wgpu::profile::take_line());
        kernels(steps as f64);
        let best = (0..4).map(|_| forward(4, &mut kv)).fold(f64::MAX, f64::min);
        eprintln!("a run of 4 rows: {best:.2} ms");
        kernels(4.);
    }

    /// A Qwen3.5 hybrid's prompt over two GPUs (its later layers and head copied to the second, each chunk's first
    /// layers run as the second runs the chunk before's) answers as on the first alone: 2,148 tokens (QWEN35_MODEL),
    /// the last logits, 8 steps after and drafts with its MTP layer bit for bit, a second prompt after them too (the
    /// second's copy of the cache brought up to the steps the first ran), and how long each takes.
    #[test]
    #[ignore = "needs two WebGPU adapters and a Qwen3.5 GGUF (QWEN35_MODEL); run with --nocapture"]
    fn a_qwen35_prompt_over_two_gpus_answers_as_on_one() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let Ok(b0) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let Some(b1) = b0.others(None).into_iter().next() else { return };
        // (QWEN35_SWAP: the other GPU first)
        let (b0, b1) = if std::env::var_os("QWEN35_SWAP").is_some() { (b1, b0) } else { (b0, b1) };
        let (w0, w1) = (Arc::new(b0), Arc::new(b1));
        let backend: Arc<dyn ggml_rs::Backend> = w0.clone();
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let mut model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        if let llama_rs::Model::Qwen35(m) = &mut model {
            m.mtp = m.load_mtp(&gguf).unwrap();
        }
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        m.chain.split_onto(w1.clone());
        let gib = |b: &ggml_rs_wgpu::WgpuBackend| b.memory_budget().map_or("?".to_string(), |(budget, used)| format!("{:.1} of {:.1} GiB", used as f64 / (1u64 << 30) as f64, budget as f64 / (1u64 << 30) as f64));
        let warm = std::time::Instant::now();
        assert!(m.warm_up());
        eprintln!("warmed up (the second's layers copied) in {:.1} s", warm.elapsed().as_secs_f64());
        // (QWEN35_LEN: the first prompt's tokens, 2,148 unless asked)
        let len: u32 = std::env::var("QWEN35_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(2148);
        let tokens: Vec<u32> = (0..len).map(|i| 1000 + (i * 7919) % 20000).collect();
        let more: Vec<u32> = (0..1100u32).map(|i| 1500 + (i * 104729) % 30000).collect();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let mut results = Vec::new();
        // (QWEN35_ROUNDS, QWEN35_STEPS: how many rounds, and steps after each prompt)
        let rounds: usize = std::env::var("QWEN35_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let steps: usize = std::env::var("QWEN35_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        for round in 0..rounds {
            for split in [false, true] {
                m.chain.split_off.store(!split, Ordering::Relaxed);
                let mut kv = model.new_kv_cache(8192);
                let mut out = Vec::new();
                let mut history = Vec::new();
                for (turn, prompt) in [&tokens, &more].into_iter().enumerate() {
                    let t = std::time::Instant::now();
                    let e = m.embed_text(prompt).to_host();
                    let last = m.forward_embeds_positions(&e, prompt.len(), &mut kv, None).unwrap().to_host();
                    let secs = t.elapsed().as_secs_f64();
                    eprintln!("round {round}, {}, turn {turn}: {} tokens in {:.0} ms ({:.0} tokens a second); memory {} and {}", if split { "two GPUs" } else { "one" }, prompt.len(), secs * 1e3, prompt.len() as f64 / secs, gib(&w0), gib(&w1));
                    if ggml_rs_wgpu::profile::chain_on() {
                        for (name, ms, n) in ggml_rs_wgpu::profile::take_kernels().into_iter().take(10) {
                            eprintln!("    {name}: {ms:.1} ms ({n})");
                        }
                    }
                    history.extend_from_slice(prompt);
                    out.push(bits(last.data()));
                    let mut next = argmax(last.data());
                    for _ in 0..steps {
                        history.push(next);
                        let drafts = m.draft(&kv, &history[history.len() - 8..], 3).unwrap_or_default();
                        out.push(drafts.clone());
                        let e = m.embed_text(&[next]);
                        let l = m.forward_embeds_positions(&e, 1, &mut kv, None).unwrap().to_host();
                        next = argmax(l.data());
                        out.push(bits(l.data()));
                    }
                }
                results.push((out, kv.len));
            }
        }
        m.chain.split_off.store(false, Ordering::Relaxed);
        for pair in results.chunks(2) {
            assert_eq!(pair[0].1, pair[1].1, "the same rows in the caches");
            for (i, (a, b)) in pair[0].0.iter().zip(&pair[1].0).enumerate() {
                assert!(a == b, "result {i} (logits or drafts) bit for bit");
            }
        }
    }

    /// A Qwen3.5 hybrid's run that keeps its recurrent states from inside it (QWEN35_MODEL; QWEN35_LEN tokens, 1,300
    /// unless asked: a chunk of 1,024 and one that keeps both; QWEN35_TWO: its chunks over two devices, of 512, each
    /// device keeping its layers') against the runs that stop where it keeps them, before
    /// the prompt's last 7 tokens and before its last: the states before the last 7 are those runs' (the same rows
    /// through the same kernels where the stopped run's last chunk is 128 rows or more: to their rounding), those
    /// before the last and the logits as close as a run of few
    /// rows is to a chunk's (other kernels for its matmuls and attention); and a cache put back to either kept state
    /// and run on from there gives the run's own logits again, the same token.
    #[test]
    #[ignore = "needs a WebGPU adapter and a Qwen3.5 GGUF (QWEN35_MODEL); run with --nocapture"]
    fn a_qwen35_run_keeps_the_states_its_stops_would() {
        use std::sync::Arc;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        // (QWEN35_TWO: the prompts' chunks over a second device too, each keeping its own layers' states)
        let second = std::env::var_os("QWEN35_TWO").and_then(|_| b.others(None).into_iter().next());
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        let two = second.is_some();
        if let Some(b1) = second {
            m.chain.split_onto(Arc::new(b1));
            assert!(m.warm_up(), "the second device's layers");
            eprintln!("over two devices");
        }
        let n: usize = std::env::var("QWEN35_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1300);
        let tokens: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let taps = [n - 7, n - 1];
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (norm(a) * norm(b)).max(1e-30)
        };
        let host = |ts: &[Option<ggml_rs::Tensor>]| -> Vec<Option<Vec<f32>>> { ts.iter().map(|t| t.as_ref().map(|t| t.to_host().data().to_vec())).collect() };
        // (the same token, or one as good to a tenth of a logit: these prompts' tokens are no text, and two of their
        // next ones can be that close, a run of few rows' kernels apart)
        let agree = |a: &[f32], b: &[f32]| argmax(a) == argmax(b) || (b[argmax(b) as usize] - b[argmax(a) as usize]).abs() < 0.1;
        let run = |from: usize, to: usize, kv: &mut llama_rs::KvCache| {
            let e = m.embed_text(&tokens[from..to]).to_host();
            m.forward_embeds_positions(&e, to - from, kv, None).unwrap().to_host()
        };
        // (the kernels warmed: a prompt and a few rows)
        {
            let mut kv = model.new_kv_cache(4096);
            run(0, n - 7, &mut kv);
            run(n - 7, n, &mut kv);
        }
        // the runs that stop
        let mut kv = model.new_kv_cache(4096);
        let t = std::time::Instant::now();
        run(0, n - 7, &mut kv);
        let first = (host(&kv.ssm_state), host(&kv.ssm_conv));
        run(n - 7, n - 1, &mut kv);
        let second = (host(&kv.ssm_state), host(&kv.ssm_conv));
        let stopped = run(n - 1, n, &mut kv);
        let ms_stopped = t.elapsed().as_secs_f64() * 1e3;
        // the run that keeps them
        let mut kv = model.new_kv_cache(4096);
        let e = m.embed_text(&tokens).to_host();
        assert!(m.can_tap(n, &kv, &taps), "a run of {n} rows keeps its states after {taps:?}");
        let t = std::time::Instant::now();
        let (logits, kept) = m.forward_embeds_tapped(&e, n, &mut kv, &taps).expect("the run keeps its states");
        let logits = logits.to_host();
        let ms_kept = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(kept.iter().map(|k| k.at).collect::<Vec<_>>(), taps, "the states' places");
        assert_eq!(kv.len, n, "the cache's rows");
        let mut worst = Vec::new();
        for (tap, (states, convs)) in kept.iter().zip([&first, &second]) {
            let (mut cs, mut cc, mut layers) = (1f64, 1f64, 0);
            for (got, want) in host(&tap.states).iter().zip(states).chain(host(&tap.convs).iter().zip(convs)).map(|(g, w)| (g.as_ref(), w.as_ref())) {
                assert_eq!(got.is_some(), want.is_some(), "a layer's state where the cache has one");
                if let (Some(g), Some(w)) = (got, want) {
                    assert_eq!(g.len(), w.len(), "a state's size");
                    let c = cosine(g, w);
                    if g.len() == first.0.iter().flatten().next().map_or(0, |v| v.len()) { cs = cs.min(c) } else { cc = cc.min(c) }
                    layers += 1;
                }
            }
            eprintln!("the states kept after {} rows against the run's that stops there: the worst cosine {cs:.7} (the conv windows' {cc:.7}), {layers} of them", tap.at);
            worst.push(cs.min(cc));
        }
        let c = cosine(logits.data(), stopped.data());
        eprintln!("{n} tokens: the runs that stop {ms_stopped:.0} ms (this test reading their states to the host between them), the run that keeps its states {ms_kept:.0} ms; the logits' cosine {c:.6}, the same token {}", argmax(logits.data()) == argmax(stopped.data()));
        // (a chunk of few rows is other kernels than a longer one's: where the stopped run's last chunk is short, its
        // rows before the last 7 are not through the same ones either)
        let before = (n - 7) % if two { 512 } else { 1024 };
        assert!(worst[0] > if before == 0 || before >= 128 { 0.99999 } else { 0.995 }, "the states before the last 7 rows: {}", worst[0]);
        assert!(worst[1] > 0.995, "the states before the last row: {}", worst[1]);
        assert!(c > 0.995 && agree(logits.data(), stopped.data()), "the logits: {c}");
        // a cache put back to a kept state runs on to the same logits
        for tap in kept.iter().rev() {
            for l in 0..kv.ssm_state.len() {
                kv.ssm_state[l] = tap.states[l].as_ref().map(|t| backend.to_device(t.to_host()));
                kv.ssm_conv[l] = tap.convs[l].as_ref().map(|t| backend.to_device(t.to_host()));
            }
            kv.len = tap.at;
            let again = run(tap.at, n, &mut kv);
            let c = cosine(again.data(), logits.data());
            eprintln!("put back to the state after {} rows and run on: the logits' cosine {c:.6}, the same token {}", tap.at, argmax(again.data()) == argmax(logits.data()));
            assert!(c > 0.995 && agree(again.data(), logits.data()), "run on from the state after {} rows: {c}", tap.at);
        }
    }

    /// A Qwen3.5 hybrid's prompt run whole (its chain recording each chunk as the one before runs) answers as its
    /// chunks run one call at a time: 2,148 tokens (QWEN35_MODEL), the last logits and 8 steps after bit for bit, and
    /// how long each takes.
    #[test]
    #[ignore = "needs a WebGPU adapter and a Qwen3.5 GGUF (QWEN35_MODEL); run with --nocapture"]
    fn a_qwen35_prompt_run_whole_answers_as_its_chunks_one_at_a_time() {
        use std::sync::Arc;
        let path = std::env::var("QWEN35_MODEL").unwrap_or_else(|_| r"E:\models\Qwen3.8-27B-Q3_K_M.gguf".into());
        let Ok(b) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(b);
        let gguf = gguf::GgufFile::open(&path).unwrap();
        let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).unwrap();
        let llama_rs::Model::Qwen35(m) = &model else { panic!("a Qwen3.5 hybrid") };
        // (QWEN35_LEN: the prompt's tokens, 2,148 unless asked)
        let len: u32 = std::env::var("QWEN35_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(2148);
        let tokens: Vec<u32> = (0..len).map(|i| 1000 + (i * 7919) % 20000).collect();
        let argmax = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let mut results = Vec::new();
        let rounds: usize = std::env::var("QWEN35_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        for round in 0..rounds {
            for whole in [false, true] {
                let mut kv = model.new_kv_cache(4096);
                let t = std::time::Instant::now();
                let last = if whole {
                    let e = m.embed_text(&tokens).to_host();
                    m.forward_embeds_positions(&e, tokens.len(), &mut kv, None).unwrap().to_host()
                } else {
                    // (a chunk a call as the chain cuts a run on one card)
                    let mut l = None;
                    for chunk in tokens.chunks(1024) {
                        let e = m.embed_text(chunk).to_host();
                        l = Some(m.forward_embeds_positions(&e, chunk.len(), &mut kv, None).unwrap().to_host());
                    }
                    l.unwrap()
                };
                let secs = t.elapsed().as_secs_f64();
                let mut next = argmax(last.data());
                let mut steps = Vec::new();
                for _ in 0..8 {
                    let e = m.embed_text(&[next]);
                    let l = m.forward_embeds_positions(&e, 1, &mut kv, None).unwrap().to_host();
                    next = argmax(l.data());
                    steps.push(bits(l.data()));
                }
                eprintln!("round {round}, {}: {} tokens in {:.0} ms ({:.0} tokens a second)", if whole { "whole" } else { "a chunk a call" }, tokens.len(), secs * 1e3, tokens.len() as f64 / secs);
                results.push((bits(last.data()), steps, kv.len));
            }
        }
        for pair in results.chunks(2) {
            assert_eq!(pair[0].2, pair[1].2, "the same rows in the caches");
            assert!(pair[0].0 == pair[1].0, "the prompt's logits bit for bit");
            assert!(pair[0].1 == pair[1].1, "the steps' logits after it bit for bit");
        }
    }

    /// Qwen3.5's hybrid chained on the GPU (QWEN35_MODEL: the 9B, or Qwen3.8 27B) answers as its host path does: the
    /// same prompt (QWEN35_PROMPT tokens, in chunks of 512 as the server sends them) then 64 greedy steps each way
    /// give the same tokens, every step's logits close (up to where the tokens part, if they do: only at a near tie,
    /// the host's two within a hair, as the prompt's matmuls' f16 sums can tip); and the chain did run.
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
        let first_diff = host_tokens.iter().zip(&chain_tokens).position(|(a, b)| a != b);
        // (past where the tokens part the steps' inputs are not the same)
        let upto = first_diff.map_or(each.len(), |i| i + 1);
        let worst = each[..upto].iter().copied().fold(1.0f64, f64::min);
        eprintln!("prompt {n}: host {host_s:.2} s, chained {chain_s:.2} s ({runs} runs) for {steps} steps; prompt logits cosine {:.6}, worst {worst:.6}; first differing token {first_diff:?}", each[0]);
        assert!(runs > prompt.len().div_ceil(512), "the chain ran ({runs} runs)");
        assert!(worst >= 0.9999, "{worst}");
        if let Some(i) = first_diff {
            let h = &host_logits[i];
            let gap = h[host_tokens[i] as usize] - h[chain_tokens[i] as usize];
            eprintln!("  at step {i} the host's logit of its token {} is {gap:.4} above the chain's {}", host_tokens[i], chain_tokens[i]);
            assert!(gap < 0.05, "the tokens part where the host's two are {gap} apart");
        }
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
