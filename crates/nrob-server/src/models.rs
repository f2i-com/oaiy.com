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

use nrob::{Error, Result};

use crate::engine::Job;
use crate::{api, disk, engine, glm, Options, STATE_FORMAT};

pub(crate) fn needs_tool_precision(body: &nrob::json::Json) -> bool {
    use nrob::json::Json;
    let nonempty=|v: Option<&Json>|v.and_then(Json::as_array).is_some_and(|a|!a.is_empty());
    (body.get("tool_choice").and_then(Json::as_str)!=Some("none") && nonempty(body.get("tools"))) ||
        body.get("messages").and_then(Json::as_array).is_some_and(|messages|messages.iter().any(|m|
            matches!(m.get("role").and_then(Json::as_str),Some("tool"|"function")) ||
            nonempty(m.get("tool_calls")) || nonempty(m.get("tools"))))
}

/// Which runtime serves a model, decided by what is on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `dsv41-cuda`: a safetensors checkpoint directory with Engram tables.
    Deepseek,
    /// `llama-rs`: a GGUF file, or a directory holding one.
    Gguf,
}

impl Kind {
    /// A `.gguf` (or a directory containing one) is served through llama-rs;
    /// anything else is taken for a DeepSeek checkpoint directory, which is what
    /// this server has always assumed.
    pub fn detect(path: &Path) -> Kind {
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
}

impl Flavour {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Deepseek(t) => t.encode(text),
            // llama-rs applies BOS through the template, so not again here.
            Self::Gguf(t) | Self::Qwen(t) => t.encode(text, false).unwrap_or_default(),
        }
    }

    /// The prompt for a chat request, in this model's own template.
    ///
    /// Both templates put the reply inside a `<think>` block the prompt opens, so
    /// the reply parser downstream needs no variant.
    pub fn chat_prompt(
        &self,
        msgs: &[nrob::json::Json],
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
                let mut out = Vec::new();
                for m in msgs {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = m
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let role = match role {
                        "system" => llama_rs::Role::System,
                        "assistant" => llama_rs::Role::Assistant,
                        _ => llama_rs::Role::User,
                    };
                    out.push(llama_rs::ChatMessage { role, content });
                }
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
        }
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
    timings: Mutex<std::collections::BTreeMap<String, nrob::json::Json>>,
}

impl Models {
    pub fn new(opts: Options, loopback: bool) -> Result<Models> {
        let image_config = opts.image_config.as_deref().map(crate::images::Config::read).transpose().map_err(Error::Arg)?;
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
        for name in opts.vision_projectors.keys() {
            if !specs.iter().any(|s| &s.name==name && s.kind==Kind::Gguf) {
                return Err(Error::Arg(format!("vision projector {name} must name a configured GGUF model")));
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
    pub fn status(&self) -> nrob::json::Json {
        use nrob::json::Json;
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
        use nrob::json::Json;
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
        // During image work all chat turns stay on the controller. The route
        // persists after a batch so its next tool turn cannot reload DeepSeek.
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
            Kind::Deepseek => self.load_deepseek(&spec),
            Kind::Gguf => self.load_gguf(&spec),
        }
        .map_err(|e| format!("loading {}: {e}", spec.name))?;
        {
            use nrob::json::Json;
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
            vision: None,
            qwen_vision: None,
            image_token_id: 0,
            local_images: o.local_images.unwrap_or(self.loopback),
        }
    }

    // ---------------------------------------------------------------- DeepSeek
    fn load_deepseek(&self, spec: &Spec) -> Result<Live> {
        use dsv41_cuda::{GpuModel, GpuOptions};

        let o = &self.opts;
        let count = dsv41_cuda::gpu::device_count()?;
        let devices = available_devices(&o.devices, count)?;
        if !o.devices.is_empty() && devices != o.devices {
            self.say(format!("available GPU selection: {:?} (requested {:?})", devices, o.devices));
        }
        let observer_device = if o.observer_device < count { o.observer_device }
            else { *devices.last().ok_or_else(||nrob::Error::Arg("no CUDA device available".into()))? };
        // Load the reviewer FIRST, so DeepSeek sizes its caches from the VRAM
        // genuinely remaining. Both stay resident; model routing is unchanged.
        let observer = o.observer_model.as_deref().map(|path| {
            self.say(format!("loading resident observer {} on GPU {}", path.display(), observer_device));
            crate::observer::Observer::load_shared(path, observer_device, o.observer_vram_gb, devices.contains(&observer_device))
        }).transpose()?;
        let tok = Arc::new(dsv41::tokenizer::Tokenizer::load(&spec.path)?);
        let gopts = GpuOptions {
            devices,
            max_seq: o.ctx,
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

        let mut cfg = self.base_cfg(spec, o.ctx);
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
            self.say("Observer available: review off by default; opt in to a single approval gate with nrob_observer_review=blocking".into());
        } else if tool_experts {
            self.say("DSML boundary precision: ternary prompt/prose; MXFP4 tool payload; 80/20 expert cache budgets; trunk retained".into());
        }
        if let Some(dir) = &o.prompt_cache {
            // Different expert banks must never share prompt states, including
            // two variants built from the same retained tensor index.
            let identity = format!("{:?}|{:?}|{:?}|ctx={}|dsml-v1", spec.path, ternary_source, tool_source, o.ctx);
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

    // ---------------------------------------------------------------- GGUF
    fn load_gguf(&self, spec: &Spec) -> Result<Live> {
        let o = &self.opts;
        let path = Kind::gguf_path(&spec.path)?;
        // Every card `--devices` names, in that order: the first carries the trunk
        // and the rest take a share of the MoE layers. `open_cards` hands back the
        // same instances the expert tier will use, so the trunk and card 0's expert
        // shard share one CUDA context rather than opening a second on that device.
        let devices = self.image_config.as_ref().filter(|c|c.controller_name==spec.name).map(|c|vec![c.controller_device]).unwrap_or_else(||o.devices.clone());
        let cards = llama_rs::glm5next::device::open_cards(&devices)
            .map_err(|e| Error::Arg(e.to_string()))?;
        let backend: Arc<dyn ggml_rs::Backend> = Arc::clone(&cards[0]) as Arc<dyn ggml_rs::Backend>;
        let gguf = gguf::GgufFile::open(&path).map_err(|e| Error::Arg(e.to_string()))?;
        if gguf.get_str("general.architecture").ok() == Some("qwen35") {
            let model = llama_rs::Model::load(&gguf, Arc::clone(&backend)).map_err(|e| Error::Arg(e.to_string()))?;
            let tok = Arc::new(model.tokenizer().clone());
            // Dense Qwen fits one card; bound the initial KV allocation.
            let max_seq = o.ctx.min(model.config().context_length).min(16384);
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
            let e = crate::qwen::QwenEngine::new(model,projector,max_seq,!o.quiet && !o.silent);
            let thread = std::thread::Builder::new().name("qwen-model".into()).spawn(move || e.run(rx)).map_err(Error::Io)?;
            return Ok(Live {name:spec.name.clone(),jobs,thread,cfg:Arc::new(cfg),flavour:Arc::new(Flavour::Qwen(tok))});
        }
        drop(gguf);
        let budget = o.expert_cache_bytes();
        if o.ram_gb == 0 {
            self.say(format!(
                "expert cache: {:.0} GB of host RAM (80% of what is free; --ram-gb pins it)",
                budget as f64 / 1e9
            ));
        }
        let mut model = llama_rs::Model::open_streaming(&path, backend, budget)
            .map_err(|e| Error::Arg(e.to_string()))?;

        // The expert hierarchy. `open_streaming` leaves a streamed model with its
        // RAM cache only -- no VRAM expert cache, one card, no CPU tier -- which for
        // GLM-5.3-Flash is 0.686 s a token against 0.139 tiered, because then every
        // routed expert of every token crosses PCIe. Nothing else served here needs
        // asking, so the arch decides.
        if let llama_rs::Model::Glm5Next(g) = &mut model {
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
            llama_rs::Model::Glm5Next(g) => g.max_len().min(o.ctx),
            _ => o.ctx,
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
fn startup_warm_profile(usage: Option<&Path>, tool_experts: bool) -> Option<(&Path, &'static str)> {
    usage.filter(|p| p.is_file())
        .map(|p| (p, if tool_experts { "ternary" } else { "model" }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_startup_warms_its_ternary_bank_from_an_existing_profile() {
        let path = std::env::temp_dir().join(format!("nrob-warm-{}.txt", std::process::id()));
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
        use nrob::json::Json;
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
        let d = std::env::temp_dir().join("nrob-kind-test");
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
    if count == 0 { return Err(nrob::Error::Arg("DeepSeek currently requires a CUDA GPU for its trunk and state".into())); }
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
