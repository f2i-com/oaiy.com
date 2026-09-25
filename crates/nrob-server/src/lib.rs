//! nrob-server: DeepSeek-V4.1-Flash behind an OpenAI-compatible HTTP API,
//! for chat clients and coding harnesses.
//!
//! The `nrob-server` binary is a thin wrapper around [`start`]. A program
//! that wants the model inside its own process (a coding CLI, say) calls
//! [`start`] itself: with `port: 0` and a random `api_key` the API is
//! reachable only by whoever holds the key, on a loopback port picked by the
//! OS, and `silent` keeps the server's log off the host's terminal.

#![forbid(unsafe_code)]

mod api;
mod disk;
mod engine;
pub mod observer;
mod tool_phase;
mod repetition;
// VENDORED-LOCAL: GLM-5.3-Flash served through the same job/event contract.
pub mod glm;
// VENDORED-LOCAL: the configured models, and swapping between them.
pub mod models;
mod http;
mod qwen;
mod orcasaq;
mod qwen_cache;
mod qwen_vision;
pub mod images;
mod media_catalog;

use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};


/// Bump when what a prompt state holds changes: older files are then left
/// alone.
///
/// 2: the glm5next KDA decay was on the wrong axis until 2026-09-21, so every
/// state written before then holds a recurrent state built with a transposed
/// decay. Reusing one puts the model straight back into the degeneration the fix
/// removed -- and it would look like the fix not working, on exactly the prompts a
/// harness repeats. See `llama_rs::glm5next::kda`.
const STATE_FORMAT: u8 = 2;

/// How to run the server; [`Options::default`] gives the binary's defaults,
/// apart from the checkpoint directory, which the caller always names.
#[derive(Clone, Debug)]
pub struct Options {
    /// Native image worker configuration; disabled unless explicitly configured.
    pub image_config: Option<PathBuf>,
    /// Trusted host override for project-owned generated media and caches.
    pub media_output_root: Option<PathBuf>,
    /// Listen address (loopback by default).
    pub host: String,
    /// Port; 0 lets the OS pick one ([`Running::addr`] tells which).
    pub port: u16,
    /// CUDA devices; the layers are split across them. Empty: the visible
    /// ones in order, at most two (the layers split only at layer 20).
    pub devices: Vec<usize>,
    /// Context length in tokens, prompt and reply together.
    pub ctx: usize,
    /// Host RAM for the expert cache, in GB.
    /// Host RAM for the expert cache, in GB. **0 means decide from what is free**:
    /// see [`Options::expert_cache_bytes`].
    pub ram_gb: usize,
    /// Checkpoint directory (required: no default).
    pub model: PathBuf,
    /// The Engram precompute, `engram_meta.safetensors` (default: in the
    /// checkpoint directory).
    pub engram_meta: Option<PathBuf>,
    /// Expert usage profile: warms the caches at start, updated after every
    /// request.
    pub usage: Option<PathBuf>,
    /// The model name clients use.
    pub name: String,
    // VENDORED-LOCAL: more than one model, switched on demand.
    /// Further models a client may ask for by name, as `(name, path)`. The
    /// `model`/`name` pair above is the default and is always available.
    ///
    /// A `.gguf` (or a directory holding one) is served through llama-rs; anything
    /// else is taken for a DeepSeek checkpoint directory. Only one is ever resident:
    /// asking for another unloads the current one first.
    pub extra_models: Vec<(String, std::path::PathBuf)>,
    /// Vision projectors keyed by model name: GGUF files, or original Qwen
    /// safetensors directories for OrcaSAQ (tools/orcasaq/download.py --vision).
    pub vision_projectors: std::collections::BTreeMap<String, PathBuf>,
    /// Which configured model to load at start, by name. `None` takes `name`.
    ///
    /// Only one model is resident, and loading one takes 10 s for GLM-5.3-Flash and
    /// 65 s for DeepSeek-V4.1-Flash, so starting with the one that will be asked for
    /// saves loading a model only to unload it.
    pub start_model: Option<String>,
    /// Original MXFP4 expert source for tool turns; trunk stays resident.
    pub tools_experts: Option<std::path::PathBuf>,
    /// Explicit routed expert sources, keyed by configured model name.
    pub ternary_experts: std::collections::BTreeMap<String, PathBuf>,
    /// Original experts used inside tool payloads for a named ternary model.
    pub tool_expert_sources: std::collections::BTreeMap<String, PathBuf>,
    /// Optional resident GGUF reviewer. No observer is loaded unless configured.
    pub observer_model: Option<PathBuf>,
    pub observer_device: usize,
    pub observer_vram_gb: usize,
    pub expert_trace: Option<std::path::PathBuf>,
    /// Stop long repeated prose/reasoning blocks; does not apply inside tool payloads.
    pub repetition_guard: bool,
    /// Require `Authorization: Bearer KEY`.
    pub api_key: Option<String>,
    /// Reason before answering unless a request says otherwise.
    pub thinking: bool,
    /// Default reasoning effort, 1-100.
    pub effort: u32,
    /// Reply length when a request gives none.
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Attention sub-chunk of a layered pass.
    pub chunk: usize,
    /// Prompt stretches shorter than this run one token at a time.
    pub step_below: usize,
    /// Longer stretches run layer by layer, up to this many tokens a pass
    /// (each pass reads the experts once; its activations take VRAM, and a
    /// pass that runs out halves this for the rest of the run). 20,480 fits
    /// two 32 GB cards with the default headroom; 32,768 does not.
    pub layered_max: usize,
    /// VRAM kept free for activations, in GB; the rest caches experts.
    pub headroom_gb: f64,
    /// Prefix-cache checkpoints kept (~10 MB each).
    pub checkpoints: usize,
    /// Where prompt states are kept between runs, so that a new process
    /// does not read a system prompt (or a conversation it resumes) again.
    /// `None`: not kept.
    pub prompt_cache: Option<PathBuf>,
    /// Disk the prompt states may take, in GB.
    pub prompt_cache_gb: f64,
    /// CPU threads for experts that miss VRAM (`None`: upload every miss).
    pub cpu_threads: Option<usize>,
    /// Load the vision tower.
    pub vision: bool,
    /// Let requests name image files on this machine (default: when
    /// listening on loopback only).
    pub local_images: Option<bool>,
    /// No per-request log.
    pub quiet: bool,
    /// No log at all, not even at start (a host with its own UI).
    pub silent: bool,
}

impl Options {
    /// The expert cache's RAM, in bytes.
    ///
    /// `ram_gb` when it is set, and otherwise 80% of what is free. A fixed default
    /// is right only on the machine it was picked on: 140 GB left this one (190 GB)
    /// with the OS and the caller's own tools competing for the rest, and would
    /// refuse to start on a smaller one. 80% of free leaves a fifth for everything
    /// else, which is what "most of it" ought to mean.
    ///
    /// 32 GB when the machine will not say -- small enough to start anywhere, and
    /// `--ram-gb` is there for a caller who knows better.
    pub fn expert_cache_bytes(&self) -> u64 {
        host_cache_budget(self.ram_gb, ggml_rs_cuda::host_memory().map(|(free, _)| free as u64))
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            image_config: None,
            media_output_root: None,
            host: "127.0.0.1".into(),
            port: 8000,
            devices: Vec::new(),
            ctx: 65536,
            // Decided at load from what the machine actually has free, rather than
            // a number that is right for one machine: see `expert_cache_bytes`.
            ram_gb: 0,
            model: PathBuf::new(),
            engram_meta: None,
            usage: None,
            name: "deepseek-v4.1-flash".into(),
            extra_models: Vec::new(),
            vision_projectors: std::collections::BTreeMap::new(),
            start_model: None,
            tools_experts: None,
            ternary_experts: std::collections::BTreeMap::new(),
            tool_expert_sources: std::collections::BTreeMap::new(),
            observer_model: None,
            observer_device: usize::MAX,
            observer_vram_gb: 16,
            expert_trace: None,
            repetition_guard: true,
            api_key: None,
            thinking: false,
            effort: 75,
            max_tokens: 8192,
            temperature: 0.6,
            top_p: 0.95,
            chunk: 1024,
            step_below: 512,
            layered_max: 20_480,
            headroom_gb: 2.0,
            checkpoints: 256,
            prompt_cache: None,
            prompt_cache_gb: 4.0,
            cpu_threads: Some(24),
            vision: true,
            local_images: None,
            quiet: false,
            silent: false,
        }
    }
}

/// A server whose model thread and accept loop run in the background.
pub struct Running {
    addr: SocketAddr,
    accept: JoinHandle<()>,
    activity: Arc<Activity>,
    images: Arc<images::Images>,
}

/// Requests under way, and when the last one ended.
struct Activity {
    active: AtomicUsize,
    /// Milliseconds after `epoch`.
    last: AtomicU64,
    epoch: Instant,
}

impl Activity {
    fn now(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }
}

impl Running {
    /// Where it listens (the port the OS picked, for `port: 0`).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// How long since a request last ended (zero while one runs), for a
    /// host that stops an idle server.
    pub fn idle_for(&self) -> Duration {
        let a = &self.activity;
        if a.active.load(Ordering::Relaxed) > 0 || self.images.busy() {
            return Duration::ZERO;
        }
        Duration::from_millis(a.now().saturating_sub(a.last.load(Ordering::Relaxed)))
    }

    /// Serve until the process ends.
    pub fn wait(self) {
        let _ = self.accept.join();
    }
}

/// Bind the listener, load the model (warming its caches from the usage
/// profile) and start serving. Returns once requests can be served; a busy
/// port fails before the model loads.
pub fn start(o: Options) -> nrob::Result<Running> {
    start_listening(o, |_| {})
}

/// [`start`], telling `listening` where the server listens as soon as it
/// does (before the model loads: a client may connect at once, and its
/// requests wait until the model is there).
pub fn start_listening(o: Options, listening: impl FnOnce(SocketAddr)) -> nrob::Result<Running> {
    if o.model.as_os_str().is_empty() {
        return Err(nrob::Error::Arg("no checkpoint directory given".into()));
    }
    // Copied out of `o` because it moves into `Models` below, which owns the
    // settings a later load needs.
    let silent = o.silent;
    let log = move |m: String| {
        if !silent {
            eprintln!("nrob-server: {m}");
        }
    };
    let listener = TcpListener::bind((o.host.as_str(), o.port))?;
    let addr = listener.local_addr()?;
    listening(addr);

    // VENDORED-LOCAL: the configured models, loaded one at a time.
    //
    // Everything a load needs moved into `models.rs`, because a switch has to be
    // able to do it again later: the DeepSeek path (GpuModel, the usage warm-up, the
    // prompt-state cache) and the GGUF path (llama-rs) are both there, chosen by what
    // is on disk. Only one model is ever resident.
    let loopback = o.host == "localhost"
        || o.host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    let request_log = !o.quiet && !o.silent;
    let ctx = o.ctx;
    let api_key = o.api_key.clone();
    let local_images = o.local_images.unwrap_or(loopback);
    // The model actually being served, which is the one `--start` names when it
    // names one -- reporting `--model`'s name while another is loaded is how the
    // wrong model gets blamed for the wait.
    let default_name = o.start_model.clone().unwrap_or_else(|| o.name.clone());
    let configured = {
        let mut names = vec![o.name.clone()];
        names.extend(o.extra_models.iter().map(|(n, _)| n.clone()));
        names
    };

    let t = Instant::now();
    // Taken before `o` moves: which model to start with, so the one a caller is
    // going to ask for is the one that loads, rather than loading another and
    // unloading it.
    let start_model = o.start_model.clone();
    let models = Arc::new(models::Models::new(o, loopback)?);
    // Load it now rather than on the first request, so a start-up failure is a
    // start-up failure and not a 400 an hour later.
    models
        .activate(start_model.as_deref())
        .map_err(|e| nrob::Error::Arg(e))?;
    log(format!(
        "{default_name} loaded in {:.1}s ({ctx} token context)",
        t.elapsed().as_secs_f64()
    ));
    if configured.len() > 1 {
        log(format!(
            "switchable models: {} — naming another in a request unloads the current one",
            configured.join(", ")
        ));
    }

    let images = Arc::new(images::Images::new(Arc::clone(&models)));
    let server = Arc::new(api::Server {
        images: Arc::clone(&images),
        models: Arc::clone(&models),
        api_key,
        local_images,
        ctx,
    });
    log(format!("serving {default_name} at http://{addr}/v1"));
    let activity = Arc::new(Activity { active: AtomicUsize::new(0), last: AtomicU64::new(0), epoch: Instant::now() });
    let requests = Arc::clone(&activity);
    let accept = std::thread::Builder::new()
        .name("accept".into())
        .spawn(move || {
            for conn in listener.incoming() {
                let Ok(stream) = conn else { continue };
                let server = Arc::clone(&server);
                let requests = Arc::clone(&requests);
                let peer = stream.peer_addr().map(|p| p.to_string()).unwrap_or_default();
                std::thread::spawn(move || {
                    http::serve(stream, |req, w| {
                        let t = Instant::now();
                        requests.active.fetch_add(1, Ordering::Relaxed);
                        let keep = server.handle(req, w);
                        requests.last.store(requests.now(), Ordering::Relaxed);
                        requests.active.fetch_sub(1, Ordering::Relaxed);
                        if request_log && req.path != "/health" {
                            eprintln!("{peer} {} {} ({:.1}s)", req.method, req.path, t.elapsed().as_secs_f64());
                        }
                        keep
                    });
                });
            }
        })
        .map_err(nrob::Error::Io)?;
    Ok(Running { addr, accept, activity, images })
}

fn host_cache_budget(requested_gib: usize, available: Option<u64>) -> u64 {
    let safe = available.map(|n|n / 5 * 4).unwrap_or(32u64 << 30);
    if requested_gib == 0 { safe } else { (requested_gib as u64).saturating_mul(1u64 << 30).min(safe) }
}
#[test]
fn ram_cache_clamps_to_actual_available_memory() {
    let gib = 1u64 << 30;
    assert_eq!(host_cache_budget(140,Some(10*gib)),8*gib);
    assert_eq!(host_cache_budget(4,Some(10*gib)),4*gib);
    assert_eq!(host_cache_budget(0,Some(10*gib)),8*gib);
}
