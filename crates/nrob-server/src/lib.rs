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
mod http;

use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use dsv41::tokenizer::Tokenizer;
use dsv41_cuda::{GpuModel, GpuOptions};

/// Bump when what a prompt state holds changes: older files are then left
/// alone.
const STATE_FORMAT: u8 = 1;

/// How to run the server; [`Options::default`] gives the binary's defaults,
/// apart from the checkpoint directory, which the caller always names.
#[derive(Clone, Debug)]
pub struct Options {
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

impl Default for Options {
    fn default() -> Options {
        Options {
            host: "127.0.0.1".into(),
            port: 8000,
            devices: Vec::new(),
            ctx: 65536,
            ram_gb: 140,
            model: PathBuf::new(),
            engram_meta: None,
            usage: None,
            name: "deepseek-v4.1-flash".into(),
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
        if a.active.load(Ordering::Relaxed) > 0 {
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
    let log = |m: String| {
        if !o.silent {
            eprintln!("nrob-server: {m}");
        }
    };
    let listener = TcpListener::bind((o.host.as_str(), o.port))?;
    let addr = listener.local_addr()?;
    listening(addr);

    let devices = if o.devices.is_empty() { (0..dsv41_cuda::gpu::device_count()?.min(2)).collect() } else { o.devices.clone() };
    let t = Instant::now();
    let tok = Arc::new(Tokenizer::load(&o.model)?);
    let opts = GpuOptions {
        devices: devices.clone(),
        max_seq: o.ctx,
        expert_cache_bytes: o.ram_gb << 30,
        direct_io: true,
        vram_expert_bytes: None,
        vram_headroom_bytes: (o.headroom_gb * (1u64 << 30) as f64) as usize,
        cpu_expert_threads: o.cpu_threads,
        vision: o.vision,
        // the free VRAM decides, pass by pass
        residual_on_device: None,
    };
    let engram_meta = o.engram_meta.clone().unwrap_or_else(|| o.model.join("engram_meta.safetensors"));
    let mut model = GpuModel::load(&o.model, &engram_meta, &opts)?;
    log(format!("model loaded on cuda:{devices:?} in {:.1}s ({} token context)", t.elapsed().as_secs_f64(), o.ctx));
    if let Some(path) = o.usage.as_ref().filter(|p| p.exists()) {
        let t = Instant::now();
        let (vram, queued) = model.warm(path, 4)?;
        log(format!("warmed {vram} experts into VRAM in {:.1}s; {queued} more loading into RAM in the background", t.elapsed().as_secs_f64()));
    }

    let vision = if model.has_vision() { model.cfg.vision.clone() } else { None };
    let image_token_id = model.cfg.image_token_id;
    if vision.is_some() {
        log("vision tower loaded; chat requests may carry images".into());
    }
    let loopback = o.host == "localhost" || o.host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    let (tx, rx) = mpsc::channel();
    let request_log = !o.quiet && !o.silent;
    let mut engine = engine::Engine::new(model, Arc::clone(&tok), o.chunk, o.step_below, o.layered_max, o.checkpoints, o.usage.clone(), request_log);
    engine.warn = !o.silent;
    if let Some(dir) = &o.prompt_cache {
        // states belong to this model (its config and weight map) and to
        // this state format
        let mut fingerprint = disk::fnv(&[STATE_FORMAT], 0);
        for name in ["config.json", "model.safetensors.index.json"] {
            fingerprint = disk::fnv(&std::fs::read(o.model.join(name)).unwrap_or_default(), fingerprint);
        }
        match disk::DiskCache::open(dir, fingerprint, (o.prompt_cache_gb * 1e9) as u64) {
            Ok(cache) => {
                log(format!("{} prompt states on disk in {}", cache.len(), dir.display()));
                engine.disk = Some(cache);
            }
            Err(e) => log(format!("prompt states are not kept ({}: {e})", dir.display())),
        }
    }
    std::thread::Builder::new().name("model".into()).spawn(move || engine.run(rx)).map_err(nrob::Error::Io)?;

    let server = Arc::new(api::Server {
        cfg: api::Config {
            model_name: o.name.clone(),
            api_key: o.api_key.clone(),
            max_seq: o.ctx,
            thinking: o.thinking,
            effort: o.effort,
            max_tokens: o.max_tokens,
            temperature: o.temperature,
            top_p: o.top_p,
            vision,
            image_token_id,
            local_images: o.local_images.unwrap_or(loopback),
        },
        tok,
        jobs: Mutex::new(tx),
    });
    log(format!("serving {} at http://{addr}/v1", o.name));
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
    Ok(Running { addr, accept, activity })
}
