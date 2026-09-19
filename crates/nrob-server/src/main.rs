//! nrob-server: DeepSeek-V4.1-Flash behind an OpenAI-compatible HTTP API,
//! for chat clients and coding harnesses.
//!
//!   nrob-server [--port 8000] [--host 127.0.0.1] [--devices 1,0] [--ctx 65536] ...
//!
//! Point a harness at `http://127.0.0.1:8000/v1` with any API key (or the
//! one given with `--api-key`) and the model name `deepseek-v4.1-flash`.
//! Run `nrob-server --help` for every option.

#![forbid(unsafe_code)]

mod api;
mod engine;
mod http;

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use dsv41::tokenizer::Tokenizer;
use dsv41_cuda::{GpuModel, GpuOptions};

struct Args {
    host: String,
    port: u16,
    devices: Vec<usize>,
    ctx: usize,
    ram_gb: usize,
    model: PathBuf,
    golden: PathBuf,
    usage: Option<PathBuf>,
    name: String,
    api_key: Option<String>,
    thinking: bool,
    effort: u32,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    chunk: usize,
    step_below: usize,
    layered_max: usize,
    headroom_gb: f64,
    checkpoints: usize,
    cpu_threads: Option<usize>,
    vision: bool,
    local_images: Option<bool>,
    quiet: bool,
}

const HELP: &str = "nrob-server: DeepSeek-V4.1-Flash behind an OpenAI-compatible API

  --host ADDR          listen address (default 127.0.0.1; 0.0.0.0 for the network)
  --port N             port (default 8000)
  --devices 1,0        CUDA devices; layers split across them (default 1,0)
  --ctx N              context length in tokens, prompt + reply (default 65536)
  --ram-gb N           host RAM for the expert cache (default 140)
  --model DIR          checkpoint directory (default D:\\deepseek\\model when that
                       drive is attached, else E:\\deepseek\\model)
  --golden DIR         directory with engram_meta.safetensors (default E:\\deepseek\\golden)
  --usage FILE|off     expert usage profile: warms the caches at start, updated
                       after every request (default E:\\deepseek\\expert_usage.txt)
  --name NAME          model name clients use (default deepseek-v4.1-flash)
  --api-key KEY        require `Authorization: Bearer KEY`
  --thinking           reason before answering unless a request says otherwise
                       (default: answer directly; requests turn reasoning on with
                       reasoning_effort, or thinking: {type: enabled})
  --effort N           default reasoning effort, 1-100 (default 75)
  --max-tokens N       reply length when a request gives none (default 8192)
  --temperature F      default temperature (default 0.6)
  --top-p F            default top-p (default 0.95)
  --step-below N       prompt stretches shorter than this run one token at a
                       time through the decode path (default 2048)
  --layered-max N      longer stretches run layer by layer, every expert read
                       once per pass, up to N tokens a pass (default 8192)
  --chunk N            attention sub-chunk of a layered pass (default 1024)
  --vram-headroom-gb F VRAM kept free for activations; the rest caches experts
                       (default 2)
  --checkpoints N      prefix-cache checkpoints kept (~10 MB each, default 256)
  --cpu-threads N      CPU threads for experts that miss VRAM (default 24; 0 = off)
  --no-vision          skip the vision tower (saves ~1 GB of VRAM; images are refused)
  --local-images on|off  let requests name image files on this machine (paths,
                       file:// URLs); default on when listening on loopback only
  --quiet              no per-request log
";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        host: "127.0.0.1".into(),
        port: 8000,
        devices: vec![1, 0],
        ctx: 65536,
        ram_gb: 140,
        // the external T9 copy reads 4x faster than the internal drive once
        // that one throttles; fall back when it is not plugged in
        model: [r"D:\deepseek\model", r"E:\deepseek\model"]
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.join("config.json").exists())
            .unwrap_or_else(|| r"E:\deepseek\model".into()),
        golden: r"E:\deepseek\golden".into(),
        usage: Some(r"E:\deepseek\expert_usage.txt".into()),
        name: "deepseek-v4.1-flash".into(),
        api_key: None,
        thinking: false,
        effort: 75,
        max_tokens: 8192,
        temperature: 0.6,
        top_p: 0.95,
        chunk: 1024,
        step_below: 2048,
        layered_max: 8192,
        headroom_gb: 2.0,
        checkpoints: 256,
        cpu_threads: Some(24),
        vision: true,
        local_images: None,
        quiet: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let num = |v: String| v.trim().parse::<usize>().map_err(|_| format!("{flag}: {v:?} is not a number"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--host" => a.host = val()?,
            "--port" => a.port = val()?.parse().map_err(|_| "--port: not a port number".to_string())?,
            "--devices" => {
                a.devices = val()?.split(',').map(|d| d.trim().parse().map_err(|_| format!("--devices: bad ordinal {d:?}"))).collect::<Result<_, _>>()?
            }
            "--ctx" => a.ctx = num(val()?)?,
            "--ram-gb" => a.ram_gb = num(val()?)?,
            "--model" => a.model = val()?.into(),
            "--golden" => a.golden = val()?.into(),
            "--usage" => {
                let v = val()?;
                a.usage = if v == "off" { None } else { Some(v.into()) };
            }
            "--name" => a.name = val()?,
            "--api-key" => a.api_key = Some(val()?),
            "--thinking" => a.thinking = true,
            "--effort" => a.effort = num(val()?)?.clamp(1, 100) as u32,
            "--max-tokens" => a.max_tokens = num(val()?)?.max(1),
            "--temperature" => a.temperature = val()?.parse().map_err(|_| "--temperature: not a number".to_string())?,
            "--top-p" => a.top_p = val()?.parse().map_err(|_| "--top-p: not a number".to_string())?,
            "--chunk" => a.chunk = num(val()?)?.max(1),
            "--step-below" => a.step_below = num(val()?)?,
            "--layered-max" => a.layered_max = num(val()?)?,
            "--vram-headroom-gb" => a.headroom_gb = val()?.parse().map_err(|_| "--vram-headroom-gb: not a number".to_string())?,
            "--checkpoints" => a.checkpoints = num(val()?)?,
            "--cpu-threads" => {
                let n = num(val()?)?;
                a.cpu_threads = if n == 0 { None } else { Some(n) };
            }
            "--no-vision" => a.vision = false,
            "--local-images" => {
                a.local_images = match val()?.as_str() {
                    "on" => Some(true),
                    "off" => Some(false),
                    v => return Err(format!("--local-images: {v:?} is not on or off")),
                }
            }
            "--quiet" => a.quiet = true,
            other => return Err(format!("unknown option {other} (try --help)")),
        }
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(m) => {
            eprintln!("{m}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args) {
        eprintln!("nrob-server: {e}");
        std::process::exit(1);
    }
}

fn run(a: Args) -> nrob::Result<()> {
    let t = Instant::now();
    let tok = Arc::new(Tokenizer::load(&a.model)?);
    let opts = GpuOptions {
        devices: a.devices.clone(),
        max_seq: a.ctx,
        expert_cache_bytes: a.ram_gb << 30,
        direct_io: true,
        vram_expert_bytes: None,
        vram_headroom_bytes: (a.headroom_gb * (1u64 << 30) as f64) as usize,
        cpu_expert_threads: a.cpu_threads,
        vision: a.vision,
    };
    let mut model = GpuModel::load(&a.model, &a.golden.join("engram_meta.safetensors"), &opts)?;
    eprintln!("nrob-server: model loaded on cuda:{:?} in {:.1}s ({} token context)", a.devices, t.elapsed().as_secs_f64(), a.ctx);
    if let Some(path) = a.usage.as_ref().filter(|p| p.exists()) {
        let t = Instant::now();
        let (vram, queued) = model.warm(path, 4)?;
        eprintln!("nrob-server: warmed {vram} experts into VRAM in {:.1}s; {queued} more loading into RAM in the background", t.elapsed().as_secs_f64());
    }

    let vision = if model.has_vision() { model.cfg.vision.clone() } else { None };
    let image_token_id = model.cfg.image_token_id;
    if vision.is_some() {
        eprintln!("nrob-server: vision tower loaded; chat requests may carry images");
    }
    let loopback = a.host == "localhost" || a.host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    let (tx, rx) = mpsc::channel();
    let engine = engine::Engine::new(model, Arc::clone(&tok), a.chunk, a.step_below, a.layered_max, a.checkpoints, a.usage.clone(), !a.quiet);
    std::thread::Builder::new().name("model".into()).spawn(move || engine.run(rx)).map_err(nrob::Error::Io)?;

    let server = Arc::new(api::Server {
        cfg: api::Config {
            model_name: a.name.clone(),
            api_key: a.api_key.clone(),
            max_seq: a.ctx,
            thinking: a.thinking,
            effort: a.effort,
            max_tokens: a.max_tokens,
            temperature: a.temperature,
            top_p: a.top_p,
            vision,
            image_token_id,
            local_images: a.local_images.unwrap_or(loopback),
        },
        tok,
        jobs: Mutex::new(tx),
    });
    let listener = TcpListener::bind((a.host.as_str(), a.port))?;
    eprintln!("nrob-server: serving {} at http://{}:{}/v1", a.name, a.host, a.port);
    let quiet = a.quiet;
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let server = Arc::clone(&server);
        let peer = stream.peer_addr().map(|p| p.to_string()).unwrap_or_default();
        std::thread::spawn(move || {
            http::serve(stream, |req, w| {
                let t = Instant::now();
                let keep = server.handle(req, w);
                if !quiet && req.path != "/health" {
                    eprintln!("{peer} {} {} ({:.1}s)", req.method, req.path, t.elapsed().as_secs_f64());
                }
                keep
            });
        });
    }
    Ok(())
}
