//! nrob-server: DeepSeek-V4.1-Flash behind an OpenAI-compatible HTTP API,
//! for chat clients and coding harnesses.
//!
//!   nrob-server --model DIR [--port 8000] [--host 127.0.0.1] [--devices 1,0] ...
//!
//! Point a harness at `http://127.0.0.1:8000/v1` with any API key (or the
//! one given with `--api-key`) and the model name `deepseek-v4.1-flash`.
//! Run `nrob-server --help` for every option.

#![forbid(unsafe_code)]

use nrob_server::Options;

const HELP: &str = "nrob-server: DeepSeek-V4.1-Flash behind an OpenAI-compatible API

  --host ADDR          listen address (default 127.0.0.1; 0.0.0.0 for the network)
  --port N             port (default 8000)
  --devices 1,0        CUDA devices; layers split across them (default: the
                       visible ones in order, at most two)
  --ctx N              context length in tokens, prompt + reply (default 65536)
  --ram-gb N           host RAM for the expert cache (default 140)
  --model DIR          checkpoint directory (required)
  --engram-meta FILE   the Engram precompute, engram_meta.safetensors (default: in
                       the checkpoint directory)
  --usage FILE         expert usage profile: warms the caches at start, updated
                       after every request (default: none)
  --name NAME          model name clients use (default deepseek-v4.1-flash)
  --also NAME=PATH     another model a client may ask for by name; repeatable.
                       A .gguf (or a directory holding one) is served through
                       llama-rs, anything else as a DeepSeek checkpoint. Only one
                       is resident: asking for another unloads the current one,
                       which takes as long as a load. /v1/models says which is
                       loaded.
  --api-key KEY        require `Authorization: Bearer KEY`
  --thinking           reason before answering unless a request says otherwise
                       (default: answer directly; requests turn reasoning on with
                       reasoning_effort, or thinking: {type: enabled})
  --effort N           default reasoning effort, 1-100 (default 75)
  --max-tokens N       reply length when a request gives none (default 8192)
  --temperature F      default temperature (default 0.6)
  --top-p F            default top-p (default 0.95)
  --step-below N       prompt stretches shorter than this run one token at a
                       time through the decode path (default 512)
  --layered-max N      longer stretches run layer by layer, every expert read
                       once per pass, up to N tokens a pass (default 20480;
                       halved if a pass runs out of VRAM)
  --chunk N            attention sub-chunk of a layered pass (default 1024)
  --vram-headroom-gb F VRAM kept free for activations; the rest caches experts
                       (default 2)
  --checkpoints N      prefix-cache checkpoints kept (~10 MB each, default 256)
  --prompt-cache DIR   keep prompt states in DIR between runs, so a restart does
                       not read a system prompt (or a conversation it resumes)
                       again (default: not kept)
  --prompt-cache-gb F  disk the prompt states may take (default 4)
  --cpu-threads N      CPU threads for experts that miss VRAM (default 24; 0 = off)
  --no-vision          skip the vision tower (saves ~1 GB of VRAM; images are refused)
  --local-images on|off  let requests name image files on this machine (paths,
                       file:// URLs); default on when listening on loopback only
  --quiet              no per-request log
";

fn parse_args() -> Result<Options, String> {
    let mut a = Options::default();
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
            "--engram-meta" => a.engram_meta = Some(val()?.into()),
            "--usage" => {
                let v = val()?;
                a.usage = if v == "off" { None } else { Some(v.into()) };
            }
            "--name" => a.name = val()?,
            // VENDORED-LOCAL: more than one model, switched on demand.
            "--also" => {
                let spec = val()?;
                let (name, path) = spec.split_once('=').ok_or_else(|| {
                    format!("--also wants NAME=PATH, got {spec}")
                })?;
                if name.is_empty() || path.is_empty() {
                    return Err(format!("--also wants NAME=PATH, got {spec}"));
                }
                a.extra_models.push((name.to_string(), path.into()));
            }
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
            "--prompt-cache" => a.prompt_cache = Some(val()?.into()),
            "--prompt-cache-gb" => a.prompt_cache_gb = val()?.parse().map_err(|_| "--prompt-cache-gb: not a number".to_string())?,
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
    if a.model.as_os_str().is_empty() {
        return Err("--model DIR is required: the checkpoint directory (try --help)".into());
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

fn run(o: Options) -> nrob::Result<()> {
    nrob_server::start(o)?.wait();
    Ok(())
}
