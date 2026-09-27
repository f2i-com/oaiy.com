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

const HELP: &str = "Observer: --observer-model FILE.gguf --observer-device auto|INDEX --observer-vram-gb N (optional; packed weights budget defaults to 16 GiB)\n\nnrob-server: DeepSeek-V4.1-Flash behind an OpenAI-compatible API

  --host ADDR          listen address (default 127.0.0.1; 0.0.0.0 for the network)
  --port N             port (default 8000)
  --devices 1,0        CUDA devices; layers split across them (default: the
                       visible ones in order, at most two)
  --ctx N|auto         context length in tokens, prompt + reply (default 65536);
                       auto: the most the model allows
  --ram-gb N           host RAM for the expert cache (default: 80% of what is
                       free, so it leaves a fifth for everything else)
  --start NAME         which configured model to load at start (default: the one
                       --model names). Only one is resident, and a load costs 10s
                       for a GGUF or 65s for a DeepSeek checkpoint, so naming the
                       one you will ask for saves loading another to unload it
  --model DIR          checkpoint directory (required)
  --media-output-root DIR  project-local generated media and worker caches
  --image-config PATH  image/video JSON configuration or live media catalog directory
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
  --ternary-experts DIR packed experts for the default model (experimental)
  --also-ternary NAME=DIR  packed experts for a named --also model
  --also-tools-experts NAME=DIR original experts for that model's tool payloads
  --tools-experts DIR  original MXFP4 experts during generated DSML tool calls;
                       ternary elsewhere, trunk stays loaded (experimental)
  --repetition-guard on|off  stop repeated prose/reasoning blocks (default on)
  --expert-trace FILE  write JSONL token/layer/expert/precision routing records
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
  --lora NAME=DIR             PEFT LoRA/rsLoRA adapter for an Orca model alias
  --vision-projector NAME=PATH  GGUF projector, or original Qwen vision directory for Orca
  --no-vision          skip the vision tower (images are refused)
  --local-images on|off  let requests name image files on this machine (paths,
                       file:// URLs); default on when listening on loopback only
  --quiet              no per-request log
  --incognito          keep nothing of any request: no prompt states on disk, no
                       request log, and each request's state is dropped when it
                       ends (requests may also send incognito: true, or the
                       header X-NROB-Incognito: 1, one at a time)
  --backend B          GGUF backend: auto (default), cuda, webgpu or cpu. auto is
                       CUDA in a CUDA build; in nrob-server-webgpu it is the
                       first WebGPU adapter, else the CPU
  --webgpu-gb N        weights WebGPU may hold (default: 8 discrete, 2 integrated);
                       the rest run on the CPU
  --watch-stdin        exit when stdin closes: a supervisor (nrob-studio) holds
                       the other end, so the server cannot outlive it
";

fn parse_args() -> Result<Options, String> {
    let mut a = parse_args_from(std::env::args().skip(1))?;
    // Compatibility for older one-model launchers; never inherited by extras.
    if let Some(path) = std::env::var_os("DSV41_TERNARY_DIR").filter(|p| !p.is_empty()) {
        a.ternary_experts.entry(a.name.clone()).or_insert(path.into());
    }
    Ok(a)
}

fn parse_args_from(mut it: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut a = Options::default();
    let mut default_ternary = None;
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let num = |v: String| v.trim().parse::<usize>().map_err(|_| format!("{flag}: {v:?} is not a number"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--host" => a.host = val()?,
            "--image-config" => a.image_config = Some(val()?.into()),
            "--media-output-root" => a.media_output_root = Some(val()?.into()),
            "--port" => a.port = val()?.parse().map_err(|_| "--port: not a port number".to_string())?,
            "--devices" => {
                a.devices = val()?.split(',').map(|d| d.trim().parse().map_err(|_| format!("--devices: bad ordinal {d:?}"))).collect::<Result<_, _>>()?
            }
            "--ctx" => {
                let v = val()?;
                a.ctx = if v.trim() == "auto" { 0 } else { num(v)? };
            }
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
            // VENDORED-LOCAL: which of them to load at start. `--help` has
            // documented this since the option was added for coder-cli's daemon,
            // but the arm never landed here, so a direct invocation was told
            // "unknown option --start" while the documented behaviour existed.
            "--start" => a.start_model = Some(val()?),
            "--lora" => {
                let value=val()?;
                let (name,path)=value.split_once('=').filter(|(n,p)|!n.is_empty() && !p.is_empty()).ok_or("--lora wants NAME=DIR")?;
                a.lora_adapters.insert(name.into(),path.into());
            }
            "--vision-projector" => {
                let value=val()?;
                let (name,path)=value.split_once('=').filter(|(n,p)|!n.is_empty() && !p.is_empty()).ok_or("--vision-projector wants NAME=PATH")?;
                a.vision_projectors.insert(name.into(),path.into());
            }
            "--ternary-experts" => default_ternary = Some(std::path::PathBuf::from(val()?)),
            "--also-ternary" | "--also-tools-experts" => {
                let value = val()?;
                let (name, path) = value.split_once('=').filter(|(n, p)| !n.is_empty() && !p.is_empty())
                    .ok_or_else(|| format!("{flag} wants NAME=DIR"))?;
                let sources = if flag == "--also-ternary" { &mut a.ternary_experts } else { &mut a.tool_expert_sources };
                sources.insert(name.into(), path.into());
            }
            "--tools-experts" => a.tools_experts = Some(val()?.into()),
            "--repetition-guard" => a.repetition_guard = match val()?.as_str() {
                "on" => true, "off" => false,
                _ => return Err("--repetition-guard wants on or off".into()),
            },
            "--observer-model" => a.observer_model = Some(val()?.into()),
            "--observer-vram-gb" => a.observer_vram_gb = num(val()?)?,
            "--observer-device" => a.observer_device = { let v = val()?; if v == "auto" { usize::MAX } else { num(v)? } },
            "--expert-trace" => a.expert_trace = Some(val()?.into()),
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
            "--incognito" => a.incognito = true,
            "--backend" => {
                a.backend = val()?;
                if !["auto", "cuda", "webgpu", "cpu"].contains(&a.backend.as_str()) {
                    return Err("--backend wants auto, cuda, webgpu or cpu".into());
                }
            }
            "--webgpu-gb" => a.webgpu_gb = Some(num(val()?)? as u64),
            "--watch-stdin" => watch_stdin(),
            other => return Err(format!("unknown option {other} (try --help)")),
        }
    }
    if let Some(path) = default_ternary { a.ternary_experts.insert(a.name.clone(), path); }
    if a.model.as_os_str().is_empty() {
        return Err("--model DIR is required: the checkpoint directory (try --help)".into());
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn named_expert_flags_keep_paths_and_default_name() {
        let a = parse_args_from(["--model", "copy", "--ternary-experts", "packed with spaces", "--name", "default",
            "--also", "other=copy2", "--also-ternary", "other=packed2", "--also-tools-experts", "other=source"]
            .into_iter().map(str::to_owned)).unwrap();
        assert_eq!(a.ternary_experts["default"], std::path::PathBuf::from("packed with spaces"));
        assert_eq!(a.ternary_experts["other"], std::path::PathBuf::from("packed2"));
        assert_eq!(a.tool_expert_sources["other"], std::path::PathBuf::from("source"));
        assert!(parse_args_from(["--model", "copy", "--also-ternary", "bad"].into_iter().map(str::to_owned)).is_err());
    }
}

/// Exit when stdin reaches end of file: the supervising process has gone
/// (however it ended), and a model left running would hold its GPUs.
fn watch_stdin() {
    std::thread::spawn(|| {
        let mut sink = [0u8; 256];
        let mut stdin = std::io::stdin();
        while std::io::Read::read(&mut stdin, &mut sink).is_ok_and(|n| n > 0) {}
        std::process::exit(0);
    });
}

pub(crate) fn main() {
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
