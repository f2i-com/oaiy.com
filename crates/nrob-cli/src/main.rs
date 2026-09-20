//! main.rs — the nrob command line, for GGUF models.
//!
//! Every command opens a `.gguf` file through the workspace's own llama-rs
//! engine. `--budget` streams a MoE model's experts from the `.gguf` file
//! through a bounded RAM cache instead of loading them all; `--cuda` runs
//! on the GPU (a `--features cuda` build). DeepSeek-V4.1 safetensors have
//! their own engine and runner (`dsv41-cuda`, docs/DEEPSEEK_V41.md).
//!
//!   nrob run        MODEL.gguf "prompt"   generate once
//!   nrob chat       MODEL.gguf            interactive conversation
//!   nrob bench      MODEL.gguf            load / prefill / decode timings
//!   nrob info       MODEL.gguf            architecture, size, experts
//!   nrob tokenize   MODEL.gguf "text"     text -> token ids
//!   nrob detokenize MODEL.gguf "1 2 3"    token ids -> text
//!   nrob version

#![forbid(unsafe_code)]

use std::io::{self, IsTerminal, Read, Write};
use std::sync::Arc;

use llama_rs::{apply_chat_template, ChatMessage, Model};
use nrob::VERSION;

/// Write to stdout and flush, ignoring errors. Output piped into `head`
/// gets its pipe closed early; `print!` would panic on that.
fn emit(bytes: &[u8]) {
    let mut so = io::stdout().lock();
    let _ = so.write_all(bytes).and_then(|()| so.flush());
}

/// `print!` that tolerates a closed stdout (see [`emit`]).
macro_rules! out {
    ($($arg:tt)*) => { $crate::emit(format!($($arg)*).as_bytes()) };
}

mod bench;

/// A byte count: `4096`, `512K`, `64M`, `24G` or `1.5T` (binary units, any
/// case, an optional trailing `B`).
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.strip_suffix(['b', 'B']).unwrap_or(s);
    let (digits, unit) = match s.chars().last()?.to_ascii_lowercase() {
        'k' => (&s[..s.len() - 1], 1u64 << 10),
        'm' => (&s[..s.len() - 1], 1 << 20),
        'g' => (&s[..s.len() - 1], 1 << 30),
        't' => (&s[..s.len() - 1], 1 << 40),
        _ => (s, 1),
    };
    let v: f64 = digits.trim().parse().ok()?;
    let bytes = v * unit as f64;
    (bytes.is_finite() && bytes >= 0.0 && bytes < u64::MAX as f64).then_some(bytes as u64)
}

/// A finite float.
fn finite(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|v| v.is_finite())
}

/// Bytes in GB, or MB below one GB (binary units).
fn human(bytes: u64) -> String {
    const MB: f64 = (1u64 << 20) as f64;
    let mb = bytes as f64 / MB;
    if mb >= 1024.0 {
        format!("{:.2} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

/// A parameter count with the unit that suits it: 15.2 M, 30.53 B, 1.20 T.
fn params_human(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000_000 => format!("{:.2} T", n as f64 / 1e12),
        n if n >= 1_000_000_000 => format!("{:.2} B", n as f64 / 1e9),
        n => format!("{:.1} M", n as f64 / 1e6),
    }
}

/* ---- options ------------------------------------------------------------ */

#[derive(Clone, Default)]
struct Opts {
    /// --budget N: stream a MoE model's experts from the .gguf through a
    /// RAM cache, holding resident weights + cache under N bytes. 0 (the
    /// default) loads everything resident.
    budget: u64,
    /// --ctx N: KV-cache length in tokens (prompt + generated).
    ctx: u32,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: u64,
    quiet: bool,
    json: bool,
    /// --json-out FILE: bench writes the schema JSON here instead of stdout.
    json_out: Option<String>,
    /// --trace-out FILE: record the routed (layer, expert) sequence per
    /// token as JSONL (see bench::write_route_trace).
    trace_out: Option<String>,
    stats: bool,
    /// --raw: plain continuation, no chat template.
    raw: bool,
    /// --cuda: run on CUDA device 0 (build with `--features cuda`).
    cuda: bool,
    /// --vram-cache N: VRAM expert cache budget for a streamed MoE model on
    /// CUDA. Absent = min(2 GiB, 25% free VRAM) when streaming on CUDA;
    /// `0` disables. Ignored on CPU and for resident loads.
    vram_cache: Option<u64>,
    file: Option<String>,
    system: Option<String>,
    /// The model, then the command's argument (prompt, text or ids).
    pos: Vec<String>,
}

fn opts_init() -> Opts {
    Opts {
        ctx: 4096,
        max_tokens: 256,
        top_p: 1.0,
        ..Opts::default()
    }
}

/// Options that take a value.
const VALUE_FLAGS: &[&str] = &[
    "--budget", "--vram-cache", "--ctx", "-n", "--top-k", "--seed", "--temp", "--top-p",
    "--file", "--system", "--trace-out", "--json-out",
];

/// An option name, as opposed to a value: `-5` and `-.5` are numbers, and a
/// lone `-` means stdin.
fn is_flag(s: &str) -> bool {
    s.len() > 1 && s.starts_with('-') && s.parse::<f64>().is_err()
}

/// Options and positionals, in any order. Anything unrecognized is an
/// error, never a silent prompt.
fn parse_opts(args: &[String], o: &mut Opts) -> Result<(), String> {
    let mut it = args.iter().map(String::as_str);
    while let Some(a) = it.next() {
        if VALUE_FLAGS.contains(&a) {
            let v = it.next().filter(|v| !is_flag(v)).ok_or_else(|| format!("{a} needs a value"))?;
            let bad = || format!("{a}: {v:?} is not a valid value");
            match a {
                "--budget" => o.budget = parse_size(v).ok_or_else(bad)?,
                "--vram-cache" => o.vram_cache = Some(parse_size(v).ok_or_else(bad)?),
                "--ctx" => o.ctx = v.trim().parse().map_err(|_| bad())?,
                "-n" => o.max_tokens = v.trim().parse().map_err(|_| bad())?,
                "--top-k" => o.top_k = v.trim().parse().map_err(|_| bad())?,
                "--seed" => o.seed = v.trim().parse().map_err(|_| bad())?,
                "--temp" => o.temperature = finite(v).ok_or_else(bad)?,
                "--top-p" => o.top_p = finite(v).ok_or_else(bad)?,
                "--file" => o.file = Some(v.to_string()),
                "--system" => o.system = Some(v.to_string()),
                "--trace-out" => o.trace_out = Some(v.to_string()),
                _ => {
                    o.json = true;
                    o.json_out = Some(v.to_string());
                }
            }
            continue;
        }
        match a {
            "-q" | "--quiet" => o.quiet = true,
            "--json" => o.json = true,
            "--raw" => o.raw = true,
            "--stats" => o.stats = true,
            "--cuda" => o.cuda = true,
            _ if is_flag(a) => return Err(format!("unknown option {a}")),
            _ if o.pos.len() < 2 => o.pos.push(a.to_string()),
            _ => return Err(format!("unexpected argument {a:?}")),
        }
    }
    if o.temperature < 0.0 {
        return Err("--temp can't be negative".into());
    }
    if !(o.top_p > 0.0 && o.top_p <= 1.0) {
        return Err("--top-p must be in (0, 1]".into());
    }
    if o.ctx == 0 || o.max_tokens == 0 {
        return Err("--ctx and -n must be at least 1".into());
    }
    Ok(())
}

/// Parse `args` for a command whose first positional is the model; prints
/// `usage` and returns the exit code when that fails.
fn parse_cmd(args: &[String], mut o: Opts, usage: &str) -> Result<Opts, i32> {
    if let Err(m) = parse_opts(args, &mut o) {
        eprintln!("{m}");
        return Err(2);
    }
    if o.pos.is_empty() {
        eprintln!("usage: {usage}");
        return Err(2);
    }
    if let Err(m) = check_gguf(&o.pos[0]) {
        eprintln!("{m}");
        return Err(1);
    }
    Ok(o)
}

fn read_stdin() -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    io::stdin().lock().read_to_end(&mut buf).ok()?;
    Some(buf)
}

/// The command's text: the argument, `--file F` (`-` for stdin), or stdin
/// when the argument is `-` or missing and stdin is a pipe. Trailing line
/// breaks are dropped.
fn read_prompt(o: &Opts) -> Option<String> {
    let bytes = match (o.file.as_deref(), o.pos.get(1).map(String::as_str)) {
        (Some("-"), _) | (None, Some("-")) => read_stdin()?,
        (Some(f), _) => match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("cannot read {f}: {e}");
                return None;
            }
        },
        (None, Some(arg)) => return Some(arg.to_string()),
        (None, None) if !io::stdin().is_terminal() => read_stdin()?,
        (None, None) => return None,
    };
    let text = String::from_utf8_lossy(&bytes);
    Some(text.trim_end_matches(['\n', '\r']).to_string())
}

/* ---- opening a model ---------------------------------------------------- */

/// A model path must be a GGUF file: "GGUF" magic at offset 0. Checked up
/// front so a directory or a safetensors shard gets a clear message rather
/// than a parse error from deep in the reader.
fn check_gguf(path: &str) -> Result<(), String> {
    let mut magic = [0u8; 4];
    let ok = std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|()| &magic == b"GGUF");
    match ok {
        Ok(true) => Ok(()),
        Ok(false) => Err(format!(
            "{path}: not a GGUF file. nrob runs .gguf models; DeepSeek-V4.1 safetensors \
             run through dsv41-cuda (docs/DEEPSEEK_V41.md)"
        )),
        Err(e) => Err(format!("{path}: {e}")),
    }
}

/// Backend for a model: CPU by default, CUDA device 0 with `--cuda`
/// (requires `cargo build -p nrob-cli --features cuda`). The concrete CUDA
/// handle rides along so a streaming model can attach its VRAM expert
/// cache to the same device.
pub(crate) struct LlamaBackend {
    pub backend: Arc<dyn ggml_rs::Backend>,
    #[cfg(feature = "cuda")]
    pub cuda: Option<Arc<ggml_rs_cuda::CudaBackend>>,
}

fn llama_backend(o: &Opts) -> Result<LlamaBackend, String> {
    if !o.cuda {
        return Ok(LlamaBackend {
            backend: ggml_rs::default_backend(),
            #[cfg(feature = "cuda")]
            cuda: None,
        });
    }
    #[cfg(feature = "cuda")]
    {
        let b = Arc::new(ggml_rs_cuda::CudaBackend::new(0).map_err(|e| format!("cuda init: {e}"))?);
        Ok(LlamaBackend {
            backend: b.clone(),
            cuda: Some(b),
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("--cuda needs a CUDA build: cargo build --release -p nrob-cli --features cuda"
            .to_string())
    }
}

/// Attach the VRAM expert cache to a streaming model. Budget: `--vram-cache
/// N` (`0` disables); without the flag the default is min(2 GiB, 25% of
/// free VRAM) when the model streams on CUDA, disabled otherwise. No-op for
/// resident loads and CPU runs.
///
/// VENDORED-LOCAL: GLM-5.3-Flash takes the tiered form of this instead — every
/// card, sized on the card, plus the CPU tier — because its experts are 182 GB
/// and a 2 GiB slice of one card holds 140 of 12,096 records. The two are
/// mutually exclusive: both put a cache on card 0, which would charge its VRAM
/// twice and split the hits between them.
pub(crate) fn maybe_enable_vram_cache(o: &Opts, lb: &LlamaBackend, model: &mut Model) {
    #[cfg(feature = "cuda")]
    {
        if let Model::Glm5Next(g) = model {
            // `--vram-cache 0` still means "no VRAM tier at all".
            if o.vram_cache == Some(0) {
                return;
            }
            let Some(cuda) = &lb.cuda else {
                if !o.quiet {
                    eprintln!("nrob: the GLM expert tier needs --cuda");
                }
                return;
            };
            // Card 0 is the trunk's own backend; the rest are opened here.
            let mut cards = vec![Arc::clone(cuda)];
            cards.extend(llama_rs::glm5next::device::extra_cards(0));
            let cap = o.vram_cache.unwrap_or(0) as usize;
            match g.enable_tiering(cards, cap) {
                Ok(()) => {
                    if !o.quiet {
                        let (budgets, layers) = g.tier_layout();
                        let gb: Vec<String> =
                            budgets.iter().map(|b| format!("{:.1}", *b as f64 / 1e9)).collect();
                        eprintln!(
                            "nrob: expert tier {} card(s), VRAM {} GB, MoE layers {:?}",
                            budgets.len(),
                            gb.join("+"),
                            layers
                        );
                    }
                }
                Err(e) => eprintln!("nrob: GLM expert tier unavailable: {e}"),
            }
            return;
        }
        let Some(shared) = model.stream_shared() else {
            if o.vram_cache.is_some() && !o.quiet {
                eprintln!("nrob: --vram-cache applies to streamed MoE models (--budget)");
            }
            return;
        };
        let budget = match o.vram_cache {
            Some(0) => return, // explicitly disabled
            Some(n) => Some(n),
            None => lb.cuda.as_ref().and_then(|c| {
                ggml_rs::Backend::vram_status(c.as_ref())
                    .map(|(free, _)| (2u64 << 30).min(free as u64 / 4))
            }),
        };
        let Some(budget) = budget else { return };
        let Some(cuda) = &lb.cuda else {
            if o.vram_cache.is_some() {
                eprintln!("nrob: --vram-cache needs --cuda");
            }
            return;
        };
        match shared.enable_device_cache(Arc::clone(cuda), budget as usize) {
            Ok(()) => {
                if !o.quiet {
                    eprintln!("nrob: VRAM expert cache {}", human(budget));
                }
            }
            Err(e) => eprintln!("nrob: VRAM expert cache unavailable: {e}"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (lb, model);
        if o.vram_cache.is_some() {
            eprintln!(
                "nrob: --vram-cache needs a CUDA build: cargo build --release -p nrob-cli --features cuda"
            );
        }
    }
}

/// Load the model at `o.pos[0]` on `backend`: resident, or with streamed
/// experts when `--budget` is set.
pub(crate) fn load_model(o: &Opts, backend: Arc<dyn ggml_rs::Backend>) -> Result<Model, String> {
    let path = &o.pos[0];
    let model = if o.budget > 0 {
        Model::open_streaming(path, backend, o.budget)
    } else {
        let g = gguf::GgufFile::open(path).map_err(|e| format!("open: {e}"))?;
        Model::load(&g, backend)
    };
    model.map_err(|e| format!("open: {e}"))
}

/// Backend + model + the VRAM cache, with the one-line status reports.
fn open_model(o: &Opts) -> Result<Model, i32> {
    let lb = llama_backend(o).map_err(|m| {
        eprintln!("{m}");
        2
    })?;
    let mut model = load_model(o, lb.backend.clone()).map_err(|m| {
        eprintln!("{m}");
        1
    })?;
    if o.cuda && !o.quiet {
        eprintln!("nrob: backend {}", model.backend().name());
    }
    if let Some(shared) = model.stream_shared() {
        if !o.quiet {
            eprintln!(
                "nrob: streaming experts from the .gguf (record {} bytes; resident ~{}, \
                 RAM expert cache {})",
                shared.record_bytes(),
                human(shared.resident_est_bytes()),
                human(shared.cache_budget_bytes() as u64),
            );
        }
    }
    maybe_enable_vram_cache(o, &lb, &mut model);
    Ok(model)
}

pub(crate) fn sample_params(o: &Opts) -> llama_rs::SampleParams {
    llama_rs::SampleParams {
        temperature: o.temperature,
        top_p: if o.top_p < 1.0 { Some(o.top_p) } else { None },
        top_k: if o.top_k > 0 { Some(o.top_k as usize) } else { None },
        seed: o.seed,
        ..llama_rs::SampleParams::default()
    }
}

/// Whether the GGUF ships a chat template, i.e. is an instruct model. Base
/// models (no template) get plain continuation.
fn has_chat_template(path: &str) -> bool {
    gguf::GgufFile::open(path)
        .map(|g| g.get_str("tokenizer.chat_template").is_ok())
        .unwrap_or(false)
}

/// Stream one generation to stdout; returns (ids, seconds) or the exit code
/// when an expert read failed mid-run.
fn generate_to_stdout(
    model: &Model,
    o: &Opts,
    ids: &[u32],
    kv: &mut llama_rs::KvCache,
) -> Result<(Vec<u32>, f64), i32> {
    let started = std::time::Instant::now();
    let mut out_ids: Vec<u32> = Vec::new();
    for tok in model.generate_with_kv(ids, sample_params(o), o.max_tokens as usize, kv) {
        if !o.quiet {
            emit(model.tokenizer().decode(&[tok]).as_bytes());
        }
        out_ids.push(tok);
        // The forward pass is infallible by signature, so a streaming read
        // failure is recorded on the model instead of returned: poll it.
        if let Some(e) = model.expert_stream_error() {
            eprintln!("\nexpert stream: {e}");
            return Err(1);
        }
    }
    Ok((out_ids, started.elapsed().as_secs_f64()))
}

fn print_stats(model: &Model, n: usize, sec: f64) {
    if n > 0 {
        eprintln!("[{n} tokens, {sec:.2} s, {:.2} tok/s]", n as f64 / sec.max(1e-9));
    }
    if let Some(st) = model.expert_cache_stats() {
        let acc = st.hits + st.misses;
        eprintln!(
            "experts: {} hit / {} miss = {:.1}% hit, {:.2} GiB read, {} evictions",
            st.hits,
            st.misses,
            100.0 * st.hits as f64 / acc.max(1) as f64,
            st.bytes_read as f64 / (1u64 << 30) as f64,
            st.evictions,
        );
    }
    #[cfg(feature = "cuda")]
    if let Some(ds) = model.device_cache_stats() {
        let tot = ds.bytes_hit + ds.h2d_bytes;
        eprintln!(
            "vram cache: {} hit / {} miss = {:.1}% byte hit, {:.2} GiB H2D, \
             {} evictions, {} entries ({} / {})",
            ds.hits,
            ds.misses,
            100.0 * ds.bytes_hit as f64 / tot.max(1) as f64,
            ds.h2d_bytes as f64 / (1u64 << 30) as f64,
            ds.evictions,
            ds.entries,
            human(ds.bytes_used),
            human(ds.budget_bytes),
        );
    }
}

/* ---- commands ----------------------------------------------------------- */

fn cmd_run(args: &[String]) -> i32 {
    let o = match parse_cmd(args, opts_init(), "nrob run MODEL.gguf [\"prompt\" | - | --file F] [options]") {
        Ok(o) => o,
        Err(rc) => return rc,
    };
    let prompt = match read_prompt(&o) {
        Some(p) if !p.is_empty() => p,
        _ => {
            eprintln!("no prompt: give one as an argument, with --file, or on stdin");
            return 2;
        }
    };
    let templated = !o.raw && has_chat_template(&o.pos[0]);
    let model = match open_model(&o) {
        Ok(m) => m,
        Err(rc) => return rc,
    };

    // An instruct model is asked a question; a base model (or --raw)
    // continues the text.
    let ids = if templated {
        let mut msgs = Vec::new();
        if let Some(s) = o.system.as_deref().filter(|s| !s.is_empty()) {
            msgs.push(ChatMessage::system(s));
        }
        msgs.push(ChatMessage::user(prompt));
        model.encode_chat(&msgs, true)
    } else {
        model.tokenizer().encode(&prompt, true).map_err(llama_rs::LlamaError::from)
    };
    let ids = match ids {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tokenize: {e}");
            return 1;
        }
    };
    if ids.len() + o.max_tokens as usize > o.ctx as usize {
        eprintln!(
            "prompt ({} tokens) + -n {} does not fit --ctx {}",
            ids.len(),
            o.max_tokens,
            o.ctx
        );
        return 2;
    }

    // --trace-out: record the routed (layer, expert) sequence per token.
    // The sink fires per moe_forward call; the ids are re-derived from the
    // router logits inside llama-rs (moe.rs).
    let trace_buf = o.trace_out.as_ref().map(|_| {
        let buf = Arc::new(std::sync::Mutex::new(Vec::<Vec<Vec<u32>>>::new()));
        let b = Arc::clone(&buf);
        llama_rs::moe::set_route_trace_sink(Some(Box::new(move |calls| {
            b.lock().unwrap_or_else(|e| e.into_inner()).push(calls.to_vec());
        })));
        buf
    });

    let mut kv = model.new_kv_cache(o.ctx as usize);
    let (out_ids, sec) = match generate_to_stdout(&model, &o, &ids, &mut kv) {
        Ok(r) => r,
        Err(rc) => return rc,
    };
    out!("\n");

    if let Some(buf) = &trace_buf {
        llama_rs::moe::set_route_trace_sink(None);
        let path = o.trace_out.as_deref().unwrap_or_default();
        let calls = buf.lock().unwrap_or_else(|e| e.into_inner());
        if calls.is_empty() {
            eprintln!("nrob: --trace-out: no MoE routing recorded (model has no MoE blocks)");
        }
        let n_layers = model.config().n_layers;
        match bench::write_route_trace(path, &o.pos[0], &calls, n_layers, ids.len()) {
            Ok(()) => eprintln!("nrob: route trace -> {path}"),
            Err(e) => {
                eprintln!("trace write {path}: {e}");
                return 1;
            }
        }
    }
    if o.stats {
        print_stats(&model, out_ids.len(), sec);
        eprintln!("ids: {out_ids:?}");
    }
    0
}

fn cmd_chat(args: &[String]) -> i32 {
    let o = match parse_cmd(args, opts_init(), "nrob chat MODEL.gguf [--system TEXT] [options]") {
        Ok(o) => o,
        Err(rc) => return rc,
    };
    let templated = has_chat_template(&o.pos[0]);
    let model = match open_model(&o) {
        Ok(m) => m,
        Err(rc) => return rc,
    };
    if !templated && !o.quiet {
        eprintln!(
            "nrob: this GGUF has no chat template (a base model?); using the {} \
             turn format anyway",
            model.config().arch.name()
        );
    }
    out!(
        "nrob {} · {} ({} layers)\n/reset clears the conversation, /stats prints counters, Ctrl-D exits\n",
        VERSION,
        model.config().arch.name(),
        model.config().n_layers
    );

    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(s) = o.system.as_deref().filter(|s| !s.is_empty()) {
        history.push(ChatMessage::system(s));
    }
    let mut kv = model.new_kv_cache(o.ctx as usize);
    let stdin = io::stdin();
    loop {
        out!("\n> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.is_empty() {
            continue;
        }
        if line == "/reset" {
            history.retain(|m| m.role == llama_rs::Role::System);
            out!("(conversation cleared)\n");
            continue;
        }
        if line == "/stats" {
            print_stats(&model, 0, 0.0);
            continue;
        }
        history.push(ChatMessage::user(line));
        // The whole conversation is re-rendered and re-prefilled each turn:
        // simple and always consistent with the template, at the cost of
        // re-reading the earlier turns.
        let text = apply_chat_template(&model.config().arch, &history, true);
        let ids = match model.tokenizer().encode(&text, false) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("tokenize: {e}");
                history.pop();
                continue;
            }
        };
        if ids.len() + o.max_tokens as usize > o.ctx as usize {
            out!("(conversation is longer than --ctx {}; /reset to start over)\n", o.ctx);
            history.pop();
            continue;
        }
        kv.reset();
        let (out_ids, sec) = match generate_to_stdout(&model, &o, &ids, &mut kv) {
            Ok(r) => r,
            Err(rc) => return rc,
        };
        out!("\n");
        if o.stats {
            print_stats(&model, out_ids.len(), sec);
        }
        history.push(ChatMessage::assistant(model.tokenizer().decode(&out_ids)));
    }
    0
}

/// `nrob bench MODEL` — one cell of the benchmark matrix (resident or
/// streamed, CPU or CUDA), timed by the bench module, which also owns the
/// JSON result schema.
///
/// `--json` alone prints the schema JSON to stdout; `--json FILE` writes it
/// to a file and keeps the human summary. `--json` takes no value in
/// parse_opts, so the optional file is split out here: a non-option token
/// right after `--json` becomes the file.
fn cmd_bench(args: &[String]) -> i32 {
    let mut rewritten: Vec<String> = Vec::with_capacity(args.len());
    for (i, a) in args.iter().enumerate() {
        if a == "--json" && i + 1 < args.len() && !is_flag(&args[i + 1]) {
            rewritten.push("--json-out".to_string());
        } else {
            rewritten.push(a.clone());
        }
    }
    let base = Opts {
        max_tokens: 64,
        ..opts_init()
    };
    match parse_cmd(&rewritten, base, "nrob bench MODEL.gguf [-n N] [--budget N] [--cuda] [--json [FILE]]") {
        Ok(o) => bench::run(&o),
        Err(rc) => rc,
    }
}

/// Architecture, size and expert layout, read from the GGUF header alone
/// (no weights are loaded).
fn cmd_info(args: &[String]) -> i32 {
    let o = match parse_cmd(args, opts_init(), "nrob info MODEL.gguf [--json]") {
        Ok(o) => o,
        Err(rc) => return rc,
    };
    let g = match gguf::GgufFile::open(&o.pos[0]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("open: {e}");
            return 1;
        }
    };
    let arch = g.architecture().unwrap_or("unknown").to_string();
    let key = |k: &str| g.get_u64(&format!("{arch}.{k}")).ok();
    let name = g.get_str("general.name").unwrap_or("").to_string();
    let params: u64 = g.tensors().iter().map(|t| t.numel()).sum();
    let bytes: u64 = g.tensors().iter().map(|t| t.nbytes()).sum();
    let experts = key("expert_count").unwrap_or(0);
    let top_k = key("expert_used_count").unwrap_or(0);
    let templated = g.get_str("tokenizer.chat_template").is_ok();
    // weight bytes per quant type, largest first
    let mut quant: Vec<(String, u64)> = Vec::new();
    for t in g.tensors() {
        let k = format!("{:?}", t.dtype);
        match quant.iter_mut().find(|(n, _)| *n == k) {
            Some((_, b)) => *b += t.nbytes(),
            None => quant.push((k, t.nbytes())),
        }
    }
    quant.sort_by(|a, b| b.1.cmp(&a.1));
    let quant_s: Vec<String> = quant
        .iter()
        .map(|(n, b)| format!("{n} {:.0}%", 100.0 * *b as f64 / bytes.max(1) as f64))
        .collect();

    if o.json {
        out!(
            "{{\"engine\":\"{}\",\"arch\":\"{}\",\"name\":\"{}\",\"gguf_version\":{},\"tensors\":{},\
             \"params\":{},\"weight_bytes\":{},\"layers\":{},\"hidden\":{},\"context\":{},\
             \"experts\":{},\"top_k\":{},\"chat_template\":{}}}\n",
            VERSION,
            bench::esc(&arch),
            bench::esc(&name),
            g.version(),
            g.tensors().len(),
            params,
            bytes,
            key("block_count").unwrap_or(0),
            key("embedding_length").unwrap_or(0),
            key("context_length").unwrap_or(0),
            experts,
            top_k,
            templated
        );
        return 0;
    }
    out!("{}\n\n", o.pos[0]);
    if !name.is_empty() {
        out!("  name          {name}\n");
    }
    out!("  arch          {arch} (GGUF v{})\n", g.version());
    out!("  layers        {}\n", key("block_count").unwrap_or(0));
    out!("  hidden        {}\n", key("embedding_length").unwrap_or(0));
    out!("  context       {}\n", key("context_length").unwrap_or(0));
    if experts > 0 {
        out!("  experts       {experts} (top-{top_k}); stream them with --budget\n");
    }
    out!("  parameters    {} in {} tensors\n", params_human(params), g.tensors().len());
    out!("  weights       {} ({})\n", human(bytes), quant_s.join(", "));
    out!("  chat template {}\n", if templated { "yes" } else { "no (base model)" });
    0
}

/// Tokenize and detokenize with the GGUF's own tokenizer (no weights are
/// loaded): the round trip every prompt goes through, exposed for checking
/// what the model actually sees.
fn cmd_tokens(args: &[String], decode: bool) -> i32 {
    let usage = if decode {
        "nrob detokenize MODEL.gguf [ids | - | --file F]"
    } else {
        "nrob tokenize MODEL.gguf [text | - | --file F]"
    };
    let o = match parse_cmd(args, opts_init(), usage) {
        Ok(o) => o,
        Err(rc) => return rc,
    };
    let text = match read_prompt(&o) {
        Some(t) if !t.is_empty() => t,
        _ => {
            eprintln!("nothing to read");
            return 2;
        }
    };
    let tok = match gguf::GgufFile::open(&o.pos[0]).map_err(|e| e.to_string()).and_then(|g| {
        tokenizer::Tokenizer::from_gguf(&g).map_err(|e| e.to_string())
    }) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("open: {e}");
            return 1;
        }
    };

    if decode {
        let mut ids: Vec<u32> = Vec::new();
        for t in text.split([' ', ',', '\t', '\n']).filter(|t| !t.is_empty()) {
            match t.trim().parse::<u32>() {
                Ok(id) if (id as usize) < tok.vocab_size() => ids.push(id),
                _ => {
                    eprintln!("detokenize: {t:?} is not a token id of this vocabulary");
                    return 2;
                }
            }
        }
        emit(tok.decode(&ids).as_bytes());
        out!("\n");
        return 0;
    }
    let ids = match tok.encode(&text, false) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tokenize: {e}");
            return 1;
        }
    };
    let list: Vec<String> = ids.iter().map(u32::to_string).collect();
    if o.json {
        out!("{{\"n\":{},\"ids\":[{}]}}\n", ids.len(), list.join(","));
    } else {
        out!("{}\n", list.join(" "));
    }
    0
}

fn help() {
    out!(
        "nrob {} — NVMe · RAM · On-GPU · Broker: run GGUF models bigger than your GPU\n\
         \n\
         \x20 nrob run        MODEL.gguf \"prompt\"   generate once\n\
         \x20 nrob chat       MODEL.gguf            interactive conversation\n\
         \x20 nrob bench      MODEL.gguf            load / prefill / decode timings\n\
         \x20 nrob info       MODEL.gguf            architecture, size, experts\n\
         \x20 nrob tokenize   MODEL.gguf \"text\"     text -> token ids\n\
         \x20 nrob detokenize MODEL.gguf \"1 2 3\"    token ids -> text\n\
         \x20 nrob version\n\
         \n\
         The prompt may be an argument, a file (--file F), or stdin —\n\
         given as - or simply piped in.\n\
         \n\
         options: -n N  --temp F  --top-p F  --top-k N  --seed N  --ctx N\n\
         \x20        --system TEXT  --raw  --stats  -q  --json\n\
         \x20        --budget N  --cuda  --vram-cache N  --trace-out F\n\
         \x20 --budget 24G   stream a MoE model's experts from the .gguf through\n\
         \x20                a RAM cache, keeping resident weights + cache under\n\
         \x20                24 GB (default: load everything)\n\
         \x20 --cuda         run on CUDA device 0 (build with --features cuda)\n\
         \x20 --vram-cache N VRAM expert cache for a streamed model on CUDA\n\
         \x20                (0 disables; default min(2 GiB, 25% free VRAM))\n\
         \x20 --raw          plain continuation, no chat template (the default\n\
         \x20                for a GGUF without one)\n\
         \x20 --json         machine-readable output for info, tokenize and\n\
         \x20                bench; `bench --json FILE` writes it to FILE\n\
         \x20 --trace-out F  on `run`, record the routed (layer, expert)\n\
         \x20                sequence per token as JSONL\n\
         \n\
         DeepSeek-V4.1 safetensors: see docs/DEEPSEEK_V41.md\n",
        VERSION
    );
}

fn run(args: &[String]) -> i32 {
    if args.is_empty() {
        help();
        return 2;
    }
    match args[0].as_str() {
        "-h" | "--help" | "help" => {
            help();
            0
        }
        "version" | "--version" | "-V" => {
            out!(
                "nrob {} ({}{})\n",
                VERSION,
                bench::REVISION.get(..12).unwrap_or(bench::REVISION),
                if cfg!(feature = "cuda") { ", cuda" } else { "" }
            );
            0
        }
        "run" => cmd_run(&args[1..]),
        "chat" => cmd_chat(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "info" => cmd_info(&args[1..]),
        "tokenize" => cmd_tokens(&args[1..], false),
        "detokenize" => cmd_tokens(&args[1..], true),
        other => {
            eprintln!("unknown command '{other}' (try --help)");
            2
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}
