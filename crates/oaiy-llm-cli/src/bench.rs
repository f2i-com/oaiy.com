//! bench.rs — unified benchmark phases and the machine-readable result
//! schema of docs/ROADMAP.md Phase 0 ("Benchmark specification" /
//! "Required result fields").
//!
//! One invocation covers one cell of the benchmark matrix: a GGUF model,
//! resident or with streamed experts (`--budget`), on CPU or CUDA (`--cuda`
//! in a `--features cuda` build). Model load, TTFT, prefill and
//! steady-state decode are timed separately, with the backend synchronized
//! at every timing boundary (`Backend::synchronize`, VENDORED-LOCAL in
//! ggml-rs).
//!
//! The JSON schema is the deliverable; counters that no public API exposes
//! yet are emitted as `null` with an entry in `notes` saying why, so CI can
//! diff the schema across waves while the counters get deeper.

use std::io::Write;
use std::time::Instant;

use crate::{human, llama_backend, load_model, maybe_enable_vram_cache, sample_params, Opts};

/// Fixed bench prompt, identified in JSON so numbers across runs compare
/// like with like.
const PROMPT_ID: &str = "bench-default-v1";
const PROMPT: &str =
    "Write a C function that parses a JSON array of integers and explain its edge cases.";

/// Git revision baked in by build.rs (`OAIY_GIT_SHA`); "unknown" without git.
pub(crate) const REVISION: &str = match option_env!("OAIY_GIT_SHA") {
    Some(s) => s,
    None => "unknown",
};

/* ---- small probes -------------------------------------------------------- */

/// Peak resident set ("high water mark") of this process, or None where std
/// cannot read it. Same constraint as `oaiy_engine::backend::physical_ram`: std
/// exposes no RSS query, so Linux reads /proc and everything else reports
/// unknown (null in JSON, with a note).
fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(kb) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb.parse::<u64>() {
                        return Some(kb.saturating_mul(1024));
                    }
                }
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Byte size of the model file.
fn model_total_bytes(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Nearest-rank percentile of an already-sorted sample.
fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.max(1).min(sorted.len()) - 1])
}

/* ---- JSON ---------------------------------------------------------------- */

/// The roadmap's required result fields plus stage durations, run
/// parameters and a hardware block. `Option` fields serialize as `null`;
/// every null must have a matching entry in `notes`.
#[derive(Default)]
pub struct BenchResult {
    // provenance
    model_path: String,
    model_format: String,
    phase: String,
    backend: String,
    model_total_bytes: u64,
    // required fields
    model_hash: Option<String>,
    active_expert_bytes_per_token: Option<u64>,
    resident_host_bytes: Option<u64>,
    resident_device_bytes: Vec<u64>,
    kv_bytes: Vec<u64>,
    host_cache_byte_hit_rate: Option<f64>,
    device_cache_byte_hit_rate: Vec<f64>,
    ssd_bytes_per_token: Option<f64>,
    host_copy_bytes_per_token: Option<u64>,
    h2d_bytes_per_token: Vec<u64>,
    kernel_launches_per_token: Option<u64>,
    host_allocations_per_token: Option<u64>,
    device_allocations_per_token: Option<u64>,
    prefill_tokens_per_second: Option<f64>,
    decode_tokens_per_second: Option<f64>,
    ttft_ms: Option<f64>,
    inter_token_ms_p50: Option<f64>,
    inter_token_ms_p95: Option<f64>,
    inter_token_ms_p99: Option<f64>,
    peak_rss_bytes: Option<u64>,
    peak_vram_bytes: Vec<u64>,
    // stage durations
    load_ms: Option<f64>,
    prefill_ms: Option<f64>,
    decode_ms: Option<f64>,
    // run parameters
    prompt_tokens: u64,
    decode_tokens: u64,
    max_new_tokens: u64,
    budget_bytes: u64,
    ctx_tokens: u64,
    temperature: f64,
    seed: u64,
    // hardware
    hw_device: String,
    hw_cpu_threads: usize,
    hw_physical_ram_bytes: u64,
    notes: Vec<String>,
}

pub(crate) fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn fnum(v: Option<f64>) -> String {
    match v {
        Some(v) if v.is_finite() => format!("{v:.4}"),
        _ => "null".to_string(),
    }
}

fn unum(v: Option<u64>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string())
}

fn uarr(v: &[u64]) -> String {
    let inner: Vec<String> = v.iter().map(u64::to_string).collect();
    format!("[{}]", inner.join(","))
}

fn farr(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|&x| fnum(Some(x))).collect();
    format!("[{}]", inner.join(","))
}

impl BenchResult {
    fn new(o: &Opts, phase: &str) -> Self {
        let mut notes = Vec::new();
        if peak_rss_bytes().is_none() {
            notes.push(
                "peak_rss_bytes: std exposes no RSS query on this OS (only Linux /proc is read)"
                    .to_string(),
            );
        }
        BenchResult {
            model_path: o.pos[0].clone(),
            model_format: "gguf".to_string(),
            phase: phase.to_string(),
            model_total_bytes: model_total_bytes(&o.pos[0]),
            max_new_tokens: u64::from(o.max_tokens),
            budget_bytes: o.budget,
            ctx_tokens: u64::from(o.ctx),
            temperature: f64::from(o.temperature),
            seed: o.seed,
            hw_cpu_threads: oaiy_engine::backend::hardware_concurrency(),
            hw_physical_ram_bytes: oaiy_engine::backend::physical_ram(),
            notes,
            ..BenchResult::default()
        }
    }

    /// Notes for counters no public API exposes yet, so the schema stays
    /// complete while the counters land.
    fn note_unmeasured(&mut self) {
        let n = &mut self.notes;
        if self.model_hash.is_none() {
            n.push("model_hash: no sha256 in a std-only build; model_total_bytes is the size \
                    cross-check until a hash lands".into());
        }
        if self.kv_bytes.is_empty() {
            n.push("kv_bytes: no KV cache was allocated".into());
        }
        if self.device_cache_byte_hit_rate.is_empty() {
            n.push("device_cache_byte_hit_rate: no VRAM expert cache active (needs a streamed \
                    MoE model on CUDA: --budget with --cuda)"
                .into());
        }
        if self.host_copy_bytes_per_token.is_none() {
            n.push("host_copy_bytes_per_token: expert-record rebuild/copy bytes are not \
                    instrumented in expert_stream".into());
        }
        if self.h2d_bytes_per_token.is_empty() {
            n.push("h2d_bytes_per_token: no VRAM expert cache active (H2D upload bytes are \
                    counted on the device-cache miss path; needs a streamed MoE model on CUDA)"
                .into());
        }
        if self.kernel_launches_per_token.is_none() {
            n.push("kernel_launches_per_token: backends do not count launches".into());
        }
        if self.host_allocations_per_token.is_none() {
            n.push("host_allocations_per_token: no allocation counting yet".into());
        }
        if self.device_allocations_per_token.is_none() {
            n.push("device_allocations_per_token: no device-allocation counting yet".into());
        }
        if self.peak_vram_bytes.is_empty() {
            n.push("peak_vram_bytes: only a post-load free-VRAM delta is available (see \
                    resident_device_bytes); no peak query is exposed".into());
        }
        if self.ttft_ms.is_some() {
            n.push("ttft_ms includes the iterator's pipelined pre-sample of the second token"
                .into());
        }
    }

    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(2048);
        s.push_str("{\n");
        s.push_str("  \"schema\": \"oaiy-bench/1\",\n");
        s.push_str(&format!("  \"revision\": \"{}\",\n", esc(REVISION)));
        s.push_str(&format!("  \"model_path\": \"{}\",\n", esc(&self.model_path)));
        s.push_str(&format!("  \"model_format\": \"{}\",\n", esc(&self.model_format)));
        s.push_str(&format!("  \"phase\": \"{}\",\n", esc(&self.phase)));
        s.push_str(&format!("  \"backend\": \"{}\",\n", esc(&self.backend)));
        s.push_str(&format!(
            "  \"model_hash\": {},\n",
            self.model_hash
                .as_ref()
                .map(|h| format!("\"{}\"", esc(h)))
                .unwrap_or_else(|| "null".to_string())
        ));
        s.push_str(&format!("  \"model_total_bytes\": {},\n", self.model_total_bytes));
        s.push_str(&format!(
            "  \"active_expert_bytes_per_token\": {},\n",
            unum(self.active_expert_bytes_per_token)
        ));
        s.push_str(&format!(
            "  \"resident_host_bytes\": {},\n",
            unum(self.resident_host_bytes)
        ));
        s.push_str(&format!(
            "  \"resident_device_bytes\": {},\n",
            uarr(&self.resident_device_bytes)
        ));
        s.push_str(&format!("  \"kv_bytes\": {},\n", uarr(&self.kv_bytes)));
        s.push_str(&format!(
            "  \"host_cache_byte_hit_rate\": {},\n",
            fnum(self.host_cache_byte_hit_rate)
        ));
        s.push_str(&format!(
            "  \"device_cache_byte_hit_rate\": {},\n",
            farr(&self.device_cache_byte_hit_rate)
        ));
        s.push_str(&format!(
            "  \"ssd_bytes_per_token\": {},\n",
            fnum(self.ssd_bytes_per_token)
        ));
        s.push_str(&format!(
            "  \"host_copy_bytes_per_token\": {},\n",
            unum(self.host_copy_bytes_per_token)
        ));
        s.push_str(&format!(
            "  \"h2d_bytes_per_token\": {},\n",
            uarr(&self.h2d_bytes_per_token)
        ));
        s.push_str(&format!(
            "  \"kernel_launches_per_token\": {},\n",
            unum(self.kernel_launches_per_token)
        ));
        s.push_str(&format!(
            "  \"host_allocations_per_token\": {},\n",
            unum(self.host_allocations_per_token)
        ));
        s.push_str(&format!(
            "  \"device_allocations_per_token\": {},\n",
            unum(self.device_allocations_per_token)
        ));
        s.push_str(&format!(
            "  \"prefill_tokens_per_second\": {},\n",
            fnum(self.prefill_tokens_per_second)
        ));
        s.push_str(&format!(
            "  \"decode_tokens_per_second\": {},\n",
            fnum(self.decode_tokens_per_second)
        ));
        s.push_str(&format!("  \"ttft_ms\": {},\n", fnum(self.ttft_ms)));
        s.push_str(&format!(
            "  \"inter_token_ms_p50\": {},\n",
            fnum(self.inter_token_ms_p50)
        ));
        s.push_str(&format!(
            "  \"inter_token_ms_p95\": {},\n",
            fnum(self.inter_token_ms_p95)
        ));
        s.push_str(&format!(
            "  \"inter_token_ms_p99\": {},\n",
            fnum(self.inter_token_ms_p99)
        ));
        s.push_str(&format!(
            "  \"peak_rss_bytes\": {},\n",
            unum(self.peak_rss_bytes)
        ));
        s.push_str(&format!(
            "  \"peak_vram_bytes\": {},\n",
            uarr(&self.peak_vram_bytes)
        ));
        s.push_str("  \"stages\": {");
        s.push_str(&format!("\"load_ms\": {}, ", fnum(self.load_ms)));
        s.push_str(&format!("\"prefill_ms\": {}, ", fnum(self.prefill_ms)));
        s.push_str(&format!("\"decode_ms\": {}}},\n", fnum(self.decode_ms)));
        s.push_str("  \"run\": {");
        s.push_str(&format!("\"prompt_id\": \"{PROMPT_ID}\", "));
        s.push_str(&format!("\"prompt_tokens\": {}, ", self.prompt_tokens));
        s.push_str(&format!("\"decode_tokens\": {}, ", self.decode_tokens));
        s.push_str(&format!("\"max_new_tokens\": {}, ", self.max_new_tokens));
        s.push_str(&format!("\"budget_bytes\": {}, ", self.budget_bytes));
        s.push_str(&format!("\"ctx_tokens\": {}, ", self.ctx_tokens));
        s.push_str(&format!("\"temperature\": {}, ", fnum(Some(self.temperature))));
        s.push_str(&format!("\"seed\": {}}},\n", self.seed));
        s.push_str("  \"hardware\": {");
        s.push_str(&format!("\"device\": \"{}\", ", esc(&self.hw_device)));
        s.push_str(&format!("\"cpu_threads\": {}, ", self.hw_cpu_threads));
        s.push_str(&format!("\"physical_ram_bytes\": {}, ", self.hw_physical_ram_bytes));
        s.push_str(&format!("\"os\": \"{}\", ", std::env::consts::OS));
        s.push_str(&format!("\"arch\": \"{}\", ", std::env::consts::ARCH));
        s.push_str("\"cuda_toolkit\": null},\n");
        let notes: Vec<String> = self.notes.iter().map(|n| format!("\"{}\"", esc(n))).collect();
        s.push_str(&format!("  \"notes\": [{}]\n", notes.join(", ")));
        s.push_str("}\n");
        s
    }

    /// Human summary (the default output; JSON is opt-in via --json).
    fn print_human(&self) {
        out!(
            "bench {} ({}, {}, backend {})\n",
            self.model_path,
            self.model_format,
            self.phase,
            self.backend
        );
        if let Some(ms) = self.load_ms {
            out!("  load      {:.2} s\n", ms / 1000.0);
        }
        if let (Some(tps), Some(ms)) = (self.prefill_tokens_per_second, self.prefill_ms) {
            out!(
                "  prefill   {:.2} tok/s ({} tokens, {:.2} s)\n",
                tps,
                self.prompt_tokens,
                ms / 1000.0
            );
        }
        if let Some(ttft) = self.ttft_ms {
            out!("  ttft      {:.0} ms\n", ttft);
        }
        if let Some(tps) = self.decode_tokens_per_second {
            out!(
                "  decode    {:.2} tok/s steady-state ({} tokens",
                tps,
                self.decode_tokens
            );
            if let (Some(p50), Some(p95), Some(p99)) = (
                self.inter_token_ms_p50,
                self.inter_token_ms_p95,
                self.inter_token_ms_p99,
            ) {
                out!(", p50 {:.1} / p95 {:.1} / p99 {:.1} ms", p50, p95, p99);
            }
            out!(")\n");
        }
        if let Some(hit) = self.host_cache_byte_hit_rate {
            out!("  cache     {:.1}% host hit rate", 100.0 * hit);
            if let Some(b) = self.ssd_bytes_per_token {
                out!(", {} read/token", human(b as u64));
            }
            out!("\n");
        }
        if let (Some(&hit), Some(&h2d)) = (
            self.device_cache_byte_hit_rate.first(),
            self.h2d_bytes_per_token.first(),
        ) {
            out!(
                "  vram      {:.1}% device byte hit rate, {} H2D/token\n",
                100.0 * hit,
                human(h2d)
            );
        }
        if let Some(rss) = self.peak_rss_bytes {
            out!("  peak rss  {}\n", human(rss));
        }
    }
}

/// Emit the result: human summary always (unless --json), JSON to stdout
/// for a bare `--json` or to the file for `--json FILE` / `--json-out FILE`.
fn emit(o: &Opts, r: &BenchResult) -> i32 {
    if !o.json && o.json_out.is_none() {
        r.print_human();
        return 0;
    }
    let js = r.to_json();
    if let Some(path) = &o.json_out {
        match std::fs::File::create(path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(js.as_bytes()) {
                    eprintln!("write {path}: {e}");
                    return 1;
                }
            }
            Err(e) => {
                eprintln!("create {path}: {e}");
                return 1;
            }
        }
        if !o.quiet {
            eprintln!("oaiy-llm: wrote {path}");
        }
        r.print_human();
        return 0;
    }
    out!("{js}");
    0
}

/* ---- the bench ------------------------------------------------------------ */

/// Resident or streamed (--budget), CPU or CUDA. Timed separately: load, a
/// standalone prefill forward, then generation with per-token inter-arrival
/// times. The backend is synchronized at every timing boundary so async
/// CUDA work is attributed to the right phase.
fn bench_gguf(o: &Opts) -> i32 {
    let phase = if o.budget > 0 { "streamed-gguf" } else { "resident-gguf" };
    let mut r = BenchResult::new(o, phase);

    let lb = match llama_backend(o) {
        Ok(b) => b,
        Err(m) => {
            eprintln!("{m}");
            return 2;
        }
    };
    let backend = lb.backend.clone();
    // Post-load free-VRAM delta is the only device-memory figure the CUDA
    // backend exposes; it is a lower bound on what the model holds.
    let vram0 = backend.vram_status();

    let t_load = Instant::now();
    let mut model = match load_model(o, backend) {
        Ok(m) => m,
        Err(m) => {
            eprintln!("{m}");
            return 1;
        }
    };
    model.backend().synchronize();
    r.load_ms = Some(t_load.elapsed().as_secs_f64() * 1000.0);
    r.backend = model.backend().name().to_string();
    r.hw_device = if o.cuda {
        // ggml-rs-cuda names the backend "cuda:<ordinal>"; it exposes no
        // device-name query, so the ordinal is all there is to report.
        format!("cuda ({}; device-name query not exposed)", model.backend().name())
    } else {
        "cpu".to_string()
    };
    if let (Some((free0, _)), Some((free1, _))) = (vram0, model.backend().vram_status()) {
        r.resident_device_bytes = vec![free0.saturating_sub(free1) as u64];
    }

    if let Some(shared) = model.stream_shared() {
        r.resident_host_bytes = Some(shared.resident_est_bytes());
        r.notes.push(format!(
            "streamed: expert record {} bytes, cache budget {} bytes",
            shared.record_bytes(),
            shared.cache_budget_bytes()
        ));
    } else {
        r.resident_host_bytes = Some(r.model_total_bytes);
        r.notes.push(
            "resident_host_bytes is the model file size; GGUF weights are mmap'd, so the \
             resident set is demand-paged (see peak_rss_bytes)".into(),
        );
    }

    // Attach the VRAM expert cache (no-op unless streaming on CUDA; see
    // maybe_enable_vram_cache for the budget default).
    maybe_enable_vram_cache(o, &lb, &mut model);
    #[cfg(feature = "cuda")]
    let dstats0 = model.device_cache_stats();

    let ids = match model.tokenizer().encode(PROMPT, true) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tokenize: {e}");
            return 1;
        }
    };
    r.prompt_tokens = ids.len() as u64;

    // -- prefill: one standalone forward over the prompt, timed on its own.
    let kv_len = (ids.len() + o.max_tokens as usize + 8).max(2048);
    let stats0 = model.expert_cache_stats();
    let t = Instant::now();
    {
        let mut kv = model.new_kv_cache(kv_len);
        r.kv_bytes = vec![kv.size_bytes() as u64];
        model.backend().synchronize();
        let logits = model.forward(&ids, &mut kv);
        model.backend().synchronize();
        drop(logits);
    }
    r.prefill_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
    r.prefill_tokens_per_second =
        Some(ids.len() as f64 / (r.prefill_ms.unwrap() / 1000.0).max(1e-9));

    // -- decode: the generate iterator (its creation re-prefills; that cost
    // is part of TTFT, not of the steady-state inter-token samples).
    let t_gen = Instant::now();
    let mut iter = model.generate(&ids, sample_params(o), o.max_tokens as usize);
    model.backend().synchronize();
    let setup_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    let mut it_ms: Vec<f64> = Vec::new();
    let mut n = 0u64;
    let mut first_ms = None;
    loop {
        let t = Instant::now();
        let next = iter.next();
        model.backend().synchronize();
        let el = t.elapsed().as_secs_f64() * 1000.0;
        if next.is_none() {
            break;
        }
        if let Some(e) = model.expert_stream_error() {
            eprintln!("\nexpert stream: {e}");
            return 1;
        }
        if n == 0 {
            first_ms = Some(el);
        } else {
            it_ms.push(el);
        }
        n += 1;
    }
    r.decode_ms = Some(it_ms.iter().sum());
    r.decode_tokens = n;
    if n > 0 {
        r.ttft_ms = Some(setup_ms + first_ms.unwrap_or(0.0));
    }
    if it_ms.is_empty() {
        r.notes.push(
            "decode produced <= 1 token: no steady-state inter-token sample".into(),
        );
    } else {
        let total: f64 = it_ms.iter().sum();
        r.decode_tokens_per_second = Some(it_ms.len() as f64 / (total / 1000.0).max(1e-9));
        it_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        r.inter_token_ms_p50 = percentile(&it_ms, 50.0);
        r.inter_token_ms_p95 = percentile(&it_ms, 95.0);
        r.inter_token_ms_p99 = percentile(&it_ms, 99.0);
    }

    // -- streaming counters over the whole run (prefill + decode; records
    // are fixed-size, so entry rates and byte rates coincide — the byte
    // counters are used where the cache reports them).
    if let (Some(before), Some(after)) = (stats0, model.expert_cache_stats()) {
        let tokens = (r.prompt_tokens + r.decode_tokens).max(1);
        let hits = after.hits - before.hits;
        let misses = after.misses - before.misses;
        let bytes_hit = after.bytes_hit - before.bytes_hit;
        let bytes_read = after.bytes_read - before.bytes_read;
        let rec = model.stream_shared().map(|s| s.record_bytes()).unwrap_or(0) as u64;
        r.host_cache_byte_hit_rate =
            Some(bytes_hit as f64 / (bytes_hit + bytes_read).max(1) as f64);
        r.ssd_bytes_per_token = Some(bytes_read as f64 / tokens as f64);
        r.active_expert_bytes_per_token = Some((hits + misses) * rec / tokens);
        r.notes.push(
            "cache counters cover BOTH prefill passes (the standalone phase and the \
             generate iterator's own) plus decode, so per-token figures run ~2x a single \
             pass; active_expert_bytes_per_token = record accesses x record size / tokens \
             (post-eviction re-fetches count again); the standalone prefill also warms \
             the cache before decode is measured".into(),
        );
    } else {
        r.notes.push(
            "not MoE-streaming: no expert cache, so cache/ssd/active-expert fields are null"
                .into(),
        );
    }

    // Device-cache counters over the same span (prefill passes +
    // decode, like the host counters above). byte_hit_rate = bytes served
    // from VRAM / (VRAM-served + H2D-uploaded); h2d_bytes_per_token is the
    // residual PCIe traffic the cache did NOT absorb.
    #[cfg(feature = "cuda")]
    if let (Some(before), Some(after)) = (dstats0, model.device_cache_stats()) {
        let tokens = (r.prompt_tokens + r.decode_tokens).max(1);
        let bytes_hit = after.bytes_hit - before.bytes_hit;
        let h2d = after.h2d_bytes - before.h2d_bytes;
        r.device_cache_byte_hit_rate =
            vec![bytes_hit as f64 / (bytes_hit + h2d).max(1) as f64];
        r.h2d_bytes_per_token = vec![h2d / tokens];
    }

    r.peak_rss_bytes = peak_rss_bytes();
    r.note_unmeasured();
    emit(o, &r)
}

/* ---- route trace (oaiy-llm run --trace-out) ---------------------------------- */

/// Write the recorded MoE routing calls as JSONL: one header line, then one
/// `{"token", "layer", "experts"}` line per (token, MoE layer). `calls` is
/// one entry per `moe_forward` invocation in execution order, each carrying
/// the top-k expert ids per sequence position.
///
/// The llama-rs routing decision is made inside `expert_stream.rs` (or the
/// resident MoE loop) and the routed ids are not exposed on any public type;
/// the trace is instead re-derived in `moe_forward_with_logits` (VENDORED-
/// LOCAL hook) from the same router logits with the same `top_k_softmax`, so
/// the recorded ids are identical by construction. The hook does not know
/// the layer index, so it is recovered here from call order: the forward
/// pass visits MoE blocks in layer order, hence `layer = call % n_layers`.
/// This is exact for the all-MoE models the streaming path targets; for an
/// arch with interleaved dense layers the layer column would be the MoE-
/// block ordinal instead (none of the models the streaming path accepts
/// mixes them).
pub fn write_route_trace(
    path: &str,
    model: &str,
    calls: &[Vec<Vec<u32>>],
    n_layers: usize,
    prompt_tokens: usize,
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "{{\"type\":\"route-trace\",\"version\":1,\"model\":\"{}\",\"moe_layers_per_forward\":{},\"prompt_tokens\":{}}}",
        esc(model),
        n_layers,
        prompt_tokens
    )?;
    if n_layers == 0 {
        return Ok(());
    }
    for (call, per_token) in calls.iter().enumerate() {
        let layer = call % n_layers;
        let fwd = call / n_layers;
        for (t, experts) in per_token.iter().enumerate() {
            // Forward 0 is the single-shot prefill (seq = prompt_tokens);
            // every later forward is one decode step (seq = 1).
            let token = if fwd == 0 { t } else { prompt_tokens + fwd - 1 + t };
            let ids: Vec<String> = experts.iter().map(u32::to_string).collect();
            writeln!(
                f,
                "{{\"token\":{},\"layer\":{},\"experts\":[{}]}}",
                token,
                layer,
                ids.join(",")
            )?;
        }
    }
    Ok(())
}

/* ---- entry --------------------------------------------------------------- */

/// `oaiy-llm bench MODEL.gguf [-n N] [--budget N] [--cuda] [--json [FILE]]`
///
/// `--json` alone prints the schema JSON to stdout; `--json FILE` (rewritten
/// to `--json-out FILE` by cmd_bench's pre-scan) writes it to a file and
/// keeps the human summary on stdout.
pub fn run(o: &Opts) -> i32 {
    bench_gguf(o)
}

/* ---- tests ---------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// Schema snapshot: every roadmap-required field is present in the JSON,
    /// and unmeasured counters serialize as explicit null with a note.
    #[test]
    fn schema_contains_all_required_fields() {
        let o = Opts {
            pos: vec!["/nonexistent/model.gguf".to_string()],
            ..Opts::default()
        };
        let mut r = BenchResult::new(&o, "resident-gguf");
        r.note_unmeasured();
        let js = r.to_json();
        for f in [
            "\"revision\"",
            "\"model_hash\"",
            "\"model_total_bytes\"",
            "\"active_expert_bytes_per_token\"",
            "\"resident_host_bytes\"",
            "\"resident_device_bytes\"",
            "\"kv_bytes\"",
            "\"host_cache_byte_hit_rate\"",
            "\"device_cache_byte_hit_rate\"",
            "\"ssd_bytes_per_token\"",
            "\"host_copy_bytes_per_token\"",
            "\"h2d_bytes_per_token\"",
            "\"kernel_launches_per_token\"",
            "\"host_allocations_per_token\"",
            "\"device_allocations_per_token\"",
            "\"prefill_tokens_per_second\"",
            "\"decode_tokens_per_second\"",
            "\"ttft_ms\"",
            "\"inter_token_ms_p50\"",
            "\"inter_token_ms_p95\"",
            "\"inter_token_ms_p99\"",
            "\"peak_rss_bytes\"",
            "\"peak_vram_bytes\"",
            "\"hardware\"",
            "\"notes\"",
        ] {
            assert!(js.contains(f), "schema field {f} missing:\n{js}");
        }
        // Every null field has a note; spot-check the unmeasured set.
        assert!(js.contains("\"model_hash\": null"), "{js}");
        assert!(js.contains("\"h2d_bytes_per_token\": []"), "{js}");
        assert!(js.contains("no VRAM expert cache"), "{js}");
    }

    #[test]
    fn percentiles_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&v, 50.0), Some(50.0));
        assert_eq!(percentile(&v, 95.0), Some(95.0));
        assert_eq!(percentile(&v, 99.0), Some(99.0));
        assert_eq!(percentile(&[], 50.0), None);
    }

    #[test]
    fn json_escapes() {
        assert_eq!(esc("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
