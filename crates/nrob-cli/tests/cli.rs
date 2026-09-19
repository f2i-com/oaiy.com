//! CLI smoke tests: the `nrob` binary driven end to end on tiny synthetic
//! GGUF models written to a temp directory (no model downloads needed).
//!
//!   - a dense llama (no chat template): info, tokenize / detokenize, run,
//!     bench JSON
//!   - a qwen3moe (with a chat template): run resident and with streamed
//!     experts (`--budget`), which must pick the same greedy tokens
//!   - a file that is not a GGUF is refused with a clear message

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use gguf::value::{Array, Value};
use gguf::{GgmlType, TensorInfo};
use tokenizer::byte_encoder::byte_to_char;

const NROB: &str = env!("CARGO_BIN_EXE_nrob");

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(NROB)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {NROB}: {e}"))
}

fn ok(out: &std::process::Output, what: &str) -> (String, String) {
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{what} failed:\n{stderr}\n{stdout}");
    (stdout, stderr)
}

/// A temp directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("nrob-cli-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn file(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const VOCAB: usize = 256; // one token per byte: any ASCII prompt encodes
const EMBED: usize = 16;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = EMBED / HEADS;
const FF: usize = 16;
const EXPERTS: usize = 4;

/// Metadata + F32 tensors under construction; weights come from a
/// deterministic SplitMix64 so every run writes the same file.
struct Builder {
    md: BTreeMap<String, Value>,
    tensors: Vec<(TensorInfo, Vec<u8>)>,
    state: u64,
}

impl Builder {
    fn new(arch: &str) -> Self {
        let mut md = BTreeMap::new();
        md.insert("general.architecture".into(), Value::String(arch.into()));
        md.insert("general.alignment".into(), Value::U32(32));
        let u = |v: usize| Value::U32(v as u32);
        md.insert(format!("{arch}.context_length"), u(256));
        md.insert(format!("{arch}.embedding_length"), u(EMBED));
        md.insert(format!("{arch}.block_count"), u(LAYERS));
        md.insert(format!("{arch}.feed_forward_length"), u(FF));
        md.insert(format!("{arch}.attention.head_count"), u(HEADS));
        md.insert(format!("{arch}.attention.head_count_kv"), u(KV_HEADS));
        md.insert(format!("{arch}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
        md.insert(format!("{arch}.rope.freq_base"), Value::F32(10000.0));
        let toks: Vec<String> = (0..VOCAB).map(|b| byte_to_char(b as u8).to_string()).collect();
        md.insert("tokenizer.ggml.model".into(), Value::String("gpt2".into()));
        md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(toks)));
        md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(vec![])));
        Builder { md, tensors: Vec::new(), state: 0xC0FFEE }
    }

    fn rand(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.5
    }

    /// An F32 tensor; `shape` is GGUF order (fastest-varying first).
    fn add(&mut self, name: &str, shape: &[usize]) {
        let n: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let v = self.rand();
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let info = TensorInfo {
            name: name.into(),
            shape: shape.iter().map(|&d| d as u64).collect(),
            dtype: GgmlType::F32,
            offset: 0, // write_to_vec lays the offsets out
        };
        self.tensors.push((info, bytes));
    }

    fn attention(&mut self, i: usize) {
        let (q, k) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
        self.add(&format!("blk.{i}.attn_norm.weight"), &[EMBED]);
        self.add(&format!("blk.{i}.attn_q.weight"), &[EMBED, q]);
        self.add(&format!("blk.{i}.attn_k.weight"), &[EMBED, k]);
        self.add(&format!("blk.{i}.attn_v.weight"), &[EMBED, k]);
        self.add(&format!("blk.{i}.attn_output.weight"), &[q, EMBED]);
        self.add(&format!("blk.{i}.ffn_norm.weight"), &[EMBED]);
    }

    fn write(mut self, path: &str) {
        self.add("token_embd.weight", &[EMBED, VOCAB]);
        self.add("output_norm.weight", &[EMBED]);
        self.add("output.weight", &[EMBED, VOCAB]);
        let raw = gguf::reader::write_to_vec(&self.md, &self.tensors, 32).expect("write gguf");
        std::fs::write(path, raw).unwrap();
    }
}

fn write_llama(path: &str) {
    let mut b = Builder::new("llama");
    for i in 0..LAYERS {
        b.attention(i);
        b.add(&format!("blk.{i}.ffn_gate.weight"), &[EMBED, FF]);
        b.add(&format!("blk.{i}.ffn_up.weight"), &[EMBED, FF]);
        b.add(&format!("blk.{i}.ffn_down.weight"), &[FF, EMBED]);
    }
    b.write(path);
}

fn write_qwen3moe(path: &str) {
    let mut b = Builder::new("qwen3moe");
    b.md.insert("qwen3moe.expert_count".into(), Value::U32(EXPERTS as u32));
    b.md.insert("qwen3moe.expert_used_count".into(), Value::U32(2));
    b.md.insert("qwen3moe.expert_feed_forward_length".into(), Value::U32(FF as u32));
    b.md.insert(
        "tokenizer.chat_template".into(),
        Value::String("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
    );
    for i in 0..LAYERS {
        b.attention(i);
        b.add(&format!("blk.{i}.attn_q_norm.weight"), &[HEAD_DIM]);
        b.add(&format!("blk.{i}.attn_k_norm.weight"), &[HEAD_DIM]);
        b.add(&format!("blk.{i}.ffn_gate_inp.weight"), &[EMBED, EXPERTS]);
        b.add(&format!("blk.{i}.ffn_gate_exps.weight"), &[EMBED, FF, EXPERTS]);
        b.add(&format!("blk.{i}.ffn_up_exps.weight"), &[EMBED, FF, EXPERTS]);
        b.add(&format!("blk.{i}.ffn_down_exps.weight"), &[FF, EMBED, EXPERTS]);
    }
    b.write(path);
}

/// The `ids: [...]` line `run --stats` prints.
fn ids_line(stderr: &str) -> String {
    stderr
        .lines()
        .find(|l| l.starts_with("ids: "))
        .unwrap_or_else(|| panic!("no ids line in:\n{stderr}"))
        .to_string()
}

#[test]
fn cli_dense_gguf_info_tokens_run_bench() {
    let tmp = TempDir::new("dense");
    let m = tmp.file("tiny-llama.gguf");
    write_llama(&m);

    // info: header facts, no weights loaded
    let (stdout, _) = ok(&run_cli(&["info", &m, "--json"]), "info");
    assert!(stdout.contains("\"arch\":\"llama\""), "{stdout}");
    assert!(stdout.contains(&format!("\"layers\":{LAYERS}")), "{stdout}");
    assert!(stdout.contains("\"chat_template\":false"), "{stdout}");
    let (stdout, _) = ok(&run_cli(&["info", &m]), "info (human)");
    assert!(stdout.contains("base model"), "{stdout}");

    // tokenize / detokenize: one token per byte in this vocabulary
    let (stdout, _) = ok(&run_cli(&["tokenize", &m, "hi"]), "tokenize");
    assert_eq!(stdout.trim(), "104 105");
    let (stdout, _) = ok(&run_cli(&["detokenize", &m, "104 105"]), "detokenize");
    assert_eq!(stdout.trim(), "hi");
    assert!(!run_cli(&["detokenize", &m, "999"]).status.success(), "out-of-vocab id accepted");

    // run: greedy, exactly -n tokens (no EOS in this vocabulary), stats line
    let (_, stderr) = ok(&run_cli(&["run", &m, "hi", "-n", "5", "--stats"]), "run");
    assert!(stderr.contains("[5 tokens,"), "{stderr}");
    let first = ids_line(&stderr);
    // deterministic across runs
    let (_, stderr) = ok(&run_cli(&["run", &m, "hi", "-n", "5", "--stats", "-q"]), "run again");
    assert_eq!(ids_line(&stderr), first);

    // bench: every schema field, JSON on stdout
    let required = [
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
        "\"notes\"",
    ];
    let (stdout, _) = ok(&run_cli(&["bench", &m, "-n", "4", "--json"]), "bench");
    for f in required {
        assert!(stdout.contains(f), "bench schema field {f} missing:\n{stdout}");
    }
    assert!(stdout.contains("\"phase\": \"resident-gguf\""), "{stdout}");
    assert!(!stdout.contains("\"decode_tokens_per_second\": null"), "{stdout}");

    // `--json FILE`: schema to the file, human summary on stdout
    let jf = tmp.file("bench.json");
    let (stdout, _) = ok(&run_cli(&["bench", &m, "-n", "4", "--json", &jf]), "bench --json FILE");
    let js = std::fs::read_to_string(&jf).expect("bench wrote no JSON file");
    assert!(js.contains("\"schema\": \"nrob-bench/1\""), "{js}");
    assert!(stdout.contains("decode"), "human summary missing: {stdout}");
}

#[test]
fn cli_moe_streamed_matches_resident() {
    let tmp = TempDir::new("moe");
    let m = tmp.file("tiny-qwen3moe.gguf");
    write_qwen3moe(&m);

    let (stdout, _) = ok(&run_cli(&["info", &m, "--json"]), "info");
    assert!(stdout.contains(&format!("\"experts\":{EXPERTS}")), "{stdout}");
    assert!(stdout.contains("\"chat_template\":true"), "{stdout}");

    let (_, resident) = ok(&run_cli(&["run", &m, "hello", "-n", "6", "--stats"]), "run resident");
    let (_, streamed) = ok(
        &run_cli(&["run", &m, "hello", "-n", "6", "--stats", "--budget", "1M"]),
        "run streamed",
    );
    assert!(streamed.contains("streaming experts"), "{streamed}");
    assert!(streamed.contains(" hit / ") && streamed.contains(" miss"), "{streamed}");
    // same greedy tokens whether experts are resident or read from the file
    assert_eq!(ids_line(&streamed), ids_line(&resident));

    // raw continuation is a different prompt, so it runs too
    ok(&run_cli(&["run", &m, "hello", "-n", "2", "--raw"]), "run --raw");

    // the streamed bench reports cache traffic
    let (stdout, _) = ok(&run_cli(&["bench", &m, "-n", "4", "--budget", "1M", "--json"]), "bench");
    assert!(stdout.contains("\"phase\": \"streamed-gguf\""), "{stdout}");
    assert!(!stdout.contains("\"host_cache_byte_hit_rate\": null"), "{stdout}");
}

#[test]
fn cli_refuses_non_gguf() {
    let tmp = TempDir::new("notgguf");
    let f = tmp.file("model.safetensors");
    std::fs::write(&f, b"{\"__metadata__\":{}}").unwrap();
    let out = run_cli(&["run", &f, "hi"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not a GGUF file"), "{stderr}");
    assert!(!run_cli(&["info", Path::new(&tmp.file("missing.gguf")).to_str().unwrap()]).status.success());
    // no command, and an unknown one, are usage errors
    assert_eq!(run_cli(&[]).status.code(), Some(2));
    assert_eq!(run_cli(&["convert"]).status.code(), Some(2));
}
