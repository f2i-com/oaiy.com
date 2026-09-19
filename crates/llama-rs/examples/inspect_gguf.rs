//! Print a summary, metadata, and tensor table of a GGUF file. Useful to
//! sanity-check that a model file is parseable before trying to load it, and
//! to understand at-a-glance what arch / dtype mix it uses.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example inspect_gguf -- path/to/model.gguf
//!   cargo run --release -p llama-rs --example inspect_gguf -- model.gguf --summary
//!
//! By default prints the summary, metadata, and tensor table. Pass `--summary`
//! (or `-s`) for just the summary block.

use std::collections::HashMap;
use std::env;

use ggml_quants::GgmlType;
use gguf::GgufFile;
use llama_rs::Architecture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: inspect_gguf <path> [--summary]")?;
    let summary_only = args.any(|a| a == "--summary" || a == "-s");

    let f = GgufFile::open(&path)?;

    println!("File:    {path}");
    println!("Version: {}", f.version());
    println!("Align:   {} bytes", f.alignment());
    println!("Data @:  0x{:x}", f.tensor_data_start());
    println!();

    print_summary(&f);

    if summary_only { return Ok(()); }

    println!("== Metadata ({} keys) ==", f.metadata().len());
    for (k, v) in f.metadata() {
        println!("  {k:48} {v}");
    }

    println!();
    println!("== Tensors ({}) ==", f.tensors().len());
    for t in f.tensors() {
        println!(
            "  {:<48} {:>10} elem  {:<8} {} bytes",
            t.name,
            t.numel(),
            t.dtype.name(),
            t.nbytes(),
        );
    }
    Ok(())
}

fn print_summary(f: &GgufFile) {
    println!("== Summary ==");

    // Architecture detection.
    match f.architecture() {
        Ok(arch_str) => {
            let arch = Architecture::from_str(arch_str);
            let supported = if arch.supported() { "supported" } else { "NOT supported by this crate" };
            println!("  arch:           {arch_str} ({supported})");
        }
        Err(_) => println!("  arch:           <missing general.architecture key>"),
    }

    // Common config knobs (best-effort — keys depend on arch).
    let md = f.metadata();
    let arch_prefix = f.architecture().ok().map(|s| s.to_string());
    let get_u64 = |k: &str| -> Option<u64> { md.get(k).and_then(|v| v.as_u64()) };
    let get_str = |k: &str| -> Option<&str> { md.get(k).and_then(|v| v.as_str()) };
    let arch_key = |suffix: &str| -> Option<String> {
        arch_prefix.as_ref().map(|p| format!("{p}.{suffix}"))
    };

    if let Some(name) = get_str("general.name") {
        println!("  name:           {name}");
    }
    if let Some(k) = arch_key("vocab_size").and_then(|k| get_u64(&k)) {
        println!("  vocab_size:     {k}");
    } else if let Some(toks) = md.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
        // Fall back to tokens array length.
        println!("  vocab_size:     {} (from tokenizer.ggml.tokens)", toks.len());
    }
    for (label, suffix) in &[
        ("context_len:   ", "context_length"),
        ("embedding_dim: ", "embedding_length"),
        ("n_layers:      ", "block_count"),
        ("n_heads:       ", "attention.head_count"),
        ("n_kv_heads:    ", "attention.head_count_kv"),
        ("ffn_dim:       ", "feed_forward_length"),
    ] {
        if let Some(v) = arch_key(suffix).and_then(|k| get_u64(&k)) {
            println!("  {label} {v}");
        }
    }

    // Parameter count + dtype breakdown.
    let mut by_dtype: HashMap<GgmlType, (u64, u64)> = HashMap::new();  // dtype -> (numel, bytes)
    let mut total_params: u64 = 0;
    let mut total_bytes: u64 = 0;
    for t in f.tensors() {
        let entry = by_dtype.entry(t.dtype).or_insert((0, 0));
        entry.0 += t.numel();
        entry.1 += t.nbytes();
        total_params += t.numel();
        total_bytes += t.nbytes();
    }

    println!("  params:         {} ({})", total_params, human_count(total_params));
    println!("  on-disk bytes:  {} ({:.2} GiB)",
             total_bytes,
             total_bytes as f64 / (1u64 << 30) as f64);

    println!();
    println!("== Dtype breakdown ==");
    let mut dtypes: Vec<_> = by_dtype.iter().collect();
    dtypes.sort_by(|a, b| b.1.1.cmp(&a.1.1));    // descending by bytes
    for (dtype, (numel, bytes)) in dtypes {
        let pct = 100.0 * (*bytes as f64) / (total_bytes.max(1) as f64);
        println!(
            "  {:<8} {:>5.1}%  {:>12} elem  {:>12} bytes",
            dtype.name(), pct, numel, bytes,
        );
    }
    println!();
}

/// Human-friendly parameter count: 596_000_000 -> "596M".
fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
