//! Greedy generation on one or more GPUs (hybrid CPU/GPU decode), streaming
//! text, with per-token timing and cache statistics.
//!
//!   echo Explain how rainbows form. > prompt.txt
//!   cargo run -p dsv41-cuda --release --example generate -- prompt.txt [max_new_tokens] [devices]
//!
//! The prompt file holds a user message (encoded with the chat format, chat
//! mode) or token ids, comma or space separated.
//!
//! `devices` is a comma-separated list of CUDA ordinals (default "1"); "1,0"
//! splits the layers across both cards, each caching its own layers' experts.
//!
//! Environment: DSV41_MODEL (default E:\deepseek\model), DSV41_GOLDEN_DIR
//! (for engram_meta.safetensors, default E:\deepseek\golden), DSV41_RAM_GB
//! (host expert cache, default 140), DSV41_USAGE (expert usage profile,
//! default E:\deepseek\expert_usage.txt: read at start to warm the RAM and
//! VRAM tiers when it exists, rewritten at exit; "off" disables both),
//! DSV41_CPU_THREADS (hybrid decode: VRAM-missing experts on this many CPU
//! threads, default 24; "off" uploads every miss instead), DSV41_RUNS (answer
//! the prompt this many times in one process, default 1), DSV41_PROFILE
//! (per-phase timing of the last run's second half; adds syncs).

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use dsv41::detok::Detokenizer;
use dsv41::model::argmax;
use dsv41_cuda::{GpuModel, GpuOptions};

fn main() -> nrob::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ids_file = args.get(1).expect("usage: generate PROMPT_IDS_FILE [max_new_tokens] [devices]");
    let max_new: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(128);
    let devices: Vec<usize> = args
        .get(3)
        .map_or("1", String::as_str)
        .split(',')
        .map(|v| v.trim().parse().map_err(|_| nrob::Error::Arg(format!("bad device {v:?}"))))
        .collect::<nrob::Result<_>>()?;
    let model_dir = std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| r"E:\deepseek\model".into());
    let golden = std::env::var_os("DSV41_GOLDEN_DIR").map(PathBuf::from).unwrap_or_else(|| r"E:\deepseek\golden".into());

    // token ids (comma or space separated), or else the text of a user
    // message, encoded here with the chat format (chat mode) and tokenizer
    let text = std::fs::read_to_string(ids_file)?;
    let ids: Result<Vec<u32>, _> = text.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).map(str::parse).collect();
    let prompt: Vec<u32> = match ids {
        Ok(ids) if !ids.is_empty() => ids,
        _ => {
            let user = nrob::json::Json::obj([("role", nrob::json::Json::str("user")), ("content", nrob::json::Json::str(text.trim()))]);
            let enc = dsv41::chat::encode(&[user], &dsv41::chat::Options::default())?;
            dsv41::tokenizer::Tokenizer::load(&model_dir)?.encode(&enc.prompt)
        }
    };
    let detok = Detokenizer::load(&model_dir)?;

    let ram_gb: usize = std::env::var("DSV41_RAM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(140);
    let usage = match std::env::var_os("DSV41_USAGE") {
        Some(v) if v == "off" => None,
        Some(v) => Some(PathBuf::from(v)),
        None => Some(PathBuf::from(r"E:\deepseek\expert_usage.txt")),
    };

    let t = Instant::now();
    let cpu_expert_threads = match std::env::var("DSV41_CPU_THREADS").as_deref() {
        Ok("off") => None,
        Ok(v) => Some(v.parse().map_err(|_| nrob::Error::Arg(format!("bad DSV41_CPU_THREADS {v:?}")))?),
        Err(_) => Some(24),
    };
    let opts = GpuOptions { devices: devices.clone(), max_seq: 4096, expert_cache_bytes: ram_gb << 30, direct_io: true, vram_expert_bytes: None, vram_headroom_bytes: 1 << 30, cpu_expert_threads, vision: false };
    let mut model = GpuModel::load(&model_dir, &golden.join("engram_meta.safetensors"), &opts)?;
    let slots: Vec<usize> = model.device_caches().map(|c| c.slots()).collect();
    eprintln!("[loaded in {:.1}s on cuda:{devices:?}, VRAM expert slots {slots:?}, RAM expert slots {}]", t.elapsed().as_secs_f64(), model.expert_cache().n_slots());
    if let Some(path) = usage.as_ref().filter(|p| p.exists()) {
        let t = Instant::now();
        let (vram, queued) = model.warm(path, 4)?;
        eprintln!(
            "[warmed {vram} experts into VRAM from {} in {:.1}s; {queued} more loading into RAM in the background]",
            path.display(),
            t.elapsed().as_secs_f64()
        );
    }
    let profile = std::env::var_os("DSV41_PROFILE").is_some();
    // DSV41_RUNS: answer the prompt this many times in one process, as a
    // long-lived server would, to see the warm tiers pay off
    let runs: usize = std::env::var("DSV41_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let eos = model.cfg.eos_token_id;
    for run in 0..runs {
        if runs > 1 {
            eprintln!("\n[run {} of {runs}]", run + 1);
        }
        let host0 = model.expert_cache().stats();
        let bg0 = model.warming().map_or(0, |(done, _)| done as u64);
        let dev0: Vec<_> = model.device_caches().map(|c| c.stats).collect();

        let t = Instant::now();
        let mut next = argmax(&model.forward(&prompt, 0)?);
        eprintln!("[prefill {} tokens: {:.1}s]", prompt.len(), t.elapsed().as_secs_f64());
        let ew0 = model.engram_wait_s();

        let (mut pending, mut times) = (Vec::new(), Vec::new());
        let mut out = std::io::stdout();
        for step in 0..max_new {
            pending.extend_from_slice(detok.bytes(next));
            // print the longest valid UTF-8 prefix; keep a split sequence for later
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            out.write_all(&pending[..valid])?;
            out.flush()?;
            pending.drain(..valid);
            if next == eos {
                break;
            }
            // profile decode only, from the second half of the last run on
            if profile && run + 1 == runs && step == max_new / 2 {
                model.enable_profile();
            }
            let t = Instant::now();
            next = argmax(&model.forward(&[next], prompt.len() + step)?);
            times.push(t.elapsed().as_secs_f64());
        }
        writeln!(out)?;
        if let Some(p) = model.profile() {
            let n = p.forwards.max(1) as f64;
            let total = p.hc + p.attn + p.route + p.fetch + p.experts + p.shared + p.cpu + p.engram + p.head;
            eprintln!(
                "\n[profile, ms/token over {} tokens: total {:.0} | expert fetch {:.0} | expert compute {:.0} | cpu experts {:.0} ({:.1}/token) | attention {:.0} | hc {:.0} | route {:.0} | shared {:.0} | engram {:.0} | head {:.0}]",
                p.forwards,
                1e3 * total / n,
                1e3 * p.fetch / n,
                1e3 * p.experts / n,
                1e3 * p.cpu / n,
                p.cpu_uses as f64 / n,
                1e3 * p.attn / n,
                1e3 * p.hc / n,
                1e3 * p.route / n,
                1e3 * p.shared / n,
                1e3 * p.engram / n,
                1e3 * p.head / n
            );
        }
        let half = &times[times.len() / 2..];
        let (hits, misses, declined) = model
            .device_caches()
            .zip(&dev0)
            .fold((0, 0, 0), |(h, m, d), (c, c0)| (h + c.stats.hits - c0.hits, m + c.stats.misses - c0.misses, d + c.stats.declined - c0.declined));
        let h = model.expert_cache().stats();
        // background warm-up reads count as host-cache misses too; leave them out
        let bg = model.warming().map_or(0, |(done, _)| done as u64) - bg0;
        let (hh, hm) = (h.hits - host0.hits, (h.misses - host0.misses).saturating_sub(bg));
        eprintln!("[waited for Engram rows: {:.1} ms a token]", 1e3 * (model.engram_wait_s() - ew0) / times.len().max(1) as f64);
        eprintln!(
            "\n[{} tokens: {:.2} tok/s overall; first half {:.2}s/token, second half {:.2}s/token ({:.2} tok/s)]",
            times.len(),
            times.len() as f64 / times.iter().sum::<f64>().max(1e-9),
            times[..times.len() / 2].iter().sum::<f64>() / (times.len() / 2).max(1) as f64,
            half.iter().sum::<f64>() / half.len().max(1) as f64,
            half.len() as f64 / half.iter().sum::<f64>().max(1e-9)
        );
        eprintln!(
            "[VRAM experts: {:.1}% hit ({} hits, {} misses, {} of them on the CPU); host RAM cache: {:.1}% hit ({} SSD reads)]",
            100.0 * hits as f64 / (hits + misses).max(1) as f64,
            hits,
            misses,
            declined,
            100.0 * hh as f64 / (hh + hm).max(1) as f64,
            hm
        );
        if let Some((done, finished)) = model.warming() {
            eprintln!("[background warm-up: {done} experts read{}]", if finished { ", finished" } else { " so far" });
        }
    }
    if let Some(path) = &usage {
        model.save_usage(path)?;
        eprintln!("[expert usage saved to {}]", path.display());
    }
    Ok(())
}
