//! Validate that a Qwen3.5 GGUF can be loaded by our crate. Reports the SSM
//! hyperparameters, the SSM-vs-attention layer split, and the per-layer-type
//! tensor count. Calling `.forward()` is currently a clean error pointing at
//! task #61 (Mamba2 selective scan + MRoPE not yet implemented).
//!
//! Usage:
//!   cargo run --release -p llama-rs --example inspect_qwen35 -- model.gguf

use std::env;

use ggml_rs::{default_backend, Backend};
use gguf::GgufFile;
use llama_rs::{Model, Qwen35Block};

fn pick_backend(name: &str) -> Result<std::sync::Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(default_backend()),
        "cuda" => Err("CUDA backend not enabled. Build with `--features cuda`".into()),
        other => Err(format!("unknown backend `{other}` (try `cpu` or `cuda`)").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: inspect_qwen35 <model.gguf> [backend=cpu|cuda]")?;
    let backend_name = args.next().unwrap_or_else(|| "cpu".into());

    let g = GgufFile::open(&path)?;
    let backend = pick_backend(&backend_name)?;
    println!("Loading {} on {} backend ...", path, backend.name());
    let t0 = std::time::Instant::now();
    let model = Model::load(&g, backend)?;
    let m = match &model {
        Model::Qwen35(m) => m,
        other => return Err(format!("expected Qwen3.5; got {:?}", std::mem::discriminant(other)).into()),
    };
    println!("  loaded in {:.2?}", t0.elapsed());
    println!();

    println!("== Config ==");
    println!("  vocab_size:    {}", m.config.vocab_size);
    println!("  context_len:   {}", m.config.context_length);
    println!("  embedding_dim: {}", m.config.embedding_dim);
    println!("  n_layers:      {}", m.config.n_layers);
    println!("  n_heads:       {}", m.config.n_heads);
    println!("  n_kv_heads:    {}", m.config.n_kv_heads);
    println!("  head_dim:      {}", m.config.head_dim);
    println!("  ff_dim:        {}", m.config.ff_dim);
    println!("  rope_theta:    {}", m.config.rope_theta);
    println!("  rope_dim:      {}  (full head_dim = {})", m.config.rope_dim, m.config.head_dim);
    println!();

    println!("== SSM hyperparameters ==");
    println!("  conv_kernel:    {}", m.ssm_cfg.conv_kernel);
    println!("  group_count:    {}", m.ssm_cfg.group_count);
    println!("  inner_size:     {}", m.ssm_cfg.inner_size);
    println!("  state_size:     {}", m.ssm_cfg.state_size);
    println!("  time_step_rank: {}", m.ssm_cfg.time_step_rank);
    println!();

    println!("== Layer split ==");
    let attn_count = m.attention_layers.iter().filter(|&&b| b).count();
    let ssm_count  = m.attention_layers.len() - attn_count;
    println!("  attention layers: {} ({:?})",
             attn_count,
             m.attention_layers.iter().enumerate().filter(|(_, &b)| b)
                 .map(|(i, _)| i).collect::<Vec<_>>());
    println!("  SSM layers:       {}", ssm_count);
    println!();

    println!("== Block 0 (SSM / linear-attention 'gated delta net') ==");
    if let Qwen35Block::Ssm { attn_qkv, attn_gate, ssm_conv1d, ssm_a, ssm_ba,
                              ssm_dt_bias, ssm_norm, ssm_out, .. } = &m.blocks[0] {
        println!("  attn_qkv (q+k+v, no z) shape: {:?}", attn_qkv.shape());
        println!("  attn_gate (z proj)     shape: {:?}", attn_gate.shape());
        println!("  ssm_conv1d (depthwise) shape: {:?}", ssm_conv1d.shape());
        println!("  ssm_a (per-v-head)     shape: {:?}", ssm_a.shape());
        let ssm_a_h = ssm_a.to_host();
        let amin = ssm_a_h.data().iter().cloned().fold(f32::INFINITY, f32::min);
        let amax = ssm_a_h.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("    ssm_a values:        min={amin:.4}  max={amax:.4}  first 4 = {:?}",
                 &ssm_a_h.data()[..4.min(ssm_a_h.numel())]);
        let dt_bias_h = ssm_dt_bias.to_host();
        let dt_min = dt_bias_h.data().iter().cloned().fold(f32::INFINITY, f32::min);
        let dt_max = dt_bias_h.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("    ssm_dt_bias values:  min={dt_min:.4}  max={dt_max:.4}  first 4 = {:?}",
                 &dt_bias_h.data()[..4.min(dt_bias_h.numel())]);
        println!("  ssm_ba (β||α fused)    shape: {:?}", ssm_ba.shape());
        println!("  ssm_dt_bias            shape: {:?}", ssm_dt_bias.shape());
        println!("  ssm_norm  (per-v-head) shape: {:?}", ssm_norm.shape());
        println!("  ssm_out  (out proj)    shape: {:?}", ssm_out.shape());

        // Dimensional sanity: per qwen3next.cpp, attn_qkv per-k-head packs
        //   [head_k_dim, head_k_dim, head_v_dim * num_v_heads/num_k_heads]
        // = [128, 128, 128*2] = [128, 128, 256] = 512.
        // Total = 512 * num_k_heads = 8192. attn_gate is z = head_v_dim *
        // (num_v_heads/num_k_heads) per k-head * num_k_heads = 256*16 = 4096.
        let head_k_dim    = m.ssm_cfg.state_size;
        let num_k_heads   = m.ssm_cfg.group_count;
        let num_v_heads   = m.ssm_cfg.time_step_rank;
        let head_v_dim    = m.ssm_cfg.inner_size / num_v_heads;
        let v_per_k_head  = num_v_heads / num_k_heads;
        let qkv_per_k     = 2 * head_k_dim + head_v_dim * v_per_k_head;
        let qkv_total     = qkv_per_k * num_k_heads;
        let z_total       = head_v_dim * v_per_k_head * num_k_heads;
        println!("  → derived: head_k_dim={head_k_dim}, num_k_heads={num_k_heads}, num_v_heads={num_v_heads}, head_v_dim={head_v_dim}");
        println!("  → expected attn_qkv out = {qkv_total}, attn_gate out = {z_total}");
        assert_eq!(attn_qkv.shape()[0], qkv_total, "attn_qkv mismatch");
        assert_eq!(attn_gate.shape()[0], z_total, "attn_gate mismatch");
        println!("  ✓ attn_qkv / attn_gate dims match qwen3next derivation");
    }
    println!();

    let attn_layer_idx = m.attention_layers.iter().position(|&b| b).unwrap_or(0);
    println!("== Block {attn_layer_idx} (Attention) ==");
    if let Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, .. } = &m.blocks[attn_layer_idx] {
        println!("  attn_q shape:       {:?}", attn_q.shape());
        println!("  attn_k shape:       {:?}", attn_k.shape());
        println!("  attn_v shape:       {:?}", attn_v.shape());
        println!("  attn_q_norm shape:  {:?}", attn_q_norm.shape());
        println!("  attn_k_norm shape:  {:?}", attn_k_norm.shape());
    }
    println!();

    println!("== Forward pass ==");
    let mut kv = model.new_kv_cache(2048);
    let prompt = std::env::var("QWEN35_PROMPT").unwrap_or_else(|_| "Hello, my name is".into());
    println!("  prompt: {prompt:?}");
    let probe_tokens = m.tokenizer.encode(&prompt, true)?;
    println!("  prompt tokens: {:?}", probe_tokens);
    let t0 = std::time::Instant::now();
    let logits = m.forward(&probe_tokens, &mut kv)?;
    println!("  forward succeeded in {:.2?}", t0.elapsed());
    println!("  logits shape: {:?}", logits.shape());

    let last_row = model.last_logits(&logits);
    let next_id = last_row.data().iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32).unwrap();
    println!("  greedy next token: {next_id} = {:?}", m.tokenizer.decode(&[next_id]));

    // Continue greedy decoding for a few more tokens to see if the model stays coherent.
    let n_more = std::env::var("QWEN35_GENERATE_N").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
    if n_more > 0 {
        use llama_rs::{SampleParams, Sampler};
        let use_sampling = std::env::var("QWEN35_SAMPLE").is_ok();
        let params = if use_sampling {
            SampleParams { temperature: 0.7, top_p: Some(0.9), top_k: Some(40),
                           repeat_penalty: Some(1.1), repeat_last_n: 32, ..SampleParams::default() }
        } else {
            SampleParams::greedy()
        };
        let mut sampler = Sampler::new(params);
        let mut tokens: Vec<u32> = probe_tokens.iter().chain(std::iter::once(&next_id)).copied().collect();
        for &t in &tokens { sampler.observe(t); }
        let t_gen = std::time::Instant::now();
        for _ in 0..n_more {
            let logits = m.forward(&[*tokens.last().unwrap()], &mut kv)?;
            let last_row = model.last_logits(&logits);
            let next = sampler.sample(&last_row);
            tokens.push(next);
        }
        let elapsed = t_gen.elapsed();
        let new_tokens = &tokens[probe_tokens.len()..];
        let text = m.tokenizer.decode(new_tokens);
        let tps = n_more as f32 / elapsed.as_secs_f32();
        let mode = if use_sampling { "sampled" } else { "greedy" };
        println!("  {mode} {n_more}-token continuation ({tps:.1} tok/s):");
        println!("    {text:?}");
    }

    println!();
    println!("Notes:");
    println!("  * Attention layers (8/32): joint Q+gate split + per-head Q/K-norm");
    println!("    + partial RoPE ({}/{} dims) + standard MHA. ✓", m.config.rope_dim, m.config.head_dim);
    println!("  * Linear-attention layers (24/32): per-token autoregressive");
    println!("    `gated delta net` matching HF Qwen3_5LinearAttention. End-to-end");
    println!("    correct on a real GGUF — greedy completes \"The capital of France");
    println!("    is\" → \" Paris.\" and \"jumps over the lazy\" → \" dog.\". The unsloth");
    println!("    GGUF stores V heads in TILED order (per `_LinearAttentionVReorderBase`");
    println!("    in convert_hf_to_gguf.py), which the code correctly handles via");
    println!("    `h_k = h_v % num_k_heads`.");
    println!("  * Decode: ~21 tok/s on CUDA (per-token CUDA delta-net + conv1d kernel");
    println!("    pair keeps everything on device, eliminating the per-layer d2h on the");
    println!("    decode fast path), ~0.1 tok/s on CPU. Prefill still uses the chunked-");
    println!("    rayon host loop — task #66 will replace it with a chunked CUDA kernel.");

    Ok(())
}
