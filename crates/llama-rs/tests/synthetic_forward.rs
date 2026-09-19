//! Build a tiny Llama-arch GGUF in memory with random-ish weights, then run a
//! forward pass end-to-end. We are NOT validating output correctness against
//! a reference implementation — that requires a real model file. We ARE
//! validating that:
//!
//!   * Config parsing reads the right keys.
//!   * Loader resolves all expected tensor names and verifies their shapes.
//!   * The forward pass runs without panic on prefill *and* incremental decode.
//!   * Output logits are finite (no NaN/inf from numerical mishaps).
//!   * KV cache append/commit is consistent across multiple steps.

use std::collections::BTreeMap;

use ggml_rs::default_backend;
use gguf::value::{Array, Value};
use gguf::{GgmlType, GgufFile, TensorInfo};
use llama_rs::{LlamaModel, SampleParams, Sampler};
use tokenizer::byte_encoder::byte_to_char;

/// Build a synthetic Llama GGUF with the given config and (random-but-deterministic) weights.
fn build_tiny_llama_gguf() -> Vec<u8> {
    let vocab = 16usize;
    let embed = 8usize;
    let layers = 2usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = embed / heads;       // 4
    let ff = 16usize;
    let ctx = 32usize;

    let mut md: BTreeMap<String, Value> = BTreeMap::new();
    md.insert("general.architecture".into(),               Value::String("llama".into()));
    md.insert("general.alignment".into(),                  Value::U32(32));
    md.insert("llama.context_length".into(),               Value::U32(ctx as u32));
    md.insert("llama.embedding_length".into(),             Value::U32(embed as u32));
    md.insert("llama.block_count".into(),                  Value::U32(layers as u32));
    md.insert("llama.feed_forward_length".into(),          Value::U32(ff as u32));
    md.insert("llama.attention.head_count".into(),         Value::U32(heads as u32));
    md.insert("llama.attention.head_count_kv".into(),      Value::U32(kv_heads as u32));
    md.insert("llama.attention.layer_norm_rms_epsilon".into(), Value::F32(1e-5));
    md.insert("llama.rope.freq_base".into(),               Value::F32(10000.0));

    // Minimal tokenizer: byte-level BPE with no merges, vocab = 16 single-byte tokens.
    let toks: Vec<String> = (0..vocab as u8).map(|b| byte_to_char(b).to_string()).collect();
    md.insert("tokenizer.ggml.model".into(),  Value::String("gpt2".into()));
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(toks)));
    md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(vec![])));

    // Deterministic PRNG for weight generation. SplitMix64.
    let mut state: u64 = 0xC0FFEE;
    let mut rand = || -> f32 {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map to (-0.05, 0.05) — small to keep activations bounded.
        ((z >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.1
    };

    // Helper: produce an F32 tensor with a given GGUF shape (fastest-varying-first).
    let mut tensors: Vec<(TensorInfo, Vec<u8>)> = Vec::new();
    let mut add = |name: &str, gguf_shape: Vec<u64>, rand: &mut dyn FnMut() -> f32| {
        let numel: u64 = gguf_shape.iter().product();
        let mut bytes = Vec::with_capacity(numel as usize * 4);
        for _ in 0..numel { bytes.extend_from_slice(&rand().to_le_bytes()); }
        tensors.push((
            TensorInfo {
                name: name.into(),
                shape: gguf_shape,
                dtype: GgmlType::F32,
                offset: 0,
            },
            bytes,
        ));
    };

    // Recall: GGUF stores fastest-varying first, so a row-major [out, in] weight
    // is [in, out] in GGUF.
    add("token_embd.weight",  vec![embed as u64, vocab as u64], &mut rand);
    add("output_norm.weight", vec![embed as u64], &mut rand);
    add("output.weight",      vec![embed as u64, vocab as u64], &mut rand);

    let q_dim = (heads * head_dim) as u64;
    let k_dim = (kv_heads * head_dim) as u64;
    let e = embed as u64;
    let f_ = ff as u64;
    for i in 0..layers {
        add(&format!("blk.{i}.attn_norm.weight"),   vec![e],          &mut rand);
        add(&format!("blk.{i}.attn_q.weight"),      vec![e, q_dim],   &mut rand);
        add(&format!("blk.{i}.attn_k.weight"),      vec![e, k_dim],   &mut rand);
        add(&format!("blk.{i}.attn_v.weight"),      vec![e, k_dim],   &mut rand);
        add(&format!("blk.{i}.attn_output.weight"), vec![q_dim, e],   &mut rand);
        add(&format!("blk.{i}.ffn_norm.weight"),    vec![e],          &mut rand);
        add(&format!("blk.{i}.ffn_gate.weight"),    vec![e, f_],      &mut rand);
        add(&format!("blk.{i}.ffn_up.weight"),      vec![e, f_],      &mut rand);
        add(&format!("blk.{i}.ffn_down.weight"),    vec![f_, e],      &mut rand);
    }

    // Compute the offsets pass and write.
    gguf::reader::write_to_vec(&md, &tensors, 32).unwrap()
}

#[test]
fn synthetic_llama_forward_runs() {
    let bytes = build_tiny_llama_gguf();
    let g = GgufFile::from_bytes(bytes).unwrap();
    let backend = default_backend();
    let model = LlamaModel::from_gguf(&g, backend).unwrap();

    assert_eq!(model.config.embedding_dim, 8);
    assert_eq!(model.config.n_layers, 2);
    assert_eq!(model.config.n_heads, 2);
    assert_eq!(model.config.n_kv_heads, 1);
    assert_eq!(model.config.head_dim, 4);
    assert_eq!(model.config.n_rep(), 2);

    let mut kv = model.new_kv_cache(32);
    let prompt = vec![1u32, 2, 3, 4]; // 4 tokens

    // Prefill. forward() returns only the last-token row of logits (the LM
    // head's matmul is sliced to [1, hidden] before the vocab projection),
    // so the shape is [1, vocab] regardless of prompt length.
    let logits = model.forward(&prompt, &mut kv);
    assert_eq!(logits.shape(), &[1, model.config.vocab_size]);
    assert!(logits.data().iter().all(|v| v.is_finite()),
            "logits contained NaN or Inf");
    assert_eq!(kv.len, 4);

    // Decode 3 more tokens.
    let mut sampler = Sampler::new(SampleParams::greedy());
    let mut current = sampler.sample(&model.last_logits(&logits));
    for step in 1..=3 {
        let logits = model.forward(&[current], &mut kv);
        assert_eq!(logits.shape(), &[1, model.config.vocab_size]);
        assert!(logits.data().iter().all(|v| v.is_finite()),
                "logits NaN/Inf at decode step {step}");
        assert_eq!(kv.len, 4 + step);
        current = sampler.sample(&model.last_logits(&logits));
    }
}

#[test]
fn forward_is_deterministic() {
    // Two passes through a fresh KV cache should produce identical logits.
    let bytes = build_tiny_llama_gguf();
    let g = GgufFile::from_bytes(bytes).unwrap();
    let model = LlamaModel::from_gguf(&g, default_backend()).unwrap();
    let prompt = vec![1u32, 2, 3];

    let mut kv1 = model.new_kv_cache(16);
    let l1 = model.forward(&prompt, &mut kv1);

    let mut kv2 = model.new_kv_cache(16);
    let l2 = model.forward(&prompt, &mut kv2);

    assert_eq!(l1.data(), l2.data());
}

#[test]
fn kv_cache_concatenated_equals_full_prefill() {
    // Two-step decode (3 tokens then 1 token) should produce the same final
    // logits as a single 4-token prefill. This is the load-bearing invariant
    // that says our KV cache is wired up correctly.
    let bytes = build_tiny_llama_gguf();
    let g = GgufFile::from_bytes(bytes).unwrap();
    let model = LlamaModel::from_gguf(&g, default_backend()).unwrap();

    let prompt = vec![5u32, 6, 7, 8];

    let mut kv_full = model.new_kv_cache(16);
    let l_full = model.forward(&prompt, &mut kv_full);
    let last_full = model.last_logits(&l_full);

    let mut kv_split = model.new_kv_cache(16);
    let _ = model.forward(&prompt[..3], &mut kv_split);
    let l_split = model.forward(&prompt[3..], &mut kv_split);
    let last_split = model.last_logits(&l_split);

    let max_diff = last_full.data().iter()
        .zip(last_split.data().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4,
            "split decode diverges from full prefill (max_diff={max_diff})");
}
