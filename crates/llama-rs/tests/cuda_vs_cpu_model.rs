//! Build a tiny synthetic Llama GGUF (same as `synthetic_forward.rs`) and run
//! a forward pass on both backends. Logits should match within ~1e-3 relative
//! tolerance (CUDA uses different float-add ordering, so they're not bitwise
//! identical, but the model output should be effectively the same).
//!
//! Skipped automatically when no CUDA device is reachable.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::sync::Arc;

use ggml_rs::{default_backend, Backend};
use ggml_rs_cuda::CudaBackend;
use gguf::value::{Array, Value};
use gguf::{GgmlType, GgufFile, TensorInfo};
use llama_rs::{Model, SampleParams, Sampler};
use tokenizer::byte_encoder::byte_to_char;

fn build_tiny_llama_gguf() -> Vec<u8> {
    let vocab = 16usize;
    let embed = 8usize;
    let layers = 2usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = embed / heads;
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

    let toks: Vec<String> = (0..vocab as u8).map(|b| byte_to_char(b).to_string()).collect();
    md.insert("tokenizer.ggml.model".into(),  Value::String("gpt2".into()));
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(toks)));
    md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(vec![])));

    let mut state: u64 = 0xC0FFEE;
    let mut rand = || -> f32 {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.1
    };

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

    gguf::reader::write_to_vec(&md, &tensors, 32).unwrap()
}

fn try_cuda() -> Option<Arc<dyn Backend>> {
    match CudaBackend::new(0) {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            eprintln!("[skipping] CUDA unavailable: {e}");
            None
        }
    }
}

#[test]
fn cpu_and_cuda_produce_close_logits() {
    let Some(cuda) = try_cuda() else { return };

    let bytes = build_tiny_llama_gguf();
    let g_cpu = GgufFile::from_bytes(bytes.clone()).unwrap();
    let g_cuda = GgufFile::from_bytes(bytes).unwrap();

    let cpu_model = Model::load(&g_cpu, default_backend()).unwrap();
    let cuda_model = Model::load(&g_cuda, cuda).unwrap();

    let prompt = vec![1u32, 3, 5, 7];

    let mut cpu_kv = cpu_model.new_kv_cache(16);
    let mut cuda_kv = cuda_model.new_kv_cache(16);

    let cpu_logits = cpu_model.forward(&prompt, &mut cpu_kv);
    let cuda_logits = cuda_model.forward(&prompt, &mut cuda_kv);

    assert_eq!(cpu_logits.shape(), cuda_logits.shape());

    // Bring both to host for comparison.
    let cpu_h = cpu_logits.to_host();
    let cuda_h = cuda_logits.to_host();

    let max_diff = cpu_h
        .data()
        .iter()
        .zip(cuda_h.data().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("max abs diff between cpu and cuda logits: {max_diff:.6}");
    assert!(
        max_diff < 5e-4,
        "CPU vs CUDA divergence too large: max_diff={max_diff}"
    );
}

#[test]
fn cpu_and_cuda_greedy_token_sequences_match() {
    let Some(cuda) = try_cuda() else { return };

    let bytes = build_tiny_llama_gguf();
    let g_cpu = GgufFile::from_bytes(bytes.clone()).unwrap();
    let g_cuda = GgufFile::from_bytes(bytes).unwrap();

    let cpu_model = Model::load(&g_cpu, default_backend()).unwrap();
    let cuda_model = Model::load(&g_cuda, cuda).unwrap();

    let prompt = vec![1u32, 2, 3];
    let mut sampler_cpu = Sampler::new(SampleParams::greedy());
    let mut sampler_cuda = Sampler::new(SampleParams::greedy());

    let mut cpu_kv = cpu_model.new_kv_cache(16);
    let mut cuda_kv = cuda_model.new_kv_cache(16);

    let l_cpu = cpu_model.forward(&prompt, &mut cpu_kv);
    let l_cuda = cuda_model.forward(&prompt, &mut cuda_kv);

    let cpu_seq: Vec<u32> = (0..6)
        .scan(sampler_cpu.sample(&cpu_model.last_logits(&l_cpu)), |cur, _| {
            let out = *cur;
            let l = cpu_model.forward(&[*cur], &mut cpu_kv);
            *cur = sampler_cpu.sample(&cpu_model.last_logits(&l));
            Some(out)
        })
        .collect();

    let cuda_seq: Vec<u32> = (0..6)
        .scan(sampler_cuda.sample(&cuda_model.last_logits(&l_cuda)), |cur, _| {
            let out = *cur;
            let l = cuda_model.forward(&[*cur], &mut cuda_kv);
            *cur = sampler_cuda.sample(&cuda_model.last_logits(&l));
            Some(out)
        })
        .collect();

    assert_eq!(cpu_seq, cuda_seq, "greedy decode diverged across backends");
}
