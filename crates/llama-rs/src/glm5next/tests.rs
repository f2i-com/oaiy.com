//! Metadata-only fixtures built from the **real** GLM-5.3-Flash header
//! (`GLM-5.3-Flash-GGUF`, Q4_K_M, GGUF v3). Every
//! value below is what that file actually carries, so these tests pin the
//! layer map and the derived geometry against the shipped model rather than
//! against numbers this module invented. No tensors are needed: both config
//! parsers read metadata only.

use std::collections::BTreeMap;

use gguf::{Array, GgufFile, Value};

use super::*;
use crate::config::{Architecture, ModelConfig};

/// Blocks whose `head_count_kv` entry is 1 in the released weights. 3, 7,
/// ... 43 are the trunk's full-attention layers; 45 is the NextN block. Note
/// that 43 is followed by 45, not 47: the stride breaks, which is why the
/// loader reads the array instead of applying a `% 4` rule.
const MLA_BLOCKS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45];
const BLOCK_COUNT: usize = 46;

fn real_metadata() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "general.architecture".to_string(),
        Value::String("glm5next".to_string()),
    );

    for (k, v) in [
        ("vocab_size", 154880u32),
        ("context_length", 1048576),
        ("embedding_length", 4096),
        ("block_count", BLOCK_COUNT as u32),
        ("feed_forward_length", 12288),
        ("attention.head_count", 64),
        ("nextn_predict_layers", 1),
        ("leading_dense_block_count", 3),
        ("kda.head_dim", 128),
        ("ssm.conv_kernel", 4),
        ("attention.q_lora_rank", 1536),
        ("attention.kv_lora_rank", 512),
        ("attention.key_length_mla", 256),
        ("attention.value_length_mla", 256),
        ("attention.indexer.head_count", 32),
        ("attention.indexer.key_length", 128),
        ("attention.indexer.top_k", 2048),
        ("attention.indexer.kpool", 4),
        ("hyper_connection.count", 4),
        ("hyper_connection.sinkhorn_iterations", 20),
        ("expert_count", 288),
        ("expert_used_count", 8),
        ("expert_shared_count", 1),
        ("expert_feed_forward_length", 2048),
        ("expert_shared_feed_forward_length", 2048),
        ("expert_gating_func", 2),
        ("rope.dimension_count", 0),
    ] {
        m.insert(format!("glm5next.{k}"), Value::U32(v));
    }

    for (k, v) in [
        ("attention.layer_norm_rms_epsilon", 1e-5f32),
        ("attention.layer_norm_epsilon", 1e-6),
        ("hyper_connection.epsilon", 1e-6),
        ("kda.gate_lower_bound", -5.0),
        ("expert_weights_scale", 2.5),
    ] {
        m.insert(format!("glm5next.{k}"), Value::F32(v));
    }

    m.insert(
        "glm5next.expert_weights_norm".to_string(),
        Value::Bool(true),
    );

    let kv: Vec<u32> = (0..BLOCK_COUNT)
        .map(|i| u32::from(MLA_BLOCKS.contains(&i)))
        .collect();
    m.insert(
        "glm5next.attention.head_count_kv".to_string(),
        Value::Array(Array::U32(kv)),
    );

    m
}

fn file_from(m: &BTreeMap<String, Value>) -> GgufFile {
    let raw = gguf::reader::write_to_vec(m, &[], 32).expect("write gguf");
    GgufFile::from_bytes(raw).expect("parse gguf")
}

// The released model, on the faster drive. Gated like the repo's other
// real-model tests:
// `cargo test -p llama-rs glm5next::tests::released -- --ignored --nocapture`
use crate::glm5next::test_paths::released;

/// The first time any of this meets a real tensor: the split reader, the
/// metadata parse, the layer map and the shape assertions.
#[test]
#[ignore = "needs the released 194 GB model on disk"]
fn released_split_model_opens_and_parses() {
    let g = GgufFile::open_streaming(released()).expect("open the split model");

    // The split reader must present all five shards as one file.
    assert_eq!(g.n_shards(), 5, "five shards");
    assert_eq!(g.tensors().len(), 1412, "split.tensors.count");
    assert_eq!(g.architecture().expect("arch"), "glm5next");

    let cfg = ModelConfig::from_gguf(&g).expect("model config");
    assert_eq!(cfg.arch, Architecture::Glm5Next);
    assert_eq!(cfg.n_layers, 46);
    assert_eq!(cfg.embedding_dim, 4096);
    assert_eq!(cfg.vocab_size, 154880);
    assert_eq!(cfg.context_length, 1_048_576);
    // The array parse, on the real array.
    assert_eq!(cfg.n_kv_heads, 1, "absorbed MLA is MQA over one latent row");
    let mask = cfg.recurrent_layers.clone().expect("per-layer mask");
    assert_eq!(mask.len(), 46);

    let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm config");
    assert_eq!(glm.n_layer, 45, "the trunk excludes the NextN block");
    assert_eq!(glm.n_layer_nextn, 1);
    assert_eq!(glm.n_dense_lead, 3);
    assert_eq!(glm.n_expert, 288);
    assert_eq!(glm.n_expert_used, 8);
    assert_eq!(glm.n_ff_exp, 2048);
    assert_eq!(glm.kda_head_dim, 128);
    assert_eq!(glm.indexer_kpool, 4);
    assert_eq!(glm.n_select(), 2051);
    assert_eq!(glm.expert_gating_func, 2);
    assert_eq!(glm.swiglu_clamp_exp.len(), 46);
    assert_eq!(glm.swiglu_clamp_exp[0], 10.0);

    let kda = glm.layer_kinds[..glm.n_layer]
        .iter()
        .filter(|k| **k == LayerKind::Kda)
        .count();
    assert_eq!((kda, glm.n_layer - kda), (34, 11), "34 KDA + 11 MLA trunk");
    assert_eq!(glm.layer_kinds[45], LayerKind::Mla, "the MTP block is MLA-shaped");

    println!(
        "glm5next: {} shards, {} tensors, trunk {} ({} KDA + {} MLA), {} experts top-{}",
        g.n_shards(),
        g.tensors().len(),
        glm.n_layer,
        kda,
        glm.n_layer - kda,
        glm.n_expert,
        glm.n_expert_used
    );

    // Tensors must resolve ACROSS shards, with the shapes the loader asserts.
    // blk.0 is in shard 1; blk.45 (NextN) is in the last shard.
    let n_embd = cfg.embedding_dim;
    let d_inner = glm.kda_head_dim * cfg.n_heads;
    for (name, want_shape) in [
        ("token_embd.weight", vec![cfg.vocab_size, n_embd]),
        ("output_norm.weight", vec![n_embd]),
        ("blk.0.attn_q.weight", vec![d_inner, n_embd]),
        ("blk.0.ssm_conv1d_q.weight", vec![d_inner, 1, glm.ssm_conv_kernel]),
        ("blk.3.attn_k_b.weight", vec![cfg.n_heads, glm.kv_lora_rank, glm.qk_head_dim]),
        ("blk.3.indexer_compressor_ape.weight", vec![glm.indexer_kpool, glm.indexer_head_dim]),
        ("blk.0.hc_attn_fn.weight", vec![glm.hc_mix(), glm.hc_dim(n_embd)]),
        ("blk.44.ffn_gate_exps.weight", vec![glm.n_expert, glm.n_ff_exp, n_embd]),
        ("blk.45.nextn.eh_proj.weight", vec![n_embd, 2 * n_embd]),
    ] {
        let t = g
            .tensor_by_name(name)
            .unwrap_or_else(|| panic!("{name} did not resolve across the shards"));
        let got: Vec<usize> = t.shape.iter().map(|&d| d as usize).rev().collect();
        assert_eq!(got, want_shape, "{name} shape (shard {})", g.shard_of(t));
    }

    // hc_* must stop at the trunk, and the NextN block must have none.
    assert!(g.tensor_by_name("blk.44.hc_attn_fn.weight").is_some());
    assert!(
        g.tensor_by_name("blk.45.hc_attn_fn.weight").is_none(),
        "the NextN block has no mHC mixer"
    );
    // Tensors really are spread over the shards.
    let shards: std::collections::BTreeSet<usize> =
        g.tensors().iter().map(|t| g.shard_of(t)).collect();
    assert_eq!(shards.len(), 5, "tensors should occupy all five shards");
}

/// The whole streaming load: experts stay in the `.gguf` behind the bounded
/// cache, everything else is resolved and shape-checked.
#[test]
#[ignore = "needs the released 194 GB model on disk"]
fn released_split_model_loads_streaming() {
    use crate::Model;
    // Resident (non-expert) weights are ~25 GB; give the cache room on top.
    let budget: u64 = 64 << 30;
    let model = Model::open_streaming(released(), ggml_rs::default_backend(), budget)
        .expect("streaming load");

    let cfg = model.config();
    assert_eq!(cfg.arch, Architecture::Glm5Next);
    match &model {
        Model::Glm5Next(m) => {
            let (kda, mla) = m.layer_census();
            assert_eq!((kda, mla), (34, 11));
            assert_eq!(m.blocks.len(), 46, "trunk plus the NextN block");
            assert!(m.stream_shared.is_some(), "experts must be streamed");
            println!(
                "loaded: {kda} KDA + {mla} MLA, {} blocks, backend {}",
                m.blocks.len(),
                m.backend.name()
            );
        }
        other => panic!("expected Model::Glm5Next, got {other:?}"),
    }
}

#[test]
fn arch_string_maps_to_glm5next() {
    assert_eq!(Architecture::from_str("glm5next"), Architecture::Glm5Next);
    assert_eq!(Architecture::Glm5Next.name(), "glm5next");
    assert!(Architecture::Glm5Next.supported());
}

#[test]
fn head_count_kv_array_decodes_to_the_layer_map() {
    let g = file_from(&real_metadata());
    let cfg = ModelConfig::from_gguf(&g).expect("model config");

    assert_eq!(cfg.arch, Architecture::Glm5Next);
    assert_eq!(cfg.n_layers, BLOCK_COUNT);
    assert_eq!(cfg.embedding_dim, 4096);
    assert_eq!(cfg.vocab_size, 154880);
    // Absorbed MLA is MQA over one latent row, so the array maximum is 1 --
    // NOT the 64 that the scalar fallback would have produced.
    assert_eq!(cfg.n_kv_heads, 1);

    let mask = cfg.recurrent_layers.expect("per-layer mask");
    assert_eq!(mask.len(), BLOCK_COUNT);
    for i in 0..BLOCK_COUNT {
        assert_eq!(mask[i], !MLA_BLOCKS.contains(&i), "block {i} recurrent flag");
    }
}

#[test]
fn config_derives_the_trunk_and_geometry() {
    let g = file_from(&real_metadata());
    let cfg = ModelConfig::from_gguf(&g).expect("model config");
    let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm config");

    // 46 blocks, but blk.45 is the NextN draft block.
    assert_eq!(glm.n_layer_all, 46);
    assert_eq!(glm.n_layer_nextn, 1);
    assert_eq!(glm.n_layer, 45);
    assert_eq!(glm.n_dense_lead, 3);

    // 34 KDA + 11 MLA over the trunk; the 12th MLA-shaped block is MTP.
    let kda = glm.layer_kinds[..glm.n_layer]
        .iter()
        .filter(|&&k| k == LayerKind::Kda)
        .count();
    assert_eq!((kda, glm.n_layer - kda), (34, 11));
    assert_eq!(glm.layer_kinds[0], LayerKind::Kda);
    assert_eq!(glm.layer_kinds[3], LayerKind::Mla);
    assert_eq!(glm.layer_kinds[44], LayerKind::Kda);
    assert_eq!(glm.layer_kinds[45], LayerKind::Mla);

    // Derived geometry the tensor shape checks are written against.
    assert_eq!(glm.d_inner(cfg.n_heads), 8192);
    assert_eq!(glm.hc_dim(cfg.embedding_dim), 16384);
    assert_eq!(glm.hc_mix(), 24);
    // top_k + kpool - 1
    assert_eq!(glm.n_select(), 2051);

    assert_eq!(glm.n_expert, 288);
    assert_eq!(glm.n_expert_used, 8);
    assert_eq!(glm.n_ff_exp, 2048);
    assert_eq!(glm.n_ff_shexp, 2048);
    assert_eq!(glm.expert_gating_func, 2, "sigmoid gating");
    assert!(glm.expert_weights_norm);
    assert_eq!(glm.expert_weights_scale, 2.5);
    assert_eq!(glm.kda_gate_lower_bound, -5.0);
    assert_eq!(glm.indexer_kpool, 4);
}

/// A scalar `head_count_kv` cannot express the hybrid layout, so the loader
/// must refuse rather than fall back to an interval rule.
#[test]
fn scalar_head_count_kv_is_rejected() {
    let mut m = real_metadata();
    m.insert(
        "glm5next.attention.head_count_kv".to_string(),
        Value::U32(1),
    );
    let g = file_from(&m);
    let cfg = ModelConfig::from_gguf(&g).expect("model config");
    assert!(cfg.recurrent_layers.is_none());
    assert!(Glm5NextConfig::from_gguf(&g, &cfg).is_err());
}

/// glm5next is NoPE. A non-zero rope dimension means the file is not the
/// architecture we think it is.
#[test]
fn nonzero_rope_dimension_is_rejected() {
    let mut m = real_metadata();
    m.insert("glm5next.rope.dimension_count".to_string(), Value::U32(128));
    let g = file_from(&m);
    let cfg = ModelConfig::from_gguf(&g).expect("model config");
    assert!(Glm5NextConfig::from_gguf(&g, &cfg).is_err());
}

/// A non-negative KDA gate bound selects a different (softplus) branch in
/// the reference, so it must not be accepted silently.
#[test]
fn nonnegative_kda_gate_bound_is_rejected() {
    let mut m = real_metadata();
    m.insert("glm5next.kda.gate_lower_bound".to_string(), Value::F32(0.0));
    let g = file_from(&m);
    let cfg = ModelConfig::from_gguf(&g).expect("model config");
    assert!(Glm5NextConfig::from_gguf(&g, &cfg).is_err());
}

/// gemma4 publishes `attention.head_count_kv` as an array too, but its
/// entries are real per-layer KV head counts rather than a recurrent mask.
/// Only glm5next may reinterpret them.
#[test]
fn other_arches_keep_their_scalar_kv_heads() {
    let mut m = BTreeMap::new();
    m.insert(
        "general.architecture".to_string(),
        Value::String("qwen3moe".to_string()),
    );
    for (k, v) in [
        ("vocab_size", 1000u32),
        ("context_length", 4096),
        ("embedding_length", 256),
        ("block_count", 4),
        ("feed_forward_length", 512),
        ("attention.head_count", 8),
        ("attention.head_count_kv", 4),
    ] {
        m.insert(format!("qwen3moe.{k}"), Value::U32(v));
    }
    let g = file_from(&m);
    let cfg = ModelConfig::from_gguf(&g).expect("model config");
    assert!(cfg.recurrent_layers.is_none());
    assert_eq!(cfg.n_kv_heads, 4, "scalar kv heads must survive untouched");
}
