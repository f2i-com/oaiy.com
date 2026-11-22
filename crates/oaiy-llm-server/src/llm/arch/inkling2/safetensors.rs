//! Direct in-place safetensors loader for `thinkingmachines/Inkling-2-Mini`
//! (and other `inkling2` / `inkling2_loop` Hugging Face checkpoints).
//!
//! Reads `config.json`, `tokenizer.json`, and `*.safetensors` in place without
//! conversion:
//!   * Dense BF16 projections stay compact via `WgpuBackend::half_linear` on GPU
//!     (when available and within budget) or host [`Bf16Linear`] (2 B/weight).
//!   * NVFP4 (layers 0..19) and MXFP8 (layers 20..23) routed experts stay packed
//!     on disk and are fetched lazily through [`PackedExpertCache`], de-interleaving
//!     `w13` rows (even = gate, odd = up) on the fly.
//!   * Built-in vision tower (`model.visual.*` / `model.vision_tower.*`) can be
//!     loaded directly from the same checkpoint directory.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ggml_rs::{Backend, Tensor};
use oaiy_engine::io::read_exact_at;
use oaiy_engine::json::Json;
use oaiy_engine::safetensors::{Dtype, StIndex};
use tokenizer::Tokenizer;

use crate::llm::loader::Weight;
use crate::llm::{Architecture, LlamaError, Result};

use super::experts::{
    bf16_bits_to_f32, Bf16Linear, Inkling2Experts, Mxfp8LayerSpec, Nvfp4LayerSpec,
    PackedExpertCache,
};
use super::state::Inkling2State;
use super::vision::{Inkling2Vision, Inkling2VisionConfig};
use super::{
    AttnNorms, EmbedTable, Inkling2Config, Inkling2Layer, Inkling2LoopLayer, Inkling2Model,
    LoopConfig,
};

fn cfg_err(msg: impl Into<String>) -> LlamaError {
    LlamaError::Config(msg.into())
}

/// True if `dir` is an Inkling-2 or Inkling-2-Loop Hugging Face safetensors checkpoint.
pub fn is_inkling2_checkpoint(path: &Path) -> bool {
    let cfg_path = path.join("config.json");
    let Ok(bytes) = std::fs::read(&cfg_path) else {
        return false;
    };
    let Ok(doc) = Json::parse(&bytes) else {
        return false;
    };
    matches!(
        doc.get("model_type").and_then(Json::as_str),
        Some("inkling2" | "inkling2_text" | "inkling2_loop" | "inkling2_loop_text")
    )
}

/// Parse `Inkling2Config` and optional `Inkling2VisionConfig` from HF `config.json`.
pub fn parse_hf_config(dir: &Path) -> Result<(Inkling2Config, Option<Inkling2VisionConfig>, Option<u32>)> {
    let raw = std::fs::read(dir.join("config.json"))
        .map_err(|e| cfg_err(format!("read config.json: {e}")))?;
    let root = Json::parse(&raw).map_err(|e| cfg_err(format!("parse config.json: {e}")))?;
    let top_eos = root.get("eos_token_id").and_then(Json::as_i64).map(|v| v as u32);
    let tc = root.get("text_config").unwrap_or(&root);

    let req_usize = |k: &str| -> Result<usize> {
        tc.get(k)
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .ok_or_else(|| cfg_err(format!("config.json missing `{k}`")))
    };
    let opt_usize = |keys: &[&str], default: usize| -> usize {
        keys.iter()
            .find_map(|k| tc.get(k).and_then(Json::as_f64).map(|v| v as usize))
            .unwrap_or(default)
    };
    let opt_f32 = |keys: &[&str], default: f32| -> f32 {
        keys.iter()
            .find_map(|k| tc.get(k).and_then(Json::as_f64).map(|v| v as f32))
            .unwrap_or(default)
    };

    let hidden = req_usize("hidden_size")?;
    let n_phys_layers = req_usize("num_hidden_layers")?;
    let vocab_size_padded = req_usize("vocab_size")?;
    let vocab_size_unpadded = opt_usize(&["unpadded_vocab_size"], vocab_size_padded);
    let context_length = opt_usize(&["model_max_length", "max_position_embeddings"], 1_048_576);
    let n_heads_full = req_usize("num_attention_heads")?;
    let n_kv_heads_full = opt_usize(&["num_key_value_heads"], n_heads_full);
    let head_dim_full = opt_usize(&["head_dim"], hidden / n_heads_full.max(1));
    let n_heads_swa = opt_usize(&["swa_num_attention_heads"], n_heads_full);
    let n_kv_heads_swa = opt_usize(&["swa_num_key_value_heads"], n_kv_heads_full);
    let head_dim_swa = opt_usize(&["swa_head_dim"], head_dim_full);
    let sliding_window = opt_usize(&["sliding_window_size"], 1024);

    let is_swa: Vec<bool> = if let Some(arr) = tc.get("layer_types").and_then(Json::as_array) {
        arr.iter()
            .map(|v| v.as_str() == Some("hybrid_sliding"))
            .collect()
    } else if let Some(arr) = tc.get("local_layer_ids").and_then(Json::as_array) {
        let local: HashSet<usize> = arr
            .iter()
            .filter_map(|v| v.as_f64().map(|x| x as usize))
            .collect();
        (0..n_phys_layers).map(|i| local.contains(&i)).collect()
    } else {
        (0..n_phys_layers).map(|i| (i + 1) % 6 != 0).collect()
    };
    if is_swa.len() != n_phys_layers {
        return Err(cfg_err("inkling2: layer_types length does not match num_hidden_layers"));
    }

    let head_counts: Vec<usize> = is_swa
        .iter()
        .map(|&s| if s { n_heads_swa } else { n_heads_full })
        .collect();
    let kv_head_counts: Vec<usize> = is_swa
        .iter()
        .map(|&s| if s { n_kv_heads_swa } else { n_kv_heads_full })
        .collect();
    let head_dims: Vec<usize> = is_swa
        .iter()
        .map(|&s| if s { head_dim_swa } else { head_dim_full })
        .collect();

    let loop_cfg = if let Some(lc) = tc.get("loop_config").filter(|v| v.as_object().is_some()) {
        let core_start = lc
            .get("core_start")
            .and_then(Json::as_f64)
            .ok_or_else(|| cfg_err("loop_config missing core_start"))? as usize;
        let core_end = lc
            .get("core_end")
            .and_then(Json::as_f64)
            .ok_or_else(|| cfg_err("loop_config missing core_end"))? as usize;
        let num_passes = lc
            .get("num_passes")
            .and_then(Json::as_f64)
            .ok_or_else(|| cfg_err("loop_config missing num_passes"))? as usize;
        let gate_rank = lc
            .get("gate_rank")
            .and_then(Json::as_f64)
            .ok_or_else(|| cfg_err("loop_config missing gate_rank"))? as usize;
        let varying = lc
            .get("is_varloop")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        Some(LoopConfig {
            core_start,
            core_end,
            num_passes,
            gate_rank,
            varying,
        })
    } else {
        None
    };

    let cfg = Inkling2Config {
        arch: if loop_cfg.is_some() {
            Architecture::Inkling2Loop
        } else {
            Architecture::Inkling2
        },
        hidden,
        n_phys_layers,
        vocab_size_padded,
        vocab_size_unpadded,
        context_length,
        head_counts,
        kv_head_counts,
        head_dims,
        is_swa,
        sliding_window,
        d_rel: opt_usize(&["d_rel"], 16),
        rel_extent: opt_usize(&["rel_extent"], 1024),
        qk_clip: opt_f32(&["qk_l2_clip_threshold"], 22.627_417),
        rel_clip: opt_f32(&["rel_l2_clip_threshold"], 8.0),
        log_scale_floor: opt_usize(&["log_scaling_n_floor"], 8192),
        log_scale_alpha: opt_f32(&["log_scaling_alpha"], 0.1),
        rms_eps: opt_f32(&["rms_norm_eps"], 1e-6),
        conv_kernel: opt_usize(&["sconv_kernel_size", "conv_kernel_size"], 4),
        attn_res_block_size: opt_usize(&["n_layers_per_attn_res_block"], 6),
        n_experts: opt_usize(&["n_routed_experts", "num_local_experts"], 128),
        n_experts_used: opt_usize(&["num_experts_per_tok"], 8),
        ff_dim: opt_usize(&["moe_intermediate_size", "intermediate_size"], 1152),
        route_scale: opt_f32(&["route_scale", "routed_scaling_factor"], 8.0),
        situ_beta: opt_f32(&["mlp_gate_softcap"], 4.0),
        situ_linear_beta: opt_f32(&["mlp_up_softcap"], 25.0),
        final_logit_softcap: opt_f32(&["final_logit_softcapping"], 30.0),
        loop_cfg,
    };
    cfg.validate()?;

    let vision_cfg = root.get("vision_config").and_then(|vc| {
        if vc.as_object().is_none() {
            return None;
        }
        let patch_size = vc
            .get("native_patch_size")
            .or_else(|| vc.get("patch_size"))
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .unwrap_or(20);
        let spatial_merge_size = vc
            .get("adapter_merge_size")
            .or_else(|| vc.get("spatial_merge_size"))
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .unwrap_or(2);
        let hidden_size = vc.get("hidden_size").and_then(Json::as_f64).map(|v| v as usize)?;
        let num_heads = vc
            .get("num_attention_heads")
            .and_then(Json::as_f64)
            .map(|v| v as usize)?;
        let num_layers = vc
            .get("num_hidden_layers")
            .and_then(Json::as_f64)
            .map(|v| v as usize)?;
        let posemb_grid_size = vc
            .get("posemb_grid_size")
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .unwrap_or(32);
        let ff_dim = vc
            .get("intermediate_size")
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let ratio = vc.get("mlp_ratio").and_then(Json::as_f64).unwrap_or(4.0);
                (hidden_size as f64 * ratio) as usize
            });
        let proj_hidden_dim = vc
            .get("projection_hidden_dim")
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .unwrap_or(spatial_merge_size * spatial_merge_size * hidden_size);
        let text_hidden_size = vc
            .get("text_hidden_size")
            .and_then(Json::as_f64)
            .map(|v| v as usize)
            .unwrap_or(hidden);
        let layer_norm_eps = vc
            .get("layer_norm_eps")
            .and_then(Json::as_f64)
            .map(|v| v as f32)
            .unwrap_or(1e-5);
        Some(Inkling2VisionConfig {
            patch_size,
            spatial_merge_size,
            posemb_grid_size,
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads.max(1),
            ff_dim,
            num_layers,
            proj_hidden_dim,
            text_hidden_size,
            layer_norm_eps,
            rope_theta: 10000.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        })
    });

    Ok((cfg, vision_cfg, top_eos))
}

/// Load Inkling-2's GPT-4o (`o200k_base`) byte-level BPE tokenizer from `tokenizer.json`.
pub fn load_hf_tokenizer(dir: &Path, eos_hint: Option<u32>) -> Result<Tokenizer> {
    let raw = std::fs::read(dir.join("tokenizer.json"))
        .map_err(|e| cfg_err(format!("read tokenizer.json: {e}")))?;
    let doc = Json::parse(&raw).map_err(|e| cfg_err(format!("parse tokenizer.json: {e}")))?;
    let model = doc
        .get("model")
        .ok_or_else(|| cfg_err("tokenizer.json missing model"))?;
    let vocab = model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or_else(|| cfg_err("tokenizer.json missing model.vocab"))?;
    let added: &[Json] = doc
        .get("added_tokens")
        .and_then(Json::as_array)
        .unwrap_or(&[]);

    let parse_id = |v: &Json| -> Result<usize> {
        let id = v
            .as_i64()
            .ok_or_else(|| cfg_err("tokenizer.json token id must be an integer"))?;
        if !(0..=2_000_000).contains(&id) {
            return Err(cfg_err(format!("tokenizer.json token id out of range: {id}")));
        }
        Ok(id as usize)
    };
    let mut max_id = 0usize;
    for (_, v) in vocab {
        max_id = max_id.max(parse_id(v)?);
    }
    for a in added {
        if let Some(v) = a.get("id") {
            max_id = max_id.max(parse_id(v)?);
        }
    }
    let mut tokens = vec![String::new(); max_id + 1];
    for (tok, v) in vocab {
        let id = parse_id(v)?;
        tokens[id] = tok.clone();
    }
    let mut special = Vec::with_capacity(added.len());
    for a in added {
        let id = parse_id(
            a.get("id")
                .ok_or_else(|| cfg_err("missing added_token id"))?,
        )?;
        let content = a
            .get("content")
            .and_then(Json::as_str)
            .ok_or_else(|| cfg_err("missing added_token content"))?;
        tokens[id] = content.to_string();
        special.push(id as u32);
    }

    let merges_arr = model
        .get("merges")
        .and_then(Json::as_array)
        .ok_or_else(|| cfg_err("tokenizer.json missing model.merges"))?;
    let mut merges = Vec::with_capacity(merges_arr.len());
    for m in merges_arr {
        if let Some(arr) = m.as_array() {
            if arr.len() == 2 {
                let a = arr[0].as_str().ok_or_else(|| cfg_err("invalid merge"))?;
                let b = arr[1].as_str().ok_or_else(|| cfg_err("invalid merge"))?;
                merges.push((a.to_string(), b.to_string()));
                continue;
            }
        }
        let (a, b) = m
            .as_str()
            .and_then(|s| s.split_once(' '))
            .ok_or_else(|| cfg_err("invalid merge entry"))?;
        merges.push((a.to_string(), b.to_string()));
    }

    let eos = eos_hint
        .or_else(|| {
            tokens
                .iter()
                .position(|t| t == "<|content_model_end_sampling|>")
                .map(|i| i as u32)
        })
        .ok_or_else(|| cfg_err("cannot determine Inkling-2 EOS token id"))?;
    let pad = tokens
        .iter()
        .position(|t| t == "<|endoftext|>")
        .map(|i| i as u32);

    Tokenizer::from_gpt4o_bpe_parts(tokens, merges, special, eos, pad)
        .map_err(LlamaError::Tokenizer)
}

pub(crate) fn read_st_raw(index: &StIndex, name: &str) -> Result<(oaiy_engine::safetensors::TensorInfo, Vec<u8>)> {
    let info = index
        .get(name)
        .ok_or_else(|| LlamaError::MissingTensor(name.to_string()))?
        .clone();
    let f = File::open(index.shard_path(info.shard))
        .map_err(|e| cfg_err(format!("open shard for {name}: {e}")))?;
    let mut buf = vec![0u8; info.nbytes as usize];
    read_exact_at(&f, &mut buf, info.start)
        .map_err(|e| cfg_err(format!("read {name}: {e}")))?;
    Ok((info, buf))
}

pub(crate) fn read_st_f32(index: &StIndex, name: &str) -> Result<Vec<f32>> {
    let (info, raw) = read_st_raw(index, name)?;
    decode_raw_to_f32(&raw, info.dtype, name)
}

fn decode_raw_to_f32(raw: &[u8], dtype: Dtype, name: &str) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        Dtype::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        other => Err(cfg_err(format!("unsupported dtype {other:?} for {name}"))),
    }
}

/// Load a 2D linear weight `[out, in]` from safetensors, staying compact in BF16/F16.
pub(crate) fn read_st_weight(index: &StIndex, name: &str, backend: &dyn Backend) -> Result<Weight> {
    let (info, raw) = read_st_raw(index, name)?;
    if info.shape.len() != 2 {
        return Err(cfg_err(format!("{name}: expected 2D weight, got {:?}", info.shape)));
    }
    let (n, k) = (info.shape[0], info.shape[1]);

    // When running on WebGPU, pack as F16 on GPU via `half_linear` (2 bytes/weight) when it fits.
    #[cfg(feature = "webgpu")]
    if let Some(wgpu) = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
        if let Ok(vals) = decode_raw_to_f32(&raw, info.dtype, name) {
            if let Ok(packed) = wgpu.half_linear(vals, n, k) {
                return Ok(Weight::Packed(packed));
            }
        }
    }

    match info.dtype {
        Dtype::BF16 => Ok(Weight::Packed(Arc::new(Bf16Linear::from_le_bytes(&raw, n, k)?))),
        _ => {
            let vals = decode_raw_to_f32(&raw, info.dtype, name)?;
            Ok(Weight::Dense(Tensor::from_vec(vals, vec![n, k])).to_device(backend))
        }
    }
}

/// Load an `Inkling2Model` (and optionally its built-in `Inkling2Vision` tower) directly
/// from a Hugging Face safetensors directory (`thinkingmachines/Inkling-2-Mini`).
pub fn load_from_safetensors(
    dir: &Path,
    backend: Arc<dyn Backend>,
    expert_cache_bytes: usize,
    load_vision: bool,
) -> Result<(Inkling2Model, Option<Inkling2Vision>)> {
    let (cfg, vision_cfg, eos_hint) = parse_hf_config(dir)?;
    let tokenizer = load_hf_tokenizer(dir, eos_hint)?;
    let index = StIndex::open(dir).map_err(|e| cfg_err(format!("open safetensors: {e}")))?;

    // Detect tensor naming convention: native Thinking Machines (`model.llm.*`) vs HF `transformers` (`model.language_model.*` or `model.*`).
    let prefix = if index.get("model.llm.embed.weight").is_some() {
        "model.llm."
    } else if index.get("model.language_model.embed_tokens.weight").is_some() {
        "model.language_model."
    } else {
        "model."
    };
    let native_names = prefix == "model.llm.";

    let p = |raw: &str, hf: &str| -> String {
        if native_names {
            format!("{prefix}{raw}")
        } else {
            format!("{prefix}{hf}")
        }
    };

    // Embed table (kept compact in BF16 on host for instant lookup without 2x F32 bloat).
    let embed_name = p("embed.weight", "embed_tokens.weight");
    let (emb_info, emb_raw) = read_st_raw(&index, &embed_name)?;
    let embed_table = match emb_info.dtype {
        Dtype::BF16 => EmbedTable::Bf16(Bf16Linear::from_le_bytes(
            &emb_raw,
            cfg.vocab_size_padded,
            cfg.hidden,
        )?),
        _ => EmbedTable::F32(
            decode_raw_to_f32(&emb_raw, emb_info.dtype, &embed_name)?,
            cfg.hidden,
        ),
    };
    let embed_norm = read_st_f32(&index, &p("embed_norm.weight", "embed_tokens.embed_norm.weight"))?;
    let output_norm = read_st_f32(&index, &p("norm.weight", "norm.weight"))?;
    let output_res_query = read_st_f32(&index, &p("attn_res.output_query", "final_input.query"))?;

    let unembed_name = if index.get(&format!("{prefix}unembed.weight")).is_some() {
        format!("{prefix}unembed.weight")
    } else {
        "lm_head.weight".to_string()
    };
    let output = read_st_weight(&index, &unembed_name, backend.as_ref())?;

    // Pre-open shard File handles for lock-free parallel expert preads.
    let mut shard_files: Vec<Arc<File>> = Vec::with_capacity(index.shard_count());
    for s in 0..index.shard_count() {
        let f = File::open(index.shard_path(s))
            .map_err(|e| cfg_err(format!("open shard {s}: {e}")))?;
        shard_files.push(Arc::new(f));
    }
    let expert_cache = Arc::new(PackedExpertCache::new(expert_cache_bytes));

    let mut layers = Vec::with_capacity(cfg.n_phys_layers);
    for i in 0..cfg.n_phys_layers {
        let is_swa = cfg.is_swa[i];
        let n_heads = cfg.head_counts[i];
        let n_kv_heads = cfg.kv_head_counts[i];
        let head_dim = cfg.head_dims[i];
        let rel_extent = if is_swa { cfg.sliding_window } else { cfg.rel_extent };

        let attn_norms = if is_swa {
            let w = read_st_f32(
                &index,
                &p(&format!("layers.{i}.attn_norm.weight"), &format!("layers.{i}.input_layernorm.weight")),
            )?;
            AttnNorms::Shared(w)
        } else {
            let q = read_st_f32(
                &index,
                &p(&format!("layers.{i}.q_input_norm.weight"), &format!("layers.{i}.q_input_layernorm.weight")),
            )?;
            let k = read_st_f32(
                &index,
                &p(&format!("layers.{i}.k_input_norm.weight"), &format!("layers.{i}.k_input_layernorm.weight")),
            )?;
            let v = read_st_f32(
                &index,
                &p(&format!("layers.{i}.v_input_norm.weight"), &format!("layers.{i}.v_input_layernorm.weight")),
            )?;
            AttnNorms::Separate { q, k, v }
        };

        let attn_res_query = if i > 0 {
            let qname = if native_names {
                format!("{prefix}attn_res.attn_queries.{}", i - 1)
            } else {
                format!("{prefix}layers.{i}.attention_input.query")
            };
            Some(read_st_f32(&index, &qname)?)
        } else {
            None
        };
        let ffn_res_query = read_st_f32(
            &index,
            &p(&format!("attn_res.mlp_queries.{i}"), &format!("layers.{i}.mlp_input.query")),
        )?;

        let wq = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.wq_du.weight"), &format!("layers.{i}.self_attn.q_proj.linear.weight")),
            backend.as_ref(),
        )?;
        let wk = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.wk_dv.weight"), &format!("layers.{i}.self_attn.k_proj.linear.weight")),
            backend.as_ref(),
        )?;
        let wv = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.wv_dv.weight"), &format!("layers.{i}.self_attn.v_proj.linear.weight")),
            backend.as_ref(),
        )?;
        let wo = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.wo_ud.weight"), &format!("layers.{i}.self_attn.o_proj.linear.weight")),
            backend.as_ref(),
        )?;
        let attn_gate = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.gate_attn.weight"), &format!("layers.{i}.self_attn.o_proj.gate.weight")),
            backend.as_ref(),
        )?;
        let attn_rel = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.wr_du.weight"), &format!("layers.{i}.self_attn.relative_position_bias.linear.weight")),
            backend.as_ref(),
        )?;
        let attn_sink_proj = read_st_weight(
            &index,
            &p(&format!("layers.{i}.attn.attn_sink_proj.weight"), &format!("layers.{i}.self_attn.sink_proj.weight")),
            backend.as_ref(),
        )?;
        let attn_q_clip = read_st_f32(
            &index,
            &p(&format!("layers.{i}.attn.q_l2_clip.scale"), &format!("layers.{i}.self_attn.q_proj.norm.scale")),
        )?;
        let attn_rel_clip = read_st_f32(
            &index,
            &p(&format!("layers.{i}.attn.rel_l2_clip.scale"), &format!("layers.{i}.self_attn.relative_position_bias.norm.scale")),
        )?;

        // `rel_logits_proj.proj` in safetensors is `[d_rel, rel_extent]` -> transpose to `[rel_extent, d_rel]`.
        let rel_raw = read_st_f32(
            &index,
            &p(&format!("layers.{i}.attn.rel_logits_proj.proj"), &format!("layers.{i}.self_attn.relative_position_bias.proj")),
        )?;
        if rel_raw.len() != rel_extent * cfg.d_rel {
            return Err(cfg_err(format!(
                "layer {i} rel_logits_proj len {} != {}",
                rel_raw.len(),
                rel_extent * cfg.d_rel
            )));
        }
        let mut attn_rel_bias = vec![0.0f32; rel_extent * cfg.d_rel];
        for r in 0..cfg.d_rel {
            for d in 0..rel_extent {
                attn_rel_bias[d * cfg.d_rel + r] = rel_raw[r * rel_extent + d];
            }
        }

        // Short convolutions:
        //   * Static `sconv.weight` (`[C, 1, K]`) is already oldest->newest (`[C, K]`).
        //   * Dynamic `sconv.static_w` (`[K, C]`, newest->oldest) is flipped and transposed to `[C, K]` oldest->newest
        //     to match GGUF's `attn_{k,v}_conv.weight` (`conversion/inkling2.py:262`).
        let (attn_k_conv, attn_k_conv_proj, attn_v_conv, attn_v_conv_proj) = if is_swa {
            let kc = read_st_f32(
                &index,
                &p(&format!("layers.{i}.attn.k_sconv.weight"), &format!("layers.{i}.self_attn.k_proj.sconv.conv1d.weight")),
            )?;
            let vc = read_st_f32(
                &index,
                &p(&format!("layers.{i}.attn.v_sconv.weight"), &format!("layers.{i}.self_attn.v_proj.sconv.conv1d.weight")),
            )?;
            (kc, None, vc, None)
        } else {
            let kv_c = n_kv_heads * head_dim;
            let k_raw = read_st_f32(
                &index,
                &p(&format!("layers.{i}.attn.k_sconv.static_w"), &format!("layers.{i}.self_attn.k_proj.sconv.static_weight")),
            )?;
            let v_raw = read_st_f32(
                &index,
                &p(&format!("layers.{i}.attn.v_sconv.static_w"), &format!("layers.{i}.self_attn.v_proj.sconv.static_weight")),
            )?;
            if k_raw.len() != cfg.conv_kernel * kv_c || v_raw.len() != cfg.conv_kernel * kv_c {
                return Err(cfg_err(format!(
                    "layer {i} dynamic sconv static_w len ({}, {}) != {}",
                    k_raw.len(),
                    v_raw.len(),
                    cfg.conv_kernel * kv_c
                )));
            }
            let kc = flip_transpose_static_w(&k_raw, cfg.conv_kernel, kv_c);
            let vc = flip_transpose_static_w(&v_raw, cfg.conv_kernel, kv_c);
            let kp = read_st_weight(
                &index,
                &p(&format!("layers.{i}.attn.k_sconv.weight_proj"), &format!("layers.{i}.self_attn.k_proj.sconv.weight_proj.weight")),
                backend.as_ref(),
            )?;
            let vp = read_st_weight(
                &index,
                &p(&format!("layers.{i}.attn.v_sconv.weight_proj"), &format!("layers.{i}.self_attn.v_proj.sconv.weight_proj.weight")),
                backend.as_ref(),
            )?;
            (kc, Some(kp), vc, Some(vp))
        };

        let attn_post_conv = read_st_f32(
            &index,
            &p(&format!("layers.{i}.attn_sconv.weight"), &format!("layers.{i}.post_attention_convolution.conv1d.weight")),
        )?;
        let ffn_post_conv = read_st_f32(
            &index,
            &p(&format!("layers.{i}.mlp_sconv.weight"), &format!("layers.{i}.post_mlp_convolution.conv1d.weight")),
        )?;
        let ffn_norm = read_st_f32(
            &index,
            &p(&format!("layers.{i}.mlp_norm.weight"), &format!("layers.{i}.post_attention_layernorm.weight")),
        )?;
        let ffn_gate_inp = read_st_weight(
            &index,
            &format!("{prefix}layers.{i}.mlp.gate.weight"),
            backend.as_ref(),
        )?;
        let exp_probs_b = {
            let b_native = format!("{prefix}layers.{i}.mlp.gate.bias");
            let b_hf = format!("{prefix}layers.{i}.mlp.gate.e_score_correction_bias");
            if index.get(&b_native).is_some() {
                Some(read_st_f32(&index, &b_native)?)
            } else if index.get(&b_hf).is_some() {
                Some(read_st_f32(&index, &b_hf)?)
            } else {
                None
            }
        };

        let experts = load_st_experts(&index, &shard_files, &expert_cache, prefix, i, &cfg)?;

        layers.push(Inkling2Layer {
            is_swa,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_norms,
            attn_norms_dev: None,
            decode_dev: None,
            attn_res_query,
            ffn_res_query,
            wq,
            wk,
            wv,
            wo,
            attn_gate,
            attn_q_clip,
            attn_sink_proj,
            attn_rel,
            wqrsg: None,
            wk_dyn: None,
            wv_dyn: None,
            attn_rel_bias,
            attn_rel_clip,
            attn_k_conv,
            attn_v_conv,
            attn_k_conv_proj,
            attn_v_conv_proj,
            attn_post_conv,
            ffn_norm,
            ffn_norm_dev: None,
            ffn_gate_inp,
            ffn_gate_inp_padded_dev: None,
            exp_probs_b,
            experts,
            ffn_post_conv,
        });
    }

    // Loop layers (cross-pass message queries & 4 controllers per core layer).
    let mut loop_layers = Vec::new();
    if let Some(lc) = &cfg.loop_cfg {
        let q_all = read_st_f32(
            &index,
            &p("middle_cycle_cross_pass_queries", "cross_pass_messages.queries"),
        )?;
        let kv_all = read_st_f32(
            &index,
            &p("middle_cycle_cross_pass_kv_queries", "cross_pass_messages.kv_queries"),
        )?;
        let core_len = lc.core_end - lc.core_start;
        if q_all.len() != 2 * core_len * cfg.hidden || kv_all.len() != q_all.len() {
            return Err(cfg_err("inkling2_loop: cross-pass queries shape mismatch"));
        }
        let ctrl_names = [("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")];
        for idx_c in 0..core_len {
            let ip = lc.core_start + idx_c;
            // Stack `[q[2*i], kv[2*i], kv[2*i+1], q[2*i+1]]` = `[Q, K, V, FFN]`.
            let h = cfg.hidden;
            let mut msg_query = Vec::with_capacity(4 * h);
            msg_query.extend_from_slice(&q_all[(2 * idx_c) * h..(2 * idx_c + 1) * h]);
            msg_query.extend_from_slice(&kv_all[(2 * idx_c) * h..(2 * idx_c + 1) * h]);
            msg_query.extend_from_slice(&kv_all[(2 * idx_c + 1) * h..(2 * idx_c + 2) * h]);
            msg_query.extend_from_slice(&q_all[(2 * idx_c + 1) * h..(2 * idx_c + 2) * h]);

            let mut ctrl_down = Vec::with_capacity(4);
            let mut ctrl_gain = Vec::with_capacity(4);
            let mut ctrl_forget = Vec::with_capacity(4);
            for &(native_c, hf_c) in &ctrl_names {
                let down = read_st_weight(
                    &index,
                    &p(
                        &format!("layers.{ip}.controllers.{native_c}.down_weight"),
                        &format!("layers.{ip}.controllers.{hf_c}.down_proj.weight"),
                    ),
                    backend.as_ref(),
                )?;
                let gain = read_st_weight(
                    &index,
                    &p(
                        &format!("layers.{ip}.controllers.{native_c}.gain_weight"),
                        &format!("layers.{ip}.controllers.{hf_c}.gain_proj.weight"),
                    ),
                    backend.as_ref(),
                )?;
                let forget = read_st_weight(
                    &index,
                    &p(
                        &format!("layers.{ip}.controllers.{native_c}.forget_weight"),
                        &format!("layers.{ip}.controllers.{hf_c}.forget_proj.weight"),
                    ),
                    backend.as_ref(),
                )?;
                ctrl_down.push(down);
                ctrl_gain.push(gain);
                ctrl_forget.push(forget);
            }
            loop_layers.push(Inkling2LoopLayer {
                msg_query,
                msg_query_dev: None,
                ctrl_down: ctrl_down.try_into().expect("4 controllers"),
                ctrl_gain: ctrl_gain.try_into().expect("4 controllers"),
                ctrl_forget: ctrl_forget.try_into().expect("4 controllers"),
                ctrl_down_qkv_f16: None,
                ctrl_down_ffn_f16: None,
                ctrl_gf_qkv_f16: None,
                ctrl_gf_ffn_f16: None,
            });
        }
    }

    let vision = if load_vision {
        if let Some(vc) = vision_cfg {
            if index.get("model.visual.backbone.ln_pre.weight").is_some()
                || index.get("model.vision_tower.pre_layernorm.weight").is_some()
            {
                Some(Inkling2Vision::from_safetensors(&index, vc, Arc::clone(&backend))?)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let model_config = cfg.to_model_config();
    let state = cfg.new_state();
    let model = Inkling2Model {
        config: model_config,
        cfg,
        tokenizer,
        backend,
        embed_table,
        embed_norm,
        output_norm,
        output_norm_dev: None,
        output_res_query,
        output_res_query_dev: None,
        output,
        layers,
        loop_layers,
        full_gpu_decode: false,
        state: Mutex::new(state),
        gpu_scratch: Mutex::new(None),
    };
    Ok((model, vision))
}

/// Flip time axis (`[K, C]` newest-first) and transpose to `[C, K]` oldest-first.
fn flip_transpose_static_w(raw: &[f32], k: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * k];
    for ch in 0..c {
        for j in 0..k {
            // Output index `j` is oldest (`j=0` -> lag `k-1`) to newest (`j=k-1` -> lag 0).
            // Source `raw[lag * c + ch]` has lag 0 at index 0.
            let lag = k - 1 - j;
            out[ch * k + j] = raw[lag * c + ch];
        }
    }
    out
}

fn load_st_experts(
    index: &StIndex,
    shards: &[Arc<File>],
    cache: &Arc<PackedExpertCache>,
    prefix: &str,
    layer_idx: usize,
    cfg: &Inkling2Config,
) -> Result<Inkling2Experts> {
    let w13_name = format!("{prefix}layers.{layer_idx}.mlp.experts.w13_weight");
    let w2_name = format!("{prefix}layers.{layer_idx}.mlp.experts.w2_weight");

    if let Some(w13_info) = index.get(&w13_name) {
        let w2_info = index
            .get(&w2_name)
            .ok_or_else(|| LlamaError::MissingTensor(w2_name.clone()))?;
        match w13_info.dtype {
            Dtype::U8 => {
                // NVFP4
                let w13_s = index
                    .get(&format!("{w13_name}_scale"))
                    .ok_or_else(|| LlamaError::MissingTensor(format!("{w13_name}_scale")))?;
                let w2_s = index
                    .get(&format!("{w2_name}_scale"))
                    .ok_or_else(|| LlamaError::MissingTensor(format!("{w2_name}_scale")))?;
                if w13_info.shard != w13_s.shard
                    || w13_info.shard != w2_info.shard
                    || w13_info.shard != w2_s.shard
                {
                    return Err(cfg_err(format!(
                        "layer {layer_idx} NVFP4 expert tensors split across different shards"
                    )));
                }
                let w13_amax = read_st_f32(index, &format!("{w13_name}_amax"))?;
                let w13_denom = read_st_f32(index, &format!("{w13_name}_outer_scale_denominator"))?
                    .first()
                    .copied()
                    .filter(|d| d.is_finite() && *d > 0.0)
                    .ok_or_else(|| cfg_err(format!("layer {layer_idx} invalid {w13_name}_outer_scale_denominator")))?;
                let w2_amax = read_st_f32(index, &format!("{w2_name}_amax"))?;
                let w2_denom = read_st_f32(index, &format!("{w2_name}_outer_scale_denominator"))?
                    .first()
                    .copied()
                    .filter(|d| d.is_finite() && *d > 0.0)
                    .ok_or_else(|| cfg_err(format!("layer {layer_idx} invalid {w2_name}_outer_scale_denominator")))?;
                if w13_amax.len() != cfg.n_experts || w2_amax.len() != cfg.n_experts {
                    return Err(cfg_err(format!(
                        "layer {layer_idx} NVFP4 amax length mismatch ({}, {} vs {})",
                        w13_amax.len(),
                        w2_amax.len(),
                        cfg.n_experts
                    )));
                }
                return Ok(Inkling2Experts::SafetensorsNvfp4 {
                    spec: Nvfp4LayerSpec {
                        layer_idx,
                        file: Arc::clone(&shards[w13_info.shard]),
                        cache: Arc::clone(cache),
                        w13_offset: w13_info.start,
                        w13_scale_offset: w13_s.start,
                        w13_amax,
                        w13_denom,
                        w2_offset: w2_info.start,
                        w2_scale_offset: w2_s.start,
                        w2_amax,
                        w2_denom,
                    },
                    n_experts: cfg.n_experts,
                    hidden: cfg.hidden,
                    ff: cfg.ff_dim,
                });
            }
            Dtype::F8E4M3 => {
                // MXFP8
                let w13_s = index
                    .get(&format!("{w13_name}_scale"))
                    .ok_or_else(|| LlamaError::MissingTensor(format!("{w13_name}_scale")))?;
                let w2_s = index
                    .get(&format!("{w2_name}_scale"))
                    .ok_or_else(|| LlamaError::MissingTensor(format!("{w2_name}_scale")))?;
                if w13_info.shard != w13_s.shard
                    || w13_info.shard != w2_info.shard
                    || w13_info.shard != w2_s.shard
                {
                    return Err(cfg_err(format!(
                        "layer {layer_idx} MXFP8 expert tensors split across different shards"
                    )));
                }
                return Ok(Inkling2Experts::SafetensorsMxfp8 {
                    spec: Mxfp8LayerSpec {
                        layer_idx,
                        file: Arc::clone(&shards[w13_info.shard]),
                        cache: Arc::clone(cache),
                        w13_offset: w13_info.start,
                        w13_scale_offset: w13_s.start,
                        w2_offset: w2_info.start,
                        w2_scale_offset: w2_s.start,
                    },
                    n_experts: cfg.n_experts,
                    hidden: cfg.hidden,
                    ff: cfg.ff_dim,
                });
            }
            _ => {
                // Unquantized interleaved BF16/F16/F32 `w13_weight` -> de-interleave to `[E, 2*ff, hidden]`.
                let raw_gu = read_st_f32(index, &w13_name)?;
                let down = read_st_f32(index, &w2_name)?;
                let (e_cnt, ff, h) = (cfg.n_experts, cfg.ff_dim, cfg.hidden);
                if raw_gu.len() != e_cnt * 2 * ff * h || down.len() != e_cnt * h * ff {
                    return Err(cfg_err(format!(
                        "layer {layer_idx} dense expert shape mismatch: w13={} w2={}",
                        raw_gu.len(),
                        down.len()
                    )));
                }
                let mut gate_up = vec![0.0f32; raw_gu.len()];
                for e in 0..e_cnt {
                    for r in 0..ff {
                        let dst_g = (e * 2 * ff + r) * h;
                        let dst_u = (e * 2 * ff + ff + r) * h;
                        let src_g = (e * 2 * ff + 2 * r) * h;
                        let src_u = (e * 2 * ff + 2 * r + 1) * h;
                        gate_up[dst_g..dst_g + h].copy_from_slice(&raw_gu[src_g..src_g + h]);
                        gate_up[dst_u..dst_u + h].copy_from_slice(&raw_gu[src_u..src_u + h]);
                    }
                }
                return Ok(Inkling2Experts::DenseF32 {
                    gate_up,
                    down,
                    n_experts: e_cnt,
                    hidden: h,
                    ff,
                });
            }
        }
    }

    // HF transformers `gate_up_proj` / `down_proj` (non-interleaved).
    let gu_name = format!("{prefix}layers.{layer_idx}.mlp.experts.gate_up_proj");
    let dn_name = format!("{prefix}layers.{layer_idx}.mlp.experts.down_proj");
    let gate_up = read_st_f32(index, &gu_name)?;
    let down = read_st_f32(index, &dn_name)?;
    Ok(Inkling2Experts::DenseF32 {
        gate_up,
        down,
        n_experts: cfg.n_experts,
        hidden: cfg.hidden,
        ff: cfg.ff_dim,
    })
}

/// Helper used by `Inkling2Model::from_safetensors` to avoid circular visibility.
#[allow(dead_code)]
fn _state_type_check(_: Inkling2State) {}
