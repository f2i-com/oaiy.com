//! Original Qwen3.8 vision tensors, read directly from the published shard.
use dsv41::safetensors::StIndex;
use ggml_rs_cuda::CudaBackend;
use ggml_rs::{Backend, Tensor};
use llama_rs::{
    loader::Weight,
    mmproj::{Qwen3VlBlock, Qwen3VlMmProj, Qwen3VlProjector},
    MmProj, MmProjConfig, ProjectorKind,
};
use oaiy_engine::{json::Json, Error, Result};
use std::{path::Path, sync::Arc};

fn bad(s: impl Into<String>) -> Error {
    Error::Format(s.into())
}
fn json(path: &Path) -> Result<Json> {
    Json::parse(&std::fs::read(path)?)
}

// Deliberately accept the matching architecture, not merely the same output
// width: another model's vision tower need not share the text embedding space.
fn config(root: &Path, width: usize) -> Result<MmProjConfig> {
    let c = json(&root.join("config.json"))?;
    let p = json(&root.join("preprocessor_config.json"))?;
    parse_config(&c, &p, width)
}
fn parse_config(c: &Json, p: &Json, width: usize) -> Result<MmProjConfig> {
    let v = c
        .get("vision_config")
        .ok_or_else(|| bad("missing Qwen vision_config"))?;
    for (key, expected) in [
        ("depth", 27),
        ("hidden_size", 1152),
        ("intermediate_size", 4304),
        ("num_heads", 16),
        ("num_position_embeddings", 2304),
        ("patch_size", 16),
        ("spatial_merge_size", 2),
        ("temporal_patch_size", 2),
        ("in_channels", 3),
    ] {
        if v.get(key).and_then(Json::as_i64) != Some(expected) {
            return Err(bad(format!("unsupported Qwen vision {key}")));
        }
    }
    if v.get("out_hidden_size").and_then(Json::as_i64) != Some(width as i64) {
        return Err(bad("unsupported Qwen vision out_hidden_size"));
    }
    // Qwen3.8-27B's tower, and Qwen3.8-Flash-Next's (the same, into a 2560-wide model).
    if !matches!((width, c.get("model_type").and_then(Json::as_str)), (5120, Some("qwen3_5")) | (2560, Some("qwen4_exp")))
        || v.get("hidden_act").and_then(Json::as_str) != Some("gelu_pytorch_tanh")
        || v.get("deepstack_visual_indexes")
            .and_then(Json::as_array)
            .is_none_or(|a| !a.is_empty())
    {
        return Err(bad(
            "vision tower is not the supported Qwen3.8-27B architecture",
        ));
    }
    for (key, expected) in [
        ("patch_size", 16),
        ("temporal_patch_size", 2),
        ("merge_size", 2),
    ] {
        if p.get(key).and_then(Json::as_i64) != Some(expected) {
            return Err(bad(format!("unsupported preprocessing {key}")));
        }
    }
    for key in ["image_mean", "image_std"] {
        if p.get(key)
            .and_then(Json::as_array)
            .is_none_or(|a| a.len() != 3 || a.iter().any(|x| x.as_f64() != Some(0.5)))
        {
            return Err(bad(format!("unsupported Qwen {key}")));
        }
    }
    Ok(MmProjConfig {
        image_size: 768,
        patch_size: 16,
        embedding_dim: 1152,
        n_layers: 27,
        n_heads: 16,
        head_dim: 72,
        ff_dim: 4304,
        layer_norm_eps: 1e-6,
        projector: ProjectorKind::Qwen3Vl,
        mean: [0.5; 3],
        std: [0.5; 3],
    })
}

fn tensor(idx: &StIndex, b: &dyn Backend, name: &str, shape: &[usize]) -> Result<Tensor> {
    let name = format!("model.visual.{name}");
    if idx.info(&name)?.shape != shape {
        return Err(bad(format!("{name}: incorrect vision tensor shape")));
    }
    let values = idx.read_f32(&name)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(bad(format!("{name}: non-finite vision weights")));
    }
    Ok(b.to_device(Tensor::from_vec(values, shape.to_vec())))
}

/// Fold the two temporal slots in memory for a still image duplicated in time.
/// Source layout is [output, RGB, time, patch_y, patch_x], not time-major.
fn static_patch(values: &[f32], d: usize, p: usize) -> Vec<f32> {
    let area = p * p;
    let mut out = vec![0.0; d * 3 * area];
    for oc in 0..d {
        for c in 0..3 {
            for i in 0..area {
                let at = (oc * 3 + c) * 2 * area + i;
                out[(oc * 3 + c) * area + i] = values[at] + values[at + area];
            }
        }
    }
    out
}

/// `exl3`: where EXL3 matrices go, for a tower stored quantized (Qwen3.8-Flash-Next's):
/// a matrix without plain weights is read from its EXL3 tiles.
pub fn load(root: &Path, backend: Arc<dyn Backend>, width: usize, exl3: Option<Arc<CudaBackend>>) -> Result<MmProj> {
    let mut cfg = config(root, width)?;
    let idx = StIndex::open(root)?;
    // A quantized tower stores its MLP padded to whole EXL3 tiles (4304 -> 4352, the padding
    // cancelling out): build it at the width stored.
    if let Some(info) = idx.get("model.visual.blocks.0.mlp.linear_fc1.bias") {
        cfg.ff_dim = info.shape[0];
    }
    let b = backend.as_ref();
    let t = |name: &str, shape: &[usize]| tensor(&idx, b, name, shape);
    let w = |name: &str, shape: &[usize]| -> Result<Weight> {
        match (&exl3, name.strip_suffix(".weight")) {
            (Some(cuda), Some(base)) if idx.get(&format!("model.visual.{name}")).is_none() => {
                crate::flashnext::exl3_weight(&idx, cuda, &format!("model.visual.{base}"), shape[1], shape[0], None, None)
            }
            _ => t(name, shape).map(Weight::Dense),
        }
    };
    let (d, ff) = (cfg.embedding_dim, cfg.ff_dim);
    let patch_name = "model.visual.patch_embed.proj.weight";
    if idx.info(patch_name)?.shape != [d, 3, 2, 16, 16] {
        return Err(bad("invalid temporal patch weights"));
    }
    let patch = idx.read_f32(patch_name)?;
    if patch.iter().any(|x| !x.is_finite()) {
        return Err(bad("non-finite patch weights"));
    }
    let patch_embd = Tensor::from_vec(static_patch(&patch, d, 16), vec![d, 3, 16, 16]);
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let name = |suffix: &str| format!("blocks.{i}.{suffix}");
        blocks.push(Qwen3VlBlock {
            ln1_w: t(&name("norm1.weight"), &[d])?,
            ln1_b: t(&name("norm1.bias"), &[d])?,
            attn_qkv: w(&name("attn.qkv.weight"), &[3 * d, d])?,
            attn_qkv_b: t(&name("attn.qkv.bias"), &[3 * d])?,
            attn_out: w(&name("attn.proj.weight"), &[d, d])?,
            attn_out_b: t(&name("attn.proj.bias"), &[d])?,
            ln2_w: t(&name("norm2.weight"), &[d])?,
            ln2_b: t(&name("norm2.bias"), &[d])?,
            ffn_up: w(&name("mlp.linear_fc1.weight"), &[ff, d])?,
            ffn_up_b: t(&name("mlp.linear_fc1.bias"), &[ff])?,
            ffn_down: w(&name("mlp.linear_fc2.weight"), &[d, ff])?,
            ffn_down_b: t(&name("mlp.linear_fc2.bias"), &[d])?,
        });
    }
    let projector = Qwen3VlProjector {
        mm0: w("merger.linear_fc1.weight", &[4 * d, 4 * d])?,
        mm0_b: t("merger.linear_fc1.bias", &[4 * d])?,
        mm2: w("merger.linear_fc2.weight", &[width, 4 * d])?,
        mm2_b: t("merger.linear_fc2.bias", &[width])?,
    };
    Ok(MmProj::Qwen3Vl(Qwen3VlMmProj {
        patch_embd,
        patch_embd_b: t("patch_embed.proj.bias", &[d])?,
        position_embd: t("pos_embed.weight", &[2304, d])?,
        blocks,
        post_ln_w: t("merger.norm.weight", &[d])?,
        post_ln_b: t("merger.norm.bias", &[d])?,
        projector,
        config: cfg,
        backend,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_incompatible_tower_and_preprocessing() {
        let c=Json::parse(br#"{"model_type":"qwen3_5","vision_config":{"depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"in_channels":3,"out_hidden_size":5120,"hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]}}"#).unwrap();
        let p=Json::parse(br#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#).unwrap();
        assert_eq!(parse_config(&c, &p, 5120).unwrap().n_soft_tokens(), 576);
        assert!(parse_config(&c, &p, 4096).is_err());
        let wrong = Json::parse(
            c.to_json()
                .replace("\"out_hidden_size\":5120", "\"out_hidden_size\":4096")
                .as_bytes(),
        )
        .unwrap();
        assert!(parse_config(&wrong, &p, 5120).is_err());
        let wrong = Json::parse(
            p.to_json()
                .replace("\"merge_size\":2", "\"merge_size\":3")
                .as_bytes(),
        )
        .unwrap();
        assert!(parse_config(&c, &wrong, 5120).is_err());
        assert!(parse_config(&c, &Json::Null, 5120).is_err());
    }
    #[test]
    fn temporal_patch_fold_preserves_rgb_and_output_axes() {
        let src: Vec<_> = (0..2 * 3 * 2 * 4).map(|i| i as f32).collect();
        let got = static_patch(&src, 2, 2);
        assert_eq!(&got[..8], &[4., 6., 8., 10., 20., 22., 24., 26.]);
        assert_eq!(&got[12..16], &[52., 54., 56., 58.]);
    }
    #[test]
    #[ignore = "requires downloaded original Qwen vision and generated HF reference"]
    fn original_vision_matches_huggingface() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let b: Arc<dyn Backend> = Arc::new(ggml_rs_cuda::CudaBackend::new(1).unwrap());
        let mm = load(&root.join("models/OrcaSAQ-2-27B/vision"), b, 5120, None).unwrap();
        let oracle =
            StIndex::open_file(&root.join("target/orca-vision-research/oracle.safetensors"))
                .unwrap();
        let input = Tensor::from_vec(oracle.read_f32("pixels").unwrap(), vec![3, 768, 768]);
        let clock=std::time::Instant::now();
        let out = mm.forward(&input).unwrap().to_host();
        eprintln!("vision encoding first: {:.3}s",clock.elapsed().as_secs_f64());
        for run in 1..=3 {
            let clock=std::time::Instant::now();
            let result=mm.forward(&input).unwrap().to_host();
            std::hint::black_box(result);
            eprintln!("vision encoding warm {run}: {:.3}s",clock.elapsed().as_secs_f64());
        }
        let expected = oracle.read_f32("embeddings").unwrap();
        assert_eq!(out.shape(), [576, 5120]);
        let max = out
            .data()
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let rms = (out
            .data()
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).powi(2) as f64)
            .sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        eprintln!("vision HF reference: max={max}, rms={rms}");
        assert!(rms < 0.0001 && max < 0.02);
    }
}
