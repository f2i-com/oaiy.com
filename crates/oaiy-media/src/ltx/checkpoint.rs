//! Whether a checkpoint is one this pipeline runs.

use super::*;

/// `store`'s checkpoint against what `r` asks of it, an error saying what does not fit: its version for the model,
/// its transformer's configuration (the audio stream's too for a clip with sound), and the shapes of the tensors
/// that tell its architecture.
pub(super) fn check(r: &Request, store: &Store) -> Result<()> {
    let expected_version = if r.model == "ltx-2.5" { "2.5." } else { "2.3." };
    if !store
        .index
        .metadata("model_version")
        .is_some_and(|v| v.starts_with(expected_version))
    {
        candle_core::bail!("{} requires an LTX {expected_version} checkpoint", r.model);
    }
    let config = Json::parse(
        store
            .index
            .metadata("config")
            .ok_or_else(|| {
                candle_core::Error::Msg("LTX checkpoint lacks architecture metadata".into())
            })?
            .as_bytes(),
    )
    .map_err(candle_core::Error::wrap)?;
    let architecture = config.get("transformer").ok_or_else(|| {
        candle_core::Error::Msg("LTX checkpoint lacks transformer configuration".into())
    })?;
    for (key, expected) in [
        ("num_layers", 48),
        ("num_attention_heads", 32),
        ("attention_head_dim", 128),
        ("connector_num_layers", 8),
        ("connector_num_attention_heads", 32),
        ("connector_attention_head_dim", 128),
    ] {
        if architecture.get(key).and_then(Json::as_i64) != Some(expected) {
            candle_core::bail!("unsupported LTX configuration: {key}");
        }
    }
    for key in [
        "cross_attention_adaln",
        "apply_gated_attention",
        "use_middle_indices_grid",
    ] {
        if architecture.get(key).and_then(Json::as_bool) != Some(true) {
            candle_core::bail!("unsupported LTX configuration: {key}");
        }
    }
    for (key, expected) in [("rope_type", "split"), ("frequencies_precision", "float64")] {
        if architecture.get(key).and_then(Json::as_str) != Some(expected) {
            candle_core::bail!("unsupported LTX configuration: {key}");
        }
    }
    if r.audio {
        for (key, expected) in [
            ("audio_num_attention_heads", 32),
            ("audio_attention_head_dim", 64),
            ("audio_cross_attention_dim", 2048),
            ("audio_connector_num_attention_heads", 32),
            ("audio_connector_attention_head_dim", 64),
        ] {
            if architecture.get(key).and_then(Json::as_i64) != Some(expected) {
                candle_core::bail!("unsupported LTX audio configuration: {key}");
            }
        }
        // Both timestep multipliers are 1000 in every LTX 2.x release; the
        // cross-attention gates depend on their ratio.
        for key in ["timestep_scale_multiplier", "av_ca_timestep_scale_multiplier"] {
            if architecture.get(key).and_then(Json::as_f64).is_some_and(|v| v != 1000.) {
                candle_core::bail!("unsupported LTX audio configuration: {key}");
            }
        }
    }
    let audio_shapes: &[(&str, Vec<usize>)] = if r.audio {
        &[
            ("audio_patchify_proj.weight", vec![2048, 128]),
            ("transformer_blocks.47.audio_scale_shift_table", vec![9, 2048]),
            ("transformer_blocks.47.scale_shift_table_a2v_ca_video", vec![5, 4096]),
        ]
    } else {
        &[]
    };
    for (key, shape) in [
        ("patchify_proj.weight", vec![4096, 128]),
        ("transformer_blocks.47.scale_shift_table", vec![9, 4096]),
    ]
    .iter()
    .chain(audio_shapes)
    {
        let name = format!("{}{key}", transformer::PREFIX);
        let info = store.index.info(&name).map_err(candle_core::Error::wrap)?;
        if &info.shape != shape {
            candle_core::bail!("unsupported LTX architecture at {name}: {:?}", info.shape);
        }
    }
    Ok(())
}
