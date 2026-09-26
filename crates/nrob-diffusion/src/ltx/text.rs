//! Gemma 3 12B and LTX's Gemma 4 Unified 12B text features.
use super::{
    store::{Group, Store},
    transformer::norm,
};
use crate::math::{heads, prefix_attention, rms, unheads};
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

fn gemma_norm(x: &Tensor, w: &Tensor, gemma4: bool) -> Result<Tensor> {
    if gemma4 {
        rms(x, w, 1e-6)
    } else {
        // Gemma applies the learned (1 + weight) in F32 before casting back.
        rms(x, &(w.to_dtype(DType::F32)? + 1.)?, 1e-6)
    }
}
fn rotary(x: &Tensor, global: bool, gemma4: bool, offset: usize) -> Result<Tensor> {
    let (_, _, n, d) = x.dims4()?;
    let base: f64 = if global { 1e6 } else { 10000. };
    let factor = if global && !gemma4 { 8. } else { 1. };
    let rotated = if global && gemma4 { d / 8 } else { d / 2 };
    let mut cos = Vec::with_capacity(n * d / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for t in 0..n {
        for j in 0..d / 2 {
            let p = if j < rotated {
                (t + offset) as f64 / (base.powf(2. * j as f64 / d as f64) * factor)
            } else {
                0.
            } as f32;
            cos.push(p.cos());
            sin.push(p.sin());
        }
    }
    let c = Tensor::from_vec(cos, (1, 1, n, d / 2), x.device())?.to_dtype(x.dtype())?;
    let s = Tensor::from_vec(sin, (1, 1, n, d / 2), x.device())?.to_dtype(x.dtype())?;
    let a = x.narrow(3, 0, d / 2)?;
    let b = x.narrow(3, d / 2, d / 2)?;
    Tensor::cat(
        &[
            (a.broadcast_mul(&c)? - b.broadcast_mul(&s)?)?,
            (b.broadcast_mul(&c)? + a.broadcast_mul(&s)?)?,
        ],
        3,
    )
}
fn repeat_kv(x: &Tensor) -> Result<Tensor> {
    let (b, h, n, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, h, 16 / h, n, d))?
        .reshape((b, 16, n, d))
}
// Gemma 4 uses scale=1. Passing it directly avoids rounding a rescaled Q.
fn gemma4_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (_, _, n, d) = q.dims4()?;
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() && d <= 256 {
        return candle_flash_attn::flash_attn(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            1.,
            true,
        )?
        .transpose(1, 2);
    }
    let _ = d;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;
    let mut chunks = Vec::new();
    for start in (0..n).step_by(128) {
        let len = 128.min(n - start);
        let scores = q
            .narrow(2, start, len)?
            .contiguous()?
            .matmul(&kt)?
            .to_dtype(DType::F32)?;
        let mask: Vec<_> = (start..start + len)
            .flat_map(|i| (0..n).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, 1, len, n), q.device())?)?;
        chunks.push(
            candle_nn::ops::softmax_last_dim(&scores)?
                .to_dtype(q.dtype())?
                .matmul(&v)?,
        );
    }
    Tensor::cat(&chunks, 2)
}
fn block(w: &Group, x: &Tensor, i: usize, gemma4: bool, offset: usize) -> Result<Tensor> {
    let global = i % 6 == 5;
    let kv = if gemma4 && global { 1 } else { 8 };
    let h = gemma_norm(x, w.get("input_layernorm.weight")?, gemma4)?;
    let q = heads(&w.linear("self_attn.q_proj", &h)?, 16)?;
    let k = heads(&w.linear("self_attn.k_proj", &h)?, kv)?;
    let v = if gemma4 && global {
        k.clone()
    } else {
        heads(&w.linear("self_attn.v_proj", &h)?, kv)?
    };
    let q = rotary(
        &gemma_norm(&q, w.get("self_attn.q_norm.weight")?, gemma4)?,
        global,
        gemma4,
        offset,
    )?;
    let k = rotary(
        &gemma_norm(&k, w.get("self_attn.k_norm.weight")?, gemma4)?,
        global,
        gemma4,
        offset,
    )?;
    let v = if gemma4 { norm(&v)? } else { v };
    let k = repeat_kv(&k)?;
    let v = repeat_kv(&v)?;
    let a = unheads(&if gemma4 {
        gemma4_attention(&q, &k, &v)?
    } else {
        prefix_attention(&q, &k, &v, true)?
    })?;
    let a = w.linear("self_attn.o_proj", &a)?;
    let x = (x + gemma_norm(&a, w.get("post_attention_layernorm.weight")?, gemma4)?)?;
    let h = gemma_norm(&x, w.get("pre_feedforward_layernorm.weight")?, gemma4)?;
    let gate = w.linear("mlp.gate_proj", &h)?;
    let h = (gate.to_dtype(DType::F32)?.gelu()?.to_dtype(gate.dtype())?
        * w.linear("mlp.up_proj", &h)?)?;
    let h = w.linear("mlp.down_proj", &h)?;
    let x = (x + gemma_norm(&h, w.get("post_feedforward_layernorm.weight")?, gemma4)?)?;
    if gemma4 {
        x.broadcast_mul(w.get("layer_scalar")?)
    } else {
        Ok(x)
    }
}
fn feature_norm(x: &Tensor) -> Result<Tensor> {
    // The published V2 extractor squares and reduces in the activation dtype.
    let variance = (x.sqr()?.mean_keepdim(candle_core::D::Minus1)? + 1e-6)?;
    let inverse = variance
        .to_dtype(DType::F32)?
        .sqrt()?
        .recip()?
        .to_dtype(x.dtype())?;
    x.broadcast_mul(&inverse)
}
fn hidden_features(
    store: &mut Store,
    ids: &[u32],
    gemma4: bool,
    dev: &Device,
    mut progress: impl FnMut(usize),
) -> Result<Tensor> {
    // Masked left-padding need not traverse 48 layers. Preserve its position
    // offset so the BF16 rotary angles match the padded reference encoder.
    let embedding = store.rows("model.embed_tokens.weight", ids, dev)?;
    let mut x = embedding
        .unsqueeze(0)?
        .broadcast_mul(&Tensor::new(3840f32.sqrt(), dev)?.to_dtype(DType::BF16)?)?;
    drop(embedding);
    let mut states = vec![feature_norm(&x)?];
    for i in 0..48 {
        let w = store.group(&format!("model.layers.{i}."), dev, false, |_| true)?;
        x = block(&w, &x, i, gemma4, 1024 - ids.len())?;
        if i == 47 {
            x = gemma_norm(&x, &store.tensor("model.norm.weight", dev, false)?, gemma4)?;
        }
        states.push(feature_norm(&x)?);
        progress(i + 1);
    }
    // Each stream rescales these by its own width (see `encode`).
    Tensor::stack(&states, 3)?.flatten_from(2)
}

/// Prompt features for the video stream, and for the audio stream when
/// `audio` (LTX 2.5: Gemma 4's `audio_aggregate_embed`).
#[allow(clippy::too_many_arguments)]
pub fn encode(
    path: &Path,
    tokenizer: Option<&Path>,
    projection: &mut Store,
    prompt: &str,
    gemma4: bool,
    audio: bool,
    dev: &Device,
    mut progress: impl FnMut(usize),
) -> Result<(Tensor, Option<Tensor>)> {
    let mut store = Store::open(path, 0)?;
    let bytes = if gemma4 {
        store
            .index
            .read("tokenizer_json")
            .map_err(candle_core::Error::wrap)?
    } else {
        std::fs::read(tokenizer.ok_or_else(|| {
            candle_core::Error::Msg("LTX 2.3 requires a Gemma 3 tokenizer path".into())
        })?)?
    };
    let mut tokenizer =
        tokenizers::Tokenizer::from_bytes(bytes).map_err(candle_core::Error::wrap)?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(candle_core::Error::wrap)?;
    let enc = tokenizer
        .encode(prompt.trim(), true)
        .map_err(candle_core::Error::wrap)?;
    let mut ids = enc.get_ids().to_vec();
    if ids.first() != Some(&2) {
        ids.insert(0, 2);
    }
    ids.truncate(1024);
    let normed = hidden_features(&mut store, &ids, gemma4, dev, &mut progress)?;
    let stacked = (&normed * (4096f64 / 3840.).sqrt())?;
    let w = if gemma4 {
        store.group("text_embedding_projection.", dev, false, |k| {
            k.starts_with("video_aggregate_embed.")
        })?
    } else {
        projection.group("text_embedding_projection.", dev, false, |k| {
            k.starts_with("video_aggregate_embed.")
        })?
    };
    let video = w.linear("video_aggregate_embed", &stacked)?;
    let audio = if audio {
        if !gemma4 {
            candle_core::bail!("audio prompt features need the LTX 2.5 Gemma 4 encoder");
        }
        let w = store.group("text_embedding_projection.", dev, false, |k| {
            k.starts_with("audio_aggregate_embed.")
        })?;
        Some(w.linear("audio_aggregate_embed", &(&normed * (2048f64 / 3840.).sqrt())?)?)
    } else {
        None
    };
    Ok((video, audio))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Gemma 4 weights and official reference features; NROB_LTX_GEMMA4"]
    fn gemma4_real_blocks_match_reference() -> Result<()> {
        gemma4_matches_reference(false)
    }
    #[test]
    #[ignore = "requires full official Gemma 4 reference features; NROB_LTX_GEMMA4"]
    fn gemma4_all_hidden_features_match_reference() -> Result<()> {
        gemma4_matches_reference(true)
    }
    fn gemma4_matches_reference(features: bool) -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
        );
        let path = std::env::var("NROB_LTX_GEMMA4").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(
            std::env::var("NROB_LTX_TEST_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )?;
        let mut store = Store::open(Path::new(&path), 0)?;
        let compare = |actual: Tensor, name: &str| -> Result<()> {
            let actual = actual.to_dtype(DType::F32)?;
            let expected = Tensor::from_raw_buffer(
                &std::fs::read(root.join(name))?,
                DType::F32,
                actual.dims(),
                &dev,
            )?;
            let error = (&actual - &expected)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?
                .sqrt();
            let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            println!("Gemma 4 {name} relative RMS error {}", error / scale);
            assert!(
                error / scale < 0.04,
                "Gemma 4 conditioning differs from reference: {}",
                error / scale
            );
            Ok(())
        };
        if features {
            let ids: Vec<u32> = std::iter::once(2).chain(100..115).collect();
            return compare(
                (hidden_features(&mut store, &ids, true, &dev, |_| {})? * (4096f64 / 3840.).sqrt())?,
                "gemma4-features.f32",
            );
        }
        let x = Tensor::from_raw_buffer(
            &std::fs::read(root.join("gemma4-input.f32"))?,
            DType::F32,
            &[1, 16, 3840],
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        for index in [0, 11] {
            let weights = store.group(&format!("model.layers.{index}."), &dev, false, |_| true)?;
            compare(
                block(&weights, &x, index, true, 1008)?,
                &format!("gemma4-{index}-output.f32"),
            )?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires full official Gemma 3 reference features; NROB_LTX_GOLDEN"]
    fn gemma3_all_hidden_features_match_reference() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
        );
        let weights = std::env::var("NROB_LTX_GEMMA").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(
            std::env::var("NROB_LTX_TEST_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )?;
        let mut store = Store::open(Path::new(&weights), 0)?;
        let ids: Vec<u32> = std::iter::once(2).chain(100..115).collect();
        let actual =
            (hidden_features(&mut store, &ids, false, &dev, |_| {})? * (4096f64 / 3840.).sqrt())?.to_dtype(DType::F32)?;
        let expected = Tensor::from_raw_buffer(
            &std::fs::read(root.join("gemma-features.f32"))?,
            DType::F32,
            &[1, 16, 3840 * 49],
            &dev,
        )?;
        let error = (&actual - &expected)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        println!(
            "Gemma complete features relative RMS error: {}",
            error / scale
        );
        assert!(
            error / scale < 0.04,
            "Gemma conditioning differs from reference: {}",
            error / scale
        );
        Ok(())
    }
    #[test]
    fn gemma4_attention_uses_unit_scale_and_causal_mask() -> Result<()> {
        let dev = Device::Cpu;
        let q = Tensor::ones((1, 1, 3, 512), DType::F32, &dev)?;
        let k = Tensor::from_vec(
            [0.001f32, 0.002, 0.003]
                .into_iter()
                .flat_map(|v| std::iter::repeat_n(v, 512))
                .collect(),
            (1, 1, 3, 512),
            &dev,
        )?;
        let v = Tensor::from_vec(
            [1f32, 2., 3.]
                .into_iter()
                .flat_map(|v| std::iter::repeat_n(v, 512))
                .collect(),
            (1, 1, 3, 512),
            &dev,
        )?;
        let y = gemma4_attention(&q, &k, &v)?
            .reshape((3, 512))?
            .to_vec2::<f32>()?;
        assert_eq!(y[0][0], 1.);
        for n in 2..=3 {
            let denominator: f64 = (1..=n).map(|j| (j as f64 * 0.512).exp()).sum();
            let expected: f64 = (1..=n)
                .map(|j| j as f64 * (j as f64 * 0.512).exp() / denominator)
                .sum();
            assert!((y[n - 1][0] as f64 - expected).abs() < 1e-4);
        }
        Ok(())
    }
    #[test]
    fn gemma4_global_rotary_preserves_unrotated_channels() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::cat(
            &[
                Tensor::ones((1, 1, 2, 256), DType::F32, &dev)?,
                Tensor::zeros((1, 1, 2, 256), DType::F32, &dev)?,
            ],
            3,
        )?;
        let y = rotary(&x, true, true, 0)?
            .reshape((2, 512))?
            .to_vec2::<f32>()?;
        assert!((y[1][0] - 1f32.cos()).abs() < 1e-6);
        assert!((y[1][256] - 1f32.sin()).abs() < 1e-6);
        assert!(y[1][64..256].iter().all(|v| *v == 1.));
        assert!(y[1][320..512].iter().all(|v| *v == 0.));
        let w = Tensor::full(2f32, 512, &dev)?;
        let input = Tensor::ones((1, 512), DType::F32, &dev)?;
        assert!((gemma_norm(&input, &w, true)?.to_vec2::<f32>()?[0][0] - 2.).abs() < 1e-5);
        assert!((gemma_norm(&input, &w, false)?.to_vec2::<f32>()?[0][0] - 3.).abs() < 1e-5);
        Ok(())
    }
    #[test]
    #[ignore = "requires local Gemma 3 weights and reference activations; NROB_LTX_GOLDEN"]
    fn gemma3_local_and_global_attention_match_reference() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
        );
        let weights = std::env::var("NROB_LTX_GEMMA").map_err(candle_core::Error::wrap)?;
        let dev = Device::new_cuda(
            std::env::var("NROB_LTX_TEST_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )?;
        let read = |name: &str| -> Result<Tensor> {
            Tensor::from_raw_buffer(
                &std::fs::read(root.join(name))?,
                candle_core::DType::F32,
                &[1, 16, 3840],
                &dev,
            )
        };
        let x = read("gemma-input.f32")?.to_dtype(candle_core::DType::BF16)?;
        let mut store = Store::open(Path::new(&weights), 0)?;
        for i in [0, 5] {
            let w = store.group(&format!("model.layers.{i}."), &dev, false, |_| true)?;
            let actual = block(&w, &x, i, false, 0)?.to_dtype(candle_core::DType::F32)?;
            let expected = read(&format!("gemma-layer-{i}.f32"))?;
            let error = (&actual - &expected)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?
                .sqrt();
            let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            assert!(
                error / scale < 0.025,
                "Gemma layer {i}: relative RMS error {}",
                error / scale
            );
        }
        Ok(())
    }
}
