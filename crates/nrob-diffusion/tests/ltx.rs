use candle_core::{DType, Device, Tensor};
use nrob::json::Json;
use nrob_diffusion::ltx::{
    vae::{causal_conv3d_forward, LtxVaeConfig, LtxVideoDecoder, PadMode},
    Request,
};

fn request(extra: &str) -> Json {
    Json::parse(format!(r#"{{"model":"ltx-2.3","transformer":"model.safetensors","text_encoder":"gemma.safetensors","tokenizer":"tokenizer.json","vae":"vae.safetensors","output_dir":"videos","prompt":"A bird flies over a lake"{extra}}}"#).as_bytes()).unwrap()
}
#[test]
fn rejects_invalid_video_geometry_and_unsupported_modes() {
    assert!(Request::parse(&request("")).is_ok());
    for extra in [
        r#", "frames":48"#,
        r#", "width":129"#,
        r#", "steps":6"#,
        r#", "memory":"unbounded""#,
        r#", "fps":0"#,
        r#", "images":["x.png"]"#,
    ] {
        assert!(Request::parse(&request(extra)).is_err(), "{extra}");
    }
}
#[test]
fn conv3d_causal_and_symmetric_boundary_values() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::from_vec(vec![1f32, 2., 4.], (1, 1, 3, 1, 1), &dev)?;
    let w = Tensor::ones((1, 1, 3, 1, 1), DType::F32, &dev)?;
    let causal = causal_conv3d_forward(&x, &w, None, true, PadMode::Zero)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let symmetric = causal_conv3d_forward(&x, &w, None, false, PadMode::Zero)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(causal, vec![3., 4., 7.]);
    assert_eq!(symmetric, vec![4., 7., 10.]);
    Ok(())
}
#[test]
#[ignore = "requires LTX VAE weights and official reference activations; NROB_LTX_GOLDEN"]
fn video_decoder_matches_official_reference() -> candle_core::Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
    );
    let path = std::env::var("NROB_LTX_VAE").map_err(candle_core::Error::wrap)?;
    let dev = Device::new_cuda(
        std::env::var("NROB_LTX_TEST_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )?;
    let model = LtxVideoDecoder::load(
        std::path::Path::new(&path),
        LtxVaeConfig::ltx_2_3_22b(),
        &dev,
        DType::BF16,
    )?;
    let input = Tensor::from_raw_buffer(
        &std::fs::read(root.join("vae-input.f32"))?,
        DType::F32,
        &[1, 128, 2, 4, 4],
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let expected = Tensor::from_raw_buffer(
        &std::fs::read(root.join("vae-output.f32"))?,
        DType::F32,
        &[1, 3, 9, 128, 128],
        &dev,
    )?;
    let actual = model.decode(&input)?.to_dtype(DType::F32)?;
    let error = (actual - &expected)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?
        .sqrt();
    let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    println!("VAE relative RMS error: {}", error / scale);
    assert!(
        error / scale < 0.04,
        "VAE differs from official decoder: {}",
        error / scale
    );
    Ok(())
}
#[test]
#[ignore = "requires LTX VAE weights and official reference activations; NROB_LTX_GOLDEN"]
fn starting_image_encoder_matches_official_reference() -> candle_core::Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var("NROB_LTX_GOLDEN").map_err(candle_core::Error::wrap)?,
    );
    let path = std::env::var("NROB_LTX_CHECKPOINT").map_err(candle_core::Error::wrap)?;
    let dev = Device::new_cuda(
        std::env::var("NROB_LTX_TEST_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )?;
    let model = nrob_diffusion::ltx::vae::LtxVideoEncoder::load(
        std::path::Path::new(&path),
        LtxVaeConfig::ltx_2_3_22b(),
        &dev,
        DType::BF16,
    )?;
    let input = Tensor::from_raw_buffer(
        &std::fs::read(root.join("encoder-input.f32"))?,
        DType::F32,
        &[1, 3, 1, 128, 128],
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let expected = Tensor::from_raw_buffer(
        &std::fs::read(root.join("encoder-output.f32"))?,
        DType::F32,
        &[1, 128, 1, 4, 4],
        &dev,
    )?;
    let actual = model.encode_means(&input)?.to_dtype(DType::F32)?;
    let error = (actual - &expected)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?
        .sqrt();
    let scale = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    println!("VAE relative RMS error: {}", error / scale);
    assert!(
        error / scale < 0.04,
        "VAE differs from official encoder: {}",
        error / scale
    );
    Ok(())
}
#[test]
#[ignore = "requires a real LTX convolutional VAE checkpoint and CUDA; set NROB_LTX_VAE"]
fn real_video_decoder_geometry_and_finite_pixels() -> candle_core::Result<()> {
    let path = std::env::var("NROB_LTX_VAE").map_err(candle_core::Error::wrap)?;
    let dev = Device::new_cuda(
        std::env::var("NROB_LTX_TEST_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )?;
    let decoder = LtxVideoDecoder::load(
        std::path::Path::new(&path),
        LtxVaeConfig::ltx_2_3_22b(),
        &dev,
        DType::BF16,
    )?;
    let z = Tensor::zeros((1, 128, 2, 4, 4), DType::BF16, &dev)?;
    let pixels = decoder.decode(&z)?;
    assert_eq!(pixels.dims(), &[1, 3, 9, 128, 128]);
    assert!(pixels
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|v| v.is_finite()));
    let wide = Tensor::zeros((1, 128, 2, 4, 6), DType::BF16, &dev)?;
    let tiled = decoder.decode_tiled(&wide, 4, 4, 1, 1)?;
    assert_eq!(tiled.dims(), &[1, 3, 9, 128, 192]);
    assert!(tiled.device().is_cpu());
    assert!(tiled
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|v| v.is_finite()));
    let normal = Tensor::zeros((1, 128, 7, 10, 16), DType::BF16, &dev)?;
    let normal = decoder.decode(&normal)?;
    assert_eq!(normal.dims(), &[1, 3, 49, 320, 512]);
    assert!(normal
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|v| v.is_finite()));
    Ok(())
}
