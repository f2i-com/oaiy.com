//! The parts of a clip made on WebGPU: the transformer's streams, the prompts' contexts, the ends' latents and
//! the decoded clip (each with a stand-in for a build without the `webgpu` feature).

use super::*;

/// The transformer a clip is denoised by: Candle's, or (`backend` "webgpu") the video stream on WebGPU with its
/// tokens' rotary table and, for a clip with sound, the audio stream beside it: the audio tokens' table and the
/// video tokens' for the attention between the streams.
pub(super) enum Model {
    Candle(transformer::Transformer),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::ltx_wgpu::WgpuLtx, ggml_rs::DeviceVec, Option<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>),
}

/// The video stream of `store`'s transformer on WebGPU (and the audio stream where the clip has sound), for a latent
/// of `f` frames of `h` by `w` and `audio_frames` of audio.
#[cfg(feature = "webgpu")]
pub(super) fn webgpu_model(r: &Request, store: &mut Store, f: usize, h: usize, w: usize, audio_frames: usize, report: &mut dyn FnMut(Json)) -> Result<Model> {
    use crate::ltx_wgpu::{rope_table, video_positions};
    report(event("loading_video_model", 0, 48));
    let m = crate::ltx_wgpu::WgpuLtx::load_streams(store, r.device, r.audio, |n| report(event("loading_video_model", n, 48)))?;
    let positions = video_positions(f, h, w, r.fps, r.end_image.is_some());
    let table = m.upload(&rope_table(&positions, &[20., 2048., 2048.], 4096, 32));
    // (each audio frame at its span's middle and each video token at its time, in seconds: 2,048 channels' tables)
    let sound = r.audio.then(|| {
        let middles: Vec<Vec<f32>> = transformer::audio_spans(audio_frames).iter().map(|&(s, e)| vec![((s + e) / 2.) as f32]).collect();
        let times: Vec<Vec<f32>> = positions.iter().map(|p| vec![p[0]]).collect();
        (m.upload(&rope_table(&middles, &[20.], 2048, 32)), m.upload(&rope_table(&times, &[20.], 2048, 32)))
    });
    Ok(Model::Wgpu(m, table, sound))
}
#[cfg(not(feature = "webgpu"))]
pub(super) fn webgpu_model(_: &Request, _: &mut Store, _: usize, _: usize, _: usize, _: usize, _: &mut dyn FnMut(Json)) -> Result<Model> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// The prompt's video context, its audio context (a clip with sound) and the negative prompt's video context (where
/// there is one) on WebGPU: from the prompt cache where it holds them (`cached` the prompt's video context,
/// `cached_audio` its audio one), else Gemma, the projections and the connectors on the GPU, both prompts in one pass
/// over Gemma's layers. On the host (F32 where new).
#[cfg(feature = "webgpu")]
pub(super) fn webgpu_contexts(r: &Request, store: &mut Store, cached: Option<Tensor>, cached_audio: Option<Tensor>, negative_text: Option<String>, report: &mut dyn FnMut(Json)) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    let negative_cache = negative_text.as_ref().and_then(|text| cache::PromptCache::new(&Request { prompt: text.clone(), audio: false, ..r.clone() }));
    let cached_negative = negative_cache.as_ref().and_then(|c| c.load(&Device::Cpu));
    // (the prompt anew where either of its contexts is not cached)
    let cached = cached.filter(|_| !r.audio || cached_audio.is_some());
    let mut prompts = Vec::new();
    if cached.is_none() {
        prompts.push(r.prompt.clone());
    }
    if let (Some(text), None) = (&negative_text, &cached_negative) {
        prompts.push(text.clone());
    }
    let mut fresh = if prompts.is_empty() {
        report(event("cached_video_prompt", 1, 1));
        Vec::new()
    } else {
        report(event("encoding_video_prompt", 0, 1));
        crate::ltx_text_wgpu::contexts_streams(&r.text_encoder, r.tokenizer.as_deref(), store, &prompts, r.device, r.audio, |n, of| report(event("encoding_video_prompt", n, of)))?
    }
    .into_iter();
    let mut next = || -> Result<(Tensor, Option<Tensor>)> {
        let (video, sound) = fresh.next().ok_or_else(|| candle_core::Error::Msg("a prompt's context is missing".into()))?;
        Ok((Tensor::from_vec(video, (1, 1024, 4096), &Device::Cpu)?, sound.map(|s| Tensor::from_vec(s, (1, 1024, 2048), &Device::Cpu)).transpose()?))
    };
    let (context, audio_context) = match cached {
        Some(c) => (c, cached_audio.filter(|_| r.audio)),
        None => {
            let (c, a) = next()?;
            if let Some(cache) = cache::PromptCache::new(r) {
                let _ = cache.save(&c);
            }
            if let (Some(cache), Some(a)) = (cache::PromptCache::audio(r).filter(|_| r.audio), &a) {
                let _ = cache.save(a);
            }
            (c, a)
        }
    };
    let negative = match (negative_text, cached_negative) {
        (Some(_), Some(c)) => Some(c),
        (Some(_), None) => {
            let (c, _) = next()?;
            if let Some(cache) = &negative_cache {
                let _ = cache.save(&c);
            }
            Some(c)
        }
        _ => None,
    };
    Ok((context, audio_context, negative))
}
#[cfg(not(feature = "webgpu"))]
pub(super) fn webgpu_contexts(_: &Request, _: &mut Store, _: Option<Tensor>, _: Option<Tensor>, _: Option<String>, _: &mut dyn FnMut(Json)) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// The clip of `latent` (`[1, f h w, 128]`) by the video decoder on WebGPU: `[1, 3, frames, height, width]`.
#[cfg(feature = "webgpu")]
pub(super) fn webgpu_decode(r: &Request, latent: &Tensor, f: usize, h: usize, w: usize) -> Result<Tensor> {
    let mut store = Store::open(&r.vae, 0)?;
    let decoder = crate::ltx_vae_wgpu::WgpuLtxVae::load(&mut store, r.device)?;
    drop(store);
    decoder.decode_fitted(&latent.flatten_all()?.to_vec1::<f32>()?, f, h, w)
}
#[cfg(not(feature = "webgpu"))]
pub(super) fn webgpu_decode(_: &Request, _: &Tensor, _: usize, _: usize, _: usize) -> Result<Tensor> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// The start and end images' latents (`[1, h w, 128]`) by the VAE's encoder on WebGPU.
#[cfg(feature = "webgpu")]
pub(super) fn webgpu_endpoints(r: &Request, (width, height): (usize, usize), dev: &Device, report: &mut dyn FnMut(Json)) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let mut store = Store::open(&r.vae, 0)?;
    let encoder = crate::ltx_vae_wgpu::WgpuLtxImageEncoder::load(&mut store, r.device)?;
    drop(store);
    let mut encode = |path: &Option<PathBuf>, stage: &str| -> Result<Option<Tensor>> {
        path.as_ref()
            .map(|path| {
                report(event(stage, 0, 1));
                let latent = encoder.encode(&image_values(path, (width, height), &r.ffmpeg, image_crf(r))?, height, width)?;
                Tensor::from_vec(latent, (1, height / 32 * (width / 32), 128), dev)
            })
            .transpose()
    };
    Ok((encode(&r.image, "encoding_starting_image")?, encode(&r.end_image, "encoding_ending_image")?))
}
#[cfg(not(feature = "webgpu"))]
pub(super) fn webgpu_endpoints(_: &Request, _: (usize, usize), _: &Device, _: &mut dyn FnMut(Json)) -> Result<(Option<Tensor>, Option<Tensor>)> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}
