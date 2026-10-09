//! What a clip leaves: its soundtrack as samples, and its file written by FFmpeg with a preview beside it.

use super::*;

/// The clip written by FFmpeg: `pixels` (`[1, 3, frames, height, width]`, -1 to 1) as H.264, the soundtrack
/// (interleaved stereo and its rate) beside it as AAC where there is one, and the middle frame as a PNG. The
/// clip's path and its preview's.
pub(super) fn export(r: &Request, pixels: &Tensor, soundtrack: Option<&(Vec<f32>, usize)>) -> Result<(PathBuf, PathBuf)> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(candle_core::Error::wrap)?
        .as_nanos();
    let path = r.output.join(format!("{}-{stamp}-{}.mp4", r.model, r.seed));
    let preview = path.with_extension("png");
    let log = path.with_extension("ffmpeg.log");
    let (b, c, frames, height, width) = pixels.dims5()?;
    if (b, c, frames, height, width) != (1, 3, r.frames, r.height, r.width) {
        candle_core::bail!("unexpected decoded video shape {:?}", pixels.dims());
    }
    let log_file = std::fs::File::create(&log)?;
    // The soundtrack goes to FFmpeg as a second, raw input beside the frames.
    let audio_file = path.with_extension("f32le");
    let mut audio_args: Vec<String> = vec!["-an".into()];
    if let Some((samples, rate)) = &soundtrack {
        let bytes: Vec<u8> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&audio_file, bytes)?;
        audio_args = ["-f", "f32le", "-ar", &rate.to_string(), "-ac", "2", "-i"]
            .iter()
            .map(|s| s.to_string())
            .chain([audio_file.to_string_lossy().into_owned()])
            .chain(["-c:a", "aac", "-b:a", "192k"].iter().map(|s| s.to_string()))
            .collect();
    }
    let spawned = command(&r.ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", r.width, r.height),
            "-r",
            &r.fps.to_string(),
            "-i",
            "pipe:0",
        ])
        .args(&audio_args)
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(log_file)
        .spawn();
    let mut encoder = match spawned {
        Ok(e) => e,
        Err(e) => {
            let _ = std::fs::remove_file(&audio_file);
            return Err(e.into());
        }
    };
    let result = (|| -> Result<()> {
        let mut stdin = encoder
            .stdin
            .take()
            .ok_or_else(|| candle_core::Error::Msg("FFmpeg stdin unavailable".into()))?;
        // Transfer/cast the decoded video once. Repeated small CPU tensor
        // permutations per frame otherwise dominate short-clip export on Windows.
        let values = pixels
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if values.iter().any(|x| !x.is_finite()) {
            candle_core::bail!("nonfinite decoded video pixels");
        }
        let plane = r.height * r.width;
        let channel_stride = r.frames * plane;
        let mut bytes = vec![0u8; plane * 3];
        for i in 0..r.frames {
            for (pixel, rgb) in bytes.chunks_exact_mut(3).enumerate() {
                for (c, value) in rgb.iter_mut().enumerate() {
                    let x = values[c * channel_stride + i * plane + pixel];
                    *value = ((x + 1.) * 127.5).round().clamp(0., 255.) as u8;
                }
            }
            if i == r.frames / 2 {
                image::save_buffer(
                    &preview,
                    &bytes,
                    r.width as u32,
                    r.height as u32,
                    image::ColorType::Rgb8,
                )
                .map_err(candle_core::Error::wrap)?;
            }
            stdin.write_all(&bytes)?;
        }
        drop(stdin);
        Ok(())
    })();
    if let Err(e) = result {
        let _ = encoder.kill();
        let _ = encoder.wait();
        let _ = std::fs::remove_file(&audio_file);
        return Err(e);
    }
    let finished = encoder.wait();
    let _ = std::fs::remove_file(&audio_file);
    if !finished?.success() {
        candle_core::bail!("FFmpeg failed; see {}", log.display());
    }
    let _ = std::fs::remove_file(log);
    Ok((path, preview))
}

/// The clip's soundtrack as interleaved stereo and its rate: the one it was given, as it was, cut to the clip (or
/// padded with silence); else its audio latent through the audio VAE's decoder and the vocoder, trimmed (or
/// padded) to exactly the clip's length and set to the reference's level. None for a silent clip.
pub(super) fn soundtrack_out(r: &Request, audio_latent: Option<&Tensor>, soundtrack_in: Option<&([Vec<f32>; 2], usize)>, reference_loudness: Option<f32>, dev: &Device, report: &mut dyn FnMut(Json)) -> Result<Option<(Vec<f32>, usize)>> {
    let soundtrack = match (&audio_latent, &r.audio_vae) {
        // A given soundtrack is kept as it was (the reference returns the
        // input audio, not its VAE round trip), cut to the clip.
        _ if soundtrack_in.is_some() => {
            let (channels, rate) = soundtrack_in.as_ref().ok_or_else(|| candle_core::Error::Msg("soundtrack missing".into()))?;
            let samples = (r.frames as f64 / r.fps as f64 * *rate as f64) as usize;
            let at = |c: &Vec<f32>, i: usize| c.get(i).copied().unwrap_or(0.);
            let interleaved: Vec<f32> = (0..samples).flat_map(|i| [at(&channels[0], i), at(&channels[1], i)]).collect();
            Some((interleaved, *rate))
        }
        (Some(latent), Some(path)) => {
            report(event("decoding_audio", 0, 1));
            let decoder = audio::AudioDecoder::load(path, &dev)?;
            // (a WebGPU job's: the vocoder and the bandwidth extension on its GPU, nearly all of the decoding)
            #[cfg(feature = "webgpu")]
            let wave = if r.webgpu { decoder.decode_webgpu(r.device, latent)? } else { decoder.decode(latent)? };
            #[cfg(not(feature = "webgpu"))]
            let wave = decoder.decode(latent)?;
            let rate = decoder.sample_rate;
            drop(decoder);
            let samples = (r.frames * rate).div_ceil(r.fps);
            let have = wave.dim(2)?;
            let wave = if have >= samples { wave.narrow(2, 0, samples)? } else { wave.pad_with_zeros(2, 0, samples - have)? };
            // Interleaved stereo, as FFmpeg's f32le input wants it.
            let mut interleaved = wave.squeeze(0)?.transpose(0, 1)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            if interleaved.iter().any(|x| !x.is_finite()) {
                candle_core::bail!("nonfinite decoded audio samples");
            }
            set_level(&mut interleaved, rate, reference_loudness);
            Some((interleaved, rate))
        }
        _ => None,
    };
    Ok(soundtrack)
}
