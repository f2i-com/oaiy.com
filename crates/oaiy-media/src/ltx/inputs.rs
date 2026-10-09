//! What a clip is given: a soundtrack (read by FFmpeg, its length in frames, its level), and the images at its
//! ends (re-compressed as the reference does, encoded, and held in the latent).

use super::*;

/// Speech loudness: the 95th percentile of 20 ms RMS, the level of its
/// voiced parts (pauses do not pull it down). None for silence.
pub(super) fn speech_loudness(samples: &[f32], rate: usize) -> Option<f32> {
    let w = (rate / 50).max(1);
    let mut rms: Vec<f32> = samples.chunks(w).map(|c| (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt()).collect();
    rms.sort_by(|a, b| a.total_cmp(b));
    let at = rms.get(rms.len() * 95 / 100).copied()?;
    (at > 1e-5).then_some(at)
}

/// The generated soundtrack's level (interleaved stereo): matched to the
/// reference voice's loudness when there is one (within 4x either way), then
/// scaled so the peak stays under -0.3 dB rather than clipping.
pub(super) fn set_level(interleaved: &mut [f32], rate: usize, reference: Option<f32>) {
    let left: Vec<f32> = interleaved.iter().step_by(2).copied().collect();
    let mut gain = match (reference, speech_loudness(&left, rate)) {
        (Some(want), Some(have)) => (want / have).clamp(0.25, 4.),
        _ => 1.,
    };
    let peak = interleaved.iter().fold(0f32, |m, x| m.max(x.abs())) * gain;
    const CEILING: f32 = 0.966; // -0.3 dBFS
    if peak > CEILING {
        gain *= CEILING / peak;
    }
    if gain != 1. {
        interleaved.iter_mut().for_each(|x| *x *= gain);
    }
}

/// The clip length for a soundtrack of `seconds`: enough frames at `fps` to
/// hold all of it, snapped up to 8k+1 (the reference snaps down, which cuts
/// the last words off), within 9..=121. The soundtrack is padded with silence
/// to the clip's length.
pub(super) fn frames_for_audio(seconds: f64, fps: usize) -> usize {
    let raw = ((seconds * fps as f64 - 1e-6).ceil().max(0.) as usize).clamp(9, 121);
    ((raw - 1).div_ceil(8) * 8 + 1).min(121)
}

/// Decode any audio FFmpeg reads to stereo F32 at its own sample rate (mono
/// is duplicated; more channels are mixed down). At most 60 seconds.
pub(super) fn read_audio(ffmpeg: &std::path::Path, path: &std::path::Path) -> Result<([Vec<f32>; 2], usize)> {
    let out = command(ffmpeg)
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(path)
        .args(["-t", "60", "-vn", "-ac", "2", "-f", "f32le", "pipe:1"])
        .stdin(Stdio::null())
        .output()?;
    let log = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        candle_core::bail!("FFmpeg could not read the soundtrack: {}", log.lines().last().unwrap_or("").trim());
    }
    let rate = regex::Regex::new(r"Audio: [^\n]*?, (\d+) Hz")
        .map_err(candle_core::Error::wrap)?
        .captures(&log)
        .and_then(|c| c[1].parse::<usize>().ok())
        .filter(|r| (1000..=384_000).contains(r))
        .ok_or_else(|| candle_core::Error::Msg("the soundtrack has no audio stream".into()))?;
    let samples: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if samples.len() < 2 * rate / 10 {
        candle_core::bail!("the soundtrack is shorter than a tenth of a second");
    }
    let left = samples.iter().step_by(2).copied().collect();
    let right = samples.iter().skip(1).step_by(2).copied().collect();
    Ok(([left, right], rate))
}

/// Audio latent frames for a clip: 25 per second of video, rounded half to
/// even as the reference does.
pub(super) fn audio_latent_frames(frames: usize, fps: usize) -> usize {
    (frames as f64 / fps as f64 * audio::LATENT_RATE).round_ties_even().max(1.) as usize
}

/// The H.264 quality an image is re-compressed at before it conditions a
/// clip, as the reference does: the models were trained on video frames, and a
/// pristine still tends to stay still. 33 for the LTX 2.3 generation (Sulphur
/// is one), 18 from 2.4 on; 0 leaves the image as it is.
pub(super) fn image_crf(r: &Request) -> u32 {
    r.image_crf.unwrap_or(if r.model == "ltx-2.5" { 18 } else { 33 })
}

/// An image through one H.264 frame at `crf` and back (FFmpeg, libx264,
/// veryfast, 4:2:0), at its own size cut to even sides.
fn recompress(ffmpeg: &std::path::Path, image: image::RgbImage, crf: u32) -> Result<image::RgbImage> {
    use std::io::Write;
    let (w, h) = (image.width() / 2 * 2, image.height() / 2 * 2);
    if crf == 0 || w < 2 || h < 2 {
        return Ok(image);
    }
    let image = image::imageops::crop_imm(&image, 0, 0, w, h).to_image();
    let run = |args: &[&str], input: &[u8]| -> Result<Vec<u8>> {
        let mut child = command(ffmpeg)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdin = child.stdin.take().ok_or_else(|| candle_core::Error::Msg("FFmpeg stdin".into()))?;
        let input = input.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let out = child.wait_with_output()?;
        writer.join().map_err(|_| candle_core::Error::Msg("FFmpeg writer".into()))??;
        if !out.status.success() || out.stdout.is_empty() {
            candle_core::bail!("FFmpeg could not re-compress the conditioning image");
        }
        Ok(out.stdout)
    };
    let size = format!("{w}x{h}");
    let crf = crf.to_string();
    let h264 = run(&["-hide_banner", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-i", "-", "-frames:v", "1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", &crf, "-pix_fmt", "yuv420p", "-f", "h264", "-"], image.as_raw())?;
    let rgb = run(&["-hide_banner", "-f", "h264", "-i", "-", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], &h264)?;
    image::RgbImage::from_raw(w, h, rgb.get(..(w * h * 3) as usize).map(<[u8]>::to_vec).unwrap_or_default())
        .ok_or_else(|| candle_core::Error::Msg("FFmpeg returned a short frame".into()))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_image(
    path: &std::path::Path,
    (width, height): (usize, usize),
    encoder: &vae::LtxVideoEncoder,
    dev: &Device,
    dtype: DType,
    ffmpeg: &std::path::Path,
    crf: u32,
) -> Result<Tensor> {
    let values = image_values(path, (width, height), ffmpeg, crf)?;
    let pixels = Tensor::from_vec(values, (1, 1, height, width, 3), &dev)?
        .permute((0, 4, 1, 2, 3))?
        .contiguous()?
        .to_dtype(dtype)?;
    let latent = encoder
        .encode_means(&pixels)?
        .permute((0, 2, 3, 4, 1))?
        .contiguous()?
        .reshape((1, height / 32 * (width / 32), 128))?
        .to_dtype(DType::F32)?;
    Ok(latent)
}

/// An endpoint image's pixels as the encoder takes them: re-compressed as the clip's frames are (`crf`), resized to
/// fill `width` by `height`, each pixel's red, green and blue in -1..1 in turn.
pub(super) fn image_values(path: &std::path::Path, (width, height): (usize, usize), ffmpeg: &std::path::Path, crf: u32) -> Result<Vec<f32>> {
    let mut reader = image::ImageReader::open(path)?
        .with_guessed_format()
        .map_err(candle_core::Error::wrap)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let decoded = reader.decode().map_err(candle_core::Error::wrap)?.to_rgb8();
    let pixels = image::DynamicImage::ImageRgb8(recompress(ffmpeg, decoded, crf)?)
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    Ok(pixels.as_raw().iter().map(|&v| v as f32 / 127.5 - 1.).collect())
}
pub(super) fn condition_endpoints(
    latent: Tensor,
    start: Option<&Tensor>,
    end: Option<&Tensor>,
) -> Result<Tensor> {
    let start_tokens = start.map(|t| t.dim(1)).transpose()?.unwrap_or(0);
    let end_tokens = end.map(|t| t.dim(1)).transpose()?.unwrap_or(0);
    let tokens = latent.dim(1)?;
    if start_tokens + end_tokens >= tokens {
        candle_core::bail!("endpoint conditioning must leave generated tokens");
    }
    let mut parts = Vec::new();
    if let Some(start) = start {
        parts.push(start.clone());
    }
    parts.push(latent.narrow(1, start_tokens, tokens - start_tokens - end_tokens)?);
    if let Some(end) = end {
        parts.push(end.clone());
    }
    Tensor::cat(&parts, 1)
}

pub(super) fn command(path: &std::path::Path) -> Command {
    let mut c = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x08000000);
    }
    c
}
