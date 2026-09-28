//! Audio in and out: a voice's reference clip read as 24 kHz mono (PCM WAV
//! natively, anything else through FFmpeg), and 16-bit output.
use std::path::Path;
use std::process::{Command, Stdio};

pub const SAMPLE_RATE: usize = crate::codec::SAMPLE_RATE;
/// The longest reference clip used; the model was trained on short prompts.
pub const MAX_CLIP_SECONDS: f64 = 30.;
/// A clip shorter than this cannot carry a voice.
pub const MIN_CLIP_SECONDS: f64 = 1.;

fn err(s: impl Into<String>) -> std::io::Error {
    std::io::Error::other(s.into())
}

/// A reference clip as the voice is made from it: 24 kHz mono, silence at
/// either end trimmed, at most 30 seconds (cut at a quiet moment near the
/// end). A transcript given with the clip must match this audio.
pub fn read_clip(path: &Path, ffmpeg: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let audio = match parse_wav(&bytes) {
        Ok((channels, rate)) => resample(&mix(&channels), rate, SAMPLE_RATE),
        // Not a plain PCM WAV (MP3, M4A, compressed WAV...): FFmpeg decodes it.
        Err(_) => decode_with_ffmpeg(path, ffmpeg)?,
    };
    let clip = limit_length(trim_silence(&audio), (MAX_CLIP_SECONDS * SAMPLE_RATE as f64) as usize);
    if (clip.len() as f64) < MIN_CLIP_SECONDS * SAMPLE_RATE as f64 {
        return Err(err(format!("{}: the voice clip needs at least a second of speech (3 to 30 seconds is best)", path.display())));
    }
    Ok(clip)
}

/// FFmpeg decodes (and resamples) any audio it knows to 24 kHz mono F32.
pub fn decode_with_ffmpeg(path: &Path, ffmpeg: &Path) -> std::io::Result<Vec<f32>> {
    let mut command = Command::new(ffmpeg);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console window for the child.
        command.creation_flags(0x0800_0000);
    }
    let rate = SAMPLE_RATE.to_string();
    let out = command
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(path)
        .args(["-t", "60", "-vn", "-ac", "1", "-ar", &rate, "-f", "f32le", "pipe:1"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| err(format!("FFmpeg ({}) could not be started to read {}: {e}", ffmpeg.display(), path.display())))?;
    if !out.status.success() {
        let log = String::from_utf8_lossy(&out.stderr);
        return Err(err(format!("FFmpeg could not read {}: {}", path.display(), log.lines().last().unwrap_or("").trim())));
    }
    Ok(out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// A RIFF WAV of integer PCM (8, 16, 24 or 32 bits) or float (32 or 64
/// bits): each channel's samples in [-1, 1], and the sample rate.
pub fn parse_wav(bytes: &[u8]) -> std::io::Result<(Vec<Vec<f32>>, usize)> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(err("not a WAV file"));
    }
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let (mut format, mut data) = (None, None);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4) as usize;
        let body = &bytes[at + 8..(at + 8 + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => format = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + size + (size & 1);
    }
    let (fmt, data) = (format.ok_or_else(|| err("WAV without a fmt chunk"))?, data.ok_or_else(|| err("WAV without a data chunk"))?);
    let mut kind = u16_at(fmt, 0);
    // WAVE_FORMAT_EXTENSIBLE: the real format is the sub-format GUID's first two bytes.
    if kind == 0xFFFE && fmt.len() >= 26 {
        kind = u16_at(fmt, 24);
    }
    let channels = u16_at(fmt, 2) as usize;
    let rate = u32_at(fmt, 4) as usize;
    let bits = u16_at(fmt, 14) as usize;
    if channels == 0 || !(1000..=384_000).contains(&rate) {
        return Err(err("WAV with no channels or an unusual sample rate"));
    }
    let width = bits / 8;
    let sample = |b: &[u8]| -> Option<f32> {
        Some(match (kind, bits) {
            (1, 8) => (b[0] as f32 - 128.) / 128.,
            (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.,
            (1, 24) => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.,
            (1, 32) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.,
            (3, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (3, 64) => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
            _ => return None,
        })
    };
    if width == 0 || sample(&[0; 8]).is_none() {
        return Err(err(format!("WAV format {kind} with {bits}-bit samples is not plain PCM")));
    }
    let frames = data.len() / (width * channels);
    let mut out = vec![Vec::with_capacity(frames); channels];
    for f in data.chunks_exact(width * channels) {
        for (c, s) in f.chunks_exact(width).enumerate() {
            out[c].push(sample(s).unwrap_or(0.).clamp(-1., 1.));
        }
    }
    Ok((out, rate))
}

/// Channels averaged to one.
pub fn mix(channels: &[Vec<f32>]) -> Vec<f32> {
    match channels {
        [] => Vec::new(),
        [one] => one.clone(),
        many => {
            let n = many.iter().map(Vec::len).min().unwrap_or(0);
            (0..n).map(|i| many.iter().map(|c| c[i]).sum::<f32>() / many.len() as f32).collect()
        }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Band-limited resampling: a Blackman-windowed sinc, 32 zero crossings a
/// side, cut off just below the lower Nyquist frequency. Rational ratios
/// (every common rate to 24 kHz) use a table of the filter's phases.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from, to);
    let (up, down) = (to / g, from / g);
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.) * 0.95;
    let zeros = 32.;
    let half = (zeros / cutoff).ceil() as isize;
    let kernel = |d: f64| -> f64 {
        let u = d / half as f64;
        if u.abs() >= 1. {
            return 0.;
        }
        let window = 0.42 + 0.5 * (std::f64::consts::PI * u).cos() + 0.08 * (2. * std::f64::consts::PI * u).cos();
        let s = if d == 0. { 1. } else { (std::f64::consts::PI * cutoff * d).sin() / (std::f64::consts::PI * cutoff * d) };
        cutoff * s * window
    };
    let taps = 2 * half as usize;
    // Output i sits at input position i * down / up: its whole part and one
    // of `up` phases.
    let table: Option<Vec<f32>> = (up <= 4096).then(|| (0..up).flat_map(|p| (0..taps).map(move |t| (p, t))).map(|(p, t)| kernel(p as f64 / up as f64 - (t as isize - half + 1) as f64) as f32).collect());
    let n_out = (x.len() as u128 * up as u128 / down as u128) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as u128 * down as u128;
        let (whole, phase) = ((pos / up as u128) as isize, (pos % up as u128) as usize);
        let mut acc = 0f64;
        for t in 0..taps {
            let j = whole + t as isize - half + 1;
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let w = match &table {
                Some(tab) => tab[phase * taps + t] as f64,
                None => kernel(phase as f64 / up as f64 - (t as isize - half + 1) as f64),
            };
            acc += x[j as usize] as f64 * w;
        }
        out.push(acc as f32);
    }
    out
}

/// Loudness of each 10 ms window (RMS).
fn window_levels(x: &[f32]) -> Vec<f32> {
    x.chunks(SAMPLE_RATE / 100).map(|w| (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt()).collect()
}

/// Silence before the first sound and after the last, down to a tenth of a
/// second before and a fifth after. "Sound" is a 10 ms window within 26 dB of
/// the clip's loud parts, so a quiet recording keeps its quiet words.
pub fn trim_silence(x: &[f32]) -> Vec<f32> {
    let levels = window_levels(x);
    if levels.is_empty() {
        return Vec::new();
    }
    let mut sorted = levels.clone();
    sorted.sort_by(f32::total_cmp);
    let loud = sorted[sorted.len() * 9 / 10];
    let threshold = (loud * 0.05).max(1e-4);
    let (Some(first), Some(last)) = (levels.iter().position(|&l| l > threshold), levels.iter().rposition(|&l| l > threshold)) else {
        return Vec::new();
    };
    let w = SAMPLE_RATE / 100;
    let start = (first * w).saturating_sub(SAMPLE_RATE / 10);
    let end = ((last + 1) * w + SAMPLE_RATE / 5).min(x.len());
    x[start..end].to_vec()
}

/// At most `max` samples; a longer clip is cut at its quietest 10 ms in the
/// last sixth before `max`, so it ends between words rather than in one.
pub fn limit_length(x: Vec<f32>, max: usize) -> Vec<f32> {
    if x.len() <= max {
        return x;
    }
    let w = SAMPLE_RATE / 100;
    let levels = window_levels(&x[..max]);
    let from = levels.len() * 5 / 6;
    let quietest = (from..levels.len()).min_by(|&a, &b| levels[a].total_cmp(&levels[b])).unwrap_or(levels.len());
    let mut x = x;
    x.truncate(((quietest + 1) * w).min(max));
    x
}

/// Samples in [-1, 1] as 16-bit PCM.
pub fn to_pcm16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|s| (s.clamp(-1., 1.) * 32767.).round() as i16).collect()
}

/// 16-bit PCM mono WAV bytes.
pub fn wav_bytes(samples: &[i16], rate: usize) -> Vec<u8> {
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(rate as u32).to_le_bytes());
    out.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// 16-bit PCM mono WAV.
pub fn write_wav(path: &Path, samples: &[f32], rate: usize) -> std::io::Result<()> {
    std::fs::write(path, wav_bytes(&to_pcm16(samples), rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: usize, seconds: f64) -> Vec<f32> {
        (0..(rate as f64 * seconds) as usize).map(|i| (0.5 * (2. * std::f64::consts::PI * freq * i as f64 / rate as f64).sin()) as f32).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn wav_files_of_every_plain_format_read_back() {
        let pcm16 = wav_bytes(&[0, 16384, -32768, 32767], 16_000);
        let (ch, rate) = parse_wav(&pcm16).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(ch, vec![vec![0., 0.5, -1., 32767. / 32768.]]);
        // Stereo 24-bit and mono float, hand-built.
        let build = |kind: u16, channels: u16, bits: u16, data: &[u8]| {
            let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
            b.extend_from_slice(&16u32.to_le_bytes());
            b.extend_from_slice(&kind.to_le_bytes());
            b.extend_from_slice(&channels.to_le_bytes());
            b.extend_from_slice(&48_000u32.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.extend_from_slice(&bits.to_le_bytes());
            b.extend_from_slice(b"LIST\x03\0\0\0abc\0"); // an odd-sized chunk to skip
            b.extend_from_slice(b"data");
            b.extend_from_slice(&(data.len() as u32).to_le_bytes());
            b.extend_from_slice(data);
            b
        };
        let s24 = build(1, 2, 24, &[0, 0, 0x40, 0, 0, 0xC0]);
        let (ch, rate) = parse_wav(&s24).unwrap();
        assert_eq!((ch, rate), (vec![vec![0.5], vec![-0.5]], 48_000));
        let f32s: Vec<u8> = [0.25f32, -0.75].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(parse_wav(&build(3, 1, 32, &f32s)).unwrap().0, vec![vec![0.25, -0.75]]);
        assert!(parse_wav(&build(2, 1, 4, &[0, 0])).is_err(), "ADPCM needs FFmpeg");
        assert!(parse_wav(b"ID3\x04 an mp3").is_err());
        assert_eq!(mix(&[vec![1., 0.5], vec![0., 0.5]]), vec![0.5, 0.5]);
    }

    #[test]
    fn resampling_keeps_what_fits_and_drops_what_does_not() {
        for from in [16_000, 22_050, 44_100, 48_000] {
            let x = tone(440., from, 0.5);
            let y = resample(&x, from, 24_000);
            assert_eq!(y.len(), (x.len() as u64 * 24_000 / from as u64) as usize, "{from}");
            // A 440 Hz tone passes at its level (away from the edges) and in phase.
            let mid = &y[2400..y.len() - 2400];
            assert!((rms(mid) - 0.5 / 2f32.sqrt()).abs() < 0.01, "{from}: {}", rms(mid));
            let want = tone(440., 24_000, 0.5);
            let d = mid.iter().zip(&want[2400..]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d < 0.01, "{from}: off by {d}");
        }
        // 15 kHz is above the new Nyquist (12 kHz): gone.
        let y = resample(&tone(15_000., 48_000, 0.5), 48_000, 24_000);
        assert!(rms(&y[2400..y.len() - 2400]) < 0.01);
    }

    #[test]
    fn clips_lose_their_silent_ends_and_long_ones_end_between_words() {
        let rate = SAMPLE_RATE;
        let mut x = vec![0.0001f32; rate];
        x.extend(tone(200., rate, 2.));
        x.extend(vec![0.; rate * 2]);
        let t = trim_silence(&x);
        assert_eq!(t.len(), rate / 10 + 2 * rate + rate / 5);
        // A quiet recording keeps its speech.
        let quiet: Vec<f32> = x.iter().map(|s| s * 0.01).collect();
        assert_eq!(trim_silence(&quiet).len(), t.len());
        assert!(trim_silence(&[0.; 1000]).is_empty());
        // 40 s of tone with a gap at 27 s: cut in the gap, under 30 s.
        let mut long = tone(200., rate, 27.);
        long.extend(vec![0.; rate / 5]);
        long.extend(tone(200., rate, 13.));
        let cut = limit_length(long, 30 * rate);
        assert!(cut.len() > 27 * rate && cut.len() <= 27 * rate + rate / 5, "{}", cut.len() as f64 / rate as f64);
    }
}
