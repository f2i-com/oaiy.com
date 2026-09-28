//! Audio in: WAV files (PCM 8/16/24/32-bit or 32-bit float, any channel
//! count, mixed to mono) resampled to the model's rate, and the request
//! encodings they arrive in (base64, data URLs, multipart forms).

/// Rates accepted from a WAV header. Anything outside is malformed or an
/// allocation attack: a 1 Hz "rate" would turn a small body into a huge
/// resampled buffer.
const MIN_RATE: u32 = 8_000;
const MAX_RATE: u32 = 192_000;

/// The longest audio accepted, in seconds (at the output rate).
pub const MAX_SECONDS: usize = 600;

#[derive(Clone, Debug, PartialEq)]
pub struct Wav {
    pub sample_rate: u32,
    /// Mono samples in [-1, 1].
    pub samples: Vec<f32>,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Parse a RIFF/WAVE file.
pub fn parse_wav(bytes: &[u8]) -> Result<Wav, String> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32_at(bytes, at + 4) as usize;
        let body_start = at + 8;
        // A streamed WAV may declare a data length past the end (or 0xFFFFFFFF): take what is there.
        let body_end = body_start.saturating_add(len).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err("fmt chunk too short".into());
                }
                let mut format = u16_at(body, 0);
                let channels = u16_at(body, 2);
                let rate = u32_at(body, 4);
                let bits = u16_at(body, 14);
                // WAVE_FORMAT_EXTENSIBLE: the real format is the sub-format GUID's first two bytes.
                if format == 0xFFFE && body.len() >= 26 {
                    format = u16_at(body, 24);
                }
                fmt = Some((format, channels, rate, bits));
            }
            b"data" => {
                data = Some(body);
                if fmt.is_some() {
                    break;
                }
            }
            _ => {}
        }
        // Chunks are padded to an even length.
        at = body_start.saturating_add(len).saturating_add(len & 1);
    }
    let (format, channels, rate, bits) = fmt.ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    if channels == 0 || channels > 32 {
        return Err(format!("{channels} channels"));
    }
    if !(MIN_RATE..=MAX_RATE).contains(&rate) {
        return Err(format!("unsupported sample rate {rate} Hz (accepted {MIN_RATE}..={MAX_RATE})"));
    }
    let width = match (format, bits) {
        (1, 8) => 1,
        (1, 16) => 2,
        (1, 24) => 3,
        (1, 32) | (3, 32) => 4,
        _ => return Err(format!("unsupported WAV encoding (format {format}, {bits} bits): PCM 8/16/24/32-bit or 32-bit float")),
    };
    let frame = width * channels as usize;
    let frames = data.len() / frame;
    let out_len = (frames as u64).saturating_mul(16_000) / rate as u64;
    if out_len > (MAX_SECONDS * 16_000) as u64 {
        return Err(format!("audio is longer than {MAX_SECONDS} s"));
    }
    let sample = |s: &[u8]| -> f32 {
        match (format, width) {
            (1, 1) => (s[0] as f32 - 128.0) / 128.0,
            (1, 2) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
            (1, 3) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
            (1, 4) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
            _ => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        }
    };
    let mut samples = Vec::with_capacity(frames);
    for f in data.chunks_exact(frame) {
        let sum: f32 = f.chunks_exact(width).map(sample).sum();
        let v = sum / channels as f32;
        samples.push(if v.is_finite() { v } else { 0.0 });
    }
    Ok(Wav { sample_rate: rate, samples })
}

/// A 16-bit mono WAV (tests and tools).
pub fn write_wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

/// torchaudio's `functional.resample` (Hann-windowed sinc, 6 zero crossings,
/// rolloff 0.99) for one channel, `from` Hz to `to` Hz (as `oaiy-media`'s
/// `ltx/audio.rs` has it).
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let g = gcd(from, to);
    let (orig, new) = (from / g, to / g);
    let (zeros, rolloff) = (6f64, 0.99f64);
    let base = orig.min(new) as f64 * rolloff;
    let width = (zeros * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    // One kernel per output phase, computed in f64 and applied in f32.
    let kernels: Vec<Vec<f32>> = (0..new)
        .map(|phase| {
            (0..taps)
                .map(|i| {
                    let idx = (i as f64 - width as f64) / orig as f64;
                    let t = ((-(phase as f64) / new as f64 + idx) * base).clamp(-zeros, zeros);
                    let window = (t * std::f64::consts::PI / zeros / 2.).cos().powi(2);
                    let t = t * std::f64::consts::PI;
                    let sinc = if t == 0. { 1. } else { t.sin() / t };
                    (sinc * window * base / orig as f64) as f32
                })
                .collect()
        })
        .collect();
    // Zeros: `width` before, `width + orig` after.
    let padded: Vec<f32> = std::iter::repeat_n(0f32, width).chain(x.iter().copied()).chain(std::iter::repeat_n(0f32, width + orig)).collect();
    let blocks = (padded.len() - taps) / orig + 1;
    let target = (new as u128 * x.len() as u128).div_ceil(orig as u128) as usize;
    let mut out = Vec::with_capacity(target);
    'outer: for b in 0..blocks {
        let window = &padded[b * orig..b * orig + taps];
        for k in &kernels {
            if out.len() == target {
                break 'outer;
            }
            out.push(window.iter().zip(k).map(|(a, w)| a * w).sum());
        }
    }
    out
}

/// Standard or URL-safe base64, padded or not, whitespace ignored; a
/// `data:...;base64,` URL is accepted too.
pub fn decode_base64_field(value: &str) -> Result<Vec<u8>, String> {
    let v = value.trim();
    let payload = if v.len() >= 5 && v[..5].eq_ignore_ascii_case("data:") {
        let comma = v.find(',').ok_or("data URL has no comma")?;
        if !v[..comma].to_ascii_lowercase().contains(";base64") {
            return Err("data URL is not base64".into());
        }
        &v[comma + 1..]
    } else {
        v
    };
    let mut out = Vec::with_capacity(payload.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut padding = false;
    for b in payload.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => {
                padding = true;
                continue;
            }
            b if b.is_ascii_whitespace() => continue,
            other => return Err(format!("invalid base64 byte 0x{other:02x}")),
        };
        if padding {
            return Err("base64 data after padding".into());
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if bits >= 6 {
        return Err("invalid base64 length".into());
    }
    Ok(out)
}

/// The parts of a `multipart/form-data` body the transcription route reads.
#[derive(Debug, Default)]
pub struct Form {
    pub file: Option<Vec<u8>>,
    pub fields: Vec<(String, String)>,
}

impl Form {
    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Parse a multipart body: the part named `file` (or any part with a
/// filename) is the audio; other parts are text fields.
pub fn parse_multipart(body: &[u8], content_type: &str) -> Result<Form, String> {
    let boundary = content_type
        .split(';')
        .filter_map(|p| p.trim().split_once('='))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("boundary"))
        .map(|(_, v)| v.trim().trim_matches('"').to_string())
        .ok_or("multipart request has no boundary")?;
    let delim = format!("--{boundary}").into_bytes();
    let mut form = Form::default();
    let mut at = find(body, &delim, 0).ok_or("multipart body has no boundary line")? + delim.len();
    loop {
        if body.get(at..at + 2) == Some(b"--") {
            break;
        }
        // Skip the CRLF after the boundary.
        if body.get(at..at + 2) == Some(b"\r\n") {
            at += 2;
        }
        let head_end = find(body, b"\r\n\r\n", at).ok_or("multipart part has no header end")?;
        let head = String::from_utf8_lossy(&body[at..head_end]).into_owned();
        let start = head_end + 4;
        let mut close = delim.clone();
        close.splice(0..0, b"\r\n".iter().copied());
        let end = find(body, &close, start).ok_or("multipart part has no closing boundary")?;
        let content = &body[start..end];
        let mut name = None;
        let mut filename = None;
        for line in head.lines() {
            let (k, v) = line.split_once(':').unwrap_or((line, ""));
            if k.trim().eq_ignore_ascii_case("content-disposition") {
                for p in v.split(';') {
                    if let Some((pk, pv)) = p.trim().split_once('=') {
                        let pv = pv.trim().trim_matches('"').to_string();
                        match pk.trim().to_ascii_lowercase().as_str() {
                            "name" => name = Some(pv),
                            "filename" => filename = Some(pv),
                            _ => {}
                        }
                    }
                }
            }
        }
        if name.as_deref() == Some("file") || (filename.is_some() && form.file.is_none()) {
            form.file = Some(content.to_vec());
        } else if let Some(n) = name {
            form.fields.push((n, String::from_utf8_lossy(content).into_owned()));
        }
        at = end + close.len();
    }
    Ok(form)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip() {
        let x: Vec<f32> = (0..160).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
        let w = parse_wav(&write_wav(&x, 16_000)).unwrap();
        assert_eq!(w.sample_rate, 16_000);
        assert_eq!(w.samples.len(), 160);
        assert!(w.samples.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn wav_stereo_float_is_mixed() {
        // A 32-bit float stereo WAV with one frame (0.5, -0.25) and a LIST chunk first.
        let mut b = b"RIFF\0\0\0\0WAVELIST\x04\0\0\0abcdfmt \x10\0\0\0\x03\0\x02\0\x80\x3e\0\0\0\0\0\0\x08\0\x20\0data\x08\0\0\0".to_vec();
        b.extend_from_slice(&0.5f32.to_le_bytes());
        b.extend_from_slice(&(-0.25f32).to_le_bytes());
        let w = parse_wav(&b).unwrap();
        assert_eq!(w.samples, vec![0.125]);
    }

    #[test]
    fn wav_rejects_bad_rates_and_encodings() {
        let mut b = write_wav(&[0.0; 4], 16_000);
        b[24..28].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_wav(&b).unwrap_err().contains("sample rate"));
        assert!(parse_wav(b"RIFF\0\0\0\0WAVE").is_err());
        assert!(parse_wav(b"not a wav").is_err());
    }

    #[test]
    fn resample_keeps_a_tone() {
        // 24 kHz to 16 kHz: a 440 Hz tone stays one (compare after the filter's edge).
        let x: Vec<f32> = (0..24_000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin()).collect();
        let y = resample(&x, 24_000, 16_000);
        assert_eq!(y.len(), 16_000);
        for i in 100..15_900 {
            let want = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
            assert!((y[i] - want).abs() < 2e-3, "{i}: {} vs {want}", y[i]);
        }
    }

    #[test]
    fn base64_forms() {
        assert_eq!(decode_base64_field("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64_field("aGVsbG8").unwrap(), b"hello");
        assert_eq!(decode_base64_field(" aGVs\nbG8= ").unwrap(), b"hello");
        assert_eq!(decode_base64_field("data:audio/wav;base64,aGk=").unwrap(), b"hi");
        assert_eq!(decode_base64_field("-_8=").unwrap(), vec![0xfb, 0xff]);
        assert!(decode_base64_field("aGk=aGk=").is_err());
        assert!(decode_base64_field("a").is_err());
        assert!(decode_base64_field("data:audio/wav,abc").is_err());
    }

    #[test]
    fn multipart_file_and_fields() {
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n--data\r\n--XyZ\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n--XyZ--\r\n";
        let f = parse_multipart(body, "multipart/form-data; boundary=XyZ").unwrap();
        assert_eq!(f.file.as_deref(), Some(&b"RIFF\r\n--data"[..]));
        assert_eq!(f.field("response_format"), Some("text"));
        assert_eq!(f.field("model"), Some("whisper-1"));
        assert!(parse_multipart(body, "multipart/form-data").is_err());
    }
}
