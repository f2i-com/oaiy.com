//! `multipart/form-data` (what OpenAI SDKs send to `/v1/images/edits`) and
//! image-header sniffing, std-only.

/// One form field: text, or a file when `filename` is set.
#[derive(Debug)]
pub struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// A `key="value"` (or `key=value`) parameter of a header value.
fn param(header: &str, key: &str) -> Option<String> {
    header.split(';').skip(1).find_map(|p| {
        let (k, v) = p.trim().split_once('=')?;
        k.trim().eq_ignore_ascii_case(key).then(|| v.trim().trim_matches('"').to_string())
    })
}

/// Split a body by the boundary named in `content_type`.
pub fn parse(content_type: &str, body: &[u8]) -> Result<Vec<Part>, String> {
    let boundary = param(content_type, "boundary").filter(|b| !b.is_empty() && b.len() <= 200).ok_or("multipart body without a boundary")?;
    let delim = format!("--{boundary}").into_bytes();
    let mut at = find(body, &delim, 0).ok_or("multipart body without its boundary")?;
    let mut parts = Vec::new();
    loop {
        at += delim.len();
        if body.get(at..at + 2) == Some(b"--") {
            return Ok(parts);
        }
        // Skip the CRLF after the delimiter.
        if body.get(at..at + 2) == Some(b"\r\n") {
            at += 2;
        }
        let head_end = find(body, b"\r\n\r\n", at).ok_or("multipart part without headers")?;
        let head = String::from_utf8_lossy(&body[at..head_end]).into_owned();
        let data_start = head_end + 4;
        let mut close = delim.clone();
        close.splice(0..0, *b"\r\n");
        let data_end = find(body, &close, data_start).ok_or("multipart part without its closing boundary")?;
        // The declared Content-Type is not trusted: images are sniffed by content.
        let (mut name, mut filename) = (None, None);
        for line in head.split("\r\n") {
            let Some((k, v)) = line.split_once(':') else { continue };
            if k.trim().eq_ignore_ascii_case("content-disposition") {
                name = param(v, "name");
                filename = param(v, "filename");
            }
        }
        parts.push(Part {
            name: name.ok_or("multipart part without a name")?,
            filename,
            data: body[data_start..data_end].to_vec(),
        });
        if parts.len() > 64 {
            return Err("too many multipart fields".into());
        }
        at = data_end + 2;
    }
}

/// Image format and pixel size from the first bytes of a PNG, JPEG or WebP.
pub fn image_info(b: &[u8]) -> Option<(&'static str, u32, u32)> {
    let be32 = |i: usize| b.get(i..i + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]));
    let be16 = |i: usize| b.get(i..i + 2).map(|s| u16::from_be_bytes([s[0], s[1]]) as u32);
    let le16 = |i: usize| b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]]) as u32);
    let le24 = |i: usize| b.get(i..i + 3).map(|s| s[0] as u32 | (s[1] as u32) << 8 | (s[2] as u32) << 16);
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some(("png", be32(16)?, be32(20)?));
    }
    if b.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2;
        while i + 9 < b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = b[i + 1];
            // Start-of-frame markers carry the size; C4, C8 and CC are not frames.
            if (0xC0..=0xCF).contains(&marker) && ![0xC4, 0xC8, 0xCC].contains(&marker) {
                return Some(("jpeg", be16(i + 7)?, be16(i + 5)?));
            }
            if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) || marker == 0xFF {
                i += if marker == 0xFF { 1 } else { 2 };
                continue;
            }
            i += 2 + be16(i + 2)? as usize;
        }
        return None;
    }
    if b.len() > 30 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        return match &b[12..16] {
            b"VP8 " => Some(("webp", le16(26)? & 0x3FFF, le16(28)? & 0x3FFF)),
            b"VP8L" => {
                let bits = u32::from_le_bytes([b[21], b[22], b[23], b[24]]);
                Some(("webp", (bits & 0x3FFF) + 1, (bits >> 14 & 0x3FFF) + 1))
            }
            b"VP8X" => Some(("webp", le24(24)? + 1, le24(27)? + 1)),
            _ => None,
        };
    }
    None
}

/// A generation size near one megapixel with the source's aspect ratio, on a
/// `step` grid within 256..=2048.
pub fn size_like(w: u32, h: u32, step: i64) -> (i64, i64) {
    let scale = ((1024.0 * 1024.0) / (w.max(1) as f64 * h.max(1) as f64)).sqrt();
    let round = |v: u32| ((v as f64 * scale / step as f64).round() as i64 * step).clamp(256, 2048);
    (round(w), round(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_fields_and_files_split_on_the_boundary() {
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it night\r\n\
--XyZ\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n\x00\x01\r\n\
--XyZ\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n2\r\n--XyZ--\r\n";
        let parts = parse("multipart/form-data; boundary=XyZ", body).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!((parts[0].name.as_str(), parts[0].data.as_slice()), ("prompt", &b"make it night"[..]));
        assert_eq!(parts[1].filename.as_deref(), Some("a.png"));
        assert_eq!(parts[1].data, b"\x89PNG\r\n\x00\x01", "binary data keeps its CRLFs");
        assert_eq!(parts[2].data, b"2");
        assert!(parse("multipart/form-data", body).is_err());
        assert!(parse("multipart/form-data; boundary=\"XyZ\"", &body[..40]).is_err());
    }

    #[test]
    fn image_sizes_are_read_from_headers() {
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        png.extend(1920u32.to_be_bytes());
        png.extend(1080u32.to_be_bytes());
        assert_eq!(image_info(&png), Some(("png", 1920, 1080)));
        // SOI, an APP0 segment, then SOF0 with height 480 and width 640.
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0xE0, 0x02, 0x80, 0x03];
        assert_eq!(image_info(&jpeg), Some(("jpeg", 640, 480)));
        let mut webp = b"RIFF\0\0\0\0WEBPVP8X\x0a\0\0\0\0\0\0\0".to_vec();
        webp.extend([0xFF, 0x03, 0x00, 0xFF, 0x02, 0x00]); // 1024 x 768 (stored minus one)
        webp.extend([0u8; 8]);
        assert_eq!(image_info(&webp), Some(("webp", 1024, 768)));
        assert_eq!(image_info(b"GIF89a"), None);
        assert_eq!(size_like(1920, 1080, 32), (1376, 768));
        assert_eq!(size_like(512, 512, 32), (1024, 1024));
    }
}
