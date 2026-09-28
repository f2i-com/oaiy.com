//! PNG decoding to 8-bit RGB, as Pillow's `Image.open(f).convert("RGB")`.
//!
//! Every colour type, bit depth and Adam7 interlacing. Pillow's conversions
//! are kept as they are rather than "fixed": alpha (a channel or `tRNS`) is
//! dropped without compositing, 16-bit samples keep their high byte, 16-bit
//! grey saturates at 255 (Pillow goes through its `I;16` mode, which clips),
//! palette indices past the palette are black. Gamma and colour profiles are
//! not applied (Pillow does not either).

use oaiy_engine::{Error, Result};

use crate::{inflate, Image, MAX_PIXELS};

fn bad(m: impl std::fmt::Display) -> Error {
    Error::Format(format!("png: {m}"))
}

pub const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

/// Adam7 passes: (x0, y0, dx, dy).
const ADAM7: [(usize, usize, usize, usize); 7] = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

struct Header {
    width: usize,
    height: usize,
    depth: u8,
    color: u8,
    interlaced: bool,
}

impl Header {
    fn channels(&self) -> usize {
        match self.color {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        }
    }

    /// Bytes of one filtered scanline of `w` pixels, filter byte excluded.
    fn row_bytes(&self, w: usize) -> usize {
        (w * self.channels() * self.depth as usize).div_ceil(8)
    }

    /// Bytes between a byte and the one the filters predict it from.
    fn bpp(&self) -> usize {
        (self.channels() * self.depth as usize).div_ceil(8)
    }
}

pub fn decode(data: &[u8]) -> Result<Image> {
    if !data.starts_with(SIGNATURE) {
        return Err(bad("not a PNG file"));
    }
    let mut pos = 8;
    let mut hdr: Option<Header> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut idat = Vec::new();
    // chunks until IEND; Pillow ignores the ones it does not know, and does
    // not check CRCs of image data, so neither do we
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        let kind = &data[pos + 4..pos + 8];
        let body = data.get(pos + 8..pos + 8 + len).ok_or_else(|| bad("truncated chunk"))?;
        pos += 12 + len;
        match kind {
            b"IHDR" => {
                if body.len() < 13 {
                    return Err(bad("short IHDR"));
                }
                let h = Header {
                    width: u32::from_be_bytes(body[0..4].try_into().expect("4 bytes")) as usize,
                    height: u32::from_be_bytes(body[4..8].try_into().expect("4 bytes")) as usize,
                    depth: body[8],
                    color: body[9],
                    interlaced: body[12] == 1,
                };
                let ok = matches!((h.color, h.depth), (0, 1 | 2 | 4 | 8 | 16) | (2 | 4 | 6, 8 | 16) | (3, 1 | 2 | 4 | 8));
                if !ok {
                    return Err(bad(format!("colour type {} at bit depth {}", h.color, h.depth)));
                }
                if body[10] != 0 || body[11] != 0 || body[12] > 1 {
                    return Err(bad("unknown compression, filter or interlace method"));
                }
                if h.width == 0 || h.height == 0 || h.width * h.height > MAX_PIXELS {
                    return Err(bad(format!("image size {}x{}", h.width, h.height)));
                }
                hdr = Some(h);
            }
            b"PLTE" => {
                palette = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
    }
    let h = hdr.ok_or_else(|| bad("no IHDR"))?;
    if h.color == 3 && palette.is_empty() {
        return Err(bad("palette image without PLTE"));
    }
    let passes: Vec<(usize, usize, usize, usize, usize, usize)> = if h.interlaced {
        ADAM7
            .iter()
            .map(|&(x0, y0, dx, dy)| (x0, y0, dx, dy, (h.width + dx - 1 - x0) / dx, (h.height + dy - 1 - y0) / dy))
            .filter(|p| p.4 > 0 && p.5 > 0)
            .collect()
    } else {
        vec![(0, 0, 1, 1, h.width, h.height)]
    };
    let need: usize = passes.iter().map(|p| p.5 * (1 + h.row_bytes(p.4))).sum();
    let raw = inflate::zlib_decompress(&idat, need)?;
    if raw.len() < need {
        return Err(bad("image data is truncated"));
    }
    let mut rgb = vec![0u8; h.width * h.height * 3];
    let mut off = 0;
    let mut samples = Vec::new();
    for &(x0, y0, dx, dy, pw, ph) in &passes {
        let rb = h.row_bytes(pw);
        let mut prev = vec![0u8; rb];
        let mut cur = vec![0u8; rb];
        for py in 0..ph {
            let filter = raw[off];
            cur.copy_from_slice(&raw[off + 1..off + 1 + rb]);
            off += 1 + rb;
            unfilter(filter, &mut cur, &prev, h.bpp())?;
            let y = y0 + py * dy;
            unpack(&h, &cur, pw, &mut samples);
            let row = &mut rgb[y * h.width * 3..(y + 1) * h.width * 3];
            to_rgb(&h, &samples, &palette, |i, px| {
                let x = (x0 + i * dx) * 3;
                row[x..x + 3].copy_from_slice(&px);
            });
            std::mem::swap(&mut prev, &mut cur);
        }
    }
    Ok(Image { width: h.width, height: h.height, rgb })
}

/// Undo one scanline's filter in place.
fn unfilter(filter: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> Result<()> {
    let n = cur.len();
    match filter {
        0 => {}
        1 => {
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        2 => {
            for i in 0..n {
                cur[i] = cur[i].wrapping_add(prev[i]);
            }
        }
        3 => {
            for i in 0..n {
                let a = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
                cur[i] = cur[i].wrapping_add(((a + prev[i] as u16) / 2) as u8);
            }
        }
        4 => {
            for i in 0..n {
                let (a, c) = if i >= bpp { (cur[i - bpp] as i16, prev[i - bpp] as i16) } else { (0, 0) };
                let b = prev[i] as i16;
                let p = a + b - c;
                let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                let pred = if pa <= pb && pa <= pc {
                    a
                } else if pb <= pc {
                    b
                } else {
                    c
                };
                cur[i] = cur[i].wrapping_add(pred as u8);
            }
        }
        f => return Err(bad(format!("unknown filter type {f}"))),
    }
    Ok(())
}

/// A scanline's samples as u16 values (channels interleaved).
fn unpack(h: &Header, row: &[u8], w: usize, out: &mut Vec<u16>) {
    out.clear();
    let n = w * h.channels();
    match h.depth {
        16 => out.extend(row.chunks_exact(2).take(n).map(|c| u16::from_be_bytes([c[0], c[1]]))),
        8 => out.extend(row[..n].iter().map(|&b| b as u16)),
        d => {
            let per = 8 / d as usize;
            let mask = (1u16 << d) - 1;
            out.extend((0..n).map(|i| {
                let shift = 8 - d as usize * (i % per + 1);
                (row[i / per] as u16 >> shift) & mask
            }));
        }
    }
}

/// Pillow's decode-then-convert("RGB") for one scanline of samples.
fn to_rgb(h: &Header, s: &[u16], palette: &[[u8; 3]], mut put: impl FnMut(usize, [u8; 3])) {
    let hi = |v: u16| -> u8 {
        if h.depth == 16 {
            (v >> 8) as u8
        } else {
            v as u8
        }
    };
    match h.color {
        0 => {
            for (i, &v) in s.iter().enumerate() {
                let g = match h.depth {
                    1 => v as u8 * 255,
                    2 => v as u8 * 0x55,
                    4 => v as u8 * 0x11,
                    8 => v as u8,
                    // mode I;16 -> RGB clips instead of scaling
                    _ => v.min(255) as u8,
                };
                put(i, [g; 3]);
            }
        }
        2 => {
            for (i, p) in s.chunks_exact(3).enumerate() {
                put(i, [hi(p[0]), hi(p[1]), hi(p[2])]);
            }
        }
        3 => {
            for (i, &v) in s.iter().enumerate() {
                put(i, palette.get(v as usize).copied().unwrap_or([0; 3]));
            }
        }
        4 => {
            for (i, p) in s.chunks_exact(2).enumerate() {
                put(i, [hi(p[0]); 3]);
            }
        }
        _ => {
            for (i, p) in s.chunks_exact(4).enumerate() {
                put(i, [hi(p[0]), hi(p[1]), hi(p[2])]);
            }
        }
    }
}
