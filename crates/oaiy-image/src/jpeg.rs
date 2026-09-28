//! JPEG decoding to 8-bit RGB, as Pillow (libjpeg-turbo 3.1, default
//! settings) decodes it: baseline, extended and progressive Huffman-coded
//! 8-bit frames; the accurate integer IDCT (`JDCT_ISLOW`); "fancy"
//! triangle-filter chroma upsampling for 2x1, 1x2 and 2x2 subsampling
//! (plain replication for other integral factors); the fixed-point YCbCr
//! tables; grey, RGB, YCbCr, CMYK and YCCK (CMYK through Pillow's inverted
//! `CMYK;I` raw mode and its `cmyk2rgb`). Arithmetic coding, lossless,
//! hierarchical and 12-bit frames are refused. EXIF orientation is not
//! applied (Pillow's `open` does not either).
//!
//! Every scan decodes into per-component coefficient arrays, then each
//! block goes through the IDCT once; for a sequential file that gives what
//! libjpeg's one-pass decoder gives. Corrupt entropy data is handled the way
//! libjpeg does: a bad Huffman code reads as 0, and once the data runs out
//! the rest of the scan stays zero.

use oaiy_engine::{Error, Result};

use crate::{Image, MAX_PIXELS};

fn bad(m: impl std::fmt::Display) -> Error {
    Error::Format(format!("jpeg: {m}"))
}

/// Zigzag position -> natural (row-major) coefficient index, with libjpeg's
/// 16 extra entries so a corrupt run length writes coefficient 63 instead
/// of past the block.
const NATURAL: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22,
    15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];

const FAST_BITS: u32 = 9;

/// A DHT table, decoded the way libjpeg's `jpeg_huff_decode` does, with a
/// lookup for codes of up to [`FAST_BITS`] bits.
#[derive(Clone)]
struct Huffman {
    /// (length << 8 | symbol) for codes of <= FAST_BITS bits, 0 otherwise.
    fast: Vec<u16>,
    /// Largest code of each length (-1 for none); [17] a sentinel.
    maxcode: [i32; 18],
    /// Symbol index = code + valoffset[length].
    valoffset: [i32; 17],
    vals: Vec<u8>,
}

impl Huffman {
    fn new(counts: &[u8; 16], vals: &[u8], dc: bool) -> Result<Huffman> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total > 256 || total > vals.len() {
            return Err(bad("bad Huffman table"));
        }
        if dc && vals[..total].iter().any(|&v| v > 15) {
            return Err(bad("bad DC Huffman table"));
        }
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 17];
        let mut fast = vec![0u16; 1 << FAST_BITS];
        let (mut code, mut k) = (0i32, 0usize);
        for len in 1..=16usize {
            let n = counts[len - 1] as usize;
            if n > 0 {
                valoffset[len] = k as i32 - code;
                for _ in 0..n {
                    // codes must fit in `len` bits, and none may be all ones
                    if code >= (1 << len) - 1 {
                        return Err(bad("bad Huffman table"));
                    }
                    if len as u32 <= FAST_BITS {
                        let shift = FAST_BITS - len as u32;
                        let base = (code as usize) << shift;
                        for e in &mut fast[base..base + (1 << shift)] {
                            *e = (len as u16) << 8 | u16::from(vals[k]);
                        }
                    }
                    code += 1;
                    k += 1;
                }
                maxcode[len] = code - 1;
            }
            code <<= 1;
        }
        maxcode[17] = 0xFFFFF;
        Ok(Huffman { fast, maxcode, valoffset, vals: vals[..total].to_vec() })
    }
}

/// MSB-first reader over entropy-coded data: un-stuffs 0xFF00, stops at a
/// marker and reads zeros from there (libjpeg's "fill with zeros").
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    /// Valid bits, left-aligned.
    buf: u64,
    cnt: u32,
    /// Zero bits at the bottom of `buf` that did not come from the data.
    pad: u32,
    marker: bool,
    /// Data ran out mid-scan: the rest of it is skipped.
    starved: bool,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Bits<'a> {
        Bits { data, pos, buf: 0, cnt: 0, pad: 0, marker: false, starved: false }
    }

    fn fill(&mut self) {
        while self.cnt <= 56 {
            let mut b = 0u8;
            if !self.marker {
                match self.data.get(self.pos) {
                    Some(0xFF) => {
                        // stuffed 0xFF00 is data; fill bytes 0xFFFF lead to a marker
                        if self.data.get(self.pos + 1) == Some(&0) {
                            self.pos += 2;
                            b = 0xFF;
                        } else {
                            self.marker = true;
                        }
                    }
                    Some(&v) => {
                        self.pos += 1;
                        b = v;
                    }
                    None => self.marker = true,
                }
            }
            if self.marker {
                self.pad += 8;
            }
            self.buf |= u64::from(b) << (56 - self.cnt);
            self.cnt += 8;
        }
    }

    #[inline]
    fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.cnt < n {
            self.fill();
        }
        let v = (self.buf >> (64 - n)) as u32;
        self.buf <<= n;
        self.cnt -= n;
        if self.cnt < self.pad {
            self.starved = true;
            self.pad = self.cnt;
        }
        v
    }

    /// A value of `s` bits, sign-extended the JPEG way (HUFF_EXTEND).
    #[inline]
    fn extend(&mut self, s: u32) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.bits(s) as i32;
        if v < 1 << (s - 1) {
            v - (1 << s) + 1
        } else {
            v
        }
    }

    #[inline]
    fn decode(&mut self, h: &Huffman) -> u32 {
        if self.cnt < 16 {
            self.fill();
        }
        let e = h.fast[(self.buf >> (64 - FAST_BITS)) as usize];
        if e != 0 {
            let len = u32::from(e >> 8);
            self.bits(len);
            return u32::from(e & 0xFF);
        }
        // one bit at a time past the fast table (libjpeg's jpeg_huff_decode)
        let mut l = FAST_BITS as usize + 1;
        let mut code = self.bits(l as u32) as i32;
        while code > h.maxcode[l] {
            code = code << 1 | self.bits(1) as i32;
            l += 1;
        }
        if l > 16 {
            return 0; // a bad code: libjpeg fakes a zero
        }
        h.vals.get((code + h.valoffset[l]) as usize).map_or(0, |&v| u32::from(v))
    }

    /// Discard buffered bits and step over the next marker if it is RSTn.
    fn restart(&mut self) {
        if !self.marker {
            // skip whatever is left of this interval up to its marker
            while self.pos + 1 < self.data.len() && !(self.data[self.pos] == 0xFF && self.data[self.pos + 1] != 0 && self.data[self.pos + 1] != 0xFF) {
                self.pos += 1;
            }
        }
        while self.data.get(self.pos) == Some(&0xFF) && self.data.get(self.pos + 1) == Some(&0xFF) {
            self.pos += 1;
        }
        if self.data.get(self.pos) == Some(&0xFF) && matches!(self.data.get(self.pos + 1), Some(0xD0..=0xD7)) {
            self.pos += 2;
            self.marker = false;
            self.starved = false;
        } else {
            // not a restart marker: leave it for the marker parser, and read
            // zeros for the rest of the scan
            self.marker = true;
        }
        self.buf = 0;
        self.cnt = 0;
        self.pad = 0;
    }

    /// Position of the marker that ends the scan.
    fn end(&self) -> usize {
        let mut p = self.pos;
        while p + 1 < self.data.len() && !(self.data[p] == 0xFF && self.data[p + 1] != 0 && !(0xD0..=0xD7).contains(&self.data[p + 1])) {
            p += 1;
        }
        p
    }
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Quantization table, latched at the component's first scan.
    quant: Option<[u16; 64]>,
    /// Blocks covering the component's samples.
    bw: usize,
    bh: usize,
    /// Blocks per row of `coefs` (whole MCUs).
    stride: usize,
    coefs: Vec<i16>,
    dc_pred: i32,
}

struct Frame {
    width: usize,
    height: usize,
    progressive: bool,
    comps: Vec<Component>,
    hmax: usize,
    vmax: usize,
    mcus_x: usize,
    mcus_y: usize,
}

struct Scan {
    comps: Vec<(usize, usize, usize)>, // (component index, DC table, AC table)
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

pub fn decode(data: &[u8]) -> Result<Image> {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return Err(bad("not a JPEG file"));
    }
    let mut pos = 2;
    let mut qt: [Option<[u16; 64]>; 4] = [None; 4];
    let mut dc: [Option<Huffman>; 4] = [None, None, None, None];
    let mut ac: [Option<Huffman>; 4] = [None, None, None, None];
    let mut frame: Option<Frame> = None;
    let mut restart_interval = 0usize;
    let (mut jfif, mut adobe) = (false, None::<u8>);
    let mut scans = 0;
    loop {
        // next marker, skipping junk and fill bytes as libjpeg's next_marker does
        while pos < data.len() && data[pos] != 0xFF {
            pos += 1;
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        let Some(&m) = data.get(pos) else { break };
        pos += 1;
        match m {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 => break,
            _ => {}
        }
        let len = match data.get(pos..pos + 2) {
            Some(b) => u16::from_be_bytes([b[0], b[1]]) as usize,
            None => break,
        };
        if len < 2 {
            return Err(bad("bad marker length"));
        }
        let seg = data.get(pos + 2..pos + len).ok_or_else(|| bad("truncated marker segment"))?;
        pos += len;
        match m {
            0xC0..=0xC2 => {
                if frame.is_some() {
                    return Err(bad("more than one frame"));
                }
                frame = Some(read_frame(seg, m == 0xC2)?);
            }
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(Error::Unsupported(format!("jpeg: SOF{} frames (lossless, hierarchical or arithmetic-coded)", m - 0xC0)));
            }
            0xC4 => read_huffman(seg, &mut dc, &mut ac)?,
            0xDB => read_quant(seg, &mut qt)?,
            0xDD => {
                if seg.len() < 2 {
                    return Err(bad("short DRI"));
                }
                restart_interval = u16::from_be_bytes([seg[0], seg[1]]) as usize;
            }
            0xE0 => jfif |= seg.len() >= 5 && &seg[..5] == b"JFIF\0",
            0xEE => {
                if seg.len() >= 12 && &seg[..5] == b"Adobe" {
                    adobe = Some(seg[11]);
                }
            }
            0xDA => {
                let f = frame.as_mut().ok_or_else(|| bad("scan before frame"))?;
                let scan = read_scan(seg, f)?;
                for &(ci, _, _) in &scan.comps {
                    let c = &mut f.comps[ci];
                    if c.quant.is_none() {
                        c.quant = Some(qt[c.tq].ok_or_else(|| bad("missing quantization table"))?);
                    }
                }
                let mut bits = Bits::new(data, pos);
                decode_scan(f, &scan, &dc, &ac, restart_interval, &mut bits)?;
                pos = bits.end();
                scans += 1;
            }
            0xDC => return Err(Error::Unsupported("jpeg: DNL marker".into())),
            _ => {}
        }
    }
    let f = frame.ok_or_else(|| bad("no frame"))?;
    if scans == 0 {
        return Err(bad("no scan"));
    }
    output(&f, jfif, adobe)
}

fn read_frame(s: &[u8], progressive: bool) -> Result<Frame> {
    if s.len() < 6 {
        return Err(bad("short SOF"));
    }
    if s[0] != 8 {
        return Err(Error::Unsupported(format!("jpeg: {}-bit samples", s[0])));
    }
    let height = u16::from_be_bytes([s[1], s[2]]) as usize;
    let width = u16::from_be_bytes([s[3], s[4]]) as usize;
    let n = s[5] as usize;
    if height == 0 {
        return Err(Error::Unsupported("jpeg: height defined by DNL".into()));
    }
    if width == 0 || width * height > MAX_PIXELS {
        return Err(bad(format!("image size {width}x{height}")));
    }
    if !matches!(n, 1 | 3 | 4) || s.len() < 6 + 3 * n {
        return Err(Error::Unsupported(format!("jpeg: {n} components")));
    }
    let mut comps = Vec::with_capacity(n);
    for c in s[6..6 + 3 * n].chunks_exact(3) {
        let (h, v, tq) = ((c[1] >> 4) as usize, (c[1] & 15) as usize, c[2] as usize);
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) || tq > 3 {
            return Err(bad("bad component"));
        }
        comps.push(Component { id: c[0], h, v, tq, quant: None, bw: 0, bh: 0, stride: 0, coefs: Vec::new(), dc_pred: 0 });
    }
    let hmax = comps.iter().map(|c| c.h).max().expect("components");
    let vmax = comps.iter().map(|c| c.v).max().expect("components");
    let mcus_x = width.div_ceil(8 * hmax);
    let mcus_y = height.div_ceil(8 * vmax);
    for c in &mut comps {
        c.bw = (width * c.h).div_ceil(hmax).div_ceil(8);
        c.bh = (height * c.v).div_ceil(vmax).div_ceil(8);
        c.stride = mcus_x * c.h;
        c.coefs = vec![0; c.stride * mcus_y * c.v * 64];
    }
    Ok(Frame { width, height, progressive, comps, hmax, vmax, mcus_x, mcus_y })
}

fn read_huffman(mut s: &[u8], dc: &mut [Option<Huffman>; 4], ac: &mut [Option<Huffman>; 4]) -> Result<()> {
    while s.len() >= 17 {
        let (class, id) = (s[0] >> 4, (s[0] & 15) as usize);
        let counts: [u8; 16] = s[1..17].try_into().expect("16 bytes");
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        let vals = s.get(17..17 + total).ok_or_else(|| bad("short DHT"))?;
        if id > 3 || class > 1 {
            return Err(bad("bad DHT table id"));
        }
        let t = Huffman::new(&counts, vals, class == 0)?;
        if class == 0 {
            dc[id] = Some(t);
        } else {
            ac[id] = Some(t);
        }
        s = &s[17 + total..];
    }
    Ok(())
}

fn read_quant(mut s: &[u8], qt: &mut [Option<[u16; 64]>; 4]) -> Result<()> {
    while !s.is_empty() {
        let (prec, id) = (s[0] >> 4, (s[0] & 15) as usize);
        if id > 3 {
            return Err(bad("bad DQT table id"));
        }
        let n = if prec == 0 { 64 } else { 128 };
        let body = s.get(1..1 + n).ok_or_else(|| bad("short DQT"))?;
        // tables arrive in zigzag order; keep them in natural order
        let mut t = [0u16; 64];
        for k in 0..64 {
            t[NATURAL[k]] = if prec == 0 { u16::from(body[k]) } else { u16::from_be_bytes([body[2 * k], body[2 * k + 1]]) };
        }
        qt[id] = Some(t);
        s = &s[1 + n..];
    }
    Ok(())
}

fn read_scan(s: &[u8], f: &mut Frame) -> Result<Scan> {
    let n = *s.first().ok_or_else(|| bad("short SOS"))? as usize;
    if n == 0 || n > 4 || s.len() < 1 + 2 * n + 3 {
        return Err(bad("bad SOS"));
    }
    let mut comps = Vec::with_capacity(n);
    for c in s[1..1 + 2 * n].chunks_exact(2) {
        let ci = f.comps.iter().position(|x| x.id == c[0]).ok_or_else(|| bad("scan names an unknown component"))?;
        comps.push((ci, (c[1] >> 4) as usize, (c[1] & 15) as usize));
    }
    let p = &s[1 + 2 * n..];
    let scan = Scan { comps, ss: p[0] as usize, se: p[1] as usize, ah: u32::from(p[2] >> 4), al: u32::from(p[2] & 15) };
    if f.progressive {
        let dc_scan = scan.ss == 0;
        if (dc_scan && scan.se != 0) || (!dc_scan && (scan.se < scan.ss || scan.se > 63 || n != 1)) || scan.al > 13 {
            return Err(bad("bad progressive scan parameters"));
        }
    }
    if scan.comps.iter().map(|&(ci, _, _)| f.comps[ci].h * f.comps[ci].v).sum::<usize>() > 10 && n > 1 {
        return Err(bad("too many blocks in an MCU"));
    }
    Ok(scan)
}

fn decode_scan(f: &mut Frame, scan: &Scan, dc: &[Option<Huffman>; 4], ac: &[Option<Huffman>; 4], restart_interval: usize, bits: &mut Bits) -> Result<()> {
    let seq = !f.progressive;
    let need_dc = seq || (scan.ss == 0 && scan.ah == 0);
    let need_ac = seq || scan.ss > 0;
    let mut tables = Vec::with_capacity(scan.comps.len());
    for &(ci, td, ta) in &scan.comps {
        let d = if need_dc { Some(dc.get(td).and_then(|t| t.as_ref()).ok_or_else(|| bad("missing DC Huffman table"))?) } else { None };
        let a = if need_ac { Some(ac.get(ta).and_then(|t| t.as_ref()).ok_or_else(|| bad("missing AC Huffman table"))?) } else { None };
        tables.push((ci, d, a));
    }
    for c in &mut f.comps {
        c.dc_pred = 0;
    }
    let mut eobrun = 0u32;
    let block = |c: &mut Component, idx: usize, d: Option<&Huffman>, a: Option<&Huffman>, bits: &mut Bits, eobrun: &mut u32| {
        let b = &mut c.coefs[idx * 64..idx * 64 + 64];
        if seq {
            let d = d.expect("DC table");
            let s = bits.decode(d);
            let diff = bits.extend(s);
            c.dc_pred = c.dc_pred.wrapping_add(diff);
            b[0] = c.dc_pred as i16;
            let a = a.expect("AC table");
            let mut k = 1;
            while k < 64 {
                let rs = bits.decode(a);
                let (r, s) = ((rs >> 4) as usize, rs & 15);
                if s != 0 {
                    k += r;
                    b[NATURAL[k]] = bits.extend(s) as i16;
                } else if r != 15 {
                    break;
                } else {
                    k += 15;
                }
                k += 1;
            }
        } else if scan.ss == 0 {
            if scan.ah == 0 {
                let s = bits.decode(d.expect("DC table"));
                let diff = bits.extend(s);
                c.dc_pred = c.dc_pred.wrapping_add(diff);
                b[0] = (c.dc_pred << scan.al) as i16;
            } else if bits.bits(1) != 0 {
                b[0] |= (1 << scan.al) as i16;
            }
        } else if scan.ah == 0 {
            ac_first(b, a.expect("AC table"), scan, bits, eobrun);
        } else {
            ac_refine(b, a.expect("AC table"), scan, bits, eobrun);
        }
    };
    let mut mcu = 0usize;
    let restart = |mcu: usize, bits: &mut Bits, f: &mut Frame, eobrun: &mut u32| {
        if restart_interval > 0 && mcu > 0 && mcu.is_multiple_of(restart_interval) {
            bits.restart();
            *eobrun = 0;
            for c in &mut f.comps {
                c.dc_pred = 0;
            }
        }
    };
    if scan.comps.len() == 1 {
        // non-interleaved: one block per MCU, only the blocks the samples cover
        let (ci, d, a) = tables[0];
        let (bw, bh, stride) = (f.comps[ci].bw, f.comps[ci].bh, f.comps[ci].stride);
        for by in 0..bh {
            for bx in 0..bw {
                restart(mcu, bits, f, &mut eobrun);
                mcu += 1;
                if bits.starved {
                    continue;
                }
                block(&mut f.comps[ci], by * stride + bx, d, a, bits, &mut eobrun);
            }
        }
    } else {
        for my in 0..f.mcus_y {
            for mx in 0..f.mcus_x {
                restart(mcu, bits, f, &mut eobrun);
                mcu += 1;
                if bits.starved {
                    continue;
                }
                for &(ci, d, a) in &tables {
                    let c = &mut f.comps[ci];
                    for v in 0..c.v {
                        for h in 0..c.h {
                            let idx = (my * c.v + v) * c.stride + mx * c.h + h;
                            block(c, idx, d, a, bits, &mut eobrun);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// First pass over an AC band (libjpeg's decode_mcu_AC_first).
fn ac_first(b: &mut [i16], a: &Huffman, scan: &Scan, bits: &mut Bits, eobrun: &mut u32) {
    if *eobrun > 0 {
        *eobrun -= 1;
        return;
    }
    let mut k = scan.ss;
    while k <= scan.se {
        let rs = bits.decode(a);
        let (r, s) = ((rs >> 4) as usize, rs & 15);
        if s != 0 {
            k += r;
            b[NATURAL[k]] = (bits.extend(s) << scan.al) as i16;
        } else if r == 15 {
            k += 15;
        } else {
            *eobrun = 1 << r;
            if r > 0 {
                *eobrun += bits.bits(r as u32);
            }
            *eobrun -= 1;
            break;
        }
        k += 1;
    }
}

/// A refinement pass over an AC band (libjpeg's decode_mcu_AC_refine).
fn ac_refine(b: &mut [i16], a: &Huffman, scan: &Scan, bits: &mut Bits, eobrun: &mut u32) {
    let p1 = 1i16 << scan.al;
    let m1 = (-1i16) << scan.al;
    let refine = |coef: &mut i16, bits: &mut Bits| {
        if bits.bits(1) != 0 && (*coef & p1) == 0 {
            *coef = coef.wrapping_add(if *coef >= 0 { p1 } else { m1 });
        }
    };
    let mut k = scan.ss;
    if *eobrun == 0 {
        while k <= scan.se {
            let rs = bits.decode(a);
            let mut r = (rs >> 4) as i32;
            let mut s = 0i16;
            if rs & 15 != 0 {
                s = if bits.bits(1) != 0 { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = 1 << r;
                if r > 0 {
                    *eobrun += bits.bits(r as u32);
                }
                break;
            }
            // step over nonzero coefficients (refining them) and r zeros
            loop {
                let z = NATURAL[k];
                if b[z] != 0 {
                    refine(&mut b[z], bits);
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
                if k > scan.se {
                    break;
                }
            }
            if s != 0 {
                b[NATURAL[k]] = s;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= scan.se {
            let z = NATURAL[k];
            if b[z] != 0 {
                refine(&mut b[z], bits);
            }
            k += 1;
        }
        *eobrun -= 1;
    }
}

// ---------------------------------------------------------------- IDCT

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

#[inline]
fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// libjpeg's post-IDCT range limit: `x & 1023` indexes a table that is
/// x + 128 clamped for |x| < 512 and wraps beyond (only corrupt data gets
/// there).
#[inline]
fn idct_limit(x: i64) -> u8 {
    let i = (x & 1023) as i32;
    match i {
        0..=127 => (i + 128) as u8,
        128..=511 => 255,
        512..=895 => 0,
        _ => (i - 896) as u8,
    }
}

/// One IDCT pass over 8 values (even/odd parts of jidctint.c): returns the
/// eight outputs before descaling.
#[inline]
fn idct_1d(v: [i64; 8]) -> [i64; 8] {
    let (z2, z3) = (v[2], v[6]);
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * -FIX_1_847759065;
    let tmp3 = z1 + z2 * FIX_0_765366865;
    let tmp0 = (v[0] + v[4]) << CONST_BITS;
    let tmp1 = (v[0] - v[4]) << CONST_BITS;
    let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);

    let (mut t0, mut t1, mut t2, mut t3) = (v[7], v[5], v[3], v[1]);
    let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
    let z5 = (z3 + z4) * FIX_1_175875602;
    t0 *= FIX_0_298631336;
    t1 *= FIX_2_053119869;
    t2 *= FIX_3_072711026;
    t3 *= FIX_1_501321110;
    let z1 = z1 * -FIX_0_899976223;
    let z2 = z2 * -FIX_2_562915447;
    let z3 = z3 * -FIX_1_961570560 + z5;
    let z4 = z4 * -FIX_0_390180644 + z5;
    t0 += z1 + z3;
    t1 += z2 + z4;
    t2 += z2 + z3;
    t3 += z1 + z4;
    [tmp10 + t3, tmp11 + t2, tmp12 + t1, tmp13 + t0, tmp13 - t0, tmp12 - t1, tmp11 - t2, tmp10 - t3]
}

/// jpeg_idct_islow: dequantize, 2-D IDCT, level shift and clamp into an
/// 8x8 block of `out` (row stride `stride`).
fn idct_block(coef: &[i16], q: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i64; 64];
    for col in 0..8 {
        let v: [i64; 8] = std::array::from_fn(|r| i64::from(coef[r * 8 + col]) * i64::from(q[r * 8 + col]));
        let o = idct_1d(v);
        for r in 0..8 {
            // the reference's int workspace
            ws[r * 8 + col] = i64::from(descale(o[r], CONST_BITS - PASS1_BITS) as i32);
        }
    }
    for r in 0..8 {
        let v: [i64; 8] = std::array::from_fn(|c| ws[r * 8 + c]);
        let o = idct_1d(v);
        let row = &mut out[r * stride..r * stride + 8];
        for c in 0..8 {
            row[c] = idct_limit(descale(o[c], CONST_BITS + PASS1_BITS + 3));
        }
    }
}

// ---------------------------------------------------------------- output

/// A component's samples at its own resolution.
struct Plane {
    w: usize,
    h: usize,
    stride: usize,
    px: Vec<u8>,
}

impl Plane {
    #[inline]
    fn at(&self, x: usize, y: usize) -> i32 {
        i32::from(self.px[y * self.stride + x])
    }
}

fn output(f: &Frame, jfif: bool, adobe: Option<u8>) -> Result<Image> {
    let (w, h) = (f.width, f.height);
    let mut full: Vec<Vec<u8>> = Vec::with_capacity(f.comps.len());
    for c in &f.comps {
        let q = c.quant.unwrap_or([0; 64]); // a component no scan named decodes as mid-grey
        let stride = c.bw * 8;
        let mut px = vec![0u8; stride * c.bh * 8];
        for by in 0..c.bh {
            for bx in 0..c.bw {
                let idx = by * c.stride + bx;
                idct_block(&c.coefs[idx * 64..idx * 64 + 64], &q, &mut px[by * 8 * stride + bx * 8..], stride);
            }
        }
        let plane = Plane { w: (w * c.h).div_ceil(f.hmax), h: (h * c.v).div_ceil(f.vmax), stride, px };
        if !f.hmax.is_multiple_of(c.h) || !f.vmax.is_multiple_of(c.v) {
            return Err(Error::Unsupported("jpeg: fractional sampling factors".into()));
        }
        full.push(upsample(&plane, f.hmax / c.h, f.vmax / c.v, w, h));
    }
    let n = w * h;
    let mut rgb = vec![0u8; n * 3];
    match full.len() {
        1 => {
            for (o, &y) in rgb.chunks_exact_mut(3).zip(&full[0]) {
                o.fill(y);
            }
        }
        3 => {
            let ids: Vec<u8> = f.comps.iter().map(|c| c.id).collect();
            let is_rgb = if jfif {
                false
            } else if let Some(t) = adobe {
                t == 0
            } else {
                ids == [b'R', b'G', b'B']
            };
            if is_rgb {
                for i in 0..n {
                    rgb[i * 3..i * 3 + 3].copy_from_slice(&[full[0][i], full[1][i], full[2][i]]);
                }
            } else {
                let t = YccTables::new();
                for i in 0..n {
                    let p = t.rgb(full[0][i], full[1][i], full[2][i]);
                    rgb[i * 3..i * 3 + 3].copy_from_slice(&p);
                }
            }
        }
        _ => {
            // libjpeg gives CMYK (YCCK converted when Adobe says so); Pillow
            // reads it as inverted CMYK and converts with cmyk2rgb
            let ycck = matches!(adobe, Some(t) if t != 0);
            let t = YccTables::new();
            let muldiv255 = |a: i32, b: i32| {
                let t = a * b + 128;
                ((t >> 8) + t) >> 8
            };
            for i in 0..n {
                let (c, m, y) = if ycck {
                    let p = t.rgb_unclamped(full[0][i], full[1][i], full[2][i]);
                    (clamp(255 - p[0]), clamp(255 - p[1]), clamp(255 - p[2]))
                } else {
                    (full[0][i], full[1][i], full[2][i])
                };
                let k = full[3][i];
                let (c, m, y, k) = (255 - i32::from(c), 255 - i32::from(m), 255 - i32::from(y), 255 - i32::from(k));
                let nk = 255 - k;
                for (o, v) in rgb[i * 3..i * 3 + 3].iter_mut().zip([c, m, y]) {
                    *o = clamp(nk - muldiv255(v, nk));
                }
            }
        }
    }
    Ok(Image { width: w, height: h, rgb })
}

#[inline]
fn clamp(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// jdsample.c's upsamplers, from a component plane to `w` x `h`. Rows past
/// the component's last real row read as that row (jdmainct's bottom
/// context); the first row's "row above" is itself.
fn upsample(p: &Plane, fx: usize, fy: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let last = p.h - 1;
    let fancy_h2 = p.w > 2;
    match (fx, fy) {
        (1, 1) => {
            for y in 0..h {
                out[y * w..(y + 1) * w].copy_from_slice(&p.px[y * p.stride..y * p.stride + w]);
            }
        }
        (2, 1) if fancy_h2 => {
            let mut row = vec![0u8; 2 * p.w];
            for y in 0..h {
                let src = &p.px[y * p.stride..y * p.stride + p.w];
                h2v1_row(src, &mut row);
                out[y * w..(y + 1) * w].copy_from_slice(&row[..w]);
            }
        }
        (1, 2) => {
            for y in 0..h {
                let i = y / 2;
                let (near, far, bias) = if y % 2 == 0 { (i, i.saturating_sub(1), 1) } else { (i, (i + 1).min(last), 2) };
                for x in 0..w {
                    out[y * w + x] = ((p.at(x, near) * 3 + p.at(x, far) + bias) >> 2) as u8;
                }
            }
        }
        (2, 2) if fancy_h2 => {
            let mut sums = vec![0i32; p.w];
            let mut row = vec![0u8; 2 * p.w];
            for y in 0..h {
                let i = y / 2;
                let far = if y % 2 == 0 { i.saturating_sub(1) } else { (i + 1).min(last) };
                for (x, s) in sums.iter_mut().enumerate() {
                    *s = p.at(x, i) * 3 + p.at(x, far);
                }
                h2v2_row(&sums, &mut row);
                out[y * w..(y + 1) * w].copy_from_slice(&row[..w]);
            }
        }
        _ => {
            // int_upsample (and h2v1/h2v2 without fancy): replicate
            for y in 0..h {
                let sy = y / fy;
                for x in 0..w {
                    out[y * w + x] = p.px[sy * p.stride + x / fx];
                }
            }
        }
    }
    out
}

/// h2v1_fancy_upsample over one row of `src.len()` > 2 samples.
fn h2v1_row(src: &[u8], out: &mut [u8]) {
    let n = src.len();
    let s = |i: usize| i32::from(src[i]);
    out[0] = src[0];
    out[1] = ((s(0) * 3 + s(1) + 2) >> 2) as u8;
    for i in 1..n - 1 {
        let v = s(i) * 3;
        out[2 * i] = ((v + s(i - 1) + 1) >> 2) as u8;
        out[2 * i + 1] = ((v + s(i + 1) + 2) >> 2) as u8;
    }
    out[2 * n - 2] = ((s(n - 1) * 3 + s(n - 2) + 1) >> 2) as u8;
    out[2 * n - 1] = src[n - 1];
}

/// h2v2_fancy_upsample's horizontal half, over column sums (3 * near row +
/// far row).
fn h2v2_row(sums: &[i32], out: &mut [u8]) {
    let n = sums.len();
    out[0] = ((sums[0] * 4 + 8) >> 4) as u8;
    out[1] = ((sums[0] * 3 + sums[1] + 7) >> 4) as u8;
    for i in 1..n - 1 {
        out[2 * i] = ((sums[i] * 3 + sums[i - 1] + 8) >> 4) as u8;
        out[2 * i + 1] = ((sums[i] * 3 + sums[i + 1] + 7) >> 4) as u8;
    }
    out[2 * n - 2] = ((sums[n - 1] * 3 + sums[n - 2] + 8) >> 4) as u8;
    out[2 * n - 1] = ((sums[n - 1] * 4 + 7) >> 4) as u8;
}

/// jdcolor.c's fixed-point YCbCr -> RGB tables (SCALEBITS 16).
struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i64; 256],
    cb_g: [i64; 256],
}

impl YccTables {
    fn new() -> YccTables {
        const ONE_HALF: i64 = 1 << 15;
        let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
        let mut t = YccTables { cr_r: [0; 256], cb_b: [0; 256], cr_g: [0; 256], cb_g: [0; 256] };
        for i in 0..256 {
            let x = i as i64 - 128;
            t.cr_r[i] = ((fix(1.40200) * x + ONE_HALF) >> 16) as i32;
            t.cb_b[i] = ((fix(1.77200) * x + ONE_HALF) >> 16) as i32;
            t.cr_g[i] = -fix(0.71414) * x;
            t.cb_g[i] = -fix(0.34414) * x + ONE_HALF;
        }
        t
    }

    #[inline]
    fn rgb_unclamped(&self, y: u8, cb: u8, cr: u8) -> [i32; 3] {
        let y = i32::from(y);
        let (cb, cr) = (cb as usize, cr as usize);
        [y + self.cr_r[cr], y + ((self.cb_g[cb] + self.cr_g[cr]) >> 16) as i32, y + self.cb_b[cb]]
    }

    #[inline]
    fn rgb(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        self.rgb_unclamped(y, cb, cr).map(clamp)
    }
}
