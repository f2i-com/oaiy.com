//! DEFLATE (RFC 1951) and the zlib wrapper around it (RFC 1950), for PNG.
//!
//! Huffman codes decode through one flat table per code, indexed by the
//! next `max_len` input bits (deflate packs codes LSB first, so the table is
//! indexed by bit-reversed codes). A dynamic block rebuilds at most 32K
//! entries, small next to the output a block carries.

use oaiy_engine::{Error, Result};

fn bad(m: &str) -> Error {
    Error::Format(format!("deflate: {m}"))
}

/// LSB-first bit reader. Past the end of the input it reads zeros and
/// counts them, so a truncated stream is caught (`overrun`) instead of read
/// out of bounds.
struct Bits<'a> {
    src: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
}

impl<'a> Bits<'a> {
    /// Top the buffer up to at least 56 bits.
    #[inline]
    fn refill(&mut self) {
        if self.pos + 8 <= self.src.len() {
            // whole bytes that fit; the bits above them are the next byte's
            // low bits, which the next refill ORs in again unchanged
            let v = u64::from_le_bytes(self.src[self.pos..self.pos + 8].try_into().expect("8 bytes"));
            self.buf |= v << self.cnt;
            let n = (63 - self.cnt) / 8;
            self.pos += n as usize;
            self.cnt += n * 8;
        } else {
            while self.cnt <= 56 {
                let b = self.src.get(self.pos).copied().unwrap_or(0);
                self.buf |= u64::from(b) << self.cnt;
                self.pos += 1;
                self.cnt += 8;
            }
        }
    }

    #[inline]
    fn take(&mut self, n: u32) -> u32 {
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.cnt -= n;
        v
    }

    /// Bytes of input consumed so far (a partly used byte counts).
    fn consumed(&self) -> usize {
        self.pos - (self.cnt / 8) as usize
    }

    fn overrun(&self) -> bool {
        self.consumed() > self.src.len()
    }
}

/// A canonical Huffman code as a flat lookup table: entry = symbol << 4 |
/// code length; 0 marks bit patterns no code starts with.
struct Huffman {
    table: Vec<u16>,
    bits: u32,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Huffman> {
        let max = lengths.iter().copied().max().unwrap_or(0) as u32;
        if max == 0 {
            // no symbols: any use is an error (a block may legally carry an
            // empty distance code if it has no matches)
            return Ok(Huffman { table: vec![0; 2], bits: 1 });
        }
        let mut count = [0u32; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut left = 1i32;
        for &c in &count[1..] {
            left = (left << 1) - c as i32;
            if left < 0 {
                return Err(bad("over-subscribed Huffman code"));
            }
        }
        // incomplete codes are accepted (zlib does too for a single
        // distance code); the unused patterns stay 0 and fail when read
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for len in 1..16 {
            code = (code + count[len - 1]) << 1;
            next[len] = code;
        }
        let size = 1usize << max;
        let mut table = vec![0u16; size];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = l as u32;
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = (c.reverse_bits() >> (32 - l)) as usize;
            let entry = (sym as u16) << 4 | l as u16;
            let mut i = rev;
            while i < size {
                table[i] = entry;
                i += 1 << l;
            }
        }
        Ok(Huffman { table, bits: max })
    }

    #[inline]
    fn decode(&self, b: &mut Bits) -> Result<u32> {
        let e = self.table[(b.buf & ((1u64 << self.bits) - 1)) as usize];
        let len = u32::from(e & 15);
        if len == 0 {
            return Err(bad("invalid Huffman code"));
        }
        b.buf >>= len;
        b.cnt -= len;
        Ok(u32::from(e >> 4))
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] =
    [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
/// Order the code-length code lengths arrive in.
const CLEN_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// How [`inflate`] ended.
#[derive(Debug, PartialEq, Eq)]
pub enum End {
    /// The final block ended; `usize` input bytes were consumed.
    Stream(usize),
    /// The output reached the limit first; the rest was not decoded.
    Limit,
}

/// Decompress raw DEFLATE data from `src`, appending to `out` until the
/// final block ends or `out` holds `limit` bytes.
pub fn inflate(src: &[u8], out: &mut Vec<u8>, limit: usize) -> Result<End> {
    let mut b = Bits { src, pos: 0, buf: 0, cnt: 0 };
    let mut fixed: Option<(Huffman, Huffman)> = None;
    loop {
        b.refill();
        let last = b.take(1) == 1;
        match b.take(2) {
            0 => {
                // stored: skip to a byte boundary, rewind the buffered bytes
                b.take(b.cnt % 8);
                b.pos -= (b.cnt / 8) as usize;
                b.buf = 0;
                b.cnt = 0;
                let hdr = src.get(b.pos..b.pos + 4).ok_or_else(|| bad("truncated stored block"))?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
                if len as u16 != !u16::from_le_bytes([hdr[2], hdr[3]]) {
                    return Err(bad("stored block length check failed"));
                }
                b.pos += 4;
                let data = src.get(b.pos..b.pos + len).ok_or_else(|| bad("truncated stored block"))?;
                b.pos += len;
                let room = limit - out.len();
                if len >= room {
                    out.extend_from_slice(&data[..room]);
                    if len > room || !last {
                        return Ok(End::Limit);
                    }
                } else {
                    out.extend_from_slice(data);
                }
            }
            1 => {
                let (lit, dist) = fixed.get_or_insert_with(|| {
                    let mut l = [8u8; 288];
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    (Huffman::new(&l).expect("fixed code"), Huffman::new(&[5u8; 30]).expect("fixed code"))
                });
                if codes(&mut b, lit, dist, out, limit)? {
                    return Ok(End::Limit);
                }
            }
            2 => {
                let (lit, dist) = dynamic_codes(&mut b)?;
                if codes(&mut b, &lit, &dist, out, limit)? {
                    return Ok(End::Limit);
                }
            }
            _ => return Err(bad("invalid block type")),
        }
        if b.overrun() {
            return Err(bad("truncated stream"));
        }
        if last {
            return Ok(End::Stream(b.consumed()));
        }
    }
}

/// Read a dynamic block's code definitions.
fn dynamic_codes(b: &mut Bits) -> Result<(Huffman, Huffman)> {
    b.refill();
    let hlit = b.take(5) as usize + 257;
    let hdist = b.take(5) as usize + 1;
    let hclen = b.take(4) as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(bad("too many length or distance codes"));
    }
    let mut clen = [0u8; 19];
    for &i in &CLEN_ORDER[..hclen] {
        b.refill();
        clen[i] = b.take(3) as u8;
    }
    let clen = Huffman::new(&clen)?;
    let mut lengths = [0u8; 286 + 30];
    let mut i = 0;
    while i < hlit + hdist {
        b.refill();
        let sym = clen.decode(b)?;
        let (val, rep) = match sym {
            0..=15 => (sym as u8, 1),
            16 => {
                if i == 0 {
                    return Err(bad("repeat with no previous length"));
                }
                (lengths[i - 1], 3 + b.take(2) as usize)
            }
            17 => (0, 3 + b.take(3) as usize),
            _ => (0, 11 + b.take(7) as usize),
        };
        if i + rep > hlit + hdist {
            return Err(bad("code lengths overflow"));
        }
        lengths[i..i + rep].fill(val);
        i += rep;
        if b.overrun() {
            return Err(bad("truncated stream"));
        }
    }
    if lengths[256] == 0 {
        return Err(bad("no end-of-block code"));
    }
    Ok((Huffman::new(&lengths[..hlit])?, Huffman::new(&lengths[hlit..hlit + hdist])?))
}

/// Decode one Huffman-coded block. True when the output hit `limit`.
fn codes(b: &mut Bits, lit: &Huffman, dist: &Huffman, out: &mut Vec<u8>, limit: usize) -> Result<bool> {
    loop {
        // 15 + 5 + 15 + 13 bits at most per literal/match
        b.refill();
        if b.overrun() {
            return Err(bad("truncated stream"));
        }
        let sym = lit.decode(b)?;
        if sym < 256 {
            if out.len() == limit {
                return Ok(true);
            }
            out.push(sym as u8);
            continue;
        }
        if sym == 256 {
            return Ok(false);
        }
        let s = sym as usize - 257;
        if s >= 29 {
            return Err(bad("invalid length symbol"));
        }
        let len = LEN_BASE[s] as usize + b.take(u32::from(LEN_EXTRA[s])) as usize;
        let d = dist.decode(b)? as usize;
        if d >= 30 {
            return Err(bad("invalid distance symbol"));
        }
        let dist = DIST_BASE[d] as usize + b.take(u32::from(DIST_EXTRA[d])) as usize;
        if dist > out.len() {
            return Err(bad("distance too far back"));
        }
        let room = limit - out.len();
        let n = len.min(room);
        let start = out.len() - dist;
        if dist >= n {
            out.extend_from_within(start..start + n);
        } else {
            // overlapping: the copy reads bytes it writes
            out.reserve(n);
            for i in 0..n {
                let v = out[start + i];
                out.push(v);
            }
        }
        if n < len {
            return Ok(true);
        }
    }
}

/// Adler-32 of `data` (RFC 1950).
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 bytes is the most that can be summed before b can overflow
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    b << 16 | a
}

/// Decompress a zlib stream, stopping once `limit` bytes are out (whatever
/// follows is not needed). The Adler-32 trailer is checked when the stream
/// ends before the limit and carries one.
pub fn zlib_decompress(src: &[u8], limit: usize) -> Result<Vec<u8>> {
    let [cmf, flg, ..] = *src else { return Err(bad("zlib header truncated")) };
    if cmf & 15 != 8 || cmf >> 4 > 7 || (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(bad("not a zlib stream"));
    }
    if flg & 0x20 != 0 {
        return Err(Error::Unsupported("zlib preset dictionary".into()));
    }
    let mut out = Vec::with_capacity(limit.min(1 << 28));
    if let End::Stream(used) = inflate(&src[2..], &mut out, limit)? {
        if let Some(t) = src.get(2 + used..2 + used + 4) {
            if adler32(&out) != u32::from_be_bytes([t[0], t[1], t[2], t[3]]) {
                return Err(bad("zlib checksum mismatch"));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_fixed_and_dynamic_blocks() {
        // b"hello hello hello hello" through Python's zlib: level 0 (a stored
        // block) and strategy Z_FIXED (a fixed-code block)
        let stored = [120, 1, 1, 23, 0, 232, 255, 104, 101, 108, 108, 111, 32, 104, 101, 108, 108, 111, 32, 104, 101, 108, 108, 111, 32, 104, 101, 108, 108, 111, 104, 3, 8, 177];
        assert_eq!(zlib_decompress(&stored, 1 << 20).unwrap(), b"hello hello hello hello");
        let fixed = [120, 1, 203, 72, 205, 201, 201, 87, 200, 64, 39, 1, 104, 3, 8, 177];
        assert_eq!(zlib_decompress(&fixed, 1 << 20).unwrap(), b"hello hello hello hello");
        assert_eq!(zlib_decompress(&fixed, 7).unwrap(), b"hello h");
        let mut broken = fixed;
        broken[12] ^= 1;
        assert!(zlib_decompress(&broken, 1 << 20).is_err());
        assert!(zlib_decompress(&fixed[..8], 1 << 20).is_err());
    }
}
