//! The hashed n-gram table: its rows read from the file and decoded as they are needed.

use super::*;

/// The hashed n-gram table: trellis-quantized 160-value rows, read from disk as needed.
pub(super) struct NgramTable {
    /// A handle per reading thread: Windows serializes the reads made through one handle.
    pub(super) files: Vec<File>,
    pub(super) start: u64,
    pub(super) rows: u64,
    pub(super) row_words: usize,
    /// A trellis row's bits a value; 0 for a GGUF's table (IQ4_NL rows, no head bias and no codebook).
    pub(super) bits: usize,
    pub(super) head_offsets: Vec<i64>,
    pub(super) head_sizes: Vec<i64>,
    pub(super) multipliers: Vec<i64>,
    /// `[heads, 160]`.
    pub(super) bias: Vec<f32>,
    pub(super) codebook: Vec<f32>,
}

pub(super) const ROW_DIM: usize = 160;
pub(super) const MUL1: u64 = 0x83DC_D12D;

/// The 65536 decoded `mul1` values, as fp16 (bit-exact with EXL3's codebook).
pub(super) fn mul1_codebook() -> Vec<f32> {
    let k_inv = dsv41::formats::f16_to_f32(0x1eee);
    let k_bias = dsv41::formats::f16_to_f32(0xc931);
    (0..65536u64).map(|s| {
        let p = (s * MUL1) & 0xffff_ffff;
        let sum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + ((p >> 24) & 255);
        round_f16((1024 + sum) as f32 * k_inv + k_bias)
    }).collect()
}

/// The bytes of a GGUF n-gram table's row: `ROW_DIM` values as IQ4_NL blocks of 32 (18 bytes each).
pub(super) const IQ4_NL_ROW: usize = ROW_DIM / 32 * 18;

/// The most u16 words an n-gram table's row takes (its scale, then `ROW_DIM` codes of up to 8 bits).
pub(super) const ROW_WORDS_MAX: usize = 1 + ROW_DIM * 8 / 16;

/// A trellis row of the n-gram table (`bytes`: its f16 scale, then `ROW_DIM` codes of `k` bits, each value's 16-bit
/// state its own code and those of the `16 / k` before it, the row a ring) decoded into `out` with its head's `bias`:
/// each code read once, each state its codes shifted together (where each state's 16 bits were read one at a time).
pub(super) fn decode_row(bytes: &[u8], k: usize, codebook: &[f32], bias: &[f32], out: &mut [f32]) {
    let word = |i: usize| bytes.get(2 * i..2 * i + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])) as u32;
    let scale = dsv41::formats::f16_to_f32(word(0) as u16);
    let mask = (1u32 << k) - 1;
    let mut codes = [0u32; ROW_DIM];
    for (j, c) in codes.iter_mut().enumerate() {
        let b = j * k;
        let w = word(1 + b / 16) | (word(2 + b / 16) << 16);
        *c = (w >> (b % 16)) & mask;
    }
    let groups = 16usize.div_ceil(k);
    for (i, o) in out.iter_mut().enumerate().take(ROW_DIM) {
        let mut state = 0u32;
        for g in 0..groups {
            state |= codes[(i + ROW_DIM - g) % ROW_DIM] << (g * k);
        }
        *o = codebook[(state & 0xffff) as usize] * scale + bias[i];
    }
}

impl NgramTable {
    pub(super) fn open(idx: &StIndex, key: &str, heads: usize) -> Result<Self> {
        let mut shards = Vec::new();
        while let Some(info) = idx.get(&format!("{key}.shard_{}.trellis", shards.len())) { shards.push(info.clone()); }
        if shards.is_empty() {
            shards.push(idx.info(&format!("{key}.trellis"))?.clone());
        }
        let first = &shards[0];
        if first.dtype != Dtype::I16 || first.shape.len() != 2 {
            return Err(bad("n-gram table: expected int16 trellis rows"));
        }
        let row_words = first.shape[1];
        let bits = (row_words - 1) * 16 / ROW_DIM;
        if 1 + ROW_DIM * bits / 16 != row_words || !(1..=8).contains(&bits) {
            return Err(bad("n-gram table: unexpected row width"));
        }
        // The shards sit back to back in one file: one table.
        let row_bytes = (row_words * 2) as u64;
        let mut rows = 0u64;
        for s in &shards {
            if s.shard != first.shard || s.start != first.start + rows * row_bytes || s.shape[1] != row_words {
                return Err(bad("n-gram table shards are not contiguous"));
            }
            rows += s.shape[0] as u64;
        }
        let head_offsets = idx.read_i64(&format!("{key}.head_offsets"))?;
        let head_sizes = idx.read_i64(&format!("{key}.head_vocab_sizes"))?;
        let multipliers = idx.read_i64(&format!("{key}.layer_multipliers"))?;
        let bias = idx.read_f32(&format!("{key}.head_bias"))?;
        if head_offsets.len() != heads || head_sizes.len() != heads || bias.len() != heads * ROW_DIM {
            return Err(bad("n-gram table: head parameters do not match the heads"));
        }
        Ok(Self {
            files: (0..rayon::current_num_threads().max(1)).map(|_| File::open(idx.shard_path(first.shard))).collect::<std::io::Result<_>>()?,
            start: first.start,
            rows,
            row_words,
            bits,
            head_offsets,
            head_sizes,
            multipliers,
            bias,
            codebook: mul1_codebook(),
        })
    }

    /// Row `row` (of hash head `head`), decoded.
    pub(super) fn row(&self, row: u64, head: usize, out: &mut [f32]) -> Result<()> {
        if row >= self.rows {
            return Err(bad("n-gram row out of range"));
        }
        // (a GGUF's table: each row five IQ4_NL blocks, its values as they are)
        if self.bits == 0 {
            let mut buf = [0u8; IQ4_NL_ROW];
            let file = &self.files[rayon::current_thread_index().unwrap_or(0) % self.files.len()];
            read_at(file, &mut buf, self.start + row * IQ4_NL_ROW as u64)?;
            ggml_quants::iq4_nl::dequantize(&buf, &mut out[..ROW_DIM]);
            return Ok(());
        }
        // (a row's 62 bytes or so: on the stack)
        let mut buf = [0u8; 2 * ROW_WORDS_MAX];
        let bytes = &mut buf[..self.row_words * 2];
        let file = &self.files[rayon::current_thread_index().unwrap_or(0) % self.files.len()];
        read_at(file, bytes, self.start + row * (self.row_words * 2) as u64)?;
        decode_row(bytes, self.bits, &self.codebook, &self.bias[head * ROW_DIM..(head + 1) * ROW_DIM], out);
        Ok(())
    }
}

#[cfg(windows)]
pub(super) fn read_at(f: &File, buf: &mut [u8], mut at: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.seek_read(&mut buf[done..], at)?;
        if n == 0 { return Err(bad("n-gram table: short read")); }
        done += n;
        at += n as u64;
    }
    Ok(())
}
#[cfg(unix)]
pub(super) fn read_at(f: &File, buf: &mut [u8], at: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, at)?;
    Ok(())
}
