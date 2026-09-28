//! Engram: n-gram hash lookups added into the residual stream at layers 1
//! and 14 (reference `engram.py`, `Engram`, `ParallelEngramEmbedding`).
//!
//! Each position hashes the n-grams (2..=4 tokens) ending at it, over a
//! *compressed* token vocabulary (tokens that normalize alike collapse), into
//! 24 rows (3 n-gram sizes x 8 heads) of a ~384M-row fp8 table per layer.
//! The rows go through `wkv` to one key per residual copy plus a shared
//! value; a signed-sqrt sigmoid gate on the normalized key/stream dot decides
//! how much value each copy receives.
//!
//! The compressed token map needs Unicode NFKC/NFD normalization, and the
//! hash moduli come from sympy primes and numpy's PCG64 — none of it is
//! reproducible from std Rust, so `tools/dsv41/oracle.py` exports it once as
//! `engram_meta.safetensors` and this module only does the arithmetic.
//!
//! The tables (2 x 101 GB) stay on disk: 24 rows x (256 + 8 bytes) per
//! token per layer, positioned reads through the page cache, which keeps
//! hot n-grams in RAM for free.

use std::fs::File;
use std::path::Path;
use std::sync::mpsc;

use oaiy_engine::{Error, Result};

use crate::config::Config;
use crate::formats::{e8m0_to_f32, fp8_e4m3_to_f32, to_bf16};
use crate::hc::HC;
use crate::io::read_exact_at;
use crate::linear::{Out, Weight};
use crate::safetensors::{Dtype, StIndex};

/// Marks a position that takes no part in an n-gram (image tokens).
const DEAD: i64 = -1;

/// Rolling n-gram hasher with the token history of the current sequence.
pub struct NgramHasher {
    token_map: Vec<i64>,
    pad: i64,
    /// `[engram layer][max_ngram]`
    multipliers: Vec<i64>,
    /// `[engram layer][max_ngram - 1][heads]`
    primes: Vec<i64>,
    /// `[engram layer][cols]`
    offsets: Vec<i64>,
    n_layers: usize,
    max_ngram: usize,
    heads: usize,
    cache: Vec<i64>,
}

impl NgramHasher {
    /// Load the precompute written by `oracle.py`.
    pub fn load(meta: &Path, cfg: &Config, max_seq: usize) -> Result<NgramHasher> {
        let m = StIndex::open_file(meta)?;
        let (n_layers, max_ngram, heads) = (cfg.engram_layer_ids.len(), cfg.engram_max_ngram_size, cfg.engram_n_heads);
        let h = NgramHasher {
            token_map: m.read_i64("token_map")?,
            pad: *m.read_i64("pad_id")?.first().ok_or_else(|| Error::Format("engram_meta: empty pad_id".into()))?,
            multipliers: m.read_i64("multipliers")?,
            primes: m.read_i64("primes")?,
            offsets: m.read_i64("offsets")?,
            n_layers,
            max_ngram,
            heads,
            cache: vec![0; max_seq],
        };
        let cols = (max_ngram - 1) * heads;
        if h.multipliers.len() != n_layers * max_ngram
            || h.primes.len() != n_layers * cols
            || h.offsets.len() != n_layers * cols
            || h.token_map.len() < cfg.vocab_size
        {
            return Err(Error::Format("engram_meta shapes do not match config.json".into()));
        }
        Ok(h)
    }

    pub fn cols(&self) -> usize {
        (self.max_ngram - 1) * self.heads
    }

    /// Hash ids for `ids` at positions `start_pos..`: `[t][engram layer][cols]`, flattened.
    pub fn forward(&mut self, ids: &[u32], start_pos: usize) -> Result<Vec<i64>> {
        self.forward_masked(ids, start_pos, None)
    }

    /// [`forward`](Self::forward) with image tokens (`image[i]`): they enter
    /// the history as DEAD, so no n-gram reaches into or across an image
    /// span (the reference's `token_mask`); their own hashes are all-pad
    /// ones, whose rows the caller gates off.
    /// Take a restored sequence's token history: `ids` are its tokens from
    /// position 0 (text only).
    pub fn set_history(&mut self, ids: &[u32]) -> Result<()> {
        if ids.len() > self.cache.len() {
            return Err(Error::Arg("sequence longer than the n-gram history".into()));
        }
        for (slot, &id) in self.cache.iter_mut().zip(ids) {
            *slot = *self.token_map.get(id as usize).ok_or_else(|| Error::Arg(format!("token {id} outside the vocabulary")))?;
        }
        Ok(())
    }

    pub fn forward_masked(&mut self, ids: &[u32], start_pos: usize, image: Option<&[bool]>) -> Result<Vec<i64>> {
        if start_pos + ids.len() > self.cache.len() {
            return Err(Error::Arg("sequence longer than the n-gram history".into()));
        }
        if image.is_some_and(|m| m.len() != ids.len()) {
            return Err(Error::Arg("image mask length differs from the tokens'".into()));
        }
        for (i, &id) in ids.iter().enumerate() {
            let c = *self.token_map.get(id as usize).ok_or_else(|| Error::Arg(format!("token {id} outside the vocabulary")))?;
            self.cache[start_pos + i] = if image.is_some_and(|m| m[i]) { DEAD } else { c };
        }
        let cols = self.cols();
        let mut out = Vec::with_capacity(ids.len() * self.n_layers * cols);
        for i in 0..ids.len() {
            let pos = start_pos + i;
            let mut tokens = vec![0i64; self.max_ngram];
            let mut blocked = false;
            for (shift, tok) in tokens.iter_mut().enumerate() {
                let src = self.cache[pos.saturating_sub(shift)];
                blocked |= pos < shift || src == DEAD;
                *tok = if blocked { self.pad } else { src };
            }
            for l in 0..self.n_layers {
                let mult = &self.multipliers[l * self.max_ngram..(l + 1) * self.max_ngram];
                let mut rolling = tokens[0].wrapping_mul(mult[0]);
                for n in 1..self.max_ngram {
                    rolling ^= tokens[n].wrapping_mul(mult[n]);
                    for h in 0..self.heads {
                        let c = (n - 1) * self.heads + h;
                        let prime = self.primes[l * cols + c];
                        out.push(rolling.rem_euclid(prime) + self.offsets[l * cols + c]);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Reader threads per table, each with its own file handles. Windows
/// serializes the reads on one synchronous handle (positioned reads from
/// several threads through one `File` wait for each other), so a shared
/// handle sends the drive one row at a time: measured on the T9, a decode
/// step's rows took 12.9 ms that way and 3.0 ms with a handle per reader; a
/// 2,000-token prompt's 20.3 s and 2.8-3.5 s (16-32 readers). The readers
/// live as long as the table: spawning them for every lookup cost 0.9 ms a
/// layer, even when the page cache held every row.
const READERS: usize = 24;

/// Where one layer's table lies in its shards.
#[derive(Clone, Copy)]
struct Layout {
    w_start: u64,
    s_start: u64,
    rows: u64,
    dim: usize,
}

/// One reader's own handles on a table.
struct Reader {
    weight: File,
    scale: File,
    at: Layout,
}

impl Reader {
    /// Row `r`, dequantized and rounded to bf16 like the reference lookup.
    fn row(&self, r: i64, dst: &mut [f32]) -> Result<()> {
        let at = &self.at;
        let r = u64::try_from(r).ok().filter(|&r| r < at.rows).ok_or_else(|| Error::Format(format!("engram row {r} out of range")))?;
        let mut w = vec![0u8; at.dim];
        let mut s = vec![0u8; at.dim / 32];
        read_exact_at(&self.weight, &mut w, at.w_start + r * at.dim as u64)?;
        read_exact_at(&self.scale, &mut s, at.s_start + r * (at.dim / 32) as u64)?;
        for (j, (d, &q)) in dst.iter_mut().zip(&w).enumerate() {
            *d = to_bf16(fp8_e4m3_to_f32(q) * e8m0_to_f32(s[j / 32]));
        }
        Ok(())
    }
}

/// A reader's share of one lookup: its rows, and where their values go.
struct Job {
    rows: Vec<i64>,
    part: usize,
    reply: mpsc::Sender<(usize, Result<Vec<f32>>)>,
}

/// One layer's table, read row by row by [`READERS`] threads, which stop
/// when the table is dropped.
struct Table {
    readers: Vec<mpsc::Sender<Job>>,
    dim: usize,
}

impl Table {
    fn open(idx: &StIndex, layer: usize, dim: usize) -> Result<Table> {
        let w = idx.info(&format!("layers.{layer}.engram.embed.weight"))?;
        let s = idx.info(&format!("layers.{layer}.engram.embed.scale"))?;
        if w.dtype != Dtype::F8E4M3 || s.dtype != Dtype::F8E8M0 || w.shape != [w.shape[0], dim] || s.shape != [w.shape[0], dim / 32] {
            return Err(Error::Format(format!("layer {layer}: unexpected engram table layout")));
        }
        let at = Layout { w_start: w.start, s_start: s.start, rows: w.shape[0] as u64, dim };
        let readers = (0..READERS)
            .map(|_| -> Result<mpsc::Sender<Job>> {
                let reader = Reader { weight: File::open(idx.shard_path(w.shard))?, scale: File::open(idx.shard_path(s.shard))?, at };
                let (tx, rx) = mpsc::channel::<Job>();
                std::thread::Builder::new().name(format!("engram {layer}")).spawn(move || {
                    for job in rx {
                        let mut out = vec![0.0f32; job.rows.len() * reader.at.dim];
                        let res = job.rows.iter().zip(out.chunks_exact_mut(reader.at.dim)).try_for_each(|(&r, dst)| reader.row(r, dst)).map(|()| out);
                        // a lookup that has failed already stopped listening
                        let _ = job.reply.send((job.part, res));
                    }
                })?;
                Ok(tx)
            })
            .collect::<Result<_>>()?;
        Ok(Table { readers, dim })
    }

    /// Rows `uniq`, `[len][dim]`, split among the readers.
    fn read(&self, uniq: &[i64]) -> Result<Vec<f32>> {
        let mut got = vec![0.0f32; uniq.len() * self.dim];
        if uniq.is_empty() {
            return Ok(got);
        }
        let gone = || Error::Io(std::io::Error::other("engram reader thread exited"));
        let per = uniq.len().div_ceil(READERS);
        let (tx, rx) = mpsc::channel();
        let mut parts = 0;
        for (part, (rows, reader)) in uniq.chunks(per).zip(&self.readers).enumerate() {
            reader.send(Job { rows: rows.to_vec(), part, reply: tx.clone() }).map_err(|_| gone())?;
            parts += 1;
        }
        drop(tx);
        for _ in 0..parts {
            let (part, vals) = rx.recv().map_err(|_| gone())?;
            let vals = vals?;
            got[part * per * self.dim..][..vals.len()].copy_from_slice(&vals);
        }
        Ok(got)
    }
}

pub struct Engram {
    /// Index of this layer among `engram_layer_ids`.
    pub hash_index: usize,
    table: Table,
    wkv: Weight,
    /// `q_weight * k_weight`, `[HC][dim]` f32.
    qk: Vec<f32>,
}

impl Engram {
    pub fn load(idx: &StIndex, cfg: &Config, layer: usize) -> Result<Engram> {
        let p = format!("layers.{layer}.engram");
        let q = idx.read_f32(&format!("{p}.q_weight"))?;
        let k = idx.read_f32(&format!("{p}.k_weight"))?;
        Ok(Engram {
            hash_index: cfg.engram_layer_ids.iter().position(|&l| l == layer).ok_or_else(|| Error::Arg(format!("layer {layer} has no engram")))?,
            table: Table::open(idx, layer, cfg.engram_head_dim)?,
            wkv: Weight::load(idx, &format!("{p}.wkv"))?,
            qk: q.iter().zip(&k).map(|(a, b)| a * b).collect(),
        })
    }

    /// `h`: `[t, HC, dim]` (bf16 values); `hashes`: `[t][cols]` for this layer.
    pub fn forward(&self, cfg: &Config, h: &[f32], t: usize, hashes: &[i64]) -> Result<Vec<f32>> {
        let emb = self.rows(hashes, cfg.engram_head_dim)?;
        let kv = self.wkv.forward(&emb, t, Out::Bf16);
        Ok(gate(h, &kv, &self.qk, cfg.dim, cfg.norm_eps))
    }

    /// The table rows for `hashes`, dequantized to bf16 values, `[len][head_dim]`.
    ///
    /// Each row is two small positioned reads (264 bytes) at a random spot in
    /// a 101 GB shard, so they are latency-bound: the table's [`READERS`]
    /// threads issue them at once rather than one after another (a drive
    /// serves dozens of such reads in about the time of one). A row that
    /// several tokens share (a repeated n-gram, the all-pad ones at the
    /// start) is read once.
    pub fn rows(&self, hashes: &[i64], head_dim: usize) -> Result<Vec<f32>> {
        if head_dim != self.table.dim {
            return Err(Error::Arg(format!("engram rows are {} wide, not {head_dim}", self.table.dim)));
        }
        let mut uniq = hashes.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let got = self.table.read(&uniq)?;
        let mut emb = vec![0.0f32; hashes.len() * head_dim];
        for (h, dst) in hashes.iter().zip(emb.chunks_exact_mut(head_dim)) {
            let i = uniq.binary_search(h).expect("every hash has its row");
            dst.copy_from_slice(&got[i * head_dim..(i + 1) * head_dim]);
        }
        Ok(emb)
    }

    pub fn wkv(&self) -> &Weight {
        &self.wkv
    }

    /// `q_weight * k_weight`, `[HC][dim]`.
    pub fn qk(&self) -> &[f32] {
        &self.qk
    }
}

/// The Engram gate: per token and residual copy, a signed-sqrt sigmoid of
/// the normalized (stream . qk . key) decides how much of the shared value
/// is added: `bf16(h + gate * value)`. `h` is `[t, HC, d]`, `kv` is
/// `[t][HC keys + 1 value][d]`.
pub fn gate(h: &[f32], kv: &[f32], qk: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let t = h.len() / (HC * d);
    let (clamp, scale) = (1e-6f32, (d as f32).powf(-0.5));
    let mut out = vec![0.0f32; h.len()];
    for i in 0..t {
        let kvr = &kv[i * (HC + 1) * d..(i + 1) * (HC + 1) * d];
        let value = &kvr[HC * d..];
        for c in 0..HC {
            let hs = &h[(i * HC + c) * d..(i * HC + c + 1) * d];
            let key = &kvr[c * d..(c + 1) * d];
            let w = &qk[c * d..(c + 1) * d];
            let ms = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let rstd = 1.0 / (ms(hs) + eps).sqrt() * (1.0 / (ms(key) + eps).sqrt());
            let dot = hs.iter().zip(w).zip(key).map(|((a, b), k)| a * b * k).sum::<f32>() * rstd * scale;
            let g = crate::ops::sigmoid(dot.abs().max(clamp).sqrt().copysign(dot));
            let o = &mut out[(i * HC + c) * d..(i * HC + c + 1) * d];
            for ((ov, hv), vv) in o.iter_mut().zip(hs).zip(value) {
                *ov = to_bf16(hv + g * vv);
            }
        }
    }
    out
}
