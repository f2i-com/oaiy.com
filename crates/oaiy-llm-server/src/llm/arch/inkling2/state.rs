//! Recurrent and KV-cache state for Inkling-2 / Inkling-2-Loop.
//!
//! Each logical layer pass (24 for non-looped `inkling2`, 48 for `inkling2_loop`
//! with 3 passes over core layers 6..18) maintains its own:
//!   * KV cache (`K` and `V` vectors, `n_kv_heads * head_dim` wide). Sliding-window
//!     layers bound their retained rows to at most `2 * sliding_window` via lazy
//!     compaction so memory stays constant regardless of context length.
//!   * Four causal short-convolution histories (`k_conv`, `v_conv`, `attn_post_conv`,
//!     `ffn_post_conv`), each storing the last `kernel - 1` inputs oldest-first.

use std::io::{Read, Write};
use std::sync::Arc;

use rayon::prelude::*;

use super::experts::{f32_axpy_fast, f32_dot_fast};

/// Causal depthwise 1D short-convolution state for a single stream.
///
/// Stores `(kernel - 1) * channels` floats in time-major order:
/// `hist[s * channels + c]` is the input from `(kernel - 1 - s)` tokens ago
/// (`s = 0` is oldest, `s = kernel - 2` is lag 1).
#[derive(Debug, Clone)]
pub struct ConvState {
    channels: usize,
    kernel: usize,
    hist: Vec<f32>,
}

impl ConvState {
    pub fn new(channels: usize, kernel: usize) -> Self {
        let past = kernel.saturating_sub(1);
        Self {
            channels,
            kernel,
            hist: vec![0.0; past * channels],
        }
    }

    pub fn reset(&mut self) {
        self.hist.fill(0.0);
    }

    pub fn hist(&self) -> &[f32] {
        &self.hist
    }

    pub fn set_hist(&mut self, src: &[f32]) {
        debug_assert_eq!(src.len(), self.hist.len());
        self.hist.copy_from_slice(src);
    }

    /// Static causal short convolution with residual and `1 / sqrt(kernel + 1)` scale.
    ///
    /// * `input`: `[seq, channels]`
    /// * `weight`: `[channels, kernel]`, oldest-to-newest (`weight[c * kernel + kernel - 1]` is lag 0).
    pub fn apply_static(&mut self, input: &[f32], seq: usize, weight: &[f32]) -> Vec<f32> {
        let c = self.channels;
        let k = self.kernel;
        let past = k - 1;
        debug_assert_eq!(input.len(), seq * c);
        debug_assert_eq!(weight.len(), c * k);

        let scale = 1.0 / ((k + 1) as f32).sqrt();
        let mut out = vec![0.0f32; seq * c];

        for t in 0..seq {
            let cur_row = &input[t * c..(t + 1) * c];
            let out_row = &mut out[t * c..(t + 1) * c];
            for ch in 0..c {
                let w = &weight[ch * k..(ch + 1) * k];
                let mut acc = 0.0f32;
                for j in 0..k {
                    // Window index `t + j` over `concat(hist, input)`:
                    // indices `< past` come from `hist`, indices `>= past` from `input`.
                    let idx = t + j;
                    let x = if idx < past {
                        self.hist[idx * c + ch]
                    } else {
                        input[(idx - past) * c + ch]
                    };
                    acc += x * w[j];
                }
                out_row[ch] = (acc + cur_row[ch]) * scale;
            }
        }

        self.advance_hist(input, seq);
        out
    }

    /// Dynamic causal short convolution (used for K and V on full-attention layers):
    /// static per-channel taps + per-KV-head dynamic taps projected from the layer input,
    /// with residual and `1 / sqrt(2 * kernel + 1)` scale.
    ///
    /// * `input`: `[seq, channels]` where `channels = n_heads * head_dim`
    /// * `dynamic`: `[seq, n_heads, kernel]`, **newest-to-oldest** (`lag = 0` at index 0,
    ///   matching `inkling2.cpp:224-226` and `Inkling2DynamicShortConvolution`).
    /// * `weight`: `[channels, kernel]`, **oldest-to-newest** (`lag = 0` at index `kernel - 1`,
    ///   matching `ggml_ssm_conv` after `conversion/inkling2.py`'s `flip(0).T`).
    pub fn apply_dynamic(
        &mut self,
        input: &[f32],
        dynamic: &[f32],
        seq: usize,
        weight: &[f32],
        n_heads: usize,
    ) -> Vec<f32> {
        let c = self.channels;
        let k = self.kernel;
        let past = k - 1;
        let head_dim = c / n_heads;
        debug_assert_eq!(input.len(), seq * c);
        debug_assert_eq!(dynamic.len(), seq * n_heads * k);
        debug_assert_eq!(weight.len(), c * k);

        let scale = 1.0 / ((2 * k + 1) as f32).sqrt();
        let mut out = vec![0.0f32; seq * c];

        for t in 0..seq {
            let cur_row = &input[t * c..(t + 1) * c];
            let dyn_t = &dynamic[t * n_heads * k..(t + 1) * n_heads * k];
            let out_row = &mut out[t * c..(t + 1) * c];
            for h in 0..n_heads {
                let taps = &dyn_t[h * k..(h + 1) * k];
                let ch_base = h * head_dim;
                for d in 0..head_dim {
                    let ch = ch_base + d;
                    let w = &weight[ch * k..(ch + 1) * k];
                    let mut acc = 0.0f32;
                    for j in 0..k {
                        let idx = t + j;
                        let x = if idx < past {
                            self.hist[idx * c + ch]
                        } else {
                            input[(idx - past) * c + ch]
                        };
                        // `j` runs oldest (lag = k - 1) to newest (lag = 0).
                        // Static weight is oldest-first (`w[j]`), dynamic taps are newest-first (`taps[k - 1 - j]`).
                        acc += x * (w[j] + taps[k - 1 - j]);
                    }
                    out_row[ch] = (acc + cur_row[ch]) * scale;
                }
            }
        }

        self.advance_hist(input, seq);
        out
    }

    fn advance_hist(&mut self, input: &[f32], seq: usize) {
        let c = self.channels;
        let past = self.kernel.saturating_sub(1);
        if past == 0 || seq == 0 {
            return;
        }
        if seq >= past {
            let start = (seq - past) * c;
            self.hist.copy_from_slice(&input[start..seq * c]);
        } else {
            let keep = past - seq;
            self.hist.copy_within(seq * c..past * c, 0);
            self.hist[keep * c..past * c].copy_from_slice(&input[..seq * c]);
        }
    }
}

/// Per-logical-slot state: KV cache + 4 short-convolution histories.
#[derive(Debug, Clone)]
pub struct LayerSlotState {
    pub is_swa: bool,
    pub sliding_window: usize,
    pub kv_dim: usize,
    /// Token position of row 0 in `k_cache` / `v_cache`.
    pub base_pos: usize,
    /// Number of rows currently stored in `k_cache` / `v_cache`.
    pub stored_rows: usize,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub k_conv: ConvState,
    pub v_conv: ConvState,
    pub attn_post_conv: ConvState,
    pub ffn_post_conv: ConvState,
}

impl LayerSlotState {
    pub fn new(
        is_swa: bool,
        sliding_window: usize,
        kv_dim: usize,
        hidden: usize,
        conv_kernel: usize,
    ) -> Self {
        Self {
            is_swa,
            sliding_window,
            kv_dim,
            base_pos: 0,
            stored_rows: 0,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            k_conv: ConvState::new(kv_dim, conv_kernel),
            v_conv: ConvState::new(kv_dim, conv_kernel),
            attn_post_conv: ConvState::new(hidden, conv_kernel),
            ffn_post_conv: ConvState::new(hidden, conv_kernel),
        }
    }

    pub fn reset(&mut self) {
        self.base_pos = 0;
        self.stored_rows = 0;
        self.k_cache.clear();
        self.v_cache.clear();
        self.k_conv.reset();
        self.v_conv.reset();
        self.attn_post_conv.reset();
        self.ffn_post_conv.reset();
    }

    /// Append `seq` new K and V rows (`[seq, kv_dim]`).
    pub fn append_kv(&mut self, k: &[f32], v: &[f32], seq: usize) {
        debug_assert_eq!(k.len(), seq * self.kv_dim);
        debug_assert_eq!(v.len(), seq * self.kv_dim);
        self.k_cache.extend_from_slice(k);
        self.v_cache.extend_from_slice(v);
        self.stored_rows += seq;
    }

    /// Compact old rows from an SWA cache once it grows past `2 * sliding_window`.
    pub fn maybe_compact(&mut self) {
        if !self.is_swa || self.sliding_window == 0 {
            return;
        }
        let keep = self.sliding_window;
        if self.stored_rows >= 2 * keep {
            let drop_rows = self.stored_rows - keep;
            let drop_elems = drop_rows * self.kv_dim;
            self.k_cache.drain(..drop_elems);
            self.v_cache.drain(..drop_elems);
            self.base_pos += drop_rows;
            self.stored_rows -= drop_rows;
        }
    }

    /// Compute GQA causal attention with relative-position bias table and per-query attention sinks.
    ///
    /// * `q`: `[seq, n_heads * head_dim]`
    /// * `rel_table`: `[seq, n_heads, rel_extent]` (already projected `R @ rel_bias^T`)
    /// * `sinks`: `[seq, n_heads]`
    /// * `past_len`: absolute position of query `t = 0`
    ///
    /// Returns `[seq, n_heads * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        q: &[f32],
        rel_table: &[f32],
        sinks: &[f32],
        seq: usize,
        past_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rel_extent: usize,
    ) -> Vec<f32> {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let base_pos = self.base_pos;
        let kv_end = base_pos + self.stored_rows;
        let is_swa = self.is_swa;
        let swa = self.sliding_window;
        let k_cache = &self.k_cache;
        let v_cache = &self.v_cache;

        let mut out = vec![0.0f32; seq * q_dim];

        let run_head = |th: usize, out_h: &mut [f32]| {
            let t = th / n_heads;
            let h = th % n_heads;
            let hk = h / group;
            let q_pos = past_len + t;

            let k_first = if is_swa {
                q_pos.saturating_sub(swa.saturating_sub(1)).max(base_pos)
            } else {
                base_pos
            };
            let k_last = (q_pos + 1).min(kv_end);
            if k_first >= k_last {
                out_h.fill(0.0);
                return;
            }

            let q_vec = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let r_row = &rel_table[(t * n_heads + h) * rel_extent..(t * n_heads + h + 1) * rel_extent];
            let sink_logit = sinks[t * n_heads + h];

            // Online softmax (FlashAttention-style) with the per-query sink as an initial zero-value entry.
            // Equivalent to: exp(s) / (exp(sink) + sum_k exp(s_k)), numerically stable for any context length.
            let mut max_s = sink_logit;
            let mut sum_exp = 1.0f32; // exp(sink_logit - sink_logit)
            out_h.fill(0.0);

            for k_pos in k_first..k_last {
                let row = k_pos - base_pos;
                let kv_off = row * kv_dim + hk * head_dim;
                let k_vec = &k_cache[kv_off..kv_off + head_dim];
                let v_vec = &v_cache[kv_off..kv_off + head_dim];

                let dot = f32_dot_fast(q_vec, k_vec);
                let dist = q_pos - k_pos;
                let bias = if dist < rel_extent { r_row[dist] } else { 0.0 };
                let score = dot * scale + bias;

                if score > max_s {
                    let rescale = (max_s - score).exp();
                    sum_exp = sum_exp * rescale + 1.0;
                    for d in 0..head_dim {
                        out_h[d] = out_h[d] * rescale + v_vec[d];
                    }
                    max_s = score;
                } else {
                    let w = (score - max_s).exp();
                    sum_exp += w;
                    f32_axpy_fast(out_h, v_vec, w);
                }
            }

            let inv = 1.0 / sum_exp;
            for v in out_h.iter_mut() {
                *v *= inv;
            }
        };

        // Avoid cross-CCD rayon wakeup overhead for short decode contexts (`eff_kv < ~1365`).
        let eff_kv = if is_swa { self.stored_rows.min(swa) } else { self.stored_rows };
        if seq > 1 || seq * n_heads * eff_kv >= 32_768 {
            out.par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(th, out_h)| run_head(th, out_h));
        } else {
            for (th, out_h) in out.chunks_mut(head_dim).enumerate() {
                run_head(th, out_h);
            }
        }

        out
    }
}

/// Complete mutable inference state for an `Inkling2Model`.
#[derive(Debug, Clone)]
pub struct Inkling2State {
    pub slots: Vec<LayerSlotState>,
    pub pos: usize,
}

impl Inkling2State {
    pub fn new(
        slot_is_swa: &[bool],
        slot_kv_dims: &[usize],
        sliding_window: usize,
        hidden: usize,
        conv_kernel: usize,
    ) -> Self {
        let slots = slot_is_swa
            .iter()
            .zip(slot_kv_dims.iter())
            .map(|(&is_swa, &kv_dim)| {
                LayerSlotState::new(is_swa, sliding_window, kv_dim, hidden, conv_kernel)
            })
            .collect();
        Self { slots, pos: 0 }
    }

    pub fn reset(&mut self) {
        for s in &mut self.slots {
            s.reset();
        }
        self.pos = 0;
    }

    /// Capture a host-side conv-only checkpoint (~3.7 MiB for Inkling-2-Mini).
    pub fn capture_conv_checkpoint(&self) -> Inkling2ConvSnapshot {
        let slots = self
            .slots
            .iter()
            .map(|s| Inkling2ConvSlotSnapshot {
                k_conv: s.k_conv.hist().to_vec(),
                v_conv: s.v_conv.hist().to_vec(),
                attn_post_conv: s.attn_post_conv.hist().to_vec(),
                ffn_post_conv: s.ffn_post_conv.hist().to_vec(),
            })
            .collect();
        Inkling2ConvSnapshot {
            pos: self.pos,
            slots,
        }
    }

    /// Capture a complete, self-contained KV + conv snapshot of `self` on the host.
    pub fn capture_snapshot(&self, hidden: usize, conv_kernel: usize, sliding_window: usize) -> Inkling2StateSnapshot {
        let slots = self
            .slots
            .iter()
            .map(|s| {
                let k_first = if s.is_swa {
                    self.pos.saturating_sub(sliding_window)
                } else {
                    0
                }
                .max(s.base_pos);
                let k_end = self.pos.min(s.base_pos + s.stored_rows);
                let count = k_end.saturating_sub(k_first);
                let (k_cache, v_cache) = if count > 0 {
                    let row0 = k_first - s.base_pos;
                    let off = row0 * s.kv_dim;
                    let len = count * s.kv_dim;
                    (
                        s.k_cache[off..off + len].to_vec(),
                        s.v_cache[off..off + len].to_vec(),
                    )
                } else {
                    (Vec::new(), Vec::new())
                };
                Inkling2SlotSnapshot {
                    is_swa: s.is_swa,
                    kv_dim: s.kv_dim,
                    base_pos: k_first,
                    stored_rows: count,
                    k_cache,
                    v_cache,
                    k_conv: s.k_conv.hist().to_vec(),
                    v_conv: s.v_conv.hist().to_vec(),
                    attn_post_conv: s.attn_post_conv.hist().to_vec(),
                    ffn_post_conv: s.ffn_post_conv.hist().to_vec(),
                }
            })
            .collect();
        Inkling2StateSnapshot {
            pos: self.pos,
            hidden,
            conv_kernel,
            sliding_window,
            slots,
        }
    }

    /// Restore `self` from a previously validated [`Inkling2StateSnapshot`].
    pub fn apply_snapshot(&mut self, snap: &Inkling2StateSnapshot) {
        for (dst, src) in self.slots.iter_mut().zip(&snap.slots) {
            dst.base_pos = src.base_pos;
            dst.stored_rows = src.stored_rows;
            dst.k_cache.clear();
            dst.k_cache.extend_from_slice(&src.k_cache);
            dst.v_cache.clear();
            dst.v_cache.extend_from_slice(&src.v_cache);
            dst.k_conv.set_hist(&src.k_conv);
            dst.v_conv.set_hist(&src.v_conv);
            dst.attn_post_conv.set_hist(&src.attn_post_conv);
            dst.ffn_post_conv.set_hist(&src.ffn_post_conv);
        }
        self.pos = snap.pos;
    }
}

/// Per-logical-slot short-convolution history snapshot.
#[derive(Debug, Clone)]
pub struct Inkling2ConvSlotSnapshot {
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub attn_post_conv: Vec<f32>,
    pub ffn_post_conv: Vec<f32>,
}

/// Lightweight host-RAM checkpoint of all logical slots' short-convolution histories at `pos`
/// (~3.7 MiB total for Inkling-2-Mini). Rewinds the live KV cache in < 0.3 ms while SWA ring
/// rows for `pos` remain intact.
#[derive(Debug, Clone)]
pub struct Inkling2ConvSnapshot {
    pub pos: usize,
    pub slots: Vec<Inkling2ConvSlotSnapshot>,
}

impl Inkling2ConvSnapshot {
    pub fn bytes(&self) -> usize {
        4 * self
            .slots
            .iter()
            .map(|s| s.k_conv.len() + s.v_conv.len() + s.attn_post_conv.len() + s.ffn_post_conv.len())
            .sum::<usize>()
    }
}

/// Full per-logical-slot KV window + short-convolution snapshot.
#[derive(Debug, Clone)]
pub struct Inkling2SlotSnapshot {
    pub is_swa: bool,
    pub kv_dim: usize,
    pub base_pos: usize,
    pub stored_rows: usize,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub attn_post_conv: Vec<f32>,
    pub ffn_post_conv: Vec<f32>,
}

/// Complete, self-contained KV + short-convolution snapshot for RAM parking and persistent disk cache.
#[derive(Debug, Clone)]
pub struct Inkling2StateSnapshot {
    pub pos: usize,
    pub hidden: usize,
    pub conv_kernel: usize,
    pub sliding_window: usize,
    pub slots: Vec<Inkling2SlotSnapshot>,
}

const SNAPSHOT_MAGIC: &[u8; 8] = b"INK2SNP1";
const SNAPSHOT_HEADER_BYTES: usize = 8 + 5 * 8; // magic + pos, hidden, conv_kernel, sliding_window, n_slots
const SLOT_HEADER_BYTES: usize = 1 + 3 * 8; // is_swa + kv_dim, base_pos, stored_rows

fn write_f32_slice(out: &mut dyn Write, data: &[f32]) -> std::io::Result<()> {
    const CHUNK_FLOATS: usize = 16_384; // 64 KiB scratch buffer
    let mut buf = [0u8; CHUNK_FLOATS * 4];
    for chunk in data.chunks(CHUNK_FLOATS) {
        let byte_len = chunk.len() * 4;
        for (i, &v) in chunk.iter().enumerate() {
            buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        out.write_all(&buf[..byte_len])?;
    }
    Ok(())
}

fn read_f32_vec(input: &mut dyn Read, count: usize) -> oaiy_engine::Result<Vec<f32>> {
    const CHUNK_FLOATS: usize = 16_384;
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; CHUNK_FLOATS * 4];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(CHUNK_FLOATS);
        let byte_len = take * 4;
        input.read_exact(&mut buf[..byte_len])?;
        for c in buf[..byte_len].chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        remaining -= take;
    }
    Ok(out)
}

impl Inkling2StateSnapshot {
    /// Approximate heap RAM bytes occupied by this snapshot.
    pub fn bytes(&self) -> usize {
        4 * self
            .slots
            .iter()
            .map(|s| {
                s.k_cache.len()
                    + s.v_cache.len()
                    + s.k_conv.len()
                    + s.v_conv.len()
                    + s.attn_post_conv.len()
                    + s.ffn_post_conv.len()
            })
            .sum::<usize>()
    }

    /// Extract a lightweight conv-only checkpoint from this complete snapshot.
    pub fn to_conv_snapshot(&self) -> Inkling2ConvSnapshot {
        Inkling2ConvSnapshot {
            pos: self.pos,
            slots: self
                .slots
                .iter()
                .map(|s| Inkling2ConvSlotSnapshot {
                    k_conv: s.k_conv.clone(),
                    v_conv: s.v_conv.clone(),
                    attn_post_conv: s.attn_post_conv.clone(),
                    ffn_post_conv: s.ffn_post_conv.clone(),
                })
                .collect(),
        }
    }

    /// Exact encoded payload byte size for `DiskCache`.
    pub fn encoded_len(&self) -> usize {
        let past = self.conv_kernel.saturating_sub(1);
        let mut total = SNAPSHOT_HEADER_BYTES;
        for s in &self.slots {
            let kv_floats = 2 * s.stored_rows * s.kv_dim;
            let conv_floats = 2 * past * s.kv_dim + 2 * past * self.hidden;
            total += SLOT_HEADER_BYTES + 4 * (kv_floats + conv_floats);
        }
        total
    }

    /// Stream this snapshot directly into `out` without duplicating the payload in memory.
    pub fn write_to(&self, out: &mut dyn Write) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::with_capacity(256 * 1024, out);
        w.write_all(SNAPSHOT_MAGIC)?;
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        w.write_all(&(self.hidden as u64).to_le_bytes())?;
        w.write_all(&(self.conv_kernel as u64).to_le_bytes())?;
        w.write_all(&(self.sliding_window as u64).to_le_bytes())?;
        w.write_all(&(self.slots.len() as u64).to_le_bytes())?;
        for s in &self.slots {
            w.write_all(&[u8::from(s.is_swa)])?;
            w.write_all(&(s.kv_dim as u64).to_le_bytes())?;
            w.write_all(&(s.base_pos as u64).to_le_bytes())?;
            w.write_all(&(s.stored_rows as u64).to_le_bytes())?;
            write_f32_slice(&mut w, &s.k_cache)?;
            write_f32_slice(&mut w, &s.v_cache)?;
            write_f32_slice(&mut w, &s.k_conv)?;
            write_f32_slice(&mut w, &s.v_conv)?;
            write_f32_slice(&mut w, &s.attn_post_conv)?;
            write_f32_slice(&mut w, &s.ffn_post_conv)?;
        }
        w.flush()
    }

    /// Decode a snapshot streamed from `input` of exact declared byte `length`.
    pub fn read_from(input: &mut dyn Read, length: u64) -> oaiy_engine::Result<Self> {
        let mut limited = std::io::BufReader::with_capacity(256 * 1024, input.take(length));
        let mut head = [0u8; SNAPSHOT_HEADER_BYTES];
        limited.read_exact(&mut head)?;
        if &head[..8] != SNAPSHOT_MAGIC {
            return Err(oaiy_engine::Error::Format("invalid Inkling-2 prompt state magic".into()));
        }
        let read_u64 = |off: usize| -> usize {
            u64::from_le_bytes(head[off..off + 8].try_into().expect("8 bytes")) as usize
        };
        let pos = read_u64(8);
        let hidden = read_u64(16);
        let conv_kernel = read_u64(24);
        let sliding_window = read_u64(32);
        let n_slots = read_u64(40);
        if hidden == 0 || hidden > (1 << 20) || conv_kernel == 0 || conv_kernel > 64 || n_slots == 0 || n_slots > 1024 {
            return Err(oaiy_engine::Error::Format("invalid Inkling-2 prompt state header".into()));
        }
        let past = conv_kernel.saturating_sub(1);
        let mut slots = Vec::with_capacity(n_slots);
        let mut sh = [0u8; SLOT_HEADER_BYTES];
        for _ in 0..n_slots {
            limited.read_exact(&mut sh)?;
            let is_swa = match sh[0] {
                0 => false,
                1 => true,
                _ => return Err(oaiy_engine::Error::Format("invalid Inkling-2 slot swa flag".into())),
            };
            let kv_dim = u64::from_le_bytes(sh[1..9].try_into().expect("8 bytes")) as usize;
            let base_pos = u64::from_le_bytes(sh[9..17].try_into().expect("8 bytes")) as usize;
            let stored_rows = u64::from_le_bytes(sh[17..25].try_into().expect("8 bytes")) as usize;
            if kv_dim == 0 || kv_dim > (1 << 20) || base_pos.saturating_add(stored_rows) > pos {
                return Err(oaiy_engine::Error::Format("invalid Inkling-2 slot dimensions".into()));
            }
            let kv_elems = stored_rows
                .checked_mul(kv_dim)
                .ok_or_else(|| oaiy_engine::Error::Format("Inkling-2 kv overflow".into()))?;
            let kv_conv_elems = past
                .checked_mul(kv_dim)
                .ok_or_else(|| oaiy_engine::Error::Format("Inkling-2 kv_conv overflow".into()))?;
            let post_conv_elems = past
                .checked_mul(hidden)
                .ok_or_else(|| oaiy_engine::Error::Format("Inkling-2 post_conv overflow".into()))?;
            let k_cache = read_f32_vec(&mut limited, kv_elems)?;
            let v_cache = read_f32_vec(&mut limited, kv_elems)?;
            let k_conv = read_f32_vec(&mut limited, kv_conv_elems)?;
            let v_conv = read_f32_vec(&mut limited, kv_conv_elems)?;
            let attn_post_conv = read_f32_vec(&mut limited, post_conv_elems)?;
            let ffn_post_conv = read_f32_vec(&mut limited, post_conv_elems)?;
            slots.push(Inkling2SlotSnapshot {
                is_swa,
                kv_dim,
                base_pos,
                stored_rows,
                k_cache,
                v_cache,
                k_conv,
                v_conv,
                attn_post_conv,
                ffn_post_conv,
            });
        }
        // Reject trailing garbage so corrupted files are caught immediately.
        let mut extra = [0u8; 1];
        if limited.read(&mut extra)? != 0 {
            return Err(oaiy_engine::Error::Format("trailing bytes in Inkling-2 prompt state".into()));
        }
        Ok(Self {
            pos,
            hidden,
            conv_kernel,
            sliding_window,
            slots,
        })
    }
}

impl crate::engines::disk::PromptState for Arc<Inkling2StateSnapshot> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn encoded_len(&self) -> usize {
        Inkling2StateSnapshot::encoded_len(self)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let _ = Inkling2StateSnapshot::write_to(self, out);
    }

    fn write_to(&self, out: &mut dyn Write) -> std::io::Result<()> {
        Inkling2StateSnapshot::write_to(self, out)
    }

    fn decode(bytes: &[u8]) -> oaiy_engine::Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        Inkling2StateSnapshot::read_from(&mut cursor, bytes.len() as u64).map(Arc::new)
    }

    fn read_from(input: &mut dyn Read, length: u64) -> oaiy_engine::Result<Self> {
        Inkling2StateSnapshot::read_from(input, length).map(Arc::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_conv_matches_streaming_token_by_token() {
        let (c, k, seq) = (6, 4, 5);
        let input: Vec<f32> = (0..seq * c).map(|i| ((i as f32) * 0.37).sin()).collect();
        let weight: Vec<f32> = (0..c * k).map(|i| ((i as f32) * 0.19).cos() * 0.5).collect();

        let mut batch = ConvState::new(c, k);
        let y_batch = batch.apply_static(&input, seq, &weight);

        let mut step = ConvState::new(c, k);
        let mut y_step = Vec::new();
        for t in 0..seq {
            y_step.extend_from_slice(&step.apply_static(&input[t * c..(t + 1) * c], 1, &weight));
        }

        for (a, b) in y_batch.iter().zip(&y_step) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert_eq!(batch.hist, step.hist);
    }

    #[test]
    fn dynamic_conv_matches_streaming_and_lag_orientation() {
        let (n_heads, head_dim, k, seq) = (2, 3, 4, 5);
        let c = n_heads * head_dim;
        let input: Vec<f32> = (0..seq * c).map(|i| ((i as f32) * 0.23).sin()).collect();
        let dyn_taps: Vec<f32> = (0..seq * n_heads * k).map(|i| ((i as f32) * 0.41).cos() * 0.3).collect();
        let weight: Vec<f32> = (0..c * k).map(|i| ((i as f32) * 0.13).sin() * 0.2).collect();

        let mut batch = ConvState::new(c, k);
        let y_batch = batch.apply_dynamic(&input, &dyn_taps, seq, &weight, n_heads);

        let mut step = ConvState::new(c, k);
        let mut y_step = Vec::new();
        for t in 0..seq {
            y_step.extend_from_slice(&step.apply_dynamic(
                &input[t * c..(t + 1) * c],
                &dyn_taps[t * n_heads * k..(t + 1) * n_heads * k],
                1,
                &weight,
                n_heads,
            ));
        }
        for (a, b) in y_batch.iter().zip(&y_step) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn online_softmax_with_sink_matches_reference_formula() {
        let mut slot = LayerSlotState::new(false, 4, 4, 4, 4);
        let k = vec![
            1.0, 0.0, 0.5, -0.5, // pos 0
            0.2, 0.8, -0.3, 0.4, // pos 1
            -0.5, 0.3, 0.9, 0.1, // pos 2
        ];
        let v = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        slot.append_kv(&k, &v, 3);
        let q = vec![0.4, -0.2, 0.7, 0.3]; // single query at pos 2
        let rel_table = vec![0.25, -0.1, 0.5, 0.0]; // extent 4: d=0 -> 0.25, d=1 -> -0.1, d=2 -> 0.5
        let sinks = vec![1.2];
        let got = slot.attend(&q, &rel_table, &sinks, 1, 2, 1, 1, 4, 4);

        let scale = 1.0 / (4.0f32).sqrt();
        let mut logits = [0.0f32; 4];
        for p in 0..3 {
            let dot: f32 = (0..4).map(|d| q[d] * k[p * 4 + d]).sum();
            logits[p] = dot * scale + rel_table[2 - p];
        }
        logits[3] = sinks[0];
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for d in 0..4 {
            let want: f32 = (0..3).map(|p| (exps[p] / sum) * v[p * 4 + d]).sum();
            assert!((got[d] - want).abs() < 1e-5, "dim {d}: got {} vs want {want}", got[d]);
        }
    }

    #[test]
    fn state_snapshot_round_trips_and_rejects_corruption() {
        use crate::engines::disk::PromptState;

        let mut st = Inkling2State::new(&[false, true], &[4, 4], 3, 6, 4);
        // Append 5 tokens to both full and SWA slots.
        for t in 0..5 {
            let k: Vec<f32> = (0..4).map(|i| (t * 4 + i) as f32 * 0.25 + 1.0).collect();
            let v: Vec<f32> = (0..4).map(|i| (t * 4 + i) as f32 * -0.5 - 2.0).collect();
            st.slots[0].append_kv(&k, &v, 1);
            st.slots[1].append_kv(&k, &v, 1);
        }
        st.pos = 5;
        st.slots[0].k_conv.set_hist(&vec![1.5; 12]);
        st.slots[0].v_conv.set_hist(&vec![-0.5; 12]);
        st.slots[0].attn_post_conv.set_hist(&vec![0.25; 18]);
        st.slots[0].ffn_post_conv.set_hist(&vec![-0.75; 18]);
        st.slots[1].k_conv.set_hist(&vec![2.0; 12]);

        let snap = Arc::new(st.capture_snapshot(6, 4, 3));
        assert_eq!(snap.pos, 5);
        assert_eq!(snap.slots[0].base_pos, 0);
        assert_eq!(snap.slots[0].stored_rows, 5);
        // SWA slot only retains the last `sliding_window = 3` rows (`[2..5)`).
        assert_eq!(snap.slots[1].base_pos, 2);
        assert_eq!(snap.slots[1].stored_rows, 3);

        let mut raw = Vec::new();
        snap.write_to(&mut raw).unwrap();
        assert_eq!(raw.len(), snap.encoded_len());

        let decoded = <Arc<Inkling2StateSnapshot> as PromptState>::decode(&raw).unwrap();
        assert_eq!(decoded.pos, 5);
        assert_eq!(decoded.slots[0].k_cache, snap.slots[0].k_cache);
        assert_eq!(decoded.slots[1].base_pos, 2);
        assert_eq!(decoded.slots[1].v_cache, snap.slots[1].v_cache);
        assert_eq!(decoded.slots[0].ffn_post_conv, snap.slots[0].ffn_post_conv);

        // Apply to a fresh state and verify exact equality.
        let mut restored = Inkling2State::new(&[false, true], &[4, 4], 3, 6, 4);
        restored.apply_snapshot(&decoded);
        assert_eq!(restored.pos, 5);
        assert_eq!(restored.slots[0].k_cache, snap.slots[0].k_cache);
        assert_eq!(restored.slots[1].base_pos, 2);
        assert_eq!(restored.slots[1].stored_rows, 3);
        assert_eq!(restored.slots[0].attn_post_conv.hist(), &vec![0.25; 18]);

        // Truncated and extended payloads must be rejected cleanly.
        assert!(<Arc<Inkling2StateSnapshot> as PromptState>::decode(&raw[..raw.len() - 1]).is_err());
        let mut extended = raw.clone();
        extended.push(0);
        assert!(<Arc<Inkling2StateSnapshot> as PromptState>::decode(&extended).is_err());
    }
}
