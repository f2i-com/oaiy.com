//! Recurrent and KV-cache state for Inkling-2 / Inkling-2-Loop.
//!
//! Each logical layer pass (24 for non-looped `inkling2`, 48 for `inkling2_loop`
//! with 3 passes over core layers 6..18) maintains its own:
//!   * KV cache (`K` and `V` vectors, `n_kv_heads * head_dim` wide). Sliding-window
//!     layers bound their retained rows to at most `2 * sliding_window` via lazy
//!     compaction so memory stays constant regardless of context length.
//!   * Four causal short-convolution histories (`k_conv`, `v_conv`, `attn_post_conv`,
//!     `ffn_post_conv`), each storing the last `kernel - 1` inputs oldest-first.

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
}
