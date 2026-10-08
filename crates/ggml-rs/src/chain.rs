//! VENDORED-LOCAL: a decode step's ops chained on a device between host round trips.
//!
//! A backend whose every call is a submit and a read back (WebGPU) spends a dense model's decode step on round trips:
//! a 3B Llama's 113 of them were most of its 37 ms. A [`DeviceChain`] keeps the activations (and a copy of the KV
//! cache) on the device and records a run of ops (quantized matmuls, RMSNorm, RoPE, adds, the SwiGLU, attention) into
//! one submit, reading back only what the host needs (the logits, the step's K and V rows for the host's cache). A
//! model asks its backend for one through [`crate::Backend::chain`] and keeps its own path where there is none.

use std::any::Any;
use std::sync::Arc;

use crate::quantized::QuantizedTensor;
use crate::tensor::Tensor;

/// An f32 vector held by a [`DeviceChain`]'s device: the backend's own buffer behind it.
#[derive(Clone)]
pub struct DeviceVec {
    pub len: usize,
    pub inner: Arc<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for DeviceVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceVec({})", self.len)
    }
}

/// The shape of a gated delta net's recurrence over a run of tokens ([`ChainRecorder::delta_net`]).
#[derive(Clone, Copy, Debug)]
pub struct DeltaNet {
    /// Tokens in the run.
    pub rows: usize,
    pub v_heads: usize,
    pub k_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub scale_q: f32,
    pub eps: f32,
    /// The output gated by `sigmoid(z)` (Qwen3.8-Flash-Next) where Qwen3.5's is `silu(z)`.
    pub sigmoid_gate: bool,
}

/// How [`ChainRecorder::norm_mod_rows`] normalizes each row before its modulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowNorm {
    /// None: the modulation alone (an affine of the row).
    None,
    /// Over the row's RMS (no centring).
    Rms,
    /// A layer norm: centred, over its deviation.
    Layer,
}

/// Which rows of a modulated op take its modulation's second set: a video's clean conditioning tokens (its first
/// frame's, its appended last frame's), their timestep 0 beside the noisy tokens' sigma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CleanRows {
    /// The rows before this one, and those from `from` on, take the second set.
    pub before: usize,
    pub from: usize,
    /// The second set's offset from the first in the modulation's vector.
    pub offset: usize,
}

impl CleanRows {
    /// Every row the first set's.
    pub const NONE: Self = Self { before: 0, from: usize::MAX, offset: 0 };
}

/// A device that runs a chain of ops on vectors it holds.
pub trait DeviceChain: Send + Sync {
    /// A zeroed vector of `len`.
    fn vec(&self, len: usize) -> DeviceVec;
    /// `values` (a matrix's, an even number) as f16, two to a word, if each is an f16 value exactly (a checkpoint's
    /// f16 weights held as f32): for [`ChainRecorder::matmul_f16_rows`]. None where the device keeps no such matrices.
    fn vec_f16(&self, values: &[f32]) -> Option<DeviceVec> {
        let _ = values;
        None
    }
    /// [`Self::vec_f16`] with each value rounded to the nearest f16 (a BF16 checkpoint's weights, its smallest values
    /// f16 holds only nearly). None where a value is past f16's range, or the device keeps no f16 matrices.
    fn vec_f16_rounded(&self, values: &[f32]) -> Option<DeviceVec> {
        let _ = values;
        None
    }
    /// A 1-D convolution's weights (`[cout, cin, k]` as PyTorch keeps them) for [`ChainRecorder::conv1d_rows`]: f16,
    /// each output's `k` taps in turn, a tap's channels padded with zeros to a multiple of 32. None where a value is past
    /// f16's range, or the device has no such kernel.
    fn conv1d_weights(&self, w: &[f32], cout: usize, cin: usize, k: usize) -> Option<DeviceVec> {
        let _ = (w, cout, cin, k);
        None
    }
    /// A convolution's weights (`[cout, cin, k, k]` as PyTorch keeps them, `k` 1, 3 or 7) for
    /// [`ChainRecorder::conv_rows`]: f16 (each rounded to the nearest), each output's `k * k` taps in turn, a tap's
    /// channels padded with zeros to a multiple of 32. None where a value is past f16's range, or the device has no such
    /// kernel.
    fn conv_weights(&self, w: &[f32], cout: usize, cin: usize, k: usize) -> Option<DeviceVec> {
        let _ = (w, cout, cin, k);
        None
    }
    /// A 3x3x3 convolution's weights (`[cout, cin, 3, 3, 3]`, PyTorch's) for [`ChainRecorder::conv3d_rows`]: f16, each
    /// output's 27 taps (time, row, column) in turn, a tap's channels padded with zeros to a multiple of 32. None where
    /// a value is past f16's range, or the device has no such kernel.
    fn conv3d_weights(&self, w: &[f32], cout: usize, cin: usize) -> Option<DeviceVec> {
        let _ = (w, cout, cin);
        None
    }
    /// An NVFP4 weight (`[rows, cols]`: `packed` two E2M1 values a byte, high nibble first, `rows x cols / 2`; `scales`
    /// one E4M3 a block of 16, row-major `rows x cols / 16`; `global` the tensor's own) for
    /// [`ChainRecorder::matmul_nvfp4_rows`]: its words, and its scale's vector. None where `cols` is not of 64 or the
    /// device has no such kernel.
    fn nvfp4_weights(&self, packed: &[u8], scales: &[u8], global: f32, rows: usize, cols: usize) -> Option<(DeviceVec, DeviceVec)> {
        let _ = (packed, scales, global, rows, cols);
        None
    }
    /// The length [`ChainRecorder::attention_rows_full`]'s `out` takes (the result, then any scratch its kernels
    /// keep there).
    fn attention_rows_full_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
        self.attention_rows_out_len(rows, n_h, head_dim, kv_len)
    }
    /// Zero `v` (before whatever is recorded next runs).
    fn zero(&self, v: &DeviceVec);
    /// A tensor of `shape` whose storage is `v` itself, not a copy (a model's recurrent state kept on the device in its
    /// cache between steps): its host copy reads `v` back as it is then, and a clone of it is a copy of `v`.
    fn alias(&self, v: &DeviceVec, shape: Vec<usize>) -> Tensor;
    /// The vector `t` is, if [`DeviceChain::alias`] made it on this device.
    fn aliased(&self, t: &Tensor) -> Option<DeviceVec>;
    /// Write `data` into `v` from element `offset` (before whatever is recorded next runs).
    fn upload_at(&self, v: &DeviceVec, offset: usize, data: &[f32]);
    /// Write `data` into `v` from its start.
    fn upload(&self, v: &DeviceVec, data: &[f32]) {
        self.upload_at(v, 0, data)
    }
    /// A vector of `len` holding `v`'s first `min(len, v.len)` elements (the rest zero).
    fn resize(&self, v: &DeviceVec, len: usize) -> DeviceVec;
    /// Whether this device holds `w` where its matmuls read it.
    fn holds(&self, w: &QuantizedTensor) -> bool;
    /// Whether this device holds the EXL3 projection `w` where [`ChainRecorder::exl3_rows`] reads it.
    fn holds_exl3(&self, w: &dyn crate::exl3::PackedLinear) -> bool;
    /// Whether this device holds a MoE layer's experts as groups [`ChainRecorder::moe_rows`] runs.
    fn holds_experts(&self, e: &dyn crate::exl3::Experts) -> bool;
    /// A copy of `w`, a weight another device of this one's API holds, here (as that one holds it, read back and put
    /// here): None where it cannot be (another API's weight, this device's own, or no room for it).
    fn copy_weight(&self, w: &QuantizedTensor) -> Option<QuantizedTensor> {
        let _ = w;
        None
    }
    /// From here on the device's queue holds at most `pieces` of its recordings' pieces at once (0: as many as are
    /// recorded, as a device starts), where it has such a queue: for a run of many pieces one after another (a
    /// prompt's chunks, a sampler's steps), which a card under a power limit otherwise throttles itself through.
    fn pieces_in_flight_at_most(&self, pieces: usize) {
        let _ = pieces;
    }
    /// Whether the device is known to have room for `bytes` more of vectors, and some to spare: false where it does
    /// not say what it has (a vector it has no room for fails the run that asked for it).
    fn has_room(&self, bytes: u64) -> bool {
        let _ = bytes;
        false
    }
    /// Whether a step's attention of `n_h` heads of `head_dim` over `n_kv` KV heads can read its cache as f16 halves
    /// ([`ChainRecorder::halve`], [`ChainRecorder::attention_halved`]).
    fn attention_halves(&self, n_h: usize, n_kv: usize, head_dim: usize) -> bool {
        let _ = (n_h, n_kv, head_dim);
        false
    }
    /// The length [`ChainRecorder::attention`]'s `out` needs for `n_h` heads of `head_dim` over a cache of `cap` rows:
    /// the result and the device's scratch.
    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize;
    /// The length [`ChainRecorder::attention_rows`]'s `out` needs for `rows` queries over `kv_len` positions.
    fn attention_rows_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize;
    /// The length of [`ChainRecorder::qsa_attention`]'s `out` for `rows` queries of `keep` blocks of `ratio` (its
    /// results first, `[rows, n_h, head_dim]`), or 0 where the device has no such kernels.
    fn qsa_attention_out_len(&self, rows: usize, n_h: usize, head_dim: usize, keep: usize, ratio: usize) -> usize {
        let _ = (rows, n_h, head_dim, keep, ratio);
        0
    }
    /// Start recording.
    fn begin(&self) -> Box<dyn ChainRecorder + '_>;
}

/// Ops recorded in order, run together by [`ChainRecorder::finish`].
pub trait ChainRecorder {
    /// `y = W x` for one row `x` (`[k]`, its first `k` elements) of a weight the device holds (`[n, k]`).
    fn matmul(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec) {
        self.matmul_rows(w, x, y, 1)
    }
    /// `y[r] = W x[r]` for `rows` rows of `x` (`[rows, k]`) into `y` (`[rows, n]`): a prompt's.
    fn matmul_rows(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, rows: usize);
    /// `out = x / rms(x) * w`.
    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32);
    /// [`Self::rmsnorm`] of each of `rows` rows of `x` (a head's q or k), all with the same `w` (`[x.len / rows]`).
    fn rmsnorm_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32);
    /// [`Self::rmsnorm_rows`] of each head of `rows` rows (`x` `[rows, heads, d]`, `d` from its length), head `h` by row
    /// `h` of `w` (`[heads, d]`): a multi-head RMS norm whose heads have weights of their own (TRELLIS's).
    fn rmsnorm_heads_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, heads: usize, eps: f32) {
        let _ = (x, w, out, rows, heads, eps);
        unimplemented!("a multi-head RMS norm on this device")
    }
    /// `acc += y`.
    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec);
    /// [`Self::rmsnorm_rows`] then SiLU, in one pass (a VAE's norms before its convolutions: no normed copy).
    fn rmsnorm_silu_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        let _ = (x, w, out, rows, eps);
        unimplemented!("a norm and SiLU on this device")
    }
    /// `x += y`, then `out` its rows' [`Self::rmsnorm_rows`] with `w`: a residual's add and the next norm in one.
    fn add_rmsnorm_rows(&mut self, x: &DeviceVec, y: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        self.add(x, y);
        self.rmsnorm_rows(x, w, out, rows, eps);
    }
    /// `out = silu(fused[..ff]) * fused[ff..]`, `ff = out.len`.
    fn silu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec) {
        self.silu_mul_split_rows(fused, out, 1)
    }
    /// `out = gelu_approx(fused[..ff]) * fused[ff..]` (the tanh approximation), `ff = out.len`.
    fn gelu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec) {
        self.gelu_mul_split_rows(fused, out, 1)
    }
    /// [`Self::silu_mul_split`] of each of `rows` rows (`fused` `[rows, 2 ff]`, `out` `[rows, ff]`).
    fn silu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize);
    /// [`Self::gelu_mul_split`] of each of `rows` rows.
    fn gelu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize);
    /// Whether this recording's bind groups are kept for the steps after (true at first). A prompt's chunk, whose
    /// vectors are its own, keeps none, so they do not hold its buffers' memory after it.
    fn keep_groups(&mut self, keep: bool) {
        let _ = keep;
    }
    /// Hold what is recorded until [`Self::finish`] submits it all (none of it submitted as it is recorded): a
    /// recording whose inputs are uploaded after it, as the next device's of a chain over several is while the last
    /// device runs.
    fn hold(&mut self) {}
    /// Submit what is recorded so far (its reads still [`Self::finish`]'s): a device's work running while the next
    /// device's is recorded.
    fn flush(&mut self) {}
    /// Rotate `x` (`[heads, head_dim]`) in place: pair `k` of each head by `table[2k]` (sine) and `table[2k + 1]`
    /// (cosine), the pairs `(2k, 2k + 1)` or with `neox` `(k, k + head_dim / 2)`.
    fn rope(&mut self, x: &DeviceVec, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool) {
        self.rope_rows(x, 1, heads, head_dim, table, neox)
    }
    /// [`Self::rope`] of each of `rows` rows (`x` `[rows, heads, head_dim]`), row `r` by `table[r * head_dim..]`.
    fn rope_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool);
    /// `rows` rows of `len` from `src` into `dst`'s rows `start..start + rows`, a row `stride` long, at `at` in it.
    #[allow(clippy::too_many_arguments)]
    fn store_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, len: usize, start: usize, stride: usize, at: usize);
    /// A prompt's attention: query `s` of `rows` (`q` `[rows, n_h, head_dim]`, at position `past + s`) over the cache
    /// `kv` (as [`Self::attention`]'s), positions up to its own and, with `window`, its last `window`; `out` the result
    /// `[rows, n_h, head_dim]` then scratch ([`DeviceChain::attention_rows_out_len`] long).
    #[allow(clippy::too_many_arguments)]
    fn attention_rows(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32);
    /// `dst[offset..offset + src.len] = src`.
    fn store(&mut self, src: &DeviceVec, dst: &DeviceVec, offset: usize) {
        self.copy(src, 0, dst, offset, src.len)
    }
    /// `dst[dst_at..dst_at + len] = src[src_at..src_at + len]`.
    fn copy(&mut self, src: &DeviceVec, src_at: usize, dst: &DeviceVec, dst_at: usize, len: usize);
    /// One query's attention over a layer's cache `kv` (row `t`: its K `[n_kv, head_dim]` then its V), positions
    /// `lo..kv_len`, each query head on its KV head (`h / (n_h / n_kv)`): `out[..n_h * head_dim]` the result, the rest
    /// of `out` scratch ([`DeviceChain::attention_out_len`] long, `cap` the cache's rows).
    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32);
    /// `len` of `src`'s values from `at` (both even) as f16 in `dst`, two a word, its words `at / 2..`: a cache's rows as
    /// the halves [`Self::attention_halved`] reads. Only where [`DeviceChain::attention_halves`].
    fn halve(&mut self, src: &DeviceVec, dst: &DeviceVec, at: usize, len: usize) {
        let _ = (src, dst, at, len);
        unreachable!("chain: no f16 halves on this device")
    }
    /// [`Self::attention`] over the cache as f16 halves (`kv` what [`Self::halve`] made of the cache's rows: `cap` rows
    /// of `n_kv * head_dim` words): half the bytes a step reads of a long cache, which bound it. Only where
    /// [`DeviceChain::attention_halves`].
    #[allow(clippy::too_many_arguments)]
    fn attention_halved(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        let _ = (q, kv, out, n_h, n_kv, head_dim, lo, kv_len, cap, scale);
        unreachable!("chain: no f16 halves on this device")
    }
    /// `dst[r * width + i] = src[r * stride + at + i]` for each of `rows` rows: a run of columns of each row of `src`
    /// (the query or the gate half of each head of Qwen3.5's q).
    #[allow(clippy::too_many_arguments)]
    fn copy_cols(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, width: usize, stride: usize, at: usize);
    /// RoPE on the first `rot` of each head's `head_dim` in place, NeoX pairs `(k, k + rot / 2)`, row `r` of `rows` by
    /// `table[r * rot..]` (each pair's sine then cosine); the rest of each head as it is (Qwen3.5's partial rotation).
    fn rope_partial_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, rot: usize, table: &DeviceVec);
    /// `out[i] = x[i] * sigmoid(gate[i])` for `i < len` (Qwen3.5's gated attention output).
    fn mul_sigmoid(&mut self, x: &DeviceVec, gate: &DeviceVec, out: &DeviceVec, len: usize);
    /// `out[i] = silu(gate[i]) * up[i]` for `i < len` (a SwiGLU whose gate and up are two weights).
    fn silu_mul(&mut self, gate: &DeviceVec, up: &DeviceVec, out: &DeviceVec, len: usize);
    /// `y[r] = W x[r]` for `rows` rows of `x` (`[rows, k]`), `W` f32 weights `[n, k]` (row-major) held in `w`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_f32_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize);
    /// An n-gram layer's gate ([`crate::Backend::ple_gate`]) of `rows` rows of `streams` streams of `d`: `gated` and
    /// `conv_in` (each `[rows, streams * d]`) from `key` and `x` (`[rows, streams * d]`), `value` (`[rows, d]`) and the
    /// norms (`[streams * d]`).
    #[allow(clippy::too_many_arguments)]
    fn ple_gate(&mut self, key: &DeviceVec, x: &DeviceVec, value: &DeviceVec, norm_key: &DeviceVec, norm_query: &DeviceVec, norm_conv: &DeviceVec, gated: &DeviceVec, conv_in: &DeviceVec, rows: usize, streams: usize, d: usize, eps: f32) {
        let _ = (key, x, value, norm_key, norm_query, norm_conv, gated, conv_in, rows, streams, d, eps);
        unreachable!("an n-gram layer's gate on a device without one")
    }
    /// An n-gram layer's dilated causal conv ([`crate::Backend::ple_conv`]) of `rows` rows of `width`: `x += gated +
    /// silu(conv)` over `window` (`[(kernel - 1) * dilation, width]`, left with the stream's last rows) then `conv_in`,
    /// `weight` `[width, kernel]`.
    #[allow(clippy::too_many_arguments)]
    fn ple_conv(&mut self, x: &DeviceVec, gated: &DeviceVec, conv_in: &DeviceVec, window: &DeviceVec, weight: &DeviceVec, rows: usize, width: usize, kernel: usize, dilation: usize) {
        let _ = (x, gated, conv_in, window, weight, rows, width, kernel, dilation);
        unreachable!("an n-gram layer's conv on a device without one")
    }
    /// [`Self::matmul_f32_rows`] of a matrix [`DeviceChain::vec_f16`] made (`k` even): half the bytes read.
    #[allow(clippy::too_many_arguments)]
    fn matmul_f16_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        let _ = (w, n, k, x, y, rows);
        unreachable!("a device that makes no f16 matrices has none to multiply")
    }
    /// A gated delta net's causal depthwise conv over `rows` tokens of `qkv` (`[rows, channels]`) after the
    /// `kernel - 1` inputs `state` holds (`[kernel - 1, channels]`, oldest first), with `weight` (`[channels, kernel]`),
    /// through SiLU into `out` (`[rows, channels]`); `state` then holds the last `kernel - 1` inputs. As the host's
    /// (`Backend::delta_net_step`).
    #[allow(clippy::too_many_arguments)]
    fn ssm_conv(&mut self, qkv: &DeviceVec, weight: &DeviceVec, state: &DeviceVec, out: &DeviceVec, rows: usize, channels: usize, kernel: usize);
    /// A gated delta net's recurrence over `d.rows` tokens, as the host's (`Backend::delta_net_step`): each token's
    /// conv output `conv` (`[rows, 2 k_heads k_dim + v_heads v_dim]`: q, k, v), gate `z` (`[rows, v_heads v_dim]`)
    /// and `beta_alpha` (`[rows, 2 v_heads]`) with the layer's `ssm_a`, `dt_bias` (`[v_heads]`) and `norm`
    /// (`[v_dim]`) update `state` (`[v_heads, v_dim, k_dim]`, the host's layout) and give the norm-gated output `out`
    /// (`[rows, v_heads v_dim]`). Head `h` reads key head `h % k_heads`.
    #[allow(clippy::too_many_arguments)]
    fn delta_net(&mut self, conv: &DeviceVec, z: &DeviceVec, beta_alpha: &DeviceVec, ssm_a: &DeviceVec, dt_bias: &DeviceVec, norm: &DeviceVec, state: &DeviceVec, out: &DeviceVec, d: DeltaNet);
    /// `y[r] = W x[r]` for `rows` rows of `x` through an EXL3 projection this device holds
    /// ([`DeviceChain::holds_exl3`]): its input and output transforms (the channel maps, the scales, the Hadamard
    /// transforms and their f16 roundings) on the device as the host makes them.
    fn exl3_rows(&mut self, w: &dyn crate::exl3::PackedLinear, x: &DeviceVec, y: &DeviceVec, rows: usize);
    /// [`Self::rmsnorm_rows`] of each of `rows * streams` rows of `x`, row `r`'s stream `s` by `w[s * d..]` (`d` its
    /// width): a hyper-connection's per-stream norm (`Backend::hc_norm`).
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_streams(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, eps: f32);
    /// A hyper-connection's gates (`Backend::hc_gates`): of each of `rows` rows of `t` (`[rows, rank + writes]`) the
    /// first `rank` become `silu(t / streams)`, the rest 0, and `post` (`[rows, writes]`) gets `2 sigmoid(t /
    /// streams)` of them.
    #[allow(clippy::too_many_arguments)]
    fn hc_gates(&mut self, t: &DeviceVec, post: &DeviceVec, rows: usize, rank: usize, writes: usize, streams: usize);
    /// A hyper-connection's branch input (`Backend::hc_mix`): `out[r] = sum over s of normed[r, s] *
    /// sigmoid(logits[r, s]) / streams`, rows of `d`.
    #[allow(clippy::too_many_arguments)]
    fn hc_mix(&mut self, logits: &DeviceVec, normed: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, d: usize);
    /// A hyper-connection's down projection and its gates: [`Self::matmul_f16_rows`] of `w` (`[rank + writes, k]`, as
    /// [`DeviceChain::vec_f16`] made it) on `rows` rows of `x` into `t`, then [`Self::hc_gates`] of `t`. One dispatch
    /// where the device has the two as one kernel (the same sums and the same gates, bit for bit).
    #[allow(clippy::too_many_arguments)]
    fn hc_down_gates(&mut self, w: &DeviceVec, k: usize, x: &DeviceVec, t: &DeviceVec, post: &DeviceVec, rows: usize, rank: usize, writes: usize, streams: usize) {
        self.matmul_f16_rows(w, rank + writes, k, x, t, rows);
        self.hc_gates(t, post, rows, rank, writes, streams);
    }
    /// A hyper-connection's up projection and its mix: [`Self::matmul_f16_rows`] of `w` (`[streams d, k]`) on `rows`
    /// rows of `t` into `logits`, then [`Self::hc_mix`] of them with `normed` into `out`. One dispatch where the
    /// device has the two as one kernel (bit for bit the same `out`; `logits` is then not written).
    #[allow(clippy::too_many_arguments)]
    fn hc_up_mix(&mut self, w: &DeviceVec, k: usize, t: &DeviceVec, logits: &DeviceVec, normed: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, d: usize) {
        self.matmul_f16_rows(w, streams * d, k, t, logits, rows);
        self.hc_mix(logits, normed, out, rows, streams, d);
    }
    /// A hyper-connection site's write-back (`Backend::stream_apply`): `x[r, s] += post[r, s] * y[r]`, rows of `d`.
    #[allow(clippy::too_many_arguments)]
    fn stream_apply(&mut self, x: &DeviceVec, y: &DeviceVec, post: &DeviceVec, rows: usize, streams: usize, d: usize);
    /// A MoE layer's experts the device holds as groups, for each of `assign.len()` rows of `x` (`[rows, hidden]`):
    /// row `r`'s experts `assign[r]` (expert, weight; the routed ones, then the shared one last, as
    /// `Experts::forward` routes them), their weighted outputs summed in that order into `out` (`[rows, hidden]`).
    fn moe_rows(&mut self, experts: &dyn crate::exl3::Experts, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]);
    /// [`Self::moe_rows`] of `rows` rows (a step's one, a check's few), their experts routed on the device from the
    /// router's logits `logits` (`[rows, routed + 1]`, the shared expert's gate last) as [`crate::exl3::route`] routes
    /// them, `top_k` of them: no trip to the host between a layer's router and its experts. False where the device
    /// cannot (and nothing is recorded): the caller routes on the host.
    #[allow(clippy::too_many_arguments)]
    fn moe_routed(&mut self, experts: &dyn crate::exl3::Experts, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize) -> bool {
        let _ = (experts, x, out, logits, top_k, rows);
        false
    }
    /// [`Self::moe_routed`] with each row's sum added to its streams as a hyper-connection site writes it back
    /// ([`Self::stream_apply`]: `streams_x[r, s] += post[r, s] * sum[r]`, `streams` of them), where `moe_routed` puts
    /// it in a vector for a write-back after: a dispatch fewer. False where the device cannot (nothing recorded).
    #[allow(clippy::too_many_arguments)]
    fn moe_routed_into(&mut self, experts: &dyn crate::exl3::Experts, x: &DeviceVec, streams_x: &DeviceVec, post: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, streams: usize) -> bool {
        let _ = (experts, x, streams_x, post, logits, top_k, rows, streams);
        false
    }
    /// The shared expert of a layer whose routed experts are on the host (`Experts::on_host`), where this device
    /// holds it: its output on each of `rows` rows of `x` into `out` (`[rows, hidden]`, not weighted), for
    /// `Experts::forward_given`. False where it does not (nothing recorded: the host runs the shared expert too).
    fn moe_shared(&mut self, experts: &dyn crate::exl3::Experts, x: &DeviceVec, out: &DeviceVec, rows: usize) -> bool {
        let _ = (experts, x, out, rows);
        false
    }
    /// `acc[i] += weights[at] * y[i]` for `i < len`: a weighted sum's term, its weight read from the device (an
    /// expert's, written by the host before the chain runs).
    fn axpy_at(&mut self, acc: &DeviceVec, y: &DeviceVec, weights: &DeviceVec, at: usize, len: usize);
    /// A draft's token and its probability from logits `x`: `out[0]` the index of the largest (the first of equals,
    /// an f32's bits), `out[1]` the largest, `out[2]` the sum of `exp(x[i] - out[1])` (the token's probability its
    /// inverse), where reading the logits back cost a draft a millisecond.
    fn argmax_softmax(&mut self, x: &DeviceVec, out: &DeviceVec);
    /// QSA's pooled block keys (`Backend::qsa_pool`): `pooled[b] = mean of raw rows b ratio .. (b + 1) ratio`, the
    /// first `blocks` blocks of `raw` (`[.., d]`).
    fn qsa_pool(&mut self, raw: &DeviceVec, pooled: &DeviceVec, blocks: usize, ratio: usize, d: usize) {
        let _ = (raw, pooled, blocks, ratio, d);
        unimplemented!("QSA on this device")
    }
    /// QSA's block scores (`Backend::qsa_block_scores`): `scores[r, j] = scale * sum over the heads of relu(q[r, h] .
    /// pooled[j])` for the `nb` blocks query `r` (at position `first + r`) sees whole, `-inf` for the rest.
    #[allow(clippy::too_many_arguments)]
    fn qsa_scores(&mut self, q: &DeviceVec, pooled: &DeviceVec, scores: &DeviceVec, rows: usize, heads: usize, d: usize, nb: usize, first: usize, ratio: usize, scale: f32) {
        let _ = (q, pooled, scores, rows, heads, d, nb, first, ratio, scale);
        unimplemented!("QSA on this device")
    }
    /// QSA's selection (`Backend::qsa_select`): each query's `keep` best whole blocks (all of them where it sees no
    /// more), the larger score first and the lower block of equals, into `list[r, ..keep]` in ascending order. At most
    /// 4096 blocks (16,384 positions of 4).
    #[allow(clippy::too_many_arguments)]
    fn qsa_select(&mut self, scores: &DeviceVec, list: &DeviceVec, rows: usize, nb: usize, first: usize, ratio: usize, keep: usize) {
        let _ = (scores, list, rows, nb, first, ratio, keep);
        unimplemented!("QSA on this device")
    }
    /// QSA's attention (`Backend::sparse_attention`): each query `r` (at position `first + r`) over its blocks in
    /// `list` (ascending) then its incomplete tail block, as the decode attention sums them; the cache `kv` as
    /// [`Self::attention_rows`] reads it, `out` [`DeviceChain::qsa_attention_out_len`] long, its first `rows * n_h *
    /// head_dim` the results.
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) {
        let _ = (q, kv, list, out, rows, n_h, n_kv, head_dim, first, ratio, keep, scale);
        unimplemented!("QSA on this device")
    }
    /// [`Self::matmul_f16_rows`] with `x`'s values as they are (f32), never rounded to f16 on the way: an activation
    /// that may pass f16's range (a diffusion transformer's text prefix).
    #[allow(clippy::too_many_arguments)]
    fn matmul_f16_rows_f32(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.matmul_f16_rows(w, n, k, x, y, rows)
    }
    /// Each of `rows` rows of `x` (`[rows, n]`) layer-normed (no weights; `eps` added to the variance) into `out`, then
    /// times `1 + mods[scale_at..scale_at + n]` and, where `shift_at` is given, plus `mods[shift_at..shift_at + n]`: a
    /// diffusion transformer's modulated norm, every row by the same modulation.
    #[allow(clippy::too_many_arguments)]
    fn layernorm_mod_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, scale_at: usize, shift_at: Option<usize>, eps: f32) {
        self.norm_mod_rows(x, out, rows, n, mods, scale_at, shift_at, RowNorm::Layer, eps)
    }
    /// [`Self::layernorm_mod_rows`] with the row's norm as `norm` says (none, an RMS norm, or a layer norm), each row's
    /// modulation the same.
    #[allow(clippy::too_many_arguments)]
    fn norm_mod_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, scale_at: usize, shift_at: Option<usize>, norm: RowNorm, eps: f32) {
        self.norm_mod_rows_clean(x, out, rows, n, mods, scale_at, shift_at, norm, eps, CleanRows::NONE)
    }
    /// [`Self::norm_mod_rows`] with `clean`'s rows modulated by the second set (`clean.offset` on in `mods`).
    #[allow(clippy::too_many_arguments)]
    fn norm_mod_rows_clean(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, scale_at: usize, shift_at: Option<usize>, norm: RowNorm, eps: f32, clean: CleanRows) {
        let _ = (x, out, rows, n, mods, scale_at, shift_at, norm, eps, clean);
        unimplemented!("a modulated norm on this device")
    }
    /// RoPE of `x` (`[rows, heads, head_dim]`) in place over each head's halves (NeoX pairs `(k, k + head_dim / 2)`),
    /// each (row, head) by its own `table[(r * heads + h) * head_dim..]` (each pair's sine then cosine): LTX's split
    /// rotary, its frequencies spread over the heads.
    fn rope_split_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec) {
        let _ = (x, rows, heads, head_dim, table);
        unimplemented!("a split rotary on this device")
    }
    /// A group norm of an image held as its pixels' rows of channels (`x`, `[pixels, c]`) into `out`: its channels in
    /// `groups` runs, each run normalised over all the pixels (the mean and the variance of its `pixels * c / groups`
    /// values, `eps` added to the variance), then times `weight[ch]` plus `bias[ch]`, and with `silu` through SiLU
    /// (a UNet's norm before each convolution). `stats` scratch of `groups * (pixels / 256 + 2) * 2` values.
    #[allow(clippy::too_many_arguments)]
    fn group_norm_rows(&mut self, x: &DeviceVec, weight: &DeviceVec, bias: &DeviceVec, out: &DeviceVec, stats: &DeviceVec, pixels: usize, c: usize, groups: usize, eps: f32, silu: bool) {
        let _ = (x, weight, bias, out, stats, pixels, c, groups, eps, silu);
        unimplemented!("a group norm on this device")
    }
    /// GEGLU of each of `rows` rows: `out = fused[..ff] * gelu(fused[ff..])`, the GELU exact (`fused` `[rows, 2 ff]`,
    /// `out` `[rows, ff]`): a UNet's transformer blocks' feed-forward gate.
    fn geglu_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize, ff: usize) {
        let _ = (fused, out, rows, ff);
        unimplemented!("GEGLU on this device")
    }
    /// `out` (`[h / 2, w / 2, c]`) the even rows' even columns of `x` (`[h, w, c]`): a 3x3 convolution of stride 1
    /// so sampled is one of stride 2 padded by 1 all round (PyTorch's, a UNet's downsampling).
    fn subsample2x_even_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        let _ = (x, out, h, w, c);
        unimplemented!("subsampling on this device")
    }
    /// Normalised attention guidance's mix of an attention's output `pos` and the same queries' over a negative
    /// context, `neg` (`rows` rows of `width` each), into `pos`: `guided = pos * scale - neg * (scale - 1)`, each row
    /// of it scaled back to `tau` times `pos`'s row where its L1 norm is more than that (`min(1, tau * (|pos| + 1e-6)
    /// / |guided|)`), then `guided * alpha + pos * (1 - alpha)`.
    #[allow(clippy::too_many_arguments)]
    fn nag_mix(&mut self, pos: &DeviceVec, neg: &DeviceVec, rows: usize, width: usize, scale: f32, tau: f32, alpha: f32) {
        let _ = (pos, neg, rows, width, scale, tau, alpha);
        unimplemented!("normalised attention guidance on this device")
    }
    /// `y[r, h, ..] *= 2 sigmoid(logits[r, h])` for `rows` rows of `heads` heads of `head_dim`: a gated attention's
    /// per-head gate.
    fn head_gate_rows(&mut self, y: &DeviceVec, logits: &DeviceVec, rows: usize, heads: usize, head_dim: usize) {
        let _ = (y, logits, rows, heads, head_dim);
        unimplemented!("a head gate on this device")
    }
    /// `x[r] += y[r] * g` for each of `rows` rows of `n`, `g` `mods[gate_at..gate_at + n]` or (with `tanh`) its tanh:
    /// a diffusion transformer's gated residual.
    #[allow(clippy::too_many_arguments)]
    fn add_gated_rows(&mut self, x: &DeviceVec, y: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, gate_at: usize, tanh: bool) {
        self.add_gated_rows_clean(x, y, rows, n, mods, gate_at, tanh, CleanRows::NONE)
    }
    /// [`Self::add_gated_rows`] with `clean`'s rows gated by the second set (`clean.offset` on in `mods`).
    #[allow(clippy::too_many_arguments)]
    fn add_gated_rows_clean(&mut self, x: &DeviceVec, y: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, gate_at: usize, tanh: bool, clean: CleanRows) {
        let _ = (x, y, rows, n, mods, gate_at, tanh, clean);
        unimplemented!("a gated residual on this device")
    }
    /// `out` (`[h / 2, w / 2, c]`) the odd rows' odd columns of `x` (`[h, w, c]`): a 3x3 convolution of stride 1 so
    /// sampled is Wan's downsampling one of stride 2 (its input padded right and below).
    fn subsample2x_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        let _ = (x, out, h, w, c);
        unimplemented!("a subsampling on this device")
    }
    /// `out[oy, ox, co] += ` the mean of `x`'s (`[h, w, cin]`) group `co` of its space-to-depth (each output pixel's
    /// `cin * ft * fs * fs` values in turn by channel, time slot, row and column; the time slots before the last zero):
    /// Wan's encoder's shortcut, added to its downsampled output (`[h / fs, w / fs, cout]`).
    #[allow(clippy::too_many_arguments)]
    fn shuffle_down_mean_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, cin: usize, cout: usize, ft: usize, fs: usize) {
        let _ = (x, out, h, w, cin, cout, ft, fs);
        unimplemented!("a shuffled mean on this device")
    }
    /// `out` (`[h / sh, w / sw, c st sh sw]`) `x`'s (`[h, w, c]`) space-to-depth, each output pixel's channels in turn
    /// by channel, time slot, row and column, its `st` time slots all one frame's: LTX's encoder's packing of a single
    /// image (its first frame repeated where time halves).
    #[allow(clippy::too_many_arguments)]
    fn space_to_depth_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, st: usize, sh: usize, sw: usize) {
        let _ = (x, out, h, w, c, st, sh, sw);
        unimplemented!("a space to depth on this device")
    }
    /// `out[r, co] += ` the mean of `x[r, co g .. (co + 1) g]` (`g = cin / cout`) for `rows` rows: LTX's encoder's
    /// shortcut, its packed input's groups averaged.
    fn group_mean_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, cin: usize, cout: usize) {
        let _ = (x, out, rows, cin, cout);
        unimplemented!("a group mean on this device")
    }
    /// `w` (an f16 matrix's `len` values, two to a word) plus `d`'s (f32), each sum rounded to f16: a LoRA's `B A`
    /// merged into its weight on the device.
    fn add_f16(&mut self, w: &DeviceVec, d: &DeviceVec, len: usize) {
        let _ = (w, d, len);
        unimplemented!("an f16 matrix's sum on this device")
    }
    /// `out[i] = gelu(x[i])` exactly (`x (1 + erf(x / sqrt 2)) / 2`) for `i < len`: a vision tower's mergers'.
    fn gelu_erf(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize) {
        let _ = (x, out, len);
        unimplemented!("an exact GELU on this device")
    }
    /// `out[i] = gelu(x[i])` (the tanh approximation) for `i < len`.
    fn gelu(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize) {
        let _ = (x, out, len);
        unimplemented!("GELU on this device")
    }
    /// [`Self::attention_rows`] with no causal mask: each of `rows` queries over all `kv_len` positions of `kv` (a
    /// diffusion transformer's image tokens over a text prefix's and their own, or a video's over a text's: fewer
    /// positions than queries); `out` [`DeviceChain::attention_rows_full_out_len`] long.
    #[allow(clippy::too_many_arguments)]
    fn attention_rows_full(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, kv_len: usize, scale: f32) {
        let _ = (q, kv, out, rows, n_h, n_kv, head_dim, kv_len, scale);
        unimplemented!("full attention on this device")
    }
    /// A `k` x `k` convolution (`k` 1, 3 or 7; stride 1, zeros past the edge) of an image `x` held as its pixels' rows of
    /// channels (`[h * w, cin]`, a row of the image after another) into `y` (`[h * w, cout]`), the weights `w` as
    /// [`DeviceChain::conv_weights`] packs them, plus the bias `b`; `x` of any range (a VAE's reach some 230,000, past
    /// f16's: where the device multiplies in f16, scaled by a power of two its largest sets, undone after).
    #[allow(clippy::too_many_arguments)]
    fn conv_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, x: &DeviceVec, h: usize, wd: usize, y: &DeviceVec) {
        let _ = (w, b, cout, cin, k, x, h, wd, y);
        unimplemented!("a convolution on this device")
    }
    /// [`Self::conv_rows`] of `x`'s pixels `xs` values apart, each its first `cin` (a concatenation's leading channels:
    /// Real-ESRGAN's dense blocks).
    #[allow(clippy::too_many_arguments)]
    fn conv_rows_strided(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, x: &DeviceVec, xs: usize, h: usize, wd: usize, y: &DeviceVec) {
        if xs == cin {
            return self.conv_rows(w, b, cout, cin, k, x, h, wd, y);
        }
        let _ = (w, b, cout, cin, k, x, xs, h, wd, y);
        unimplemented!("a strided convolution on this device")
    }
    /// ComfyUI's W4A8 matrix (`rows` by `cols`) decoded into `out` as f16 pairs (`rows * cols / 2` words, the even
    /// column the low half): `codes` its codes two to a byte (the even column the low nibble) and `rel` its FP8 E4M3
    /// group scales (16 columns a scale), both as bytes four to a word; `channel` its rows' scales, `book` its codes'
    /// 16 values. A value is `round(book[code] rel)` (ties to even, within -127..127) times its row's scale; with
    /// `rotation` (a power of four; 0 none) each group that long of a row then times the Hadamard matrix ConvRot
    /// rotated by (the 4x4 `[[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]`'s Kronecker power over the
    /// group's square root).
    #[allow(clippy::too_many_arguments)]
    fn w4a8_f16(&mut self, codes: &DeviceVec, rel: &DeviceVec, channel: &DeviceVec, book: &DeviceVec, rows: usize, cols: usize, rotation: usize, out: &DeviceVec) {
        let _ = (codes, rel, channel, book, rows, cols, rotation, out);
        unimplemented!("W4A8's decode on this device")
    }
    /// Swin's windows: `x`'s `h` by `w` tokens of `c` (rows) into `out`'s windows of `win` by `win` (`[windows, win²,
    /// c]`, the windows row by row), the grid padded with zeros to whole windows (`hp` by `wp`) and, with `shift`,
    /// rolled up and left by it first (torch.roll by -shift).
    #[allow(clippy::too_many_arguments)]
    fn window_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, win: usize, shift: usize) {
        let _ = (x, out, h, w, c, win, shift);
        unimplemented!("Swin's windows on this device")
    }
    /// [`Self::window_rows`]' way back: each of the `h` by `w` tokens' place in `windows` added to its row of `acc` (the
    /// roll undone, the padding dropped).
    #[allow(clippy::too_many_arguments)]
    fn unwindow_add_rows(&mut self, windows: &DeviceVec, acc: &DeviceVec, h: usize, w: usize, c: usize, win: usize, shift: usize) {
        let _ = (windows, acc, h, w, c, win, shift);
        unimplemented!("Swin's windows on this device")
    }
    /// Swin's window attention, heads 32 wide: `qkv` the windows' tokens' queries, keys and values (`[windows win²,
    /// 3 heads 32]`, from [`Self::window_rows`] of a `h` by `w` grid), `table` the relative positions' bias (`[(2 win -
    /// 1)², heads]`), into `out` (`[windows win², heads 32]`); the queries scaled by `scale`; with `shift`, a key in
    /// another region of the rolled grid than its query's -100 (Swin's mask).
    #[allow(clippy::too_many_arguments)]
    fn window_attention(&mut self, qkv: &DeviceVec, table: &DeviceVec, out: &DeviceVec, h: usize, w: usize, heads: usize, win: usize, shift: usize, scale: f32) {
        let _ = (qkv, table, out, h, w, heads, win, shift, scale);
        unimplemented!("Swin's window attention on this device")
    }
    /// An image's rows (`[h * w, c]`) resized to `oh` by `ow` into `out`, bilinear with the corners aligned (PyTorch's
    /// `align_corners=True`).
    #[allow(clippy::too_many_arguments)]
    fn resize_bilinear_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, oh: usize, ow: usize) {
        let _ = (x, out, h, w, c, oh, ow);
        unimplemented!("a bilinear resize on this device")
    }
    /// A square picture's rows (`[s * s, c]`) as patches `size` a side, each channel of the result one whole block of
    /// it (`[size², c g²]`, `g = s / size`; einops' `c (hg h) (wg w) -> (c hg wg) h w`).
    fn blocks_to_channels_rows(&mut self, x: &DeviceVec, out: &DeviceVec, s: usize, c: usize, size: usize) {
        let _ = (x, out, s, c, size);
        unimplemented!("patches on this device")
    }
    /// A modulated deformable convolution's taps (torchvision's `deform_conv2d`) for `pixels` of an `h` by `w` image
    /// `x` (`[h w, c]`) from pixel `first`: `out` `[pixels, k² c]` (taps outer), tap `t` of pixel `(y, x)` bilinear at
    /// `(y - k/2 + t / k + dy, x - k/2 + t % k + dx)` (zeros outside), `(dy, dx)` its `offsets` (`[h w, 2 k²]`), times
    /// its `modulators` (`[h w, k²]`).
    #[allow(clippy::too_many_arguments)]
    fn deform_im2col_rows(&mut self, x: &DeviceVec, offsets: &DeviceVec, modulators: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, k: usize, first: usize, pixels: usize) {
        let _ = (x, offsets, modulators, out, h, w, c, k, first, pixels);
        unimplemented!("a deformable convolution on this device")
    }
    /// Each of `x`'s `rows` (`[rows, c]`) times the sigmoid of its `gate` (`[rows]`), in place.
    fn mul_sigmoid_rows(&mut self, x: &DeviceVec, gate: &DeviceVec, rows: usize, c: usize) {
        let _ = (x, gate, rows, c);
        unimplemented!("a gate of rows on this device")
    }
    /// `out`'s `rows` rows of `c` (from row `first` of `index`): row `r` is `x`'s row `index[first + r]` (`index` u32s
    /// in a vector's words), zeros where that is `src_rows` or past it: a sparse convolution's neighbours gathered (a
    /// missing one zeros), or rows picked out.
    #[allow(clippy::too_many_arguments)]
    fn gather_rows(&mut self, x: &DeviceVec, index: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, first: usize, src_rows: usize) {
        let _ = (x, index, out, rows, c, first, src_rows);
        unimplemented!("a gather of rows on this device")
    }
    /// `out[r, k] += x[r, k / repeat]` for each of `rows` rows (`x` `[rows, c]`, `out` `[rows, c repeat]`): each channel
    /// repeated `repeat` times, added (a sparse decoder's skip as it goes up a level).
    fn repeat_cols_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, repeat: usize) {
        let _ = (x, out, rows, c, repeat);
        unimplemented!("a repeat of columns on this device")
    }
    /// `out[ch]` the mean of `x`'s `rows` (`[rows, c]`) at `ch`.
    fn mean_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, c: usize) {
        let _ = (x, out, rows, c);
        unimplemented!("a mean on this device")
    }
    /// `src`'s `c` values into each of `rows` rows of `dst` (`stride` apart) at `at`.
    fn broadcast_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, c: usize, stride: usize, at: usize) {
        let _ = (src, dst, rows, c, stride, at);
        unimplemented!("a broadcast on this device")
    }
    /// A 1-D convolution of `x`'s `len` steps (`[len, cin]`) into `y` (`[len, cout]`): `k` taps `dilation` apart, padded
    /// to keep the length (zeros past the ends), the weights as [`DeviceChain::conv1d_weights`] packs them, plus `b`.
    #[allow(clippy::too_many_arguments)]
    fn conv1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, dilation: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        self.conv1d_padded_rows(w, b, cout, cin, k, dilation, k / 2 * dilation, x, len, y);
    }
    /// [`Self::conv1d_rows`] with `pad` zeros before the first step (a causal convolution's `(k - 1) dilation`; tap `t`
    /// of step `s` from step `s + t dilation - pad`), the length kept.
    #[allow(clippy::too_many_arguments)]
    fn conv1d_padded_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, dilation: usize, pad: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        let _ = (w, b, cout, cin, k, dilation, pad, x, len, y);
        unimplemented!("a 1-D convolution on this device")
    }
    /// A depthwise causal 1-D convolution (each channel its own `k` taps, `w` `[c, k]`, plus `b`) of `x`'s `len` steps
    /// (`[len, c]`) into `y`: step `s` from steps `s - k + 1 ..= s` (zeros before the first).
    #[allow(clippy::too_many_arguments)]
    fn depthwise_causal_conv1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, c: usize, k: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        let _ = (w, b, c, k, x, len, y);
        unimplemented!("a depthwise convolution on this device")
    }
    /// `x + scale sin²(freq x)` of `rows` of `c`, `freq` and `scale` a channel's, in place (SnakeBeta: `freq` its
    /// `e^alpha`, `scale` `1 / (e^beta + 1e-9)`).
    fn snake_beta_rows(&mut self, x: &DeviceVec, freq: &DeviceVec, scale: &DeviceVec, rows: usize, c: usize) {
        let _ = (x, freq, scale, rows, c);
        unimplemented!("SnakeBeta on this device")
    }
    /// SnakeBeta without aliasing (BigVGAN's `Activation1d`) of `x`'s `len` steps (`[len, c]`) into `y`: each channel
    /// upsampled twice over by `filters`' first `ku` taps (PyTorch's transposed convolution of stride 2 over the steps
    /// padded by their ends with `ku / 2 - 1` each side, times 2, its edges cropped to `2 len` steps), SnakeBeta of
    /// those (`freq` and `scale` as [`Self::snake_beta_rows`]) into `mid` (`[2 len, c]`), then low-passed by the `kd`
    /// taps after them over the steps padded by their ends (`kd / 2` after, that or one fewer before where `kd` is
    /// even), every second step kept. The filters are every channel's.
    #[allow(clippy::too_many_arguments)]
    fn snake_beta_alias_rows(&mut self, x: &DeviceVec, filters: &DeviceVec, ku: usize, kd: usize, freq: &DeviceVec, scale: &DeviceVec, len: usize, c: usize, mid: &DeviceVec, y: &DeviceVec) {
        let _ = (x, filters, ku, kd, freq, scale, len, c, mid, y);
        unimplemented!("SnakeBeta without aliasing on this device")
    }
    /// `len` values clamped to `lo..=hi`, in place.
    fn clamp_in_place(&mut self, x: &DeviceVec, len: usize, lo: f32, hi: f32) {
        let _ = (x, len, lo, hi);
        unimplemented!("a clamp on this device")
    }
    /// A transposed 1-D convolution (PyTorch's `ConvTranspose1d`, `pad` dropped from the start) of `x`'s `len` steps
    /// (`[len, cin]`): its first `out` steps into `y` (`[out, cout]`; PyTorch's whole output `(len - 1) stride - 2 pad +
    /// k + out_pad`, a causal one's `len stride`, its tail trimmed); `w` its weight `[cout, k, cin]` (f32, PyTorch's
    /// `[cin, cout, k]` rearranged), plus `b`.
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, stride: usize, pad: usize, x: &DeviceVec, len: usize, out: usize, y: &DeviceVec) {
        let _ = (w, b, cout, cin, k, stride, pad, x, len, out, y);
        unimplemented!("a transposed 1-D convolution on this device")
    }
    /// The Snake activation of `rows` of `c` (`x + sin(alpha x)² / (alpha + 1e-9)`, `alpha` a channel's), in place.
    fn snake_rows(&mut self, x: &DeviceVec, alpha: &DeviceVec, rows: usize, c: usize) {
        let _ = (x, alpha, rows, c);
        unimplemented!("Snake on this device")
    }
    /// `tanh` of `len` values, in place.
    fn tanh_in_place(&mut self, x: &DeviceVec, len: usize) {
        let _ = (x, len);
        unimplemented!("tanh on this device")
    }
    /// `out = x` where `x` is positive, else `slope x`, for `len` values (a leaky ReLU; `out` may be `x`).
    fn leaky_relu(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize, slope: f32) {
        let _ = (x, out, len, slope);
        unimplemented!("a leaky ReLU on this device")
    }
    /// `y[r] = W x[r] + b` for `rows` rows of `x` (`[rows, k]`), `W` an NVFP4 weight `[n, k]` as
    /// [`DeviceChain::nvfp4_weights`] made it (`w` its words, `scale` its scale's vector), `b` its bias (`[n]`).
    #[allow(clippy::too_many_arguments)]
    fn matmul_nvfp4_rows(&mut self, w: &DeviceVec, scale: &DeviceVec, b: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        let _ = (w, scale, b, n, k, x, y, rows);
        unimplemented!("NVFP4 on this device")
    }
    /// A 3x3x3 convolution (stride 1) of a video `x` held as its voxels' rows of channels (`[frames * h * w, cin]`, a
    /// frame after another) into `y` (`[frames * h * w, cout]`) plus the bias `b`: in time the first and last frames
    /// repeated past the clip's ends (LTX's decoder, not causal), zeros past each frame's edge; `x` of any range (as
    /// [`Self::conv_rows`]).
    #[allow(clippy::too_many_arguments)]
    fn conv3d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, x: &DeviceVec, frames: usize, h: usize, wd: usize, y: &DeviceVec) {
        let _ = (w, b, cout, cin, x, frames, h, wd, y);
        unimplemented!("a 3D convolution on this device")
    }
    /// Depth to space: a video `x` (`[frames * h * w, c * st * sh * sw]`) into `out` (`[(frames * st - drop) * (h *
    /// sh) * (w * sw), c]`), voxel `(d st + i, y sh + j, x sw + k)`'s channel `c'` the input's `(d, y, x)`'s `c' st sh
    /// sw + i sh sw + j sw + k`; with `drop` its first frame left out (LTX's after doubling time).
    #[allow(clippy::too_many_arguments)]
    fn depth_to_space_rows(&mut self, x: &DeviceVec, out: &DeviceVec, frames: usize, h: usize, w: usize, c: usize, st: usize, sh: usize, sw: usize, drop: usize) {
        let _ = (x, out, frames, h, w, c, st, sh, sw, drop);
        unimplemented!("depth to space on this device")
    }
    /// `y[r * n + i] += b[i]` for each of `rows` rows.
    fn add_bias_rows(&mut self, y: &DeviceVec, b: &DeviceVec, rows: usize, n: usize) {
        let _ = (y, b, rows, n);
        unimplemented!("a bias on this device")
    }
    /// An image held as its pixels' rows of channels (`[h * w, c]`) twice as wide and high into `out` (`[2h * 2w,
    /// c]`), each pixel four (nearest).
    fn upsample2x_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        let _ = (x, out, h, w, c);
        unimplemented!("an upsampling on this device")
    }
    /// Wan's upsampling shortcut added: `out` (`[2h * 2w, cout]`) gets `x` (`[h * w, cin]`) each channel repeated
    /// `cout ft 4 / cin` times then shuffled into pixels (`(cout, ft, 2, 2)`), the last of `ft` frames kept:
    /// `out[(2y + a, 2x + b), co] += x[(y, x), ci]`, `ci = (4 co ft + 4 (ft - 1) + 2 a + b) / repeats`.
    #[allow(clippy::too_many_arguments)]
    fn shuffle_up_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, cin: usize, cout: usize, ft: usize) {
        let _ = (x, out, h, w, cin, cout, ft);
        unimplemented!("an upsampling shortcut on this device")
    }
    /// Read `v` back once the chain has run.
    fn read(&mut self, v: &DeviceVec) {
        self.read_range(v, 0, v.len)
    }
    /// Read `v[offset..offset + len]` back once the chain has run.
    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize);
    /// Run what was recorded (one submit) and return what was read, in order.
    fn finish(self: Box<Self>) -> Vec<Vec<f32>>;
}
