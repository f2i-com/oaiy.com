//! Flash-Next's attention past the context the GPU keeps its cache for (`resident_rows`): the cache's rows are the
//! host's alone, in the computer's memory, and the GPU holds none of them.
//!
//! Its attention layers read little of a long cache: past the dense span each query attends to the 512 blocks of
//! four positions its indexer scores best, and its own last few. So a step's attention is the host's: the GPU's
//! run waits for its queries, the host scores the cache's blocks by their pooled keys (kept here, pooled and
//! rotated once each), chooses and attends as the device's QSA does, and the GPU goes on with the output. A
//! prompt's chunk, whose thousand rows choose most of the blocks between them, is the GPU's still: each attention
//! layer's rows are streamed up from the host's cache as the layer begins, into one layer's vectors that every
//! layer then uses in turn. The GPU holds one layer's rows for a prompt where it held every layer's for the whole
//! context, and its expert slots keep the room.

use super::*;
use rayon::prelude::*;

/// The positions the GPU holds Flash-Next's attention cache for (OAIY_FN_RESIDENT, 4,096 at least; 32,768 unless
/// said: what a 24 GB card holds beside its expert slots, the room its slots are sized around). A request that needs
/// more is run with the cache on the host ([`self`]).
pub(crate) fn resident_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| std::env::var("OAIY_FN_RESIDENT").ok().and_then(|v| v.parse::<usize>().ok()).map_or(32_768, |n| n.max(4096)))
}

/// A layer's pooled indexer keys on the host: the mean of each whole block of its raw keys, normed and rotated at the
/// block's start (`[valid, index_dim]`), as the device's QSA pools them at each step.
#[derive(Default)]
pub(crate) struct HostPooled {
    pub(super) valid: usize,
    pub(super) keys: Vec<f32>,
}

/// The device's vectors of an attention layer's run with the cache on the host.
pub(crate) struct Offload {
    /// A step's or a few rows' index queries (`[CHECK_ROWS, index_heads, index_dim]`, then normed and rotated) and
    /// their attention's output, which the host gives (`[CHECK_ROWS, heads, head_dim]`).
    pub(super) iqv: ggml_rs::DeviceVec,
    pub(super) iqn: ggml_rs::DeviceVec,
    pub(super) out: ggml_rs::DeviceVec,
    /// A prompt's: one attention layer's rows (each K then V) and raw indexer keys, `cap` of them, uploaded as each
    /// layer begins.
    pub(super) stream: Option<Stream>,
}

pub(crate) struct Stream {
    pub(super) layer: ggml_rs::DeviceVec,
    pub(super) raw: ggml_rs::DeviceVec,
    pub(super) cap: usize,
}

/// The host's own backend, for the few rows' norms and rotations it does as the reference path does.
fn cpu() -> &'static ggml_rs::CpuBackend {
    static CPU: std::sync::OnceLock<ggml_rs::CpuBackend> = std::sync::OnceLock::new();
    CPU.get_or_init(ggml_rs::CpuBackend::new)
}

/// `a . b`, eight lanes at a time (the lengths here are multiples of eight).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ac.remainder().iter().zip(bc.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ac.zip(bc) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// A tensor's values on the host (a copy only where it is a device's).
fn host_values(t: &Tensor) -> std::borrow::Cow<'_, [f32]> {
    if t.is_device() {
        std::borrow::Cow::Owned(t.to_host().data().to_vec())
    } else {
        std::borrow::Cow::Borrowed(t.data())
    }
}

/// The shapes [`qsa_rows`] reads.
pub(crate) struct QsaDims {
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) index_heads: usize,
    pub(crate) index_dim: usize,
    pub(crate) ratio: usize,
    pub(crate) keep: usize,
}

/// `t` rows at `past` attending over a cache's `kd` and `vd` (`[rows, kv_heads, head_dim]`, those to `past + t` there)
/// as the device's QSA attends: each row's `keep` whole blocks its indexer scores best (the larger first, and of equal
/// ones the lower), all of them where it sees no more, and its incomplete last block. `qn` `[t, heads, head_dim]`
/// and `iqn` `[t, index_heads, index_dim]` normed and rotated; `pooled` the blocks' keys. The output `[t, heads,
/// head_dim]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_rows(dims: &QsaDims, kd: &[f32], vd: &[f32], pooled: &HostPooled, qn: &[f32], iqn: &[f32], past: usize, t: usize) -> Vec<f32> {
    let QsaDims { heads: nh, kv_heads: nkv, head_dim: hd, index_heads: ih, index_dim: id, ratio, keep } = *dims;
    let (iscale, scale) = (1.0 / (id as f32).sqrt(), 1.0 / (hd as f32).sqrt());
    let group = nh / nkv;
    let mut out = vec![0f32; t * nh * hd];
    for r in 0..t {
        let p = past + r;
        let visible = ((p + 1) / ratio).min(pooled.valid);
        let mut blocks: Vec<u32> = (0..visible as u32).collect();
        if visible > keep {
            let q = &iqn[r * ih * id..(r + 1) * ih * id];
            let mut scores = vec![0f32; visible];
            scores.par_chunks_mut(512).enumerate().for_each(|(c, s)| {
                for (k, v) in s.iter_mut().enumerate() {
                    let key = &pooled.keys[(c * 512 + k) * id..(c * 512 + k + 1) * id];
                    let total: f32 = (0..ih).map(|h| dot(&q[h * id..(h + 1) * id], key).max(0.0)).sum();
                    *v = total * iscale;
                }
            });
            blocks.select_nth_unstable_by(keep - 1, |&x, &y| scores[y as usize].total_cmp(&scores[x as usize]).then(x.cmp(&y)));
            blocks.truncate(keep);
            blocks.sort_unstable();
        }
        let at: Vec<usize> = blocks.iter().flat_map(|&b| b as usize * ratio..(b as usize + 1) * ratio).chain(visible * ratio..=p).collect();
        out[r * nh * hd..(r + 1) * nh * hd].par_chunks_mut(hd).enumerate().for_each(|(h, o)| {
            let g = h / group;
            let q = &qn[(r * nh + h) * hd..(r * nh + h + 1) * hd];
            let mut w: Vec<f32> = at.iter().map(|&x| dot(q, &kd[(x * nkv + g) * hd..(x * nkv + g + 1) * hd]) * scale).collect();
            let most = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in w.iter_mut() {
                *v = (*v - most).exp();
                sum += *v;
            }
            for (&e, &x) in w.iter().zip(&at) {
                for (y, v) in o.iter_mut().zip(&vd[(x * nkv + g) * hd..(x * nkv + g + 1) * hd]) {
                    *y += e * v;
                }
            }
            for y in o.iter_mut() {
                *y /= sum;
            }
        });
    }
    out
}

impl FlashNext {
    /// The host's pooled keys of attention layer `a` up to `blocks` whole blocks of its cache (those it has are its
    /// cache's still: the caller says so of them, as of a device's copy).
    pub(super) fn pool_host(&self, a: &Attn, kv: &KvCache, pooled: &mut HostPooled, blocks: usize) {
        let cfg = &self.config;
        let (id, ratio) = (cfg.index_dim, cfg.index_ratio);
        if pooled.valid >= blocks {
            return;
        }
        let raw = host_values(kv.k_buffer(a.index_slot));
        let from = pooled.valid;
        let n = blocks - from;
        let mut means = vec![0f32; n * id];
        means.par_chunks_mut(id).enumerate().for_each(|(k, m)| {
            let j = from + k;
            for c in 0..ratio {
                for (x, r) in m.iter_mut().zip(&raw[(j * ratio + c) * id..(j * ratio + c + 1) * id]) {
                    *x += r / ratio as f32;
                }
            }
        });
        let b = cpu();
        let w = a.index_k_norm.to_host();
        let mut y = ggml_rs::ops::rmsnorm(b, &Tensor::from_vec(means, vec![n, 1, id]), &w, cfg.eps);
        let starts: Vec<u32> = (from..blocks).map(|j| (j * ratio) as u32).collect();
        b.rope_partial_neox(&mut y, &starts, id, cfg.rope_dim, cfg.rope_theta);
        pooled.keys.truncate(from * id);
        pooled.keys.extend_from_slice(y.to_host().data());
        pooled.valid = blocks;
    }

    /// The attention of `t` rows at `past` over attention layer `layer`'s cache on the host (its rows there up to
    /// `past + t`): [`qsa_rows`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn host_attention(&self, layer: usize, kv: &KvCache, pooled: &HostPooled, qn: &[f32], iqn: &[f32], past: usize, t: usize) -> Vec<f32> {
        let cfg = &self.config;
        let (kd, vd) = (host_values(kv.k_buffer(layer)), host_values(kv.v_buffer(layer)));
        let dims = QsaDims { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim, index_heads: cfg.index_heads, index_dim: cfg.index_dim, ratio: cfg.index_ratio, keep: cfg.index_budget / cfg.index_ratio };
        qsa_rows(&dims, &kd, &vd, pooled, qn, iqn, past, t)
    }

    /// Attention layer `layer`'s first `rows` rows of the host's cache up to the device's stream (each row's K then
    /// V, and its raw indexer keys), through `staging` (kept by the caller from layer to layer).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stream_rows(&self, c: &dyn ggml_rs::DeviceChain, kv: &KvCache, layer: usize, index_slot: usize, rows: usize, s: &Stream, staging: &mut Vec<f32>) {
        if rows == 0 {
            return;
        }
        let cfg = &self.config;
        let inner = cfg.kv_heads * cfg.head_dim;
        let (kd, vd) = (host_values(kv.k_buffer(layer)), host_values(kv.v_buffer(layer)));
        staging.resize(rows * 2 * inner, 0.0);
        staging.par_chunks_mut(2 * inner).enumerate().for_each(|(p, r)| {
            r[..inner].copy_from_slice(&kd[p * inner..(p + 1) * inner]);
            r[inner..].copy_from_slice(&vd[p * inner..(p + 1) * inner]);
        });
        c.upload_at(&s.layer, 0, staging);
        let raw = host_values(kv.k_buffer(index_slot));
        c.upload_at(&s.raw, 0, &raw[..rows * cfg.index_dim]);
    }
}

#[cfg(test)]
mod tests {
    use super::{dot, qsa_rows, HostPooled, QsaDims};
    use ggml_rs::{Backend, Tensor};

    /// The host's sparse attention is the reference's (the CPU backend's pool, scores, selection and attention, op by
    /// op, which the device's kernels are held to): rows that see no more blocks than are kept and rows that choose,
    /// over a cache past the dense span, a step's row and a few rows', and the pooled keys as the device pools them.
    #[test]
    fn the_hosts_sparse_attention_is_the_references() {
        let (nh, nkv, hd, ih, id, ratio, keep) = (8usize, 2usize, 32usize, 2usize, 16usize, 4usize, 6usize);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let cpu = ggml_rs::CpuBackend::new();
        let dims = QsaDims { heads: nh, kv_heads: nkv, head_dim: hd, index_heads: ih, index_dim: id, ratio, keep };
        for (past, t) in [(9usize, 3usize), (61, 1), (97, 5), (200, 8)] {
            let total = past + t;
            let k: Vec<f32> = (0..total * nkv * hd).map(|_| next()).collect();
            let v: Vec<f32> = (0..total * nkv * hd).map(|_| next()).collect();
            let raw: Vec<f32> = (0..total * id).map(|_| next()).collect();
            let qn: Vec<f32> = (0..t * nh * hd).map(|_| next()).collect();
            let iqn: Vec<f32> = (0..t * ih * id).map(|_| next()).collect();
            let nb = total / ratio;
            // the reference: pool (no norm or rotation here: both paths are given the same keys), score, choose, attend
            let pooled = cpu.qsa_pool(&Tensor::from_vec(raw.clone(), vec![total, 1, id]), nb, ratio);
            let scores = cpu.qsa_block_scores(&Tensor::from_vec(iqn.clone(), vec![t, ih, id]), &pooled, past, ratio, 1.0 / (id as f32).sqrt());
            let sel = cpu.qsa_select(&scores, past, ratio, keep);
            let want = cpu.sparse_attention(&Tensor::from_vec(qn.clone(), vec![t, nh, hd]), &Tensor::from_vec(k.clone(), vec![total, nkv, hd]), &Tensor::from_vec(v.clone(), vec![total, nkv, hd]), &sel, 1.0 / (hd as f32).sqrt());
            let host = HostPooled { valid: nb, keys: pooled.data().to_vec() };
            let got = qsa_rows(&dims, &k, &v, &host, &qn, &iqn, past, t);
            for (i, (g, w)) in got.iter().zip(want.data()).enumerate() {
                assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "past {past}, {t} rows: value {i}: {g} against {w}");
            }
        }
    }

    #[test]
    fn a_dot_product_eight_lanes_at_a_time_is_the_plain_one() {
        let a: Vec<f32> = (0..259).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..259).map(|i| (i as f32 * 0.11).cos()).collect();
        let plain: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - plain).abs() < 1e-4, "{} against {plain}", dot(&a, &b));
    }
}
