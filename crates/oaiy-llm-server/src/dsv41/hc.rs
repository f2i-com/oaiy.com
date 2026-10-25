//! Hyper-connections: the residual stream is `hc_mult` (4) parallel copies
//! of the model width. Each sublayer derives three coefficient sets from the
//! stream itself — `pre` (collapse the copies into one input), `post`
//! (spread the output back) and `comb` (mix the residual copies, made
//! doubly stochastic by Sinkhorn iterations) — reference `Block.hc_mixes`,
//! `hc_split_sinkhorn`, `hc_pre`, `hc_post`.

use oaiy_engine::Result;

use crate::dsv41::formats::to_bf16;
use crate::dsv41::linear::Weight;
use crate::dsv41::ops::sigmoid;
use crate::dsv41::safetensors::StIndex;

pub const HC: usize = 4;
/// Mix coefficients per token and sublayer: pre, post, comb.
pub const MIX: usize = (2 + HC) * HC; // 24

/// One sublayer's mix parameters (`hc_{attn,ffn}_{fn,base,scale}`).
pub struct HcParams {
    /// The `[24, 4 * dim]` projection kept the other way about, `[4 * dim, 24]`: element `k`'s 24 weights side by
    /// side, as [`mixes`] takes them.
    by_k: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    /// The projection as stored (`[24, 4 * dim]`), for a device to hold ([`crate::dsv41::model::Model::offload`]): a
    /// decode step's 24 sums are then made there, beside a call the sublayer makes anyway ([`Self::on_device`]).
    weight: Weight,
}

impl HcParams {
    pub fn load(idx: &StIndex, prefix: &str, which: &str) -> Result<HcParams> {
        Ok(HcParams::new(
            idx.read_f32(&format!("{prefix}.hc_{which}_fn"))?,
            idx.read_f32(&format!("{prefix}.hc_{which}_base"))?,
            idx.read_f32(&format!("{prefix}.hc_{which}_scale"))?,
        ))
    }

    /// From raw parts: `fn_` `[24, 4 * dim]`, `base` `[24]`, `scale` `[3]`.
    pub fn new(fn_: Vec<f32>, base: Vec<f32>, scale: Vec<f32>) -> HcParams {
        let n = fn_.len() / MIX;
        let mut by_k = vec![0.0f32; fn_.len()];
        for (j, row) in fn_.chunks_exact(n.max(1)).enumerate() {
            for (k, &w) in row.iter().enumerate() {
                by_k[k * MIX + j] = w;
            }
        }
        HcParams { by_k, base, scale, weight: Weight::F32 { w: fn_, n: MIX, k: n } }
    }

    /// The projection as a weight a device may take ([`crate::dsv41::model::Model::offload`]).
    pub(crate) fn weight_mut(&mut self) -> &mut Weight {
        &mut self.weight
    }

    /// The projection where a device holds it: a decode step's 24 sums made there ([`mixes_of`]), beside the
    /// sublayer's own call. None where it does not (they are made here, [`mixes`]).
    pub fn on_device(&self) -> Option<&Weight> {
        matches!(self.weight, Weight::Device { .. }).then_some(&self.weight)
    }

    /// `(base, scale)`, as a device's [`mixes_of`] takes them.
    pub(crate) fn base_scale(&self) -> (&[f32], &[f32]) {
        (&self.base, &self.scale)
    }
}

/// [`mixes`] of one token's stream `x` (`[HC * dim]`) from its 24 projections made elsewhere (`proj`: a device's,
/// [`HcParams::on_device`]), its sum of squares here: the device sums in an order of its own, so the coefficients are
/// within roundings of the host's.
pub fn mixes_of(proj: &[f32], x: &[f32], p: &HcParams, norm_eps: f32, iters: usize, hc_eps: f32) -> Mix {
    let proj: [f32; MIX] = proj[..MIX].try_into().expect("24 projections");
    mixes_from_projection(&proj, sumsq(x), x.len(), p, norm_eps, iters, hc_eps)
}

/// The sum of squares of a token's stream [`mixes_of`] scales by, in eight running sums (the one chain of 20,480
/// additions was a quarter of a step's mixing): a device's mixing takes it from here.
pub fn sumsq(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for c in x.chunks_exact(8) {
        for (s, v) in lanes.iter_mut().zip(c) {
            *s += v * v;
        }
    }
    let tail: f32 = x.chunks_exact(8).remainder().iter().map(|v| v * v).sum();
    lanes.iter().sum::<f32>() + tail
}

/// Per-token coefficients.
#[derive(Clone, Copy, Debug)]
pub struct Mix {
    pub pre: [f32; HC],
    pub post: [f32; HC],
    /// `comb[i][j]`: weight of residual copy `i` in output copy `j`.
    pub comb: [[f32; HC]; HC],
}

impl Mix {
    /// The 24 laid end to end as a device program takes them: pre, post, then comb by rows (`comb[i][j]` at `8 + 4 i +
    /// j`).
    pub fn flat(&self) -> [f32; MIX] {
        let mut out = [0.0f32; MIX];
        out[..HC].copy_from_slice(&self.pre);
        out[HC..2 * HC].copy_from_slice(&self.post);
        for (i, row) in self.comb.iter().enumerate() {
            out[2 * HC + i * HC..2 * HC + (i + 1) * HC].copy_from_slice(row);
        }
        out
    }
}

/// `hc_mixes` for one token: `x` is its `[HC * dim]` stream (bf16 values).
pub fn mixes(x: &[f32], p: &HcParams, norm_eps: f32, iters: usize, hc_eps: f32) -> Mix {
    let n = x.len();
    assert_eq!(p.by_k.len(), n * MIX, "hc: the stream is not the projection's width");
    // The 24 projections' sums side by side, each in its own order (k ascending, as one projection at a time sums
    // it): the same sums to the bit, and 24 lanes of one loop where each was a chain of 20,480 additions, each waiting
    // for the one before (a decode step's 80 of these were 16 ms of it).
    let mut proj = [0.0f32; MIX];
    for (weights, &v) in p.by_k.chunks_exact(MIX).zip(x) {
        for (acc, w) in proj.iter_mut().zip(weights) {
            *acc += w * v;
        }
    }
    let sumsq = x.iter().map(|v| v * v).sum::<f32>();
    mixes_from_projection(&proj, sumsq, n, p, norm_eps, iters, hc_eps)
}

/// The cheap tail of [`mixes`], given the 24 raw projections `fn . x` and
/// `sum(x^2)` over the `n`-wide stream (a device computes those).
pub fn mixes_from_projection(proj: &[f32; MIX], sumsq: f32, n: usize, p: &HcParams, norm_eps: f32, iters: usize, hc_eps: f32) -> Mix {
    let r = 1.0 / (sumsq / n as f32 + norm_eps).sqrt();
    let m: [f32; MIX] = std::array::from_fn(|j| proj[j] * r);
    let mut mix = Mix { pre: [0.0; HC], post: [0.0; HC], comb: [[0.0; HC]; HC] };
    for j in 0..HC {
        mix.pre[j] = sigmoid(m[j] * p.scale[0] + p.base[j]) + hc_eps;
        mix.post[j] = 2.0 * sigmoid(m[j + HC] * p.scale[1] + p.base[j + HC]);
    }
    let c = &mut mix.comb;
    for (j, row) in c.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = m[2 * HC + j * HC + k] * p.scale[2] + p.base[2 * HC + j * HC + k];
        }
        // softmax over k, then + eps
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
        }
        let s: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v = *v / s + hc_eps;
        }
    }
    normalize_cols(c, hc_eps);
    for _ in 1..iters {
        for row in c.iter_mut() {
            let s: f32 = row.iter().sum::<f32>() + hc_eps;
            for v in row.iter_mut() {
                *v /= s;
            }
        }
        normalize_cols(c, hc_eps);
    }
    mix
}

fn normalize_cols(c: &mut [[f32; HC]; HC], eps: f32) {
    for k in 0..HC {
        let s: f32 = (0..HC).map(|j| c[j][k]).sum::<f32>() + eps;
        for row in c.iter_mut() {
            row[k] /= s;
        }
    }
}

/// `hc_pre`: collapse the copies, `bf16(sum_i pre[i] * x[i])`. `x` is `[HC, dim]`.
pub fn pre(x: &[f32], pre: &[f32; HC]) -> Vec<f32> {
    let d = x.len() / HC;
    (0..d)
        .map(|k| to_bf16((0..HC).map(|i| pre[i] * x[i * d + k]).sum::<f32>()))
        .collect()
}

/// `hc_post`: `bf16(post[j] * out + sum_i comb[i][j] * residual[i])` for each copy `j`.
pub fn post(out: &[f32], residual: &[f32], mix: &Mix) -> Vec<f32> {
    let d = out.len();
    let mut y = vec![0.0f32; HC * d];
    for j in 0..HC {
        for k in 0..d {
            let r: f32 = (0..HC).map(|i| mix.comb[i][j] * residual[i * d + k]).sum();
            y[j * d + k] = to_bf16(mix.post[j] * out[k] + r);
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_comb_is_doubly_stochastic() {
        let n = HC * 8;
        let p = HcParams::new((0..MIX * n).map(|i| ((i * 7919 % 101) as f32 - 50.0) / 50.0).collect(), (0..MIX).map(|i| i as f32 * 0.1 - 1.0).collect(), vec![0.5, 0.7, 1.3]);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) / 10.0).collect();
        let m = mixes(&x, &p, 1e-20, 20, 1e-6);
        for j in 0..HC {
            let row: f32 = m.comb[j].iter().sum();
            let col: f32 = (0..HC).map(|i| m.comb[i][j]).sum();
            assert!((row - 1.0).abs() < 1e-3 && (col - 1.0).abs() < 1e-3, "row {row} col {col}");
            assert!(m.pre[j] > 0.0 && m.post[j] > 0.0 && m.post[j] < 2.0);
        }
    }

    /// The mixes are the ones the 24 projections summed one at a time give, to the bit: at the model's width, on
    /// values of mixed signs and sizes (sums that would differ in another order).
    #[test]
    fn the_projections_side_by_side_are_the_ones_summed_one_at_a_time() {
        let n = HC * 5120;
        let mut s = 0x2545_f491u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 20_001) as f32 / 10_000.0 - 1.0
        };
        let fn_: Vec<f32> = (0..MIX * n).map(|_| next() * 0.05).collect();
        let x: Vec<f32> = (0..n).map(|i| next() * if i % 97 == 0 { 300.0 } else { 3.0 }).collect();
        let (base, scale): (Vec<f32>, Vec<f32>) = ((0..MIX).map(|i| i as f32 * 0.1 - 1.0).collect(), vec![0.5, 0.7, 1.3]);
        let mut proj = [0.0f32; MIX];
        for (j, mj) in proj.iter_mut().enumerate() {
            for (w, v) in fn_[j * n..(j + 1) * n].iter().zip(&x) {
                *mj += w * v;
            }
        }
        let p = HcParams::new(fn_, base, scale);
        let mut sumsq = 0.0f32;
        for v in &x {
            sumsq += v * v;
        }
        let want = mixes_from_projection(&proj, sumsq, n, &p, 1e-6, 20, 1e-6);
        let got = mixes(&x, &p, 1e-6, 20, 1e-6);
        let bits = |m: &Mix| -> Vec<u32> { m.pre.iter().chain(&m.post).chain(m.comb.iter().flatten()).map(|v| v.to_bits()).collect() };
        assert_eq!(bits(&got), bits(&want));
    }

    /// What one token's mixing costs on one thread, at DeepSeek-V4.1's width (HC streams of 5,120): a prompt's tokens
    /// take it twice a layer, 40 layers.
    #[test]
    #[ignore = "a timing; run with --nocapture"]
    fn measure_a_tokens_mixing() {
        let n = HC * 5120;
        let p = HcParams::new((0..MIX * n).map(|i| ((i * 7919 % 101) as f32 - 50.0) / 5000.0).collect(), (0..MIX).map(|i| i as f32 * 0.1 - 1.0).collect(), vec![0.5, 0.7, 1.3]);
        let x: Vec<f32> = (0..n).map(|i| ((i % 33) as f32 - 16.0) / 10.0).collect();
        let tokens = 2000;
        let t = std::time::Instant::now();
        let mut keep = 0.0;
        for _ in 0..tokens {
            keep += mixes(std::hint::black_box(&x), &p, 1e-6, 20, 1e-6).pre[0];
        }
        let secs = t.elapsed().as_secs_f64();
        eprintln!("{tokens} tokens' mixing in {secs:.2} s on one thread: a 2,000-token prompt's 80 would take {:.0} s ({keep})", secs * 80.0);
    }
}
