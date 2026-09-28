//! Drawing codec ids from logits: temperature, top-k, top-p, as the
//! reference's samplers do, with a seeded generator. On the host
//! ([`sample`]), or on the device ([`pick`]), where the draw needs no trip to
//! the host: a frame's 16 draws then cost one synchronization instead of 16.
use candle_core::{Result, Tensor};

/// A seeded generator (splitmix64) for sampling.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Temperature, top-k, top-p, then a draw (or the argmax).
pub fn sample(logits: &[f32], temperature: f64, top_k: usize, top_p: f64, greedy: bool, rng: &mut Rng) -> u32 {
    let mut order: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    if order.is_empty() {
        return 0;
    }
    // Only the top k need ordering; ties go to the lower id, as a stable sort
    // (and the reference's argmax) would have it.
    let k = if greedy { 1 } else { top_k.max(1).min(order.len()) };
    let by_logit = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, by_logit);
        order.truncate(k);
    }
    order.sort_by(by_logit);
    if greedy || order.len() == 1 {
        return order[0] as u32;
    }
    let top = logits[order[0]] as f64;
    let weights: Vec<f64> = order.iter().map(|&i| ((logits[i] as f64 - top) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut kept = order.len();
    if top_p < 1.0 {
        let mut acc = 0.;
        for (n, w) in weights.iter().enumerate() {
            acc += w / total;
            if acc >= top_p {
                kept = n + 1;
                break;
            }
        }
    }
    let total: f64 = weights[..kept].iter().sum();
    let mut target = rng.uniform() * total;
    for (n, w) in weights[..kept].iter().enumerate() {
        target -= w;
        if target <= 0. {
            return order[n] as u32;
        }
    }
    order[kept - 1] as u32
}

/// Gumbel noise, `-ln(-ln(u))` for uniform `u`: adding it to tempered
/// logits and taking the argmax draws from their softmax.
pub fn gumbel(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| -(-(rng.uniform().clamp(1e-12, 1. - 1e-12)).ln()).ln() as f32).collect()
}

/// Top-k sampling on the device: the k largest of `logits` (1, V) F32,
/// tempered, plus `noise` (1, k) Gumbel noise, argmax. The drawn id as a (1,)
/// U32 tensor, still on the device. Greedy (or k = 1) is the plain argmax.
pub fn pick(logits: &Tensor, top_k: usize, temperature: f64, noise: &Tensor, greedy: bool) -> Result<Tensor> {
    let k = top_k.clamp(1, logits.dim(1)?);
    if greedy || k == 1 {
        return logits.argmax(1);
    }
    let top = logits.arg_sort_last_dim(false)?.narrow(1, 0, k)?.contiguous()?;
    let score = (logits.gather(&top, 1)?.affine(1. / temperature, 0.)? + noise.narrow(1, 0, k)?)?;
    top.gather(&score.argmax_keepdim(1)?, 1)?.flatten_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn device_draws_follow_the_tempered_top_k() -> Result<()> {
        // ln 3, 0, and a third id cut by top-2: drawn 3:1 between the first two.
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[0.0f32, (3f32).ln(), -1.0]], &dev)?;
        let mut rng = Rng::new(5);
        let n = 20_000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            let noise = Tensor::from_vec(gumbel(&mut rng, 2), (1, 2), &dev)?;
            counts[pick(&logits, 2, 1.0, &noise, false)?.to_vec1::<u32>()?[0] as usize] += 1;
        }
        assert_eq!(counts[2], 0, "outside the top 2");
        let p = counts[1] as f64 / n as f64;
        assert!((p - 0.75).abs() < 0.015, "{p}");
        // Temperature 0.5 squares the odds: 9:1.
        let mut first = 0;
        for _ in 0..n {
            let noise = Tensor::from_vec(gumbel(&mut rng, 2), (1, 2), &dev)?;
            first += (pick(&logits, 2, 0.5, &noise, false)?.to_vec1::<u32>()?[0] == 1) as usize;
        }
        let p = first as f64 / n as f64;
        assert!((p - 0.9).abs() < 0.012, "{p}");
        let noise = Tensor::zeros((1, 2), candle_core::DType::F32, &dev)?;
        assert_eq!(pick(&logits, 50, 0.9, &noise, true)?.to_vec1::<u32>()?, vec![1]);
        Ok(())
    }

    #[test]
    fn sampling_respects_top_k_and_greedy() {
        let logits = [0.0f32, 5.0, 4.9, f32::NEG_INFINITY, -3.0];
        let mut rng = Rng::new(7);
        assert_eq!(sample(&logits, 0.9, 50, 1.0, true, &mut rng), 1);
        for _ in 0..200 {
            let s = sample(&logits, 1.0, 2, 1.0, false, &mut rng);
            assert!(s == 1 || s == 2, "top-2 only, got {s}");
        }
        // Everything masked but one.
        assert_eq!(sample(&[f32::NEG_INFINITY, 2.0], 1.0, 50, 1.0, false, &mut rng), 1);
    }

    #[test]
    fn draws_follow_the_tempered_distribution() {
        // Two ids with logits ln 3 and 0: at temperature 1 the first is drawn
        // three times as often.
        let logits = [(3f32).ln(), 0.0];
        let mut rng = Rng::new(11);
        let n = 20_000;
        let first = (0..n).filter(|_| sample(&logits, 1.0, 50, 1.0, false, &mut rng) == 0).count();
        let p = first as f64 / n as f64;
        assert!((p - 0.75).abs() < 0.015, "{p}");
    }
}
