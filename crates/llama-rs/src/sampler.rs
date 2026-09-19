//! Logit-to-token samplers: greedy + temperature + top-k + top-p + min-p
//! + repetition penalty + mirostat v2.

use ggml_rs::Tensor;
use std::collections::VecDeque;

/// Mirostat v2 configuration. When attached to `SampleParams.mirostat`, it
/// replaces the top-k / top-p / min-p chain — mirostat does its own
/// distribution truncation based on a running estimate of token surprisal.
///
/// Algorithm (Basu et al., 2020 — v2 simplification): track `μ` = the maximum
/// allowed surprisal. After each sample, compute observed surprisal
/// `s = -log2(p(chosen))` and update `μ ← μ - η · (s - τ)`. Filter the next
/// step's candidates to those with `surprisal ≤ μ` (i.e. `p ≥ 2^-μ`). Tends
/// to produce more stable long-form text than fixed top-k/top-p, which can
/// drift into repetition or randomness depending on local distribution
/// sharpness.
#[derive(Debug, Clone, Copy)]
pub struct MirostatConfig {
    /// Target surprisal (bits per token). Lower = more focused / less random.
    /// Typical: 3.0–5.0. The original paper uses 5.0.
    pub tau: f32,
    /// Learning rate for `μ` updates. Lower = slower to adapt, higher = more
    /// reactive. Typical: 0.05–0.2.
    pub eta: f32,
}

impl Default for MirostatConfig {
    fn default() -> Self { Self { tau: 5.0, eta: 0.1 } }
}

#[derive(Debug, Clone)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k:       Option<usize>,
    pub top_p:       Option<f32>,
    /// Min-p (Minerva-style nucleus): keep tokens whose probability is at least
    /// `min_p × max_token_probability`. Adapts to distribution sharpness — for
    /// confident predictions it lets through fewer alternatives than top-p.
    /// Typical: 0.05–0.1. None disables.
    pub min_p:       Option<f32>,
    /// Repetition penalty: divide the logit of any token in `last_n` recent
    /// tokens by this value before softmax. Typical: 1.05–1.3. None or 1.0
    /// disables. Combine with `repeat_last_n` to bound the lookback window.
    pub repeat_penalty:   Option<f32>,
    pub repeat_last_n:    usize,
    /// Mirostat v2 adaptive sampler. When `Some`, the top-k / top-p / min-p
    /// chain is *bypassed* — mirostat replaces them. Temperature still applies
    /// (standard practice is `temperature = 1.0` with mirostat).
    pub mirostat: Option<MirostatConfig>,
    pub seed:        u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k:       Some(40),
            top_p:       Some(0.95),
            min_p:       None,
            repeat_penalty: None,
            repeat_last_n:  64,
            mirostat:    None,
            seed:        0xC0FFEE,
        }
    }
}

impl SampleParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: 0,
            mirostat: None,
            seed: 0,
        }
    }

    /// Convenience builder: standard mirostat v2 setup with `tau`, default `eta=0.1`,
    /// `temperature=1.0`, no other filters. Repetition penalty stays at the caller's
    /// existing setting.
    pub fn mirostat(tau: f32) -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: 0,
            mirostat: Some(MirostatConfig { tau, eta: 0.1 }),
            seed: 0xC0FFEE,
        }
    }
}

/// Tiny SplitMix64 PRNG. We don't pull in `rand` for this — it'd be the
/// crate's heaviest dep otherwise.
#[derive(Debug)]
pub struct Sampler {
    state: u64,
    params: SampleParams,
    /// Recently-sampled tokens for repetition-penalty lookback. Newest at the back.
    /// Bounded by `params.repeat_last_n`; older tokens fall off the front.
    history: VecDeque<u32>,
    /// Mirostat v2 running estimate of max surprisal. Initialised to `2τ`
    /// (the original paper's suggestion) on first mirostat-mode sample.
    /// Persists across samples so the filter adapts to the prior token's
    /// observed surprisal.
    mirostat_mu: Option<f32>,
}

impl Sampler {
    pub fn new(params: SampleParams) -> Self {
        let cap = params.repeat_last_n;
        let mirostat_mu = params.mirostat.map(|cfg| 2.0 * cfg.tau);
        Self {
            state: params.seed.wrapping_add(0x9E3779B97F4A7C15),
            params,
            history: VecDeque::with_capacity(cap),
            mirostat_mu,
        }
    }

    /// Add a token to the history without sampling. Use this to seed the sampler
    /// with the prompt tokens so repetition penalty considers them.
    pub fn observe(&mut self, token: u32) {
        if self.params.repeat_last_n == 0 { return; }
        if self.history.len() == self.params.repeat_last_n {
            self.history.pop_front();
        }
        self.history.push_back(token);
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        // 24-bit float in [0, 1).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Sample one token from a row of logits. `logits` shape: `[vocab_size]`.
    /// Side effect: records the chosen token in the rolling repetition-penalty history.
    pub fn sample(&mut self, logits: &Tensor) -> u32 {
        self.sample_biased(logits, &[])
    }

    /// Sample with additive logit biases. `bias` is a slice of `(token_id, delta)`
    /// pairs added to the logit before any other transformation (repetition
    /// penalty, temperature, top-k/p, min-p). Use a large negative delta (e.g.
    /// `f32::NEG_INFINITY`) to ban a token; a large positive delta to force it.
    /// Compatible with the OpenAI Chat Completions `logit_bias` parameter.
    /// Linear in `bias.len()`, so keep the slice small (typical usage: stop
    /// tokens, format constraints).
    pub fn sample_biased(&mut self, logits: &Tensor, bias: &[(u32, f32)]) -> u32 {
        debug_assert_eq!(logits.rank(), 1);
        let n = logits.numel();
        let l_src = logits.data();

        // Whether we need an owned copy of the logits to mutate. We need one
        // if either logit_bias or repetition penalty applies.
        let need_owned = !bias.is_empty()
            || matches!(self.params.repeat_penalty, Some(p) if p != 1.0 && !self.history.is_empty());

        let logits_owned: Option<Vec<f32>> = if need_owned {
            let mut v: Vec<f32> = l_src.to_vec();

            // Logit biases first (additive). NEG_INFINITY effectively bans.
            for &(tok, delta) in bias {
                let i = tok as usize;
                if i < v.len() { v[i] += delta; }
            }

            // Repetition penalty: positive logit → divide, negative → multiply
            // (matches llama.cpp semantics — true soft suppression regardless of sign).
            if let Some(p) = self.params.repeat_penalty {
                if p != 1.0 && !self.history.is_empty() {
                    for &tok in &self.history {
                        let i = tok as usize;
                        if i < v.len() {
                            v[i] = if v[i] > 0.0 { v[i] / p } else { v[i] * p };
                        }
                    }
                }
            }
            Some(v)
        } else {
            None
        };
        let l: &[f32] = logits_owned.as_deref().unwrap_or(l_src);

        // Greedy fast path.
        if self.params.temperature <= 0.0 {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in l.iter().enumerate() {
                if v > best_v { best_v = v; best_i = i; }
            }
            self.observe(best_i as u32);
            return best_i as u32;
        }

        // Apply temperature, exp, then optionally restrict to top-k, top-p, min-p.
        let inv_t = 1.0 / self.params.temperature;
        let mut scored: Vec<(usize, f32)> = (0..n).map(|i| (i, l[i] * inv_t)).collect();

        // Subtract max before exp for numerical stability.
        let mut m = f32::NEG_INFINITY;
        for &(_, v) in &scored { if v > m { m = v; } }
        for s in &mut scored { s.1 = (s.1 - m).exp(); }

        // Mirostat v2 short-circuits the top-k / top-p / min-p chain.
        if let Some(cfg) = self.params.mirostat {
            // Snapshot μ — avoid holding a mutable borrow on self while we
            // also call self.next_f32(). We write the new μ back at the end.
            let mut mu = self.mirostat_mu.unwrap_or(2.0 * cfg.tau);

            // Normalise to probabilities for surprisal computation.
            let total: f32 = scored.iter().map(|s| s.1).sum();
            if total > 0.0 {
                for s in &mut scored { s.1 /= total; }
            }

            // Filter to tokens with surprisal ≤ μ, i.e. p ≥ 2^-μ.
            let p_threshold = (-mu).exp2();
            let mut filtered: Vec<(usize, f32)> = scored.iter()
                .copied()
                .filter(|s| s.1 >= p_threshold)
                .collect();
            // Always keep the argmax — protects against pathological μ where
            // the threshold filters everything out.
            if filtered.is_empty() {
                let argmax = scored.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .copied().unwrap_or((0, 1.0));
                filtered.push(argmax);
            }

            // Sample from filtered distribution.
            let f_total: f32 = filtered.iter().map(|s| s.1).sum();
            let r = self.next_f32() * f_total;
            let mut acc = 0.0f32;
            let mut chosen = filtered.last().unwrap().0;
            let mut chosen_p = filtered.last().unwrap().1;
            for &(i, w) in &filtered {
                acc += w;
                if acc >= r { chosen = i; chosen_p = w; break; }
            }

            // Update μ from observed surprisal of the chosen token.
            // s = -log2(p). Then μ ← μ - η · (s - τ).
            let surprisal = -chosen_p.max(1e-30).log2();
            mu -= cfg.eta * (surprisal - cfg.tau);
            self.mirostat_mu = Some(mu);

            self.observe(chosen as u32);
            return chosen as u32;
        }

        // top-k cut (sort by descending prob).
        if let Some(k) = self.params.top_k {
            if k < scored.len() {
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(k);
            }
        }

        // top-p cut: sort, take prefix until cumulative prob >= p.
        if let Some(p) = self.params.top_p {
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let total: f32 = scored.iter().map(|s| s.1).sum();
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for s in &scored {
                cum += s.1 / total;
                keep += 1;
                if cum >= p { break; }
            }
            scored.truncate(keep);
        }

        // min-p cut: keep tokens with prob >= min_p × max_prob.
        if let Some(mp) = self.params.min_p {
            let max_p = scored.iter().map(|s| s.1).fold(0.0f32, f32::max);
            let cutoff = max_p * mp;
            scored.retain(|s| s.1 >= cutoff);
            // Pathological case: if min_p ≥ 1 and ties exist we may still have ≥1; if
            // somehow we ended up empty, fall back to argmax.
            if scored.is_empty() {
                let mut best_i = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in l.iter().enumerate() {
                    if v > best_v { best_v = v; best_i = i; }
                }
                self.observe(best_i as u32);
                return best_i as u32;
            }
        }

        // Sample.
        let total: f32 = scored.iter().map(|s| s.1).sum();
        let r = self.next_f32() * total;
        let mut acc = 0.0f32;
        let chosen = {
            let mut out = scored.last().unwrap().0;
            for &(i, w) in &scored {
                acc += w;
                if acc >= r { out = i; break; }
            }
            out
        };
        self.observe(chosen as u32);
        chosen as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(SampleParams::greedy());
        let logits = Tensor::from_vec(vec![0.1, 0.5, 0.2, 0.9, 0.0], vec![5]);
        assert_eq!(s.sample(&logits), 3);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut s = Sampler::new(SampleParams { temperature: 0.0, ..SampleParams::default() });
        let logits = Tensor::from_vec(vec![0.1, 0.5, 0.2, 0.9, 0.0], vec![5]);
        assert_eq!(s.sample(&logits), 3);
    }

    #[test]
    fn high_temperature_diversity() {
        // With high temperature, we should sometimes pick non-argmax tokens.
        let mut hits = [0u32; 5];
        let logits = Tensor::from_vec(vec![1.0, 1.1, 1.0, 1.0, 1.0], vec![5]);
        let mut s = Sampler::new(SampleParams {
            temperature: 5.0, top_k: None, top_p: None, min_p: None,
            repeat_penalty: None, repeat_last_n: 0, mirostat: None, seed: 42,
        });
        for _ in 0..1000 {
            hits[s.sample(&logits) as usize] += 1;
        }
        let nonzero = hits.iter().filter(|&&c| c > 0).count();
        assert!(nonzero >= 4, "expected diverse sampling, got {hits:?}");
    }

    /// Repetition penalty: greedy normally picks the highest logit; with penalty
    /// applied to the top token via history, the next-best should win instead.
    #[test]
    fn repetition_penalty_demotes_history_tokens() {
        let logits = Tensor::from_vec(vec![1.0, 5.0, 4.0, 0.5, 0.1], vec![5]);
        let mut s = Sampler::new(SampleParams {
            temperature: 0.0, top_k: None, top_p: None, min_p: None,
            repeat_penalty: Some(2.0), repeat_last_n: 4, mirostat: None, seed: 0,
        });
        // No history -> argmax = idx 1 (logit 5.0).
        assert_eq!(s.sample(&logits), 1);
        // Now token 1 is in history; with penalty=2.0, logit becomes 5.0/2 = 2.5,
        // and idx 2 (logit 4.0) becomes the new max.
        assert_eq!(s.sample(&logits), 2);
    }

    /// Min-p: with sharp distribution and min_p=0.5, only the top token should
    /// survive the filter (others are well below 0.5 × max_prob).
    #[test]
    fn min_p_filters_low_probability_tokens() {
        let mut hits = [0u32; 5];
        let logits = Tensor::from_vec(vec![10.0, 0.1, 0.1, 0.1, 0.1], vec![5]);
        let mut s = Sampler::new(SampleParams {
            temperature: 1.0, top_k: None, top_p: None, min_p: Some(0.5),
            repeat_penalty: None, repeat_last_n: 0, mirostat: None, seed: 7,
        });
        for _ in 0..200 {
            hits[s.sample(&logits) as usize] += 1;
        }
        // Only token 0 should be picked (its prob >> 0.5 × 1.0 = 0.5; others are tiny).
        assert_eq!(hits[0], 200, "min_p should keep only the dominant token: {hits:?}");
    }

    /// History should bound at `repeat_last_n` and drop oldest entries first.
    #[test]
    fn repetition_history_bounded_to_last_n() {
        let mut s = Sampler::new(SampleParams {
            temperature: 0.0, top_k: None, top_p: None, min_p: None,
            repeat_penalty: Some(1.5), repeat_last_n: 3, mirostat: None, seed: 0,
        });
        s.observe(10);
        s.observe(20);
        s.observe(30);
        s.observe(40);  // pushes 10 out
        assert_eq!(s.history.len(), 3);
        assert_eq!(s.history[0], 20);
        assert_eq!(s.history[2], 40);
    }

    /// Logit bias: large negative bias bans a token; large positive bias forces one.
    #[test]
    fn logit_bias_bans_and_forces_tokens() {
        let logits = Tensor::from_vec(vec![1.0, 5.0, 4.0, 0.5, 0.1], vec![5]);

        // Greedy: ban the natural argmax (idx 1) -> idx 2 wins.
        let mut s = Sampler::new(SampleParams::greedy());
        let banned = [(1u32, f32::NEG_INFINITY)];
        assert_eq!(s.sample_biased(&logits, &banned), 2);

        // Greedy: force a low-logit token by adding a huge bias.
        let mut s2 = Sampler::new(SampleParams::greedy());
        let forced = [(4u32, 100.0f32)];
        assert_eq!(s2.sample_biased(&logits, &forced), 4);
    }

    /// Bias survives all the post-bias filters (temperature, top-k, etc.). With a
    /// soft positive bias on a previously-mid-rank token, it should win argmax.
    #[test]
    fn logit_bias_modifies_ranking_under_temperature() {
        let logits = Tensor::from_vec(vec![1.0, 5.0, 4.0, 0.5, 0.1], vec![5]);
        let mut s = Sampler::new(SampleParams {
            temperature: 0.0,  // greedy after bias
            ..SampleParams::default()
        });
        // +2 to idx 2 makes it 6.0, beating idx 1 at 5.0.
        let bias = [(2u32, 2.0)];
        assert_eq!(s.sample_biased(&logits, &bias), 2);
    }

    /// Mirostat: with a sharp distribution and τ=1.0, the dominant token's
    /// surprisal (~0.01 bits) is well below μ=2τ=2.0, so it should be admitted
    /// every time. After many samples μ should drift down toward observed
    /// surprisal but never push the dominant token out of the filter.
    #[test]
    fn mirostat_picks_dominant_token_in_sharp_distribution() {
        let logits = Tensor::from_vec(vec![10.0, 0.1, 0.1, 0.1, 0.1], vec![5]);
        let mut s = Sampler::new(SampleParams::mirostat(5.0));
        let mut hits = [0u32; 5];
        for _ in 0..200 {
            hits[s.sample(&logits) as usize] += 1;
        }
        assert_eq!(hits[0], 200, "dominant token should always win: {hits:?}");
    }

    /// Mirostat: μ should *increase* (widen the allowed surprisal band) when
    /// observed surprisal stays well below τ — the filter relaxes because
    /// we're already sampling more confidently than the target.
    #[test]
    fn mirostat_mu_increases_when_surprisal_below_tau() {
        let logits = Tensor::from_vec(vec![10.0, 0.1, 0.1, 0.1, 0.1], vec![5]);
        // Sharp distribution → dominant token has surprisal ≈ 0, well below τ=5.
        let mut s = Sampler::new(SampleParams::mirostat(5.0));
        let mu_initial = s.mirostat_mu.unwrap();
        for _ in 0..50 {
            s.sample(&logits);
        }
        let mu_after = s.mirostat_mu.unwrap();
        assert!(mu_after > mu_initial,
            "μ should grow when observed surprisal < τ; before={mu_initial}, after={mu_after}");
    }

    /// Mirostat: μ should *decrease* (tighten the filter) when observed
    /// surprisal exceeds τ — pushing the sampler toward more confident tokens
    /// to bring surprisal back down.
    #[test]
    fn mirostat_mu_decreases_when_surprisal_above_tau() {
        // Uniform 8-way distribution → surprisal = log2(8) = 3 bits.
        let logits = Tensor::from_vec(vec![1.0; 8], vec![8]);
        // τ=1.0 < 3.0, so observed surprisal exceeds τ and μ should shrink.
        let mut s = Sampler::new(SampleParams::mirostat(1.0));
        let mu_initial = s.mirostat_mu.unwrap();
        for _ in 0..50 {
            s.sample(&logits);
        }
        let mu_after = s.mirostat_mu.unwrap();
        assert!(mu_after < mu_initial,
            "μ should shrink when observed surprisal > τ; before={mu_initial}, after={mu_after}");
    }
}
