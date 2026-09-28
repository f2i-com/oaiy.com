// Ported from the user-owned F2I plugin-diffusion SDXL implementation.
//! SDXL scheduler — sigma schedules + multiple sampler kernels.
//!
//! SDXL was trained with a scaled-linear beta schedule over 1000
//! timesteps. For inference we build a sigma schedule (Karras / Normal /
//! Exponential), then integrate via one of the discrete samplers
//! (Euler, Euler a, DPM++ 2M, DPM++ SDE). The samplers all take the
//! standard SDXL eps-prediction as input and apply Karras-style input
//! pre-conditioning (`x_scaled = x / sqrt(sigma^2 + 1)`).
//!
//! ## Schedules
//!
//! **Karras** (rho=7) — the diffusers default. Front-loads steps where
//! sigma changes most.
//!
//! ```text
//!   sigmas[i] = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
//! ```
//!
//! **Normal** — uniform spacing in DDPM-timestep space. Maps i = 999 →
//! 0 over the N steps and looks up `sigma_train[round(i)]` from the
//! scaled-linear beta cumulative-product table.
//!
//! **Exponential** — uniform spacing in log-sigma space.
//!
//! ```text
//!   sigmas[i] = exp(log(sigma_max) + i/(N-1) * (log(sigma_min) - log(sigma_max)))
//! ```
//!
//! ## Samplers
//!
//! **Euler** — `x_new = x + eps * (sigma_next - sigma_curr)`. Simplest;
//! one model forward per step. Deterministic.
//!
//! **Euler Ancestral** — drift to a `sigma_down` (less than sigma_next)
//! then re-add noise scaled by `sigma_up`. Same one-forward cost. The
//! noise injection means the trajectory depends on the seed beyond just
//! the initial latent.
//!
//! **DPM++ 2M** — second-order multistep. Reuses last step's `denoised`
//! to form a higher-order extrapolation. One forward per step but needs
//! the previous step's state. Deterministic. Faster convergence than
//! Euler at low step counts.
//!
//! **DPM++ SDE** — single-step second-order with stochastic re-noising.
//! TWO forwards per step (the predictor pass evaluates at an
//! intermediate sigma). Best quality for hard prompts at moderate
//! steps; ~2× slower per step than the others.
//!
//! ## Classifier-free guidance
//!
//! Applied by the pipeline outside the sampler:
//! ```text
//!   eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
//! ```

use candle_core::{Device, Result, Tensor};

/// Build the Karras sigma schedule for `num_steps` denoising steps.
/// Output has `num_steps + 1` entries: the integration walks from
/// `sigmas[0]` (max noise) to `sigmas[N] = 0` (clean).
pub fn karras_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(num_steps + 1);
    let inv_rho_min = sigma_min.powf(1.0 / rho);
    let inv_rho_max = sigma_max.powf(1.0 / rho);
    for i in 0..num_steps {
        let t = i as f64 / (num_steps - 1).max(1) as f64;
        let s = inv_rho_max + t * (inv_rho_min - inv_rho_max);
        out.push(s.powf(rho));
    }
    out.push(0.0); // terminal
    out
}

/// SDXL Karras defaults — taken from the diffusers SDXL pipeline.
pub fn sdxl_default_sigmas(num_steps: usize) -> Vec<f64> {
    karras_sigmas(num_steps, 0.0292, 14.6146, 7.0)
}

/// Translate sigma to a fractional DDPM training timestep using log-sigma
/// interpolation, matching the continuous k-diffusion denoiser convention.
///
/// Uses the diffusers convention: at training, `sigma(t) = sqrt((1 -
/// alpha_cumprod(t)) / alpha_cumprod(t))`. Inverting that closed-form
/// gives the timestep. We search the scaled-linear training sigma table.
pub fn sigma_to_timestep(sigma: f64, num_train_steps: usize) -> f64 {
    let (beta_start, beta_end) = (0.00085f64, 0.012f64);
    // SDXL uses `scaled_linear` betas: linear in sqrt(beta).
    let sqrt_betas: Vec<f64> = (0..num_train_steps)
        .map(|i| {
            let t = i as f64 / (num_train_steps - 1).max(1) as f64;
            let s = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
            s * s
        })
        .collect();
    let mut alpha_cumprod = 1.0;
    let mut sigmas_train = Vec::with_capacity(num_train_steps);
    for &b in &sqrt_betas {
        let alpha = 1.0 - b;
        alpha_cumprod *= alpha;
        sigmas_train.push(((1.0 - alpha_cumprod) / alpha_cumprod).sqrt());
    }
    // Find the bracketing training sigmas, then interpolate in log space.
    if sigma <= sigmas_train[0] {
        return 0.0;
    }
    if sigma >= sigmas_train[num_train_steps - 1] {
        return (num_train_steps - 1) as f64;
    }
    for i in 0..num_train_steps - 1 {
        if sigma >= sigmas_train[i] && sigma <= sigmas_train[i + 1] {
            let frac = (sigma.ln() - sigmas_train[i].ln())
                / (sigmas_train[i + 1].ln() - sigmas_train[i].ln());
            return i as f64 + frac;
        }
    }
    (num_train_steps - 1) as f64
}

/// Pre-conditioning factor: Euler-discrete uses sigma-scaled input,
/// `x_scaled = x / sqrt(sigma^2 + 1)`. The UNet takes the scaled tensor.
pub fn scale_input_for_euler(x: &Tensor, sigma: f64) -> Result<Tensor> {
    let denom = (sigma * sigma + 1.0).sqrt();
    x.affine(1.0 / denom, 0.0)
}

/// One Euler-discrete step: `x_new = x + eps * (sigma_next - sigma_curr)`.
///
/// The UNet was trained to predict the noise (eps-prediction), so the
/// step is a simple forward Euler in sigma. No corrector pass.
pub fn euler_step(x: &Tensor, eps: &Tensor, sigma_curr: f64, sigma_next: f64) -> Result<Tensor> {
    let dt = sigma_next - sigma_curr;
    let step = eps.affine(dt, 0.0)?;
    x + &step
}

/// Build the random noise tensor `(B, 4, H/8, W/8)` scaled to `sigmas[0]`
/// — the standard SDXL initial state for text-to-image.
pub fn build_initial_noise(
    batch: usize,
    height_latent: usize,
    width_latent: usize,
    sigma_init: f64,
    device: &Device,
) -> Result<Tensor> {
    let noise = Tensor::randn(0f32, 1f32, (batch, 4, height_latent, width_latent), device)?;
    noise.affine(sigma_init, 0.0)
}

// ---------------------------------------------------------------------------
// Schedule + Sampler enums (selectable from the diffusion_image node)
// ---------------------------------------------------------------------------

/// Sigma-schedule selector. Each variant returns a sigma list of length
/// `num_steps + 1` (terminal 0 appended).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    Karras,
    Normal,
    Exponential,
}

impl Schedule {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "normal" | "linear" | "ddpm" => Schedule::Normal,
            "exponential" | "exp" => Schedule::Exponential,
            _ => Schedule::Karras,
        }
    }
}

/// Sigma min/max bounds used by Karras + Exponential. The "Normal"
/// schedule ignores these and walks the DDPM training timesteps.
pub const SDXL_SIGMA_MIN: f64 = 0.0292;
pub const SDXL_SIGMA_MAX: f64 = 14.6146;

/// Dispatch on the schedule variant. Same signature as `karras_sigmas`
/// but selectable at runtime.
pub fn build_sigmas(schedule: Schedule, num_steps: usize) -> Vec<f64> {
    match schedule {
        Schedule::Karras => karras_sigmas(num_steps, SDXL_SIGMA_MIN, SDXL_SIGMA_MAX, 7.0),
        Schedule::Normal => normal_sigmas_sdxl(num_steps),
        Schedule::Exponential => exponential_sigmas(num_steps, SDXL_SIGMA_MIN, SDXL_SIGMA_MAX),
    }
}

/// Uniform spacing in DDPM-timestep space. Walks i = 999 → 0 over N
/// steps and looks up the SDXL `scaled_linear` beta cumulative-product
/// sigma table at each rounded position. Matches diffusers'
/// `EulerDiscreteScheduler::set_timesteps` with `use_karras_sigmas=False,
/// timestep_spacing="linspace"`.
pub fn normal_sigmas_sdxl(num_steps: usize) -> Vec<f64> {
    let train_sigmas = sdxl_training_sigmas(1000);
    let n = num_steps.max(1);
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        // linspace from t=999 (max noise) down to t=0 (clean) inclusive
        let t = if n == 1 {
            999.0
        } else {
            999.0 - (i as f64) * 999.0 / ((n - 1) as f64)
        };
        let idx = t.round().clamp(0.0, 999.0) as usize;
        out.push(train_sigmas[idx]);
    }
    out.push(0.0);
    out
}

/// Uniform spacing in log-sigma space.
pub fn exponential_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    let log_min = sigma_min.ln();
    let log_max = sigma_max.ln();
    let n = num_steps.max(1);
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            i as f64 / (n - 1) as f64
        };
        out.push((log_max + t * (log_min - log_max)).exp());
    }
    out.push(0.0);
    out
}

/// Precomputed SDXL training-sigma table: σ(t) = sqrt((1 - ᾱ_t) / ᾱ_t)
/// for t in `0..num_train_steps`, using SDXL's `scaled_linear` betas.
/// This is the inverse of the lookup `sigma_to_timestep` does — exposed
/// publicly so the Normal schedule can index by integer timestep.
pub fn sdxl_training_sigmas(num_train_steps: usize) -> Vec<f64> {
    let (beta_start, beta_end) = (0.00085f64, 0.012f64);
    let mut alpha_cumprod = 1.0;
    let mut sigmas = Vec::with_capacity(num_train_steps);
    for i in 0..num_train_steps {
        let t = i as f64 / (num_train_steps - 1).max(1) as f64;
        let s = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
        let beta = s * s;
        let alpha = 1.0 - beta;
        alpha_cumprod *= alpha;
        sigmas.push(((1.0 - alpha_cumprod) / alpha_cumprod).sqrt());
    }
    sigmas
}

/// Sampler selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    Euler,
    EulerAncestral,
    DpmPp2m,
    DpmPpSde,
}

impl Sampler {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "euler_a" | "euler ancestral" | "euler-ancestral" => Sampler::EulerAncestral,
            "dpmpp_2m" | "dpm++ 2m" | "dpmpp-2m" => Sampler::DpmPp2m,
            "dpmpp_sde" | "dpm++ sde" | "dpmpp-sde" => Sampler::DpmPpSde,
            _ => Sampler::Euler,
        }
    }

    /// True iff this sampler needs a SECOND model forward per step
    /// (the predictor at intermediate sigma). The pipeline loop has to
    /// run CFG twice per step in this case.
    pub fn needs_corrector(&self, sigma_next: f64) -> bool {
        matches!(self, Sampler::DpmPpSde) && sigma_next > 0.0
    }
}

/// Per-call multistep sampler state. Holds the previous step's `denoised`
/// + sigma so DPM++ 2M can form its second-order extrapolation. Reset
/// to `Default::default()` at the start of each generation.
#[derive(Default)]
pub struct SamplerState {
    pub prev_denoised: Option<Tensor>,
    pub prev_sigma: Option<f64>,
}

/// One Euler-Ancestral step. Drifts to `sigma_down` deterministically,
/// then re-adds Gaussian noise scaled by `sigma_up`. Defaults match
/// k-diffusion's `sample_euler_ancestral` with eta=1.0, s_noise=1.0.
pub fn euler_ancestral_step(
    x: &Tensor,
    eps: &Tensor,
    sigma_curr: f64,
    sigma_next: f64,
    device: &Device,
) -> Result<Tensor> {
    let (sigma_down, sigma_up) = ancestral_split(sigma_curr, sigma_next, 1.0);
    // Drift via deterministic Euler to sigma_down
    let drift = eps.affine(sigma_down - sigma_curr, 0.0)?;
    let x_drifted = (x + &drift)?;
    if sigma_up <= 0.0 {
        return Ok(x_drifted);
    }
    let noise = Tensor::randn(0f32, 1f32, x.shape(), device)?.to_dtype(x.dtype())?;
    x_drifted + &noise.affine(sigma_up, 0.0)?
}

/// One DPM++ 2M (multistep, deterministic) step. The first time this is
/// called for a run (`state.prev_denoised == None`), or when stepping
/// onto the terminal `sigma_next == 0`, it falls back to the first-order
/// exponential-integrator update. Otherwise it forms the second-order
/// extrapolation against the previous step's denoised estimate.
///
/// Math follows k-diffusion's `sample_dpmpp_2m`. The exponential
/// integrator in `lambda = -log(sigma)` space is unconditionally stable.
pub fn dpmpp_2m_step(
    x: &Tensor,
    eps: &Tensor,
    sigma_curr: f64,
    sigma_next: f64,
    state: &mut SamplerState,
) -> Result<Tensor> {
    // denoised = x - eps * sigma_curr  (eps-prediction → x0 estimate)
    let denoised = (x - &eps.affine(sigma_curr, 0.0)?)?;

    let coeff_x = sigma_next / sigma_curr;
    let h = (-sigma_next.ln()) - (-sigma_curr.ln());
    let em = (-h).exp() - 1.0;

    let x_new = match (state.prev_denoised.as_ref(), state.prev_sigma) {
        (Some(prev_d), Some(prev_s)) if sigma_next > 0.0 && prev_s > 0.0 => {
            let h_last = (-sigma_curr.ln()) - (-prev_s.ln());
            let r = h_last / h;
            let w_new = 1.0 + 1.0 / (2.0 * r);
            let w_old = 1.0 / (2.0 * r);
            let denoised_d = (&denoised.affine(w_new, 0.0)? - &prev_d.affine(w_old, 0.0)?)?;
            (x.affine(coeff_x, 0.0)? - denoised_d.affine(em, 0.0)?)?
        }
        _ => {
            // First step (or terminal): first-order exponential integrator
            (x.affine(coeff_x, 0.0)? - denoised.affine(em, 0.0)?)?
        }
    };
    state.prev_denoised = Some(denoised);
    state.prev_sigma = Some(sigma_curr);
    Ok(x_new)
}

/// DPM++ SDE — predictor (first pass). Returns `(x_2, sigma_mid)` and
/// the `denoised_1` tensor the corrector needs. The pipeline loop is
/// expected to evaluate the model at `sigma_mid` against `x_2` and
/// hand both back to `dpmpp_sde_corrector`.
///
/// Uses k-diffusion's defaults: r=1/2, eta=1, s_noise=1, Gaussian noise.
pub fn dpmpp_sde_predictor(
    x: &Tensor,
    eps: &Tensor,
    sigma_curr: f64,
    sigma_next: f64,
    device: &Device,
) -> Result<(Tensor, f64, Tensor)> {
    let denoised_1 = (x - &eps.affine(sigma_curr, 0.0)?)?;

    // Midpoint in lambda-space (r=1/2)
    let t = -sigma_curr.ln();
    let t_next = -sigma_next.ln();
    let h = t_next - t;
    let s = t + h * 0.5;
    let sigma_mid = (-s).exp();

    // Ancestral split for the predictor step (sigma_curr → sigma_mid)
    let (sigma_d_1, sigma_u_1) = ancestral_split(sigma_curr, sigma_mid, 1.0);
    let t_d_1 = -sigma_d_1.ln();

    // x_2 = (sigma_d_1 / sigma_curr) * x - (exp(t - t_d_1) - 1) * denoised_1
    // (the multiplier `exp(t - t_d_1) - 1` is `expm1(t - t_d_1)`)
    let em = (t - t_d_1).exp() - 1.0;
    let coeff_x = sigma_d_1 / sigma_curr;
    let mut x_2 = (x.affine(coeff_x, 0.0)? - denoised_1.affine(em, 0.0)?)?;

    if sigma_u_1 > 0.0 {
        let noise = Tensor::randn(0f32, 1f32, x.shape(), device)?.to_dtype(x.dtype())?;
        x_2 = (x_2 + &noise.affine(sigma_u_1, 0.0)?)?;
    }

    Ok((x_2, sigma_mid, denoised_1))
}

/// DPM++ SDE — corrector (second pass). Combines the predictor's
/// `denoised_1` with the corrector's `denoised_2` (from the second
/// forward at `sigma_mid`) into the final step from `sigma_curr` to
/// `sigma_next`.
pub fn dpmpp_sde_corrector(
    x: &Tensor,
    denoised_1: &Tensor,
    x_2: &Tensor,
    eps_2: &Tensor,
    sigma_curr: f64,
    sigma_next: f64,
    sigma_mid: f64,
    device: &Device,
) -> Result<Tensor> {
    let denoised_2 = (x_2 - &eps_2.affine(sigma_mid, 0.0)?)?;

    // r=1/2 ⇒ fac = 1 / (2r) = 1, so denoised_d = denoised_2 exactly.
    // Keeping the convex combo here so future r != 1/2 stays correct.
    let fac = 1.0;
    let denoised_d = (denoised_1.affine(1.0 - fac, 0.0)? + denoised_2.affine(fac, 0.0)?)?;

    let t = -sigma_curr.ln();
    let (sigma_d_2, sigma_u_2) = ancestral_split(sigma_curr, sigma_next, 1.0);
    let t_d_2 = -sigma_d_2.ln();

    let em = (t - t_d_2).exp() - 1.0;
    let coeff_x = sigma_d_2 / sigma_curr;
    let mut x_new = (x.affine(coeff_x, 0.0)? - denoised_d.affine(em, 0.0)?)?;

    if sigma_u_2 > 0.0 {
        let noise = Tensor::randn(0f32, 1f32, x.shape(), device)?.to_dtype(x.dtype())?;
        x_new = (x_new + &noise.affine(sigma_u_2, 0.0)?)?;
    }
    Ok(x_new)
}

/// k-diffusion `get_ancestral_step` — given a `from → to` sigma jump,
/// return `(sigma_down, sigma_up)` such that the SDE-equivalent split
/// is `drift to sigma_down + noise of magnitude sigma_up`. eta=0 gives
/// pure deterministic (sigma_down == sigma_to, sigma_up == 0).
fn ancestral_split(sigma_from: f64, sigma_to: f64, eta: f64) -> (f64, f64) {
    if eta == 0.0 || sigma_to <= 0.0 || sigma_from <= 0.0 {
        return (sigma_to, 0.0);
    }
    let var_ratio = (sigma_from * sigma_from - sigma_to * sigma_to) / (sigma_from * sigma_from);
    let raw_up = sigma_to * var_ratio.max(0.0).sqrt();
    let sigma_up = (eta * raw_up).min(sigma_to);
    let sigma_down_sq = (sigma_to * sigma_to - sigma_up * sigma_up).max(0.0);
    (sigma_down_sq.sqrt(), sigma_up)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dpmpp_2m_preserves_multistep_history_and_finishes_at_denoised() {
        let dev = Device::Cpu;
        let mut state = SamplerState::default();
        let x = Tensor::new(&[5f32], &dev).unwrap();
        let eps = Tensor::new(&[1f32], &dev).unwrap();
        let x = dpmpp_2m_step(&x, &eps, 4., 2., &mut state).unwrap();
        assert!((x.to_vec1::<f32>().unwrap()[0] - 3.).abs() < 1e-6);
        let eps = Tensor::new(&[0.5f32], &dev).unwrap();
        let x = dpmpp_2m_step(&x, &eps, 2., 1., &mut state).unwrap();
        // Equal log-sigma intervals give 1.5*x0_new - 0.5*x0_previous.
        assert!((x.to_vec1::<f32>().unwrap()[0] - 2.75).abs() < 1e-6);
        let eps = Tensor::new(&[0f32], &dev).unwrap();
        let x = dpmpp_2m_step(&x, &eps, 1., 0., &mut state).unwrap();
        assert!((x.to_vec1::<f32>().unwrap()[0] - 2.75).abs() < 1e-6);
    }

    #[test]
    fn karras_endpoints() {
        let s = sdxl_default_sigmas(30);
        assert_eq!(s.len(), 31);
        assert!((s[0] - 14.6146).abs() < 1e-3);
        assert_eq!(s[30], 0.0);
        // Monotone decreasing
        for i in 0..s.len() - 1 {
            assert!(
                s[i] >= s[i + 1],
                "schedule not monotone at i={i}: {} < {}",
                s[i],
                s[i + 1]
            );
        }
    }

    #[test]
    fn schedules_all_monotone_decreasing() {
        for schedule in [Schedule::Karras, Schedule::Normal, Schedule::Exponential] {
            let s = build_sigmas(schedule, 28);
            assert_eq!(s.len(), 29);
            assert_eq!(s[28], 0.0);
            for i in 0..s.len() - 1 {
                assert!(
                    s[i] >= s[i + 1],
                    "{:?} not monotone at i={i}: {} < {}",
                    schedule,
                    s[i],
                    s[i + 1]
                );
            }
        }
    }

    #[test]
    fn ancestral_split_eta_zero_is_deterministic() {
        let (d, u) = ancestral_split(2.0, 1.0, 0.0);
        assert_eq!(d, 1.0);
        assert_eq!(u, 0.0);
    }

    #[test]
    fn ancestral_split_eta_one_within_bounds() {
        let (d, u) = ancestral_split(2.0, 1.0, 1.0);
        assert!(u > 0.0 && u <= 1.0);
        assert!(d >= 0.0 && d <= 1.0);
        // sigma_up² + sigma_down² == sigma_to²
        assert!((u * u + d * d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sampler_parse_aliases() {
        assert_eq!(Sampler::parse("Euler"), Sampler::Euler);
        assert_eq!(Sampler::parse("euler_a"), Sampler::EulerAncestral);
        assert_eq!(Sampler::parse("DPM++ 2M"), Sampler::DpmPp2m);
        assert_eq!(Sampler::parse("dpmpp_sde"), Sampler::DpmPpSde);
        assert_eq!(Sampler::parse("nonexistent"), Sampler::Euler);
    }

    #[test]
    fn sigma_to_timestep_endpoints() {
        // sigma=0 → t=0; sigma very large → t≈999
        let t0 = sigma_to_timestep(0.0001, 1000);
        let t1 = sigma_to_timestep(100.0, 1000);
        assert!(t0 < 1.0);
        assert!(t1 > 998.0);
    }
}
