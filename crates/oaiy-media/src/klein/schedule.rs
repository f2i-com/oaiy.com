//! BFL generalized time/SNR shift and forward-velocity Euler integration.
use candle_core::{Result, Tensor};
pub fn sigmas(steps: usize, tokens: usize) -> Result<Vec<f64>> {
    if !(1..=100).contains(&steps) || tokens == 0 {
        candle_core::bail!("invalid Klein schedule dimensions");
    }
    let seq = tokens as f64;
    let mu = if tokens > 4300 {
        0.00016927 * seq + 0.45666666
    } else {
        let m200 = 0.00016927 * seq + 0.45666666;
        let m10 = 8.73809524e-5 * seq + 1.89833333;
        let a = (m200 - m10) / 190.;
        a * steps as f64 + m200 - 200. * a
    };
    let shift = mu.exp();
    Ok((0..=steps)
        .map(|i| {
            let t = 1. - i as f64 / steps as f64;
            if i == steps {
                0.
            } else {
                shift / (shift + 1. / t - 1.)
            }
        })
        .collect())
}
pub fn euler(x: &Tensor, velocity: &Tensor, now: f64, next: f64) -> Result<Tensor> {
    x + (velocity * (next - now))?
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedule_matches_bfl_reference_and_velocity_moves_toward_clean() -> Result<()> {
        let s = sigmas(4, 4096)?;
        // Independently evaluated reference formula, not a linear/Qwen schedule.
        let mu: f64 = 2.2911798941155705;
        assert!((s[1] - mu.exp() / (mu.exp() + 1. / 0.75 - 1.)).abs() < 1e-8);
        assert_eq!((s[0], s[4]), (1., 0.));
        assert!(s.windows(2).all(|p| p[0] > p[1]));
        let x = Tensor::new(&[10f32], &candle_core::Device::Cpu)?;
        let v = Tensor::new(&[2f32], x.device())?;
        assert_eq!(euler(&x, &v, 1., 0.)?.to_vec1::<f32>()?, vec![8.]);
        Ok(())
    }
}
