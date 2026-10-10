//! FlowMatch Euler with the resolution-dependent exponential time shift.

/// Qwen-Image-2.1-Turbo's own schedule (the distilled checkpoint, not an adapter): its 8 steps at CFG 1, as its
/// release samples them (stable-diffusion.cpp's `--sigmas` in AtomicChat's GGUF release).
pub const DISTILLED: [f64; 9] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0];

/// The distilled checkpoint's sigmas for `steps` (8, the steps it was distilled to).
pub fn distilled(steps: usize) -> Result<Vec<f64>, String> {
    if steps != 8 {
        return Err("Qwen-Image-2.1-Turbo is distilled to 8 steps".into());
    }
    Ok(DISTILLED.to_vec())
}

pub fn sigmas(steps: usize, tokens: usize, turbo: bool) -> Result<Vec<f64>, String> {
    let raw = if turbo {
        match steps {
            4 => vec![1., 0.75, 0.5, 0.25],
            6 => vec![1., 0.9375, 0.875, 0.75, 0.5, 0.25],
            _ => return Err("Viggle turbo supports 4 or 6 steps".into()),
        }
    } else {
        if !(2..=100).contains(&steps) {
            return Err("base steps must be between 2 and 100".into());
        }
        (0..steps).map(|i| 1. - i as f64 / steps as f64).collect()
    };
    let mu = 0.5 + (tokens as f64 - 256.) * (0.9 - 0.5) / (8192. - 256.);
    let shift = mu.exp();
    let mut sigmas: Vec<_> = raw
        .into_iter()
        .map(|s| shift / (shift + 1. / s - 1.))
        .collect();
    if !turbo {
        let scale = (1. - sigmas[sigmas.len() - 1]) / (1. - 0.02);
        for s in &mut sigmas {
            *s = 1. - (1. - *s) / scale;
        }
    }
    sigmas.push(0.);
    Ok(sigmas)
}

#[test]
fn turbo_preserves_trained_nodes() {
    let four = sigmas(4, 4096, true).unwrap();
    let six = sigmas(6, 4096, true).unwrap();
    assert_eq!(four[0], 1.);
    assert_eq!(six[0], 1.);
    assert_eq!(&four[1..], &six[3..]);
    assert!(six.windows(2).all(|s| s[0] > s[1]));
    assert_eq!(six[6], 0.);
    assert!(sigmas(5, 4096, true).is_err());
}
