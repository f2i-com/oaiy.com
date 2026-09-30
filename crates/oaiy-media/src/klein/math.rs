//! FLUX.2 conventions: four coordinate axes, adjacent-pair rotary, and
//! channel-first 2x2 VAE patches. See the BFL reference in docs/KLEIN.md.
use candle_core::{DType, Device, Result, Tensor, D};

pub fn patchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if h % 2 != 0 || w % 2 != 0 {
        candle_core::bail!("Flux2 latent dimensions must be even");
    }
    x.reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * 4, h / 2, w / 2))
}
pub fn unpatchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if c % 4 != 0 {
        candle_core::bail!("Flux2 patch channels must be divisible by four");
    }
    x.reshape((b, c / 4, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, c / 4, h * 2, w * 2))
}
pub fn pack(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.flatten_from(2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, h * w, c))
}
pub fn unpack(x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (b, s, c) = x.dims3()?;
    if h.checked_mul(w) != Some(s) {
        candle_core::bail!("Flux2 token grid disagrees with latent shape");
    }
    x.transpose(1, 2)?.contiguous()?.reshape((b, c, h, w))
}
pub fn ids(text: usize, h: usize, w: usize) -> Vec<[f32; 4]> {
    (0..text)
        .map(|i| [0., 0., 0., i as f32])
        .chain((0..h).flat_map(|y| (0..w).map(move |x| [0., y as f32, x as f32, 0.])))
        .collect()
}
pub fn rotary(
    coords: &[[f32; 4]],
    axes: [usize; 4],
    theta: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    if axes.iter().any(|d| *d == 0 || d % 2 != 0) {
        candle_core::bail!("invalid Flux2 rotary axes");
    }
    let half = axes.iter().sum::<usize>() / 2;
    let mut cos = Vec::with_capacity(coords.len() * half);
    let mut sin = Vec::with_capacity(coords.len() * half);
    for coord in coords {
        for (axis, width) in axes.iter().enumerate() {
            for i in 0..width / 2 {
                let angle = coord[axis] as f64 / theta.powf((2 * i) as f64 / *width as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (1, 1, coords.len(), half), dev)?,
        Tensor::from_vec(sin, (1, 1, coords.len(), half), dev)?,
    ))
}
pub fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, n, s, d) = x.dims4()?;
    if d % 2 != 0 {
        candle_core::bail!("rotary head dimension must be even");
    }
    let pairs = x.to_dtype(DType::F32)?.reshape((b, n, s, d / 2, 2))?;
    let a = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let z = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let real = (a.broadcast_mul(cos)? - z.broadcast_mul(sin)?)?;
    let imag = (a.broadcast_mul(sin)? + z.broadcast_mul(cos)?)?;
    Tensor::stack(&[real, imag], D::Minus1)?
        .reshape((b, n, s, d))?
        .to_dtype(x.dtype())
}
pub fn timestep(t: f64, dev: &Device, dtype: DType) -> Result<Tensor> {
    let phase: Vec<f32> = (0..128)
        .map(|i| (t * 1000. * (-10000f64.ln() * i as f64 / 128.).exp()) as f32)
        .collect();
    let values = phase
        .iter()
        .map(|a| a.cos())
        .chain(phase.iter().map(|a| a.sin()))
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 256), dev)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patches_have_the_published_channel_pixel_order() -> Result<()> {
        let x = Tensor::from_vec(
            (0..16).map(|i| i as f32).collect(),
            (1, 2, 2, 4),
            &Device::Cpu,
        )?;
        let patched = patchify(&x)?;
        assert_eq!(
            patched.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 2., 1., 3., 4., 6., 5., 7., 8., 10., 9., 11., 12., 14., 13., 15.]
        );
        assert_eq!(
            unpatchify(&patched)?.flatten_all()?.to_vec1::<f32>()?,
            x.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            unpack(&pack(&patched)?, 1, 2)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            patched.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
    #[test]
    fn rotary_uses_adjacent_pairs_and_four_independent_axes() -> Result<()> {
        let (c, s) = rotary(
            &[[0., std::f32::consts::FRAC_PI_2, 0., 0.]],
            [2; 4],
            2000.,
            &Device::Cpu,
        )?;
        let x = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 1, 1, 8),
            &Device::Cpu,
        )?;
        let y = rotate(&x, &c, &s)?.flatten_all()?.to_vec1::<f32>()?;
        for (a, b) in y.iter().zip([1., 2., -4., 3., 5., 6., 7., 8.]) {
            assert!((a - b).abs() < 1e-5);
        }
        Ok(())
    }
}
