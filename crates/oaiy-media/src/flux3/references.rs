//! A FLUX 3 edit's reference pictures, prepared as `flux3/pipeline.py` prepares them: turned upright by their EXIF
//! orientation, RGB; a single one fitted within its canvas (at most 4 MiB pixels, sides 64 to 4,096, multiples of 16)
//! and mirrored out to it, centred (the picture made is cut back to it); each of several fitted under 1 MiB pixels and
//! cut, centred, to multiples of 16.
use candle_core::{bail, Result};
use image::{imageops::FilterType, RgbImage};
use std::path::{Path, PathBuf};

const MIN_SIDE: usize = 64;
const MAX_SIDE: usize = 4096;
/// Each of several references at most this many pixels.
const MULTI_PIXELS: usize = 1024 * 1024;
/// The VAE's and the patches' reduction together.
const DOWN: usize = 16;

/// A box in a picture: its left, top, right and bottom.
pub type Box4 = (usize, usize, usize, usize);

/// A prepared reference: its pixels' rows of red, green and blue in -1..1, as the VAE's encoder takes them.
pub struct Picture {
    pub rgb: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

pub struct Prepared {
    pub pictures: Vec<Picture>,
    /// The first's size: the picture's where none is asked for.
    pub size: Option<(usize, usize)>,
    /// A single padded reference's own box in its canvas (left, top, right, bottom): what the picture is cut to.
    pub crop: Option<Box4>,
}

/// The references at `paths`, a single one's canvas at most `max_pixels`.
pub fn prepare(paths: &[PathBuf], max_pixels: usize) -> Result<Prepared> {
    if paths.len() > 1 {
        let pictures = paths.iter().map(|p| Ok(pixels(&multi(open(p)?)?))).collect::<Result<Vec<_>>>()?;
        let size = pictures.first().map(|p| (p.width, p.height));
        return Ok(Prepared { pictures, size, crop: None });
    }
    let Some(path) = paths.first() else { return Ok(Prepared { pictures: Vec::new(), size: None, crop: None }) };
    let (picture, crop) = single(open(path)?, max_pixels)?;
    let picture = pixels(&picture);
    let size = Some((picture.width, picture.height));
    Ok(Prepared { pictures: vec![picture], size, crop })
}

/// A picture's file decoded, upright, as RGB.
fn open(path: &Path) -> Result<RgbImage> {
    use image::ImageDecoder;
    let meta = std::fs::metadata(path)?;
    if !meta.is_file() || meta.len() > 64 * 1024 * 1024 {
        bail!("a reference picture must be a regular image file of at most 64 MiB: {}", path.display());
    }
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(1024 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(candle_core::Error::wrap)?;
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut picture = image::DynamicImage::from_decoder(decoder).map_err(candle_core::Error::wrap)?;
    picture.apply_orientation(orientation);
    Ok(picture.to_rgb8())
}

/// `v` up to the next multiple of [`DOWN`].
fn ceil16(v: usize) -> usize {
    v.div_ceil(DOWN) * DOWN
}

/// A single reference's fitted size and canvas (`reference_canvas`): the canvas the fitted size rounded up to 16, its
/// sides 64 to 4,096, its area at most `max_pixels`, the scale cut by half a percent at a time until it is.
pub fn canvas(width: usize, height: usize, max_pixels: usize) -> ((usize, usize), (usize, usize)) {
    let (w, h) = (width as f64, height as f64);
    let mut scale = 1f64.min((max_pixels as f64 / (w * h)).sqrt()).min(MAX_SIDE as f64 / w.max(h));
    loop {
        // (Python's round: halves to even)
        let fit = (((w * scale).round_ties_even() as usize).max(1), ((h * scale).round_ties_even() as usize).max(1));
        let canvas = (ceil16(fit.0).clamp(MIN_SIDE, MAX_SIDE), ceil16(fit.1).clamp(MIN_SIDE, MAX_SIDE));
        if canvas.0 * canvas.1 <= max_pixels {
            return (fit, canvas);
        }
        scale *= 0.995;
    }
}

/// `prepare_reference`: fitted (Lanczos), then mirrored out to its canvas, centred; with the box it fills where padded.
fn single(picture: RgbImage, max_pixels: usize) -> Result<(RgbImage, Option<Box4>)> {
    let (fit, (cw, ch)) = canvas(picture.width() as usize, picture.height() as usize, max_pixels);
    let picture = if fit == (picture.width() as usize, picture.height() as usize) { picture } else { image::imageops::resize(&picture, fit.0 as u32, fit.1 as u32, FilterType::Lanczos3) };
    let (w, h) = fit;
    if (cw, ch) == (w, h) {
        return Ok((picture, None));
    }
    let (left, top) = ((cw - w) / 2, (ch - h) / 2);
    // numpy's reflect (the edge not repeated) where the picture is wider than a patch, else its symmetric
    let reflect = w.min(h) > DOWN;
    let mirror = |i: isize, n: usize| -> u32 {
        let n = n as isize;
        let period = if reflect { 2 * (n - 1) } else { 2 * n };
        if period == 0 {
            return 0;
        }
        let m = i.rem_euclid(period);
        (if reflect { if m < n { m } else { period - m } } else if m < n { m } else { period - 1 - m }) as u32
    };
    let padded = RgbImage::from_fn(cw as u32, ch as u32, |x, y| *picture.get_pixel(mirror(x as isize - left as isize, w), mirror(y as isize - top as isize, h)));
    Ok((padded, Some((left, top, left + w, top + h))))
}

/// `prepare_multi_reference`: fitted under 1 MiB pixels (its sides cut down), then cut, centred, to multiples of 16.
fn multi(picture: RgbImage) -> Result<RgbImage> {
    let (mut w, mut h) = (picture.width() as usize, picture.height() as usize);
    let picture = if w * h > MULTI_PIXELS {
        let scale = (MULTI_PIXELS as f64 / (w * h) as f64).sqrt();
        (w, h) = ((w as f64 * scale) as usize, (h as f64 * scale) as usize);
        image::imageops::resize(&picture, w as u32, h as u32, FilterType::Lanczos3)
    } else {
        picture
    };
    let (cw, ch) = (w / DOWN * DOWN, h / DOWN * DOWN);
    if cw == 0 || ch == 0 {
        bail!("a reference picture of {w}x{h} is smaller than the model's {DOWN}-pixel patches");
    }
    Ok(image::imageops::crop_imm(&picture, ((w - cw) / 2) as u32, ((h - ch) / 2) as u32, cw as u32, ch as u32).to_image())
}

/// A picture's pixels as `to_tensor` makes them: each byte over 127.5, less 1.
fn pixels(picture: &RgbImage) -> Picture {
    Picture { rgb: picture.as_raw().iter().map(|&b| b as f32 / 127.5 - 1.).collect(), width: picture.width() as usize, height: picture.height() as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `reference_canvas` as the pipeline computes it (its values from Python): a photograph's 4:3 under 4 MiB, a
    /// small square padded up to 64, a long strip's sides capped.
    #[test]
    fn canvases_are_the_pipelines() {
        const FOUR: usize = 4 * 1024 * 1024;
        assert_eq!(canvas(4032, 3024, FOUR), ((2341, 1756), (2352, 1760)));
        assert_eq!(canvas(1000, 667, FOUR), ((1000, 667), (1008, 672)));
        assert_eq!(canvas(40, 40, FOUR), ((40, 40), (64, 64)));
        assert_eq!(canvas(8000, 500, FOUR), ((4096, 256), (4096, 256)));
    }

    /// A small picture mirrored out to its canvas: the reflection doesn't repeat the edge, the symmetric one does.
    #[test]
    fn padding_mirrors_as_numpy_does() -> Result<()> {
        let picture = RgbImage::from_fn(20, 18, |x, y| image::Rgb([x as u8, y as u8, 0]));
        let (padded, crop) = single(picture, 4 * 1024 * 1024)?;
        assert_eq!(padded.dimensions(), (64, 64));
        assert_eq!(crop, Some((22, 23, 42, 41)));
        // (left 22: column 21 is the picture's column 1, column 20 its 2)
        assert_eq!(padded.get_pixel(21, 30)[0], 1);
        assert_eq!(padded.get_pixel(20, 30)[0], 2);
        assert_eq!((padded.get_pixel(30, 22)[1], padded.get_pixel(30, 41)[1]), (1, 16));
        let tiny = RgbImage::from_fn(10, 10, |x, _| image::Rgb([x as u8, 0, 0]));
        let (padded, _) = single(tiny, 4 * 1024 * 1024)?;
        // (left 27: symmetric, column 26 is the picture's column 0 again)
        assert_eq!(padded.get_pixel(26, 5)[0], 0);
        assert_eq!(padded.get_pixel(25, 5)[0], 1);
        assert_eq!(padded.get_pixel(37, 5)[0], 9);
        Ok(())
    }

    #[test]
    fn several_references_are_cut_to_patches() -> Result<()> {
        let cut = multi(RgbImage::new(1500, 1000))?;
        // (1500 x 1000 over 1 MiB: 1254 x 836, cut to 1248 x 832)
        assert_eq!(cut.dimensions(), (1248, 832));
        assert_eq!(multi(RgbImage::new(100, 70))?.dimensions(), (96, 64));
        Ok(())
    }
}
