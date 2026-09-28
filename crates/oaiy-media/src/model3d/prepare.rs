//! The input picture, as Pixal3D wants it (its `preprocess_image`): scaled to
//! at most 1024, the object cut out, cropped square to 1.1× its bounding box,
//! and put on black.
//!
//! The object is cut out by the picture's own alpha when it has one; otherwise
//! by BiRefNet (as Pixal3D does, with BiRefNet's MIT general-use weights rather
//! than RMBG-2.0's non-commercial ones); without BiRefNet, by flood-filling a
//! plain background from the border.
//!
//! With Real-ESRGAN, the square is made 2048 pixels a side: cut from the
//! picture at its own resolution (not the 1024 the object is found at), and
//! when that has fewer pixels, made four times larger by Real-ESRGAN rather than
//! stretched. DINOv3 and NAF still see it at 512 and 1024, as Pixal3D is
//! trained, now from a sharp picture scaled down; the cut-out kept beside the
//! model is the full 2048.
use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, Rgb, RgbImage, Rgba, RgbaImage};
use std::collections::VecDeque;
use std::path::Path;

/// The side the square is made with Real-ESRGAN.
const UPSCALED: u32 = 2048;

pub struct Prepared {
    /// Square, the object on black.
    pub image: RgbImage,
    /// The same, with the background transparent.
    pub cutout: RgbaImage,
    /// How the object was found: "alpha", "birefnet" or "background".
    pub matte: &'static str,
    /// The square's side in the picture, and as made (2048), when Real-ESRGAN was there to make it.
    pub upscaled: Option<(u32, u32)>,
}

/// The models that help prepare a picture, loaded while they are used.
pub struct Helpers<'a> {
    /// A BiRefNet folder, to cut the object out of a picture without transparency.
    pub matte: Option<&'a Path>,
    /// Real-ESRGAN x4plus weights: the square is made 2048 a side, enlarged where it has fewer pixels.
    pub upscaler: Option<&'a Path>,
    pub dev: &'a Device,
}

/// A plain background's colour, when the border is (nearly) one colour.
fn plain_background(img: &RgbaImage) -> Option<[f32; 3]> {
    let (w, h) = img.dimensions();
    let mut border = Vec::new();
    for x in 0..w {
        border.push(*img.get_pixel(x, 0));
        border.push(*img.get_pixel(x, h - 1));
    }
    for y in 0..h {
        border.push(*img.get_pixel(0, y));
        border.push(*img.get_pixel(w - 1, y));
    }
    let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for p in &border {
        for c in 0..3 {
            channels[c].push(p[c]);
        }
    }
    let median: Vec<f32> = channels
        .iter_mut()
        .map(|v| {
            v.sort_unstable();
            v[v.len() / 2] as f32
        })
        .collect();
    let near = border.iter().filter(|p| (0..3).all(|c| (p[c] as f32 - median[c]).abs() < 24.)).count();
    (near as f64 / border.len() as f64 > 0.85).then(|| [median[0], median[1], median[2]])
}

/// Background pixels: those like `bg` reachable from the border.
fn flood(img: &RgbaImage, bg: [f32; 3], tolerance: f32) -> Vec<bool> {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let like = |x: usize, y: usize| {
        let p = img.get_pixel(x as u32, y as u32);
        (0..3).map(|c| (p[c] as f32 - bg[c]).abs()).fold(0f32, f32::max) < tolerance
    };
    let mut background = vec![false; w * h];
    let mut queue = VecDeque::new();
    let seed = |x: usize, y: usize, bgd: &mut Vec<bool>, q: &mut VecDeque<(usize, usize)>| {
        if !bgd[y * w + x] && like(x, y) {
            bgd[y * w + x] = true;
            q.push_back((x, y));
        }
    };
    for x in 0..w {
        seed(x, 0, &mut background, &mut queue);
        seed(x, h - 1, &mut background, &mut queue);
    }
    for y in 0..h {
        seed(0, y, &mut background, &mut queue);
        seed(w - 1, y, &mut background, &mut queue);
    }
    while let Some((x, y)) = queue.pop_front() {
        let mut visit = |nx: usize, ny: usize| {
            if !background[ny * w + nx] && like(nx, ny) {
                background[ny * w + nx] = true;
                queue.push_back((nx, ny));
            }
        };
        if x > 0 {
            visit(x - 1, y);
        }
        if x + 1 < w {
            visit(x + 1, y);
        }
        if y > 0 {
            visit(x, y - 1);
        }
        if y + 1 < h {
            visit(x, y + 1);
        }
    }
    background
}

/// Does the picture's own alpha cut the object out? Pixal3D trusts any alpha
/// below 255; here it takes clearly transparent pixels (alpha under 32) over at
/// least half a percent of the picture, so a picture with a translucent edge or
/// a stray alpha channel still has its background removed.
pub fn cuts_out(img: &RgbaImage) -> bool {
    let transparent = img.pixels().filter(|p| p[3] < 32).count();
    transparent * 200 >= img.pixels().len().max(1)
}

pub fn prepare(path: &Path, helpers: &Helpers) -> Result<Prepared> {
    let img = image::ImageReader::open(path)?.with_guessed_format()?.decode().map_err(candle_core::Error::wrap)?;
    let mut original = img.to_rgba8();
    let has_alpha = cuts_out(&original);
    // A picture merely a little translucent (an alpha channel an export left
    // behind) has an opaque background still: it is cut out like one without.
    if !has_alpha {
        for p in original.pixels_mut() {
            p[3] = 255;
        }
    }
    let mut rgba = original.clone();
    // At most 1024 on its longer side (LANCZOS, as PIL).
    let (w, h) = rgba.dimensions();
    let longest = w.max(h);
    if longest > 1024 {
        let s = 1024. / longest as f64;
        rgba = image::imageops::resize(&rgba, ((w as f64 * s) as u32).max(1), ((h as f64 * s) as u32).max(1), FilterType::Lanczos3);
    }
    let (w, h) = rgba.dimensions();
    let matte = if has_alpha {
        "alpha"
    } else if let Some(dir) = helpers.matte {
        let net = crate::birefnet::BiRefNet::load(dir, helpers.dev)?;
        let rgb: Vec<u8> = rgba.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let alpha = net.matte(&rgb, w as usize, h as usize)?;
        for (p, a) in rgba.pixels_mut().zip(alpha) {
            p[3] = a;
        }
        "birefnet"
    } else {
        let Some(bg) = plain_background(&rgba) else {
            candle_core::bail!("the picture has no transparency and no plain background to cut the object out of: give it a transparent or plain (white, grey or black) background, with the whole object in view (or add BiRefNet to the 3D model, which cuts objects out of any background)");
        };
        let background = flood(&rgba, bg, 28.);
        for (i, p) in rgba.pixels_mut().enumerate() {
            if background[i] {
                p[3] = 0;
            }
        }
        "background"
    };
    // The object's box: alpha above 0.8.
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for (x, y, p) in rgba.enumerate_pixels() {
        if p[3] as f32 > 0.8 * 255. {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x0 == u32::MAX {
        candle_core::bail!("no object found in the picture (it is all background)");
    }
    let cx = (x0 + x1) as f64 / 2.;
    let cy = (y0 + y1) as f64 / 2.;
    let size = ((x1 - x0).max(y1 - y0) as f64 * 1.1) as i64;
    // PIL's crop rounds its box; outside the picture is transparent.
    let left = (cx - (size / 2) as f64).round_ties_even() as i64;
    let top = (cy - (size / 2) as f64).round_ties_even() as i64;
    let right = (cx + (size / 2) as f64).round_ties_even() as i64;
    let bottom = (cy + (size / 2) as f64).round_ties_even() as i64;
    let (cw, ch) = ((right - left).max(1) as u32, (bottom - top).max(1) as u32);
    let mut upscaled = None;
    let crop = match helpers.upscaler {
        None => cut(&rgba, left, top, cw, ch),
        Some(weights) => {
            // The same square at the picture's own resolution, its alpha from the matte found at 1024.
            let (ow, oh) = original.dimensions();
            let k = ow as f64 / w as f64;
            let mut full = original;
            if !has_alpha && k > 1. {
                let alpha: Vec<u8> = rgba.pixels().map(|p| p[3]).collect();
                let alpha = oaiy_image::resize::resample(&alpha, 1, w as usize, h as usize, ow as usize, oh as usize, oaiy_image::resize::Filter::Bicubic);
                for (p, a) in full.pixels_mut().zip(alpha) {
                    p[3] = a;
                }
            } else if !has_alpha {
                full = rgba;
            }
            let side = ((cw as f64 * k).round() as u32).max(1);
            let square = cut(&full, (left as f64 * k).round() as i64, (top as f64 * k).round() as i64, side, side);
            drop(full);
            // Fewer than half the pixels wanted: four times larger with Real-ESRGAN (its alpha bicubic, as Pillow).
            let square = if side < UPSCALED / 2 {
                let net = crate::esrgan::Esrgan::load(weights, helpers.dev)?;
                let s = side as usize;
                let rgb: Vec<u8> = square.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
                let alpha: Vec<u8> = square.pixels().map(|p| p[3]).collect();
                let big = net.upscale(&rgb, s, s)?;
                let b = s * crate::esrgan::SCALE;
                let big_alpha = oaiy_image::resize::resample(&alpha, 1, s, s, b, b, oaiy_image::resize::Filter::Bicubic);
                let mut bigger = RgbaImage::new(b as u32, b as u32);
                for (i, p) in bigger.pixels_mut().enumerate() {
                    *p = Rgba([big[i * 3], big[i * 3 + 1], big[i * 3 + 2], big_alpha[i]]);
                }
                bigger
            } else {
                square
            };
            upscaled = Some((side, UPSCALED));
            if square.width() == UPSCALED {
                square
            } else {
                image::imageops::resize(&square, UPSCALED, UPSCALED, FilterType::Lanczos3)
            }
        }
    };
    // On black, as Pixal3D composites it; and with the background transparent.
    let (cw, ch) = crop.dimensions();
    let mut image = RgbImage::new(cw, ch);
    for (x, y, p) in crop.enumerate_pixels() {
        let a = p[3] as f32 / 255.;
        let c = |v: u8| ((v as f32 / 255. * a).clamp(0., 1.) * 255.) as u8;
        image.put_pixel(x, y, Rgb([c(p[0]), c(p[1]), c(p[2])]));
    }
    Ok(Prepared { image, cutout: crop, matte, upscaled })
}

/// `side`-wide square (`w` × `h` at `left`, `top`) of `img`; outside it is transparent (as PIL's crop).
fn cut(img: &RgbaImage, left: i64, top: i64, w: u32, h: u32) -> RgbaImage {
    let (iw, ih) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (left + x as i64, top + y as i64);
            if sx >= 0 && sy >= 0 && (sx as u32) < iw && (sy as u32) < ih {
                out.put_pixel(x, y, *img.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    out
}

/// The prepared picture at `size`² (LANCZOS) as [3, size, size] in [0, 1].
pub fn tensor(img: &RgbImage, size: u32, dev: &Device) -> Result<Tensor> {
    let resized = image::imageops::resize(img, size, size, FilterType::Lanczos3);
    let data: Vec<f32> = resized.into_raw().into_iter().map(|v| v as f32 / 255.).collect();
    Tensor::from_vec(data, (size as usize, size as usize, 3), dev)?.permute((2, 0, 1))?.contiguous()
}

/// ImageNet normalization for DINOv3.
pub fn imagenet(x: &Tensor) -> Result<Tensor> {
    let dev = x.device();
    let mean = Tensor::from_vec(vec![0.485f32, 0.456, 0.406], (3, 1, 1), dev)?;
    let std = Tensor::from_vec(vec![0.229f32, 0.224, 0.225], (3, 1, 1), dev)?;
    x.broadcast_sub(&mean)?.broadcast_div(&std)
}
