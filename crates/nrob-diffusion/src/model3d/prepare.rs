//! The input picture, as Pixal3D wants it (its `preprocess_image`): the object
//! cut out (the picture's own alpha when it has one; otherwise its background,
//! when it is plain, flood-filled from the border), scaled to at most 1024,
//! cropped square to 1.1× the object's bounding box, and put on black.
//!
//! Pixal3D cuts objects out with RMBG-2.0, which is for non-commercial use only;
//! this takes pictures with transparency, or of an object on a plain background
//! (as generated product shots and cut-outs are).
use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, Rgb, RgbImage, Rgba, RgbaImage};
use std::collections::VecDeque;
use std::path::Path;

pub struct Prepared {
    /// Square, the object on black.
    pub image: RgbImage,
    /// How the object was found: "alpha" or "background".
    pub matte: &'static str,
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

pub fn prepare(path: &Path) -> Result<Prepared> {
    let img = image::ImageReader::open(path)?.with_guessed_format()?.decode().map_err(candle_core::Error::wrap)?;
    let mut rgba = img.to_rgba8();
    let has_alpha = rgba.pixels().any(|p| p[3] != 255);
    // At most 1024 on its longer side (LANCZOS, as PIL).
    let (w, h) = rgba.dimensions();
    let longest = w.max(h);
    if longest > 1024 {
        let s = 1024. / longest as f64;
        rgba = image::imageops::resize(&rgba, ((w as f64 * s) as u32).max(1), ((h as f64 * s) as u32).max(1), FilterType::Lanczos3);
    }
    let matte = if has_alpha {
        "alpha"
    } else {
        let Some(bg) = plain_background(&rgba) else {
            candle_core::bail!("the picture has no transparency and no plain background to cut the object out of: give it a transparent or plain (white, grey or black) background, with the whole object in view");
        };
        let background = flood(&rgba, bg, 28.);
        let (w, _) = rgba.dimensions();
        for (i, p) in rgba.pixels_mut().enumerate() {
            if background[i] {
                p[3] = 0;
            }
            let _ = w;
        }
        "background"
    };
    // The object's box: alpha above 0.8.
    let (w, h) = rgba.dimensions();
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
    let mut out = RgbImage::new(cw, ch);
    for y in 0..ch {
        for x in 0..cw {
            let (sx, sy) = (left + x as i64, top + y as i64);
            let p: Rgba<u8> = if sx >= 0 && sy >= 0 && (sx as u32) < w && (sy as u32) < h { *rgba.get_pixel(sx as u32, sy as u32) } else { Rgba([0, 0, 0, 0]) };
            let a = p[3] as f32 / 255.;
            let c = |v: u8| ((v as f32 / 255. * a).clamp(0., 1.) * 255.) as u8;
            out.put_pixel(x, y, Rgb([c(p[0]), c(p[1]), c(p[2])]));
        }
    }
    Ok(Prepared { image: out, matte })
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
