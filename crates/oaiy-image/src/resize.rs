//! Pillow's resampling (`Image.resize`, libImaging/Resample.c) and the
//! `ImageOps.contain` / `ImageOps.pad` built on it, pixel-exact.
//!
//! Resample.c filters in two passes, horizontal then vertical, each with
//! per-output-pixel coefficients: the filter is stretched by the scale when
//! shrinking (antialiasing), its taps normalized to sum to 1, then turned
//! into 22-bit fixed point; every pass rounds to 8 bits. Bicubic (Keys,
//! a = -0.5, what `ImageOps.pad` and `resize` default to) and bilinear (what
//! torchvision's `Resize` asks of a PIL image) are here, for any number of
//! 8-bit channels.

use crate::Image;

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// A resampling filter, as Pillow's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Bilinear,
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Bicubic => {
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// Taps for one axis: per output pixel the first input pixel, the tap
/// count and `ksize` fixed-point weights (precompute_coeffs +
/// normalize_coeffs_8bpc).
struct Taps {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    k: Vec<i32>,
}

fn taps(in_size: usize, out_size: usize, filter: Filter) -> Taps {
    // the box edges are C floats
    let (in0, in1) = (0f32, in_size as f32);
    let scale = f64::from(in1 - in0) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut k = vec![0i32; out_size * ksize];
    let mut w = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C's (int) truncates toward zero
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let mut ww = 0.0;
        for (x, wx) in w.iter_mut().enumerate().take(xmax) {
            *wx = filter.weight((x as f64 + xmin as f64 - center + 0.5) * ss);
            ww += *wx;
        }
        for (x, wx) in w.iter().enumerate().take(xmax) {
            let v = if ww != 0.0 { wx / ww } else { *wx };
            let f = v * f64::from(1u32 << PRECISION_BITS);
            k[xx * ksize + x] = if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    Taps { ksize, bounds, k }
}

#[inline]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((w, h), filter)` on `c`-channel 8-bit pixels of a `sw` × `sh` image.
pub fn resample(src: &[u8], c: usize, sw: usize, sh: usize, w: usize, h: usize, filter: Filter) -> Vec<u8> {
    if (w, h) == (sw, sh) {
        return src.to_vec();
    }
    let horiz = taps(sw, w, filter);
    let vert = taps(sh, h, filter);
    let mut pixels: std::borrow::Cow<'_, [u8]> = std::borrow::Cow::Borrowed(src);
    let mut src_w = sw;
    let mut row0 = 0; // first source row the vertical pass reads
    if w != sw {
        // horizontal pass, over just the rows the vertical pass will read
        let (first, last) = if h == sh { (0, sh) } else { (vert.bounds[0].0, vert.bounds[h - 1].0 + vert.bounds[h - 1].1) };
        let mut tmp = vec![0u8; w * (last - first) * c];
        for y in first..last {
            let line = &src[y * sw * c..(y + 1) * sw * c];
            let out = &mut tmp[(y - first) * w * c..(y - first + 1) * w * c];
            for (xx, &(xmin, n)) in horiz.bounds.iter().enumerate() {
                let k = &horiz.k[xx * horiz.ksize..xx * horiz.ksize + n];
                for ch in 0..c {
                    let mut ss = 1i32 << (PRECISION_BITS - 1);
                    for (x, &kx) in k.iter().enumerate() {
                        ss = ss.wrapping_add(i32::from(line[(xmin + x) * c + ch]).wrapping_mul(kx));
                    }
                    out[xx * c + ch] = clip8(ss);
                }
            }
        }
        pixels = std::borrow::Cow::Owned(tmp);
        src_w = w;
        row0 = first;
    }
    if h == sh {
        return pixels.into_owned();
    }
    let mut out = vec![0u8; w * h * c];
    let stride = src_w * c;
    for (yy, &(ymin, n)) in vert.bounds.iter().enumerate() {
        let k = &vert.k[yy * vert.ksize..yy * vert.ksize + n];
        let o = &mut out[yy * w * c..(yy + 1) * w * c];
        for (i, px) in o.iter_mut().enumerate() {
            let mut ss = 1i32 << (PRECISION_BITS - 1);
            for (y, &ky) in k.iter().enumerate() {
                ss = ss.wrapping_add(i32::from(pixels[(ymin - row0 + y) * stride + i]).wrapping_mul(ky));
            }
            *px = clip8(ss);
        }
    }
    out
}

/// `Image.resize((w, h), Image.BICUBIC)` on an RGB image.
pub fn resize_bicubic(img: &Image, w: usize, h: usize) -> Image {
    Image { width: w, height: h, rgb: resample(&img.rgb, 3, img.width, img.height, w, h, Filter::Bicubic) }
}

/// `Image.resize((w, h), Image.BILINEAR)` on an RGB image.
pub fn resize_bilinear(img: &Image, w: usize, h: usize) -> Image {
    Image { width: w, height: h, rgb: resample(&img.rgb, 3, img.width, img.height, w, h, Filter::Bilinear) }
}

/// Python's `round()` on a float: half to even.
fn round_py(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The size `ImageOps.contain(img, (w, h))` resizes to: the largest that
/// fits with the aspect ratio kept, in the reference's float arithmetic.
pub fn contain_size(width: usize, height: usize, w: usize, h: usize) -> (usize, usize) {
    let im_ratio = width as f64 / height as f64;
    let dest_ratio = w as f64 / h as f64;
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let nh = round_py(height as f64 / width as f64 * w as f64).max(1) as usize;
            return (w, nh);
        }
        let nw = round_py(width as f64 / height as f64 * h as f64).max(1) as usize;
        return (nw, h);
    }
    (w, h)
}

/// `ImageOps.pad(img, (w, h), method=BICUBIC, color=color)`: resize to fit,
/// centered on a `color` background.
pub fn pad(img: &Image, w: usize, h: usize, color: [u8; 3]) -> Image {
    let (rw, rh) = contain_size(img.width, img.height, w, h);
    let resized = resize_bicubic(img, rw, rh);
    if (rw, rh) == (w, h) {
        return resized;
    }
    let mut out = Image::filled(w, h, color);
    // Python rounds the half-pixel offsets to even
    let (x0, y0) = if rw != w { (round_py((w - rw) as f64 * 0.5) as usize, 0) } else { (0, round_py((h - rh) as f64 * 0.5) as usize) };
    for y in 0..rh.min(h - y0) {
        let n = rw.min(w - x0) * 3;
        let dst = ((y0 + y) * w + x0) * 3;
        out.rgb[dst..dst + n].copy_from_slice(&resized.rgb[y * rw * 3..y * rw * 3 + n]);
    }
    out
}
