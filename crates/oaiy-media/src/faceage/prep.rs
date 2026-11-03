//! Pictures made ready for the models: crops, MiVOLO's letterbox (`class_letterbox`: scaled to fit 384 by 384 with
//! OpenCV's bilinear resize, centred on black, Python's rounding) and its normalisation (ImageNet's mean and spread,
//! RGB, as `prepare_classification_images` makes them from OpenCV's BGR).
use candle_core::{Device, Result, Tensor};

/// An 8-bit RGB picture, row by row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// A box in a picture, in whole pixels: x1 and y1 included, x2 and y2 not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelBox {
    pub x1: usize,
    pub y1: usize,
    pub x2: usize,
    pub y2: usize,
}

impl Rgb {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Option<Self> {
        (width > 0 && height > 0 && data.len() == width * height * 3).then_some(Self { width, height, data })
    }

    /// A JPEG or PNG, decoded: refused from its header when it says it is larger than the job takes, before anything
    /// is made for it. A picture of a few hundred bytes can say it is 23000 pixels square, and decoded first and
    /// measured after, it took 2 GB.
    pub fn decode(bytes: &[u8]) -> std::result::Result<Self, String> {
        let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| format!("a picture that does not decode: {e}"))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(super::MAX_SIDE as u32);
        limits.max_image_height = Some(super::MAX_SIDE as u32);
        reader.limits(limits);
        let img = reader
            .decode()
            .map_err(|e| match e {
                image::ImageError::Limits(_) => format!("a picture larger than {} pixels on a side", super::MAX_SIDE),
                e => format!("a picture that does not decode: {e}"),
            })?
            // In place where it is RGB already: no second copy of the picture.
            .into_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Self::new(w, h, img.into_raw()).ok_or_else(|| "an empty picture".into())
    }

    /// The box's part of the picture (clipped to it).
    pub fn crop(&self, b: PixelBox) -> Option<Self> {
        let (x2, y2) = (b.x2.min(self.width), b.y2.min(self.height));
        if b.x1 >= x2 || b.y1 >= y2 {
            return None;
        }
        let mut data = Vec::with_capacity((x2 - b.x1) * (y2 - b.y1) * 3);
        for y in b.y1..y2 {
            data.extend_from_slice(&self.data[(y * self.width + b.x1) * 3..(y * self.width + x2) * 3]);
        }
        Self::new(x2 - b.x1, y2 - b.y1, data)
    }

    /// The picture with the box blacked out, as MiVOLO cuts a face out of the body it belongs to.
    pub fn without(&self, b: PixelBox) -> Self {
        let mut out = self.clone();
        out.black_out(b);
        out
    }

    /// The box blacked out, in place.
    pub fn black_out(&mut self, b: PixelBox) {
        let (x2, y2) = (b.x2.min(self.width), b.y2.min(self.height));
        if b.x1 >= x2 {
            return;
        }
        for y in b.y1..y2 {
            self.data[(y * self.width + b.x1) * 3..(y * self.width + x2) * 3].fill(0);
        }
    }

    /// OpenCV's `resize` with `INTER_LINEAR`: each output pixel's centre mapped back (half-pixel centres), clamped at
    /// the edges, two neighbours a direction weighted by distance, rounded.
    pub fn resize_linear(&self, width: usize, height: usize) -> Self {
        let taps = |dst: usize, src: usize| -> Vec<(usize, usize, f64)> {
            let scale = src as f64 / dst as f64;
            (0..dst)
                .map(|d| {
                    let f = ((d as f64 + 0.5) * scale - 0.5) as f32 as f64;
                    let mut s = f.floor();
                    let mut frac = f - s;
                    if s < 0. {
                        s = 0.;
                        frac = 0.;
                    }
                    if s as usize >= src - 1 {
                        s = (src - 1) as f64;
                        frac = 0.;
                    }
                    let s = s as usize;
                    (s, (s + 1).min(src - 1), frac)
                })
                .collect()
        };
        let xs = taps(width, self.width);
        let ys = taps(height, self.height);
        let mut data = vec![0u8; width * height * 3];
        for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
            for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
                for c in 0..3 {
                    let at = |yy: usize, xx: usize| self.data[(yy * self.width + xx) * 3 + c] as f64;
                    let top = at(y0, x0) * (1. - fx) + at(y0, x1) * fx;
                    let bottom = at(y1, x0) * (1. - fx) + at(y1, x1) * fx;
                    data[(y * width + x) * 3 + c] = (top * (1. - fy) + bottom * fy).round().clamp(0., 255.) as u8;
                }
            }
        }
        Self { width, height, data }
    }

    /// OpenCV's `resize` with `INTER_AREA`, for a smaller picture: each pixel the mean of the part of the picture it
    /// covers, a pixel cut by its edge counted for the share of it covered. Shrunk by four, the two-tap resize reads two
    /// pixels in four a direction and can miss a thin line or a fine pattern altogether; this reads every one. Row by row,
    /// so it holds no more than the two pictures and one row of sums.
    pub fn resize_area(&self, width: usize, height: usize) -> Self {
        // For each pixel out, the pixels in it covers and the share of its area each is.
        let spans = |dst: usize, src: usize| -> Vec<Vec<(usize, f32)>> {
            let scale = src as f64 / dst as f64;
            (0..dst)
                .map(|d| {
                    let (a, b) = (d as f64 * scale, ((d + 1) as f64 * scale).min(src as f64));
                    (a.floor() as usize..(b.ceil() as usize).min(src))
                        .filter_map(|s| {
                            let cover = b.min(s as f64 + 1.) - a.max(s as f64);
                            (cover > 1e-9).then_some((s, (cover / (b - a)) as f32))
                        })
                        .collect()
                })
                .collect()
        };
        let xs = spans(width, self.width);
        let ys = spans(height, self.height);
        let mut data = vec![0u8; width * height * 3];
        let mut sums = vec![0f32; width * 3];
        for (y, rows) in ys.iter().enumerate() {
            sums.fill(0.);
            for &(sy, wy) in rows {
                let row = &self.data[sy * self.width * 3..(sy + 1) * self.width * 3];
                for (x, cols) in xs.iter().enumerate() {
                    for &(sx, wx) in cols {
                        let w = wx * wy;
                        for c in 0..3 {
                            sums[x * 3 + c] += f32::from(row[sx * 3 + c]) * w;
                        }
                    }
                }
            }
            for (out, sum) in data[y * width * 3..(y + 1) * width * 3].iter_mut().zip(&sums) {
                *out = sum.round().clamp(0., 255.) as u8;
            }
        }
        Self { width, height, data }
    }

    /// MiVOLO's `class_letterbox`: scaled to fit `side` by `side` keeping its shape, centred on black.
    pub fn letterbox(&self, side: usize) -> Self {
        if self.width == side && self.height == side {
            return self.clone();
        }
        let r = (side as f64 / self.height as f64).min(side as f64 / self.width as f64);
        let (nw, nh) = ((self.width as f64 * r).round_ties_even() as usize, (self.height as f64 * r).round_ties_even() as usize);
        let scaled = if (nw, nh) != (self.width, self.height) { self.resize_linear(nw.max(1), nh.max(1)) } else { self.clone() };
        let (dw, dh) = ((side - scaled.width) as f64 / 2., (side - scaled.height) as f64 / 2.);
        let top = (dh - 0.1).round_ties_even() as usize;
        let left = (dw - 0.1).round_ties_even() as usize;
        let mut data = vec![0u8; side * side * 3];
        for y in 0..scaled.height {
            let row = &scaled.data[y * scaled.width * 3..(y + 1) * scaled.width * 3];
            let at = ((top + y) * side + left) * 3;
            data[at..at + row.len()].copy_from_slice(row);
        }
        Self { width: side, height: side, data }
    }

    /// [1, 3, H, W], ImageNet-normalised RGB.
    pub fn normalised(&self, dev: &Device) -> Result<Tensor> {
        const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
        const STD: [f64; 3] = [0.229, 0.224, 0.225];
        let plane = self.width * self.height;
        let mut out = vec![0f32; plane * 3];
        for (i, px) in self.data.chunks_exact(3).enumerate() {
            for c in 0..3 {
                out[c * plane + i] = ((px[c] as f64 / 255. - MEAN[c]) / STD[c]) as f32;
            }
        }
        Tensor::from_vec(out, (1, 3, self.height, self.width), dev)
    }
}

/// MiVOLO's input for no picture: black, normalised.
pub fn absent(side: usize, dev: &Device) -> Result<Tensor> {
    Rgb { width: side, height: side, data: vec![0; side * side * 3] }.normalised(dev)
}
