//! oaiy-image: image decoding for OAIY's vision models, std-only.
//!
//! A vision model sees images through the preprocessing it was trained with,
//! and DeepSeek's reference runs Pillow: `Image.open(f).convert("RGB")`, then
//! `ImageOps.pad` with bicubic resampling. This crate reproduces those
//! pixels, checked against Pillow's own output (`tests/golden.rs`):
//!
//! - [`png`] — every colour type, bit depth and interlacing, over our own
//!   [`inflate`]; exact
//! - [`jpeg`] — baseline and progressive, as libjpeg-turbo decodes them
//!   (integer IDCT, fancy upsampling, its colour tables); exact
//! - [`resize`] — Pillow's two-pass fixed-point resampling and
//!   `ImageOps.contain` / `ImageOps.pad`; exact
//!
//! Everything decodes to 8-bit RGB with alpha dropped, as `convert("RGB")`
//! does.

#![forbid(unsafe_code)]

pub mod inflate;
pub mod jpeg;
pub mod png;
pub mod resize;

use oaiy_engine::{Error, Result};

/// Pillow's decompression-bomb limit (twice `Image.MAX_IMAGE_PIXELS`, past
/// which it refuses to open an image).
pub const MAX_PIXELS: usize = 178_956_970;

/// An 8-bit RGB image, rows top to bottom, pixels packed `rgbrgb…`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Image {
    /// A `width` x `height` image of one colour.
    pub fn filled(width: usize, height: usize, color: [u8; 3]) -> Image {
        Image { width, height, rgb: color.repeat(width * height) }
    }
}

/// The formats [`decode`] recognizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
}

/// The format of an encoded image, from its first bytes.
pub fn format(data: &[u8]) -> Option<Format> {
    if data.starts_with(png::SIGNATURE) {
        Some(Format::Png)
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(Format::Jpeg)
    } else {
        None
    }
}

/// Decode a PNG or JPEG file to RGB, as Pillow's
/// `Image.open(f).convert("RGB")` does.
pub fn decode(data: &[u8]) -> Result<Image> {
    match format(data) {
        Some(Format::Png) => png::decode(data),
        Some(Format::Jpeg) => jpeg::decode(data),
        None => {
            let other = if data.starts_with(b"GIF8") {
                "GIF"
            } else if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
                "WebP"
            } else if data.starts_with(b"BM") {
                "BMP"
            } else {
                return Err(Error::Format("not a PNG or JPEG image".into()));
            };
            Err(Error::Unsupported(format!("{other} images (send PNG or JPEG)")))
        }
    }
}
