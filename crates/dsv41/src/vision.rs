//! Vision for DeepSeek-V4.1 (reference `image_processor.py`, `vision.py`).
//!
//! An image becomes an `n_vit_h x n_vit_w` grid of 14-pixel patches for the
//! ViT, and the aligner merges each 3x3 block of patch features into one
//! LLM token. The LLM sees the span
//!
//! ```text
//! [IMAGE_START] ([IMAGE] * n_llm_w [IMAGE_NEWLINE]) * n_llm_h [IMAGE_END]
//! ```
//!
//! every position carrying `image_token_id`; the delimiters take learned
//! embeddings and the IMAGE slots the aligner's rows, in reading order.
//!
//! - [`plan_image_grid`], [`prepare`] — sizing, Pillow's resize/pad (via
//!   `oaiy-image`), normalization and patches, bit-exact to the reference
//! - [`VisionTower`] — the ViT and aligner on the CPU, the oracle the CUDA
//!   tower is tested against (and a fallback); `bf16` rounding wherever the
//!   reference's tensors are bf16

use std::path::Path;

use oaiy_engine::{Error, Result};
use oaiy_image::Image;

use crate::config::{Config, VisionConfig};
use crate::formats::{bf16_to_f32, f32_to_bf16, to_bf16};
use crate::linear::load_vec;
use crate::ops::{rmsnorm, silu};
use crate::safetensors::StIndex;

/// Token types, as the reference numbers them.
pub const TEXT: i8 = -1;
pub const IMAGE_START: i8 = 0;
pub const IMAGE: i8 = 1;
pub const IMAGE_NEWLINE: i8 = 2;
pub const IMAGE_END: i8 = 3;

/// The reference normalizes with RMSNorm eps 1e-6 throughout the tower.
const NORM_EPS: f32 = 1e-6;

/// How an image of some size is laid out: its LLM token grid and the pixel
/// size it is resized/padded to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub best_h: usize,
    pub best_w: usize,
}

/// LLM tokens of an `n_llm_h x n_llm_w` grid, delimiters included.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

fn llm_grid(best_h: usize, best_w: usize, cfg: &VisionConfig) -> (usize, usize) {
    let p = cfg.patch_size;
    ((best_h / p).div_ceil(cfg.downsample), (best_w / p).div_ceil(cfg.downsample))
}

/// `solve_resize_ratio`: the largest aspect-preserving pixel size whose
/// token grid fits `max_tokens`, in the reference's float arithmetic.
fn solve_resize_ratio(height: f64, width: f64, cfg: &VisionConfig) -> (usize, usize) {
    let (p, max_n) = (cfg.patch_size, cfg.max_tokens);
    let r = height / width;
    let max_w = ((max_n - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h = max_w * r;
    let cell = p * cfg.downsample;
    if max_w < 1.0 {
        return ((max_n - 2) / 2 * cell, cell); // very tall: one column
    }
    if max_h < 1.0 {
        return (cell, (max_n - 3) * cell); // very wide: one row
    }
    let beta = (max_w.floor() * cell as f64 / width).min(max_h.floor() * cell as f64 / height);
    ((height * beta / p as f64).floor() as usize * p, (width * beta / p as f64).floor() as usize * p)
}

/// The reference `plan_image_grid` (with `safe_resize`) for an image of
/// `width x height` pixels.
pub fn plan_image_grid(width: usize, height: usize, cfg: &VisionConfig) -> Plan {
    let p = cfg.patch_size;
    let (mut w, mut h) = (width as f64, height as f64);
    if let Some(r) = cfg.max_wh_ratio {
        if w > h * r as f64 {
            w = h * r as f64;
        }
    }
    if w * h > 0.0 && w * h < cfg.min_pixels as f64 {
        // Python's `** 0.5` is the C library's pow(); an opaque exponent
        // keeps LLVM from turning this one into a sqrt
        let ratio = (cfg.min_pixels as f64 / (w * h)).powf(std::hint::black_box(0.5));
        w = (w * ratio).trunc();
        h = (h * ratio).trunc();
    }
    let best_w = (w / p as f64).ceil() as usize * p;
    let best_h = (h / p as f64).ceil() as usize * p;
    let (mut n_llm_h, mut n_llm_w) = llm_grid(best_h, best_w, cfg);
    let (mut best_h, mut best_w) = (best_h, best_w);
    if num_image_tokens(n_llm_h, n_llm_w) > cfg.max_tokens {
        (best_h, best_w) = solve_resize_ratio(h, w, cfg);
        (n_llm_h, n_llm_w) = llm_grid(best_h, best_w, cfg);
    }
    Plan { n_llm_h, n_llm_w, best_h, best_w }
}

/// An image ready for the ViT.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// `[n_vit_h * n_vit_w][3 * patch * patch]`, bf16 values; each patch is
    /// channel-major, then rows, then columns.
    pub patches: Vec<f32>,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
}

impl Prepared {
    pub fn n_patches(&self) -> usize {
        self.n_vit_h * self.n_vit_w
    }

    /// LLM tokens the image takes.
    pub fn n_tokens(&self) -> usize {
        num_image_tokens(self.n_llm_h, self.n_llm_w)
    }

    /// The span's token types (`image_token_types`).
    pub fn token_types(&self) -> Vec<i8> {
        let mut t = Vec::with_capacity(self.n_tokens());
        t.push(IMAGE_START);
        for _ in 0..self.n_llm_h {
            t.extend(std::iter::repeat_n(IMAGE, self.n_llm_w));
            t.push(IMAGE_NEWLINE);
        }
        t.push(IMAGE_END);
        t
    }
}

/// The reference `load_image` after decoding: plan, resize/pad onto grey
/// 127 (or a plain resize past `max_wh_ratio`), scale to [-1, 1] as bf16,
/// cut into patches.
pub fn prepare(img: &Image, cfg: &VisionConfig) -> Prepared {
    let p = cfg.patch_size;
    let plan = plan_image_grid(img.width, img.height, cfg);
    let (n_vit_h, n_vit_w) = (plan.best_h / p, plan.best_w / p);
    let sized = match cfg.max_wh_ratio {
        Some(r) if img.width >= r * img.height => oaiy_image::resize::resize_bicubic(img, plan.best_w, plan.best_h),
        _ => oaiy_image::resize::pad(img, plan.best_w, plan.best_h, [127; 3]),
    };
    // (x / 255 - 0.5) / 0.5 in f32, then bf16: one value per byte
    let lut: Vec<f32> = (0..256).map(|v| to_bf16((v as f32 / 255.0 - 0.5) / 0.5)).collect();
    let pp = p * p;
    let mut patches = vec![0.0f32; n_vit_h * n_vit_w * 3 * pp];
    for y in 0..plan.best_h {
        for x in 0..plan.best_w {
            let patch = (y / p) * n_vit_w + x / p;
            let inner = (y % p) * p + x % p;
            let px = &sized.rgb[(y * plan.best_w + x) * 3..][..3];
            for c in 0..3 {
                patches[patch * 3 * pp + c * pp + inner] = lut[px[c] as usize];
            }
        }
    }
    Prepared { patches, n_vit_h, n_vit_w, n_llm_h: plan.n_llm_h, n_llm_w: plan.n_llm_w }
}

/// Decode (PNG or JPEG) and [`prepare`] an encoded image.
pub fn load_image(bytes: &[u8], cfg: &VisionConfig) -> Result<Prepared> {
    Ok(prepare(&oaiy_image::decode(bytes)?, cfg))
}

/// The bytes of an image record (the reference `load_image_bytes`): `data`
/// (base64), `source` (`{data}` in base64 or `{url}`), or `url`: a base64
/// `data:` URL, or — when `local_files` — a local path or `file://` URL.
/// Remote `http(s)` URLs are refused (no TLS in a std-only build); send the
/// image inline instead.
pub fn image_bytes(record: &oaiy_engine::json::Json, local_files: bool) -> Result<Vec<u8>> {
    use oaiy_engine::json::Json;
    if let Some(Json::Str(d)) = record.get("data") {
        return base64_decode(d);
    }
    if let Some(src) = record.get("source").filter(|s| matches!(s, Json::Obj(_))) {
        if let Some(Json::Str(d)) = src.get("data") {
            return base64_decode(d);
        }
        if let Some(Json::Str(u)) = src.get("url") {
            return url_bytes(u, local_files);
        }
    }
    match record.get("url") {
        Some(Json::Str(u)) if !u.is_empty() => url_bytes(u, local_files),
        _ => Err(Error::Arg("image has no data, source or url".into())),
    }
}

fn url_bytes(url: &str, local_files: bool) -> Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, payload) = rest.split_once(',').ok_or_else(|| Error::Arg("malformed data URL".into()))?;
        if !header.split(';').any(|p| p == "base64") {
            return Err(Error::Arg(format!("unsupported data URL encoding: data:{header}")));
        }
        return base64_decode(payload);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(Error::Unsupported("image URLs over the network (send the image as a base64 data: URL)".into()));
    }
    if !local_files {
        return Err(Error::Arg("local image files are not allowed here (send the image as a base64 data: URL)".into()));
    }
    let path = url.strip_prefix("file://").unwrap_or(url);
    // file:///C:/x on Windows
    let path = if cfg!(windows) && path.len() > 2 && path.starts_with('/') && path.as_bytes()[2] == b':' { &path[1..] } else { path };
    Ok(std::fs::read(path)?)
}

/// Base64 (standard or URL-safe alphabet, padding optional, whitespace
/// ignored), as the reference's `base64.b64decode` reads it.
pub fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err(Error::Arg(format!("invalid base64 character {:?}", c as char))),
        };
        acc = acc << 6 | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Expand each image placeholder (`image_token_id`) of a prompt into its
/// image's span of `n_tokens` ids, as `prepare_vl_inputs` does; returns
/// the new ids and where each image's span starts.
pub fn expand_placeholders(ids: &[u32], image_token_id: u32, images: &[Prepared]) -> Result<(Vec<u32>, Vec<usize>)> {
    let found = ids.iter().filter(|&&t| t == image_token_id).count();
    if found != images.len() {
        return Err(Error::Arg(format!("found {found} image tokens but got {} images", images.len())));
    }
    let mut out = Vec::with_capacity(ids.len() + images.iter().map(Prepared::n_tokens).sum::<usize>());
    let mut starts = Vec::with_capacity(images.len());
    let mut next = images.iter();
    for &t in ids {
        if t == image_token_id {
            let img = next.next().expect("counted above");
            starts.push(out.len());
            out.extend(std::iter::repeat_n(image_token_id, img.n_tokens()));
        } else {
            out.push(t);
        }
    }
    Ok((out, starts))
}

/// 2-D rotary tables `[n_h * n_w][rope_dim]` (`get_vision_cos_sin`): the
/// first half of the frequencies turn with the patch row, the second with
/// the column. f32 throughout, as the reference.
pub fn rope_tables(n_h: usize, n_w: usize, rope_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / rope_dim as f32)).collect();
    let (mut cos, mut sin) = (Vec::with_capacity(n_h * n_w * rope_dim), Vec::with_capacity(n_h * n_w * rope_dim));
    for y in 0..n_h {
        for x in 0..n_w {
            for (pos, f) in [(y, &inv), (x, &inv)] {
                for &fr in f {
                    let a = pos as f32 * fr;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
    }
    (cos, sin)
}

/// The aligner's `F.unfold(3, stride=3)` over the `[n_h][n_w][dim]` feature
/// grid, zero-padded to whole 3x3 blocks: `[blocks][dim * 9]`, each row
/// channel-major (`c * 9 + ki * 3 + kj`), blocks in reading order.
pub fn unfold(feats: &[f32], n_h: usize, n_w: usize, dim: usize, r: usize) -> Vec<f32> {
    let (bh, bw) = (n_h.div_ceil(r), n_w.div_ceil(r));
    let rr = r * r;
    let mut out = vec![0.0f32; bh * bw * dim * rr];
    for by in 0..bh {
        for bx in 0..bw {
            let row = &mut out[(by * bw + bx) * dim * rr..][..dim * rr];
            for ki in 0..r {
                for kj in 0..r {
                    let (y, x) = (by * r + ki, bx * r + kj);
                    if y >= n_h || x >= n_w {
                        continue;
                    }
                    let f = &feats[(y * n_w + x) * dim..][..dim];
                    for (c, &v) in f.iter().enumerate() {
                        row[c * rr + ki * r + kj] = v;
                    }
                }
            }
        }
    }
    out
}

/// erf in f64: its Maclaurin series near 0, a continued fraction for erfc
/// in the tails (both far past f32 accuracy).
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    let v = if a < 3.5 {
        let (mut term, mut sum, x2) = (a, a, a * a);
        // at most ~60 terms for |x| < 3.5
        for n in 1..200 {
            let n = n as f64;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(a) = exp(-a^2) / sqrt(pi) * 1 / (a + (1/2) / (a + 1 / (a + (3/2) / (a + ...))))
        let mut f = a;
        for k in (1..60).rev() {
            f = a + (k as f64 / 2.0) / f;
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / f
    };
    v.copysign(x)
}

/// torch's exact `F.gelu`: `x * Φ(x)`.
pub fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

// ---------------------------------------------------------------- CPU tower

/// A bf16 `nn.Linear` held as f32 (exact), `[n][k]` plus an optional bias.
pub struct Linear {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub n: usize,
    pub k: usize,
}

impl Linear {
    fn load(idx: &StIndex, prefix: &str, bias: bool) -> Result<Linear> {
        let name = format!("{prefix}.weight");
        let info = idx.info(&name)?;
        let [n, k] = info.shape[..] else {
            return Err(Error::Format(format!("{name}: expected 2-D, got {:?}", info.shape)));
        };
        let w = idx.read_f32(&name)?;
        let b = if bias { Some(load_vec(idx, &format!("{prefix}.bias"))?) } else { None };
        if w.len() != n * k || b.as_ref().is_some_and(|b| b.len() != n) {
            return Err(Error::Format(format!("{prefix}: size mismatch")));
        }
        Ok(Linear { w, b, n, k })
    }

    /// `bf16(x . W^T + b)` for `t` rows of `x`, threaded over the rows.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (n, k) = (self.n, self.k);
        assert_eq!(x.len(), t * k, "linear: input is not [t, k]");
        let mut y = vec![0.0f32; t * n];
        let threads = oaiy_engine::backend::hardware_concurrency().clamp(1, 64);
        let per = t.div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for (ci, ys) in y.chunks_mut(per * n).enumerate() {
                let xs = &x[ci * per * k..(ci * per * k + ys.len() / n * k)];
                s.spawn(move || {
                    // 32 weight rows at a time against every row of this
                    // thread's share, so the rows stay in cache
                    for o0 in (0..n).step_by(32) {
                        let o1 = (o0 + 32).min(n);
                        for (xr, yr) in xs.chunks_exact(k).zip(ys.chunks_exact_mut(n)) {
                            for o in o0..o1 {
                                let acc = dot(xr, &self.w[o * k..(o + 1) * k]) + self.b.as_ref().map_or(0.0, |b| b[o]);
                                yr[o] = to_bf16(acc);
                            }
                        }
                    }
                });
            }
        });
        y
    }
}

/// Dot product with 16 partial sums (the order a vector unit takes).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

struct Block {
    norm1: Vec<f32>,
    wqkv: Linear,
    wo: Linear,
    norm2: Vec<f32>,
    w1: Linear,
    w2: Linear,
}

/// The ViT, the aligner and the span delimiter embeddings.
pub struct VisionTower {
    pub cfg: VisionConfig,
    patch_embed: Linear,
    blocks: Vec<Block>,
    norm: Vec<f32>,
    aligner: [Linear; 2],
    /// `[dim]` each: the learned `image_start`, `image_newline`, `image_end`.
    pub start: Vec<f32>,
    pub newline: Vec<f32>,
    pub end: Vec<f32>,
}

impl VisionTower {
    /// Load `vision.*`, `aligner.*` and the delimiter embeddings.
    pub fn load(model_dir: &Path, cfg: &Config) -> Result<VisionTower> {
        let vc = cfg.vision.clone().ok_or_else(|| Error::Unsupported("this checkpoint has no vision tower".into()))?;
        let idx = StIndex::open(model_dir)?;
        let blocks = (0..vc.n_layers)
            .map(|i| {
                let p = format!("vision.blocks.{i}");
                Ok(Block {
                    norm1: load_vec(&idx, &format!("{p}.norm1.weight"))?,
                    wqkv: Linear::load(&idx, &format!("{p}.attn.wqkv"), true)?,
                    wo: Linear::load(&idx, &format!("{p}.attn.wo"), true)?,
                    norm2: load_vec(&idx, &format!("{p}.norm2.weight"))?,
                    w1: Linear::load(&idx, &format!("{p}.mlp.w1"), false)?,
                    w2: Linear::load(&idx, &format!("{p}.mlp.w2"), false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(VisionTower {
            patch_embed: Linear::load(&idx, "vision.patch_embed.proj", true)?,
            blocks,
            norm: load_vec(&idx, "vision.norm.weight")?,
            aligner: [Linear::load(&idx, "aligner.w1", true)?, Linear::load(&idx, "aligner.w2", true)?],
            start: load_vec(&idx, "image_start")?,
            newline: load_vec(&idx, "image_newline")?,
            end: load_vec(&idx, "image_end")?,
            cfg: vc,
        })
    }

    /// Patch embedding of a prepared image, `[patches][dim]`.
    pub fn patch_embed(&self, img: &Prepared) -> Vec<f32> {
        self.patch_embed.forward(&img.patches, img.n_patches())
    }

    /// One ViT block over `x` (`[n][dim]`) with the image's rotary tables.
    pub fn block(&self, i: usize, x: &[f32], n: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let b = &self.blocks[i];
        let h = rmsnorm(x, &b.norm1, NORM_EPS);
        let qkv = b.wqkv.forward(&h, n);
        let o = attention(&qkv, n, self.cfg.n_heads, self.cfg.head_dim(), cos, sin);
        let o = b.wo.forward(&o, n);
        let x: Vec<f32> = x.iter().zip(&o).map(|(a, b)| to_bf16(a + b)).collect();
        let h = rmsnorm(&x, &b.norm2, NORM_EPS);
        let gu = b.w1.forward(&h, n);
        let inter = self.cfg.inter_dim;
        let mut m = vec![0.0f32; n * inter];
        for (i, mr) in m.chunks_exact_mut(inter).enumerate() {
            let (g, u) = gu[i * 2 * inter..(i + 1) * 2 * inter].split_at(inter);
            for ((mv, &gv), &uv) in mr.iter_mut().zip(g).zip(u) {
                *mv = to_bf16(to_bf16(silu(gv)) * uv);
            }
        }
        let y = b.w2.forward(&m, n);
        x.iter().zip(&y).map(|(a, b)| to_bf16(a + b)).collect()
    }

    /// The ViT's output features, `[patches][dim]`.
    pub fn vit(&self, img: &Prepared) -> Vec<f32> {
        let n = img.n_patches();
        let (cos, sin) = rope_tables(img.n_vit_h, img.n_vit_w, self.cfg.head_dim() / 2, self.cfg.rope_theta);
        let mut x = self.patch_embed(img);
        for i in 0..self.blocks.len() {
            x = self.block(i, &x, n, &cos, &sin);
        }
        rmsnorm(&x, &self.norm, NORM_EPS)
    }

    /// The aligner over ViT features: `[n_llm_h * n_llm_w][llm dim]`.
    pub fn align(&self, feats: &[f32], img: &Prepared) -> Vec<f32> {
        let u = unfold(feats, img.n_vit_h, img.n_vit_w, self.cfg.dim, self.cfg.downsample);
        let rows = u.len() / (self.cfg.dim * self.cfg.downsample * self.cfg.downsample);
        let mut y = self.aligner[0].forward(&u, rows);
        for v in &mut y {
            *v = to_bf16(gelu(*v));
        }
        self.aligner[1].forward(&y, rows)
    }

    /// The image's aligner rows (the reference `encode_image`).
    pub fn encode(&self, img: &Prepared) -> Vec<f32> {
        self.align(&self.vit(img), img)
    }

    /// The whole span's embeddings, `[n_tokens][llm dim]`: the delimiters'
    /// learned rows around the aligner rows, in reading order.
    pub fn span(&self, img: &Prepared, aligned: &[f32]) -> Vec<f32> {
        span_rows(img, aligned, &self.start, &self.newline, &self.end)
    }
}

/// [`VisionTower::span`] for rows computed elsewhere (the GPU tower).
pub fn span_rows(img: &Prepared, aligned: &[f32], start: &[f32], newline: &[f32], end: &[f32]) -> Vec<f32> {
    let d = start.len();
    let mut out = Vec::with_capacity(img.n_tokens() * d);
    let mut next = aligned.chunks_exact(d);
    for t in img.token_types() {
        match t {
            IMAGE_START => out.extend_from_slice(start),
            IMAGE_NEWLINE => out.extend_from_slice(newline),
            IMAGE_END => out.extend_from_slice(end),
            _ => out.extend_from_slice(next.next().expect("one aligner row per IMAGE slot")),
        }
    }
    out
}

/// Full bidirectional attention with 2-D rotary: `qkv` is `[n][3][heads]
/// [head_dim]` (bf16 values); q and k are rotated (half-split pairs `i`,
/// `i + head_dim/2`) and rounded to bf16, scores scaled by 1/sqrt(head_dim),
/// softmax in f32; the output `[n][heads * head_dim]` is bf16.
pub fn attention(qkv: &[f32], n: usize, heads: usize, hd: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let d = heads * hd;
    let half = hd / 2;
    let rotate = |src: &[f32], i: usize, dst: &mut [f32]| {
        let (c, s) = (&cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
        for j in 0..half {
            let (x1, x2) = (src[j], src[j + half]);
            dst[j] = to_bf16(x1 * c[j] - x2 * s[j]);
            dst[j + half] = to_bf16(x2 * c[j] + x1 * s[j]);
        }
    };
    // per head, contiguous rotated q, k and plain v
    let mut q = vec![0.0f32; heads * n * hd];
    let mut k = vec![0.0f32; heads * n * hd];
    let mut v = vec![0.0f32; heads * n * hd];
    for i in 0..n {
        for h in 0..heads {
            let base = i * 3 * d + h * hd;
            let o = (h * n + i) * hd;
            rotate(&qkv[base..base + hd], i, &mut q[o..o + hd]);
            rotate(&qkv[base + d..base + d + hd], i, &mut k[o..o + hd]);
            v[o..o + hd].copy_from_slice(&qkv[base + 2 * d..base + 2 * d + hd]);
        }
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; n * d];
    let threads = oaiy_engine::backend::hardware_concurrency().clamp(1, 64);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, os) in out.chunks_mut(per * d).enumerate() {
            let (q, k, v) = (&q, &k, &v);
            s.spawn(move || {
                let mut sc = vec![0.0f32; n];
                for (r, orow) in os.chunks_exact_mut(d).enumerate() {
                    let i = ci * per + r;
                    for h in 0..heads {
                        let qi = &q[(h * n + i) * hd..][..hd];
                        let kh = &k[h * n * hd..(h + 1) * n * hd];
                        let mut mx = f32::NEG_INFINITY;
                        for (j, s) in sc.iter_mut().enumerate() {
                            *s = dot(qi, &kh[j * hd..(j + 1) * hd]) * scale;
                            mx = mx.max(*s);
                        }
                        let mut sum = 0.0f32;
                        for s in sc.iter_mut() {
                            *s = (*s - mx).exp();
                            sum += *s;
                        }
                        let o = &mut orow[h * hd..(h + 1) * hd];
                        let vh = &v[h * n * hd..(h + 1) * n * hd];
                        let mut acc = vec![0.0f32; hd];
                        for (j, &p) in sc.iter().enumerate() {
                            for (a, &vv) in acc.iter_mut().zip(&vh[j * hd..(j + 1) * hd]) {
                                *a += p * vv;
                            }
                        }
                        for (ov, a) in o.iter_mut().zip(acc) {
                            *ov = to_bf16(a / sum);
                        }
                    }
                }
            });
        }
    });
    out
}

/// bf16 bit patterns of f32 values (for comparisons with reference dumps).
pub fn bf16_bits(x: &[f32]) -> Vec<u16> {
    x.iter().map(|&v| f32_to_bf16(v)).collect()
}

/// f32 values of bf16 bit patterns.
pub fn from_bf16_bits(x: &[u16]) -> Vec<f32> {
    x.iter().map(|&b| bf16_to_f32(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [(0.0, 0.0), (0.5, 0.520_499_877_813_046_5), (1.0, 0.842_700_792_949_714_9), (2.0, 0.995_322_265_018_952_7), (3.0, 0.999_977_909_503_001_4), (4.0, 0.999_999_984_582_742_1), (6.0, 1.0)] {
            assert!((erf(x) - want).abs() < 1e-12, "erf({x}) = {} not {want}", erf(x));
            assert!((erf(-x) + want).abs() < 1e-12);
        }
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-3.0) + 0.004_049_694).abs() < 1e-8);
    }

    #[test]
    fn unfold_is_channel_major_and_zero_padded() {
        // a 4x4 grid of 2-channel features: [c0 = y*4+x, c1 = 100 + y*4+x]
        let feats: Vec<f32> = (0..16).flat_map(|i| [i as f32, 100.0 + i as f32]).collect();
        let u = unfold(&feats, 4, 4, 2, 3);
        assert_eq!(u.len(), 4 * 18);
        assert_eq!(&u[..9], &[0., 1., 2., 4., 5., 6., 8., 9., 10.]);
        assert_eq!(&u[9..18], &[100., 101., 102., 104., 105., 106., 108., 109., 110.]);
        // the block right of it holds column 3 only
        assert_eq!(&u[18..27], &[3., 0., 0., 7., 0., 0., 11., 0., 0.]);
    }

    #[test]
    fn base64_and_data_urls() {
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ").unwrap(), b"hello world");
        assert_eq!(base64_decode("aGVs\nbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("-_8=").unwrap(), [0xfb, 0xff]);
        assert!(base64_decode("a$b").is_err());
        let rec = oaiy_engine::json::Json::parse(br#"{"type": "image", "url": "data:image/png;base64,aGk="}"#).unwrap();
        assert_eq!(image_bytes(&rec, false).unwrap(), b"hi");
        let rec = oaiy_engine::json::Json::parse(br#"{"type": "image", "source": {"type": "base64", "data": "aGk="}}"#).unwrap();
        assert_eq!(image_bytes(&rec, false).unwrap(), b"hi");
        let rec = oaiy_engine::json::Json::parse(br#"{"type": "image", "url": "C:/nope.png"}"#).unwrap();
        assert!(matches!(image_bytes(&rec, false), Err(Error::Arg(_))));
        let rec = oaiy_engine::json::Json::parse(br#"{"type": "image", "url": "https://example.com/a.png"}"#).unwrap();
        assert!(matches!(image_bytes(&rec, true), Err(Error::Unsupported(_))));
    }

    #[test]
    fn placeholders_expand_to_spans() {
        let p = |h, w| Prepared { patches: Vec::new(), n_vit_h: 3 * h, n_vit_w: 3 * w, n_llm_h: h, n_llm_w: w };
        let (ids, starts) = expand_placeholders(&[5, 9, 6, 9, 7], 9, &[p(1, 1), p(1, 2)]).unwrap();
        assert_eq!(ids, [5, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 7]);
        assert_eq!(starts, [1, 6]);
        assert!(expand_placeholders(&[5, 9], 9, &[]).is_err());
    }

    #[test]
    fn token_types_frame_each_row() {
        let p = Prepared { patches: Vec::new(), n_vit_h: 6, n_vit_w: 9, n_llm_h: 2, n_llm_w: 3 };
        assert_eq!(p.token_types(), vec![0, 1, 1, 1, 2, 1, 1, 1, 2, 3]);
        assert_eq!(p.n_tokens(), 10);
    }
}
