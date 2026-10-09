// VENDORED-LOCAL: whole module. GLM-5.3-Flash vision tower.
//! The `glm5next` mmproj: a 24-block ViT and the GLM-4V projector, as a host-f32
//! reference.
//!
//! Reference: llama.cpp `clip_graph_glm5next::build()`
//! (`tools/mtmd/models/glm5next-vision.cpp`), which is three asserts and then a
//! call to `clip_graph_glm4v::build()` — so the graph is GLM-4V's, in
//! `tools/mtmd/models/glm4v.cpp`, with the shared ViT loop in
//! `clip_graph::build_vit` (`tools/mtmd/clip.cpp`). PR #27754, pinned at
//! `86ebfef`.
//!
//! ## What glm5next changes versus GLM-4V
//!
//! The asserts are the whole difference:
//!
//!   1. `hparams.ffn_op == FFN_SILU_CLAMP` — a **new op the PR adds**. GLM-4V
//!      uses a plain `swiglu_split`; glm5next clamps at
//!      `clip.vision.swiglu_limit` (10.0), in every ViT block *and* the
//!      projector FFN. See [`swiglu_clamped_pre`].
//!   2. `model.norm_embd_w == nullptr` — no post-convolution embedding norm, so
//!      GLM-4V's `build_norm` after the patch bias is a no-op here. (`build_norm`
//!      still normalises when the weight is null, but glm5next's mmproj carries
//!      no such tensor and the assert makes that explicit.)
//!   3. `model.position_embeddings == nullptr` — **no learned position
//!      embeddings.** Position information reaches the tower only through the
//!      per-block M-RoPE.
//!
//! Also set for glm5next: `rope_theta = 10000`, `n_merge = 2` (from
//! `spatial_merge_size`), bicubic resize, and image tokens clamped to 16..8000.
//!
//! ## The clamp is NOT the text model's clamp
//!
//! This is the trap. Both clamp a SwiGLU at 10.0, but on different sides of the
//! activation:
//!
//! ```text
//! vision (FFN_SILU_CLAMP):  up = clamp(up, -L, L)
//!                           gate = clamp(gate, -inf, L)      // BEFORE silu
//!                           out = silu(gate) * up
//!
//! text (build_ffn):         up = clamp(up, -L, L)
//!                           g = clamp(silu(gate), -inf, L)   // AFTER silu
//!                           out = g * up
//! ```
//!
//! llama.cpp's own callback names say so: `ffn_gate_clamped` in the vision path,
//! `ffn_silu_clamped` in the text path. [`super::forward::swiglu_clamped`] is the
//! text one; do not use it here. [`clamp_order_differs_from_the_text_model`] pins
//! the difference.
//!
//! ## The tower
//!
//! ```text
//! patches = conv(patch_embd_0, img) + conv(patch_embd_1, img) + patch_bias
//! tokens  = reorder_2x2(patches)        // each 2x2 block becomes 4 consecutive
//! for il in 0..24:
//!     cur = rms_norm(x, ln1)
//!     qkv = attn_qkv @ cur + bias       // [3 * n_embd]
//!     Q, K = rms_norm per head (q_norm / k_norm, [d_head])
//!     Q, K = mrope(Q), mrope(K)         // see MRope
//!     x   += attn_out @ softmax(QK^T / sqrt(d_head)) V + bias
//!     cur  = rms_norm(x, ln2)
//!     x   += ffn_down(clamped_swiglu(ffn_gate(cur), ffn_up(cur))) + bias
//! x = rms_norm(x, post_ln)
//! ```
//!
//! Attention is **bidirectional** — `build_vit` is called with no mask.
//! `ln1`/`ln2`/`post_ln` and the q/k norms are **RMSNorm** (`norm_type =
//! NORM_TYPE_RMS`), weight-only, which matches the mmproj carrying no bias for
//! them.
//!
//! The two patch convolutions are summed over the **same** input — a conv3d with
//! temporal patch 2 applied to a single repeated frame, which the converter
//! splits into `v.patch_embd.weight` and `v.patch_embd.weight.1`.
//!
//! ## The projector
//!
//! ```text
//! merge 4 consecutive tokens -> patch_merger (a 2x2 conv = one linear over
//!                               n_merge^2 * n_embd) + bias      -> 4096
//! fc            (mm.model.fc)                                   -> 4096
//! LayerNorm     (mm.post_norm, WITH bias, eps 1e-5, not RMS)
//! gelu_erf
//! clamped SwiGLU (mm.gate / mm.up -> 10240, mm.down -> 4096)
//! ```
//!
//! Note the order: the norm is *after* the fc, and a GELU sits between the norm
//! and the FFN.
//!
//! ## Positional information: M-RoPE
//!
//! `build_vit`'s `add_pos` hook rotates Q and K in **every** block with
//! `ggml_rope_multi(..., n_dims = d_head/2, sections = {d_head/4 x 4},
//! GGML_ROPE_TYPE_VISION, freq_base = 10000, ...)`, reading a `[n_patches * 4]`
//! position tensor. This stack had only `RopeType::Normal` and `RopeType::NeoX`
//! (`ggml-rs/src/cpu.rs`), so [`MRope`] is a fresh implementation, transcribed
//! from `ggml_mrope_cache_init` and `rotate_pairs`. Its docs record what the
//! implementation does where the ggml header's worked example disagrees.
//!
//! [`VitRope`] keeps the seam so a tower can be run without it; [`NoRope`] is
//! that stand-in and is **positionally blind**, for isolating the other stages in
//! tests. Real use wants [`MRope::for_grid`].
//!
//! **Wired into `Glm5NextModel`: no.** The remaining gap for end-to-end vision is
//! preprocessing — resize to a multiple of `patch * n_merge` (bicubic) and
//! normalise by `clip.vision.image_mean` / `image_std`. `oaiy-image` already has
//! Pillow-exact resize, and `vision::preprocess_image` wraps it.

use crate::{LlamaError, Result};

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Abramowitz & Stegun 7.1.26: `|error| < 1.5e-7`, which is below f32 rounding
/// over the range GELU cares about. ggml calls libm's `erff` here; Rust's std
/// has no `erf`, so this is the stand-in and the source of the only deliberate
/// numerical difference in this module.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// The exact GELU, `0.5 * x * (1 + erf(x / sqrt(2)))` — `ggml_gelu_erf`, not the
/// tanh approximation.
fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// `out = w @ x` for `w` of `[out_dim, in_dim]` row-major, plus an optional bias.
fn matvec(w: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) -> Result<()> {
    let n = x.len();
    if w.len() != out.len() * n {
        return Err(LlamaError::Config(format!(
            "vision: matvec weight is {} for [{}, {n}]",
            w.len(),
            out.len()
        )));
    }
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    if let Some(b) = bias {
        if b.len() != out.len() {
            return Err(LlamaError::Config(format!(
                "vision: bias is {} for a {}-wide output",
                b.len(),
                out.len()
            )));
        }
        for (o, &bb) in out.iter_mut().zip(b) {
            *o += bb;
        }
    }
    Ok(())
}

fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) -> Result<()> {
    if w.len() != x.len() {
        return Err(LlamaError::Config(format!(
            "vision: rms_norm weight is {} for a {}-wide vector",
            w.len(),
            x.len()
        )));
    }
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (v, &s) in x.iter_mut().zip(w) {
        *v = *v * inv * s;
    }
    Ok(())
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) -> Result<()> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for ((v, &s), &bb) in x.iter_mut().zip(w).zip(b) {
        *v = (*v - mean) * inv * s + bb;
    }
    Ok(())
}

/// The vision tower's `FFN_SILU_CLAMP`: **clamp before the activation**.
///
/// ```text
/// up   = clamp(up,   -L,   L)
/// gate = clamp(gate, -inf, L)
/// out  = silu(gate) * up
/// ```
///
/// This is not [`super::forward::swiglu_clamped`], which clamps `silu(gate)`
/// *after* the activation for the text FFN. See the module docs.
pub fn swiglu_clamped_pre(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || out.len() != gate.len() {
        return Err(LlamaError::Config(format!(
            "vision: swiglu widths {} / {} / {}",
            gate.len(),
            up.len(),
            out.len()
        )));
    }
    if !(limit > 0.0) {
        return Err(LlamaError::Config(
            "vision: FFN_SILU_CLAMP needs a positive swiglu_limit".into(),
        ));
    }
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        let gc = if g > limit { limit } else { g };
        let uc = u.clamp(-limit, limit);
        *o = silu(gc) * uc;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// positional rotation seam
// ---------------------------------------------------------------------------

/// Applies the ViT's positional rotation to one token's `[n_head * d_head]` Q or
/// K buffer, in place.
///
/// glm5next uses M-RoPE (`ggml_rope_multi`, `GGML_ROPE_TYPE_VISION`), which this
/// stack does not implement yet — see the module docs.
pub trait VitRope {
    fn rotate(&self, buf: &mut [f32], token: usize, n_head: usize, d_head: usize);
}

/// A stand-in that applies no rotation. **Makes the tower positionally blind**:
/// useful to exercise every other stage, not to produce correct features.
pub struct NoRope;

impl VitRope for NoRope {
    fn rotate(&self, _buf: &mut [f32], _token: usize, _n_head: usize, _d_head: usize) {}
}

// ---------------------------------------------------------------------------
// config and weights
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VisionShape {
    /// `clip.vision.embedding_length` — 1024.
    pub n_embd: usize,
    /// `clip.vision.block_count` — 24.
    pub n_layer: usize,
    /// `clip.vision.attention.head_count` — 16.
    pub n_head: usize,
    /// `clip.vision.feed_forward_length` — 4096.
    pub n_ff: usize,
    /// `clip.vision.patch_size` — 14.
    pub patch: usize,
    /// `clip.vision.spatial_merge_size` — 2.
    pub n_merge: usize,
    /// `clip.vision.projection_dim` — 4096.
    pub proj_dim: usize,
    /// The projector FFN's hidden width — 10240 for the released mmproj.
    pub proj_ff: usize,
    /// `clip.vision.swiglu_limit` — 10.0.
    pub swiglu_limit: f32,
    /// `clip.vision.attention.layer_norm_epsilon` — 1e-5. Used for the RMSNorms.
    pub eps: f32,
    /// The projector's LayerNorm epsilon; the reference hardcodes 1e-5.
    pub proj_norm_eps: f32,
}

impl VisionShape {
    pub fn d_head(&self) -> usize {
        self.n_embd / self.n_head
    }
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.d_head() as f32).sqrt()
    }
    /// Tokens out of the projector for a `nx * ny` patch grid.
    pub fn n_out_tokens(&self, nx: usize, ny: usize) -> usize {
        (nx * ny) / (self.n_merge * self.n_merge)
    }
}

pub struct VitBlockW<'a> {
    /// `[n_embd]` each, RMSNorm weights.
    pub ln1: &'a [f32],
    pub ln2: &'a [f32],
    /// `[3 * n_embd, n_embd]` and `[3 * n_embd]`.
    pub qkv: &'a [f32],
    pub qkv_b: &'a [f32],
    /// `[d_head]` each.
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    /// `[n_embd, n_embd]` and `[n_embd]`.
    pub out: &'a [f32],
    pub out_b: &'a [f32],
    /// `[n_ff, n_embd]` and `[n_ff]` each.
    pub ffn_gate: &'a [f32],
    pub ffn_gate_b: &'a [f32],
    pub ffn_up: &'a [f32],
    pub ffn_up_b: &'a [f32],
    /// `[n_embd, n_ff]` and `[n_embd]`.
    pub ffn_down: &'a [f32],
    pub ffn_down_b: &'a [f32],
}

pub struct VisionW<'a> {
    /// `[n_embd, 3, patch, patch]` each — the two temporal halves, summed.
    pub patch_embd_0: &'a [f32],
    pub patch_embd_1: &'a [f32],
    /// `[n_embd]`
    pub patch_bias: &'a [f32],
    pub blocks: Vec<VitBlockW<'a>>,
    /// `[n_embd]`
    pub post_ln: &'a [f32],

    // --- projector ---
    /// `[proj_dim, n_merge * n_merge * n_embd]` and `[proj_dim]`.
    pub merger: &'a [f32],
    pub merger_b: &'a [f32],
    /// `[proj_dim, proj_dim]`
    pub fc: &'a [f32],
    /// `[proj_dim]` each — a LayerNorm, with bias.
    pub post_norm: &'a [f32],
    pub post_norm_b: &'a [f32],
    /// `[proj_ff, proj_dim]` each.
    pub gate: &'a [f32],
    pub up: &'a [f32],
    /// `[proj_dim, proj_ff]`
    pub down: &'a [f32],
}

// ---------------------------------------------------------------------------
// stages
// ---------------------------------------------------------------------------

/// Sequence position of the patch at `(px, py)` after the 2x2 reorder, in
/// tokens.
///
/// GLM-4V's permute/reshape dance rearranges the patch grid so each `n_merge x
/// n_merge` spatial block occupies `n_merge^2` **consecutive** sequence slots,
/// which is what lets the projector's merger read groups of four. Derived from
/// the contiguous order of
/// `cont_3d(permute(reshape_4d(cont_4d(permute(inp)))))`: fastest to slowest the
/// axes are (channel, dx), dy, block-x, block-y.
pub fn reordered_index(px: usize, py: usize, nx: usize, m: usize) -> usize {
    let (bx, dx) = (px / m, px % m);
    let (by, dy) = (py / m, py % m);
    let blocks_x = nx / m;
    ((by * blocks_x + bx) * m + dy) * m + dx
}

/// Patch embedding: two convolutions over the same image, summed, plus the patch
/// bias — then reordered so each 2x2 block is consecutive.
///
/// `img` is `[3, h, w]` f32, already resized and normalised. `h` and `w` must be
/// multiples of `patch * n_merge` (the reference asserts exactly that).
/// Returns `[nx * ny, n_embd]` row-major in sequence order.
pub fn patch_embed(
    sh: &VisionShape,
    w: &VisionW<'_>,
    img: &[f32],
    img_w: usize,
    img_h: usize,
) -> Result<Vec<f32>> {
    let p = sh.patch;
    let m = sh.n_merge;
    if p == 0 || m == 0 {
        return Err(LlamaError::Config("vision: patch and n_merge must be > 0".into()));
    }
    if img_w % (p * m) != 0 || img_h % (p * m) != 0 {
        return Err(LlamaError::Config(format!(
            "vision: {img_w}x{img_h} is not a multiple of patch*n_merge = {}",
            p * m
        )));
    }
    if img.len() != 3 * img_w * img_h {
        return Err(LlamaError::Config(format!(
            "vision: image is {} floats, expected 3*{img_w}*{img_h}",
            img.len()
        )));
    }
    let kern = 3 * p * p;
    if w.patch_embd_0.len() != sh.n_embd * kern || w.patch_embd_1.len() != sh.n_embd * kern {
        return Err(LlamaError::Config(format!(
            "vision: patch_embd is {} / {}, expected n_embd*3*patch^2 = {}",
            w.patch_embd_0.len(),
            w.patch_embd_1.len(),
            sh.n_embd * kern
        )));
    }

    let (nx, ny) = (img_w / p, img_h / p);
    let mut out = vec![0.0f32; nx * ny * sh.n_embd];

    // Stride equals the kernel, so the convolution is one linear map per patch.
    let mut win = vec![0.0f32; kern];
    for py in 0..ny {
        for px in 0..nx {
            for ch in 0..3 {
                for dy in 0..p {
                    let row = (ch * img_h + py * p + dy) * img_w + px * p;
                    let dst = (ch * p + dy) * p;
                    win[dst..dst + p].copy_from_slice(&img[row..row + p]);
                }
            }
            let t = reordered_index(px, py, nx, m);
            let dst = &mut out[t * sh.n_embd..(t + 1) * sh.n_embd];
            for (c, o) in dst.iter_mut().enumerate() {
                let r0 = &w.patch_embd_0[c * kern..(c + 1) * kern];
                let r1 = &w.patch_embd_1[c * kern..(c + 1) * kern];
                // Both temporal halves see the same pixels for a still image.
                let a: f32 = r0.iter().zip(win.iter()).map(|(a, b)| a * b).sum();
                let b: f32 = r1.iter().zip(win.iter()).map(|(a, b)| a * b).sum();
                *o = a + b + w.patch_bias[c];
            }
        }
    }
    Ok(out)
}

/// One ViT block over all tokens, in place. Bidirectional attention — the
/// reference builds the tower with no mask.
pub fn vit_block(
    sh: &VisionShape,
    b: &VitBlockW<'_>,
    rope: &dyn VitRope,
    x: &mut [f32],
    n_tok: usize,
) -> Result<()> {
    let (e, nh, dh) = (sh.n_embd, sh.n_head, sh.d_head());
    if x.len() != n_tok * e {
        return Err(LlamaError::Config(format!(
            "vision: hidden state is {} for {n_tok} tokens of {e}",
            x.len()
        )));
    }

    // --- attention -------------------------------------------------------
    let mut q = vec![0.0f32; n_tok * nh * dh];
    let mut k = vec![0.0f32; n_tok * nh * dh];
    let mut v = vec![0.0f32; n_tok * nh * dh];
    let mut qkv = vec![0.0f32; 3 * e];

    for t in 0..n_tok {
        let mut cur = x[t * e..(t + 1) * e].to_vec();
        rms_norm(&mut cur, b.ln1, sh.eps)?;
        matvec(b.qkv, &cur, Some(b.qkv_b), &mut qkv)?;

        // The fused projection is Q | K | V, each n_head * d_head wide.
        q[t * nh * dh..(t + 1) * nh * dh].copy_from_slice(&qkv[0..e]);
        k[t * nh * dh..(t + 1) * nh * dh].copy_from_slice(&qkv[e..2 * e]);
        v[t * nh * dh..(t + 1) * nh * dh].copy_from_slice(&qkv[2 * e..3 * e]);

        // Per-head RMSNorm, weight only.
        for h in 0..nh {
            let off = t * nh * dh + h * dh;
            rms_norm(&mut q[off..off + dh], b.q_norm, sh.eps)?;
            rms_norm(&mut k[off..off + dh], b.k_norm, sh.eps)?;
        }
        rope.rotate(&mut q[t * nh * dh..(t + 1) * nh * dh], t, nh, dh);
        rope.rotate(&mut k[t * nh * dh..(t + 1) * nh * dh], t, nh, dh);
    }

    let scale = sh.kq_scale();
    let mut ctx = vec![0.0f32; n_tok * e];
    let mut scores = vec![0.0f32; n_tok];
    for t in 0..n_tok {
        for h in 0..nh {
            let qo = t * nh * dh + h * dh;
            let qh = &q[qo..qo + dh];
            let mut max = f32::NEG_INFINITY;
            for (s, sc) in scores.iter_mut().enumerate() {
                let ko = s * nh * dh + h * dh;
                let d: f32 = qh.iter().zip(&k[ko..ko + dh]).map(|(a, b)| a * b).sum();
                *sc = d * scale;
                if *sc > max {
                    max = *sc;
                }
            }
            let mut sum = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - max).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            let co = t * e + h * dh;
            for (s, &sc) in scores.iter().enumerate() {
                let p = sc * inv;
                let vo = s * nh * dh + h * dh;
                for (c, &vv) in ctx[co..co + dh].iter_mut().zip(&v[vo..vo + dh]) {
                    *c += p * vv;
                }
            }
        }
    }

    let mut proj = vec![0.0f32; e];
    for t in 0..n_tok {
        matvec(b.out, &ctx[t * e..(t + 1) * e], Some(b.out_b), &mut proj)?;
        for (dst, &p) in x[t * e..(t + 1) * e].iter_mut().zip(proj.iter()) {
            *dst += p; // residual
        }
    }

    // --- FFN -------------------------------------------------------------
    let mut g = vec![0.0f32; sh.n_ff];
    let mut u = vec![0.0f32; sh.n_ff];
    let mut h = vec![0.0f32; sh.n_ff];
    let mut d = vec![0.0f32; e];
    for t in 0..n_tok {
        let mut cur = x[t * e..(t + 1) * e].to_vec();
        rms_norm(&mut cur, b.ln2, sh.eps)?;
        matvec(b.ffn_gate, &cur, Some(b.ffn_gate_b), &mut g)?;
        matvec(b.ffn_up, &cur, Some(b.ffn_up_b), &mut u)?;
        swiglu_clamped_pre(&g, &u, sh.swiglu_limit, &mut h)?;
        matvec(b.ffn_down, &h, Some(b.ffn_down_b), &mut d)?;
        for (dst, &dv) in x[t * e..(t + 1) * e].iter_mut().zip(d.iter()) {
            *dst += dv; // residual
        }
    }
    Ok(())
}

/// Lay out one merge group's features the way `mm.patch_merger`'s kernel reads
/// them: **channel-major**, `win[ic * group + slot]`.
///
/// `mm.patch_merger.weight` is GGUF `[n_merge, n_merge, n_embd, proj_dim]`, which
/// the loader reverses to `[proj_dim, n_embd, n_merge, n_merge]`. A row's input
/// index is therefore `ic * group + slot`, **not** `slot * n_embd + ic`. The token
/// buffer is slot-major, so the group has to be transposed on the way in.
///
/// Derivation from the reference: `reshape_4d(cur, n_embd, n_merge, n_merge,
/// n_token_out)` then `permute(cur, 2, 0, 1, 3)` puts the two token axes on dims
/// 0 and 1 and the channel on dim 2, and `conv_2d` then contracts
/// `w[kw][kh][ic][oc]` against `in[kw][kh][ic]` with `slot = kw + n_merge * kh`.
///
/// Getting this backwards is invisible on synthetic weights and wrong on real
/// ones, which is what [`the_merger_window_is_channel_major`] pins.
pub fn merge_window(tokens: &[f32], group_idx: usize, group: usize, n_embd: usize, win: &mut [f32]) {
    for slot in 0..group {
        let base = (group_idx * group + slot) * n_embd;
        for c in 0..n_embd {
            win[c * group + slot] = tokens[base + c];
        }
    }
}

/// The GLM-4V projector: merge `n_merge^2` consecutive tokens, then
/// fc -> LayerNorm -> gelu_erf -> clamped SwiGLU.
///
/// `tokens` is `[n_tok, n_embd]`; returns `[n_tok / n_merge^2, proj_dim]`.
pub fn project(sh: &VisionShape, w: &VisionW<'_>, tokens: &[f32]) -> Result<Vec<f32>> {
    let (e, m, pd) = (sh.n_embd, sh.n_merge, sh.proj_dim);
    let group = m * m;
    if tokens.len() % e != 0 {
        return Err(LlamaError::Config(format!(
            "vision: token buffer {} is not a multiple of n_embd {e}",
            tokens.len()
        )));
    }
    let n_tok = tokens.len() / e;
    if n_tok % group != 0 {
        return Err(LlamaError::Config(format!(
            "vision: {n_tok} tokens is not a multiple of n_merge^2 = {group}"
        )));
    }
    if w.merger.len() != pd * group * e {
        return Err(LlamaError::Config(format!(
            "vision: merger is {} , expected proj_dim*n_merge^2*n_embd = {}",
            w.merger.len(),
            pd * group * e
        )));
    }

    let n_out = n_tok / group;
    let mut out = vec![0.0f32; n_out * pd];
    let mut win = vec![0.0f32; group * e];
    let mut merged = vec![0.0f32; pd];
    let mut fced = vec![0.0f32; pd];
    let mut g = vec![0.0f32; sh.proj_ff];
    let mut u = vec![0.0f32; sh.proj_ff];
    let mut h = vec![0.0f32; sh.proj_ff];

    for o in 0..n_out {
        merge_window(tokens, o, group, e, &mut win);
        matvec(w.merger, &win, Some(w.merger_b), &mut merged)?;

        matvec(w.fc, &merged, None, &mut fced)?;
        layer_norm(&mut fced, w.post_norm, w.post_norm_b, sh.proj_norm_eps)?;
        for v in fced.iter_mut() {
            *v = gelu_erf(*v);
        }

        matvec(w.gate, &fced, None, &mut g)?;
        matvec(w.up, &fced, None, &mut u)?;
        swiglu_clamped_pre(&g, &u, sh.swiglu_limit, &mut h)?;
        matvec(w.down, &h, None, &mut out[o * pd..(o + 1) * pd])?;
    }
    Ok(out)
}

/// The whole tower: patches -> 24 ViT blocks -> post_ln -> projector.
///
/// Returns `[n_out_tokens, proj_dim]`, the image's contribution to the text
/// model's embedding sequence.
pub fn encode_image(
    sh: &VisionShape,
    w: &VisionW<'_>,
    rope: &dyn VitRope,
    img: &[f32],
    img_w: usize,
    img_h: usize,
) -> Result<Vec<f32>> {
    if w.blocks.len() != sh.n_layer {
        return Err(LlamaError::Config(format!(
            "vision: {} block weights for a {}-block tower",
            w.blocks.len(),
            sh.n_layer
        )));
    }
    let mut x = patch_embed(sh, w, img, img_w, img_h)?;
    let n_tok = x.len() / sh.n_embd;

    // glm5next asserts there is no post-convolution embedding norm and no
    // learned position embedding, so nothing sits between here and the blocks.
    for b in w.blocks.iter() {
        vit_block(sh, b, rope, &mut x, n_tok)?;
    }
    for t in 0..n_tok {
        rms_norm(&mut x[t * sh.n_embd..(t + 1) * sh.n_embd], w.post_ln, sh.eps)?;
    }
    project(sh, w, &x)
}

// ---------------------------------------------------------------------------
// vision M-RoPE
// ---------------------------------------------------------------------------

/// Per-token `(row, col)` patch coordinates, in sequence order.
///
/// Reference: `clip.cpp`'s `set_input` case shared by `QWEN2VL` / `QWEN3VL` /
/// `GLM4V` (and `GLM5NEXT`, which the PR adds to it), which fills a
/// `[n_pos * 4]` I32 tensor as four planes:
///
/// ```text
/// for y in (0..ph).step_by(m)
///   for x in (0..pw).step_by(m)
///     for dy in 0..2
///       for dx in 0..2
///         plane0[ptr] = y + dy   // row
///         plane1[ptr] = x + dx   // col
///         plane2[ptr] = y + dy   // row again
///         plane3[ptr] = x + dx   // col again
/// ```
///
/// Planes 2 and 3 duplicate 0 and 1, and are unreachable for this geometry
/// anyway (see [`MRope`]). The loop nesting is the same one
/// [`reordered_index`] derives from GLM-4V's permute/reshape dance — two
/// independent derivations of the same token order, which is a useful check on
/// both.
pub fn vision_positions(nx: usize, ny: usize, m: usize) -> Vec<(i32, i32)> {
    let mut out = Vec::with_capacity(nx * ny);
    for by in (0..ny).step_by(m) {
        for bx in (0..nx).step_by(m) {
            for dy in 0..m {
                for dx in 0..m {
                    out.push(((by + dy) as i32, (bx + dx) as i32));
                }
            }
        }
    }
    out
}

/// The ViT's multi-dimensional RoPE, `GGML_ROPE_TYPE_VISION`.
///
/// Reference: `ggml_mrope_cache_init` + `rotate_pairs` in
/// `ggml/src/ggml-cpu/ops.cpp`, as called by GLM-4V with
/// `n_dims = d_head/2`, `sections = {d_head/4, d_head/4, d_head/4, d_head/4}`,
/// `freq_base = 10000`, and YaRN disabled (`ext_factor = 0`, `attn_factor = 1`,
/// `freq_scale = 1`), which reduces `rope_yarn` to a plain `cos`/`sin`.
///
/// ## What the implementation actually does
///
/// The header's worked example for VISION mode (`sections = [y=4, x=4, 0, 0]`,
/// `n_dims = 4`, `--> [yyyyxxxx]`) does not line up with the code for GLM-4V's
/// parameters, and the header itself says other `n_dims` values are
/// "undefined behavior". The code is authoritative, so this follows it:
///
///   * The cache loop runs `i0` over `0..ne0` step 2, so the **sector** index is
///     `i0/2 ∈ 0..d_head/2`. With `sect_dims = 64` for a 64-wide head, `sector`
///     never wraps.
///   * With `sections[0] = 16` and `sec_w = 32`, only the **first two** sections
///     are ever reached: sectors `0..16` take position plane 0 (the row), sectors
///     `16..32` take plane 1 (the column). Planes 2 and 3 are dead for this
///     geometry — which is why `vision_positions` need only produce two values.
///   * `indep_sects` is true for VISION, so each section's theta **resets** at
///     its first sector: `theta = row * ts^i` for `i < 16` and
///     `theta = col * ts^(i-16)` for `i >= 16`, with
///     `ts = freq_base^(-2/n_dims)`.
///   * `rotate_pairs(ne0, n_dims, ...)` pairs element `i` with element
///     `i + n_dims` across the **whole** head, and every element is rotated —
///     which is why the `if (!is_vision)` copy-through of unrotated channels is
///     skipped.
#[derive(Debug, Clone)]
pub struct MRope {
    /// `freq_base^(-2 / n_dims)`.
    theta_scale: f32,
    /// Cos/sin pairs, `d_head / 2`. Pair `i` rotates elements `i` and
    /// `i + n_dims`.
    n_dims: usize,
    /// Pairs taking the row component before the column takes over:
    /// `sections[0] = d_head / 4`.
    sec0: usize,
    /// `(row, col)` per token.
    pos: Vec<(i32, i32)>,
}

impl MRope {
    /// `d_head` must be divisible by 4: `n_dims = d_head/2` pairs, split into two
    /// sections of `d_head/4`.
    pub fn new(d_head: usize, freq_base: f32, pos: Vec<(i32, i32)>) -> Result<Self> {
        if d_head == 0 || d_head % 4 != 0 {
            return Err(LlamaError::Config(format!(
                "vision mrope: d_head {d_head} must be a non-zero multiple of 4"
            )));
        }
        if !(freq_base > 0.0) {
            return Err(LlamaError::Config(
                "vision mrope: freq_base must be positive".into(),
            ));
        }
        let n_dims = d_head / 2;
        Ok(Self {
            theta_scale: freq_base.powf(-2.0 / n_dims as f32),
            n_dims,
            sec0: d_head / 4,
            pos,
        })
    }

    /// Build from an image's patch grid, in the same token order the tower uses.
    pub fn for_grid(d_head: usize, freq_base: f32, nx: usize, ny: usize, m: usize) -> Result<Self> {
        Self::new(d_head, freq_base, vision_positions(nx, ny, m))
    }

    /// The angle pair `i` rotates by for a token at `(row, col)`.
    fn theta(&self, i: usize, row: i32, col: i32) -> f32 {
        let (base, k) = if i < self.sec0 {
            (row, i)
        } else {
            (col, i - self.sec0)
        };
        base as f32 * self.theta_scale.powi(k as i32)
    }
}

impl VitRope for MRope {
    fn rotate(&self, buf: &mut [f32], token: usize, n_head: usize, d_head: usize) {
        // A token with no position, or a mis-sized buffer, is left alone rather
        // than panicking mid-forward; the tower validates shapes up front.
        let Some(&(row, col)) = self.pos.get(token) else {
            return;
        };
        if d_head != 2 * self.n_dims || buf.len() < n_head * d_head {
            return;
        }
        for h in 0..n_head {
            let head = &mut buf[h * d_head..(h + 1) * d_head];
            for i in 0..self.n_dims {
                let theta = self.theta(i, row, col);
                let (s, c) = theta.sin_cos();
                let x0 = head[i];
                let x1 = head[i + self.n_dims];
                // NEOX pairing, forced for MROPE/VISION.
                head[i] = x0 * c - x1 * s;
                head[i + self.n_dims] = x0 * s + x1 * c;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// preprocessing
// ---------------------------------------------------------------------------

/// `clip.vision.image_mean` from the released mmproj (OpenAI CLIP's constants).
pub const IMAGE_MEAN: [f32; 3] = [0.48145467, 0.45782751, 0.40821072];
/// `clip.vision.image_std` from the released mmproj.
pub const IMAGE_STD: [f32; 3] = [0.26862955, 0.26130259, 0.27577710];

/// `hparams.set_limit_image_tokens(16, 8000)` for glm5next.
pub const MIN_IMAGE_TOKENS: usize = 16;
pub const MAX_IMAGE_TOKENS: usize = 8000;

/// Pixels one output token covers: `patch^2 * n_merge^2`. 784 for the released
/// model, which is how `set_limit_image_tokens` turns a token budget into a
/// pixel budget (`clip-model.h`).
pub const fn patch_area(patch: usize, n_merge: usize) -> usize {
    patch * patch * n_merge * n_merge
}

/// Pick the canvas an image is resized to.
///
/// Reference: `mtmd_image_preprocessor_glm5next::smart_resize`
/// (`tools/mtmd/mtmd-image.cpp`, added by PR #27754). Both edges are aligned up
/// to a multiple of `factor = patch * n_merge`, then the area is pulled inside
/// `[min_tokens, max_tokens] * patch_area`.
///
/// The interesting part is the **upper** clamp. Qwen's usual
/// `sqrt(area / max_pixels)` rescale is not monotone once both edges are aligned
/// up, so the reference **binary-searches** the content height instead, keeping
/// the aspect ratio via a float64 divide-then-floor. This reproduces that,
/// including the search bounds and the `low = mid + 1` / `high = mid - 1`
/// updates, so the chosen canvas is the same one.
///
/// Returns `(width, height)`, each a multiple of `factor`. A degenerate input
/// gives `(0, 0)` — reachable through a public API, per the reference's comment.
pub fn smart_resize(
    width: usize,
    height: usize,
    patch: usize,
    n_merge: usize,
    min_tokens: usize,
    max_tokens: usize,
) -> Result<(usize, usize)> {
    let factor = patch * n_merge;
    if factor == 0 {
        return Err(LlamaError::Config(
            "vision: patch * n_merge must be > 0".into(),
        ));
    }
    if min_tokens == 0 || max_tokens < min_tokens {
        return Err(LlamaError::Config(format!(
            "vision: token limits {min_tokens}..{max_tokens} are not a usable range"
        )));
    }
    if width == 0 || height == 0 {
        return Ok((0, 0));
    }

    let area = patch_area(patch, n_merge) as i64;
    let min_pixels = min_tokens as i64 * area;
    let max_pixels = max_tokens as i64 * area;

    let align = |v: i64| -> i64 { ((v + factor as i64 - 1) / factor as i64) * factor as i64 };

    let (w, h) = (width as i64, height as i64);
    let mut ah = align(h);
    let mut aw = align(w);

    if ah * aw < min_pixels {
        let scale = ((min_pixels as f64) / (h as f64 * w as f64)).sqrt();
        ah = align(((h as f64 * scale).ceil() as i64).max(1));
        aw = align(((w as f64 * scale).ceil() as i64).max(1));
    }

    if ah * aw > max_pixels {
        // Aligning both edges is not monotone in the sqrt-area scale, so search.
        let mut low: i64 = 1;
        let mut high: i64 = h;
        ah = factor as i64;
        aw = factor as i64;
        while low <= high {
            let content_h = (low + high) / 2;
            // float64 divide then floor, to stay bit-identical to the reference
            let content_w = ((w as f64 * content_h as f64 / h as f64).floor() as i64).max(1);
            let cand_h = align(content_h);
            let cand_w = align(content_w);
            if cand_h * cand_w <= max_pixels {
                ah = cand_h;
                aw = cand_w;
                low = content_h + 1;
            } else {
                high = content_h - 1;
            }
        }
    }

    Ok((aw as usize, ah as usize))
}

/// How many tokens the text model receives for a canvas of `(w, h)`.
pub fn n_image_tokens(w: usize, h: usize, patch: usize, n_merge: usize) -> usize {
    let a = patch_area(patch, n_merge);
    if a == 0 {
        0
    } else {
        (w * h) / a
    }
}

/// Resize with [`smart_resize`], normalise, and lay out `[3, H, W]` f32 — what
/// [`patch_embed`] consumes.
///
/// Resampling is bicubic (Catmull-Rom). The reference asks for
/// `PILImageResampling.BICUBIC` and notes its own bicubic "only approximates"
/// it, so exact agreement with the Python preprocessor is not achievable on
/// either side; this is the same class of approximation.
///
/// Returns the buffer and the canvas it was resized to.
pub fn preprocess(
    img: &image::DynamicImage,
    sh: &VisionShape,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(Vec<f32>, usize, usize)> {
    let (w, h) = smart_resize(
        img.width() as usize,
        img.height() as usize,
        sh.patch,
        sh.n_merge,
        MIN_IMAGE_TOKENS,
        MAX_IMAGE_TOKENS,
    )?;
    if w == 0 || h == 0 {
        return Err(LlamaError::Config("vision: empty image".into()));
    }
    for (i, &s) in std.iter().enumerate() {
        if s == 0.0 {
            return Err(LlamaError::Config(format!(
                "vision: image_std[{i}] is zero"
            )));
        }
    }

    let resized = img
        .resize_exact(w as u32, h as u32, image::imageops::FilterType::CatmullRom)
        .to_rgb8();

    // CHW, normalised from the 0..1 range.
    let mut out = vec![0.0f32; 3 * w * h];
    for (y, row) in resized.rows().enumerate() {
        for (x, px) in row.enumerate() {
            for c in 0..3 {
                let v = px[c] as f32 / 255.0;
                out[(c * h + y) * w + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok((out, w, h))
}

/// Preprocess an image and run the whole tower, with M-RoPE.
///
/// This is the end-to-end entry point: bytes in, `[n_tokens, proj_dim]`
/// features out, ready to splice into the text model's embedding sequence.
pub fn encode_image_bytes(
    sh: &VisionShape,
    w: &VisionW<'_>,
    bytes: &[u8],
    rope_theta: f32,
) -> Result<(Vec<f32>, usize)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| LlamaError::Config(format!("vision: cannot decode image: {e}")))?;
    let (pixels, iw, ih) = preprocess(&img, sh, IMAGE_MEAN, IMAGE_STD)?;
    let (nx, ny) = (iw / sh.patch, ih / sh.patch);
    let rope = MRope::for_grid(sh.d_head(), rope_theta, nx, ny, sh.n_merge)?;
    let feats = encode_image(sh, w, &rope, &pixels, iw, ih)?;
    Ok((feats, sh.n_out_tokens(nx, ny)))
}

#[cfg(test)]
mod tests;
