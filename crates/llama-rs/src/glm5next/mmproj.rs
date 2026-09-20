// VENDORED-LOCAL: whole module. GLM-5.3-Flash mmproj loader.
//! Loads a `projector_type = "glm5next"` mmproj GGUF into the f32 buffers
//! [`super::vision`] operates on.
//!
//! The released mmproj (`mmproj-GLM-5.3-Flash-F16.gguf`, 564M) stores
//! its matrices as **BF16** and its norms, biases and patch embeddings as F32.
//! `ggml_quants::dequantize` handles BF16, so `loader::load_tensor_f32` gives
//! f32 for everything and no bespoke conversion is needed here.
//!
//! Dequantised, the tower is roughly 2.3 GB of f32 — small enough to hold
//! resident, unlike the text model's experts (see
//! [`super::forward::ExpertSource`]).
//!
//! ## Tensor names
//!
//! | name | GGUF dims | dtype |
//! |---|---|---|
//! | `v.patch_embd.weight`, `.weight.1` | `[patch, patch, 3, n_embd]` | F32 |
//! | `v.patch_embd.bias` | `[n_embd]` | F32 |
//! | `v.blk.N.ln1.weight`, `ln2.weight` | `[n_embd]` | F32 |
//! | `v.blk.N.attn_qkv.weight` / `.bias` | `[n_embd, 3*n_embd]` / `[3*n_embd]` | BF16 / F32 |
//! | `v.blk.N.attn_q_norm.weight`, `attn_k_norm.weight` | `[d_head]` | F32 |
//! | `v.blk.N.attn_out.weight` / `.bias` | `[n_embd, n_embd]` / `[n_embd]` | BF16 / F32 |
//! | `v.blk.N.ffn_gate.weight` / `.bias` | `[n_embd, n_ff]` / `[n_ff]` | BF16 / F32 |
//! | `v.blk.N.ffn_up.weight` / `.bias` | `[n_embd, n_ff]` / `[n_ff]` | BF16 / F32 |
//! | `v.blk.N.ffn_down.weight` / `.bias` | `[n_ff, n_embd]` / `[n_embd]` | BF16 / F32 |
//! | `v.post_ln.weight` | `[n_embd]` | F32 |
//! | `mm.patch_merger.weight` / `.bias` | `[m, m, n_embd, proj_dim]` / `[proj_dim]` | F32 |
//! | `mm.post_norm.weight` / `.bias` | `[proj_dim]` | F32 |
//! | `mm.model.fc.weight` | `[proj_dim, proj_dim]` | BF16 |
//! | `mm.gate.weight`, `mm.up.weight` | `[proj_dim, proj_ff]` | BF16 |
//! | `mm.down.weight` | `[proj_ff, proj_dim]` | BF16 |
//!
//! There is deliberately **no** `v.position_embd` and no post-convolution
//! embedding norm — `clip_graph_glm5next::build()` asserts both are absent.
//! Their absence is checked here so a GLM-4V mmproj cannot be loaded by mistake.

use std::path::Path;

use ggml_rs::Tensor;
use gguf::GgufFile;

use super::vision::{VisionShape, VisionW, VitBlockW};
use crate::loader::TensorIndex;
use crate::{LlamaError, Result};

/// `hparams.rope_theta` for glm5next's vision tower. Hardcoded in `clip.cpp`'s
/// `PROJECTOR_TYPE_GLM5NEXT` case rather than carried in the GGUF.
pub const ROPE_THETA: f32 = 10000.0;

struct BlockT {
    ln1: Tensor,
    ln2: Tensor,
    qkv: Tensor,
    qkv_b: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    out: Tensor,
    out_b: Tensor,
    ffn_gate: Tensor,
    ffn_gate_b: Tensor,
    ffn_up: Tensor,
    ffn_up_b: Tensor,
    ffn_down: Tensor,
    ffn_down_b: Tensor,
}

/// An owned, dequantised vision tower. Hand [`Self::view`] to
/// [`super::vision::encode_image_bytes`].
pub struct VisionWeights {
    patch_embd_0: Tensor,
    patch_embd_1: Tensor,
    patch_bias: Tensor,
    blocks: Vec<BlockT>,
    post_ln: Tensor,
    merger: Tensor,
    merger_b: Tensor,
    fc: Tensor,
    post_norm: Tensor,
    post_norm_b: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

fn want(name: &str, got: &[usize], expect: &[usize]) -> Result<()> {
    if got != expect {
        return Err(LlamaError::Config(format!(
            "mmproj: {name} has shape {got:?}, expected {expect:?}"
        )));
    }
    Ok(())
}

impl VisionWeights {
    /// Open an mmproj file and load it.
    pub fn open(path: impl AsRef<Path>) -> Result<(VisionShape, Self)> {
        let g = GgufFile::open(path)?;
        Self::from_gguf(&g)
    }

    pub fn from_gguf(g: &GgufFile) -> Result<(VisionShape, Self)> {
        let projector = g
            .metadata()
            .get("clip.projector_type")
            .or_else(|| g.metadata().get("clip.vision.projector_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if projector != "glm5next" {
            return Err(LlamaError::Config(format!(
                "mmproj: clip.projector_type is {projector:?}, expected \"glm5next\""
            )));
        }

        let u = |k: &str| -> Result<usize> {
            g.get_u64(k)
                .map(|v| v as usize)
                .map_err(|_| LlamaError::Config(format!("mmproj: missing {k}")))
        };
        let uo = |k: &str, d: usize| g.get_u64(k).map(|v| v as usize).unwrap_or(d);

        let n_embd = u("clip.vision.embedding_length")?;
        let n_layer = u("clip.vision.block_count")?;
        let n_head = u("clip.vision.attention.head_count")?;
        let n_ff = u("clip.vision.feed_forward_length")?;
        let patch = u("clip.vision.patch_size")?;
        let n_merge = uo("clip.vision.spatial_merge_size", 2);
        let proj_dim = u("clip.vision.projection_dim")?;
        let swiglu_limit = g
            .get_f32("clip.vision.swiglu_limit")
            .map_err(|_| LlamaError::Config("mmproj: missing clip.vision.swiglu_limit".into()))?;
        let eps = g
            .get_f32("clip.vision.attention.layer_norm_epsilon")
            .unwrap_or(1e-5);

        if n_head == 0 || n_embd % n_head != 0 {
            return Err(LlamaError::Config(format!(
                "mmproj: n_embd {n_embd} is not divisible by n_head {n_head}"
            )));
        }
        if swiglu_limit <= 0.0 {
            return Err(LlamaError::Config(format!(
                "mmproj: swiglu_limit is {swiglu_limit}; glm5next's FFN_SILU_CLAMP needs a \
                 positive limit"
            )));
        }

        let idx = TensorIndex::new(g);

        // The two asserts glm5next's graph makes, checked as load-time errors so
        // a GLM-4V mmproj cannot be mistaken for this one.
        for absent in [
            "v.position_embd.weight",
            "v.pre_ln.weight",
            "v.patch_embd.norm.weight",
        ] {
            if idx.has(absent) {
                return Err(LlamaError::Config(format!(
                    "mmproj: {absent} is present, but glm5next asserts no learned position \
                     embedding and no post-convolution embedding norm"
                )));
            }
        }

        let d_head = n_embd / n_head;
        let kern = 3 * patch * patch;

        let patch_embd_0 = idx.take("v.patch_embd.weight", &[])?;
        want("v.patch_embd.weight", patch_embd_0.shape(), &[n_embd, 3, patch, patch])?;
        let patch_embd_1 = idx.take("v.patch_embd.weight.1", &[]).map_err(|_| {
            LlamaError::Config(
                "mmproj: v.patch_embd.weight.1 is missing; glm5next's patch embedding is a \
                 conv3d split into two temporal halves"
                    .into(),
            )
        })?;
        want("v.patch_embd.weight.1", patch_embd_1.shape(), &[n_embd, 3, patch, patch])?;
        let patch_bias = idx.take("v.patch_embd.bias", &[])?;
        want("v.patch_embd.bias", patch_bias.shape(), &[n_embd])?;
        debug_assert_eq!(patch_embd_0.numel(), n_embd * kern);

        let mut blocks = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let t = |suffix: &str| -> Result<Tensor> {
                idx.take(&format!("v.blk.{il}.{suffix}"), &[])
            };
            let qkv = t("attn_qkv.weight")?;
            want(&format!("v.blk.{il}.attn_qkv.weight"), qkv.shape(), &[3 * n_embd, n_embd])?;
            let ffn_gate = t("ffn_gate.weight")?;
            want(&format!("v.blk.{il}.ffn_gate.weight"), ffn_gate.shape(), &[n_ff, n_embd])?;
            let ffn_down = t("ffn_down.weight")?;
            want(&format!("v.blk.{il}.ffn_down.weight"), ffn_down.shape(), &[n_embd, n_ff])?;
            let q_norm = t("attn_q_norm.weight")?;
            want(&format!("v.blk.{il}.attn_q_norm.weight"), q_norm.shape(), &[d_head])?;

            blocks.push(BlockT {
                ln1: t("ln1.weight")?,
                ln2: t("ln2.weight")?,
                qkv,
                qkv_b: t("attn_qkv.bias")?,
                q_norm,
                k_norm: t("attn_k_norm.weight")?,
                out: t("attn_out.weight")?,
                out_b: t("attn_out.bias")?,
                ffn_gate,
                ffn_gate_b: t("ffn_gate.bias")?,
                ffn_up: t("ffn_up.weight")?,
                ffn_up_b: t("ffn_up.bias")?,
                ffn_down,
                ffn_down_b: t("ffn_down.bias")?,
            });
        }

        let post_ln = idx.take("v.post_ln.weight", &[])?;
        want("v.post_ln.weight", post_ln.shape(), &[n_embd])?;

        let merger = idx.take("mm.patch_merger.weight", &[])?;
        // GGUF [m, m, n_embd, proj_dim] reverses to [proj_dim, n_embd, m, m];
        // vision::merge_window lays its input out to match.
        want(
            "mm.patch_merger.weight",
            merger.shape(),
            &[proj_dim, n_embd, n_merge, n_merge],
        )?;
        let merger_b = idx.take("mm.patch_merger.bias", &[])?;
        want("mm.patch_merger.bias", merger_b.shape(), &[proj_dim])?;

        let fc = idx.take("mm.model.fc.weight", &[])?;
        want("mm.model.fc.weight", fc.shape(), &[proj_dim, proj_dim])?;
        let post_norm = idx.take("mm.post_norm.weight", &[])?;
        want("mm.post_norm.weight", post_norm.shape(), &[proj_dim])?;
        let post_norm_b = idx.take("mm.post_norm.bias", &[])?;
        want("mm.post_norm.bias", post_norm_b.shape(), &[proj_dim])?;

        let gate = idx.take("mm.gate.weight", &[])?;
        if gate.shape().len() != 2 || gate.shape()[1] != proj_dim {
            return Err(LlamaError::Config(format!(
                "mmproj: mm.gate.weight has shape {:?}, expected [proj_ff, {proj_dim}]",
                gate.shape()
            )));
        }
        // The projector FFN width is not in the metadata; take it from the tensor.
        let proj_ff = gate.shape()[0];
        let up = idx.take("mm.up.weight", &[])?;
        want("mm.up.weight", up.shape(), &[proj_ff, proj_dim])?;
        let down = idx.take("mm.down.weight", &[])?;
        want("mm.down.weight", down.shape(), &[proj_dim, proj_ff])?;

        let shape = VisionShape {
            n_embd,
            n_layer,
            n_head,
            n_ff,
            patch,
            n_merge,
            proj_dim,
            proj_ff,
            swiglu_limit,
            eps,
            // The reference hardcodes 1e-5 for the projector's LayerNorm.
            proj_norm_eps: 1e-5,
        };

        Ok((
            shape,
            Self {
                patch_embd_0,
                patch_embd_1,
                patch_bias,
                blocks,
                post_ln,
                merger,
                merger_b,
                fc,
                post_norm,
                post_norm_b,
                gate,
                up,
                down,
            },
        ))
    }

    /// Borrow the buffers as the view [`super::vision`] takes.
    pub fn view(&self) -> VisionW<'_> {
        VisionW {
            patch_embd_0: self.patch_embd_0.data(),
            patch_embd_1: self.patch_embd_1.data(),
            patch_bias: self.patch_bias.data(),
            blocks: self
                .blocks
                .iter()
                .map(|b| VitBlockW {
                    ln1: b.ln1.data(),
                    ln2: b.ln2.data(),
                    qkv: b.qkv.data(),
                    qkv_b: b.qkv_b.data(),
                    q_norm: b.q_norm.data(),
                    k_norm: b.k_norm.data(),
                    out: b.out.data(),
                    out_b: b.out_b.data(),
                    ffn_gate: b.ffn_gate.data(),
                    ffn_gate_b: b.ffn_gate_b.data(),
                    ffn_up: b.ffn_up.data(),
                    ffn_up_b: b.ffn_up_b.data(),
                    ffn_down: b.ffn_down.data(),
                    ffn_down_b: b.ffn_down_b.data(),
                })
                .collect(),
            post_ln: self.post_ln.data(),
            merger: self.merger.data(),
            merger_b: self.merger_b.data(),
            fc: self.fc.data(),
            post_norm: self.post_norm.data(),
            post_norm_b: self.post_norm_b.data(),
            gate: self.gate.data(),
            up: self.up.data(),
            down: self.down.data(),
        }
    }

    /// Resident f32 footprint, for reporting.
    pub fn size_bytes(&self) -> usize {
        let t = |x: &Tensor| x.numel() * 4;
        let mut n = t(&self.patch_embd_0)
            + t(&self.patch_embd_1)
            + t(&self.patch_bias)
            + t(&self.post_ln)
            + t(&self.merger)
            + t(&self.merger_b)
            + t(&self.fc)
            + t(&self.post_norm)
            + t(&self.post_norm_b)
            + t(&self.gate)
            + t(&self.up)
            + t(&self.down);
        for b in &self.blocks {
            n += t(&b.ln1)
                + t(&b.ln2)
                + t(&b.qkv)
                + t(&b.qkv_b)
                + t(&b.q_norm)
                + t(&b.k_norm)
                + t(&b.out)
                + t(&b.out_b)
                + t(&b.ffn_gate)
                + t(&b.ffn_gate_b)
                + t(&b.ffn_up)
                + t(&b.ffn_up_b)
                + t(&b.ffn_down)
                + t(&b.ffn_down_b);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released mmproj, if it is on disk. Gated like the repo's other
    /// real-model tests: run with
    /// `cargo test -p llama-rs glm5next::mmproj -- --ignored --nocapture`.
    // D: is the faster drive; E: holds an identical copy.
    const MMPROJ: &str = r"D:\glm5.3_flash\mmproj-GLM-5.3-Flash-F16.gguf";

    #[test]
    #[ignore = "needs the real mmproj on disk"]
    fn loads_the_released_mmproj_and_encodes_an_image() {
        let (sh, w) = VisionWeights::open(MMPROJ).expect("load mmproj");

        // The released geometry.
        assert_eq!(sh.n_embd, 1024);
        assert_eq!(sh.n_layer, 24);
        assert_eq!(sh.n_head, 16);
        assert_eq!(sh.d_head(), 64);
        assert_eq!(sh.n_ff, 4096);
        assert_eq!(sh.patch, 14);
        assert_eq!(sh.n_merge, 2);
        assert_eq!(sh.proj_dim, 4096);
        assert_eq!(sh.proj_ff, 10240);
        assert_eq!(sh.swiglu_limit, 10.0);
        println!(
            "mmproj: {} blocks, {:.2} GB resident f32",
            sh.n_layer,
            w.size_bytes() as f64 / 1e9
        );

        // A small synthetic image, so the test needs no fixtures.
        let mut raw = image::RgbImage::new(140, 84);
        for (x, y, px) in raw.enumerate_pixels_mut() {
            *px = image::Rgb([
                (x * 7 % 256) as u8,
                (y * 11 % 256) as u8,
                ((x * y) % 256) as u8,
            ]);
        }
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(raw)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");

        let view = w.view();
        let (feats, n_tok) =
            super::super::vision::encode_image_bytes(&sh, &view, &bytes, ROPE_THETA)
                .expect("encode image");

        println!("image -> {n_tok} tokens x {} dims", sh.proj_dim);
        assert_eq!(feats.len(), n_tok * sh.proj_dim);
        assert!(n_tok >= 16, "at least the minimum token count, got {n_tok}");
        assert!(
            feats.iter().all(|x| x.is_finite()),
            "features must be finite"
        );
        assert!(feats.iter().any(|&x| x != 0.0), "features must not be zero");

        let mean = feats.iter().sum::<f32>() / feats.len() as f32;
        let max = feats.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        println!("features: mean {mean:.5}, max |x| {max:.5}");
        assert!(
            max < 1e4,
            "features should not have exploded; max |x| = {max}"
        );
    }
}
