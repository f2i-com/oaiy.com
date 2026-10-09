//! Gemma 4's vision tower and projector: the weights, their loading and checks, and the forward pass.

use super::*;

/// One pre-LN+post-LN sandwich block of Gemma 4's vision tower. RMSNorm on
/// both sides of each sublayer, no biases anywhere, SwiGLU FFN.
#[derive(Debug)]
pub struct Gemma4VBlock {
    pub ln1:        Tensor,        // pre-attention RMSNorm scale
    pub attn_q:     Weight,
    pub attn_q_norm: Tensor,       // per-head RMSNorm scale (size head_dim)
    pub attn_k:     Weight,
    pub attn_k_norm: Tensor,
    pub attn_v:     Weight,
    pub attn_out:   Weight,
    pub attn_post:  Tensor,        // post-attention RMSNorm scale
    pub ln2:        Tensor,        // pre-FFN RMSNorm scale
    pub ffn_gate:   Weight,
    pub ffn_up:     Weight,
    pub ffn_down:   Weight,
    pub ffn_post:   Tensor,        // post-FFN RMSNorm scale
}

#[derive(Debug)]
pub struct Gemma4VProjector {
    /// `[lm_hidden, vision_dim]` after our load-time transpose. Same convention
    /// as the Gemma 3 projector.
    pub input_projection: Weight,
}

#[derive(Debug)]
pub struct Gemma4VMmProj {
    pub config:        MmProjConfig,
    pub patch_embd:    Tensor,            // [D, 3, P, P]
    pub position_embd: Tensor,            // [N_max, D] — slice first n_patches
    pub blocks:        Vec<Gemma4VBlock>,
    pub projector:     Gemma4VProjector,
    pub backend:       Arc<dyn Backend>,
}

impl Gemma4VMmProj {
    pub fn from_gguf_with_config(
        g: &GgufFile,
        config: MmProjConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        debug_assert_eq!(config.projector, ProjectorKind::Gemma4V);
        let idx = TensorIndex::new(g);

        // ----- patch + position embed ------------------------------------
        let patch_embd    = idx.take("v.patch_embd.weight",    &[])?;
        let position_embd = idx.take("v.position_embd.weight", &[])?;

        // ----- transformer blocks ---------------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(Gemma4VBlock {
                ln1:         idx.take(&format!("v.blk.{i}.ln1.weight"),            &[])?,
                attn_q:      idx.take_weight(&format!("v.blk.{i}.attn_q.weight"),  &[])?,
                attn_q_norm: idx.take(&format!("v.blk.{i}.attn_q_norm.weight"),    &[])?,
                attn_k:      idx.take_weight(&format!("v.blk.{i}.attn_k.weight"),  &[])?,
                attn_k_norm: idx.take(&format!("v.blk.{i}.attn_k_norm.weight"),    &[])?,
                attn_v:      idx.take_weight(&format!("v.blk.{i}.attn_v.weight"),  &[])?,
                attn_out:    idx.take_weight(&format!("v.blk.{i}.attn_out.weight"),&[])?,
                attn_post:   idx.take(&format!("v.blk.{i}.attn_post_norm.weight"), &[])?,
                ln2:         idx.take(&format!("v.blk.{i}.ln2.weight"),            &[])?,
                ffn_gate:    idx.take_weight(&format!("v.blk.{i}.ffn_gate.weight"),&[])?,
                ffn_up:      idx.take_weight(&format!("v.blk.{i}.ffn_up.weight"),  &[])?,
                ffn_down:    idx.take_weight(&format!("v.blk.{i}.ffn_down.weight"),&[])?,
                ffn_post:    idx.take(&format!("v.blk.{i}.ffn_post_norm.weight"),  &[])?,
            });
        }

        // ----- projector (single linear, NO transpose) ------------------
        // Unlike Gemma 3 (which stores `mm.input_projection` as
        // `[vision_dim, lm_hidden]` and needs transposing), Gemma 4 V already
        // stores the projector in `[lm_hidden, vision_dim]` order — exactly
        // what `linear(x, w) = x @ w^T` expects. Use as-is.
        let projector = Gemma4VProjector {
            input_projection: idx.take_weight("mm.input_projection.weight", &[])?,
        };

        verify_gemma4v_shapes(&config, &patch_embd, &position_embd, &blocks, &projector)?;

        // ----- upload to backend ----------------------------------------
        // Vision is one-shot per image; 1 GB safety margin is plenty.
        const M: usize = 1 * 1024 * 1024 * 1024;
        let upload = |t: Tensor| backend.to_device(t);
        let blocks: Vec<Gemma4VBlock> = blocks.into_iter().map(|b| Gemma4VBlock {
            ln1:         upload(b.ln1),
            attn_q:      b.attn_q.try_to_device(&*backend, M),
            attn_q_norm: upload(b.attn_q_norm),
            attn_k:      b.attn_k.try_to_device(&*backend, M),
            attn_k_norm: upload(b.attn_k_norm),
            attn_v:      b.attn_v.try_to_device(&*backend, M),
            attn_out:    b.attn_out.try_to_device(&*backend, M),
            attn_post:   upload(b.attn_post),
            ln2:         upload(b.ln2),
            ffn_gate:    b.ffn_gate.try_to_device(&*backend, M),
            ffn_up:      b.ffn_up.try_to_device(&*backend, M),
            ffn_down:    b.ffn_down.try_to_device(&*backend, M),
            ffn_post:    upload(b.ffn_post),
        }).collect();
        let projector = Gemma4VProjector {
            input_projection: projector.input_projection.try_to_device(&*backend, M),
        };
        let position_embd = upload(position_embd);
        // patch_embd kept on host until forward time (unfold runs CPU-side).

        Ok(Self { config, patch_embd, position_embd, blocks, projector, backend })
    }

    /// Run image preprocessing → Gemma 4 V tower → projector. Same input/output
    /// contract as [`SigLipMmProj::forward`]: `[3, H, W]` → `[n_soft, lm_hidden]`.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let backend = &*self.backend;
        if image.rank() != 3 || image.dim(0) != 3
            || image.dim(1) != cfg.image_size || image.dim(2) != cfg.image_size
        {
            return Err(LlamaError::Config(format!(
                "image shape {:?} doesn't match expected [3, {sz}, {sz}]",
                image.shape(), sz = cfg.image_size
            )));
        }

        // ----- 1. Patch-conv as flattened matmul ------------------------
        let patches = unfold_patches_to_host(image, cfg.patch_size)?;
        let patches = backend.to_device(patches);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_w_host = self.patch_embd.to_host();
        let patch_w_flat = patch_w_host.reshape(vec![cfg.embedding_dim, in_dim])
            .map_err(|e| LlamaError::Config(format!("patch_embd reshape: {e:?}")))?;
        let patch_w_flat = backend.to_device(patch_w_flat);
        let mut x = backend.linear(&patches, &patch_w_flat);
        // No patch_embd_b — Gemma 4 V conv has no bias.

        // ----- 2. Position embed (axial: row + col) ----------------------
        // The table is `[2, max_pos, D]`. For patch (py, px) at flat index
        // `py*side + px`, we add `pos[0][py] + pos[1][px]`. Build the per-
        // patch slice on host then add as one tensor.
        let pe_host = self.position_embd.to_host();
        let pe_shape = pe_host.shape();
        let max_pos = pe_shape[1];
        let d = cfg.embedding_dim;
        let side = cfg.image_size / cfg.patch_size;
        let np = side * side;
        let pe_data = pe_host.data();
        let row_off = 0;                  // pos[0] starts at index 0
        let col_off = max_pos * d;        // pos[1] starts at max_pos*d
        let mut pos_block = vec![0.0f32; np * d];
        for py in 0..side {
            let row_base = row_off + py * d;
            for px in 0..side {
                let col_base = col_off + px * d;
                let dst = (py * side + px) * d;
                for j in 0..d {
                    pos_block[dst + j] = pe_data[row_base + j] + pe_data[col_base + j];
                }
            }
        }
        let pos_block = backend.to_device(Tensor::from_vec(pos_block, vec![np, d]));
        backend.add_inplace(&mut x, &pos_block);

        // ----- 3. Transformer blocks ------------------------------------
        for blk in &self.blocks {
            x = gemma4v_block_forward(backend, &x, blk, cfg);
        }

        // ----- 4. No global post-LN (per-block sandwich handles it) ------

        // ----- 5. Projector: single linear -----------------------------
        let out = self.projector.input_projection.linear(backend, &x);
        Ok(out)
    }
}

/// One Gemma-style pre-LN+post-LN sandwich block applied to image patches.
///   `x' = x + attn_post(Attn(ln1(x)))`
///   `out = x' + ffn_post(MLP(ln2(x')))`
/// Bidirectional attention via `past = seq` trick (no causal mask).
fn gemma4v_block_forward(
    b:   &dyn Backend,
    x:   &Tensor,
    blk: &Gemma4VBlock,
    cfg: &MmProjConfig,
) -> Tensor {
    let n_seq = x.dim(0);
    let n_h   = cfg.n_heads;
    let hd    = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let eps   = cfg.layer_norm_eps;

    // ----- Pre-attn RMSNorm + Q/K/V proj (no bias) -------------------
    let xn = ggml_rs::ops::rmsnorm(b, x, &blk.ln1, eps);
    let q  = blk.attn_q.linear(b, &xn);
    let k  = blk.attn_k.linear(b, &xn);
    let v  = blk.attn_v.linear(b, &xn);

    let q_3d = q.reshape(vec![n_seq, n_h, hd]).expect("Q reshape");
    let k_3d = k.reshape(vec![n_seq, n_h, hd]).expect("K reshape");
    let v_3d = v.reshape(vec![n_seq, n_h, hd]).expect("V reshape");

    // Per-head Q/K RMSNorm (Gemma 3 does this in the text tower too).
    let q_3d = ggml_rs::ops::rmsnorm(b, &q_3d, &blk.attn_q_norm, eps);
    let k_3d = ggml_rs::ops::rmsnorm(b, &k_3d, &blk.attn_k_norm, eps);

    // Bidirectional attention via past = n_seq (mask never fires).
    let attn = b.attention(&q_3d, &k_3d, &v_3d, n_seq, scale, n_seq, None);
    let attn = attn.reshape(vec![n_seq, n_h * hd]).expect("attn reshape");

    let mut out = blk.attn_out.linear(b, &attn);
    // Post-attn RMSNorm (Gemma sandwich), then residual.
    out = ggml_rs::ops::rmsnorm(b, &out, &blk.attn_post, eps);
    b.add_inplace(&mut out, x);

    // ----- Pre-FFN RMSNorm + SwiGLU + post-FFN norm + residual --------
    let xn2 = ggml_rs::ops::rmsnorm(b, &out, &blk.ln2, eps);
    let gate = blk.ffn_gate.linear(b, &xn2);
    let up   = blk.ffn_up.linear(b, &xn2);
    // Gemma uses approx-GeLU on the gate, then multiplies by up — fused.
    let gated = b.gelu_approx_mul(&gate, &up);
    let mut down = blk.ffn_down.linear(b, &gated);
    down = ggml_rs::ops::rmsnorm(b, &down, &blk.ffn_post, eps);
    b.add_inplace(&mut down, &out);
    down
}

fn verify_gemma4v_shapes(
    cfg:           &MmProjConfig,
    patch_embd:    &Tensor,
    position_embd: &Tensor,
    blocks:        &[Gemma4VBlock],
    projector:     &Gemma4VProjector,
) -> Result<()> {
    let bad = |name: &str, got: &[usize], expected: &[usize]| -> LlamaError {
        LlamaError::BadTensorShape {
            name: name.into(),
            got: got.iter().map(|&v| v as u64).collect(),
            expected: expected.iter().map(|&v| v as u64).collect(),
        }
    };
    let check_dense = |t: &Tensor, expected: &[usize], name: &str| -> Result<()> {
        if t.shape() != expected { Err(bad(name, t.shape(), expected)) } else { Ok(()) }
    };
    let check_w = |w: &Weight, expected: &[usize], name: &str| -> Result<()> {
        if w.shape() != expected { Err(bad(name, w.shape(), expected)) } else { Ok(()) }
    };

    let d  = cfg.embedding_dim;
    let p  = cfg.patch_size;
    let ff = cfg.ff_dim;
    let hd = cfg.head_dim;

    check_dense(patch_embd, &[d, 3, p, p], "v.patch_embd")?;

    // Position embed is `[2, max_positions, D]` — two axial embedding tables
    // (row + column), each indexed by a per-axis position. For a single
    // 224×224 image with patch 16 we use indices 0..14 from each axis. The
    // outer "2" is the row/col split; we just sanity-check the inner dim.
    let pe_shape = position_embd.shape();
    if pe_shape.len() != 3 || pe_shape[0] != 2 || pe_shape[2] != d {
        return Err(bad("v.position_embd", pe_shape, &[2, 0, d]));
    }
    let side = cfg.image_size / cfg.patch_size;
    if pe_shape[1] < side {
        return Err(LlamaError::Config(format!(
            "v.position_embd axial table has only {} positions per axis, need at least {} for {}×{} input",
            pe_shape[1], side, cfg.image_size, cfg.image_size
        )));
    }

    for (i, blk) in blocks.iter().enumerate() {
        check_dense(&blk.ln1,         &[d],     &format!("v.blk.{i}.ln1"))?;
        check_w    (&blk.attn_q,      &[d, d],  &format!("v.blk.{i}.attn_q"))?;
        check_dense(&blk.attn_q_norm, &[hd],    &format!("v.blk.{i}.attn_q_norm"))?;
        check_w    (&blk.attn_k,      &[d, d],  &format!("v.blk.{i}.attn_k"))?;
        check_dense(&blk.attn_k_norm, &[hd],    &format!("v.blk.{i}.attn_k_norm"))?;
        check_w    (&blk.attn_v,      &[d, d],  &format!("v.blk.{i}.attn_v"))?;
        check_w    (&blk.attn_out,    &[d, d],  &format!("v.blk.{i}.attn_out"))?;
        check_dense(&blk.attn_post,   &[d],     &format!("v.blk.{i}.attn_post_norm"))?;
        check_dense(&blk.ln2,         &[d],     &format!("v.blk.{i}.ln2"))?;
        check_w    (&blk.ffn_gate,    &[ff, d], &format!("v.blk.{i}.ffn_gate"))?;
        check_w    (&blk.ffn_up,      &[ff, d], &format!("v.blk.{i}.ffn_up"))?;
        check_w    (&blk.ffn_down,    &[d, ff], &format!("v.blk.{i}.ffn_down"))?;
        check_dense(&blk.ffn_post,    &[d],     &format!("v.blk.{i}.ffn_post_norm"))?;
    }

    // Projector: [lm_hidden, vision_dim] after our transpose. Inner dim = d.
    let s = projector.input_projection.shape();
    if s.len() != 2 || s[1] != d {
        return Err(bad("mm.input_projection.weight", s, &[0, d]));
    }
    Ok(())
}
