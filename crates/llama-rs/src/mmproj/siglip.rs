//! The SigLIP tower with its projector (Gemma 3's pooling one, or an MLP): the weights, their loading and
//! checks, and the forward pass.

use super::*;

/// One transformer block of the SigLIP / CLIP vision tower. Pre-LN
/// architecture: `x + Attn(LN1(x))`; `x + MLP(LN2(x))`. All projections have
/// biases (unlike Llama/Gemma text towers).
#[derive(Debug)]
pub struct VisionBlock {
    pub ln1_w:       Tensor,
    pub ln1_b:       Tensor,
    pub attn_q:      Weight,
    pub attn_q_b:    Tensor,
    pub attn_k:      Weight,
    pub attn_k_b:    Tensor,
    pub attn_v:      Weight,
    pub attn_v_b:    Tensor,
    pub attn_out:    Weight,
    pub attn_out_b:  Tensor,
    pub ln2_w:       Tensor,
    pub ln2_b:       Tensor,
    pub ffn_up:      Weight,
    pub ffn_up_b:    Tensor,
    pub ffn_down:    Weight,
    pub ffn_down_b:  Tensor,
}

/// The Gemma multimodal projector: `avg_pool_4x4(patches) → RMSNorm → linear`.
#[derive(Debug)]
pub struct GemmaProjector {
    /// Per-channel RMSNorm weight applied to the post-pool patches.
    /// Shape `[vision_dim]`.
    pub soft_emb_norm: Tensor,
    /// `[lm_hidden_dim, vision_dim]` linear mapping into the LM token-embedding
    /// space. No bias. Quantized in the GGUF, so we keep it packed.
    pub input_projection: Weight,
}

/// Llava-style 2-layer projector with GeLU between.
#[derive(Debug)]
pub struct MlpProjector {
    pub linear0:   Weight,
    pub linear0_b: Tensor,
    pub linear1:   Weight,
    pub linear1_b: Tensor,
}

#[derive(Debug)]
pub enum Projector {
    Gemma(GemmaProjector),
    Mlp(MlpProjector),
}

/// Vanilla SigLIP/CLIP-style mmproj: pre-LN ViT with biased projections, GeLU
/// MLP, single LayerNorm per sublayer, optional pre-LN, mandatory post-LN.
/// Used by Gemma 3 (with Gemma avg-pool projector) and Llava (with Mlp
/// projector).
#[derive(Debug)]
pub struct SigLipMmProj {
    pub config:        MmProjConfig,
    /// Patch-embedding convolution weight. Stored shape: `[D, 3, P, P]` where
    /// D is `embedding_dim` and P is `patch_size`. Acts as a stride-P conv
    /// from [3, H, W] to [D, H/P, W/P]. We treat it as the matmul
    /// `patches[N, 3*P*P] @ W^T` after unfolding the input.
    pub patch_embd:    Tensor,
    pub patch_embd_b:  Tensor,
    /// Learned position-embedding table. Shape `[n_patches, D]`. Added
    /// element-wise to the patch-embedded sequence before block 0.
    pub position_embd: Tensor,
    /// Optional pre-block layer-norm (CLIP has it, SigLIP does not).
    pub pre_ln:        Option<(Tensor, Tensor)>,
    pub blocks:        Vec<VisionBlock>,
    /// Final post-block layer-norm `(weight, bias)`. Always present.
    pub post_ln:       (Tensor, Tensor),
    pub projector:     Projector,
    pub backend:       Arc<dyn Backend>,
}

impl SigLipMmProj {
    /// Loader for the vanilla SigLIP/CLIP path. Called by
    /// [`MmProj::from_gguf`] after dispatching on `projector_kind`. Loads +
    /// shape-validates every tensor against the parsed config — any mismatch
    /// surfaces as `BadTensorShape` rather than a downstream NaN.
    pub fn from_gguf_with_config(
        g: &GgufFile,
        config: MmProjConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        debug_assert!(config.projector.supports_siglip_loader());
        let idx = TensorIndex::new(g);

        // ----- patch embedding (conv as a flattened matmul) ---------------
        // The HF→GGUF converter writes the conv weight in NCHW order
        // `[D, 3, P, P]`. We load it as F32 since the shape isn't a 2D matrix
        // — backend.linear can't consume it directly. The forward pass will
        // reshape it to `[D, 3*P*P]` before matmul.
        let patch_embd   = idx.take("v.patch_embd.weight", &[])?;
        let patch_embd_b = idx.take("v.patch_embd.bias",   &[])?;
        let position_embd = idx.take("v.position_embd.weight", &[])?;

        // ----- optional pre-LN (CLIP family only) -------------------------
        let pre_ln = match (idx.try_take("v.pre_ln.weight"), idx.try_take("v.pre_ln.bias")) {
            (Some(w), Some(b)) => Some((w?, b?)),
            _ => None,
        };

        // ----- transformer blocks -----------------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(VisionBlock {
                ln1_w:      idx.take(&format!("v.blk.{i}.ln1.weight"),     &[])?,
                ln1_b:      idx.take(&format!("v.blk.{i}.ln1.bias"),       &[])?,
                attn_q:     idx.take_weight(&format!("v.blk.{i}.attn_q.weight"),   &[])?,
                attn_q_b:   idx.take(&format!("v.blk.{i}.attn_q.bias"),    &[])?,
                attn_k:     idx.take_weight(&format!("v.blk.{i}.attn_k.weight"),   &[])?,
                attn_k_b:   idx.take(&format!("v.blk.{i}.attn_k.bias"),    &[])?,
                attn_v:     idx.take_weight(&format!("v.blk.{i}.attn_v.weight"),   &[])?,
                attn_v_b:   idx.take(&format!("v.blk.{i}.attn_v.bias"),    &[])?,
                attn_out:   idx.take_weight(&format!("v.blk.{i}.attn_out.weight"), &[])?,
                attn_out_b: idx.take(&format!("v.blk.{i}.attn_out.bias"),  &[])?,
                ln2_w:      idx.take(&format!("v.blk.{i}.ln2.weight"),     &[])?,
                ln2_b:      idx.take(&format!("v.blk.{i}.ln2.bias"),       &[])?,
                ffn_up:     idx.take_weight(&format!("v.blk.{i}.ffn_up.weight"),   &[])?,
                ffn_up_b:   idx.take(&format!("v.blk.{i}.ffn_up.bias"),    &[])?,
                ffn_down:   idx.take_weight(&format!("v.blk.{i}.ffn_down.weight"), &[])?,
                ffn_down_b: idx.take(&format!("v.blk.{i}.ffn_down.bias"),  &[])?,
            });
        }

        // ----- post-LN (always present) -----------------------------------
        let post_ln = (
            idx.take("v.post_ln.weight", &[])?,
            idx.take("v.post_ln.bias",   &[])?,
        );

        // ----- projector head ---------------------------------------------
        // Note on `mm.input_projection.weight`: the Gemma converter writes
        // this matrix in HuggingFace's `nn.Parameter([vision_dim, lm_hidden])`
        // form (used as `out = in @ W` in the reference implementation),
        // *not* in `nn.Linear`'s `[out, in]` form. Our `linear(x, w) = x @ w^T`
        // expects `[out, in]`, so we transpose at load time. F16 dequantises
        // to F32 before the transpose; the matrix is small (1152×2560 ≈ 12MB)
        // so the host round-trip is cheap.
        let projector = match config.projector {
            ProjectorKind::Gemma3 => {
                let raw = idx.take("mm.input_projection.weight", &[])?;
                Projector::Gemma(GemmaProjector {
                    soft_emb_norm:    idx.take("mm.soft_emb_norm.weight", &[])?,
                    input_projection: Weight::Dense(transpose_2d_host(raw)?),
                })
            }
            ProjectorKind::Mlp => Projector::Mlp(MlpProjector {
                linear0:   idx.take_weight("mm.0.weight", &[])?,
                linear0_b: idx.take("mm.0.bias", &[])?,
                linear1:   idx.take_weight("mm.2.weight", &[])?,
                linear1_b: idx.take("mm.2.bias", &[])?,
            }),
            // Gated by `supports_siglip_loader()` above — unreachable.
            ProjectorKind::Gemma4V => unreachable!("Gemma4V loader not yet implemented"),
            ProjectorKind::Qwen3Vl => unreachable!("Qwen3Vl handled by separate loader (returns Err in MmProj::from_gguf)"),
        };

        verify_shapes(&config, &patch_embd, &patch_embd_b, &position_embd,
                      pre_ln.as_ref(), &blocks, &post_ln, &projector)?;

        // Move everything onto the backend's preferred storage. Norms / biases
        // stay dense; projection matrices use `Weight::try_to_device` which
        // keeps packed quants packed AND falls back to host-resident if the
        // remaining VRAM budget would be breached. The vision tower is one-shot
        // per image, so a host-resident weight just means a slightly slower
        // single-image preprocess — not per-token cost.
        const M: usize = 1 * 1024 * 1024 * 1024;       // 1 GB margin (vision is one-shot)
        let upload = |t: Tensor| backend.to_device(t);
        let blocks: Vec<VisionBlock> = blocks.into_iter().map(|b| VisionBlock {
            ln1_w:      upload(b.ln1_w),
            ln1_b:      upload(b.ln1_b),
            attn_q:     b.attn_q.try_to_device(&*backend, M),
            attn_q_b:   upload(b.attn_q_b),
            attn_k:     b.attn_k.try_to_device(&*backend, M),
            attn_k_b:   upload(b.attn_k_b),
            attn_v:     b.attn_v.try_to_device(&*backend, M),
            attn_v_b:   upload(b.attn_v_b),
            attn_out:   b.attn_out.try_to_device(&*backend, M),
            attn_out_b: upload(b.attn_out_b),
            ln2_w:      upload(b.ln2_w),
            ln2_b:      upload(b.ln2_b),
            ffn_up:     b.ffn_up.try_to_device(&*backend, M),
            ffn_up_b:   upload(b.ffn_up_b),
            ffn_down:   b.ffn_down.try_to_device(&*backend, M),
            ffn_down_b: upload(b.ffn_down_b),
        }).collect();
        let projector = match projector {
            Projector::Gemma(p) => Projector::Gemma(GemmaProjector {
                soft_emb_norm:    upload(p.soft_emb_norm),
                input_projection: p.input_projection.try_to_device(&*backend, M),
            }),
            Projector::Mlp(p) => Projector::Mlp(MlpProjector {
                linear0:   p.linear0.try_to_device(&*backend, M),
                linear0_b: upload(p.linear0_b),
                linear1:   p.linear1.try_to_device(&*backend, M),
                linear1_b: upload(p.linear1_b),
            }),
        };
        let pre_ln = pre_ln.map(|(w, b)| (upload(w), upload(b)));
        let post_ln = (upload(post_ln.0), upload(post_ln.1));
        // `patch_embd` keeps its NCHW shape on host until forward time, since
        // the upcoming unfold step happens CPU-side anyway.

        Ok(Self {
            config, patch_embd, patch_embd_b, position_embd,
            pre_ln, blocks, post_ln, projector, backend,
        })
    }
}

fn verify_shapes(
    cfg:           &MmProjConfig,
    patch_embd:    &Tensor,
    patch_embd_b:  &Tensor,
    position_embd: &Tensor,
    pre_ln:        Option<&(Tensor, Tensor)>,
    blocks:        &[VisionBlock],
    post_ln:       &(Tensor, Tensor),
    projector:     &Projector,
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

    let d = cfg.embedding_dim;
    let p = cfg.patch_size;
    let np = cfg.n_patches();
    let ff = cfg.ff_dim;

    // Patch conv: [D, 3, P, P]. Bias: [D]. Position embed: [N, D].
    check_dense(patch_embd,    &[d, 3, p, p], "v.patch_embd")?;
    check_dense(patch_embd_b,  &[d],          "v.patch_embd.bias")?;
    check_dense(position_embd, &[np, d],      "v.position_embd")?;

    if let Some((w, b)) = pre_ln {
        check_dense(w, &[d], "v.pre_ln.weight")?;
        check_dense(b, &[d], "v.pre_ln.bias")?;
    }
    check_dense(&post_ln.0, &[d], "v.post_ln.weight")?;
    check_dense(&post_ln.1, &[d], "v.post_ln.bias")?;

    for (i, blk) in blocks.iter().enumerate() {
        check_dense(&blk.ln1_w,     &[d],    &format!("v.blk.{i}.ln1.weight"))?;
        check_dense(&blk.ln1_b,     &[d],    &format!("v.blk.{i}.ln1.bias"))?;
        check_w    (&blk.attn_q,    &[d, d], &format!("v.blk.{i}.attn_q.weight"))?;
        check_dense(&blk.attn_q_b,  &[d],    &format!("v.blk.{i}.attn_q.bias"))?;
        check_w    (&blk.attn_k,    &[d, d], &format!("v.blk.{i}.attn_k.weight"))?;
        check_dense(&blk.attn_k_b,  &[d],    &format!("v.blk.{i}.attn_k.bias"))?;
        check_w    (&blk.attn_v,    &[d, d], &format!("v.blk.{i}.attn_v.weight"))?;
        check_dense(&blk.attn_v_b,  &[d],    &format!("v.blk.{i}.attn_v.bias"))?;
        check_w    (&blk.attn_out,  &[d, d], &format!("v.blk.{i}.attn_out.weight"))?;
        check_dense(&blk.attn_out_b,&[d],    &format!("v.blk.{i}.attn_out.bias"))?;
        check_dense(&blk.ln2_w,     &[d],    &format!("v.blk.{i}.ln2.weight"))?;
        check_dense(&blk.ln2_b,     &[d],    &format!("v.blk.{i}.ln2.bias"))?;
        check_w    (&blk.ffn_up,    &[ff, d],&format!("v.blk.{i}.ffn_up.weight"))?;
        check_dense(&blk.ffn_up_b,  &[ff],   &format!("v.blk.{i}.ffn_up.bias"))?;
        check_w    (&blk.ffn_down,  &[d, ff],&format!("v.blk.{i}.ffn_down.weight"))?;
        check_dense(&blk.ffn_down_b,&[d],    &format!("v.blk.{i}.ffn_down.bias"))?;
    }

    match projector {
        Projector::Gemma(p) => {
            check_dense(&p.soft_emb_norm, &[d], "mm.soft_emb_norm.weight")?;
            // [lm_hidden, d] — we don't know lm_hidden here without the LM
            // config, so just sanity-check the inner dim matches.
            let s = p.input_projection.shape();
            if s.len() != 2 || s[1] != d {
                return Err(bad("mm.input_projection.weight", s, &[0, d]));
            }
        }
        Projector::Mlp(p) => {
            // First layer: [hidden, vision_dim], second: [hidden, hidden].
            // Hidden dim is implicit — just check inner dim consistency.
            let s0 = p.linear0.shape();
            if s0.len() != 2 || s0[1] != d {
                return Err(bad("mm.0.weight", s0, &[0, d]));
            }
            let s1 = p.linear1.shape();
            if s1.len() != 2 || s1[0] != s0[0] || s1[1] != s0[0] {
                return Err(bad("mm.2.weight", s1, &[s0[0], s0[0]]));
            }
            check_dense(&p.linear0_b, &[s0[0]], "mm.0.bias")?;
            check_dense(&p.linear1_b, &[s1[0]], "mm.2.bias")?;
        }
    }
    Ok(())
}

impl SigLipMmProj {
    /// Run image preprocessing → ViT → projector. Input is a host tensor
    /// `[3, H, W]` produced by [`crate::preprocess_image`]; output is the
    /// LM-space soft-token tensor `[n_soft_tokens, lm_hidden_dim]` ready to be
    /// spliced into the text token embedding stream.
    ///
    /// Cost: one-shot per image (not per token), so the LayerNorm /
    /// bias-add / patch-unfold paths run on host even when the backend is
    /// CUDA — same pattern as `rmsnorm_no_scale` in its initial port. GPU
    /// fast paths can be added later if vision becomes a hotspot.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        if image.rank() != 3 || image.dim(0) != 3
            || image.dim(1) != cfg.image_size || image.dim(2) != cfg.image_size
        {
            return Err(LlamaError::Config(format!(
                "image shape {:?} doesn't match expected [3, {sz}, {sz}]",
                image.shape(), sz = cfg.image_size
            )));
        }
        let backend = &*self.backend;

        // ----- 1. Patch-conv as flattened matmul -------------------------
        // Unfold the image into [n_patches, 3*P*P] so the conv becomes a
        // standard linear. The conv weight is stored as [D, 3, P, P] in
        // GGUF; reshape to [D, 3*P*P] and then `backend.linear` does the
        // job (linear computes y = x · W^T, so [N, 3PP] @ [D, 3PP]^T = [N, D]).
        let patches = unfold_patches_to_host(image, cfg.patch_size)?;
        let patches = backend.to_device(patches);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_w_host = self.patch_embd.to_host();
        let patch_w_flat = patch_w_host.reshape(vec![cfg.embedding_dim, in_dim])
            .map_err(|e| LlamaError::Config(format!("patch_embd reshape: {e:?}")))?;
        let patch_w_flat = backend.to_device(patch_w_flat);
        let mut x = backend.linear(&patches, &patch_w_flat);
        add_bias_inplace(backend, &mut x, &self.patch_embd_b);

        // ----- 2. Position embedding (additive) --------------------------
        backend.add_inplace(&mut x, &self.position_embd);

        // ----- 3. Optional pre-LN (CLIP only) ----------------------------
        if let Some((w, b)) = &self.pre_ln {
            x = layer_norm(backend, &x, w, b, cfg.layer_norm_eps);
        }

        // ----- 4. Transformer blocks -------------------------------------
        for blk in &self.blocks {
            x = vision_block_forward(backend, &x, blk, cfg);
        }

        // ----- 5. Post-LN ------------------------------------------------
        x = layer_norm(backend, &x, &self.post_ln.0, &self.post_ln.1, cfg.layer_norm_eps);

        // ----- 6. Projector head -----------------------------------------
        let out = match &self.projector {
            Projector::Gemma(p) => gemma_project(backend, &x, p, cfg),
            Projector::Mlp(p)   => mlp_project(backend, &x, p),
        };
        Ok(out)
    }
}

// ============================================================================
// Gemma 4 V mmproj — bespoke vision tower
// ============================================================================
//
// Gemma 4 (E2B / E4B) ships a vision tower that's structurally a Gemma-style
// transformer applied to image patches, not a vanilla SigLIP. Differences
// from SigLIP:
//   * RMSNorm everywhere (no LayerNorm bias terms).
//   * Sandwich norms — both pre and post-norm around attention and FFN
//     (mirrors Gemma 3's text tower's `attn_post_norm` + `ffn_post_norm`).
//   * Per-head Q/K-norm (RMSNorm of size head_dim) before attention scaling.
//   * SwiGLU FFN — `ffn_gate` + `ffn_up` + `ffn_down` (vs vanilla GeLU MLP).
//   * No biases on any projection.
//   * No global post-LN — the final norm is the per-block ffn_post_norm.
//   * Smaller dimensions: SigLIP-base sized (768 dim, 16 blocks, 12 heads).
//   * Multi-tile position embedding (~20480 entries) so the same tower can
//     handle pan-and-scan crops at varying tile counts. For a single
//     224×224 image, we slice the first `n_patches` (=196) positions.
// And the projector:
//   * Just a single `mm.input_projection.weight` linear (no soft_emb_norm,
//     no avg-pool). 1:1 patches → soft tokens.

/// One pre-LN ViT block. Standard pattern:
///   `x' = x + Attn(LN(x))`
///   `out = x' + MLP(LN(x'))`
/// Vision attention is **bidirectional** (no causal mask) — we get that out
/// of the existing causal `attention()` op by setting `past = seq`, which
/// makes the mask predicate `t > past + s` false for every (t, s) in the
/// valid range. Avoids needing a separate non-causal kernel until vision
/// becomes hot enough to warrant one.
fn vision_block_forward(
    b:   &dyn Backend,
    x:   &Tensor,
    blk: &VisionBlock,
    cfg: &MmProjConfig,
) -> Tensor {
    let n_seq = x.dim(0);
    let n_h   = cfg.n_heads;
    let hd    = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    // ----- Pre-attn LN + Q/K/V projections + biases -------------------
    let h0 = layer_norm(b, x, &blk.ln1_w, &blk.ln1_b, cfg.layer_norm_eps);
    let mut q = blk.attn_q.linear(b, &h0); add_bias_inplace(b, &mut q, &blk.attn_q_b);
    let mut k = blk.attn_k.linear(b, &h0); add_bias_inplace(b, &mut k, &blk.attn_k_b);
    let mut v = blk.attn_v.linear(b, &h0); add_bias_inplace(b, &mut v, &blk.attn_v_b);

    let q3 = q.reshape(vec![n_seq, n_h, hd]).expect("Q reshape");
    let k3 = k.reshape(vec![n_seq, n_h, hd]).expect("K reshape");
    let v3 = v.reshape(vec![n_seq, n_h, hd]).expect("V reshape");

    // No RoPE — vision uses learned absolute positions (already added at the
    // start of forward()). past = n_seq disables the causal mask.
    let attn = b.attention(&q3, &k3, &v3, n_seq, scale, n_seq, None);
    let attn = attn.reshape(vec![n_seq, n_h * hd]).expect("attn reshape");

    let mut out = blk.attn_out.linear(b, &attn);
    add_bias_inplace(b, &mut out, &blk.attn_out_b);

    // Residual: out += x
    b.add_inplace(&mut out, x);

    // ----- Pre-FFN LN + MLP + bias + residual -------------------------
    let h1 = layer_norm(b, &out, &blk.ln2_w, &blk.ln2_b, cfg.layer_norm_eps);
    let mut up = blk.ffn_up.linear(b, &h1); add_bias_inplace(b, &mut up, &blk.ffn_up_b);
    let activated = b.gelu_approx(&up);
    let mut down = blk.ffn_down.linear(b, &activated);
    add_bias_inplace(b, &mut down, &blk.ffn_down_b);

    b.add_inplace(&mut down, &out);
    down
}

/// Gemma 3 / 4 multimodal projector: 4×4 spatial avg-pool over the 64×64
/// patch grid (4096 → 256 soft tokens), then RMSNorm with `soft_emb_norm`,
/// then linear to LM hidden dim.
fn gemma_project(
    b:   &dyn Backend,
    x:   &Tensor,
    p:   &GemmaProjector,
    cfg: &MmProjConfig,
) -> Tensor {
    let np = cfg.n_patches();
    let d  = cfg.embedding_dim;
    // Side length of the patch grid (e.g. 64 for SigLIP-Gemma).
    let side = (np as f32).sqrt() as usize;
    debug_assert_eq!(side * side, np, "n_patches {np} isn't a perfect square");
    let pool = 4;
    let new_side = side / pool;          // 16
    let n_pooled = new_side * new_side;  // 256

    // Avg-pool on host: vision is one-shot, and the output (256 × D) is small
    // enough that the d2h round-trip is negligible.
    let host = x.to_host();
    let data = host.data();
    let mut pooled = vec![0.0f32; n_pooled * d];
    let inv = 1.0 / ((pool * pool) as f32);
    for big_y in 0..new_side {
        for big_x in 0..new_side {
            let out_off = (big_y * new_side + big_x) * d;
            for dy in 0..pool {
                for dx in 0..pool {
                    let ny = big_y * pool + dy;
                    let nx = big_x * pool + dx;
                    let in_off = (ny * side + nx) * d;
                    for j in 0..d {
                        pooled[out_off + j] += data[in_off + j];
                    }
                }
            }
            for j in 0..d {
                pooled[out_off + j] *= inv;
            }
        }
    }
    let pooled = b.to_device(Tensor::from_vec(pooled, vec![n_pooled, d]));

    // RMSNorm with the per-channel `soft_emb_norm` weight. Gemma uses the
    // same eps here as the LM's RMSNorm (1e-6 in practice).
    let normed = b.rmsnorm(&pooled, &p.soft_emb_norm, 1e-6);

    // Project into LM hidden dim. Output: [n_pooled, lm_hidden].
    p.input_projection.linear(b, &normed)
}

/// Llava-style MLP projector: linear → GeLU → linear, both with bias.
fn mlp_project(b: &dyn Backend, x: &Tensor, p: &MlpProjector) -> Tensor {
    let mut h = p.linear0.linear(b, x);
    add_bias_inplace(b, &mut h, &p.linear0_b);
    let activated = b.gelu_approx(&h);
    let mut out = p.linear1.linear(b, &activated);
    add_bias_inplace(b, &mut out, &p.linear1_b);
    out
}

// ============================================================================
// Qwen3-VL mmproj — fused-QKV ViT + 2x2 spatial-merge MLP projector
// ============================================================================
//
// Qwen3-VL ships its vision tower as a "merger" projector. Differences from
// SigLIP:
//   * Fused `attn_qkv.weight` (with bias) instead of separate q/k/v.
//   * Pre-LN with full LayerNorm (mean + var, with bias) — same as SigLIP/CLIP.
//   * GeLU MLP with biases on `ffn_up` and `ffn_down`. NO `ffn_gate` (Qwen3-VL
//     uses a vanilla 2-layer MLP, not SwiGLU).
//   * Post-LN at the end of the tower.
//   * Temporal patch embed: TWO `v.patch_embd.weight` tensors named `.weight`
//     and `.weight.1`. For static images (single time step) these are summed at
//     load time — equivalent to feeding the same image at both temporal slots.
//   * 2x2 spatial merge collapses 4 neighbouring patches into one 4*D row, then
//     a 2-layer MLP (`mm.0` → GELU → `mm.2`) projects into LM hidden dim.
// And optional DeepStack (when `clip.vision.is_deepstack_layers` flags any
// block AND per-block `deepstack_norm`/`fc1`/`fc2` tensors are present): each
// flagged block's output is also extracted, normed, projected, and concatenated
// as additional context tokens. The current Q3.6-27B mmproj declares the bool
// array but ships no such per-block tensors → DeepStack is inactive for now;
// loader detects this and emits warning rather than failing.
