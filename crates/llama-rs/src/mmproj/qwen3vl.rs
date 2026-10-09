//! Qwen3-VL's vision tower and merger: the weights, their loading and checks, the forward pass, and the
//! tower's chain on a device.

use super::*;

/// One pre-LN ViT block as used by the Qwen3-VL merger. Fused QKV reduces
/// three matmuls into one, lowering memory bandwidth at decode time.
#[derive(Debug)]
pub struct Qwen3VlBlock {
    pub ln1_w:        Tensor,
    pub ln1_b:        Tensor,
    pub attn_qkv:     Weight,    // [3*D, D] — Q | K | V stacked along output axis
    pub attn_qkv_b:   Tensor,    // [3*D]
    pub attn_out:     Weight,    // [D, D]
    pub attn_out_b:   Tensor,    // [D]
    pub ln2_w:        Tensor,
    pub ln2_b:        Tensor,
    pub ffn_up:       Weight,    // [ff, D]
    pub ffn_up_b:     Tensor,    // [ff]
    pub ffn_down:     Weight,    // [D, ff]
    pub ffn_down_b:   Tensor,    // [D]
}

/// 2-layer MLP merger head. The 2x2 spatial reshape happens at forward time;
/// these are just the two linear layers + their biases.
#[derive(Debug)]
pub struct Qwen3VlProjector {
    /// `[merge_dim, merge_dim]` where `merge_dim = 4 * vision_dim` (the
    /// 4-patch concatenation). For 1152-dim vision: 4608x4608.
    pub mm0:    Weight,
    pub mm0_b:  Tensor,
    /// `[lm_hidden, merge_dim]`. For Qwen3.6-27B: 5120x4608.
    pub mm2:    Weight,
    pub mm2_b:  Tensor,
}

#[derive(Debug)]
pub struct Qwen3VlMmProj {
    pub config:        MmProjConfig,
    /// `[D, 3, P, P]` patch convolution. If a `.weight.1` temporal slot exists
    /// in the GGUF, it has been summed into this tensor at load time so the
    /// runtime path stays a single conv. (For a static image, summing the
    /// two temporal weights is equivalent to feeding [img, img] through the
    /// 2-temporal-step conv.)
    pub patch_embd:    Tensor,
    pub patch_embd_b:  Tensor,                 // [D]
    pub position_embd: Tensor,                 // [n_patches, D]
    pub blocks:        Vec<Qwen3VlBlock>,
    pub post_ln_w:     Tensor,                 // [D]
    pub post_ln_b:     Tensor,                 // [D]
    pub projector:     Qwen3VlProjector,
    pub backend:       Arc<dyn Backend>,
    /// VENDORED-LOCAL: the tower chained on the backend's device (`chain_mmproj`), made at the first picture; a
    /// loader gives `Default::default()`.
    pub chain:         TowerChain,
}

/// VENDORED-LOCAL: [`Qwen3VlMmProj::chain`]: the tower's weights on the device once a picture has asked for them
/// (None inside where they cannot go there).
#[derive(Default)]
pub struct TowerChain(std::sync::OnceLock<Option<crate::chain_mmproj::Tower>>);

impl TowerChain {
    pub(crate) fn get_or_init(&self, make: impl FnOnce() -> Option<crate::chain_mmproj::Tower>) -> &Option<crate::chain_mmproj::Tower> {
        self.0.get_or_init(make)
    }
}

impl std::fmt::Debug for TowerChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0.get() {
            None => "TowerChain(not yet)",
            Some(None) => "TowerChain(none)",
            Some(Some(_)) => "TowerChain(on the device)",
        })
    }
}

impl Qwen3VlMmProj {
    pub fn from_gguf_with_config(
        g: &GgufFile,
        config: MmProjConfig,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        debug_assert_eq!(config.projector, ProjectorKind::Qwen3Vl);
        let idx = TensorIndex::new(g);

        // ----- patch conv (sum the two temporal slots if both present) ---
        let patch_embd_0 = idx.take("v.patch_embd.weight", &[])?;
        let patch_embd = match idx.try_take("v.patch_embd.weight.1") {
            Some(Ok(t1)) => sum_two_dense(patch_embd_0, t1)?,
            Some(Err(e)) => return Err(e),
            None         => patch_embd_0,
        };
        let patch_embd_b  = idx.take("v.patch_embd.bias",   &[])?;
        let position_embd = idx.take("v.position_embd.weight", &[])?;

        // ----- blocks ----------------------------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(Qwen3VlBlock {
                ln1_w:      idx.take(&format!("v.blk.{i}.ln1.weight"), &[])?,
                ln1_b:      idx.take(&format!("v.blk.{i}.ln1.bias"),   &[])?,
                attn_qkv:   idx.take_weight(&format!("v.blk.{i}.attn_qkv.weight"), &[])?,
                attn_qkv_b: idx.take(&format!("v.blk.{i}.attn_qkv.bias"), &[])?,
                attn_out:   idx.take_weight(&format!("v.blk.{i}.attn_out.weight"), &[])?,
                attn_out_b: idx.take(&format!("v.blk.{i}.attn_out.bias"), &[])?,
                ln2_w:      idx.take(&format!("v.blk.{i}.ln2.weight"), &[])?,
                ln2_b:      idx.take(&format!("v.blk.{i}.ln2.bias"),   &[])?,
                ffn_up:     idx.take_weight(&format!("v.blk.{i}.ffn_up.weight"), &[])?,
                ffn_up_b:   idx.take(&format!("v.blk.{i}.ffn_up.bias"), &[])?,
                ffn_down:   idx.take_weight(&format!("v.blk.{i}.ffn_down.weight"), &[])?,
                ffn_down_b: idx.take(&format!("v.blk.{i}.ffn_down.bias"), &[])?,
            });
        }

        // ----- post-LN + projector -------------------------------------
        let post_ln_w = idx.take("v.post_ln.weight", &[])?;
        let post_ln_b = idx.take("v.post_ln.bias",   &[])?;

        // The projector dims aren't fully fixed by config — `mm.0` is square
        // `[merge_dim, merge_dim]` and `mm.2` is `[lm_hidden, merge_dim]`.
        // We pull them with no shape constraint and just sanity-check them in
        // verify_qwen3vl_shapes below.
        let mm0   = idx.take_weight("mm.0.weight", &[])?;
        let mm0_b = idx.take("mm.0.bias", &[])?;
        let mm2   = idx.take_weight("mm.2.weight", &[])?;
        let mm2_b = idx.take("mm.2.bias", &[])?;
        let projector = Qwen3VlProjector { mm0, mm0_b, mm2, mm2_b };

        verify_qwen3vl_shapes(&config, &patch_embd, &position_embd, &blocks, &projector)?;

        // ----- upload to backend (1 GB margin — vision is one-shot) ----
        const M: usize = 1024 * 1024 * 1024;
        let upload = |t: Tensor| backend.to_device(t);
        let blocks: Vec<Qwen3VlBlock> = blocks.into_iter().map(|b| Qwen3VlBlock {
            ln1_w:      upload(b.ln1_w),
            ln1_b:      upload(b.ln1_b),
            attn_qkv:   b.attn_qkv.try_to_device(&*backend, M),
            attn_qkv_b: upload(b.attn_qkv_b),
            attn_out:   b.attn_out.try_to_device(&*backend, M),
            attn_out_b: upload(b.attn_out_b),
            ln2_w:      upload(b.ln2_w),
            ln2_b:      upload(b.ln2_b),
            ffn_up:     b.ffn_up.try_to_device(&*backend, M),
            ffn_up_b:   upload(b.ffn_up_b),
            ffn_down:   b.ffn_down.try_to_device(&*backend, M),
            ffn_down_b: upload(b.ffn_down_b),
        }).collect();
        let projector = Qwen3VlProjector {
            mm0:   projector.mm0.try_to_device(&*backend, M),
            mm0_b: upload(projector.mm0_b),
            mm2:   projector.mm2.try_to_device(&*backend, M),
            mm2_b: upload(projector.mm2_b),
        };
        let post_ln_w = upload(post_ln_w);
        let post_ln_b = upload(post_ln_b);
        let position_embd = upload(position_embd);
        // patch_embd kept on host until forward time (unfold runs CPU-side).

        Ok(Self {
            config, patch_embd, patch_embd_b: upload(patch_embd_b),
            position_embd, blocks, post_ln_w, post_ln_b, projector, backend,
            chain: Default::default(),
        })
    }

    /// Image → soft tokens. Same contract as the other mmproj variants.
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

        // ----- 1. Patch conv as flattened matmul + bias ------------------
        let patches = unfold_patches_to_host(image, cfg.patch_size)?;
        // VENDORED-LOCAL: the whole tower chained on the backend's device where it can be (the plain matrices are
        // otherwise multiplied on the host by a backend that keeps dense weights there: 20 s a picture).
        if let Some(out) = crate::chain_mmproj::forward(self, &patches) {
            return Ok(out);
        }
        let patches = backend.to_device(patches);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_w_host = self.patch_embd.to_host();
        let patch_w_flat = patch_w_host.reshape(vec![cfg.embedding_dim, in_dim])
            .map_err(|e| LlamaError::Config(format!("patch_embd reshape: {e:?}")))?;
        let patch_w_flat = backend.to_device(patch_w_flat);
        let mut x = backend.linear(&patches, &patch_w_flat);
        backend.add_inplace_broadcast_last(&mut x, &self.patch_embd_b);

        // ----- 2. Add learned absolute positions ------------------------
        backend.add_inplace(&mut x, &self.position_embd);

        // ----- 3. Transformer blocks ------------------------------------
        for blk in &self.blocks {
            x = qwen3vl_block_forward(backend, &x, blk, cfg);
        }

        // ----- 4. Post-LN ----------------------------------------------
        x = backend.layer_norm(&x, &self.post_ln_w, &self.post_ln_b, cfg.layer_norm_eps);

        // ----- 5. 2x2 spatial merge → [N/4, 4*D] -----------------------
        // The patch grid is `side` x `side`; collapse 2x2 neighbourhoods into
        // one row by concatenating their D-vectors in row-major order. Done
        // on host because it's a contiguous shuffle, not a matmul, and the
        // tensor is small (2304 * 1152 = ~2.7M elements).
        let side = cfg.image_size / cfg.patch_size;
        let new_side = side / 2;
        let d = cfg.embedding_dim;
        let merge_dim = 4 * d;
        let n_merged = new_side * new_side;
        let host = x.to_host();
        let src = host.data();
        let mut merged = vec![0.0f32; n_merged * merge_dim];
        for by in 0..new_side {
            for bx in 0..new_side {
                let dst_off = (by * new_side + bx) * merge_dim;
                // Sub-quadrant order: (0,0), (0,1), (1,0), (1,1) — row-major
                // within the 2x2 block. Matches HF's `Qwen3VLPatchMerger`.
                for sy in 0..2 {
                    for sx in 0..2 {
                        let py = by * 2 + sy;
                        let px = bx * 2 + sx;
                        let src_off = (py * side + px) * d;
                        let quad = sy * 2 + sx;
                        let dst_quad_off = dst_off + quad * d;
                        merged[dst_quad_off..dst_quad_off + d]
                            .copy_from_slice(&src[src_off..src_off + d]);
                    }
                }
            }
        }
        let mut h = backend.to_device(Tensor::from_vec(merged, vec![n_merged, merge_dim]));

        // ----- 6. mm.0 → GELU → mm.2 -----------------------------------
        let mut h0 = self.projector.mm0.linear(backend, &h);
        backend.add_inplace_broadcast_last(&mut h0, &self.projector.mm0_b);
        // VENDORED-LOCAL: HF's patch merger uses erf GELU; its ViT blocks
        // use tanh GELU. Evaluate this once per image on host (not per token).
        h = qwen_merger_gelu(backend, &h0);
        let mut out = self.projector.mm2.linear(backend, &h);
        backend.add_inplace_broadcast_last(&mut out, &self.projector.mm2_b);
        Ok(out)
    }
}

fn qwen_merger_gelu(b: &dyn Backend, x: &Tensor) -> Tensor {
    let mut host=x.to_host();
    for v in host.data_mut() {
        // Abramowitz-Stegun 7.1.26: absolute erf error below 1.5e-7.
        let z=*v*std::f32::consts::FRAC_1_SQRT_2;
        let t=1.0/(1.0+0.3275911*z.abs());
        let erf=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*(-z*z).exp();
        *v *= 0.5*(1.0+erf.copysign(z));
    }
    b.to_device(host)
}

/// Pre-LN attention + FFN block for the Qwen3-VL ViT. Fused QKV split into
/// three [N, D] tensors, bidirectional self-attention (no causal mask via
/// `past = n_seq` trick), GeLU MLP.
fn qwen3vl_block_forward(
    b:   &dyn Backend,
    x:   &Tensor,
    blk: &Qwen3VlBlock,
    cfg: &MmProjConfig,
) -> Tensor {
    let n_seq = x.dim(0);
    let n_h   = cfg.n_heads;
    let hd    = cfg.head_dim;
    let d     = cfg.embedding_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    // ----- Pre-LN + fused QKV --------------------------------------------
    let h0 = b.layer_norm(x, &blk.ln1_w, &blk.ln1_b, cfg.layer_norm_eps);
    let mut qkv = blk.attn_qkv.linear(b, &h0);            // [N, 3*D]
    b.add_inplace_broadcast_last(&mut qkv, &blk.attn_qkv_b);

    // Split [N, 3*D] → q, k, v each [N, D] via the backend's GPU-resident
    // 3-way splitter — no host round-trip on CUDA.
    let (q, k, v) = b.split_qkv_3way(&qkv, d);

    let mut q3 = q.reshape(vec![n_seq, n_h, hd]).expect("Q reshape");
    let mut k3 = k.reshape(vec![n_seq, n_h, hd]).expect("K reshape");
    let v3 = v.reshape(vec![n_seq, n_h, hd]).expect("V reshape");

    crate::multimodal_rope::vision(b, &mut q3, cfg.image_size / cfg.patch_size);
    crate::multimodal_rope::vision(b, &mut k3, cfg.image_size / cfg.patch_size);

    // Bidirectional via past = n_seq.
    let attn = b.attention(&q3, &k3, &v3, n_seq, scale, n_seq, None);
    let attn = attn.reshape(vec![n_seq, n_h * hd]).expect("attn reshape");

    let mut out = blk.attn_out.linear(b, &attn);
    b.add_inplace_broadcast_last(&mut out, &blk.attn_out_b);
    b.add_inplace(&mut out, x);

    // ----- Pre-LN + 2-layer GeLU MLP + residual --------------------------
    let h1 = b.layer_norm(&out, &blk.ln2_w, &blk.ln2_b, cfg.layer_norm_eps);
    let mut up = blk.ffn_up.linear(b, &h1);
    b.add_inplace_broadcast_last(&mut up, &blk.ffn_up_b);
    let activated = b.gelu_approx(&up);
    let mut down = blk.ffn_down.linear(b, &activated);
    b.add_inplace_broadcast_last(&mut down, &blk.ffn_down_b);
    b.add_inplace(&mut down, &out);
    down
}

/// Sum two same-shape dense tensors element-wise into a fresh tensor. Used to
/// fold the two temporal-slot patch convs into a single conv at load time.
fn sum_two_dense(a: Tensor, b: Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(LlamaError::Config(format!(
            "patch_embd temporal slot 0 shape {:?} != slot 1 shape {:?}",
            a.shape(), b.shape()
        )));
    }
    let shape = a.shape().to_vec();
    let a_h = a.to_host();
    let b_h = b.to_host();
    let mut out = a_h.data().to_vec();
    for (o, v) in out.iter_mut().zip(b_h.data()) {
        *o += *v;
    }
    Ok(Tensor::from_vec(out, shape))
}

fn verify_qwen3vl_shapes(
    cfg:           &MmProjConfig,
    patch_embd:    &Tensor,
    position_embd: &Tensor,
    blocks:        &[Qwen3VlBlock],
    projector:     &Qwen3VlProjector,
) -> Result<()> {
    let bad = |name: &str, got: &[usize], expected: &[usize]| -> LlamaError {
        LlamaError::BadTensorShape {
            name: name.into(),
            got: got.iter().map(|&v| v as u64).collect(),
            expected: expected.iter().map(|&v| v as u64).collect(),
        }
    };
    let d = cfg.embedding_dim;
    let p = cfg.patch_size;
    let np = cfg.n_patches();

    // Patch conv shape: [D, 3, P, P]
    if patch_embd.shape() != [d, 3, p, p] {
        return Err(bad("v.patch_embd", patch_embd.shape(), &[d, 3, p, p]));
    }
    if position_embd.shape() != [np, d] {
        return Err(bad("v.position_embd", position_embd.shape(), &[np, d]));
    }

    // mm.0: [merge_dim, merge_dim] where merge_dim = 4*D.
    let merge_dim = 4 * d;
    let s0 = projector.mm0.shape();
    if s0.len() != 2 || s0[0] != merge_dim || s0[1] != merge_dim {
        return Err(bad("mm.0.weight", s0, &[merge_dim, merge_dim]));
    }
    if projector.mm0_b.shape() != [merge_dim] {
        return Err(bad("mm.0.bias", projector.mm0_b.shape(), &[merge_dim]));
    }
    // mm.2: [lm_hidden, merge_dim] — lm_hidden unknown here, just check inner.
    let s2 = projector.mm2.shape();
    if s2.len() != 2 || s2[1] != merge_dim {
        return Err(bad("mm.2.weight", s2, &[0, merge_dim]));
    }
    if projector.mm2_b.shape() != [s2[0]] {
        return Err(bad("mm.2.bias", projector.mm2_b.shape(), &[s2[0]]));
    }

    // Block-level shapes are already checked by `idx.take_weight(.., expected)`
    // calls in the loader. This pass just guards against zero-block mmprojs.
    if blocks.len() != cfg.n_layers {
        return Err(LlamaError::Config(format!(
            "loaded {} vision blocks, expected {}", blocks.len(), cfg.n_layers
        )));
    }
    Ok(())
}
