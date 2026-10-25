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

/// One of Qwen3-VL's DeepStack mergers (`v.deepstack.{block}.*`): the states after vision block `block`, each 2 x 2
/// patches side by side, layer-normed (over the four together), then `fc1` → erf-GELU → `fc2` to the text model's
/// width. Its tokens are added to the text model's hidden states at the picture's rows after one of its first layers
/// (the first merger's after layer 0, the next's after layer 1, …), as transformers' `_deepstack_process` adds them.
#[derive(Debug)]
pub struct Qwen3VlDeepstack {
    pub block:  usize,
    pub norm_w: Tensor,  // [4 D]
    pub norm_b: Tensor,  // [4 D]
    pub fc1:    Weight,  // [4 D, 4 D]
    pub fc1_b:  Tensor,
    pub fc2:    Weight,  // [lm_hidden, 4 D]
    pub fc2_b:  Tensor,
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
    /// Qwen3-VL's DeepStack mergers, by the vision block they read (none in Qwen3.5's tower, which has no DeepStack).
    pub deepstack:     Vec<Qwen3VlDeepstack>,
    pub backend:       Arc<dyn Backend>,
    /// the tower chained on the backend's device (`chain::mmproj`), made at the first picture; a
    /// loader gives `Default::default()`.
    pub chain:         TowerChain,
}

/// [`Qwen3VlMmProj::chain`]: the tower's weights on the device once a picture has asked for them
/// (None inside where they cannot go there), and whether the host's copy of its matrices was let go
/// ([`Qwen3VlMmProj::keep_on_device`]).
#[derive(Default)]
pub struct TowerChain(std::sync::OnceLock<Option<crate::llm::chain::mmproj::Tower>>, std::sync::atomic::AtomicBool);

impl TowerChain {
    pub(crate) fn get_or_init(&self, make: impl FnOnce() -> Option<crate::llm::chain::mmproj::Tower>) -> &Option<crate::llm::chain::mmproj::Tower> {
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

        // ----- DeepStack mergers (Qwen3-VL's; Qwen3.5's tower has none) --
        let mut deepstack = Vec::new();
        for block in 0..config.n_layers {
            let name = |part: &str| format!("v.deepstack.{block}.{part}");
            let Some(norm_w) = idx.try_take(&name("norm.weight")) else { continue };
            deepstack.push(Qwen3VlDeepstack {
                block,
                norm_w: norm_w?,
                norm_b: idx.take(&name("norm.bias"), &[])?,
                fc1:    idx.take_weight(&name("fc1.weight"), &[])?,
                fc1_b:  idx.take(&name("fc1.bias"), &[])?,
                fc2:    idx.take_weight(&name("fc2.weight"), &[])?,
                fc2_b:  idx.take(&name("fc2.bias"), &[])?,
            });
        }
        let merged = 4 * config.embedding_dim;
        for ds in &deepstack {
            let (s1, s2) = (ds.fc1.shape(), ds.fc2.shape());
            if ds.norm_w.shape() != [merged] || s1.len() != 2 || s1[1] != merged || s2.len() != 2 || s2[1] != s1[0] || s2[0] != projector.mm2.shape()[0] {
                return Err(LlamaError::Config(format!("v.deepstack.{}: not a merger of 2 x 2 patches to the projector's width", ds.block)));
            }
        }

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
        let deepstack = deepstack.into_iter().map(|d| Qwen3VlDeepstack {
            block:  d.block,
            norm_w: upload(d.norm_w),
            norm_b: upload(d.norm_b),
            fc1:    d.fc1.try_to_device(&*backend, M),
            fc1_b:  upload(d.fc1_b),
            fc2:    d.fc2.try_to_device(&*backend, M),
            fc2_b:  upload(d.fc2_b),
        }).collect();
        let post_ln_w = upload(post_ln_w);
        let post_ln_b = upload(post_ln_b);
        let position_embd = upload(position_embd);
        // patch_embd kept on host until forward time (unfold runs CPU-side).

        Ok(Self {
            config, patch_embd, patch_embd_b: upload(patch_embd_b),
            position_embd, blocks, post_ln_w, post_ln_b, projector, deepstack, backend,
            chain: Default::default(),
        })
    }

    /// its tower made on the backend's device now (else at the first picture) and the host's f32 copy
    /// of the matrices there let go: the device's f16 copy is what runs, and the host's was the same weights a second
    /// time (1.8 GB of a 24 GB Mac's memory, beside the card's copy, or beside its own GPU's in the same memory). The
    /// bytes let go; 0 where the device holds no tower (it runs op by op, on the host's weights). A picture the
    /// device's tower then does not take is refused, never run on weights let go.
    pub fn keep_on_device(&mut self) -> usize {
        if !crate::llm::chain::mmproj::prepare(self) {
            return 0;
        }
        let mut bytes = 0;
        for b in &mut self.blocks {
            for w in [&mut b.attn_qkv, &mut b.attn_out, &mut b.ffn_up, &mut b.ffn_down] {
                bytes += let_go(w);
            }
        }
        bytes += let_go(&mut self.projector.mm0) + let_go(&mut self.projector.mm2);
        for d in &mut self.deepstack {
            bytes += let_go(&mut d.fc1) + let_go(&mut d.fc2);
        }
        self.chain.1.store(true, std::sync::atomic::Ordering::Relaxed);
        bytes
    }

    /// Image → soft tokens. Same contract as the other mmproj variants.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        self.forward_deepstack(image).map(|(tokens, _)| tokens)
    }

    /// Image → soft tokens, and each DeepStack merger's tokens beside them (`[n / 4, lm_hidden]` each, in
    /// [`Self::deepstack`]'s order; none for a tower without them).
    pub fn forward_deepstack(&self, image: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
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
        // the whole tower chained on the backend's device where it can be (the plain matrices are
        // otherwise multiplied on the host by a backend that keeps dense weights there: 20 s a picture).
        if let Some(out) = crate::llm::chain::mmproj::forward_taps(self, &patches) {
            return Ok(out);
        }
        if self.chain.1.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(LlamaError::Config("the vision tower is on the GPU alone, which could not take this picture".into()));
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

        // ----- 3. Transformer blocks (DeepStack's mergers reading theirs) --
        let mut taps = Vec::with_capacity(self.deepstack.len());
        for (i, blk) in self.blocks.iter().enumerate() {
            x = qwen3vl_block_forward(backend, &x, blk, cfg);
            for ds in self.deepstack.iter().filter(|d| d.block == i) {
                let merged = backend.to_device(merge_2x2(&x.to_host(), cfg));
                let normed = backend.layer_norm(&merged, &ds.norm_w, &ds.norm_b, cfg.layer_norm_eps);
                let mut h = ds.fc1.linear(backend, &normed);
                backend.add_inplace_broadcast_last(&mut h, &ds.fc1_b);
                let h = qwen_merger_gelu(backend, &h);
                let mut out = ds.fc2.linear(backend, &h);
                backend.add_inplace_broadcast_last(&mut out, &ds.fc2_b);
                taps.push(out.to_host());
            }
        }

        // ----- 4. Post-LN ----------------------------------------------
        x = backend.layer_norm(&x, &self.post_ln_w, &self.post_ln_b, cfg.layer_norm_eps);

        // ----- 5. 2x2 spatial merge → [N/4, 4*D] -----------------------
        let mut h = backend.to_device(merge_2x2(&x.to_host(), cfg));

        // ----- 6. mm.0 → GELU → mm.2 -----------------------------------
        let mut h0 = self.projector.mm0.linear(backend, &h);
        backend.add_inplace_broadcast_last(&mut h0, &self.projector.mm0_b);
        // HF's patch merger uses erf GELU; its ViT blocks
        // use tanh GELU. Evaluate this once per image on host (not per token).
        h = qwen_merger_gelu(backend, &h0);
        let mut out = self.projector.mm2.linear(backend, &h);
        backend.add_inplace_broadcast_last(&mut out, &self.projector.mm2_b);
        Ok((out, taps))
    }
}

/// The patch grid's states (`[side * side, D]`, row-major) with each 2 x 2 neighbourhood's four side by side in one
/// row (`[side / 2 * side / 2, 4 D]`), each block's (0,0), (0,1), (1,0), (1,1): HF's `Qwen3VLPatchMerger`'s order.
/// On the host: a contiguous shuffle, and small (2304 x 1152 values).
fn merge_2x2(host: &Tensor, cfg: &MmProjConfig) -> Tensor {
    let side = cfg.image_size / cfg.patch_size;
    let new_side = side / 2;
    let d = cfg.embedding_dim;
    let merge_dim = 4 * d;
    let n_merged = new_side * new_side;
    let src = host.data();
    let mut merged = vec![0.0f32; n_merged * merge_dim];
    for by in 0..new_side {
        for bx in 0..new_side {
            let dst_off = (by * new_side + bx) * merge_dim;
            for sy in 0..2 {
                for sx in 0..2 {
                    let src_off = ((by * 2 + sy) * side + bx * 2 + sx) * d;
                    let dst = dst_off + (sy * 2 + sx) * d;
                    merged[dst..dst + d].copy_from_slice(&src[src_off..src_off + d]);
                }
            }
        }
    }
    Tensor::from_vec(merged, vec![n_merged, merge_dim])
}

pub(super) fn qwen_merger_gelu(b: &dyn Backend, x: &Tensor) -> Tensor {
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

    crate::llm::multimodal_rope::vision(b, &mut q3, cfg.image_size / cfg.patch_size);
    crate::llm::multimodal_rope::vision(b, &mut k3, cfg.image_size / cfg.patch_size);

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

// (its one test takes a WebGPU adapter)
#[cfg(all(test, feature = "webgpu"))]
mod tests {
    use super::*;

    /// Each row's cosine with the reference's, the worst first, and the mean.
    fn cosines(got: &[f32], want: &[f32], width: usize) -> (f64, f64) {
        let rows = want.len() / width;
        let (mut worst, mut sum) = (1f64, 0f64);
        for r in 0..rows {
            let (a, b) = (&got[r * width..(r + 1) * width], &want[r * width..(r + 1) * width]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let c = dot / (na * nb);
            worst = worst.min(c);
            sum += c;
        }
        (worst, sum / rows as f64)
    }

    /// Qwen3-VL-4B's tower from llama.cpp's mmproj GGUF, on the device, against transformers' (float32) on the same
    /// picture: its soft tokens and its three DeepStack features
    /// (`tools/qwen3vl/vision_reference.py` makes the reference).
    #[test]
    #[ignore = "needs Qwen3-VL's mmproj GGUF (OAIY_QWEN3VL_MMPROJ), the reference (OAIY_QWEN3VL_VISION_EXPECTED) and a WebGPU adapter"]
    fn qwen3vl_tower_and_deepstack_are_transformers() {
        let (Ok(mmproj), Ok(expected)) = (std::env::var("OAIY_QWEN3VL_MMPROJ"), std::env::var("OAIY_QWEN3VL_VISION_EXPECTED")) else { return };
        let Ok(gpu) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
        let backend: Arc<dyn Backend> = Arc::new(gpu);
        let file = GgufFile::open(&mmproj).unwrap();
        let MmProj::Qwen3Vl(mm) = MmProj::from_gguf(&file, backend).unwrap() else { panic!("not a Qwen3-VL tower") };
        assert_eq!(mm.deepstack.iter().map(|d| d.block).collect::<Vec<_>>(), [5, 11, 17]);
        let want = oaiy_engine::safetensors::StIndex::open_file(std::path::Path::new(&expected)).unwrap();
        let image = Tensor::from_vec(want.read_f32("image").unwrap(), vec![3, 768, 768]);
        let (tokens, deepstack) = (want.read_f32("tokens").unwrap(), want.read_f32("deepstack").unwrap());
        let width = mm.projector.mm2.shape()[0];
        let t = std::time::Instant::now();
        let (got, taps) = mm.forward_deepstack(&image).unwrap();
        eprintln!("the tower in {:.2} s", t.elapsed().as_secs_f64());
        assert_eq!(got.shape(), [576, width]);
        let (worst, mean) = cosines(got.to_host().data(), &tokens, width);
        eprintln!("soft tokens: cosine worst {worst:.5}, mean {mean:.5}");
        assert!(worst > 0.995, "soft tokens: worst cosine {worst}");
        assert_eq!(taps.len(), 3);
        for (i, tap) in taps.iter().enumerate() {
            let (worst, mean) = cosines(tap.to_host().data(), &deepstack[i * 576 * width..(i + 1) * 576 * width], width);
            eprintln!("DeepStack {i}: cosine worst {worst:.5}, mean {mean:.5}");
            assert!(worst > 0.995, "DeepStack {i}: worst cosine {worst}");
        }
    }
}
