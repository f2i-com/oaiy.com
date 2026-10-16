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
    /// each linear layer's learned input and output ranges (Transformers' `Gemma4ClippableLinear`).
    pub clamps:     Gemma4VBlockClamps,
}

/// a linear layer's input and output ranges (`<name>.input_min` etc. in the mmproj; all of `f32`
/// where it has none).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clamp {
    pub input: (f32, f32),
    pub output: (f32, f32),
}

impl Clamp {
    pub const NONE: Clamp = Clamp { input: (f32::MIN, f32::MAX), output: (f32::MIN, f32::MAX) };

    /// `<weight>.{input,output}_{min,max}`, where the file has them.
    fn of(idx: &TensorIndex<'_>, weight: &str) -> Clamp {
        let base = weight.trim_end_matches(".weight");
        let scalar = |suffix: &str, default: f32| {
            idx.take(&format!("{base}.{suffix}"), &[]).ok().and_then(|t| t.data().first().copied()).unwrap_or(default)
        };
        Clamp {
            input: (scalar("input_min", f32::MIN), scalar("input_max", f32::MAX)),
            output: (scalar("output_min", f32::MIN), scalar("output_max", f32::MAX)),
        }
    }
}

/// a block's linear layers' ranges.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4VBlockClamps {
    pub attn_q: Clamp,
    pub attn_k: Clamp,
    pub attn_v: Clamp,
    pub attn_out: Clamp,
    pub ffn_gate: Clamp,
    pub ffn_up: Clamp,
    pub ffn_down: Clamp,
}

#[derive(Debug)]
pub struct Gemma4VProjector {
    /// `[lm_hidden, vision_dim]` after our load-time transpose. Same convention
    /// as the Gemma 3 projector.
    pub input_projection: Weight,
    /// its ranges.
    pub clamps: Clamp,
}

#[derive(Debug)]
pub struct Gemma4VMmProj {
    pub config:        MmProjConfig,
    pub patch_embd:    Tensor,            // [D, 3, P, P]
    pub position_embd: Tensor,            // [N_max, D] — slice first n_patches
    pub blocks:        Vec<Gemma4VBlock>,
    pub projector:     Gemma4VProjector,
    pub backend:       Arc<dyn Backend>,
    /// the tower chained on the backend's device (`chain::gemma4v`), made at the first picture.
    pub chain:         Gemma4VTowerChain,
}

/// [`Gemma4VMmProj::chain`]: the tower's weights on the device once a picture has asked for them.
#[derive(Default)]
pub struct Gemma4VTowerChain(std::sync::OnceLock<Option<crate::llm::chain::gemma4v::Tower>>);

impl Gemma4VTowerChain {
    pub(crate) fn get_or_init(&self, make: impl FnOnce() -> Option<crate::llm::chain::gemma4v::Tower>) -> &Option<crate::llm::chain::gemma4v::Tower> {
        self.0.get_or_init(make)
    }
}

impl std::fmt::Debug for Gemma4VTowerChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gemma4VTowerChain({})", match self.0.get() { None => "not yet", Some(None) => "none", Some(Some(_)) => "on the device" })
    }
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
                clamps: Gemma4VBlockClamps {
                    attn_q:   Clamp::of(&idx, &format!("v.blk.{i}.attn_q.weight")),
                    attn_k:   Clamp::of(&idx, &format!("v.blk.{i}.attn_k.weight")),
                    attn_v:   Clamp::of(&idx, &format!("v.blk.{i}.attn_v.weight")),
                    attn_out: Clamp::of(&idx, &format!("v.blk.{i}.attn_out.weight")),
                    ffn_gate: Clamp::of(&idx, &format!("v.blk.{i}.ffn_gate.weight")),
                    ffn_up:   Clamp::of(&idx, &format!("v.blk.{i}.ffn_up.weight")),
                    ffn_down: Clamp::of(&idx, &format!("v.blk.{i}.ffn_down.weight")),
                },
            });
        }

        // ----- projector (single linear, NO transpose) ------------------
        // Unlike Gemma 3 (which stores `mm.input_projection` as
        // `[vision_dim, lm_hidden]` and needs transposing), Gemma 4 V already
        // stores the projector in `[lm_hidden, vision_dim]` order — exactly
        // what `linear(x, w) = x @ w^T` expects. Use as-is.
        let projector = Gemma4VProjector {
            input_projection: idx.take_weight("mm.input_projection.weight", &[])?,
            clamps: Clamp::of(&idx, "mm.input_projection.weight"),
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
            clamps:      b.clamps,
        }).collect();
        let projector = Gemma4VProjector {
            input_projection: projector.input_projection.try_to_device(&*backend, M),
            clamps: projector.clamps,
        };
        let position_embd = upload(position_embd);
        // patch_embd kept on host until forward time (unfold runs CPU-side).

        Ok(Self { config, patch_embd, position_embd, blocks, projector, backend, chain: Default::default() })
    }

    /// `image` (`[3, h, w]`, values in `[0, 1]`, each side a multiple of `patch_size * 3`, as
    /// [`crate::llm::vision::gemma4_image`] makes it) through the tower and the projector: the soft tokens `[(h / 48)
    /// (w / 48), lm_hidden]`, row by row. On the backend's device ([`crate::llm::chain::gemma4v`]); a backend without a
    /// chain cannot (the op-by-op pass this replaced left out the rotary positions, the pooling, the clamps and the
    /// projector's norm, and its tokens were not the model's).
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let side = self.config.patch_size * crate::llm::chain::gemma4v::POOL;
        if image.rank() != 3 || image.dim(0) != 3 || image.dim(1) % side != 0 || image.dim(2) % side != 0 {
            return Err(LlamaError::Config(format!(
                "image shape {:?} is not [3, h, w] with each side a multiple of {side}", image.shape()
            )));
        }
        crate::llm::chain::gemma4v::forward(self, image).ok_or_else(|| LlamaError::Config(match self.quantized() {
            // (the chain takes the projector's matrices as the checkpoint has them, f16 on the device: a quantized
            // mmproj loads, and every picture would otherwise be refused for a GPU that is there)
            Some(dtype) => format!("this Gemma 4 vision projector is quantized ({dtype:?}): its tower runs from an F16, BF16 or F32 mmproj"),
            None => "Gemma 4's vision tower runs on a GPU (WebGPU), and this backend has none for it".into(),
        }))
    }

    /// The type of the first of the tower's matrices that is quantized (an mmproj in Q8_0 or a K-quant), if any.
    fn quantized(&self) -> Option<ggml_quants::GgmlType> {
        let blocks = self.blocks.iter().flat_map(|b| [&b.attn_q, &b.attn_k, &b.attn_v, &b.attn_out, &b.ffn_gate, &b.ffn_up, &b.ffn_down]);
        blocks.chain([&self.projector.input_projection]).find_map(|w| match w {
            Weight::Quant(q) => Some(q.dtype()),
            _ => None,
        })
    }
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
