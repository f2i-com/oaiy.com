// Ported from the user-owned F2I plugin-diffusion SDXL implementation.
//! SDXL dual text encoder — CLIP-L (OpenAI) + CLIP-G (OpenCLIP).
//!
//! SDXL concatenates the last-hidden-states of two CLIP variants along
//! the channel dim to form the UNet cross-attention context (2048-d).
//! CLIP-G's pooled EOT-token output (× `text_projection`) is fed into
//! the UNet's `label_emb` ADM conditioner alongside the size/crop
//! micro-conditioning.
//!
//! Both encoders use **causal** self-attention over 77 tokens. The two
//! layouts differ enough to warrant separate structs:
//!
//! | aspect              | CLIP-L (HF naming)                        | CLIP-G (OpenCLIP naming)              |
//! |---------------------|-------------------------------------------|---------------------------------------|
//! | top-level prefix    | `text_model.*`                            | (bare)                                |
//! | layers prefix       | `text_model.encoder.layers.{i}.*`         | `transformer.resblocks.{i}.*`         |
//! | qkv layout          | separate `self_attn.{q,k,v,out}_proj`     | FUSED `attn.in_proj_{weight,bias}`    |
//! | layer norms         | `layer_norm1`, `layer_norm2`              | `ln_1`, `ln_2`                        |
//! | MLP names           | `mlp.fc1`, `mlp.fc2`                      | `mlp.c_fc`, `mlp.c_proj`              |
//! | final norm          | `text_model.final_layer_norm`             | `ln_final`                            |
//! | embeddings          | `text_model.embeddings.{token,position}_embedding.weight` | `token_embedding.weight`, `positional_embedding` (no `.weight`) |
//! | activation          | **QuickGELU** (`x * sigmoid(1.702 x)`)    | **plain GELU**                        |
//! | num layers          | 12                                        | 32                                    |
//! | hidden / heads / ff | 768 / 12 / 3072                           | 1280 / 20 / 5120                      |
//! | pooled output       | (unused by SDXL)                          | `ln_final(x)[:, eot] @ text_projection` |
//!
//! ## Output selection (SDXL convention)
//!
//! SDXL uses the **penultimate** hidden state (skip the last
//! transformer layer's post-block output but stop BEFORE
//! `final_layer_norm`). HF's `CLIPTextModel` calls this
//! `clip_skip = 1` / "second-to-last layer". The reference Stability
//! AI pipeline applies this for BOTH CLIP-L and CLIP-G.
//!
//! Pooled output is only used from CLIP-G: take the final hidden state
//! (post-`ln_final`), pick the EOT-token position (49407 in stock
//! CLIP), and project via `text_projection` to (1, 1280).

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, ops, Embedding, LayerNorm, Linear, VarBuilder};

use crate::sdxl::config::CLIPConfig;

// ---------------------------------------------------------------------------
// Shared math helpers
// ---------------------------------------------------------------------------

/// OpenAI's "QuickGELU" approximation: `x * sigmoid(1.702 * x)`. Differs
/// from PyTorch's `gelu` (which uses `0.5 * x * (1 + erf(x / sqrt(2)))`)
/// — the two diverge by up to ~0.05 at the bend, enough to shift the
/// final CLIP-L embeddings noticeably. SDXL was trained against
/// QuickGELU for CLIP-L; using plain GELU here produces visibly worse
/// results.
fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    let s = (x * 1.702f64)?.apply(&candle_nn::Activation::Sigmoid)?;
    x * s
}

/// Build a `(1, 1, seq, seq)` causal attention bias suitable for adding
/// to pre-softmax scores: 0 on the lower triangle, `f32::NEG_INFINITY`
/// above. Causal because CLIP's text encoder is auto-regressive (each
/// token attends only to itself and earlier tokens).
fn causal_bias(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq)
        .flat_map(|i| (0..seq).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let t = Tensor::from_vec(mask, (1usize, 1, seq, seq), device)?;
    t.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// CLIP-L (OpenAI CLIP-ViT-L/14, HF naming)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ClipLAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl ClipLAttention {
    fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let q = linear(h, h, vb.pp("q_proj"))?;
        let k = linear(h, h, vb.pp("k_proj"))?;
        let v = linear(h, h, vb.pp("v_proj"))?;
        let o = linear(h, h, vb.pp("out_proj"))?;
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor, causal: &Tensor) -> Result<Tensor> {
        let (b, t, _h) = x.dims3()?;
        let q = x.apply(&self.q)?;
        let k = x.apply(&self.k)?;
        let v = x.apply(&self.v)?;
        // (b, t, h) -> (b, n_heads, t, head_dim)
        let split = |x: &Tensor| -> Result<Tensor> {
            x.reshape((b, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)
        };
        let q = split(&q)?.contiguous()?;
        let k = split(&k)?.contiguous()?;
        let v = split(&v)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        // Broadcast causal bias (1, 1, t, t) over batch + heads.
        scores = scores.broadcast_add(causal)?;
        // F32 softmax — CLIP encoders are post-LN with growing scales;
        // BF16 softmax accumulates error fast.
        let scores_dtype = scores.dtype();
        let scores_f32 = scores.to_dtype(DType::F32)?;
        let attn = ops::softmax_last_dim(&scores_f32)?.to_dtype(scores_dtype)?;
        let out = attn.matmul(&v)?;
        // (b, n_heads, t, head_dim) -> (b, t, h)
        out.transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.head_dim))?
            .apply(&self.o)
    }
}

#[derive(Debug, Clone)]
struct ClipLMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipLMlp {
    fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.fc1)?;
        let h = quick_gelu(&h)?;
        h.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct ClipLLayer {
    ln1: LayerNorm,
    attn: ClipLAttention,
    ln2: LayerNorm,
    mlp: ClipLMlp,
}

impl ClipLLayer {
    fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.hidden_size, 1e-5, vb.pp("layer_norm1"))?,
            attn: ClipLAttention::load(cfg, vb.pp("self_attn"))?,
            ln2: layer_norm(cfg.hidden_size, 1e-5, vb.pp("layer_norm2"))?,
            mlp: ClipLMlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, causal: &Tensor) -> Result<Tensor> {
        let h = (x + self.attn.forward(&x.apply(&self.ln1)?, causal)?)?;
        let m = self.mlp.forward(&h.apply(&self.ln2)?)?;
        h + m
    }
}

/// CLIP-L (OpenAI CLIP-ViT-L/14) text encoder. Stripped-prefix loader:
/// expects keys under `text_model.*` (matches the splitter output).
#[derive(Debug, Clone)]
pub struct ClipL {
    token_embedding: Embedding,
    position_embedding: Tensor,
    layers: Vec<ClipLLayer>,
    final_layer_norm: LayerNorm,
    cfg: CLIPConfig,
}

impl ClipL {
    pub fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        let tm = vb.pp("text_model");
        let emb = tm.pp("embeddings");
        let token_embedding =
            embedding(cfg.vocab_size, cfg.hidden_size, emb.pp("token_embedding"))?;
        // position_embedding stored as a normal Linear-style tensor with
        // a `.weight` suffix (HF convention).
        let position_embedding = emb
            .pp("position_embedding")
            .get((cfg.max_position_embeddings, cfg.hidden_size), "weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_vb = tm.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipLLayer::load(cfg, layers_vb.pp(i.to_string()))?);
        }
        let final_layer_norm = layer_norm(cfg.hidden_size, 1e-5, tm.pp("final_layer_norm"))?;
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            cfg: cfg.clone(),
        })
    }

    /// Forward returning EVERY layer's post-block hidden state (length
    /// `num_hidden_layers + 1` — the first entry is the post-embedding
    /// pre-block input, then one per layer's output). Caller picks
    /// `[-1]` for "use final layer" or `[-2]` for SDXL's penultimate.
    ///
    /// `input_ids` shape: `(B, T)` int. T must be ≤ `max_position_embeddings`.
    pub fn forward_all(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let (b, t) = input_ids.dims2()?;
        if t > self.cfg.max_position_embeddings {
            candle_core::bail!(
                "CLIP-L: sequence length {t} exceeds max_position_embeddings {}",
                self.cfg.max_position_embeddings
            );
        }
        let tok = input_ids.apply(&self.token_embedding)?;
        // Position embedding lookup: take rows [0..t) of (max, hidden)
        // → broadcast-add to (b, t, hidden).
        let pos = self.position_embedding.narrow(0, 0, t)?.unsqueeze(0)?;
        let mut h = tok.broadcast_add(&pos)?;
        let causal = causal_bias(t, h.dtype(), h.device())?;
        let mut all = Vec::with_capacity(self.layers.len() + 1);
        all.push(h.clone());
        for layer in &self.layers {
            h = layer.forward(&h, &causal)?;
            all.push(h.clone());
        }
        let _ = b;
        Ok(all)
    }

    /// SDXL convention: return the **penultimate** (layer[-2] in the
    /// `forward_all` list) hidden state. NOT passed through
    /// `final_layer_norm`. Shape: `(B, T, 768)`.
    pub fn forward_penultimate(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_at_skip(input_ids, 1)
    }

    /// CLIP-Skip variant: return the `(layer[-(1+skip)])` hidden state.
    /// `skip=1` ≡ `forward_penultimate` (the SDXL default). `skip=2`
    /// reaches one layer earlier; other UIs may number this differently. Higher skips climb
    /// further up the stack; values past `num_hidden_layers` clamp to
    /// the post-embedding pre-block input.
    pub fn forward_at_skip(&self, input_ids: &Tensor, skip: usize) -> Result<Tensor> {
        let all = self.forward_all(input_ids)?;
        let n = all.len();
        let s = skip.max(1).min(n - 1);
        Ok(all[n - 1 - s].clone())
    }

    /// Apply the loaded `final_layer_norm` to a hidden state. Exposed
    /// so the pipeline can compute the "use last layer post-norm"
    /// variant when needed (some non-stock workflows do this).
    pub fn apply_final_norm(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.final_layer_norm)
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

// ---------------------------------------------------------------------------
// CLIP-G (OpenCLIP ViT-bigG-14, OpenCLIP naming)
// ---------------------------------------------------------------------------
//
// OpenCLIP packs Q, K, V into a single (3*h, h) `in_proj_weight` and a
// (3*h,) `in_proj_bias`. We split the matrix at load time so the
// forward pass uses three independent matmuls (matches the candle
// Linear API). The fused storage is purely a torch.nn.MultiheadAttention
// historical artifact.

#[derive(Debug, Clone)]
struct ClipGAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl ClipGAttention {
    fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        // Fused in_proj_weight (3h, h) → split into three (h, h) blocks.
        let in_proj_weight = vb.pp("attn").get((3 * h, h), "in_proj_weight")?;
        let in_proj_bias = vb.pp("attn").get(3 * h, "in_proj_bias")?;
        let q_w = in_proj_weight.narrow(0, 0, h)?.contiguous()?;
        let k_w = in_proj_weight.narrow(0, h, h)?.contiguous()?;
        let v_w = in_proj_weight.narrow(0, 2 * h, h)?.contiguous()?;
        let q_b = in_proj_bias.narrow(0, 0, h)?.contiguous()?;
        let k_b = in_proj_bias.narrow(0, h, h)?.contiguous()?;
        let v_b = in_proj_bias.narrow(0, 2 * h, h)?.contiguous()?;
        let q = Linear::new(q_w, Some(q_b));
        let k = Linear::new(k_w, Some(k_b));
        let v = Linear::new(v_w, Some(v_b));
        // out_proj uses standard Linear naming.
        let o = linear(h, h, vb.pp("attn").pp("out_proj"))?;
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor, causal: &Tensor) -> Result<Tensor> {
        let (b, t, _h) = x.dims3()?;
        let q = x.apply(&self.q)?;
        let k = x.apply(&self.k)?;
        let v = x.apply(&self.v)?;
        let split = |x: &Tensor| -> Result<Tensor> {
            x.reshape((b, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)
        };
        let q = split(&q)?.contiguous()?;
        let k = split(&k)?.contiguous()?;
        let v = split(&v)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        scores = scores.broadcast_add(causal)?;
        let scores_dtype = scores.dtype();
        let scores_f32 = scores.to_dtype(DType::F32)?;
        let attn = ops::softmax_last_dim(&scores_f32)?.to_dtype(scores_dtype)?;
        let out = attn.matmul(&v)?;
        out.transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.head_dim))?
            .apply(&self.o)
    }
}

#[derive(Debug, Clone)]
struct ClipGMlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl ClipGMlp {
    fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_fc: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("c_fc"))?,
            c_proj: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // OpenCLIP G/14 uses standard GELU, NOT QuickGELU. Confirmed
        // by the OpenCLIP G/14 trained config (`act_layer = "gelu"`).
        // candle's `Activation::Gelu` is the standard erf-based GELU.
        let h = x.apply(&self.c_fc)?.apply(&candle_nn::Activation::Gelu)?;
        h.apply(&self.c_proj)
    }
}

#[derive(Debug, Clone)]
struct ClipGLayer {
    ln1: LayerNorm,
    attn: ClipGAttention,
    ln2: LayerNorm,
    mlp: ClipGMlp,
}

impl ClipGLayer {
    fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.hidden_size, 1e-5, vb.pp("ln_1"))?,
            attn: ClipGAttention::load(cfg, vb.clone())?,
            ln2: layer_norm(cfg.hidden_size, 1e-5, vb.pp("ln_2"))?,
            mlp: ClipGMlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, causal: &Tensor) -> Result<Tensor> {
        let h = (x + self.attn.forward(&x.apply(&self.ln1)?, causal)?)?;
        let m = self.mlp.forward(&h.apply(&self.ln2)?)?;
        h + m
    }
}

/// CLIP-G (OpenCLIP ViT-bigG-14, LAION-2B) text encoder. Stripped-prefix
/// loader: expects bare keys (post-strip of `conditioner.embedders.1.model.`).
#[derive(Debug, Clone)]
pub struct ClipG {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    layers: Vec<ClipGLayer>,
    ln_final: LayerNorm,
    text_projection: Linear,
    cfg: CLIPConfig,
}

impl ClipG {
    pub fn load(cfg: &CLIPConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embedding"))?;
        let positional_embedding = vb.get(
            (cfg.max_position_embeddings, cfg.hidden_size),
            "positional_embedding",
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let res_vb = vb.pp("transformer").pp("resblocks");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipGLayer::load(cfg, res_vb.pp(i.to_string()))?);
        }
        let ln_final = layer_norm(cfg.hidden_size, 1e-5, vb.pp("ln_final"))?;
        // `text_projection` is stored as a bare matrix (no `.weight`
        // suffix) — instantiate as a Linear with no bias.
        let proj_w = vb.get((cfg.hidden_size, cfg.hidden_size), "text_projection")?;
        // OpenCLIP applies text_projection as `x @ text_projection`
        // (row-vector × matrix), which corresponds to `Linear(W=proj.T)`.
        // We transpose at load so the forward uses Linear's standard
        // `x @ W.T` convention.
        let text_projection = Linear::new(proj_w.t()?.contiguous()?, None);
        Ok(Self {
            token_embedding,
            positional_embedding,
            layers,
            ln_final,
            text_projection,
            cfg: cfg.clone(),
        })
    }

    /// All hidden states (length = num_hidden_layers + 1).
    pub fn forward_all(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let (_b, t) = input_ids.dims2()?;
        if t > self.cfg.max_position_embeddings {
            candle_core::bail!(
                "CLIP-G: sequence length {t} exceeds max_position_embeddings {}",
                self.cfg.max_position_embeddings
            );
        }
        let tok = input_ids.apply(&self.token_embedding)?;
        let pos = self.positional_embedding.narrow(0, 0, t)?.unsqueeze(0)?;
        let mut h = tok.broadcast_add(&pos)?;
        let causal = causal_bias(t, h.dtype(), h.device())?;
        let mut all = Vec::with_capacity(self.layers.len() + 1);
        all.push(h.clone());
        for layer in &self.layers {
            h = layer.forward(&h, &causal)?;
            all.push(h.clone());
        }
        Ok(all)
    }

    /// SDXL context: penultimate hidden state. Shape: `(B, T, 1280)`.
    pub fn forward_penultimate(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_at_skip(input_ids, 1)
    }

    /// See `ClipL::forward_at_skip` — same explicit worker clip-skip semantics
    /// applied to the 32-layer OpenCLIP-G stack.
    pub fn forward_at_skip(&self, input_ids: &Tensor, skip: usize) -> Result<Tensor> {
        let all = self.forward_all(input_ids)?;
        let n = all.len();
        let s = skip.max(1).min(n - 1);
        Ok(all[n - 1 - s].clone())
    }

    /// SDXL pooled output: take the FINAL layer's hidden state, apply
    /// `ln_final`, pick the EOT-token position per row, project to
    /// 1280-d via `text_projection`. Shape: `(B, 1280)`.
    ///
    /// `eot_positions[i]` is the column index of the EOT (`<|endoftext|>`)
    /// token in batch row i. For stock CLIP this is the highest token-id
    /// seen at index ≥1 (the BOT token is always at 0).
    pub fn forward_pooled(&self, input_ids: &Tensor, eot_positions: &[usize]) -> Result<Tensor> {
        let all = self.forward_all(input_ids)?;
        let last = all
            .last()
            .expect("forward_all always pushes ≥1 entry")
            .clone();
        let normed = last.apply(&self.ln_final)?;
        // Per-row gather of eot_positions. For batch=1 the common case,
        // a single .i((0, p)) suffices; we still handle b>1 via stack.
        let (b, _t, _h) = normed.dims3()?;
        if eot_positions.len() != b {
            candle_core::bail!(
                "CLIP-G pooled: got {} EOT positions for batch size {}",
                eot_positions.len(),
                b
            );
        }
        let mut rows: Vec<Tensor> = Vec::with_capacity(b);
        for (i, &p) in eot_positions.iter().enumerate() {
            rows.push(normed.i((i, p))?);
        }
        let stacked = Tensor::stack(&rows, 0)?;
        stacked.apply(&self.text_projection)
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

// ---------------------------------------------------------------------------
// Dual encoder: runs both, concatenates context, returns pooled-G
// ---------------------------------------------------------------------------

/// Stitched output of running both CLIPs. Matches the format the SDXL
/// UNet consumes:
///   * `context`: concatenation of CLIP-L penultimate (768) and CLIP-G
///     penultimate (1280) along the channel dim → (B, T, 2048).
///   * `pooled`: CLIP-G's pooled+projected vector (B, 1280) — fed into
///     UNet's `label_emb` ADM conditioner alongside size/crop micro-cond.
#[derive(Debug)]
pub struct DualClipOutput {
    pub context: Tensor,
    pub pooled: Tensor,
}

/// Convenience helper that runs both encoders and concatenates their
/// (clip-skipped) penultimate states. Both `input_ids_l` and
/// `input_ids_g` must have the same batch + sequence length (SDXL
/// tokenizes the same prompt with both BPE vocabs and pads to 77).
///
/// `clip_skip = 1` is the SDXL default (penultimate layer). `2` is the
/// worker setting for one layer earlier in the stack. UI conventions differ.
pub fn dual_encode(
    clip_l: &ClipL,
    clip_g: &ClipG,
    input_ids_l: &Tensor,
    input_ids_g: &Tensor,
    eot_positions_g: &[usize],
    clip_skip: usize,
) -> Result<DualClipOutput> {
    let ctx_l = clip_l.forward_at_skip(input_ids_l, clip_skip)?;
    let ctx_g = clip_g.forward_at_skip(input_ids_g, clip_skip)?;
    // Cast to a uniform dtype before concat — both encoders typically
    // load at F16/BF16 to match the rest of the SDXL pipeline.
    let target = ctx_l.dtype();
    let ctx_g = ctx_g.to_dtype(target)?;
    let context = Tensor::cat(&[&ctx_l, &ctx_g], D::Minus1)?;
    let pooled = clip_g
        .forward_pooled(input_ids_g, eot_positions_g)?
        .to_dtype(target)?;
    Ok(DualClipOutput { context, pooled })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdxl::config::CLIPConfig;

    #[test]
    fn quick_gelu_at_zero_is_zero() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1usize, 4), DType::F32, &dev).unwrap();
        let g = quick_gelu(&x).unwrap();
        let v: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        for v in v {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn causal_bias_blocks_future() {
        let dev = Device::Cpu;
        let mask = causal_bias(3, DType::F32, &dev).unwrap();
        let v: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // shape (1, 1, 3, 3) flattened row-major: row0 = [0, -inf, -inf]
        assert_eq!(v[0], 0.0);
        assert!(v[1].is_infinite() && v[1].is_sign_negative());
        assert!(v[2].is_infinite() && v[2].is_sign_negative());
        // row2 = [0, 0, 0]
        assert_eq!(v[6], 0.0);
        assert_eq!(v[7], 0.0);
        assert_eq!(v[8], 0.0);
    }

    #[test]
    fn config_shapes() {
        let l = CLIPConfig::clip_l_14_336();
        assert_eq!(l.hidden_size, 768);
        assert_eq!(l.num_hidden_layers, 12);
        assert_eq!(l.num_attention_heads, 12);
        let g = CLIPConfig::open_clip_g_14_laion2b();
        assert_eq!(g.hidden_size, 1280);
        assert_eq!(g.num_hidden_layers, 32);
        assert_eq!(g.num_attention_heads, 20);
        assert_eq!(g.intermediate_size, 5120);
    }
}
