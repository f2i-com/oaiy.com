//! Qwen3-MoE forward pass — Qwen3 attention block + Mixture-of-Experts FFN.
//!
//! Targets Qwen3-30B-A3B (typically 128 experts, top-8 routing) and any other
//! GGUF with `general.architecture = "qwen3moe"`. The attention path is the
//! same as `Qwen3Model` (per-head Q/K RMSNorm before RoPE); only the FFN block
//! differs — instead of a single `ffn_gate`/`ffn_up`/`ffn_down` SwiGLU, each
//! block has a small router + N expert FFNs and per-token top-K dispatch.
//!
//! Two GGUF tensor layouts for the experts are common:
//!   * **Stacked** (newer): `blk.N.ffn_gate_exps.weight` shape
//!     `[n_experts, ff, hidden]`; sliced into per-expert weights at load time.
//!   * **Per-expert** (older): `blk.N.ffn_gate.M.weight` shape `[ff, hidden]`.
//! Loader auto-detects which layout the file uses.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{load_lm_head_or_tied, upload_tok_embd_and_lm_head, TensorIndex, Weight};
use crate::moe::{moe_forward, MoeFfn};
use crate::{LlamaError, ModelConfig, Result};

#[derive(Debug)]
pub struct Qwen3MoeBlock {
    pub attn_norm:   Tensor,
    pub attn_q:      Weight,
    pub attn_q_norm: Tensor,
    pub attn_k:      Weight,
    pub attn_k_norm: Tensor,
    pub attn_v:      Weight,
    pub attn_output: Weight,
    pub ffn_norm:    Tensor,
    pub moe:         MoeFfn,
}

pub struct Qwen3MoeModel {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub blocks:    Vec<Qwen3MoeBlock>,
    pub tok_embd:  Arc<Tensor>,
    pub output_norm: Tensor,
    pub output:    Weight,
    pub n_experts:      usize,
    pub n_experts_used: usize,
    pub backend:   Arc<dyn Backend>,
    /// VENDORED-LOCAL: shared streaming-expert state when the model was
    /// opened with `from_gguf_streaming` (experts stay in the .gguf file behind
    /// a bounded cache, not in RAM). `None` for the resident load.
    pub stream_shared: Option<Arc<crate::expert_stream::StreamShared>>,
}

impl std::fmt::Debug for Qwen3MoeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3MoeModel")
            .field("config", &self.config)
            .field("n_experts", &self.n_experts)
            .field("n_experts_used", &self.n_experts_used)
            .field("backend", &self.backend.name())
            .finish()
    }
}

impl Qwen3MoeModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::from_gguf_impl(g, backend, None)
    }

    /// VENDORED-LOCAL: streaming variant. `shared` carries the expert store +
    /// bounded cache; expert tensors are NOT materialized — each block's
    /// `MoeFfn` gets a `LayerStream` and fetches routed experts per dispatch.
    /// Non-expert weights load resident exactly as in `from_gguf`.
    pub fn from_gguf_streaming(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        shared: Arc<crate::expert_stream::StreamShared>,
    ) -> Result<Self> {
        Self::from_gguf_impl(g, backend, Some(shared))
    }

    fn from_gguf_impl(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        stream_shared: Option<Arc<crate::expert_stream::StreamShared>>,
    ) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        let arch_name = g.architecture()?.to_string();
        let n_experts = g.get_u64(&format!("{arch_name}.expert_count"))
            .map(|v| v as usize)
            .map_err(|_| LlamaError::Config(format!("missing {arch_name}.expert_count")))?;
        let n_experts_used = g.get_u64(&format!("{arch_name}.expert_used_count"))
            .map(|v| v as usize)
            .map_err(|_| LlamaError::Config(format!("missing {arch_name}.expert_used_count")))?;

        let tok_embd    = Arc::new(idx.take("token_embd.weight",  &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        const M: usize = 2 * 1024 * 1024 * 1024;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let attn_norm   = idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?;
            let attn_q      = idx.take_weight(&format!("blk.{i}.attn_q.weight"), &[])?;
            let attn_q_norm = idx.take(&format!("blk.{i}.attn_q_norm.weight"), &[])?;
            let attn_k      = idx.take_weight(&format!("blk.{i}.attn_k.weight"), &[])?;
            let attn_k_norm = idx.take(&format!("blk.{i}.attn_k_norm.weight"), &[])?;
            let attn_v      = idx.take_weight(&format!("blk.{i}.attn_v.weight"), &[])?;
            let attn_output = idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?;
            let ffn_norm    = idx.take(&format!("blk.{i}.ffn_norm.weight"), &[])?;

            let router = idx.take_weight(&format!("blk.{i}.ffn_gate_inp.weight"), &[])?;
            // VENDORED-LOCAL: streaming mode skips materializing the experts;
            // the MoeFfn carries a per-layer stream handle instead.
            let moe = match &stream_shared {
                Some(shared) => MoeFfn {
                    router,
                    gate_up_experts: Vec::new(),
                    down_experts: Vec::new(),
                    top_k: n_experts_used,
                    stream: Some(shared.layer(i as u32)),
                    #[cfg(feature = "cuda")]
                    gpu_plan: std::sync::OnceLock::new(),
                }
                .move_to_device(&*backend, M),
                None => {
                    let (gate_up_experts, down_experts) =
                        load_per_layer_experts_pair(&idx, i, n_experts)?;
                    MoeFfn { router, gate_up_experts, down_experts, top_k: n_experts_used, stream: None,
                        #[cfg(feature = "cuda")]
                        gpu_plan: std::sync::OnceLock::new(),
                    }
                        .move_to_device(&*backend, M)
                }
            };

            blocks.push(Qwen3MoeBlock {
                attn_norm:   backend.to_device(attn_norm),
                attn_q:      attn_q.try_to_device(&*backend, M),
                attn_q_norm: backend.to_device(attn_q_norm),
                attn_k:      attn_k.try_to_device(&*backend, M),
                attn_k_norm: backend.to_device(attn_k_norm),
                attn_v:      attn_v.try_to_device(&*backend, M),
                attn_output: attn_output.try_to_device(&*backend, M),
                ffn_norm:    backend.to_device(ffn_norm),
                moe,
            });
        }

        let (tok_embd, output) = upload_tok_embd_and_lm_head(&*backend, tok_embd, output);
        let output = output.try_to_device(&*backend, M);
        Ok(Self {
            config, tokenizer, blocks,
            tok_embd,
            output_norm: backend.to_device(output_norm),
            output,
            n_experts,
            n_experts_used,
            backend,
            stream_shared,
        })
    }

    /// Look up token embeddings. Companion to [`forward_embeds`] for the
    /// vision-language splice path; standard text-only callers stay on
    /// [`forward`].
    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        self.backend.embed_lookup(&self.tok_embd, tokens, self.config.embedding_dim)
    }

    /// Vision-language splice helper mirroring `Qwen35Model::embed_with_vision_at_placeholder`:
    /// tokenize `prompt` (which must contain exactly one `placeholder_token_id`
    /// marker — `<|image_pad|>` for Qwen3-VL), expand into `soft_tokens.dim(0)`
    /// repetitions, embed, and splice the vision soft tokens into those rows.
    pub fn embed_with_vision_at_placeholder(
        &self,
        prompt:               &str,
        soft_tokens:          &Tensor,
        placeholder_token_id: u32,
        add_bos:              bool,
    ) -> Result<(Vec<u32>, Tensor)> {
        let prompt_ids = self.tokenizer.encode(prompt, add_bos)?;
        let pos = prompt_ids.iter().position(|&id| id == placeholder_token_id)
            .ok_or_else(|| crate::LlamaError::Config(format!(
                "placeholder token id {placeholder_token_id} not found in prompt"
            )))?;
        let n_soft = soft_tokens.dim(0);
        let d = self.config.embedding_dim;
        debug_assert_eq!(soft_tokens.dim(soft_tokens.rank() - 1), d);
        let mut tokens = Vec::with_capacity(prompt_ids.len() - 1 + n_soft);
        tokens.extend_from_slice(&prompt_ids[..pos]);
        tokens.extend(std::iter::repeat(placeholder_token_id).take(n_soft));
        tokens.extend_from_slice(&prompt_ids[pos + 1..]);
        let embeds = self.embed_text(&tokens);
        let mut host = embeds.to_host();
        let dst = host.data_mut();
        let src = soft_tokens.to_host();
        let src_data = src.data();
        let splice_off = pos * d;
        let splice_len = n_soft * d;
        dst[splice_off..splice_off + splice_len].copy_from_slice(&src_data[..splice_len]);
        let embeds = self.backend.to_device(host);
        Ok((tokens, embeds))
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        let embeds = self.embed_text(tokens);
        self.forward_embeds(&embeds, tokens.len(), kv)
    }

    /// Run the transformer stack on a pre-computed embedding tensor. Used by
    /// the multimodal path so vision soft tokens can be spliced into the
    /// embedding sequence before the LM stack runs.
    pub fn forward_embeds(&self, embeds: &Tensor, seq: usize, kv: &mut KvCache) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let past = kv.len;
        debug_assert_eq!(embeds.dim(0), seq);
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut x = embeds.clone();
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..cfg.n_layers {
            let b = &self.blocks[layer];
            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);
            let q_flat = b.attn_q.linear(backend, &xn);
            let k_flat = b.attn_k.linear(backend, &xn);
            let v_flat = b.attn_v.linear(backend, &xn);

            let q_3d = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let k_3d = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
            let mut q = ops::rmsnorm(backend, &q_3d, &b.attn_q_norm, cfg.rms_eps);
            let mut k = ops::rmsnorm(backend, &k_3d, &b.attn_k_norm, cfg.rms_eps);
            let v = v_flat.reshape(vec![seq, n_kv, hd]).expect("v reshape");

            let rope_type = cfg.arch.rope_type();
            ops::rope(backend, &mut q, &positions, hd, rope_type, cfg.rope_theta);
            ops::rope(backend, &mut k, &positions, hd, rope_type, cfg.rope_theta);

            kv.append(backend, layer, &k, &v);
            let kv_len = kv.len + seq;
            let attn_out = ops::attention(
                backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                kv_len, scale, past,
            );
            let attn_t = attn_out.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let attn_proj = b.attn_output.linear(backend, &attn_t);

            // Fused residual + pre-FFN rmsnorm: `x += attn_proj; xn2 = rmsnorm(x)`.
            let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &b.ffn_norm, cfg.rms_eps);
            let ffn_out = moe_forward(backend, &xn2, &b.moe);
            ops::add_inplace(backend, &mut x, &ffn_out);
        }

        kv.commit(seq);
        let x = ops::rmsnorm(backend, &x, &self.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        self.output.linear(backend, &x_last)
    }
}

/// Load per-layer expert FFN weights, auto-detecting stacked-vs-separate layout.
/// Stacked (newer): `blk.N.ffn_{gate,up,down}_exps.weight` with shape
/// `[n_experts, ff, hidden]` — sliced into per-expert weights at load time.
/// Separate (older Mixtral-era): `blk.N.ffn_{gate,up,down}.M.weight` per expert.
/// Public alias used by `mixtral.rs` (same per-layer expert loader).
pub fn load_per_layer_experts_compat(
    idx: &TensorIndex<'_>,
    layer: usize,
    n_experts: usize,
) -> Result<(Vec<crate::loader::FfnPair>, Vec<Weight>)> {
    load_per_layer_experts_pair(idx, layer, n_experts)
}

/// Load all of layer `layer`'s expert weights, fusing each expert's gate+up
/// pair into one stacked weight (`FfnPair::Fused`) when dtype/shape allow.
/// Cuts the per-expert matmul launch count from 3→2 at decode time.
// VENDORED-LOCAL: pub(crate) so glm5next.rs can reuse the stacked-expert loader.
pub(crate) fn load_per_layer_experts_pair(
    idx: &TensorIndex<'_>,
    layer: usize,
    n_experts: usize,
) -> Result<(Vec<crate::loader::FfnPair>, Vec<Weight>)> {
    use crate::loader::FfnPair;
    // Try stacked first.
    let stacked_gate = format!("blk.{layer}.ffn_gate_exps.weight");
    if idx.has(&stacked_gate) {
        let gate_stacked = idx.take_weight(&stacked_gate, &[])?;
        let up_stacked   = idx.take_weight(&format!("blk.{layer}.ffn_up_exps.weight"), &[])?;
        let down_stacked = idx.take_weight(&format!("blk.{layer}.ffn_down_exps.weight"), &[])?;
        let gate = split_stacked_experts(gate_stacked, n_experts)?;
        let up   = split_stacked_experts(up_stacked,   n_experts)?;
        let down = split_stacked_experts(down_stacked, n_experts)?;
        let pairs: Vec<FfnPair> = gate.into_iter().zip(up.into_iter())
            .map(|(g, u)| FfnPair::from_halves(g, u))
            .collect();
        return Ok((pairs, down));
    }
    // Fall back to per-expert tensors.
    let mut pairs = Vec::with_capacity(n_experts);
    let mut down  = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let g = idx.take_weight(&format!("blk.{layer}.ffn_gate.{e}.weight"), &[])?;
        let u = idx.take_weight(&format!("blk.{layer}.ffn_up.{e}.weight"),   &[])?;
        pairs.push(FfnPair::from_halves(g, u));
        down.push(idx.take_weight(&format!("blk.{layer}.ffn_down.{e}.weight"), &[])?);
    }
    Ok((pairs, down))
}

/// Split a stacked `[n_experts, dim_a, dim_b]` weight into `n_experts` per-
/// expert `[dim_a, dim_b]` weights. For Dense F32 this is a vector slice; for
/// packed quantized weights it's a byte-range slice — Q4_K and friends store
/// rows as contiguous block sequences, so axis-0 partition is just byte split.
pub fn split_stacked_experts(w: Weight, n_experts: usize) -> Result<Vec<Weight>> {
    match w {
        Weight::Dense(t) => {
            let shape = t.shape();
            if shape.len() != 3 || shape[0] != n_experts {
                return Err(LlamaError::Config(format!(
                    "stacked expert tensor expected shape [{n_experts}, dim_a, dim_b], got {shape:?}"
                )));
            }
            let per_expert_shape = vec![shape[1], shape[2]];
            let per_expert_n     = shape[1] * shape[2];
            let data = t.data().to_vec();
            let mut out = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let slice = data[e * per_expert_n..(e + 1) * per_expert_n].to_vec();
                out.push(Weight::Dense(Tensor::from_vec(slice, per_expert_shape.clone())));
            }
            Ok(out)
        }
        Weight::Quant(qt) => {
            let shape = qt.shape().to_vec();
            if shape.len() != 3 || shape[0] != n_experts {
                return Err(LlamaError::Config(format!(
                    "stacked expert tensor expected shape [{n_experts}, dim_a, dim_b], got {shape:?}"
                )));
            }
            let per_expert_shape = vec![shape[1], shape[2]];
            let total_bytes = qt.bytes().len();
            if total_bytes % n_experts != 0 {
                return Err(LlamaError::Config(format!(
                    "stacked quant expert bytes ({total_bytes}) not divisible by n_experts ({n_experts})"
                )));
            }
            let per_expert_bytes = total_bytes / n_experts;
            let dtype = qt.dtype();
            let bytes = qt.bytes().to_vec();
            let mut out = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let slice = bytes[e * per_expert_bytes..(e + 1) * per_expert_bytes].to_vec();
                out.push(Weight::Quant(ggml_rs::quantized::QuantizedTensor::from_bytes_cpu(
                    slice, per_expert_shape.clone(), dtype,
                )));
            }
            Ok(out)
        }
        Weight::TiedEmbed(_) => Err(LlamaError::Config(
            "split_stacked_experts: TiedEmbed is LM-head-only".into(),
        )),
    }
}

/// Split a 2D `[d0, d1]` weight along axis 0 into `n_chunks` weights of
/// `[d0/n_chunks, d1]`. Same byte-level split logic as `split_stacked_experts`
/// (Q4_K / Q5_1 / Q8_0 store rows as contiguous block sequences). Used by
/// Gemma 4 MoE to split each per-expert `ffn_gate_up_exps` `[2*ff, hidden]`
/// into separate gate `[ff, hidden]` and up `[ff, hidden]` weights at load.
pub fn split_axis0_2d(w: Weight, n_chunks: usize) -> Result<Vec<Weight>> {
    match w {
        Weight::Dense(t) => {
            let shape = t.shape();
            if shape.len() != 2 || shape[0] % n_chunks != 0 {
                return Err(LlamaError::Config(format!(
                    "axis-0 split: tensor shape {shape:?} not 2D or first dim not divisible by {n_chunks}"
                )));
            }
            let chunk_shape = vec![shape[0] / n_chunks, shape[1]];
            let chunk_n     = chunk_shape[0] * chunk_shape[1];
            let data = t.data().to_vec();
            let mut out = Vec::with_capacity(n_chunks);
            for c in 0..n_chunks {
                let slice = data[c * chunk_n..(c + 1) * chunk_n].to_vec();
                out.push(Weight::Dense(Tensor::from_vec(slice, chunk_shape.clone())));
            }
            Ok(out)
        }
        Weight::Quant(qt) => {
            let shape = qt.shape().to_vec();
            if shape.len() != 2 || shape[0] % n_chunks != 0 {
                return Err(LlamaError::Config(format!(
                    "axis-0 split: quant tensor shape {shape:?} not 2D or first dim not divisible by {n_chunks}"
                )));
            }
            let chunk_shape = vec![shape[0] / n_chunks, shape[1]];
            let total_bytes = qt.bytes().len();
            if total_bytes % n_chunks != 0 {
                return Err(LlamaError::Config(format!(
                    "axis-0 split: quant bytes ({total_bytes}) not divisible by {n_chunks}"
                )));
            }
            let per_chunk_bytes = total_bytes / n_chunks;
            let dtype = qt.dtype();
            let bytes = qt.bytes().to_vec();
            let mut out = Vec::with_capacity(n_chunks);
            for c in 0..n_chunks {
                let slice = bytes[c * per_chunk_bytes..(c + 1) * per_chunk_bytes].to_vec();
                out.push(Weight::Quant(ggml_rs::quantized::QuantizedTensor::from_bytes_cpu(
                    slice, chunk_shape.clone(), dtype,
                )));
            }
            Ok(out)
        }
        Weight::TiedEmbed(_) => Err(LlamaError::Config(
            "split_axis0_2d: TiedEmbed is LM-head-only".into(),
        )),
    }
}
