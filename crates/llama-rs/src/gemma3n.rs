//! Gemma 3n (a.k.a. "Gemma 4 E2B / E4B") — Google's MatFormer architecture.
//!
//! **Status: working.** Loads `gemma-3n-E2B-it` and produces coherent answers
//! end-to-end. The final fix was setting `f_attention_scale = 1.0` (Gemma 3n's
//! own quirk) instead of `1/sqrt(head_dim)`. Cross-referenced against
//! `llama.cpp/src/models/gemma3n.cpp`. Original HF reference:
//! `transformers/models/gemma3n/modeling_gemma3n.py`.
//!
//! Implemented (matches the upstream forward pass to the best of our reading):
//!   * **AltUp expand** at the start: stream 0 = scaled tok_embd, streams 1..3 =
//!     `altup_proj[i] @ stream_0` rescaled to match stream 0's per-token RMS.
//!   * **AltUp predict** per layer: routes the active stream through
//!     `router_norm` × `1/hidden_size` → `altup_router` → `tanh` → `predict_coef`
//!     to get a per-token 4×4 mixing matrix; mixes the streams + adds the
//!     residual.
//!   * **Laurel** low-rank (rank 64) parallel residual on the attention branch.
//!     `Laurel(x) = x + post_norm(W_r @ W_l @ x)`. Output combines into
//!     `(attn_gated + laurel) / sqrt(2)`.
//!   * **Per-Layer Embeddings** — `per_layer_token_embd` (a packed Q5_1 table
//!     of shape `[vocab × n_layers × per_layer_dim]`) is dequantized to host
//!     F32 at load time (~7.7 GB). Per forward we slice the seq tokens × all
//!     layers, project the hidden state via `per_layer_model_proj`, RMSNorm,
//!     and combine `(projection + lookup) / sqrt(2)`.
//!   * **Activation sparsity** — Gaussian-top-k mask on FFN gate per layer
//!     (the first 10 layers are 95% sparse — their `activation_sparsity_scale`
//!     value `1.6448` is `Φ⁻¹(0.95)`).
//!   * **AltUp correct** — re-routes from `attn_ffw_laurel_gated`, applies
//!     `correction_coefs(modalities) + 1.0` per stream as the innovation gain.
//!   * **PLE injection** — active stream × `correct_scale`, `inp_gate` linear,
//!     GeLU, multiply with the layer's PLE slice, `proj` linear, RMSNorm,
//!     added to streams 1..3.
//!   * **AltUp collapse** at the end — magnitude-rescale streams 1..3 via
//!     `altup_unembd_proj`, average all 4 streams, final `output_norm`,
//!     tied LM head.
//!   * **V-norm** — `with_scale=False` RMSNorm applied to the flat v projection
//!     (across all heads) for non-shared layers (i.e. layers `< n_layers − num_kv_shared`).
//!   * **Shared K/V layer mapping** — for the E2B 30-layer model, the last 10
//!     layers reuse earlier layers' K/V cache. We compute `kv_shared_source[i]`
//!     at load time per upstream's "find the last earlier non-shared layer of
//!     the same attention type". For Gemma 3n E2B's 4L1G pattern: layers 20-23
//!     and 25-28 (local) reuse layer 18; layers 24 and 29 (global) reuse layer 19.

use std::sync::Arc;

use ggml_quants::GgmlType;
use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{CommonTensors, FfnPair, TensorIndex, Weight};
use crate::{LlamaError, ModelConfig, Result};

const ALTUP_NUM_INPUTS: usize = 4;
const ALTUP_ACTIVE_IDX: usize = 0;

#[derive(Debug)]
pub struct Gemma3nGlobalTensors {
    /// 3 projection matrices for AltUp expand: streams 1..4 = altup_proj[i] @ stream0.
    /// Each is shape `[embedding_dim, embedding_dim]`.
    pub altup_proj: Vec<Tensor>,
    /// 3 projection matrices for AltUp collapse.
    pub altup_unembd_proj: Vec<Tensor>,
    /// PLE: project hidden state to per-layer space.
    /// Shape: `[n_layers * per_layer_dim, embedding_dim]`.
    pub per_layer_model_proj: Tensor,
    /// PLE: RMSNorm weight applied to per-layer projection. Shape: `[per_layer_dim]`.
    pub per_layer_proj_norm: Tensor,
    /// PLE: per-layer token embedding table on the model's backend (CPU or
    /// CUDA). Layout: `[vocab, n_layers * per_layer_dim]`. ~8 GB F32 for
    /// Gemma 3n E2B. Sliced per-token via `embed_lookup` at forward time.
    pub per_layer_token_embd: Tensor,
}

#[derive(Debug)]
pub struct Gemma3nBlockExtras {
    pub attn_post_norm:      Tensor,
    pub ffn_post_norm:       Tensor,
    pub post_norm:           Tensor,
    pub attn_q_norm:         Tensor,
    pub attn_k_norm:         Tensor,
    // AltUp coefficients (per block).
    pub altup_predict_coef:  Tensor,    // `[num_inputs² = 16, num_inputs = 4]`
    pub altup_correct_coef:  Tensor,    // `[num_inputs = 4, num_inputs = 4]`
    pub altup_correct_scale: Tensor,    // `[embedding_dim]`
    pub altup_router:        Tensor,    // `[num_inputs = 4, embedding_dim]`
    pub altup_router_norm:   Tensor,    // `[embedding_dim]`
    // Laurel.
    pub laurel_l:            Tensor,    // `[laurel_rank, embedding_dim]`
    pub laurel_r:            Tensor,    // `[embedding_dim, laurel_rank]`
    pub laurel_post_norm:    Tensor,    // `[embedding_dim]`
    // PLE injection (per block).
    pub inp_gate:            Weight,    // `[per_layer_dim, embedding_dim]`
    pub proj:                Weight,    // `[embedding_dim, per_layer_dim]`
}

pub struct Gemma3nModel {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub common:    CommonTensors,
    pub global:    Gemma3nGlobalTensors,
    pub extras:    Vec<Gemma3nBlockExtras>,
    pub backend:   Arc<dyn Backend>,
    embed_scale:    f32,
    per_layer_dim:  usize,
    /// For each layer index, the index of the source layer whose K/V cache it
    /// reuses (None = compute its own K/V). Computed at load time per upstream's
    /// "find the last earlier non-shared layer of the same attention type".
    kv_shared_source: Vec<Option<usize>>,
}

impl std::fmt::Debug for Gemma3nModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma3nModel")
            .field("config", &self.config)
            .field("backend", &self.backend.name())
            .field("vocab_size", &self.tokenizer.vocab_size())
            .finish()
    }
}

impl Gemma3nModel {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);
        // Force the FFN gate+up pair to stay split — Gemma 3n applies activation
        // sparsity to `gate` between the matmul and the GeLU, which the fused
        // `gelu_approx_mul_split` kernel can't express. Other Gemma variants get
        // auto-fused.
        let mut common = CommonTensors::load_with_index(&idx, &config)?;
        for b in &mut common.blocks {
            let pair = std::mem::replace(&mut b.ffn_pair,
                FfnPair::Fused(Weight::Dense(Tensor::zeros(vec![0, 0]))));
            b.ffn_pair = pair.into_split();
        }
        let common = common.upload_to(&*backend);

        let per_layer_dim = g.get_u64("gemma3n.embedding_length_per_layer_input")
            .map_err(|e| LlamaError::Config(format!("missing embedding_length_per_layer_input: {e}")))?
            as usize;

        // Global AltUp projections — stored as concatenated `[3 × embed, embed]`,
        // split into 3 separate matrices for per-stream linear application.
        let altup_proj_combined = idx.take("altup_proj.weight", &[])?;
        let altup_unembd_proj_combined = idx.take("altup_unembd_proj.weight", &[])?;
        let altup_proj = split_3way(altup_proj_combined, config.embedding_dim, &*backend);
        let altup_unembd_proj = split_3way(altup_unembd_proj_combined, config.embedding_dim, &*backend);

        let per_layer_model_proj = backend.to_device(idx.take("per_layer_model_proj.weight", &[])?);
        let per_layer_proj_norm = backend.to_device(idx.take("per_layer_proj_norm.weight", &[])?);

        // PLE table: ~2 GB packed; dequantize once and keep on host. Per-forward
        // we slice [seq × n_layers × per_layer_dim] bytes out — small enough to
        // round-trip to GPU each step.
        let ple_info = g.tensor_by_name("per_layer_token_embd.weight")
            .ok_or_else(|| LlamaError::MissingTensor("per_layer_token_embd.weight".into()))?;
        let n_elem = ple_info.numel() as usize;
        let mut ple_host = vec![0.0f32; n_elem];
        // VENDORED-LOCAL: byte-source seam (works for mmap'd and source-backed files).
        ggml_quants::dequantize(ple_info.dtype, &g.tensor_bytes(ple_info)?, &mut ple_host)?;
        let row_size = config.n_layers * per_layer_dim;
        let per_layer_token_embd = backend.to_device(
            Tensor::from_vec(ple_host, vec![config.vocab_size, row_size]),
        );

        let global = Gemma3nGlobalTensors {
            altup_proj,
            altup_unembd_proj,
            per_layer_model_proj,
            per_layer_proj_norm,
            per_layer_token_embd,
        };

        let mut extras = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            extras.push(Gemma3nBlockExtras {
                attn_post_norm:      backend.to_device(idx.take(&format!("blk.{i}.post_attention_norm.weight"), &[])?),
                ffn_post_norm:       backend.to_device(idx.take(&format!("blk.{i}.post_ffw_norm.weight"), &[])?),
                post_norm:           backend.to_device(idx.take(&format!("blk.{i}.post_norm.weight"), &[])?),
                attn_q_norm:         backend.to_device(idx.take(&format!("blk.{i}.attn_q_norm.weight"), &[])?),
                attn_k_norm:         backend.to_device(idx.take(&format!("blk.{i}.attn_k_norm.weight"), &[])?),
                altup_predict_coef:  backend.to_device(idx.take(&format!("blk.{i}.altup_predict_coef.weight"), &[])?),
                altup_correct_coef:  backend.to_device(idx.take(&format!("blk.{i}.altup_correct_coef.weight"), &[])?),
                altup_correct_scale: backend.to_device(idx.take(&format!("blk.{i}.altup_correct_scale.weight"), &[])?),
                altup_router:        backend.to_device(idx.take(&format!("blk.{i}.altup_router.weight"), &[])?),
                altup_router_norm:   backend.to_device(idx.take(&format!("blk.{i}.altup_router_norm.weight"), &[])?),
                laurel_l:            backend.to_device(idx.take(&format!("blk.{i}.laurel_l.weight"), &[])?),
                laurel_r:            backend.to_device(idx.take(&format!("blk.{i}.laurel_r.weight"), &[])?),
                laurel_post_norm:    backend.to_device(idx.take(&format!("blk.{i}.laurel_post_norm.weight"), &[])?),
                inp_gate:            idx.take_weight(&format!("blk.{i}.inp_gate.weight"), &[])?.try_to_device(&*backend, 2 * 1024 * 1024 * 1024),
                proj:                idx.take_weight(&format!("blk.{i}.proj.weight"), &[])?.try_to_device(&*backend, 2 * 1024 * 1024 * 1024),
            });
        }

        // Avoid an unused-import warning in the no-cuda build.
        let _ = GgmlType::F32;

        let embed_scale = (config.embedding_dim as f32).sqrt();
        let num_kv_shared = g.get_u64("gemma3n.attention.shared_kv_layers").unwrap_or(0) as usize;
        let first_kv_shared_layer_idx = config.n_layers.saturating_sub(num_kv_shared);

        // Build kv_shared_source: per-layer index of the K/V source layer.
        // Upstream: for each shared layer, find the LAST earlier non-shared layer
        // of the same attention type (sliding/local vs global). Treats unknown
        // layer types as "all the same" (degrades to layer first_kv_shared - 1).
        let layer_types: Vec<bool> = config.sliding_window_layers
            .clone()
            .unwrap_or_else(|| (0..config.n_layers).map(|i| (i + 1) % config.sliding_window_pattern != 0).collect());
        let mut kv_shared_source = vec![None; config.n_layers];
        for layer_idx in first_kv_shared_layer_idx..config.n_layers {
            let want = layer_types.get(layer_idx).copied().unwrap_or(true);
            // Search prev_layers (0..first_kv_shared_layer_idx) reversed for matching type.
            let source = (0..first_kv_shared_layer_idx).rev()
                .find(|&i| layer_types.get(i).copied().unwrap_or(true) == want);
            kv_shared_source[layer_idx] = source;
        }

        Ok(Self {
            config, tokenizer, common, global, extras, backend,
            embed_scale, per_layer_dim, kv_shared_source,
        })
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Tensor> {
        let cfg = &self.config;
        let backend = &*self.backend;
        let seq = tokens.len();
        let past = kv.len;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let n_rep = n_h / n_kv;
        let _ = n_rep;
        // Gemma 3n hardcodes attention scale to 1.0 (NOT 1/sqrt(head_dim)).
        // Per llama.cpp's `llama_model_gemma3n::load_arch_hparams`:
        //   hparams.f_attention_scale = 1.0f;
        // and the same value is passed to `build_attn`. Using 1/sqrt(256) = 1/16
        // would shrink attention logits 16× and badly degrade output.
        let scale = 1.0f32;
        let embed_dim = cfg.embedding_dim;
        let n_layers = cfg.n_layers;
        let pld = self.per_layer_dim;

        // ----- Initial embedding + scaling -----
        let mut hidden = backend.embed_lookup(&self.common.tok_embd, tokens, embed_dim);
        scale_inplace(backend, &mut hidden, self.embed_scale);

        // ----- Compute per-layer inputs (lookup + projection, combined per layer) -----
        // per_layer_inputs[seq, layer, per_layer_dim] = (per_layer_lookup + projected) / sqrt(2)
        let per_layer_inputs = self.compute_per_layer_inputs(&hidden, tokens, seq, n_layers, pld);

        // ----- AltUp expand: 1 stream → 4 streams with magnitude normalization -----
        // Streams live as a single stacked `[n_alt, seq, hidden]` tensor for the
        // whole forward, so AltUp predict/correct can pass it straight to the
        // fused kernel without per-call stack/unstack memcpys.
        let target_magnitude = compute_per_token_rms(&hidden, seq, embed_dim);
        let mut streams = backend.alloc_zeros(vec![ALTUP_NUM_INPUTS, seq, embed_dim]);
        backend.copy_axis0_into(&mut streams, 0, &hidden);
        for i in 0..(ALTUP_NUM_INPUTS - 1) {
            let projected = backend.linear(&hidden, &self.global.altup_proj[i]);
            let stream_i = rescale_to_target_rms(backend, projected, &target_magnitude, seq, embed_dim);
            backend.copy_axis0_into(&mut streams, i + 1, &stream_i);
        }

        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        // ----- Per-layer forward -----
        for layer in 0..n_layers {
            let b = &self.common.blocks[layer];
            let gx = &self.extras[layer];

            // -- AltUp predict --
            let active_stream = extract_stream(backend, &streams, ALTUP_ACTIVE_IDX, seq, embed_dim);
            let modalities = compute_router_modalities(
                backend, &active_stream, &gx.altup_router_norm, &gx.altup_router, embed_dim,
            );
            // `predictions` stays stacked [n_alt, seq, hidden] — fused kernel
            // reads `streams` and writes the predictions in one launch.
            let predictions = altup_predict_stacked(backend, &streams, &gx.altup_predict_coef, &modalities);
            let active_pred = extract_stream(backend, &predictions, ALTUP_ACTIVE_IDX, seq, embed_dim);

            // -- Pre-attn norm --
            let active_pred_norm = ops::rmsnorm(backend, &active_pred, &b.attn_norm, cfg.rms_eps);

            // -- Laurel parallel branch --
            let laurel_low = backend.linear(&active_pred_norm, &gx.laurel_l);   // [seq, laurel_rank]
            let laurel_hi  = backend.linear(&laurel_low, &gx.laurel_r);          // [seq, embed_dim]
            let laurel_normed = ops::rmsnorm(backend, &laurel_hi, &gx.laurel_post_norm, cfg.rms_eps);
            // Upstream: `Laurel(x) = x + RMSNorm(W_r W_l x)`. We add the residual after.
            let laurel_output = add(backend, &active_pred_norm, &laurel_normed);

            // -- Self-Attention (Q-norm, K-norm, RoPE, post-attn-norm) --
            // Q is always computed (per-layer), even for shared-K/V layers.
            let q_flat = b.attn_q.linear(backend, &active_pred_norm);
            let q_3d = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let mut q = ops::rmsnorm(backend, &q_3d, &gx.attn_q_norm, cfg.rms_eps);
            let rope_type = cfg.arch.rope_type();
            // The local (sliding-window) layers rotate with base 10,000, as llama.cpp sets Gemma 3n
            // (`rope_freq_base_train_swa`) unless the GGUF says; only the global ones with `rope.freq_base`
            // (1,000,000). A shared-KV layer reads the cache of a layer of its own kind, rotated the same way.
            let theta = if cfg.layer_uses_sliding_window(layer) { cfg.rope_theta_swa.unwrap_or(10000.0) } else { cfg.rope_theta };
            ops::rope(backend, &mut q, &positions, hd, rope_type, theta);

            // K/V: compute fresh for non-shared layers; reuse from source layer's
            // KV cache for shared layers.
            let kv_layer_for_attn = match self.kv_shared_source[layer] {
                Some(src) => src,
                None => {
                    let k_flat = b.attn_k.linear(backend, &active_pred_norm);
                    let v_flat = b.attn_v.linear(backend, &active_pred_norm);
                    // V-norm: upstream's `with_scale=False` RMSNorm applied to the FLAT
                    // v projection (across all heads), non-shared layers only.
                    let v_flat_normed = ops::rmsnorm_no_scale(backend, &v_flat, cfg.rms_eps);

                    let k_3d = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
                    let v = v_flat_normed.reshape(vec![seq, n_kv, hd]).expect("v reshape");

                    let mut k = ops::rmsnorm(backend, &k_3d, &gx.attn_k_norm, cfg.rms_eps);
                    ops::rope(backend, &mut k, &positions, hd, rope_type, theta);

                    kv.append(backend, layer, &k, &v);
                    layer
                }
            };

            let kv_len = kv.len + seq;
            let sw = if cfg.layer_uses_sliding_window(layer) { cfg.sliding_window } else { None };
            let attn_out = ops::attention_swa(
                backend, &q,
                kv.k_buffer(kv_layer_for_attn), kv.v_buffer(kv_layer_for_attn),
                kv_len, scale, past, sw,
            );

            let attn_t = attn_out.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let attn_proj = b.attn_output.linear(backend, &attn_t);
            let attn_normed = ops::rmsnorm(backend, &attn_proj, &gx.attn_post_norm, cfg.rms_eps);

            // -- Combine: (active_pred + attn) + laurel, divided by sqrt(2) --
            let attn_gated = add(backend, &active_pred, &attn_normed);
            let mut attn_laurel = add(backend, &attn_gated, &laurel_output);
            scale_inplace(backend, &mut attn_laurel, 1.0 / 2.0_f32.sqrt());

            // -- FFN: pre-norm, GeGLU (approx-tanh GeLU), post-norm --
            let attn_norm_for_ffn = ops::rmsnorm(backend, &attn_laurel, &b.ffn_norm, cfg.rms_eps);
            let (ffn_gate, ffn_up) = match &b.ffn_pair {
                FfnPair::Split { gate, up } => (gate, up),
                FfnPair::Fused(_) => unreachable!(
                    "gemma3n loader force-splits ffn_pair to support per-layer activation sparsity"
                ),
            };
            let mut gate_proj = ffn_gate.linear(backend, &attn_norm_for_ffn);
            // Apply Gaussian-top-k activation sparsity on `gate_proj` before the
            // activation (Gemma 3n's per-layer aggressive sparsification — the
            // first 10 layers are 95% sparse). Skip layers whose multiplier is
            // -inf (the "no sparsity" sentinel from upstream).
            if let Some(scales) = &cfg.activation_sparsity_scale {
                let s = scales[layer];
                if s.is_finite() {
                    ops::gaussian_topk_inplace(backend, &mut gate_proj, s);
                }
            }
            let up = ffn_up.linear(backend, &attn_norm_for_ffn);
            let gated = backend.gelu_approx_mul(&gate_proj, &up);
            let ffn_out = b.ffn_down.linear(backend, &gated);
            let ffn_normed = ops::rmsnorm(backend, &ffn_out, &gx.ffn_post_norm, cfg.rms_eps);
            let attn_ffw_laurel_gated = add(backend, &attn_laurel, &ffn_normed);

            // -- AltUp correct --
            let modalities_corr = compute_router_modalities(
                backend, &attn_ffw_laurel_gated, &gx.altup_router_norm, &gx.altup_router, embed_dim,
            );
            // Reuses stacked `predictions`; writes back to a new stacked tensor.
            streams = altup_correct_stacked(
                backend, &predictions, &attn_ffw_laurel_gated, &gx.altup_correct_coef,
                &modalities_corr,
            );

            // -- PLE injection --
            // Plain element-wise multiply by `altup_correct_scale` per llama.cpp
            // (`ggml_mul(first_prediction, model.layers[il].altup_correct_scale)`).
            // The HF Python `scale_corrected_output` uses `(1 + scale)` but the
            // GGUF converter pre-folds the +1 baseline into the stored weights.
            let mut active_corrected = extract_stream(backend, &streams, ALTUP_ACTIVE_IDX, seq, embed_dim);
            ops::mul_inplace_broadcast_last(backend, &mut active_corrected, &gx.altup_correct_scale);
            let mut gated = gx.inp_gate.linear(backend, &active_corrected); // [seq, per_layer_dim]
            gated = ops::gelu_approx(backend, &gated);
            let layer_pli = ops::slice_axis1_2d(backend, &per_layer_inputs, layer);
            ops::mul_inplace(backend, &mut gated, &layer_pli);
            let projected_back = gx.proj.linear(backend, &gated);
            let normed = ops::rmsnorm(backend, &projected_back, &gx.post_norm, cfg.rms_eps);
            // streams[1..n_alt] += normed — single device launch instead of
            // (n_alt - 1) separate slice + add + copy_back round trips.
            ops::add_to_axis0_range(backend, &mut streams, 1, ALTUP_NUM_INPUTS - 1, &normed);
        }

        kv.commit(seq);

        // ----- AltUp collapse: project streams 1..3 back, magnitude-normalize, average -----
        let stream0 = extract_stream(backend, &streams, 0, seq, embed_dim);
        let target = compute_per_token_rms(&stream0, seq, embed_dim);
        let mut accum = stream0;
        for i in 1..ALTUP_NUM_INPUTS {
            let stream_i = extract_stream(backend, &streams, i, seq, embed_dim);
            let projected = backend.linear(&stream_i, &self.global.altup_unembd_proj[i - 1]);
            let rescaled = rescale_to_target_rms(backend, projected, &target, seq, embed_dim);
            ops::add_inplace(backend, &mut accum, &rescaled);
        }
        scale_inplace(backend, &mut accum, 1.0 / ALTUP_NUM_INPUTS as f32);

        let normed = ops::rmsnorm(backend, &accum, &self.common.output_norm, cfg.rms_eps);
        let n_tok = normed.dim(0);
        let normed_last = if n_tok > 1 {
            backend.slice_axis0_range(&normed, n_tok - 1, 1)
        } else { normed };
        let logits = self.common.output.linear(backend, &normed_last);

        Ok(logits)
    }

    /// Compute `per_layer_inputs[seq, n_layers, per_layer_dim]` as a device
    /// tensor. Per-layer slicing in the main forward loop reads slices in one
    /// device kernel launch via `ops::slice_axis1_2d` — no host roundtrips.
    /// Combination: `(per_layer_projection + per_layer_lookup * sqrt(pld)) / sqrt(2)`.
    fn compute_per_layer_inputs(
        &self,
        hidden: &Tensor,
        tokens: &[u32],
        seq: usize,
        n_layers: usize,
        pld: usize,
    ) -> Tensor {
        let backend = &*self.backend;
        let cfg = &self.config;
        let row_size = n_layers * pld;

        // 1. Per-layer projection + scale by 1/sqrt(embed_dim).
        let mut projection = backend.linear(hidden, &self.global.per_layer_model_proj);
        scale_inplace(backend, &mut projection, 1.0 / (cfg.embedding_dim as f32).sqrt());

        // 2. RMSNorm — reshape [seq, n_layers*pld] -> [seq*n_layers, pld].
        let projection_2d = projection
            .reshape(vec![seq * n_layers, pld])
            .expect("per-layer projection reshape");
        let projection_normed = ops::rmsnorm(backend, &projection_2d, &self.global.per_layer_proj_norm, cfg.rms_eps);

        // 3. Lookup PLE rows on device.
        let mut lookup = backend.embed_lookup(&self.global.per_layer_token_embd, tokens, row_size);

        // 4. Scale lookup by sqrt(pld).
        scale_inplace(backend, &mut lookup, (pld as f32).sqrt());

        // 5. lookup += projection_normed (reshape projection back to [seq, row_size]).
        let projection_back = projection_normed
            .reshape(vec![seq, row_size])
            .expect("projection back-reshape");
        ops::add_inplace(backend, &mut lookup, &projection_back);

        // 6. Final scale by 1/sqrt(2).
        scale_inplace(backend, &mut lookup, 1.0 / 2.0_f32.sqrt());

        // Reshape to 3D for axis-1 slicing per layer in the forward loop.
        lookup.reshape(vec![seq, n_layers, pld]).expect("per_layer_inputs 3D reshape")
    }
}

/// Split a `[3*N, K]` tensor into 3 `[N, K]` tensors along axis 0, each uploaded
/// to the backend.
fn split_3way(t: Tensor, n: usize, backend: &dyn Backend) -> Vec<Tensor> {
    debug_assert_eq!(t.dim(0), 3 * n, "expected [3*N, K], got {:?}", t.shape());
    let k = t.dim(1);
    let data = t.data();
    let mut out = Vec::with_capacity(3);
    for i in 0..3 {
        let start = i * n * k;
        let end = (i + 1) * n * k;
        let slice = data[start..end].to_vec();
        let sub = Tensor::from_vec(slice, vec![n, k]);
        out.push(backend.to_device(sub));
    }
    out
}

/// In-place scalar multiply: `t *= s`. Thin wrapper kept for call-site
/// readability; delegates to `ops::mul_scalar_inplace`.
fn scale_inplace(backend: &dyn Backend, t: &mut Tensor, s: f32) {
    ops::mul_scalar_inplace(backend, t, s);
}

/// `c = a + b`. Allocates a new tensor.
fn add(backend: &dyn Backend, a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = a.clone();
    ops::add_inplace(backend, &mut out, b);
    out
}

/// Compute per-token RMS magnitude: `sqrt(mean(x^2))` along the last axis.
/// Returns a host-side `Vec<f32>` of length `seq`. We do this on host because
/// the magnitudes are small and used for per-row scaling that we also compute on
/// host.
fn compute_per_token_rms(t: &Tensor, seq: usize, hidden: usize) -> Vec<f32> {
    let host = t.to_host();
    let data = host.data();
    let mut out = vec![0.0f32; seq];
    for s in 0..seq {
        let off = s * hidden;
        let mut sum_sq = 0.0f32;
        for h in 0..hidden { sum_sq += data[off + h] * data[off + h]; }
        out[s] = (sum_sq / hidden as f32).sqrt();
    }
    out
}

/// Rescale `t` per-token to match `target_rms`, with a 1e-5 floor on the
/// computed RMS to avoid division by zero. Mirrors upstream's
/// `current * target_magnitude / new_magnitude` step in altup expand/collapse.
fn rescale_to_target_rms(
    backend: &dyn Backend,
    t: Tensor,
    target_rms: &[f32],
    seq: usize,
    hidden: usize,
) -> Tensor {
    let mut host = t.to_host();
    let host_data = host.data_mut();
    for s in 0..seq {
        let off = s * hidden;
        let mut sum_sq = 0.0f32;
        for h in 0..hidden { sum_sq += host_data[off + h] * host_data[off + h]; }
        let new_rms = (sum_sq / hidden as f32).sqrt().max(1e-5);
        let factor = target_rms[s] / new_rms;
        for h in 0..hidden { host_data[off + h] *= factor; }
    }
    backend.to_device(host)
}

/// Compute router modalities: `tanh(altup_router(router_norm(x) * router_input_scale))`.
/// `router_input_scale = 1 / hidden_size`. Result shape: `[seq, num_inputs]`.
fn compute_router_modalities(
    backend: &dyn Backend,
    x: &Tensor,
    router_norm: &Tensor,
    router: &Tensor,
    hidden: usize,
) -> Tensor {
    let mut router_input = ops::rmsnorm(backend, x, router_norm, 1e-6);
    scale_inplace(backend, &mut router_input, 1.0 / hidden as f32);
    let mut modalities = backend.linear(&router_input, router);
    ops::tanh_inplace(backend, &mut modalities);
    modalities
}

/// AltUp predict step (stacked-streams variant). Both `streams` and the returned
/// predictions are `[n_alt, seq, hidden]` tensors that live on the device for
/// the whole forward — no per-call stack/unstack memcpys.
fn altup_predict_stacked(
    backend: &dyn Backend,
    streams: &Tensor,
    predict_coef: &Tensor,
    modalities: &Tensor,
) -> Tensor {
    let coefs = backend.linear(modalities, predict_coef); // [seq, n_alt²]
    backend.altup_predict(streams, &coefs, ALTUP_NUM_INPUTS)
}

/// AltUp correct step (stacked-streams variant). `predictions` and the returned
/// corrected tensor are both `[n_alt, seq, hidden]` device tensors.
fn altup_correct_stacked(
    backend: &dyn Backend,
    predictions: &Tensor,
    activated: &Tensor,
    correct_coef: &Tensor,
    modalities: &Tensor,
) -> Tensor {
    let coefs = backend.linear(modalities, correct_coef); // [seq, n_alt]
    backend.altup_correct(predictions, activated, &coefs, ALTUP_NUM_INPUTS, ALTUP_ACTIVE_IDX)
}

/// Extract one stream from a `[n_alt, seq, hidden]` stacked tensor as a 2D
/// `[seq, hidden]` device tensor. CUDA backend uses a single d2d memcpy.
fn extract_stream(backend: &dyn Backend, stacked: &Tensor, idx: usize, seq: usize, hidden: usize) -> Tensor {
    backend
        .slice_axis0_range(stacked, idx, 1)
        .reshape(vec![seq, hidden])
        .expect("stream slice reshape")
}

