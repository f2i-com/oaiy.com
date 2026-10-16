//! Gemma 4 — Apache-2.0-licensed dense forward pass.
//!
//! Architecturally a stripped-down Gemma 3n: keeps **Per-Layer Embeddings (PLE)**
//! and shared K/V layers, drops AltUp 4-stream routing and Laurel low-rank
//! residual. Per-layer head_dim (256 SWA / 512 global), per-layer RoPE base
//! (10K SWA / 1M global), per-layer FFN size, sliding-window pattern via bool
//! array, final logit softcap (=30 typical), `f_attention_scale = 1.0`.
//!
//! **Status: produces coherent output.** All architecture-level pieces wired:
//! PLE injection, shared K/V, V-norm, attention scale=1.0, final softcap,
//! correct `<|turn>` / `<turn|>` chat template, `rope_freqs` long-rope scaling
//! on global layers (via `ops::rope_with_factors`), and the per-block
//! `layer_output_scale` scalar multiplier (which was the final missing piece —
//! the model was gibberish without it even after everything else was in
//! place). All matched against `llama.cpp/src/models/gemma4.cpp`.
//!
//! Per-layer block sequence (per `llama.cpp/src/models/gemma4.cpp`):
//! ```text
//!   cur = rmsnorm(inpL, attn_norm)
//!   Q = wq @ cur, q_norm, RoPE (with rope_freqs only on global layers)
//!   K, V = wk, wv @ cur, k_norm, V is rms-normed (no scale), RoPE on K
//!   cur = attention(Q, K, V, scale=1.0)
//!   cur = rmsnorm(cur, attn_post_norm)
//!   attn_out = cur + inpL                                  // residual
//!   cur = rmsnorm(attn_out, ffn_norm)
//!   cur = ffn_gelu(cur)
//!   cur = rmsnorm(cur, ffn_post_norm)
//!   cur = cur + attn_out                                   // residual
//!   if PLE:                                                // Per-Layer Embedding injection
//!     pe_in = cur
//!     cur = per_layer_inp_gate @ cur, gelu, * inp_per_layer[il], per_layer_proj @, per_layer_post_norm
//!     cur = pe_in + cur
//!   if layer_output_scale: cur *= layer_output_scale       // TODO
//!   inpL = cur
//! ```
//! Final: rmsnorm(output_norm) → output projection → tanh-softcap.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::llm::kv_cache::KvCache;
use crate::llm::loader::{CommonTensors, TensorIndex, Weight};
use crate::llm::{LlamaError, ModelConfig, Result};

#[derive(Debug)]
pub struct Gemma4GlobalTensors {
    /// `[n_layers * per_layer_dim, embed_dim]` — projects hidden state to per-layer signal.
    pub per_layer_model_proj: Tensor,
    /// `[per_layer_dim]` — RMSNorm scale for the projected signal.
    pub per_layer_proj_norm:  Tensor,
    /// The PLE table, `[vocab_size, n_layers * per_layer_dim]`, as the GGUF packs it; a forward decodes its tokens'
    /// rows ([`PleTable::rows`]).
    pub per_layer_token_embd: PleTable,
    /// Per-frequency-dim divisors for RoPE (long-rope scaling). Length is
    /// `head_dim_global / 2`. Applied only on global (non-SWA) layers per
    /// `llama.cpp/src/models/gemma4.cpp`.
    pub rope_freq_factors: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct Gemma4BlockExtras {
    pub attn_post_norm:        Tensor,
    pub ffn_post_norm:         Tensor,
    pub attn_q_norm:           Tensor,
    pub attn_k_norm:           Tensor,
    /// PLE per-block tensors (None ⇒ layer skips PLE injection — every Gemma 4
    /// layer has these in practice, but we keep them optional for safety).
    pub per_layer_inp_gate:    Option<Weight>,    // `[per_layer_dim, embed_dim]`
    pub per_layer_proj:        Option<Weight>,    // `[embed_dim, per_layer_dim]`
    pub per_layer_post_norm:   Option<Tensor>,    // `[embed_dim]`
    /// Per-block scalar (1-element F32) multiplied into the layer's residual
    /// output. Loaded as a host scalar; applied via element-wise scale.
    pub layer_output_scale:    Option<f32>,
}

pub struct Gemma4Model {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub common:    CommonTensors,
    pub global:    Option<Gemma4GlobalTensors>,
    pub extras:    Vec<Gemma4BlockExtras>,
    pub backend:   Arc<dyn Backend>,
    embed_scale_const: Option<f32>,
    per_layer_dim:     usize,
    /// For each layer: `Some(src)` if K/V should be reused from layer `src`,
    /// `None` if this layer computes its own K/V. Same convention as Gemma 3n.
    /// Last `shared_kv_layers` layers reuse the latest earlier non-shared layer
    /// of the same SWA type.
    kv_shared_source:  Vec<Option<usize>>,
    /// the multimodal soft tokens (`<|image|>`, `<|audio|>`, `<|video|>`), whose PLE rows are the
    /// padding's (as Transformers' `Gemma4Model` looks them up).
    soft_token_ids:    Vec<u32>,
    /// the token embeddings as the GGUF packs them, where the f32 table was let go (see `from_gguf`).
    tok_rows:          Option<PleTable>,
    /// the layers chained on the backend's device, where it has a chain.
    chain:             crate::llm::chain_gemma4::Gemma4Chain,
}

impl std::fmt::Debug for Gemma4Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4Model")
            .field("config", &self.config)
            .field("backend", &self.backend.name())
            .field("vocab_size", &self.tokenizer.vocab_size())
            .finish()
    }
}

impl Gemma4Model {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        // MoE detection: Gemma 4 26B-A4B uses arch=gemma4 with `gemma4.expert_count > 0`.
        // Routed at the Model::load level to Gemma4MoeModel; this branch should
        // not be reached in practice, but stays as a defensive guard.
        if let Ok(n_experts) = g.get_u64("gemma4.expert_count") {
            if n_experts > 0 {
                return Err(crate::llm::LlamaError::Config(format!(
                    "Gemma 4 MoE detected ({n_experts} experts) — Gemma4Model is the dense loader; \
                     should have been routed to Gemma4MoeModel via Model::load."
                )));
            }
        }
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        let mut common = CommonTensors::load_with_index(&idx, &config)?.upload_to(&*backend);
        // where the head is a matmul of its own (a backend that keeps a packed head), the embeddings are
        // read from the packed table a token's row at a time, and the f32 table is let go.
        let tok_rows = match (&common.output, common.tok_embd.is_device()) {
            (Weight::TiedEmbed(_), _) | (_, true) => None,
            _ => PleTable::load_named(g, "token_embd.weight", config.vocab_size, config.embedding_dim).ok(),
        };
        if tok_rows.is_some() {
            common.tok_embd = Arc::new(Tensor::from_vec(Vec::new(), vec![0, config.embedding_dim]));
        }

        // PLE config + globals (optional — small Gemma 4 variants might omit).
        let per_layer_dim = g.get_u64("gemma4.embedding_length_per_layer_input")
            .unwrap_or(0) as usize;

        let global = if per_layer_dim > 0 {
            let per_layer_token_embd = PleTable::load(g, config.vocab_size, config.n_layers * per_layer_dim)?;

            let per_layer_model_proj = backend.to_device(idx.take("per_layer_model_proj.weight", &[])?);
            let per_layer_proj_norm  = backend.to_device(idx.take("per_layer_proj_norm.weight",  &[])?);

            // RoPE long-rope factors. Stored as 1D F32 of length head_dim_global/2.
            // Optional — small variants might omit. Only used on global layers.
            let rope_freq_factors = idx.take("rope_freqs.weight", &[]).ok().map(|t| {
                let host = t.to_host();
                host.data().to_vec()
            });

            Some(Gemma4GlobalTensors {
                per_layer_model_proj,
                per_layer_proj_norm,
                per_layer_token_embd,
                rope_freq_factors,
            })
        } else {
            None
        };

        let mut extras = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let hd_layer = config.layer_head_dim(i);

            let per_layer_inp_gate = if global.is_some() {
                idx.take_weight(&format!("blk.{i}.inp_gate.weight"), &[])
                    .ok().map(|w| w.try_to_device(&*backend, 2 * 1024 * 1024 * 1024))
            } else { None };
            let per_layer_proj = if global.is_some() {
                idx.take_weight(&format!("blk.{i}.proj.weight"), &[])
                    .ok().map(|w| w.try_to_device(&*backend, 2 * 1024 * 1024 * 1024))
            } else { None };
            let per_layer_post_norm = if global.is_some() {
                idx.take(&format!("blk.{i}.post_norm.weight"), &[])
                    .ok().map(|t| backend.to_device(t))
            } else { None };
            // layer_output_scale: 1-element F32 scalar per block.
            let layer_output_scale = idx.take(&format!("blk.{i}.layer_output_scale.weight"), &[])
                .ok()
                .and_then(|t| t.data().first().copied());

            extras.push(Gemma4BlockExtras {
                attn_post_norm: backend.to_device(idx.take(
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &[&format!("blk.{i}.attn_post_norm.weight")],
                )?),
                ffn_post_norm: backend.to_device(idx.take(
                    &format!("blk.{i}.post_ffw_norm.weight"),
                    &[&format!("blk.{i}.ffn_post_norm.weight")],
                )?),
                attn_q_norm: backend.to_device(load_or_unit(
                    &idx, &format!("blk.{i}.attn_q_norm.weight"), hd_layer,
                )?),
                attn_k_norm: backend.to_device(load_or_unit(
                    &idx, &format!("blk.{i}.attn_k_norm.weight"), hd_layer,
                )?),
                per_layer_inp_gate,
                per_layer_proj,
                per_layer_post_norm,
                layer_output_scale,
            });
        }

        let embed_scale_const = if config.embedding_scale {
            Some((config.embedding_dim as f32).sqrt())
        } else {
            None
        };

        // Shared K/V: last `shared_kv_layers` (=20 for Gemma 4 E2B) reuse the
        // latest earlier non-shared layer of the same SWA type.
        let num_kv_shared = g.get_u64("gemma4.attention.shared_kv_layers").unwrap_or(0) as usize;
        let first_kv_shared_layer_idx = config.n_layers.saturating_sub(num_kv_shared);
        let layer_types: Vec<bool> = config.sliding_window_layers
            .clone()
            .unwrap_or_else(|| (0..config.n_layers).map(|i| (i + 1) % config.sliding_window_pattern != 0).collect());
        let mut kv_shared_source = vec![None; config.n_layers];
        for layer_idx in first_kv_shared_layer_idx..config.n_layers {
            let want = layer_types.get(layer_idx).copied().unwrap_or(true);
            let source = (0..first_kv_shared_layer_idx).rev()
                .find(|&i| layer_types.get(i).copied().unwrap_or(true) == want);
            kv_shared_source[layer_idx] = source;
        }

        let soft_token_ids = ["<|image|>", "<|audio|>", "<|video|>"].iter().filter_map(|t| tokenizer.token_id(t)).collect();
        Ok(Self {
            config, tokenizer, common, global, extras, backend, embed_scale_const, per_layer_dim,
            kv_shared_source, soft_token_ids, tok_rows, chain: Default::default(),
        })
    }

    /// the width of a layer's PLE input (0: none).
    pub(crate) fn per_layer_dim(&self) -> usize { self.per_layer_dim }

    /// the layer whose cache layer `l`'s attention reads (itself unless it shares an earlier one's).
    pub(crate) fn kv_source(&self, l: usize) -> usize { self.kv_shared_source[l].unwrap_or(l) }

    /// the token whose PLE row each of `tokens` reads: a multimodal soft token reads the padding's.
    pub fn ple_ids(&self, tokens: &[u32]) -> Vec<u32> {
        tokens.iter().map(|t| if self.soft_token_ids.contains(t) { 0 } else { *t }).collect()
    }

    /// `tokens` with `soft` rows (`(position in tokens, [n, d] values)`) in place of their
    /// embeddings (a picture's soft tokens), through the model: the last row's logits. Chained on the device where
    /// the backend can, else op by op.
    pub fn forward_spliced(&self, tokens: &[u32], soft: &[(usize, &[f32])], kv: &mut KvCache) -> Tensor {
        let d = self.config.embedding_dim;
        let embeds = self.embed_text(tokens);
        let mut host = if embeds.is_device() { embeds.to_host() } else { embeds };
        for (at, rows) in soft {
            let n = rows.len() / d;
            assert!(rows.len() % d == 0 && at + n <= tokens.len(), "soft tokens of width {d} within the tokens");
            host.data_mut()[at * d..(at + n) * d].copy_from_slice(rows);
        }
        let ple = self.ple_ids(tokens);
        if let Some(logits) = self.chain.run(self, host.data(), &ple, kv) {
            return logits;
        }
        let embeds = self.backend.to_device(host);
        self.forward_embeds(&embeds, tokens, kv)
    }

    /// Look up token embeddings + apply Gemma's `sqrt(d)` scaling. Mirrors
    /// `Gemma3Model::embed_text` for use in the multimodal splice path.
    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        let backend = &*self.backend;
        let cfg = &self.config;
        let mut x = match &self.tok_rows {
            Some(rows) => backend.to_device(Tensor::from_vec(rows.rows(tokens), vec![tokens.len(), cfg.embedding_dim])),
            None => backend.embed_lookup(&self.common.tok_embd, tokens, cfg.embedding_dim),
        };
        if let Some(s) = self.embed_scale_const {
            ops::mul_scalar_inplace(backend, &mut x, s);
        }
        x
    }

    /// Standard text-only forward. Composes [`embed_text`] + [`forward_embeds`].
    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        self.forward_spliced(tokens, &[], kv)
    }

    /// Vision-language helper: tokenize `prompt` (which must contain exactly
    /// one `placeholder_token_id` marker — typically `<|image|>` = 258880 for
    /// Gemma 4), expand the placeholder into `soft_tokens.dim(0)` repetitions
    /// of the same token id, embed the resulting sequence, and splice the
    /// vision soft tokens into the placeholder positions.
    ///
    /// Returns `(tokens, embeds)` ready to be fed into [`forward_embeds`]:
    /// the tokens drive PLE lookup (placeholder positions
    /// read the padding's row, as Transformers' `Gemma4Model` does), and the
    /// embeds carry the actual image content. [`Self::forward_spliced`] does
    /// both in one, chained on a device where it can be.
    pub fn embed_with_vision_at_placeholder(
        &self,
        prompt:               &str,
        soft_tokens:          &Tensor,
        placeholder_token_id: u32,
        add_bos:              bool,
    ) -> Result<(Vec<u32>, Tensor)> {
        let prompt_ids = self.tokenizer.encode(prompt, add_bos)?;
        let pos = prompt_ids.iter().position(|&id| id == placeholder_token_id)
            .ok_or_else(|| crate::llm::LlamaError::Config(format!(
                "placeholder token id {placeholder_token_id} not found in prompt"
            )))?;

        let n_soft = soft_tokens.dim(0);
        let d = self.config.embedding_dim;
        debug_assert_eq!(soft_tokens.dim(soft_tokens.rank() - 1), d,
            "soft-token width {} doesn't match LM hidden dim {d}",
            soft_tokens.dim(soft_tokens.rank() - 1));

        // Build the expanded token sequence: pre + N×placeholder + post.
        let mut tokens = Vec::with_capacity(prompt_ids.len() - 1 + n_soft);
        tokens.extend_from_slice(&prompt_ids[..pos]);
        tokens.extend(std::iter::repeat(placeholder_token_id).take(n_soft));
        tokens.extend_from_slice(&prompt_ids[pos + 1..]);

        // Embed normally; this gives [seq, d] with placeholder embeddings at
        // the vision positions. Then overwrite those rows with the projected
        // soft tokens so the LM sees the actual image content there.
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

    /// Run the transformer stack on a pre-computed embedding tensor. Unlike
    /// Gemma 3's variant, this also takes `tokens` because Gemma 4's PLE
    /// (Per-Layer Embeddings) is keyed by token id — the caller passes the
    /// token sequence (with placeholder ids at vision positions, e.g.
    /// `<image_soft_token>` = 262144) and the embeddings (with vision soft
    /// tokens spliced into those positions). PLE lookup uses tokens; the
    /// main hidden state uses embeds.
    pub fn forward_embeds(
        &self,
        embeds: &Tensor,
        tokens: &[u32],
        kv:     &mut KvCache,
    ) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let seq = tokens.len();
        debug_assert_eq!(embeds.dim(0), seq,
            "embeds first dim {} doesn't match tokens.len() {seq}", embeds.dim(0));
        let past = kv.len;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let n_layers = cfg.n_layers;
        let pld = self.per_layer_dim;

        // Take ownership-equivalent of the passed-in embeddings. Clone is
        // device-aware (`Tensor::clone` calls `clone_to_device` on CUDA), so
        // this stays on-device — no host round-trip.
        let mut x = embeds.clone();

        // Per-layer inputs: computed once at start as a device tensor of shape
        // `[seq, n_layers, pld]`. The forward loop reads each layer's slice via
        // `ops::slice_axis1_2d` — one device kernel per layer, no host syncs.
        let per_layer_inputs: Option<Tensor> = self.global.as_ref()
            .map(|g| compute_per_layer_inputs(backend, &x, &self.ple_ids(tokens), &g.per_layer_token_embd,
                                              &g.per_layer_model_proj, &g.per_layer_proj_norm,
                                              cfg.embedding_dim, seq, n_layers, pld, cfg.rms_eps));

        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..n_layers {
            let b = &self.common.blocks[layer];
            let gx = &self.extras[layer];

            let hd = cfg.layer_head_dim(layer);
            // Gemma 4 (and 3n) hardcode attention scale to 1.0, NOT 1/sqrt(head_dim).
            // Per llama.cpp/src/models/gemma4.cpp: hparams.f_attention_scale = 1.0f.
            let scale = 1.0f32;
            let rope_theta = cfg.layer_rope_theta(layer);

            // ----- pre-attn norm + Q (always computed) -----
            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);

            // RoPE freq factors: applied ONLY on global (non-SWA) layers per
            // llama.cpp's `freq_factors = is_swa(il) ? nullptr : rope_freqs`.
            let is_swa = cfg.layer_uses_sliding_window(layer);
            let freq_factors: Option<&[f32]> = if !is_swa {
                self.global.as_ref().and_then(|g| g.rope_freq_factors.as_deref())
            } else { None };

            let q_flat = b.attn_q.linear(backend, &xn);
            let q_3d = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let mut q = ops::rmsnorm(backend, &q_3d, &gx.attn_q_norm, cfg.rms_eps);
            let rope_type = cfg.arch.rope_type();
            ops::rope_with_factors(backend, &mut q, &positions, hd, rope_type, rope_theta, freq_factors);

            // K/V: compute fresh for non-shared layers; reuse from `src` for shared.
            // Per llama.cpp's `has_kv(il)` check: layers `[0, n_layer - shared_kv_layers)`
            // store their own K/V; later layers leave the buffers untouched and
            // attention reads the source layer's cache.
            let kv_layer_for_attn = match self.kv_shared_source[layer] {
                Some(src) => src,
                None => {
                    let k_flat = b.attn_k.linear(backend, &xn);
                    let v_flat = b.attn_v.linear(backend, &xn);

                    // V-norm: per llama.cpp, Vcur = ggml_rms_norm(Vcur, eps) BEFORE
                    // reshape — no learnable scale weight, just unit RMS.
                    let v_normed = ops::rmsnorm_no_scale(backend, &v_flat, cfg.rms_eps);

                    let k_3d = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
                    let v = v_normed.reshape(vec![seq, n_kv, hd]).expect("v reshape");

                    let mut k = ops::rmsnorm(backend, &k_3d, &gx.attn_k_norm, cfg.rms_eps);
                    ops::rope_with_factors(backend, &mut k, &positions, hd, rope_type, rope_theta, freq_factors);

                    kv.append(backend, layer, &k, &v);
                    layer
                }
            };

            let kv_len = kv.len + seq;
            let sw = if cfg.layer_uses_sliding_window(layer) {
                cfg.sliding_window
            } else {
                None
            };
            let attn_out = ops::attention_swa(
                backend, &q,
                kv.k_buffer(kv_layer_for_attn), kv.v_buffer(kv_layer_for_attn),
                kv_len, scale, past, sw,
            );

            let attn_t = attn_out.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let mut attn_proj = b.attn_output.linear(backend, &attn_t);
            attn_proj = ops::rmsnorm(backend, &attn_proj, &gx.attn_post_norm, cfg.rms_eps);

            // attn_out = cur + inpL (the input to attn norm, i.e., x before this layer's modifications)
            let mut attn_added = x.clone();
            ops::add_inplace(backend, &mut attn_added, &attn_proj);

            // ----- FFN: GeGLU with approximate-tanh GeLU -----
            let xn2 = ops::rmsnorm(backend, &attn_added, &b.ffn_norm, cfg.rms_eps);
            let gated = b.ffn_pair.geglu(backend, &xn2);
            let mut ffn_out = b.ffn_down.linear(backend, &gated);
            ffn_out = ops::rmsnorm(backend, &ffn_out, &gx.ffn_post_norm, cfg.rms_eps);

            // FFN residual: cur = ffn_out + attn_added. attn_added is unused
            // after this point in the layer, so move it instead of cloning.
            let mut cur = attn_added;
            ops::add_inplace(backend, &mut cur, &ffn_out);

            // ----- PLE injection (after FFN+residual) -----
            // `inp_gate_w.linear(&cur)` borrows `cur`, so we don't need the
            // pre-PLE snapshot — just accumulate `normed` into `cur` directly.
            if let (Some(pli_dev), Some(inp_gate_w), Some(proj_w), Some(post_norm_t))
                = (per_layer_inputs.as_ref(),
                   gx.per_layer_inp_gate.as_ref(),
                   gx.per_layer_proj.as_ref(),
                   gx.per_layer_post_norm.as_ref())
            {
                let mut gated = inp_gate_w.linear(backend, &cur);  // [seq, per_layer_dim]
                gated = ops::gelu_approx(backend, &gated);
                let layer_pli = ops::slice_axis1_2d(backend, pli_dev, layer);
                ops::mul_inplace(backend, &mut gated, &layer_pli);
                let projected_back = proj_w.linear(backend, &gated);
                let normed = ops::rmsnorm(backend, &projected_back, post_norm_t, cfg.rms_eps);
                ops::add_inplace(backend, &mut cur, &normed);
            }

            // Per-block residual scalar. Per llama.cpp gemma4.cpp:
            //   if (model.layers[il].out_scale) cur = ggml_mul(cur, out_scale);
            if let Some(s) = gx.layer_output_scale {
                ops::mul_scalar_inplace(backend, &mut cur, s);
            }

            x = cur;
        }

        kv.commit(seq);

        let x = ops::rmsnorm(backend, &x, &self.common.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        let mut logits = self.common.output.linear(backend, &x_last);

        // Final logit softcap, on device.
        if let Some(softcap) = cfg.final_logit_softcap {
            ops::mul_scalar_inplace(backend, &mut logits, 1.0 / softcap);
            ops::tanh_inplace(backend, &mut logits);
            ops::mul_scalar_inplace(backend, &mut logits, softcap);
        }
        logits
    }
}

/// a table of rows as the GGUF packs it (a view of the mapped file, or its bytes from a streamed one), a
/// token's row decoded when a forward needs it: Gemma's per-layer token table, and its token embeddings where the head
/// is a packed matmul of its own. Gemma 4 E2B's PLE table is 262,144 rows of 8,960: 1.6 GB as Q5_K, 9.4 GB as the f32
/// the model held on the host before, which took an engine of it past a 16 GB machine's memory; its embeddings were
/// 1.6 GB more of f32.
pub struct PleTable {
    packed: ggml_rs::QuantizedTensor,
    vocab: usize,
    row: usize,
    row_bytes: usize,
}

impl std::fmt::Debug for PleTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PleTable({} rows of {}, {:?})", self.vocab, self.row, self.packed.dtype())
    }
}

impl PleTable {
    /// `per_layer_token_embd.weight`, `vocab` rows of `row`.
    pub fn load(g: &GgufFile, vocab: usize, row: usize) -> Result<Self> {
        Self::load_named(g, "per_layer_token_embd.weight", vocab, row)
    }

    /// The tensor `name`, `vocab` rows of `row`.
    pub fn load_named(g: &GgufFile, name: &str, vocab: usize, row: usize) -> Result<Self> {
        let info = g.tensor_by_name(name).ok_or_else(|| LlamaError::MissingTensor(name.into()))?;
        let dtype = info.dtype;
        if info.numel() as usize != vocab * row || row % dtype.block_size() != 0 || !ggml_quants::is_supported(dtype) {
            return Err(LlamaError::Config(format!(
                "{name}: {} values, not {vocab} rows of {row} in {dtype:?}'s blocks", info.numel()
            )));
        }
        let packed = crate::llm::loader::load_tensor_packed(g, info)?;
        Ok(Self { packed, vocab, row, row_bytes: row / dtype.block_size() * dtype.type_size() })
    }

    /// The width of a row (`n_layers * per_layer_dim`).
    pub fn row(&self) -> usize { self.row }

    /// `tokens`' rows, `[tokens.len(), row]`; a token past the table reads row 0 (the padding's).
    pub fn rows(&self, tokens: &[u32]) -> Vec<f32> {
        let bytes = self.packed.bytes();
        let mut out = vec![0f32; tokens.len() * self.row];
        for (t, dst) in tokens.iter().zip(out.chunks_mut(self.row)) {
            let id = if (*t as usize) < self.vocab { *t as usize } else { 0 };
            ggml_quants::dequantize(self.packed.dtype(), &bytes[id * self.row_bytes..(id + 1) * self.row_bytes], dst)
                .expect("a PLE row in its own blocks");
        }
        out
    }
}

/// Load an `[N]` norm weight, or substitute a unit-weight tensor of size `n` if
/// the GGUF doesn't include it. Lets us keep the forward pass unconditional.
pub fn load_or_unit(idx: &TensorIndex<'_>, name: &str, n: usize) -> Result<Tensor> {
    match idx.take(name, &[]) {
        Ok(t) => Ok(t),
        Err(LlamaError::MissingTensor(_)) => Ok(Tensor::from_vec(vec![1.0; n], vec![n])),
        Err(e) => Err(e),
    }
}

/// Compute `per_layer_inputs[seq, n_layers, per_layer_dim]` once per forward
/// as a device tensor. Per-layer slicing in the main loop reads it via
/// `ops::slice_axis1_2d` — a single device kernel launch, no host roundtrip.
///
/// Combination per upstream `project_per_layer_inputs`:
///   `(per_layer_lookup * sqrt(per_layer_dim) + RMSNorm(proj(hidden) / sqrt(embed_dim))) / sqrt(2)`.
pub fn compute_per_layer_inputs(
    backend: &dyn Backend,
    hidden: &Tensor,
    tokens: &[u32],
    table: &PleTable,
    per_layer_model_proj: &Tensor,
    per_layer_proj_norm: &Tensor,
    embed_dim: usize,
    seq: usize,
    n_layers: usize,
    pld: usize,
    rms_eps: f32,
) -> Tensor {
    let row_size = n_layers * pld;
    let inv_sqrt_embed = 1.0 / (embed_dim as f32).sqrt();
    let pl_scale       = (pld as f32).sqrt();
    let inv_sqrt2      = 1.0 / 2.0_f32.sqrt();

    // 1. Project hidden through per_layer_model_proj and scale by 1/sqrt(embed_dim).
    let mut projection = backend.linear(hidden, per_layer_model_proj);
    ops::mul_scalar_inplace(backend, &mut projection, inv_sqrt_embed);

    // 2. RMSNorm — reshape [seq, n_layers*pld] -> [seq*n_layers, pld] for per-layer normalisation.
    let projection_2d = projection
        .reshape(vec![seq * n_layers, pld])
        .expect("per-layer projection reshape");
    let projection_normed = ops::rmsnorm(backend, &projection_2d, per_layer_proj_norm, rms_eps);

    // 3. The tokens' PLE rows, decoded from the packed table: [seq, row_size].
    let mut lookup = backend.to_device(Tensor::from_vec(table.rows(tokens), vec![seq, row_size]));

    // 4. Scale lookup by sqrt(per_layer_dim).
    ops::mul_scalar_inplace(backend, &mut lookup, pl_scale);

    // 5. Combine: lookup += projection_normed (after reshape back to [seq, row_size]).
    let projection_back = projection_normed
        .reshape(vec![seq, row_size])
        .expect("projection back-reshape");
    ops::add_inplace(backend, &mut lookup, &projection_back);

    // 6. Final scale by 1/sqrt(2).
    ops::mul_scalar_inplace(backend, &mut lookup, inv_sqrt2);

    // Reshape to 3D for axis-1 slicing per layer in the forward loop.
    lookup.reshape(vec![seq, n_layers, pld]).expect("per_layer_inputs 3D reshape")
}
