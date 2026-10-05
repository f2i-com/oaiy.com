//! Gemma 3 dense forward pass.

use std::sync::Arc;

use ggml_rs::{ops, Backend, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{CommonTensors, TensorIndex};
use crate::{ModelConfig, Result};

#[derive(Debug)]
pub struct Gemma3BlockExtras {
    pub attn_post_norm: Tensor,
    pub ffn_post_norm:  Tensor,
    pub attn_q_norm:    Tensor,
    pub attn_k_norm:    Tensor,
}

pub struct Gemma3Model {
    pub config:    ModelConfig,
    pub tokenizer: Tokenizer,
    pub common:    CommonTensors,
    pub extras:    Vec<Gemma3BlockExtras>,
    pub backend:   Arc<dyn Backend>,
    /// Pre-computed `[1]`-shape device tensor holding `sqrt(embedding_dim)`,
    /// used to scale embeddings without round-tripping the constant per call.
    embed_scale_const: Option<f32>,
}

impl std::fmt::Debug for Gemma3Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma3Model")
            .field("config", &self.config)
            .field("backend", &self.backend.name())
            .field("vocab_size", &self.tokenizer.vocab_size())
            .finish()
    }
}

impl Gemma3Model {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);
        let common = CommonTensors::load_with_index(&idx, &config)?.upload_to(&*backend);

        let mut extras = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            extras.push(Gemma3BlockExtras {
                attn_post_norm: backend.to_device(idx.take(
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &[&format!("blk.{i}.attn_post_norm.weight")],
                )?),
                ffn_post_norm: backend.to_device(idx.take(
                    &format!("blk.{i}.post_ffw_norm.weight"),
                    &[&format!("blk.{i}.ffn_post_norm.weight")],
                )?),
                attn_q_norm: backend.to_device(idx.take(
                    &format!("blk.{i}.attn_q_norm.weight"),
                    &[],
                )?),
                attn_k_norm: backend.to_device(idx.take(
                    &format!("blk.{i}.attn_k_norm.weight"),
                    &[],
                )?),
            });
        }

        let embed_scale_const = if config.embedding_scale {
            Some((config.embedding_dim as f32).sqrt())
        } else {
            None
        };

        Ok(Self { config, tokenizer, common, extras, backend, embed_scale_const })
    }

    /// Look up token embeddings and apply Gemma 3's per-token `sqrt(d)` scaling.
    /// Returns `[seq, embedding_dim]` host or device tensor (matches the
    /// backend's preferred storage). Use this as the first step of a vision-
    /// language pipeline: embed the text tokens, splice in the vision
    /// soft-tokens at the placeholder position, then call [`forward_embeds`].
    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        let backend = &*self.backend;
        let cfg = &self.config;
        let mut x = backend.embed_lookup(&self.common.tok_embd, tokens, cfg.embedding_dim);
        if let Some(s) = self.embed_scale_const {
            ops::mul_scalar_inplace(backend, &mut x, s);
        }
        x
    }

    /// Standard text-only forward: tokens → logits. Composes [`embed_text`]
    /// and [`forward_embeds`].
    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
        let embeds = self.embed_text(tokens);
        self.forward_embeds(&embeds, kv)
    }

    /// Vision-language helper: concatenate `[embed_text(pre_tokens) ;
    /// vision_soft_tokens ; embed_text(post_tokens)]` into one
    /// `[seq, embedding_dim]` embedding stream ready for
    /// [`forward_embeds`]. The vision tokens come from
    /// `MmProj::forward(image)` and are inserted *unscaled* — only text
    /// embeddings get Gemma 3's `sqrt(d)` multiplier.
    pub fn embed_with_vision(
        &self,
        pre_tokens:  &[u32],
        soft_tokens: &Tensor,
        post_tokens: &[u32],
    ) -> Tensor {
        let cfg = &self.config;
        let d = cfg.embedding_dim;
        debug_assert_eq!(soft_tokens.dim(soft_tokens.rank() - 1), d,
            "vision soft-token width {} doesn't match LM hidden dim {d}",
            soft_tokens.dim(soft_tokens.rank() - 1));

        let pre  = self.embed_text(pre_tokens).to_host();
        let post = self.embed_text(post_tokens).to_host();
        let soft = soft_tokens.to_host();

        let n_pre  = pre_tokens.len();
        let n_soft = soft.dim(0);
        let n_post = post_tokens.len();
        let total  = n_pre + n_soft + n_post;

        let mut out = vec![0.0f32; total * d];
        out[..n_pre * d].copy_from_slice(pre.data());
        out[n_pre * d..(n_pre + n_soft) * d].copy_from_slice(soft.data());
        out[(n_pre + n_soft) * d..].copy_from_slice(post.data());

        self.backend.to_device(Tensor::from_vec(out, vec![total, d]))
    }

    /// Run the transformer stack on a pre-computed embedding tensor — same
    /// shape as the output of [`embed_text`] (`[seq, embedding_dim]`). The
    /// vision-language path uses this to splice projected image soft-tokens
    /// into the embedding stream before the transformer sees them.
    ///
    /// Note: vision soft-tokens come from the mmproj projector, which already
    /// emits values in the LM hidden space — the caller should NOT additionally
    /// scale them by `sqrt(d)`. Only text embeds get that scaling, and
    /// [`embed_text`] applies it for you.
    pub fn forward_embeds(&self, embeds: &Tensor, kv: &mut KvCache) -> Tensor {
        let cfg = &self.config;
        let backend = &*self.backend;
        let seq = embeds.dim(0);
        let past = kv.len;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let n_rep = cfg.n_rep();
        let scale = 1.0 / (hd as f32).sqrt();

        // Take ownership-equivalent: clone (device-aware) so we can mutate
        // x in place across the loop without disturbing the caller's tensor.
        let mut x = embeds.clone();

        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for layer in 0..cfg.n_layers {
            let b = &self.common.blocks[layer];
            let gx = &self.extras[layer];

            // ----- attention -----
            let xn = ops::rmsnorm(backend, &x, &b.attn_norm, cfg.rms_eps);

            let [q_flat, k_flat, v_flat]: [Tensor; 3] =
                crate::loader::Weight::linear_many(backend, &xn, &[&b.attn_q, &b.attn_k, &b.attn_v]).try_into().expect("q, k and v");

            let q_3d = q_flat.reshape(vec![seq, n_h, hd]).expect("q reshape");
            let k_3d = k_flat.reshape(vec![seq, n_kv, hd]).expect("k reshape");
            let v = v_flat.reshape(vec![seq, n_kv, hd]).expect("v reshape");

            // Gemma 3: per-head Q/K norm before RoPE (same shape pattern as Qwen 3).
            let mut q = ops::rmsnorm(backend, &q_3d, &gx.attn_q_norm, cfg.rms_eps);
            let mut k = ops::rmsnorm(backend, &k_3d, &gx.attn_k_norm, cfg.rms_eps);

            let rope_type = cfg.arch.rope_type();
            ops::rope(backend, &mut q, &positions, hd, rope_type, cfg.rope_theta);
            ops::rope(backend, &mut k, &positions, hd, rope_type, cfg.rope_theta);

            kv.append(backend, layer, &k, &v);
            let _ = n_rep;
            let kv_len = kv.len + seq;
            // Gemma 3 alternates local (sliding-window) and global attention layers.
            let sw = if cfg.layer_uses_sliding_window(layer) {
                cfg.sliding_window
            } else {
                None
            };
            let attn_out = ops::attention_swa(
                backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                kv_len, scale, past, sw,
            );

            let attn_t = attn_out.reshape(vec![seq, n_h * hd]).expect("attn reshape");
            let mut attn_proj = b.attn_output.linear(backend, &attn_t);

            // Gemma 3: post-attention norm before residual.
            attn_proj = ops::rmsnorm(backend, &attn_proj, &gx.attn_post_norm, cfg.rms_eps);
            // Fused residual + pre-FFN rmsnorm.
            let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, &b.ffn_norm, cfg.rms_eps);
            let gated = b.ffn_pair.geglu(backend, &xn2);
            let mut ffn_out = b.ffn_down.linear(backend, &gated);

            // Gemma 3: post-FFN norm before residual.
            ffn_out = ops::rmsnorm(backend, &ffn_out, &gx.ffn_post_norm, cfg.rms_eps);
            ops::add_inplace(backend, &mut x, &ffn_out);
        }

        kv.commit(seq);

        let x = ops::rmsnorm(backend, &x, &self.common.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        let mut logits = self.common.output.linear(backend, &x_last);

        // Final logit softcap, on device: scale → tanh → scale-back.
        if let Some(softcap) = cfg.final_logit_softcap {
            ops::mul_scalar_inplace(backend, &mut logits, 1.0 / softcap);
            ops::tanh_inplace(backend, &mut logits);
            ops::mul_scalar_inplace(backend, &mut logits, softcap);
        }
        logits
    }
}
