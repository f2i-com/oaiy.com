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
    /// RoPE base of the sliding-window ("local") layers: 10,000 as llama.cpp
    /// sets it for Gemma 3 (`rope_freq_base_train_swa`), unless the GGUF says
    /// (`rope.freq_base_swa`). Only the global layers use `rope.freq_base`
    /// (1,000,000). One base for every layer rotated the local layers' keys with
    /// the global layers' frequencies, and Gemma 3 4B lost the thread of a
    /// prompt of a few hundred tokens (markdown answered with fragments of it).
    rope_theta_local: f32,
    /// The global layers' linear RoPE scaling as per-frequency divisors
    /// (`rope.scaling.type = linear`, `rope.scaling.factor`: 8 on the 4B, 12B
    /// and 27B; none on the 1B). The local layers are not scaled.
    rope_global_factors: Option<Vec<f32>>,
    /// VENDORED-LOCAL: a decode step chained on the backend's device ([`crate::chain_decode`]).
    chain: crate::chain_decode::ChainDecoder,
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

        let ns = g.get_str("general.architecture").unwrap_or("gemma3");
        let rope_theta_local = config.rope_theta_swa.unwrap_or(10000.0);
        let linear = g.get_str(&format!("{ns}.rope.scaling.type")).is_ok_and(|t| t == "linear");
        let factor = g.get_f32(&format!("{ns}.rope.scaling.factor")).unwrap_or(1.0);
        let rope_global_factors = (linear && factor > 0.0 && factor != 1.0).then(|| vec![factor; config.head_dim / 2]);

        Ok(Self { config, tokenizer, common, extras, backend, embed_scale_const, rope_theta_local, rope_global_factors, chain: Default::default() })
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
        // VENDORED-LOCAL: a decode step chained on the device when the backend has one for this model.
        {
            let cfg = &self.config;
            let layers = (0..cfg.n_layers)
                .map(|l| {
                    if cfg.layer_uses_sliding_window(l) {
                        (self.rope_theta_local, None, cfg.sliding_window)
                    } else {
                        (cfg.rope_theta, self.rope_global_factors.as_deref(), None)
                    }
                })
                .collect();
            let dense = crate::chain_decode::Dense {
                qk_norms: Some(self.extras.iter().map(|e| (&e.attn_q_norm, &e.attn_k_norm)).collect()),
                post_norms: Some(self.extras.iter().map(|e| (&e.attn_post_norm, &e.ffn_post_norm)).collect()),
                gelu: true,
                embed_scale: self.embed_scale_const,
                layers,
                softcap: cfg.final_logit_softcap,
                ..crate::chain_decode::Dense::plain(cfg, &self.common)
            };
            let chained = if tokens.len() == 1 {
                self.chain.step(&*self.backend, &dense, tokens[0], kv)
            } else {
                self.chain.prompt(&*self.backend, &dense, tokens, kv)
            };
            if let Some(logits) = chained {
                return logits;
            }
        }
        self.forward_host(tokens, kv)
    }

    /// VENDORED-LOCAL: [`Self::forward`] op by op through the backend, never chained (what a chained step is
    /// checked against).
    pub fn forward_host(&self, tokens: &[u32], kv: &mut KvCache) -> Tensor {
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

            // Gemma 3 alternates local (sliding-window) and global attention layers, each with its own RoPE: the
            // local ones base 10,000 unscaled, the global ones `rope.freq_base` with the linear scaling.
            let local = cfg.layer_uses_sliding_window(layer);
            let (theta, factors) = if local {
                (self.rope_theta_local, None)
            } else {
                (cfg.rope_theta, self.rope_global_factors.as_deref())
            };
            let rope_type = cfg.arch.rope_type();
            ops::rope_with_factors(backend, &mut q, &positions, hd, rope_type, theta, factors);
            ops::rope_with_factors(backend, &mut k, &positions, hd, rope_type, theta, factors);

            kv.append(backend, layer, &k, &v);
            let _ = n_rep;
            let kv_len = kv.len + seq;
            let sw = if local { cfg.sliding_window } else { None };
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
