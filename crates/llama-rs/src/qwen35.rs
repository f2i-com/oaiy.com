//! Qwen 3.5 — gated-delta-net + attention hybrid (Apache-2.0). The 9B variant:
//!   * 32 blocks, 4096 dim, 16 heads, 4 KV heads, 12288 ff_dim
//!   * 248 K vocab, 262 K context, RoPE base 1e7
//!   * `full_attention_interval = 4` — every 4th layer (8 of 32) is full
//!     attention; the other 24 are linear-attention "gated delta net" layers.
//!
//! **The unsloth GGUF arch `qwen35` is what llama.cpp calls `qwen3next`.**
//! Reference: `llama.cpp/src/models/qwen3next.cpp` (build_layer_attn_full
//! and build_layer_attn_linear). Tensor naming differs slightly from llama.cpp:
//!
//! | unsloth GGUF        | llama.cpp qwen3next | shape (9B)        | role |
//! |---------------------|---------------------|-------------------|------|
//! | `attn_qkv.weight`   | (subset of ssm_in)  | `[8192, 4096]`    | fused q+k+v projection (no z) |
//! | `attn_gate.weight`  | (subset of ssm_in)  | `[4096, 4096]`    | z (output gate) projection |
//! | `ssm_alpha.weight`  | (subset of ssm_beta_alpha) | `[32, 4096]` | α projection |
//! | `ssm_beta.weight`   | (subset of ssm_beta_alpha) | `[32, 4096]` | β projection |
//! | `ssm_a`             | `ssm_a`             | `[32]`            | per-head gate scale |
//! | `ssm_dt.bias`       | `ssm_dt`            | `[32]`            | α bias (added before softplus) |
//! | `ssm_conv1d.weight` | `ssm_conv1d`        | `[8192, 4]`       | depthwise conv on q‖k‖v after projection |
//! | `ssm_norm.weight`   | `ssm_norm`          | `[128]`           | per-v-head RMSNorm before output |
//! | `ssm_out.weight`    | `ssm_out`           | `[4096, 4096]`    | output projection |
//!
//! Per llama.cpp: `head_k_dim = ssm.state_size = 128`,
//! `num_k_heads = ssm.group_count = 16`, `num_v_heads = ssm.time_step_rank = 32`,
//! `head_v_dim = ssm.inner_size / num_v_heads = 128`,
//! `qkv_per_k_head = 2*head_k_dim + head_v_dim*(num_v_heads/num_k_heads) = 512`,
//! `total qkv = 512 * num_k_heads = 8192` (matches our `attn_qkv` out dim).
//!
//! **Status (this commit):**
//!   * Loader correct for both layer types ✓
//!   * Layer structure matches qwen3next ✓
//!   * **V-HEAD ORDERING (v0.48 fix):** unsloth's converter
//!     (`_LinearAttentionVReorderBase` in `convert_hf_to_gguf.py`) reorders V
//!     heads from HF's "grouped" `[K0_V0, K0_V1, K1_V0, K1_V1, ...]` to "tiled"
//!     `[K0_V0, K1_V0, ..., K15_V0, K0_V1, ...]` for ggml broadcast efficiency.
//!     Affects 7 tensor groups: `attn_qkv` V rows, `attn_gate` (z) rows,
//!     `ssm_alpha`/`ssm_beta` rows, `ssm_a`/`ssm_dt` 1-D, `ssm_conv1d` V channels,
//!     `ssm_out` input dim. Code now uses `h_k = h_v % num_k_heads` (tiled), NOT
//!     the grouped `h_k = h_v / v_per_k`.
//!   * Full-attention layers (8/32): joint Q+gate split + per-head Q/K-norm +
//!     partial RoPE on the first `cfg.rope_dim = 64` dims of each 256-head
//!     (matches Qwen3.5's `partial_rotary_factor: 0.25`) + standard MHA. MRoPE
//!     collapses to standard NeoX RoPE for text-only inference. ✓
//!   * Linear-attention "gated delta net" layers (24/32): `run_delta_net_layer_host`
//!     (prefill) and `Backend::delta_net_decode_step` (decode, on-device CUDA
//!     kernel pair) implement per-token autoregressive update (input projections →
//!     conv1d state → l2_norm Q/K → softplus α gate → exp → state decay →
//!     state @ k → outer-product update → state @ q → norm-gated output →
//!     ssm_out projection). Verified end-to-end on real prompts: greedy
//!     completion of "The quick brown fox jumps over the lazy" → " dog.";
//!     "The capital of France is" → " Paris."; "Hello, my name is" → " John.
//!     I am a 25-year-old male. I am a student at a university." ✓
//!
//! **Performance:** decode **21.5 tok/s**, prefill **162 tok/s** on CUDA
//! (96-tok prompt). The SSM block uses `Backend::delta_net_step` for both
//! decode (seq=1) and prefill (seq>1). The CUDA impl uses fused-loop kernels
//! (`delta_net_conv1d_loop_f32` + `delta_net_step_loop_f32`) that internally
//! iterate over the seq axis — the rolling conv window and the 128×128 per-head
//! state matrix both live in registers across all seq iterations, so each
//! layer needs only 2 kernel launches regardless of seq length, and per-token
//! global-state I/O is eliminated. On CPU the trait's default fallback runs
//! the equivalent per-token math (~0.1 tok/s on a 9B model). The next prefill
//! speedup is the chunked-recurrence kernel (task #66) which would parallelize
//! across tokens within a CHUNK_SIZE=64 window.

use std::sync::Arc;

use ggml_rs::{ops, Backend, QuantizedTensor, Tensor};
use gguf::GgufFile;
use tokenizer::Tokenizer;

use crate::kv_cache::KvCache;
use crate::loader::{load_lm_head_or_tied, upload_tok_embd_and_lm_head, FfnPair, TensorIndex, Weight};
use crate::{LlamaError, ModelConfig, Result};

/// Per-layer extras, depending on whether the layer is SSM (Mamba2) or full
/// attention. Most layers are SSM; only every `full_attention_interval`-th
/// layer is full attention.
#[derive(Debug)]
pub enum Qwen35Block {
    /// Linear-attention "gated delta net" block (NOT vanilla Mamba2). Per
    /// `qwen3next.cpp::build_layer_attn_linear`: input → joint q+k+v projection
    /// (`attn_qkv`) and z projection (`attn_gate`); separate α/β projections
    /// (`ssm_alpha`/`ssm_beta`); α is bias-added (`ssm_dt_bias`), softplus'd,
    /// and multiplied by `ssm_a` to produce the per-head gate; q‖k‖v is conv1d'd
    /// (kernel 4, depthwise) + silu'd; then a chunked recurrence updates a
    /// per-head state matrix; output is `ssm_norm` RMS-normalized and projected
    /// out via `ssm_out`. The recurrence is what's not yet ported.
    Ssm {
        /// Pre-block RMSNorm scale (shape `[embed_dim]`).
        attn_norm:    Tensor,
        /// Joint q+k+v projection (no z; z is in `attn_gate`). Per k-head the
        /// 512-wide row packs `[head_k_dim]` query, `[head_k_dim]` key, then
        /// `[head_v_dim * num_v_heads/num_k_heads]` value. Shape: `[8192, 4096]`.
        attn_qkv:     Weight,
        /// z (output gate) projection. Each k-head has `head_v_dim *
        /// (num_v_heads/num_k_heads) = 256` z dims; total 4096.
        /// Shape: `[4096, 4096]`.
        attn_gate:    Weight,
        /// Depthwise conv1d weight applied along the sequence axis to
        /// q‖k‖v (concatenated). Kernel size 4. Shape: `[8192, 4]`.
        ssm_conv1d:   Tensor,
        /// Per-v-head log-eigenvalue scale: gate = softplus(α + bias) * ssm_a.
        /// Shape: `[num_v_heads]` = `[32]`.
        ssm_a:        Tensor,
        /// Fused β+α projection. Built at load time by concatenating the GGUF's
        /// `ssm_beta.weight` and `ssm_alpha.weight` along axis 0 (both Q8_0 of
        /// `[num_v_heads, embed_dim] = [32, 4096]`). Shape: `[2 * num_v_heads,
        /// embed_dim]` = `[64, 4096]`. Saves one matmul + one d2h per layer per
        /// token vs running the two projections separately.
        ssm_ba:       Weight,
        /// α bias added before softplus. Shape: `[num_v_heads]` = `[32]`.
        ssm_dt_bias:  Tensor,
        /// Pre-output RMSNorm scale (per-v-head). Shape: `[head_v_dim]` = `[128]`.
        ssm_norm:     Tensor,
        /// Output projection. Shape: `[embed_dim, embed_dim]` = `[4096, 4096]`.
        ssm_out:      Weight,
        /// Pre-FFN RMSNorm scale (called `post_attention_norm` in the GGUF /
        /// `attn_post_norm` in qwen3next.cpp; equivalent to qwen3's `ffn_norm`).
        post_norm:    Tensor,
        /// FFN: SwiGLU (gate + up + down) — gate and up are auto-fused into
        /// one matmul at load time, halved by `silu_mul_split`.
        ffn_pair:     FfnPair,
        ffn_down:     Weight,
    },
    /// Standard multi-head attention block with MRoPE. Same shape conventions
    /// as Qwen 3 dense, plus per-head Q/K-norm.
    Attention {
        attn_norm:    Tensor,
        attn_q:       Weight,
        attn_q_norm:  Tensor,
        attn_k:       Weight,
        attn_k_norm:  Tensor,
        attn_v:       Weight,
        attn_output:  Weight,
        /// Pre-FFN RMSNorm scale — see Ssm variant docstring.
        post_norm:    Tensor,
        ffn_pair:     FfnPair,
        ffn_down:     Weight,
    },
}

/// SSM hyperparameters, parsed from the `qwen35.ssm.*` metadata namespace.
#[derive(Debug, Clone, Copy)]
pub struct SsmConfig {
    pub conv_kernel:    usize,  // 4
    pub group_count:    usize,  // 16
    pub inner_size:     usize,  // 4096 (= embed_dim for Qwen3.5-9B)
    pub state_size:     usize,  // 128
    pub time_step_rank: usize,  // 32
}

#[derive(Debug)]
pub struct Qwen35Model {
    pub config:    ModelConfig,
    pub ssm_cfg:   SsmConfig,
    /// Layers where index `i` is full attention. Computed from
    /// `full_attention_interval`: `i % interval == interval - 1`. For the
    /// 9B variant with interval=4, this is layers `[3, 7, 11, …, 31]` (8 of 32).
    pub attention_layers: Vec<bool>,
    pub tokenizer: Tokenizer,
    pub blocks:    Vec<Qwen35Block>,
    pub tok_embd:  Arc<Tensor>,
    pub output_norm: Tensor,
    pub output:    Weight,
    pub backend:   Arc<dyn Backend>,
}

impl Qwen35Model {
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        // MoE detection: Qwen3.6-35B-A3B uses arch=qwen35 + `qwen35.expert_count > 0`.
        // The dense Qwen3.5/3.6 loader doesn't handle per-expert FFN tensors,
        // so we bail with a clear pointer to #88 instead of a missing-tensor crash.
        if let Ok(n_experts) = g.get_u64("qwen35.expert_count") {
            if n_experts > 0 {
                let used = g.get_u64("qwen35.expert_used_count").unwrap_or(0);
                return Err(LlamaError::Config(format!(
                    "Qwen3.6-MoE detected ({n_experts} experts, top-{used}). \
                     Needs the SSM hybrid (existing Qwen3.5/3.6 path) + MoE FFN \
                     (existing infra) combined into a single forward. Loader \
                     integration pending (task #88). MoeFfn infra is in \
                     `llama_rs::moe` and works on real GGUFs (Qwen3-30B-A3B + \
                     Mixtral 8x7B), but Qwen3.6-MoE additionally needs the \
                     gated-deltanet-vs-attention layer split per qwen35.full_attention_interval."
                )));
            }
        }
        let config = ModelConfig::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        let idx = TensorIndex::new(g);

        // ----- SSM hyperparameters --------------------------------------
        let get = |suffix: &str| -> Result<u64> {
            Ok(g.get_u64(&format!("qwen35.{suffix}"))?)
        };
        let ssm_cfg = SsmConfig {
            conv_kernel:    get("ssm.conv_kernel")?     as usize,
            group_count:    get("ssm.group_count")?     as usize,
            inner_size:     get("ssm.inner_size")?      as usize,
            state_size:     get("ssm.state_size")?      as usize,
            time_step_rank: get("ssm.time_step_rank")?  as usize,
        };

        // ----- Attention layer mask --------------------------------------
        let interval = g.get_u64("qwen35.full_attention_interval")
            .map(|v| v as usize).unwrap_or(4);
        let attention_layers: Vec<bool> = (0..config.n_layers)
            .map(|i| i % interval == interval - 1)
            .collect();

        // ----- Common embeddings + LM head -----------------------------
        let tok_embd    = Arc::new(idx.take("token_embd.weight",  &["tok_embeddings.weight"])?);
        let output_norm = idx.take("output_norm.weight", &["norm.weight"])?;
        let output = load_lm_head_or_tied(&idx, &tok_embd)?;

        // ----- Per-layer block tensors ---------------------------------
        let mut blocks = Vec::with_capacity(config.n_layers);
        for (i, &is_attn) in attention_layers.iter().enumerate() {
            let blk = if is_attn {
                Qwen35Block::Attention {
                    attn_norm:   idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?,
                    attn_q:      idx.take_weight(&format!("blk.{i}.attn_q.weight"), &[])?,
                    attn_q_norm: idx.take(&format!("blk.{i}.attn_q_norm.weight"), &[])?,
                    attn_k:      idx.take_weight(&format!("blk.{i}.attn_k.weight"), &[])?,
                    attn_k_norm: idx.take(&format!("blk.{i}.attn_k_norm.weight"), &[])?,
                    attn_v:      idx.take_weight(&format!("blk.{i}.attn_v.weight"), &[])?,
                    attn_output: idx.take_weight(&format!("blk.{i}.attn_output.weight"), &[])?,
                    post_norm:   idx.take(&format!("blk.{i}.post_attention_norm.weight"), &[])?,
                    ffn_pair:    {
                        let g = idx.take_weight(&format!("blk.{i}.ffn_gate.weight"), &[])?;
                        let u = idx.take_weight(&format!("blk.{i}.ffn_up.weight"),   &[])?;
                        FfnPair::from_halves(g, u)
                    },
                    ffn_down:    idx.take_weight(&format!("blk.{i}.ffn_down.weight"), &[])?,
                }
            } else {
                let ssm_alpha = idx.take_weight(&format!("blk.{i}.ssm_alpha.weight"), &[])?;
                let ssm_beta  = idx.take_weight(&format!("blk.{i}.ssm_beta.weight"), &[])?;
                let ssm_ba    = fuse_alpha_beta(ssm_beta, ssm_alpha);
                Qwen35Block::Ssm {
                    attn_norm:   idx.take(&format!("blk.{i}.attn_norm.weight"), &[])?,
                    attn_qkv:    idx.take_weight(&format!("blk.{i}.attn_qkv.weight"), &[])?,
                    attn_gate:   idx.take_weight(&format!("blk.{i}.attn_gate.weight"), &[])?,
                    ssm_conv1d:  idx.take(&format!("blk.{i}.ssm_conv1d.weight"), &[])?,
                    ssm_a:       idx.take(&format!("blk.{i}.ssm_a"), &[])?,
                    ssm_ba,
                    ssm_dt_bias: idx.take(&format!("blk.{i}.ssm_dt.bias"), &[])?,
                    ssm_norm:    idx.take(&format!("blk.{i}.ssm_norm.weight"), &[])?,
                    ssm_out:     idx.take_weight(&format!("blk.{i}.ssm_out.weight"), &[])?,
                    post_norm:   idx.take(&format!("blk.{i}.post_attention_norm.weight"), &[])?,
                    ffn_pair:    {
                        let g = idx.take_weight(&format!("blk.{i}.ffn_gate.weight"), &[])?;
                        let u = idx.take_weight(&format!("blk.{i}.ffn_up.weight"),   &[])?;
                        FfnPair::from_halves(g, u)
                    },
                    ffn_down:    idx.take_weight(&format!("blk.{i}.ffn_down.weight"), &[])?,
                }
            };
            blocks.push(blk);
        }

        // Move embeddings + LM head onto the backend (tied path shares Arc).
        let (tok_embd, output) = upload_tok_embd_and_lm_head(&*backend, tok_embd, output);
        let output_norm = backend.to_device(output_norm);
        let output      = output.to_device(&*backend);
        let blocks: Vec<Qwen35Block> = blocks.into_iter().map(|b| upload_block(b, &*backend)).collect();

        Ok(Self {
            config, ssm_cfg, attention_layers,
            tokenizer, blocks,
            tok_embd, output_norm, output,
            backend,
        })
    }

    /// Forward — **partial implementation**: full-attention layers are
    /// numerically correct (per llama.cpp's `qwen3next.cpp` reference), the
    /// "linear-attention" / gated-delta-net layers (24 of 32) are stubbed as
    /// identity. Output won't be coherent until the delta-net path lands.
    ///
    /// Attention layer details (matched to the llama.cpp reference):
    ///   * `attn_q.weight` is `[n_embd_head * 2, n_head, embed]` packed flat —
    ///     the first half of each per-head 512-dim slice is the **actual Q**
    ///     (matching k's 256), the second half is a **per-channel gate** that
    ///     gets `sigmoid`'d and element-wise multiplied with the attention
    ///     output before the final output projection. This was the previously-
    ///     mysterious 8192 vs 4096 dim mismatch.
    ///   * Standard NeoX RoPE on Q and K (no MRoPE here — the
    ///     `rope.dimension_sections` metadata applies to the recurrent path,
    ///     not these full-attention layers).
    ///   * Per-head Q/K-norm before RoPE — same pattern as Qwen 3 dense.
    ///
    /// What's still stubbed:
    ///   * Recurrent layers (`Qwen35Block::Ssm`) — actually a "gated delta net"
    ///     (linear attention with chunked recurrence + sigmoid gating + state
    ///     propagation), NOT vanilla Mamba2. Reference: `build_layer_attn_linear`
    ///     in qwen3next.cpp. Substantial port (chunked computation with
    ///     `CHUNK_SIZE=64` blocks, delta-net updates, A-noscan multiplication).
    ///     Currently those layers run only the SwiGLU FFN.
    /// Look up token embeddings. Companion to [`forward_embeds`] for the
    /// vision-language splice path; standard text-only callers can stay on
    /// [`forward`] and don't need to touch this directly.
    pub fn embed_text(&self, tokens: &[u32]) -> Tensor {
        self.backend.embed_lookup(&self.tok_embd, tokens, self.config.embedding_dim)
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Tensor> {
        let embeds = self.embed_text(tokens);
        self.forward_embeds(&embeds, tokens.len(), kv)
    }

    /// Vision-language helper mirroring [`Gemma4Model::embed_with_vision_at_placeholder`]:
    /// tokenize `prompt` (which must contain exactly one `placeholder_token_id`
    /// marker — `<|vision_pad|>` = 151654 for Qwen3-VL), expand the placeholder
    /// into `soft_tokens.dim(0)` repetitions of the same token id, embed the
    /// resulting sequence, and splice the vision soft tokens into those
    /// positions. Returns `(tokens, embeds)` — feed both back into
    /// [`forward_embeds`] to run the LM stack.
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
        debug_assert_eq!(soft_tokens.dim(soft_tokens.rank() - 1), d,
            "soft-token width {} doesn't match LM hidden dim {d}",
            soft_tokens.dim(soft_tokens.rank() - 1));
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

    /// Run the transformer stack on a pre-computed embedding tensor. Same
    /// shape as [`forward`] otherwise — used by the multimodal path so the
    /// vision soft tokens can be spliced into the embedding sequence before
    /// running the LM. `seq` must equal `embeds.dim(0)`.
    pub fn forward_embeds(&self, embeds: &Tensor, seq: usize, kv: &mut KvCache) -> Result<Tensor> {
        let cfg = &self.config;
        let backend = &*self.backend;
        let past = kv.len;
        debug_assert_eq!(embeds.dim(0), seq,
            "embeds first dim {} doesn't match seq {seq}", embeds.dim(0));

        let head_dim = cfg.head_dim;             // 256
        let n_h_kv  = cfg.n_kv_heads;            // 4
        // Per qwen3next.cpp: q dim per head is 2 × head_dim. First half is
        // the actual query (matched to k's head_dim); second half is a
        // per-channel gate that gets sigmoid'd and element-wise multiplied
        // with the attention output before the final output projection.
        let q_per_head_full = self.first_attn_q_out_dim() / cfg.n_heads;  // 512 for 9B
        debug_assert_eq!(q_per_head_full, 2 * head_dim,
            "expected q_per_head_full=2*head_dim; got {q_per_head_full} vs {}", 2 * head_dim);
        let n_h = cfg.n_heads;                   // 16
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut x = embeds.clone();
        let positions: Vec<u32> = (past..past + seq).map(|p| p as u32).collect();

        for (layer, blk) in self.blocks.iter().enumerate() {
            match blk {
                Qwen35Block::Attention { attn_norm, attn_q, attn_q_norm, attn_k, attn_k_norm,
                                         attn_v, attn_output, post_norm, ffn_pair, ffn_down } => {
                    let xn = ops::rmsnorm(backend, &x, attn_norm, cfg.rms_eps);
                    let q_full = attn_q.linear(backend, &xn);    // [seq, n_h * 2 * head_dim]
                    let k_flat = attn_k.linear(backend, &xn);
                    let v_flat = attn_v.linear(backend, &xn);

                    // Split q_full per head: first head_dim → query, second head_dim → gate.
                    // On CUDA this is a single kernel writing both halves directly on
                    // device — eliminates the per-attention-layer host roundtrip.
                    let (q_only, q_gate) = backend.split_q_and_gate(&q_full, n_h, head_dim);
                    let q_3d = q_only.reshape(vec![seq, n_h, head_dim]).expect("q reshape");
                    let k_3d = k_flat.reshape(vec![seq, n_h_kv, head_dim]).expect("k reshape");
                    let v_3d = v_flat.reshape(vec![seq, n_h_kv, head_dim]).expect("v reshape");

                    let mut q = ops::rmsnorm(backend, &q_3d, attn_q_norm, cfg.rms_eps);
                    let mut k = ops::rmsnorm(backend, &k_3d, attn_k_norm, cfg.rms_eps);

                    // Qwen3.5: partial RoPE — only the first `cfg.rope_dim` dims of
                    // each head's `head_dim`-wide slice get rotated (the
                    // "rope.dimension_count" metadata = 64; head_dim = 256 ⇒ rotates
                    // only 25%, leaves the back 192 dims untouched). MRoPE
                    // (mrope_section=[11,11,10]) collapses to standard NeoX RoPE for
                    // text-only inference (no image positions).
                    if cfg.rope_dim < head_dim {
                        backend.rope_partial_neox(&mut q, &positions, head_dim, cfg.rope_dim, cfg.rope_theta);
                        backend.rope_partial_neox(&mut k, &positions, head_dim, cfg.rope_dim, cfg.rope_theta);
                    } else {
                        let rope_type = cfg.arch.rope_type();
                        ops::rope(backend, &mut q, &positions, head_dim, rope_type, cfg.rope_theta);
                        ops::rope(backend, &mut k, &positions, head_dim, rope_type, cfg.rope_theta);
                    }

                    kv.append(backend, layer, &k, &v_3d);
                    let kv_len = kv.len + seq;
                    let attn_out = ops::attention(
                        backend, &q, kv.k_buffer(layer), kv.v_buffer(layer),
                        kv_len, scale, past,
                    );

                    // Gate the attention output: `attn_out *= sigmoid(q_gate)`,
                    // fused into a single kernel.
                    let mut attn_t = attn_out.reshape(vec![seq, n_h * head_dim]).expect("attn reshape");
                    backend.mul_sigmoid_inplace(&mut attn_t, &q_gate);

                    let attn_proj = attn_output.linear(backend, &attn_t);
                    // FFN (SwiGLU). post_attention_norm is the *pre-FFN* norm
                    // (qwen3next convention; equivalent to qwen3's ffn_norm).
                    let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, post_norm, cfg.rms_eps);
                    let activated = ffn_pair.swiglu(backend, &xn2);
                    let ffn_out = ffn_down.linear(backend, &activated);
                    ops::add_inplace(backend, &mut x, &ffn_out);
                }
                Qwen35Block::Ssm { attn_norm, attn_qkv, attn_gate, ssm_conv1d, ssm_a,
                                   ssm_ba, ssm_dt_bias, ssm_norm, ssm_out,
                                   post_norm, ffn_pair, ffn_down } => {
                    // ----- Pre-block RMSNorm + on-device input projections -----
                    let xn = ops::rmsnorm(backend, &x, attn_norm, cfg.rms_eps);
                    let mixed_qkv = attn_qkv.linear(backend, &xn);   // [seq, 8192]
                    let z         = attn_gate.linear(backend, &xn);  // [seq, 4096]
                    // ssm_ba is the fused [β | α] projection: one matmul gives
                    // [seq, 2 * num_v_heads], split into β (first half) and α
                    // (second half). Saves one matmul + one d2h per layer per
                    // token vs running β/α as two separate projs.
                    let beta_alpha = ssm_ba.linear(backend, &xn);    // [seq, 64]

                    // SSM block runs entirely on device — `delta_net_step` loops
                    // the conv1d + per-head update kernels per token internally.
                    // For decode (seq=1) it's one kernel pair; for prefill it's
                    // `seq` kernel pairs, all without leaving the GPU. State +
                    // conv buffers live on device for the cache lifetime.
                    let num_v_heads = self.ssm_cfg.time_step_rank;
                    let num_k_heads = self.ssm_cfg.group_count;
                    let head_v_dim  = self.ssm_cfg.inner_size / num_v_heads;
                    let head_k_dim  = self.ssm_cfg.state_size;
                    let v_per_k     = num_v_heads / num_k_heads;
                    let conv_kernel = self.ssm_cfg.conv_kernel;
                    let conv_dim    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
                    let scale_q     = 1.0 / (head_v_dim as f32).sqrt();

                    if kv.ssm_state[layer].is_none() {
                        kv.ssm_state[layer] = Some(backend.to_device(
                            Tensor::zeros(vec![num_v_heads, head_v_dim, head_v_dim])));
                    }
                    if kv.ssm_conv[layer].is_none() {
                        kv.ssm_conv[layer] = Some(backend.to_device(
                            Tensor::zeros(vec![conv_kernel - 1, conv_dim])));
                    }
                    let mut state_dev = kv.ssm_state[layer].take().unwrap();
                    if state_dev.is_cpu() { state_dev = backend.to_device(state_dev); }
                    let mut conv_dev  = kv.ssm_conv [layer].take().unwrap();
                    if conv_dev.is_cpu()  { conv_dev  = backend.to_device(conv_dev);  }
                    let attn_out_dev = backend.delta_net_step(
                        &mixed_qkv, &z, &beta_alpha,
                        ssm_conv1d, ssm_a, ssm_dt_bias, ssm_norm,
                        &mut conv_dev, &mut state_dev,
                        seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k,
                        scale_q, cfg.rms_eps,
                    );
                    kv.ssm_state[layer] = Some(state_dev);
                    kv.ssm_conv [layer] = Some(conv_dev);

                    // ----- Output projection + fused residual+pre-FFN norm + FFN ---
                    let attn_proj = ssm_out.linear(backend, &attn_out_dev);
                    let xn2 = ops::add_inplace_then_rmsnorm(backend, &mut x, &attn_proj, post_norm, cfg.rms_eps);
                    let activated = ffn_pair.swiglu(backend, &xn2);
                    let ffn_out = ffn_down.linear(backend, &activated);
                    ops::add_inplace(backend, &mut x, &ffn_out);
                }
            }
        }

        kv.commit(seq);
        let x = ops::rmsnorm(backend, &x, &self.output_norm, cfg.rms_eps);
        let x_last = if seq > 1 { backend.slice_axis0_range(&x, seq - 1, 1) } else { x };
        Ok(self.output.linear(backend, &x_last))
    }

    /// Look up the q-projection's output dim from the first attention layer.
    fn first_attn_q_out_dim(&self) -> usize {
        for blk in &self.blocks {
            if let Qwen35Block::Attention { attn_q, .. } = blk {
                return attn_q.shape()[0];
            }
        }
        self.config.n_heads * self.config.head_dim
    }
}

/// Fuse the GGUF's separate `ssm_beta` and `ssm_alpha` weights (each shape
/// `[num_v_heads, embed_dim]`) into a single weight of shape
/// `[2*num_v_heads, embed_dim]` by concatenating along axis 0. The HF model has
/// these fused as `ssm_beta_alpha` already; unsloth's GGUF ships them split.
/// For Q8_0 (Qwen3.5 9B), each row is a contiguous block sequence (block_size=32,
/// type_size=34 bytes), so axis-0 concat is a byte-level append. For F32 (Qwen3.6
/// 27B keeps these full-precision), we concat the f32 element vectors. Either
/// way: saves one matmul + one d2h sync per layer per token at decode time.
pub fn fuse_alpha_beta(beta: Weight, alpha: Weight) -> Weight {
    match (beta, alpha) {
        (Weight::Quant(b_qt), Weight::Quant(a_qt)) => {
            debug_assert_eq!(b_qt.dtype(), a_qt.dtype(),
                "fuse_alpha_beta: dtype mismatch β={:?} α={:?}", b_qt.dtype(), a_qt.dtype());
            debug_assert_eq!(b_qt.shape(), a_qt.shape(),
                "fuse_alpha_beta: shape mismatch β={:?} α={:?}", b_qt.shape(), a_qt.shape());
            let dtype = b_qt.dtype();
            let mut shape = b_qt.shape().to_vec();
            shape[0] *= 2;
            let mut bytes = b_qt.bytes().to_vec();
            bytes.extend_from_slice(a_qt.bytes());
            Weight::Quant(QuantizedTensor::from_bytes_cpu(bytes, shape, dtype))
        }
        (Weight::Dense(b_t), Weight::Dense(a_t)) => {
            debug_assert_eq!(b_t.shape(), a_t.shape(),
                "fuse_alpha_beta: shape mismatch β={:?} α={:?}", b_t.shape(), a_t.shape());
            let mut shape = b_t.shape().to_vec();
            shape[0] *= 2;
            let mut data = b_t.data().to_vec();
            data.extend_from_slice(a_t.data());
            Weight::Dense(Tensor::from_vec(data, shape))
        }
        (b, a) => panic!(
            "fuse_alpha_beta: mismatched Weight kinds β={} α={}",
            weight_kind(&b), weight_kind(&a)
        ),
    }
}

fn weight_kind(w: &Weight) -> &'static str {
    match w {
        Weight::Dense(_) => "Dense(f32)",
        Weight::Quant(_) => "Quant",
        Weight::TiedEmbed(_) => "TiedEmbed",
    }
}

/// (Pre-v0.50: host-side per-head split. The live attention code now uses
/// `Backend::split_q_and_gate` which has a CUDA-kernel override.)
#[allow(dead_code)]
fn split_q_and_gate_host(
    backend: &dyn Backend,
    q_full:  &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> (Tensor, Tensor) {
    let seq = q_full.dim(0);
    let stride = 2 * head_dim;
    let host = q_full.to_host();
    let src = host.data();
    let mut q  = vec![0.0f32; seq * n_heads * head_dim];
    let mut g  = vec![0.0f32; seq * n_heads * head_dim];
    for s in 0..seq {
        let src_row_off = s * n_heads * stride;
        let dst_row_off = s * n_heads * head_dim;
        for h in 0..n_heads {
            let src_off = src_row_off + h * stride;
            let dst_off = dst_row_off + h * head_dim;
            q[dst_off..dst_off + head_dim]
                .copy_from_slice(&src[src_off..src_off + head_dim]);
            g[dst_off..dst_off + head_dim]
                .copy_from_slice(&src[src_off + head_dim..src_off + stride]);
        }
    }
    (
        backend.to_device(Tensor::from_vec(q, vec![seq, n_heads * head_dim])),
        backend.to_device(Tensor::from_vec(g, vec![seq, n_heads * head_dim])),
    )
}

/// Bundle of host-side inputs to one delta-net layer. The per-token outputs
/// (mixed_qkv, z, beta_alpha) come in as `&[f32]` slices so callers can hand
/// us a single fused-d2h buffer without re-copying.
#[allow(dead_code)] // historical: pre-v0.49 host SSM path; the SSM block now
// uses `Backend::delta_net_step` for both decode and prefill.
struct SsmStepHostInputs<'a> {
    mixed_qkv:   &'a [f32],
    z:           &'a [f32],
    beta_alpha:  &'a [f32],
    ssm_a:       &'a Tensor,
    ssm_conv1d:  &'a Tensor,
    ssm_dt_bias: &'a Tensor,
    ssm_norm:    &'a Tensor,
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline(always)]
fn delta_net_head_step(
    h_v: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    scale_q: f32,
    eps: f32,
    q_buf: &[f32], k_buf: &[f32], v_buf: &[f32], z_buf: &[f32],
    beta_t: &[f32], alph_t: &[f32], dt_h: &[f32], a_h: &[f32], nm_h: &[f32],
    st: &mut [f32], out_h: &mut [f32],
) {
    let q_h = &q_buf[h_v * head_k_dim..(h_v + 1) * head_k_dim];
    let k_h = &k_buf[h_v * head_k_dim..(h_v + 1) * head_k_dim];
    let v_h = &v_buf[h_v * head_v_dim..(h_v + 1) * head_v_dim];
    let z_h = &z_buf[h_v * head_v_dim..(h_v + 1) * head_v_dim];

    // a/b: l2_norm q and k. y = x / sqrt(sum(x²) + eps)
    let mut q_n = [0.0f32; 256]; let mut k_n = [0.0f32; 256];
    debug_assert!(head_k_dim <= 256);
    let inv_q = 1.0 / (q_h.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    let inv_k = 1.0 / (k_h.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    for i in 0..head_k_dim {
        q_n[i] = q_h[i] * inv_q * scale_q;
        k_n[i] = k_h[i] * inv_k;
    }

    // c: beta_h = sigmoid(beta[h_v])
    let bh = 1.0 / (1.0 + (-beta_t[h_v]).exp());

    // d: g_t = exp(softplus(alpha[h_v] + dt_bias[h_v]) * ssm_a[h_v])
    let alph_b = alph_t[h_v] + dt_h[h_v];
    let alph_sp = if alph_b > 20.0 { alph_b } else { alph_b.exp().ln_1p() };
    let g_t = (alph_sp * a_h[h_v]).exp();

    // e: state[h_v] *= g_t
    for s in st.iter_mut() { *s *= g_t; }

    // f: kv_mem[i] = sum_j state[i, j] * k_n[j]
    let mut kv_mem = [0.0f32; 256];
    debug_assert!(head_v_dim <= 256);
    for i in 0..head_v_dim {
        let row = &st[i * head_v_dim..(i + 1) * head_v_dim];
        let mut acc = 0.0f32;
        for j in 0..head_k_dim { acc += row[j] * k_n[j]; }
        kv_mem[i] = acc;
    }

    // g: delta[i] = (v[i] - kv_mem[i]) * bh
    let mut delta = [0.0f32; 256];
    for i in 0..head_v_dim { delta[i] = (v_h[i] - kv_mem[i]) * bh; }

    // h: state[i, j] += delta[i] * k_n[j]
    for i in 0..head_v_dim {
        let row = &mut st[i * head_v_dim..(i + 1) * head_v_dim];
        let di = delta[i];
        for j in 0..head_k_dim { row[j] += di * k_n[j]; }
    }

    // i: core_attn[i] = sum_j state[i, j] * q_n[j]
    let mut core = [0.0f32; 256];
    for i in 0..head_v_dim {
        let row = &st[i * head_v_dim..(i + 1) * head_v_dim];
        let mut acc = 0.0f32;
        for j in 0..head_k_dim { acc += row[j] * q_n[j]; }
        core[i] = acc;
    }

    // j: build_norm_gated: y = rmsnorm(core, ssm_norm) * silu(z[h_v])
    let mean_sq = core[..head_v_dim].iter().map(|v| v * v).sum::<f32>() / head_v_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..head_v_dim {
        let normed = core[i] * inv_rms * nm_h[i];
        let silu_z = z_h[i] / (1.0 + (-z_h[i]).exp());
        out_h[i] = normed * silu_z;
    }
}

/// Pre-v0.49 host-only SSM forward. Kept as a reference impl; the live SSM
/// block now uses `Backend::delta_net_step` (which has its own host fallback
/// for non-CUDA backends). Slated for deletion once we're confident the new
/// path is stable.
#[allow(dead_code)]
fn run_delta_net_layer_host(
    cfg:     &ModelConfig,
    ssm_cfg: &SsmConfig,
    inp:     SsmStepHostInputs<'_>,
    kv:      &mut KvCache,
    layer:   usize,
    seq:     usize,
) -> Tensor {
    let num_k_heads = ssm_cfg.group_count;          // 16
    let num_v_heads = ssm_cfg.time_step_rank;       // 32
    let head_k_dim  = ssm_cfg.state_size;           // 128
    let head_v_dim  = ssm_cfg.inner_size / num_v_heads; // 128
    let conv_kernel = ssm_cfg.conv_kernel;          // 4
    let conv_dim    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads; // 8192
    let embed_dim   = cfg.embedding_dim;

    // ----- Lazy-allocate state + conv buffers -----------------------------
    if kv.ssm_state[layer].is_none() {
        kv.ssm_state[layer] = Some(Tensor::zeros(vec![num_v_heads, head_v_dim, head_v_dim]));
    }
    if kv.ssm_conv[layer].is_none() {
        kv.ssm_conv[layer]  = Some(Tensor::zeros(vec![conv_kernel - 1, conv_dim]));
    }
    let mut state = kv.ssm_state[layer].take().unwrap();
    let mut conv  = kv.ssm_conv [layer].take().unwrap();
    debug_assert_eq!(state.shape(), &[num_v_heads, head_v_dim, head_v_dim]);
    debug_assert_eq!(conv.shape(),  &[conv_kernel - 1, conv_dim]);

    // ----- Pull all small per-block tensors to host slices ---------------
    let mqkv  = inp.mixed_qkv;          // [seq, 8192] — component-grouped [q|k|v]
    let zs    = inp.z;                  // [seq, 4096] — per-v-head natural
    let ba    = inp.beta_alpha;         // [seq, 2 * num_v_heads]; β first then α
    // SSM constants may be on device (when CUDA decode path is wired); .to_host()
    // is a no-op for already-host tensors. Pulled once per layer per call, not
    // per token, so the cost is negligible relative to the per-token loop.
    let a_h_t  = inp.ssm_a.to_host();       let a_h  = a_h_t.data();
    let dt_h_t = inp.ssm_dt_bias.to_host(); let dt_h = dt_h_t.data();
    let cn_h_t = inp.ssm_conv1d.to_host();  let cn_h = cn_h_t.data();
    let nm_h_t = inp.ssm_norm.to_host();    let nm_h = nm_h_t.data();

    let scale_q = 1.0f32 / (head_v_dim as f32).sqrt();
    let eps     = cfg.rms_eps;

    // Working buffers reused across timesteps.
    let mut qkv_re      = vec![0.0f32; conv_dim];   // re-laid-out [q_all|k_all|v_all]
    let mut conv_input  = vec![0.0f32; conv_kernel * conv_dim];
    let mut conv_output = vec![0.0f32; conv_dim];
    let mut q_buf       = vec![0.0f32; num_v_heads * head_k_dim];  // q repeated to v-heads
    let mut k_buf       = vec![0.0f32; num_v_heads * head_k_dim];
    let mut v_buf       = vec![0.0f32; num_v_heads * head_v_dim];
    let mut z_buf       = vec![0.0f32; num_v_heads * head_v_dim];  // [num_v_heads, head_v_dim]
    let mut output      = vec![0.0f32; seq * embed_dim];

    let state_data = state.data_mut();
    let conv_data  = conv.data_mut();

    for t in 0..seq {
        // 1. attn_qkv output is already in component-grouped layout
        //    [q_all (2048) | k_all (2048) | v_all (4096)] — matches HF
        //    transformers' Qwen3_5LinearAttention `torch.split(mixed_qkv,
        //    [key_dim, key_dim, value_dim], dim=-1)` convention. Copy as-is.
        let mqkv_t = &mqkv[t * conv_dim..(t + 1) * conv_dim];
        qkv_re.copy_from_slice(mqkv_t);

        // 2. Build conv input window: [conv_state; new_sample], shape [K, conv_dim].
        for k in 0..conv_kernel - 1 {
            let dst = k * conv_dim;
            let src = k * conv_dim;
            conv_input[dst..dst + conv_dim].copy_from_slice(&conv_data[src..src + conv_dim]);
        }
        let last = (conv_kernel - 1) * conv_dim;
        conv_input[last..last + conv_dim].copy_from_slice(&qkv_re);

        // 3. Depthwise conv1d: out[c] = sum_{k} input[k, c] * weight[c, k]; then silu.
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            let w_off = c * conv_kernel;
            for k in 0..conv_kernel {
                acc += conv_input[k * conv_dim + c] * cn_h[w_off + k];
            }
            // silu: x / (1 + exp(-x))
            conv_output[c] = acc / (1.0 + (-acc).exp());
        }

        // 4. Slide conv window forward: drop oldest, append the new sample.
        for k in 0..conv_kernel - 2 {
            let dst = k * conv_dim;
            let src = (k + 1) * conv_dim;
            conv_data[dst..dst + conv_dim].copy_from_slice(&conv_input[src..src + conv_dim]);
        }
        let last_dst = (conv_kernel - 2) * conv_dim;
        conv_data[last_dst..last_dst + conv_dim].copy_from_slice(&qkv_re);

        // 5. Split conv_output into q/k/v and repeat q,k along v-head axis.
        //    Unsloth's converter (per `_LinearAttentionVReorderBase` in
        //    convert_hf_to_gguf.py) reorders V heads from HF's "grouped"
        //    `[K0_V0, K0_V1, K1_V0, K1_V1, ...]` to "tiled" `[K0_V0, K1_V0, ...,
        //    K15_V0, K0_V1, ..., K15_V1]` for ggml broadcast efficiency. So
        //    `h_v` in the GGUF flat index decodes as `h_k = h_v % num_k_heads`
        //    (NOT the grouped `h_k = h_v / v_per_k`). This applies to attn_gate
        //    (z), ssm_alpha, ssm_beta, ssm_a, ssm_dt, ssm_out (input dim), and
        //    the V portion of ssm_conv1d.
        let q_base = 0;
        let k_base = head_k_dim * num_k_heads;
        let v_base = 2 * head_k_dim * num_k_heads;
        for h_v in 0..num_v_heads {
            let h_k = h_v % num_k_heads;
            let dst = h_v * head_k_dim;
            let q_src = q_base + h_k * head_k_dim;
            let k_src = k_base + h_k * head_k_dim;
            let v_src = v_base + h_v * head_v_dim;
            q_buf[dst..dst + head_k_dim].copy_from_slice(&conv_output[q_src..q_src + head_k_dim]);
            k_buf[dst..dst + head_k_dim].copy_from_slice(&conv_output[k_src..k_src + head_k_dim]);
            v_buf[dst..dst + head_v_dim].copy_from_slice(&conv_output[v_src..v_src + head_v_dim]);
        }

        // 6. z is natively in TILED V order (the unsloth converter reorders the
        //    in_proj_z weight rows, so the projection output is already tiled).
        //    Direct copy — `z_buf[h_v * head_v_dim..]` matches `z_t[h_v * ..]`.
        let z_t = &zs[t * (num_v_heads * head_v_dim)..(t + 1) * (num_v_heads * head_v_dim)];
        z_buf.copy_from_slice(z_t);

        // 7. Per-v-head delta-net update. Heads are fully independent: each
        //    touches its own slice of `state_data` and `out_t`. We chunk into
        //    `PAR_CHUNK` heads per task (4 tasks × 8 heads at the default 32
        //    v-heads). Earlier 1-head-per-task chunking was net negative — the
        //    rayon scheduler overhead exceeded the per-head ~12 µs of work; at
        //    8 heads/task each chunk does ~100 µs of work, well above scheduler
        //    overhead.
        // ba is [seq, 2*num_v_heads], with β occupying the first num_v_heads
        // and α the next num_v_heads per token.
        let ba_off = t * (2 * num_v_heads);
        let beta_t = &ba[ba_off..ba_off + num_v_heads];
        let alph_t = &ba[ba_off + num_v_heads..ba_off + 2 * num_v_heads];
        let out_t  = &mut output[t * embed_dim..(t + 1) * embed_dim];

        const PAR_CHUNK: usize = 4;
        let head_state_size = head_v_dim * head_v_dim;
        use rayon::prelude::*;
        state_data
            .par_chunks_exact_mut(PAR_CHUNK * head_state_size)
            .zip(out_t.par_chunks_exact_mut(PAR_CHUNK * head_v_dim))
            .enumerate()
            .for_each(|(chunk_idx, (states_chunk, outs_chunk))| {
                let h_base = chunk_idx * PAR_CHUNK;
                for local in 0..PAR_CHUNK {
                    let h_v = h_base + local;
                    let st = &mut states_chunk[local * head_state_size..(local + 1) * head_state_size];
                    let out_h = &mut outs_chunk[local * head_v_dim..(local + 1) * head_v_dim];
                    delta_net_head_step(
                        h_v, head_k_dim, head_v_dim, scale_q, eps,
                        &q_buf, &k_buf, &v_buf, &z_buf,
                        beta_t, alph_t, dt_h, a_h, nm_h,
                        st, out_h,
                    );
                }
            });
    }

    // Restore the (now-updated) state and conv into the cache.
    kv.ssm_state[layer] = Some(state);
    kv.ssm_conv [layer] = Some(conv);

    Tensor::from_vec(output, vec![seq, embed_dim])
}

/// Reserve this much VRAM for KV cache + activations + intermediates when
/// deciding whether a Weight fits. Tuned for the largest model in the lineup
/// (27B at ~131k context); smaller models leave correspondingly more headroom.
/// Big Weights that would breach this margin stay host-resident — `linear`/
/// `linear_q` then dispatch to the CPU op and we still produce correct output,
/// just slower per-tensor. Norms and biases are always device-resident (they
/// are tiny and the per-layer ops would do per-step h2d transfers otherwise).
const VRAM_SAFETY_MARGIN_BYTES: usize = 2 * 1024 * 1024 * 1024;

fn upload_block(b: Qwen35Block, backend: &dyn Backend) -> Qwen35Block {
    let m = VRAM_SAFETY_MARGIN_BYTES;
    match b {
        Qwen35Block::Attention {
            attn_norm, attn_q, attn_q_norm, attn_k, attn_k_norm, attn_v,
            attn_output, post_norm, ffn_pair, ffn_down,
        } => Qwen35Block::Attention {
            attn_norm:   backend.to_device(attn_norm),
            attn_q:      attn_q.try_to_device(backend, m),
            attn_q_norm: backend.to_device(attn_q_norm),
            attn_k:      attn_k.try_to_device(backend, m),
            attn_k_norm: backend.to_device(attn_k_norm),
            attn_v:      attn_v.try_to_device(backend, m),
            attn_output: attn_output.try_to_device(backend, m),
            post_norm:   backend.to_device(post_norm),
            ffn_pair:    ffn_pair.try_to_device(backend, m),
            ffn_down:    ffn_down.try_to_device(backend, m),
        },
        Qwen35Block::Ssm {
            attn_norm, attn_qkv, attn_gate, ssm_conv1d, ssm_a, ssm_ba,
            ssm_dt_bias, ssm_norm, ssm_out, post_norm, ffn_pair, ffn_down,
        } => Qwen35Block::Ssm {
            attn_norm:   backend.to_device(attn_norm),
            attn_qkv:    attn_qkv.try_to_device(backend, m),
            attn_gate:   attn_gate.try_to_device(backend, m),
            // ssm_conv1d / ssm_a / ssm_dt_bias / ssm_norm are device-resident
            // for the GPU `delta_net_step` kernel — they're small and read every
            // layer per token.
            ssm_conv1d:  backend.to_device(ssm_conv1d),
            ssm_a:       backend.to_device(ssm_a),
            ssm_ba:      ssm_ba.try_to_device(backend, m),
            ssm_dt_bias: backend.to_device(ssm_dt_bias),
            ssm_norm:    backend.to_device(ssm_norm),
            ssm_out:     ssm_out.try_to_device(backend, m),
            post_norm:   backend.to_device(post_norm),
            ffn_pair:    ffn_pair.try_to_device(backend, m),
            ffn_down:    ffn_down.try_to_device(backend, m),
        },
    }
}
