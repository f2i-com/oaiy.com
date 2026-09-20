//! Configuration parsed from GGUF metadata.

use ggml_rs::RopeType;
use gguf::{Array, GgufFile};

use crate::{LlamaError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
    Gemma3,
    /// Gemma 3n (a.k.a. "Gemma 4 E2B/E4B") — the embeddable on-device variant
    /// with MatFormer architecture. Multimodal capable (vision + audio in the
    /// official model; in this crate text-only for now). Adds AltUp routing,
    /// Per-Layer Embeddings, Laurel low-rank residual paths, shared KV layers,
    /// per-layer FFN sizes, activation sparsity, and a per-layer sliding-window
    /// pattern on top of the Gemma 3 base. Loaded but forward-pass not yet
    /// fully implemented.
    Gemma3n,
    /// Gemma 4 — Apache-2.0-licensed successor to Gemma 3, released April 2026.
    /// E2B / E4B / 26B-MoE / 31B-Dense variants. Per-layer head_dim (different
    /// for SWA vs global layers), per-layer FFN sizes, per-layer RoPE base,
    /// per-layer SWA bool array, and final logit softcap.
    Gemma4,
    /// Qwen 3.5 — Mamba2/attention hybrid (released early 2026). Most layers
    /// run a Mamba2 SSM block (`ssm_*` tensors); every `full_attention_interval`-th
    /// layer (default 4) runs full attention with MRoPE. SSM/Mamba2 forward is
    /// the substantial-new-code piece — the loader recognises both layer types
    /// and reports a clear error from forward until the SSM path lands.
    Qwen35,
    /// Qwen3-MoE — Llama backbone + per-block Mixture-of-Experts FFN. Targets
    /// Qwen3-30B-A3B (typically 128 experts, top-8). MoE infrastructure landed
    /// in v0.59 (`llama_rs::moe`); arch loader integration pending #85.
    Qwen3Moe,
    /// Qwen3-VL-MoE — same per-block layout as `qwen3moe` (full attention +
    /// per-block MoE, no shared expert) plus the Qwen3-VL vision tower. Adds
    /// `n_deepstack_layers` metadata. Routes to `Qwen3MoeModel` (same LM
    /// stack) + the existing `qwen3vl_merger` mmproj loader.
    Qwen3VlMoe,
    /// Qwen3.6-MoE — qwen35 SSM-hybrid backbone + 256-expert MoE + shared
    /// expert (DeepSeek-V2 / Qwen3-Next pattern). Targets Qwen3.6-35B-A3B.
    /// `arch=qwen35moe`. Forward in `crates/llama-rs/src/qwen35moe.rs`.
    Qwen35Moe,
    /// Qwen3.6-MoE-VL — same as Qwen3.6 (qwen35 backbone) + MoE + Qwen3-VL
    /// vision tower. Three-way pending: MoE loader (#85), vision tower (#84),
    /// integration of both into the qwen35 dense path.
    Qwen36MoeVl,
    // VENDORED-LOCAL: GLM-5.3-Flash (`glm5next`).
    /// GLM-5.3-Flash — `arch=glm5next`, 313B-A17B (46 blocks, 288 experts,
    /// top-8 + 1 shared). Three things no other arch here has at once:
    ///   * **Hybrid attention by per-layer array**, not a modular interval.
    ///     `attention.head_count_kv` is a 0/1 array: 0 = KDA linear-attention
    ///     layer, 1 = full MLA. 34 KDA + 11 MLA over a 45-layer trunk.
    ///   * **MLA + a sparse "lightning indexer"** (`top_k=2048`, `kpool=4`)
    ///     that scores pooled key groups and gathers the winning cells. NoPE:
    ///     `rope.dimension_count = 0` for the whole text tower.
    ///   * **Hyper-connections** — the residual stream is 4 parallel copies,
    ///     mixed per sublayer through a Sinkhorn-normalised 4x4. Same
    ///     formulation as DeepSeek-V4.1 (`dsv41::hc`), whose reference
    ///     `glm5next` inherits from verbatim.
    ///
    /// `block_count` is 46 but the trunk is 45: `blk.45` is the NextN/MTP
    /// draft block, which is why `hc_*` stops at `blk.44`. Loaded and
    /// validated; forward lands with the KDA scan (see `glm5next.rs`).
    Glm5Next,
    /// Architectures recognized but not yet implemented.
    Unsupported(String),
}

impl Architecture {
    pub fn from_str(s: &str) -> Self {
        match s {
            "llama"        => Self::Llama,
            "mistral"      => Self::Mistral,
            "qwen2"        => Self::Qwen2,
            "qwen3"        => Self::Qwen3,
            "gemma3"       => Self::Gemma3,
            "gemma3n"      => Self::Gemma3n,
            "gemma4"       => Self::Gemma4,
            "qwen35"       => Self::Qwen35,
            "qwen3moe"     => Self::Qwen3Moe,
            "qwen3vlmoe"   => Self::Qwen3VlMoe,
            "qwen35moe"    => Self::Qwen35Moe,
            "qwen36moevl"  => Self::Qwen36MoeVl,
            "glm5next"     => Self::Glm5Next,
            other          => Self::Unsupported(other.to_string()),
        }
    }

    pub fn supported(&self) -> bool { !matches!(self, Self::Unsupported(_)) }

    pub fn name(&self) -> &str {
        match self {
            Self::Llama       => "llama",
            Self::Mistral     => "mistral",
            Self::Qwen2       => "qwen2",
            Self::Qwen3       => "qwen3",
            Self::Gemma3      => "gemma3",
            Self::Gemma3n     => "gemma3n",
            Self::Gemma4      => "gemma4",
            Self::Qwen35      => "qwen35",
            Self::Qwen3Moe    => "qwen3moe",
            Self::Qwen3VlMoe  => "qwen3vlmoe",
            Self::Qwen35Moe   => "qwen35moe",
            Self::Qwen36MoeVl => "qwen36moevl",
            Self::Glm5Next    => "glm5next",
            Self::Unsupported(s) => s.as_str(),
        }
    }

    /// RoPE flavor as used by this arch's GGUF convention.
    ///   * Llama / Mistral: `Normal` (interleaved). The HF→GGUF converter
    ///     pre-permutes Q/K weights so this matches HF's rotated-half RoPE.
    ///   * Qwen-2 / Qwen-3 / Gemma 3: `NeoX` (rotated halves) — converters
    ///     don't permute; runtime applies the rotation directly.
    pub fn rope_type(&self) -> RopeType {
        match self {
            Self::Llama | Self::Mistral => RopeType::Normal,
            Self::Qwen2 | Self::Qwen3 | Self::Qwen35 | Self::Qwen35Moe | Self::Qwen3Moe |
            Self::Qwen3VlMoe | Self::Qwen36MoeVl |
            Self::Gemma3 | Self::Gemma3n | Self::Gemma4 => RopeType::NeoX,
            // VENDORED-LOCAL: glm5next is NoPE — `rope.dimension_count` is 0 and
            // the GGUF carries no rope tensors, so this value is never applied.
            // It is reported rather than left to the fallback so `nrob info`
            // does not imply a rotation the arch never performs.
            Self::Glm5Next => RopeType::NeoX,
            Self::Unsupported(_) => RopeType::NeoX,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch:           Architecture,
    pub vocab_size:     usize,
    pub context_length: usize,
    pub embedding_dim:  usize,
    pub n_layers:       usize,
    pub n_heads:        usize,
    pub n_kv_heads:     usize,
    pub head_dim:       usize,
    pub ff_dim:         usize,
    pub rms_eps:        f32,
    pub rope_theta:     f32,
    pub rope_dim:       usize,
    /// Gemma 3: optional softcap for the final logits (None for non-Gemma archs).
    pub final_logit_softcap: Option<f32>,
    /// Gemma 3: tok_embd output is scaled by `sqrt(embedding_dim)`.
    pub embedding_scale: bool,
    /// Gemma 3: sliding-window size for "local" attention layers. None = full attention everywhere.
    pub sliding_window: Option<usize>,
    /// Gemma 3: pattern for global vs sliding layers. Layer `i` is global iff `(i+1) % pattern == 0`.
    /// Defaults to 6 if `sliding_window` is set but the metadata key is absent.
    pub sliding_window_pattern: usize,
    /// Gemma 4: explicit per-layer SWA bool (true = sliding-window, false = global).
    /// When present, overrides the modular `sliding_window_pattern` rule.
    pub sliding_window_layers: Option<Vec<bool>>,
    /// Gemma 4: per-layer FFN sizes (each entry = ff_dim for layer i). When present,
    /// overrides the scalar `ff_dim` for that layer.
    pub ff_dims: Option<Vec<usize>>,
    /// Gemma 4: head_dim for sliding-window layers. None = use the scalar `head_dim`
    /// for all layers.
    pub head_dim_swa: Option<usize>,
    /// Gemma 4: RoPE freq_base for sliding-window layers. None = use scalar `rope_theta`.
    pub rope_theta_swa: Option<f32>,
    /// Gemma 4: RoPE dimension count for sliding-window layers. None = use scalar `rope_dim`.
    pub rope_dim_swa: Option<usize>,
    /// Gemma 3n: per-layer activation sparsity multiplier (precomputed inverse normal CDF
    /// of the target sparsity). Finite values trigger Gaussian-top-k masking on the FFN
    /// gate; `-inf` (the upstream sentinel for "no sparsity") means skip.
    pub activation_sparsity_scale: Option<Vec<f32>>,
    // VENDORED-LOCAL: glm5next per-layer attention kind.
    /// glm5next: per-layer recurrent mask, decoded from the
    /// `attention.head_count_kv` **array** (0 = KDA linear attention, 1 = full
    /// MLA). `true` = this layer is recurrent/linear. Covers all `block_count`
    /// entries including the trailing NextN block. `None` for every arch that
    /// stores `head_count_kv` as a scalar.
    pub recurrent_layers: Option<Vec<bool>>,
}

/// Backwards-compat alias for the v0 name.
pub type LlamaConfig = ModelConfig;

impl ModelConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch_str = g.architecture()?.to_string();
        let arch = Architecture::from_str(&arch_str);
        if !arch.supported() {
            return Err(LlamaError::UnsupportedArch(arch_str));
        }

        // Metadata keys are namespaced by architecture name. Most schemas
        // mirror Llama's, so we fall back to the `llama.*` namespace.
        let ns: &str = arch.name();

        let get_u64 = |suffix: &str| -> Result<u64> {
            let primary = format!("{ns}.{suffix}");
            if let Ok(v) = g.get_u64(&primary) { return Ok(v); }
            let fallback = format!("llama.{suffix}");
            Ok(g.get_u64(&fallback)?)
        };
        let get_f32 = |suffix: &str| -> Result<f32> {
            let primary = format!("{ns}.{suffix}");
            if let Ok(v) = g.get_f32(&primary) { return Ok(v); }
            let fallback = format!("llama.{suffix}");
            Ok(g.get_f32(&fallback)?)
        };
        let get_u64_or = |suffix: &str, default: u64| -> u64 {
            get_u64(suffix).unwrap_or(default)
        };

        let context_length = get_u64("context_length")? as usize;
        let embedding_dim  = get_u64("embedding_length")? as usize;
        let n_layers       = get_u64("block_count")? as usize;
        // Gemma 3n stores `feed_forward_length` as a per-layer array, not a
        // scalar — the stub loader doesn't use the value, so 0 is a safe sentinel.
        let ff_dim         = get_u64("feed_forward_length").map(|v| v as usize).unwrap_or(0);
        let n_heads        = get_u64("attention.head_count")? as usize;

        // VENDORED-LOCAL: glm5next stores `attention.head_count_kv` as a 0/1
        // array (0 = KDA linear-attention layer, 1 = full MLA), not a scalar.
        // Read the array form FIRST: the scalar read below fails on an array and
        // would silently fall back to `n_heads`, claiming 64 KV heads for a model
        // whose attention layers are absorbed-MLA MQA with a single latent row.
        let recurrent_layers: Option<Vec<bool>> = if arch != Architecture::Glm5Next {
            // gemma4 also stores this key as an array, but its entries are real
            // per-layer KV head counts, not a 0/1 recurrent mask. Only glm5next
            // means "0 = this layer is recurrent" by it.
            None
        } else {
            g.metadata()
            .get(&format!("{ns}.attention.head_count_kv"))
            .and_then(|v| v.as_array())
            .and_then(|a| match a {
                Array::U32(v) => Some(v.iter().map(|&x| x == 0).collect()),
                Array::I32(v) => Some(v.iter().map(|&x| x == 0).collect()),
                Array::U64(v) => Some(v.iter().map(|&x| x == 0).collect()),
                Array::I64(v) => Some(v.iter().map(|&x| x == 0).collect()),
                _ => None,
            })
        };

        // With the per-layer array, the KV head count is the array's maximum: the
        // full-attention layers are MQA over one latent row. Layers marked 0 keep
        // no KV cache at all — they carry a recurrent state instead.
        let n_kv_heads = match &recurrent_layers {
            Some(mask) if mask.iter().any(|&r| !r) => 1,
            Some(_) => return Err(LlamaError::Config(
                "attention.head_count_kv array marks every layer recurrent; the arch needs \n                 at least one full-attention layer".into())),
            None => get_u64_or("attention.head_count_kv", n_heads as u64) as usize,
        };
        let rms_eps        = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_theta     = get_f32("rope.freq_base").unwrap_or(10000.0);

        if n_heads == 0 || n_kv_heads == 0 || embedding_dim == 0 {
            return Err(LlamaError::Config("zero-sized dimension in config".into()));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(LlamaError::Config(format!(
                "n_heads ({n_heads}) not a multiple of n_kv_heads ({n_kv_heads})"
            )));
        }

        // Most archs derive head_dim from embed/n_heads. Gemma 3 stores
        // attention.key_length explicitly because key_dim can differ from
        // embed/n_heads. Same for value_length.
        let head_dim = match get_u64("attention.key_length") {
            Ok(v) => v as usize,
            Err(_) => {
                if embedding_dim % n_heads != 0 {
                    return Err(LlamaError::Config(format!(
                        "embedding_dim ({embedding_dim}) not divisible by n_heads ({n_heads}) and no attention.key_length"
                    )));
                }
                embedding_dim / n_heads
            }
        };

        let rope_dim = get_u64_or("rope.dimension_count", head_dim as u64) as usize;

        let vocab_size = if let Ok(v) = get_u64("vocab_size") {
            v as usize
        } else {
            match g.metadata().get("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
                Some(arr) => arr.len(),
                None => return Err(LlamaError::Config("can't determine vocab size".into())),
            }
        };

        let final_logit_softcap = get_f32("final_logit_softcapping").ok();
        let embedding_scale = matches!(arch, Architecture::Gemma3 | Architecture::Gemma3n | Architecture::Gemma4);

        let sliding_window = get_u64("attention.sliding_window").ok().map(|v| v as usize);
        // sliding_window_pattern can be either a scalar (Gemma 3) or a bool array
        // (Gemma 3n / Gemma 4). Look up the array form first so we don't lose it.
        let sliding_window_layers = g.metadata()
            .get(&format!("{ns}.attention.sliding_window_pattern"))
            .and_then(|v| v.as_array())
            .and_then(|a| if let Array::Bool(b) = a { Some(b.clone()) } else { None });
        let sliding_window_pattern = get_u64("attention.sliding_window_pattern")
            .map(|v| v as usize)
            .unwrap_or(6);

        // Per-layer FFN sizes (Gemma 3n / Gemma 4 store as i32 array).
        let ff_dims = g.metadata()
            .get(&format!("{ns}.feed_forward_length"))
            .and_then(|v| v.as_array())
            .and_then(|a| match a {
                Array::I32(v) => Some(v.iter().map(|&x| x as usize).collect()),
                Array::U32(v) => Some(v.iter().map(|&x| x as usize).collect()),
                _ => None,
            });

        // Gemma 4: SWA-specific head_dim / rope params.
        let head_dim_swa  = get_u64("attention.key_length_swa").ok().map(|v| v as usize);
        let rope_theta_swa = get_f32("rope.freq_base_swa").ok();
        let rope_dim_swa  = get_u64("rope.dimension_count_swa").ok().map(|v| v as usize);

        // Gemma 3n: per-layer activation-sparsity multiplier (already an inverse
        // normal CDF; -inf = no sparsity for that layer).
        let activation_sparsity_scale = g.metadata()
            .get(&format!("{ns}.activation_sparsity_scale"))
            .and_then(|v| v.as_array())
            .and_then(|a| if let Array::F32(v) = a { Some(v.clone()) } else { None });

        Ok(Self {
            arch,
            vocab_size,
            context_length,
            embedding_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ff_dim,
            rms_eps,
            rope_theta,
            rope_dim,
            final_logit_softcap,
            embedding_scale,
            sliding_window,
            sliding_window_pattern,
            sliding_window_layers,
            ff_dims,
            head_dim_swa,
            rope_theta_swa,
            rope_dim_swa,
            activation_sparsity_scale,
            recurrent_layers,
        })
    }

    /// True if layer `i` should use sliding-window attention. False = full/global attention.
    pub fn layer_uses_sliding_window(&self, i: usize) -> bool {
        if self.sliding_window.is_none() { return false; }
        // Explicit per-layer bool array (Gemma 3n / Gemma 4) wins over the modular rule.
        if let Some(arr) = &self.sliding_window_layers {
            if let Some(&b) = arr.get(i) { return b; }
        }
        // Gemma 3 convention: every Nth layer is global. The "+1" makes layers 5, 11, 17, ...
        // global with pattern=6 (matches HuggingFace gemma3 reference impl).
        (i + 1) % self.sliding_window_pattern != 0
    }

    /// Effective head_dim for layer `i`. Most archs ignore the layer; Gemma 4
    /// uses different head_dim for SWA vs global layers.
    pub fn layer_head_dim(&self, i: usize) -> usize {
        match (self.head_dim_swa, self.layer_uses_sliding_window(i)) {
            (Some(swa), true) => swa,
            _                 => self.head_dim,
        }
    }

    /// Effective RoPE freq_base for layer `i`.
    pub fn layer_rope_theta(&self, i: usize) -> f32 {
        match (self.rope_theta_swa, self.layer_uses_sliding_window(i)) {
            (Some(swa), true) => swa,
            _                 => self.rope_theta,
        }
    }

    /// Effective RoPE dimension count for layer `i`.
    pub fn layer_rope_dim(&self, i: usize) -> usize {
        match (self.rope_dim_swa, self.layer_uses_sliding_window(i)) {
            (Some(swa), true) => swa,
            _                 => self.rope_dim,
        }
    }

    /// Effective FFN dim for layer `i`. Falls back to scalar `ff_dim`.
    pub fn layer_ff_dim(&self, i: usize) -> usize {
        self.ff_dims.as_ref().and_then(|v| v.get(i).copied()).unwrap_or(self.ff_dim)
    }

    pub fn n_rep(&self) -> usize { self.n_heads / self.n_kv_heads }
}
