//! Model hyperparameters, read from the checkpoint's `config.json`
//! (`text_config`). Field meanings follow the reference `ModelArgs`; the
//! names on the right of each `field(...)` are the Hugging Face keys.
//!
//! Anything the forward pass does not implement is rejected here rather than
//! silently run wrong: another scoring function, top-k method, or `hc_mult`
//! would change the numerics without failing any shape check.

use std::path::Path;

use oaiy_engine::json::Json;
use oaiy_engine::{Error, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub norm_eps: f32,
    pub swiglu_limit: f32,
    // rope
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_seq_len: usize,
    // moe
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub route_scale: f32,
    // sparse attention
    pub window_size: usize,
    /// One per layer (MTP layers included); 0 = sliding window only.
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    // hyper-connections
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    // engram
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<usize>,
    pub engram_max_ngram_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    // tokens
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// The id every position of an image span carries (`<｜deepseek_image｜>`).
    pub image_token_id: u32,
    /// The vision tower, when the checkpoint has one.
    pub vision: Option<VisionConfig>,
}

/// `vision_config`: the ViT, its aligner and how images are sized for it.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub inter_dim: usize,
    pub patch_size: usize,
    pub rope_theta: f32,
    /// The aligner merges `downsample x downsample` patches into one token.
    pub downsample: usize,
    /// Most LLM tokens one image may take, delimiters included.
    pub max_tokens: usize,
    /// Smaller images are scaled up to at least this many pixels.
    pub min_pixels: usize,
    /// Wider images are squeezed to this aspect ratio (none in V4.1).
    pub max_wh_ratio: Option<usize>,
}

impl Config {
    /// Parse `<model_dir>/config.json`.
    pub fn load(model_dir: &Path) -> Result<Config> {
        let src = std::fs::read(model_dir.join("config.json"))?;
        Self::parse(&src)
    }

    pub fn parse(src: &[u8]) -> Result<Config> {
        let root = Json::parse(src)?;
        let t = root.get("text_config").ok_or_else(|| Error::Format("config.json has no text_config".into()))?;
        let need = |key: &str| -> Result<&Json> {
            t.get(key).ok_or_else(|| Error::Format(format!("config.json: text_config.{key} missing")))
        };
        let count = |v: &Json| v.as_i64().and_then(|v| usize::try_from(v).ok());
        let int = |key: &str| -> Result<usize> {
            let v = need(key)?;
            count(v).ok_or_else(|| Error::Format(format!("config.json: {key} = {v:?} is not a count")))
        };
        let num = |key: &str| -> Result<f32> { Ok(need(key)?.as_f64().unwrap_or(f64::NAN) as f32) };
        let list = |key: &str| -> Result<Vec<usize>> {
            let items = need(key)?.as_array().unwrap_or(&[]);
            items.iter().map(|v| count(v).ok_or_else(|| Error::Format(format!("config.json: bad {key}")))).collect()
        };
        let text = |key: &str| t.get(key).and_then(Json::as_str).unwrap_or("");

        // numerics the forward pass implements, and nothing else
        for (key, want) in [("scoring_func", "sqrtsoftplus"), ("topk_method", "noaux_tc"), ("hidden_act", "silu")] {
            if text(key) != want {
                return Err(Error::Unsupported(format!("{key} = {:?} (only {want:?} is implemented)", text(key))));
            }
        }
        let rope = need("rope_scaling")?;
        let rnum = |key: &str| rope.get(key).and_then(Json::as_f64).unwrap_or(f64::NAN) as f32;
        let int_or = |v: &Json, key: &str, dflt: i64| v.get(key).and_then(Json::as_i64).unwrap_or(dflt);
        let candidate = int_or(t, "candidate_source_layer_id", -1);

        let cfg = Config {
            vocab_size: int("vocab_size")?,
            dim: int("hidden_size")?,
            moe_inter_dim: int("moe_intermediate_size")?,
            n_layers: int("num_hidden_layers")?,
            n_heads: int("num_attention_heads")?,
            head_dim: int("head_dim")?,
            rope_head_dim: int("qk_rope_head_dim")?,
            q_lora_rank: int("q_lora_rank")?,
            o_lora_rank: int("o_lora_rank")?,
            o_groups: int("o_groups")?,
            norm_eps: num("rms_norm_eps")?,
            swiglu_limit: num("swiglu_limit")?,
            rope_theta: num("rope_theta")?,
            compress_rope_theta: num("compress_rope_theta")?,
            rope_factor: rnum("factor"),
            beta_fast: rnum("beta_fast"),
            beta_slow: rnum("beta_slow"),
            original_seq_len: int_or(rope, "original_max_position_embeddings", 0).max(0) as usize,
            n_routed_experts: int("n_routed_experts")?,
            n_activated_experts: int("num_experts_per_tok")?,
            route_scale: num("routed_scaling_factor")?,
            window_size: int("sliding_window")?,
            compress_ratios: list("compress_ratios")?,
            kv_source_layers: list("kv_source_layer_ids")?,
            index_source_layers: list("index_source_layer_ids")?,
            index_n_heads: int("index_n_heads")?,
            index_head_dim: int("index_head_dim")?,
            index_topk: int("index_topk")?,
            candidate_source_layer: usize::try_from(candidate).ok(),
            candidate_topk_blocks: int("candidate_topk_blocks")?,
            candidate_block_size: int("candidate_block_size")?,
            hc_mult: int("hc_mult")?,
            hc_sinkhorn_iters: int("hc_sinkhorn_iters")?,
            hc_eps: num("hc_eps")?,
            engram_layer_ids: list("engram_layer_ids")?,
            engram_num_embeddings: list("engram_num_embeddings")?,
            engram_max_ngram_size: int("engram_max_ngram_size")?,
            engram_n_heads: int("engram_n_heads")?,
            engram_head_dim: int("engram_head_dim")?,
            bos_token_id: int_or(&root, "bos_token_id", 0) as u32,
            eos_token_id: int_or(&root, "eos_token_id", 1) as u32,
            image_token_id: int_or(&root, "image_token_id", 129264) as u32,
            vision: match root.get("vision_config") {
                Some(v) if int_or(v, "num_hidden_layers", 0) > 0 => Some(VisionConfig::parse(v)?),
                _ => None,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Format(format!("config.json: {m}")));
        if !d_ok(self.norm_eps) || !d_ok(self.rope_factor) || !d_ok(self.beta_fast) || !d_ok(self.beta_slow) {
            return bad("a float field is missing".into());
        }
        if self.compress_ratios.len() < self.n_layers {
            return bad(format!("{} compress_ratios for {} layers", self.compress_ratios.len(), self.n_layers));
        }
        if !(self.n_heads * self.head_dim).is_multiple_of(self.o_groups) || self.rope_head_dim > self.head_dim {
            return bad("head geometry does not divide".into());
        }
        if self.engram_layer_ids.len() != self.engram_num_embeddings.len() {
            return bad("engram_layer_ids / engram_num_embeddings length mismatch".into());
        }
        if self.hc_mult != 4 {
            return Err(Error::Unsupported(format!("hc_mult {} (the hc split assumes 4)", self.hc_mult)));
        }
        Ok(())
    }

    /// Compress ratio of backbone layer `layer` (0 = sliding window only).
    pub fn ratio(&self, layer: usize) -> usize {
        self.compress_ratios[layer]
    }

    /// Hash columns per Engram lookup: (max n-gram - 1) n-gram sizes x heads.
    pub fn engram_cols(&self) -> usize {
        (self.engram_max_ngram_size - 1) * self.engram_n_heads
    }
}

impl VisionConfig {
    fn parse(v: &Json) -> Result<VisionConfig> {
        let int = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(Json::as_i64)
                .and_then(|x| usize::try_from(x).ok())
                .filter(|&x| x > 0)
                .ok_or_else(|| Error::Format(format!("config.json: vision_config.{key} missing or not a count")))
        };
        let cfg = VisionConfig {
            n_layers: int("num_hidden_layers")?,
            dim: int("hidden_size")?,
            n_heads: int("num_attention_heads")?,
            inter_dim: int("intermediate_size")?,
            patch_size: int("patch_size")?,
            rope_theta: v.get("rope_theta").and_then(Json::as_f64).unwrap_or(10000.0) as f32,
            downsample: int("downsample_ratio")?,
            max_tokens: int("max_image_tokens")?,
            min_pixels: int("min_pixels")?,
            max_wh_ratio: v.get("max_wh_ratio").and_then(Json::as_i64).and_then(|x| usize::try_from(x).ok()).filter(|&x| x > 0),
        };
        if !cfg.dim.is_multiple_of(cfg.n_heads) || !(cfg.dim / cfg.n_heads).is_multiple_of(4) {
            return Err(Error::Format("config.json: vision head size must split into two rotary halves".into()));
        }
        if cfg.max_tokens < 3 * cfg.downsample {
            return Err(Error::Format("config.json: vision max_image_tokens too small".into()));
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

fn d_ok(x: f32) -> bool {
    x.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_real_config_when_present() {
        let dir = std::path::PathBuf::from(std::env::var("DSV41_MODEL").unwrap_or_else(|_| r"E:\deepseek\model".into()));
        if !dir.join("config.json").exists() {
            return;
        }
        let c = Config::load(&dir).unwrap();
        assert_eq!((c.n_layers, c.dim, c.n_heads, c.head_dim), (40, 5120, 64, 512));
        assert_eq!(c.compress_ratios[2], 2);
        assert_eq!(c.candidate_source_layer, Some(20));
        assert_eq!(c.engram_cols(), 24);
        assert_eq!(c.norm_eps, 1e-20);
        assert_eq!(c.image_token_id, 129264);
        let v = c.vision.expect("vision_config");
        assert_eq!((v.n_layers, v.dim, v.n_heads, v.inter_dim, v.patch_size), (32, 1024, 16, 2816, 14));
        assert_eq!((v.downsample, v.max_tokens, v.min_pixels, v.max_wh_ratio), (3, 1024, 544 * 544, None));
    }

    #[test]
    fn rejects_unimplemented_scoring() {
        let src = br#"{"text_config": {"scoring_func": "softmax", "topk_method": "noaux_tc", "hidden_act": "silu"}}"#;
        assert!(matches!(Config::parse(src), Err(Error::Unsupported(_))));
    }
}
