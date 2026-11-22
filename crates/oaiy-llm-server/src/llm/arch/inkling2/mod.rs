//! Thinking Machines Lab Inkling-2 & Inkling-2-Mini (`inkling2`, `inkling2_loop`).
//!
//! Supports:
//!   * Mainline GGUF checkpoints (`general.architecture = "inkling2_loop"` and `"inkling2"`,
//!     e.g., `ggml-org/Inkling-2-Mini-GGUF`, `bartowski/thinkingmachines_Inkling-2-Mini-GGUF`,
//!     `lmstudio-community/Inkling-2-Mini-GGUF`, `unsloth/Inkling-2-Mini-GGUF/Mainline_Support`)
//!   * Companion `mmproj` GGUF vision projectors (`clip.vision.projector_type = "inkling2v"`)
//!   * Native Hugging Face safetensors checkpoints (`thinkingmachines/Inkling-2-Mini`),
//!     reading BF16 dense weights, NVFP4 / MXFP8 routed experts, and built-in `model.visual.*`
//!     vision tower directly in place with zero conversion step.

pub mod experts;
pub mod forward;
pub mod safetensors;
pub mod state;
pub mod vision;

use std::path::Path;
use std::sync::{Arc, Mutex};

use ggml_rs::chain::DeviceVec;
use ggml_rs::{Backend, QuantizedTensor, Tensor};
use gguf::{Array, GgufFile};
use tokenizer::Tokenizer;

use crate::llm::kv_cache::KvCache;
use crate::llm::loader::{load_tensor_f32, load_tensor_packed, TensorIndex, Weight};
use crate::llm::{Architecture, LlamaError, ModelConfig, Result};

use self::experts::{Bf16Linear, Inkling2Experts};
use self::state::Inkling2State;

#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub varying: bool,
}

#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub arch: Architecture,
    pub hidden: usize,
    pub n_phys_layers: usize,
    pub vocab_size_padded: usize,
    pub vocab_size_unpadded: usize,
    pub context_length: usize,
    pub head_counts: Vec<usize>,
    pub kv_head_counts: Vec<usize>,
    pub head_dims: Vec<usize>,
    pub is_swa: Vec<bool>,
    pub sliding_window: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_floor: usize,
    pub log_scale_alpha: f32,
    pub rms_eps: f32,
    pub conv_kernel: usize,
    pub attn_res_block_size: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub ff_dim: usize,
    pub route_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub final_logit_softcap: f32,
    pub loop_cfg: Option<LoopConfig>,
}

impl Inkling2Config {
    pub fn n_logical_slots(&self) -> usize {
        match &self.loop_cfg {
            Some(lc) => self.n_phys_layers + (lc.num_passes - 1) * (lc.core_end - lc.core_start),
            None => self.n_phys_layers,
        }
    }

    pub fn layer_phys(&self, il: usize) -> usize {
        if il < self.n_phys_layers {
            il
        } else if let Some(lc) = &self.loop_cfg {
            lc.core_start + (il - self.n_phys_layers) % (lc.core_end - lc.core_start)
        } else {
            il
        }
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |msg: &str| Err(LlamaError::Config(format!("inkling2: {msg}")));
        if self.hidden == 0
            || self.n_phys_layers == 0
            || self.d_rel == 0
            || self.rel_extent == 0
            || self.sliding_window == 0
            || self.attn_res_block_size == 0
            || self.log_scale_floor == 0
            || self.conv_kernel < 2
            || self.conv_kernel > 64
            || self.vocab_size_unpadded == 0
            || self.vocab_size_unpadded > self.vocab_size_padded
            || self.n_experts == 0
            || self.n_experts_used == 0
            || self.n_experts_used > self.n_experts
            || self.ff_dim == 0
        {
            return bad("invalid dimensions, vocabulary, or expert counts");
        }
        for v in [
            self.rms_eps,
            self.qk_clip,
            self.rel_clip,
            self.situ_beta,
            self.situ_linear_beta,
        ] {
            if !v.is_finite() || v <= 0.0 {
                return bad("normalization, clipping, and SiTU parameters must be finite and positive");
            }
        }
        if !self.log_scale_alpha.is_finite() || self.log_scale_alpha < 0.0 {
            return bad("invalid log_scale_alpha");
        }
        for i in 0..self.n_phys_layers {
            if self.kv_head_counts[i] == 0
                || self.head_counts[i] == 0
                || self.head_counts[i] % self.kv_head_counts[i] != 0
                || self.head_dims[i] == 0
            {
                return bad("invalid attention head counts or head_dim");
            }
        }
        if let Some(lc) = &self.loop_cfg {
            let min_passes = if lc.varying { 1 } else { 2 };
            if lc.num_passes < min_passes
                || lc.gate_rank == 0
                || lc.core_start == 0
                || lc.core_start >= lc.core_end
                || lc.core_end > self.n_phys_layers
                || lc.core_start % self.attn_res_block_size != 0
                || lc.core_end % self.attn_res_block_size != 0
            {
                return bad("invalid loop_config bounds or alignment with attn_res_block_size");
            }
        }
        Ok(())
    }

    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch_str = g.architecture()?.to_string();
        let arch = Architecture::from_str(&arch_str);
        if !matches!(arch, Architecture::Inkling2 | Architecture::Inkling2Loop) {
            return Err(LlamaError::UnsupportedArch(arch_str));
        }
        let ns = arch_str.as_str();

        let get_u64 = |suffix: &str| -> Result<u64> {
            Ok(g.get_u64(&format!("{ns}.{suffix}"))?)
        };
        let get_f32 = |suffix: &str| -> Result<f32> {
            Ok(g.get_f32(&format!("{ns}.{suffix}"))?)
        };
        let get_u64_arr = |suffix: &str, len: usize| -> Result<Vec<usize>> {
            let key = format!("{ns}.{suffix}");
            if let Some(val) = g.metadata().get(&key) {
                if let Some(arr) = val.as_array() {
                    let v: Option<Vec<usize>> = match arr {
                        Array::U32(a) => Some(a.iter().map(|&x| x as usize).collect()),
                        Array::I32(a) => Some(a.iter().map(|&x| x.max(0) as usize).collect()),
                        Array::U64(a) => Some(a.iter().map(|&x| x as usize).collect()),
                        Array::I64(a) => Some(a.iter().map(|&x| x.max(0) as usize).collect()),
                        _ => None,
                    };
                    if let Some(v) = v {
                        return Ok(v);
                    }
                }
            }
            let scalar = get_u64(suffix)? as usize;
            Ok(vec![scalar; len])
        };

        let n_phys_layers = get_u64("block_count")? as usize;
        let hidden = get_u64("embedding_length")? as usize;
        let context_length = get_u64("context_length")? as usize;
        let vocab_size_padded = get_u64("vocab_size")? as usize;
        let vocab_size_unpadded = get_u64("vocab_size_unpadded")? as usize;
        let head_counts = get_u64_arr("attention.head_count", n_phys_layers)?;
        let kv_head_counts = get_u64_arr("attention.head_count_kv", n_phys_layers)?;
        let key_len = get_u64("attention.key_length")? as usize;
        let key_len_swa = get_u64("attention.key_length_swa").map(|v| v as usize).unwrap_or(key_len);
        let sliding_window = get_u64("attention.sliding_window")? as usize;
        let is_swa = match g
            .metadata()
            .get(&format!("{ns}.attention.sliding_window_pattern"))
            .and_then(|v| v.as_array())
        {
            Some(Array::Bool(b)) if b.len() == n_phys_layers => b.clone(),
            _ => {
                return Err(LlamaError::Config(format!(
                    "{ns}.attention.sliding_window_pattern must be a bool array of length {n_phys_layers}"
                )));
            }
        };
        let head_dims: Vec<usize> = is_swa
            .iter()
            .map(|&s| if s { key_len_swa } else { key_len })
            .collect();

        let ff_dim = get_u64_arr("expert_feed_forward_length", n_phys_layers)
            .or_else(|_| get_u64_arr("feed_forward_length", n_phys_layers))?
            .into_iter()
            .next()
            .unwrap_or(0);

        let loop_cfg = if arch == Architecture::Inkling2Loop {
            Some(LoopConfig {
                core_start: get_u64("loop.start")? as usize,
                core_end: get_u64("loop.end")? as usize,
                num_passes: get_u64("loop.count")? as usize,
                gate_rank: get_u64("loop.gate_rank")? as usize,
                varying: g.get_bool(&format!("{ns}.loop.varying")).unwrap_or(false),
            })
        } else {
            None
        };

        let cfg = Self {
            arch,
            hidden,
            n_phys_layers,
            vocab_size_padded,
            vocab_size_unpadded,
            context_length,
            head_counts,
            kv_head_counts,
            head_dims,
            is_swa,
            sliding_window,
            d_rel: get_u64("attention.relative_length")? as usize,
            rel_extent: get_u64("attention.relative_extent")? as usize,
            qk_clip: get_f32("attention.qk_clip")?,
            rel_clip: get_f32("attention.relative_clip")?,
            log_scale_floor: get_u64("attention.log_scale_floor")? as usize,
            log_scale_alpha: get_f32("attention.log_scale_alpha")?,
            rms_eps: get_f32("attention.layer_norm_rms_epsilon")?,
            conv_kernel: get_u64("shortconv.l_cache")? as usize,
            attn_res_block_size: get_u64("attn_res.block_size")? as usize,
            n_experts: get_u64("expert_count")? as usize,
            n_experts_used: get_u64_arr("expert_used_count", n_phys_layers)?
                .into_iter()
                .next()
                .unwrap_or(8),
            ff_dim,
            route_scale: get_f32("expert_weights_scale").unwrap_or(1.0),
            situ_beta: get_f32("activation.situ_beta")?,
            situ_linear_beta: get_f32("activation.situ_linear_beta")?,
            final_logit_softcap: get_f32("final_logit_softcapping").unwrap_or(0.0),
            loop_cfg,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn new_state(&self) -> Inkling2State {
        let n_slots = self.n_logical_slots();
        let mut slot_is_swa = Vec::with_capacity(n_slots);
        let mut slot_kv_dims = Vec::with_capacity(n_slots);
        for il in 0..n_slots {
            let p = self.layer_phys(il);
            slot_is_swa.push(self.is_swa[p]);
            slot_kv_dims.push(self.kv_head_counts[p] * self.head_dims[p]);
        }
        Inkling2State::new(
            &slot_is_swa,
            &slot_kv_dims,
            self.sliding_window,
            self.hidden,
            self.conv_kernel,
        )
    }

    pub fn to_model_config(&self) -> ModelConfig {
        let n_heads = self.head_counts.iter().copied().max().unwrap_or(1);
        let n_kv_heads = self.kv_head_counts.iter().copied().max().unwrap_or(1);
        let head_dim = self.head_dims.iter().copied().max().unwrap_or(1);
        ModelConfig {
            arch: self.arch.clone(),
            vocab_size: self.vocab_size_unpadded,
            context_length: self.context_length,
            embedding_dim: self.hidden,
            n_layers: self.n_logical_slots(),
            n_heads,
            n_kv_heads,
            head_dim,
            ff_dim: self.ff_dim,
            rms_eps: self.rms_eps,
            rope_theta: 0.0,
            rope_dim: 0,
            final_logit_softcap: (self.final_logit_softcap > 0.0).then_some(self.final_logit_softcap),
            embedding_scale: false,
            sliding_window: Some(self.sliding_window),
            sliding_window_pattern: self.attn_res_block_size,
            sliding_window_layers: Some(self.is_swa.clone()),
            ff_dims: None,
            head_dim_swa: None,
            rope_theta_swa: None,
            rope_dim_swa: None,
            activation_sparsity_scale: None,
            recurrent_layers: None,
            kv_heads_per_layer: Some(self.kv_head_counts.clone()),
            mrope_sections: None,
            deepstack_layers: 0,
        }
    }

    /// Bytes per position for full-attention logical slots (SWA slots stay bounded at `<= 2 * sliding_window`).
    pub fn full_kv_bytes_per_position(&self) -> u64 {
        let mut vals = 0usize;
        for il in 0..self.n_logical_slots() {
            let p = self.layer_phys(il);
            if !self.is_swa[p] {
                vals += self.kv_head_counts[p] * self.head_dims[p];
            }
        }
        (2 * 4 * vals) as u64
    }
}

#[derive(Debug)]
pub(crate) enum EmbedTable {
    GgufQuant(QuantizedTensor),
    Bf16(Bf16Linear),
    F32(Vec<f32>, usize),
}

#[derive(Debug)]
pub(crate) enum AttnNorms {
    Shared(Vec<f32>),
    Separate {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
    },
}

#[derive(Debug)]
pub(crate) enum AttnNormsDev {
    Shared(DeviceVec),
    Separate {
        q: DeviceVec,
        k: DeviceVec,
        v: DeviceVec,
        /// Stacked `[q, k, v]` (`[3 * hidden]`) for single-dispatch `rmsnorm_multi_q8`.
        qkv: DeviceVec,
    },
}

/// Small per-layer F32 parameter vectors resident on GPU VRAM (~80 KB/layer) for single-submit
/// full-GPU decode (`seq == 1`).
#[derive(Debug)]
pub(crate) struct LayerDecodeDev {
    pub(crate) attn_res_query: Option<DeviceVec>,
    pub(crate) ffn_res_query: DeviceVec,
    /// Packed `[attn_rel_bias, attn_q_clip, attn_rel_clip]` for single-dispatch fused attention.
    pub(crate) attn_consts: DeviceVec,
    /// Packed `[attn_k_conv, attn_v_conv]` for single-dispatch fused K/V conv+clip+store.
    pub(crate) kv_conv_w: DeviceVec,
    pub(crate) attn_post_conv: DeviceVec,
    pub(crate) ffn_post_conv: DeviceVec,
}

#[derive(Debug)]
pub struct Inkling2Layer {
    pub(crate) is_swa: bool,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) attn_norms: AttnNorms,
    pub(crate) attn_norms_dev: Option<AttnNormsDev>,
    pub(crate) decode_dev: Option<LayerDecodeDev>,
    /// `[hidden]` for SWA (`i > 0`), `[3 * hidden]` for full-attention (`i > 0`), `None` for layer 0.
    pub(crate) attn_res_query: Option<Vec<f32>>,
    pub(crate) ffn_res_query: Vec<f32>,
    pub(crate) wq: Weight,
    pub(crate) wk: Weight,
    pub(crate) wv: Weight,
    pub(crate) wo: Weight,
    pub(crate) attn_gate: Weight,
    pub(crate) attn_q_clip: Vec<f32>,
    pub(crate) attn_sink_proj: Weight,
    pub(crate) attn_rel: Weight,
    /// Optional GPU-stacked `[wq, attn_rel, attn_sink_proj, attn_gate]` (`[q_dim + r_dim + n_heads + q_dim, hidden]`)
    /// so all 4 Q-stream projections run in ONE Q8_0 dispatch with zero duplicate VRAM.
    pub(crate) wqrsg: Option<Weight>,
    /// Optional GPU-stacked `[wk, attn_k_conv_proj]` on full-attention layers.
    pub(crate) wk_dyn: Option<Weight>,
    /// Optional GPU-stacked `[wv, attn_v_conv_proj]` on full-attention layers.
    pub(crate) wv_dyn: Option<Weight>,
    /// `[rel_extent, d_rel]` row-major.
    pub(crate) attn_rel_bias: Vec<f32>,
    pub(crate) attn_rel_clip: Vec<f32>,
    /// `[kv_dim, conv_kernel]` oldest-to-newest.
    pub(crate) attn_k_conv: Vec<f32>,
    pub(crate) attn_v_conv: Vec<f32>,
    pub(crate) attn_k_conv_proj: Option<Weight>,
    pub(crate) attn_v_conv_proj: Option<Weight>,
    /// `[hidden, conv_kernel]` oldest-to-newest.
    pub(crate) attn_post_conv: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) ffn_norm_dev: Option<DeviceVec>,
    pub(crate) ffn_gate_inp: Weight,
    /// Optional exact-F32 GPU copy of `ffn_gate_inp` padded with one zero row (`[n_experts + 1, hidden]`)
    /// so `QuantMoe::record_routed` runs directly after `matmul_f32_rows` in a single GPU submission.
    pub(crate) ffn_gate_inp_padded_dev: Option<DeviceVec>,
    pub(crate) exp_probs_b: Option<Vec<f32>>,
    pub(crate) experts: Inkling2Experts,
    pub(crate) ffn_post_conv: Vec<f32>,
}

#[derive(Debug)]
pub struct Inkling2LoopLayer {
    /// `[4, hidden]` row-major: `[Q, K, V, FFN]`.
    pub(crate) msg_query: Vec<f32>,
    pub(crate) msg_query_dev: Option<DeviceVec>,
    pub(crate) ctrl_down: [Weight; 4],
    pub(crate) ctrl_gain: [Weight; 4],
    pub(crate) ctrl_forget: [Weight; 4],
    /// Optional stacked Q+K+V F16 `[3 * gate_rank, 2 * hidden]` for single-dispatch `inkling2_ctrl_cat_down_multi`.
    pub(crate) ctrl_down_qkv_f16: Option<DeviceVec>,
    /// Optional FFN F16 `[gate_rank, 2 * hidden]` for single-dispatch `inkling2_ctrl_cat_down`.
    pub(crate) ctrl_down_ffn_f16: Option<DeviceVec>,
    /// Optional stacked Q+K+V F16 `[3 * 2 * hidden, gate_rank]` for single-dispatch `inkling2_ctrl_up_blend_multi`.
    pub(crate) ctrl_gf_qkv_f16: Option<DeviceVec>,
    /// Optional FFN F16 `[2 * hidden, gate_rank]` for single-dispatch `inkling2_ctrl_up_blend`.
    pub(crate) ctrl_gf_ffn_f16: Option<DeviceVec>,
}

#[derive(Debug)]
pub struct Inkling2Model {
    pub config: ModelConfig,
    pub cfg: Inkling2Config,
    pub tokenizer: Tokenizer,
    pub backend: Arc<dyn Backend>,
    pub(crate) embed_table: EmbedTable,
    pub(crate) embed_norm: Vec<f32>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) output_norm_dev: Option<DeviceVec>,
    pub(crate) output_res_query: Vec<f32>,
    pub(crate) output_res_query_dev: Option<DeviceVec>,
    pub(crate) output: Weight,
    pub(crate) layers: Vec<Inkling2Layer>,
    pub(crate) loop_layers: Vec<Inkling2LoopLayer>,
    /// True when every layer weight, QuantMoe, and decode vector lives on `DeviceChain` VRAM,
    /// enabling single-submit 48-layer GPU decode for `seq == 1`.
    pub(crate) full_gpu_decode: bool,
    pub(crate) state: Mutex<Inkling2State>,
    pub(crate) gpu_scratch: Mutex<Option<forward::GpuScratch>>,
}

impl Inkling2Model {
    /// Load an Inkling-2 or Inkling-2-Loop model from a GGUF file.
    pub fn from_gguf(g: &GgufFile, backend: Arc<dyn Backend>) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(g)?;
        let tokenizer = Tokenizer::from_gguf(g)?;
        if tokenizer.vocab_size() != cfg.vocab_size_unpadded {
            return Err(LlamaError::Config(format!(
                "inkling2: tokenizer vocab ({}) != vocab_size_unpadded ({})",
                tokenizer.vocab_size(),
                cfg.vocab_size_unpadded
            )));
        }
        let idx = TensorIndex::new(g);
        let req = |name: &str| -> Result<&gguf::TensorInfo> {
            idx.info(name)
                .ok_or_else(|| LlamaError::MissingTensor(name.to_string()))
        };
        let load_vec = |name: &str| -> Result<Vec<f32>> {
            Ok(load_tensor_f32(g, req(name)?)?.data().to_vec())
        };
        let load_w = |name: &str| -> Result<Weight> {
            Ok(Weight::load(g, req(name)?)?.to_device(backend.as_ref()))
        };
        let load_w_host = |name: &str| -> Result<Weight> { Ok(Weight::load(g, req(name)?)?) };
        let stack_quant = |ws: &[&Weight]| -> Option<Weight> {
            let chain = backend.chain()?;
            let first = match ws.first()? {
                Weight::Quant(qt) => qt,
                _ => return None,
            };
            let dtype = first.dtype();
            let in_dim = first.shape().get(1).copied()?;
            let mut total_rows = 0usize;
            let mut total_bytes = 0usize;
            for w in ws {
                match w {
                    Weight::Quant(qt)
                        if qt.dtype() == dtype
                            && qt.shape().len() == 2
                            && qt.shape()[1] == in_dim =>
                    {
                        total_rows += qt.shape()[0];
                        total_bytes += qt.bytes().len();
                    }
                    _ => return None,
                }
            }
            let mut bytes = Vec::with_capacity(total_bytes);
            for w in ws {
                if let Weight::Quant(qt) = w {
                    bytes.extend_from_slice(qt.bytes());
                }
            }
            let stacked = Weight::Quant(ggml_rs::QuantizedTensor::from_bytes_cpu(
                bytes,
                vec![total_rows, in_dim],
                dtype,
            ))
            .to_device(backend.as_ref());
            match &stacked {
                Weight::Quant(qt) if chain.holds(qt) => Some(stacked),
                _ => None,
            }
        };

        // Keep token_embd zero-copy packed when quantized (Q8_0, etc.) to avoid ~1.8 GB F32 inflation.
        let tok_info = req("token_embd.weight")?;
        let embed_table = if ggml_quants::is_supported(tok_info.dtype) && tok_info.dtype != ggml_quants::GgmlType::F32 {
            EmbedTable::GgufQuant(load_tensor_packed(g, tok_info)?)
        } else {
            EmbedTable::F32(load_tensor_f32(g, tok_info)?.data().to_vec(), cfg.hidden)
        };
        let embed_norm = load_vec("token_embd_norm.weight")?;
        let output_norm = load_vec("output_norm.weight")?;
        let output_res_query = load_vec("output_res_query.weight")?;
        let output = load_w("output.weight")?;
        let (output_norm_dev, output_res_query_dev) = if let Some(chain) = backend.chain() {
            let on = chain.vec(output_norm.len());
            chain.upload(&on, &output_norm);
            let oq = chain.vec(output_res_query.len());
            chain.upload(&oq, &output_res_query);
            (Some(on), Some(oq))
        } else {
            (None, None)
        };

        let mut layers = Vec::with_capacity(cfg.n_phys_layers);
        for i in 0..cfg.n_phys_layers {
            let is_swa = cfg.is_swa[i];
            let attn_norms = if is_swa {
                AttnNorms::Shared(load_vec(&format!("blk.{i}.attn_norm.weight"))?)
            } else {
                AttnNorms::Separate {
                    q: load_vec(&format!("blk.{i}.attn_q_inp_norm.weight"))?,
                    k: load_vec(&format!("blk.{i}.attn_k_inp_norm.weight"))?,
                    v: load_vec(&format!("blk.{i}.attn_v_inp_norm.weight"))?,
                }
            };
            let attn_res_query = if i > 0 {
                Some(load_vec(&format!("blk.{i}.attn_res_query.weight"))?)
            } else {
                None
            };
            let (k_conv_host, v_conv_host) = if is_swa {
                (None, None)
            } else {
                (
                    Some(load_w_host(&format!("blk.{i}.attn_k_conv_proj.weight"))?),
                    Some(load_w_host(&format!("blk.{i}.attn_v_conv_proj.weight"))?),
                )
            };
            let wq_host = load_w_host(&format!("blk.{i}.attn_q.weight"))?;
            let wk_host = load_w_host(&format!("blk.{i}.attn_k.weight"))?;
            let wv_host = load_w_host(&format!("blk.{i}.attn_v.weight"))?;
            let attn_rel_host = load_w_host(&format!("blk.{i}.attn_rel.weight"))?;
            let attn_sink_host = load_w_host(&format!("blk.{i}.attn_sink_proj.weight"))?;
            let attn_gate_host = load_w_host(&format!("blk.{i}.attn_gate.weight"))?;

            // Stack Q-stream `[wq, attn_rel, attn_sink_proj, attn_gate]` (and full-attn `[wk, dyn_k]`
            // / `[wv, dyn_v]`) into single GPU Q8_0 matrices with zero duplicate VRAM.
            let wqrsg = stack_quant(&[&wq_host, &attn_rel_host, &attn_sink_host, &attn_gate_host]);
            let (wq, attn_rel, attn_sink_proj, attn_gate) = if wqrsg.is_some() {
                (wq_host, attn_rel_host, attn_sink_host, attn_gate_host)
            } else {
                (
                    wq_host.to_device(backend.as_ref()),
                    attn_rel_host.to_device(backend.as_ref()),
                    attn_sink_host.to_device(backend.as_ref()),
                    attn_gate_host.to_device(backend.as_ref()),
                )
            };
            let wk_dyn = k_conv_host
                .as_ref()
                .and_then(|kp| stack_quant(&[&wk_host, kp]));
            let (wk, attn_k_conv_proj) = if wk_dyn.is_some() {
                (wk_host, k_conv_host)
            } else {
                (
                    wk_host.to_device(backend.as_ref()),
                    k_conv_host.map(|w| w.to_device(backend.as_ref())),
                )
            };
            let wv_dyn = v_conv_host
                .as_ref()
                .and_then(|vp| stack_quant(&[&wv_host, vp]));
            let (wv, attn_v_conv_proj) = if wv_dyn.is_some() {
                (wv_host, v_conv_host)
            } else {
                (
                    wv_host.to_device(backend.as_ref()),
                    v_conv_host.map(|w| w.to_device(backend.as_ref())),
                )
            };
            let exp_probs_b = match idx.info(&format!("blk.{i}.exp_probs_b.bias")) {
                Some(info) => Some(load_tensor_f32(g, info)?.data().to_vec()),
                None => None,
            };
            let experts = Inkling2Experts::from_gguf(
                g,
                req(&format!("blk.{i}.ffn_gate_up_exps.weight"))?,
                req(&format!("blk.{i}.ffn_down_exps.weight"))?,
                cfg.n_experts,
                cfg.hidden,
                cfg.ff_dim,
            )?;

            let ffn_res_query = load_vec(&format!("blk.{i}.ffn_res_query.weight"))?;
            let attn_q_clip = load_vec(&format!("blk.{i}.attn_q_clip.weight"))?;
            let attn_rel_bias = load_vec(&format!("blk.{i}.attn_rel_bias.weight"))?;
            let attn_rel_clip = load_vec(&format!("blk.{i}.attn_rel_clip.weight"))?;
            let attn_k_conv = load_vec(&format!("blk.{i}.attn_k_conv.weight"))?;
            let attn_v_conv = load_vec(&format!("blk.{i}.attn_v_conv.weight"))?;
            let attn_post_conv = load_vec(&format!("blk.{i}.attn_post_conv.weight"))?;
            let ffn_norm = load_vec(&format!("blk.{i}.ffn_norm.weight"))?;
            let ffn_gate_inp = load_w(&format!("blk.{i}.ffn_gate_inp.weight"))?;
            let ffn_post_conv = load_vec(&format!("blk.{i}.ffn_post_conv.weight"))?;

            // Upload tiny per-layer parameter vectors (~90 KB) and padded exact-F32 router (~1.2 MB)
            // when a DeviceChain backend is active so attention + FFN + single-submit GPU decode run
            // with zero host hops.
            let (attn_norms_dev, decode_dev, ffn_norm_dev, ffn_gate_inp_padded_dev) = if let Some(chain) = backend.chain() {
                let up_vec = |vals: &[f32]| -> DeviceVec {
                    let v = chain.vec(vals.len());
                    chain.upload(&v, vals);
                    v
                };
                let an_dev = match &attn_norms {
                    AttnNorms::Shared(w) => AttnNormsDev::Shared(up_vec(w)),
                    AttnNorms::Separate { q, k, v } => {
                        let mut stacked = Vec::with_capacity(q.len() + k.len() + v.len());
                        stacked.extend_from_slice(q);
                        stacked.extend_from_slice(k);
                        stacked.extend_from_slice(v);
                        AttnNormsDev::Separate {
                            q: up_vec(q),
                            k: up_vec(k),
                            v: up_vec(v),
                            qkv: up_vec(&stacked),
                        }
                    }
                };
                let mut attn_consts_host =
                    Vec::with_capacity(attn_rel_bias.len() + attn_q_clip.len() + attn_rel_clip.len());
                attn_consts_host.extend_from_slice(&attn_rel_bias);
                attn_consts_host.extend_from_slice(&attn_q_clip);
                attn_consts_host.extend_from_slice(&attn_rel_clip);
                let mut kv_conv_host = Vec::with_capacity(attn_k_conv.len() + attn_v_conv.len());
                kv_conv_host.extend_from_slice(&attn_k_conv);
                kv_conv_host.extend_from_slice(&attn_v_conv);
                let dec_dev = LayerDecodeDev {
                    attn_res_query: attn_res_query.as_deref().map(&up_vec),
                    ffn_res_query: up_vec(&ffn_res_query),
                    attn_consts: up_vec(&attn_consts_host),
                    kv_conv_w: up_vec(&kv_conv_host),
                    attn_post_conv: up_vec(&attn_post_conv),
                    ffn_post_conv: up_vec(&ffn_post_conv),
                };
                let fn_dev = up_vec(&ffn_norm);
                let router_dev = match &ffn_gate_inp {
                    Weight::Dense(t) if !t.is_device() && t.rank() == 2 && t.dim(0) == cfg.n_experts && t.dim(1) == cfg.hidden => {
                        let mut padded = Vec::with_capacity((cfg.n_experts + 1) * cfg.hidden);
                        padded.extend_from_slice(t.data());
                        padded.resize((cfg.n_experts + 1) * cfg.hidden, 0.0);
                        Some(up_vec(&padded))
                    }
                    _ => None,
                };
                (Some(an_dev), Some(dec_dev), Some(fn_dev), router_dev)
            } else {
                (None, None, None, None)
            };

            layers.push(Inkling2Layer {
                is_swa,
                n_heads: cfg.head_counts[i],
                n_kv_heads: cfg.kv_head_counts[i],
                head_dim: cfg.head_dims[i],
                attn_norms,
                attn_norms_dev,
                decode_dev,
                attn_res_query,
                ffn_res_query,
                wq,
                wk,
                wv,
                wo: load_w(&format!("blk.{i}.attn_output.weight"))?,
                attn_gate,
                attn_q_clip,
                attn_sink_proj,
                attn_rel,
                wqrsg,
                wk_dyn,
                wv_dyn,
                attn_rel_bias,
                attn_rel_clip,
                attn_k_conv,
                attn_v_conv,
                attn_k_conv_proj,
                attn_v_conv_proj,
                attn_post_conv,
                ffn_norm,
                ffn_norm_dev,
                ffn_gate_inp,
                ffn_gate_inp_padded_dev,
                exp_probs_b,
                experts,
                ffn_post_conv,
            });
        }

        // Optional F16 GPU upload for dense loop controllers (~56 MiB total across all 12 core layers)
        // so all 3 Q/K/V cross-pass controllers execute in one chained GPU submit instead of host F32.
        let load_ctrl_w = |name: &str| -> Result<Weight> {
            let w = Weight::load(g, req(name)?)?;
            #[cfg(feature = "webgpu")]
            if let Weight::Dense(ref t) = w {
                if t.rank() == 2
                    && t.dim(1) % 2 == 0
                    && backend
                        .vram_status()
                        .is_some_and(|(avail, _)| avail > 512 * 1024 * 1024)
                {
                    if let Some(wgpu) = backend.as_any().downcast_ref::<ggml_rs_wgpu::WgpuBackend>() {
                        if let Ok(packed) = wgpu.half_linear(t.data().to_vec(), t.dim(0), t.dim(1)) {
                            return Ok(Weight::Packed(packed));
                        }
                    }
                }
            }
            Ok(w.to_device(backend.as_ref()))
        };

        let mut loop_layers = Vec::new();
        if let Some(lc) = &cfg.loop_cfg {
            let ctrl_names = ["q", "k", "v", "ffn"];
            for ip in lc.core_start..lc.core_end {
                let msg_query = load_vec(&format!("blk.{ip}.loop_msg_query.weight"))?;
                let msg_query_dev = backend.chain().map(|chain| {
                    let v = chain.vec(msg_query.len());
                    chain.upload(&v, &msg_query);
                    v
                });
                let mut ctrl_down = Vec::with_capacity(4);
                let mut ctrl_gain = Vec::with_capacity(4);
                let mut ctrl_forget = Vec::with_capacity(4);
                let mut d_qkv_host: Vec<f32> = Vec::new();
                let mut gf_qkv_host: Vec<f32> = Vec::new();
                let mut ctrl_down_ffn_f16: Option<DeviceVec> = None;
                let mut ctrl_gf_ffn_f16: Option<DeviceVec> = None;
                for (ci, name) in ctrl_names.into_iter().enumerate() {
                    let d_raw = load_vec(&format!("blk.{ip}.loop_{name}_ctrl_down.weight"))?;
                    let g_raw = load_vec(&format!("blk.{ip}.loop_{name}_ctrl_gain.weight"))?;
                    let f_raw = load_vec(&format!("blk.{ip}.loop_{name}_ctrl_forget.weight"))?;
                    if let Some(chain) = backend.chain() {
                        if cfg.hidden % 2 == 0 {
                            if ci < 3 {
                                d_qkv_host.extend_from_slice(&d_raw);
                            } else {
                                ctrl_down_ffn_f16 = chain.vec_f16_rounded(&d_raw);
                            }
                        }
                        if lc.gate_rank % 2 == 0 && lc.gate_rank <= 256 {
                            if ci < 3 {
                                gf_qkv_host.extend_from_slice(&g_raw);
                                gf_qkv_host.extend_from_slice(&f_raw);
                            } else {
                                let mut both = Vec::with_capacity(g_raw.len() + f_raw.len());
                                both.extend_from_slice(&g_raw);
                                both.extend_from_slice(&f_raw);
                                ctrl_gf_ffn_f16 = chain.vec_f16_rounded(&both);
                            }
                        }
                    }
                    ctrl_down.push(load_ctrl_w(&format!("blk.{ip}.loop_{name}_ctrl_down.weight"))?);
                    ctrl_gain.push(load_ctrl_w(&format!("blk.{ip}.loop_{name}_ctrl_gain.weight"))?);
                    ctrl_forget.push(load_ctrl_w(&format!("blk.{ip}.loop_{name}_ctrl_forget.weight"))?);
                }
                let ctrl_down_qkv_f16 = backend
                    .chain()
                    .filter(|_| !d_qkv_host.is_empty())
                    .and_then(|c| c.vec_f16_rounded(&d_qkv_host));
                let ctrl_gf_qkv_f16 = backend
                    .chain()
                    .filter(|_| !gf_qkv_host.is_empty())
                    .and_then(|c| c.vec_f16_rounded(&gf_qkv_host));
                loop_layers.push(Inkling2LoopLayer {
                    msg_query,
                    msg_query_dev,
                    ctrl_down: ctrl_down.try_into().expect("4 controllers"),
                    ctrl_gain: ctrl_gain.try_into().expect("4 controllers"),
                    ctrl_forget: ctrl_forget.try_into().expect("4 controllers"),
                    ctrl_down_qkv_f16,
                    ctrl_down_ffn_f16,
                    ctrl_gf_qkv_f16,
                    ctrl_gf_ffn_f16,
                });
            }
        }

        // Offload routed MoE expert layers to GPU VRAM while remaining budget exceeds 640 MiB
        // (reserving headroom for the vision projector ~330 MiB, controllers ~56 MiB, and scratch).
        // Prioritize looped core layers first since each runs 3x per token, then postlude (Q8_0),
        // then prelude.
        if backend.chain().is_some() {
            const VISION_AND_SCRATCH_RESERVE: usize = 640 * 1024 * 1024;
            let mut order: Vec<usize> = Vec::with_capacity(cfg.n_phys_layers);
            if let Some(lc) = &cfg.loop_cfg {
                for i in lc.core_start..lc.core_end {
                    order.push(i);
                }
                for i in (lc.core_end..cfg.n_phys_layers).rev() {
                    order.push(i);
                }
                for i in 0..lc.core_start {
                    order.push(i);
                }
            } else {
                for i in 0..cfg.n_phys_layers {
                    order.push(i);
                }
            }
            let mut n_gpu_exp = 0usize;
            let mut n_quant_moe = 0usize;
            for idx_l in order {
                let bias = layers[idx_l].exp_probs_b.as_deref();
                if let Some(gpu_exps) = layers[idx_l].experts.try_upload_to_gpu(
                    backend.as_ref(),
                    VISION_AND_SCRATCH_RESERVE,
                    bias,
                    cfg.route_scale,
                    cfg.situ_beta,
                    cfg.situ_linear_beta,
                ) {
                    if gpu_exps.quant_moe().is_some() {
                        n_quant_moe += 1;
                    }
                    layers[idx_l].experts = gpu_exps;
                    n_gpu_exp += 1;
                }
            }
            if n_gpu_exp > 0 {
                eprintln!(
                    "inkling2: offloaded {n_gpu_exp}/{} MoE expert layers to GPU VRAM ({n_quant_moe} fused QuantMoe)",
                    layers.len()
                );
            }
        }

        // Single-submit full-GPU decode is eligible when every layer, loop controller, QuantMoe,
        // and output weight is resident on `DeviceChain` VRAM.
        let full_gpu_decode = if let Some(chain) = backend.chain() {
            let on_dev = |w: &Weight| match w {
                Weight::Quant(qt) => chain.holds(qt),
                Weight::Packed(p) => chain.holds_exl3(p.as_ref()),
                _ => false,
            };
            let max_blocks = cfg.n_phys_layers.div_ceil(cfg.attn_res_block_size) + 2;
            max_blocks < 32
                && output_norm_dev.is_some()
                && output_res_query_dev.is_some()
                && on_dev(&output)
                && layers.iter().all(|l| {
                    l.head_dim <= 256
                        && l.attn_norms_dev.is_some()
                        && l.decode_dev.is_some()
                        && l.ffn_norm_dev.is_some()
                        && l.ffn_gate_inp_padded_dev.is_some()
                        && l.experts.quant_moe().is_some_and(|q| chain.holds_experts(q))
                        && (l.wqrsg.as_ref().is_some_and(&on_dev)
                            || (on_dev(&l.wq)
                                && on_dev(&l.attn_gate)
                                && on_dev(&l.attn_sink_proj)
                                && on_dev(&l.attn_rel)))
                        && (l.wk_dyn.as_ref().is_some_and(&on_dev)
                            || (on_dev(&l.wk) && l.attn_k_conv_proj.as_ref().is_none_or(&on_dev)))
                        && (l.wv_dyn.as_ref().is_some_and(&on_dev)
                            || (on_dev(&l.wv) && l.attn_v_conv_proj.as_ref().is_none_or(&on_dev)))
                        && on_dev(&l.wo)
                })
                && loop_layers.iter().all(|ll| {
                    ll.msg_query_dev.is_some()
                        && ((ll.ctrl_down_qkv_f16.is_some() && ll.ctrl_down_ffn_f16.is_some())
                            || ll.ctrl_down.iter().all(&on_dev))
                        && ((ll.ctrl_gf_qkv_f16.is_some() && ll.ctrl_gf_ffn_f16.is_some())
                            || (ll.ctrl_gain.iter().all(&on_dev)
                                && ll.ctrl_forget.iter().all(&on_dev)))
                })
        } else {
            false
        };
        if full_gpu_decode {
            eprintln!("inkling2: single-submit full-GPU decode enabled (all {} logical layers on GPU)", cfg.n_logical_slots());
        }

        let config = cfg.to_model_config();
        let state = cfg.new_state();
        Ok(Self {
            config,
            cfg,
            tokenizer,
            backend,
            embed_table,
            embed_norm,
            output_norm,
            output_norm_dev,
            output_res_query,
            output_res_query_dev,
            output,
            layers,
            loop_layers,
            full_gpu_decode,
            state: Mutex::new(state),
            gpu_scratch: Mutex::new(None),
        })
    }

    /// Load directly from a Hugging Face safetensors directory (`thinkingmachines/Inkling-2-Mini`).
    pub fn from_safetensors(
        dir: &Path,
        backend: Arc<dyn Backend>,
        expert_cache_bytes: usize,
        load_vision: bool,
    ) -> Result<(Self, Option<vision::Inkling2Vision>)> {
        safetensors::load_from_safetensors(dir, backend, expert_cache_bytes, load_vision)
    }

    pub fn reset(&self) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).reset();
        if let Some(scratch) = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            scratch.reset_decode_state();
        }
    }

    pub fn state_pos(&self) -> usize {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).pos
    }

    pub(crate) fn embed_row(&self, token_id: u32) -> Result<Vec<f32>> {
        let idx = token_id as usize;
        if idx >= self.cfg.vocab_size_padded {
            return Err(LlamaError::Config(format!(
                "token id {token_id} out of range (vocab_padded = {})",
                self.cfg.vocab_size_padded
            )));
        }
        match &self.embed_table {
            EmbedTable::GgufQuant(qt) => {
                let d = self.cfg.hidden;
                let dtype = qt.dtype();
                let row_bytes = (d / dtype.block_size()) * dtype.type_size();
                let raw = &qt.bytes()[idx * row_bytes..(idx + 1) * row_bytes];
                let mut out = vec![0.0f32; d];
                ggml_quants::dequantize(dtype, raw, &mut out)?;
                Ok(out)
            }
            EmbedTable::Bf16(bf) => Ok(bf.row_f32(idx)),
            EmbedTable::F32(data, d) => Ok(data[idx * d..(idx + 1) * d].to_vec()),
        }
    }

    /// Run a forward pass compatible with `crate::llm::Model::forward` (returns `[seq, vocab]`
    /// logits and syncs `kv.len` with internal recurrent + KV state; resets when `kv.len == 0`).
    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Tensor> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if kv.len == 0 && st.pos != 0 {
            st.reset();
            if let Some(scratch) = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                scratch.reset_decode_state();
            }
        }
        let logits = self.forward_internal(tokens, &[], &mut st, forward::OutputMode::All)?;
        kv.len = st.pos;
        Ok(logits)
    }

    /// Run a forward pass with spliced soft vision embeddings, returning last-token logits
    /// (`[1, unpadded_vocab_size]`).
    pub fn forward_spliced(
        &self,
        tokens: &[u32],
        soft: &[(usize, &[f32])],
        kv: &mut KvCache,
    ) -> Result<Tensor> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if kv.len == 0 && st.pos != 0 {
            st.reset();
            if let Some(scratch) = self.gpu_scratch.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                scratch.reset_decode_state();
            }
        }
        let logits = self.forward_internal(tokens, soft, &mut st, forward::OutputMode::LastOnly)?;
        kv.len = st.pos;
        Ok(logits)
    }

    /// Advance KV and short-convolution states across an intermediate prompt chunk without
    /// computing unembed logits or transferring logits back to the host.
    pub fn prefill_only(&self, tokens: &[u32], soft: &[(usize, &[f32])]) -> Result<()> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let _ = self.forward_internal(tokens, soft, &mut st, forward::OutputMode::None)?;
        Ok(())
    }

    /// Fast server step/prefill helper: returns last-token logits directly (`Vec<f32>` of length `vocab_size_unpadded`).
    pub fn forward_last(&self, tokens: &[u32], soft: &[(usize, &[f32])]) -> Result<Vec<f32>> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let t = self.forward_internal(tokens, soft, &mut st, forward::OutputMode::LastOnly)?;
        Ok(t.data().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_rs::CpuBackend;

    fn det_vec(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i.wrapping_mul(37).wrapping_add(seed * 101)) as f32) * 0.13).sin() * scale)
            .collect()
    }

    fn det_w(n: usize, k: usize, seed: usize, scale: f32) -> Weight {
        Weight::Dense(Tensor::from_vec(det_vec(n * k, seed, scale), vec![n, k]))
    }

    /// Build a tiny looped `Inkling2Model` (6 physical layers: prelude 0..2, core 2..4 x 3 passes, postlude 4..6;
    /// attn_res_block_size = 2; SWA on even layers, full-attn on odd layers) and verify that chunked prefill
    /// matches token-by-token decode to floating-point precision, and spliced vision tokens bypass embed_norm.
    #[test]
    fn tiny_looped_model_prefill_matches_token_by_token_and_soft_splice() {
        let hidden = 16;
        let n_phys = 6;
        let n_heads = 2;
        let n_kv = 1;
        let head_dim = 8;
        let d_rel = 4;
        let rel_extent = 8;
        let conv_k = 4;
        let n_exp = 4;
        let top_k = 2;
        let ff = 8;
        let vocab_padded = 24;
        let vocab_unpadded = 20;
        let gate_rank = 4;

        let is_swa = vec![true, false, true, false, true, false];
        let cfg = Inkling2Config {
            arch: Architecture::Inkling2Loop,
            hidden,
            n_phys_layers: n_phys,
            vocab_size_padded: vocab_padded,
            vocab_size_unpadded: vocab_unpadded,
            context_length: 64,
            head_counts: vec![n_heads; n_phys],
            kv_head_counts: vec![n_kv; n_phys],
            head_dims: vec![head_dim; n_phys],
            is_swa: is_swa.clone(),
            sliding_window: 8,
            d_rel,
            rel_extent,
            qk_clip: 22.627_417,
            rel_clip: 8.0,
            log_scale_floor: 4,
            log_scale_alpha: 0.1,
            rms_eps: 1e-6,
            conv_kernel: conv_k,
            attn_res_block_size: 2,
            n_experts: n_exp,
            n_experts_used: top_k,
            ff_dim: ff,
            route_scale: 2.0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            final_logit_softcap: 30.0,
            loop_cfg: Some(LoopConfig {
                core_start: 2,
                core_end: 4,
                num_passes: 3,
                gate_rank,
                varying: false,
            }),
        };
        assert_eq!(cfg.n_logical_slots(), 10); // 6 + 2 * 2

        let mut layers = Vec::new();
        for i in 0..n_phys {
            let swa = is_swa[i];
            let s = 100 * (i + 1);
            let attn_norms = if swa {
                AttnNorms::Shared(vec![1.0; hidden])
            } else {
                AttnNorms::Separate {
                    q: vec![1.0; hidden],
                    k: vec![1.0; hidden],
                    v: vec![1.0; hidden],
                }
            };
            layers.push(Inkling2Layer {
                is_swa: swa,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                attn_norms,
                attn_norms_dev: None,
                decode_dev: None,
                attn_res_query: if i > 0 {
                    Some(det_vec(if swa { hidden } else { 3 * hidden }, s + 1, 0.2))
                } else {
                    None
                },
                ffn_res_query: det_vec(hidden, s + 2, 0.2),
                wq: det_w(n_heads * head_dim, hidden, s + 3, 0.2),
                wk: det_w(n_kv * head_dim, hidden, s + 4, 0.2),
                wv: det_w(n_kv * head_dim, hidden, s + 5, 0.2),
                wo: det_w(hidden, n_heads * head_dim, s + 6, 0.2),
                attn_gate: det_w(n_heads * head_dim, hidden, s + 7, 0.2),
                attn_q_clip: vec![1.0; n_heads * head_dim],
                attn_sink_proj: det_w(n_heads, hidden, s + 8, 0.2),
                attn_rel: det_w(n_heads * d_rel, hidden, s + 9, 0.2),
                wqrsg: None,
                wk_dyn: None,
                wv_dyn: None,
                attn_rel_bias: det_vec(rel_extent * d_rel, s + 10, 0.15),
                attn_rel_clip: vec![1.0; n_heads * d_rel],
                attn_k_conv: det_vec(n_kv * head_dim * conv_k, s + 11, 0.1),
                attn_v_conv: det_vec(n_kv * head_dim * conv_k, s + 12, 0.1),
                attn_k_conv_proj: (!swa).then(|| det_w(n_kv * conv_k, hidden, s + 13, 0.1)),
                attn_v_conv_proj: (!swa).then(|| det_w(n_kv * conv_k, hidden, s + 14, 0.1)),
                attn_post_conv: det_vec(hidden * conv_k, s + 15, 0.1),
                ffn_norm: vec![1.0; hidden],
                ffn_norm_dev: None,
                ffn_gate_inp: det_w(n_exp, hidden, s + 16, 0.2),
                ffn_gate_inp_padded_dev: None,
                exp_probs_b: Some(det_vec(n_exp, s + 17, 0.05)),
                experts: Inkling2Experts::DenseF32 {
                    gate_up: det_vec(n_exp * 2 * ff * hidden, s + 18, 0.15),
                    down: det_vec(n_exp * hidden * ff, s + 19, 0.15),
                    n_experts: n_exp,
                    hidden,
                    ff,
                },
                ffn_post_conv: det_vec(hidden * conv_k, s + 20, 0.1),
            });
        }

        let mut loop_layers = Vec::new();
        for c in 0..2 {
            let s = 900 + c * 50;
            loop_layers.push(Inkling2LoopLayer {
                msg_query: det_vec(4 * hidden, s, 0.2),
                msg_query_dev: None,
                ctrl_down: [
                    det_w(gate_rank, 2 * hidden, s + 1, 0.15),
                    det_w(gate_rank, 2 * hidden, s + 2, 0.15),
                    det_w(gate_rank, 2 * hidden, s + 3, 0.15),
                    det_w(gate_rank, 2 * hidden, s + 4, 0.15),
                ],
                ctrl_gain: [
                    det_w(hidden, gate_rank, s + 5, 0.15),
                    det_w(hidden, gate_rank, s + 6, 0.15),
                    det_w(hidden, gate_rank, s + 7, 0.15),
                    det_w(hidden, gate_rank, s + 8, 0.15),
                ],
                ctrl_forget: [
                    det_w(hidden, gate_rank, s + 9, 0.15),
                    det_w(hidden, gate_rank, s + 10, 0.15),
                    det_w(hidden, gate_rank, s + 11, 0.15),
                    det_w(hidden, gate_rank, s + 12, 0.15),
                ],
                ctrl_down_qkv_f16: None,
                ctrl_down_ffn_f16: None,
                ctrl_gf_qkv_f16: None,
                ctrl_gf_ffn_f16: None,
            });
        }

        let tokens_list: Vec<String> = (0..vocab_unpadded).map(|i| format!("t{i}")).collect();
        let tokenizer = Tokenizer::from_gpt4o_bpe_parts(tokens_list, vec![], vec![], 19, Some(0)).unwrap();
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let config = cfg.to_model_config();
        let state = cfg.new_state();

        let model = Inkling2Model {
            config,
            cfg,
            tokenizer,
            backend,
            embed_table: EmbedTable::F32(det_vec(vocab_padded * hidden, 1, 0.3), hidden),
            embed_norm: vec![1.2; hidden],
            output_norm: vec![1.0; hidden],
            output_norm_dev: None,
            output_res_query: det_vec(hidden, 2, 0.2),
            output_res_query_dev: None,
            output: det_w(vocab_padded, hidden, 3, 0.2),
            layers,
            loop_layers,
            full_gpu_decode: false,
            state: Mutex::new(state),
            gpu_scratch: Mutex::new(None),
        };

        let prompt = [1u32, 4, 7, 2, 9, 3];

        // 1. Single batched prefill
        model.reset();
        let logits_batch = model.forward_last(&prompt, &[]).unwrap();
        assert_eq!(logits_batch.len(), vocab_unpadded);
        assert!(logits_batch.iter().all(|v| v.is_finite() && v.abs() <= 30.0));

        // 2. Token-by-token step decode
        model.reset();
        let mut logits_step = Vec::new();
        for &tok in &prompt {
            logits_step = model.forward_last(&[tok], &[]).unwrap();
        }
        assert_eq!(model.state_pos(), prompt.len());

        let max_diff = logits_batch
            .iter()
            .zip(&logits_step)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "batch vs step logits diverged: max_diff={max_diff}");

        // 3. Soft vision embedding splice works and changes logits cleanly.
        model.reset();
        let soft_emb = det_vec(2 * hidden, 777, 0.5);
        let logits_soft = model.forward_last(&prompt, &[(2, &soft_emb)]).unwrap();
        assert_eq!(logits_soft.len(), vocab_unpadded);
        assert!(logits_soft.iter().all(|v| v.is_finite()));
        let diff_soft: f32 = logits_batch
            .iter()
            .zip(&logits_soft)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_soft > 1e-3, "spliced vision embeddings had no effect");
    }
}
