//! Text-only Qwen3-VL-8B conditioning. The image transformer consumes the
//! last decoder layer BEFORE the final language-model RMS normalization.
use crate::{
    math::*,
    weights::{Linear, Weights},
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

struct Block {
    input: Tensor,
    post: Tensor,
    qn: Tensor,
    kn: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub struct TextEncoder {
    embedding: Tensor,
    blocks: Vec<Block>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dtype: DType,
}

impl TextEncoder {
    pub fn load(root: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let config =
            nrob::json::Json::parse(&std::fs::read(root.join("text_encoder/config.json"))?)
                .map_err(candle_core::Error::wrap)?;
        if config.get("model_type").and_then(|x| x.as_str()) != Some("qwen3_vl") {
            candle_core::bail!("expected Qwen3-VL text encoder");
        }
        let mut w = Weights::open(&root.join("text_encoder"))?;
        let p = "model.language_model";
        let embedding = w.tensor(&format!("{p}.embed_tokens.weight"), device, dtype)?;
        let mut blocks = Vec::new();
        for i in 0..36 {
            let p = format!("{p}.layers.{i}");
            let mut linear =
                |name: &str| w.linear(&format!("{p}.{name}"), device, dtype, &mut None);
            let (q, k, v, o) = (
                linear("self_attn.q_proj")?,
                linear("self_attn.k_proj")?,
                linear("self_attn.v_proj")?,
                linear("self_attn.o_proj")?,
            );
            let (gate, up, down) = (
                linear("mlp.gate_proj")?,
                linear("mlp.up_proj")?,
                linear("mlp.down_proj")?,
            );
            blocks.push(Block {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                input: w.tensor(&format!("{p}.input_layernorm.weight"), device, dtype)?,
                post: w.tensor(
                    &format!("{p}.post_attention_layernorm.weight"),
                    device,
                    dtype,
                )?,
                qn: w.tensor(&format!("{p}.self_attn.q_norm.weight"), device, dtype)?,
                kn: w.tensor(&format!("{p}.self_attn.k_norm.weight"), device, dtype)?,
            });
        }
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("processor/tokenizer.json"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(Self {
            embedding,
            blocks,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        let sys = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
        let text = format!("{sys}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let encode = |s: &str| {
            self.tokenizer
                .encode(s, false)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };
        let ids = encode(&text)?.get_ids().to_vec();
        let drop = encode(sys)?.len();
        let s = ids.len();
        if s > 1024 {
            candle_core::bail!("image prompt exceeds 1024 text tokens");
        }
        let mut x = self
            .embedding
            .index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?
            .unsqueeze(0)?;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for pos in 0..s {
            for j in 0..64 {
                let phase = pos as f64 / 5_000_000f64.powf(j as f64 / 64.);
                cos.push(phase.cos() as f32);
                sin.push(phase.sin() as f32);
            }
        }
        let cos = Tensor::from_vec(cos, (s, 64), &self.device)?.to_dtype(self.dtype)?;
        let sin = Tensor::from_vec(sin, (s, 64), &self.device)?.to_dtype(self.dtype)?;
        for block in &self.blocks {
            let norm = rms(&x, &block.input, 1e-6)?;
            let q = rms(&heads(&block.q.forward(&norm)?, 32)?, &block.qn, 1e-6)?;
            let k = rms(&heads(&block.k.forward(&norm)?, 8)?, &block.kn, 1e-6)?;
            let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            let repeat = |v: Tensor| {
                v.unsqueeze(2)?
                    .broadcast_as((1, 8, 4, s, 128))?
                    .contiguous()?
                    .reshape((1, 32, s, 128))
            };
            let k = repeat(k)?;
            let v = repeat(heads(&block.v.forward(&norm)?, 8)?)?;
            x = (x + block.o.forward(&unheads(&attention(&q, &k, &v, s)?)?)?)?;
            let norm = rms(&x, &block.post, 1e-6)?;
            x = (x + swiglu(&norm, &block.gate, &block.up, &block.down)?)?;
        }
        x.narrow(1, drop, s - drop)?.contiguous()
    }
}
