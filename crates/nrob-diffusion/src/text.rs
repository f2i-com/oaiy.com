//! Qwen3-VL-8B text/vision conditioning. The image transformer consumes the
//! last decoder layer BEFORE the final language-model RMS normalization.
use crate::vision::Features;
use crate::{
    math::*,
    weights::{Linear, Weights},
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

pub struct Conditioning {
    pub states: Tensor,
    /// Ordered (start, VLM slot count) ranges, after removing the system prefix.
    pub spans: Vec<(usize, usize)>,
}

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

    pub fn encode(&self, prompt: &str, images: &[Features]) -> Result<Conditioning> {
        let sys = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
        let image_prefix = (1..=images.len())
            .map(|i| format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = format!(
            "{sys}<|im_start|>user\n{image_prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encode = |s: &str| {
            self.tokenizer
                .encode(s, false)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };
        let tokens = encode(&text)?.get_ids().to_vec();
        if tokens.len() > 1024 {
            candle_core::bail!("image prompt exceeds 1024 text tokens");
        }
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut positions = Vec::new();
        let mut position = 0usize;
        for id in tokens {
            if id == 151655 {
                let image = images.get(spans.len()).ok_or_else(|| {
                    candle_core::Error::Msg("unexpected image placeholder in prompt".into())
                })?;
                spans.push((ids.len(), image.h * image.w));
                for y in 0..image.h {
                    for x in 0..image.w {
                        ids.push(id);
                        positions.push([position, position + y, position + x]);
                    }
                }
                position += image.h.max(image.w);
            } else {
                ids.push(id);
                positions.push([position; 3]);
                position += 1;
            }
        }
        if spans.len() != images.len() {
            candle_core::bail!("reference image placeholder mismatch");
        }
        let drop = encode(sys)?.len();
        let s = ids.len();
        let mut x = self
            .embedding
            .index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?
            .unsqueeze(0)?;
        x = inject(
            &x,
            &spans,
            &images
                .iter()
                .map(|v| v.embedding.clone())
                .collect::<Vec<_>>(),
            false,
        )?;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for pos in positions {
            for j in 0..64 {
                let axis = if j < 60 && j % 3 != 0 { j % 3 } else { 0 };
                let phase = pos[axis] as f64 / 5_000_000f64.powf(j as f64 / 64.);
                cos.push(phase.cos() as f32);
                sin.push(phase.sin() as f32);
            }
        }
        let cos = Tensor::from_vec(cos, (s, 64), &self.device)?.to_dtype(self.dtype)?;
        let sin = Tensor::from_vec(sin, (s, 64), &self.device)?.to_dtype(self.dtype)?;
        for (layer, block) in self.blocks.iter().enumerate() {
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
            if layer < 3 {
                x = inject(
                    &x,
                    &spans,
                    &images
                        .iter()
                        .map(|v| v.deep[layer].clone())
                        .collect::<Vec<_>>(),
                    true,
                )?;
            }
        }
        Ok(Conditioning {
            states: x.narrow(1, drop, s - drop)?.contiguous()?,
            spans: spans.into_iter().map(|(s, n)| (s - drop, n)).collect(),
        })
    }
}

fn inject(x: &Tensor, spans: &[(usize, usize)], features: &[Tensor], add: bool) -> Result<Tensor> {
    if spans.is_empty() {
        return Ok(x.clone());
    }
    let mut parts = Vec::new();
    let mut cursor = 0;
    for ((start, len), v) in spans.iter().zip(features) {
        if *start > cursor {
            parts.push(x.narrow(1, cursor, start - cursor)?);
        }
        parts.push(if add {
            (x.narrow(1, *start, *len)? + v)?
        } else {
            v.clone()
        });
        cursor = start + len;
    }
    if cursor < x.dim(1)? {
        parts.push(x.narrow(1, cursor, x.dim(1)? - cursor)?);
    }
    Tensor::cat(&parts, 1)
}
