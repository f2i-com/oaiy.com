//! Qwen3-4B causal conditioning. Hidden states 9/18/27 (HF tuple indices)
//! are concatenated, before final RMS normalization, including padded tokens.
use crate::{
    math::{heads, swiglu, unheads},
    residency::{Budget, Resident, Tiered},
    weights::{bytes_of, Linear, Weights},
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
impl Resident for Block {
    fn bytes(&self) -> u64 {
        [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
        ]
        .iter()
        .map(|l| l.bytes())
        .sum::<u64>()
            + [&self.input, &self.post, &self.qn, &self.kn]
                .iter()
                .map(|t| bytes_of(t))
                .sum::<u64>()
    }
    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            input: self.input.to_device(d)?,
            post: self.post.to_device(d)?,
            qn: self.qn.to_device(d)?,
            kn: self.kn.to_device(d)?,
            q: self.q.to_device(d)?,
            k: self.k.to_device(d)?,
            v: self.v.to_device(d)?,
            o: self.o.to_device(d)?,
            gate: self.gate.to_device(d)?,
            up: self.up.to_device(d)?,
            down: self.down.to_device(d)?,
        })
    }
}
impl Block {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        valid: usize,
        q_heads: usize,
        kv_heads: usize,
    ) -> Result<Tensor> {
        let norm = rms(x, &self.input)?;
        let q = rotate_half(
            &rms(&heads(&self.q.forward(&norm)?, q_heads)?, &self.qn)?,
            cos,
            sin,
        )?;
        let k = rotate_half(
            &rms(&heads(&self.k.forward(&norm)?, kv_heads)?, &self.kn)?,
            cos,
            sin,
        )?;
        let v = heads(&self.v.forward(&norm)?, kv_heads)?;
        let seq = x.dim(1)?;
        let dim = q.dim(3)?;
        let repeat = |t: Tensor| {
            t.unsqueeze(2)?
                .broadcast_as((1, kv_heads, q_heads / kv_heads, seq, dim))?
                .contiguous()?
                .reshape((1, q_heads, seq, dim))
        };
        let a = masked_attention(&q, &repeat(k)?, &repeat(v)?, valid)?;
        let next = (x + self.o.forward(&unheads(&a)?)?)?;
        &next + swiglu(&rms(&next, &self.post)?, &self.gate, &self.up, &self.down)?
    }
}
fn rms(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    f.broadcast_div(&(f.sqr()?.mean_keepdim(candle_core::D::Minus1)? + 1e-6)?.sqrt()?)?
        .to_dtype(x.dtype())?
        .broadcast_mul(weight)
}
fn load_block(w: &mut Weights, i: usize, d: &Device, t: DType) -> Result<Block> {
    let p = format!("model.layers.{i}");
    let mut l = crate::lora::Loras::default();
    Ok(Block {
        input: w.tensor(&format!("{p}.input_layernorm.weight"), d, t)?,
        post: w.tensor(&format!("{p}.post_attention_layernorm.weight"), d, t)?,
        qn: w.tensor(&format!("{p}.self_attn.q_norm.weight"), d, t)?,
        kn: w.tensor(&format!("{p}.self_attn.k_norm.weight"), d, t)?,
        q: w.linear(&format!("{p}.self_attn.q_proj"), d, t, &mut l)?,
        k: w.linear(&format!("{p}.self_attn.k_proj"), d, t, &mut l)?,
        v: w.linear(&format!("{p}.self_attn.v_proj"), d, t, &mut l)?,
        o: w.linear(&format!("{p}.self_attn.o_proj"), d, t, &mut l)?,
        gate: w.linear(&format!("{p}.mlp.gate_proj"), d, t, &mut l)?,
        up: w.linear(&format!("{p}.mlp.up_proj"), d, t, &mut l)?,
        down: w.linear(&format!("{p}.mlp.down_proj"), d, t, &mut l)?,
    })
}
pub struct TextEncoder {
    embedding: Tensor,
    blocks: Tiered<Block>,
    weights: Weights,
    tokenizer: tokenizers::Tokenizer,
    dev: Device,
    dtype: DType,
}
impl TextEncoder {
    pub fn load(
        checkpoint: &Path,
        tokenizer: &Path,
        dev: &Device,
        dtype: DType,
        budget: &Budget,
    ) -> Result<Self> {
        let mut w = Weights::open(checkpoint)?;
        if w.shape("model.embed_tokens.weight")? != [151936, 2560]
            || w.shape("model.layers.0.self_attn.q_proj.weight")? != [4096, 2560]
        {
            candle_core::bail!(
                "Klein 4B requires the Qwen3-4B text encoder, not Qwen3-VL or Qwen3.5"
            );
        }
        let embedding = w.tensor("model.embed_tokens.weight", dev, dtype)?;
        // Nothing after layer 27 participates in conditioning; omit the unused
        // nine layers and LM head, preserving HF hidden-state tuple indexing.
        let blocks = Tiered::load(
            27,
            budget,
            dev,
            |i| load_block(&mut w, i, dev, dtype),
            |_| {},
        )?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        for (token, id) in [
            ("<|endoftext|>", 151643),
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
        ] {
            if tokenizer.token_to_id(token) != Some(id) {
                candle_core::bail!("incompatible Qwen3 tokenizer: {token} must have id {id}");
            }
        }
        Ok(Self {
            embedding,
            blocks,
            weights: w,
            tokenizer,
            dev: dev.clone(),
            dtype,
        })
    }
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        let chat = chat_template(prompt);
        let encoded = self
            .tokenizer
            .encode(chat, false)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut ids = encoded.get_ids().to_vec();
        // Official Klein conditioning is right-padded/truncated to 512.
        ids.truncate(512);
        let valid = ids.len();
        if valid == 0 {
            candle_core::bail!("empty Klein token sequence");
        }
        let pad = self.tokenizer.token_to_id("<|endoftext|>").ok_or_else(|| {
            candle_core::Error::Msg("Qwen3 tokenizer lacks its padding token".into())
        })?;
        ids.resize(512, pad);
        let mut x = self
            .embedding
            .index_select(&Tensor::from_vec(ids, (512,), &self.dev)?, 0)?
            .unsqueeze(0)?;
        let (cos, sin) = rotary(512, 128, &self.dev)?;
        let mut taps = Vec::with_capacity(3);
        let dev = &self.dev;
        let dtype = self.dtype;
        let weights = &mut self.weights;
        for i in 0..27 {
            x = self.blocks.with(
                i,
                |i| load_block(weights, i, dev, dtype),
                |b| b.forward(&x, &cos, &sin, valid, 32, 8),
            )?;
            if [9, 18, 27].contains(&(i + 1)) {
                taps.push(x.clone());
            }
        }
        Tensor::cat(&taps, 2)
    }
}
pub fn chat_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}
fn rotary(seq: usize, dim: usize, d: &Device) -> Result<(Tensor, Tensor)> {
    let values = (0..seq)
        .flat_map(|p| {
            (0..dim).map(move |i| {
                p as f32 * 1_000_000f32.powf(-((i % (dim / 2)) as f32) / (dim / 2) as f32)
            })
        })
        .collect::<Vec<_>>();
    let t = Tensor::from_vec(values, (1, 1, seq, dim), d)?;
    Ok((t.cos()?, t.sin()?))
}
fn rotate_half(x: &Tensor, c: &Tensor, s: &Tensor) -> Result<Tensor> {
    let a = x.clone();
    let n = a.dim(3)? / 2;
    let rot = Tensor::cat(&[a.narrow(3, n, n)?.neg()?, a.narrow(3, 0, n)?], 3)?;
    a.broadcast_mul(&c.to_dtype(x.dtype())?)? + rot.broadcast_mul(&s.to_dtype(x.dtype())?)?
}
fn masked_attention(q: &Tensor, k: &Tensor, v: &Tensor, valid: usize) -> Result<Tensor> {
    let seq = q.dim(2)?;
    let mut parts = Vec::new();
    let kt = k.transpose(2, 3)?.contiguous()?;
    for start in (0..seq).step_by(128) {
        let n = 128.min(seq - start);
        let mask = (start..start + n)
            .flat_map(|i| {
                (0..seq).map(move |j| {
                    if j > i || j >= valid {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect::<Vec<_>>();
        let scores = (q.narrow(2, start, n)?.contiguous()?.matmul(&kt)?
            / (q.dim(3)? as f64).sqrt())?
        .to_dtype(DType::F32)?
        .broadcast_add(&Tensor::from_vec(mask, (1, 1, n, seq), q.device())?)?;
        parts.push(
            candle_nn::ops::softmax_last_dim(&scores)?
                .to_dtype(q.dtype())?
                .matmul(&v.contiguous()?)?,
        );
    }
    Tensor::cat(&parts, 2)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen3_decoder_layer_matches_hugging_face_with_gqa_and_right_padding() -> Result<()> {
        use oaiy_engine::json::Json;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/klein");
        let j = Json::parse(&std::fs::read(root.join("qwen-expected.json"))?)
            .map_err(candle_core::Error::wrap)?;
        let values = |key: &str| -> Result<Vec<f32>> {
            j.get(key)
                .and_then(Json::as_array)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {key}")))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|n| n as f32)
                        .ok_or_else(|| candle_core::Error::Msg("invalid reference number".into()))
                })
                .collect()
        };
        let mut w = Weights::open(&root.join("qwen-tiny.safetensors"))?;
        let block = load_block(&mut w, 0, &Device::Cpu, DType::F32)?;
        let x = Tensor::from_vec(values("input")?, (1, 4, 8), &Device::Cpu)?;
        let (c, s) = rotary(4, 4, &Device::Cpu)?;
        let actual = block
            .forward(&x, &c, &s, 2, 2, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = values("output")?;
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error < 2e-5, "Qwen3 reference error {error}");
        Ok(())
    }
    #[test]
    fn padded_queries_cannot_attend_to_padding_keys() -> Result<()> {
        let q = Tensor::zeros((1, 1, 4, 1), DType::F32, &Device::Cpu)?;
        let v = Tensor::from_vec(vec![2f32, 4., 100., 1000.], (1, 1, 4, 1), &Device::Cpu)?;
        assert_eq!(
            masked_attention(&q, &q, &v, 2)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![2., 3., 3., 3.]
        );
        Ok(())
    }
    #[test]
    fn thinking_disabled_template_matches_qwen3() {
        assert_eq!(
            chat_template("fox"),
            "<|im_start|>user\nfox<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
