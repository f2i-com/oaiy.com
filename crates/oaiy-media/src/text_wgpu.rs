//! Qwen Image 2.1's text encoder (Qwen3-VL 8B's language model) on WebGPU, as [`crate::text::TextEncoder`] encodes a
//! prompt without reference images: the last decoder layer's states before the final norm, the system prefix dropped.
//! The weights f16 on the GPU (the BF16 checkpoint's rounded), the embedding's rows looked up on the host; the matmuls
//! read their inputs as f32 (an LLM's hidden states reach some 500).
use crate::{text::Conditioning, weights::Weights};
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::path::Path;

const LAYERS: usize = 36;
const D: usize = 4096;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const FF: usize = 12288;
const EPS: f32 = 1e-6;
const THETA: f64 = 5_000_000.;
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A matrix on the GPU as f16 (`[n, k]`).
struct Mat {
    v: DeviceVec,
    n: usize,
    k: usize,
}

struct Layer {
    input: DeviceVec,
    post: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
}

pub struct WgpuTextEncoder {
    gpu: ggml_rs_wgpu::WgpuBackend,
    /// The token embedding (`[vocab, D]`), on the host: a prompt's rows looked up there.
    embedding: Vec<f32>,
    layers: Vec<Layer>,
    tokenizer: tokenizers::Tokenizer,
}

impl WgpuTextEncoder {
    /// The text encoder under `root` (`text_encoder/`, or the `checkpoint` given; `processor/tokenizer.json`) on GPU
    /// `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(root: &Path, checkpoint: Option<&Path>, device: usize) -> Result<Self> {
        let config = oaiy_engine::json::Json::parse(&std::fs::read(root.join("text_encoder/config.json"))?).map_err(candle_core::Error::wrap)?;
        if config.get("model_type").and_then(|x| x.as_str()) != Some("qwen3_vl") {
            candle_core::bail!("expected Qwen3-VL text encoder");
        }
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut w = Weights::open(checkpoint.unwrap_or(&root.join("text_encoder")))?;
        let prefix = if w.has("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" };
        let embedding = w.tensor(&format!("{prefix}.embed_tokens.weight"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut none = crate::lora::Loras::open(&[])?;
        let mut mat = |w: &mut Weights, name: &str| -> Result<Mat> {
            let (v, n, k) = crate::wgpu_weights::f16_matrix(w, &gpu, name, &mut none)?;
            Ok(Mat { v, n, k })
        };
        let vector = |w: &mut Weights, name: &str| -> Result<DeviceVec> {
            let values = w.tensor(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let v = gpu.vec(values.len());
            gpu.upload(&v, &values);
            Ok(v)
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("{prefix}.layers.{i}");
            layers.push(Layer {
                input: vector(&mut w, &format!("{p}.input_layernorm.weight"))?,
                post: vector(&mut w, &format!("{p}.post_attention_layernorm.weight"))?,
                qn: vector(&mut w, &format!("{p}.self_attn.q_norm.weight"))?,
                kn: vector(&mut w, &format!("{p}.self_attn.k_norm.weight"))?,
                q: mat(&mut w, &format!("{p}.self_attn.q_proj"))?,
                k: mat(&mut w, &format!("{p}.self_attn.k_proj"))?,
                v: mat(&mut w, &format!("{p}.self_attn.v_proj"))?,
                o: mat(&mut w, &format!("{p}.self_attn.o_proj"))?,
                gate: mat(&mut w, &format!("{p}.mlp.gate_proj"))?,
                up: mat(&mut w, &format!("{p}.mlp.up_proj"))?,
                down: mat(&mut w, &format!("{p}.mlp.down_proj"))?,
            });
        }
        let l = &layers[0];
        if (l.q.n, l.q.k, l.k.n, l.gate.n) != (HEADS * HD, D, KV_HEADS * HD, FF) {
            candle_core::bail!("not Qwen3-VL 8B's language model (q {}x{}, k {}, gate {})", l.q.n, l.q.k, l.k.n, l.gate.n);
        }
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("processor/tokenizer.json")).map_err(err)?;
        Ok(Self { gpu, embedding, layers, tokenizer })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// `prompt`'s conditioning (as [`crate::text::TextEncoder::encode`] gives it with no reference images): its states
    /// on the CPU.
    pub fn encode(&mut self, prompt: &str) -> Result<Conditioning> {
        let text = format!("{SYSTEM}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let encode = |s: &str| self.tokenizer.encode(s, false).map_err(err);
        let ids = encode(&text)?.get_ids().to_vec();
        if ids.len() > 1024 {
            candle_core::bail!("image prompt exceeds 1024 text tokens");
        }
        let drop = encode(SYSTEM)?.len();
        let s = ids.len();
        let vocab = self.embedding.len() / D;
        let mut x0 = Vec::with_capacity(s * D);
        for &id in &ids {
            let id = id as usize;
            if id >= vocab {
                candle_core::bail!("token {id} past the embedding's {vocab}");
            }
            x0.extend_from_slice(&self.embedding[id * D..(id + 1) * D]);
        }
        // RoPE over each head's halves (NeoX), a token's position its index (a text-only prompt's three axes alike)
        let mut table = Vec::with_capacity(s * HD);
        for p in 0..s {
            for j in 0..HD / 2 {
                let a = p as f64 / THETA.powf(j as f64 / (HD / 2) as f64);
                table.push(a.sin() as f32);
                table.push(a.cos() as f32);
            }
        }
        let (x, norm, q, k, v, qq, kk) = (self.vec(s * D), self.vec(s * D), self.vec(s * D), self.vec(s * KV_HEADS * HD), self.vec(s * KV_HEADS * HD), self.vec(s * D), self.vec(s * KV_HEADS * HD));
        let (o, g, u, act) = (self.vec(s * D), self.vec(s * FF), self.vec(s * FF), self.vec(s * FF));
        let kv = self.vec(s * 2 * KV_HEADS * HD);
        let att = self.vec(self.gpu.attention_rows_out_len(s.div_ceil(32) * 32, HEADS, HD, s));
        let tv = self.vec(s * HD);
        self.gpu.upload(&x, &x0);
        self.gpu.upload(&tv, &table);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let mul = |r: &mut dyn ChainRecorder, m: &Mat, a: &DeviceVec, y: &DeviceVec| r.matmul_f16_rows_f32(&m.v, m.n, m.k, a, y, s);
        let row = 2 * KV_HEADS * HD;
        for l in &self.layers {
            r.rmsnorm_rows(&x, &l.input, &norm, s, EPS);
            mul(r, &l.q, &norm, &q);
            mul(r, &l.k, &norm, &k);
            mul(r, &l.v, &norm, &v);
            r.rmsnorm_rows(&q, &l.qn, &qq, s * HEADS, EPS);
            r.rmsnorm_rows(&k, &l.kn, &kk, s * KV_HEADS, EPS);
            r.rope_rows(&qq, s, HEADS, HD, &tv, true);
            r.rope_rows(&kk, s, KV_HEADS, HD, &tv, true);
            r.store_rows(&kk, &kv, s, KV_HEADS * HD, 0, row, 0);
            r.store_rows(&v, &kv, s, KV_HEADS * HD, 0, row, KV_HEADS * HD);
            r.attention_rows(&qq, &kv, &att, s, HEADS, KV_HEADS, HD, 0, None, 1.0 / (HD as f32).sqrt());
            mul(r, &l.o, &att, &o);
            r.add(&x, &o);
            r.rmsnorm_rows(&x, &l.post, &norm, s, EPS);
            mul(r, &l.gate, &norm, &g);
            mul(r, &l.up, &norm, &u);
            r.silu_mul(&g, &u, &act, s * FF);
            mul(r, &l.down, &act, &o);
            r.add(&x, &o);
        }
        r.read_range(&x, drop * D, (s - drop) * D);
        let states = rec.finish().pop().ok_or_else(|| err("the states were not read"))?;
        Ok(Conditioning { states: Tensor::from_vec(states, (1, s - drop, D), &Device::Cpu)?, spans: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WebGPU text encoder gives the Candle one's states (CPU, f32) for a prompt, from the published weights
    /// (`OAIY_QWEN_IMAGE_BASE`, e.g. `D:\Qwen-Image-2.1`).
    #[test]
    #[ignore = "needs Qwen Image 2.1's text encoder (OAIY_QWEN_IMAGE_BASE) and a WebGPU adapter"]
    fn the_webgpu_text_encoder_is_the_candle_one() -> Result<()> {
        let Some(base) = std::env::var_os("OAIY_QWEN_IMAGE_BASE").map(std::path::PathBuf::from) else { return Ok(()) };
        let prompt = "A cosy mountain cabin at dusk beside a frozen lake, a sign above the door that reads \"WebGPU Lodge\"";
        let t = std::time::Instant::now();
        let mut gpu = WgpuTextEncoder::load(&base, None, 0)?;
        eprintln!("WebGPU text encoder loaded in {:.1} s", t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let got = gpu.encode(prompt)?;
        eprintln!("WebGPU encode {:.3} s", t.elapsed().as_secs_f64());
        drop(gpu);
        let budget = crate::residency::Budget::default();
        let mut cpu = crate::text::TextEncoder::load(&base, None, &Device::Cpu, DType::F32, &budget)?;
        let want = cpu.encode(prompt, &[])?;
        assert_eq!(got.states.dims(), want.states.dims());
        let (g, e) = (got.states.flatten_all()?.to_vec1::<f32>()?, want.states.flatten_all()?.to_vec1::<f32>()?);
        let n = e.len() / D;
        let mut worst = 1f64;
        for t in 0..n {
            let (a, b) = (&g[t * D..(t + 1) * D], &e[t * D..(t + 1) * D]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            worst = worst.min(dot / (na * nb));
        }
        eprintln!("{n} tokens: the worst token's cosine {worst:.6}");
        assert!(worst > 0.999, "the worst token's cosine {worst}");
        Ok(())
    }
}
