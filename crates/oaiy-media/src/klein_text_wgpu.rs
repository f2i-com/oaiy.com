//! FLUX.2 Klein's text conditioning (Qwen3-4B) on WebGPU, as [`crate::klein::text::TextEncoder`] encodes a prompt: its
//! chat template's tokens right-padded to 512, causal, a padded position attending to the prompt's own tokens only;
//! the states after layers 9, 18 and 27 side by side (7,680 wide, the padded positions' too), no final norm.
//! The weights f16 on the GPU (the BF16 checkpoint's rounded) a layer at a time, every prompt through it (its 5.4 GB
//! never all on the card), the embedding's rows looked up on the host; the matmuls read their inputs as f32 (an LLM's
//! hidden states reach some hundreds).
use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::path::Path;

const LAYERS: usize = 27;
/// The layers whose output states are the conditioning's (counted from one, as Hugging Face's tuple of hidden states).
const TAPS: [usize; 3] = [9, 18, 27];
const D: usize = 2560;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const SEQ: usize = 512;
const EPS: f32 = 1e-6;
const THETA: f64 = 1_000_000.;

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

pub struct WgpuKleinText {
    gpu: ggml_rs_wgpu::WgpuBackend,
    /// The token embedding (`[vocab, D]`), on the host: a prompt's rows looked up there.
    embedding: Vec<f32>,
    /// The weights, a layer's read as the prompts reach it.
    w: Weights,
    /// The MLP's width.
    ff: usize,
    tokenizer: tokenizers::Tokenizer,
}

/// A prompt's vectors through the layers.
struct Prompt {
    /// Its own tokens (the rest of the 512 padding).
    valid: usize,
    x: DeviceVec,
    norm: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    /// The padded positions' queries, and their attention's output.
    qpad: DeviceVec,
    apad: DeviceVec,
    o: DeviceVec,
    g: DeviceVec,
    u: DeviceVec,
    act: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    /// Every position's attention output, the prompt's then the padding's.
    attended: DeviceVec,
    table: DeviceVec,
}

impl WgpuKleinText {
    /// The Qwen3-4B text encoder at `checkpoint` with `tokenizer` (its `tokenizer.json`) on GPU `device` (as the
    /// computer counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(checkpoint: &Path, tokenizer: &Path, device: usize) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut w = Weights::open(checkpoint)?;
        let shape = |w: &Weights, name: &str| w.shape(&format!("model.layers.0.{name}.weight"));
        let (q, k, gate) = (shape(&w, "self_attn.q_proj")?, shape(&w, "self_attn.k_proj")?, shape(&w, "mlp.gate_proj")?);
        if w.shape("model.embed_tokens.weight")? != [151936, D] || q != [HEADS * HD, D] || k != [KV_HEADS * HD, D] || gate.len() != 2 || gate[1] != D {
            candle_core::bail!("Klein 4B requires the Qwen3-4B text encoder, not Qwen3-VL or Qwen3.5");
        }
        let embedding = w.tensor("model.embed_tokens.weight", &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(err)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(err)?;
        for (token, id) in [("<|endoftext|>", 151643), ("<|im_start|>", 151644), ("<|im_end|>", 151645)] {
            if tokenizer.token_to_id(token) != Some(id) {
                candle_core::bail!("incompatible Qwen3 tokenizer: {token} must have id {id}");
            }
        }
        Ok(Self { gpu, embedding, w, ff: gate[0], tokenizer })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// Layer `i`'s weights on the GPU.
    fn layer(&mut self, i: usize) -> Result<Layer> {
        let p = format!("model.layers.{i}");
        let gpu = &self.gpu;
        let mut none = crate::lora::Loras::open(&[])?;
        let w = &mut self.w;
        let mut mat = |w: &mut Weights, name: &str| -> Result<Mat> {
            let (v, n, k) = crate::wgpu_weights::f16_matrix(w, gpu, &format!("{p}.{name}"), &mut none)?;
            Ok(Mat { v, n, k })
        };
        let vector = |w: &mut Weights, name: &str| -> Result<DeviceVec> {
            let values = w.tensor(&format!("{p}.{name}.weight"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let v = gpu.vec(values.len());
            gpu.upload(&v, &values);
            Ok(v)
        };
        Ok(Layer {
            input: vector(w, "input_layernorm")?,
            post: vector(w, "post_attention_layernorm")?,
            qn: vector(w, "self_attn.q_norm")?,
            kn: vector(w, "self_attn.k_norm")?,
            q: mat(w, "self_attn.q_proj")?,
            k: mat(w, "self_attn.k_proj")?,
            v: mat(w, "self_attn.v_proj")?,
            o: mat(w, "self_attn.o_proj")?,
            gate: mat(w, "mlp.gate_proj")?,
            up: mat(w, "mlp.up_proj")?,
            down: mat(w, "mlp.down_proj")?,
        })
    }

    /// `prompt`'s conditioning (`[1, 512, 7680]` on the CPU), as [`crate::klein::text::TextEncoder::encode`] gives it.
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        self.encode_all(&[prompt])?.pop().ok_or_else(|| err("no conditioning"))
    }

    /// Each prompt's conditioning, every prompt through a layer while its weights are on the card.
    pub fn encode_all(&mut self, prompts: &[&str]) -> Result<Vec<Tensor>> {
        let (ff, vocab) = (self.ff, self.embedding.len() / D);
        let (qd, kd) = (HEADS * HD, KV_HEADS * HD);
        let row = 2 * kd;
        let pad = self.tokenizer.token_to_id("<|endoftext|>").ok_or_else(|| err("Qwen3 tokenizer lacks its padding token"))?;
        // RoPE over each head's halves (NeoX), a position a token
        let mut table = Vec::with_capacity(SEQ * HD);
        for p in 0..SEQ {
            for j in 0..HD / 2 {
                let a = p as f64 / THETA.powf(j as f64 / (HD / 2) as f64);
                table.push(a.sin() as f32);
                table.push(a.cos() as f32);
            }
        }
        let mut all = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let chat = crate::klein::text::chat_template(prompt);
            let mut ids = self.tokenizer.encode(chat, false).map_err(err)?.get_ids().to_vec();
            // (the published conditioning: right-padded or cut to 512)
            ids.truncate(SEQ);
            let valid = ids.len();
            if valid == 0 {
                candle_core::bail!("empty Klein token sequence");
            }
            ids.resize(SEQ, pad);
            let mut x0 = Vec::with_capacity(SEQ * D);
            for &id in &ids {
                let id = id as usize;
                if id >= vocab {
                    candle_core::bail!("token {id} past the embedding's {vocab}");
                }
                x0.extend_from_slice(&self.embedding[id * D..(id + 1) * D]);
            }
            let padded = SEQ - valid;
            let p = Prompt {
                valid,
                x: self.vec(SEQ * D),
                norm: self.vec(SEQ * D),
                q: self.vec(SEQ * qd),
                k: self.vec(SEQ * kd),
                v: self.vec(SEQ * kd),
                qq: self.vec(SEQ * qd),
                kk: self.vec(SEQ * kd),
                qpad: self.vec(padded * qd),
                apad: self.vec(if padded == 0 { 1 } else { self.gpu.attention_rows_full_out_len(padded, HEADS, HD, valid) }),
                o: self.vec(SEQ * D),
                g: self.vec(SEQ * ff),
                u: self.vec(SEQ * ff),
                act: self.vec(SEQ * ff),
                kv: self.vec(SEQ * row),
                att: self.vec(self.gpu.attention_rows_out_len(valid, HEADS, HD, valid)),
                attended: self.vec(SEQ * qd),
                table: self.vec(SEQ * HD),
            };
            self.gpu.upload(&p.x, &x0);
            self.gpu.upload(&p.table, &table);
            all.push(p);
        }
        let scale = 1.0 / (HD as f32).sqrt();
        // each prompt's states after the tapped layers, in turn
        let mut taps: Vec<Vec<Vec<f32>>> = all.iter().map(|_| Vec::with_capacity(TAPS.len())).collect();
        for i in 0..LAYERS {
            let l = self.layer(i)?;
            let tapped = TAPS.contains(&(i + 1));
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            for p in &all {
                let (valid, padded) = (p.valid, SEQ - p.valid);
                let mul = |r: &mut dyn ChainRecorder, m: &Mat, a: &DeviceVec, y: &DeviceVec| r.matmul_f16_rows_f32(&m.v, m.n, m.k, a, y, SEQ);
                r.rmsnorm_rows(&p.x, &l.input, &p.norm, SEQ, EPS);
                mul(r, &l.q, &p.norm, &p.q);
                mul(r, &l.k, &p.norm, &p.k);
                mul(r, &l.v, &p.norm, &p.v);
                r.rmsnorm_rows(&p.q, &l.qn, &p.qq, SEQ * HEADS, EPS);
                r.rmsnorm_rows(&p.k, &l.kn, &p.kk, SEQ * KV_HEADS, EPS);
                r.rope_rows(&p.qq, SEQ, HEADS, HD, &p.table, true);
                r.rope_rows(&p.kk, SEQ, KV_HEADS, HD, &p.table, true);
                r.store_rows(&p.kk, &p.kv, SEQ, kd, 0, row, 0);
                r.store_rows(&p.v, &p.kv, SEQ, kd, 0, row, kd);
                // the prompt's own tokens causal among themselves; a padded position over all of them and no padding
                r.attention_rows(&p.qq, &p.kv, &p.att, valid, HEADS, KV_HEADS, HD, 0, None, scale);
                r.copy(&p.att, 0, &p.attended, 0, valid * qd);
                if padded > 0 {
                    r.copy(&p.qq, valid * qd, &p.qpad, 0, padded * qd);
                    r.attention_rows_full(&p.qpad, &p.kv, &p.apad, padded, HEADS, KV_HEADS, HD, valid, scale);
                    r.copy(&p.apad, 0, &p.attended, valid * qd, padded * qd);
                }
                mul(r, &l.o, &p.attended, &p.o);
                r.add(&p.x, &p.o);
                r.rmsnorm_rows(&p.x, &l.post, &p.norm, SEQ, EPS);
                mul(r, &l.gate, &p.norm, &p.g);
                mul(r, &l.up, &p.norm, &p.u);
                r.silu_mul(&p.g, &p.u, &p.act, SEQ * ff);
                mul(r, &l.down, &p.act, &p.o);
                r.add(&p.x, &p.o);
                if tapped {
                    r.read(&p.x);
                }
            }
            let reads = rec.finish();
            if tapped {
                for (tap, states) in taps.iter_mut().zip(reads) {
                    tap.push(states);
                }
            }
            drop(l);
        }
        // a token's three states side by side
        taps.into_iter()
            .map(|tap| {
                if tap.len() != TAPS.len() || tap.iter().any(|t| t.len() != SEQ * D) {
                    return Err(err("the conditioning's tapped states were not all read"));
                }
                let mut joined = Vec::with_capacity(SEQ * D * TAPS.len());
                for t in 0..SEQ {
                    for states in &tap {
                        joined.extend_from_slice(&states[t * D..(t + 1) * D]);
                    }
                }
                Tensor::from_vec(joined, (1, SEQ, D * TAPS.len()), &Device::Cpu)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WebGPU text encoder gives the Candle one's conditioning (CPU, f32) for a prompt, from the published weights
    /// (`OAIY_KLEIN_TEXT_ENCODER`, Qwen3-4B's checkpoint, and `OAIY_KLEIN_TOKENIZER`, its `tokenizer.json`): every
    /// token's states, the padded positions' too.
    #[test]
    #[ignore = "needs Qwen3-4B (OAIY_KLEIN_TEXT_ENCODER), its tokenizer (OAIY_KLEIN_TOKENIZER) and a WebGPU adapter"]
    fn the_webgpu_text_encoder_is_the_candle_one() -> Result<()> {
        let (Some(checkpoint), Some(tokenizer)) = (std::env::var_os("OAIY_KLEIN_TEXT_ENCODER").map(std::path::PathBuf::from), std::env::var_os("OAIY_KLEIN_TOKENIZER").map(std::path::PathBuf::from)) else { return Ok(()) };
        let prompt = "A cosy mountain cabin at dusk beside a frozen lake, a sign above the door that reads \"WebGPU Lodge\"";
        let t = std::time::Instant::now();
        let mut gpu = WgpuKleinText::load(&checkpoint, &tokenizer, 0)?;
        eprintln!("WebGPU text encoder loaded in {:.1} s", t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let got = gpu.encode(prompt)?;
        eprintln!("WebGPU encode {:.3} s", t.elapsed().as_secs_f64());
        drop(gpu);
        let t = std::time::Instant::now();
        // (every layer resident: the default streams each from the file as it is used)
        let budget = crate::residency::Budget { memory: crate::residency::Memory::Gpu, ..Default::default() };
        let mut cpu = crate::klein::text::TextEncoder::load(&checkpoint, &tokenizer, &Device::Cpu, DType::F32, &budget)?;
        let want = cpu.encode(prompt)?;
        eprintln!("Candle load and encode {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!(got.dims(), want.dims());
        let (g, e) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
        let width = D * TAPS.len();
        let mut worst = 1f64;
        for t in 0..SEQ {
            let (a, b) = (&g[t * width..(t + 1) * width], &e[t * width..(t + 1) * width]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            worst = worst.min(dot / (na * nb));
        }
        eprintln!("{SEQ} tokens: the worst token's cosine {worst:.6}");
        assert!(worst > 0.999, "the worst token's cosine {worst}");
        Ok(())
    }
}
