//! FLUX.2 Klein's text conditioning (Qwen3-4B) on WebGPU, as [`crate::klein::text::TextEncoder`] encodes a prompt: its
//! chat template's tokens right-padded to 512, causal, a padded position attending to the prompt's own tokens only;
//! the states after layers 9, 18 and 27 side by side (7,680 wide, the padded positions' too), no final norm.
//! FLUX 3 Image Klein's ([`FLUX3`]: Qwen3-VL-4B's language model, as BFL's `flux3/text_encoder.py` encodes a prompt)
//! the same but for its taps (every fourth layer's to the 32nd: 20,480 wide), its RoPE's base, its padding (to the
//! next multiple of 80 tokens, at most 4,096) and its padded positions' rotation (all at position 1).
//! The weights on the GPU a layer at a time, every prompt through it (its 5.4 GB never all on the card): a GGUF's
//! K-quant blocks as they are (the 4-bit file's 2.5 GB sent, where its rows were made 8 GB of halves on the host for
//! each picture, 2.5 s of a 3.3 s encode), anything else f16 (the BF16 checkpoint's rounded); the embedding's rows of a
//! prompt's tokens alone read from the file (not the whole table made f32, 1.5 GB, at each load). The f16 matmuls read
//! their inputs as f32 (an LLM's hidden states reach some hundreds).
use crate::common::weights::{Raw, Weights};
use crate::qwen_image::webgpu::transformer::coop_type;
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::{Backend, ChainRecorder, DeviceChain, DeviceVec, QuantizedTensor};
use std::path::Path;

const D: usize = 2560;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const EPS: f32 = 1e-6;

/// How a prompt's tokens are padded.
#[derive(Clone, Copy, Debug)]
pub enum Pad {
    /// To this many, cut to it.
    To(usize),
    /// To the next multiple of `of`, cut to `max`.
    Multiple { of: usize, max: usize },
}

impl Pad {
    /// A prompt of `tokens` tokens: how many of them are kept, and how many positions it takes padded.
    fn sizes(self, tokens: usize) -> (usize, usize) {
        match self {
            Pad::To(n) => (tokens.min(n), n),
            Pad::Multiple { of, max } => (tokens.min(max), (tokens.div_ceil(of) * of).min(max)),
        }
    }
}

/// A FLUX text encoder: which Qwen decoder, which of its layers' states, and how a prompt is made into its tokens.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// The decoder's tensors' prefix.
    pub prefix: &'static str,
    /// The layers whose output states are the conditioning's (counted from one, as Hugging Face's tuple of hidden
    /// states), side by side in this order.
    pub taps: &'static [usize],
    /// RoPE's base.
    pub theta: f64,
    pub pad: Pad,
    /// The position a padded token is rotated at: its own (None), or this one.
    pub pad_position: Option<usize>,
    /// The prompt in its chat template.
    pub template: fn(&str) -> String,
    /// What the checkpoint must be.
    pub what: &'static str,
}

/// FLUX.2 Klein 4B's: Qwen3-4B.
pub const FLUX2: Spec = Spec { prefix: "model.", taps: &[9, 18, 27], theta: 1_000_000., pad: Pad::To(512), pad_position: None, template: crate::klein::text::chat_template, what: "Klein 4B requires the Qwen3-4B text encoder, not Qwen3-VL or Qwen3.5" };

/// FLUX 3 Image Klein's: Qwen3-VL-4B-Instruct's language model (its vision tower unused), as `flux3/text_encoder.py`
/// runs it: positions counted over the prompt, the padding's all 1 (`masked_fill(mask == 0, 1)`).
pub const FLUX3: Spec = Spec { prefix: "model.language_model.", taps: &[4, 8, 12, 16, 20, 24, 28, 32], theta: 5_000_000., pad: Pad::Multiple { of: 80, max: 4096 }, pad_position: Some(1), template: flux3_template, what: "FLUX 3 Image Klein requires the Qwen3-VL-4B-Instruct text encoder" };

/// Qwen3-VL's chat template of one user turn and the assistant's to come (no system prompt).
pub fn flux3_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A matrix on the GPU as f16 (`[n, k]`).
/// A matrix on the GPU: a GGUF's K-quant blocks, or f16.
enum Mat {
    F16 { v: DeviceVec, n: usize, k: usize },
    Quant(QuantizedTensor),
}

impl Mat {
    /// `y` (`[rows, n]`) = `x` (`[rows, k]`, f32) times the matrix's transpose.
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match self {
            Self::F16 { v, n, k } => r.matmul_f16_rows_f32(v, *n, *k, x, y, rows),
            Self::Quant(q) => r.matmul_rows(q, x, y, rows),
        }
    }
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
    spec: Spec,
    /// The token embedding's rows (`[vocab, D]`): a prompt's read from the file as it is encoded.
    vocab: usize,
    /// The weights, a layer's read as the prompts reach it.
    w: Weights,
    /// The MLP's width.
    ff: usize,
    tokenizer: tokenizers::Tokenizer,
}

/// A prompt's vectors through the layers.
struct Prompt {
    /// Its own tokens, and all its positions (the rest padding).
    valid: usize,
    seq: usize,
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
        Self::load_spec(checkpoint, tokenizer, device, FLUX2)
    }

    /// [`Self::load`] of the text encoder `spec` says.
    pub fn load_spec(checkpoint: &Path, tokenizer: &Path, device: usize, spec: Spec) -> Result<Self> {
        let gpu = crate::common::webgpu_weights::text_encoder_gpu(device)?;
        let w = Weights::open(checkpoint)?;
        let p = spec.prefix;
        let shape = |w: &Weights, name: &str| w.shape(&format!("{p}layers.0.{name}.weight"));
        let last = spec.taps.iter().copied().max().unwrap_or(0);
        let (Ok(q), Ok(k), Ok(gate), Ok(embedding)) = (shape(&w, "self_attn.q_proj"), shape(&w, "self_attn.k_proj"), shape(&w, "mlp.gate_proj"), w.shape(&format!("{p}embed_tokens.weight"))) else {
            candle_core::bail!("{}", spec.what);
        };
        if embedding != [151936, D] || q != [HEADS * HD, D] || k != [KV_HEADS * HD, D] || gate.len() != 2 || gate[1] != D || !w.has(&format!("{p}layers.{}.mlp.down_proj.weight", last - 1)) {
            candle_core::bail!("{}", spec.what);
        }
        let vocab = embedding[0];
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(err)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(err)?;
        for (token, id) in [("<|endoftext|>", 151643), ("<|im_start|>", 151644), ("<|im_end|>", 151645)] {
            if tokenizer.token_to_id(token) != Some(id) {
                candle_core::bail!("incompatible Qwen3 tokenizer: {token} must have id {id}");
            }
        }
        Ok(Self { gpu, spec, vocab, w, ff: gate[0], tokenizer })
    }

    /// The GPU it is on.
    pub fn backend(&self) -> &ggml_rs_wgpu::WgpuBackend {
        &self.gpu
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// Layer `i`'s weights on the GPU.
    fn layer(&mut self, i: usize) -> Result<Layer> {
        let p = format!("{}layers.{i}", self.spec.prefix);
        let gpu = &self.gpu;
        let mut none = crate::common::lora::Loras::open(&[])?;
        let w = &mut self.w;
        let mut mat = |w: &mut Weights, name: &str| -> Result<Mat> {
            let key = format!("{p}.{name}.weight");
            if let Some(g) = w.ggml_dtype(&key).and_then(coop_type) {
                if let (&[n, k], Some(Raw::Ggml(_, bytes))) = (w.shape(&key)?.as_slice(), w.raw(&key)?) {
                    if k % 256 == 0 {
                        let q = gpu.to_device_quant(QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], g));
                        if q.is_device() {
                            return Ok(Mat::Quant(q));
                        }
                    }
                }
            }
            let (v, n, k) = crate::common::webgpu_weights::f16_matrix(w, gpu, &format!("{p}.{name}"), &mut none)?;
            Ok(Mat::F16 { v, n, k })
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

    /// `prompt`'s conditioning (`[1, 512, 7680]` on the CPU, FLUX.2's; FLUX 3's `[1, its padded length, 20480]`), as
    /// [`crate::klein::text::TextEncoder::encode`] gives it.
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        self.encode_all(&[prompt])?.pop().ok_or_else(|| err("no conditioning"))
    }

    /// Each prompt's conditioning, every prompt through a layer while its weights are on the card.
    pub fn encode_all(&mut self, prompts: &[&str]) -> Result<Vec<Tensor>> {
        let (ff, vocab, spec) = (self.ff, self.vocab, self.spec);
        let embedding = format!("{}embed_tokens.weight", spec.prefix);
        let (qd, kd) = (HEADS * HD, KV_HEADS * HD);
        let row = 2 * kd;
        let pad = self.tokenizer.token_to_id("<|endoftext|>").ok_or_else(|| err("Qwen3 tokenizer lacks its padding token"))?;
        let layers = spec.taps.iter().copied().max().unwrap_or(0);
        let mut all = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let chat = (spec.template)(prompt);
            let mut ids = self.tokenizer.encode(chat, false).map_err(err)?.get_ids().to_vec();
            // (the published conditioning: right-padded, or cut)
            let (valid, seq) = spec.pad.sizes(ids.len());
            ids.truncate(valid);
            if valid == 0 {
                candle_core::bail!("empty Klein token sequence");
            }
            ids.resize(seq, pad);
            // RoPE over each head's halves (NeoX), a position a token: a padded one at its own or the spec's
            let mut table = Vec::with_capacity(seq * HD);
            for at in 0..seq {
                let p = if at < valid { at } else { spec.pad_position.unwrap_or(at) };
                for j in 0..HD / 2 {
                    let a = p as f64 / spec.theta.powf(j as f64 / (HD / 2) as f64);
                    table.push(a.sin() as f32);
                    table.push(a.cos() as f32);
                }
            }
            if let Some(id) = ids.iter().find(|&&id| id as usize >= vocab) {
                candle_core::bail!("token {id} past the embedding's {vocab}");
            }
            let x0 = self.w.rows(&embedding, &ids)?;
            let padded = seq - valid;
            let p = Prompt {
                valid,
                seq,
                x: self.vec(seq * D),
                norm: self.vec(seq * D),
                q: self.vec(seq * qd),
                k: self.vec(seq * kd),
                v: self.vec(seq * kd),
                qq: self.vec(seq * qd),
                kk: self.vec(seq * kd),
                qpad: self.vec(padded * qd),
                apad: self.vec(if padded == 0 { 1 } else { self.gpu.attention_rows_full_out_len(padded, HEADS, HD, valid) }),
                o: self.vec(seq * D),
                g: self.vec(seq * ff),
                u: self.vec(seq * ff),
                act: self.vec(seq * ff),
                kv: self.vec(seq * row),
                att: self.vec(self.gpu.attention_rows_out_len(valid, HEADS, HD, valid)),
                attended: self.vec(seq * qd),
                table: self.vec(seq * HD),
            };
            self.gpu.upload(&p.x, &x0);
            self.gpu.upload(&p.table, &table);
            all.push(p);
        }
        let scale = 1.0 / (HD as f32).sqrt();
        // each prompt's states after the tapped layers, in turn
        let mut taps: Vec<Vec<Vec<f32>>> = all.iter().map(|_| Vec::with_capacity(spec.taps.len())).collect();
        for i in 0..layers {
            let l = self.layer(i)?;
            let tapped = spec.taps.contains(&(i + 1));
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            for p in &all {
                let (valid, seq) = (p.valid, p.seq);
                let padded = seq - valid;
                let mul = |r: &mut dyn ChainRecorder, m: &Mat, a: &DeviceVec, y: &DeviceVec| m.run(r, a, y, seq);
                r.rmsnorm_rows(&p.x, &l.input, &p.norm, seq, EPS);
                mul(r, &l.q, &p.norm, &p.q);
                mul(r, &l.k, &p.norm, &p.k);
                mul(r, &l.v, &p.norm, &p.v);
                r.rmsnorm_rows(&p.q, &l.qn, &p.qq, seq * HEADS, EPS);
                r.rmsnorm_rows(&p.k, &l.kn, &p.kk, seq * KV_HEADS, EPS);
                r.rope_rows(&p.qq, seq, HEADS, HD, &p.table, true);
                r.rope_rows(&p.kk, seq, KV_HEADS, HD, &p.table, true);
                r.store_rows(&p.kk, &p.kv, seq, kd, 0, row, 0);
                r.store_rows(&p.v, &p.kv, seq, kd, 0, row, kd);
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
                r.rmsnorm_rows(&p.x, &l.post, &p.norm, seq, EPS);
                mul(r, &l.gate, &p.norm, &p.g);
                mul(r, &l.up, &p.norm, &p.u);
                r.silu_mul(&p.g, &p.u, &p.act, seq * ff);
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
        // a token's tapped states side by side
        taps.into_iter()
            .zip(&all)
            .map(|(tap, p)| {
                let seq = p.seq;
                if tap.len() != spec.taps.len() || tap.iter().any(|t| t.len() != seq * D) {
                    return Err(err("the conditioning's tapped states were not all read"));
                }
                let mut joined = Vec::with_capacity(seq * D * tap.len());
                for t in 0..seq {
                    for states in &tap {
                        joined.extend_from_slice(&states[t * D..(t + 1) * D]);
                    }
                }
                Tensor::from_vec(joined, (1, seq, D * tap.len()), &Device::Cpu)
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
        let budget = crate::common::residency::Budget { memory: crate::common::residency::Memory::Gpu, ..Default::default() };
        let mut cpu = crate::klein::text::TextEncoder::load(&checkpoint, &tokenizer, &Device::Cpu, DType::F32, &budget)?;
        let want = cpu.encode(prompt)?;
        eprintln!("Candle load and encode {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!(got.dims(), want.dims());
        let (g, e) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
        let (seq, width) = (512, D * FLUX2.taps.len());
        let mut worst = 1f64;
        for t in 0..seq {
            let (a, b) = (&g[t * width..(t + 1) * width], &e[t * width..(t + 1) * width]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            worst = worst.min(dot / (na * nb));
        }
        eprintln!("{seq} tokens: the worst token's cosine {worst:.6}");
        assert!(worst > 0.999, "the worst token's cosine {worst}");
        Ok(())
    }
    /// FLUX 3's text encoder on WebGPU gives what Hugging Face's Qwen3-VL gives in float32 as BFL's encoder runs it
    /// (`tools/flux3/text_reference.py`: `OAIY_FLUX3_TEXT_EXPECTED`, its file; `OAIY_FLUX3_TEXT_ENCODER`, the encoder's
    /// folder): the prompt's tokens and its padding's (a multiple of 80, at position 1), the tapped layers' states.
    #[test]
    #[ignore = "needs Qwen3-VL-4B-Instruct (OAIY_FLUX3_TEXT_ENCODER), the reference's states (OAIY_FLUX3_TEXT_EXPECTED) and a WebGPU adapter"]
    fn flux3_text_is_hugging_faces() -> Result<()> {
        let (Some(dir), Some(expected)) = (std::env::var_os("OAIY_FLUX3_TEXT_ENCODER").map(std::path::PathBuf::from), std::env::var_os("OAIY_FLUX3_TEXT_EXPECTED").map(std::path::PathBuf::from)) else { return Ok(()) };
        let mut w = Weights::open(&expected)?;
        let want = w.tensor("tokens", &Device::Cpu, DType::F32)?;
        let taps: Vec<usize> = w.tensor("taps", &Device::Cpu, DType::F32)?.to_vec1::<f32>()?.into_iter().map(|t| t as usize).collect();
        let Weights::Safe(index) = &w else { unreachable!("a safetensors file") };
        let prompt = index.metadata("prompt").expect("the reference's prompt").to_owned();
        let spec = Spec { taps: if taps == [4, 8] { &[4, 8] } else { FLUX3.taps }, ..FLUX3 };
        let t = std::time::Instant::now();
        let mut gpu = WgpuKleinText::load_spec(&dir, &dir.join("tokenizer.json"), 0, spec)?;
        let got = gpu.encode(&prompt)?;
        eprintln!("WebGPU load and encode {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!(got.dims()[1..], want.dims()[..]);
        let (g, e) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
        let (seq, width) = (want.dims()[0], want.dims()[1]);
        let mut worst = (1f64, 0);
        for t in 0..seq {
            let (a, b) = (&g[t * width..(t + 1) * width], &e[t * width..(t + 1) * width]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            if dot / (na * nb) < worst.0 {
                worst = (dot / (na * nb), t);
            }
        }
        eprintln!("{seq} tokens, taps {taps:?}: the worst token's cosine {:.6} (token {})", worst.0, worst.1);
        assert!(worst.0 > 0.999, "the worst token's cosine {worst:?}");
        Ok(())
    }

    /// A GGUF of Qwen3-VL-4B-Instruct (llama.cpp's names, its K-quant blocks made f16 as each layer loads) in the
    /// checkpoint's place: the same states, within what its quantization costs.
    #[test]
    #[ignore = "needs Qwen3-VL-4B-Instruct (OAIY_FLUX3_TEXT_ENCODER), a GGUF of it (OAIY_FLUX3_TEXT_GGUF) and a WebGPU adapter"]
    fn flux3_text_from_a_gguf_is_the_checkpoints() -> Result<()> {
        let (Some(dir), Some(gguf)) = (std::env::var_os("OAIY_FLUX3_TEXT_ENCODER").map(std::path::PathBuf::from), std::env::var_os("OAIY_FLUX3_TEXT_GGUF").map(std::path::PathBuf::from)) else { return Ok(()) };
        let prompt = "Make the mug dark blue and change the words on it to \"GOOD NIGHT\" in white letters; keep everything else the same";
        let tokenizer = dir.join("tokenizer.json");
        let t = std::time::Instant::now();
        let want = WgpuKleinText::load_spec(&dir, &tokenizer, 0, FLUX3)?.encode(prompt)?;
        let checkpoint = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let got = WgpuKleinText::load_spec(&gguf, &tokenizer, 0, FLUX3)?.encode(prompt)?;
        let quantized = t.elapsed().as_secs_f64();
        assert_eq!(got.dims(), want.dims());
        let (g, e) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
        let width = want.dims()[2];
        let (mut worst, mut sum, n) = (1f64, 0f64, g.len() / width);
        for t in 0..n {
            let (a, b) = (&g[t * width..(t + 1) * width], &e[t * width..(t + 1) * width]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let c = dot / (na * nb);
            worst = worst.min(c);
            sum += c;
        }
        eprintln!("{n} positions: cosine mean {:.5}, worst {worst:.5}; loaded and encoded in {checkpoint:.1} s (checkpoint), {quantized:.1} s (GGUF)", sum / n as f64);
        assert!(worst > 0.95, "the worst position's cosine {worst}");
        Ok(())
    }
}
