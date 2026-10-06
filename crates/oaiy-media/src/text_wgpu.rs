//! Qwen Image 2.1's text encoder (Qwen3-VL 8B's language model) on WebGPU, as [`crate::text::TextEncoder`] encodes a
//! prompt, its reference images' vision features in place of their placeholders (each token's three rotary axes, the
//! features' deeper layers added after the first three): the last decoder layer's states before the final norm, the
//! system prefix dropped.
//! The weights f16 on the GPU (the BF16 checkpoint's rounded) a layer at a time, every prompt through it (its 16 GB
//! never all on the card), the embedding's rows looked up on the host; the matmuls read their inputs as f32 (an LLM's
//! hidden states reach some 500).
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
    /// The weights, a layer's read as the prompts reach it, and their names' prefix.
    w: Weights,
    prefix: String,
    tokenizer: tokenizers::Tokenizer,
}

/// A prompt's vectors through the layers.
struct Prompt {
    s: usize,
    drop: usize,
    /// Its images' token runs (`(start, len)`, among all its tokens) and the vectors their deeper features are added
    /// from (a layer's at a time: zeros past the runs).
    spans: Vec<(usize, usize)>,
    deep: DeviceVec,
    x: DeviceVec,
    norm: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    o: DeviceVec,
    g: DeviceVec,
    u: DeviceVec,
    act: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    table: DeviceVec,
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
        let shape = |w: &Weights, name: &str| w.shape(&format!("{prefix}.layers.0.{name}.weight"));
        let (q, k, gate) = (shape(&w, "self_attn.q_proj")?, shape(&w, "self_attn.k_proj")?, shape(&w, "mlp.gate_proj")?);
        if q != [HEADS * HD, D] || k != [KV_HEADS * HD, D] || gate != [FF, D] {
            candle_core::bail!("not Qwen3-VL 8B's language model (q {q:?}, k {k:?}, gate {gate:?})");
        }
        let embedding = w.tensor(&format!("{prefix}.embed_tokens.weight"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("processor/tokenizer.json")).map_err(err)?;
        Ok(Self { gpu, embedding, w, prefix: prefix.to_owned(), tokenizer })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// Layer `i`'s weights on the GPU.
    fn layer(&mut self, i: usize) -> Result<Layer> {
        let p = format!("{}.layers.{i}", self.prefix);
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

    /// `prompt`'s conditioning (as [`crate::text::TextEncoder::encode`] gives it with no reference images): its states
    /// on the CPU.
    pub fn encode(&mut self, prompt: &str) -> Result<Conditioning> {
        self.encode_all(&[prompt], &[])?.pop().ok_or_else(|| err("no conditioning"))
    }

    /// Each prompt's conditioning (as [`crate::text::TextEncoder::encode`] gives it, `images` the reference images'
    /// vision features for every prompt), every prompt through a layer while its weights are on the card.
    pub fn encode_all(&mut self, prompts: &[&str], images: &[crate::vision::Features]) -> Result<Vec<Conditioning>> {
        const IMAGE_PAD: u32 = 151655;
        let system = self.tokenizer.encode(SYSTEM, false).map_err(err)?.len();
        let vocab = self.embedding.len() / D;
        let host = |t: &Tensor| -> Result<Vec<f32>> { t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let embeddings: Vec<Vec<f32>> = images.iter().map(|f| host(&f.embedding)).collect::<Result<_>>()?;
        let deeps: Vec<Vec<Vec<f32>>> = images.iter().map(|f| f.deep.iter().map(host).collect::<Result<Vec<_>>>()).collect::<Result<_>>()?;
        let image_prefix = (1..=images.len()).map(|i| format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>")).collect::<Vec<_>>().join(" ");
        let mut all = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let text = format!("{SYSTEM}<|im_start|>user\n{image_prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n");
            let tokens = self.tokenizer.encode(text.as_str(), false).map_err(err)?.get_ids().to_vec();
            if tokens.len() > 1024 {
                candle_core::bail!("image prompt exceeds 1024 text tokens");
            }
            // each image's placeholder as its features' grid (their positions its rows and columns from where it
            // starts), the text's a position a token on all three axes
            let (mut ids, mut spans, mut positions, mut position) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
            for id in tokens {
                if id == IMAGE_PAD {
                    let image = images.get(spans.len()).ok_or_else(|| err("unexpected image placeholder in prompt"))?;
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
            let s = ids.len();
            let mut x0 = Vec::with_capacity(s * D);
            for &id in &ids {
                let id = id as usize;
                if id >= vocab {
                    candle_core::bail!("token {id} past the embedding's {vocab}");
                }
                x0.extend_from_slice(&self.embedding[id * D..(id + 1) * D]);
            }
            for (&(start, len), e) in spans.iter().zip(&embeddings) {
                if e.len() != len * D {
                    candle_core::bail!("an image's features: {} values for {len} tokens", e.len());
                }
                x0[start * D..(start + len) * D].copy_from_slice(e);
            }
            // RoPE over each head's halves (NeoX): Qwen3-VL's interleaved axes, a pair's from its index (every third
            // past the first of the 60 the rows' or the columns', the rest time's)
            let mut table = Vec::with_capacity(s * HD);
            for pos in &positions {
                for j in 0..HD / 2 {
                    let axis = if j < 60 && j % 3 != 0 { j % 3 } else { 0 };
                    let a = pos[axis] as f64 / THETA.powf(j as f64 / (HD / 2) as f64);
                    table.push(a.sin() as f32);
                    table.push(a.cos() as f32);
                }
            }
            let p = Prompt {
                s,
                drop: system,
                deep: self.vec(if spans.is_empty() { 1 } else { s * D }),
                spans,
                x: self.vec(s * D),
                norm: self.vec(s * D),
                q: self.vec(s * D),
                k: self.vec(s * KV_HEADS * HD),
                v: self.vec(s * KV_HEADS * HD),
                qq: self.vec(s * D),
                kk: self.vec(s * KV_HEADS * HD),
                o: self.vec(s * D),
                g: self.vec(s * FF),
                u: self.vec(s * FF),
                act: self.vec(s * FF),
                kv: self.vec(s * 2 * KV_HEADS * HD),
                att: self.vec(self.gpu.attention_rows_out_len(s.div_ceil(32) * 32, HEADS, HD, s)),
                table: self.vec(s * HD),
            };
            self.gpu.upload(&p.x, &x0);
            self.gpu.upload(&p.table, &table);
            all.push(p);
        }
        let row = 2 * KV_HEADS * HD;
        for i in 0..LAYERS {
            let l = self.layer(i)?;
            // the images' deeper features for this layer's output (the first three's), zeros past their runs
            let deep_at = (i < 3 && !images.is_empty()).then_some(i);
            if let Some(layer) = deep_at {
                for p in &all {
                    let mut v = vec![0f32; p.s * D];
                    for (&(start, len), d) in p.spans.iter().zip(&deeps) {
                        let f = d.get(layer).ok_or_else(|| err("an image's deeper features"))?;
                        if f.len() != len * D {
                            candle_core::bail!("an image's deeper features: {} values for {len} tokens", f.len());
                        }
                        v[start * D..(start + len) * D].copy_from_slice(f);
                    }
                    self.gpu.upload(&p.deep, &v);
                }
            }
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            for p in &all {
                let s = p.s;
                let mul = |r: &mut dyn ChainRecorder, m: &Mat, a: &DeviceVec, y: &DeviceVec| r.matmul_f16_rows_f32(&m.v, m.n, m.k, a, y, s);
                r.rmsnorm_rows(&p.x, &l.input, &p.norm, s, EPS);
                mul(r, &l.q, &p.norm, &p.q);
                mul(r, &l.k, &p.norm, &p.k);
                mul(r, &l.v, &p.norm, &p.v);
                r.rmsnorm_rows(&p.q, &l.qn, &p.qq, s * HEADS, EPS);
                r.rmsnorm_rows(&p.k, &l.kn, &p.kk, s * KV_HEADS, EPS);
                r.rope_rows(&p.qq, s, HEADS, HD, &p.table, true);
                r.rope_rows(&p.kk, s, KV_HEADS, HD, &p.table, true);
                r.store_rows(&p.kk, &p.kv, s, KV_HEADS * HD, 0, row, 0);
                r.store_rows(&p.v, &p.kv, s, KV_HEADS * HD, 0, row, KV_HEADS * HD);
                r.attention_rows(&p.qq, &p.kv, &p.att, s, HEADS, KV_HEADS, HD, 0, None, 1.0 / (HD as f32).sqrt());
                mul(r, &l.o, &p.att, &p.o);
                r.add(&p.x, &p.o);
                r.rmsnorm_rows(&p.x, &l.post, &p.norm, s, EPS);
                mul(r, &l.gate, &p.norm, &p.g);
                mul(r, &l.up, &p.norm, &p.u);
                r.silu_mul(&p.g, &p.u, &p.act, s * FF);
                mul(r, &l.down, &p.act, &p.o);
                r.add(&p.x, &p.o);
                if deep_at.is_some() {
                    r.add(&p.x, &p.deep);
                }
            }
            rec.finish();
            drop(l);
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        for p in &all {
            rec.read_range(&p.x, p.drop * D, (p.s - p.drop) * D);
        }
        let reads = rec.finish();
        all.iter()
            .zip(reads)
            .map(|(p, states)| Ok(Conditioning { states: Tensor::from_vec(states, (1, p.s - p.drop, D), &Device::Cpu)?, spans: p.spans.iter().map(|&(s, n)| (s - p.drop, n)).collect() }))
            .collect()
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

    /// How long a 1024x1024 reference's encoding takes on the CPU (`--ignored --nocapture`): its VAE latent and its
    /// vision features (Candle's, F32), each.
    #[test]
    #[ignore = "a timing; needs Qwen Image 2.1 (OAIY_QWEN_IMAGE_BASE) and a reference image (OAIY_QWEN_IMAGE_REFERENCE)"]
    fn measure_a_references_encoding_on_the_cpu() -> Result<()> {
        let (Some(base), Some(image)) = (std::env::var_os("OAIY_QWEN_IMAGE_BASE").map(std::path::PathBuf::from), std::env::var_os("OAIY_QWEN_IMAGE_REFERENCE").map(std::path::PathBuf::from)) else { return Ok(()) };
        let reference = crate::reference::Reference::load(&image, 1024)?;
        let t = std::time::Instant::now();
        let vae = crate::vae::Vae::load_encoder(&base, &Device::Cpu, DType::F32)?;
        let loaded = t.elapsed().as_secs_f64();
        let latent = vae.encode(&reference.pixels(&Device::Cpu, DType::F32)?)?;
        eprintln!("VAE encoder: loaded {loaded:.1} s, encoded {:.1} s ({:?})", t.elapsed().as_secs_f64() - loaded, latent.dims());
        drop(vae);
        let t = std::time::Instant::now();
        let vision = crate::vision::VisionEncoder::load(&base.join("text_encoder"), &Device::Cpu, DType::F32)?;
        let loaded = t.elapsed().as_secs_f64();
        let f = vision.encode(&reference)?;
        eprintln!("vision encoder: loaded {loaded:.1} s, encoded {:.1} s ({} by {})", t.elapsed().as_secs_f64() - loaded, f.h, f.w);
        Ok(())
    }

    /// With a reference image (`OAIY_QWEN_IMAGE_REFERENCE`, at 512 by 512): its vision features (Candle's, on the CPU)
    /// in place of its placeholder, its tokens' three rotary axes, its deeper features added; the states and the
    /// image's span as Candle's.
    #[test]
    #[ignore = "needs Qwen Image 2.1's text encoder (OAIY_QWEN_IMAGE_BASE), a reference image (OAIY_QWEN_IMAGE_REFERENCE) and a WebGPU adapter"]
    fn a_reference_images_conditioning_is_the_candle_ones() -> Result<()> {
        let (Some(base), Some(image)) = (std::env::var_os("OAIY_QWEN_IMAGE_BASE").map(std::path::PathBuf::from), std::env::var_os("OAIY_QWEN_IMAGE_REFERENCE").map(std::path::PathBuf::from)) else { return Ok(()) };
        let prompt = "Make it night time, the cafe lit by warm lamps";
        let reference = crate::reference::Reference::load(&image, 512)?;
        let vision = crate::vision::VisionEncoder::load(&base.join("text_encoder"), &Device::Cpu, DType::F32)?;
        let features = vec![vision.encode(&reference)?];
        drop(vision);
        let mut gpu = WgpuTextEncoder::load(&base, None, 0)?;
        let got = gpu.encode_all(&[prompt], &features)?.remove(0);
        drop(gpu);
        let budget = crate::residency::Budget::default();
        let mut cpu = crate::text::TextEncoder::load(&base, None, &Device::Cpu, DType::F32, &budget)?;
        let want = cpu.encode(prompt, &features)?;
        assert_eq!(got.spans, want.spans);
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
        eprintln!("{n} tokens ({:?} the image's): the worst token's cosine {worst:.6}", want.spans);
        assert!(worst > 0.999, "the worst token's cosine {worst}");
        Ok(())
    }
}
