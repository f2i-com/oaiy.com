//! LTX 2.3's prompt context on WebGPU, as [`crate::ltx::text::encode`] and [`crate::ltx::transformer::connector`]
//! make it: Gemma 3 12B's hidden states (the embedding's and every layer's, each over its RMS, stacked), the video
//! stream's aggregate projection, then its connector's eight blocks over 1,024 rows (the prompt's, then learned
//! registers). One device throughout (its vectors stay there): Gemma's weights (24 GB as f16) let go before the
//! projection's and the connector's are loaded.
use crate::ltx::store::Store;
use crate::ltx_wgpu::{rope_table, Linear};
use candle_core::{Device, Result};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::path::Path;

const LAYERS: usize = 48;
const G: usize = 3840;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const GHD: usize = 256;
const FF: usize = 15360;
const EPS: f32 = 1e-6;
/// The connector's rows (the prompt's, then registers), width and heads.
const ROWS: usize = 1024;
const D: usize = 4096;
const CHEADS: usize = 32;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A bias-free matrix as f16 (`[n, k]`).
struct Mat {
    v: DeviceVec,
    n: usize,
    k: usize,
}

fn mat(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, names: &[String]) -> Result<Mat> {
    let mut values = Vec::new();
    let (mut n, mut k) = (0, 0);
    for name in names {
        let t = store.tensor_f32(name, &Device::Cpu)?;
        let (rows, cols) = t.dims2()?;
        if k != 0 && cols != k {
            candle_core::bail!("{name}: {cols} columns, not {k}");
        }
        k = cols;
        n += rows;
        values.extend(t.flatten_all()?.to_vec1::<f32>()?);
    }
    let v = gpu.vec_f16_rounded(&values).ok_or_else(|| err(format!("{}: a weight past f16's range", names[0])))?;
    Ok(Mat { v, n, k })
}

fn vector(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, add: f32) -> Result<DeviceVec> {
    let values: Vec<f32> = store.tensor_f32(name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| v + add).collect();
    let v = gpu.vec(values.len());
    gpu.upload(&v, &values);
    Ok(v)
}

struct Layer {
    /// The norms' `1 + w`.
    input: DeviceVec,
    post_attn: DeviceVec,
    pre_ff: DeviceVec,
    post_ff: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    /// The gate's rows, then the up projection's (one matmul).
    gate_up: Mat,
    down: Mat,
}

fn mul(r: &mut dyn ChainRecorder, m: &Mat, x: &DeviceVec, y: &DeviceVec, rows: usize) {
    // (an LLM's hidden states reach hundreds: their f32 into the matmul as they are)
    r.matmul_f16_rows_f32(&m.v, m.n, m.k, x, y, rows);
}

/// Each prompt's video context (`[1024, 4096]`, the connector's), from Gemma 3 12B at `gemma` (its text weights
/// under `model.` or the multimodal release's `language_model.model.`; `tokenizer` its tokenizer.json) and the
/// projection and connector in `transformer` (LTX 2.3's checkpoint), on GPU `device` (as CUDA counts them;
/// OAIY_WEBGPU_ADAPTER naming one instead). `progress(step, of)` as it goes.
pub fn contexts(gemma: &Path, tokenizer: &Path, transformer: &mut Store, prompts: &[String], device: usize, mut progress: impl FnMut(usize, usize)) -> Result<Vec<Vec<f32>>> {
    let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
    let mut tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(err)?;
    tok.with_padding(None);
    tok.with_truncation(None).map_err(err)?;
    let mut ids_all = Vec::new();
    for p in prompts {
        let mut ids = tok.encode(p.trim(), true).map_err(err)?.get_ids().to_vec();
        if ids.first() != Some(&2) {
            ids.insert(0, 2);
        }
        ids.truncate(ROWS);
        ids_all.push(ids);
    }
    let steps = LAYERS + 10;
    // Gemma's hidden states, each prompt's stacked on the GPU
    let stacked: Vec<(DeviceVec, usize)> = {
        let mut store = Store::open(gemma, 0)?;
        let prefix = if store.index.get("language_model.model.embed_tokens.weight").is_some() { "language_model.model." } else { "model." };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("{prefix}layers.{i}.");
            let n = |s: &str| format!("{p}{s}");
            layers.push(Layer {
                input: vector(&mut store, &gpu, &n("input_layernorm.weight"), 1.)?,
                post_attn: vector(&mut store, &gpu, &n("post_attention_layernorm.weight"), 1.)?,
                pre_ff: vector(&mut store, &gpu, &n("pre_feedforward_layernorm.weight"), 1.)?,
                post_ff: vector(&mut store, &gpu, &n("post_feedforward_layernorm.weight"), 1.)?,
                qn: vector(&mut store, &gpu, &n("self_attn.q_norm.weight"), 1.)?,
                kn: vector(&mut store, &gpu, &n("self_attn.k_norm.weight"), 1.)?,
                q: mat(&mut store, &gpu, &[n("self_attn.q_proj.weight")])?,
                k: mat(&mut store, &gpu, &[n("self_attn.k_proj.weight")])?,
                v: mat(&mut store, &gpu, &[n("self_attn.v_proj.weight")])?,
                o: mat(&mut store, &gpu, &[n("self_attn.o_proj.weight")])?,
                gate_up: mat(&mut store, &gpu, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?,
                down: mat(&mut store, &gpu, &[n("mlp.down_proj.weight")])?,
            });
            progress(i + 1, steps);
        }
        if layers[0].q.n != HEADS * GHD || layers[0].k.n != KV_HEADS * GHD || layers[0].gate_up.n != 2 * FF {
            candle_core::bail!("not Gemma 3 12B (q {}, k {}, gate and up {})", layers[0].q.n, layers[0].k.n, layers[0].gate_up.n);
        }
        let final_norm = vector(&mut store, &gpu, &format!("{prefix}norm.weight"), 1.)?;
        let ones = gpu.vec(G);
        gpu.upload(&ones, &vec![1.0; G]);
        let mut out = Vec::new();
        for ids in &ids_all {
            let s = ids.len();
            let emb = store.rows(&format!("{prefix}embed_tokens.weight"), ids, &Device::Cpu)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let scale = (G as f32).sqrt();
            let x0: Vec<f32> = emb.iter().map(|v| v * scale).collect();
            // RoPE over each head's halves: the local layers' (base 10,000) and the global ones' (1e6, positions over 8);
            // positions after the reference's left padding to 1,024
            let offset = ROWS - s;
            let table = |base: f64, factor: f64| -> Vec<f32> {
                let mut t = Vec::with_capacity(s * GHD);
                for p in 0..s {
                    for j in 0..GHD / 2 {
                        let a = (p + offset) as f64 / (base.powf(2. * j as f64 / GHD as f64) * factor);
                        t.extend([a.sin() as f32, a.cos() as f32]);
                    }
                }
                t
            };
            let (local, global) = (gpu.vec(s * GHD), gpu.vec(s * GHD));
            gpu.upload(&local, &table(10000., 1.));
            gpu.upload(&global, &table(1e6, 8.));
            let x = gpu.vec(s * G);
            gpu.upload(&x, &x0);
            let v = |len: usize| gpu.vec(len.max(1));
            let (h, q, k, vv, qq, kk, kv, o, gu, act) = (v(s * G), v(s * HEADS * GHD), v(s * KV_HEADS * GHD), v(s * KV_HEADS * GHD), v(s * HEADS * GHD), v(s * KV_HEADS * GHD), v(s * 2 * KV_HEADS * GHD), v(s * G), v(s * 2 * FF), v(s * FF));
            let (an, fs) = (v(s * G), v(s * G));
            let att = v(gpu.attention_rows_out_len(s.div_ceil(32) * 32, HEADS, GHD, s));
            let feats = v(s * (LAYERS + 1) * G);
            let row = 2 * KV_HEADS * GHD;
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            // each state over its RMS, into its place among the token's 49
            let stack = |r: &mut dyn ChainRecorder, x: &DeviceVec, i: usize| {
                r.rmsnorm_rows(x, &ones, &fs, s, EPS);
                r.store_rows(&fs, &feats, s, G, 0, (LAYERS + 1) * G, i * G);
            };
            stack(r, &x, 0);
            for (i, l) in layers.iter().enumerate() {
                let tv = if i % 6 == 5 { &global } else { &local };
                r.rmsnorm_rows(&x, &l.input, &h, s, EPS);
                mul(r, &l.q, &h, &q, s);
                mul(r, &l.k, &h, &k, s);
                mul(r, &l.v, &h, &vv, s);
                r.rmsnorm_rows(&q, &l.qn, &qq, s * HEADS, EPS);
                r.rmsnorm_rows(&k, &l.kn, &kk, s * KV_HEADS, EPS);
                r.rope_rows(&qq, s, HEADS, GHD, tv, true);
                r.rope_rows(&kk, s, KV_HEADS, GHD, tv, true);
                r.store_rows(&kk, &kv, s, KV_HEADS * GHD, 0, row, 0);
                r.store_rows(&vv, &kv, s, KV_HEADS * GHD, 0, row, KV_HEADS * GHD);
                r.attention_rows(&qq, &kv, &att, s, HEADS, KV_HEADS, GHD, 0, None, 1.0 / (GHD as f32).sqrt());
                mul(r, &l.o, &att, &o, s);
                r.rmsnorm_rows(&o, &l.post_attn, &an, s, EPS);
                r.add(&x, &an);
                r.rmsnorm_rows(&x, &l.pre_ff, &h, s, EPS);
                mul(r, &l.gate_up, &h, &gu, s);
                r.gelu_mul_split_rows(&gu, &act, s);
                mul(r, &l.down, &act, &o, s);
                r.rmsnorm_rows(&o, &l.post_ff, &an, s, EPS);
                r.add(&x, &an);
                if i == LAYERS - 1 {
                    r.rmsnorm_rows(&x, &final_norm, &h, s, EPS);
                    stack(r, &h, i + 1);
                } else {
                    stack(r, &x, i + 1);
                }
            }
            rec.finish();
            out.push((feats, s));
        }
        out
    };
    gpu.release_cached();
    // the projection (its input over sqrt(3840 / 4096), folded into its weights) and the connector
    let g = |n: &str| format!("model.diffusion_model.{n}");
    let proj = projection(transformer, &gpu, "text_embedding_projection.video_aggregate_embed", (D as f32 / G as f32).sqrt())?;
    progress(LAYERS + 1, steps);
    let registers = transformer.tensor_f32(&g("video_embeddings_connector.learnable_registers"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut blocks = Vec::with_capacity(8);
    for i in 0..8 {
        let p = g(&format!("video_embeddings_connector.transformer_1d_blocks.{i}"));
        blocks.push(crate::ltx_wgpu::ConnectorBlock::load(transformer, &gpu, &p)?);
        progress(LAYERS + 2 + i, steps);
    }
    let rope = rope_table(&(0..ROWS).map(|i| vec![i as f32]).collect::<Vec<_>>(), &[4096.], D, CHEADS);
    let table = gpu.vec(rope.len());
    gpu.upload(&table, &rope);
    let mut contexts = Vec::new();
    for (feats, s) in stacked {
        let x = gpu.vec(ROWS * D);
        // the learned registers past the prompt's rows (register i % 128 at row i)
        let pad: Vec<f32> = (s..ROWS).flat_map(|i| registers[(i % 128) * D..(i % 128 + 1) * D].iter().copied()).collect();
        if !pad.is_empty() {
            gpu.upload_at(&x, s * D, &pad);
        }
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        proj.forward(r, &feats, &x, s);
        let out = crate::ltx_wgpu::connector(&gpu, r, &blocks, &x, ROWS, &table);
        r.read(&out);
        let c = rec.finish().pop().ok_or_else(|| err("the context was not read"))?;
        contexts.push(c);
    }
    progress(steps, steps);
    Ok(contexts)
}

/// The aggregate projection `name` (`[n, 3840 x 49]`) for the features as stacked here (each state's 3,840 channels
/// in turn, where the reference's stack has each channel's 49 states together: its columns permuted to match), its
/// input's scale folded into its weights.
fn projection(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, scale: f32) -> Result<Linear> {
    let t = store.tensor_f32(&format!("{name}.weight"), &Device::Cpu)?;
    let (n, k) = t.dims2()?;
    let states = LAYERS + 1;
    if k != G * states {
        candle_core::bail!("{name}: {k} inputs, not {}", G * states);
    }
    let w = t.flatten_all()?.to_vec1::<f32>()?;
    let mut permuted = vec![0f32; n * k];
    for o in 0..n {
        let (src, dst) = (&w[o * k..(o + 1) * k], &mut permuted[o * k..(o + 1) * k]);
        for c in 0..G {
            for i in 0..states {
                dst[i * G + c] = src[c * states + i] * scale;
            }
        }
    }
    let v = gpu.vec_f16_rounded(&permuted).ok_or_else(|| err(format!("{name}: past f16's range")))?;
    let bias = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let b = gpu.vec(n);
    gpu.upload(&b, &bias);
    Ok(Linear { weight: crate::ltx_wgpu::Weight::F16(v), bias: b, n, k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// The WebGPU prompt context is Candle's (on CUDA, BF16): Gemma 3 12B (`OAIY_LTX_GEMMA`, its folder or one file;
    /// its tokenizer `OAIY_LTX_TOKENIZER`, else the folder's tokenizer.json), then the projection and connector of an
    /// LTX 2.3 checkpoint (`OAIY_LTX_NVFP4`), a prompt.
    #[test]
    #[ignore = "needs Gemma 3 12B (OAIY_LTX_GEMMA), LTX 2.3 (OAIY_LTX_NVFP4), a WebGPU adapter and CUDA (the cuda feature)"]
    fn the_webgpu_prompt_context_is_the_candle_one() -> Result<()> {
        let (Some(gemma), Some(ltx)) = (std::env::var_os("OAIY_LTX_GEMMA"), std::env::var_os("OAIY_LTX_NVFP4")) else { return Ok(()) };
        let (gemma, ltx) = (std::path::PathBuf::from(gemma), std::path::PathBuf::from(ltx));
        let tokenizer = std::env::var_os("OAIY_LTX_TOKENIZER").map_or_else(|| gemma.join("tokenizer.json"), std::path::PathBuf::from);
        let prompt = "A red fox trots through fresh snow at sunrise, its breath steaming, the camera tracking beside it".to_string();
        let t = std::time::Instant::now();
        let mut store = Store::open(&ltx, 0)?;
        let got = contexts(&gemma, &tokenizer, &mut store, std::slice::from_ref(&prompt), 0, |_, _| {})?.remove(0);
        eprintln!("WebGPU context {:.1} s", t.elapsed().as_secs_f64());
        #[cfg(feature = "cuda")]
        let dev = Device::new_cuda(std::env::var("OAIY_LTX_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        #[cfg(not(feature = "cuda"))]
        let dev = Device::Cpu;
        let t = std::time::Instant::now();
        let (features, _) = crate::ltx::text::encode(&gemma, Some(&tokenizer), &mut store, &prompt, false, false, &dev, |_| {})?;
        let want = crate::ltx::transformer::connector(&mut store, &features, "video_embeddings_connector", &dev, |_| {})?;
        eprintln!("Candle context {:.1} s", t.elapsed().as_secs_f64());
        let want = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got.len(), want.len());
        let mut worst = 1f64;
        for t in 0..ROWS {
            let (a, b) = (&got[t * D..(t + 1) * D], &want[t * D..(t + 1) * D]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            worst = worst.min(dot / (na * nb));
        }
        eprintln!("1,024 rows: the worst row's cosine {worst:.6}");
        assert!(worst > 0.99, "the worst row's cosine {worst}");
        Ok(())
    }
}
