//! LTX 2.3's prompt context on WebGPU, as [`crate::ltx::text::encode`] and [`crate::ltx::transformer::connector`]
//! make it: Gemma 3 12B's hidden states (the embedding's and every layer's, each over its RMS, stacked), the video
//! stream's aggregate projection, then its connector's eight blocks over 1,024 rows (the prompt's, then learned
//! registers). One device throughout (its vectors stay there): Gemma's weights (24 GB as f16) a layer at a time,
//! every prompt through it, before the projection's and the connector's are loaded.
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

/// `names`' matrices one after another (their rows) as f16, a BF16 checkpoint's converted from its bytes on every core.
fn mat(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, names: &[String]) -> Result<Mat> {
    let mut words: Vec<f32> = Vec::new();
    let (mut n, mut k) = (0, 0);
    for name in names {
        let shape = store.index.info(name).map_err(err)?.shape.clone();
        let &[rows, cols] = shape.as_slice() else { candle_core::bail!("{name}: a matrix of shape {shape:?}") };
        if k != 0 && cols != k {
            candle_core::bail!("{name}: {cols} columns, not {k}");
        }
        k = cols;
        n += rows;
        let part = match store.bf16_bytes(name)? {
            Some(bytes) => crate::wgpu_weights::f16_words(&bytes),
            None => crate::wgpu_weights::f16_words_f32(&store.tensor_f32(name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
        }
        .ok_or_else(|| err(format!("{name}: a weight past f16's range")))?;
        if words.is_empty() {
            words = part;
        } else {
            words.extend(part);
        }
    }
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
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

/// Gemma's layer `i` (its weights under `prefix`) on `gpu`.
fn layer(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, prefix: &str, i: usize) -> Result<Layer> {
    let p = format!("{prefix}layers.{i}.");
    let n = |s: &str| format!("{p}{s}");
    let l = Layer {
        input: vector(store, gpu, &n("input_layernorm.weight"), 1.)?,
        post_attn: vector(store, gpu, &n("post_attention_layernorm.weight"), 1.)?,
        pre_ff: vector(store, gpu, &n("pre_feedforward_layernorm.weight"), 1.)?,
        post_ff: vector(store, gpu, &n("post_feedforward_layernorm.weight"), 1.)?,
        qn: vector(store, gpu, &n("self_attn.q_norm.weight"), 1.)?,
        kn: vector(store, gpu, &n("self_attn.k_norm.weight"), 1.)?,
        q: mat(store, gpu, &[n("self_attn.q_proj.weight")])?,
        k: mat(store, gpu, &[n("self_attn.k_proj.weight")])?,
        v: mat(store, gpu, &[n("self_attn.v_proj.weight")])?,
        o: mat(store, gpu, &[n("self_attn.o_proj.weight")])?,
        gate_up: mat(store, gpu, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?,
        down: mat(store, gpu, &[n("mlp.down_proj.weight")])?,
    };
    if l.q.n != HEADS * GHD || l.k.n != KV_HEADS * GHD || l.gate_up.n != 2 * FF {
        candle_core::bail!("not Gemma 3 12B (q {}, k {}, gate and up {})", l.q.n, l.k.n, l.gate_up.n);
    }
    Ok(l)
}

/// A prompt's vectors through Gemma: its rows `x`, its rotary tables, its scratch, and its stacked states.
struct Prompt {
    s: usize,
    x: DeviceVec,
    local: DeviceVec,
    global: DeviceVec,
    h: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    vv: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    kv: DeviceVec,
    o: DeviceVec,
    gu: DeviceVec,
    act: DeviceVec,
    an: DeviceVec,
    fs: DeviceVec,
    att: DeviceVec,
    feats: DeviceVec,
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
    // Gemma's hidden states, each prompt's stacked on the GPU: a layer's weights at a time, every prompt through it
    let stacked: Vec<(DeviceVec, usize)> = {
        let mut store = Store::open(gemma, 0)?;
        let prefix = if store.index.get("language_model.model.embed_tokens.weight").is_some() { "language_model.model." } else { "model." };
        let ones = gpu.vec(G);
        gpu.upload(&ones, &vec![1.0; G]);
        let mut prompts = Vec::with_capacity(ids_all.len());
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
            prompts.push(Prompt {
                s,
                x,
                local,
                global,
                h: v(s * G),
                q: v(s * HEADS * GHD),
                k: v(s * KV_HEADS * GHD),
                vv: v(s * KV_HEADS * GHD),
                qq: v(s * HEADS * GHD),
                kk: v(s * KV_HEADS * GHD),
                kv: v(s * 2 * KV_HEADS * GHD),
                o: v(s * G),
                gu: v(s * 2 * FF),
                act: v(s * FF),
                an: v(s * G),
                fs: v(s * G),
                att: v(gpu.attention_rows_out_len(s.div_ceil(32) * 32, HEADS, GHD, s)),
                feats: v(s * (LAYERS + 1) * G),
            });
        }
        // each state over its RMS, into its place among the token's 49
        let stack = |r: &mut dyn ChainRecorder, p: &Prompt, x: &DeviceVec, i: usize| {
            r.rmsnorm_rows(x, &ones, &p.fs, p.s, EPS);
            r.store_rows(&p.fs, &p.feats, p.s, G, 0, (LAYERS + 1) * G, i * G);
        };
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        for p in &prompts {
            stack(rec.as_mut(), p, &p.x, 0);
        }
        rec.finish();
        let row = 2 * KV_HEADS * GHD;
        for i in 0..LAYERS {
            let l = layer(&mut store, &gpu, prefix, i)?;
            let final_norm = if i == LAYERS - 1 { Some(vector(&mut store, &gpu, &format!("{prefix}norm.weight"), 1.)?) } else { None };
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            for p in &prompts {
                let (s, x, h) = (p.s, &p.x, &p.h);
                let tv = if i % 6 == 5 { &p.global } else { &p.local };
                r.rmsnorm_rows(x, &l.input, h, s, EPS);
                mul(r, &l.q, h, &p.q, s);
                mul(r, &l.k, h, &p.k, s);
                mul(r, &l.v, h, &p.vv, s);
                r.rmsnorm_rows(&p.q, &l.qn, &p.qq, s * HEADS, EPS);
                r.rmsnorm_rows(&p.k, &l.kn, &p.kk, s * KV_HEADS, EPS);
                r.rope_rows(&p.qq, s, HEADS, GHD, tv, true);
                r.rope_rows(&p.kk, s, KV_HEADS, GHD, tv, true);
                r.store_rows(&p.kk, &p.kv, s, KV_HEADS * GHD, 0, row, 0);
                r.store_rows(&p.vv, &p.kv, s, KV_HEADS * GHD, 0, row, KV_HEADS * GHD);
                r.attention_rows(&p.qq, &p.kv, &p.att, s, HEADS, KV_HEADS, GHD, 0, None, 1.0 / (GHD as f32).sqrt());
                mul(r, &l.o, &p.att, &p.o, s);
                r.rmsnorm_rows(&p.o, &l.post_attn, &p.an, s, EPS);
                r.add(x, &p.an);
                r.rmsnorm_rows(x, &l.pre_ff, h, s, EPS);
                mul(r, &l.gate_up, h, &p.gu, s);
                r.gelu_mul_split_rows(&p.gu, &p.act, s);
                mul(r, &l.down, &p.act, &p.o, s);
                r.rmsnorm_rows(&p.o, &l.post_ff, &p.an, s, EPS);
                r.add(x, &p.an);
                match &final_norm {
                    Some(norm) => {
                        r.rmsnorm_rows(x, norm, h, s, EPS);
                        stack(r, p, h, i + 1);
                    }
                    None => stack(r, p, x, i + 1),
                }
            }
            rec.finish();
            drop(l);
            progress(i + 1, steps);
        }
        prompts.into_iter().map(|p| (p.feats, p.s)).collect()
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
    let key = format!("{name}.weight");
    let shape = store.index.info(&key).map_err(err)?.shape.clone();
    let &[n, k] = shape.as_slice() else { candle_core::bail!("{key}: a matrix of shape {shape:?}") };
    let states = LAYERS + 1;
    if k != G * states {
        candle_core::bail!("{name}: {k} inputs, not {}", G * states);
    }
    let w = match store.bf16_bytes(&key)? {
        Some(bytes) => bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        None => store.tensor_f32(&key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?,
    };
    // each row's (state, channel) word pairs, rounded to f16 on every core
    let mut words = vec![0f32; n * k / 2];
    let ok = std::sync::atomic::AtomicBool::new(true);
    let rows: Vec<(usize, &mut [f32])> = words.chunks_mut(k / 2).enumerate().collect();
    std::thread::scope(|s| {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
        let mut rows = rows.into_iter();
        for _ in 0..threads {
            let mine: Vec<(usize, &mut [f32])> = rows.by_ref().take(n.div_ceil(threads)).collect();
            let (w, ok) = (&w, &ok);
            s.spawn(move || {
                for (o, dst) in mine {
                    let src = &w[o * k..(o + 1) * k];
                    let mut fine = true;
                    for (j, word) in dst.iter_mut().enumerate() {
                        let (i, c) = ((2 * j) / G, (2 * j) % G);
                        let (lo, hi) = (src[c * states + i] * scale, src[(c + 1) * states + i] * scale);
                        fine &= lo.abs() <= 65504.0 && hi.abs() <= 65504.0;
                        *word = f32::from_bits(half::f16::from_f32(lo).to_bits() as u32 | (half::f16::from_f32(hi).to_bits() as u32) << 16);
                    }
                    if !fine {
                        ok.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });
    if !ok.into_inner() {
        candle_core::bail!("{name}: past f16's range");
    }
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
    let bias = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let b = gpu.vec(n);
    gpu.upload(&b, &bias);
    Ok(Linear { weight: crate::ltx_wgpu::Weight::F16(v), bias: b, n, k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// Where a Gemma layer's load goes (`--ignored --nocapture`, `OAIY_LTX_GEMMA`): its bytes read, converted to f16,
    /// put on the GPU.
    #[test]
    #[ignore = "a timing; needs Gemma 3 12B (OAIY_LTX_GEMMA) and a WebGPU adapter"]
    fn measure_a_gemma_layers_load() -> Result<()> {
        let Some(gemma) = std::env::var_os("OAIY_LTX_GEMMA") else { return Ok(()) };
        let mut store = Store::open(std::path::Path::new(&gemma), 0)?;
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(err)?;
        let names: Vec<String> = store.index.names().filter(|n| n.contains("layers.10.") && n.ends_with("proj.weight")).map(str::to_owned).collect();
        for round in 0..2 {
            let (mut read, mut convert, mut upload, mut bytes) = (0f64, 0f64, 0f64, 0usize);
            for name in &names {
                let t = std::time::Instant::now();
                let b = store.bf16_bytes(name)?.expect("BF16");
                read += t.elapsed().as_secs_f64();
                bytes += b.len();
                let t = std::time::Instant::now();
                let words = crate::wgpu_weights::f16_words(&b).expect("in f16's range");
                convert += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let v = gpu.vec(words.len());
                gpu.upload(&v, &words);
                gpu.settle();
                upload += t.elapsed().as_secs_f64();
            }
            eprintln!("round {round}: {:.0} MB read in {read:.3} s, converted in {convert:.3} s, uploaded in {upload:.3} s", bytes as f64 / 1e6);
        }
        Ok(())
    }

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
        let started = std::time::Instant::now();
        let got = contexts(&gemma, &tokenizer, &mut store, std::slice::from_ref(&prompt), 0, |n, of| {
            if [1, 24, LAYERS, LAYERS + 1, LAYERS + 9, of].contains(&n) {
                eprintln!("step {n} of {of}: {:.1} s", started.elapsed().as_secs_f64());
            }
        })?
        .remove(0);
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
