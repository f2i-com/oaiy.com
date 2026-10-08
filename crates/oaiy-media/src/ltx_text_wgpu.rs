//! LTX's prompt context on WebGPU, as [`crate::ltx::text::encode`] and [`crate::ltx::transformer::connector`] make
//! it: the text model's hidden states (the embedding's and every layer's, each over its RMS, stacked: Gemma 3 12B's
//! for LTX 2.3, LTX 2.5's Gemma 4 12B's), the video stream's aggregate projection, then its connector's eight blocks
//! over 1,024 rows (the prompt's, then learned registers). One device throughout (its vectors stay there): the text
//! model's weights (24 GB as f16) a layer at a time, every prompt through it, before the projection's and the
//! connector's are loaded.
use crate::ltx::store::Store;
use crate::ltx_wgpu::{rope_table, Linear};
use candle_core::{Device, Result};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use std::path::Path;

const LAYERS: usize = 48;
const G: usize = 3840;
const HEADS: usize = 16;
/// A local layer's head, and a global one's (Gemma 4's: Gemma 3's are the local width).
const HD: usize = 256;
const GLOBAL_HD: usize = 512;
/// The widest a layer's keys are (local: 8 heads of 256; Gemma 4's global: one of 512).
const KV_WIDTH: usize = 8 * HD;
const FF: usize = 15360;
const EPS: f32 = 1e-6;
/// The connector's rows (the prompt's, then registers), width and heads.
const ROWS: usize = 1024;
const D: usize = 4096;
const CHEADS: usize = 32;
/// The audio stream's connector's width.
const AUDIO: usize = 2048;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A bias-free matrix (`[n, k]`) as f16, or (`bf16`) as BF16 the way its file holds it.
struct Mat {
    v: DeviceVec,
    n: usize,
    k: usize,
    bf16: bool,
}

/// [`Mat`] on the host, read and not yet on the GPU: a BF16 checkpoint's bytes as they are (the kernels widen BF16
/// by a shift: no pass over 460 MB a layer to make f16 of them), else f16 words converted from what the file holds.
struct MatHost {
    weights: HostWeights,
    n: usize,
    k: usize,
}

enum HostWeights {
    Bf16(Vec<u8>),
    F16(Vec<f32>),
    /// BF16 matrices still in their file: each one's file, its bytes' start there and how many, one after another.
    Spans(Vec<(std::path::PathBuf, u64, usize)>),
}

/// The bytes from `at` on of `spans`' (one after another), `part.len()` of them, read into `part`.
fn read_spans(spans: &[(std::path::PathBuf, u64, usize)], at: usize, part: &mut [u8]) -> std::io::Result<()> {
    let (mut begin, end) = (0usize, at + part.len());
    for (path, start, len) in spans {
        let (a, b) = (at.max(begin), end.min(begin + len));
        if a < b {
            dsv41::io::read_exact_at(&std::fs::File::open(path)?, &mut part[a - at..b - at], start + (a - begin) as u64)?;
        }
        begin += len;
    }
    Ok(())
}

impl MatHost {
    fn up(self, gpu: &ggml_rs_wgpu::WgpuBackend) -> Result<Mat> {
        Ok(match self.weights {
            HostWeights::Bf16(bytes) => Mat { v: gpu.vec_of_bytes(&bytes), n: self.n, k: self.k, bf16: true },
            HostWeights::F16(words) => {
                let v = gpu.vec(words.len());
                gpu.upload(&v, &words);
                Mat { v, n: self.n, k: self.k, bf16: false }
            }
            // the file's bytes read straight into the write's staging memory, each thread its own part of them: read
            // into the host's memory first they were copied twice, the second copy beside the next layer's first
            HostWeights::Spans(spans) => {
                let total: usize = spans.iter().map(|s| s.2).sum();
                let failed = std::sync::Mutex::new(None::<String>);
                let fill = |at: usize, part: &mut [u8]| {
                    if let Err(e) = read_spans(&spans, at, part) {
                        *failed.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                    }
                };
                let filled = gpu.vec_filled(total, &fill);
                if let Some(e) = failed.into_inner().unwrap_or_else(|p| p.into_inner()) {
                    return Err(err(e));
                }
                let v = match filled {
                    Some(v) => v,
                    None => {
                        let mut bytes = vec![0u8; total];
                        read_spans(&spans, 0, &mut bytes).map_err(err)?;
                        gpu.vec_of_bytes(&bytes)
                    }
                };
                Mat { v, n: self.n, k: self.k, bf16: true }
            }
        })
    }
}

/// `names`' matrices one after another (their rows) as f16 words on the host, a BF16 checkpoint's converted from its
/// bytes on every core.
fn mat_host(store: &mut Store, names: &[String]) -> Result<MatHost> {
    let (mut n, mut k) = (0, 0);
    for name in names {
        let shape = store.index.info(name).map_err(err)?.shape.clone();
        let &[rows, cols] = shape.as_slice() else { candle_core::bail!("{name}: a matrix of shape {shape:?}") };
        if k != 0 && cols != k {
            candle_core::bail!("{name}: {cols} columns, not {k}");
        }
        k = cols;
        n += rows;
    }
    // BF16 (an even width, so a row is whole words): the file's bytes, one matrix's after another; where they lie
    // in it, for the upload to read them there (OAIY_LTX_TEXT_COPY: read here, into the host's memory, as before)
    let direct = std::env::var_os("OAIY_LTX_TEXT_COPY").is_none();
    let mut bytes: Vec<u8> = Vec::new();
    let mut spans = Vec::new();
    let mut all = k % 2 == 0 && std::env::var_os("OAIY_LTX_TEXT_F16").is_none();
    for name in names {
        if !all {
            break;
        }
        if direct {
            match store.bf16_span(name)? {
                Some(span) => spans.push(span),
                None => all = false,
            }
            continue;
        }
        match store.bf16_bytes(name)? {
            Some(part) if bytes.is_empty() => bytes = part,
            Some(part) => bytes.extend(part),
            None => all = false,
        }
    }
    if all {
        return Ok(MatHost { weights: if direct { HostWeights::Spans(spans) } else { HostWeights::Bf16(bytes) }, n, k });
    }
    let mut words: Vec<f32> = Vec::new();
    for name in names {
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
    Ok(MatHost { weights: HostWeights::F16(words), n, k })
}

fn vector(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, add: f32) -> Result<DeviceVec> {
    Ok(vector_up(gpu, &vector_host(store, name, add)?))
}

/// A vector's values on the host, `add` added to each.
fn vector_host(store: &mut Store, name: &str, add: f32) -> Result<Vec<f32>> {
    Ok(store.tensor_f32(name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| v + add).collect())
}

fn vector_up(gpu: &ggml_rs_wgpu::WgpuBackend, values: &[f32]) -> DeviceVec {
    let v = gpu.vec(values.len());
    gpu.upload(&v, values);
    v
}

struct Layer {
    /// The norms' scales (Gemma 3's `1 + w`, Gemma 4's `w`).
    input: DeviceVec,
    post_attn: DeviceVec,
    pre_ff: DeviceVec,
    post_ff: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    q: Mat,
    k: Mat,
    /// None where the keys serve as values too (Gemma 4's global layers).
    v: Option<Mat>,
    o: Mat,
    /// The gate's rows, then the up projection's (one matmul).
    gate_up: Mat,
    down: Mat,
    /// Its head's width and its key heads.
    hd: usize,
    kv: usize,
    /// Gemma 4's layer scalar, less one, for every channel (the layer's output times it: an affine's scale).
    scalar: Option<DeviceVec>,
}

fn mul(r: &mut dyn ChainRecorder, m: &Mat, x: &DeviceVec, y: &DeviceVec, rows: usize) {
    // (an LLM's hidden states reach hundreds: their f32 into the matmul as they are)
    if m.bf16 {
        r.matmul_bf16_rows_f32(&m.v, m.n, m.k, x, y, rows);
    } else {
        r.matmul_f16_rows_f32(&m.v, m.n, m.k, x, y, rows);
    }
}

/// [`Layer`] on the host: its weights read and converted to f16 words, not yet on the GPU (a thread of their own
/// makes them a layer ahead of the one the GPU is given: [`contexts_streams`]).
struct LayerHost {
    input: Vec<f32>,
    post_attn: Vec<f32>,
    pre_ff: Vec<f32>,
    post_ff: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    q: MatHost,
    k: MatHost,
    v: Option<MatHost>,
    o: MatHost,
    gate_up: MatHost,
    down: MatHost,
    scalar: Option<Vec<f32>>,
}

/// The text model's layer `i` (its weights under `prefix`; `gemma4` LTX 2.5's Gemma 4) read and converted.
fn layer_host(store: &mut Store, prefix: &str, i: usize, gemma4: bool) -> Result<LayerHost> {
    let p = format!("{prefix}layers.{i}.");
    let n = |s: &str| format!("{p}{s}");
    // Gemma 3's norms scale by 1 + w, Gemma 4's by w
    let add = if gemma4 { 0. } else { 1. };
    let v_name = n("self_attn.v_proj.weight");
    let scalar = if gemma4 {
        let s = store.tensor_f32(&n("layer_scalar"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        Some(vec![s.first().copied().ok_or_else(|| err(format!("{p}layer_scalar is empty")))? - 1.; G])
    } else {
        None
    };
    Ok(LayerHost {
        input: vector_host(store, &n("input_layernorm.weight"), add)?,
        post_attn: vector_host(store, &n("post_attention_layernorm.weight"), add)?,
        pre_ff: vector_host(store, &n("pre_feedforward_layernorm.weight"), add)?,
        post_ff: vector_host(store, &n("post_feedforward_layernorm.weight"), add)?,
        qn: vector_host(store, &n("self_attn.q_norm.weight"), add)?,
        kn: vector_host(store, &n("self_attn.k_norm.weight"), add)?,
        q: mat_host(store, &[n("self_attn.q_proj.weight")])?,
        k: mat_host(store, &[n("self_attn.k_proj.weight")])?,
        v: if store.index.get(&v_name).is_some() { Some(mat_host(store, &[v_name])?) } else { None },
        o: mat_host(store, &[n("self_attn.o_proj.weight")])?,
        gate_up: mat_host(store, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?,
        down: mat_host(store, &[n("mlp.down_proj.weight")])?,
        scalar,
    })
}

/// Layer `i`'s weights put on `gpu` (`gemma4` LTX 2.5's Gemma 4).
fn layer_up(gpu: &ggml_rs_wgpu::WgpuBackend, h: LayerHost, i: usize, gemma4: bool) -> Result<Layer> {
    let hd = h.qn.len();
    let (q, k) = (h.q.up(gpu)?, h.k.up(gpu)?);
    let l = Layer {
        input: vector_up(gpu, &h.input),
        post_attn: vector_up(gpu, &h.post_attn),
        pre_ff: vector_up(gpu, &h.pre_ff),
        post_ff: vector_up(gpu, &h.post_ff),
        kn: vector_up(gpu, &h.kn),
        qn: vector_up(gpu, &h.qn),
        v: h.v.map(|v| v.up(gpu)).transpose()?,
        o: h.o.up(gpu)?,
        gate_up: h.gate_up.up(gpu)?,
        down: h.down.up(gpu)?,
        kv: k.n / hd.max(1),
        hd,
        q,
        k,
        scalar: h.scalar.map(|s| vector_up(gpu, &s)),
    };
    let fits = matches!(l.hd, HD | GLOBAL_HD) && l.q.n == HEADS * l.hd && l.k.n == l.kv * l.hd && l.kv >= 1 && l.kv * l.hd <= KV_WIDTH && l.o.k == l.q.n && l.gate_up.n == 2 * FF && (l.v.is_some() || gemma4);
    if !fits {
        candle_core::bail!("not Gemma 3 or Gemma 4 12B at layer {i} (q {}, k {}, head {}, gate and up {})", l.q.n, l.k.n, l.hd, l.gate_up.n);
    }
    Ok(l)
}

/// A prompt's vectors through the text model: its rows `x`, its rotary tables, its scratch (each its widest layer's
/// size), and its stacked states.
struct Prompt {
    s: usize,
    x: DeviceVec,
    local: DeviceVec,
    global: DeviceVec,
    h: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    vv: DeviceVec,
    vn: DeviceVec,
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

/// `v`'s first `len` values (a norm takes a row's width from its vector's length).
fn first(v: &DeviceVec, len: usize) -> DeviceVec {
    DeviceVec { len, inner: v.inner.clone() }
}

/// Each prompt's video context (`[1024, 4096]`, the connector's), from the text model at `gemma` (Gemma 3 12B, its
/// weights under `model.` or the multimodal release's `language_model.model.`, `tokenizer` its tokenizer.json; or LTX
/// 2.5's Gemma 4 12B, its tokenizer and aggregate projection in its own file) and the projection (where Gemma's file
/// has none) and connector in `transformer` (LTX's checkpoint), on GPU `device` (as CUDA counts them;
/// OAIY_WEBGPU_ADAPTER naming one instead). `progress(step, of)` as it goes.
pub fn contexts(gemma: &Path, tokenizer: Option<&Path>, transformer: &mut Store, prompts: &[String], device: usize, progress: impl FnMut(usize, usize)) -> Result<Vec<Vec<f32>>> {
    Ok(contexts_streams(gemma, tokenizer, transformer, prompts, device, false, progress)?.into_iter().map(|(video, _)| video).collect())
}

/// [`contexts`], and with `audio` each prompt's audio context beside its video one (`[1024, 2048]`: the same stacked
/// states through the audio stream's aggregate projection and its own connector, as
/// [`crate::ltx::text::encode`] and the reference's `audio_embeddings_connector`).
pub fn contexts_streams(gemma: &Path, tokenizer: Option<&Path>, transformer: &mut Store, prompts: &[String], device: usize, audio: bool, mut progress: impl FnMut(usize, usize)) -> Result<Vec<(Vec<f32>, Option<Vec<f32>>)>> {
    // (OAIY_LOAD_PROFILE: where a prompt's context's time goes, part by part)
    let profile = std::env::var_os("OAIY_LOAD_PROFILE").is_some();
    let mut lap = std::time::Instant::now();
    let mut said = |what: &str| {
        if profile {
            eprintln!("the text context: {what} {:.2} s", lap.elapsed().as_secs_f64());
        }
        lap = std::time::Instant::now();
    };
    let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
    said("the device");
    let mut store = Store::open(gemma, 0)?;
    said("the text model's file opened");
    let prefix = if store.index.get("language_model.model.embed_tokens.weight").is_some() { "language_model.model." } else { "model." };
    let gemma4 = store.index.get(&format!("{prefix}layers.0.layer_scalar")).is_some();
    let mut tok = match tokenizer {
        Some(path) => tokenizers::Tokenizer::from_file(path).map_err(err)?,
        None if gemma4 => tokenizers::Tokenizer::from_bytes(store.index.read("tokenizer_json").map_err(err)?).map_err(err)?,
        None => candle_core::bail!("LTX 2.3 requires a Gemma 3 tokenizer path"),
    };
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
    said("the tokenizer and the prompts' tokens");
    let steps = LAYERS + 10 + if audio { 9 } else { 0 };
    // the global layers' heads: Gemma 3's the local width (positions over 8), Gemma 4's twice it (a quarter of its
    // pairs turned, positions as they are); attention unscaled in Gemma 4
    let (global_hd, global_factor, global_turned) = if gemma4 { (GLOBAL_HD, 1., GLOBAL_HD / 8) } else { (HD, 8., HD / 2) };
    let scale = if gemma4 { 1. } else { 1. / (HD as f32).sqrt() };
    // its hidden states, each prompt's stacked on the GPU: a layer's weights at a time, every prompt through it
    let stacked: Vec<(DeviceVec, usize)> = {
        let ones = gpu.vec(G);
        gpu.upload(&ones, &vec![1.0; G]);
        let mut prompts = Vec::with_capacity(ids_all.len());
        for ids in &ids_all {
            let s = ids.len();
            let emb = store.rows(&format!("{prefix}embed_tokens.weight"), ids, &Device::Cpu)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let x0: Vec<f32> = emb.iter().map(|v| v * (G as f32).sqrt()).collect();
            // RoPE over each head's halves, its first `turned` pairs (the rest at angle 0), positions after the
            // reference's left padding to 1,024
            let offset = ROWS - s;
            let table = |hd: usize, base: f64, factor: f64, turned: usize| -> Vec<f32> {
                let mut t = Vec::with_capacity(s * hd);
                for p in 0..s {
                    for j in 0..hd / 2 {
                        let a = if j < turned { (p + offset) as f64 / (base.powf(2. * j as f64 / hd as f64) * factor) } else { 0. };
                        t.extend([a.sin() as f32, a.cos() as f32]);
                    }
                }
                t
            };
            let (local, global) = (gpu.vec(s * HD), gpu.vec(s * global_hd));
            gpu.upload(&local, &table(HD, 10000., 1., HD / 2));
            gpu.upload(&global, &table(global_hd, 1e6, global_factor, global_turned));
            let x = gpu.vec(s * G);
            gpu.upload(&x, &x0);
            let v = |len: usize| gpu.vec(len.max(1));
            let qw = HEADS * global_hd.max(HD);
            let att = gpu.attention_rows_out_len(s, HEADS, HD, s).max(gpu.attention_rows_out_len(s, HEADS, global_hd, s));
            prompts.push(Prompt {
                s,
                x,
                local,
                global,
                h: v(s * G),
                q: v(s * qw),
                k: v(s * KV_WIDTH),
                vv: v(s * KV_WIDTH),
                vn: v(s * KV_WIDTH),
                qq: v(s * qw),
                kk: v(s * KV_WIDTH),
                kv: v(s * 2 * KV_WIDTH),
                o: v(s * G),
                gu: v(s * 2 * FF),
                act: v(s * FF),
                an: v(s * G),
                fs: v(s * G),
                att: v(att),
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
        said("the prompts' embeddings and vectors");
        // a layer's weights are read and converted to f16 by a thread of their own (its own handle on the file), a
        // layer ahead of the one the GPU is given: one thread reading, converting, uploading and running each layer in
        // turn took 0.27 s a layer with the file in the system's cache, of it 0.06 the read and 0.06 the conversion
        std::thread::scope(|scope| -> Result<()> {
            // (at most a layer waiting and a layer being made: a gigabyte; the receiver gone, at an error here, ends
            // the reader at its next layer)
            let (made, ahead) = std::sync::mpsc::sync_channel::<std::result::Result<LayerHost, String>>(1);
            scope.spawn(move || {
                let mut file = match Store::open(gemma, 0) {
                    Ok(file) => file,
                    Err(e) => {
                        let _ = made.send(Err(e.to_string()));
                        return;
                    }
                };
                for i in 0..LAYERS {
                    if made.send(layer_host(&mut file, prefix, i, gemma4).map_err(|e| e.to_string())).is_err() {
                        return;
                    }
                }
            });
            // (OAIY_LOAD_PROFILE: where the layers' time went)
            let (mut waited, mut up, mut recorded, mut ran) = (0f64, 0f64, 0f64, 0f64);
            for i in 0..LAYERS {
                let clock = std::time::Instant::now();
                let host = ahead.recv().map_err(err)?.map_err(err)?;
                waited += clock.elapsed().as_secs_f64();
                let clock = std::time::Instant::now();
                let l = layer_up(&gpu, host, i, gemma4)?;
                up += clock.elapsed().as_secs_f64();
                let clock = std::time::Instant::now();
                let final_norm = if i == LAYERS - 1 { Some(vector(&mut store, &gpu, &format!("{prefix}norm.weight"), if gemma4 { 0. } else { 1. })?) } else { None };
                let (hd, kv) = (l.hd, l.kv);
                let (qd, kd) = (HEADS * hd, kv * hd);
                let mut rec = gpu.begin();
                rec.keep_groups(false);
                let r = rec.as_mut();
                for p in &prompts {
                    let (s, x, h) = (p.s, &p.x, &p.h);
                    let tv = if i % 6 == 5 { &p.global } else { &p.local };
                    r.rmsnorm_rows(x, &l.input, h, s, EPS);
                    mul(r, &l.q, h, &p.q, s);
                    mul(r, &l.k, h, &p.k, s);
                    match &l.v {
                        Some(v) => mul(r, v, h, &p.vv, s),
                        // (the keys as they come, before their norm and rotary, as values)
                        None => r.copy(&p.k, 0, &p.vv, 0, s * kd),
                    }
                    r.rmsnorm_rows(&first(&p.q, s * qd), &l.qn, &first(&p.qq, s * qd), s * HEADS, EPS);
                    r.rmsnorm_rows(&first(&p.k, s * kd), &l.kn, &first(&p.kk, s * kd), s * kv, EPS);
                    r.rope_rows(&p.qq, s, HEADS, hd, tv, true);
                    r.rope_rows(&p.kk, s, kv, hd, tv, true);
                    let row = 2 * kd;
                    r.store_rows(&p.kk, &p.kv, s, kd, 0, row, 0);
                    if gemma4 {
                        // Gemma 4's values over their RMS (no weights)
                        r.rmsnorm_rows(&first(&p.vv, s * kd), &ones, &first(&p.vn, s * kd), s * kv, EPS);
                        r.store_rows(&p.vn, &p.kv, s, kd, 0, row, kd);
                    } else {
                        r.store_rows(&p.vv, &p.kv, s, kd, 0, row, kd);
                    }
                    r.attention_rows(&p.qq, &p.kv, &p.att, s, HEADS, kv, hd, 0, None, scale);
                    mul(r, &l.o, &p.att, &p.o, s);
                    r.rmsnorm_rows(&p.o, &l.post_attn, &p.an, s, EPS);
                    r.add(x, &p.an);
                    r.rmsnorm_rows(x, &l.pre_ff, h, s, EPS);
                    mul(r, &l.gate_up, h, &p.gu, s);
                    r.gelu_mul_split_rows(&p.gu, &p.act, s);
                    mul(r, &l.down, &p.act, &p.o, s);
                    r.rmsnorm_rows(&p.o, &l.post_ff, &p.an, s, EPS);
                    r.add(x, &p.an);
                    if let Some(sc) = &l.scalar {
                        // the layer's output times its scalar (an affine: x (1 + (scalar - 1)))
                        r.norm_mod_rows(x, &p.an, s, G, sc, 0, None, RowNorm::None, EPS);
                        r.copy(&p.an, 0, x, 0, s * G);
                    }
                    match &final_norm {
                        Some(norm) => {
                            r.rmsnorm_rows(x, norm, h, s, EPS);
                            stack(r, p, h, i + 1);
                        }
                        None => stack(r, p, x, i + 1),
                    }
                }
                recorded += clock.elapsed().as_secs_f64();
                let clock = std::time::Instant::now();
                rec.finish();
                drop(l);
                ran += clock.elapsed().as_secs_f64();
                progress(i + 1, steps);
            }
            if std::env::var_os("OAIY_LOAD_PROFILE").is_some() {
                eprintln!("the text model's {LAYERS} layers: waiting for their weights' reader {waited:.2} s, putting them on the GPU {up:.2} s, recording {recorded:.2} s, their runs {ran:.2} s");
            }
            Ok(())
        })?;
        prompts.into_iter().map(|p| (p.feats, p.s)).collect()
    };
    said("the layers");
    gpu.release_cached();
    // a stream's projection (its input over sqrt(3840 / its width), folded into its weights; Gemma 4's in its own file)
    // and its connector: the video's 4,096 wide, the audio's 2,048
    let g = |n: &str| format!("model.diffusion_model.{n}");
    let mut done = LAYERS;
    let mut stream = |store: &mut Store, transformer: &mut Store, which: &str, width: usize| -> Result<Vec<Vec<f32>>> {
        let name = format!("text_embedding_projection.{which}_aggregate_embed");
        let own = store.index.get(&format!("{name}.weight")).is_some();
        let proj = projection(if own { store } else { &mut *transformer }, &gpu, &name, (width as f32 / G as f32).sqrt())?;
        said("a stream's projection");
        done += 1;
        progress(done, steps);
        let registers = transformer.tensor_f32(&g(&format!("{which}_embeddings_connector.learnable_registers")), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        if proj.n != width || registers.len() != 128 * width {
            candle_core::bail!("the {which} stream's context: a projection to {} and {} register values, for a width of {width}", proj.n, registers.len());
        }
        let mut blocks = Vec::with_capacity(8);
        for i in 0..8 {
            let p = g(&format!("{which}_embeddings_connector.transformer_1d_blocks.{i}"));
            blocks.push(crate::ltx_wgpu::ConnectorBlock::load(transformer, &gpu, &p)?);
            done += 1;
            progress(done, steps);
        }
        said("its registers and its connector's eight blocks");
        let rope = rope_table(&(0..ROWS).map(|i| vec![i as f32]).collect::<Vec<_>>(), &[4096.], width, CHEADS);
        let table = gpu.vec(rope.len());
        gpu.upload(&table, &rope);
        let mut contexts = Vec::new();
        for (feats, s) in &stacked {
            let s = *s;
            let x = gpu.vec(ROWS * width);
            // the learned registers past the prompt's rows (register i % 128 at row i)
            let pad: Vec<f32> = (s..ROWS).flat_map(|i| registers[(i % 128) * width..(i % 128 + 1) * width].iter().copied()).collect();
            if !pad.is_empty() {
                gpu.upload_at(&x, s * width, &pad);
            }
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            proj.forward(r, feats, &x, s);
            let out = crate::ltx_wgpu::connector(&gpu, r, &blocks, &x, ROWS, &table);
            r.read(&out);
            contexts.push(rec.finish().pop().ok_or_else(|| err("the context was not read"))?);
        }
        said("its runs");
        Ok(contexts)
    };
    let video = stream(&mut store, transformer, "video", D)?;
    let sound = if audio { Some(stream(&mut store, transformer, "audio", AUDIO)?) } else { None };
    drop(stream);
    drop(store);
    progress(steps, steps);
    let mut sound = sound.map(|s| s.into_iter());
    Ok(video.into_iter().map(|v| (v, sound.as_mut().and_then(|s| s.next()))).collect())
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
    // (BF16 as its file holds it, widened where it is permuted: 770 million values made f32 on one core first took
    // half of the projection's time)
    let (halves, wide) = match store.bf16_bytes(&key)? {
        Some(bytes) => (bytes, Vec::new()),
        None => (Vec::new(), store.tensor_f32(&key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
    };
    let at = |i: usize| if wide.is_empty() { f32::from_bits((u16::from_le_bytes([halves[2 * i], halves[2 * i + 1]]) as u32) << 16) } else { wide[i] };
    // each row's (state, channel) word pairs, rounded to f16 on every core
    let mut words = vec![0f32; n * k / 2];
    let ok = std::sync::atomic::AtomicBool::new(true);
    let rows: Vec<(usize, &mut [f32])> = words.chunks_mut(k / 2).enumerate().collect();
    std::thread::scope(|s| {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
        let mut rows = rows.into_iter();
        for _ in 0..threads {
            let mine: Vec<(usize, &mut [f32])> = rows.by_ref().take(n.div_ceil(threads)).collect();
            let (at, ok) = (&at, &ok);
            s.spawn(move || {
                for (o, dst) in mine {
                    let row = o * k;
                    let mut fine = true;
                    for (j, word) in dst.iter_mut().enumerate() {
                        let (i, c) = ((2 * j) / G, (2 * j) % G);
                        let (lo, hi) = (at(row + c * states + i) * scale, at(row + (c + 1) * states + i) * scale);
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

    /// What a prompt's context costs (`--ignored --nocapture`): Gemma 3 12B (`OAIY_LTX_GEMMA`; its tokenizer
    /// `OAIY_LTX_TOKENIZER`, else the folder's tokenizer.json) and the projection and connector of an LTX 2.3
    /// checkpoint (`OAIY_LTX_NVFP4`), twice (the second with the files in the system's cache): the time at a few of
    /// its steps, and the context's values' sum and a hash of their bits, the same from one build to the next unless
    /// its arithmetic changes.
    #[test]
    #[ignore = "a timing; needs Gemma 3 12B (OAIY_LTX_GEMMA), LTX 2.3 (OAIY_LTX_NVFP4) and a WebGPU adapter"]
    fn measure_a_prompts_context() -> Result<()> {
        let (Some(gemma), Some(ltx)) = (std::env::var_os("OAIY_LTX_GEMMA"), std::env::var_os("OAIY_LTX_NVFP4")) else { return Ok(()) };
        let (gemma, ltx) = (std::path::PathBuf::from(gemma), std::path::PathBuf::from(ltx));
        let gemma4 = Store::open(&gemma, 0)?.index.get("model.layers.0.layer_scalar").is_some();
        let tokenizer = (!gemma4).then(|| std::env::var_os("OAIY_LTX_TOKENIZER").map_or_else(|| gemma.join("tokenizer.json"), std::path::PathBuf::from));
        let prompt = "A red fox trots through fresh snow at sunrise, its breath steaming, the camera tracking beside it".to_string();
        for round in 0..2 {
            let mut store = Store::open(&ltx, 0)?;
            let started = std::time::Instant::now();
            let mut marks = Vec::new();
            let got = contexts(&gemma, tokenizer.as_deref(), &mut store, std::slice::from_ref(&prompt), 0, |n, of| {
                if [1, 12, 24, 36, LAYERS, of].contains(&n) {
                    marks.push(format!("step {n} at {:.1} s", started.elapsed().as_secs_f64()));
                }
            })?
            .remove(0);
            let sum: f64 = got.iter().map(|v| v.abs() as f64).sum();
            let hash = got.iter().fold(0xcbf29ce484222325u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3));
            eprintln!("round {round}: a prompt's context in {:.1} s ({}); {} values, their sum {sum:.3}, their bits' hash {hash:016x}", started.elapsed().as_secs_f64(), marks.join(", "), got.len());
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
        // (Gemma 4's in its own file; Gemma 3's OAIY_LTX_TOKENIZER or the folder's)
        let gemma4 = Store::open(&gemma, 0)?.index.get("model.layers.0.layer_scalar").is_some();
        let tokenizer = (!gemma4).then(|| std::env::var_os("OAIY_LTX_TOKENIZER").map_or_else(|| gemma.join("tokenizer.json"), std::path::PathBuf::from));
        let prompt = "A red fox trots through fresh snow at sunrise, its breath steaming, the camera tracking beside it".to_string();
        let t = std::time::Instant::now();
        let mut store = Store::open(&ltx, 0)?;
        let started = std::time::Instant::now();
        let got = contexts(&gemma, tokenizer.as_deref(), &mut store, std::slice::from_ref(&prompt), 0, |n, of| {
            if [1, 24, LAYERS, LAYERS + 1, LAYERS + 9, of].contains(&n) {
                eprintln!("step {n} of {of}: {:.1} s", started.elapsed().as_secs_f64());
            }
        })?
        .remove(0);
        eprintln!("WebGPU context {:.1} s", t.elapsed().as_secs_f64());
        let dev = Device::Cpu;
        let t = std::time::Instant::now();
        let (features, _) = crate::ltx::text::encode(&gemma, tokenizer.as_deref(), &mut store, &prompt, gemma4, false, &dev, |_| {})?;
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
