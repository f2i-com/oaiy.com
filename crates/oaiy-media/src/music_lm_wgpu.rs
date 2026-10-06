//! MiniMax Music 3's autoregressive stage on WebGPU ([`crate::music::lm`]'s; a music job's `backend` "webgpu"): the 8B
//! Qwen3 language model and the 4-layer depth decoder on the device, the conditional and the unconditional row stepped
//! together (each matrix read once for both, each row attending over its own keys and values), the sampling on the
//! host. The language model's layers f16 where the device has room for them (13.9 GB: as fast as Candle's on CUDA, and
//! nearer the BF16 checkpoint), else Q8_0 (7.4 GB), as [`lm_f16`] decides; a quantized file's as it stores them; the
//! depth decoder and the output heads f16.
use crate::ltx::store::Store;
use crate::music::lm::{self, Rng, AUDIO_END, AUDIO_VOCAB, CODEBOOKS, CODE_OFFSET, HIDDEN, SEMANTIC_CODES};
use crate::wgpu_weights::{f16_words, f16_words_f32, q8_0_bf16};
use candle_core::{quantized::gguf_file, Device, Result, Tensor};
use ggml_rs::{Backend, ChainRecorder, DeviceChain, DeviceVec, QuantizedTensor};
use ggml_rs_wgpu::WgpuBackend;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("music on WebGPU: {e}"))
}

fn upload(gpu: &WgpuBackend, v: &[f32]) -> DeviceVec {
    let d = gpu.vec(v.len());
    gpu.upload(&d, v);
    d
}

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// The language model's layers as f16: their 36 layers' four matrices, 47,104 rows of 4096 each.
const F16_LAYERS: u64 = 36 * 47_104 * 4096 * 2;
/// The depth decoder as f16, and a position's keys and values in the language model's caches (both rows, F32).
const F16_DEPTH: u64 = 1_232_000_000;
const KV_POSITION: u64 = 36 * 2 * 8 * 128 * 4 * 2;

/// Whether the language model's layers go f16 rather than Q8_0: as OAIY_MUSIC_WEBGPU_WEIGHTS says (`f16` or `q8_0`),
/// else where the device has room for them, the depth decoder and `positions`' keys and values with 2 GB to spare.
/// (Q8_0's matrices take a step twice f16's for two rows: no int8 kernel of a few rows for its blocks yet.)
pub fn lm_f16(gpu: &WgpuBackend, positions: usize) -> bool {
    match std::env::var("OAIY_MUSIC_WEBGPU_WEIGHTS").map(|v| v.to_ascii_lowercase()).as_deref() {
        Ok("f16") => return true,
        Ok("q8_0") => return false,
        _ => {}
    }
    let need = F16_LAYERS + F16_DEPTH + positions as u64 * KV_POSITION + (2 << 30);
    match gpu.memory_budget() {
        Some((budget, usage)) => budget.saturating_sub(usage) >= need,
        None => gpu.usage().1 >= need,
    }
}

/// A matrix `[n, k]` on the device: f16, or quantized (Q8_0, or a quantized file's type).
enum Mat {
    F16 { w: DeviceVec, n: usize, k: usize },
    Quant(QuantizedTensor),
}

impl Mat {
    /// f32 values (`[n, k]`) as f16.
    fn f16(gpu: &WgpuBackend, v: &[f32], n: usize, k: usize, name: &str) -> Result<Self> {
        let words = f16_words_f32(v).ok_or_else(|| err(format!("{name}: past f16's range")))?;
        Ok(Self::F16 { w: upload(gpu, &words), n, k })
    }

    /// BF16 bytes (`[n, k]`, row-major) as Q8_0 where `quantize` and `k` is a multiple of 256, else f16.
    fn bf16(gpu: &WgpuBackend, bytes: &[u8], n: usize, k: usize, quantize: bool, name: &str) -> Result<Self> {
        if bytes.len() != n * k * 2 {
            return Err(err(format!("{name}: {} bytes, not [{n}, {k}] BF16", bytes.len())));
        }
        if quantize && k % 256 == 0 {
            return Self::quant(gpu, QuantizedTensor::from_bytes_cpu(q8_0_bf16(bytes), vec![n, k], ggml_quants::GgmlType::Q8_0), name);
        }
        let words = f16_words(bytes).ok_or_else(|| err(format!("{name}: past f16's range")))?;
        Ok(Self::F16 { w: upload(gpu, &words), n, k })
    }

    fn quant(gpu: &WgpuBackend, q: QuantizedTensor, name: &str) -> Result<Self> {
        let bytes = q.shape().iter().product::<usize>();
        let q = gpu.to_device_quant(q);
        if !q.is_device() {
            return Err(err(format!("{name}: no room on the GPU for its {bytes} values")));
        }
        Ok(Self::Quant(q))
    }

    /// A quantized file's matrix as it stores it.
    fn gguf(gpu: &WgpuBackend, q: &candle_core::quantized::QTensor, name: &str) -> Result<Self> {
        let t = crate::wgpu_weights::ggml(q.dtype()).filter(|t| ggml_quants::is_supported(*t)).ok_or_else(|| err(format!("{name}: no kernel for {:?}", q.dtype())))?;
        let &[n, k] = q.shape().dims() else { return Err(err(format!("{name}: not a matrix"))) };
        Self::quant(gpu, QuantizedTensor::from_bytes_cpu(q.data()?.into_owned(), vec![n, k], t), name)
    }

    fn n(&self) -> usize {
        match self {
            Self::F16 { n, .. } => *n,
            Self::Quant(q) => q.shape()[0],
        }
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match self {
            Self::F16 { w, n, k } => r.matmul_f16_rows(w, *n, *k, x, y, rows),
            Self::Quant(q) => r.matmul_rows(q, x, y, rows),
        }
    }
}

struct Layer {
    input_norm: DeviceVec,
    /// q, k and v's rows in one matrix.
    qkv: Mat,
    q_norm: Option<DeviceVec>,
    k_norm: Option<DeviceVec>,
    o: Mat,
    post_norm: DeviceVec,
    /// The gate's rows, then the up projection's.
    gate_up: Mat,
    down: Mat,
}

/// A decoder's layers (the language model's: per-head q/k norms and RoPE; the depth decoder's: neither).
struct Decoder {
    layers: Vec<Layer>,
    norm: DeviceVec,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    ff: usize,
    eps: f32,
    rope: bool,
}

/// A sequence's keys and values: each layer's rows (keys, then values) for `capacity` positions.
pub struct Cache {
    kv: Vec<DeviceVec>,
    len: usize,
    capacity: usize,
}

impl Decoder {
    fn cache(&self, gpu: &WgpuBackend, capacity: usize) -> Cache {
        let row = 2 * self.kv_heads * self.head_dim;
        Cache { kv: (0..self.layers.len()).map(|_| gpu.vec(capacity * row)).collect(), len: 0, capacity }
    }

    /// `t` rows for each of `caches`' sequences (`x` `[caches.len() t, HIDDEN]`, a sequence's rows after another's,
    /// each at its cache's next positions, the same for all): every row's final, normed state into `out` (exactly
    /// that size), their keys and values into the caches. `table` (with RoPE) each row's (sin, cos) pairs.
    fn step(&self, gpu: &WgpuBackend, r: &mut dyn ChainRecorder, x: &DeviceVec, t: usize, caches: &mut [Cache], table: Option<&DeviceVec>, out: &DeviceVec) {
        let (h, hd, nq, nkv, ff, eps) = (HIDDEN, self.head_dim, self.heads, self.kv_heads, self.ff, self.eps);
        let (seqs, past) = (caches.len(), caches[0].len);
        let rows = seqs * t;
        assert!(caches.iter().all(|c| c.len == past && past + t <= c.capacity) && out.len == rows * h, "music on WebGPU: a step's caches");
        let (wq, wkv) = (nq * hd, nkv * hd);
        let w3 = wq + 2 * wkv;
        let scale = 1. / (hd as f32).sqrt();
        let v = |n: usize| gpu.vec(n);
        let (hs, n, qkv, q, k, vv, o) = (v(rows * h), v(rows * h), v(rows * w3), v(rows * wq), v(rows * wkv), v(rows * wkv), v(rows * h));
        let (qn, kn) = if self.layers[0].q_norm.is_some() { (v(rows * wq), v(rows * wkv)) } else { (q.clone(), k.clone()) };
        let att_len = gpu.attention_rows_out_len(t, nq, hd, past + t);
        let att = v(if seqs == 1 { att_len } else { rows * wq });
        // (a sequence's queries, keys and values on their own, where there are several)
        let (qs, ks, vs, atts) = if seqs > 1 { (v(t * wq), v(t * wkv), v(t * wkv), v(att_len)) } else { (q.clone(), k.clone(), vv.clone(), att.clone()) };
        let (gu, act) = (v(rows * 2 * ff), v(rows * ff));
        r.copy(x, 0, &hs, 0, rows * h);
        for (li, l) in self.layers.iter().enumerate() {
            r.rmsnorm_rows(&hs, &l.input_norm, &n, rows, eps);
            l.qkv.run(r, &n, &qkv, rows);
            r.copy_cols(&qkv, &q, rows, wq, w3, 0);
            r.copy_cols(&qkv, &k, rows, wkv, w3, wq);
            r.copy_cols(&qkv, &vv, rows, wkv, w3, wq + wkv);
            if let (Some(qw), Some(kw)) = (&l.q_norm, &l.k_norm) {
                r.rmsnorm_rows(&q, qw, &qn, rows * nq, eps);
                r.rmsnorm_rows(&k, kw, &kn, rows * nkv, eps);
            }
            if let (true, Some(table)) = (self.rope, table) {
                r.rope_rows(&qn, rows, nq, hd, table, true);
                r.rope_rows(&kn, rows, nkv, hd, table, true);
            }
            for (s, c) in caches.iter().enumerate() {
                let kv = &c.kv[li];
                if seqs == 1 {
                    r.store_rows(&kn, kv, t, wkv, past, 2 * wkv, 0);
                    r.store_rows(&vv, kv, t, wkv, past, 2 * wkv, wkv);
                    r.attention_rows(&qn, kv, &att, t, nq, nkv, hd, past, None, scale);
                } else {
                    r.copy(&kn, s * t * wkv, &ks, 0, t * wkv);
                    r.copy(&vv, s * t * wkv, &vs, 0, t * wkv);
                    r.store_rows(&ks, kv, t, wkv, past, 2 * wkv, 0);
                    r.store_rows(&vs, kv, t, wkv, past, 2 * wkv, wkv);
                    r.copy(&qn, s * t * wq, &qs, 0, t * wq);
                    r.attention_rows(&qs, kv, &atts, t, nq, nkv, hd, past, None, scale);
                    r.copy(&atts, 0, &att, s * t * wq, t * wq);
                }
            }
            l.o.run(r, &att, &o, rows);
            r.add(&hs, &o);
            r.rmsnorm_rows(&hs, &l.post_norm, &n, rows, eps);
            l.gate_up.run(r, &n, &gu, rows);
            r.silu_mul_split_rows(&gu, &act, rows);
            l.down.run(r, &act, &o, rows);
            r.add(&hs, &o);
        }
        r.rmsnorm_rows(&hs, &self.norm, out, rows, eps);
        for c in caches {
            c.len += t;
        }
    }
}

/// Where a prompt's embedding rows come from.
enum Text {
    Safe(Store),
    Gguf { content: gguf_file::Content, file: std::fs::File },
}

/// The language model on the device.
pub struct WgpuMusicLm {
    gpu: WgpuBackend,
    decoder: Decoder,
    /// Output rows the stage can emit: end of audio, then the 16384 codes.
    head: Mat,
    /// The semantic codes' input rows (`[16384, HIDDEN]`, on the host: a frame's feedback is summed there).
    codes: Vec<f32>,
    text: Text,
    theta: f64,
}

impl WgpuMusicLm {
    /// `dir`: the `language_model` folder; `quantized`: a file made by [`crate::music::quant::convert`] instead;
    /// `positions` the song's (its caches', for [`lm_f16`]). `progress(layers loaded)`.
    pub fn load(dir: &Path, quantized: Option<&Path>, positions: usize, gpu: &WgpuBackend, mut progress: impl FnMut(usize)) -> Result<Self> {
        let code_rows: Vec<u32> = (CODE_OFFSET..CODE_OFFSET + SEMANTIC_CODES as u32).collect();
        let head_rows: Vec<u32> = std::iter::once(AUDIO_END).chain(code_rows.iter().copied()).collect();
        let vector = |v: Vec<f32>| upload(gpu, &v);
        let (layers, norm, head, codes, text, (heads, kv_heads, head_dim, eps, theta)) = match quantized {
            None => {
                let config = lm::config_of(dir)?;
                let int = |k: &str| config.get(k).and_then(oaiy_engine::json::Json::as_i64).map(|v| v as usize).ok_or_else(|| err(format!("language model config lacks {k}")));
                let (count, heads, kv_heads, head_dim, ff) = (int("num_hidden_layers")?, int("num_attention_heads")?, int("num_key_value_heads")?, int("head_dim")?, int("intermediate_size")?);
                let eps = config.get("rms_norm_eps").and_then(oaiy_engine::json::Json::as_f64).unwrap_or(1e-6) as f32;
                let theta = config.get("rope_parameters").and_then(|r| r.get("rope_theta")).or_else(|| config.get("rope_theta")).and_then(oaiy_engine::json::Json::as_f64).unwrap_or(1e6);
                let mut store = Store::open(dir, 0)?;
                let s = &mut store;
                let bytes = |s: &mut Store, k: &str| -> Result<Vec<u8>> { s.bf16_bytes(k)?.ok_or_else(|| err(format!("{k}: not BF16"))) };
                let f32s = |s: &mut Store, k: &str| -> Result<Vec<f32>> { values(&s.tensor_f32(k, &Device::Cpu)?) };
                let w3 = (heads + 2 * kv_heads) * head_dim;
                let q8 = !lm_f16(gpu, positions);
                let mut layers = Vec::with_capacity(count);
                for i in 0..count {
                    let l = format!("model.layers.{i}");
                    let k = |n: &str| format!("{l}.{n}.weight");
                    let qkv = [bytes(s, &k("self_attn.q_proj"))?, bytes(s, &k("self_attn.k_proj"))?, bytes(s, &k("self_attn.v_proj"))?].concat();
                    let gate_up = [bytes(s, &k("mlp.gate_proj"))?, bytes(s, &k("mlp.up_proj"))?].concat();
                    layers.push(Layer {
                        input_norm: vector(f32s(s, &k("input_layernorm"))?),
                        qkv: Mat::bf16(gpu, &qkv, w3, HIDDEN, q8, &l)?,
                        q_norm: Some(vector(f32s(s, &k("self_attn.q_norm"))?)),
                        k_norm: Some(vector(f32s(s, &k("self_attn.k_norm"))?)),
                        o: Mat::bf16(gpu, &bytes(s, &k("self_attn.o_proj"))?, HIDDEN, heads * head_dim, q8, &l)?,
                        post_norm: vector(f32s(s, &k("post_attention_layernorm"))?),
                        gate_up: Mat::bf16(gpu, &gate_up, 2 * ff, HIDDEN, q8, &l)?,
                        down: Mat::bf16(gpu, &bytes(s, &k("mlp.down_proj"))?, HIDDEN, ff, q8, &l)?,
                    });
                    progress(i + 1);
                }
                let norm = vector(f32s(s, "model.norm.weight")?);
                let head = Mat::f16(gpu, &values(&s.rows("lm_head.weight", &head_rows, &Device::Cpu)?)?, head_rows.len(), HIDDEN, "lm_head")?;
                let codes = values(&s.rows("model.embed_tokens.weight", &code_rows, &Device::Cpu)?)?;
                (layers, norm, head, codes, Text::Safe(store), (heads, kv_heads, head_dim, eps, theta))
            }
            Some(path) => {
                let mut file = std::fs::File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
                let content = gguf_file::Content::read(&mut file).map_err(|e| err(format!("{}: {e}", path.display())))?;
                let u = |k: &str| content.metadata.get(k).and_then(|v| v.to_u32().ok()).map(|v| v as usize).ok_or_else(|| err(format!("{}: lacks {k}", path.display())));
                let f = |k: &str| content.metadata.get(k).and_then(|v| v.to_f32().ok()).ok_or_else(|| err(format!("{}: lacks {k}", path.display())));
                let (count, heads, kv_heads, head_dim, eps, theta) = (u("music3.layers")?, u("music3.heads")?, u("music3.kv_heads")?, u("music3.head_dim")?, f("music3.eps")?, f("music3.rope_theta")? as f64);
                let mut tensor = |name: &str| -> Result<candle_core::quantized::QTensor> { content.tensor(&mut file, name, &Device::Cpu) };
                let mut layers = Vec::with_capacity(count);
                for i in 0..count {
                    let l = format!("layers.{i}");
                    let mut small = |n: &str| -> Result<DeviceVec> { Ok(vector(values(&tensor(&format!("{l}.{n}"))?.dequantize(&Device::Cpu)?)?)) };
                    let (input_norm, q_norm, k_norm, post_norm) = (small("input_norm")?, small("q_norm")?, small("k_norm")?, small("post_norm")?);
                    layers.push(Layer {
                        input_norm,
                        qkv: Mat::gguf(gpu, &tensor(&format!("{l}.qkv"))?, &l)?,
                        q_norm: Some(q_norm),
                        k_norm: Some(k_norm),
                        o: Mat::gguf(gpu, &tensor(&format!("{l}.o"))?, &l)?,
                        post_norm,
                        gate_up: Mat::gguf(gpu, &tensor(&format!("{l}.gate_up"))?, &l)?,
                        down: Mat::gguf(gpu, &tensor(&format!("{l}.down"))?, &l)?,
                    });
                    progress(i + 1);
                }
                let norm = vector(values(&tensor("norm")?.dequantize(&Device::Cpu)?)?);
                let head = Mat::f16(gpu, &values(&tensor("head")?.dequantize(&Device::Cpu)?)?, head_rows.len(), HIDDEN, "head")?;
                let codes = values(&tensor("codes")?.dequantize(&Device::Cpu)?)?;
                (layers, norm, head, codes, Text::Gguf { content, file }, (heads, kv_heads, head_dim, eps, theta))
            }
        };
        let ff = match &layers[0].down {
            Mat::F16 { k, .. } => *k,
            Mat::Quant(q) => q.shape()[1],
        };
        if codes.len() != SEMANTIC_CODES * HIDDEN || head.n() != SEMANTIC_CODES + 1 {
            return Err(err("the language model's code rows are not 16384"));
        }
        Ok(Self { gpu: gpu.clone(), decoder: Decoder { layers, norm, heads, kv_heads, head_dim, ff, eps, rope: true }, head, codes, text, theta })
    }

    /// A prompt's embedding rows (`[ids, HIDDEN]`), read from the file row by row.
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        match &mut self.text {
            Text::Safe(store) => values(&store.rows("model.embed_tokens.weight", ids, &Device::Cpu)?),
            Text::Gguf { content, file } => {
                use std::io::{Read, Seek, SeekFrom};
                let info = content.tensor_infos.get("text").ok_or_else(|| err("the quantized language model lacks its text rows"))?;
                let (rows, width) = info.shape.dims2()?;
                let dtype = info.ggml_dtype;
                let row_bytes = width / dtype.block_size() * dtype.type_size();
                let mut bytes = vec![0u8; ids.len() * row_bytes];
                for (i, &id) in ids.iter().enumerate() {
                    if id as usize >= rows {
                        return Err(err(format!("token {id} past the quantized language model's text rows")));
                    }
                    file.seek(SeekFrom::Start(content.tensor_data_offset + info.offset + id as u64 * row_bytes as u64))?;
                    file.read_exact(&mut bytes[i * row_bytes..(i + 1) * row_bytes])?;
                }
                let q = candle_core::quantized::ggml_file::qtensor_from_ggml(dtype, &bytes, vec![ids.len(), width], &Device::Cpu)?;
                values(&q.dequantize(&Device::Cpu)?)
            }
        }
    }

    /// Rotate-half RoPE's table for `n` positions from `start`, each `repeat` times over (a step's rows: a sequence's,
    /// then the next's).
    fn table(&self, start: usize, n: usize, repeat: usize) -> Vec<f32> {
        let hd = self.decoder.head_dim;
        let one: Vec<f32> = (start..start + n)
            .flat_map(|p| (0..hd / 2).flat_map(move |i| {
                let a = p as f64 / self.theta.powf(2. * i as f64 / hd as f64);
                [a.sin() as f32, a.cos() as f32]
            }))
            .collect();
        one.repeat(repeat)
    }

    /// The semantic code's input row.
    fn code_row(&self, code: u32) -> &[f32] {
        &self.codes[code as usize * HIDDEN..(code as usize + 1) * HIDDEN]
    }
}

/// The depth decoder on the device.
pub struct WgpuDepth {
    decoder: Decoder,
    projection: Mat,
    heads_out: Vec<Mat>,
    /// The residual codebooks' input rows (`[7 * 1024, HIDDEN]`) and the learned positions (`[16, HIDDEN]`), on the
    /// host.
    audio: Vec<f32>,
    positions: Vec<f32>,
}

impl WgpuDepth {
    /// The `rvq_depth_decoder` folder onto `gpu`.
    pub fn load(dir: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let config = crate::music::acoustic::read_config(&dir.join("config.json"))?;
        let int = |k: &str, d: usize| config.get(k).and_then(oaiy_engine::json::Json::as_i64).map_or(d, |v| v as usize);
        let (count, heads, ff) = (int("num_layers", 4), int("num_attention_heads", 16), int("intermediate_size", 6144));
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let s = &mut store;
        let bytes = |s: &mut Store, k: &str| -> Result<Vec<u8>> { s.bf16_bytes(k)?.ok_or_else(|| err(format!("{k}: not BF16"))) };
        let f32s = |s: &mut Store, k: &str| -> Result<Vec<f32>> { values(&s.tensor_f32(k, &Device::Cpu)?) };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let k = |n: &str| format!("layers.{i}.{n}.weight");
            let qkv = [bytes(s, &k("attn.to_q"))?, bytes(s, &k("attn.to_k"))?, bytes(s, &k("attn.to_v"))?].concat();
            let gate_up = [bytes(s, &k("gate_proj"))?, bytes(s, &k("up_proj"))?].concat();
            layers.push(Layer {
                input_norm: upload(gpu, &f32s(s, &k("input_layernorm"))?),
                qkv: Mat::bf16(gpu, &qkv, 3 * HIDDEN, HIDDEN, false, &k("attn"))?,
                q_norm: None,
                k_norm: None,
                o: Mat::bf16(gpu, &bytes(s, &k("attn.to_out"))?, HIDDEN, HIDDEN, false, &k("attn.to_out"))?,
                post_norm: upload(gpu, &f32s(s, &k("post_attention_layernorm"))?),
                gate_up: Mat::bf16(gpu, &gate_up, 2 * ff, HIDDEN, false, &k("gate_up"))?,
                down: Mat::bf16(gpu, &bytes(s, &k("down_proj"))?, HIDDEN, ff, false, &k("down_proj"))?,
            });
        }
        let heads_out = (0..CODEBOOKS - 1).map(|i| Mat::f16(gpu, &f32s(s, &format!("audio_heads.{i}.weight"))?, AUDIO_VOCAB, HIDDEN, "audio_heads")).collect::<Result<_>>()?;
        Ok(Self {
            decoder: Decoder { layers, norm: upload(gpu, &f32s(s, "norm.weight")?), heads, kv_heads: heads, head_dim: HIDDEN / heads, ff, eps: 1e-6, rope: false },
            projection: Mat::f16(gpu, &f32s(s, "projection.weight")?, HIDDEN, HIDDEN, "projection")?,
            heads_out,
            audio: f32s(s, "audio_embeddings.weight")?,
            positions: f32s(s, "pos_embedding.weight")?,
        })
    }

    /// Codebook `index`'s (1..7) input row for `code`.
    fn audio_row(&self, code: u32, index: usize) -> &[f32] {
        let id = code as usize + (index - 1) * AUDIO_VOCAB;
        &self.audio[id * HIDDEN..(id + 1) * HIDDEN]
    }

    /// The feedback input for a finished frame: every codebook's input row summed, scaled by 8^-1/2.
    fn feedback(&self, semantic: &[f32], residual: &[u32]) -> Vec<f32> {
        let mut sum = semantic.to_vec();
        for (i, &c) in residual.iter().enumerate() {
            for (a, b) in sum.iter_mut().zip(self.audio_row(c, i + 1)) {
                *a += b;
            }
        }
        let scale = (CODEBOOKS as f64).powf(-0.5) as f32;
        sum.iter().map(|v| v * scale).collect()
    }

    /// A frame's seven residual codes from the language model's states `last` (`[2, HIDDEN]`: the conditional row's,
    /// then the unconditional's) and its semantic code's input row; `pick(index, guided logits)` chooses each. The
    /// codes, and the conditional row's depth states (`[7 HIDDEN]`).
    fn codes(&self, gpu: &WgpuBackend, last: &DeviceVec, semantic_row: &[f32], semantic: u32, pick: &mut dyn FnMut(usize, &[f32]) -> u32) -> Result<([u32; CODEBOOKS], Vec<f32>)> {
        let h = HIDDEN;
        let mut caches = vec![self.decoder.cache(gpu, CODEBOOKS), self.decoder.cache(gpu, CODEBOOKS)];
        let mut codes = [0u32; CODEBOOKS];
        codes[0] = semantic;
        let mut hidden = Vec::with_capacity((CODEBOOKS - 1) * h);
        let sem = upload(gpu, semantic_row);
        for index in 1..CODEBOOKS {
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            let (t, x) = if index == 1 {
                // positions 0 and 1: the language model's state, then the semantic code's, each projected
                let (pl, ps, x) = (gpu.vec(2 * h), gpu.vec(h), gpu.vec(4 * h));
                self.projection.run(r, last, &pl, 2);
                self.projection.run(r, &sem, &ps, 1);
                r.copy(&pl, 0, &x, 0, h);
                r.copy(&ps, 0, &x, h, h);
                r.copy(&pl, h, &x, 2 * h, h);
                r.copy(&ps, 0, &x, 3 * h, h);
                (2, x)
            } else {
                // position `index`: the last code's input row projected, both rows the same
                let (e, pe, x) = (upload(gpu, self.audio_row(codes[index - 1], index - 1)), gpu.vec(h), gpu.vec(2 * h));
                self.projection.run(r, &e, &pe, 1);
                r.copy(&pe, 0, &x, 0, h);
                r.copy(&pe, 0, &x, h, h);
                (1, x)
            };
            let past = caches[0].len;
            let pos: Vec<f32> = (past..past + t).flat_map(|p| self.positions[p * h..(p + 1) * h].iter().copied()).collect::<Vec<_>>().repeat(2);
            r.add(&x, &upload(gpu, &pos));
            let out = gpu.vec(2 * t * h);
            self.decoder.step(gpu, r, &x, t, &mut caches, None, &out);
            // each row's last position
            let (hl, logits) = (gpu.vec(2 * h), gpu.vec(2 * AUDIO_VOCAB));
            r.copy(&out, (t - 1) * h, &hl, 0, h);
            r.copy(&out, (2 * t - 1) * h, &hl, h, h);
            self.heads_out[index - 1].run(r, &hl, &logits, 2);
            r.read(&logits);
            r.read_range(&hl, 0, h);
            let mut got = rec.finish();
            let (state, l) = (got.pop().ok_or_else(|| err("a depth state was not read"))?, got.pop().ok_or_else(|| err("depth logits were not read"))?);
            hidden.extend_from_slice(&state);
            codes[index] = pick(index, &lm::guide(&l[..AUDIO_VOCAB], &l[AUDIO_VOCAB..]));
        }
        Ok((codes, hidden))
    }
}

/// A frame: its codes (semantic, then the seven residual), and the conditional row's eight hidden states (`[8
/// HIDDEN]`: the language model's, then the depth decoder's).
pub struct Frame {
    pub codes: [u32; CODEBOOKS],
    pub hidden: Vec<f32>,
}

/// Up to `max_frames` frames for the prompt pair `ids` (conditional, unconditional), as [`crate::music::lm::generate`]
/// writes them; `progress(frames)` after each.
pub fn generate(lm: &mut WgpuMusicLm, depth: &WgpuDepth, ids: &[Vec<u32>; 2], max_frames: usize, mut rng: Option<&mut Rng>, mut progress: impl FnMut(usize) -> Result<()>) -> Result<Vec<Frame>> {
    let g = lm.gpu.clone();
    let h = HIDDEN;
    if ids[0].len() != ids[1].len() || ids[0].is_empty() {
        return Err(err("the two prompts differ in length"));
    }
    let n = ids[0].len();
    let mut caches = vec![lm.decoder.cache(&g, n + max_frames + 2), lm.decoder.cache(&g, n + max_frames + 2)];
    // each prompt's prefill, its last state kept
    let last = g.vec(2 * h);
    let table = upload(&g, &lm.table(0, n, 1));
    for (s, ids) in ids.iter().enumerate() {
        let x = upload(&g, &lm.embed(ids)?);
        let out = g.vec(n * h);
        let mut rec = g.begin();
        rec.keep_groups(false);
        lm.decoder.step(&g, rec.as_mut(), &x, n, std::slice::from_mut(&mut caches[s]), Some(&table), &out);
        rec.as_mut().copy(&out, (n - 1) * h, &last, s * h, h);
        rec.finish();
    }
    let v = lm.head.n();
    let mut frames = Vec::new();
    // The first step only moves past <|audio_start|>; its frame is fed back but not kept.
    for index in 0..=max_frames {
        let logits = g.vec(2 * v);
        let mut rec = g.begin();
        lm.head.run(rec.as_mut(), &last, &logits, 2);
        rec.as_mut().read(&logits);
        rec.as_mut().read_range(&last, 0, h);
        let mut got = rec.finish();
        let (state, l) = (got.pop().ok_or_else(|| err("a state was not read"))?, got.pop().ok_or_else(|| err("logits were not read"))?);
        let Some(semantic) = lm::pick_semantic_rows(&l[..v], &l[v..], rng.as_deref_mut()) else { break };
        let (codes, depth_hidden) = depth.codes(&g, &last, lm.code_row(semantic), semantic, &mut |_, guided| lm::sample_top_k(guided, rng.as_deref_mut()) as u32)?;
        if index > 0 {
            frames.push(Frame { codes, hidden: [state, depth_hidden].concat() });
            progress(frames.len())?;
            if frames.len() >= max_frames {
                break;
            }
        }
        let feedback = depth.feedback(lm.code_row(semantic), &codes[1..]);
        let x = upload(&g, &feedback.repeat(2));
        let table = upload(&g, &lm.table(caches[0].len, 1, 2));
        let out = g.vec(2 * h);
        let mut rec = g.begin();
        rec.keep_groups(false);
        lm.decoder.step(&g, rec.as_mut(), &x, 1, &mut caches, Some(&table), &out);
        rec.as_mut().copy(&out, 0, &last, 0, 2 * h);
        rec.finish();
    }
    Ok(frames)
}

#[cfg(test)]
mod golden {
    use super::*;
    use std::path::PathBuf;

    fn dirs() -> (PathBuf, PathBuf) {
        (
            PathBuf::from(std::env::var("OAIY_MUSIC_GOLDEN").unwrap_or_else(|_| "E:/deepseek/nrob/target/music3-golden".into())),
            PathBuf::from(std::env::var("OAIY_MUSIC_MODEL").unwrap_or_else(|_| "E:/models/MiniMax-Music3".into())),
        )
    }

    fn load(dir: &Path, name: &str) -> Vec<f32> {
        std::fs::read(dir.join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn relative(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        (a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>()).sqrt()
    }

    /// The autoregressive stage on WebGPU against the official pipeline's activations (its language model and depth
    /// decoder BF16; `--ignored --nocapture`; OAIY_MUSIC_GOLDEN, OAIY_MUSIC_MODEL as the acoustic stage's test, and
    /// OAIY_MUSIC_LM a quantized file for the safetensors): the depth decoder on the reference's own inputs, the
    /// prefill's last states and guided logits, then 40 frames with the reference's codes fed back, their hidden states
    /// and how often our greedy pick is the reference's.
    #[test]
    #[ignore = "needs MiniMax-Music3 and the reference's dumps"]
    fn the_webgpu_language_model_is_the_references() -> Result<()> {
        let (g, m) = dirs();
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let p = oaiy_engine::json::Json::parse(&std::fs::read(g.join("prompt.json"))?).map_err(candle_core::Error::wrap)?;
        let s = |k: &str| p.get(k).and_then(oaiy_engine::json::Json::as_str).unwrap().to_string();
        let ids = crate::music::prompt_ids(&m, &s("prompt"), &s("lyrics"))?;
        let quantized = std::env::var("OAIY_MUSIC_LM").ok().map(PathBuf::from);
        let started = std::time::Instant::now();
        let mut lm = WgpuMusicLm::load(&m.join("language_model"), quantized.as_deref(), ids[0].len() + 64, &gpu, |_| {})?;
        let depth = WgpuDepth::load(&m.join("rvq_depth_decoder"), &gpu)?;
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());
        let h = HIDDEN;
        // the depth decoder on the reference's inputs (its positions added as its forward adds them)
        for i in 0..7 {
            let mut x = load(&g, &format!("depth_in{i}.f32"));
            let t = x.len() / (2 * h);
            for (r, row) in x.chunks_exact_mut(h).enumerate() {
                for (a, b) in row.iter_mut().zip(&depth.positions[(r % t) * h..(r % t + 1) * h]) {
                    *a += b;
                }
            }
            let mut caches = vec![depth.decoder.cache(&gpu, CODEBOOKS), depth.decoder.cache(&gpu, CODEBOOKS)];
            let (xd, out) = (upload(&gpu, &x), gpu.vec(2 * t * h));
            let mut rec = gpu.begin();
            depth.decoder.step(&gpu, rec.as_mut(), &xd, t, &mut caches, None, &out);
            rec.as_mut().read(&out);
            let y = rec.finish().pop().unwrap();
            eprintln!("depth step {i}: {:.2e}", relative(&y, &load(&g, &format!("depth_out{i}.f32"))));
        }
        // the prefill
        let n = ids[0].len();
        let mut caches = vec![lm.decoder.cache(&gpu, n + 64), lm.decoder.cache(&gpu, n + 64)];
        let last = gpu.vec(2 * h);
        let table = upload(&gpu, &lm.table(0, n, 1));
        for (si, ids) in ids.iter().enumerate() {
            let (x, out) = (upload(&gpu, &lm.embed(ids)?), gpu.vec(n * h));
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            lm.decoder.step(&gpu, rec.as_mut(), &x, n, std::slice::from_mut(&mut caches[si]), Some(&table), &out);
            rec.as_mut().copy(&out, (n - 1) * h, &last, si * h, h);
            rec.finish();
        }
        let v = lm.head.n();
        let logits_of = |lm: &WgpuMusicLm, last: &DeviceVec| -> (Vec<f32>, Vec<f32>) {
            let logits = gpu.vec(2 * v);
            let mut rec = gpu.begin();
            lm.head.run(rec.as_mut(), last, &logits, 2);
            rec.as_mut().read(&logits);
            rec.as_mut().read(last);
            let mut got = rec.finish();
            let state = got.pop().unwrap();
            (got.pop().unwrap(), state)
        };
        let (l, state) = logits_of(&lm, &last);
        eprintln!("prefill states: {:.2e}", relative(&state, &load(&g, "lm_hidden.f32")[..2 * h]));
        let guided = lm::guide(&l[..v], &l[v..]);
        let reference = load(&g, "lm_logits.f32");
        let expected: Vec<f32> = std::iter::once(AUDIO_END as usize).chain(CODE_OFFSET as usize..CODE_OFFSET as usize + SEMANTIC_CODES).map(|i| reference[i]).collect();
        let finite: Vec<usize> = (0..expected.len()).filter(|&i| expected[i].is_finite()).collect();
        let (a, e): (Vec<f32>, Vec<f32>) = finite.iter().map(|&i| (guided[i], expected[i])).unzip();
        eprintln!("guided logits ({} candidates): {:.2e}", finite.len(), relative(&a, &e));
        // the reference's codes fed back: each frame's hidden states, and our greedy picks against its
        let codes: Vec<u32> = std::fs::read(g.join("codes.i32"))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect();
        let (c0, c1) = (load(&g, "cond_in0.f32"), load(&g, "cond_in1.f32"));
        let reference_hidden: Vec<f32> = c0.iter().chain(&c1[100 * 8 * h..]).copied().collect();
        let (mut agree, mut total, mut worst_margin, mut worst) = (0, 0, 0f32, 0f64);
        let started = std::time::Instant::now();
        let forced = 40;
        for f in 0..=forced {
            let want = &codes[f * 8..f * 8 + 8];
            let (l, state) = logits_of(&lm, &last);
            total += 1;
            let ours = lm::pick_semantic_rows(&l[..v], &l[v..], None);
            if ours == Some(want[0]) {
                agree += 1;
            } else {
                let mut sorted = l[..v].to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let threshold = sorted[lm::TOP_K - 1];
                let gd: Vec<f32> = lm::guide(&l[..v], &l[v..]).into_iter().zip(&l[..v]).map(|(g, &c)| if c < threshold { f32::NEG_INFINITY } else { g }).collect();
                let best = gd.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                worst_margin = worst_margin.max(best - gd[want[0] as usize + 1]);
            }
            let (got, depth_hidden) = depth.codes(&gpu, &last, lm.code_row(want[0]), want[0], &mut |index, guided| {
                let ours = lm::sample_top_k(guided, None) as u32;
                total += 1;
                if ours == want[index] {
                    agree += 1
                } else {
                    worst_margin = worst_margin.max(guided[ours as usize] - guided[want[index] as usize])
                }
                want[index]
            })?;
            assert_eq!(&got[..], want);
            if f > 0 {
                let hidden = [&state[..h], &depth_hidden[..]].concat();
                worst = worst.max(relative(&hidden, &reference_hidden[(f - 1) * 8 * h..f * 8 * h]));
            }
            let feedback = depth.feedback(lm.code_row(want[0]), &want[1..]);
            let (x, out) = (upload(&gpu, &feedback.repeat(2)), gpu.vec(2 * h));
            let table = upload(&gpu, &lm.table(caches[0].len, 1, 2));
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            lm.decoder.step(&gpu, rec.as_mut(), &x, 1, &mut caches, Some(&table), &out);
            rec.as_mut().copy(&out, 0, &last, 0, 2 * h);
            rec.finish();
        }
        eprintln!(
            "{forced} frames teacher-forced in {:.2} s: the worst frame's hidden states {worst:.2e}; {agree} of {total} greedy picks the reference's, the largest margin where not {worst_margin:.3}",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }
}
