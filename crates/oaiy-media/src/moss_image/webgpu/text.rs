//! MOSS-Image's text conditioning on WebGPU, as `MOSSImagePipeline.encode_prompt` makes it: the prompt (trimmed) in
//! Qwen3.5's chat template, one user turn and the assistant's opened with thinking on, at most 4,096 tokens, through
//! Qwen3.5-9B's language model (its vision tower unused); the last hidden state after the final norm, `[n, 4096]`, each
//! prompt its own length (no padding). Its 24 gated delta net layers and 8 of attention run on the ops the chat
//! engine's Qwen3.5 chain runs on (the depthwise conv and the recurrence over the prompt's rows, the attention's
//! sigmoid gate), a layer at a time on the GPU with every prompt through it: the 9B's 18 GB are never all there.
//!
//! A checkpoint as Hugging Face keeps it (the stock `Qwen/Qwen3.5-9B`, or MOSS-Image's `text_encoder` folder: the same
//! weights, its few F32 ones as BF16) is read as llama.cpp's converter makes a GGUF of it: each delta net's value heads
//! from grouped (a key head's values side by side) to tiled order, `-exp(A_log)`, `1 + w` of every RMS norm but the
//! delta nets' own. A GGUF of it (llama.cpp's names) is read as it is, its K-quant blocks kept; its own tokenizer where
//! no tokenizer.json is given. The matmuls of a checkpoint's f16 weights read their inputs as f32.
use crate::common::weights::{Raw, Weights};
use crate::common::webgpu_weights::{f16_words, f16_words_f32, f16_words_ggml};
use crate::qwen_image::webgpu::transformer::{coop_type, err, first};
use candle_core::{DType, Device, Result};
use ggml_rs::{Backend, ChainRecorder, DeltaNet, DeviceChain, DeviceVec, QuantizedTensor};
use std::path::Path;

const D: usize = 4096;
const VOCAB: usize = 248_320;
const HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HD: usize = 256;
/// The rotated part of an attention head (`partial_rotary_factor` 0.25), and RoPE's base.
const ROT: usize = 64;
const THETA: f32 = 10_000_000.;
const FF: usize = 12_288;
/// A delta net's key heads and value heads, each 128 wide, and its conv's channels (q, k and v) and width.
const NK: usize = 16;
const NV: usize = 32;
const DK: usize = 128;
const CH: usize = 2 * NK * DK + NV * DK;
const KERN: usize = 4;
const EPS: f32 = 1e-6;
/// The pipeline's `max_sequence_length`: a prompt's tokens past it are cut.
pub const MAX_TOKENS: usize = 4096;

/// The prompt as the pipeline gives it to the tokenizer: Qwen3.5's chat template of one user turn (its text trimmed
/// as the template's `|trim` does), the assistant's turn opened with thinking on (`enable_thinking=True`).
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", prompt.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)))
}

/// Which names the checkpoint's tensors go by: Hugging Face's under a prefix, or llama.cpp's in a GGUF.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Names {
    Hf(&'static str),
    Gguf,
}

/// A layer's tensors, by what they are.
#[derive(Clone, Copy)]
enum Part {
    InputNorm,
    PostNorm,
    Qkv,
    Z,
    Beta,
    Alpha,
    A,
    DtBias,
    Conv,
    SsmNorm,
    Out,
    Q,
    K,
    V,
    O,
    QNorm,
    KNorm,
    Gate,
    Up,
    Down,
}

impl Names {
    fn key(self, i: usize, part: Part) -> String {
        match self {
            Names::Hf(p) => {
                let name = match part {
                    Part::InputNorm => "input_layernorm.weight",
                    Part::PostNorm => "post_attention_layernorm.weight",
                    Part::Qkv => "linear_attn.in_proj_qkv.weight",
                    Part::Z => "linear_attn.in_proj_z.weight",
                    Part::Beta => "linear_attn.in_proj_b.weight",
                    Part::Alpha => "linear_attn.in_proj_a.weight",
                    Part::A => "linear_attn.A_log",
                    Part::DtBias => "linear_attn.dt_bias",
                    Part::Conv => "linear_attn.conv1d.weight",
                    Part::SsmNorm => "linear_attn.norm.weight",
                    Part::Out => "linear_attn.out_proj.weight",
                    Part::Q => "self_attn.q_proj.weight",
                    Part::K => "self_attn.k_proj.weight",
                    Part::V => "self_attn.v_proj.weight",
                    Part::O => "self_attn.o_proj.weight",
                    Part::QNorm => "self_attn.q_norm.weight",
                    Part::KNorm => "self_attn.k_norm.weight",
                    Part::Gate => "mlp.gate_proj.weight",
                    Part::Up => "mlp.up_proj.weight",
                    Part::Down => "mlp.down_proj.weight",
                };
                format!("{p}layers.{i}.{name}")
            }
            Names::Gguf => {
                let name = match part {
                    Part::InputNorm => "attn_norm.weight",
                    Part::PostNorm => "post_attention_norm.weight",
                    Part::Qkv => "attn_qkv.weight",
                    Part::Z => "attn_gate.weight",
                    Part::Beta => "ssm_beta.weight",
                    Part::Alpha => "ssm_alpha.weight",
                    Part::A => "ssm_a",
                    Part::DtBias => "ssm_dt.bias",
                    Part::Conv => "ssm_conv1d.weight",
                    Part::SsmNorm => "ssm_norm.weight",
                    Part::Out => "ssm_out.weight",
                    Part::Q => "attn_q.weight",
                    Part::K => "attn_k.weight",
                    Part::V => "attn_v.weight",
                    Part::O => "attn_output.weight",
                    Part::QNorm => "attn_q_norm.weight",
                    Part::KNorm => "attn_k_norm.weight",
                    Part::Gate => "ffn_gate.weight",
                    Part::Up => "ffn_up.weight",
                    Part::Down => "ffn_down.weight",
                };
                format!("blk.{i}.{name}")
            }
        }
    }

    fn embedding(self) -> String {
        match self {
            Names::Hf(p) => format!("{p}embed_tokens.weight"),
            Names::Gguf => "token_embd.weight".into(),
        }
    }

    fn final_norm(self) -> String {
        match self {
            Names::Hf(p) => format!("{p}norm.weight"),
            Names::Gguf => "output_norm.weight".into(),
        }
    }
}

/// The tiled value head `n` is (llama.cpp's order: `ggml_repeat` takes key head `n % NK` for it) as Hugging Face's
/// grouped one (`repeat_interleave`: key head `o / (NV / NK)`) holds it.
fn grouped(n: usize) -> usize {
    (n % NK) * (NV / NK) + n / NK
}

/// How a Hugging Face tensor's rows (or columns) are reordered to llama.cpp's tiled value heads.
#[derive(Clone, Copy, PartialEq)]
enum Order {
    Same,
    /// Its rows from `from`, value heads `head` rows each.
    Rows { from: usize, head: usize },
    /// Its columns, value heads `head` columns each.
    Cols { head: usize },
}

/// `values` (`[n, k]` of `size`-byte elements) reordered as `order` says.
fn reorder<T: Copy>(values: &mut [T], n: usize, k: usize, order: Order) {
    match order {
        Order::Same => {}
        Order::Rows { from, head } => {
            let part = values[from * k..n * k].to_vec();
            for v in 0..NV {
                let o = grouped(v);
                values[(from + v * head) * k..(from + (v + 1) * head) * k].copy_from_slice(&part[o * head * k..(o + 1) * head * k]);
            }
        }
        Order::Cols { head } => {
            for row in values.chunks_exact_mut(k) {
                let old = row.to_vec();
                for v in 0..NV {
                    let o = grouped(v);
                    row[v * head..(v + 1) * head].copy_from_slice(&old[o * head..(o + 1) * head]);
                }
            }
        }
    }
}

/// A matrix on the GPU (`[n, k]`): a checkpoint's as f16 (its inputs read as f32), a GGUF's K-quant blocks as they
/// are, or (a delta net's beta and alpha) f32.
enum W {
    F16(DeviceVec),
    Quant(QuantizedTensor),
    F32(DeviceVec),
}

struct Mat {
    w: W,
    n: usize,
    k: usize,
}

impl Mat {
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match &self.w {
            W::F16(v) => r.matmul_f16_rows_f32(v, self.n, self.k, x, y, rows),
            W::Quant(q) => r.matmul_rows(q, x, y, rows),
            W::F32(v) => r.matmul_f32_rows(v, self.n, self.k, x, y, rows),
        }
    }
}

enum Mixer {
    Delta { qkv: Mat, z: Mat, beta_alpha: Mat, conv: DeviceVec, a: DeviceVec, dt: DeviceVec, norm: DeviceVec, out: Mat },
    Attention { q: Mat, k: Mat, v: Mat, o: Mat, q_norm: DeviceVec, k_norm: DeviceVec },
}

struct Layer {
    input: DeviceVec,
    post: DeviceVec,
    mixer: Mixer,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// The prompt's tokens: by a tokenizer.json, or by the GGUF's own tokenizer.
enum Tok {
    Json(tokenizers::Tokenizer),
    Gguf(tokenizer::Tokenizer),
}

/// A prompt's vectors through the layers: its states (`x`), the rest scratch for its `t` rows.
struct Work {
    t: usize,
    x: DeviceVec,
    xn: DeviceVec,
    table: DeviceVec,
    qkv: DeviceVec,
    z: DeviceVec,
    ba: DeviceVec,
    conv: DeviceVec,
    core: DeviceVec,
    conv_state: DeviceVec,
    state: DeviceVec,
    qfull: DeviceVec,
    q: DeviceVec,
    gate: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    gated: DeviceVec,
    proj: DeviceVec,
    g: DeviceVec,
    u: DeviceVec,
    act: DeviceVec,
}

pub struct WgpuMossText {
    gpu: ggml_rs_wgpu::WgpuBackend,
    w: Weights,
    names: Names,
    layers: usize,
    tok: Tok,
}

impl WgpuMossText {
    /// Qwen3.5-9B at `checkpoint` (a folder of safetensors, one safetensors file, or a GGUF) with `tokenizer` (a
    /// tokenizer.json; a GGUF's own where none), on GPU `device` (as the computer counts them; OAIY_WEBGPU_ADAPTER
    /// naming one instead).
    pub fn load(checkpoint: &Path, tokenizer: Option<&Path>, device: usize) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let w = Weights::open(checkpoint)?;
        let names = if checkpoint.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            Names::Gguf
        } else {
            ["model.language_model.", "language_model.", "model.", ""]
                .into_iter()
                .find(|p| w.has(&format!("{p}embed_tokens.weight")))
                .map(Names::Hf)
                .ok_or_else(|| err("MOSS-Image requires the Qwen3.5-9B text encoder: the checkpoint has no embed_tokens"))?
        };
        let what = "MOSS-Image requires the Qwen3.5-9B text encoder";
        let layers = (0..).take_while(|&i| w.has(&names.key(i, Part::InputNorm))).count();
        let embedding = w.shape(&names.embedding())?;
        let delta = names.key(0, Part::Qkv);
        if embedding != [VOCAB, D] || layers == 0 || !w.has(&delta) || w.shape(&delta)? != [CH, D] || w.shape(&names.key(0, Part::Gate))? != [FF, D] {
            candle_core::bail!("{what} (embedding {embedding:?}, {layers} layers)");
        }
        let tok = match tokenizer {
            Some(path) => {
                let mut t = tokenizers::Tokenizer::from_file(path).map_err(err)?;
                t.with_padding(None);
                t.with_truncation(None).map_err(err)?;
                for (token, id) in [("<|im_start|>", 248_045), ("<|im_end|>", 248_046), ("<think>", 248_068)] {
                    if t.token_to_id(token) != Some(id) {
                        candle_core::bail!("not Qwen3.5's tokenizer: {token} must have id {id}");
                    }
                }
                Tok::Json(t)
            }
            None if names == Names::Gguf => Tok::Gguf(tokenizer::Tokenizer::from_gguf(&gguf::GgufFile::open(checkpoint).map_err(err)?).map_err(err)?),
            None => candle_core::bail!("MOSS-Image's text encoder needs Qwen3.5's tokenizer.json"),
        };
        Ok(Self { gpu, w, names, layers, tok })
    }

    /// The prompt's tokens as the pipeline makes them: its chat template, at most [`MAX_TOKENS`].
    pub fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let text = template(prompt);
        let mut ids = match &self.tok {
            Tok::Json(t) => t.encode(text, false).map_err(err)?.get_ids().to_vec(),
            Tok::Gguf(t) => t.encode(&text, false).map_err(err)?,
        };
        ids.truncate(MAX_TOKENS);
        Ok(ids)
    }

    /// Each prompt's final states (`[n, 4096]`, a token a row) and its token count.
    pub fn encode_all(&mut self, prompts: &[&str]) -> Result<Vec<(Vec<f32>, usize)>> {
        let ids = prompts.iter().map(|p| self.tokens(p)).collect::<Result<Vec<_>>>()?;
        let layers = self.layers;
        let states = self.encode_ids(&ids, layers)?;
        Ok(states.into_iter().zip(&ids).map(|(s, i)| (s, i.len())).collect())
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// A matrix `[n, k]` onto the GPU, a Hugging Face one's rows or columns reordered as `order` says: a GGUF's
    /// K-quant blocks as they are, anything else as f16.
    fn matrix(&mut self, key: &str, order: Order) -> Result<Mat> {
        let shape = self.w.shape(key)?;
        let &[n, k] = shape.as_slice() else { candle_core::bail!("{key}: a matrix of shape {shape:?}") };
        let order = if self.names == Names::Gguf { Order::Same } else { order };
        if let (Some(g), 0) = (self.w.ggml_dtype(key).and_then(coop_type), k % 256) {
            if let Some(Raw::Ggml(_, bytes)) = self.w.raw(key)? {
                let q = self.gpu.to_device_quant(QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], g));
                if !q.is_device() {
                    candle_core::bail!("{key}: no room on the GPU");
                }
                return Ok(Mat { w: W::Quant(q), n, k });
            }
        }
        let words = match self.w.raw(key)? {
            Some(Raw::Bf16(bytes)) => {
                let mut halves: Vec<[u8; 2]> = bytes.chunks_exact(2).map(|b| [b[0], b[1]]).collect();
                reorder(&mut halves, n, k, order);
                f16_words(&halves.concat())
            }
            Some(Raw::Ggml(t, bytes)) => f16_words_ggml(t, &bytes, n, k),
            None => {
                let mut values = self.w.tensor(key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                reorder(&mut values, n, k, order);
                f16_words_f32(&values)
            }
        }
        .ok_or_else(|| err(format!("{key}: past f16's range")))?;
        let v = self.vec(words.len());
        self.gpu.upload(&v, &words);
        Ok(Mat { w: W::F16(v), n, k })
    }

    /// A tensor's values on the host as f32, a Hugging Face one's rows reordered as `order` says (its rows `from + NV
    /// head`, whatever their length: a conv's weights four a channel, a head's gate one).
    fn values(&mut self, key: &str, order: Order) -> Result<Vec<f32>> {
        let mut v = self.w.tensor(key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if let (Names::Hf(_), Order::Rows { from, head }) = (self.names, order) {
            let n = from + NV * head;
            if v.len() % n != 0 {
                candle_core::bail!("{key}: {} values are not {n} rows", v.len());
            }
            let k = v.len() / n;
            reorder(&mut v, n, k, order);
        }
        Ok(v)
    }

    fn upload(&self, values: &[f32]) -> DeviceVec {
        let v = self.vec(values.len());
        self.gpu.upload(&v, values);
        v
    }

    /// An RMS norm's weight as it multiplies: Hugging Face's `1 + w` (Qwen3.5's), a GGUF's as it is.
    fn norm(&mut self, key: &str) -> Result<DeviceVec> {
        let add = if self.names == Names::Gguf { 0. } else { 1. };
        let v: Vec<f32> = self.w.tensor(key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| v + add).collect();
        Ok(self.upload(&v))
    }

    /// Layer `i`'s weights on the GPU.
    fn layer(&mut self, i: usize) -> Result<Layer> {
        let names = self.names;
        let key = |p| names.key(i, p);
        let attention = self.w.has(&key(Part::Q));
        let mixer = if attention {
            let q = self.matrix(&key(Part::Q), Order::Same)?;
            let (k, v, o) = (self.matrix(&key(Part::K), Order::Same)?, self.matrix(&key(Part::V), Order::Same)?, self.matrix(&key(Part::O), Order::Same)?);
            if q.n != HEADS * 2 * HD || k.n != KV_HEADS * HD || o.n != D {
                candle_core::bail!("MOSS-Image requires the Qwen3.5-9B text encoder (layer {i}'s attention)");
            }
            Mixer::Attention { q, k, v, o, q_norm: self.norm(&key(Part::QNorm))?, k_norm: self.norm(&key(Part::KNorm))? }
        } else {
            let qkv = self.matrix(&key(Part::Qkv), Order::Rows { from: 2 * NK * DK, head: DK })?;
            let z = self.matrix(&key(Part::Z), Order::Rows { from: 0, head: DK })?;
            let out = self.matrix(&key(Part::Out), Order::Cols { head: DK })?;
            // beta's rows then alpha's, f32 (as the chat engine keeps them: they gate the recurrence)
            let mut beta_alpha = self.values(&key(Part::Beta), Order::Rows { from: 0, head: 1 })?;
            beta_alpha.extend(self.values(&key(Part::Alpha), Order::Rows { from: 0, head: 1 })?);
            let mut a = self.values(&key(Part::A), Order::Rows { from: 0, head: 1 })?;
            if self.names != Names::Gguf {
                a.iter_mut().for_each(|v| *v = -v.exp());
            }
            let dt = self.values(&key(Part::DtBias), Order::Rows { from: 0, head: 1 })?;
            let conv = self.values(&key(Part::Conv), Order::Rows { from: 2 * NK * DK, head: DK })?;
            let norm = self.values(&key(Part::SsmNorm), Order::Same)?;
            if beta_alpha.len() != 2 * NV * D || a.len() != NV || dt.len() != NV || conv.len() != CH * KERN || norm.len() != DK || z.n != NV * DK || out.k != NV * DK {
                candle_core::bail!("MOSS-Image requires the Qwen3.5-9B text encoder (layer {i}'s delta net)");
            }
            let beta_alpha = Mat { w: W::F32(self.upload(&beta_alpha)), n: 2 * NV, k: D };
            Mixer::Delta { qkv, z, beta_alpha, conv: self.upload(&conv), a: self.upload(&a), dt: self.upload(&dt), norm: self.upload(&norm), out }
        };
        Ok(Layer {
            input: self.norm(&key(Part::InputNorm))?,
            post: self.norm(&key(Part::PostNorm))?,
            mixer,
            gate: self.matrix(&key(Part::Gate), Order::Same)?,
            up: self.matrix(&key(Part::Up), Order::Same)?,
            down: self.matrix(&key(Part::Down), Order::Same)?,
        })
    }

    /// A prompt of `ids`' vectors, its embeddings in its states and its positions' RoPE table.
    fn work(&mut self, ids: &[u32]) -> Result<Work> {
        let t = ids.len();
        let v = |n: usize| self.vec(n);
        let w = Work {
            t,
            x: v(t * D),
            xn: v(t * D),
            table: v(t * ROT),
            qkv: v(t * CH),
            z: v(t * NV * DK),
            ba: v(t * 2 * NV),
            conv: v(t * CH),
            core: v(t * NV * DK),
            conv_state: v((KERN - 1) * CH),
            state: v(NV * DK * DK),
            qfull: v(t * HEADS * 2 * HD),
            q: v(t * HEADS * HD),
            gate: v(t * HEADS * HD),
            k: v(t * KV_HEADS * HD),
            v: v(t * KV_HEADS * HD),
            qq: v(t * HEADS * HD),
            kk: v(t * KV_HEADS * HD),
            kv: v(t * 2 * KV_HEADS * HD),
            att: v(self.gpu.attention_rows_out_len(t, HEADS, HD, t)),
            gated: v(t * HEADS * HD),
            proj: v(t * D),
            g: v(t * FF),
            u: v(t * FF),
            act: v(t * FF),
        };
        let embedding = self.w.rows(&self.names.embedding(), ids)?;
        self.gpu.upload(&w.x, &embedding);
        // the partial RoPE's (sine, cosine) pairs for positions 0..t, as transformers' rotary embedding makes them
        // for text (its three axes the same position)
        let table: Vec<f32> = (0..t)
            .flat_map(|pos| (0..ROT / 2).flat_map(move |k| {
                let (s, c) = (pos as f32 * THETA.powf(-2.0 * k as f32 / ROT as f32)).sin_cos();
                [s, c]
            }))
            .collect();
        self.gpu.upload(&w.table, &table);
        Ok(w)
    }

    /// Layer `l` of a prompt's `t` rows recorded on `r`: its states (`p.x`) through it.
    fn record(r: &mut dyn ChainRecorder, l: &Layer, p: &Work) {
        let t = p.t;
        let (x, xn, proj) = (first(&p.x, t * D), first(&p.xn, t * D), first(&p.proj, t * D));
        r.rmsnorm_rows(&x, &l.input, &xn, t, EPS);
        match &l.mixer {
            Mixer::Delta { qkv, z, beta_alpha, conv, a, dt, norm, out } => {
                qkv.run(r, &xn, &p.qkv, t);
                z.run(r, &xn, &p.z, t);
                beta_alpha.run(r, &xn, &p.ba, t);
                r.ssm_conv(&p.qkv, conv, &p.conv_state, &p.conv, t, CH, KERN);
                let delta = DeltaNet { rows: t, v_heads: NV, k_heads: NK, k_dim: DK, v_dim: DK, scale_q: 1.0 / (DK as f32).sqrt(), eps: EPS, sigmoid_gate: false };
                r.delta_net(&p.conv, &p.z, &p.ba, a, dt, norm, &p.state, &p.core, delta);
                out.run(r, &p.core, &proj, t);
            }
            Mixer::Attention { q, k, v, o, q_norm, k_norm } => {
                let kvd = KV_HEADS * HD;
                // each head's q its query then its gate
                q.run(r, &xn, &p.qfull, t);
                k.run(r, &xn, &p.k, t);
                v.run(r, &xn, &p.v, t);
                r.copy_cols(&p.qfull, &p.q, t * HEADS, HD, 2 * HD, 0);
                r.copy_cols(&p.qfull, &p.gate, t * HEADS, HD, 2 * HD, HD);
                r.rmsnorm_rows(&p.q, q_norm, &p.qq, t * HEADS, EPS);
                r.rmsnorm_rows(&p.k, k_norm, &p.kk, t * KV_HEADS, EPS);
                r.rope_partial_rows(&p.qq, t, HEADS, HD, ROT, &p.table);
                r.rope_partial_rows(&p.kk, t, KV_HEADS, HD, ROT, &p.table);
                r.store_rows(&p.kk, &p.kv, t, kvd, 0, 2 * kvd, 0);
                r.store_rows(&p.v, &p.kv, t, kvd, 0, 2 * kvd, kvd);
                r.attention_rows(&p.qq, &p.kv, &p.att, t, HEADS, KV_HEADS, HD, 0, None, 1.0 / (HD as f32).sqrt());
                r.mul_sigmoid(&p.att, &p.gate, &p.gated, t * HEADS * HD);
                o.run(r, &p.gated, &proj, t);
            }
        }
        r.add_rmsnorm_rows(&x, &proj, &l.post, &xn, t, EPS);
        l.gate.run(r, &xn, &p.g, t);
        l.up.run(r, &xn, &p.u, t);
        r.silu_mul(&p.g, &p.u, &p.act, t * FF);
        l.down.run(r, &p.act, &proj, t);
        r.add(&x, &proj);
    }

    /// Each prompt of `ids`' states after the first `layers` layers, through the final norm.
    fn encode_ids(&mut self, ids: &[Vec<u32>], layers: usize) -> Result<Vec<Vec<f32>>> {
        if ids.iter().any(|i| i.is_empty()) {
            candle_core::bail!("a prompt of no tokens");
        }
        let work = ids.iter().map(|i| self.work(i)).collect::<Result<Vec<_>>>()?;
        for i in 0..layers.min(self.layers) {
            let layer = self.layer(i)?;
            for p in &work {
                // (each delta net's recurrence from nothing: its conv window and state zero)
                if matches!(layer.mixer, Mixer::Delta { .. }) {
                    self.gpu.zero(&p.conv_state);
                    self.gpu.zero(&p.state);
                }
                let mut rec = self.gpu.begin();
                rec.keep_groups(false);
                Self::record(rec.as_mut(), &layer, p);
                rec.finish();
            }
            drop(layer);
            self.gpu.settle();
        }
        let key = self.names.final_norm();
        let norm = self.norm(&key)?;
        let mut out = Vec::with_capacity(work.len());
        for p in &work {
            let (x, xn) = (first(&p.x, p.t * D), first(&p.xn, p.t * D));
            let mut rec = self.gpu.begin();
            rec.rmsnorm_rows(&x, &norm, &xn, p.t, EPS);
            rec.read(&xn);
            out.push(rec.finish().pop().ok_or_else(|| err("the text encoder's states were not read"))?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn the_template_is_the_pipelines() {
        assert_eq!(template("  A cat\n "), "<|im_start|>user\nA cat<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_eq!(template(""), "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n");
    }

    /// Hugging Face's grouped value heads (a key head's two side by side) to llama.cpp's tiled ones (every key head's
    /// first, then every one's second), as `_reorder_v_heads` makes them.
    #[test]
    fn value_heads_go_from_grouped_to_tiled() {
        assert_eq!((0..NV).map(grouped).collect::<Vec<_>>()[..4], [0, 2, 4, 6]);
        assert_eq!(grouped(NK), 1);
        assert_eq!(grouped(NV - 1), NV - 1);
        // rows from 2 (two rows a head), and columns (one a head)
        let (n, k) = (2 + 2 * NV, 3);
        let mut m: Vec<usize> = (0..n * k).collect();
        reorder(&mut m, n, k, Order::Rows { from: 2, head: 2 });
        assert_eq!(&m[..6], &[0, 1, 2, 3, 4, 5]);
        // (tiled head 1 is grouped head 2: its rows 2 + 4 and 2 + 5)
        assert_eq!(&m[(2 + 2) * k..(2 + 4) * k], &[(2 + 4) * k, (2 + 4) * k + 1, (2 + 4) * k + 2, (2 + 5) * k, (2 + 5) * k + 1, (2 + 5) * k + 2]);
        let mut c: Vec<usize> = (0..NV).collect();
        reorder(&mut c, 1, NV, Order::Cols { head: 1 });
        assert_eq!(c, (0..NV).map(grouped).collect::<Vec<_>>());
    }

    /// The states of Qwen3.5-9B (`OAIY_MOSS_TEXT_ENCODER`, Hugging Face's checkpoint or a GGUF of it; with
    /// `OAIY_MOSS_TOKENIZER`, MOSS-Image's tokenizer.json, where a checkpoint needs one) for the prompts of
    /// `tools/moss_image/text_reference.py`'s file (`OAIY_MOSS_TEXT_REFERENCE`: transformers in float32): the same tokens,
    /// the states after four layers and after all of them, each through the final norm, alike.
    #[test]
    #[ignore = "needs Qwen3.5-9B (OAIY_MOSS_TEXT_ENCODER), its reference states (OAIY_MOSS_TEXT_REFERENCE) and a WebGPU adapter"]
    fn the_states_are_transformers() -> Result<()> {
        let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
        let (Some(checkpoint), Some(reference)) = (var("OAIY_MOSS_TEXT_ENCODER"), var("OAIY_MOSS_TEXT_REFERENCE")) else { return Ok(()) };
        let tokenizer = var("OAIY_MOSS_TOKENIZER");
        let quantized = checkpoint.extension().is_some_and(|e| e == "gguf");
        let mut want = Weights::open(&reference)?;
        let prompts: Vec<String> = match want.metadata("prompts") {
            Some(p) => oaiy_engine::json::Json::parse(p.as_bytes()).map_err(err)?.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect(),
            None => candle_core::bail!("the reference names no prompts"),
        };
        let mut model = WgpuMossText::load(&checkpoint, tokenizer.as_deref(), 0)?;
        let mut ids = Vec::new();
        for (j, prompt) in prompts.iter().enumerate() {
            let tokens: Vec<u32> = want.tensor(&format!("prompt{j}.tokens"), &Device::Cpu, DType::F32)?.to_vec1::<f32>()?.into_iter().map(|v| v as u32).collect();
            assert_eq!(model.tokens(prompt)?, tokens, "prompt {j}'s tokens");
            ids.push(tokens);
        }
        for (stage, layers) in [("early", 4), ("final", usize::MAX)] {
            // (every prompt through each layer: the checkpoint read once)
            let clock = std::time::Instant::now();
            let all = model.encode_ids(&ids, layers)?;
            let seconds = clock.elapsed().as_secs_f64();
            for (j, (got, tokens)) in all.into_iter().zip(&ids).enumerate() {
                // (OAIY_MOSS_TEXT_DUMP: each prompt's states written beside, little-endian f32, to set against others')
                if let Some(dir) = var("OAIY_MOSS_TEXT_DUMP") {
                    std::fs::write(dir.join(format!("prompt{j}.{stage}.f32")), got.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                }
                let reference = want.tensor(&format!("prompt{j}.{stage}"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(got.len(), reference.len());
                let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cosine = got.iter().zip(&reference).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (norm(&got) * norm(&reference));
                // (each token's own: the worst of them)
                let worst = got.chunks_exact(D).zip(reference.chunks_exact(D)).map(|(a, b)| a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>() / (norm(a) * norm(b))).fold(1f64, f64::min);
                eprintln!("prompt {j} ({} tokens) {stage}: cosine {cosine:.6}, the worst token's {worst:.6} (the prompts in {seconds:.1} s)", tokens.len());
                // (a GGUF's own error: llama.cpp's states of unsloth's Q4_K_M are 0.9945 of these for the first prompt,
                // 0.980 for the empty one, the worst tokens' 0.970 and 0.953; these 0.9952 and 0.983, the worst 0.954
                // and 0.957, and 0.9986 and 0.997 of llama.cpp's)
                let (floor, token) = if quantized { (0.975, 0.94) } else { (0.9995, 0.999) };
                assert!(cosine > floor && worst > token, "prompt {j} {stage}: {cosine} (worst {worst})");
            }
        }
        Ok(())
    }
}
