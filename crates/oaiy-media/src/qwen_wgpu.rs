//! Qwen Image 2.1's transformer on WebGPU ([`ggml_rs_wgpu`]'s chain of ops), as [`crate::transformer`] computes it: the
//! text prefix conditioned at time zero, its keys and values made once a prompt; each step's image tokens over them
//! and their own. The weights are f16 on the GPU (the BF16 checkpoint's rounded to the nearest: none past f16's range,
//! and the 0.01% it holds only nearly all below 2^-17), LoRA factors merged in as they load; a GGUF's K-quant blocks
//! stay as they are where the adapter has tensor cores (Q4_K_M's 4.5 GB, not 14 as f16), each LoRA's factors beside
//! them, added as it runs. A step's matmuls run on the tensor cores where the adapter has them (their inputs f16: a
//! step's largest some 130, its MLP's); the prefix's f16 ones read their inputs as f32 (its MLP's reach some 5,000,
//! f16's range 65,504).
use crate::{lora::Loras, text::Conditioning, weights::{Raw, Weights}};
use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use ggml_rs::{Backend, ChainRecorder, DeviceChain, DeviceVec, QuantizedTensor};
use std::path::{Path, PathBuf};

const D: usize = 4096;
const HEADS: usize = 32;
const HD: usize = 128;
const FF: usize = 12288;
const BLOCKS: usize = 32;
const CH: usize = 64;
/// A position's key then its value, each `D`: a layer's cache row as the chain's attention reads it.
const ROW: usize = 2 * D;
const EPS: f32 = 1e-6;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A matrix on the GPU (`[n, k]`): f16 (a checkpoint's float weights, LoRA factors merged in as they load), or a GGUF's
/// K-quant blocks as they are, each adapter's factors then kept beside them and added as it runs (`B A x`: `A`
/// `[rank, k]` and `B` `[n, rank]` as f16, its scale in `B`).
struct Mat {
    w: W,
    n: usize,
    k: usize,
    lora: Vec<(DeviceVec, DeviceVec, usize)>,
}

enum W {
    F16(DeviceVec),
    Quant(QuantizedTensor),
}

/// A LoRA's running vectors: its `A x` (rows of the largest rank) and its `B A x` (rows of the widest output).
struct Low {
    t: DeviceVec,
    y: DeviceVec,
}

/// `v`'s first `len` values.
fn first(v: &DeviceVec, len: usize) -> DeviceVec {
    DeviceVec { len, inner: v.inner.clone() }
}

/// The K-quants with tensor-core kernels.
fn coop_type(t: GgmlDType) -> Option<ggml_quants::GgmlType> {
    use ggml_quants::GgmlType as G;
    Some(match t {
        GgmlDType::Q3K => G::Q3_K,
        GgmlDType::Q4K => G::Q4_K,
        GgmlDType::Q5K => G::Q5_K,
        GgmlDType::Q6K => G::Q6_K,
        GgmlDType::Q8_0 => G::Q8_0,
        _ => return None,
    })
}

/// A tensor's values as an f16 matrix on `gpu`.
fn f16_vec(gpu: &ggml_rs_wgpu::WgpuBackend, t: &Tensor, what: &str) -> Result<DeviceVec> {
    let words = crate::wgpu_weights::f16_words_f32(&t.flatten_all()?.to_vec1::<f32>()?).ok_or_else(|| err(format!("{what}: past f16's range")))?;
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
    Ok(v)
}

struct Block {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    qn: DeviceVec,
    kn: DeviceVec,
}

/// A prompt's conditioning: each layer's keys and values of its text prefix (`nt` positions of [`ROW`]), and the
/// position its image starts at.
pub struct WgpuPrefix {
    kv: Vec<DeviceVec>,
    nt: usize,
    position: usize,
}

/// A step's vectors, kept for the next step of the same image size whatever its prompt (a guided step's two prompts
/// alternate: a set each time would leave the old in the kept bind groups), the keys' and values' for the longest
/// prefix so far.
struct Scratch {
    ni: usize,
    /// The prefix positions `kv` and `att` have room for.
    nt: usize,
    lat: DeviceVec,
    table: DeviceVec,
    t: DeviceVec,
    x: DeviceVec,
    norm: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    o: DeviceVec,
    g: DeviceVec,
    u: DeviceVec,
    act: DeviceVec,
    vel: DeviceVec,
    time: [DeviceVec; 4],
    mods: DeviceVec,
    scale: DeviceVec,
    low: Low,
}

pub struct WgpuTransformer {
    gpu: ggml_rs_wgpu::WgpuBackend,
    /// The largest rank of the LoRA factors kept beside quantized weights (0: none).
    rank: usize,
    img: Mat,
    text1: Mat,
    text2: Mat,
    time1: Mat,
    time2: Mat,
    modulation: Mat,
    norm_out: Mat,
    out: Mat,
    /// `1 + w` of the text states' RMS norm.
    text_norm: DeviceVec,
    blocks: Vec<Block>,
    lora_notes: Vec<String>,
    scratch: Option<Scratch>,
}

/// `name`'s weight `[n, k]` on `gpu`: a GGUF's K-quant blocks as they are where the tensor cores take them (rows a
/// multiple of 256 long; OAIY_WEBGPU_DEQUANTIZE f16 throughout), each adapter's factors beside them; else f16, each
/// adapter's factors merged in (`W + B A`).
fn matrix(w: &mut Weights, lora: &mut Loras, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<Mat> {
    let key = format!("{name}.weight");
    let shape = w.shape(&key)?;
    let quant = w.ggml_dtype(&key).and_then(coop_type).filter(|_| gpu.tensor_cores() && std::env::var_os("OAIY_WEBGPU_DEQUANTIZE").is_none());
    if let (Some(g), &[n, k]) = (quant, shape.as_slice()) {
        if let (0, Some(Raw::Ggml(_, bytes))) = (k % 256, w.raw(&key)?) {
            let size = bytes.len();
            let q = gpu.to_device_quant(QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], g));
            if !q.is_device() {
                candle_core::bail!("{key}: no room on the GPU for its {} MB", size >> 20);
            }
            let mut low = Vec::new();
            if !lora.is_empty() {
                for (a, b) in lora.factors(name, n, k, &Device::Cpu, DType::F32)? {
                    let rank = a.dim(0)?;
                    low.push((f16_vec(gpu, &a, name)?, f16_vec(gpu, &b, name)?, rank));
                }
            }
            return Ok(Mat { w: W::Quant(q), n, k, lora: low });
        }
    }
    let (v, n, k) = crate::wgpu_weights::f16_matrix(w, gpu, name, lora)?;
    Ok(Mat { w: W::F16(v), n, k, lora: Vec::new() })
}

/// `name` (a norm's weight) as f32 on `gpu`, plus `add`.
fn vector(w: &mut Weights, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, add: f32) -> Result<DeviceVec> {
    let values: Vec<f32> = w.tensor(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| v + add).collect();
    let v = gpu.vec(values.len());
    gpu.upload(&v, &values);
    Ok(v)
}

/// Each position's RoPE pairs (axes 16, 56 and 56 wide) as the chain's table has them: a pair's sine, then its cosine.
fn rope_table(positions: &[[f64; 3]]) -> Vec<f32> {
    let mut t = Vec::with_capacity(positions.len() * HD);
    for pos in positions {
        for (axis, dim) in [16usize, 56, 56].into_iter().enumerate() {
            for j in 0..dim / 2 {
                let a = pos[axis] / 10000f64.powf((j * 2) as f64 / dim as f64);
                t.push(a.sin() as f32);
                t.push(a.cos() as f32);
            }
        }
    }
    t
}

/// The image's positions after the prefix's (as [`crate::transformer`]'s): its rows and columns about its centre.
fn image_positions(position: usize, h: usize, w: usize) -> Vec<[f64; 3]> {
    let mut p = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            p.push([position as f64, y as f64 - (h - h / 2) as f64, x as f64 - (w - w / 2) as f64]);
        }
    }
    p
}

/// The timestep's 256 sinusoids (cosines, then sines).
fn timestep(sigma: f64) -> Vec<f32> {
    let mut v = Vec::with_capacity(256);
    for kind in 0..2 {
        for j in 0..128 {
            let a = 1000. * sigma / 10000f64.powf(j as f64 / 128.);
            v.push(if kind == 0 { a.cos() as f32 } else { a.sin() as f32 });
        }
    }
    v
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

impl WgpuTransformer {
    /// The transformer at `path` on GPU `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming one instead), with
    /// the turbo `adapter` and `loras` merged into its weights. `progress(block)` as each block loads.
    pub fn load(path: &Path, device: usize, adapter: Option<&Path>, loras: &[(PathBuf, f64)], mut progress: impl FnMut(usize)) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut w = Weights::open(path)?;
        let list: Vec<(PathBuf, f64)> = adapter.map(|a| (a.to_owned(), 1.)).into_iter().chain(loras.iter().cloned()).collect();
        let mut lora = Loras::open(&list)?;
        if adapter.is_some() && !lora.first_adapts("transformer_blocks.0.attn.to_q") {
            candle_core::bail!("not a supported turbo adapter (expected LoRA factors for transformer_blocks.0.attn.to_q)");
        }
        let mut m = |name: &str| matrix(&mut w, &mut lora, &gpu, name);
        let (img, text1, text2) = (m("img_in")?, m("txt_in.in_layer")?, m("txt_in.out_layer")?);
        let (time1, time2) = (m("time_text_embed.timestep_embedder.linear_1")?, m("time_text_embed.timestep_embedder.linear_2")?);
        let (modulation, norm_out, out) = (m("modulation.1")?, m("norm_out.linear")?, m("proj_out")?);
        if img.n != D || img.k != CH || modulation.n != 4 * D || norm_out.n != D || out.n != CH {
            candle_core::bail!("not a Qwen Image 2.1 transformer (img_in {}x{}, modulation {}, norm_out {}, proj_out {})", img.n, img.k, modulation.n, norm_out.n, out.n);
        }
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let p = format!("transformer_blocks.{i}");
            let mut m = |name: &str| matrix(&mut w, &mut lora, &gpu, &format!("{p}.{name}"));
            blocks.push(Block {
                q: m("attn.to_q")?,
                k: m("attn.to_k")?,
                v: m("attn.to_v")?,
                o: m("attn.to_out.0")?,
                gate: m("img_mlp.gate_layer")?,
                up: m("img_mlp.proj")?,
                down: m("img_mlp.out")?,
                qn: vector(&mut w, &gpu, &format!("{p}.attn.norm_q.weight"), 0.)?,
                kn: vector(&mut w, &gpu, &format!("{p}.attn.norm_k.weight"), 0.)?,
            });
            progress(i + 1);
        }
        let text_norm = vector(&mut w, &gpu, "txt_in.text_norm.weight", 1.)?;
        // every block has been read once: an adapter that fit nothing is for another model
        let lora_notes = lora.check()?;
        let mats = [&img, &text1, &text2, &time1, &time2, &modulation, &norm_out, &out].into_iter().chain(blocks.iter().flat_map(|b| [&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down]));
        let rank = mats.flat_map(|m| m.lora.iter().map(|l| l.2)).max().unwrap_or(0);
        Ok(Self { gpu, rank, img, text1, text2, time1, time2, modulation, norm_out, out, text_norm, blocks, lora_notes, scratch: None })
    }

    /// LoRA adapters that fit only in part, and what of them was left out.
    pub fn lora_notes(&self) -> &[String] {
        &self.lora_notes
    }

    /// Let go of a step's vectors and what the device keeps for the next (some 10 GB at 1024x1024 past the weights; the
    /// VAE's decode wants the room), made again by the next step.
    pub fn release_scratch(&mut self) {
        self.scratch = None;
        self.gpu.release_cached();
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// A LoRA's running vectors for `rows` rows (nothing to speak of where no factors are kept).
    fn low(&self, rows: usize) -> Low {
        let rows = if self.rank == 0 { 0 } else { rows };
        Low { t: self.vec(rows * self.rank), y: self.vec(rows * FF) }
    }

    /// `y = W x` for `rows` rows, each kept LoRA's `B A x` added: the tensor cores' f16 inputs, or (`exact`) an f16
    /// weight's inputs as f32 (a K-quant's matmul takes them as f16).
    fn mul(rec: &mut dyn ChainRecorder, m: &Mat, x: &DeviceVec, y: &DeviceVec, rows: usize, exact: bool, low: &Low) {
        match &m.w {
            W::F16(v) if exact => rec.matmul_f16_rows_f32(v, m.n, m.k, x, y, rows),
            W::F16(v) => rec.matmul_f16_rows(v, m.n, m.k, x, y, rows),
            W::Quant(q) => rec.matmul_rows(q, x, y, rows),
        }
        for (a, b, rank) in &m.lora {
            rec.matmul_f16_rows(a, *rank, m.k, x, &low.t, rows);
            rec.matmul_f16_rows(b, m.n, *rank, &low.t, &low.y, rows);
            rec.add(&first(y, rows * m.n), &first(&low.y, rows * m.n));
        }
    }

    /// The modulation of a timestep: `mods` the blocks' (`[scale, gate, scale, gate]`, each `D`) and `scale` the
    /// output norm's, from `t` (its sinusoids) through the time embedder; `time` its four vectors.
    fn time(&self, rec: &mut dyn ChainRecorder, t: &DeviceVec, time: &[DeviceVec; 4], mods: &DeviceVec, scale: &DeviceVec, low: &Low) {
        let [t1, t1s, emb, embs] = time;
        Self::mul(rec, &self.time1, t, t1, 1, true, low);
        rec.mul_sigmoid(t1, t1, t1s, D);
        Self::mul(rec, &self.time2, t1s, emb, 1, true, low);
        rec.mul_sigmoid(emb, emb, embs, D);
        Self::mul(rec, &self.modulation, embs, mods, 1, true, low);
        Self::mul(rec, &self.norm_out, embs, scale, 1, true, low);
    }

    /// The vectors of a step of `ni` image tokens after `nt` of the prefix's (kept for the next of the same size, the
    /// keys' and values' grown for a longer prefix).
    fn scratch(&mut self, ni: usize, nt: usize) -> Scratch {
        if let Some(mut s) = self.scratch.take().filter(|s| s.ni == ni) {
            if s.nt < nt {
                s.kv = self.vec((nt + ni) * ROW);
                s.att = self.vec(self.gpu.attention_rows_full_out_len(ni, HEADS, HD, nt + ni));
                s.nt = nt;
            }
            return s;
        }
        let kv_len = nt + ni;
        let att = self.gpu.attention_rows_full_out_len(ni, HEADS, HD, kv_len);
        Scratch {
            ni,
            nt,
            lat: self.vec(ni * CH),
            table: self.vec(ni * HD),
            t: self.vec(256),
            x: self.vec(ni * D),
            norm: self.vec(ni * D),
            q: self.vec(ni * D),
            k: self.vec(ni * D),
            v: self.vec(ni * D),
            qq: self.vec(ni * D),
            kk: self.vec(ni * D),
            kv: self.vec(kv_len * ROW),
            att: self.vec(att),
            o: self.vec(ni * D),
            g: self.vec(ni * FF),
            u: self.vec(ni * FF),
            act: self.vec(ni * FF),
            vel: self.vec(ni * CH),
            time: [self.vec(D), self.vec(D), self.vec(D), self.vec(D)],
            mods: self.vec(4 * D),
            scale: self.vec(D),
            low: self.low(ni),
        }
    }

    /// The prefix's keys and values in every layer, conditioned at time zero as [`crate::transformer::Transformer::
    /// prepare`] makes them: the text's tokens (causal among themselves) and each reference image's latent in place of
    /// its vision tokens (its tokens over each other and everything before them), `refs` the latents (`[1, h w, 64]`)
    /// and their grids.
    pub fn prepare(&mut self, text: &Conditioning, refs: &[(Tensor, usize, usize)]) -> Result<WgpuPrefix> {
        let nt = text.states.dim(1)?;
        if text.states.dims() != [1, nt, D] || nt == 0 {
            candle_core::bail!("text states must be [1, n, {D}], not {:?}", text.states.dims());
        }
        if text.spans.len() != refs.len() {
            candle_core::bail!("conditioning/reference count mismatch");
        }
        // its parts in turn: (where the text's projected states start, or which reference; its tokens; causal), each
        // part's positions (an image's its rows and columns about its centre, a position its larger side)
        enum Part {
            /// The text's projected states from this one.
            Text(usize),
            /// This reference's latent.
            Image(usize),
        }
        let mut parts: Vec<(Part, usize, bool)> = Vec::new();
        let (mut cursor, mut position, mut positions) = (0, 0, Vec::new());
        for (i, (&(start, len), (latent, h, w))) in text.spans.iter().zip(refs).enumerate() {
            if latent.dims() != [1, h * w, CH] || len * 4 != h * w || start < cursor {
                candle_core::bail!("reference latent/vision grid mismatch");
            }
            if start > cursor {
                parts.push((Part::Text(cursor), start - cursor, true));
                positions.extend((position..position + start - cursor).map(|p| [p as f64; 3]));
                position += start - cursor;
            }
            parts.push((Part::Image(i), h * w, false));
            positions.extend(image_positions(position, *h, *w));
            position += (*h).max(*w);
            cursor = start + len;
        }
        if nt > cursor {
            parts.push((Part::Text(cursor), nt - cursor, true));
            positions.extend((position..position + nt - cursor).map(|p| [p as f64; 3]));
            position += nt - cursor;
        }
        let np = positions.len();
        let states = host(&text.states)?;
        let (sv, sn, t1, t1g, xt) = (self.vec(nt * D), self.vec(nt * D), self.vec(nt * D), self.vec(nt * D), self.vec(nt * D));
        self.gpu.upload(&sv, &states);
        // each reference's latent and its tokens through the image projection
        let mut images = Vec::with_capacity(refs.len());
        for (latent, h, w) in refs {
            let lv = self.vec(h * w * CH);
            self.gpu.upload(&lv, &host(latent)?);
            images.push((lv, self.vec(h * w * D), h * w));
        }
        let table = self.vec(np * HD);
        self.gpu.upload(&table, &rope_table(&positions));
        let t = self.vec(256);
        self.gpu.upload(&t, &timestep(0.));
        let time = [self.vec(D), self.vec(D), self.vec(D), self.vec(D)];
        let (mods, scale) = (self.vec(4 * D), self.vec(D));
        let (x, norm, q, k, v, qq, kk, o) = (self.vec(np * D), self.vec(np * D), self.vec(np * D), self.vec(np * D), self.vec(np * D), self.vec(np * D), self.vec(np * D), self.vec(np * D));
        let (g, u, act) = (self.vec(np * FF), self.vec(np * FF), self.vec(np * FF));
        // a part's queries, and its attention's output (with its partials' room), at the start of their own vectors
        let (mut at, mut longest, mut room) = (0, 0, 0);
        let mut spans = Vec::with_capacity(parts.len());
        for (_, n, causal) in &parts {
            let need = if *causal { self.gpu.attention_rows_out_len(n.div_ceil(32) * 32, HEADS, HD, at + n) } else { self.gpu.attention_rows_full_out_len(*n, HEADS, HD, at + n) };
            (longest, room) = (longest.max(*n), room.max(need));
            spans.push(at);
            at += n;
        }
        let (qs, atts, att) = (self.vec(longest * D), self.vec(room), self.vec(np * D));
        let kv: Vec<DeviceVec> = (0..BLOCKS).map(|_| self.vec(np * ROW)).collect();
        let low = self.low(np);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        r.rmsnorm_rows(&sv, &self.text_norm, &sn, nt, EPS);
        Self::mul(r, &self.text1, &sn, &t1, nt, true, &low);
        r.gelu(&t1, &t1g, nt * D);
        Self::mul(r, &self.text2, &t1g, &xt, nt, true, &low);
        for (lv, xi, n) in &images {
            Self::mul(r, &self.img, lv, xi, *n, true, &low);
        }
        // the parts into the prefix's rows: the text's projected states but its images' vision tokens, the images'
        for ((from, n, _), &at) in parts.iter().zip(&spans) {
            match from {
                Part::Text(start) => r.copy(&xt, start * D, &x, at * D, n * D),
                Part::Image(i) => r.copy(&images[*i].1, 0, &x, at * D, n * D),
            }
        }
        self.time(r, &t, &time, &mods, &scale, &low);
        rec.finish();
        let s = 1.0 / (HD as f32).sqrt();
        // a recording a block: what its ops keep for themselves (their inputs' f16 copies, partial sums) let go before
        // the next's (a 1024x1024 reference's 4,096 tokens filled the card otherwise)
        for (b, kvl) in self.blocks.iter().zip(&kv) {
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            r.layernorm_mod_rows(&x, &norm, np, D, &mods, 0, None, EPS);
            Self::mul(r, &b.q, &norm, &q, np, true, &low);
            Self::mul(r, &b.k, &norm, &k, np, true, &low);
            Self::mul(r, &b.v, &norm, &v, np, true, &low);
            r.rmsnorm_rows(&q, &b.qn, &qq, np * HEADS, EPS);
            r.rmsnorm_rows(&k, &b.kn, &kk, np * HEADS, EPS);
            r.rope_rows(&qq, np, HEADS, HD, &table, false);
            r.rope_rows(&kk, np, HEADS, HD, &table, false);
            r.store_rows(&kk, kvl, np, D, 0, ROW, 0);
            r.store_rows(&v, kvl, np, D, 0, ROW, D);
            // each part's queries over the rows to its end: a text's causal, an image's all of them
            for ((_, n, causal), &at) in parts.iter().zip(&spans) {
                r.copy(&qq, at * D, &qs, 0, n * D);
                if *causal {
                    r.attention_rows(&qs, kvl, &atts, *n, HEADS, HEADS, HD, at, None, s);
                } else {
                    r.attention_rows_full(&qs, kvl, &atts, *n, HEADS, HEADS, HD, at + n, s);
                }
                r.copy(&atts, 0, &att, at * D, n * D);
            }
            Self::mul(r, &b.o, &att, &o, np, true, &low);
            r.add_gated_rows(&x, &o, np, D, &mods, D, true);
            r.layernorm_mod_rows(&x, &norm, np, D, &mods, 2 * D, None, EPS);
            Self::mul(r, &b.gate, &norm, &g, np, true, &low);
            Self::mul(r, &b.up, &norm, &u, np, true, &low);
            r.silu_mul(&g, &u, &act, np * FF);
            Self::mul(r, &b.down, &act, &o, np, true, &low);
            r.add_gated_rows(&x, &o, np, D, &mods, 3 * D, true);
            rec.finish();
        }
        Ok(WgpuPrefix { kv, nt: np, position })
    }

    /// The velocity of `latent` (`[1, h w, 64]`) at `sigma`, conditioned on `prefix`: `[1, h w, 64]` on the CPU.
    pub fn conditioned(&mut self, latent: &Tensor, prefix: &WgpuPrefix, sigma: f64, h: usize, w: usize) -> Result<Tensor> {
        let ni = h * w;
        if latent.dims() != [1, ni, CH] {
            candle_core::bail!("latent must be [1, {ni}, {CH}], not {:?}", latent.dims());
        }
        let nt = prefix.nt;
        let s = self.scratch(ni, nt);
        self.gpu.upload(&s.lat, &host(latent)?);
        self.gpu.upload(&s.table, &rope_table(&image_positions(prefix.position, h, w)));
        self.gpu.upload(&s.t, &timestep(sigma));
        let mut rec = self.gpu.begin();
        // (bind groups kept would hold each step's own scratch, the tensor cores' f16 copies and partial sums: some
        // 0.8 GB a step at 1024x1024, never let go)
        rec.keep_groups(false);
        let r = rec.as_mut();
        self.time(r, &s.t, &s.time, &s.mods, &s.scale, &s.low);
        Self::mul(r, &self.img, &s.lat, &s.x, ni, false, &s.low);
        let scale = 1.0 / (HD as f32).sqrt();
        for (b, pkv) in self.blocks.iter().zip(&prefix.kv) {
            r.copy(pkv, 0, &s.kv, 0, nt * ROW);
            r.layernorm_mod_rows(&s.x, &s.norm, ni, D, &s.mods, 0, None, EPS);
            Self::mul(r, &b.q, &s.norm, &s.q, ni, false, &s.low);
            Self::mul(r, &b.k, &s.norm, &s.k, ni, false, &s.low);
            Self::mul(r, &b.v, &s.norm, &s.v, ni, false, &s.low);
            r.rmsnorm_rows(&s.q, &b.qn, &s.qq, ni * HEADS, EPS);
            r.rmsnorm_rows(&s.k, &b.kn, &s.kk, ni * HEADS, EPS);
            r.rope_rows(&s.qq, ni, HEADS, HD, &s.table, false);
            r.rope_rows(&s.kk, ni, HEADS, HD, &s.table, false);
            r.store_rows(&s.kk, &s.kv, ni, D, nt, ROW, 0);
            r.store_rows(&s.v, &s.kv, ni, D, nt, ROW, D);
            r.attention_rows_full(&s.qq, &s.kv, &s.att, ni, HEADS, HEADS, HD, nt + ni, scale);
            Self::mul(r, &b.o, &s.att, &s.o, ni, false, &s.low);
            r.add_gated_rows(&s.x, &s.o, ni, D, &s.mods, D, true);
            r.layernorm_mod_rows(&s.x, &s.norm, ni, D, &s.mods, 2 * D, None, EPS);
            Self::mul(r, &b.gate, &s.norm, &s.g, ni, false, &s.low);
            Self::mul(r, &b.up, &s.norm, &s.u, ni, false, &s.low);
            r.silu_mul(&s.g, &s.u, &s.act, ni * FF);
            Self::mul(r, &b.down, &s.act, &s.o, ni, false, &s.low);
            r.add_gated_rows(&s.x, &s.o, ni, D, &s.mods, 3 * D, true);
        }
        r.layernorm_mod_rows(&s.x, &s.norm, ni, D, &s.scale, 0, None, EPS);
        Self::mul(r, &self.out, &s.norm, &s.vel, ni, false, &s.low);
        r.read(&s.vel);
        let vel = rec.finish().pop().ok_or_else(|| err("the step's velocity was not read"))?;
        self.scratch = Some(s);
        Tensor::from_vec(vel, (1, ni, CH), &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How long a block's load takes (`--ignored --nocapture`): its seven matrices from `OAIY_QWEN_IMAGE_TRANSFORMER`,
    /// with and without the turbo adapter `OAIY_QWEN_IMAGE_ADAPTER` merged (on the GPU; its kernels' first build in the
    /// second's).
    #[test]
    #[ignore = "a timing; needs Qwen Image 2.1's transformer and its turbo adapter, and a WebGPU adapter"]
    fn measure_a_blocks_load() -> Result<()> {
        let (Some(path), Some(adapter)) = (std::env::var_os("OAIY_QWEN_IMAGE_TRANSFORMER"), std::env::var_os("OAIY_QWEN_IMAGE_ADAPTER")) else { return Ok(()) };
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(err)?;
        let mut w = Weights::open(std::path::Path::new(&path))?;
        for list in [vec![], vec![(PathBuf::from(&adapter), 1.)]] {
            let mut lora = Loras::open(&list)?;
            let t = std::time::Instant::now();
            for name in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"] {
                matrix(&mut w, &mut lora, &gpu, &format!("transformer_blocks.10.{name}"))?;
            }
            gpu.settle();
            eprintln!("block 10 with {} adapters: {:.3} s", list.len(), t.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Each step's time over many (`--ignored --nocapture`): the transformer of `OAIY_QWEN_IMAGE_TRANSFORMER` with its
    /// turbo adapter `OAIY_QWEN_IMAGE_ADAPTER`, a 1024x1024 picture's latent (QWEN_IMAGE_GRID another side), 15 steps
    /// (QWEN_IMAGE_STEPS another count); with OAIY_CHAIN_PROFILE each step's costliest kernels.
    #[test]
    #[ignore = "a timing; needs Qwen Image 2.1's transformer and a WebGPU adapter"]
    fn measure_many_steps() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_QWEN_IMAGE_TRANSFORMER").map(PathBuf::from) else { return Ok(()) };
        let adapter = std::env::var_os("OAIY_QWEN_IMAGE_ADAPTER").map(PathBuf::from);
        let side: usize = std::env::var("QWEN_IMAGE_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let steps: usize = std::env::var("QWEN_IMAGE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
        let nt = 31usize;
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let states: Vec<f32> = (0..nt * D).map(|i| (next() * if i % D % 997 == 5 { 300. } else { 3. }) as f32).collect();
        let latent: Vec<f32> = (0..side * side * CH).map(|_| next() as f32).collect();
        let text = Conditioning { states: Tensor::from_vec(states, (1, nt, D), &Device::Cpu)?, spans: Vec::new() };
        let mut latent = Tensor::from_vec(latent, (1, side * side, CH), &Device::Cpu)?;
        let mut gpu = WgpuTransformer::load(&path, 0, adapter.as_deref(), &[], |_| {})?;
        let prefix = gpu.prepare(&text, &[])?;
        for step in 0..steps {
            let t = std::time::Instant::now();
            let sigma = 1. - step as f64 / (steps + 1) as f64;
            let v = gpu.conditioned(&latent, &prefix, sigma, side, side)?;
            latent = (latent + (v * -0.0625)?)?;
            eprintln!("step {step}: {:.3} s", t.elapsed().as_secs_f64());
            if std::env::var_os("OAIY_CHAIN_PROFILE").is_some() {
                eprintln!("  {:?}", ggml_rs_wgpu::profile::take_kernels().into_iter().take(6).collect::<Vec<_>>());
            }
        }
        Ok(())
    }

    /// With a reference image: 40 text states, 16 of them an image's vision tokens (at 8), its 8 x 8 latent in their
    /// place in the prefix (its tokens over each other and all before them); an 8 x 8 picture's velocity as Candle's
    /// (CPU, f32) on the published weights (`OAIY_QWEN_IMAGE_TRANSFORMER`), at two sigmas.
    #[test]
    #[ignore = "needs the Qwen Image 2.1 transformer (OAIY_QWEN_IMAGE_TRANSFORMER) and a WebGPU adapter"]
    fn a_reference_images_prefix_is_the_candle_ones() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_QWEN_IMAGE_TRANSFORMER").map(PathBuf::from) else { return Ok(()) };
        let (nt, side) = (40usize, 8usize);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let states: Vec<f32> = (0..nt * D).map(|i| (next() * if i % D % 997 == 5 { 300. } else { 3. }) as f32).collect();
        let reference: Vec<f32> = (0..side * side * CH).map(|_| next() as f32).collect();
        let latent: Vec<f32> = (0..side * side * CH).map(|_| (next() * 1.7) as f32).collect();
        let text = Conditioning { states: Tensor::from_vec(states, (1, nt, D), &Device::Cpu)?, spans: vec![(8, side * side / 4)] };
        let refs = vec![(Tensor::from_vec(reference, (1, side * side, CH), &Device::Cpu)?, side, side)];
        let latent = Tensor::from_vec(latent, (1, side * side, CH), &Device::Cpu)?;
        let mut gpu = WgpuTransformer::load(&path, 0, None, &[], |_| {})?;
        let prefix = gpu.prepare(&text, &refs)?;
        let got: Vec<Vec<f32>> = [0.9, 0.3].iter().map(|&s| gpu.conditioned(&latent, &prefix, s, side, side)?.flatten_all()?.to_vec1::<f32>()).collect::<Result<_>>()?;
        drop(gpu);
        let budget = crate::residency::Budget::default();
        let mut cpu = crate::transformer::Transformer::load(&path, None, &[], &Device::Cpu, DType::F32, &budget, |_| {})?;
        let prefix = cpu.prepare(&text, &refs)?;
        for (i, &sigma) in [0.9, 0.3].iter().enumerate() {
            let want = cpu.conditioned(&latent, &prefix, sigma, side, side)?.flatten_all()?.to_vec1::<f32>()?;
            let dot: f64 = got[i].iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got[i]) * norm(&want));
            eprintln!("sigma {sigma}, a reference's prefix: cosine {cos:.6}");
            assert!(cos > 0.999, "sigma {sigma}: cosine {cos}");
        }
        Ok(())
    }

    /// The WebGPU transformer gives the Candle one's velocity (CPU, f32) on the published weights
    /// (`OAIY_QWEN_IMAGE_TRANSFORMER`, e.g. `D:\Qwen-Image-2.1\transformer`): 16 text tokens' states (random, as an
    /// encoder's are: a few channels large) and an 8 x 8 image's latent (QWEN_IMAGE_GRID another side) at two sigmas;
    /// with OAIY_QWEN_IMAGE_ADAPTER a turbo adapter on both.
    #[test]
    #[ignore = "needs the Qwen Image 2.1 transformer (OAIY_QWEN_IMAGE_TRANSFORMER) and a WebGPU adapter"]
    fn the_webgpu_transformer_is_the_candle_one() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_QWEN_IMAGE_TRANSFORMER").map(PathBuf::from) else { return Ok(()) };
        // QWEN_IMAGE_GRID: the latent's side (8 by default; 32 is a 512x512 picture's)
        let side: usize = std::env::var("QWEN_IMAGE_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let (nt, h, w) = (16usize, side, side);
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let states: Vec<f32> = (0..nt * D).map(|i| (next() * if i % D % 997 == 5 { 300. } else { 3. }) as f32).collect();
        let latent: Vec<f32> = (0..h * w * CH).map(|_| (next() * 1.7) as f32).collect();
        let text = Conditioning { states: Tensor::from_vec(states, (1, nt, D), &Device::Cpu)?, spans: Vec::new() };
        let latent = Tensor::from_vec(latent, (1, h * w, CH), &Device::Cpu)?;
        let t = std::time::Instant::now();
        // OAIY_QWEN_IMAGE_ADAPTER: a turbo adapter on both (merged into the weights here, added as it runs there)
        let adapter = std::env::var_os("OAIY_QWEN_IMAGE_ADAPTER").map(PathBuf::from);
        let mut gpu = WgpuTransformer::load(&path, 0, adapter.as_deref(), &[], |_| {})?;
        eprintln!("WebGPU transformer loaded in {:.1} s", t.elapsed().as_secs_f64());
        let prefix = gpu.prepare(&text, &[])?;
        let got: Vec<Vec<f32>> = [0.9, 0.3].iter().map(|&s| gpu.conditioned(&latent, &prefix, s, h, w)?.flatten_all()?.to_vec1::<f32>()).collect::<Result<_>>()?;
        drop(gpu);
        let t = std::time::Instant::now();
        let budget = crate::residency::Budget::default();
        let mut cpu = crate::transformer::Transformer::load(&path, adapter.as_deref(), &[], &Device::Cpu, DType::F32, &budget, |_| {})?;
        eprintln!("Candle transformer loaded in {:.1} s", t.elapsed().as_secs_f64());
        let prefix = cpu.prepare(&text, &[])?;
        for (i, &sigma) in [0.9, 0.3].iter().enumerate() {
            let want = cpu.conditioned(&latent, &prefix, sigma, h, w)?.flatten_all()?.to_vec1::<f32>()?;
            let dot: f64 = got[i].iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got[i]) * norm(&want));
            let rms = norm(&want) / (want.len() as f64).sqrt();
            let worst = got[i].iter().zip(&want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
            eprintln!("sigma {sigma}: cosine {cos:.6}, the worst error {worst:.4} of an RMS {rms:.4}");
            assert!(cos > 0.999, "sigma {sigma}: cosine {cos}");
        }
        Ok(())
    }
}
