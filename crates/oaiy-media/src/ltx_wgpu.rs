//! LTX 2.3's transformer on WebGPU, as [`crate::ltx::transformer`] computes its video stream (text- and
//! image-to-video, its clean first and last frames' tokens at timestep 0, a negative prompt by NAG) and, where it is
//! loaded with it, its audio stream beside (a clip's sound denoised with its picture: each block's audio attentions
//! and feed-forward, and the two attentions between the streams): its weights on the GPU as they are stored where the chain has a kernel
//! for them (NVFP4 packed, the tensor cores decoding it as they multiply: Lightricks' `-nvfp4` release's 44 blocks of
//! 48), else Q8_0 where the tensor cores take it (a BF16 checkpoint's video stream is 28 GB as f16, 15 as Q8_0), else
//! f16 (BF16 rounded); its activations f32.
use crate::ltx::store::{untile_scales, Store};
use candle_core::{Device, Result};
use dsv41::safetensors::Dtype;
use ggml_rs::{Backend, ChainRecorder, CleanRows, DeviceChain, DeviceVec, QuantizedTensor, RowNorm};
use crate::wgpu_weights::{f16_words, f16_words_f32, q8_0, q8_0_bf16};

const PREFIX: &str = "model.diffusion_model.";
/// The video stream's width, heads and head.
const D: usize = 4096;
const HEADS: usize = 32;
const HD: usize = 128;
const BLOCKS: usize = 48;
const EPS: f32 = 1e-6;
/// The audio stream's width (its heads 64 wide), which the attentions between the streams work at too.
const A: usize = 2048;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A linear layer on the GPU: its weight NVFP4 (its words and its scale's vector), Q8_0 or f16, and its bias.
pub enum Weight {
    Nvfp4 { w: DeviceVec, scale: DeviceVec },
    Q8(QuantizedTensor),
    F16(DeviceVec),
}

/// A dense weight's values as stored: a BF16 checkpoint's bytes (converted on every core), else f32.
enum Values {
    Bf16(Vec<u8>),
    F32(Vec<f32>),
}

impl Values {
    fn of(store: &mut Store, key: &str) -> Result<Self> {
        // (a weight a LoRA adapts: the store's sum of the two, rounded as the reference rounds it, where the bytes as
        // stored are the weight alone)
        if store.adapted(key) {
            return Ok(Self::F32(store.tensor(key, &Device::Cpu, false)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?));
        }
        Ok(match store.bf16_bytes(key)? {
            Some(bytes) => Self::Bf16(bytes),
            None => Self::F32(store.tensor_f32(key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
        })
    }
}

/// A dense weight (`[n, k]`) on `gpu`: Q8_0 where its rows are a multiple of 256 long (the tensor cores' kernels, else
/// the f32 tiled one: without tensor cores a 512x320 clip's steps 10.3 s where f16's 11.5, 14.4 GB where 29.5), else
/// f16 (f16 throughout with OAIY_LTX_WEBGPU_WEIGHTS=f16).
fn dense(gpu: &ggml_rs_wgpu::WgpuBackend, key: &str, values: Values, n: usize, k: usize) -> Result<Weight> {
    let f16 = std::env::var("OAIY_LTX_WEBGPU_WEIGHTS").is_ok_and(|v| v.eq_ignore_ascii_case("f16"));
    if k % 256 == 0 && !f16 {
        let bytes = match values {
            Values::Bf16(b) => q8_0_bf16(&b),
            Values::F32(v) => q8_0(&v),
        };
        let size = bytes.len();
        let w = gpu.to_device_quant(QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], ggml_quants::GgmlType::Q8_0));
        if !w.is_device() {
            candle_core::bail!("{key}: no room on the GPU for its {} MB (as Q8_0)", size >> 20);
        }
        return Ok(Weight::Q8(w));
    }
    let words = match values {
        Values::Bf16(b) => f16_words(&b),
        Values::F32(v) => f16_words_f32(&v),
    }
    .ok_or_else(|| err(format!("{key}: past f16's range")))?;
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
    Ok(Weight::F16(v))
}

pub struct Linear {
    pub weight: Weight,
    pub bias: DeviceVec,
    pub n: usize,
    pub k: usize,
}

impl Linear {
    /// `name`'s weight and bias (`{name}.weight`, `{name}.bias`; zeros where none) from `store` onto `gpu`.
    pub fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<Self> {
        let key = format!("{name}.weight");
        let info = store.index.get(&key).cloned().ok_or_else(|| err(format!("missing LTX tensor {key}")))?;
        let global_name = format!("{name}.weight_scale_2");
        let (weight, n, k) = if info.dtype == Dtype::U8 && store.index.get(&global_name).is_some() {
            let (n, half) = match info.shape.as_slice() {
                [n, h] => (*n, *h),
                _ => candle_core::bail!("{key}: an NVFP4 weight of shape {:?}", info.shape),
            };
            let k = 2 * half;
            let packed = store.index.read(&key).map_err(err)?;
            let scale_name = format!("{name}.weight_scale");
            let sinfo = store.index.get(&scale_name).cloned().ok_or_else(|| err(format!("missing {scale_name}")))?;
            let (sr, sc) = match sinfo.shape.as_slice() {
                [r, c] => (*r, *c),
                _ => candle_core::bail!("{scale_name}: scales of shape {:?}", sinfo.shape),
            };
            // (the scales' tiles cover rows to 128's: a layer of fewer rows keeps its own)
            let tiled = store.index.read(&scale_name).map_err(err)?;
            let all = untile_scales(&tiled, sr, sc)?;
            if sc != k / 16 || sr < n {
                candle_core::bail!("{scale_name}: {sr} x {sc} scales for a weight of {n} x {k}");
            }
            let scales = &all[..n * sc];
            let g = store.index.read(&global_name).map_err(err)?;
            let global = f32::from_le_bytes(g.get(..4).and_then(|b| b.try_into().ok()).ok_or_else(|| err(format!("{global_name} is not one F32")))?);
            // (a weight a LoRA adapts is no longer the NVFP4 stored: dense, with the LoRA's part)
            let stored = if store.adapted(&key) { None } else { gpu.nvfp4_weights(&packed, scales, global, n, k) };
            match stored {
                Some((w, scale)) => (Weight::Nvfp4 { w, scale }, n, k),
                None => (dense(gpu, &key, Values::of(store, &key)?, n, k)?, n, k),
            }
        } else {
            let (n, k) = match info.shape.as_slice() {
                [n, k] => (*n, *k),
                _ => candle_core::bail!("{key}: a weight of shape {:?}", info.shape),
            };
            (dense(gpu, &key, Values::of(store, &key)?, n, k)?, n, k)
        };
        let bias_name = format!("{name}.bias");
        let bias: Vec<f32> = if store.index.get(&bias_name).is_some() { store.tensor_f32(&bias_name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()? } else { vec![0.0; n] };
        let b = gpu.vec(n);
        gpu.upload(&b, &bias);
        Ok(Self { weight, bias: b, n, k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows.
    pub fn forward(&self, rec: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match &self.weight {
            Weight::Nvfp4 { w, scale } => rec.matmul_nvfp4_rows(w, scale, &self.bias, self.n, self.k, x, y, rows),
            Weight::Q8(w) => {
                rec.matmul_rows(w, x, y, rows);
                rec.add_bias_rows(y, &self.bias, rows, self.n);
            }
            Weight::F16(w) => {
                rec.matmul_f16_rows(w, self.n, self.k, x, y, rows);
                rec.add_bias_rows(y, &self.bias, rows, self.n);
            }
        }
    }
}

/// One attention's layers: q, k, v, its gate's logits (where it has a gate) and its output, and q's and k's RMS norms
/// (over the width). Its width is q's output's (the video's 4,096, or 2,048 for the audio's own and for those between
/// the streams), its heads [`HEADS`].
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    gate: Option<Linear>,
    out: Linear,
    qn: DeviceVec,
    kn: DeviceVec,
}

impl Attn {
    fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        let gate = format!("{p}.to_gate_logits");
        Ok(Self {
            q: Linear::load(store, gpu, &format!("{p}.to_q"))?,
            k: Linear::load(store, gpu, &format!("{p}.to_k"))?,
            v: Linear::load(store, gpu, &format!("{p}.to_v"))?,
            gate: if store.index.get(&format!("{gate}.weight")).is_some() { Some(Linear::load(store, gpu, &gate)?) } else { None },
            out: Linear::load(store, gpu, &format!("{p}.to_out.0"))?,
            qn: vector(store, gpu, &format!("{p}.q_norm.weight"))?,
            kn: vector(store, gpu, &format!("{p}.k_norm.weight"))?,
        })
    }
}

struct Block {
    attn1: Attn,
    attn2: Attn,
    ff0: Linear,
    ff2: Linear,
    /// The block's nine modulation rows' own part (`[9, D]`), and the prompt's two (`[2, D]`).
    table: DeviceVec,
    prompt_table: DeviceVec,
    audio: Option<AudioBlock>,
}

/// A block's audio stream: its self-attention, text attention and feed-forward as the video's ([`A`] wide), the
/// attention of the video's tokens over the audio's and of the audio's over the video's, and each side's part of
/// those two's modulation (four rows: a scale and a shift for each direction; then its direction's gate).
struct AudioBlock {
    attn1: Attn,
    attn2: Attn,
    ff0: Linear,
    ff2: Linear,
    table: DeviceVec,
    prompt_table: DeviceVec,
    a2v: Attn,
    v2a: Attn,
    cross_video: (DeviceVec, DeviceVec),
    cross_audio: (DeviceVec, DeviceVec),
}

impl AudioBlock {
    fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        // a table of five rows as its first four and its last
        let split = |store: &mut Store, name: &str, width: usize| -> Result<(DeviceVec, DeviceVec)> {
            let values = store.tensor_f32(name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            if values.len() != 5 * width {
                candle_core::bail!("{name}: {} values, not five rows of {width}", values.len());
            }
            let (rows, gate) = (gpu.vec(4 * width), gpu.vec(width));
            gpu.upload(&rows, &values[..4 * width]);
            gpu.upload(&gate, &values[4 * width..]);
            Ok((rows, gate))
        };
        Ok(Self {
            attn1: Attn::load(store, gpu, &format!("{p}.audio_attn1"))?,
            attn2: Attn::load(store, gpu, &format!("{p}.audio_attn2"))?,
            ff0: Linear::load(store, gpu, &format!("{p}.audio_ff.net.0.proj"))?,
            ff2: Linear::load(store, gpu, &format!("{p}.audio_ff.net.2"))?,
            table: vector(store, gpu, &format!("{p}.audio_scale_shift_table"))?,
            prompt_table: vector(store, gpu, &format!("{p}.audio_prompt_scale_shift_table"))?,
            a2v: Attn::load(store, gpu, &format!("{p}.audio_to_video_attn"))?,
            v2a: Attn::load(store, gpu, &format!("{p}.video_to_audio_attn"))?,
            cross_video: split(store, &format!("{p}.scale_shift_table_a2v_ca_video"), D)?,
            cross_audio: split(store, &format!("{p}.scale_shift_table_a2v_ca_audio"), A)?,
        })
    }
}

/// The audio stream's own layers outside the blocks: its patchify, its timestep's and its prompt's embedders, its
/// output's table and projection, and the embedders of the attention between the streams (each side's four rows of
/// scale and shift, each direction's gate).
struct AudioStreams {
    patchify: Linear,
    adaln: TimeEmbed,
    prompt: Option<TimeEmbed>,
    out_table: DeviceVec,
    proj_out: Linear,
    video_rows: TimeEmbed,
    audio_rows: TimeEmbed,
    video_gate: TimeEmbed,
    audio_gate: TimeEmbed,
}

/// A step's audio beside its video ([`WgpuLtx::forward_av`]): the audio latent (`tokens` rows of the audio patchify's
/// width) at `sigma` over its own context (`rows` rows of 2,048, the audio connector's); its tokens rotated by `table`
/// and, in the attention between the streams, the video's by `video_cross` (each a [`rope_table`] of times over 2,048
/// channels, [`WgpuLtx::upload`]ed). With `skip_self` that block's audio self-attention passed through.
pub struct AudioPass<'a> {
    pub latent: &'a [f32],
    pub tokens: usize,
    pub context: &'a [f32],
    pub rows: usize,
    pub sigma: f64,
    pub table: &'a DeviceVec,
    pub video_cross: &'a DeviceVec,
    pub skip_self: Option<usize>,
}

/// A timestep's embedder (`{p}.emb.timestep_embedder.linear_1/2`) and its modulation's linear (`{p}.linear`).
struct TimeEmbed {
    t1: Linear,
    t2: Linear,
    linear: Linear,
}

impl TimeEmbed {
    fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        Ok(Self {
            t1: Linear::load(store, gpu, &format!("{p}.emb.timestep_embedder.linear_1"))?,
            t2: Linear::load(store, gpu, &format!("{p}.emb.timestep_embedder.linear_2"))?,
            linear: Linear::load(store, gpu, &format!("{p}.linear"))?,
        })
    }

    /// `sigma`'s modulation into `m` and its embedding into `e` (`t` its 256 sinusoids, uploaded; `s1`, `s2` scratch
    /// of the embedder's width).
    fn record(&self, r: &mut dyn ChainRecorder, t: &DeviceVec, s1: &DeviceVec, s2: &DeviceVec, e: &DeviceVec, m: &DeviceVec) {
        self.t1.forward(r, t, s1, 1);
        r.mul_sigmoid(s1, s1, s2, self.t1.n);
        self.t2.forward(r, s2, e, 1);
        r.mul_sigmoid(e, e, s1, self.t2.n);
        self.linear.forward(r, s1, m, 1);
    }
}

/// A norm's weight (or any vector) as f32 on `gpu`.
fn vector(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<DeviceVec> {
    let values = store.tensor_f32(name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let v = gpu.vec(values.len());
    gpu.upload(&v, &values);
    Ok(v)
}

/// The timestep's 256 sinusoids as the reference's embedder takes them (cosines, then sines).
fn sinusoids(sigma: f64) -> Vec<f32> {
    let phase: Vec<f32> = (0..128).map(|j| (sigma as f32 * 1000.) * (-10000f32.ln() * j as f32 / 128.).exp()).collect();
    phase.iter().map(|p| p.cos()).chain(phase.iter().map(|p| p.sin())).collect()
}

/// LTX's split rotary tables for `positions` (each its coordinates, `maxima` their ranges) over `dim` channels of
/// `heads` heads, as [`crate::ltx::transformer::Rope::positions`] makes them (in f64): each (position, head) its
/// `dim / heads / 2` pairs' sine then cosine, the chain's layout.
pub fn rope_table(positions: &[Vec<f32>], maxima: &[f32], dim: usize, heads: usize) -> Vec<f32> {
    let axes = maxima.len();
    let half = dim / 2;
    let count = dim / (2 * axes);
    let pad = half - count * axes;
    let mut t = Vec::with_capacity(positions.len() * dim);
    for p in positions {
        for _ in 0..pad {
            t.extend([0f32, 1f32]);
        }
        for j in 0..count {
            let freq = 10000f64.powf(j as f64 / (count - 1) as f64) * std::f64::consts::FRAC_PI_2;
            for axis in 0..axes {
                let phase = freq * (2. * p[axis] as f64 / maxima[axis] as f64 - 1.);
                t.extend([phase.sin() as f32, phase.cos() as f32]);
            }
        }
    }
    let _ = heads;
    t
}

/// A video's tokens' coordinates (time in seconds, then its pixels' row and column), as the reference places them;
/// with `end_image`, an appended end frame's tokens at the last frame's time.
pub fn video_positions(frames: usize, height: usize, width: usize, fps: usize, end_image: bool) -> Vec<Vec<f32>> {
    let mut positions = Vec::with_capacity((frames + usize::from(end_image)) * height * width);
    for t in 0..frames {
        for y in 0..height {
            for x in 0..width {
                let start = (t * 8).saturating_sub(7);
                let end = (t + 1) * 8 - 7;
                positions.push(vec![(start + end) as f32 / (2 * fps) as f32, (y as f32 + 0.5) * 32., (x as f32 + 0.5) * 32.]);
            }
        }
    }
    if end_image {
        let last = ((frames - 1) * 8) as f32 + 0.5;
        for y in 0..height {
            for x in 0..width {
                positions.push(vec![last / fps as f32, (y as f32 + 0.5) * 32., (x as f32 + 0.5) * 32.]);
            }
        }
    }
    positions
}

pub struct WgpuLtx {
    gpu: ggml_rs_wgpu::WgpuBackend,
    patchify: Linear,
    /// LTX 2.5's learned marker of the first latent frame's tokens (added after the patchify).
    keyframe: Option<DeviceVec>,
    adaln: TimeEmbed,
    prompt: Option<TimeEmbed>,
    out_table: DeviceVec,
    proj_out: Linear,
    blocks: Vec<Block>,
    /// The audio stream's layers outside the blocks, where it was loaded with it ([`Self::load_streams`]).
    audio: Option<AudioStreams>,
    /// A negative prompt by normalised attention guidance, for every forward until cleared (as the reference's
    /// `ltx::transformer::Transformer::nag`).
    pub nag: Option<Nag>,
}

/// A negative prompt's video context (`rows` rows of `D`, the connector's, as the prompt's) and normalised attention
/// guidance's scale, tau and alpha: each block's text cross-attention is run over it too, and the two outputs mixed
/// ([`ChainRecorder::nag_mix`]) before the heads' gate and the output projection. One more cross-attention a block,
/// not a second pass of the model.
pub struct Nag {
    pub context: Vec<f32>,
    pub rows: usize,
    pub scale: f32,
    pub tau: f32,
    pub alpha: f32,
}

impl WgpuLtx {
    /// The video stream of the transformer in `store` on GPU `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming
    /// one instead). `progress(block)` as each loads.
    pub fn load(store: &mut Store, device: usize, progress: impl FnMut(usize)) -> Result<Self> {
        Self::load_streams(store, device, false, progress)
    }

    /// [`Self::load`], with the audio stream beside the video's where `audio` (a clip with sound: each block's audio
    /// layers and those between the streams, some half as much again).
    pub fn load_streams(store: &mut Store, device: usize, audio: bool, mut progress: impl FnMut(usize)) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let g = |n: &str| format!("{PREFIX}{n}");
        let patchify = Linear::load(store, &gpu, &g("patchify_proj"))?;
        let adaln = TimeEmbed::load(store, &gpu, &g("adaln_single"))?;
        let prompt = if store.index.get(&g("prompt_adaln_single.linear.weight")).is_some() { Some(TimeEmbed::load(store, &gpu, &g("prompt_adaln_single"))?) } else { None };
        let out_table = vector(store, &gpu, &g("scale_shift_table"))?;
        let keyframe = if store.index.get(&g("keyframes_abs_pos_embedding")).is_some() { Some(vector(store, &gpu, &g("keyframes_abs_pos_embedding"))?) } else { None };
        let proj_out = Linear::load(store, &gpu, &g("proj_out"))?;
        if patchify.n != D || adaln.linear.n != 9 * D || out_table.len != 2 * D {
            candle_core::bail!("not LTX 2.3's video stream (patchify {}, modulation {}, output table {})", patchify.n, adaln.linear.n, out_table.len);
        }
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let p = g(&format!("transformer_blocks.{i}"));
            blocks.push(Block {
                attn1: Attn::load(store, &gpu, &format!("{p}.attn1"))?,
                attn2: Attn::load(store, &gpu, &format!("{p}.attn2"))?,
                ff0: Linear::load(store, &gpu, &format!("{p}.ff.net.0.proj"))?,
                ff2: Linear::load(store, &gpu, &format!("{p}.ff.net.2"))?,
                table: vector(store, &gpu, &format!("{p}.scale_shift_table"))?,
                prompt_table: vector(store, &gpu, &format!("{p}.prompt_scale_shift_table"))?,
                audio: if audio { Some(AudioBlock::load(store, &gpu, &p)?) } else { None },
            });
            progress(i + 1);
        }
        let audio = if audio {
            let streams = AudioStreams {
                patchify: Linear::load(store, &gpu, &g("audio_patchify_proj"))?,
                adaln: TimeEmbed::load(store, &gpu, &g("audio_adaln_single"))?,
                prompt: if store.index.get(&g("audio_prompt_adaln_single.linear.weight")).is_some() { Some(TimeEmbed::load(store, &gpu, &g("audio_prompt_adaln_single"))?) } else { None },
                out_table: vector(store, &gpu, &g("audio_scale_shift_table"))?,
                proj_out: Linear::load(store, &gpu, &g("audio_proj_out"))?,
                video_rows: TimeEmbed::load(store, &gpu, &g("av_ca_video_scale_shift_adaln_single"))?,
                audio_rows: TimeEmbed::load(store, &gpu, &g("av_ca_audio_scale_shift_adaln_single"))?,
                video_gate: TimeEmbed::load(store, &gpu, &g("av_ca_a2v_gate_adaln_single"))?,
                audio_gate: TimeEmbed::load(store, &gpu, &g("av_ca_v2a_gate_adaln_single"))?,
            };
            let (s, widths) = (&streams, [D, A]);
            if s.patchify.n != A || s.adaln.linear.n != 9 * A || s.out_table.len != 2 * A || s.video_rows.linear.n != 4 * D || s.audio_rows.linear.n != 4 * A || s.video_gate.linear.n != D || s.audio_gate.linear.n != A || [&s.adaln, &s.video_rows, &s.audio_rows, &s.video_gate, &s.audio_gate].iter().any(|e| !widths.contains(&e.t1.n) || !widths.contains(&e.t2.n)) {
                candle_core::bail!("not LTX 2.3's audio stream (patchify {}, modulation {}, output table {})", s.patchify.n, s.adaln.linear.n, s.out_table.len);
            }
            Some(streams)
        } else {
            None
        };
        Ok(Self { gpu, patchify, keyframe, adaln, prompt, out_table, proj_out, blocks, audio, nag: None })
    }

    /// Whether it was loaded with its audio stream.
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Let go of the audio stream (its layers' memory the card's again): the video's alone from here on.
    pub fn drop_audio(&mut self) {
        self.audio = None;
        for b in &mut self.blocks {
            b.audio = None;
        }
        self.gpu.settle();
    }

    /// Whether the card has room for a step of `tokens` video tokens beside what is loaded, where it says what it has
    /// (true where it does not): a step's vectors are some 0.41 MiB a token (measured: 17,408 tokens 6.9 GiB), asked
    /// for here as 450 KiB a token (OAIY_LTX_TOKEN_KIB another figure) with a GiB to spare.
    pub fn room_for_step(&self, tokens: usize) -> bool {
        let token_bytes = std::env::var("OAIY_LTX_TOKEN_KIB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(450) << 10;
        self.gpu.memory_budget().is_none_or(|(budget, used)| budget.saturating_sub(used) >= tokens as u64 * token_bytes + (1 << 30))
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// `values` on this transformer's GPU (a rotary table, kept for every step).
    pub fn upload(&self, values: &[f32]) -> DeviceVec {
        let v = self.vec(values.len());
        self.gpu.upload(&v, values);
        v
    }

    /// One attention: `xq`'s `tq` rows' queries over `xkv`'s `tk` rows (each rotated by its table where given), the
    /// heads gated, out through the output projection into `y`. `s` the scratch.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, xkv: &DeviceVec, tk: usize, rope: Option<&DeviceVec>, passthrough: bool, s: &Scratch, y: &DeviceVec) {
        attend(r, a, xq, tq, xkv, tk, rope, rope, passthrough, s, y)
    }

    /// The video velocity of `latent` (`tokens` rows of 128) at `sigma` over `context` (`lc` rows of `D`, the
    /// connector's), its tokens rotated by `table` ([`rope_table`] of [`video_positions`], [`Self::upload`]ed):
    /// `[tokens, 128]`. `clean`: the leading tokens of a starting image and the trailing ones of an end image, at
    /// timestep 0; `first_frame` the first latent frame's tokens (LTX 2.5 marks them). With `skip_self` that block's
    /// self-attention passed through (spatio-temporal guidance's perturbed pass).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], tokens: usize, context: &[f32], lc: usize, sigma: f64, table: &DeviceVec, clean: (usize, usize), first_frame: usize, skip_self: Option<usize>) -> Result<Vec<f32>> {
        self.pass(latent, tokens, context, lc, sigma, table, clean, first_frame, skip_self, false, None)?.pop().ok_or_else(|| err("the velocity was not read"))
    }

    /// [`Self::forward`] with the clip's sound beside its picture: the video's velocity and the audio's (`[audio
    /// tokens, the audio latent's width]`), each block's two streams attending to each other, as the reference's
    /// `av_block` (the model loaded with its audio stream: [`Self::load_streams`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_av(&self, latent: &[f32], tokens: usize, context: &[f32], lc: usize, sigma: f64, table: &DeviceVec, clean: (usize, usize), first_frame: usize, skip_self: Option<usize>, audio: &AudioPass<'_>) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut reads = self.pass(latent, tokens, context, lc, sigma, table, clean, first_frame, skip_self, false, Some(audio))?;
        let sound = reads.pop().ok_or_else(|| err("the audio's velocity was not read"))?;
        Ok((reads.pop().ok_or_else(|| err("the velocity was not read"))?, sound))
    }

    /// [`Self::forward`]'s pass: its reads, the velocity last (with `trace`, every block's output before it, one after
    /// another in one vector: a read is the vector as the recording leaves it).
    #[allow(clippy::too_many_arguments)]
    fn pass(&self, latent: &[f32], tokens: usize, context: &[f32], lc: usize, sigma: f64, table: &DeviceVec, clean: (usize, usize), first_frame: usize, skip_self: Option<usize>, trace: bool, audio: Option<&AudioPass<'_>>) -> Result<Vec<Vec<f32>>> {
        if latent.len() != tokens * self.patchify.k || context.len() != lc * D || table.len != tokens * D {
            candle_core::bail!("an LTX step's inputs: {} latent values for {tokens} tokens, {} context for {lc}, {} rotary", latent.len(), context.len(), table.len);
        }
        let (start, end) = clean;
        if start + end >= tokens {
            candle_core::bail!("conditioning must leave generated video tokens");
        }
        // the clean tokens' rows modulated by timestep 0's set, after sigma's: the blocks' nine rows, the output's two
        let two = start + end > 0;
        let rows = if two { CleanRows { before: start, from: tokens - end, offset: 9 * D } } else { CleanRows::NONE };
        let out_rows = if two { CleanRows { offset: 2 * D, ..rows } } else { CleanRows::NONE };
        let nag = self.nag.as_ref();
        if let Some(n) = nag {
            if n.context.len() != n.rows * D || n.rows == 0 {
                candle_core::bail!("a negative context of {} values for {} rows", n.context.len(), n.rows);
            }
        }
        let ln = nag.map_or(0, |n| n.rows);
        // the audio stream's step, where there is sound: its layers, its inputs and its vectors
        let sound = match audio {
            Some(a) => {
                let g = self.audio.as_ref().ok_or_else(|| err("this transformer was loaded without its audio stream"))?;
                if a.latent.len() != a.tokens * g.patchify.k || a.context.len() != a.rows * A || a.table.len != a.tokens * A || a.video_cross.len != tokens * A || a.tokens == 0 || a.rows == 0 {
                    candle_core::bail!("an LTX step's sound: {} latent values for {} tokens, {} context for {}, {} and {} rotary", a.latent.len(), a.tokens, a.context.len(), a.rows, a.table.len, a.video_cross.len);
                }
                Some((g, a, Sound::new(self, g, a, tokens, if two { 2 } else { 1 })))
            }
            None => None,
        };
        let (ta, la) = audio.map_or((0, 0), |a| (a.tokens, a.rows));
        let s = Scratch::new(&self.gpu, tokens.max(lc).max(ln).max(ta).max(la));
        let (lat, ctx) = (self.vec(latent.len()), self.vec(context.len()));
        self.gpu.upload(&lat, latent);
        self.gpu.upload(&ctx, context);
        // the negative context, its modulated rows and the plain attention's output kept beside the negative's
        let (nctx, ncm, plain) = (self.vec(ln * D), self.vec(ln * D), self.vec(if nag.is_some() { tokens * D } else { 1 }));
        if let Some(n) = nag {
            self.gpu.upload(&nctx, &n.context);
        }
        let t = self.vec(256);
        self.gpu.upload(&t, &sinusoids(sigma));
        let (x, h, cm, y, f, fg) = (self.vec(tokens * D), self.vec(tokens * D), self.vec(lc * D), self.vec(tokens * D), self.vec(tokens * 4 * D), self.vec(tokens * 4 * D));
        let (emb, modulation, prompt) = (self.vec(D), self.vec(9 * D), self.vec(2 * D));
        let trail = self.vec(if trace { self.blocks.len() * tokens * D } else { 1 });
        let sets = if two { 2 } else { 1 };
        let (mm, pm, mo) = (self.vec(sets * 9 * D), self.vec(2 * D), self.vec(sets * 2 * D));
        let (s1, s2) = (self.vec(D), self.vec(D));
        // timestep 0's, for the clean tokens: its sinusoids, embedding and modulation, and a set's scratch
        let (t0, emb0, modulation0, set, out_set) = (self.vec(256), self.vec(D), self.vec(9 * D), self.vec(9 * D), self.vec(2 * D));
        if two {
            self.gpu.upload(&t0, &sinusoids(0.));
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        // the timestep's modulation and embedding, the prompt's modulation
        self.adaln.record(r, &t, &s1, &s2, &emb, &modulation);
        if two {
            self.adaln.record(r, &t0, &s1, &s2, &emb0, &modulation0);
        }
        match &self.prompt {
            Some(pe) => {
                let pe_emb = self.vec(D);
                pe.record(r, &t, &s1, &s2, &pe_emb, &prompt);
            }
            None => r.copy(&self.vec(2 * D), 0, &prompt, 0, 2 * D),
        }
        self.patchify.forward(r, &lat, &x, tokens);
        if let Some(marker) = self.keyframe.as_ref().filter(|_| first_frame > 0) {
            r.add_bias_rows(&x, marker, first_frame.min(tokens), D);
        }
        // the audio's step: every timestep the audio's own sigma, but the video's rows of the attention between the
        // streams (its tokens' own timesteps: sigma's, and 0's for the clean ones) and the audio's gate (the video's)
        if let Some((g, a, v)) = &sound {
            g.adaln.record(r, &v.t, &s1, &s2, &v.emb, &v.modulation);
            match &g.prompt {
                Some(pe) => pe.record(r, &v.t, &s1, &s2, &v.spare, &v.prompt),
                None => r.copy(&self.vec(2 * A), 0, &v.prompt, 0, 2 * A),
            }
            g.audio_rows.record(r, &v.t, &s1, &s2, &v.spare, &v.audio_rows);
            g.video_gate.record(r, &v.t, &s1, &s2, &v.spare, &v.video_gate);
            g.audio_gate.record(r, &t, &s1, &s2, &v.spare, &v.audio_gate);
            for (at, step) in [(0, &t), (4 * D, &t0)].into_iter().take(sets) {
                g.video_rows.record(r, step, &s1, &s2, &v.spare, &v.rows_set);
                r.copy(&v.rows_set, 0, &v.video_rows, at, 4 * D);
            }
            g.patchify.forward(r, &v.lat, &v.x, a.tokens);
        }
        // (the clean video tokens' rows of the attention between the streams: timestep 0's set)
        let cross_rows = if two { CleanRows { offset: 4 * D, ..rows } } else { CleanRows::NONE };
        for (i, b) in self.blocks.iter().enumerate() {
            // the block's modulation: the step's and its own table (shift, scale, gate: self-attention 0-2, the
            // feed-forward 3-5, text attention 6-8), the prompt's two rows the same
            if two {
                for (at, step) in [(0, &modulation), (9 * D, &modulation0)] {
                    r.copy(step, 0, &set, 0, 9 * D);
                    r.add(&set, &b.table);
                    r.copy(&set, 0, &mm, at, 9 * D);
                }
            } else {
                r.copy(&modulation, 0, &mm, 0, 9 * D);
                r.add(&mm, &b.table);
            }
            r.copy(&prompt, 0, &pm, 0, 2 * D);
            r.add(&pm, &b.prompt_table);
            r.norm_mod_rows_clean(&x, &h, tokens, D, &mm, D, Some(0), RowNorm::Rms, EPS, rows);
            self.attend(r, &b.attn1, &h, tokens, &h, tokens, Some(table), skip_self == Some(i), &s, &y);
            r.add_gated_rows_clean(&x, &y, tokens, D, &mm, 2 * D, false, rows);
            r.norm_mod_rows_clean(&x, &h, tokens, D, &mm, 7 * D, Some(6 * D), RowNorm::Rms, EPS, rows);
            r.norm_mod_rows(&ctx, &cm, lc, D, &pm, D, Some(0), RowNorm::None, EPS);
            match nag {
                Some(n) => {
                    // the negative context modulated as the prompt's, the two attentions' outputs mixed
                    r.norm_mod_rows(&nctx, &ncm, ln, D, &pm, D, Some(0), RowNorm::None, EPS);
                    attend_guided(r, &b.attn2, &h, tokens, (&cm, lc), (&ncm, ln), n, &s, &plain, &y);
                }
                None => self.attend(r, &b.attn2, &h, tokens, &cm, lc, None, false, &s, &y),
            }
            r.add_gated_rows_clean(&x, &y, tokens, D, &mm, 8 * D, false, rows);
            if let (Some((_, a, v)), Some(ab)) = (&sound, &b.audio) {
                let (ta, la) = (a.tokens, a.rows);
                // the audio's own attentions, as the video's: its nine rows (shift, scale, gate each), its prompt's two
                r.copy(&v.modulation, 0, &v.mm, 0, 9 * A);
                r.add(&v.mm, &ab.table);
                r.copy(&v.prompt, 0, &v.pm, 0, 2 * A);
                r.add(&v.pm, &ab.prompt_table);
                r.norm_mod_rows(&v.x, &v.h, ta, A, &v.mm, A, Some(0), RowNorm::Rms, EPS);
                attend(r, &ab.attn1, &v.h, ta, &v.h, ta, Some(a.table), Some(a.table), a.skip_self == Some(i), &s, &v.y);
                r.add_gated_rows(&v.x, &v.y, ta, A, &v.mm, 2 * A, false);
                r.norm_mod_rows(&v.x, &v.h, ta, A, &v.mm, 7 * A, Some(6 * A), RowNorm::Rms, EPS);
                r.norm_mod_rows(&v.ctx, &v.cm, la, A, &v.pm, A, Some(0), RowNorm::None, EPS);
                attend(r, &ab.attn2, &v.h, ta, &v.cm, la, None, None, false, &s, &v.y);
                r.add_gated_rows(&v.x, &v.y, ta, A, &v.mm, 8 * A, false);
                // the two streams over each other: each side's rows (a scale then a shift for each direction) and each
                // direction's gate, the step's plus the block's; both directions from the streams as they are now
                for at in (0..sets).map(|set| set * 4 * D) {
                    r.copy(&v.video_rows, at, &v.rows_set, 0, 4 * D);
                    r.add(&v.rows_set, &ab.cross_video.0);
                    r.copy(&v.rows_set, 0, &v.video_mod, at, 4 * D);
                }
                r.copy(&v.audio_rows, 0, &v.audio_mod, 0, 4 * A);
                r.add(&v.audio_mod, &ab.cross_audio.0);
                r.copy(&v.video_gate, 0, &v.video_by, 0, D);
                r.add(&v.video_by, &ab.cross_video.1);
                r.copy(&v.audio_gate, 0, &v.audio_by, 0, A);
                r.add(&v.audio_by, &ab.cross_audio.1);
                r.norm_mod_rows_clean(&x, &h, tokens, D, &v.video_mod, 0, Some(D), RowNorm::Rms, EPS, cross_rows);
                r.norm_mod_rows(&v.x, &v.h, ta, A, &v.audio_mod, 0, Some(A), RowNorm::Rms, EPS);
                attend(r, &ab.a2v, &h, tokens, &v.h, ta, Some(a.video_cross), Some(a.table), false, &s, &y);
                r.norm_mod_rows(&v.x, &v.h, ta, A, &v.audio_mod, 2 * A, Some(3 * A), RowNorm::Rms, EPS);
                r.norm_mod_rows_clean(&x, &h, tokens, D, &v.video_mod, 2 * D, Some(3 * D), RowNorm::Rms, EPS, cross_rows);
                attend(r, &ab.v2a, &v.h, ta, &h, tokens, Some(a.table), Some(a.video_cross), false, &s, &v.y);
                r.add_gated_rows(&x, &y, tokens, D, &v.video_by, 0, false);
                r.add_gated_rows(&v.x, &v.y, ta, A, &v.audio_by, 0, false);
                // the audio's feed-forward
                r.norm_mod_rows(&v.x, &v.h, ta, A, &v.mm, 4 * A, Some(3 * A), RowNorm::Rms, EPS);
                ab.ff0.forward(r, &v.h, &v.f, ta);
                r.gelu(&v.f, &v.fg, ta * ab.ff0.n);
                ab.ff2.forward(r, &v.fg, &v.y, ta);
                r.add_gated_rows(&v.x, &v.y, ta, A, &v.mm, 5 * A, false);
            }
            r.norm_mod_rows_clean(&x, &h, tokens, D, &mm, 4 * D, Some(3 * D), RowNorm::Rms, EPS, rows);
            b.ff0.forward(r, &h, &f, tokens);
            r.gelu(&f, &fg, tokens * b.ff0.n);
            b.ff2.forward(r, &fg, &y, tokens);
            r.add_gated_rows_clean(&x, &y, tokens, D, &mm, 5 * D, false, rows);
            if trace {
                r.copy(&x, 0, &trail, i * tokens * D, tokens * D);
            }
        }
        // the output: a layer norm, its table's (shift, scale) plus the timestep's embedding, the projection
        for (at, e) in [(0, &emb), (2 * D, &emb0)].into_iter().take(sets) {
            r.copy(&self.out_table, 0, &out_set, 0, 2 * D);
            r.add_bias_rows(&out_set, e, 2, D);
            r.copy(&out_set, 0, &mo, at, 2 * D);
        }
        r.norm_mod_rows_clean(&x, &h, tokens, D, &mo, D, Some(0), RowNorm::Layer, EPS, out_rows);
        let vel = self.vec(tokens * self.proj_out.n);
        self.proj_out.forward(r, &h, &vel, tokens);
        if trace {
            r.read(&trail);
        }
        r.read(&vel);
        // the audio's output, as the video's: a layer norm, its table plus its timestep's embedding, its projection
        if let Some((g, a, v)) = &sound {
            r.copy(&g.out_table, 0, &v.pm, 0, 2 * A);
            r.add_bias_rows(&v.pm, &v.emb, 2, A);
            r.norm_mod_rows(&v.x, &v.h, a.tokens, A, &v.pm, A, Some(0), RowNorm::Layer, EPS);
            g.proj_out.forward(r, &v.h, &v.vel, a.tokens);
            r.read(&v.vel);
        }
        Ok(rec.finish())
    }
}

/// A step's audio vectors ([`WgpuLtx::pass`] with sound).
struct Sound {
    lat: DeviceVec,
    ctx: DeviceVec,
    /// The audio's sigma's sinusoids.
    t: DeviceVec,
    x: DeviceVec,
    h: DeviceVec,
    cm: DeviceVec,
    y: DeviceVec,
    f: DeviceVec,
    fg: DeviceVec,
    emb: DeviceVec,
    /// An embedder's embedding nobody reads.
    spare: DeviceVec,
    modulation: DeviceVec,
    prompt: DeviceVec,
    mm: DeviceVec,
    pm: DeviceVec,
    /// The attention between the streams: the step's rows (the video's a set a timestep) and gates, a set's scratch,
    /// then each with the block's part.
    video_rows: DeviceVec,
    audio_rows: DeviceVec,
    video_gate: DeviceVec,
    audio_gate: DeviceVec,
    rows_set: DeviceVec,
    video_mod: DeviceVec,
    audio_mod: DeviceVec,
    video_by: DeviceVec,
    audio_by: DeviceVec,
    vel: DeviceVec,
}

impl Sound {
    fn new(m: &WgpuLtx, g: &AudioStreams, a: &AudioPass<'_>, _tokens: usize, sets: usize) -> Self {
        let v = |len: usize| m.vec(len);
        let (ta, la) = (a.tokens, a.rows);
        let ff = m.blocks.first().and_then(|b| b.audio.as_ref()).map_or(4 * A, |b| b.ff0.n);
        let (lat, ctx, t) = (v(a.latent.len()), v(a.context.len()), v(256));
        m.gpu.upload(&lat, a.latent);
        m.gpu.upload(&ctx, a.context);
        m.gpu.upload(&t, &sinusoids(a.sigma));
        Sound {
            lat,
            ctx,
            t,
            x: v(ta * A),
            h: v(ta * A),
            cm: v(la * A),
            y: v(ta * A),
            f: v(ta * ff),
            fg: v(ta * ff),
            emb: v(A),
            spare: v(D),
            modulation: v(9 * A),
            prompt: v(2 * A),
            mm: v(9 * A),
            pm: v(2 * A),
            video_rows: v(sets * 4 * D),
            audio_rows: v(4 * A),
            video_gate: v(D),
            audio_gate: v(A),
            rows_set: v(4 * D),
            video_mod: v(sets * 4 * D),
            audio_mod: v(4 * A),
            video_by: v(D),
            audio_by: v(A),
            vel: v(ta * g.proj_out.n),
        }
    }
}

/// One attention: `xq`'s `tq` rows' queries over `xkv`'s `tk` rows (the queries rotated by `rope_q` and the keys by
/// `rope_k` where given: the same table for a stream over itself, each stream's own between the two), the heads gated
/// where the attention has a gate, out through the output projection into `y`. Its width is its q's (the video's
/// 4,096, the audio's and the streams' between them 2,048), its inputs' and its output's their layers'. With
/// `passthrough` the attention is its value projection (the reference's perturbation for spatio-temporal guidance),
/// its gate and output still applied. `s` the scratch.
#[allow(clippy::too_many_arguments)]
fn attend(r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, xkv: &DeviceVec, tk: usize, rope_q: Option<&DeviceVec>, rope_k: Option<&DeviceVec>, passthrough: bool, s: &Scratch, y: &DeviceVec) {
    let width = a.q.n;
    let hd = width / HEADS;
    let gate_out = |r: &mut dyn ChainRecorder| {
        if let Some(gate) = &a.gate {
            gate.forward(r, xq, &s.logits, tq);
            r.head_gate_rows(&s.att, &s.logits, tq, HEADS, hd);
        }
        a.out.forward(r, &s.att, y, tq);
    };
    if passthrough {
        a.v.forward(r, xkv, &s.att, tk);
        return gate_out(r);
    }
    // (the norms take a row's width from their vectors' lengths: views of the scratch's first rows)
    let first = |v: &DeviceVec, rows: usize| DeviceVec { len: rows * width, inner: v.inner.clone() };
    a.q.forward(r, xq, &s.q, tq);
    r.rmsnorm_rows(&first(&s.q, tq), &a.qn, &first(&s.qn, tq), tq, EPS);
    a.k.forward(r, xkv, &s.k, tk);
    r.rmsnorm_rows(&first(&s.k, tk), &a.kn, &first(&s.kn, tk), tk, EPS);
    a.v.forward(r, xkv, &s.v, tk);
    if let Some(t) = rope_q {
        r.rope_split_rows(&s.qn, tq, HEADS, hd, t);
    }
    if let Some(t) = rope_k {
        r.rope_split_rows(&s.kn, tk, HEADS, hd, t);
    }
    r.store_rows(&s.kn, &s.kv, tk, width, 0, 2 * width, 0);
    r.store_rows(&s.v, &s.kv, tk, width, 0, 2 * width, width);
    r.attention_rows_full(&s.qn, &s.kv, &s.att, tq, HEADS, HEADS, hd, tk, 1.0 / (hd as f32).sqrt());
    gate_out(r);
}

/// A text cross-attention guided away from a negative context ([`Nag`]; the reference's `nag_attn`): `xq`'s `tq`
/// rows' queries over the prompt's context and over the negative one (`(rows, count)` each), the two outputs mixed in
/// `plain`, then the heads' gate and the output projection into `y`.
#[allow(clippy::too_many_arguments)]
fn attend_guided(r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, positive: (&DeviceVec, usize), negative: (&DeviceVec, usize), nag: &Nag, s: &Scratch, plain: &DeviceVec, y: &DeviceVec) {
    let first = |v: &DeviceVec, rows: usize| DeviceVec { len: rows * D, inner: v.inner.clone() };
    a.q.forward(r, xq, &s.q, tq);
    r.rmsnorm_rows(&first(&s.q, tq), &a.qn, &first(&s.qn, tq), tq, EPS);
    for (which, (xkv, tk)) in [positive, negative].into_iter().enumerate() {
        a.k.forward(r, xkv, &s.k, tk);
        r.rmsnorm_rows(&first(&s.k, tk), &a.kn, &first(&s.kn, tk), tk, EPS);
        a.v.forward(r, xkv, &s.v, tk);
        r.store_rows(&s.kn, &s.kv, tk, D, 0, 2 * D, 0);
        r.store_rows(&s.v, &s.kv, tk, D, 0, 2 * D, D);
        r.attention_rows_full(&s.qn, &s.kv, &s.att, tq, HEADS, HEADS, HD, tk, 1.0 / (HD as f32).sqrt());
        if which == 0 {
            r.copy(&s.att, 0, plain, 0, tq * D);
        }
    }
    r.nag_mix(plain, &s.att, tq, D, nag.scale, nag.tau, nag.alpha);
    if let Some(gate) = &a.gate {
        gate.forward(r, xq, &s.logits, tq);
        r.head_gate_rows(plain, &s.logits, tq, HEADS, HD);
    }
    a.out.forward(r, plain, y, tq);
}

/// An attention's vectors, for `rows` rows at the most.
struct Scratch {
    q: DeviceVec,
    qn: DeviceVec,
    k: DeviceVec,
    kn: DeviceVec,
    v: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    logits: DeviceVec,
}

impl Scratch {
    fn new(gpu: &ggml_rs_wgpu::WgpuBackend, rows: usize) -> Self {
        let v = |len: usize| gpu.vec(len.max(1));
        Self {
            q: v(rows * D),
            qn: v(rows * D),
            k: v(rows * D),
            kn: v(rows * D),
            v: v(rows * D),
            kv: v(rows * 2 * D),
            // (the video's heads, or the audio's of half the width: whichever's scratch is the longer)
            att: v(gpu.attention_rows_full_out_len(rows, HEADS, HD, rows).max(gpu.attention_rows_full_out_len(rows, HEADS, A / HEADS, rows))),
            logits: v(rows * HEADS),
        }
    }
}

/// One of a text connector's blocks: its gated self-attention and its feed-forward.
pub struct ConnectorBlock {
    attn1: Attn,
    ff0: Linear,
    ff2: Linear,
}

impl ConnectorBlock {
    pub fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        Ok(Self { attn1: Attn::load(store, gpu, &format!("{p}.attn1"))?, ff0: Linear::load(store, gpu, &format!("{p}.ff.net.0.proj"))?, ff2: Linear::load(store, gpu, &format!("{p}.ff.net.2"))? })
    }
}

/// The text connector over `x` (`rows` of its blocks' width: the video's 4,096 or the audio's 2,048; changed in place)
/// as [`crate::ltx::transformer::connector`] runs it: each block RMS-normed (no weights) into its self-attention
/// (rotated by `table`) and its feed-forward, each added back; the result over its RMS (a new vector).
pub fn connector(gpu: &ggml_rs_wgpu::WgpuBackend, r: &mut dyn ChainRecorder, blocks: &[ConnectorBlock], x: &DeviceVec, rows: usize, table: &DeviceVec) -> DeviceVec {
    let s = Scratch::new(gpu, rows);
    let (d, ff) = blocks.first().map_or((D, 4 * D), |b| (b.attn1.q.k, b.ff0.n));
    let ones = gpu.vec(d);
    gpu.upload(&ones, &vec![1.0; d]);
    let v = |len: usize| gpu.vec(len.max(1));
    let (h, y, f, fg, out) = (v(rows * d), v(rows * d), v(rows * ff), v(rows * ff), v(rows * d));
    for b in blocks {
        r.rmsnorm_rows(x, &ones, &h, rows, EPS);
        attend(r, &b.attn1, &h, rows, &h, rows, Some(table), Some(table), false, &s, &y);
        r.add(x, &y);
        r.rmsnorm_rows(x, &ones, &h, rows, EPS);
        b.ff0.forward(r, &h, &f, rows);
        r.gelu(&f, &fg, rows * b.ff0.n);
        b.ff2.forward(r, &fg, &y, rows);
        r.add(x, &y);
    }
    r.rmsnorm_rows(x, &ones, &out, rows, EPS);
    out
}

#[cfg(test)]
mod tests;
