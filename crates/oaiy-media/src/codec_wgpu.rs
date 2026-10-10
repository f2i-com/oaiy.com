//! The Qwen3-TTS 12 Hz speech codec's decoder on WebGPU (a speech job's `backend` "webgpu": any GPU), as
//! [`oaiy_tts::codec::CodecDecoder`] runs it: its features as rows (`[steps, channels]`), F32 but for the 1-D
//! convolutions' weights (f16, as [`DeviceChain::conv1d_weights`] packs them). A frame's 16 codebook rows are summed on
//! the host (the first codebook's, then the other 15's), their two output projections one matrix; then the causal
//! convolutions ([`ChainRecorder::conv1d_padded_rows`], `(k - 1) dilation` zeros before), the transformer's 72-frame
//! sliding-window attention and layer scales, the ConvNeXt upsamplers, SnakeBeta, and the transposed convolutions
//! trimmed on the right ([`ChainRecorder::conv_transpose1d_rows`]'s first `len stride` steps), in the official
//! 300-frame chunks with 25 frames of left context.
use crate::ltx::store::Store;
use candle_core::{Device, Result};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use ggml_rs_wgpu::WgpuBackend;
use oaiy_engine::json::Json;
use oaiy_tts::codec::{CodecConfig, CHUNK, LEFT_CONTEXT, SAMPLES_PER_FRAME};
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("the speech codec on WebGPU: {e}"))
}

/// A tensor's values (F32) and shape.
fn host(s: &mut Store, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let t = s.tensor_f32(key, &Device::Cpu)?;
    Ok((t.flatten_all()?.to_vec1::<f32>()?, t.dims().to_vec()))
}

fn upload(gpu: &WgpuBackend, v: &[f32]) -> DeviceVec {
    let d = gpu.vec(v.len());
    gpu.upload(&d, v);
    d
}

fn vector(s: &mut Store, gpu: &WgpuBackend, key: &str) -> Result<DeviceVec> {
    Ok(upload(gpu, &host(s, key)?.0))
}

/// A linear layer in F32: its weight `[n, k]` and bias.
struct Lin {
    w: DeviceVec,
    b: Option<DeviceVec>,
    n: usize,
    k: usize,
}

impl Lin {
    fn load(s: &mut Store, gpu: &WgpuBackend, prefix: &str) -> Result<Self> {
        let (w, shape) = host(s, &format!("{prefix}.weight"))?;
        let &[n, k] = shape.as_slice() else { return Err(err(format!("{prefix}: a matrix of {shape:?}"))) };
        let bias = format!("{prefix}.bias");
        let b = if s.index.get(&bias).is_some() { Some(vector(s, gpu, &bias)?) } else { None };
        Ok(Self { w: upload(gpu, &w), b, n, k })
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f32_rows(&self.w, self.n, self.k, x, y, rows);
        if let Some(b) = &self.b {
            r.add_bias_rows(y, b, rows, self.n);
        }
    }
}

/// A causal 1-D convolution: its weights as [`DeviceChain::conv1d_weights`] packs them, its bias, and its shape.
struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cout: usize,
    cin: usize,
    k: usize,
}

impl Conv {
    fn load(s: &mut Store, gpu: &WgpuBackend, prefix: &str) -> Result<Self> {
        let (w, shape) = host(s, &format!("{prefix}.weight"))?;
        let &[cout, cin, k] = shape.as_slice() else { return Err(err(format!("{prefix}: a convolution of {shape:?}"))) };
        let w = gpu.conv1d_weights(&w, cout, cin, k).ok_or_else(|| err(format!("{prefix} past f16's range")))?;
        Ok(Self { w, b: vector(s, gpu, &format!("{prefix}.bias"))?, cout, cin, k })
    }

    /// `x`'s `len` steps (`[len, cin]`) into `y` (`[len, cout]`): tap `t` of step `s` from step `s - (k - 1 - t)
    /// dilation`, zeros before the first.
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, len: usize, dilation: usize, y: &DeviceVec) {
        r.conv1d_padded_rows(&self.w, &self.b, self.cout, self.cin, self.k, dilation, (self.k - 1) * dilation, x, len, y);
    }
}

/// A causal transposed convolution: its weight as `[cout, k, cin]` (f32, PyTorch's `[cin, cout, k]` rearranged), its
/// bias, and its stride.
struct Up {
    w: DeviceVec,
    b: DeviceVec,
    cout: usize,
    cin: usize,
    k: usize,
    stride: usize,
}

impl Up {
    fn load(s: &mut Store, gpu: &WgpuBackend, prefix: &str, stride: Option<usize>) -> Result<Self> {
        let (w, shape) = host(s, &format!("{prefix}.weight"))?;
        let &[cin, cout, k] = shape.as_slice() else { return Err(err(format!("{prefix}: a transposed convolution of {shape:?}"))) };
        let packed: Vec<f32> = (0..cout * k * cin)
            .map(|i| {
                let (co, j, c) = (i / (k * cin), (i / cin) % k, i % cin);
                w[(c * cout + co) * k + j]
            })
            .collect();
        Ok(Self { w: upload(gpu, &packed), b: vector(s, gpu, &format!("{prefix}.bias"))?, cout, cin, k, stride: stride.unwrap_or(k) })
    }

    /// `x`'s `len` steps into a new `[len stride, cout]`: PyTorch's whole output less its last `k - stride` steps.
    fn run(&self, gpu: &WgpuBackend, r: &mut dyn ChainRecorder, x: &DeviceVec, len: usize) -> DeviceVec {
        let out = len * self.stride;
        let y = gpu.vec(out * self.cout);
        r.conv_transpose1d_rows(&self.w, &self.b, self.cout, self.cin, self.k, self.stride, 0, x, len, out, &y);
        y
    }
}

/// SnakeBeta's `e^alpha` and `1 / (e^beta + 1e-9)`, a channel's.
struct Snake {
    freq: DeviceVec,
    scale: DeviceVec,
    c: usize,
}

impl Snake {
    fn load(s: &mut Store, gpu: &WgpuBackend, prefix: &str) -> Result<Self> {
        let (alpha, beta) = (host(s, &format!("{prefix}.alpha"))?.0, host(s, &format!("{prefix}.beta"))?.0);
        let freq: Vec<f32> = alpha.iter().map(|a| a.exp()).collect();
        let scale: Vec<f32> = beta.iter().map(|b| 1. / (b.exp() + 1e-9)).collect();
        Ok(Self { freq: upload(gpu, &freq), scale: upload(gpu, &scale), c: alpha.len() })
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, rows: usize) {
        r.snake_beta_rows(x, &self.freq, &self.scale, rows, self.c);
    }
}

struct Layer {
    input_norm: DeviceVec,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    attn_scale: DeviceVec,
    post_norm: DeviceVec,
    gate: Lin,
    up: Lin,
    down: Lin,
    mlp_scale: DeviceVec,
}

/// A ConvNeXt block: a depthwise causal convolution (`[c, k]`), a layer norm (its weight less one, then its bias), the
/// pointwise MLP, and the scale of what it adds.
struct ConvNext {
    dw: DeviceVec,
    dw_b: DeviceVec,
    k: usize,
    norm: DeviceVec,
    pw1: Lin,
    pw2: Lin,
    gamma: DeviceVec,
}

/// A residual unit: SnakeBeta, a dilated causal convolution, SnakeBeta, a pointwise one.
struct Unit {
    act1: Snake,
    conv1: Conv,
    dilation: usize,
    act2: Snake,
    conv2: Conv,
}

/// A vocoder block: SnakeBeta, the transposed convolution up by its rate, three residual units.
struct Block {
    snake: Snake,
    up: Up,
    units: Vec<Unit>,
}

pub struct WgpuCodec {
    gpu: WgpuBackend,
    /// The 16 codebooks (`[2048, dim]` each, on the host: a frame's rows are summed there).
    codebooks: Vec<Vec<f32>>,
    dim: usize,
    /// The first codebook's output projection and the rest's side by side (`[512, 2 dim]`).
    project: Lin,
    pre_conv: Conv,
    input_proj: Lin,
    layers: Vec<Layer>,
    norm: DeviceVec,
    output_proj: Lin,
    upsample: Vec<(Up, ConvNext)>,
    first: Conv,
    blocks: Vec<Block>,
    last_snake: Snake,
    last: Conv,
    heads: usize,
    head_dim: usize,
    window: usize,
    theta: f64,
    eps: f32,
    one: DeviceVec,
    /// Frames the stages after the transformer reach back (their causal convolutions, through each upsampling): a
    /// frame's sound is settled by the transformer's output for it and these frames before it.
    reach: usize,
}

/// Codes a codebook holds.
const CODES: usize = 2048;

/// Frames the stages after the transformer reach back, from their shapes: walked from the waveform back to the frames,
/// each causal convolution adds `(k - 1) dilation` steps at its rate, and each transposed convolution (stride `s`,
/// `k` taps) turns `n` steps after it into `(n + k - 1) / s` before it, rounded up and one more. Counted generously:
/// a frame too many costs a little time, a frame too few changes the sound.
fn vocoder_reach(upsample: &[(Up, ConvNext)], first_k: usize, blocks: &[Block], last_k: usize) -> usize {
    let up = |n: usize, u: &Up| (n + u.k - 1).div_ceil(u.stride) + 1;
    let mut n = last_k - 1;
    for b in blocks.iter().rev() {
        n += b.units.iter().map(|u| (u.conv1.k - 1) * u.dilation + (u.conv2.k - 1)).sum::<usize>();
        n = up(n, &b.up);
    }
    n += first_k - 1;
    for (u, c) in upsample.iter().rev() {
        n += c.k - 1;
        n = up(n, u);
    }
    n + 1
}

impl WgpuCodec {
    /// `decoder.*` of a speech tokenizer's `model.safetensors` (its `config.json` beside it) onto `gpu`.
    pub fn load(path: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let cfg_path = path.parent().map(|p| p.join("config.json")).ok_or_else(|| err("the speech tokenizer's path has no folder"))?;
        let config = Json::parse(&std::fs::read(&cfg_path)?).map_err(candle_core::Error::wrap)?;
        let cfg = CodecConfig::from_json(&config)?;
        let eps = config.get("decoder_config").and_then(|c| c.get("rms_norm_eps")).and_then(Json::as_f64).unwrap_or(1e-5) as f32;
        let mut store = Store::open(path, 0)?;
        let s = &mut store;
        // the codebooks: each code's summed embedding over its use
        let codebook = |s: &mut Store, group: &str, i: usize| -> Result<(Vec<f32>, usize)> {
            let base = format!("decoder.quantizer.{group}.vq.layers.{i}._codebook");
            let (sum, shape) = host(s, &format!("{base}.embedding_sum"))?;
            let (usage, _) = host(s, &format!("{base}.cluster_usage"))?;
            let &[codes, dim] = shape.as_slice() else { return Err(err(format!("{base}: {shape:?}"))) };
            if codes != CODES || usage.len() != codes {
                return Err(err(format!("{base}: {codes} codes, not {CODES}")));
            }
            Ok((sum.iter().enumerate().map(|(i, v)| v / usage[i / dim].max(1e-5)).collect(), dim))
        };
        let (first, dim) = codebook(s, "rvq_first", 0)?;
        let mut codebooks = vec![first];
        for i in 0..cfg.quantizers - 1 {
            codebooks.push(codebook(s, "rvq_rest", i)?.0);
        }
        if codebooks.len() != 16 {
            return Err(err(format!("{} codebooks, not 16", codebooks.len())));
        }
        let project = {
            let (a, shape) = host(s, "decoder.quantizer.rvq_first.output_proj.weight")?;
            let (b, _) = host(s, "decoder.quantizer.rvq_rest.output_proj.weight")?;
            let n = shape[0];
            if a.len() != n * dim || b.len() != n * dim {
                return Err(err(format!("the output projections: {shape:?}")));
            }
            let w: Vec<f32> = (0..n).flat_map(|o| a[o * dim..(o + 1) * dim].iter().chain(&b[o * dim..(o + 1) * dim]).copied()).collect();
            Lin { w: upload(gpu, &w), b: None, n, k: 2 * dim }
        };
        let p = "decoder.pre_transformer";
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("{p}.layers.{i}");
            layers.push(Layer {
                input_norm: vector(s, gpu, &format!("{l}.input_layernorm.weight"))?,
                q: Lin::load(s, gpu, &format!("{l}.self_attn.q_proj"))?,
                k: Lin::load(s, gpu, &format!("{l}.self_attn.k_proj"))?,
                v: Lin::load(s, gpu, &format!("{l}.self_attn.v_proj"))?,
                o: Lin::load(s, gpu, &format!("{l}.self_attn.o_proj"))?,
                attn_scale: vector(s, gpu, &format!("{l}.self_attn_layer_scale.scale"))?,
                post_norm: vector(s, gpu, &format!("{l}.post_attention_layernorm.weight"))?,
                gate: Lin::load(s, gpu, &format!("{l}.mlp.gate_proj"))?,
                up: Lin::load(s, gpu, &format!("{l}.mlp.up_proj"))?,
                down: Lin::load(s, gpu, &format!("{l}.mlp.down_proj"))?,
                mlp_scale: vector(s, gpu, &format!("{l}.mlp_layer_scale.scale"))?,
            });
        }
        let width = layers.first().map_or(0, |l| l.q.n);
        if width % cfg.heads != 0 || layers.iter().any(|l| l.k.n != width || l.v.n != width) {
            return Err(err("the transformer's heads are not its configuration's"));
        }
        let mut upsample = Vec::new();
        let mut i = 0;
        while s.index.get(&format!("decoder.upsample.{i}.0.conv.weight")).is_some() {
            let u = format!("decoder.upsample.{i}");
            let up = Up::load(s, gpu, &format!("{u}.0.conv"), None)?;
            let b = format!("{u}.1");
            let (dw, shape) = host(s, &format!("{b}.dwconv.conv.weight"))?;
            let norm: Vec<f32> = host(s, &format!("{b}.norm.weight"))?.0.iter().map(|v| v - 1.).chain(host(s, &format!("{b}.norm.bias"))?.0).collect();
            upsample.push((
                up,
                ConvNext {
                    dw: upload(gpu, &dw),
                    dw_b: vector(s, gpu, &format!("{b}.dwconv.conv.bias"))?,
                    k: shape[2],
                    norm: upload(gpu, &norm),
                    pw1: Lin::load(s, gpu, &format!("{b}.pwconv1"))?,
                    pw2: Lin::load(s, gpu, &format!("{b}.pwconv2"))?,
                    gamma: vector(s, gpu, &format!("{b}.gamma"))?,
                },
            ));
            i += 1;
        }
        let v = "decoder.decoder";
        let mut blocks = Vec::new();
        for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
            let b = format!("{v}.{}.block", i + 1);
            let mut units = Vec::new();
            for (j, dilation) in [1, 3, 9].into_iter().enumerate() {
                let u = format!("{b}.{}", j + 2);
                units.push(Unit {
                    act1: Snake::load(s, gpu, &format!("{u}.act1"))?,
                    conv1: Conv::load(s, gpu, &format!("{u}.conv1.conv"))?,
                    dilation,
                    act2: Snake::load(s, gpu, &format!("{u}.act2"))?,
                    conv2: Conv::load(s, gpu, &format!("{u}.conv2.conv"))?,
                });
            }
            blocks.push(Block { snake: Snake::load(s, gpu, &format!("{b}.0"))?, up: Up::load(s, gpu, &format!("{b}.1.conv"), Some(rate))?, units });
        }
        let n = cfg.upsample_rates.len();
        let mut codec = Self {
            gpu: gpu.clone(),
            codebooks,
            dim,
            project,
            pre_conv: Conv::load(s, gpu, "decoder.pre_conv.conv")?,
            input_proj: Lin::load(s, gpu, &format!("{p}.input_proj"))?,
            layers,
            norm: vector(s, gpu, &format!("{p}.norm.weight"))?,
            output_proj: Lin::load(s, gpu, &format!("{p}.output_proj"))?,
            upsample,
            first: Conv::load(s, gpu, &format!("{v}.0.conv"))?,
            blocks,
            last_snake: Snake::load(s, gpu, &format!("{v}.{}", n + 1))?,
            last: Conv::load(s, gpu, &format!("{v}.{}.conv", n + 2))?,
            heads: cfg.heads,
            head_dim: width / cfg.heads,
            window: cfg.window,
            theta: cfg.rope_theta,
            eps,
            one: upload(gpu, &[1.0]),
            reach: 0,
        };
        codec.reach = vocoder_reach(&codec.upsample, codec.first.k, &codec.blocks, codec.last.k);
        Ok(codec)
    }

    /// Frames of 16 codes to a 24 kHz waveform, in the official 300-frame chunks with 25 frames of left context.
    pub fn decode(&self, frames: &[[u32; 16]]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(frames.len() * SAMPLES_PER_FRAME);
        let mut start = 0;
        while start < frames.len() {
            let end = (start + CHUNK).min(frames.len());
            let context = start.min(LEFT_CONTEXT);
            let wave = self.forward(&frames[start - context..end])?;
            out.extend_from_slice(&wave[context * SAMPLES_PER_FRAME..]);
            start = end;
        }
        Ok(out)
    }

    /// The samples of `new` (1920 a frame), which follow `before`: decoded with the last [`LEFT_CONTEXT`] frames of
    /// `before` ahead of them, as [`Self::decode`] reads each of its chunks. A stream's chunk: every stage is causal,
    /// so what a frame sounds like is settled once it is drawn, and the context is what the stages reach back for.
    pub fn decode_after(&self, before: &[[u32; 16]], new: &[[u32; 16]]) -> Result<Vec<f32>> {
        self.decode_after_with(before, new, LEFT_CONTEXT)
    }

    /// [`Self::decode_after`] with `context` frames of `before` read ahead of `new` by the transformer (the stages
    /// after it run on `new` and the frames they reach back for alone: [`Self::forward_tail`]).
    pub fn decode_after_with(&self, before: &[[u32; 16]], new: &[[u32; 16]], context: usize) -> Result<Vec<f32>> {
        if new.is_empty() {
            return Ok(Vec::new());
        }
        let context = before.len().min(context);
        let mut frames = before[before.len() - context..].to_vec();
        frames.extend_from_slice(new);
        self.forward_tail(&frames, new.len())
    }

    /// One whole decode of `frames` (`frames.len() * 1920` samples).
    fn forward(&self, frames: &[[u32; 16]]) -> Result<Vec<f32>> {
        let x = upload(&self.gpu, &self.rows(frames)?);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let y = self.record(rec.as_mut(), &x, frames.len(), &mut |_, _, _| {});
        rec.as_mut().read(&y);
        rec.finish().pop().ok_or_else(|| err("the waveform was not read"))
    }

    /// The samples of `frames`' last `tail` frames: the transformer reads every frame, and the stages after it (whose
    /// cost is nearly all of a decode's: they run at up to 1920 steps a frame) only the tail and the [`Self::reach`]
    /// frames before it, which is all they look back at. The same samples as [`Self::forward`]'s for the tail.
    fn forward_tail(&self, frames: &[[u32; 16]], tail: usize) -> Result<Vec<f32>> {
        let t = frames.len();
        let kept = (tail + self.reach).min(t);
        let x = upload(&self.gpu, &self.rows(frames)?);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let mut h = self.front(rec.as_mut(), &x, t, &mut |_, _, _| {});
        if kept < t {
            let c = self.output_proj.n;
            let part = self.gpu.vec(kept * c);
            rec.as_mut().copy(&h, (t - kept) * c, &part, 0, kept * c);
            h = part;
        }
        let y = self.back(rec.as_mut(), h, kept, &mut |_, _, _| {});
        rec.as_mut().read(&y);
        let wave = rec.finish().pop().ok_or_else(|| err("the waveform was not read"))?;
        Ok(wave[(kept - tail.min(kept)) * SAMPLES_PER_FRAME..].to_vec())
    }

    /// Each frame's codebook rows (`[frames, 2 dim]`): the first codebook's, then the sum of the other 15's.
    fn rows(&self, frames: &[[u32; 16]]) -> Result<Vec<f32>> {
        let d = self.dim;
        let mut rows = vec![0f32; frames.len() * 2 * d];
        for (f, row) in frames.iter().zip(rows.chunks_exact_mut(2 * d)) {
            for (q, &code) in f.iter().enumerate() {
                let code = code as usize;
                if code >= CODES {
                    return Err(err(format!("code {code} of codebook {q}, past its {CODES}")));
                }
                let at = if q == 0 { 0 } else { d };
                for (a, b) in row[at..at + d].iter_mut().zip(&self.codebooks[q][code * d..(code + 1) * d]) {
                    *a += b;
                }
            }
        }
        Ok(rows)
    }

    /// A whole decode recorded on `r` from the frames' codebook rows `x` ([`Self::rows`]): the waveform (`[t 1920]`,
    /// clamped), each stage's output shown to `tap` (by the reference's name for it) as it is made.
    fn record(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, t: usize, tap: &mut dyn FnMut(&mut dyn ChainRecorder, &str, &DeviceVec)) -> DeviceVec {
        let h = self.front(r, x, t, tap);
        self.back(r, h, t, tap)
    }

    /// The stages up to the transformer's output (`[t, 1024]`), recorded on `r` from the codebook rows `x`.
    fn front(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, t: usize, tap: &mut dyn FnMut(&mut dyn ChainRecorder, &str, &DeviceVec)) -> DeviceVec {
        let g = &self.gpu;
        let q = g.vec(t * self.project.n);
        self.project.run(r, x, &q, t);
        tap(r, "quantized", &q);
        let h = g.vec(t * self.pre_conv.cout);
        self.pre_conv.run(r, &q, t, 1, &h);
        tap(r, "pre_conv", &h);
        let h = self.transformer(r, &h, t);
        tap(r, "pre_transformer", &h);
        h
    }

    /// The stages after the transformer, from its output for `t` frames (`h`, `[t, 1024]`) to their waveform.
    fn back(&self, r: &mut dyn ChainRecorder, mut h: DeviceVec, t: usize, tap: &mut dyn FnMut(&mut dyn ChainRecorder, &str, &DeviceVec)) -> DeviceVec {
        let g = &self.gpu;
        let mut len = t;
        for (i, (up, block)) in self.upsample.iter().enumerate() {
            h = up.run(g, r, &h, len);
            len *= up.stride;
            self.convnext(r, block, &h, len);
            tap(r, &format!("upsample{i}"), &h);
        }
        let y = g.vec(len * self.first.cout);
        self.first.run(r, &h, len, 1, &y);
        h = y;
        tap(r, "decoder0", &h);
        for (i, b) in self.blocks.iter().enumerate() {
            b.snake.run(r, &h, len);
            h = b.up.run(g, r, &h, len);
            len *= b.up.stride;
            let c = b.up.cout;
            let (u, v) = (g.vec(len * c), g.vec(len * c));
            for unit in &b.units {
                r.copy(&h, 0, &u, 0, len * c);
                unit.act1.run(r, &u, len);
                unit.conv1.run(r, &u, len, unit.dilation, &v);
                unit.act2.run(r, &v, len);
                unit.conv2.run(r, &v, len, 1, &u);
                r.axpy_at(&h, &u, &self.one, 0, len * c);
            }
            tap(r, &format!("decoder{}", i + 1), &h);
        }
        self.last_snake.run(r, &h, len);
        let y = g.vec(len * self.last.cout);
        self.last.run(r, &h, len, 1, &y);
        r.clamp_in_place(&y, len * self.last.cout, -1., 1.);
        y
    }

    /// input_proj, the layers (sliding-window attention and the MLP, each scaled), the final norm, output_proj: `x`
    /// (`[t, 1024]`) to a new `[t, 1024]`.
    fn transformer(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, t: usize) -> DeviceVec {
        let g = &self.gpu;
        let (hd, nh) = (self.head_dim, self.heads);
        let (hidden, width) = (self.input_proj.n, nh * hd);
        // rotate-half RoPE's table for positions 0..t: each position's (sin, cos) a pair
        let table: Vec<f32> = (0..t)
            .flat_map(|p| (0..hd / 2).flat_map(move |i| {
                let a = p as f64 / self.theta.powf(2. * i as f64 / hd as f64);
                [a.sin() as f32, a.cos() as f32]
            }))
            .collect();
        let td = upload(g, &table);
        let v = |n: usize| g.vec(n);
        let ff = self.layers.first().map_or(0, |l| l.gate.n);
        let (h, n, q, k, vv, kv, att, o) = (v(t * hidden), v(t * hidden), v(t * width), v(t * width), v(t * width), v(t * 2 * width), v(g.attention_rows_out_len(t, nh, hd, t)), v(t * hidden));
        let (gate, up, act) = (v(t * ff), v(t * ff), v(t * ff));
        self.input_proj.run(r, x, &h, t);
        for l in &self.layers {
            r.rmsnorm_rows(&h, &l.input_norm, &n, t, self.eps);
            l.q.run(r, &n, &q, t);
            l.k.run(r, &n, &k, t);
            l.v.run(r, &n, &vv, t);
            r.rope_rows(&q, t, nh, hd, &td, true);
            r.rope_rows(&k, t, nh, hd, &td, true);
            r.store_rows(&k, &kv, t, width, 0, 2 * width, 0);
            r.store_rows(&vv, &kv, t, width, 0, 2 * width, width);
            r.attention_rows(&q, &kv, &att, t, nh, nh, hd, 0, Some(self.window), 1. / (hd as f32).sqrt());
            l.o.run(r, &att, &o, t);
            r.add_gated_rows(&h, &o, t, hidden, &l.attn_scale, 0, false);
            r.rmsnorm_rows(&h, &l.post_norm, &n, t, self.eps);
            l.gate.run(r, &n, &gate, t);
            l.up.run(r, &n, &up, t);
            r.silu_mul(&gate, &up, &act, t * ff);
            l.down.run(r, &act, &o, t);
            r.add_gated_rows(&h, &o, t, hidden, &l.mlp_scale, 0, false);
        }
        r.rmsnorm_rows(&h, &self.norm, &n, t, self.eps);
        let y = v(t * self.output_proj.n);
        self.output_proj.run(r, &n, &y, t);
        y
    }

    /// A ConvNeXt block on `x` (`[len, c]`) in place: the depthwise causal convolution, the layer norm, the pointwise
    /// MLP (an exact GELU), times gamma, added.
    fn convnext(&self, r: &mut dyn ChainRecorder, b: &ConvNext, x: &DeviceVec, len: usize) {
        let g = &self.gpu;
        let (c, f) = (b.pw2.n, b.pw1.n);
        let (y, n, h, hg) = (g.vec(len * c), g.vec(len * c), g.vec(len * f), g.vec(len * f));
        r.depthwise_causal_conv1d_rows(&b.dw, &b.dw_b, c, b.k, x, len, &y);
        r.norm_mod_rows(&y, &n, len, c, &b.norm, 0, Some(c), RowNorm::Layer, 1e-6);
        b.pw1.run(r, &n, &h, len);
        r.gelu_erf(&h, &hg, len * f);
        b.pw2.run(r, &hg, &y, len);
        r.add_gated_rows(x, &y, len, c, &b.gamma, 0, false);
    }
}

#[cfg(test)]
mod golden {
    use super::*;

    fn relative(a: &[f32], b: &[f32]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>()).sqrt()
    }

    /// The decoder on WebGPU against the official implementation's F32 stages for `sampled_codes.i32` (90 frames;
    /// `--ignored --nocapture`; OAIY_TTS_GOLDEN, else E:/deepseek/nrob/target/qwen-tts-golden; OAIY_TTS_MODEL, else the
    /// 1.7B VoiceDesign): each stage's relative RMS error, and the decode's time.
    #[test]
    #[ignore = "needs Qwen3-TTS's speech tokenizer and the reference's dumps"]
    fn the_webgpu_codec_is_the_references() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("OAIY_TTS_GOLDEN").unwrap_or_else(|_| "E:/deepseek/nrob/target/qwen-tts-golden".into()));
        let model = std::path::PathBuf::from(std::env::var("OAIY_TTS_MODEL").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign".into()));
        let f32s = |name: &str| -> Vec<f32> { std::fs::read(root.join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
        let codes: Vec<u32> = std::fs::read(root.join("sampled_codes.i32"))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect();
        let frames: Vec<[u32; 16]> = codes.chunks_exact(16).map(|c| c.try_into().unwrap()).collect();
        let t = frames.len();
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let codec = WgpuCodec::load(&model.join("speech_tokenizer").join("model.safetensors"), &gpu)?;
        // each stage's output copied as it is made (later stages work in place)
        let x = upload(&gpu, &codec.rows(&frames)?);
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let mut names = Vec::new();
        let y = codec.record(rec.as_mut(), &x, t, &mut |r, name, v| {
            if root.join(format!("dec_{name}.f32")).exists() {
                let c = gpu.vec(v.len);
                r.copy(v, 0, &c, 0, v.len);
                r.read(&c);
                names.push(name.to_string());
            }
        });
        rec.as_mut().read(&y);
        let mut got = rec.finish();
        names.push("wave".into());
        let mut worst = 0f64;
        for (name, ours) in names.iter().zip(&mut got) {
            let want = f32s(&format!("dec_{name}.f32"));
            assert_eq!(ours.len(), want.len(), "{name}");
            // the reference's convolutions' outputs channels first
            if name != "pre_transformer" && name != "wave" {
                let steps = match name.as_str() {
                    "quantized" | "pre_conv" => t,
                    n if n.starts_with("upsample") => t << (n[8..].parse::<usize>().unwrap() + 1),
                    _ => 4 * t * codec.blocks.iter().take(name[7..].parse::<usize>().unwrap()).map(|b| b.up.stride).product::<usize>(),
                };
                let c = ours.len() / steps;
                *ours = (0..ours.len()).map(|i| ours[(i % steps) * c + i / steps]).collect();
            }
            let e = relative(ours, &want);
            eprintln!("{name}: relative RMS error {e:.3e}");
            worst = worst.max(e);
        }
        let started = std::time::Instant::now();
        let wave = codec.decode(&frames)?;
        eprintln!("{t} frames decoded in {:.3} s ({} samples)", started.elapsed().as_secs_f64(), wave.len());
        assert!(worst < 1e-3, "{worst}");
        Ok(())
    }

    /// A stream's chunk decoded after its context with the stages after the transformer on the tail alone
    /// ([`WgpuCodec::forward_tail`]) against the whole decode's last samples (`--ignored --nocapture`; OAIY_TTS the
    /// model's folder, OAIY_TTS_DEVICE the adapter): the same, for chunks of one to eight frames after 120.
    #[test]
    #[ignore = "needs Qwen3-TTS's speech tokenizer and a WebGPU adapter"]
    fn a_chunks_tail_is_the_whole_decodes() -> Result<()> {
        let model = std::path::PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let device = std::env::var("OAIY_TTS_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let gpu = WgpuBackend::nth(device, None).map_err(err)?;
        let mut codec = WgpuCodec::load(&model.join("speech_tokenizer").join("model.safetensors"), &gpu)?;
        eprintln!("the stages after the transformer reach {} frames back", codec.reach);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut code = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % CODES as u64) as u32
        };
        for tail in [1, 4, 8] {
            let frames: Vec<[u32; 16]> = (0..120 + tail).map(|_| std::array::from_fn(|_| code())).collect();
            let whole = codec.forward(&frames)?;
            let started = std::time::Instant::now();
            let ours = codec.forward_tail(&frames, tail)?;
            let took = started.elapsed().as_secs_f64();
            let theirs = &whole[(frames.len() - tail) * SAMPLES_PER_FRAME..];
            assert_eq!(ours.len(), theirs.len());
            let e = relative(&ours, theirs);
            eprintln!("{tail} frame(s) after 120: relative RMS error {e:.2e}, {:.0} ms", took * 1e3);
            assert!(e < 1e-5, "{e}");
        }
        // and a reach too short is heard: the check above can tell
        let frames: Vec<[u32; 16]> = (0..124).map(|_| std::array::from_fn(|_| code())).collect();
        let whole = codec.forward(&frames)?;
        codec.reach = 2;
        let e = relative(&codec.forward_tail(&frames, 4)?, &whole[120 * SAMPLES_PER_FRAME..]);
        eprintln!("with a reach of 2 frames: relative RMS error {e:.2e}");
        assert!(e > 1e-3, "{e}");
        Ok(())
    }

    /// [`WgpuCodec::decode`] against Candle's decoder on the CPU (F32, the same chunks) for `sampled_codes.i32` four
    /// times over (360 frames: a second chunk of speech behind 25 frames of context; `--ignored --nocapture`, the
    /// folders as above): each chunk's error over the whole waveform's RMS.
    #[test]
    #[ignore = "needs Qwen3-TTS's speech tokenizer and the reference's dumps"]
    fn the_webgpu_codecs_chunks_are_candles() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("OAIY_TTS_GOLDEN").unwrap_or_else(|_| "E:/deepseek/nrob/target/qwen-tts-golden".into()));
        let model = std::path::PathBuf::from(std::env::var("OAIY_TTS_MODEL").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign".into()));
        let codes: Vec<u32> = std::fs::read(root.join("sampled_codes.i32"))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect();
        let frames: Vec<[u32; 16]> = codes.chunks_exact(16).map(|c| c.try_into().unwrap()).cycle().take(4 * codes.len() / 16).collect();
        assert!(frames.len() > CHUNK, "{} frames: one chunk", frames.len());
        let path = model.join("speech_tokenizer").join("model.safetensors");
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let started = std::time::Instant::now();
        let ours = WgpuCodec::load(&path, &gpu)?.decode(&frames)?;
        eprintln!("WebGPU: loaded and decoded in {:.2} s", started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let theirs = oaiy_tts::codec::CodecDecoder::load(&path, &Device::Cpu)?.decode(&frames)?;
        eprintln!("Candle on the CPU: {:.1} s", started.elapsed().as_secs_f64());
        assert_eq!(ours.len(), theirs.len());
        let split = CHUNK * SAMPLES_PER_FRAME;
        let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let diff: Vec<f32> = ours.iter().zip(&theirs).map(|(a, b)| a - b).collect();
        let (first, second) = (rms(&diff[..split]) / rms(&theirs), rms(&diff[split..]) / rms(&theirs));
        eprintln!(
            "{} frames: relative RMS error {:.3e}; each chunk's error over the whole's RMS {first:.3e} and {second:.3e} (their own RMS {:.4} and {:.4}; the largest difference {:.2e})",
            frames.len(),
            relative(&ours, &theirs),
            rms(&theirs[..split]),
            rms(&theirs[split..]),
            diff.iter().fold(0f32, |m, d| m.max(d.abs())),
        );
        assert!(first < 1e-3 && second < 1e-3, "{first} {second}");
        Ok(())
    }
}
