//! LTX 2.5's diffusion video decoder on WebGPU (`CausalDiffusionVAE`'s `NADiffusionDecoder`), as Lightricks'
//! `DiffusionVideoDecoder` computes it. Four deterministic stages of neighborhood-attention blocks (3D NATTEN windows,
//! absolute 3D RoPE, SwiGLU), each followed by a linear pixel shuffle, carry the latent up to a context the size of the
//! clip's patches (4x4 pixels); a fifth stage of diffusion blocks then predicts the clip's patches from noise, each
//! block taking the context through its own projection and the timestep through a shared AdaLN. LTX 2.5's release
//! predicts the clean clip (x0) in one step from pure noise: a clip is decoded once, its detail drawn from the noise.
//!
//! The weights are f16 on the device (the checkpoint's BF16), every activation f32. Each block is its own recording,
//! settled after (a 121-frame clip's stage-5 vectors are 3 GB each); stages 4 and 5, which carry the clip at full
//! size, run in tiles of the stage-4 grid, as the reference's `diffusion_tiling` plans them, their pixels blended.
use crate::ltx::store::Store;
use candle_core::{Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use oaiy_engine::json::Json;

const EPS: f32 = 1e-6;
/// Every attention's heads are this wide (the release's `head_dim`).
const HEAD: usize = 64;
/// The latent's frames repeated past its last through stages 1 to 4 (the reference's `(K_t // 2) * 2` of stage 1),
/// so the last frames' windows are not shifted inward, then cropped off before stage 5.
const GHOST_FRAMES: usize = 2;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

pub use crate::ltx::vae::is_diffusion_vae;

/// The decoder's shape, from the file's config (the reference's defaults where a field is not there).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Each stage's channels (the last: the context's, and stage 5's).
    pub channels: Vec<usize>,
    pub depths: Vec<usize>,
    /// Each stage's neighborhood (time, rows, columns).
    pub kernels: Vec<[usize; 3]>,
    /// Each upsampling's stride (time, rows, columns) and the factor its channels are reduced by.
    pub upsamples: Vec<([usize; 3], usize)>,
    pub kernel5: [usize; 3],
    pub patch: usize,
    /// The timestep as the embedding takes it, a step's `t` times this.
    pub t_scale: f32,
    pub steps: usize,
    /// The model predicts the clean clip (x0), else its velocity.
    pub x0: bool,
}

impl Config {
    pub fn parse(config: &Json) -> Result<Self> {
        let vae = config.get("vae").unwrap_or(config);
        let d = vae.get("decoder").unwrap_or(vae);
        let ints = |j: &Json| -> Result<Vec<usize>> { j.as_array().ok_or_else(|| err("a list of numbers"))?.iter().map(|v| v.as_i64().map(|n| n as usize).ok_or_else(|| err("a number"))).collect() };
        let triple = |j: &Json| -> Result<[usize; 3]> { ints(j)?.try_into().map_err(|_| err("three numbers")) };
        let list = |key: &str| d.get(key).ok_or_else(|| err(format!("the diffusion VAE's config has no decoder {key}")));
        let channels = ints(list("stage_channels")?)?;
        let depths = ints(list("stage_depths")?)?;
        let kernels = list("stage_kernels")?.as_array().ok_or_else(|| err("stage_kernels"))?.iter().map(triple).collect::<Result<Vec<_>>>()?;
        let upsamples = list("upsamples")?
            .as_array()
            .ok_or_else(|| err("upsamples"))?
            .iter()
            .map(|u| {
                let pair = u.as_array().filter(|p| p.len() == 2).ok_or_else(|| err("an upsampling's stride and reduction"))?;
                Ok((triple(&pair[0])?, pair[1].as_i64().ok_or_else(|| err("a reduction"))? as usize))
            })
            .collect::<Result<Vec<_>>>()?;
        let kernel5 = d.get("stage5_kernel").map(triple).transpose()?.unwrap_or([3, 7, 7]);
        let head = d.get("head_dim").or_else(|| d.get("na_head_dim")).and_then(Json::as_i64).unwrap_or(64) as usize;
        let ok = channels.len() == 5 && depths.len() == 5 && kernels.len() == 5 && upsamples.len() == 4 && head == HEAD && channels.iter().all(|c| c % HEAD == 0) && d.get("stage5_channels").is_none_or(|c| c.as_i64() == Some(channels[4] as i64));
        if !ok {
            candle_core::bail!("a diffusion VAE decoder this does not run: five stages of heads 64 wide, four upsamplings, stage 5 as wide as the context ({channels:?}, {depths:?}, head {head})");
        }
        Ok(Self {
            channels,
            depths,
            kernels,
            upsamples,
            kernel5,
            patch: d.get("patch_size").and_then(Json::as_i64).unwrap_or(4) as usize,
            t_scale: d.get("timestep_scale_multiplier").and_then(Json::as_f64).unwrap_or(1.0) as f32,
            steps: d.get("default_num_inference_steps").and_then(Json::as_i64).unwrap_or(2).max(1) as usize,
            x0: vae.get("model_output_type").and_then(Json::as_str) == Some("x0"),
        })
    }

    /// The latent grid's floor, each stage's neighborhood within its axes (the reference's `all_stages_min_tile_size`).
    pub fn min_latent(&self) -> [usize; 3] {
        let mut mins = [1usize; 3];
        let mut strides = [1usize; 3];
        for (kernel, (stride, _)) in self.kernels.iter().zip(&self.upsamples) {
            for ((least, k), (at, s)) in mins.iter_mut().zip(kernel).zip(strides.iter_mut().zip(stride)) {
                *least = (*least).max(k.div_ceil(*at));
                *at *= s;
            }
        }
        for ((least, k), at) in mins.iter_mut().zip(&self.kernel5).zip(&strides) {
            *least = (*least).max(k.div_ceil(*at));
        }
        mins
    }
}

/// A linear layer as the file holds it: its matrix (`[n, k]`), its bias, `n` and `k`.
type HostLinear = (Vec<f32>, Option<Vec<f32>>, usize, usize);

/// A linear layer on the device: its f16 matrix (`[n, k]`) and its bias.
struct Linear {
    w: DeviceVec,
    b: Option<DeviceVec>,
    n: usize,
    k: usize,
}

/// A neighborhood attention's weights: the queries', keys' and values' projections (the checkpoint's fused one split),
/// the per-head norms (the queries' times the scale), the output's projection.
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    proj: Linear,
    heads: usize,
    kernel: [usize; 3],
}

/// A deterministic stage's block: `x + NA(norm1 x)`, then `x + SwiGLU(norm2 x)`.
struct Block {
    norm1: DeviceVec,
    attn: Attention,
    norm2: DeviceVec,
    mlp: Mlp,
}

/// SwiGLU: `down(silu(gate x) * up x)`, no biases.
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// A diffusion block: the context through its projection into the patches' stream, then attention and the MLP each
/// on the norm modulated by the timestep (`norm1 x (1 + scale) + shift`, the shared AdaLN's plus the block's table).
struct DiffBlock {
    context: Linear,
    /// The block's own scale and shift table (`[7, dim]`), on the host.
    table: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    attn: Attention,
    mlp: Mlp,
}

/// The decoder on the GPU.
pub struct WgpuLtxDiffusionVae {
    gpu: ggml_rs_wgpu::WgpuBackend,
    pub config: Config,
    conv_in: Linear,
    stages: Vec<Vec<Block>>,
    ups: Vec<Linear>,
    conv_in_x_t: Linear,
    blocks: Vec<DiffBlock>,
    norm_out: DeviceVec,
    conv_out: Linear,
    /// The timestep's MLP (`[384, 256]`, `[384]`, `[384, 384]`, `[384]`) and the shared AdaLN (`[7 dim, 384]`, `[7 dim]`), on
    /// the host: one row each a step.
    t_mlp: [Vec<f32>; 4],
    adaln: [Vec<f32>; 2],
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl WgpuLtxDiffusionVae {
    /// The decoder in `store` (`decoder.*`, `per_channel_statistics.*`, or under `vae.`) on GPU `device` (as CUDA
    /// counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(store: &mut Store, device: usize) -> Result<Self> {
        let config = Config::parse(&Json::parse(store.index.metadata("config").ok_or_else(|| err("the diffusion VAE file lacks its config metadata"))?.as_bytes()).map_err(err)?)?;
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let prefix = if store.index.names().any(|n| n.starts_with("vae.decoder.")) { "vae." } else { "" };
        // (its tensors are read one at a time below, in the model's order: the file read through first)
        let warming = std::time::Instant::now();
        let warmed = store.warm(&format!("{prefix}decoder."), |_| true);
        if std::env::var_os("OAIY_MEDIA_MEMORY").is_some() {
            eprintln!("the diffusion decoder's file: {:.2} GB read through in {:.1} s", warmed as f64 / 1e9, warming.elapsed().as_secs_f64());
        }
        let host = |store: &mut Store, name: &str| -> Result<Vec<f32>> { store.tensor_f32(&format!("{prefix}{name}"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>() };
        let up = |gpu: &ggml_rs_wgpu::WgpuBackend, values: &[f32]| -> DeviceVec {
            let v = gpu.vec(values.len().max(1));
            gpu.upload(&v, values);
            v
        };
        let half = |gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, values: &[f32]| -> Result<DeviceVec> { gpu.vec_f16_rounded(values).ok_or_else(|| err(format!("{name}: a weight past f16's range"))) };
        let linear = |store: &mut Store, name: &str, bias: bool| -> Result<HostLinear> {
            let info = store.index.info(&format!("{prefix}decoder.{name}.weight")).map_err(err)?.shape.clone();
            let &[n, k] = info.as_slice() else { candle_core::bail!("{name}: a matrix of shape {info:?}") };
            let w = host(store, &format!("decoder.{name}.weight"))?;
            let b = bias.then(|| host(store, &format!("decoder.{name}.bias"))).transpose()?;
            Ok((w, b, n, k))
        };
        let lin = |gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, (w, b, n, k): HostLinear| -> Result<Linear> { Ok(Linear { w: half(gpu, name, &w)?, b: b.map(|b| up(gpu, &b)), n, k }) };
        let attention = |store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str, kernel: [usize; 3]| -> Result<Attention> {
            let (w, b, n3, k) = linear(store, &format!("{p}.attn.qkv"), true)?;
            let b = b.ok_or_else(|| err("qkv bias"))?;
            let n = n3 / 3;
            let part = |i: usize, name: &str| -> Result<Linear> { Ok(Linear { w: half(gpu, name, &w[i * n * k..(i + 1) * n * k])?, b: Some(up(gpu, &b[i * n..(i + 1) * n])), n, k }) };
            let (q, kk, v) = (part(0, p)?, part(1, p)?, part(2, p)?);
            // (the queries' scale in their norm's weight: `q_norm(q) * head ** -0.5`)
            let scale = (HEAD as f32).powf(-0.5);
            let q_norm: Vec<f32> = host(store, &format!("decoder.{p}.attn.q_norm.weight"))?.iter().map(|w| w * scale).collect();
            let k_norm = host(store, &format!("decoder.{p}.attn.k_norm.weight"))?;
            let proj = lin(gpu, p, linear(store, &format!("{p}.attn.proj"), true)?)?;
            Ok(Attention { q, k: kk, v, q_norm: up(gpu, &q_norm), k_norm: up(gpu, &k_norm), proj, heads: n / HEAD, kernel })
        };
        let mlp = |store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str| -> Result<Mlp> {
            Ok(Mlp {
                gate: lin(gpu, p, linear(store, &format!("{p}.mlp.w_gate"), false)?)?,
                up: lin(gpu, p, linear(store, &format!("{p}.mlp.w_up"), false)?)?,
                down: lin(gpu, p, linear(store, &format!("{p}.mlp.w_down"), false)?)?,
            })
        };
        let conv_in = lin(&gpu, "conv_in", linear(store, "conv_in", true)?)?;
        let mut stages = Vec::new();
        let mut ups = Vec::new();
        for s in 0..4 {
            let mut blocks = Vec::new();
            for i in 0..config.depths[s] {
                let p = format!("det_stages.{s}.{i}");
                blocks.push(Block {
                    norm1: up(&gpu, &host(store, &format!("decoder.{p}.norm1.weight"))?),
                    attn: attention(store, &gpu, &p, config.kernels[s])?,
                    norm2: up(&gpu, &host(store, &format!("decoder.{p}.norm2.weight"))?),
                    mlp: mlp(store, &gpu, &p)?,
                });
            }
            stages.push(blocks);
            ups.push(lin(&gpu, "upsample", linear(store, &format!("upsamples.{s}.proj"), true)?)?);
        }
        let conv_in_x_t = lin(&gpu, "conv_in_x_t", linear(store, "conv_in_x_t", true)?)?;
        let mut blocks = Vec::new();
        for i in 0..config.depths[4] {
            let p = format!("diff_blocks.{i}");
            blocks.push(DiffBlock {
                context: lin(&gpu, &p, linear(store, &format!("{p}.context_proj"), true)?)?,
                table: host(store, &format!("decoder.{p}.scale_shift_table"))?,
                norm1: host(store, &format!("decoder.{p}.norm1.weight"))?,
                norm2: host(store, &format!("decoder.{p}.norm2.weight"))?,
                attn: attention(store, &gpu, &p, config.kernel5)?,
                mlp: mlp(store, &gpu, &p)?,
            });
        }
        let norm_out = up(&gpu, &host(store, "decoder.norm_out.weight")?);
        let conv_out = lin(&gpu, "conv_out", linear(store, "conv_out", true)?)?;
        let t_mlp = [host(store, "decoder.t_embedder.mlp.0.weight")?, host(store, "decoder.t_embedder.mlp.0.bias")?, host(store, "decoder.t_embedder.mlp.2.weight")?, host(store, "decoder.t_embedder.mlp.2.bias")?];
        let adaln = [host(store, "decoder.shared_adaln.proj.weight")?, host(store, "decoder.shared_adaln.proj.bias")?];
        let (std, mean) = (host(store, "per_channel_statistics.std-of-means")?, host(store, "per_channel_statistics.mean-of-means")?);
        if std::env::var_os("OAIY_MEDIA_MEMORY").is_some() {
            eprintln!("the diffusion decoder's weights read, converted and on the GPU in {:.1} s", warming.elapsed().as_secs_f64());
        }
        Ok(Self { gpu, config, conv_in, stages, ups, conv_in_x_t, blocks, norm_out, conv_out, t_mlp, adaln, mean, std })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    fn upload(&self, values: &[f32]) -> DeviceVec {
        let v = self.vec(values.len());
        self.gpu.upload(&v, values);
        v
    }

    fn linear(&self, r: &mut dyn ChainRecorder, l: &Linear, x: &DeviceVec, rows: usize) -> DeviceVec {
        let y = self.vec(rows * l.n);
        r.matmul_f16_rows_ranged(&l.w, l.n, l.k, x, &y, rows);
        if let Some(b) = &l.b {
            r.add_bias_rows(&y, b, rows, l.n);
        }
        y
    }

    /// Each row of `x` RMS-normed by `w` (and `shift` added).
    fn norm(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, w: &DeviceVec, shift: Option<&DeviceVec>, rows: usize, c: usize) -> DeviceVec {
        let y = self.vec(rows * c);
        r.rmsnorm_rows(x, w, &y, rows, EPS);
        if let Some(s) = shift {
            r.add_bias_rows(&y, s, rows, c);
        }
        y
    }

    /// The absolute 3D RoPE's table for a `t` by `h` by `w` grid (the reference's split of a 64-wide head: 8 pairs by
    /// time, 12 by row, 12 by column; each axis's frequencies `10000 ^ (-2 i / dims)`, its angles in f32).
    fn rope_table(t: usize, h: usize, w: usize) -> (Vec<f32>, [usize; 3]) {
        let pairs = rope_pairs();
        let mut table = Vec::with_capacity(2 * (t * pairs[0] + h * pairs[1] + w * pairs[2]));
        for (len, p) in [(t, pairs[0]), (h, pairs[1]), (w, pairs[2])] {
            let inv: Vec<f32> = (0..p).map(|i| (1.0 / 10000f64.powf((2 * i) as f64 / (2 * p) as f64)) as f32).collect();
            for place in 0..len {
                for f in &inv {
                    let a = (place as f32 * f) as f64;
                    table.extend([a.cos() as f32, a.sin() as f32]);
                }
            }
        }
        (table, pairs)
    }

    /// One recording of `f`'s ops, settled after: what it made and let go of is freed before the next is recorded.
    fn recorded(&self, f: impl FnOnce(&mut dyn ChainRecorder)) {
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        f(rec.as_mut());
        rec.finish();
        self.gpu.settle();
    }

    /// `x += attn(norm x)` over a `t` by `h` by `w` grid (`norm` making the normed rows), in recordings that each hold
    /// only what they make and the next needs, at most six vectors of the grid's (a 512K-token tile's are 512 MB each):
    /// the queries normed and turned, the keys, the values, the attention, its projection.
    fn attend(&self, a: &Attention, x: &DeviceVec, norm: impl FnOnce(&mut dyn ChainRecorder) -> DeviceVec, rope: &DeviceVec, (t, h, w): (usize, usize, usize)) {
        let rows = t * h * w;
        let c = a.q.n;
        let (mut y, mut qn, mut kn, mut v, mut o) = (None, None, None, None, None);
        self.recorded(|r| {
            let normed = norm(r);
            let q = self.linear(r, &a.q, &normed, rows);
            let n = self.vec(rows * c);
            r.rmsnorm_rows(&q, &a.q_norm, &n, rows * a.heads, EPS);
            r.rope_3d_rows(&n, rope, t, h, w, a.heads, rope_pairs());
            (y, qn) = (Some(normed), Some(n));
        });
        let (y, qn) = (y.expect("normed rows"), qn.expect("queries"));
        self.recorded(|r| {
            let k = self.linear(r, &a.k, &y, rows);
            let n = self.vec(rows * c);
            r.rmsnorm_rows(&k, &a.k_norm, &n, rows * a.heads, EPS);
            r.rope_3d_rows(&n, rope, t, h, w, a.heads, rope_pairs());
            kn = Some(n);
        });
        self.recorded(|r| v = Some(self.linear(r, &a.v, &y, rows)));
        drop(y);
        let (kn, v) = (kn.expect("keys"), v.expect("values"));
        self.recorded(|r| {
            let out = self.vec(rows * c);
            r.neighborhood_attention_3d(&qn, &kn, &v, &out, t, h, w, a.heads, a.kernel);
            o = Some(out);
        });
        drop((qn, kn, v));
        let o = o.expect("the attention");
        self.recorded(|r| {
            let p = self.linear(r, &a.proj, &o, rows);
            r.add(x, &p);
        });
    }

    /// `x += mlp(y)`, its rows a part at a time (the hidden vectors of at most 256 MB each).
    fn feed(&self, r: &mut dyn ChainRecorder, m: &Mlp, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        let (c, hidden) = (m.gate.k, m.gate.n);
        let part = ((256usize << 20) / (4 * hidden)).max(1).min(rows);
        let out = self.vec(rows * c);
        let (yp, gate, upv, hv) = (self.vec(part * c), self.vec(part * hidden), self.vec(part * hidden), self.vec(part * hidden));
        let dp = self.vec(part * c);
        for r0 in (0..rows).step_by(part) {
            let n = part.min(rows - r0);
            let src = if part == rows { y.clone() } else {
                r.copy(y, r0 * c, &yp, 0, n * c);
                yp.clone()
            };
            r.matmul_f16_rows_ranged(&m.gate.w, hidden, c, &src, &gate, n);
            r.matmul_f16_rows_ranged(&m.up.w, hidden, c, &src, &upv, n);
            r.silu_mul(&gate, &upv, &hv, n * hidden);
            r.matmul_f16_rows_ranged(&m.down.w, c, hidden, &hv, &dp, n);
            r.copy(&dp, 0, &out, r0 * c, n * c);
        }
        r.add(x, &out);
    }

    /// One deterministic block: its attention, then its MLP (its own recording).
    fn block(&self, b: &Block, x: &DeviceVec, rope: &DeviceVec, (t, h, w): (usize, usize, usize)) {
        let rows = t * h * w;
        let c = b.attn.q.n;
        self.attend(&b.attn, x, |r| self.norm(r, x, &b.norm1, None, rows, c), rope, (t, h, w));
        self.recorded(|r| {
            let y = self.norm(r, x, &b.norm2, None, rows, c);
            self.feed(r, &b.mlp, x, &y, rows);
        });
    }

    /// Stage `s`'s blocks then its upsampling (`origin`: the doubled first frame left out, the clip's origin's): the
    /// grid it makes and its rows.
    fn stage(&self, s: usize, x: DeviceVec, (t, h, w): (usize, usize, usize), origin: bool) -> ((usize, usize, usize), DeviceVec) {
        let (table, _) = Self::rope_table(t, h, w);
        let rope = self.upload(&table);
        for b in &self.stages[s] {
            self.block(b, &x, &rope, (t, h, w));
        }
        let ([st, sh, sw], reduction) = self.config.upsamples[s];
        let c = self.config.channels[s];
        let cout = c / reduction;
        let dropped = usize::from(st == 2 && origin);
        let (nt, nh, nw) = (t * st - dropped, h * sh, w * sw);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let y = self.linear(r, &self.ups[s], &x, t * h * w);
        let out = self.vec(nt * nh * nw * cout);
        r.depth_to_space_rows(&y, &out, t, h, w, cout, st, sh, sw, dropped);
        rec.finish();
        drop((y, x));
        self.gpu.settle();
        ((nt, nh, nw), out)
    }

    /// Stages 1 to 3 on a latent already padded (`[t h w, 128]`, normalized as the transformer makes it): the stage-4
    /// grid's input.
    fn stages_1_to_3(&self, latent: &[f32], (t, h, w): (usize, usize, usize)) -> ((usize, usize, usize), DeviceVec) {
        let values: Vec<f32> = latent.iter().enumerate().map(|(i, v)| v * self.std[i % 128] + self.mean[i % 128]).collect();
        let z = self.upload(&values);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let x = self.linear(rec.as_mut(), &self.conv_in, &z, t * h * w);
        rec.finish();
        drop(z);
        let (mut grid, mut x) = ((t, h, w), x);
        for s in 0..3 {
            (grid, x) = self.stage(s, x, grid, true);
        }
        (grid, x)
    }

    /// The timestep's modulation for a step at `t` (`[7, dim]`: scale and shift for attention, a gate, scale and
    /// shift for the MLP, two gates), on the host.
    fn modulation(&self, t: f32) -> Vec<f32> {
        let x = self.config.t_scale * t;
        // the sinusoid (cosines first), 256 wide
        let half = 128;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
        let mut s: Vec<f32> = freqs.iter().map(|f| (x * f).cos()).collect();
        s.extend(freqs.iter().map(|f| (x * f).sin()));
        let mv = |w: &[f32], b: &[f32], x: &[f32]| -> Vec<f32> { b.iter().enumerate().map(|(o, b)| b + w[o * x.len()..(o + 1) * x.len()].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect() };
        let silu = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|v| v / (1.0 + (-v).exp())).collect() };
        let e = mv(&self.t_mlp[2], &self.t_mlp[3], &silu(mv(&self.t_mlp[0], &self.t_mlp[1], &s)));
        mv(&self.adaln[0], &self.adaln[1], &silu(e))
    }

    /// One stage-5 step: the context (`[t h w, dim]`) and the noised patches (`x_t`, `[t h w, 48]`) to the model's
    /// prediction (`[t h w, 48]`), on the host.
    fn step(&self, context: &DeviceVec, x_t: &[f32], (t, h, w): (usize, usize, usize), at: f32) -> Result<Vec<f32>> {
        let rows = t * h * w;
        let c = self.config.channels[4];
        let modulation = self.modulation(at);
        let xt = self.upload(x_t);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let x = self.linear(rec.as_mut(), &self.conv_in_x_t, &xt, rows);
        rec.finish();
        drop(xt);
        let (table, _) = Self::rope_table(t, h, w);
        let rope = self.upload(&table);
        for b in &self.blocks {
            // (each norm's weight times its scale, its shift a bias: the timestep is the tile's own)
            let chunk = |i: usize| -> Vec<f32> { (0..c).map(|j| modulation[i * c + j] + b.table[i * c + j]).collect() };
            let (s1, h1, s2, h2) = (chunk(0), chunk(1), chunk(3), chunk(4));
            let w1: Vec<f32> = b.norm1.iter().zip(&s1).map(|(w, s)| w * (1.0 + s)).collect();
            let w2: Vec<f32> = b.norm2.iter().zip(&s2).map(|(w, s)| w * (1.0 + s)).collect();
            let (w1, h1, w2, h2) = (self.upload(&w1), self.upload(&h1), self.upload(&w2), self.upload(&h2));
            self.recorded(|r| {
                let injected = self.linear(r, &b.context, context, rows);
                r.add(&x, &injected);
            });
            self.attend(&b.attn, &x, |r| self.norm(r, &x, &w1, Some(&h1), rows, c), &rope, (t, h, w));
            self.recorded(|r| {
                let y = self.norm(r, &x, &w2, Some(&h2), rows, c);
                self.feed(r, &b.mlp, &x, &y, rows);
            });
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let y = self.norm(r, &x, &self.norm_out, None, rows, c);
        let out = self.linear(r, &self.conv_out, &y, rows);
        r.read(&out);
        rec.finish().pop().ok_or_else(|| err("the clip's patches were not read"))
    }

    /// Stage 4 on its input (`x`, a stage-4 grid of `grid`) and stage 5's steps from `noise` (`[3, frames, rows,
    /// columns]` pixels, the stage-5 canvas): the clip's pixels `[3, frames, rows, columns]`. `drop`: the tile holds
    /// the clip's first frame; `trailing`: it holds its last (the ghost frames, then cropped).
    pub fn stages_4_and_5(&self, x: DeviceVec, grid: (usize, usize, usize), noise: &[f32], drop: bool, trailing: bool) -> Result<Vec<f32>> {
        let ((t5, h5, w5), context) = self.stage(3, x, grid, drop);
        let c = self.config.channels[4];
        // the ghost frames off: as many as they became, the stage-5 kernel's floor kept
        let (t5, context) = if trailing {
            let ghost = GHOST_FRAMES * 8;
            let keep = t5.min((t5.saturating_sub(ghost)).max(1).max(self.config.kernel5[0]));
            if keep < t5 {
                let kept = self.vec(keep * h5 * w5 * c);
                let mut rec = self.gpu.begin();
                rec.copy(&context, 0, &kept, 0, keep * h5 * w5 * c);
                rec.finish();
                (keep, kept)
            } else {
                (t5, context)
            }
        } else {
            (t5, context)
        };
        let p = self.config.patch;
        let frames = noise.len() / (3 * h5 * p * w5 * p);
        if frames != t5 || noise.len() != 3 * t5 * h5 * p * w5 * p {
            candle_core::bail!("stage 5's noise ({} values) is not the context's canvas ({t5}x{}x{})", noise.len(), h5 * p, w5 * p);
        }
        let mut x_t = patchify(noise, t5, h5, w5, p);
        let steps = self.config.steps;
        let times: Vec<f32> = (0..steps).map(|i| if steps == 1 { 1.0 } else { 1.0 - i as f32 * (1.0 - 1.0 / steps as f32) / (steps - 1) as f32 }).collect();
        let mut out = Vec::new();
        for (i, &at) in times.iter().enumerate() {
            let pred = self.step(&context, &x_t, (t5, h5, w5), at)?;
            let next = times.get(i + 1).copied().unwrap_or(0.0);
            if i + 1 == steps && self.config.x0 {
                out = pred;
                break;
            }
            // Euler from `at` to `next` on the velocity (an x0 prediction's: (x_t - x0) / t)
            for (v, p) in x_t.iter_mut().zip(&pred) {
                let vel = if self.config.x0 { (*v - p) / at } else { *p };
                *v -= (at - next) * vel;
            }
            if i + 1 == steps {
                out = x_t.clone();
            }
        }
        Ok(unpatchify(&out, t5, h5, w5, p))
    }

    /// The stage-4 grid a latent grid makes (the first three upsamplings, the doubled first frames dropped).
    fn stage4_grid(&self, (t, h, w): (usize, usize, usize)) -> [usize; 3] {
        let mut g = [t, h, w];
        for (stride, _) in &self.config.upsamples[..3] {
            for a in 0..3 {
                g[a] *= stride[a];
            }
            if stride[0] == 2 {
                g[0] -= 1;
            }
        }
        g
    }

    /// Pixels (frames, rows, columns) a stage-4 cell makes: stage 4's upsampling, then the patch.
    fn pixel_scale(&self) -> [usize; 3] {
        let ([st, sh, sw], _) = self.config.upsamples[3];
        [st, sh * self.config.patch, sw * self.config.patch]
    }

    /// The tiles' overlaps (frames, pixels) and their least size on the stage-4 grid (the reference's
    /// `recommended_pixel_overlaps` and `compute_tile_min_size`): stage 4's and stage 5's reach each side (each block's
    /// half window, stage 5's in stage-4 cells), in pixels, the frames on 8's and the pixels on 32's.
    fn overlaps(&self) -> ([usize; 2], [usize; 3]) {
        let c = &self.config;
        let ([st, sh, sw], _) = c.upsamples[3];
        let stride = [st, sh, sw];
        let reach = |a: usize| (c.depths[3] * (c.kernels[3][a] / 2)).max((c.depths[4] * (c.kernel5[a] / 2)).div_ceil(stride[a]));
        let scale = self.pixel_scale();
        let least = [0, 1, 2].map(|a| c.kernels[3][a].max(c.kernel5[a].div_ceil(stride[a])));
        ([(reach(0) * scale[0]).div_ceil(8) * 8, (reach(1).max(reach(2)) * scale[1]).div_ceil(32) * 32], least)
    }

    /// The tile sizes (frames, rows, columns in pixels) for a clip of `full` pixels: None where its stage 5 fits
    /// `tokens` whole, else the reference's choice (`recommended_decode_tiling_config`): every size on the grid's steps
    /// whose stage 5 fits, the least overlap recomputed, then the largest, then the fewest tiles.
    fn tile_sizes(&self, full: [usize; 3], tokens: usize) -> Result<Option<[usize; 3]>> {
        let p = self.config.patch;
        if full[0] * (full[1] / p) * (full[2] / p) <= tokens {
            return Ok(None);
        }
        let ([ot, ohw], least) = self.overlaps();
        let scale = self.pixel_scale();
        let steps = [lcm(scale[0], 8), lcm(scale[1], 32), lcm(scale[2], 32)];
        let lowest = |a: usize, overlap: usize, floor: usize| (2 * scale[a]).max(2 * overlap).max(least[a] * scale[a]).max(floor).div_ceil(steps[a]) * steps[a];
        let mins = [lowest(0, ot, 16), lowest(1, ohw, 64), lowest(2, ohw, 64)];
        let overlap = [ot, ohw, ohw];
        let candidates = |a: usize| -> Vec<(usize, usize)> {
            let most = full[a].div_ceil(steps[a]) * steps[a];
            (mins[a]..=most.max(mins[a])).step_by(steps[a]).filter(|&size| size > overlap[a]).map(|size| (size, split_by_size(full[a], size, overlap[a], 1).len())).collect()
        };
        let mut best: Option<(f64, i64, usize, [usize; 3])> = None;
        for &(tt, nt) in &candidates(0) {
            for &(th, nh) in &candidates(1) {
                for &(tw, nw) in &candidates(2) {
                    if tt * (th / p).max(1) * (tw / p).max(1) > tokens {
                        continue;
                    }
                    let waste = (nt * nh * nw * tt * th * tw) as f64 / (full[0] * full[1] * full[2]) as f64;
                    let key = (waste, -((tt * th * tw) as i64), nt * nh * nw, [tt, th, tw]);
                    if best.as_ref().is_none_or(|b| (key.0, key.1, key.2) < (b.0, b.1, b.2)) {
                        best = Some(key);
                    }
                }
            }
        }
        best.map(|b| Some(b.3)).ok_or_else(|| err(format!("the GPU has no room for a tile of the diffusion decoder ({} frames by {}x{} pixels at least): a smaller or shorter clip, or more free memory", mins[0], mins[1], mins[2])))
    }

    /// How many stage-5 tokens a tile may have: the GPU's free memory (8 GB where the API does not say) less the
    /// stage-4 features held through the decode and a GB to spare, at seven of stage 5's vectors a token (its blocks'
    /// most), each vector within a binding.
    fn tile_tokens(&self, features: usize) -> usize {
        let c = self.config.channels[4];
        let free = self.gpu.memory_budget().map_or(8 << 30, |(budget, used)| budget.saturating_sub(used));
        let usable = free.saturating_sub(1 << 30).saturating_sub(if self.gpu.memory_budget().is_some() { 0 } else { (features * 4) as u64 });
        let by_memory = usable / (7 * c as u64 * 4);
        let by_binding = self.gpu.max_binding() / (c as u64 * 4);
        by_memory.min(by_binding) as usize
    }

    /// The clip of `latent` (`[f h w, 128]`, the transformer's tokens: `f` frames of `h` by `w`): `[1, 3, 8 (f - 1) +
    /// 1, 32 h, 32 w]` (RGB in -1..1) on the CPU, whole where the GPU has room, else in tiles; each tile's stage-5 noise
    /// from `seed`.
    pub fn decode(&self, latent: &[f32], f: usize, h: usize, w: usize, seed: u64) -> Result<Tensor> {
        self.decode_with(latent, f, h, w, None, |tile, n| crate::common::job::noise(seed.wrapping_add(tile as u64 * 0x9e37_79b9_7f4a_7c15), n))
    }

    /// [`Self::decode`] in tiles of `sizes` (frames, rows, columns in pixels; None: as the GPU's room has them), each
    /// tile's stage-5 noise from `noise` (asked for its tile's and its number of values), as the reference's
    /// `_decode_pixels` makes them: stages 1 to 3 once on the whole latent (padded to the grid's floor, its ghost frames
    /// after), stages 4 and 5 a tile of the stage-4 grid at a time, each alone (the clip's first frame's tile dropping
    /// the doubled frame, its last's carrying the ghost frames), their pixels blended by their masks.
    pub fn decode_with(&self, latent: &[f32], f: usize, h: usize, w: usize, sizes: Option<[usize; 3]>, mut noise: impl FnMut(usize, usize) -> Vec<f32>) -> Result<Tensor> {
        if latent.len() != f * h * w * 128 || f == 0 || h == 0 || w == 0 {
            candle_core::bail!("an LTX latent of {} values for {f}x{h}x{w} tokens", latent.len());
        }
        let Padded { mut values, grid: (ft, fh, fw), before: (top, left) } = floor_padded(latent, (f, h, w), self.config.min_latent());
        // the ghost frames: the last latent frame repeated
        let frame = fh * fw * 128;
        let last = values[values.len() - frame..].to_vec();
        for _ in 0..GHOST_FRAMES {
            values.extend_from_slice(&last);
        }
        let ((gt, gh, gw), features) = self.stages_1_to_3(&values, (ft + GHOST_FRAMES, fh, fw));
        drop(values);
        let s4 = self.stage4_grid((ft, fh, fw));
        let scale = self.pixel_scale();
        let p = self.config.patch;
        let causal = self.config.upsamples[3].0[0] == 2;
        let full = [s4[0] * scale[0] - usize::from(causal), s4[1] * scale[1], s4[2] * scale[2]];
        let sizes = match sizes {
            Some(s) => Some(s),
            None => self.tile_sizes(full, self.tile_tokens(gt * gh * gw * self.config.channels[3]))?,
        };
        let ([ot, ohw], least) = self.overlaps();
        let tiles = plan(s4, full, sizes, [ot, ohw], least, self.config.upsamples[3].0, p);
        // (OAIY_MEDIA_MEMORY: the tiles the GPU's room made, and what it said it had)
        if std::env::var_os("OAIY_MEDIA_MEMORY").is_some() {
            let room = self.gpu.memory_budget().map(|(budget, used)| format!("{:.1} of {:.1} GB free", budget.saturating_sub(used) as f64 / 1e9, budget as f64 / 1e9));
            eprintln!("the diffusion decoder: {} tile(s) of {sizes:?} for {full:?} ({})", tiles.len(), room.unwrap_or_else(|| "its room not said".into()));
        }
        let c = self.config.channels[3];
        let mut acc = vec![0f32; 3 * full[0] * full[1] * full[2]];
        let mut weight = vec![0f32; full[0] * full[1] * full[2]];
        for (i, tile) in tiles.iter().enumerate() {
            let [it, ih, iw] = tile.grid;
            let origin = it.start == 0;
            let trailing = it.end == s4[0];
            // (the last tile in time carries the ghost frames' features)
            let t_end = if trailing { gt } else { it.end };
            let (nt, nh, nw) = (t_end - it.start, ih.end - ih.start, iw.end - iw.start);
            let index: Vec<f32> = (it.start..t_end).flat_map(|t| (ih.start..ih.end).flat_map(move |y| (iw.start..iw.end).map(move |x| f32::from_bits(((t * gh + y) * gw + x) as u32)))).collect();
            let index = self.upload(&index);
            let x = self.vec(nt * nh * nw * c);
            self.recorded(|r| r.gather_rows(&features, &index, &x, nt * nh * nw, c, 0, gt * gh * gw));
            drop(index);
            // stage 5's canvas: the content's frames (the floor of its kernel's on the last tile), its rows and columns
            let ([st, sh, sw], _) = self.config.upsamples[3];
            let mut frames5 = (it.end - it.start) * st - usize::from(origin && st == 2);
            if trailing {
                frames5 = frames5.max(self.config.kernel5[0]);
            }
            let (rows5, cols5) = (nh * sh * p, nw * sw * p);
            let pixels = self.stages_4_and_5(x, (nt, nh, nw), &noise(i, 3 * frames5 * rows5 * cols5), origin, trailing)?;
            // into the clip: the tile's pixels (its frames past them cut), each by its masks
            let [pt, py, px] = tile.pixels;
            let (lt, ly, lx) = (pt.end - pt.start, py.end - py.start, px.end - px.start);
            if ly != rows5 || lx != cols5 || lt > frames5 {
                candle_core::bail!("a tile's pixels ({frames5}x{rows5}x{cols5}) are not its place in the clip ({lt}x{ly}x{lx})");
            }
            let [mt, my, mx] = &tile.masks;
            for (t, mtv) in mt.iter().enumerate().take(lt) {
                for (y, myv) in my.iter().enumerate().take(ly) {
                    let m = mtv * myv;
                    let at = ((pt.start + t) * full[1] + py.start + y) * full[2] + px.start;
                    for (xx, mxv) in mx.iter().enumerate() {
                        weight[at + xx] += m * mxv;
                    }
                    for ch in 0..3 {
                        let from = ((ch * frames5 + t) * rows5 + y) * cols5;
                        let to = ch * full[0] * full[1] * full[2] + at;
                        for (xx, mxv) in mx.iter().enumerate() {
                            acc[to + xx] += pixels[from + xx] * m * mxv;
                        }
                    }
                }
            }
        }
        drop(features);
        // back to the content: its frames, its rows and columns past the floor's padding, each pixel by its weight
        let frames = 8 * (f - 1) + 1;
        let (top, left) = (top * 32, left * 32);
        let plane = full[0] * full[1] * full[2];
        let mut out = vec![0f32; 3 * frames * h * 32 * w * 32];
        for ch in 0..3 {
            for t in 0..frames {
                for y in 0..h * 32 {
                    let from = (t * full[1] + top + y) * full[2] + left;
                    let to = ((ch * frames + t) * h * 32 + y) * w * 32;
                    for xx in 0..w * 32 {
                        let wt = weight[from + xx];
                        if wt <= 0.0 {
                            candle_core::bail!("a pixel no tile covered");
                        }
                        out[to + xx] = acc[ch * plane + from + xx] / wt;
                    }
                }
            }
        }
        Tensor::from_vec(out, (1, 3, frames, h * 32, w * 32), &Device::Cpu)
    }
}

fn lcm(a: usize, b: usize) -> usize {
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    a / gcd(a, b) * b
}

/// An interval of one axis and the ramps it shares with its neighbours (the reference's `DimensionInterval`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
    pub left: usize,
    pub right: usize,
}

impl Interval {
    /// Through an upsampling of `stride` (the reference's `_propagate_interval_through_upsample_hops`): times the
    /// stride, and where time doubles (`causal`) the duplicate first frame dropped, a later interval starting one
    /// earlier.
    fn up(self, stride: usize, causal: bool) -> Self {
        let mut i = Interval { start: self.start * stride, end: self.end * stride, left: self.left * stride, right: self.right * stride };
        if causal && stride == 2 {
            i.end -= 1;
            if self.start != 0 {
                i.start -= 1;
            }
        }
        i
    }
}

/// An axis `len` long in intervals of `size` sharing `overlap`, the last grown back to `least` where it is shorter
/// (the reference's `split_by_size`); one interval where it fits one.
pub fn split_by_size(len: usize, size: usize, overlap: usize, least: usize) -> Vec<Interval> {
    if len < least || len <= size {
        return vec![Interval { start: 0, end: len, left: 0, right: 0 }];
    }
    let amount = (len + size - 2 * overlap - 1) / (size - overlap);
    let mut v = vec![Interval { start: 0, end: size, left: 0, right: overlap }];
    for i in 1..amount - 1 {
        v.push(Interval { start: i * (size - overlap), end: i * (size - overlap) + size, left: overlap, right: overlap });
    }
    v.push(Interval { start: (amount - 1) * (size - overlap), end: len, left: overlap, right: 0 });
    let n = v.len();
    if n > 1 && v[n - 1].end - v[n - 1].start < least {
        let start = v[n - 1].end - least;
        let shared = v[n - 2].end - start;
        v[n - 2].right = shared;
        v[n - 1].start = start;
        v[n - 1].left = shared;
    }
    v
}

/// A blend mask `len` long ramped in over `left` and out over `right` (the reference's `compute_trapezoidal_mask_1d`,
/// its ramps starting past 0: `i / (ramp + 1)`).
pub fn trapezoid(len: usize, left: usize, right: usize) -> Vec<f32> {
    let (left, right) = (left.min(len), right.min(len));
    let mut m = vec![1f32; len];
    for (i, v) in m.iter_mut().take(left).enumerate() {
        *v *= (i + 1) as f32 / (left + 1) as f32;
    }
    for (i, v) in m[len - right..].iter_mut().enumerate() {
        *v *= 1.0 - (i + 1) as f32 / (right + 1) as f32;
    }
    m
}

/// A tile: its intervals of the stage-4 grid, its pixels' in the clip, and their blend masks.
#[derive(Clone, Debug)]
pub struct Tile {
    pub grid: [Interval; 3],
    pub pixels: [Interval; 3],
    pub masks: [Vec<f32>; 3],
}

/// The tiles of a stage-4 grid of `s4` making a clip of `full` pixels: one where `sizes` is None, else the reference's
/// `prepare_tile_schedule` for tiles of `sizes` pixels (frames, rows, columns) sharing `overlap` (frames, pixels), each
/// axis split on the stage-4 grid (at least `least` cells a tile) and carried to pixels through stage 4's upsampling of
/// `stride` and the patch (time causal: the first frame not doubled), time slowest.
pub fn plan(s4: [usize; 3], full: [usize; 3], sizes: Option<[usize; 3]>, overlap: [usize; 2], least: [usize; 3], stride: [usize; 3], patch: usize) -> Vec<Tile> {
    let scale = [stride[0], stride[1] * patch, stride[2] * patch];
    let axis = |a: usize| -> Vec<(Interval, Interval)> {
        let intervals = match sizes {
            None => vec![Interval { start: 0, end: s4[a], left: 0, right: 0 }],
            Some(sizes) => {
                let ov = overlap[a.min(1)] / scale[a];
                let size = (sizes[a] / scale[a]).max(2.max(ov + 1));
                split_by_size(s4[a], size, ov, least[a])
            }
        };
        intervals
            .into_iter()
            .map(|iv| {
                let px = if a == 0 { iv.up(stride[0], true) } else { iv.up(stride[a], false).up(patch, false) };
                (iv, px)
            })
            .collect()
    };
    let (ts, hs, ws) = (axis(0), axis(1), axis(2));
    let _ = full;
    let mut tiles = Vec::new();
    for &(gt, pt) in &ts {
        for &(gh, ph) in &hs {
            for &(gw, pw) in &ws {
                let mask = |i: Interval| trapezoid(i.end - i.start, i.left, i.right);
                tiles.push(Tile { grid: [gt, gh, gw], pixels: [pt, ph, pw], masks: [mask(pt), mask(ph), mask(pw)] });
            }
        }
    }
    tiles
}

/// The 3D RoPE's pairs by time, row and column for a 64-wide head (the reference's `default_rope_dim_split`: 16, 24,
/// 24 dims).
fn rope_pairs() -> [usize; 3] {
    let d_t = (HEAD / 4) / 2 * 2;
    let d_hw = (HEAD - d_t) / 2;
    [d_t / 2, d_hw / 2, d_hw / 2]
}

/// A latent padded to the grid's floor: its values (`[t h w, 128]`), its grid, and the rows and columns added before it.
struct Padded {
    values: Vec<f32>,
    grid: (usize, usize, usize),
    before: (usize, usize),
}

/// `latent` (`[f h w, 128]`) up to the grid's `floor`, as the reference's `ensure_min_latent_shape` pads it: frames the
/// last repeated after it, rows and columns their edges' (half before, the rest after).
fn floor_padded(latent: &[f32], (f, h, w): (usize, usize, usize), floor: [usize; 3]) -> Padded {
    let (nt, nh, nw) = (f.max(floor[0]), h.max(floor[1]), w.max(floor[2]));
    let (top, left) = ((nh - h) / 2, (nw - w) / 2);
    let mut values = Vec::with_capacity(nt * nh * nw * 128);
    for t in 0..nt {
        let t = t.min(f - 1);
        for y in 0..nh {
            let y = y.saturating_sub(top).min(h - 1);
            for x in 0..nw {
                let x = x.saturating_sub(left).min(w - 1);
                values.extend_from_slice(&latent[((t * h + y) * w + x) * 128..][..128]);
            }
        }
    }
    Padded { values, grid: (nt, nh, nw), before: (top, left) }
}

/// Pixels `[3, t, h p, w p]` as patches `[t h w, 48]`: a patch's channel `(c p + column) p + row` (the reference's
/// `b c (f 1) (h q) (w r) -> b (c 1 r q) f h w`, columns before rows).
pub fn patchify(pixels: &[f32], t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (oh, ow) = (h * p, w * p);
    let mut out = vec![0f32; t * h * w * 3 * p * p];
    for f in 0..t {
        for y in 0..h {
            for x in 0..w {
                let row = &mut out[((f * h + y) * w + x) * 3 * p * p..][..3 * p * p];
                for c in 0..3 {
                    for col in 0..p {
                        for r in 0..p {
                            row[(c * p + col) * p + r] = pixels[((c * t + f) * oh + y * p + r) * ow + x * p + col];
                        }
                    }
                }
            }
        }
    }
    out
}

/// [`patchify`]'s inverse.
pub fn unpatchify(patches: &[f32], t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (oh, ow) = (h * p, w * p);
    let mut out = vec![0f32; 3 * t * oh * ow];
    for f in 0..t {
        for y in 0..h {
            for x in 0..w {
                let row = &patches[((f * h + y) * w + x) * 3 * p * p..][..3 * p * p];
                for c in 0..3 {
                    for col in 0..p {
                        for r in 0..p {
                            out[((c * t + f) * oh + y * p + r) * ow + x * p + col] = row[(c * p + col) * p + r];
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agree(got: &[f32], want: &[f32]) -> (f64, f64) {
        assert_eq!(got.len(), want.len(), "lengths");
        let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs() as f64).fold(0f64, f64::max);
        (dot / (norm(got) * norm(want)), worst)
    }

    /// A latent's grid padded to the floor as the reference pads it: frames repeated after, rows and columns their
    /// edges' (an odd count's extra after), and nothing where the grid has room.
    #[test]
    fn a_latent_is_padded_to_the_floor_as_the_reference_pads_it() {
        let latent: Vec<f32> = (0..2 * 2 * 3 * 128).map(|i| (i / 128) as f32).collect();
        let p = floor_padded(&latent, (2, 2, 3), [3, 5, 4]);
        assert_eq!((p.grid, p.before), ((3, 5, 4), (1, 0)));
        let token = |t: usize, y: usize, x: usize| p.values[((t * 5 + y) * 4 + x) * 128];
        // (source token t*6 + y*3 + x)
        assert_eq!([token(0, 0, 0), token(0, 1, 0), token(0, 2, 2), token(0, 4, 3), token(2, 3, 1)], [0., 0., 5., 5., 10.]);
        let same = floor_padded(&latent, (2, 2, 3), [1, 1, 1]);
        assert_eq!((same.values, same.grid, same.before), (latent, (2, 2, 3), (0, 0)));
    }

    /// The patches' channels as the reference lays them: colour, then the column in the patch, then its row.
    #[test]
    fn patches_lay_columns_before_rows() {
        let (t, h, w, p) = (2usize, 3usize, 2usize, 4usize);
        let pixels: Vec<f32> = (0..3 * t * h * p * w * p).map(|i| i as f32).collect();
        let patches = patchify(&pixels, t, h, w, p);
        // token (1, 2, 1)'s channel (c 2, column 3, row 1): pixel (c 2, frame 1, row 2*4+1, column 1*4+3)
        assert_eq!(patches[((h + 2) * w + 1) * 48 + (2 * 4 + 3) * 4 + 1], pixels[((2 * t + 1) * h * p + 9) * w * p + 7]);
        assert_eq!(unpatchify(&patches, t, h, w, p), pixels);
    }

    /// The reference's tilings: a 3 x 12 x 12 latent's grid in 9 tiles of 256 pixels (the last column's 192), a 12 x 7
    /// x 7 one's in 3 of 64 frames (the first 63, the last 42 with the ghost frames), each 40 frames and 160 pixels
    /// shared, every frame covered.
    #[test]
    fn tiles_are_the_references() {
        let tiles = plan([9, 48, 48], [17, 384, 384], Some([64, 256, 256]), [40, 160], [6, 6, 6], [2, 2, 2], 4);
        assert_eq!(tiles.len(), 9);
        let widths: Vec<usize> = tiles[..3].iter().map(|t| t.pixels[2].end - t.pixels[2].start).collect();
        assert_eq!(widths, [256, 256, 192]);
        assert_eq!(tiles[1].grid[2], Interval { start: 12, end: 44, left: 20, right: 20 });
        let tiles = plan([45, 28, 28], [89, 224, 224], Some([64, 256, 256]), [40, 160], [6, 6, 6], [2, 2, 2], 4);
        let frames: Vec<(usize, usize)> = tiles.iter().map(|t| (t.pixels[0].start, t.pixels[0].end)).collect();
        assert_eq!(frames, [(0, 63), (23, 87), (47, 89)]);
        // every frame covered (where three tiles share frames the masks sum past one: the decode divides by the sum, as
        // the reference's does where they are not complementary)
        let mut sum = vec![0f32; 89];
        for t in &tiles {
            for (i, m) in t.masks[0].iter().enumerate() {
                sum[t.pixels[0].start + i] += m;
            }
        }
        assert!(sum.iter().all(|&s| s > 0.0) && sum.iter().any(|&s| s > 1.01), "{sum:?}");
        assert_eq!(trapezoid(6, 2, 3), vec![1. / 3., 2. / 3., 1., 0.75, 0.5, 0.25]);
    }

    /// The tiled decode against the reference's (`--ignored`; `OAIY_LTX_DVAE`, and `OAIY_LTX_DVAE_TILED`: folders of the
    /// reference's tiled decodes, `ref.py tiled`, each tile's noise in the order drawn).
    #[test]
    #[ignore = "needs LTX 2.5's diffusion VAE and the reference's tensors"]
    fn tiled_decodes_are_the_references() -> Result<()> {
        let (Ok(file), Ok(dirs)) = (std::env::var("OAIY_LTX_DVAE"), std::env::var("OAIY_LTX_DVAE_TILED")) else { return Ok(()) };
        let mut store = Store::open(std::path::Path::new(&file), 0)?;
        let dec = WgpuLtxDiffusionVae::load(&mut store, 0)?;
        for dir in dirs.split(':') {
            let dir = std::path::PathBuf::from(dir);
            let meta = Json::parse(&std::fs::read(dir.join("shapes.json"))?).map_err(err)?;
            let num = |v: &Json| v.as_i64().unwrap() as usize;
            let shape: Vec<usize> = meta.get("shapes").and_then(|s| s.get("latent")).and_then(Json::as_array).unwrap().iter().map(num).collect();
            let tile: Vec<usize> = meta.get("tile").and_then(Json::as_array).unwrap().iter().map(num).collect();
            let read = |name: &str| -> Vec<f32> { std::fs::read(dir.join(format!("{name}.f32"))).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() };
            let [_, _, f, h, w] = shape[..] else { panic!("shape") };
            let latent = read("latent");
            let tokens: Vec<f32> = (0..f * h * w * 128).map(|i| latent[(i % 128) * f * h * w + i / 128]).collect();
            let started = std::time::Instant::now();
            let clip = dec.decode_with(&tokens, f, h, w, Some([tile[0], tile[1], tile[1]]), |i, n| {
                let noise = read(&format!("noise{i}"));
                assert_eq!(noise.len(), n, "tile {i}'s noise");
                noise
            })?;
            let (cos, worst) = agree(&clip.flatten_all()?.to_vec1::<f32>()?, &read("pixels"));
            eprintln!("{}: {:?} in {:.1} s: cosine {cos:.6}, worst {worst:.6}", dir.display(), clip.dims(), started.elapsed().as_secs_f64());
            assert!(cos > 0.999, "{}: cosine {cos}", dir.display());
        }
        Ok(())
    }

    /// A clip's decode time (`--ignored --nocapture`; `OAIY_LTX_DVAE`, `OAIY_LTX_DVAE_SIZE`: the latent's frames, rows and
    /// columns, 7,10,16 by default: 49 frames of 512x320): a latent of noise, its tiles as the GPU's room has them.
    #[test]
    #[ignore = "needs LTX 2.5's diffusion VAE; a measurement"]
    fn measure_a_clips_decode() -> Result<()> {
        let Ok(file) = std::env::var("OAIY_LTX_DVAE") else { return Ok(()) };
        let size: Vec<usize> = std::env::var("OAIY_LTX_DVAE_SIZE").unwrap_or("7,10,16".into()).split(',').map(|v| v.trim().parse().unwrap()).collect();
        let [f, h, w] = size[..] else { panic!("OAIY_LTX_DVAE_SIZE: frames,rows,columns") };
        let mut store = Store::open(std::path::Path::new(&file), 0)?;
        let dec = WgpuLtxDiffusionVae::load(&mut store, 0)?;
        let latent = crate::common::job::noise(5, f * h * w * 128);
        let started = std::time::Instant::now();
        let mut tiles = 0;
        let clip = dec.decode_with(&latent, f, h, w, None, |i, n| {
            tiles = i + 1;
            crate::common::job::noise(9 + i as u64, n)
        })?;
        let values = clip.flatten_all()?.to_vec1::<f32>()?;
        let finite = values.iter().all(|v| v.is_finite());
        eprintln!("{:?} in {:.1} s, {tiles} tile(s), every value finite: {finite}", clip.dims(), started.elapsed().as_secs_f64());
        assert!(finite);
        Ok(())
    }

    /// The decoder against Lightricks' own (`--ignored`; `OAIY_LTX_DVAE`: the release's video VAE file,
    /// `OAIY_LTX_DVAE_GOLDEN`: a folder of the reference's tensors for one latent, `ref.py golden` in OAIY's notes):
    /// stages 1 to 3's output, and the clip from the reference's stage-5 noise.
    #[test]
    #[ignore = "needs LTX 2.5's diffusion VAE and the reference's tensors"]
    fn the_diffusion_decoder_is_the_references() -> Result<()> {
        let (Ok(file), Ok(dir)) = (std::env::var("OAIY_LTX_DVAE"), std::env::var("OAIY_LTX_DVAE_GOLDEN")) else { return Ok(()) };
        let dir = std::path::PathBuf::from(dir);
        let shapes = Json::parse(&std::fs::read(dir.join("shapes.json"))?).map_err(err)?;
        let shape = |name: &str| -> Vec<usize> { shapes.get("shapes").and_then(|s| s.get(name)).and_then(Json::as_array).unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect() };
        let read = |name: &str| -> Vec<f32> { std::fs::read(dir.join(format!("{name}.f32"))).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() };
        let mut store = Store::open(std::path::Path::new(&file), 0)?;
        let started = std::time::Instant::now();
        let dec = WgpuLtxDiffusionVae::load(&mut store, 0)?;
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());
        // (the reference's latents channels first: as tokens)
        let [_, _, t, h, w] = shape("latent_padded")[..] else { panic!("shape") };
        let padded = read("latent_padded");
        let tokens: Vec<f32> = (0..t * h * w * 128).map(|i| padded[(i % 128) * t * h * w + i / 128]).collect();
        let started = std::time::Instant::now();
        let (grid, x) = dec.stages_1_to_3(&tokens, (t, h, w));
        let mut rec = dec.gpu.begin();
        rec.read(&x);
        let got = rec.finish().pop().unwrap();
        let (cos, worst) = agree(&got, &read("stage3_up"));
        eprintln!("stages 1 to 3 ({grid:?}) in {:.2} s: cosine {cos:.6}, worst {worst:.4}", started.elapsed().as_secs_f64());
        assert!(cos > 0.9999, "stages 1 to 3: cosine {cos}");
        let started = std::time::Instant::now();
        let pixels = dec.stages_4_and_5(x, grid, &read("noise"), true, true)?;
        let want = read("pixels");
        let (cos, worst) = agree(&pixels, &want);
        eprintln!("stages 4 and 5 in {:.2} s: the clip's cosine {cos:.6}, worst {worst:.6}", started.elapsed().as_secs_f64());
        // (OAIY_CHAIN_PROFILE: their kernels' times)
        if ggml_rs_wgpu::profile::chain_on() {
            let mut kernels = ggml_rs_wgpu::profile::take_kernels();
            kernels.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (name, seconds, n) in kernels.iter().take(12) {
                eprintln!("  {name}: {seconds:.3} s in {n}");
            }
        }
        assert!(cos > 0.999, "the clip: cosine {cos}");
        // the whole decode from the unpadded latent, the same noise
        let [_, _, f, lh, lw] = shape("latent")[..] else { panic!("shape") };
        let latent = read("latent");
        let tokens: Vec<f32> = (0..f * lh * lw * 128).map(|i| latent[(i % 128) * f * lh * lw + i / 128]).collect();
        let noise = read("noise");
        let clip = dec.decode_with(&tokens, f, lh, lw, None, |tile, n| {
            assert_eq!((tile, n), (0, noise.len()));
            noise.clone()
        })?;
        let (cos, worst) = agree(&clip.flatten_all()?.to_vec1::<f32>()?, &want[..3 * (8 * (f - 1) + 1) * lh * 32 * lw * 32]);
        eprintln!("decoded whole: cosine {cos:.6}, worst {worst:.4}");
        assert!(cos > 0.999, "decoded whole: cosine {cos}");
        Ok(())
    }
}
