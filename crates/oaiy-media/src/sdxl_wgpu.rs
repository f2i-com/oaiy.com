//! SDXL's UNet on WebGPU ([`ggml_rs_wgpu`]'s chain of ops), as [`crate::sdxl::unet`] computes it: an image its
//! pixels' rows of channels (the tensor cores' tokens), the convolutions and the matmuls on the tensor cores where the
//! adapter has them, the weights f16 (the checkpoint's FP16 as it is, an f32 one's rounded to the nearest). A guided
//! step's two images (the prompt's and the negative prompt's) are two passes of one recording, each with its own
//! embedding and its own text keys and values: a group norm's statistics, a self-attention's keys and a
//! convolution's edges are one image's. The timestep's and the label's embeddings are two small MLPs on the host; a
//! prompt's keys and values for each cross-attention are made once ([`WgpuUnet::prepare`]). A pass's vectors are
//! taken from a pool and given back as their values die (a layer's would otherwise all stand until the step ran:
//! some 10 GB at 1024x1024).
use crate::sdxl::config::UNetConfig;
use crate::weights::Weights;
use candle_core::{DType, Device, Module, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use std::cell::RefCell;
use std::collections::HashMap;

/// The group norms' groups (the reference's: 32 whatever the config's, but for the last norm, the config's).
const GROUPS: usize = 32;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A convolution's weights as [`DeviceChain::conv_weights`] packs them (`k` by `k`, 1 or 3), and its bias.
struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cout: usize,
    cin: usize,
    k: usize,
}

/// A group norm's weight and bias.
struct Norm {
    w: DeviceVec,
    b: DeviceVec,
}

/// A matrix `[n, k]` as f16, and its bias where it has one.
struct Lin {
    w: DeviceVec,
    b: Option<DeviceVec>,
    n: usize,
    k: usize,
}

struct Res {
    n1: Norm,
    c1: Conv,
    emb: Lin,
    n2: Norm,
    c2: Conv,
    skip: Option<Conv>,
}

/// A transformer block: each layer norm as a modulation (`[weight - 1, bias]`), each attention's keys' and values'
/// matrices one (`to_k`'s rows, then `to_v`'s: a position's key then its value, the row the chain's attention reads).
struct Layer {
    norm1: DeviceVec,
    q1: Lin,
    kv1: Lin,
    o1: Lin,
    norm2: DeviceVec,
    q2: Lin,
    kv2: Lin,
    o2: Lin,
    norm3: DeviceVec,
    ff1: Lin,
    ff2: Lin,
}

struct Spatial {
    norm: Norm,
    proj_in: Lin,
    layers: Vec<Layer>,
    proj_out: Lin,
    heads: usize,
}

enum Op {
    Conv(Conv),
    Res(Res),
    Spatial(Spatial),
    /// A 3x3 convolution of stride 2: of stride 1, its even pixels kept.
    Down(Conv),
    /// Each pixel four, then a 3x3 convolution.
    Up(Conv),
}

/// `W2 silu(W0 x + b0) + b2` on the host (the timestep's and the label's embeddings: a few million products a step).
struct Mlp {
    a: candle_nn::Linear,
    b: candle_nn::Linear,
}

impl Mlp {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_slice(x, (1, x.len()), &Device::Cpu)?;
        let h = self.a.forward(&x)?;
        self.b.forward(&(&h * candle_nn::ops::sigmoid(&h)?)?)?.flatten_all()?.to_vec1::<f32>()
    }
}

struct Loader<'a> {
    w: &'a mut Weights,
    gpu: &'a ggml_rs_wgpu::WgpuBackend,
    prefix: &'a str,
}

impl Loader<'_> {
    fn values(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self.w.tensor(&format!("{}{key}", self.prefix), &Device::Cpu, DType::F32)?;
        Ok((t.flatten_all()?.to_vec1::<f32>()?, t.dims().to_vec()))
    }

    fn vec(&mut self, key: &str) -> Result<DeviceVec> {
        let (values, _) = self.values(key)?;
        let v = self.gpu.vec(values.len());
        self.gpu.upload(&v, &values);
        Ok(v)
    }

    fn conv(&mut self, name: &str) -> Result<Conv> {
        let (values, dims) = self.values(&format!("{name}.weight"))?;
        let &[cout, cin, k, kw] = dims.as_slice() else { candle_core::bail!("{name}: a convolution of shape {dims:?}") };
        if k != kw || !matches!(k, 1 | 3) {
            candle_core::bail!("{name}: a {k}x{kw} convolution");
        }
        let w = self.gpu.conv_weights(&values, cout, cin, k).ok_or_else(|| err(format!("{name}: a convolution's weight past f16's range")))?;
        Ok(Conv { w, b: self.vec(&format!("{name}.bias"))?, cout, cin, k })
    }

    fn norm(&mut self, name: &str) -> Result<Norm> {
        Ok(Norm { w: self.vec(&format!("{name}.weight"))?, b: self.vec(&format!("{name}.bias"))? })
    }

    /// `names`' matrices one below another as f16 (each `[_, k]`), with the first's bias where `bias`.
    fn lin(&mut self, names: &[&str], bias: bool) -> Result<Lin> {
        let (mut all, mut n, mut k) = (Vec::new(), 0, 0);
        for name in names {
            let (values, dims) = self.values(&format!("{name}.weight"))?;
            let &[rows, cols] = dims.as_slice() else { candle_core::bail!("{name}: a matrix of shape {dims:?}") };
            if k != 0 && cols != k {
                candle_core::bail!("{name}: {cols} wide beside {k}");
            }
            (n, k) = (n + rows, cols);
            all.extend(values);
        }
        let words = crate::wgpu_weights::f16_words_f32(&all).ok_or_else(|| err(format!("{}: a weight past f16's range", names[0])))?;
        let w = self.gpu.vec(words.len());
        self.gpu.upload(&w, &words);
        Ok(Lin { w, b: if bias { Some(self.vec(&format!("{}.bias", names[0]))?) } else { None }, n, k })
    }

    /// A layer norm's weight and bias as the modulation a modulated norm takes: `[weight - 1, bias]`.
    fn layer_norm(&mut self, name: &str) -> Result<DeviceVec> {
        let (mut values, _) = self.values(&format!("{name}.weight"))?;
        values.iter_mut().for_each(|v| *v -= 1.0);
        values.extend(self.values(&format!("{name}.bias"))?.0);
        let v = self.gpu.vec(values.len());
        self.gpu.upload(&v, &values);
        Ok(v)
    }

    fn mlp(&mut self, a: &str, b: &str) -> Result<Mlp> {
        let prefix = self.prefix;
        let mut one = |name: &str| -> Result<candle_nn::Linear> {
            Ok(candle_nn::Linear::new(
                self.w.tensor(&format!("{prefix}{name}.weight"), &Device::Cpu, DType::F32)?,
                Some(self.w.tensor(&format!("{prefix}{name}.bias"), &Device::Cpu, DType::F32)?),
            ))
        };
        Ok(Mlp { a: one(a)?, b: one(b)? })
    }

    fn res(&mut self, p: &str, cin: usize, cout: usize) -> Result<Res> {
        Ok(Res {
            n1: self.norm(&format!("{p}.in_layers.0"))?,
            c1: self.conv(&format!("{p}.in_layers.2"))?,
            emb: self.lin(&[&format!("{p}.emb_layers.1")], true)?,
            n2: self.norm(&format!("{p}.out_layers.0"))?,
            c2: self.conv(&format!("{p}.out_layers.3"))?,
            skip: if cin != cout { Some(self.conv(&format!("{p}.skip_connection"))?) } else { None },
        })
    }

    fn spatial(&mut self, p: &str, c: usize, layers: usize, head: usize) -> Result<Spatial> {
        Ok(Spatial {
            norm: self.norm(&format!("{p}.norm"))?,
            proj_in: self.lin(&[&format!("{p}.proj_in")], true)?,
            layers: (0..layers)
                .map(|i| {
                    let b = format!("{p}.transformer_blocks.{i}");
                    Ok(Layer {
                        norm1: self.layer_norm(&format!("{b}.norm1"))?,
                        q1: self.lin(&[&format!("{b}.attn1.to_q")], false)?,
                        kv1: self.lin(&[&format!("{b}.attn1.to_k"), &format!("{b}.attn1.to_v")], false)?,
                        o1: self.lin(&[&format!("{b}.attn1.to_out.0")], true)?,
                        norm2: self.layer_norm(&format!("{b}.norm2"))?,
                        q2: self.lin(&[&format!("{b}.attn2.to_q")], false)?,
                        kv2: self.lin(&[&format!("{b}.attn2.to_k"), &format!("{b}.attn2.to_v")], false)?,
                        o2: self.lin(&[&format!("{b}.attn2.to_out.0")], true)?,
                        norm3: self.layer_norm(&format!("{b}.norm3"))?,
                        ff1: self.lin(&[&format!("{b}.ff.net.0.proj")], true)?,
                        ff2: self.lin(&[&format!("{b}.ff.net.2")], true)?,
                    })
                })
                .collect::<Result<_>>()?,
            proj_out: self.lin(&[&format!("{p}.proj_out")], true)?,
            heads: c / head,
        })
    }
}

/// A prompt's conditioning: each cross-attention's keys and values of its text (`rows` positions, in the order a pass
/// meets them), and its label's embedding (the pooled text's and the picture's size's, through the label's MLP).
pub struct Cond {
    kv: Vec<DeviceVec>,
    rows: usize,
    label: Vec<f32>,
}

pub struct WgpuUnet {
    gpu: ggml_rs_wgpu::WgpuBackend,
    cfg: UNetConfig,
    time: Mlp,
    label: Mlp,
    input: Vec<Vec<Op>>,
    middle: Vec<Op>,
    output: Vec<Vec<Op>>,
    out_norm: Norm,
    out_conv: Conv,
    /// The vectors free for a pass to take, by their length.
    pool: RefCell<HashMap<usize, Vec<DeviceVec>>>,
}

impl WgpuUnet {
    /// The UNet `cfg` describes from `w`'s tensors under `prefix` (a checkpoint's `model.diffusion_model.`), on GPU
    /// `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(w: &mut Weights, prefix: &str, cfg: &UNetConfig, device: usize) -> Result<Self> {
        Self::load_on(w, prefix, cfg, ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?)
    }

    pub fn load_on(w: &mut Weights, prefix: &str, cfg: &UNetConfig, gpu: ggml_rs_wgpu::WgpuBackend) -> Result<Self> {
        let (cs, per, head) = (&cfg.block_out_channels, &cfg.transformer_layers_per_block, cfg.attention_head_dim);
        if cs.is_empty() || per.len() != cs.len() || cs.iter().any(|c| c % GROUPS != 0) || cs.iter().zip(per).any(|(c, n)| *n > 0 && c % head != 0) || cs[0] % cfg.norm_groups != 0 {
            candle_core::bail!("a UNet of channels {cs:?}, transformer layers {per:?} and heads of {head}");
        }
        let mut l = Loader { w, gpu: &gpu, prefix };
        let time = l.mlp("time_embed.0", "time_embed.2")?;
        let label = l.mlp("label_emb.0.0", "label_emb.0.2")?;
        // the input blocks: the first convolution, then each level's residual blocks (with their transformers) and
        // its downsampling; `skips` their outputs' channels, what the output blocks take beside their own
        let mut input = vec![vec![Op::Conv(l.conv("input_blocks.0.0")?)]];
        let mut skips = vec![cs[0]];
        let (mut at, mut c) = (1, cs[0]);
        for (i, &out) in cs.iter().enumerate() {
            for _ in 0..cfg.layers_per_block {
                let mut ops = vec![Op::Res(l.res(&format!("input_blocks.{at}.0"), c, out)?)];
                if per[i] > 0 {
                    ops.push(Op::Spatial(l.spatial(&format!("input_blocks.{at}.1"), out, per[i], head)?));
                }
                input.push(ops);
                skips.push(out);
                (at, c) = (at + 1, out);
            }
            if i + 1 < cs.len() {
                input.push(vec![Op::Down(l.conv(&format!("input_blocks.{at}.0.op"))?)]);
                skips.push(out);
                at += 1;
            }
        }
        let last = cs.len() - 1;
        let middle = vec![
            Op::Res(l.res("middle_block.0", c, c)?),
            Op::Spatial(l.spatial("middle_block.1", c, per[last], head)?),
            Op::Res(l.res("middle_block.2", c, c)?),
        ];
        let mut output = Vec::new();
        let mut at = 0;
        for i in (0..cs.len()).rev() {
            let out = cs[i];
            for j in 0..=cfg.layers_per_block {
                let skip = skips.pop().ok_or_else(|| err("a UNet's output blocks past its input blocks"))?;
                let mut ops = vec![Op::Res(l.res(&format!("output_blocks.{at}.0"), c + skip, out)?)];
                if per[i] > 0 {
                    ops.push(Op::Spatial(l.spatial(&format!("output_blocks.{at}.1"), out, per[i], head)?));
                }
                if j == cfg.layers_per_block && i != 0 {
                    ops.push(Op::Up(l.conv(&format!("output_blocks.{at}.{}.conv", if per[i] > 0 { 2 } else { 1 }))?));
                }
                output.push(ops);
                (at, c) = (at + 1, out);
            }
        }
        let (out_norm, out_conv) = (l.norm("out.0")?, l.conv("out.2")?);
        Ok(Self { gpu, cfg: cfg.clone(), time, label, input, middle, output, out_norm, out_conv, pool: RefCell::new(HashMap::new()) })
    }

    /// A vector of `len` from the pool, or a new one.
    fn take(&self, len: usize) -> DeviceVec {
        let len = len.max(1);
        self.pool.borrow_mut().get_mut(&len).and_then(Vec::pop).unwrap_or_else(|| self.gpu.vec(len))
    }

    /// `v` back to the pool: nothing recorded after reads what it holds.
    fn give(&self, v: DeviceVec) {
        self.pool.borrow_mut().entry(v.len).or_default().push(v);
    }

    /// Let go of the pool's vectors (a picture of another size takes others).
    pub fn forget(&self) {
        self.pool.borrow_mut().clear();
        self.gpu.settle();
    }

    fn spatials(&self) -> impl Iterator<Item = &Spatial> {
        self.input.iter().flatten().chain(&self.middle).chain(self.output.iter().flatten()).filter_map(|op| match op {
            Op::Spatial(s) => Some(s),
            _ => None,
        })
    }

    /// A prompt's conditioning: `context` its text's states (`[rows, cross_attention_dim]`), `y` its label
    /// (`[label_emb_in_dim]`).
    pub fn prepare(&self, context: &[f32], y: &[f32]) -> Result<Cond> {
        let width = self.cfg.cross_attention_dim;
        if context.is_empty() || context.len() % width != 0 || y.len() != self.cfg.label_emb_in_dim {
            candle_core::bail!("a prompt's context of {} values ({width} a position) and label of {}", context.len(), y.len());
        }
        let rows = context.len() / width;
        let ctx = self.gpu.vec(context.len());
        self.gpu.upload(&ctx, context);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let kv = self
            .spatials()
            .flat_map(|s| &s.layers)
            .map(|l| {
                let kv = self.gpu.vec(rows * l.kv2.n);
                rec.matmul_f16_rows(&l.kv2.w, l.kv2.n, l.kv2.k, &ctx, &kv, rows);
                kv
            })
            .collect();
        rec.finish();
        Ok(Cond { kv, rows, label: self.label.forward(y)? })
    }

    fn conv(&self, r: &mut dyn ChainRecorder, c: &Conv, x: &DeviceVec, h: usize, w: usize) -> DeviceVec {
        let y = self.take(h * w * c.cout);
        r.conv_rows(&c.w, &c.b, c.cout, c.cin, c.k, x, h, w, &y);
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn group_norm(&self, r: &mut dyn ChainRecorder, n: &Norm, x: &DeviceVec, px: usize, c: usize, groups: usize, eps: f32, silu: bool) -> DeviceVec {
        let (y, stats) = (self.take(px * c), self.take(groups * (px.div_ceil(256) + 1) * 2));
        r.group_norm_rows(x, &n.w, &n.b, &y, &stats, px, c, groups, eps, silu);
        self.give(stats);
        y
    }

    fn lin(&self, r: &mut dyn ChainRecorder, l: &Lin, x: &DeviceVec, rows: usize) -> DeviceVec {
        let y = self.take(rows * l.n);
        r.matmul_f16_rows(&l.w, l.n, l.k, x, &y, rows);
        if let Some(b) = &l.b {
            r.add_bias_rows(&y, b, rows, l.n);
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn res(&self, r: &mut dyn ChainRecorder, b: &Res, x: &DeviceVec, c: usize, h: usize, w: usize, emb: &DeviceVec) -> DeviceVec {
        let (px, cout) = (h * w, b.c2.cout);
        let n = self.group_norm(r, &b.n1, x, px, c, GROUPS, 1e-5, true);
        let h1 = self.conv(r, &b.c1, &n, h, w);
        self.give(n);
        // the embedding's projection (its SiLU the host's), on every pixel
        let e = self.lin(r, &b.emb, emb, 1);
        r.add_bias_rows(&h1, &e, px, cout);
        self.give(e);
        let n = self.group_norm(r, &b.n2, &h1, px, cout, GROUPS, 1e-5, true);
        self.give(h1);
        let h2 = self.conv(r, &b.c2, &n, h, w);
        self.give(n);
        match &b.skip {
            Some(s) => {
                let shortcut = self.conv(r, s, x, h, w);
                r.add(&h2, &shortcut);
                self.give(shortcut);
            }
            None => r.add(&h2, x),
        }
        h2
    }

    /// An attention of `q`'s `px` queries over `kv`'s `positions`, through `o` and onto the stream `t`.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, r: &mut dyn ChainRecorder, q: &DeviceVec, kv: &DeviceVec, positions: usize, o: &Lin, t: &DeviceVec, px: usize, heads: usize) {
        let hd = o.k / heads;
        let att = self.take(self.gpu.attention_rows_full_out_len(px, heads, hd, positions));
        r.attention_rows_full(q, kv, &att, px, heads, heads, hd, positions, 1.0 / (hd as f32).sqrt());
        let out = self.lin(r, o, &att, px);
        self.give(att);
        r.add(t, &out);
        self.give(out);
    }

    #[allow(clippy::too_many_arguments)]
    fn spatial(&self, r: &mut dyn ChainRecorder, s: &Spatial, x: &DeviceVec, c: usize, px: usize, cond: &Cond, site: &mut usize) -> DeviceVec {
        let n = self.group_norm(r, &s.norm, x, px, c, GROUPS, 1e-6, false);
        let t = self.lin(r, &s.proj_in, &n, px);
        self.give(n);
        let d = s.proj_in.n;
        for l in &s.layers {
            // the self-attention: each pixel over all of them
            let a = self.take(px * d);
            r.norm_mod_rows(&t, &a, px, d, &l.norm1, 0, Some(d), RowNorm::Layer, 1e-5);
            let (q, kv) = (self.lin(r, &l.q1, &a, px), self.lin(r, &l.kv1, &a, px));
            self.attend(r, &q, &kv, px, &l.o1, &t, px, s.heads);
            self.give(q);
            self.give(kv);
            // the text's: its keys and values the prompt's
            r.norm_mod_rows(&t, &a, px, d, &l.norm2, 0, Some(d), RowNorm::Layer, 1e-5);
            let q = self.lin(r, &l.q2, &a, px);
            self.attend(r, &q, &cond.kv[*site], cond.rows, &l.o2, &t, px, s.heads);
            self.give(q);
            *site += 1;
            // GEGLU's feed-forward
            r.norm_mod_rows(&t, &a, px, d, &l.norm3, 0, Some(d), RowNorm::Layer, 1e-5);
            let fused = self.lin(r, &l.ff1, &a, px);
            self.give(a);
            let ff = l.ff1.n / 2;
            let gated = self.take(px * ff);
            r.geglu_rows(&fused, &gated, px, ff);
            self.give(fused);
            let out = self.lin(r, &l.ff2, &gated, px);
            self.give(gated);
            r.add(&t, &out);
            self.give(out);
        }
        let out = self.lin(r, &s.proj_out, &t, px);
        self.give(t);
        r.add(&out, x);
        out
    }

    /// A block's ops in turn over `x` (`[h w, c]`), which is given back to the pool after its first with `own`: its
    /// output, channels, rows and columns.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, r: &mut dyn ChainRecorder, ops: &[Op], x: &DeviceVec, own: bool, c: usize, h: usize, w: usize, emb: &DeviceVec, cond: &Cond, site: &mut usize) -> (DeviceVec, usize, usize, usize) {
        let (mut cur, mut own, mut c, mut h, mut w) = (x.clone(), own, c, h, w);
        for op in ops {
            let (y, cy, hy, wy) = match op {
                Op::Conv(cv) => (self.conv(r, cv, &cur, h, w), cv.cout, h, w),
                Op::Res(b) => (self.res(r, b, &cur, c, h, w, emb), b.c2.cout, h, w),
                Op::Spatial(s) => (self.spatial(r, s, &cur, c, h * w, cond, site), c, h, w),
                Op::Down(cv) => {
                    let full = self.conv(r, cv, &cur, h, w);
                    let y = self.take(h * w * cv.cout / 4);
                    r.subsample2x_even_rows(&full, &y, h, w, cv.cout);
                    self.give(full);
                    (y, cv.cout, h / 2, w / 2)
                }
                Op::Up(cv) => {
                    let up = self.take(4 * h * w * c);
                    r.upsample2x_rows(&cur, &up, h, w, c);
                    let y = self.conv(r, cv, &up, 2 * h, 2 * w);
                    self.give(up);
                    (y, cv.cout, 2 * h, 2 * w)
                }
            };
            if own {
                self.give(cur);
            }
            (cur, own, c, h, w) = (y, true, cy, hy, wy);
        }
        (cur, c, h, w)
    }

    /// One image's pass: `x` its latent (`[h w, in_channels]`), `emb` the SiLU of its embedding, into a vector of the
    /// noise (`[h w, out_channels]`, the caller's to give back); with `trace`, a copy of each block's output there.
    #[allow(clippy::too_many_arguments)]
    fn pass(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, emb: &DeviceVec, cond: &Cond, h: usize, w: usize, mut trace: Option<&mut Vec<DeviceVec>>) -> DeviceVec {
        let mut keep = |r: &mut dyn ChainRecorder, v: &DeviceVec, len: usize| {
            if let Some(t) = trace.as_deref_mut() {
                let copy = self.gpu.vec(len);
                r.copy(v, 0, &copy, 0, len);
                t.push(copy);
            }
        };
        let mut site = 0;
        let (mut cur, mut c, mut h, mut w) = (x.clone(), self.cfg.in_channels, h, w);
        // each input block's output is kept for the output block that takes it beside its own input
        let mut skips: Vec<(DeviceVec, usize)> = Vec::new();
        for ops in &self.input {
            let (y, cy, hy, wy) = self.block(r, ops, &cur, false, c, h, w, emb, cond, &mut site);
            (cur, c, h, w) = (y, cy, hy, wy);
            keep(r, &cur, h * w * c);
            skips.push((cur.clone(), c));
        }
        // (the middle reads the last input block's output, a skip: its own output, and each output block's, the pool's)
        (cur, c, h, w) = self.block(r, &self.middle, &cur, false, c, h, w, emb, cond, &mut site);
        keep(r, &cur, h * w * c);
        for ops in &self.output {
            let (skip, sc) = skips.pop().expect("an output block's skip");
            let px = h * w;
            let cat = self.take(px * (c + sc));
            r.store_rows(&cur, &cat, px, c, 0, c + sc, 0);
            r.store_rows(&skip, &cat, px, sc, 0, c + sc, c);
            self.give(cur);
            // (an input block's output is no one's after its output block: the first, the first convolution's, too)
            self.give(skip);
            (cur, c, h, w) = self.block(r, ops, &cat, true, c + sc, h, w, emb, cond, &mut site);
            keep(r, &cur, h * w * c);
        }
        let n = self.group_norm(r, &self.out_norm, &cur, h * w, c, self.cfg.norm_groups, 1e-5, true);
        self.give(cur);
        let out = self.conv(r, &self.out_conv, &n, h, w);
        self.give(n);
        out
    }

    /// The SiLU of an image's embedding at `timestep`: the timestep's sinusoids through their MLP, plus the label's.
    fn embedding(&self, timestep: f32, cond: &Cond) -> Result<DeviceVec> {
        let half = self.cfg.block_out_channels[0] / 2;
        let args: Vec<f32> = (0..half).map(|i| timestep * (-(10_000f64.ln()) * i as f64 / half as f64).exp() as f32).collect();
        let sinusoids: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let e: Vec<f32> = self.time.forward(&sinusoids)?.iter().zip(&cond.label).map(|(t, l)| { let v = t + l; v / (1.0 + (-v).exp()) }).collect();
        let v = self.gpu.vec(e.len());
        self.gpu.upload(&v, &e);
        Ok(v)
    }

    /// The noise the UNet predicts in `latent` (`[h w, in_channels]`, a pixel's channels a row) at `timestep` for
    /// each of `conds` (a guided step's prompt and negative prompt: a pass each, one recording): each
    /// `[h w, out_channels]`.
    pub fn eps(&self, latent: &[f32], h: usize, w: usize, timestep: f32, conds: &[&Cond]) -> Result<Vec<Vec<f32>>> {
        Ok(self.run(latent, h, w, timestep, conds, false)?.0)
    }

    /// [`Self::eps`], and with `traced` each block's output of each pass (the input blocks', the middle's, the output
    /// blocks'; `[h w, c]` at its own size).
    fn run(&self, latent: &[f32], h: usize, w: usize, timestep: f32, conds: &[&Cond], traced: bool) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let levels = self.cfg.block_out_channels.len() - 1;
        if latent.len() != h * w * self.cfg.in_channels || h == 0 || w == 0 || h % (1 << levels) != 0 || w % (1 << levels) != 0 {
            candle_core::bail!("a latent of {} values for {h}x{w} pixels of {} (each a multiple of {})", latent.len(), self.cfg.in_channels, 1 << levels);
        }
        let x = self.gpu.vec(latent.len());
        self.gpu.upload(&x, latent);
        let embs = conds.iter().map(|c| self.embedding(timestep, c)).collect::<Result<Vec<_>>>()?;
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let mut trace = Vec::new();
        let outs: Vec<DeviceVec> = conds.iter().zip(&embs).map(|(c, e)| self.pass(rec.as_mut(), &x, e, c, h, w, traced.then_some(&mut trace))).collect();
        for o in &outs {
            rec.read_range(o, 0, h * w * self.cfg.out_channels);
        }
        for t in &trace {
            rec.read(t);
        }
        let mut read = rec.finish();
        let traces = read.split_off(outs.len());
        outs.into_iter().for_each(|o| self.give(o));
        Ok((read, traces))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdxl::unet::UNet2DConditionModel;
    use candle_nn::{VarBuilder, VarMap};

    fn rng(mut seed: u64) -> impl FnMut() -> f32 {
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
        }
    }

    /// `[1, c, h, w]` as its pixels' rows of channels.
    fn rows_of(t: &Tensor) -> Result<Vec<f32>> {
        t.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    fn compare(what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        let (mut dot, mut gg, mut ww, mut worst) = (0f64, 0f64, 0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            let (g, w) = (*g as f64, *w as f64);
            dot += g * w;
            gg += g * g;
            ww += w * w;
            worst = worst.max((g - w).abs());
        }
        let (cosine, rms) = (dot / (gg.sqrt() * ww.sqrt()).max(1e-30), (ww / want.len() as f64).sqrt());
        eprintln!("{what}: cosine {cosine:.6}, the worst error {worst:.2e} of an RMS of {rms:.3}");
        (cosine, worst / rms.max(1e-30))
    }

    /// A UNet's noise on WebGPU against Candle's on the CPU in f32, block by block: `cfg`'s UNet from `path`, a latent
    /// of `h` by `w` and a text of 77 positions, two conditionings in one recording (each its own pass).
    fn against_candle(path: &std::path::Path, cfg: &UNetConfig, gpu: ggml_rs_wgpu::WgpuBackend, h: usize, w: usize, least: f64) -> Result<()> {
        const PREFIX: &str = "model.diffusion_model.";
        let mut weights = Weights::open(path)?;
        let tensors: HashMap<String, Tensor> = weights.names().into_iter().filter_map(|n| n.strip_prefix(PREFIX).map(|k| (n.clone(), k.to_owned()))).map(|(n, k)| Ok((k, weights.tensor(&n, &Device::Cpu, DType::F32)?))).collect::<Result<_>>()?;
        let reference = UNet2DConditionModel::load(cfg, VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu))?;
        let unet = WgpuUnet::load_on(&mut weights, PREFIX, cfg, gpu)?;
        let mut next = rng(0x5d);
        let rows = 77;
        let latent: Vec<f32> = (0..cfg.in_channels * h * w).map(|_| 2.0 * next()).collect();
        let x = Tensor::from_vec(latent, (1, cfg.in_channels, h, w), &Device::Cpu)?;
        let conds: Vec<(Vec<f32>, Vec<f32>)> = (0..2).map(|_| ((0..rows * cfg.cross_attention_dim).map(|_| next()).collect(), (0..cfg.label_emb_in_dim).map(|_| next()).collect())).collect();
        let timestep = 731.0f32;
        let prepared = conds.iter().map(|(c, y)| unet.prepare(c, y)).collect::<Result<Vec<_>>>()?;
        let clock = std::time::Instant::now();
        let (eps, traces) = unet.run(&rows_of(&x)?, h, w, timestep, &prepared.iter().collect::<Vec<_>>(), true)?;
        eprintln!("two passes on WebGPU, traced: {:.2} s", clock.elapsed().as_secs_f64());
        let blocks = traces.len() / 2;
        for (i, (context, y)) in conds.iter().enumerate() {
            let mut trace = Vec::new();
            let clock = std::time::Instant::now();
            let want = reference.forward_traced(
                &x,
                &Tensor::from_vec(vec![timestep], (1,), &Device::Cpu)?,
                &Tensor::from_slice(context, (1, rows, cfg.cross_attention_dim), &Device::Cpu)?,
                &Tensor::from_slice(y, (1, cfg.label_emb_in_dim), &Device::Cpu)?,
                Some(&mut trace),
            )?;
            eprintln!("conditioning {i}: Candle's pass {:.2} s", clock.elapsed().as_secs_f64());
            assert_eq!(trace.len(), blocks, "the blocks traced");
            for (b, t) in trace.iter().enumerate() {
                let (cosine, _) = compare(&format!("conditioning {i}, block {b} {:?}", t.dims()), &traces[i * blocks + b], &rows_of(t)?);
                assert!(cosine > least, "conditioning {i}, block {b}: cosine {cosine}");
            }
            let (cosine, worst) = compare(&format!("conditioning {i}, the noise"), &eps[i], &rows_of(&want)?);
            assert!(cosine > least && worst < 0.2, "conditioning {i}: cosine {cosine}, worst {worst}");
        }
        // the two conditionings' passes are their own: the noise differs between them as Candle's does
        let (between, _) = compare("the noise between the two conditionings", &eps[0], &eps[1]);
        assert!(between < 0.9999, "the two conditionings' noise is one: {between}");
        // a second step from the pool's vectors is the first
        let again = unet.eps(&rows_of(&x)?, h, w, timestep, &prepared.iter().collect::<Vec<_>>())?;
        assert!(compare("the step again", &again[0], &eps[0]).0 > 0.999999 && compare("the step again", &again[1], &eps[1]).0 > 0.999999);
        Ok(())
    }

    /// A small UNet of SDXL's shape (three levels, the lower two with transformers, heads of 64) with random weights,
    /// norms' and biases' too: WebGPU's noise is Candle's, block by block.
    #[test]
    fn a_small_unet_on_webgpu_is_candles() -> Result<()> {
        let Ok(gpu) = ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)) else { return Ok(()) };
        let cfg = UNetConfig {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![64, 128, 256],
            layers_per_block: 2,
            transformer_layers_per_block: vec![0, 1, 2],
            attention_head_dim: 64,
            time_embed_dim: 256,
            label_emb_in_dim: 96,
            cross_attention_dim: 128,
            norm_groups: 32,
        };
        let map = VarMap::new();
        UNet2DConditionModel::load(&cfg, VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu).pp("model.diffusion_model"))?;
        // (the norms' weights and every bias are ones and zeros as made: random ones tell a norm or a bias left out)
        let mut next = rng(0xa11ce);
        for (name, var) in map.data().lock().unwrap().iter() {
            if var.dims().len() == 1 {
                let n = var.dims()[0];
                let weight = name.ends_with("weight");
                let values: Vec<f32> = (0..n).map(|_| if weight { 1.0 + 0.3 * next() } else { 0.2 * next() }).collect();
                var.set(&Tensor::from_vec(values, n, &Device::Cpu)?)?;
            }
        }
        let dir = std::env::temp_dir().join(format!("oaiy-wgpu-unet-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("unet.safetensors");
        map.save(&path)?;
        let result = against_candle(&path, &cfg, gpu, 24, 32, 0.9995);
        std::fs::remove_dir_all(dir)?;
        result
    }

    /// SDXL's own UNet (`OAIY_SDXL_CHECKPOINT`, a checkpoint on disk) on WebGPU against Candle's on the CPU in f32,
    /// block by block, a latent of 64 by 64 (`OAIY_SDXL_LATENT` its side).
    #[test]
    #[ignore = "needs an SDXL checkpoint (OAIY_SDXL_CHECKPOINT) and some 16 GB of memory"]
    fn sdxls_unet_on_webgpu_is_candles() -> Result<()> {
        let path = std::path::PathBuf::from(std::env::var("OAIY_SDXL_CHECKPOINT").map_err(err)?);
        let side = std::env::var("OAIY_SDXL_LATENT").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(err)?;
        against_candle(&path, &UNetConfig::sdxl_1_0(), gpu, side, side, 0.999)
    }
}
