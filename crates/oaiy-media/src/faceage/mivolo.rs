//! MiVOLO v2 (Irina Tolstykh and Maxim Kuprashevich; `iitolstykh/mivolo_v2`, Apache-2.0): a person's age from a crop of
//! their face and a crop of their body, computed as the authors' model computes it (their MiVOLO code on timm 0.8's
//! VOLO-D1, 384 by 384).
//!
//! Two stems (a 7x7 convolution at stride 2 and two 3x3, each with batch norm and ReLU) and a 4x4 patch projection make
//! the face and the body each 192 channels on 48 by 48; a cross attention (one head, relative positions; face to body
//! and body to face, a layer norm across channels and positions, a 1x1 MLP) fuses them. Then VOLO-D1: four outlooker
//! blocks (outlook attention, 6 heads, 3x3 at stride 2), a 2x2 downsample to 384 channels on 24 by 24 with a learnt
//! position embedding, fourteen transformer blocks (12 heads), two class-attention blocks on a class token, a layer
//! norm, and the head plus half the auxiliary head's largest answer over the tokens. Its third output is the age,
//! scaled by the model's configuration (0 to 122 around 61); the first two (gender) are not read.
//!
//! Float32, on the device given. The weights are the published `model.safetensors`, read whole (no memory mapping).
use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

/// The side of both crops the model takes.
pub const SIDE: usize = 384;
const PREFIX: &str = "mivolo.model.";
/// `config.json`: the age is `raw * (MAX_AGE - MIN_AGE) + AVG_AGE`.
const MIN_AGE: f64 = 0.;
const MAX_AGE: f64 = 122.;
const AVG_AGE: f64 = 61.;
/// Torch's layer and batch norm epsilon.
const EPS: f64 = 1e-5;

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.0
            .get(&format!("{PREFIX}{name}"))
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("not MiVOLO v2's weights: {PREFIX}{name} is missing")))
    }
}

struct Linear {
    /// [in, out]
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    fn load(w: &Weights, name: &str, bias: bool) -> Result<Self> {
        Ok(Self { w: w.get(&format!("{name}.weight"))?.t()?.contiguous()?, b: if bias { Some(w.get(&format!("{name}.bias"))?) } else { None } })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (last, lead) = dims.split_last().ok_or_else(|| candle_core::Error::Msg("a linear layer on a scalar".into()))?;
        let rows: usize = lead.iter().product();
        let y = x.reshape((rows, *last))?.matmul(&self.w)?;
        let y = match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut out = lead.to_vec();
        out.push(self.w.dim(1)?);
        y.reshape(out)
    }
}

/// Layer norm over the last dimension.
struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self { w: w.get(&format!("{name}.weight"))?, b: w.get(&format!("{name}.bias"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let centred = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
        centred.broadcast_div(&(var + EPS)?.sqrt()?)?.broadcast_mul(&self.w)?.broadcast_add(&self.b)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self { fc1: Linear::load(w, &format!("{name}.fc1"), true)?, fc2: Linear::load(w, &format!("{name}.fc2"), true)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

/// A convolution, with its bias if it has one.
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv {
    fn load(w: &Weights, name: &str, bias: bool, stride: usize, padding: usize) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{name}.weight"))?,
            b: if bias { Some(w.get(&format!("{name}.bias"))?.reshape((1, (), 1, 1))?) } else { None },
            stride,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d(&self.w, self.padding, self.stride, 1, 1)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// Batch norm as it runs in evaluation: a scale and a shift a channel.
struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        let scale = w.get(&format!("{name}.weight"))?.broadcast_div(&(w.get(&format!("{name}.running_var"))? + EPS)?.sqrt()?)?;
        let shift = w.get(&format!("{name}.bias"))?.broadcast_sub(&w.get(&format!("{name}.running_mean"))?.broadcast_mul(&scale)?)?;
        Ok(Self { scale: scale.reshape((1, (), 1, 1))?, shift: shift.reshape((1, (), 1, 1))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)
    }
}

/// A stem and its patch projection: [B, 3, 384, 384] to [B, 192, 48, 48].
struct Stem {
    convs: [(Conv, BatchNorm); 3],
    proj: Conv,
}

impl Stem {
    fn load(w: &Weights, stem: &str, proj: &str) -> Result<Self> {
        Ok(Self {
            convs: [
                (Conv::load(w, &format!("{stem}.0"), false, 2, 3)?, BatchNorm::load(w, &format!("{stem}.1"))?),
                (Conv::load(w, &format!("{stem}.3"), false, 1, 1)?, BatchNorm::load(w, &format!("{stem}.4"))?),
                (Conv::load(w, &format!("{stem}.6"), false, 1, 1)?, BatchNorm::load(w, &format!("{stem}.7"))?),
            ],
            proj: Conv::load(w, proj, true, 4, 0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (conv, bn) in &self.convs {
            x = bn.forward(&conv.forward(&x)?)?.relu()?;
        }
        self.proj.forward(&x)
    }
}

/// The relative position logits of timm's `PosEmbedRel` for queries [B, H*W, d] on an H by W grid: for the query at
/// (h, w) and the key at (h', w'), `q . height_rel[h' - h + H - 1] + q . width_rel[w' - w + W - 1]`, as [B, HW, HW].
fn relative_logits(q: &Tensor, height_rel: &Tensor, width_rel: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (b, _, d) = q.dims3()?;
    let q4 = q.reshape((b, h, w, d))?;
    // [B, H, Wq, Wk]: along the width, for each query column, its window of the 2W - 1 relative logits.
    let along = |q4: &Tensor, rel: &Tensor, n: usize| -> Result<Tensor> {
        let (b, m, _, d) = q4.dims4()?;
        let logits = q4.reshape((b * m * n, d))?.matmul(&rel.t()?)?.reshape((b, m, n, 2 * n - 1))?;
        let rows = (0..n).map(|i| logits.narrow(2, i, 1)?.narrow(3, n - 1 - i, n)).collect::<Result<Vec<_>>>()?;
        Tensor::cat(&rows, 2)
    };
    let width = along(&q4, width_rel, w)?; // [B, H, Wq, Wk]
    let height = along(&q4.transpose(1, 2)?.contiguous()?, height_rel, h)?; // [B, Wq, Hq, Hk]
    let width = width.unsqueeze(3)?; // [B, Hq, Wq, 1, Wk]
    let height = height.transpose(1, 2)?.unsqueeze(4)?; // [B, Hq, Wq, Hk, 1]
    width.broadcast_add(&height)?.reshape((b, h * w, h * w))
}

/// MiVOLO's fusion of the face and the body (`CrossBottleneckAttn`, one head, stride 1).
struct Fusion {
    qkv_face: Conv,
    qkv_body: Conv,
    height_rel: Tensor,
    width_rel: Tensor,
    norm_w: Tensor,
    norm_b: Tensor,
    fc1: Conv,
    fc2: Conv,
    dim: usize,
}

impl Fusion {
    fn load(w: &Weights) -> Result<Self> {
        let m = "patch_embed.map";
        Ok(Self {
            qkv_face: Conv::load(w, &format!("{m}.qkv_f"), false, 1, 0)?,
            qkv_body: Conv::load(w, &format!("{m}.qkv_p"), false, 1, 0)?,
            height_rel: w.get(&format!("{m}.pos_embed.height_rel"))?,
            width_rel: w.get(&format!("{m}.pos_embed.width_rel"))?,
            norm_w: w.get(&format!("{m}.norm.weight"))?.unsqueeze(0)?,
            norm_b: w.get(&format!("{m}.norm.bias"))?.unsqueeze(0)?,
            fc1: Conv::load(w, &format!("{m}.mlp.fc1"), true, 1, 0)?,
            fc2: Conv::load(w, &format!("{m}.mlp.fc2"), true, 1, 0)?,
            dim: 192,
        })
    }

    /// q [B, HW, d], k [B, d, HW], v [B, HW, d] from a [B, 192, H, W] map.
    fn qkv(&self, x: &Tensor, conv: &Conv) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, _, h, w) = x.dims4()?;
        let y = conv.forward(x)?;
        let d = self.dim;
        let q = y.narrow(1, 0, d)?.reshape((b, d, h * w))?.transpose(1, 2)?.contiguous()?;
        let k = y.narrow(1, d, d)?.reshape((b, d, h * w))?.contiguous()?;
        let v = y.narrow(1, 2 * d, d)?.reshape((b, d, h * w))?.transpose(1, 2)?.contiguous()?;
        Ok((q, k, v))
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let scale = (self.dim as f64).powf(-0.5);
        let attn = (q.matmul(k)? * scale)?.add(&relative_logits(q, &self.height_rel, &self.width_rel, h, w)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let (b, _, d) = v.dims3()?;
        attn.matmul(v)?.transpose(1, 2)?.reshape((b, d, h, w))
    }

    fn forward(&self, face: &Tensor, body: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = face.dims4()?;
        let (qf, kf, vf) = self.qkv(face, &self.qkv_face)?;
        let (qp, kp, vp) = self.qkv(body, &self.qkv_body)?;
        let to_face = self.attend(&qf, &kp, &vp, h, w)?; // person to face
        let to_body = self.attend(&qp, &kf, &vf, h, w)?; // face to person
        let x = Tensor::cat(&[to_face, to_body], 1)?;
        // A layer norm over channels and positions together, with an affine a channel and position.
        let flat = x.reshape((b, ()))?;
        let centred = flat.broadcast_sub(&flat.mean_keepdim(1)?)?;
        let var = centred.sqr()?.mean_keepdim(1)?;
        let x = centred.broadcast_div(&(var + EPS)?.sqrt()?)?.reshape(x.dims())?;
        let x = x.broadcast_mul(&self.norm_w)?.broadcast_add(&self.norm_b)?;
        self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)
    }
}

/// Outlook attention, 3x3 at stride 2, padding 1, on [B, H, W, C].
struct Outlook {
    v: Linear,
    attn: Linear,
    proj: Linear,
    heads: usize,
}

const K: usize = 3;

impl Outlook {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
        let d = c / self.heads;
        // unfold: [B, C, 9, oh, ow], the value at (2i + kh - 1, 2j + kw - 1) for each (kh, kw).
        let v = self.v.forward(x)?.permute((0, 3, 1, 2))?;
        let padded = v.pad_with_zeros(2, 1, 2 * oh + 1 - h)?.pad_with_zeros(3, 1, 2 * ow + 1 - w)?;
        let mut taps = Vec::with_capacity(K * K);
        for kh in 0..K {
            for kw in 0..K {
                let window = padded.narrow(2, kh, 2 * oh)?.narrow(3, kw, 2 * ow)?;
                taps.push(window.reshape((b, c, oh, 2, ow, 2))?.narrow(3, 0, 1)?.narrow(5, 0, 1)?.reshape((b, c, oh, ow))?);
            }
        }
        let unfolded = Tensor::stack(&taps, 2)?; // [B, C, 9, oh, ow]
        let v = unfolded.reshape((b, self.heads, d, K * K, oh * ow))?.permute((0, 1, 4, 3, 2))?.contiguous()?; // [B, H, N, 9, d]
        // The attention: from the 2x2 average of the input, 9x9 weights for each head and output position.
        let pooled = x.reshape((b, oh, 2, ow, 2, c))?.mean(4)?.mean(2)?; // h and w are even here
        let attn = self.attn.forward(&pooled)?.reshape((b, oh * ow, self.heads, K * K, K * K))?.permute((0, 2, 1, 3, 4))?;
        let attn = candle_nn::ops::softmax_last_dim(&(attn * (d as f64).powf(-0.5))?.contiguous()?)?;
        // (candle multiplies batches of matrices in three dimensions, not five)
        let n = oh * ow;
        let out = attn.reshape((b * self.heads * n, K * K, K * K))?.matmul(&v.reshape((b * self.heads * n, K * K, d))?)?;
        let out = out.reshape((b, self.heads, n, K * K, d))?; // [B, H, N, 9, d]
        let out = out.permute((0, 1, 4, 3, 2))?.reshape((b, c, K * K, oh, ow))?;
        // fold: each tap back to (2i + kh - 1, 2j + kw - 1), summed where they overlap.
        let mut folded: Option<Tensor> = None;
        for kh in 0..K {
            for kw in 0..K {
                let tap = out.narrow(2, kh * K + kw, 1)?.reshape((b, c, oh, 1, ow, 1))?;
                let spread = tap.pad_with_zeros(3, 0, 1)?.pad_with_zeros(5, 0, 1)?.reshape((b, c, 2 * oh, 2 * ow))?;
                let placed = spread.pad_with_zeros(2, kh, K - 1 - kh)?.pad_with_zeros(3, kw, K - 1 - kw)?;
                folded = Some(match folded {
                    Some(f) => f.add(&placed)?,
                    None => placed,
                });
            }
        }
        let folded = folded.expect("nine taps").narrow(2, 1, h)?.narrow(3, 1, w)?;
        self.proj.forward(&folded.permute((0, 2, 3, 1))?.contiguous()?)
    }
}

/// Multi-head self attention on [B, N, C].
struct Attention {
    qkv: Linear,
    proj: Linear,
    heads: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let d = c / self.heads;
        let qkv = self.qkv.forward(x)?.reshape((b, n, 3, self.heads, d))?.permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (qkv.get(0)?.contiguous()?, qkv.get(1)?.contiguous()?, qkv.get(2)?.contiguous()?);
        let attn = (q.matmul(&k.t()?)? * (d as f64).powf(-0.5))?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        self.proj.forward(&attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?)
    }
}

enum Mixer {
    Outlook(Outlook),
    Attention(Attention),
}

/// An outlooker or a transformer block: pre-norm, a mixer, a pre-norm MLP, both residual.
struct Block {
    norm1: Norm,
    mixer: Mixer,
    norm2: Norm,
    mlp: Mlp,
}

impl Block {
    fn load(w: &Weights, name: &str, outlook: bool, heads: usize) -> Result<Self> {
        let a = format!("{name}.attn");
        let mixer = if outlook {
            Mixer::Outlook(Outlook {
                v: Linear::load(w, &format!("{a}.v"), false)?,
                attn: Linear::load(w, &format!("{a}.attn"), true)?,
                proj: Linear::load(w, &format!("{a}.proj"), true)?,
                heads,
            })
        } else {
            Mixer::Attention(Attention { qkv: Linear::load(w, &format!("{a}.qkv"), false)?, proj: Linear::load(w, &format!("{a}.proj"), true)?, heads })
        };
        Ok(Self { norm1: Norm::load(w, &format!("{name}.norm1"))?, mixer, norm2: Norm::load(w, &format!("{name}.norm2"))?, mlp: Mlp::load(w, &format!("{name}.mlp"))? })
    }

    /// On [B, H, W, C].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.norm1.forward(x)?;
        let mixed = match &self.mixer {
            Mixer::Outlook(o) => o.forward(&y)?,
            Mixer::Attention(a) => {
                let (b, h, w, c) = y.dims4()?;
                a.forward(&y.reshape((b, h * w, c))?)?.reshape((b, h, w, c))?
            }
        };
        let x = x.add(&mixed)?;
        x.add(&self.mlp.forward(&self.norm2.forward(&x)?)?)
    }
}

/// Class attention: the class token attends to every token.
struct ClassBlock {
    norm1: Norm,
    kv: Linear,
    q: Linear,
    proj: Linear,
    norm2: Norm,
    mlp: Mlp,
    heads: usize,
}

impl ClassBlock {
    fn load(w: &Weights, name: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            norm1: Norm::load(w, &format!("{name}.norm1"))?,
            kv: Linear::load(w, &format!("{name}.attn.kv"), false)?,
            q: Linear::load(w, &format!("{name}.attn.q"), false)?,
            proj: Linear::load(w, &format!("{name}.attn.proj"), true)?,
            norm2: Norm::load(w, &format!("{name}.norm2"))?,
            mlp: Mlp::load(w, &format!("{name}.mlp"))?,
            heads,
        })
    }

    /// On [B, N, C], the class token first.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let d = c / self.heads;
        let y = self.norm1.forward(x)?;
        let kv = self.kv.forward(&y)?.reshape((b, n, 2, self.heads, d))?.permute((2, 0, 3, 1, 4))?;
        let (k, v) = (kv.get(0)?.contiguous()?, kv.get(1)?.contiguous()?);
        let q = self.q.forward(&y.narrow(1, 0, 1)?)?.reshape((b, self.heads, 1, d))?;
        let attn = candle_nn::ops::softmax_last_dim(&(q * (d as f64).powf(-0.5))?.matmul(&k.t()?)?)?;
        let attended = self.proj.forward(&attn.matmul(&v)?.transpose(1, 2)?.reshape((b, 1, c))?)?;
        let cls = x.narrow(1, 0, 1)?.add(&attended)?;
        let cls = cls.add(&self.mlp.forward(&self.norm2.forward(&cls)?)?)?;
        Tensor::cat(&[cls, x.narrow(1, 1, n - 1)?], 1)
    }
}

/// What the network gives at each part, for checking a port against the reference.
#[derive(Default)]
pub struct Trace {
    pub parts: Vec<(String, Tensor)>,
}

pub struct MiVolo {
    face_stem: Stem,
    body_stem: Stem,
    fusion: Fusion,
    stage1: Vec<Block>,
    downsample: Conv,
    pos_embed: Tensor,
    stage2: Vec<Vec<Block>>,
    cls_token: Tensor,
    post: Vec<ClassBlock>,
    norm: Norm,
    head: Linear,
    aux_head: Linear,
    dev: Device,
}

impl MiVolo {
    /// The published `model.safetensors` of `iitolstykh/mivolo_v2`.
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let all = candle_core::safetensors::load(path, dev)?;
        let w = Weights(all.into_iter().filter(|(_, t)| t.dtype() != DType::I64).map(|(k, t)| t.to_dtype(DType::F32).map(|t| (k, t))).collect::<Result<_>>()?);
        // MiVOLO v2's shape: VOLO-D1 on both stems, 3 outputs (2 gender, 1 age).
        if w.get("head.weight")?.dims() != [3, 384] || w.get("patch_embed.proj2.weight")?.dims() != [192, 64, 4, 4] || w.get("pos_embed")?.dims() != [1, 24, 24, 384] {
            candle_core::bail!("{}: not MiVOLO v2 (mivolo_d1_384 on a face and a body)", path.display());
        }
        let stage = |index: usize, count: usize, outlook: bool, heads: usize| -> Result<Vec<Block>> {
            (0..count).map(|i| Block::load(&w, &format!("network.{index}.{i}"), outlook, heads)).collect()
        };
        Ok(Self {
            face_stem: Stem::load(&w, "patch_embed.conv1", "patch_embed.proj1")?,
            body_stem: Stem::load(&w, "patch_embed.conv2", "patch_embed.proj2")?,
            fusion: Fusion::load(&w)?,
            stage1: stage(0, 4, true, 6)?,
            downsample: Conv::load(&w, "network.1.proj", true, 2, 0)?,
            pos_embed: w.get("pos_embed")?,
            stage2: vec![stage(2, 4, false, 12)?, stage(3, 8, false, 12)?, stage(4, 2, false, 12)?],
            cls_token: w.get("cls_token")?,
            post: (0..2).map(|i| ClassBlock::load(&w, &format!("post_network.{i}"), 12)).collect::<Result<_>>()?,
            norm: Norm::load(&w, "norm")?,
            head: Linear::load(&w, "head", true)?,
            aux_head: Linear::load(&w, "aux_head", true)?,
            dev: dev.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The model's three outputs for faces and bodies [B, 3, 384, 384] each, normalised as `prep` makes them.
    pub fn head(&self, faces: &Tensor, bodies: &Tensor, mut trace: Option<&mut Trace>) -> Result<Tensor> {
        let mut note = |name: &str, t: &Tensor| {
            if let Some(trace) = trace.as_deref_mut() {
                trace.parts.push((name.to_string(), t.clone()));
            }
        };
        let face = self.face_stem.forward(faces)?;
        note("embed_face", &face);
        let body = self.body_stem.forward(bodies)?;
        note("embed_body", &body);
        let fused = self.fusion.forward(&face, &body)?;
        note("fused", &fused);
        let mut x = fused.permute((0, 2, 3, 1))?.contiguous()?;
        for block in &self.stage1 {
            x = block.forward(&x)?;
        }
        note("network.0", &x);
        x = self.downsample.forward(&x.permute((0, 3, 1, 2))?)?.permute((0, 2, 3, 1))?.contiguous()?;
        note("network.1", &x);
        x = x.broadcast_add(&self.pos_embed)?;
        for (i, stage) in self.stage2.iter().enumerate() {
            for block in stage {
                x = block.forward(&x)?;
            }
            note(&format!("network.{}", i + 2), &x);
        }
        let (b, h, w, c) = x.dims4()?;
        let tokens = x.reshape((b, h * w, c))?;
        let mut x = Tensor::cat(&[self.cls_token.broadcast_as((b, 1, c))?, tokens], 1)?;
        for block in &self.post {
            x = block.forward(&x)?;
        }
        note("post", &x);
        let x = self.norm.forward(&x)?;
        note("norm", &x);
        let out = self.head.forward(&x.narrow(1, 0, 1)?.squeeze(1)?)?;
        let aux = self.aux_head.forward(&x.narrow(1, 1, h * w)?)?.max(1)?;
        let out = out.add(&(aux * 0.5)?)?;
        note("head", &out);
        Ok(out)
    }

    /// The ages, in years, for faces and bodies [B, 3, 384, 384] each.
    pub fn ages(&self, faces: &Tensor, bodies: &Tensor) -> Result<Vec<f64>> {
        let head = self.head(faces, bodies, None)?;
        let raw: Vec<f32> = head.narrow(1, 2, 1)?.flatten_all()?.to_vec1()?;
        Ok(raw.into_iter().map(|r| r as f64 * (MAX_AGE - MIN_AGE) + AVG_AGE).collect())
    }
}
