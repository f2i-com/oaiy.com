//! Pixal3D's decoders (TRELLIS.2's):
//! - `StructureDecoder`: the sparse structure latent (8 × 16³) to 64³ occupancy,
//!   a dense 3D conv net with 3D pixel shuffles;
//! - `SparseDecoder`: a structured latent (32 channels per voxel) up four
//!   levels of sparse ConvNeXt blocks, each voxel split into the children the
//!   decoder keeps (channel-to-space), to 7 values per voxel (shape: a dual
//!   vertex, three edge-crossing flags and a quad split weight) or 6 (texture:
//!   base colour, metallic, roughness, alpha).
//! As the reference runs them: the blocks in F16, norms in F32, the input and
//! output layers in F32.
use super::sparse::{layer_norm, Conv, Level, Subdivision};
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::path::Path;
use std::sync::Arc;

/// A linear layer in a chosen dtype.
struct Lin {
    w: Tensor,
    b: Tensor,
}
impl Lin {
    fn load(store: &mut Store, prefix: &str, dtype: DType, dev: &Device) -> Result<Self> {
        Ok(Self { w: store.tensor(&format!("{prefix}.weight"), dev, false)?.to_dtype(dtype)?.t()?.contiguous()?, b: store.tensor(&format!("{prefix}.bias"), dev, false)?.to_dtype(dtype)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.to_dtype(self.w.dtype())?.matmul(&self.w)?.broadcast_add(&self.b)
    }
}

struct Norm {
    w: Option<Tensor>,
    b: Option<Tensor>,
    eps: f64,
}
impl Norm {
    fn affine(store: &mut Store, prefix: &str, eps: f64, dev: &Device) -> Result<Self> {
        Ok(Self { w: Some(store.tensor_f32(&format!("{prefix}.weight"), dev)?), b: Some(store.tensor_f32(&format!("{prefix}.bias"), dev)?), eps })
    }
    fn plain(eps: f64) -> Self {
        Self { w: None, b: None, eps }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, self.w.as_ref(), self.b.as_ref(), self.eps)
    }
}

fn conv(store: &mut Store, prefix: &str, dense: bool, dtype: DType, dev: &Device) -> Result<Conv> {
    let w = store.tensor(&format!("{prefix}.weight"), dev, false)?;
    let b = store.tensor(&format!("{prefix}.bias"), dev, false)?;
    if dense {
        Conv::dense(&w, Some(b), dtype)
    } else {
        Conv::sparse(&w, Some(b), dtype)
    }
}

fn read_json(path: &Path) -> Result<Json> {
    Json::parse(&std::fs::read(path)?).map_err(candle_core::Error::wrap)
}

fn ints(j: &Json, key: &str) -> Vec<usize> {
    j.get(key).and_then(Json::as_array).map(|a| a.iter().filter_map(Json::as_i64).map(|v| v as usize).collect()).unwrap_or_default()
}

// --- the sparse structure decoder -----------------------------------------------------

struct DenseRes {
    norm1: Norm,
    norm2: Norm,
    conv1: Conv,
    conv2: Conv,
}

enum DenseBlock {
    Res(DenseRes),
    /// A conv to 8× the channels, then a 3D pixel shuffle.
    Up(Conv),
}

pub struct StructureDecoder {
    input: Conv,
    middle: Vec<DenseRes>,
    blocks: Vec<DenseBlock>,
    out_norm: Norm,
    out_conv: Conv,
    /// The latent's grid (16).
    pub res: usize,
}

impl StructureDecoder {
    /// `path`: the checkpoint without its extension (`.json` and `.safetensors` beside).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let config = read_json(&path.with_extension("json"))?;
        let args = config.get("args").ok_or_else(|| candle_core::Error::Msg("structure decoder config: no args".into()))?;
        let channels = ints(args, "channels");
        let res_blocks = args.get("num_res_blocks").and_then(Json::as_i64).unwrap_or(2) as usize;
        let middle = args.get("num_res_blocks_middle").and_then(Json::as_i64).unwrap_or(2) as usize;
        let mut store = Store::open(&path.with_extension("safetensors"), 0)?;
        let f16 = DType::F16;
        let res_block = |store: &mut Store, p: String| -> Result<DenseRes> {
            Ok(DenseRes { norm1: Norm::affine(store, &format!("{p}.norm1"), 1e-5, dev)?, norm2: Norm::affine(store, &format!("{p}.norm2"), 1e-5, dev)?, conv1: conv(store, &format!("{p}.conv1"), true, f16, dev)?, conv2: conv(store, &format!("{p}.conv2"), true, f16, dev)? })
        };
        let mut blocks = Vec::new();
        let mut i = 0;
        for (level, _) in channels.iter().enumerate() {
            for _ in 0..res_blocks {
                blocks.push(DenseBlock::Res(res_block(&mut store, format!("blocks.{i}"))?));
                i += 1;
            }
            if level + 1 < channels.len() {
                blocks.push(DenseBlock::Up(conv(&mut store, &format!("blocks.{i}.conv"), true, f16, dev)?));
                i += 1;
            }
        }
        Ok(Self {
            input: conv(&mut store, "input_layer", true, DType::F32, dev)?,
            middle: (0..middle).map(|m| res_block(&mut store, format!("middle_block.{m}"))).collect::<Result<_>>()?,
            blocks,
            out_norm: Norm::affine(&mut store, "out_layer.0", 1e-5, dev)?,
            out_conv: conv(&mut store, "out_layer.2", true, DType::F32, dev)?,
            res: 16,
        })
    }

    fn res_block(b: &DenseRes, level: &Level, x: &Tensor) -> Result<Tensor> {
        let h = b.norm1.forward(x)?.silu()?;
        let h = b.conv1.forward(level, &h)?;
        let h = b.norm2.forward(&h)?.silu()?;
        b.conv2.forward(level, &h)? + x
    }

    /// The latent [16³, 8] (voxels x slowest) to occupancy logits at the finest grid, and its size.
    pub fn forward(&self, latent: &Tensor) -> Result<(Tensor, usize)> {
        let dev = latent.device();
        let mut res = self.res;
        let mut level = Level::dense(res, dev)?;
        let mut h = self.input.forward(&level, &latent.to_dtype(DType::F32)?)?.to_dtype(DType::F16)?;
        for b in &self.middle {
            h = Self::res_block(b, &level, &h)?;
        }
        for block in &self.blocks {
            match block {
                DenseBlock::Res(b) => h = Self::res_block(b, &level, &h)?,
                DenseBlock::Up(c) => {
                    let y = c.forward(&level, &h)?;
                    h = pixel_shuffle(&y, res)?;
                    res *= 2;
                    level = Level::dense(res, dev)?;
                }
            }
        }
        let h = self.out_norm.forward(&h.to_dtype(DType::F32)?)?.silu()?;
        Ok((self.out_conv.forward(&level, &h)?, res))
    }
}

/// 3D pixel shuffle of dense voxels (x slowest): [r³, 8·C] to [(2r)³, C], a
/// channel's sub-index being x·4 + y·2 + z as in `pixel_shuffle_3d`.
fn pixel_shuffle(y: &Tensor, r: usize) -> Result<Tensor> {
    let (n, c8) = y.dims2()?;
    let c = c8 / 8;
    // [N, C, 8] -> [N, 8, C]: row parent·8 + sub is that child's features.
    let rows = y.reshape((n, c, 8))?.transpose(1, 2)?.contiguous()?.reshape((n * 8, c))?;
    let r2 = r * 2;
    let mut ids = Vec::with_capacity(n * 8);
    for x in 0..r2 {
        for yy in 0..r2 {
            for z in 0..r2 {
                let parent = ((x / 2) * r + yy / 2) * r + z / 2;
                let sub = (x % 2) * 4 + (yy % 2) * 2 + z % 2;
                ids.push((parent * 8 + sub) as u32);
            }
        }
    }
    rows.index_select(&Tensor::from_vec(ids, n * 8, y.device())?, 0)
}

// --- the sparse decoders ------------------------------------------------------------------

struct ConvNeXt {
    conv: Conv,
    norm: Norm,
    fc1: Lin,
    fc2: Lin,
}

impl ConvNeXt {
    fn forward(&self, level: &Level, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(level, x)?;
        let h = self.norm.forward(&h)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.silu()?)?;
        h + x
    }
}

/// Up one level: channel-to-space into the children kept.
struct Up {
    norm1: Norm,
    norm2: Norm,
    conv1: Conv,
    conv2: Conv,
    to_subdiv: Option<Lin>,
    channels: usize,
    out: usize,
}

pub struct SparseDecoder {
    from_latent: Lin,
    levels: Vec<(Vec<ConvNeXt>, Option<Up>)>,
    output: Lin,
    /// The finest grid (the shape decoder's `resolution`, e.g. 256 for a 16³-voxel latent... set per run).
    pub resolution: usize,
    pub out_channels: usize,
}

/// What a sparse decoder made: the finest level's voxels and their values, and the
/// children it kept at each step up (a texture decoder follows a shape decoder's).
pub struct Decoded {
    pub level: Arc<Level>,
    pub feats: Tensor,
    pub subdivisions: Vec<Subdivision>,
}

impl SparseDecoder {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let config = read_json(&path.with_extension("json"))?;
        let args = config.get("args").ok_or_else(|| candle_core::Error::Msg("decoder config: no args".into()))?;
        let channels = ints(args, "model_channels");
        let counts = ints(args, "num_blocks");
        let pred_subdiv = args.get("pred_subdiv").and_then(Json::as_bool).unwrap_or(true);
        let name = config.get("name").and_then(Json::as_str).unwrap_or("");
        let out_channels = if name == "FlexiDualGridVaeDecoder" { 7 } else { args.get("out_channels").and_then(Json::as_i64).unwrap_or(6) as usize };
        let resolution = args.get("resolution").and_then(Json::as_i64).unwrap_or(256) as usize;
        let mut store = Store::open(&path.with_extension("safetensors"), 0)?;
        let f16 = DType::F16;
        let mut levels = Vec::new();
        for (i, &ch) in channels.iter().enumerate() {
            let mut blocks = Vec::new();
            for j in 0..counts[i] {
                let p = format!("blocks.{i}.{j}");
                blocks.push(ConvNeXt {
                    conv: conv(&mut store, &format!("{p}.conv"), false, f16, dev)?,
                    norm: Norm::affine(&mut store, &format!("{p}.norm"), 1e-6, dev)?,
                    fc1: Lin::load(&mut store, &format!("{p}.mlp.0"), f16, dev)?,
                    fc2: Lin::load(&mut store, &format!("{p}.mlp.2"), f16, dev)?,
                });
            }
            let up = if i + 1 < channels.len() {
                let p = format!("blocks.{i}.{}", counts[i]);
                Some(Up {
                    norm1: Norm::affine(&mut store, &format!("{p}.norm1"), 1e-6, dev)?,
                    norm2: Norm::plain(1e-6),
                    conv1: conv(&mut store, &format!("{p}.conv1"), false, f16, dev)?,
                    conv2: conv(&mut store, &format!("{p}.conv2"), false, f16, dev)?,
                    to_subdiv: if pred_subdiv { Some(Lin::load(&mut store, &format!("{p}.to_subdiv"), f16, dev)?) } else { None },
                    channels: ch,
                    out: channels[i + 1],
                })
            } else {
                None
            };
            levels.push((blocks, up));
        }
        Ok(Self { from_latent: Lin::load(&mut store, "from_latent", DType::F32, dev)?, levels, output: Lin::load(&mut store, "output_layer", DType::F32, dev)?, resolution, out_channels })
    }

    /// The voxels a latent would have after `times` steps up (the cascade's finer
    /// structure: TRELLIS.2's `upsample`), without decoding them.
    pub fn upsample(&self, level: Arc<Level>, latent: &Tensor, times: usize) -> Result<Arc<Level>> {
        self.run(level, latent, None, times, |_, _| {}).map(|d| d.level)
    }

    /// From a latent [N, 32] on `level` (its coordinates) to the finest level. A
    /// texture decoder gets the shape decoder's `guide` (its choice of children).
    pub fn forward(&self, level: Arc<Level>, latent: &Tensor, guide: Option<&[Subdivision]>, report: impl FnMut(usize, usize)) -> Result<Decoded> {
        self.run(level, latent, guide, usize::MAX, report)
    }

    fn run(&self, level: Arc<Level>, latent: &Tensor, guide: Option<&[Subdivision]>, stop_at: usize, mut report: impl FnMut(usize, usize)) -> Result<Decoded> {
        let dev = latent.device().clone();
        let mut level = level;
        let mut h = self.from_latent.forward(&latent.to_dtype(DType::F32)?)?.to_dtype(DType::F16)?;
        let mut subdivisions = Vec::new();
        let total = self.levels.len();
        for (i, (blocks, up)) in self.levels.iter().enumerate() {
            if i == stop_at {
                return Ok(Decoded { level, feats: h, subdivisions });
            }
            report(i, total);
            for b in blocks {
                h = b.forward(&level, &h)?;
            }
            let Some(up) = up else { continue };
            let sub = match (&up.to_subdiv, guide) {
                (Some(lin), _) => Subdivision::from_logits(&lin.forward(&h)?)?,
                (None, Some(g)) => Subdivision { parent: g[i].parent.clone(), child: g[i].child.clone() },
                (None, None) => candle_core::bail!("a decoder that does not choose its children needs a guide"),
            };
            let x = h.clone();
            let hh = up.norm1.forward(&h)?.silu()?;
            let hh = up.conv1.forward(&level, &hh)?;
            let next = Level::new(sub.coords(&level), level.res * 2, &dev)?;
            let hh = sub.channel_to_space(&hh)?;
            let x = sub.channel_to_space(&x)?;
            let hh = up.norm2.forward(&hh)?.silu()?;
            let hh = up.conv2.forward(&next, &hh)?;
            // The skip repeats each of the child's channels out / (channels / 8) times.
            let repeat = up.out / (up.channels / 8);
            let (n, c) = x.dims2()?;
            let skip = x.unsqueeze(2)?.expand((n, c, repeat))?.reshape((n, c * repeat))?;
            h = (hh + skip)?;
            level = next;
            subdivisions.push(sub);
        }
        let h = layer_norm(&h.to_dtype(DType::F32)?, None, None, 1e-5)?;
        Ok(Decoded { level, feats: self.output.forward(&h)?, subdivisions })
    }
}
