//! Bounded weight residency. SSD reads use the published safetensors directly.
//!
//! Besides BF16/F16/F32 weights, fp8 (F8_E4M3) and int8 (I8) checkpoints load
//! as published: each weight may carry a per-tensor (or per-row)
//! `<name>.weight_scale`, as ComfyUI's scaled checkpoints store it. They stay at
//! their stored size on disk and in the RAM tier, and become BF16 on the device.
//! Transformers saved with bare names (`patchify_proj.weight`) are served under
//! the `model.diffusion_model.` prefix the loader uses.
use candle_core::{DType, Device, Result, Tensor};
use dsv41::safetensors::{Dtype, StIndex};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedding_rows_preserve_order_and_check_bounds() -> Result<()> {
        let path = std::env::temp_dir().join(format!("nrob-rows-{}-{}.safetensors", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let header = br#"{"embedding":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header);
        file.extend((1..=6).flat_map(|n| (n as f32).to_le_bytes()));
        std::fs::write(&path, file)?;
        let mut store = Store::open(&path, 0)?;
        let selected = store.rows("embedding", &[2, 0, 2], &Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(selected, vec![5., 6., 1., 2., 5., 6.]);
        assert!(store.rows("embedding", &[3], &Device::Cpu).is_err());
        assert_eq!(store.disk_bytes, 24);
        std::fs::remove_file(path)?;
        Ok(())
    }

    fn write_st(name: &str, tensors: &[(&str, &str, &[usize], Vec<u8>)], meta: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("nrob-{name}-{}.safetensors", std::process::id()));
        let (mut header, mut data) = (format!("{{\"__metadata__\":{meta}"), Vec::new());
        for (k, dtype, shape, bytes) in tensors {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header += &format!(",\"{k}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", dims.join(","), data.len(), data.len() + bytes.len());
            data.extend_from_slice(bytes);
        }
        header.push('}');
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend(data);
        std::fs::write(&path, file).unwrap();
        path
    }

    #[test]
    #[ignore = "needs a CUDA device and --features cuda"]
    fn quantized_weights_decode_the_same_on_the_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let bytes: Vec<u8> = (0..=255u8).collect();
        for (dtype, scale) in [(Dtype::F8E4M3, Some(&[0.37f32][..])), (Dtype::I8, Some(&[0.01f32][..])), (Dtype::F8E4M3, None)] {
            let cpu = decode("x.weight", dtype, &[16, 16], &bytes, scale, &Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let gpu = decode("x.weight", dtype, &[16, 16], &bytes, scale, &dev)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
                assert!(a == b || (a.is_nan() && b.is_nan()), "{dtype:?} byte {i}: cpu {a} gpu {b}");
            }
        }
        Ok(())
    }

    #[test]
    fn the_e4m3_table_matches_the_reference_conversion() -> Result<()> {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let reference = Tensor::from_raw_buffer(&bytes, DType::F8E4M3, &[256], &Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for (b, (want, got)) in reference.iter().zip(e4m3_table()).enumerate() {
            assert!(want == got || (want.is_nan() && got.is_nan()), "byte {b:#04x}: {want} vs {got}");
        }
        Ok(())
    }

    #[test]
    fn scaled_fp8_and_int8_weights_decode_to_bf16() -> Result<()> {
        // fp8 e4m3: 0x38 = 1.0, 0x40 = 2.0, 0xB8 = -1.0, 0x00 = 0.
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let path = write_st("quant", &[
            ("patchify_proj.weight", "F8_E4M3", &[2, 2], vec![0x38, 0x40, 0xB8, 0x00]),
            ("patchify_proj.weight_scale", "F32", &[1], f32s(&[0.5])),
            ("emb.weight", "I8", &[3, 2], vec![1, 255, 127, 128, 0, 2]),
            ("emb.weight_scale", "F32", &[1], f32s(&[0.25])),
            ("rows.weight", "I8", &[2, 2], vec![2, 4, 2, 4]),
            ("rows.weight_scale", "F32", &[2], f32s(&[1.0, 0.5])),
        ], "{\"model_version\":\"2.5.0\"}");
        let mut store = Store::open(&path, 1 << 20)?;
        let get = |t: Tensor| t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Bare transformer names are served under the loader's prefix.
        let w = store.tensor("model.diffusion_model.patchify_proj.weight", &Device::Cpu, true)?;
        assert_eq!(get(w), vec![0.5, 1.0, -0.5, 0.0]);
        // The RAM tier keeps the stored bytes and decodes the same way.
        let again = store.tensor("model.diffusion_model.patchify_proj.weight", &Device::Cpu, true)?;
        assert_eq!(get(again), vec![0.5, 1.0, -0.5, 0.0]);
        assert_eq!(store.host_bytes, 4);
        let rows = store.rows("model.diffusion_model.emb.weight", &[1, 0], &Device::Cpu)?;
        assert_eq!(get(rows), vec![31.75, -32.0, 0.25, -0.25]);
        let per_row = store.tensor("model.diffusion_model.rows.weight", &Device::Cpu, false)?;
        assert_eq!(get(per_row), vec![2.0, 4.0, 1.0, 2.0]);
        // Scales are applied, not loaded as weights of their own.
        let g = store.group("model.diffusion_model.", &Device::Cpu, false, |_| true)?;
        assert!(g.tensors.keys().all(|k| !k.ends_with("weight_scale")));
        std::fs::remove_file(path)?;
        Ok(())
    }
}

pub struct Store {
    pub index: StIndex,
    host: HashMap<String, Vec<u8>>,
    /// `weight_scale` values already read, by weight name.
    scales: HashMap<String, Option<Vec<f32>>>,
    pub host_bytes: u64,
    pub disk_bytes: u64,
    budget: u64,
}
impl Store {
    pub fn open(path: &Path, budget: u64) -> Result<Self> {
        let index = if path.is_dir() {
            StIndex::open(path)
        } else {
            StIndex::open_file(path)
        }
        .map_err(candle_core::Error::wrap)?;
        let mut index = index;
        let prefix = super::transformer::PREFIX;
        if index.get("patchify_proj.weight").is_some() && index.get(&format!("{prefix}patchify_proj.weight")).is_none() {
            index.prefix_names(prefix);
        }
        Ok(Self {
            index,
            host: HashMap::new(),
            scales: HashMap::new(),
            host_bytes: 0,
            disk_bytes: 0,
            budget,
        })
    }
    pub fn tensor(&mut self, key: &str, dev: &Device, cache: bool) -> Result<Tensor> {
        let info = self
            .index
            .info(key)
            .map_err(candle_core::Error::wrap)?
            .clone();
        let scale = self.scale(key)?;
        if let Some(bytes) = self.host.get(key) {
            return decode(key, info.dtype, &info.shape, bytes, scale.as_deref(), dev);
        }
        let bytes = self.index.read(key).map_err(candle_core::Error::wrap)?;
        self.disk_bytes += bytes.len() as u64;
        let t = decode(key, info.dtype, &info.shape, &bytes, scale.as_deref(), dev)?;
        if cache && self.host_bytes + bytes.len() as u64 <= self.budget {
            self.host_bytes += bytes.len() as u64;
            self.host.insert(key.to_owned(), bytes);
        }
        Ok(t)
    }
    /// Fetch only prompt vocabulary rows, avoiding a multi-gigabyte embedding upload.
    pub fn rows(&mut self, key: &str, ids: &[u32], dev: &Device) -> Result<Tensor> {
        use std::io::{Read, Seek, SeekFrom};
        let info = self.index.info(key).map_err(candle_core::Error::wrap)?;
        if info.shape.len() != 2 || ids.iter().any(|&i| i as usize >= info.shape[0]) {
            candle_core::bail!("invalid embedding row selection for {key}");
        }
        let stored = info.dtype;
        if !matches!(stored, Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F8E4M3 | Dtype::I8) {
            candle_core::bail!("unsupported embedding dtype {stored:?}");
        }
        let width = info.shape[1];
        let row_bytes = width
            .checked_mul(stored.size())
            .ok_or_else(|| candle_core::Error::Msg("embedding row overflow".into()))?;
        let size = ids
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| candle_core::Error::Msg("embedding selection overflow".into()))?;
        let mut bytes = vec![0u8; size];
        let mut file = std::fs::File::open(self.index.shard_path(info.shard))?;
        for (i, &id) in ids.iter().enumerate() {
            let offset = (id as u64)
                .checked_mul(row_bytes as u64)
                .ok_or_else(|| candle_core::Error::Msg("embedding offset overflow".into()))?;
            if offset + row_bytes as u64 > info.nbytes {
                candle_core::bail!("embedding row outside tensor");
            }
            file.seek(SeekFrom::Start(info.start.checked_add(offset).ok_or_else(
                || candle_core::Error::Msg("embedding file offset overflow".into()),
            )?))?;
            file.read_exact(&mut bytes[i * row_bytes..(i + 1) * row_bytes])?;
        }
        self.disk_bytes += size as u64;
        let scale = match self.scale(key)? {
            // A per-row scale follows the selected rows.
            Some(s) if s.len() > 1 => Some(ids.iter().map(|&i| s[i as usize]).collect()),
            other => other,
        };
        decode(key, stored, &[ids.len(), width], &bytes, scale.as_deref(), dev)
    }

    /// The `weight_scale` stored beside a quantized weight, if any.
    fn scale(&mut self, key: &str) -> Result<Option<Vec<f32>>> {
        let Some(base) = key.strip_suffix(".weight") else { return Ok(None) };
        if let Some(s) = self.scales.get(key) {
            return Ok(s.clone());
        }
        let name = format!("{base}.weight_scale");
        let found = match self.index.get(&name).cloned() {
            None => None,
            Some(info) => {
                let bytes = self.index.read(&name).map_err(candle_core::Error::wrap)?;
                let values: Vec<f32> = match info.dtype {
                    Dtype::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                    Dtype::BF16 => bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
                    other => candle_core::bail!("LTX tensor {name}: unsupported scale dtype {other:?}"),
                };
                Some(values)
            }
        };
        self.scales.insert(key.to_owned(), found.clone());
        Ok(found)
    }
    pub fn group_bytes(&self, prefix: &str, select: impl Fn(&str) -> bool) -> Result<u64> {
        let mut bytes = 0;
        for key in self.index.names() {
            let Some(short) = key.strip_prefix(prefix).filter(|s| select(s) && !is_scale(s)) else {
                continue;
            };
            let info = self.index.info(key).map_err(candle_core::Error::wrap)?;
            // All selected floating-point weights become BF16 on the device.
            bytes += info.shape.iter().product::<usize>() as u64 * 2;
            if info.shape.len() == 2 {
                if let Some(base) = short.strip_suffix(".weight") {
                    let bias = format!("{base}.bias");
                    if select(&bias) && self.index.get(&format!("{prefix}{bias}")).is_some() {
                        bytes += info.shape[0] as u64 * 8 * 2;
                    }
                }
            }
        }
        Ok(bytes)
    }
    pub fn group(
        &mut self,
        prefix: &str,
        dev: &Device,
        cache: bool,
        select: impl Fn(&str) -> bool,
    ) -> Result<Group> {
        let mut names: Vec<_> = self
            .index
            .names()
            .filter_map(|k| {
                k.strip_prefix(prefix)
                    .filter(|s| select(s) && !is_scale(s))
                    .map(|s| (k.to_owned(), s.to_owned()))
            })
            .collect();
        names.sort();
        let mut tensors = HashMap::new();
        let mut bytes = 0;
        for (key, short) in names {
            let t = self.tensor(&key, dev, cache)?;
            bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            tensors.insert(short, t);
        }
        // Include the bias in the GEMM accumulator before its BF16 rounding.
        // Eight extra channels keep the contraction dimension tensor-core aligned.
        let mut biased = HashSet::new();
        let biases: Vec<_> = tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".bias").map(str::to_owned))
            .collect();
        for key in biases {
            let name = format!("{key}.weight");
            let Some(w) = tensors.get(&name) else {
                continue;
            };
            if w.rank() != 2 {
                continue;
            }
            let out = w.dim(0)?;
            let b = tensors.get(&format!("{key}.bias")).unwrap().unsqueeze(1)?;
            let zero = Tensor::zeros((out, 7), w.dtype(), dev)?;
            let packed = Tensor::cat(&[w, &b, &zero], 1)?;
            bytes += (out * 8 * w.dtype().size_in_bytes()) as u64;
            tensors.insert(name, packed);
            biased.insert(key);
        }
        Ok(Group {
            tensors,
            bytes,
            biased,
        })
    }
}
/// Quantization side tensors, applied with their weight rather than loaded.
fn is_scale(name: &str) -> bool {
    name.ends_with(".weight_scale") || name.ends_with(".input_scale")
}

/// Every fp8 E4M3 (fn) byte's value: 1 sign, 4 exponent (bias 7) and 3
/// mantissa bits; no infinities, and 0x7F/0xFF are NaN.
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0f32; 256];
        for (b, v) in t.iter_mut().enumerate() {
            let (sign, exp, man) = (if b & 0x80 != 0 { -1.0 } else { 1.0 }, (b >> 3) & 0xF, (b & 7) as f32);
            *v = sign * match (exp, man as u8) {
                (0xF, 7) => f32::NAN,
                (0, _) => man / 8.0 * 2f32.powi(-6),
                _ => (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7),
            };
        }
        t
    })
}

/// Stored bytes as a BF16 tensor on `dev`. fp8 converts on the device; int8
/// (which candle has no type for) goes up as bytes and is re-signed there.
/// `scale` is one value for the whole tensor or one per row.
fn decode(key: &str, dtype: Dtype, shape: &[usize], bytes: &[u8], scale: Option<&[f32]>, dev: &Device) -> Result<Tensor> {
    let t = match dtype {
        Dtype::BF16 => Tensor::from_raw_buffer(bytes, DType::BF16, shape, dev)?,
        Dtype::F16 => Tensor::from_raw_buffer(bytes, DType::F16, shape, dev)?,
        Dtype::F32 => Tensor::from_raw_buffer(bytes, DType::F32, shape, dev)?,
        Dtype::U8 if scale.is_none() => Tensor::from_raw_buffer(bytes, DType::U8, shape, dev)?,
        // candle's own fp8 casts are not built for every GPU (sm_120 lacks
        // them), so each byte is looked up in the E4M3 table instead: exact,
        // and the same everywhere.
        Dtype::F8E4M3 => {
            let lut = Tensor::from_vec(e4m3_table().to_vec(), 256, dev)?;
            let codes = Tensor::from_raw_buffer(bytes, DType::U8, &[bytes.len()], dev)?;
            lut.index_select(&codes, 0)?.reshape(shape)?
        }
        Dtype::I8 => {
            let u = Tensor::from_raw_buffer(bytes, DType::U8, shape, dev)?.to_dtype(DType::F32)?;
            let negative = u.ge(128f64)?.to_dtype(DType::F32)?;
            (u - (negative * 256.)?)?
        }
        other => candle_core::bail!("LTX tensor {key}: unsupported dtype {other:?}"),
    };
    let t = match scale {
        None => t,
        Some([s]) => t.to_dtype(DType::F32)?.affine(*s as f64, 0.)?,
        Some(rows) if shape.first() == Some(&rows.len()) => {
            let mut dims = vec![1; shape.len()];
            dims[0] = rows.len();
            let s = Tensor::from_slice(rows, dims.as_slice(), dev)?;
            t.to_dtype(DType::F32)?.broadcast_mul(&s)?
        }
        Some(s) => candle_core::bail!("LTX tensor {key}: {} scales for shape {shape:?}", s.len()),
    };
    t.to_dtype(DType::BF16)
}

pub struct Group {
    pub tensors: HashMap<String, Tensor>,
    pub bytes: u64,
    biased: HashSet<String>,
}
impl Group {
    pub fn get(&self, k: &str) -> Result<&Tensor> {
        self.tensors
            .get(k)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing LTX tensor {k}")))
    }
    pub fn linear(&self, key: &str, x: &Tensor) -> Result<Tensor> {
        let w = self.get(&format!("{key}.weight"))?;
        let input = x.dim(candle_core::D::Minus1)?;
        let rows = x.elem_count() / input;
        let flat = x.reshape((rows, input))?;
        let flat = if self.biased.contains(key) {
            Tensor::cat(
                &[
                    &flat,
                    &Tensor::ones((rows, 1), x.dtype(), x.device())?,
                    &Tensor::zeros((rows, 7), x.dtype(), x.device())?,
                ],
                1,
            )?
        } else {
            flat
        };
        let y = flat.matmul(&w.t()?)?;
        let mut shape = x.dims().to_vec();
        let last = shape
            .last_mut()
            .ok_or_else(|| candle_core::Error::Msg("linear requires a channel axis".into()))?;
        *last = w.dim(0)?;
        y.reshape(shape)
    }
}
