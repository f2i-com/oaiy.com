//! Bounded weight residency. SSD reads use the published safetensors directly.
use candle_core::{DType, Device, Result, Tensor};
use dsv41::safetensors::{Dtype, StIndex};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

pub struct Store {
    pub index: StIndex,
    host: HashMap<String, Vec<u8>>,
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
        Ok(Self {
            index,
            host: HashMap::new(),
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
        let dtype = match info.dtype {
            Dtype::BF16 => DType::BF16,
            Dtype::F16 => DType::F16,
            Dtype::F32 => DType::F32,
            Dtype::U8 => DType::U8,
            _ => candle_core::bail!(
                "LTX tensor {key}: unsupported dtype {:?}; use BF16 weights",
                info.dtype
            ),
        };
        if let Some(bytes) = self.host.get(key) {
            return Tensor::from_raw_buffer(bytes, dtype, &info.shape, dev)?.to_dtype(DType::BF16);
        }
        let bytes = self.index.read(key).map_err(candle_core::Error::wrap)?;
        self.disk_bytes += bytes.len() as u64;
        let t = Tensor::from_raw_buffer(&bytes, dtype, &info.shape, dev)?.to_dtype(DType::BF16)?;
        if cache && self.host_bytes + bytes.len() as u64 <= self.budget {
            self.host_bytes += bytes.len() as u64;
            self.host.insert(key.to_owned(), bytes);
        }
        Ok(t)
    }
    pub fn group_bytes(&self, prefix: &str, select: impl Fn(&str) -> bool) -> Result<u64> {
        let mut bytes = 0;
        for key in self.index.names() {
            let Some(short) = key.strip_prefix(prefix).filter(|s| select(s)) else {
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
                    .filter(|s| select(s))
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
