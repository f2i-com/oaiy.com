use candle_core::{
    quantized::{ggml_file, gguf_file, GgmlDType, QMatMul},
    DType, Device, Module, Result, Tensor,
};
use dsv41::safetensors::{Dtype, StIndex};
use std::{fs::File, path::Path};

/// A tensor's bytes as stored, where a device converts them itself: BF16 safetensors, or GGUF's blocks (of their type).
pub enum Raw {
    Bf16(Vec<u8>),
    Ggml(GgmlDType, Vec<u8>),
}

pub enum Weights {
    Safe(StIndex),
    Gguf {
        content: gguf_file::Content,
        file: File,
    },
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            let mut file = File::open(path)?;
            let content = gguf_file::Content::read(&mut file)?;
            Ok(Self::Gguf { content, file })
        } else {
            let index = if path.is_dir() {
                StIndex::open(path)
            } else {
                StIndex::open_file(path)
            };
            Ok(Self::Safe(index.map_err(candle_core::Error::wrap)?))
        }
    }

    pub fn names(&self) -> Vec<String> {
        match self {
            Self::Safe(s) => s.names().map(str::to_owned).collect(),
            Self::Gguf { content, .. } => content.tensor_infos.keys().cloned().collect(),
        }
    }

    fn resolve(&self, name: &str) -> Result<String> {
        for candidate in [
            name.to_owned(),
            format!("model.diffusion_model.{name}"),
            format!("diffusion_model.{name}"),
        ] {
            let exists = match self {
                Self::Safe(s) => s.get(&candidate).is_some(),
                Self::Gguf { content, .. } => content.tensor_infos.contains_key(&candidate),
            };
            if exists {
                return Ok(candidate);
            }
        }
        candle_core::bail!("missing tensor {name}")
    }

    pub fn has(&self, name: &str) -> bool {
        self.resolve(name).is_ok()
    }

    /// `name`'s GGUF block type (a GGUF file's tensor; None for safetensors), without reading it.
    pub fn ggml_dtype(&self, name: &str) -> Option<GgmlDType> {
        let key = self.resolve(name).ok()?;
        match self {
            Self::Gguf { content, .. } => content.tensor_infos.get(&key).map(|i| i.ggml_dtype),
            Self::Safe(_) => None,
        }
    }

    /// `name`'s bytes as stored where [`Raw`] holds them; None where only [`Self::tensor`] reads it (ComfyUI's
    /// quantized weights, F16, F32, a fused gate and up it splits).
    pub fn raw(&mut self, name: &str) -> Result<Option<Raw>> {
        let Ok(key) = self.resolve(name) else { return Ok(None) };
        match self {
            Self::Safe(s) => {
                let info = s.info(&key).map_err(candle_core::Error::wrap)?;
                let quantized = key.strip_suffix(".weight").is_some_and(|p| s.get(&format!("{p}.comfy_quant")).is_some());
                if info.dtype != Dtype::BF16 || quantized {
                    return Ok(None);
                }
                Ok(Some(Raw::Bf16(s.read_par(&key).map_err(candle_core::Error::wrap)?)))
            }
            Self::Gguf { content, file } => {
                use std::io::{Read, Seek, SeekFrom};
                let info = content.tensor_infos.get(&key).ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {key}")))?;
                let t = info.ggml_dtype;
                let size = info.shape.elem_count() / t.block_size() * t.type_size();
                let mut bytes = vec![0u8; size];
                file.seek(SeekFrom::Start(content.tensor_data_offset + info.offset))?;
                file.read_exact(&mut bytes)?;
                Ok(Some(Raw::Ggml(t, bytes)))
            }
        }
    }

    /// `name` as ComfyUI's W4A8 keeps it ([`crate::comfy_quant::W4a8`]: a GPU's to decode), None for anything else.
    pub fn w4a8(&self, name: &str) -> Result<Option<crate::comfy_quant::W4a8>> {
        let Ok(key) = self.resolve(name) else { return Ok(None) };
        match self {
            Self::Safe(s) => crate::comfy_quant::w4a8(s, &key),
            Self::Gguf { .. } => Ok(None),
        }
    }

    /// Inspect dimensions without materializing tensor payloads.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let key = self.resolve(name)?;
        match self {
            // (ComfyUI's W4A8: its codes two to a byte, the matrix [rows, 2 x the bytes'], as Self::tensor reads it)
            Self::Safe(s) => Ok(crate::comfy_quant::logical_shape(s, &key).unwrap_or(s.info(&key).map_err(candle_core::Error::wrap)?.shape.clone())),
            Self::Gguf { content, .. } => Ok(content.tensor_infos.get(&key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {key}")))?.shape.dims().to_vec()),
        }
    }

    pub fn tensor(&mut self, name: &str, dev: &Device, dtype: DType) -> Result<Tensor> {
        // Comfy's fused Qwen 2.1 MLP stores gate rows followed by up rows.
        // Read only the requested half, preserving the original logical LoRA keys.
        if !self.has(name) {
            for (suffix, half) in [("img_mlp.gate_layer.weight", 0), ("img_mlp.proj.weight", 1)] {
                if let Some(prefix) = name.strip_suffix(suffix) {
                    let key = self.resolve(&format!("{prefix}img_mlp.gate_up.weight"))?;
                    if let Self::Safe(s) = self {
                        use std::io::{Read, Seek, SeekFrom};
                        let info = s.info(&key).map_err(candle_core::Error::wrap)?;
                        if info.shape.len() != 2 || info.shape[0] % 2 != 0 {
                            candle_core::bail!("invalid fused gate/up shape for {key}");
                        }
                        let kind = match info.dtype {
                            Dtype::F16 => DType::F16, Dtype::BF16 => DType::BF16,
                            Dtype::F32 => DType::F32,
                            _ => candle_core::bail!("unsupported fused gate/up dtype for {key}"),
                        };
                        let len = usize::try_from(info.nbytes / 2).map_err(candle_core::Error::wrap)?;
                        let mut bytes = vec![0; len];
                        let mut file = File::open(s.shard_path(info.shard))?;
                        file.seek(SeekFrom::Start(info.start + half * info.nbytes / 2))?;
                        file.read_exact(&mut bytes)?;
                        return Tensor::from_raw_buffer(&bytes, kind, &[info.shape[0]/2, info.shape[1]], dev)?.to_dtype(dtype);
                    }
                }
            }
        }
        let name = self.resolve(name)?;
        let t = match self {
            Self::Safe(s) => {
                if let Some(t) = crate::comfy_quant::load(s, &name, dev, dtype)? { return Ok(t); }
                let info = s.info(&name).map_err(candle_core::Error::wrap)?;
                let kind = match info.dtype {
                    Dtype::F32 => DType::F32,
                    Dtype::F16 => DType::F16,
                    Dtype::BF16 => DType::BF16,
                    other => {
                        candle_core::bail!("unsupported safetensors dtype {other:?} for {name}")
                    }
                };
                let bytes = s.read(&name).map_err(candle_core::Error::wrap)?;
                Tensor::from_raw_buffer(&bytes, kind, &info.shape, dev)?
            }
            Self::Gguf { content, file } => content.tensor(file, &name, dev)?.dequantize(dev)?,
        };
        t.to_dtype(dtype)
    }

    pub fn linear(
        &mut self,
        name: &str,
        dev: &Device,
        dtype: DType,
        lora: &mut crate::lora::Loras,
    ) -> Result<Linear> {
        let key = format!("{name}.weight");
        let key = if matches!(self, Self::Gguf { .. }) { self.resolve(&key)? } else { key };
        let weight = match self {
            Self::Gguf { content, file } => {
                Weight::Quant(QMatMul::from_qtensor(content.tensor(file, &key, dev)?)?)
            }
            _ => Weight::Dense(self.tensor(&key, dev, dtype)?),
        };
        let bias_key = format!("{name}.bias");
        let bias = if self.has(&bias_key) {
            Some(self.tensor(&bias_key, dev, dtype)?)
        } else {
            None
        };
        // Each adapter's factors, checked against the projection's shape [out, input].
        let adapters = if lora.is_empty() {
            Vec::new()
        } else {
            let (out, input) = match &weight {
                Weight::Dense(t) => t.dims2()?,
                Weight::Quant(QMatMul::QTensor(q)) => q.shape().dims2()?,
                Weight::Quant(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => t.dims2()?,
                Weight::HostQuant { dims, .. } => (dims[0], dims[1]),
            };
            lora.factors(name, out, input, dev, dtype)?
        };
        Ok(Linear {
            weight,
            bias,
            adapters,
        })
    }
}

enum Weight {
    Dense(Tensor),
    Quant(QMatMul),
    /// A quantized matrix parked in RAM as its GGML bytes, so a RAM-tier block
    /// uploads the ~4.5-bit payload rather than a dequantized copy.
    HostQuant {
        dtype: GgmlDType,
        bytes: Vec<u8>,
        dims: Vec<usize>,
    },
}

fn tensor_bytes(t: &Tensor) -> u64 {
    (t.elem_count() * t.dtype().size_in_bytes()) as u64
}
pub struct Linear {
    weight: Weight,
    bias: Option<Tensor>,
    /// LoRA factors (down, up with its scale folded in), one pair per adapter.
    adapters: Vec<(Tensor, Tensor)>,
}
impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        let input = *shape.last().filter(|&&n|n>0).ok_or_else(||candle_core::Error::Msg("linear input requires a nonempty channel axis".into()))?;
        let n = x.elem_count() / input;
        let flat = x.reshape((n, input))?;
        let mut y = match &self.weight {
            Weight::Dense(w) => flat.matmul(&w.t()?)?,
            Weight::Quant(q) => q
                .forward(&flat.to_dtype(DType::F32)?)?
                .to_dtype(x.dtype())?,
            Weight::HostQuant { .. } => {
                candle_core::bail!("a RAM-resident weight was used without uploading it")
            }
        };
        for (a, b) in &self.adapters {
            y = (y + flat.matmul(&a.t()?)?.matmul(&b.t()?)?)?;
        }
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        let mut out_shape = shape.to_vec();
        *out_shape
            .last_mut()
            .ok_or_else(|| candle_core::Error::Msg("linear input has no dimensions".into()))? =
            y.dim(1)?;
        y.reshape(out_shape)
    }
}

impl Linear {
    /// Device bytes this projection holds (weight, bias and LoRA factors).
    pub fn bytes(&self) -> u64 {
        let weight = match &self.weight {
            Weight::Dense(t) => tensor_bytes(t),
            Weight::Quant(QMatMul::QTensor(q)) => q.storage_size_in_bytes() as u64,
            Weight::Quant(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => tensor_bytes(t),
            Weight::HostQuant { bytes, .. } => bytes.len() as u64,
        };
        weight
            + self.bias.as_ref().map_or(0, tensor_bytes)
            + self.adapters.iter().map(|(a, b)| tensor_bytes(a) + tensor_bytes(b)).sum::<u64>()
    }

    /// The same projection on `dev`. Quantized GGUF weights move as their raw
    /// blocks and are rebuilt there, never dequantized on the way.
    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        let weight = match &self.weight {
            Weight::Dense(t) => Weight::Dense(t.to_device(dev)?),
            Weight::Quant(QMatMul::QTensor(q)) if dev.is_cpu() => Weight::HostQuant {
                dtype: q.dtype(),
                bytes: q.data()?.into_owned(),
                dims: q.shape().dims().to_vec(),
            },
            Weight::Quant(QMatMul::QTensor(q)) => Weight::Quant(QMatMul::from_qtensor(
                ggml_file::qtensor_from_ggml(q.dtype(), &q.data()?, q.shape().dims().to_vec(), dev)?,
            )?),
            Weight::Quant(QMatMul::Tensor(t)) => Weight::Quant(QMatMul::Tensor(t.to_device(dev)?)),
            Weight::Quant(QMatMul::TensorF16(t)) => Weight::Quant(QMatMul::TensorF16(t.to_device(dev)?)),
            Weight::HostQuant { dtype, bytes, dims } if dev.is_cpu() => {
                Weight::HostQuant { dtype: *dtype, bytes: bytes.clone(), dims: dims.clone() }
            }
            Weight::HostQuant { dtype, bytes, dims } => Weight::Quant(QMatMul::from_qtensor(
                ggml_file::qtensor_from_ggml(*dtype, bytes, dims.clone(), dev)?,
            )?),
        };
        Ok(Self {
            weight,
            bias: self.bias.as_ref().map(|b| b.to_device(dev)).transpose()?,
            adapters: self
                .adapters
                .iter()
                .map(|(a, b)| Ok::<_, candle_core::Error>((a.to_device(dev)?, b.to_device(dev)?)))
                .collect::<Result<_>>()?,
        })
    }
}

/// Device bytes of a plain tensor (norm scales and the like), for block sizes.
pub fn bytes_of(t: &Tensor) -> u64 {
    tensor_bytes(t)
}
