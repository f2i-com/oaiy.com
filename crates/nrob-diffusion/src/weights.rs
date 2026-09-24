use candle_core::{
    quantized::{gguf_file, QMatMul},
    DType, Device, Module, Result, Tensor,
};
use dsv41::safetensors::{Dtype, StIndex};
use std::{fs::File, path::Path};

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
        lora: &mut Option<Weights>,
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
        let adapter = if let Some(lora) = lora {
            let mut adapter = None;
            for prefix in [format!("transformer.{name}"), name.to_owned()] {
                let key = format!("{prefix}.lora_A.weight");
                if lora.has(&key) {
                    let a = lora.tensor(&key, dev, dtype)?;
                    let b = lora.tensor(&format!("{prefix}.lora_B.weight"), dev, dtype)?;
                    // Viggle v0.2.1: alpha == rank, so runtime scale is exactly one.
                    adapter = Some((a, b));
                    break;
                }
            }
            adapter
        } else {
            None
        };
        Ok(Linear {
            weight,
            bias,
            adapter,
        })
    }
}

enum Weight {
    Dense(Tensor),
    Quant(QMatMul),
}
pub struct Linear {
    weight: Weight,
    bias: Option<Tensor>,
    adapter: Option<(Tensor, Tensor)>,
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
        };
        if let Some((a, b)) = &self.adapter {
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
