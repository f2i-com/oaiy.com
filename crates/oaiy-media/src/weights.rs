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
        ]
        .into_iter()
        .chain(llama_name(name))
        {
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

    /// A GGUF's fused gate or up half as a quantized tensor of its own (None where `name` is not one).
    fn gguf_half_qtensor(&mut self, name: &str, dev: &Device) -> Result<Option<candle_core::quantized::QTensor>> {
        let Some((key, half)) = self.gguf_half(name) else { return Ok(None) };
        let (t, bytes, dims) = self.gguf_bytes(&key, Some(half))?;
        Ok(Some(ggml_file::qtensor_from_ggml(t, &bytes, dims, dev)?))
    }

    /// Where a GGUF keeps a block's MLP gate and up projections as one `img_mlp.gate_up` (gate rows first, as
    /// AtomicChat's Qwen-Image-2.1-Turbo GGUFs do): the fused tensor's key and the half `name` is (0 gate, 1 up).
    fn gguf_half(&self, name: &str) -> Option<(String, usize)> {
        if !matches!(self, Self::Gguf { .. }) || self.has(name) {
            return None;
        }
        [("img_mlp.gate_layer.weight", 0), ("img_mlp.proj.weight", 1)]
            .into_iter()
            .find_map(|(suffix, half)| Some((self.resolve(&format!("{}img_mlp.gate_up.weight", name.strip_suffix(suffix)?)).ok()?, half)))
    }

    /// A GGUF tensor's bytes as stored (its blocks): rows `half` of two, or all of it.
    fn gguf_bytes(&mut self, key: &str, half: Option<usize>) -> Result<(GgmlDType, Vec<u8>, Vec<usize>)> {
        use std::io::{Read, Seek, SeekFrom};
        let Self::Gguf { content, file } = self else { candle_core::bail!("{key}: not a GGUF tensor") };
        let info = content.tensor_infos.get(key).ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {key}")))?;
        let t = info.ggml_dtype;
        let mut dims = info.shape.dims().to_vec();
        let size = info.shape.elem_count() / t.block_size() * t.type_size();
        let (start, len) = match half {
            None => (0, size),
            Some(h) => {
                // (each row whole blocks: a half's rows are a half's bytes)
                if dims.len() != 2 || dims[0] % 2 != 0 || dims[1] % t.block_size() != 0 {
                    candle_core::bail!("{key}: a fused gate and up of shape {dims:?} in {t:?} blocks");
                }
                dims[0] /= 2;
                (h * size / 2, size / 2)
            }
        };
        let mut bytes = vec![0u8; len];
        file.seek(SeekFrom::Start(content.tensor_data_offset + info.offset + start as u64))?;
        file.read_exact(&mut bytes)?;
        Ok((t, bytes, dims))
    }

    /// `name`'s GGUF block type (a GGUF file's tensor; None for safetensors), without reading it.
    pub fn ggml_dtype(&self, name: &str) -> Option<GgmlDType> {
        let key = self.resolve(name).ok().or_else(|| self.gguf_half(name).map(|(k, _)| k))?;
        match self {
            Self::Gguf { content, .. } => content.tensor_infos.get(&key).map(|i| i.ggml_dtype),
            Self::Safe(_) => None,
        }
    }

    /// `name`'s bytes as stored where [`Raw`] holds them; None where only [`Self::tensor`] reads it (ComfyUI's
    /// quantized weights, F16, F32, a fused gate and up it splits).
    pub fn raw(&mut self, name: &str) -> Result<Option<Raw>> {
        if let Some((key, half)) = self.gguf_half(name) {
            let (t, bytes, _) = self.gguf_bytes(&key, Some(half))?;
            return Ok(Some(Raw::Ggml(t, bytes)));
        }
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

    /// `name`'s bytes and shape where a safetensors file keeps it as F16 (little-endian halves, read on several
    /// cores): what a device that holds f16 takes as they are. None for anything else (ComfyUI's quantized weights
    /// too).
    pub fn raw_f16(&self, name: &str) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let Ok(key) = self.resolve(name) else { return Ok(None) };
        let Self::Safe(s) = self else { return Ok(None) };
        let info = s.info(&key).map_err(candle_core::Error::wrap)?;
        let quantized = key.strip_suffix(".weight").is_some_and(|p| s.get(&format!("{p}.comfy_quant")).is_some());
        if info.dtype != Dtype::F16 || quantized {
            return Ok(None);
        }
        let shape = info.shape.clone();
        Ok(Some((s.read_par(&key).map_err(candle_core::Error::wrap)?, shape)))
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
        if let Some((key, _)) = self.gguf_half(name) {
            let Self::Gguf { content, .. } = self else { unreachable!() };
            let mut dims = content.tensor_infos[&key].shape.dims().to_vec();
            dims[0] /= 2;
            return Ok(dims);
        }
        let key = self.resolve(name)?;
        match self {
            // (ComfyUI's W4A8: its codes two to a byte, the matrix [rows, 2 x the bytes'], as Self::tensor reads it)
            Self::Safe(s) => Ok(crate::comfy_quant::logical_shape(s, &key).unwrap_or(s.info(&key).map_err(candle_core::Error::wrap)?.shape.clone())),
            Self::Gguf { content, .. } => Ok(content.tensor_infos.get(&key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {key}")))?.shape.dims().to_vec()),
        }
    }

    pub fn tensor(&mut self, name: &str, dev: &Device, dtype: DType) -> Result<Tensor> {
        if let Some(q) = self.gguf_half_qtensor(name, dev)? {
            return q.dequantize(dev)?.to_dtype(dtype);
        }
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
        let key = if matches!(self, Self::Gguf { .. }) && self.gguf_half(&key).is_none() { self.resolve(&key)? } else { key };
        // ([`gguf_dense`], or OAIY_GGUF_DENSE: a GGUF's matrices dequantized whole, their products exact)
        static ASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dense = GGUF_DENSE.load(std::sync::atomic::Ordering::Relaxed) || *ASKED.get_or_init(|| std::env::var_os("OAIY_GGUF_DENSE").is_some());
        let weight = match self {
            Self::Gguf { .. } if !dense && self.gguf_half(&key).is_some() => Weight::Quant(QMatMul::from_qtensor(self.gguf_half_qtensor(&key, dev)?.expect("a fused half"))?),
            Self::Gguf { content, file } if !dense => {
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

static GGUF_DENSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether [`Weights::linear`] dequantizes a GGUF's matrices whole from here on (four bytes a weight, their products
/// exact), where it otherwise keeps their blocks for Candle's quantized matmul. That matmul rounds each row of
/// activations to 8 bits a block before a K-quant's product, as ggml's does (some 1% of a product's worst element,
/// FLUX.2 Klein's velocity then 0.9987 by cosine from the exact one): a port that multiplies by the dequantized
/// weights is held to the exact products, so its tests ask for them.
pub fn gguf_dense(on: bool) {
    GGUF_DENSE.store(on, std::sync::atomic::Ordering::Relaxed);
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

/// A language model's tensor as llama.cpp's GGUF names it (a text encoder in a GGUF, e.g. Qwen3-VL 8B's as
/// stable-diffusion.cpp's `--llm` takes it), for its Hugging Face name: `model.layers.3.self_attn.q_proj.weight` is
/// `blk.3.attn_q.weight`.
fn llama_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("model.language_model.").or_else(|| name.strip_prefix("model."))?;
    match rest {
        "embed_tokens.weight" => return Some("token_embd.weight".into()),
        "norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }
    let rest = rest.strip_prefix("layers.")?;
    let (layer, part) = rest.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let (module, kind) = part.rsplit_once('.')?;
    let llama = match module {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "ffn_norm",
        _ => return None,
    };
    Some(format!("blk.{layer}.{llama}.{kind}"))
}

#[cfg(test)]
mod llama_names {
    use super::llama_name;

    #[test]
    fn a_language_models_hugging_face_names_are_llama_cpps_in_a_gguf() {
        for (hf, llama) in [
            ("model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight"),
            ("model.language_model.layers.35.mlp.down_proj.weight", "blk.35.ffn_down.weight"),
            ("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight"),
            ("model.layers.7.self_attn.k_norm.weight", "blk.7.attn_k_norm.weight"),
            ("model.language_model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
        ] {
            assert_eq!(llama_name(hf).as_deref(), Some(llama), "{hf}");
        }
        assert_eq!(llama_name("transformer_blocks.0.attn.to_q.weight"), None);
        assert_eq!(llama_name("model.layers.x.self_attn.q_proj.weight"), None);
    }
}
