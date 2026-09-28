//! A Parakeet checkpoint's tensors, by NeMo's parameter names, from either
//! form they are published in: a `.nemo` (NeMo's names) or a Hugging Face
//! `model.safetensors` (transformers' `ParakeetForTDT` names, mapped back).

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use dsv41::safetensors::{Dtype, StIndex};

use super::nemo::{NemoArchive, RawTensor, StorageType};
use super::{bad, Result};

pub enum Weights {
    Nemo(NemoArchive),
    Safetensors(StIndex),
}

/// The transformers name of a NeMo parameter (`convert_nemo_to_hf.py`'s
/// renaming, inverted).
pub fn hf_name(nemo: &str) -> String {
    nemo.replace("encoder.pre_encode.conv.", "encoder.subsampling.layers.")
        .replace("encoder.pre_encode.out.", "encoder.subsampling.linear.")
        .replace(".self_attn.linear_q.", ".self_attn.q_proj.")
        .replace(".self_attn.linear_k.", ".self_attn.k_proj.")
        .replace(".self_attn.linear_v.", ".self_attn.v_proj.")
        .replace(".self_attn.linear_out.", ".self_attn.o_proj.")
        .replace(".self_attn.linear_pos.", ".self_attn.relative_k_proj.")
        .replace(".self_attn.pos_bias_u", ".self_attn.bias_u")
        .replace(".self_attn.pos_bias_v", ".self_attn.bias_v")
        .replace(".conv.batch_norm.", ".conv.norm.")
        .replace("decoder.prediction.embed.", "decoder.embedding.")
        .replace("decoder.prediction.dec_rnn.lstm.", "decoder.lstm.")
        .replace("joint.pred.", "decoder.decoder_projector.")
        .replace("joint.enc.", "encoder_projector.")
        .replace("joint.joint_net.2.", "joint.head.")
}

fn decode_f32(dtype: StorageType, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        StorageType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        StorageType::F16 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        StorageType::BF16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => return Err(bad(format!("{other:?} is not a float tensor"))),
    })
}

impl Weights {
    /// A `.nemo` file, or a `model.safetensors` (or a folder holding one).
    pub fn open(path: &Path) -> Result<Self> {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("nemo")) {
            return Ok(Self::Nemo(NemoArchive::open(path)?));
        }
        let index = if path.is_dir() { StIndex::open(path) } else { StIndex::open_file(path) };
        Ok(Self::Safetensors(index.map_err(|e| bad(format!("{}: {e}", path.display())))?))
    }

    pub fn has(&self, name: &str) -> bool {
        match self {
            Self::Nemo(n) => n.has(name),
            Self::Safetensors(s) => s.get(&hf_name(name)).is_some(),
        }
    }

    /// A tensor's shape, without reading it.
    pub fn shape(&mut self, name: &str) -> Result<Vec<usize>> {
        match self {
            Self::Nemo(n) => n.shape(name).ok_or_else(|| bad(format!("tensor {name} not in {}", n.path().display()))),
            Self::Safetensors(s) => Ok(s.info(&hf_name(name)).map_err(|e| bad(e.to_string()))?.shape.clone()),
        }
    }

    /// A float tensor, decoded to f32 on the CPU, with its shape.
    pub fn f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Nemo(n) => {
                let RawTensor { dtype, shape, bytes } = n.tensor(name)?;
                Ok((decode_f32(dtype, &bytes).map_err(|e| bad(format!("{name}: {e}")))?, shape))
            }
            Self::Safetensors(s) => {
                let hf = hf_name(name);
                let info = s.info(&hf).map_err(|e| bad(e.to_string()))?;
                let shape = info.shape.clone();
                if !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                    return Err(bad(format!("{hf}: {:?} is not a float tensor", info.dtype)));
                }
                Ok((s.read_f32(&hf).map_err(|e| bad(e.to_string()))?, shape))
            }
        }
    }

    /// A float tensor on the CPU as f32, checked against `shape`.
    pub fn f32_shaped(&mut self, name: &str, shape: &[usize]) -> Result<(Vec<f32>, Vec<usize>)> {
        let (v, s) = self.f32(name)?;
        if s != shape {
            return Err(bad(format!("{name}: shape {s:?}, expected {shape:?}")));
        }
        Ok((v, s))
    }

    /// Like [`f32_shaped`](Self::f32_shaped), for a parameter that may be absent.
    pub fn optional_f32(&mut self, name: &str, shape: &[usize]) -> Result<Option<Vec<f32>>> {
        if self.has(name) {
            Ok(Some(self.f32_shaped(name, shape)?.0))
        } else {
            Ok(None)
        }
    }

    /// A float tensor on `dev` as `dtype`, checked against `shape`.
    pub fn tensor(&mut self, name: &str, shape: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
        let (v, s) = self.f32(name)?;
        if s != shape {
            return Err(bad(format!("{name}: shape {s:?}, expected {shape:?}")));
        }
        Tensor::from_vec(v, s, &Device::Cpu)?.to_dtype(dtype)?.to_device(dev)
    }

    /// Like [`tensor`](Self::tensor), for a parameter that may be absent
    /// (biases of models trained without them).
    pub fn optional(&mut self, name: &str, shape: &[usize], dtype: DType, dev: &Device) -> Result<Option<Tensor>> {
        if self.has(name) {
            self.tensor(name, shape, dtype, dev).map(Some)
        } else {
            Ok(None)
        }
    }

    /// A `.nemo`'s member file by name suffix (`model_config.yaml`, `tokenizer.model`).
    pub fn nemo_member(&mut self, suffix: &str) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Nemo(n) => n.member(suffix),
            Self::Safetensors(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemo_names_map_to_transformers_names() {
        for (nemo, hf) in [
            ("encoder.pre_encode.conv.0.weight", "encoder.subsampling.layers.0.weight"),
            ("encoder.pre_encode.out.bias", "encoder.subsampling.linear.bias"),
            ("encoder.layers.3.self_attn.linear_pos.weight", "encoder.layers.3.self_attn.relative_k_proj.weight"),
            ("encoder.layers.3.self_attn.pos_bias_u", "encoder.layers.3.self_attn.bias_u"),
            ("encoder.layers.3.self_attn.linear_out.weight", "encoder.layers.3.self_attn.o_proj.weight"),
            ("encoder.layers.3.conv.batch_norm.running_var", "encoder.layers.3.conv.norm.running_var"),
            ("encoder.layers.3.conv.depthwise_conv.weight", "encoder.layers.3.conv.depthwise_conv.weight"),
            ("encoder.layers.3.norm_feed_forward1.weight", "encoder.layers.3.norm_feed_forward1.weight"),
            ("decoder.prediction.embed.weight", "decoder.embedding.weight"),
            ("decoder.prediction.dec_rnn.lstm.weight_ih_l1", "decoder.lstm.weight_ih_l1"),
            ("joint.pred.weight", "decoder.decoder_projector.weight"),
            ("joint.enc.bias", "encoder_projector.bias"),
            ("joint.joint_net.2.weight", "joint.head.weight"),
        ] {
            assert_eq!(hf_name(nemo), hf, "{nemo}");
        }
    }
}
