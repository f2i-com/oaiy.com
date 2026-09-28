//! A one-off conversion of the Music 3 language model to a smaller file:
//! its 36 layers quantized (GGUF block formats, as llama.cpp uses them), with
//! only the vocabulary the model can read or write kept beside them.
//!
//! | quant  | layers | whole file |
//! |--------|--------|------------|
//! | bf16   | 13.6 GB (the published safetensors, 16.4 GB with both 200k tables) | - |
//! | q8_0   | 7.2 GB | ~8.2 GB |
//! | q6_k   | 5.6 GB | ~6.6 GB |
//! | q5_k   | 4.7 GB | ~5.7 GB |
//! | q4_k   | 3.8 GB | ~4.8 GB |
//!
//! Kept at full precision: the norms (F32), the output rows for the 16384
//! codes and end of audio, and the codes' input rows (BF16). The text
//! embeddings (the prompt's rows are read from the file) become Q8_0.
use super::lm::{self, AUDIO_END, CODE_OFFSET, SEMANTIC_CODES};
use crate::ltx::store::Store;
use candle_core::{
    quantized::{gguf_file, ggml_file, GgmlDType, QTensor},
    DType, Device, Result, Tensor,
};
use oaiy_engine::json::Json;
use std::path::{Path, PathBuf};

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// The layer formats offered.
pub fn parse(name: &str) -> std::result::Result<GgmlDType, String> {
    match name {
        "q8_0" => Ok(GgmlDType::Q8_0),
        "q6_k" => Ok(GgmlDType::Q6K),
        "q5_k" => Ok(GgmlDType::Q5K),
        "q4_k" => Ok(GgmlDType::Q4K),
        _ => Err(format!("quant must be q8_0, q6_k, q5_k or q4_k, not {name}")),
    }
}

/// Where a conversion is written: beside the model's folders.
pub fn default_path(model_dir: &Path, quant: &str) -> PathBuf {
    model_dir.join(format!("language_model-{quant}.gguf"))
}

/// A (rows, cols) matrix quantized on every core (rows are independent).
fn quantize(t: &Tensor, dtype: GgmlDType) -> Result<QTensor> {
    let t = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
    let (rows, cols) = t.dims2()?;
    if matches!(dtype, GgmlDType::F32 | GgmlDType::BF16 | GgmlDType::F16) {
        return QTensor::quantize(&t, dtype);
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).clamp(1, 32);
    let per = rows.div_ceil(threads).max(1);
    let parts = std::thread::scope(|s| -> Result<Vec<Vec<u8>>> {
        let handles: Vec<_> = (0..rows)
            .step_by(per)
            .map(|start| {
                let t = &t;
                // Storage of its own: `quantize` reads a tensor's storage from
                // the start whatever view it is given (and `copy` keeps the view).
                s.spawn(move || -> Result<Vec<u8>> {
                    let n = per.min(rows - start);
                    let chunk = Tensor::from_vec(t.narrow(0, start, n)?.flatten_all()?.to_vec1::<f32>()?, (n, cols), &Device::Cpu)?;
                    Ok(QTensor::quantize(&chunk, dtype)?.data()?.into_owned())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().map_err(|_| msg("a quantizing thread panicked"))?).collect()
    })?;
    ggml_file::qtensor_from_ggml(dtype, &parts.concat(), vec![rows, cols], &Device::Cpu)
}

/// Convert `model_dir/language_model` to `out` with `quant` layers.
/// `progress(done, total)` counts layers.
pub fn convert(model_dir: &Path, quant: &str, out: &Path, mut progress: impl FnMut(usize, usize)) -> Result<Json> {
    let started = std::time::Instant::now();
    let dtype = parse(quant).map_err(msg)?;
    let dir = model_dir.join("language_model");
    let config = lm::config_of(&dir)?;
    let int = |k: &str| config.get(k).and_then(Json::as_i64).map(|v| v as u32).ok_or_else(|| msg(format!("language model config lacks {k}")));
    let layers = int("num_hidden_layers")?;
    let mut store = Store::open(&dir, 0)?;
    let cpu = Device::Cpu;
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let total = layers as usize + 1;
    for i in 0..layers {
        let l = format!("model.layers.{i}");
        let mut get = |n: &str| store.tensor(&format!("{l}.{n}.weight"), &cpu, false);
        let qkv = Tensor::cat(&[get("self_attn.q_proj")?, get("self_attn.k_proj")?, get("self_attn.v_proj")?], 0)?;
        let gate_up = Tensor::cat(&[get("mlp.gate_proj")?, get("mlp.up_proj")?], 0)?;
        tensors.push((format!("layers.{i}.qkv"), quantize(&qkv, dtype)?));
        tensors.push((format!("layers.{i}.o"), quantize(&get("self_attn.o_proj")?, dtype)?));
        tensors.push((format!("layers.{i}.gate_up"), quantize(&gate_up, dtype)?));
        tensors.push((format!("layers.{i}.down"), quantize(&get("mlp.down_proj")?, dtype)?));
        for (name, key) in [("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"), ("input_norm", "input_layernorm"), ("post_norm", "post_attention_layernorm")] {
            tensors.push((format!("layers.{i}.{name}"), QTensor::quantize(&get(key)?.to_dtype(DType::F32)?, GgmlDType::F32)?));
        }
        progress(i as usize + 1, total);
    }
    let code_rows: Vec<u32> = (CODE_OFFSET..CODE_OFFSET + SEMANTIC_CODES as u32).collect();
    let head_rows: Vec<u32> = std::iter::once(AUDIO_END).chain(code_rows.iter().copied()).collect();
    tensors.push(("norm".into(), QTensor::quantize(&store.tensor("model.norm.weight", &cpu, false)?.to_dtype(DType::F32)?, GgmlDType::F32)?));
    tensors.push(("head".into(), quantize(&store.rows("lm_head.weight", &head_rows, &cpu)?, GgmlDType::BF16)?));
    tensors.push(("codes".into(), quantize(&store.rows("model.embed_tokens.weight", &code_rows, &cpu)?, GgmlDType::BF16)?));
    // Every token a prompt can hold lies below the first audio code.
    let text_rows: Vec<u32> = (0..CODE_OFFSET).collect();
    tensors.push(("text".into(), quantize(&store.rows("model.embed_tokens.weight", &text_rows, &cpu)?, GgmlDType::Q8_0)?));
    progress(total, total);

    let f = |k: &str, d: f64| config.get(k).and_then(Json::as_f64).unwrap_or(d) as f32;
    let theta = config.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Json::as_f64).unwrap_or(1e6) as f32;
    let meta: Vec<(&str, gguf_file::Value)> = vec![
        ("general.architecture", gguf_file::Value::String("music3-lm".into())),
        ("music3.quant", gguf_file::Value::String(quant.into())),
        ("music3.layers", gguf_file::Value::U32(layers)),
        ("music3.heads", gguf_file::Value::U32(int("num_attention_heads")?)),
        ("music3.kv_heads", gguf_file::Value::U32(int("num_key_value_heads")?)),
        ("music3.head_dim", gguf_file::Value::U32(int("head_dim")?)),
        ("music3.eps", gguf_file::Value::F32(f("rms_norm_eps", 1e-6))),
        ("music3.rope_theta", gguf_file::Value::F32(theta)),
    ];
    let meta_refs: Vec<(&str, &gguf_file::Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    // Written beside the target and renamed, so a half-written file is never
    // mistaken for a finished one.
    let partial = out.with_extension("gguf.partial");
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&partial)?);
        gguf_file::write(&mut file, &meta_refs, &tensor_refs)?;
        std::io::Write::flush(&mut file)?;
    }
    std::fs::rename(&partial, out)?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(Json::obj([
        ("path", Json::str(out.to_string_lossy())),
        ("quant", Json::str(quant)),
        ("bytes", Json::Int(bytes as i64)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
    ]))
}

/// A worker request: `{"kind": "music_quantize", "model_dir", "quant", "output"?}`.
pub struct Request {
    pub model_dir: PathBuf,
    pub quant: String,
    pub output: PathBuf,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let model_dir: PathBuf = j.get("model_dir").and_then(Json::as_str).filter(|p| !p.trim().is_empty()).ok_or("music_quantize: missing model_dir")?.into();
        let quant = j.get("quant").and_then(Json::as_str).unwrap_or("q4_k").to_string();
        parse(&quant)?;
        let output = j.get("output").and_then(Json::as_str).filter(|p| !p.trim().is_empty()).map(PathBuf::from).unwrap_or_else(|| default_path(&model_dir, &quant));
        Ok(Self { model_dir, quant, output })
    }
}

pub fn run(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    convert(&r.model_dir, &r.quant, &r.output, |done, total| {
        report(Json::obj([("stage", Json::str("quantizing")), ("current", Json::Int(done as i64)), ("total", Json::Int(total as i64))]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_matrices_round_trip() -> Result<()> {
        let t = Tensor::randn(0f32, 1., (70, 512), &Device::Cpu)?;
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K] {
            let whole = QTensor::quantize(&t, dtype)?.dequantize(&Device::Cpu)?;
            let parallel = quantize(&t, dtype)?.dequantize(&Device::Cpu)?;
            let err = |a: &Tensor| -> Result<f32> { Ok(((a - &t)?.sqr()?.mean_all()?.to_scalar::<f32>()?).sqrt()) };
            println!("{dtype:?}: whole {:.3e}, parallel {:.3e}", err(&whole)?, err(&parallel)?);
            assert!(err(&parallel)? < 0.1, "{dtype:?}");
        }
        Ok(())
    }
}
