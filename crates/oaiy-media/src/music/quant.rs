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
    quantized::{ggml_file, GgmlDType, QTensor},
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

/// A metadata value of the converted file.
enum Meta {
    Str(String),
    U32(u32),
    F32(f32),
}

/// GGUF's number for a tensor format (candle keeps its own to itself).
fn ggml_type(d: GgmlDType) -> Result<u32> {
    Ok(match d {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::BF16 => 30,
        other => return Err(msg(format!("no GGUF number for {other:?} here"))),
    })
}

/// The converted file as it is made: each tensor's bytes appended to a spill file beside the target as soon as they
/// are quantized (only its name, shape and format kept), then the GGUF header (candle's layout: version 2, its
/// tensors 32 bytes apart) and the spill copied after it. Held in memory until the end, the 36 layers and the text
/// table were 13.7 GB at once for `q8_0`, and a 16 GB machine (the kind the smaller file is for) killed the
/// conversion.
struct Spill {
    file: std::io::BufWriter<std::fs::File>,
    path: PathBuf,
    tensors: Vec<(String, Vec<usize>, GgmlDType)>,
    open: usize,
}

impl Spill {
    fn create(path: PathBuf) -> Result<Self> {
        Ok(Self { file: std::io::BufWriter::new(std::fs::File::create(&path)?), path, tensors: Vec::new(), open: 0 })
    }

    /// A tensor whose bytes follow ([`Self::bytes`], then [`Self::end`]).
    fn begin(&mut self, name: &str, dims: Vec<usize>, dtype: GgmlDType) {
        self.tensors.push((name.to_string(), dims, dtype));
        self.open = 0;
    }

    fn bytes(&mut self, b: &[u8]) -> Result<()> {
        std::io::Write::write_all(&mut self.file, b)?;
        self.open += b.len();
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        let (_, dims, dtype) = self.tensors.last().ok_or_else(|| msg("no tensor begun"))?;
        let want = dims.iter().product::<usize>() / dtype.block_size() * dtype.type_size();
        if self.open != want {
            return Err(msg(format!("{}: {} bytes written, {want} its shape's", self.tensors.last().map_or("", |t| t.0.as_str()), self.open)));
        }
        std::io::Write::write_all(&mut self.file, &[0u8; 32][..(32 - self.open % 32) % 32])?;
        Ok(())
    }

    fn push(&mut self, name: &str, t: QTensor) -> Result<()> {
        self.begin(name, t.shape().dims().to_vec(), t.dtype());
        self.bytes(&t.data()?)?;
        self.end()
    }

    /// The header, then the spill's bytes, into `out` (through a partial file renamed, so a half-written file is
    /// never mistaken for a finished one); the spill removed.
    fn finish(mut self, out: &Path, meta: &[(&str, Meta)]) -> Result<()> {
        use std::io::{Seek, Write};
        self.file.flush()?;
        drop(self.file);
        let string = |w: &mut dyn Write, s: &str| -> Result<()> {
            w.write_all(&(s.len() as u64).to_le_bytes())?;
            w.write_all(s.as_bytes())?;
            Ok(())
        };
        let partial = out.with_extension("gguf.partial");
        {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&partial)?);
            w.write_all(&0x4655_4747u32.to_le_bytes())?;
            w.write_all(&2u32.to_le_bytes())?;
            w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
            w.write_all(&(meta.len() as u64).to_le_bytes())?;
            for (key, value) in meta {
                string(&mut w, key)?;
                match value {
                    Meta::Str(v) => {
                        w.write_all(&8u32.to_le_bytes())?;
                        string(&mut w, v)?;
                    }
                    Meta::U32(v) => {
                        w.write_all(&4u32.to_le_bytes())?;
                        w.write_all(&v.to_le_bytes())?;
                    }
                    Meta::F32(v) => {
                        w.write_all(&6u32.to_le_bytes())?;
                        w.write_all(&v.to_le_bytes())?;
                    }
                }
            }
            let mut offset = 0u64;
            for (name, dims, dtype) in &self.tensors {
                string(&mut w, name)?;
                w.write_all(&(dims.len() as u32).to_le_bytes())?;
                for d in dims.iter().rev() {
                    w.write_all(&(*d as u64).to_le_bytes())?;
                }
                w.write_all(&ggml_type(*dtype)?.to_le_bytes())?;
                w.write_all(&offset.to_le_bytes())?;
                let size = (dims.iter().product::<usize>() / dtype.block_size() * dtype.type_size()) as u64;
                offset += size + (32 - size % 32) % 32;
            }
            let at = w.stream_position()?;
            w.write_all(&[0u8; 32][..((32 - at % 32) % 32) as usize])?;
            std::io::copy(&mut std::fs::File::open(&self.path)?, &mut w)?;
            w.flush()?;
        }
        std::fs::remove_file(&self.path)?;
        std::fs::rename(&partial, out)?;
        Ok(())
    }
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
    let mut spill = Spill::create(out.with_extension("gguf.spill"))?;
    let total = layers as usize + 1;
    for i in 0..layers {
        let l = format!("model.layers.{i}");
        let mut get = |n: &str| store.tensor(&format!("{l}.{n}.weight"), &cpu, false);
        let qkv = Tensor::cat(&[get("self_attn.q_proj")?, get("self_attn.k_proj")?, get("self_attn.v_proj")?], 0)?;
        spill.push(&format!("layers.{i}.qkv"), quantize(&qkv, dtype)?)?;
        drop(qkv);
        spill.push(&format!("layers.{i}.o"), quantize(&get("self_attn.o_proj")?, dtype)?)?;
        let gate_up = Tensor::cat(&[get("mlp.gate_proj")?, get("mlp.up_proj")?], 0)?;
        spill.push(&format!("layers.{i}.gate_up"), quantize(&gate_up, dtype)?)?;
        drop(gate_up);
        spill.push(&format!("layers.{i}.down"), quantize(&get("mlp.down_proj")?, dtype)?)?;
        for (name, key) in [("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"), ("input_norm", "input_layernorm"), ("post_norm", "post_attention_layernorm")] {
            spill.push(&format!("layers.{i}.{name}"), QTensor::quantize(&get(key)?.to_dtype(DType::F32)?, GgmlDType::F32)?)?;
        }
        progress(i as usize + 1, total);
    }
    let code_rows: Vec<u32> = (CODE_OFFSET..CODE_OFFSET + SEMANTIC_CODES as u32).collect();
    let head_rows: Vec<u32> = std::iter::once(AUDIO_END).chain(code_rows.iter().copied()).collect();
    spill.push("norm", QTensor::quantize(&store.tensor("model.norm.weight", &cpu, false)?.to_dtype(DType::F32)?, GgmlDType::F32)?)?;
    spill.push("head", quantize(&store.rows("lm_head.weight", &head_rows, &cpu)?, GgmlDType::BF16)?)?;
    spill.push("codes", quantize(&store.rows("model.embed_tokens.weight", &code_rows, &cpu)?, GgmlDType::BF16)?)?;
    // Every token a prompt can hold lies below the first audio code: its rows a batch at a time (the whole table as
    // f32 is 2.5 GB, and its copies as many again).
    let hidden = int("hidden_size")? as usize;
    spill.begin("text", vec![CODE_OFFSET as usize, hidden], GgmlDType::Q8_0);
    const BATCH: u32 = 8192;
    for first in (0..CODE_OFFSET).step_by(BATCH as usize) {
        let rows: Vec<u32> = (first..(first + BATCH).min(CODE_OFFSET)).collect();
        let part = quantize(&store.rows("model.embed_tokens.weight", &rows, &cpu)?, GgmlDType::Q8_0)?;
        spill.bytes(&part.data()?)?;
    }
    spill.end()?;
    progress(total, total);

    let f = |k: &str, d: f64| config.get(k).and_then(Json::as_f64).unwrap_or(d) as f32;
    let theta = config.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Json::as_f64).unwrap_or(1e6) as f32;
    let meta = [
        ("general.architecture", Meta::Str("music3-lm".into())),
        ("music3.quant", Meta::Str(quant.into())),
        ("music3.layers", Meta::U32(layers)),
        ("music3.heads", Meta::U32(int("num_attention_heads")?)),
        ("music3.kv_heads", Meta::U32(int("num_key_value_heads")?)),
        ("music3.head_dim", Meta::U32(int("head_dim")?)),
        ("music3.eps", Meta::F32(f("rms_norm_eps", 1e-6))),
        ("music3.rope_theta", Meta::F32(theta)),
    ];
    spill.finish(out, &meta)?;
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

    /// The spilled file is candle's writer's to the byte (the music model reads it with candle's reader).
    #[test]
    fn a_spilled_file_is_candles() -> Result<()> {
        use candle_core::quantized::gguf_file;
        let dir = std::env::temp_dir().join(format!("oaiy-spill-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let made = [
            ("a", QTensor::quantize(&Tensor::randn(0f32, 1., (3, 256), &Device::Cpu)?, GgmlDType::Q4K)?),
            ("b", QTensor::quantize(&Tensor::randn(0f32, 1., (5,), &Device::Cpu)?, GgmlDType::F32)?),
            ("c", QTensor::quantize(&Tensor::randn(0f32, 1., (7, 64), &Device::Cpu)?, GgmlDType::Q8_0)?),
            ("d", QTensor::quantize(&Tensor::randn(0f32, 1., (3, 6), &Device::Cpu)?, GgmlDType::BF16)?),
        ];
        let theirs = dir.join("theirs.gguf");
        {
            let meta = [("general.architecture", gguf_file::Value::String("music3-lm".into())), ("n", gguf_file::Value::U32(7)), ("x", gguf_file::Value::F32(0.5))];
            let meta: Vec<(&str, &gguf_file::Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
            let tensors: Vec<(&str, &QTensor)> = made.iter().map(|(k, t)| (*k, t)).collect();
            let mut f = std::fs::File::create(&theirs)?;
            gguf_file::write(&mut f, &meta, &tensors)?;
        }
        let ours = dir.join("ours.gguf");
        let mut spill = Spill::create(dir.join("ours.gguf.spill"))?;
        for (name, t) in made {
            spill.push(name, t)?;
        }
        spill.finish(&ours, &[("general.architecture", Meta::Str("music3-lm".into())), ("n", Meta::U32(7)), ("x", Meta::F32(0.5))])?;
        let (a, b) = (std::fs::read(&theirs)?, std::fs::read(&ours)?);
        std::fs::remove_dir_all(&dir)?;
        assert!(a == b, "{} bytes against candle's {}", b.len(), a.len());
        Ok(())
    }

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
