//! Where the model's tensors come from. The layers load through
//! [`TensorSource`], so the same code serves this crate's own safetensors
//! reader and oaiy-media's weight store (with its quantized formats).
use candle_core::{DType, Device, Result, Tensor};
use dsv41::safetensors::{Dtype, StIndex};
use std::path::Path;

/// Stop Candle's CUDA tensors made from here on (on this device) from
/// carrying events. Candle runs each thread's work on one stream, where the
/// events cudarc records at every use and waits on when a buffer is freed
/// order nothing; they cost a driver call per use, and one recorded while a
/// CUDA graph is captured cannot be waited on afterwards (freeing such a
/// buffer then leaves an error for the next call to report). Call it before
/// making the tensors a graph will use; the weights' loaders do.
pub fn untracked(dev: &Device) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = dev {
        // SAFETY: the events only order work across streams. Candle queues a
        // thread's work on that thread's one stream, and this crate hands
        // tensors between threads only after synchronizing the device (at
        // the end of loading and of every line), so no use depends on them.
        unsafe { cuda.disable_event_tracking() }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = dev;
}

/// Named tensors, read on demand.
pub trait TensorSource {
    /// A weight on `dev`, in the model's compute type (BF16 for the
    /// checkpoints).
    fn load(&mut self, name: &str, dev: &Device) -> Result<Tensor>;
    /// A weight on `dev`, in F32 (the codec and the encoders run in F32).
    fn load_f32(&mut self, name: &str, dev: &Device) -> Result<Tensor>;
    /// Only the rows `ids` of a 2-D table, in BF16: a large embedding stays
    /// in the file and only a prompt's rows are read.
    fn load_rows(&mut self, name: &str, ids: &[u32], dev: &Device) -> Result<Tensor>;
    fn has(&self, name: &str) -> bool;
    fn tensor_names(&self) -> Vec<String>;
}

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// A safetensors file (or a folder of them), read in place.
pub struct Weights {
    index: StIndex,
    /// Bytes read from disk so far.
    pub disk_bytes: u64,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let index = if path.is_dir() { StIndex::open(path) } else { StIndex::open_file(path) };
        Ok(Self { index: index.map_err(|e| msg(format!("{}: {e}", path.display())))?, disk_bytes: 0 })
    }

    fn read(&mut self, name: &str, out: DType, dev: &Device) -> Result<Tensor> {
        let info = self.index.info(name).map_err(candle_core::Error::wrap)?.clone();
        let bytes = self.index.read(name).map_err(candle_core::Error::wrap)?;
        self.disk_bytes += bytes.len() as u64;
        decode(name, info.dtype, &info.shape, &bytes, out, dev)
    }
}

/// Stored bytes as a tensor of type `out` on `dev`.
fn decode(name: &str, dtype: Dtype, shape: &[usize], bytes: &[u8], out: DType, dev: &Device) -> Result<Tensor> {
    let stored = match dtype {
        Dtype::BF16 => DType::BF16,
        Dtype::F16 => DType::F16,
        Dtype::F32 => DType::F32,
        other => return Err(msg(format!("tensor {name}: unsupported dtype {other:?}"))),
    };
    let t = Tensor::from_raw_buffer(bytes, stored, shape, dev)?;
    if stored == out {
        Ok(t)
    } else {
        t.to_dtype(out)
    }
}

impl TensorSource for Weights {
    fn load(&mut self, name: &str, dev: &Device) -> Result<Tensor> {
        self.read(name, DType::BF16, dev)
    }

    fn load_f32(&mut self, name: &str, dev: &Device) -> Result<Tensor> {
        self.read(name, DType::F32, dev)
    }

    fn load_rows(&mut self, name: &str, ids: &[u32], dev: &Device) -> Result<Tensor> {
        use std::io::{Read, Seek, SeekFrom};
        let info = self.index.info(name).map_err(candle_core::Error::wrap)?.clone();
        if info.shape.len() != 2 || ids.iter().any(|&i| i as usize >= info.shape[0]) {
            return Err(msg(format!("invalid row selection for {name}")));
        }
        let width = info.shape[1];
        let row_bytes = width * info.dtype.size();
        let mut bytes = vec![0u8; ids.len() * row_bytes];
        let mut file = std::fs::File::open(self.index.shard_path(info.shard))?;
        for (i, &id) in ids.iter().enumerate() {
            file.seek(SeekFrom::Start(info.start + id as u64 * row_bytes as u64))?;
            file.read_exact(&mut bytes[i * row_bytes..(i + 1) * row_bytes])?;
        }
        self.disk_bytes += bytes.len() as u64;
        decode(name, info.dtype, &[ids.len(), width], &bytes, DType::BF16, dev)
    }

    fn has(&self, name: &str) -> bool {
        self.index.get(name).is_some()
    }

    fn tensor_names(&self) -> Vec<String> {
        self.index.names().map(str::to_owned).collect()
    }
}

/// Tensors already in memory (tests, and models assembled by hand). `load`
/// gives them as `dtype` (BF16 unless set: the CPU cannot multiply BF16, so
/// CPU checks use F32).
pub struct InMemory {
    pub tensors: std::collections::HashMap<String, Tensor>,
    pub dtype: DType,
}

impl Default for InMemory {
    fn default() -> Self {
        Self { tensors: Default::default(), dtype: DType::BF16 }
    }
}

impl InMemory {
    fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| msg(format!("missing tensor {name}")))
    }
}

impl TensorSource for InMemory {
    fn load(&mut self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.to_device(dev)?.to_dtype(self.dtype)
    }

    fn load_f32(&mut self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.to_device(dev)?.to_dtype(DType::F32)
    }

    fn load_rows(&mut self, name: &str, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let t = self.get(name)?;
        t.index_select(&Tensor::new(ids, t.device())?, 0)?.to_device(dev)?.to_dtype(self.dtype)
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

/// Every tensor whose name starts with `prefix`, in F32.
pub fn load_prefix_f32(source: &mut (impl TensorSource + ?Sized), prefix: &str, dev: &Device) -> Result<std::collections::HashMap<String, Tensor>> {
    let names: Vec<String> = source.tensor_names().into_iter().filter(|k| k.starts_with(prefix)).collect();
    if names.is_empty() {
        return Err(msg(format!("no {prefix}* tensors")));
    }
    let mut out = std::collections::HashMap::new();
    for k in names {
        let t = source.load_f32(&k, dev)?;
        out.insert(k, t);
    }
    Ok(out)
}

/// Bytes held by tensors (for reporting device memory).
pub fn tensor_bytes<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    tensors.into_iter().map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_st(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let (mut header, mut data) = (String::from("{"), Vec::new());
        for (i, (k, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header += &format!("\"{k}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{},{}]}}", data.len(), data.len() + bytes.len());
            data.extend_from_slice(bytes);
        }
        header.push('}');
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend(data);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn safetensors_load_whole_tensors_and_selected_rows() -> Result<()> {
        let path = std::env::temp_dir().join(format!("oaiy-tts-weights-{}.safetensors", std::process::id()));
        let f32s: Vec<u8> = (1..=6).flat_map(|n| (n as f32).to_le_bytes()).collect();
        let bf16s: Vec<u8> = [1.5f32, -2.0].iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        write_st(&path, &[("emb", "F32", &[3, 2], f32s), ("w", "BF16", &[2], bf16s)]);
        let mut w = Weights::open(&path)?;
        assert!(w.has("emb") && !w.has("nope"));
        let rows = w.load_rows("emb", &[2, 0, 2], &Device::Cpu)?;
        assert_eq!(rows.dtype(), DType::BF16);
        assert_eq!(rows.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, vec![5., 6., 1., 2., 5., 6.]);
        assert!(w.load_rows("emb", &[3], &Device::Cpu).is_err());
        assert_eq!(w.load_f32("w", &Device::Cpu)?.to_vec1::<f32>()?, vec![1.5, -2.0]);
        assert_eq!(w.load("emb", &Device::Cpu)?.dtype(), DType::BF16);
        let mut names = w.tensor_names();
        names.sort();
        assert_eq!(names, vec!["emb", "w"]);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
