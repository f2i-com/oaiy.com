//! Read-only index over a sharded safetensors checkpoint.
//!
//! A shard is `[u64 LE header length][JSON header][raw tensor bytes]`; the
//! header maps each tensor name to its dtype, shape and `[start, end)` byte
//! range relative to the end of the header. Only the headers are read at
//! open (about 200 KB per shard here), so indexing 96,085 tensors across 48
//! shards costs milliseconds and no tensor bytes. The checkpoint is used in
//! place: nothing is converted or copied.
//!
//! Untrusted input rules apply (CONVENTIONS.md): a truncated header, a range
//! past the end of the file, or a byte count that disagrees with
//! `shape x dtype` is an `Error::Format`, never a panic.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use oaiy_engine::json::Json;
use oaiy_engine::{Error, Result};

use crate::io::read_exact_at;

/// Largest header accepted. Real shards carry ~200 KB; this only stops a
/// corrupt length prefix from asking for gigabytes.
const MAX_HEADER_BYTES: u64 = 64 << 20;

/// Element types that appear in DeepSeek-V4.1 checkpoints and golden files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I8,
    I16,
    U8,
    I32,
    I64,
    /// fp8 e4m3fn (bias 7, no infinities, 0x7f/0xff are NaN).
    F8E4M3,
    /// Power-of-two scale: value = 2^(bits - 127).
    F8E8M0,
}

impl Dtype {
    fn parse(tag: &str) -> Option<Self> {
        Some(match tag {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "U8" => Self::U8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            _ => return None,
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 | Self::F8E4M3 | Self::F8E8M0 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
        }
    }
}

/// Where one tensor lives.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Index into [`StIndex::shard_path`].
    pub shard: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte offset of the data inside the shard file.
    pub start: u64,
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Name -> location for every tensor in a checkpoint directory (or file).
pub struct StIndex {
    shards: Vec<PathBuf>,
    tensors: HashMap<String, TensorInfo>,
    /// String pairs from the shards' `__metadata__` objects.
    metadata: HashMap<String, String>,
}

impl StIndex {
    /// Index every `*.safetensors` file in `dir`, in name order.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(Error::Format(format!("no .safetensors files in {}", dir.display())));
        }
        Self::from_files(paths)
    }

    /// Index a single safetensors file (golden fixtures, small checkpoints).
    pub fn open_file(path: &Path) -> Result<Self> {
        Self::from_files(vec![path.to_path_buf()])
    }

    fn from_files(shards: Vec<PathBuf>) -> Result<Self> {
        let (mut tensors, mut metadata) = (HashMap::new(), HashMap::new());
        for (i, path) in shards.iter().enumerate() {
            index_shard(i, path, &mut tensors, &mut metadata)
                .map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        }
        Ok(StIndex { shards, tensors, metadata })
    }

    /// A `__metadata__` string value (safetensors allows string pairs only).
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn shard_path(&self, shard: usize) -> &Path {
        &self.shards[shard]
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Serve every tensor as `prefix` + its stored name, for checkpoints
    /// saved without the prefix their loader expects.
    pub fn prefix_names(&mut self, prefix: &str) {
        self.tensors = std::mem::take(&mut self.tensors).into_iter().map(|(k, v)| (format!("{prefix}{k}"), v)).collect();
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Like [`get`](Self::get), but a missing tensor is an error naming it.
    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.get(name)
            .ok_or_else(|| Error::Format(format!("tensor {name} not in checkpoint")))
    }

    /// The tensor's raw bytes (one positioned read through the page cache).
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.info(name)?;
        let len = usize::try_from(t.nbytes)
            .map_err(|_| Error::Format(format!("{name}: {} bytes do not fit in memory", t.nbytes)))?;
        let mut buf = vec![0u8; len];
        let file = File::open(&self.shards[t.shard])?;
        read_exact_at(&file, &mut buf, t.start)?;
        Ok(buf)
    }

    /// F32 / BF16 / F16 tensor decoded to `f32`.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let dtype = self.info(name)?.dtype;
        let bytes = self.read(name)?;
        Ok(match dtype {
            Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|c| crate::formats::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| crate::formats::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => return Err(Error::Format(format!("{name}: {other:?} is not a float tensor"))),
        })
    }

    /// I64 / I32 tensor widened to `i64`.
    pub fn read_i64(&self, name: &str) -> Result<Vec<i64>> {
        let dtype = self.info(name)?.dtype;
        let bytes = self.read(name)?;
        Ok(match dtype {
            Dtype::I64 => bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            Dtype::I32 => bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
            other => return Err(Error::Format(format!("{name}: {other:?} is not an integer tensor"))),
        })
    }
}

fn index_shard(
    shard: usize,
    path: &Path,
    out: &mut HashMap<String, TensorInfo>,
    metadata: &mut HashMap<String, String>,
) -> Result<()> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES || 8 + header_len > file_len {
        return Err(Error::Format(format!("header length {header_len} is implausible")));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let data_base = 8 + header_len;

    let doc = Json::parse(&header)?;
    let tensors = doc.as_object().ok_or_else(|| Error::Format("header is not a JSON object".into()))?;
    for (name, v) in tensors {
        let name = name.as_str();
        if name == "__metadata__" {
            for (mk, mv) in v.members() {
                if let Some(mv) = mv.as_str() {
                    metadata.insert(mk.to_string(), mv.to_string());
                }
            }
            continue;
        }
        let bad = |what: &str| Error::Format(format!("tensor {name}: {what}"));
        let tag = v.get("dtype").and_then(Json::as_str).ok_or_else(|| bad("missing dtype"))?;
        // Complex buffers (precomputed RoPE phases, say) are left out: nothing reads them.
        if tag == "C64" || tag == "C128" {
            continue;
        }
        let dtype = Dtype::parse(tag).ok_or_else(|| bad(&format!("unsupported dtype {tag}")))?;
        let dims = v.get("shape").and_then(Json::as_array).ok_or_else(|| bad("missing shape"))?;
        let mut shape = Vec::with_capacity(dims.len());
        for d in dims {
            let d = d.as_i64().unwrap_or(-1);
            if d < 0 {
                return Err(bad("negative or non-numeric dimension"));
            }
            shape.push(d as usize);
        }
        let offs = v.get("data_offsets").ok_or_else(|| bad("missing data_offsets"))?;
        let off = |i: usize| offs.at(i).and_then(Json::as_i64).unwrap_or(-1);
        let (a, b) = (off(0), off(1));
        if a < 0 || b < a || data_base + b as u64 > file_len {
            return Err(bad("data_offsets outside the file"));
        }
        let nbytes = (b - a) as u64;
        let expect = shape.iter().try_fold(dtype.size() as u64, |acc, &d| acc.checked_mul(d as u64));
        if expect != Some(nbytes) {
            return Err(bad(&format!("{nbytes} bytes but shape {shape:?} x {dtype:?} needs {expect:?}")));
        }
        let info = TensorInfo { shard, dtype, shape, start: data_base + a as u64, nbytes };
        if out.insert(name.to_string(), info).is_some() {
            return Err(bad("appears in more than one shard"));
        }
    }
    Ok(())
}
