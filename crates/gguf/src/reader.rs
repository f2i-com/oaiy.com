//! GGUF file reader. Memory-maps the file and exposes typed metadata + raw
//! tensor data slices.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{GgufError, Result};
use crate::source::TensorBytes; // VENDORED-LOCAL
use crate::tensor::{GgmlType, TensorInfo};
use crate::value::{Array, Value, ValueType};
use crate::{DEFAULT_ALIGNMENT, GGUF_MAGIC, SUPPORTED_VERSIONS};

/// Backing storage for a parsed GGUF file. Keeps either an mmap or owned bytes
/// alive so tensor-data slices stay valid.
#[derive(Debug)]
enum Backing {
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => &m[..],
            Backing::Bytes(b) => b,
        }
    }
}

/// A loaded GGUF file. Cheap to clone (`Arc` internally).
#[derive(Clone, Debug)]
pub struct GgufFile {
    inner: Arc<GgufInner>,
}

#[derive(Debug)]
struct GgufInner {
    backing:           Backing,
    version:           u32,
    metadata:          BTreeMap<String, Value>,
    tensors:           Vec<TensorInfo>,
    tensors_by_name:   BTreeMap<String, usize>,
    tensor_data_start: u64,
    alignment:         u64,
    // VENDORED-LOCAL: when set, tensor bodies are served through this source
    // (e.g. positioned reads from the file, for streamed experts) instead of
    // `backing`, which then holds only the header bytes (header + metadata
    // KV + tensor index + padding).
    source:            Option<Arc<dyn TensorBytes>>,
}

impl GgufFile {
    /// Memory-map the file at `path` and parse its header / tensor table.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // Safety: GGUF files we read are not concurrently mutated.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mmap(mmap))
    }

    /// Parse from an in-memory buffer. Useful for tests / pipelines.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Backing::Bytes(bytes))
    }

    // VENDORED-LOCAL: everything from here to `tensor_source` is the
    // byte-source seam — see `crate::source`.

    /// Parse `header_bytes` (raw GGUF bytes up to the tensor data section:
    /// header + metadata KV + tensor index, padding optional) and serve
    /// tensor bodies from `source` instead of a file mapping, through the
    /// exact same parsed-metadata view as a mmap'd `.gguf`.
    pub fn from_bytes_with_source(
        header_bytes: Vec<u8>,
        source: Arc<dyn TensorBytes>,
    ) -> Result<Self> {
        let mut inner = parse(Backing::Bytes(header_bytes))?;
        inner.source = Some(source);
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Open the `.gguf` at `path` with tensor bodies read on demand by
    /// positioned reads ([`FileSource`](crate::source::FileSource)) instead
    /// of a mapping: only the header is held in memory. This is how a model
    /// streams its experts from disk.
    pub fn open_streaming(path: impl AsRef<Path>) -> Result<Self> {
        let probe = Self::open(path.as_ref())?;
        let start = probe.tensor_data_start();
        let header = probe.raw_slice(0, start as usize).to_vec();
        drop(probe);
        let source = crate::source::FileSource::open(path.as_ref(), start)?;
        Self::from_bytes_with_source(header, Arc::new(source))
    }

    /// The external tensor-byte source, if this file was opened with
    /// [`from_bytes_with_source`](Self::from_bytes_with_source) or
    /// [`open_streaming`](Self::open_streaming).
    pub fn tensor_source(&self) -> Option<&Arc<dyn TensorBytes>> {
        self.inner.source.as_ref()
    }

    /// Raw bytes for the tensor body as an owned-or-borrowed buffer:
    /// borrowed from the mmap on the classic path, owned from the external
    /// source on a source-backed file. Prefer this over [`tensor_data`] in
    /// code that should work for both.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>> {
        match &self.inner.source {
            Some(src) => Ok(Cow::Owned(src.read_tensor(t)?)),
            None => Ok(Cow::Borrowed(self.tensor_data(t))),
        }
    }

    /// Total length of the backing byte buffer in bytes.
    pub fn raw_len(&self) -> usize {
        self.inner.backing.as_slice().len()
    }

    fn from_backing(backing: Backing) -> Result<Self> {
        let inner = parse(backing)?;
        Ok(Self { inner: Arc::new(inner) })
    }

    pub fn version(&self) -> u32 { self.inner.version }
    pub fn alignment(&self) -> u64 { self.inner.alignment }
    pub fn tensor_data_start(&self) -> u64 { self.inner.tensor_data_start }

    pub fn metadata(&self) -> &BTreeMap<String, Value> { &self.inner.metadata }
    pub fn tensors(&self) -> &[TensorInfo] { &self.inner.tensors }

    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.inner.tensors_by_name.get(name).map(|&i| &self.inner.tensors[i])
    }

    /// Raw bytes for the tensor body — zero-copy slice into the mmap.
    ///
    /// Panics on a source-backed file ([`from_bytes_with_source`]): the
    /// backing then holds only the header. Use [`tensor_bytes`] there.
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        // VENDORED-LOCAL: fail loudly instead of slicing past the header.
        assert!(
            self.inner.source.is_none(),
            "tensor_data() on a source-backed GgufFile; use tensor_bytes()"
        );
        let start = (self.inner.tensor_data_start + t.offset) as usize;
        let end   = start + t.nbytes() as usize;
        &self.inner.backing.as_slice()[start..end]
    }

    /// Absolute file offset of `t`'s data body. Combine with [`raw_slice`] to
    /// build long-lived zero-copy views of the mmap (so a clone of the
    /// `GgufFile` can keep the slice alive past a lookup-by-name borrow).
    pub fn tensor_data_offset(&self, t: &TensorInfo) -> usize {
        (self.inner.tensor_data_start + t.offset) as usize
    }

    /// Zero-copy slice into the file backing. Caller is responsible for the
    /// `(offset, len)` being in bounds — paired with `tensor_data_offset` and
    /// `TensorInfo::nbytes()`. Used by the host-resident-via-mmap weight tier
    /// so the loader can hold an `Arc<GgufFile>` view without copying bytes
    /// into a `Vec<u8>` heap buffer.
    pub fn raw_slice(&self, offset: usize, len: usize) -> &[u8] {
        // VENDORED-LOCAL: same guard as tensor_data — header-only backing.
        assert!(
            self.inner.source.is_none(),
            "raw_slice() on a source-backed GgufFile; use tensor_source()"
        );
        &self.inner.backing.as_slice()[offset..offset + len]
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.inner
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        let v = self.get(key)?;
        v.as_u64().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "integer",
            actual: v.type_str(),
        })
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        let v = self.get(key)?;
        v.as_f32().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "float",
            actual: v.type_str(),
        })
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        let v = self.get(key)?;
        v.as_str().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "string",
            actual: v.type_str(),
        })
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let v = self.get(key)?;
        v.as_bool().ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected: "bool",
            actual: v.type_str(),
        })
    }

    /// Convenience: `general.architecture`.
    pub fn architecture(&self) -> Result<&str> {
        self.get_str("general.architecture")
    }
}

// ----- parsing -------------------------------------------------------------

struct Cursor<'a> {
    bytes: &'a [u8],
    pos:   usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }

    #[inline]
    fn ensure(&self, n: usize) -> Result<()> {
        if self.pos + n > self.bytes.len() {
            Err(GgufError::Truncated {
                offset: self.pos as u64,
                needed: (self.pos + n - self.bytes.len()) as u64,
            })
        } else {
            Ok(())
        }
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }

    #[inline]
    fn read_i8(&mut self) -> Result<i8> { self.read_u8().map(|b| b as i8) }

    #[inline]
    fn read_u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.bytes[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16> { self.read_u16().map(|v| v as i16) }

    #[inline]
    fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32> { self.read_u32().map(|v| v as i32) }

    #[inline]
    fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64> { self.read_u64().map(|v| v as i64) }

    #[inline]
    fn read_f32(&mut self) -> Result<f32> { self.read_u32().map(f32::from_bits) }

    #[inline]
    fn read_f64(&mut self) -> Result<f64> { self.read_u64().map(f64::from_bits) }

    #[inline]
    fn read_bool(&mut self) -> Result<bool> { Ok(self.read_u8()? != 0) }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = String::from_utf8(self.bytes[self.pos..self.pos + len].to_vec())?;
        self.pos += len;
        Ok(s)
    }

    fn read_value(&mut self, ty: ValueType, key: &str) -> Result<Value> {
        Ok(match ty {
            ValueType::U8     => Value::U8(self.read_u8()?),
            ValueType::I8     => Value::I8(self.read_i8()?),
            ValueType::U16    => Value::U16(self.read_u16()?),
            ValueType::I16    => Value::I16(self.read_i16()?),
            ValueType::U32    => Value::U32(self.read_u32()?),
            ValueType::I32    => Value::I32(self.read_i32()?),
            ValueType::F32    => Value::F32(self.read_f32()?),
            ValueType::Bool   => Value::Bool(self.read_bool()?),
            ValueType::String => Value::String(self.read_string()?),
            ValueType::U64    => Value::U64(self.read_u64()?),
            ValueType::I64    => Value::I64(self.read_i64()?),
            ValueType::F64    => Value::F64(self.read_f64()?),
            ValueType::Array  => Value::Array(self.read_array(key)?),
        })
    }

    fn read_array(&mut self, key: &str) -> Result<Array> {
        let elem_ty = ValueType::from_u32(self.read_u32()?)?;
        let len = self.read_u64()? as usize;

        macro_rules! collect {
            ($variant:ident, $reader:ident) => {{
                let mut out = Vec::with_capacity(len);
                for _ in 0..len { out.push(self.$reader()?); }
                Array::$variant(out)
            }};
        }

        Ok(match elem_ty {
            ValueType::U8     => collect!(U8, read_u8),
            ValueType::I8     => collect!(I8, read_i8),
            ValueType::U16    => collect!(U16, read_u16),
            ValueType::I16    => collect!(I16, read_i16),
            ValueType::U32    => collect!(U32, read_u32),
            ValueType::I32    => collect!(I32, read_i32),
            ValueType::F32    => collect!(F32, read_f32),
            ValueType::Bool   => collect!(Bool, read_bool),
            ValueType::U64    => collect!(U64, read_u64),
            ValueType::I64    => collect!(I64, read_i64),
            ValueType::F64    => collect!(F64, read_f64),
            ValueType::String => {
                let mut out = Vec::with_capacity(len);
                for _ in 0..len { out.push(self.read_string()?); }
                Array::String(out)
            }
            ValueType::Array => return Err(GgufError::NestedArray(key.to_string())),
        })
    }
}

fn parse(backing: Backing) -> Result<GgufInner> {
    let bytes = backing.as_slice();
    let mut c = Cursor::new(bytes);

    let magic = c.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic(magic));
    }

    let version = c.read_u32()?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion(version, SUPPORTED_VERSIONS));
    }

    let tensor_count = c.read_u64()?;
    let kv_count     = c.read_u64()?;

    // Metadata KVs.
    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = c.read_string()?;
        let ty  = ValueType::from_u32(c.read_u32()?)?;
        let val = c.read_value(ty, &key)?;
        metadata.insert(key, val);
    }

    // Optional alignment override.
    let alignment = match metadata.get("general.alignment") {
        Some(v) => v.as_u64().unwrap_or(DEFAULT_ALIGNMENT),
        None    => DEFAULT_ALIGNMENT,
    };

    // Tensor info table.
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut tensors_by_name = BTreeMap::new();
    for i in 0..tensor_count {
        let name = c.read_string()?;
        let n_dims = c.read_u32()?;
        if n_dims > 4 {
            return Err(GgufError::TooManyDims { name, n_dims });
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(c.read_u64()?);
        }
        let dtype = GgmlType::from_u32(c.read_u32()?)?;
        let offset = c.read_u64()?;

        // Verify block alignment.
        let numel: u64 = shape.iter().product();
        let block = dtype.block_size();
        if block > 1 && numel % (block as u64) != 0 {
            return Err(GgufError::NotBlockAligned {
                name: name.clone(),
                block,
                numel,
            });
        }

        tensors_by_name.insert(name.clone(), i as usize);
        tensors.push(TensorInfo { name, shape, dtype, offset });
    }

    // Pad to alignment.
    let unaligned = c.pos as u64;
    let tensor_data_start = align_up(unaligned, alignment);

    Ok(GgufInner {
        backing,
        version,
        metadata,
        tensors,
        tensors_by_name,
        tensor_data_start,
        alignment,
        source: None, // VENDORED-LOCAL: set by from_bytes_with_source
    })
}

#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two() || align != 0);
    if align == 0 { return value; }
    let r = value % align;
    if r == 0 { value } else { value + (align - r) }
}

// ----- minimal write-side for round-trip tests -----------------------------

/// Writes a GGUF file from already-parsed pieces. Intended primarily for
/// round-trip testing; production writers will likely want a streaming variant.
#[doc(hidden)]
pub fn write_to_vec(
    metadata: &BTreeMap<String, Value>,
    tensors: &[(TensorInfo, Vec<u8>)],
    alignment: u64,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());                    // version
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (k, v) in metadata {
        write_string(&mut out, k);
        write_value(&mut out, v);
    }

    // First pass: figure out tensor offsets relative to tensor_data_start.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for (info, data) in tensors {
        offsets.push(cursor);
        let nbytes = info.nbytes();
        debug_assert_eq!(nbytes as usize, data.len(), "tensor `{}` data length mismatch", info.name);
        cursor += nbytes;
        // Tensor data is *not* internally re-aligned per-tensor in GGUF; only
        // the start of the data section is aligned.
    }

    for ((info, _), &offset) in tensors.iter().zip(offsets.iter()) {
        write_string(&mut out, &info.name);
        out.extend_from_slice(&(info.shape.len() as u32).to_le_bytes());
        for d in &info.shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(info.dtype as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    // Pad to alignment.
    let unaligned = out.len() as u64;
    let aligned = align_up(unaligned, alignment);
    out.resize(aligned as usize, 0);

    for (_, data) in tensors {
        out.extend_from_slice(data);
    }

    Ok(out)
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    let tag = match v {
        Value::U8(_)     => ValueType::U8,
        Value::I8(_)     => ValueType::I8,
        Value::U16(_)    => ValueType::U16,
        Value::I16(_)    => ValueType::I16,
        Value::U32(_)    => ValueType::U32,
        Value::I32(_)    => ValueType::I32,
        Value::F32(_)    => ValueType::F32,
        Value::Bool(_)   => ValueType::Bool,
        Value::String(_) => ValueType::String,
        Value::Array(_)  => ValueType::Array,
        Value::U64(_)    => ValueType::U64,
        Value::I64(_)    => ValueType::I64,
        Value::F64(_)    => ValueType::F64,
    };
    out.extend_from_slice(&(tag as u32).to_le_bytes());

    match v {
        Value::U8(x)     => out.push(*x),
        Value::I8(x)     => out.push(*x as u8),
        Value::U16(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x)   => out.push(*x as u8),
        Value::String(s) => write_string(out, s),
        Value::U64(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x)    => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(a)  => write_array(out, a),
    }
}

fn write_array(out: &mut Vec<u8>, a: &Array) {
    out.extend_from_slice(&(a.element_type() as u32).to_le_bytes());
    out.extend_from_slice(&(a.len() as u64).to_le_bytes());
    match a {
        Array::U8(v)     => out.extend_from_slice(v),
        Array::I8(v)     => out.extend_from_slice(bytemuck_i8(v)),
        Array::U16(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::I16(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::U32(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::I32(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::F32(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::Bool(v)   => for x in v { out.push(*x as u8); },
        Array::String(v) => for s in v { write_string(out, s); },
        Array::U64(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::I64(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
        Array::F64(v)    => for x in v { out.extend_from_slice(&x.to_le_bytes()); },
    }
}

#[inline]
fn bytemuck_i8(v: &[i8]) -> &[u8] {
    // Safety: i8 and u8 have the same layout.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

// ----- tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_gguf() -> Vec<u8> {
        let mut md: BTreeMap<String, Value> = BTreeMap::new();
        md.insert("general.architecture".into(), Value::String("llama".into()));
        md.insert("general.alignment".into(),    Value::U32(32));
        md.insert("llama.embedding_length".into(), Value::U32(4));
        md.insert("llama.block_count".into(),      Value::U32(2));
        md.insert("test.array.u32".into(),
                  Value::Array(Array::U32(vec![1, 2, 3, 4])));
        md.insert("test.array.string".into(),
                  Value::Array(Array::String(vec!["alpha".into(), "beta".into()])));

        let t1_data: Vec<u8> = (0u32..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let t1 = TensorInfo {
            name: "weight.a".into(),
            shape: vec![4, 2],
            dtype: GgmlType::F32,
            offset: 0,
        };

        let t2_data: Vec<u8> = (0u32..4).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect();
        let t2 = TensorInfo {
            name: "weight.b".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            offset: 0,
        };

        write_to_vec(&md, &[(t1, t1_data), (t2, t2_data)], 32).unwrap()
    }

    /// VENDORED-LOCAL: a streaming open serves the same metadata and tensor
    /// bytes through positioned reads as the mmap does.
    #[test]
    fn open_streaming_matches_mmap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("simple.gguf");
        std::fs::write(&path, make_simple_gguf()).unwrap();
        let mapped = GgufFile::open(&path).unwrap();
        let streamed = GgufFile::open_streaming(&path).unwrap();
        assert!(mapped.tensor_source().is_none());
        let src = streamed.tensor_source().expect("streaming file has a source");
        assert_eq!(streamed.architecture().unwrap(), "llama");
        assert_eq!(streamed.tensor_data_start(), mapped.tensor_data_start());
        for t in mapped.tensors() {
            let want = mapped.tensor_data(t);
            assert_eq!(&*streamed.tensor_bytes(t).unwrap(), want, "{}", t.name);
            assert_eq!(src.read_tensor(t).unwrap(), want, "{}", t.name);
        }
        // a range across both tensors, and one past the end
        let b = mapped.tensor_by_name("weight.b").unwrap();
        let span = (b.offset + b.nbytes()) as usize;
        let mut got = vec![0u8; span - 4];
        src.read_range(4, &mut got).unwrap();
        let start = mapped.tensor_data_start() as usize + 4;
        assert_eq!(got, mapped.raw_slice(start, span - 4));
        assert!(src.read_range(span as u64 + (1 << 20), &mut [0u8; 8]).is_err());
    }

    #[test]
    fn roundtrip_simple() {
        let bytes = make_simple_gguf();
        let f = GgufFile::from_bytes(bytes).unwrap();

        assert_eq!(f.architecture().unwrap(), "llama");
        assert_eq!(f.get_u64("llama.embedding_length").unwrap(), 4);
        assert_eq!(f.get_u64("llama.block_count").unwrap(), 2);
        assert_eq!(f.alignment(), 32);

        let arr = f.get("test.array.u32").unwrap().as_array().unwrap();
        match arr {
            Array::U32(v) => assert_eq!(v, &vec![1, 2, 3, 4]),
            _ => panic!("expected U32 array"),
        }

        let arr = f.get("test.array.string").unwrap().as_array().unwrap();
        match arr {
            Array::String(v) => assert_eq!(v, &vec!["alpha".to_string(), "beta".into()]),
            _ => panic!("expected String array"),
        }

        assert_eq!(f.tensors().len(), 2);
        let a = f.tensor_by_name("weight.a").unwrap();
        assert_eq!(a.shape, vec![4, 2]);
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.numel(), 8);
        assert_eq!(a.nbytes(), 32);
        assert_eq!(f.tensor_data(a).len(), 32);

        let b = f.tensor_by_name("weight.b").unwrap();
        assert_eq!(b.numel(), 4);

        // Verify tensor data round-trips.
        let bytes_a = f.tensor_data(a);
        let recovered: Vec<f32> = bytes_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(recovered, (0..8).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = vec![0u8; 64];
        let err = GgufFile::from_bytes(bytes).unwrap_err();
        assert!(matches!(err, GgufError::BadMagic(_)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let err = GgufFile::from_bytes(bytes).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(99, _)));
    }

    #[test]
    fn alignment_padding_is_correct() {
        let bytes = make_simple_gguf();
        let f = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(f.tensor_data_start() % f.alignment(), 0);
    }
}
