//! GGUF file reader. Memory-maps the file and exposes typed metadata + raw
//! tensor data slices.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{GgufError, Result};
use crate::source::TensorBytes;
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

    /// Ask the OS to read `offset..offset + len` of a mapped file now, in large reads (`MADV_WILLNEED`), rather than a
    /// page at a time as a copy faults it in. Advice only: nothing changes if it is not taken.
    fn will_need(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        if let Backing::Mmap(m) = self {
            let _ = m.advise_range(memmap2::Advice::WillNeed, offset, len);
        }
        #[cfg(not(unix))]
        let _ = (offset, len);
    }
}

// one shard of a split GGUF beyond the first. The primary
// shard's bytes stay in `GgufInner::backing` so the single-file path is
// untouched.
#[derive(Debug)]
struct ShardBacking {
    backing:           Backing,
    tensor_data_start: u64,
    source:            Option<Arc<dyn TensorBytes>>,
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
    // when set, tensor bodies are served through this source
    // (e.g. positioned reads from the file, for streamed experts) instead of
    // `backing`, which then holds only the header bytes (header + metadata
    // KV + tensor index + padding).
    source:            Option<Arc<dyn TensorBytes>>,
    // split GGUF. `tensor_shard[i]` is which shard
    // `tensors[i]`'s body lives in: 0 for `backing`, otherwise `extra[n - 1]`.
    // A parallel vector rather than a field on `TensorInfo` so the public
    // struct — and every literal that builds one in tests — stays unchanged.
    tensor_shard:      Vec<u32>,
    extra:             Vec<ShardBacking>,
}

impl GgufFile {
    /// Memory-map the file at `path` and parse its header / tensor table.
    ///
    /// if the file declares `split.count > 1` this follows the
    /// sibling shards and presents them as one logical file — `tensors()` spans
    /// all of them and `tensor_by_name` resolves across them. Any shard may be
    /// named; the primary (`split.no == 0`) is always the one whose metadata is
    /// kept, since the others carry only their own `split.*` keys.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_following_splits(path.as_ref(), false)
    }

    /// Parse exactly one file, following no splits.
    fn open_one(path: &Path, streaming: bool) -> Result<GgufInner> {
        let file = File::open(path)?;
        // Safety: GGUF files we read are not concurrently mutated.
        let mmap = unsafe { Mmap::map(&file)? };
        if !streaming {
            return parse(Backing::Mmap(mmap), Bodies::Here);
        }
        // Streaming: keep only the header resident and serve bodies by
        // positioned reads. Each shard gets its own source, whose offsets are
        // relative to that shard own data start.
        let probe = parse(Backing::Mmap(mmap), Bodies::Here)?;
        let start = probe.tensor_data_start;
        // (past the end of a file that stops at its table, or whose alignment is larger than it is: parse has held every
        // tensor to the file, so such a file has no bytes of body to lose)
        let bytes = probe.backing.as_slice();
        let header = bytes[..start.min(bytes.len() as u64) as usize].to_vec();
        drop(probe);
        let source = crate::source::FileSource::open(path, start)?;
        let mut inner = parse(Backing::Bytes(header), Bodies::Elsewhere)?;
        inner.source = Some(Arc::new(source));
        Ok(inner)
    }

    fn open_following_splits(path: &Path, streaming: bool) -> Result<Self> {
        let probe = Self::open_one(path, streaming)?;
        let count = probe
            .metadata
            .get("split.count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if count <= 1 {
            return Ok(Self { inner: Arc::new(probe) });
        }
        let paths = split_shard_paths(path, count as usize).ok_or_else(|| {
            GgufError::Split(format!(
                "{} declares split.count {count} but its name does not follow the \
                 <prefix>-%05d-of-%05d.gguf convention, so the siblings cannot be found",
                path.display()
            ))
        })?;
        let no = probe
            .metadata
            .get("split.no")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        // The primary shard carries the full metadata KV, so it has to be the
        // one we keep even when the caller named a later shard.
        let mut inner = if no == 0 {
            probe
        } else {
            drop(probe);
            Self::open_one(&paths[0], streaming)?
        };

        for sp in paths.iter().skip(1) {
            let shard = Self::open_one(sp, streaming)?;
            inner.merge_shard(shard, sp)?;
        }

        if let Some(total) = inner
            .metadata
            .get("split.tensors.count")
            .and_then(|v| v.as_u64())
        {
            if inner.tensors.len() as u64 != total {
                return Err(GgufError::Split(format!(
                    "split.tensors.count is {total} but {} tensors were found across \
                     {count} shards",
                    inner.tensors.len()
                )));
            }
        }

        Ok(Self { inner: Arc::new(inner) })
    }

    /// Parse from an in-memory buffer. Useful for tests / pipelines.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Backing::Bytes(bytes))
    }

    // everything from here to `tensor_source` is the
    // byte-source seam — see `crate::source`.

    /// Parse `header_bytes` (raw GGUF bytes up to the tensor data section:
    /// header + metadata KV + tensor index, padding optional) and serve
    /// tensor bodies from `source` instead of a file mapping, through the
    /// exact same parsed-metadata view as a mmap'd `.gguf`.
    pub fn from_bytes_with_source(
        header_bytes: Vec<u8>,
        source: Arc<dyn TensorBytes>,
    ) -> Result<Self> {
        let mut inner = parse(Backing::Bytes(header_bytes), Bodies::Elsewhere)?;
        inner.source = Some(source);
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Open the `.gguf` at `path` with tensor bodies read on demand by
    /// positioned reads ([`FileSource`](crate::source::FileSource)) instead
    /// of a mapping: only the header is held in memory. This is how a model
    /// streams its experts from disk.
    pub fn open_streaming(path: impl AsRef<Path>) -> Result<Self> {
        // split-aware, and each shard gets its own FileSource.
        Self::open_following_splits(path.as_ref(), true)
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
        // resolve the shard first; each has its own source.
        match self.shard_source(self.shard_of(t)) {
            Some(src) => Ok(Cow::Owned(src.read_tensor(t)?)),
            None => Ok(Cow::Borrowed(self.tensor_data(t))),
        }
    }

    // shard resolution for a split GGUF. All of these collapse
    // to the single-file behaviour when `extra` is empty.

    /// Which shard holds this tensor body. Looked up by name, which is the
    /// authoritative mapping; a `TensorInfo` from some other file falls back to
    /// the primary shard.
    pub fn shard_of(&self, t: &TensorInfo) -> usize {
        self.inner
            .tensors_by_name
            .get(&t.name)
            .and_then(|&i| self.inner.tensor_shard.get(i))
            .copied()
            .unwrap_or(0) as usize
    }

    /// Number of shards backing this file: 1 unless it is a split GGUF.
    pub fn n_shards(&self) -> usize {
        1 + self.inner.extra.len()
    }

    /// Read ahead `len` bytes at `offset` of shard `shard` (as [`Self::raw_slice_shard`] addresses them), for a copy
    /// about to read them all: a model's weights going to a GPU. Under memory pressure a page-at-a-time read of a
    /// mapped file runs at a fraction of the disk's speed (64 MB/s against the drive's 980 measured on a Mac).
    pub fn will_need_shard(&self, shard: usize, offset: usize, len: usize) {
        let backing = if shard == 0 { &self.inner.backing } else { &self.inner.extra[shard - 1].backing };
        backing.will_need(offset, len);
    }

    fn shard_slice(&self, shard: usize) -> &[u8] {
        if shard == 0 {
            self.inner.backing.as_slice()
        } else {
            self.inner.extra[shard - 1].backing.as_slice()
        }
    }

    fn shard_start(&self, shard: usize) -> u64 {
        if shard == 0 {
            self.inner.tensor_data_start
        } else {
            self.inner.extra[shard - 1].tensor_data_start
        }
    }

    fn shard_source(&self, shard: usize) -> Option<&Arc<dyn TensorBytes>> {
        if shard == 0 {
            self.inner.source.as_ref()
        } else {
            self.inner.extra[shard - 1].source.as_ref()
        }
    }

    /// The byte source for one shard, if that shard is source-backed. On a
    /// split GGUF each shard has its own, so a tensor body must be read through
    /// the source for *its* shard — [`Self::tensor_source`] is only the primary.
    pub fn shard_source_at(&self, shard: usize) -> Option<&Arc<dyn TensorBytes>> {
        self.shard_source(shard)
    }

    /// Whether one shard serves bodies through a source rather than a mapping.
    pub fn shard_is_source_backed(&self, shard: usize) -> bool {
        self.shard_source(shard).is_some()
    }

    /// The source for this tensor own shard.
    pub fn tensor_source_of(&self, t: &TensorInfo) -> Option<&Arc<dyn TensorBytes>> {
        self.shard_source(self.shard_of(t))
    }

    /// Zero-copy slice into a specific shard backing. [`Self::raw_slice`] is
    /// this with `shard = 0`.
    pub fn raw_slice_shard(&self, shard: usize, offset: usize, len: usize) -> &[u8] {
        assert!(
            self.shard_source(shard).is_none(),
            "raw_slice_shard() on a source-backed shard; use tensor_source()"
        );
        &self.shard_slice(shard)[offset..offset + len]
    }

    /// Total length of the backing byte buffer in bytes.
    pub fn raw_len(&self) -> usize {
        self.inner.backing.as_slice().len()
    }

    fn from_backing(backing: Backing) -> Result<Self> {
        let inner = parse(backing, Bodies::Here)?;
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
        // fail loudly instead of slicing past the header.
        assert!(
            self.shard_source(self.shard_of(t)).is_none(),
            "tensor_data() on a source-backed GgufFile; use tensor_bytes()"
        );
        let shard = self.shard_of(t);
        let start = (self.shard_start(shard) + t.offset) as usize;
        let end   = start + t.nbytes() as usize;
        &self.shard_slice(shard)[start..end]
    }

    /// Absolute file offset of `t`'s data body. Combine with [`raw_slice`] to
    /// build long-lived zero-copy views of the mmap (so a clone of the
    /// `GgufFile` can keep the slice alive past a lookup-by-name borrow).
    /// for a split GGUF this is the offset within the tensor
    /// own shard, so it pairs with [`Self::raw_slice_shard`] and
    /// [`Self::shard_of`], not with [`Self::raw_slice`].
    pub fn tensor_data_offset(&self, t: &TensorInfo) -> usize {
        (self.shard_start(self.shard_of(t)) + t.offset) as usize
    }

    /// Zero-copy slice into the file backing. Caller is responsible for the
    /// `(offset, len)` being in bounds — paired with `tensor_data_offset` and
    /// `TensorInfo::nbytes()`. Used by the host-resident-via-mmap weight tier
    /// so the loader can hold an `Arc<GgufFile>` view without copying bytes
    /// into a `Vec<u8>` heap buffer.
    pub fn raw_slice(&self, offset: usize, len: usize) -> &[u8] {
        // same guard as tensor_data — header-only backing.
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

    /// `n` is compared with what is left, not added to `pos`: a length comes from the file, and one near the top of the
    /// address space wraps past the end and passes.
    #[inline]
    fn ensure(&self, n: usize) -> Result<()> {
        let left = self.bytes.len() - self.pos;
        if n > left {
            Err(GgufError::Truncated {
                offset: self.pos as u64,
                needed: (n - left) as u64,
            })
        } else {
            Ok(())
        }
    }

    /// A count from the file, of things that each take `each` bytes at the least, checked against what is left of it.
    /// Call before reserving for them: `Vec::with_capacity` with a count no file could hold does not fail with an
    /// error, it aborts the process.
    fn ensure_count(&self, count: u64, each: usize) -> Result<usize> {
        let left = self.bytes.len() - self.pos;
        if count > (left / each) as u64 {
            return Err(GgufError::Truncated {
                offset: self.pos as u64,
                needed: count.saturating_mul(each as u64).saturating_sub(left as u64),
            });
        }
        Ok(count as usize)
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
        let len = self.read_u64()?;
        let len = self.ensure_count(len, 1)?;
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
        let len = self.read_u64()?;
        // The fewest bytes one element takes (a string's, its length alone; a nested array is refused below).
        let each = match elem_ty {
            ValueType::U8 | ValueType::I8 | ValueType::Bool | ValueType::Array => 1,
            ValueType::U16 | ValueType::I16 => 2,
            ValueType::U32 | ValueType::I32 | ValueType::F32 => 4,
            ValueType::U64 | ValueType::I64 | ValueType::F64 | ValueType::String => 8,
        };
        let len = self.ensure_count(len, each)?;

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

/// Where a file's tensor bodies are, so whether [`parse`] can hold its tensors to them.
#[derive(Clone, Copy, PartialEq)]
enum Bodies {
    /// In the backing, after the header: a whole file.
    Here,
    /// Behind a source, the backing being the header alone (which a streaming open has checked as a whole file first).
    Elsewhere,
}

/// The fewest bytes a tensor's entry in the table takes: a name's length, `n_dims`, the dtype and the offset.
const SMALLEST_TENSOR_ENTRY: usize = 8 + 4 + 4 + 8;

fn parse(backing: Backing, bodies: Bodies) -> Result<GgufInner> {
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
    let mut tensors = Vec::with_capacity(c.ensure_count(tensor_count, SMALLEST_TENSOR_ENTRY)?);
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

        // A shape's size must be a number: callers take `shape` as real, and one whose product wrapped (or a zero
        // dimension among others no tensor could have) sized their buffers by it.
        let info = TensorInfo { name, shape, dtype, offset };
        let numel = match info.checked_nbytes() {
            Some(_) => info.numel(),
            None => return Err(GgufError::TensorSize { name: info.name, shape: info.shape }),
        };

        // Verify block alignment.
        let block = dtype.block_size();
        if block > 1 && numel % (block as u64) != 0 {
            return Err(GgufError::NotBlockAligned {
                name: info.name,
                block,
                numel,
            });
        }

        tensors_by_name.insert(info.name.clone(), i as usize);
        tensors.push(info);
    }

    // Pad to alignment.
    let unaligned = c.pos as u64;
    let tensor_data_start = align_up(unaligned, alignment);

    // Every tensor inside the file: what this says here is what makes `tensor_data` and the other accessors that slice
    // an invariant. Mappings are lazy, so a file cut short used to load and then panic in whichever kernel first
    // touched the missing tensor, in the middle of a request. (A split file comes here a shard at a time, each against
    // its own file.)
    if bodies == Bodies::Here {
        let file_len = bytes.len() as u64;
        for t in &tensors {
            let end = t.checked_nbytes().and_then(|n| t.offset.checked_add(n)).and_then(|n| n.checked_add(tensor_data_start));
            if end.is_none_or(|end| end > file_len) {
                return Err(GgufError::Truncated {
                    offset: file_len,
                    needed: end.unwrap_or(u64::MAX).saturating_sub(file_len),
                });
            }
        }
    }

    // a freshly parsed file is one shard, so every tensor is 0.
    let tensor_shard = vec![0u32; tensors.len()];

    Ok(GgufInner {
        backing,
        version,
        metadata,
        tensors,
        tensors_by_name,
        tensor_data_start,
        alignment,
        source: None, // set by from_bytes_with_source
        tensor_shard,
        extra: Vec::new(),
    })
}

// split-GGUF helpers.

impl GgufInner {
    /// Fold another shard tensor table into this one. Bodies stay in that
    /// shard own backing; only the index is merged.
    fn merge_shard(&mut self, other: GgufInner, path: &Path) -> Result<()> {
        if other.alignment != self.alignment {
            return Err(GgufError::Split(format!(
                "{} has alignment {} but the primary shard has {}",
                path.display(),
                other.alignment,
                self.alignment
            )));
        }
        let idx = (self.extra.len() + 1) as u32;
        for t in other.tensors {
            if self.tensors_by_name.contains_key(&t.name) {
                return Err(GgufError::Split(format!(
                    "tensor {} appears in more than one shard (second in {})",
                    t.name,
                    path.display()
                )));
            }
            self.tensors_by_name.insert(t.name.clone(), self.tensors.len());
            self.tensors.push(t);
            self.tensor_shard.push(idx);
        }
        self.extra.push(ShardBacking {
            backing: other.backing,
            tensor_data_start: other.tensor_data_start,
            source: other.source,
        });
        Ok(())
    }
}

/// Sibling paths for a split GGUF, given any one shard and the shard count.
///
/// llama.cpp names shards `<prefix>-%05d-of-%05d.gguf`, so the prefix is
/// recovered by stripping that suffix. Returns the full set in shard order
/// (index 0 is `-00001-of-`), or `None` if the name does not match, which the
/// caller reports rather than guessing.
pub fn split_shard_paths(path: &Path, count: usize) -> Option<Vec<PathBuf>> {
    if count == 0 {
        return None;
    }
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (prefix, cur) = head.rsplit_once('-')?;
    let five = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !five(total) || !five(cur) {
        return None;
    }
    // Trust the metadata count over the filename, but they should agree.
    if total.parse::<usize>().ok()? != count {
        return None;
    }
    let dir = path.parent()?;
    Some(
        (1..=count)
            .map(|i| dir.join(format!("{prefix}-{i:05}-of-{count:05}.gguf")))
            .collect(),
    )
}

#[inline]
fn align_up(value: u64, align: u64) -> u64 {
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

    /// a streaming open serves the same metadata and tensor
    /// bytes through positioned reads as the mmap does.
    // split-GGUF tests.

    /// Build a two-shard split GGUF on disk and read it back as one file.
    #[test]
    fn split_shards_present_as_one_file() {
        use crate::tensor::GgmlType;

        let dir = std::env::temp_dir().join(format!("gguf_split_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let mk = |no: u32, names: &[&str]| -> Vec<u8> {
            let mut md = BTreeMap::new();
            md.insert("general.architecture".to_string(), Value::String("llama".into()));
            md.insert("split.no".to_string(), Value::U16(no as u16));
            md.insert("split.count".to_string(), Value::U16(2));
            md.insert("split.tensors.count".to_string(), Value::I32(4));
            let mut ts: Vec<(TensorInfo, Vec<u8>)> = Vec::new();
            for (k, n) in names.iter().enumerate() {
                // Distinct bytes per tensor so a mis-attributed shard shows up.
                let body: Vec<u8> = (0..16u32)
                    .map(|i| (i as u8).wrapping_add((no * 100 + k as u32) as u8))
                    .collect();
                ts.push((
                    TensorInfo {
                        name: (*n).to_string(),
                        shape: vec![4],
                        dtype: GgmlType::F32,
                        offset: 0,
                    },
                    body,
                ));
            }
            crate::reader::write_to_vec(&md, &ts, 32).expect("write shard")
        };

        let p0 = dir.join("m-00001-of-00002.gguf");
        let p1 = dir.join("m-00002-of-00002.gguf");
        std::fs::write(&p0, mk(0, &["a", "b"])).expect("write 0");
        std::fs::write(&p1, mk(1, &["c", "d"])).expect("write 1");

        let f = GgufFile::open(&p0).expect("open split");
        assert_eq!(f.n_shards(), 2);
        assert_eq!(f.tensors().len(), 4, "the index spans both shards");
        for n in ["a", "b", "c", "d"] {
            assert!(f.tensor_by_name(n).is_some(), "{n} must resolve");
        }
        // Shard attribution, and bodies read from the right file.
        assert_eq!(f.shard_of(f.tensor_by_name("a").unwrap()), 0);
        assert_eq!(f.shard_of(f.tensor_by_name("d").unwrap()), 1);
        let d = f.tensor_data(f.tensor_by_name("d").unwrap());
        // shard 1, tensor index 1 -> base 101
        assert_eq!(d[0], 101u8, "shard 1 bodies must come from shard 1");
        let a = f.tensor_data(f.tensor_by_name("a").unwrap());
        assert_eq!(a[0], 0u8);

        // Opening a LATER shard must still give the whole model, because the
        // primary carries the metadata.
        let f2 = GgufFile::open(&p1).expect("open from shard 2");
        assert_eq!(f2.tensors().len(), 4);
        assert_eq!(f2.architecture().unwrap(), "llama");

        // Streaming follows splits too, with one source per shard.
        let fs = GgufFile::open_streaming(&p0).expect("open split streaming");
        assert_eq!(fs.n_shards(), 2);
        let td = fs.tensor_bytes(fs.tensor_by_name("d").unwrap()).expect("bytes");
        assert_eq!(td[0], 101u8, "streamed shard-1 body");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A declared split whose siblings cannot be named is an error, not a
    /// silently truncated model.
    #[test]
    fn a_split_with_an_unparseable_name_is_rejected() {
        use crate::tensor::GgmlType;
        let dir = std::env::temp_dir().join(format!("gguf_split_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let mut md = BTreeMap::new();
        md.insert("general.architecture".to_string(), Value::String("llama".into()));
        md.insert("split.count".to_string(), Value::U16(3));
        let ts = vec![(
            TensorInfo {
                name: "a".into(),
                shape: vec![4],
                dtype: GgmlType::F32,
                offset: 0,
            },
            vec![0u8; 16],
        )];
        let raw = crate::reader::write_to_vec(&md, &ts, 32).expect("write");
        let bad = dir.join("not-a-split.gguf");
        std::fs::write(&bad, raw).expect("write bad");

        let err = GgufFile::open(&bad).expect_err("must refuse");
        assert!(
            matches!(err, GgufError::Split(_)),
            "expected a Split error, got {err:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A single-file GGUF keeps exactly its old behaviour.
    #[test]
    fn a_single_file_reports_one_shard() {
        use crate::tensor::GgmlType;
        let mut md = BTreeMap::new();
        md.insert("general.architecture".to_string(), Value::String("llama".into()));
        let ts = vec![(
            TensorInfo {
                name: "a".into(),
                shape: vec![4],
                dtype: GgmlType::F32,
                offset: 0,
            },
            vec![7u8; 16],
        )];
        let raw = crate::reader::write_to_vec(&md, &ts, 32).expect("write");
        let f = GgufFile::from_bytes(raw).expect("parse");
        assert_eq!(f.n_shards(), 1);
        assert_eq!(f.shard_of(f.tensor_by_name("a").unwrap()), 0);
        assert_eq!(f.tensor_data(f.tensor_by_name("a").unwrap())[0], 7u8);
    }

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

    // crafted files: what a count, a length, an offset or a shape says is only a claim.

    /// A file's first bytes: magic, version 3, and its counts of tensors and of metadata keys.
    fn head(tensors: u64, keys: u64) -> Vec<u8> {
        let mut out = GGUF_MAGIC.to_le_bytes().to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&tensors.to_le_bytes());
        out.extend_from_slice(&keys.to_le_bytes());
        out
    }

    /// A tensor's entry in the table, as a file writes it.
    fn entry(out: &mut Vec<u8>, name: &str, shape: &[u64], dtype: GgmlType, offset: u64) {
        write_string(out, name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(dtype as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    /// A metadata key that is an array of `ty` claiming `len` elements, and none of them there.
    fn claimed_array(ty: ValueType, len: u64) -> Vec<u8> {
        let mut out = head(0, 1);
        write_string(&mut out, "last.key");
        out.extend_from_slice(&(ValueType::Array as u32).to_le_bytes());
        out.extend_from_slice(&(ty as u32).to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out
    }

    fn truncated(got: Result<GgufFile>, what: &str) {
        assert!(matches!(got, Err(GgufError::Truncated { .. })), "{what}: {:?}", got.map(|_| "parsed"));
    }

    /// `Vec::with_capacity` of 2^40 bytes aborts the process, and of 2^61 `u64`s panics: the claim is compared with what
    /// is left of the file before anything is reserved.
    #[test]
    fn an_array_longer_than_the_file_is_refused_before_it_is_reserved() {
        truncated(GgufFile::from_bytes(claimed_array(ValueType::U8, 1 << 40)), "u8");
        truncated(GgufFile::from_bytes(claimed_array(ValueType::String, 1 << 40)), "string");
        truncated(GgufFile::from_bytes(claimed_array(ValueType::U64, 1 << 61)), "u64");
        truncated(GgufFile::from_bytes(claimed_array(ValueType::U32, u64::MAX)), "u32");
    }

    /// Every element of a string array takes eight bytes at the least, its length: a list that could fit no more is
    /// refused as a whole, and one that fits is read.
    #[test]
    fn an_array_that_fits_the_file_is_still_read() {
        let mut bytes = claimed_array(ValueType::U16, 2);
        bytes.extend_from_slice(&[1, 0, 2, 0]);
        let f = GgufFile::from_bytes(bytes).expect("two u16s are there");
        assert!(matches!(f.get("last.key").unwrap().as_array(), Some(Array::U16(v)) if v == &vec![1, 2]));
        truncated(GgufFile::from_bytes(claimed_array(ValueType::U16, 1)), "a u16 with no bytes");
        let f = GgufFile::from_bytes(claimed_array(ValueType::F64, 0)).expect("an empty array asks for nothing");
        assert_eq!(f.get("last.key").unwrap().as_array().map(Array::len), Some(0));
    }

    #[test]
    fn a_tensor_count_past_the_file_is_refused_before_it_is_reserved() {
        truncated(GgufFile::from_bytes(head(1 << 36, 0)), "2^36 tensors");
        truncated(GgufFile::from_bytes(head(u64::MAX, 0)), "u64::MAX tensors");
    }

    /// `pos + n` wrapped past the end of the address space, so a length near `u64::MAX` passed the check and the slice
    /// after it panicked.
    #[test]
    fn a_string_length_near_the_top_of_u64_is_truncated_not_a_panic() {
        let mut bytes = head(0, 1);
        bytes.extend_from_slice(&(u64::MAX - 7).to_le_bytes());
        bytes.extend_from_slice(b"not that long");
        truncated(GgufFile::from_bytes(bytes), "a key");
    }

    /// One F32 tensor at `offset`, `body` after the padding.
    fn one_tensor(shape: &[u64], offset: u64, body: &[u8]) -> Vec<u8> {
        let mut out = head(1, 0);
        entry(&mut out, "t", shape, GgmlType::F32, offset);
        out.resize(align_up(out.len() as u64, DEFAULT_ALIGNMENT) as usize, 0);
        out.extend_from_slice(body);
        out
    }

    /// Nothing compared a tensor with the file, so a file cut short loaded and then panicked in the first kernel to
    /// touch that tensor, in the middle of a request.
    #[test]
    fn a_tensor_past_the_end_of_the_file_is_refused_at_open() {
        assert_eq!(GgufFile::from_bytes(one_tensor(&[2], 0, &[0; 8])).unwrap().tensors().len(), 1, "a whole tensor");
        assert_eq!(GgufFile::from_bytes(one_tensor(&[2], 4, &[0; 12])).unwrap().tensors().len(), 1, "one that ends the file");
        truncated(GgufFile::from_bytes(one_tensor(&[2], 0, &[0; 7])), "a byte short");
        truncated(GgufFile::from_bytes(one_tensor(&[2], 4, &[0; 8])), "an offset past its room");
        truncated(GgufFile::from_bytes(one_tensor(&[2], 1 << 40, &[0; 8])), "an offset past the file");
        truncated(GgufFile::from_bytes(one_tensor(&[2], u64::MAX - 3, &[0; 8])), "an offset that wraps");
        truncated(GgufFile::from_bytes(one_tensor(&[1 << 40], 0, &[0; 8])), "a shape past the file");
    }

    /// A shape's product wrapped: `[2^33, 2^33]` was a tensor of no bytes that every caller took for sixteen billion
    /// billion elements.
    #[test]
    fn a_shape_whose_size_does_not_fit_is_refused() {
        for shape in [&[1u64 << 33, 1 << 33][..], &[1 << 62, 4], &[1 << 63, 0, 1 << 63], &[u64::MAX, u64::MAX, u64::MAX, u64::MAX]] {
            let got = GgufFile::from_bytes(one_tensor(shape, 0, &[0; 8]));
            assert!(matches!(got, Err(GgufError::TensorSize { .. })), "{shape:?}: {:?}", got.map(|_| "parsed"));
        }
        assert_eq!(GgufFile::from_bytes(one_tensor(&[0, 3], 0, &[])).unwrap().tensors()[0].nbytes(), 0, "an empty tensor is one");
    }

    #[test]
    fn a_zero_alignment_is_no_alignment() {
        let mut md = BTreeMap::new();
        md.insert("general.alignment".to_string(), Value::U32(0));
        let bytes = write_to_vec(&md, &[], 1).unwrap();
        let len = bytes.len() as u64;
        assert_eq!(GgufFile::from_bytes(bytes).unwrap().tensor_data_start(), len);
    }

    /// A scratch folder of this test's, gone when it is dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!("gguf_{name}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("mkdir");
            Scratch(dir)
        }

        fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).expect("write");
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A streaming open kept `..tensor_data_start` of the file, a start that may lie past its end: a file that stops at
    /// its table, or whose `general.alignment` is larger than it is.
    #[test]
    fn a_streaming_open_keeps_the_header_a_file_has() {
        let dir = Scratch::new("streaming_header");
        let mut key = BTreeMap::new();
        key.insert("general.architecture".to_string(), Value::String("llama".into()));
        let no_padding = write_to_vec(&key, &[], 1).unwrap();
        assert_ne!(no_padding.len() % DEFAULT_ALIGNMENT as usize, 0, "a table that padding would have followed");
        let f = GgufFile::open_streaming(dir.file("no-padding.gguf", &no_padding)).expect("no tensors, no padding");
        assert_eq!(f.architecture().unwrap(), "llama");

        key.insert("general.alignment".to_string(), Value::U64(1 << 40));
        let f = GgufFile::open_streaming(dir.file("alignment.gguf", &write_to_vec(&key, &[], 1).unwrap())).expect("an alignment past the end");
        assert_eq!(f.tensors().len(), 0);

        let mut cut = head(1, 0);
        entry(&mut cut, "t", &[2], GgmlType::F32, 0);
        truncated(GgufFile::open_streaming(dir.file("cut.gguf", &cut)), "streaming, a tensor with no body");
        truncated(GgufFile::open(dir.file("cut2.gguf", &cut)), "mapped, a tensor with no body");
        let mut empty = head(1, 0);
        entry(&mut empty, "t", &[0], GgmlType::F32, 0);
        truncated(GgufFile::open(dir.file("empty.gguf", &empty)), "a tensor whose data section is past the end");
    }

    #[test]
    fn a_bad_tensor_is_refused_by_a_streaming_open_too() {
        let dir = Scratch::new("streaming_tensor");
        truncated(GgufFile::open_streaming(dir.file("short.gguf", &one_tensor(&[2], 0, &[0; 7]))), "a byte short");
        truncated(GgufFile::open_streaming(dir.file("far.gguf", &one_tensor(&[2], 1 << 40, &[0; 8]))), "an offset past the file");
    }

    /// Each shard is held to its own file, the second as much as the first.
    #[test]
    fn a_later_shard_cut_short_is_refused() {
        let dir = Scratch::new("split_short");
        let shard = |no: u16, name: &str| {
            let mut md = BTreeMap::new();
            md.insert("split.no".to_string(), Value::U16(no));
            md.insert("split.count".to_string(), Value::U16(2));
            let t = TensorInfo { name: name.into(), shape: vec![4], dtype: GgmlType::F32, offset: 0 };
            write_to_vec(&md, &[(t, vec![0u8; 16])], 32).unwrap()
        };
        let first = dir.file("m-00001-of-00002.gguf", &shard(0, "a"));
        let mut second = shard(1, "b");
        second.truncate(second.len() - 1);
        dir.file("m-00002-of-00002.gguf", &second);
        truncated(GgufFile::open(&first), "mapped");
        truncated(GgufFile::open_streaming(&first), "streaming");
    }

    /// A bad tensor's bytes were allocated before they were checked: the abort of a count past the file, not the
    /// `Truncated` two lines on.
    #[test]
    fn a_source_refuses_a_read_past_the_file_before_allocating_it() {
        use crate::source::FileSource;
        let dir = Scratch::new("source_read");
        let src = FileSource::open(dir.file("body.bin", &[7u8; 64]), 32).unwrap();
        let huge = TensorInfo { name: "t".into(), shape: vec![1 << 42], dtype: GgmlType::F32, offset: 0 };
        assert!(matches!(src.read_tensor(&huge), Err(GgufError::Truncated { .. })));
        let wrapped = TensorInfo { name: "t".into(), shape: vec![1 << 33, 1 << 33], dtype: GgmlType::F32, offset: 0 };
        assert!(matches!(src.read_tensor(&wrapped), Err(GgufError::Truncated { .. })));
        let far = TensorInfo { name: "t".into(), shape: vec![2], dtype: GgmlType::F32, offset: u64::MAX - 3 };
        assert!(matches!(src.read_tensor(&far), Err(GgufError::Truncated { .. })));
        assert!(matches!(src.read_range(u64::MAX - 3, &mut [0u8; 8]), Err(GgufError::Truncated { .. })));
        let fits = TensorInfo { name: "t".into(), shape: vec![2], dtype: GgmlType::F32, offset: 24 };
        assert_eq!(src.read_tensor(&fits).unwrap(), vec![7u8; 8]);
    }

    #[test]
    fn a_mapped_source_refuses_a_range_that_wraps() {
        use crate::source::GgufSource;
        let src = GgufSource(GgufFile::from_bytes(make_simple_gguf()).unwrap());
        assert!(matches!(src.read_range(u64::MAX - 3, &mut [0u8; 8]), Err(GgufError::Truncated { .. })));
    }
}
