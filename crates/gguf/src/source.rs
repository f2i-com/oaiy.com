//! VENDORED-LOCAL: pluggable tensor-byte source seam (nrob streaming).
//!
//! The classic [`GgufFile`] path memory-maps a `.gguf` file and lends out
//! `&[u8]` slices into the mapping. A model whose expert weights are
//! *streamed* instead reads tensor bodies on demand through this trait:
//! [`FileSource`] serves them with positioned reads straight from the
//! `.gguf` file, so nothing beyond what a caller asks for is brought into
//! memory (the header is always parsed from an in-memory buffer).
//! Everything returns owned `Vec<u8>` / fills caller buffers; the mmap
//! implementation ([`GgufSource`]) is a plain copy. Callers that want
//! zero-copy slices can still use [`GgufFile::tensor_data`] when
//! [`GgufFile::tensor_source`] is `None`.

use std::fs::File;
use std::path::Path;

use crate::error::{GgufError, Result};
use crate::reader::GgufFile;
use crate::tensor::TensorInfo;

/// A source of raw tensor body bytes for a parsed GGUF.
///
/// Offsets in [`TensorBytes::read_range`] are relative to the start of the
/// GGUF tensor data section (the same frame as [`TensorInfo::offset`]), so a
/// caller can translate an absolute file offset by subtracting
/// [`GgufFile::tensor_data_start`].
pub trait TensorBytes: Send + Sync + std::fmt::Debug {
    /// Owned copy of one tensor's raw (still ggml-quantized) body bytes.
    /// Implementations should verify the length matches `info.nbytes()`.
    fn read_tensor(&self, info: &TensorInfo) -> Result<Vec<u8>>;

    /// Fill `dst` with data-section bytes `[data_offset, data_offset + dst.len())`.
    /// Used for ranged streaming of a slice of a (stacked) tensor without
    /// materializing the whole object.
    fn read_range(&self, data_offset: u64, dst: &mut [u8]) -> Result<()>;
}

/// The classic mmap-backed source: wraps a [`GgufFile`] and copies out of
/// the mapping. Byte-identical to slicing the mmap directly, just owned.
#[derive(Debug, Clone)]
pub struct GgufSource(pub GgufFile);

impl TensorBytes for GgufSource {
    fn read_tensor(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        Ok(self.0.tensor_data(info).to_vec())
    }

    fn read_range(&self, data_offset: u64, dst: &mut [u8]) -> Result<()> {
        let start = self.0.tensor_data_start() + data_offset;
        let end = start + dst.len() as u64;
        let file_len = self.0.raw_len() as u64;
        if end > file_len {
            return Err(GgufError::Truncated {
                offset: file_len,
                needed: end - file_len,
            });
        }
        dst.copy_from_slice(self.0.raw_slice(start as usize, dst.len()));
        Ok(())
    }
}

/// Tensor bodies read on demand from the `.gguf` file itself, with
/// positioned reads (no mapping, no shared file cursor, so concurrent
/// readers never interfere). This is what streams a model's experts:
/// only the bytes a caller asks for are ever read.
#[derive(Debug)]
pub struct FileSource {
    file: File,
    /// Absolute file offset of the tensor data section.
    data_start: u64,
    /// File length, for a clean error on reads past the end.
    len: u64,
}

impl FileSource {
    /// Serve the tensor data section of the GGUF at `path`, which starts
    /// at absolute offset `data_start` ([`GgufFile::tensor_data_start`]).
    pub fn open(path: impl AsRef<Path>, data_start: u64) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let len = file.metadata()?.len();
        Ok(Self { file, data_start, len })
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let end = offset + dst.len() as u64;
        if end > self.len {
            return Err(GgufError::Truncated {
                offset: self.len,
                needed: end - self.len,
            });
        }
        read_exact_at(&self.file, dst, offset)?;
        Ok(())
    }
}

impl TensorBytes for FileSource {
    fn read_tensor(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let mut out = vec![0u8; info.nbytes() as usize];
        self.read_at(self.data_start + info.offset, &mut out)?;
        Ok(out)
    }

    fn read_range(&self, data_offset: u64, dst: &mut [u8]) -> Result<()> {
        self.read_at(self.data_start + data_offset, dst)
    }
}

/// `pread`-style read of exactly `dst.len()` bytes at `offset`.
#[cfg(unix)]
fn read_exact_at(file: &File, dst: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dst, offset)
}

/// `pread`-style read of exactly `dst.len()` bytes at `offset`.
#[cfg(windows)]
fn read_exact_at(file: &File, mut dst: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !dst.is_empty() {
        match file.seek_read(dst, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                dst = &mut dst[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
