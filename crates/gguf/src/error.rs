use std::io;

use ggml_quants::dtype::UnknownDtype;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("not a GGUF file: bad magic 0x{0:08x}")]
    BadMagic(u32),

    #[error("unsupported GGUF version {0} (supported: {1:?})")]
    UnsupportedVersion(u32, &'static [u32]),

    #[error("unknown ValueType tag {0}")]
    UnknownValueType(u32),

    #[error("unknown ggml dtype tag {0}")]
    UnknownGgmlType(u32),

    #[error(transparent)]
    UnknownDtype(#[from] UnknownDtype),

    #[error("string was not valid UTF-8")]
    BadUtf8(#[from] std::string::FromUtf8Error),

    #[error("expected metadata key `{0}` was not found")]
    MissingKey(String),

    #[error("metadata key `{key}` had wrong type: expected {expected}, got {actual}")]
    TypeMismatch {
        key: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("file truncated: needed {needed} more bytes at offset {offset}")]
    Truncated { offset: u64, needed: u64 },

    #[error("nested arrays of arrays are not supported (key `{0}`)")]
    NestedArray(String),

    #[error("tensor `{name}` has unsupported {n_dims} dimensions (max 4)")]
    TooManyDims { name: String, n_dims: u32 },

    #[error("tensor `{name}` length not a multiple of block size {block} (numel = {numel})")]
    NotBlockAligned {
        name: String,
        block: usize,
        numel: u64,
    },

    // VENDORED-LOCAL: error from an external `TensorBytes` source.
    #[error("tensor byte source error: {0}")]
    Source(String),
}
