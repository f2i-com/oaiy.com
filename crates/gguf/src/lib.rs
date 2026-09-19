//! Pure-Rust GGUF v2 / v3 reader.
//!
//! GGUF is the container format used by llama.cpp. This crate implements just the
//! container — it does not interpret tensor *contents* (that lives in
//! `ggml-quants` and `ggml-rs`).
//!
//! Layout (little-endian):
//!
//! ```text
//!   magic          u32        "GGUF"
//!   version        u32        2 or 3
//!   tensor_count   u64
//!   kv_count       u64
//!   kv_pairs       kv_count of (string, value_type, value)
//!   tensor_infos   tensor_count of (string, n_dims:u32, dims:[u64;n_dims], ty:u32, offset:u64)
//!   padding        zero-bytes up to `general.alignment` (default 32) past the tensor info table
//!   tensor_data    contiguous; tensor i lives at tensor_data_start + tensor_infos[i].offset
//! ```

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]
// Quant variant names like `Q4_0`, `IQ2_XXS` match upstream ggml naming.
#![allow(non_camel_case_types)]

pub mod error;
pub mod reader;
// VENDORED-LOCAL: byte-source seam (see source.rs); additive public API.
pub mod source;
pub mod tensor;
pub mod value;

pub use error::{GgufError, Result};
pub use reader::GgufFile;
pub use source::{FileSource, GgufSource, TensorBytes};
pub use tensor::{GgmlType, TensorInfo};
pub use value::{Array, Value, ValueType};

/// Magic number: ASCII "GGUF" little-endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default alignment for tensor data (overridable via `general.alignment` KV).
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Supported GGUF versions.
pub const SUPPORTED_VERSIONS: &[u32] = &[2, 3];
