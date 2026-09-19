use thiserror::Error;

use crate::dtype::GgmlType;

pub type Result<T> = std::result::Result<T, QuantError>;

#[derive(Debug, Error)]
pub enum QuantError {
    #[error("unsupported dtype for dequantization: {0:?}")]
    Unsupported(GgmlType),

    #[error("element count {n} not a multiple of block size {block}")]
    NotBlockAligned { n: usize, block: usize },

    #[error("source byte length {got} != expected {expected}")]
    WrongLength { got: usize, expected: usize },
}
