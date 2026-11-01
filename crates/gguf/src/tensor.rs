//! Per-tensor metadata. The dtype tag itself lives in `ggml-quants` because
//! quantization owns the layout semantics; we just re-export here.

pub use ggml_quants::GgmlType;

/// Per-tensor metadata, parsed from the GGUF tensor info table.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,

    /// Logical shape, *as stored* in GGUF. Note GGUF dimensions are stored
    /// fastest-varying-first (the mirror of NumPy / PyTorch convention).
    pub shape: Vec<u64>,

    pub dtype: GgmlType,

    /// Offset *within the tensor data section* (not the file). Caller adds
    /// `GgufFile::tensor_data_start` to get the absolute file offset.
    pub offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 { self.shape.iter().product() }

    pub fn nbytes(&self) -> u64 {
        let n = self.numel();
        let block = self.dtype.block_size() as u64;
        let ts = self.dtype.type_size() as u64;
        (n / block) * ts
    }

    /// [`Self::nbytes`] where a shape from a file may be anything: `None` if the size does not fit a `u64`, its zero
    /// dimensions set aside (a `[2^63, 0, 2^63]` holds no bytes, but its neighbours are dimensions no caller can use).
    pub fn checked_nbytes(&self) -> Option<u64> {
        let n = self.shape.iter().filter(|&&d| d != 0).try_fold(1u64, |n, &d| n.checked_mul(d))?;
        let n = if self.shape.contains(&0) { 0 } else { n };
        (n / self.dtype.block_size() as u64).checked_mul(self.dtype.type_size() as u64)
    }
}
