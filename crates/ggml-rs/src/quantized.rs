//! Tensors that hold *packed* quantized data (Q4_K, Q8_0, etc.) rather than
//! F32 floats. Used as weight storage so we don't pay the 4–8× memory bloat
//! of dequantizing at load time.
//!
//! `QuantizedTensor` mirrors `Tensor` but its storage is a flat byte buffer
//! whose length matches the GGML block layout: `n_blocks * type_size` bytes,
//! where `n_blocks = numel / block_size`.

use std::any::Any;
use std::fmt;

use ggml_quants::GgmlType;

/// Backend-provided device storage for packed quant bytes (analog of
/// `DeviceStorage` for F32 tensors).
pub trait QuantizedDeviceStorage: fmt::Debug + Send + Sync + 'static {
    /// Number of bytes occupied by this storage on the device.
    fn nbytes(&self) -> usize;

    fn dtype(&self) -> GgmlType;

    fn device_name(&self) -> &str;

    /// Materialize a host copy of the raw bytes.
    fn copy_to_host(&self) -> Vec<u8>;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn clone_to_device(&self) -> Box<dyn QuantizedDeviceStorage>;
}

/// Host-resident byte source backed by something that owns the underlying
/// allocation (typically an `Arc<GgufFile>` clone holding the mmap alive).
/// Used by [`QuantizedStorage::Mmap`] so weights that don't fit in VRAM can
/// stay as zero-copy views into the .gguf file — OS page cache handles
/// eviction to disk transparently, no `.to_vec()` doubling at load time.
pub trait QuantizedHostBytes: Send + Sync + 'static {
    fn as_bytes(&self) -> &[u8];
}

impl<T> QuantizedHostBytes for T
where T: AsRef<[u8]> + Send + Sync + 'static
{
    fn as_bytes(&self) -> &[u8] { self.as_ref() }
}

pub enum QuantizedStorage {
    Cpu(Vec<u8>),
    /// Zero-copy host view (e.g. into a memory-mapped GGUF file). The Arc
    /// keeps the backing alive for the storage's lifetime.
    Mmap(std::sync::Arc<dyn QuantizedHostBytes>),
    Device(Box<dyn QuantizedDeviceStorage>),
}

impl fmt::Debug for QuantizedStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(v) => write!(f, "Cpu({} bytes)", v.len()),
            Self::Mmap(s) => write!(f, "Mmap({} bytes)", s.as_bytes().len()),
            Self::Device(s) => write!(f, "Device:{}({} bytes)", s.device_name(), s.nbytes()),
        }
    }
}

/// Packed-weight tensor. Logical shape is preserved (e.g. `[out, in]`) but
/// the underlying storage is the GGML block-quant byte layout for `dtype`.
pub struct QuantizedTensor {
    storage: QuantizedStorage,
    shape:   Vec<usize>,
    dtype:   GgmlType,
}

impl QuantizedTensor {
    pub fn from_bytes_cpu(bytes: Vec<u8>, shape: Vec<usize>, dtype: GgmlType) -> Self {
        let numel: usize = shape.iter().product();
        debug_assert_eq!(numel % dtype.block_size(), 0);
        let expected = (numel / dtype.block_size()) * dtype.type_size();
        debug_assert_eq!(bytes.len(), expected,
                         "QuantizedTensor::from_bytes_cpu: byte length {} != expected {}",
                         bytes.len(), expected);
        Self { storage: QuantizedStorage::Cpu(bytes), shape, dtype }
    }

    pub fn from_device(storage: Box<dyn QuantizedDeviceStorage>, shape: Vec<usize>) -> Self {
        let dtype = storage.dtype();
        let numel: usize = shape.iter().product();
        let expected = (numel / dtype.block_size()) * dtype.type_size();
        debug_assert_eq!(storage.nbytes(), expected);
        Self {
            storage: QuantizedStorage::Device(storage),
            shape,
            dtype,
        }
    }

    /// Build from a host byte source that's already in memory but owned by
    /// something we don't want to clone (e.g. an `Arc<GgufFile>` view into a
    /// memory-mapped file). Cheaper than [`from_bytes_cpu`] — no `.to_vec()` —
    /// and the OS page cache becomes the SSD-tier fallback when host RAM is
    /// also tight.
    pub fn from_mmap(
        bytes: std::sync::Arc<dyn QuantizedHostBytes>,
        shape: Vec<usize>,
        dtype: GgmlType,
    ) -> Self {
        let numel: usize = shape.iter().product();
        let expected = (numel / dtype.block_size()) * dtype.type_size();
        debug_assert_eq!(bytes.as_bytes().len(), expected,
                         "from_mmap: byte length {} != expected {}",
                         bytes.as_bytes().len(), expected);
        Self { storage: QuantizedStorage::Mmap(bytes), shape, dtype }
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn dim(&self, axis: usize) -> usize { self.shape[axis] }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
    pub fn dtype(&self) -> GgmlType { self.dtype }

    pub fn nbytes(&self) -> usize {
        match &self.storage {
            QuantizedStorage::Cpu(v) => v.len(),
            QuantizedStorage::Mmap(s) => s.as_bytes().len(),
            QuantizedStorage::Device(s) => s.nbytes(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.storage, QuantizedStorage::Cpu(_) | QuantizedStorage::Mmap(_))
    }
    pub fn is_device(&self) -> bool { matches!(self.storage, QuantizedStorage::Device(_)) }

    pub fn device_name(&self) -> &str {
        match &self.storage {
            QuantizedStorage::Cpu(_) => "cpu",
            QuantizedStorage::Mmap(_) => "mmap",
            QuantizedStorage::Device(s) => s.device_name(),
        }
    }

    /// Direct host access to the packed bytes. Panics if storage is on a device.
    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            QuantizedStorage::Cpu(v) => v,
            QuantizedStorage::Mmap(s) => s.as_bytes(),
            QuantizedStorage::Device(s) => panic!(
                "QuantizedTensor::bytes() on {} tensor (use to_host() first)",
                s.device_name()
            ),
        }
    }

    pub fn to_host(&self) -> QuantizedTensor {
        let bytes = match &self.storage {
            QuantizedStorage::Cpu(v) => v.clone(),
            QuantizedStorage::Mmap(s) => s.as_bytes().to_vec(),
            QuantizedStorage::Device(s) => s.copy_to_host(),
        };
        QuantizedTensor::from_bytes_cpu(bytes, self.shape.clone(), self.dtype)
    }

    pub fn device_storage(&self) -> Option<&dyn QuantizedDeviceStorage> {
        match &self.storage {
            QuantizedStorage::Device(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn into_storage(self) -> QuantizedStorage { self.storage }
}

impl Clone for QuantizedTensor {
    fn clone(&self) -> Self {
        match &self.storage {
            QuantizedStorage::Cpu(v) => QuantizedTensor::from_bytes_cpu(
                v.clone(), self.shape.clone(), self.dtype),
            QuantizedStorage::Mmap(s) => QuantizedTensor::from_mmap(
                s.clone(), self.shape.clone(), self.dtype),
            QuantizedStorage::Device(s) => QuantizedTensor::from_device(
                s.clone_to_device(), self.shape.clone()),
        }
    }
}

impl fmt::Debug for QuantizedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantizedTensor({}, dtype={:?}, shape={:?}, nbytes={})",
               self.device_name(), self.dtype, self.shape, self.nbytes())
    }
}
