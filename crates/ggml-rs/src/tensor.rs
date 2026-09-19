//! Owned tensor with pluggable storage (CPU `Vec<f32>` or device-resident
//! buffer provided by a GPU backend).

use std::any::Any;
use std::fmt;

use thiserror::Error;

pub type Shape = Vec<usize>;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Shape, got: Shape },

    #[error("can't reshape numel={numel} into {shape:?}")]
    BadReshape { numel: usize, shape: Shape },

    #[error("rank {rank} is too small for axis {axis}")]
    BadAxis { rank: usize, axis: usize },

    #[error("incompatible matmul shapes: a={a:?}, b={b:?}")]
    BadMatmulShape { a: Shape, b: Shape },

    #[error("expected tensor on `{expected}`, got `{got}`")]
    WrongDevice { expected: String, got: String },
}

/// Backend-provided device storage. Ownership of the underlying buffer lives
/// inside the trait object; the backend (e.g. `ggml-rs-cuda`'s `CudaStorage`)
/// downcasts to its concrete type via `as_any` / `as_any_mut` to access it.
pub trait DeviceStorage: fmt::Debug + Send + Sync + 'static {
    fn len(&self) -> usize;

    /// Short device label, e.g. `"cuda:0"`. Used for diagnostics.
    fn device_name(&self) -> &str;

    /// Materialize a host copy. Allocates.
    fn copy_to_host(&self) -> Vec<f32>;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Deep-copy on device. Used by `Tensor::clone()` so the original keeps
    /// independent ownership of its buffer.
    fn clone_to_device(&self) -> Box<dyn DeviceStorage>;
}

/// Underlying storage for a tensor.
pub enum TensorStorage {
    Cpu(Vec<f32>),
    Device(Box<dyn DeviceStorage>),
}

impl fmt::Debug for TensorStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(v) => write!(f, "Cpu({} elem)", v.len()),
            Self::Device(s) => write!(f, "Device:{}({} elem)", s.device_name(), s.len()),
        }
    }
}

/// Row-major tensor. Data may live on the host (`Vec<f32>`) or on a device
/// (e.g. CUDA). Use [`Tensor::data`] / [`Tensor::data_mut`] for direct host
/// access; [`Tensor::to_host`] downloads from a device. Backends that operate
/// on device-resident data downcast through [`DeviceStorage::as_any`].
pub struct Tensor {
    storage: TensorStorage,
    shape:   Shape,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        let n: usize = shape.iter().product();
        debug_assert_eq!(n, data.len(), "Tensor::from_vec: numel != data.len()");
        Self { storage: TensorStorage::Cpu(data), shape }
    }

    pub fn from_device(storage: Box<dyn DeviceStorage>, shape: Shape) -> Self {
        let n: usize = shape.iter().product();
        debug_assert_eq!(n, storage.len(), "Tensor::from_device: numel != storage.len()");
        Self { storage: TensorStorage::Device(storage), shape }
    }

    pub fn zeros(shape: Shape) -> Self {
        let n = shape.iter().product();
        Self { storage: TensorStorage::Cpu(vec![0.0; n]), shape }
    }

    pub fn ones(shape: Shape) -> Self {
        let n = shape.iter().product();
        Self { storage: TensorStorage::Cpu(vec![1.0; n]), shape }
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn rank(&self) -> usize { self.shape.len() }

    pub fn dim(&self, axis: usize) -> usize {
        assert!(axis < self.shape.len(), "dim: axis {axis} >= rank {}", self.shape.len());
        self.shape[axis]
    }

    pub fn numel(&self) -> usize { self.shape.iter().product() }

    pub fn is_cpu(&self) -> bool { matches!(self.storage, TensorStorage::Cpu(_)) }
    pub fn is_device(&self) -> bool { matches!(self.storage, TensorStorage::Device(_)) }

    pub fn device_name(&self) -> &str {
        match &self.storage {
            TensorStorage::Cpu(_) => "cpu",
            TensorStorage::Device(s) => s.device_name(),
        }
    }

    /// Direct host access. **Panics** if storage is on a device — call
    /// `to_host()` first or use the backend's device-aware ops.
    pub fn data(&self) -> &[f32] {
        match &self.storage {
            TensorStorage::Cpu(v) => v,
            TensorStorage::Device(s) => panic!(
                "Tensor::data() on {} tensor (use to_host() first)",
                s.device_name()
            ),
        }
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            TensorStorage::Cpu(v) => v,
            TensorStorage::Device(s) => panic!(
                "Tensor::data_mut() on {} tensor (use to_host() first)",
                s.device_name()
            ),
        }
    }

    /// Get a host copy of the tensor data, transferring from device if needed.
    pub fn to_host(&self) -> Tensor {
        let data = match &self.storage {
            TensorStorage::Cpu(v) => v.clone(),
            TensorStorage::Device(s) => s.copy_to_host(),
        };
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Borrow the device storage, if any. Used by GPU backends to downcast to
    /// their concrete storage type.
    pub fn device_storage(&self) -> Option<&dyn DeviceStorage> {
        match &self.storage {
            TensorStorage::Device(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn device_storage_mut(&mut self) -> Option<&mut dyn DeviceStorage> {
        match &mut self.storage {
            TensorStorage::Device(s) => Some(s.as_mut()),
            _ => None,
        }
    }

    /// Replace the storage entirely. Caller must ensure the new storage matches
    /// the tensor's `numel()`.
    pub fn replace_storage(&mut self, storage: TensorStorage) {
        match &storage {
            TensorStorage::Cpu(v) => debug_assert_eq!(v.len(), self.numel()),
            TensorStorage::Device(s) => debug_assert_eq!(s.len(), self.numel()),
        }
        self.storage = storage;
    }

    pub fn into_storage(self) -> TensorStorage { self.storage }

    /// Reshape (no copy). New shape must have the same number of elements.
    pub fn reshape(mut self, shape: Shape) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        if n != self.numel() {
            return Err(TensorError::BadReshape { numel: self.numel(), shape });
        }
        self.shape = shape;
        Ok(self)
    }

    /// Per-row strides for a row-major contiguous tensor: `stride[i] = product of dims[i+1..]`.
    pub fn strides(&self) -> Vec<usize> {
        let r = self.shape.len();
        let mut strides = vec![1usize; r];
        for i in (0..r.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    /// View shape as 2D: `(rows, cols)` where cols is the last axis.
    pub fn as_2d(&self) -> (usize, usize) {
        assert!(!self.shape.is_empty(), "as_2d: 0-rank tensor");
        let cols = *self.shape.last().unwrap();
        let rows = self.numel() / cols;
        (rows, cols)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        match &self.storage {
            TensorStorage::Cpu(v) => Tensor::from_vec(v.clone(), self.shape.clone()),
            TensorStorage::Device(s) => Tensor::from_device(s.clone_to_device(), self.shape.clone()),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({}, shape={:?}, numel={}", self.device_name(), self.shape, self.numel())?;
        if self.is_cpu() && self.numel() <= 8 {
            write!(f, ", data={:?}", self.data())?;
        }
        write!(f, ")")
    }
}
