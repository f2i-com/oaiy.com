//! A quantized weight on the device: its buffers, and what `ggml-rs` asks of a device's storage.


// The quantized matmul's WGSL.
pub mod shaders;
// A GGUF's matrix where a chain takes a packed projection.
pub mod linear;
// A MoE layer's experts as a GGUF holds them, on the card, and on the host for the layers no card has room for.
pub mod moe;
pub mod host;
// Weights past the GPU's budget, sent to the card as they are used.
pub(crate) mod stream;

use super::*;

/// A weight matrix on the GPU: its rows in one or more buffers.
pub(super) struct WgpuQuant {
    pub(super) gpu: Arc<Gpu>,
    pub(super) dtype: GgmlType,
    /// `(buffer, first_row, rows)`; each buffer holds whole rows.
    pub(super) chunks: Vec<(wgpu::Buffer, u32, u32)>,
    pub(super) row_bytes: usize,
    pub(super) nbytes: usize,
    pub(super) used: Arc<AtomicU64>,
    /// Its chunks are in the host's memory, which the card reads over the bus ([`crate::quant::stream`]).
    pub(super) host: bool,
    /// Its bytes are in the model's file (no chunks): written into the card's slots each time a matmul takes it
    /// ([`crate::quant::stream`]); the blocks' padding on the card, where its type has one (`shaders::padded_block`).
    pub(super) file: Option<(QuantizedTensor, Option<(usize, usize, usize)>)>,
}

impl std::fmt::Debug for WgpuQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuQuant({:?}, {} bytes, {} chunk(s))", self.dtype, self.nbytes, self.chunks.len())
    }
}

impl Drop for WgpuQuant {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes as u64, Ordering::Relaxed);
    }
}

impl WgpuQuant {
    /// The weights as the GPU holds them.
    pub(super) fn gpu_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nbytes);
        for (buffer, _, rows) in &self.chunks {
            let len = *rows as usize * self.row_bytes;
            out.extend_from_slice(&self.gpu.read(buffer, len as u64)[..len]);
        }
        out
    }
}

impl QuantizedDeviceStorage for WgpuQuant {
    fn nbytes(&self) -> usize {
        self.nbytes
    }
    fn dtype(&self) -> GgmlType {
        self.dtype
    }
    fn device_name(&self) -> &str {
        "webgpu"
    }
    fn copy_to_host(&self) -> Vec<u8> {
        let out = self.gpu_bytes();
        // ggml's layout, where the GPU's blocks are padded
        match shaders::padded_block(self.dtype) {
            Some((host, gpu, at)) => shaders::pad_blocks(&out, gpu, host, at),
            None => out,
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_to_device(&self) -> Box<dyn QuantizedDeviceStorage> {
        let bytes = match &self.file {
            Some((w, padded)) => crate::quant::stream::card_bytes(w, *padded).into_owned(),
            None => self.gpu_bytes(),
        };
        self.used.fetch_add(self.nbytes as u64, Ordering::Relaxed);
        Box::new(WgpuQuant {
            gpu: Arc::clone(&self.gpu),
            dtype: self.dtype,
            chunks: self.gpu.upload_rows(&bytes, self.row_bytes, self.chunks.len()),
            row_bytes: self.row_bytes,
            nbytes: self.nbytes,
            used: Arc::clone(&self.used),
            host: false,
            file: None,
        })
    }
}
