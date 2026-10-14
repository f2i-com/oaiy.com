//! Weights past the GPU's budget, sent to the card as they are used: a model larger than the card. A weight the
//! budget has no room for stays in the model's file and is held as the card's ([`crate::WgpuQuant`] with `file`), so
//! a model's chained runs take it with the rest; each time a matmul takes it, the work recorded so far is sent and its
//! bytes are written into one of two sets of slots on the card ([`Recorder::stage`]), through the host's memory as
//! any large write is. The drive, then the system's memory (what of the file it keeps), then the card: each byte over
//! the bus once a step and once a prompt's chunk. Before this, such a weight stayed in the file for the CPU, and the
//! GPU's chained runs did not take a model that had one: Qwen3.5 9B held to a 3 GB budget read a 300-token prompt for
//! over ten minutes on an NVIDIA A2 (Thunderbolt 3); its 1.9 GB past the budget now take it 1.8 s, and its reply is
//! 1.15 tokens a second (the bus's 2.2 GB a second; the CPU alone replies at 0.36).
//!
//! With OAIY_STREAM_PIN=1 such a weight goes in the host's memory instead, in a buffer the card's kernels read where
//! it lies (a step's rows over the bus; a prompt's rows copied into the slots by a kernel, in the recording's pass):
//! as fast for a prompt, slower for a step (0.95 tokens a second there), and it holds the system's memory.
use super::*;
use crate::chain::{grid, Recorder};

/// Rows of a matmul from which a weight in the host's memory is copied into the card first: below, read in place.
pub(crate) const STAGE_ROWS: usize = 16;

/// Whether weights past the budget go in the host's memory for the card to read (OAIY_STREAM_WEIGHTS=0: not; they
/// stay in the file for the CPU, as before).
pub(crate) fn streaming_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OAIY_STREAM_WEIGHTS").map_or(true, |v| v != "0"))
}

/// The host's memory to leave free besides what the weights there take: 2 GiB, or a sixth of it where that is more.
pub(crate) fn host_reserve(total: usize) -> usize {
    (2usize << 30).max(total / 6)
}

/// Two sets of buffers on the card that weights from the host's memory or a file are copied into, in turn (each
/// matmul's into the set the matmul before it did not read), a buffer each of a weight's chunks, grown to the
/// largest.
#[derive(Default)]
pub(crate) struct Slots {
    bufs: [Vec<(wgpu::Buffer, u64)>; 2],
    next: usize,
}

/// A copy of `p[0].x` words, a thread four of them ([`Recorder::stage`]).
const COPY_WORDS: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = (id.x + id.y * 16776960u) * 4u;
    for (var j = 0u; j < 4u; j++) {
        let i = base + j;
        if (i < p[0].x) { dst[i] = src[i]; }
    }
}
"#;

impl Gpu {
    /// `bytes` in whole rows of `row_bytes` as buffers in the HOST's memory below the binding limit, as
    /// [`Self::upload_rows`] lays them out on the card: written through their mapping, none of them over the bus.
    /// None where the device has no such buffers.
    pub(crate) fn host_rows(&self, bytes: &[u8], row_bytes: usize) -> Option<Vec<(wgpu::Buffer, u32, u32)>> {
        if !self.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
            return None;
        }
        let rows = bytes.len() / row_bytes;
        let per = ((chunk_limit(&self.limits) as usize) / row_bytes).max(1);
        let mut chunks = Vec::new();
        let mut r = 0;
        while r < rows {
            let n = per.min(rows - r);
            let data = &bytes[r * row_bytes..(r + n) * row_bytes];
            let size = (data.len() as u64).div_ceil(4) * 4;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-host-weight"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                let mut view = buffer.slice(..).get_mapped_range_mut().ok()?;
                let mut whole = view.slice(..);
                // SAFETY: the buffer's `size` bytes, mapped at creation until the view is dropped; nothing reads them yet
                unsafe {
                    let dst = whole.as_raw_ptr().cast::<u8>().as_ptr();
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                    std::ptr::write_bytes(dst.add(data.len()), 0, size as usize - data.len());
                }
            }
            buffer.unmap();
            chunks.push((buffer, r as u32, n as u32));
            r += n;
        }
        Some(chunks)
    }

    /// The next set of slots, a buffer for each of `sizes` (a weight's chunks' bytes).
    pub(crate) fn stream_slots(&self, sizes: &[u64]) -> Vec<wgpu::Buffer> {
        let mut s = self.stream_slots.lock().unwrap_or_else(|p| p.into_inner());
        let k = s.next;
        s.next = (k + 1) % 2;
        let set = &mut s.bufs[k];
        while set.len() < sizes.len() {
            set.push((self.dummy_rw().clone(), 0));
        }
        sizes
            .iter()
            .zip(set.iter_mut())
            .map(|(&bytes, slot)| {
                if slot.1 < bytes {
                    let size = bytes.div_ceil(1 << 20) << 20;
                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-stream-slot"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    *slot = (buffer, size);
                }
                slot.0.clone()
            })
            .collect()
    }
}

/// `w`'s bytes as the card lays them (its blocks padded where `padded` says, `shaders::pad_blocks`).
pub(crate) fn card_bytes(w: &QuantizedTensor, padded: Option<(usize, usize, usize)>) -> std::borrow::Cow<'_, [u8]> {
    match padded {
        Some((host, gpu, at)) => std::borrow::Cow::Owned(shaders::pad_blocks(w.bytes(), host, gpu, at)),
        None => std::borrow::Cow::Borrowed(w.bytes()),
    }
}

impl WgpuBackend {
    /// `w` (whole rows of `row_bytes` as the card lays them, `bytes` of them) in the host's memory for the card to read,
    /// where the budget has no room for it on the card and the host's memory has (its reserve kept): counted as
    /// [`Self::streamed`]. None otherwise (it stays in the file, for the CPU).
    pub(crate) fn upload_to_host(&self, bytes: &[u8], row_bytes: usize, dtype: GgmlType, shape: &[usize]) -> Option<QuantizedTensor> {
        // (only with OAIY_STREAM_PIN=1: from its file is as fast for a prompt, faster for a step, and takes no memory
        // of the system's for itself)
        if !streaming_on() || !std::env::var("OAIY_STREAM_PIN").is_ok_and(|v| v == "1") {
            return None;
        }
        self.pin_to_host(bytes, row_bytes, dtype, shape)
    }

    /// [`Self::upload_to_host`] whatever OAIY_STREAM_PIN says.
    pub(crate) fn pin_to_host(&self, bytes: &[u8], row_bytes: usize, dtype: GgmlType, shape: &[usize]) -> Option<QuantizedTensor> {
        if !self.host_weights() {
            return None;
        }
        let (available, total) = host_memory()?;
        if available < bytes.len() + host_reserve(total) {
            return None;
        }
        let chunks = self.gpu.host_rows(bytes, row_bytes)?;
        let nbytes = bytes.len();
        self.streamed.fetch_add(nbytes as u64, Ordering::Relaxed);
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype, chunks, row_bytes, nbytes, used: Arc::clone(&self.streamed), host: true, file: None };
        Some(QuantizedTensor::from_device(Box::new(storage), shape.to_vec()))
    }

    /// `w` (its rows `row_bytes` on the card) left in its file: held as the card's, its bytes written into the card's
    /// slots each time a matmul takes it ([`Recorder::stage`]), counted as [`Self::filed`]. Where the host's memory has
    /// no room for it beside the reserve: the system keeps what of the file it can, and reads the rest from the drive.
    pub(crate) fn upload_as_file(&self, w: QuantizedTensor, row_bytes: usize, padded: Option<(usize, usize, usize)>) -> QuantizedTensor {
        let nbytes = row_bytes * w.dim(0);
        if !streaming_on() {
            self.left.fetch_add(nbytes as u64, Ordering::Relaxed);
            return w;
        }
        self.filed.fetch_add(nbytes as u64, Ordering::Relaxed);
        let (dtype, shape) = (w.dtype(), w.shape().to_vec());
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype, chunks: Vec::new(), row_bytes, nbytes, used: Arc::clone(&self.filed), host: false, file: Some((w, padded)) };
        QuantizedTensor::from_device(Box::new(storage), shape)
    }

    /// Bytes of weights the budget had no room for that the card reads from the host's memory.
    pub fn streamed(&self) -> u64 {
        self.streamed.load(Ordering::Relaxed)
    }

    /// Bytes of weights the budget had no room for that stay in their file and are written to the card as they are
    /// used.
    pub fn filed(&self) -> u64 {
        self.filed.load(Ordering::Relaxed)
    }
}

impl Recorder<'_> {
    /// `w` for `m` rows of a matmul, where it is not on the card: from the host's memory, for a prompt's rows, a copy of
    /// it in the card's slots (recorded before the matmul, in its pass: the bus carries it once; a step's rows read it
    /// where it lies); from its file, for any rows, its bytes written into the slots (the work recorded so far sent
    /// first, so the write is after it). None where the matmul reads `w` as it is.
    pub(crate) fn stage(&mut self, w: &QuantizedTensor, m: usize) -> Option<QuantizedTensor> {
        let q = w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?;
        let chunks: Vec<(wgpu::Buffer, u32, u32)> = if let Some((file, padded)) = &q.file {
            let bytes = card_bytes(file, *padded);
            let rows = w.dim(0);
            let per = ((chunk_limit(&self.gpu().limits) as usize) / q.row_bytes).max(1);
            let ranges: Vec<(usize, usize)> = (0..rows).step_by(per).map(|r| (r, per.min(rows - r))).collect();
            let sizes: Vec<u64> = ranges.iter().map(|(_, n)| ((n * q.row_bytes) as u64).div_ceil(4) * 4).collect();
            self.submit_so_far();
            // (the feed's pieces all submitted first: a write behind them is made by the feed through the queue's
            // staging memory, 0.21 GB/s on an NVIDIA A2 over Thunderbolt 3, where the host's is 2)
            self.gpu().settle();
            let slots = self.gpu().stream_slots(&sizes);
            let chunks: Vec<(wgpu::Buffer, u32, u32)> = ranges
                .iter()
                .zip(slots)
                .map(|(&(r, n), slot)| {
                    let data = &bytes[r * q.row_bytes..(r + n) * q.row_bytes];
                    if data.len() % 4 == 0 {
                        self.gpu().write(&slot, 0, data);
                    } else {
                        let mut padded = data.to_vec();
                        padded.resize(data.len().div_ceil(4) * 4, 0);
                        self.gpu().write(&slot, 0, &padded);
                    }
                    (slot, r as u32, n as u32)
                })
                .collect();
            chunks
        } else if q.host && m >= STAGE_ROWS {
            let drw = self.gpu().dummy_rw().clone();
            let dd = self.gpu().dummy().clone();
            let sizes: Vec<u64> = q.chunks.iter().map(|(b, _, _)| b.size()).collect();
            let slots = self.gpu().stream_slots(&sizes);
            let mut chunks = Vec::with_capacity(q.chunks.len());
            for ((buffer, row0, rows), slot) in q.chunks.iter().zip(slots) {
                let words = (buffer.size() / 4) as u32;
                self.dispatch_wide("stream-copy-words", COPY_WORDS, [buffer, &dd, &dd, &dd, &dd, &dd, &slot, &drw], &[words], grid(words.div_ceil(4).div_ceil(256)));
                chunks.push((slot, *row0, *rows));
            }
            chunks
        } else {
            return None;
        };
        let storage = WgpuQuant { gpu: Arc::clone(&q.gpu), dtype: q.dtype, chunks, row_bytes: q.row_bytes, nbytes: 0, used: Arc::new(AtomicU64::new(0)), host: false, file: None };
        Some(QuantizedTensor::from_device(Box::new(storage), w.shape().to_vec()))
    }
}
