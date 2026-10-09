//! Writes to the device's buffers through the queue's staging memory.

/// `data` (a multiple of 4 bytes) into `buffer` at `offset` through a write's staging memory, copied there from every
/// core: one core's copy into the card's memory (Resizable BAR) took 1.4 GB/s, most of a weight's load. Small
/// writes as `write_buffer`'s.
pub(super) fn write_bytes(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    let size = match wgpu::BufferSize::new(data.len() as u64) {
        Some(size) if data.len() >= 8 << 20 => size,
        _ => return queue.write_buffer(buffer, offset, data),
    };
    let Some(mut view) = queue.write_buffer_with(buffer, offset, size) else { return queue.write_buffer(buffer, offset, data) };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let each = data.len().div_ceil(threads).div_ceil(4096) * 4096;
    let mut whole = view.slice(..);
    let base = whole.as_raw_ptr().cast::<u8>().as_ptr() as usize;
    std::thread::scope(|s| {
        for (i, part) in data.chunks(each).enumerate() {
            let at = base + i * each;
            // SAFETY: each thread writes its own bytes of the staging memory, mapped for writing until the view
            // (which outlives this scope) is dropped; nothing reads them
            s.spawn(move || unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), at as *mut u8, part.len()) });
        }
    });
}

/// `len` bytes (a multiple of 4) of `buffer` from `offset`, made in place by `fill` on every core: `fill(at, part)` puts
/// the bytes from `at` on into `part`, a part of the write's staging memory, a thread a part. A file read there
/// straight is copied once, where bytes read into the host's memory first are copied again by [`write_bytes`]. False
/// (nothing written) where the queue gives no such memory.
pub(super) fn write_with(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, len: usize, fill: &(dyn Fn(usize, &mut [u8]) + Sync)) -> bool {
    let Some(size) = wgpu::BufferSize::new(len as u64) else { return false };
    let Some(mut view) = queue.write_buffer_with(buffer, offset, size) else { return false };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let each = len.div_ceil(threads).div_ceil(4096) * 4096;
    let mut whole = view.slice(..);
    let base = whole.as_raw_ptr().cast::<u8>().as_ptr() as usize;
    std::thread::scope(|s| {
        for at in (0..len).step_by(each) {
            let n = each.min(len - at);
            // SAFETY: each thread is given its own bytes of the staging memory, mapped for writing until the view
            // (which outlives this scope) is dropped; nothing else reads or writes them
            s.spawn(move || fill(at, unsafe { std::slice::from_raw_parts_mut((base + at) as *mut u8, n) }));
        }
    });
    true
}

/// The bytes from which a write goes a piece at a time ([`Gpu::write`]), and a piece's.
pub(super) const WRITE_BY_PIECES: usize = 64 << 20;
pub(super) const WRITE_PIECE: usize = 32 << 20;

/// OAIY_WHOLE_WRITES: a large write whole and weights in turn flushed every 256 MB, as they were before a write went
/// by pieces (a measurement's other side).
pub(super) fn whole_writes() -> bool {
    static ASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| std::env::var_os("OAIY_WHOLE_WRITES").is_some())
}
