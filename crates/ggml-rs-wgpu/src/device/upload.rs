//! Writes to the device's buffers through the queue's staging memory, or a large one's through the host's
//! ([`Gpu::write_through_host`](crate::Gpu::write_through_host)).

/// `data` (a multiple of 4 bytes) into `buffer` at `offset` through a write's staging memory, copied there from every
/// core: one core's copy into the card's memory (Resizable BAR) took 1.4 GB/s, most of a weight's load. Small
/// writes as `write_buffer`'s.
pub(crate) fn write_bytes(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
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
pub(crate) fn write_with(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, len: usize, fill: &(dyn Fn(usize, &mut [u8]) + Sync)) -> bool {
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
pub(crate) const WRITE_BY_PIECES: usize = 64 << 20;
pub(crate) const WRITE_PIECE: usize = 32 << 20;

/// OAIY_WHOLE_WRITES: a large write whole and weights in turn flushed every 256 MB, as they were before a write went
/// by pieces (a measurement's other side).
pub(crate) fn whole_writes() -> bool {
    static ASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| std::env::var_os("OAIY_WHOLE_WRITES").is_some())
}

/// The bytes a write needs before it goes through the host's memory ([`Gpu::write_through_host`]), where the device's
/// writes do; a smaller one goes through the queue's staging memory as before.
pub(crate) const HOST_WRITE_MIN: usize = 1 << 20;

/// The two staging buffers in the host's memory a large write goes through a piece ([`WRITE_PIECE`]) at a time, each
/// with the submission of its last copy to the card ([`Gpu::write_through_host`]).
#[derive(Default)]
pub(crate) struct HostPieces(Vec<(wgpu::Buffer, Option<wgpu::SubmissionIndex>)>, usize);

impl crate::Gpu {
    /// Whether this device's large writes go through the host's memory ([`Self::write_through_host`]): where the
    /// queue's staging memory, which wgpu's Vulkan backend puts in the card's own host-visible memory (its BAR) where
    /// there is such, takes the CPU's writes at under 0.5 GB/s. An NVIDIA A2 in a Thunderbolt 3 enclosure took them
    /// at 0.21 GB/s from every core, where the host's memory took 20 and the card read it at 2.2: Qwen3.5 9B's 5.1 GB
    /// of weights loaded in 27 s. OAIY_HOST_STAGING=1: wherever it can be (a Vulkan device with read-mappable
    /// buffers); 0: never.
    pub(crate) fn host_staging(&self) -> bool {
        *self.host_staging.get_or_init(|| {
            let asked = std::env::var("OAIY_HOST_STAGING").ok();
            if asked.as_deref() == Some("0") || !self.host_staging_can() {
                return false;
            }
            if asked.as_deref() == Some("1") {
                return true;
            }
            let rate = self.queue_staging_rate();
            let slow = rate < 0.5;
            // (said once a process: a picture's text encoder, transformer and VAE each open the device)
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if slow && !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("{}: the queue's staging memory takes the CPU's writes at {rate:.2} GB/s: large writes go through the host's memory", self.device.adapter_info().name);
            }
            slow
        })
    }

    /// Whether a read-mappable buffer can be a large write's staging: a Vulkan device with read-mappable buffers that
    /// can be copied from (MAPPABLE_PRIMARY_BUFFERS), whose host-visible memory is all coherent (the CPU's writes are
    /// the GPU's to see with no flush, which a buffer mapped for reading is not given).
    fn host_staging_can(&self) -> bool {
        if self.device.adapter_info().backend != wgpu::Backend::Vulkan || !self.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
            return false;
        }
        #[cfg(any(windows, target_os = "linux"))]
        {
            use ash::vk::MemoryPropertyFlags as F;
            // SAFETY: the device's handles are only read, by a query that creates and frees nothing.
            let Some(d) = (unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() }) else { return false };
            let props = unsafe { d.shared_instance().raw_instance().get_physical_device_memory_properties(d.raw_physical_device()) };
            let types = &props.memory_types[..(props.memory_type_count as usize).min(props.memory_types.len())];
            types.iter().filter(|t| t.property_flags.contains(F::HOST_VISIBLE)).all(|t| t.property_flags.contains(F::HOST_COHERENT))
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        false
    }

    /// GB a second one core writes into a write-mappable buffer's memory, the queue's staging memory's (8 MB).
    fn queue_staging_rate(&self) -> f64 {
        const BYTES: usize = 8 << 20;
        let probe = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-staging-probe"), size: BYTES as u64, usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        probe.slice(..).map_async(wgpu::MapMode::Write, |_| {});
        self.wait(None);
        let data = vec![0x5au8; BYTES];
        let began = std::time::Instant::now();
        {
            let Ok(mut view) = probe.slice(..).get_mapped_range_mut() else { return f64::INFINITY };
            let mut whole = view.slice(..);
            // SAFETY: the buffer's BYTES bytes, mapped for writing until the view is dropped; nothing reads them
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), whole.as_raw_ptr().cast::<u8>().as_ptr(), BYTES) };
        }
        let secs = began.elapsed().as_secs_f64();
        probe.unmap();
        BYTES as f64 / secs.max(1e-9) / 1e9
    }

    /// `parts` of buffers made mappable for writing (MAP_WRITE: the card's memory, through its BAR), each `(buffer,
    /// offset, len, fill)` (offset a multiple of 8, len of 4), mapped, made in place by `fill` (as [`write_with`]'s,
    /// from the part's own byte 0) on the host's threads, all the parts' bytes shared among them, and unmapped: there
    /// for the next submission. Nothing may be using the buffers. One staging copy fewer than the queue's write, whose
    /// staging in the host's memory the GPU copied at 1.7 to 2.5 GB/s, where the threads write the card's memory at
    /// 21.7 over eight lanes.
    pub(crate) fn write_mapped(&self, parts: &[(&wgpu::Buffer, u64, usize, &(dyn Fn(usize, &mut [u8]) + Sync))]) {
        use std::sync::atomic::{AtomicU8, Ordering};
        // (0 waiting, 1 mapped, 2 refused)
        let states: Vec<std::sync::Arc<AtomicU8>> = parts
            .iter()
            .map(|&(buffer, offset, len, _)| {
                let state = std::sync::Arc::new(AtomicU8::new(0));
                let flag = std::sync::Arc::clone(&state);
                buffer.slice(offset..offset + len as u64).map_async(wgpu::MapMode::Write, move |r| flag.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release));
                state
            })
            .collect();
        let _ = self.device.poll(wgpu::PollType::Poll);
        if states.iter().any(|s| s.load(Ordering::Acquire) == 0) {
            self.wait(None);
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
        assert!(states.iter().all(|s| s.load(Ordering::Acquire) == 1), "webgpu: a slot's buffer was not mapped for writing");
        {
            let mut views: Vec<wgpu::BufferViewMut> = parts.iter().map(|&(buffer, offset, len, _)| buffer.slice(offset..offset + len as u64).get_mapped_range_mut().expect("webgpu: a mapped buffer")).collect();
            let total: usize = parts.iter().map(|p| p.2).sum();
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
            let each = total.div_ceil(threads).div_ceil(4096) * 4096;
            std::thread::scope(|s| {
                for (view, &(_, _, len, fill)) in views.iter_mut().zip(parts) {
                    let mut whole = view.slice(..);
                    let base = whole.as_raw_ptr().cast::<u8>().as_ptr() as usize;
                    for at in (0..len).step_by(each.max(4096)) {
                        let n = each.max(4096).min(len - at);
                        // SAFETY: each thread is given its own bytes of the part's mapped range, mapped for writing
                        // until its view (which outlives this scope) is dropped; nothing else reads or writes them
                        s.spawn(move || fill(at, unsafe { std::slice::from_raw_parts_mut((base + at) as *mut u8, n) }));
                    }
                }
            });
        }
        for &(buffer, ..) in parts {
            buffer.unmap();
        }
    }

    /// `len` bytes (a multiple of 4) of `buffer` from `offset`, made by `fill` on every core ([`write_with`]'s `fill`)
    /// in a staging buffer in the host's memory a piece at a time, each piece copied to the card by the GPU: a
    /// read-mappable buffer, which wgpu's Vulkan backend puts in the host's cached memory, written through its mapping.
    /// Two such buffers in turn, so a piece is made as the last one is copied.
    pub(crate) fn write_through_host(&self, buffer: &wgpu::Buffer, offset: u64, len: usize, fill: &(dyn Fn(usize, &mut [u8]) + Sync)) {
        use std::sync::atomic::{AtomicU8, Ordering};
        let mut pieces = self.host_pieces.lock().unwrap_or_else(|p| p.into_inner());
        if pieces.0.is_empty() {
            for _ in 0..2 {
                let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-host-staging"), size: WRITE_PIECE as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                pieces.0.push((staging, None));
            }
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        for at in (0..len).step_by(WRITE_PIECE) {
            let n = WRITE_PIECE.min(len - at);
            let k = pieces.1;
            pieces.1 = (k + 1) % pieces.0.len();
            let (staging, last) = &mut pieces.0[k];
            let slice = staging.slice(..n as u64);
            // (0 waiting, 1 mapped, 2 refused)
            let state = std::sync::Arc::new(AtomicU8::new(0));
            let flag = std::sync::Arc::clone(&state);
            slice.map_async(wgpu::MapMode::Read, move |r| flag.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release));
            // (mapped once the GPU's copy of its last piece is done: that waited for, then the device polled; where it is
            // not mapped yet, everything waited for)
            if let Some(index) = last.take() {
                self.wait(Some(index));
            }
            let _ = self.device.poll(wgpu::PollType::Poll);
            if state.load(Ordering::Acquire) == 0 {
                self.wait(None);
                let _ = self.device.poll(wgpu::PollType::Poll);
            }
            assert_eq!(state.load(Ordering::Acquire), 1, "webgpu: a host staging buffer was not mapped");
            {
                let mut view = slice.get_mapped_range_mut().expect("webgpu: a mapped host staging buffer");
                let mut whole = view.slice(..);
                let base = whole.as_raw_ptr().cast::<u8>().as_ptr() as usize;
                let each = n.div_ceil(threads).div_ceil(4096) * 4096;
                std::thread::scope(|s| {
                    for from in (0..n).step_by(each) {
                        let m = each.min(n - from);
                        // SAFETY: each thread is given its own bytes of the staging buffer, mapped until the view (which
                        // outlives this scope) is dropped; nothing else reads or writes them
                        s.spawn(move || fill(at + from, unsafe { std::slice::from_raw_parts_mut((base + from) as *mut u8, m) }));
                    }
                });
            }
            staging.unmap();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("oaiy-host-write") });
            encoder.copy_buffer_to_buffer(staging, 0, buffer, offset + at as u64, n as u64);
            *last = Some(self.queue().submit([encoder.finish()]));
        }
    }
}
