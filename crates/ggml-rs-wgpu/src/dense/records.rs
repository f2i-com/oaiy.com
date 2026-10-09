//! Routed experts' records in slots on the GPU, read as stored.

use super::*;

/// Routed experts' records on the GPU, a slot a record, made once and written call after call: a prompt's busy experts
/// pass through them a group at a time. A record goes up as stored, one write, and its MXFP4 matrices are read in place
/// with their e8m0 scales, so nothing is converted on the host. The slots count against the weight budget while they
/// live.
pub struct RecordSlots {
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
    slots: Vec<wgpu::Buffer>,
    record_bytes: usize,
    nbytes: u64,
    used: Arc<AtomicU64>,
}

impl Drop for RecordSlots {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes, Ordering::Relaxed);
        forget(&self.gpu, &mut self.slots.iter());
    }
}

impl RecordSlots {
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// `record` into slot `i`, there before the matmuls of the next call: written, submitted and waited for. A
    /// write's staging memory is let go when its submission has run, so each record's is then the memory the one
    /// before had; queued one behind another with no wait, as they were, every record's staging was new memory,
    /// which the system hands over zeroed. `measure_the_upload_rate`, 32 records (604 MB), an RTX 5090 on eight
    /// lanes: 6.4 ms a record queued together (2.9 GB a second), 0.93 ms each waited for (20 GB a second; on four
    /// lanes 5.2 and 1.8 ms). One core's copy, as `write_buffer` makes it: a record copied on every core
    /// ([`Gpu::write`]) was 1.26 ms, its threads' start more than the copy they shared.
    pub fn write(&self, i: usize, record: &[u8]) {
        assert_eq!(record.len(), self.record_bytes, "dense: a record of another size");
        self.gpu.queue().write_buffer(&self.slots[i], 0, record);
        self.gpu.queue().submit([]);
        self.gpu.wait(None);
    }

    /// Slot `i`'s MXFP4 matrix `[n, k]`: its nibbles from byte `w` (`[n, k]`, two to a byte, low first), its e8m0 scales
    /// from byte `s` (`[n, k/32]`). It reads whatever the slot holds when it is used.
    pub fn mxfp4(&self, i: usize, w: usize, s: usize, n: usize, k: usize) -> DenseGpu {
        assert!(k % 32 == 0 && w % 4 == 0, "dense: a record's matrix off its words");
        assert!(w + n * k / 2 <= self.record_bytes && s + n * (k / 32) <= self.record_bytes, "dense: a matrix past its record");
        DenseGpu {
            gpu: Arc::clone(&self.gpu),
            serial: Arc::clone(&self.serial),
            chunks: vec![(self.slots[i].clone(), 0, n as u32, s as u32, (w / 4) as u32)],
            n,
            k,
            kind: Kind::Record,
            nbytes: 0,
            used: Arc::clone(&self.used),
        }
    }
}

impl WgpuBackend {
    /// Up to `count` slots for records of `record_bytes`, as many as the weight budget has room for; None if not one,
    /// or if a record is not whole words or is past the binding limit.
    pub fn record_slots(&self, count: usize, record_bytes: usize) -> Option<RecordSlots> {
        if record_bytes == 0 || record_bytes % 4 != 0 || record_bytes as u64 > chunk_limit(&self.gpu.limits) {
            return None;
        }
        let free = self.budget.saturating_sub(self.used.load(Ordering::Relaxed));
        let n = count.min((free / record_bytes as u64) as usize);
        if n == 0 {
            return None;
        }
        let nbytes = (n * record_bytes) as u64;
        self.used.fetch_add(nbytes, Ordering::Relaxed);
        let slots = (0..n)
            .map(|_| {
                self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("oaiy-record"),
                    size: record_bytes as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Some(RecordSlots { gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial), slots, record_bytes, nbytes, used: Arc::clone(&self.used) })
    }
}
