//! The device's own functions: its pipelines and bind groups, its pools of scratch, its waits and read-backs,
//! its uploads.

use super::*;

impl Drop for Gpu {
    fn drop(&mut self) {
        self.feed.state.lock().unwrap_or_else(|p| p.into_inner()).closed = true;
        self.feed.changed.notify_all();
    }
}

impl Gpu {
    /// The EXL3 projections' few-rows scratch ([`exl3::FewScratch`]), made by `b` when first asked for.
    pub(crate) fn few(&self, b: &WgpuBackend) -> &exl3::FewScratch {
        self.few.get_or_init(|| exl3::FewScratch::new(b))
    }

    /// How many units (SMs) the GPU's workgroups are shared out to ([`shaders::COOP_UNITS_PROBE`]'s count of 4,096
    /// workgroups of 1024 threads that each work about 0.1 ms: those started before any had finished), counted once
    /// (a millisecond or so); 1 where a workgroup cannot be that large.
    pub(crate) fn coop_units(&self) -> u32 {
        *self.coop_units.get_or_init(|| {
            if self.limits.max_compute_invocations_per_workgroup < 1024 || self.limits.max_compute_workgroup_size_x < 1024 {
                return 1;
            }
            let pipeline = self.named_pipeline("coop-units-probe", || shaders::COOP_UNITS_PROBE.to_string());
            let counts = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-coop-units"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let params = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-coop-units-params"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut words = [0u8; 32];
            words[..4].copy_from_slice(&65536u32.to_le_bytes());
            self.queue().write_buffer(&params, 0, &words);
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-coop-units"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.dummy().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.dummy().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                ],
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(4096, 1, 1);
            }
            self.queue().submit([enc.finish()]);
            let got = self.read(&counts, 8);
            u32::from_le_bytes([got[0], got[1], got[2], got[3]]).max(1)
        })
    }

    /// Copy `len` bytes of `src` back to the host.
    pub(super) fn read(&self, src: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let len = len.div_ceil(4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-readback"),
            size: len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, len);
        self.queue().submit([enc.finish()]);
        self.map_read(&staging, len)
    }

    /// The device's queue, for a write or a submission of the caller's own: after everything handed to the feed has
    /// gone to it (a write is before the submissions after it; a piece still waiting would run after a write made
    /// since it was recorded).
    pub(super) fn queue(&self) -> &wgpu::Queue {
        self.settle();
        &self.queue_raw
    }

    /// Whether the device's pieces in flight are limited: its chains' pieces go by its feed.
    pub(crate) fn feeds(&self) -> bool {
        pieces_in_flight_asked().unwrap_or_else(|| self.in_flight_limit.load(Ordering::Relaxed)) > 0
    }

    /// Waits until everything handed to the feed is submitted (at once where nothing is waiting).
    pub(super) fn settle(&self) {
        let mut s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
        while s.gone < s.handed {
            if let Some(why) = &s.failed {
                panic!("webgpu: {why}");
            }
            s = self.feed.changed.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// `fare` handed to the feed (its thread begun at the first), where its submission will be said.
    pub(super) fn hand(&self, fare: Fare) -> Piece {
        let slot = Arc::new(Mutex::new(None));
        let mut s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(why) = &s.failed {
            panic!("webgpu: {why}");
        }
        s.waiting.push_back((fare, Arc::clone(&slot)));
        s.handed += 1;
        if !s.fed {
            s.fed = true;
            let (feed, device, queue, watch, lost, limit) = (Arc::clone(&self.feed), self.device.clone(), self.queue_raw.clone(), Arc::clone(&self.watch), Arc::clone(&self.lost), Arc::clone(&self.in_flight_limit));
            std::thread::Builder::new().name("oaiy-webgpu-feed".into()).spawn(move || feed_pieces(feed, device, queue, watch, lost, limit)).expect("webgpu: a thread for the device's queue");
        }
        drop(s);
        self.feed.changed.notify_all();
        Piece::Fed(slot)
    }

    /// A chain's piece as its dispatches: where the device's pieces in flight are limited
    /// ([`WgpuBackend::pieces_in_flight_at_most`]), to its feed, which encodes and submits it once all but the limit
    /// less one of those before it have run (the caller goes on at once); else encoded and submitted here.
    pub(crate) fn feed(&self, dispatches: Vec<Dispatch>) {
        if self.feeds() {
            self.hand(Fare::Dispatches(dispatches));
        } else {
            self.queue().submit([encode(&self.device, &dispatches)]);
        }
    }

    /// A chain's piece its recorder encoded (its command buffers, in turn) submitted: where the device's pieces in
    /// flight are limited, by its feed a command buffer a piece, and waited for until the last of them has gone to the
    /// queue (all but the limit less one of those before it have run: such a piece keeps its recorder to the queue's
    /// pace). The piece returned is the last of them.
    pub(crate) fn submit_piece(&self, commands: Vec<wgpu::CommandBuffer>) -> Piece {
        if !self.feeds() || commands.is_empty() {
            return Piece::Gone(self.queue().submit(commands));
        }
        let mut last = None;
        for command in commands {
            last = Some(self.hand(Fare::Commands(vec![command])));
        }
        Piece::Gone(self.gone(last.expect("a command buffer")))
    }

    /// A recording's copies out (its reads, its timestamps) submitted after its pieces: by the feed where the device's
    /// pieces in flight are limited (the caller goes on at once), else here.
    pub(crate) fn submit_after(&self, commands: Vec<wgpu::CommandBuffer>) -> Piece {
        if self.feeds() {
            self.hand(Fare::Commands(commands))
        } else {
            Piece::Gone(self.queue().submit(commands))
        }
    }

    /// `piece`'s submission, once it has gone to the queue (a buffer it copies into can be mapped from then, not
    /// before).
    pub(crate) fn gone(&self, piece: Piece) -> wgpu::SubmissionIndex {
        let slot = match piece {
            Piece::Gone(index) => return index,
            Piece::Fed(slot) => slot,
        };
        let mut s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(index) = slot.lock().unwrap_or_else(|p| p.into_inner()).clone() {
                return index;
            }
            if let Some(why) = &s.failed {
                panic!("webgpu: {why}");
            }
            s = self.feed.changed.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Waits for submission `index` (else everything submitted, the feed's pieces too), a second at a time: a lost
    /// device (its callback has said so) fails the job rather than leaving it waiting for good.
    pub(crate) fn wait(&self, index: Option<wgpu::SubmissionIndex>) {
        if index.is_none() {
            self.settle();
        }
        if let Err(why) = wait_for(&self.device, &self.watch, &self.lost, index) {
            panic!("webgpu: {why}");
        }
    }

    /// [`Self::wait`] for submission `index` whose read-back sets `mapped` when it is there: the device polled for up
    /// to `spin` first, so a round trip a fraction of a millisecond away does not park the thread and wake it (the
    /// OS's wake was most of such a wait).
    pub(crate) fn wait_soon(&self, index: wgpu::SubmissionIndex, mapped: &std::sync::atomic::AtomicBool, spin: std::time::Duration) {
        let began = std::time::Instant::now();
        while !mapped.load(Ordering::Acquire) {
            if began.elapsed() >= spin {
                return self.wait(Some(index));
            }
            let _ = self.device.poll(wgpu::PollType::Poll);
            std::hint::spin_loop();
        }
        if let Some(why) = self.lost.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            panic!("webgpu: the device was lost ({why})");
        }
    }

    pub(super) fn map_read(&self, staging: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let slice = staging.slice(..len);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.wait(None);
        let out = slice.get_mapped_range().expect("webgpu: mapping a finished buffer").to_vec();
        staging.unmap();
        out
    }

    /// [`Self::map_read`] for a read-back that is a fraction of a millisecond away (a decode step's dense call): the
    /// device polled for [`SPIN`] before it is waited for, so the thread is not parked and woken for each (the OS's
    /// wake of a parked thread was most of such a call's wait). OAIY_DENSE_NO_SPIN: waited for from the start.
    pub(super) fn map_read_soon(&self, staging: &wgpu::Buffer, len: u64) -> Vec<u8> {
        /// How long the device is polled before the thread waits.
        const SPIN: std::time::Duration = std::time::Duration::from_millis(2);
        static NO_SPIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *NO_SPIN.get_or_init(|| std::env::var_os("OAIY_DENSE_NO_SPIN").is_some()) {
            return self.map_read(staging, len);
        }
        let slice = staging.slice(..len);
        let mapped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&mapped);
        slice.map_async(wgpu::MapMode::Read, move |_| flag.store(true, Ordering::Release));
        self.settle();
        let began = std::time::Instant::now();
        while !mapped.load(Ordering::Acquire) {
            if began.elapsed() > SPIN {
                if let Err(why) = wait_for(&self.device, &self.watch, &self.lost, None) {
                    panic!("webgpu: {why}");
                }
                break;
            }
            let _ = self.device.poll(wgpu::PollType::Poll);
            std::hint::spin_loop();
        }
        let out = slice.get_mapped_range().expect("webgpu: mapping a finished buffer").to_vec();
        staging.unmap();
        out
    }

    /// `data` (a multiple of 4 bytes) into `buffer` at `offset` ([`write_bytes`]): at once, or, behind pieces still
    /// waiting for the queue, in its turn after them by the feed (a copy of the bytes), the caller going on (a
    /// prompt's next chunk is recorded as the one before runs: its embeddings' upload waiting for the queue kept the
    /// recording to the GPU's last pieces). A large write ([`WRITE_BY_PIECES`]) goes a piece at a time, each waited
    /// for: a write's staging memory is let go when its submission has run, so each piece's is the allocator's same
    /// block again, where a whole write's (or pieces' not waited for) was new memory, first touched as it was
    /// written: 1 GiB in 53 ms where 280 to 360 (`measure_an_uploads_ways`), and no more of the card held than a
    /// piece (a model's weights whole held 31 GB of a 32 GB card for Qwen Image's 14).
    pub(crate) fn write(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let holds = {
            let s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
            s.gone < s.handed && s.failed.is_none()
        };
        if holds {
            self.hand(Fare::Write(buffer.clone(), offset, data.to_vec()));
        } else if data.len() < WRITE_BY_PIECES || whole_writes() {
            write_bytes(self.queue(), buffer, offset, data);
        } else {
            for (i, piece) in data.chunks(WRITE_PIECE).enumerate() {
                write_bytes(self.queue(), buffer, offset + (i * WRITE_PIECE) as u64, piece);
                self.queue().submit([]);
                self.wait(None);
            }
        }
    }

    /// [`write_with`]: `len` bytes of `buffer` from `offset` made in place by `fill`, a large one a piece at a time as
    /// [`Self::write`]'s. False where a piece still waits for the queue (nothing written: such a write goes in its
    /// turn, by its bytes), or the queue gives no staging memory (the caller writes the bytes, all of them).
    pub(crate) fn write_with(&self, buffer: &wgpu::Buffer, offset: u64, len: usize, fill: &(dyn Fn(usize, &mut [u8]) + Sync)) -> bool {
        let holds = {
            let s = self.feed.state.lock().unwrap_or_else(|p| p.into_inner());
            s.gone < s.handed && s.failed.is_none()
        };
        if holds {
            return false;
        }
        let pieces = len >= WRITE_BY_PIECES && !whole_writes();
        let piece = if pieces { WRITE_PIECE } else { len };
        for at in (0..len).step_by(piece.max(1)) {
            let n = piece.min(len - at);
            if !write_with(self.queue(), buffer, offset + at as u64, n, &|from, part| fill(at + from, part)) {
                return false;
            }
            if pieces {
                self.queue().submit([]);
                self.wait(None);
            }
        }
        true
    }

    /// `bytes` more written since the queue was last flushed: flushed and waited for at a piece's worth
    /// ([`WRITE_PIECE`]), as a large write is by itself ([`Self::write`]): weights uploaded in turn and never waited
    /// for each took new staging memory, and all of it was held until the next submission (loading a 16 GB model
    /// exhausted it).
    pub(super) fn staged_more(&self, bytes: u64) {
        if bytes >= WRITE_BY_PIECES as u64 && !whole_writes() {
            return self.staged.store(0, Ordering::Relaxed);
        }
        let pending = self.staged.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if pending >= if whole_writes() { 256 << 20 } else { WRITE_PIECE as u64 } {
            self.staged.store(0, Ordering::Relaxed);
            self.queue().submit([]);
            self.wait(None);
        }
    }

    /// `bytes` (a multiple of 4 of them) in a storage buffer in the HOST's memory, which a kernel reads over the bus:
    /// a read-mappable storage buffer, for which wgpu's Vulkan backend asks for the host's cached memory (the system's,
    /// on a card of its own; a write-mappable one it puts in the card's own host-visible memory where there is such,
    /// which is the card's speed and the card's room). None where the device has no mappable storage buffers. It
    /// costs the card none of its memory and a kernel its bytes at the bus's speed: an RTX 5090 on eight lanes of
    /// PCIe 5 reads it at 26.6 GB a second, on four at 13.3 (`measure_a_matrix_in_the_hosts_memory`), where its own
    /// memory gives 1,400.
    pub(crate) fn host_buffer(&self, bytes: &[u8]) -> Option<wgpu::Buffer> {
        if bytes.is_empty() || bytes.len() % 4 != 0 || !self.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
            return None;
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-host-weights"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.write(&buffer, 0, bytes);
        self.staged_more(bytes.len() as u64);
        Some(buffer)
    }

    /// Upload whole rows into buffers below the binding limit.
    pub(super) fn upload_rows(&self, bytes: &[u8], row_bytes: usize, _hint: usize) -> Vec<(wgpu::Buffer, u32, u32)> {
        let rows = bytes.len() / row_bytes;
        let per = ((chunk_limit(&self.limits) as usize) / row_bytes).max(1);
        let mut chunks = Vec::new();
        let mut r = 0;
        while r < rows {
            let n = per.min(rows - r);
            let data = &bytes[r * row_bytes..(r + n) * row_bytes];
            let size = (data.len() as u64).div_ceil(4) * 4;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-weight"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if data.len() % 4 == 0 {
                self.write(&buffer, 0, data);
            } else {
                let mut padded = data.to_vec();
                padded.resize(size as usize, 0);
                self.write(&buffer, 0, &padded);
            }
            chunks.push((buffer, r as u32, n as u32));
            r += n;
            self.staged_more(size);
        }
        chunks
    }

    /// A kernel's module, its workgroup memory checked first where the device has less than 48 KB of it (Apple's 32,
    /// OAIY_PORTABLE_LIMITS's): wgpu 30 does not check it, and a kernel past the limit fails where it runs, if its
    /// driver says at all.
    pub(super) fn shader(&self, name: &str, source: String) -> wgpu::ShaderModule {
        if self.limits.max_compute_workgroup_storage_size < 48 << 10 {
            if let Some(bytes) = workgroup_bytes(&source) {
                assert!(
                    bytes <= self.limits.max_compute_workgroup_storage_size as u64,
                    "webgpu: kernel {name} takes {bytes} bytes of a workgroup's memory, past this device's {}",
                    self.limits.max_compute_workgroup_storage_size
                );
            }
        }
        // OAIY_KERNEL_DUMP: a folder each kernel's text is written to as it is made, one file a text (its name and a
        // hash of it): to read one, and to give them all to another API's translator where no such GPU is at hand (a
        // Mac's is naga's Metal writer, which runs anywhere)
        if let Some(dir) = std::env::var_os("OAIY_KERNEL_DUMP") {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            let plain: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{plain}-{:016x}.wgsl", h.finish())), &source);
        }
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(source.into()) })
    }

    pub(super) fn exl3_pipeline(&self, many: bool) -> Arc<wgpu::ComputePipeline> {
        let mut slots = self.exl3.lock().unwrap_or_else(|p| p.into_inner());
        let slot = &mut slots[many as usize];
        if let Some(p) = slot.as_ref() {
            return Arc::clone(p);
        }
        let module = self.shader("oaiy-exl3", exl3::shader(many));
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oaiy-exl3"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        *slot = Some(Arc::clone(&pipeline));
        pipeline
    }

    /// The pipeline `name`, made from the WGSL `source` gives the first time it is asked for.
    pub(super) fn named_pipeline(&self, name: &'static str, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        self.named_pipeline_in(name, &self.pipeline_layout, source)
    }

    /// The layout of eight storage buffers (the first six read, the last two written) and the parameters at 8.
    pub(super) fn wide_layout(&self) -> &(wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        self.wide.get_or_init(|| {
            let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8).map(|b| storage(b, b < 6)).collect();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("oaiy-chain-wide"), entries: &entries });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("oaiy-chain-wide"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            (layout, pipeline_layout)
        })
    }

    /// A small buffer for the bindings a kernel does not read.
    pub(super) fn dummy(&self) -> &wgpu::Buffer {
        self.dummy.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
        })
    }

    /// Another, for a written binding a kernel does not write (one buffer may not be bound written twice).
    pub(super) fn dummy_rw(&self) -> &wgpu::Buffer {
        self.dummy_rw.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-dummy-rw"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
        })
    }

    /// A named kernel of eight buffers ([`Gpu::wide_layout`]).
    /// A scratch buffer of `bytes` (a power of two) from the pool, or a new one.
    pub(crate) fn pooled(&self, bytes: u64) -> wgpu::Buffer {
        let taken = {
            let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
            pool.iter().position(|(b, _)| *b == bytes).map(|i| pool.swap_remove(i).1)
        };
        taken.unwrap_or_else(|| {
            SCRATCH_MADE.fetch_add(bytes, Ordering::Relaxed);
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-chain-scratch"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
    }

    /// A staging buffer of `bytes` (a power of two) for a read-back: one a read before used, or a new one.
    pub(crate) fn staging(&self, bytes: u64) -> wgpu::Buffer {
        let taken = {
            let mut pool = self.staging.lock().unwrap_or_else(|p| p.into_inner());
            pool.iter().position(|(b, _)| *b == bytes).map(|i| pool.swap_remove(i).1)
        };
        taken.unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-chain-read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Staging buffers back (unmapped), to be used again; past [`STAGING_BYTES`] the largest are let go.
    pub(crate) fn unstage(&self, buffers: Vec<(u64, wgpu::Buffer)>) {
        let mut pool = self.staging.lock().unwrap_or_else(|p| p.into_inner());
        pool.extend(buffers);
        let mut total: u64 = pool.iter().map(|(b, _)| b).sum();
        if total > STAGING_BYTES {
            pool.sort_by_key(|(b, _)| *b);
            while total > STAGING_BYTES {
                let Some((b, _)) = pool.pop() else { break };
                total -= b;
            }
        }
    }

    /// Scratch buffers back to the pool, once what used them has run; past [`POOL_BYTES`] the largest are let go.
    pub(crate) fn unpool(&self, buffers: Vec<(u64, wgpu::Buffer)>) {
        let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
        pool.extend(buffers);
        let mut total: u64 = pool.iter().map(|(b, _)| b).sum();
        if total > POOL_BYTES {
            pool.sort_by_key(|(b, _)| *b);
            while total > POOL_BYTES {
                let Some((b, _)) = pool.pop() else { break };
                total -= b;
            }
        }
    }

    /// A pipeline's name, for a chain's profile.
    pub(crate) fn name_of(&self, pipeline: &Arc<wgpu::ComputePipeline>) -> &'static str {
        self.names.lock().unwrap_or_else(|p| p.into_inner()).get(&(Arc::as_ptr(pipeline) as usize)).copied().unwrap_or("other")
    }

    /// A generated kernel's source, made once for its name and kept (what a recorder's wide dispatch is given each
    /// time, its pipeline made of it the first).
    pub(super) fn named_source(&self, name: &'static str, make: impl FnOnce() -> String) -> &'static str {
        static MADE: Mutex<Vec<(&'static str, &'static str)>> = Mutex::new(Vec::new());
        let mut made = MADE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, source)) = made.iter().find(|(n, _)| *n == name) {
            return source;
        }
        let source: &'static str = Box::leak(make().into_boxed_str());
        made.push((name, source));
        source
    }

    pub(super) fn named_pipeline_wide(&self, name: &'static str, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let layout = &self.wide_layout().1;
        self.named_pipeline_in(name, layout, source)
    }

    pub(super) fn named_pipeline_in(&self, name: &'static str, layout: &wgpu::PipelineLayout, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self.named.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(name) {
            return Arc::clone(p);
        }
        let began = std::time::Instant::now();
        let module = self.shader(name, source());
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        made(name, began);
        cache.insert(name, Arc::clone(&pipeline));
        self.names.lock().unwrap_or_else(|p| p.into_inner()).insert(Arc::as_ptr(&pipeline) as usize, name);
        pipeline
    }

    /// `dtype`'s kernel for `m` rows of `x`: the decode kernel for one, the one-row kernel for a few, the tiled one
    /// a prompt takes.
    pub(super) fn pipeline(&self, dtype: GgmlType, m: usize) -> Option<Arc<wgpu::ComputePipeline>> {
        let kind = shaders::kind(dtype, m);
        let mut cache = self.pipelines.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(&(dtype, kind)) {
            return Some(Arc::clone(p));
        }
        let source = match kind {
            1 => shaders::source_many(dtype)?,
            2 => shaders::source_decode(dtype)?,
            3 => shaders::source_multi(dtype)?,
            _ => shaders::source(dtype)?,
        };
        let began = std::time::Instant::now();
        let module = self.shader(&format!("oaiy-linear-q ({dtype:?}, kind {kind})"), source);
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oaiy-linear-q"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert((dtype, kind), Arc::clone(&pipeline));
        // a name for a chain's profile (a few, made once each)
        let name: &'static str = Box::leak(format!("matmul-{dtype:?}-{}", ["one", "tiled", "decode", "multi"][kind as usize]).into_boxed_str());
        made(name, began);
        self.names.lock().unwrap_or_else(|p| p.into_inner()).insert(Arc::as_ptr(&pipeline) as usize, name);
        Some(pipeline)
    }
}
