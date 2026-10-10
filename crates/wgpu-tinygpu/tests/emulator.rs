//! WGSL's meaning, kept by the translation to CUDA: kernels run through the adapter on the server's emulator
//! (`tools/tinygpu/webgpu_server.py --emulate`: each kernel compiled for the CPU), so no card is needed. Skipped where
//! the emulator cannot start (no python3 or clang++). With `TINYGPU_TEST_SOCKET` set they run on the server there
//! instead: the card itself.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

/// The emulator, for as long as the test holds it (none where the test runs on a server of its own).
struct Emulator {
    child: Option<Child>,
    socket: PathBuf,
}

impl Drop for Emulator {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = Command::new("kill").arg("-TERM").arg(child.id().to_string()).status();
            let _ = child.wait();
            let _ = std::fs::remove_file(&self.socket);
        }
    }
}

fn emulator() -> Option<(Emulator, wgpu::Device, wgpu::Queue)> {
    if let Some(socket) = std::env::var_os("TINYGPU_TEST_SOCKET") {
        let emu = Emulator { child: None, socket: socket.into() };
        let adapter = wgpu_tinygpu::adapter(&emu.socket).unwrap_or_else(|e| panic!("{e}"));
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
        return Some((emu, device, queue));
    }
    let server = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/tinygpu/webgpu_server.py");
    static STARTED: AtomicUsize = AtomicUsize::new(0);
    let socket = std::env::temp_dir().join(format!("tinygpu-emu-{}-{}.sock", std::process::id(), STARTED.fetch_add(1, Ordering::Relaxed)));
    let child = Command::new("python3").arg(&server).arg("--emulate").arg(&socket).stdout(Stdio::null()).spawn().ok()?;
    let emu = Emulator { child: Some(child), socket };
    let start = Instant::now();
    while !emu.socket.exists() {
        if start.elapsed() > Duration::from_secs(20) {
            eprintln!("skipping: the emulator did not start");
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let adapter = wgpu_tinygpu::adapter(&emu.socket).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
    Some((emu, device, queue))
}

/// `wgsl`'s `main` over `groups` workgroups, its bindings 0.. the buffers `data` (read and written); what they hold after.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, data: &[Vec<u32>], groups: u32) -> Vec<Vec<u32>> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let buffers: Vec<wgpu::Buffer> = data.iter().map(|d| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage })).collect();
    let entries: Vec<wgpu::BindGroupEntry> = buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit([enc.finish()]);
    buffers.iter().zip(data).map(|(b, d)| read(device, queue, b, d.len() * 4)).collect()
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, b: &wgpu::Buffer, bytes: usize) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(b, 0, &staging, 0, bytes as u64);
    queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
    staging.unmap();
    out
}

#[test]
fn wgsl_keeps_its_meaning_on_the_emulator() {
    let Some((_emu, device, queue)) = emulator() else { return };

    // integer division and remainder never fault: by zero the dividend and 0, the most negative by -1 itself and 0
    let out = run(&device, &queue, "
        @group(0) @binding(0) var<storage, read_write> u: array<u32>;
        @group(0) @binding(1) var<storage, read_write> i: array<i32>;
        @compute @workgroup_size(1) fn main() {
            let z = u[0]; let m1 = i[0]; let lo = i[1];
            u[1] = 7u / z; u[2] = 7u % z; u[3] = 7u / 2u;
            i[2] = lo / m1; i[3] = lo % m1; i[4] = -7 / i[5]; i[6] = -7 % 2;
        }", &[vec![0, 0, 0, 0], vec![-1i32 as u32, i32::MIN as u32, 0, 0, 0, 0, 0]], 1);
    assert_eq!(out[0], [0, 7, 0, 3]);
    assert_eq!(out[1].iter().map(|&v| v as i32).collect::<Vec<_>>(), [-1, i32::MIN, i32::MIN, 0, -7, 0, -1]);

    // shifts by the amount modulo the width
    let out = run(&device, &queue, "
        @group(0) @binding(0) var<storage, read_write> u: array<u32>;
        @compute @workgroup_size(1) fn main() { let s = u[0]; u[1] = 1u << s; u[2] = 0x80000000u >> s; u[3] = u32(i32(-8) >> (s - 31u)); }",
        &[vec![33, 0, 0, 0]], 1);
    assert_eq!(out[0], [33, 2, 0x4000_0000, -2i32 as u32]);

    // workgroup memory starts zeroed; a barrier lets each invocation read another's write
    let out = run(&device, &queue, "
        @group(0) @binding(0) var<storage, read_write> o: array<u32>;
        var<workgroup> t: array<u32, 64>;
        var<workgroup> untouched: array<u32, 4>;
        @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
            t[li] = li * 10u + wg.x;
            workgroupBarrier();
            o[wg.x * 64u + li] = t[63u - li] + untouched[li % 4u];
        }", &[vec![0; 128]], 2);
    assert!((0..128).all(|k| out[0][k] == (63 - k as u32 % 64) * 10 + k as u32 / 64), "{:?}", &out[0][..8]);

    // atomics: every invocation of every workgroup counts once; a workgroup's maximum; workgroupUniformLoad
    let out = run(&device, &queue, "
        @group(0) @binding(0) var<storage, read_write> o: array<atomic<u32>>;
        var<workgroup> top: atomic<u32>;
        var<workgroup> pick: u32;
        @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
            atomicAdd(&o[0], 1u);
            atomicMax(&top, li * 3u + wg.x);
            if (li == 5u) { pick = 40u + wg.x; }
            let p = workgroupUniformLoad(&pick);
            workgroupBarrier();
            if (li == 0u) { atomicAdd(&o[1], atomicLoad(&top)); atomicAdd(&o[2], p); }
        }", &[vec![0; 3]], 4);
    assert_eq!(out[0], [512, (127 * 3) * 4 + 6, 40 * 4 + 6]);

    // a vec3 in an array is 16 bytes apart; a struct's members lie where WGSL puts them
    let out = run(&device, &queue, "
        struct S { a: vec3<f32>, b: u32, c: vec2<u32> }
        @group(0) @binding(0) var<storage, read_write> s: array<S>;
        @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) li: u32) {
            s[li].a = vec3<f32>(1.0, 2.0, 3.0) * f32(li + 1u); s[li].b = 7u + li; s[li].c = vec2<u32>(li, 9u);
        }", &[vec![0; 16]], 1);
    let f = |v: f32| v.to_bits();
    assert_eq!(out[0], [f(1.0), f(2.0), f(3.0), 7, 0, 9, 0, 0, f(2.0), f(4.0), f(6.0), 8, 1, 9, 0, 0]);

    // the float built-ins: a float's remainder truncates; a float made an integer saturates (NaN 0)
    let out = run(&device, &queue, "
        @group(0) @binding(0) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(1) fn main() {
            let big = o[0]; let x = o[1];
            o[2] = x % 2.0; o[3] = f32(i32(big)); o[4] = f32(u32(-big)); o[5] = clamp(x, -1.0, 1.0); o[6] = fract(x); o[7] = sign(-x);
        }", &[vec![f(3e9), f(-5.5), 0, 0, 0, 0, 0, 0]], 1);
    assert_eq!(out[0][2..], [f(-1.5), f(2147483647.0), 0, f(-1.0), f(0.5), f(1.0)]);
}

#[test]
fn a_buffer_outlives_its_handle_while_work_holds_it() {
    let Some((_emu, device, queue)) = emulator() else { return };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read> a: array<u32>; @group(0) @binding(1) var<storage, read_write> b: array<u32>;
            @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) li: u32) { b[li] = a[li] + 1u; }".into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[1u32, 2, 3, 4]), usage: wgpu::BufferUsages::STORAGE });
    let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() }],
    });
    drop(a);
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    drop(group);
    queue.submit([enc.finish()]);
    assert_eq!(read(&device, &queue, &b, 16), [2, 3, 4, 5]);
}

/// An integer's f16 bits (exact for the small ones the test uses).
fn f16_bits(v: i32) -> u16 {
    if v == 0 {
        return 0;
    }
    let (sign, mut m) = (if v < 0 { 0x8000u16 } else { 0 }, v.unsigned_abs());
    let mut e = 0u16;
    while m >= 2 {
        m >>= 1;
        e += 1;
    }
    let frac = ((v.unsigned_abs() << (10 - e)) & 0x3ff) as u16;
    sign | ((e + 15) << 10) | frac
}

#[test]
fn the_tensor_cores_multiply_as_wgsl_says() {
    let Some((_emu, device, queue)) = emulator() else { return };
    if !device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let a = |m: i32, k: i32| (m + 2 * k) % 7 - 3;
    let b = |k: i32, n: i32| (3 * k + n) % 5 - 2;
    let c = |m: i32, n: i32| m - n;
    // A row-major in vec4<f16>s, B column-major in f16s (two to a word), C row-major in f32s
    let pack = |v: Vec<u16>| -> Vec<u32> { v.chunks(2).map(|p| p[0] as u32 | (p[1] as u32) << 16).collect() };
    let a_mem = pack((0..256).map(|i| f16_bits(a(i / 16, i % 16))).collect());
    let b_mem = pack((0..256).map(|i| f16_bits(b(i % 16, i / 16))).collect());
    let c_mem: Vec<u32> = (0..256).map(|i| (c(i / 16, i % 16) as f32).to_bits()).collect();
    let out = run(&device, &queue, "
        enable f16;
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> a: array<vec4<f16>>;
        @group(0) @binding(1) var<storage, read_write> b: array<f16>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @group(0) @binding(3) var<storage, read_write> rows: array<f32>;
        @group(0) @binding(4) var<storage, read_write> cols: array<f32>;
        @group(0) @binding(5) var<storage, read_write> halves: array<f16>;
        var<workgroup> t: array<vec4<f16>, 64>;
        @compute @workgroup_size(32) fn main(@builtin(local_invocation_index) li: u32) {
            for (var i = li; i < 64u; i += 32u) { t[i] = a[i]; }
            workgroupBarrier();
            let z = 0u;
            let s4 = 4u;
            let s16 = 16u;
            let x = coopLoadT<coop_mat16x16<f16, A>>(&t[z], s4);
            let y = coopLoad<coop_mat16x16<f16, B>>(&b[z], s16);
            let c0 = coopLoadT<coop_mat16x16<f32, C>>(&c[z], s16);
            let d = coopMultiplyAdd(x, y, c0);
            coopStoreT(d, &rows[z], s16);
            coopStore(d, &cols[z], s16);
            var h = coop_mat16x16<f16, C>();
            h = coopMultiplyAdd(x, y, h);
            coopStoreT(h, &halves[z], s16);
        }", &[a_mem, b_mem, c_mem, vec![0; 256], vec![0; 256], vec![0; 128]], 1);
    let want = |m: i32, n: i32| (0..16).map(|k| a(m, k) * b(k, n)).sum::<i32>();
    for m in 0..16 {
        for n in 0..16 {
            let i = (m * 16 + n) as usize;
            assert_eq!(f32::from_bits(out[3][i]), (want(m, n) + c(m, n)) as f32, "row-major D[{m}][{n}]");
            assert_eq!(f32::from_bits(out[4][(n * 16 + m) as usize]), (want(m, n) + c(m, n)) as f32, "column-major D[{m}][{n}]");
            let half = (out[5][i / 2] >> (16 * (i % 2))) as u16;
            assert_eq!(half, f16_bits(want(m, n)), "f16 sums H[{m}][{n}]");
        }
    }
}

#[test]
fn a_fragment_past_its_array_reads_zeros_there_and_writes_nothing_there() {
    let Some((_emu, device, queue)) = emulator() else { return };
    if !device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    // A whole; B's binding 200 halves of a buffer of 256 whose last 56 hold 1.0 (outside the binding: read as 0); D's
    // binding 200 floats of a buffer of 256 whose last 56 hold 9.0 (outside it: not written)
    let pack = |v: Vec<u16>| -> Vec<u32> { v.chunks(2).map(|p| p[0] as u32 | (p[1] as u32) << 16).collect() };
    let a = |m: i32, k: i32| (m * 3 + k) % 5 - 2;
    let b = |k: i32, n: i32| if k * 16 + n < 200 { (k + 2 * n) % 7 - 3 } else { 0 };
    let a_mem = pack((0..256).map(|i| f16_bits(a(i / 16, i % 16))).collect());
    let b_mem = pack((0..256).map(|i| if i < 200 { f16_bits(b(i / 16, i % 16)) } else { f16_bits(1) }).collect());
    let d_mem: Vec<u32> = (0..256).map(|i| if i < 200 { 0 } else { 9f32.to_bits() }).collect();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("
            enable f16;
            enable wgpu_cooperative_matrix;
            @group(0) @binding(0) var<storage, read_write> a: array<f16>;
            @group(0) @binding(1) var<storage, read_write> b: array<f16>;
            @group(0) @binding(2) var<storage, read_write> d: array<f32>;
            @compute @workgroup_size(32) fn main() {
                let z = 0u;
                let s = 16u;
                let x = coopLoadT<coop_mat16x16<f16, A>>(&a[z], s);
                let y = coopLoadT<coop_mat16x16<f16, B>>(&b[z], s);
                var acc = coop_mat16x16<f32, C>();
                acc = coopMultiplyAdd(x, y, acc);
                coopStoreT(acc, &d[z], s);
            }".into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let make = |words: &[u32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(words), usage });
    let (ab, bb, db) = (make(&a_mem), make(&b_mem), make(&d_mem));
    fn part(buf: &wgpu::Buffer, bytes: u64) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buf, offset: 0, size: wgpu::BufferSize::new(bytes) })
    }
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ab.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: part(&bb, 400) },
            wgpu::BindGroupEntry { binding: 2, resource: part(&db, 800) },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([enc.finish()]);
    let d = read(&device, &queue, &db, 1024);
    for i in 0..200 {
        let (m, n) = (i / 16, i % 16);
        let want = (0..16).map(|k| a(m, k) * b(k, n)).sum::<i32>();
        assert_eq!(f32::from_bits(d[i as usize]), want as f32, "D[{m}][{n}] (B's elements past its binding read as 0)");
    }
    assert!(d[200..].iter().all(|&v| f32::from_bits(v) == 9.0), "nothing written past D's binding");
}

#[test]
fn a_write_larger_than_the_servers_pieces_lands_whole_where_it_was_put() {
    let Some((_emu, device, queue)) = emulator() else { return };
    // 40 MB at an offset: the server takes it in pieces of 16 MB, each sent on to the card as it comes.
    let words = 10 << 20;
    let data: Vec<u32> = (0..words as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (words as u64 + 1024) * 4, usage, mapped_at_creation: false });
    queue.write_buffer(&b, 4096, bytemuck::cast_slice(&data));
    let back = read(&device, &queue, &b, (words + 1024) * 4);
    assert!(back[..1024].iter().all(|&v| v == 0), "before the offset: as it was");
    assert!(back[1024..] == data[..], "the write, whole and in order");
}

#[test]
fn a_buffer_from_the_pool_begins_as_zeros_and_keeps_what_is_written_into_it() {
    let Some((_emu, device, queue)) = emulator() else { return };
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let size = 1 << 20;
    let make = || device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
    // one written and let go: the next of its size is the same buffer on the server, and is zeros
    let a = make();
    queue.write_buffer(&a, 0, &vec![0xabu8; size as usize]);
    assert!(read(&device, &queue, &a, size as usize).iter().all(|&v| v == 0xabababab));
    drop(a);
    let b = make();
    assert!(read(&device, &queue, &b, size as usize).iter().all(|&v| v == 0), "a buffer from the pool is zeros");
    drop(b);
    // written as soon as it is made: the write is after its clear, not under it
    let c = make();
    queue.write_buffer(&c, 4096, &[7u8; 64]);
    let back = read(&device, &queue, &c, size as usize);
    assert!(back[1024..1040].iter().all(|&v| v == 0x07070707) && back[..1024].iter().all(|&v| v == 0) && back[1040..].iter().all(|&v| v == 0));
    // and one used in a dispatch first: cleared at the head of that submission
    drop(c);
    let out = run(&device, &queue, "@group(0) @binding(0) var<storage, read_write> v: array<u32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { v[id.x] = v[id.x] + 1u; }", &[vec![5u32; 256]], 4);
    assert!(out[0].iter().all(|&v| v == 6));
}
