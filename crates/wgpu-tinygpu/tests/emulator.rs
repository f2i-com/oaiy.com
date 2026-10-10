//! WGSL's meaning, kept by the translation to CUDA: kernels run through the adapter on the server's emulator
//! (`tools/tinygpu/webgpu_server.py --emulate`: each kernel compiled for the CPU), so no card is needed. Skipped where
//! the emulator cannot start (no python3 or clang++).

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

/// The emulator, for as long as the test holds it.
struct Emulator {
    child: Child,
    socket: PathBuf,
}

impl Drop for Emulator {
    fn drop(&mut self) {
        let _ = Command::new("kill").arg("-TERM").arg(self.child.id().to_string()).status();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

fn emulator() -> Option<(Emulator, wgpu::Device, wgpu::Queue)> {
    let server = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/tinygpu/webgpu_server.py");
    static STARTED: AtomicUsize = AtomicUsize::new(0);
    let socket = std::env::temp_dir().join(format!("tinygpu-emu-{}-{}.sock", std::process::id(), STARTED.fetch_add(1, Ordering::Relaxed)));
    let child = Command::new("python3").arg(&server).arg("--emulate").arg(&socket).stdout(Stdio::null()).spawn().ok()?;
    let emu = Emulator { child, socket };
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
