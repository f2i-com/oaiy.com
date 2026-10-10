//! A wgpu program as any is written, on the TinyGPU adapter: `cargo run --release -p wgpu-tinygpu --example smoke`
//! (with tools/tinygpu/webgpu_server.py running).
use wgpu::util::DeviceExt;

fn main() {
    let adapter = wgpu_tinygpu::adapter(&wgpu_tinygpu::default_socket()).unwrap_or_else(|e| panic!("{e}"));
    println!("adapter: {:?}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let n = 1 << 20;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
    let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[n as u32, 7, 0, 0]), usage: wgpu::BufferUsages::UNIFORM });
    let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("smoke"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> a: array<f32>;
             @group(0) @binding(1) var<storage, read_write> b: array<f32>;
             @group(0) @binding(2) var<uniform> p: vec4<u32>;
             var<workgroup> tile: array<f32, 256>;
             @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
               let i = id.x;
               if (i < p.x) { tile[li] = a[i]; }
               workgroupBarrier();
               if (i < p.x) { b[i] = tile[255u - li] * 2.0 + f32(p.y); }
             }"
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("smoke"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: p.as_entire_binding() },
        ],
    });
    let t = std::time::Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(n as u32 / 256, 1, 1);
    }
    enc.copy_buffer_to_buffer(&b, 0, &read, 0, n as u64 * 4);
    queue.submit([enc.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let got: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
    read.unmap();
    let bad = (0..n).filter(|&i| got[i] != ((i / 256) * 256 + 255 - i % 256) as f32 * 2.0 + 7.0).count();
    println!("{} values in {:.1} ms (with the readback): {}", n, t.elapsed().as_secs_f64() * 1e3, if bad == 0 { "OK".to_string() } else { format!("{bad} WRONG") });
}
