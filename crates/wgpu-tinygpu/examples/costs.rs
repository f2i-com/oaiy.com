//! What the TinyGPU adapter's work costs, apart from the kernels' own time: a dispatch in a long command buffer, a
//! submission, a write, a read, and the bandwidth of each. `cargo run --release -p wgpu-tinygpu --example costs`
//! (with tools/tinygpu/webgpu_server.py running; TINYGPU_SOCKET for another).
use std::time::Instant;
use wgpu::util::DeviceExt;

fn main() {
    let adapter = wgpu_tinygpu::adapter(&wgpu_tinygpu::default_socket()).unwrap_or_else(|e| panic!("{e}"));
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> v: array<u32>;
            @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { v[id.x] = v[id.x] + 1u; }".into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &[0u8; 256], usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: v.as_entire_binding() }] });
    let wait = || device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let dispatches = |n: u32, per_submit: u32| {
        let t = Instant::now();
        for _ in 0..n / per_submit {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                for _ in 0..per_submit {
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            queue.submit([enc.finish()]);
        }
        wait();
        t.elapsed().as_secs_f64()
    };
    dispatches(64, 64);
    let s = dispatches(2048, 1024);
    println!("a dispatch, 1024 to a submission: {:.1} us", s / 2048.0 * 1e6);
    let s = dispatches(256, 1);
    println!("a submission of one dispatch: {:.1} us", s / 256.0 * 1e6);

    for size in [4usize << 10, 1 << 20, 64 << 20] {
        let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: size as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let data = vec![7u8; size];
        let reps = if size > (1 << 20) { 4 } else { 64 };
        let t = Instant::now();
        for _ in 0..reps {
            queue.write_buffer(&b, 0, &data);
        }
        wait();
        let w = t.elapsed().as_secs_f64() / reps as f64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: size as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let t = Instant::now();
        for _ in 0..reps {
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&b, 0, &staging, 0, size as u64);
            queue.submit([enc.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
            wait();
            assert_eq!(staging.slice(..).get_mapped_range().unwrap()[size - 1], 7);
            staging.unmap();
        }
        let r = t.elapsed().as_secs_f64() / reps as f64;
        let mb = size as f64 / 1e6;
        println!("{:>9} bytes: a write {:.2} ms ({:.0} MB/s), a copy and read back {:.2} ms ({:.0} MB/s)", size, w * 1e3, mb / w, r * 1e3, mb / r);
    }
}
