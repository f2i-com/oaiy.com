"""tools/tinygpu/wgpu_py_smoke.py: Python's wgpu (pip install wgpu numpy) on the TinyGPU card, through webgpu.h.

    WGPU_LIB_PATH=target/release/libwebgpu_tinygpu.dylib python tools/tinygpu/wgpu_py_smoke.py

A program as any wgpu one is written: an adapter, a WGSL kernel with workgroup memory and a barrier, buffers in and
out, a dispatch, the answer read back and checked. With TINYGPU_SOCKET at the server's emulator, it runs on the CPU.
"""
import time

import numpy as np
import wgpu

SHADER = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
var<workgroup> tile: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = id.x;
  if (i < p.x) { tile[li] = a[i]; }
  workgroupBarrier();
  if (i < p.x) { b[i] = tile[255u - li] * 2.0 + f32(p.y); }
}
"""

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print("adapter:", adapter.info["device"], "|", adapter.info["description"])
device = adapter.request_device_sync()
n = 1 << 20
a = np.arange(n, dtype=np.float32)
buf_a = device.create_buffer_with_data(data=a, usage=wgpu.BufferUsage.STORAGE)
buf_b = device.create_buffer(size=a.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
buf_p = device.create_buffer_with_data(data=np.array([n, 7, 0, 0], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM)
pipeline = device.create_compute_pipeline(layout="auto", compute={"module": device.create_shader_module(code=SHADER), "entry_point": "main"})
group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
  {"binding": 0, "resource": {"buffer": buf_a}}, {"binding": 1, "resource": {"buffer": buf_b}}, {"binding": 2, "resource": {"buffer": buf_p}}])
t = time.time()
enc = device.create_command_encoder()
pass_ = enc.begin_compute_pass()
pass_.set_pipeline(pipeline)
pass_.set_bind_group(0, group)
pass_.dispatch_workgroups(n // 256)
pass_.end()
device.queue.submit([enc.finish()])
b = np.frombuffer(device.queue.read_buffer(buf_b), dtype=np.float32)
i = np.arange(n)
want = ((i // 256) * 256 + 255 - i % 256).astype(np.float32) * 2 + 7
bad = int((b != want).sum())
print(f"{n} values in {(time.time() - t) * 1e3:.1f} ms (with the readback):", "OK" if bad == 0 else f"{bad} WRONG")
