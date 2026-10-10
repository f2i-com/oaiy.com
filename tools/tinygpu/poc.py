"""One WGSL kernel on the eGPU: wgsl-cuda's CUDA, nvcc's cubin (in Docker), loaded and run by tinygrad's NV runtime.

    DEV=NV ~/tinygrad/.venv/bin/python tools/tinygpu/poc.py target/release/wgsl-cuda
"""
import hashlib, json, os, pathlib, struct, subprocess, sys
from tinygrad import Device
from tinygrad.device import BufferSpec, TinyELF
from tinygrad.helpers import Target
from tinygrad.runtime.ops_nv import NVProgram

WGSL = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < p.x) { b[i] = a[i] * 2.0 + f32(p.y); }
}
"""

cache = pathlib.Path.home() / ".cache" / "tinygpu-webgpu"
cache.mkdir(parents=True, exist_ok=True)
src = cache / "poc.wgsl"; src.write_text(WGSL)
cu = cache / "poc.cu"
meta = json.loads(subprocess.run([sys.argv[1], str(src), str(cu)], capture_output=True, text=True, check=True).stderr.strip().splitlines()[-1])
cubin = cache / "poc.cubin"
subprocess.run(["nvcc", "-arch=sm_89", "-cubin", "-o", str(cubin), str(cu)], check=True)
dev = Device["NV"]
prg = NVProgram(dev, TinyELF(cubin.read_bytes(), "oaiy_main", Target(device="NV"), ()))
n = 1000
a = dev.allocator.alloc(n * 4, BufferSpec()); b = dev.allocator.alloc(n * 4, BufferSpec()); p = dev.allocator.alloc(16, BufferSpec())
dev.allocator._copyin(a, memoryview(struct.pack(f"{n}f", *[float(i) for i in range(n)])))
dev.allocator._copyin(p, memoryview(struct.pack("4I", n, 7, 0, 0)))
wg = meta["workgroup_size"]
grid = ((n + wg[0] - 1) // wg[0], 1, 1)
prg.cbuf_0[0:6] = [*wg, *grid]  # CUDA's blockDim and gridDim: the driver's words, which tinygrad's launches leave 0
prg(a, b, p, global_size=grid, local_size=tuple(wg), wait=True)
out = memoryview(bytearray(n * 4)); dev.allocator._copyout(out, b)
got = struct.unpack(f"{n}f", out)
bad = [i for i in range(n) if got[i] != i * 2.0 + 7]
print("bindings", meta["bindings"], "workgroup", wg)
print("first", got[:5], "last", got[-2:], "->", "OK" if not bad else f"{len(bad)} WRONG, e.g. [{bad[0]}] = {got[bad[0]]}")
