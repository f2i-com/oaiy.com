"""A smoke test of webgpu_server.py from Python: a buffer zeroed, written, a kernel translated by wgsl-cuda run on it,
then a copy and a clear, read back.  python3 tools/tinygpu/client_test.py target/release/wgsl-cuda [SOCKET]"""
import json, os, pathlib, socket, struct, subprocess, sys, tempfile

sock_path = sys.argv[2] if len(sys.argv) > 2 else str(pathlib.Path.home() / ".cache" / "tinygpu-webgpu" / "server.sock")
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(sock_path)

def call(cmd, payload=b""):
  s.sendall(struct.pack("<IQ", cmd, len(payload)) + payload)
  hdr = b""
  while len(hdr) < 12: hdr += s.recv(12 - len(hdr))
  status, n = struct.unpack("<IQ", hdr)
  data = b""
  while len(data) < n: data += s.recv(n - len(data))
  if status: raise RuntimeError(data.decode())
  return data

print("hello:", json.loads(call(1)))
WGSL = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < p.x) { b[i] = a[i] * 2.0 + f32(p.y); }
}"""
with tempfile.TemporaryDirectory() as d:
  pathlib.Path(d, "k.wgsl").write_text(WGSL)
  run = subprocess.run([sys.argv[1], f"{d}/k.wgsl", f"{d}/k.cu"], capture_output=True, text=True, check=True)
  meta = json.loads(run.stderr.strip().splitlines()[-1]); cu = pathlib.Path(d, "k.cu").read_text()
n = 100_000
A = struct.unpack("<Q", call(2, struct.pack("<Q", n * 4)))[0]
B = struct.unpack("<Q", call(2, struct.pack("<Q", n * 4)))[0]
P = struct.unpack("<Q", call(2, struct.pack("<Q", 16)))[0]
zeros = struct.unpack(f"<{n}f", call(5, struct.pack("<QQQ", B, 0, n * 4)))
print("new buffer zeroed:", all(v == 0.0 for v in zeros))
call(4, struct.pack("<QQ", A, 0) + struct.pack(f"<{n}f", *[float(i) for i in range(n)]))
call(4, struct.pack("<QQ", P, 0) + struct.pack("<4I", n, 7, 0, 0))
prog = struct.unpack("<Q", call(6, struct.pack("<III", *meta["workgroup_size"]) + cu.encode()))[0]
grid = (n + 63) // 64
ops = struct.pack("<BQIIII", 1, prog, grid, 1, 1, 3) + b"".join(struct.pack("<QQ", h, 0) for h in (A, B, P))
ops += struct.pack("<BQQQQQ", 2, B, 0, A, 4 * 10, 4 * 5)       # copy b[0..5] over a[10..15]
ops += struct.pack("<BQQQ", 3, B, 4 * 100, 4 * 3)               # clear b[100..103]
call(7, ops); call(8)
b = struct.unpack(f"<{n}f", call(5, struct.pack("<QQQ", B, 0, n * 4)))
a = struct.unpack("<20f", call(5, struct.pack("<QQQ", A, 0, 80)))
want = [i * 2.0 + 7 for i in range(n)]; want[100:103] = [0.0] * 3
bad = [i for i in range(n) if b[i] != want[i]]
print("dispatch + clear:", "OK" if not bad else f"{len(bad)} wrong, e.g. [{bad[0]}] {b[bad[0]]} want {want[bad[0]]}")
print("copy:", "OK" if list(a[10:15]) == [7.0, 9.0, 11.0, 13.0, 15.0] and a[9] == 9.0 and a[15] == 15.0 else f"wrong {a[8:16]}")
for h in (A, B, P): call(3, struct.pack("<Q", h))
