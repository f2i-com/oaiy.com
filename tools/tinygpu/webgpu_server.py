"""tools/tinygpu/webgpu_server.py [SOCKET]: WebGPU's compute on an NVIDIA card a Mac reaches only through tinygrad (TinyGPU).

The card is held by this one process for as long as it runs (each process that opens it resets it, and a reset over
Thunderbolt has taken the card's link down): a client (crates/wgpu-tinygpu, a wgpu backend) connects to SOCKET, a Unix
socket, and asks for what WebGPU's compute does: buffers, writes and reads, kernels, and submissions of dispatches,
copies and clears, run in order on the card's compute queue. Kernels come as CUDA C++ (crates/wgsl-cuda's), compiled
by nvcc (tinygrad's Docker shim on a Mac) to a cubin kept in ~/.cache/tinygpu-webgpu by the source's hash.

    DEV=NV PATH=~/.local/bin:$PATH ~/tinygrad/.venv/bin/python tools/tinygpu/webgpu_server.py ~/.cache/tinygpu-webgpu/server.sock

A request is `<u32 command><u64 length><payload>`; its answer `<u32 status><u64 length><payload>`, status 0 with the
command's result, else 1 with the error's text. Numbers are little-endian. Commands:

    1 HELLO                                         -> JSON: the card (arch, name, memory)
    2 ALLOC   u64 size                              -> u64 buffer (zeroed)
    3 FREE    u64 buffer
    4 WRITE   u64 buffer, u64 offset, bytes         (after all that was submitted)
    5 READ    u64 buffer, u64 offset, u64 size      -> bytes (after all that was submitted)
    6 PROGRAM u32 x, u32 y, u32 z, CUDA source       -> u64 program (its workgroup x by y by z)
    7 SUBMIT  ops                                   (queued; in order)
        u8 1 DISPATCH u64 program, u32 gx, gy, gz, u32 n, n x (u64 buffer, u64 offset)
        u8 2 COPY     u64 source, u64 source offset, u64 destination, u64 destination offset, u64 size
        u8 3 CLEAR    u64 buffer, u64 offset, u64 size
    8 SYNC                                          (until the card has done all that was submitted)
    9 SHUTDOWN                                      (the card released, and the server gone)

It ends by releasing the card (tinygrad's exit: an NVIDIA card's GSP firmware unloaded), at SHUTDOWN, SIGTERM, SIGHUP or
an interrupt: a process that holds the card and is killed leaves its firmware up, and the next one to open it resets
the card, which over Thunderbolt has taken its link down until the enclosure was powered off and on.
"""
import hashlib, json, os, pathlib, signal, socket, struct, subprocess, sys, time, traceback

from tinygrad import Device, dtypes
from tinygrad.device import BufferSpec, TinyELF
from tinygrad.helpers import Target
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.elf import elf_loader

CACHE = pathlib.Path.home() / ".cache" / "tinygpu-webgpu"
ALIGN = 256

# the queue's own copies and clears: kernels on the compute queue, so they keep their place among the dispatches. One
# kernel a translation unit: tinygrad reads a program's registers and stack from the cubin's .nv.info, taking the last
# kernel's of several (a fill run with a copy's 14 registers where it used 16 faulted every SM, and a fault costs the
# card: tinygrad's reset as it is opened again took its Thunderbolt link down)
BUILTINS = {
  "tg_copy4": r"""extern "C" __global__ void tg_copy4(const unsigned* __restrict__ s, unsigned* __restrict__ d, unsigned n) {
  for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n; i += 256u * 1024u) d[i] = s[i];
}""",
  "tg_copy1": r"""extern "C" __global__ void tg_copy1(const unsigned char* __restrict__ s, unsigned char* __restrict__ d, unsigned n) {
  for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n; i += 256u * 1024u) d[i] = s[i];
}""",
  "tg_fill4": r"""extern "C" __global__ void tg_fill4(unsigned* __restrict__ d, unsigned n) {
  for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n; i += 256u * 1024u) d[i] = 0u;
}""",
  "tg_fill1": r"""extern "C" __global__ void tg_fill1(unsigned char* __restrict__ d, unsigned n) {
  for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n; i += 256u * 1024u) d[i] = 0;
}""",
}


def compile_cuda(src: str, arch: str) -> bytes:
  """A translation unit's cubin, from the cache or nvcc."""
  CACHE.mkdir(parents=True, exist_ok=True)
  key = hashlib.sha256((arch + "\0" + src).encode()).hexdigest()[:24]
  cubin = CACHE / f"{key}.cubin"
  if not cubin.exists():
    cu = CACHE / f"{key}.cu"
    cu.write_text(src)
    tmp = CACHE / f"{key}.{os.getpid()}.cubin"
    run = subprocess.run(["nvcc", f"-arch={arch}", "-cubin", "-o", str(tmp), str(cu)], capture_output=True, text=True)
    if run.returncode != 0: raise RuntimeError(f"nvcc: {run.stderr.strip()[-2000:]}")
    tmp.rename(cubin)
  lib = cubin.read_bytes()
  kernels = [s.name for s in elf_loader(lib)[1] if s.name.startswith(".text.")]
  if len(kernels) != 1: raise RuntimeError(f"a cubin of one kernel is what tinygrad loads, and this has {kernels}")
  return lib


class Program:
  def __init__(self, dev, lib: bytes, name: str, workgroup: tuple, vals: int = 0):
    sig = tuple((f"v{i}", i, dtypes.uint32, ()) for i in range(vals))
    self.prg = NVProgram(dev, TinyELF(lib, name, Target(device="NV"), sig))
    self.workgroup = workgroup

  def launch(self, bufs, grid, vals=()):
    # CUDA's blockDim and gridDim are the driver's words in constant bank 0, which tinygrad's launches leave 0
    self.prg.cbuf_0[0:6] = [*self.workgroup, *grid]
    self.prg(*bufs, global_size=grid, local_size=self.workgroup, vals=vals, wait=False)


class Server:
  def __init__(self):
    self.dev = Device["NV"]
    self.arch = self.dev.arch
    self.buffers: dict[int, object] = {}
    self.sizes: dict[int, int] = {}
    self.programs: dict[int, Program] = {}
    self.next_id = 1
    built = {name: Program(self.dev, compile_cuda(src, self.arch), name, (256, 1, 1), 1) for name, src in BUILTINS.items()}
    self.copy4, self.copy1, self.fill4, self.fill1 = built["tg_copy4"], built["tg_copy1"], built["tg_fill4"], built["tg_fill1"]

  def new_id(self) -> int:
    self.next_id += 1
    return self.next_id - 1

  def view(self, buf: int, offset: int, size: int | None = None):
    b = self.buffers[buf]
    if offset == 0 and size is None: return b
    return b.offset(offset, size if size is not None else self.sizes[buf] - offset)

  def copy(self, src, soff, dst, doff, size):
    if size == 0: return
    s, d = self.view(src, soff, size), self.view(dst, doff, size)
    if (soff | doff | size) % 4 == 0: self.copy4.launch((s, d), (min(1024, (size // 4 + 255) // 256), 1, 1), (size // 4,))
    else: self.copy1.launch((s, d), (min(1024, (size + 255) // 256), 1, 1), (size,))

  def clear(self, buf, off, size):
    if size == 0: return
    d = self.view(buf, off, size)
    if (off | size) % 4 == 0: self.fill4.launch((d,), (min(1024, (size // 4 + 255) // 256), 1, 1), (size // 4,))
    else: self.fill1.launch((d,), (min(1024, (size + 255) // 256), 1, 1), (size,))

  # ---- the commands ----

  def hello(self, _p):
    return json.dumps({"arch": self.arch, "name": f"NVIDIA {self.arch} via tinygrad (TinyGPU)", "device": "NV",
                       "memory": int(os.environ.get("TINYGPU_MEMORY", 24 << 30))}).encode()

  def alloc(self, p):
    (size,) = struct.unpack_from("<Q", p)
    rounded = max(ALIGN, (size + ALIGN - 1) // ALIGN * ALIGN)
    h = self.new_id()
    self.buffers[h], self.sizes[h] = self.dev.allocator.alloc(rounded, BufferSpec()), rounded
    self.clear(h, 0, rounded)
    return struct.pack("<Q", h)

  def free(self, p):
    (h,) = struct.unpack_from("<Q", p)
    self.dev.synchronize()
    b = self.buffers.pop(h, None)
    if b is not None: self.dev.allocator.free(b, self.sizes.pop(h), BufferSpec())
    return b""

  def write(self, p):
    h, off = struct.unpack_from("<QQ", p)
    data = memoryview(p)[16:]
    if len(data):
      self.dev.synchronize()
      self.dev.allocator._copyin(self.view(h, off, len(data)), data)
    return b""

  def read(self, p):
    h, off, size = struct.unpack_from("<QQQ", p)
    out = memoryview(bytearray(size))
    if size:
      self.dev.synchronize()
      self.dev.allocator._copyout(out, self.view(h, off, size))
    return bytes(out)

  def program(self, p):
    x, y, z = struct.unpack_from("<III", p)
    src = bytes(p[12:]).decode()
    lib = compile_cuda(src, self.arch)
    h = self.new_id()
    self.programs[h] = Program(self.dev, lib, "oaiy_main", (x, y, z))
    return struct.pack("<Q", h)

  def submit(self, p):
    p, at = memoryview(p), 0
    while at < len(p):
      op = p[at]; at += 1
      if op == 1:
        prog, gx, gy, gz, n = struct.unpack_from("<QIIII", p, at); at += 24
        bufs = []
        for _ in range(n):
          b, off = struct.unpack_from("<QQ", p, at); at += 16
          bufs.append(self.view(b, off))
        if gx and gy and gz: self.programs[prog].launch(bufs, (gx, gy, gz))
      elif op == 2:
        s, so, d, do, size = struct.unpack_from("<QQQQQ", p, at); at += 40
        self.copy(s, so, d, do, size)
      elif op == 3:
        b, off, size = struct.unpack_from("<QQQ", p, at); at += 24
        self.clear(b, off, size)
      else: raise RuntimeError(f"unknown op {op}")
    return b""

  def sync(self, _p):
    self.dev.synchronize()
    return b""

  def shutdown(self, _p):
    raise SystemExit(0)

  COMMANDS = {1: hello, 2: alloc, 3: free, 4: write, 5: read, 6: program, 7: submit, 8: sync, 9: shutdown}


def recv_exact(conn: socket.socket, n: int) -> bytes:
  buf = bytearray(n)
  view, got = memoryview(buf), 0
  while got < n:
    k = conn.recv_into(view[got:], n - got)
    if k == 0: raise ConnectionError("client closed")
    got += k
  return bytes(buf)


def serve(path: str):
  # (a stop by signal is a normal exit, so tinygrad's own exit releases the card)
  for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *_: sys.exit(0))
  server = Server()
  if os.path.exists(path): os.unlink(path)
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.bind(path)
  os.chmod(path, 0o600)
  sock.listen(4)
  print(f"tinygpu-webgpu: {server.arch} ready on {path}", flush=True)
  while True:
    conn, _ = sock.accept()
    try:
      while True:
        cmd, n = struct.unpack("<IQ", recv_exact(conn, 12))
        payload = recv_exact(conn, n) if n else b""
        if cmd == 9: conn.sendall(struct.pack("<IQ", 0, 0))
        try:
          out, status = Server.COMMANDS[cmd](server, payload), 0
        except Exception as e:
          out, status = f"{type(e).__name__}: {e}".encode(), 1
          traceback.print_exc()
        conn.sendall(struct.pack("<IQ", status, len(out)) + out)
    except (ConnectionError, BrokenPipeError):
      pass
    finally:
      conn.close()


if __name__ == "__main__":
  try: serve(sys.argv[1] if len(sys.argv) > 1 else str(CACHE / "server.sock"))
  except KeyboardInterrupt: pass
  print("tinygpu-webgpu: the card released", flush=True)
