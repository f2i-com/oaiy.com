"""tools/tinygpu/webgpu_server.py [--emulate] [SOCKET]: WebGPU's compute on an NVIDIA card a Mac reaches only through
tinygrad (TinyGPU).

The card is held by this one process for as long as it runs (each process that opens it resets it, and a reset over
Thunderbolt has taken the card's link down): a client (crates/wgpu-tinygpu, a wgpu backend) connects to SOCKET, a Unix
socket, and asks for what WebGPU's compute does: buffers, writes and reads, kernels, and submissions of dispatches,
copies and clears, run in order on the card's compute queue. Kernels come as CUDA C++ (crates/wgsl-cuda's), compiled
by nvcc (tinygrad's Docker shim on a Mac) to a cubin kept in ~/.cache/tinygpu-webgpu by the source's hash.

    DEV=NV PATH=~/.local/bin:$PATH ~/tinygrad/.venv/bin/python tools/tinygpu/webgpu_server.py ~/.cache/tinygpu-webgpu/server.sock

With --emulate there is no card: each kernel is compiled for the CPU (clang++, with tools/tinygpu/emu/cuda_emu.h) and
run there, workgroups on every core, so a translation and a program are checked before they go to the card (a kernel
that faults costs the card its link). No tinygrad is needed for it.

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

Clients connect at once, each its own; their requests run one at a time, and a client's buffers are freed when it
goes.

It ends by releasing the card (tinygrad's exit: an NVIDIA card's GSP firmware unloaded), at SHUTDOWN, SIGTERM, SIGHUP or
an interrupt: a process that holds the card and is killed leaves its firmware up, and the next one to open it resets
the card, which over Thunderbolt has taken its link down until the enclosure was powered off and on.
"""
from __future__ import annotations

import ctypes, hashlib, json, mmap, os, pathlib, signal, socket, struct, subprocess, sys, threading, traceback
from concurrent.futures import ThreadPoolExecutor

CACHE = pathlib.Path.home() / ".cache" / "tinygpu-webgpu"
EMU = pathlib.Path(__file__).resolve().parent / "emu"
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


def cached(suffix: str, key: bytes, build) -> pathlib.Path:
  """The file of `key`'s hash in the cache, made by build(path) where it is not there yet."""
  CACHE.mkdir(parents=True, exist_ok=True)
  out = CACHE / f"{hashlib.sha256(key).hexdigest()[:24]}{suffix}"
  if not out.exists():
    tmp = out.with_name(f"{out.stem}.{os.getpid()}{suffix}")
    build(tmp)
    tmp.rename(out)
  return out


def run_compiler(argv: list, what: str):
  run = subprocess.run(argv, capture_output=True, text=True)
  if run.returncode != 0: raise RuntimeError(f"{what}: {run.stderr.strip()[-3000:]}")


# ---- the card, through tinygrad ----

def compile_cuda(src: str, arch: str) -> bytes:
  """A translation unit's cubin, from the cache or nvcc."""
  from tinygrad.runtime.support.elf import elf_loader
  def build(tmp: pathlib.Path):
    cu = tmp.with_suffix(".cu")
    cu.write_text(src)
    run_compiler(["nvcc", f"-arch={arch}", "-cubin", "-o", str(tmp), str(cu)], "nvcc")
  lib = cached(".cubin", (arch + "\0" + src).encode(), build).read_bytes()
  kernels = [s.name for s in elf_loader(lib)[1] if s.name.startswith(".text.")]
  if len(kernels) != 1: raise RuntimeError(f"a cubin of one kernel is what tinygrad loads, and this has {kernels}")
  return lib


class Program:
  def __init__(self, dev, lib: bytes, name: str, workgroup: tuple, vals: int = 0):
    from tinygrad import dtypes
    from tinygrad.device import TinyELF
    from tinygrad.helpers import Target
    from tinygrad.runtime.ops_nv import NVProgram
    sig = tuple((f"v{i}", i, dtypes.uint32, ()) for i in range(vals))
    self.prg = NVProgram(dev, TinyELF(lib, name, Target(device="NV"), sig))
    self.workgroup = workgroup

  def args(self, bufs, grid, vals=()):
    # CUDA's blockDim and gridDim are the driver's words in constant bank 0, which tinygrad's launches leave 0 (the
    # words are copied into each launch's arguments as they are filled)
    self.prg.cbuf_0[0:6] = [*self.workgroup, *grid]
    return self.prg.fill_kernargs(tuple(bufs), tuple(vals))


class Card:
  """The NVIDIA card tinygrad holds (DEV=NV). A buffer is (tinygrad's buffer, its size).

  A submission's kernels go in one queue of the card's (one doorbell), each launched as the one before it ends, as
  tinygrad's own graphs chain them; a write goes by the copy queue, which waits for what was submitted before it, so
  neither waits for the card. A freed buffer goes to tinygrad's cache, from where the next of its size reuses it in the
  queue's order."""
  BATCH = 256      # kernels a queue, before it goes to the card and the next starts
  ARGS_RING = 4096  # launches between waits: their arguments are a ring of tinygrad's (16 MB), reused as it wraps

  def __init__(self):
    from tinygrad import Device
    from tinygrad.device import BufferSpec
    self.dev, self.spec = Device["NV"], BufferSpec()
    self.arch = self.dev.arch
    self.q, self.queued, self.since_sync = None, 0, 0
    built = {name: Program(self.dev, compile_cuda(src, self.arch), name, (256, 1, 1), 1) for name, src in BUILTINS.items()}
    self.copy4, self.copy1, self.fill4, self.fill1 = built["tg_copy4"], built["tg_copy1"], built["tg_fill4"], built["tg_fill1"]

  def begin(self):
    if self.q is None:
      self.q = self.dev.hw_compute_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()

  def end(self):
    if self.q is not None and self.queued: self.q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
    self.q, self.queued = None, 0

  def launch(self, prog: Program, bufs, grid, vals=()):
    inside = self.q is not None   # (a submission's, else a launch of its own: an ALLOC's clear)
    if self.since_sync >= self.ARGS_RING: self.sync()
    self.begin()
    self.q.exec(prog.prg, prog.args(bufs, grid, vals), grid, prog.workgroup)
    self.queued, self.since_sync = self.queued + 1, self.since_sync + 1
    if not inside: self.end()
    elif self.queued >= self.BATCH:
      self.end()
      self.begin()

  def info(self) -> dict:
    return {"arch": self.arch, "name": f"NVIDIA {self.arch} via tinygrad (TinyGPU)", "device": "NV",
            "memory": int(os.environ.get("TINYGPU_MEMORY", 24 << 30))}

  @staticmethod
  def view(buf, offset: int, size: int):
    b, whole = buf
    return b if offset == 0 and size == whole else b.offset(offset, size)

  def alloc(self, size: int):
    buf = (self.dev.allocator.alloc(size, self.spec), size)
    self.clear(buf, 0, size)
    return buf

  def free(self, buf):
    self.dev.allocator.free(buf[0], buf[1], self.spec)

  def write(self, buf, off: int, data: memoryview):
    self.dev.allocator._copyin(self.view(buf, off, len(data)), data)

  def read(self, buf, off: int, size: int) -> bytes:
    out = memoryview(bytearray(size))
    self.dev.allocator._copyout(out, self.view(buf, off, size))
    self.since_sync = 0
    return bytes(out)

  def program(self, workgroup: tuple, src: str):
    return Program(self.dev, compile_cuda(src, self.arch), "oaiy_main", workgroup)

  def dispatch(self, prog: Program, bufs: list, grid: tuple):
    self.launch(prog, [self.view(b, off, b[1] - off) for b, off in bufs], grid)

  def copy(self, src, soff: int, dst, doff: int, size: int):
    s, d = self.view(src, soff, size), self.view(dst, doff, size)
    if (soff | doff | size) % 4 == 0: self.launch(self.copy4, (s, d), (min(1024, (size // 4 + 255) // 256), 1, 1), (size // 4,))
    else: self.launch(self.copy1, (s, d), (min(1024, (size + 255) // 256), 1, 1), (size,))

  def clear(self, buf, off: int, size: int):
    d = self.view(buf, off, size)
    if (off | size) % 4 == 0: self.launch(self.fill4, (d,), (min(1024, (size // 4 + 255) // 256), 1, 1), (size // 4,))
    else: self.launch(self.fill1, (d,), (min(1024, (size + 255) // 256), 1, 1), (size,))

  def sync(self):
    self.end()
    self.dev.synchronize()
    self.since_sync = 0


# ---- no card: the kernels on the CPU ----

class Emulator:
  """The card's work on the CPU: memory the process's own, each kernel compiled by clang++ with emu/cuda_emu.h and its
  workgroups shared out among the cores (ctypes lets go of Python's lock while a kernel runs). Each piece of work is
  done before the next is taken, so the queue is always done. A buffer is (its address, its size)."""
  STACK = 256 << 10   # an invocation's stack
  MAX_INVOCATIONS = 1024

  def __init__(self):
    libc = ctypes.CDLL(None, use_errno=True)
    self.mmap, self.munmap, self.mprotect = libc.mmap, libc.munmap, libc.mprotect
    self.mmap.restype = ctypes.c_void_p
    self.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    self.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    self.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    self.workers = os.cpu_count() or 4
    self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix="emu")
    self.local = threading.local()

  def info(self) -> dict:
    return {"arch": "cpu", "name": "the card's work on the CPU (tinygpu-webgpu --emulate)", "device": "CPU",
            "memory": int(os.environ.get("TINYGPU_MEMORY", 16 << 30))}

  def map(self, size: int) -> int:
    # (anonymous memory: zeroed, and given pages as they are touched)
    at = self.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_ANON | mmap.MAP_PRIVATE, -1, 0)
    if at in (None, ctypes.c_void_p(-1).value): raise MemoryError(f"mmap of {size} bytes: errno {ctypes.get_errno()}")
    return at

  def alloc(self, size: int):
    return (self.map(size), size)

  def free(self, buf):
    self.munmap(buf[0], buf[1])

  def write(self, buf, off: int, data: memoryview):
    ctypes.memmove(buf[0] + off, bytes(data), len(data))

  def read(self, buf, off: int, size: int) -> bytes:
    return ctypes.string_at(buf[0] + off, size)

  def program(self, workgroup: tuple, src: str):
    header = (EMU / "cuda_emu.h").read_bytes()
    barriers = "__syncthreads()" in src
    def build(tmp: pathlib.Path):
      cpp = tmp.with_suffix(".cpp")
      cpp.write_text(f"{src}\nEMU_LAUNCHER(oaiy_main, {'true' if barriers else 'false'})\n")
      run_compiler(["clang++", "-std=c++20", "-O2", "-fwrapv", "-shared", "-fPIC", "-w", "-include", str(EMU / "cuda_emu.h"),
                    "-I", str(EMU), str(cpp), "-o", str(tmp)], "clang++")
    lib = ctypes.CDLL(str(cached(".dylib", b"cpu\0" + header + b"\0" + src.encode(), build)))
    launch = lib.emu_launch
    launch.restype = None
    launch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint64,
                       ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    return (lib, launch, workgroup)

  def stacks(self) -> int:
    """This thread's room for a workgroup's stacks, each with a guard page under it."""
    at = getattr(self.local, "stacks", None)
    if at is None:
      at = self.map(self.STACK * self.MAX_INVOCATIONS)
      for i in range(self.MAX_INVOCATIONS): self.mprotect(at + i * self.STACK, mmap.PAGESIZE, 0)   # (PROT_NONE)
      self.local.stacks = at
    return at

  def run(self, launch, args, grid, first, count):
    launch(args, *grid, first, count, self.stacks(), self.STACK)

  def dispatch(self, prog, bufs: list, grid: tuple):
    _, launch, workgroup = prog
    if workgroup[0] * workgroup[1] * workgroup[2] > self.MAX_INVOCATIONS: raise RuntimeError(f"a workgroup of {workgroup}")
    args = (ctypes.c_void_p * max(1, len(bufs)))(*[b[0] + off for b, off in bufs])
    total = grid[0] * grid[1] * grid[2]
    parts = min(total, self.workers * 4)
    if parts <= 1: return self.run(launch, args, grid, 0, total)
    edges = [total * i // parts for i in range(parts + 1)]
    for f in [self.pool.submit(self.run, launch, args, grid, a, b - a) for a, b in zip(edges, edges[1:])]: f.result()

  def copy(self, src, soff: int, dst, doff: int, size: int):
    ctypes.memmove(dst[0] + doff, src[0] + soff, size)

  def clear(self, buf, off: int, size: int):
    ctypes.memset(buf[0] + off, 0, size)

  def begin(self): pass
  def end(self): pass
  def sync(self): pass


# ---- the protocol ----

class Server:
  def __init__(self, card):
    self.card = card
    self.buffers: dict = {}
    self.programs: dict = {}
    self.next_id = 1

  def new_id(self) -> int:
    self.next_id += 1
    return self.next_id - 1

  def hello(self, _p):
    return json.dumps(self.card.info()).encode()

  def alloc(self, p):
    (size,) = struct.unpack_from("<Q", p)
    h = self.new_id()
    self.buffers[h] = self.card.alloc(max(ALIGN, (size + ALIGN - 1) // ALIGN * ALIGN))
    return struct.pack("<Q", h)

  def free(self, p):
    (h,) = struct.unpack_from("<Q", p)
    b = self.buffers.pop(h, None)
    if b is not None: self.card.free(b)
    return b""

  def write(self, p):
    h, off = struct.unpack_from("<QQ", p)
    data = memoryview(p)[16:]
    if len(data): self.card.write(self.buffers[h], off, data)
    return b""

  def read(self, p):
    h, off, size = struct.unpack_from("<QQQ", p)
    return self.card.read(self.buffers[h], off, size) if size else b""

  def program(self, p):
    x, y, z = struct.unpack_from("<III", p)
    h = self.new_id()
    self.programs[h] = self.card.program((x, y, z), bytes(p[12:]).decode())
    return struct.pack("<Q", h)

  def submit(self, p):
    p, at = memoryview(p), 0
    self.card.begin()
    try: self.ops(p)
    finally: self.card.end()
    return b""

  def ops(self, p):
    at = 0
    while at < len(p):
      op = p[at]; at += 1
      if op == 1:
        prog, gx, gy, gz, n = struct.unpack_from("<QIIII", p, at); at += 24
        bufs = []
        for _ in range(n):
          b, off = struct.unpack_from("<QQ", p, at); at += 16
          bufs.append((self.buffers[b], off))
        if gx and gy and gz: self.card.dispatch(self.programs[prog], bufs, (gx, gy, gz))
      elif op == 2:
        s, so, d, do, size = struct.unpack_from("<QQQQQ", p, at); at += 40
        if size: self.card.copy(self.buffers[s], so, self.buffers[d], do, size)
      elif op == 3:
        b, off, size = struct.unpack_from("<QQQ", p, at); at += 24
        if size: self.card.clear(self.buffers[b], off, size)
      else: raise RuntimeError(f"unknown op {op}")

  def sync(self, _p):
    self.card.sync()
    return b""

  COMMANDS = {1: hello, 2: alloc, 3: free, 4: write, 5: read, 6: program, 7: submit, 8: sync}


def recv_exact(conn: socket.socket, n: int) -> bytes:
  buf = bytearray(n)
  view, got = memoryview(buf), 0
  while got < n:
    k = conn.recv_into(view[got:], n - got)
    if k == 0: raise ConnectionError("client closed")
    got += k
  return bytes(buf)


def handle(server: Server, conn: socket.socket, lock: threading.Lock, closing: threading.Event):
  """One client's requests, each run with the card to itself; the buffers it did not free are freed as it goes."""
  owned = set()
  try:
    while True:
      cmd, n = struct.unpack("<IQ", recv_exact(conn, 12))
      payload = recv_exact(conn, n) if n else b""
      if cmd == 9:
        conn.sendall(struct.pack("<IQ", 0, 0))
        os.kill(os.getpid(), signal.SIGTERM)   # (the main thread ends the server: tinygrad's exit is the process's)
        return
      with lock:
        if closing.is_set(): return
        try:
          if cmd not in Server.COMMANDS: raise RuntimeError(f"unknown command {cmd}")
          out, status = Server.COMMANDS[cmd](server, payload), 0
          if cmd == 2: owned.add(struct.unpack("<Q", out)[0])
          elif cmd == 3: owned.discard(struct.unpack_from("<Q", payload)[0])
        except Exception as e:
          out, status = f"{type(e).__name__}: {e}".encode(), 1
          traceback.print_exc()
      conn.sendall(struct.pack("<IQ", status, len(out)) + out)
  except OSError:   # (the client gone: ConnectionError, BrokenPipeError)
    pass
  finally:
    conn.close()
    with lock:
      if not closing.is_set():
        for h in owned: server.free(struct.pack("<Q", h))


def serve(path: str, emulate: bool):
  # (a stop by signal is a normal exit, so tinygrad's own exit releases the card)
  for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *_: sys.exit(0))
  server = Server(Emulator() if emulate else Card())
  if os.path.exists(path): os.unlink(path)
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.bind(path)
  os.chmod(path, 0o600)
  sock.listen(16)
  print(f"tinygpu-webgpu: {server.card.info()['name']} ready on {path}", flush=True)
  # clients at once, each on a thread of its own; their requests one at a time
  lock, closing = threading.Lock(), threading.Event()
  try:
    while True:
      conn, _ = sock.accept()
      threading.Thread(target=handle, args=(server, conn, lock, closing), daemon=True).start()
  finally:
    closing.set()
    lock.acquire(timeout=60)   # (a request under way ends first: the card is let go between requests)
    sock.close()


if __name__ == "__main__":
  argv = sys.argv[1:]
  emulate = "--emulate" in argv
  argv = [a for a in argv if a != "--emulate"]
  CACHE.mkdir(parents=True, exist_ok=True)
  try: serve(argv[0] if argv else str(CACHE / ("emulator.sock" if emulate else "server.sock")), emulate)
  except KeyboardInterrupt: pass
  print("tinygpu-webgpu: " + ("the emulator ended" if emulate else "the card released"), flush=True)
