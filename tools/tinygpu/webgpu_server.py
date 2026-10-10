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

    1 HELLO                                         -> JSON: the card (arch, name, memory) and the protocol's version
    2 ALLOC   u64 size                              -> u64 buffer (zeroed)
    3 FREE    u64 buffer
    4 WRITE   u64 buffer, u64 offset, bytes         (after all that was submitted)
    5 READ    u64 buffer, u64 offset, u64 size      -> bytes (after all that was submitted)
    6 PROGRAM u32 x, u32 y, u32 z, u32 n, u32 flags, CUDA source  -> u64 program (its workgroup x by y by z; after
                                                       its n pointers it takes the scratch where flags' bit 0 says,
                                                       then n uints, each binding's size)
    7 SUBMIT  ops                                   (queued; in order)
        u8 1 DISPATCH u64 program, u32 gx, gy, gz, u32 n, n x (u64 buffer, u64 offset, u64 size)
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

import ctypes, hashlib, json, mmap, os, pathlib, signal, socket, struct, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

CACHE = pathlib.Path.home() / ".cache" / "tinygpu-webgpu"
EMU = pathlib.Path(__file__).resolve().parent / "emu"
ALIGN = 256
PROTOCOL = 3   # the requests as below; a client of another version is told so at HELLO (crates/wgpu-tinygpu's PROTOCOL)
SCRATCH = 160 * 64 * 1024   # a kernel's scratch, where it takes one (wgsl-cuda's SCRATCH_BYTES: 1 KB a warp)

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
  """A translation unit's cubin, from the cache or nvcc (its name, the source's hash, is the program's label)."""
  from tinygrad.runtime.support.elf import elf_loader
  def build(tmp: pathlib.Path):
    cu = tmp.with_suffix(f".{threading.get_ident()}.cu")
    cu.write_text(src)
    run_compiler(["nvcc", f"-arch={arch}", "-cubin", "-o", str(tmp), str(cu)], "nvcc")
    cu.rename(tmp.with_name(tmp.name.split(".")[0] + ".cu"))   # (beside its cubin, as it will be named)
  path = cached(".cubin", (arch + "\0" + src).encode(), build)
  compile_cuda.last = path.stem
  lib = path.read_bytes()
  kernels = [s.name for s in elf_loader(lib)[1] if s.name.startswith(".text.")]
  if len(kernels) != 1: raise RuntimeError(f"a cubin of one kernel is what tinygrad loads, and this has {kernels}")
  return lib


class Program:
  def __init__(self, dev, lib: bytes, name: str, workgroup: tuple, vals: int = 0, scratch: bool = False):
    from tinygrad import dtypes
    from tinygrad.device import TinyELF
    from tinygrad.helpers import Target
    from tinygrad.runtime.ops_nv import NVProgram
    sig = tuple((f"v{i}", i, dtypes.uint32, ()) for i in range(vals))
    self.prg = NVProgram(dev, TinyELF(lib, name, Target(device="NV"), sig))
    self.workgroup, self.scratch = workgroup, scratch
    self.label = getattr(compile_cuda, "last", name)

  def args(self, bufs, grid, vals=()):
    # CUDA's blockDim and gridDim are the driver's words in constant bank 0, which tinygrad's launches leave 0 (the
    # words are copied into each launch's arguments as they are filled)
    self.prg.cbuf_0[0:6] = [*self.workgroup, *grid]
    return self.prg.fill_kernargs(tuple(bufs), tuple(vals))


class Fast:
  """A launch as tinygrad's NVComputeQueue.exec writes it, without the objects it makes a launch (a view of memory a
  word, its QMD's fields set by name): a launch's arguments and its QMD written in a few copies, its fields' places
  found once, from tinygrad's own tables. On with TINYGPU_FAST=1 (else tinygrad's own launch); TINYGPU_FAST_CHECK=1 has
  tinygrad write each launch too (in its own slot, never submitted) and compares the two before the card is given it.

  A launch's arguments (a slot of tinygrad's ring) are the program's constant bank 0 (its words, the workgroup's size
  and the grid in the first six), its pointers, then its 32-bit values; its QMD is the program's (its workgroup's size
  set once), the grid and where the arguments are set, and the queue's last QMD is pointed at it (or it is sent, the
  queue's first)."""

  def __init__(self, dev):
    import tinygrad.runtime.ops_nv as nv
    self.nv, self.dev = nv, dev
    probe = nv.QMD(dev)
    if probe.ver != 3: raise NotImplementedError(f"QMD version {probe.ver}")
    if dev.pma_enabled: raise NotImplementedError("tinygrad's performance counters")
    f = nv.QMD.fields[probe.pref]
    self.ring, self.ring_va = dev.kernargs_buf.cpu_view().mv, dev.kernargs_buf.va_addr
    self.grid_at = f["CTA_RASTER_WIDTH"][1] // 8
    self.dims = (f["CTA_THREAD_DIMENSION0"][1] // 8, f["CTA_THREAD_DIMENSION2"][1] // 8)
    self.cbuf_hi, self.cbuf_lo = self.bits(f, "CONSTANT_BUFFER_ADDR_UPPER_0"), self.bits(f, "CONSTANT_BUFFER_ADDR_LOWER_0")
    self.next_qmd = self.bits(f, "DEPENDENT_QMD0_POINTER")
    # (the dependent QMD's action, prefetch and enable, each 1: tinygrad's)
    self.next_flags = [self.bits(f, k) for k in ("DEPENDENT_QMD0_ACTION", "DEPENDENT_QMD0_PREFETCH", "DEPENDENT_QMD0_ENABLE")]
    # a field that is a whole 32-bit word written as one; the flags, where they share their bytes, in one write
    self.word = lambda spec: spec[0] if spec[3] == 0 and spec[4] == 32 and spec[1] - spec[0] == 4 else None
    self.cbuf_lo_word, self.next_qmd_word = self.word(self.cbuf_lo), self.word(self.next_qmd)
    spans = {(sp[0], sp[1]) for sp in self.next_flags}
    self.flags = None
    if len(spans) == 1:
      (b0, b1), = spans
      self.flags = (b0, b1, sum(sp[2] for sp in self.next_flags), sum((1 << sp[3]) & sp[2] for sp in self.next_flags))
    self.pcas = (nv.nv_gpu.NVC6C0_SEND_PCAS_A, nv.nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B)
    self.qmd_size = probe.sz * 4
    self.last = None   # the ring's offset of the queue's last QMD, None where the queue has none since its start

  @staticmethod
  def bits(fields, name):
    """A field's bytes (first, past last), its mask in them and its shift: what tinygrad's QMD._rw_bits works out a write."""
    hi, lo = fields[name]
    return lo // 8, hi // 8 + 1, ((1 << (hi - lo + 1)) - 1) << (lo % 8), lo % 8, hi - lo + 1

  def put(self, at: int, spec, value: int):
    b0, b1, mask, shift, width = spec
    if value >> width: raise ValueError(f"{value:#x} does not fit {width} bits")
    num = int.from_bytes(self.ring[at + b0:at + b1], "little")
    self.ring[at + b0:at + b1] = ((num & ~mask) | ((value << shift) & mask)).to_bytes(b1 - b0, "little")

  def program(self, prog):
    """What a launch of `prog` writes that is the same each time: its constant bank's words and its QMD (its
    workgroup's size in both), their layout in its slot."""
    p = prog.prg
    words = list(p.cbuf_0)
    words[0:3] = prog.workgroup
    qmd = bytearray(p.qmd.mv)
    struct.pack_into("<HH", qmd, self.dims[0], *prog.workgroup[:2])
    struct.pack_into("<B", qmd, self.dims[1], prog.workgroup[2])
    prog.fast = (struct.pack(f"<{len(words)}I", *words), len(words) * 4, self.nv.round_up(p.constbufs[0][1], 1 << 8), bytes(qmd), p.kernargs_alloc_size)
    return prog.fast

  def launch(self, q, prog, ptrs: list, grid: tuple, vals: tuple):
    """`prog` over `grid`, its pointers and values, added to queue `q` (its last QMD pointed at this one)."""
    prefix, plen, qmd_at, qmd, size = getattr(prog, "fast", None) or self.program(prog)
    off = self.dev.kernargs_offset_allocator.alloc(size, 8)
    ring, va = self.ring, self.ring_va + off
    ring[off:off + plen] = prefix
    struct.pack_into("<3I", ring, off + 12, *grid)
    struct.pack_into(f"<{len(ptrs)}Q{len(vals)}I", ring, off + plen, *ptrs, *vals)
    qo = off + qmd_at
    ring[qo:qo + len(qmd)] = qmd
    struct.pack_into("<3I", ring, qo + self.grid_at, *grid)
    self.put(qo, self.cbuf_hi, va >> 32)
    if self.cbuf_lo_word is not None: struct.pack_into("<I", ring, qo + self.cbuf_lo_word, va & 0xffffffff)
    else: self.put(qo, self.cbuf_lo, va & 0xffffffff)
    if self.last is None:
      q.nvm(1, self.pcas[0], (va + qmd_at) >> 8)
      q.nvm(1, self.pcas[1], 9)
    else:
      at = self.last
      if self.next_qmd_word is not None: struct.pack_into("<I", ring, at + self.next_qmd_word, (va + qmd_at) >> 8)
      else: self.put(at, self.next_qmd, (va + qmd_at) >> 8)
      if self.flags is not None:
        b0, b1, mask, value = self.flags
        num = int.from_bytes(ring[at + b0:at + b1], "little")
        ring[at + b0:at + b1] = ((num & ~mask) | value).to_bytes(b1 - b0, "little")
      else:
        for spec in self.next_flags: self.put(at, spec, 1)
    self.last = qo
    return off

  def ending(self, q):
    """The queue's last QMD as tinygrad's queue keeps it, for its signal to be written into (a release of it)."""
    if self.last is None: return
    buf = self.dev.kernargs_buf.offset(offset=self.last, size=self.qmd_size)
    q.active_qmd, q.active_qmd_buf = self.nv.QMD(dev=self.dev, view=buf.cpu_view()), buf
    self.last = None


class Card:
  """The NVIDIA card tinygrad holds (DEV=NV). A buffer is (tinygrad's buffer, its size).

  A submission's kernels go in one queue of the card's (one doorbell), each launched as the one before it ends, as
  tinygrad's own graphs chain them; a write goes by the copy queue, which waits for what was submitted before it, so
  neither waits for the card. A freed buffer goes to tinygrad's cache, from where the next of its size reuses it in the
  queue's order."""
  BATCH = 256      # kernels a queue, before it goes to the card and the next starts
  # TINYGPU_SYNC_EACH=1: each kernel waited for as it is launched, so a fault is the kernel that made it (slow; to
  # find a kernel the emulator lets through)
  SYNC_EACH = os.environ.get("TINYGPU_SYNC_EACH") == "1"
  ARGS_RING = 4096  # launches between waits: their arguments are a ring of tinygrad's (16 MB), reused as it wraps
  # TINYGPU_PROFILE=1: each kernel run alone and timed (the wall clock's, its own submission's 45 us in it), the times
  # by kernel written to ~/.cache/tinygpu-webgpu/profile.txt as each client goes: where a program's time is spent
  PROFILE = os.environ.get("TINYGPU_PROFILE") == "1"

  def __init__(self):
    from tinygrad import Device
    from tinygrad.device import BufferSpec
    self.dev, self.spec = Device["NV"], BufferSpec()
    self.arch = self.dev.arch
    self.reserve_firmware_memory()
    self.local_memory()
    self.q, self.queued, self.since_sync = None, 0, 0
    self.scratch = None
    self.profiled: dict = {}   # (TINYGPU_PROFILE: a kernel's calls, its time, its last grid, its workgroup)
    built = {name: Program(self.dev, compile_cuda(src, self.arch), name, (256, 1, 1), 1) for name, src in BUILTINS.items()}
    self.copy4, self.copy1, self.fill4, self.fill1 = built["tg_copy4"], built["tg_copy1"], built["tg_fill4"], built["tg_fill1"]
    self.fast = None
    if os.environ.get("TINYGPU_FAST") == "1" and not (self.SYNC_EACH or self.PROFILE):
      try: self.fast = Fast(self.dev)
      except (NotImplementedError, KeyError, AttributeError) as e: print(f"tinygpu-webgpu: launches through tinygrad ({e})", file=sys.stderr, flush=True)
    self.check = self.fast is not None and os.environ.get("TINYGPU_FAST_CHECK") == "1"
    if self.fast is not None:
      # (SIGUSR1 turns the check on or off, as the server runs)
      def toggle(*_):
        self.check = not self.check
        print(f"tinygpu-webgpu: fast launches {'checked' if self.check else 'not checked'}", file=sys.stderr, flush=True)
      signal.signal(signal.SIGUSR1, toggle)

  # The least of the card's memory kept from buffers at its top (the GSP firmware's, with room to spare)
  FIRMWARE_RESERVE = 512 << 20

  def reserve_firmware_memory(self):
    """The card's memory where the GSP firmware lives kept from tinygrad's allocator. tinygrad keeps the last 64 MB of
    the card's memory out of it, but lays the firmware out below that (tinygrad's GspFwWprMeta: its heap, 129 MB, its
    image, its boot code, from the top down to gspFwRsvdStart): with the card's memory all but full, a buffer is given
    pages there, and the first write to them faults the card (a REGION_VIOLATION: the firmware's region is protected).
    On 2026-10-10 a language model loaded after pictures (the pictures' buffers in tinygrad's cache, 23.2 GB in all)
    faulted so three times at the same place, and a picture after a language model once. The range from the
    firmware's start (or FIRMWARE_RESERVE below the top, where that is lower) is taken out of the allocator's free
    space once, as the card opens, before any buffer can be there."""
    impl = getattr(self.dev.iface, "dev_impl", None)
    mm = getattr(impl, "mm", None)
    if mm is None or not hasattr(mm, "pa_allocator"): return   # (a card the kernel's driver manages)
    pa, top = mm.pa_allocator, impl.vram_size
    start = top - self.FIRMWARE_RESERVE
    try:
      from tinygrad.runtime.autogen import nv as nvs
      meta = nvs.GspFwWprMeta.from_buffer_copy(bytes(impl.gsp.wpr_meta[:ctypes.sizeof(nvs.GspFwWprMeta)]))
      start = min(start, meta.gspFwRsvdStart)
    except Exception as e:
      print(f"tinygpu-webgpu: the firmware's layout not read ({type(e).__name__}: {e}); the top {self.FIRMWARE_RESERVE >> 20} MB kept", file=sys.stderr, flush=True)
    start -= start % (2 << 20)
    lo, hi = start - pa.base, pa.size
    if lo >= hi: return
    # the free block that holds [lo, hi), cut to it and taken
    for at, (size, _nxt, _prev, free) in list(pa.blocks.items()):
      if free and at <= lo and lo + (hi - lo) <= at + size: break
    else:
      raise RuntimeError(f"the card's memory from {start:#x} is already in use: the firmware's region cannot be kept from buffers")
    if at < lo:
      pa._split_block(at, size, lo - at)
      at, size = lo, pa.blocks[lo][0]
    if size > hi - lo: pa._split_block(at, size, hi - lo)
    pa._remove_block(at, hi - lo)
    print(f"tinygpu-webgpu: the card's memory from {start:#x} to its top ({(top - start) >> 20} MB: the firmware's) kept from buffers", file=sys.stderr, flush=True)

  def local_memory(self):
    """The card's local memory (a thread's spills and stack) grown as tinygrad's NVDevice._ensure_has_local_memory does,
    but safely: the work in flight, which spills into the old buffer, waited for first; the new buffer made before the
    old is let go, and none given to the card that it was not made for. tinygrad's sets the new size a TPC first, and
    where the larger buffer cannot be had (the card's memory full: another program's models loaded) keeps the old one
    and still gives the card the new size: threads then spill past its end, a write to memory that is no one's. Two
    faults on 2026-10-10, each as a second program's models went to a card the first had filled, were writes at a page's
    start. Each growth, and each emptying of tinygrad's cache of freed buffers, is said on stderr."""
    import tinygrad.runtime.ops_nv as nv
    from tinygrad.helpers import round_up
    dev = self.dev

    def ensure(required: int):
      if dev.slm_per_thread >= required: return
      slm = round_up(required, 32)
      per_tpc = round_up(round_up(slm * 32, 0x200) * dev.max_warps_per_sm * dev.num_sm_per_tpc, 0x8000)
      size = round_up(per_tpc * dev.num_tpc_per_gpc * dev.num_gpcs, 0x20000)
      dev.synchronize()
      try: buf = dev.allocator.alloc(size)
      except MemoryError as e: raise RuntimeError(f"the card has no room for a kernel's local memory ({size >> 20} MB): {e}") from e
      old = dev.shader_local_mem
      nv.NVComputeQueue().wait(dev.timeline_signal, dev.timeline_value - 1).setup(local_mem=buf.va_addr, local_mem_tpc_bytes=per_tpc) \
                         .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
      dev.synchronize()
      dev.shader_local_mem, dev.slm_per_thread = buf, slm
      if old is not None: dev.allocator.free(old, old.size)
      print(f"tinygpu-webgpu: local memory {slm} bytes a thread, {size >> 20} MB at {buf.va_addr:#x}..{buf.va_addr + size:#x}", file=sys.stderr, flush=True)

    dev._ensure_has_local_memory = ensure
    allocator = dev.allocator
    if hasattr(allocator, "free_cache"):
      emptied = allocator.free_cache

      def free_cache(*args, **kwargs):
        n = sum(len(v) for v in getattr(allocator, "cache", {}).values())
        print(f"tinygpu-webgpu: the card's memory full: tinygrad's {n} cached buffers let go", file=sys.stderr, flush=True)
        return emptied(*args, **kwargs)

      allocator.free_cache = free_cache

  def begin(self):
    if self.q is None:
      self.q = self.dev.hw_compute_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()
      if self.fast: self.fast.last = None
      self.ref = self.dev.hw_compute_queue_t() if self.check else None

  def end(self):
    if self.q is not None and self.queued:
      if self.fast: self.fast.ending(self.q)
      self.q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
    self.q, self.queued = None, 0

  def launch_fast(self, prog: Program, ptrs: list, grid, vals):
    """`prog` launched by Fast: as launch(), its arguments as addresses."""
    inside = self.q is not None
    if self.since_sync >= self.ARGS_RING: self.sync()
    self.begin()
    before = self.fast.last
    off = self.fast.launch(self.q, prog, ptrs, grid, vals)
    if self.check and self.ref is not None:
      try: self.compare(prog, ptrs, grid, vals, off, before)
      except Exception:
        # (the queue let go, never submitted: what differs from tinygrad's must not reach the card)
        self.q, self.queued, self.fast.last = None, 0, None
        raise
    # (a checked launch takes two slots of the arguments' ring, its own and tinygrad's: counted so, or the ring would
    # wrap onto launches the card has yet to run)
    self.queued, self.since_sync = self.queued + 1, self.since_sync + (2 if self.check else 1)
    if not inside: self.end()
    elif self.queued >= self.BATCH:
      self.end()
      self.begin()

  def compare(self, prog, ptrs, grid, vals, off, before):
    """TINYGPU_FAST_CHECK: the launch written by tinygrad as well, into a queue never submitted, and the two compared:
    the arguments byte for byte, the QMD with its arguments' address set to Fast's, and the QMD before it in the
    queue pointed the same way (each at its own next)."""
    from tinygrad.runtime.support.hcq import HCQBuffer
    nv, f, ring = self.fast.nv, self.fast, self.fast.ring
    prefix, plen, qmd_at, qmd, size = prog.fast
    views = [HCQBuffer(a, 8) for a in ptrs]
    prog.prg.cbuf_0[0:6] = [*prog.workgroup, *grid]
    args = prog.prg.fill_kernargs(tuple(views), tuple(vals))
    ref_prev = self.ref.active_qmd_buf.va_addr - f.ring_va if self.ref.active_qmd is not None else None
    self.ref.exec(prog.prg, args, grid, prog.workgroup)
    ro = args.buf.va_addr - f.ring_va
    n = plen + 8 * len(ptrs) + 4 * len(vals)
    if bytes(ring[off:off + n]) != bytes(ring[ro:ro + n]):
      raise RuntimeError(f"fast launch: {prog.label}'s arguments differ from tinygrad's")
    mine, theirs = bytearray(ring[off + qmd_at:off + qmd_at + f.qmd_size]), bytearray(ring[ro + qmd_at:ro + qmd_at + f.qmd_size])
    def same(a, b, at_a, at_b, specs):
      # (each with the given fields cleared: what differs by where each one is)
      for spec in specs:
        for buf in (a, b):
          b0, b1, mask, _, _ = spec
          num = int.from_bytes(buf[b0:b1], "little") & ~mask
          buf[b0:b1] = num.to_bytes(b1 - b0, "little")
      return a == b
    if not same(mine, theirs, 0, 0, [f.cbuf_hi, f.cbuf_lo]):
      raise RuntimeError(f"fast launch: {prog.label}'s QMD differs from tinygrad's")
    if (before is None) != (ref_prev is None):
      raise RuntimeError("fast launch: the queue's QMDs chained otherwise than tinygrad's")
    if before is not None:
      a, b = bytearray(ring[before:before + f.qmd_size]), bytearray(ring[ref_prev:ref_prev + f.qmd_size])
      if not same(a, b, 0, 0, [f.cbuf_hi, f.cbuf_lo, f.next_qmd]):
        raise RuntimeError(f"fast launch: the QMD before {prog.label}'s chained otherwise than tinygrad's")

  def launch(self, prog: Program, bufs, grid, vals=()):
    inside = self.q is not None   # (a submission's, else a launch of its own: an ALLOC's clear)
    if self.PROFILE: return self.timed(prog, bufs, grid, vals, inside)
    if self.since_sync >= self.ARGS_RING: self.sync()
    self.begin()
    self.q.exec(prog.prg, prog.args(bufs, grid, vals), grid, prog.workgroup)
    self.queued, self.since_sync = self.queued + 1, self.since_sync + 1
    if self.SYNC_EACH:
      self.end()
      try: self.dev.synchronize()
      except Exception:
        print(f"tinygpu-webgpu: the card faulted at {prog.label} {grid} {prog.workgroup} (its CUDA: {CACHE / (prog.label + '.cu')})", file=sys.stderr, flush=True)
        raise
      if inside: self.begin()
      return
    if not inside: self.end()
    elif self.queued >= self.BATCH:
      self.end()
      self.begin()

  def timed(self, prog: Program, bufs, grid, vals, inside: bool):
    """`prog` launched alone and waited for (TINYGPU_PROFILE), its time added to its kernel's."""
    self.end()
    self.dev.synchronize()
    t = time.perf_counter()
    self.begin()
    self.q.exec(prog.prg, prog.args(bufs, grid, vals), grid, prog.workgroup)
    self.queued = 1
    self.end()
    self.dev.synchronize()
    took = time.perf_counter() - t
    self.since_sync = 0
    seen = self.profiled.setdefault(prog.label, [0, 0.0, grid, prog.workgroup])
    seen[0] += 1
    seen[1] += took
    seen[2] = grid
    if inside: self.begin()

  def report(self):
    """The kernels' times so far (TINYGPU_PROFILE), the most first, written to the cache's profile.txt."""
    if not self.PROFILE or not self.profiled: return
    total = sum(s[1] for s in self.profiled.values())
    lines = [f"{total * 1e3:10.1f} ms in {sum(s[0] for s in self.profiled.values())} kernels (each alone: a submission's 45 us in each)",
             f"{'ms':>10} {'%':>5} {'calls':>7} {'ms each':>9}  {'grid':<16} {'workgroup':<12} kernel (its CUDA: <cache>/<kernel>.cu)"]
    for label, (n, t, grid, wg) in sorted(self.profiled.items(), key=lambda kv: -kv[1][1]):
      lines.append(f"{t * 1e3:10.1f} {100 * t / total:5.1f} {n:7d} {t * 1e3 / n:9.3f}  {str(grid):<16} {str(wg):<12} {label}")
    (CACHE / "profile.txt").write_text("\n".join(lines) + "\n")

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

  def program(self, workgroup: tuple, src: str, sizes: int, scratch: bool):
    if scratch and self.scratch is None: self.scratch = self.alloc(SCRATCH)
    return Program(self.dev, compile_cuda(src, self.arch), "oaiy_main", workgroup, sizes, scratch)

  def dispatch(self, prog: Program, bufs: list, grid: tuple, sizes: list):
    if self.fast:
      ptrs = [b[0].va_addr + off for b, off in bufs]
      if prog.scratch: ptrs.append(self.scratch[0].va_addr)
      return self.launch_fast(prog, ptrs, grid, tuple(sizes))
    views = [self.view(b, off, b[1] - off) for b, off in bufs] + ([self.scratch[0]] if prog.scratch else [])
    self.launch(prog, views, grid, sizes)

  def copy(self, src, soff: int, dst, doff: int, size: int):
    words = (soff | doff | size) % 4 == 0
    prog, n = (self.copy4, size // 4) if words else (self.copy1, size)
    grid = (min(1024, (n + 255) // 256), 1, 1)
    if self.fast: return self.launch_fast(prog, [src[0].va_addr + soff, dst[0].va_addr + doff], grid, (n,))
    self.launch(prog, (self.view(src, soff, size), self.view(dst, doff, size)), grid, (n,))

  def clear(self, buf, off: int, size: int):
    words = (off | size) % 4 == 0
    prog, n = (self.fill4, size // 4) if words else (self.fill1, size)
    grid = (min(1024, (n + 255) // 256), 1, 1)
    if self.fast: return self.launch_fast(prog, [buf[0].va_addr + off], grid, (n,))
    self.launch(prog, (self.view(buf, off, size),), grid, (n,))

  def sync(self):
    self.end()
    self.dev.synchronize()
    self.since_sync = 0


# ---- no card: the kernels on the CPU ----

class Emulator:
  """The card's work on the CPU: memory the process's own, each kernel compiled by clang++ with emu/cuda_emu.h and its
  workgroups shared out among the cores (ctypes lets go of Python's lock while a kernel runs). Each piece of work is
  done before the next is taken, so the queue is always done. A buffer is (its address, its size, its mapping's address
  and size): it ends where a page that is not readable begins, so a kernel that reads or writes past a buffer's end
  faults here as it would fault the card (where it costs the card its link). TINYGPU_EMU_TRACE=1 says each dispatch's
  kernel (its file in the cache) on stderr first, so the last said is a fault's."""
  STACK = int(os.environ.get("TINYGPU_EMU_STACK_KB", 1024)) << 10   # an invocation's stack (a fragment of a tensor core's is all its matrix here)
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
    self.trace = os.environ.get("TINYGPU_EMU_TRACE") == "1"

  def info(self) -> dict:
    return {"arch": "cpu", "name": "the card's work on the CPU (tinygpu-webgpu --emulate)", "device": "CPU",
            "memory": int(os.environ.get("TINYGPU_MEMORY", 16 << 30))}

  def map(self, size: int) -> int:
    # (anonymous memory: zeroed, and given pages as they are touched)
    at = self.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_ANON | mmap.MAP_PRIVATE, -1, 0)
    if at in (None, ctypes.c_void_p(-1).value): raise MemoryError(f"mmap of {size} bytes: errno {ctypes.get_errno()}")
    return at

  def alloc(self, size: int):
    page = mmap.PAGESIZE
    span = (size + page - 1) // page * page + page
    at = self.map(span)
    self.mprotect(at + span - page, page, 0)   # (PROT_NONE)
    if self.trace: print(f"emu: buffer 0x{at + span - page - size:016x} to 0x{at + span - page:016x} ({size} bytes)", file=sys.stderr, flush=True)
    return (at + span - page - size, size, at, span)

  def free(self, buf):
    self.munmap(buf[2], buf[3])

  def write(self, buf, off: int, data: memoryview):
    ctypes.memmove(buf[0] + off, bytes(data), len(data))

  def read(self, buf, off: int, size: int) -> bytes:
    return ctypes.string_at(buf[0] + off, size)

  def program(self, workgroup: tuple, src: str, sizes: int, scratch: bool):
    header = (EMU / "cuda_emu.h").read_bytes() + (EMU / "mma.h").read_bytes()
    barriers = "__syncthreads()" in src
    def build(tmp: pathlib.Path):
      # (the source under this build's own name, then beside the library as it will be named: two builds of one
      # kernel at once, two servers', must not write one file)
      cpp = tmp.with_suffix(f".{threading.get_ident()}.cpp")
      cpp.write_text(f"{src}\nEMU_LAUNCHER(oaiy_main, {'true' if barriers else 'false'})\n")
      run_compiler(["clang++", "-std=c++20", "-O2", "-fwrapv", "-shared", "-fPIC", "-w", "-include", str(EMU / "cuda_emu.h"),
                    "-I", str(EMU), str(cpp), "-o", str(tmp)], "clang++")
      cpp.rename(tmp.with_name(tmp.name.split(".")[0] + ".cpp"))
    path = cached(".dylib", b"cpu\0" + header + b"\0" + src.encode(), build)
    lib = ctypes.CDLL(str(path))
    launch = lib.emu_launch
    launch.restype = None
    launch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint64,
                       ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    return (lib, launch, workgroup, path.with_suffix(".cpp"), scratch)

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

  def dispatch(self, prog, bufs: list, grid: tuple, sizes: list):
    _, launch, workgroup, name, scratch = prog
    if self.trace: print(f"emu: {name} {grid} " + " ".join(f"0x{b[0] + off:x}+{b[1] - off}" for b, off in bufs), file=sys.stderr, flush=True)
    if workgroup[0] * workgroup[1] * workgroup[2] > self.MAX_INVOCATIONS: raise RuntimeError(f"a workgroup of {workgroup}")
    # (the pointers, the scratch where it takes one (the CPU's kernels stage on their own stacks), then the bindings'
    # sizes: integers in the same words)
    args = (ctypes.c_void_p * max(1, 2 * len(bufs) + 1))(*[b[0] + off for b, off in bufs], *([0] if scratch else []), *sizes)
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
  # TINYGPU_TRACE_ALLOC=FILE: each buffer made (its id, its place on the card, its size) and let go, and each program
  # made, a line in FILE: a fault's address against the buffers there were
  TRACE = os.environ.get("TINYGPU_TRACE_ALLOC")

  def __init__(self, card):
    self.card = card
    self.trace = open(self.TRACE, "a", buffering=1) if self.TRACE else None
    self.buffers: dict = {}
    self.programs: dict = {}
    self.next_id = 1
    self.dispatched = 0   # (the last submission's dispatches, for TINYGPU_STATS)

  def new_id(self) -> int:
    self.next_id += 1
    return self.next_id - 1

  def hello(self, _p):
    return json.dumps({**self.card.info(), "protocol": PROTOCOL}).encode()

  def alloc(self, p):
    (size,) = struct.unpack_from("<Q", p)
    h = self.new_id()
    self.buffers[h] = self.card.alloc(max(ALIGN, (size + ALIGN - 1) // ALIGN * ALIGN))
    if self.trace:
      b = self.buffers[h][0]
      va = getattr(b, "va_addr", b) if not isinstance(b, int) else b
      self.trace.write(f"alloc {h} {va:#x}..{va + self.buffers[h][1]:#x} {size}\n")
    return struct.pack("<Q", h)

  def free(self, p):
    (h,) = struct.unpack_from("<Q", p)
    b = self.buffers.pop(h, None)
    if b is not None:
      if self.trace: self.trace.write(f"free {h}\n")
      self.card.free(b)
    return b""

  def within(self, h: int, off: int, size: int, what: str):
    """The buffer, where off..off + size is in it (wgpu's own checks of a copy, a write, a binding: a range past a
    buffer is the card's memory past it, and a write there may fault the card)."""
    b = self.buffers[h]
    if off + size > b[1]: raise RuntimeError(f"{what}: bytes {off}..{off + size} of a buffer of {b[1]}")
    return b

  def write(self, p):
    h, off = struct.unpack_from("<QQ", p)
    data = memoryview(p)[16:]
    if len(data): self.card.write(self.within(h, off, len(data), "a write"), off, data)
    return b""

  def read(self, p):
    h, off, size = struct.unpack_from("<QQQ", p)
    return self.card.read(self.within(h, off, size, "a read"), off, size) if size else b""

  def program(self, p):
    x, y, z, n, flags = struct.unpack_from("<IIIII", p)
    h = self.new_id()
    self.programs[h] = self.card.program((x, y, z), bytes(p[20:]).decode(), n, bool(flags & 1))
    if self.trace: self.trace.write(f"program {h} {getattr(self.programs[h], 'label', '')}\n")
    return struct.pack("<Q", h)

  def submit(self, p):
    p, at = memoryview(p), 0
    self.card.begin()
    try: self.ops(p)
    finally: self.card.end()
    return b""

  def ops(self, p):
    at = 0
    self.dispatched = 0
    while at < len(p):
      op = p[at]; at += 1
      if op == 1:
        self.dispatched += 1
        prog, gx, gy, gz, n = struct.unpack_from("<QIIII", p, at); at += 24
        bufs, sizes = [], []
        for _ in range(n):
          b, off, size = struct.unpack_from("<QQQ", p, at); at += 24
          bufs.append((self.within(b, off, size, "a binding"), off))
          sizes.append(min(size, 0xffffffff))
        if gx and gy and gz: self.card.dispatch(self.programs[prog], bufs, (gx, gy, gz), sizes)
      elif op == 2:
        s, so, d, do, size = struct.unpack_from("<QQQQQ", p, at); at += 40
        if size: self.card.copy(self.within(s, so, size, "a copy's source"), so, self.within(d, do, size, "a copy's destination"), do, size)
      elif op == 3:
        b, off, size = struct.unpack_from("<QQQ", p, at); at += 24
        if size: self.card.clear(self.within(b, off, size, "a clear"), off, size)
      else: raise RuntimeError(f"unknown op {op}")

  def sync(self, _p):
    self.card.sync()
    return b""

  COMMANDS = {1: hello, 2: alloc, 3: free, 4: write, 5: read, 6: program, 7: submit, 8: sync}


def recv_exact(conn: socket.socket, n: int, into: bytearray | None = None) -> bytearray:
  """`n` bytes from the client (in `into`'s first n, where it is given: a buffer kept for it)."""
  buf = bytearray(n) if into is None else into
  view, got = memoryview(buf), 0
  while got < n:
    k = conn.recv_into(view[got:n], n - got)
    if k == 0: raise ConnectionError("client closed")
    got += k
  return buf


# TINYGPU_STATS=1: each kind of request's count and time (the server's, inside it), the dispatches submitted, and
# the time between requests (the client's), said on stderr every 5 seconds: where a program's time goes, the server's
# Python or the card's work or the client.
STATS = os.environ.get("TINYGPU_STATS") == "1"
# TINYGPU_CPROFILE=FILE: the server's Python profiled while it answers (cProfile), written to FILE as a client goes
CPROFILE = os.environ.get("TINYGPU_CPROFILE")
NAMES = {1: "hello", 2: "alloc", 3: "free", 4: "write", 5: "read", 6: "program", 7: "submit", 8: "sync"}
stats = {"since": time.perf_counter(), "idle": 0.0, "dispatches": 0}


def counted(cmd: int, took: float, idle: float, dispatches: int):
  """A request's time added to its kind's (TINYGPU_STATS), and the totals said every 5 seconds."""
  n, t = stats.get(cmd, (0, 0.0))
  stats[cmd] = (n + 1, t + took)
  stats["idle"] += idle
  stats["dispatches"] += dispatches
  now = time.perf_counter()
  if now - stats["since"] < 5: return
  span = now - stats["since"]
  kinds = "  ".join(f"{NAMES.get(k, k)} {n} {1e3 * t:.0f} ms" for k, (n, t) in sorted((k, v) for k, v in stats.items() if isinstance(k, int)))
  print(f"tinygpu-webgpu: {span:.1f} s: {kinds}  dispatches {stats['dispatches']} ({1e6 * stats.get(7, (0, 0.0))[1] / max(1, stats['dispatches']):.1f} us each in submits)  between requests {1e3 * stats['idle']:.0f} ms", file=sys.stderr, flush=True)
  stats.clear()
  stats.update({"since": now, "idle": 0.0, "dispatches": 0})


# A large write is taken in pieces of this size, each sent on to the card as it comes: the card's copy (tinygrad's, by
# its staging buffers, which it does not wait for) runs while the next piece is received.
PIECE = 16 << 20


def write_in_pieces(server: Server, conn: socket.socket, n: int, piece: bytearray):
  """A WRITE of `n` bytes (its buffer, its offset, its bytes) taken from the socket piece by piece. Its bytes are all
  read, an error or not, so the next request is where the client put it."""
  h, off = struct.unpack("<QQ", recv_exact(conn, 16))
  size, error = n - 16, None
  try: buf = server.within(h, off, size, "a write")
  except Exception as e: buf, error = None, e
  at = 0
  while at < size:
    k = min(PIECE, size - at)
    recv_exact(conn, k, piece)
    if error is None:
      try: server.card.write(buf, off + at, memoryview(piece)[:k])
      except Exception as e: error = e
    at += k
  if error is not None: raise error


# How many clients are connected, and a word to whoever waits for that to change (egpu_serve.py, pausing OAIY's engine:
# its memory on the card is free once its connection's buffers are).
CLIENTS, connected = threading.Condition(), [0]


def wait_clients(most: int, timeout: float) -> bool:
  """Whether the clients came down to `most` within `timeout` seconds, each one's buffers freed as it went."""
  with CLIENTS: return CLIENTS.wait_for(lambda: connected[0] <= most, timeout)


def handle(server: Server, conn: socket.socket, lock: threading.Lock, closing: threading.Event):
  """One client's requests, each run with the card to itself; the buffers it did not free are freed as it goes."""
  owned, piece = set(), None
  profile = None
  if CPROFILE:
    import cProfile
    profile = cProfile.Profile()
  with CLIENTS: connected[0] += 1
  answered = time.perf_counter()
  try:
    while True:
      cmd, n = struct.unpack("<IQ", recv_exact(conn, 12))
      asked = time.perf_counter()
      if cmd == 4 and n > 16 + PIECE:
        # (with the card to itself throughout, as any request: its pieces are not another client's requests' business)
        with lock:
          if closing.is_set(): return
          try:
            if piece is None: piece = bytearray(PIECE)
            write_in_pieces(server, conn, n, piece)
            out, status = b"", 0
          except (ConnectionError, OSError): raise
          except Exception as e:
            out, status = f"{type(e).__name__}: {e}".encode(), 1
            traceback.print_exc()
        conn.sendall(struct.pack("<IQ", status, len(out)) + out)
        answered = time.perf_counter()
        continue
      payload = recv_exact(conn, n) if n else b""
      if cmd == 9:
        conn.sendall(struct.pack("<IQ", 0, 0))
        os.kill(os.getpid(), signal.SIGTERM)   # (the main thread ends the server: tinygrad's exit is the process's)
        return
      with lock:
        if closing.is_set(): return
        try:
          if cmd not in Server.COMMANDS: raise RuntimeError(f"unknown command {cmd}")
          t = time.perf_counter()
          if profile: profile.enable()
          try: out, status = Server.COMMANDS[cmd](server, payload), 0
          finally:
            if profile: profile.disable()
          if STATS: counted(cmd, time.perf_counter() - t, asked - answered, server.dispatched if cmd == 7 else 0)
          if cmd == 2: owned.add(struct.unpack("<Q", out)[0])
          elif cmd == 3: owned.discard(struct.unpack_from("<Q", payload)[0])
        except Exception as e:
          out, status = f"{type(e).__name__}: {e}".encode(), 1
          traceback.print_exc()
      conn.sendall(struct.pack("<IQ", status, len(out)) + out)
      answered = time.perf_counter()
  except OSError:   # (the client gone: ConnectionError, BrokenPipeError)
    pass
  finally:
    conn.close()
    try:
      with lock:
        if not closing.is_set():
          for h in owned: server.free(struct.pack("<Q", h))
          if hasattr(server.card, "report"): server.card.report()
          if profile: profile.dump_stats(CPROFILE)
    finally:
      with CLIENTS:
        connected[0] -= 1
        CLIENTS.notify_all()


def listen(path: str) -> socket.socket:
  """The socket clients connect to, at `path` (for this user alone)."""
  if os.path.exists(path): os.unlink(path)
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.bind(path)
  os.chmod(path, 0o600)
  sock.listen(16)
  return sock


def accept(server: Server, sock: socket.socket, lock: threading.Lock, closing: threading.Event):
  """Clients at once, each on a thread of its own; their requests one at a time. Until the socket is closed."""
  while not closing.is_set():
    try: conn, _ = sock.accept()
    except OSError: return
    # (a Mac gives a local socket 8 KB each way: a read's bytes cross it in 4 MB calls instead)
    for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
      try: conn.setsockopt(socket.SOL_SOCKET, option, 4 << 20)
      except OSError: pass
    threading.Thread(target=handle, args=(server, conn, lock, closing), daemon=True).start()


# The signals that end a process unless it handles them. A holder of the card ended by one of them leaves its firmware
# up, and the next open resets the card: one such reset (after a SIGUSR1 a server of older code had no handler for)
# took the card's link down until the enclosure was powered off and on. Each is a normal exit here instead.
ENDING = [getattr(signal, n) for n in ("SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGALRM", "SIGVTALRM", "SIGPROF", "SIGXCPU", "SIGXFSZ") if hasattr(signal, n)]


def exit_cleanly_on_signals():
  """Every signal that would end the process without tinygrad's exit made a normal exit (Card's SIGUSR1 replaces its)."""
  for sig in ENDING: signal.signal(sig, lambda *_: sys.exit(0))


def serve(path: str, emulate: bool):
  # (a stop by signal is a normal exit, so tinygrad's own exit releases the card)
  exit_cleanly_on_signals()
  server = Server(Emulator() if emulate else Card())
  sock = listen(path)
  print(f"tinygpu-webgpu: {server.card.info()['name']} ready on {path}", flush=True)
  lock, closing = threading.Lock(), threading.Event()
  try:
    accept(server, sock, lock, closing)
  finally:
    closing.set()
    lock.acquire(timeout=60)   # (a request under way ends first: the card is let go between requests)
    sock.close()


if __name__ == "__main__":
  argv = sys.argv[1:]
  emulate = "--emulate" in argv
  argv = [a for a in argv if a != "--emulate"]
  CACHE.mkdir(parents=True, exist_ok=True)
  if emulate:
    import faulthandler
    faulthandler.enable()   # (a kernel's fault: where the server was, with TINYGPU_EMU_TRACE's last kernel)
  try: serve(argv[0] if argv else str(CACHE / ("emulator.sock" if emulate else "server.sock")), emulate)
  except KeyboardInterrupt: pass
  finally: print("tinygpu-webgpu: " + ("the emulator ended" if emulate else "the card released as Python exits"), flush=True)
