"""tools/tinygpu/fast_check.py: webgpu_server.py's Fast launches against tinygrad's own, with no card.

    ~/tinygrad/.venv/bin/python tools/tinygpu/fast_check.py [LAUNCHES]

A stand-in for tinygrad's NV device (an Ada card's compute class; its argument ring host memory) is given to Fast and to
tinygrad's NVComputeQueue.exec alike, and each writes the same launches: programs of several constant banks and
signatures, random grids, pointers and values, chained in queues and ended by a release, as the server does. Their
arguments, QMDs, chains and releases must be the same bytes (each QMD's own address aside), and so must the queues'
words but for the addresses in them. The card's server checks the same as it runs with TINYGPU_FAST_CHECK=1; this
checks it before a change to Fast reaches a card (a QMD written wrong faults the card).
"""
import ctypes, os, random, struct, sys, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import webgpu_server as ws
import tinygrad.runtime.ops_nv as nv
from tinygrad import dtypes
from tinygrad.runtime.support.hcq import HCQBuffer, HCQProgram, BumpAllocator
from tinygrad.runtime.support.memory import MMIOInterface

RING = 4 << 20
ring_host = (ctypes.c_uint8 * RING)()
ring_va = 0x2300000000   # (where the card would see it: below 2^40, as a QMD must be, the same for both sides)
ring = HCQBuffer(ring_va, RING, view=MMIOInterface(ctypes.addressof(ring_host), RING))

dev = types.SimpleNamespace(
  iface=types.SimpleNamespace(compute_class=nv.nv_gpu.ADA_COMPUTE_A), pma_enabled=False,
  kernargs_buf=ring, kernargs_offset_allocator=BumpAllocator(RING, wrap=True))


def program(rng, nbufs: int, cbuf0: int):
  """A stand-in for an NVProgram: its constant bank 0 (`cbuf0` bytes), its QMD as tinygrad makes one, `nbufs` 32-bit
  values after its pointers (the server's binding sizes)."""
  p = types.SimpleNamespace()
  p.dev, p.name = dev, "oaiy_main"
  p.cbuf_0 = [rng.getrandbits(32) for _ in range(max(cbuf0 // 4, 12))]
  p.constbufs = {0: (0, cbuf0)}
  prog_addr = 0x7f1100000000 + rng.getrandbits(20) * 256
  p.qmd = nv.QMD(dev, qmd_major_version=3, sm_global_caching_enable=1, program_address_upper=prog_addr >> 32,
                 program_address_lower=prog_addr & 0xffffffff, shared_memory_size=rng.choice([0x400, 0x2000, 0xc400]),
                 register_count_v=rng.choice([32, 64, 128, 255]))
  p.signature = tuple((f"v{i}", i, dtypes.uint32, ()) for i in range(nbufs))
  p.kernargs_alloc_size = nv.round_up(cbuf0, 1 << 8) + 0x100 + 0x40
  p.args_state_t = nv.NVArgsState
  p.fill_kernargs = types.MethodType(HCQProgram.fill_kernargs, p)
  prog = types.SimpleNamespace(prg=p, workgroup=rng.choice([(32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1), (16, 16, 1), (8, 8, 4)]),
                               label=f"prog{nbufs}_{cbuf0}", scratch=False)
  return prog


def main(launches: int):
  rng = random.Random(7)
  progs = [program(rng, n, c) for n in (1, 2, 5, 9) for c in (0x160, 0x200, 0x380)]
  fast = ws.Fast(dev)
  card = ws.Card.__new__(ws.Card)
  card.fast, card.check = fast, True
  signal = types.SimpleNamespace(value_addr=0x7f3300000000 + 0x40)
  done, queues = 0, 0
  while done < launches:
    q, card.ref, fast.last = nv.NVComputeQueue(), nv.NVComputeQueue(), None
    for _ in range(rng.randint(1, 40)):
      prog = rng.choice(progs)
      n = len(prog.prg.signature)
      ptrs = [0x7f0000000000 + rng.getrandbits(32) * 16 for _ in range(n)]
      grid = (rng.randint(1, 65535), rng.choice([1, rng.randint(1, 1024)]), rng.choice([1, 1, rng.randint(1, 64)]))
      vals = tuple(rng.getrandbits(32) for _ in range(n))
      before = fast.last
      off = fast.launch(q, prog, ptrs, grid, vals)
      card.compare(prog, ptrs, grid, vals, off, before)
      done += 1
    # the queue's end: its last QMD released, as each side's queue writes it
    value = rng.getrandbits(40)
    fast.ending(q)
    q.signal(signal, value)
    card.ref.signal(signal, value)
    mine, theirs = q.active_qmd_buf.va_addr - ring_va, card.ref.active_qmd_buf.va_addr - ring_va
    a, b = bytearray(ring.cpu_view().mv[mine:mine + fast.qmd_size]), bytearray(ring.cpu_view().mv[theirs:theirs + fast.qmd_size])
    for spec in (fast.cbuf_hi, fast.cbuf_lo):
      for buf in (a, b):
        b0, b1, mask, _, _ = spec
        buf[b0:b1] = (int.from_bytes(buf[b0:b1], "little") & ~mask).to_bytes(b1 - b0, "little")
    assert a == b, "the last QMD's release differs from tinygrad's"
    # the queues' words: the same but for the first QMD's address (each its own)
    assert len(q._q) == len(card.ref._q), f"{len(q._q)} words, tinygrad's {len(card.ref._q)}"
    differ = [i for i, (x, y) in enumerate(zip(q._q, card.ref._q)) if x != y]
    assert len(differ) <= 1, f"words {differ} differ"
    queues += 1
  print(f"fast launches: {done} in {queues} queues, each the same bytes as tinygrad's")


if __name__ == "__main__":
  main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
