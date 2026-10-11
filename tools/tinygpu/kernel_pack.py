"""tools/tinygpu/kernel_pack.py: a kernel pack, the eGPU's kernels compiled ahead of time, so a card needs no compiler.

    python3 tools/tinygpu/kernel_pack.py OUT [--arch sm_89,sm_86,...] [--sources DIR ...] [--jobs N]

The card's kernels come to webgpu_server.py as CUDA C++ (crates/wgsl-cuda's), and nvcc makes their cubins: a Mac has
nvcc only through Docker (a Linux container with NVIDIA's toolkit). A pack is a folder of those cubins, each named as
the server's cache names it (the hash of the architecture and the source), which the server looks in first: shipped
in the eGPU plugin, the Mac that runs it needs neither nvcc nor Docker for any kernel the pack has.

Its kernels are the sources the server has seen: each cubin in the cache has its .cu beside it, and a kernel no pack
had, on a card with no nvcc, is kept in the cache's `missing` folder. Those (and the folders --sources names) are
compiled for every --arch (default sm_89, the RTX 40 series; sm_86 the RTX 30, sm_120 the RTX 50), a cubin already in
the cache copied rather than compiled again. This is where nvcc is needed: on a Linux machine with NVIDIA's toolkit (no
GPU), or on a Mac through the Docker shim. OUT gets the cubins and `index.json` (what the pack holds), and a pack built
again into the same folder keeps what it had.
"""
import argparse, datetime, hashlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

CACHE = pathlib.Path(os.environ.get("TINYGPU_CACHE") or pathlib.Path.home() / ".cache" / "tinygpu-webgpu")


def kernel_name(arch: str, src: str) -> str:
  """As webgpu_server.kernel_name names a kernel's cubin."""
  return hashlib.sha256((arch + "\0" + src).encode()).hexdigest()[:24]


def sources(folders: list, named: bool) -> dict:
  """Every kernel source in `folders` (their .cu files), by the hash of its text: each once. `named`: only those the
  server wrote, named by their kernel's hash (HASH.cu; an earlier server left HASH.PID.cu), not a test's own."""
  out = {}
  for folder in folders:
    for cu in sorted(pathlib.Path(folder).glob("*.cu")):
      if not named or re.fullmatch(r"[0-9a-f]{24}(\.\d+)*", cu.stem):
        text = cu.read_text()
        out[hashlib.sha256(text.encode()).hexdigest()] = text
  return out


def compile_one(arch: str, src: str, out: pathlib.Path) -> str:
  """`src`'s cubin for `arch` into `out`: from the cache where it is there, else nvcc's. What happened, in a word."""
  name = kernel_name(arch, src)
  target = out / f"{name}.cubin"
  if target.exists():
    return "had"
  cached = CACHE / f"{name}.cubin"
  if cached.exists():
    shutil.copyfile(cached, target)
    return "copied"
  with tempfile.TemporaryDirectory(dir=os.environ.get("TINYGPU_PACK_TMP")) as tmp:
    cu, cubin = pathlib.Path(tmp) / "k.cu", pathlib.Path(tmp) / "k.cubin"
    cu.write_text(src)
    run = subprocess.run(["nvcc", f"-arch={arch}", "-cubin", "-o", str(cubin), str(cu)], capture_output=True, text=True)
    if run.returncode != 0:
      raise RuntimeError(f"nvcc ({arch}, {name}): {run.stderr.strip()[-2000:]}")
    shutil.copyfile(cubin, target)
  return "compiled"


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("out")
  ap.add_argument("--arch", default="sm_89", help="comma-separated: sm_86, sm_89, sm_120 ...")
  ap.add_argument("--sources", action="append", default=[], help="a folder of kernel sources (.cu), besides the cache's")
  ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  archs = [a.strip() for a in args.arch.split(",") if a.strip()]
  found = {**sources([CACHE, CACHE / "missing"], named=True), **sources(args.sources, named=False)}
  # the server's own kernels (its copies and clears), compiled as it starts: a pack without them is no start
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
  from webgpu_server import BUILTINS
  found.update({hashlib.sha256(src.encode()).hexdigest(): src for src in BUILTINS.values()})
  if not found:
    sys.exit(f"no kernel sources in {CACHE}: run the eGPU once (with nvcc) to make them, or give --sources")
  work = [(arch, src) for arch in archs for src in found.values()]
  if shutil.which("nvcc") is None and any(not (CACHE / f"{kernel_name(a, s)}.cubin").exists() and not (out / f"{kernel_name(a, s)}.cubin").exists() for a, s in work):
    sys.exit("nvcc is needed to compile what the cache does not have (NVIDIA's toolkit on Linux, or the Docker shim on a Mac)")
  done, failed = {"had": 0, "copied": 0, "compiled": 0}, []
  with ThreadPoolExecutor(max_workers=args.jobs) as pool:
    for (arch, src), result in zip(work, pool.map(lambda w: _try(compile_one, *w, out), work)):
      if isinstance(result, Exception):
        failed.append(f"{arch} {kernel_name(arch, src)}: {result}")
      else:
        done[result] += 1
  previous = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else {}
  cubins = list(out.glob("*.cubin"))
  index = {"format": 1, "archs": sorted(set(archs) | set(previous.get("archs", []))), "kernels": len(cubins), "sources": len(found),
           "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "bytes": sum(p.stat().st_size for p in cubins)}
  (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
  print(f"{len(found)} kernel sources for {', '.join(archs)}: {done['had']} in the pack already, {done['copied']} from the cache, "
        f"{done['compiled']} compiled; {len(failed)} failed. The pack: {index['kernels']} cubins, {index['bytes'] / 1e6:.1f} MB, in {out}")
  for line in failed[:20]:
    print("  FAILED", line[:400])
  sys.exit(1 if failed else 0)


def _try(fn, *args):
  try:
    return fn(*args)
  except Exception as e:  # (reported with its kernel; the others go on)
    return e


if __name__ == "__main__":
  main()
