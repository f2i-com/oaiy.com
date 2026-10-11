"""tools/tinygpu/plugin/bundle.py: the eGPU plugin for OAIY Desktop, built into a folder (docs/TINYGPU.md, "The eGPU
plugin").

    python3 tools/tinygpu/plugin/bundle.py [OUT] [--tinygrad DIR] [--python DIR] [--arch sm_89,...] [--no-firmware] [--zip]

OUT (target/egpu-plugin unless named) is the plugin, for OAIY's Plugins page to install (or OUT.zip, with --zip):

  manifest.json
  python/          a CPython 3.12 of its own (python-build-standalone's, as uv installs it: one program that needs only
                   macOS's libraries), without what the card never uses (Tk, IDLE, the tests, the headers)
  lib/             tinygrad (the checkout --tinygrad names; its commit in tinygrad-commit.txt), jinja2 and markupsafe
  card/            plugin.py (what OAIY talks to), egpu_serve.py (OAIY's launcher), webgpu_server.py (the card's server)
  card/kernels/    the kernel pack for --arch (tools/tinygpu/kernel_pack.py: nvcc only for what the cache lacks)
  card/firmware/   what tinygrad downloads as it first opens a card: NVIDIA's GSP firmware (linux-firmware's, for every
                   card tinygrad knows) and its TinyGPU app, from tinygrad's download cache where it has them. Not
                   with --no-firmware, for a plugin given to others: both are other people's programs (NVIDIA's
                   under LICENCE.nvidia, tinygrad's app with no licence stated where it is published), and tinygrad
                   downloads them itself, from where they are published, the first time it opens a card

There are no links in it (OAIY refuses a plugin with one), and every Python file has its .pyc beside it, checked by
nothing (unchecked-hash): a plugin is trusted by the digest of its files, so its Python runs with -B and writes none.
Nothing in it names the folders of the machine that built it: the bytecode records its files from the plugin's root
(Python puts a file's real place in as it loads one), and the Python's own build notes name its builder's folder.
"""
from __future__ import annotations

import argparse, hashlib, os, pathlib, re, shutil, subprocess, sys, urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
# what OAIY takes in a plugin (platform/desktop's plugins/install.rs and trust.rs: entries, files and bytes)
MOST_FILES, MOST_BYTES = 20_000, 512 << 20
# what the card never uses, of the Python's own library
TRIMMED = {"test", "idlelib", "tkinter", "turtledemo", "turtle.py", "ensurepip", "lib2to3", "venv", "site-packages", "__pycache__"}


def copy_files(src: pathlib.Path, dst: pathlib.Path, skip=lambda rel: False) -> int:
  """The regular files under `src` copied to `dst` (links and __pycache__ left out). How many."""
  n = 0
  for root, dirs, files in os.walk(src):
    rel = pathlib.Path(root).relative_to(src)
    dirs[:] = [d for d in dirs if d != "__pycache__" and not (pathlib.Path(root) / d).is_symlink() and not skip(rel / d)]
    for f in files:
      if (pathlib.Path(root) / f).is_symlink() or skip(rel / f) or f == ".DS_Store": continue
      (dst / rel).mkdir(parents=True, exist_ok=True)
      shutil.copy2(pathlib.Path(root) / f, dst / rel / f)
      n += 1
  return n


def ask_python(python: pathlib.Path, code: str) -> str:
  return subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def the_python(src: pathlib.Path, out: pathlib.Path):
  """`src`'s CPython (a python-build-standalone install: bin/python3.12 and lib/python3.12) into `out`."""
  program = src / "bin" / "python3.12"
  if not program.is_file(): sys.exit(f"{src} has no bin/python3.12: --python names a CPython 3.12 install (uv's)")
  linked = subprocess.run(["otool", "-L", str(program)], capture_output=True, text=True).stdout
  if any(l.strip() and not l.strip().startswith(("/usr/lib/", "/System/")) for l in linked.splitlines()[1:]):
    sys.exit(f"{program} needs libraries besides macOS's:\n{linked}")
  (out / "bin").mkdir(parents=True)
  shutil.copy2(program.resolve(), out / "bin" / "python3.12")
  stdlib = src / "lib" / "python3.12"
  def skip(rel: pathlib.Path) -> bool:
    return rel.parts[0] in TRIMMED or rel.parts[0].startswith("config-") or (rel.parts[0] == "lib-dynload" and "_tkinter" in rel.name)
  n = copy_files(stdlib, out / "lib" / "python3.12", skip)
  # What the Python was built with (read only to build an extension, which nothing here does) names the folder uv
  # installed it in, the home folder of whoever builds the plugin in it: back to python-build-standalone's own.
  for notes in (out / "lib" / "python3.12").glob("_sysconfigdata*.py"):
    text = notes.read_text()
    if str(src) in text: notes.write_text(text.replace(str(src), "/install"))
  print(f"python: {src} ({n} files of its library)")


def tinygrad(src: pathlib.Path, venv_python: pathlib.Path, out: pathlib.Path):
  """tinygrad's package from the checkout `src`, and what its LLM server uses from the checkout's Python: jinja2 (and
  markupsafe, which it needs) for a model's chat template."""
  if not (src / "tinygrad" / "__init__.py").is_file(): sys.exit(f"{src} is not a checkout of tinygrad (no tinygrad/__init__.py)")
  n = copy_files(src / "tinygrad", out / "tinygrad")
  commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
  dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no", "tinygrad"], capture_output=True, text=True).stdout.strip()
  (out / "tinygrad-commit.txt").write_text(commit + (" (with changes not committed)" if dirty else "") + "\n")
  if (src / "LICENSE").is_file(): shutil.copy2(src / "LICENSE", out / "tinygrad-LICENSE")
  print(f"tinygrad: {src} at {commit[:12]}{' (dirty)' if dirty else ''} ({n} files)")
  site = pathlib.Path(ask_python(venv_python, "import jinja2, os; print(os.path.dirname(os.path.dirname(jinja2.__file__)))"))
  for package in ("jinja2", "markupsafe"):
    copy_files(site / package, out / package)
    for info in site.glob(f"{package}-*.dist-info"):
      copy_files(info, out / info.name)
  print(f"jinja2, markupsafe: {site}")


def tinygrads_downloads() -> pathlib.Path:
  """Where tinygrad keeps what it downloads (its helpers' cache_dir: XDG_CACHE_HOME, else ~/Library/Caches on a Mac)."""
  base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/Library/Caches" if sys.platform == "darwin" else "~/.cache")
  return pathlib.Path(base) / "tinygrad" / "downloads"


def fetch(url: str, kept: pathlib.Path | None, to: pathlib.Path) -> str:
  """`url` into `to`: from tinygrad's download of it where there is one (`kept`), else downloaded. What happened."""
  to.parent.mkdir(parents=True, exist_ok=True)
  if kept is not None and kept.is_file():
    shutil.copyfile(kept, to)
    return "from tinygrad's cache"
  try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "oaiy-egpu-bundle"}), timeout=120) as r:
      to.write_bytes(r.read())
    return "downloaded"
  except urllib.error.HTTPError as e:
    return f"not in linux-firmware ({e.code}): not one that card loads"


def firmware(src: pathlib.Path, out: pathlib.Path):
  """NVIDIA's GSP firmware for every card tinygrad's NV driver knows, as it names and loads it (tinygrad checks each
  file's sha256 as it loads it), its licence, and tinygrad's TinyGPU app (the zip of the commit tinygrad asks for)."""
  helpers = (src / "tinygrad" / "helpers.py").read_text()
  ip = (src / "tinygrad" / "runtime" / "support" / "nv" / "ip.py").read_text()
  nvdev = (src / "tinygrad" / "runtime" / "support" / "nv" / "nvdev.py").read_text()
  system = (src / "tinygrad" / "runtime" / "support" / "system.py").read_text()
  base = re.search(r'"(https://gitlab\.com/kernel-firmware/linux-firmware/-/raw/[0-9a-f]{40})/', helpers)
  families = re.findall(r'"(?:GB2|AD1|GA1|GH1|GB1)":\s*"(\w+)"', nvdev)
  loads = re.findall(r'fetch_fw\(f?"(nvidia/[^"]+)",\s*"([^"]+)"', ip)
  if not (base and families and loads): sys.exit("tinygrad's firmware is not named as this script reads it (helpers.fetch_fw, nv/ip.py): update firmware()")
  wanted = sorted({(path.replace("{self.nvdev.fw_name}", fam), name) for path, name in loads for fam in families})
  for path, name in wanted:
    url = f"{base.group(1)}/{path}/{name}"
    said = fetch(url, tinygrads_downloads() / "fw" / hashlib.md5(url.encode()).hexdigest(), out / path / name)
    print(f"firmware: {path}/{name}: {said}")
  print(f"firmware: LICENCE.nvidia: {fetch(base.group(1) + '/LICENSES/LICENCE.nvidia', None, out / 'LICENCE.nvidia')}")
  app = re.search(r'def ensure_app[\s\S]*?commit = "([0-9a-f]{40})"', system)
  if not app: sys.exit("tinygrad's TinyGPU app is not named as this script reads it (system.py's ensure_app): update firmware()")
  zipped = f"TinyGPU_{app.group(1)}.zip"
  said = fetch(f"https://github.com/tinygrad/tinygpu_releases/raw/{app.group(1)}/TinyGPU.zip", tinygrads_downloads() / zipped, out / zipped)
  print(f"firmware: {zipped}: {said}")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("out", nargs="?", default=str(REPO / "target" / "egpu-plugin"))
  ap.add_argument("--tinygrad", default=os.path.expanduser("~/tinygrad"), help="a checkout of tinygrad, with its Python in .venv")
  ap.add_argument("--python", help="a CPython 3.12 install (python-build-standalone's); the checkout's Python's own by default")
  ap.add_argument("--arch", default="sm_89", help="the cards' architectures, comma-separated (sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50)")
  ap.add_argument("--sources", action="append", default=[], help="more kernel sources for the pack (kernel_pack.py --sources)")
  ap.add_argument("--no-firmware", action="store_true", help="without NVIDIA's firmware and tinygrad's TinyGPU app: tinygrad downloads them as it first opens a card")
  ap.add_argument("--zip", action="store_true", help="and OUT.zip of it, to install")
  args = ap.parse_args()
  sys.stdout.reconfigure(line_buffering=True)   # (in order with the kernel pack's)
  out, src = pathlib.Path(args.out).resolve(), pathlib.Path(args.tinygrad).expanduser().resolve()
  venv_python = src / ".venv" / "bin" / "python3"
  python = pathlib.Path(args.python).expanduser() if args.python else pathlib.Path(ask_python(venv_python, "import sys; print(sys.base_prefix)"))
  if out.exists():
    if not (out / "manifest.json").is_file(): sys.exit(f"{out} is there and is not a plugin this made: name another folder")
    shutil.rmtree(out)
  out.mkdir(parents=True)
  shutil.copy2(HERE / "manifest.json", out / "manifest.json")
  the_python(python.resolve(), out / "python")
  tinygrad(src, venv_python, out / "lib")
  card = out / "card"
  card.mkdir()
  for f in (HERE / "plugin.py", REPO / "crates" / "oaiy-studio" / "src" / "egpu_serve.py", HERE.parent / "webgpu_server.py"):
    shutil.copy2(f, card / f.name)
  if args.no_firmware: print("firmware: left out (tinygrad downloads NVIDIA's and its TinyGPU app as it first opens a card)")
  else: firmware(src, card / "firmware")
  pack = [sys.executable, str(HERE.parent / "kernel_pack.py"), str(card / "kernels"), "--arch", args.arch]
  for s in args.sources: pack += ["--sources", s]
  subprocess.run(pack, check=True)
  program = out / "python" / "bin" / "python3.12"
  # (-s, -p: each file recorded as egpu/... and not by where this machine built it)
  subprocess.run([str(program), "-I", "-B", "-m", "compileall", "-q", "-f", "-j", "0", "--invalidation-mode", "unchecked-hash",
                  "-s", str(out), "-p", "egpu", str(out / "python" / "lib" / "python3.12"), str(out / "lib"), str(card)], check=True)
  said = subprocess.run([str(program), "-I", "-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import tinygrad, jinja2, ssl, sqlite3, ctypes; print(sys.version.split()[0])",
                         str(out / "lib")], capture_output=True, text=True)
  if said.returncode != 0: sys.exit(f"the plugin's Python does not start with tinygrad:\n{said.stderr}")
  everything = sorted(out.rglob("*"))
  links = [p for p in everything if p.is_symlink()]
  files = [p for p in everything if p.is_file() and not p.is_symlink()]
  size = sum(p.stat().st_size for p in files)
  if links: sys.exit(f"links in the plugin (OAIY refuses them): {[str(p) for p in links[:10]]}")
  home = str(pathlib.Path.home()).encode()
  named = [p for p in files if p.suffix in (".py", ".pyc", ".json", ".txt") and home in p.read_bytes()]
  if named: sys.exit(f"files in the plugin that name this machine's home folder: {[str(p.relative_to(out)) for p in named[:10]]}")
  if len(everything) > MOST_FILES or size > MOST_BYTES:
    sys.exit(f"{len(everything)} files and folders, {size / 2**20:.0f} MiB: more than OAIY takes in a plugin ({MOST_FILES}, {MOST_BYTES >> 20} MiB)")
  print(f"the eGPU plugin, Python {said.stdout.strip()}: {len(files)} files, {size / 2**20:.0f} MiB, in {out}")
  if args.zip:
    import zipfile
    zipped = out.with_suffix(".zip")
    # (each file's mode with it: OAIY's install keeps a program's executable bit from the archive)
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
      for p in everything:
        z.write(p, p.relative_to(out))
    print(f"zipped: {zipped} ({zipped.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
  main()
