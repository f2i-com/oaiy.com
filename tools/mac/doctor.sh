#!/bin/sh
# tools/mac/doctor.sh [--egpu]: what this Mac has for OAIY, as text to read or send on (docs/MAC.md).
#
# It changes nothing and installs nothing. With --egpu it also has tinygrad add three numbers on the card in the
# Thunderbolt enclosure (DEV=NV, or the DEV you set), which shows whether tinygrad reaches it: the first time that
# takes a while (tinygrad compiles for the card).
#
# Written on Windows and not yet run on a Mac by its authors: a line that fails says so and the rest goes on.
say() { printf '%s\n' "$*"; }
have() { command -v "$1" > /dev/null 2>&1; }

say "== This Mac"
say "macOS $(sw_vers -productVersion 2>/dev/null) ($(uname -m))"
say "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
mem=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
say "memory: $((mem / 1073741824)) GB"
wired=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "")
say "the GPU's share of it (iogpu.wired_limit_mb): ${wired:-not readable}$( [ "${wired:-0}" = "0" ] && echo " (0 = macOS decides; OAIY then gives its GPU what a card with two thirds of the memory would hold)")"

say
say "== To build OAIY"
if xcode-select -p > /dev/null 2>&1; then say "Apple's command line tools: $(xcode-select -p)"; else say "Apple's command line tools: MISSING (xcode-select --install)"; fi
if have cargo; then say "Rust: $(cargo --version)"; else say "Rust: MISSING (https://rustup.rs)"; fi
if have node; then say "Node: $(node --version) (the desktop app's pages need 22 or later)"; else say "Node: missing (only the desktop app needs it, not the engines)"; fi

say
say "== The eGPU (tinygrad's TinyGPU)"
if [ -d /Applications/TinyGPU.app ]; then say "TinyGPU.app: in /Applications"; else say "TinyGPU.app: not in /Applications (it may be elsewhere)"; fi
ext=$(systemextensionsctl list 2>/dev/null | grep -i tinygpu | head -3)
say "its driver extension: ${ext:-none listed by systemextensionsctl}"
if have docker; then say "Docker: $(docker --version 2>/dev/null)"; else say "Docker: not on this shell's PATH (an NVIDIA card's kernels are compiled in it)"; fi
for t in nvcc nvdisasm; do
  if [ -x "$HOME/.local/bin/$t" ]; then say "$t: $HOME/.local/bin/$t"; else say "$t: not in ~/.local/bin (tinygrad's extra/setup_nvcc_osx.sh puts it there)"; fi
done
say "Thunderbolt devices:"
system_profiler SPThunderboltDataType 2>/dev/null | grep -E "Device Name|Vendor Name" | sed 's/^ */  /' | head -12

say
say "== Pythons, and which has tinygrad"
check='
import sys, os, importlib.util
out = "Python " + sys.version.split()[0]
try:
    import tinygrad
    folder = os.path.dirname(os.path.abspath(tinygrad.__file__))
    server = next((n for n in ("tinygrad.llm", "tinygrad.apps.llm") if importlib.util.find_spec(n) is not None), None)
    commit = ""
    try:
        git = os.path.join(os.path.dirname(folder), ".git")
        head = open(os.path.join(git, "HEAD")).read().strip()
        if head.startswith("ref: "): head = open(os.path.join(git, *head[5:].split("/"))).read().strip()
        commit = ", commit " + head[:12]
    except Exception: pass
    out += ", tinygrad in " + folder + commit + (", LLM server: " + server if server else ", NO LLM server (too old)")
    try:
        import jinja2
    except Exception:
        out += ", jinja2 MISSING (pip install jinja2: chat formats and tool calls need it)"
except Exception as e:
    out += ", no tinygrad (" + str(e) + ")"
print(out)
'
found=""
for p in "${TINYGRAD:-}/.venv/bin/python3" "${TINYGRAD:-}/venv/bin/python3" /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3 "$(command -v python3 2>/dev/null)"; do
  [ -x "$p" ] || continue
  case " $found " in *" $p "*) continue ;; esac
  found="$found $p"
  if [ -n "${TINYGRAD:-}" ]; then line=$(PYTHONPATH="$TINYGRAD" "$p" -c "$check" 2>&1 | tail -1); else line=$("$p" -c "$check" 2>&1 | tail -1); fi
  say "$p: $line"
done
[ -n "$found" ] || say "no python3 found"
say "(a checkout of tinygrad that is not installed: run this as  TINYGRAD=/path/to/tinygrad sh tools/mac/doctor.sh)"

if [ "${1:-}" = "--egpu" ]; then
  say
  say "== tinygrad on the card (DEV=${DEV:-NV})"
  py="${PYTHON:-$(command -v python3)}"
  export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
  if [ -n "${TINYGRAD:-}" ]; then export PYTHONPATH="$TINYGRAD"; fi
  DEV="${DEV:-NV}" "$py" -c '
from tinygrad import Tensor, Device
print("tinygrad computes on:", Device.DEFAULT)
print("[1, 2, 3] + 1 =", (Tensor([1, 2, 3]) + 1).tolist())
' 2>&1 | tail -6
  say "(with another Python: PYTHON=/path/to/python3 sh tools/mac/doctor.sh --egpu)"
fi
