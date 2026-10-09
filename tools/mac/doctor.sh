#!/bin/sh
# tools/mac/doctor.sh [--egpu]: what this Mac has for OAIY, as text to read or send on (docs/MAC.md).
#
# It changes nothing and installs nothing. With --egpu it also has tinygrad add three numbers on the card in the
# Thunderbolt enclosure (DEV=NV, or the DEV you set), which shows whether tinygrad reaches it: the first time that
# takes a while (tinygrad compiles for the card).
#
# Written on Windows and not yet run on a Mac by its authors (there it was run with stand-ins for a Mac's own
# programs): a line that fails says so and the rest goes on.
say() { printf '%s\n' "$*"; }
have() { command -v "$1" > /dev/null 2>&1; }
cd "$(dirname "$0")/../.." 2>/dev/null || true

say "== This Mac"
say "macOS $(sw_vers -productVersion 2>/dev/null) ($(uname -m))"
say "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
fast=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "")
slow=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "")
[ -n "$fast" ] && say "cores: $fast performance, ${slow:-0} efficiency"
mem=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
say "memory: $((mem / 1073741824)) GB"
vm_stat 2>/dev/null | awk '/page size of/ { p = $8 } /^Pages free/ { f = $3 } /^Pages inactive/ { i = $3 } /^Pages speculative/ { s = $3 }
  END { if (p) printf "free for a model now: %.1f GB (free, inactive and speculative pages)\n", (f + i + s) * p / 1073741824 }'
wired=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "")
case "$wired" in
  "") say "the GPU's share of the memory (iogpu.wired_limit_mb): not readable (OAIY then gives its GPU what a card with two thirds of the memory would hold)" ;;
  0) say "the GPU's share of the memory (iogpu.wired_limit_mb): 0, which is macOS deciding (OAIY then gives its GPU what a card with two thirds of the memory would hold)" ;;
  *) say "the GPU's share of the memory (iogpu.wired_limit_mb): $wired MB, set on this Mac (OAIY gives its GPU what a card with that much would hold)" ;;
esac
system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores|Metal" | sed 's/^ */GPU: /' | head -4
say "free on this drive: $(df -h . 2>/dev/null | awk 'NR == 2 { print $4 }')"

say
say "== To build OAIY"
if xcode-select -p > /dev/null 2>&1; then say "Apple's command line tools: $(xcode-select -p)"; else say "Apple's command line tools: MISSING (xcode-select --install)"; fi
if have cargo; then say "Rust: $(cargo --version) / $(rustc --version 2>/dev/null)"; else say "Rust: MISSING (https://rustup.rs)"; fi
if have node; then say "Node: $(node --version) (the desktop app's pages need 22 or later)"; else say "Node: missing (only the desktop app needs it, not the engines)"; fi

say
say "== OAIY here"
say "this checkout: $(git rev-parse --short HEAD 2>/dev/null || echo "not a git checkout") in $(pwd)"
for b in oaiy-studio oaiy-llm-server oaiy-media; do
  if [ -x "target/release/$b" ]; then say "$b: built"; else say "$b: not built yet (sh tools/mac/build.sh)"; fi
done

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
# (the first of them that has tinygrad's LLM server is the one --egpu tries the card with)
found=""
with=""
probe() {
  [ -x "$1" ] || return 0
  case " $found " in *" $1 "*) return 0 ;; esac
  found="$found $1"
  if [ -n "${TINYGRAD:-}" ]; then line=$(PYTHONPATH="$TINYGRAD" "$1" -c "$check" 2>&1 | tail -1); else line=$("$1" -c "$check" 2>&1 | tail -1); fi
  say "$1: $line"
  case "$line" in *"LLM server: "*) [ -n "$with" ] || with="$1" ;; esac
}
if [ -n "${TINYGRAD:-}" ]; then probe "$TINYGRAD/.venv/bin/python3"; probe "$TINYGRAD/venv/bin/python3"; fi
probe /opt/homebrew/bin/python3
probe /usr/local/bin/python3
probe /usr/bin/python3
probe "$(command -v python3 2>/dev/null)"
[ -n "$found" ] || say "no python3 found"
say "(a checkout of tinygrad that is not installed: run this as  TINYGRAD=/path/to/tinygrad sh tools/mac/doctor.sh)"

if [ "${1:-}" = "--egpu" ]; then
  say
  say "== tinygrad on the card (DEV=${DEV:-NV})"
  py="${PYTHON:-${with:-$(command -v python3 2>/dev/null)}}"
  if [ -z "$py" ]; then
    say "no python3 to try it with (PYTHON=/path/to/python3 sh tools/mac/doctor.sh --egpu)"
  else
    say "with $py"
    export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    if [ -n "${TINYGRAD:-}" ]; then export PYTHONPATH="$TINYGRAD"; fi
    DEV="${DEV:-NV}" "$py" -c '
from tinygrad import Tensor, Device
print("tinygrad computes on:", Device.DEFAULT)
print("[1, 2, 3] + 1 =", (Tensor([1, 2, 3]) + 1).tolist())
' 2>&1 | tail -6
    say "(with another Python: PYTHON=/path/to/python3 sh tools/mac/doctor.sh --egpu)"
  fi
fi
