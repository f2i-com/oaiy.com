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
# Without Apple's command line tools, /usr/bin/git and /usr/bin/python3 are stand-ins that put up the dialog to
# install them: this script installs nothing, so it does not run those two then.
if xcode-select -p > /dev/null 2>&1; then tools=yes; else tools=""; fi
apples() { [ -z "$tools" ] && [ "$(command -v "$1" 2>/dev/null)" = "/usr/bin/$1" ]; }

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
if apples git; then say "this checkout: in $(pwd) (git not asked: it is Apple's stand-in until the command line tools are there)"
else say "this checkout: $(git rev-parse --short HEAD 2>/dev/null || echo "not a git checkout") in $(pwd)"; fi
for b in oaiy-studio oaiy-llm-server oaiy-media; do
  if [ -x "${CARGO_TARGET_DIR:-target}/release/$b" ]; then say "$b: built"; else say "$b: not built yet (sh tools/mac/build.sh)"; fi
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
say "== A USB Bluetooth dongle (for the phone link, Aokie, which does not run on a Mac yet: docs/MAC.md)"
# (each USB device that looks like a Bluetooth controller, with what macOS has attached to it: a driver of its own
# on the dongle is what a program that drives the dongle itself would have to take it from)
dongles=$(ioreg -r -c IOUSBHostDevice -w 0 2>/dev/null | sed 's/, id 0x[0-9a-f]*//; s/, retain [0-9]*//; s/, busy [0-9]* ([0-9]* ms)//' | awk '
  /^\+-o / { if (block ~ /[Bb]luetooth|BCM2070|RTL87|CSR8510/) printf "%s", block; block = "" }
  { block = block "  " $0 "\n" }
  END { if (block ~ /[Bb]luetooth|BCM2070|RTL87|CSR8510/) printf "%s", block }' | cut -c 1-150 | head -30)
if [ -n "$dongles" ]; then say "USB devices that look like one, and what macOS has attached to each:"; say "$dongles"; else say "no USB device that looks like one (is it plugged in?)"; fi
say "what macOS's own Bluetooth runs on:"
system_profiler SPBluetoothDataType 2>/dev/null | grep -E "^ *(State|Chipset|Transport|Vendor ID|Product ID|Firmware Version):" | sed 's/^ */  /' | head -8
switch=$(nvram bluetoothHostControllerSwitchBehavior 2>/dev/null | awk '{ print $2 }')
say "bluetoothHostControllerSwitchBehavior (whether macOS moves its own Bluetooth onto a dongle): ${switch:-not set}"
if have brew && brew list --versions libusb > /dev/null 2>&1; then say "libusb: $(brew list --versions libusb)"; else say "libusb: not installed by Homebrew"; fi

say
say "== Pythons, and which has tinygrad"
check='
import sys, os, importlib.util
out = "Python " + sys.version.split()[0]
try:
    import tinygrad
    folder = os.path.dirname(os.path.abspath(tinygrad.__file__))
    def has(n):
        try: return importlib.util.find_spec(n) is not None
        except Exception: return False
    server = next((n for n in ("tinygrad.llm", "tinygrad.apps.llm") if has(n)), None)
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
  if [ "$1" = /usr/bin/python3 ] && [ -z "$tools" ]; then say "$1: not asked (Apple's stand-in until the command line tools are there)"; return 0; fi
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
  if [ -z "$py" ] || { [ "$py" = /usr/bin/python3 ] && [ -z "$tools" ]; }; then
    say "no python3 to try it with (PYTHON=/path/to/python3 sh tools/mac/doctor.sh --egpu)"
  else
    say "with $py"
    export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"
    if [ -n "${TINYGRAD:-}" ]; then export PYTHONPATH="$TINYGRAD"; fi
    DEV="${DEV:-NV}" "$py" -c '
from tinygrad import Tensor, Device
print("tinygrad computes on:", Device.DEFAULT)
print("[1, 2, 3] + 1 =", (Tensor([1, 2, 3]) + 1).tolist())
' 2>&1 | tail -6
    say "(with another Python: PYTHON=/path/to/python3 sh tools/mac/doctor.sh --egpu)"
  fi
fi
