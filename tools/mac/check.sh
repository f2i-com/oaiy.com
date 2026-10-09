#!/bin/sh
# tools/mac/check.sh: the GPU backend's own tests, on this Mac's GPU (docs/MAC.md).
#
# Each test makes kernels, runs them on the GPU and holds the results against the CPU's: on a Mac that is Apple's
# compiler reading every kernel, Metal's limits and Metal's arithmetic, none of which the authors could run (they
# held another GPU to Metal's limits instead). It prints a summary to read or send on and keeps the whole text in a
# file. It changes nothing but the build folder. The first run builds the tests, which takes several minutes.
set -u
cd "$(dirname "$0")/../.."

[ "$(uname -s)" = "Darwin" ] || { echo "This is for a Mac (uname says $(uname -s))."; exit 1; }
xcode-select -p > /dev/null 2>&1 || { echo "Apple's command line tools are missing. Install them with: xcode-select --install"; exit 1; }
command -v cargo > /dev/null 2>&1 || { echo "Rust is missing. Install it from https://rustup.rs and open a new terminal."; exit 1; }

out="${TMPDIR:-/tmp}/oaiy-mac-check.txt"
out=$(printf '%s' "$out" | sed 's|//|/|g')
echo "Running the GPU backend's tests on this Mac's GPU (two at a time; several minutes)..."
# (two at a time: each test opens the GPU for itself, and on a Mac they all share the computer's memory)
cargo test --release --locked -p ggml-rs-wgpu --lib -- --test-threads=2 > "$out" 2>&1
code=$?

echo
echo "== $(sysctl -n machdep.cpu.brand_string 2>/dev/null), macOS $(sw_vers -productVersion 2>/dev/null), OAIY $(git rev-parse --short HEAD 2>/dev/null)"
if grep -q "^test result" "$out"; then
  grep "^test result" "$out"
  failed=$(grep -c "\.\.\. FAILED$" "$out")
  if [ "$failed" -gt 0 ]; then
    echo
    echo "== The $failed that failed, and what each said"
    grep "\.\.\. FAILED$" "$out" | sed 's/ \.\.\. FAILED$//; s/^test /  /'
    echo
    grep -A 6 "panicked at" "$out" | grep -v "^note: \|^$" | cut -c 1-600 | head -120
  fi
else
  echo "The tests did not run (exit $code). The end of what was said:"
  tail -30 "$out" | cut -c 1-600
fi
echo
echo "The whole text is in $out"
exit "$code"
