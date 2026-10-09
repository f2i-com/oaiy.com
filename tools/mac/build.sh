#!/bin/sh
# tools/mac/build.sh [--media]: build OAIY's engines on an Apple-silicon Mac (docs/MAC.md).
#
#   the Engines host (oaiy-studio: the control pages and the API gateway)
#   the language-model server (oaiy-llm-server: every model through WebGPU, which is Metal on a Mac)
#   --media: the picture, video and speech worker too (oaiy-media; a long build, and not yet tried on a Mac)
#
# Written and type-checked on Windows (`cargo check --target aarch64-apple-darwin`); not yet run on a Mac by its
# authors. It installs nothing: it says what is missing and stops.
set -eu
cd "$(dirname "$0")/../.."

[ "$(uname -s)" = "Darwin" ] || { echo "This is for a Mac (uname says $(uname -s))."; exit 1; }
[ "$(uname -m)" = "arm64" ] || echo "Note: this Mac is not Apple silicon ($(uname -m)); the engine's memory rule for a Mac's GPU is for Apple silicon."
xcode-select -p > /dev/null 2>&1 || { echo "Apple's command line tools are missing. Install them with: xcode-select --install"; exit 1; }
command -v cargo > /dev/null 2>&1 || { echo "Rust is missing. Install it from https://rustup.rs and open a new terminal."; exit 1; }

echo "Building the Engines host and the language-model server (the first build takes several minutes)..."
cargo build --release --locked -p oaiy-studio -p oaiy-llm-server
if [ "${1:-}" = "--media" ]; then
  echo "Building the picture, video and speech worker (a long build)..."
  cargo build --release --locked -p oaiy-media
fi

echo
echo "Built. Start the engines with:"
echo "  \"${CARGO_TARGET_DIR:-$(pwd)/target}/release/oaiy-studio\""
echo "Their control pages open in the browser (http://127.0.0.1:7860); the API is http://127.0.0.1:8080/v1."
