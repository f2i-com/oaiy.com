#!/bin/sh
# tools/mac/build.sh [--media] [--voice]: build OAIY's engines on an Apple-silicon Mac (docs/MAC.md).
#
#   the Engines host (oaiy-studio: the control pages and the API gateway)
#   the language-model server (oaiy-llm-server: every model through WebGPU, which is Metal on a Mac)
#   --media: the picture, video and speech worker too (oaiy-media; a long build)
#   --voice: OAIY Voice's server too (oaiy-voice: speech to text on the CPU, speech through WebGPU; a long build).
#     OAIY Desktop's service runs it from its own bin folder, where the last lines say to copy it.
#
# Run on an M5 Pro Mac (macOS 27). It installs nothing: it says what is missing and stops.
set -eu
cd "$(dirname "$0")/../.."

media=0 voice=0
for arg in "$@"; do
  case "$arg" in
    --media) media=1 ;;
    --voice) voice=1 ;;
    *) echo "unknown option: $arg (tools/mac/build.sh [--media] [--voice])"; exit 1 ;;
  esac
done

[ "$(uname -s)" = "Darwin" ] || { echo "This is for a Mac (uname says $(uname -s))."; exit 1; }
[ "$(uname -m)" = "arm64" ] || echo "Note: this Mac is not Apple silicon ($(uname -m)); the engine's memory rule for a Mac's GPU is for Apple silicon."
xcode-select -p > /dev/null 2>&1 || { echo "Apple's command line tools are missing. Install them with: xcode-select --install"; exit 1; }
command -v cargo > /dev/null 2>&1 || { echo "Rust is missing. Install it from https://rustup.rs and open a new terminal."; exit 1; }

echo "Building the Engines host and the language-model server (the first build takes several minutes)..."
cargo build --release --locked -p oaiy-studio -p oaiy-llm-server
if [ "$media" = 1 ]; then
  echo "Building the picture, video and speech worker (a long build)..."
  cargo build --release --locked -p oaiy-media
fi
if [ "$voice" = 1 ]; then
  echo "Building OAIY Voice's server (a long build)..."
  cargo build --release --locked -p oaiy-voice
fi

echo
echo "Built. Start the engines with:"
echo "  \"${CARGO_TARGET_DIR:-$(pwd)/target}/release/oaiy-studio\""
echo "Their control pages open in the browser (http://127.0.0.1:7860); the API is http://127.0.0.1:8080/v1."
if [ "$voice" = 1 ]; then
  echo "For OAIY Desktop's OAIY Voice service, copy oaiy-voice into its bin folder:"
  echo "  mkdir -p \"\$HOME/Library/Application Support/com.oaiy.app/bin\""
  echo "  cp \"${CARGO_TARGET_DIR:-$(pwd)/target}/release/oaiy-voice\" \"\$HOME/Library/Application Support/com.oaiy.app/bin/\""
fi
