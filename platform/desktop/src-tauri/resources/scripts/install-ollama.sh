#!/usr/bin/env sh
# install-ollama.sh
#
# Installs Ollama on Linux via the official install script. Ollama puts itself
# on PATH, so the template's `command: "ollama"` works without OAIY_BIN_DIR.
# No-op if already installed.

set -e

if command -v ollama >/dev/null 2>&1; then
  echo "[install-ollama] already installed: $(command -v ollama)"
  exit 0
fi

echo "[install-ollama] installing via https://ollama.com/install.sh ..."
# `curl ... | sh` masks a download failure: the pipeline's exit status is sh's,
# and sh exits 0 on empty stdin (POSIX dash has no pipefail). Download to a file
# first so a network/proxy error aborts here under `set -e`.
tmp="$(mktemp 2>/dev/null || echo "/tmp/ollama-install.$$.sh")"
curl -fsSL https://ollama.com/install.sh -o "$tmp"
sh "$tmp"
rm -f "$tmp"

# The official installer enables + starts a systemd `ollama` service on :11434.
# oaiy manages ollama itself (its template runs `ollama serve`), so stop + disable
# the unit to free the port — otherwise oaiy's `ollama serve` can't bind. Guarded
# so a non-systemd / non-root / container host doesn't abort the install (set -e).
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop ollama 2>/dev/null || true
  sudo systemctl disable ollama 2>/dev/null || true
fi

# Authoritative post-check: a missing binary (failed install) must exit non-zero so
# reap_exited maps it to Errored instead of the UI showing a false success. The
# official installer writes to /usr/local/bin or /usr/bin (already on PATH).
if ! command -v ollama >/dev/null 2>&1; then
  echo "[install-ollama] ollama not found after install — see https://ollama.com/download" >&2
  exit 1
fi
echo "[install-ollama] installed: $(command -v ollama)"

# Pre-pull the tiny default model the companion's Ollama node targets
# (qwen2.5:0.5b, ~0.4 GB), so a freshly-installed node RUNS without a manual
# pull — Ollama's OpenAI /v1/chat/completions 404s an un-pulled model. The
# systemd unit was disabled above, so start a transient server, pull, then stop
# it. Best-effort: never fail the install on a pull error.
if command -v ollama >/dev/null 2>&1; then
  echo "[install-ollama] pre-pulling default model qwen2.5:0.5b"
  ollama serve >/dev/null 2>&1 &
  serve_pid=$!
  i=0
  while [ $i -lt 20 ]; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 0.5
    i=$((i + 1))
  done
  ollama pull qwen2.5:0.5b || echo "[install-ollama] pull failed; run 'ollama pull qwen2.5:0.5b' manually."
  kill "$serve_pid" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
fi
