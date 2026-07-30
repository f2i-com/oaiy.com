#!/usr/bin/env sh
# install-playwright.sh
#
# Sets up the Playwright browser backend (Linux): a Python venv with playwright
# + Chromium. Reuses the companion's bundled Python runtime when present, else a
# system python3. Env from the companion: OAIY_BIN_DIR (= <dataDir>/bin).

set -e

if [ -z "$OAIY_BIN_DIR" ]; then
  echo "[install-playwright] OAIY_BIN_DIR not set; run via the OAIY app/server." >&2
  exit 1
fi

dataDir=$(dirname "$OAIY_BIN_DIR")
venv="$dataDir/venvs/playwright"
venvPy="$venv/bin/python"

companionPy="$dataDir/python/bin/python3"
if [ -x "$companionPy" ]; then
  basePy="$companionPy"
  echo "[install-playwright] using companion Python: $basePy"
elif command -v python3 >/dev/null 2>&1; then
  basePy="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  basePy="$(command -v python)"
else
  echo "[install-playwright] no Python found. Run 'oaiy python install' or install system python3." >&2
  exit 1
fi

if [ ! -x "$venvPy" ]; then
  echo "[install-playwright] creating venv at $venv"
  "$basePy" -m venv "$venv"
fi

echo "[install-playwright] installing playwright (pip)"
"$venvPy" -m pip install --upgrade pip playwright

echo "[install-playwright] downloading Chromium (~150 MB, one-time)..."
"$venvPy" -m playwright install chromium

# Chromium needs system libs on Linux; install them when we have the privileges.
if "$venvPy" -m playwright install-deps chromium 2>/dev/null; then
  echo "[install-playwright] installed Chromium system deps"
else
  echo "[install-playwright] (could not auto-install Chromium system deps — if Chromium fails to launch, run: sudo \$venvPy -m playwright install-deps chromium)"
fi

echo "[install-playwright] done."
