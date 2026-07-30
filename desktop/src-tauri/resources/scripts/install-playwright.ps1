# install-playwright.ps1
#
# Sets up the Playwright browser backend for the companion's
# "Playwright Browser" service: a Python venv with `playwright` + the
# Chromium browser binary. Reuses the companion's bundled Python runtime
# when present, otherwise falls back to a system Python 3 on PATH.
#
# Env supplied by the companion: OAIY_BIN_DIR (= <dataDir>/bin),
# OAIY_MODELS_DIR. We derive <dataDir> from OAIY_BIN_DIR.
#
# One-time cost: the Chromium download is ~150 MB. Subsequent installs
# (re-run to upgrade) are quick -- pip + playwright cache aggressively.

$ErrorActionPreference = 'Stop'

if (-not $env:OAIY_BIN_DIR) {
    Write-Error 'OAIY_BIN_DIR is not set; this script expects to be invoked by the OAIY app.'
    exit 1
}

$dataDir = Split-Path $env:OAIY_BIN_DIR -Parent
$venv = Join-Path $dataDir 'venvs\playwright'
$venvPy = Join-Path $venv 'Scripts\python.exe'

# Pick a base Python: companion runtime first (self-contained), then a
# system python / py launcher.
$companionPy = Join-Path $dataDir 'python\python.exe'
$basePy = $null
if (Test-Path $companionPy) {
    $basePy = $companionPy
    Write-Host "[install-playwright] using companion Python: $basePy"
} else {
    foreach ($cand in @('python', 'py')) {
        try {
            $null = & $cand --version 2>&1
            if ($LASTEXITCODE -eq 0) { $basePy = $cand; break }
        } catch {}
    }
}
if (-not $basePy) {
    Write-Error 'No Python found. Install Python in the companion''s Python tab (one click), or install Python 3 system-wide, then re-run Install.'
    exit 1
}

if (-not (Test-Path $venvPy)) {
    Write-Host "[install-playwright] creating venv at $venv"
    & $basePy -m venv $venv
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed (exit $LASTEXITCODE)"; exit 2 }
}

Write-Host "[install-playwright] installing playwright (pip)"
& $venvPy -m pip install --upgrade pip playwright
if ($LASTEXITCODE -ne 0) { Write-Error "pip install playwright failed (exit $LASTEXITCODE)"; exit 3 }

Write-Host "[install-playwright] downloading Chromium (~150 MB, one-time)..."
& $venvPy -m playwright install chromium
if ($LASTEXITCODE -ne 0) { Write-Error "playwright install chromium failed (exit $LASTEXITCODE)"; exit 4 }

Write-Host "[install-playwright] done. Start the 'Playwright Browser' service, then browser_* nodes in oaiy-web will use it."
