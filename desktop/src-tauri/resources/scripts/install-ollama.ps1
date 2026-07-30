# install-ollama.ps1
#
# Installs Ollama via its official Windows installer. Ollama puts itself
# on PATH globally, so the template's `command: "ollama"` works without
# any additional wiring (no OAIY_BIN_DIR step).
#
# If Ollama is already installed, this script is a no-op.

$ErrorActionPreference = 'Stop'
# Invoke-WebRequest renders a progress bar that is very slow and, in a piped
# (no-console) context like the companion's install runner, can stall the
# download outright. Silencing it makes the download fast + non-blocking.
$ProgressPreference = 'SilentlyContinue'

# Already installed? Don't re-download.
$existing = $null
try { $existing = (Get-Command ollama -ErrorAction Stop).Source } catch {}
if ($existing) {
    Write-Host "[install-ollama] Ollama already installed at $existing"
    exit 0
}

$installerUrl = 'https://ollama.com/download/OllamaSetup.exe'
$installerPath = Join-Path $env:TEMP 'OllamaSetup.exe'

Write-Host "[install-ollama] downloading installer"
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing

Write-Host "[install-ollama] running installer (silent)"
# The official installer supports /VERYSILENT for unattended install.
# NOTE: -PassThru + .WaitForExit() (NOT -Wait). PowerShell's -Wait blocks until
# the whole process *tree* exits, and the installer auto-launches a persistent
# "ollama app" child — so -Wait would hang forever. .WaitForExit() waits only
# for the installer process itself.
$proc = Start-Process -FilePath $installerPath -ArgumentList '/VERYSILENT', '/NORESTART' -PassThru
$proc.WaitForExit()
Remove-Item $installerPath -Force -ErrorAction SilentlyContinue

if ($proc.ExitCode -ne 0) {
    Write-Error "Ollama installer exited with code $($proc.ExitCode)"
    exit $proc.ExitCode
}

# The installer auto-launches the Ollama tray app, which runs its own
# `ollama serve` on the default port. The companion manages the service itself
# (its template runs `ollama serve`), so stop the auto-started app + server
# now — otherwise it holds the port (clashing with the companion's serve) and
# the lingering process can keep this install job from completing.
Start-Sleep -Seconds 2
foreach ($n in 'ollama app', 'ollama') {
    Get-Process -Name $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

# Refresh PATH for the current session so a subsequent `ollama serve`
# in this same session can find the binary.
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User')

try {
    $resolved = (Get-Command ollama -ErrorAction Stop).Source
    Write-Host "[install-ollama] installed at $resolved"
} catch {
    Write-Warning '[install-ollama] installed, but ollama is not yet on PATH for the current session. Restart the OAIY app if needed.'
}

# Pre-pull the tiny default model the companion's Ollama node targets
# (qwen2.5:0.5b, ~0.4 GB), so a freshly-installed Ollama node RUNS without a
# hidden manual pull — Ollama's OpenAI /v1/chat/completions returns 404 for an
# un-pulled model rather than pulling on demand. Best-effort: never fail the
# install on a pull error. `ollama pull` starts a transient server, so stop any
# lingering one afterwards to free the port for the companion's own serve.
try {
    Write-Host '[install-ollama] pre-pulling default model qwen2.5:0.5b'
    & ollama pull qwen2.5:0.5b
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[install-ollama] 'ollama pull qwen2.5:0.5b' exited $LASTEXITCODE; pull it manually if the Ollama node 404s."
    }
} catch {
    Write-Warning "[install-ollama] could not pre-pull qwen2.5:0.5b ($_). Run 'ollama pull qwen2.5:0.5b' manually."
}
Start-Sleep -Seconds 1
foreach ($n in 'ollama app', 'ollama') {
    Get-Process -Name $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
