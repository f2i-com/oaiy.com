$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
# The engines' GPU is WebGPU's (a default feature of each): nothing of a GPU's is needed to build them, and cargo
# finds the C++ linker itself.
Push-Location $repo
try {
    cargo build --release -p oaiy-media
    if ($LASTEXITCODE) { throw 'Native diffusion build failed.' }
    # Both of its programs: oaiy-llm-server, and the same server as oaiy-llm-server-webgpu (the name the desktop's
    # installer stages it by).
    cargo build --release -p oaiy-llm-server
    if ($LASTEXITCODE) { throw 'OAIY server build failed.' }
    cargo build --release -p oaiy-studio -p oaiy-studio-tray
    if ($LASTEXITCODE) { throw 'OAIY build failed.' }
} finally { Pop-Location }
