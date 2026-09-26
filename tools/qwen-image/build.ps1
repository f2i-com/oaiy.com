$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue) -or $env:VSCMD_VER -notlike '17.*') {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    # CUDA 12.8's host-code compiler supports VS 2022, not VS 2026.
    $installation = & $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installation) { throw 'Install Visual Studio 2022 C++ build tools to compile the CUDA kernels.' }
    Import-Module (Join-Path $installation 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
}
Push-Location $repo
try {
    cargo build --release -p nrob-diffusion --features flash-attn
    if ($LASTEXITCODE) { throw 'Native diffusion build failed.' }
    cargo build --release -p nrob-server
    if ($LASTEXITCODE) { throw 'NROB server build failed.' }
    # The same server without CUDA, for machines that lack it (WebGPU, else CPU).
    cargo build --release -p nrob-server --no-default-features --features webgpu --bin nrob-server-webgpu
    if ($LASTEXITCODE) { throw 'NROB WebGPU server build failed.' }
    cargo build --release -p nrob-studio -p nrob-studio-tray
    if ($LASTEXITCODE) { throw 'NROB Studio build failed.' }
} finally { Pop-Location }
