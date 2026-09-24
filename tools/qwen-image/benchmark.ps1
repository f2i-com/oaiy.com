param(
    [string]$Config = (Join-Path $PSScriptRoot '../../config/qwen-image.local.json'),
    [ValidateSet('gguf', 'safetensors')][string]$Weights = 'gguf',
    [ValidateRange(256, 2048)][int]$Size = 1024,
    [ValidateRange(2, 1000)][int]$Count = 4,
    [string]$Prompt = 'A cheerful orange robot artist painting a colorful landscape in a sunlit studio, detailed illustration'
)
$ErrorActionPreference = 'Stop'
if ($Size % 32) { throw 'Size must be a multiple of 32.' }
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if (-not $cfg.adapter) { throw 'The six-step benchmark requires a turbo adapter.' }
$run = Join-Path $repo ('target/qwen-image-bench/' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run -Force | Out-Null
$requestPath = Join-Path $run 'request.json'
$request = @{
    base = $cfg.base
    transformer = $(if ($Weights -eq 'gguf') { $cfg.transformer } else { Join-Path $cfg.base 'transformer' })
    adapter = $cfg.adapter
    output_dir = (Join-Path $run 'images')
    prompt = $Prompt
    n = $Count; width = $Size; height = $Size; steps = 6; cfg = 1; seed = 90
    device = $cfg.image_device
}
[IO.File]::WriteAllText($requestPath, ($request | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$oldCache = $env:CUDA_CACHE_PATH
try {
    if (-not $oldCache) {
        $env:CUDA_CACHE_PATH = Join-Path $cfg.output_root '.cuda-cache'
        New-Item -ItemType Directory -Path $env:CUDA_CACHE_PATH -Force | Out-Null
    }
    $process = Start-Process -FilePath $cfg.worker -ArgumentList '--request', ('"' + $requestPath + '"') -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $run 'result.json') -RedirectStandardError (Join-Path $run 'events.jsonl')
    if ($process.ExitCode) { throw (Get-Content (Join-Path $run 'events.jsonl') -Tail 8 | Out-String) }
} finally {
    $env:CUDA_CACHE_PATH = $oldCache
}
$result = Get-Content (Join-Path $run 'result.json') -Raw | ConvertFrom-Json
$records = @(Get-Content (Join-Path $result.output_dir 'manifest.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$warm = @($records | Select-Object -Skip 1 | ForEach-Object { $_.image_seconds } | Sort-Object)
$median = ($warm[[int][math]::Floor(($warm.Count - 1) / 2)] + $warm[[int][math]::Floor($warm.Count / 2)]) / 2
[pscustomobject]@{
    weights = $Weights; size = $Size; steps = 6; images = $Count
    total_seconds = $result.seconds
    text_load_seconds = $result.text_load_seconds
    encoding_seconds = $result.encoding_seconds
    transformer_load_seconds = $result.transformer_load_seconds
    vae_load_seconds = $result.vae_load_seconds
    first_image_seconds = $records[0].image_seconds
    warm_median_seconds = $median
    warm_min_seconds = $warm[0]
    warm_max_seconds = $warm[-1]
    output_dir = $result.output_dir
} | ConvertTo-Json
