param([string]$Destination = 'E:/models/sdxl/clip-tokenizer')
$ErrorActionPreference = 'Stop'
$revision = '32bd64288804d66eefd0ccbe215aa642df71cc41'
$url = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/$revision/tokenizer.json"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$target = Join-Path $Destination 'tokenizer.json'
$temporary = Join-Path $Destination 'tokenizer.download.json'
try {
    Invoke-WebRequest -Uri $url -OutFile $temporary
    $sha256 = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sha256 -ne 'a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4') {
        throw 'Downloaded CLIP tokenizer does not match the pinned SHA-256.'
    }
    Move-Item -LiteralPath $temporary -Destination $target -Force
    $source = @{ repo='openai/clip-vit-large-patch14'; revision=$revision; sha256=$sha256 }
    [IO.File]::WriteAllText((Join-Path ([IO.Path]::GetFullPath($Destination)) 'source.json'), ($source | ConvertTo-Json))
    Write-Host "CLIP tokenizer ready: $target"
} finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
}
