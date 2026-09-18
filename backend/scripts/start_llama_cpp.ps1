param(
    [Parameter(Mandatory = $true)]
    [string]$LlamaServerPath,

    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$ModelAlias = "qwen3:4b-instruct",
    [int]$Threads = 5,
    [int]$BatchSize = 128,
    [int]$ContextSize = 4096,
    [int]$Port = 8080,
    [string]$HostAddress = "127.0.0.1",
    [ValidateSet("f16", "bf16", "q8_0", "q4_0")]
    [string]$CacheTypeK = "f16",
    [ValidateSet("f16", "bf16", "q8_0", "q4_0")]
    [string]$CacheTypeV = "f16"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LlamaServerPath -PathType Leaf)) {
    throw "llama-server executable not found: $LlamaServerPath"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "GGUF model not found: $ModelPath"
}
if ($Threads -lt 1 -or $BatchSize -lt 1 -or $ContextSize -lt 512) {
    throw "Threads and batch size must be positive; context size must be at least 512."
}

Write-Host "Starting llama.cpp for SovereignAI"
Write-Host "  Model:      $ModelPath"
Write-Host "  API alias:  $ModelAlias"
Write-Host "  Endpoint:   http://${HostAddress}:$Port"
Write-Host "  Threads:    $Threads"
Write-Host "  Batch:      $BatchSize"
Write-Host "  Context:    $ContextSize"
Write-Host "  KV cache:   K=$CacheTypeK V=$CacheTypeV"
Write-Host "  Parallel:   1"
Write-Host ""
Write-Host "Note: --mlock is intentionally not used on the measured 16 GB CPU-laptop profile."

& $LlamaServerPath `
    --model $ModelPath `
    --alias $ModelAlias `
    --host $HostAddress `
    --port $Port `
    --threads $Threads `
    --threads-batch $Threads `
    --ctx-size $ContextSize `
    --batch-size $BatchSize `
    --ubatch-size $BatchSize `
    --parallel 1 `
    --cache-type-k $CacheTypeK `
    --cache-type-v $CacheTypeV

if ($LASTEXITCODE -ne 0) {
    throw "llama-server exited with code $LASTEXITCODE"
}
