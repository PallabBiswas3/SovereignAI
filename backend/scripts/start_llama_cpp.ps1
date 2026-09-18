param(
    [Parameter(Mandatory = $true)]
    [string]$LlamaServerPath,

    [string]$ModelPath,
    [string]$OllamaModelTag = "qwen3:4b-instruct",
    [string]$OllamaModelsRoot,
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

function Resolve-OllamaGgufBlob {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModelTag,
        [string]$ModelsRoot
    )

    if ([string]::IsNullOrWhiteSpace($ModelsRoot)) {
        if (-not [string]::IsNullOrWhiteSpace($env:OLLAMA_MODELS)) {
            $ModelsRoot = $env:OLLAMA_MODELS
        }
        else {
            $ModelsRoot = Join-Path $HOME ".ollama\models"
        }
    }

    $name = $ModelTag
    $tag = "latest"
    $colon = $ModelTag.LastIndexOf(":")
    if ($colon -gt $ModelTag.LastIndexOf("/")) {
        $name = $ModelTag.Substring(0, $colon)
        $tag = $ModelTag.Substring($colon + 1)
    }

    $segments = @($name -split "/" | Where-Object { $_ })
    if ($segments.Count -eq 0) {
        throw "Invalid Ollama model tag: $ModelTag"
    }

    $registry = "registry.ollama.ai"
    if ($segments[0] -match "\.") {
        $registry = $segments[0]
        $segments = @($segments | Select-Object -Skip 1)
    }
    elseif ($segments.Count -eq 1) {
        $segments = @("library", $segments[0])
    }

    $manifestPath = Join-Path $ModelsRoot "manifests"
    $manifestPath = Join-Path $manifestPath $registry
    foreach ($segment in $segments) {
        $manifestPath = Join-Path $manifestPath $segment
    }
    $manifestPath = Join-Path $manifestPath $tag

    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Ollama manifest not found for '$ModelTag': $manifestPath"
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $modelLayers = @(
        $manifest.layers | Where-Object {
            $_.mediaType -eq "application/vnd.ollama.image.model"
        }
    )
    if ($modelLayers.Count -ne 1) {
        throw "Expected one GGUF model layer for '$ModelTag', found $($modelLayers.Count). Split/multi-layer models require an explicit -ModelPath."
    }

    $digest = [string]$modelLayers[0].digest
    if ($digest -notmatch "^sha256:[0-9a-fA-F]{64}$") {
        throw "Unexpected Ollama model-layer digest: $digest"
    }
    $blobName = $digest.Replace(":", "-")
    $blobPath = Join-Path (Join-Path $ModelsRoot "blobs") $blobName
    if (-not (Test-Path -LiteralPath $blobPath -PathType Leaf)) {
        throw "Ollama model blob not found: $blobPath"
    }

    $stream = [System.IO.File]::OpenRead($blobPath)
    try {
        $signatureBytes = New-Object byte[] 4
        if ($stream.Read($signatureBytes, 0, 4) -ne 4) {
            throw "Ollama model blob is too small to be a GGUF file: $blobPath"
        }
        $signature = [System.Text.Encoding]::ASCII.GetString($signatureBytes)
        if ($signature -ne "GGUF" -and $signature -ne "FUGG") {
            throw "Ollama model layer is not a GGUF blob (signature '$signature'): $blobPath"
        }
    }
    finally {
        $stream.Dispose()
    }

    Write-Host "Resolved Ollama model '$ModelTag' to its local GGUF blob."
    Write-Host "  Ollama root: $ModelsRoot"
    Write-Host "  Manifest:    $manifestPath"
    return $blobPath
}

if (-not (Test-Path -LiteralPath $LlamaServerPath -PathType Leaf)) {
    throw "llama-server executable not found: $LlamaServerPath"
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Resolve-OllamaGgufBlob -ModelTag $OllamaModelTag -ModelsRoot $OllamaModelsRoot
}
elseif (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
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
