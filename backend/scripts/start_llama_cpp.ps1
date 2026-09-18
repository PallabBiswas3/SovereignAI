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

function Test-PortAvailable {
    param([string]$Address, [int]$CandidatePort)
    $listener = $null
    try {
        $ip = [System.Net.IPAddress]::Parse($Address)
        $listener = [System.Net.Sockets.TcpListener]::new($ip, $CandidatePort)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $listener) {
            try { $listener.Stop() } catch { }
        }
    }
}

function Test-CompatibleLlamaServer {
    param([string]$Address, [int]$CandidatePort, [string]$ExpectedModel)
    try {
        $base = "http://${Address}:$CandidatePort"
        Invoke-RestMethod -Uri "$base/health" -Method Get -TimeoutSec 2 | Out-Null
        $models = Invoke-RestMethod -Uri "$base/v1/models" -Method Get -TimeoutSec 2
        $ids = @($models.data | ForEach-Object { [string]$_.id })
        return $ids -contains $ExpectedModel
    }
    catch {
        return $false
    }
}

function Get-PortOwnerSummary {
    param([int]$CandidatePort)
    try {
        $connection = Get-NetTCPConnection -LocalPort $CandidatePort -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($null -eq $connection) { return "unknown process" }
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            return "PID $($connection.OwningProcess) ($($process.ProcessName))"
        }
        return "PID $($connection.OwningProcess)"
    }
    catch {
        return "unknown process"
    }
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

$portWasExplicit = $PSBoundParameters.ContainsKey("Port")
if (-not (Test-PortAvailable -Address $HostAddress -CandidatePort $Port)) {
    if (Test-CompatibleLlamaServer -Address $HostAddress -CandidatePort $Port -ExpectedModel $ModelAlias) {
        Write-Host "Compatible llama.cpp server is already running at http://${HostAddress}:$Port"
        Write-Host "Model alias '$ModelAlias' is available there; no second server is needed."
        Write-Host ""
        Write-Host "Run the backend benchmark in another terminal:"
        Write-Host "  python backend/scripts/benchmark_inference_backends.py --model $ModelAlias --output workspace/backend_comparison.json"
        return
    }

    $owner = Get-PortOwnerSummary -CandidatePort $Port
    if ($portWasExplicit) {
        throw "Port $Port is already in use by $owner. Choose another port, e.g. -Port $($Port + 1)."
    }

    $originalPort = $Port
    $replacement = $null
    foreach ($candidate in (($originalPort + 1)..($originalPort + 20))) {
        if (Test-PortAvailable -Address $HostAddress -CandidatePort $candidate) {
            $replacement = $candidate
            break
        }
    }
    if ($null -eq $replacement) {
        throw "Port $originalPort is occupied by $owner and no free port was found in $($originalPort + 1)-$($originalPort + 20)."
    }
    $Port = $replacement
    Write-Host "Port $originalPort is occupied by $owner; using port $Port instead."
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
Write-Host "The backend benchmark auto-discovers this server if the default port was occupied."

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
