param(
    [string]$OllamaExe = "ollama",
    [string]$ProfilePath = "backend/data/kv_cache_optimization.json",
    [string]$Model = "qwen3:4b-instruct",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 11434,
    [ValidateSet("f16", "q8_0", "q4_0")]
    [string]$CacheType
)

$ErrorActionPreference = "Stop"

function Resolve-CacheType {
    param([string]$Path, [string]$ModelName, [string]$Override)
    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        return $Override
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Write-Host "No KV-cache profile found at '$Path'; using f16."
        return "f16"
    }
    try {
        $data = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $key = $ModelName.Trim().ToLowerInvariant()
        $entry = $data.models.$key
        if ($null -eq $entry -or [string]::IsNullOrWhiteSpace([string]$entry.cache_type)) {
            Write-Host "No KV-cache profile for '$ModelName'; using f16."
            return "f16"
        }
        $selected = ([string]$entry.cache_type).ToLowerInvariant()
        if ($selected -notin @("f16", "q8_0", "q4_0")) {
            throw "Unsupported persisted KV cache type '$selected'."
        }
        return $selected
    }
    catch {
        throw "Could not read KV-cache profile '$Path': $($_.Exception.Message)"
    }
}

if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}

$resolvedCache = Resolve-CacheType -Path $ProfilePath -ModelName $Model -Override $CacheType
$command = Get-Command $OllamaExe -ErrorAction SilentlyContinue
if ($null -eq $command) {
    throw "Ollama executable not found: $OllamaExe"
}

$env:OLLAMA_HOST = "${HostAddress}:$Port"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = $resolvedCache

Write-Host "Starting optimized Ollama for SovereignAI"
Write-Host "  Endpoint:       http://${HostAddress}:$Port"
Write-Host "  Model profile:  $Model"
Write-Host "  Flash Attention: enabled"
Write-Host "  KV cache:       $resolvedCache"
Write-Host ""
Write-Host "These variables apply only to this Ollama server process and its children."
Write-Host "If another Ollama server already owns port $Port, stop it first or choose a different -Port."

& $command.Source serve
if ($LASTEXITCODE -ne 0) {
    throw "ollama serve exited with code $LASTEXITCODE"
}
