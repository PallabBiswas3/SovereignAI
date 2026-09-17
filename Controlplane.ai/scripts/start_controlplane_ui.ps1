param(
    [string]$Model = "cross-encoder/nli-deberta-v3-small",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Project virtual environment was not found at '$pythonExe'. Create .venv and install the project dependencies first."
}

$env:CONTROLPLANE_ADAPTIVE_VERIFICATION = "1"
$env:CONTROLPLANE_NLI_MODEL = $Model
if ($Device -eq "auto") {
    Remove-Item Env:CONTROLPLANE_NLI_DEVICE -ErrorAction SilentlyContinue
} else {
    $env:CONTROLPLANE_NLI_DEVICE = $Device
}

Write-Output "ControlPlane AdaptiveFact/NLI: enabled"
Write-Output "NLI model: $Model"
Write-Output "NLI device: $Device"
Write-Output "Starting Streamlit at http://localhost:8501"

Push-Location $projectRoot
try {
    & $pythonExe -m streamlit run "demo\app.py" --server.fileWatcherType none
    $streamlitExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $streamlitExitCode
