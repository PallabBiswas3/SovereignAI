[CmdletBinding()]
param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$failures = [System.Collections.Generic.List[string]]::new()

function Invoke-ProjectCheck {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$WorkingDirectory,
        [Parameter(Mandatory)] [scriptblock]$Command
    )

    Write-Host "`n== $Name ==" -ForegroundColor Cyan
    Push-Location $WorkingDirectory
    try {
        & $Command
        if ($LASTEXITCODE -ne 0) {
            $failures.Add("$Name (exit $LASTEXITCODE)")
        }
    }
    catch {
        $failures.Add("$Name ($($_.Exception.Message))")
    }
    finally {
        Pop-Location
    }
}

$sovereignPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $sovereignPython)) {
    $sovereignPython = "python"
}

Invoke-ProjectCheck "SovereignAI backend" $workspaceRoot {
    if ($Quick) { & $sovereignPython -m pytest -q tests/test_phase1_foundation.py tests/test_phase38_system_integration.py }
    else { & $sovereignPython -m pytest -q }
}

Invoke-ProjectCheck "SovereignAI frontend" (Join-Path $workspaceRoot "frontend") {
    npm.cmd run typecheck
}

Invoke-ProjectCheck "ControlPlane.ai" (Join-Path $workspaceRoot "Controlplane.ai") {
    if ($Quick) { python -m pytest -q tests/test_controlplane.py }
    else { python -m pytest -q }
}

Invoke-ProjectCheck "Graph-RAG client" (Join-Path $workspaceRoot "Graph-RAG\client") {
    npm.cmd run build
}

Invoke-ProjectCheck "Graph-RAG server" (Join-Path $workspaceRoot "Graph-RAG\server") {
    npm.cmd run build
    if ($LASTEXITCODE -eq 0) { npm.cmd run test:metrics }
    if (-not $Quick -and $LASTEXITCODE -eq 0) { npm.cmd run test:retrieval-core }
    if (-not $Quick -and $LASTEXITCODE -eq 0) { npm.cmd run test:agent-policy }
    if (-not $Quick -and $LASTEXITCODE -eq 0) { npm.cmd run test:verification-policy }
}

Invoke-ProjectCheck "Time-Series Diagnostic Agent" (Join-Path $workspaceRoot "Time-Series-Diagnostic-Agent") {
    if ($Quick) { python -m pytest -q tests/test_public_pipeline.py tests/test_diagnostic_tool.py tests/test_api.py }
    else { python -m pytest -q }
}

if ($failures.Count -gt 0) {
    Write-Host "`nWorkspace verification failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "`nAll workspace checks passed." -ForegroundColor Green
