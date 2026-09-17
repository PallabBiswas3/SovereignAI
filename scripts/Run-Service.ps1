[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("sovereign-api", "sovereign-ui", "graph-rag", "controlplane", "diagnostics")]
    [string]$Service
)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot

switch ($Service) {
    "sovereign-api" {
        Set-Location (Join-Path $workspaceRoot "backend")
        & (Join-Path $workspaceRoot ".venv\Scripts\python.exe") -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    }
    "sovereign-ui" {
        Set-Location (Join-Path $workspaceRoot "frontend")
        npm.cmd run dev -- --hostname 127.0.0.1 --port 3000
    }
    "graph-rag" {
        Set-Location (Join-Path $workspaceRoot "Graph-RAG\server")
        $env:PORT = "3100"
        npm.cmd run dev
    }
    "controlplane" {
        Set-Location (Join-Path $workspaceRoot "Controlplane.ai")
        python -m uvicorn --app-dir src controlplane.api:app --host 127.0.0.1 --port 8100
    }
    "diagnostics" {
        Set-Location (Join-Path $workspaceRoot "Time-Series-Diagnostic-Agent")
        python -m uvicorn --app-dir src tsdiag.api:app --host 127.0.0.1 --port 8200
    }
}

exit $LASTEXITCODE
