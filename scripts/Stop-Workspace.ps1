[CmdletBinding()]
param()

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $workspaceRoot ".workspace\services.json"

function Stop-ProcessTree {
    param([int]$ProcessId)
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) { Stop-ProcessTree -ProcessId $child.ProcessId }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Get-ListeningProcessId {
    param([int]$Port)
    $pattern = "^\s*TCP\s+(?:127\.0\.0\.1|\[::1\]):$Port\s+\S+\s+LISTENING\s+(\d+)\s*$"
    foreach ($line in (& netstat.exe -ano -p tcp)) {
        if ($line -match $pattern) { return [int]$Matches[1] }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No workspace process manifest was found."
    exit 0
}

$services = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
foreach ($service in $services) {
    $port = if ($service.port) { [int]$service.port } else {
        switch ($service.name) {
            "sovereign-ui" { 3000 }
            "graph-rag" { 3100 }
            "sovereign-api" { 8000 }
            "controlplane" { 8100 }
            "diagnostics" { 8200 }
        }
    }
    $listenerPid = Get-ListeningProcessId -Port $port
    if ($listenerPid) { Stop-ProcessTree -ProcessId $listenerPid }
    if ($service.launcher_pid -and $service.launcher_pid -ne $listenerPid) {
        Stop-ProcessTree -ProcessId $service.launcher_pid
    }
    elseif ($service.pid -and $service.pid -ne $listenerPid) {
        Stop-ProcessTree -ProcessId $service.pid
    }
    Write-Host "Stopped $($service.name) (port $port)."
}
Remove-Item -LiteralPath $pidFile -Force
