[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$stateRoot = Join-Path $workspaceRoot ".workspace"
$logRoot = Join-Path $stateRoot "logs"
$pidFile = Join-Path $stateRoot "services.json"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

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

$services = @(
    @{ name = "controlplane"; port = 8100; url = "http://127.0.0.1:8100/health" },
    @{ name = "diagnostics"; port = 8200; url = "http://127.0.0.1:8200/health" },
    @{ name = "graph-rag"; port = 3100; url = "http://127.0.0.1:3100/health" },
    @{ name = "sovereign-api"; port = 8000; url = "http://127.0.0.1:8000/health" },
    @{ name = "sovereign-ui"; port = 3000; url = "http://127.0.0.1:3000" }
)

if (Test-Path -LiteralPath $pidFile) {
    try {
        $existing = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
        $running = @($existing | Where-Object { Get-Process -Id $_.pid -ErrorAction SilentlyContinue })
        if ($running.Count -eq @($existing).Count -and $running.Count -gt 0) {
            Write-Host "The integrated workspace is already running."
            foreach ($service in $existing) {
                Write-Host ("{0,-16} {1}" -f $service.name, $service.url)
            }
            exit 0
        }
        if ($running.Count -gt 0) {
            throw "The workspace is partially running. Run .\scripts\Stop-Workspace.ps1, then start it again."
        }
        Remove-Item -LiteralPath $pidFile -Force
        Write-Host "Removed a stale workspace process manifest."
    }
    catch {
        if ($_.Exception.Message -like "The workspace is partially running*") { throw }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
        Write-Host "Removed an unreadable workspace process manifest."
    }
}

foreach ($service in $services) {
    $occupant = Get-ListeningProcessId -Port $service.port
    if ($occupant) {
        throw "Port $($service.port) for $($service.name) is already owned by PID $occupant. Run .\scripts\Stop-Workspace.ps1 first."
    }
}

$processes = @()
try {
    foreach ($service in $services) {
        $stdout = Join-Path $logRoot "$($service.name).out.log"
        $stderr = Join-Path $logRoot "$($service.name).err.log"
        $process = Start-Process powershell.exe -WindowStyle Hidden -PassThru `
            -WorkingDirectory $workspaceRoot `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "Run-Service.ps1"), "-Service", $service.name) `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $processes += [pscustomobject]@{
            name = $service.name; pid = $process.Id; launcher_pid = $process.Id
            port = $service.port; url = $service.url
        }
    }
    $processes | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8

    $deadline = (Get-Date).AddSeconds(45)
    $pending = @($processes)
    while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 750
        $pending = @($pending | Where-Object {
            try {
                $response = Invoke-WebRequest -Uri $_.url -UseBasicParsing -TimeoutSec 2
                $response.StatusCode -ne 200
            }
            catch { $true }
        })
    }

    foreach ($service in $processes) {
        $ready = -not ($pending.name -contains $service.name)
        if ($ready) {
            $listenerPid = Get-ListeningProcessId -Port $service.port
            if ($listenerPid) { $service.pid = $listenerPid }
        }
        $state = if ($ready) { "ready" } else { "not ready; inspect .workspace/logs" }
        Write-Host ("{0,-16} {1,-31} {2}" -f $service.name, $service.url, $state)
    }
    $processes | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8
    if ($pending.Count -gt 0) { exit 1 }
}
catch {
    foreach ($process in $processes) {
        Stop-ProcessTree -ProcessId $process.pid
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    throw
}
