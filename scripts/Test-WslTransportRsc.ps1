[CmdletBinding()]
param()

# A diagnostic only: preserve TLS/certificate verification, change only the WSL
# virtual adapter's coalescing flags, and restore the original flags in finally.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$evidence = Join-Path $root 'local_testing\deployment-remediation-20261023'
$adapterName = 'vEthernet (WSL (Hyper-V firewall))'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'This reversible WSL-adapter diagnostic requires an Administrator terminal.' }
if (@(& docker ps -q).Count -gt 0) { throw 'Running Docker workloads found; diagnostic refused to interrupt them.' }
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
Start-Transcript -Path (Join-Path $evidence 'rsc-admin.log') -NoClobber
try {
    $before = Get-NetAdapterRsc -Name $adapterName
    $before | Select-Object Name,IPv4Enabled,IPv6Enabled | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence 'rsc-before.json')
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $runner = Join-Path $root 'scripts\deployment_evidence.py'
    $probe = Join-Path $root 'tests\deployment\transport_probe.cjs'
    function Invoke-TransportProbe([string]$label) {
        & $python $runner --label $label --timeout 180 -- docker run --rm --mount "type=bind,source=$probe,target=/probe.cjs,readonly" node:22-alpine node /probe.cjs
        $script:probeExit = $LASTEXITCODE
    }
    Invoke-TransportProbe 'rsc-before-probe'
    $baseline = $script:probeExit
    $changed = $false
    try {
        $changed = $true
        Disable-NetAdapterRsc -Name $adapterName -IPv4 -IPv6 -Confirm:$false
        Get-NetAdapterRsc -Name $adapterName | Select-Object Name,IPv4Enabled,IPv6Enabled | ConvertTo-Json
        Start-Sleep -Seconds 3
        Invoke-TransportProbe 'rsc-disabled-probe'
        $disabled = $script:probeExit
    }
    finally {
        if ($changed) {
            Set-NetAdapterRsc -Name $adapterName -IPv4Enabled $before.IPv4Enabled -IPv6Enabled $before.IPv6Enabled -Confirm:$false
        }
    }
    Start-Sleep -Seconds 3
    Invoke-TransportProbe 'rsc-restored-probe'
    $restored = $script:probeExit
    $after = Get-NetAdapterRsc -Name $adapterName
    [pscustomobject]@{
        baselineExit = $baseline; disabledExit = $disabled; restoredExit = $restored
        originalFlagsRestored = ($after.IPv4Enabled -eq $before.IPv4Enabled -and $after.IPv6Enabled -eq $before.IPv6Enabled)
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence 'rsc-result.json')
}
finally { Stop-Transcript }
