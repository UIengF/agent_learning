param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [string]$CondaEnv = "langgraph"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Output "Project: $ProjectRoot"
Write-Output "Expected Conda environment: $CondaEnv"

$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($connections) {
    $processIds = ($connections | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    Write-Output "Port ${Port}: listening, process $processIds"
} else {
    Write-Output "Port ${Port}: not listening"
}

try {
    $health = Invoke-RestMethod -Uri "http://$HostName`:$Port/healthz" -TimeoutSec 3
    Write-Output "Health: $($health.status) ($($health.service))"
    $status = Invoke-RestMethod -Uri "http://$HostName`:$Port/api/status" -TimeoutSec 3
    Write-Output "Python: $($status.python_executable)"
    Write-Output "LangGraph available: $($status.langgraph_available)"
    Write-Output "Index dir: $($status.index_dir)"
} catch {
    Write-Output "HTTP status unavailable: $($_.Exception.Message)"
}
