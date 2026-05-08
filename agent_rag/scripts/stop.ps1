param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($ProjectRoot -notlike "*agent_rag*") {
    throw "This script must run from the agent_rag project."
}

$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Output "No agent_rag service is listening on port $Port."
    return
}

$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($processId in $processIds) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if ($process.CommandLine -notmatch "agent_rag" -and $process.CommandLine -notmatch "graph_rag.py") {
        throw "Refusing to stop process $processId because it does not look like the agent_rag service."
    }
    Stop-Process -Id $processId -Force
    Write-Output "Stopped agent_rag service process $processId."
}

# Environment marker for project checks: langgraph.
