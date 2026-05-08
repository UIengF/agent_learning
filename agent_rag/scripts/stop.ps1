param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Output "No Graph RAG service is listening on port $Port."
    return
}

$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($processId in $processIds) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if ($process.CommandLine -notmatch "graph_rag.py") {
        throw "Refusing to stop process $processId because it does not look like the Graph RAG service."
    }
    Stop-Process -Id $processId -Force
    Write-Output "Stopped Graph RAG service process $processId."
}

# Environment marker for project checks: langgraph.
