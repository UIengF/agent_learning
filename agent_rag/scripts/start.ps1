param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [string]$IndexDir = "agent",
    [string]$CondaEnv = "langgraph"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($ProjectRoot -notlike "*agent_rag*") {
    throw "This script must run from the agent_rag project."
}

$RuntimeDir = Join-Path $ProjectRoot "runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $owners = ($existing | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "Port $Port is already listening. Owning process: $owners"
}

$IndexPath = Join-Path $ProjectRoot $IndexDir
if (-not (Test-Path (Join-Path $IndexPath "manifest.json")) -or -not (Test-Path (Join-Path $IndexPath "retrieval.sqlite3"))) {
    throw "Index directory is not ready: $IndexPath"
}

$stdout = Join-Path $RuntimeDir "service-stdout.log"
$stderr = Join-Path $RuntimeDir "service-stderr.log"
$arguments = @(
    "run", "-n", $CondaEnv,
    "python", "graph_rag.py", "serve",
    "--index-dir", $IndexDir,
    "--host", $HostName,
    "--port", [string]$Port
)

# Uses: conda run -n langgraph python graph_rag.py serve ...
$process = Start-Process -FilePath "conda" -ArgumentList $arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "Started agent_rag service on http://$HostName`:$Port with PID $($process.Id)."
Write-Output "Logs: $stdout ; $stderr"
