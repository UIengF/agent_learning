[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$KbPath,

    [Parameter()]
    [string]$Question = "What changed recently about OpenAI agents?",

    [Parameter()]
    [string]$IndexDir = ".\agent",

    [Parameter()]
    [string]$CheckpointDb = ".\runtime\checkpoints.db",

    [Parameter()]
    [string]$SessionId = ("run-" + (Get-Date -Format "yyyyMMdd-HHmmss")),

    [switch]$RebuildIndex,
    [switch]$InspectIndex,
    [switch]$RunTests,
    [switch]$DisableWeb
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    Write-Host ""
    Write-Host ">> $($Command -join ' ')" -ForegroundColor Cyan
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source
$resolvedKbPath = Resolve-ProjectPath -PathValue $KbPath -ProjectRoot $projectRoot
$resolvedIndexDir = Resolve-ProjectPath -PathValue $IndexDir -ProjectRoot $projectRoot
$resolvedCheckpointDb = Resolve-ProjectPath -PathValue $CheckpointDb -ProjectRoot $projectRoot
$indexDbPath = Join-Path $resolvedIndexDir "retrieval.sqlite3"
$manifestPath = Join-Path $resolvedIndexDir "manifest.json"

if (-not (Test-Path $resolvedKbPath)) {
    throw "Knowledge base path does not exist: $resolvedKbPath"
}

if (-not $env:DASHSCOPE_API_KEY) {
    throw "Missing DASHSCOPE_API_KEY in the current shell environment."
}

New-Item -ItemType Directory -Force -Path $resolvedIndexDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedCheckpointDb) | Out-Null

Push-Location $projectRoot
try {
    if ($DisableWeb) {
        $env:RAG_WEB_ENABLED = "false"
    }
    else {
        $env:RAG_WEB_ENABLED = "true"
    }

    if ($RunTests) {
        Invoke-Checked -Command @($python, "-m", "unittest", "discover", "-s", "tests", "-v")
    }

    $needsBuild = $RebuildIndex -or -not (Test-Path $indexDbPath) -or -not (Test-Path $manifestPath)
    if ($needsBuild) {
        Invoke-Checked -Command @(
            $python,
            ".\graph_rag.py",
            "index",
            "build",
            "--kb-path",
            $resolvedKbPath,
            "--output-dir",
            $resolvedIndexDir
        )
    }
    else {
        Write-Host ">> Reusing existing index: $resolvedIndexDir" -ForegroundColor Yellow
    }

    if ($InspectIndex) {
        Invoke-Checked -Command @(
            $python,
            ".\graph_rag.py",
            "index",
            "inspect",
            "--index-dir",
            $resolvedIndexDir
        )
    }

    Invoke-Checked -Command @(
        $python,
        ".\graph_rag.py",
        "ask",
        "--index-dir",
        $resolvedIndexDir,
        "--question",
        $Question,
        "--session-id",
        $SessionId,
        "--checkpoint-db",
        $resolvedCheckpointDb
    )

    Write-Host ""
    Write-Host "Run completed." -ForegroundColor Green
    Write-Host "Session ID: $SessionId"
    Write-Host "Index Dir:  $resolvedIndexDir"
    Write-Host "Checkpoint: $resolvedCheckpointDb"
}
finally {
    Pop-Location
}
