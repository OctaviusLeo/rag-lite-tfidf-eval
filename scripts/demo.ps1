param(
    [string]$Workspace = "$PSScriptRoot/.."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $Workspace

function RunStep {
    param(
        [string]$Label,
        [string]$Cmd
    )
    Write-Host ''
    Write-Host "### $Label" -ForegroundColor Cyan
    Write-Host $Cmd
    Invoke-Expression $Cmd
}

RunStep -Label "Build index" -Cmd "python src/build_index.py"
RunStep -Label "Query (top-3)" -Cmd "python src/query.py --q 'What is reinforcement learning?' --k 3"
RunStep -Label "Eval (Recall@3)" -Cmd "python src/evaluate.py --k 3"
