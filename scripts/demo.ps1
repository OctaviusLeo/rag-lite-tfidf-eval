param(
    [string]$Workspace = "$PSScriptRoot/.."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $Workspace

function Run-Step {
    param([string]$Label, [string]$Cmd)
    Write-Host ''
    Write-Host "### $Label" -ForegroundColor Cyan
    Write-Host "```bash"; Write-Host "$Cmd"; Write-Host "```"
    Invoke-Expression $Cmd
}

Run-Step -Label "Build index" -Cmd "python src/build_index.py"
Run-Step -Label "Query (top-3)" -Cmd "python src/query.py --q 'What is reinforcement learning?' --k 3"
Run-Step -Label "Eval (Recall@3)" -Cmd "python src/evaluate.py --k 3"
