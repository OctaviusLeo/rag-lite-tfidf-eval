param(
    [string]$Workspace = "$PSScriptRoot/..",
    [string]$Python = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $Python) {
    $venv = Join-Path $Workspace ".venv/Scripts/python.exe"
    if (Test-Path $venv) {
        $Python = $venv
    }
    else {
        $Python = "python"
    }
}

Set-Location $Workspace

function Run-Step {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$Args
    )
    Write-Host ''
    Write-Host ("### " + $Label) -ForegroundColor Cyan
    Write-Host '---'
    Write-Host ($Exe + ' ' + ($Args -join ' '))
    Write-Host '---'
    & $Exe @Args
}

Run-Step -Label 'Build index' -Exe $Python -Args @('src/build_index.py')
Run-Step -Label 'Query (top-3)' -Exe $Python -Args @('src/query.py','--q','What is reinforcement learning?','--k','3')
Run-Step -Label 'Eval (Recall@3)' -Exe $Python -Args @('src/evaluate.py','--k','3')
