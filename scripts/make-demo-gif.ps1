param(
    [string]$Workspace = "$PSScriptRoot/..",
    [string]$Cast = "demo.cast",
    [string]$Out = "demo.gif",
    [string]$Theme = "dracula",
    [double]$Speed = 1.05,
    [int]$Padding = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $Workspace

if (-not (Test-Path $Cast)) {
    throw "Missing cast: $Cast. Record one with: asciinema rec $Cast -c \"pwsh -File scripts/demo.ps1\""
}

Write-Host "Rendering GIF -> $Out" -ForegroundColor Cyan
npx agg $Cast $Out --theme $Theme --padding $Padding --speed $Speed
Write-Host "Done: $Out" -ForegroundColor Green
