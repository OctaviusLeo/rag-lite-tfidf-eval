# Quick Start Script for RAG-Lite
# Run this script to set up and test RAG-Lite

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "RAG-Lite Quick Start" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green

# Install package
Write-Host ""
Write-Host "Installing RAG-Lite..." -ForegroundColor Yellow
pip install -e ".[api]"

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Installation failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Installation complete" -ForegroundColor Green

# Build index
Write-Host ""
Write-Host "Building index from sample data..." -ForegroundColor Yellow
python -m src.cli build --docs data/docs.txt --bm25

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Index build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Index built successfully" -ForegroundColor Green

# Run sample query
Write-Host ""
Write-Host "Running sample query..." -ForegroundColor Yellow
python -m src.cli query "machine learning" --method bm25 --k 3

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Query failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "✓ Quick Start Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  • Start API server: " -NoNewline -ForegroundColor White
Write-Host "python -m src.api" -ForegroundColor Cyan
Write-Host "  • Run benchmarks: " -NoNewline -ForegroundColor White
Write-Host "python -m src.cli benchmark" -ForegroundColor Cyan
Write-Host "  • View docs: " -NoNewline -ForegroundColor White
Write-Host "Visit http://localhost:8000/docs after starting API" -ForegroundColor Cyan
Write-Host ""
