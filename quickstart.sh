#!/bin/bash
# Quick Start Script for RAG-Lite
# Run this script to set up and test RAG-Lite

set -e

echo "=============================================================="
echo "RAG-Lite Quick Start"
echo "=============================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Install package
echo "Installing RAG-Lite..."
pip install -e ".[api]"
echo "✓ Installation complete"
echo ""

# Build index
echo "Building index from sample data..."
python3 -m src.cli build --docs data/docs.txt --bm25
echo "✓ Index built successfully"
echo ""

# Run sample query
echo "Running sample query..."
python3 -m src.cli query "machine learning" --method bm25 --k 3
echo ""

echo "=============================================================="
echo "✓ Quick Start Complete!"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "  • Start API server: python3 -m src.api"
echo "  • Run benchmarks: python3 -m src.cli benchmark"
echo "  • View docs: Visit http://localhost:8000/docs after starting API"
echo ""
