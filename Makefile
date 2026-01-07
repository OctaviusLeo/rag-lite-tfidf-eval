# Makefile for RAG-Lite

.PHONY: help install install-dev install-all test lint format clean docker-build docker-up docker-down benchmark

help:
	@echo "RAG-Lite Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make install-all    Install with all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Running:"
	@echo "  make api           Start API server"
	@echo "  make benchmark     Run benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start Docker services"
	@echo "  make docker-down   Stop Docker services"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

api:
	python -m uvicorn src.api:app --reload

benchmark:
	python -c "from src.benchmark_comparison import main; main()"

docker-build:
	docker build -t rag-lite:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

build-package:
	python -m build

check-package:
	twine check dist/*

# One-command demo
demo: install-all
	@echo "Building index..."
	rag-lite build --docs data/docs.txt --bm25
	@echo ""
	@echo "Running query..."
	rag-lite query "machine learning" --method bm25 --k 3
	@echo ""
	@echo "✓ Demo complete!"
