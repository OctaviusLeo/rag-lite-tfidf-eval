# RAG‑Lite: Production-Ready Retrieval System

[![CI](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/actions)
[![codecov](https://codecov.io/gh/OctaviusLeo/rag-lite-tfidf-eval/branch/main/graph/badge.svg)](https://codecov.io/gh/OctaviusLeo/rag-lite-tfidf-eval)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG‑Lite is a compact, production‑style retrieval stack: multiple retrieval methods (TF‑IDF, BM25, dense embeddings), optional cross‑encoder reranking, chunking with stable citations, and an evaluation + benchmarking harness.

This repo is optimized for demonstrating real engineering trade‑offs (quality vs. latency vs. memory) with reproducible metrics.

## Table of Contents

- [Overview (features & modules)](#core-features)
- [Setup (requirements & installation)](#requirements)
- [Quickstart](#quickstart)
  - [More examples](#basic-usage-tf-idf-only)
    - [Basic usage](#basic-usage-tf-idf-only)
    - [Performance benchmarking](#performance-benchmarking)
    - [Benchmark comparison](#comprehensive-benchmark-comparison)
    - [Chunking with grounding](#chunking-with-citation-grounding)
    - [Hybrid + reranking](#hybrid-retrieval--chunking--reranking-full-pipeline)
    - [Ablation study](#ablation-study-compare-all-methods)
- [Reference: Performance & Evaluation](#performance-analysis)
  - [Performance Analysis](#performance-analysis)
  - [Citation Grounding](#citation-grounding)
  - [Evaluation Framework](#evaluation-framework)
- [Reference: Project Structure](#project-structure)
  - [System Architecture](#system-architecture)
- [Development](#development)
  - [Testing](#testing)
  - [Code Quality](#code-quality)
- [Troubleshooting](#troubleshooting)
  - [CI & Contributing](#continuous-integration)
  - [Technical Notes](#technical-notes)

---

### Demo

![Demo RAG](assets/Demo-rag.png)

---

<details>
<summary><strong>Overview (features & modules)</strong></summary>

---

### Core Features

**Multi-Method Retrieval** ([src/rag.py](src/rag.py))
- TF-IDF baseline implementation
- BM25 (Okapi) statistical ranking
- Dense embeddings using Sentence-BERT (all-MiniLM-L6-v2)
- Hybrid score fusion with configurable weights
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)

**Document Chunking and Citation Tracking** ([src/rag.py](src/rag.py))
- Configurable chunk size and overlap parameters
- Stable citation identifiers (`[doc_0_chunk_2]`)
- Character-level position tracking
- Source document attribution
- Automated snippet generation

**Performance Benchmarking** ([src/benchmark.py](src/benchmark.py))
- Latency measurement: mean, median, P95, P99 query times
- Memory profiling: usage tracking and peak consumption
- Throughput calculation: queries per second, passages per second
- Index build time analysis
- System information capture (CPU, memory, Python version)
- Cross-method comparison framework

**Evaluation and Analysis**
- Advanced metrics: MRR@K, nDCG@K, Precision@K, Recall@K ([src/evaluate.py](src/evaluate.py))
- Ablation study framework for method comparison ([src/ablation.py](src/ablation.py))
- Per-query detailed reports with error analysis
- Comprehensive benchmark comparison tool ([src/benchmark_comparison.py](src/benchmark_comparison.py))

**Command-Line Interface**
- Index building with hybrid components ([src/build_index.py](src/build_index.py))
- Query execution with multiple retrieval methods ([src/query.py](src/query.py))
- Grounded retrieval demonstration ([src/demo_grounded.py](src/demo_grounded.py))

</details>

<details>
<summary><strong>Setup (requirements & installation)</strong></summary>

## Requirements
- Python 3.10 or higher
- pip package manager
- Git version control

## Installation

Clone the repository:
```bash
git clone https://github.com/OctaviusLeo/rag-lite-tfidf-eval.git
cd rag-lite-tfidf-eval

# Basic installation
pip install -e .

# With API support
pip install -e ".[api]"

# With all features (dev tools, API, Redis cache)
pip install -e ".[all]"
```

### With Docker

```bash
# Build and run
docker-compose up -d

# API available at http://localhost:8000
curl http://localhost:8000/health
```

---

## 🎯 Features

### Core Retrieval Methods
- **TF-IDF**: Fast baseline, minimal memory footprint
- **BM25**: Statistical ranking with better relevance
- **Dense Embeddings**: Semantic search using Sentence-BERT
- **Hybrid**: Combines lexical + semantic for best quality
- **Cross-Encoder Reranking**: Highest quality, for top candidates

### Production Engineering
✅ **CLI**: Unified command interface (`rag-lite`)  
✅ **REST API**: FastAPI with OpenAPI docs  
✅ **Configuration**: YAML/TOML config files  
✅ **Caching**: File-based + optional Redis  
✅ **Benchmarking**: JSON + Markdown reports  
✅ **CI/CD**: Tests, linting, benchmark sanity checks  
✅ **Docker**: Ready for deployment  
✅ **Pip-installable**: Standard Python packaging  

---

## 📖 Usage

### CLI

```bash
# Build index
rag-lite build --docs data/docs.txt --bm25 --embeddings

# Query
rag-lite query "what is machine learning?" --method hybrid --k 5

# Evaluate
rag-lite eval --eval-file data/eval.jsonl --method hybrid

# Benchmark
rag-lite benchmark --output outputs/benchmark.json
```

### REST API

Start the server:
```bash
rag-api
# or
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Query via HTTP:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is machine learning?",
    "method": "hybrid",
    "k": 5
  }'
```

Build index:
```bash
curl -X POST http://localhost:8000/build-index \
  -H "Content-Type: application/json" \
  -d '{
    "docs_path": "data/docs.txt",
    "bm25": true,
    "embeddings": true
  }'
```

### Python API

```python
from src.rag import build_index, retrieve_hybrid
from src.io_utils import read_text

# Build index
passages = read_text("data/docs.txt")
index = build_index(passages, bm25=True, embeddings=True)

# Query
results = retrieve_hybrid(index, "machine learning", k=5)

for text, score in results:
    print(f"[{score:.4f}] {text[:100]}...")
```

### Configuration

Create `config.yaml`:
```yaml
retrieval:
  default_method: hybrid
  default_k: 10
  enable_reranking: true

models:
  embedder_model: sentence-transformers/all-MiniLM-L6-v2
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

cache:
  enabled: true
  cache_dir: .cache/rag-lite
  query_cache_ttl: 3600

api:
  host: 0.0.0.0
  port: 8000
```

Set via environment:
```bash
export RAG_LITE_CONFIG=config.yaml
rag-api
```

---

## 📊 Performance Results

### Benchmark Summary

| Method | Mean Latency | P95 | Memory | QPS |
|--------|-------------|-----|--------|-----|
| TF-IDF | 2.3ms | 3.1ms | 45 MB | 435 |
| BM25 | 3.7ms | 5.2ms | 52 MB | 270 |
| Embeddings | 12.5ms | 18.3ms | 380 MB | 80 |
| Hybrid | 15.8ms | 22.1ms | 420 MB | 63 |
| + Reranking | 45.2ms | 61.7ms | 520 MB | 22 |

### Quality Metrics (on eval.jsonl)

| Method | MRR@10 | nDCG@10 | Recall@10 |
|--------|--------|---------|-----------|
| TF-IDF | 0.523 | 0.612 | 0.741 |
| BM25 | 0.587 | 0.668 | 0.798 |
| Embeddings | 0.652 | 0.721 | 0.856 |
| Hybrid | **0.698** | **0.769** | **0.891** |
| + Reranking | **0.734** | **0.801** | **0.905** |

*Tested on: Intel i7-9750H, 16GB RAM, Python 3.11*

**Full reports**: [outputs/benchmark_report.md](outputs/benchmark_report.md)

---

## 🏗️ Architecture

```
rag-lite/
├── src/
│   ├── cli.py              # Unified CLI interface
│   ├── api.py              # FastAPI REST API
│   ├── rag.py              # Core retrieval logic
│   ├── config.py           # Configuration management
│   ├── cache.py            # Caching layer
│   ├── benchmark.py        # Performance measurement
│   ├── evaluate.py         # Evaluation metrics
│   └── ...
├── tests/                  # Test suite
├── data/
│   ├── docs.txt           # Sample documents
│   └── eval.jsonl         # Evaluation dataset
├── config.yaml            # Default configuration
├── Dockerfile             # Container image
├── docker-compose.yml     # Multi-service setup
└── pyproject.toml         # Package metadata
```

### Key Design Decisions

1. **Multi-Method**: Supports TF-IDF, BM25, embeddings, hybrid retrieval
2. **Benchmarking First**: Built-in performance tracking with detailed reports
3. **Caching**: Reduces redundant computations (embeddings, queries)
4. **API + CLI**: Both programmatic and command-line access
5. **Docker Ready**: One-command deployment for production

---

## 🧪 Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Fast tests only
pytest -m "not slow"
```

### Code Quality

```bash
# Format
black src tests

# Lint
ruff check src tests

# Type check (if using mypy)
mypy src
```

### Benchmark Generation

```bash
# Run full benchmarks and generate reports
python -c "
from src.benchmark import benchmark_all_methods, generate_markdown_report
from src.io_utils import read_text

passages = read_text('data/docs.txt')
results = benchmark_all_methods(passages, num_trials=20)
generate_markdown_report(results, 'outputs/benchmark_report.md')
"
```

---

## 🐳 Docker Deployment

### Single Container

```bash
docker build -t rag-lite .
docker run -p 8000:8000 -v $(pwd)/data:/app/data rag-lite
```

### With Docker Compose (includes Redis)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with corresponding tests
4. Run tests and linting locally
5. Submit a pull request with a clear description

## Technical Notes
- Index output defaults to `outputs/index.pkl` (configurable with `--out` flag)
- Performance metrics available via `--benchmark` flag
- Hybrid index includes all retrieval methods for ablation studies
- Document chunking is optional (defaults to full passages)
- Chunk overlap preserves context across boundaries
- Citation identifiers remain stable across index rebuilds
- First run downloads pre-trained models (approximately 180MB for embeddings and reranker)
- Embedding-based methods trade latency for semantic quality
- Ablation studies help identify optimal method for specific use cases
- Benchmark comparisons quantify speed, quality, and memory trade-offs

</details>

# Future Plans
- AG-Lite as a real product (shipping + retrieval/eval work look like software).
