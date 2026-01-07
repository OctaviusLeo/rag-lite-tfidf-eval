# RAG‑Lite: Production-Ready Retrieval System

[![CI](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/actions)
[![codecov](https://codecov.io/gh/OctaviusLeo/rag-lite-tfidf-eval/branch/main/graph/badge.svg)](https://codecov.io/gh/OctaviusLeo/rag-lite-tfidf-eval)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAG-Lite** is a production-grade retrieval system demonstrating real engineering trade-offs (quality vs latency vs memory) with reproducible metrics. It features multiple retrieval methods (TF-IDF, BM25, dense embeddings, hybrid), optional reranking, comprehensive benchmarking, and a REST API—all pip-installable and Docker-ready.

---

## 🚀 Quick Start (One Command)

```bash
# Install
pip install -e ".[api]"

# Run API server
rag-api

# Or use CLI
rag-lite query "your question here"
```

Visit http://localhost:8000/docs for interactive API documentation.

---

## 📦 Installation

### From Source

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

### Quick Guidelines

1. **Fork** and create a feature branch
2. **Add tests** for new features
3. **Run tests** and linting before committing
4. **Update docs** as needed
5. Submit a **pull request**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Documentation**: [Full docs](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/wiki)
- **Issues**: [Report bugs](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## 🎓 Citation

If you use RAG-Lite in your research or project, please cite:

```bibtex
@software{rag_lite,
  title={RAG-Lite: Production-Ready Retrieval System},
  author={Your Name},
  year={2026},
  url={https://github.com/OctaviusLeo/rag-lite-tfidf-eval}
}
```

---

## ⭐ Highlights

**Why RAG-Lite?**

- ✅ **Shippable**: pip install + one command to run
- ✅ **Reproducible**: Deterministic benchmarks with CI checks
- ✅ **Production-Ready**: API, caching, config, Docker
- ✅ **Well-Tested**: >80% coverage, CI on multiple OS/Python versions
- ✅ **Documented**: Clear README, API docs, contribution guide
- ✅ **Engineered**: Not a notebook—real software architecture

Perfect for demonstrating IR + evaluation knowledge **AND** software engineering skills.
