# RAG-Lite Quick Reference

## Installation

```bash
# Basic
pip install -e .

# With API
pip install -e ".[api]"

# Everything
pip install -e ".[all]"

# Docker
docker-compose up -d
```

## CLI Commands

```bash
# Build index
rag-lite build --docs data/docs.txt --bm25 --embeddings --reranker

# Query
rag-lite query "your question" --method hybrid --k 5 --rerank

# Evaluate
rag-lite eval --eval-file data/eval.jsonl --method hybrid

# Benchmark
rag-lite benchmark --output results.json --json
```

## API Endpoints

```bash
# Start server
rag-api

# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "method": "hybrid", "k": 5}'

# Build index
curl -X POST http://localhost:8000/build-index \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "data/docs.txt", "bm25": true, "embeddings": true}'

# Metrics
curl http://localhost:8000/metrics

# Docs
open http://localhost:8000/docs
```

## Python API

```python
from src.rag import build_index, retrieve_hybrid
from src.io_utils import read_text

# Build
passages = read_text("data/docs.txt")
index = build_index(passages, bm25=True, embeddings=True)

# Query
results = retrieve_hybrid(index, "machine learning", k=5)

# With reranking
from src.rag import retrieve
results = retrieve(index, "query", method="hybrid", rerank=True)
```

## Configuration

```yaml
# config.yaml
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
```

## Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

## Development

```bash
# Install dev deps
pip install -e ".[dev]"

# Run tests
pytest
pytest --cov=src --cov-report=html

# Format & lint
black src tests
ruff check src tests

# With Makefile
make test
make format
make lint
make api
```

## Method Selection

| Method | Latency | Memory | Quality | Use When |
|--------|---------|--------|---------|----------|
| TF-IDF | ~2ms | 45MB | Baseline | Keywords, speed critical |
| BM25 | ~4ms | 52MB | Good | Recommended baseline |
| Embeddings | ~13ms | 380MB | Better | Semantic search |
| Hybrid | ~16ms | 420MB | Best | Quality matters |
| +Reranking | ~45ms | 520MB | Highest | <100 candidates |

## Benchmarking

```bash
# Run benchmarks
rag-lite benchmark --docs data/docs.txt --trials 20 --output bench.json

# Generate reports
python -c "
from src.benchmark import generate_markdown_report
import json
with open('bench.json') as f:
    results = json.load(f)
generate_markdown_report(results, 'report.md')
"
```

## Cache Management

```python
from src.cache import get_cache_manager

cache = get_cache_manager()

# Stats
print(cache.get_stats())

# Clear
cache.clear_all()
cache.clear_embeddings()
cache.clear_queries()
```

## Troubleshooting

### Index not found
```bash
rag-lite build --docs data/docs.txt --output outputs/index.pkl
```

### API not starting
```bash
# Check port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Change port
uvicorn src.api:app --port 8001
```

### Out of memory
```yaml
# config.yaml
models:
  batch_size: 16  # Reduce from 32
  
cache:
  max_cache_size_mb: 500  # Reduce from 1000
```

### Slow queries
- Enable caching
- Use BM25 instead of embeddings
- Reduce k value
- Don't use reranking

## File Structure

```
rag-lite/
├── src/
│   ├── cli.py          # CLI interface
│   ├── api.py          # REST API
│   ├── rag.py          # Core retrieval
│   ├── config.py       # Configuration
│   ├── cache.py        # Caching
│   ├── benchmark.py    # Benchmarking
│   └── evaluate.py     # Evaluation
├── tests/              # Tests
├── data/               # Data files
├── outputs/            # Results
├── config.yaml         # Configuration
├── Dockerfile          # Docker image
└── docker-compose.yml  # Services
```

## Links

- **Docs**: http://localhost:8000/docs (after starting API)
- **README**: [README.md](README.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Summary**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
