# Changelog

All notable changes to RAG-Lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-07

### Added - Production Features

#### Infrastructure
- **Unified CLI** (`rag-lite`) with subcommands for build, query, eval, and benchmark
- **FastAPI REST API** with OpenAPI documentation
  - `/query` - Query the index
  - `/build-index` - Build index from documents
  - `/load-index` - Load existing index
  - `/health` - Health check endpoint
  - `/metrics` - System metrics
- **Configuration Management**
  - YAML and TOML configuration file support
  - Environment variable support (`RAG_LITE_CONFIG`)
  - Configurable retrieval, models, caching, and API settings
- **Caching Layer**
  - File-based caching for embeddings and query results
  - Optional Redis support for distributed caching
  - Configurable cache size limits and TTL
  - LRU eviction policy

#### Benchmarking & Reporting
- **Enhanced Benchmark Reports**
  - Markdown report generation with detailed metrics
  - JSON report output for programmatic access
  - Latency statistics (mean, median, P95, P99)
  - Memory usage tracking
  - Throughput measurements (QPS, passages/sec)
  - Method comparison tables
- **Performance Recommendations**
  - Guidance on method selection based on use case

#### Development & CI/CD
- **Enhanced CI Pipeline**
  - Benchmark sanity checks in CI
  - Artifact uploads for benchmark results
  - Multi-OS testing (Ubuntu, Windows, macOS)
  - Multi-Python version testing (3.10, 3.11, 3.12)
- **Docker Support**
  - Production-ready Dockerfile
  - Docker Compose with Redis integration
  - Health checks and volume mounts
  - `.dockerignore` optimization

#### Documentation
- **Comprehensive README**
  - One-command quick start
  - Installation instructions (pip, Docker)
  - Usage examples (CLI, API, Python)
  - Performance results and benchmarks
  - Architecture overview
- **Contributing Guidelines** (CONTRIBUTING.md)
  - Development setup
  - Testing guidelines
  - Code style conventions
  - Pull request process
- **Changelog** (this file)

#### Package Management
- **Updated pyproject.toml**
  - Added CLI entry point: `rag-lite`
  - Added API entry point: `rag-api`
  - Optional dependencies: `[api]`, `[cache]`, `[all]`
  - Added dependencies: `pyyaml`, `fastapi`, `uvicorn`, `pydantic`
  - Pip-installable package

### Core Features (Existing)

#### Retrieval Methods
- TF-IDF baseline implementation
- BM25 (Okapi) statistical ranking
- Dense embeddings (Sentence-BERT)
- Hybrid retrieval with configurable fusion
- Cross-encoder reranking

#### Document Processing
- Configurable chunking with overlap
- Citation tracking with stable identifiers
- Source attribution and snippet generation

#### Evaluation
- Comprehensive metrics: MRR@K, nDCG@K, Precision@K, Recall@K
- Per-query analysis and error reporting
- Ablation study framework

### Fixed
- N/A (initial production release)

### Changed
- Consolidated CLI commands into single `rag-lite` entry point
- Enhanced error handling and user feedback across all commands
- Improved logging and progress indicators

### Deprecated
- Individual CLI scripts (`rag-build`, etc.) still work but `rag-lite` is preferred

### Removed
- N/A

### Security
- N/A

---

## [Unreleased]

### Planned Features
- [ ] Async query processing for better API throughput
- [ ] Query result pagination
- [ ] Vector database integration (Weaviate, Qdrant)
- [ ] Additional embedding models (OpenAI, Cohere)
- [ ] Streaming API responses
- [ ] Authentication/authorization for API
- [ ] Rate limiting
- [ ] Distributed caching improvements
- [ ] Monitoring and observability (Prometheus, Grafana)
- [ ] Query analytics dashboard

---

## Version History

- **0.1.0** (2026-01-07) - Production-ready release with API, CLI, caching, and Docker

---

## Migration Guide

### From Previous Versions

If upgrading from a pre-0.1.0 version:

1. **Install new dependencies:**
   ```bash
   pip install -e ".[api]"
   ```

2. **Update CLI commands:**
   - Old: `rag-build --docs data/docs.txt`
   - New: `rag-lite build --docs data/docs.txt`

3. **Optional: Create config file:**
   ```bash
   python -m src.config  # Generates config.yaml
   ```

4. **Indices are compatible** - No need to rebuild

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this changelog.

---

## Links

- [GitHub Repository](https://github.com/OctaviusLeo/rag-lite-tfidf-eval)
- [Issue Tracker](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/issues)
- [Documentation](https://github.com/OctaviusLeo/rag-lite-tfidf-eval/wiki)
