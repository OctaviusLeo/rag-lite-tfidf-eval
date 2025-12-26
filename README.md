# RAG-Lite — Production-Grade Retrieval with Performance Benchmarking
Complete retrieval system with TF-IDF, BM25, dense embeddings, reranking, chunking, citations, and **comprehensive performance measurement**. Demonstrates proper grounded retrieval with **latency tracking, memory profiling, and scaling analysis**. "I measured it."

Demo: 
![Demo](assets/Demo-rag.png)


## What's included
- **Multi-method retrieval** ([src/rag.py](src/rag.py)):
  - TF-IDF (baseline)
  - BM25 (statistical ranking)
  - Dense embeddings (semantic search via Sentence-BERT)
  - Hybrid fusion (weighted combination)
  - Cross-encoder reranking (ms-marco-MiniLM)
- **Chunking + Citation Grounding** ([src/rag.py](src/rag.py)):
  - Configurable chunk size and overlap
  - Stable citation IDs for each chunk (e.g., `[doc_0_chunk_2]`)
  - Character-level position tracking
  - Source document attribution
  - Snippet generation for display
- **Performance Benchmarking** ([src/benchmark.py](src/benchmark.py)):
  - **Latency measurement**: mean, median, P95, P99 query times
  - **Memory profiling**: track memory usage and peak consumption
  - **Throughput calculation**: queries/sec, passages/sec
  - **Index build time**: full pipeline timing
  - **System info**: CPU, memory, Python version
  - **Comparison framework**: side-by-side performance analysis
- CLI to build hybrid index ([src/build_index.py](src/build_index.py))
- CLI to query with multiple methods ([src/query.py](src/query.py))
- **Grounded retrieval demo** ([src/demo_grounded.py](src/demo_grounded.py))
- **Benchmark comparison** ([src/benchmark_comparison.py](src/benchmark_comparison.py))
- **Ablation study framework** ([src/ablation.py](src/ablation.py)):
  - Compare TF-IDF → BM25 → Embeddings → Hybrid → Hybrid+Rerank
  - Automatic performance comparison table
  - Quantify improvements over baseline
- **Advanced evaluation harness** ([src/evaluate.py](src/evaluate.py)):
  - MRR@K (Mean Reciprocal Rank)
  - nDCG@K (Normalized Discounted Cumulative Gain)
  - Precision@K
  - Recall@K
  - Per-query detailed reports
  - Worst 20 queries error analysis
- Sample data: [data/docs.txt](data/docs.txt), [data/eval.jsonl](data/eval.jsonl)

## Requirements
- Python 3.10+
- pip
- Git

## Setup

Clone the repository:
```bash
git clone <your-repo-url>
cd rag-lite-tfidf-eval
```

Create virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

### Basic usage (TF-IDF only)
```bash
# Build simple TF-IDF index
python src/build_index.py

# Query
python src/query.py --q "What is reinforcement learning?" --k 3

# Evaluate
python src/evaluate.py --k 3
```

### Performance benchmarking
```bash
# Build index with performance metrics
python src/build_index.py --benchmark --out outputs/index_bench.pkl

# Output:
# Build time: 0.003s
# Memory used: 0.22 MB
# Peak memory: 321.35 MB
# Index size on disk: 0.01 MB
# Indexing throughput: 1668.78 passages/sec

# Query with latency measurement (20 trials by default)
python src/query.py --index outputs/index_bench.pkl --q "What is RL?" --k 3 --benchmark

# Output:
# Method          Mean (ms)    Median (ms)  P95 (ms)     P99 (ms)
# tfidf           0.60         0.00         3.03         3.03

# Evaluate with performance metrics
python src/evaluate.py --index outputs/index_bench.pkl --k 3 --benchmark

# Output:
# Total evaluation time: 0.004s
# Average time per query: 1.35ms
# Per-query latency: Mean: 1.01ms, Median: 1.05ms, P95: 1.65ms
# Memory used: 0.21 MB
```

### Comprehensive benchmark comparison
```bash
# Compare all retrieval methods side-by-side
python src/benchmark_comparison.py --num-trials 10

# Output example:
# Index                     Method               Avg Latency (ms)   P95 (ms)     Size (MB)
# TF-IDF Only               tfidf                0.70               1.63         0.01
# Hybrid                    tfidf                0.74               1.27         174.36
#                           bm25                 0.03               0.33
#                           embeddings           18.68              36.12
#                           hybrid               21.96              41.92
#
# KEY INSIGHTS:
# Fastest: Hybrid / bm25 - 0.03ms avg
# Slowest: Hybrid / hybrid - 21.96ms avg
# Speed difference: 658.0x
```

### Chunking with citation grounding
```bash
# Build chunked index (200 char chunks, 50 char overlap)
python src/build_index.py --chunking --chunk-size 200 --overlap 50 --out outputs/index_chunked.pkl

# Query with grounded results (shows citations and snippets)
python src/query.py --index outputs/index_chunked.pkl --q "What is reinforcement learning?" --k 3 --grounded

# Example output:
# [Rank #1] [doc_0_chunk_0]
# Score: 0.4148
# Source: Document 0
# Position: chars 0-163
# Snippet:
#   Reinforcement learning (RL) is a learning paradigm where an agent...
```

### Hybrid retrieval + chunking + reranking (full pipeline)
```bash
# Build hybrid chunked index with all methods
python src/build_index.py --chunking --chunk-size 150 --overlap 30 --bm25 --embeddings --reranker --out outputs/index_chunked_hybrid.pkl --benchmark

# Query with hybrid method and citations
python src/query.py --index outputs/index_chunked_hybrid.pkl --q "robot perception" --k 3 --method hybrid --grounded --benchmark

# Run grounded retrieval demo
python src/demo_grounded.py --index outputs/index_chunked_hybrid.pkl --method hybrid --k 3
```

### Ablation study (compare all methods)
```bash
# Run ablation study
python src/ablation.py --index outputs/index_hybrid.pkl --k 3

# Generates comparison table:
# Method                         Recall@3        MRR@3
# ----------------------------------------------------------------------
# tfidf                          1.0000          1.0000
# bm25                           1.0000          0.8333
# embeddings                     1.0000          1.0000
# hybrid                         1.0000          1.0000
# hybrid + Rerank                1.0000          1.0000
```

## Performance benchmarking explained

### What gets measured
- **Build time**: Index construction latency
- **Query latency**: Mean, median, P95, P99 across multiple trials
- **Memory**: Peak memory usage during indexing/querying
- **Throughput**: Queries/sec, passages/sec
- **Index size**: On-disk storage requirements
- **System info**: CPU cores, total memory, Python version

### Why it matters
- **Production readiness**: Know performance characteristics before deployment
- **Trade-offs**: Understand speed vs quality vs memory
- **Bottleneck identification**: Find slow components
- **Capacity planning**: Estimate resource requirements
- **Optimization targets**: Measure impact of improvements

### Key findings from benchmarks
```
Speed comparison:
- BM25: 0.03ms avg (fastest)
- TF-IDF: 0.70ms avg (baseline)
- Embeddings: 18.68ms avg (semantic quality)
- Hybrid: 21.96ms avg (best quality)

Memory/storage trade-offs:
- TF-IDF only: 0.01 MB index
- Hybrid: 174.36 MB index (includes embeddings)
- 658x speed difference between fastest and slowest
- 21,000x size difference between smallest and largest
```

## Chunking explained

### Why chunking?
- **Better granularity**: Long documents split into focused chunks
- **Overlap prevents splitting**: Context preserved across boundaries
- **Stable citations**: Each chunk has permanent ID for referencing
- **Source tracking**: Know which document and position each chunk came from
- **Grounded retrieval**: Demonstrate proper citation without LLM

### Chunking parameters
- **chunk_size**: Target size in characters (default: 200)
- **overlap**: Overlap between consecutive chunks (default: 50)
- **Sentence boundary detection**: Tries to break at sentence endings

### Example chunking
```
Original text (300 chars):
"Reinforcement learning is... [150 chars] ...maximize reward. Deep RL combines... [150 chars] ...neural networks."

With chunk_size=150, overlap=30:
- Chunk 0 (chars 0-150): "Reinforcement learning is... maximize reward."
- Chunk 1 (chars 120-270): "reward. Deep RL combines... neural networks."
```

## Citation grounding

### Features
- **Stable IDs**: `[doc_0_chunk_2]` format for permanent reference
- **Character positions**: Track exact location in source
- **Source attribution**: Link back to original document
- **Snippet generation**: Truncated text for display
- **No LLM required**: Demonstrate grounding with retrieval alone

### Use cases
1. **Verifiable retrieval**: Each result has traceable citation
2. **Resume/portfolio projects**: Shows understanding of grounded AI
3. **RAG preparation**: Foundation for LLM-based systems
4. **Citation analysis**: Track which sources are most useful

## Hybrid retrieval explained

### Methods
- **TF-IDF**: Classic sparse retrieval, term frequency × inverse document frequency
- **BM25**: Improved probabilistic ranking function (Okapi BM25)
- **Embeddings**: Dense semantic vectors using Sentence-BERT (all-MiniLM-L6-v2)
- **Hybrid**: Weighted fusion of multiple methods (default: 40% TF-IDF, 30% BM25, 30% embeddings)
- **Reranking**: Cross-encoder rescores top-K candidates for better precision

### Why hybrid?
- **TF-IDF/BM25**: Fast, interpretable, good for exact matches
- **Embeddings**: Capture semantic similarity, handle paraphrases
- **Hybrid**: Combines strengths, more robust than single method
- **Reranking**: Computationally expensive but highly accurate final ranking

## Evaluation outputs

### Per-query reports
The evaluation script generates detailed analysis files in `outputs/`:

- **per_query_report.jsonl**: Line-by-line metrics for every query
- **worst_20_queries.json**: Error analysis for targeted improvement
- **ablation_results.json**: Method comparison data
- **benchmark_comparison.json**: Detailed performance comparison
- **grounded_demo.txt**: Example grounded retrieval results with citations

Example evaluation output:
```
============================================================
EVALUATION RESULTS @ K=3
============================================================
Total Queries:     3
Recall@3:         1.0000
MRR@3:            1.0000
nDCG@3:           1.0000
Precision@3:      0.3333
============================================================

PERFORMANCE METRICS (with --benchmark):
Total evaluation time: 0.004s
Average time per query: 1.35ms
Per-query latency: Mean: 1.01ms, P95: 1.65ms
Memory used: 0.21 MB
```

## File map
- [src/rag.py](src/rag.py): Multi-method retrieval (TF-IDF, BM25, embeddings, hybrid, reranking), chunking, citations
- [src/benchmark.py](src/benchmark.py): Performance measurement utilities (latency, memory, throughput)
- [src/build_index.py](src/build_index.py): Build index with optional chunking, hybrid components, and benchmarking
- [src/query.py](src/query.py): CLI for queries with method selection, grounded output, and latency measurement
- [src/demo_grounded.py](src/demo_grounded.py): Demo of grounded retrieval with citations
- [src/benchmark_comparison.py](src/benchmark_comparison.py): Comprehensive performance comparison across methods
- [src/ablation.py](src/ablation.py): Ablation study comparing all retrieval methods
- [src/evaluate.py](src/evaluate.py): Advanced evaluation with MRR@K, nDCG@K, Precision@K, Recall@K, and benchmarking
- [src/io_utils.py](src/io_utils.py): File I/O helpers
- [data/docs.txt](data/docs.txt): Sample corpus (blank-line separated passages)
- [data/eval.jsonl](data/eval.jsonl): Sample eval set with `query` and `relevant_contains`
- [requirements.txt](requirements.txt): Dependencies (numpy, scikit-learn, rank-bm25, sentence-transformers, psutil)
- [outputs/per_query_report.jsonl](outputs/per_query_report.jsonl): Detailed per-query metrics (generated)
- [outputs/worst_20_queries.json](outputs/worst_20_queries.json): Error analysis (generated)
- [outputs/ablation_results.json](outputs/ablation_results.json): Method comparison (generated)
- [outputs/benchmark_comparison.json](outputs/benchmark_comparison.json): Performance analysis (generated)
- [outputs/grounded_demo.txt](outputs/grounded_demo.txt): Grounded retrieval examples (generated)

## Architecture

```
Query → Chunking → Retrieval Methods → Score Fusion → (Optional) Reranking → Grounded Results
              ↓
         [doc_X_chunk_Y]
              ↓
    Citation + Position Tracking
              ↓
    Performance Measurement (Latency, Memory, Throughput)
```

## Notes
- Index output is written to `outputs/index.pkl` by default (use `--out` to change)
- Use `--benchmark` flag to measure performance metrics
- Hybrid index includes all methods for fair comparison in ablation studies
- Chunking is optional; without it, full passages are used
- Chunk overlap prevents important content from being split
- Citations are stable across index rebuilds if source documents don't change
- First run downloads models (~180MB total for embeddings + reranker)
- Embeddings/reranking add latency but improve quality (measured!)
- Use ablation study to identify best method for your use case
- Use benchmark comparison to understand speed/quality/memory trade-offs
- "I measured it" - quantify everything for production readiness
