# RAG-Lite — Hybrid Retrieval + Reranking + Advanced Evaluation
Production-grade retrieval system with TF-IDF, BM25, dense embeddings, cross-encoder reranking, and comprehensive evaluation metrics. Includes ablation studies comparing retrieval methods. No external APIs required.

Demo: 
![Demo](assets/Demo-rag.png)


## What's included
- **Multi-method retrieval** ([src/rag.py](src/rag.py)):
  - TF-IDF (baseline)
  - BM25 (statistical ranking)
  - Dense embeddings (semantic search via Sentence-BERT)
  - Hybrid fusion (weighted combination)
  - Cross-encoder reranking (ms-marco-MiniLM)
- CLI to build hybrid index ([src/build_index.py](src/build_index.py))
- CLI to query with multiple methods ([src/query.py](src/query.py))
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

### Hybrid retrieval + reranking
```bash
# Build hybrid index with all methods
python src/build_index.py --bm25 --embeddings --reranker --out outputs/index_hybrid.pkl

# Query with hybrid method and reranking
python src/query.py --index outputs/index_hybrid.pkl --q "What is reinforcement learning?" --k 3 --method hybrid --rerank

# Query with specific method
python src/query.py --index outputs/index_hybrid.pkl --q "What is TF-IDF?" --k 3 --method bm25
python src/query.py --index outputs/index_hybrid.pkl --q "What is Recall@K?" --k 3 --method embeddings
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
```

### Ablation study output
```
======================================================================
ABLATION STUDY RESULTS
======================================================================
Method                         Recall@3        MRR@3
----------------------------------------------------------------------
tfidf                          1.0000          1.0000
bm25                           1.0000          0.8333
embeddings                     1.0000          1.0000
hybrid                         1.0000          1.0000
hybrid + Rerank                1.0000          1.0000
======================================================================

Improvements over TF-IDF baseline:
  Recall@3: +0.0%
  MRR@3: +0.0%
```

## File map
- [src/rag.py](src/rag.py): Multi-method retrieval (TF-IDF, BM25, embeddings, hybrid, reranking)
- [src/build_index.py](src/build_index.py): Build index with optional hybrid components
- [src/query.py](src/query.py): CLI for queries with method selection
- [src/ablation.py](src/ablation.py): Ablation study comparing all retrieval methods
- [src/evaluate.py](src/evaluate.py): Advanced evaluation with MRR@K, nDCG@K, Precision@K, Recall@K
- [src/io_utils.py](src/io_utils.py): File I/O helpers
- [data/docs.txt](data/docs.txt): Sample corpus (blank-line separated passages)
- [data/eval.jsonl](data/eval.jsonl): Sample eval set with `query` and `relevant_contains`
- [requirements.txt](requirements.txt): Dependencies (numpy, scikit-learn, rank-bm25, sentence-transformers)
- [outputs/per_query_report.jsonl](outputs/per_query_report.jsonl): Detailed per-query metrics (generated)
- [outputs/worst_20_queries.json](outputs/worst_20_queries.json): Error analysis (generated)
- [outputs/ablation_results.json](outputs/ablation_results.json): Method comparison (generated)

## Architecture

```
Query → Retrieval Methods → Score Fusion → (Optional) Reranking → Top-K Results
         ├─ TF-IDF
         ├─ BM25
         └─ Embeddings
```

## Notes
- Index output is written to `outputs/index.pkl` by default (use `--out` to change)
- Hybrid index includes all methods for fair comparison in ablation studies
- Passages are split on blank lines; adjust `split_passages` in [src/rag.py](src/rag.py) if needed
- First run downloads models (~180MB total for embeddings + reranker)
- Embeddings/reranking add latency but improve quality
- Use ablation study to identify best method for your use case
