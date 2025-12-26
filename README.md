# RAG-Lite — Grounded Retrieval with Chunking + Citations
Production-grade retrieval system with TF-IDF, BM25, dense embeddings, cross-encoder reranking, **chunking with overlap**, and **citation-grounded results**. Demonstrates proper grounded retrieval without requiring an LLM. Includes comprehensive evaluation metrics and ablation studies.

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
  - Grounded retrieval without LLM
- CLI to build hybrid index ([src/build_index.py](src/build_index.py))
- CLI to query with multiple methods ([src/query.py](src/query.py))
- **Grounded retrieval demo** ([src/demo_grounded.py](src/demo_grounded.py))
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
python src/build_index.py --chunking --chunk-size 150 --overlap 30 --bm25 --embeddings --reranker --out outputs/index_chunked_hybrid.pkl

# Query with hybrid method and citations
python src/query.py --index outputs/index_chunked_hybrid.pkl --q "robot perception" --k 3 --method hybrid --grounded

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
```

### Grounded output example
```
Query: What is reinforcement learning?
======================================================================
Retrieved Information:

[doc_0_chunk_0]
  Score: 0.6659
  Source: Document 0
  Reinforcement learning (RL) is a learning paradigm where an agent...

Citation Summary:
  [doc_0_chunk_0] - Document 0, chars (0, 150)
```

## File map
- [src/rag.py](src/rag.py): Multi-method retrieval, chunking, citation grounding
- [src/build_index.py](src/build_index.py): Build index with optional chunking and hybrid components
- [src/query.py](src/query.py): CLI for queries with method selection and grounded output
- [src/demo_grounded.py](src/demo_grounded.py): Demo of grounded retrieval with citations
- [src/ablation.py](src/ablation.py): Ablation study comparing all retrieval methods
- [src/evaluate.py](src/evaluate.py): Advanced evaluation with MRR@K, nDCG@K, Precision@K, Recall@K
- [src/io_utils.py](src/io_utils.py): File I/O helpers
- [data/docs.txt](data/docs.txt): Sample corpus (blank-line separated passages)
- [data/eval.jsonl](data/eval.jsonl): Sample eval set with `query` and `relevant_contains`
- [requirements.txt](requirements.txt): Dependencies (numpy, scikit-learn, rank-bm25, sentence-transformers)
- [outputs/per_query_report.jsonl](outputs/per_query_report.jsonl): Detailed per-query metrics (generated)
- [outputs/worst_20_queries.json](outputs/worst_20_queries.json): Error analysis (generated)
- [outputs/ablation_results.json](outputs/ablation_results.json): Method comparison (generated)
- [outputs/grounded_demo.txt](outputs/grounded_demo.txt): Grounded retrieval examples (generated)

## Architecture

```
Query → Chunking → Retrieval Methods → Score Fusion → (Optional) Reranking → Grounded Results
              ↓
         [doc_X_chunk_Y]
              ↓
    Citation + Position Tracking
```

## Notes
- Index output is written to `outputs/index.pkl` by default (use `--out` to change)
- Hybrid index includes all methods for fair comparison in ablation studies
- Chunking is optional; without it, full passages are used
- Chunk overlap prevents important content from being split
- Citations are stable across index rebuilds if source documents don't change
- First run downloads models (~180MB total for embeddings + reranker)
- Embeddings/reranking add latency but improve quality
- Use ablation study to identify best method for your use case
- Grounded retrieval works without LLM, demonstrating proper citation practices
