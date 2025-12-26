# RAG-Lite — TF-IDF Retrieval + Advanced Evaluation
Minimal retrieval baseline: build a TF-IDF index over passages, query it, and evaluate with production-grade metrics (Recall@K, MRR@K, nDCG@K, Precision@K). Includes detailed per-query reports and error analysis. No external model APIs required.

Demo: 
![Demo](assets/Demo-rag.png)


## What's included
- TF-IDF indexing and retrieval ([src/rag.py](src/rag.py))
- CLI to build index ([src/build_index.py](src/build_index.py))
- CLI to query top-k passages ([src/query.py](src/query.py))
- **Advanced evaluation harness** ([src/evaluate.py](src/evaluate.py)):
  - **MRR@K** (Mean Reciprocal Rank)
  - **nDCG@K** (Normalized Discounted Cumulative Gain)
  - **Precision@K**
  - **Recall@K**
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

Build index, query, and evaluate:
```bash
# Build index from sample docs
python src/build_index.py

# Query (returns top passages)
python src/query.py --q "What is reinforcement learning?" --k 3

# Run evaluation with advanced metrics
python src/evaluate.py --k 3

# One-shot quickstart (build -> query -> eval -> summary)
python src/run_quickstart.py --k 3 --summary outputs/quickstart_summary.txt
```

## Evaluation outputs
The evaluation script generates detailed analysis files in `outputs/`:

- **per_query_report.jsonl**: Line-by-line metrics for every query, including:
  - All metrics (Recall@K, MRR@K, nDCG@K, Precision@K)
  - Retrieved passages with scores and relevance labels
  - First relevant document rank

- **worst_20_queries.json**: Error analysis identifying the 20 worst-performing queries sorted by MRR@K, nDCG@K, and Recall@K for targeted improvement

Example output:
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

## File map
- [src/rag.py](src/rag.py): TF-IDF index and retrieve
- [src/build_index.py](src/build_index.py): build and persist index
- [src/query.py](src/query.py): CLI for queries
- [src/evaluate.py](src/evaluate.py): Advanced evaluation with MRR@K, nDCG@K, Precision@K, Recall@K, per-query reports, and error analysis
- [src/io_utils.py](src/io_utils.py): small file helpers
- [data/docs.txt](data/docs.txt): sample corpus (blank-line separated passages)
- [data/eval.jsonl](data/eval.jsonl): sample eval set with `query` and `relevant_contains`
- [requirements.txt](requirements.txt): numpy, scikit-learn
- [outputs/per_query_report.jsonl](outputs/per_query_report.jsonl): Detailed per-query metrics (generated)
- [outputs/worst_20_queries.json](outputs/worst_20_queries.json): Error analysis of worst performers (generated)

## Notes
- Index output is written to `outputs/index.pkl` by default.
- Passages are split on blank lines; adjust `split_passages` in [src/rag.py](src/rag.py) if your corpus differs.
- Increase `--k` in query/eval to change the number of retrieved passages.
- Use the worst queries analysis to identify failure modes and improve your retrieval system.
