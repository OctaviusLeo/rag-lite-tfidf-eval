# RAG-Lite — TF-IDF Retrieval + Simple Evaluation
A minimal **retrieval + evaluation** scaffold that works without any external model APIs.
Use it as a clean engineering baseline, then swap in embeddings + an LLM later.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Build index from sample docs
python src/build_index.py

# Query (returns top passages)
python src/query.py --q "What is reinforcement learning?"

# Run a tiny eval set (Recall@K)
python src/evaluate.py
```

## Future Features
- Replace TF-IDF with embeddings + ANN index (FAISS), add latency benchmarks
- Add reranking and ablations (retrieval-only vs retrieval+rerrank)
- Add offline evaluation harness and error analysis notebook
