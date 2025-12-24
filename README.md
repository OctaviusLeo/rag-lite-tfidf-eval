# RAG-Lite — TF-IDF Retrieval + Simple Evaluation
A minimal **retrieval + evaluation** scaffold that works without any external model APIs.
Use it as a clean engineering baseline, then swap in embeddings + an LLM later.

Demo: 

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

# (Optional) Run the scripted demo used for GIF capture
pwsh -File scripts/demo.ps1
```

### Capture the CLI as a GIF (asciinema + agg)
```bash
# 1) Record a terminal cast (WSL/mac/Linux recommended)
asciinema rec demo.cast -c "pwsh -File scripts/demo.ps1"

# 2) Render to GIF (no editing needed)
npx agg demo.cast demo.gif --theme dracula --padding 12 --speed 1.05
```
Keep the terminal around 100 columns and the run under ~20s for a crisp, lightweight GIF.

## Future Features
- Replace TF-IDF with embeddings + ANN index (FAISS), add latency benchmarks
- Add reranking and ablations (retrieval-only vs retrieval+rerrank)
- Add offline evaluation harness and error analysis notebook
