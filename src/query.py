# query.py
# This script provides a command-line interface for querying a retrieval-based answer generation (RAG) model using a pre-built index.
from __future__ import annotations
import argparse
import pickle
import os

from rag import retrieve, retrieve_hybrid

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/index.pkl")
    parser.add_argument("--q", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--method", default="tfidf", 
                       choices=["tfidf", "bm25", "embeddings", "hybrid"],
                       help="Retrieval method")
    parser.add_argument("--rerank", action="store_true", help="Apply reranking")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Missing index: {args.index} (run build_index.py first)")

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    # Use hybrid retrieval if method is specified (other than tfidf) or rerank is enabled
    if args.method != "tfidf" or args.rerank:
        hits = retrieve_hybrid(
            index, args.q, k=args.k, method=args.method, rerank=args.rerank
        )
        print(f"Method: {args.method}{' + rerank' if args.rerank else ''}")
    else:
        hits = retrieve(index, args.q, k=args.k)
        print("Method: tfidf (legacy)")
    
    print(f"Query: {args.q}")
    print("=" * 60)
    
    for rank, (i, score, passage) in enumerate(hits, start=1):
        print(f"#{rank} score={score:.3f} passage_id={i}")
        print(passage)
        print("-" * 60)

if __name__ == "__main__":
    main()
