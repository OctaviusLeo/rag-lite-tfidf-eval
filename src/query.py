# query.py
# This script provides a command-line interface for querying a retrieval-based answer generation (RAG) model using a pre-built index.
from __future__ import annotations
import argparse
import pickle
import os

from rag import retrieve, retrieve_hybrid, retrieve_grounded

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/index.pkl")
    parser.add_argument("--q", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--method", default="tfidf", 
                       choices=["tfidf", "bm25", "embeddings", "hybrid"],
                       help="Retrieval method")
    parser.add_argument("--rerank", action="store_true", help="Apply reranking")
    parser.add_argument("--grounded", action="store_true", 
                       help="Show grounded results with citations and snippets")
    parser.add_argument("--snippet-length", type=int, default=150,
                       help="Max snippet length for grounded results")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Missing index: {args.index} (run build_index.py first)")

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    # Determine if we should use grounded retrieval
    use_grounded = args.grounded or (index.chunks is not None)
    
    if use_grounded and (args.method != "tfidf" or args.rerank or index.chunks):
        # Use grounded retrieval with citations
        results = retrieve_grounded(
            index, args.q, k=args.k, method=args.method, 
            rerank=args.rerank, snippet_length=args.snippet_length
        )
        
        print(f"Method: {args.method}{' + rerank' if args.rerank else ''} (grounded)")
        if index.chunks:
            print(f"Mode: Chunked retrieval ({len(index.chunks)} chunks)")
        print(f"Query: {args.q}")
        print("=" * 70)
        
        for result in results:
            print(f"\n[Rank #{result.rank}] {result.citation}")
            print(f"Score: {result.score:.4f}")
            if result.source_doc_id is not None:
                print(f"Source: Document {result.source_doc_id}")
            if result.char_range:
                print(f"Position: chars {result.char_range[0]}-{result.char_range[1]}")
            print(f"\nSnippet:")
            print(f"  {result.snippet if result.snippet else result.text}")
            print("-" * 70)
    
    else:
        # Use legacy retrieval (backward compatible)
        if args.method != "tfidf" or args.rerank:
            hits = retrieve_hybrid(
                index, args.q, k=args.k, method=args.method, rerank=args.rerank
            )
            print(f"Method: {args.method}{' + rerank' if args.rerank else ''}")
        else:
            hits = retrieve(index, args.q, k=args.k)
            print("Method: tfidf (legacy)")
        
        print(f"Query: {args.q}")
        print("=" * 70)
        
        for rank, (i, score, passage) in enumerate(hits, start=1):
            print(f"\n#{rank} score={score:.3f} passage_id={i}")
            print(passage)
            print("-" * 70)

if __name__ == "__main__":
    main()
