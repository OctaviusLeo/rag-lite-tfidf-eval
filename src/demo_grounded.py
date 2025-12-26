# demo_grounded.py
# This script demonstrates grounded retrieval with chunking and citations.
from __future__ import annotations
import pickle
import os
from typing import List

from rag import retrieve_grounded, GroundedResult


def format_grounded_answer(query: str, results: List[GroundedResult]) -> str:
    """Format a grounded answer with citations in-line.
    
    This demonstrates how to present retrieval results with proper citations,
    similar to how a RAG system would ground LLM responses.
    """
    answer_parts = [f"Query: {query}\n"]
    answer_parts.append("=" * 70)
    answer_parts.append("\nRetrieved Information:\n")
    
    for result in results:
        answer_parts.append(f"\n{result.citation}")
        answer_parts.append(f"  Score: {result.score:.4f}")
        if result.source_doc_id is not None:
            answer_parts.append(f"  Source: Document {result.source_doc_id}")
        answer_parts.append(f"  {result.snippet}\n")
    
    answer_parts.append("\n" + "=" * 70)
    answer_parts.append("\nCitation Summary:")
    for result in results:
        answer_parts.append(f"  {result.citation} - Document {result.source_doc_id}, chars {result.char_range}")
    
    return "\n".join(answer_parts)


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo grounded retrieval with citations")
    parser.add_argument("--index", default="outputs/index_chunked.pkl", 
                       help="Path to chunked index")
    parser.add_argument("--queries", nargs="+", 
                       default=["What is reinforcement learning?", 
                               "How do you evaluate retrieval?"],
                       help="Queries to run")
    parser.add_argument("--k", type=int, default=3, help="Number of results")
    parser.add_argument("--method", default="hybrid", 
                       choices=["tfidf", "bm25", "embeddings", "hybrid"])
    parser.add_argument("--output", default="outputs/grounded_demo.txt",
                       help="Output file for results")
    args = parser.parse_args()
    
    if not os.path.exists(args.index):
        print(f"Error: Index not found at {args.index}")
        print("Build a chunked index first:")
        print("  python src/build_index.py --chunking --chunk-size 200 --overlap 50 --out outputs/index_chunked.pkl")
        return
    
    print("Loading index...")
    with open(args.index, "rb") as f:
        index = pickle.load(f)
    
    if not index.chunks:
        print("Warning: Index does not have chunking enabled. Citations will be basic.")
    else:
        print(f"Index loaded: {len(index.chunks)} chunks from {len(set(c.source_doc_id for c in index.chunks))} documents")
    
    print(f"\nRunning {len(args.queries)} queries with method: {args.method}")
    print("=" * 70)
    
    all_outputs = []
    
    for i, query in enumerate(args.queries, 1):
        print(f"\n[Query {i}/{len(args.queries)}] {query}")
        
        results = retrieve_grounded(
            index, query, k=args.k, method=args.method
        )
        
        # Display results
        for result in results:
            print(f"  {result.citation} (score={result.score:.3f}) - {result.snippet[:80]}...")
        
        # Generate formatted output
        formatted = format_grounded_answer(query, results)
        all_outputs.append(formatted)
    
    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(all_outputs))
    
    print(f"\n{'=' * 70}")
    print(f"Grounded results saved to: {args.output}")
    print("\nKey Features Demonstrated:")
    print("  ✓ Chunking with overlap for better context")
    print("  ✓ Stable citation IDs for each chunk")
    print("  ✓ Character-level position tracking")
    print("  ✓ Source document attribution")
    print("  ✓ Grounded retrieval without LLM")


if __name__ == "__main__":
    main()
