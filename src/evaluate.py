# evaluate.py
# This script evaluates the recall of a retrieval system using a pre-built index and a set of evaluation queries.
from __future__ import annotations
import json
import argparse
import pickle
import os
import math
from typing import List, Tuple, Dict, Any

from rag import retrieve


def calculate_mrr_at_k(relevant_rank: int | None, k: int) -> float:
    """Calculate Mean Reciprocal Rank at K.
    
    Args:
        relevant_rank: Position of first relevant document (1-indexed), None if not found
        k: Cut-off rank
    
    Returns:
        Reciprocal rank (0 if not found in top K)
    """
    if relevant_rank is None or relevant_rank > k:
        return 0.0
    return 1.0 / relevant_rank


def calculate_ndcg_at_k(relevant_positions: List[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        relevant_positions: List of positions where relevant docs appear (1-indexed)
        k: Cut-off rank
    
    Returns:
        nDCG@K score
    """
    # DCG calculation
    dcg = 0.0
    for pos in relevant_positions:
        if pos <= k:
            # Gain = 1 for relevant, discount by log2(pos + 1)
            dcg += 1.0 / math.log2(pos + 1)
    
    # IDCG (ideal DCG) - assumes all relevant docs are at top
    num_relevant = len([p for p in relevant_positions if p <= k])
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_precision_at_k(num_relevant: int, k: int) -> float:
    """Calculate Precision at K.
    
    Args:
        num_relevant: Number of relevant documents in top K
        k: Cut-off rank
    
    Returns:
        Precision@K score
    """
    return num_relevant / k


def evaluate_query(query: str, relevant_contains: str, index: Any, k: int) -> Dict[str, Any]:
    """Evaluate a single query and return detailed metrics.
    
    Args:
        query: The query string
        relevant_contains: Text that relevant documents must contain
        index: The retrieval index
        k: Number of documents to retrieve
    
    Returns:
        Dictionary with query results and metrics
    """
    hits = retrieve(index, query, k=k)
    
    # Find relevant documents
    relevant_positions = []
    for rank, (doc_id, score, passage) in enumerate(hits, start=1):
        if relevant_contains.lower() in passage.lower():
            relevant_positions.append(rank)
    
    # Calculate metrics
    has_relevant = len(relevant_positions) > 0
    first_relevant_rank = relevant_positions[0] if relevant_positions else None
    
    recall_at_k = 1.0 if has_relevant else 0.0
    mrr_at_k = calculate_mrr_at_k(first_relevant_rank, k)
    ndcg_at_k = calculate_ndcg_at_k(relevant_positions, k)
    precision_at_k = calculate_precision_at_k(len(relevant_positions), k)
    
    return {
        "query": query,
        "relevant_contains": relevant_contains,
        "recall@k": recall_at_k,
        "mrr@k": mrr_at_k,
        "ndcg@k": ndcg_at_k,
        "precision@k": precision_at_k,
        "first_relevant_rank": first_relevant_rank,
        "num_relevant_found": len(relevant_positions),
        "retrieved_passages": [
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "passage": passage[:200],  # Truncate for readability
                "is_relevant": rank in relevant_positions
            }
            for rank, (doc_id, score, passage) in enumerate(hits, start=1)
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/index.pkl")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Missing index: {args.index} (run build_index.py first)")

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    # Evaluate all queries
    query_results = []
    with open(args.eval, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result = evaluate_query(row["query"], row["relevant_contains"], index, args.k)
            query_results.append(result)

    # Aggregate metrics
    total = len(query_results)
    avg_recall = sum(r["recall@k"] for r in query_results) / total
    avg_mrr = sum(r["mrr@k"] for r in query_results) / total
    avg_ndcg = sum(r["ndcg@k"] for r in query_results) / total
    avg_precision = sum(r["precision@k"] for r in query_results) / total

    # Print summary
    print("=" * 60)
    print(f"EVALUATION RESULTS @ K={args.k}")
    print("=" * 60)
    print(f"Total Queries:     {total}")
    print(f"Recall@{args.k}:         {avg_recall:.4f}")
    print(f"MRR@{args.k}:            {avg_mrr:.4f}")
    print(f"nDCG@{args.k}:           {avg_ndcg:.4f}")
    print(f"Precision@{args.k}:      {avg_precision:.4f}")
    print("=" * 60)

    # Save per-query report
    os.makedirs(args.output_dir, exist_ok=True)
    per_query_path = os.path.join(args.output_dir, "per_query_report.jsonl")
    with open(per_query_path, "w", encoding="utf-8") as f:
        for result in query_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n✓ Per-query report saved to: {per_query_path}")

    # Identify and save worst 20 queries (sorted by MRR, then nDCG)
    sorted_by_performance = sorted(
        query_results, 
        key=lambda x: (x["mrr@k"], x["ndcg@k"], x["recall@k"])
    )
    worst_queries = sorted_by_performance[:min(20, len(sorted_by_performance))]
    
    worst_queries_path = os.path.join(args.output_dir, "worst_20_queries.json")
    with open(worst_queries_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": f"Worst performing queries sorted by MRR@{args.k}, nDCG@{args.k}, Recall@{args.k}",
            "k": args.k,
            "worst_queries": worst_queries
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Worst 20 queries analysis saved to: {worst_queries_path}")
    
    # Print summary of worst queries
    print(f"\n{'=' * 60}")
    print(f"WORST {min(20, len(worst_queries))} QUERIES")
    print("=" * 60)
    for i, result in enumerate(worst_queries[:10], 1):  # Show top 10 in console
        print(f"{i}. Query: {result['query'][:60]}...")
        print(f"   MRR@{args.k}: {result['mrr@k']:.3f} | nDCG@{args.k}: {result['ndcg@k']:.3f} | Recall@{args.k}: {result['recall@k']:.3f}")
    
    if len(worst_queries) > 10:
        print(f"\n   ... and {len(worst_queries) - 10} more (see {worst_queries_path})")
    print("=" * 60)


if __name__ == "__main__":
    main()
