# evaluate.py
# Evaluation harness for retrieval system performance measurement.
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from typing import Any

from src.benchmark import Benchmark, format_system_info, get_system_info
from src.rag import retrieve


def calculate_mrr_at_k(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    """Calculate Mean Reciprocal Rank at K.

    Args:
        relevant_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs (in ranked order)

    Returns:
        Reciprocal rank of first relevant document (0 if not found)
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        relevant_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs (in ranked order)

    Returns:
        nDCG@K score
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # DCG calculation
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG (ideal DCG) - assumes all relevant docs at top
    k = len(retrieved_ids)
    num_relevant_in_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant_in_k))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_precision_at_k(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    """Calculate Precision at K.

    Args:
        relevant_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs

    Returns:
        Precision@K score
    """
    if not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    num_relevant_retrieved = len(relevant_set & retrieved_set)
    return num_relevant_retrieved / len(retrieved_ids)


def calculate_recall_at_k(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    """Calculate Recall at K.

    Args:
        relevant_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs

    Returns:
        Recall@K score (proportion of relevant docs retrieved)
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    num_relevant_retrieved = len(relevant_set & retrieved_set)
    return num_relevant_retrieved / len(relevant_set)


def evaluate_query(
    index: Any, query: str, relevant_ids: list[int], k: int, _method: str = "tfidf"
) -> dict[str, Any]:
    """Evaluate a single query and return detailed metrics.

    Args:
        index: The retrieval index
        query: The query string
        relevant_ids: List of relevant document IDs
        k: Number of documents to retrieve
        _method: Retrieval method to use (ignored, kept for backwards compatibility)

    Returns:
        Dictionary with query results and metrics
    """
    hits = retrieve(index, query, k=k)

    # Extract document IDs from results (tuples of (doc_id, score, passage))
    retrieved_ids = [doc_id for doc_id, _score, _passage in hits]

    # Calculate metrics
    recall = calculate_recall_at_k(relevant_ids, retrieved_ids)
    mrr = calculate_mrr_at_k(relevant_ids, retrieved_ids)
    ndcg = calculate_ndcg_at_k(relevant_ids, retrieved_ids)
    precision = calculate_precision_at_k(relevant_ids, retrieved_ids)

    return {
        "query": query,
        "relevant_ids": relevant_ids,
        "retrieved_ids": retrieved_ids,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "precision": precision,
    }


def evaluate_query_with_pattern(
    query: str, relevant_contains: str, index: Any, k: int
) -> dict[str, Any]:
    """Evaluate a single query using text pattern matching (legacy API).

    Args:
        query: The query string
        relevant_contains: Text pattern that relevant documents must contain
        index: The retrieval index
        k: Number of documents to retrieve

    Returns:
        Dictionary with query results and metrics (keys with @k suffix)
    """
    hits = retrieve(index, query, k=k)

    # Find relevant documents by text pattern
    relevant_positions = []
    for rank, (_doc_id, _score, passage) in enumerate(hits, start=1):
        if relevant_contains.lower() in passage.lower():
            relevant_positions.append(rank)

    # Calculate metrics using positions
    has_relevant = len(relevant_positions) > 0
    first_relevant_rank = relevant_positions[0] if relevant_positions else None

    recall_at_k = 1.0 if has_relevant else 0.0
    mrr_at_k = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    # For nDCG and Precision, use position-based calculation
    dcg = sum(1.0 / math.log2(pos + 1) for pos in relevant_positions)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_positions), k)))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    precision_at_k = len(relevant_positions) / k

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
                "score": float(score),
                "passage": passage[:200],
                "is_relevant": rank in relevant_positions,
            }
            for rank, (doc_id, score, passage) in enumerate(hits, start=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/index.pkl")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument(
        "--benchmark", action="store_true", help="Show detailed performance metrics"
    )
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Missing index: {args.index} (run build_index.py first)")

    if args.benchmark:
        print(format_system_info(get_system_info()))
        print()

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    # Evaluate all queries with benchmarking
    query_results = []
    query_times = []

    with (
        Benchmark("Evaluation", track_memory=True) as bench,
        open(args.eval, encoding="utf-8") as f,
    ):
        for line in f:
            row = json.loads(line)

            # Measure per-query time if benchmarking
            if args.benchmark:
                query_start = time.time()

            result = evaluate_query_with_pattern(
                row["query"], row["relevant_contains"], index, args.k
            )

            if args.benchmark:
                query_time = time.time() - query_start
                query_times.append(query_time)

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
    print(f"\nPer-query report saved to: {per_query_path}")

    # Identify and save worst 20 queries (sorted by MRR, then nDCG)
    sorted_by_performance = sorted(
        query_results, key=lambda x: (x["mrr@k"], x["ndcg@k"], x["recall@k"])
    )
    worst_queries = sorted_by_performance[: min(20, len(sorted_by_performance))]

    worst_queries_path = os.path.join(args.output_dir, "worst_20_queries.json")
    with open(worst_queries_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": f"Worst performing queries sorted by MRR@{args.k}, nDCG@{args.k}, Recall@{args.k}",
                "k": args.k,
                "worst_queries": worst_queries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Worst 20 queries analysis saved to: {worst_queries_path}")

    # Print summary of worst queries
    print(f"\n{'=' * 60}")
    print(f"WORST {min(20, len(worst_queries))} QUERIES")
    print("=" * 60)
    for i, result in enumerate(worst_queries[:10], 1):  # Show top 10 in console
        print(f"{i}. Query: {result['query'][:60]}...")
        print(
            f"   MRR@{args.k}: {result['mrr@k']:.3f} | nDCG@{args.k}: {result['ndcg@k']:.3f} | Recall@{args.k}: {result['recall@k']:.3f}"
        )

    if len(worst_queries) > 10:
        print(f"\n   ... and {len(worst_queries) - 10} more (see {worst_queries_path})")
    print("=" * 60)

    # Show benchmark results if requested
    if args.benchmark:
        result = bench.get_result(num_queries=total)

        print(f"\n{'=' * 60}")
        print("PERFORMANCE METRICS")
        print(f"{'=' * 60}")
        print(f"Total evaluation time: {result.wall_time:.3f}s")
        print(f"Queries evaluated: {total}")
        print(f"Average time per query: {result.wall_time / total * 1000:.2f}ms")

        if query_times:
            query_times.sort()
            print("\nPer-query latency:")
            print(f"  Mean: {sum(query_times) / len(query_times) * 1000:.2f}ms")
            print(f"  Median: {query_times[len(query_times) // 2] * 1000:.2f}ms")
            print(f"  P95: {query_times[int(len(query_times) * 0.95)] * 1000:.2f}ms")
            print(f"  P99: {query_times[int(len(query_times) * 0.99)] * 1000:.2f}ms")
            print(f"  Min: {min(query_times) * 1000:.2f}ms")
            print(f"  Max: {max(query_times) * 1000:.2f}ms")

        if result.memory_used_mb is not None:
            print(f"\nMemory used: {result.memory_used_mb:.2f} MB")

        if result.throughput is not None:
            print(f"Throughput: {result.throughput:.2f} queries/sec")

        print(f"{'=' * 60}")


def evaluate_retrieval(
    index: Any,
    eval_file: str,
    k: int = 5,
    method: str = "tfidf",
    _rerank: bool = False,
    output_file: str | None = None,
) -> dict[str, float]:
    """
    Evaluate retrieval performance on a test set.

    Args:
        index: The retrieval index
        eval_file: Path to JSONL file with queries and relevant_contains patterns
        k: Number of results to retrieve
        method: Retrieval method to use
        rerank: Whether to apply reranking
        output_file: Optional path to save detailed results

    Returns:
        Dictionary with aggregate metrics (mrr, ndcg, precision, recall)
    """
    from src.rag import retrieve_hybrid

    # Load evaluation queries
    queries = []
    with open(eval_file, encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))

    # Run evaluation
    all_results = []
    mrr_scores = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []

    for item in queries:
        query = item['query']
        relevant_pattern = item['relevant_contains']

        # Retrieve results
        hits = retrieve_hybrid(index, query, k, method=method)

        # Check which results are relevant
        relevant_positions = []
        for rank, (_idx, _score, passage) in enumerate(hits, start=1):
            if relevant_pattern.lower() in passage.lower():
                relevant_positions.append(rank)

        # Calculate metrics
        has_relevant = len(relevant_positions) > 0
        first_relevant = relevant_positions[0] if relevant_positions else None

        mrr = 1.0 / first_relevant if first_relevant else 0.0
        recall = 1.0 if has_relevant else 0.0
        precision = len(relevant_positions) / k

        # nDCG
        dcg = sum(1.0 / math.log2(pos + 1) for pos in relevant_positions)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_positions), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        precision_scores.append(precision)
        recall_scores.append(recall)

        all_results.append({
            'query': query,
            'relevant_contains': relevant_pattern,
            'mrr@k': mrr,
            'ndcg@k': ndcg,
            'precision@k': precision,
            'recall@k': recall,
            'num_relevant_found': len(relevant_positions),
        })

    # Save detailed results if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

    # Return aggregate metrics
    return {
        'mrr': sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        'ndcg': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        'precision': sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        'recall': sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
    }


if __name__ == "__main__":
    main()

