# ablation.py
# This script runs an ablation study comparing different retrieval methods.
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

from rag import retrieve, retrieve_hybrid


def evaluate_method(
    index: Any, eval_path: str, k: int, method: str, rerank: bool = False, **kwargs
) -> dict[str, Any]:
    """Evaluate a specific retrieval method.

    Args:
        index: The retrieval index
        eval_path: Path to eval.jsonl file
        k: Number of results to retrieve
        method: Retrieval method name
        rerank: Whether to apply reranking
        **kwargs: Additional arguments for retrieve_hybrid

    Returns:
        Dictionary with evaluation metrics
    """
    total = 0
    recall_hit = 0
    mrr_sum = 0.0

    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query = row["query"]
            relevant_contains = row["relevant_contains"]

            # Retrieve using specified method
            if method == "tfidf_legacy":
                hits = retrieve(index, query, k=k)
            else:
                hits = retrieve_hybrid(index, query, k=k, method=method, rerank=rerank, **kwargs)

            # Check for relevant documents
            relevant_rank = None
            for rank, (doc_id, score, passage) in enumerate(hits, start=1):
                if relevant_contains.lower() in passage.lower():
                    if relevant_rank is None:
                        relevant_rank = rank
                    recall_hit += 1
                    break

            # Calculate MRR
            if relevant_rank is not None:
                mrr_sum += 1.0 / relevant_rank

            total += 1

    recall = recall_hit / max(total, 1)
    mrr = mrr_sum / max(total, 1)

    return {
        "method": method,
        "rerank": rerank,
        "recall@k": recall,
        "mrr@k": mrr,
        "total_queries": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study on retrieval methods")
    parser.add_argument("--index", default="outputs/index_hybrid.pkl", help="Path to hybrid index")
    parser.add_argument("--eval", default="data/eval.jsonl", help="Path to evaluation file")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--output", default="outputs/ablation_results.json", help="Output file")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"Error: Index not found at {args.index}")
        print("Build a hybrid index first:")
        print(
            "  python src/build_index.py --bm25 --embeddings --reranker --out outputs/index_hybrid.pkl"
        )
        return

    print("Loading index...")
    with open(args.index, "rb") as f:
        index = pickle.load(f)

    print(f"Running ablation study @ K={args.k}")
    print("=" * 70)

    results = []

    # 1. TF-IDF only (baseline)
    print("\n[1/5] Evaluating TF-IDF (baseline)...")
    result = evaluate_method(index, args.eval, args.k, "tfidf")
    results.append(result)
    print(f"  Recall@{args.k}: {result['recall@k']:.4f} | MRR@{args.k}: {result['mrr@k']:.4f}")

    # 2. BM25 only
    if index.bm25 is not None:
        print("\n[2/5] Evaluating BM25...")
        result = evaluate_method(index, args.eval, args.k, "bm25")
        results.append(result)
        print(f"  Recall@{args.k}: {result['recall@k']:.4f} | MRR@{args.k}: {result['mrr@k']:.4f}")
    else:
        print("\n[2/5] BM25 not available in index (skipped)")

    # 3. Embeddings only
    if index.embedder is not None:
        print("\n[3/5] Evaluating Dense Embeddings...")
        result = evaluate_method(index, args.eval, args.k, "embeddings")
        results.append(result)
        print(f"  Recall@{args.k}: {result['recall@k']:.4f} | MRR@{args.k}: {result['mrr@k']:.4f}")
    else:
        print("\n[3/5] Embeddings not available in index (skipped)")

    # 4. Hybrid (TF-IDF + BM25 + Embeddings)
    if index.bm25 is not None or index.embedder is not None:
        print("\n[4/5] Evaluating Hybrid Retrieval...")
        result = evaluate_method(
            index,
            args.eval,
            args.k,
            "hybrid",
            tfidf_weight=0.4,
            bm25_weight=0.3,
            embedding_weight=0.3,
        )
        results.append(result)
        print(f"  Recall@{args.k}: {result['recall@k']:.4f} | MRR@{args.k}: {result['mrr@k']:.4f}")
    else:
        print("\n[4/5] Hybrid not available (need BM25 or embeddings)")

    # 5. Hybrid + Reranking
    if index.reranker is not None and (index.bm25 is not None or index.embedder is not None):
        print("\n[5/5] Evaluating Hybrid + Reranking...")
        result = evaluate_method(
            index,
            args.eval,
            args.k,
            "hybrid",
            rerank=True,
            rerank_top_k=20,
            tfidf_weight=0.4,
            bm25_weight=0.3,
            embedding_weight=0.3,
        )
        results.append(result)
        print(f"  Recall@{args.k}: {result['recall@k']:.4f} | MRR@{args.k}: {result['mrr@k']:.4f}")
    else:
        print("\n[5/5] Reranking not available (need reranker + hybrid methods)")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"k": args.k, "results": results}, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Method':<30} {'Recall@' + str(args.k):<15} {'MRR@' + str(args.k):<15}")
    print("-" * 70)

    for result in results:
        method_name = result["method"]
        if result.get("rerank"):
            method_name += " + Rerank"
        print(f"{method_name:<30} {result['recall@k']:<15.4f} {result['mrr@k']:<15.4f}")

    print("=" * 70)
    print(f"\nResults saved to: {args.output}")

    # Calculate improvements
    if len(results) >= 2:
        baseline_recall = results[0]["recall@k"]
        baseline_mrr = results[0]["mrr@k"]
        best_recall = max(r["recall@k"] for r in results)
        best_mrr = max(r["mrr@k"] for r in results)

        recall_improvement = ((best_recall - baseline_recall) / max(baseline_recall, 0.001)) * 100
        mrr_improvement = ((best_mrr - baseline_mrr) / max(baseline_mrr, 0.001)) * 100

        print("\nImprovements over TF-IDF baseline:")
        print(f"  Recall@{args.k}: {recall_improvement:+.1f}%")
        print(f"  MRR@{args.k}: {mrr_improvement:+.1f}%")


if __name__ == "__main__":
    main()
